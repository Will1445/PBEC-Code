"""Fixed-pump scan-speed and direction-order polarization experiment.

The experiment is designed to distinguish:

* a genuine scan-rate-dependent switching delay;
* a shift caused by which scan was recorded first or second;
* slow thermal/cavity drift;
* a speed-independent piezo-direction offset.

The caller must already have created these instrument objects:

    laser
    controller
    powermeter
    powermeter2

Notebook example
----------------
from scan_speed_direction_test import run_scan_speed_direction_test

csv_path, metadata_path = run_scan_speed_direction_test(
    laser=laser,
    controller=controller,
    powermeter=powermeter,
    powermeter2=powermeter2,
)
"""

from __future__ import annotations

import csv
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_VOLTAGES = np.linspace(24.0, 28.0, 201)
DEFAULT_POINT_SETTLE_VALUES_S = (0.0, 0.5, 2.0)


def _as_float(value: Any) -> float:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected a scalar instrument reading, got {value!r}")
    return float(array.reshape(-1)[0])


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    temporary_path.replace(path)


def _ramp_voltage_without_measurement(
    controller: Any,
    target_voltage: float,
    maximum_step_v: float,
    settle_s: float,
) -> None:
    current = _as_float(controller.get_x_voltage())
    number_of_steps = max(
        1, int(math.ceil(abs(target_voltage - current) / maximum_step_v))
    )
    for voltage in np.linspace(
        current, target_voltage, number_of_steps + 1
    )[1:]:
        controller.set_x_voltage(float(voltage))
        time.sleep(settle_s)


def _make_balanced_schedule(
    point_settle_values_s: np.ndarray,
    number_of_blocks: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    """Use every speed once with each first direction in every block.

    The first direction alternates strictly within each block. The speed order
    is independently randomized for increasing-first and decreasing-first
    acquisitions. Odd blocks start with the opposite direction, counterbalancing
    the earliest and latest positions in the block.
    """
    rng = np.random.default_rng(random_seed)
    schedule: list[dict[str, Any]] = []
    acquisition_index = 0

    for block_index in range(number_of_blocks):
        speed_for_increasing_first = list(
            rng.permutation(point_settle_values_s)
        )
        speed_for_decreasing_first = list(
            rng.permutation(point_settle_values_s)
        )
        number_of_conditions = 2 * len(point_settle_values_s)

        for order_in_block in range(number_of_conditions):
            increasing_first = (order_in_block + block_index) % 2 == 0
            if increasing_first:
                first_direction = "increasing"
                point_settle_s = speed_for_increasing_first.pop(0)
            else:
                first_direction = "decreasing"
                point_settle_s = speed_for_decreasing_first.pop(0)

            schedule.append(
                {
                    "block_index": block_index,
                    "acquisition_index": acquisition_index,
                    "order_in_block": order_in_block,
                    "point_settle_s": float(point_settle_s),
                    "first_direction": first_direction,
                    "direction_order": (
                        ["increasing", "decreasing"]
                        if increasing_first
                        else ["decreasing", "increasing"]
                    ),
                }
            )
            acquisition_index += 1

    return schedule


def run_scan_speed_direction_test(
    laser: Any,
    controller: Any,
    powermeter: Any,
    powermeter2: Any,
    *,
    pump_power: float = 0.045,
    voltages: Iterable[float] = DEFAULT_VOLTAGES,
    point_settle_values_s: Iterable[float] = (
        DEFAULT_POINT_SETTLE_VALUES_S
    ),
    number_of_blocks: int = 3,
    random_seed: int = 20260730,
    initial_pump_settle_s: float = 30.0,
    endpoint_dwell_s: float = 10.0,
    positioning_step_v: float = 0.05,
    positioning_settle_s: float = 0.03,
    estimated_meter_overhead_per_point_s: float = 1.15,
    output_directory: str | Path = "polarization_scan_data",
) -> tuple[Path, Path]:
    """Run a balanced fixed-pump speed/direction-order experiment.

    Each block contains every combination of:

    * requested point-settling time;
    * increasing-first or decreasing-first direction order.

    The direction order alternates within a block. The condition speeds are
    randomized independently in each block. The pump is held constant throughout
    the experiment, including the short voltage repositioning ramps.
    """
    voltage_up = np.asarray(list(voltages), dtype=float)
    settle_values = np.asarray(list(point_settle_values_s), dtype=float)

    if voltage_up.ndim != 1 or len(voltage_up) < 2:
        raise ValueError("voltages must be a one-dimensional sequence")
    if not np.all(np.isfinite(voltage_up)):
        raise ValueError("voltages must all be finite")
    if not np.all(np.diff(voltage_up) > 0):
        raise ValueError("voltages must be strictly increasing")
    if settle_values.ndim != 1 or len(settle_values) == 0:
        raise ValueError(
            "point_settle_values_s must be a non-empty sequence"
        )
    if not np.all(np.isfinite(settle_values)) or np.any(settle_values < 0):
        raise ValueError(
            "point_settle_values_s must contain finite non-negative values"
        )
    if len(np.unique(settle_values)) != len(settle_values):
        raise ValueError("point_settle_values_s must not contain duplicates")
    if number_of_blocks < 1:
        raise ValueError("number_of_blocks must be at least one")
    if endpoint_dwell_s < 0 or initial_pump_settle_s < 0:
        raise ValueError("settling and dwell times must be non-negative")

    voltage_down = voltage_up[::-1]
    scans_by_direction = {
        "increasing": voltage_up,
        "decreasing": voltage_down,
    }
    schedule = _make_balanced_schedule(
        settle_values, number_of_blocks, random_seed
    )

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_directory / f"{run_id}_speed_direction_test.csv"
    metadata_path = output_directory / (
        f"{run_id}_speed_direction_test_metadata.json"
    )

    fieldnames = [
        "run_id",
        "block_index",
        "acquisition_index",
        "order_in_block",
        "pump_setpoint",
        "point_settle_s",
        "first_direction",
        "direction_order",
        "scan_order_within_acquisition",
        "scan_index",
        "direction",
        "point_index",
        "commanded_voltage_v",
        "measured_voltage_v",
        "power_meter_1_uW",
        "power_meter_2_uW",
        "elapsed_s",
        "meter_1_elapsed_s",
        "meter_2_elapsed_s",
        "timestamp_utc",
    ]

    estimated_duration_s = initial_pump_settle_s + sum(
        2
        * (
            len(voltage_up)
            * (
                estimated_meter_overhead_per_point_s
                + acquisition["point_settle_s"]
            )
            + endpoint_dwell_s
        )
        for acquisition in schedule
    )
    metadata: dict[str, Any] = {
        "run_id": run_id,
        "experiment": "fixed_pump_speed_and_direction_order",
        "status": "initializing",
        "started_utc": _utc_now(),
        "csv_file": str(csv_path),
        "pump_setpoint": float(pump_power),
        "voltage_start_v": float(voltage_up[0]),
        "voltage_stop_v": float(voltage_up[-1]),
        "points_per_direction": int(len(voltage_up)),
        "voltage_step_v": float(np.median(np.diff(voltage_up))),
        "point_settle_values_s": settle_values.tolist(),
        "number_of_blocks": int(number_of_blocks),
        "conditions_per_block": int(2 * len(settle_values)),
        "total_acquisitions": int(len(schedule)),
        "total_planned_scans": int(2 * len(schedule)),
        "random_seed": int(random_seed),
        "initial_pump_settle_s": float(initial_pump_settle_s),
        "endpoint_dwell_s": float(endpoint_dwell_s),
        "positioning_step_v": float(positioning_step_v),
        "positioning_settle_s": float(positioning_settle_s),
        "estimated_meter_overhead_per_point_s": float(
            estimated_meter_overhead_per_point_s
        ),
        "estimated_duration_hours": float(estimated_duration_s / 3600),
        "power_units": "uW",
        "power_conversion_from_meter_reading": 1e6,
        "schedule": schedule,
        "completed_scans": 0,
        "completed_acquisitions": 0,
    }
    _write_metadata(metadata_path, metadata)

    print(
        f"Planned: {len(schedule)} bidirectional acquisitions, "
        f"{2 * len(schedule)} individual scans"
    )
    print(
        "Estimated duration using "
        f"{estimated_meter_overhead_per_point_s:.2f} s meter overhead: "
        f"{estimated_duration_s / 3600:.2f} h"
    )

    run_start = time.perf_counter()
    scan_index = 0
    completed_acquisitions = 0

    try:
        # Start at the first condition's correct endpoint with the pump off,
        # then leave the pump fixed for the whole experimental schedule.
        first_direction = schedule[0]["first_direction"]
        initial_voltage = (
            float(voltage_up[0])
            if first_direction == "increasing"
            else float(voltage_up[-1])
        )
        laser.set(0)
        _ramp_voltage_without_measurement(
            controller,
            target_voltage=initial_voltage,
            maximum_step_v=positioning_step_v,
            settle_s=positioning_settle_s,
        )
        laser.set(float(pump_power))
        time.sleep(initial_pump_settle_s)

        metadata["status"] = "running"
        _write_metadata(metadata_path, metadata)

        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            handle.flush()

            for acquisition in schedule:
                first_direction = acquisition["first_direction"]
                starting_voltage = (
                    float(voltage_up[0])
                    if first_direction == "increasing"
                    else float(voltage_up[-1])
                )

                # The preceding bidirectional acquisition ends where it began.
                # Repositioning to the opposite endpoint before the next
                # condition also preconditions the recorded branch correctly.
                _ramp_voltage_without_measurement(
                    controller,
                    target_voltage=starting_voltage,
                    maximum_step_v=positioning_step_v,
                    settle_s=positioning_settle_s,
                )

                direction_order = acquisition["direction_order"]
                point_settle_s = acquisition["point_settle_s"]
                print(
                    "Block "
                    f"{acquisition['block_index'] + 1}/{number_of_blocks}, "
                    f"condition "
                    f"{acquisition['order_in_block'] + 1}/"
                    f"{2 * len(settle_values)}, "
                    f"settle={point_settle_s:.3f} s, "
                    f"order={direction_order[0]}→{direction_order[1]}"
                )

                for scan_order, direction in enumerate(direction_order):
                    time.sleep(endpoint_dwell_s)
                    scan_voltages = scans_by_direction[direction]

                    for point_index, commanded_voltage in enumerate(
                        scan_voltages
                    ):
                        controller.set_x_voltage(float(commanded_voltage))
                        time.sleep(point_settle_s)

                        measured_voltage = _as_float(
                            controller.get_x_voltage()
                        )
                        meter_1_time = time.perf_counter() - run_start
                        power_1_uW = (
                            1e6
                            * _as_float(powermeter.take_power_reading())
                        )
                        meter_2_time = time.perf_counter() - run_start
                        power_2_uW = (
                            1e6
                            * _as_float(powermeter2.take_power_reading())
                        )
                        elapsed = time.perf_counter() - run_start

                        writer.writerow(
                            {
                                "run_id": run_id,
                                "block_index": acquisition["block_index"],
                                "acquisition_index": acquisition[
                                    "acquisition_index"
                                ],
                                "order_in_block": acquisition[
                                    "order_in_block"
                                ],
                                "pump_setpoint": float(pump_power),
                                "point_settle_s": point_settle_s,
                                "first_direction": first_direction,
                                "direction_order": (
                                    f"{direction_order[0]}_then_"
                                    f"{direction_order[1]}"
                                ),
                                "scan_order_within_acquisition": scan_order,
                                "scan_index": scan_index,
                                "direction": direction,
                                "point_index": point_index,
                                "commanded_voltage_v": float(
                                    commanded_voltage
                                ),
                                "measured_voltage_v": measured_voltage,
                                "power_meter_1_uW": power_1_uW,
                                "power_meter_2_uW": power_2_uW,
                                "elapsed_s": elapsed,
                                "meter_1_elapsed_s": meter_1_time,
                                "meter_2_elapsed_s": meter_2_time,
                                "timestamp_utc": _utc_now(),
                            }
                        )
                        # Preserve all completed points if the run is stopped.
                        handle.flush()

                    scan_index += 1
                    metadata["completed_scans"] = scan_index
                    _write_metadata(metadata_path, metadata)

                completed_acquisitions += 1
                metadata["completed_acquisitions"] = completed_acquisitions
                _write_metadata(metadata_path, metadata)

        metadata["status"] = "complete"
        metadata["completed_utc"] = _utc_now()

    except BaseException as error:
        metadata["status"] = "aborted"
        metadata["aborted_utc"] = _utc_now()
        metadata["error_type"] = type(error).__name__
        metadata["error_message"] = str(error)
        raise

    finally:
        try:
            laser.set(0)
        finally:
            metadata["completed_scans"] = scan_index
            metadata["completed_acquisitions"] = completed_acquisitions
            try:
                metadata["final_voltage_v"] = _as_float(
                    controller.get_x_voltage()
                )
            except BaseException as final_voltage_error:
                metadata["final_voltage_v"] = None
                metadata["final_voltage_read_error"] = (
                    f"{type(final_voltage_error).__name__}: "
                    f"{final_voltage_error}"
                )
            _write_metadata(metadata_path, metadata)

    print(f"Saved measurements to {csv_path}")
    print(f"Saved metadata to {metadata_path}")
    return csv_path, metadata_path


if __name__ == "__main__":
    raise SystemExit(
        "Import run_scan_speed_direction_test from a notebook that already "
        "contains the connected instrument objects."
    )
