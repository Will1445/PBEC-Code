"""Bidirectional, drift-aware polarization scan.

This module assumes that the caller has already created the four instrument
objects used in the existing notebook:

    laser
    controller
    powermeter
    powermeter2

Example
-------
from bidirectional_polarization_scan import run_bidirectional_scan

csv_path, metadata_path = run_bidirectional_scan(
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


# The previous measurements used a six-volt range and found approximately
# 4.05 backside-cavity FSRs within it.  The 20 mV point spacing resolves the
# narrow persistent domains while keeping the acquisition time manageable.
DEFAULT_VOLTAGES = np.linspace(23.3, 29.3, 301)

# Clean, essentially unsaturated range from the previous dataset.  It covers
# the region where the polarization-domain count changes most strongly.
DEFAULT_LASER_VALUES = np.asarray(
    [
        0.031,
        0.033,
        0.035,
        0.037,
        0.039,
        0.041,
        0.043,
        0.045,
        0.047,
        0.049,
        0.051,
        0.053,
        0.055,
        0.057,
        0.060,
    ],
    dtype=float,
)


def _as_float(value: Any) -> float:
    """Convert scalar-like instrument return values to a plain float."""
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"Expected a scalar instrument reading, got {value!r}")
    return float(array.reshape(-1)[0])


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ramp_voltage_without_measurement(
    controller: Any,
    target_voltage: float,
    maximum_step_v: float,
    settle_s: float,
) -> None:
    """Move to a scan endpoint through small steps instead of one large jump."""
    current = _as_float(controller.get_x_voltage())
    number_of_steps = max(
        1, int(math.ceil(abs(target_voltage - current) / maximum_step_v))
    )
    for voltage in np.linspace(
        current, target_voltage, number_of_steps + 1
    )[1:]:
        controller.set_x_voltage(float(voltage))
        time.sleep(settle_s)


def _make_pump_schedule(
    laser_values: np.ndarray,
    number_of_blocks: int,
    random_seed: int,
    anchor_power: float | None,
    anchor_every: int,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(random_seed)
    schedule: list[dict[str, Any]] = []
    acquisition_index = 0

    for block_index in range(number_of_blocks):
        randomized = rng.permutation(laser_values)

        for order_in_block, pump_power in enumerate(randomized):
            schedule.append(
                {
                    "acquisition_index": acquisition_index,
                    "block_index": block_index,
                    "order_in_block": order_in_block,
                    "pump_power": float(pump_power),
                    "is_anchor": False,
                }
            )
            acquisition_index += 1

            should_add_anchor = (
                anchor_power is not None
                and anchor_every > 0
                and (order_in_block + 1) % anchor_every == 0
                and order_in_block + 1 < len(randomized)
                and not np.isclose(pump_power, anchor_power)
            )
            if should_add_anchor:
                schedule.append(
                    {
                        "acquisition_index": acquisition_index,
                        "block_index": block_index,
                        "order_in_block": order_in_block,
                        "pump_power": float(anchor_power),
                        "is_anchor": True,
                    }
                )
                acquisition_index += 1

    return schedule


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    temporary_path.replace(path)


def run_bidirectional_scan(
    laser: Any,
    controller: Any,
    powermeter: Any,
    powermeter2: Any,
    *,
    voltages: Iterable[float] = DEFAULT_VOLTAGES,
    laser_values: Iterable[float] = DEFAULT_LASER_VALUES,
    number_of_blocks: int = 2,
    random_seed: int = 20260729,
    anchor_power: float | None = 0.045,
    anchor_every: int = 4,
    pump_settle_s: float = 0.5,
    point_settle_s: float = 0.1,
    reversal_settle_s: float = 0.2,
    positioning_step_v: float = 0.05,
    positioning_settle_s: float = 0.03,
    output_directory: str | Path = "polarization_scan_data",
) -> tuple[Path, Path]:
    """Acquire paired increasing/decreasing voltage scans and save every point.

    Each pump acquisition starts at the low-voltage endpoint, scans upwards,
    then immediately scans downwards at the same pump power.  It therefore ends
    where it started and never needs a large voltage reset between pump powers.

    Pump values are randomized independently in each repeat block.  Optional
    repeated anchor scans provide a direct measurement of slow drift.
    """
    voltage_up = np.asarray(list(voltages), dtype=float)
    pump_values = np.asarray(list(laser_values), dtype=float)

    if voltage_up.ndim != 1 or len(voltage_up) < 2:
        raise ValueError("voltages must be a one-dimensional sequence")
    if not np.all(np.diff(voltage_up) > 0):
        raise ValueError("voltages must be strictly increasing")
    if pump_values.ndim != 1 or len(pump_values) == 0:
        raise ValueError("laser_values must be a non-empty sequence")
    if number_of_blocks < 1:
        raise ValueError("number_of_blocks must be at least one")

    voltage_down = voltage_up[::-1]
    schedule = _make_pump_schedule(
        pump_values,
        number_of_blocks,
        random_seed,
        anchor_power,
        anchor_every,
    )

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_directory / f"{run_id}_bidirectional_scan.csv"
    metadata_path = output_directory / f"{run_id}_metadata.json"

    fieldnames = [
        "run_id",
        "block_index",
        "acquisition_index",
        "order_in_block",
        "is_anchor",
        "pump_setpoint",
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

    metadata: dict[str, Any] = {
        "run_id": run_id,
        "status": "initializing",
        "started_utc": _utc_now(),
        "csv_file": str(csv_path),
        "voltage_start_v": float(voltage_up[0]),
        "voltage_stop_v": float(voltage_up[-1]),
        "points_per_direction": int(len(voltage_up)),
        "voltage_step_v": float(np.median(np.diff(voltage_up))),
        "directions_per_pump": ["increasing", "decreasing"],
        "number_of_blocks": number_of_blocks,
        "random_seed": random_seed,
        "base_pump_values": pump_values.tolist(),
        "anchor_power": anchor_power,
        "anchor_every": anchor_every,
        "pump_settle_s": pump_settle_s,
        "point_settle_s": point_settle_s,
        "reversal_settle_s": reversal_settle_s,
        "power_units": "uW",
        "power_conversion_from_meter_reading": 1e6,
        "schedule": schedule,
        "completed_scans": 0,
    }
    _write_metadata(metadata_path, metadata)

    run_start = time.perf_counter()
    scan_index = 0

    try:
        # Position with the pump off so that the first recorded scan also begins
        # from the same endpoint as every subsequent pump acquisition.
        laser.set(0)
        _ramp_voltage_without_measurement(
            controller,
            target_voltage=float(voltage_up[0]),
            maximum_step_v=positioning_step_v,
            settle_s=positioning_settle_s,
        )

        metadata["status"] = "running"
        _write_metadata(metadata_path, metadata)

        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            handle.flush()

            for acquisition in schedule:
                pump_power = float(acquisition["pump_power"])
                print(
                    "Block "
                    f"{acquisition['block_index'] + 1}/{number_of_blocks}, "
                    f"pump={pump_power:.5f}"
                    + (" (anchor)" if acquisition["is_anchor"] else "")
                )

                laser.set(pump_power)
                time.sleep(pump_settle_s)

                for direction, scan_voltages in (
                    ("increasing", voltage_up),
                    ("decreasing", voltage_down),
                ):
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
                                "is_anchor": acquisition["is_anchor"],
                                "pump_setpoint": pump_power,
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
                        # If the run is interrupted, at most one point remains
                        # buffered rather than an entire scan.
                        handle.flush()

                    scan_index += 1
                    metadata["completed_scans"] = scan_index
                    _write_metadata(metadata_path, metadata)
                    time.sleep(reversal_settle_s)

        metadata["status"] = "complete"
        metadata["completed_utc"] = _utc_now()

    except BaseException as error:
        metadata["status"] = "aborted"
        metadata["aborted_utc"] = _utc_now()
        metadata["error_type"] = type(error).__name__
        metadata["error_message"] = str(error)
        raise

    finally:
        laser.set(0)
        metadata["completed_scans"] = scan_index
        metadata["final_voltage_v"] = _as_float(
            controller.get_x_voltage()
        )
        _write_metadata(metadata_path, metadata)

    print(f"Saved measurements to {csv_path}")
    print(f"Saved metadata to {metadata_path}")
    return csv_path, metadata_path
