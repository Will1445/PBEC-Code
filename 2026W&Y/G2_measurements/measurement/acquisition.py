"""Background HydraHarp and Toptica power-sweep worker."""

import os
from pathlib import Path
import sys
import struct
from threading import Event
import time

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from measurement.adaptive_scan import AdaptiveScan
from measurement.analysis import fit_histogram
from measurement.models import AcquisitionConfig
from measurement.storage import save_histogram

class MeasurementWorker(QThread):
    """Acquire a power sweep without blocking the GUI thread."""

    log_message = pyqtSignal(str)
    progress = pyqtSignal(float, float, float, float)
    measurement_saved = pyqtSignal(str)
    adaptive_plan_changed = pyqtSignal(object)
    filter_prompt = pyqtSignal(float, object)
    filter_reset_prompt = pyqtSignal(int, object)
    nd_filter_count_changed = pyqtSignal(int)
    run_finished = pyqtSignal(bool, str)

    def __init__(self, config: AcquisitionConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self._stop_requested = Event()
        self._pause_requested = Event()
        self._hydra_factory = None
        self._laser_factory = None

    def request_stop(self):
        self._stop_requested.set()
        self._pause_requested.clear()

    def set_paused(self, paused: bool):
        if paused:
            self._pause_requested.set()
        else:
            self._pause_requested.clear()

    def _wait_while_paused(self) -> bool:
        while self._pause_requested.is_set() and not self._stop_requested.is_set():
            self._stop_requested.wait(0.1)
        return not self._stop_requested.is_set()

    def _load_hardware(self):
        if os.name != "nt":
            raise RuntimeError("The HydraHarp HHLIB interface requires Windows and the connected lab hardware.")
        self.log_message.emit(
            f"Python runtime: {sys.version.split()[0]}, {struct.calcsize('P') * 8}-bit, "
            f"{sys.executable}"
        )
        self.log_message.emit(r"HydraHarp DLL path: C:\Program Files\PicoQuant\UniHarp\hhlib64.dll")
        repo_root = Path(__file__).resolve().parents[3]
        wy_root = repo_root / "2026W&Y"
        for import_path in (repo_root, wy_root):
            if str(import_path) not in sys.path:
                sys.path.insert(0, str(import_path))

        from utils.hydraharp import HydraHarp
        from pylablib.devices import Toptica

        adapter_path = sys.modules[HydraHarp.__module__].__file__
        self.log_message.emit(f"HydraHarp adapter: {adapter_path}")
        hydra = self._hydra_factory() if self._hydra_factory else HydraHarp()
        try:
            laser = (
                self._laser_factory()
                if self._laser_factory
                else Toptica.TopticaIBeam(self.config.laser_com_port)
            )
        except Exception:
            hydra.close()
            raise
        return hydra, laser

    def _wait_for_filter(self, count_rate: float) -> bool:
        decision = {"continue": False}
        acknowledged = Event()
        self.filter_prompt.emit(
            count_rate,
            (decision, acknowledged, self.config.saturation_limit_cps),
        )
        while not acknowledged.wait(0.1):
            if self._stop_requested.is_set():
                return False
        return decision["continue"] and not self._stop_requested.is_set()

    def _filter_limit_reached(self, count_rate: float) -> bool:
        self.log_message.emit(
            f"Count rate {count_rate:,.0f} counts/s reached the configured limit."
        )
        return self._wait_for_filter(count_rate)

    def _wait_for_filter_reset(self, initial_count: int) -> bool:
        decision = {"continue": False}
        acknowledged = Event()
        self.filter_reset_prompt.emit(initial_count, (decision, acknowledged))
        while not acknowledged.wait(0.1):
            if self._stop_requested.is_set():
                return False
        return decision["continue"] and not self._stop_requested.is_set()

    def _acquire_block(self, hydra, seconds: float):
        hydra.start(seconds)
        deadline = time.monotonic() + seconds + 10.0
        while not hydra.complete() and not self._stop_requested.is_set():
            if time.monotonic() > deadline:
                if hydra.running:
                    hydra.stop_read()
                raise TimeoutError("HydraHarp acquisition timed out")
            time.sleep(0.05)
        result = hydra.stop_read()
        elapsed = float(result.get("elapsed_ms", 0.0)) / 1000.0
        if elapsed <= 0 and not self._stop_requested.is_set():
            elapsed = seconds
        return result, elapsed

    def _acquire_power_point(self, hydra, laser, power_mw, nd_filter_count, point_type):
        power_w = float(power_mw) / 1000.0
        laser.set_channel_power(1, power_w)
        time.sleep(0.25)
        self.log_message.emit(f"Set laser power to {power_mw:.4f} mW ({point_type} point)")

        count_rate = float(hydra.rates()["input_cps"][self.config.channel])
        while count_rate >= self.config.saturation_limit_cps:
            if not self._filter_limit_reached(count_rate):
                self._stop_requested.set()
                return None, nd_filter_count, None
            nd_filter_count += 1
            self.nd_filter_count_changed.emit(nd_filter_count)
            self.log_message.emit(f"ND filter count updated to {nd_filter_count}.")
            time.sleep(0.25)
            count_rate = float(hydra.rates()["input_cps"][self.config.channel])
        if self._stop_requested.is_set():
            return None, nd_filter_count, None

        # Preserve the detector rate measured immediately before acquisition.
        initial_count_rate = count_rate

        histogram = np.zeros(hydra.n, dtype=np.uint64)
        elapsed_total = 0.0
        saturated = False
        while elapsed_total < self.config.collection_time_s:
            if not self._wait_while_paused():
                break
            block_seconds = min(1.0, self.config.collection_time_s - elapsed_total)
            result, elapsed = self._acquire_block(hydra, block_seconds)
            elapsed_total += elapsed
            histogram += np.asarray(
                result["histograms"][self.config.channel], dtype=np.uint64
            )

            count_rate = float(hydra.rates()["input_cps"][self.config.channel])
            self.progress.emit(
                float(power_mw), elapsed_total,
                self.config.collection_time_s, count_rate,
            )
            status = result.get("status", {})
            warning_text = status.get("warning_text", "")
            if warning_text and "WARNING_INPUT_RATE_RATIO" not in warning_text:
                self.log_message.emit(f"HydraHarp warning: {warning_text}")

            if self._stop_requested.is_set():
                break
            if count_rate >= self.config.saturation_limit_cps:
                saturated = True
                if not self._filter_limit_reached(count_rate):
                    self._stop_requested.set()
                    break
                nd_filter_count += 1
                self.nd_filter_count_changed.emit(nd_filter_count)
                self.log_message.emit(
                    f"ND filter count updated to {nd_filter_count}; restarting this power point."
                )
                time.sleep(0.25)
                count_rate = float(hydra.rates()["input_cps"][self.config.channel])
                initial_count_rate = count_rate
                histogram.fill(0)
                elapsed_total = 0.0
                saturated = False

        if elapsed_total <= 0 or saturated:
            return None, nd_filter_count, None

        resolution_ns = float(hydra.settings["resolution_ps"]) / 1000.0
        delays_ns = np.arange(hydra.n, dtype=float) * resolution_ns
        saved_path = save_histogram(
            self.config.output_directory,
            delays_ns,
            histogram,
            collection_time_s=elapsed_total,
            nd_filter_count=nd_filter_count,
            initial_count_rate_cps=initial_count_rate,
            power_w=power_w,
            wavelength_nm=self.config.wavelength_nm,
            point_type=point_type,
        )
        self.log_message.emit(f"Saved histogram: {saved_path}")
        return saved_path, nd_filter_count, initial_count_rate

    def run(self):
        hydra = None
        laser = None
        success = False
        summary = "Measurement stopped."
        try:
            self.log_message.emit("Connecting to HydraHarp and Toptica laser…")
            hydra, laser = self._load_hardware()
            if not 0 <= self.config.channel < hydra.channels:
                raise ValueError(
                    f"HydraHarp channel {self.config.channel} is unavailable; "
                    f"this unit has {hydra.channels} input channels."
                )

            powers_mw = np.arange(
                self.config.start_power_mw,
                self.config.end_power_mw + self.config.step_power_mw * 1e-9,
                self.config.step_power_mw,
            )
            if powers_mw.size == 0:
                raise ValueError("The configured power sweep contains no points")

            scan = AdaptiveScan(
                powers_mw,
                target_spacing_cps=self.config.target_count_rate_spacing_cps,
                change_threshold_percent=self.config.jump_threshold_percent,
                max_refinements=self.config.max_refinements,
            )
            nd_filter_count = self.config.initial_nd_filter_count
            last_requested_power_mw = None
            self.nd_filter_count_changed.emit(nd_filter_count)
            self.adaptive_plan_changed.emit([])
            self.log_message.emit(
                f"Connected. Starting {len(powers_mw)} initial power points; "
                f"collection time {self.config.collection_time_s:g} s each."
            )
            completed_points = 0

            def process_saved_point(power_mw, initial_count_rate, saved_path):
                nonlocal completed_points
                completed_points += 1
                try:
                    fit = fit_histogram(saved_path)
                except (OSError, ValueError, RuntimeError) as exc:
                    fit = None
                    self.log_message.emit(
                        f"Could not fit {Path(saved_path).name}; no refinement from this point: {exc}"
                    )
                if fit is None:
                    new_points = scan.mark_measured(
                        power_mw, count_rate_cps=initial_count_rate
                    )
                else:
                    new_points = scan.mark_measured(
                        power_mw,
                        count_rate_cps=initial_count_rate,
                        peak_height=fit.peak_height,
                        width_ns=fit.width_ns,
                    )
                if new_points:
                    self.log_message.emit(
                        "Adaptive powers queued: "
                        + ", ".join(f"{power:.4f} mW" for power in new_points)
                    )
                self.adaptive_plan_changed.emit(scan.pending_count_rates_cps)
                self.measurement_saved.emit(str(saved_path))

            for power_mw in powers_mw:
                if not self._wait_while_paused():
                    break
                saved_path, nd_filter_count, initial_count_rate = self._acquire_power_point(
                    hydra, laser, power_mw, nd_filter_count, "initial"
                )
                last_requested_power_mw = float(power_mw)
                if saved_path is not None:
                    process_saved_point(power_mw, initial_count_rate, saved_path)
                if self._stop_requested.is_set():
                    break

            while not self._stop_requested.is_set() and scan.pending_powers_mw:
                if not self._wait_while_paused():
                    break
                power_mw = scan.pending_powers_mw[0]
                if (
                    last_requested_power_mw is not None
                    and power_mw < last_requested_power_mw
                    and nd_filter_count > self.config.initial_nd_filter_count
                ):
                    self.log_message.emit(
                        f"Adaptive point {power_mw:.4f} mW is below the previous "
                        "power; requesting the initial ND filter count."
                    )
                    if not self._wait_for_filter_reset(self.config.initial_nd_filter_count):
                        self._stop_requested.set()
                        break
                    nd_filter_count = self.config.initial_nd_filter_count
                    self.nd_filter_count_changed.emit(nd_filter_count)
                    self.log_message.emit(
                        f"ND filter count reset to {nd_filter_count}."
                    )
                saved_path, nd_filter_count, initial_count_rate = self._acquire_power_point(
                    hydra, laser, power_mw, nd_filter_count, "adaptive"
                )
                last_requested_power_mw = float(power_mw)
                if saved_path is not None:
                    process_saved_point(power_mw, initial_count_rate, saved_path)
                elif self._stop_requested.is_set():
                    break
                else:
                    scan.mark_measured(power_mw, count_rate_cps=0.0)
                    self.adaptive_plan_changed.emit(scan.pending_count_rates_cps)

            success = not self._stop_requested.is_set()
            summary = (
                f"Sweep complete: {completed_points} measurement(s) saved."
                if success
                else f"Stopped: {completed_points} measurement(s) saved."
            )
            self.log_message.emit(summary)
        except Exception as exc:
            summary = str(exc)
            self.log_message.emit(f"Measurement error: {exc}")
        finally:
            self.adaptive_plan_changed.emit([])
            if laser is not None:
                close_laser = getattr(laser, "close", None)
                if callable(close_laser):
                    try:
                        close_laser()
                    except Exception as exc:
                        self.log_message.emit(f"Laser close warning: {exc}")
            if hydra is not None:
                try:
                    hydra.close()
                except Exception as exc:
                    self.log_message.emit(f"HydraHarp close warning: {exc}")
            self.run_finished.emit(success, summary)
