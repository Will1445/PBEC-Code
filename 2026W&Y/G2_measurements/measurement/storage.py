"""Persistence for raw histograms, fitted values, and scan metadata."""

from datetime import datetime
from pathlib import Path

import numpy as np


def save_histogram(
    output_directory: Path | str,
    delays_ns,
    counts,
    *,
    collection_time_s: float,
    nd_filter_count: int,
    initial_count_rate_cps: float,
    power_w: float,
    wavelength_nm: int,
    point_type: str = "initial",
) -> Path:
    """Write a histogram CSV with acquisition metadata in comment lines.

    ``nd_filter_count`` is the filter count at the time of acquisition, so the
    caller should pass the updated count after any operator filter prompt.
    """
    delays = np.asarray(delays_ns, dtype=float)
    histogram = np.asarray(counts)
    if delays.ndim != 1 or histogram.ndim != 1 or delays.size != histogram.size:
        raise ValueError("Delays and counts must be one-dimensional arrays of equal length")
    if delays.size == 0 or not np.all(np.isfinite(delays)) or not np.all(np.isfinite(histogram)):
        raise ValueError("Histogram must contain finite data")
    if not np.isfinite(collection_time_s) or collection_time_s <= 0:
        raise ValueError("Collection time must be positive")
    if not isinstance(nd_filter_count, (int, np.integer)) or nd_filter_count < 0:
        raise ValueError("ND filter count must be a non-negative integer")
    if not np.isfinite(power_w) or power_w < 0 or wavelength_nm <= 0:
        raise ValueError("Power must be non-negative and wavelength must be positive")
    if not np.isfinite(initial_count_rate_cps) or initial_count_rate_cps < 0:
        raise ValueError("Initial count rate must be finite and non-negative")
    if point_type not in {"initial", "adaptive"}:
        raise ValueError("Point type must be 'initial' or 'adaptive'")

    output_path = Path(output_directory).expanduser()
    output_path.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    power_label = f"{power_w:.10f}".rstrip("0").rstrip(".")
    filename = f"{power_label}W_g2_{stamp}.csv"
    csv_path = output_path / filename

    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        stream.write(f"# collection_time_s={collection_time_s:.10g}\n")
        stream.write(f"# nd_filter_count={nd_filter_count}\n")
        stream.write(f"# initial_count_rate_cps={initial_count_rate_cps:.10g}\n")
        stream.write(f"# power_W={power_w:.10g}\n")
        stream.write(f"# wavelength_nm={int(wavelength_nm)}\n")
        stream.write(f"# point_type={point_type}\n")
        stream.write("delay_ns,counts\n")
        np.savetxt(stream, np.column_stack((delays, histogram)), delimiter=",", fmt=("%.9f", "%.0f"))
    return csv_path

