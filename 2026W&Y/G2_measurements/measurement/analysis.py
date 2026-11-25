"""Analysis of saved g2 histograms."""

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class HistogramFit:
    """Fit results and the source histogram's laser power."""

    path: Path
    power_mw: float
    count_rate_cps: float | None
    background: float
    amplitude: float
    peak_position_ns: float
    tau_slow_ns: float
    peak_height: float
    peak_height_error: float
    width_ns: float
    width_error_ns: float
    fit_range_ns: float
    point_type: str


def uncertainty_outliers(results, *, factor: float = 3.0):
    """Return fits whose height or width uncertainty is a large outlier.

    A comparison needs at least three fitted measurements. Each uncertainty is
    compared with the median for that same quantity, so the differently scaled
    height and width errors are never compared with each other.
    """
    results = list(results)
    if len(results) < 3:
        return []

    errors = (
        np.asarray([result.peak_height_error for result in results], dtype=float),
        np.asarray([result.width_error_ns for result in results], dtype=float),
    )
    rejected = np.zeros(len(results), dtype=bool)
    for values in errors:
        valid = np.isfinite(values) & (values >= 0)
        rejected |= ~np.isfinite(values)
        if np.count_nonzero(valid) < 3:
            continue
        median = float(np.median(values[valid]))
        if median > 0:
            rejected |= valid & (values > factor * median)
    return [result for result, is_outlier in zip(results, rejected) if is_outlier]


_POWER_PATTERN = re.compile(
    r"(?P<power>\d+(?:\.\d+)?)\s*(?P<unit>mW|W)(?:_g2)?(?:_\d{8}_\d{6}(?:_\d{6})?)?$",
    re.IGNORECASE,
)


def power_from_filename(path: Path) -> float:
    """Extract laser power from names such as ``0.0302W_g2.csv``; return mW."""
    match = _POWER_PATTERN.search(path.stem)
    if not match:
        raise ValueError("Filename must end in a power such as 0.0302W_g2.csv")
    power = float(match.group("power"))
    return power if match.group("unit").lower() == "mw" else power * 1000.0


def load_histogram(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a two-column delay/count CSV with optional metadata and headers."""
    data_start = None
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.split(",")
            try:
                float(fields[0])
                float(fields[1])
            except (IndexError, ValueError):
                continue
            data_start = line_number
            break
    if data_start is None:
        raise ValueError("No numeric delay/count rows found in CSV")
    values = np.loadtxt(path, delimiter=",", skiprows=data_start, ndmin=2)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("Expected a CSV with delay and count columns")
    delays, counts = values[:, 0], values[:, 1]
    valid = np.isfinite(delays) & np.isfinite(counts)
    delays, counts = delays[valid], counts[valid]
    if delays.size < 5:
        raise ValueError("Histogram contains too few valid samples")
    return delays, counts


def load_metadata(path: Path) -> dict[str, str]:
    """Read key/value metadata comment lines from a histogram CSV."""
    metadata = {}
    with Path(path).open("r", encoding="utf-8") as stream:
        for line in stream:
            stripped = line.strip()
            if not stripped.startswith("#"):
                break
            key, separator, value = stripped[1:].strip().partition("=")
            if separator:
                metadata[key.strip()] = value.strip()
    return metadata


def fit_histogram(path: Path, *, max_delay_ns: float = 80.0) -> HistogramFit:
    """Fit the exponential peak model used by the existing g2 analysis script."""
    path = Path(path)
    delays, counts = load_histogram(path)
    mask = (delays >= 0) & (delays <= max_delay_ns)
    delays, counts = delays[mask], counts[mask]
    if delays.size < 5:
        raise ValueError(f"Fewer than five samples between 0 and {max_delay_ns:g} ns")

    background_guess = float(np.median(counts[-10:]))
    amplitude_guess = float(np.max(counts) - background_guess)
    if background_guess <= 0 or amplitude_guess <= 0:
        raise ValueError("Could not determine a positive peak and background")
    guesses = [background_guess, amplitude_guess, float(delays[np.argmax(counts)]), 20.0]

    def model(x, background, amplitude, peak_position, tau_slow):
        return background + amplitude * np.exp(-np.abs(x - peak_position) / tau_slow)

    parameters, covariance = curve_fit(
        model,
        delays,
        counts,
        p0=guesses,
        bounds=([0.0, 0.0, 0.0, 0.01], [np.inf, np.inf, max_delay_ns, np.inf]),
        maxfev=30000,
    )
    background, amplitude, peak_position, tau_slow = map(float, parameters)
    if background <= 0:
        raise ValueError("The fitted background is zero; normalised peak height is undefined")

    parameter_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    peak_height = 1.0 + amplitude / background
    d_height_d_background = -amplitude / (background**2)
    d_height_d_amplitude = 1.0 / background
    height_variance = (
        d_height_d_background**2 * covariance[0, 0]
        + d_height_d_amplitude**2 * covariance[1, 1]
        + 2 * d_height_d_background * d_height_d_amplitude * covariance[0, 1]
    )

    return HistogramFit(
        path=path,
        power_mw=power_from_filename(path),
        background=background,
        amplitude=amplitude,
        peak_position_ns=peak_position,
        tau_slow_ns=tau_slow,
        peak_height=peak_height,
        peak_height_error=float(np.sqrt(max(height_variance, 0.0))),
        width_ns=tau_slow / 2.0,
        width_error_ns=float(parameter_errors[3] / 2.0),
        fit_range_ns=max_delay_ns,
        count_rate_cps=(
            float(metadata["initial_count_rate_cps"])
            if (metadata := load_metadata(path)).get("initial_count_rate_cps") is not None
            else None
        ),
        point_type=metadata.get("point_type", "initial"),
    )

