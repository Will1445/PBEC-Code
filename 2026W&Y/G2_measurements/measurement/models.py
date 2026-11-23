"""Configuration and measurement record models."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AcquisitionConfig:
    start_power_mw: float
    end_power_mw: float
    step_power_mw: float
    collection_time_s: float
    channel: int
    wavelength_nm: int
    initial_nd_filter_count: int
    saturation_limit_cps: int
    target_resolution_mw: float
    jump_threshold_percent: float
    max_refinements: int
    output_directory: Path
    laser_com_port: str = "COM5"

