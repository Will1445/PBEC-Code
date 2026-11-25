"""Adaptive midpoint planning for sharp changes in fitted g2 results."""


class AdaptiveScan:
    """Refine measured intervals when either fitted metric changes sharply.

    Refinement points are placed at interval midpoints. Each child interval is
    evaluated only after its endpoints have been measured, which lets a
    detected feature be progressively localized without duplicating queued
    powers.
    """

    def __init__(
        self,
        initial_powers_mw,
        *,
        target_spacing_cps: float,
        change_threshold_percent: float,
        max_refinements: int,
    ):
        if target_spacing_cps <= 0:
            raise ValueError("Target spacing must be positive")
        if change_threshold_percent < 0:
            raise ValueError("Change threshold must not be negative")
        if max_refinements < 0:
            raise ValueError("Maximum refinements must not be negative")

        self.target_spacing_cps = float(target_spacing_cps)
        self.change_threshold_percent = float(change_threshold_percent)
        self.max_refinements = int(max_refinements)
        self.point_levels = {self._key(power): 0 for power in initial_powers_mw}
        self.initial_points = set(self.point_levels)
        self.pending_points = set()
        self.measured_results = {}
        self.measured_count_rates = {}
        self.pending_count_rates = {}

    @staticmethod
    def _key(power_mw: float) -> float:
        return round(float(power_mw), 9)

    @property
    def pending_powers_mw(self):
        return sorted(self.pending_points)

    @property
    def pending_count_rates_cps(self):
        return sorted(self.pending_count_rates.values())

    def mark_measured(self, power_mw: float, *, count_rate_cps: float, peak_height=None, width_ns=None):
        """Mark a point complete; provide both fit values to use it for refinement."""
        power = self._key(power_mw)
        count_rate = float(count_rate_cps)
        if count_rate < 0:
            raise ValueError("Count rate must not be negative")
        self.pending_points.discard(power)
        self.pending_count_rates.pop(power, None)
        self.measured_count_rates[power] = count_rate
        if peak_height is not None and width_ns is not None:
            self.measured_results[power] = (float(peak_height), float(width_ns))
        return self._generate_refinements()

    def _generate_refinements(self):
        new_points = []
        planned = sorted(self.point_levels)
        for left, right in zip(planned, planned[1:]):
            if left not in self.measured_results or right not in self.measured_results:
                continue
            spacing_cps = abs(self.measured_count_rates[right] - self.measured_count_rates[left])
            if spacing_cps <= self.target_spacing_cps:
                continue
            if max(self.point_levels[left], self.point_levels[right]) >= self.max_refinements:
                continue

            left_height, left_width = self.measured_results[left]
            right_height, right_width = self.measured_results[right]
            height_change = self._relative_change_percent(left_height, right_height)
            width_change = self._relative_change_percent(left_width, right_width)
            if max(height_change, width_change) < self.change_threshold_percent:
                continue

            midpoint = self._key((left + right) / 2.0)
            if midpoint in self.point_levels:
                continue
            level = max(self.point_levels[left], self.point_levels[right]) + 1
            self.point_levels[midpoint] = level
            self.pending_points.add(midpoint)
            self.pending_count_rates[midpoint] = (
                self.measured_count_rates[left] + self.measured_count_rates[right]
            ) / 2.0
            new_points.append(midpoint)
        return sorted(new_points)

    @staticmethod
    def _relative_change_percent(first: float, second: float) -> float:
        denominator = max(abs(first), abs(second), 1e-12)
        return abs(second - first) / denominator * 100.0
