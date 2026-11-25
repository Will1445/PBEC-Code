"""Matplotlib canvas used for live analysis plots."""

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np


class AnalysisPlot(FigureCanvasQTAgg):
    """Reusable plot panel with a clear empty state."""

    def __init__(self, title: str, x_label: str, y_label: str, parent=None):
        self.figure = Figure(figsize=(5, 2.5), tight_layout=True)
        super().__init__(self.figure)
        self.setParent(parent)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_title(title)
        self.axes.set_xlabel(x_label)
        self.axes.set_ylabel(y_label)
        self.axes.grid(True, alpha=0.25)
        self.axes.text(
            0.5,
            0.5,
            "Waiting for measurements",
            ha="center",
            va="center",
            transform=self.axes.transAxes,
            color="#758195",
        )

    def show_series(self, x, y, *, yerr=None, label="Measurements", marker="x"):
        title = self.axes.get_title()
        x_label = self.axes.get_xlabel()
        y_label = self.axes.get_ylabel()
        self.axes.clear()
        self.axes.set_title(title)
        self.axes.set_xlabel(x_label)
        self.axes.set_ylabel(y_label)
        self.axes.grid(True, alpha=0.25)
        self.axes.errorbar(x, y, yerr=yerr, marker=marker, linestyle="-", capsize=3, label=label)
        self.axes.legend()
        self.draw_idle()

    def show_measurement_series(
        self,
        *,
        initial_x,
        initial_y,
        initial_yerr,
        adaptive_x,
        adaptive_y,
        adaptive_yerr,
        pending_x,
    ):
        title = self.axes.get_title()
        x_label = self.axes.get_xlabel()
        y_label = self.axes.get_ylabel()
        self.axes.clear()
        self.axes.set_title(title)
        self.axes.set_xlabel(x_label)
        self.axes.set_ylabel(y_label)
        self.axes.grid(True, alpha=0.25)

        if initial_x:
            self.axes.errorbar(
                initial_x, initial_y, yerr=initial_yerr, marker="o", linestyle="none",
                color="#316da8", ecolor="#8fb2d3", capsize=3, label="Initial points",
            )
        if adaptive_x:
            self.axes.errorbar(
                adaptive_x, adaptive_y, yerr=adaptive_yerr, marker="o", linestyle="none",
                color="#27834b", ecolor="#8cc79f", capsize=3, label="Adaptive points",
            )
        if pending_x:
            for index, count_rate in enumerate(pending_x):
                self.axes.axvline(
                    count_rate,
                    color="#d44b4b",
                    linestyle=":",
                    linewidth=1.5,
                    label="Queued adaptive count rate" if index == 0 else None,
                )
        if initial_x or adaptive_x or pending_x:
            self.axes.legend()
        else:
            self.axes.text(
                0.5, 0.5, "Waiting for measurements", ha="center", va="center",
                transform=self.axes.transAxes, color="#758195",
            )
        self.draw_idle()

    def show_histogram(self, delays, counts, *, max_delay_ns, fit=None):
        self.axes.clear()
        mask = delays <= max_delay_ns
        self.axes.step(
            delays[mask], counts[mask], where="mid", color="#2867a6", label="Measured counts"
        )
        if fit is not None:
            fit_end = min(max_delay_ns, fit.fit_range_ns)
            fit_x = np.linspace(0.0, fit_end, 600)
            fit_y = fit.background + fit.amplitude * np.exp(
                -np.abs(fit_x - fit.peak_position_ns) / fit.tau_slow_ns
            )
            self.axes.plot(fit_x, fit_y, color="#d24a43", linewidth=1.8, label="Exponential fit")
        self.axes.set_xlabel("Time delay (ns)")
        self.axes.set_ylabel("Counts")
        self.axes.set_xlim(0, max_delay_ns)
        self.axes.grid(True, alpha=0.25)
        if fit is not None:
            self.axes.legend()
        self.draw_idle()

    def show_empty(self, message="Waiting for measurements"):
        title = self.axes.get_title()
        x_label = self.axes.get_xlabel()
        y_label = self.axes.get_ylabel()
        self.axes.clear()
        self.axes.set_title(title)
        self.axes.set_xlabel(x_label)
        self.axes.set_ylabel(y_label)
        self.axes.grid(True, alpha=0.25)
        self.axes.text(
            0.5, 0.5, message, ha="center", va="center",
            transform=self.axes.transAxes, color="#758195"
        )
        self.draw_idle()

