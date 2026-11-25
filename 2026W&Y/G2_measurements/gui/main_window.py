"""Main window and first-pass layout for the g2 measurement application."""

from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSlider,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gui.plot_widget import AnalysisPlot
from measurement.analysis import fit_histogram, load_histogram, uncertainty_outliers
from measurement.acquisition import MeasurementWorker
from measurement.models import AcquisitionConfig


def number_input(value: float, *, minimum=0.0, maximum=1_000_000.0, decimals=4, step=0.01):
    field = QDoubleSpinBox()
    field.setRange(minimum, maximum)
    field.setDecimals(decimals)
    field.setSingleStep(step)
    field.setValue(value)
    return field


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PBEC · g² Measurement")
        self.resize(1440, 920)
        self.setMinimumSize(1100, 760)
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #f3f5f8; color: #1c2736; }
            QTabWidget::pane { border: 1px solid #dce2ea; background: #f3f5f8; }
            QTabBar::tab { background: #e5eaf1; padding: 9px 18px; margin-right: 3px;
                           border-top-left-radius: 6px; border-top-right-radius: 6px; }
            QTabBar::tab:selected { background: #ffffff; font-weight: 600; }
            QGroupBox { background: #ffffff; border: 1px solid #dce2ea;
                        border-radius: 10px; margin-top: 12px; padding: 14px 12px 12px; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px;
                               color: #26364a; font-weight: 600; }
            QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit { background: #ffffff; border: 1px solid #cbd3df;
                border-radius: 5px; padding: 6px; min-height: 20px; }
            QPushButton { background: #e8edf4; border: 0; border-radius: 6px;
                          padding: 8px 13px; font-weight: 600; }
            QPushButton:hover { background: #d9e3f0; }
            QPushButton#startButton { background: #2463a6; color: white; }
            QPushButton#startButton:hover { background: #1d548f; }
            QPlainTextEdit { background: #111a27; color: #d5e0ed; border: 0;
                             border-radius: 6px; padding: 8px; }
            QListWidget { background: #ffffff; border: 1px solid #dce2ea; border-radius: 6px; }
            """
        )
        self._build_ui()

    def _build_ui(self):
        root = QWidget()
        page = QVBoxLayout(root)
        page.setContentsMargins(10, 8, 10, 8)
        page.setSpacing(8)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._measurement_tab(), "Measurement")
        self.tabs.addTab(self._analysis_tab(), "Analysis & saved data")
        page.addWidget(self.tabs, 1)
        self.setCentralWidget(root)

    def _measurement_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 4, 10, 10)

        upper = QSplitter(Qt.Orientation.Horizontal)
        upper.setChildrenCollapsible(False)
        self.settings_panel = self._settings_panel()
        upper.addWidget(self.settings_panel)
        upper.addWidget(self._status_panel())
        upper.setStretchFactor(0, 4)
        upper.setStretchFactor(1, 6)
        layout.addWidget(upper, 1)
        layout.addWidget(self._terminal_panel())
        return tab

    def _settings_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(3, 0, 6, 0)
        layout.setSpacing(8)
        layout.addWidget(self._measurement_parameters())
        layout.addWidget(self._analysis_parameters())
        layout.addWidget(self._safety_parameters())
        layout.addStretch(1)
        scroll.setWidget(content)
        return scroll

    def _measurement_parameters(self):
        group = QGroupBox("Measurement parameters")
        form = QFormLayout(group)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.power_start = number_input(30.0, maximum=100.0, step=0.1)
        self.power_end = number_input(33.0, maximum=100.0, step=0.1)
        self.power_step = number_input(0.2, maximum=10.0, step=0.05)
        self.collection_time = number_input(20.0, maximum=3600.0, decimals=1, step=1.0)
        self.channel = QSpinBox()
        self.channel.setRange(0, 7)
        self.wavelength = QSpinBox()
        self.wavelength.setRange(300, 2000)
        self.wavelength.setValue(949)
        self.nd_filter_count = QSpinBox()
        self.nd_filter_count.setRange(0, 20)
        self.nd_filter_count.setValue(3)

        form.addRow("Start power (mW)", self.power_start)
        form.addRow("End power (mW)", self.power_end)
        form.addRow("Initial step (mW)", self.power_step)
        form.addRow("Collection time (s)", self.collection_time)
        form.addRow("HydraHarp channel", self.channel)
        form.addRow("Wavelength (nm)", self.wavelength)
        form.addRow("Initial ND filter count", self.nd_filter_count)
        return group

    def _analysis_parameters(self):
        group = QGroupBox("Adaptive resolution")
        form = QFormLayout(group)
        self.target_resolution = number_input(
            50_000, minimum=1, maximum=100_000_000, decimals=0, step=10_000
        )
        self.jump_threshold = number_input(10.0, maximum=100.0, step=0.5)
        self.max_refinements = QSpinBox()
        self.max_refinements.setRange(0, 12)
        self.max_refinements.setValue(3)
        form.addRow("Target count-rate spacing (counts/s)", self.target_resolution)
        form.addRow("Change threshold (%)", self.jump_threshold)
        form.addRow("Max refinement levels", self.max_refinements)
        return group

    def _safety_parameters(self):
        group = QGroupBox("Detector safety")
        form = QFormLayout(group)
        self.saturation_limit = number_input(1_000_000, minimum=1, maximum=100_000_000, decimals=0, step=100_000)
        form.addRow("Count-rate limit (counts/s)", self.saturation_limit)
        note = QLabel("At this limit, the scan asks you to add an ND filter and restarts the current power point.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #68778a; font-size: 11px")
        form.addRow("", note)
        return group

    def _status_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(5, 0, 0, 0)
        layout.addWidget(self._current_measurement())
        layout.addWidget(self._output_directory_panel())
        layout.addStretch(1)
        return panel

    def _output_directory_panel(self):
        group = QGroupBox("Measurement data folder")
        layout = QHBoxLayout(group)
        self.output_directory = QLineEdit(str(Path(__file__).resolve().parents[1] / "Data"))
        self.output_directory.setToolTip("Folder used for saved measurements and analysis.")
        self.browse_directory_button = QPushButton("Browse…")
        layout.addWidget(self.output_directory, 1)
        layout.addWidget(self.browse_directory_button)
        self.output_directory.editingFinished.connect(self.refresh_saved_measurements)
        self.browse_directory_button.clicked.connect(self.choose_output_directory)
        return group

    def _current_measurement(self):
        group = QGroupBox("Current measurement")
        grid = QGridLayout(group)
        self.current_values = {}
        self.run_status = QLabel("●  Ready")
        self.run_status.setStyleSheet("color: #267044; font-weight: 600")
        grid.addWidget(QLabel("Status"), 0, 0)
        grid.addWidget(self.run_status, 0, 1, alignment=Qt.AlignmentFlag.AlignRight)
        rows = [
            ("Laser power", "— mW"),
            ("Elapsed", "— / — s"),
            ("Input count rate", "— counts/s"),
            ("ND filter count", "—"),
            ("Fit status", "Waiting"),
            ("Measurements", "0 completed"),
        ]
        for row, (name, value) in enumerate(rows):
            key = QLabel(name)
            key.setStyleSheet("color: #68778a")
            val = QLabel(value)
            val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            val.setStyleSheet("font-weight: 600")
            grid.addWidget(key, row + 1, 0)
            grid.addWidget(val, row + 1, 1)
            self.current_values[name] = val
        return group

    def _analysis_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 4, 10, 10)
        layout.setSpacing(8)

        plots = QSplitter(Qt.Orientation.Horizontal)
        plots.setChildrenCollapsible(False)
        peak_box = QGroupBox("Peak height vs power")
        peak_layout = QVBoxLayout(peak_box)
        self.peak_plot = AnalysisPlot("", "Initial count rate (counts/s)", "g²(τ₀)")
        peak_layout.addWidget(self.peak_plot)
        width_box = QGroupBox("Peak width vs power")
        width_layout = QVBoxLayout(width_box)
        self.width_plot = AnalysisPlot("", "Initial count rate (counts/s)", "Width (ns)")
        width_layout.addWidget(self.width_plot)
        plots.addWidget(peak_box)
        plots.addWidget(width_box)
        plots.setStretchFactor(0, 1)
        plots.setStretchFactor(1, 1)
        layout.addWidget(plots, 1)

        data_box = QGroupBox("Saved measurements")
        data_layout = QVBoxLayout(data_box)
        self.refresh_measurements_button = QPushButton("Refresh")
        refresh_row = QHBoxLayout()
        refresh_row.addStretch(1)
        refresh_row.addWidget(self.refresh_measurements_button)
        data_layout.addLayout(refresh_row)

        data_split = QSplitter(Qt.Orientation.Horizontal)
        self.measurement_list = QListWidget()
        self.measurement_list.setMinimumWidth(230)
        histogram_box = QGroupBox("Selected histogram")
        histogram_layout = QVBoxLayout(histogram_box)
        self.histogram_plot = AnalysisPlot("", "Time delay (ns)", "Counts")
        histogram_layout.addWidget(self.histogram_plot)
        limit_row = QHBoxLayout()
        limit_row.addWidget(QLabel("Max delay (ns)"))
        self.histogram_limit_slider = QSlider(Qt.Orientation.Horizontal)
        self.histogram_limit_slider.setRange(1, 100)
        self.histogram_limit_slider.setMaximumHeight(18)
        self.histogram_limit_spin = number_input(80.0, minimum=0.01, maximum=100.0, decimals=2, step=0.1)
        self.histogram_limit_spin.setFixedWidth(92)
        self.histogram_limit_spin.setMaximumHeight(28)
        limit_row.addWidget(self.histogram_limit_slider, 1)
        limit_row.addWidget(self.histogram_limit_spin)
        histogram_layout.addLayout(limit_row)
        self.histogram_limit_slider.setEnabled(False)
        self.histogram_limit_spin.setEnabled(False)
        data_split.addWidget(self.measurement_list)
        data_split.addWidget(histogram_box)
        data_split.setStretchFactor(0, 2)
        data_split.setStretchFactor(1, 5)
        data_layout.addWidget(data_split, 1)
        layout.addWidget(data_box, 1)

        self.refresh_measurements_button.clicked.connect(self.refresh_saved_measurements)
        self.measurement_list.currentItemChanged.connect(self.show_selected_histogram)
        self.histogram_limit_slider.valueChanged.connect(self._histogram_limit_from_slider)
        self.histogram_limit_spin.valueChanged.connect(self._histogram_limit_from_spin)
        self._histogram_delays = None
        self._histogram_counts = None
        self._histogram_fit = None
        self._preferred_histogram_limit_ns = 80.0
        self.analysis_results = []
        self.pending_adaptive_count_rates_cps = []
        self.refresh_saved_measurements()
        return tab

    def _terminal_panel(self):
        group = QGroupBox("Output log")
        outer = QVBoxLayout(group)
        toolbar = QHBoxLayout()
        self.clear_output_button = QPushButton("Clear")
        toolbar.addStretch(1)
        toolbar.addWidget(self.clear_output_button)
        self.output_terminal = QPlainTextEdit()
        self.output_terminal.setReadOnly(True)
        self.output_terminal.setPlaceholderText("Ready. Configure a sweep to begin.")
        self.output_terminal.setMaximumBlockCount(2000)
        self.output_terminal.setMinimumHeight(100)
        actions = QHBoxLayout()
        self.start_button = QPushButton("Start measurement")
        self.start_button.setObjectName("startButton")
        self.pause_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_measurement)
        self.pause_button.clicked.connect(self.toggle_pause_measurement)
        self.stop_button.clicked.connect(self.stop_measurement)
        actions.addWidget(self.start_button)
        actions.addWidget(self.pause_button)
        actions.addWidget(self.stop_button)
        actions.addStretch(1)
        outer.addLayout(toolbar)
        outer.addWidget(self.output_terminal)
        outer.addLayout(actions)
        self.clear_output_button.clicked.connect(self.output_terminal.clear)
        self.log_message("Application layout ready. Hardware connection is not started yet.")
        return group

    def choose_output_directory(self):
        current = self.output_directory.text().strip()
        selected = QFileDialog.getExistingDirectory(self, "Choose measurement output directory", current)
        if selected:
            self.output_directory.setText(selected)
            self.refresh_saved_measurements()

    def refresh_saved_measurements(self, *_, select_path=None):
        folder = Path(self.output_directory.text()).expanduser()
        selected_path = str(select_path) if select_path else None
        current_item = self.measurement_list.currentItem()
        if current_item and selected_path is None:
            selected_path = current_item.data(Qt.ItemDataRole.UserRole)
        self.measurement_list.clear()
        if not folder.is_dir():
            self.histogram_plot.show_empty("Choose an existing output folder")
            self._clear_analysis_plots("Choose a data folder")
            return
        csv_paths = sorted(folder.rglob("*.csv"))
        for csv_path in csv_paths:
            label = str(csv_path.relative_to(folder))
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, str(csv_path))
            self.measurement_list.addItem(item)
            if selected_path and str(csv_path) == selected_path:
                self.measurement_list.setCurrentItem(item)
        if self.measurement_list.count() == 0:
            self.histogram_plot.show_empty("No CSV measurements found")
        elif self.measurement_list.currentItem() is None:
            self.measurement_list.setCurrentRow(0)
        self._update_analysis_plots(csv_paths)

    def _clear_analysis_plots(self, message):
        self.peak_plot.show_empty(message)
        self.width_plot.show_empty(message)
        self.analysis_results = []

    def _update_analysis_plots(self, csv_paths):
        results = []
        failures = []
        for csv_path in csv_paths:
            try:
                results.append(fit_histogram(csv_path))
            except (OSError, ValueError, RuntimeError) as exc:
                failures.append(f"{csv_path.name}: {exc}")

        self.analysis_results = sorted(results, key=lambda result: result.power_mw)
        if not self.analysis_results:
            self._clear_analysis_plots("No files could be fitted")
            if failures:
                self.log_message(f"Analysis skipped {len(failures)} file(s); check the CSV names and data format.")
            return

        missing_rates = [result for result in self.analysis_results if result.count_rate_cps is None]
        if missing_rates:
            self.log_message(
                f"Analysis omitted {len(missing_rates)} legacy measurement(s) without saved initial count rate."
            )
        rejected = uncertainty_outliers(
            result for result in self.analysis_results if result.count_rate_cps is not None
        )
        if rejected:
            names = ", ".join(result.path.name for result in rejected)
            self.log_message(
                f"Excluded {len(rejected)} measurement(s) from analysis plots because "
                f"fit uncertainty exceeded 3× the median: {names}"
            )
        self._render_analysis_plots()
        rejected = uncertainty_outliers(self.analysis_results)
        if rejected:
            names = ", ".join(result.path.name for result in rejected)
            self.log_message(
                f"Excluded {len(rejected)} measurement(s) from analysis plots because "
                f"fit uncertainty exceeded 3× the median: {names}"
            )
        if failures:
            self.log_message(f"Analysis fitted {len(results)} file(s); skipped {len(failures)} file(s).")
        else:
            self.log_message(f"Analysis fitted {len(results)} saved measurement(s).")

    def _render_analysis_plots(self):
        count_rate_results = [
            result for result in self.analysis_results if result.count_rate_cps is not None
        ]
        rejected = set(uncertainty_outliers(count_rate_results))
        plot_results = [result for result in count_rate_results if result not in rejected]
        initial = [result for result in plot_results if result.point_type != "adaptive"]
        adaptive = [result for result in plot_results if result.point_type == "adaptive"]
        pending = self.pending_adaptive_count_rates_cps

        self.peak_plot.show_measurement_series(
            initial_x=[result.count_rate_cps for result in initial],
            initial_y=[result.peak_height for result in initial],
            initial_yerr=[result.peak_height_error for result in initial],
            adaptive_x=[result.count_rate_cps for result in adaptive],
            adaptive_y=[result.peak_height for result in adaptive],
            adaptive_yerr=[result.peak_height_error for result in adaptive],
            pending_x=pending,
        )
        self.width_plot.show_measurement_series(
            initial_x=[result.count_rate_cps for result in initial],
            initial_y=[result.width_ns for result in initial],
            initial_yerr=[result.width_error_ns for result in initial],
            adaptive_x=[result.count_rate_cps for result in adaptive],
            adaptive_y=[result.width_ns for result in adaptive],
            adaptive_yerr=[result.width_error_ns for result in adaptive],
            pending_x=pending,
        )

    def update_adaptive_plan(self, pending_count_rates_cps):
        self.pending_adaptive_count_rates_cps = list(pending_count_rates_cps)
        self._render_analysis_plots()

    def show_selected_histogram(self, current, _previous=None):
        if current is None:
            self._histogram_delays = None
            self._histogram_counts = None
            self._histogram_fit = None
            self.histogram_plot.show_empty("Select a saved measurement")
            self.histogram_limit_slider.setEnabled(False)
            self.histogram_limit_spin.setEnabled(False)
            return
        csv_path = Path(current.data(Qt.ItemDataRole.UserRole))
        try:
            delays, counts = load_histogram(csv_path)
            fit = None
            try:
                fit = fit_histogram(csv_path)
            except (OSError, ValueError, RuntimeError) as fit_error:
                self.log_message(f"Fit overlay unavailable for {csv_path.name}: {fit_error}")
            self._histogram_delays = delays
            self._histogram_counts = counts
            self._histogram_fit = fit
            max_delay = max(float(np.max(delays)), 0.01)
            display_limit = min(self._preferred_histogram_limit_ns, max_delay)
            self.histogram_limit_slider.blockSignals(True)
            self.histogram_limit_spin.blockSignals(True)
            self.histogram_limit_spin.setRange(0.01, max_delay)
            self.histogram_limit_slider.setRange(1, max(1, int(round(max_delay * 100))))
            self.histogram_limit_spin.setValue(display_limit)
            self.histogram_limit_slider.setValue(max(1, int(round(display_limit * 100))))
            self.histogram_limit_spin.blockSignals(False)
            self.histogram_limit_slider.blockSignals(False)
            self.histogram_limit_slider.setEnabled(True)
            self.histogram_limit_spin.setEnabled(True)
            self._redraw_histogram(display_limit)
        except (OSError, ValueError) as exc:
            self.histogram_plot.show_empty("Could not load this histogram")
            self.log_message(f"Could not load {csv_path}: {exc}")

    def _histogram_limit_from_slider(self, slider_value: int):
        self.histogram_limit_spin.setValue(slider_value / 100.0)

    def _histogram_limit_from_spin(self, value: float):
        self._preferred_histogram_limit_ns = value
        self.histogram_limit_slider.blockSignals(True)
        self.histogram_limit_slider.setValue(int(round(value * 100)))
        self.histogram_limit_slider.blockSignals(False)
        self._redraw_histogram(value)

    def _redraw_histogram(self, max_delay_ns: float):
        if self._histogram_delays is None or self._histogram_counts is None:
            return
        self.histogram_plot.show_histogram(
            self._histogram_delays,
            self._histogram_counts,
            max_delay_ns=max_delay_ns,
            fit=self._histogram_fit,
        )

    def log_message(self, message: str):
        self.output_terminal.appendPlainText(message)

    def start_measurement(self):
        if getattr(self, "worker", None) is not None and self.worker.isRunning():
            return
        start_mw = self.power_start.value()
        end_mw = self.power_end.value()
        step_mw = self.power_step.value()
        collection_s = self.collection_time.value()
        if step_mw <= 0 or end_mw < start_mw or collection_s <= 0:
            QMessageBox.warning(
                self,
                "Invalid measurement settings",
                "Check that the end power is at or above the start power, the step is positive, and collection time is positive.",
            )
            return
        if (end_mw - start_mw) / step_mw + 1 > 10000:
            QMessageBox.warning(self, "Sweep too large", "This sweep would create more than 10,000 power points.")
            return

        output_value = self.output_directory.text().strip()
        if not output_value:
            QMessageBox.warning(self, "Invalid output folder", "Choose a folder for saved measurements.")
            return
        output_directory = Path(output_value).expanduser()
        if output_directory.exists() and not output_directory.is_dir():
            QMessageBox.warning(self, "Invalid output folder", "The selected output path is a file, not a folder.")
            return

        config = AcquisitionConfig(
            start_power_mw=start_mw,
            end_power_mw=end_mw,
            step_power_mw=step_mw,
            collection_time_s=collection_s,
            channel=self.channel.value(),
            wavelength_nm=self.wavelength.value(),
            initial_nd_filter_count=self.nd_filter_count.value(),
            saturation_limit_cps=int(self.saturation_limit.value()),
            target_count_rate_spacing_cps=self.target_resolution.value(),
            jump_threshold_percent=self.jump_threshold.value(),
            max_refinements=self.max_refinements.value(),
            output_directory=output_directory,
        )
        self.worker = MeasurementWorker(config, self)
        self.worker.log_message.connect(self.log_message)
        self.worker.progress.connect(self.update_measurement_progress)
        self.worker.measurement_saved.connect(self.on_measurement_saved)
        self.worker.adaptive_plan_changed.connect(self.update_adaptive_plan)
        self.worker.filter_prompt.connect(self.handle_filter_prompt)
        self.worker.filter_reset_prompt.connect(self.handle_filter_reset_prompt)
        self.worker.nd_filter_count_changed.connect(self.update_nd_filter_count)
        self.worker.run_finished.connect(self.on_measurement_run_finished)
        self.worker.finished.connect(self.on_measurement_thread_finished)

        self.settings_panel.setEnabled(False)
        self.output_directory.setEnabled(False)
        self.browse_directory_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.pause_button.setEnabled(True)
        self.pause_button.setText("Pause")
        self.stop_button.setEnabled(True)
        self.run_status.setText("●  Connecting")
        self.run_status.setStyleSheet("color: #9a6516; font-weight: 600")
        self.current_values["Measurements"].setText("0 completed")
        self.current_values["Fit status"].setText("Waiting")
        self._run_saved_count = 0
        self.pending_adaptive_count_rates_cps = []
        self._render_analysis_plots()
        self.worker.start()

    def toggle_pause_measurement(self):
        worker = getattr(self, "worker", None)
        if worker is None or not worker.isRunning():
            return
        paused = self.pause_button.text() == "Pause"
        worker.set_paused(paused)
        self.pause_button.setText("Resume" if paused else "Pause")
        self.run_status.setText("●  Paused" if paused else "●  Measuring")

    def stop_measurement(self):
        worker = getattr(self, "worker", None)
        if worker is None or not worker.isRunning():
            return
        worker.request_stop()
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.run_status.setText("●  Stopping")

    def update_measurement_progress(self, power_mw, elapsed_s, total_s, count_rate):
        self.run_status.setText("●  Measuring")
        self.run_status.setStyleSheet("color: #2463a6; font-weight: 600")
        self.current_values["Laser power"].setText(f"{power_mw:.4f} mW")
        self.current_values["Elapsed"].setText(f"{elapsed_s:.1f} / {total_s:.1f} s")
        self.current_values["Input count rate"].setText(f"{count_rate:,.0f} counts/s")

    def update_nd_filter_count(self, count):
        self.current_values["ND filter count"].setText(str(count))

    def handle_filter_prompt(self, count_rate, response_state):
        decision, acknowledged, saturation_limit = response_state
        self.run_status.setText("●  Waiting for ND filter")
        self.run_status.setStyleSheet("color: #9a6516; font-weight: 600")
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle("Add an ND filter to continue")
        dialog.setText(
            f"Input count rate {count_rate:,.0f} counts/s has reached the "
            f"configured limit of {saturation_limit:,.0f} counts/s."
        )
        dialog.setInformativeText(
            "Pause here and add one ND filter to the optical setup. Confirm only after it is in place. "
            "The partial histogram will be discarded, the filter count will increase by one, and this power point will restart."
        )
        continue_button = dialog.addButton("Filter added — continue", QMessageBox.ButtonRole.AcceptRole)
        dialog.addButton("Stop scan", QMessageBox.ButtonRole.RejectRole)
        dialog.exec()
        decision["continue"] = dialog.clickedButton() is continue_button
        acknowledged.set()

    def handle_filter_reset_prompt(self, initial_count, response_state):
        decision, acknowledged = response_state
        self.run_status.setText("●  Waiting for ND filter reset")
        self.run_status.setStyleSheet("color: #9a6516; font-weight: 600")
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Information)
        dialog.setWindowTitle("Reset ND filter stack")
        dialog.setText(
            "The adaptive scan is returning to a lower laser power after adding filters."
        )
        dialog.setInformativeText(
            f"Remove filters until the setup has its initial count of {initial_count}. "
            "Confirm only after the optical setup matches that count."
        )
        continue_button = dialog.addButton(
            "Filters reset — continue", QMessageBox.ButtonRole.AcceptRole
        )
        dialog.addButton("Stop scan", QMessageBox.ButtonRole.RejectRole)
        dialog.exec()
        decision["continue"] = dialog.clickedButton() is continue_button
        acknowledged.set()

    def on_measurement_saved(self, saved_path):
        self.refresh_saved_measurements(select_path=saved_path)
        self._run_saved_count += 1
        self.current_values["Measurements"].setText(f"{self._run_saved_count} saved this run")
        self.current_values["Fit status"].setText("Analysis updated")

    def on_measurement_run_finished(self, success, summary):
        self.log_message(summary)
        if success:
            self.run_status.setText("●  Complete")
            self.run_status.setStyleSheet("color: #267044; font-weight: 600")
        else:
            self.run_status.setText("●  Stopped / error")
            self.run_status.setStyleSheet("color: #9a6516; font-weight: 600")

    def on_measurement_thread_finished(self):
        self.settings_panel.setEnabled(True)
        self.output_directory.setEnabled(True)
        self.browse_directory_button.setEnabled(True)
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.pause_button.setText("Pause")
        self.stop_button.setEnabled(False)
        worker = self.worker
        self.worker = None
        if worker is not None:
            worker.deleteLater()

    def closeEvent(self, event):
        worker = getattr(self, "worker", None)
        if worker is not None and worker.isRunning():
            worker.request_stop()
            self.log_message("Stopping the active acquisition before closing…")
            if not worker.wait(5000):
                QMessageBox.information(
                    self,
                    "Acquisition stopping",
                    "The hardware is still stopping. Close the window again once the run has ended.",
                )
                event.ignore()
                return
        event.accept()

