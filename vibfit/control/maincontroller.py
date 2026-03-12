from __future__ import annotations

import copy
import datetime as dt
from pathlib import Path

import numpy as np
from PyQt6 import QtCore, QtWidgets

from ..model import (
    BackgroundArea,
    BackgroundFitResult,
    BackgroundSpec,
    FitResultBundle,
    ParameterConstraint,
    PeakResult,
    PeakSpec,
    SavedSection,
    build_fit,
    clone_region,
    default_region,
    export_saved_sections,
    export_plot_npy,
    fit_background,
    get_param_dir,
    list_backup_events,
    load_spectrum,
    load_session_from_backup,
    save_session,
    update_backup_comment,
)
from ..view import MainWindow


class MainController:
    DEFAULT_SECTION_COMMENT = "Write your comment here"

    def __init__(self):
        self.widget = MainWindow()
        self.settings = QtCore.QSettings("vibfit", "vibfit")
        self.spectrum = None
        self.current_region = clone_region(default_region())
        self.current_region.background.fit_areas = []
        self.current_region.peaks = []
        self.background_result = None
        self.fit_result = None
        self.saved_sections = []
        self._syncing_region_boxes = False
        self._bg_area_press_cid = None
        self._bg_area_motion_cid = None
        self._bg_area_release_cid = None
        self._bg_area_press_x = None
        self._bg_area_preview = None
        self._fit_range_press_cid = None
        self._fit_range_motion_cid = None
        self._fit_range_release_cid = None
        self._fit_range_press_x = None
        self._fit_range_preview = None
        self._peak_pick_press_cid = None
        self._peak_pick_motion_cid = None
        self._peak_pick_release_cid = None
        self._peak_pick_press_x = None
        self._peak_pick_press_button = None
        self._peak_pick_preview = None
        self._build_initial_state()
        self.connect_channel()

    def _build_initial_state(self):
        self._populate_background_area_table()
        self._populate_peak_table()
        self._populate_sections_table()
        self._draw()

    def connect_channel(self):
        self.widget.pushButton_LoadSpectrum.clicked.connect(self.load_spectrum_from_dialog)
        self.widget.pushButton_SetFitRange.toggled.connect(self._toggle_fit_range_selector)
        self.widget.toolButton_ZoomOut.clicked.connect(self.zoom_out_full_region)
        self.widget.toolButton_ZoomIn.clicked.connect(self.zoom_in_active_region)
        self.widget.pushButton_AdjustYForSpectrum.clicked.connect(self.adjust_y_for_spectrum)
        self.widget.pushButton_FindViewMinMax.clicked.connect(self.find_view_minmax)
        self.widget.pushButton_ApplyView.clicked.connect(self.apply_view_limits)
        self.widget.pushButton_SelectBackgroundArea.toggled.connect(self._toggle_background_area_selector)
        self.widget.pushButton_RemoveBackgroundArea.clicked.connect(self.remove_selected_background_area)
        self.widget.pushButton_ClearBackgroundAreas.clicked.connect(self.clear_background_areas)
        self.widget.pushButton_FitBackground.clicked.connect(self.fit_background_model)
        self.widget.pushButton_PickPeaks.toggled.connect(self._toggle_peak_picker)
        self.widget.pushButton_ClearPeaks.clicked.connect(self.clear_peaks)
        self.widget.pushButton_RemovePeak.clicked.connect(self.remove_peak)
        self.widget.pushButton_ClearFitRange.clicked.connect(self.clear_fit_range)
        self.widget.pushButton_SaveToSection.clicked.connect(self.save_to_section)
        self.widget.pushButton_SaveFitResults.clicked.connect(self.save_fit_results)
        self.widget.pushButton_ExportNPY.clicked.connect(self.export_plot_npy)
        self.widget.pushButton_Fit.clicked.connect(self.fit_region)
        self.widget.pushButton_SectionSetCurrent.clicked.connect(self.set_selected_section_current)
        self.widget.pushButton_SectionRemove.clicked.connect(self.remove_selected_sections)
        self.widget.pushButton_SectionClear.clicked.connect(self.clear_section_list)
        self.widget.pushButton_SaveSession.clicked.connect(self.save_current_session)
        self.widget.pushButton_RestoreSession.clicked.connect(self.restore_selected_backup)
        self.widget.pushButton_EditBackupComment.clicked.connect(self.edit_selected_backup_comment)
        self.widget.tableWidget_Peaks.itemChanged.connect(self.handle_peak_table_change)
        self.widget.tableWidget_Sections.itemChanged.connect(self.handle_section_table_change)

    def show_window(self):
        self.widget.show()
        self.widget.raise_()
        self.widget.activateWindow()

    def log(self, message: str):
        self.widget.status_box.appendPlainText(message)

    def load_spectrum_from_dialog(self):
        start_dir = self._last_data_dir()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self.widget,
            "Open vibEELS spectrum",
            start_dir,
            "Supported (*.csv *.txt *.dat *.vxy *.npy *.npz *.dm3 *.dm4 *.hspy);;All files (*)",
        )
        if not path:
            return
        try:
            self.spectrum = load_spectrum(path)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self.widget, "Load failed", str(exc))
            self.log(f"Load failed: {exc}")
            return
        self.settings.setValue("paths/last_data_dir", str(Path(path).expanduser().resolve().parent))
        self.widget.label_SpectrumPath.setText(self.spectrum.path)
        self.widget.label_SpectrumShape.setText(str(self.spectrum.intensity.shape))
        self.log(f"Loaded spectrum from {self.spectrum.path}")
        self.current_region = clone_region(default_region())
        x_values = np.asarray(self.spectrum.x_cminv, dtype=float)
        if x_values.size:
            self.current_region.x_min_cminv = float(np.nanmin(x_values))
            self.current_region.x_max_cminv = float(np.nanmax(x_values))
        self.current_region.background.fit_areas = []
        self.current_region.peaks = []
        self.widget.pushButton_PickPeaks.setChecked(False)
        self.background_result = None
        self.fit_result = None
        self.saved_sections = []
        self.widget.tableWidget_Results.setRowCount(0)
        self.widget.plainTextEdit_BackgroundReport.clear()
        self.widget.plainTextEdit_FitReport.clear()
        self._populate_background_area_table()
        self._populate_peak_table()
        self._populate_sections_table()
        self.refresh_backup_table()
        self._draw()

    def _last_data_dir(self) -> str:
        saved = self.settings.value("paths/last_data_dir", "", type=str)
        if saved:
            saved_path = Path(saved).expanduser()
            if saved_path.exists() and saved_path.is_dir():
                return str(saved_path)
        return str(Path.home())

    def clear_peaks(self):
        if not self.current_region.peaks and self.fit_result is None:
            return
        preserve_limits = self._current_plot_limits()
        self.widget.pushButton_PickPeaks.setChecked(False)
        self.current_region.peaks = []
        self._clear_peak_fit_results()
        self._populate_peak_table()
        self._draw(preserve_limits=preserve_limits)

    def clear_fit_range(self):
        if self.spectrum is None:
            return
        preserve_limits = self._current_plot_limits()
        x_values = np.asarray(self.spectrum.x_cminv, dtype=float)
        if x_values.size == 0:
            return
        self.widget.pushButton_SetFitRange.setChecked(False)
        self.widget.pushButton_PickPeaks.setChecked(False)
        self.current_region.x_min_cminv = float(np.nanmin(x_values))
        self.current_region.x_max_cminv = float(np.nanmax(x_values))
        self._clear_background_results()
        self._draw(
            preserve_limits={
                "raw_xlim": preserve_limits["raw_xlim"],
                "raw_ylim": preserve_limits["raw_ylim"],
                "fit_xlim": preserve_limits["fit_xlim"],
                "fit_ylim": preserve_limits["fit_ylim"],
            }
        )
        self.log(
            f"Fit range cleared to full spectrum: "
            f"{self.current_region.x_min_cminv:.2f} - {self.current_region.x_max_cminv:.2f} cm$^{{-1}}$"
        )

    def remove_peak(self):
        row = self.widget.tableWidget_Peaks.currentRow()
        if row < 0 or row >= len(self.current_region.peaks):
            return
        preserve_limits = self._current_plot_limits()
        self.current_region.peaks.pop(row)
        self._populate_peak_table()
        self.fit_result = None
        self.widget.tableWidget_Results.setRowCount(0)
        self.widget.plainTextEdit_FitReport.clear()
        self._draw(preserve_limits=preserve_limits)

    def fit_background_model(self):
        if self.spectrum is None:
            QtWidgets.QMessageBox.information(self.widget, "No spectrum", "Load a spectrum first.")
            return
        self._clear_background_mouse_mode()
        raw_ax = self.widget.mpl.canvas.ax_raw
        fit_ax = self.widget.mpl.canvas.ax_fit
        preserve_limits = {
            "raw_xlim": raw_ax.get_xlim(),
            "raw_ylim": raw_ax.get_ylim(),
            "fit_xlim": fit_ax.get_xlim(),
            "fit_ylim": fit_ax.get_ylim(),
        }
        try:
            self.background_result = fit_background(self.current_region, self.spectrum)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self.widget, "Background fit failed", str(exc))
            self.log(f"Background fit failed: {exc}")
            return
        self.fit_result = None
        self.widget.tableWidget_Results.setRowCount(0)
        self.widget.plainTextEdit_FitReport.clear()
        self.widget.plainTextEdit_BackgroundReport.setPlainText(self.background_result.fit_report)
        self._draw(preserve_limits={"raw_xlim": preserve_limits["raw_xlim"], "fit_xlim": preserve_limits["fit_xlim"]})
        self.find_view_minmax()
        self.log(
            "Background fit complete: "
            f"redchi={self.background_result.redchi:.4g}, "
            f"aic={self.background_result.aic:.4g}, "
            f"bic={self.background_result.bic:.4g}"
        )

    def fit_region(self):
        if self.spectrum is None:
            QtWidgets.QMessageBox.information(self.widget, "No spectrum", "Load a spectrum first.")
            return
        if self.background_result is None:
            QtWidgets.QMessageBox.information(self.widget, "Background first", "Fit the background before running PeakFit.")
            return
        preserve_limits = self._current_plot_limits()
        self._clear_peakfit_mouse_modes()
        try:
            self.fit_result = build_fit(self.current_region, self.spectrum, self.background_result)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self.widget, "Fit failed", str(exc))
            self.log(f"Fit failed: {exc}")
            return
        self._update_results_table()
        self.widget.plainTextEdit_FitReport.setPlainText(self.fit_result.fit_report)
        self._draw(
            preserve_limits={
                "raw_xlim": preserve_limits["raw_xlim"],
                "raw_ylim": preserve_limits["raw_ylim"],
                "fit_xlim": preserve_limits["fit_xlim"],
            }
        )
        self._autoscale_bottom_panel()
        self.log(
            f"Fit complete for {self.fit_result.region_name}: "
            f"redchi={self.fit_result.redchi:.4g}, aic={self.fit_result.aic:.4g}, bic={self.fit_result.bic:.4g}"
        )

    def _toggle_fit_range_selector(self, checked: bool):
        if checked:
            self._activate_peakfit_mouse_mode(fit_range=True)
        else:
            self._deactivate_fit_range_selector()

    def _populate_peak_table(self):
        table = self.widget.tableWidget_Peaks
        table.blockSignals(True)
        table.setRowCount(len(self.current_region.peaks))
        for row, peak in enumerate(self.current_region.peaks):
            peak = self._normalized_peak_constraints(peak)
            self.current_region.peaks[row] = peak
            values = [
                peak.name,
                f"{peak.amplitude.value:.4g}",
                f"{self._constraint_minimum(peak.amplitude):.4g}",
                "" if peak.amplitude.max is None else f"{peak.amplitude.max:.4g}",
                f"{peak.center.value:.2f}",
                f"{self._constraint_minimum(peak.center):.2f}",
                "" if peak.center.max is None else f"{peak.center.max:.2f}",
                f"{peak.sigma.value:.2f}",
                f"{self._constraint_minimum(peak.sigma):.2f}",
                "" if peak.sigma.max is None else f"{peak.sigma.max:.2f}",
                f"{peak.fraction.value:.4g}",
                f"{self._constraint_minimum(peak.fraction):.4g}",
                "" if peak.fraction.max is None else f"{peak.fraction.max:.4g}",
            ]
            text_columns = [0, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15, 16]
            for col, text in zip(text_columns, values, strict=False):
                table.setItem(row, col, QtWidgets.QTableWidgetItem(text))
            self._set_vary_checkbox(row, 1, peak.amplitude.vary)
            self._set_vary_checkbox(row, 5, peak.center.vary)
            self._set_vary_checkbox(row, 9, peak.sigma.vary)
            self._set_vary_checkbox(row, 13, peak.fraction.vary)
        table.blockSignals(False)

    def _populate_background_area_table(self):
        table = self.widget.tableWidget_BackgroundAreas
        table.setRowCount(len(self.current_region.background.fit_areas))
        for row, area in enumerate(self.current_region.background.fit_areas):
            values = [f"{area.x_min_cminv:.2f}", f"{area.x_max_cminv:.2f}"]
            for col, text in enumerate(values):
                item = QtWidgets.QTableWidgetItem(text)
                item.setFlags(QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled)
                table.setItem(row, col, item)

    def _populate_sections_table(self):
        table = self.widget.tableWidget_Sections
        table.blockSignals(True)
        table.setRowCount(len(self.saved_sections))
        for row, section in enumerate(self.saved_sections):
            xbg_min, xbg_max = self._section_background_bounds(section)
            xpfit_min, xpfit_max = self._section_peakfit_bounds(section)
            values = [
                section.timestamp,
                section.label,
                f"{xbg_min:.2f}" if xbg_min is not None else "",
                f"{xbg_max:.2f}" if xbg_max is not None else "",
                f"{xpfit_min:.2f}" if xpfit_min is not None else "",
                f"{xpfit_max:.2f}" if xpfit_max is not None else "",
                str(len(section.region.peaks)),
                "Yes" if section.fit_result is not None else "No",
            ]
            for col, text in enumerate(values):
                item = QtWidgets.QTableWidgetItem(text)
                flags = QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled
                if col == 1:
                    flags |= QtCore.Qt.ItemFlag.ItemIsEditable
                item.setFlags(flags)
                table.setItem(row, col, item)
        table.blockSignals(False)

    def _current_plot_limits(self):
        raw_ax = self.widget.mpl.canvas.ax_raw
        fit_ax = self.widget.mpl.canvas.ax_fit
        return {
            "raw_xlim": raw_ax.get_xlim(),
            "raw_ylim": raw_ax.get_ylim(),
            "fit_xlim": fit_ax.get_xlim(),
            "fit_ylim": fit_ax.get_ylim(),
        }

    def _set_button_checked(self, button, checked: bool):
        if button.isChecked() == checked:
            return
        button.blockSignals(True)
        button.setChecked(checked)
        button.blockSignals(False)
        if button.property("status_toggle_base_text") is not None:
            self.widget._update_status_toggle_text(button, checked)

    def _clear_background_mouse_mode(self):
        self._set_button_checked(self.widget.pushButton_SelectBackgroundArea, False)
        self._deactivate_background_area_selector()

    def _clear_peakfit_mouse_modes(self):
        self._set_button_checked(self.widget.pushButton_SetFitRange, False)
        self._set_button_checked(self.widget.pushButton_PickPeaks, False)
        self._deactivate_fit_range_selector()
        self._deactivate_peak_picker()

    def _activate_background_mouse_mode(self):
        self._clear_peakfit_mouse_modes()
        self._clear_background_mouse_mode()
        self._set_button_checked(self.widget.pushButton_SelectBackgroundArea, True)
        self._activate_background_area_selector()

    def _activate_peakfit_mouse_mode(self, *, fit_range: bool = False, peak_pick: bool = False):
        self._clear_background_mouse_mode()
        self._clear_peakfit_mouse_modes()
        if fit_range:
            self._set_button_checked(self.widget.pushButton_SetFitRange, True)
            self._activate_fit_range_selector()
        elif peak_pick:
            self._set_button_checked(self.widget.pushButton_PickPeaks, True)
            self._activate_peak_picker()

    def _sync_view_controls_from_axes(self):
        raw_ax = self.widget.mpl.canvas.ax_raw
        fit_ax = self.widget.mpl.canvas.ax_fit
        controls = [
            (self.widget.doubleSpinBox_TopYMin, raw_ax.get_ylim()[0]),
            (self.widget.doubleSpinBox_TopYMax, raw_ax.get_ylim()[1]),
            (self.widget.doubleSpinBox_BottomYMin, fit_ax.get_ylim()[0]),
            (self.widget.doubleSpinBox_BottomYMax, fit_ax.get_ylim()[1]),
        ]
        for box, value in controls:
            box.blockSignals(True)
            box.setValue(float(value))
            box.blockSignals(False)

    @staticmethod
    def _find_line_range(axis) -> tuple[float, float] | None:
        xlim = axis.get_xlim()
        y_values = []
        for line in axis.lines:
            if line.get_gid() == "helper_overlay":
                continue
            x_data = np.asarray(line.get_xdata(), dtype=float)
            y_data = np.asarray(line.get_ydata(), dtype=float)
            if x_data.size == 0 or y_data.size == 0:
                continue
            mask = (x_data >= min(xlim)) & (x_data <= max(xlim))
            if np.any(mask):
                y_values.append(y_data[mask])
        if not y_values:
            return None
        y_concat = np.concatenate(y_values)
        y_min = float(np.nanmin(y_concat))
        y_max = float(np.nanmax(y_concat))
        if not np.isfinite(y_min) or not np.isfinite(y_max):
            return None
        span = max(y_max - y_min, 1e-9)
        pad = 0.08 * span
        return y_min - pad, y_max + pad

    @staticmethod
    def _section_peakfit_bounds(section: SavedSection) -> tuple[float | None, float | None]:
        if section.fit_result is not None:
            x_values = np.asarray(section.fit_result.x_cminv, dtype=float)
            if x_values.size:
                return float(np.nanmin(x_values)), float(np.nanmax(x_values))
        return float(section.region.x_min_cminv), float(section.region.x_max_cminv)

    @staticmethod
    def _section_background_bounds(section: SavedSection) -> tuple[float | None, float | None]:
        if section.background_result is not None:
            x_values = np.asarray(section.background_result.x_cminv, dtype=float)
            if x_values.size:
                return float(np.nanmin(x_values)), float(np.nanmax(x_values))
        areas = list(section.region.background.fit_areas)
        if areas:
            x_min = min(min(float(area.x_min_cminv), float(area.x_max_cminv)) for area in areas)
            x_max = max(max(float(area.x_min_cminv), float(area.x_max_cminv)) for area in areas)
            return x_min, x_max
        return None, None

    def find_view_minmax(self):
        raw_ax = self.widget.mpl.canvas.ax_raw
        fit_ax = self.widget.mpl.canvas.ax_fit
        raw_range = self._find_line_range(raw_ax)
        fit_range = self._find_line_range(fit_ax)
        if raw_range is not None:
            raw_ax.set_ylim(*raw_range)
        if fit_range is not None:
            fit_ax.set_ylim(*fit_range)
        self._sync_view_controls_from_axes()
        self.widget.mpl.canvas.draw_idle()

    def _autoscale_bottom_panel(self):
        fit_ax = self.widget.mpl.canvas.ax_fit
        fit_range = None
        if self.fit_result is not None:
            xlim = fit_ax.get_xlim()
            x = np.asarray(self.fit_result.x_cminv, dtype=float)
            mask = (x >= min(xlim)) & (x <= max(xlim))
            if np.any(mask):
                y_values = [
                    np.asarray(self.fit_result.y_bgsub, dtype=float)[mask],
                    np.asarray(self.fit_result.best_fit_bgsub, dtype=float)[mask],
                ]
                for peak in self.fit_result.peaks:
                    y_values.append(np.asarray(peak.curve, dtype=float)[mask])
                residual = np.asarray(self.fit_result.residual_bgsub, dtype=float)
                fit_span = float(np.nanmax(self.fit_result.y_bgsub) - np.nanmin(self.fit_result.y_bgsub)) if self.fit_result.y_bgsub.size else 0.0
                fit_residual_span = float(np.nanmax(residual) - np.nanmin(residual)) if residual.size else 0.0
                fit_residual_offset = float(
                    np.nanmin(self.fit_result.y_bgsub) - max(0.08 * max(fit_span, 1.0), 1.5 * fit_residual_span)
                )
                y_values.append((residual + fit_residual_offset)[mask])
                y_concat = np.concatenate(y_values)
                y_min = float(np.nanmin(y_concat))
                y_max = float(np.nanmax(y_concat))
                span = max(y_max - y_min, 1e-9)
                fit_range = (y_min - 0.08 * span, y_max + 0.08 * span)
        if fit_range is None:
            fit_range = self._find_line_range(fit_ax)
        if fit_range is not None:
            fit_ax.set_ylim(*fit_range)
            self._sync_view_controls_from_axes()
            self.widget.mpl.canvas.draw_idle()

    def apply_view_limits(self):
        raw_min = self.widget.doubleSpinBox_TopYMin.value()
        raw_max = self.widget.doubleSpinBox_TopYMax.value()
        fit_min = self.widget.doubleSpinBox_BottomYMin.value()
        fit_max = self.widget.doubleSpinBox_BottomYMax.value()
        if raw_max <= raw_min or fit_max <= fit_min:
            QtWidgets.QMessageBox.warning(self.widget, "Invalid limits", "Each max must be greater than its min.")
            return
        raw_ax = self.widget.mpl.canvas.ax_raw
        fit_ax = self.widget.mpl.canvas.ax_fit
        raw_ax.set_ylim(raw_min, raw_max)
        fit_ax.set_ylim(fit_min, fit_max)
        self.widget.mpl.canvas.draw_idle()

    def zoom_out_full_region(self):
        if self.spectrum is None:
            return
        raw_ax = self.widget.mpl.canvas.ax_raw
        fit_ax = self.widget.mpl.canvas.ax_fit
        x_values = np.asarray(self.spectrum.x_cminv, dtype=float)
        if x_values.size == 0:
            return
        raw_ax.set_xlim(float(np.nanmin(x_values)), float(np.nanmax(x_values)))
        fit_ax.set_xlim(float(np.nanmin(x_values)), float(np.nanmax(x_values)))
        self.find_view_minmax()

    def zoom_in_active_region(self):
        if self.spectrum is None:
            return
        if self.fit_result is not None:
            x_values = np.asarray(self.fit_result.x_cminv, dtype=float)
        elif self.background_result is not None:
            x_values = np.asarray(self.background_result.x_cminv, dtype=float)
        else:
            x_values = np.asarray([], dtype=float)
        if x_values.size == 0:
            x_min = float(self.current_region.x_min_cminv)
            x_max = float(self.current_region.x_max_cminv)
        else:
            x_min = float(np.nanmin(x_values))
            x_max = float(np.nanmax(x_values))
        raw_ax = self.widget.mpl.canvas.ax_raw
        fit_ax = self.widget.mpl.canvas.ax_fit
        raw_ax.set_xlim(x_min, x_max)
        fit_ax.set_xlim(x_min, x_max)
        self.find_view_minmax()

    def adjust_y_for_spectrum(self):
        if self.spectrum is None:
            return
        raw_ax = self.widget.mpl.canvas.ax_raw
        fit_ax = self.widget.mpl.canvas.ax_fit
        xlim_raw = raw_ax.get_xlim()
        xlim_fit = fit_ax.get_xlim()

        x_raw = np.asarray(self.spectrum.x_cminv, dtype=float)
        y_raw = np.asarray(self.spectrum.intensity, dtype=float)
        raw_mask = (x_raw >= min(xlim_raw)) & (x_raw <= max(xlim_raw))
        if np.any(raw_mask):
            raw_vals = y_raw[raw_mask]
            raw_min = float(np.nanmin(raw_vals))
            raw_max = float(np.nanmax(raw_vals))
            raw_span = max(raw_max - raw_min, 1e-9)
            raw_ax.set_ylim(raw_min - 0.08 * raw_span, raw_max + 0.08 * raw_span)

        if self.background_result is not None:
            x_fit = np.asarray(self.background_result.x_cminv, dtype=float)
            y_fit = np.asarray(self.background_result.y_bgsub, dtype=float)
            fit_mask = (x_fit >= min(xlim_fit)) & (x_fit <= max(xlim_fit))
            if np.any(fit_mask):
                fit_vals = y_fit[fit_mask]
                fit_min = float(np.nanmin(fit_vals))
                fit_max = float(np.nanmax(fit_vals))
                fit_span = max(fit_max - fit_min, 1e-9)
                fit_ax.set_ylim(fit_min - 0.08 * fit_span, fit_max + 0.08 * fit_span)
        else:
            fit_mask = (x_raw >= min(xlim_fit)) & (x_raw <= max(xlim_fit))
            if np.any(fit_mask):
                fit_vals = y_raw[fit_mask]
                fit_min = float(np.nanmin(fit_vals))
                fit_max = float(np.nanmax(fit_vals))
                fit_span = max(fit_max - fit_min, 1e-9)
                fit_ax.set_ylim(fit_min - 0.08 * fit_span, fit_max + 0.08 * fit_span)

        self._sync_view_controls_from_axes()
        self.widget.mpl.canvas.draw_idle()

    def remove_selected_background_area(self):
        row = self.widget.tableWidget_BackgroundAreas.currentRow()
        if row < 0 or row >= len(self.current_region.background.fit_areas):
            return
        preserve_limits = self._current_plot_limits()
        self.current_region.background.fit_areas.pop(row)
        self._clear_background_results()
        self._populate_background_area_table()
        self._draw(preserve_limits=preserve_limits)

    def clear_background_areas(self):
        if not self.current_region.background.fit_areas:
            return
        preserve_limits = self._current_plot_limits()
        self.current_region.background.fit_areas = []
        self.widget.pushButton_PickPeaks.setChecked(False)
        self._clear_background_results()
        self._populate_background_area_table()
        self._draw(preserve_limits=preserve_limits)

    def _clear_background_results(self):
        if self.background_result is not None:
            self.widget.pushButton_PickPeaks.setChecked(False)
        self.background_result = None
        self.fit_result = None
        self.widget.plainTextEdit_BackgroundReport.clear()
        self.widget.tableWidget_Results.setRowCount(0)
        self.widget.plainTextEdit_FitReport.clear()

    def _clear_peak_fit_results(self):
        self.fit_result = None
        self.widget.tableWidget_Results.setRowCount(0)
        self.widget.plainTextEdit_FitReport.clear()

    def _activate_fit_range_selector(self):
        if self.spectrum is None:
            self.widget.pushButton_SetFitRange.setChecked(False)
            return
        toolbar = self.widget.mpl.ntb
        if getattr(toolbar, "mode", ""):
            if "zoom" in toolbar.mode.lower():
                toolbar.zoom()
            elif "pan" in toolbar.mode.lower():
                toolbar.pan()
        canvas = self.widget.mpl.canvas
        self._deactivate_fit_range_selector()
        self._fit_range_press_x = None
        self._fit_range_press_cid = canvas.mpl_connect("button_press_event", self._on_fit_range_press)
        self._fit_range_motion_cid = canvas.mpl_connect("motion_notify_event", self._on_fit_range_motion)
        self._fit_range_release_cid = canvas.mpl_connect("button_release_event", self._on_fit_range_release)
        self.log("Drag on the bottom plot to set the fit range.")

    def _deactivate_fit_range_selector(self):
        canvas = self.widget.mpl.canvas
        for attr in ("_fit_range_press_cid", "_fit_range_motion_cid", "_fit_range_release_cid"):
            cid = getattr(self, attr)
            if cid is not None:
                canvas.mpl_disconnect(cid)
                setattr(self, attr, None)
        self._fit_range_press_x = None
        if self._fit_range_preview is not None:
            try:
                self._fit_range_preview.remove()
            except Exception:
                pass
            self._fit_range_preview = None
        canvas.draw_idle()

    def _on_fit_range_press(self, event):
        if event.inaxes is not self.widget.mpl.canvas.ax_fit or event.xdata is None:
            return
        self._fit_range_press_x = float(event.xdata)
        self._update_fit_range_preview(float(event.xdata))

    def _on_fit_range_motion(self, event):
        if self._fit_range_press_x is None or event.inaxes is not self.widget.mpl.canvas.ax_fit or event.xdata is None:
            return
        self._update_fit_range_preview(float(event.xdata))

    def _on_fit_range_release(self, event):
        if self._fit_range_press_x is None:
            return
        preserve_limits = self._current_plot_limits()
        x_release = self._fit_range_press_x if event.xdata is None else float(event.xdata)
        x0 = min(self._fit_range_press_x, x_release)
        x1 = max(self._fit_range_press_x, x_release)
        if abs(x1 - x0) > 1e-9:
            self.current_region.x_min_cminv = max(x0, 0.0)
            self.current_region.x_max_cminv = max(x1, self.current_region.x_min_cminv)
            self._clear_peak_fit_results()
            self._draw(preserve_limits=preserve_limits)
            self.log(
                f"Fit range set to {self.current_region.x_min_cminv:.2f} - "
                f"{self.current_region.x_max_cminv:.2f} cm$^{{-1}}$"
            )
        self.widget.pushButton_SetFitRange.setChecked(False)

    def _update_fit_range_preview(self, x_current: float):
        ax = self.widget.mpl.canvas.ax_fit
        x0 = min(float(self._fit_range_press_x), float(x_current))
        x1 = max(float(self._fit_range_press_x), float(x_current))
        if self._fit_range_preview is not None:
            try:
                self._fit_range_preview.remove()
            except Exception:
                pass
        self._fit_range_preview = ax.axvspan(x0, x1, color="#3b82f6", alpha=0.18)
        self.widget.mpl.canvas.draw_idle()

    def _toggle_background_area_selector(self, checked: bool):
        if checked:
            self._activate_background_mouse_mode()
        else:
            self._deactivate_background_area_selector()

    def _activate_background_area_selector(self):
        if self.spectrum is None:
            self.widget.pushButton_SelectBackgroundArea.setChecked(False)
            return
        toolbar = self.widget.mpl.ntb
        if getattr(toolbar, "mode", ""):
            if "zoom" in toolbar.mode.lower():
                toolbar.zoom()
            elif "pan" in toolbar.mode.lower():
                toolbar.pan()
        canvas = self.widget.mpl.canvas
        self._deactivate_background_area_selector()
        self._bg_area_press_x = None
        self._bg_area_press_cid = canvas.mpl_connect("button_press_event", self._on_background_area_press)
        self._bg_area_motion_cid = canvas.mpl_connect("motion_notify_event", self._on_background_area_motion)
        self._bg_area_release_cid = canvas.mpl_connect("button_release_event", self._on_background_area_release)
        self.log("Drag on the top plot to add a background fit area.")

    def _deactivate_background_area_selector(self):
        canvas = self.widget.mpl.canvas
        for attr in ("_bg_area_press_cid", "_bg_area_motion_cid", "_bg_area_release_cid"):
            cid = getattr(self, attr)
            if cid is not None:
                canvas.mpl_disconnect(cid)
                setattr(self, attr, None)
        self._bg_area_press_x = None
        if self._bg_area_preview is not None:
            try:
                self._bg_area_preview.remove()
            except Exception:
                pass
            self._bg_area_preview = None
        canvas.draw_idle()

    def _toggle_peak_picker(self, checked: bool):
        if checked:
            self._activate_peakfit_mouse_mode(peak_pick=True)
        else:
            self._deactivate_peak_picker()

    def _activate_peak_picker(self):
        if self.background_result is None:
            self.widget.pushButton_PickPeaks.setChecked(False)
            QtWidgets.QMessageBox.information(
                self.widget,
                "Background first",
                "Fit the background before picking peaks on the bg-subtracted spectrum.",
            )
            return
        toolbar = self.widget.mpl.ntb
        if getattr(toolbar, "mode", ""):
            if "zoom" in toolbar.mode.lower():
                toolbar.zoom()
            elif "pan" in toolbar.mode.lower():
                toolbar.pan()
        canvas = self.widget.mpl.canvas
        self._deactivate_peak_picker()
        self._peak_pick_press_x = None
        self._peak_pick_press_button = None
        self._peak_pick_press_cid = canvas.mpl_connect("button_press_event", self._on_peak_pick_press)
        self._peak_pick_motion_cid = canvas.mpl_connect("motion_notify_event", self._on_peak_pick_motion)
        self._peak_pick_release_cid = canvas.mpl_connect("button_release_event", self._on_peak_pick_release)
        self.log("Peak picker active: move to preview, left-drag from left FWHM edge to right FWHM edge, right click removes nearest peak.")

    def _deactivate_peak_picker(self):
        canvas = self.widget.mpl.canvas
        for attr in ("_peak_pick_press_cid", "_peak_pick_motion_cid", "_peak_pick_release_cid"):
            cid = getattr(self, attr)
            if cid is not None:
                canvas.mpl_disconnect(cid)
                setattr(self, attr, None)
        self._peak_pick_press_x = None
        self._peak_pick_press_button = None
        if self._peak_pick_preview is not None:
            try:
                self._peak_pick_preview.remove()
            except Exception:
                pass
            self._peak_pick_preview = None
        canvas.draw_idle()

    def _on_peak_pick_press(self, event):
        if event.inaxes is not self.widget.mpl.canvas.ax_fit or event.xdata is None:
            return
        if event.button == 3:
            self._remove_nearest_peak(float(event.xdata))
            return
        if event.button != 1:
            return
        self._peak_pick_press_x = float(event.xdata)
        self._peak_pick_press_button = event.button
        self._update_peak_pick_preview(float(event.xdata))

    def _on_peak_pick_motion(self, event):
        if event.inaxes is not self.widget.mpl.canvas.ax_fit or event.xdata is None:
            return
        self._update_peak_pick_preview(float(event.xdata))

    def _on_peak_pick_release(self, event):
        if self._peak_pick_press_x is None:
            return
        if self.background_result is None:
            self.widget.pushButton_PickPeaks.setChecked(False)
            return
        preserve_limits = self._current_plot_limits()
        x_release = self._peak_pick_press_x if event.xdata is None else float(event.xdata)
        x0 = min(self._peak_pick_press_x, x_release)
        x1 = max(self._peak_pick_press_x, x_release)
        if abs(x1 - x0) <= 1e-9:
            self._upsert_peak_from_click(x_release)
        else:
            self._upsert_peak_from_drag(x0, x1)
        self._reset_peak_pick_gesture()
        self._draw(preserve_limits=preserve_limits)

    def _update_peak_pick_preview(self, x_current: float):
        ax = self.widget.mpl.canvas.ax_fit
        center, sigma, height = self._preview_peak_parameters(float(x_current))
        amplitude = self._pseudo_voigt_height_to_amplitude(height, sigma, 0.5)
        x_axis, y_axis = self._peak_profile_curve(center, sigma, amplitude, 0.5)
        if x_axis.size == 0:
            return
        if self._peak_pick_preview is not None:
            try:
                self._peak_pick_preview.remove()
            except Exception:
                pass
        (self._peak_pick_preview,) = ax.plot(x_axis, y_axis, color="#22c55e", linewidth=1.3, linestyle="--")
        self.widget.mpl.canvas.draw_idle()

    def _reset_peak_pick_gesture(self):
        self._peak_pick_press_x = None
        self._peak_pick_press_button = None
        if self._peak_pick_preview is not None:
            try:
                self._peak_pick_preview.remove()
            except Exception:
                pass
            self._peak_pick_preview = None

    @staticmethod
    def _default_peak_pick_sigma() -> float:
        # lmfit's pseudo-Voigt uses sigma such that FWHM ~= 2 * sigma.
        return 0.5

    def _peak_signal_y(self, x_value: float) -> float:
        if self.background_result is None:
            return 0.0
        x = np.asarray(self.background_result.x_cminv, dtype=float)
        y = np.asarray(self.background_result.y_bgsub, dtype=float)
        if x.size == 0:
            return 0.0
        x_clamped = min(max(float(x_value), float(np.min(x))), float(np.max(x)))
        return float(np.interp(x_clamped, x, y))

    def _preview_peak_parameters(self, x_current: float) -> tuple[float, float, float]:
        if self._peak_pick_press_x is None:
            center = float(x_current)
            sigma = self._default_peak_pick_sigma()
        else:
            x0 = min(float(self._peak_pick_press_x), float(x_current))
            x1 = max(float(self._peak_pick_press_x), float(x_current))
            if abs(x1 - x0) <= 1e-9:
                center = float(x_current)
                sigma = self._default_peak_pick_sigma()
            else:
                center = 0.5 * (x0 + x1)
                sigma = max((x1 - x0) / 2.0, 1e-5)
        height = max(self._peak_signal_y(center), 0.0)
        return center, sigma, height

    def _peak_profile_curve(
        self,
        center: float,
        sigma: float,
        amplitude: float,
        fraction: float,
        x_axis: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.background_result is None:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        x = np.asarray(
            self.background_result.x_cminv if x_axis is None else x_axis,
            dtype=float,
        )
        if x.size == 0:
            return x, np.asarray([], dtype=float)
        mask = (x >= self.current_region.x_min_cminv) & (x <= self.current_region.x_max_cminv)
        x_use = x[mask]
        if x_use.size == 0:
            return x_use, np.asarray([], dtype=float)
        sigma = max(float(sigma), 1e-5)
        fraction = min(max(float(fraction), 0.0), 1.0)
        dx = x_use - float(center)
        gaussian_sigma = sigma / np.sqrt(2.0 * np.log(2.0))
        gaussian = np.exp(-0.5 * np.square(dx / gaussian_sigma))
        lorentz = 1.0 / (1.0 + np.square(dx / sigma))
        gaussian_norm = 1.0 / (gaussian_sigma * np.sqrt(2.0 * np.pi))
        lorentz_norm = 1.0 / (np.pi * sigma)
        profile = float(amplitude) * (
            ((1.0 - fraction) * gaussian_norm * gaussian) + (fraction * lorentz_norm * lorentz)
        )
        return x_use, profile

    @staticmethod
    def _pseudo_voigt_height_to_amplitude(height: float, sigma: float, fraction: float) -> float:
        sigma = max(float(sigma), 1e-5)
        fraction = min(max(float(fraction), 0.0), 1.0)
        gaussian_sigma = sigma / np.sqrt(2.0 * np.log(2.0))
        gaussian_norm = 1.0 / (gaussian_sigma * np.sqrt(2.0 * np.pi))
        lorentz_norm = 1.0 / (np.pi * sigma)
        peak_scale = ((1.0 - fraction) * gaussian_norm) + (fraction * lorentz_norm)
        if peak_scale <= 0.0:
            return 0.0
        return float(height) / peak_scale

    def _selected_peak_index(self) -> int | None:
        row = self.widget.tableWidget_Peaks.currentRow()
        if 0 <= row < len(self.current_region.peaks):
            return row
        return None

    def _set_peak_selection(self, row: int):
        if 0 <= row < self.widget.tableWidget_Peaks.rowCount():
            self.widget.tableWidget_Peaks.selectRow(row)

    def _upsert_peak_from_click(self, x_center: float):
        amplitude = max(self._peak_signal_y(x_center), 0.0)
        width = self._default_peak_pick_sigma()
        x0 = max(self.current_region.x_min_cminv, x_center - 2.0 * width)
        x1 = min(self.current_region.x_max_cminv, x_center + 2.0 * width)
        peak = self._build_peak_guess(x_center, x0, x1, amplitude)
        self._append_picked_peak(peak)

    def _upsert_peak_from_drag(self, x0: float, x1: float):
        center = 0.5 * (x0 + x1)
        amplitude = max(self._peak_signal_y(center), 0.0)
        peak = self._build_peak_guess(center, x0, x1, amplitude)
        self._append_picked_peak(peak)

    def _build_peak_guess(self, center: float, x0: float, x1: float, amplitude: float) -> PeakSpec:
        x_min = max(self.current_region.x_min_cminv, min(float(x0), float(x1)))
        x_max = min(self.current_region.x_max_cminv, max(float(x0), float(x1)))
        span = max(x_max - x_min, 1e-6)
        sigma = max(span / 2.0, 1e-5)
        row = self._selected_peak_index()
        name = self.current_region.peaks[row].name if row is not None else f"p{len(self.current_region.peaks) + 1}"
        model_amplitude = self._pseudo_voigt_height_to_amplitude(max(amplitude, 0.0), sigma, 0.5)
        amp_min, amp_max = self._default_bounds(model_amplitude, lower_floor=0.0)
        center_min, center_max = self._default_bounds(center, lower_floor=0.0)
        sigma_min, sigma_max = self._default_bounds(sigma, lower_floor=0.0)
        fraction_min, fraction_max = 0.0, 1.0
        return PeakSpec(
            name=name,
            guess_min_cminv=x_min,
            guess_max_cminv=x_max,
            amplitude=ParameterConstraint(value=max(model_amplitude, 0.0), vary=True, min=amp_min, max=amp_max),
            center=ParameterConstraint(value=max(float(center), 0.0), vary=True, min=center_min, max=center_max),
            sigma=ParameterConstraint(value=sigma, vary=True, min=sigma_min, max=sigma_max),
            fraction=ParameterConstraint(value=0.5, vary=True, min=fraction_min, max=fraction_max),
        )

    def _upsert_peak(self, peak: PeakSpec):
        row = self._selected_peak_index()
        if row is None:
            self.current_region.peaks.append(peak)
            row = len(self.current_region.peaks) - 1
        else:
            self.current_region.peaks[row] = peak
        self._clear_peak_fit_results()
        self._populate_peak_table()
        self._set_peak_selection(row)

    def _append_picked_peak(self, peak: PeakSpec):
        self.current_region.peaks.append(peak)
        self._clear_peak_fit_results()
        self._populate_peak_table()
        self.widget.tableWidget_Peaks.clearSelection()

    def _remove_nearest_peak(self, x_center: float):
        if not self.current_region.peaks:
            return
        preserve_limits = self._current_plot_limits()
        centers = [abs(float(peak.center.value) - float(x_center)) for peak in self.current_region.peaks]
        row = int(np.argmin(centers))
        self.current_region.peaks.pop(row)
        self._clear_peak_fit_results()
        self._populate_peak_table()
        if self.current_region.peaks:
            self._set_peak_selection(min(row, len(self.current_region.peaks) - 1))
        self._draw(preserve_limits=preserve_limits)

    def _on_background_area_press(self, event):
        if event.inaxes is not self.widget.mpl.canvas.ax_raw or event.xdata is None:
            return
        if event.button == 3:
            self._remove_nearest_background_area(float(event.xdata))
            return
        if event.button != 1:
            return
        self._bg_area_press_x = float(event.xdata)
        self._update_background_area_preview(float(event.xdata))

    def _on_background_area_motion(self, event):
        if self._bg_area_press_x is None or event.inaxes is not self.widget.mpl.canvas.ax_raw or event.xdata is None:
            return
        self._update_background_area_preview(float(event.xdata))

    def _on_background_area_release(self, event):
        if self._bg_area_press_x is None:
            return
        preserve_limits = self._current_plot_limits()
        x_release = self._bg_area_press_x if event.xdata is None else float(event.xdata)
        x0 = min(self._bg_area_press_x, x_release)
        x1 = max(self._bg_area_press_x, x_release)
        if abs(x1 - x0) > 1e-9:
            self.current_region.background.fit_areas.append(BackgroundArea(x0, x1))
            self.current_region.background.fit_areas.sort(key=lambda area: min(area.x_min_cminv, area.x_max_cminv))
            self._clear_background_results()
            self._populate_background_area_table()
            self._draw(preserve_limits=preserve_limits)
        self._bg_area_press_x = None
        if self._bg_area_preview is not None:
            try:
                self._bg_area_preview.remove()
            except Exception:
                pass
            self._bg_area_preview = None
        self.widget.mpl.canvas.draw_idle()

    def _update_background_area_preview(self, x_current: float):
        ax = self.widget.mpl.canvas.ax_raw
        x0 = min(float(self._bg_area_press_x), float(x_current))
        x1 = max(float(self._bg_area_press_x), float(x_current))
        if self._bg_area_preview is not None:
            try:
                self._bg_area_preview.remove()
            except Exception:
                pass
        self._bg_area_preview = ax.axvspan(x0, x1, color="#fbbf24", alpha=0.20)
        self.widget.mpl.canvas.draw_idle()

    def _remove_nearest_background_area(self, x_center: float):
        areas = self.current_region.background.fit_areas
        if not areas:
            return
        preserve_limits = self._current_plot_limits()
        centers = [
            abs((0.5 * (float(area.x_min_cminv) + float(area.x_max_cminv))) - float(x_center))
            for area in areas
        ]
        row = int(np.argmin(centers))
        areas.pop(row)
        self._clear_background_results()
        self._populate_background_area_table()
        self._draw(preserve_limits=preserve_limits)

    def handle_peak_table_change(self, _item):
        table = self.widget.tableWidget_Peaks
        peaks = []
        for row in range(table.rowCount()):
            name = self._table_text(row, 0, table) or f"p{row + 1}"
            center_value = self._table_float(row, 6, 0.0)
            center_min = self._table_optional_float(row, 7, default=0.0)
            center_max = self._table_optional_float(row, 8)
            guess_min = center_min if center_min is not None else center_value
            guess_max = center_max if center_max is not None else center_value
            peaks.append(
                PeakSpec(
                    name=name,
                    guess_min_cminv=guess_min,
                    guess_max_cminv=max(guess_max, guess_min),
                    amplitude=ParameterConstraint(
                        value=self._table_float(row, 2, 100.0),
                        vary=self._table_checkbox(row, 1),
                        min=self._table_optional_float(row, 3, default=0.0),
                        max=self._table_optional_float(row, 4),
                    ),
                    center=ParameterConstraint(
                        value=center_value,
                        vary=self._table_checkbox(row, 5),
                        min=center_min,
                        max=center_max,
                    ),
                    sigma=ParameterConstraint(
                        value=self._table_float(row, 10, 25.0),
                        vary=self._table_checkbox(row, 9),
                        min=self._table_optional_float(row, 11, default=0.0),
                        max=self._table_optional_float(row, 12),
                    ),
                    fraction=ParameterConstraint(
                        value=self._table_float(row, 14, 0.5),
                        vary=self._table_checkbox(row, 13),
                        min=self._table_optional_float(row, 15, default=0.0),
                        max=self._table_optional_float(row, 16),
                    ),
                )
            )
        self.current_region.peaks = [self._normalized_peak_constraints(peak) for peak in peaks]
        self._clear_peak_fit_results()

    def handle_section_table_change(self, item):
        if item is None or item.column() != 1:
            return
        row = item.row()
        if 0 <= row < len(self.saved_sections):
            self.saved_sections[row].label = item.text().strip()

    def _update_results_table(self):
        table = self.widget.tableWidget_Results
        if self.fit_result is None:
            table.setRowCount(0)
            return
        table.setRowCount(len(self.fit_result.peaks))
        for row, peak in enumerate(self.fit_result.peaks):
            values = [
                peak.name,
                f"{peak.center_cminv:.2f}",
                f"{peak.amplitude:.4g}",
                f"{peak.sigma_cminv:.2f}",
                f"{peak.fraction:.4g}",
            ]
            for col, text in enumerate(values):
                item = QtWidgets.QTableWidgetItem(text)
                item.setFlags(QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled)
                table.setItem(row, col, item)

    def _draw(self, preserve_limits=None):
        canvas = self.widget.mpl.canvas
        canvas.clear()
        raw_ax = canvas.ax_raw
        fit_ax = canvas.ax_fit
        region = self.current_region
        has_explicit_fit_range = False

        def region_mask(x_values):
            x_values = np.asarray(x_values, dtype=float)
            return (x_values >= region.x_min_cminv) & (x_values <= region.x_max_cminv)

        if self.spectrum is not None:
            spectrum_x = np.asarray(self.spectrum.x_cminv, dtype=float)
            if spectrum_x.size:
                full_x_min = float(np.nanmin(spectrum_x))
                full_x_max = float(np.nanmax(spectrum_x))
                has_explicit_fit_range = (
                    abs(region.x_min_cminv - full_x_min) > 1e-9
                    or abs(region.x_max_cminv - full_x_max) > 1e-9
                )
            raw_ax.plot(
                self.spectrum.x_cminv,
                self.spectrum.intensity,
                linestyle="-",
                color="#f2f2f2",
                linewidth=1.0,
                label="vibEELS data",
            )
            if has_explicit_fit_range:
                raw_ax.axvspan(region.x_min_cminv, region.x_max_cminv, color="#3b82f6", alpha=0.12, label="Fit window")
            for idx, area in enumerate(region.background.fit_areas):
                label = "Bg fit area" if idx == 0 else None
                raw_ax.axvspan(
                    min(area.x_min_cminv, area.x_max_cminv),
                    max(area.x_min_cminv, area.x_max_cminv),
                    color="#f59e0b",
                    alpha=0.12,
                    label=label,
                )
            for peak in region.peaks:
                raw_ax.axvspan(peak.guess_min_cminv, peak.guess_max_cminv, color="#10b981", alpha=0.08)

        if self.background_result is not None:
            bg = self.background_result
            if self.fit_result is None:
                raw_ax.plot(bg.x_cminv, bg.background, color="#60a5fa", linewidth=1.0, label="background")
                bg_residual = np.asarray(bg.y_raw - bg.background, dtype=float)
                top_mask = region_mask(bg.x_cminv)
                if np.any(top_mask):
                    top_y = np.asarray(bg.y_raw[top_mask], dtype=float)
                    top_residual = np.asarray(bg_residual[top_mask], dtype=float)
                else:
                    top_y = np.asarray(bg.y_raw, dtype=float)
                    top_residual = np.asarray(bg_residual, dtype=float)
                raw_span = float(np.nanmax(top_y) - np.nanmin(top_y)) if top_y.size else 0.0
                residual_span = float(np.nanmax(top_residual) - np.nanmin(top_residual)) if top_residual.size else 0.0
                top_zero_line = float(np.nanmin(top_y) - max(0.06 * max(raw_span, 1.0), 1.5 * residual_span))
                raw_ax.axhline(
                    top_zero_line,
                    color="#a1a1aa",
                    linewidth=0.8,
                    linestyle="--",
                    label="shifted zero",
                ).set_gid("helper_overlay")
                residual_y = bg_residual + top_zero_line
                raw_ax.fill_between(
                    bg.x_cminv,
                    top_zero_line,
                    residual_y,
                    color="#e5e7eb",
                    alpha=0.28,
                    label="residue",
                )
                raw_ax.plot(bg.x_cminv, residual_y, color="#e5e7eb", linewidth=0.9)[0].set_gid("helper_overlay")
            fit_mask = region_mask(bg.x_cminv)
            fit_x = bg.x_cminv[fit_mask]
            fit_y = bg.y_bgsub[fit_mask]
            fit_ax.plot(fit_x, fit_y, color="#f2f2f2", linewidth=1.0, label="_nolegend_", zorder=99)
            fit_ax.axhline(0.0, color="#52525b", linewidth=0.8, linestyle="--")
            if self.fit_result is None:
                for idx, peak in enumerate(self.current_region.peaks):
                    x_curve, y_curve = self._peak_profile_curve(
                        peak.center.value,
                        peak.sigma.value,
                        peak.amplitude.value,
                        peak.fraction.value,
                    )
                    if x_curve.size == 0:
                        continue
                    label = f"{peak.name} guess" if idx == 0 else None
                    fit_ax.plot(x_curve, y_curve, color="#22c55e", linewidth=1.0, linestyle="--", label=label)

        if self.fit_result is not None:
            result = self.fit_result
            raw_ax.plot(result.x_cminv, result.best_fit, color="#ff4d6d", linewidth=1.0, label="best fit", zorder=99)
            raw_span = float(np.nanmax(result.y_raw) - np.nanmin(result.y_raw)) if result.y_raw.size else 0.0
            residual_span = float(np.nanmax(result.residual_raw) - np.nanmin(result.residual_raw)) if result.residual_raw.size else 0.0
            top_zero_line = float(np.nanmin(result.y_raw) - max(0.06 * max(raw_span, 1.0), 1.5 * residual_span))
            raw_ax.axhline(
                top_zero_line,
                color="#a1a1aa",
                linewidth=0.8,
                linestyle="--",
                label="shifted zero",
            ).set_gid("helper_overlay")
            residual_y = result.residual_raw + top_zero_line
            raw_ax.fill_between(
                result.x_cminv,
                top_zero_line,
                residual_y,
                color="#e5e7eb",
                alpha=0.28,
                label="residue",
            )
            raw_ax.plot(result.x_cminv, residual_y, color="#e5e7eb", linewidth=0.9)[0].set_gid("helper_overlay")

            fit_ax.plot(result.x_cminv, result.y_bgsub, color="#f2f2f2", linewidth=1.0, label="_nolegend_", zorder=99)
            fit_ax.plot(result.x_cminv, result.best_fit_bgsub, color="#ff4d6d", linewidth=1.0, label="best fit")
            fit_span = float(np.nanmax(result.y_bgsub) - np.nanmin(result.y_bgsub)) if result.y_bgsub.size else 0.0
            fit_residual_span = float(np.nanmax(result.residual_bgsub) - np.nanmin(result.residual_bgsub)) if result.residual_bgsub.size else 0.0
            fit_residual_offset = float(np.nanmin(result.y_bgsub) - max(0.08 * max(fit_span, 1.0), 1.5 * fit_residual_span))
            for peak in sorted(result.peaks, key=lambda peak_result: float(peak_result.center_cminv)):
                fit_ax.plot(result.x_cminv, peak.curve, linewidth=1.0, label=f"{peak.center_cminv:.0f} cm$^{{-1}}$")
            fit_ax.axhline(
                fit_residual_offset,
                color="#a1a1aa",
                linewidth=0.8,
                linestyle="--",
                label="_nolegend_",
            ).set_gid("helper_overlay")
            fit_residual_y = result.residual_bgsub + fit_residual_offset
            fit_ax.fill_between(
                result.x_cminv,
                fit_residual_offset,
                fit_residual_y,
                color="#93c5fd",
                alpha=0.25,
                label="residue",
            )
            fit_ax.plot(
                result.x_cminv,
                fit_residual_y,
                color="#93c5fd",
                linewidth=0.9,
                label="_nolegend_",
            )[0].set_gid("helper_overlay")
        elif self.spectrum is not None:
            mask = (self.spectrum.x_cminv >= self.current_region.x_min_cminv) & (self.spectrum.x_cminv <= self.current_region.x_max_cminv)
            if self.background_result is None:
                fit_label = "Selected region" if has_explicit_fit_range else "_nolegend_"
                fit_ax.plot(self.spectrum.x_cminv[mask], self.spectrum.intensity[mask], color="#d0d0d0", linewidth=1.0, label=fit_label)

        raw_handles, raw_labels = raw_ax.get_legend_handles_labels()
        raw_pairs = [(handle, label) for handle, label in zip(raw_handles, raw_labels, strict=False) if label and not label.startswith("_")]
        if raw_pairs:
            raw_ax.legend(
                [handle for handle, _ in raw_pairs],
                [label for _, label in raw_pairs],
                loc="upper right",
                fontsize=8,
            )
        fit_handles, fit_labels = fit_ax.get_legend_handles_labels()
        fit_pairs = [(handle, label) for handle, label in zip(fit_handles, fit_labels, strict=False) if label and not label.startswith("_")]
        if fit_pairs:
            fit_ax.legend(
                [handle for handle, _ in fit_pairs],
                [label for _, label in fit_pairs],
                loc="upper right",
                fontsize=8,
            )
        if preserve_limits is None and self.background_result is not None:
            raw_ax.set_xlim(region.x_min_cminv, region.x_max_cminv)
            fit_ax.set_xlim(region.x_min_cminv, region.x_max_cminv)
        if preserve_limits is not None:
            if "raw_xlim" in preserve_limits:
                raw_ax.set_xlim(*preserve_limits["raw_xlim"])
            if "raw_ylim" in preserve_limits:
                raw_ax.set_ylim(*preserve_limits["raw_ylim"])
            if "fit_xlim" in preserve_limits:
                fit_ax.set_xlim(*preserve_limits["fit_xlim"])
            if "fit_ylim" in preserve_limits:
                fit_ax.set_ylim(*preserve_limits["fit_ylim"])
        self._sync_view_controls_from_axes()
        canvas.draw_idle()

    @staticmethod
    def _table_text(row: int, col: int, table=None):
        if table is None:
            return ""
        item = table.item(row, col)
        return "" if item is None else item.text().strip()

    def _table_float(self, row: int, col: int, default: float) -> float:
        text = self._table_text(row, col, self.widget.tableWidget_Peaks)
        return default if text == "" else float(text)

    def _table_optional_float(self, row: int, col: int, default=None):
        text = self._table_text(row, col, self.widget.tableWidget_Peaks)
        return default if text == "" else float(text)

    def _table_checkbox(self, row: int, col: int) -> bool:
        widget = self.widget.tableWidget_Peaks.cellWidget(row, col)
        if widget is None:
            return True
        checkbox = widget.findChild(QtWidgets.QCheckBox)
        return True if checkbox is None else checkbox.isChecked()

    def _set_vary_checkbox(self, row: int, col: int, checked: bool):
        checkbox = QtWidgets.QCheckBox()
        checkbox.setChecked(checked)
        checkbox.stateChanged.connect(lambda _state: self.handle_peak_table_change(None))
        container = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(checkbox)
        self.widget.tableWidget_Peaks.setCellWidget(row, col, container)

    @staticmethod
    def _constraint_minimum(param: ParameterConstraint) -> float:
        return 0.0 if param.min is None else float(param.min)

    @staticmethod
    def _default_bounds(value: float, *, lower_floor: float = 0.0, upper_ceiling: float | None = None) -> tuple[float, float]:
        value = float(value)
        delta = abs(value) * 0.2
        if delta <= 0.0:
            delta = 0.1
        lower = max(lower_floor, value - delta)
        upper = value + delta
        if upper_ceiling is not None:
            upper = min(upper, upper_ceiling)
        if upper < lower:
            upper = lower
        return lower, upper

    def _normalized_peak_constraints(self, peak: PeakSpec) -> PeakSpec:
        peak.amplitude.min = self._constraint_minimum(peak.amplitude)
        peak.center.min = self._constraint_minimum(peak.center)
        peak.sigma.min = self._constraint_minimum(peak.sigma)
        peak.fraction.min = self._constraint_minimum(peak.fraction)
        if peak.fraction.max is not None:
            peak.fraction.max = min(float(peak.fraction.max), 1.0)
        peak.amplitude.value = max(float(peak.amplitude.value), 0.0)
        peak.center.value = max(float(peak.center.value), 0.0)
        peak.sigma.value = max(float(peak.sigma.value), 1e-5)
        peak.fraction.value = min(max(float(peak.fraction.value), 0.0), 1.0)
        peak.guess_min_cminv = max(float(peak.guess_min_cminv), 0.0)
        peak.guess_max_cminv = max(float(peak.guess_max_cminv), peak.guess_min_cminv)
        return peak

    def save_fit_results(self):
        if self.spectrum is None or not self.saved_sections:
            QtWidgets.QMessageBox.information(
                self.widget,
                "No saved sections",
                "Save one or more fitted sections first.",
            )
            return
        base_dir = get_param_dir(self.spectrum.path)
        default_base = str(Path(base_dir) / f"{Path(self.spectrum.path).stem}_sections")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.widget,
            "Save section results",
            default_base,
            "Excel (*.xlsx);;JSON (*.json);;All files (*)",
        )
        if not path:
            return
        output_base = str(Path(path).with_suffix(""))
        try:
            result = export_saved_sections(
                output_base,
                self.spectrum,
                self.saved_sections,
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self.widget, "Save failed", str(exc))
            self.log(f"Save fit results failed: {exc}")
            return
        self.log(f"Saved {len(self.saved_sections)} section(s) to {result.json_path} and {result.excel_path}")

    def export_plot_npy(self):
        if self.spectrum is None or self.background_result is None or self.fit_result is None:
            QtWidgets.QMessageBox.information(
                self.widget,
                "No fit result",
                "No PeakFit result is currently in queue.\n\n"
                "Go to the Sections menu, choose a saved range, and press Set current.\n"
                "Or run a new background fit and PeakFit before exporting.",
            )
            return
        base_dir = get_param_dir(self.spectrum.path)
        target_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self.widget,
            "Select folder for NPY export",
            base_dir,
        )
        if not target_dir:
            return
        base_export_dir = Path(target_dir) / f"{Path(self.spectrum.path).stem}-npy"
        export_dir = base_export_dir
        suffix = 1
        while export_dir.exists():
            export_dir = Path(f"{base_export_dir}-{suffix}")
            suffix += 1
        output_base = str(export_dir / f"{Path(self.spectrum.path).stem}_plot_export")
        try:
            result = export_plot_npy(
                output_base,
                self.spectrum,
                self.current_region,
                self.background_result,
                self.fit_result,
                view_limits=self._current_plot_limits(),
            )
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self.widget, "Export failed", str(exc))
            self.log(f"Export NPY failed: {exc}")
            return
        self.log(
            f"Exported {result.npy_path}, {result.script_path}, {result.png_path}, and {result.pdf_path}"
        )

    def save_current_session(self):
        if self.spectrum is None:
            QtWidgets.QMessageBox.information(self.widget, "No spectrum", "Load a spectrum first.")
            return
        reason, ok = QtWidgets.QInputDialog.getText(
            self.widget,
            "Save vibfit session",
            "Backup comment:",
            text="manual save",
        )
        if not ok:
            return
        result = save_session(
            self.spectrum.path,
            self.spectrum,
            self.current_region,
            self.background_result,
            self.fit_result,
            self.saved_sections,
            reason=(reason or "manual save"),
        )
        self.refresh_backup_table()
        self.log(f"Saved session backup {result.backup_id} to {result.param_dir}")

    def save_to_section(self):
        if self.spectrum is None or self.background_result is None or self.fit_result is None:
            QtWidgets.QMessageBox.information(
                self.widget,
                "No fit result",
                "Run background fitting and PeakFit before saving a section.",
            )
            return
        preserve_limits = self._current_plot_limits()
        section_region = copy.deepcopy(self.current_region)
        section_x_min, section_x_max = self._section_peakfit_bounds(
            SavedSection(
                timestamp="",
                label=section_region.name,
                region=section_region,
                background_result=self.background_result,
                fit_result=self.fit_result,
            )
        )
        section_region.x_min_cminv = section_x_min
        section_region.x_max_cminv = section_x_max
        self.saved_sections.append(
            SavedSection(
                timestamp=dt.datetime.now().isoformat(timespec="seconds"),
                label=self.DEFAULT_SECTION_COMMENT,
                region=section_region,
                background_result=copy.deepcopy(self.background_result),
                fit_result=copy.deepcopy(self.fit_result),
            )
        )
        self.current_region.peaks = []
        self._clear_peak_fit_results()
        self._populate_peak_table()
        self._populate_sections_table()
        self._draw(preserve_limits=preserve_limits)
        self.log("Saved current fit to Sections and cleared the active fit result.")

    def _selected_section_rows(self) -> list[int]:
        selection = self.widget.tableWidget_Sections.selectionModel()
        if selection is None:
            return []
        rows = sorted({index.row() for index in selection.selectedRows()})
        return [row for row in rows if 0 <= row < len(self.saved_sections)]

    def set_selected_section_current(self):
        rows = self._selected_section_rows()
        if len(rows) != 1:
            QtWidgets.QMessageBox.information(self.widget, "Select one section", "Select one section row first.")
            return
        section = copy.deepcopy(self.saved_sections[rows[0]])
        self.current_region = section.region
        self.background_result = section.background_result
        self.fit_result = section.fit_result
        self._populate_background_area_table()
        self._populate_peak_table()
        self._populate_sections_table()
        self.widget.plainTextEdit_BackgroundReport.setPlainText(
            "" if self.background_result is None else self.background_result.fit_report
        )
        self._update_results_table()
        self.widget.plainTextEdit_FitReport.setPlainText(
            "" if self.fit_result is None else self.fit_result.fit_report
        )
        self._draw()
        x_min, x_max = self._section_peakfit_bounds(section)
        self.log(
            f"Loaded section {section.label} "
            f"({x_min:.2f}-{x_max:.2f} cm$^{{-1}}$) into the current queue."
        )

    def remove_selected_sections(self):
        rows = self._selected_section_rows()
        if not rows:
            return
        for row in reversed(rows):
            self.saved_sections.pop(row)
        self._populate_sections_table()
        self.log(f"Removed {len(rows)} section(s) from the saved list.")

    def clear_section_list(self):
        if not self.saved_sections:
            return
        self.saved_sections = []
        self._populate_sections_table()
        self.log("Cleared the saved section list.")

    def refresh_backup_table(self):
        table = self.widget.tableWidget_Backups
        table.setRowCount(0)
        if self.spectrum is None:
            return
        events = list(reversed(list_backup_events(get_param_dir(self.spectrum.path))))
        table.setRowCount(len(events))
        for row, event in enumerate(events):
            for col, key in enumerate(("id", "timestamp", "reason")):
                item = QtWidgets.QTableWidgetItem(str(event.get(key, "")))
                item.setFlags(QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled)
                table.setItem(row, col, item)

    def autoload_latest_session(self):
        if self.spectrum is None:
            return
        payload = load_session_from_backup(get_param_dir(self.spectrum.path))
        if not payload:
            return
        self._apply_session_payload(payload)
        self.log("Autoloaded latest vibfit session.")

    def restore_selected_backup(self):
        if self.spectrum is None:
            return
        row = self.widget.tableWidget_Backups.currentRow()
        if row < 0:
            QtWidgets.QMessageBox.information(self.widget, "No backup selected", "Select a backup row first.")
            return
        backup_id_item = self.widget.tableWidget_Backups.item(row, 0)
        if backup_id_item is None:
            return
        payload = load_session_from_backup(get_param_dir(self.spectrum.path), backup_id_item.text())
        if not payload:
            QtWidgets.QMessageBox.warning(self.widget, "Restore failed", "Backup payload could not be read.")
            return
        self._apply_session_payload(payload)
        self.log(f"Restored backup {backup_id_item.text()}")

    def edit_selected_backup_comment(self):
        if self.spectrum is None:
            return
        row = self.widget.tableWidget_Backups.currentRow()
        if row < 0:
            QtWidgets.QMessageBox.information(self.widget, "No backup selected", "Select a backup row first.")
            return
        backup_id_item = self.widget.tableWidget_Backups.item(row, 0)
        comment_item = self.widget.tableWidget_Backups.item(row, 2)
        if backup_id_item is None:
            return
        backup_id = backup_id_item.text()
        current_comment = "" if comment_item is None else comment_item.text()
        comment, ok = QtWidgets.QInputDialog.getText(
            self.widget,
            "Edit backup comment",
            "Comment:",
            text=current_comment,
        )
        if not ok:
            return
        param_dir = get_param_dir(self.spectrum.path)
        if not update_backup_comment(param_dir, backup_id, comment):
            QtWidgets.QMessageBox.warning(self.widget, "Edit failed", "Backup comment could not be updated.")
            return
        self.refresh_backup_table()
        self.log(f"Updated comment for backup {backup_id}")

    def _apply_session_payload(self, payload):
        region_payload = payload.get("region") or {}
        if region_payload:
            self.current_region = self._region_from_payload(region_payload)
        else:
            self.current_region = clone_region(default_region())
        self.background_result = self._background_result_from_payload(payload.get("background_result"))
        self.fit_result = self._fit_result_from_payload(payload.get("fit_result"))
        self.saved_sections = []
        for section_payload in payload.get("saved_sections", []) or []:
            section = self._saved_section_from_payload(section_payload)
            if section is not None:
                self.saved_sections.append(section)
        self._populate_background_area_table()
        self._populate_peak_table()
        self._populate_sections_table()
        self.widget.plainTextEdit_BackgroundReport.setPlainText(
            "" if self.background_result is None else self.background_result.fit_report
        )
        self._update_results_table()
        self.widget.plainTextEdit_FitReport.setPlainText(
            "" if self.fit_result is None else self.fit_result.fit_report
        )
        self._draw()

    def _region_from_payload(self, region_payload):
        region = clone_region(default_region())
        region.name = region_payload.get("name", region.name)
        region.x_min_cminv = float(region_payload.get("x_min_cminv", region.x_min_cminv))
        region.x_max_cminv = float(region_payload.get("x_max_cminv", region.x_max_cminv))
        bg = region_payload.get("background") or {}
        area_payloads = bg.get("fit_areas") or []
        if not area_payloads and bg.get("anchor_left_cminv") is not None and bg.get("anchor_right_cminv") is not None:
            area_payloads = [
                {"x_min_cminv": region.x_min_cminv, "x_max_cminv": float(bg["anchor_left_cminv"])},
                {"x_min_cminv": float(bg["anchor_right_cminv"]), "x_max_cminv": region.x_max_cminv},
            ]
        region.background = BackgroundSpec(
            model_name=bg.get("model_name", "PowerLaw"),
            anchor_left_cminv=bg.get("anchor_left_cminv"),
            anchor_right_cminv=bg.get("anchor_right_cminv"),
            fit_areas=[
                BackgroundArea(
                    x_min_cminv=float(area.get("x_min_cminv", region.x_min_cminv)),
                    x_max_cminv=float(area.get("x_max_cminv", region.x_max_cminv)),
                )
                for area in area_payloads
            ],
        )
        region.peaks = []
        for peak in region_payload.get("peaks", []):
            region.peaks.append(
                self._normalized_peak_constraints(
                    PeakSpec(
                        name=peak.get("name", f"p{len(region.peaks) + 1}"),
                        guess_min_cminv=float(peak.get("guess_min_cminv", region.x_min_cminv)),
                        guess_max_cminv=float(peak.get("guess_max_cminv", region.x_max_cminv)),
                        amplitude=ParameterConstraint(**(peak.get("amplitude") or {"value": 100.0})),
                        center=ParameterConstraint(
                            **(peak.get("center") or {"value": (region.x_min_cminv + region.x_max_cminv) / 2.0})
                        ),
                        sigma=ParameterConstraint(**(peak.get("sigma") or {"value": 20.0})),
                        fraction=ParameterConstraint(**(peak.get("fraction") or {"value": 0.5})),
                    )
                )
            )
        return region

    def _background_result_from_payload(self, payload):
        if not payload:
            return None
        return BackgroundFitResult(
            x_cminv=np.asarray(payload.get("x_cminv", []), dtype=float),
            y_raw=np.asarray(payload.get("y_raw", []), dtype=float),
            background=np.asarray(payload.get("background", []), dtype=float),
            y_bgsub=np.asarray(payload.get("y_bgsub", []), dtype=float),
            bgsub_offset=float(payload.get("bgsub_offset", 0.0)),
            area_mask=np.asarray(payload.get("area_mask", []), dtype=bool),
            fit_report=payload.get("fit_report", ""),
            success=bool(payload.get("success", False)),
            chisqr=float(payload.get("chisqr", 0.0)),
            redchi=float(payload.get("redchi", 0.0)),
            aic=float(payload.get("aic", 0.0)),
            bic=float(payload.get("bic", 0.0)),
        )

    def _fit_result_from_payload(self, payload):
        if not payload:
            return None
        peaks = [
            PeakResult(
                name=peak.get("name", ""),
                center_ev=float(peak.get("center_ev", 0.0)),
                center_cminv=float(peak.get("center_cminv", 0.0)),
                amplitude=float(peak.get("amplitude", 0.0)),
                sigma_ev=float(peak.get("sigma_ev", 0.0)),
                sigma_cminv=float(peak.get("sigma_cminv", 0.0)),
                fraction=float(peak.get("fraction", 0.0)),
                curve=np.asarray(peak.get("curve", []), dtype=float),
            )
            for peak in payload.get("peaks", [])
        ]
        return FitResultBundle(
            region_name=payload.get("region_name", self.current_region.name),
            x_cminv=np.asarray(payload.get("x_cminv", []), dtype=float),
            y_raw=np.asarray(payload.get("y_raw", []), dtype=float),
            best_fit=np.asarray(payload.get("best_fit", []), dtype=float),
            background=np.asarray(payload.get("background", []), dtype=float),
            y_bgsub=np.asarray(payload.get("y_bgsub", []), dtype=float),
            bgsub_offset=float(payload.get("bgsub_offset", 0.0)),
            best_fit_bgsub=np.asarray(payload.get("best_fit_bgsub", []), dtype=float),
            residual_raw=np.asarray(payload.get("residual_raw", []), dtype=float),
            residual_bgsub=np.asarray(payload.get("residual_bgsub", []), dtype=float),
            peaks=peaks,
            fit_report=payload.get("fit_report", ""),
            success=bool(payload.get("success", False)),
            chisqr=float(payload.get("chisqr", 0.0)),
            redchi=float(payload.get("redchi", 0.0)),
            aic=float(payload.get("aic", 0.0)),
            bic=float(payload.get("bic", 0.0)),
        )

    def _saved_section_from_payload(self, payload):
        region_payload = payload.get("region")
        if not region_payload:
            return None
        return SavedSection(
            timestamp=payload.get("timestamp", ""),
            label=payload.get("label", self.DEFAULT_SECTION_COMMENT),
            region=self._region_from_payload(region_payload),
            background_result=self._background_result_from_payload(payload.get("background_result")),
            fit_result=self._fit_result_from_payload(payload.get("fit_result")),
        )
