import os
import sys
import tempfile
import json
import datetime
from contextlib import contextmanager
from qtpy import QtWidgets, QtCore
from .mplcontroller import MplController
from .waterfalltablecontroller import WaterfallTableController
from .jcpdstablecontroller import JcpdsTableController
from .peakfittablecontroller import PeakfitTableController
from .ccdprocesscontroller import CCDProcessController
from ..utils import convert_wl_to_energy, get_temp_dir, make_filename
from ..model.param_session_io import (
    save_model_to_param,
    load_model_from_param,
    list_backup_events,
    is_new_param_folder,
    BACKUP_INDEX_FILE,
)

class SessionController(object):

    def __init__(self, model, widget):
        self.model = model
        self.widget = widget
        self._carryover_source_chi = None
        self._last_param_category_presence = {
            "backup_information": False,
            "jcpds": False,
            "pressure": False,
            "temperature": False,
            "spectrum_smoothing": False,
            "cake_z_scale": False,
            "ccd_roi": False,
            "background": False,
            "waterfall_list": False,
            "fits_information": False,
        }
        self.plot_ctrl = MplController(self.model, self.widget)
        self.waterfalltable_ctrl = \
            WaterfallTableController(self.model, self.widget)
        self.jcpdstable_ctrl = JcpdsTableController(self.model, self.widget)
        self.peakfit_table_ctrl = PeakfitTableController(
            self.model, self.widget)
        self.ccdprocess_ctrl = CCDProcessController(self.model, self.widget)
        self.connect_channel()

    def connect_channel(self):
        if hasattr(self.widget, "pushButton_SaveDPP"):
            self.widget.pushButton_SaveDPP.clicked.connect(self.save_dpp)
        if hasattr(self.widget, "pushButton_LoadDPP"):
            self.widget.pushButton_LoadDPP.clicked.connect(self.save_dpp)
        if hasattr(self.widget, "pushButton_OpenBackupInfo"):
            self.widget.pushButton_OpenBackupInfo.clicked.connect(
                self.open_backup_info)
        if hasattr(self.widget, "pushButton_SaveJlist"):
            self.widget.pushButton_SaveJlist.clicked.connect(self.save_dpp)
        if hasattr(self.widget, "pushButton_S_SaveSession"):
            self.widget.pushButton_S_SaveSession.clicked.connect(self.save_dpp)
        if hasattr(self.widget, "pushButton_BackupRestore"):
            self.widget.pushButton_BackupRestore.clicked.connect(
                self.restore_selected_backup)
        if hasattr(self.widget, "pushButton_BackupEditComment"):
            self.widget.pushButton_BackupEditComment.clicked.connect(
                self.edit_selected_backup_comment)
        if hasattr(self.widget, "tabWidget_3"):
            self.widget.tabWidget_3.currentChanged.connect(
                self._handle_file_subtab_changed)

    def _commit_inputs_before_save(self):
        # Commit any active spinbox editor text first (e.g., JCPDS twk cells).
        fw = QtWidgets.QApplication.focusWidget()
        if fw is not None:
            try:
                fw.clearFocus()
            except Exception:
                pass
        if isinstance(fw, QtWidgets.QAbstractSpinBox):
            try:
                fw.interpretText()
            except Exception:
                pass
        QtWidgets.QApplication.processEvents()
        # Sync all JCPDS table widgets to model.
        self.jcpdstable_ctrl.sync_model_from_table()
        # Commit P/T editor text and mirror to model scalars.
        self.widget.doubleSpinBox_Pressure.interpretText()
        self.widget.doubleSpinBox_Temperature.interpretText()
        self.model.save_pressure(self.widget.doubleSpinBox_Pressure.value())
        self.model.save_temperature(self.widget.doubleSpinBox_Temperature.value())

    def _handle_file_subtab_changed(self, idx):
        # Backup table is shown in File > Data; keep it current on tab changes.
        self.refresh_backup_table()

    def refresh_backup_table(self):
        if not hasattr(self.widget, "tableWidget_BackupInfo"):
            return
        table = self.widget.tableWidget_BackupInfo
        headers = ["ID", "Comment", "Changes", "Timestamp", "Files"]
        table.clear()
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(0)
        table.verticalHeader().setVisible(False)
        if not self.model.base_ptn_exist():
            return
        param_dir = get_temp_dir(self.model.get_base_ptn_filename())
        if not is_new_param_folder(param_dir):
            return
        events = list_backup_events(param_dir)
        table.setRowCount(len(events))
        for row, (idx, ev) in enumerate(reversed(list(enumerate(events)))):
            reason = self._format_backup_comment(ev.get("reason", ""))
            values = [
                str(ev.get("id", "")),
                reason,
                ", ".join(ev.get("highlights", [])) or "none",
                str(ev.get("timestamp", "")),
                str(len(ev.get("changed_files", []))),
            ]
            for col, txt in enumerate(values):
                item = QtWidgets.QTableWidgetItem(txt)
                item.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)
                if col == 0:
                    item.setData(QtCore.Qt.UserRole, idx)
                table.setItem(row, col, item)
        table.resizeColumnsToContents()
        table.resizeRowsToContents()
        if table.rowCount() > 0:
            table.selectRow(0)

    def _format_backup_comment(self, reason):
        comment = str(reason or "").strip()
        if comment in ("manual-save", "save", ""):
            comment = "snapshot"
        return comment

    def _selected_backup_index_from_table(self):
        if not hasattr(self.widget, "tableWidget_BackupInfo"):
            return None
        table = self.widget.tableWidget_BackupInfo
        rows = table.selectionModel().selectedRows()
        if len(rows) != 1:
            return None
        item = table.item(rows[0].row(), 0)
        if item is None:
            return None
        idx = item.data(QtCore.Qt.UserRole)
        if idx is None:
            return None
        return int(idx)

    def _selected_backup_id_from_table(self):
        if not hasattr(self.widget, "tableWidget_BackupInfo"):
            return None
        table = self.widget.tableWidget_BackupInfo
        rows = table.selectionModel().selectedRows()
        if len(rows) != 1:
            return None
        item = table.item(rows[0].row(), 0)
        if item is None:
            return None
        return str(item.text() or "")

    def _write_json_atomic(self, path, payload):
        dname = os.path.dirname(path)
        os.makedirs(dname, exist_ok=True)
        with tempfile.NamedTemporaryFile(
                "w", delete=False, dir=dname, encoding="utf-8") as tmpf:
            json.dump(payload, tmpf, indent=2)
            tmp_path = tmpf.name
        os.replace(tmp_path, path)

    def edit_selected_backup_comment(self):
        backup_idx = self._selected_backup_index_from_table()
        backup_id = self._selected_backup_id_from_table()
        if backup_idx is None:
            QtWidgets.QMessageBox.warning(
                self.widget, "Warning",
                "Select one backup row first.")
            return
        if not self.model.base_ptn_exist():
            QtWidgets.QMessageBox.warning(
                self.widget, "Warning",
                "Open a spectrum first.")
            return
        param_dir = get_temp_dir(self.model.get_base_ptn_filename())
        index_path = os.path.join(param_dir, BACKUP_INDEX_FILE)
        if not os.path.exists(index_path):
            QtWidgets.QMessageBox.warning(
                self.widget, "Warning",
                "Backup index file not found.")
            return
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index_data = json.load(f)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self.widget, "Warning",
                "Cannot read backup index:\n" + str(exc))
            return
        events = index_data.get("events", [])
        if (backup_idx < 0) or (backup_idx >= len(events)):
            QtWidgets.QMessageBox.warning(
                self.widget, "Warning",
                "Selected backup row is no longer valid.")
            return
        ev = events[backup_idx]
        current_comment = self._format_backup_comment(ev.get("reason", ""))
        new_comment, ok = QtWidgets.QInputDialog.getText(
            self.widget,
            "Edit Backup Comment",
            "Comment:",
            text=current_comment,
        )
        if not ok:
            return
        updated_comment = str(new_comment or "").strip()
        if updated_comment == "":
            updated_comment = "snapshot"
        events[backup_idx]["reason"] = updated_comment
        index_data["events"] = events
        try:
            self._write_json_atomic(index_path, index_data)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self.widget, "Warning",
                "Cannot update backup comment:\n" + str(exc))
            return
        self.refresh_backup_table()
        # Restore selection to edited backup id when possible.
        if hasattr(self.widget, "tableWidget_BackupInfo") and (backup_id not in (None, "")):
            table = self.widget.tableWidget_BackupInfo
            for r in range(table.rowCount()):
                it = table.item(r, 0)
                if (it is not None) and (str(it.text()) == backup_id):
                    table.selectRow(r)
                    break

    def restore_selected_backup(self):
        # Do not refresh here: it resets selection to row 0 and can restore
        # a different backup than the user highlighted.
        if hasattr(self.widget, "tableWidget_BackupInfo") and \
                self.widget.tableWidget_BackupInfo.rowCount() == 0:
            self.refresh_backup_table()
        backup_idx = self._selected_backup_index_from_table()
        if backup_idx is None:
            QtWidgets.QMessageBox.warning(
                self.widget, "Warning",
                "Select one backup row first.")
            return
        self._restore_backup_by_index(backup_idx)

    def _restore_backup_by_index(self, backup_idx):
        if not self.model.base_ptn_exist():
            QtWidgets.QMessageBox.warning(
                self.widget, "Warning",
                "Open a spectrum first.")
            return
        param_dir = get_temp_dir(self.model.get_base_ptn_filename())
        if not is_new_param_folder(param_dir):
            QtWidgets.QMessageBox.information(
                self.widget, "Backup Info",
                "No saved session folder was found for this spectrum.")
            return
        events = list_backup_events(param_dir)
        if (backup_idx < 0) or (backup_idx >= len(events)):
            QtWidgets.QMessageBox.warning(
                self.widget, "Warning",
                "Selected backup index is no longer valid.")
            return
        backup_id = events[backup_idx].get("id")
        self._commit_inputs_before_save()
        base_chi = self.model.get_base_ptn_filename()
        success, meta = load_model_from_param(
            self.model, base_chi, backup_event_index=backup_idx)
        if not success:
            QtWidgets.QMessageBox.warning(
                self.widget, "Warning",
                "Backup restore failed: " + str(meta.get("reason")))
            return
        self._sync_ui_from_model(
            manifest_path=str(meta.get("manifest", "")),
            ui_state=meta.get("ui_state", {}),
        )
        missing_csv = meta.get("missing_section_csv_files", []) or []
        if missing_csv:
            QtWidgets.QMessageBox.warning(
                self.widget, "Missing Section CSV",
                "Some saved section CSV files were missing, so those sections "
                "were skipped.\n\n"
                "Missing files:\n" + "\n".join(map(str, missing_csv[:20])) +
                ("\n..." if len(missing_csv) > 20 else "")
            )
        self.refresh_backup_table()
        msg = f"Restored backup: {backup_id}"
        QtWidgets.QMessageBox.information(
            self.widget, "Backup Restored", msg)

    def _collect_ui_state(self):
        smoothing = {
            "active": bool(self.plot_ctrl._smoothing_active()),
            "despike_kernel": int(self.widget.spinBox_SpectrumDespike.value())
            if hasattr(self.widget, "spinBox_SpectrumDespike") else 0,
            "sg_window": int(self.widget.spinBox_SpectrumSGWindow.value())
            if hasattr(self.widget, "spinBox_SpectrumSGWindow") else 0,
            "sg_polyorder": int(self.widget.spinBox_SpectrumSGPoly.value())
            if hasattr(self.widget, "spinBox_SpectrumSGPoly") else 3,
            "raw_file": (
                os.path.splitext(os.path.basename(self.model.base_ptn.fname))[0] + ".chi"
            ) if self.model.base_ptn_exist() else None,
            "smooth_file": "smooth.chi",
        }
        cake_hist = {}
        if hasattr(self.widget, "cake_hist_widget"):
            hist = self.widget.cake_hist_widget
            cake_hist = {
                "log_y": bool(hist.check_log.isChecked()),
            }
        return {
            "pt_controls": {
                "p_step": self.widget.doubleSpinBox_PStep.value(),
                "t_step": self.widget.spinBox_TStep.value(),
                "jcpds_step": self.widget.doubleSpinBox_JCPDSStep.value(),
            },
            "background": {
                "roi_min": float(self.widget.doubleSpinBox_Background_ROI_min.value()),
                "roi_max": float(self.widget.doubleSpinBox_Background_ROI_max.value()),
                "poly_order": int(self.widget.spinBox_BGParam1.value()),
                "areas": self._collect_background_areas(),
            },
            "spectrum": smoothing,
            "cake": {
                "vmin": float(self.widget.doubleSpinBox_CCDScaleMin.value())
                if hasattr(self.widget, "doubleSpinBox_CCDScaleMin") else 0.0,
                "vmax": float(self.widget.doubleSpinBox_CCDScaleMax.value())
                if hasattr(self.widget, "doubleSpinBox_CCDScaleMax") else 1.0,
                "mask_min": self.widget.spinBox_MaskMin.value(),
                "mask_max": self.widget.spinBox_MaskMax.value(),
                "hist": cake_hist,
            },
            "ccd_roi": {
                "row_min": int(self.widget.spinBox_CCDRowMin.value())
                if hasattr(self.widget, "spinBox_CCDRowMin") else 0,
                "row_max": int(self.widget.spinBox_CCDRowMax.value())
                if hasattr(self.widget, "spinBox_CCDRowMax") else 0,
            },
            "diff": self._collect_diff_ui_state(),
        }

    def _collect_diff_ui_state(self):
        if (not hasattr(self.widget, "checkBox_Diff")) and \
                (not hasattr(self.widget, "checkBox_UseDiffMode")):
            return {}
        enabled = False
        diff = {
            "ref_chi_path": str(self.widget.lineEdit_DiffRefChi.text()).strip(),
            "cmap_2d": str(self.widget.comboBox_DiffCmap.currentText()),
            "scale_mode": str(self.widget.comboBox_DiffScaleMode.currentText()),
            "vmin": float(self.widget.doubleSpinBox_DiffVmin.value()),
            "vmax": float(self.widget.doubleSpinBox_DiffVmax.value()),
        }
        if hasattr(self.model, "diff_state") and (self.model.diff_state is not None):
            self.model.diff_state.apply_ui_dict(diff)
        return diff

    def _apply_ui_state(self, ui_state):
        pt = (ui_state or {}).get("pt_controls", {})
        if pt != {}:
            if "p_step" in pt:
                self.widget.doubleSpinBox_PStep.setValue(float(pt["p_step"]))
            if "t_step" in pt:
                self.widget.spinBox_TStep.setValue(int(pt["t_step"]))
            if "jcpds_step" in pt:
                self.widget.doubleSpinBox_JCPDSStep.setValue(float(pt["jcpds_step"]))
        bg = (ui_state or {}).get("background", {})
        if bg != {}:
            if "roi_min" in bg:
                self.widget.doubleSpinBox_Background_ROI_min.setValue(float(bg["roi_min"]))
            if "roi_max" in bg:
                self.widget.doubleSpinBox_Background_ROI_max.setValue(float(bg["roi_max"]))
            self.widget.spinBox_BGParam1.setValue(int(bg.get("poly_order", 3)))
            self._apply_background_areas(bg.get("areas", []))
        spectrum = (ui_state or {}).get("spectrum", {})
        if spectrum != {}:
            if hasattr(self.widget, "spinBox_SpectrumDespike"):
                self.widget.spinBox_SpectrumDespike.setValue(
                    int(spectrum.get("despike_kernel", 0)))
            if hasattr(self.widget, "spinBox_SpectrumSGWindow"):
                self.widget.spinBox_SpectrumSGWindow.setValue(
                    int(spectrum.get("sg_window", 0)))
            if hasattr(self.widget, "spinBox_SpectrumSGPoly"):
                self.widget.spinBox_SpectrumSGPoly.setValue(
                    int(spectrum.get("sg_polyorder", 3)))
        cake = (ui_state or {}).get("cake", {})
        if cake != {}:
            if "vmin" in cake and hasattr(self.widget, "doubleSpinBox_CCDScaleMin"):
                self.widget.doubleSpinBox_CCDScaleMin.setValue(float(cake["vmin"]))
            if "vmax" in cake and hasattr(self.widget, "doubleSpinBox_CCDScaleMax"):
                self.widget.doubleSpinBox_CCDScaleMax.setValue(float(cake["vmax"]))
            if "mask_min" in cake:
                self.widget.spinBox_MaskMin.setValue(int(cake["mask_min"]))
            if "mask_max" in cake:
                self.widget.spinBox_MaskMax.setValue(int(cake["mask_max"]))
            hist = cake.get("hist", {})
            if hasattr(self.widget, "cake_hist_widget") and hist != {}:
                if "log_y" in hist:
                    self.widget.cake_hist_widget.check_log.setChecked(bool(hist["log_y"]))
        ccd_roi = (ui_state or {}).get("ccd_roi", {})
        if ccd_roi != {} and hasattr(self.widget, "spinBox_CCDRowMin") and \
                hasattr(self.widget, "spinBox_CCDRowMax"):
            row_min = int(ccd_roi.get("row_min", self.widget.spinBox_CCDRowMin.value()))
            row_max = int(ccd_roi.get("row_max", self.widget.spinBox_CCDRowMax.value()))
            self.widget.spinBox_CCDRowMin.blockSignals(True)
            self.widget.spinBox_CCDRowMax.blockSignals(True)
            self.widget.spinBox_CCDRowMin.setValue(row_min)
            self.widget.spinBox_CCDRowMax.setValue(row_max)
            self.widget.spinBox_CCDRowMin.blockSignals(False)
            self.widget.spinBox_CCDRowMax.blockSignals(False)
            try:
                self.model.base_ptn.set_spe_row_roi(row_min, row_max)
            except Exception:
                pass
        self._apply_diff_ui_state((ui_state or {}).get("diff", {}))

    def _restore_background_ui_from_model(self):
        if not self.model.base_ptn_exist():
            return
        base_ptn = self.model.base_ptn
        roi = getattr(base_ptn, "roi", None)
        if roi is not None and len(roi) >= 2:
            try:
                self.widget.doubleSpinBox_Background_ROI_min.setValue(float(roi[0]))
                self.widget.doubleSpinBox_Background_ROI_max.setValue(float(roi[1]))
            except Exception:
                pass
        try:
            params = getattr(base_ptn, "params_chbg", None) or [3]
            self.widget.spinBox_BGParam1.setValue(int(params[0]))
        except Exception:
            pass
        self._apply_background_areas(getattr(base_ptn, "bg_fit_areas", []) or [])

    def _apply_diff_ui_state(self, diff):
        if ((not hasattr(self.widget, "checkBox_Diff")) and
                (not hasattr(self.widget, "checkBox_UseDiffMode"))) or (diff == {}):
            return
        if "ref_chi_path" in diff:
            self.widget.lineEdit_DiffRefChi.setText(str(diff["ref_chi_path"] or ""))
        # Diff toggle status is not loaded from JSON; always start unchecked.
        if hasattr(self.widget, "checkBox_Diff"):
            self.widget.checkBox_Diff.setChecked(False)
        if hasattr(self.widget, "checkBox_UseDiffMode"):
            self.widget.checkBox_UseDiffMode.setChecked(False)
        if "cmap_2d" in diff:
            cmap = str(diff["cmap_2d"])
            if self.widget.comboBox_DiffCmap.findText(cmap) >= 0:
                self.widget.comboBox_DiffCmap.setCurrentText(cmap)
        if "scale_mode" in diff:
            mode = str(diff["scale_mode"])
            if mode in ("Symmetric (0 centered)", "Asymmetric (0 centered)", "0 centered"):
                mode = "0 Centered"
            elif mode in ("Cake-like free range", "Positive only (0 as min)", "Negative only (0 as max)"):
                mode = "Free range"
            if self.widget.comboBox_DiffScaleMode.findText(mode) >= 0:
                self.widget.comboBox_DiffScaleMode.setCurrentText(mode)
        if "vmin" in diff:
            self.widget.doubleSpinBox_DiffVmin.setValue(float(diff["vmin"]))
        if "vmax" in diff:
            self.widget.doubleSpinBox_DiffVmax.setValue(float(diff["vmax"]))
        if hasattr(self.model, "diff_state") and (self.model.diff_state is not None):
            self.model.diff_state.apply_ui_dict(diff)

    def _sync_ui_from_model(self, manifest_path="", ui_state=None):
        """
        Repopulate GUI state from current model after session load/restore.
        """
        with self._block_plot_ui_signals():
            if self.model.base_ptn_exist():
                self.widget.lineEdit_DiffractionPatternFileName.setText(
                    str(self.model.base_ptn.fname))
                self.widget.doubleSpinBox_SetWavelength.setValue(
                    self.model.get_base_ptn_wavelength())
                self.widget.label_XRayEnergy.setText("nm")
                if self.model.exist_in_waterfall(self.model.base_ptn.fname):
                    self.widget.pushButton_AddBasePtn.setChecked(True)
                else:
                    self.widget.pushButton_AddBasePtn.setChecked(False)
            self.widget.textEdit_Jlist.setText(str(manifest_path))
            self.widget.textEdit_SessionFileName.setText(str(manifest_path))
            if self.model.diff_img_exist():
                self.widget.textEdit_DiffractionImageFilename.setText(
                    self.model.diff_img.img_filename)
            else:
                self.widget.textEdit_DiffractionImageFilename.setText(
                    'Image file must have the same name ' +
                    'as base ptn in the same folder.')
            self.widget.doubleSpinBox_Pressure.setValue(self.model.get_saved_pressure())
            self.widget.doubleSpinBox_Temperature.setValue(self.model.get_saved_temperature())
            self._apply_ui_state(ui_state or {})
            self._restore_background_ui_from_model()
            self.update_inputs()
        self._sync_peakfit_selection_to_current_section()
        self.plot_ctrl.zoom_out_graph()

    @contextmanager
    def _block_plot_ui_signals(self):
        blockers = []
        for w in self._plot_signal_widgets():
            try:
                blockers.append(QtCore.QSignalBlocker(w))
            except Exception:
                continue
        try:
            yield
        finally:
            # Keep blocker objects alive until context exit.
            del blockers

    def _plot_signal_widgets(self):
        names = [
            "doubleSpinBox_SetWavelength",
            "doubleSpinBox_Pressure",
            "doubleSpinBox_Temperature",
            "doubleSpinBox_Background_ROI_min",
            "doubleSpinBox_Background_ROI_max",
            "spinBox_BGParam1",
            "doubleSpinBox_CCDScaleMin",
            "doubleSpinBox_CCDScaleMax",
            "spinBox_MaskMin",
            "spinBox_MaskMax",
            "checkBox_Diff",
            "checkBox_UseDiffMode",
            "comboBox_DiffCmap",
            "comboBox_DiffScaleMode",
            "doubleSpinBox_DiffVmin",
            "doubleSpinBox_DiffVmax",
        ]
        widgets = []
        for name in names:
            if hasattr(self.widget, name):
                widgets.append(getattr(self.widget, name))
        if hasattr(self.widget, "cake_hist_widget"):
            hist = self.widget.cake_hist_widget
            for name in ("check_log",):
                if hasattr(hist, name):
                    widgets.append(getattr(hist, name))
        return widgets

    def _sync_peakfit_selection_to_current_section(self):
        if not self.model.current_section_exist():
            self.widget.tableWidget_PkFtSections.clearSelection()
            return
        current_ts = self.model.current_section.get_timestamp()
        if current_ts is None:
            self.widget.tableWidget_PkFtSections.clearSelection()
            return
        for i, sec in enumerate(self.model.section_lst):
            if sec.get_timestamp() == current_ts:
                self.widget.tableWidget_PkFtSections.selectRow(i)
                return

    def _infer_base_chi_from_manifest(self, manifest_file):
        candidate_dir = os.path.dirname(manifest_file)
        basename = os.path.basename(candidate_dir)
        if basename.isdigit():
            param_dir = os.path.dirname(candidate_dir)
        else:
            param_dir = candidate_dir
        basename = os.path.basename(param_dir)
        if not basename.endswith("-rampo"):
            return None
        session_file = None
        if os.path.isdir(candidate_dir):
            candidate_session = os.path.join(candidate_dir, "rampo_session.json")
            if os.path.exists(candidate_session):
                session_file = candidate_session
        if session_file is None:
            for entry in sorted(os.scandir(param_dir), key=lambda e: e.name, reverse=True):
                if not entry.is_dir():
                    continue
                candidate_session = os.path.join(entry.path, "rampo_session.json")
                if os.path.exists(candidate_session):
                    session_file = candidate_session
                    break
        if session_file is not None and os.path.exists(session_file):
            try:
                with open(session_file, "r", encoding="utf-8") as f:
                    session_data = json.load(f)
                base_payload = (session_data or {}).get("base_pattern", {})
                stored = base_payload.get("fname")
                if stored:
                    candidate = stored if os.path.isabs(stored) else os.path.normpath(
                        os.path.join(os.path.dirname(param_dir), stored))
                    if os.path.exists(candidate):
                        return candidate
            except Exception:
                pass
        base_no_ext = basename[:-6]
        for ext in (".spe", ".SPE", ".chi", ".CHI"):
            candidate = os.path.join(os.path.dirname(param_dir), base_no_ext + ext)
            if os.path.exists(candidate):
                return candidate
        return None

    def _load_new_param_session(self, selected_file):
        ext = os.path.splitext(selected_file)[1].lower()
        selected_dir = os.path.dirname(selected_file)
        if ext == ".chi" and os.path.basename(selected_dir).endswith("-rampo"):
            base_chi = self._infer_base_chi_from_manifest(selected_file)
            param_dir = selected_dir
            if base_chi is None:
                QtWidgets.QMessageBox.warning(
                    self.widget, "Warning",
                    "Cannot infer original spectrum from the selected Rampo folder file.")
                return False
        elif ext == ".chi":
            base_chi = selected_file
            param_dir = os.path.join(
                os.path.dirname(base_chi),
                os.path.splitext(os.path.basename(base_chi))[0] + "-rampo",
            )
        else:
            base_chi = self._infer_base_chi_from_manifest(selected_file)
            param_dir = os.path.dirname(selected_file)
            if base_chi is None:
                QtWidgets.QMessageBox.warning(
                    self.widget, "Warning",
                    "Cannot infer base .chi file from selected manifest.")
                return False
        if not is_new_param_folder(param_dir):
            QtWidgets.QMessageBox.warning(
                self.widget, "Warning",
                "No valid session manifest was found in the session folder.")
            return False

        backup_events = list_backup_events(param_dir)
        backup_idx = None
        if backup_events:
            items = ["Current (latest)"]
            labels_to_idx = {}
            for idx, event in reversed(list(enumerate(backup_events))):
                highlights = ", ".join(event.get("highlights", []))
                if highlights == "":
                    highlights = "none"
                label = (
                    f"{event.get('id')} | "
                    f"{event.get('timestamp', '')} | "
                    f"{event.get('reason', 'save')} | "
                    f"{highlights}"
                )
                items.append(label)
                labels_to_idx[label] = idx
            selected, ok = QtWidgets.QInputDialog.getItem(
                self.widget,
                "Load Session Version",
                "Choose setup timestamp:",
                items, 0, False)
            if not ok:
                return False
            if selected != "Current (latest)":
                backup_idx = labels_to_idx.get(selected)

        success, meta = load_model_from_param(
            self.model,
            base_chi,
            backup_event_index=backup_idx,
        )
        if not success:
            QtWidgets.QMessageBox.warning(
                self.widget, "Warning",
                "Failed to load session folder: " + str(meta.get("reason")))
            return False

        self._sync_ui_from_model(
            manifest_path=str(meta.get("manifest", "")),
            ui_state=meta.get("ui_state", {}),
        )
        missing_csv = meta.get("missing_section_csv_files", []) or []
        if missing_csv:
            QtWidgets.QMessageBox.warning(
                self.widget, "Missing Section CSV",
                "Some saved section CSV files were missing, so those sections "
                "were skipped.\n\n"
                "Missing files:\n" + "\n".join(map(str, missing_csv[:20])) +
                ("\n..." if len(missing_csv) > 20 else "")
            )
        fallback_wf = meta.get("fallback_waterfall_files", []) or []
        if fallback_wf:
            QtWidgets.QMessageBox.warning(
                self.widget, "Waterfall Fallback Used",
                "Some waterfall files were missing at their original paths.\n"
                "Rampo loaded fallback copies from the session folder waterfall cache.\n\n"
                "Files:\n" + "\n".join(map(str, fallback_wf[:20])) +
                ("\n..." if len(fallback_wf) > 20 else "")
            )
        return True

    def autoload_param_for_chi(self, base_chi_file):
        """
        Automatically load saved session data for an opened spectrum file, if found.
        Returns True when a saved session is loaded, False otherwise.
        """
        param_dir = os.path.join(
            os.path.dirname(base_chi_file),
            os.path.splitext(os.path.basename(base_chi_file))[0] + "-rampo",
        )
        if not is_new_param_folder(param_dir):
            self._last_param_category_presence = {
                "backup_information": False,
                "jcpds": False,
                "pressure": False,
                "temperature": False,
                "spectrum_smoothing": False,
                "cake_z_scale": False,
                "ccd_roi": False,
                "background": False,
                "waterfall_list": False,
                "fits_information": False,
            }
            return False
        success, meta = load_model_from_param(self.model, base_chi_file)
        if not success:
            print(str(datetime.datetime.now())[:-7],
                  ": Session autoload failed:", str(meta.get("reason")))
            self._last_param_category_presence = {
                "backup_information": False,
                "jcpds": False,
                "pressure": False,
                "temperature": False,
                "spectrum_smoothing": False,
                "cake_z_scale": False,
                "ccd_roi": False,
                "background": False,
                "waterfall_list": False,
                "fits_information": False,
            }
            return False
        self._last_param_category_presence = meta.get("category_presence", {}) or {
            "backup_information": False,
            "jcpds": False,
            "pressure": False,
            "temperature": False,
            "spectrum_smoothing": False,
            "cake_z_scale": False,
            "ccd_roi": False,
            "background": False,
            "waterfall_list": False,
            "fits_information": False,
        }
        self._sync_ui_from_model(
            manifest_path=str(meta.get("manifest", "")),
            ui_state=meta.get("ui_state", {}),
        )
        missing_csv = meta.get("missing_section_csv_files", []) or []
        if missing_csv:
            QtWidgets.QMessageBox.warning(
                self.widget, "Missing Section CSV",
                "Some saved section CSV files were missing, so those sections "
                "were skipped.\n\n"
                "Missing files:\n" + "\n".join(map(str, missing_csv[:20])) +
                ("\n..." if len(missing_csv) > 20 else "")
            )
        fallback_wf = meta.get("fallback_waterfall_files", []) or []
        if fallback_wf:
            QtWidgets.QMessageBox.warning(
                self.widget, "Waterfall Fallback Used",
                "Some waterfall files were missing at their original paths.\n"
                "Rampo loaded fallback copies from the session folder waterfall cache.\n\n"
                "Files:\n" + "\n".join(map(str, fallback_wf[:20])) +
                ("\n..." if len(fallback_wf) > 20 else "")
            )
        return True

    def get_last_param_category_presence(self):
        return dict(self._last_param_category_presence)

    def set_carryover_source_chi(self, chi_filename):
        if chi_filename in (None, ""):
            self._carryover_source_chi = None
        else:
            self._carryover_source_chi = str(chi_filename)

    def get_carryover_source_chi(self):
        return self._carryover_source_chi

    def open_backup_info(self):
        self.refresh_backup_table()
        if hasattr(self.widget, "tabWidget_3Page1") and hasattr(self.widget, "tabWidget_3"):
            self.widget.tabWidget_3.setCurrentWidget(self.widget.tabWidget_3Page1)

    def _load_cake_format_file(self):
        # get filename
        temp_dir = get_temp_dir(self.model.get_base_ptn_filename())
        """
        filen = QtWidgets.QFileDialog.getOpenFileName(
            self.widget, "Open a cake format File", temp_dir,
            # self.model.chi_path,
            "Data files (*.cakeformat)")[0]
        """
        ext = "cakeformat"
        #filen_t = self.model.make_filename(ext)
        filen = make_filename(self.model.base_ptn.fname, ext,
                              temp_dir=temp_dir)
        if os.path.exists(filen):
            temp_values = {}
            with open(filen, "r") as f:
                for line in f:
                    if ':' not in line:
                        continue
                    key, value = line.split(':', 1)
                    temp_values[key.strip()] = value.strip()
            if "vmin" in temp_values and hasattr(self.widget, "doubleSpinBox_CCDScaleMin"):
                self.widget.doubleSpinBox_CCDScaleMin.setValue(float(temp_values["vmin"]))
            if "vmax" in temp_values and hasattr(self.widget, "doubleSpinBox_CCDScaleMax"):
                self.widget.doubleSpinBox_CCDScaleMax.setValue(float(temp_values["vmax"]))

    def _save_cake_format_file(self):
        # make filename
        temp_dir = get_temp_dir(self.model.get_base_ptn_filename())
        ext = "cakeformat"
        #filen_t = self.model.make_filename(ext)
        filen = make_filename(self.model.base_ptn.fname, ext,
                              temp_dir=temp_dir)
        # save cake related Values
        names = ['vmin', 'vmax']
        values = [
            self.widget.doubleSpinBox_CCDScaleMin.value()
            if hasattr(self.widget, "doubleSpinBox_CCDScaleMin") else 0.0,
            self.widget.doubleSpinBox_CCDScaleMax.value()
            if hasattr(self.widget, "doubleSpinBox_CCDScaleMax") else 1.0,
        ]

        with open(filen, "w") as f:
            for n, v in zip(names, values):
                f.write(n + ' : ' + str(v) + '\n')

    def update_inputs(self):
        self.reset_bgsub()
        self.waterfalltable_ctrl.update()
        step = float(self.widget.doubleSpinBox_JCPDSStep.value())
        self.jcpdstable_ctrl.update(step=step)
        self.peakfit_table_ctrl.update_sections()
        self.peakfit_table_ctrl.update_peak_parameters()
        self.peakfit_table_ctrl.update_peak_constraints()
        self._restore_background_ui_from_model()

    # JSON-only Rampo session scheme. Legacy session formats are disabled.
    def migrate_dpp_for_chi_if_exists(self, chi_file):
        return False

    def load_ppss(self):
        return self.load_dpp()

    def load_dpp(self):
        fn = QtWidgets.QFileDialog.getOpenFileName(
            self.widget,
            "Choose A Session File",
            self.model.chi_path,
            "Session files (*.chi *.spe *rampo_manifest.json)")[0]
        if fn == '':
            return False
        success = self._load_new_param_session(fn)
        if success:
            if self.model.exist_in_waterfall(self.model.base_ptn.fname):
                self.widget.pushButton_AddBasePtn.setChecked(True)
            else:
                self.widget.pushButton_AddBasePtn.setChecked(False)
            self._load_cake_format_file()
            self.plot_ctrl.zoom_out_graph()
            self.update_inputs()
        else:
            QtWidgets.QMessageBox.warning(
                self.widget, "Warning", "Session loading was not successful.")
        return success

    def _load_ppss(self, fsession, jlistonly=False):
        if jlistonly:
            QtWidgets.QMessageBox.information(
                self.widget, "JSON Only",
                "Legacy PPSS import is not supported in Rampo.")
            return False
        return self._load_new_param_session(fsession)

    def _load_dpp(self, filen_dpp, jlistonly=False):
        if jlistonly:
            QtWidgets.QMessageBox.information(
                self.widget, "JSON Only",
                "Legacy DPP import is not supported in Rampo.")
            return False
        return self._load_new_param_session(filen_dpp)

    def zip_ppss(self):
        QtWidgets.QMessageBox.information(
            self.widget, "JSON Only",
            "Zip/PPSS export has been removed. Rampo saves everything in the JSON session subfolder.")

    def save_dpp_ppss(self):
        return self.save_dpp()

    def save_dpp(self, quiet=False):
        if not self.model.base_ptn_exist():
            QtWidgets.QMessageBox.warning(
                self.widget, "Warning", "Open a spectrum first.")
            return
        diff_img = getattr(self.model, "diff_img", None)
        if (diff_img is not None) and (getattr(diff_img, "img", None) is not None):
            self.ccdprocess_ctrl.cook()
            self.model.diff_img.write_temp_cakefiles(
                temp_dir=get_temp_dir(self.model.get_base_ptn_filename()))
        elif (diff_img is not None) and \
                (getattr(diff_img, "tth_cake", None) is not None) and \
                (getattr(diff_img, "chi_cake", None) is not None) and \
                (getattr(diff_img, "intensity_cake", None) is not None):
            self.model.diff_img.write_temp_cakefiles(
                temp_dir=get_temp_dir(self.model.get_base_ptn_filename()))
        self._commit_inputs_before_save()
        self._write_smoothed_spectrum_file()
        try:
            result = save_model_to_param(
                self.model,
                ui_state=self._collect_ui_state(),
                reason="manual-save",
                create_backup=True,
                force_backup=True,
            )
        except Exception as inst:
            QtWidgets.QMessageBox.warning(
                self.widget, "Warning",
                "Saving session failed:\n" + str(inst))
            return
        print(str(datetime.datetime.now())[:-7],
              ": Save session:", result.manifest_path)
        self.refresh_backup_table()
        self._save_cake_format_file()
        try:
            env = os.environ['CONDA_DEFAULT_ENV']
        except Exception:
            env = 'unknown'
        temp_dir = get_temp_dir(self.model.get_base_ptn_filename())
        filen = make_filename(self.model.base_ptn.fname, "sysinfo.txt", temp_dir=temp_dir)
        with open(filen, "w") as f:
            f.write('OS: ' + os.name + '\n')
            f.write('Python ver.: ' + sys.version + '\n')
            f.write("Environment: " + env + '\n')
        self.widget.textEdit_SessionFileName.setText(str(result.manifest_path))
        self.widget.tableWidget_PkFtSections.setStyleSheet(
            "Background-color:None;color:rgb(0,0,0);")

    def _write_smoothed_spectrum_file(self):
        if not self.model.base_ptn_exist():
            return
        temp_dir = get_temp_dir(self.model.get_base_ptn_filename())
        smooth_path = os.path.join(temp_dir, "smooth.chi")
        x_raw, y_raw = self.model.base_ptn.get_raw()
        if x_raw is None or y_raw is None:
            return
        settings = self.plot_ctrl._get_smoothing_settings()
        active = bool(self.plot_ctrl._smoothing_active())
        if not active:
            if os.path.exists(smooth_path):
                try:
                    os.remove(smooth_path)
                except Exception:
                    pass
            return
        x_s, y_s = self.plot_ctrl._get_smoothed_pattern_xy(x_raw, y_raw)
        row_roi = getattr(self.model.base_ptn, "row_roi", None)
        header_lines = []
        if row_roi is not None:
            header_lines.append(
                '# CCD ROI rows: {0:d}, {1:d} \n'.format(
                    int(row_roi[0]), int(row_roi[1]))
            )
        header_lines.append(
            '# Smoothed spectrum: despike={0:d}, sg_window={1:d}, sg_polyorder={2:d} \n'.format(
                int(settings.get("despike_kernel", 0)),
                int(settings.get("sg_window", 0)),
                int(settings.get("sg_polyorder", 3)),
            )
        )
        header_lines.append('\n')
        self.model.base_ptn.write_temporary_processed_file(
            temp_dir=temp_dir,
            x_data=x_s,
            y_data=y_s,
            output_filename="smooth.chi",
            preheader=''.join(header_lines),
        )

    def save_ppss(self, quiet=False):
        return self.save_dpp(quiet=quiet)

    def save_ppss_with_default_name(self):
        return self.save_dpp()

    def reset_bgsub(self):
        '''
        this is to read from session file and put to the table
        '''
        self.widget.spinBox_BGParam1.setValue(
            self.model.base_ptn.params_chbg[0])
        self.widget.doubleSpinBox_Background_ROI_min.setValue(
            self.model.base_ptn.x_bg[0])
        self.widget.doubleSpinBox_Background_ROI_max.setValue(
            self.model.base_ptn.x_bg[-1])

    def _collect_background_areas(self):
        areas = []
        table = getattr(self.widget, "tableWidget_BackgroundConstraints", None)
        if table is None:
            return areas
        for row in range(table.rowCount()):
            item_min = table.item(row, 0)
            item_max = table.item(row, 1)
            if item_min is None or item_max is None:
                continue
            try:
                xmin = float(item_min.text())
                xmax = float(item_max.text())
            except Exception:
                continue
            if xmax < xmin:
                xmin, xmax = xmax, xmin
            areas.append([xmin, xmax])
        return areas

    def _apply_background_areas(self, areas):
        table = getattr(self.widget, "tableWidget_BackgroundConstraints", None)
        if table is None:
            return
        table.setRowCount(0)
        for area in (areas or []):
            try:
                xmin = float(area[0])
                xmax = float(area[1])
            except Exception:
                continue
            row = table.rowCount()
            table.insertRow(row)
            table.setItem(row, 0, QtWidgets.QTableWidgetItem(f"{xmin:.3f}"))
            table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{xmax:.3f}"))
        # the line below seems to be unnecessary, as there should be bgsub
        # self.model.base_ptn.subtract_bg(bg_roi, bg_params, yshift=0)
        # if self.model.waterfall_exist():
        #    for pattern in self.model.waterfall_ptn:
        #        pattern.get_chbg(bg_roi, bg_params, yshift=0)
