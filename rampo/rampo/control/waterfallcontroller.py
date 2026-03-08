import os
import copy
from qtpy import QtWidgets
from qtpy import QtCore
from qtpy import QtGui
from .mplcontroller import MplController
from .waterfalltablecontroller import WaterfallTableController
from ..utils import get_directory, get_temp_dir
from ..ds_ramspec import PatternPeakPo, Spectrum


class WaterfallController(object):

    def __init__(self, model, widget):
        self.model = model
        self.widget = widget
        self.base_ptn_ctrl = None
        self.capture_nav_state_cb = None
        self.apply_nav_state_cb = None
        self.waterfall_table_ctrl = \
            WaterfallTableController(self.model, self.widget)
        self.plot_ctrl = MplController(self.model, self.widget)
        self.connect_channel()

    def set_navigation_helpers(
            self, base_ptn_ctrl=None, capture_nav_state_cb=None,
            apply_nav_state_cb=None):
        self.base_ptn_ctrl = base_ptn_ctrl
        self.capture_nav_state_cb = capture_nav_state_cb
        self.apply_nav_state_cb = apply_nav_state_cb

    def connect_channel(self):
        self.widget.pushButton_MakeBasePtn.clicked.connect(self.make_base_ptn)
        self.widget.pushButton_AddPatterns.clicked.connect(self.add_patterns)
        self.widget.pushButton_CleanPatterns.clicked.connect(
            self.erase_waterfall_list)
        self.widget.pushButton_RemovePatterns.clicked.connect(
            self.remove_waterfall)
        self.widget.pushButton_UpPattern.clicked.connect(
            self.move_up_waterfall)
        self.widget.pushButton_DownPattern.clicked.connect(
            self.move_down_waterfall)
        self.widget.checkBox_IntNorm.clicked.connect(
            self._apply_changes_to_graph)
        self.widget.checkBox_ShowWaterfall.clicked.connect(
            self._apply_changes_to_graph)
        self.widget.pushButton_CheckAllWaterfall.clicked.connect(
            self.check_all_waterfall)
        self.widget.pushButton_UncheckAllWaterfall.clicked.connect(
            self.uncheck_all_waterfall)
        self.widget.pushButton_AddBasePtn.clicked.connect(
            self.add_base_pattern_to_waterfall)

    def make_base_ptn(self):
        # read selected from the table.  It should be single item
        idx_selected = self._find_a_waterfall_ptn()
        if idx_selected is None:
            QtWidgets.QMessageBox.warning(self.widget, "Warning",
                                          "Highlight an item to switch with.")
            return

        i = idx_selected
        target = copy.deepcopy(self.model.waterfall_ptn[i])
        original_fname = getattr(target, "_pkpo_original_fname", None) or target.fname
        if (original_fname is None) or (not os.path.exists(original_fname)):
            QtWidgets.QMessageBox.warning(
                self.widget, "Warning",
                "Set base is not allowed for fallback waterfall entries.\n"
                "Original waterfall file path is missing:\n"
                + str(original_fname)
            )
            return
        nav_state = None
        if callable(self.capture_nav_state_cb):
            nav_state = self.capture_nav_state_cb()
        # Preserve selected waterfall wavelength for no-PARAM fallback path.
        self.widget.doubleSpinBox_SetWavelength.setValue(target.wavelength)

        if (self.base_ptn_ctrl is not None) and hasattr(self.base_ptn_ctrl, "_load_a_new_pattern"):
            self.base_ptn_ctrl._load_a_new_pattern(original_fname)
        else:
            self.model.set_base_ptn(original_fname, target.wavelength)
            self.widget.lineEdit_DiffractionPatternFileName.setText(
                str(self.model.get_base_ptn_filename()))
            self.widget.doubleSpinBox_SetWavelength.setValue(
                self.model.get_base_ptn_wavelength())
            self.widget.label_XRayEnergy.setText("nm")

        if callable(self.apply_nav_state_cb) and (nav_state is not None):
            self.apply_nav_state_cb(nav_state)

        self.waterfall_table_ctrl.update()
        self._apply_changes_to_graph(reinforced=True)

    def check_all_waterfall(self):
        if not self.model.waterfall_exist():
            return
        for ptn in self.model.waterfall_ptn:
            ptn.display = True
        self.waterfall_table_ctrl.update()
        self._apply_changes_to_graph(reinforced=True)

    def uncheck_all_waterfall(self):
        if not self.model.waterfall_exist():
            return
        for ptn in self.model.waterfall_ptn:
            ptn.display = False
        self.waterfall_table_ctrl.update()
        self._apply_changes_to_graph(reinforced=True)

    def _apply_changes_to_graph(self, reinforced=False):
        """
        this does not do actual nomalization but the processing.
        actual normalization takes place in plotting.
        """
        self._reprocess_waterfall_patterns()
        if reinforced:
            pass
        else:
            if not self.model.waterfall_exist():
                return
            count = 0
            for ptn in self.model.waterfall_ptn:
                if ptn.display:
                    count += 1
            if count == 0:
                return
        self.plot_ctrl.update()

    def _current_background_fit_areas(self):
        fit_areas = []
        table = getattr(self.widget, "tableWidget_BackgroundConstraints", None)
        if table is None:
            return fit_areas
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
            fit_areas.append([xmin, xmax])
        return fit_areas

    def _reprocess_waterfall_patterns(self):
        if not self.model.waterfall_exist():
            return
        for idx, pattern in enumerate(list(self.model.waterfall_ptn)):
            try:
                self.model.waterfall_ptn[idx] = self._build_processed_waterfall_pattern(
                    getattr(pattern, "_pkpo_original_fname", None) or getattr(pattern, "fname", None),
                    display=bool(getattr(pattern, "display", False)),
                    color=getattr(pattern, "color", "white"),
                )
            except Exception:
                continue

    def _find_a_waterfall_ptn(self):
        idx_checked = [
            s.row() for s in
            self.widget.tableWidget_wfPatterns.selectionModel().selectedRows()]
        if idx_checked == []:
            return None
        else:
            return idx_checked[0]

    def add_patterns(self):
        """
        get files for waterfall plot
        """
        if not self.model.base_ptn_exist():
            QtWidgets.QMessageBox.warning(
                self.widget, "Warning",
                "Pick a base pattern first.")
            return
        f_input = QtWidgets.QFileDialog.getOpenFileNames(
            self.widget, "Choose additional data files", self.model.chi_path,
            "Data files (*.spe *.SPE *.chi)")
        files = f_input[0]
        self._add_patterns(files)

    def _add_patterns(self, files):
        if files is not None:
            for f in files:
                filename = str(f)
                self._append_processed_waterfall_pattern(filename, display=False)
            self.waterfall_table_ctrl.update()
            self._apply_changes_to_graph()
        return

    def add_base_pattern_to_waterfall(self):
        if not self.model.base_ptn_exist():
            QtWidgets.QMessageBox.warning(
                self.widget, "Warning",
                "Pick a base pattern first.")
            return
        filename = self.model.get_base_ptn_filename()
        if self.model.exist_in_waterfall(filename):
            self.widget.pushButton_AddBasePtn.setChecked(True)
            return
        self._append_processed_waterfall_pattern(filename, display=False)
        self.waterfall_table_ctrl.update()
        self._apply_changes_to_graph()

    def _append_processed_waterfall_pattern(self, filename, display=False):
        pattern = self._build_processed_waterfall_pattern(
            filename, display=display, color='white')
        self.model.waterfall_ptn.append(pattern)

    def _build_processed_waterfall_pattern(self, filename, display=False, color='white'):
        wavelength = float(self.widget.doubleSpinBox_SetWavelength.value())
        bg_roi = [
            float(self.widget.doubleSpinBox_Background_ROI_min.value()),
            float(self.widget.doubleSpinBox_Background_ROI_max.value()),
        ]
        bg_params = [int(self.widget.spinBox_BGParam1.value())]
        fit_areas = self._current_background_fit_areas()

        spectrum = Spectrum(filename)
        if str(filename).lower().endswith(".spe"):
            spectrum.apply_excitation_wavelength(wavelength)
            if hasattr(self.widget, "spinBox_CCDRowMin") and hasattr(self.widget, "spinBox_CCDRowMax"):
                spectrum.set_spe_row_roi(
                    int(self.widget.spinBox_CCDRowMin.value()),
                    int(self.widget.spinBox_CCDRowMax.value()))

        x_raw, y_raw = spectrum.get_raw()
        x_raw = spectrum.x_raw
        y_fit = y_raw
        if self.plot_ctrl is not None and bool(self.plot_ctrl._smoothing_active()):
            __, y_fit = self.plot_ctrl._get_smoothed_pattern_xy(x_raw, y_raw)
        spectrum.get_chbg(
            bg_roi,
            params=bg_params,
            yshift=0,
            fit_areas=fit_areas,
            y_source=y_fit,
        )

        pattern = PatternPeakPo()
        pattern.fname = filename
        pattern._pkpo_original_fname = filename
        pattern.wavelength = wavelength
        pattern.display = bool(display)
        pattern.color = str(color)
        pattern.x_raw = spectrum.x_raw
        pattern.y_raw = spectrum.y_raw
        pattern.raw_image = getattr(spectrum, "raw_image", None)
        pattern.x_wavelength_raw = getattr(spectrum, "x_wavelength_raw", None)
        pattern.row_roi = getattr(spectrum, "row_roi", None)
        pattern.set_bg(
            spectrum.x_bg,
            spectrum.y_bg,
            spectrum.x_bgsub,
            spectrum.y_bgsub,
            spectrum.roi,
            spectrum.params_chbg,
            fit_areas=getattr(spectrum, "bg_fit_areas", []),
        )
        return pattern

    def move_up_waterfall(self):
        # get selected cell number
        idx_selected = self._find_a_waterfall_ptn()
        if idx_selected is None:
            QtWidgets.QMessageBox.warning(self.widget, "Warning",
                                          "Highlight the item to move first.")
            return
        i = idx_selected
        self.model.waterfall_ptn[i - 1], self.model.waterfall_ptn[i] = \
            self.model.waterfall_ptn[i], self.model.waterfall_ptn[i - 1]
        self.widget.tableWidget_wfPatterns.selectRow(i - 1)
        self.waterfall_table_ctrl.update()
        self._apply_changes_to_graph()

    def move_down_waterfall(self):
        idx_selected = self._find_a_waterfall_ptn()
        if idx_selected is None:
            QtWidgets.QMessageBox.warning(self.widget, "Warning",
                                          "Highlight the item to move first.")
            return
        i = idx_selected
        self.model.waterfall_ptn[i + 1], self.model.waterfall_ptn[i] = \
            self.model.waterfall_ptn[i], self.model.waterfall_ptn[i + 1]
        self.widget.tableWidget_wfPatterns.selectRow(i + 1)
        self.waterfall_table_ctrl.update()
        self._apply_changes_to_graph()

    def erase_waterfall_list(self):
        self.model.reset_waterfall_ptn()
        # self.widget.tableWidget_wfPatterns.clearContents()
        self.waterfall_table_ctrl.update()
        self._apply_changes_to_graph(reinforced=True)

    def remove_waterfall(self):
        reply = QtWidgets.QMessageBox.question(
            self.widget, 'Message',
            'Are you sure you want to remove the highlighted pattern?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes)
        if reply == QtWidgets.QMessageBox.No:
            return
        # print self.widget.tableWidget_JCPDS.selectedIndexes().__len__()
        idx_checked = [
            s.row() for s in
            self.widget.tableWidget_wfPatterns.selectionModel().selectedRows()]
        if idx_checked == []:
            QtWidgets.QMessageBox.warning(
                self.widget, 'Warning',
                'In order to remove, highlight the names.')
            return
        else:
            idx_checked.reverse()
            for idx in idx_checked:
                self.model.waterfall_ptn.remove(self.model.waterfall_ptn[idx])
                self.widget.tableWidget_wfPatterns.removeRow(idx)
#        self._list_jcpds()
            self._apply_changes_to_graph()
