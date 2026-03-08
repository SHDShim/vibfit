import os
from qtpy import QtWidgets
from ..utils import get_sorted_filelist, find_from_filelist, readchi, \
    make_filename, writechi, get_directory
from ..utils import undo_button_press, get_temp_dir
import datetime
from .mplcontroller import MplController
from .ccdcontroller import CCDController


class BaseSpectrumController(object):

    def __init__(self, model, widget, session_ctrl=None):
        self.model = model
        self.widget = widget
        self.session_ctrl = session_ctrl
        self.plot_ctrl = MplController(self.model, self.widget)
        self.ccd_ctrl = CCDController(self.model, self.widget)
        self.connect_channel()

    def connect_channel(self):
        self.widget.pushButton_NewBasePtn.clicked.connect(
            self.select_base_ptn)
        self.widget.lineEdit_DiffractionPatternFileName.editingFinished.\
            connect(self.load_new_base_pattern_from_name)

    def select_base_ptn(self):
        """
        opens a file select dialog
        """
        filen = QtWidgets.QFileDialog.getOpenFileName(
            self.widget, "Open an SPE or CHI File", self.model.chi_path,
            "Data files (*.spe *.SPE *.chi)")[0]
        self._setshow_new_base_ptn(str(filen))

    def load_new_base_pattern_from_name(self):
        if self.widget.lineEdit_DiffractionPatternFileName.isModified():
            filen = self.widget.lineEdit_DiffractionPatternFileName.text()
            self._setshow_new_base_ptn(filen)

    def _setshow_new_base_ptn(self, filen):
        """
        load and then send signal to update_graph
        """
        if os.path.exists(filen):
            self.model.set_chi_path(os.path.split(filen)[0])
            if self.model.base_ptn_exist():
                old_filename = self.model.get_base_ptn_filename()
            else:
                old_filename = None
            new_filename = filen
            self._load_a_new_pattern(new_filename)
            if old_filename is None:
                self.plot_new_graph()
            else:
                self.apply_changes_to_graph()
        else:
            QtWidgets.QMessageBox.warning(
                self.widget, 'Warning', 'Cannot find ' + filen)
            # self.widget.lineEdit_DiffractionPatternFileName.setText(
            #    self.model.get_base_ptn_filename())

    def _load_a_new_pattern(self, new_filename):
        """
        load and process base pattern.  does not signal to update_graph
        """
        if self.session_ctrl is not None:
            # Reset carry-over provenance for generic/manual loads.
            self.session_ctrl.set_carryover_source_chi(None)
        self.model.set_base_ptn(
            new_filename, self.widget.doubleSpinBox_SetWavelength.value())
        # self.widget.textEdit_DiffractionPatternFileName.setText(
        #    '1D Pattern: ' + self.model.get_base_ptn_filename())
        self.widget.lineEdit_DiffractionPatternFileName.setText(
            str(self.model.get_base_ptn_filename()))
        main_ctrl = getattr(self.widget, "_main_controller", None)
        if main_ctrl is not None and hasattr(main_ctrl, "sync_background_roi_spinboxes"):
            try:
                main_ctrl.sync_background_roi_spinboxes(force_full_range=True)
            except Exception:
                pass
        # Prefer loading full PARAM session state when available for this CHI.
        if self.session_ctrl is not None:
            loaded_param = self.session_ctrl.autoload_param_for_chi(new_filename)
            if loaded_param:
                if self.ccd_ctrl._is_spe_source():
                    try:
                        self.ccd_ctrl.process_temp_cake()
                    except Exception:
                        pass
                # Ensure File > Data backup table reflects the newly loaded CHI
                # immediately, without requiring tab changes.
                self.session_ctrl.refresh_backup_table()
                print(str(datetime.datetime.now())[:-7],
                    ': Loaded PARAM session for this CHI.')
                return
        print(str(datetime.datetime.now())[:-7], 
                ": Receive request to open ", 
                str(self.model.get_base_ptn_filename()))
        temp_dir = get_temp_dir(self.model.get_base_ptn_filename())
        if True:
            if os.path.exists(temp_dir):
                success = self.model.base_ptn.read_bg_from_tempfile(
                    temp_dir=temp_dir)
                if success:
                    self._update_bg_params_in_widget()
                    print(str(datetime.datetime.now())[:-7], 
                        ': Read temp chi successfully.')
                else:
                    self._update_bgsub_from_current_values()
                    print(str(datetime.datetime.now())[:-7], 
                        ': No temp chi file found. Force new bgsub fit.')
            else:
                os.makedirs(temp_dir)
                self._update_bgsub_from_current_values()
                print(str(datetime.datetime.now())[:-7], 
                    ': No temp chi file found. Force new bgsub fit.')
        if (not self.ccd_ctrl._is_spe_source()) and \
                (not self.model.associated_image_exists()) and \
                (not self.ccd_ctrl._ignore_raw_data_missing()):
            return

        success = self.ccd_ctrl.process_temp_cake()
        if (not success) and \
                self.ccd_ctrl._ignore_raw_data_missing() and \
                (not self.model.associated_image_exists()):
            QtWidgets.QMessageBox.warning(
                self.widget, 'Warning',
                'Rampo cannot process the CCD view: no raw image and no existing cached files were found.')
        # Keep backup table in File > Data synchronized right after CHI load.
        if self.session_ctrl is not None:
            self.session_ctrl.refresh_backup_table()

    def _update_bg_params_in_widget(self):
        self.widget.spinBox_BGParam1.setValue(
            self.model.base_ptn.params_chbg[0])
        self.widget.doubleSpinBox_Background_ROI_min.setValue(
            self.model.base_ptn.roi[0])
        self.widget.doubleSpinBox_Background_ROI_max.setValue(
            self.model.base_ptn.roi[1])
        table = getattr(self.widget, "tableWidget_BackgroundConstraints", None)
        if table is not None:
            table.setRowCount(0)
            for area in getattr(self.model.base_ptn, "bg_fit_areas", []) or []:
                try:
                    xmin = float(area[0])
                    xmax = float(area[1])
                except Exception:
                    continue
                row = table.rowCount()
                table.insertRow(row)
                table.setItem(row, 0, QtWidgets.QTableWidgetItem(f"{xmin:.3f}"))
                table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{xmax:.3f}"))

    def _update_bgsub_from_current_values(self):
        fit_areas = []
        table = getattr(self.widget, "tableWidget_BackgroundConstraints", None)
        if table is not None:
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
        x_raw, y_raw = self.model.base_ptn.get_raw()
        __, y_fit = self.plot_ctrl._get_smoothed_pattern_xy(x_raw, y_raw)
        if not self.plot_ctrl._smoothing_active():
            y_fit = y_raw
        if (x_raw.min() >= self.widget.doubleSpinBox_Background_ROI_min.value()) or \
                (x_raw.max() <= self.widget.doubleSpinBox_Background_ROI_min.value()):
            self.widget.doubleSpinBox_Background_ROI_min.setValue(x_raw.min())
        if (x_raw.max() <= self.widget.doubleSpinBox_Background_ROI_max.value()) or \
                (x_raw.min() >= self.widget.doubleSpinBox_Background_ROI_max.value()):
            self.widget.doubleSpinBox_Background_ROI_max.setValue(x_raw.max())
        self.model.base_ptn.subtract_bg(
            [self.widget.doubleSpinBox_Background_ROI_min.value(),
                self.widget.doubleSpinBox_Background_ROI_max.value()],
            [self.widget.spinBox_BGParam1.value()],
            yshift=0,
            fit_areas=fit_areas,
            y_source=y_fit)
        temp_dir = get_temp_dir(self.model.get_base_ptn_filename())
        self.model.base_ptn.write_temporary_bgfiles(temp_dir)

    def apply_changes_to_graph(self):
        self.plot_ctrl.update()

    def plot_new_graph(self):
        self.plot_ctrl.zoom_out_graph()


BasePatternController = BaseSpectrumController
