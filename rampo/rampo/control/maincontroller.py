import os
import glob
import numpy as np
import copy
#from matplotlib.backend_bases import key_press_handler
from qtpy import QtWidgets
from qtpy import QtCore
import gc
import datetime
from contextlib import contextmanager
from matplotlib.widgets import RectangleSelector
from ..view import MainWindow
from ..model import PeakPoModel, PeakPoModel8
from .basespectrumcontroller import BaseSpectrumController
from .mplcontroller import MplController
# CCD controller is called in BasePatternController already.
from .waterfallcontroller import WaterfallController
from .jcpdscontroller import JcpdsController
from .waterfalltablecontroller import WaterfallTableController
from .jcpdstablecontroller import JcpdsTableController
from .sessioncontroller import SessionController
from .diffcontroller import DiffController
from .peakfitcontroller import PeakFitController
from .peakfittablecontroller import PeakfitTableController
from .exportpythoncontroller import ExportPythonController
from .mapcontroller import MapController
from .sequencecontroller import SequenceController
from ..utils import dialog_savefile, writechi, extract_extension, \
    convert_wl_to_energy, get_sorted_filelist, find_from_filelist, \
    make_filename, get_directory, get_temp_dir
from ..ds_ramspec import get_data_section
#from utils import readchi, make_filename, writechi


class MainController(object):

    def __init__(self):
        self._shutdown_done = False
        self._defer_plot_update_count = 0

        self.widget = MainWindow()
        self.widget._main_controller = self

        self.model = PeakPoModel8()

        self.session_ctrl = SessionController(self.model, self.widget)

        self.base_spectrum_ctrl = BaseSpectrumController(
            self.model, self.widget, session_ctrl=self.session_ctrl)
        self.base_ptn_ctrl = self.base_spectrum_ctrl
        
        self.plot_ctrl = MplController(self.model, self.widget)
        self._replace_toolbar_save_action()

        self.diff_ctrl = DiffController(self.model, self.widget, self.plot_ctrl)
        self.plot_ctrl.set_diff_controller(self.diff_ctrl)
        self._propagate_diff_controller()

        self.map_ctrl = MapController(self.model, self.widget)
        self.map_ctrl.set_helpers(
            base_ptn_ctrl=self.base_spectrum_ctrl,
            plot_ctrl=self.plot_ctrl)

        self.seq_ctrl = SequenceController(self.model, self.widget)
        self.seq_ctrl.set_helpers(
            base_ptn_ctrl=self.base_spectrum_ctrl,
            plot_ctrl=self.plot_ctrl)
        
        self.ccdazi_ctrl = None
        
        self.waterfall_ctrl = WaterfallController(self.model, self.widget)
        self.waterfall_ctrl.set_navigation_helpers(
            base_ptn_ctrl=self.base_spectrum_ctrl,
            capture_nav_state_cb=self._capture_nav_carry_state,
            apply_nav_state_cb=self._apply_nav_carry_state,
        )
        
        self.jcpds_ctrl = JcpdsController(self.model, self.widget)
        
        self.waterfalltable_ctrl = WaterfallTableController(self.model, self.widget)
        
        self.jcpdstable_ctrl = JcpdsTableController(self.model, self.widget)
        
        self.peakfit_ctrl = PeakFitController(self.model, self.widget)
        
        self.peakfit_table_ctrl = PeakfitTableController(self.model, self.widget)

        self.export_py_ctrl = ExportPythonController(
            self.model, self.widget, plot_ctrl=self.plot_ctrl)
        self._propagate_diff_controller()
        self._bg_area_selector = None
        
        self.read_setting()
        
        self.connect_channel()
        
        self.clip = QtWidgets.QApplication.clipboard()

    def _propagate_diff_controller(self):
        # Multiple controllers keep their own MplController instances.
        # Keep Diff behavior consistent across all redraw paths.
        for ctrl_name in (
            "session_ctrl",
            "base_spectrum_ctrl",
            "waterfall_ctrl",
            "jcpds_ctrl",
            "waterfalltable_ctrl",
            "jcpdstable_ctrl",
            "peakfit_ctrl",
            "peakfit_table_ctrl",
            "ccdazi_ctrl",
        ):
            ctrl = getattr(self, ctrl_name, None)
            plot_ctrl = getattr(ctrl, "plot_ctrl", None)
            if (plot_ctrl is not None) and hasattr(plot_ctrl, "set_diff_controller"):
                plot_ctrl.set_diff_controller(self.diff_ctrl)
        # Nested cake controller under base pattern controller.
        if hasattr(self, "base_spectrum_ctrl") and hasattr(self.base_spectrum_ctrl, "ccd_ctrl"):
            ccd_plot_ctrl = getattr(self.base_spectrum_ctrl.ccd_ctrl, "plot_ctrl", None)
            if (ccd_plot_ctrl is not None) and hasattr(ccd_plot_ctrl, "set_diff_controller"):
                ccd_plot_ctrl.set_diff_controller(self.diff_ctrl)

    def _replace_toolbar_save_action(self):
        toolbar = getattr(getattr(self.widget, "mpl", None), "ntb", None)
        if toolbar is None:
            return

        def _save_session_from_toolbar(*args, **kwargs):
            del args, kwargs
            self.session_ctrl.save_dpp()

        try:
            toolbar.save_figure = _save_session_from_toolbar
        except Exception:
            pass

        try:
            for action in toolbar.actions():
                text = str(action.text() or "").strip().lower()
                tip = str(action.toolTip() or "").strip().lower()
                if text == "save" or tip.startswith("save the figure"):
                    try:
                        action.triggered.disconnect()
                    except Exception:
                        pass
                    action.triggered.connect(_save_session_from_toolbar)
                    action.setToolTip("Save Rampo session")
                    action.setStatusTip("Save Rampo session")
                    break
        except Exception:
            pass

    def show_window(self):
        """Show the main window and ensure it renders"""
        # Show and let Qt/Matplotlib render in normal event flow.
        self.widget.show()
        
        # Bring to front (important on macOS)
        self.widget.raise_()
        self.widget.activateWindow()

    def shutdown(self):
        if self._shutdown_done:
            return
        self._shutdown_done = True
        try:
            self.write_setting()
        except Exception:
            pass
        try:
            if hasattr(self.widget, 'mpl') and hasattr(self.widget.mpl, 'shutdown'):
                self.widget.mpl.shutdown()
        except Exception:
            pass
        try:
            if self.widget is not None:
                self.widget.close()
        except Exception:
            pass
        
    def connect_channel(self):
        # connecting events
        self.widget.mpl.canvas.mpl_connect(
            'button_press_event', self.deliver_mouse_signal)
        self.widget.mpl.canvas.mpl_connect(
            'key_press_event', self.on_key_press)
        self.widget.doubleSpinBox_Pressure.valueChanged.connect(
            self.apply_pt_to_graph)
        self.widget.pushButton_S_PIncrease.clicked.connect(
            lambda: self.quick_p_change(1))
        self.widget.pushButton_S_PDecrease.clicked.connect(
            lambda: self.quick_p_change(-1))
        self.widget.doubleSpinBox_Temperature.valueChanged.connect(
            self.apply_pt_to_graph)
        self.widget.doubleSpinBox_SetWavelength.valueChanged.connect(
            self.apply_wavelength)
        for name in ("spinBox_SpectrumDespike", "spinBox_SpectrumSGWindow", "spinBox_SpectrumSGPoly"):
            if hasattr(self.widget, name):
                getattr(self.widget, name).valueChanged.connect(
                    self.apply_spectrum_smoothing)
        if hasattr(self.widget, "pushButton_BGAreaAdd"):
            self.widget.pushButton_BGAreaAdd.toggled.connect(
                self.toggle_background_area_selector)
        if hasattr(self.widget, "pushButton_BGAreaRemove"):
            self.widget.pushButton_BGAreaRemove.clicked.connect(
                self.remove_selected_background_area)
        if hasattr(self.widget, "pushButton_BGAreaClear"):
            self.widget.pushButton_BGAreaClear.clicked.connect(
                self.clear_background_areas)
        if hasattr(self.widget, "pushButton_BGFit"):
            self.widget.pushButton_BGFit.clicked.connect(
                self.update_bgsub)
        if hasattr(self.widget, "pushButton_ExportPythonView"):
            self.widget.pushButton_ExportPythonView.clicked.connect(
                self.export_py_ctrl.export_current_view)
        self.widget.pushButton_SetXEat30.clicked.connect(
            lambda: self.setXEat(532.22))
        self.widget.pushButton_SetXEat37.clicked.connect(
            lambda: self.setXEat(660.00))
        self.widget.pushButton_SetXEat42.clicked.connect(
            lambda: self.setXEat(785.00))
        self.widget.pushButton_ImportJlist.clicked.connect(
            self.load_jlist_from_session)
        if hasattr(self.widget, "pushButton_SpectrumRaw"):
            self.widget.pushButton_SpectrumRaw.clicked.connect(
                self.reset_spectrum_smoothing_to_raw)
        self.widget.checkBox_LongCursor.stateChanged.connect(
            self._handle_cursor_toggle)  # Changed from clicked to stateChanged
        # ✅ ADD: Connect checkbox to deactivate toolbar
        self.widget.checkBox_LongCursor.stateChanged.connect(
            self._on_long_cursor_changed)
        self.widget.checkBox_ShowMillerIndices.clicked.connect(
            self.apply_changes_to_graph)
        self.widget.comboBox_BasePtnLineThickness.currentIndexChanged.connect(
            self.apply_changes_to_graph)
        self.widget.comboBox_PtnJCPDSBarThickness.currentIndexChanged.connect(
            self.apply_changes_to_graph)
        self.widget.comboBox_CakeJCPDSBarThickness.currentIndexChanged.connect(
            self.apply_changes_to_graph)
        self.widget.comboBox_BkgnLineThickness.currentIndexChanged.connect(
            self.apply_changes_to_graph)
        self.widget.comboBox_WaterfallLineThickness.currentIndexChanged.connect(
            self.apply_changes_to_graph)
        self.widget.comboBox_HKLFontSize.currentIndexChanged.connect(
            self.apply_changes_to_graph)
        self.widget.comboBox_PnTFontSize.currentIndexChanged.connect(
            self.apply_changes_to_graph)
        if hasattr(self.widget, "comboBox_LegendFontSize"):
            self.widget.comboBox_LegendFontSize.currentIndexChanged.connect(
                self.apply_changes_to_graph)
        if hasattr(self.widget, "comboBox_WaterfallFontSize"):
            self.widget.comboBox_WaterfallFontSize.currentIndexChanged.connect(
                self.apply_changes_to_graph)
        self.widget.checkBox_ShortPlotTitle.clicked.connect(
            self.apply_changes_to_graph)
        if hasattr(self.widget, "spinBox_TitleFontSize"):
            self.widget.spinBox_TitleFontSize.valueChanged.connect(
                self.apply_changes_to_graph)
        if hasattr(self.widget, "spinBox_TitleMaxLength"):
            self.widget.spinBox_TitleMaxLength.valueChanged.connect(
                self.apply_changes_to_graph)
        if hasattr(self.widget, "checkBox_TitleTruncateMiddle"):
            self.widget.checkBox_TitleTruncateMiddle.clicked.connect(
                self.apply_changes_to_graph)
        self.widget.checkBox_ShowCakeLabels.clicked.connect(
            self.apply_changes_to_graph)
        self.widget.checkBox_ShowLargePnT.clicked.connect(
            self.apply_changes_to_graph)
        # navigation toolbar modification.  Do not move the followings to
        # other controller files.
        #self.widget.pushButton_toPkFt.clicked.connect(self.to_PkFt)
        #self.widget.pushButton_fromPkFt.clicked.connect(self.from_PkFt)
        self.widget.checkBox_NightView.clicked.connect(self.set_nightday_view)
        if hasattr(self.widget, "comboBox_CakeColormap"):
            self.widget.comboBox_CakeColormap.currentIndexChanged.connect(
                self.apply_changes_to_graph)
        self.widget.pushButton_S_Zoom.clicked.connect(self.plot_new_graph)
        self.widget.checkBox_AutoY.clicked.connect(self.apply_changes_to_graph)
        self.widget.checkBox_BgSub.clicked.connect(self.apply_changes_to_graph)
        if hasattr(self.widget, "checkBox_ShowBg"):
            self.widget.checkBox_ShowBg.clicked.connect(
                self.apply_changes_to_graph)
        self.widget.checkBox_ShowWaterfallLabels.clicked.connect(
            self.apply_changes_to_graph)
        self.widget.checkBox_ShowMillerIndices_Cake.clicked.connect(
            self.apply_changes_to_graph)
        # self.widget.actionClose.triggered.connect(self.closeEvent)
        self.widget.tabWidget.currentChanged.connect(self.check_for_peakfit)
        # self.widget.tabWidget.setTabEnabled(8, False)
        if hasattr(self.widget, "doubleSpinBox_CCDScaleMin"):
            self.widget.doubleSpinBox_CCDScaleMin.valueChanged.connect(
                self.apply_changes_to_graph)
        if hasattr(self.widget, "doubleSpinBox_CCDScaleMax"):
            self.widget.doubleSpinBox_CCDScaleMax.valueChanged.connect(
                self.apply_changes_to_graph)
        self.widget.horizontalSlider_CakeAxisSize.valueChanged.connect(
            self.apply_changes_to_graph)
        self.widget.horizontalSlider_JCPDSBarScale.valueChanged.connect(
            self.apply_changes_to_graph)
        self.widget.horizontalSlider_JCPDSBarPosition.valueChanged.connect(
            self.apply_changes_to_graph)
        self.widget.horizontalSlider_WaterfallGaps.valueChanged.connect(
            self.apply_changes_to_graph)
        self.widget.doubleSpinBox_JCPDS_cake_Alpha.valueChanged.connect(
            self.apply_changes_to_graph)
        self.widget.doubleSpinBox_JCPDS_ptn_Alpha.valueChanged.connect(
            self.apply_changes_to_graph)
        self.widget.pushButton_UpdateJCPDSSteps.clicked.connect(
            self.update_jcpds_table)
        self.widget.pushButton_PrevBasePtn.clicked.connect(
            lambda: self.goto_next_file('previous'))
        self.widget.pushButton_NextBasePtn.clicked.connect(
            lambda: self.goto_next_file('next'))
        self.widget.pushButton_S_PrevBasePtn.clicked.connect(
            lambda: self.goto_next_file('previous'))
        self.widget.pushButton_S_NextBasePtn.clicked.connect(
            lambda: self.goto_next_file('next'))
        self.widget.pushButton_LastBasePtn.clicked.connect(
            lambda: self.goto_next_file('last'))
        self.widget.pushButton_FirstBasePtn.clicked.connect(
            lambda: self.goto_next_file('first'))

    def _on_long_cursor_changed(self, state):
        """Deactivate pan/zoom when vertical cursor is enabled"""
        if state == QtCore.Qt.Checked:
            # Deactivate any active toolbar mode
            toolbar = self.widget.mpl.canvas.toolbar
            if toolbar and toolbar.mode:
                # Click the active button again to deactivate it
                if toolbar.mode == 'zoom rect':
                    toolbar.zoom()  # Toggle off
                elif toolbar.mode == 'pan/zoom':
                    toolbar.pan()   # Toggle off
            
            # Update the plot to show cursor
            self.plot_ctrl.update()
        else:
            # Update the plot to remove cursor
            self.plot_ctrl.update()

    def _handle_cursor_toggle(self, state):
        """Handle vertical cursor checkbox - implement mutual exclusivity with toolbar"""
        if state == QtCore.Qt.Checked:
            # Cursor was just checked - deactivate toolbar pan/zoom
            toolbar = self.widget.mpl.canvas.toolbar
            if toolbar:
                # Check which mode is active
                current_mode = ''
                if hasattr(toolbar, 'mode'):
                    # New matplotlib API (3.3+)
                    current_mode = toolbar.mode
                elif hasattr(toolbar, '_active'):
                    # Old matplotlib API
                    current_mode = toolbar._active or ''
                
                # Deactivate zoom or pan if active
                if current_mode == 'zoom rect' or current_mode == 'ZOOM':
                    toolbar.zoom()  # Toggle zoom off
                    print("  ✓ Zoom deactivated (cursor enabled)")
                elif current_mode == 'pan/zoom' or current_mode == 'PAN':
                    toolbar.pan()   # Toggle pan off
                    print("  ✓ Pan deactivated (cursor enabled)")
                
                # Ensure toolbar state is cleared
                if hasattr(toolbar, 'mode'):
                    toolbar.mode = ''
                if hasattr(self.plot_ctrl, '_toolbar_active'):
                    self.plot_ctrl._toolbar_active = False
        
        # Update plot to show/hide cursor
        self.apply_changes_to_graph()

    def integrate_to_1d(self):
        # ccdazi_ctrl is pointing CCDAziController.
        filen = self.ccdazi_ctrl.integrate_to_1d()

        if filen is None:
            return
        else:
            reply = QtWidgets.QMessageBox.question(
                self.widget, 'Message',
                'Do you want to add this file ({:s}) to the waterfall list?'.
                format(filen),
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes)
            if reply == QtWidgets.QMessageBox.No:
                return
            else:
                # add to waterfall
                self.waterfall_ctrl._add_patterns([filen])

    def quick_p_change(self, direction):
        step = self.widget.doubleSpinBox_PStep.value()
        p_value = self.widget.doubleSpinBox_Pressure.value()
        self.widget.doubleSpinBox_Pressure.setValue(p_value + step * direction)

    def quick_temp_change(self, direction):
        step = self.widget.spinBox_TStep.value()
        temp_value = self.widget.doubleSpinBox_Temperature.value()
        self.widget.doubleSpinBox_Temperature.setValue(
            temp_value + step * direction)

    def update_jcpds_table(self):
        step = float(self.widget.doubleSpinBox_JCPDSStep.value())
        self.jcpdstable_ctrl.update_steps_only(step)

    def del_temp_chi(self):
        reply = QtWidgets.QMessageBox.question(
            self.widget, 'Message',
            'This can slow down PeakPo, but update the background. Proceed?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes)
        if reply == QtWidgets.QMessageBox.No:
            return
        if self._temporary_pkpo_exists():
            temp_dir = get_temp_dir(self.model.get_base_ptn_filename())
            temp_chi = os.path.join(temp_dir, '*.chi')
            for f in glob.glob(temp_chi):
                os.remove(f)

    def del_temp_cake(self):
        reply = QtWidgets.QMessageBox.question(
            self.widget, 'Message',
            'This deletes cached CCD files and may slow the next redraw. Proceed?',
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes)
        if reply == QtWidgets.QMessageBox.No:
            return
        if self._temporary_pkpo_exists():
            temp_dir = get_temp_dir(self.model.get_base_ptn_filename())
            temp_cake = os.path.join(temp_dir, '*.npy')
            for f in glob.glob(temp_cake):
                os.remove(f)

    def _temporary_pkpo_exists(self):
        temp_dir = get_temp_dir(self.model.get_base_ptn_filename())
        return os.path.exists(temp_dir)

    def check_for_peakfit(self, i):
        if hasattr(self.widget, "tab_PkFt"):
            is_peakfit_tab = (self.widget.tabWidget.widget(i) == self.widget.tab_PkFt)
        else:
            is_peakfit_tab = (i == 8)
        if is_peakfit_tab:
            self.widget.checkBox_AutoY.setChecked(False)
            self.apply_changes_to_graph()

    def apply_changes_to_graph(self):
        if self._plot_update_deferred():
            return
        self.plot_ctrl.update()
        if hasattr(self, "map_ctrl") and (self.map_ctrl is not None):
            try:
                self.map_ctrl.refresh_roi_overlays()
            except Exception:
                pass
        if hasattr(self, "seq_ctrl") and (self.seq_ctrl is not None):
            try:
                self.seq_ctrl.refresh_roi_overlays()
            except Exception:
                pass

    def plot_new_graph(self):
        self.plot_ctrl.zoom_out_graph()
        if hasattr(self, "map_ctrl") and (self.map_ctrl is not None):
            try:
                self.map_ctrl.refresh_roi_overlays()
            except Exception:
                pass
        if hasattr(self, "seq_ctrl") and (self.seq_ctrl is not None):
            try:
                self.seq_ctrl.refresh_roi_overlays()
            except Exception:
                pass

    def load_jlist_from_session(self):
        QtWidgets.QMessageBox.information(
            self.widget, "JSON Only",
            "Legacy session import is removed. Load RAPO entries from the current JSON session folder instead.")

    def save_bgsubchi(self):
        """
        Save bg subtractd pattern to a chi file
        """
        if not self.model.base_ptn_exist():
            return
        filen_chi_t = self.model.make_filename('bgsub.chi')
        filen_chi = dialog_savefile(self.widget, filen_chi_t)
        if str(filen_chi) == '':
            return
        x, y = self.model.base_ptn.get_bgsub()
        preheader_line0 = \
            '2-theta # BG ROI: {0: .5e}, {1: .5e} \n'.format(
                self.widget.doubleSpinBox_Background_ROI_min.value(),
                self.widget.doubleSpinBox_Background_ROI_max.value())
        preheader_line1 = \
            '2-theta # BG Params: {0: d} \n'.format(
                self.widget.spinBox_BGParam1.value())
        preheader_line2 = '\n'
        writechi(filen_chi, x, y, preheader=preheader_line0 +
                 preheader_line1 + preheader_line2)

    def write_setting(self):
        """
        Write default setting
        """
        # self.settings = QtCore.QSettings('DS', 'PeakPo')
        self.settings = QtCore.QSettings('DS', 'PeakPo')
        # print('write:' + self.model.chi_path)
        self.settings.setValue('chi_path', self.model.chi_path)
        self.settings.setValue('jcpds_path', self.model.jcpds_path)
        self.settings.setValue(
            'fontsize_pt_label', self.widget.comboBox_PnTFontSize.currentText())
        self.settings.setValue(
            'fontsize_miller', self.widget.comboBox_HKLFontSize.currentText())
        if hasattr(self.widget, "comboBox_LegendFontSize"):
            self.settings.setValue(
                'fontsize_legend',
                self.widget.comboBox_LegendFontSize.currentText())
        if hasattr(self.widget, "comboBox_WaterfallFontSize"):
            self.settings.setValue(
                'fontsize_waterfall_label',
                self.widget.comboBox_WaterfallFontSize.currentText())
        if hasattr(self.widget, "spinBox_SpectrumDespike"):
            self.settings.setValue(
                'spectrum_smoothing_despike',
                int(self.widget.spinBox_SpectrumDespike.value()))
        if hasattr(self.widget, "spinBox_SpectrumSGWindow"):
            self.settings.setValue(
                'spectrum_smoothing_sg_window',
                int(self.widget.spinBox_SpectrumSGWindow.value()))
        if hasattr(self.widget, "spinBox_SpectrumSGPoly"):
            self.settings.setValue(
                'spectrum_smoothing_sg_polyorder',
                int(self.widget.spinBox_SpectrumSGPoly.value()))
        # CHI navigation carry-over policy
        nav_keys = [
            ("carry_nav_jcpds", "checkBox_CarryNavJCPDS"),
            ("carry_nav_pressure", "checkBox_CarryNavPressure"),
            ("carry_nav_spectrum_smoothing", "checkBox_CarryNavSpectrumSmoothing"),
            ("carry_nav_cake_z_scale", "checkBox_CarryNavCakeZScale"),
            ("carry_nav_ccd_roi", "checkBox_CarryNavCCDRoi"),
            ("carry_nav_background", "checkBox_CarryNavBackground"),
            ("carry_nav_waterfall_list", "checkBox_CarryNavWaterfall"),
            ("carry_nav_fits_information", "checkBox_CarryNavFits"),
        ]
        for key, attr in nav_keys:
            if hasattr(self.widget, attr):
                self.settings.setValue(key, bool(getattr(self.widget, attr).isChecked()))

        for key, attr in self._plot_config_setting_bindings():
            if hasattr(self.widget, attr):
                self._save_widget_to_settings(key, getattr(self.widget, attr))
        

    def read_setting(self):
        """
        Read default setting
        """
        self.settings = QtCore.QSettings('DS', 'PeakPo')
        # self.settings.setFallbacksEnabled(False)
        self.model.set_chi_path(self.settings.value('chi_path'))
        self.model.set_jcpds_path(self.settings.value('jcpds_path'))
        pnt_fs = str(self.settings.value(
            'fontsize_pt_label', self.widget.comboBox_PnTFontSize.currentText()))
        hkl_fs = str(self.settings.value(
            'fontsize_miller', self.widget.comboBox_HKLFontSize.currentText()))
        if self.widget.comboBox_PnTFontSize.findText(pnt_fs) >= 0:
            self.widget.comboBox_PnTFontSize.setCurrentText(pnt_fs)
        if self.widget.comboBox_HKLFontSize.findText(hkl_fs) >= 0:
            self.widget.comboBox_HKLFontSize.setCurrentText(hkl_fs)
        if hasattr(self.widget, "comboBox_LegendFontSize"):
            leg_fs = str(self.settings.value(
                'fontsize_legend',
                self.widget.comboBox_LegendFontSize.currentText()))
            if self.widget.comboBox_LegendFontSize.findText(leg_fs) >= 0:
                self.widget.comboBox_LegendFontSize.setCurrentText(leg_fs)
        if hasattr(self.widget, "comboBox_WaterfallFontSize"):
            wf_fs = str(self.settings.value(
                'fontsize_waterfall_label',
                self.widget.comboBox_WaterfallFontSize.currentText()))
            if self.widget.comboBox_WaterfallFontSize.findText(wf_fs) >= 0:
                self.widget.comboBox_WaterfallFontSize.setCurrentText(wf_fs)
        if hasattr(self.widget, "comboBox_CakeColormap"):
            # Always start with inferno regardless of previous sessions/settings.
            self.widget.comboBox_CakeColormap.setCurrentText("inferno")
        if hasattr(self.widget, "spinBox_SpectrumDespike"):
            try:
                despike = int(self.settings.value('spectrum_smoothing_despike', 0))
            except Exception:
                despike = 0
            self.widget.spinBox_SpectrumDespike.setValue(despike)
        if hasattr(self.widget, "spinBox_SpectrumSGWindow"):
            try:
                sg_window = int(self.settings.value('spectrum_smoothing_sg_window', 0))
            except Exception:
                sg_window = 0
            self.widget.spinBox_SpectrumSGWindow.setValue(sg_window)
        if hasattr(self.widget, "spinBox_SpectrumSGPoly"):
            try:
                sg_poly = int(self.settings.value('spectrum_smoothing_sg_polyorder', 3))
            except Exception:
                sg_poly = 3
            self.widget.spinBox_SpectrumSGPoly.setValue(sg_poly)
        if hasattr(self.widget, "spinBox_SpectrumDespike"):
            self.apply_spectrum_smoothing()
        nav_defaults = {
            "checkBox_CarryNavJCPDS": True,
            "checkBox_CarryNavPressure": True,
            "checkBox_CarryNavSpectrumSmoothing": True,
            "checkBox_CarryNavCakeZScale": False,
            "checkBox_CarryNavCCDRoi": True,
            "checkBox_CarryNavBackground": False,
            "checkBox_CarryNavWaterfall": True,
            "checkBox_CarryNavFits": False,
        }
        nav_keys = {
            "checkBox_CarryNavJCPDS": "carry_nav_jcpds",
            "checkBox_CarryNavPressure": "carry_nav_pressure",
            "checkBox_CarryNavSpectrumSmoothing": "carry_nav_spectrum_smoothing",
            "checkBox_CarryNavCakeZScale": "carry_nav_cake_z_scale",
            "checkBox_CarryNavCCDRoi": "carry_nav_ccd_roi",
            "checkBox_CarryNavBackground": "carry_nav_background",
            "checkBox_CarryNavWaterfall": "carry_nav_waterfall_list",
            "checkBox_CarryNavFits": "carry_nav_fits_information",
        }
        for attr, key in nav_keys.items():
            if hasattr(self.widget, attr):
                raw = self.settings.value(key, nav_defaults[attr])
                val = str(raw).lower() in ("1", "true", "yes") if isinstance(raw, str) else bool(raw)
                getattr(self.widget, attr).setChecked(val)

        for key, attr in self._plot_config_setting_bindings():
            if hasattr(self.widget, attr):
                self._load_widget_from_settings(
                    key, getattr(self.widget, attr))

    def _plot_config_setting_bindings(self):
        return [
            ("plot_cfg/night_view", "checkBox_NightView"),
            ("plot_cfg/night_cake", "checkBox_WhiteForPeak"),
            ("plot_cfg/show_large_pt", "checkBox_ShowLargePnT"),
            ("plot_cfg/title_filename_only", "checkBox_ShortPlotTitle"),
            ("plot_cfg/title_truncate_middle", "checkBox_TitleTruncateMiddle"),
            ("plot_cfg/title_font_size", "spinBox_TitleFontSize"),
            ("plot_cfg/title_max_length", "spinBox_TitleMaxLength"),
            ("plot_cfg/base_line_thickness", "comboBox_BasePtnLineThickness"),
            ("plot_cfg/background_line_thickness", "comboBox_BkgnLineThickness"),
            ("plot_cfg/waterfall_line_thickness", "comboBox_WaterfallLineThickness"),
            ("plot_cfg/vcursor_thickness", "comboBox_VertCursorThickness"),
            ("plot_cfg/fontsize_pt_label", "comboBox_PnTFontSize"),
            ("plot_cfg/fontsize_miller", "comboBox_HKLFontSize"),
            ("plot_cfg/fontsize_legend", "comboBox_LegendFontSize"),
            ("plot_cfg/fontsize_waterfall_label", "comboBox_WaterfallFontSize"),
            ("plot_cfg/jcpds_alpha_pattern", "doubleSpinBox_JCPDS_ptn_Alpha"),
            ("plot_cfg/jcpds_alpha_cake", "doubleSpinBox_JCPDS_cake_Alpha"),
            ("plot_cfg/jcpds_thickness_pattern", "comboBox_PtnJCPDSBarThickness"),
            ("plot_cfg/jcpds_thickness_cake", "comboBox_CakeJCPDSBarThickness"),
        ]

    def _save_widget_to_settings(self, key, widget):
        if isinstance(widget, QtWidgets.QCheckBox):
            self.settings.setValue(key, bool(widget.isChecked()))
            return
        if isinstance(widget, QtWidgets.QComboBox):
            self.settings.setValue(key, str(widget.currentText()))
            return
        if isinstance(widget, QtWidgets.QSpinBox):
            self.settings.setValue(key, int(widget.value()))
            return
        if isinstance(widget, QtWidgets.QDoubleSpinBox):
            self.settings.setValue(key, float(widget.value()))
            return

    def _load_widget_from_settings(self, key, widget):
        if isinstance(widget, QtWidgets.QCheckBox):
            raw = self.settings.value(key, widget.isChecked())
            val = str(raw).lower() in ("1", "true", "yes") if isinstance(raw, str) else bool(raw)
            widget.setChecked(val)
            return
        if isinstance(widget, QtWidgets.QComboBox):
            raw = str(self.settings.value(key, widget.currentText()))
            if widget.findText(raw) >= 0:
                widget.setCurrentText(raw)
            return
        if isinstance(widget, QtWidgets.QSpinBox):
            raw = self.settings.value(key, widget.value())
            try:
                widget.setValue(int(raw))
            except Exception:
                pass
            return
        if isinstance(widget, QtWidgets.QDoubleSpinBox):
            raw = self.settings.value(key, widget.value())
            try:
                widget.setValue(float(raw))
            except Exception:
                pass
            return

    def _capture_nav_carry_state(self):
        source_chi = None
        if self.model.base_ptn_exist():
            source_chi = os.path.basename(self.model.get_base_ptn_filename())
        cake_hist = {}
        if hasattr(self.widget, "cake_hist_widget"):
            hist = self.widget.cake_hist_widget
            cake_hist = {
                "log_y": bool(hist.check_log.isChecked()),
            }
        return {
            "source_chi": source_chi,
            "jcpds_lst": copy.deepcopy(self.model.jcpds_lst),
            "pressure": float(self.model.get_saved_pressure()),
            "temperature": float(self.model.get_saved_temperature()),
            "spectrum_smoothing": self.get_spectrum_smoothing_settings(),
            "cake_z_scale": {
                "vmin": float(self.widget.doubleSpinBox_CCDScaleMin.value())
                if hasattr(self.widget, "doubleSpinBox_CCDScaleMin") else 0.0,
                "vmax": float(self.widget.doubleSpinBox_CCDScaleMax.value())
                if hasattr(self.widget, "doubleSpinBox_CCDScaleMax") else 1.0,
                "hist": cake_hist,
            },
            "ccd_roi": {
                "row_min": int(self.widget.spinBox_CCDRowMin.value())
                if hasattr(self.widget, "spinBox_CCDRowMin") else 0,
                "row_max": int(self.widget.spinBox_CCDRowMax.value())
                if hasattr(self.widget, "spinBox_CCDRowMax") else 0,
            },
            "background": {
                "roi_min": float(self.widget.doubleSpinBox_Background_ROI_min.value()),
                "roi_max": float(self.widget.doubleSpinBox_Background_ROI_max.value()),
                "poly_order": int(self.widget.spinBox_BGParam1.value()),
                "areas": self.get_background_fit_areas(),
            },
            "waterfall_list": copy.deepcopy(self.model.waterfall_ptn),
            "fits_information": {
                "section_lst": copy.deepcopy(self.model.section_lst),
                "current_section": copy.deepcopy(self.model.current_section),
            },
        }

    def _should_carry_nav_category(self, key, checkbox_attr):
        presence = self.session_ctrl.get_last_param_category_presence()
        if not bool(presence.get(key, False)):
            # If target CHI has no existing info, always carry from current.
            return True
        if not hasattr(self.widget, checkbox_attr):
            return True
        return bool(getattr(self.widget, checkbox_attr).isChecked())

    def _apply_nav_carry_state(self, snap):
        carried_any = False
        if self._should_carry_nav_category("jcpds", "checkBox_CarryNavJCPDS"):
            self.model.jcpds_lst = copy.deepcopy(snap["jcpds_lst"])
            self.jcpdstable_ctrl.update()
            carried_any = True

        if self._should_carry_nav_category("pressure", "checkBox_CarryNavPressure"):
            self.model.save_pressure(float(snap["pressure"]))
            self.widget.doubleSpinBox_Pressure.setValue(float(snap["pressure"]))
            carried_any = True

        if self._should_carry_nav_category("spectrum_smoothing", "checkBox_CarryNavSpectrumSmoothing"):
            params = snap.get("spectrum_smoothing", {}) or {}
            if hasattr(self.widget, "spinBox_SpectrumDespike"):
                self.widget.spinBox_SpectrumDespike.setValue(
                    int(params.get("despike_kernel", 0)))
            if hasattr(self.widget, "spinBox_SpectrumSGWindow"):
                self.widget.spinBox_SpectrumSGWindow.setValue(
                    int(params.get("sg_window", 0)))
            if hasattr(self.widget, "spinBox_SpectrumSGPoly"):
                self.widget.spinBox_SpectrumSGPoly.setValue(
                    int(params.get("sg_polyorder", 3)))
            self.apply_spectrum_smoothing()
            carried_any = True

        if self._should_carry_nav_category("cake_z_scale", "checkBox_CarryNavCakeZScale"):
            cake = snap["cake_z_scale"]
            if hasattr(self.widget, "doubleSpinBox_CCDScaleMin"):
                self.widget.doubleSpinBox_CCDScaleMin.setValue(float(cake.get("vmin", 0.0)))
            if hasattr(self.widget, "doubleSpinBox_CCDScaleMax"):
                self.widget.doubleSpinBox_CCDScaleMax.setValue(float(cake.get("vmax", 1.0)))
            hist = cake.get("hist", {})
            if hasattr(self.widget, "cake_hist_widget") and hist != {}:
                self.widget.cake_hist_widget.check_log.setChecked(bool(hist.get("log_y", True)))
            carried_any = True

        if self._should_carry_nav_category("ccd_roi", "checkBox_CarryNavCCDRoi"):
            ccd_roi = snap.get("ccd_roi", {})
            if hasattr(self.widget, "spinBox_CCDRowMin") and hasattr(self.widget, "spinBox_CCDRowMax"):
                row_min = int(ccd_roi.get("row_min", self.widget.spinBox_CCDRowMin.value()))
                row_max = int(ccd_roi.get("row_max", self.widget.spinBox_CCDRowMax.value()))
                self.widget.spinBox_CCDRowMin.blockSignals(True)
                self.widget.spinBox_CCDRowMax.blockSignals(True)
                self.widget.spinBox_CCDRowMin.setValue(row_min)
                self.widget.spinBox_CCDRowMax.setValue(row_max)
                self.widget.spinBox_CCDRowMin.blockSignals(False)
                self.widget.spinBox_CCDRowMax.blockSignals(False)
                try:
                    self.base_ptn_ctrl.ccd_ctrl._apply_row_roi_to_spectrum()
                except Exception:
                    pass
            carried_any = True

        if self._should_carry_nav_category("background", "checkBox_CarryNavBackground"):
            bg = snap["background"]
            self.widget.doubleSpinBox_Background_ROI_min.setValue(float(bg["roi_min"]))
            self.widget.doubleSpinBox_Background_ROI_max.setValue(float(bg["roi_max"]))
            self.widget.spinBox_BGParam1.setValue(int(bg.get("poly_order", 3)))
            self.set_background_fit_areas(bg.get("areas", []))
            if self.model.base_ptn_exist():
                self.update_bgsub()
            carried_any = True

        if self._should_carry_nav_category("waterfall_list", "checkBox_CarryNavWaterfall"):
            self.model.waterfall_ptn = copy.deepcopy(snap["waterfall_list"])
            self.waterfalltable_ctrl.update()
            carried_any = True

        if self._should_carry_nav_category("fits_information", "checkBox_CarryNavFits"):
            self.model.section_lst = copy.deepcopy(snap["fits_information"]["section_lst"])
            self.model.current_section = copy.deepcopy(snap["fits_information"]["current_section"])
            self.peakfit_table_ctrl.update_sections()
            self.peakfit_table_ctrl.update_peak_parameters()
            self.peakfit_table_ctrl.update_baseline_constraints()
            self.peakfit_table_ctrl.update_peak_constraints()
            carried_any = True

        if carried_any:
            self.session_ctrl.set_carryover_source_chi(snap.get("source_chi"))
        else:
            self.session_ctrl.set_carryover_source_chi(None)

        # Never carry over backup information across CHI navigation.
        # Always show backup info for the newly loaded file.
        self.session_ctrl.refresh_backup_table()

    """
    def closeEvent(self, event):
        self.write_setting()
        self.widget.deleteLater()
        gc.collect()
        self.deleteLater()
        event.accept()
    """

    def on_key_press(self, event):
        from matplotlib.backend_bases import key_press_handler
        
        if event.key == 'i':
            if self.widget.mpl.ntb._active == 'PAN':
                self.widget.mpl.ntb.pan()
            if self.widget.mpl.ntb._active == 'ZOOM':
                self.widget.mpl.ntb.zoom()
        elif event.key == 's':
            self.session_ctrl.save_dpp_ppss()
        elif event.key == 'w':
            self.plot_new_graph()
        elif event.key == 'v':
            lims = self.widget.mpl.canvas.ax_pattern.axis()
            if self.widget.checkBox_BgSub.isChecked():
                x, y = self.model.base_ptn.get_bgsub()
            else:
                x, y = self.model.base_ptn.get_raw()
            xroi, yroi = get_data_section(x, y, [lims[0], lims[1]])
            self.plot_ctrl.update([lims[0], lims[1], yroi.min(), yroi.max()])
        else:
            key_press_handler(event, self.widget.mpl.canvas,
                              self.widget.mpl.ntb)
    """
    def to_PkFt(self):
        # listen
        if not self.model.base_ptn_exist():
            return
        lims = self.widget.mpl.canvas.ax_pattern.axis()
        talk = "PeakPo,{0},{1: .2f},{2: .2f},{3: .2f},{4: .2f}".format(
            self.model.base_ptn.fname, lims[0], lims[1], lims[2], lims[3])
        self.clip.setText(talk)

    def from_PkFt(self):
        l = self.clip.text()
        listen = str(l)
        if listen.find("PeakFt") == -1:
            return
        a = listen.split(',')
        new_filen = a[1]
        new_lims = [float(i) for i in a[2:6]]
        self.base_ptn_ctrl._load_a_new_pattern(new_filen)
        self.plot_ctrl.update(new_lims)
    """

    def set_nightday_view(self):
        self.plot_ctrl._set_nightday_view()
        self.waterfalltable_ctrl.update()
        self.plot_ctrl.update()

    def deliver_mouse_signal(self, event):
        # Map ROI selection uses mouse drag on the main plot/cake axes.
        # Suppress default click handling (position popup / peak pick)
        # while ROI selector is active.
        if hasattr(self, "map_ctrl") and (self.map_ctrl is not None):
            try:
                if self.map_ctrl.is_roi_selection_active():
                    return
            except Exception:
                pass
        if hasattr(self, "seq_ctrl") and (self.seq_ctrl is not None):
            try:
                if self.seq_ctrl.is_roi_selection_active():
                    return
            except Exception:
                pass
        # ✅ Compatible with matplotlib 3.3+
        if hasattr(self.widget.mpl.ntb, 'mode'):
            # New matplotlib API
            if self.widget.mpl.ntb.mode != '':
                return
        elif hasattr(self.widget.mpl.ntb, '_active'):
            # Old matplotlib API
            if self.widget.mpl.ntb._active is not None:
                return
        if (event.xdata is None) or (event.ydata is None):
            return
        # Peak add/remove must come from the main 1D pattern axes.
        if event.inaxes != self.widget.mpl.canvas.ax_pattern:
            return
        if (event.button != 1) and (event.button != 3):
            return
        if event.button == 1:
            mouse_button = 'left'
        elif event.button == 3:
            mouse_button = 'right'
        fits_active = False
        if hasattr(self.widget, "tab_PkFt"):
            fits_active = (self.widget.tabWidget.currentWidget() == self.widget.tab_PkFt)
        else:
            # Backward-compatible fallback for older/newer tab orders.
            fits_active = (self.widget.tabWidget.currentIndex() in (4, 5))

        if fits_active and \
                (self.widget.pushButton_AddRemoveFromMouse.isChecked()):
            if not self.model.current_section_exist():
                QtWidgets.QMessageBox.warning(
                    self.widget, "Warning", "Set section first.")
                return
            """ lines below causes issues
            if self.model.current_section.fitted():
                reply = QtWidgets.QMessageBox.question(
                    self.widget, 'Message',
                    'Do you want to add to the last fitting result without save?',
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.Yes)
                if reply == QtWidgets.QMessageBox.No:
                    return
                else:
                    self.model.current_section.invalidate_fit_result()
            """
            if self.model.current_section.fitted():
                    self.model.current_section.invalidate_fit_result()
            self.pick_peak(mouse_button, event.xdata, event.ydata)
        else:
            return

    def pick_peak(self, mouse_button, xdata, ydata):
        """
        """
        if mouse_button == 'left':  # left click
            if (self.model.current_section is None) or \
                    (self.model.current_section.x is None) or \
                    (len(self.model.current_section.x) == 0):
                QtWidgets.QMessageBox.warning(
                    self.widget, "Warning", "Set section first.")
                return
            # Robust against tiny range mismatches: map click to nearest x
            # sample in the current section and use that as initial center.
            x_arr = np.asarray(self.model.current_section.x, dtype=float)
            idx = int(np.abs(x_arr - float(xdata)).argmin())
            x_center = float(x_arr[idx])
            success = self.model.current_section.set_single_peak(
                x_center,
                self.widget.doubleSpinBox_InitialFWHM.value())
            if not success:
                QtWidgets.QMessageBox.warning(
                    self.widget, "Warning",
                    "You picked outside of the current section.")
                return
        elif mouse_button == 'right':  # right button for removal
            if not self.model.current_section.peaks_exist():
                return
            self.model.current_section.remove_single_peak_nearby(xdata)
        else:
            return
        self.peakfit_ctrl.set_tableWidget_PkParams_unsaved()
        self.peakfit_table_ctrl.update_peak_parameters()
        self.peakfit_table_ctrl.update_peak_constraints()
        self.plot_ctrl.update()

    def setXEat(self, wavelength):
        self.widget.doubleSpinBox_SetWavelength.setValue(wavelength)
        self.apply_wavelength()

    def apply_wavelength(self):
        if self.model.base_ptn_exist():
            self.model.set_base_ptn_wavelength(
                self.widget.doubleSpinBox_SetWavelength.value())
            self.sync_background_roi_spinboxes(force_full_range=True)
        self.widget.label_XRayEnergy.setText("nm")
        if self._plot_update_deferred():
            return
        if self.model.base_ptn_exist():
            self.update_bgsub()
        self.plot_ctrl.update()

    def get_spectrum_smoothing_settings(self):
        if not hasattr(self.widget, "spinBox_SpectrumDespike"):
            return {
                "despike_kernel": 0,
                "sg_window": 0,
                "sg_polyorder": 3,
            }
        despike = int(self.widget.spinBox_SpectrumDespike.value())
        sg_window = int(self.widget.spinBox_SpectrumSGWindow.value()) \
            if hasattr(self.widget, "spinBox_SpectrumSGWindow") else 0
        sg_poly = int(self.widget.spinBox_SpectrumSGPoly.value()) \
            if hasattr(self.widget, "spinBox_SpectrumSGPoly") else 3
        if despike > 0 and despike % 2 == 0:
            despike += 1
        if sg_window > 0 and sg_window % 2 == 0:
            sg_window += 1
        if sg_window <= 1:
            sg_window = 0
        sg_poly = max(0, sg_poly)
        if sg_window > 0:
            sg_poly = min(sg_poly, sg_window - 1)
        return {
            "despike_kernel": despike,
            "sg_window": sg_window,
            "sg_polyorder": sg_poly,
        }

    def apply_spectrum_smoothing(self):
        params = self.get_spectrum_smoothing_settings()
        if hasattr(self.widget, "spinBox_SpectrumDespike"):
            desired = int(params["despike_kernel"])
            if self.widget.spinBox_SpectrumDespike.value() != desired:
                self.widget.spinBox_SpectrumDespike.blockSignals(True)
                self.widget.spinBox_SpectrumDespike.setValue(desired)
                self.widget.spinBox_SpectrumDespike.blockSignals(False)
        if hasattr(self.widget, "spinBox_SpectrumSGWindow"):
            desired = int(params["sg_window"])
            if self.widget.spinBox_SpectrumSGWindow.value() != desired:
                self.widget.spinBox_SpectrumSGWindow.blockSignals(True)
                self.widget.spinBox_SpectrumSGWindow.setValue(desired)
                self.widget.spinBox_SpectrumSGWindow.blockSignals(False)
        if hasattr(self.widget, "spinBox_SpectrumSGPoly"):
            desired = int(params["sg_polyorder"])
            if self.widget.spinBox_SpectrumSGPoly.value() != desired:
                self.widget.spinBox_SpectrumSGPoly.blockSignals(True)
                self.widget.spinBox_SpectrumSGPoly.setValue(desired)
                self.widget.spinBox_SpectrumSGPoly.blockSignals(False)
        if self._plot_update_deferred():
            return
        self.plot_ctrl.update()

    def force_update_spectrum_process(self):
        fw = QtWidgets.QApplication.focusWidget()
        if isinstance(fw, QtWidgets.QAbstractSpinBox):
            try:
                fw.interpretText()
            except Exception:
                pass
        if fw is not None:
            try:
                fw.clearFocus()
            except Exception:
                pass
        QtWidgets.QApplication.processEvents()
        self.apply_spectrum_smoothing()
        self.update_bgsub()

    def reset_spectrum_smoothing_to_raw(self):
        if hasattr(self.widget, "spinBox_SpectrumDespike"):
            self.widget.spinBox_SpectrumDespike.setValue(0)
        if hasattr(self.widget, "spinBox_SpectrumSGWindow"):
            self.widget.spinBox_SpectrumSGWindow.setValue(0)
        if hasattr(self.widget, "spinBox_SpectrumSGPoly"):
            self.widget.spinBox_SpectrumSGPoly.setValue(3)
        self.force_update_spectrum_process()

    def update_bgsub(self):
        '''
        this is only to read the current inputs and replot
        '''
        if not self.model.base_ptn_exist():
            QtWidgets.QMessageBox.warning(self.widget, "Warning",
                                          "Load a base pattern first.")
            return
        x_raw_base = getattr(self.model.base_ptn, "x_raw", None)
        y_raw_base = getattr(self.model.base_ptn, "y_raw", None)
        if (x_raw_base is None) or (y_raw_base is None) or \
                (len(x_raw_base) == 0) or (len(y_raw_base) == 0):
            QtWidgets.QMessageBox.warning(
                self.widget, "Warning",
                "Base pattern has no raw data for background fitting.")
            return
        self.sync_background_roi_spinboxes(force_full_range=False)
        """receive new bg parameters and update the graph"""
        bg_params = [self.widget.spinBox_BGParam1.value()]
        bg_roi = [self.widget.doubleSpinBox_Background_ROI_min.value(),
                  self.widget.doubleSpinBox_Background_ROI_max.value()]
        if (bg_roi[0] <= x_raw_base.min()):
            bg_roi[0] = x_raw_base.min()
            self.widget.doubleSpinBox_Background_ROI_min.setValue(bg_roi[0])
        if (bg_roi[1] >= x_raw_base.max()):
            bg_roi[1] = x_raw_base.max()
            self.widget.doubleSpinBox_Background_ROI_max.setValue(bg_roi[1])
        bg_fit_areas = self.get_background_fit_areas()
        __, y_fit_base = self._get_background_fit_source_xy(
            x_raw_base, y_raw_base)
        self.model.base_ptn.subtract_bg(
            bg_roi, bg_params, yshift=0,
            fit_areas=bg_fit_areas, y_source=y_fit_base)
        temp_dir = get_temp_dir(self.model.get_base_ptn_filename())
        self.model.base_ptn.write_temporary_bgfiles(temp_dir=temp_dir)
        if self.model.waterfall_exist():
            print(str(datetime.datetime.now())[:-7], 
                ": BGfit and BGsub for waterfall patterns even if they are displayed.\n",
                "Yes this is a bit of waste.  Future fix needed.")
            n_skipped = 0
            for pattern in self.model.waterfall_ptn:
                x_raw = getattr(pattern, "x_raw", None)
                y_raw = getattr(pattern, "y_raw", None)
                if (x_raw is None) or (y_raw is None) or \
                        (len(x_raw) == 0) or (len(y_raw) == 0):
                    n_skipped += 1
                    continue
                __, y_fit_pattern = self._get_background_fit_source_xy(
                    x_raw, y_raw)
                pattern.subtract_bg(
                    bg_roi, bg_params, yshift=0,
                    fit_areas=bg_fit_areas, y_source=y_fit_pattern)
            if n_skipped > 0:
                print(str(datetime.datetime.now())[:-7],
                    ": Skipped BG subtraction for {0:d} waterfall item(s) "
                    "without raw data.".format(n_skipped))
        if self._plot_update_deferred():
            return
        self.plot_new_graph()

    def sync_background_roi_spinboxes(self, force_full_range=False):
        if not self.model.base_ptn_exist():
            return
        x_raw = getattr(self.model.base_ptn, "x_raw", None)
        if x_raw is None or len(x_raw) == 0:
            return
        x_arr = np.asarray(x_raw, dtype=float)
        finite = x_arr[np.isfinite(x_arr)]
        if finite.size == 0:
            return
        x_min = float(np.nanmin(finite))
        x_max = float(np.nanmax(finite))
        if x_max < x_min:
            x_min, x_max = x_max, x_min
        for box in (
            self.widget.doubleSpinBox_Background_ROI_min,
            self.widget.doubleSpinBox_Background_ROI_max,
        ):
            box.blockSignals(True)
            box.setMinimum(x_min)
            box.setMaximum(x_max)
            box.setDecimals(3)
            box.blockSignals(False)
        cur_min = float(self.widget.doubleSpinBox_Background_ROI_min.value())
        cur_max = float(self.widget.doubleSpinBox_Background_ROI_max.value())
        if force_full_range or (cur_min < x_min) or (cur_min > x_max):
            cur_min = x_min
        if force_full_range or (cur_max < x_min) or (cur_max > x_max):
            cur_max = x_max
        if cur_max < cur_min:
            cur_min, cur_max = x_min, x_max
        self.widget.doubleSpinBox_Background_ROI_min.blockSignals(True)
        self.widget.doubleSpinBox_Background_ROI_max.blockSignals(True)
        self.widget.doubleSpinBox_Background_ROI_min.setValue(cur_min)
        self.widget.doubleSpinBox_Background_ROI_max.setValue(cur_max)
        self.widget.doubleSpinBox_Background_ROI_min.blockSignals(False)
        self.widget.doubleSpinBox_Background_ROI_max.blockSignals(False)

    def _get_background_fit_source_xy(self, x_data, y_data):
        x_arr = np.asarray(x_data, dtype=float)
        y_arr = np.asarray(y_data, dtype=float)
        if (self.plot_ctrl is not None) and \
                bool(getattr(self.plot_ctrl, "_smoothing_active", lambda: False)()):
            try:
                return self.plot_ctrl._get_smoothed_pattern_xy(x_arr, y_arr)
            except Exception:
                pass
        return x_arr, y_arr

    def _get_explicit_background_fit_areas(self):
        areas = []
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
                areas.append([xmin, xmax])
        return areas

    def get_background_fit_areas(self):
        areas = self._get_explicit_background_fit_areas()
        if areas:
            return areas
        return [[
            float(self.widget.doubleSpinBox_Background_ROI_min.value()),
            float(self.widget.doubleSpinBox_Background_ROI_max.value()),
        ]]

    def set_background_fit_areas(self, areas):
        table = getattr(self.widget, "tableWidget_BackgroundConstraints", None)
        if table is None:
            return
        table.setRowCount(0)
        normalized = []
        for area in (areas or []):
            try:
                xmin = float(area[0])
                xmax = float(area[1])
            except Exception:
                continue
            if xmax < xmin:
                xmin, xmax = xmax, xmin
            normalized.append([xmin, xmax])
        normalized.sort(key=lambda area: area[0])
        for xmin, xmax in normalized:
            row = table.rowCount()
            table.insertRow(row)
            table.setItem(row, 0, QtWidgets.QTableWidgetItem(f"{xmin:.3f}"))
            table.setItem(row, 1, QtWidgets.QTableWidgetItem(f"{xmax:.3f}"))

    def append_background_fit_area(self, xmin, xmax):
        table = getattr(self.widget, "tableWidget_BackgroundConstraints", None)
        if table is None:
            return
        xmin = float(xmin)
        xmax = float(xmax)
        if xmax < xmin:
            xmin, xmax = xmax, xmin
        if abs(xmax - xmin) < 1.0e-9:
            return
        for area in self._get_explicit_background_fit_areas():
            if abs(area[0] - xmin) < 1.0e-6 and abs(area[1] - xmax) < 1.0e-6:
                return
        row = table.rowCount()
        table.insertRow(row)
        item_min = QtWidgets.QTableWidgetItem(f"{xmin:.3f}")
        item_max = QtWidgets.QTableWidgetItem(f"{xmax:.3f}")
        table.setItem(row, 0, item_min)
        table.setItem(row, 1, item_max)
        table.sortItems(0)

    def add_background_area_from_inputs(self):
        xmin = float(self.widget.doubleSpinBox_Background_ROI_min.value())
        xmax = float(self.widget.doubleSpinBox_Background_ROI_max.value())
        self.append_background_fit_area(xmin, xmax)
        self.plot_ctrl.update()

    def toggle_background_area_selector(self, checked):
        if checked:
            self.activate_background_area_selector()
        else:
            self.deactivate_background_area_selector()

    def activate_background_area_selector(self):
        if not self.model.base_ptn_exist():
            if hasattr(self.widget, "pushButton_BGAreaAdd"):
                self.widget.pushButton_BGAreaAdd.blockSignals(True)
                self.widget.pushButton_BGAreaAdd.setChecked(False)
                self.widget.pushButton_BGAreaAdd.blockSignals(False)
            return
        ax = getattr(self.widget.mpl.canvas, "ax_pattern", None)
        if ax is None:
            return
        self.deactivate_background_area_selector()
        toolbar = getattr(self.widget.mpl.canvas, "toolbar", None)
        if toolbar is not None:
            try:
                if getattr(toolbar, "mode", "") == 'zoom rect':
                    toolbar.zoom()
                elif getattr(toolbar, "mode", "") == 'pan/zoom':
                    toolbar.pan()
            except Exception:
                pass
        self._bg_area_selector = RectangleSelector(
            ax,
            self._on_background_area_selected,
            useblit=True,
            button=[1],
            interactive=False,
            drag_from_anywhere=False,
            spancoords="data",
            minspanx=1.0e-6,
        )
        try:
            self.widget.mpl.canvas.setCursor(QtCore.Qt.CrossCursor)
        except Exception:
            pass
        self.plot_ctrl.update()

    def deactivate_background_area_selector(self):
        if self._bg_area_selector is not None:
            try:
                self._bg_area_selector.set_active(False)
            except Exception:
                pass
            self._bg_area_selector = None
        try:
            self.widget.mpl.canvas.unsetCursor()
        except Exception:
            pass
        if hasattr(self.widget, "pushButton_BGAreaAdd"):
            self.widget.pushButton_BGAreaAdd.blockSignals(True)
            self.widget.pushButton_BGAreaAdd.setChecked(False)
            self.widget.pushButton_BGAreaAdd.blockSignals(False)

    def _on_background_area_selected(self, eclick, erelease):
        ax = getattr(self.widget.mpl.canvas, "ax_pattern", None)
        if ax is None:
            self.deactivate_background_area_selector()
            return
        try:
            x0, __ = ax.transData.inverted().transform((float(eclick.x), float(eclick.y)))
            x1, __ = ax.transData.inverted().transform((float(erelease.x), float(erelease.y)))
        except Exception:
            x0 = eclick.xdata
            x1 = erelease.xdata
        if (x0 is None) or (x1 is None):
            self.deactivate_background_area_selector()
            return
        xmin = min(float(x0), float(x1))
        xmax = max(float(x0), float(x1))
        if xmax <= xmin:
            self.deactivate_background_area_selector()
            return
        self.widget.doubleSpinBox_Background_ROI_min.setValue(xmin)
        self.widget.doubleSpinBox_Background_ROI_max.setValue(xmax)
        self.append_background_fit_area(xmin, xmax)
        self.plot_ctrl.update()
        self.deactivate_background_area_selector()

    def remove_selected_background_area(self):
        table = getattr(self.widget, "tableWidget_BackgroundConstraints", None)
        if table is None:
            return
        row = table.currentRow()
        if row < 0:
            return
        table.removeRow(row)
        self.plot_ctrl.update()

    def clear_background_areas(self):
        table = getattr(self.widget, "tableWidget_BackgroundConstraints", None)
        if table is None:
            return
        table.setRowCount(0)
        self.plot_ctrl.update()

    def apply_pt_to_graph(self):
        """
        if self.model.jcpds_exist():
            self.plot_ctrl.update_jcpds_only()
        else:
        """
        if self._plot_update_deferred():
            return
        self.plot_ctrl.update()

    def _plot_update_deferred(self):
        return self._defer_plot_update_count > 0

    @contextmanager
    def _defer_plot_updates(self):
        self._defer_plot_update_count += 1
        try:
            yield
        finally:
            self._defer_plot_update_count -= 1

    def _find_closestjcpds(self, x):
        jcount = 0
        for phase in self.model.jcpds_lst:
            if phase.display:
                jcount += 1
        if jcount == 0:
            return ''
        if jcount != 0:
            idx_j = []
            diff_j = []
            tth_j = []
            h_j = []
            k_j = []
            l_j = []
            names_j = []
            dsp_j = []
            int_j = []
            for j in self.model.jcpds_lst:
                if j.display:
                    i, d, t = j.find_DiffLine(
                        x, self.widget.doubleSpinBox_SetWavelength.value())
                    idx_j.append(i)
                    diff_j.append(d)
                    tth_j.append(t)
                    h_j.append(j.DiffLines[i].h)
                    k_j.append(j.DiffLines[i].k)
                    l_j.append(j.DiffLines[i].l)
                    dsp_j.append(j.DiffLines[i].dsp)
                    int_j.append(j.DiffLines[i].intensity)
                    names_j.append(j.name)
        idx_min = diff_j.index(min(diff_j))
        tth_min = tth_j[idx_min]
        dsp_min = dsp_j[idx_min]
        int_min = int_j[idx_min]
        h_min = h_j[idx_min]
        k_min = k_j[idx_min]
        l_min = l_j[idx_min]
        name_min = names_j[idx_min]
        line1 = '2\u03B8 = {0:.4f} \u00B0, d-sp = {1:.4f} \u212B'.format(
            float(tth_min), float(dsp_min))
        line2 = 'intensity = {0: .0f}, hkl = {1: .0f} {2: .0f} {3: .0f}'.\
            format(int(int_min), int(h_min), int(k_min), int(l_min))
        textoutput = name_min + '\n' + line1 + '\n' + line2
        return textoutput

    def goto_next_file(self, move):
        """
        quick move to the next base pattern file
        """
        if not self.model.base_ptn_exist():
            QtWidgets.QMessageBox.warning(
                self.widget, "Warning", "Choose a base pattern first.")
            return
        use_dpp_nav = bool(
            hasattr(self.widget, "checkBox_NavDPP") and
            self.widget.checkBox_NavDPP.isChecked())
        if use_dpp_nav:
            self._goto_dpp_next_file(move)
        else:
            self._goto_chi_next_file(move)
        return

    def _goto_chi_next_file(self, move):
        nav_state = self._capture_nav_carry_state()
        filelist_chi = self._get_spectrum_filelist()
        idx_chi = self._find_current_spectrum_index(filelist_chi)

        if idx_chi == -1:
            QtWidgets.QMessageBox.warning(
                self.widget, "Warning", "Cannot find current spectrum file")
            return  # added newly

        step = self.widget.spinBox_FileStep.value()
        if move == 'next':
            idx_chi_new = idx_chi + step
        elif move == 'previous':
            idx_chi_new = idx_chi - step
        elif move == 'last':
            idx_chi_new = filelist_chi.__len__() - 1
            if idx_chi == idx_chi_new:
                QtWidgets.QMessageBox.warning(
                    self.widget, "Warning", "It is already the last file.")
                return
        elif move == 'first':
            idx_chi_new = 0
            if idx_chi == idx_chi_new:
                QtWidgets.QMessageBox.warning(
                    self.widget, "Warning", "It is already the first file.")
                return

        if idx_chi_new > filelist_chi.__len__() - 1:
            idx_chi_new = filelist_chi.__len__() - 1
            if idx_chi == idx_chi_new:
                QtWidgets.QMessageBox.warning(
                    self.widget, "Warning", "It is already the last file.")
                return
        if idx_chi_new < 0:
            idx_chi_new = 0
            if idx_chi == idx_chi_new:
                QtWidgets.QMessageBox.warning(
                    self.widget, "Warning", "It is already the first file.")
                return
        new_filename_chi = filelist_chi[idx_chi_new]
        if os.path.exists(new_filename_chi):
            with self._defer_plot_updates():
                self.base_ptn_ctrl._load_a_new_pattern(new_filename_chi)
                self._apply_nav_carry_state(nav_state)
            # self.model.set_base_ptn_color(self.obj_color)
            self.plot_ctrl.update()
        else:
            QtWidgets.QMessageBox.warning(self.widget, "Warning",
                                          new_filename_chi +
                                          " does not exist.")

    def _goto_dpp_next_file(self, move):
        def _session_manifest_for(data_file):
            param_dir = get_temp_dir(data_file, branch='-rampo')
            return os.path.join(param_dir, "rampo_manifest.json")

        filelist_chi = self._get_spectrum_filelist()
        filelist_session = [
            filen for filen in filelist_chi
            if os.path.exists(_session_manifest_for(filen))
        ]

        idx_chi = self._find_current_spectrum_index(filelist_chi)
        idx_session = self._find_current_spectrum_index(filelist_session)

        if idx_chi == -1:
            QtWidgets.QMessageBox.warning(
                self.widget, "Warning", "Cannot find current spectrum file")
            return  # added newly

        if idx_session == -1:
            QtWidgets.QMessageBox.warning(
                self.widget, "Warning",
                "Cannot find a saved session for the current spectrum.\n"
                "Save one first.")
            return  # added newly

        step = self.widget.spinBox_FileStep.value()
        if move == 'next':
            idx_chi_new = idx_chi + step
        elif move == 'previous':
            idx_chi_new = idx_chi - step
        elif move == 'last':
            idx_chi_new = filelist_chi.__len__() - 1
            if idx_chi == idx_chi_new:
                QtWidgets.QMessageBox.warning(
                    self.widget, "Warning", "It is already the last file.")
                return
        elif move == 'first':
            idx_chi_new = 0
            if idx_chi == idx_chi_new:
                QtWidgets.QMessageBox.warning(
                    self.widget, "Warning", "It is already the first file.")
                return
        if idx_chi_new > filelist_chi.__len__() - 1:
            idx_chi_new = filelist_chi.__len__() - 1
            if idx_chi == idx_chi_new:
                QtWidgets.QMessageBox.warning(
                    self.widget, "Warning", "It is already the last file.")
                return
        if idx_chi_new < 0:
            idx_chi_new = 0
            if idx_chi == idx_chi_new:
                QtWidgets.QMessageBox.warning(
                    self.widget, "Warning", "It is already the first file.")
                return

        auto_save_move = True
        if hasattr(self.widget, "checkBox_SaveDPPMove"):
            auto_save_move = bool(self.widget.checkBox_SaveDPPMove.isChecked())
        if auto_save_move:
            self.session_ctrl.save_dpp(quiet=True)
        else:
            reply = QtWidgets.QMessageBox.question(
                self.widget, 'Message',
                'Do you want to save this session before moving to the next file?',
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes)
            if reply == QtWidgets.QMessageBox.Yes:
                self.session_ctrl.save_dpp()

        new_filename_chi = filelist_chi[idx_chi_new]
        new_manifest = _session_manifest_for(new_filename_chi)
        idx = self._find_file_index(filelist_session, new_filename_chi)

        if idx == -1:
            auto_gen_dpp = False
            if hasattr(self.widget, "checkBox_AutoGenDPP"):
                auto_gen_dpp = bool(self.widget.checkBox_AutoGenDPP.isChecked())
            if auto_gen_dpp:
                self.base_ptn_ctrl._load_a_new_pattern(new_filename_chi)
                self.session_ctrl.save_dpp(quiet=True)
                self.model.clear_section_list()
                self.plot_ctrl.update()
            else:
                QtWidgets.QMessageBox.warning(
                    self.widget, "Warning",
                    "Cannot find a saved session for the target spectrum.\n"
                    "Enable auto-create with move or save one manually first.")
                return
        else:
            auto_gen_dpp = False
            if hasattr(self.widget, "checkBox_AutoGenDPP"):
                auto_gen_dpp = bool(self.widget.checkBox_AutoGenDPP.isChecked())
            auto_gen_only_missing = True
            if hasattr(self.widget, "checkBox_AutogenMissing"):
                auto_gen_only_missing = bool(self.widget.checkBox_AutogenMissing.isChecked())
            if auto_gen_dpp and (not auto_gen_only_missing):
                reply = QtWidgets.QMessageBox.question(
                    self.widget, 'Message',
                    "The next spectrum already has a saved session.\n"
                    "Choose Yes to overwrite it from the current spectrum.\n"
                    "Choose No to keep and open the existing saved session.",
                    QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                    QtWidgets.QMessageBox.No)
                if reply == QtWidgets.QMessageBox.Yes:
                    self.base_ptn_ctrl._load_a_new_pattern(new_filename_chi)
                    self.session_ctrl.save_dpp(quiet=True)
                    self.model.clear_section_list()
                    self.plot_ctrl.update()
                else:
                    success = self.session_ctrl._load_dpp(new_manifest)
                    if success:
                        if self.model.exist_in_waterfall(
                            self.model.base_ptn.fname):
                            self.widget.pushButton_AddBasePtn.setChecked(True)
                        else:
                            self.widget.pushButton_AddBasePtn.setChecked(False)
                        if self.widget.checkBox_ShowCake.isChecked():
                            self.session_ctrl._load_cake_format_file()
                        self.plot_ctrl.update()
                    else:
                        QtWidgets.QMessageBox.warning(
                            self.widget, "Warning",
                            "Session loading was not successful.")
                        return
            else:
                success = self.session_ctrl._load_dpp(new_manifest)
                if success:
                    if self.model.exist_in_waterfall(
                        self.model.base_ptn.fname):
                        self.widget.pushButton_AddBasePtn.setChecked(True)
                    else:
                        self.widget.pushButton_AddBasePtn.setChecked(False)
                    if self.widget.checkBox_ShowCake.isChecked():
                        self.session_ctrl._load_cake_format_file()
                    self.plot_ctrl.update()
                else:
                    QtWidgets.QMessageBox.warning(
                        self.widget, "Warning",
                        "Session loading was not successful.")
                    return
        self.jcpdstable_ctrl.update()
        self.peakfit_table_ctrl.update_sections()
        self.peakfit_table_ctrl.update_peak_parameters()
        self.peakfit_table_ctrl.update_baseline_constraints()
        self.peakfit_table_ctrl.update_peak_constraints()
        return

    def _get_spectrum_filelist(self):
        sorted_by_name = self.widget.radioButton_SortbyNme.isChecked()
        patterns = ("*.spe", "*.SPE", "*.chi", "*.CHI")
        seen = set()
        filelist = []
        for pattern in patterns:
            for filen in get_sorted_filelist(
                    self.model.chi_path,
                    sorted_by_name=sorted_by_name,
                    search_ext=pattern):
                if filen not in seen:
                    seen.add(filen)
                    filelist.append(filen)
        if sorted_by_name:
            filelist = sorted(
                filelist,
                key=lambda filen: os.path.basename(filen).lower())
        else:
            filelist = sorted(filelist, key=os.path.getmtime)
        return filelist

    def _find_current_spectrum_index(self, filelist):
        if (not self.model.base_ptn_exist()) or (filelist == []):
            return -1
        current = os.path.normcase(os.path.abspath(self.model.base_ptn.fname))
        return self._find_file_index(filelist, current)

    def _find_file_index(self, filelist, target_file):
        current = os.path.normcase(os.path.abspath(target_file))
        for i, filen in enumerate(filelist):
            if os.path.normcase(os.path.abspath(filen)) == current:
                return i
        current_name = os.path.basename(target_file)
        for i, filen in enumerate(filelist):
            if os.path.basename(filen) == current_name:
                return i
        return -1

        # QtWidgets.QMessageBox.warning(self.widget, "Warning",
        #                              new_filename_chi + " does not exist.")
