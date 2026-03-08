import os
import time
import datetime
import numpy as np
import numpy.ma as ma
try:
    from scipy.signal import medfilt, savgol_filter
except Exception:
    medfilt = None
    savgol_filter = None
from matplotlib.widgets import MultiCursor
#from matplotlib.widgets import MultiCursor
#import matplotlib.transforms as transforms
#import matplotlib.colors as colors
#import matplotlib.patches as patches
#from matplotlib.textpath import TextPath
#import matplotlib.pyplot as plt
from qtpy import QtWidgets
from qtpy import QtCore
class MplController(object):

    def __init__(self, model, widget):
        self.model = model
        self.widget = widget
        self.obj_color = 'k'
        self.diff_ctrl = None
        self._cached_title = None
        self._cached_filename = None
        self._is_drawing = False
        self._toolbar_active = False
        self._update_delay_ms = 25
        self._pending_update_args = None
        self._update_timer = QtCore.QTimer(self.widget)
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self._flush_update_request)
        
        # ✅ Wrap toolbar methods to track state
        toolbar = self.widget.mpl.canvas.toolbar
        if toolbar:
            self._original_zoom = toolbar.zoom
            self._original_pan = toolbar.pan
            self._original_home = toolbar.home
            self._original_back = toolbar.back
            self._original_forward = toolbar.forward
            
            def zoom_wrapper(*args, **kwargs):
                result = self._original_zoom(*args, **kwargs)
                self._toolbar_active = (toolbar.mode != '')
                # ✅ NEW: Uncheck cursor when zoom is activated
                if toolbar.mode == 'zoom rect':
                    self.widget.checkBox_LongCursor.setChecked(False)
                return result
            
            def pan_wrapper(*args, **kwargs):
                result = self._original_pan(*args, **kwargs)
                self._toolbar_active = (toolbar.mode != '')
                # ✅ NEW: Uncheck cursor when pan is activated
                if toolbar.mode == 'pan/zoom':
                    self.widget.checkBox_LongCursor.setChecked(False)
                return result
            
            def home_wrapper(*args, **kwargs):
                self._toolbar_active = True
                result = self._original_home(*args, **kwargs)
                self._toolbar_active = False
                return result
            
            def back_wrapper(*args, **kwargs):
                self._toolbar_active = True
                result = self._original_back(*args, **kwargs)
                self._toolbar_active = False
                return result
            
            def forward_wrapper(*args, **kwargs):
                self._toolbar_active = True
                result = self._original_forward(*args, **kwargs)
                self._toolbar_active = False
                return result
            
            toolbar.zoom = zoom_wrapper
            toolbar.pan = pan_wrapper
            toolbar.home = home_wrapper
            toolbar.back = back_wrapper
            toolbar.forward = forward_wrapper

    def set_diff_controller(self, diff_ctrl):
        self.diff_ctrl = diff_ctrl

    def _is_spe_mode(self):
        if not self.model.base_ptn_exist():
            return False
        return str(getattr(self.model.base_ptn, "fname", "")).lower().endswith(".spe")

    def _set_nightday_view(self):
        if not self.widget.checkBox_NightView.isChecked():
            self.widget.mpl.canvas.set_toNight(False)
            # reset plot objects with white
            if self.model.base_ptn_exist():
                self.model.base_ptn.color = 'k'
            if self.model.waterfall_exist():
                for pattern in self.model.waterfall_ptn:
                    if (pattern.color == 'white') or \
                            (pattern.color == '#ffffff'):
                        pattern.color = 'k'
            self.obj_color = 'k'
        else:
            self.widget.mpl.canvas.set_toNight(True)
            if self.model.base_ptn_exist():
                self.model.base_ptn.color = 'white'
            if self.model.waterfall_exist():
                for pattern in self.model.waterfall_ptn:
                    if (pattern.color == 'k') or (pattern.color == '#000000'):
                        pattern.color = 'white'
            self.obj_color = 'white'

    def get_cake_range(self):
        if self.model.diff_img_exist():
            return self.widget.mpl.canvas.ax_cake.get_xlim(),\
                self.widget.mpl.canvas.ax_cake.get_ylim()
        else:
            return None, None

    def _read_azilist(self):
        n_row = self.widget.tableWidget_DiffImgAzi.rowCount()
        if n_row == 0:
            return None, None, None
        azi_list = []
        tth_list = []
        note_list = []
        for i in range(n_row):
            azi_min = float(
                self.widget.tableWidget_DiffImgAzi.item(i, 2).text())
            azi_max = float(
                self.widget.tableWidget_DiffImgAzi.item(i, 4).text())
            tth_min = float(
                self.widget.tableWidget_DiffImgAzi.item(i, 1).text())
            tth_max = float(
                self.widget.tableWidget_DiffImgAzi.item(i, 3).text())
            note_i = self.widget.tableWidget_DiffImgAzi.item(i, 0).text()
            tth_list.append([tth_min, tth_max])
            azi_list.append([azi_min, azi_max])
            note_list.append(note_i)
        return tth_list, azi_list, note_list

    def zoom_out_graph(self):
        if not self.model.base_ptn_exist():
            return
        data_limits = self._get_data_limits()
        self.update(limits=data_limits,
                    cake_ylimits=self._get_cake_y_limits())

    def _get_data_limits(self, y_margin=0.):
        if self.widget.checkBox_BgSub.isChecked():
            x_raw, y_raw = self.model.base_ptn.get_bgsub()
        else:
            x_raw, y_raw = self.model.base_ptn.get_raw()
        x_plot, y_plot = self._get_smoothed_pattern_xy(x_raw, y_raw)
        if self.diff_ctrl is not None:
            try:
                x_plot, y_plot = self.diff_ctrl.get_display_pattern(x_plot, y_plot)
                __, y_raw = self.diff_ctrl.get_display_pattern(x_raw, y_raw)
            except Exception:
                pass
        y_min = float(np.min(y_raw))
        y_max = float(np.max(y_raw))
        y_span = y_max - y_min
        if y_span == 0:
            y_span = max(abs(y_max), 1.0) * 1.0e-6
        return (x_plot.min(), x_plot.max(),
                y_min - y_span * y_margin,
                y_max + y_span * y_margin)

    def _get_cake_y_limits(self):
        if self.model.diff_img_exist():
            try:
                __, __, chi_cake = self.model.diff_img.get_cake()
            except Exception:
                chi_cake = None
            if chi_cake is not None and len(chi_cake) > 0:
                return (float(np.min(chi_cake)), float(np.max(chi_cake)))
        return None

    def _get_smoothing_settings(self):
        despike = int(self.widget.spinBox_SpectrumDespike.value()) \
            if hasattr(self.widget, "spinBox_SpectrumDespike") else 0
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

    def _smoothing_active(self):
        settings = self._get_smoothing_settings()
        return (settings["despike_kernel"] > 1) or (settings["sg_window"] > 1)

    def _smooth_xy(self, x, y):
        settings = self._get_smoothing_settings()
        if settings["despike_kernel"] <= 1 and settings["sg_window"] <= 1:
            return x, y
        if (medfilt is None) or (savgol_filter is None):
            return x, y
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        if x_arr.ndim != 1 or y_arr.ndim != 1:
            return x, y
        n = min(x_arr.size, y_arr.size)
        if n == 0:
            return x_arr[:0], y_arr[:0]
        x_arr = x_arr[:n]
        y_arr = y_arr[:n]
        y_smooth = np.asarray(y_arr, dtype=float)
        despike = int(settings["despike_kernel"])
        if despike > 1 and despike <= n:
            y_smooth = medfilt(y_smooth, kernel_size=despike)
        sg_window = int(settings["sg_window"])
        sg_poly = int(settings["sg_polyorder"])
        if sg_window > 1 and sg_window <= n and sg_poly < sg_window:
            y_smooth = savgol_filter(
                y_smooth,
                window_length=sg_window,
                polyorder=sg_poly,
            )
        return x_arr, np.asarray(y_smooth, dtype=float)

    def _get_smoothed_pattern_xy(self, x, y):
        if y is None:
            return x, y
        return self._smooth_xy(x, y)

    def _get_background_fit_display_xy(self):
        if not self.model.base_ptn_exist():
            return None, None
        x_raw, y_raw = self.model.base_ptn.get_raw()
        if x_raw is None or y_raw is None:
            return None, None
        x_fit, y_fit = self._get_smoothed_pattern_xy(x_raw, y_raw)
        if x_fit is None or y_fit is None:
            return None, None
        x_fit = np.asarray(x_fit, dtype=float).reshape(-1)
        y_fit = np.asarray(y_fit, dtype=float).reshape(-1)
        n = min(x_fit.size, y_fit.size)
        if n == 0:
            return None, None
        x_fit = x_fit[:n]
        y_fit = y_fit[:n]
        valid = np.isfinite(x_fit) & np.isfinite(y_fit)
        fit_areas = list(getattr(self.model.base_ptn, "bg_fit_areas", []) or [])
        if fit_areas:
            fit_mask = np.zeros(n, dtype=bool)
            for area in fit_areas:
                try:
                    xmin = float(area[0])
                    xmax = float(area[1])
                except Exception:
                    continue
                if xmax < xmin:
                    xmin, xmax = xmax, xmin
                fit_mask |= (x_fit >= xmin) & (x_fit <= xmax)
            valid &= fit_mask
        if np.count_nonzero(valid) == 0:
            return None, None
        return x_fit[valid], y_fit[valid]

    def _smooth_cake_x(self, intensity, x):
        settings = self._get_smoothing_settings()
        if settings["despike_kernel"] <= 1 and settings["sg_window"] <= 1:
            return intensity, x
        if (medfilt is None) or (savgol_filter is None):
            return intensity, x
        if intensity is None or x is None:
            return intensity, x
        x_arr = np.asarray(x, dtype=float).reshape(-1)
        if x_arr.size == 0:
            return intensity, x_arr
        is_masked = ma.isMaskedArray(intensity)
        data_arr = ma.asarray(intensity, dtype=float)
        if data_arr.ndim != 2:
            return intensity, x
        n_cols = min(data_arr.shape[1], x_arr.size)
        if n_cols == 0:
            return intensity, x_arr[:n_cols]
        data_arr = data_arr[:, :n_cols]
        x_arr = x_arr[:n_cols]
        smooth = np.asarray(ma.filled(data_arr, np.nan), dtype=float)
        despike = int(settings["despike_kernel"])
        if despike > 1 and despike <= n_cols:
            smooth = np.apply_along_axis(
                lambda row: medfilt(np.nan_to_num(row, nan=0.0), kernel_size=despike),
                1,
                smooth,
            )
        sg_window = int(settings["sg_window"])
        sg_poly = int(settings["sg_polyorder"])
        if sg_window > 1 and sg_window <= n_cols and sg_poly < sg_window:
            smooth = np.apply_along_axis(
                lambda row: savgol_filter(
                    np.nan_to_num(row, nan=0.0),
                    window_length=sg_window,
                    polyorder=sg_poly,
                ),
                1,
                smooth,
            )
        if is_masked:
            smooth = ma.masked_where(ma.getmaskarray(data_arr), smooth, copy=False)
        return smooth, x_arr

    def _plot_cake(self):
        """
        Controls cake viewing as well as mask
        """
        import matplotlib.patches as patches
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors

        #print(str(datetime.datetime.now())[:-7], ': Num of tth points = {0:.0f}, azi strips = {1:.0f}'.format(len(tth_cake), len(chi_cake)))

        # make a copy of intensity_cake and make sure it also has mask information 
        #intensity_cake_plot = ma.masked_values(intensity_cake, 0.)
        # intensity_cake_plot = ma.masked_equal(intensity_cake, 0.0, copy=False)
        #intensity_cake_plot = ma.array(intensity_cake, mask=self.model.diff_img.mask)

        # Get cake data
        intensity_cake, tth_cake, chi_cake = self.model.diff_img.get_cake()
        int_plot = np.array(intensity_cake, copy=True)

        diff_mode = False
        if self.diff_ctrl is not None:
            try:
                int_plot, tth_cake, chi_cake = self.diff_ctrl.get_display_cake(
                    int_plot, tth_cake, chi_cake)
                diff_mode = self.diff_ctrl.is_diff_mode_active()
            except Exception:
                diff_mode = False

        int_plot, tth_cake = self._smooth_cake_x(int_plot, tth_cake)
        if (int_plot is None) or (tth_cake is None) or (chi_cake is None):
            return
        if np.size(int_plot) == 0 or np.size(tth_cake) == 0 or np.size(chi_cake) == 0:
            return

        # Apply azimuthal shift after diff subtraction so the same shift is
        # effectively applied to both current and reference cake images.
        # Get image contrast parameters from UI unless diff mode overrides.
        if hasattr(self.widget, "doubleSpinBox_CCDScaleMin"):
            vmin = float(self.widget.doubleSpinBox_CCDScaleMin.value())
            vmax = float(self.widget.doubleSpinBox_CCDScaleMax.value())
            if vmax <= vmin:
                vmax = vmin + max(1e-6, 1e-6 * max(abs(vmin), 1.0))
                self.widget.doubleSpinBox_CCDScaleMax.setValue(vmax)
            climits = np.asarray([vmin, vmax], dtype=float)
        else:
            min_slider_pos = self.widget.horizontalSlider_VMin.value()
            max_slider_pos = self.widget.horizontalSlider_VMax.value()
            if (max_slider_pos <= min_slider_pos):
                self.widget.horizontalSlider_VMin.setValue(1)
                self.widget.horizontalSlider_VMax.setValue(99)
            prefactor = self.widget.spinBox_MaxCakeScale.value() / \
                (10. ** self.widget.horizontalSlider_MaxScaleBars.value())
            climits = np.asarray([
                self.widget.horizontalSlider_VMin.value(),
                self.widget.horizontalSlider_VMax.value()]) / \
                100. * prefactor

        # Check if ApplyMask is on
        # If so, get mask range from UI and set mask, then process cake for new mask.  Note that if mask from UI is for entire range of data, do not re-integrate.

        """        mask_range = self.model.diff_img.get_mask_range()
        if (self.widget.pushButton_ApplyMask.isChecked() and mask_range != None):
            vmin_mask, vmax_mask = mask_range
            mask = (int_plot < vmin_mask) | (int_plot > vmax_mask) | ~np.isfinite(int_plot)
            int_new = ma.masked_where(mask, int_plot, copy=False)
        else:
            if np.ma.isMaskedArray(int_plot):
                int_new = int_plot
            else:
                int_new = ma.MaskedArray(int_plot)  # no mask
        """

        # Colormap + mask handling.
        if diff_mode and (self.diff_ctrl is not None):
            cfg = self.diff_ctrl.get_cake_render_config(int_plot) or {}
            cmap = plt.get_cmap(cfg.get("cmap", "RdBu_r")).copy()
            climits = np.asarray([cfg.get("vmin", -1.0), cfg.get("vmax", 1.0)])
            norm = None
            if bool(cfg.get("center_zero", False)):
                try:
                    norm = mcolors.TwoSlopeNorm(
                        vmin=float(climits[0]), vcenter=0.0, vmax=float(climits[1]))
                except Exception:
                    norm = None
            zero_mask = np.zeros(np.shape(int_plot), dtype=bool)
            cmap.set_bad(color=(0.0, 0.0, 0.0, 0.0))
        else:
            # Non-diff mode uses user-selected colormap from Plot > Control.
            cmap_name = "gray_r"
            if hasattr(self.widget, "comboBox_CakeColormap"):
                cmap_name = str(self.widget.comboBox_CakeColormap.currentText() or "gray_r")
            cmap = plt.get_cmap(cmap_name).copy()
            norm = None
            # 0-values are typically masked pixels in cake data.
            zero_mask = (int_plot == 0)
            # Opaque pale yellow for masked pixels.
            cmap.set_bad(color=(1.0, 0.97, 0.55, 1.0))

        mask = self.model.diff_img.get_mask()
        use_user_mask = (self.widget.pushButton_ApplyMask.isChecked() and
                         (mask is not None) and np.any(mask))
        if use_user_mask:
            combined_mask = zero_mask | mask | ~np.isfinite(int_plot)
        else:
            combined_mask = zero_mask | ~np.isfinite(int_plot)
        int_new = ma.masked_where(combined_mask, int_plot, copy=False)


        imshow_kwargs = {
            "origin": "lower",
            "extent": [tth_cake.min(), tth_cake.max(), chi_cake.min(), chi_cake.max()],
            "aspect": "auto",
            "cmap": cmap,
        }
        if norm is None:
            imshow_kwargs["vmin"] = climits[0]
            imshow_kwargs["vmax"] = climits[1]
        else:
            imshow_kwargs["norm"] = norm
        self.widget.mpl.canvas.ax_cake.imshow(int_new, **imshow_kwargs)
        if hasattr(self.widget, "cake_hist_widget"):
            self.widget.cake_hist_widget.set_data(
                int_new, vmin=float(climits[0]), vmax=float(climits[1]))

        # get gray scale color map and make sure masked data points are colored red
        """
        if self.widget.checkBox_WhiteForPeak.isChecked():
            #cmap = 'gray'
            cmap = plt.cm.gray.copy()
        else:
            #cmap = 'gray_r'
            cmap = plt.cm.gray_r.copy()
        cmap.set_bad(color='red')
        """

        # plot the data as an image
        """
        self.widget.mpl.canvas.ax_cake.imshow(
            int_new, origin="lower",
            extent=[tth_cake.min(), tth_cake.max(),
                    chi_cake.min(), chi_cake.max()],
            aspect="auto", cmap=cmap, clim=climits)  # gray_r
        """
        #print(str(datetime.datetime.now())[:-7], ': Cake intensity min, max = ', climits)

        # overlay azimuthal sections information
        tth_list, azi_list, note_list = self._read_azilist()
        tth_min = tth_cake.min()
        tth_max = tth_cake.max()
        if azi_list is not None:
            for tth, azi, note in zip(tth_list, azi_list, note_list):
                rect = patches.Rectangle(
                    (tth_min, azi[0]), (tth_max - tth_min), (azi[1] - azi[0]),
                    linewidth=0, edgecolor='b', facecolor='b', alpha=0.2)
                rect1 = patches.Rectangle(
                    (tth[0], azi[0]), (tth[1] - tth[0]), (azi[1] - azi[0]),
                    linewidth=1, edgecolor='b', facecolor='None')
                self.widget.mpl.canvas.ax_cake.add_patch(rect)
                self.widget.mpl.canvas.ax_cake.add_patch(rect1)
                if self.widget.checkBox_ShowCakeLabels.isChecked():
                    self.widget.mpl.canvas.ax_cake.text(
                        tth[1], azi[1], note, color=self.obj_color)
        rows = self.widget.tableWidget_DiffImgAzi.selectionModel().\
            selectedRows()
        if rows != []:
            for r in rows:
                azi_min = float(
                    self.widget.tableWidget_DiffImgAzi.item(r.row(), 2).text())
                azi_max = float(
                    self.widget.tableWidget_DiffImgAzi.item(r.row(), 4).text())
                rect = patches.Rectangle(
                    (tth_min, azi_min), (tth_max - tth_min),
                    (azi_max - azi_min),
                    linewidth=0, facecolor='r', alpha=0.2)
                self.widget.mpl.canvas.ax_cake.add_patch(rect)
        if hasattr(self.widget, "spinBox_CCDRowMin") and hasattr(self.widget, "spinBox_CCDRowMax"):
            row_min = float(self.widget.spinBox_CCDRowMin.value())
            row_max = float(self.widget.spinBox_CCDRowMax.value())
            if row_max >= row_min:
                rect = patches.Rectangle(
                    (tth_min, row_min),
                    (tth_max - tth_min),
                    (row_max - row_min + 1.0),
                    linewidth=1.5,
                    edgecolor='#00e676',
                    facecolor='#00e676',
                    alpha=0.12)
                self.widget.mpl.canvas.ax_cake.add_patch(rect)

    def _plot_jcpds(self, axisrange):
        import matplotlib.transforms as transforms

        # t_start = time.time()
        if (not self.widget.checkBox_JCPDSinPattern.isChecked()) and \
                (not self.widget.checkBox_JCPDSinCake.isChecked()):
            return
        selected_phases = []
        for phase in self.model.jcpds_lst:
            if phase.display:
                selected_phases.append(phase)
        if selected_phases == []:
            return
        n_displayed_jcpds = len(selected_phases)
        # axisrange = self.widget.mpl.canvas.ax_pattern.axis()
        cakerange = self.widget.mpl.canvas.ax_cake.axis()
        bar_scale = 1. / 100. * axisrange[3] * \
            self.widget.horizontalSlider_JCPDSBarScale.value() / 100.
        bar_pos = self.widget.horizontalSlider_JCPDSBarPosition.value() / 100.
        show_intensity = self.widget.checkBox_Intensity.isChecked()
        if not show_intensity:
            data_limits = self._get_data_limits()
            start_intensity = data_limits[2] + bar_pos * axisrange[3]
        pressure = self.widget.doubleSpinBox_Pressure.value()
        temperature = self.widget.doubleSpinBox_Temperature.value()
        wavelength = self.widget.doubleSpinBox_SetWavelength.value()
        use_table_0gpa = self.widget.checkBox_UseJCPDSTable1bar.isChecked()
        for i, phase in enumerate(selected_phases):
#            try:
            phase.cal_dsp(pressure,
                            temperature,
                            use_table_for_0GPa=use_table_0gpa)
#            except:
#                QtWidgets.QMessageBox.warning(
#                    self.widget, "Warning",
#                    phase.name+" created issues with pressure calculation.")
#                break
            tth, inten = phase.get_tthVSint(
                wavelength)
            if self.widget.checkBox_JCPDSinPattern.isChecked():
                intensity = inten * phase.twk_int
                if show_intensity:
                    bar_min = np.ones_like(tth) * axisrange[2] + \
                        bar_pos * axisrange[3]
                    bar_max = intensity * bar_scale + bar_min
                else:
                    starting_intensity = np.ones_like(tth) * start_intensity
                    bar_max = starting_intensity - \
                        i * 100. * bar_scale / n_displayed_jcpds
                    bar_min = starting_intensity - \
                        (i+0.7) * 100. * bar_scale / n_displayed_jcpds
                if (pressure == 0.) or (phase.symmetry == 'nosymmetry'):
                    volume = phase.v
                else:
                    volume = phase.v.item()
                self.widget.mpl.canvas.ax_pattern.vlines(
                    tth, bar_min, bar_max, colors=phase.color,
                    label="{0:}, {1:.3f} A^3".format(
                        phase.name, volume),
                    lw=float(
                        self.widget.comboBox_PtnJCPDSBarThickness.
                        currentText()),
                    alpha=self.widget.doubleSpinBox_JCPDS_ptn_Alpha.value())
                # hkl
                if self.widget.checkBox_ShowMillerIndices.isChecked():
                    hkl_list = phase.get_hkl_in_text()
                    for j, hkl in enumerate(hkl_list):
                        self.widget.mpl.canvas.ax_pattern.text(
                            tth[j], bar_max[j], hkl, color=phase.color,
                            rotation=90, verticalalignment='bottom',
                            horizontalalignment='center',
                            fontsize=int(
                                self.widget.comboBox_HKLFontSize.currentText()),
                            alpha=self.widget.doubleSpinBox_JCPDS_ptn_Alpha.value())
                # phase.name, phase.v.item()))
            if self.model.diff_img_exist() and \
                    self.widget.checkBox_JCPDSinCake.isChecked():
                self.widget.mpl.canvas.ax_cake.vlines(
                    tth, np.ones_like(tth) * cakerange[2],
                    np.ones_like(tth) * cakerange[3], colors=phase.color,
                    lw=float(
                        self.widget.comboBox_CakeJCPDSBarThickness.currentText()),
                    alpha=self.widget.doubleSpinBox_JCPDS_cake_Alpha.value())
                if self.widget.checkBox_ShowMillerIndices_Cake.isChecked():
                    hkl_list = phase.get_hkl_in_text()
                    trans = transforms.blended_transform_factory(
                        self.widget.mpl.canvas.ax_cake.transData,
                        self.widget.mpl.canvas.ax_cake.transAxes)
                    for j, hkl in enumerate(hkl_list):
                        self.widget.mpl.canvas.ax_cake.text(
                            tth[j], 0.99, hkl, color=phase.color,
                            rotation=90, verticalalignment='top',
                            transform=trans, horizontalalignment='right',
                            fontsize=int(
                                self.widget.comboBox_HKLFontSize.currentText()),
                            alpha=self.widget.doubleSpinBox_JCPDS_cake_Alpha.value())
        if self.widget.checkBox_JCPDSinPattern.isChecked():
            legend_fontsize = 14
            if hasattr(self.widget, "comboBox_LegendFontSize"):
                try:
                    legend_fontsize = int(
                        self.widget.comboBox_LegendFontSize.currentText())
                except Exception:
                    pass
            leg_jcpds = self.widget.mpl.canvas.ax_pattern.legend(
                loc=1, framealpha=0.,
                fontsize=legend_fontsize,
                handlelength=1)
            for line, txt in zip(leg_jcpds.get_lines(), leg_jcpds.get_texts()):
                txt.set_color(line.get_color())
        # print("JCPDS update takes {0:.2f}s at".format(time.time() - t_start),
        #      str(datetime.datetime.now())[:-7])

    def _plot_waterfallpatterns(self):
        if not self.widget.checkBox_ShowWaterfall.isChecked():
            return
        # t_start = time.time()
        # count how many are dispaly
        i = 0
        for pattern in self.model.waterfall_ptn:
            if pattern.display:
                i += 1
        if i == 0:
            return
        n_display = i
        j = 0  # this is needed for waterfall gaps
        # get y_max
        for pattern in self.model.waterfall_ptn[::-1]:
            if pattern.display:
                j += 1
                """
                self.widget.mpl.canvas.ax_pattern.text(
                    0.01, 0.97 - n_display * 0.05 + j * 0.05,
                    os.path.basename(pattern.fname),
                    transform=self.widget.mpl.canvas.ax_pattern.transAxes,
                    color=pattern.color)
                """
                if self.widget.checkBox_BgSub.isChecked():
                    base_y = np.asarray(self.model.base_ptn.y_bgsub, dtype=float)
                    if base_y.size == 0:
                        continue
                    y_span = float(np.nanmax(base_y) - np.nanmin(base_y))
                    if (not np.isfinite(y_span)) or (y_span <= 0):
                        y_span = float(np.nanmax(np.abs(base_y)))
                    ygap = self.widget.horizontalSlider_WaterfallGaps.value() * \
                        y_span * float(j) / 10000.
                    y_bgsub = pattern.y_bgsub
                    if self.widget.checkBox_IntNorm.isChecked():
                        y = y_bgsub / y_bgsub.max() * \
                            self.model.base_ptn.y_bgsub.max()
                    else:
                        y = y_bgsub
                    x_t = pattern.x_bgsub
                else:
                    base_y = np.asarray(self.model.base_ptn.y_raw, dtype=float)
                    if base_y.size == 0:
                        continue
                    y_span = float(np.nanmax(base_y) - np.nanmin(base_y))
                    if (not np.isfinite(y_span)) or (y_span <= 0):
                        y_span = float(np.nanmax(np.abs(base_y)))
                    ygap = self.widget.horizontalSlider_WaterfallGaps.value() * \
                        y_span * float(j) / 10000.
                    if self.widget.checkBox_IntNorm.isChecked():
                        y = pattern.y_raw / pattern.y_raw.max() *\
                            self.model.base_ptn.y_raw.max()
                    else:
                        y = pattern.y_raw
                    x_t = pattern.x_raw
                x, y = self._get_smoothed_pattern_xy(x_t, y)
                if x is None or y is None or len(x) == 0 or len(y) == 0:
                    continue
                if len(x) != len(y):
                    n = min(len(x), len(y))
                    x = x[:n]
                    y = y[:n]
                self.widget.mpl.canvas.ax_pattern.plot(
                    x, y + ygap, c=pattern.color, lw=float(
                        self.widget.comboBox_WaterfallLineThickness.
                        currentText()))
                if self.widget.checkBox_ShowWaterfallLabels.isChecked():
                    wf_fontsize = 12
                    if hasattr(self.widget, "comboBox_WaterfallFontSize"):
                        try:
                            wf_fontsize = int(
                                self.widget.comboBox_WaterfallFontSize.currentText())
                        except Exception:
                            pass
                    self.widget.mpl.canvas.ax_pattern.text(
                        (x[-1] - x[0]) * 0.01 + x[0], y[0] + ygap,
                        os.path.basename(pattern.fname),
                        verticalalignment='bottom', horizontalalignment='left',
                        color=pattern.color, fontsize=wf_fontsize)
        """
        self.widget.mpl.canvas.ax_pattern.text(
            0.01, 0.97 - n_display * 0.05,
            os.path.basename(self.model.base_ptn.fname),
            transform=self.widget.mpl.canvas.ax_pattern.transAxes,
            color=self.model.base_ptn.color)
        """

    def _plot_diffpattern(self, gsas_style=False):
        if self.widget.checkBox_BgSub.isChecked():
            x_raw, y_raw = self.model.base_ptn.get_bgsub()
        else:
            x_raw, y_raw = self.model.base_ptn.get_raw()
        x, y = x_raw, y_raw
        x_s, y_s = self._get_smoothed_pattern_xy(x_raw, y_raw)
        if self.diff_ctrl is not None:
            try:
                x, y = self.diff_ctrl.get_display_pattern(x, y)
                x_s, y_s = self.diff_ctrl.get_display_pattern(x_s, y_s)
            except Exception:
                pass
        if gsas_style:
            self.widget.mpl.canvas.ax_pattern.plot(
                x, y, c=self.model.base_ptn.color, marker='o',
                linestyle='None', ms=3)
        elif self._smoothing_active():
            self.widget.mpl.canvas.ax_pattern.plot(
                x, y, c=self.model.base_ptn.color,
                marker='.', linestyle='None', ms=3, alpha=0.5)
            self.widget.mpl.canvas.ax_pattern.plot(
                x_s, y_s, c=self.model.base_ptn.color,
                lw=float(
                    self.widget.comboBox_BasePtnLineThickness.
                    currentText()))
        else:
            self.widget.mpl.canvas.ax_pattern.plot(
                x, y, c=self.model.base_ptn.color,
                lw=float(
                    self.widget.comboBox_BasePtnLineThickness.
                    currentText()))
        if self.diff_ctrl is not None and self.diff_ctrl.is_diff_mode_active():
            self.widget.mpl.canvas.ax_pattern.axhline(
                0.0, ls='--', c='tab:red', lw=0.8)
            return
        if (not self.widget.checkBox_BgSub.isChecked()) and \
                hasattr(self.widget, "checkBox_ShowBg") and \
                self.widget.checkBox_ShowBg.isChecked():
            x_bg, y_bg = self.model.base_ptn.get_background()
            x_bg, y_bg = self._get_smoothed_pattern_xy(x_bg, y_bg)
            self.widget.mpl.canvas.ax_pattern.plot(
                x_bg, y_bg, c=self.model.base_ptn.color, ls='--',
                lw=float(
                    self.widget.comboBox_BkgnLineThickness.
                    currentText()))
            x_fit, y_fit = self._get_background_fit_display_xy()
            if x_fit is not None and y_fit is not None:
                self.widget.mpl.canvas.ax_pattern.plot(
                    x_fit, y_fit,
                    c=self.model.base_ptn.color,
                    marker='o',
                    linestyle='None',
                    ms=5,
                    mfc='none',
                    mew=1.0,
                    alpha=0.9)

    def _plot_peakfit(self):
        if not self.model.current_section_exist():
            return
        if self.model.current_section.peaks_exist():
            for x_c in self.model.current_section.get_peak_positions():
                self.widget.mpl.canvas.ax_pattern.axvline(
                    x_c, ls='--', dashes=(10, 5))
        if self.model.current_section.fitted():
            bgsub = self.widget.checkBox_BgSub.isChecked()
            x_plot = self.model.current_section.x
            profiles = self.model.current_section.get_individual_profiles(
                bgsub=bgsub)
            for key, value in profiles.items():
                self.widget.mpl.canvas.ax_pattern.plot(
                    x_plot, value, ls='-', c=self.obj_color, lw=float(
                        self.widget.comboBox_BasePtnLineThickness.
                        currentText()))
            total_profile = self.model.current_section.get_fit_profile(
                bgsub=bgsub)
            residue = self.model.current_section.get_fit_residue(bgsub=bgsub)
            self.widget.mpl.canvas.ax_pattern.plot(
                x_plot, total_profile, 'r-', lw=float(
                    self.widget.comboBox_BasePtnLineThickness.
                    currentText()))
            y_range = self.model.current_section.get_yrange(bgsub=bgsub)
            y_shift = y_range[0] - (y_range[1] - y_range[0]) * 0.05
            #(y_range[1] - y_range[0]) * 1.05
            self.widget.mpl.canvas.ax_pattern.fill_between(
                x_plot, self.model.current_section.get_fit_residue_baseline(
                    bgsub=bgsub) + y_shift, residue + y_shift, facecolor='r')
            """
            self.widget.mpl.canvas.ax_pattern.plot(
                x_plot, residue + y_shift, 'r-')
            self.widget.mpl.canvas.ax_pattern.axhline(
                self.model.current_section.get_fit_residue_baseline(
                    bgsub=bgsub) + y_shift, c='r', ls='-', lw=0.5)
            """
        else:
            pass

    def _plot_peakfit_in_gsas_style(self):
        # get all the highlights
        # iteratively run plot
        rows = self.widget.tableWidget_PkFtSections.selectionModel().\
            selectedRows()
        if rows == []:
            return
        else:
            selected_rows = [r.row() for r in rows]
        bgsub = self.widget.checkBox_BgSub.isChecked()
        data_limits = self._get_data_limits()
        y_shift = data_limits[2] - (data_limits[3] - data_limits[2]) * 0.05
        i = 0
        for section in self.model.section_lst:
            if i in selected_rows:
                x_plot = section.x
                total_profile = section.get_fit_profile(bgsub=bgsub)
                residue = section.get_fit_residue(bgsub=bgsub)
                self.widget.mpl.canvas.ax_pattern.plot(
                    x_plot, total_profile, 'r-', lw=float(
                        self.widget.comboBox_BasePtnLineThickness.
                        currentText()))
                self.widget.mpl.canvas.ax_pattern.fill_between(
                    x_plot, section.get_fit_residue_baseline(bgsub=bgsub) +
                    y_shift, residue + y_shift, facecolor='r')
            i += 1

    def _plot_background_fit_areas(self):
        table = getattr(self.widget, "tableWidget_BackgroundConstraints", None)
        if table is None:
            return
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
            self.widget.mpl.canvas.ax_pattern.axvspan(
                xmin, xmax,
                ymin=0.0, ymax=1.0,
                facecolor="#00c853",
                edgecolor="#00c853",
                alpha=0.10,
                linewidth=1.0,
            )

    def _fits_tab_active(self):
        """
        Determine if the Fits tab is currently active.
        Avoid hardcoded tab indices because UI tab order can change.
        """
        if hasattr(self.widget, "tab_PkFt"):
            try:
                return self.widget.tabWidget.currentWidget() == self.widget.tab_PkFt
            except Exception:
                pass
        # Backward-compatible fallback.
        return self.widget.tabWidget.currentIndex() in (4, 5)

    def update(self, limits=None, gsas_style=False, cake_ylimits=None):
        if limits is not None:
            limits = tuple(limits)
        if cake_ylimits is not None:
            cake_ylimits = tuple(cake_ylimits)
        self._pending_update_args = (limits, bool(gsas_style), cake_ylimits)
        self._update_timer.start(self._update_delay_ms)

    def _flush_update_request(self):
        if self._pending_update_args is None:
            return
        if self._is_drawing or self._toolbar_active:
            self._update_timer.start(self._update_delay_ms)
            return
        limits, gsas_style, cake_ylimits = self._pending_update_args
        self._pending_update_args = None
        self._update_impl(limits=limits, gsas_style=gsas_style, cake_ylimits=cake_ylimits)
        if self._pending_update_args is not None:
            self._update_timer.start(self._update_delay_ms)

    def _update_impl(self, limits=None, gsas_style=False, cake_ylimits=None):
        """Updates the graph"""
        # ✅ Block updates during drawing OR toolbar interaction
        if self._is_drawing or self._toolbar_active:
            return
        
        # ✅ Pre-check conditions BEFORE setting flag
        if (not self.model.base_ptn_exist()) and \
                (not self.model.jcpds_exist()):
            return
        
        t_start = time.time()
        self.widget.setCursor(QtCore.Qt.WaitCursor)
        
        # ✅ Set drawing flag AFTER pre-checks
        self._is_drawing = True
        
        try:
            if limits is None:
                limits = self.widget.mpl.canvas.ax_pattern.axis()
            if cake_ylimits is None:
                cake_ylimits = self._get_cake_y_limits()
                if cake_ylimits is None and hasattr(self.widget.mpl.canvas, 'ax_cake'):
                    c_limits = self.widget.mpl.canvas.ax_cake.axis()
                    cake_ylimits = c_limits[2:4]
            
            if self.model.diff_img_exist():
                new_height = self.widget.horizontalSlider_CakeAxisSize.value()
                self.widget.mpl.canvas.resize_axes(new_height)
                self._plot_cake()
            else:
                self.widget.mpl.canvas.resize_axes(1)
            
            self._set_nightday_view()
            
            if self.model.base_ptn_exist():
                title_font_size = 12
                if hasattr(self.widget, "spinBox_TitleFontSize"):
                    try:
                        title_font_size = int(self.widget.spinBox_TitleFontSize.value())
                    except Exception:
                        title_font_size = 12
                max_title_chars = 140
                if hasattr(self.widget, "spinBox_TitleMaxLength"):
                    try:
                        max_title_chars = int(self.widget.spinBox_TitleMaxLength.value())
                    except Exception:
                        max_title_chars = 140

                if self.widget.checkBox_ShortPlotTitle.isChecked():
                    raw_title = os.path.basename(self.model.base_ptn.fname)
                else:
                    raw_title = self.model.base_ptn.fname

                truncate_middle = True
                if hasattr(self.widget, "checkBox_TitleTruncateMiddle"):
                    truncate_middle = bool(
                        self.widget.checkBox_TitleTruncateMiddle.isChecked())
                title = truncate_title_by_chars(
                    raw_title, max_title_chars, truncate_middle=truncate_middle)
                fig_width_pixels = \
                    self.widget.mpl.canvas.fig.get_size_inches()[0] * \
                    self.widget.mpl.canvas.fig.dpi
                max_width = 0.85 * fig_width_pixels
                title = truncate_title(title, title_font_size, max_width)
                
                self.widget.mpl.canvas.fig.suptitle(
                    title, color=self.obj_color, fontsize=title_font_size)
                
                self._plot_diffpattern(gsas_style)
                
                if self.model.waterfall_exist():
                    self._plot_waterfallpatterns()
            
            if self._fits_tab_active():
                if gsas_style:
                    self._plot_peakfit_in_gsas_style()
                else:
                    self._plot_peakfit()
            
            self.widget.mpl.canvas.ax_pattern.set_xlim(limits[0], limits[1])
            
            if not self.widget.checkBox_AutoY.isChecked():
                self.widget.mpl.canvas.ax_pattern.set_ylim(limits[2], limits[3])
            
            # ✅ Check if ax_cake exists before setting ylim
            if hasattr(self.widget.mpl.canvas, 'ax_cake') and (cake_ylimits is not None):
                self.widget.mpl.canvas.ax_cake.set_ylim(cake_ylimits)
            
            if self.model.jcpds_exist():
                self._plot_jcpds(limits)
                if not self.widget.checkBox_Intensity.isChecked():
                    new_low_limit = -1.1 * limits[3] * \
                        self.widget.horizontalSlider_JCPDSBarScale.value() / 100.
                    self.widget.mpl.canvas.ax_pattern.set_ylim(
                        new_low_limit, limits[3])
            
            if self.widget.checkBox_ShowLargePnT.isChecked():
                label_p_t = "{0: 5.1f} GPa".format(
                    self.widget.doubleSpinBox_Pressure.value())
                self.widget.mpl.canvas.ax_pattern.text(
                    0.01, 0.98, label_p_t, horizontalalignment='left',
                    verticalalignment='top',
                    transform=self.widget.mpl.canvas.ax_pattern.transAxes,
                    fontsize=int(
                        self.widget.comboBox_PnTFontSize.currentText()))
            
            if self._is_spe_mode():
                self.widget.mpl.canvas.ax_pattern.set_xlabel(r"Raman Shift (cm$^{-1}$)")
                self.widget.mpl.canvas.ax_pattern.format_coord = \
                    lambda x, y: "\n Shift={0:.3f} cm-1, I={1:.4e}".format(x, y)
                if hasattr(self.widget.mpl.canvas, 'ax_cake'):
                    self.widget.mpl.canvas.ax_cake.set_ylabel("CCD Pixel")
            else:
                xlabel = "Two Theta (degrees), {:6.4f} \u212B".\
                    format(self.widget.doubleSpinBox_SetWavelength.value())
                self.widget.mpl.canvas.ax_pattern.set_xlabel(xlabel)
                self.widget.mpl.canvas.ax_pattern.format_coord = \
                    lambda x, y: \
                    "\n 2\u03B8={0:.3f}\u00B0, I={1:.4e}, d-sp={2:.4f}\u212B".\
                    format(x, y,
                        self.widget.doubleSpinBox_SetWavelength.value()
                        / 2. / np.sin(np.radians(x / 2.)))
            
            # ✅ Only set cake format_coord if ax_cake exists
            if hasattr(self.widget.mpl.canvas, 'ax_cake'):
                """
                self.widget.mpl.canvas.ax_cake.format_coord = \
                    lambda x, y: \
                    "\n 2\u03B8={0:.3f}\u00B0, azi={1:.1f}, d-sp={2:.4f}\u212B".\
                    format(x, y,  
                        self.widget.doubleSpinBox_SetWavelength.value()
                        / 2. / np.sin(np.radians(x / 2.)))
                """
                self.widget.mpl.canvas.ax_cake.format_coord = self._format_coord_x_y_z_dsp
            
            # ✅ MOVED: Set up cursor BEFORE drawing (inside try block)
            if self.widget.checkBox_LongCursor.isChecked():
                # Determine which axes to use
                if hasattr(self.widget.mpl.canvas, 'ax_cake') and \
                   self.model.diff_img_exist():
                    # Use both axes
                    axes_list = (self.widget.mpl.canvas.ax_pattern,
                                self.widget.mpl.canvas.ax_cake)
                else:
                    # Use only pattern axis
                    axes_list = (self.widget.mpl.canvas.ax_pattern,)
                
                # Get line width
                try:
                    lw_value = float(
                        self.widget.comboBox_VertCursorThickness.currentText())
                except:
                    lw_value = 1.0
                
                # Create MultiCursor
                self.widget.cursor = MultiCursor(
                    self.widget.mpl.canvas.fig,  # Use figure, not canvas
                    axes_list,
                    color='r',
                    lw=lw_value,
                    ls='--',
                    useblit=False,
                    horizOn=False)  # Only vertical line
            else:
                # Clear cursor if checkbox is unchecked
                if hasattr(self.widget, 'cursor'):
                    self.widget.cursor = None
            
            # ✅ Draw canvas (deferred to Qt event loop)
            QtCore.QTimer.singleShot(0, self.widget.mpl.canvas.draw)
            
            print(str(datetime.datetime.now())[:-7], 
                ": Plot takes {0:.2f}s".format(time.time() - t_start))
        
        except Exception as e:
            print(f"Error during plot update: {e}")
            import traceback
            traceback.print_exc()
        
        # ✅ Always clear flag and restore cursor
        finally:
            self._is_drawing = False
            self.widget.unsetCursor()
            if self._pending_update_args is not None:
                self._update_timer.start(0)

    def _format_coord_x_y_z_dsp(self, x, y):
        """
        Read 2theta, azimuthal angle, intensity, and d-spacing from the image
        
        :param x: 2 theta angle
        :param y: azimuthal angle
        """
        if self._is_spe_mode():
            ax = self.widget.mpl.canvas.ax_cake
            if not ax.images:
                return "Shift={:.3f} cm-1, pixel={:.1f}, I=NA".format(x, y)
            img = ax.images[0]
            data = img.get_array()
            xmin, xmax, ymin, ymax = img.get_extent()
            try:
                ny, nx = data.shape
                fx = (x - xmin) / (xmax - xmin) * (nx - 1)
                fy = (y - ymin) / (ymax - ymin) * (ny - 1)
                col = int(round(fx))
                row = int(round(fy))
                col = min(max(col, 0), nx - 1)
                row = min(max(row, 0), ny - 1)
                z_val = data[row, col]
                z_text = "{:.0f}".format(float(z_val))
            except Exception:
                z_text = "NA"
            return "Shift={:.3f} cm-1, pixel={:.1f}, I={}".format(x, y, z_text)
        ax = self.widget.mpl.canvas.ax_cake

        # compute d-spacing from x (2-theta)
        try:
            dsp = (self.widget.doubleSpinBox_SetWavelength.value()
                   / 2.0 / np.sin(np.radians(x / 2.0)))
        except Exception:
            dsp = None

        # If no image on the axis, return x,y,dsp only
        if not ax.images:
            if dsp is None:
                return "2\u03B8={:.3f}\u00B0, azi={:.1f}, I=NA, d-sp=NA".format(x, y)
            return "2\u03B8={:.3f}\u00B0, azi={:.1f}, I=NA, d-sp={:.4f}\u212B".format(x, y, dsp)

        img = ax.images[0]
        data = img.get_array()
        if data is None:
            if dsp is None:
                return "2\u03B8={:.3f}\u00B0, azi={:.1f}, I=NA, d-sp=NA".format(x, y)
            return "2\u03B8={:.3f}\u00B0, azi={:.1f}, I=NA, d-sp={:.4f}\u212B".format(x, y, dsp)

        # extent -> map data coords to pixel indices
        xmin, xmax, ymin, ymax = img.get_extent()
        if xmax == xmin or ymax == ymin:
            # degenerate extent
            if dsp is None:
                return "2\u03B8={:.3f}\u00B0, azi={:.1f}, I=NA, d-sp=NA".format(x, y)
            return "2\u03B8={:.3f}\u00B0, azi={:.1f}, I=NA, d-sp={:.4f}\u212B".format(x, y, dsp)

        # ensure 2D image
        try:
            ny, nx = data.shape
        except Exception:
            # not a 2D image
            if dsp is None:
                return "2\u03B8={:.3f}\u00B0, azi={:.1f}, I=NA, d-sp=NA".format(x, y)
            return "2\u03B8={:.3f}\u00B0, azi={:.1f}, I=NA, d-sp={:.4f}\u212B".format(x, y, dsp)

        # fractional positions (0..nx-1, 0..ny-1)
        fx = (x - xmin) / (xmax - xmin) * (nx - 1)
        fy = (y - ymin) / (ymax - ymin) * (ny - 1)

        # nearest-neighbor
        col = int(round(fx))
        row = int(round(fy))

        # handle origin
        origin = getattr(img, 'origin', None)
        if origin == 'upper':
            row = (ny - 1) - row

        # clamp & check bounds
        if col < 0 or col >= nx or row < 0 or row >= ny:
            if dsp is None:
                return "2\u03B8={:.3f}\u00B0, azi={:.1f}, I=NA, d-sp=NA".format(x, y)
            return "2\u03B8={:.3f}\u00B0, azi={:.1f}, I=NA, d-sp={:.4f}\u212B".format(x, y, dsp)

        # read intensity, handle masked/invalid
        try:
            if np.ma.isMaskedArray(data):
                mask = data.mask
                if mask is not None and mask.shape == data.shape and mask[row, col]:
                    z_text = "NA"
                else:
                    z_val = data.data[row, col]
                    if np.isnan(z_val) or np.isinf(z_val):
                        z_text = "(invalid)"
                    else:
                        z_text = "{:.0f}".format(float(z_val))
            else:
                z_val = data[row, col]
                if isinstance(z_val, (float, np.floating)) and (np.isnan(z_val) or np.isinf(z_val)):
                    z_text = "(invalid)"
                else:
                    z_text = "{:.0f}".format(float(z_val))
        except Exception:
            z_text = "NA"

        # format final string: x, y, z, d-sp
        if dsp is None:
            return "2\u03B8={:.3f}\u00B0, azi={:.1f}, I={}, d-sp=NA".format(x, y, z_text)
        return "2\u03B8={:.3f}\u00B0, azi={:.1f}, I={}, d-sp={:.4f}\u212B".format(x, y, z_text, dsp)

from matplotlib.textpath import TextPath

def truncate_title_by_chars(title, max_chars, truncate_middle=True):
    if title is None:
        return ""
    title = str(title)
    try:
        max_chars = int(max_chars)
    except Exception:
        max_chars = 140
    if max_chars < 20:
        max_chars = 20
    if len(title) <= max_chars:
        return title
    if not truncate_middle:
        tail_len = max_chars - 4
        if tail_len < 1:
            tail_len = 1
        return "... " + title[-tail_len:]
    head = int(max_chars * 0.45)
    tail = max_chars - head - 5
    if tail < 1:
        tail = 1
    return title[:head] + " ... " + title[-tail:]

def truncate_title(title, font_size, max_width):
    """Fast truncation without expensive TextPath calculations"""
    # ✅ Simple character-based truncation
    # Approximate: average character is ~7 pixels at size 12
    if isinstance(font_size, str):
        font_size = 12  # Default
    else:
        font_size = float(font_size)
    
    # Rough estimate of characters that fit
    approx_chars = int(max_width / (font_size * 0.6))
    
    if len(title) <= approx_chars:
        return title
    
    # Keep first 30% and last 50% of available space
    first_chars = int(approx_chars * 0.3)
    last_chars = int(approx_chars * 0.5)
    
    if first_chars + last_chars + 5 >= len(title):
        return title
    
    return title[:first_chars] + " ... " + title[-last_chars:]
