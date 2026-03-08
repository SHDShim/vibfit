import os
import glob
from qtpy import QtWidgets, QtCore
import numpy as np
import matplotlib.patches as patches
from ..utils import get_temp_dir, extract_extension, InformationBox
from .mplcontroller import MplController
from .ccdprocesscontroller import CCDProcessController
from PIL import Image
import json
import datetime

class CCDController(object):

    def __init__(self, model, widget):
        self.model = model
        self.widget = widget
        self.ccdprocess_ctrl = CCDProcessController(self.model, self.widget)
        self.plot_ctrl = MplController(self.model, self.widget)
        self._row_roi_selector = None
        self._row_roi_press_cid = None
        self._row_roi_motion_cid = None
        self._row_roi_release_cid = None
        self._row_roi_press_y = None
        self._row_roi_preview = None
        self.connect_channel()

    def connect_channel(self):
        if hasattr(self.widget, "pushButton_Info"):
            self.widget.pushButton_Info.clicked.connect(self.show_tif_header)
        self.widget.pushButton_ApplyCakeView.clicked.connect(self.update_cake)
        self.widget.pushButton_ApplyMask.clicked.connect(self.apply_mask)
        self.widget.pushButton_MaskReset.clicked.connect(self.reset_maskrange)
        self.widget.pushButton_ResetCakeScale.clicked.connect(
            self.reset_max_cake_scale)
        if hasattr(self.widget, "spinBox_CCDRowMin"):
            self.widget.spinBox_CCDRowMin.valueChanged.connect(
                self._on_row_roi_spin_changed)
        if hasattr(self.widget, "spinBox_CCDRowMax"):
            self.widget.spinBox_CCDRowMax.valueChanged.connect(
                self._on_row_roi_spin_changed)
        if hasattr(self.widget, "pushButton_CCDSelectRoi"):
            self.widget.pushButton_CCDSelectRoi.toggled.connect(
                self._toggle_row_roi_selector)
        if hasattr(self.widget, "pushButton_CCDFullRoi"):
            self.widget.pushButton_CCDFullRoi.clicked.connect(
                self._reset_row_roi_to_full)
        if hasattr(self.widget, "comboBox_CakeColormap"):
            self.widget.comboBox_CakeColormap.currentIndexChanged.connect(
                self._apply_changes_to_graph)
        if hasattr(self.widget, "doubleSpinBox_CCDScaleMin"):
            self.widget.doubleSpinBox_CCDScaleMin.valueChanged.connect(
                self._on_cake_scale_spin_changed)
        if hasattr(self.widget, "doubleSpinBox_CCDScaleMax"):
            self.widget.doubleSpinBox_CCDScaleMax.valueChanged.connect(
                self._on_cake_scale_spin_changed)
        if hasattr(self.widget, "cake_hist_widget"):
            self.widget.cake_hist_widget.boundChanged.connect(
                self._set_cake_bound_from_hist)
        """
        self.widget.pushButton_Load_CakeFormatFile.clicked.connect(
            self.load_cake_format_file)
        self.widget.pushButton_Save_CakeFormatFile.clicked.connect(
            self.save_cake_format_file)
        """

    def update_cake(self):
        if not self.model.base_ptn_exist():
            QtWidgets.QMessageBox.warning(
                self.widget, 'Warning', 'Choose a spectrum file first.')
            return
        if (not self._is_spe_source()) and (not self.model.associated_image_exists()):
            self._set_image_file_box_missing()
            QtWidgets.QMessageBox.warning(
                self.widget, 'Warning',
                'CCD image file does not exist in the spectrum folder.\n'
                'Move the image file into the same folder as the spectrum first.')
            return

        success = self.produce_cake()
        if not success:
            return
        temp_dir = get_temp_dir(self.model.get_base_ptn_filename())
        self.model.diff_img.write_temp_cakefiles(temp_dir=temp_dir)
        self._set_image_file_box()
        self._apply_changes_to_graph()

    def _set_row_roi_spin_limits(self):
        if (not hasattr(self.widget, "spinBox_CCDRowMin")) or \
                (not self.model.diff_img_exist()):
            return
        img = getattr(self.model.diff_img, "img", None)
        if img is None or np.ndim(img) < 2 or img.shape[0] <= 0:
            return
        n_rows = int(img.shape[0])
        for box in (self.widget.spinBox_CCDRowMin, self.widget.spinBox_CCDRowMax):
            box.blockSignals(True)
            box.setMaximum(n_rows - 1)
            box.blockSignals(False)
        current = getattr(self.model.base_ptn, "row_roi", None)
        if current is None:
            current = (0, n_rows - 1)
        row_min = max(0, min(int(current[0]), n_rows - 1))
        row_max = max(0, min(int(current[1]), n_rows - 1))
        if row_max < row_min:
            row_min, row_max = row_max, row_min
        self.widget.spinBox_CCDRowMin.blockSignals(True)
        self.widget.spinBox_CCDRowMax.blockSignals(True)
        self.widget.spinBox_CCDRowMin.setValue(row_min)
        self.widget.spinBox_CCDRowMax.setValue(row_max)
        self.widget.spinBox_CCDRowMin.blockSignals(False)
        self.widget.spinBox_CCDRowMax.blockSignals(False)

    def _apply_row_roi_to_spectrum(self):
        if (not self.model.base_ptn_exist()) or \
                (extract_extension(str(self.model.get_base_ptn_filename())).lower() != 'spe'):
            return
        if (not hasattr(self.widget, "spinBox_CCDRowMin")) or \
                (getattr(self.model.base_ptn, "raw_image", None) is None):
            return
        row_min = int(self.widget.spinBox_CCDRowMin.value())
        row_max = int(self.widget.spinBox_CCDRowMax.value())
        if row_max < row_min:
            row_min, row_max = row_max, row_min
            self.widget.spinBox_CCDRowMin.blockSignals(True)
            self.widget.spinBox_CCDRowMax.blockSignals(True)
            self.widget.spinBox_CCDRowMin.setValue(row_min)
            self.widget.spinBox_CCDRowMax.setValue(row_max)
            self.widget.spinBox_CCDRowMin.blockSignals(False)
            self.widget.spinBox_CCDRowMax.blockSignals(False)
        success = self.model.base_ptn.set_spe_row_roi(row_min, row_max)
        if not success:
            return
        self._refresh_bgsub_for_current_spectrum()
        self._apply_changes_to_graph()

    def _refresh_bgsub_for_current_spectrum(self):
        if not self.model.base_ptn_exist():
            return
        x_raw, __ = self.model.base_ptn.get_raw()
        if x_raw is None or len(x_raw) == 0:
            return
        roi_min = float(self.widget.doubleSpinBox_Background_ROI_min.value())
        roi_max = float(self.widget.doubleSpinBox_Background_ROI_max.value())
        if (x_raw.min() >= roi_min) or (x_raw.max() <= roi_min):
            roi_min = float(x_raw.min())
            self.widget.doubleSpinBox_Background_ROI_min.setValue(roi_min)
        if (x_raw.max() <= roi_max) or (x_raw.min() >= roi_max):
            roi_max = float(x_raw.max())
            self.widget.doubleSpinBox_Background_ROI_max.setValue(roi_max)
        params = [
            int(self.widget.spinBox_BGParam1.value()),
        ]
        x_raw, y_raw = self.model.base_ptn.get_raw()
        __, y_fit = self.plot_ctrl._get_smoothed_pattern_xy(x_raw, y_raw)
        if not self.plot_ctrl._smoothing_active():
            y_fit = y_raw
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
        self.model.base_ptn.get_chbg(
            [roi_min, roi_max], params, yshift=0,
            fit_areas=fit_areas, y_source=y_fit)

    def _on_row_roi_spin_changed(self, value):
        del value
        self._apply_row_roi_to_spectrum()

    def _toggle_row_roi_selector(self, checked):
        if checked:
            self._activate_row_roi_selector()
        else:
            self._deactivate_row_roi_selector()

    def _activate_row_roi_selector(self):
        if not self.model.diff_img_exist():
            self.widget.pushButton_CCDSelectRoi.setChecked(False)
            return
        if not hasattr(self.widget.mpl.canvas, "ax_cake"):
            self.widget.pushButton_CCDSelectRoi.setChecked(False)
            return
        self._deactivate_row_roi_selector()
        toolbar = getattr(self.widget.mpl.canvas, "toolbar", None)
        if toolbar is not None:
            try:
                if getattr(toolbar, "mode", "") == 'zoom rect':
                    toolbar.zoom()
                elif getattr(toolbar, "mode", "") == 'pan/zoom':
                    toolbar.pan()
            except Exception:
                pass
        canvas = self.widget.mpl.canvas
        self._row_roi_press_y = None
        try:
            canvas.setCursor(QtWidgets.QCursor(QtCore.Qt.CrossCursor))
        except Exception:
            pass
        self._row_roi_press_cid = canvas.mpl_connect(
            "button_press_event", self._on_row_roi_press)
        self._row_roi_motion_cid = canvas.mpl_connect(
            "motion_notify_event", self._on_row_roi_motion)
        self._row_roi_release_cid = canvas.mpl_connect(
            "button_release_event", self._on_row_roi_release)
        if hasattr(self.widget, "pushButton_CCDSelectRoi"):
            self.widget.pushButton_CCDSelectRoi.blockSignals(True)
            self.widget.pushButton_CCDSelectRoi.setChecked(True)
            self.widget.pushButton_CCDSelectRoi.blockSignals(False)
        try:
            self.widget.mpl.canvas.draw_idle()
        except Exception:
            pass

    def _deactivate_row_roi_selector(self):
        canvas = getattr(self.widget, "mpl", None)
        canvas = getattr(canvas, "canvas", None)
        if canvas is not None:
            try:
                if self._row_roi_press_cid is not None:
                    canvas.mpl_disconnect(self._row_roi_press_cid)
            except Exception:
                pass
            try:
                if self._row_roi_motion_cid is not None:
                    canvas.mpl_disconnect(self._row_roi_motion_cid)
            except Exception:
                pass
            try:
                if self._row_roi_release_cid is not None:
                    canvas.mpl_disconnect(self._row_roi_release_cid)
            except Exception:
                pass
        if self._row_roi_preview is not None:
            try:
                self._row_roi_preview.remove()
            except Exception:
                pass
        self._row_roi_selector = None
        self._row_roi_press_cid = None
        self._row_roi_motion_cid = None
        self._row_roi_release_cid = None
        self._row_roi_press_y = None
        self._row_roi_preview = None
        canvas = getattr(self.widget, "mpl", None)
        canvas = getattr(canvas, "canvas", None)
        if canvas is not None:
            try:
                canvas.unsetCursor()
            except Exception:
                pass
        if hasattr(self.widget, "pushButton_CCDSelectRoi"):
            self.widget.pushButton_CCDSelectRoi.blockSignals(True)
            self.widget.pushButton_CCDSelectRoi.setChecked(False)
            self.widget.pushButton_CCDSelectRoi.blockSignals(False)

    def _on_row_roi_press(self, event):
        ax = getattr(self.widget.mpl.canvas, "ax_cake", None)
        if event.inaxes != ax:
            return
        if event.button != 1 or event.ydata is None:
            return
        self._row_roi_press_y = float(event.ydata)
        self._update_row_roi_preview(float(event.ydata))

    def _on_row_roi_motion(self, event):
        if self._row_roi_press_y is None:
            return
        ax = getattr(self.widget.mpl.canvas, "ax_cake", None)
        y_now = self._event_y_to_cake_row(event, ax)
        if y_now is None:
            return
        self._set_row_roi_spin_values_from_pixels(self._row_roi_press_y, y_now)
        self._update_row_roi_preview(float(y_now))

    def _on_row_roi_release(self, event):
        if self._row_roi_press_y is None:
            return
        ax = getattr(self.widget.mpl.canvas, "ax_cake", None)
        if event.button != 1:
            self._deactivate_row_roi_selector()
            return
        y_release = self._event_y_to_cake_row(event, ax)
        if y_release is None:
            self._deactivate_row_roi_selector()
            return
        img = getattr(self.model.diff_img, "img", None)
        if img is None or np.ndim(img) < 2:
            self._deactivate_row_roi_selector()
            return
        self._set_row_roi_spin_values_from_pixels(self._row_roi_press_y, y_release)
        self._apply_row_roi_to_spectrum()
        self._deactivate_row_roi_selector()

    def _event_y_to_cake_row(self, event, ax):
        if ax is None:
            return None
        if event.ydata is not None and event.inaxes == ax:
            return float(event.ydata)
        if event.y is None:
            return None
        try:
            __, ydata = ax.transData.inverted().transform((event.x, event.y))
            return float(ydata)
        except Exception:
            return None

    def _update_row_roi_preview(self, y_current):
        ax = getattr(self.widget.mpl.canvas, "ax_cake", None)
        if ax is None:
            return
        try:
            x0, x1 = ax.get_xlim()
        except Exception:
            return
        y0 = min(float(self._row_roi_press_y), float(y_current))
        y1 = max(float(self._row_roi_press_y), float(y_current))
        if self._row_roi_preview is None:
            self._row_roi_preview = patches.Rectangle(
                (x0, y0),
                (x1 - x0),
                max(y1 - y0, 1.0),
                linewidth=1.5,
                edgecolor="#00e676",
                facecolor="#00e676",
                alpha=0.18,
                linestyle="--",
            )
            ax.add_patch(self._row_roi_preview)
        else:
            self._row_roi_preview.set_x(x0)
            self._row_roi_preview.set_y(y0)
            self._row_roi_preview.set_width(x1 - x0)
            self._row_roi_preview.set_height(max(y1 - y0, 1.0))
        try:
            self.widget.mpl.canvas.draw_idle()
        except Exception:
            pass

    def _set_row_roi_spin_values_from_pixels(self, y0, y1):
        img = getattr(self.model.diff_img, "img", None)
        if img is None or np.ndim(img) < 2:
            return
        ymin = int(np.floor(min(float(y0), float(y1))))
        ymax = int(np.ceil(max(float(y0), float(y1))))
        ymax = max(ymin, ymax - 1)
        n_rows = int(img.shape[0])
        ymin = max(0, min(ymin, n_rows - 1))
        ymax = max(0, min(ymax, n_rows - 1))
        self.widget.spinBox_CCDRowMin.blockSignals(True)
        self.widget.spinBox_CCDRowMax.blockSignals(True)
        self.widget.spinBox_CCDRowMin.setValue(ymin)
        self.widget.spinBox_CCDRowMax.setValue(ymax)
        self.widget.spinBox_CCDRowMin.blockSignals(False)
        self.widget.spinBox_CCDRowMax.blockSignals(False)

    def _reset_row_roi_to_full(self):
        img = getattr(getattr(self.model, "diff_img", None), "img", None)
        if img is None or np.ndim(img) < 2 or img.shape[0] <= 0:
            return
        self.widget.spinBox_CCDRowMin.blockSignals(True)
        self.widget.spinBox_CCDRowMax.blockSignals(True)
        self.widget.spinBox_CCDRowMin.setValue(0)
        self.widget.spinBox_CCDRowMax.setValue(int(img.shape[0] - 1))
        self.widget.spinBox_CCDRowMin.blockSignals(False)
        self.widget.spinBox_CCDRowMax.blockSignals(False)
        self._apply_row_roi_to_spectrum()

    """
    def load_cake_format_file(self):
        # get filename
        temp_dir = get_temp_dir(self.model.get_base_ptn_filename())
        filen = QtWidgets.QFileDialog.getOpenFileName(
            self.widget, "Open a cake format File", temp_dir,  # self.model.chi_path,
            "Data files (*.cakeformat)")[0]
        if filen == '':
            return
        temp_values = []
        with open(filen, "r") as f:
            for line in f:
                temp_values.append(float(line.split(':')[1]))
        self.widget.spinBox_AziShift.setValue(temp_values[0])
        self.widget.spinBox_MaxCakeScale.setValue(temp_values[1])
        self.widget.horizontalSlider_VMin.setValue(temp_values[2])
        self.widget.horizontalSlider_VMax.setValue(temp_values[3])
        self.widget.horizontalSlider_MaxScaleBars.setValue(temp_values[4])
        self._apply_changes_to_graph()

    def save_cake_format_file(self):
        # make filename
        temp_dir = get_temp_dir(self.model.get_base_ptn_filename())
        ext = "cakeformat"
        #filen_t = self.model.make_filename(ext)
        filen_t = make_filename(self.model.base_ptn.fname, ext,
                                temp_dir=temp_dir)
        filen = dialog_savefile(self.widget, filen_t)
        if str(filen) == '':
            return
        # save cake related Values
        names = ['azi_shift', 'int_max', 'min_bar', 'max_bar', 'scale_bar']
        values = [self.widget.spinBox_AziShift.value(),
                  self.widget.spinBox_MaxCakeScale.value(),
                  self.widget.horizontalSlider_VMin.value(),
                  self.widget.horizontalSlider_VMax.value(),
                  self.widget.horizontalSlider_MaxScaleBars.value()]

        with open(filen, "w") as f:
            for n, v in zip(names, values):
                f.write(n + ' : ' + str(v) + '\n')
    """

    def reset_max_cake_scale(self):
        if hasattr(self.widget, "checkBox_Diff") and self.widget.checkBox_Diff.isChecked():
            return
        intensity_cake, _, _ = self.model.diff_img.get_cake()
        arr = np.asarray(intensity_cake, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return
        vmin = float(np.min(arr))
        vmax = float(np.max(arr))
        self._set_cake_scale_spinboxes(vmin, vmax)
        self._apply_changes_to_graph()

    def _apply_changes_to_graph(self):
        self.plot_ctrl.update()

    def _set_cake_bound_from_hist(self, bound_type, intensity_value):
        if bound_type == "min":
            self.widget.doubleSpinBox_CCDScaleMin.setValue(float(intensity_value))
        elif bound_type == "max":
            self.widget.doubleSpinBox_CCDScaleMax.setValue(float(intensity_value))

    def _set_cake_scale_spinboxes(self, vmin, vmax):
        if not hasattr(self.widget, "doubleSpinBox_CCDScaleMin"):
            return
        if vmax < vmin:
            vmin, vmax = vmax, vmin
        self.widget.doubleSpinBox_CCDScaleMin.blockSignals(True)
        self.widget.doubleSpinBox_CCDScaleMax.blockSignals(True)
        self.widget.doubleSpinBox_CCDScaleMin.setValue(float(vmin))
        self.widget.doubleSpinBox_CCDScaleMax.setValue(float(vmax))
        self.widget.doubleSpinBox_CCDScaleMin.blockSignals(False)
        self.widget.doubleSpinBox_CCDScaleMax.blockSignals(False)

    def _on_cake_scale_spin_changed(self, value):
        del value
        if not hasattr(self.widget, "doubleSpinBox_CCDScaleMin"):
            return
        vmin = float(self.widget.doubleSpinBox_CCDScaleMin.value())
        vmax = float(self.widget.doubleSpinBox_CCDScaleMax.value())
        if vmax <= vmin:
            vmax = vmin + max(1e-6, 1e-6 * max(abs(vmin), 1.0))
            self.widget.doubleSpinBox_CCDScaleMax.blockSignals(True)
            self.widget.doubleSpinBox_CCDScaleMax.setValue(vmax)
            self.widget.doubleSpinBox_CCDScaleMax.blockSignals(False)
        self._apply_changes_to_graph()

    def _ignore_raw_data_missing(self):
        return True

    def _set_image_file_box_missing(self):
        self.widget.textEdit_DiffractionImageFilename.setText(
            'CCD image file is missing. Move the image file into the same folder as the spectrum.')

    def _set_image_file_box(self):
        if self.model.diff_img_exist() and (self.model.diff_img.img_filename is not None):
            self.widget.textEdit_DiffractionImageFilename.setText(
                self.model.diff_img.img_filename)
        else:
            self._set_image_file_box_missing()

    def _warn_cannot_process_cake(self):
        QtWidgets.QMessageBox.warning(
            self.widget, 'Warning',
            'Rampo cannot process the CCD view: no image data and no cached CCD files were found.')

    def _load_cake_from_temp_without_raw_image(self, temp_dir):
        # Temp cake files are named from the base pattern root name.
        if self.model.diff_img is None:
            self.model.reset_diff_img()
        self.model.diff_img.img_filename = self.model.make_filename(
            'tif', original=True)
        return self.model.diff_img.read_cake_from_tempfile(temp_dir=temp_dir)

    def show_tif_header(self):
        if not self.model.base_ptn_exist():
            return
        filen_tif = self.model.make_filename('tif', original=True)
        filen_tiff = self.model.make_filename('tiff', original=True)
        if not (os.path.exists(filen_tif) or
                os.path.exists(filen_tiff)):
            QtWidgets.QMessageBox.warning(
                self.widget, 'Warning',
                'Cannot find image file: %s or %s in the chi folder.' %
                (filen_tif, filen_tiff))
        else:
            textoutput = ''
            if os.path.exists(filen_tif):
                f = filen_tif
            else:
                f = filen_tiff
            metadata = {}
            with Image.open(f) as img:
                for key in img.tag:
                    metadata[key] = img.tag[key]
            infobox = InformationBox()
            infobox.setText(json.dumps(metadata, indent=4))
            print(str(datetime.datetime.now())[:-7], ': TIF metadata\n', 
                json.dumps(metadata, indent=4))
            infobox.exec()
            #self.widget.plainTextEdit_ViewJCPDS.setPlainText(textoutput)


    def addremove_cake(self):
        """
        add / remove cake to the graph
        """
        update = self._addremove_cake()
        if update:
            self._apply_changes_to_graph()
    """
    def image_file_exists(self):
        # if no image file, no cake
        filen_tif = self.model.make_filename('tif', original=True)
        filen_tiff = self.model.make_filename('tiff', original=True)
        filen_mar3450 = self.model.make_filename('mar3450',
            original=True)
        filen_cbf = self.model.make_filename('cbf', original=True)
        filen_h5 = self.model.make_filename('h5', original=True)
        if not (os.path.exists(filen_tif) or
                os.path.exists(filen_tiff) or
                os.path.exists(filen_mar3450) or
                os.path.exists(filen_h5) or
                os.path.exists(filen_cbf)):
            return False
        else:
            return True
    """

    def _addremove_cake(self):
        """
        add / remove cake
        no signal to update_graph
        """
        # if no base ptn, no cake
        if not self.model.base_ptn_exist():
            QtWidgets.QMessageBox.warning(
                self.widget, 'Warning', 'Choose a spectrum file first.')
            return False
        if self._is_spe_source():
            if not self.process_temp_cake():
                return False
            return True
        if not self.model.associated_image_exists():
            if not self._ignore_raw_data_missing():
                self._set_image_file_box_missing()
                QtWidgets.QMessageBox.warning(
                    self.widget, 'Warning',
                    'Cannot find CCD image file.')
                return False
            if not self.process_temp_cake():
                self._warn_cannot_process_cake()
                return False
            return True

        if not self.process_temp_cake():
            self._warn_cannot_process_cake()
            return False
        return True

    def _load_new_image(self):
        """
        Load new image for cake view.  Cake should be the same as base pattern.
        no signal to update_graph
        """
        self.model.reset_diff_img()
        if not self.model.associated_image_exists():
            self._set_image_file_box_missing()
            return False
        self.model.load_associated_img()
        self._set_image_file_box()
        return True

    def _is_spe_source(self):
        if not self.model.base_ptn_exist():
            return False
        return extract_extension(str(self.model.get_base_ptn_filename())).lower() == 'spe'

    def apply_mask(self):
        # self.produce_cake()
        min_mask = float(self.widget.spinBox_MaskMin.value())
        max_mask = float(self.widget.spinBox_MaskMax.value())
        zrange = self.model.diff_img.get_img_zrange()
        print('img z range', zrange)
        print('mask range', min_mask, max_mask)
        if (zrange[0] < min_mask) or (zrange[1] > max_mask):
            # case for meaningful mask
            if self.widget.pushButton_ApplyMask.isChecked():
                self.ccdprocess_ctrl.cook()
        else:
            self.model.diff_img.set_mask(None)
        self._apply_changes_to_graph()

    def reset_maskrange(self):
        # get min and max of the cake image
        #intensity_cake, _, _ = self.model.diff_img.get_cake()
        zrange = self.model.diff_img.get_img_zrange()
        if zrange != None:
            # push those values to spinboxes
            self.widget.spinBox_MaskMin.setValue(int(zrange[0]))
            self.widget.spinBox_MaskMax.setValue(int(zrange[1]))
            self.model.diff_img.set_mask(None)
            self._apply_changes_to_graph()
        # reprocess the image
        # self.apply_mask()

    def produce_cake(self):
        """
        Reprocess to get cake.  Slower re - processing
        does not signal to update_graph
        """
        success = self._load_new_image()
        if not success:
            return False
        self._set_row_roi_spin_limits()
        self.ccdprocess_ctrl.cook()
        self._apply_row_roi_to_spectrum()
        self._set_image_file_box()
        return True

    def process_temp_cake(self):
        """
        load cake through either temporary file or make a new cake
        """
        temp_dir = get_temp_dir(self.model.get_base_ptn_filename())
        if self._is_spe_source():
            success = self._load_new_image()
            if not success:
                return False
            self._set_row_roi_spin_limits()
            self._apply_row_roi_to_spectrum()
            self.model.diff_img.write_temp_cakefiles(temp_dir=temp_dir)
            return True
        has_raw_image = self.model.associated_image_exists()
        if not has_raw_image:
            self._set_image_file_box_missing()
            if not self._ignore_raw_data_missing():
                QtWidgets.QMessageBox.warning(
                    self.widget, "Warning",
                    "Image file for the base pattern does not exist.")
                return False
            return self._load_cake_from_temp_without_raw_image(temp_dir)
        #temp_dir = os.path.join(self.model.chi_path, 'temporary_pkpo')
        success = self._load_new_image()
        if not success:
            return False
        success = self.model.diff_img.read_cake_from_tempfile(
            temp_dir=temp_dir)
        if success:
            print(str(datetime.datetime.now())[:-7],
                ": Load cake image from temporary file.")
            return True
        print(str(datetime.datetime.now())[:-7],
            ": Create new temporary file for cake image.")
        self._update_temp_cake_files(temp_dir)
        return True

    def _update_temp_cake_files(self, temp_dir):
        success = self.produce_cake()
        if not success:
            return
        self.model.diff_img.write_temp_cakefiles(temp_dir=temp_dir)

    def temp_cake_exists(self):
        temp_dir = get_temp_dir(self.model.get_base_ptn_filename())
        search_pattern = os.path.join(temp_dir, "*.cake.npy")
        cake_all = glob.glob(search_pattern)
        print(len(cake_all))
        if len(cake_all) < 3:
            return False
        else:
            return True


CakeController = CCDController
