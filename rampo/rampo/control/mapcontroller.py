import os
import math
import re
import numpy as np
from qtpy import QtWidgets, QtCore
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.widgets import RectangleSelector
from matplotlib import colors as mcolors
import matplotlib.patches as mpatches

from .ramaniohelpers import (
    load_spectrum_xy,
)
from ..ds_ramspec import Spectrum


class MapController(object):
    def __init__(self, model, widget):
        self.model = model
        self.widget = widget
        self.base_spectrum_ctrl = None
        self.plot_ctrl = None

        self._map_canvas = None
        self._map_ax = None
        self._map_cax = None
        self._map_cbar = None
        self._map_im = None

        self._selector_1d = None
        self._selector_2d = None
        self._roi_artist_1d = None
        self._roi_artist_2d = None

        self._chi_files = []
        self._pos_idx = []
        self._lin_to_file = {}
        self._chi_cache = {}
        self._grid = (1, 1)
        self._roi_1d = None
        self._map_data = None

        self._sync_ui_from_roi = False

        self._build_canvas()
        self._connect_channel()

    def set_helpers(self, base_spectrum_ctrl=None, base_ptn_ctrl=None, plot_ctrl=None):
        self.base_spectrum_ctrl = base_spectrum_ctrl or base_ptn_ctrl
        self.plot_ctrl = plot_ctrl

    def _build_canvas(self):
        if not hasattr(self.widget, "verticalLayout_MapCanvas"):
            return
        self._map_fig = Figure()
        self._map_fig.subplots_adjust(left=0.005, right=0.94, top=0.985, bottom=0.03)
        self._recreate_map_axes()
        self._map_canvas = FigureCanvasQTAgg(self._map_fig)
        self.widget.verticalLayout_MapCanvas.addWidget(self._map_canvas, 1)
        self._map_canvas.mpl_connect("button_press_event", self._on_map_click)

    def _recreate_map_axes(self):
        if getattr(self, "_map_fig", None) is None:
            return
        self._map_fig.clf()
        self._map_ax = self._map_fig.add_subplot(111)
        self._map_cax = None
        self._map_cbar = None

    def _ensure_map_axes(self):
        if self._map_ax is None:
            self._recreate_map_axes()
            return
        # Axes can become detached after colorbar removal in some mpl states.
        ax_fig = getattr(self._map_ax, "figure", None)
        if (ax_fig is None) or (ax_fig is not self._map_fig):
            self._recreate_map_axes()

    def _connect_channel(self):
        if not hasattr(self.widget, "tabWidget"):
            return
        self.widget.tabWidget.currentChanged.connect(self._on_main_tab_changed)

        self.widget.pushButton_MapLoadChi.clicked.connect(self._load_spectrum_files)
        self.widget.spinBox_MapNx.valueChanged.connect(self._on_grid_changed)
        self.widget.spinBox_MapNy.valueChanged.connect(self._on_grid_changed)
        self.widget.comboBox_MapOrder.currentIndexChanged.connect(self._on_grid_changed)

        self.widget.pushButton_MapSetRoi.clicked.connect(self._arm_roi_selection)
        self.widget.pushButton_MapClearRoi.clicked.connect(self._clear_roi)
        self.widget.pushButton_MapCompute.clicked.connect(self._compute_map)

        self.widget.comboBox_MapCmap.currentIndexChanged.connect(self._draw_map)
        self.widget.checkBox_MapReverseCmap.stateChanged.connect(self._draw_map)
        self.widget.doubleSpinBox_MapVmin.valueChanged.connect(self._draw_map)
        self.widget.doubleSpinBox_MapVmax.valueChanged.connect(self._draw_map)
        self.widget.checkBox_MapLog.stateChanged.connect(self._draw_map)

        self.widget.pushButton_MapScaleAuto.clicked.connect(self._auto_scale)
        self.widget.pushButton_MapScalePercentile.clicked.connect(self._scale_percentile)
        self.widget.pushButton_MapScaleReset.clicked.connect(self._scale_reset)

        self.widget.pushButton_MapExportImage.clicked.connect(self._export_image)
        self.widget.pushButton_MapExportNpy.clicked.connect(self._export_npy)

    def _on_main_tab_changed(self, _idx):
        if not hasattr(self.widget, "tab_Map"):
            return
        if self.widget.tabWidget.currentWidget() != self.widget.tab_Map:
            self.deactivate_interactions()
            self._clear_roi_overlays()
        else:
            self.refresh_roi_overlays()

    def deactivate_interactions(self):
        self._disable_roi_selectors()
        if hasattr(self.widget, "pushButton_MapSetRoi"):
            self.widget.pushButton_MapSetRoi.setChecked(False)

    def is_roi_selection_active(self):
        sel_1d_active = (self._selector_1d is not None) and \
            bool(getattr(self._selector_1d, "active", False))
        return bool(sel_1d_active)

    def _set_status(self, msg):
        if hasattr(self.widget, "label_MapStatus"):
            self.widget.label_MapStatus.setText(str(msg))

    def _set_loaded_count(self):
        if hasattr(self.widget, "label_MapLoaded"):
            self.widget.label_MapLoaded.setText(f"Loaded: {len(self._chi_files)}")

    def _load_spectrum_files(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self.widget,
            "Select SPE files for map",
            self.model.chi_path,
            "Spectra (*.spe *.SPE *.chi)",
        )
        if not files:
            return

        self._chi_files = sorted(list(files), key=self._filename_sort_key)
        self._pos_idx = self._derive_position_indices(self._chi_files)
        self._rebuild_linear_lookup()
        self._chi_cache = {}
        self._roi_1d = None
        self._map_data = None

        nx, ny = self._guess_grid_dims(len(self._chi_files))
        self._sync_ui_from_roi = True
        self.widget.spinBox_MapNx.setValue(nx)
        self.widget.spinBox_MapNy.setValue(ny)
        self._sync_ui_from_roi = False
        self._grid = (nx, ny)
        self._set_loaded_count()
        self._set_status(f"Loaded {len(self._chi_files)} spectra. Guessed grid: {nx} x {ny}")

        self._preview_center_file()
        self._set_default_1d_full_range_roi()
        if self._roi_1d is not None:
            self._compute_map()
        else:
            self._draw_map()

    def _guess_grid_dims(self, n_files):
        if n_files <= 0:
            return 1, 1
        best = (n_files, 1)
        best_score = abs(n_files - 1)
        root = int(math.sqrt(n_files))
        for a in range(1, root + 1):
            if (n_files % a) != 0:
                continue
            b = n_files // a
            score = abs(b - a)
            if score < best_score:
                best = (b, a)
                best_score = score
        return int(best[0]), int(best[1])

    def _filename_sort_key(self, filename):
        name = os.path.splitext(os.path.basename(filename))[0]
        name_lower = name.lower()
        map_match = re.search(r"map_(\d+)", name_lower)
        if map_match:
            return (
                0,
                name_lower[:map_match.start()],
                int(map_match.group(1)),
                name_lower[map_match.end():],
            )
        tail_match = re.search(r"(\d+)$", name_lower)
        if tail_match:
            return (1, name_lower[:tail_match.start()], int(tail_match.group(1)))
        return (2, name_lower)

    def _derive_position_indices(self, files):
        nums = []
        ok = True
        for f in files:
            b = os.path.basename(f).lower()
            m = re.search(r"map_(\d+)", b)
            if m is None:
                ok = False
                break
            nums.append(int(m.group(1)))
        if not ok:
            return list(range(len(files)))
        min_num = min(nums)
        return [n - min_num for n in nums]

    def _rebuild_linear_lookup(self):
        self._lin_to_file = {}
        for i, lin in enumerate(self._pos_idx):
            if lin not in self._lin_to_file:
                self._lin_to_file[lin] = i

    def _on_grid_changed(self):
        if self._sync_ui_from_roi:
            return
        nx = int(self.widget.spinBox_MapNx.value())
        ny = int(self.widget.spinBox_MapNy.value())
        self._grid = (nx, ny)
        self._preview_center_file()
        n = len(self._chi_files)
        if n > 0 and (nx * ny != n):
            self._set_status(
                f"Grid mismatch: Nx*Ny ({nx}x{ny}={nx * ny}) must equal loaded files ({n})."
            )
        self._draw_map()

    def _set_default_1d_full_range_roi(self):
        if not self._chi_files:
            return
        try:
            x, __ = self._load_processed_xy(self._chi_files[0])
            if x.size == 0:
                return
            xmin = float(np.nanmin(x))
            xmax = float(np.nanmax(x))
            if (not np.isfinite(xmin)) or (not np.isfinite(xmax)) or (xmax <= xmin):
                return
            self._roi_1d = (xmin, xmax)
            self.widget.lineEdit_MapRoiSummary.setText(
                f"Shift [{xmin:.3f}, {xmax:.3f}]"
            )
        except Exception:
            pass

    def _preview_center_file(self):
        if not self._chi_files:
            return
        nx, ny = self._grid
        n = len(self._chi_files)
        if nx * ny != n:
            self._set_status(
                f"Grid mismatch: Nx*Ny ({nx}x{ny}={nx * ny}) must equal loaded files ({n})."
            )
            return
        cx = nx // 2
        cy = ny // 2
        idx = self._grid_to_linear(cx, cy)
        if idx is None:
            return
        file_idx = self._lin_to_file.get(idx, None)
        if file_idx is None:
            return
        self._load_file_to_main_plot(file_idx)

    def _load_file_to_main_plot(self, idx):
        if self.base_spectrum_ctrl is None:
            return
        if (idx < 0) or (idx >= len(self._chi_files)):
            return
        try:
            self.base_spectrum_ctrl._load_a_new_pattern(self._chi_files[idx])
            if self.plot_ctrl is not None:
                self.plot_ctrl.update()
            self._schedule_overlay_refresh()
        except Exception as exc:
            self._set_status(f"Failed to preview file: {exc}")

    def _schedule_overlay_refresh(self):
        # Some redraw paths are deferred; refresh twice to survive delayed clears.
        self.refresh_roi_overlays()
        QtCore.QTimer.singleShot(0, self.refresh_roi_overlays)
        QtCore.QTimer.singleShot(80, self.refresh_roi_overlays)

    def _grid_to_linear(self, x, y):
        nx, ny = self._grid
        if (x < 0) or (y < 0) or (x >= nx) or (y >= ny):
            return None
        order = self.widget.comboBox_MapOrder.currentText()
        if order == "Snake":
            if (y % 2) == 0:
                i = y * nx + x
            else:
                i = y * nx + (nx - 1 - x)
        else:
            i = y * nx + x
        return int(i)

    def _linear_to_grid(self, idx):
        nx, _ny = self._grid
        order = self.widget.comboBox_MapOrder.currentText()
        y = idx // nx
        x = idx % nx
        if order == "Snake" and ((y % 2) == 1):
            x = nx - 1 - x
        return int(x), int(y)

    def _arm_roi_selection(self):
        if self.widget.tabWidget.currentWidget() != self.widget.tab_Map:
            self._set_status("Open Map tab first.")
            return
        self._disable_roi_selectors()
        self.widget.pushButton_MapSetRoi.setChecked(True)
        self._selector_1d = RectangleSelector(
            self.widget.mpl.canvas.ax_pattern,
            self._on_roi_1d_selected,
            useblit=True,
            button=[1],
            interactive=False,
            drag_from_anywhere=False,
        )
        self._set_status("Draw ROI on the spectrum to define the map range.")

    def _disable_roi_selectors(self):
        if self._selector_1d is not None:
            try:
                self._selector_1d.set_active(False)
            except Exception:
                pass
            self._selector_1d = None

    def _clear_roi(self):
        self._roi_1d = None
        self.widget.lineEdit_MapRoiSummary.setText("")
        self._set_status("ROI cleared.")
        self.deactivate_interactions()
        self._clear_roi_overlays()

    def _on_roi_1d_selected(self, eclick, erelease):
        if (eclick.xdata is None) or (erelease.xdata is None):
            return
        xmin = min(float(eclick.xdata), float(erelease.xdata))
        xmax = max(float(eclick.xdata), float(erelease.xdata))
        if xmax <= xmin:
            return
        self._roi_1d = (xmin, xmax)
        self.widget.lineEdit_MapRoiSummary.setText(
            f"Shift [{xmin:.3f}, {xmax:.3f}]"
        )
        self._set_status("1D ROI selected.")
        self.deactivate_interactions()
        self.refresh_roi_overlays()
        self._compute_map()

    def _load_spectrum_xy(self, spectrum_path):
        return load_spectrum_xy(spectrum_path, self._chi_cache)

    def _load_processed_xy(self, spectrum_path):
        spectrum = Spectrum(spectrum_path)
        if str(spectrum_path).lower().endswith(".spe"):
            spectrum.apply_excitation_wavelength(
                float(self.widget.doubleSpinBox_SetWavelength.value()))
            if hasattr(self.widget, "spinBox_CCDRowMin") and hasattr(self.widget, "spinBox_CCDRowMax"):
                spectrum.set_spe_row_roi(
                    int(self.widget.spinBox_CCDRowMin.value()),
                    int(self.widget.spinBox_CCDRowMax.value()))
        use_bgsub = bool(
            getattr(self.widget, "checkBox_BgSub", None) and
            self.widget.checkBox_BgSub.isChecked()
        )
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
        if use_bgsub:
            x_raw, __ = spectrum.get_raw()
            x_raw = np.asarray(x_raw, dtype=float)
            if x_raw.size == 0:
                return x_raw, np.asarray([], dtype=float)
            roi_min = float(self.widget.doubleSpinBox_Background_ROI_min.value())
            roi_max = float(self.widget.doubleSpinBox_Background_ROI_max.value())
            roi_min = max(float(np.nanmin(x_raw)), roi_min)
            roi_max = min(float(np.nanmax(x_raw)), roi_max)
            if roi_max <= roi_min:
                roi_min = float(np.nanmin(x_raw))
                roi_max = float(np.nanmax(x_raw))
            y_fit = spectrum.get_raw()[1]
            if (self.plot_ctrl is not None) and \
                    bool(getattr(self.plot_ctrl, "_smoothing_active", lambda: False)()):
                __, y_fit = self.plot_ctrl._get_smoothed_pattern_xy(x_raw, y_fit)
            spectrum.get_chbg(
                [roi_min, roi_max],
                [int(self.widget.spinBox_BGParam1.value())],
                yshift=0,
                fit_areas=fit_areas,
                y_source=y_fit,
            )
            x, y = spectrum.get_bgsub()
        else:
            x, y = spectrum.get_raw()
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            if (self.plot_ctrl is not None) and \
                    bool(getattr(self.plot_ctrl, "_smoothing_active", lambda: False)()):
                x, y = self.plot_ctrl._get_smoothed_pattern_xy(x, y)
            return x, y
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if not use_bgsub and (self.plot_ctrl is not None) and \
                bool(getattr(self.plot_ctrl, "_smoothing_active", lambda: False)()):
            x, y = self.plot_ctrl._get_smoothed_pattern_xy(x, y)
        return x, y

    def _compute_map(self):
        if not self._chi_files:
            self._set_status("Load SPE files first.")
            return
        nx, ny = self._grid
        n = len(self._chi_files)
        if nx * ny != n:
            self._set_status(
                f"Grid mismatch: Nx*Ny ({nx}x{ny}={nx * ny}) must equal loaded files ({n})."
            )
            return
        if self._roi_1d is None:
            self._set_status("Select ROI first.")
            return

        values = np.full(n, np.nan, dtype=float)
        failures = []
        for i, chi_path in enumerate(self._chi_files):
            try:
                x, y = self._load_processed_xy(chi_path)
                xmin, xmax = self._roi_1d
                m = (x >= xmin) & (x <= xmax)
                if not np.any(m):
                    values[i] = np.nan
                else:
                    values[i] = float(np.nansum(y[m]))
            except Exception as exc:
                failures.append((chi_path, str(exc)))
                values[i] = np.nan

        grid = np.full((ny, nx), np.nan, dtype=float)
        for i, val in enumerate(values):
            lin_pos = self._pos_idx[i] if i < len(self._pos_idx) else i
            x, y = self._linear_to_grid(lin_pos)
            if (y < ny) and (x < nx):
                grid[y, x] = val
        self._map_data = grid

        # Recomputing map should always refresh Min/Max from new data.
        self._scale_reset()
        self._draw_map()

        if failures:
            first_path, first_err = failures[0]
            first_name = os.path.basename(first_path)
            self._set_status(
                f"Map computed with {len(failures)} failures. "
                f"First: {first_name} ({first_err})")
        else:
            self._set_status("Map computed from processed spectrum intensity in ROI.")
        self._schedule_overlay_refresh()

    def _effective_cmap(self):
        cmap = str(self.widget.comboBox_MapCmap.currentText())
        if self.widget.checkBox_MapReverseCmap.isChecked() and (not cmap.endswith("_r")):
            cmap = cmap + "_r"
        return cmap

    def _current_vrange(self, data):
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return None
        return float(np.nanmin(finite)), float(np.nanmax(finite))

    def _scale_reset(self):
        if self._map_data is None:
            return
        vr = self._current_vrange(self._map_data)
        if vr is None:
            return
        self.widget.doubleSpinBox_MapVmin.setValue(vr[0])
        self.widget.doubleSpinBox_MapVmax.setValue(vr[1])

    def _auto_scale(self):
        self._scale_reset()
        self._draw_map()

    def _scale_percentile(self):
        if self._map_data is None:
            return
        finite = self._map_data[np.isfinite(self._map_data)]
        if finite.size == 0:
            return
        p_low = float(self.widget.doubleSpinBox_MapPctLow.value())
        p_high = float(self.widget.doubleSpinBox_MapPctHigh.value())
        if p_high <= p_low:
            self._set_status("Percentile high must be larger than low.")
            return
        vmin = float(np.nanpercentile(finite, p_low))
        vmax = float(np.nanpercentile(finite, p_high))
        self.widget.doubleSpinBox_MapVmin.setValue(vmin)
        self.widget.doubleSpinBox_MapVmax.setValue(vmax)
        self._draw_map()

    def _draw_map(self):
        # Recreate axes each draw so colorbar layout adjustments do not accumulate.
        self._recreate_map_axes()
        if self._map_ax is None:
            return
        self._map_ax.clear()
        self._map_cbar = None
        if self._map_data is None:
            self._map_ax.set_axis_off()
            self._map_canvas.draw_idle()
            return

        data = np.array(self._map_data, copy=True)
        vmin = float(self.widget.doubleSpinBox_MapVmin.value())
        vmax = float(self.widget.doubleSpinBox_MapVmax.value())

        norm = None
        label = "Integrated intensity"
        if self.widget.checkBox_MapLog.isChecked():
            finite = data[np.isfinite(data)]
            if finite.size == 0 or np.nanmax(finite) <= 0:
                self._set_status("Log scale disabled: map has no positive values.")
                self.widget.checkBox_MapLog.setChecked(False)
            else:
                min_pos = np.nanmin(finite[finite > 0]) if np.any(finite > 0) else np.nan
                if (not np.isfinite(min_pos)):
                    self._set_status("Log scale disabled: map has no positive values.")
                    self.widget.checkBox_MapLog.setChecked(False)
                else:
                    data[data <= 0] = np.nan
                    if vmin <= 0:
                        vmin = min_pos
                    vmax = max(vmax, vmin * 1.001)
                    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
                    label = "log10(Integrated intensity)"

        im_kwargs = {
            "origin": "lower",
            "cmap": self._effective_cmap(),
            "extent": [-0.5, data.shape[1] - 0.5, -0.5, data.shape[0] - 0.5],
            "interpolation": "nearest",
        }
        if norm is None:
            im_kwargs["vmin"] = vmin
            im_kwargs["vmax"] = vmax
        else:
            im_kwargs["norm"] = norm
        self._map_im = self._map_ax.imshow(data, **im_kwargs)
        self._map_cbar = self._map_fig.colorbar(
            self._map_im, ax=self._map_ax, pad=0.0075, fraction=0.046
        )
        self._map_cbar.set_label("")
        self._map_cbar.set_ticks([])
        self._map_cbar.ax.tick_params(
            left=False, right=False, labelleft=False, labelright=False,
            bottom=False, top=False, labelbottom=False, labeltop=False
        )

        self._map_ax.set_aspect("equal", adjustable="box")
        self._map_ax.set_anchor("C")
        self._map_ax.set_xlim(-0.5, data.shape[1] - 0.5)
        self._map_ax.set_ylim(-0.5, data.shape[0] - 0.5)
        self._map_ax.set_axis_off()
        self._map_canvas.draw_idle()

    def _on_map_click(self, event):
        if self._map_data is None:
            return
        if event.inaxes != self._map_ax:
            return
        if (event.xdata is None) or (event.ydata is None):
            return
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        lin = self._grid_to_linear(x, y)
        if lin is None:
            return
        file_idx = self._lin_to_file.get(lin, None)
        if file_idx is None:
            self._set_status(f"No file for map pixel ({x}, {y})")
            return
        self._load_file_to_main_plot(file_idx)
        self._set_status(f"Selected map pixel ({x}, {y})")

    def _is_map_tab_active(self):
        return hasattr(self.widget, "tab_Map") and \
            (self.widget.tabWidget.currentWidget() == self.widget.tab_Map)

    def _should_show_roi_overlay(self):
        if self._map_data is None:
            return False
        if not self._is_map_tab_active():
            return False
        return self._roi_1d is not None

    def _clear_roi_overlays(self):
        changed = False
        if self._roi_artist_1d is not None:
            try:
                self._roi_artist_1d.remove()
            except Exception:
                pass
            self._roi_artist_1d = None
            changed = True
        if self._roi_artist_2d is not None:
            try:
                self._roi_artist_2d.remove()
            except Exception:
                pass
            self._roi_artist_2d = None
            changed = True
        if changed and hasattr(self.widget, "mpl") and hasattr(self.widget.mpl, "canvas"):
            try:
                self.widget.mpl.canvas.draw_idle()
            except Exception:
                pass

    def refresh_roi_overlays(self):
        self._clear_roi_overlays()
        if not self._should_show_roi_overlay():
            return
        try:
            if self._roi_1d is not None:
                xmin, xmax = self._roi_1d
                ax = self.widget.mpl.canvas.ax_pattern
                self._roi_artist_1d = ax.axvspan(
                    xmin, xmax, ymin=0.0, ymax=1.0,
                    facecolor="red", edgecolor="red", alpha=0.2, linewidth=1.2
                )
            self.widget.mpl.canvas.draw_idle()
        except Exception:
            # Keep map functionality robust even if overlay draw fails.
            self._roi_artist_1d = None
            self._roi_artist_2d = None

    def _export_image(self):
        if self._map_data is None:
            self._set_status("No map data to export.")
            return
        base = os.path.join(self.model.chi_path, "map.png")
        filen, _ = QtWidgets.QFileDialog.getSaveFileName(
            self.widget,
            "Export map image",
            base,
            "Image files (*.png *.pdf)",
        )
        if not filen:
            return
        self._map_fig.savefig(filen, dpi=300, bbox_inches="tight")
        self._set_status("Image exported.")

    def _export_npy(self):
        if self._map_data is None:
            self._set_status("No map data to export.")
            return
        root = self.model.chi_path if str(getattr(self.model, "chi_path", "")).strip() else os.getcwd()
        out_dir = os.path.join(root, "map_py")
        os.makedirs(out_dir, exist_ok=True)

        npy_path = os.path.join(out_dir, "map.npy")
        py_path = os.path.join(out_dir, "plot_map.py")

        np.save(npy_path, self._map_data)
        cmap = self._effective_cmap()
        vmin = float(self.widget.doubleSpinBox_MapVmin.value())
        vmax = float(self.widget.doubleSpinBox_MapVmax.value())
        use_log = bool(self.widget.checkBox_MapLog.isChecked())

        script = (
            "import numpy as np\n"
            "import matplotlib.pyplot as plt\n\n"
            "from matplotlib import colors as mcolors\n\n"
            "def main():\n"
            "    data = np.load('map.npy')\n"
            "    fig, ax = plt.subplots(figsize=(7, 5), facecolor='white')\n"
            "    ax.set_facecolor('white')\n"
            f"    cmap = {cmap!r}\n"
            f"    vmin = {vmin!r}\n"
            f"    vmax = {vmax!r}\n"
            f"    use_log = {use_log!r}\n"
            "    im_kwargs = {\n"
            "        'origin': 'lower',\n"
            "        'cmap': cmap,\n"
            "        'extent': [-0.5, data.shape[1] - 0.5, -0.5, data.shape[0] - 0.5],\n"
            "        'interpolation': 'nearest',\n"
            "    }\n"
            "    if use_log:\n"
            "        finite = data[np.isfinite(data)]\n"
            "        pos = finite[finite > 0]\n"
            "        if pos.size == 0:\n"
            "            use_log = False\n"
            "        else:\n"
            "            data = np.array(data, copy=True)\n"
            "            data[data <= 0] = np.nan\n"
            "            if vmin <= 0:\n"
            "                vmin = float(np.nanmin(pos))\n"
            "            vmax = max(vmax, vmin * 1.001)\n"
            "            im_kwargs['norm'] = mcolors.LogNorm(vmin=vmin, vmax=vmax)\n"
            "    if 'norm' not in im_kwargs:\n"
            "        im_kwargs['vmin'] = vmin\n"
            "        im_kwargs['vmax'] = vmax\n"
            "    im = ax.imshow(data, **im_kwargs)\n"
            "    cbar = fig.colorbar(im, ax=ax, pad=0.0075, fraction=0.046)\n"
            "    cbar.set_label('')\n"
            "    cbar.set_ticks([])\n"
            "    cbar.ax.tick_params(left=False, right=False, labelleft=False, labelright=False,\n"
            "                        bottom=False, top=False, labelbottom=False, labeltop=False)\n"
            "    ax.set_title('Map')\n"
            "    ax.set_aspect('equal', adjustable='box')\n"
            "    ax.set_anchor('C')\n"
            "    ax.set_xlim(-0.5, data.shape[1] - 0.5)\n"
            "    ax.set_ylim(-0.5, data.shape[0] - 0.5)\n"
            "    ax.set_axis_off()\n"
            "    fig.tight_layout()\n"
            "    plt.show()\n\n"
            "if __name__ == '__main__':\n"
            "    main()\n"
        )
        with open(py_path, "w", encoding="utf-8") as fh:
            fh.write(script)

        self._set_status("Exported map_py (map.npy + plot_map.py).")
