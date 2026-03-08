from PyQt6 import QtCore, QtWidgets
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from ..model import cminv_to_ev, ev_to_cminv


class DualSpectrumCanvas(FigureCanvasQTAgg):
    def __init__(self):
        self.fig = Figure(figsize=(8, 6), constrained_layout=True)
        self.ax_raw = self.fig.add_subplot(211)
        self.ax_fit = self.fig.add_subplot(212, sharex=self.ax_raw)
        self.ax_raw.set_ylabel("Raw intensity")
        self.ax_fit.set_ylabel("Bg-sub intensity")
        self.ax_fit.set_xlabel("Wavenumber (cm$^{-1}$)")
        super().__init__(self.fig)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.updateGeometry()
        self._secondary_axes = []
        self._configure_axes()

    def _configure_axes(self):
        for axis in (self.ax_raw, self.ax_fit):
            axis.set_facecolor("#1e1f22")
            axis.grid(True, color="#43464d", alpha=0.35, linewidth=0.6)
            axis.tick_params(colors="#f0f0f0")
            axis.xaxis.label.set_color("#f0f0f0")
            axis.yaxis.label.set_color("#f0f0f0")
            for spine in axis.spines.values():
                spine.set_color("#c9c9c9")
        self.fig.set_facecolor("#2b2d31")
        self.ax_raw.tick_params(bottom=False, labelbottom=False)
        self.ax_fit.tick_params(top=False, labeltop=False)
        self._secondary_axes = [
            self.ax_raw.secondary_xaxis("top", functions=(cminv_to_ev, ev_to_cminv)),
        ]
        self._secondary_axes[0].set_xlabel("Energy (eV)")
        self._secondary_axes[0].tick_params(colors="#f0f0f0")
        self._secondary_axes[0].xaxis.label.set_color("#f0f0f0")
        for spine in self._secondary_axes[0].spines.values():
            spine.set_color("#c9c9c9")

    def clear(self):
        self.ax_raw.cla()
        self.ax_fit.cla()
        self.ax_raw.set_ylabel("Raw intensity")
        self.ax_fit.set_ylabel("Bg-sub intensity")
        self.ax_fit.set_xlabel("Wavenumber (cm$^{-1}$)")
        self._configure_axes()


class MplWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = DualSpectrumCanvas()
        self.canvas.setParent(self)
        self.canvas.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)
        self.canvas.setFocus()
        self.vbl = QtWidgets.QVBoxLayout(self)
        self.vbl.setContentsMargins(0, 0, 0, 0)
        self.vbl.setSpacing(0)
        self.ntb = NavigationToolbar(self.canvas, self)
        self.setStyleSheet("MplWidget, QWidget { border: 0px; }")
        self.canvas.setStyleSheet("border: 0px;")
        self.ntb.setStyleSheet("border: 0px;")
        self.vbl.addWidget(self.ntb)
        self.vbl.addWidget(self.canvas)
