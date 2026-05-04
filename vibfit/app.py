import os
import sys
import faulthandler
from importlib import resources

faulthandler.enable()

if sys.platform == "darwin":
    os.environ.setdefault("MPLBACKEND", "QtAgg")
    os.environ.setdefault("QT_OPENGL", "software")
    os.environ.setdefault("QT_QUICK_BACKEND", "software")

from PyQt6.QtCore import QCoreApplication, Qt

if sys.platform == "darwin":
    use_sw_gl = Qt.ApplicationAttribute.AA_UseSoftwareOpenGL
    if use_sw_gl is not None:
        QCoreApplication.setAttribute(use_sw_gl, True)

import matplotlib

matplotlib.use("QtAgg")

from PyQt6 import QtWidgets
from PyQt6.QtGui import QColor, QIcon, QPalette

from .control import MainController


def _build_dark_palette() -> QPalette:
    palette = QPalette()
    role = QPalette.ColorRole
    group = QPalette.ColorGroup
    palette.setColor(role.Window, QColor(40, 42, 46))
    palette.setColor(role.WindowText, Qt.GlobalColor.white)
    palette.setColor(role.Base, QColor(26, 28, 31))
    palette.setColor(role.AlternateBase, QColor(45, 48, 53))
    palette.setColor(role.ToolTipBase, QColor(26, 28, 31))
    palette.setColor(role.ToolTipText, Qt.GlobalColor.white)
    palette.setColor(role.Text, Qt.GlobalColor.white)
    palette.setColor(role.Button, QColor(52, 55, 61))
    palette.setColor(role.ButtonText, Qt.GlobalColor.white)
    palette.setColor(role.BrightText, Qt.GlobalColor.red)
    palette.setColor(role.Link, QColor(88, 166, 255))
    palette.setColor(role.Highlight, QColor(88, 166, 255))
    palette.setColor(role.HighlightedText, QColor(20, 20, 20))
    palette.setColor(group.Disabled, role.Text, QColor(130, 130, 130))
    palette.setColor(group.Disabled, role.ButtonText, QColor(130, 130, 130))
    palette.setColor(group.Disabled, role.WindowText, QColor(130, 130, 130))
    return palette


def _icon_path() -> str | None:
    try:
        return str(resources.files("vibfit.assets").joinpath("icon.png"))
    except (FileNotFoundError, ModuleNotFoundError):
        return None


def _set_macos_dock_icon(icon_path: str):
    if sys.platform != "darwin":
        return
    try:
        from AppKit import NSApplication, NSImage
    except ImportError:
        return
    image = NSImage.alloc().initWithContentsOfFile_(icon_path)
    if image is not None:
        NSApplication.sharedApplication().setApplicationIconImage_(image)


app = QtWidgets.QApplication(sys.argv)
app.setApplicationName("vibfit")
app.setDesktopFileName("vibfit")
app.setStyle("Fusion")
app.setPalette(_build_dark_palette())
icon_path = _icon_path()
if icon_path is not None:
    app_icon = QIcon(icon_path)
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)
        _set_macos_dock_icon(icon_path)

controller = MainController()
if icon_path is not None:
    controller.widget.setWindowIcon(app.windowIcon())
controller.show_window()

ret = app.exec()
sys.exit(ret)
