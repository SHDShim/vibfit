from PyQt6 import QtCore, QtWidgets

from ..version import __version__
from .mplwidget import MplWidget


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"vibfit {__version__}")
        self.resize(1540, 960)
        self._build_ui()

    def _build_ui(self):
        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)
        outer = QtWidgets.QHBoxLayout(central)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        outer.addWidget(splitter)

        self.mpl = MplWidget(self)
        splitter.addWidget(self.mpl)

        right_panel = QtWidgets.QWidget(self)
        right_layout = QtWidgets.QVBoxLayout(right_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self._build_toolbar(right_layout)

        self.tabs = QtWidgets.QTabWidget(right_panel)
        right_layout.addWidget(self.tabs, 1)

        self.status_box = QtWidgets.QPlainTextEdit(right_panel)
        self.status_box.setReadOnly(True)
        self.status_box.setMaximumHeight(180)
        right_layout.addWidget(self.status_box)

        self._build_file_tab()
        self._build_plot_tab()
        self._build_background_tab()
        self._build_peakfit_tab()
        self._build_sections_tab()
        self._align_spinboxes_right()

    def _align_spinboxes_right(self):
        alignment = QtCore.Qt.AlignmentFlag.AlignRight
        for spinbox in self.findChildren(QtWidgets.QAbstractSpinBox):
            if hasattr(spinbox, "setAlignment"):
                spinbox.setAlignment(alignment)

    @staticmethod
    def _accent_button_style(bg: str, border: str) -> str:
        return (
            "QPushButton {"
            f"background-color: {bg};"
            f"border: 1px solid {border};"
            "border-radius: 4px;"
            "color: #f3f4f6;"
            "font-weight: 600;"
            "padding: 6px 10px;"
            "}"
            "QPushButton:pressed {"
            "padding-top: 7px;"
            "padding-bottom: 5px;"
            "}"
            "QPushButton:checked {"
            f"background-color: {border};"
            f"border: 1px solid {border};"
            "}"
        )

    def _build_toolbar(self, parent_layout):
        toolbar_row = QtWidgets.QHBoxLayout()
        self.toolButton_ZoomOut = QtWidgets.QPushButton("Zoom out")
        self.toolButton_ZoomOut.setToolTip("Zoom out")
        self.toolButton_ZoomIn = QtWidgets.QPushButton("Zoom in")
        self.toolButton_ZoomIn.setToolTip("Zoom to active fit area")
        self.pushButton_AdjustYForSpectrum = QtWidgets.QPushButton("Yspec")
        self.pushButton_FindViewMinMax = QtWidgets.QPushButton("Yadj")
        self.pushButton_SaveSession = QtWidgets.QPushButton("Save")
        toolbar_buttons = (
            self.toolButton_ZoomOut,
            self.toolButton_ZoomIn,
            self.pushButton_AdjustYForSpectrum,
            self.pushButton_FindViewMinMax,
            self.pushButton_SaveSession,
        )
        target_height = max(button.sizeHint().height() for button in toolbar_buttons)
        base_font = self.pushButton_SaveSession.font()
        for button in toolbar_buttons:
            button.setFont(base_font)
            button.setMinimumHeight(target_height)
        for button in (
            self.toolButton_ZoomOut,
            self.toolButton_ZoomIn,
            self.pushButton_AdjustYForSpectrum,
            self.pushButton_FindViewMinMax,
            self.pushButton_SaveSession,
        ):
            toolbar_row.addWidget(button, 1)
        parent_layout.addLayout(toolbar_row)

    def _build_file_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QFormLayout(tab)

        self.pushButton_LoadSpectrum = QtWidgets.QPushButton("Load spectrum")
        self.pushButton_LoadSpectrum.setStyleSheet(self._accent_button_style("#166534", "#22c55e"))
        self.label_SpectrumPath = QtWidgets.QLabel("No spectrum loaded")
        self.label_SpectrumPath.setWordWrap(True)
        self.label_SpectrumShape = QtWidgets.QLabel("-")
        self.comboBox_SourceMode = QtWidgets.QComboBox()
        self.comboBox_SourceMode.addItems(["Auto detect", "Text/CSV", "NumPy", "Signal/DM"])
        self.pushButton_RestoreSession = QtWidgets.QPushButton("Restore selected backup")
        self.pushButton_EditBackupComment = QtWidgets.QPushButton("Edit comment")

        self.tableWidget_Backups = QtWidgets.QTableWidget(0, 3)
        self.tableWidget_Backups.setHorizontalHeaderLabels(["ID", "Timestamp", "Comment"])
        backup_header = self.tableWidget_Backups.horizontalHeader()
        backup_header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        backup_header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        backup_header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.tableWidget_Backups.verticalHeader().setVisible(False)
        self.tableWidget_Backups.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.tableWidget_Backups.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)

        layout.addRow(self.pushButton_LoadSpectrum)
        layout.addRow("Path", self.label_SpectrumPath)
        layout.addRow("Shape", self.label_SpectrumShape)
        layout.addRow("Loader", self.comboBox_SourceMode)
        restore_row = QtWidgets.QHBoxLayout()
        restore_row.addWidget(self.pushButton_RestoreSession)
        restore_row.addWidget(self.pushButton_EditBackupComment)
        layout.addRow("Restore", restore_row)
        layout.addRow(self.tableWidget_Backups)

        self.tabs.addTab(tab, "File")

    def _build_plot_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        self.doubleSpinBox_TopYMin = QtWidgets.QDoubleSpinBox()
        self.doubleSpinBox_TopYMax = QtWidgets.QDoubleSpinBox()
        self.doubleSpinBox_BottomYMin = QtWidgets.QDoubleSpinBox()
        self.doubleSpinBox_BottomYMax = QtWidgets.QDoubleSpinBox()
        for box in (
            self.doubleSpinBox_TopYMin,
            self.doubleSpinBox_TopYMax,
            self.doubleSpinBox_BottomYMin,
            self.doubleSpinBox_BottomYMax,
        ):
            box.setDecimals(2)
            box.setRange(-1e12, 1e12)
            box.setKeyboardTracking(False)
        self.pushButton_ApplyView = QtWidgets.QPushButton("Apply view")

        grid = QtWidgets.QGridLayout()
        grid.setColumnStretch(0, 0)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)

        self.label_TopYMinMax = QtWidgets.QLabel("Top Y min/max")
        self.label_BottomYMinMax = QtWidgets.QLabel("Btm Y min/max")
        grid.addWidget(self.label_TopYMinMax, 0, 0)
        grid.addWidget(self.doubleSpinBox_TopYMin, 0, 1)
        grid.addWidget(self.doubleSpinBox_TopYMax, 0, 2)
        grid.addWidget(self.label_BottomYMinMax, 1, 0)
        grid.addWidget(self.doubleSpinBox_BottomYMin, 1, 1)
        grid.addWidget(self.doubleSpinBox_BottomYMax, 1, 2)
        layout.addLayout(grid)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(self.pushButton_ApplyView)
        button_row.addStretch(1)
        layout.addLayout(button_row)
        layout.addStretch(1)
        self.tabs.addTab(tab, "Plot")

    def _build_background_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        info = QtWidgets.QLabel(
            "Select one or more background areas on the top plot, then fit a power-law background."
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        controls = QtWidgets.QHBoxLayout()
        self.pushButton_SelectBackgroundArea = QtWidgets.QPushButton("Select area")
        self.pushButton_SelectBackgroundArea.setCheckable(True)
        self.pushButton_SelectBackgroundArea.setStyleSheet(self._accent_button_style("#a16207", "#facc15"))
        self.pushButton_RemoveBackgroundArea = QtWidgets.QPushButton("Remove selected")
        self.pushButton_ClearBackgroundAreas = QtWidgets.QPushButton("Clear areas")
        self.pushButton_FitBackground = QtWidgets.QPushButton("Fit background")
        self.pushButton_FitBackground.setStyleSheet(self._accent_button_style("#991b1b", "#ef4444"))
        background_buttons = (
            self.pushButton_SelectBackgroundArea,
            self.pushButton_RemoveBackgroundArea,
            self.pushButton_ClearBackgroundAreas,
            self.pushButton_FitBackground,
        )
        target_height = max(button.sizeHint().height() for button in background_buttons)
        for button in background_buttons:
            button.setMinimumHeight(target_height)
        controls.addWidget(self.pushButton_SelectBackgroundArea)
        controls.addWidget(self.pushButton_RemoveBackgroundArea)
        controls.addWidget(self.pushButton_ClearBackgroundAreas)
        controls.addStretch(1)
        controls.addWidget(self.pushButton_FitBackground)
        layout.addLayout(controls)

        self.tableWidget_BackgroundAreas = QtWidgets.QTableWidget(0, 2)
        self.tableWidget_BackgroundAreas.setHorizontalHeaderLabels(["Min (cm^-1)", "Max (cm^-1)"])
        area_header = self.tableWidget_BackgroundAreas.horizontalHeader()
        area_header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Stretch)
        self.tableWidget_BackgroundAreas.verticalHeader().setVisible(False)
        self.tableWidget_BackgroundAreas.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.tableWidget_BackgroundAreas.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        layout.addWidget(self.tableWidget_BackgroundAreas, 1)

        self.plainTextEdit_BackgroundReport = QtWidgets.QPlainTextEdit()
        self.plainTextEdit_BackgroundReport.setReadOnly(True)
        layout.addWidget(self.plainTextEdit_BackgroundReport, 1)

        self.tabs.addTab(tab, "Background")

    def _build_peakfit_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        button_row0 = QtWidgets.QHBoxLayout()
        self.pushButton_SetFitRange = QtWidgets.QPushButton("Set fit range")
        self.pushButton_SetFitRange.setCheckable(True)
        self.pushButton_ClearFitRange = QtWidgets.QPushButton("Clear fit range")
        self.pushButton_SetFitRange.setStyleSheet(self._accent_button_style("#a16207", "#facc15"))
        peakfit_row0_buttons = (
            self.pushButton_SetFitRange,
            self.pushButton_ClearFitRange,
        )
        target_height = max(button.sizeHint().height() for button in peakfit_row0_buttons)
        for button in peakfit_row0_buttons:
            button.setMinimumHeight(target_height)
            button_row0.addWidget(button, 1)
        layout.addLayout(button_row0)

        button_row1 = QtWidgets.QHBoxLayout()
        self.pushButton_PickPeaks = QtWidgets.QPushButton("Pick peaks")
        self.pushButton_PickPeaks.setCheckable(True)
        self.pushButton_PickPeaks.setStyleSheet(self._accent_button_style("#a16207", "#facc15"))
        self.pushButton_RemovePeak = QtWidgets.QPushButton("Remove peak")
        self.pushButton_ClearPeaks = QtWidgets.QPushButton("Clear")
        self.pushButton_Fit = QtWidgets.QPushButton("Fit region")
        self.pushButton_Fit.setStyleSheet(self._accent_button_style("#991b1b", "#ef4444"))
        peakfit_row1_buttons = (
            self.pushButton_PickPeaks,
            self.pushButton_RemovePeak,
            self.pushButton_ClearPeaks,
            self.pushButton_Fit,
        )
        target_height = max(button.sizeHint().height() for button in peakfit_row1_buttons)
        for button in peakfit_row1_buttons:
            button.setMinimumHeight(target_height)
            button_row1.addWidget(button, 1)
        layout.addLayout(button_row1)

        button_row2 = QtWidgets.QHBoxLayout()
        self.pushButton_SaveToSection = QtWidgets.QPushButton("Save to section")
        self.pushButton_SaveToSection.setStyleSheet(self._accent_button_style("#166534", "#22c55e"))
        self.pushButton_SaveFitResults = QtWidgets.QPushButton("Save XLS")
        self.pushButton_ExportNPY = QtWidgets.QPushButton("Export NPY")
        peakfit_row2_buttons = (
            self.pushButton_SaveToSection,
            self.pushButton_SaveFitResults,
            self.pushButton_ExportNPY,
        )
        target_height = max(button.sizeHint().height() for button in peakfit_row2_buttons)
        for button in peakfit_row2_buttons:
            button.setMinimumHeight(target_height)
            button_row2.addWidget(button, 1)
        layout.addLayout(button_row2)

        layout.addWidget(QtWidgets.QLabel("Fitting setup"))

        self.tableWidget_Peaks = QtWidgets.QTableWidget(0, 17)
        self.tableWidget_Peaks.setHorizontalHeaderLabels(
            [
                "Name",
                "Avari",
                "Amp",
                "min A",
                "max A",
                "Cvari",
                "Cent",
                "min C",
                "max C",
                "Wvari",
                "Width",
                "min W",
                "max W",
                "nLvari",
                "nL",
                "min nL",
                "max nL",
            ]
        )
        header = self.tableWidget_Peaks.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(False)
        self.tableWidget_Peaks.verticalHeader().setVisible(False)
        self.tableWidget_Peaks.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.tableWidget_Peaks.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        layout.addWidget(self.tableWidget_Peaks, 2)

        layout.addWidget(QtWidgets.QLabel("Fit result"))

        self.tableWidget_Results = QtWidgets.QTableWidget(0, 5)
        self.tableWidget_Results.setHorizontalHeaderLabels(
            ["Peak", "Center (cm^-1)", "Amplitude", "Sigma (cm^-1)", "Fraction"]
        )
        results_header = self.tableWidget_Results.horizontalHeader()
        results_header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        results_header.setStretchLastSection(True)
        self.tableWidget_Results.verticalHeader().setVisible(False)
        layout.addWidget(self.tableWidget_Results, 1)

        self.plainTextEdit_FitReport = QtWidgets.QPlainTextEdit()
        self.plainTextEdit_FitReport.setReadOnly(True)
        layout.addWidget(self.plainTextEdit_FitReport, 1)

        self.tabs.addTab(tab, "PeakFit")

    def _build_sections_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)

        controls = QtWidgets.QHBoxLayout()
        self.pushButton_SectionSetCurrent = QtWidgets.QPushButton("Set current")
        self.pushButton_SectionSetCurrent.setStyleSheet(self._accent_button_style("#166534", "#22c55e"))
        self.pushButton_SectionZoom = QtWidgets.QPushButton("Zoom to section")
        self.pushButton_SectionRemove = QtWidgets.QPushButton("Remove selected")
        self.pushButton_SectionClear = QtWidgets.QPushButton("Clear list")
        section_buttons = (
            self.pushButton_SectionSetCurrent,
            self.pushButton_SectionZoom,
            self.pushButton_SectionRemove,
            self.pushButton_SectionClear,
        )
        target_height = max(button.sizeHint().height() for button in section_buttons)
        for button in section_buttons:
            button.setMinimumHeight(target_height)
            controls.addWidget(button, 1)
        layout.addLayout(controls)

        self.tableWidget_Sections = QtWidgets.QTableWidget(0, 6)
        self.tableWidget_Sections.setHorizontalHeaderLabels(
            ["Saved", "Region", "Min (cm$^{-1}$)", "Max (cm$^{-1}$)", "Peaks", "Fit"]
        )
        header = self.tableWidget_Sections.horizontalHeader()
        header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.tableWidget_Sections.verticalHeader().setVisible(False)
        self.tableWidget_Sections.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        self.tableWidget_Sections.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        layout.addWidget(self.tableWidget_Sections, 1)

        self.tabs.addTab(tab, "Sections")
