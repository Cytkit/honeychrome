from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QPushButton, QSlider, QSpinBox, QHBoxLayout,
                               QVBoxLayout, QDockWidget, QLabel, QToolBar, QSizePolicy, QFrame, QGraphicsOpacityEffect)
from PySide6.QtCore import Qt, QPropertyAnimation, QEasingCurve, QTimer

from honeychrome import settings
from honeychrome.view_components.icon_loader import icon


class AcquisitionWidget(QWidget):
    def __init__(self, bus, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bus = bus

        self.setMaximumHeight(180)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        layout = QVBoxLayout(self)

        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel)  # Add border
        frame_layout = QVBoxLayout(frame)

        # Title
        title = QLabel("Acquisition")
        title.setAlignment(Qt.AlignmentFlag.AlignLeft)
        frame_layout.addWidget(title)

        acquisition_toolbar = QToolBar("Acquisition Toolbar")
        acquisition_toolbar.setMovable(False)
        self.action_initialise = QAction(icon('plug-connected'), "Initialise", self)
        self.action_start_acquisition = QAction(icon('player-record', colour='red'), "Start Acquisition", self)
        self.action_start_acquisition.setEnabled(False)
        self.action_stop_acquisition = QAction(icon('player-stop'), "Stop Acquisition", self)
        self.action_stop_acquisition.setEnabled(False)
        self.action_restart_acquisition = QAction(icon('repeat'), "Restart Acquisition", self)
        self.action_restart_acquisition.setEnabled(False)
        self.action_flush = QAction(icon('wash'), "Flush", self)
        self.action_flush.setEnabled(False)
        self.action_backflush = QAction(icon('wiper-wash'), "Backflush", self)
        self.action_backflush.setEnabled(False)
        self.action_initialise.triggered.connect(self.initialise)
        self.action_start_acquisition.triggered.connect(self.start_acquisition)
        self.action_stop_acquisition.triggered.connect(self.stop_acquisition)
        self.action_restart_acquisition.triggered.connect(self.restart_acquisition)
        self.action_flush.triggered.connect(self.flush)
        self.action_backflush.triggered.connect(self.backflush)
        acquisition_toolbar.addAction(self.action_initialise)
        acquisition_toolbar.addAction(self.action_start_acquisition)
        acquisition_toolbar.addAction(self.action_stop_acquisition)
        acquisition_toolbar.addAction(self.action_restart_acquisition)
        acquisition_toolbar.addAction(self.action_flush)
        acquisition_toolbar.addAction(self.action_backflush)

        frame_layout.addWidget(acquisition_toolbar)


        # Get the QToolButton for the action
        self.btn = acquisition_toolbar.widgetForAction(self.action_start_acquisition)
        # apply an opacity effect and animate it
        effect = QGraphicsOpacityEffect(self.btn)
        self.btn.setGraphicsEffect(effect)
        self.animation = QPropertyAnimation(effect, b"opacity", self)

        # Flow rate slider and spinbox
        slider_layout = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(10, 100)
        self.spinbox = QSpinBox()
        self.spinbox.setRange(10, 100)

        # Sync slider and spinbox
        self.slider.valueChanged.connect(self.spinbox.setValue)
        self.spinbox.valueChanged.connect(self.slider.setValue)

        slider_layout.addWidget(QLabel("Sample [uL/min]:"))
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.spinbox)
        frame_layout.addLayout(slider_layout)

        layout.addWidget(frame)

        self.initialised = False

        self.standby_timer = QTimer()
        self.standby_timer.setSingleShot(True)
        self.standby_timer.timeout.connect(self.initialise)

    def initialise(self):
        if self.bus is not None:
            self.bus.initialiseInstrument.emit() # if already initialised, this puts it to standby

        if self.initialised:
            self.initialised = False
            self.action_start_acquisition.setEnabled(False)
            self.action_flush.setEnabled(False)
            self.action_backflush.setEnabled(False)
            self.standby_timer.stop()
        else:
            self.initialised = True
            self.action_start_acquisition.setEnabled(True)
            self.action_flush.setEnabled(True)
            self.action_backflush.setEnabled(True)
            self.standby_timer.start(settings.standby_time)

    def start_acquisition(self):
        self.action_start_acquisition.setEnabled(False)
        self.action_stop_acquisition.setIcon(icon('player-stop', colour='red'))
        self.action_stop_acquisition.setEnabled(True)
        self.action_restart_acquisition.setEnabled(True)
        self.action_flush.setEnabled(False)
        self.action_backflush.setEnabled(False)
        if self.bus is not None:
            self.bus.startAcquisition.emit()
        self.start_flashing()
        self.standby_timer.stop()

    def start_flashing(self):
        self.animation.setDuration(900)
        self.animation.setStartValue(1.0)
        self.animation.setKeyValueAt(0.5, 0.2)  # fade mid-way
        self.animation.setEndValue(1.0)
        self.animation.setEasingCurve(QEasingCurve.InOutQuad)
        self.animation.setLoopCount(-1)
        self.animation.start()

    def stop_acquisition(self):
        self.action_start_acquisition.setEnabled(True)
        self.action_stop_acquisition.setIcon(icon('player-stop'))
        self.action_stop_acquisition.setEnabled(False)
        self.action_restart_acquisition.setEnabled(False)
        self.action_flush.setEnabled(True)
        self.action_backflush.setEnabled(True)
        if self.bus is not None:
            self.bus.stopAcquisition.emit()
        self.animation.stop()
        self.standby_timer.start(settings.standby_time)

    def restart_acquisition(self):
        if self.bus is not None:
            self.bus.restartAcquisition.emit()

    def flush(self):
        if self.bus is not None:
            self.bus.flushSip.emit()
        self.standby_timer.start(settings.standby_time)

    def backflush(self):
        if self.bus is not None:
            self.bus.backFlushSip.emit()
        self.standby_timer.start(settings.standby_time)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Dockable Control Panel")
        self.resize(800, 600)

        # Central widget placeholder
        central = QLabel("Main Content Area")
        central.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(central)

        # Create dock widget
        self.dock = QDockWidget("Controls", self)
        self.dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        self.acquisition_widget = AcquisitionWidget(None)
        self.dock.setWidget(self.acquisition_widget)

        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
