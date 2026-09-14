import sys
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QTransform, QHideEvent, QShowEvent
from PySide6.QtMultimedia import QCamera, QMediaCaptureSession, QMediaDevices, QVideoSink
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QSlider, QFormLayout
import numpy as np

# Define logarithmic boundaries
min_exposure = 0.005 # s
max_exposure = 1.0 # s
log_min = np.log10(min_exposure)
log_max = np.log10(max_exposure)

class AlignmentCameraWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.camera = None
        self.capture_session = None
        self.video_sink = None

        self.layout = QHBoxLayout(self)
        self.control_layout = QFormLayout()
        self.layout.addLayout(self.control_layout)
        self.image_label = QLabel("")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.image_label)

        self.find_connect_and_run_btn = QPushButton("Start Camera")
        self.find_connect_and_run_btn.clicked.connect(self.find_connect_and_run)
        self.control_layout.addWidget(self.find_connect_and_run_btn)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)  # 1000 discrete log steps
        self.value_label = QLabel("0.0000 s")
        self.control_layout.addRow("Exposure:", self.slider)
        self.control_layout.addRow("Value:", self.value_label)
        self.slider.valueChanged.connect(self.on_exposure_changed)
        default_val = 0.005
        default_step = int(1000 * (np.log10(default_val) - log_min) / (log_max - log_min))
        self.slider.setValue(default_step)

        # self.autoexposure_btn = QPushButton("Autoexposure (laser off, find channel)")
        # self.autoexposure_btn.clicked.connect(self.autoexposure)
        # self.control_layout.addWidget(self.autoexposure_btn)
        # self.min_exposure_btn = QPushButton("Minimum exposure (laser on, align laser)")
        # self.min_exposure_btn.clicked.connect(self.min_exposure)
        # self.control_layout.addWidget(self.min_exposure_btn)

        # self.control_layout.addStretch()

    def find_connect_and_run(self):
        if not self.camera:
            # Match exact camera name
            selected_device = None
            for device in QMediaDevices.videoInputs():
                if 'HD Camera: HD Camera' in device.description():
                    selected_device = device
                    break

            if not selected_device:
                print('Camera not found.')
                return

            # Configure Qt Camera Pipeline
            self.camera = QCamera(selected_device)

            self.capture_session = QMediaCaptureSession()
            self.video_sink = QVideoSink()

            self.capture_session.setCamera(self.camera)
            self.capture_session.setVideoSink(self.video_sink)

            # 3. Handle incoming frame events
            self.video_sink.videoFrameChanged.connect(self.process_frame)
            self.camera.start()

            self.find_connect_and_run_btn.setText('Stop Camera')

        else:
            self.camera.stop()
            self.camera = None
            self.find_connect_and_run_btn.setText('Start Camera')



    def process_frame(self, frame):
        if not frame.isValid():
            return

        # Convert frame directly to QPixmap
        image = frame.toImage()
        pixmap = QPixmap.fromImage(image)

        # Rotate 90 degrees clockwise without OpenCV
        rotated_pixmap = pixmap.transformed(QTransform().rotate(90))

        # Render in GUI
        self.image_label.setPixmap(rotated_pixmap)

    # def autoexposure(self):
    #     # Check if ExposureAuto (not ExposureManual) is supported by the camera driver
    #     if self.camera and self.camera.isExposureModeSupported(QCamera.ExposureMode.ExposureAuto):
    #         self.camera.setExposureMode(QCamera.ExposureMode.ExposureAuto)
    #         print("Switched camera to Auto Exposure mode.")
    #     else:
    #         print("Auto exposure mode is not supported on this device.")

    def min_exposure(self):
        # Switch camera to Manual Exposure Mode
        if self.camera and self.camera.isExposureModeSupported(QCamera.ExposureMode.ExposureManual):
            self.camera.setExposureMode(QCamera.ExposureMode.ExposureManual)

            # Set Exposure Time (in seconds, e.g., 0.01 = 10ms / 1/100s)
            self.camera.setManualExposureTime(0.005)
            print(f"Current Exposure Time: {self.camera.exposureTime()}s")

            ### The items below are not supported by camera
            # # Set Gain via ISO Sensitivity (e.g., 100, 200, 400, 800)
            # self.camera.setManualIsoSensitivity(0)
            # self.camera.setExposureCompensation(-2.0)
            # print(f"Current ISO (Gain): {self.camera.isoSensitivity()}")

        else:
            print("Manual exposure mode not supported by driver. Using EV offset instead.")
            self.camera.setExposureCompensation(-2.0)

    def on_exposure_changed(self, value: int):
        # 1. Calculate normalized position (0.0 to 1.0)
        t = value / 1000.0

        # 2. Map linearly in log space, then exponentiate
        log_val = log_min + t * (log_max - log_min)
        exposure_seconds = np.pow(10, log_val)

        # 3. Update QCamera settings
        if self.camera:
            if self.camera.isExposureModeSupported(QCamera.ExposureMode.ExposureManual):
                self.camera.setExposureMode(QCamera.ExposureMode.ExposureManual)
                self.camera.setManualExposureTime(exposure_seconds)

        # 4. Display formatted UI text
        if exposure_seconds < 0.001:
            self.value_label.setText(f"{exposure_seconds * 1000000:.1f} µs")
        elif exposure_seconds < 1.0:
            self.value_label.setText(f"{exposure_seconds * 1000:.1f} ms")
        else:
            self.value_label.setText(f"{exposure_seconds:.2f} s")

    def showEvent(self, event: QShowEvent):
        super().showEvent(event)
        # Power on the hardware sensor and resume frame flow
        if self.camera and not self.camera.isActive():
            self.camera.start()

    def hideEvent(self, event: QHideEvent):
        super().hideEvent(event)
        # Power off hardware, turn off LED light, and freeze CPU pipeline
        if self.camera and self.camera.isActive():
            self.camera.stop()

    def closeEvent(self, event):
        if self.camera:
            self.camera.stop()
        event.accept()


if __name__ == "__main__":
    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Pure PySide6 Stream")
            self.resize(600, 800)
            self.setCentralWidget(AlignmentCameraWidget())

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())