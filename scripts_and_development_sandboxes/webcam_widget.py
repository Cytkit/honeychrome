import sys
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QTransform
from PySide6.QtMultimedia import QCamera, QMediaCaptureSession, QMediaDevices, QVideoSink
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget


class WebcamWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.layout = QVBoxLayout(self)
        self.image_label = QLabel("Initializing camera...")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.image_label)

        self.camera = None
        self.capture_session = None
        self.video_sink = None

    def find_connect_and_run(self):
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

        print("Auto Supported:", self.camera.isExposureModeSupported(QCamera.ExposureMode.ExposureAuto))
        print("Manual Supported:", self.camera.isExposureModeSupported(QCamera.ExposureMode.ExposureManual))

        if self.camera and self.camera.isExposureModeSupported(QCamera.ExposureMode.ExposureAuto):
            self.camera.setExposureMode(QCamera.ExposureMode.ExposureAuto)

        # if self.camera.isExposureModeSupported(QCamera.ExposureMode.ExposureManual):
        #     # Set Exposure Time (in seconds, e.g., 0.01 = 10ms / 1/100s)
        #     self.camera.setExposureMode(QCamera.ExposureMode.ExposureManual)
        #     self.camera.setManualExposureTime(0.005)
        #     print(f"Current Exposure Time: {self.camera.exposureTime()}s")
        #
        #     ### The items below are not supported by camera
        #     # # Set Gain via ISO Sensitivity (e.g., 100, 200, 400, 800)
        #     # self.camera.setManualIsoSensitivity(0)
        #     # self.camera.setExposureCompensation(-2.0)
        #     # print(f"Current ISO (Gain): {self.camera.isoSensitivity()}")
        #
        # else:
        #     print("Manual exposure mode not supported by driver. Using EV offset instead.")
        #     self.camera.setExposureCompensation(-2.0)

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
            self.cam_widget = WebcamWidget()
            self.setCentralWidget(self.cam_widget)

            self.cam_widget.find_connect_and_run()


    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())