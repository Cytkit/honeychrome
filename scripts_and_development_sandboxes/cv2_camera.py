import glob
import os
import sys
import cv2

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget


def get_linux_camera_index(target_name="HD Camera"):
    """Scans V4L2 devices on Linux to find the OpenCV camera index matching a target name."""
    for dev_path in sorted(glob.glob("/sys/class/video4linux/video*")):
        name_file = os.path.join(dev_path, "name")
        if os.path.exists(name_file):
            with open(name_file, "r") as f:
                device_name = f.read().strip()

            index = int(os.path.basename(dev_path).replace("video", ""))
            print(f"Linux V4L2 Device [/dev/video{index}]: {device_name}")

            if target_name.lower() in device_name.lower():
                # Test if OpenCV can capture from this node (filters out metadata nodes)
                cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
                if cap.isOpened():
                    ret, _ = cap.read()
                    cap.release()
                    if ret:
                        print(f"--> Matched '{device_name}' on /dev/video{index}")
                        return index

    print(f"Target '{target_name}' not found. Defaulting to index 0.")
    return 0


class LinuxWebcamWidget(QWidget):
    def __init__(self, target_camera_name="HD Camera", parent=None):
        super().__init__(parent)

        # 1. Setup UI Layout
        self.layout = QVBoxLayout(self)
        self.image_label = QLabel("Initializing Linux Camera...")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.image_label)

        # 2. Find V4L2 camera index by name
        camera_index = get_linux_camera_index(target_camera_name)

        # 3. Connect via OpenCV V4L2 backend
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)

        if not self.cap.isOpened():
            self.image_label.setText("Error: Could not open camera.")
            return

        # 4. Linux V4L2 Hardware Exposure & Gain Controls
        # On V4L2 Linux: 1 = Manual Mode, 3 = Auto Mode
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)

        # Exposure values on V4L2 are absolute time in 100µs units (e.g., 100 = 10ms)
        self.cap.set(cv2.CAP_PROP_EXPOSURE, 50)
        self.cap.set(cv2.CAP_PROP_GAIN, 1)

        # 5. Timer for frame updates (~30 FPS)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Rotate frame 90 degrees clockwise
        rotated = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        # Convert OpenCV BGR to PySide QImage
        rgb_frame = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape

        q_img = QImage(rgb_frame.data, w, h, ch * w, QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img))

    def closeEvent(self, event):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        event.accept()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Linux V4L2 Webcam Stream")
        self.resize(600, 800)
        self.setCentralWidget(LinuxWebcamWidget(target_camera_name="HD Camera"))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())