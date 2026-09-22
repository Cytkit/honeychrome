import sys
import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel

WIDTH, HEIGHT, PAGES = 128, 64, 8
SCALE = 6
ON_COLOR = (255, 255, 0)
OFF_COLOR = (0, 0, 0)


class SimWindow(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.label = QLabel()
        self.label.setFixedSize(WIDTH * SCALE, HEIGHT * SCALE)
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

    def paint_buffer(self, page_buf: bytes):
        arr = np.frombuffer(page_buf, dtype=np.uint8).reshape(PAGES, WIDTH)  # (8, 128)

        # Unpack each byte into 8 bits along a new last axis
        bits = np.unpackbits(arr[:, :, None], axis=2, bitorder="little")  # (8, 128, 8)

        # Reorder to (page, bit_within_page, column) → (8, 8, 128)
        bits = bits.transpose(0, 2, 1)  # (8, 8, 128)

        # Flatten pages and bits into rows: (64, 128)
        bits = bits.reshape(HEIGHT, WIDTH)

        rgb = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        rgb[bits == 1] = ON_COLOR
        rgb[bits == 0] = OFF_COLOR

        qimg = QImage(rgb.data, WIDTH, HEIGHT, WIDTH * 3, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            WIDTH * SCALE, HEIGHT * SCALE,
            Qt.IgnoreAspectRatio, Qt.FastTransformation,
        )
        self.label.setPixmap(pix)


def run_sim(frame_queue):
    app = QApplication(sys.argv)
    win = SimWindow()
    win.show()

    timer = QTimer()
    timer.setInterval(30)  # ~33 Hz poll

    def poll():
        # Drain all pending frames, keep only the latest
        latest = None
        while True:
            try:
                latest = frame_queue.get_nowait()
            except Exception:
                break
        if latest is not None:
            if latest is None:  # sentinel → quit
                app.quit()
            else:
                win.paint_buffer(latest)

    timer.timeout.connect(poll)
    timer.start()
    sys.exit(app.exec())


if __name__ == "__main__":
    # Not used when launched via multiprocessing
    pass