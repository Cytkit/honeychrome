import sys
import time
import math
import ft4222
from PySide6.QtCore import QTimer
from PySide6.QtGui import Qt, QImage, QPixmap
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QApplication
from ft4222.SPIMaster import Mode as MasterSingle, Clock, SlaveSelect
from ft4222.SPI import Cpha, Cpol
from ft4222.GPIO import Dir, Port
from PIL import Image, ImageDraw
from PIL.ImageQt import ImageQt

width=128
height=64

class SSD1309_sim(QWidget):
    scale = 6
    ON_COLOR  = (255, 255, 0)   # yellow
    OFF_COLOR = (0, 0, 0)       # black

    def __init__(self):
        super().__init__()

        print('Opening OLED sim...')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.image_label = QLabel('Waiting for image...')
        self.image_label.setFixedSize(width * self.scale, height * self.scale)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setScaledContents(False)

        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)

    def display(self, pil_img):
        # Convert to RGB — ImageQt does not handle mode '1' well
        pil_img = pil_img.convert('RGB')

        # Recolor: any pixel that isn't black becomes yellow
        pixels = pil_img.load()
        for y in range(pil_img.height):
            for x in range(pil_img.width):
                r, g, b = pixels[x, y]
                if r or g or b:  # was "on"
                    pixels[x, y] = self.ON_COLOR
                else:
                    pixels[x, y] = self.OFF_COLOR

        q_image = ImageQt(pil_img)   # needs `from PIL.ImageQt import ImageQt`
        pixmap = QPixmap.fromImage(q_image)

        scaled = pixmap.scaled(
            width  * self.scale,
            height * self.scale,
            Qt.IgnoreAspectRatio,      # exact multiple, no distortion
            Qt.FastTransformation,     # <-- nearest-neighbor, keeps pixels crisp
        )
        self.image_label.setPixmap(scaled)


class SSD1309:
    def __init__(self):

        print("Opening FT4222 devices...")
        # Open Channel A for SPI Master transfers
        self.dev_spi = ft4222.openByDescription('FT4222 A')
        # Open Channel B for GPIO control (DC and RST pins)
        self.dev_gpio = ft4222.openByDescription('FT4222 B')

        print("Initializing SPI Master on FT4222 A...")
        self.dev_spi.spiMaster_Init(
            MasterSingle.SINGLE, Clock.DIV_32, Cpol.IDLE_LOW, Cpha.CLK_LEADING, SlaveSelect.SS0
        )

        print("Initializing GPIOs on FT4222 B...")
        self.dev_gpio.gpio_Init(gpio0=Dir.OUTPUT, gpio1=Dir.OUTPUT)

        print("Resetting OLED...")
        self.reset()
        print("Sending SSD1309 initialization commands...")
        self.init_display()
        print("Initialization complete! Starting render loop...\n")

    def set_dc(self, is_data: bool):
        # Write DC signal to Channel B
        self.dev_gpio.gpio_Write(Port.P0, is_data)

    def reset(self):
        # Write Reset signal to Channel B
        self.dev_gpio.gpio_Write(Port.P1, True)
        time.sleep(0.01)
        self.dev_gpio.gpio_Write(Port.P1, False)
        time.sleep(0.02)
        self.dev_gpio.gpio_Write(Port.P1, True)
        time.sleep(0.01)

    def command(self, *cmds):
        self.set_dc(False)
        for cmd in cmds:
            # SPI write on Channel nA
            self.dev_spi.spiMaster_SingleWrite(bytes([cmd]), True)

    def data(self, data_bytes):
        self.set_dc(True)
        # SPI write on Channel A
        self.dev_spi.spiMaster_SingleWrite(bytes(data_bytes), True)

    def init_display(self):
        init_cmds = [0xAE,  # Display OFF
            0xD5, 0x80,  # Set Clock Divide Ratio
            0xA8, 0x3F,  # Set Multiplex Ratio (64 lines)
            0xD3, 0x00,  # Set Display Offset
            0x40,  # Set Start Line
            0x20, 0x00,  # Horizontal Addressing Mode
            0xA1,  # Segment Re-map
            0xC8,  # COM Scan Direction
            0xDA, 0x12,  # Set COM Pins
            0x81, 0xFF,  # Set Contrast to MAX (0xFF)
            0xD9, 0xF1,  # Set Pre-charge Period
            0xDB, 0x40,  # Set VCOMH Deselect
            0xA4,  # Resume display RAM
            0xA6,  # Normal Display
            0xAF  # Display ON
        ]
        self.command(*init_cmds)
        time.sleep(0.1)  # Allow power supply rail to stabilize

    def display(self, image: Image.Image):
        """Converts image to SSD1309 page layout and transmits via SPI."""
        img = image.convert('1')
        self.command(0x21, 0, width - 1)
        self.command(0x22, 0, (height // 8) - 1)

        buf = bytearray(1024)
        pix = img.load()
        for y in range(height):
            for x in range(width):
                if pix[x, y]:
                    buf[x + (y // 8) * width] |= (1 << (y % 8))

        self.data(buf)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    try:
        oled = SSD1309()
    except:
        oled = SSD1309_sim()
        oled.show()

    box_x, box_y = 10.0, 30.0
    vx, vy = 45.0, 30.0
    text_x = 64
    ticker_text = "AZIZ"


    last_time = time.monotonic()
    frame_count = 0
    fps = 0
    fps_timer = last_time


    def run_animation():
        global last_time, box_x, box_y, vx, vy, text_x, frame_count, fps_timer, fps

        current_time = time.monotonic()
        dt = current_time - last_time
        last_time = current_time

        # Update positions
        box_x += vx * dt
        box_y += vy * dt

        if box_x <= 0 or box_x >= 118:
            vx *= -1
        if box_y <= 20 or box_y >= 54:
            vy *= -1

        text_x -= 40 * dt
        if text_x < -220:
            text_x = 128

        frame_count += 1
        if current_time - fps_timer >= 1.0:
            fps = frame_count
            frame_count = 0
            fps_timer = current_time
            print(f"Status: Rendering at {fps} FPS")

        # Draw frame
        frame = Image.new('1', (128, 64), 0)
        draw = ImageDraw.Draw(frame)

        draw.text((2, 2), f"FPS: {fps}", fill=1)
        draw.line((0, 14, 127, 14), fill=1)
        draw.text((int(text_x), 2), ticker_text, fill=1)

        wave_val = int((math.sin(current_time * 4) + 1) * 60)
        draw.rectangle((2, 58, 125, 62), outline=1, fill=0)
        draw.rectangle((3, 59, 3 + wave_val, 61), outline=0, fill=1)
        draw.rectangle((int(box_x), int(box_y), int(box_x) + 9, int(box_y) + 9), outline=1, fill=1)

        oled.display(frame)


    timer = QTimer()
    timer.timeout.connect(run_animation)
    timer.start(10)

    app.exec()