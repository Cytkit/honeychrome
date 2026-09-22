import sys
import time
from pathlib import Path
import numpy as np

import ft4222
from PySide6.QtCore import QTimer
from PySide6.QtGui import Qt, QImage, QPixmap
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QApplication
from ft4222.SPIMaster import Mode as MasterSingle, Clock, SlaveSelect
from ft4222.SPI import Cpha, Cpol
from ft4222.GPIO import Dir, Port
from PIL import Image, ImageDraw, ImageFont
from PIL.ImageQt import ImageQt

WIDTH = 128
HEIGHT = 64
PAGES = 8

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
        self.image_label.setFixedSize(WIDTH * self.scale, HEIGHT * self.scale)
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
            WIDTH * self.scale,
            HEIGHT * self.scale,
            Qt.IgnoreAspectRatio,      # exact multiple, no distortion
            Qt.FastTransformation,     # <-- nearest-neighbor, keeps pixels crisp
        )
        self.image_label.setPixmap(scaled)

def _dirty_rect(new: bytes, old: bytes):
    """Return (x0, x1, page0, page1) bounding box of changed bytes, or None.

    x0/x1 are inclusive column indices, page0/page1 inclusive page indices.
    """
    new_arr = np.frombuffer(new, dtype=np.uint8).reshape(PAGES, WIDTH)
    old_arr = np.frombuffer(old, dtype=np.uint8).reshape(PAGES, WIDTH)

    # Byte-level diff mask per (page, column)
    diff = new_arr != old_arr
    if not diff.any():
        return None

    # Find columns that changed anywhere
    col_any = diff.any(axis=0)  # shape (128,)
    cols = np.flatnonzero(col_any)
    x0, x1 = int(cols[0]), int(cols[-1])

    # Find pages that changed anywhere in that column span
    page_any = diff[:, x0:x1 + 1].any(axis=1)  # shape (8,)
    pages = np.flatnonzero(page_any)
    page0, page1 = int(pages[0]), int(pages[-1])

    return x0, x1, page0, page1

def _dirty_rects(new: bytes, old: bytes):
    new_arr = np.frombuffer(new, dtype=np.uint8).reshape(PAGES, WIDTH)
    old_arr = np.frombuffer(old, dtype=np.uint8).reshape(PAGES, WIDTH)
    diff = new_arr != old_arr
    if not diff.any():
        return []

    page_changed = diff.any(axis=1)
    rects = []
    p = 0
    while p < PAGES:
        if not page_changed[p]:
            p += 1
            continue
        start = p
        while p < PAGES and page_changed[p]:
            p += 1
        end = p - 1
        cols = np.flatnonzero(diff[start:end + 1].any(axis=0))
        rects.append((int(cols[0]), int(cols[-1]), start, end))
    return rects

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

        # Cache of the last transmitted buffer for diffing
        self._last_buf = None
        self.byte_tally = 0

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

    def write_data(self, data_bytes):
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
        self._last_buf = None

    def _to_page_buffer(self, image: Image.Image) -> bytes:
        """Convert a Pillow image into the SSD1309 page-addressed byte layout."""
        img = image.convert('1')
        arr = np.array(img, dtype=np.uint8)              # (64, 128)
        pages = arr.reshape(PAGES, 8, WIDTH)   # (8, 8, 128)
        bits = np.packbits(pages, axis=1, bitorder='little')  # (8, 1, 128)
        return bits.reshape(PAGES * WIDTH).tobytes()


    # def display(self, image: Image.Image):
    #     """Converts image to SSD1309 page layout and transmits via SPI."""
    #     img = image.convert('1')
    #     self.command(0x21, 0, WIDTH - 1)
    #     self.command(0x22, 0, (HEIGHT // 8) - 1)
    #
    #     arr = np.array(img, dtype=np.uint8)  # shape (64, 128)
    #     pages = arr.reshape(8, 8, 128)  # split rows into 8 pages
    #     bits = np.packbits(pages, axis=1, bitorder='little')  # shape (8, 1, 128)
    #     buf = bits.reshape(1024).tobytes()
    #
    #     self.write_data(buf)


    def _send_rect(self, buf: bytes, x0: int, x1: int, page0: int, page1: int):
        """Transmit the rectangle [x0..x1] x [page0..page1] from buf."""
        buf_arr = np.frombuffer(buf, dtype=np.uint8).reshape(PAGES, WIDTH)

        # Restrict the column and page windows
        self.command(0x21, x0, x1)
        self.command(0x22, page0, page1)

        # Extract the sub-array and flatten row-major (page-major), matching
        # the SSD1309's horizontal addressing auto-increment order.
        region = buf_arr[page0:page1 + 1, x0:x1 + 1]
        region_bytes = region.tobytes()
        self.byte_tally += len(region_bytes)
        self.write_data(region_bytes)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def display(self, image: Image.Image, force_full: bool = False):
        """Send only the changed region of the framebuffer to the SSD1309."""
        new_buf = self._to_page_buffer(image)

        if force_full or self._last_buf is None:
            self._send_rect(new_buf, 0, WIDTH - 1, 0, PAGES - 1)
            self._last_buf = new_buf
            return

        # rect = _dirty_rect(new_buf, self._last_buf)
        # if not rect:
        #     return  # nothing changed
        #
        # x0, x1, page0, page1 = rect
        # self._send_rect(new_buf, x0, x1, page0, page1)
        # self._last_buf = new_buf

        rects = _dirty_rects(new_buf, self._last_buf)
        if not rects:
            return  # nothing changed

        for rect in rects:
            x0, x1, page0, page1 = rect
            self._send_rect(new_buf, x0, x1, page0, page1)
            self._last_buf = new_buf

    def close(self):
        try:
            self.dev_spi.spiMaster_Uninit()
        except Exception:
            pass
        self.dev_spi.close()
        self.dev_gpio.close()

ASSETS_DIR = Path(__file__).parent.parent / "src" / "honeychrome" / "instrument_driver_components" / "cytkit_components" / "oled_frames"

# frame = Image.new('1', (128, 64), 0)
font4x6 = ImageFont.load(ASSETS_DIR / 'fonts' / 'pil' / '4x6.pil')
font5x7 = ImageFont.load(ASSETS_DIR / 'fonts' / 'pil' / '5x7.pil')
font5x8 = ImageFont.load(ASSETS_DIR / 'fonts' / 'pil' / '5x8.pil')
font6x9 = ImageFont.load(ASSETS_DIR / 'fonts' / 'pil' / '6x9.pil')
font6x10 = ImageFont.load(ASSETS_DIR / 'fonts' / 'pil' / '6x10.pil')
font6x12 = ImageFont.load(ASSETS_DIR / 'fonts' / 'pil' / '6x12.pil')
font6x13 = ImageFont.load(ASSETS_DIR / 'fonts' / 'pil' / '6x13.pil')
font6x13B = ImageFont.load(ASSETS_DIR / 'fonts' / 'pil' / '6x13B.pil')
font6x13O = ImageFont.load(ASSETS_DIR / 'fonts' / 'pil' / '6x13O.pil')

def load_and_convert_frame(filename=None):
    if filename:
        frame = Image.open(ASSETS_DIR / filename)
        # print(frame.mode, frame.size)
        frame = frame.convert("L").point(lambda x: 255 if x > 128 else 0, mode="1")
    else:
        frame = Image.new('1', (128, 64), 0)

    draw = ImageDraw.Draw(frame)
    return frame, draw

def draw_text_lr(draw, x, y, text, font, fill, anchor='l'):
    if anchor == 'r':
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        draw.text((x - w, y), text, font=font, fill=fill)
    else:
        draw.text((x, y), text, font=font, fill=fill)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    try:
        oled = SSD1309()
    except:
        oled = SSD1309_sim()
        oled.show()

    last_time = time.monotonic()
    frame_count = 0
    fps = 0
    fps_timer = last_time
    chapter_time = last_time
    chapter = 0

    def run_animation():
        global last_time, frame_count, fps_timer, fps, chapter, chapter_time

        current_time = time.monotonic()
        dt = current_time - last_time
        last_time = current_time

        frame_count += 1
        if current_time - fps_timer >= 1.0:
            fps = frame_count
            frame_count = 0
            fps_timer = current_time
            print(f"Status: Rendering at {fps} FPS, transfer rate {oled.byte_tally} bytes/s")
            oled.byte_tally = 0

        chapter = 6
        match chapter:
            case 0:
                frame, draw = load_and_convert_frame("connection front panel template.png")
                chapter_interval = 1
                connection_alive_animation = False

                draw_text_lr(draw, 3, 0, "Connect", font6x13, 1)
                draw_text_lr(draw, 3, 15, "USB 2.0", font6x13, 1)
                draw_text_lr(draw, 3, 36, "Cytkit is powered on.", font4x6, 1)
                draw_text_lr(draw, 3, 44, "Connect to host PC", font4x6, 1)
                draw_text_lr(draw, 3, 52, "then run Honeychrome.", font4x6, 1)

            case 1:
                frame, draw = load_and_convert_frame("logo front panel dark antenna.png")
                chapter_interval = 1
                connection_alive_animation = False

            case 2:
                frame, draw = load_and_convert_frame("logo front panel dark flash.png")
                chapter_interval = 1
                connection_alive_animation = False

            case 3:
                if frame_count % 20 > 10:
                    frame, draw = load_and_convert_frame("logo front panel template.png")
                else:
                    frame, draw = load_and_convert_frame("logo front panel template off.png")
                chapter_interval = 1
                connection_alive_animation = False


            case 4:
                frame, draw = load_and_convert_frame("logo front panel present light.png")
                chapter_interval = 2
                connection_alive_animation = True

                draw_text_lr(draw, 127, 0, "Cytkit", font6x13, 0, anchor='r')
                draw_text_lr(draw, 127, 15, "Open", font5x8, 0, anchor='r')
                draw_text_lr(draw, 127, 23, "Spectral", font5x8, 0, anchor='r')
                draw_text_lr(draw, 127, 31, "Cytometry", font5x8, 0, anchor='r')
                draw_text_lr(draw, 127, 50, "Connected!", font4x6, 0, anchor='r')


            case 5:
                frame, draw = load_and_convert_frame("info front panel flying off.png")
                chapter_interval = 1
                connection_alive_animation = True

                draw_text_lr(draw, 0, 50, "Acquiring!", font6x13, 1)

            case 6:
                frame, draw = load_and_convert_frame("info front panel arriving.png")
                chapter_interval = 10
                connection_alive_animation = True

                x_right = 65
                y_array = [0 + n*10 for n in range(5)]
                draw_text_lr(draw, x_right, y_array[0], "Trig", font5x8, 1, anchor='r')
                draw_text_lr(draw, x_right, y_array[1], "Flow", font5x8, 1, anchor='r')
                draw_text_lr(draw, x_right, y_array[2], "Pres", font5x8, 1, anchor='r')
                draw_text_lr(draw, x_right, y_array[3], "Temp", font5x8, 1, anchor='r')
                draw_text_lr(draw, x_right, y_array[4], "", font5x8, 1, anchor='r')

                x_right += 3
                draw_text_lr(draw, x_right, y_array[0], f"{np.random.randint(5000):5.0f} ev/s", font5x8, 1, anchor='l')
                draw_text_lr(draw, x_right, y_array[1], f"{np.random.random()+30:5.2f} uL/min", font5x8, 1, anchor='l')
                draw_text_lr(draw, x_right, y_array[2], f"{np.random.random()*20:5.2f} Pa", font5x8, 1, anchor='l')
                draw_text_lr(draw, x_right, y_array[3], f"{np.random.random()*20:5.2f} C", font5x8, 1, anchor='l')
                draw_text_lr(draw, x_right, y_array[4], "Laser On", font5x8, 1, anchor='l')

            case 7:
                frame, draw = load_and_convert_frame("info front panel flying off.png")
                chapter_interval = 1
                connection_alive_animation = True

                draw_text_lr(draw, 0, 50, "Stopped!", font6x13, 1)


            case _:
                chapter = 0
                return


        if connection_alive_animation:
            track_length = 256
            x_left = (current_time*500 % track_length) - track_length//2
            x_right = x_left + 64
            draw.rectangle((0, 63, 128, 63), outline=0)
            draw.rectangle((x_left, 63, x_right, 63), outline=1)

        # oled.display(frame, force_full=True)
        oled.display(frame)

        if current_time - chapter_time >= chapter_interval:
            chapter_time = current_time
            chapter += 1
            print(chapter)



    timer = QTimer()
    timer.timeout.connect(run_animation)
    timer.start(40)

    app.exec()