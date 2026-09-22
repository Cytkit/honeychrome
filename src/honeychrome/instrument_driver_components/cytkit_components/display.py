import sys
import time
from copy import deepcopy
from pathlib import Path
from threading import Thread, Event, Lock

import numpy as np

import ft4222
from ft4222.SPIMaster import Mode as MasterSingle, Clock, SlaveSelect
from ft4222.SPI import Cpha, Cpol
from ft4222.GPIO import Dir, Port
from PIL import Image, ImageDraw, ImageFont
from PIL.ImageQt import ImageQt


WIDTH = 128
HEIGHT = 64
PAGES = 8
SCALE = 6

ASSETS_DIR = Path(__file__).parent / "oled_frames"

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

frame_info, draw_info = load_and_convert_frame("info front panel arriving.png")
frame_flying, draw_flying = load_and_convert_frame("info front panel flying off.png")


def draw_text_lr(draw, x, y, text, font, fill, anchor='l'):
    if anchor == 'r':
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        draw.text((x - w, y), text, font=font, fill=fill)
    else:
        draw.text((x, y), text, font=font, fill=fill)

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

    def display(self, image: Image.Image, force_full: bool = False):
        """Send only the changed region of the framebuffer to the SSD1309."""
        new_buf = self._to_page_buffer(image)

        if force_full or self._last_buf is None:
            self._send_rect(new_buf, 0, WIDTH - 1, 0, PAGES - 1)
            self._last_buf = new_buf
            return

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

class SimProxy:
    """Drop-in replacement for SSD1309 that ships frames to a subprocess."""
    def __init__(self):
        import multiprocessing as mp
        from honeychrome.instrument_driver_components.cytkit_components.display_simulator import run_sim

        self.queue = mp.Queue(maxsize=4)
        self.proc = mp.Process(target=run_sim, args=(self.queue,), daemon=True)
        self.proc.start()
        self.byte_tally = 0

    def display(self, pil_img, force_full=False):
        # Convert to the same 1024-byte page layout the real driver uses
        from PIL import Image
        import numpy as np
        img = pil_img.convert("1")
        arr = np.array(img, dtype=np.uint8)
        pages = arr.reshape(8, 8, 128)
        bits = np.packbits(pages, axis=1, bitorder="little")
        buf = bits.reshape(1024).tobytes()

        self.byte_tally += 1024
        try:
            self.queue.put_nowait(buf)
        except Exception:
            pass  # drop frame if child is slow

    def close(self):
        self.queue.put(None)          # sentinel
        self.proc.join(timeout=2)


class Display(Thread):
    def __init__(self, transfer_object=None, sample_pump_object=None, pressure_object=None, temperature_object=None, laser_object=None):
        super().__init__(daemon=True)
        try:
            self.oled = SSD1309()
        except:
            self.oled = SimProxy()

        self.frame = Image.new('1', (128, 64), 0)
        self.draw = ImageDraw.Draw(self.frame)

        self.transmission_animation = False
        self.animate_logo = True

        self.message_timeout = 0
        self.interval = 0.1
        self._stop_event = Event()
        self._lock = Lock()
        self._closed = False

        self.transfer_object = transfer_object
        self.sample_pump_object = sample_pump_object
        self.pressure_object = pressure_object
        self.temperature_object = temperature_object
        self.laser_object = laser_object

    def run(self):
        while not self._stop_event.is_set():
            if self._closed:
                return

            if self.animate_logo:
                frame, draw = load_and_convert_frame("logo front panel dark antenna.png")
                self.oled.display(frame)
                time.sleep(1)

                frame, draw = load_and_convert_frame("logo front panel dark flash.png")
                self.oled.display(frame)
                time.sleep(1)

                for frame_count in range(30):
                    if frame_count % 2 == 0:
                        frame, draw = load_and_convert_frame("logo front panel template.png")
                    else:
                        frame, draw = load_and_convert_frame("logo front panel template off.png")
                    self.oled.display(frame)
                    time.sleep(0.033)

                frame, draw = load_and_convert_frame("logo front panel present light.png")
                draw_text_lr(draw, 127, 0, "Cytkit", font6x13, 0, anchor='r')
                draw_text_lr(draw, 127, 15, "Open", font5x8, 0, anchor='r')
                draw_text_lr(draw, 127, 23, "Spectral", font5x8, 0, anchor='r')
                draw_text_lr(draw, 127, 31, "Cytometry", font5x8, 0, anchor='r')
                draw_text_lr(draw, 127, 50, "Connected!", font4x6, 0, anchor='r')
                self.frame = frame
                self.draw = draw
                self.message_timeout = 3
                self.animate_logo = False
                self.transmission_animation = True

            self.message_timeout -= self.interval
            if self.message_timeout < 0 and self.transmission_animation:
                self.info_display()

            if self.transmission_animation:
                track_length = 256
                current_time = time.monotonic()
                x_left = (current_time * 500 % track_length) - track_length // 2
                x_right = x_left + 64
                self.draw.rectangle((0, 63, 128, 63), outline=0)
                self.draw.rectangle((x_left, 63, x_right, 63), outline=1)

            self.oled.display(self.frame)
            self._stop_event.wait(self.interval)   # interruptible sleep

    def stop(self):
        self._stop_event.set()

    def close(self):
        self.transmission_animation = False

        self.frame, self.draw = load_and_convert_frame("connection front panel template.png")
        draw_text_lr(self.draw, 3, 0, "Connect", font6x13, 1)
        draw_text_lr(self.draw, 3, 15, "USB 2.0", font6x13, 1)
        draw_text_lr(self.draw, 3, 36, "Cytkit is powered on.", font4x6, 1)
        draw_text_lr(self.draw, 3, 44, "Connect to host PC", font4x6, 1)
        draw_text_lr(self.draw, 3, 52, "then run Honeychrome.", font4x6, 1)

        self._closed = True
        self.stop()

    def disconnect(self):
        self.close()

    def action_message(self, message):
        self.frame = deepcopy(frame_flying)
        self.draw = ImageDraw.Draw(self.frame)

        if type(message) == str:
            draw_text_lr(self.draw, 0, 50, message, font6x13, 1)
        elif type(message) == list:
            draw_text_lr(self.draw, 0, 37, message[0], font6x13, 1)
            draw_text_lr(self.draw, 0, 52, message[1], font5x8, 1)

        self.message_timeout = 1

    def info_display(self):
        self.frame = deepcopy(frame_info)
        self.draw = ImageDraw.Draw(self.frame)

        event_rate = self.transfer_object.event_rate if self.transfer_object else 0
        sample_flow_rate = self.sample_pump_object.flow_rate if self.sample_pump_object else 0
        pressure = self.pressure_object.pressure if self.pressure_object else 0
        temperature = self.temperature_object.temperature if self.temperature_object else 0
        laser_enabled = self.laser_object.enabled if self.laser_object else False

        x_right = 65
        y_array = [0 + n*10 for n in range(5)]
        draw_text_lr(self.draw, x_right, y_array[0], "Trig", font5x8, 1, anchor='r')
        draw_text_lr(self.draw, x_right, y_array[1], "Flow", font5x8, 1, anchor='r')
        draw_text_lr(self.draw, x_right, y_array[2], "Pres", font5x8, 1, anchor='r')
        draw_text_lr(self.draw, x_right, y_array[3], "Temp", font5x8, 1, anchor='r')
        draw_text_lr(self.draw, x_right, y_array[4], "", font5x8, 1, anchor='r')

        x_right += 3
        draw_text_lr(self.draw, x_right, y_array[0], f"{event_rate:5.0f} ev/s", font5x8, 1, anchor='l')
        draw_text_lr(self.draw, x_right, y_array[1], f"{sample_flow_rate:5.2f} uL/min", font5x8, 1, anchor='l')
        draw_text_lr(self.draw, x_right, y_array[2], f"{pressure:5.2f} Pa", font5x8, 1, anchor='l')
        draw_text_lr(self.draw, x_right, y_array[3], f"{temperature:5.2f} C", font5x8, 1, anchor='l')
        draw_text_lr(self.draw, x_right, y_array[4], "Laser On" if laser_enabled else "Laser off", font5x8, 1, anchor='l')


if __name__ == '__main__':

    display = Display()
    display.start()
    time.sleep(10)
    display.action_message('Acquiring!')
    display.set_info(1000, 53, -18, 26, True)
    time.sleep(4)
    display.disconnect()