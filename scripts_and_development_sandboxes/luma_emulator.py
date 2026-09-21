from luma.core.render import canvas
from luma.emulator import pygame as emulator_pygame
from luma.oled.device import ssd1309
from PIL import ImageFont
import time

# 1. Initialize the pygame emulator
# The size parameter defines the virtual display resolution (128x64 for SSD1309)
emulator = emulator_pygame(width=128, height=64, scale=2)

# 2. Initialize the SSD1309 device with the emulator
# This replaces the usual "serial" interface with the emulator
device = ssd1309(emulator)

# 3. Create a Pillow-compatible drawing canvas
with canvas(device) as draw:
    # Draw a border
    draw.rectangle(device.bounding_box, outline="white", fill="black")

    # Load a small pixel font suitable for the display
    # Note: Adjust the font path and size to fit your display
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSans.ttf", 10)
    except:
        font = None  # Fall back to default if font not found

    # Draw text at a specific position
    draw.text((10, 10), "Emulator Test", font=font, fill="white")
    draw.text((10, 30), "SSD1309 128x64", font=font, fill="white")

# 4. Keep the emulator window open
# The 'with' block flushes the buffer when it exits, but the window may close immediately.
# An explicit loop is needed to keep the pygame window running.
print("Close the emulator window to exit...")
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    pass