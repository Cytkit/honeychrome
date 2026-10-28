# Makes honeychrome importable from external plugin scripts loaded at runtime.
import sys, os

# When frozen, sys._MEIPASS is the unpacked bundle directory.
# Adding it to sys.path lets code outside the bundle (e.g. ~/Experiments/plugins/)
# do `from honeychrome.settings import ...` successfully.
meipass = getattr(sys, '_MEIPASS', None)
if meipass and meipass not in sys.path:
    sys.path.insert(0, meipass)
    print(f"[runtime-hook] inserted {meipass} into sys.path")
