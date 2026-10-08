import re
import numpy as np
from pathlib import Path
import time

def by_find(data, needle=b'\xCA\xFE\xF0\x0D'):
    out, i = [], 0
    while (i := data.find(needle, i)) != -1:
        out.append(i)
        i += 1
    return out

def by_re(data):
    return [m.start() for m in re.finditer(rb'\xCA\xFE\xF0\x0D', data)]

def by_np_even(data):
    mask = (data[:-1] == 0xCAFE) & (data[1:] == 0xF00D)
    return np.where(mask)[0] * 2


test_data = Path(__file__).resolve().parent.parent / 'tests' / 'test_data' / 'fpga_test_buffer.bin'
with open(test_data, 'rb') as f:
    buffer = f.read()


buffer_np = np.frombuffer(buffer, dtype='>u2')
buffer_np_odd = np.frombuffer(buffer[1:-1], dtype='>u2')

t0 = time.perf_counter()
out = by_find(buffer)
t1 = time.perf_counter()
print(t1 - t0, out)

t0 = time.perf_counter()
out = by_re(buffer)
t1 = time.perf_counter()
print(t1 - t0, out)

t0 = time.perf_counter()
out = by_np_even(buffer_np)
t1 = time.perf_counter()
print(t1 - t0, out)

t0 = time.perf_counter()
out = by_np_even(buffer_np_odd)
t1 = time.perf_counter()
print(t1 - t0, out)

