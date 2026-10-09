###############################################################
# Name:      Capture_Decoder.py
# Purpose:   Decodes the capture sample stream into packets (NumPy)
# Author:    Catherine Vant ()
# Created:   2026-09-26
# Copyright: Catherine Vant ()
###############################################################

from typing import List, Optional
import numpy as np
from typing import NamedTuple
from honeychrome import settings
from honeychrome.instrument_driver_components.cytkit_components.cytkit_configuration import adc_dictionary

SYNC_WORD_0 = 0xCAFE
SYNC_WORD_1 = 0xF00D
SAMPLE_INVALID_MASK = 0xC000
MAX_SAMPLES = 10000
HEADER_LEN = 16  # sync0, sync1, ts0..ts3, channel, count
N_CHANNELS = 16


class CapturePacket(NamedTuple):
    timestamp: int
    channel: int
    samples: np.ndarray

class CaptureDecoder:
    def __init__(self) -> None:
        self.error_count: int = 0
        self._carry = b''  # trailing bytes of an incomplete packet, prepended to the next read

    def reset(self) -> None:
        self.error_count = 0
        self._carry = b''

    def decode(self, buffer):
        """Decode the buffer, assemble an array of traces.

        `buffer` is the raw byte stream from the device (big-endian 16-bit words). Any partial packet
        at the end of the block is retained and prepended to the next call, so a packet that straddles
        two reads is not lost.
        """
        if buffer is None or len(buffer) == 0:
            return None

        packets = self.extract_packets(bytes(buffer), carry=True)

        if packets:
            N = len({p.timestamp for p in packets}) # number of events
            traces = np.zeros((N, settings.n_channels_trace, settings.n_time_points_in_event), dtype=np.uint16)

            # Map timestamp -> event index, assigning indices in first-seen order.
            event_index = {}
            for packet in packets:
                idx = event_index.get(packet.timestamp)
                if idx is None:
                    idx = len(event_index)
                    event_index[packet.timestamp] = idx

                n = min(packet.samples.size, settings.n_time_points_in_event)
                if n > 0:
                    traces[idx, packet.channel, :n] = packet.samples[:n]

            return traces
        else:
            return None

    def extract_packets(self, buffer, carry=False):
        """Scan `buffer` (bytes), return packets.

        The stream is a sequence of big-endian 16-bit words, so every search and step is done in whole
        words (a byte-wise search can false-match across a word boundary). If `carry` is True the unused
        tail is kept for the next call, otherwise it is discarded.
        """
        if carry:
            buffer = self._carry + buffer
            self._carry = b''

        n = len(buffer) // 2  # whole words only
        words = np.frombuffer(buffer, dtype='>u2', count=n)
        pos = 0
        packets = []
        keep_from = None  # word index of an incomplete packet to carry over

        while pos < n:
            # ---- 1. Find next candidate header at or after `pos`: SYNC0 followed by SYNC1.
            candidates = np.flatnonzero(words[pos:] == SYNC_WORD_0)
            if candidates.size == 0:
                break
            pos += int(candidates[0])

            if pos + 1 >= n:
                keep_from = pos  # SYNC0 is the last word, wait for more data
                break
            if words[pos + 1] != SYNC_WORD_1:
                pos += 1  # not a header; keep hunting from the next word
                continue

            # ---- 2. Do we have a full header in this buffer?
            if pos + HEADER_LEN // 2 > n:
                keep_from = pos
                break

            # ---- 3. Parse header fields.
            timestamp = (int(words[pos + 2]) | (int(words[pos + 3]) << 16) |
                         (int(words[pos + 4]) << 32) | (int(words[pos + 5]) << 48))
            channel = int(words[pos + 6])
            count = int(words[pos + 7])

            # ---- 4. Validate header.
            if channel >= N_CHANNELS or count > MAX_SAMPLES:
                # Bad header. Re-sync from the word after the first sync word.
                self.error_count += 1
                pos += 1
                continue

            # ---- 5. Do we have the full payload?
            payload_start = pos + HEADER_LEN // 2
            payload_end = payload_start + count
            if payload_end > n:
                keep_from = pos
                break

            payload = words[payload_start:payload_end]

            # ---- 6. Validate payload in bulk: any word with bits 15:14 set?
            if count > 0:
                bad = np.flatnonzero(payload & SAMPLE_INVALID_MASK)
                if bad.size:
                    # Lost alignment inside payload. Abandon, re-sync from
                    # the offending word so it can be re-examined as a header.
                    self.error_count += 1
                    pos = payload_start + int(bad[0])
                    continue

            # ---- 7. Emit packet.
            packets.append(CapturePacket(timestamp, channel, payload.astype(np.uint16)))

            pos = payload_end

        if carry:
            if keep_from is not None:
                self._carry = buffer[keep_from * 2:]  # includes any odd trailing byte
            elif len(buffer) % 2:
                self._carry = buffer[-1:]  # odd trailing byte belongs to the next word

        return packets


if __name__ == "__main__":
    from pathlib import Path
    test_data = Path(__file__).resolve().parent.parent.parent.parent.parent / 'tests' / 'test_data' / 'fpga_test_buffer.bin'
    with open(test_data, 'rb') as f:
        data = f.read()

    decoder = CaptureDecoder()
    packets = decoder.extract_packets(data)
    print(packets)
    decoder.decode(data)