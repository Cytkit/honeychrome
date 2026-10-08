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

SYNC_WORD_0 = b'\xCA\xFE'
SYNC_WORD_1 = b'\xF0\x0D'
SAMPLE_INVALID_MASK = 0xC000
MAX_SAMPLES = 10000
HEADER_LEN = 16  # sync0, sync1, ts0..ts3, channel, count
N_CHANNELS = len(adc_dictionary)


class CapturePacket(NamedTuple):
    timestamp: int
    channel: int
    samples: np.ndarray

class CaptureDecoder:
    def __init__(self) -> None:
        self.error_count: int = 0
        # Carry-over state between decode() calls: a partial header/payload
        # at the tail of the previous block.
        self._pending: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.error_count = 0

    def decode(self, buffer):
        """Decode the buffer, assemble a list of traces.

        `words` must be a 1-D numpy array of dtype uint16. Any partial packet
        at the end of the block is retained for the next call.
        """
        if buffer is None or buffer.size == 0:
            return None

        packets = self.extract_packets(buffer)

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

    def extract_packets(self, buffer):
        """Scan `buf`, return packets.

        `buf` is a uint16 numpy array. Called with the concatenation of any
        pending data and the new block.
        """
        n = len(buffer)
        pos = 0
        packets = []

        while pos < n:
            # ---- 1. Find next candidate header at or after `pos`.
            #        We search for SYNC0, then verify SYNC1 follows.
            pos = buffer.find(SYNC_WORD_0, pos)
            if pos == -1:
                break
            if buffer[pos + 2] != SYNC_WORD_1:
                continue

            # ---- 2. Do we have a full header in this buffer?
            if pos + HEADER_LEN > n:
                break

            # ---- 3. Parse header fields (vectorized, tiny).
            ts_words = buffer[pos + 4: pos + 12].astype(np.uint64)
            timestamp = int(ts_words[0] | (ts_words[1] << 16) | (ts_words[2] << 32) | (ts_words[3] << 48))
            channel = int(buffer[pos + 12])
            count = int(buffer[pos + 14])

            # ---- 4. Validate header.
            if channel >= N_CHANNELS or count > MAX_SAMPLES:
                # Bad header. Re-sync from the word *after* the first sync word,
                # so we can still find a header that overlaps the bad one.
                self.error_count += 1
                continue

            # ---- 5. Do we have the full payload?
            payload_start = pos + HEADER_LEN
            payload_end = payload_start + count
            if payload_end > n:
                break

            payload = buffer[payload_start:payload_end]

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
            packets.append(CapturePacket(timestamp, channel, payload.copy()))

            pos = payload_end

        return packets
