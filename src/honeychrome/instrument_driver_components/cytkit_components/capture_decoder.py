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


class CapturePacket(NamedTuple):
    timestamp: int
    channel: int
    samples: np.ndarray


class CaptureDecoder:
    SYNC_WORD_0 = 0xCAFE
    SYNC_WORD_1 = 0xF00D
    SAMPLE_INVALID_MASK = 0xC000
    NUM_CHANNELS = 16
    MAX_SAMPLES = 10000

    def __init__(self) -> None:
        self.error_count: int = 0
        # Carry-over state between decode() calls: a partial header/payload
        # at the tail of the previous block.
        self._pending: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def reset(self) -> None:
        self._pending = None
        self.error_count = 0

    def decode(
        self,
        words: np.ndarray
    ) -> list[CapturePacket] | None:
        """Decode a block of uint16 words, appending completed packets to `packets`.

        `words` must be a 1-D numpy array of dtype uint16. Any partial packet
        at the end of the block is retained for the next call.
        """
        if words is None or words.size == 0:
            return

        # Ensure we're working with a contiguous uint16 array.
        if words.dtype != np.uint16:
            words = words.astype(np.uint16, copy=False)
        if not words.flags["C_CONTIGUOUS"]:
            words = np.ascontiguousarray(words)

        # Prepend any leftover from last call.
        if self._pending is not None and self._pending.size:
            buf = np.concatenate((self._pending, words))
        else:
            buf = words
        self._pending = None

        return self._process(buf)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _process(self, buf: np.ndarray):
        """Scan `buf`, return packets.

        `buf` is a uint16 numpy array. Called with the concatenation of any
        pending data and the new block.
        """
        n = buf.size
        pos = 0
        packets: list[CapturePacket] = []

        # Pre-extract constants into locals for speed.
        SYNC0 = self.SYNC_WORD_0
        SYNC1 = self.SYNC_WORD_1
        INVALID = self.SAMPLE_INVALID_MASK
        NCH = self.NUM_CHANNELS
        MAXS = self.MAX_SAMPLES
        HEADER_LEN = 8  # sync0, sync1, ts0..ts3, channel, count

        while pos < n:
            # ---- 1. Find next candidate header at or after `pos`.
            #        We search for SYNC0, then verify SYNC1 follows.
            idx = self._find_header(buf, pos, SYNC0, SYNC1)
            if idx is None:
                # No header in the remainder. Keep the last 7 words (a header
                # can't be split shorter than that) as pending, drop the rest.
                keep = max(0, n - pos - 7)
                if keep > 0:
                    self._pending = buf[n - keep:].copy()
                return

            # ---- 2. Do we have a full header in this buffer?
            if idx + HEADER_LEN > n:
                # Need more data. Keep from idx onward as pending.
                self._pending = buf[idx:].copy()
                return

            # ---- 3. Parse header fields (vectorized, tiny).
            ts_words = buf[idx + 2 : idx + 6].astype(np.uint64)
            timestamp = int(
                ts_words[0]
                | (ts_words[1] << 16)
                | (ts_words[2] << 32)
                | (ts_words[3] << 48)
            )
            channel = int(buf[idx + 6])
            count = int(buf[idx + 7])

            # ---- 4. Validate header.
            if channel >= NCH or count > MAXS:
                # Bad header. Re-sync from the word *after* the first sync word,
                # so we can still find a header that overlaps the bad one.
                self.error_count += 1
                pos = idx + 1
                continue

            # ---- 5. Do we have the full payload?
            payload_start = idx + HEADER_LEN
            payload_end = payload_start + count
            if payload_end > n:
                # Need more data. Keep from idx onward as pending.
                self._pending = buf[idx:].copy()
                return

            payload = buf[payload_start:payload_end]

            # ---- 6. Validate payload in bulk: any word with bits 15:14 set?
            if count > 0:
                bad = np.flatnonzero(payload & INVALID)
                if bad.size:
                    # Lost alignment inside payload. Abandon, re-sync from
                    # the offending word so it can be re-examined as a header.
                    self.error_count += 1
                    pos = payload_start + int(bad[0])
                    continue

            # ---- 7. Emit packet.
            packets.append(CapturePacket(timestamp, channel, payload.copy()))

            pos = payload_end

        # Looped off the end cleanly.
        self._pending = None

        return packets

    @staticmethod
    def _find_header(
        buf: np.ndarray,
        start: int,
        sync0: int,
        sync1: int,
    ) -> Optional[int]:
        """Return the index of the next (SYNC0, SYNC1) pair at or after `start`.

        Returns None if no candidate is found.
        """
        n = buf.size
        # Search for SYNC0 starting at `start`.
        while True:
            # np.flatnonzero on a boolean compare is C-speed.
            hits = np.flatnonzero(buf[start:n - 1] == sync0)
            if hits.size == 0:
                return None
            # Filter to those where the next word is SYNC1.
            cand = hits + start
            good = cand[buf[cand + 1] == sync1]
            if good.size:
                return int(good[0])
            # No SYNC1 followed any SYNC0. Advance past the last SYNC0 and retry
            # (its position could still be a *start* of another attempt only if
            # it's the last word — which we excluded by n-1).
            start = int(cand[-1]) + 1
            if start >= n - 1:
                return None