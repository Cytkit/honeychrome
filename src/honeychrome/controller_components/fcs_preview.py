"""Read a deterministic uniform preview without materialising the FCS DATA segment."""
from pathlib import Path
import random

from flowio import FlowData
import numpy as np


class PreviewSample:
    """Display-only sample; full-file exports continue to use sample_from_fcs."""

    def __init__(self, path, metadata, events, labels):
        self.sample_id = Path(path).name
        self.pnn_labels = list(labels)
        self.event_count = metadata.event_count
        self.preview_event_count = len(events)
        self.is_display_subset = len(events) < self.event_count
        self.metadata = dict(metadata.text)
        self._events = events

    def get_metadata(self):
        return self.metadata

    def get_events(self, source='raw', col_order=None):
        if source != 'raw':
            raise ValueError('PreviewSample stores only raw display events.')
        if col_order is None:
            return self._events
        return self._events[:, [self.pnn_labels.index(name) for name in col_order]]


def read_preview_metadata(path):
    return FlowData(path, only_text=True, use_header_offsets=True)


def load_fcs_preview(path, max_events, col_order=None, metadata=None, seed=42):
    """Return uniformly sampled events with FlowIO gain/log/time scaling.

    Reads are bounded. Unsupported or truncated files raise without a full load.
    """
    if max_events <= 0:
        raise ValueError('A positive preview event limit is required.')
    path = Path(path)
    meta = metadata if metadata is not None else read_preview_metadata(path)
    text = meta.text
    kind = text.get('datatype', '').upper()
    if text.get('mode', 'L').upper() != 'L' or kind not in ('F', 'D'):
        raise ValueError('Streaming preview supports list-mode float32/float64 FCS files.')
    byteorder = text.get('byteord', '').replace(' ', '')
    if byteorder not in ('1,2,3,4', '4,3,2,1', '1,2,3,4,5,6,7,8', '8,7,6,5,4,3,2,1'):
        raise ValueError(f'Unsupported FCS byte order: {byteorder!r}')
    dtype = np.dtype(
        ('<' if byteorder.startswith('1,') else '>') + ('f4' if kind == 'F' else 'f8')
    )
    expected_bits = dtype.itemsize * 8
    if any(int(text.get(f'p{i}b', expected_bits)) != expected_bits
           for i in range(1, meta.channel_count + 1)):
        raise ValueError('FCS parameter bit widths disagree with the DATA type.')
    offset = int(text.get('begindata', 0) or 0) or meta.header['data_start']
    row_bytes = meta.channel_count * dtype.itemsize
    if offset <= 0 or path.stat().st_size < offset + meta.event_count * row_bytes:
        raise ValueError('Invalid offset or truncated FCS DATA segment.')
    labels = meta.pnn_labels if col_order is None else list(col_order)
    columns = [meta.pnn_labels.index(name) for name in labels]
    count = min(int(max_events), meta.event_count)
    if count * len(columns) * 8 > 512 * 1024 ** 2:
        raise ValueError('Preview exceeds 512 MB; reduce Max display events.')
    # random.sample(range(...)) avoids an array proportional to total events.
    indices = np.array(
        sorted(random.Random(seed).sample(range(meta.event_count), count)), dtype=np.int64
    )
    events = np.empty((count, len(columns)), dtype=np.float64)
    breaks = np.r_[0, np.flatnonzero(np.diff(indices) != 1) + 1, count]
    with path.open('rb', buffering=0) as stream:
        for first, stop in zip(breaks[:-1], breaks[1:]):
            # Limit each temporary read even when every event is selected.
            for start in range(int(first), int(stop), 4096):
                end = min(start + 4096, int(stop))
                stream.seek(offset + int(indices[start]) * row_bytes)
                block = stream.read((end - start) * row_bytes)
                if len(block) != (end - start) * row_bytes:
                    raise ValueError('FCS DATA segment changed or was truncated while reading.')
                rows = np.frombuffer(block, dtype=dtype).reshape(-1, meta.channel_count)
                events[start:end] = rows[:, columns]
    for output_column, file_column in enumerate(columns):
        channel = meta.channels[file_column + 1]
        values = events[:, output_column]
        if file_column == meta.time_index:
            values *= float(text.get('timestep', '1').strip() or '1')
        decades, log_zero = channel['pne']
        if decades > 0:
            values[:] = 10 ** (decades * values / channel['pnr']) * log_zero
        gain = channel['png']
        if gain not in (0, 1):
            values /= gain
    np.nan_to_num(events, copy=False, nan=0.0, posinf=np.inf, neginf=-np.inf)
    return PreviewSample(path, meta, events, labels), events
