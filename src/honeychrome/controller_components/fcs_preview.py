"""Read a uniform random preview of an FCS file without loading the whole DATA segment.

The bounded-read approach (metadata first, then only the sampled events) originates
from Joan Escrivà Font's pull request #16 (Cytkit/honeychrome).
"""
from pathlib import Path

from flowio import FlowData
import numpy as np

PREVIEW_SEED = 42
_BLOCK_ROWS = 16384
_BYTE_ORDER_PREFIX = {
    '1,2,3,4': '<', '1,2,3,4,5,6,7,8': '<',
    '4,3,2,1': '>', '8,7,6,5,4,3,2,1': '>',
}


class PreviewSample:
    """Display-only stand-in for a flowkit Sample, holding the preview events and the file metadata."""

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
    """Parse only the HEADER and TEXT segments."""
    return FlowData(path, only_text=True, use_header_offsets=True)


def preview_indices(n_events, max_events, seed=PREVIEW_SEED):
    """Sorted random event rows; every row when the file is within the limit."""
    if max_events >= n_events:
        return np.arange(n_events)
    return np.sort(np.random.default_rng(seed).choice(n_events, int(max_events), replace=False))


def load_fcs_preview(path, max_events, col_order=None, metadata=None, seed=PREVIEW_SEED):
    """Return (PreviewSample, events) for at most max_events randomly chosen events.

    Events carry the same gain, log and time scaling as FlowIO's as_array().
    Only list-mode float32/float64 files are supported; anything else raises
    ValueError so the caller can fall back to a full load.
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
    if byteorder not in _BYTE_ORDER_PREFIX:
        raise ValueError(f'Unsupported FCS byte order: {byteorder!r}')
    dtype = np.dtype(_BYTE_ORDER_PREFIX[byteorder] + ('f4' if kind == 'F' else 'f8'))
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
    indices = preview_indices(meta.event_count, max_events, seed)
    events = np.empty((len(indices), len(columns)), dtype=np.float64)

    # Gather rows block by block; the OS pages in only what is touched
    data = np.memmap(path, dtype=dtype, mode='r', offset=offset,
                     shape=(meta.event_count, meta.channel_count))
    try:
        for start in range(0, len(indices), _BLOCK_ROWS):
            rows = indices[start:start + _BLOCK_ROWS]
            events[start:start + len(rows)] = data[rows][:, columns]
    finally:
        del data

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
