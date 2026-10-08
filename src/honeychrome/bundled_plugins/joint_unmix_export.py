"""
joint_unmix_export.py
---------------------
Batch FCS export through a caller-supplied unmixing function, with a
run-time estimate and a confirm/cancel prompt for exports that use the
compiled joint AF + fluorophore-variant kernel.

Used by the AutoSpectral Optimization plugin and by plugins that build on
its kernel. Typical use, on the main thread:

    layout = build_export_layout(controller)
    exporter = JointUnmixExporter(sample_paths, layout, unmix_fn, bus=bus)
    run = JointUnmixRun(exporter, parent_widget=self, probe_fn=probe_fn)
    run.done.connect(on_done)
    run.start()

``unmix_fn(raw_fl_chunk, raw_chunk, af)`` is called once per chunk of each
sample. ``raw_fl_chunk`` holds the fluorescence detectors used for
unmixing, ``raw_chunk`` every exported raw channel (``layout.pnn_raw_export``
order) and ``af`` the sample's ``AFLibrary``. It returns a dict with
``unmixed`` (n x F uncompensated fluorophore abundances, in
``layout.fluor_names`` order), ``af_scale`` (n,), ``af_idx`` (n,) 1-based
into ``af.spectra``, and ``extra`` (n x len(extra_channels)) when the
exporter writes extra channels.

``probe_fn(raw_fl, af)`` runs the slow part of the same unmixing on a block
of raw fluorescence events. When given, ``JointUnmixRun`` times it on a
probe of ``PROBE_EVENTS`` events drawn from every selected sample in
proportion to its event count, unmixed with the most frequently assigned AF
profile set, and asks the user to confirm the estimated time for the whole
run before exporting. Leave it out for fast unmixing (e.g. AF only).
"""

from __future__ import annotations

import gc
import logging
import math
import time
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal
from PySide6.QtWidgets import QMessageBox

import honeychrome.settings as settings
from honeychrome.controller_components.autospectral_functions import af_index_lookup
from honeychrome.controller_components.functions import (
    apply_transfer_matrix,
    export_unmixed_sample,
    sample_from_fcs,
)

logger = logging.getLogger(__name__)

# Events per chunk: bounds the per-cell variant-search buffers regardless of
# file size (unmix_fcs.R's chunk.size default).
UNMIX_CHUNK_SIZE = 2_000_000

# Probe size for the run-time estimate (estimate.unmix.time()'s default).
PROBE_EVENTS = 5000


# ---------------------------------------------------------------------------
# Export layout
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class AFLibrary:
    """Combined AF library for one sample's profile assignment.

    ``index_map`` holds the experiment-wide AF Index of each library row.
    """
    profile_names: tuple
    spectra: np.ndarray
    index_map: np.ndarray


@dataclass(frozen=True, eq=False)
class ExportLayout:
    """Everything an export needs, snapshotted from the controller.

    The raw channel space is the whitelisted raw channels plus any
    FACSDiscover imaging channels, which are carried straight through to the
    output by an identity block in ``transfer_matrix``. ``transfer_matrix``
    (raw -> unmixed, applied as ``raw @ transfer_matrix``) is uncompensated:
    files are written uncompensated with the fine-tuning matrix in
    ``$SPILLOVER``.
    """
    experiment_dir: Path
    raw_subdir_abs: Path
    unmixed_subdirectory: str
    pnn_raw_export: tuple
    fl_ids_raw: tuple
    pnn_unmixed: tuple
    fl_ids_unmixed: np.ndarray
    transfer_matrix: np.ndarray
    spillover: np.ndarray
    reference_spectra: np.ndarray
    fluor_names: tuple
    unmixing_spectra: np.ndarray | None
    raw_settings: dict
    unmixed_settings: dict
    spectral_model: list
    all_samples: dict
    sample_af_profiles: dict
    af_profiles: dict
    n_af_index: int | None


def _imaging_channels(raw_settings: dict, pnn_raw: list, pnn_raw_full: list) -> list:
    """FACSDiscover imaging channels not already in the whitelist."""
    if 'FACSDiscover' not in raw_settings.get('cytometer', ''):
        return []
    import re
    from honeychrome.controller_components.cytometer_whitelist import _CYTOMETER_PARAMS

    params = _CYTOMETER_PARAMS.get('FACSDiscover')
    if params is None:
        return []
    exclude_from_imaging = {'FSC', 'SSC', 'Time'}
    imaging_prefixes = [
        p for p in params.non_spectral_pat
        if not p.startswith('-') and p not in exclude_from_imaging
    ]
    imaging_pat = re.compile('|'.join(rf'(?:^|\b){re.escape(p)}' for p in imaging_prefixes))
    already_whitelisted = set(pnn_raw)
    if raw_settings.get('event_id_channel_id') is not None:
        already_whitelisted = already_whitelisted | {pnn_raw_full[raw_settings['event_id_channel_id']]}
    return [ch for ch in pnn_raw_full if re.search(imaging_pat, ch) and ch not in already_whitelisted]


def build_export_layout(controller) -> ExportLayout:
    """Snapshot the export layout from *controller*. Main thread only."""
    exp = controller.experiment
    raw_settings = deepcopy(exp.settings['raw'])
    unmixed_settings = deepcopy(exp.settings['unmixed'])
    experiment_dir = Path(controller.experiment_dir)

    pnn_raw_full = raw_settings['event_channels_pnn']
    pnn_raw = list(raw_settings.get('whitelisted_pnn') or pnn_raw_full)
    imaging_pnn = _imaging_channels(raw_settings, pnn_raw, pnn_raw_full)
    pnn_raw_export = pnn_raw + imaging_pnn
    pnn_unmixed = list(unmixed_settings['event_channels_pnn'])
    n_unmixed_before_imaging = len(pnn_unmixed)
    pnn_unmixed = pnn_unmixed + imaging_pnn

    fl_ids_raw = [pnn_raw_export.index(pnn_raw_full[i]) for i in controller.filtered_raw_fluorescence_channel_ids]
    sc_ids_raw = [pnn_raw_export.index(pnn_raw_full[i]) for i in raw_settings['scatter_channel_ids']]
    fl_ids_unmixed = np.array(unmixed_settings['fluorescence_channel_ids'])
    sc_ids_unmixed = np.array(unmixed_settings['scatter_channel_ids'])
    n_scatter = unmixed_settings['n_scatter_channels']

    unmixing_matrix = np.array(exp.process['unmixing_matrix'])
    spillover = np.array(exp.process['spillover'])

    transfer_matrix = np.zeros((len(pnn_unmixed), len(pnn_raw_export)))
    transfer_matrix[np.ix_(fl_ids_unmixed, fl_ids_raw)] = unmixing_matrix
    transfer_matrix[np.ix_(sc_ids_unmixed, sc_ids_raw)] = np.eye(n_scatter)
    if raw_settings.get('time_channel_id') is not None:
        raw_time_id = pnn_raw_export.index(pnn_raw_full[raw_settings['time_channel_id']])
        transfer_matrix[unmixed_settings['time_channel_id'], raw_time_id] = 1
    for k in range(len(imaging_pnn)):
        transfer_matrix[n_unmixed_before_imaging + k, len(pnn_raw) + k] = 1.0
    transfer_matrix = transfer_matrix.T

    profiles = exp.process.get('profiles', {})
    fluor_names = [c['label'] for c in exp.process.get('spectral_model', []) if c['label'] in profiles]
    spectra_matrix = exp.process.get('spectra_matrix')

    return ExportLayout(
        experiment_dir=experiment_dir,
        raw_subdir_abs=(experiment_dir / raw_settings['raw_samples_subdirectory']).resolve(),
        unmixed_subdirectory=unmixed_settings['unmixed_samples_subdirectory'],
        pnn_raw_export=tuple(pnn_raw_export),
        fl_ids_raw=tuple(fl_ids_raw),
        pnn_unmixed=tuple(pnn_unmixed),
        fl_ids_unmixed=fl_ids_unmixed,
        transfer_matrix=transfer_matrix,
        spillover=spillover,
        reference_spectra=controller._build_fluor_spectra(),
        fluor_names=tuple(fluor_names),
        unmixing_spectra=np.array(spectra_matrix) if spectra_matrix is not None else None,
        raw_settings=raw_settings,
        unmixed_settings=unmixed_settings,
        spectral_model=deepcopy(exp.process.get('spectral_model', [])),
        all_samples=dict(exp.samples.get('all_samples', {})),
        sample_af_profiles=deepcopy(exp.samples.get('sample_af_profiles', {})),
        af_profiles=deepcopy(exp.process.get('af_profiles', {})),
        n_af_index=controller.n_af_spectra(),
    )


def af_library_for_profiles(layout: ExportLayout, profile_names) -> AFLibrary | None:
    """Combined AF library for *profile_names*, or None with fewer than 2 spectra."""
    names = [str(n) for n in (profile_names or [])]
    stored = [n for n in names if n in layout.af_profiles]
    if not stored:
        return None
    spectra = np.vstack([np.array(layout.af_profiles[n]['spectra']) for n in stored])
    if spectra.shape[0] < 2:
        return None
    return AFLibrary(
        profile_names=tuple(stored),
        spectra=spectra,
        index_map=af_index_lookup(layout.af_profiles, names),
    )


def af_library_for_sample(layout: ExportLayout, sample_path: str) -> AFLibrary | None:
    """AF library assigned to *sample_path*, or None if it cannot be AF-unmixed."""
    return af_library_for_profiles(layout, layout.sample_af_profiles.get(sample_path, []))


def read_export_events(full_path, layout: ExportLayout, bus=None) -> tuple:
    """Raw events in ``layout.pnn_raw_export`` order (missing channels 0, NaN 0).

    Returns ``(raw_event_data, raw_keywords, n_events)``.
    """
    sample = sample_from_fcs(full_path, bus)
    all_events = sample.get_events(source='raw')
    sample_ch_idx = {ch: i for i, ch in enumerate(sample.pnn_labels)}
    pnn = list(layout.pnn_raw_export)
    if set(pnn) <= set(sample.pnn_labels):
        raw_event_data = all_events[:, [sample_ch_idx[ch] for ch in pnn]]
    else:
        raw_event_data = np.zeros((all_events.shape[0], len(pnn)), dtype=all_events.dtype)
        for dst, ch in enumerate(pnn):
            if ch in sample_ch_idx:
                raw_event_data[:, dst] = all_events[:, sample_ch_idx[ch]]
    np.nan_to_num(raw_event_data, copy=False, nan=0.0)
    return raw_event_data, sample.get_metadata(), sample.event_count


def export_relative_path(sample_path: str, layout: ExportLayout) -> Path:
    """Path of *sample_path* relative to the raw samples subdirectory."""
    p = Path(sample_path)
    sample_abs = p.resolve() if p.is_absolute() else (layout.experiment_dir / p).resolve()
    for parent in [sample_abs] + list(sample_abs.parents):
        try:
            if parent.samefile(layout.raw_subdir_abs):
                return sample_abs.relative_to(parent)
        except OSError:
            pass
    return Path(*sample_abs.parts[len(layout.raw_subdir_abs.parts):])


# ---------------------------------------------------------------------------
# Exporter
# ---------------------------------------------------------------------------

class JointUnmixExporter(QObject):
    """Unmix and write each sample, chunk by chunk, with ``unmix_fn``.

    Output columns: the unmixed channels (scatter, time and imaging channels
    via ``layout.transfer_matrix``, fluorophores from ``unmix_fn``), AF
    Abundance, AF Index, then ``extra_channels``. Files go to
    ``output_root`` (default: the experiment's unmixed samples
    subdirectory), mirroring each sample's location under the raw samples
    subdirectory, named ``<sample><name_suffix> (Unmixed).fcs``. Samples
    without an AF library of at least 2 spectra are skipped.

    ``sample_done(sample_path, n_events)``, when given, is called on the
    export thread after each sample's file is written, so a caller can
    attribute per-chunk results of ``unmix_fn`` to their sample.

    ``prepare_fn(raw_fl, raw, af)``, when given, is called once per sample
    with all its events (fluorescence detectors, every exported raw
    channel) and its AF library before the first chunk, for estimates that
    need the whole sample.

    FlowKit loads each file whole; chunking bounds only the unmixing
    buffers.
    """
    progress = Signal(int, int)
    finished = Signal()
    error = Signal(str)

    def __init__(self, sample_paths, layout: ExportLayout, unmix_fn, bus=None,
                 extra_channels=(), output_root=None, name_suffix: str = '',
                 unmixing_method: str = 'AutoSpectral Optimization',
                 extra_keywords: dict | None = None, chunk_size: int = UNMIX_CHUNK_SIZE,
                 sample_done=None, prepare_fn=None):
        super().__init__()
        self.sample_paths = list(sample_paths)
        self.layout = layout
        self.unmix_fn = unmix_fn
        self.bus = bus
        self.extra_channels = list(extra_channels)
        self.output_root = Path(output_root) if output_root is not None else None
        self.name_suffix = name_suffix
        self.unmixing_method = unmixing_method
        self.extra_keywords = dict(extra_keywords) if extra_keywords else None
        self.chunk_size = int(chunk_size)
        self.sample_done = sample_done
        self.prepare_fn = prepare_fn

    def _export_pnn(self) -> list:
        export_pnn = list(self.layout.pnn_unmixed)
        for label in settings.af_channels:
            if label not in export_pnn:
                export_pnn.append(label)
        for name in self.extra_channels:
            if name in export_pnn:
                raise ValueError(f'Extra channel "{name}" has the same name as an existing channel.')
            export_pnn.append(name)
        return export_pnn

    def _output_folder(self, sample_path: str) -> Path:
        rel = export_relative_path(sample_path, self.layout)
        root = self.output_root or (self.layout.experiment_dir / self.layout.unmixed_subdirectory)
        return (root / rel).parent

    def run(self):
        try:
            from honeychrome import __version__

            layout = self.layout
            export_pnn = self._export_pnn()
            n_unmixed_cols = len(layout.pnn_unmixed)
            af_abundance_col = export_pnn.index(settings.af_abundance_channel)
            af_index_col = export_pnn.index(settings.af_index_channel)
            extra_start = len(export_pnn) - len(self.extra_channels)
            fl_ids_raw = list(layout.fl_ids_raw)

            total = len(self.sample_paths)
            n_exported = 0
            n_skipped = 0
            for n, sample_path in enumerate(self.sample_paths):
                self.progress.emit(n, total)

                af = af_library_for_sample(layout, sample_path)
                if af is None:
                    logger.warning(f'{self.unmixing_method} export: "{sample_path}" has no AF profile '
                                   f'with at least 2 spectra assigned — skipping.')
                    n_skipped += 1
                    continue

                raw_event_data, raw_keywords, n_events = read_export_events(
                    layout.experiment_dir / sample_path, layout, self.bus,
                )
                if n_events == 0:
                    n_skipped += 1
                    continue

                if self.prepare_fn is not None:
                    self.prepare_fn(raw_event_data[:, fl_ids_raw], raw_event_data, af)
                export_event_data = np.zeros((n_events, len(export_pnn)), dtype=np.float64)
                n_chunks = math.ceil(n_events / self.chunk_size)
                for chunk_i in range(n_chunks):
                    s_row = chunk_i * self.chunk_size
                    e_row = min((chunk_i + 1) * self.chunk_size, n_events)
                    if n_chunks > 1:
                        logger.debug(f'{self.unmixing_method} export: "{sample_path}" '
                                     f'chunk {chunk_i + 1}/{n_chunks} (events {s_row}-{e_row})')

                    raw_chunk = raw_event_data[s_row:e_row]
                    raw_fl_chunk = raw_chunk[:, fl_ids_raw]
                    result = self.unmix_fn(raw_fl_chunk, raw_chunk, af)

                    unmixed_chunk = apply_transfer_matrix(layout.transfer_matrix, raw_chunk)
                    unmixed_chunk[:, layout.fl_ids_unmixed] = result['unmixed']

                    export_event_data[s_row:e_row, :n_unmixed_cols] = unmixed_chunk
                    export_event_data[s_row:e_row, af_abundance_col] = result['af_scale']
                    export_event_data[s_row:e_row, af_index_col] = (
                        af.index_map[np.asarray(result['af_idx'], dtype=np.int64) - 1]
                    )
                    if self.extra_channels:
                        export_event_data[s_row:e_row, extra_start:] = result['extra']

                    del raw_chunk, raw_fl_chunk, result, unmixed_chunk
                    if n_chunks > 1 and chunk_i % 5 == 4:
                        gc.collect()

                del raw_event_data

                output_folder = self._output_folder(sample_path)
                output_folder.mkdir(parents=True, exist_ok=True)
                sample_name = layout.all_samples.get(sample_path, Path(sample_path).stem)

                export_unmixed_sample(
                    sample_name=f'{sample_name}{self.name_suffix}',
                    unmixed_folder=output_folder,
                    export_event_data=export_event_data,
                    export_pnn=export_pnn,
                    spillover=layout.spillover,
                    raw_keywords=raw_keywords,
                    spectral_model=layout.spectral_model,
                    unmixed_settings=layout.unmixed_settings,
                    raw_settings=layout.raw_settings,
                    af_spectra=af.spectra,
                    unmixing_spectra=layout.unmixing_spectra,
                    version=__version__,
                    subsample=None,
                    extra_null_channels=None,
                    unmixing_method=self.unmixing_method,
                    unmixing_weights=None,
                    af_index_map=af.index_map,
                    n_af_index=layout.n_af_index,
                    extra_keywords=self.extra_keywords,
                )
                n_exported += 1
                if self.sample_done is not None:
                    self.sample_done(sample_path, n_events)

            self.progress.emit(total, total)
            if self.bus:
                destination = (str(self.output_root) if self.output_root is not None
                               else f'"{layout.unmixed_subdirectory}" folder in experiment folder')
                message = (f'Exported {n_exported} sample(s) with {self.unmixing_method} '
                           f'unmixing, to \n{destination}')
                if n_skipped:
                    message += f'\n\n{n_skipped} sample(s) skipped (no AF profile assigned, or no events).'
                self.bus.popupMessage.emit(message)
            self.finished.emit()
        except Exception as e:
            logger.exception(f'{self.unmixing_method}: export failed')
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Run-time estimate
# ---------------------------------------------------------------------------

@dataclass
class UnmixTimeEstimate:
    """Estimated unmixing time for a whole export run.

    ``probe_events`` is 0 when no probe was run (the whole run is no larger
    than the probe), in which case no confirmation is needed.
    """
    n_samples: int
    total_events: int
    probe_events: int = 0
    probe_seconds: float = float('nan')
    events_per_second: float = float('nan')
    seconds: float = float('nan')
    af_profile_names: tuple = ()
    skipped_samples: tuple = ()


def format_duration(seconds) -> str:
    """'45s', '12.5min' or '1.2hr'; 'unknown' for a missing or non-finite value."""
    if seconds is None or not math.isfinite(seconds):
        return 'unknown'
    if seconds < 60:
        return f'{seconds:.0f}s'
    if seconds < 3600:
        return f'{seconds / 60:.1f}min'
    return f'{seconds / 3600:.1f}hr'


def fcs_event_count(path) -> int | None:
    """``$TOT`` from the TEXT segment only; None if the header cannot be read."""
    import flowio
    try:
        fd = flowio.FlowData(str(path), only_text=True,
                             ignore_offset_error=True, ignore_offset_discrepancy=True)
        return int(fd.text['tot'])
    except Exception as exc:
        logger.debug(f'fcs_event_count: could not read the header of {path}: {exc}')
        return None


def read_fcs_rows(path, rows, channels) -> np.ndarray | None:
    """Selected events of selected channels, read from the DATA segment only.

    For list-mode float (``$DATATYPE`` F or D) files, which spectral
    cytometers write. Values are pre-processed as FlowKit ``'raw'`` events are
    (PnE log decades, PnG gain except on Time). Channels absent from the file
    are 0. Returns None for files this cannot read (integer or ASCII data,
    mixed bit widths, unsupported byte order, unreadable header); callers fall
    back to a full read.

    rows     : event indices (0-based) into the file.
    channels : PnN names, in output column order.
    """
    import flowio
    try:
        fd = flowio.FlowData(str(path), only_text=True)
        text = fd.text
        data_type = text.get('datatype', '').lower()
        if text.get('mode', 'l').lower() != 'l' or data_type not in ('f', 'd'):
            return None
        n_par = int(text['par'])
        n_tot = int(text['tot'])
        width = 32 if data_type == 'f' else 64
        if any(int(text.get(f'p{i}b', width)) != width for i in range(1, n_par + 1)):
            return None
        byte_order = text.get('byteord', '1,2,3,4').replace(' ', '')
        if byte_order in ('1,2,3,4', '1,2,3,4,5,6,7,8'):
            order = '<'
        elif byte_order in ('4,3,2,1', '8,7,6,5,4,3,2,1'):
            order = '>'
        else:
            return None
        if fd.version == '2.0':
            data_start = int(fd.header['data_start'])
        else:
            data_start = int(text.get('begindata', 0)) or int(fd.header['data_start'])
        item = np.dtype(f'{order}f{width // 8}')
        if data_start <= 0 or data_start + n_tot * n_par * item.itemsize > Path(path).stat().st_size:
            return None

        rows = np.asarray(rows, dtype=np.int64)
        if rows.size and (rows.min() < 0 or rows.max() >= n_tot):
            raise IndexError(f'rows out of range for {n_tot} events')
        events = np.memmap(str(path), dtype=item, mode='r', offset=data_start, shape=(n_tot, n_par))
        channel_number = {meta['pnn']: num for num, meta in fd.channels.items()}

        out = np.zeros((len(rows), len(channels)), dtype=np.float64)
        for j, ch in enumerate(channels):
            num = channel_number.get(ch)
            if num is None:
                continue
            col = np.asarray(events[rows, num - 1], dtype=np.float64)
            meta = fd.channels[num]
            decades, log0 = meta['pne']
            if decades > 0:
                col = 10 ** (decades * col / meta['pnr']) * log0
            gain = meta['png']
            if gain not in (0.0, 1.0) and ch.lower() != 'time':
                col = col / gain
            out[:, j] = col
        del events
    except IndexError:
        raise
    except Exception as exc:
        logger.debug(f'read_fcs_rows: falling back to a full read for {path}: {exc}')
        return None
    np.nan_to_num(out, copy=False, nan=0.0)
    return out


def allocate_probe_rows(counts, sample_n: int) -> list:
    """Split ``min(sample_n, sum(counts))`` probe events across files in
    proportion to their event counts (largest remainder), never more than a
    file holds."""
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if total <= 0:
        return [0] * len(counts)
    n = min(int(sample_n), total)
    exact = counts * n / total
    alloc = np.floor(exact).astype(np.int64)
    remainder = n - int(alloc.sum())
    order = np.argsort(-(exact - alloc), kind='stable')
    alloc[order[:remainder]] += 1
    return np.minimum(alloc, counts).astype(int).tolist()


def most_common_af_assignment(sample_paths, layout: ExportLayout) -> tuple:
    """The AF profile assignment shared by most of the samples that can be
    AF-unmixed; ties go to the assignment met first in *sample_paths*."""
    assignments = [
        tuple(str(n) for n in (layout.sample_af_profiles.get(p, []) or []))
        for p in sample_paths if af_library_for_sample(layout, p) is not None
    ]
    if not assignments:
        return ()
    return Counter(assignments).most_common(1)[0][0]


def estimate_unmix_time(sample_paths, layout: ExportLayout, probe_fn,
                        sample_n: int = PROBE_EVENTS, seed: int = 42, bus=None) -> UnmixTimeEstimate:
    """Estimate the unmixing time of a whole run from one timed probe.

    Samples that would be skipped (no usable AF library) are left out. The
    probe takes ``sample_n`` events spread across the remaining samples in
    proportion to their event counts, so a mix of dim and bright samples is
    represented, reads only those events where the file format allows, and
    times ``probe_fn`` on them with the most frequently assigned AF profile
    set (AF assignment has little effect on speed). The per-event rate is
    scaled to the run's total event count. File reading and writing are not
    included. No probe is run when the whole run is no larger than the
    probe.
    """
    usable = []
    skipped = []
    for p in sample_paths:
        (usable if af_library_for_sample(layout, p) is not None else skipped).append(p)
    if not usable:
        return UnmixTimeEstimate(n_samples=0, total_events=0, skipped_samples=tuple(skipped))

    fl_ids_raw = list(layout.fl_ids_raw)
    fl_names = [layout.pnn_raw_export[i] for i in fl_ids_raw]

    counts = []
    preloaded = {}
    for p in usable:
        path = layout.experiment_dir / p
        n = fcs_event_count(path)
        if n is None:
            raw, _keywords, n = read_export_events(path, layout, bus)
            preloaded[p] = raw[:, fl_ids_raw]
        counts.append(int(n))
    total = int(sum(counts))

    estimate = UnmixTimeEstimate(n_samples=len(usable), total_events=total,
                                 skipped_samples=tuple(skipped))
    if total <= sample_n:
        return estimate

    rng = np.random.default_rng(seed)
    blocks = []
    for p, n, k in zip(usable, counts, allocate_probe_rows(counts, sample_n)):
        if k == 0:
            continue
        rows = np.sort(rng.choice(n, size=k, replace=False))
        if p in preloaded:
            block = preloaded[p][rows]
        else:
            path = layout.experiment_dir / p
            block = read_fcs_rows(path, rows, fl_names)
            if block is None:
                raw, _keywords, _n = read_export_events(path, layout, bus)
                block = raw[rows][:, fl_ids_raw]
        blocks.append(block)
    probe_block = np.ascontiguousarray(np.vstack(blocks), dtype=np.float64)

    names = most_common_af_assignment(usable, layout)
    af = af_library_for_profiles(layout, list(names))

    start = time.perf_counter()
    probe_fn(probe_block, af)
    elapsed = max(time.perf_counter() - start, 1e-9)

    estimate.probe_events = int(len(probe_block))
    estimate.probe_seconds = elapsed
    estimate.events_per_second = estimate.probe_events / elapsed
    estimate.seconds = total / estimate.events_per_second
    estimate.af_profile_names = tuple(af.profile_names)
    return estimate


def estimate_message(estimate: UnmixTimeEstimate) -> str:
    """Text for the confirm/cancel prompt."""
    lines = [
        f'Estimated unmixing time: about {format_duration(estimate.seconds)} for '
        f'{estimate.n_samples} sample(s), {estimate.total_events:,} events in total.',
        '',
        f'Based on a {estimate.probe_events:,}-event probe drawn from all selected samples '
        f'({estimate.events_per_second:,.0f} events/sec), unmixed with the most frequently '
        f'assigned AF profiles ({", ".join(estimate.af_profile_names)}).',
        'Reading and writing files is not included. Treat this as a rough guide: '
        'the real time can differ by a factor of 2 or more.',
    ]
    if estimate.skipped_samples:
        lines += ['', f'{len(estimate.skipped_samples)} selected sample(s) have no AF profile '
                      f'with at least 2 spectra assigned and will be skipped.']
    lines += ['', 'Start unmixing?']
    return '\n'.join(lines)


class UnmixTimeEstimateWorker(QObject):
    """Runs estimate_unmix_time() in a QThread."""
    finished = Signal(object)   # UnmixTimeEstimate
    error = Signal(str)

    def __init__(self, sample_paths, layout: ExportLayout, probe_fn,
                 sample_n: int = PROBE_EVENTS, bus=None):
        super().__init__()
        self.sample_paths = list(sample_paths)
        self.layout = layout
        self.probe_fn = probe_fn
        self.sample_n = sample_n
        self.bus = bus

    def run(self):
        try:
            self.finished.emit(estimate_unmix_time(
                self.sample_paths, self.layout, self.probe_fn,
                sample_n=self.sample_n, bus=self.bus,
            ))
        except Exception as e:
            logger.exception('Unmixing time estimate failed')
            self.error.emit(str(e))


# ---------------------------------------------------------------------------
# Estimate -> confirm -> export
# ---------------------------------------------------------------------------

class JointUnmixRun(QObject):
    """Estimate the run time, ask the user to confirm, then export.

    Create and start on the main thread. Without ``probe_fn`` the export
    starts straight away. ``done`` is emitted once at the end whatever the
    outcome (finished, error or cancelled).
    """
    status = Signal(str)
    progress = Signal(int, int)
    finished = Signal()
    error = Signal(str)
    cancelled = Signal()
    done = Signal()

    def __init__(self, exporter: JointUnmixExporter, parent_widget=None, probe_fn=None,
                 sample_n: int = PROBE_EVENTS, bus=None):
        super().__init__()
        self._exporter = exporter
        self._parent_widget = parent_widget
        self._probe_fn = probe_fn
        self._sample_n = sample_n
        self._bus = bus
        self._est_thread = None
        self._est_worker = None
        self._exp_thread = None

    def start(self):
        if self._probe_fn is None:
            self._start_export()
            return
        self.status.emit('Estimating unmixing time...')
        self._est_thread = QThread()
        self._est_worker = UnmixTimeEstimateWorker(
            self._exporter.sample_paths, self._exporter.layout, self._probe_fn,
            sample_n=self._sample_n, bus=self._bus,
        )
        self._est_worker.moveToThread(self._est_thread)
        self._est_thread.started.connect(self._est_worker.run)
        self._est_worker.finished.connect(self._on_estimate)
        self._est_worker.error.connect(self._on_estimate_error)
        self._est_worker.finished.connect(self._est_thread.quit)
        self._est_worker.error.connect(self._est_thread.quit)
        self._est_thread.finished.connect(self._est_thread.deleteLater)
        self._est_thread.finished.connect(self._on_est_thread_finished)
        self._est_thread.start()

    def _on_est_thread_finished(self):
        self._est_thread = None
        self._est_worker = None

    def _confirm(self, text: str) -> bool:
        reply = QMessageBox.question(
            self._parent_widget, 'Unmixing time', text,
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Yes,
        )
        return reply == QMessageBox.Yes

    def _cancel(self):
        self.status.emit('Unmixing cancelled.')
        self.cancelled.emit()
        self.done.emit()

    def _on_estimate(self, estimate: UnmixTimeEstimate):
        if estimate.n_samples == 0:
            self.error.emit('None of the selected samples has an AF profile with at least '
                            '2 spectra assigned.')
            self.done.emit()
            return
        if estimate.probe_events == 0:
            self._start_export()
            return
        self.status.emit(f'Estimated unmixing time: about {format_duration(estimate.seconds)}')
        if self._confirm(estimate_message(estimate)):
            self._start_export()
        else:
            self._cancel()

    def _on_estimate_error(self, msg: str):
        if self._confirm(f'The unmixing time could not be estimated:\n{msg}\n\nStart unmixing anyway?'):
            self._start_export()
        else:
            self._cancel()

    def _start_export(self):
        self._exp_thread = QThread()
        self._exporter.moveToThread(self._exp_thread)
        self._exp_thread.started.connect(self._exporter.run)
        self._exporter.progress.connect(self.progress)
        self._exporter.finished.connect(self.finished)
        self._exporter.error.connect(self.error)
        self._exporter.finished.connect(self._exp_thread.quit)
        self._exporter.error.connect(self._exp_thread.quit)
        self._exp_thread.finished.connect(self._exp_thread.deleteLater)
        self._exp_thread.finished.connect(self._on_export_thread_finished)
        self._exp_thread.start()

    def _on_export_thread_finished(self):
        self._exp_thread = None
        self.done.emit()
