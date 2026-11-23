"""
landmark_gates.py
-----------------
Default Cells and Singlets gates placed from the single-stain controls.

The fixed default gates (the middle 60% of each scatter axis) often miss the
cells: on an Aurora, for example, lymphocytes sit at about 15% of the SSC
range, below the default lower edge. Here the gates are placed where the
stained cells are instead, following define.gate.landmarks() in AutoSpectral
(which credits N. Laniewski's flowstate): the brightest events of each
single-stain control in its own peak channel are "landmarks", stained cells
that the scatter gates must keep.

Steps
-----
1. Controls used: every single-stain control except unstained samples, bead
   controls (by name, the bead list, or a tight scatter core as judged by
   particle_type), viability dyes (``is.viability`` in the fluorophore
   database), and controls whose stain is not where the database expects it:
   when a control's name matches a database fluorophore and its empirical peak
   channel is on a different laser from the database's peak channel, the
   bright events are not the stain (an empty, failed or mislabelled control)
   and are left out.
2. Events: LANDMARK_BLOCKS evenly spaced runs of LANDMARK_BLOCK_ROWS events
   from each file (contiguous reads keep this fast on large files).
3. Peak channel: find_empirical_peak() against the unstained cell control's
   mean, as Auto generate does; without one, against the dimmest quarter of
   the control itself.
4. Landmarks: the top LANDMARK_FRACTION of events (at most
   LANDMARK_MAX_EVENTS) in the peak channel, unsaturated on the peak and
   scatter channels.
5. Cells: the landmarks of each control are binned on the displayed FSC/SSC
   plot, each control weighted equally, smoothed, and the densest region
   holding DENSITY_SHARE of the pooled landmarks is found (the 70% density
   contour of define.gate.landmarks). The Cells rectangle is that region's
   bounding box, with its lower FSC and SSC edges taken down, where needed, to
   LOWER_EDGE_HEADROOM x the LOWER_EDGE_QUANTILE of the pooled landmarks:
   brightly stained cells sit slightly higher on scatter than the population
   they belong to (on UCL IMMPROVE PBMC the density box alone cut the lower
   half of the lymphocytes), while the upper edges must stay clear of the
   sparse high-SSC events that bright staining also brings.
6. Singlets: the landmarks inside the Cells rectangle give the singlet ratio
   (FSC height over area, on the displayed scale). The band runs from
   median - SINGLET_MAD_FACTOR x MAD to median + SINGLET_MAD_FACTOR x MAD,
   and at least SINGLET_MIN_HALF_WIDTH x median either side; doublets have a
   lower ratio and fall below it. For a width channel the band is a range of
   widths instead.

With fewer than MIN_CONTROLS controls giving landmarks (e.g. bead controls
with an unstained cell sample), Cells and Singlets are placed the same way on
the unstained cell sample's own events, holding UNSTAINED_SHARE of them. With
neither, the caller keeps the fixed defaults.

Bead samples
------------
bead_scatter_gates() places one Cells box and one Singlets gate for all bead
samples together, which the caller installs as the same per-sample gates on
each bead sample (so the hierarchy is shared with the cells). The bead
samples' events are pooled, each sample weighted equally; the box is the
densest DENSITY_SHARE of them, widened by BEAD_BOX_PADDING of the box size on
every side (a bead population is a few bins wide on the displayed scale), and
the singlet band comes from the events inside it. Bead samples with less than
BEAD_MIN_INSIDE of their events in the box (another bead type, or a poor
sample) are listed so the user can give them their own gate.

The rectangle is a deliberate simplification of the landmark contour, so the
default gates keep the shapes users already know and edit.

Public API
----------
landmark_scatter_gates(experiment_dir, samples, raw_settings, transformations,
                       morph_x, morph_y, sing_x=None, sing_y=None)
    LandmarkGates or None.

bead_scatter_gates(experiment_dir, bead_paths, raw_settings, transformations,
                   morph_x, morph_y, sing_x=None, sing_y=None)
    BeadGates (one Cells box and Singlets for every bead sample) or None.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from honeychrome.controller_components.fcs_preview import read_fcs_rows, read_preview_metadata
from honeychrome.controller_components.particle_type import (
    classify_particle_type, is_bead_name, scatter_ceilings, scatter_core)
from honeychrome.controller_components.spectral_cleaning import find_empirical_peak

logger = logging.getLogger(__name__)

LANDMARK_BLOCKS = 20            # runs of contiguous events read per file
LANDMARK_BLOCK_ROWS = 1000      # events per run
LANDMARK_FRACTION = 0.025       # brightest share of events kept as landmarks
LANDMARK_MAX_EVENTS = 500       # landmarks per control, at most
LANDMARK_MIN_EVENTS = 50        # landmarks per control, at least
MIN_CONTROLS = 3                # controls with landmarks needed to place gates
DENSITY_SHARE = 0.70            # share of pooled landmarks in the densest region
DENSITY_GRID = 128              # bins per axis of the displayed plot
DENSITY_SMOOTH_BINS = 2.0       # Gaussian smoothing, standard deviation in bins
LOWER_EDGE_QUANTILE = 0.01      # pooled landmark quantile setting the lowest Cells edges
LOWER_EDGE_HEADROOM = 0.8       # lower edges at most this fraction of that quantile
UNSTAINED_SHARE = 0.90          # share of unstained cell events in Cells (fallback)
BEAD_BOX_PADDING = 0.5          # bead Cells box widened by this fraction of its size per side
BEAD_MIN_INSIDE = 0.5           # bead samples with less of their events in the box are listed
SINGLET_MAD_FACTOR = 2.5        # half-width of the singlet band, in MADs
SINGLET_MIN_HALF_WIDTH = 0.1    # half-width of the singlet band, at least this fraction of its median
SATURATION = 0.999              # share of a channel's ceiling counted as saturated


@dataclass
class LandmarkGates:
    """Gate geometry on the displayed (transformed) scale."""
    cells: tuple                         # (x_min, x_max, y_min, y_max)
    singlets: list | tuple | None        # polygon vertices (height) or (y_min, y_max) (width)
    source: str = 'controls'             # 'controls' or 'unstained'
    used: list = field(default_factory=list)
    excluded: dict = field(default_factory=dict)   # sample path -> reason
    landmarks: list = field(default_factory=list)  # per control, displayed (x, y, singlet y)

    def summary(self) -> str:
        if self.source == 'unstained':
            text = (f'Cells and Singlets gates placed on the unstained cell sample '
                    f'{Path(self.used[0]).stem} (too few single-stain cell controls).')
        else:
            text = (f'Cells and Singlets gates placed from the brightest events of '
                    f'{len(self.used)} single-stain controls.')
        if self.excluded:
            names = ', '.join(f'{Path(p).stem} ({why})' for p, why in self.excluded.items())
            text += f'\nControls not used: {names}.'
        return text


@dataclass
class BeadGates:
    """One Cells box and Singlets for all bead samples, on the displayed scale."""
    cells: tuple                         # (x_min, x_max, y_min, y_max)
    singlets: list | tuple | None        # polygon vertices (height) or (y_min, y_max) (width)
    samples: list = field(default_factory=list)      # bead samples read
    outside: dict = field(default_factory=dict)      # sample path -> share of its events in the box

    def summary(self) -> str:
        text = (f'Bead samples ({len(self.samples)}) share one Cells and Singlets gate placed on their '
                f'beads, as per-sample gates.')
        if self.outside:
            names = ', '.join(f'{Path(p).stem} ({share:.0%} inside)' for p, share in self.outside.items())
            text += f'\nBead samples mostly outside that gate, to check: {names}.'
        return text


def landmark_rows(n_events: int) -> np.ndarray:
    """Row positions of LANDMARK_BLOCKS evenly spaced runs of events."""
    if n_events <= LANDMARK_BLOCKS * LANDMARK_BLOCK_ROWS:
        return np.arange(n_events)
    starts = np.linspace(0, n_events - LANDMARK_BLOCK_ROWS, LANDMARK_BLOCKS).astype(np.int64)
    return np.unique(np.concatenate([np.arange(s, s + LANDMARK_BLOCK_ROWS) for s in starts]))


def read_landmark_events(path, channels) -> np.ndarray:
    """Raw events of *channels* at landmark_rows(); integer FCS files are read whole."""
    meta = read_preview_metadata(path)
    rows = landmark_rows(meta.event_count)
    try:
        return read_fcs_rows(path, rows, channels, meta)
    except ValueError:
        from honeychrome.controller_components.functions import sample_from_fcs
        return sample_from_fcs(str(path)).get_events('raw', col_order=list(channels))[rows]


def _smooth(grid: np.ndarray, sigma: float) -> np.ndarray:
    radius = int(np.ceil(3 * sigma))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    grid = np.apply_along_axis(np.convolve, 0, grid, kernel, mode='same')
    return np.apply_along_axis(np.convolve, 1, grid, kernel, mode='same')


def densest_box(points: list, x_limits, y_limits, share: float = DENSITY_SHARE) -> tuple | None:
    """Bounding box (x_min, x_max, y_min, y_max) of the densest region holding
    *share* of the pooled points, each array in *points* weighted equally."""
    grid = np.zeros((DENSITY_GRID, DENSITY_GRID))
    for xy in points:
        if len(xy):
            counts, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=DENSITY_GRID,
                                          range=[x_limits, y_limits])
            grid += counts / len(xy)
    if grid.sum() <= 0:
        return None
    density = _smooth(grid, DENSITY_SMOOTH_BINS)
    ordered = np.sort(density.ravel())[::-1]
    cumulative = np.cumsum(ordered) / ordered.sum()
    threshold = ordered[min(np.searchsorted(cumulative, share), len(ordered) - 1)]
    ix, iy = np.nonzero(density >= threshold)
    x_edges = np.linspace(*x_limits, DENSITY_GRID + 1)
    y_edges = np.linspace(*y_limits, DENSITY_GRID + 1)
    return (float(x_edges[ix.min()]), float(x_edges[ix.max() + 1]),
            float(y_edges[iy.min()]), float(y_edges[iy.max() + 1]))


def robust_band(values: np.ndarray, factor: float = SINGLET_MAD_FACTOR,
                min_half_width: float = SINGLET_MIN_HALF_WIDTH) -> tuple | None:
    """median -/+ the larger of factor x MAD (scaled to a standard deviation)
    and min_half_width x median."""
    values = values[np.isfinite(values)]
    if len(values) < LANDMARK_MIN_EVENTS:
        return None
    median = float(np.median(values))
    mad = 1.4826 * float(np.median(np.abs(values - median)))
    half = max(factor * mad, min_half_width * abs(median))
    return median - half, median + half


def singlet_geometry(points: np.ndarray, cells: tuple, sing_y: str | None):
    """Singlets on the morphology FSC channel: polygon vertices for a height
    channel, (y_min, y_max) for a width channel, from the displayed (x, y,
    singlet y) points inside the Cells box; None when it cannot be placed."""
    if not sing_y or points is None or len(points) == 0:
        return None
    inside = points[(points[:, 0] >= cells[0]) & (points[:, 0] <= cells[1])
                    & (points[:, 1] >= cells[2]) & (points[:, 1] <= cells[3])]
    if sing_y.endswith('-W'):
        return robust_band(inside[:, 2])
    band = robust_band(inside[:, 2] / np.maximum(inside[:, 0], 1e-12))
    if band is None:
        return None
    low, high = band
    x_lo, x_hi = cells[0], cells[1]
    return [(x_lo, low * x_lo), (x_hi, low * x_hi), (x_hi, high * x_hi), (x_lo, high * x_lo)]


def _shown_unsaturated(events: np.ndarray, channels: list, transformations: dict,
                       ceilings: np.ndarray) -> np.ndarray:
    """Events on the displayed scale, dropping non-positive and saturated ones."""
    keep = np.all(events > 0, axis=1) & np.all(events < SATURATION * ceilings, axis=1)
    return np.column_stack([transformations[ch].xform.apply(events[keep, i])
                            for i, ch in enumerate(channels)])


def _database_peak(name: str, db_col: str | None) -> tuple:
    """(database peak channel, is viability) of the fluorophore in a sample name."""
    from honeychrome.controller_components.label_matching import get_fluorophore_db, match_fluorophore
    database = get_fluorophore_db()
    canonical = match_fluorophore(name, database)
    if not canonical:
        return None, False
    row = next((r for r in database if r.get('fluorophore') == canonical), {})
    channel = (row.get(f'channel.{db_col}') or '').strip() if db_col else ''
    return channel or None, str(row.get('is.viability', '')).strip().upper() == 'TRUE'


def landmark_scatter_gates(experiment_dir, samples: dict, raw_settings: dict, transformations: dict,
                           morph_x: str, morph_y: str,
                           sing_x: str | None = None, sing_y: str | None = None) -> LandmarkGates | None:
    """Cells (and Singlets) gate geometry from the single-stain controls, or None."""
    experiment_dir = Path(experiment_dir)
    pnn = raw_settings['event_channels_pnn']
    whitelisted = set(raw_settings.get('whitelisted_pnn') or pnn)
    fluor = [pnn[i] for i in raw_settings['fluorescence_channel_ids'] if pnn[i] in whitelisted]
    scatter = [morph_x, morph_y] + ([sing_y] if sing_y else [])
    if not fluor or any(ch not in transformations for ch in scatter):
        return None
    channels = scatter + fluor
    n_sc = len(scatter)
    ceilings = np.asarray(scatter_ceilings(raw_settings, scatter))
    fluor_ceiling = float(raw_settings.get('magnitude_ceiling') or np.inf)
    names = samples.get('all_samples', {})
    unstained = set(samples.get('unstained_samples', []))
    bead_list = set(samples.get('bead_samples', []))
    db_col = raw_settings.get('cytometer_db_col')

    def core_area(events):
        core = scatter_core(events[:, :2], ceilings[:2])
        return None if core is None else core[0]

    # unstained cell control: the AF reference for the peak channel, and the
    # anchor for judging beads; ones in the controls folder first
    controls = list(samples.get('single_stain_controls', []))
    unstained_events, unstained_path, anchor = None, None, None
    for path in sorted(unstained, key=lambda p: p not in controls):
        if path in bead_list or is_bead_name(path, names.get(path)):
            continue
        try:
            events = read_landmark_events(experiment_dir / path, channels)
        except Exception as e:
            logger.warning(f'landmark_scatter_gates: {path} not read: {e}')
            continue
        area = core_area(events)
        if classify_particle_type(area) == 'Beads':
            continue
        unstained_events, unstained_path, anchor = events, path, area
        break
    af_mean = unstained_events[:, n_sc:].mean(axis=0) if unstained_events is not None else None

    lasers = None
    landmarks, used, excluded = [], [], {}
    for path in controls:
        name = names.get(path, Path(path).stem)
        if path in unstained:
            continue
        if path in bead_list or is_bead_name(path, name):
            excluded[path] = 'beads'
            continue
        db_peak, viability = _database_peak(name, db_col)
        if viability:
            excluded[path] = 'viability dye'
            continue
        try:
            events = read_landmark_events(experiment_dir / path, channels)
        except Exception as e:
            logger.warning(f'landmark_scatter_gates: {path} not read: {e}')
            excluded[path] = 'not read'
            continue
        if classify_particle_type(core_area(events), anchor) == 'Beads':
            excluded[path] = 'beads'
            continue

        spectra = events[:, n_sc:]
        if af_mean is not None:
            reference = af_mean
        else:
            order = np.argsort(spectra.max(axis=1))
            reference = spectra[order[:max(2, len(spectra) // 4)]].mean(axis=0)
        peak = find_empirical_peak(spectra, reference)

        if db_peak in fluor:
            if lasers is None:
                from honeychrome.controller_components.gls_unmix import detector_lasers
                lasers = dict(zip(fluor, detector_lasers(raw_settings, fluor, str(experiment_dir / path))))
            expected, found = lasers.get(db_peak), lasers.get(fluor[peak])
            if expected and found and expected != found:
                excluded[path] = f'peak in {fluor[peak]}, expected near {db_peak}'
                continue

        values = spectra[:, peak]
        keep = ((values < SATURATION * fluor_ceiling)
                & np.all(events[:, :n_sc] > 0, axis=1)
                & np.all(events[:, :n_sc] < SATURATION * ceilings, axis=1))
        rows = np.flatnonzero(keep)
        n_top = int(np.clip(LANDMARK_FRACTION * len(events), LANDMARK_MIN_EVENTS, LANDMARK_MAX_EVENTS))
        if len(rows) < n_top:
            excluded[path] = 'too few events'
            continue
        top = rows[np.argsort(values[rows])[-n_top:]]
        shown = np.column_stack([transformations[ch].xform.apply(events[top, i])
                                 for i, ch in enumerate(scatter)])
        landmarks.append(shown)
        used.append(path)

    x_limits = tuple(transformations[morph_x].limits)
    y_limits = tuple(transformations[morph_y].limits)
    singlet_y = sing_y if sing_x == morph_x else None

    if len(landmarks) < MIN_CONTROLS:
        if unstained_events is None:
            logger.info(f'landmark_scatter_gates: {len(landmarks)} controls with landmarks and no '
                        f'unstained cell sample; fixed default gates kept')
            return None
        shown = _shown_unsaturated(unstained_events[:, :n_sc], scatter, transformations, ceilings)
        cells = densest_box([shown[:, :2]], x_limits, y_limits, UNSTAINED_SHARE)
        if cells is None:
            return None
        result = LandmarkGates(cells=cells, singlets=singlet_geometry(shown, cells, singlet_y),
                               source='unstained', used=[unstained_path], excluded=excluded)
        logger.info(f'landmark_scatter_gates: {len(landmarks)} controls with landmarks; Cells '
                    f'{np.round(cells, 3).tolist()} from unstained {unstained_path}')
        return result

    cells = densest_box(landmarks, x_limits, y_limits)
    if cells is None:
        return None
    pooled = np.vstack(landmarks)
    lowest = LOWER_EDGE_HEADROOM * np.quantile(pooled[:, :2], LOWER_EDGE_QUANTILE, axis=0)
    cells = (min(cells[0], max(float(lowest[0]), x_limits[0])), cells[1],
             min(cells[2], max(float(lowest[1]), y_limits[0])), cells[3])

    result = LandmarkGates(cells=cells, singlets=singlet_geometry(pooled, cells, singlet_y),
                           used=used, excluded=excluded, landmarks=landmarks)
    logger.info(f'landmark_scatter_gates: Cells {np.round(cells, 3).tolist()}, '
                f'Singlets {result.singlets}; {len(used)} controls used, excluded {excluded}')
    return result


def bead_scatter_gates(experiment_dir, bead_paths, raw_settings: dict, transformations: dict,
                       morph_x: str, morph_y: str,
                       sing_x: str | None = None, sing_y: str | None = None) -> BeadGates | None:
    """One Cells box and Singlets placed on the pooled events of all bead samples, or None."""
    experiment_dir = Path(experiment_dir)
    scatter = [morph_x, morph_y] + ([sing_y] if sing_y else [])
    if not bead_paths or any(ch not in transformations for ch in scatter):
        return None
    ceilings = np.asarray(scatter_ceilings(raw_settings, scatter))
    x_limits = tuple(transformations[morph_x].limits)
    y_limits = tuple(transformations[morph_y].limits)
    singlet_y = sing_y if sing_x == morph_x else None
    shown = {}
    for path in bead_paths:
        try:
            events = read_landmark_events(experiment_dir / path, scatter)
        except Exception as e:
            logger.warning(f'bead_scatter_gates: {path} not read: {e}')
            continue
        points = _shown_unsaturated(events, scatter, transformations, ceilings)
        if len(points) >= LANDMARK_MIN_EVENTS:
            shown[path] = points
    if not shown:
        return None
    box = densest_box([p[:, :2] for p in shown.values()], x_limits, y_limits)
    if box is None:
        return None
    pad_x = BEAD_BOX_PADDING * (box[1] - box[0])
    pad_y = BEAD_BOX_PADDING * (box[3] - box[2])
    cells = (max(box[0] - pad_x, x_limits[0]), min(box[1] + pad_x, x_limits[1]),
             max(box[2] - pad_y, y_limits[0]), min(box[3] + pad_y, y_limits[1]))

    def share_inside(points):
        return float(np.mean((points[:, 0] >= cells[0]) & (points[:, 0] <= cells[1])
                             & (points[:, 1] >= cells[2]) & (points[:, 1] <= cells[3])))

    outside = {path: share_inside(p) for path, p in shown.items() if share_inside(p) < BEAD_MIN_INSIDE}
    result = BeadGates(cells=cells, singlets=singlet_geometry(np.vstack(list(shown.values())), cells, singlet_y),
                       samples=list(shown), outside=outside)
    logger.info(f'bead_scatter_gates: Cells {np.round(cells, 3).tolist()} from {len(shown)} bead samples; '
                f'mostly outside: {outside}')
    return result
