"""
particle_type.py
----------------
Cells or beads from forward and side scatter, for single-stain controls whose
name does not say which they are.

Beads are monodisperse, so their main population is far tighter on log scatter
than any cell population. The measure is the area of the tightest ellipse that
holds a quarter of the events in log10(FSC) x log10(SSC), found with the
concentration steps of FAST-MCD (Rousseeuw & Van Driessen 1999). Log scatter
makes the area independent of detector gain, so controls run at different
voltages or on different days stay comparable.

Thresholds were set on 305 controls named cells or beads (117 bead, 188 cell)
from 18 experiments on 10 cytometers (Aurora, Northern Lights, FACSDiscover A8,
ID7000, CytoFLEX LX, CytoFLEX Mosaic, Attune NxT, Attune Xenith, Bigfoot,
NovoCyte Opteon): log10 core area of bead controls -4.74 to -3.05, of cell
controls -2.78 to -0.95.

A control is called beads when its core area is below BEAD_CORE_LOG_AREA and,
when the experiment has an unstained cell control, also more than
BEAD_ANCHOR_MARGIN decades below that control's core area.

Public API
----------
scatter_core_log_area(scatter, ceilings)
    log10 area of the core ellipse of an (n_events, 2) FSC/SSC array, or None.

classify_particle_type(log_area, anchor_log_area=None)
    'Beads', 'Cells', or None when log_area is None.

scatter_channel_pair(raw_settings, raw_plots=None)
    The (FSC, SSC) channel names used for the measure.

scatter_ceilings(raw_settings, channels)
    Saturation ceiling of each scatter channel, as the raw transforms use.

Bead samples
------------
samples['bead_samples'] lists the samples treated as beads everywhere a
sample (rather than a spectral control) is judged bead or cell, e.g. which
unstained sample is a bead or a cell negative. update_bead_samples() checks
each sample once, listed in samples['bead_checked_samples']: a name
containing "bead" marks it; an unstained sample is otherwise scored on
scatter. After that check only the user's Mark/Unmark as Beads changes it.
In the scatter check the anchor is the broadest unstained sample checked in
the same call (v1: earlier calls' scores are not kept).

is_bead_name(path, name)
    Whether the path or name contains "bead".

is_bead_sample(samples, path, name=None)
    Whether a sample is a bead sample; by name in experiments not yet scanned.

sample_scatter_log_area(abs_path, pair, ceilings)
    Core area of a sample file, read from at most CORE_MAX_EVENTS events.

update_bead_samples(samples, experiment_dir, raw_settings, raw_plots=None)
    Check samples not yet checked; returns the paths newly marked as beads.
"""

import logging
import re
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

CORE_FRACTION = 0.25            # fraction of events in the core ellipse
CORE_MAX_EVENTS = 20000         # events used per sample
CORE_MIN_EVENTS = 200           # fewer usable events: no call
CORE_STEPS = 4                  # concentration steps
BEAD_CORE_LOG_AREA = -2.9       # log10 core area; beads below
BEAD_ANCHOR_MARGIN = 0.75       # decades below the unstained cell control


def scatter_core_log_area(scatter: np.ndarray, ceilings,
                          fraction: float = CORE_FRACTION,
                          max_events: int = CORE_MAX_EVENTS,
                          seed: int = 0) -> float | None:
    """log10 area of the tightest ellipse holding `fraction` of the events
    in log10 scatter. Non-positive and saturated events are dropped;
    `ceilings` is one value per column or a single value."""
    scatter = np.asarray(scatter, dtype=np.float64)
    if scatter.ndim != 2 or scatter.shape[1] != 2:
        return None
    ceilings = np.broadcast_to(np.asarray(ceilings, dtype=np.float64), (2,))
    keep = (np.all(np.isfinite(scatter), axis=1)
            & np.all(scatter > 0, axis=1)
            & np.all(scatter < 0.999 * ceilings, axis=1))
    xy = np.log10(scatter[keep])
    if len(xy) < CORE_MIN_EVENTS:
        return None
    if len(xy) > max_events:
        rng = np.random.default_rng(seed)
        xy = xy[rng.choice(len(xy), max_events, replace=False)]

    h = max(3, int(len(xy) * fraction))
    centre = np.median(xy, axis=0)
    cov = np.cov(xy.T)
    for _ in range(CORE_STEPS):
        d = xy - centre
        md = np.einsum('ij,jk,ik->i', d, np.linalg.pinv(cov), d)
        core = xy[np.argpartition(md, h - 1)[:h]]
        centre = core.mean(axis=0)
        cov = np.cov(core.T)

    det = np.linalg.det(cov)
    if not np.isfinite(det) or det <= 0:
        return None
    return float(np.log10(np.pi * np.sqrt(det)))


def classify_particle_type(log_area: float | None,
                           anchor_log_area: float | None = None) -> str | None:
    """'Beads' or 'Cells' from a core area; anchor_log_area is the unstained
    cell control's core area, ignored when it is itself bead-like."""
    if log_area is None:
        return None
    tight = log_area < BEAD_CORE_LOG_AREA
    if anchor_log_area is not None and anchor_log_area >= BEAD_CORE_LOG_AREA:
        tight = tight and log_area < anchor_log_area - BEAD_ANCHOR_MARGIN
    return 'Beads' if tight else 'Cells'


def scatter_channel_pair(raw_settings: dict, raw_plots: list | None = None) -> tuple[str, str] | None:
    """FSC and SSC channel names: scatter_param, else the root morphology plot."""
    pnn = raw_settings.get('event_channels_pnn', [])
    pair = raw_settings.get('scatter_param') or []
    if len(pair) >= 2 and pair[0] in pnn and pair[1] in pnn:
        return pair[0], pair[1]
    for plot in raw_plots or []:
        if (plot.get('type') == 'hist2d' and plot.get('source_gate') == 'root'
                and plot.get('channel_x') in pnn and plot.get('channel_y') in pnn):
            return plot['channel_x'], plot['channel_y']
    return None


def scatter_ceilings(raw_settings: dict, channels) -> list[float]:
    """Per-channel $PnR when known, else the magnitude ceiling."""
    pnn = raw_settings.get('event_channels_pnn', [])
    pnr = raw_settings.get('channel_pnr') or []
    fallback = float(raw_settings.get('magnitude_ceiling') or np.inf)
    out = []
    for ch in channels:
        i = pnn.index(ch) if ch in pnn else -1
        out.append(float(pnr[i]) if 0 <= i < len(pnr) and pnr[i] else fallback)
    return out


def is_bead_name(path: str, name: str | None) -> bool:
    return bool(re.search(r'bead', str(path), re.IGNORECASE)
                or re.search(r'bead', str(name or ''), re.IGNORECASE))


def is_bead_sample(samples: dict, path: str, name: str | None = None) -> bool:
    beads = samples.get('bead_samples')
    if beads is not None:
        return path in beads
    if name is None:
        name = samples.get('all_samples', {}).get(path, '')
    return is_bead_name(path, name)


def sample_scatter_log_area(abs_path, pair, ceilings) -> float | None:
    from honeychrome.controller_components.fcs_preview import load_fcs_preview
    try:
        _, events = load_fcs_preview(abs_path, CORE_MAX_EVENTS, col_order=list(pair))
    except ValueError:
        # integer FCS data: full read
        from honeychrome.controller_components.functions import sample_from_fcs
        events = sample_from_fcs(str(abs_path)).get_events('raw', col_order=list(pair))
    return scatter_core_log_area(events, ceilings)


def update_bead_samples(samples: dict, experiment_dir, raw_settings: dict,
                        raw_plots: list | None = None) -> list:
    all_samples = samples.get('all_samples', {})
    unstained = set(samples.get('unstained_samples', []))
    beads = [p for p in samples.get('bead_samples', []) if p in all_samples]
    checked = [p for p in samples.get('bead_checked_samples', []) if p in all_samples]
    samples['bead_samples'] = beads
    samples['bead_checked_samples'] = checked

    newly = []
    to_score = []
    for path, name in all_samples.items():
        if path in checked:
            continue
        if is_bead_name(path, name):
            checked.append(path)
            if path not in beads:
                beads.append(path)
                newly.append(path)
        elif path in unstained:
            to_score.append(path)

    pair = scatter_channel_pair(raw_settings, raw_plots) if to_score else None
    if pair is None:
        return newly
    ceilings = scatter_ceilings(raw_settings, pair)
    scores = {}
    for path in to_score:
        try:
            score = sample_scatter_log_area(Path(experiment_dir) / path, pair, ceilings)
        except Exception as e:
            logger.warning(f'update_bead_samples: scatter of {path} not read: {e}')
            continue
        if score is not None:
            scores[path] = score

    anchor = max(scores.values()) if scores else None
    for path, score in scores.items():
        checked.append(path)
        particle_type = classify_particle_type(score, anchor)
        logger.info(f'update_bead_samples: "{path}" {particle_type} from scatter '
                    f'(core log10 area {score:.2f}, anchor {anchor:.2f})')
        if particle_type == 'Beads' and path not in beads:
            beads.append(path)
            newly.append(path)

    rank = {p: i for i, p in enumerate(all_samples)}
    beads.sort(key=rank.__getitem__)
    return newly
