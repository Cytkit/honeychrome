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
"""

import numpy as np

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
