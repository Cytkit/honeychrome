"""
stain_detection.py
------------------
Is a single-stain cell control stained at all, compared with an unstained cell sample?

A control tube that was never stained (an unstained sample that is not named or marked
as unstained, so auto-generate takes it for a control) adds an autofluorescence row to
the spectral model and degrades the unmixing. Its cells match the unstained sample's in
every detector. A stained control has an excess of bright cells in the detectors its dye
reaches.

Excess z
--------
In one detector, let t be the unstained sample's 99.5th percentile: 0.5% of unstained
cells are above t by definition. Of the n control cells, k are above t. The excess
k - 0.005 n counts the stained cells the unstained cannot account for, and

    z = (k - 0.005 n) / sqrt(0.005 x 0.995 x n)

is how many standard deviations that excess sits above what an unstained sample would
give by chance. z grows with both the share of stained cells and the number of cells.

Rectangle search
----------------
A dim stain on a minority of cells can be hidden when the Cells gate also holds bright
autofluorescent cells (high-SSC myeloid cells, for example): they set t so high that the
stained cells do not pass it. The search therefore also scores Cells rectangles on the
FSC/SSC plot, each with the unstained cut to the same rectangle. The rectangle edges are
at twelfths of the control's FSC and SSC distributions (all 78 x 78 pairs of edges;
quantile edges make the search the same on any monotone scatter scale). Rectangles with
fewer than MIN_BOX_CONTROL control or MIN_BOX_NEGATIVE unstained events are not scored.
Thresholds come from histograms of the detector with HIST_BINS quantile bins.

Taking the largest z over thousands of rectangles and every detector gives some z by
chance alone: two unstained cell samples of the same mouse spleen compared this way
reached 14 and 26, a control whose stain was hidden by bright myeloid cells 349, and the
dimmest stained controls of a 40-colour panel 199 to 214 (FlowCode MSA02).
UNSTAINED_Z_MAX sits between them.
"""
from __future__ import annotations

import numpy as np

NULL_SHARE = 0.005              # share of unstained cells above the threshold, by definition
EDGE_QUANTILES = np.linspace(0.0, 1.0, 13)   # rectangle edges, per scatter axis
MIN_BOX_CONTROL = 5000          # control events a rectangle must hold to be scored
MIN_BOX_NEGATIVE = 2000         # unstained events a rectangle must hold to be scored
HIST_BINS = 2000                # quantile bins of the detector for rectangle thresholds
UNSTAINED_Z_MAX = 50.0          # a largest excess z below this: no stain found


def excess_z(control: np.ndarray, negative: np.ndarray) -> np.ndarray:
    """Per detector (column), the excess z of *control* events above the 99.5th percentile of *negative*."""
    t = np.quantile(negative, 1.0 - NULL_SHARE, axis=0)
    n = len(control)
    k = (control > t).sum(axis=0)
    return (k - NULL_SHARE * n) / np.sqrt(NULL_SHARE * (1.0 - NULL_SHARE) * n)


def rectangle_scores(control_xy: np.ndarray, control_values: np.ndarray,
                     negative_xy: np.ndarray, negative_values: np.ndarray) -> list[dict]:
    """
    Excess z of every scored rectangle in one detector.

    control_xy, negative_xy        : (n, 2) FSC and SSC of the control and unstained events.
    control_values, negative_values: (n,) the detector's values for the same events.

    Returns one dict per rectangle: edges x0, x1, y0, y1 (scatter units of the input),
    n_control, n_negative, threshold, excess, fraction (excess / n_control) and z.
    """
    if len(control_xy) < MIN_BOX_CONTROL or len(negative_xy) < MIN_BOX_NEGATIVE:
        return []
    xe = np.quantile(control_xy[:, 0], EDGE_QUANTILES)
    ye = np.quantile(control_xy[:, 1], EDGE_QUANTILES)
    xe[0], ye[0] = min(xe[0], negative_xy[:, 0].min()), min(ye[0], negative_xy[:, 1].min())
    xe[-1] = max(xe[-1], negative_xy[:, 0].max()) + 1e-9
    ye[-1] = max(ye[-1], negative_xy[:, 1].max()) + 1e-9
    ve = np.unique(np.quantile(np.concatenate([control_values, negative_values]),
                               np.linspace(0.0, 1.0, HIST_BINS + 1)))
    ve[-1] += 1e-9
    g = len(EDGE_QUANTILES) - 1
    nv = len(ve) - 1

    def cumulative(xy, v):
        ix = np.clip(np.searchsorted(xe, xy[:, 0], side='right') - 1, 0, g - 1)
        iy = np.clip(np.searchsorted(ye, xy[:, 1], side='right') - 1, 0, g - 1)
        iv = np.clip(np.searchsorted(ve, v, side='right') - 1, 0, nv - 1)
        h = np.zeros((g, g, nv))
        np.add.at(h, (ix, iy, iv), 1.0)
        c = np.zeros((g + 1, g + 1, nv))
        c[1:, 1:] = h.cumsum(0).cumsum(1)
        return c

    cc, cn = cumulative(control_xy, control_values), cumulative(negative_xy, negative_values)
    rows = []
    for a in range(g):
        for b in range(a + 1, g + 1):
            for p in range(g):
                for q in range(p + 1, g + 1):
                    hc = cc[b, q] - cc[a, q] - cc[b, p] + cc[a, p]
                    hn = cn[b, q] - cn[a, q] - cn[b, p] + cn[a, p]
                    n_c, n_n = hc.sum(), hn.sum()
                    if n_c < MIN_BOX_CONTROL or n_n < MIN_BOX_NEGATIVE:
                        continue
                    t_bin = int(np.searchsorted(np.cumsum(hn) / n_n, 1.0 - NULL_SHARE, side='left'))
                    excess = hc[t_bin + 1:].sum() - NULL_SHARE * n_c
                    rows.append({
                        'x0': float(xe[a]), 'x1': float(xe[b]), 'y0': float(ye[p]), 'y1': float(ye[q]),
                        'n_control': int(n_c), 'n_negative': int(n_n), 'threshold': float(ve[t_bin + 1]),
                        'excess': float(excess), 'fraction': float(excess / n_c),
                        'z': float(excess / np.sqrt(NULL_SHARE * (1.0 - NULL_SHARE) * n_c)),
                    })
    return rows


def largest_rectangle_z(control_xy: np.ndarray, control_events: np.ndarray,
                        negative_xy: np.ndarray, negative_events: np.ndarray) -> tuple[float, int] | None:
    """The largest rectangle excess z over all detectors (columns) and its detector index;
    None when no rectangle holds enough events."""
    best = None
    for j in range(control_events.shape[1]):
        rows = rectangle_scores(control_xy, control_events[:, j], negative_xy, negative_events[:, j])
        if rows:
            z = max(r['z'] for r in rows)
            if best is None or z > best[0]:
                best = (z, j)
    return best
