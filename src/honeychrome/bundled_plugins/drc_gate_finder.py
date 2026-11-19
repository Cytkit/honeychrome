"""
drc_gate_finder.py — Gating strategies that reproduce a Honeycluster cluster
=============================================================================
Companion module to ``dr_clustering_tab.py`` (filename intentionally does
NOT end in ``_tab.py``, so it is not picked up as a separate plugin tab).

What it answers
---------------
Can a cluster, found by the algorithm using all markers at once, also be
found with a few ordinary gates? The search looks for the fewest marker
thresholds ("CD4 above x", "CD62L below y") that, applied one after another
to the cells that were clustered, keep at least ``recall`` of the cluster's
cells (default 90%) while at least ``purity`` of the cells left in the gate
belong to the cluster (default 50%). The thresholds are found on one half of
the cells and every reported number is measured on the other half, so a
gate that only fits the cells it was tuned on is caught.

    recall = cluster cells inside the gate / all cluster cells
    purity = cluster cells inside the gate / all cells inside the gate

Each cluster ends with one of three verdicts:

* ``confirmed``: a gating strategy meets both targets.
* ``not_gateable``: no gating strategy does, but one weighted combination of
  all markers separates the cluster (it is a real population whose border
  with its neighbours runs through many markers at once, which a few biaxial
  gates cannot follow).
* ``continuum``: not even that combination separates it at the target
  recall; the cluster is a slice of a continuous population.

(``not_confirmed`` when the separability check is not run.)

For clusters that are not confirmed, the best compromise gate (the one that
balances recall and purity) and the clusters making up the impurity are
reported too.

How the search works
--------------------
The gate is a hyper-rectangle: a low and/or high cut on each marker used,
which is what range and rectangle gates are. With each marker's values
sorted once, a running total of cluster and non-cluster cells gives recall
and purity for every possible cut position in one pass, the other cuts held
fixed. The search repeatedly applies the single best cut change over all
markers (tightening or loosening) until nothing improves the objective by at
least ``MIN_GAIN``:

* Stepped search (the strategy): the highest purity with recall above a
  floor. The floor starts at 0.99 and is lowered in steps (``RECALL_STEPS``)
  to the training goal, so cuts are added one at a time. Asking for the
  final recall at once lets the first cut use the whole recall allowance,
  after which no second cut can be added.
* F-beta search (the compromise): the cut that best balances recall and
  purity, F_beta = (1 + b^2) P R / (b^2 P + R), where b below 1 favours
  purity and above 1 recall. Each of ``BETAS`` is tried and the best kept.

Markers are then removed, least useful first, re-fitting the others after
each removal; the strategy is the fewest markers that still meet the
targets on the training half (or the best one within ``max_markers``).
Training aims at recall + ``TRAIN_MARGIN`` and purity + ``TRAIN_MARGIN``,
because the validation half scatters around the training value; without the
margin, pruning stops exactly at a target and misses it on validation about
half the time.

Large runs are downsampled before the search (``TARGET_CAP`` cluster events,
``BACKGROUND_CAP`` others); each kept event carries a weight equal to the
inverse of its sampling fraction, so recall and purity are those of the full
data. Without the weights, dropping background events would inflate purity.

The separability check fits a logistic regression (a weighted sum of all
markers) on the training half, cuts it to keep the training recall goal and
measures its purity on the validation half.

v1 simplifications
------------------
* Rectangles only. Two-marker polygon gates were tested and rescued few
  clusters that rectangles could not.
* One pooled threshold per marker for every sample; per-sample results show
  where it fits some samples worse than others.
* Thresholds are on the run's own transform scale (the values the cluster
  was found on).

Pure NumPy apart from the separability check (scikit-learn); no Qt.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

DEFAULT_RECALL = 0.90
DEFAULT_PURITY = 0.50
DEFAULT_MAX_MARKERS = 6
TRAIN_MARGIN = 0.02
RECALL_STEPS = (0.99, 0.98, 0.96, 0.94)
BETAS = (0.5, 1.0, 2.0)
MIN_GAIN = 1e-3
TARGET_CAP = 50_000
BACKGROUND_CAP = 200_000
MIN_CLUSTER_EVENTS = 20
VALLEY_BINS = 128
VALLEY_WINDOW = 0.05

CONFIRMED = 'confirmed'
NOT_GATEABLE = 'not_gateable'
CONTINUUM = 'continuum'
NOT_CONFIRMED = 'not_confirmed'

HOLDOUT_EVENTS = 'events'
HOLDOUT_SAMPLES = 'samples'


@dataclass
class Targets:
    """What a gating strategy must achieve on the validation half."""
    recall: float = DEFAULT_RECALL
    purity: float = DEFAULT_PURITY
    max_markers: int | None = DEFAULT_MAX_MARKERS
    train_margin: float = TRAIN_MARGIN

    @property
    def train_recall(self) -> float:
        return min(self.recall + self.train_margin, 1.0)

    @property
    def train_purity(self) -> float:
        return min(self.purity + self.train_margin, 1.0)


@dataclass
class EventSet:
    """Events searched for one target: values (run transform scale), whether
    each is in the target, its weight, sample index, cluster label and
    whether it is in the training half."""
    X: np.ndarray
    y: np.ndarray
    w: np.ndarray
    sample: np.ndarray
    label: np.ndarray
    train: np.ndarray
    cluster_fraction: float


@dataclass
class Strategy:
    """Low and high cut per channel (-inf / inf where a side is uncut)."""
    lo: np.ndarray
    hi: np.ndarray

    def markers(self) -> list[int]:
        return [j for j in range(len(self.lo))
                if np.isfinite(self.lo[j]) or np.isfinite(self.hi[j])]

    def inside(self, X: np.ndarray, channels: list[int] | None = None) -> np.ndarray:
        """Events inside the cuts of *channels* (default: every cut)."""
        mask = np.ones(len(X), dtype=bool)
        for j in (self.markers() if channels is None else channels):
            mask &= (X[:, j] >= self.lo[j]) & (X[:, j] <= self.hi[j])
        return mask

    def describe(self, names: list[str]) -> str:
        parts = []
        for j in self.markers():
            cut = []
            if np.isfinite(self.lo[j]):
                cut.append(f'> {self.lo[j]:.4g}')
            if np.isfinite(self.hi[j]):
                cut.append(f'< {self.hi[j]:.4g}')
            parts.append(f'{names[j]} {" and ".join(cut)}')
        return '; '.join(parts)


@dataclass
class Metrics:
    recall: float
    purity: float

    @property
    def f1(self) -> float:
        s = self.recall + self.purity
        return 2 * self.recall * self.purity / s if s > 0 else 0.0


@dataclass
class Step:
    """One biaxial plot of the hierarchy (y is None for a single-marker
    step) and the validation metrics after applying it and every earlier
    step."""
    x: int
    y: int | None
    metrics: Metrics


@dataclass
class GateFinding:
    status: str
    met: bool
    strategy: Strategy
    train: Metrics
    validation: Metrics
    enrichment: float
    steps: list[Step]
    path: list[tuple[int, Metrics]]
    per_sample: list[dict]
    impurity: list[tuple[int, float]]
    compromise: Strategy | None = None
    compromise_metrics: Metrics | None = None
    tradeoff: list[dict] = field(default_factory=list)
    separability: Metrics | None = None
    seconds: float = 0.0


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

def select_events(X: np.ndarray, labels: np.ndarray, sample: np.ndarray, target,
                  *, weights: np.ndarray | None = None, holdout: str = HOLDOUT_EVENTS,
                  target_cap: int | None = TARGET_CAP,
                  background_cap: int | None = BACKGROUND_CAP, seed: int = 0) -> EventSet:
    """
    The events to search for one target (a cluster label or a list of
    labels treated as one): every target event up to *target_cap* and other
    events up to *background_cap*, weighted back to the full data, split
    into training and validation halves by event within each sample
    (``HOLDOUT_EVENTS``) or by whole samples (``HOLDOUT_SAMPLES``).

    weights: optional weight of each event when the events are already a
        sample of the data (for example a fixed number per cluster): the
        number of cells each event stands for. The capping weights multiply
        it.
    """
    labels = np.asarray(labels)
    targets = np.atleast_1d(np.asarray(target))
    in_target = np.isin(labels, targets)
    base = np.ones(len(labels)) if weights is None else np.asarray(weights, dtype=np.float64)
    rng = np.random.default_rng(seed)
    tgt, bg = np.flatnonzero(in_target), np.flatnonzero(~in_target)
    if len(tgt) < 2 * MIN_CLUSTER_EVENTS:
        raise ValueError(f'the target has {len(tgt)} events; at least '
                         f'{2 * MIN_CLUSTER_EVENTS} are needed')
    w_t = w_b = 1.0
    if target_cap and len(tgt) > target_cap:
        w_t = len(tgt) / target_cap
        tgt = rng.choice(tgt, target_cap, replace=False)
    if background_cap and len(bg) > background_cap:
        w_b = len(bg) / background_cap
        bg = rng.choice(bg, background_cap, replace=False)
    idx = np.sort(np.concatenate([tgt, bg]))
    y = in_target[idx]
    s = np.asarray(sample)[idx]
    if holdout == HOLDOUT_SAMPLES:
        ids = np.unique(s)
        if len(ids) < 2:
            raise ValueError('holding out samples needs at least two samples')
        held = rng.permutation(ids)[:len(ids) // 2]
        train = ~np.isin(s, held)
    else:
        train = rng.random(len(idx)) < 0.5
    for half in (train, ~train):
        if y[half].sum() < MIN_CLUSTER_EVENTS:
            raise ValueError('too few target events in the training or validation half')
    return EventSet(X=np.asarray(X[idx], dtype=np.float32), y=y,
                    w=base[idx] * np.where(y, w_t, w_b), sample=s,
                    label=labels[idx], train=train,
                    cluster_fraction=float(base[in_target].sum() / base.sum()))


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def constrained_score(recall_floor: float):
    """Purity while recall >= recall_floor; below it, how close recall is."""
    def score(r, p):
        return np.where(r >= recall_floor - 1e-12, p + 1e-6 * r, r - 1.0)
    return score


def fbeta_score(beta: float):
    b2 = beta * beta

    def score(r, p):
        den = b2 * p + r
        with np.errstate(invalid='ignore', divide='ignore'):
            return np.where(den > 0, (1 + b2) * p * r / den, 0.0)
    return score


class RectangleSearch:
    """
    A hyper-rectangle (low and high cut per channel) over weighted events.
    Arrays are kept column-major so each channel's sorted values, weights and
    candidate cuts are contiguous.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, w: np.ndarray):
        self.X = np.asfortranarray(X)
        self.m, self.d = self.X.shape
        order = np.argsort(self.X, axis=0, kind='stable').astype(np.int32)
        self.order = np.asfortranarray(order)
        self.Xs = np.asfortranarray(np.take_along_axis(self.X, order, axis=0))
        self.w = np.asarray(w, dtype=np.float64)
        self.tw = self.w * np.asarray(y, dtype=bool)
        self.t_total = float(self.tw.sum())
        self.tw_sorted = np.asfortranarray(self.tw[order])
        self.w_sorted = np.asfortranarray(self.w[order])
        distinct = self.Xs[1:] > self.Xs[:-1]
        ones = np.ones((1, self.d), dtype=bool)
        self.valid_lo = np.asfortranarray(np.vstack([ones, distinct]))
        self.valid_hi = np.asfortranarray(np.vstack([distinct, ones]))
        mid = 0.5 * (self.Xs[1:] + self.Xs[:-1])
        self.thr_lo = np.asfortranarray(np.vstack([np.full((1, self.d), -np.inf), mid]))
        self.thr_hi = np.asfortranarray(np.vstack([mid, np.full((1, self.d), np.inf)]))
        self.lo = np.full(self.d, -np.inf)
        self.hi = np.full(self.d, np.inf)
        self.outside = np.zeros((self.m, self.d), dtype=bool, order='F')
        self.n_outside = np.zeros(self.m, dtype=np.int16)

    def set(self, j: int, lo: float, hi: float) -> None:
        new = (self.X[:, j] < lo) | (self.X[:, j] > hi)
        self.n_outside += new.astype(np.int16) - self.outside[:, j].astype(np.int16)
        self.outside[:, j] = new
        self.lo[j], self.hi[j] = lo, hi

    def strategy(self) -> Strategy:
        return Strategy(self.lo.copy(), self.hi.copy())

    def load(self, strategy: Strategy) -> None:
        for j in range(self.d):
            if strategy.lo[j] != self.lo[j] or strategy.hi[j] != self.hi[j]:
                self.set(j, strategy.lo[j], strategy.hi[j])

    def used(self) -> list[int]:
        return [j for j in range(self.d) if np.isfinite(self.lo[j]) or np.isfinite(self.hi[j])]

    def metrics(self) -> Metrics:
        inside = self.n_outside == 0
        t = float(self.tw[inside].sum())
        a = float(self.w[inside].sum())
        return Metrics(t / self.t_total, t / a if a > 0 else 0.0)

    def candidates(self, j: int, side: str):
        """Every cut position on channel j for one side, the other cuts
        fixed: (thresholds, recall, purity, valid), one per sorted event."""
        keep = ((self.n_outside - self.outside[:, j]) == 0)[self.order[:, j]]
        xs = self.Xs[:, j]
        if side == 'lo':
            keep &= xs <= self.hi[j]
            t = (self.tw_sorted[:, j] * keep)[::-1].cumsum()[::-1]
            a = (self.w_sorted[:, j] * keep)[::-1].cumsum()[::-1]
            valid = self.valid_lo[:, j] & keep
            thr = self.thr_lo[:, j]
        else:
            keep &= xs >= self.lo[j]
            t = (self.tw_sorted[:, j] * keep).cumsum()
            a = (self.w_sorted[:, j] * keep).cumsum()
            valid = self.valid_hi[:, j] & keep
            thr = self.thr_hi[:, j]
        valid &= a > 0
        with np.errstate(invalid='ignore', divide='ignore'):
            p = t / a
        return thr, t / self.t_total, p, valid

    def search(self, score, channels: list[int] | None = None,
               max_moves: int | None = None) -> Metrics:
        """Apply the best single cut change until none improves *score*."""
        channels = list(range(self.d)) if channels is None else list(channels)
        m = self.metrics()
        current = float(score(np.array([m.recall]), np.array([m.purity]))[0])
        for _ in range(max_moves or 4 * self.d):
            best = (current + MIN_GAIN, None, None, None)
            for j in channels:
                for side in ('lo', 'hi'):
                    thr, r, p, valid = self.candidates(j, side)
                    s = score(r, p)
                    s[~valid] = -np.inf
                    k = int(np.argmax(s))
                    if s[k] > best[0]:
                        best = (float(s[k]), j, side, float(thr[k]))
            if best[1] is None:
                break
            current, j, side, cut = best
            if side == 'lo':
                self.set(j, cut, self.hi[j])
            else:
                self.set(j, self.lo[j], cut)
        return self.metrics()

    def stepped(self, recall: float) -> Metrics:
        """Constrained search with the recall floor lowered in steps."""
        for floor in [f for f in RECALL_STEPS if f > recall] + [recall]:
            self.search(constrained_score(floor))
        return self.metrics()

    def prune_path(self, score, recall: float, purity: float) -> list[tuple[Strategy, Metrics]]:
        """Remove markers, least useful first, re-fitting the rest after each
        removal: the strategy at each number of markers, from the starting
        number down to one."""
        path = [(self.strategy(), self.metrics())]
        while len(self.used()) > 1:
            used = self.used()
            trials = []
            for j in used:
                lo, hi = self.lo[j], self.hi[j]
                self.set(j, -np.inf, np.inf)
                m = self.metrics()
                trials.append((m.recall >= recall - 1e-9 and m.purity >= purity,
                               m.purity, m.recall, j))
                self.set(j, lo, hi)
            j = max(trials)[3]
            self.set(j, -np.inf, np.inf)
            rest = [k for k in used if k != j]
            self.search(score, channels=rest, max_moves=2 * len(rest) + 2)
            path.append((self.strategy(), self.metrics()))
        return path


def choose_from_path(path, recall: float, purity: float,
                     max_markers: int | None) -> tuple[Strategy, Metrics, bool]:
    """The fewest markers meeting both targets within *max_markers*; if none
    does, the purest within it that keeps the recall, else the best recall."""
    allowed = [e for e in path if max_markers is None or len(e[0].markers()) <= max_markers]
    if not allowed:
        allowed = [path[-1]]
    ok = [e for e in allowed if e[1].recall >= recall - 1e-9 and e[1].purity >= purity]
    if ok:
        s, m = min(ok, key=lambda e: len(e[0].markers()))
        return s, m, True
    s, m = max(allowed, key=lambda e: (e[1].purity if e[1].recall >= recall - 1e-9 else -1.0,
                                       e[1].recall))
    return s, m, False


def snap_to_valleys(search: RectangleSearch, recall: float, purity: float) -> int:
    """Move each threshold to the deepest density valley within
    VALLEY_WINDOW of the marker's range when both targets still hold, so
    gates sit where a person would draw them. Returns how many moved."""
    moved = 0
    kernel = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
    kernel /= kernel.sum()
    for j in search.used():
        others_in = (search.n_outside - search.outside[:, j]) == 0
        x, wx = search.X[others_in, j], search.w[others_in]
        q0, q1 = np.quantile(search.X[:, j], [0.001, 0.999])
        if len(x) < 50 or not q1 > q0:
            continue
        edges = np.linspace(q0, q1, VALLEY_BINS + 1)
        dens = np.convolve(np.histogram(x, bins=edges, weights=wx)[0], kernel, mode='same')
        centres = 0.5 * (edges[1:] + edges[:-1])
        minima = np.flatnonzero((dens[1:-1] < dens[:-2]) & (dens[1:-1] <= dens[2:])) + 1
        for side in ('lo', 'hi'):
            cut = search.lo[j] if side == 'lo' else search.hi[j]
            if not np.isfinite(cut) or len(minima) == 0:
                continue
            near = minima[np.abs(centres[minima] - cut) <= VALLEY_WINDOW * (q1 - q0)]
            if len(near) == 0:
                continue
            new = float(centres[near[np.argmin(dens[near])]])
            lo, hi = search.lo[j], search.hi[j]
            if side == 'lo':
                search.set(j, new, hi)
            else:
                search.set(j, lo, new)
            m = search.metrics()
            if m.recall >= recall - 1e-9 and m.purity >= purity:
                moved += 1
            else:
                search.set(j, lo, hi)
    return moved


# ---------------------------------------------------------------------------
# Measuring a strategy
# ---------------------------------------------------------------------------

def measure(inside: np.ndarray, y: np.ndarray, w: np.ndarray) -> Metrics:
    tw = w * y
    total = float(tw.sum())
    t = float(tw[inside].sum())
    a = float(w[inside].sum())
    return Metrics(t / total if total > 0 else 0.0, t / a if a > 0 else 0.0)


def hierarchy_steps(strategy: Strategy, X: np.ndarray, y: np.ndarray,
                    w: np.ndarray) -> list[Step]:
    """
    The strategy as a sequence of biaxial plots: markers ordered by how much
    purity is lost without them, each paired with the remaining marker that
    gives the purest gate together with the steps before it. Metrics are
    cumulative.
    """
    used = strategy.markers()
    full = measure(strategy.inside(X), y, w).purity
    loss = {j: full - measure(strategy.inside(X, [k for k in used if k != j]), y, w).purity
            for j in used}
    remaining = sorted(used, key=lambda j: loss[j], reverse=True)
    steps, applied = [], []
    while remaining:
        first = remaining.pop(0)
        partner = None
        if remaining:
            partner = max(remaining, key=lambda k: measure(
                strategy.inside(X, applied + [first, k]), y, w).purity)
            remaining.remove(partner)
        applied += [first] + ([partner] if partner is not None else [])
        steps.append(Step(first, partner, measure(strategy.inside(X, applied), y, w)))
    return steps


def per_sample_metrics(inside: np.ndarray, es: EventSet, half: np.ndarray) -> list[dict]:
    """Recall and purity in each sample (validation events)."""
    rows = []
    y, w, s = es.y[half], es.w[half], es.sample[half]
    for sid in np.unique(s):
        m = s == sid
        n = int(y[m].sum())
        if n == 0:
            continue
        mm = measure(inside[m], y[m], w[m])
        rows.append({'sample': int(sid), 'n_target': n, 'recall': mm.recall, 'purity': mm.purity})
    return rows


def impurity_by_cluster(inside: np.ndarray, es: EventSet, half: np.ndarray,
                        top: int = 3) -> list[tuple[int, float]]:
    """The clusters making up the non-target cells in the gate, as shares of
    all cells in the gate, largest first."""
    y, w, lab = es.y[half], es.w[half], es.label[half]
    total = float(w[inside].sum())
    other = inside & ~y
    if total <= 0 or not other.any():
        return []
    ids = np.unique(lab[other])
    shares = [(int(c), float(w[other & (lab == c)].sum()) / total) for c in ids]
    return sorted(shares, key=lambda t: -t[1])[:top]


def linear_separability(es: EventSet, recall: float) -> Metrics:
    """Validation recall and purity of a weighted sum of all markers
    (logistic regression), cut to keep *recall* of the target on the
    training half."""
    from sklearn.linear_model import LogisticRegression
    tr, va = es.train, ~es.train
    model = LogisticRegression(max_iter=500, C=10.0)
    model.fit(es.X[tr], es.y[tr], sample_weight=es.w[tr])
    score = model.decision_function(es.X[tr])
    order = np.argsort(-score)
    tw = (es.w[tr] * es.y[tr])[order]
    k = min(int(np.searchsorted(np.cumsum(tw) / tw.sum(), recall)), len(order) - 1)
    return measure(model.decision_function(es.X[va]) >= score[order][k], es.y[va], es.w[va])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def find_gating_strategy(es: EventSet, targets: Targets | None = None, *,
                         snap: bool = False, compromise: bool = True,
                         separability: bool = True) -> GateFinding:
    """
    The minimal rectangle gating strategy for the target of *es* and, when
    it does not meet the targets, the best compromise, the recall-purity
    trade-off and the separability check that decides between
    ``NOT_GATEABLE`` and ``CONTINUUM``.
    """
    targets = targets or Targets()
    start = time.perf_counter()
    tr, va = es.train, ~es.train
    goal, goal_purity = targets.train_recall, targets.train_purity
    search = RectangleSearch(es.X[tr], es.y[tr], es.w[tr])
    search.stepped(goal)
    path = search.prune_path(constrained_score(goal), goal, goal_purity)
    strategy, train_m, _ = choose_from_path(path, goal, goal_purity, targets.max_markers)
    if snap:
        search.load(strategy)
        snap_to_valleys(search, goal, goal_purity)
        strategy, train_m = search.strategy(), search.metrics()

    inside_v = strategy.inside(es.X[va])
    val = measure(inside_v, es.y[va], es.w[va])
    met = val.recall >= targets.recall and val.purity >= targets.purity
    finding = GateFinding(
        status=CONFIRMED if met else NOT_CONFIRMED, met=met, strategy=strategy,
        train=train_m, validation=val,
        enrichment=val.purity / es.cluster_fraction if es.cluster_fraction > 0 else float('nan'),
        steps=hierarchy_steps(strategy, es.X[va], es.y[va], es.w[va]),
        path=[(len(s.markers()), measure(s.inside(es.X[va]), es.y[va], es.w[va])) for s, _ in path],
        per_sample=per_sample_metrics(inside_v, es, va),
        impurity=impurity_by_cluster(inside_v, es, va))

    if not met and compromise:
        best = None
        for beta in BETAS:
            fb = RectangleSearch(es.X[tr], es.y[tr], es.w[tr])
            fb.search(fbeta_score(beta))
            fb_path = fb.prune_path(fbeta_score(beta), goal, goal_purity)
            s, m, ok = choose_from_path(fb_path, goal, goal_purity, targets.max_markers)
            v = measure(s.inside(es.X[va]), es.y[va], es.w[va])
            finding.tradeoff.append({'beta': beta, 'n_markers': len(s.markers()),
                                     'recall': v.recall, 'purity': v.purity})
            key = (ok, m.purity if ok else m.f1)
            if best is None or key > best[0]:
                best = (key, s, v)
        finding.compromise, finding.compromise_metrics = best[1], best[2]

    if not met and separability:
        sep = linear_separability(es, goal)
        finding.separability = sep
        separable = sep.recall >= targets.recall and sep.purity >= targets.purity
        finding.status = NOT_GATEABLE if separable else CONTINUUM
    finding.seconds = time.perf_counter() - start
    return finding


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def _cut(v: float):
    return None if not np.isfinite(v) else float(v)


def _uncut(v, default: float) -> float:
    return default if v is None else float(v)


def _strategy_to_dict(strategy: Strategy | None, channels: list[str]) -> dict | None:
    if strategy is None:
        return None
    return {channels[j]: [_cut(strategy.lo[j]), _cut(strategy.hi[j])]
            for j in strategy.markers()}


def strategy_from_dict(cuts: dict | None, channels: list[str]) -> Strategy | None:
    """Rebuild a strategy over *channels* from its saved {channel: [lo, hi]};
    None when a channel it cuts is not in *channels*."""
    if cuts is None:
        return None
    lo, hi = np.full(len(channels), -np.inf), np.full(len(channels), np.inf)
    for ch, (a, b) in cuts.items():
        if ch not in channels:
            return None
        j = channels.index(ch)
        lo[j], hi[j] = _uncut(a, -np.inf), _uncut(b, np.inf)
    return Strategy(lo, hi)


def _metrics_dict(m: Metrics | None) -> dict | None:
    return None if m is None else {'recall': m.recall, 'purity': m.purity}


def finding_to_dict(f: GateFinding, channels: list[str]) -> dict:
    """A JSON-ready copy of *f*, markers named by *channels* (the columns
    of the events searched)."""
    return {
        'status': f.status, 'met': bool(f.met),
        'strategy': _strategy_to_dict(f.strategy, channels),
        'train': _metrics_dict(f.train), 'validation': _metrics_dict(f.validation),
        'enrichment': float(f.enrichment),
        'steps': [{'x': channels[s.x], 'y': None if s.y is None else channels[s.y],
                   **_metrics_dict(s.metrics)} for s in f.steps],
        'path': [{'n_markers': n, **_metrics_dict(m)} for n, m in f.path],
        'per_sample': list(f.per_sample),
        'impurity': [[int(c), float(v)] for c, v in f.impurity],
        'compromise': _strategy_to_dict(f.compromise, channels),
        'compromise_metrics': _metrics_dict(f.compromise_metrics),
        'tradeoff': list(f.tradeoff),
        'separability': _metrics_dict(f.separability),
        'seconds': float(f.seconds),
    }

