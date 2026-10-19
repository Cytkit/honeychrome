"""
af_auto_assign.py
-----------------
AF profiles for samples from forward and side scatter, and folder groups for
assigning profiles by hand.

Each AF profile's source unstained sample (the reference) is modelled as a
Gaussian mixture in log10(FSC) x log10(SSC), with 1 to MAX_COMPONENTS
components chosen by BIC. A sample's scatter is then modelled as a mixture of
these fixed reference densities, with weights fitted by EM. The candidates
are the best single reference plus the references holding at least
CANDIDATE_WEIGHT in an EM fit over all references (at most MAX_CANDIDATES,
by weight). Every subset of 1 to MAX_REFERENCES candidates in which each
reference keeps at least MIN_WEIGHT of the events is scored by
    BIC = -2 log L + (m - 1) ln n      (m references, n events)
and the lowest wins. The sample is given every profile extracted from the
references chosen, in stored profile order.

Fit check: an event is outside a reference when its log density under that
reference is below the OUTSIDE_QUANTILE quantile of the reference's own
events. The share of a sample's events outside every chosen reference is
about OUTSIDE_QUANTILE for a good match; samples at or above POOR_FIT_SHARE
are listed for review.

Simplifications (v1):
  - every event is used, without a gate, as AF extraction uses the whole
    unstained sample; non-positive and saturated events are dropped;
  - at most MAX_EVENTS randomly chosen events are read per sample;
  - only the mixing weights count as parameters in the BIC, since the
    reference densities are fitted beforehand on other data;
  - references from bead samples are not used.

Public API
----------
folder_groups(sample_paths)
    {folder: [sample paths in it or below it]} for every folder holding a sample.
reference_sources(af_profiles, sample_paths, exclude=())
    {source sample path: [profile names]} for profiles extracted from a sample
    in the experiment.
log_scatter(scatter, ceilings)
    log10 of the usable events of an (n, 2) FSC/SSC array.
read_scatter(abs_path, pair, max_events)
    Raw (n, 2) FSC/SSC events of a sample file.
fit_reference(xy, max_components, seed)
    GaussianMixture of a reference, its number of components chosen by BIC.
mixture_weights(log_dens)
    EM weights and log-likelihood of a mixture of fixed densities.
choose_references(log_dens, min_weight, max_references)
    BIC choice of references for one sample.
suggest_af_profiles(experiment_dir, references, targets, pair, ceilings, progress)
    Suggested profiles for each target sample, with the evidence.
"""

import logging
from pathlib import Path, PurePosixPath

import numpy as np

logger = logging.getLogger(__name__)

MAX_EVENTS = 20000          # events read per sample
MIN_EVENTS = 200            # fewer usable events: no call
MAX_COMPONENTS = 6          # components per reference mixture
MIN_WEIGHT = 0.10           # smallest share a second or third reference may take
MAX_REFERENCES = 3          # references per sample
CANDIDATE_WEIGHT = 0.02     # share in the all-reference fit to be a candidate
MAX_CANDIDATES = 6
OUTSIDE_QUANTILE = 0.01     # reference's own events below this log density are outside it
POOR_FIT_SHARE = 0.05       # share of events outside every chosen reference to flag
EM_MAX_ITER = 500
EM_TOL = 1e-9               # relative change in log-likelihood


def folder_groups(sample_paths) -> dict:
    """{folder: [paths]} for every folder holding a sample, directly or in a
    subfolder; paths keep their input order and folders are sorted."""
    groups = {}
    for path in sample_paths:
        for parent in PurePosixPath(Path(str(path)).as_posix()).parents:
            folder = str(parent)
            if folder in ('', '.'):
                continue
            groups.setdefault(folder, []).append(path)
    return {folder: groups[folder] for folder in sorted(groups, key=str.lower)}


def reference_sources(af_profiles: dict, sample_paths, exclude=()) -> dict:
    """{source path: [profile names]} for profiles whose source sample is in
    *sample_paths* and not in *exclude*."""
    present = {str(p) for p in sample_paths}
    excluded = {str(p) for p in exclude}
    out = {}
    for name, entry in (af_profiles or {}).items():
        source = str(entry.get('source_fcs', ''))
        if source in present and source not in excluded:
            out.setdefault(source, []).append(name)
    return out


def log_scatter(scatter: np.ndarray, ceilings) -> np.ndarray:
    """log10 of the events that are finite, positive and below 0.999 of the
    ceiling on both channels."""
    scatter = np.asarray(scatter, dtype=np.float64)
    ceilings = np.broadcast_to(np.asarray(ceilings, dtype=np.float64), (2,))
    keep = (np.all(np.isfinite(scatter), axis=1)
            & np.all(scatter > 0, axis=1)
            & np.all(scatter < 0.999 * ceilings, axis=1))
    return np.log10(scatter[keep])


def read_scatter(abs_path, pair, max_events: int = MAX_EVENTS, seed: int = 0) -> np.ndarray:
    from honeychrome.controller_components.fcs_preview import load_fcs_preview
    try:
        _, events = load_fcs_preview(abs_path, max_events, col_order=list(pair))
    except ValueError:
        # integer FCS data: full read
        from honeychrome.controller_components.functions import sample_from_fcs
        events = sample_from_fcs(str(abs_path)).get_events('raw', col_order=list(pair))
        if len(events) > max_events:
            rows = np.random.default_rng(seed).choice(len(events), max_events, replace=False)
            events = events[np.sort(rows)]
    return np.asarray(events, dtype=np.float64)


def fit_reference(xy: np.ndarray, max_components: int = MAX_COMPONENTS, seed: int = 0):
    """Full-covariance GaussianMixture with the lowest BIC over 1..max_components."""
    from sklearn.mixture import GaussianMixture

    best, best_bic = None, np.inf
    for k in range(1, max_components + 1):
        if len(xy) < 10 * k:
            break
        gm = GaussianMixture(n_components=k, covariance_type='full',
                             random_state=seed, max_iter=200).fit(xy)
        bic = gm.bic(xy)
        if bic < best_bic:
            best, best_bic = gm, bic
    return best


def _mixture_loglik(log_dens: np.ndarray, weights: np.ndarray):
    a = log_dens + np.log(np.maximum(weights, 1e-300))
    top = a.max(axis=1, keepdims=True)
    s = np.exp(a - top)
    total = s.sum(axis=1, keepdims=True)
    return float(np.sum(top[:, 0] + np.log(total[:, 0]))), s / total


def mixture_weights(log_dens: np.ndarray, max_iter: int = EM_MAX_ITER, tol: float = EM_TOL):
    """EM mixing weights of fixed densities; *log_dens* is (n events, m)
    log densities. Returns (weights, log-likelihood)."""
    m = log_dens.shape[1]
    weights = np.full(m, 1.0 / m)
    ll, resp = _mixture_loglik(log_dens, weights)
    if m == 1:
        return weights, ll
    for _ in range(max_iter):
        weights = resp.mean(axis=0)
        new_ll, resp = _mixture_loglik(log_dens, weights)
        converged = new_ll - ll <= tol * abs(new_ll)
        ll = new_ll
        if converged:
            break
    return weights, ll


def choose_references(log_dens: np.ndarray, min_weight: float = MIN_WEIGHT,
                      max_references: int = MAX_REFERENCES) -> dict:
    """
    BIC choice of references for one sample. *log_dens* is (n events, R references).

    Returns {'indices', 'weights', 'loglik', 'bic', 'bic_single'}: the chosen
    reference columns by decreasing weight, their weights, and the BIC of the
    choice and of the best single reference.
    """
    from itertools import combinations

    n, n_ref = log_dens.shape
    single = log_dens.sum(axis=0)
    best = int(np.argmax(single))
    bic_single = -2.0 * float(single[best])
    result = ([best], np.array([1.0]), float(single[best]), bic_single)

    w_all, _ = mixture_weights(log_dens)
    ranked = [int(r) for r in np.argsort(-w_all, kind='stable') if w_all[r] >= CANDIDATE_WEIGHT]
    candidates = sorted(set(ranked[:MAX_CANDIDATES]) | {best})
    for m in range(2, min(max_references, len(candidates)) + 1):
        for idx in combinations(candidates, m):
            w, ll = mixture_weights(log_dens[:, list(idx)])
            if w.min() < min_weight:
                continue
            bic = -2.0 * ll + (m - 1) * np.log(n)
            if bic < result[3]:
                result = (list(idx), w, ll, bic)

    idx, w, ll, bic = result
    order = np.argsort(-np.asarray(w), kind='stable')
    return {'indices': [int(idx[i]) for i in order], 'weights': [float(w[i]) for i in order],
            'loglik': float(ll), 'bic': float(bic), 'bic_single': float(bic_single)}


def suggest_af_profiles(experiment_dir, references: dict, targets, pair, ceilings,
                        progress=None, max_events: int = MAX_EVENTS,
                        min_weight: float = MIN_WEIGHT,
                        max_references: int = MAX_REFERENCES) -> dict:
    """
    Suggested AF profiles for each target sample from its scatter.

    references : {source sample path: [profile names]} (reference_sources).
    targets    : sample paths to score.
    pair       : (FSC, SSC) channel names; ceilings one value per channel.
    progress   : optional callable taking a status text.

    Returns
      'assignments' : {target: [profile names]}
      'details'     : {target: {'references', 'weights', 'n_events',
                       'delta_bic' (single minus chosen), 'outside'}}
      'skipped'     : {target: reason}
      'unused_references' : {source: reason}
    'outside' is the share of the target's events outside every chosen
    reference (see the module docstring).
    """
    experiment_dir = Path(experiment_dir)
    report = progress or (lambda _text: None)
    models, floors, unused = {}, {}, {}
    for i, source in enumerate(references, start=1):
        report(f'Modelling reference scatter ({i}/{len(references)}): {Path(source).stem}')
        try:
            xy = log_scatter(read_scatter(experiment_dir / source, pair, max_events), ceilings)
        except Exception as exc:
            logger.warning(f'af_auto_assign: reference {source} not read: {exc}')
            unused[source] = f'not read ({exc})'
            continue
        if len(xy) < MIN_EVENTS:
            unused[source] = f'{len(xy)} usable events'
            continue
        gm = fit_reference(xy)
        models[source] = gm
        floors[source] = float(np.quantile(gm.score_samples(xy), OUTSIDE_QUANTILE))

    out = {'assignments': {}, 'details': {}, 'skipped': {}, 'unused_references': unused}
    sources = list(models)
    if not sources:
        for target in targets:
            out['skipped'][target] = 'no usable reference'
        return out

    targets = list(targets)
    for i, target in enumerate(targets, start=1):
        report(f'Scoring sample scatter ({i}/{len(targets)}): {Path(target).stem}')
        try:
            xy = log_scatter(read_scatter(experiment_dir / target, pair, max_events), ceilings)
        except Exception as exc:
            logger.warning(f'af_auto_assign: {target} not read: {exc}')
            out['skipped'][target] = f'not read ({exc})'
            continue
        if len(xy) < MIN_EVENTS:
            out['skipped'][target] = f'{len(xy)} usable events'
            continue
        log_dens = np.column_stack([models[s].score_samples(xy) for s in sources])
        choice = choose_references(log_dens, min_weight, max_references)
        chosen = [sources[j] for j in choice['indices']]
        names = []
        for source in sorted(chosen, key=sources.index):
            names.extend(n for n in references[source] if n not in names)
        inside = np.zeros(len(xy), dtype=bool)
        for j, source in zip(choice['indices'], chosen):
            inside |= log_dens[:, j] >= floors[source]
        out['assignments'][target] = names
        out['details'][target] = {
            'references': chosen,
            'weights': choice['weights'],
            'n_events': int(len(xy)),
            'delta_bic': choice['bic_single'] - choice['bic'],
            'outside': float(1.0 - inside.mean()),
        }
        logger.info(f'af_auto_assign: {target} -> {names} (weights '
                    f'{[round(w, 3) for w in choice["weights"]]}, ΔBIC '
                    f'{choice["bic_single"] - choice["bic"]:.1f})')
    report('')
    return out
