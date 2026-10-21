"""
spectral_correction.py
----------------------
Fix My Unmix on the experiment's fully stained samples (AutoSpectral tab,
Spectral Correction page). No Qt. The numerics are ``fix_my_unmix.py``.

Samples and groups
    The samples offered are every sample that is not a bead sample, an
    unstained sample or a single-stain control, plus those types the page's
    *Include* tick boxes add (``offered_samples``); the default selection is
    the samples of none of those types (``default_samples``). They run
    together as one group, or one group per raw subfolder
    (``sample_groups``, *Concatenate per group*); each group gets its own
    corrected spectra, and each sample is unmixed with its group's.
Unstained match
    Each group's unstained sample (its positivity thresholds) is the
    unstained cell sample whose scatter fits the group's samples best: every
    sample is matched to one reference by ``af_auto_assign`` (log FSC x SSC
    mixtures, BIC), and the reference matched by the most events wins
    (``match_unstained``). The page can override it.
Run
    ``correction_snapshot`` (main thread) gathers the base spectra, each
    sample's AF library, the spillover spread, the spectral variants (the
    variant prior) and the gating; ``run_corrections`` (worker) loads each
    group's samples in the gate, takes an equal share of the group's event
    budget from each, unmixes every sample by per-cell AF with its own AF
    library (unpenalised, on the base spectra), and runs ``fix_my_unmix``.
    Every run starts from the base spectra. With *Smooth updates* (the
    default, ``FixSettings.smooth_updates``) the group's raw events are kept
    and ``fix_my_unmix.smooth_fix`` runs instead: each round draws a fresh
    share of them, unmixes it with the current spectra (each AF library
    precomputed again with its envelopes rebuilt for them,
    ``unmix_method.rebuilt_envelope``) and moves the spectra part of the way
    toward the round's result.
    After the groups, the spillover spread is computed again from the
    single-stain controls under each group's corrected spectra
    (``spread_model.compute_spread_entry`` with projection sets: one pass
    over the controls for every group).
Storage
    ``store_corrections`` writes the overlay ``unmix_method`` applies
    (``process['spectral_corrections']``; format in its module docstring)
    with the accepted rows Active, each group's spread, and the run's log
    per group; the page's choices are kept in
    ``process['spectral_correction_setup']``. The overlay is dropped when the
    base spectra change (``clear_if_stale``). The envelopes, variants and
    thresholds that follow the corrected spectra are derived by
    ``unmix_method.set_extras``.

v1 simplifications
    * Every sample is read whole and gated before its share of events is
      drawn (as the noise model and the spread do).
    * Smooth updates: every round's envelopes carry the base spillover
      spread (the run's own thresholds use it throughout); the spread under
      the corrected spectra is computed once, after the run.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import PurePosixPath

import numpy as np

from honeychrome.controller_components import fix_my_unmix as fm
from honeychrome.controller_components import unmix_method

logger = logging.getLogger(__name__)

STORE_KEY = unmix_method.CORRECTIONS_KEY
SETUP_KEY = 'spectral_correction_setup'
AUTO = '__auto__'
ALL_GROUP = 'All samples'
TOP_FOLDER = '(experiment folder)'
DEFAULT_EVENTS = 200_000
DEFAULT_SEED = 42
SAMPLE_TYPES = ('unstained', 'single_stain', 'beads')
MIN_EVENTS = 200

# FixSettings fields the page exposes (Advanced Settings), in display order:
# (field, label, kind, minimum, maximum, step, decimals); kind 'int', 'float',
# 'bool' or a tuple of (value, label) choices.
EXPOSED_SETTINGS = (
    ('smooth_updates', 'Smooth updates (rounds averaged; slower)', 'bool', None, None, None, None),
    ('max_iter', 'Maximum iterations', 'int', 1, 200, 1, None),
    ('convergence_threshold', 'Convergence threshold', 'float', 1e-5, 0.5, 0.001, 5),
    ('convergence_quantile', 'Convergence quantile', 'float', 0.5, 1.0, 0.01, 3),
    ('unstained_threshold', 'Unstained quantile', 'float', 0.9, 0.9999, 0.001, 4),
    ('unstained_margin', 'Unstained margin', 'float', 1.0, 5.0, 0.1, 2),
    ('spread_kappa', 'Spread kappa (SDs)', 'float', 0.0, 10.0, 0.5, 2),
    ('max_coefficient', 'Maximum coefficient', 'float', 0.01, 2.0, 0.05, 3),
    ('coefficient_decay', 'Coefficient decay', 'float', 0.0, 0.99, 0.01, 3),
    ('min_hypernegative_frac', 'Hypernegative: trust starts falling at', 'float', 0.0, 0.5, 0.005, 3),
    ('max_hypernegative_frac', 'Hypernegative: zero trust at', 'float', 0.001, 1.0, 0.005, 3),
    ('hypernegative_rule', 'Hypernegative: judged on',
     (('after', 'Share left below the boundary'), ('new', 'Share newly pushed below it')), None, None, None, None),
    ('source_events_low', 'Clear source positives beyond unstained: no trust at', 'int', 0, 1_000_000, 10, None),
    ('source_events_high', 'Clear source positives beyond unstained: full trust at', 'int', 1, 1_000_000, 10,
     None),
    ('glasso_check', 'Joint lasso check on pair coefficients', 'bool', None, None, None, None),
    ('af_gate', 'AF gate on row changes', 'bool', None, None, None, None),
    ('af_gate_max', 'AF gate: maximum likeness (cos² to an AF spectrum)', 'float', 0.0, 1.0, 0.05, 3),
    ('phase_two_rows', 'Phase two: rows re-measured',
     (('flagged', 'Only rows phase one changed'), ('all', 'All')), None, None, None, None),
    ('phase_two_adopt', 'Phase two: row adopted when the gates pass',
     (('remeasured', 'Re-measured signature'), ('backsolved', 'Phase-one back-solved row')), None, None, None, None),
    ('max_angle', 'Phase two: maximum change (deg)', 'float', 0.1, 90.0, 1.0, 2),
    ('downsample', 'Stratified downsample (0 = off)', 'int', 0, 10_000_000, 10_000, None),
)


# ---------------------------------------------------------------------------
# Setup (stored with the experiment)
# ---------------------------------------------------------------------------

@dataclass
class CorrectionSetup:
    """The page's choices; ``samples`` None means the default samples."""
    samples: list | None = None
    per_group: bool = False
    gate: str | None = None
    n_events: int = DEFAULT_EVENTS
    seed: int = DEFAULT_SEED
    unstained: dict | None = None            # {group: sample path or AUTO}
    include: dict | None = None               # {sample type: offered}, see SAMPLE_TYPES
    fix: dict | None = None                   # exposed FixSettings fields changed from their defaults
    fixed: list | None = None                 # fluorophores Fix Unmix never corrects (FixSettings.fixed_fluorophores)


def load_setup(process: dict) -> CorrectionSetup:
    stored = process.get(SETUP_KEY) or {}
    names = {f.name for f in fields(CorrectionSetup)}
    return CorrectionSetup(**{k: v for k, v in stored.items() if k in names})


def save_setup(process: dict, setup: CorrectionSetup) -> None:
    process[SETUP_KEY] = asdict(setup)


def fix_settings(setup: CorrectionSetup) -> fm.FixSettings:
    """FixSettings from the exposed fields stored in the setup; the rest at their defaults."""
    allowed = {key for key, *_rest in EXPOSED_SETTINGS}
    values = {k: v for k, v in (setup.fix or {}).items() if k in allowed}
    s = fm.FixSettings(**values)
    s.n_threads = unmix_method.n_threads_default()
    s.fixed_fluorophores = tuple(str(n) for n in (setup.fixed or ()))
    return s


# ---------------------------------------------------------------------------
# Samples and groups
# ---------------------------------------------------------------------------

def sample_types(samples: dict, path: str, name: str | None = None) -> set:
    """The control types of a sample among SAMPLE_TYPES (empty for a fully stained sample)."""
    from honeychrome.controller_components.gls_unmix import _is_unstained
    from honeychrome.controller_components.particle_type import is_bead_sample

    name = (samples.get('all_samples') or {}).get(path, '') if name is None else name
    out = set()
    if _is_unstained(samples, path, name):
        out.add('unstained')
    if path in set(samples.get('single_stain_controls') or []):
        out.add('single_stain')
    if is_bead_sample(samples, path, name):
        out.add('beads')
    return out


def offered_samples(samples: dict, include: dict | None = None) -> list:
    """
    Samples offered for correction, in experiment order: a sample of a
    control type is offered only when one of its types is included.
    """
    include = include or {}
    out = []
    for path, name in (samples.get('all_samples') or {}).items():
        types = sample_types(samples, path, name)
        if types and not any(include.get(t) for t in types):
            continue
        out.append(path)
    return out


def default_samples(samples: dict) -> list:
    """Every sample that is not a bead sample, an unstained sample or a single-stain control."""
    return offered_samples(samples, None)


def group_of(path: str) -> str:
    """The raw subfolder a sample sits in."""
    parent = str(PurePosixPath(str(path).replace('\\', '/')).parent)
    return TOP_FOLDER if parent in ('', '.') else parent


def sample_groups(paths, per_group: bool) -> dict:
    """{group: [paths]}: one group of every sample, or one per raw subfolder (in sample order)."""
    paths = [str(p) for p in paths]
    if not per_group:
        return {ALL_GROUP: paths} if paths else {}
    groups = {}
    for p in paths:
        groups.setdefault(group_of(p), []).append(p)
    return groups


def unstained_choices(controller) -> list:
    """(path, name, has_af) of every unstained cell sample: the candidates for a group's unstained."""
    from honeychrome.controller_components.gls_unmix import unstained_cell_samples

    return [(p, n, bool(_unstained_profiles(controller, p))) for p, n in
            unstained_cell_samples(controller.experiment.samples)]


def _unstained_profiles(controller, path: str) -> list:
    """AF profiles for an unstained sample's events: those assigned to it, else the one extracted from it."""
    from honeychrome.controller_components.af_auto_assign import reference_sources
    from honeychrome.controller_components.sample_loader import af_profile_names

    assigned = af_profile_names(controller, path)
    if assigned:
        return assigned
    sources = reference_sources(controller.experiment.process.get('af_profiles') or {}, [path])
    return list(sources.get(path, []))


def match_unstained(experiment_dir, groups: dict, references: list, pair, ceilings, progress=None) -> dict:
    """
    Each group's unstained sample by scatter: every sample of the group is
    matched to its single best reference (``af_auto_assign``), and the
    reference matched by the most events wins. Returns {group: {'unstained':
    path or None, 'votes': {path: events}, 'skipped': {sample: reason}}}.
    """
    from honeychrome.controller_components.af_auto_assign import suggest_af_profiles

    targets = list(dict.fromkeys(p for members in groups.values() for p in members))
    out = {g: {'unstained': None, 'votes': {}, 'skipped': {}} for g in groups}
    if not references or not targets or pair is None:
        for g in groups:
            out[g]['skipped'] = {p: 'no unstained reference or scatter channels' for p in groups[g]}
        return out
    refs = {str(p): [] for p in references}
    found = suggest_af_profiles(experiment_dir, refs, targets, pair, ceilings, progress=progress,
                                max_references=1)
    for g, members in groups.items():
        votes = Counter()
        for p in members:
            det = found['details'].get(p)
            if det is None or not det.get('references'):
                out[g]['skipped'][p] = found['skipped'].get(p, 'no match')
                continue
            votes[det['references'][0]] += int(det.get('n_events', 1))
        if votes:
            best = max(votes.values())
            out[g]['unstained'] = next(r for r in references if votes.get(str(r)) == best)
        out[g]['votes'] = dict(votes)
    return out


# ---------------------------------------------------------------------------
# Snapshot (main thread)
# ---------------------------------------------------------------------------

def _drop_envelope_rows(envelope: dict | None, keep: list) -> dict | None:
    """A runtime envelope over the kept fluorophores only."""
    if envelope is None:
        return None
    out = dict(envelope)
    out['null_var'] = np.asarray(envelope['null_var'], dtype=np.float64)[..., keep]
    if envelope.get('spread') is not None:
        out['spread'] = np.asarray(envelope['spread'], dtype=np.float64)[np.ix_(keep, keep)]
    return out


def _af_library(controller, profiles: list, fl: list, n_names: int):
    """``(spectra, precomputed)`` of an AF profile assignment over the panel's non-AF rows, or None."""
    from honeychrome.controller_components.autospectral_functions import build_af_precomputed
    from honeychrome.controller_components.sample_loader import resolve_af_for_profiles

    if not profiles:
        return None
    for name in profiles:
        try:
            controller.precomputed_for_profile(name)
        except Exception as exc:
            logger.warning('Spectral correction: AF profile %s not cached (%s)', name, exc)
    af = resolve_af_for_profiles(controller, profiles)
    if af is None:
        return None
    pre = af.precomputed
    if len(fl) != n_names:
        spectra_fl = pre['S_t'].T[fl]
        pre = build_af_precomputed(spectra_fl, af.spectra, _drop_envelope_rows(pre.get('envelope'), fl))
    return {'profiles': tuple(af.profile_names), 'spectra': np.asarray(af.spectra, dtype=np.float64),
            'precomputed': pre}


def variant_deltas(controller, fluor_names: list) -> list:
    """Per fluorophore, its spectral variants' deltas (Spectral Variants page) or None."""
    store = getattr(controller, 'autospectral_variants', None) or {}
    out = []
    for name in fluor_names:
        delta = (store.get(name) or {}).get('delta')
        out.append(None if delta is None or np.asarray(delta).size == 0 else np.asarray(delta, dtype=np.float64))
    return out


def correction_snapshot(controller, setup: CorrectionSetup, unstained: dict) -> dict:
    """
    Everything ``run_corrections`` needs, read on the main thread.
    ``unstained`` is {group: unstained sample path} (already matched).
    """
    from honeychrome.controller_components.functions import resolve_base_gate
    from honeychrome.controller_components.gls_unmix import fluor_names_and_spectra, spectra_fingerprint
    from honeychrome.controller_components.per_sample_gating_functions import SampleGating
    from honeychrome.controller_components.sample_loader import af_profile_names
    from honeychrome.controller_components.spread_model import raw_layout, spread_is_current, spread_snapshot

    process = controller.experiment.process
    names, spectra = fluor_names_and_spectra(controller)
    if spectra is None:
        raise ValueError('No fluorophore spectra: complete the spectral model first.')
    fl = [i for i, n in enumerate(names) if n != fm.AF_NAME]
    paths = default_samples(controller.experiment.samples) if setup.samples is None else list(setup.samples)
    groups = sample_groups(paths, setup.per_group)
    if not groups:
        raise ValueError('Choose at least one fully stained sample.')
    missing = [g for g in groups if not unstained.get(g)]
    if missing:
        raise ValueError(f'No unstained sample for {", ".join(missing)}: choose one.')
    gate = setup.gate
    if not gate:
        gate, _ = resolve_base_gate(controller.raw_gating, process.get('base_gate_priority_order', []))
    libraries = {}
    af_of = {}
    for key in list(dict.fromkeys(paths + [unstained[g] for g in groups])):
        profiles = (_unstained_profiles(controller, key) if key in unstained.values() and key not in paths
                    else af_profile_names(controller, key))
        lib = _af_library(controller, profiles, fl, len(names))
        af_of[key] = None if lib is None else lib['profiles']
        if lib is not None:
            libraries.setdefault(lib['profiles'], lib)
    profiles = process.get('af_profiles') or {}
    af_profiles = {n: {'spectra': np.asarray(profiles[n]['spectra'], dtype=np.float64),
                       'envelope': profiles[n].get('envelope')}
                   for lib in libraries.values() for n in lib['profiles'] if n in profiles}
    spread = None
    entry = process.get('af_spread')
    if spread_is_current(entry, names, spectra):
        spread = np.asarray(entry['spread'], dtype=np.float64)
    deltas = variant_deltas(controller, names)
    return {
        'experiment_dir': str(controller.experiment_dir), 'layout': raw_layout(controller),
        'gating': SampleGating.from_controller(controller, scope='raw'), 'gate': gate or 'root',
        'names': list(names), 'spectra': np.asarray(spectra, dtype=np.float64), 'fl': fl,
        'groups': {g: {'samples': members, 'unstained': unstained[g]} for g, members in groups.items()},
        'libraries': libraries, 'af_of': af_of, 'af_profiles': af_profiles, 'spread': spread, 'deltas': deltas,
        'n_variants': int(sum(d is not None for d in deltas)),
        'settings': fix_settings(setup), 'n_events': int(setup.n_events), 'seed': int(setup.seed),
        'per_group': bool(setup.per_group), 'base_fingerprint': spectra_fingerprint(list(names), spectra),
        'spread_snapshot': spread_snapshot(controller, list(names), np.asarray(spectra, dtype=np.float64),
                                           unstained[next(iter(groups))]),
    }


# ---------------------------------------------------------------------------
# Run (worker)
# ---------------------------------------------------------------------------

def _load(snap: dict, path: str, cap: int, loader) -> np.ndarray:
    """A sample's fluorescence events in the gate, at most ``cap`` of them (seeded per sample)."""
    from honeychrome.controller_components.sample_loader import seeded_subsample_indices

    fl, _sc = loader(snap['experiment_dir'], path, snap['gate'], snap['layout'], snap['gating'])
    fl = np.asarray(fl, dtype=np.float64)
    if len(fl) > cap:
        fl = fl[seeded_subsample_indices(len(fl), cap, path, snap['seed'])]
    return fl


def _project(snap: dict, path: str, raw: np.ndarray, spectra_fl: np.ndarray | None = None,
             libraries: dict | None = None) -> dict:
    """A sample's events unmixed with the base spectra and libraries, or the given ones."""
    spectra_fl = snap['spectra'][snap['fl']] if spectra_fl is None else spectra_fl
    libraries = snap['libraries'] if libraries is None else libraries
    key = snap['af_of'].get(path)
    lib = None if key is None else libraries[key]
    if lib is None:
        return fm.project_events(raw, spectra_fl)
    return fm.project_events(raw, spectra_fl, lib['precomputed'], lib['spectra'])


def run_group(snap: dict, group: str, progress, loader=None) -> dict:
    """Fix My Unmix on one group: its samples concatenated, its unstained sample's thresholds."""
    from honeychrome.controller_components.spread_model import load_events

    loader = loader or load_events
    entry = snap['groups'][group]
    members = entry['samples']
    say = (lambda text: progress(f'{group}: {text}')) if snap['per_group'] else progress
    s = snap['settings']
    smooth = bool(s.smooth_updates) and int(s.smooth_rounds) > 1
    say(f'Loading the unstained sample {entry["unstained"]}')
    u_raw = _load(snap, entry['unstained'], snap['n_events'], loader)
    if len(u_raw) < MIN_EVENTS:
        raise ValueError(f'Only {len(u_raw)} unstained events of {entry["unstained"]} in gate "{snap["gate"]}".')
    n_unstained = int(len(u_raw))
    if not smooth:
        unstained = _project(snap, entry['unstained'], u_raw)
        del u_raw
    share = max(1, int(np.ceil(snap['n_events'] / max(len(members), 1))))
    parts, raws, keys, counts = [], [], [], {}
    for i, path in enumerate(members, start=1):
        say(f'Loading {path} ({i}/{len(members)})')
        raw = _load(snap, path, share, loader)
        counts[path] = int(len(raw))
        if len(raw) == 0:
            continue
        if smooth:
            raws.append(raw.astype(np.float32))         # every round draws from these
        else:
            parts.append(_project(snap, path, raw))
        keys.append(path)
        del raw
    if sum(counts.values()) < MIN_EVENTS:
        raise ValueError(f'Only {sum(counts.values())} fully stained events in gate "{snap["gate"]}".')
    # libraries in order of events (the first gives the design's AF row), then the unstained sample's
    order = Counter()
    for path in keys:
        if snap['af_of'].get(path) is not None:
            order[snap['af_of'][path]] += counts[path]
    lib_keys = [k for k, _n in order.most_common()]
    u_key = snap['af_of'].get(entry['unstained'])
    if u_key is not None and u_key not in lib_keys:
        lib_keys.append(u_key)
    offsets_of, offset = {}, 0
    for k in lib_keys:
        offsets_of[k] = offset
        offset += len(snap['libraries'][k]['spectra'])
    n_det = snap['spectra'].shape[1]
    background = fm.background_for([snap['libraries'][k]['spectra'] for k in lib_keys], n_det)
    names = snap['names']
    fl_names = [names[i] for i in snap['fl']]
    spread = None if snap['spread'] is None else snap['spread'][np.ix_(snap['fl'], snap['fl'])]
    deltas = [snap['deltas'][i] for i in snap['fl']] if snap['n_variants'] else None
    run_kw = dict(spread=spread, variant_deltas=deltas, detector_names=list(snap['layout']['detectors']),
                  progress=say)
    started = time.perf_counter()
    if smooth:
        project = _smooth_projector(snap, entry['unstained'], u_raw, keys, raws, offsets_of, background,
                                    fl_names, spread)
        result = fm.smooth_fix(snap['spectra'][snap['fl']], fl_names, project, sum(len(r) for r in raws),
                               settings=s, **run_kw)
    else:
        stained = fm.concatenate(parts, [offsets_of.get(snap['af_of'].get(p), 0) for p in keys])
        unstained = fm.concatenate([unstained], [offsets_of.get(u_key, 0)])
        result = fm.fix_my_unmix(snap['spectra'][snap['fl']], fl_names, unstained, stained, background, s, **run_kw)
    logger.info('Spectral correction: %s, %d events, %.1f s', group, result.n_events,
                time.perf_counter() - started)
    return {'result': result, 'samples': members, 'events': counts, 'unstained': entry['unstained'],
            'n_unstained': n_unstained}


def _smooth_projector(snap: dict, unstained_path: str, u_raw: np.ndarray, paths: list, raws: list,
                      offsets_of: dict, background, fl_names: list, spread: np.ndarray | None):
    """
    ``project(spectra, keep)`` for ``smooth_fix``: the unstained sample and
    the stained events ``keep`` (indices into the concatenated ``raws``)
    unmixed with ``spectra``, each sample with its AF library precomputed on
    them and its envelopes rebuilt for them (the base libraries for the base
    spectra; a library's base envelope when a profile cannot be rebuilt).
    """
    from honeychrome.controller_components.autospectral_functions import build_af_precomputed

    base_fl = snap['spectra'][snap['fl']]
    sizes = [len(r) for r in raws]
    part_of = np.repeat(np.arange(len(raws)), sizes)
    starts = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)

    def libraries(spectra_fl: np.ndarray) -> dict:
        if np.array_equal(spectra_fl, base_fl):
            return snap['libraries']
        spread_entry = None if spread is None else {'fluor_names': list(fl_names), 'fluor_spectra': spectra_fl,
                                                    'spread': spread}
        out = {}
        for key, lib in snap['libraries'].items():
            base_env = lib['precomputed'].get('envelope')
            env = None
            if base_env is not None:
                env = unmix_method.rebuilt_envelope(lib['profiles'], snap['af_profiles'], fl_names, spectra_fl,
                                                    snap['experiment_dir'], snap['layout']['ceiling'], spread_entry)
                env = base_env if env is None else env
            out[key] = {**lib, 'precomputed': build_af_precomputed(spectra_fl, lib['spectra'], env)}
        return out

    def project(spectra_fl: np.ndarray, keep: np.ndarray):
        libs = libraries(spectra_fl)
        u_key = snap['af_of'].get(unstained_path)
        unstained = fm.concatenate([_project(snap, unstained_path, u_raw, spectra_fl, libs)],
                                   [offsets_of.get(u_key, 0)])
        parts, offsets = [], []
        for p, (path, raw) in enumerate(zip(paths, raws)):
            rows = keep[part_of[keep] == p] - starts[p]
            if rows.size:
                parts.append(_project(snap, path, raw[rows].astype(np.float64), spectra_fl, libs))
                offsets.append(offsets_of.get(snap['af_of'].get(path), 0))
        return unstained, fm.concatenate(parts, offsets), background

    return project


def corrected_spectra(snap: dict, result: fm.FixResult) -> np.ndarray:
    """The base spectra (every row of the panel) with a run's accepted rows."""
    out = np.array(snap['spectra'], dtype=np.float64, copy=True)
    for i in np.flatnonzero(result.accepted):
        out[snap['fl'][i]] = result.spectra[i]
    return out


def run_corrections(snap: dict, progress, loader=None) -> dict:
    """
    Every group of the snapshot, then the spread under each group's
    corrected spectra; {'groups': {group: run_group output}, 'spread_sets':
    {group: spread entry}, plus the snapshot's metadata}.
    """
    from honeychrome.controller_components.spread_model import compute_spread_entry

    out = {}
    for group in snap['groups']:
        out[group] = run_group(snap, group, progress, loader)
    sets = {g: corrected_spectra(snap, r['result']) for g, r in out.items() if r['result'].accepted.any()}
    spread_sets = {}
    spread_snap = snap.get('spread_snapshot')
    if sets and spread_snap and spread_snap.get('controls'):
        spread_sets = compute_spread_entry(spread_snap, lambda text: progress(f'Corrected spectra: {text}'),
                                           loader, projection_sets=sets)
    return {'groups': out, 'spread_sets': spread_sets, 'base_fingerprint': snap['base_fingerprint'],
            'gate': snap['gate'], 'n_events': snap['n_events'], 'per_group': snap['per_group'],
            'seed': snap['seed'], 'spread': snap['spread'] is not None, 'n_variants': snap['n_variants'],
            'settings': asdict(snap['settings'])}


# ---------------------------------------------------------------------------
# Results and storage
# ---------------------------------------------------------------------------

def result_rows(result: fm.FixResult) -> list:
    """
    One row per fluorophore phase one changed or phase two accepted:
    fluorophore, the change adopted (deg, 0 when rejected), the change phase
    one implied (deg), phase two's result and the events it dominated.
    """
    implied = fm.row_angle(result.spectra_start, result.spectra_backsolved)
    adopted = fm.row_angle(result.spectra_start, result.spectra)
    flagged = np.max(np.abs(result.spillover - np.eye(len(result.fluor_names))), axis=1) \
        >= result.settings.get('convergence_threshold', 0.01)
    rows = []
    for entry in result.signature_log:
        f = entry['index']
        if not (entry['accepted'] or flagged[f]):
            continue
        rows.append({'fluorophore': result.fluor_names[f],
                     'angle': float(adopted[f]) if entry['accepted'] else 0.0,
                     'phase_one_angle': float(implied[f]), 'result': entry['reason'],
                     'accepted': bool(entry['accepted']), 'n_events': int(entry['n_events'])})
    rows.sort(key=lambda r: (-r['accepted'], -r['angle'], -r['phase_one_angle']))
    return rows


def store_corrections(process: dict, run: dict) -> None:
    """
    Write a run as the corrections overlay (accepted rows Active) and tick
    *corrections* in ``process['unmixing']``.
    """
    groups = {}
    for group, g in run['groups'].items():
        res = g['result']
        fluors = {}
        for r in result_rows(res):
            if r['accepted']:
                f = res.fluor_names.index(r['fluorophore'])
                fluors[r['fluorophore']] = {'spectrum': [float(v) for v in res.spectra[f]], 'active': True,
                                            'angle': r['angle']}
        groups[group] = {
            'samples': list(g['samples']), 'unstained': g['unstained'], 'events': dict(g['events']),
            'n_unstained': g['n_unstained'], 'iterations': len(res.convergence_log),
            'smooth_rounds': len(res.smooth_log) if res.smooth_log else 1,
            'stop_reason': res.settings.get('stop_reason', ''),
            'n_coefficients': int(((res.trust > 0) & ~np.eye(len(res.fluor_names), dtype=bool)).sum()),
            'fluorophores': fluors, 'log': result_rows(res),
            'spread': (run.get('spread_sets') or {}).get(group),
        }
    process[STORE_KEY] = {
        'base_fingerprint': run['base_fingerprint'], 'fitted': time.strftime('%Y-%m-%d %H:%M'),
        'gate': run['gate'], 'n_events': run['n_events'], 'per_group': run['per_group'],
        'spread': run['spread'], 'n_variants': run['n_variants'], 'groups': groups,
    }
    unmix_method.set_unmixing(process, corrections=True)


def set_active(process: dict, group: str, label: str, active: bool) -> bool:
    """Tick or untick one corrected fluorophore; False when it is not stored."""
    fc = (((process.get(STORE_KEY) or {}).get('groups') or {}).get(group) or {}).get('fluorophores', {}).get(label)
    if fc is None:
        return False
    fc['active'] = bool(active)
    return True


def corrections_stale(process: dict, names: list, spectra: np.ndarray) -> bool:
    """True when stored corrections were made on other base spectra."""
    from honeychrome.controller_components.gls_unmix import spectra_fingerprint

    entry = process.get(STORE_KEY)
    if not entry:
        return False
    return entry.get('base_fingerprint') != spectra_fingerprint(list(names), spectra)


def clear_corrections(process: dict) -> None:
    process.pop(STORE_KEY, None)
    unmix_method.set_unmixing(process, corrections=False)


def clear_if_stale(process: dict, names: list, spectra: np.ndarray | None) -> bool:
    """Drop the overlay when the base spectra changed since it was made; True when dropped."""
    if spectra is None or not corrections_stale(process, names, spectra):
        return False
    clear_corrections(process)
    return True


def group_spectra(entry: dict, group: str, names: list, base: np.ndarray, active_only: bool = False) -> np.ndarray:
    """The base spectra with a stored group's corrected rows (only the Active ones with ``active_only``)."""
    corrected = np.array(base, dtype=np.float64, copy=True)
    for label, fc in (entry['groups'][group].get('fluorophores') or {}).items():
        if label in names and (fc.get('active', True) or not active_only):
            corrected[names.index(label)] = np.asarray(fc['spectrum'], dtype=np.float64)
    return corrected


def group_for_sample(entry: dict | None, sample_key) -> str | None:
    """The stored group holding a sample, or None."""
    for group, g in ((entry or {}).get('groups') or {}).items():
        if str(sample_key) in [str(s) for s in g.get('samples') or []]:
            return group
    return None
