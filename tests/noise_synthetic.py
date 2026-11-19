"""
noise_synthetic.py
------------------
Simulated instrument and experiment for the noise model and M2pg tests: smooth
synthetic spectra, per-detector sigma0 and kappa, an AF spectrum with shape
modes, per-fluorophore variant modes, and an experiment of an unstained cell
sample, cell controls and bead controls with internal negatives (the GLS Test
plugin's simulation).
"""

import numpy as np

from honeychrome.controller_components import gls_unmix as gu
from honeychrome.controller_components import noise_model as nm


def _bump(d, centre, width):
    det = np.arange(d)
    return np.exp(-0.5 * ((det - centre) / width) ** 2)


def simulated_panel(n_fluor: int = 12, d: int = 48, seed: int = 0) -> dict:
    """
    Smooth synthetic spectra (one or two emission bumps each, L-inf
    normalised), per-detector kappa around 5e-3 and sigma0 around 150, an AF
    spectrum with two shape modes, and two smooth variant modes per
    fluorophore at 3% of the peak.
    """
    rng = np.random.default_rng(seed)
    centres = np.linspace(3, d - 4, n_fluor)
    spectra = []
    for i, c in enumerate(centres):
        s = _bump(d, c, rng.uniform(1.5, 3.0))
        if i % 3 == 1:
            s = s + 0.3 * _bump(d, min(d - 1, c + rng.uniform(6, 12)), 2.5)
        s = s + 0.004
        spectra.append(s / s.max())
    spectra = np.array(spectra)
    af0 = _bump(d, 0.3 * d, 0.15 * d) + 0.4 * _bump(d, 0.55 * d, 0.1 * d) + 0.02
    af0 /= af0.max()
    det = np.arange(d)
    af_modes = np.vstack([np.sin(det / 5.0) * af0, np.cos(det / 7.0) * af0]) * 0.08
    modes = []
    for f in range(n_fluor):
        on = np.flatnonzero(spectra[f] > 0.05)
        m = np.vstack([_bump(d, rng.choice(on), rng.uniform(1.5, 4)) * rng.choice([-1, 1]) for _ in range(2)])
        modes.append(m * 0.03)
    return {
        'spectra': spectra, 'names': [f'F{i}' for i in range(n_fluor)],
        'detectors': [f'D{i}-A' for i in range(d)],
        'kappa': 5e-3 * np.exp(rng.normal(0, 0.5, d)), 'sigma0': 150 * np.exp(rng.normal(0, 0.4, d)),
        'af0': af0, 'af_modes': af_modes, 'var_modes': modes, 'rng': rng,
    }


def simulate_events(panel: dict, x: np.ndarray, af_abund: np.ndarray, offset=None, variants: bool = True,
                    af_shape: bool = True) -> np.ndarray:
    rng = panel['rng']
    s = panel['spectra']
    mu = x @ s + af_abund[:, None] * panel['af0']
    if af_shape:
        mu += (af_abund[:, None] * rng.standard_normal((len(x), 2))) @ panel['af_modes']
    if variants:
        for f, m in enumerate(panel['var_modes']):
            on = x[:, f] > 0
            if on.any():
                mu[on] += (x[on, f, None] * rng.standard_normal((on.sum(), m.shape[0]))) @ m
    if offset is not None:
        mu = mu + offset
    var = panel['sigma0'] ** 2 + np.maximum(mu, 0) / panel['kappa']
    return mu + rng.standard_normal(mu.shape) * np.sqrt(var)


def simulated_experiment(n_cell_controls: int = 8, n_bead_controls: int = 4, n_events: int = 12_000,
                         seed: int = 0) -> tuple:
    """
    ``(panel, samples, snap)``: an unstained cell sample, cell controls with
    40% unstained events, and bead controls with a 30% internal negative and
    a constant bead background. ``samples`` maps a path to ``(fl, sc)``;
    ``snap`` is a snapshot for ``gls_unmix.fit_experiment_noise_model`` with
    ``loader_for(samples)``.
    """
    n_fluor = n_cell_controls + n_bead_controls
    panel = simulated_panel(n_fluor, seed=seed)
    rng = panel['rng']
    spectra = panel['spectra']
    d = spectra.shape[1]
    bead_bg = 40.0 + 20.0 * rng.random(d)

    def cells(n, f=None):
        af = np.exp(rng.normal(np.log(3e3), 0.5, n))
        x = np.zeros((n, n_fluor))
        if f is not None:
            pos = rng.random(n) < 0.6
            x[pos, f] = np.exp(rng.normal(np.log(3e4), 1.0, pos.sum()))
        fl = simulate_events(panel, x, af)
        sc = np.column_stack([rng.normal(1e5, 1e4, n), 2e4 + 3.0 * af + rng.normal(0, 2e3, n)])
        return fl, sc

    def beads(n, f):
        x = np.zeros((n, n_fluor))
        pos = rng.random(n) < 0.7
        x[pos, f] = np.exp(rng.normal(np.log(8e4), 0.2, pos.sum()))
        fl = simulate_events(panel, x, np.zeros(n), offset=bead_bg, af_shape=False)
        sc = np.column_stack([rng.normal(5e4, 2e3, n), rng.normal(2e4, 1e3, n)])
        return fl, sc

    samples = {'Unstained.fcs': cells(3 * n_events)}
    records = []
    for f in range(n_fluor):
        is_cell = f < n_cell_controls
        path = f'{panel["names"][f]} {"Cells" if is_cell else "Beads"}.fcs'
        samples[path] = cells(n_events, f) if is_cell else beads(n_events, f)
        records.append({'label': panel['names'][f], 'fluor_idx': f, 'path': path,
                        'neg_path': 'Unstained.fcs' if is_cell else None,
                        'peak': int(np.argmax(spectra[f])), 'is_cell': is_cell})
    layout = {'fl_ids': list(range(d)), 'sc_ids': [d, d + 1], 'col_order': None,
              'detectors': panel['detectors'], 'ceiling': float(2 ** 22), 'n_scatter': 2}
    settings = gu.NoiseSetupSettings(max_test_unstained=8_000, max_test_control=3_000, max_fit_events=4_000,
                                     max_cal_events=1_500)
    snap = {
        'experiment_dir': '/tmp', 'unstained_candidates': ['Unstained.fcs'], 'gate': 'root', 'base_gate': 'root',
        'layout': layout, 'gating': None, 'fluor_names': panel['names'], 'spectra': spectra,
        'settings': settings, 'controls': records, 'bead_unstained': [], 'stored_af': {'Unstained.fcs': None},
        'stored_sidecar': {}, 'variant_cache': None, 'scatter_columns': (0, 1, None),
        'lasers': [f'L{j * 4 // d}' for j in range(d)],
    }
    return panel, samples, snap


def loader_for(samples: dict):
    def loader(_experiment_dir, path, gate, _layout, _gating):
        fl, sc = samples[path]
        if gate == 'Neg Unstained' and path != 'Unstained.fcs':
            return fl[:0], sc[:0]
        return fl, sc
    return loader


def simulated_groups(panel: dict, n: int, cells_every: int = 2) -> list:
    """Noise-model groups with known designs: one per fluorophore (cells or beads) and an unstained one."""
    rng = panel['rng']
    s = panel['spectra']
    f_n = s.shape[0]
    afb = np.vstack([panel['af0'], panel['af_modes']])
    groups = []
    for f in range(f_n):
        cell = f % cells_every == 0
        x = np.zeros((n, f_n))
        x[:, f] = np.exp(rng.uniform(np.log(1e3), np.log(2e5), n))
        af = np.exp(rng.normal(np.log(3e3), 0.5, n)) if cell else np.zeros(n)
        y = simulate_events(panel, x, af, af_shape=cell)
        design = np.vstack([s[f], panel['var_modes'][f]] + ([afb] if cell else []))
        groups.append(nm.Group(f'F{f}', 'control', y, design))
    af = np.exp(rng.normal(np.log(3e3), 0.5, 3 * n))
    groups.append(nm.Group('Unstained', 'unstained', simulate_events(panel, np.zeros((3 * n, f_n)), af), afb))
    return groups


# ---------------------------------------------------------------------------
# Cases for the estimator options and the automatic selection
# ---------------------------------------------------------------------------

def laser_blocks(d: int, n_lasers: int = 4) -> list:
    """A laser label per detector: ``n_lasers`` contiguous blocks."""
    return [f'L{j * n_lasers // d}' for j in range(d)]


def option_groups(panel: dict, n: int, bead_af_cv: float = 0.0, laser_jitter: float = 0.0,
                  control_spread=None, sigma0_cells_ratio: float = 1.0, bead_af_rows: int = 0,
                  laser_split: bool = False, n_lasers: int = 4, background=None, seed: int = 1) -> tuple:
    """
    ``(fit groups, calibration groups, lasers)`` with known designs: cell
    controls (AF rows of the unstained cells), bead controls with a bead
    background, an unstained cell group and a bead negative. Options:

    * ``bead_af_cv``: the bead background is a bright AF spectrum (UV/violet
      end, peak 8,000) whose abundance varies with that CV per bead;
    * ``laser_jitter``: a per-event relative gain per laser block;
    * ``control_spread``: per-control relative SD of off-design spread
      (variance ``(rel mu)^2`` per detector);
    * ``sigma0_cells_ratio``: cells' sigma0 is that times the beads' one;
    * ``background`` (D,): a constant signal in every event (shot noise
      included), carried in every group's offset, e.g. a detector with no
      dark events.

    ``bead_af_rows`` and ``laser_split`` set the designs as gls_unmix does
    (``af_basis``, ``control_design``).
    """
    rng = np.random.default_rng(seed)
    s = panel['spectra']
    f_n, d = s.shape
    lasers = laser_blocks(d, n_lasers)
    las = np.array(lasers)
    det = np.arange(d)
    bead_af = 8000.0 * np.exp(-0.5 * ((det - 0.12 * d) / (0.08 * d)) ** 2) + 30.0
    afb = np.vstack([panel['af0'], panel['af_modes']])

    def events(x, af_abund, cell, spread=0.0, bead_scale=None):
        mu = x @ s + af_abund[:, None] * panel['af0']
        if cell:
            mu += (af_abund[:, None] * rng.standard_normal((len(x), 2))) @ panel['af_modes']
        else:
            scale = np.ones(len(x)) if bead_scale is None else bead_scale
            mu += scale[:, None] * bead_af
        if laser_jitter:
            gain = np.ones_like(mu)
            for laser in np.unique(las):
                gain[:, las == laser] *= 1.0 + laser_jitter * rng.standard_normal((len(x), 1))
            mu *= gain
        if spread:
            mu += spread * mu * rng.standard_normal(mu.shape)
        if background is not None:
            mu = mu + background
        sigma0 = panel['sigma0'] * (sigma0_cells_ratio if cell else 1.0)
        var = sigma0 ** 2 + np.maximum(mu, 0) / panel['kappa']
        return mu + rng.standard_normal(mu.shape) * np.sqrt(var)

    def bead_scale(m):
        return np.maximum(1.0 + bead_af_cv * rng.standard_normal(m), 0.05)

    bg = np.zeros(d) if background is None else np.asarray(background, dtype=np.float64)

    def off(x):
        return x if background is None else (bg if x is None else x + bg)

    def build(m):
        neg = events(np.zeros((m, f_n)), np.zeros(m), False, bead_scale=bead_scale(m))
        groups = []
        neg_rows = gu.af_basis(neg, bead_af_rows, seed=0) if bead_af_rows else None
        for f in range(f_n):
            cell = f % 2 == 0
            x = np.zeros((m, f_n))
            x[:, f] = np.exp(rng.uniform(np.log(1e3), np.log(2e5), m))
            spread = 0.0 if control_spread is None else control_spread[f]
            if cell:
                y = events(x, np.exp(rng.normal(np.log(3e3), 0.5, m)), True, spread)
                af_rows, offset = afb, None
            else:
                y = events(x, np.zeros(m), False, spread, bead_scale(m))
                af_rows, offset = (neg_rows, None) if bead_af_rows else (None, bead_af + 0.0)
            vb = gu.VariantBasis(f'F{f}', np.zeros((0, d)), np.zeros(0), np.zeros(d))
            design = gu.control_design(s[f], vb, af_rows, lasers=lasers if laser_split else None)
            groups.append(nm.Group(f'F{f}', 'control', y, design, off(offset), cls='cells' if cell else 'beads'))
        unst = events(np.zeros((3 * m, f_n)), np.exp(rng.normal(np.log(3e3), 0.5, 3 * m)), True)
        groups.append(nm.Group('Unstained', 'unstained', unst, afb, off(None), cls='cells'))
        groups.append(nm.Group('Beads', 'beads', neg, neg_rows, off(None), cls='beads'))
        return groups

    return build(n), build(max(n // 2, 500)), lasers


def scatter_population(n_singlets: int = 20_000, n_doublets: int = 2_000, n_debris: int = 3_000,
                       saturated_frac: float = 0.0, ceiling: float = 4_194_304.0, seed: int = 3) -> tuple:
    """
    ``(fsc_a, ssc_a, fsc_h, kind)`` of simulated events: singlets, doublets
    (twice the area, 1.65 times the pulse width) and debris (low, broad
    FSC/SSC). With ``saturated_frac`` the singlets' SSC is centred so that
    that share of them lies above the ceiling, and SSC is clipped there (as
    bright beads on some cytometers). ``kind`` is 0 singlet, 1 doublet, 2
    debris.
    """
    from scipy.stats import norm

    rng = np.random.default_rng(seed)
    ssc_sd = 0.18
    ssc_median = 4e4 if saturated_frac <= 0 else ceiling * np.exp(-norm.ppf(1.0 - saturated_frac) * ssc_sd)
    fsc = np.exp(rng.normal(np.log(1.5e5), 0.12, n_singlets))
    ssc = np.exp(rng.normal(np.log(ssc_median), ssc_sd, n_singlets))
    width = np.exp(rng.normal(np.log(1.25), 0.03, n_singlets))
    dfsc = np.exp(rng.normal(np.log(1.5e5), 0.12, n_doublets)) * 2.0
    dssc = np.exp(rng.normal(np.log(ssc_median), ssc_sd, n_doublets)) * 2.0
    dwidth = np.exp(rng.normal(np.log(1.25 * 1.65), 0.05, n_doublets))
    rfsc = np.exp(rng.normal(np.log(2e4), 0.8, n_debris))
    rssc = np.exp(rng.normal(np.log(5e3), 0.8, n_debris))
    rwidth = np.exp(rng.normal(np.log(1.25), 0.2, n_debris))
    fsc_a = np.concatenate([fsc, dfsc, rfsc])
    ssc_a = np.minimum(np.concatenate([ssc, dssc, rssc]), ceiling)
    fsc_h = fsc_a / np.concatenate([width, dwidth, rwidth])
    kind = np.concatenate([np.zeros(n_singlets), np.ones(n_doublets), np.full(n_debris, 2)]).astype(int)
    return fsc_a, ssc_a, fsc_h, kind
