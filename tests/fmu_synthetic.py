"""
fmu_synthetic.py
----------------
Synthetic panel for the Fix My Unmix tests: Gaussian-peak spectra over 40
detectors, two autofluorescent cell types (whose spectra, after their
population mean, are the AF library), photon and read noise, and a known
error injected into the spectra handed to the method.
"""

import numpy as np

from honeychrome.controller_components import fix_my_unmix as fm
from honeychrome.controller_components.autospectral_functions import (
    build_af_precomputed, build_runtime_envelope, make_envelope_entry,
)

CEILING = 4_194_304.0
N_DET = 40


def panel(n_fluor: int = 8):
    """L-inf normalised spectra with overlapping peaks and tandem shoulders."""
    x = np.arange(N_DET, dtype=float)
    rows = []
    for i in range(n_fluor):
        centre = 3 + i * (N_DET - 6) / max(n_fluor - 1, 1)
        row = np.exp(-0.5 * ((x - centre) / 1.5) ** 2)
        if i % 3 == 2:
            row += 0.35 * np.exp(-0.5 * ((x - centre + 9) / 2.5) ** 2)
        rows.append(row / row.max())
    return np.array(rows), [f'Dye {i}' for i in range(n_fluor)]


def af_shapes() -> np.ndarray:
    x = np.arange(N_DET, dtype=float)
    af1 = np.exp(-0.5 * ((x - 6) / 5) ** 2) + 0.5 * np.exp(-0.5 * ((x - 18) / 6) ** 2)
    af2 = np.exp(-0.5 * ((x - 14) / 6) ** 2) + 0.3 * np.exp(-0.5 * ((x - 30) / 5) ** 2)
    return np.vstack([af1 / af1.max(), af2 / af2.max()])


def af_cells(n: int, rng) -> np.ndarray:
    """Raw unstained events: two AF cell types, photon and read noise."""
    af1, af2 = af_shapes()
    kind = rng.random(n) < 0.7
    level = np.where(kind, rng.lognormal(np.log(2500), 0.4, n), rng.lognormal(np.log(12000), 0.4, n))
    raw = level[:, None] * np.where(kind[:, None], af1, af2)
    return raw + rng.normal(size=raw.shape) * np.sqrt(np.maximum(raw, 0.0)) + rng.normal(0.0, 50.0, raw.shape)


def af_library() -> np.ndarray:
    """The population mean, then each cell type."""
    shapes = af_shapes()
    mean = 0.7 * 2500 * shapes[0] + 0.3 * 12000 * shapes[1]
    return np.vstack([mean / mean.max(), shapes])


def inject(truth: np.ndarray, names: list, injections) -> np.ndarray:
    """Spectra with ``target += eps * source`` for each (target, source, eps), L-inf normalised."""
    wrong = truth.copy()
    for target, source, eps in injections:
        t = names.index(target)
        row = wrong[t] + eps * truth[names.index(source)]
        wrong[t] = row / row.max()
    return wrong


def stained_events(n: int, truth: np.ndarray, rng, positive: float = 0.25) -> tuple:
    """Fully stained raw events (each dye positive in a share of cells) on AF cells, and their abundances."""
    n_fluor = truth.shape[0]
    on = rng.random((n, n_fluor)) < positive
    abundance = np.where(on, rng.lognormal(np.log(20_000), 0.8, (n, n_fluor)), 0.0)
    signal = abundance @ truth
    raw = af_cells(n, rng) + signal + rng.normal(size=signal.shape) * np.sqrt(signal)
    return raw, abundance


def experiment(n: int = 20_000, injections=(('Dye 4', 'Dye 3', 0.10),), seed: int = 3):
    rng = np.random.default_rng(seed)
    truth, names = panel()
    wrong = inject(truth, names, injections)
    unstained = af_cells(n, rng)
    stained, abundance = stained_events(n, truth, rng)
    return truth, wrong, names, unstained, stained, abundance


def library_for(spectra: np.ndarray, names: list, unstained: np.ndarray, envelope: bool = True) -> dict:
    """The synthetic AF library precomputed on ``spectra``, with a negative envelope built from ``unstained``."""
    lib = af_library()
    env = None
    if envelope:
        entry, _sample = make_envelope_entry(unstained, names, spectra, lib)
        env = build_runtime_envelope(entry, None, names, spectra, CEILING)
    return {'spectra': lib, 'precomputed': build_af_precomputed(spectra, lib, env)}


def projections(spectra: np.ndarray, names: list, unstained: np.ndarray, stained: np.ndarray,
                envelope: bool = True) -> tuple:
    """(unstained projection, stained projection, background) with the synthetic AF library."""
    lib = library_for(spectra, names, unstained, envelope)
    pu = fm.project_events(unstained, spectra, lib['precomputed'], lib['spectra'])
    ps = fm.project_events(stained, spectra, lib['precomputed'], lib['spectra'])
    return pu, ps, fm.background_for([lib['spectra']], spectra.shape[1])


def run(spectra, names, unstained, stained, settings=None, truth=None, **kw):
    """fix_my_unmix with the synthetic AF library (per-cell AF with an envelope)."""
    pu, ps, bg = projections(spectra, names, unstained, stained)
    return fm.fix_my_unmix(spectra, names, pu, ps, bg, settings or fm.FixSettings(), true_spectra=truth, **kw)


def coexpressed_experiment(n: int = 20_000, partners=(0, 6), target: int = 3, k: float = 0.04, seed: int = 3):
    """
    Correct spectra, but the cells positive for each ``partners`` dye also
    carry ``target`` at ``k`` x their partner's abundance (a dim marker on
    the same cells). Returns truth, names, unstained, stained.
    """
    rng = np.random.default_rng(seed)
    truth, names = panel()
    unstained = af_cells(n, rng)
    n_fluor = truth.shape[0]
    on = rng.random((n, n_fluor)) < 0.25
    abundance = np.where(on, rng.lognormal(np.log(20_000), 0.8, (n, n_fluor)), 0.0)
    extra = sum(k * abundance[:, p] * rng.lognormal(0.0, 0.2, n) for p in partners)
    abundance[:, target] = np.where(abundance[:, target] > 0, abundance[:, target], extra)
    signal = abundance @ truth
    stained = af_cells(n, rng) + signal + rng.normal(size=signal.shape) * np.sqrt(signal)
    return truth, names, unstained, stained
