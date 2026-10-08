"""
spectral_synthetic.py
---------------------
Synthetic single-stained controls shared by the spectral refinement tests:
Aurora reference spectra, per-cell autofluorescence that varies in shape
between two broad UV/violet components, and Poisson-like detector noise.
"""

from pathlib import Path

import numpy as np

from honeychrome.controller_components import spectral_refinement as sr

LIBRARY = (Path(__file__).resolve().parents[1] / 'src' / 'honeychrome' / 'data'
           / 'Aurora_spectral_reference_library.csv')

PANEL = ('BUV395', 'BUV496', 'BUV563', 'BUV615', 'BUV661', 'BUV737', 'BUV805',
         'BV421', 'BV480', 'BV510', 'BV605', 'BV650', 'BV711', 'BV785',
         'BB515', 'PE', 'PE-CF594', 'PE-Cy5', 'PerCP-Cy5.5', 'PE-Fire 810',
         'APC', 'Alexa Fluor 700', 'APC-Fire 750')
CONTAMINATED = 'BV510'


def panel_spectra(panel=PANEL):
    """(len(panel), 64) Aurora rows, negatives clipped, L-inf normalised,
    and the detector names."""
    rows = {}
    with open(LIBRARY) as fh:
        header = fh.readline().strip().split(',')
        for line in fh:
            parts = line.strip().split(',')
            rows[parts[0]] = np.array(parts[1:], dtype=float)
    spectra = np.clip(np.array([rows[name] for name in panel]), 0.0, None)
    return spectra / spectra.max(axis=1, keepdims=True), header[1:]


def af_shapes(n_det=64):
    """Two broad AF shapes peaking in the UV and violet detectors."""
    ch = np.arange(n_det)

    def bumps(*centres_weights):
        s = sum(w * np.exp(-0.5 * ((ch - c) / 3.0) ** 2) for c, w in centres_weights)
        return s / s.max()

    return np.array([bumps((7, 0.6), (22, 1.0), (34, 0.4)),
                     bumps((9, 1.0), (24, 0.7), (36, 0.5))])


def simulate(spectra_row, n, rng, positive_frac=0.4, brightness=3e4, af_scale=600.0, n_det=64):
    """Raw events of one single-stained cell control, or of an unstained
    control when spectra_row is None."""
    af = af_shapes(n_det)
    mix = rng.beta(2.0, 2.0, n)
    signal = rng.gamma(4.0, af_scale, n)[:, None] * (mix[:, None] * af[0] + (1 - mix[:, None]) * af[1])
    if spectra_row is not None:
        dye = rng.lognormal(np.log(brightness), 0.8, n) * (rng.random(n) < positive_frac)
        signal = signal + dye[:, None] * spectra_row
    noise_sd = np.sqrt(30.0 ** 2 + np.abs(signal))
    return signal + rng.normal(0.0, 1.0, signal.shape) * noise_sd


def contaminate(row, amount=0.04):
    """A first-pass row with residual autofluorescence left in it."""
    bad = row + amount * af_shapes(len(row)).mean(axis=0)
    return bad / bad.max()


def angle(a, b):
    return sr._angle_deg(np.asarray(a, float), np.asarray(b, float))
