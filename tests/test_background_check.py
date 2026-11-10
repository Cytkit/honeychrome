"""Background check of cleaned controls (spectral_cleaning.background_check)."""
import numpy as np

from honeychrome.controller_components.spectral_cleaning import (
    background_check, cosine_filter, knn_scatter_match,
)
from spectral_synthetic import panel_spectra, simulate

N_CANDIDATES, N_SPECTRAL = 1000, 200


def _clean_and_check(pos, neg, rng):
    """The external-negative cleaning path of SpectralCleaner._clean_one, then the check."""
    pos_scatter = rng.normal(size=(len(pos), 2))
    neg_scatter = rng.normal(size=(len(neg), 2))
    af_mean = neg.mean(axis=0)
    af_unit = af_mean / np.linalg.norm(af_mean)
    peak = int(np.argmax((pos - np.outer(pos @ af_unit, af_unit)).mean(axis=0)))
    sel, _ = cosine_filter(pos, np.median(neg, axis=0), peak, N_CANDIDATES, N_SPECTRAL)
    sub, _ = knn_scatter_match(pos[sel], pos_scatter[sel], neg, neg_scatter)
    spectrum = np.median(sub, axis=0).clip(0, None)
    return background_check(pos, neg, neg_scatter, sub, spectrum / spectrum.max(), peak,
                            N_CANDIDATES, N_SPECTRAL)


def test_stained_control_passes():
    rng = np.random.default_rng(1)
    spectra, _ = panel_spectra(('BV421',))
    check = _clean_and_check(simulate(spectra[0], 20_000, rng), simulate(None, 20_000, rng), rng)
    assert check['verdict'] is None
    assert check['ratio'] > 2


def test_unstained_as_control_is_background():
    rng = np.random.default_rng(2)
    check = _clean_and_check(simulate(None, 20_000, rng), simulate(None, 20_000, rng), rng)
    assert check['verdict'] == 'background'
    assert check['ratio'] < 2
    assert check['angle_deg'] < 15
