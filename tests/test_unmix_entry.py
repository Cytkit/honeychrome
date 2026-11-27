"""
The display (Controller._apply_unmixing), unmix_events (statistics, plugins)
and the FCS export all unmix through unmix_method.unmix, also when no
unmixing method is set, so they cannot drift apart.
"""

from dataclasses import replace

import numpy as np
import pytest

from honeychrome.controller import Controller
from honeychrome.controller_components import af_posterior, unmix_method
from honeychrome.controller_components import sample_loader as sl
from honeychrome.controller_components.autospectral_functions import build_af_precomputed
from honeychrome.controller_components.unmixed_exporter import UnmixedExporter

import af_synthetic as syn

pytestmark = pytest.mark.filterwarnings('ignore:No events bus connected')

PROFILE = 'AF profile'
SPILLOVER = np.eye(syn.N_FLUOR) + 0.05 * np.eye(syn.N_FLUOR, k=1)
CASES = ['plain', 'af', 'af_noise_aware']


def _transfer_matrix():
    """raw @ T = unmixed for af_synthetic's channels, fine-tuned with SPILLOVER."""
    fl = syn.fluor_spectra()
    unmixing = np.linalg.solve(fl @ fl.T, fl)
    t = np.zeros((len(syn.UNMIXED_PNN), len(syn.RAW_PNN)))
    t[0, 0] = t[2, 1] = t[3, 2] = 1
    t[np.ix_(range(4, 4 + syn.N_FLUOR), range(3, 3 + syn.N_DET))] = np.linalg.inv(SPILLOVER).T @ unmixing
    return t.T


def _controller(sample_key):
    """A Controller showing *sample_key* of af_synthetic's experiment; AF_SAMPLE has an AF profile."""
    controller = Controller()
    controller.experiment.settings['raw'].update(syn.raw_settings())
    controller.experiment.settings['unmixed'].update(syn.unmixed_settings())
    controller.experiment.process.update({'spillover': SPILLOVER.tolist(),
                                          'af_profiles': {PROFILE: {'spectra': syn.af_spectra().tolist()}}})
    controller.experiment.samples['sample_af_profiles'] = {syn.AF_SAMPLE: [PROFILE]}
    controller.transfer_matrix = _transfer_matrix()
    controller.filtered_raw_fluorescence_channel_ids = list(range(3, 3 + syn.N_DET))
    controller.af_precomputed_cache = {PROFILE: build_af_precomputed(syn.fluor_spectra(), syn.af_spectra())}
    controller.current_sample_path = sample_key
    af = sl.snapshot_unmix_state(controller, [sample_key], include_qc=False).af_for(sample_key)
    controller.af_precomputed = None if af is None else af.precomputed
    controller.af_spectra = None if af is None else af.spectra
    controller.af_index_map = None if af is None else af.index_map
    return controller


class FixedPosterior:
    """Stands in for af_posterior.AFNoiseContext: one posterior for every AF library."""

    def __init__(self):
        prior = af_posterior.AFPrior(intercept=600.0, slope=0.0, tau2=200.0 ** 2, delta=900.0,
                                     tau2_broad=1500.0 ** 2, pi_broad=0.1, choice_shrinkage=1.0,
                                     k_typical=600.0, log_ssc_median=11.0, n_events=10_000,
                                     fingerprint='f', source='s')
        self.posterior = af_posterior.AFPosterior(prior, np.full(syn.N_DET, 400.0), np.full(syn.N_DET, 0.8),
                                                  np.zeros(syn.N_DET), ssc_name='SSC-A')

    def attach(self, af, sample_key, live=False):
        return None if af is None else replace(af, posterior=self.posterior)


@pytest.fixture(params=CASES)
def sample_key(request, monkeypatch):
    """The displayed sample: without AF, with AF, or with AF and the noise-aware AF abundance."""
    if request.param == 'af_noise_aware':
        context = FixedPosterior()
        monkeypatch.setattr(af_posterior, 'noise_context', lambda *args, **kwargs: context)
    return syn.PLAIN_SAMPLE if request.param == 'plain' else syn.AF_SAMPLE


@pytest.fixture
def unmix_calls(monkeypatch):
    """Sample keys of the calls to unmix_method.unmix."""
    calls = []
    original = unmix_method.unmix

    def spy(*args, **kwargs):
        calls.append(kwargs.get('sample_key', args[7] if len(args) > 7 else None))
        return original(*args, **kwargs)

    monkeypatch.setattr(unmix_method, 'unmix', spy)
    return calls


def test_display_and_unmix_events_give_the_same_values(sample_key):
    controller = _controller(sample_key)
    snap = sl.snapshot_unmix_state(controller, [sample_key], include_qc=False)
    raw = syn.raw_events(2000, seed=4)

    shown = controller._apply_unmixing(raw)
    loaded = sl.unmix_events(raw, snap, snap.af_for(sample_key), return_af=True, sample_key=sample_key)

    np.testing.assert_array_equal(shown, loaded['unmixed'])
    if snap.af_for(sample_key) is None:
        assert controller.af_sidecar_data is None
    else:
        np.testing.assert_array_equal(controller.af_sidecar_data,
                                      np.column_stack([loaded['af_scale'], loaded['af_idx']]))


def test_display_unmixes_through_unmix_method(sample_key, unmix_calls):
    _controller(sample_key)._apply_unmixing(syn.raw_events(500, seed=4))

    assert len(unmix_calls) == 1


def test_unmix_events_unmixes_through_unmix_method(sample_key, unmix_calls):
    snap = sl.snapshot_unmix_state(_controller(sample_key), [sample_key], include_qc=False)

    sl.unmix_events(syn.raw_events(500, seed=4), snap, snap.af_for(sample_key), sample_key=sample_key)

    assert unmix_calls == [sample_key]


@pytest.mark.numpy_only
def test_export_unmixes_through_unmix_method(tmp_path, unmix_calls):
    controller = syn.make_experiment(tmp_path, n_events=500)

    UnmixedExporter('Raw', False, None, controller).run()

    assert set(unmix_calls) == {syn.AF_SAMPLE, syn.PLAIN_SAMPLE}
