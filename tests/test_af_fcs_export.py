"""
test_af_fcs_export.py
---------------------
FCS export of the per-cell AF channels (AF Abundance, AF Index) from
UnmixedExporter, run on a synthetic experiment (see af_synthetic.py) and
read back from disk.

Usage:
    pytest tests/test_af_fcs_export.py -m numpy_only
"""

import sys
from pathlib import Path

import numpy as np
import pytest

import honeychrome.settings as hc_settings
from honeychrome.controller_components import functions as fn
from honeychrome.controller_components.autospectral_functions import (
    af_index_lookup,
    apply_af_unmixing,
    precompute_af_matrices,
    precompute_joint_cov_extras,
)
from honeychrome.controller_components.unmixed_exporter import UnmixedExporter

import af_synthetic as syn

_PLUGIN_DIR = Path(__file__).resolve().parents[1] / 'src' / 'honeychrome' / 'bundled_plugins'
if str(_PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_DIR))

AF_A = hc_settings.af_abundance_channel
AF_I = hc_settings.af_index_channel


def _keyword(text, n, field):
    """$PnX from a keyword dict whose keys may or may not keep '$' / case."""
    for key in (f'$P{n}{field}', f'P{n}{field}', f'p{n}{field.lower()}', f'$p{n}{field.lower()}'):
        if key in text:
            return text[key]
    raise KeyError(f'P{n}{field}')


def _run_unmixed_exporter(controller):
    UnmixedExporter('Raw', False, None, controller).run()


# ---------------------------------------------------------------------------
# UnmixedExporter
# ---------------------------------------------------------------------------

@pytest.mark.numpy_only
def test_unmixed_export_writes_af_channels_once_with_af_unmixing_values(tmp_path):
    controller = syn.make_experiment(tmp_path)
    _run_unmixed_exporter(controller)

    pnn, events, _text = syn.read_fcs(syn.exported_path(tmp_path, syn.AF_SAMPLE))
    assert len(pnn) == len(set(pnn)), f'duplicate $PnN: {pnn}'
    assert pnn.count(AF_A) == 1 and pnn.count(AF_I) == 1

    raw_pnn, raw, _ = syn.read_fcs(tmp_path / syn.AF_SAMPLE)
    raw_fl = raw[:, [raw_pnn.index(c) for c in syn.DET_PNN]]
    fl, af = syn.fluor_spectra(), syn.af_spectra()
    pre = precompute_af_matrices(fl, af)
    pre.update(precompute_joint_cov_extras(pre, af))
    expected = apply_af_unmixing(raw_fl, pre, af)

    lookup = af_index_lookup(controller.experiment.process['af_profiles'], ['AF profile'])
    af_index = events[:, pnn.index(AF_I)]
    assert set(np.unique(af_index)) <= set(range(syn.N_OTHER_AF + 1, syn.N_AF_TOTAL + 1))
    np.testing.assert_array_equal(af_index, lookup[expected['af_idx'] - 1])
    np.testing.assert_allclose(events[:, pnn.index(AF_A)], expected['af_scale'], rtol=1e-5, atol=1e-2)
    fl_cols = [pnn.index(c) for c in syn.FLUOR_LABELS]
    np.testing.assert_allclose(events[:, fl_cols], expected['unmixed'], rtol=1e-5, atol=1e-2)


@pytest.mark.numpy_only
def test_unmixed_export_without_af_profile_omits_af_channels(tmp_path):
    controller = syn.make_experiment(tmp_path)
    _run_unmixed_exporter(controller)

    pnn, events, _text = syn.read_fcs(syn.exported_path(tmp_path, syn.PLAIN_SAMPLE))
    assert AF_A not in pnn and AF_I not in pnn
    assert pnn == [c for c in syn.UNMIXED_PNN if c not in hc_settings.af_channels]

    raw_pnn, raw, _ = syn.read_fcs(tmp_path / syn.PLAIN_SAMPLE)
    raw_fl = raw[:, [raw_pnn.index(c) for c in syn.DET_PNN]]
    unmixing = np.array(controller.experiment.process['unmixing_matrix'])
    fl_cols = [pnn.index(c) for c in syn.FLUOR_LABELS]
    np.testing.assert_allclose(events[:, fl_cols], raw_fl @ unmixing.T, rtol=1e-5, atol=1e-2)


@pytest.mark.numpy_only
def test_af_index_range_covers_every_library_index(tmp_path):
    """AF Index values run 1..N over the whole experiment, so $PnR must
    exceed N for the top index to sit inside the declared range."""
    controller = syn.make_experiment(tmp_path)
    _run_unmixed_exporter(controller)

    pnn, events, text = syn.read_fcs(syn.exported_path(tmp_path, syn.AF_SAMPLE))
    n = pnn.index(AF_I) + 1
    assert float(_keyword(text, n, 'R')) == syn.N_AF_TOTAL + 1
    assert events[:, n - 1].max() < float(_keyword(text, n, 'R'))


# ---------------------------------------------------------------------------
# Keyword construction
# ---------------------------------------------------------------------------

@pytest.mark.numpy_only
def test_define_fcs_keywords_af_channels_and_spillover_names():
    af = syn.af_spectra()
    pnn = list(syn.UNMIXED_PNN)
    data = np.zeros((5, len(pnn)))
    kw = fn.define_fcs_keywords(
        raw_keywords={'$P1N': 'Time', '$P1R': '100', '$CYT': 'Synthetic'},
        pnn=pnn, event_data=data,
        spectral_model=[{'label': c, 'antigen': ''} for c in syn.FLUOR_LABELS],
        unmixed_settings=syn.unmixed_settings(), raw_settings=syn.raw_settings(),
        spillover=np.eye(syn.N_FLUOR), af_spectra=af,
        unmixing_spectra=syn.fluor_spectra(), file_name='x.fcs', version='test',
    )
    names = [kw[f'$P{i}N'] for i in range(1, len(pnn) + 1)]
    assert names == pnn
    assert kw['$SPILLOVER'].split(',')[1:1 + syn.N_FLUOR] == syn.FLUOR_LABELS
    assert kw['AUTOFLUORESCENCE'].startswith(f'{syn.N_AF},{syn.N_DET},')
    i_af = pnn.index(AF_A) + 1
    assert kw[f'$P{i_af}DISPLAY'] == 'LOG'


@pytest.mark.numpy_only
def test_autofluorescence_keyword_rows_follow_the_af_index():
    """AUTOFLUORESCENCE rows are named by their AF Index value and written
    in AF Index order, so an exported AF Index can be matched to its row."""
    af = syn.af_spectra()
    index_map = np.array([7, 5, 4, 6])
    pnn = list(syn.UNMIXED_PNN)
    kw = fn.define_fcs_keywords(
        raw_keywords={'$CYT': 'Synthetic'}, pnn=pnn, event_data=np.zeros((1, len(pnn))),
        spectral_model=[], unmixed_settings=syn.unmixed_settings(), raw_settings=syn.raw_settings(),
        spillover=np.eye(syn.N_FLUOR), af_spectra=af, unmixing_spectra=None,
        file_name='x.fcs', version='test', af_index_map=index_map, n_af_index=7,
    )
    fields = kw['AUTOFLUORESCENCE'].split(',')
    assert fields[2:2 + syn.N_AF] == ['AF4', 'AF5', 'AF6', 'AF7']
    values = np.array(fields[2 + syn.N_AF + syn.N_DET:], dtype=float).reshape(syn.N_AF, syn.N_DET)
    np.testing.assert_allclose(values, af[[2, 1, 3, 0]], rtol=1e-7)
    assert kw[f'$P{pnn.index(AF_I) + 1}R'] == '8'
