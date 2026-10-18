"""Cells or beads from scatter (controller_components/particle_type.py)."""
import numpy as np
import pytest

import honeychrome.controller_components.particle_type as pt
from honeychrome.controller_components.particle_type import (
    BEAD_CORE_LOG_AREA, BEAD_ANCHOR_MARGIN,
    scatter_core_log_area, classify_particle_type, scatter_channel_pair, scatter_ceilings,
    is_bead_sample, set_bead_sample, update_bead_samples, match_bead_unstained,
)

CEILING = 262144.0


def _log_normal(rng, n, centre, sd):
    return 10 ** rng.normal(centre, sd, size=(n, 2))


def _beads(rng, n=5000):
    """Singlets (CV ~4 %) with 10 % doublets and 10 % debris."""
    singlets = _log_normal(rng, int(n * 0.8), (4.6, 4.8), (0.015, 0.02))
    doublets = _log_normal(rng, int(n * 0.1), (4.9, 5.1), (0.02, 0.03))
    debris = _log_normal(rng, n - len(singlets) - len(doublets), (3.8, 3.9), (0.3, 0.3))
    return np.vstack([singlets, doublets, debris])


def _cells(rng, n=5000):
    """Lymphocytes, myeloid cells and debris."""
    lymph = _log_normal(rng, int(n * 0.6), (4.85, 4.3), (0.06, 0.12))
    myeloid = _log_normal(rng, int(n * 0.25), (5.0, 4.9), (0.08, 0.15))
    debris = _log_normal(rng, n - len(lymph) - len(myeloid), (4.0, 3.9), (0.25, 0.3))
    return np.vstack([lymph, myeloid, debris])


def test_beads_tighter_than_cells():
    rng = np.random.default_rng(1)
    beads = scatter_core_log_area(_beads(rng), CEILING)
    cells = scatter_core_log_area(_cells(rng), CEILING)
    assert beads < BEAD_CORE_LOG_AREA < cells
    assert cells - beads > 1.0


def test_area_independent_of_gain():
    rng = np.random.default_rng(2)
    events = _cells(rng)
    assert scatter_core_log_area(events, 1e12) == pytest.approx(
        scatter_core_log_area(events * 3.0, 1e12), abs=1e-9)


def test_saturated_and_non_positive_events_dropped():
    rng = np.random.default_rng(3)
    beads = _beads(rng)
    piled = np.vstack([beads, np.full((3000, 2), CEILING), np.zeros((500, 2))])
    assert scatter_core_log_area(piled, CEILING) == pytest.approx(
        scatter_core_log_area(beads, CEILING), abs=0.05)


def test_per_channel_ceilings():
    rng = np.random.default_rng(4)
    events = _cells(rng)
    assert scatter_core_log_area(events, [10.0, CEILING]) is None
    assert scatter_core_log_area(events, [CEILING, CEILING]) is not None


def test_too_few_events_gives_no_call():
    rng = np.random.default_rng(5)
    assert scatter_core_log_area(_beads(rng, 100), CEILING) is None
    assert classify_particle_type(None) is None


def test_classify_without_anchor():
    assert classify_particle_type(BEAD_CORE_LOG_AREA - 0.01) == 'Beads'
    assert classify_particle_type(BEAD_CORE_LOG_AREA + 0.01) == 'Cells'


def test_anchor_keeps_tight_cells_as_cells():
    anchor = -3.0                               # bead-like anchor is ignored
    assert classify_particle_type(-3.1, anchor) == 'Beads'
    anchor = -2.75                              # cell anchor
    assert classify_particle_type(-3.1, anchor) == 'Cells'
    # Slingshot beads (-3.15) against cell controls of the same experiment (-2.57)
    assert classify_particle_type(-3.15, -2.57) == 'Beads'
    assert classify_particle_type(anchor - BEAD_ANCHOR_MARGIN - 0.01, anchor) == 'Beads'


def test_scatter_channel_pair_and_ceilings():
    raw = {'event_channels_pnn': ['FSC-A', 'SSC-A', 'BSSC-A', 'V1-A'],
           'scatter_param': ['FSC-A', 'BSSC-A'],
           'channel_pnr': [16777216.0, 16777216.0, 8000000.0, 10000.0],
           'magnitude_ceiling': 10000.0}
    assert scatter_channel_pair(raw) == ('FSC-A', 'BSSC-A')
    assert scatter_ceilings(raw, ('FSC-A', 'BSSC-A')) == [16777216.0, 8000000.0]

    raw['scatter_param'] = None
    plots = [{'type': 'hist1d', 'channel_x': 'V1-A', 'source_gate': 'root'},
             {'type': 'hist2d', 'channel_x': 'FSC-A', 'channel_y': 'SSC-A', 'source_gate': 'root'}]
    assert scatter_channel_pair(raw, plots) == ('FSC-A', 'SSC-A')
    assert scatter_channel_pair(raw, []) is None

    raw['channel_pnr'] = None
    assert scatter_ceilings(raw, ('FSC-A', 'SSC-A')) == [10000.0, 10000.0]


RAW = {'event_channels_pnn': ['FSC-A', 'SSC-A'], 'scatter_param': ['FSC-A', 'SSC-A'],
       'channel_pnr': [CEILING, CEILING], 'magnitude_ceiling': CEILING}


def _samples():
    return {
        'all_samples': {'c/A01 Unstained.fcs': 'A01 Unstained',
                        'c/G06 Unstained Cells.fcs': 'G06 Unstained Cells',
                        'c/B01 CD3 Beads.fcs': 'B01 CD3 Beads',
                        'c/A02 BUV395.fcs': 'A02 BUV395'},
        'unstained_samples': ['c/A01 Unstained.fcs', 'c/G06 Unstained Cells.fcs'],
    }


def test_is_bead_sample_by_name_until_listed():
    samples = _samples()
    assert is_bead_sample(samples, 'c/B01 CD3 Beads.fcs')
    assert not is_bead_sample(samples, 'c/A01 Unstained.fcs')
    samples['bead_samples'] = ['c/A01 Unstained.fcs']
    assert is_bead_sample(samples, 'c/A01 Unstained.fcs')
    assert not is_bead_sample(samples, 'c/B01 CD3 Beads.fcs')


def test_update_bead_samples_checks_each_sample_once(monkeypatch):
    scores = {'A01 Unstained.fcs': -3.76, 'G06 Unstained Cells.fcs': -1.74}
    calls = []

    def fake_score(abs_path, pair, ceilings):
        calls.append(abs_path.name)
        return scores.get(abs_path.name, -4.0), [5.0, 5.0]

    monkeypatch.setattr(pt, 'sample_scatter_core', fake_score)
    samples = _samples()
    newly = update_bead_samples(samples, '/exp', RAW)
    assert newly == ['c/B01 CD3 Beads.fcs', 'c/A01 Unstained.fcs']
    assert samples['bead_samples'] == ['c/A01 Unstained.fcs', 'c/B01 CD3 Beads.fcs']
    assert sorted(calls) == sorted(list(scores) + ['B01 CD3 Beads.fcs'])   # unstained, and beads for their centre

    # an unmark sticks: checked samples are not looked at again
    samples['bead_samples'].remove('c/B01 CD3 Beads.fcs')
    assert update_bead_samples(samples, '/exp', RAW) == []
    assert samples['bead_samples'] == ['c/A01 Unstained.fcs']
    assert len(calls) == 3

    # deleted files drop out of both lists
    samples['all_samples'].pop('c/A01 Unstained.fcs')
    update_bead_samples(samples, '/exp', RAW)
    assert samples['bead_samples'] == []
    assert 'c/A01 Unstained.fcs' not in samples['bead_checked_samples']


def test_update_bead_samples_waits_for_channels(monkeypatch):
    monkeypatch.setattr(pt, 'sample_scatter_core', lambda *a: (-4.0, [5.0, 5.0]))
    samples = _samples()
    update_bead_samples(samples, '/exp', {'event_channels_pnn': []})
    assert samples['bead_samples'] == ['c/B01 CD3 Beads.fcs']
    assert 'c/A01 Unstained.fcs' not in samples['bead_checked_samples']


def test_update_bead_samples_controls_and_known_types(monkeypatch):
    scores = {'A01 Unstained.fcs': -3.76, 'G06 Unstained Cells.fcs': -1.74,
              'A02 BUV395.fcs': -3.9, 'A03 BV421.fcs': -2.0, 'B01 CD3 Beads.fcs': -4.0,
              'A04 PE.fcs': -4.0}
    calls = []

    def fake_score(abs_path, pair, ceilings):
        calls.append(abs_path.name)
        return scores[abs_path.name], [5.0, 5.0]

    monkeypatch.setattr(pt, 'sample_scatter_core', fake_score)
    samples = _samples()
    samples['all_samples']['c/A03 BV421.fcs'] = 'A03 BV421'
    samples['all_samples']['c/A04 PE.fcs'] = 'A04 PE'
    samples['single_stain_controls'] = ['c/A02 BUV395.fcs', 'c/A03 BV421.fcs', 'c/A04 PE.fcs']
    update_bead_samples(samples, '/exp', RAW, known_types={'c/A04 PE.fcs': 'Beads'})
    assert samples['bead_samples'] == ['c/A01 Unstained.fcs', 'c/B01 CD3 Beads.fcs',
                                       'c/A02 BUV395.fcs', 'c/A04 PE.fcs']
    assert calls.count('A04 PE.fcs') == 1                 # Spectral Model type; read for its centre only
    assert samples['bead_scatter_scores']['c/G06 Unstained Cells.fcs'] == -1.74

    # a control added later is judged against the stored unstained anchor
    samples['all_samples']['c/A05 FITC.fcs'] = 'A05 FITC'
    samples['single_stain_controls'].append('c/A05 FITC.fcs')
    scores['A05 FITC.fcs'] = -2.95                        # below -2.9 and 1.21 below the anchor
    update_bead_samples(samples, '/exp', RAW)
    assert 'c/A05 FITC.fcs' in samples['bead_samples']
    assert calls.count('G06 Unstained Cells.fcs') == 1


def test_set_bead_sample():
    samples = _samples()
    samples['bead_samples'] = []
    assert set_bead_sample(samples, 'c/A02 BUV395.fcs', True)
    assert not set_bead_sample(samples, 'c/A02 BUV395.fcs', True)
    assert samples['bead_checked_samples'] == ['c/A02 BUV395.fcs']
    assert set_bead_sample(samples, 'c/A02 BUV395.fcs', False)
    assert samples['bead_samples'] == []


def test_match_bead_unstained():
    # centres from the Aurora bead_types files: Slingshot, UltraComp, cells
    samples = {
        'all_samples': {'u/Slingshot unstained.fcs': 'Slingshot unstained',
                        'u/UltraComp unstained.fcs': 'UltraComp unstained',
                        'u/Cells unstained.fcs': 'Cells unstained',
                        'c/BB515 Slingshot.fcs': 'BB515 Slingshot',
                        'c/BB515 UltraComp.fcs': 'BB515 UltraComp',
                        'c/PE Other beads.fcs': 'PE Other beads'},
        'unstained_samples': ['u/Slingshot unstained.fcs', 'u/UltraComp unstained.fcs', 'u/Cells unstained.fcs'],
        'bead_samples': ['u/Slingshot unstained.fcs', 'u/UltraComp unstained.fcs',
                         'c/BB515 Slingshot.fcs', 'c/BB515 UltraComp.fcs', 'c/PE Other beads.fcs'],
        'bead_scatter_centres': {'u/Slingshot unstained.fcs': [6.294, 6.347],
                                 'u/UltraComp unstained.fcs': [5.901, 6.296],
                                 'u/Cells unstained.fcs': [6.197, 5.831],
                                 'c/BB515 Slingshot.fcs': [6.297, 6.344],
                                 'c/BB515 UltraComp.fcs': [5.902, 6.299],
                                 'c/PE Other beads.fcs': [6.10, 6.30]},
    }
    assert match_bead_unstained(samples, 'c/BB515 Slingshot.fcs') == 'u/Slingshot unstained.fcs'
    assert match_bead_unstained(samples, 'c/BB515 UltraComp.fcs') == 'u/UltraComp unstained.fcs'
    # 0.19 / 0.20 decades from either unstained: no match, internal negative
    assert match_bead_unstained(samples, 'c/PE Other beads.fcs') is None
    # only one bead type has an unstained: the other gets none
    samples['unstained_samples'].remove('u/Slingshot unstained.fcs')
    assert match_bead_unstained(samples, 'c/BB515 Slingshot.fcs') is None
    assert match_bead_unstained(samples, 'c/BB515 UltraComp.fcs') == 'u/UltraComp unstained.fcs'
    # the cell unstained is never a bead match
    assert match_bead_unstained(samples, 'c/missing.fcs') is None
