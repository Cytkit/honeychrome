import pandas as pd
import pytest

from honeychrome.controller_components import sample_table as stt

KEYS = ['Raw/Fully stained/A1 POMS_008_FB_D_Plate 1.fcs',
        'Raw/Fully stained/E4 POMS_008_MB_D_Plate 2.fcs',
        'Raw/Fully stained/A1 Nonpreg_002_Plate 4.fcs',
        'Raw/Fully stained/C4 IC_BuffyCoat_Donor_E_Plate 4.fcs']


def _key_table():
    return pd.DataFrame({
        'Sample': ['POMS_008_FB_D', 'POMS_008_MB_D', 'Nonpreg_002', 'CB_001', '', 'D', 'Total'],
        'Age Group': ['D', 'D', 'na', 'D', '', 'Moderate', ''],
        'Vaccinated': ['Y', 'N', '', '', '', '', '']})


def test_ids_inside_file_names_match_and_legend_rows_do_not():
    m = stt.match_table(_key_table(), KEYS)
    assert m.rows[0] == [KEYS[0]] and m.rows[1] == [KEYS[1]] and m.rows[2] == [KEYS[2]]
    assert m.how[0] == stt.INSIDE
    assert m.unmatched == ['CB_001', 'D', 'Total']
    assert m.samples_without_row == [KEYS[3]]
    vals = stt.table_values(_key_table(), m, None, ['Age Group', 'Vaccinated'])
    assert vals['groups'] == {}
    assert vals['covariates']['Vaccinated'] == {KEYS[0]: 'Y', KEYS[1]: 'N'}


def test_longest_name_wins_and_exact_beats_inside():
    df = pd.DataFrame({'sample': ['POMS_008', 'POMS_008_FB_D', 'A1 Nonpreg_002_Plate 4.fcs']})
    m = stt.match_table(df, KEYS)
    assert m.rows[1] == [KEYS[0]]
    assert m.rows[0] == [KEYS[1]]
    assert m.rows[2] == [KEYS[2]] and m.how[2] == stt.EXACT


def test_name_fitting_several_samples_is_reported():
    keys = ['Raw/A1 S_001_Plate 1.fcs', 'Raw/B1 S_001_Plate 2.fcs']
    m = stt.match_table(pd.DataFrame({'sample': ['S_001']}), keys)
    assert m.rows[0] == keys and m.multiple == ['S_001']


def test_no_sample_column_raises():
    with pytest.raises(ValueError):
        stt.match_table(pd.DataFrame({'x': ['1']}), KEYS)


def test_export_round_trips_with_shown_names(tmp_path):
    groups = {KEYS[0]: 'Foetal', KEYS[1]: 'Maternal'}
    covs = {'donor': {KEYS[0]: 'POMS_008', KEYS[1]: 'POMS_008'}}
    path = tmp_path / 't.csv'
    stt.write_table(stt.table_frame(KEYS, groups, covs), path)
    assert path.read_text().splitlines()[1].startswith('"A1 POMS_008_FB_D_Plate 1",')
    df = stt.read_table_file(path)
    m = stt.match_table(df, KEYS)
    group_col = stt.find_column(df, stt.GROUP_COLUMNS)
    vals = stt.table_values(df, m, group_col, stt.covariate_candidates(df, m, group_col))
    assert vals['groups'] == groups and vals['covariates']['donor'] == covs['donor']


def test_read_table_file_strips_headers_and_drops_empty_rows(tmp_path):
    path = tmp_path / 'k.csv'
    path.write_text('Sample,Age Group ,Cells\nPOMS_008_FB_D,D,"768,500"\n,,\n')
    df = stt.read_table_file(path)
    assert list(df.columns) == ['Sample', 'Age Group', 'Cells'] and len(df) == 1
    assert df.loc[0, 'Cells'] == '768,500'


def test_default_export_path_names_experiment_and_plugin():
    assert stt.default_export_path('/x/UCL IMMPROVE', 'Automated Gating').endswith(
        'UCL IMMPROVE/UCL IMMPROVE_automated_gating_sample_table.csv')
