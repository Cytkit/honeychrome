"""Unit tests for the curated Spectral Reference Library store (steps 1-2).

No UI, no app — exercises compute_config_key + the SQLite CRUD store against a
temp DB. See _local_docs/SPECTRAL_REFERENCE_LIBRARY_PLAN.
"""
import pytest

from honeychrome.controller_components.spectral_reference_library import (
    compute_config_key,
    ReferenceProfile,
    SpectralReferenceLibrary,
    CONFIG_KEY_UNKNOWN_CYTOMETER,
)

# Layout shipped by early development builds: no antigen/particle_type/lot_number.
_OLD_TABLE_SQL = """
CREATE TABLE reference_library_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT, fluorophore TEXT NOT NULL,
    display_name TEXT NOT NULL, origin TEXT NOT NULL, cytometer_key TEXT NOT NULL,
    config_key TEXT NOT NULL, channel_names_json TEXT NOT NULL,
    profile_json TEXT NOT NULL, gate_channel TEXT, source_sample_name TEXT,
    source_experiment_dir TEXT, notes TEXT, created_at REAL NOT NULL,
    updated_at REAL NOT NULL, is_deletable INTEGER NOT NULL DEFAULT 1,
    is_reference INTEGER NOT NULL DEFAULT 0, is_qc_target INTEGER NOT NULL DEFAULT 0);
"""


# --- compute_config_key -----------------------------------------------------

def test_config_key_is_order_independent():
    a = compute_config_key('Aurora', ['B1-A', 'B2-A', 'V5-A'])
    b = compute_config_key('Aurora', ['V5-A', 'B1-A', 'B2-A'])
    assert a == b


def test_config_key_strips_whitespace():
    a = compute_config_key('Aurora', ['B1-A', 'B2-A'])
    b = compute_config_key('Aurora', [' B1-A ', 'B2-A '])
    assert a == b


def test_config_key_differs_by_channel_set():
    a = compute_config_key('Aurora', ['B1-A', 'B2-A', 'V5-A'])
    b = compute_config_key('Aurora', ['B1-A', 'B2-A'])
    assert a != b


def test_config_key_differs_by_cytometer():
    a = compute_config_key('Aurora', ['B1-A', 'B2-A'])
    b = compute_config_key('ID7000', ['B1-A', 'B2-A'])
    assert a != b


def test_config_key_none_cytometer_uses_placeholder():
    key = compute_config_key(None, ['B1-A'])
    assert key.startswith(CONFIG_KEY_UNKNOWN_CYTOMETER + '::')


def test_config_key_has_channel_count_suffix():
    key = compute_config_key('Aurora', ['B1-A', 'B2-A', 'V5-A'])
    assert key.endswith('::3ch')


# --- store fixtures ---------------------------------------------------------

@pytest.fixture
def lib(tmp_path):
    store = SpectralReferenceLibrary(library_path=tmp_path / 'test_library.db')
    store.ensure_schema()
    return store


def _save(lib, fluor='BUV805', cyt='Aurora', channels=('B1-A', 'B2-A'), **kw):
    cfg = compute_config_key(cyt, list(channels))
    profile = {c: float(i + 1) / len(channels) for i, c in enumerate(channels)}
    return lib.save_profile(
        fluorophore=fluor, profile=profile, cytometer_key=cyt, config_key=cfg,
        channel_names=list(channels), **kw,
    )


# --- save / get round-trip --------------------------------------------------

def test_save_and_get_roundtrip(lib):
    p = _save(lib)
    assert p.id is not None
    got = lib.get_profile(p.id)
    assert got.fluorophore == 'BUV805'
    assert got.origin == 'user'
    assert got.is_deletable is True
    assert got.profile == p.profile
    assert got.channel_names == ['B1-A', 'B2-A']


def test_display_name_defaults_to_fluorophore(lib):
    p = _save(lib, fluor='PE')
    assert p.display_name == 'PE'


# --- listing ----------------------------------------------------------------

def test_list_by_cytometer_and_config(lib):
    _save(lib, fluor='BUV805', cyt='Aurora', channels=('B1-A', 'B2-A'))
    _save(lib, fluor='PE', cyt='Aurora', channels=('B1-A', 'B2-A'))
    _save(lib, fluor='PE', cyt='ID7000', channels=('B1-A', 'B2-A'))

    assert len(lib.list_profiles('Aurora')) == 2
    assert len(lib.list_profiles('ID7000')) == 1

    cfg = compute_config_key('Aurora', ['B1-A', 'B2-A'])
    assert len(lib.list_profiles_for_config(cfg)) == 2


# --- reference flag (per fluorophore+config) --------------------------------

def test_set_reference_is_unique_within_group(lib):
    p1 = _save(lib, fluor='PE')
    p2 = _save(lib, fluor='PE')
    lib.set_reference(p1.id)
    assert lib.get_profile(p1.id).is_reference is True

    lib.set_reference(p2.id)  # should clear p1
    assert lib.get_profile(p1.id).is_reference is False
    assert lib.get_profile(p2.id).is_reference is True

    cfg = p2.config_key
    ref = lib.get_reference_for_config('PE', cfg)
    assert ref.id == p2.id


def test_reference_scoped_by_config(lib):
    # same fluorophore, different config -> independent reference flags
    a = _save(lib, fluor='PE', channels=('B1-A', 'B2-A'))
    b = _save(lib, fluor='PE', channels=('B1-A', 'B2-A', 'V5-A'))
    lib.set_reference(a.id)
    lib.set_reference(b.id)
    assert lib.get_profile(a.id).is_reference is True   # not cleared: different config
    assert lib.get_profile(b.id).is_reference is True


# --- qc target flag (per fluorophore+cytometer) -----------------------------

def test_set_qc_target_is_unique_within_cytometer(lib):
    p1 = _save(lib, fluor='PE', cyt='Aurora', channels=('B1-A', 'B2-A'))
    p2 = _save(lib, fluor='PE', cyt='Aurora', channels=('B1-A', 'B2-A', 'V5-A'))
    lib.set_qc_target(p1.id)
    lib.set_qc_target(p2.id)  # different config but same cytometer -> clears p1
    assert lib.get_profile(p1.id).is_qc_target is False
    assert lib.get_profile(p2.id).is_qc_target is True
    tgt = lib.get_qc_target_for_cytometer('PE', 'Aurora')
    assert tgt.id == p2.id


# --- rename -----------------------------------------------------------------

def test_rename_profile(lib):
    p = _save(lib, fluor='PE')
    lib.rename_profile(p.id, 'PE (lot 42)')
    assert lib.get_profile(p.id).display_name == 'PE (lot 42)'


# --- delete + deletability --------------------------------------------------

def test_delete_user_profile(lib):
    p = _save(lib, fluor='PE')
    lib.delete_profile(p.id)
    assert lib.get_profile(p.id) is None


def test_cannot_delete_honeychrome_profile(lib):
    p = _save(lib, fluor='PE', origin='honeychrome')
    assert p.is_deletable is False
    with pytest.raises(ValueError):
        lib.delete_profile(p.id)
    assert lib.get_profile(p.id) is not None


def test_delete_unknown_id_raises(lib):
    with pytest.raises(ValueError):
        lib.delete_profile(9999)


# --- acquisition metadata + user-editable fields ----------------------------

def test_metadata_fields_roundtrip(lib):
    p = _save(lib, fluor='PE', antigen='CD3', particle_type='Cells',
              lot_number='LOT-42', gate_channel='B2-A', notes='fixed PBMC')
    got = lib.get_profile(p.id)
    assert got.antigen == 'CD3'
    assert got.particle_type == 'Cells'
    assert got.lot_number == 'LOT-42'
    assert got.gate_channel == 'B2-A'
    assert got.notes == 'fixed PBMC'


def test_metadata_fields_default_to_none(lib):
    got = lib.get_profile(_save(lib).id)
    assert got.antigen is None
    assert got.particle_type is None
    assert got.lot_number is None


def test_update_fields_edits_user_entered_values(lib):
    p = _save(lib, fluor='PE')
    lib.update_fields(p.id, lot_number='LOT-7', notes='new note', antigen='CD4')
    got = lib.get_profile(p.id)
    assert (got.lot_number, got.notes, got.antigen) == ('LOT-7', 'new note', 'CD4')


def test_update_fields_ignores_unknown_columns(lib):
    p = _save(lib, fluor='PE')
    lib.update_fields(p.id, origin='honeychrome', not_a_column='x')
    assert lib.get_profile(p.id).origin == 'user'


def test_table_from_an_earlier_dev_build_gains_the_new_columns(tmp_path):
    """A colleague's DB from before antigen/particle_type/lot_number existed."""
    import sqlite3
    path = tmp_path / 'stale.db'
    conn = sqlite3.connect(path)
    conn.executescript(_OLD_TABLE_SQL)
    conn.commit()
    conn.close()

    store = SpectralReferenceLibrary(library_path=path)
    store.ensure_schema()
    p = _save(store, fluor='PE', lot_number='LOT-1')
    assert store.get_profile(p.id).lot_number == 'LOT-1'


def test_peak_detector_backfilled_for_rows_without_one(lib):
    p = _save(lib, fluor='PE', channels=('B1-A', 'B2-A', 'B3-A'))
    assert lib.get_profile(p.id).gate_channel is None
    lib._backfill_peak_detectors()
    # profile values increase with index, so the last channel is the peak
    assert lib.get_profile(p.id).gate_channel == 'B3-A'


# --- CSV import / export ----------------------------------------------------

def _write_csv(path, rows, channels=('B1-A', 'B2-A', 'B3-A')):
    import csv
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['fluorophore', *channels])
        for name, values in rows.items():
            writer.writerow([name, *values])
    return path


def test_import_csv_creates_user_profiles(lib, tmp_path):
    path = _write_csv(tmp_path / 'ref.csv', {'PE': [1, 2, 4], 'APC': [0, 5, 5]})
    imported = lib.import_csv(path, 'Aurora')
    assert len(imported) == 2
    assert {p.fluorophore for p in imported} == {'PE', 'APC'}
    assert all(p.origin == 'user' and p.is_deletable for p in imported)


def test_import_csv_normalises_rows_to_peak_of_one(lib, tmp_path):
    """Imported spectra must be scaled like the shipped CSVs, else they cannot be
    compared with them on the same plot."""
    path = _write_csv(tmp_path / 'ref.csv', {'PE': [1, 2, 4]})
    profile = lib.import_csv(path, 'Aurora')[0].profile
    assert profile == {'B1-A': 0.25, 'B2-A': 0.5, 'B3-A': 1.0}


def test_import_csv_sets_peak_detector(lib, tmp_path):
    path = _write_csv(tmp_path / 'ref.csv', {'PE': [1, 9, 4]})
    assert lib.import_csv(path, 'Aurora')[0].gate_channel == 'B2-A'


def test_import_csv_skips_rows_with_no_signal(lib, tmp_path):
    path = _write_csv(tmp_path / 'ref.csv', {'PE': [1, 2, 4], 'Empty': [0, 0, 0]})
    imported = lib.import_csv(path, 'Aurora')
    assert [p.fluorophore for p in imported] == ['PE']


def test_import_csv_profiles_are_listed_for_that_instrument(lib, tmp_path):
    path = _write_csv(tmp_path / 'ref.csv', {'PE': [1, 2, 4]})
    lib.import_csv(path, 'Aurora')
    assert [p.fluorophore for p in lib.list_profiles('Aurora')] == ['PE']


def test_export_csv_roundtrips_through_import(lib, tmp_path):
    source = _write_csv(tmp_path / 'in.csv', {'PE': [1, 2, 4], 'APC': [8, 4, 2]})
    imported = lib.import_csv(source, 'Aurora')

    out = tmp_path / 'out.csv'
    lib.export_csv(out, imported)

    fresh = SpectralReferenceLibrary(library_path=tmp_path / 'fresh.db')
    fresh.ensure_schema()
    reimported = {p.fluorophore: p.profile for p in fresh.import_csv(out, 'Aurora')}
    assert reimported == {p.fluorophore: p.profile for p in imported}


def test_export_csv_keeps_detector_order(lib, tmp_path):
    path = _write_csv(tmp_path / 'in.csv', {'PE': [1, 2, 4]})
    imported = lib.import_csv(path, 'Aurora')
    out = tmp_path / 'out.csv'
    lib.export_csv(out, imported)

    import csv
    with open(out, newline='') as f:
        header = next(csv.reader(f))
    assert header[1:] == ['B1-A', 'B2-A', 'B3-A']


def test_export_csv_pads_mixed_configurations_with_zero(lib, tmp_path):
    a = _save(lib, fluor='PE', channels=('B1-A', 'B2-A'))
    b = _save(lib, fluor='APC', channels=('B1-A', 'B2-A', 'V1-A'))
    out = tmp_path / 'out.csv'
    lib.export_csv(out, [a, b])

    import csv
    with open(out, newline='') as f:
        rows = list(csv.reader(f))
    assert rows[0][1:] == ['B1-A', 'B2-A', 'V1-A']
    pe = next(r for r in rows if r[0] == 'PE')
    assert float(pe[3]) == 0.0   # PE has no V1-A


def test_export_csv_with_no_profiles_raises(lib, tmp_path):
    with pytest.raises(ValueError):
        lib.export_csv(tmp_path / 'out.csv', [])
