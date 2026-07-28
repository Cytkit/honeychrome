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
