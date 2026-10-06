"""
test_autospectral.py
--------------------
Integration tests for the AutoSpectral AF extraction and unmixing pipeline.

Tests are split into two groups:

  Group A — Pure-numpy (no FCS files, no Qt, fast):
    These test the numerical correctness of the autospectral_functions module
    in isolation and should be run on every push to dev.

  Group B — Controller-level (require a real experiment on disk):
    These test the wiring between autospectral_functions and the Controller,
    including experiment persistence. They mirror the pattern of
    test_controller.py and test_experiment_model.py and are intended as the
    dev-branch merge gate.

    They expect the same experiment used by the existing tests:
        ~/spectral_cytometry/20240620 Spectral Symposium-poor cell unmixed.kit

    and at least one sample in that experiment that has a computed unmixing
    matrix (i.e. refresh_spectral_process() has been run at least once and
    saved).

Usage:
    pytest test_autospectral.py                   # all tests
    pytest test_autospectral.py -m numpy_only     # Group A only (CI fast path)
    pytest test_autospectral.py -m controller     # Group B only
"""

import multiprocessing as mp
import os
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

mp.set_start_method("spawn", force=True)

# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

N_CHANNELS = 14   # fluorescence channels only (matches default Honeychrome config)
N_FLUORS   = 8    # number of synthetic fluorophores
N_CELLS    = 500  # synthetic event count
RNG        = np.random.default_rng(42)


def _make_fluor_spectra(n_fluors=N_FLUORS, n_channels=N_CHANNELS, rng=RNG):
    """
    Return a (n_fluors, n_channels) array of L-inf-normalised synthetic
    fluorophore spectra.  Each row peaks in a different channel so they are
    well-separated and the OLS solve is numerically stable.
    """
    spectra = np.zeros((n_fluors, n_channels))
    for i in range(n_fluors):
        peak = i % n_channels
        spectra[i, peak] = 1.0
        # add small Gaussian tails either side
        for offset in range(1, 4):
            if peak - offset >= 0:
                spectra[i, peak - offset] = 0.5 ** offset
            if peak + offset < n_channels:
                spectra[i, peak + offset] = 0.5 ** offset
        spectra[i] /= spectra[i].max()
    return spectra


def _make_af_spectra(n_af=3, n_channels=N_CHANNELS, rng=RNG):
    """
    Return (n_af, n_channels) L-inf-normalised AF spectra that span the
    full detector range and are distinct from the synthetic fluorophores
    (broad, multi-peak profiles).
    """
    af = rng.uniform(0.1, 0.8, size=(n_af, n_channels))
    af /= af.max(axis=1, keepdims=True)
    return af


def _make_raw_events(fluor_spectra, af_spectra, n_cells=N_CELLS, rng=RNG):
    """
    Simulate raw fluorescence events as a mixture of fluorophore signal plus
    one AF component plus Poisson noise.  Returns (n_cells, n_channels).
    """
    n_fluors, n_channels = fluor_spectra.shape
    abundances = rng.exponential(scale=500.0, size=(n_cells, n_fluors))
    signal = abundances @ fluor_spectra  # (n_cells, n_channels)

    # each cell gets a random amount of the first AF spectrum
    af_scale = rng.exponential(scale=200.0, size=(n_cells, 1))
    signal += af_scale * af_spectra[0]

    noise = rng.poisson(signal.clip(min=0)).astype(float)
    return noise


# ---------------------------------------------------------------------------
# GROUP A — Pure-numpy tests (fast, no FCS files required)
# ---------------------------------------------------------------------------

@pytest.mark.numpy_only
def test_precompute_af_matrices_shapes():
    """precompute_af_matrices returns a dict whose arrays have the expected shapes."""
    from honeychrome.controller_components.autospectral_functions import precompute_af_matrices

    fluor_spectra = _make_fluor_spectra()
    af_spectra    = _make_af_spectra(n_af=3)

    result = precompute_af_matrices(fluor_spectra, af_spectra)

    assert result['P'].shape           == (N_FLUORS, N_CHANNELS), \
        f"P shape mismatch: {result['P'].shape}"
    assert result['S_t'].shape         == (N_CHANNELS, N_FLUORS), \
        f"S_t shape mismatch: {result['S_t'].shape}"
    assert result['v_library'].shape   == (N_FLUORS, 3), \
        f"v_library shape mismatch: {result['v_library'].shape}"
    assert result['r_library'].shape   == (N_CHANNELS, 3), \
        f"r_library shape mismatch: {result['r_library'].shape}"
    assert result['r_dots'].shape      == (3,), \
        f"r_dots shape mismatch: {result['r_dots'].shape}"
    assert (result['r_dots'] > 0).all(), \
        "r_dots must be positive (clamped at 1e-20)"


@pytest.mark.numpy_only
def test_apply_af_unmixing_output_shapes_and_no_nan():
    """apply_af_unmixing returns correctly shaped arrays with no NaN or Inf."""
    from honeychrome.controller_components.autospectral_functions import (
        precompute_af_matrices, apply_af_unmixing,
    )

    fluor_spectra = _make_fluor_spectra()
    af_spectra    = _make_af_spectra(n_af=3)
    raw_fl        = _make_raw_events(fluor_spectra, af_spectra)

    precomputed = precompute_af_matrices(fluor_spectra, af_spectra)
    result = apply_af_unmixing(raw_fl, precomputed, af_spectra)

    assert 'unmixed'  in result
    assert 'af_scale' in result
    assert 'af_idx'   in result

    assert result['unmixed'].shape  == (N_CELLS, N_FLUORS), \
        f"unmixed shape: {result['unmixed'].shape}"
    assert result['af_scale'].shape == (N_CELLS,), \
        f"af_scale shape: {result['af_scale'].shape}"
    assert result['af_idx'].shape   == (N_CELLS,), \
        f"af_idx shape: {result['af_idx'].shape}"

    assert not np.isnan(result['unmixed']).any(),  "NaN in unmixed output"
    assert not np.isinf(result['unmixed']).any(),  "Inf in unmixed output"
    assert not np.isnan(result['af_scale']).any(), "NaN in af_scale"


@pytest.mark.numpy_only
def test_apply_af_unmixing_corrects_af_signal():
    """
    When events contain a known AF contribution, AF unmixing should reduce
    the mean absolute value of the first fluorophore channel compared to
    plain OLS (because AF was bleeding into that channel).
    """
    from honeychrome.controller_components.autospectral_functions import (
        precompute_af_matrices, apply_af_unmixing,
    )
    from honeychrome.controller_components.functions import apply_transfer_matrix

    fluor_spectra = _make_fluor_spectra()
    af_spectra    = _make_af_spectra(n_af=1)

    # Simulate events: pure AF signal with no real fluorophore content
    rng = np.random.default_rng(7)
    af_scale = rng.exponential(scale=1000.0, size=(N_CELLS, 1))
    raw_fl = af_scale * af_spectra[0]  # all signal is AF

    # Plain OLS unmix via P matrix
    P = np.linalg.solve(fluor_spectra @ fluor_spectra.T, fluor_spectra)
    ols_unmixed = raw_fl @ P.T  # (n_cells, n_fluors)

    # AF-corrected unmix
    precomputed = precompute_af_matrices(fluor_spectra, af_spectra)
    result = apply_af_unmixing(raw_fl, precomputed, af_spectra)

    ols_mean_abs   = np.abs(ols_unmixed).mean()
    af_mean_abs    = np.abs(result['unmixed']).mean()

    assert af_mean_abs < ols_mean_abs, (
        f"AF unmixing should reduce residuals vs plain OLS on pure-AF data. "
        f"OLS mean abs={ols_mean_abs:.4f}, AF mean abs={af_mean_abs:.4f}"
    )


@pytest.mark.numpy_only
def test_af_idx_is_1_based():
    """af_idx must use 1-based indexing (matching R convention)."""
    from honeychrome.controller_components.autospectral_functions import (
        precompute_af_matrices, apply_af_unmixing,
    )

    fluor_spectra = _make_fluor_spectra()
    af_spectra    = _make_af_spectra(n_af=3)
    raw_fl        = _make_raw_events(fluor_spectra, af_spectra)

    precomputed = precompute_af_matrices(fluor_spectra, af_spectra)
    result = apply_af_unmixing(raw_fl, precomputed, af_spectra)

    assert result['af_idx'].min() >= 1, \
        f"af_idx minimum should be 1 (1-based), got {result['af_idx'].min()}"
    assert result['af_idx'].max() <= af_spectra.shape[0], \
        f"af_idx maximum should be <= n_af={af_spectra.shape[0]}, got {result['af_idx'].max()}"


@pytest.mark.numpy_only
def test_combine_af_precomputed_matches_single():
    """
    Combining a single precomputed dict via combine_af_precomputed should return
    the same dict unchanged.
    """
    from honeychrome.controller_components.autospectral_functions import (
        precompute_af_matrices, combine_af_precomputed,
    )

    fluor_spectra = _make_fluor_spectra()
    af_spectra    = _make_af_spectra(n_af=2)
    pc = precompute_af_matrices(fluor_spectra, af_spectra)

    combined = combine_af_precomputed([pc])
    assert combined is pc, "combine_af_precomputed([single]) should return the original dict"


@pytest.mark.numpy_only
def test_combine_af_precomputed_concatenates_columns():
    """
    Combining two precomputed dicts should horizontally stack v_library,
    r_library, and r_dots, but leave P and S_t unchanged.
    """
    from honeychrome.controller_components.autospectral_functions import (
        precompute_af_matrices, combine_af_precomputed,
    )

    fluor_spectra = _make_fluor_spectra()
    af1 = _make_af_spectra(n_af=2)
    af2 = _make_af_spectra(n_af=3, rng=np.random.default_rng(99))

    pc1 = precompute_af_matrices(fluor_spectra, af1)
    pc2 = precompute_af_matrices(fluor_spectra, af2)
    combined = combine_af_precomputed([pc1, pc2])

    assert combined['v_library'].shape == (N_FLUORS,  5), \
        f"v_library columns should be 2+3=5, got {combined['v_library'].shape}"
    assert combined['r_library'].shape == (N_CHANNELS, 5), \
        f"r_library columns should be 2+3=5, got {combined['r_library'].shape}"
    assert combined['r_dots'].shape    == (5,), \
        f"r_dots length should be 2+3=5, got {combined['r_dots'].shape}"

    # P and S_t must be identical to the first profile's values
    np.testing.assert_array_equal(combined['P'],   pc1['P'],   err_msg="P must come from first profile")
    np.testing.assert_array_equal(combined['S_t'], pc1['S_t'], err_msg="S_t must come from first profile")


@pytest.mark.numpy_only
def test_save_and_load_af_profile_csv_roundtrip():
    """
    save_af_profile_csv writes a CSV; load_af_profile_csv reads it back.
    The reloaded spectra must be numerically identical to the originals, the
    profile name must be derived correctly, and the channel names must round-trip.
    """
    from honeychrome.controller_components.autospectral_functions import (
        save_af_profile_csv, load_af_profile_csv,
    )

    af_spectra    = _make_af_spectra(n_af=4)
    channel_names = [f'Ch{i:02d}' for i in range(N_CHANNELS)]

    with tempfile.TemporaryDirectory() as tmp:
        experiment_dir = Path(tmp)
        fcs_path = 'Raw/Single stain controls/Spleen_unstained.fcs'

        profile_name = save_af_profile_csv(af_spectra, channel_names, fcs_path, experiment_dir)

        assert profile_name == 'Spleen_unstained AutoSpectral AF', \
            f"Unexpected profile name: {profile_name!r}"

        csv_path = experiment_dir / 'AutoSpectral' / f'{profile_name}.csv'
        assert csv_path.exists(), f"CSV not written to expected path: {csv_path}"

        loaded_name, loaded_spectra, loaded_channels = load_af_profile_csv(csv_path)

        assert loaded_name == profile_name, \
            f"Loaded profile name mismatch: {loaded_name!r} vs {profile_name!r}"
        assert loaded_channels == channel_names, \
            f"Channel names did not round-trip: {loaded_channels}"
        np.testing.assert_allclose(
            loaded_spectra, af_spectra, rtol=1e-6,
            err_msg="AF spectra did not survive CSV round-trip within tolerance",
        )


@pytest.mark.numpy_only
def test_get_af_spectra_returns_valid_output():
    """
    get_af_spectra on synthetic unstained data should return:
    - at least 1 row (the population mean is always kept)
    - L-inf-normalised rows (max value == 1.0 for each row)
    - no NaN or Inf
    """
    from honeychrome.controller_components.autospectral_functions import get_af_spectra

    fluor_spectra = _make_fluor_spectra()
    # Pure AF events: spectral shape independent of any fluorophore
    rng = np.random.default_rng(21)
    unstained = rng.exponential(scale=300.0, size=(1000, N_CHANNELS))

    af_spectra = get_af_spectra(unstained, fluor_spectra, som_dim=4)

    assert af_spectra.shape[0] >= 1, "Expected at least the population mean row"
    assert af_spectra.shape[1] == N_CHANNELS, \
        f"Channel count mismatch: {af_spectra.shape[1]}"
    assert not np.isnan(af_spectra).any(), "NaN in returned AF spectra"
    assert not np.isinf(af_spectra).any(), "Inf in returned AF spectra"

    row_maxima = np.abs(af_spectra).max(axis=1)
    np.testing.assert_allclose(
        row_maxima, 1.0, atol=1e-9,
        err_msg="Each AF spectrum row must be L-inf normalised (max == 1.0)",
    )


@pytest.mark.numpy_only
def test_get_af_spectra_raises_on_too_few_cells():
    """get_af_spectra must raise ValueError when given fewer than min_cells events."""
    from honeychrome.controller_components.autospectral_functions import get_af_spectra

    fluor_spectra = _make_fluor_spectra()
    tiny = RNG.uniform(0, 500, size=(50, N_CHANNELS))  # below default min_cells=100

    with pytest.raises(ValueError, match="Insufficient cells"):
        get_af_spectra(tiny, fluor_spectra)


def _contaminated_unstained(seed, n_cells=1000, contaminated_fraction=0.15):
    """Unstained events from two AF spectra, with a fraction of the events
    also carrying a bright fluorophore-0 signal (a single-stained control
    mixed into the unstained sample)."""
    rng = np.random.default_rng(seed)
    fluor_spectra = _make_fluor_spectra()
    af = _make_af_spectra(n_af=2, rng=np.random.default_rng(seed + 100))
    abundance = rng.lognormal(np.log(1500), 0.3, n_cells)
    events = (abundance[:, None] * af[rng.integers(0, 2, n_cells)]
              + rng.normal(0, 20, (n_cells, N_CHANNELS)))
    k = int(contaminated_fraction * n_cells)
    events[:k] += rng.lognormal(np.log(100000), 0.2, k)[:, None] * fluor_spectra[0]
    return events.clip(0), fluor_spectra


def _max_fluor_cosine(af_spectra, fluor_spectra):
    a = af_spectra / np.linalg.norm(af_spectra, axis=1, keepdims=True)
    f = fluor_spectra / np.linalg.norm(fluor_spectra, axis=1, keepdims=True)
    return float((a @ f.T).max())


@pytest.mark.numpy_only
@pytest.mark.parametrize('seed', [0, 1, 2])
def test_get_af_spectra_removes_fluorophore_contaminants(seed):
    """
    Events contaminated with a fluorophore must not produce an AF spectrum
    resembling that fluorophore. Without the contamination filter the same
    data does (cosine similarity above 0.99), so the filter is what removes it.
    """
    from honeychrome.controller_components.autospectral_functions import get_af_spectra

    unstained, fluor_spectra = _contaminated_unstained(seed)

    filtered = get_af_spectra(unstained, fluor_spectra, som_dim=4)
    assert filtered.shape[0] >= 1
    assert _max_fluor_cosine(filtered, fluor_spectra) < 0.9

    unfiltered = get_af_spectra(unstained, fluor_spectra, som_dim=4,
                                remove_contaminants=False, refine=False)
    assert _max_fluor_cosine(unfiltered, fluor_spectra) > 0.99


@pytest.mark.numpy_only
def test_get_af_spectra_rejects_fluorophore_only_unstained():
    """
    An "unstained" sample whose every event looks like one fluorophore has no
    AF signal to extract: every candidate spectrum, including the population
    mean, fails contamination QC and get_af_spectra raises.
    """
    from honeychrome.controller_components.autospectral_functions import get_af_spectra

    rng = np.random.default_rng(5)
    fluor_spectra = _make_fluor_spectra()
    n_cells = 500
    fluor0_signal = np.tile(fluor_spectra[0] * 1000, (n_cells, 1))
    noise = rng.normal(0, 10, size=(n_cells, N_CHANNELS))
    unstained = (fluor0_signal + noise).clip(0)

    with pytest.raises(ValueError, match='contamination QC'):
        get_af_spectra(unstained, fluor_spectra, som_dim=3)


# ---------------------------------------------------------------------------
# GROUP B — Controller-level integration tests (require experiment on disk)
# ---------------------------------------------------------------------------

# Set HONEYCHROME_TEST_EXPERIMENT to a .kit file to run this group on another
# experiment.
BASE_DIR = Path.home() / 'Experiments'
EXPERIMENT_PATH = Path(os.environ.get('HONEYCHROME_TEST_EXPERIMENT',
                                      str(BASE_DIR / '7C panel 2.kit'))).expanduser()

# Skip the whole group if the experiment file isn't present
requires_experiment = pytest.mark.skipif(
    not EXPERIMENT_PATH.exists(),
    reason=f"Experiment file not found at {EXPERIMENT_PATH}",
)


@pytest.fixture(scope='module')
def experiment_copy(tmp_path_factory):
    """
    A copy of the test experiment's .kit file, so nothing a test does can
    change the original. The experiment folder (FCS files) is linked beside
    the copy rather than copied. Opening the copy does not add it to the
    app's recent-files list.
    """
    import honeychrome.controller as controller_module

    tmp = tmp_path_factory.mktemp('experiment')
    kit = tmp / EXPERIMENT_PATH.name
    shutil.copy2(EXPERIMENT_PATH, kit)
    data_dir = EXPERIMENT_PATH.parent / EXPERIMENT_PATH.stem
    if data_dir.is_dir():
        (tmp / EXPERIMENT_PATH.stem).symlink_to(data_dir, target_is_directory=True)
    patcher = pytest.MonkeyPatch()
    patcher.setattr(controller_module, 'add_recent_file', lambda path: None)
    yield kit
    patcher.undo()


@pytest.fixture(scope='module')
def loaded_controller(experiment_copy):
    """
    Module-scoped fixture: opens the experiment copy, runs the spectral
    process, and loads the first sample.  Shared across all controller-level
    tests to avoid redundant disk I/O.
    """
    from honeychrome.controller import Controller

    kc = Controller()
    kc.load_experiment(experiment_copy)
    kc.regenerate_spectral_model()
    kc.refresh_spectral_process()
    kc.set_mode('Unmixed Data')

    first_sample = list(kc.experiment.samples['all_samples'].keys())[0]
    kc.load_sample(first_sample)

    return kc


def _raw_fluor_columns(kc):
    """Positions of the filtered fluorescence channels in kc.raw_event_data.

    filtered_raw_fluorescence_channel_ids index the full raw channel list,
    but raw_event_data holds only the whitelisted channels, in whitelist
    order (as in Controller._apply_unmixing).
    """
    raw = kc.experiment.settings['raw']
    pnn_loaded = raw.get('whitelisted_pnn') or raw['event_channels_pnn']
    return [pnn_loaded.index(raw['event_channels_pnn'][i])
            for i in kc.filtered_raw_fluorescence_channel_ids]


@requires_experiment
@pytest.mark.controller
def test_controller_af_assignment_changes_unmixed_output(loaded_controller):
    """
    Assigning an AF profile to the current sample and re-running unmixing
    must produce different fluorescence output from plain OLS.  After
    the assignment, _apply_unmixing should take the AF path.
    """
    kc = loaded_controller

    assert kc.raw_event_data is not None, "No raw event data loaded"
    assert kc.transfer_matrix is not None, "No transfer matrix — run spectral process first"

    ols_result = kc.unmixed_event_data.copy()

    # Build a synthetic AF profile from the current sample's raw fluorescence data
    fluor_ids = kc.filtered_raw_fluorescence_channel_ids
    raw_fl    = kc.raw_event_data[:, _raw_fluor_columns(kc)]
    fluor_spectra = kc._build_fluor_spectra()

    from honeychrome.controller_components.autospectral_functions import get_af_spectra
    af_spectra = get_af_spectra(raw_fl, fluor_spectra, som_dim=4)

    channel_names = [kc.experiment.settings['raw']['event_channels_pnn'][i] for i in fluor_ids]

    with tempfile.TemporaryDirectory() as tmp:
        from honeychrome.controller_components.autospectral_functions import save_af_profile_csv
        profile_name = save_af_profile_csv(
            af_spectra, channel_names,
            source_fcs_path=kc.current_sample_path,
            experiment_dir=Path(tmp),
        )

    # Store the profile and assign it to the current sample
    kc.experiment.process['af_profiles'][profile_name] = {
        'spectra': af_spectra.tolist(),
        'channel_names': channel_names,
    }
    sample_af = kc.experiment.samples.setdefault('sample_af_profiles', {})
    sample_af[kc.current_sample_path] = [profile_name]

    kc.cache_af_profile(profile_name)
    kc.initialise_af_matrices()

    assert kc.af_precomputed is not None, \
        "af_precomputed should be set after assignment"
    assert kc.af_spectra is not None, \
        "af_spectra should be set after assignment"

    af_result = kc._apply_unmixing(kc.raw_event_data)

    fl_ids_unmixed = np.array(kc.experiment.settings['unmixed']['fluorescence_channel_ids'])
    ols_fl = ols_result[:, fl_ids_unmixed]
    af_fl  = af_result[:,  fl_ids_unmixed]

    assert not np.allclose(ols_fl, af_fl, atol=1e-6), (
        "AF-corrected unmixing produced identical output to plain OLS — "
        "the AF path is not being taken or has no effect"
    )

    # Clean up: remove the test profile so we don't pollute other tests
    kc.experiment.process['af_profiles'].pop(profile_name, None)
    kc.experiment.samples['sample_af_profiles'][kc.current_sample_path] = []
    kc.initialise_af_matrices()


@requires_experiment
@pytest.mark.controller
def test_controller_af_unmixing_preserves_scatter_channels(loaded_controller):
    """
    AF unmixing must not modify scatter (FSC/SSC) columns.  The scatter
    columns in the AF-corrected result must be identical to plain OLS.
    """
    kc = loaded_controller

    assert kc.raw_event_data is not None
    assert kc.transfer_matrix is not None

    raw_fl        = kc.raw_event_data[:, _raw_fluor_columns(kc)]
    fluor_spectra = kc._build_fluor_spectra()

    from honeychrome.controller_components.autospectral_functions import (
        get_af_spectra, precompute_af_matrices, apply_af_transfer,
    )
    af_spectra  = get_af_spectra(raw_fl, fluor_spectra, som_dim=4)
    precomputed = precompute_af_matrices(fluor_spectra, af_spectra)

    ols_result = kc.raw_event_data @ kc.transfer_matrix

    af_result = apply_af_transfer(
        kc.raw_event_data,
        kc.transfer_matrix,
        precomputed,
        af_spectra,
        kc.experiment.settings,
        filtered_fl_ids_raw=_raw_fluor_columns(kc),
    )

    sc_ids = np.array(kc.experiment.settings['unmixed']['scatter_channel_ids'])
    np.testing.assert_array_equal(
        ols_result[:, sc_ids],
        af_result['unmixed'][:, sc_ids],
        err_msg="Scatter columns must not be modified by AF unmixing",
    )


@requires_experiment
@pytest.mark.controller
def test_controller_clear_af_reverts_to_ols(loaded_controller):
    """
    After assigning an AF profile and then clearing it, _apply_unmixing must
    return output that is numerically identical to plain OLS.
    """
    from honeychrome.controller_components.autospectral_functions import (
        get_af_spectra, save_af_profile_csv,
    )
    from honeychrome.controller_components.functions import apply_transfer_matrix

    kc = loaded_controller
    assert kc.raw_event_data is not None
    assert kc.transfer_matrix is not None

    fluor_ids     = kc.filtered_raw_fluorescence_channel_ids
    raw_fl        = kc.raw_event_data[:, _raw_fluor_columns(kc)]
    fluor_spectra = kc._build_fluor_spectra()
    af_spectra    = get_af_spectra(raw_fl, fluor_spectra, som_dim=4)
    channel_names = [kc.experiment.settings['raw']['event_channels_pnn'][i] for i in fluor_ids]

    profile_name = 'test_clear_af_profile AutoSpectral AF'
    kc.experiment.process['af_profiles'][profile_name] = {
        'spectra': af_spectra.tolist(),
        'channel_names': channel_names,
    }
    kc.experiment.samples.setdefault('sample_af_profiles', {})[kc.current_sample_path] = [profile_name]
    kc.cache_af_profile(profile_name)
    kc.initialise_af_matrices()

    # Verify AF is active
    assert kc.af_precomputed is not None

    # Now clear
    kc.experiment.samples['sample_af_profiles'][kc.current_sample_path] = []
    kc.initialise_af_matrices()

    assert kc.af_precomputed is None, \
        "af_precomputed should be None after clearing all AF profiles for this sample"
    assert kc.af_spectra is None, \
        "af_spectra should be None after clearing"

    reverted_result = kc._apply_unmixing(kc.raw_event_data)
    ols_result      = apply_transfer_matrix(kc.transfer_matrix, kc.raw_event_data)

    np.testing.assert_array_equal(
        reverted_result, ols_result,
        err_msg="After clearing AF, _apply_unmixing should give identical output to plain OLS",
    )

    # Cleanup
    kc.experiment.process['af_profiles'].pop(profile_name, None)


@requires_experiment
@pytest.mark.controller
def test_controller_regenerate_spectral_process_with_af_assigned(loaded_controller):
    """
    After regenerating the spectral process, AF profiles already stored in the
    experiment must still be present and the af_precomputed cache must be
    rebuilt so that a subsequent load_sample() uses AF correction.

    This is the most important test: it guards against the bug class where
    refresh_spectral_process() silently invalidates the AF cache without
    reinitialising it.
    """
    from honeychrome.controller_components.autospectral_functions import get_af_spectra

    kc = loaded_controller
    assert kc.raw_event_data is not None
    assert kc.transfer_matrix is not None

    fluor_ids     = kc.filtered_raw_fluorescence_channel_ids
    raw_fl        = kc.raw_event_data[:, _raw_fluor_columns(kc)]
    fluor_spectra = kc._build_fluor_spectra()
    af_spectra    = get_af_spectra(raw_fl, fluor_spectra, som_dim=4)
    channel_names = [kc.experiment.settings['raw']['event_channels_pnn'][i] for i in fluor_ids]

    profile_name = 'test_regen_profile AutoSpectral AF'
    kc.experiment.process['af_profiles'][profile_name] = {
        'spectra': af_spectra.tolist(),
        'channel_names': channel_names,
    }
    kc.experiment.samples.setdefault('sample_af_profiles', {})[kc.current_sample_path] = [profile_name]
    kc.cache_af_profile(profile_name)
    kc.initialise_af_matrices()
    assert kc.af_precomputed is not None, "Precondition: AF should be active before regen"

    # Simulate what happens when the user re-runs the spectral process
    kc.regenerate_spectral_model()
    kc.refresh_spectral_process()

    # Profile must still be present in the experiment
    assert profile_name in kc.experiment.process['af_profiles'], \
        "AF profile was lost from experiment.process after regenerating spectral model"

    assigned = kc.experiment.samples['sample_af_profiles'].get(kc.current_sample_path, [])
    assert profile_name in assigned, \
        "AF profile assignment was lost from experiment.samples after regenerating spectral model"

    # Cache must have been rebuilt (initialise_transfer_matrix calls cache_all_af_profiles)
    assert profile_name in kc.af_precomputed_cache, \
        "AF precomputed cache was not rebuilt after refresh_spectral_process"

    # Loading the sample must activate AF correction
    kc.load_sample(kc.current_sample_path)
    assert kc.af_precomputed is not None, \
        "af_precomputed is None after load_sample — AF correction not reinstated after spectral regen"

    # Cleanup
    kc.experiment.process['af_profiles'].pop(profile_name, None)
    kc.experiment.samples['sample_af_profiles'][kc.current_sample_path] = []
    kc.initialise_af_matrices()


@requires_experiment
@pytest.mark.controller
def test_experiment_save_load_preserves_af_state(experiment_copy):
    """
    AF profiles and per-sample assignments survive a save/load cycle.
    Uses DeepDiff, matching the pattern of test_experiment_model.py.
    """
    from honeychrome.controller import Controller
    from honeychrome.experiment_model import ExperimentModel
    from deepdiff import DeepDiff
    from honeychrome.controller_components.autospectral_functions import get_af_spectra

    kc = Controller()
    kc.load_experiment(experiment_copy)
    kc.regenerate_spectral_model()
    kc.refresh_spectral_process()

    first_sample = list(kc.experiment.samples['all_samples'].keys())[0]
    kc.load_sample(first_sample)

    fluor_ids     = kc.filtered_raw_fluorescence_channel_ids
    raw_fl        = kc.raw_event_data[:, _raw_fluor_columns(kc)]
    fluor_spectra = kc._build_fluor_spectra()
    af_spectra    = get_af_spectra(raw_fl, fluor_spectra, som_dim=4)
    channel_names = [kc.experiment.settings['raw']['event_channels_pnn'][i] for i in fluor_ids]

    profile_name = 'test_persistence_profile AutoSpectral AF'
    kc.experiment.process['af_profiles'][profile_name] = {
        'spectra': af_spectra.tolist(),
        'channel_names': channel_names,
    }
    kc.experiment.samples.setdefault('sample_af_profiles', {})[first_sample] = [profile_name]

    with tempfile.TemporaryDirectory() as tmp:
        save_path = Path(tmp) / 'test_af_persistence.kit'
        # Use experiment.save() directly — save_experiment(path) intentionally
        # strips samples (it's the "save as template" path in the controller).
        kc.experiment.experiment_path = str(save_path)
        kc.experiment.save()

        reloaded = ExperimentModel()
        reloaded.load(save_path)

    assert profile_name in reloaded.process.get('af_profiles', {}), \
        "AF profile not found in reloaded experiment.process['af_profiles']"

    reloaded_assigned = reloaded.samples.get('sample_af_profiles', {}).get(first_sample, [])
    assert profile_name in reloaded_assigned, \
        "AF profile assignment not found in reloaded experiment.samples['sample_af_profiles']"

    # Check the spectra themselves survived serialisation without drift
    original_spectra = np.array(kc.experiment.process['af_profiles'][profile_name]['spectra'])
    reloaded_spectra = np.array(reloaded.process['af_profiles'][profile_name]['spectra'])
    np.testing.assert_allclose(
        reloaded_spectra, original_spectra, rtol=1e-6,
        err_msg="AF spectra changed during experiment save/load",
    )
