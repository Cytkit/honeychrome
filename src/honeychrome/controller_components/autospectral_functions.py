"""
autospectral_functions.py
--------------------------
AutoSpectral AF extraction for Honeychrome.

Public API
----------
get_af_spectra(unstained_raw, fluor_spectra, som_dim)
    Identifies AF spectral profiles from an unstained sample using SOM
    clustering (KMeans when the compiled SOM kernel is unavailable), with
    optional solver-validated refinement for poorly corrected cells.
    Returns an (n_af, n_channels) ndarray of L-inf-normalised AF spectra,
    with the population mean prepended as row 0.

get_som_codes(data, som_dim, dist)
    Batch SOM codebook via the compiled SOM kernel, falling back to KMeans.

assign_af_joint_l2(raw_data, fluor_spectra, af_spectra)
    Per-cell AF assignment by the joint covariance-weighted L2 fluorophore
    error x L2 residual error criterion.

apply_af_unmixing(raw_data, precomputed, af_spectra)
    Per-cell AF extraction and OLS unmixing for fluorescence channels only.

precompute_af_matrices(fluor_spectra, af_spectra)
    Precomputes projection matrices; call once after spectral process refresh
    and cache the result on the controller.

apply_af_transfer(raw_event_data, transfer_matrix, af_precomputed, af_spectra, settings)
    Assembles a full unmixed event array, overwriting fluorescence columns
    with AF-corrected OLS values.

rank_af_spectra(af_spectra) / af_index_lookup(af_profiles, profile_names)
    Experiment-wide AF Index numbering: every spectrum of every stored
    profile gets one index, ranked within its profile by spectral angle to
    the profile's mean spectrum.

save_af_profile_csv(af_spectra, channel_names, source_fcs_path, experiment_dir)
    Saves an AF profile as a CSV file in the experiment's AutoSpectral folder.
    Returns the profile name (str) used as the key in experiment.process['af_profiles'].

load_af_profile_csv(csv_path)
    Loads an AF profile from a CSV file previously saved by save_af_profile_csv.
    Returns (profile_name, spectra_ndarray, channel_names).
"""

import logging
import sys
from pathlib import Path

import numpy as np


logger = logging.getLogger(__name__)

# Sub-folder inside the experiment directory where CSV files are stored
AF_SUBDIR = 'AutoSpectral'


# ---------------------------------------------------------------------------
# CSV save / load
# ---------------------------------------------------------------------------

def save_af_profile_csv(
    af_spectra: np.ndarray,
    channel_names: list,
    source_fcs_path: str,
    experiment_dir: Path,
) -> str:
    """
    Save an AF profile to CSV and return the profile name.

    The CSV file has:
      - Column headers: 'AF_index', then one column per detector
      - Rows: 0 = population mean, 1..n = cluster AF spectra

    File name: "<stem of source_fcs_path> AutoSpectral AF.csv"
    Location:  <experiment_dir>/AutoSpectral/

    Parameters
    ----------
    af_spectra : ndarray, shape (n_af, n_channels)
    channel_names : list[str]
        Detector names for the fluorescence channels, in column order.
    source_fcs_path : str
        Relative (or absolute) path of the FCS file used to extract the profile.
        Only the stem is used for naming.
    experiment_dir : Path
        Root experiment directory (contains the .kit file's sibling folder).

    Returns
    -------
    str
        Profile name, e.g. "Spleen_unstained AutoSpectral AF".
        This is the key used in experiment.process['af_profiles'].
    """
    import pandas as pd

    stem = Path(source_fcs_path).stem
    profile_name = f'{stem} AutoSpectral AF'

    out_dir = Path(experiment_dir) / AF_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f'{profile_name}.csv'

    n_af = af_spectra.shape[0]
    row_labels = ['mean'] + [str(i) for i in range(1, n_af)]

    df = pd.DataFrame(af_spectra, columns=channel_names)
    df.insert(0, 'AF_index', row_labels)
    df.to_csv(csv_path, index=False)

    logger.info(f'AutoSpectral: saved AF profile "{profile_name}" to {csv_path}')
    return profile_name


def load_af_profile_csv(csv_path: str | Path):
    """
    Load an AF profile from a CSV file.

    Parameters
    ----------
    csv_path : str or Path

    Returns
    -------
    tuple (profile_name, af_spectra, channel_names)
        profile_name : str — derived from the file stem
        af_spectra   : ndarray, shape (n_af, n_channels)
        channel_names: list[str]
    """
    import pandas as pd

    csv_path = Path(csv_path)
    profile_name = csv_path.stem  # e.g. "Spleen_unstained AutoSpectral AF"

    df = pd.read_csv(csv_path)
    channel_names = [c for c in df.columns if c != 'AF_index']
    af_spectra = df[channel_names].to_numpy(dtype=float)

    logger.info(
        f'AutoSpectral: loaded AF profile "{profile_name}" '
        f'({af_spectra.shape[0]} spectra, {af_spectra.shape[1]} channels) '
        f'from {csv_path}'
    )
    return profile_name, af_spectra, channel_names


# ---------------------------------------------------------------------------
# Precomputation (run once per spectral process / per-sample AF assignment)
# ---------------------------------------------------------------------------

def precompute_af_matrices(fluor_spectra: np.ndarray, af_spectra: np.ndarray) -> dict:
    """
    Precompute the matrices needed for per-cell AF unmixing.

    Parameters
    ----------
    fluor_spectra : ndarray, shape (n_fluors, n_channels)
        L-infinity-normalised fluorophore spectral profiles.
    af_spectra : ndarray, shape (n_af, n_channels)
        AF spectral profiles from get_af_spectra().

    Returns
    -------
    dict with keys:
        P           : (n_fluors, n_channels)  OLS unmixing matrix
        S_t         : (n_channels, n_fluors)
        v_library   : (n_fluors, n_af)  in-span projection of each AF variant
        r_library   : (n_channels, n_af)  out-of-span residual of each variant
        r_dots      : (n_af,)  squared norm of each r_library column. Not
                      floored for identifiability here; that floor depends on
                      the whole library, which may be several profiles
                      combined, so it is applied at scoring time.
    """
    P = np.linalg.solve(fluor_spectra @ fluor_spectra.T, fluor_spectra)
    S_t = fluor_spectra.T
    AF_t = af_spectra.T

    v_library = P @ AF_t
    r_library = AF_t - S_t @ v_library
    r_dots = np.einsum('ij,ij->j', r_library, r_library)
    r_dots = np.where(r_dots <= 0.0, 1e-10, r_dots)

    return {
        'P': P,
        'S_t': S_t,
        'v_library': v_library,
        'r_library': r_library,
        'r_dots': r_dots,
    }


def precompute_joint_cov_extras(precomputed: dict, af_spectra: np.ndarray) -> dict:
    """
    Covariance-based fluorophore error weights for joint-cov L2 scoring: the
    AF library's spectral covariance propagated into fluorophore space, with
    the square root of its diagonal as the per-fluorophore weight. Call once
    after precompute_af_matrices() (or after combining profiles, since the
    weights depend on the full library); cache alongside af_precomputed.
    """
    P = precomputed['P']   # (n_fluors, n_channels)
    n_channels = af_spectra.shape[1]
    af_cov = (np.cov(af_spectra, rowvar=False)
              if af_spectra.shape[0] > 1
              else np.zeros((n_channels, n_channels)))
    fluor_cov = P @ af_cov @ P.T
    af_error_weights = np.sqrt(np.abs(np.diag(fluor_cov)))
    if af_error_weights.max() < 1e-12:
        af_error_weights = np.ones(P.shape[0])
    return {'af_error_weights': af_error_weights}


def combine_af_precomputed(precomputed_list: list) -> dict:
    """
    Combine a list of per-profile precomputed dicts into one combined dict.

    Because all the column-wise arrays (v_library, r_library, r_dots) are
    independent across profiles, combination is simply np.hstack — no further
    matrix algebra is needed.  P and S_t are identical for all profiles (they
    depend only on fluor_spectra) so we take them from the first entry.
    af_error_weights depend on the combined library and must be recomputed
    with precompute_joint_cov_extras() afterwards.

    Parameters
    ----------
    precomputed_list : list of dict
        Each element is the output of precompute_af_matrices() for one profile.
        Must be non-empty.

    Returns
    -------
    dict — same structure as precompute_af_matrices() output, but with
    v_library, r_library, and r_dots spanning all profiles combined.
    """
    if len(precomputed_list) == 1:
        return precomputed_list[0]

    return {
        'P':         precomputed_list[0]['P'],        # (n_fluors, n_channels) — shared
        'S_t':       precomputed_list[0]['S_t'],      # (n_channels, n_fluors) — shared
        'v_library': np.hstack([d['v_library'] for d in precomputed_list]),
        'r_library': np.hstack([d['r_library'] for d in precomputed_list]),
        'r_dots':    np.concatenate([d['r_dots'] for d in precomputed_list]),
    }


# ---------------------------------------------------------------------------
# Helper: assemble full unmixed event array
# ---------------------------------------------------------------------------

def rank_af_spectra(af_spectra: np.ndarray) -> np.ndarray:
    """
    Order of one AF profile's spectra by spectral angle to the profile's mean
    spectrum, closest first.

    The spectral angle between two spectra is arccos of their cosine
    similarity, so ranking by descending cosine similarity to the mean is
    ranking by ascending angle. Ties keep their stored order.

    Returns an int array of row indices into af_spectra.
    """
    spectra = np.asarray(af_spectra, dtype=np.float64)
    if spectra.ndim != 2 or spectra.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    norms = np.linalg.norm(spectra, axis=1)
    norms = np.where(norms < 1e-12, 1.0, norms)
    mean = spectra.mean(axis=0)
    mean_norm = max(float(np.linalg.norm(mean)), 1e-12)
    cosine = (spectra @ mean) / (norms * mean_norm)
    return np.argsort(-cosine, kind='stable')


def af_index_lookup(af_profiles: dict, profile_names) -> np.ndarray | None:
    """
    Experiment-wide AF Index for each row of a sample's combined AF library.

    Every spectrum of every stored profile gets one index, 1..N, where N is
    the total number of AF spectra in the experiment. Profiles are numbered
    in their stored order (af_profiles is insertion ordered), and within a
    profile the spectra are numbered by rank_af_spectra(), so index order
    follows spectral angle from that profile's mean.

    Parameters
    ----------
    af_profiles   : experiment.process['af_profiles']
    profile_names : the sample's assigned profile names, in the order their
                    spectra are stacked into its combined library. Names not
                    in af_profiles are skipped, as they are when the library
                    is built.

    Returns
    -------
    int64 ndarray, one experiment-wide index per combined-library row (so
    lookup[af_idx - 1] converts a per-sample af_idx), or None when none of
    the names is stored.
    """
    af_profiles = af_profiles or {}
    offsets = {}
    offset = 0
    for name, entry in af_profiles.items():
        spectra = np.asarray(entry.get('spectra') or [], dtype=np.float64)
        rank = np.empty(len(spectra), dtype=np.int64)
        rank[rank_af_spectra(spectra)] = np.arange(len(spectra))
        offsets[name] = offset + rank + 1
        offset += len(spectra)
    parts = [offsets[name] for name in (profile_names or []) if name in offsets]
    if not parts:
        return None
    return np.concatenate(parts)


def apply_af_transfer(raw_event_data, transfer_matrix, af_precomputed, af_spectra, settings,
                      filtered_fl_ids_raw=None, spillover=None, af_index_map=None):
    """
    Assemble a full unmixed event array with AF-corrected fluorescence columns.
    Scatter, time, and event_id columns come from the standard transfer_matrix path;
    the AF Abundance and AF Index columns, when present, hold af_scale and the
    AF Index. With af_index_map (from af_index_lookup) the AF Index column holds
    the experiment-wide index; otherwise it holds af_idx. The returned 'af_idx'
    is always the per-sample index into af_spectra.

    The AF unmixing (apply_af_unmixing) produces abundances in plain OLS fluorophore
    space. If a spillover matrix is provided, compensation (inv(spillover).T) is applied
    to those fluorescence columns so the result matches the compensated transfer_matrix
    path. The transpose matches spillover's row-spills-into-column convention to the
    column-vector multiplication used below (see controller.py::initialise_transfer_matrix).
    """
    from honeychrome.controller_components.functions import apply_transfer_matrix

    raw_settings = settings['raw']
    unmixed_settings = settings['unmixed']

    if filtered_fl_ids_raw is not None:
        fl_ids_raw = np.array(filtered_fl_ids_raw)
    else:
        fl_ids_raw = np.array(raw_settings['fluorescence_channel_ids'])
        
    fl_ids_unmixed = np.array(unmixed_settings['fluorescence_channel_ids'])

    unmixed = raw_event_data @ transfer_matrix

    raw_fl = raw_event_data[:, fl_ids_raw]
    result = apply_af_unmixing(raw_fl, af_precomputed, af_spectra)
    af_unmixed_fl = result['unmixed']  # (n_cells, n_fluors) — plain OLS space

    if spillover is not None:
        compensation = np.linalg.inv(np.array(spillover)).T
        af_unmixed_fl = (compensation @ af_unmixed_fl.T).T

    unmixed[:, fl_ids_unmixed] = af_unmixed_fl

    # Per-cell AF abundance and library index go in the AF channels when the
    # unmixed channel list has them (the transfer matrix leaves them at 0).
    from honeychrome.settings import af_abundance_channel, af_index_channel
    pnn_unmixed = unmixed_settings.get('event_channels_pnn') or []
    af_index = result['af_idx']
    if af_index_map is not None:
        af_index = np.asarray(af_index_map)[result['af_idx'] - 1]
    for label, values in ((af_abundance_channel, result['af_scale']),
                          (af_index_channel, af_index)):
        if label in pnn_unmixed and pnn_unmixed.index(label) < unmixed.shape[1]:
            unmixed[:, pnn_unmixed.index(label)] = values

    return {
        'unmixed': unmixed,
        'af_scale': result['af_scale'],   # (n_cells,)
        'af_idx':   result['af_idx'],     # (n_cells,)
    }


# ---------------------------------------------------------------------------
# Per-cell AF assignment and unmixing
# ---------------------------------------------------------------------------

def _joint_l2_library_terms(precomputed: dict, af_spectra: np.ndarray):
    """
    Per-variant constants for the joint L2 score, taken across the whole
    (possibly combined) AF library.

    Returns (w, k_denom, c_fluor):
        w       : (n_fluors,) covariance-derived fluorophore error weights
        k_denom : (n_af,) r_dots floored at 1% of the largest. An AF variant
                  lying almost inside the fluorophore span has a vanishing
                  out-of-span residual, so its abundance is not identifiable
                  and the raw ratio explodes; the floor caps that. Used for
                  the abundance estimate both inside the score and in the
                  reported abundance (apply_af_unmixing), as in the compiled
                  joint kernel.
        c_fluor : (n_af,) sum_f w_f * v_library[f, j]^2, the curvature of the
                  weighted squared fluorophore error.
    """
    w = precomputed.get('af_error_weights')
    if w is None:
        w = precompute_joint_cov_extras(precomputed, af_spectra)['af_error_weights']
    r_dots = precomputed['r_dots']
    k_denom = np.maximum(r_dots, 0.01 * max(float(r_dots.max()), 1e-10))
    c_fluor = w @ (precomputed['v_library'] ** 2)
    return w, k_denom, c_fluor


def _assign_af_chunk(chunk, precomputed, w, k_denom, c_fluor):
    """
    Joint covariance-weighted L2 fluorophore x L2 residual AF scoring for one
    chunk of cells (port of AutoSpectral's assign.af.joint.cov.l2()).

    For variant j with abundance k (clamped >= 0), both error terms are
    quadratics in k:
        e_fluor_j = base_e_fluor - 2k <w*u, v_j> + k^2 c_fluor_j
        e_resid_j = base_e_resid - 2k <resid, r_j> + k^2 |r_j|^2
    where u is the AF-free OLS unmix and resid the raw-space residual against
    its non-negative part. Each term is one matrix product across the chunk,
    so no (cells x fluors x variants) temporary is formed. The score is the
    product of the two proportional errors; the variant minimising it wins.

    Returns (best_j, numerator, unmixed):
        best_j    : (B,) 0-based variant index
        numerator : (B, n_af) chunk @ r_library
        unmixed   : (B, n_fluors) AF-free OLS unmix
    """
    P         = precomputed['P']
    S_t       = precomputed['S_t']
    v_library = precomputed['v_library']
    r_library = precomputed['r_library']
    r_dots    = precomputed['r_dots']

    unmixed = chunk @ P.T
    resid   = chunk - np.maximum(unmixed, 0.0) @ S_t.T

    base_e_fluor = (unmixed * unmixed) @ w + 1e-6
    base_e_resid = np.einsum('ij,ij->i', resid, resid) + 1e-6

    numerator = chunk @ r_library
    k  = np.maximum(numerator / k_denom, 0.0)
    k2 = k * k

    e_fluor = (unmixed * w) @ v_library
    e_fluor *= -2.0 * k
    e_fluor += k2 * c_fluor
    e_fluor += base_e_fluor[:, np.newaxis]
    np.maximum(e_fluor, 0.0, out=e_fluor)
    e_fluor /= base_e_fluor[:, np.newaxis]

    e_resid = resid @ r_library
    e_resid *= -2.0 * k
    e_resid += k2 * r_dots
    e_resid += base_e_resid[:, np.newaxis]
    np.maximum(e_resid, 0.0, out=e_resid)
    e_resid /= base_e_resid[:, np.newaxis]

    e_fluor *= e_resid
    best_j = np.argmin(e_fluor, axis=1)
    return best_j, numerator, unmixed


def assign_af_joint_l2(
    raw_data: np.ndarray,
    fluor_spectra: np.ndarray,
    af_spectra: np.ndarray,
    chunk_size: int = 50_000,
) -> np.ndarray:
    """
    Assign each cell its best-fitting AF variant by the joint
    covariance-weighted L2 fluorophore x L2 residual criterion.

    Parameters
    ----------
    raw_data      : (n_cells, n_channels) raw fluorescence only
    fluor_spectra : (n_fluors, n_channels)
    af_spectra    : (n_af, n_channels)
    chunk_size    : cells per processing batch

    Returns
    -------
    ndarray (n_cells,) int64, 0-based row index into af_spectra.
    """
    precomputed = precompute_af_matrices(fluor_spectra, af_spectra)
    precomputed.update(precompute_joint_cov_extras(precomputed, af_spectra))
    w, k_denom, c_fluor = _joint_l2_library_terms(precomputed, af_spectra)

    n_cells = raw_data.shape[0]
    best = np.empty(n_cells, dtype=np.int64)
    for start in range(0, n_cells, chunk_size):
        end   = min(start + chunk_size, n_cells)
        chunk = np.ascontiguousarray(raw_data[start:end], dtype=np.float64)
        best[start:end] = _assign_af_chunk(chunk, precomputed, w, k_denom, c_fluor)[0]
    return best


def apply_af_unmixing(
    raw_data: np.ndarray,
    precomputed: dict,
    af_spectra: np.ndarray,
    chunk_size: int = 50_000,
) -> dict:
    """
    Per-cell AF extraction and OLS unmixing (fluorescence channels only).

    Each cell is assigned the AF variant minimising the joint
    covariance-weighted L2 fluorophore x L2 residual score (see
    _assign_af_chunk), then solved jointly with the fluorophores by
    Frisch-Waugh-Lovell, as in AutoSpectral's unmix.af.fwl(): the AF
    abundance is the projection of the cell onto the variant's out-of-span
    residual direction, and the fluorophore abundances are the AF-free OLS
    solution minus that abundance times the variant's in-span projection.
    As in the compiled joint kernel, the AF abundance is clamped at 0 and its
    denominator is floored (see _joint_l2_library_terms) so a variant lying
    almost inside the fluorophore span cannot receive an exploding abundance.
    Where neither applies this is identical to an OLS solve against
    [fluor_spectra; af_spectra[j]].

    Parameters
    ----------
    raw_data    : (n_cells, n_channels) raw fluorescence only
    precomputed : dict from precompute_af_matrices(), optionally extended
                  with precompute_joint_cov_extras() merged in (computed here
                  from af_spectra when absent)
    af_spectra  : (n_af, n_channels)
    chunk_size  : cells per processing batch

    Returns
    -------
    dict with keys: unmixed (n_cells, n_fluors), af_scale (n_cells,),
                    af_idx (n_cells,) 1-based
    """
    v_library = precomputed['v_library']   # (n_fluors, n_af)
    w, k_denom, c_fluor = _joint_l2_library_terms(precomputed, af_spectra)

    n_cells  = raw_data.shape[0]
    n_fluors = precomputed['P'].shape[0]
    v_library_t = np.ascontiguousarray(v_library.T)

    unmixed_out  = np.empty((n_cells, n_fluors), dtype=np.float64)
    af_scale_out = np.empty(n_cells,             dtype=np.float64)
    af_idx_out   = np.empty(n_cells,             dtype=np.int32)

    for start in range(0, n_cells, chunk_size):
        end   = min(start + chunk_size, n_cells)
        chunk = np.ascontiguousarray(raw_data[start:end], dtype=np.float64)

        best_j, numerator, unmixed = _assign_af_chunk(
            chunk, precomputed, w, k_denom, c_fluor
        )

        best_k = np.maximum(numerator[np.arange(end - start), best_j], 0.0) / k_denom[best_j]
        unmixed_out[start:end]  = unmixed - best_k[:, np.newaxis] * v_library_t[best_j]
        af_scale_out[start:end] = best_k
        af_idx_out[start:end]   = best_j + 1

    return {'unmixed': unmixed_out, 'af_scale': af_scale_out, 'af_idx': af_idx_out}


# ---------------------------------------------------------------------------
# Clustering engine: batch SOM with KMeans fallback
# ---------------------------------------------------------------------------

def _load_som_kernel():
    """
    Return the som_kernel_wrapper module from bundled_plugins/ when its
    compiled kernel is importable, else None.
    """
    meipass = getattr(sys, '_MEIPASS', None)
    if meipass:
        plugin_dir = Path(meipass) / 'honeychrome' / 'bundled_plugins'
    else:
        plugin_dir = Path(__file__).resolve().parent.parent / 'bundled_plugins'
    if not (plugin_dir / 'som_kernel_wrapper.py').exists():
        return None
    if str(plugin_dir) not in sys.path:
        sys.path.append(str(plugin_dir))
    try:
        import som_kernel_wrapper
    except Exception as e:
        logger.info(f'get_som_codes: SOM kernel wrapper could not be imported ({e}).')
        return None
    return som_kernel_wrapper if som_kernel_wrapper.SOM_KERNEL_AVAILABLE else None


def get_som_codes(
    data: np.ndarray,
    som_dim: int,
    dist: int = 2,
    rlen: int = 10,
    random_state: int = 42,
    n_threads: int = 0,
    unit_norm: bool = False,
):
    """
    Codebook of a som_dim x som_dim self-organising map trained on `data`.

    Mirrors AutoSpectral's get.som.codes(): a batch SOM on a square grid with
    Chebyshev neighbour distances, the neighbourhood radius annealing over
    `rlen` epochs from the 67th percentile of grid distances to a tenth of
    that, and the codebook initialised from a random sample of rows. Uses the
    compiled OpenMP kernel (bundled_plugins/som_kernel_wrapper.py) when it is
    available; otherwise falls back to KMeans (MiniBatchKMeans above 200,000
    rows) with som_dim**2 clusters. Under dist=4 the fallback clusters
    unit-length rows but returns each cluster's mean on the original scale,
    matching the SOM's cosine assignment with raw-value code updates.

    With unit_norm=True every row is scaled to unit L2 length before
    training (all-zero rows are dropped), as get.som.codes(unit.norm = TRUE)
    does. Under dist=4 this makes each code the mean of unit event vectors,
    so bright events do not dominate it. Cosine assignments are unchanged,
    and the returned codes are on the unit-normalised scale.

    Parameters
    ----------
    data         : (n, n_features)
    som_dim      : grid side length; the codebook has som_dim**2 rows
    dist         : 1 manhattan, 2 euclidean, 3 chebyshev, 4 cosine. The
                   KMeans fallback treats anything other than 4 as euclidean.
    rlen         : SOM training epochs
    random_state : seed for the initial codebook / KMeans
    n_threads    : OpenMP threads for the SOM kernel, 0 = all cores
    unit_norm    : scale rows to unit L2 length before training

    Returns
    -------
    (codes, engine) — codes (som_dim**2, n_features) float64; engine is
    'som' or 'kmeans'.
    """
    data = np.ascontiguousarray(data, dtype=np.float64)
    if unit_norm:
        norms = np.linalg.norm(data, axis=1)
        keep = norms > 0
        data = np.ascontiguousarray(data[keep] / norms[keep, np.newaxis])
    n_codes = int(som_dim) ** 2
    if data.shape[0] < n_codes:
        raise ValueError(
            f'Not enough events ({data.shape[0]}) to initialise a '
            f'{som_dim}x{som_dim} SOM ({n_codes} nodes).'
        )

    som = _load_som_kernel()
    if som is not None:
        grid = np.array([(i, j) for j in range(1, som_dim + 1) for i in range(1, som_dim + 1)],
                        dtype=np.float64)
        nhbrdist = np.abs(grid[:, np.newaxis, :] - grid[np.newaxis, :, :]).max(axis=2)
        radius_start = float(np.percentile(nhbrdist, 67))
        radii = np.linspace(radius_start, 0.1 * radius_start, rlen)
        rng = np.random.default_rng(random_state)
        init_codes = data[rng.choice(data.shape[0], n_codes, replace=False)]
        codes = som.train_som_batch(
            data, init_codes, nhbrdist, radii, dist=dist, n_threads=n_threads,
        )
        return codes, 'som'

    from sklearn.cluster import KMeans, MiniBatchKMeans

    if data.shape[0] > 200_000:
        km = MiniBatchKMeans(n_clusters=n_codes, random_state=random_state, n_init='auto')
    else:
        km = KMeans(n_clusters=n_codes, random_state=random_state, n_init='auto')

    if dist == 4:
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        labels = km.fit_predict(data / norms)
        codes = np.zeros((n_codes, data.shape[1]), dtype=np.float64)
        counts = np.bincount(labels, minlength=n_codes)
        np.add.at(codes, labels, data)
        filled = counts > 0
        codes[filled] /= counts[filled, np.newaxis]
        codes[~filled] = km.cluster_centers_[~filled]
        return codes, 'kmeans'

    km.fit(data)
    return km.cluster_centers_, 'kmeans'


# ---------------------------------------------------------------------------
# AF spectra identification (training step)
# ---------------------------------------------------------------------------

def _row_normalise(m: np.ndarray) -> np.ndarray:
    """Scale each row to unit L2 length (zero rows left as zero)."""
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return m / norms


def _cosine_similarity_matrix(a: np.ndarray) -> np.ndarray:
    """
    Compute pairwise cosine similarity for rows of a.
    Returns an (n, n) matrix in [-1, 1].
    """
    a_norm = _row_normalise(a)
    return a_norm @ a_norm.T


def _deduplicate_spectra(
    spectra: np.ndarray,
    cosine_threshold: float = 0.995,
) -> np.ndarray:
    """
    Greedy cosine-similarity deduplication.

    Iterates through rows in order, keeping a row only if its cosine
    similarity to every already-kept row is below cosine_threshold. Row 0 is
    always kept.

    Parameters
    ----------
    spectra : ndarray, shape (n, n_channels)
        L-inf-normalised spectra.
    cosine_threshold : float
        Rows at or above this similarity to any kept row are dropped.

    Returns
    -------
    ndarray, shape (m, n_channels), m <= n
    """
    if len(spectra) == 0:
        return spectra

    sim = _cosine_similarity_matrix(spectra)
    kept = []
    for i in range(len(spectra)):
        if all(sim[i, j] < cosine_threshold for j in kept):
            kept.append(i)
    return spectra[kept]


def _qc_af_spectra(
    af_spectra: np.ndarray,
    fluor_spectra: np.ndarray,
    cosine_threshold: float = 0.99,
) -> np.ndarray:
    """
    Remove any AF spectrum whose cosine similarity to any fluorophore
    spectrum exceeds cosine_threshold — these are likely contamination
    from single-stained controls in the unstained sample.

    Parameters
    ----------
    af_spectra : ndarray, shape (n_af, n_channels)
    fluor_spectra : ndarray, shape (n_fluors, n_channels)
    cosine_threshold : float

    Returns
    -------
    ndarray — filtered af_spectra (may be shorter than input)
    """
    if len(af_spectra) == 0:
        return af_spectra

    # sim[i, j] = cosine similarity of af_spectra[i] to fluor_spectra[j]
    sim = _row_normalise(af_spectra) @ _row_normalise(fluor_spectra).T
    contaminated = (sim > cosine_threshold).any(axis=1)
    n_removed = contaminated.sum()
    if n_removed:
        logger.warning(
            f'get_af_spectra: removed {n_removed} AF spectrum/spectra '
            f'with cosine similarity > {cosine_threshold} to a fluorophore '
            f'(likely control contamination in unstained sample).'
        )
    return af_spectra[~contaminated]


def _filter_contaminant_events(
    event_mat: np.ndarray,
    spectra_mat: np.ndarray,
    threshold: float = 0.99,
) -> np.ndarray:
    """
    Return a boolean mask (True = keep) for events whose cosine similarity
    to every fluorophore spectrum is below threshold.

    More sensitive than post-clustering centroid QC: a small number of
    contaminating events will not dominate an entire cluster node.

    Parameters
    ----------
    event_mat   : ndarray (n_events, n_channels)
    spectra_mat : ndarray (n_fluors, n_channels)
    threshold   : float

    Returns
    -------
    ndarray, bool, shape (n_events,)
    """
    event_norms = np.sqrt(np.sum(event_mat ** 2, axis=1)) + 1e-9  # (n_events,)
    keep = np.ones(len(event_mat), dtype=bool)

    for spec in spectra_mat:
        spec_norm = np.sqrt(np.dot(spec, spec)) + 1e-9
        dots = event_mat @ spec          # (n_events,)
        cs = dots / (event_norms * spec_norm)
        keep &= cs < threshold
        if not keep.any():
            break

    return keep


def get_af_spectra(
    unstained_raw: np.ndarray,
    fluor_spectra: np.ndarray,
    som_dim: int = 10,
    min_cells: int = 100,
    random_state: int = 42,
    deduplicate: bool = True,
    duplication_threshold: float = 0.995,
    refine: bool = True,
    problem_quantile: float = 0.99,
    k_neighbors: int = 15,
    refine_improvement_threshold: float = 0.005,
    refine_min_shift_n: int = 8,
    remove_contaminants: bool = True,
    contaminant_threshold: float = 0.99,
) -> np.ndarray:
    """
    Identify AF spectral profiles from an unstained sample. Port of
    AutoSpectral's get.af.spectra().

    Stage 1 — Base spectra
    ----------------------
    Events whose background-subtracted cosine similarity to any fluorophore
    reaches contaminant_threshold are dropped. The remaining events are
    clustered in raw + OLS-unmixed space with a som_dim x som_dim SOM
    (KMeans fallback, see get_som_codes). Node codes are L-inf normalised,
    their mean is prepended as row 0, spectra resembling a fluorophore are
    removed, and near-duplicates are collapsed by cosine similarity.

    Stage 2 — Refine
    ----------------
    Every event is assigned an AF variant (assign_af_joint_l2) and unmixed.
    Cells whose post-correction fluorophore L2 norm is above problem_quantile
    are "problem cells". They are grouped by the pattern of their
    detector-space error (normalised by AF abundance) with a second SOM, and
    each group's cells seed a k_neighbors nearest-neighbour search across the
    whole population, so a candidate spectrum is built from a density-boosted
    set rather than from the sparse seeds alone. A candidate is appended only
    when, added to the library, the per-cell solver moves at least
    refine_min_shift_n of its seeds onto it and those cells' cosine
    similarity to their assigned spectrum improves: median gain at least
    refine_improvement_threshold and 25th-percentile gain above zero, paired
    per cell.

    Parameters
    ----------
    unstained_raw : ndarray, shape (n_cells, n_channels)
        Raw fluorescence channel data from the unstained control.
    fluor_spectra : ndarray, shape (n_fluors, n_channels)
        L-infinity-normalised fluorophore spectra (from spectral model).
    som_dim : int
        SOM grid side length; up to som_dim**2 base spectra before QC and
        deduplication. Shrunk automatically below 500 events.
    min_cells : int
        Minimum events required after contaminant filtering.
    random_state : int
        Random seed for reproducibility.
    deduplicate : bool
        Whether to collapse near-identical base spectra.
    duplication_threshold : float
        Cosine similarity at or above which a spectrum counts as a duplicate
        of one already kept; also the novelty bar for refine candidates.
    refine : bool
        Whether to run the refinement stage.
    problem_quantile : float
        Quantile of post-correction fluorophore L2 norm defining problem
        cells; stepped down by 0.05 until at least 500 cells qualify.
    k_neighbors : int
        Nearest neighbours recruited per seed cell for each candidate.
    refine_improvement_threshold : float
        Minimum median paired cosine gain for a candidate to be accepted.
    refine_min_shift_n : int
        Minimum seed cells that must switch to a candidate before it is
        evaluated.
    remove_contaminants : bool
        Whether to filter fluorophore-like events and spectra.
    contaminant_threshold : float
        Cosine similarity to a fluorophore for the per-event filter (>=) and
        for rejecting refine candidates (>=).

    Returns
    -------
    ndarray, shape (n_af, n_channels)
        Row 0 is the population mean of the base spectra; subsequent rows
        are the retained base spectra followed by any accepted refine
        spectra.
    """
    unstained_raw = np.asarray(unstained_raw, dtype=np.float64)
    fluor_spectra = np.asarray(fluor_spectra, dtype=np.float64)
    n_channels = unstained_raw.shape[1]

    # -------------------------------------------------------------------------
    # Stage 1 — Base spectra
    # -------------------------------------------------------------------------

    # Per-event contaminant filter, on mean-background-subtracted data so the
    # check targets contamination spikes rather than the baseline AF shape.
    if remove_contaminants:
        unstained_orth = unstained_raw - unstained_raw.mean(axis=0)[np.newaxis, :]
        keep = _filter_contaminant_events(unstained_orth, fluor_spectra, contaminant_threshold)
        n_removed = int((~keep).sum())
        if n_removed > 0:
            logger.info(
                f'get_af_spectra: removed {n_removed} event(s) prior to clustering '
                f'(cosine similarity >= {contaminant_threshold} to a fluorophore spectrum '
                f'on background-subtracted data)'
            )
            unstained_raw = unstained_raw[keep]

    n_cells = unstained_raw.shape[0]
    if n_cells < min_cells:
        raise ValueError(
            f'Insufficient cells in unstained sample: {n_cells} < {min_cells}. '
            f'Provide a larger unstained control.'
        )
    if n_cells < 500:
        som_dim = max(2, int(np.floor(np.sqrt(n_cells / 3))))

    # OLS unmix without AF — used as additional clustering features
    P = np.linalg.solve(fluor_spectra @ fluor_spectra.T, fluor_spectra)
    unmixed_no_af = unstained_raw @ P.T
    cluster_input = np.concatenate([unstained_raw, unmixed_no_af], axis=1)

    codes, engine = get_som_codes(cluster_input, som_dim, dist=2, random_state=random_state)
    logger.info(f'get_af_spectra: {som_dim}x{som_dim} codebook via {engine}')

    # L-infinity normalise; codes with no signal are dropped
    centres_spectral = codes[:, :n_channels]
    peak_vals = np.abs(centres_spectral).max(axis=1)
    usable = peak_vals > 1e-12
    af_candidates = centres_spectral[usable] / peak_vals[usable, np.newaxis]
    af_candidates = af_candidates[~np.isnan(af_candidates).any(axis=1)]

    # Prepend the population mean of all node spectra
    mean_af = af_candidates.mean(axis=0)
    mean_peak = np.abs(mean_af).max()
    if mean_peak > 1e-12:
        mean_af = mean_af / mean_peak
    af_spectra = np.vstack([mean_af[np.newaxis, :], af_candidates])

    # Contamination QC: remove any spectrum resembling a fluorophore
    if remove_contaminants:
        af_spectra = _qc_af_spectra(af_spectra, fluor_spectra, contaminant_threshold)
        if len(af_spectra) == 0:
            raise ValueError(
                'All AF candidate spectra were removed by contamination QC. '
                'Check whether the unstained sample contains single-stained events.'
            )

    if deduplicate:
        n_before = len(af_spectra)
        af_spectra = _deduplicate_spectra(af_spectra, duplication_threshold)
        logger.info(
            f'get_af_spectra: {len(af_spectra)} base spectra retained after '
            f'deduplication (dropped {n_before - len(af_spectra)})'
        )

    logger.info(f'get_af_spectra: {af_spectra.shape[0]} spectra after stage 1')

    # -------------------------------------------------------------------------
    # Stage 2 — Refine
    # -------------------------------------------------------------------------

    if refine:
        af_spectra = _refine_af_spectra(
            unstained_raw, fluor_spectra, af_spectra,
            problem_quantile=problem_quantile,
            k_neighbors=k_neighbors,
            improvement_threshold=refine_improvement_threshold,
            min_shift_n=refine_min_shift_n,
            duplication_threshold=duplication_threshold,
            remove_contaminants=remove_contaminants,
            contaminant_threshold=contaminant_threshold,
            random_state=random_state,
        )

    logger.info(f'get_af_spectra: returning {af_spectra.shape[0]} AF spectra total')
    return af_spectra


def _refine_af_spectra(
    unstained_raw: np.ndarray,
    fluor_spectra: np.ndarray,
    af_spectra: np.ndarray,
    problem_quantile: float,
    k_neighbors: int,
    improvement_threshold: float,
    min_shift_n: int,
    duplication_threshold: float,
    remove_contaminants: bool,
    contaminant_threshold: float,
    random_state: int,
) -> np.ndarray:
    """
    Refinement stage of get_af_spectra(): discover AF spectra for cells the
    base library under-corrects, keeping only candidates the per-cell solver
    demonstrably prefers. Returns the (possibly extended) library.
    """
    from sklearn.neighbors import NearestNeighbors

    # First-pass per-cell AF assignment and unmixing
    precomputed = precompute_af_matrices(fluor_spectra, af_spectra)
    precomputed.update(precompute_joint_cov_extras(precomputed, af_spectra))
    first_pass = apply_af_unmixing(unstained_raw, precomputed, af_spectra)

    unmixed_fluors = first_pass['unmixed']
    af_abundance   = first_pass['af_scale']
    af_assign      = first_pass['af_idx'].astype(np.int64) - 1

    # Detector-space error: raw minus fitted AF, i.e. residual plus the
    # fluorophore projection — all of it is error in an unstained sample.
    error = unstained_raw - af_abundance[:, np.newaxis] * af_spectra[af_assign]

    # Problem cells: still furthest from zero after correction. Step the
    # quantile down in 5% increments until at least 500 cells qualify.
    error_magnitude = np.sqrt(np.sum(unmixed_fluors ** 2, axis=1))
    pq = problem_quantile
    while True:
        threshold   = np.quantile(error_magnitude, pq)
        problem_idx = np.where(error_magnitude > threshold)[0]
        if len(problem_idx) >= 500:
            break
        pq -= 0.05
        if pq < 0.5:
            threshold   = np.quantile(error_magnitude, pq)
            problem_idx = np.where(error_magnitude > threshold)[0]
            break
    problem_n = len(problem_idx)

    logger.info(
        f'get_af_spectra refine: {problem_n} problem cells selected '
        f'(quantile = {pq:.2f}, threshold = {threshold:.2f})'
    )
    if problem_n <= 10:
        logger.info('get_af_spectra refine: insufficient problem cells - skipping.')
        return af_spectra

    # Group problem cells by their error pattern (spill ratios)
    af_abundance_problem = af_abundance[problem_idx].copy()
    af_abundance_problem[af_abundance_problem == 0] = 1e-6
    spill_ratios = error[problem_idx] / af_abundance_problem[:, np.newaxis]

    som_dim_error = min(10, max(2, int(np.floor(np.sqrt(problem_n / 3)))))
    codes_error, _engine = get_som_codes(
        spill_ratios, som_dim_error, dist=2, random_state=random_state,
    )
    sq_dist = (
        np.sum(spill_ratios ** 2, axis=1)[:, np.newaxis]
        - 2.0 * spill_ratios @ codes_error.T
        + np.sum(codes_error ** 2, axis=1)[np.newaxis, :]
    )
    error_assign = np.argmin(sq_dist, axis=1)
    _, first_seen = np.unique(error_assign, return_index=True)
    cluster_ids = error_assign[np.sort(first_seen)]

    # Density boost: each seed recruits its nearest neighbours (unit-length
    # spectra) from the whole population.
    pool_unit = _row_normalise(unstained_raw)
    k_eff = min(int(k_neighbors), len(pool_unit))
    nn_index = NearestNeighbors(n_neighbors=k_eff).fit(pool_unit).kneighbors(
        pool_unit[problem_idx], return_distance=False
    )

    accepted_n = 0
    for cl in cluster_ids:
        cl_sub_idx = np.where(error_assign == cl)[0]
        seed_idx   = problem_idx[cl_sub_idx]
        enriched_idx = np.unique(np.concatenate([seed_idx, nn_index[cl_sub_idx].ravel()]))

        candidate = pool_unit[enriched_idx].mean(axis=0)
        peak = np.abs(candidate).max()
        if peak <= 1e-12:
            continue
        candidate = candidate / peak
        candidate_unit = candidate / np.linalg.norm(candidate)

        # Cheap novelty and contamination filters before the solver check
        if np.max(_row_normalise(af_spectra) @ candidate_unit) >= duplication_threshold:
            continue
        if remove_contaminants and \
                np.max(_row_normalise(fluor_spectra) @ candidate_unit) >= contaminant_threshold:
            continue

        # Does the solver prefer this candidate for its own seeds?
        trial_spectra = np.vstack([af_spectra, candidate[np.newaxis, :]])
        candidate_row = trial_spectra.shape[0] - 1
        trial_assign = assign_af_joint_l2(unstained_raw[seed_idx], fluor_spectra, trial_spectra)
        shifted = np.where(trial_assign == candidate_row)[0]
        if len(shifted) < min_shift_n:
            continue

        # Paired before/after cosine similarity of each switching cell to its
        # assigned spectrum
        shifted_global  = seed_idx[shifted]
        raw_shifted     = unstained_raw[shifted_global]
        before_spectrum = af_spectra[af_assign[shifted_global]]
        raw_norm = np.linalg.norm(raw_shifted, axis=1)

        before_denom = raw_norm * np.linalg.norm(before_spectrum, axis=1)
        after_denom  = raw_norm * np.linalg.norm(candidate)
        with np.errstate(invalid='ignore', divide='ignore'):
            before_cos = np.where(
                before_denom > 0,
                np.einsum('ij,ij->i', raw_shifted, before_spectrum) / before_denom,
                np.nan,
            )
            after_cos = np.where(after_denom > 0, (raw_shifted @ candidate) / after_denom, np.nan)
        delta = after_cos - before_cos
        if not np.isfinite(delta).any():
            continue
        median_delta = float(np.nanmedian(delta))
        q25_delta    = float(np.nanquantile(delta, 0.25))

        # The typical switching cell must clear the margin, and the worse-off
        # quarter must still gain, so a few large gains cannot carry the rest.
        if median_delta < improvement_threshold or q25_delta <= 0:
            continue

        af_spectra = trial_spectra
        accepted_n += 1
        logger.info(
            f'get_af_spectra refine: group {cl} accepted - {len(shifted)}/{len(seed_idx)} '
            f'seed cells shifted (n={len(enriched_idx)} enriched), median cosine gain '
            f'{median_delta:.4f}.'
        )

    if accepted_n > 0:
        if remove_contaminants:
            af_spectra = _qc_af_spectra(af_spectra, fluor_spectra, contaminant_threshold)
        logger.info(
            f'get_af_spectra refine: {af_spectra.shape[0]} total AF spectra '
            f'after discovery and QC'
        )
    else:
        logger.info('get_af_spectra refine: no candidate spectra cleared validation - nothing appended.')

    return af_spectra
