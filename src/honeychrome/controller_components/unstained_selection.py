"""Choice of the default unstained cell sample for the spectral model.

The unstained cell sample supplies the autofluorescence (AF) taken away from
every cell control, and the AF reference used to find each control's peak
channel, so it should hold the same kind of cells as the controls. When an
experiment has more than one, they can differ: a sample from another tissue
or preparation can be twice as bright as the controls' own unstained cells.

Every non-bead sample marked as unstained, or with "unstained" in its path or
name, is a candidate. Each candidate's dimmest cells are compared with those
of the cell single-stain controls: the quarter of base-gate events whose
brightest detector is lowest, which in a control are mostly its unstained
cells. The mismatch is the median over detectors of
|log2(candidate / reference)|, where the reference is the per-detector median
across all cell controls. A mismatch of 0.25 is about a 19% difference in
brightness; the controls themselves typically sit about 0.12 from the
reference. Among the candidates within MATCH_TOLERANCE, the one with the most
base-gate events is chosen, since more cells give a steadier AF estimate.
When none is within it, the closest is chosen and a warning is logged.

Events are read as a random preview of at most PREVIEW_EVENTS per file, which
gives the same scores as the whole files and keeps the comparison to a few
seconds; the number of base-gate events in a file is estimated from the
fraction of the preview inside the base gate.
"""
import logging
import re

import numpy as np

from honeychrome.controller_components.particle_type import is_bead_sample

logger = logging.getLogger(__name__)

MATCH_TOLERANCE = 0.25
PREVIEW_EVENTS = 20000

_UNSTAINED_NAME = re.compile(r'unstained', re.IGNORECASE)
_UNSTAINED_OR_NEGATIVE = re.compile(r'(unstained|negative)', re.IGNORECASE)


def cell_unstained_candidates(samples: dict) -> list:
    """Paths of the non-bead samples marked as unstained or named "unstained", in sample order."""
    marked = set(samples.get('unstained_samples', []))
    return [
        path for path, name in samples.get('all_samples', {}).items()
        if (path in marked or _UNSTAINED_NAME.search(path) or _UNSTAINED_NAME.search(name))
        and not is_bead_sample(samples, path, name)
    ]


def default_cell_unstained(samples: dict, process: dict) -> str | None:
    """Tube name of the default unstained cell sample: the one stored in
    process['default_cell_unstained'] while it is still a candidate, else the
    first candidate; None without candidates."""
    names = [samples['all_samples'][path] for path in cell_unstained_candidates(samples)]
    stored = process.get('default_cell_unstained')
    if stored in names:
        return stored
    return names[0] if names else None


def dim_cells_spectrum(events: np.ndarray) -> np.ndarray:
    """Per-detector median of the quarter of events with the lowest brightest-detector value."""
    order = np.argsort(events.max(axis=1))
    return np.median(events[order[:max(2, len(events) // 4)]], axis=0)


def af_mismatch(spectrum: np.ndarray, reference: np.ndarray) -> float:
    """Median over detectors of |log2(spectrum / reference)|, values below 1 taken as 1."""
    ratio = np.clip(spectrum, 1, None) / np.clip(reference, 1, None)
    return float(np.median(np.abs(np.log2(ratio))))


def choose_cell_unstained(rows: list) -> str | None:
    """Path chosen from select_cell_unstained's rows: the most base-gate events
    among those within MATCH_TOLERANCE, else the smallest mismatch. Without any
    mismatch (no cell controls could be read), the most base-gate events."""
    if not rows:
        return None
    scored = [row for row in rows if row['mismatch'] is not None]
    if not scored:
        return max(rows, key=lambda row: row['n_events'])['path']
    matched = [row for row in scored if row['mismatch'] <= MATCH_TOLERANCE]
    if matched:
        return max(matched, key=lambda row: row['n_events'])['path']
    closest = min(scored, key=lambda row: row['mismatch'])
    logger.warning(
        'No unstained cell sample matches the cell controls\' autofluorescence within %.2f; '
        'using the closest, "%s" (mismatch %.2f).', MATCH_TOLERANCE, closest['name'], closest['mismatch'])
    return closest['path']


def _reference_controls(samples: dict, candidates: list) -> list:
    """Paths of the cell single-stain controls: not beads, not marked as
    unstained and not named "unstained" or "negative"."""
    marked = set(samples.get('unstained_samples', [])) | set(candidates)
    all_samples = samples.get('all_samples', {})
    return [
        path for path in samples.get('single_stain_controls', [])
        if path in all_samples and path not in marked
        and not _UNSTAINED_OR_NEGATIVE.search(path)
        and not _UNSTAINED_OR_NEGATIVE.search(all_samples[path])
        and not is_bead_sample(samples, path, all_samples[path])
    ]


def _base_gate_events(controller, path: str, gating, base_gate: str, layout: dict):
    """Fluorescence of the base-gate events in a preview of *path*, and the
    estimated number of base-gate events in the whole file."""
    from honeychrome.controller_components.fcs_preview import load_fcs_preview
    from honeychrome.controller_components.functions import sample_from_fcs
    from honeychrome.controller_components.spectral_functions import get_raw_events

    full_path = str(controller.experiment_dir / path)
    try:
        sample, events = load_fcs_preview(full_path, PREVIEW_EVENTS, col_order=layout['col_order'])
        n_read, n_file = len(events), sample.event_count
    except ValueError:
        # integer FCS data: full read
        sample = sample_from_fcs(full_path)
        n_read = n_file = sample.event_count
    gated = get_raw_events(
        sample, layout['fl_ids'],
        gate_label=None if base_gate == 'root' else base_gate,
        gating_strategy=gating,
        col_order=layout['col_order'],
        sample_key=path,
    )
    n_estimate = int(round(len(gated) / n_read * n_file)) if n_read else 0
    return gated, n_estimate


def select_cell_unstained(controller, gating, base_gate: str):
    """Path of the default unstained cell sample and the rows it was chosen
    from (path, name, estimated base-gate events, mismatch). With one
    candidate it is returned without reading any file; (None, []) without
    candidates. *gating* is the raw SampleGating, *base_gate* the gate the
    spectral controls are drawn from."""
    from honeychrome.controller_components.spread_model import raw_layout

    samples = controller.experiment.samples
    candidates = cell_unstained_candidates(samples)
    if len(candidates) <= 1:
        return (candidates[0] if candidates else None), []

    layout = raw_layout(controller)
    reference_spectra = []
    for path in _reference_controls(samples, candidates):
        try:
            events, _ = _base_gate_events(controller, path, gating, base_gate, layout)
        except Exception as e:
            logger.warning(f'select_cell_unstained: could not read control "{path}": {e}')
            continue
        if len(events) >= 2:
            reference_spectra.append(dim_cells_spectrum(events))
    reference = np.median(reference_spectra, axis=0) if reference_spectra else None

    rows = []
    for path in candidates:
        row = {'path': path, 'name': samples['all_samples'][path], 'n_events': 0, 'mismatch': None}
        try:
            events, row['n_events'] = _base_gate_events(controller, path, gating, base_gate, layout)
            if reference is not None and len(events) >= 2:
                row['mismatch'] = af_mismatch(dim_cells_spectrum(events), reference)
        except Exception as e:
            logger.warning(f'select_cell_unstained: could not read unstained "{path}": {e}')
        rows.append(row)

    chosen = choose_cell_unstained(rows)
    summary = '; '.join(
        f'"{row["name"]}" {row["n_events"]} events, mismatch '
        + ('n/a' if row['mismatch'] is None else f'{row["mismatch"]:.2f}')
        for row in rows)
    logger.info(f'select_cell_unstained: {len(reference_spectra)} cell controls as reference; {summary}; '
                f'chose "{samples["all_samples"].get(chosen, chosen)}"')
    return chosen, rows
