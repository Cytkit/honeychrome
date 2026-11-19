"""
plate_layout.py
---------------
Where each sample sat on its plate and when it was acquired, and which
neighbouring wells could have contaminated an unstained sample.

Contamination of an unstained well by single-stain controls has two
physical routes:

* spatial: liquid or cells moving between adjacent wells (a splash, a
  misaimed pipette tip), so the 8 wells around it;
* carryover: the sampling probe, so the well acquired immediately before
  it, when it was acquired shortly before.

Plate position and acquisition time come from the FCS keywords the
instrument writes ($WELLID or $LOCATIONID, $PLATENAME, PLATEROWS,
PLATECOLS, $DATE, $BTIM), with the well read from a leading file-name
prefix such as "A1 " when the keyword is missing. Tube racks (layouts of
fewer than 96 positions) give no candidates: separate tubes neither splash
into each other nor carry over appreciably.

A neighbour contributes the dyes it holds: a spectral-model control its
fluorophore; another sample the panel dyes named in its file name; a
sample with no recognisable dye (a fully stained sample) the whole panel;
an unstained sample nothing.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

MIN_PLATE_POSITIONS = 96          # smaller layouts are tube racks
CARRYOVER_MAX_GAP_S = 15 * 60     # previous acquisition counts within this gap
WHOLE_PANEL = '<panel>'
_WELL_PREFIX = re.compile(r'^([A-P])(\d{1,2})(?=[\s_-])')


@dataclass(frozen=True)
class WellInfo:
    well: str | None
    plate: str | None
    rows: int
    cols: int
    start: datetime | None

    @property
    def row_col(self) -> tuple[int, int] | None:
        m = re.fullmatch(r'([A-P])(\d{1,2})', self.well or '')
        return (ord(m.group(1)) - 65, int(m.group(2)) - 1) if m else None

    @property
    def is_plate(self) -> bool:
        return self.rows * self.cols >= MIN_PLATE_POSITIONS


@dataclass
class Candidates:
    """Dyes that neighbouring wells could have put into a sample."""
    spatial: list = field(default_factory=list)      # sample keys of adjacent wells
    carryover: list = field(default_factory=list)    # sample key acquired just before
    single: list = field(default_factory=list)       # dyes from single-stain neighbours
    whole_panel: bool = False                        # a neighbour holds the whole panel
    sources: dict = field(default_factory=dict)      # dye -> neighbour sample keys

    def describe(self, names: dict | None = None) -> str:
        names = names or {}
        parts = [f'{dye} ({", ".join(Path(names.get(k, k)).stem for k in keys)})'
                 for dye, keys in sorted(self.sources.items())]
        return '; '.join(parts)


def _start_time(text: dict) -> datetime | None:
    date, btim = text.get('date', ''), (text.get('btim', '') or '')[:8]
    for fmt in ('%d-%b-%Y %H:%M:%S', '%Y-%m-%d %H:%M:%S', '%d-%m-%Y %H:%M:%S'):
        try:
            return datetime.strptime(f'{date} {btim}', fmt)
        except ValueError:
            continue
    return None


def _int(value) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return 0


def read_well_info(abs_path) -> WellInfo:
    """Plate position and start time of one FCS file (TEXT segment only)."""
    from honeychrome.controller_components.fcs_preview import read_preview_metadata
    try:
        text = read_preview_metadata(str(abs_path)).text
    except Exception as exc:
        logger.warning('plate_layout: %s not read (%s)', abs_path, exc)
        text = {}
    well = (text.get('wellid') or text.get('locationid') or '').strip().upper() or None
    if well is None:
        m = _WELL_PREFIX.match(Path(abs_path).name)
        well = f'{m.group(1)}{int(m.group(2))}' if m else None
    return WellInfo(well=well, plate=(text.get('platename') or '').strip() or None,
                    rows=_int(text.get('platerows')), cols=_int(text.get('platecols')),
                    start=_start_time(text))


def read_layout(experiment_dir, sample_keys) -> dict:
    """{sample key: WellInfo} for every readable sample."""
    out = {}
    for key in sample_keys:
        path = Path(experiment_dir) / key
        if path.is_file():
            out[key] = read_well_info(path)
    return out


def sample_dyes(key: str, sample_name: str, panel: list, control_labels: dict, unstained: set) -> list:
    """Dyes a sample can contribute: its control label, else panel dyes named in its file name, else the panel."""
    if key in unstained:
        return []
    if key in control_labels:
        return [control_labels[key]]
    from honeychrome.controller_components.label_matching import get_fluorophore_db, match_fluorophore
    stem = re.sub(r'^[A-P]\d{1,2}\s+', '', Path(sample_name or key).stem)
    found = [d for d in panel
             if re.search(r'(?<![A-Za-z0-9])' + re.escape(d) + r'(?![A-Za-z0-9])', stem, re.IGNORECASE)]
    if not found:
        db = get_fluorophore_db()
        for token in re.split(r'[\s_()]+', stem):
            match = match_fluorophore(token, db)
            if match in panel and match not in found:
                found.append(match)
    return found or [WHOLE_PANEL]


def neighbour_candidates(key: str, layout: dict, panel: list, control_labels: dict, unstained: set,
                         sample_names: dict | None = None) -> Candidates:
    """
    Candidate contaminating dyes of sample ``key``: from the 8 adjacent wells
    of the same plate and from the sample acquired immediately before it
    (within CARRYOVER_MAX_GAP_S). Empty for tube racks and for samples
    without a plate position.
    """
    out = Candidates()
    info = layout.get(key)
    if info is None or not info.is_plate:
        return out
    rc = info.row_col
    for other, o in layout.items():
        if other == key or o.plate != info.plate or (o.rows, o.cols) != (info.rows, info.cols):
            continue
        orc = o.row_col
        if rc is not None and orc is not None and max(abs(rc[0] - orc[0]), abs(rc[1] - orc[1])) == 1:
            out.spatial.append(other)
    if info.start is not None:
        earlier = [(o.start, other) for other, o in layout.items()
                   if other != key and o.start is not None and o.start < info.start]
        if earlier:
            t, prev = max(earlier)
            if (info.start - t).total_seconds() <= CARRYOVER_MAX_GAP_S:
                out.carryover.append(prev)
    names = sample_names or {}
    for other in sorted(set(out.spatial + out.carryover)):
        for dye in sample_dyes(other, names.get(other, other), panel, control_labels, unstained):
            if dye == WHOLE_PANEL:
                out.whole_panel = True
            elif dye not in out.single:
                out.single.append(dye)
            out.sources.setdefault(dye, []).append(other)
    out.single.sort()
    return out
