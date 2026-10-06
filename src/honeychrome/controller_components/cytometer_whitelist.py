# cytometer_whitelist.py
#
# Cytometer identification and fluorescence channel whitelisting.
#
# Mirrors the logic in get_cytometer_param.R / .CYTOMETER_PARAMS:
#   1. Match $CYT (and CREATOR as fallback) to a known cytometer entry.
#   2. Derive fluorescence detector column names from the full $PnN list using
#      a two-stage filter:
#        - Exclusion pass  : remove any channel whose name matches a non-spectral pattern.
#        - Positive pass   : (FACSDiscover only) additionally require the name to match
#                            a spectral pattern such as "UV3 (420)-A".
#   3. Order the surviving channels against cytometer_database.csv.
#
# Returns None for unrecognised cytometers so the caller can fall back to
# flowio's own classification.

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd
from PySide6.QtGui import QColor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-cytometer parameter table  (mirrors .CYTOMETER_PARAMS in R)
# ---------------------------------------------------------------------------

@dataclass
class _CytParams:
    cyt_label: str
    cyt_kw_pattern: str           # regex applied to $CYT  (case-insensitive)
    creator_pattern: Optional[str]  # regex applied to CREATOR (case-insensitive); None = not used
    non_spectral_pat: list[str]   # channel name substrings/patterns to EXCLUDE
    spectral_pat: Optional[str]   # positive regex that surviving channels must match (FACSDiscover only)
    db_col: str                   # column name in cytometer_database.csv
    scatter_param: list[str]      # canonical scatter parameters
    sat_value: int                # max fluorescence value (saturation threshold)
    scatter_extra_pat: list[str]   # additional channel name patterns to include as scatter
    singlet_y_preference: str      # 'FSC-W' or 'FSC-H'
    scatter_display_ceiling: Optional[dict[str, float]] = None  # per-channel display limit override; None = use PNR
    db_col_bare_names: bool = False  # True when cytometer_database.csv stores this cytometer's detector names WITHOUT the -A/-H/-W suffix (CytoStellar only)


_CYTOMETER_PARAMS: dict[str, _CytParams] = {

    "Aurora": _CytParams(
        cyt_label       = "Cytek Aurora",
        cyt_kw_pattern  = r"^Aurora$",
        creator_pattern = r"SpectroFlo",
        non_spectral_pat = ["FSC", "SSC", "Time"],
        spectral_pat    = None,
        db_col          = "Aurora",
        scatter_param = ["FSC-A", "SSC-A"],
        sat_value = 4194304,
        scatter_extra_pat = ["SSC-B"],
        singlet_y_preference = "FSC-H",
    ),

    "NorthernLights": _CytParams(
        cyt_label       = "Cytek Northern Lights",
        cyt_kw_pattern  = r"^Aurora$",          # same $CYT as Aurora; distinguished by UV presence
        creator_pattern = r"SpectroFlo",
        non_spectral_pat = ["FSC", "SSC", "Time"],
        spectral_pat    = None,
        db_col          = "NorthernLights",
        scatter_param = ["FSC-A", "SSC-A"],
        sat_value = 4194304,
        scatter_extra_pat = ["SSC-B"],
        singlet_y_preference = "FSC-H",
    ),

    "ID7000": _CytParams(
        cyt_label       = "Sony ID7000",
        cyt_kw_pattern  = r"ID7000",
        creator_pattern = None,
        non_spectral_pat = ["FSC", "SSC", "TIME"],
        spectral_pat    = None,
        db_col          = "ID7000",
        scatter_param = ["FSC-A", "SSC-A"],
        sat_value = 1048576,
        scatter_extra_pat = [],
        singlet_y_preference = "FSC-H",
    ),

    "FACSDiscover": _CytParams(
        cyt_label       = "BD FACSDiscover (S8 / A8)",
        cyt_kw_pattern  = r"FACSDiscover",
        creator_pattern = r"FACSDiva|FACSSuite",
        non_spectral_pat = [
            "FSC", "SSC", "Time", "LightLoss", "Delta", "Plate",
            "Radial", "Correlation", "Intensity", "Eccentricity",
            "Diffusivity", "Center", "Moment", "Size", "Saturated",
            "Sorted", "Row", "Column", "Img", "Protocol", "EventLabel",
            "Region", "Gate", "Index", "Phase", "Event", "Drop",
            "Spectral", "Waveform", "Merged", "Flow", "Packet", "Reserved",
            r"-T$",
        ],
        spectral_pat    = r"\([0-9]+\)-A$",     # must match e.g. "UV3 (420)-A"
        db_col          = "Discover",
        scatter_param = ["LightLoss (Violet)-A", "SSC (Imaging)-A"],
        sat_value = 24140237,
        scatter_extra_pat = [r"^LightLoss", r"^FSC", r"^SSC \(Imaging\)"],
        singlet_y_preference = "LightLoss (Violet)-H",
        scatter_display_ceiling = {
            "FSC-A": 1e8, "SSC (Violet)-A": 3e7, "SSC (Imaging)-A": 3e7, "LightLoss (Violet)-A": 1e8, "LightLoss (Violet)-H": 1e5
        },
    ),

    "Opteon": _CytParams(
        cyt_label       = "Agilent NovoCyte Opteon",
        cyt_kw_pattern  = r"Opteon",
        creator_pattern = r"NovoExpress|Opteon",
        non_spectral_pat = ["FSC", "SSC", "VSSC", "Time", "Width"],
        spectral_pat    = None,
        db_col          = "Opteon",
        scatter_param = ["FSC-A", "VSSC-A"],
        sat_value = 16777216,
        scatter_extra_pat = ["FSC", "SSC"],
        singlet_y_preference = "FSC-H",
    ),

    "Mosaic": _CytParams(
        cyt_label       = "Beckman Coulter CytoFLEX Mosaic",
        cyt_kw_pattern  = r"mosaic",
        creator_pattern = r"CytExpert",
        non_spectral_pat = ["FSC", "SSC", "BSSC", "Time"],
        spectral_pat    = None,
        db_col          = "Mosaic",
        scatter_param = ["FSC-A", "BSSC-A"],
        sat_value = 10000,
        scatter_extra_pat = ["FSC", "SSC"],
        singlet_y_preference = "FSC-H",
        scatter_display_ceiling = {"FSC-A": 1e5, "BSSC-A": 1e5, "FSC-H": 1e5},
    ),

    "Xenith": _CytParams(
        cyt_label       = "ThermoFisher Attune Xenith",
        cyt_kw_pattern  = r"Xenith",
        creator_pattern = r"Attune|VitesseSQ",
        non_spectral_pat = ["FSC", "SSC", "Time", "Event", "Gate", "Sort",
                             "Comp"],
        spectral_pat    = None,
        db_col          = "Xenith",
        scatter_param = ["FSC51-A", "SSC52-A"],
        sat_value = 100000,
        scatter_extra_pat = ["FSC", "SSC"],
        singlet_y_preference = "FSC51-H",
    ),

    "Symphony": _CytParams(
        cyt_label       = "BD FACSymphony A5 SE",
        cyt_kw_pattern  = r"FACSymphony",
        creator_pattern = r"FACSDiva|FACSSuite",
        non_spectral_pat = ["FSC", "SSC", "Time"],
        spectral_pat    = None,
        db_col          = "A5SE",
        scatter_param = ["FSC-A", "SSC-A"],
        sat_value = 262144,
        scatter_extra_pat = [],
        singlet_y_preference = "FSC-H",
    ),

    "Bigfoot": _CytParams(
        cyt_label       = "Thermo Fisher Bigfoot",
        cyt_kw_pattern  = r"Bigfoot",
        creator_pattern = None,
        non_spectral_pat = [
            "FSC", "SSC", "Time", "Clock", "GateMatch", "SortIndex",
            "DropsSorted", "SortDestination",
            r"^Comp-",  # conventional-mode compensated channel
            r"-Comp$",  # spectral-mode unmixed channel (incl. stray FSC07-Comp)
        ],
        spectral_pat    = None,
        db_col          = "Bigfoot",
        scatter_param = ["FSC07-A", "SSC58-A"],
        sat_value = 100000,
        scatter_extra_pat = ["FSC56", "FSC57", "SSC59"],
        singlet_y_preference = "FSC07-H",
    ),

    "CytoStellar": _CytParams(
        cyt_label       = "CytoStellar",
        # NOTE: current CytoStellar firmware writes a "default" placeholder
        # (Chinese "默认", not literal ASCII "default") into $CYT instead of
        # an instrument name; engineers expect this fixed in a future
        # release. This pattern matches once that ships -- the literal
        # placeholder is matched separately in _match_cytometer() below as
        # an interim workaround (confirmed value from Oliver, 2026-09-27).
        cyt_kw_pattern  = r"CytoStellar",
        creator_pattern = None,
        non_spectral_pat = ["FSC", "SSC", "Time"],
        spectral_pat    = None,
        db_col          = "CytoStellar",
        scatter_param = ["FSC-A", "BSSC-A"],
        sat_value = 4000000,
        scatter_extra_pat = ["FSC", "SSC"],
        singlet_y_preference = "FSC-H",
        scatter_display_ceiling = {"FSC-A": 4e6, "BSSC-A": 4e6, "FSC-H": 4e6},
        # cytometer_database.csv's CytoStellar column stores bare
        # (unsuffixed) detector names -- the actual -A/-H suffix is chosen
        # per acquisition, and a detector acquired as BOTH -A and -H is
        # deliberately NOT deduplicated here: that choice belongs to
        # experiment.process['fluorescence_channel_filter'] ('area_only' vs
        # not) via Controller.filter_raw_fluorescence_channels(), which
        # already does -A-only filtering downstream of this module.
        db_col_bare_names = True,
    ),
}

# Path to the bundled cytometer_database.csv
_DB_PATH = Path(__file__).parent.parent / "data" / "cytometer_database.csv"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class CytometerInfo:
    """Returned by resolve_cytometer_params()."""
    cyt_label: str                   # human-readable name
    db_col: str                      # column in cytometer_database.csv
    fluorescence_channel_ids: list[int]  # indices into the full $PnN list
    scatter_param: list[str]
    sat_value: int
    scatter_extra_pat: list[str]
    singlet_y_preference: str
    scatter_display_ceiling: dict[str, float]


def resolve_cytometer_params(
    all_pnn: list[str],
    text_keywords: dict[str, str],
) -> Optional[CytometerInfo]:
    """
    Identify the cytometer and derive the fluorescence channel indices.

    Parameters
    ----------
    all_pnn : list[str]
        Full ordered list of $PnN channel names from the FCS file (as returned
        by FlowData.pnn_labels).
    text_keywords : dict[str, str]
        The FCS TEXT segment as a flat dict (FlowData.text).  Keys are used
        case-insensitively.

    Returns
    -------
    CytometerInfo  if the cytometer is recognised, otherwise None.
    """
    # Normalise: uppercase keys, also build a $-stripped version for lookup
    kw_upper = {k.upper(): v for k, v in text_keywords.items()}
    kw_noprefix = {k.lstrip("$").upper(): v for k, v in text_keywords.items()}

    def _get_kw(*names: str) -> str:
        for name in names:
            v = kw_upper.get(name.upper()) or kw_noprefix.get(name.lstrip("$").upper())
            if v and str(v).strip():
                return str(v).strip()
        return ""

    cyt_kw           = _get_kw("$CYT", "CYT")
    creator_combined = f"{_get_kw('CREATOR', '$CREATOR')}".strip()

    matched = _match_cytometer(all_pnn, cyt_kw, creator_combined)
    if matched is None:
        return None

    detector_names = _derive_detector_cols(all_pnn, matched)
    detector_names = _order_detectors(detector_names, matched.db_col, bare_names=matched.db_col_bare_names)

    # Map back to indices in the original all_pnn list
    name_to_idx = {name: i for i, name in enumerate(all_pnn)}
    fl_ids = [name_to_idx[name] for name in detector_names if name in name_to_idx]

    scatter_param = _resolve_scatter(all_pnn, matched.scatter_param)

    return CytometerInfo(
        cyt_label=matched.cyt_label,
        db_col=matched.db_col,
        fluorescence_channel_ids=fl_ids,
        scatter_param=scatter_param,
        sat_value=matched.sat_value,
        scatter_extra_pat=matched.scatter_extra_pat,
        singlet_y_preference=matched.singlet_y_preference,
        scatter_display_ceiling=matched.scatter_display_ceiling or {},
    )


def singlet_channels(cytometer: str) -> Optional[tuple[str, str]]:
    """
    Canonical forward-scatter singlet pair for a cytometer.

    Parameters
    ----------
    cytometer : str
        The cytometer label stored in ``experiment.settings['raw']['cytometer']``
        (e.g. "Cytek Aurora") or its registry key (e.g. "Aurora").

    Returns
    -------
    (area_channel, singlet_y_channel), e.g. ("FSC-A", "FSC-H"), or None when
    the cytometer is not recognised.
    """
    name = str(cytometer or "").strip()
    params = _CYTOMETER_PARAMS.get(name)
    if params is None:
        params = next((p for p in _CYTOMETER_PARAMS.values() if p.cyt_label == name), None)
    if params is None or not params.scatter_param:
        return None
    return params.scatter_param[0], params.singlet_y_preference


# ---------------------------------------------------------------------------
# Internal matching logic  (mirrors .match_cytometer in R)
# ---------------------------------------------------------------------------

def _match_cytometer(
    all_pnn: list[str],
    cyt_kw: str,
    creator_combined: str,
) -> Optional[_CytParams]:

    def kw_match(pattern: str) -> bool:
        return bool(re.search(pattern, cyt_kw, re.IGNORECASE))

    def creator_match(pattern: str) -> bool:
        return bool(re.search(pattern, creator_combined, re.IGNORECASE))

    # --- Priority 1: $CYT-based matching ---

    if kw_match(r"ID7000"):
        return _CYTOMETER_PARAMS["ID7000"]

    if kw_match(r"FACSDiscover"):
        return _CYTOMETER_PARAMS["FACSDiscover"]

    if kw_match(r"FACSymphony"):
        return _CYTOMETER_PARAMS["Symphony"]

    if kw_match(r"Opteon"):
        return _CYTOMETER_PARAMS["Opteon"]

    if kw_match(r"mosaic"):
        return _CYTOMETER_PARAMS["Mosaic"]

    if kw_match(r"Xenith"):
        return _CYTOMETER_PARAMS["Xenith"]

    if kw_match(r"Bigfoot"):
        return _CYTOMETER_PARAMS["Bigfoot"]

    if kw_match(r"CytoStellar"):
        return _CYTOMETER_PARAMS["CytoStellar"]

    # Current-firmware bug workaround: $CYT is written as a "default"
    # placeholder in Chinese (默认: default / tacit consent / take for
    # granted) instead of an instrument name. Distinctive enough to match
    # directly until the firmware fix ships.
    if cyt_kw.strip() == "默认":
        return _CYTOMETER_PARAMS["CytoStellar"]

    # Aurora / NL share $CYT = "Aurora"; distinguished by presence of UV channels
    if re.match(r"^Aurora$", cyt_kw, re.IGNORECASE):
        has_uv = any(re.match(r"^UV[0-9]+-A$", ch) for ch in all_pnn)
        return _CYTOMETER_PARAMS["Aurora"] if has_uv else _CYTOMETER_PARAMS["NorthernLights"]

    # --- Priority 2: CREATOR-based fallback ---

    if creator_match(r"FACSDiva|FACSSuite"):
        has_discover_style = any(re.search(r"\([0-9]+\)-A$", ch) for ch in all_pnn)
        return _CYTOMETER_PARAMS["FACSDiscover"] if has_discover_style else _CYTOMETER_PARAMS["Symphony"]

    if creator_match(r"SpectroFlo"):
        has_uv = any(re.match(r"^UV[0-9]+-A$", ch) for ch in all_pnn)
        return _CYTOMETER_PARAMS["Aurora"] if has_uv else _CYTOMETER_PARAMS["NorthernLights"]

    if creator_match(r"NovoExpress"):
        return _CYTOMETER_PARAMS["Opteon"]

    if creator_match(r"CytExpert"):
        return _CYTOMETER_PARAMS["Mosaic"]

    if creator_match(r"Attune|VitesseSQ|Xenith"):
        return _CYTOMETER_PARAMS["Xenith"]

    return None  # unrecognised — caller falls back to flowio


# ---------------------------------------------------------------------------
# Internal derivation logic  (mirrors .derive_detector_cols in R)
# ---------------------------------------------------------------------------

def _derive_detector_cols(all_pnn: list[str], params: _CytParams) -> list[str]:
    """Apply exclusion (and optionally positive) filters to the $PnN list."""

    excl_pattern = "|".join(params.non_spectral_pat)

    cols = [
        ch for ch in all_pnn
        if ch and not re.search(excl_pattern, ch)
    ]

    # Positive pass — FACSDiscover only
    if params.spectral_pat:
        cols = [ch for ch in cols if re.search(params.spectral_pat, ch)]
        if not cols:
            logger.error(
                "No spectral detector columns survived the positive-match filter "
                "(pattern '%s'). The FCS file may contain only pre-unmixed channels.",
                params.spectral_pat,
            )

    return cols


def _order_detectors(detector_names: list[str], db_col: str, bare_names: bool = False) -> list[str]:
    """
    Re-order detector_names to match the canonical excitation/emission order
    defined in cytometer_database.csv.  Channels not present in the database
    are appended at the end in their original order.

    bare_names : when True, the database column stores detector names
        WITHOUT an acquisition-parameter suffix (-A/-H/-W) -- currently
        CytoStellar only, where the suffix actually acquired varies per
        file and both suffixes may legitimately be present for the same
        detector. Ordering then matches on each channel's suffix-stripped
        name; channels sharing a bare name (e.g. both "V7-A" and "V7-H")
        stay adjacent, in their original relative order.
    """
    if not _DB_PATH.exists():
        logger.warning(
            "cytometer_database.csv not found at %s; "
            "detector ordering follows FCS acquisition order.",
            _DB_PATH,
        )
        return detector_names

    try:
        db = pd.read_csv(_DB_PATH)
        if db_col not in db.columns:
            logger.warning(
                "Column '%s' not found in cytometer_database.csv; "
                "detector ordering follows FCS acquisition order.",
                db_col,
            )
            return detector_names

        ref_order = [v for v in db[db_col].dropna() if str(v).strip()]

        if bare_names:
            suffix_re = re.compile(r"-(A|H|W)$")
            bare_lookup: dict[str, list[str]] = {}
            for ch in detector_names:
                bare_lookup.setdefault(suffix_re.sub('', ch), []).append(ch)
            ordered = []
            for ref in ref_order:
                ordered.extend(bare_lookup.get(ref, []))
            ordered_set = set(ordered)
            remainder = [ch for ch in detector_names if ch not in ordered_set]
            return ordered + remainder

        ordered   = [ch for ch in ref_order   if ch in set(detector_names)]
        remainder = [ch for ch in detector_names if ch not in set(ref_order)]
        return ordered + remainder

    except Exception as exc:
        logger.warning(
            "Could not read cytometer_database.csv for ordering: %s. "
            "Detector ordering follows FCS acquisition order.",
            exc,
        )
        return detector_names
    
def _resolve_scatter(all_pnn: list[str], canonical: list[str]) -> list[str]:
    """Use canonical scatter names if present; otherwise grep for FSC/SSC area channels."""
    if all(ch in all_pnn for ch in canonical):
        return canonical
    fallback = [ch for ch in all_pnn
                if re.match(r"^(FSC|SSC|BSSC|VSSC)", ch) and ch.endswith("-A")]
    if not fallback:
        logger.warning("No scatter channels found; scatter plots will be unavailable.")
    return fallback

# ---------------------------------------------------------------------------
# Laser colour map and detector-laser lookup  (shared by all plot widgets)
# ---------------------------------------------------------------------------

# Keys match the laser values found in the *_laser columns of cytometer_database.csv.
LASER_LABEL_COLORS: dict[str, QColor] = {
    'DeepUV':      QColor("#EDD5F6"),
    'UV':          QColor("#D886F9"),
    'Violet':      QColor("#7F00FF"),
    'Blue':        QColor("#328FE7"),
    'YellowGreen': QColor("#ACF312"),
    'Red':         QColor('#E74C3C'),
    'IR':          QColor('#A93226'),
}


def get_detector_laser_map(db_col: str) -> dict[str, str]:
    """
    Return a dict mapping detector channel name → laser name for the given
    cytometer database column (e.g. 'ID7000').

    Reads the <db_col>_laser column from cytometer_database.csv paired with
    the <db_col> column.  Returns an empty dict if the column is missing or
    the CSV cannot be read — callers treat that as "no colour information".
    """
    if not db_col or not _DB_PATH.exists():
        return {}
    laser_col = f"{db_col}_laser"
    try:
        db = pd.read_csv(_DB_PATH)
        if db_col not in db.columns or laser_col not in db.columns:
            logger.warning(
                "Column '%s' or '%s' not found in cytometer_database.csv; "
                "laser colour-coding will be skipped.",
                db_col, laser_col,
            )
            return {}
        return {
            str(row[db_col]): str(row[laser_col])
            for _, row in db.iterrows()
            if pd.notna(row[db_col]) and pd.notna(row[laser_col])
            and str(row[db_col]).strip() and str(row[laser_col]).strip()
        }
    except Exception as exc:
        logger.warning("Could not build detector–laser map: %s", exc)
        return {}