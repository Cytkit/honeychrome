# label_matching.py
import re
import csv
from pathlib import Path
from importlib.resources import files   # Python 3.9+; use importlib_resources backport if needed

import logging
logger = logging.getLogger(__name__)


def _load_csv(path: Path) -> list[dict]:
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


_DELIM = r'(?<![A-Za-z0-9\-]){}(?![A-Za-z0-9\-])'

# Cache of precompiled (pattern, value, canonical) triples, keyed by
# (id(database), canonical_col, tuple(synonym_cols)) -- see
# _compiled_patterns(). Stable across calls because both call sites
# (drc_cluster_id.build_channel_marker_map, spectral_controller) fetch
# the database via get_marker_db()/get_fluorophore_db(), which each
# cache and return the SAME list object every time -- id() doesn't
# change between calls.
_pattern_cache: dict[tuple, list[tuple]] = {}


def _compiled_patterns(database: list[dict], canonical_col: str,
                        synonym_cols: list[str]) -> list[tuple]:
    """
    Precompile every (pattern, value, canonical) triple for this
    database/column-set ONCE, sorted longest-value-first so _best_match
    can return on the first hit. Building each row's regex (re.escape +
    string-format + re.compile) here instead of inside _best_match turns
    an O(channels x rows x cols) recompile storm into a one-time cost
    amortised across every call for this database.
    """
    key = (id(database), canonical_col, tuple(synonym_cols))
    cached = _pattern_cache.get(key)
    if cached is not None:
        return cached

    entries = []
    for row in database:
        canonical = row.get(canonical_col)
        for col in [canonical_col] + synonym_cols:
            val = row.get(col, '') or ''
            if not val:
                continue
            escaped = re.escape(val)
            escaped = escaped.replace(r'\ ', r'\s*')   # mirror R's gsub(" ", "\\s*", ...)
            pattern = re.compile(_DELIM.format(escaped), flags=re.IGNORECASE)
            entries.append((pattern, val, canonical))

    # Longest value first -- ties keep original row/column order via
    # Python's stable sort, matching the old strict len(val) > len(best_text)
    # tie-break (first max-length match wins, not last).
    entries.sort(key=lambda e: len(e[1]), reverse=True)
    _pattern_cache[key] = entries
    return entries


def _best_match(name: str, database: list[dict], canonical_col: str, synonym_cols: list[str]) -> str | None:
    """
    Return the canonical value from `canonical_col` for the longest synonym
    that appears as a word-boundary match inside `name`.
    Returns None if nothing matches.
    """
    for pattern, _val, canonical in _compiled_patterns(database, canonical_col, synonym_cols):
        if pattern.search(name):
            return canonical
    return None


def match_fluorophore(name: str, fluorophore_db: list[dict]) -> str | None:
    """Return the canonical fluorophore name, or None."""
    synonym_cols = [f'synonym{i}' for i in range(1, 5)]
    result = _best_match(name, fluorophore_db, 'fluorophore', synonym_cols)
    if result:
        logger.debug(f'Fluorophore match: "{name}" -> "{result}"')
    else:
        logger.debug(f'No fluorophore match for: "{name}"')
    return result


def match_marker(name: str, marker_db: list[dict]) -> str | None:
    """Return the canonical marker/antigen name, or None."""
    synonym_cols = [f'synonym{i}' for i in range(1, 10)]
    result = _best_match(name, marker_db, 'marker', synonym_cols)
    if result:
        logger.debug(f'Marker match: "{name}" -> "{result}"')
    else:
        logger.debug(f'No marker match for: "{name}"')
    return result

_DATA_DIR = Path(__file__).parent.parent / 'data'

_fluorophore_db: list[dict] | None = None
_marker_db: list[dict] | None = None


def get_fluorophore_db() -> list[dict]:
    global _fluorophore_db
    if _fluorophore_db is None:
        _fluorophore_db = _load_csv(_DATA_DIR / 'fluorophore_database.csv')
    return _fluorophore_db


def get_marker_db() -> list[dict]:
    global _marker_db
    if _marker_db is None:
        _marker_db = _load_csv(_DATA_DIR / 'marker_database.csv')
    return _marker_db