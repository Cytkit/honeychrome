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

# Cache of precompiled (pattern, value, canonical) triples and of the
# matches already found, keyed by (id(database), canonical_col,
# tuple(synonym_cols)) -- see _compiled_patterns(). Each entry keeps the
# database it was built from, so a list that reuses a freed list's id is
# not served the old patterns. The databases from get_marker_db() /
# get_fluorophore_db() are cached and returned as the SAME list every
# time, so their entries are built once per session.
_pattern_cache: dict[tuple, tuple] = {}
# Matches remembered per database and column set; cleared when larger.
_MATCH_CACHE_SIZE = 50_000


def _cache_entry(database: list[dict], canonical_col: str, synonym_cols: list[str]) -> tuple:
    """(database, patterns, {name: canonical or None}) for this database and
    column set, built on first use."""
    key = (id(database), canonical_col, tuple(synonym_cols))
    cached = _pattern_cache.get(key)
    if cached is not None and cached[0] is database:
        return cached
    entry = (database, _build_patterns(database, canonical_col, synonym_cols), {})
    _pattern_cache[key] = entry
    return entry


def _compiled_patterns(database: list[dict], canonical_col: str,
                        synonym_cols: list[str]) -> list[tuple]:
    """
    Every (pattern, value, canonical) triple for this database/column-set,
    precompiled ONCE and sorted longest-value-first so _best_match can
    return on the first hit.
    """
    return _cache_entry(database, canonical_col, synonym_cols)[1]


def _build_patterns(database: list[dict], canonical_col: str,
                    synonym_cols: list[str]) -> list[tuple]:
    """Compile the (pattern, value, canonical) triples of a database:
    building each row's regex (re.escape + string-format + re.compile)
    once instead of inside _best_match turns an O(channels x rows x cols)
    recompile storm into a one-time cost."""

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
    return entries


def _best_match(name: str, database: list[dict], canonical_col: str, synonym_cols: list[str]) -> str | None:
    """
    Return the canonical value from `canonical_col` for the longest synonym
    that appears as a word-boundary match inside `name`.
    Returns None if nothing matches. A name matched before against the
    same database and columns is answered from the cache.
    """
    _db, patterns, matches = _cache_entry(database, canonical_col, synonym_cols)
    if name in matches:
        return matches[name]
    result = None
    for pattern, _val, canonical in patterns:
        if pattern.search(name):
            result = canonical
            break
    if len(matches) >= _MATCH_CACHE_SIZE:
        matches.clear()
    matches[name] = result
    return result


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