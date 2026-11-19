"""
sample_table.py — read and write per-sample tables of groups and covariates
==========================================================================

Shared by the plugins that assign samples to comparison groups (Automated
Gating, Honeycluster, and plugins built on Automated Gating's tables). A table
has one row per sample: a sample column, optionally a group column, and one
column per covariate (donor, age, sex, CRP).

Sample names in a table rarely match file names exactly. A study key names a
sample ``POMS_008_FB_D`` while the file is ``A1 POMS_008_FB_D_Plate 1.fcs``.
Rows are therefore matched in two steps:

1. Exactly (any case): the sample key, the file name, or the file name
   without its extension.
2. Inside the file name: the table name appears in the file name without
   its extension as a whole run of characters, bounded on each side by the
   start or end of the name or by a character that is not a letter or digit
   (space, ``_``, ``-``, ``.``). Only names of three or more characters
   holding a digit are matched this way, so a legend row such as ``D`` or
   ``Total`` never matches a sample. When several table names fit one file,
   the longest wins (``POMS_008_FB_D`` over ``POMS_008_FB``).

A row whose name fits several samples is applied to every one of them and
reported, so the user can see it before applying (the same donor acquired
twice, or a name that is too short to be specific).
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

SAMPLE_COLUMNS = ('sample', 'file', 'filename', 'name')
GROUP_COLUMNS = ('group', 'condition')
IGNORED_COLUMNS = ('full_path',)

EXACT = 'exact'
INSIDE = 'inside file name'

_MIN_INSIDE_LENGTH = 3


def sample_label(key: str) -> str:
    """The name a sample is shown by: its file name without extension."""
    return Path(str(key).replace('\\', '/')).stem


def _exact_names(key: str) -> list[str]:
    """Ways a table may name a sample exactly: its key, file name and stem (lower case)."""
    path = str(key).replace('\\', '/')
    name = path.rsplit('/', 1)[-1]
    stem = name.rsplit('.', 1)[0] if '.' in name else name
    return [path.lower(), name.lower(), stem.lower()]


def _matchable_inside(name: str) -> bool:
    return len(name) >= _MIN_INSIDE_LENGTH and any(ch.isdigit() for ch in name)


def _inside_pattern(name: str) -> re.Pattern:
    return re.compile(r'(?:^|[^a-z0-9])' + re.escape(name) + r'(?:$|[^a-z0-9])')


def read_table_file(path) -> pd.DataFrame:
    """A CSV, tab-separated or semicolon-separated table as text, with
    column names stripped of surrounding spaces. Rows with no value at all
    are dropped."""
    text = Path(path).read_text(encoding='utf-8-sig', errors='replace')
    first = text.splitlines()[0] if text else ''
    sep = max((',', '\t', ';'), key=first.count)
    df = pd.read_csv(path, sep=sep, dtype=str, encoding='utf-8-sig', encoding_errors='replace',
                     keep_default_na=False)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.apply(lambda col: col.str.strip())
    return df[(df != '').any(axis=1)].reset_index(drop=True)


def find_column(df: pd.DataFrame, candidates) -> str | None:
    """The first column of *df* whose name (any case) is one of *candidates*."""
    lower = {str(c).strip().lower(): c for c in df.columns}
    return next((lower[c] for c in candidates if c in lower), None)


@dataclass
class SampleTableMatch:
    """How a table's rows were matched to samples.

    ``rows`` holds, per table row index, the matched sample keys (empty when
    the row matched nothing), ``how`` the matching step per row, ``names``
    the row's sample name. ``unmatched`` lists rows' names with no sample,
    ``multiple`` the names matched to more than one sample, ``samples_without_row``
    the samples no row matched.
    """
    sample_column: str
    names: dict[int, str] = field(default_factory=dict)
    rows: dict[int, list[str]] = field(default_factory=dict)
    how: dict[int, str] = field(default_factory=dict)
    unmatched: list[str] = field(default_factory=list)
    multiple: list[str] = field(default_factory=list)
    samples_without_row: list[str] = field(default_factory=list)

    @property
    def matched_rows(self) -> list[int]:
        return [i for i, keys in self.rows.items() if keys]

    @property
    def matched_samples(self) -> list[str]:
        seen: dict[str, None] = {}
        for i in self.matched_rows:
            for key in self.rows[i]:
                seen.setdefault(key, None)
        return list(seen)


def match_table(df: pd.DataFrame, sample_keys: list[str],
                sample_column: str | None = None) -> SampleTableMatch:
    """Match each row of *df* to *sample_keys* (exactly, else by the name
    inside the file name; see the module docstring). Raises ValueError when
    the table has no sample column."""
    sample_column = sample_column or find_column(df, SAMPLE_COLUMNS)
    if sample_column is None or sample_column not in df.columns:
        raise ValueError("The table needs a 'sample' column naming each sample.")
    exact: dict[str, list[str]] = {}
    for key in sample_keys:
        for n in dict.fromkeys(_exact_names(key)):
            exact.setdefault(n, []).append(key)
    stems = {key: sample_label(key).lower() for key in sample_keys}

    out = SampleTableMatch(sample_column=sample_column)
    pending: dict[int, str] = {}
    for i, raw in df[sample_column].items():
        if pd.isna(raw) or not str(raw).strip():
            continue
        name = str(raw).strip()
        out.names[i] = name
        hit = next((exact[n] for n in _exact_names(name) if n in exact), None)
        if hit is not None:
            out.rows[i] = list(dict.fromkeys(hit))
            out.how[i] = EXACT
        else:
            out.rows[i] = []
            if _matchable_inside(name.lower()):
                pending[i] = name.lower()

    taken = {key for i in out.rows for key in out.rows[i]}
    patterns = {i: _inside_pattern(n) for i, n in pending.items()}
    for key, stem in stems.items():
        if key in taken:
            continue
        fits = [i for i, pat in patterns.items() if pat.search(stem)]
        if not fits:
            continue
        longest = max(len(pending[i]) for i in fits)
        for i in fits:
            if len(pending[i]) == longest:
                out.rows[i].append(key)
                out.how[i] = INSIDE

    out.unmatched = [out.names[i] for i in out.names if not out.rows[i]]
    out.multiple = [out.names[i] for i in out.names if len(out.rows[i]) > 1]
    matched = set(out.matched_samples)
    out.samples_without_row = [k for k in sample_keys if k not in matched]
    return out


def table_values(df: pd.DataFrame, match: SampleTableMatch, group_column: str | None,
                 covariate_columns: list[str], rows: list[int] | None = None) -> dict:
    """Groups and covariates for the matched samples.

    Only the given table *rows* (default every matched row) are used. Empty
    cells are skipped. Returns {'groups': {key: group}, 'covariates':
    {column: {key: value}}}; every requested covariate column is present,
    even when it holds no value.
    """
    rows = match.matched_rows if rows is None else rows
    groups: dict[str, str] = {}
    covs: dict[str, dict[str, str]] = {str(c): {} for c in covariate_columns}
    for i in rows:
        row = df.loc[i]
        for key in match.rows.get(i, []):
            if group_column is not None:
                g = row[group_column]
                if pd.notna(g) and str(g).strip():
                    groups[key] = str(g).strip()
            for c in covariate_columns:
                v = row[c]
                if pd.notna(v) and str(v).strip():
                    covs[str(c)][key] = str(v).strip()
    return {'groups': groups, 'covariates': covs}


def covariate_candidates(df: pd.DataFrame, match: SampleTableMatch,
                         group_column: str | None) -> list[str]:
    """Columns that may be imported as covariates: every column except the
    sample and group columns and a 'full_path' column."""
    skip = {match.sample_column, group_column}
    return [c for c in df.columns
            if c not in skip and str(c).strip().lower() not in IGNORED_COLUMNS]


def filled_columns(df: pd.DataFrame, match: SampleTableMatch, columns: list[str]) -> list[str]:
    """The *columns* holding at least one value in a matched row."""
    rows = match.matched_rows
    if not rows:
        return []
    sub = df.loc[rows, columns].fillna('').astype(str).apply(lambda col: col.str.strip())
    return [c for c in columns if (sub[c] != '').any()]


def table_frame(samples: list[str], groups: dict, covariates: dict,
                label=sample_label) -> pd.DataFrame:
    """One row per sample: 'sample' (the name the sample is shown by,
    ``label(key)``), 'group' and one column per covariate; the format
    match_table reads back."""
    rows = []
    for key in samples:
        row = {'sample': label(key), 'group': groups.get(key, '')}
        for name, values in (covariates or {}).items():
            row[name] = (values or {}).get(key, '')
        rows.append(row)
    return pd.DataFrame(rows, columns=['sample', 'group'] + list(covariates or {}))


def write_table(df: pd.DataFrame, path) -> None:
    """Write a sample table as CSV with every value quoted, so names holding
    spaces or separators stay in one cell whichever program opens it."""
    df.to_csv(path, index=False, quoting=csv.QUOTE_ALL)


def default_export_path(experiment_dir, plugin_label: str) -> str:
    """'<experiment folder>/<experiment name>_<plugin>_sample_table.csv'."""
    folder = Path(str(experiment_dir or '.'))
    stem = folder.name or 'experiment'
    slug = re.sub(r'[^A-Za-z0-9]+', '_', plugin_label).strip('_').lower()
    return str(folder / f"{stem}_{slug}_sample_table.csv")
