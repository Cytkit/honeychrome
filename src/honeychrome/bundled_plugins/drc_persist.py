"""
drc_persist.py — Pickle-free storage for the DR/Clustering plugin
=================================================================
Companion module to ``dr_clustering_tab.py`` (filename intentionally does
NOT end in ``_tab.py``, so it is not picked up as a separate plugin tab).

Stores one payload dict as a *bundle* directory:

    <bundle>/
        data.json      the payload's structure: every scalar, string, list,
                       tuple and dict inline, with placeholders for arrays
                       and tables
        arrays.npz     every numpy array, under positional keys (a0, a1, …),
                       so no sample path (which contains '/') is ever used
                       as an archive member name
        tables/*.csv   one CSV per pandas DataFrame; its index, column
                       labels and dtypes are recorded in data.json so it
                       reads back exactly

Nothing is unpickled on load: arrays are read with ``allow_pickle=False``
and data.json uses a few tagged objects for what JSON lacks:

    {"__tuple__": [...]}                 tuple
    {"__items__": [[key, value], ...]}   dict with non-string keys
    {"__float__": "nan" | "inf" | "-inf"}
    {"__na__": null}                     pandas.NA
    {"__ndarray__": "a0"}                array stored in arrays.npz
    {"__objarray__": {...}}              object-dtype array, stored inline
    {"__dataframe__": {...}}             table stored under tables/

A bundle is written to a sibling ``<name>.tmp`` directory and swapped into
place, so an interrupted save leaves the previous bundle readable.
"""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from drc_logging import get_logger

log = get_logger(__name__)

FORMAT_NAME = 'honeychrome-drc-bundle'
FORMAT_VERSION = 1

_DATA_FILE = 'data.json'
_ARRAYS_FILE = 'arrays.npz'
_TABLES_DIR = 'tables'


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_bundle(directory: Path, payload: dict,
                skip_unstorable: bool = False) -> list[str]:
    """
    Write *payload* (a dict with string keys) to *directory*, replacing any
    bundle already there.

    skip_unstorable: when True, a top-level value containing a type this
        module cannot store is left out (and its key returned) instead of
        failing the whole save.

    Returns the list of skipped top-level keys.
    """
    directory = Path(directory)
    tmp = _sibling(directory, '.tmp')
    old = _sibling(directory, '.old')
    _remove(tmp)
    tmp.mkdir(parents=True)
    try:
        enc = _Encoder()
        tree: dict = {}
        skipped: list[str] = []
        for key, value in payload.items():
            if not isinstance(key, str):
                raise TypeError(f"top-level payload keys must be str, got {key!r}")
            mark = enc.mark()
            try:
                tree[key] = enc.encode(value, key)
            except TypeError as exc:
                if not skip_unstorable:
                    raise
                enc.rollback(mark)
                skipped.append(key)
                log.warning("bundle %s: skipping '%s' (%s)", directory.name, key, exc)
        enc.write(tmp)
        with open(tmp / _DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump({'format': FORMAT_NAME, 'version': FORMAT_VERSION, 'payload': tree},
                      f, indent=1, allow_nan=False, ensure_ascii=False)
    except Exception:
        _remove(tmp)
        raise

    _remove(old)
    if directory.exists():
        directory.rename(old)
    tmp.rename(directory)
    _remove(old)
    return skipped


def load_bundle(directory: Path) -> dict | None:
    """
    Read a bundle written by save_bundle(). Returns None if there is no
    bundle at *directory*; raises on a damaged or unrecognised one.
    """
    directory = Path(directory)
    _recover(directory)
    data_path = directory / _DATA_FILE
    if not data_path.exists():
        return None
    with open(data_path, 'r', encoding='utf-8') as f:
        doc = json.load(f)
    if doc.get('format') != FORMAT_NAME or doc.get('version') != FORMAT_VERSION:
        raise ValueError(f"unrecognised bundle format in {data_path}: "
                         f"{doc.get('format')!r} v{doc.get('version')!r}")
    arrays: dict[str, np.ndarray] = {}
    npz_path = directory / _ARRAYS_FILE
    if npz_path.exists():
        with np.load(npz_path, allow_pickle=False) as npz:
            arrays = {name: npz[name] for name in npz.files}
    return _Decoder(directory, arrays).decode(doc['payload'])


def bundle_exists(directory: Path) -> bool:
    """True if a complete bundle is (or can be recovered) at *directory*."""
    directory = Path(directory)
    return ((directory / _DATA_FILE).exists()
            or (_sibling(directory, '.tmp') / _DATA_FILE).exists())


def delete_bundle(directory: Path) -> None:
    directory = Path(directory)
    for path in (directory, _sibling(directory, '.tmp'), _sibling(directory, '.old')):
        _remove(path)


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def _sibling(directory: Path, suffix: str) -> Path:
    return directory.with_name(directory.name + suffix)


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def _recover(directory: Path) -> None:
    """Finish a swap interrupted between save_bundle()'s two renames."""
    if directory.exists():
        return
    for candidate in (_sibling(directory, '.tmp'), _sibling(directory, '.old')):
        if (candidate / _DATA_FILE).exists():
            candidate.rename(directory)
            log.info("bundle %s: recovered from %s", directory.name, candidate.name)
            return


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def _flat_label(value) -> str:
    """Single-line CSV header text for an index/column label."""
    if isinstance(value, tuple):
        return ' | '.join(str(v) for v in value)
    return '' if value is None else str(value)


def _table_stem(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', name).strip('._')[:60] or 'table'


class _Encoder:
    """Turns a payload into a JSON tree, collecting arrays and tables."""

    def __init__(self):
        self.arrays: dict[str, np.ndarray] = {}
        self.tables: list[tuple[str, pd.DataFrame]] = []

    def mark(self) -> tuple[int, int]:
        return len(self.arrays), len(self.tables)

    def rollback(self, mark: tuple[int, int]) -> None:
        n_arrays, n_tables = mark
        for key in list(self.arrays)[n_arrays:]:
            del self.arrays[key]
        del self.tables[n_tables:]

    def write(self, directory: Path) -> None:
        if self.arrays:
            np.savez(directory / _ARRAYS_FILE, **self.arrays)
        if self.tables:
            tables_dir = directory / _TABLES_DIR
            tables_dir.mkdir()
            for stem, df in self.tables:
                _write_csv(df, tables_dir / f'{stem}.csv')

    def encode(self, obj, name: str):
        if obj is None or isinstance(obj, (bool, str)):
            return obj
        if obj is pd.NA:
            return {'__na__': None}
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (int, np.integer)):
            return int(obj)
        if isinstance(obj, (float, np.floating)):
            value = float(obj)
            if math.isfinite(value):
                return value
            return {'__float__': 'nan' if math.isnan(value) else ('inf' if value > 0 else '-inf')}
        if isinstance(obj, np.ndarray):
            if obj.dtype.kind == 'O':
                return {'__objarray__': {
                    'shape': list(obj.shape),
                    'values': [self.encode(v, name) for v in obj.ravel().tolist()],
                }}
            key = f'a{len(self.arrays)}'
            self.arrays[key] = obj
            return {'__ndarray__': key}
        if isinstance(obj, pd.DataFrame):
            return {'__dataframe__': self._table(obj, name)}
        if isinstance(obj, tuple):
            return {'__tuple__': [self.encode(v, name) for v in obj]}
        if isinstance(obj, list):
            return [self.encode(v, name) for v in obj]
        if isinstance(obj, dict):
            if all(isinstance(k, str) and not k.startswith('__') for k in obj):
                return {k: self.encode(v, k) for k, v in obj.items()}
            return {'__items__': [[self.encode(k, name), self.encode(v, name)]
                                  for k, v in obj.items()]}
        raise TypeError(f"cannot store a {type(obj).__module__}.{type(obj).__qualname__}")

    def _table(self, df: pd.DataFrame, name: str) -> dict:
        used = {stem for stem, _ in self.tables}
        base = _table_stem(name)
        stem, n = base, 2
        while stem in used:
            stem, n = f'{base}_{n}', n + 1
        schema = {
            'file': f'{_TABLES_DIR}/{stem}.csv',
            'n_rows': int(len(df)),
            'index': self._axis(df.index),
            'columns': self._axis(df.columns),
            'dtypes': [self._column(df.iloc[:, pos]) for pos in range(df.shape[1])],
        }
        self.tables.append((stem, df))
        return schema

    def _axis(self, axis: pd.Index) -> dict:
        names = [self.encode(n, 'label') for n in axis.names]
        if isinstance(axis, pd.RangeIndex):
            return {'range': [axis.start, axis.stop, axis.step], 'names': names}
        return {
            'values': [self.encode(v, 'label') for v in axis.tolist()],
            'names': names,
            'nlevels': int(axis.nlevels),
            'dtype': str(axis.dtype) if axis.nlevels == 1 else None,
        }

    def _column(self, col: pd.Series) -> dict:
        dtype = col.dtype
        spec: dict = {'dtype': str(dtype)}
        if isinstance(dtype, pd.CategoricalDtype):
            spec['categories'] = [self.encode(v, 'category') for v in dtype.categories.tolist()]
            spec['ordered'] = bool(dtype.ordered)
            spec['codes'] = col.cat.codes.tolist()
        elif isinstance(dtype, np.dtype) and dtype.kind in 'biuf':
            pass  # read back from the CSV text
        elif isinstance(dtype, pd.StringDtype):
            spec['nulls'] = np.flatnonzero(col.isna().to_numpy()).tolist()
        elif dtype == object and all(isinstance(v, str) for v in col.tolist()):
            spec['nulls'] = []
        else:
            spec['values'] = [self.encode(v, 'value') for v in col.tolist()]
        return spec


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    flat = df.copy(deep=False)
    flat.columns = [_flat_label(c) for c in df.columns]
    flat.index = pd.Index([_flat_label(v) for v in df.index],
                          name=' | '.join(_flat_label(n) for n in df.index.names))
    flat.to_csv(path, index=True)


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

class _Decoder:
    def __init__(self, directory: Path, arrays: dict[str, np.ndarray]):
        self.directory = directory
        self.arrays = arrays

    def decode(self, obj):
        if isinstance(obj, list):
            return [self.decode(v) for v in obj]
        if not isinstance(obj, dict):
            return obj
        if len(obj) == 1:
            (tag, body), = obj.items()
            if tag == '__ndarray__':
                return self.arrays[body]
            if tag == '__tuple__':
                return tuple(self.decode(v) for v in body)
            if tag == '__items__':
                return {self.decode(k): self.decode(v) for k, v in body}
            if tag == '__float__':
                return float(body)
            if tag == '__na__':
                return pd.NA
            if tag == '__objarray__':
                values = [self.decode(v) for v in body['values']]
                arr = np.empty(len(values), dtype=object)
                for i, v in enumerate(values):
                    arr[i] = v
                return arr.reshape(body['shape'])
            if tag == '__dataframe__':
                return self._table(body)
        return {k: self.decode(v) for k, v in obj.items()}

    def _axis(self, spec: dict) -> pd.Index:
        names = [self.decode(n) for n in spec['names']]
        if 'range' in spec:
            return pd.RangeIndex(*spec['range'], name=names[0])
        values = [self.decode(v) for v in spec['values']]
        if spec['nlevels'] > 1:
            if not values:
                return pd.MultiIndex.from_arrays([[] for _ in names], names=names)
            return pd.MultiIndex.from_tuples(values, names=names)
        try:
            return pd.Index(values, name=names[0], dtype=spec['dtype'])
        except (TypeError, ValueError):
            return pd.Index(values, name=names[0])

    def _table(self, schema: dict) -> pd.DataFrame:
        n_rows = schema['n_rows']
        specs = schema['dtypes']
        text_cols: list[list[str]] = [[] for _ in specs]
        if n_rows and any(_reads_text(s) for s in specs):
            raw = pd.read_csv(self.directory / schema['file'], header=0, dtype=str,
                              keep_default_na=False, na_filter=False)
            if raw.shape != (n_rows, len(specs) + 1):
                raise ValueError(f"{schema['file']}: expected {n_rows} x {len(specs) + 1} "
                                 f"cells, found {raw.shape[0]} x {raw.shape[1]}")
            # column 0 holds the index labels, which come from the schema instead
            text_cols = [raw.iloc[:, pos + 1].tolist() for pos in range(len(specs))]
        data = {pos: self._column(spec, text_cols[pos])
                for pos, spec in enumerate(specs)}
        df = pd.DataFrame(data, index=pd.RangeIndex(n_rows))
        df.columns = self._axis(schema['columns'])
        df.index = self._axis(schema['index'])
        return df

    def _column(self, spec: dict, text: list[str]) -> pd.Series:
        dtype_name = spec['dtype']
        if 'codes' in spec:
            categories = [self.decode(v) for v in spec['categories']]
            return pd.Series(pd.Categorical.from_codes(
                spec['codes'], categories=categories, ordered=spec['ordered']))
        if 'values' in spec:
            series = pd.Series([self.decode(v) for v in spec['values']], dtype=object)
            if dtype_name == 'object':
                return series
            try:
                return series.astype(dtype_name)
            except (TypeError, ValueError):
                return series
        if 'nulls' in spec:
            values: list = list(text)
            for row in spec['nulls']:
                values[row] = None
            return pd.Series(values, dtype=dtype_name)
        dtype = np.dtype(dtype_name)
        if dtype.kind == 'f':
            arr = np.array([float(t) if t else np.nan for t in text], dtype=dtype)
        elif dtype.kind in 'iu':
            arr = np.array([int(t) for t in text], dtype=dtype)
        else:
            arr = np.array([t == 'True' for t in text], dtype=dtype)
        return pd.Series(arr)


def _reads_text(spec: dict) -> bool:
    return 'codes' not in spec and 'values' not in spec
