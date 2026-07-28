import numpy as np
import pandas as pd
from functools import lru_cache
from pathlib import Path

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

from honeychrome.settings import experiments_folder, library_file

base_directory = Path.home() / experiments_folder

_CYTOMETER_TO_CSV = {
    'Aurora':         'Aurora_spectral_reference_library.csv',
    'NorthernLights': 'Aurora_spectral_reference_library.csv',
    'Discover':       'Discover_spectral_reference_library.csv',
    'ID7000':         'ID7000_spectral_reference_library.csv',
    'Opteon':         'Opteon_spectral_reference_library.csv',
    'Mosaic':         'Mosaic_spectral_reference_library.csv',
    'Xenith':         'Xenith_spectral_reference_library.csv',
    'A5SE':           'Symphony_spectral_reference_library.csv',
}

@lru_cache(maxsize=8)
def load_reference_library(cytometer_key: str) -> pd.DataFrame | None:
    """Load and cache spectral reference CSV; rows normalised to [0, 1]."""
    csv_name = _CYTOMETER_TO_CSV.get(cytometer_key)
    if csv_name is None:
        return None
    data_dir = Path(__file__).parent.parent / 'data'
    path = data_dir / csv_name
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0)
    row_max = df.max(axis=1).replace(0, np.nan)
    return df.div(row_max, axis=0).fillna(0)


def cosine_similarity_to_reference(
    spectrum: np.ndarray,
    channel_names: list[str],
    fluorophore: str,
    cytometer_key: str,
) -> float | None:
    ref = load_reference_library(cytometer_key)
    if ref is None:
        return None
    if fluorophore not in ref.index:
        return None
    common = [c for c in channel_names if c in ref.columns]
    if not common:
        return None
    v_exp = np.array([spectrum[channel_names.index(c)] for c in common], dtype=float)
    v_ref = ref.loc[fluorophore, common].values.astype(float)  # type: ignore[index]
    denom = (np.linalg.norm(v_exp) * np.linalg.norm(v_ref)) + 1e-9
    cs = float(np.dot(v_exp, v_ref) / denom)
    return cs


# ---------------------------------------------------------------------------
# Curated reference library (SQLite store)
#
# See _local_docs/SPECTRAL_REFERENCE_LIBRARY_PLAN. A ReferenceProfile is one
# fluorophore's spectral fingerprint on one instrument configuration. The store
# backs both Spectral Process (pick a spectrum to unmix with, exact config match)
# and Spectral QC (compare a measured control to a target, loose cytometer match).
# ---------------------------------------------------------------------------

CONFIG_KEY_UNKNOWN_CYTOMETER = 'unknown'


def compute_config_key(cytometer_key: str | None, channel_names: list[str]) -> str:
    """Stable identifier for an exact fluorescence-channel configuration.

    The channel names are **sorted** before hashing, so the key is independent
    of detector ordering (two runs with the same detectors in a different order
    get the same key). This is safe because a profile is stored as a
    ``{channel_name: value}`` dict, so unmixing re-aligns by name, not position.
    (NOTE: deviates from the original proposal, which hashed the raw ordered
    list — flagged for Oliver.)
    """
    key = cytometer_key or CONFIG_KEY_UNKNOWN_CYTOMETER
    normalised = sorted(c.strip() for c in channel_names)
    channel_sig = hashlib.sha1('|'.join(normalised).encode()).hexdigest()[:12]
    return f'{key}::{channel_sig}::{len(normalised)}ch'


@dataclass
class ReferenceProfile:
    """In-memory representation of one saved reference spectrum."""
    id: int | None                       # None until persisted
    fluorophore: str                     # dye/fluorophore identity — matching key
    display_name: str                    # user-editable label shown in the UI
    origin: str                          # 'honeychrome' | 'user'
    cytometer_key: str                   # instrument family, e.g. 'Aurora'
    config_key: str                      # exact channel-config identity
    channel_names: list[str]             # ordered PNN list — keys of profile
    profile: dict[str, float]            # channel name -> normalised intensity [0, 1]
    gate_channel: str | None = None      # peak channel, for display only
    source_sample_name: str | None = None
    source_experiment_dir: str | None = None
    notes: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0
    is_deletable: bool = True            # False for origin == 'honeychrome'
    is_reference: bool = False           # "the" profile for (fluorophore, config_key)
    is_qc_target: bool = False           # "the" profile for (fluorophore, cytometer_key)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS reference_library_profiles (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    fluorophore             TEXT    NOT NULL,
    display_name            TEXT    NOT NULL,
    origin                  TEXT    NOT NULL CHECK (origin IN ('honeychrome', 'user')),
    cytometer_key           TEXT    NOT NULL,
    config_key              TEXT    NOT NULL,
    channel_names_json      TEXT    NOT NULL,
    profile_json            TEXT    NOT NULL,
    gate_channel            TEXT,
    source_sample_name      TEXT,
    source_experiment_dir   TEXT,
    notes                   TEXT,
    created_at              REAL    NOT NULL,
    updated_at              REAL    NOT NULL,
    is_deletable            INTEGER NOT NULL DEFAULT 1,
    is_reference            INTEGER NOT NULL DEFAULT 0,
    is_qc_target            INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_ref_lib_config
    ON reference_library_profiles (config_key, fluorophore);
CREATE INDEX IF NOT EXISTS idx_ref_lib_cytometer
    ON reference_library_profiles (cytometer_key, fluorophore);
"""


class SpectralReferenceLibrary:
    """CRUD store for curated reference spectra (shares the SpectralLibrary DB)."""

    def __init__(self, library_path: Path | None = None):
        # same DB file as SpectralLibrary; overridable for tests.
        self.library_path = library_path or (base_directory / library_file)

    # --- connection helper ---------------------------------------------------
    def _open_connection(self) -> sqlite3.Connection:
        self.library_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.library_path)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _connect(self):
        """Yield a connection, commit on success, roll back on error, always close."""
        conn = self._open_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def ensure_honeychrome_rows_populated(self) -> None:
        """Ingest the bundled reference CSVs as ``origin='honeychrome'`` rows.

        Idempotent: a shipped row is inserted only if one does not already exist
        for that ``(fluorophore, cytometer_key)``. Each shipped row is the QC
        target for its ``(fluorophore, cytometer_key)`` by default, but only when
        no QC target exists yet — so re-running never clobbers a user override.
        """
        self.ensure_schema()
        now = time.time()
        with self._connect() as conn:
            for cytometer_key in _CYTOMETER_TO_CSV:
                df = load_reference_library(cytometer_key)
                if df is None:
                    continue
                channel_names = [str(c) for c in df.columns]
                config_key = compute_config_key(cytometer_key, channel_names)
                for fluor, row in df.iterrows():
                    fluor = str(fluor)
                    exists = conn.execute(
                        'SELECT 1 FROM reference_library_profiles '
                        'WHERE fluorophore = ? AND cytometer_key = ? AND origin = ? LIMIT 1',
                        (fluor, cytometer_key, 'honeychrome'),
                    ).fetchone()
                    if exists:
                        continue
                    profile = {c: float(row[c]) for c in channel_names}
                    has_qc = conn.execute(
                        'SELECT 1 FROM reference_library_profiles '
                        'WHERE fluorophore = ? AND cytometer_key = ? AND is_qc_target = 1 LIMIT 1',
                        (fluor, cytometer_key),
                    ).fetchone()
                    conn.execute(
                        """INSERT INTO reference_library_profiles
                           (fluorophore, display_name, origin, cytometer_key, config_key,
                            channel_names_json, profile_json, gate_channel, source_sample_name,
                            source_experiment_dir, notes, created_at, updated_at, is_deletable,
                            is_reference, is_qc_target)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,0,0,?)""",
                        (fluor, fluor, 'honeychrome', cytometer_key, config_key,
                         json.dumps(channel_names), json.dumps(profile), None, None,
                         None, None, now, now, 0 if has_qc else 1),
                    )

    def list_cytometer_keys(self) -> list[str]:
        """Distinct instrument families present in the library."""
        with self._connect() as conn:
            rows = conn.execute(
                'SELECT DISTINCT cytometer_key FROM reference_library_profiles ORDER BY cytometer_key'
            ).fetchall()
        return [r['cytometer_key'] for r in rows]

    # --- row <-> dataclass ---------------------------------------------------
    @staticmethod
    def _row_to_profile(row: sqlite3.Row) -> ReferenceProfile:
        return ReferenceProfile(
            id=row['id'],
            fluorophore=row['fluorophore'],
            display_name=row['display_name'],
            origin=row['origin'],
            cytometer_key=row['cytometer_key'],
            config_key=row['config_key'],
            channel_names=json.loads(row['channel_names_json']),
            profile=json.loads(row['profile_json']),
            gate_channel=row['gate_channel'],
            source_sample_name=row['source_sample_name'],
            source_experiment_dir=row['source_experiment_dir'],
            notes=row['notes'],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
            is_deletable=bool(row['is_deletable']),
            is_reference=bool(row['is_reference']),
            is_qc_target=bool(row['is_qc_target']),
        )

    # --- create --------------------------------------------------------------
    def save_profile(
        self,
        fluorophore: str,
        profile: dict[str, float],
        cytometer_key: str,
        config_key: str,
        channel_names: list[str],
        display_name: str | None = None,
        origin: str = 'user',
        gate_channel: str | None = None,
        source_sample_name: str | None = None,
        source_experiment_dir: str | None = None,
        notes: str | None = None,
        is_deletable: bool | None = None,
    ) -> ReferenceProfile:
        """The explicit, intentional save — replaces the old auto-deposit."""
        now = time.time()
        if is_deletable is None:
            is_deletable = origin != 'honeychrome'
        self.ensure_schema()
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO reference_library_profiles
                   (fluorophore, display_name, origin, cytometer_key, config_key,
                    channel_names_json, profile_json, gate_channel, source_sample_name,
                    source_experiment_dir, notes, created_at, updated_at, is_deletable,
                    is_reference, is_qc_target)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,0,0)""",
                (fluorophore, display_name or fluorophore, origin, cytometer_key, config_key,
                 json.dumps(list(channel_names)), json.dumps(profile), gate_channel,
                 source_sample_name, source_experiment_dir, notes, now, now, int(is_deletable)),
            )
            new_id = cur.lastrowid
        return self.get_profile(new_id)

    # --- read ----------------------------------------------------------------
    def get_profile(self, profile_id: int) -> ReferenceProfile | None:
        with self._connect() as conn:
            row = conn.execute(
                'SELECT * FROM reference_library_profiles WHERE id = ?', (profile_id,)
            ).fetchone()
        return self._row_to_profile(row) if row else None

    def list_profiles(self, cytometer_key: str) -> list[ReferenceProfile]:
        with self._connect() as conn:
            rows = conn.execute(
                'SELECT * FROM reference_library_profiles WHERE cytometer_key = ? '
                'ORDER BY fluorophore, display_name', (cytometer_key,)
            ).fetchall()
        return [self._row_to_profile(r) for r in rows]

    def list_profiles_for_config(self, config_key: str) -> list[ReferenceProfile]:
        with self._connect() as conn:
            rows = conn.execute(
                'SELECT * FROM reference_library_profiles WHERE config_key = ? '
                'ORDER BY fluorophore, display_name', (config_key,)
            ).fetchall()
        return [self._row_to_profile(r) for r in rows]

    def get_reference_for_config(self, fluorophore: str, config_key: str) -> ReferenceProfile | None:
        with self._connect() as conn:
            row = conn.execute(
                'SELECT * FROM reference_library_profiles '
                'WHERE fluorophore = ? AND config_key = ? AND is_reference = 1 LIMIT 1',
                (fluorophore, config_key),
            ).fetchone()
        return self._row_to_profile(row) if row else None

    def get_qc_target_for_cytometer(self, fluorophore: str, cytometer_key: str) -> ReferenceProfile | None:
        with self._connect() as conn:
            row = conn.execute(
                'SELECT * FROM reference_library_profiles '
                'WHERE fluorophore = ? AND cytometer_key = ? AND is_qc_target = 1 LIMIT 1',
                (fluorophore, cytometer_key),
            ).fetchone()
        return self._row_to_profile(row) if row else None

    # --- update --------------------------------------------------------------
    def rename_profile(self, profile_id: int, new_display_name: str) -> None:
        with self._connect() as conn:
            conn.execute(
                'UPDATE reference_library_profiles SET display_name = ?, updated_at = ? WHERE id = ?',
                (new_display_name, time.time(), profile_id),
            )

    def set_reference(self, profile_id: int, value: bool = True) -> None:
        """Make ``profile_id`` the reference for its (fluorophore, config_key)
        group (clearing any sibling first), or clear it when ``value`` is False."""
        now = time.time()
        with self._connect() as conn:
            row = conn.execute(
                'SELECT fluorophore, config_key FROM reference_library_profiles WHERE id = ?',
                (profile_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f'No reference profile with id {profile_id}')
            if value:
                conn.execute(
                    'UPDATE reference_library_profiles SET is_reference = 0, updated_at = ? '
                    'WHERE fluorophore = ? AND config_key = ?',
                    (now, row['fluorophore'], row['config_key']),
                )
            conn.execute(
                'UPDATE reference_library_profiles SET is_reference = ?, updated_at = ? WHERE id = ?',
                (1 if value else 0, now, profile_id),
            )

    def set_qc_target(self, profile_id: int, value: bool = True) -> None:
        """Make ``profile_id`` the QC target for its (fluorophore, cytometer_key)
        group (clearing any sibling first), or clear it when ``value`` is False."""
        now = time.time()
        with self._connect() as conn:
            row = conn.execute(
                'SELECT fluorophore, cytometer_key FROM reference_library_profiles WHERE id = ?',
                (profile_id,),
            ).fetchone()
            if row is None:
                raise ValueError(f'No reference profile with id {profile_id}')
            if value:
                conn.execute(
                    'UPDATE reference_library_profiles SET is_qc_target = 0, updated_at = ? '
                    'WHERE fluorophore = ? AND cytometer_key = ?',
                    (now, row['fluorophore'], row['cytometer_key']),
                )
            conn.execute(
                'UPDATE reference_library_profiles SET is_qc_target = ?, updated_at = ? WHERE id = ?',
                (1 if value else 0, now, profile_id),
            )

    # --- delete --------------------------------------------------------------
    def delete_profile(self, profile_id: int) -> None:
        with self._connect() as conn:
            row = conn.execute(
                'SELECT is_deletable FROM reference_library_profiles WHERE id = ?', (profile_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f'No reference profile with id {profile_id}')
            if not row['is_deletable']:
                raise ValueError('Honeychrome-origin profiles cannot be deleted')
            conn.execute('DELETE FROM reference_library_profiles WHERE id = ?', (profile_id,))
