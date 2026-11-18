"""
Where an experiment keeps its working files.

Next to each ``.kit`` file is the experiment folder, which holds two working
folders besides the sample folders:

``cache/``
    Files Honeychrome can make again at any time from the FCS files and the
    experiment: unmixed events kept so that samples are not unmixed again,
    and compact per-sample event sets that make counting and phenotype-shift
    runs fast. When a file is missing it is simply made again, so these
    folders are capped in size (App Configuration, "Temporary files per
    experiment") and can be cleared.

``state/``
    Files that are part of the user's work: plugin settings, the saved
    results of runs (Automated Gating, Honeycluster, FlowCode statistics),
    spectral cleaning and spectral variants. Nothing here is ever removed
    automatically.

Only the folders in ``REBUILDABLE`` are ever pruned or cleared. Everything
else found in ``cache/`` (from experiments made before the split, or written
by an older version) is moved to the same place under ``state/`` when the
experiment is opened. A folder that already exists under ``state/`` is
merged item by item. Moves never overwrite: an item whose destination exists
is left where it is and logged, and it is still never deleted.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_FOLDER = 'cache'
STATE_FOLDER = 'state'

# Folders under cache/ whose files are remade when missing, as relative paths.
REBUILDABLE = (
    'unmixed',
    'flowcode_unmixed',
    'flowcode_analysis/count_events',
    'flowcode_analysis/shift_events',
)

_TMP_MARK = '.tmp.'


def cache_dir(experiment_dir, *parts) -> Path:
    """``<experiment>/cache/<parts>``: files that are remade when missing."""
    return Path(experiment_dir, CACHE_FOLDER, *parts)


def state_dir(experiment_dir, *parts) -> Path:
    """``<experiment>/state/<parts>``: files that are part of the user's work."""
    return Path(experiment_dir, STATE_FOLDER, *parts)


def _rebuildable_parents() -> set[str]:
    """Every folder above a REBUILDABLE folder (e.g. ``flowcode_analysis``)."""
    parents = set()
    for rel in REBUILDABLE:
        parts = rel.split('/')
        for i in range(1, len(parts)):
            parents.add('/'.join(parts[:i]))
    return parents


def migrate(experiment_dir) -> int:
    """Move everything in ``cache/`` that is not REBUILDABLE to ``state/``.

    Runs whenever an experiment is opened; when nothing needs moving it only
    lists the top of ``cache/``. Returns the number of entries moved."""
    cache = cache_dir(experiment_dir)
    if not cache.is_dir():
        return 0
    keep, parents = set(REBUILDABLE), _rebuildable_parents()
    moved = 0
    pending = [cache]
    while pending:
        folder = pending.pop()
        for entry in sorted(folder.iterdir()):
            rel = entry.relative_to(cache).as_posix()
            if rel in keep:
                continue
            if rel in parents and entry.is_dir():
                pending.append(entry)
                continue
            dest = state_dir(experiment_dir, rel)
            if dest.is_dir() and entry.is_dir():
                pending.append(entry)
                continue
            if dest.exists():
                logger.warning('Experiment folder: %s was not moved to state/, which already has it', entry)
                continue
            try:
                dest.parent.mkdir(parents=True, exist_ok=True)
                entry.rename(dest)
                moved += 1
            except OSError as exc:
                logger.warning('Experiment folder: could not move %s to state/ (%s)', entry, exc)
    if moved:
        logger.info('Experiment folder: moved %d item(s) from cache/ to state/ in %s', moved, experiment_dir)
    return moved


def _cache_files(experiment_dir) -> list[tuple[float, int, Path]]:
    """(last used, size, path) of every file in the REBUILDABLE folders,
    leaving out files still being written."""
    files = []
    for rel in REBUILDABLE:
        folder = cache_dir(experiment_dir, rel)
        if not folder.is_dir():
            continue
        for p in folder.rglob('*'):
            if _TMP_MARK in p.name:
                continue
            try:
                if not p.is_file():
                    continue
                st = p.stat()
            except OSError:
                continue
            files.append((st.st_mtime, st.st_size, p))
    return files


def cache_size(experiment_dir) -> int:
    """Bytes in the experiment's REBUILDABLE folders."""
    return sum(size for _t, size, _p in _cache_files(experiment_dir))


def max_cache_bytes() -> int:
    """The App Configuration limit for one experiment, in bytes."""
    from honeychrome import settings as app_settings
    return int(app_settings.unmixing_cache_gb_retrieved) * 1024 ** 3


def prune_cache(experiment_dir, max_bytes: int | None = None) -> int:
    """Remove the least recently used REBUILDABLE files until the experiment
    is within ``max_bytes`` (default: the App Configuration limit). Returns
    the bytes removed."""
    if max_bytes is None:
        max_bytes = max_cache_bytes()
    files = _cache_files(experiment_dir)
    total = sum(f[1] for f in files)
    freed = 0
    for _mtime, size, p in sorted(files):
        if total <= max_bytes:
            break
        try:
            p.unlink()
            total -= size
            freed += size
        except OSError:
            pass
    return freed


def _folder_size(folder: Path) -> int:
    total = 0
    for p in folder.rglob('*'):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            pass
    return total


def clear_cache(experiment_dir) -> int:
    """Remove every REBUILDABLE folder of the experiment. Files in use (on
    Windows) are left in place. Returns the bytes removed."""
    freed = 0
    for rel in REBUILDABLE:
        folder = cache_dir(experiment_dir, rel)
        if not folder.is_dir():
            continue
        before = _folder_size(folder)
        shutil.rmtree(folder, ignore_errors=True)
        freed += before - (_folder_size(folder) if folder.exists() else 0)
    return freed


def known_experiment_dirs() -> list[Path]:
    """Experiment folders of every ``.kit`` in the Experiments folder and in
    the recent-files list."""
    from honeychrome.settings import experiments_folder, q_settings
    kits = set()
    root = Path.home() / experiments_folder
    if root.is_dir():
        kits.update(p for p in root.rglob('*.kit') if p.is_file())
    recent = q_settings.value('recent_files', []) or []
    if isinstance(recent, str):
        recent = [recent]
    for path in recent:
        p = Path(path)
        if p.suffix == '.kit' and p.is_file():
            kits.add(p)
    dirs = {p.parent / p.stem for p in kits}
    return sorted(d for d in dirs if d.is_dir())
