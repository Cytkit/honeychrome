"""
Experiment-specific plugin state, kept in the experiment's folder.

A plugin keeps its settings for one experiment in
``<experiment folder>/cache/<folder>/state.json``, stamped with the id of
the ``.kit`` they belong to (``ExperimentModel.experiment_id``). The plugin's
results live in the same cache folder. When a new ``.kit`` takes over a folder
(an experiment deleted and created again at the same path), the folder's
contents carry another experiment's id: they are moved to
``cache/<folder>.previous`` and the new experiment starts clean. QSettings
keeps only window layout and application-wide preferences.

Experiments saved before ids existed keep their plugin state: on first use
the state is read from the plugin's old QSettings group (keyed by the
experiment folder) and the group is removed.

``ExperimentStateStore`` is read and written like QSettings (``value``,
``setValue``, ``remove``), so a plugin's existing per-key code is unchanged;
``sync()`` writes the file when something changed.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

from honeychrome.experiment_model import safe_save

log = logging.getLogger(__name__)

__all__ = ['ExperimentStateStore', 'open_store', 'legacy_group_key', 'STATE_FILE']

STATE_FILE = 'state.json'


def legacy_group_key(experiment_dir) -> str:
    """The QSettings group the plugins used for an experiment folder."""
    return str(experiment_dir).replace('\\', '/').replace(':', '_').replace(' ', '_')


class ExperimentStateStore:
    """One plugin's settings for one experiment (see the module docstring)."""

    def __init__(self, path: Path, experiment_id: str, values: dict):
        self.path = Path(path)
        self.experiment_id = experiment_id
        self._values = dict(values)
        self._dirty = False

    def value(self, key: str, default=None, type=None):
        """The stored value, or *default*; *type* converts it as QSettings does."""
        if key not in self._values:
            return default
        v = self._values[key]
        if type is bool and isinstance(v, str):
            return v.strip().lower() in ('true', '1', 'yes')
        if type is not None and v is not None:
            try:
                return type(v)
            except (TypeError, ValueError):
                return default
        return v

    def setValue(self, key: str, value):
        try:
            changed = bool(self._values.get(key, object()) != value)
        except (TypeError, ValueError):
            changed = True
        if changed:
            self._values[key] = value
            self._dirty = True

    def remove(self, key: str):
        if key in self._values:
            del self._values[key]
            self._dirty = True

    def contains(self, key: str) -> bool:
        return key in self._values

    def sync(self, force: bool = False):
        """Write the file when a value changed (or *force*)."""
        if not (self._dirty or force):
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            text = json.dumps({'experiment_id': self.experiment_id, 'values': self._values},
                              indent=1, default=_jsonable)
            safe_save(text, str(self.path))
            self._dirty = False
        except Exception:
            log.exception("could not write %s", self.path)


def _jsonable(value):
    """JSON stand-in for numpy scalars and arrays; text for anything else."""
    for method in ('tolist', 'item'):
        convert = getattr(value, method, None)
        if callable(convert):
            try:
                return convert()
            except Exception:
                pass
    return str(value)


def _read(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        log.exception("could not read %s", path)
        return None


def _set_aside(folder: Path):
    """Move another experiment's cache folder to <folder>.previous."""
    previous = folder.with_name(folder.name + '.previous')
    try:
        if previous.exists():
            shutil.rmtree(previous)
        folder.rename(previous)
        log.info("moved %s, which belongs to another experiment, to %s", folder, previous)
    except Exception:
        log.exception("could not move %s aside; removing it", folder)
        shutil.rmtree(folder, ignore_errors=True)


def _from_qsettings(app_key: str, experiment_dir) -> dict:
    """Read and remove the plugin's old QSettings group for a folder."""
    from PySide6.QtCore import QSettings
    settings = QSettings('honeychrome', app_key)
    group = legacy_group_key(experiment_dir)
    settings.beginGroup(group)
    try:
        values = {k: settings.value(k) for k in settings.childKeys()}
    finally:
        settings.endGroup()
    if values:
        settings.remove(group)
    return values


def open_store(controller, folder: str, legacy_app_key: str | None = None,
               experiment_dir=None) -> ExperimentStateStore:
    """Open *folder*'s state for the experiment the controller shows (or
    the folder *experiment_dir* of it), claiming the cache folder for this
    ``.kit``: another experiment's contents are moved aside, and for an
    experiment saved before ids existed the old QSettings group
    *legacy_app_key* is read once. The file is written at once, so the
    folder is stamped from its first use."""
    experiment = controller.experiment
    exp_dir = Path(experiment_dir or controller.experiment_dir)
    exp_id = str(getattr(experiment, 'experiment_id', '') or '')
    legacy = bool(getattr(experiment, 'legacy_plugin_state', False))
    cache = exp_dir / 'cache' / folder
    path = cache / STATE_FILE
    data = _read(path)
    if data is not None and data.get('experiment_id') == exp_id:
        values = dict(data.get('values') or {})
    elif legacy:
        if data is not None:
            values = dict(data.get('values') or {})
        else:
            values = _from_qsettings(legacy_app_key, exp_dir) if legacy_app_key else {}
            if values:
                log.info("moved %d %s settings from QSettings to %s", len(values), folder, path)
    else:
        if cache.exists() and any(cache.iterdir()):
            _set_aside(cache)
        values = {}
    store = ExperimentStateStore(path, exp_id, values)
    store.sync(force=True)
    return store
