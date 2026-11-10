"""
Tests for controller_components/job_gate.py: acquire and refuse, idempotent
release, hand-over, release on an exception path, forced release on an
experiment change, waiting requests of automatic jobs, main-thread check,
greying out of guarded widgets, the module helpers.
"""

import threading
from types import SimpleNamespace

import pytest
import shiboken6
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication, QPushButton

from honeychrome.controller_components.job_gate import (
    JobGate, acquire_job, guard_while_busy, wait_if_busy,
)


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def _process(app):
    app.processEvents()
    app.processEvents()


def test_acquire_refuse_release(app):
    gate = JobGate()
    seen = []
    gate.changed.connect(lambda: seen.append(gate.holder))
    token = gate.try_acquire('DR: UMAP')
    assert token is not None and token.held
    assert gate.holder[0] == 'DR: UMAP'
    assert gate.busy_text().startswith('Busy: DR: UMAP (started ')
    assert gate.try_acquire('Clustering') is None
    token.release()
    assert not gate.is_held() and gate.holder is None and gate.busy_text() == ''
    token.release()
    assert len(seen) == 2
    again = gate.try_acquire('Clustering')
    assert again is not None
    again.release()


def test_stale_token_does_not_release_new_holder(app):
    gate = JobGate()
    old = gate.try_acquire('A')
    gate.force_release('experiment change')
    new = gate.try_acquire('B')
    old.release()
    assert new.held and gate.holder[0] == 'B'
    new.release()


def test_hand_over(app):
    gate = JobGate()
    seen = []
    gate.changed.connect(lambda: seen.append(gate.holder[0] if gate.holder else None))
    token = gate.try_acquire('Auto generate')
    started = token.started
    token.hand_over('Clean Controls')
    assert gate.holder == ('Clean Controls', started)
    assert gate.try_acquire('Export') is None
    token.release()
    token.hand_over('Recalculate')
    assert gate.holder is None
    assert seen == ['Auto generate', 'Clean Controls', None]


def test_release_in_finally(app):
    gate = JobGate()
    token = gate.try_acquire('Statistics')
    with pytest.raises(ValueError):
        try:
            raise ValueError('worker failed')
        finally:
            token.release()
    assert not gate.is_held()


def test_force_release_drops_waiting(app):
    gate = JobGate()
    token = gate.try_acquire('Fix Unmix')
    ran = []
    assert gate.when_free('cmp', lambda: ran.append(1)) is False
    gate.force_release('experiment change')
    assert not token.held and not gate.is_waiting('cmp')
    _process(app)
    assert ran == []
    gate.force_release('again')


def test_when_free_runs_now_or_after_release(app):
    gate = JobGate()
    ran = []
    assert gate.when_free('a', lambda: ran.append('now')) is True
    assert ran == ['now']
    token = gate.try_acquire('Model Noise')
    gate.when_free('a', lambda: ran.append('old'))
    gate.when_free('a', lambda: ran.append('a'))
    gate.when_free('b', lambda: ran.append('b'))
    gate.cancel_wait('b')
    assert gate.is_waiting('a') and not gate.is_waiting('b')
    token.release()
    assert ran == ['now']
    _process(app)
    assert ran == ['now', 'a']
    assert not gate.is_waiting('a')


def test_waiting_kept_when_gate_taken_again_first(app):
    gate = JobGate()
    ran = []
    first = gate.try_acquire('A')
    gate.when_free('cmp', lambda: ran.append(1))
    first.release()
    second = gate.try_acquire('B')
    _process(app)
    assert ran == [] and gate.is_waiting('cmp')
    second.release()
    _process(app)
    assert ran == [1]


def test_waiting_callback_error_does_not_stop_others(app):
    gate = JobGate()
    ran = []
    token = gate.try_acquire('A')

    def boom():
        raise RuntimeError('stale')

    gate.when_free('x', boom)
    gate.when_free('y', lambda: ran.append('y'))
    token.release()
    _process(app)
    assert ran == ['y']


def test_main_thread_only(app):
    gate = JobGate()
    errors = []

    def worker():
        try:
            gate.try_acquire('from a worker')
        except RuntimeError as e:
            errors.append(str(e))

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert errors and not gate.is_held()


def test_guard_greys_and_restores(app):
    gate = JobGate()
    run_btn, off_btn, action = QPushButton('Run'), QPushButton('Off'), QAction('Export')
    run_btn.setToolTip('Start the run')
    off_btn.setEnabled(False)
    gate.guard(run_btn, off_btn, action)
    assert run_btn.isEnabled() and action.isEnabled()
    token = gate.try_acquire('Honeycluster: UMAP training')
    assert not run_btn.isEnabled() and not off_btn.isEnabled() and not action.isEnabled()
    assert run_btn.toolTip().startswith('Busy: Honeycluster: UMAP training')
    assert run_btn.toolTip().endswith('Start the run')
    token.hand_over('Honeycluster: UMAP on all samples')
    assert 'UMAP on all samples' in run_btn.toolTip() and run_btn.toolTip().endswith('Start the run')
    token.release()
    assert run_btn.isEnabled() and action.isEnabled()
    assert not off_btn.isEnabled()
    assert run_btn.toolTip() == 'Start the run' and off_btn.toolTip() == ''


def test_guard_leaves_owner_changes(app):
    gate = JobGate()
    btn = QPushButton('Run')
    gate.guard(btn)
    token = gate.try_acquire('A')
    btn.setEnabled(True)   # the owner re-enabled it during the hold
    token.release()
    assert btn.isEnabled() and btn.toolTip() == ''


def test_guard_while_held_and_deleted_widgets(app):
    gate = JobGate()
    token = gate.try_acquire('A')
    btn = QPushButton('Run')
    gate.guard(btn, btn)
    assert not btn.isEnabled() and len(gate._guarded) == 1
    gone = QPushButton('Gone')
    gate.guard(gone)
    shiboken6.delete(gone)
    token.release()
    assert btn.isEnabled() and len(gate._guarded) == 1


def test_module_helpers_with_and_without_gate(app):
    gate = JobGate()
    controller = SimpleNamespace(job_gate=gate, bus=None)
    plain = SimpleNamespace()
    free = acquire_job(plain, 'X')
    assert free is not None and free.held
    free.hand_over('Y')
    free.release()
    token = acquire_job(controller, 'A')
    assert token is not None and acquire_job(controller, 'B') is None
    ran, shown = [], []
    assert wait_if_busy(plain, 'k', lambda: ran.append(1)) is False
    assert wait_if_busy(controller, 'k', lambda: ran.append(2), shown.append) is True
    assert shown == ['Waiting for A …']
    btn = QPushButton('Run')
    guard_while_busy(plain, btn)
    guard_while_busy(controller, btn)
    assert not btn.isEnabled()
    token.release()
    _process(app)
    assert ran == [2] and btn.isEnabled()
