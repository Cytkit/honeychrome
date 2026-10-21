"""
job_gate.py
-----------
One heavy job at a time across the application.

A heavy job (DR, clustering, statistics, AF extraction, noise model fit,
Clean Controls, export, ...) asks ``controller.job_gate.try_acquire(label)``
on the main thread before it starts anything, prompts included. It gets a
``JobToken`` or ``None`` when another job holds the gate, in which case it
shows ``busy_text()`` and does not start. The token is released once, in the
slot that ends the job, on every path (finish, error, cancel, declined
prompt); ``release`` is idempotent. A chain of threads (estimate -> confirm ->
run) keeps one token and renames it with ``hand_over``.

Automatic jobs (background display unmixing, AF envelope build, comparison
plots) do not take the gate: they call ``when_free(key, callback)``, which
runs the callback at once when the gate is free, else after the next release.
A later request under the same key replaces the waiting one.

``guard(*widgets)`` greys run buttons and controls that change the unmixing
out while the gate is held, naming the holder in their tooltip.

Plugins look the gate up with ``getattr(controller, 'job_gate', None)``;
``None`` (an older Honeychrome) means no gating. The module functions
``acquire_job``, ``wait_if_busy`` and ``guard_while_busy`` do that lookup.
"""

from __future__ import annotations

import logging
import time

from PySide6.QtCore import QCoreApplication, QObject, QThread, QTimer, Signal
from shiboken6 import isValid

logger = logging.getLogger(__name__)


class JobToken:
    """The hold on a ``JobGate`` by one job (or one chain of jobs)."""

    def __init__(self, gate: 'JobGate', label: str):
        self._gate = gate
        self.label = str(label)
        self.started = time.time()
        self._released = False

    @property
    def held(self) -> bool:
        return not self._released and self._gate._token is self

    def hand_over(self, label: str) -> None:
        """Keep the hold for the next step of a chain, under a new label."""
        if not self.held:
            return
        self.label = str(label)
        logger.debug('JobGate: hand over to %s', self.label)
        self._gate._emit_changed()

    def release(self) -> None:
        """End the hold. Safe to call more than once, and after a forced release."""
        if self._released:
            return
        self._released = True
        self._gate._release(self)


class JobGate(QObject):
    """Mutual exclusion of heavy jobs; main thread only."""

    changed = Signal()   # acquired, handed over or released

    def __init__(self, parent=None):
        super().__init__(parent)
        self._token: JobToken | None = None
        self._waiting: dict = {}
        self._guarded: list = []
        self._guard_state: dict = {}

    # ------------------------------------------------------------------
    # Holding
    # ------------------------------------------------------------------

    @staticmethod
    def _check_main_thread() -> None:
        app = QCoreApplication.instance()
        if app is not None and QThread.currentThread() is not app.thread():
            raise RuntimeError('JobGate is main-thread only')

    def try_acquire(self, label: str) -> JobToken | None:
        """A token for *label*, or ``None`` when another job holds the gate."""
        self._check_main_thread()
        if self._token is not None:
            logger.debug('JobGate: %s refused, held by %s', label, self._token.label)
            return None
        self._token = JobToken(self, label)
        logger.debug('JobGate: acquired by %s', self._token.label)
        self._emit_changed()
        return self._token

    def _emit_changed(self) -> None:
        self._apply_guards()
        self.changed.emit()

    def _release(self, token: JobToken) -> None:
        if self._token is not token:
            return
        self._token = None
        logger.debug('JobGate: released by %s', token.label)
        self._emit_changed()
        if self._waiting:
            if QCoreApplication.instance() is None:
                self._run_waiting()
            else:
                QTimer.singleShot(0, self._run_waiting)

    def force_release(self, reason: str) -> None:
        """Drop the hold and every waiting request, e.g. to recover from a job
        whose finish never arrived; its token's later release does nothing."""
        self._waiting.clear()
        token, self._token = self._token, None
        if token is None:
            return
        token._released = True
        logger.warning('JobGate: %s released by force (%s)', token.label, reason)
        self._emit_changed()

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def is_held(self) -> bool:
        return self._token is not None

    @property
    def holder(self) -> tuple | None:
        """``(label, started)`` of the job holding the gate, or ``None``."""
        if self._token is None:
            return None
        return self._token.label, self._token.started

    def busy_text(self) -> str:
        """'Busy: <label> (started hh:mm)', or '' when free."""
        if self._token is None:
            return ''
        started = time.strftime('%H:%M', time.localtime(self._token.started))
        return f'Busy: {self._token.label} (started {started})'

    # ------------------------------------------------------------------
    # Greying out
    # ------------------------------------------------------------------

    def guard(self, *widgets) -> None:
        """Grey *widgets* (QWidget or QAction) out while the gate is held, with
        the busy text leading their tooltip. On release, a widget is enabled
        again only if this disabled it and it is still disabled; owners whose
        enabled state depends on other things refresh it on ``changed``."""
        for w in widgets:
            if w is not None and all(w is not g for g in self._guarded):
                self._guarded.append(w)
        if self._token is not None:
            self._apply_guards()

    def _apply_guards(self) -> None:
        busy = self.busy_text()
        alive = []
        for w in self._guarded:
            if not isValid(w):
                self._guard_state.pop(id(w), None)
                continue
            state = self._guard_state.setdefault(id(w), {})
            try:
                if busy:
                    if 'tip' not in state:
                        state['tip'] = w.toolTip()
                        state['disabled'] = w.isEnabled()
                        if state['disabled']:
                            w.setEnabled(False)
                    w.setToolTip(f'{busy}\n\n{state["tip"]}' if state['tip'] else busy)
                elif 'tip' in state:
                    if state['disabled'] and not w.isEnabled():
                        w.setEnabled(True)
                    w.setToolTip(state['tip'])
                    state.clear()
            except RuntimeError:   # the C++ object has been deleted
                self._guard_state.pop(id(w), None)
                continue
            alive.append(w)
        self._guarded = alive

    # ------------------------------------------------------------------
    # Automatic jobs
    # ------------------------------------------------------------------

    def when_free(self, key, callback) -> bool:
        """Run *callback* now if the gate is free (returns True), else after
        the next release (returns False); a later request under *key*
        replaces the waiting one."""
        if self._token is None:
            self._waiting.pop(key, None)
            callback()
            return True
        self._waiting[key] = callback
        return False

    def cancel_wait(self, key) -> None:
        self._waiting.pop(key, None)

    def is_waiting(self, key) -> bool:
        return key in self._waiting

    def _run_waiting(self) -> None:
        while self._waiting and self._token is None:
            key = next(iter(self._waiting))
            callback = self._waiting.pop(key)
            try:
                callback()
            except Exception:
                logger.exception('JobGate: waiting request %r failed', key)


class _NoGateToken:
    """Stands in for a token when the application has no job gate."""
    held = True
    label = ''

    def hand_over(self, label: str) -> None:
        pass

    def release(self) -> None:
        pass


def acquire_job(controller, label: str):
    """A token for *label* from ``controller.job_gate``. ``None`` when another
    job holds the gate, after showing the busy text in the status bar; a
    no-op token when the application has no gate."""
    gate = getattr(controller, 'job_gate', None)
    if gate is None:
        return _NoGateToken()
    token = gate.try_acquire(label)
    if token is None:
        bus = getattr(controller, 'bus', None)
        if bus is not None:
            bus.statusMessage.emit(gate.busy_text())
    return token


def guard_while_busy(controller, *widgets) -> None:
    """``JobGate.guard`` on the application's gate; nothing without one."""
    gate = getattr(controller, 'job_gate', None)
    if gate is not None:
        gate.guard(*widgets)


def wait_if_busy(controller, key, callback, status=None) -> bool:
    """For automatic jobs: True when a heavy job holds the gate, in which case
    *callback* runs after the release (a later request under *key* replaces
    this one) and *status*, if given, is called with 'Waiting for <label> …'.
    False when the gate is free (or absent) and the job can start now."""
    gate = getattr(controller, 'job_gate', None)
    if gate is None or not gate.is_held():
        return False
    gate.when_free(key, callback)
    if status is not None:
        status(f'Waiting for {gate.holder[0]} …')
    return True
