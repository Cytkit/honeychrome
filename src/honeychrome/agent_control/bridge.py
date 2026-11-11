"""
bridge.py
---------
Runs the agent's tool calls on the Qt main thread, one at a time, and
collects the messages Honeychrome shows while a call runs
(design/agent_control.md §5.2 and §5.6).

The MCP server thread calls ``call``, which queues the call to the main
thread and waits for its result. Controller methods with
``with_busy_cursor`` spin a nested event loop, which can deliver the next
queued call in the middle of the current one; ``_run`` queues it instead,
so calls never interleave.
"""

from __future__ import annotations

import concurrent.futures
import logging
import re
import shutil
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal, Slot

from honeychrome.agent_control.tools import TOOLS, ToolContext, ToolError, ToolSpec

logger = logging.getLogger(__name__)

MAX_MESSAGES = 20
MAX_STATUS = 5
# status-bar progress that says nothing to the agent
PROGRESS = re.compile(r'^(Agent:|Ready\.?$|Calculating \d+/\d+ |Calculating \d+ histograms|Initialising plots|Processing \d+/\d+)')


@dataclass
class _Call:
    spec: ToolSpec
    args: dict
    future: concurrent.futures.Future


class AgentBridge(QObject):
    _requested = Signal(object)

    def __init__(self, view):
        super().__init__()
        self.context = ToolContext(view)
        self._queue: deque = deque()
        self._running = False
        self._messages: dict | None = None
        self._backed_up: set = set()
        self._requested.connect(self._run, Qt.QueuedConnection)
        view.bus.popupMessage.connect(lambda message: self._note('popups', message))
        view.bus.warningMessage.connect(lambda message: self._note('warnings', message))
        view.bus.statusMessage.connect(lambda message: self._note('status', message))

    @property
    def capturing(self) -> bool:
        """True while a tool runs: pop-ups go to the agent instead of a dialog."""
        return self._messages is not None

    # server thread
    def call(self, name: str, args: dict) -> tuple:
        """``(result, messages)`` of the tool, run on the main thread; raises ``ToolError``."""
        spec = TOOLS[name]
        job = _Call(spec, args, concurrent.futures.Future())
        self._requested.emit(job)
        try:
            return job.future.result(timeout=spec.timeout)
        except concurrent.futures.TimeoutError:
            raise ToolError(f'{name} did not finish within {spec.timeout:.0f} s. It may still be running in '
                            'Honeychrome: call get_state before trying again.') from None

    # main thread
    @Slot(object)
    def _run(self, job: _Call) -> None:
        self._queue.append(job)
        if self._running:
            return
        self._running = True
        try:
            while self._queue:
                self._execute(self._queue.popleft())
        finally:
            self._running = False

    def _execute(self, job: _Call) -> None:
        if not job.future.set_running_or_notify_cancel():
            return
        spec, ctx = job.spec, self.context
        self._messages = {'popups': [], 'warnings': [], 'status': []}
        try:
            self._check(spec)
            logger.info('Agent tool %s %s', spec.name, job.args)
            ctx.bus.statusMessage.emit(f'Agent: {spec.title}')
            result = spec.fn(ctx, **job.args)
            job.future.set_result((result, self._messages))
        except ToolError as exc:
            logger.info('Agent tool %s refused: %s', spec.name, exc)
            exc.messages = self._messages
            job.future.set_exception(exc)
        except Exception as exc:
            logger.exception('Agent tool %s failed', spec.name)
            error = ToolError(f'{type(exc).__name__}: {exc}')
            error.messages = self._messages
            job.future.set_exception(error)
        finally:
            self._messages = None

    def _check(self, spec: ToolSpec) -> None:
        ctx = self.context
        c = ctx.controller
        if spec.needs_experiment and not ctx.experiment_open():
            raise ToolError('No experiment is open in Honeychrome. Use list_experiments and open_experiment.')
        if not spec.writes:
            return
        if c.job_gate.is_held():
            raise ToolError(f'Honeychrome is busy ({c.job_gate.busy_text()}). Try again when it has finished.')
        if not c.stop_live_data_processing.is_set():
            raise ToolError('Honeychrome is acquiring data; the agent cannot make changes during acquisition.')
        if ctx.experiment_open():
            self._backup(Path(c.experiment.experiment_path))

    def _backup(self, kit: Path) -> None:
        """Copy the .kit once per session before the agent's first change to it."""
        key = str(kit.resolve())
        if key in self._backed_up or not kit.exists():
            return
        backup = kit.with_name(kit.name + '.agent-backup')
        try:
            shutil.copy2(kit, backup)
            self._backed_up.add(key)
            logger.info('Agent control: backed up %s to %s', kit, backup)
        except OSError as exc:
            logger.warning('Agent control: %s not backed up (%s)', kit, exc)

    def _note(self, kind: str, message) -> None:
        if self._messages is None:
            return
        text = str(message)
        if kind == 'status' and PROGRESS.match(text):
            return
        entries = self._messages[kind]
        if not entries or entries[-1] != text:
            entries.append(text)
        del entries[:-(MAX_STATUS if kind == 'status' else MAX_MESSAGES)]
