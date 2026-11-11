"""
Agent control: lets an AI agent drive the open Honeychrome window through a
local MCP server (design/agent_control.md). Off by default; turned on in
Edit > App Configuration. Only this package's own modules are imported, so
the feature adds no dependency.
"""

from __future__ import annotations

import json
import logging
import secrets

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QLabel
from shiboken6 import isValid

logger = logging.getLogger(__name__)

SETTING_ENABLED = 'agent_control_enabled'
SETTING_PORT = 'agent_control_port'
SETTING_TOKEN = 'agent_control_token'
DEFAULT_PORT = 8765


def _settings() -> QSettings:
    return QSettings('honeychrome', 'app_configuration')


def is_enabled() -> bool:
    return _settings().value(SETTING_ENABLED, False, type=bool)


def port() -> int:
    return _settings().value(SETTING_PORT, DEFAULT_PORT, type=int)


def token() -> str:
    """The bearer token, created on first use."""
    settings = _settings()
    value = str(settings.value(SETTING_TOKEN, '') or '')
    if not value:
        value = new_token()
    return value


def new_token() -> str:
    value = secrets.token_urlsafe(32)
    _settings().setValue(SETTING_TOKEN, value)
    return value


def client_config(port_number: int | None = None, token_value: str | None = None) -> str:
    """The MCP client settings (VS Code mcp.json form) for this server."""
    return json.dumps({'servers': {'honeychrome': {
        'type': 'http',
        'url': f'http://127.0.0.1:{port_number or port()}/mcp',
        'headers': {'Authorization': f'Bearer {token_value or token()}'},
    }}}, indent=2)


class AgentControl:
    """Starts and stops the MCP server for a View; one per application."""

    _instance: 'AgentControl | None' = None

    def __init__(self, view):
        self.view = view
        self.bridge = None
        self.server = None
        self.error = None
        self._indicator = None
        AgentControl._instance = self

    @classmethod
    def instance(cls) -> 'AgentControl | None':
        return cls._instance

    @property
    def running(self) -> bool:
        return self.server is not None

    @property
    def capturing_messages(self) -> bool:
        return self.bridge is not None and self.bridge.capturing

    def apply_settings(self) -> None:
        """Start, stop or restart the server to match App Configuration."""
        self.stop()
        if is_enabled():
            self.start()
        self.update_indicator()

    def start(self) -> None:
        from honeychrome.agent_control.bridge import AgentBridge
        from honeychrome.agent_control.server import McpHttpServer

        if self.bridge is None:
            self.bridge = AgentBridge(self.view)
        try:
            self.server = McpHttpServer(self.bridge, port(), token())
            self.server.start()
            self.error = None
        except OSError as exc:
            self.server = None
            self.error = f'Agent control could not listen on 127.0.0.1:{port()} ({exc}).'
            logger.warning(self.error)
            self.view.bus.warningMessage.emit(f'{self.error} Choose another port in App Configuration.')

    def stop(self) -> None:
        if self.server is not None:
            self.server.stop()
            self.server = None

    def status_text(self) -> str:
        if self.running:
            return f'Agent control: on ({self.server.url})'
        return self.error or 'Agent control: off'

    def update_indicator(self) -> None:
        """A permanent label in the main window's status bar while the server listens."""
        window = self.view.main_window
        if window is None or not isValid(window):
            return
        if self._indicator is None or not isValid(self._indicator) or self._indicator.window() is not window:
            self._indicator = QLabel()
            window.statusBar().addPermanentWidget(self._indicator)
        self._indicator.setText(self.status_text() if self.running else '')
        self._indicator.setToolTip('An AI agent can control Honeychrome through this local MCP server. '
                                   'Turn it off in Edit > App Configuration.')
        self._indicator.setVisible(self.running)
