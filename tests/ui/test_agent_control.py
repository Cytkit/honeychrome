"""Agent control (MCP): the protocol, tools that keep the open window in sync, one call at a time,
calls that time out, and unmixed reads that wait for the background unmixing.

Tools are called as the MCP server calls them: from another thread, while the Qt loop runs here.
"""
import json
import shutil
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PySide6.QtCore import QEventLoop, QModelIndex, QSettings, QTimer

import honeychrome
from honeychrome.agent_control import tools as agent_tools
from honeychrome.agent_control.bridge import AgentBridge
from honeychrome.agent_control.server import McpHttpServer
from honeychrome.agent_control.tools import TOOLS, ToolError, ToolSpec
from honeychrome.controller import Controller
from honeychrome.controller_components import unmix_method
from honeychrome.controller_components.functions import apply_transfer_matrix
from honeychrome.view import View

pytestmark = pytest.mark.filterwarnings('ignore:No events bus connected')

EXAMPLE_FCS = (Path(honeychrome.__file__).parent / 'instrument_driver_components' / 'data'
               / 'example_for_dummy_acquisition.fcs')
TOKEN = 'test-token'


@pytest.fixture(autouse=True)
def user_settings():
    """Window and recent-file settings the app writes, restored afterwards."""
    stores = {name: QSettings('honeychrome', name) for name in ('MainWindow', 'app_configuration', 'ExperimentSelector')}
    saved = {name: {k: s.value(k) for k in s.allKeys()} for name, s in stores.items()}
    yield
    for name, store in stores.items():
        store.clear()
        for key, value in saved[name].items():
            store.setValue(key, value)
        store.sync()


@pytest.fixture
def app(qtbot, tmp_path, monkeypatch):
    """A real Controller, View and main window on a new experiment with two samples, and a bridge."""
    monkeypatch.setattr(agent_tools, '_experiments_folder', lambda: tmp_path.resolve())
    controller = Controller()
    controller.new_experiment(tmp_path / 'agent_test')
    raw = controller.experiment_dir / 'Raw'
    raw.mkdir(parents=True, exist_ok=True)
    for name in ('Tube_001.fcs', 'Tube_002.fcs'):
        shutil.copy2(EXAMPLE_FCS, raw / name)
    view = View(controller=controller)
    controller.bus = view.bus
    qtbot.addWidget(view.splash)
    view.load_main_window_with_experiment_and_template(controller.experiment.experiment_path)
    qtbot.addWidget(view.main_window)
    view.main_window.resize(1200, 800)
    view.main_window.show()
    qtbot.wait(500)   # let the grids' relayout after showing run before a test changes their plots
    bridge = AgentBridge(view)
    confirmations = []
    bridge.context.confirm_hook = lambda text: confirmations.append(text) or True
    yield SimpleNamespace(controller=controller, view=view, bridge=bridge, confirmations=confirmations)
    qtbot.waitUntil(lambda: not controller.background_unmixing_running() and controller._method_thread is None,
                    timeout=10_000)


def call(app, qtbot, tool, **args):
    """The tool's result, called from another thread as the MCP server does; raises ToolError."""
    box = {}

    def worker():
        try:
            box['result'] = app.bridge.call(tool, args)[0]
        except ToolError as exc:
            box['error'] = exc

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    qtbot.waitUntil(lambda: not thread.is_alive(), timeout=60_000)
    if 'error' in box:
        raise box['error']
    return box['result']


def run_threads(qtbot, *targets, timeout=30_000):
    threads = [threading.Thread(target=t, daemon=True) for t in targets]
    for thread in threads:
        thread.start()
    qtbot.waitUntil(lambda: not any(t.is_alive() for t in threads), timeout=timeout)


def release_later(event, seconds=8.0):
    """Stops a probe even if the code under test never does, so a failing test cannot hang."""
    timer = threading.Timer(seconds, event.set)
    timer.daemon = True
    timer.start()


@pytest.fixture
def probe(monkeypatch):
    """Registers a tool that runs until released, keeping the window responsive like a long job."""
    def register(timeout=60.0):
        state = SimpleNamespace(runs=0, started=threading.Event(), release=threading.Event())

        def fn(ctx):
            state.runs += 1
            state.started.set()
            loop = QEventLoop()
            while not state.release.is_set():
                QTimer.singleShot(20, loop.quit)
                loop.exec()
            return {'done': True}

        monkeypatch.setitem(TOOLS, 'probe', ToolSpec('probe', 'Probe', 'Test tool.', {'type': 'object', 'properties': {}},
                                                     fn, False, False, False, timeout))
        return state
    return register


# ---------------------------------------------------------------------------
# Protocol over HTTP
# ---------------------------------------------------------------------------

def test_http_endpoint_checks_token_and_origin_and_runs_tools(app, qtbot):
    server = McpHttpServer(app.bridge, 0, TOKEN)
    port = server._httpd.server_address[1]
    server.start()
    try:
        def post(payload, token=TOKEN, origin=None):
            request = urllib.request.Request(f'http://127.0.0.1:{port}/mcp', data=json.dumps(payload).encode(),
                                             method='POST', headers={'Content-Type': 'application/json'})
            if token:
                request.add_header('Authorization', f'Bearer {token}')
            if origin:
                request.add_header('Origin', origin)
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    body = response.read()
                    return response.status, json.loads(body) if body else None
            except urllib.error.HTTPError as exc:
                return exc.code, None

        replies = {}

        def client():
            replies['init'] = post({'jsonrpc': '2.0', 'id': 1, 'method': 'initialize',
                                    'params': {'protocolVersion': '2025-06-18', 'capabilities': {}}})
            replies['wrong_token'] = post({'jsonrpc': '2.0', 'id': 2, 'method': 'ping'}, token='wrong')
            replies['foreign_origin'] = post({'jsonrpc': '2.0', 'id': 3, 'method': 'ping'},
                                             origin='http://evil.example')
            replies['list'] = post({'jsonrpc': '2.0', 'id': 4, 'method': 'tools/list'})
            replies['state'] = post({'jsonrpc': '2.0', 'id': 5, 'method': 'tools/call',
                                     'params': {'name': 'get_state', 'arguments': {}}})
            replies['bad_args'] = post({'jsonrpc': '2.0', 'id': 6, 'method': 'tools/call',
                                        'params': {'name': 'list_plots', 'arguments': {'scope': 3}}})
            replies['unknown'] = post({'jsonrpc': '2.0', 'id': 7, 'method': 'tools/call',
                                       'params': {'name': 'no_such_tool', 'arguments': {}}})

        run_threads(qtbot, client)
    finally:
        server.stop()

    assert replies['init'][1]['result']['protocolVersion'] == '2025-06-18'
    assert replies['wrong_token'][0] == 401
    assert replies['foreign_origin'][0] == 403
    assert {t['name'] for t in replies['list'][1]['result']['tools']} == set(TOOLS)
    state = replies['state'][1]['result']
    assert not state['isError'] and json.loads(state['content'][0]['text'])['experiment_open'] is True
    assert replies['bad_args'][1]['result']['isError']
    assert replies['unknown'][1]['error']['code'] == -32602


# ---------------------------------------------------------------------------
# Tools keep the window in sync
# ---------------------------------------------------------------------------

def _roi_names(app, plot_index):
    widget = app.view.main_window.cytometry_grid_raw.plot_widgets[plot_index]
    return [roi.label.gate_name for roi in widget.rois]


def _tree_text(app):
    model = app.view.main_window.gating_tree_raw.model
    texts = []

    def walk(parent):
        for row in range(model.rowCount(parent)):
            index = model.index(row, 0, parent)
            texts.append(str(model.data(index)))
            walk(index)
    walk(QModelIndex())
    return texts


def test_gate_tools_update_the_plots_and_gating_tree(app, qtbot):
    call(app, qtbot, 'load_sample', sample='Tube_001')
    plots = call(app, qtbot, 'list_plots', scope='raw')['plots']
    index = next(p['index'] for p in plots if p['type'] == 'hist2d')
    population = call(app, qtbot, 'summarise_plot', scope='raw', plot_index=index)['populations'][0]
    box = {'x_min': population['x_range'][0], 'x_max': population['x_range'][1],
           'y_min': population['y_range'][0], 'y_max': population['y_range'][1]}

    created = call(app, qtbot, 'create_gate', scope='raw', plot_index=index, gate_type='rectangle',
                   coordinates=box, name='AgentCells')
    assert created['statistics']['events'] > 0
    assert 'AgentCells' in _roi_names(app, index)
    assert any('AgentCells' in text for text in _tree_text(app))
    with pytest.raises(ToolError, match='already exists'):
        call(app, qtbot, 'create_gate', scope='raw', plot_index=index, gate_type='rectangle', coordinates=box,
             name='AgentCells')

    wider = dict(box, x_max=box['x_max'] * 1.2)
    call(app, qtbot, 'update_gate', scope='raw', gate_name='AgentCells', coordinates=wider)
    gate = app.controller.data_for_cytometry_plots_raw['gating'].get_gate('AgentCells')
    roi = next(r for r in app.view.main_window.cytometry_grid_raw.plot_widgets[index].rois
               if r.label.gate_name == 'AgentCells')
    assert roi.getState()['size'][0] == pytest.approx(gate.dimensions[0].max - gate.dimensions[0].min)

    call(app, qtbot, 'rename_gate', scope='raw', gate_name='AgentCells', new_name='Lymphocytes')
    assert 'Lymphocytes' in _roi_names(app, index)
    names = [row['gate'] for row in call(app, qtbot, 'get_statistics', scope='raw')['statistics']]
    assert 'Lymphocytes' in names and 'AgentCells' not in names

    call(app, qtbot, 'delete_gate', scope='raw', gate_name='Lymphocytes')
    assert app.confirmations and 'Lymphocytes' in app.confirmations[-1]
    assert 'Lymphocytes' not in _roi_names(app, index)
    assert not any('Lymphocytes' in text for text in _tree_text(app))


# ---------------------------------------------------------------------------
# One call at a time; calls that time out
# ---------------------------------------------------------------------------

def test_a_call_made_while_another_runs_is_refused(app, qtbot, probe):
    state = probe()
    release_later(state.release)
    out = {}

    def first():
        out['first'] = app.bridge.call('probe', {})[0]

    def second():
        state.started.wait(10)
        try:
            out['second'] = app.bridge.call('get_state', {})[0]
        except ToolError as exc:
            out['second'] = str(exc)
        state.release.set()

    run_threads(qtbot, first, second)
    assert out['first'] == {'done': True}
    assert 'still running probe' in out['second']
    assert state.runs == 1
    assert call(app, qtbot, 'get_state')['experiment_open'] is True


def test_a_call_that_times_out_before_it_starts_never_runs(app, qtbot, probe):
    state = probe(timeout=0.3)
    state.release.set()
    out = {}

    def worker():
        try:
            app.bridge.call('probe', {})
        except ToolError as exc:
            out['error'] = str(exc)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(5)   # the Qt loop does not run meanwhile, so the call cannot start
    assert 'was cancelled' in out['error']
    qtbot.wait(300)   # deliver the queued call
    assert state.runs == 0
    assert call(app, qtbot, 'get_state')['experiment_open'] is True


def test_a_call_that_times_out_while_running_blocks_others_until_it_ends(app, qtbot, probe):
    state = probe(timeout=0.5)
    release_later(state.release)
    out = {}

    def agent():
        try:
            app.bridge.call('probe', {})
        except ToolError as exc:
            out['first'] = str(exc)
        try:
            app.bridge.call('probe', {})   # the agent retries at once
        except ToolError as exc:
            out['retry'] = str(exc)
        state.release.set()

    run_threads(qtbot, agent)
    qtbot.waitUntil(lambda: not app.bridge.capturing, timeout=5000)
    qtbot.wait(300)   # a queued retry would run now
    assert 'other calls are refused until it ends' in out['first']
    assert 'still running probe' in out['retry']
    assert state.runs == 1
    assert call(app, qtbot, 'get_state')['experiment_open'] is True


# ---------------------------------------------------------------------------
# Unmixed data is read only once the background unmixing has finished
# ---------------------------------------------------------------------------

def test_background_unmixing_is_running_until_its_result_is_shown_or_dropped(qtbot, monkeypatch):
    controller = Controller()
    job = {'generation': controller._unmix_generation}
    current = {'value': True}
    shown = []
    monkeypatch.setattr(controller, '_method_job_current', lambda j: current['value'])
    monkeypatch.setattr(controller, '_show_method_result', lambda j, result: shown.append(result))

    controller._display_job = job
    assert controller.background_unmixing_running()
    controller._on_method_job_finished(job, 'final')
    assert shown == ['final'] and not controller.background_unmixing_running()

    controller._display_job = job
    controller._on_method_job_error(job, 'failed')
    assert not controller.background_unmixing_running()

    controller._display_job = job
    current['value'] = False   # e.g. another sample was loaded meanwhile
    controller._on_method_job_finished(job, 'stale')
    assert controller._display_job is None and shown == ['final']


def test_waiting_for_unmixing_stops_at_the_deadline_or_when_abandoned(app, qtbot, monkeypatch):
    ctx = app.bridge.context
    with monkeypatch.context() as patch:
        patch.setattr(app.controller, 'background_unmixing_running', lambda: True)
        ctx.deadline = time.monotonic() + 0.3
        assert ctx.wait_for_unmixing() is False
        ctx.deadline = float('inf')
        ctx.abandoned = lambda: True
        with pytest.raises(ToolError, match='timed out'):
            ctx.wait_for_unmixing()


class _SlowMethod:
    """What the display path reads of unmix_method.MethodState for a noise-model experiment."""
    prior_lambda = 1.0

    def effective_method(self, af, sample_key=None):
        return unmix_method.M2PG


FINAL_OFFSET = 1000.0


def _channel_assignment_model(controller):
    pnn = controller.experiment.settings['raw']['event_channels_pnn']
    fl_ids = list(controller.filtered_raw_fluorescence_channel_ids)
    model = controller.experiment.process['spectral_model']
    profiles = controller.experiment.process['profiles']
    model.clear()
    profiles.clear()
    for k, label in enumerate(('Dye A', 'Dye B')):
        model.append({'label': label, 'control_type': 'Channel Assignment', 'particle_type': '',
                      'gate_channel': pnn[fl_ids[k]], 'sample_name': '', 'sample_path': '', 'gate_label': ''})
        profile = np.full(len(fl_ids), 0.05)
        profile[k] = 1.0
        profiles[label] = profile.tolist()


def test_unmixed_reads_wait_for_the_background_unmixing(app, qtbot, monkeypatch):
    call(app, qtbot, 'load_sample', sample='Tube_001')
    _channel_assignment_model(app.controller)
    call(app, qtbot, 'refresh_unmixing')
    assert app.controller.transfer_matrix is not None

    # The noise model unmixes in the background for a second; until then the sample shows per-cell AF.
    background = []

    def provisional(raw, transfer, settings, fl_ids, spillover, af, state, key, force_r1=False):
        assert force_r1
        return {'unmixed': apply_transfer_matrix(transfer, raw)}

    def slow_method(raw_fl, state, af, sample_key=None, method=None):
        background.append(sample_key)
        time.sleep(1.0)
        return {}

    def final(raw, transfer, settings, fl_result, spillover=None, af=None):
        return {'unmixed': apply_transfer_matrix(transfer, raw) + FINAL_OFFSET}

    monkeypatch.setattr(unmix_method, 'method_state', lambda controller, corrections=None: _SlowMethod())
    monkeypatch.setattr(unmix_method, 'display_cache_key', lambda *args, **kwargs: None)
    monkeypatch.setattr(unmix_method, 'unmix', provisional)
    monkeypatch.setattr(unmix_method, 'unmix_fluorescence', slow_method)
    monkeypatch.setattr(unmix_method, 'assemble', final)

    loaded = call(app, qtbot, 'load_sample', sample='Tube_002')
    intensities = call(app, qtbot, 'get_gate_intensities', scope='unmixed', gates=['root'], channels=['Dye A'])
    median = next(iter(intensities['gates'][0]['channels'].values()))['median']
    c = app.controller
    data = c.data_for_cytometry_plots_unmixed
    column = data['pnn'].index(app.bridge.context.resolve_channel(data, 'Dye A'))
    per_cell_af = float(np.median(apply_transfer_matrix(c.transfer_matrix, c.raw_event_data)[:, column]))
    assert median == pytest.approx(per_cell_af + FINAL_OFFSET, abs=0.01)
    assert background == [loaded['loaded']]
    assert loaded['background_unmixing'] is False
