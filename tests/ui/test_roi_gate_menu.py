"""The per-sample custom gate actions offered on an ROI's own context menu.

Samson asked for the same Customise / Revert / Adopt items that the gating
hierarchy tree offers to appear when right-clicking the gate drawn on the plot.
These tests exercise the shared helper directly, with a stand-in for the ROI's
DraggableRoiLabel, so no plot or sample has to be built.
"""
import pytest
from PySide6.QtWidgets import QMenu

from honeychrome.view_components.regions_of_interest import exec_gate_menu


class _FakeBus:
    """Records the (scope, gate_name) each request signal was emitted with."""

    class _Signal:
        def __init__(self):
            self.emitted = []

        def emit(self, *args):
            self.emitted.append(args)

    def __init__(self):
        self.customiseGateRequested = self._Signal()
        self.revertGateRequested = self._Signal()
        self.adoptGateRequested = self._Signal()


class _FakeGating:
    def __init__(self, custom=(), raises=False):
        self._custom = set(custom)
        self._raises = raises

    def is_custom_gate(self, sample_id, gate_name):
        if self._raises:
            raise ValueError('no such gate')
        return gate_name in self._custom


class _FakeLabel:
    """Stands in for DraggableRoiLabel, including its live sample lookup."""

    def __init__(self, gate_name='Lymphocytes', mode='raw', sample_id='A1.fcs',
                 custom=(), bus=True, raises=False):
        self.gate_name = gate_name
        self.mode = mode
        self.gating = _FakeGating(custom, raises)
        self.bus = _FakeBus() if bus else None
        self.sample_id = sample_id

    @property
    def current_sample_id(self):
        return self.sample_id

    def is_customised(self):
        if not self.current_sample_id:
            return False
        try:
            return bool(self.gating.is_custom_gate(self.current_sample_id, self.gate_name))
        except Exception:
            return False


def _menu_with(label, base_items=('Delete Gate',)):
    """A menu built once (as the ROIs do) then shown via the shared helper."""
    menu = QMenu()
    for item in base_items:
        menu.addAction(item)
    menu.exec = lambda *_: None  # never actually block on a popup
    exec_gate_menu(menu, label, None)
    return menu


def _labels(menu):
    return [a.text() for a in menu.actions() if not a.isSeparator()]


def test_template_gate_offers_customise(qtbot):
    label = _FakeLabel()
    menu = _menu_with(label)
    assert _labels(menu) == ['Delete Gate', "Customise 'Lymphocytes' for this sample"]


def test_customised_gate_offers_revert_and_adopt(qtbot):
    label = _FakeLabel(custom=['Lymphocytes'])
    menu = _menu_with(label)
    assert _labels(menu) == [
        'Delete Gate',
        "Revert 'Lymphocytes' to template",
        "Adopt 'Lymphocytes' custom gate as template",
    ]


def test_actions_emit_the_same_bus_signals_as_the_tree(qtbot):
    label = _FakeLabel(mode='unmixed')
    menu = _menu_with(label)
    menu.actions()[-1].trigger()
    assert label.bus.customiseGateRequested.emitted == [('unmixed', 'Lymphocytes')]

    label = _FakeLabel(mode='unmixed', custom=['Lymphocytes'])
    menu = _menu_with(label)
    menu.actions()[-2].trigger()
    menu.actions()[-1].trigger()
    assert label.bus.revertGateRequested.emitted == [('unmixed', 'Lymphocytes')]
    assert label.bus.adoptGateRequested.emitted == [('unmixed', 'Lymphocytes')]


def test_actions_follow_a_rename(qtbot):
    """The gate name is read at trigger time, so renaming keeps the action valid."""
    label = _FakeLabel()
    menu = _menu_with(label)
    label.gate_name = 'Live cells'
    menu.actions()[-1].trigger()
    assert label.bus.customiseGateRequested.emitted == [('raw', 'Live cells')]


def test_menu_is_rebuilt_on_every_right_click(qtbot):
    """Customising a gate must flip the menu without leaving stale entries."""
    label = _FakeLabel()
    menu = _menu_with(label)
    assert _labels(menu) == ['Delete Gate', "Customise 'Lymphocytes' for this sample"]

    label.gating = _FakeGating(custom=['Lymphocytes'])
    exec_gate_menu(menu, label, None)
    assert _labels(menu) == [
        'Delete Gate',
        "Revert 'Lymphocytes' to template",
        "Adopt 'Lymphocytes' custom gate as template",
    ]

    label.gating = _FakeGating()
    exec_gate_menu(menu, label, None)
    assert _labels(menu) == ['Delete Gate', "Customise 'Lymphocytes' for this sample"]


def test_customise_is_greyed_out_with_no_sample_loaded(qtbot):
    """Visible but disabled: hiding it makes the feature look missing (which is
    exactly what happened when the selected tube was empty)."""
    menu = _menu_with(_FakeLabel(sample_id=None))
    assert _labels(menu) == ['Delete Gate', "Customise 'Lymphocytes' for this sample"]
    assert not menu.actions()[-1].isEnabled()


def test_customise_is_enabled_once_a_sample_is_loaded(qtbot):
    menu = _menu_with(_FakeLabel())
    assert menu.actions()[-1].isEnabled()


def test_customise_is_offered_when_the_gate_state_cannot_be_read(qtbot):
    """A gate missing from the strategy still offers the action rather than
    silently dropping it."""
    menu = _menu_with(_FakeLabel(raises=True))
    assert _labels(menu) == ['Delete Gate', "Customise 'Lymphocytes' for this sample"]


@pytest.mark.parametrize('label', [
    None,                                   # a menu whose owner has no label yet
    _FakeLabel(bus=False),                  # no event bus
    _FakeLabel(gate_name='root'),           # root is not gateable
])
def test_nothing_is_added_when_there_is_no_gate_to_act_on(qtbot, label):
    menu = _menu_with(label)
    assert _labels(menu) == ['Delete Gate']


# --- against the real ROI classes -------------------------------------------
#
# The tests above drive the shared helper. These build the actual ROIs and send
# them a real right-click, so the wiring each ROI needs (its label, and the
# gate_label handed to the range region / quadrant target) is covered too.

import flowkit as fk
from flowkit import Dimension
from flowkit._models.gates import RectangleGate as FkRectangleGate
from PySide6.QtCore import Qt, QPointF
from PySide6.QtWidgets import QFrame
import pyqtgraph as pg

from honeychrome.view_components.event_bus import EventBus
from honeychrome.view_components.regions_of_interest import (
    EllipseROI, PolygonROI, QuadROI, RangeROI, RectangleROI,
)


class _FakePlotWidget(QFrame):
    """The bit of CytometryPlotWidget an ROI label reaches for, with the real
    EventBus and a real ViewBox parented to it as CytometryPlotWidget does."""

    def __init__(self, gating, sample_id='A1.fcs'):
        super().__init__()
        self.bus = EventBus()
        self.data_for_cytometry_plots = {'sample_id': sample_id, 'statistics': {}}
        self.gating = gating
        self.graphics_widget = pg.GraphicsLayoutWidget()
        self.vb = pg.ViewBox()
        self.vb.setParent(self)
        self.graphics_widget.addItem(self.vb, row=0, col=0)

        self.requests = []
        self.bus.customiseGateRequested.connect(
            lambda *a: self.requests.append(('customise', *a)))
        self.bus.revertGateRequested.connect(
            lambda *a: self.requests.append(('revert', *a)))
        self.bus.adoptGateRequested.connect(
            lambda *a: self.requests.append(('adopt', *a)))


class _RightClick:
    def __init__(self):
        self.accepted = False

    def button(self):
        return Qt.MouseButton.RightButton

    def screenPos(self):
        return QPointF(0, 0)

    def accept(self):
        self.accepted = True


@pytest.fixture
def gating():
    """A real flowkit strategy holding one 1-D range gate named 'R1'."""
    gs = fk.GatingStrategy()
    dim = Dimension('Time', compensation_ref='uncompensated', range_min=0.2, range_max=0.8)
    gs.add_gate(FkRectangleGate('R1', dimensions=[dim]), gate_path=('root',))
    return gs


def _build(kind, gating, vb):
    if kind == 'rectangle':
        roi = RectangleROI((0.1, 0.1), (0.4, 0.4), 'R1', gating, 'raw', vb)
        return roi, roi
    if kind == 'ellipse':
        roi = EllipseROI((0.1, 0.1), (0.4, 0.4), 0, 'R1', gating, 'raw', vb)
        return roi, roi
    if kind == 'polygon':
        roi = PolygonROI([(0.1, 0.1), (0.5, 0.1), (0.3, 0.5)], 'R1', gating, 'raw', vb)
        return roi, roi
    if kind == 'range':
        roi = RangeROI(0.2, 0.8, 'R1', gating, 'raw', vb)
        return roi, roi.region      # the shaded band carries the menu
    if kind == 'quadrant':
        roi = QuadROI(0.5, 0.5, 'R1', gating, 'raw', vb)
        return roi, roi.target      # the centre marker carries the menu
    raise AssertionError(kind)


@pytest.mark.parametrize('kind', ['rectangle', 'ellipse', 'polygon', 'range', 'quadrant'])
def test_right_clicking_a_real_roi_offers_the_custom_gate_actions(qtbot, gating, kind):
    widget = _FakePlotWidget(gating)
    qtbot.addWidget(widget)
    roi, clickable = _build(kind, gating, widget.vb)
    clickable.menu.exec = lambda *_: None

    event = _RightClick()
    clickable.mouseClickEvent(event)
    assert event.accepted
    assert "Customise 'R1' for this sample" in _labels(clickable.menu)


@pytest.mark.parametrize('kind', ['rectangle', 'range', 'quadrant'])
def test_a_real_roi_flips_to_revert_and_adopt_once_customised(qtbot, gating, kind):
    widget = _FakePlotWidget(gating)
    qtbot.addWidget(widget)
    roi, clickable = _build(kind, gating, widget.vb)
    clickable.menu.exec = lambda *_: None

    import copy
    gating.add_gate(copy.deepcopy(gating.get_gate('R1')),
                    gate_path=('root',), sample_id='A1.fcs')

    clickable.mouseClickEvent(_RightClick())
    labels = _labels(clickable.menu)
    assert "Revert 'R1' to template" in labels
    assert "Adopt 'R1' custom gate as template" in labels
    assert "Customise 'R1' for this sample" not in labels


def test_real_roi_reads_the_sample_live_not_the_dict_captured_at_construction(qtbot, gating):
    """The plot widget rebinds data_for_cytometry_plots when the mode changes,
    so an ROI built earlier must still see the current sample."""
    widget = _FakePlotWidget(gating, sample_id='A1.fcs')
    qtbot.addWidget(widget)
    roi = RectangleROI((0.1, 0.1), (0.4, 0.4), 'R1', gating, 'raw', widget.vb)
    roi.menu.exec = lambda *_: None

    import copy
    gating.add_gate(copy.deepcopy(gating.get_gate('R1')),
                    gate_path=('root',), sample_id='B2.fcs')

    # a whole new dict, as happens on a mode switch
    widget.data_for_cytometry_plots = {'sample_id': 'B2.fcs', 'statistics': {}}
    roi.mouseClickEvent(_RightClick())
    assert "Revert 'R1' to template" in _labels(roi.menu)
