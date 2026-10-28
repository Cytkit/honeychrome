"""Per-sample custom gate actions in the gating hierarchy tree's context menu.

A QuadrantGate's four quadrants appear in the tree as children, but FlowKit
refuses to hand them out on their own ("specify the owning QuadrantGate"), so
offering Customise / Revert / Adopt on them produced an action that silently did
nothing. Only the owning QuadrantGate may be customised.
"""
import pytest
import flowkit as fk
from flowkit import gates
from PySide6.QtWidgets import QMenu

from honeychrome.controller_components.functions import define_quad_gates, define_range_gate
from honeychrome.controller_components.transform import Transform
from honeychrome.view_components import gating_hierarchy_widget as ghw
from honeychrome.view_components.event_bus import EventBus
from honeychrome.view_components.gating_hierarchy_widget import GatingHierarchyWidget


@pytest.fixture
def gating():
    """A strategy with one plain range gate and one quadrant gate."""
    transforms = {'FSC-A': Transform(), 'SSC-A': Transform()}
    gs = fk.GatingStrategy()
    gs.add_gate(gates.RectangleGate(
        'Cells', dimensions=[define_range_gate(0.2, 0.8, 'FSC-A', transforms)]),
        gate_path=('root',))
    dividers, quadrants = define_quad_gates(0.5, 0.5, 'FSC-A', 'SSC-A', transforms)
    gs.add_gate(gates.QuadrantGate('Q1', dividers, quadrants), gate_path=('root',))
    return gs


class _RecordingMenu(QMenu):
    """A QMenu that records itself instead of blocking on a popup."""
    last = None

    def exec(self, *_args):
        _RecordingMenu.last = self


@pytest.fixture
def tree(qtbot, gating, monkeypatch):
    monkeypatch.setattr(ghw, 'QMenu', _RecordingMenu)
    widget = GatingHierarchyWidget(bus=EventBus(), mode='raw')
    qtbot.addWidget(widget)
    widget.model.set_dict(gating.get_gate_hierarchy(output='dict'), {})
    widget.tree_view.expandAll()
    widget.resize(400, 400)
    return widget


def _menu_for(tree, name):
    """Right-click the row called ``name`` and return its menu's action texts."""
    matches = tree.model.match(
        tree.model.index(0, 0), ghw.Qt.DisplayRole, name, 1,
        ghw.Qt.MatchRecursive | ghw.Qt.MatchExactly,
    )
    assert matches, f'{name!r} is not in the tree'
    _RecordingMenu.last = None
    tree.show_context_menu(tree.tree_view.visualRect(matches[0]).center())
    assert _RecordingMenu.last is not None, 'no context menu was shown'
    return [a.text() for a in _RecordingMenu.last.actions() if not a.isSeparator()]


def test_quadrants_are_listed_in_the_tree(tree):
    """Guard the premise: the quadrants really are rows a user can right-click."""
    names = []

    def walk(item):
        names.append(item.name)
        for child in item.child_items:
            walk(child)

    walk(tree.model.root_item)
    assert 'Q1' in names
    assert 'FSC-A+ SSC-A+' in names


def test_a_normal_gate_offers_customise(tree):
    assert "Customise 'Cells' for this sample" in _menu_for(tree, 'Cells')


def test_gating_tree_has_no_multiple_template_picker(tree):
    assert not hasattr(tree, 'template_bar')
    assert not hasattr(tree, 'template_combo')


def test_the_owning_quadrant_gate_offers_customise(tree):
    assert "Customise 'Q1' for this sample" in _menu_for(tree, 'Q1')


@pytest.mark.parametrize('quadrant', [
    'FSC-A+ SSC-A+', 'FSC-A+ SSC-A-', 'FSC-A- SSC-A+', 'FSC-A- SSC-A-',
])
def test_a_quadrant_offers_no_custom_gate_actions(tree, quadrant):
    labels = _menu_for(tree, quadrant)
    assert not any('Customise' in t or 'Revert' in t or 'Adopt' in t for t in labels)
    # the rest of the menu is untouched
    assert any('Copy hierarchy statistics' in t for t in labels)


def test_flowkit_still_refuses_to_return_a_quadrant(gating):
    """The reason the actions are hidden — fail loudly here if FlowKit changes."""
    from flowkit.exceptions import QuadrantReferenceError
    with pytest.raises(QuadrantReferenceError):
        gating.get_gate('FSC-A+ SSC-A+')
