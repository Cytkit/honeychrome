"""
Focus mode: more of the window for figures.

While focus mode is on, the main window hides the sample browser and, in the
tab on screen, every widget marked as page "chrome": help toggles, section
navigation lists, page heads and the controls above a plugin's figures, and
the section headers of accordion (QToolBox) pages, whose open section stays.
Turning it off shows them again. Widgets are marked once, where they are
built, with ``mark_focus_chrome``; accordions with ``mark_focus_headers``.

``hide_chrome`` and ``restore`` do the hiding for the main window: only
widgets that were showing are hidden, and only those are shown again, so a
widget a page had hidden for its own reasons stays hidden.
"""

from PySide6.QtWidgets import QAbstractButton, QToolBox, QWidget

__all__ = ['FOCUS_CHROME', 'mark_focus_chrome', 'mark_focus_headers', 'hide_chrome', 'restore']

FOCUS_CHROME = 'focusChrome'
_FOCUS_HEADERS = 'focusHeaders'
# Object name Qt gives the section buttons of a QToolBox.
_TOOLBOX_BUTTON = 'qt_toolbox_toolboxbutton'


def mark_focus_chrome(*widgets: QWidget):
    """Hide *widgets* in focus mode."""
    for widget in widgets:
        if widget is not None:
            widget.setProperty(FOCUS_CHROME, True)


def mark_focus_headers(toolbox: QToolBox):
    """Hide the section headers of *toolbox* in focus mode (its open
    section stays on screen)."""
    toolbox.setProperty(_FOCUS_HEADERS, True)


def _chrome(root: QWidget) -> list:
    out = [w for w in root.findChildren(QWidget) if w.property(FOCUS_CHROME)]
    for toolbox in root.findChildren(QToolBox):
        if toolbox.property(_FOCUS_HEADERS):
            out += [b for b in toolbox.findChildren(QAbstractButton)
                    if b.objectName() == _TOOLBOX_BUTTON and b.parent() is toolbox]
    return out


def hide_chrome(root: QWidget) -> list:
    """Hide the marked widgets under *root* that are showing; returns them."""
    hidden = [w for w in _chrome(root) if not w.isHidden()]
    for widget in hidden:
        widget.hide()
    return hidden


def restore(hidden: list):
    """Show again the widgets *hide_chrome* hid (those still alive)."""
    for widget in hidden:
        try:
            widget.show()
        except RuntimeError:
            pass    # deleted since
