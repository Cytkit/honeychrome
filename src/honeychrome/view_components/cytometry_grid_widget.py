from PySide6.QtWidgets import QApplication, QWidget, QGridLayout, QLabel, QVBoxLayout, QScrollArea, QFrame, QMessageBox, \
    QDialog, QPushButton
from PySide6.QtGui import QColor, QPalette, QResizeEvent
from PySide6.QtCore import Qt, QTimer, Slot
import sys

import honeychrome.settings as settings
from honeychrome.view_components.cytometry_plot_widget import CytometryPlotWidget
from honeychrome.view_components.cytometry_toolbar_popout import CytometryToolbarPopout
from honeychrome.view_components.new_plot_widget import NewPlotWidget

import logging
logger = logging.getLogger(__name__)

class CytometryGridWidget(QScrollArea):
    def __init__(self, bus=None, mode=None, gating_tree=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.main_layout = QVBoxLayout(self)
        self.setWidgetResizable(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.container = QWidget(parent=self)
        self.layout = QGridLayout(self.container)
        self.layout.setSpacing(5)
        self.setWidget(self.container)

        self.debounce_timer = QTimer(parent=self)
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self.init_grid)

        # Track which grid cells are occupied
        self.n_columns = None
        self.cytometry_plot_real_width = None
        self.occupied = None
        self.row = None
        self.last_row = None

        self.bus = bus
        self.mode = mode
        self.gating_tree = gating_tree
        self.toolbar = None
        self.data_for_cytometry_plots = None
        self.plot_widgets = []

        if self.bus is not None:
            self.bus.showNewPlot.connect(self.add_new_plot)

        self.selected_plot = None

    def set_toolbar(self, toolbar):
        self.toolbar = toolbar

    def select_plot(self, widget):
        # Deselect previous
        if widget != self.selected_plot:
            self.deselect_plot()
            # Select new
            widget.setFrameShape(QFrame.Box)
            widget.setFrameShadow(QFrame.Plain)
            widget.setLineWidth(1)
            self.selected_plot = widget
            self.toolbar.update_button_state(self.selected_plot.plot['type'])

    def deselect_plot(self):
        if self.selected_plot is not None:
            self.selected_plot.setFrameShape(QFrame.NoFrame)
            self.selected_plot = None
            self.toolbar.update_button_state(None)

    def open_plot_in_modal(self, plot_widget): # Note: called by child CytometryPlotWidget
        self.debounce_timer.stop()
        self.debounce_timer.blockSignals(True)

        # Create modal dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Pop-out plot")
        dialog.setModal(True)
        dialog.setMinimumSize(800, 800)

        # Create layout for dialog
        layout = QVBoxLayout()
        cytometry_toolbar_popout = CytometryToolbarPopout(self.bus, parent=self)
        cytometry_toolbar_popout.update_button_state(self.selected_plot.plot['type'])

        layout.addWidget(cytometry_toolbar_popout)
        plot_widget._mouse_events_enabled = False
        layout.addWidget(plot_widget)
        logger.info(f'CytometryGridWidget {self.mode}: popped out plot {plot_widget.n_in_plot_sequence}')

        # Add close button
        close_btn = QPushButton("Pop back in")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.setLayout(layout)

        # Show dialog and wait for it to close
        dialog.exec()

        # re initialise grid when closed
        plot_widget._mouse_events_enabled = True
        self.place_tile(plot_widget, 1, 1)
        # self.debounce_timer.start(300)
        self.init_grid()
        self.debounce_timer.blockSignals(False)

    @Slot()
    def permute_plot_widgets(self, destination): #todo fix bug: permuted plot seems to lose connection ***
        if self.selected_plot is not None:
            plots = self.data_for_cytometry_plots['plots']
            histograms = self.data_for_cytometry_plots['histograms']
            widgets = self.plot_widgets

            n = widgets.index(self.selected_plot)
            N = len(plots)-1
            scrollbar = self.verticalScrollBar()
            if destination == 'start':
                m = 0
                scrollbar.setValue(scrollbar.minimum())
            elif destination == 'left':
                m = max([n-1, 0])
            elif destination == 'right':
                m = min([n+1, N])
            else: # destination == 'end':
                m = N
                scrollbar.setValue(scrollbar.maximum())

            if destination == 'left' or destination == 'right':
                plots[n], plots[m] = plots[m], plots[n]
                widgets[n], widgets[m] = widgets[m], widgets[n]
                widgets[n].n_in_plot_sequence, widgets[m].n_in_plot_sequence = widgets[m].n_in_plot_sequence, widgets[n].n_in_plot_sequence
                logger.info(f'CytometryGridWidget {self.mode}: Permuting plots. len(histograms), n, m = {n}, {m}, {len(histograms)}')
                histograms[n], histograms[m] = histograms[m], histograms[n]
            else: # start or end
                plot = plots[n]
                plots.pop(n)
                plots.insert(m, plot)
                widget = widgets[n]
                widgets.pop(n)
                widgets.insert(m, widget)
                histogram = histograms[n]
                histograms.pop(n)
                histograms.insert(m, histogram)
                for i, widget in enumerate(widgets):
                    widgets[i].n_in_plot_sequence = i

            self.bus.autoSaveRequested.emit()
            self.debounce_timer.start(300)

    @Slot(str)
    def add_new_plot(self, mode):
        if mode == self.mode:
            n_in_plot_sequence = len(self.data_for_cytometry_plots['plots'])-1
            plot = self.data_for_cytometry_plots['plots'][-1]
            logger.info(f'CytometryGridWidget {self.mode}: new plot {n_in_plot_sequence}, {self.data_for_cytometry_plots['plots'][n_in_plot_sequence]}')
            new_widget = CytometryPlotWidget(bus=self.bus, mode=self.mode, n_in_plot_sequence=n_in_plot_sequence, plot=plot,
                                             data_for_cytometry_plots=self.data_for_cytometry_plots, parent=self.container)
            self.plot_widgets.append(new_widget)
            self.select_plot(new_widget)
            self.debounce_timer.start(300)
            if self.bus is not None:
                self.bus.plotChangeRequested.emit(self.mode, n_in_plot_sequence)

    def show_new_plot_widget(self):
        new_plot_widget = NewPlotWidget(bus=self.bus, mode=self.mode, data_for_cytometry_plots=self.data_for_cytometry_plots)
        self.place_tile(new_plot_widget, 1, 1)
        self.debounce_timer.start(300)
        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum()+1000)

    def delete_current_plot(self):
        if self.selected_plot is not None:
            if len(self.selected_plot.rois):
                # Create the message box
                msg_box = QMessageBox()
                msg_box.setWindowTitle("Delete current plot")
                msg_box.setText("Do you wish to delete the child gates?")
                msg_box.setIcon(QMessageBox.Question)

                # Add Yes and Cancel buttons
                yes_button = msg_box.addButton("Yes", QMessageBox.YesRole)
                cancel_button = msg_box.addButton("Cancel", QMessageBox.RejectRole)

                # Show the dialog and wait for user response
                msg_box.exec()

                # Check which button was clicked
                if msg_box.clickedButton() == yes_button:
                    self.proceed_to_delete_plot()
                else:
                    logger.info("Action cancelled")
            else:
                self.proceed_to_delete_plot()

    def proceed_to_delete_plot(self):
        plots = self.data_for_cytometry_plots['plots']
        hists = self.data_for_cytometry_plots['histograms']
        widgets = self.plot_widgets

        n = widgets.index(self.selected_plot)
        self.selected_plot.remove_gate_and_roi()
        plots.pop(n)
        hists.pop(n)
        widgets.pop(n)
        self.selected_plot.deleteLater()
        self.selected_plot = None
        self.debounce_timer.start(300)
        if self.bus:
            self.bus.autoSaveRequested.emit()

    def init_plots(self, data_for_cytometry_plots):
        self.data_for_cytometry_plots = data_for_cytometry_plots

        # old_widgets = self.plot_widgets[:]
        self.plot_widgets = []
        self.n_columns = None

        # # Properly delete old widgets
        # for widget in old_widgets:
        #     # widget.setParent(None)  # Remove parent first
        #     widget.deleteLater()
        # Process events to help cleanup (optional)
        # QApplication.processEvents()

        self.selected_plot = None
        self.clear_layout(self.layout)

        # Create new widgets efficiently
        if self.data_for_cytometry_plots['plots']:
            for n, plot in enumerate(self.data_for_cytometry_plots['plots']):
                new_widget = CytometryPlotWidget(bus=self.bus, mode=self.mode, n_in_plot_sequence=n, plot=plot,
                    data_for_cytometry_plots=self.data_for_cytometry_plots, parent=self.container)
                self.plot_widgets.append(new_widget)

            self.debounce_timer.start(300)

    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)

        # Check if width actually changed
        if event.oldSize().width() != event.size().width():
            self.debounce_timer.start(300)

    def init_grid(self):
        if self.data_for_cytometry_plots is None:
            return
        # called every time width changes
        if self.data_for_cytometry_plots['plots']:
            old_n_columns = self.n_columns
            self.n_columns = max([self.width() // settings.cytometry_plot_width_target_retrieved, 1])
            self.cytometry_plot_real_width = (self.width() - 45)// self.n_columns
            for n in range(self.n_columns):
                self.layout.setColumnMinimumWidth(n, self.cytometry_plot_real_width)

            if True: # old_n_columns != self.n_columns:
                self.occupied = []
                self.row = 0
                self.last_row = 0
                if self.n_columns < 1:
                    self.n_columns = 1
                elif self.n_columns > 10:
                    self.n_columns = 10

                logger.info(f'Cytometry Grid Widget: setting width to {self.n_columns} columns')

                # # Create container widget
                # if self.container is not None:
                #     self.container.deleteLater()
                #     self.container = None
                # self.container = QWidget(parent=self)
                # self.layout = QGridLayout(self.container)  # Layout assigned to container

                # Place each plot as a plot_widget
                for n, plot in enumerate(self.data_for_cytometry_plots['plots']):

                    # set width to 3 if plot is ribbon and width not set
                    if plot['type'] == 'ribbon' and 'width' not in plot.keys():
                        plot['width'] = 3

                    # set tile to 1x1 if width/height not previously set
                    w = min([plot.get("width", 1), self.n_columns])
                    h = plot.get("height", 1)
                    # Create tile widget
                    plot_widget = self.plot_widgets[n]

                    self.place_tile(plot_widget, w, h)
                    plot_widget.n_in_plot_sequence = n

                    # print(self.n_columns)
                    # print(n, w, h, self.data_for_cytometry_plots['plots'])
                    # print(self.occupied)


    def fits(self, row, col, w, h):
        """Check if tile of size (w,h) fits at (row,col)."""
        for r in range(row, row + h):
            for c in range(col, col + w):
                if c >= self.n_columns or (r, c) in self.occupied:
                    return False
        return True

    def occupy(self, row, col, w, h):
        """Mark cells as occupied."""
        for r in range(row, row + h):
            for c in range(col, col + w):
                self.occupied.append((r, c))

    def place_tile(self, tile, w, h):
        tile.setMinimumSize(self.cytometry_plot_real_width * w, self.cytometry_plot_real_width * h)

        placed = False
        # Try to find first spot where it fits
        while not placed:
            for col in range(self.n_columns):
                # print(self.row, col, w, h, self.fits(self.row, col, w, h))
                if self.fits(self.row, col, w, h):
                    pass
                elif col==0 and self.fits(self.row, col, self.n_columns, h):
                    # this is the case of the wide plot, e.g. ribbon plot, having to squeeze
                    w = self.n_columns
                    tile.setMinimumWidth(self.width())
                else:
                    continue

                # Add widget spanning w columns × h rows
                self.layout.addWidget(tile, self.row, col, h, w)
                self.occupy(self.row, col, w, h)
                self.last_row = max(self.last_row, self.row + h)
                placed = True
                # print('placed', self.row, col, w, h)
                break
            if not placed:
                self.row += 1  # move down a row and try again

        self.set_last_row_stretch()

    def set_last_row_stretch(self):
        # Clear stretch from all rows
        for row in range(self.last_row):
            self.layout.setRowStretch(row, 0)

        # Set stretch only on the last row
        self.layout.setRowStretch(self.last_row, 1)  # Stretch factor of 1

    def clear_layout(self, layout):
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            else:
                self.clear_layout(item.layout())


if __name__ == "__main__":
    from honeychrome.controller import Controller
    from pathlib import Path
    import pyqtgraph as pg
    app = QApplication(sys.argv)

    # app.setStyle("Fusion")

    kc = Controller()
    base_directory = Path.home() / 'spectral_cytometry'
    experiment_name = base_directory / '20240620 Spectral Symposium-poor cell unmixed'
    experiment_path = experiment_name.with_suffix('.kit')
    kc.load_experiment(experiment_path) # note this loads first sample too and runs calculate all histograms and statistics

    # note every time tab is changed (or every time plots or transforms changed), calculate all histograms and statistics
    kc.current_mode = 'unmixed'
    kc.experiment.cytometry['plots'][0]['channel_x'] = 'FSC-A'
    kc.initialise_data_for_cytometry_plots()

    window = CytometryGridWidget()
    window.init_plots(kc.data_for_cytometry_plots_raw)
    window.setWindowTitle("Grid-Filling Tiles (integer cell units)")
    window.resize(500, 400)
    window.show()

    sys.exit(app.exec())
