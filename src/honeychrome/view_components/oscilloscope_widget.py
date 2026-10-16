import numpy as np
from PySide6.QtCore import Slot, QTimer, Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from queue import Empty
import pyqtgraph as pg

from honeychrome.settings import n_time_points_in_event, adc_rate, peak_detection_channel_retrieved, instrument_settings, peak_detection_channel, threshold
from honeychrome.settings import analyser_target_repeat_time, line_colors, adc_channels, scatter_channels, fluorescence_channels
from honeychrome.view_components.profiles_viewer import FlowLayout, LegendEntry

import logging
logger = logging.getLogger(__name__)

class OscilloscopeWidget(QWidget):
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, controller, bus):
        if not hasattr(self, '_initialized'):
            super().__init__()
            self._initialized = True
            self.controller = controller
            self.bus = bus

            self.setGeometry(10, 10, 800, 700)
            self.setWindowTitle("Oscilloscope Viewer")
            self.layout = QVBoxLayout(self)
            self.layout.setContentsMargins(0, 0, 0, 0)

            self.label = QLabel('')

            self.plot_widget1 = pg.PlotWidget()
            self.plot_widget1.setLabel('left', 'Intensity [uv]')
            self.plot_widget1.setLabel('bottom', 'Time', units='us')
            self.plot_widget1.showGrid(x=True, y=True, alpha=0.3)

            self.plot_widget2 = pg.PlotWidget()
            self.plot_widget2.setLabel('left', 'Intensity [uv]')
            self.plot_widget2.setLabel('bottom', 'Time', units='us')
            self.plot_widget2.showGrid(x=True, y=True, alpha=0.3)

            self.plot_widget2.setXLink(self.plot_widget1)

            # ---- Flowing Legend ----
            self.legendContainer1 = QWidget()
            self.legendContainer2 = QWidget()
            self.legendLayout1 = FlowLayout(self.legendContainer1)
            self.legendLayout2 = FlowLayout(self.legendContainer2)

            self.layout.addWidget(self.label)
            self.layout.addWidget(self.plot_widget1)
            self.layout.addWidget(self.legendContainer1)
            self.layout.addWidget(self.plot_widget2)
            self.layout.addWidget(self.legendContainer2)

            trace_length = n_time_points_in_event
            trace_mid = trace_length // 2
            self.x = (np.arange(n_time_points_in_event) - trace_mid) / adc_rate

            self.scatter_channel_indices = [i for i,channel in enumerate(adc_channels) if channel in scatter_channels]
            self.fluorescence_channel_indices = [i for i,channel in enumerate(adc_channels) if channel in fluorescence_channels]
            for i in self.scatter_channel_indices:
                color = line_colors[i % len(line_colors)]
                entry = LegendEntry(color, adc_channels[i])
                self.legendLayout1.addWidget(entry)

            for i in self.fluorescence_channel_indices:
                color = line_colors[i % len(line_colors)]
                entry = LegendEntry(color, adc_channels[i])
                self.legendLayout2.addWidget(entry)

            self.peak_detection_channel_index = adc_channels.index(str(instrument_settings.value("peak_detection_channel", peak_detection_channel)))
            self.threshold = instrument_settings.value("threshold", threshold, type=int) * 1000

            # Use a timer to check the queue periodically
            self.trace = None
            self.timer = QTimer()
            self.timer.timeout.connect(self.check_queue)
            self.timer.setInterval(int(analyser_target_repeat_time * 1000 / 4))
            self.timer.start()

    @Slot()
    def check_queue(self):
        try:
            self.trace = self.controller.oscilloscope_traces_queue.get_nowait()
            self._update_plot()
        except Empty:
            pass

    @Slot(dict)
    def _update_plot(self):
        """Thread-safe method to update the plot"""
        try:
            # print(self.trace) # debug only

            start = self.trace['n_start']
            end = self.trace['n_end']
            peak_span = np.array([self.x[start], self.x[end]])
            peak_start = np.array([self.x[start], self.x[start]])
            peak_end = np.array([self.x[end], self.x[end]])
            self.plot_widget1.clear()
            for i in self.scatter_channel_indices:
                color = line_colors[i % len(line_colors)]
                y = self.trace['traces'][i]

                dotted_pen = pg.mkPen(color=color, width=2, style=Qt.DotLine)
                dashed_pen = pg.mkPen(color=color, width=2, style=Qt.DashLine)
                solid_pen = pg.mkPen(color=color, width=2, style=Qt.SolidLine)

                self.plot_widget1.plot(self.x, y, pen=dashed_pen)
                self.plot_widget1.plot(self.x, y-self.trace['baselines'][i], pen=solid_pen)
                self.plot_widget1.plot(peak_span, self.trace['baselines'][i] * np.ones(2), pen=dotted_pen)
                self.plot_widget1.plot(peak_start, np.array([0, self.trace['traces'][i][start:end].max()]), pen=solid_pen)
                self.plot_widget1.plot(peak_end, np.array([0, self.trace['traces'][i][start:end].max()]), pen=solid_pen)

                if i==self.peak_detection_channel_index:
                    pen = pg.mkPen(color=color, width=5, style=Qt.DotLine)
                    self.plot_widget1.plot(peak_span, self.threshold * np.ones(2), pen=pen)

            self.plot_widget2.clear()
            for i in self.fluorescence_channel_indices:
                color = line_colors[i % len(line_colors)]
                y = self.trace['traces'][i]

                dotted_pen = pg.mkPen(color=color, width=2, style=Qt.DotLine)
                dashed_pen = pg.mkPen(color=color, width=2, style=Qt.DashLine)
                solid_pen = pg.mkPen(color=color, width=2, style=Qt.SolidLine)

                self.plot_widget2.plot(self.x, y, pen=dashed_pen)
                self.plot_widget2.plot(self.x, y-self.trace['baselines'][i], pen=solid_pen)
                self.plot_widget2.plot(peak_span, self.trace['baselines'][i] * np.ones(2), pen=dotted_pen)
                self.plot_widget2.plot(peak_start, np.array([0, self.trace['traces'][i][start:end].max()]), pen=solid_pen)
                self.plot_widget2.plot(peak_end, np.array([0, self.trace['traces'][i][start:end].max()]), pen=solid_pen)

                if i==self.peak_detection_channel_index:
                    pen = pg.mkPen(color=color, width=5, style=Qt.DotLine)
                    self.plot_widget2.plot(peak_span, self.threshold * np.ones(2), pen=pen)

            self.label.setText(f"event_id={self.trace['event_id']} time={self.trace['time']}")





        except Exception as e:
            logger.info(f"Error updating plot: {e}")

    def closeEvent(self, event):
        self.timer.stop()
        event.accept()