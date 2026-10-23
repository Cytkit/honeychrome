import numpy as np
from PySide6.QtCore import Slot, QTimer, Qt
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from queue import Empty
import pyqtgraph as pg

from honeychrome.settings import n_time_points_in_event, adc_rate, instrument_settings, peak_detection_channel, threshold
from honeychrome.settings import analyser_target_repeat_time, line_colors, spectral_array_colors, adc_channels, scatter_channels, fluorescence_channels
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
        if getattr(self, '_initialized', False):
            return
        super().__init__()
        self._initialized = True
        self.controller = controller
        self.bus = bus

        self.setGeometry(10, 10, 800, 700)
        self.setWindowTitle("Oscilloscope Viewer")
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel('')
        self._last_label_text = None
        self.local_trace_queue = []

        # ---- Plot widgets ----
        self.plot_widget1 = pg.PlotWidget()
        self.plot_widget1.setLabel('left', 'Intensity [uv]')
        self.plot_widget1.setLabel('bottom', 'Time', units='us')
        self.plot_widget1.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget1.setClipToView(True)
        self.plot_widget1.setDownsampling(auto=True, mode='peak')

        self.plot_widget2 = pg.PlotWidget()
        self.plot_widget2.setLabel('left', 'Intensity [uv]')
        self.plot_widget2.setLabel('bottom', 'Time', units='us')
        self.plot_widget2.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget2.setClipToView(True)
        self.plot_widget2.setDownsampling(auto=True, mode='peak')
        self.plot_widget2.setXLink(self.plot_widget1)

        # ---- Flowing legend ----
        self.legendContainer1 = QWidget()
        self.legendContainer2 = QWidget()
        self.legendLayout1 = FlowLayout(self.legendContainer1)
        self.legendLayout2 = FlowLayout(self.legendContainer2)

        self.layout.addWidget(self.label)
        self.layout.addWidget(self.plot_widget1)
        self.layout.addWidget(self.legendContainer1)
        self.layout.addWidget(self.plot_widget2)
        self.layout.addWidget(self.legendContainer2)

        # ---- X axis (static) ----
        trace_length = n_time_points_in_event
        trace_mid = trace_length // 2
        self.x = (np.arange(n_time_points_in_event) - trace_mid) / adc_rate

        # ---- Channel selection ----
        self.scatter_channel_indices = [
            i for i, channel in enumerate(adc_channels) if channel in scatter_channels
        ]
        self.fluorescence_channel_indices = [
            i for i, channel in enumerate(adc_channels) if channel in fluorescence_channels
        ]

        # ---- Legend entries ----
        for i in self.scatter_channel_indices:
            color = line_colors[i % len(line_colors)]
            self.legendLayout1.addWidget(LegendEntry(color, adc_channels[i]))
        for i in self.fluorescence_channel_indices:
            color = spectral_array_colors[i % len(spectral_array_colors)]
            self.legendLayout2.addWidget(LegendEntry(color, adc_channels[i]))

        self.peak_detection_channel_index = adc_channels.index(
            str(instrument_settings.value("peak_detection_channel", peak_detection_channel))
        )
        self.threshold = instrument_settings.value("threshold", threshold, type=int) * 1000

        # ---- Pre-create pens (hoisted out of the hot path) ----
        self._scatter_solid_pens = [
            pg.mkPen(color=line_colors[i % len(line_colors)], width=2, style=Qt.SolidLine)
            for i in self.scatter_channel_indices
        ]
        self._scatter_dotted_pens = [
            pg.mkPen(color=line_colors[i % len(line_colors)], width=5, style=Qt.DotLine)
            for i in self.scatter_channel_indices
        ]
        self._fluorescence_solid_pens = [
            pg.mkPen(color=spectral_array_colors[i % len(spectral_array_colors)],
                     width=2, style=Qt.SolidLine)
            for i in self.fluorescence_channel_indices
        ]
        self._fluorescence_dotted_pens = [
            pg.mkPen(color=spectral_array_colors[i % len(spectral_array_colors)],
                     width=5, style=Qt.DotLine)
            for i in self.fluorescence_channel_indices
        ]

        # ---- Pre-create persistent PlotDataItems ----
        # For each channel we need:
        #   - a solid line for the trace
        #   - a dotted vertical at the peak start
        #   - a dotted vertical at the peak end
        # Optionally, on the peak-detection channel, a dotted threshold line.

        self.scatter_curves = []       # list[(channel_idx, PlotDataItem)]
        self.scatter_vstart = []       # list[(channel_idx, PlotDataItem)]
        self.scatter_vend = []         # list[(channel_idx, PlotDataItem)]
        self.scatter_thresh = None     # PlotDataItem or None

        for slot, i in enumerate(self.scatter_channel_indices):
            solid = self._scatter_solid_pens[slot]
            dotted = self._scatter_dotted_pens[slot]
            c = self.plot_widget1.plot([], [], pen=solid)
            v0 = self.plot_widget1.plot([], [], pen=dotted)
            v1 = self.plot_widget1.plot([], [], pen=dotted)
            self.scatter_curves.append((i, c))
            self.scatter_vstart.append((i, v0))
            self.scatter_vend.append((i, v1))
            if i == self.peak_detection_channel_index:
                self.scatter_thresh = self.plot_widget1.plot([], [], pen=dotted)

        self.fluorescence_curves = []
        self.fluorescence_vstart = []
        self.fluorescence_vend = []
        self.fluorescence_thresh = None

        for slot, i in enumerate(self.fluorescence_channel_indices):
            solid = self._fluorescence_solid_pens[slot]
            dotted = self._fluorescence_dotted_pens[slot]
            c = self.plot_widget2.plot([], [], pen=solid)
            v0 = self.plot_widget2.plot([], [], pen=dotted)
            v1 = self.plot_widget2.plot([], [], pen=dotted)
            self.fluorescence_curves.append((i, c))
            self.fluorescence_vstart.append((i, v0))
            self.fluorescence_vend.append((i, v1))
            if i == self.peak_detection_channel_index:
                self.fluorescence_thresh = self.plot_widget2.plot([], [], pen=dotted)

        # ---- Timer ----
        self.trace = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_queue)
        self.timer.setInterval(int(analyser_target_repeat_time * 1000 / 5))
        self.timer.start()

    @Slot()
    def start_polling(self):
        if not self.timer.isActive():
            self.timer.start(int(analyser_target_repeat_time * 1000 / 5))

    # ------------------------------------------------------------------
    # Queue handling
    # ------------------------------------------------------------------
    @Slot()
    def check_queue(self):
        """Drain the queue completely; render only the newest traces."""
        while True:
            try:
                latest = self.controller.oscilloscope_traces_queue.get_nowait()
                self.local_trace_queue.append(latest)
                self.send_baselines_to_instrument_driver(latest) #todo consider removing this or moving to controller / trace analyst
            except Empty:
                break
        if len(self.local_trace_queue) > 0:
            if len(self.local_trace_queue) > 5:
                self.local_trace_queue = self.local_trace_queue[-5:]
            self.trace = self.local_trace_queue.pop(0)
            self._update_plot()


    # ------------------------------------------------------------------
    # Plot update (in-place setData — no clear(), no new items)
    # ------------------------------------------------------------------
    @Slot()
    def _update_plot(self):
        """Update existing curves in place. Never create/destroy graphics items."""
        if self.trace is None:
            return
        try:
            if self.controller.peak_detection_threshold:
                self.threshold = self.controller.peak_detection_threshold * 1000 # uV

            start = int(self.trace['n_start'])
            end = int(self.trace['n_end'])

            x_start = self.x[start]
            x_end = self.x[end]
            peak_span = np.array([x_start, x_end])

            traces = self.trace['traces']
            baselines = self.trace['baselines']

            # ---------- Scatter plot ----------
            for slot, (i, curve) in enumerate(self.scatter_curves):
                y = traces[i] - baselines[i]
                curve.setData(self.x, y)

                bound_height = float(traces[i][start:end].max() - baselines[i])
                v0 = self.scatter_vstart[slot][1]
                v1 = self.scatter_vend[slot][1]
                v0.setData(np.array([x_start, x_start]),
                           np.array([0.0, bound_height]))
                v1.setData(np.array([x_end, x_end]),
                           np.array([0.0, bound_height]))

            if self.scatter_thresh is not None:
                self.scatter_thresh.setData(
                    peak_span,
                    np.array([self.threshold, self.threshold]),
                )

            # ---------- Fluorescence plot ----------
            for slot, (i, curve) in enumerate(self.fluorescence_curves):
                y = traces[i] - baselines[i]
                curve.setData(self.x, y)

                bound_height = float(traces[i][start:end].max() - baselines[i])
                v0 = self.fluorescence_vstart[slot][1]
                v1 = self.fluorescence_vend[slot][1]
                v0.setData(np.array([x_start, x_start]),
                           np.array([0.0, bound_height]))
                v1.setData(np.array([x_end, x_end]),
                           np.array([0.0, bound_height]))

            if self.fluorescence_thresh is not None:
                self.fluorescence_thresh.setData(
                    peak_span,
                    np.array([self.threshold, self.threshold]),
                )

            # ---------- Label ----------
            text = f"event_id={self.trace.get('event_id')} time={self.trace.get('time')}"
            if text != self._last_label_text:
                self.label.setText(text)
                self._last_label_text = text

            logger.info("_update_plot: event_id=%r", self.trace.get("event_id"))

        except Exception as e:
            logger.info(f"Error updating plot: {e}")

    # ------------------------------------------------------------------
    def closeEvent(self, event):
        self.timer.stop()
        event.accept()

    def send_baselines_to_instrument_driver(self, trace):
        self.controller.pipe_connection_instrument.send({'command': 'set_instrument_state', 'data': {'oscilloscope_trace': trace}})
        response = self.controller.pipe_connection_instrument.recv()
        logger.info(response)