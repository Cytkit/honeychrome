"""
cosine_qc_viewer.py
--------------------
QC widget in the AutoSpectral Control Cleaning box (Spectral Process tab).

Shows for the selected control:
  - Calculated spectrum (purple) from cleaned_events['spectrum']
  - Reference library spectrum (green) from spectral_reference_library.py
  - Cosine similarity score, surviving event count, empirical vs expected peak

Replaces AfCleaningViewer
"""
from __future__ import annotations

import logging

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel,
    QSizePolicy, QVBoxLayout, QWidget,
)

from honeychrome.settings import heading_style
from honeychrome.view_components.help_toggle_widget import WheelBlocker
from honeychrome.view_components.profiles_viewer import (
    BottomAxisVerticalTickLabels, TransparentPlotWidget,
)
from honeychrome.controller_components.cytometer_whitelist import (
    get_detector_laser_map, LASER_LABEL_COLORS,
)

logger = logging.getLogger(__name__)

# Controls below this cosine similarity to the reference library trigger a popup warning.
COSINE_QC_WARNING_THRESHOLD = 0.97

class CosineQCViewer(QFrame):
    """Spectral QC plot: calculated vs. reference spectrum with cosine score."""

    _COL_CALC = (160,  50, 220)   # purple — calculated spectrum
    _COL_REF  = ( 80, 200,  80)   # green  — reference library spectrum

    def __init__(self, bus, controller, parent=None):
        super().__init__(parent)
        self.bus        = bus
        self.controller = controller

        # ------------------------------------------------------------------
        # Layout
        # ------------------------------------------------------------------
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(2)

        title = QLabel('Spectral QC')
        title.setStyleSheet(heading_style)
        outer.addWidget(title)

        self._toggle = QCheckBox('Show spectral QC plot')
        self._toggle.setChecked(False)
        self._toggle.setEnabled(False)
        self._toggle.setToolTip('Run Clean Controls first.')
        outer.addWidget(self._toggle)

        self._content = QWidget()
        content_layout = QVBoxLayout(self._content)
        content_layout.setContentsMargins(0, 4, 0, 0)
        content_layout.setSpacing(6)

        # Control selector + status row
        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel('Control:'))
        self._combo = QComboBox()
        self._combo.setMinimumWidth(220)
        self._combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._combo.installEventFilter(WheelBlocker(self._combo))
        self._combo.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        ctrl_row.addWidget(self._combo)
        ctrl_row.addStretch()
        self._status = QLabel('')
        ctrl_row.addWidget(self._status)
        content_layout.addLayout(ctrl_row)

        # Legend row (coloured square + label, matching ProfilesViewer style)
        legend_row = QHBoxLayout()
        for color, text in (
            (self._COL_CALC, 'Calculated spectrum'),
            (self._COL_REF,  'Reference spectrum'),
        ):
            r, g, b = color
            swatch = QLabel()
            swatch.setFixedSize(14, 14)
            swatch.setStyleSheet(
                f'background-color: rgb({r},{g},{b}); border: 1px solid #444;'
            )
            legend_row.addWidget(swatch)
            legend_row.addWidget(QLabel(text))
            legend_row.addSpacing(10)
        legend_row.addStretch()
        content_layout.addLayout(legend_row)

        # Plot — same axis class and widget type as ProfilesViewer
        self._axis_bottom = BottomAxisVerticalTickLabels(angle=45)
        self._plot_widget = TransparentPlotWidget(
            axisItems={'bottom': self._axis_bottom}
        )
        self._plot_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._plot_widget.setMinimumHeight(200)
        vb = self._plot_widget.getViewBox()
        vb.setMenuEnabled(False)
        self._plot_widget.setLabel('left', 'Normalised intensity')
        self._plot_widget.setLabel('bottom', 'Channel')
        self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self._plot_widget.setMouseEnabled(x=False, y=False)
        self._plot_widget.setYRange(0, 1, padding=0)

        content_layout.addWidget(self._plot_widget, stretch=1)

        self._content.setVisible(False)
        outer.addWidget(self._content)

        # ------------------------------------------------------------------
        # Signal connections
        # ------------------------------------------------------------------
        self._toggle.toggled.connect(self._content.setVisible)
        self._toggle.toggled.connect(self._on_toggle)
        self._combo.currentTextChanged.connect(self._refresh_plot)

        if self.bus:
            self.bus.cleaningResultsReady.connect(self._on_cleaning_results_ready)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def refresh_combo(self):
        """Repopulate the control combo from cleaned_events, in spectral model order."""
        cleaned        = self.controller.cleaned_events
        spectral_model = self.controller.experiment.process.get('spectral_model', [])
        model_order    = [c['label'] for c in spectral_model if 'label' in c]
        labels         = [lbl for lbl in model_order
                          if lbl in cleaned and cleaned[lbl].get('spectrum')]
        
        current = self._combo.currentText()
        self._combo.blockSignals(True)
        self._combo.clear()
        self._combo.addItems(labels)
        if current in labels:
            self._combo.setCurrentText(current)
        self._combo.blockSignals(False)

        has_data = bool(labels)
        self._toggle.setEnabled(has_data)
        self._toggle.setToolTip(
            'Show spectral QC plot for the selected control.'
            if has_data else
            'Run Clean Controls first.'
        )

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    @Slot()
    def _on_cleaning_results_ready(self):
        self.refresh_combo()
        if self._toggle.isChecked():
            self._refresh_plot()

    @Slot(bool)
    def _on_toggle(self, checked: bool):
        if checked:
            self.refresh_combo()
            self._refresh_plot()

    @Slot(str)
    def _refresh_plot(self, _=''):
        if not self._toggle.isChecked():
            return
        label = self._combo.currentText()
        if not label:
            self._plot_widget.clear()
            self._status.setText('')
            return
        self._draw(label)

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def _draw(self, label: str):
        self._plot_widget.clear()

        cleaned = self.controller.cleaned_events.get(label)
        if not cleaned or not cleaned.get('spectrum'):
            self._status.setText('No cleaned spectrum for this control.')
            return

        spectrum = np.array(cleaned['spectrum'], dtype=float)

        # Resolve fluorescence channel names
        fluor_ch_ids = (cleaned.get('fluor_ch_ids')
                        or self.controller.filtered_raw_fluorescence_channel_ids)
        pnn = self.controller.experiment.settings['raw']['event_channels_pnn']
        try:
            fluor_pnn = [pnn[i] for i in fluor_ch_ids]
        except (IndexError, KeyError) as exc:
            logger.debug(f'CosineQCViewer: channel name resolution failed: {exc}')
            fluor_pnn = [str(i) for i in range(len(spectrum))]

        x = np.arange(len(spectrum)) + 0.5

        # Calculated spectrum
        self._plot_widget.plot(
            x, spectrum,
            pen=pg.mkPen(self._COL_CALC, width=2),
        )

        # Reference spectrum + cosine similarity
        cs            = None
        cytometer_key = cleaned.get('cytometer_key')
        if cytometer_key:
            try:
                from honeychrome.controller_components.spectral_reference_library import (
                    load_reference_library, cosine_similarity_to_reference,
                )
                ref = load_reference_library(cytometer_key)
                if ref is not None and label in ref.index:
                    common = [c for c in fluor_pnn if c in ref.columns]
                    if common:
                        ref_vals = np.zeros(len(spectrum))
                        for c in common:
                            ref_vals[fluor_pnn.index(c)] = ref.loc[label, c]
                        self._plot_widget.plot(
                            x, ref_vals,
                            pen=pg.mkPen(self._COL_REF, width=2),
                        )
                        cs = cosine_similarity_to_reference(
                            spectrum, fluor_pnn, label, cytometer_key
                        )
            except Exception as exc:
                logger.debug(
                    f'CosineQCViewer: reference lookup failed for "{label}": {exc}'
                )

        # X-axis: vertical tick labels with per-laser colour coding
        LABEL_DENSITY_TARGET = 80
        stride    = max(1, round(len(fluor_pnn) / LABEL_DENSITY_TARGET))
        all_pairs = [(i + 0.5, fluor_pnn[i].removesuffix('-A')) for i in range(len(fluor_pnn))]
        ticks     = [
            [(m, label_) for m, label_ in all_pairs if round(m - 0.5) % stride == 0],
            [],
        ]
        db_col = cleaned.get('cytometer_key')
        if db_col:
            detector_laser_map = get_detector_laser_map(db_col)
            tick_colors = {
                label_: LASER_LABEL_COLORS[laser]
                for _, label_ in all_pairs
                if (laser := detector_laser_map.get(label_ + '-A')) in LASER_LABEL_COLORS
            }
        else:
            tick_colors = {}
        self._axis_bottom.tick_colors = tick_colors
        self._axis_bottom.setTicks(ticks)

        n = len(fluor_pnn)
        self._plot_widget.setXRange(-0.5, n - 0.5, padding=0.02)

        # Status bar
        cs_txt  = f'{cs:.4f}' if cs is not None else 'N/A'
        n_surv  = cleaned.get('n_surviving_positive', '?')
        emp_idx = cleaned.get('empirical_peak_ch_idx')
        exp_idx = cleaned.get('expected_peak_ch_idx')
        try:
            emp_ch = fluor_pnn[emp_idx].removesuffix('-A') if emp_idx is not None else '?'
            exp_ch = fluor_pnn[exp_idx].removesuffix('-A') if exp_idx is not None else '?'
        except IndexError:
            emp_ch = str(emp_idx)
            exp_ch = str(exp_idx)
        match_flag = '' if emp_ch == exp_ch else '  ⚠ peak mismatch'
        self._status.setText(
            f'Cosine: {cs_txt}  ·  {n_surv} events  ·  '
            f'peak: {emp_ch} (expected: {exp_ch}){match_flag}'
        )