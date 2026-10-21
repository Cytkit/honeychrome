import sys
from PySide6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QSlider, QSpinBox, QLabel, QPushButton, QFrame, QGroupBox)
from PySide6.QtCore import Qt, Signal
from honeychrome.settings import fluorescence_channels, default_gains_immuno, default_gains_xfp, gain_range, dac_bias_retrieved

class GainsWidget(QWidget):
    setAllChannels = Signal(int)

    def __init__(self, bus, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bus = bus

        self.channels = []
        self.presets = {'Immuno': default_gains_immuno, 'xFP': default_gains_xfp}

        layout = QVBoxLayout(self)

        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel)  # Add border

        main_layout = QVBoxLayout(frame)

        # Title and Preset buttons
        presets_layout = QHBoxLayout()

        title = QLabel("Gains")
        presets_layout.addWidget(title)

        immuno_btn = QPushButton("Default AB")
        immuno_btn.clicked.connect(lambda: self.apply_preset("Immuno"))

        xfp_btn = QPushButton("Default xFP")
        xfp_btn.clicked.connect(lambda: self.apply_preset("xFP"))
        presets_layout.addStretch()
        presets_layout.addWidget(immuno_btn)
        presets_layout.addWidget(xfp_btn)

        main_layout.addLayout(presets_layout)

        all_channel_widget = self.create_channel('All Channels')
        main_layout.addWidget(all_channel_widget)

        show_individual_btn = QPushButton("Individual Channels ▼")
        main_layout.addWidget(show_individual_btn)
        # Create bands stacked vertically
        self.indiv_widget = QGroupBox()
        indiv_layout = QVBoxLayout(self.indiv_widget)
        for c in fluorescence_channels:
            channel_widget = self.create_channel(c)
            self.channels.append(channel_widget)
            indiv_layout.addWidget(channel_widget)

        main_layout.addWidget(self.indiv_widget)
        self.indiv_widget.setVisible(False)
        show_individual_btn.clicked.connect(lambda: self.indiv_widget.setVisible(False if self.indiv_widget.isVisible() else True))

        self.setLayout(main_layout)

        layout.addWidget(frame)

    def create_channel(self, ch_name):
        """Create a single band with horizontal slider and spinbox"""
        channel_widget = QWidget()
        layout = QHBoxLayout()

        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Channel label
        ch_label = QLabel(ch_name)
        ch_label.setFixedWidth(100)

        # Horizontal slider
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, gain_range)
        slider.setValue(dac_bias_retrieved[ch_name])

        # Spin box
        spinbox = QSpinBox()
        spinbox.setRange(0, gain_range)
        spinbox.setValue(dac_bias_retrieved[ch_name])
        spinbox.valueChanged.connect(slider.setValue)

        slider.valueChanged.connect(spinbox.setValue)
        if ch_name != 'All Channels':
            slider.valueChanged.connect(lambda value: self.bus.gainChanged.emit(ch_name, value))
            self.setAllChannels.connect(spinbox.setValue)
            self.setAllChannels.connect(slider.setValue)
        else:
            slider.valueChanged.connect(self.setAllChannels)

        # Add widgets to layout
        layout.addWidget(ch_label)
        layout.addWidget(slider)
        layout.addWidget(spinbox)

        channel_widget.setLayout(layout)
        return channel_widget

    def apply_preset(self, preset_name):
        if preset_name in self.presets.keys():
            values = self.presets[preset_name]
            for i, channel_widget in enumerate(self.channels):
                slider = channel_widget.findChild(QSlider)
                slider.setValue(values[fluorescence_channels[i]])
            print(f"Applied preset: {preset_name}")

    def apply_master(self):
        value = self.master_spin.value()
        for i, channel_widget in enumerate(self.channels):
            slider = channel_widget.findChild(QSlider)
            slider.setValue(value)
        print(f"Applied master value: {value}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GainsWidget()
    window.show()
    sys.exit(app.exec())