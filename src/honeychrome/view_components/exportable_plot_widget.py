from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QLabel, QVBoxLayout, QMessageBox
import honeychrome.settings as settings
from PySide6.QtCore import Qt


def apply_plot_theme():
    """Clean appearance for publication-like aesthetics: the seaborn
    whitegrid theme with DejaVu Sans first in the font list."""
    import seaborn as sns
    import matplotlib as mpl
    sns.set_theme(style="whitegrid", font_scale=1.2)
    mpl.rcParams['font.sans-serif'] = ['DejaVu Sans'] + [
        f for f in mpl.rcParams['font.sans-serif'] if f != 'DejaVu Sans'
    ]


class ExportablePlotWidget(QWidget):
    def __init__(self, figure, title='figure', experiment_dir='', parent=None):
        super().__init__(parent)

        self.experiment_dir = experiment_dir
        self.export_filename = title
        self.plot_label = QLabel(title)
        self.plot_label.setTextFormat(Qt.RichText)
        self.plot_label.setWordWrap(True)
        self.plot_label.setMinimumWidth(200)
        self.plot_label.setMaximumWidth(500)

        # Matplotlib imports for embedding
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

        # Create matplotlib figure + canvas
        self.figure = figure
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setFixedSize(800, 600)

        self.delete_button = QPushButton('Delete')
        self.delete_button.clicked.connect(self.delete_plot)
        self.export_graphic_button = QPushButton('Export Graphic')
        self.export_graphic_button.clicked.connect(self.export_graphic)
        buttons_layout = QVBoxLayout()
        buttons_layout.addWidget(self.plot_label)
        buttons_layout.addWidget(self.delete_button)
        buttons_layout.addWidget(self.export_graphic_button)
        buttons_layout.addStretch()

        # Layout to hold the canvas
        layout = QHBoxLayout(self)
        layout.addWidget(self.canvas)
        layout.addLayout(buttons_layout)
        layout.addStretch()

        apply_plot_theme()

        self.canvas.draw()

    def delete_plot(self):
        reply = QMessageBox.question(self, "Delete Plot", f"This will delete the current statistical comparison. Are you sure you wish to continue?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.deleteLater()

    def export_graphic(self):
        self.figure.savefig(f"{self.experiment_dir /self.export_filename}.{settings.graphics_export_format_retrieved}", bbox_inches="tight")
        QMessageBox.information(self, "Exported", f"Exported {settings.graphics_export_format_retrieved} graphic file: \n{self.export_filename}.{settings.graphics_export_format_retrieved}\nto {self.experiment_dir}")

