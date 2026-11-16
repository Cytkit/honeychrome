import os
from pathlib import Path

from PySide6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel, QListView, QMenu, QDialog, QDialogButtonBox, QVBoxLayout as QVLayout, QStyledItemDelegate, QCheckBox, QFrame)
from PySide6.QtCore import Qt, QSettings, QStringListModel, QPoint, QObject, QModelIndex
from PySide6.QtGui import QPixmap, QPainter, QIcon

from honeychrome.controller_components.functions import q_settings, add_recent_file
from honeychrome.settings import experiments_folder, file_extension, send_debug_data
from honeychrome.settings import q_settings as q_settings_app_config
from honeychrome.view_components.busy_cursor import with_busy_cursor
from honeychrome.__init__ import __version__

base_directory = str(Path.home() / experiments_folder)

# Get the assets directory
ASSETS_DIR = Path(__file__).parent / "assets"
n_recent = 10

import logging
logger = logging.getLogger(__name__)

import requests

def check_for_updates():
    try:
        response = requests.get("https://api.github.com/repos/salmansamson/honeychrome/releases/latest")
        latest_tag = response.json()["tag_name"]

        if latest_tag != f"v{__version__}":
            return f'New version {latest_tag} available! (You have v{__version__}.) <a href="https://honeychrome.cytkit.com">Update!</a>'
        else:
            return None
    except Exception:
        return None  # Stay silent if offline

def toggle_send_debug_data_setting(checked):
    q_settings_app_config.setValue("send_debug_data", checked)

class HoverDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.hover_index = QModelIndex()

    def setHoverIndex(self, index):
        self.hover_index = index

    def paint(self, painter: QPainter, option, index):
        painter.save()

        text = index.data(Qt.DisplayRole)
        rect = option.rect

        font = painter.font()

        # underline when hovered
        if index == self.hover_index:
            font.setUnderline(True)
            painter.setFont(font)
            # painter.setPen(Qt.blue)  # optional hyperlink color

        painter.drawText(rect, Qt.AlignVCenter | Qt.TextSingleLine, text)

        painter.restore()


class HoverListView(QListView):
    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.delegate = HoverDelegate(self)
        self.setItemDelegate(self.delegate)

    def mouseMoveEvent(self, event):
        index = self.indexAt(event.position().toPoint())
        self.delegate.setHoverIndex(index)

        # 👇 set pointer cursor when hovering an item
        if index.isValid():
            self.viewport().setCursor(Qt.PointingHandCursor)
        else:
            self.viewport().setCursor(Qt.ArrowCursor)

        self.viewport().update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self.delegate.setHoverIndex(QModelIndex())
        self.viewport().setCursor(Qt.ArrowCursor)  # reset cursor
        self.viewport().update()
        super().leaveEvent(event)



class SplashScreen(QDialog):
    def __init__(self, view):
        super().__init__()
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMinimizeButtonHint | Qt.WindowType.WindowMaximizeButtonHint | Qt.WindowType.WindowCloseButtonHint)
        self.view = view
        self.setWindowTitle("Honeychrome by Cytkit")

        layout = QVBoxLayout()

        # Create QLabel to display image
        image_label = QLabel()
        logo_path = ASSETS_DIR / 'honeychrome_by_cytkit_logo_bee.png'
        pixmap = QPixmap(logo_path)
        # Scale the image
        scaled_pixmap = pixmap.scaled(
            450,
            450,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        image_label.setPixmap(scaled_pixmap)
        layout.addWidget(image_label)

        # update link and version check
        update = check_for_updates()
        if update:
            frame = QFrame(self)
            frame_layout = QVBoxLayout()
            frame.setLayout(frame_layout)
            frame_layout.setContentsMargins(10, 10, 10, 10)
            frame.setFrameShape(QFrame.StyledPanel)
            frame.setLineWidth(1)

            update_label = QLabel(update)
            update_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
            update_label.setOpenExternalLinks(True)

            frame_layout.addWidget(update_label)
            layout.addWidget(frame)

        layout.addWidget(QLabel("Select an option to start:"))

        self.btn_new = QPushButton("New Experiment")
        self.btn_new.clicked.connect(self.view.new_experiment)
        layout.addWidget(self.btn_new)

        self.btn_open = QPushButton("Open Experiment")
        self.btn_open.clicked.connect(self.view.open_experiment)
        layout.addWidget(self.btn_open)

        self.btn_template = QPushButton("New Experiment from Template")
        self.btn_template.setToolTip('Copy settings and spectral model from an existing experiment file')
        self.btn_template.clicked.connect(self.view.new_from_template)
        layout.addWidget(self.btn_template)

        self.btn_config = QPushButton("App Configuration")
        self.btn_config.setToolTip('Application settings, including letting AI agents control Honeychrome')
        self.btn_config.clicked.connect(self.open_app_config)
        layout.addWidget(self.btn_config)

        # recent files
        self.recent_model = QStringListModel()
        full_recent_files = q_settings.value("recent_files", [])
        if isinstance(full_recent_files, str):
            full_recent_files = [full_recent_files]
        full_recent_files = [file for file in full_recent_files if Path(file).exists()]

        if full_recent_files:
            layout.addWidget(QLabel("Recent Experiments:"))
            self.display_path_list = full_recent_files[:n_recent]
            display_list = [os.path.basename(path) for path in self.display_path_list]
            self.recent_model.setStringList(display_list)

            self.recent_view = HoverListView()
            self.recent_view.setModel(self.recent_model)
            self.recent_view.clicked.connect(self.handle_recent_clicked)
            self.recent_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            self.recent_view.customContextMenuRequested.connect(self.show_context_menu)

            layout.addWidget(self.recent_view)

        # display checkbox for send_debug_data if settings not yet saved (i.e. first run)
        if not q_settings_app_config.contains('send_debug_data'):
            toggle_send_debug_data_setting(send_debug_data)

            frame = QFrame(self)
            frame_layout = QVBoxLayout()
            frame.setLayout(frame_layout)
            frame_layout.setContentsMargins(10, 10, 10, 10)
            frame.setFrameShape(QFrame.StyledPanel)
            frame.setLineWidth(1)

            frame_layout.addWidget(QLabel('Honeychrome does not collect personal identifiable information'))
            self.send_debug_data = QCheckBox("Send debug data to improve Honeychrome")
            self.send_debug_data_label = QLabel('<a href="https://honeychrome.cytkit.com/docs/send_debug_data">What data is collected?</a>')

            self.send_debug_data_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
            self.send_debug_data_label.setOpenExternalLinks(True)
            self.send_debug_data.setChecked(q_settings_app_config.value("send_debug_data", send_debug_data, type=bool))

            frame_layout.addWidget(self.send_debug_data)
            frame_layout.addWidget(self.send_debug_data_label)
            layout.addWidget(frame)
            self.send_debug_data.toggled.connect(toggle_send_debug_data_setting)
        else:
            self.send_debug_data = None


        self.setLayout(layout)

    def open_app_config(self):
        from honeychrome.view_components.configuration_dialogs import AppConfigDialog
        dialog = AppConfigDialog(self, getattr(self.view, 'bus', None), getattr(self.view, 'controller', None))
        dialog.exec()

    def handle_recent_clicked(self, index):
        self.btn_new.setEnabled(False)
        self.btn_open.setEnabled(False)
        self.btn_template.setEnabled(False)
        self.recent_view.setEnabled(False)
        #path = self.recent_model.data(index, Qt.ItemDataRole.DisplayRole)
        path = self.display_path_list[index.row()]
        self.view.load_main_window_with_experiment_and_template(path)

    def show_context_menu(self, pos: QPoint):
        index = self.recent_view.indexAt(pos)
        if not index.isValid():
            return
        # path = self.recent_model.data(index, Qt.ItemDataRole.DisplayRole)
        path = self.display_path_list[index.row()]

        menu = QMenu(self)
        action_new_template = menu.addAction("New From Template")

        action = menu.exec(self.recent_view.mapToGlobal(pos))
        if action == action_new_template:
            self.view.new_from_this_template(path)


if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication

    class ExperimentView(QObject):
        def __init__(self):
            super().__init__()
            self.main_window = None
            self.splash = SplashScreen(self)
            self.splash.show()

        def load_main_window_with_experiment_and_template(self, experiment_file, new=False, template_path=None):
            self.splash.close()
            logger.info([experiment_file, new, template_path])
            # self.main_window = MainWindow(experiment_file, experiment_model, new=new, template=template)
            # self.main_window.show()

        def new_experiment(self):
            pass

        def open_experiment(self):
            pass

        def new_from_template(self):
            pass

    app =  QApplication(sys.argv)

    view = ExperimentView()

    exit_code = app.exec()
    sys.exit(exit_code)

