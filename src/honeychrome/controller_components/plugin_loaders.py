import importlib.util
import logging
from pathlib import Path
import sys

from PySide6.QtCore import QSettings
from honeychrome.settings import experiments_folder

PLUGIN_DIR = Path.home() / experiments_folder / "plugins"
plugins_path = Path(PLUGIN_DIR)

settings = QSettings("honeychrome", "app_configuration")
logger = logging.getLogger(__name__)

meipass = getattr(sys, '_MEIPASS', None)

# Bundled/approved plugins — shipped inside the app itself, loaded directly
# from here in both dev and frozen builds. Never copied into the
# user-writable plugins_path above.
if meipass:
    BUNDLED_PLUGIN_DIR = Path(meipass) / "honeychrome" / "bundled_plugins"
else:
    BUNDLED_PLUGIN_DIR = Path(__file__).resolve().parent.parent / "bundled_plugins"
bundled_plugins_path = Path(BUNDLED_PLUGIN_DIR)

def _load_module(module_name, file_path):
    """Import one plugin file and run its _bootstrap(); None if it fails.

    A failing plugin is logged with its traceback and skipped, so it cannot
    stop the other plugins from loading."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if not (spec and spec.loader):
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        if hasattr(module, '_bootstrap'):
            module._bootstrap()
    except Exception:
        sys.modules.pop(module_name, None)
        logger.exception(f'Plugin {module_name} failed to load from {file_path}')
        return None
    return module


def load_plugin_modules():
    """Import and execute plugin modules only — no Qt widget creation.
    Safe to call from a background thread."""
    loaded_modules = {}

    # The old Gating Model Builder setting enables Automated Gating, which
    # now holds its pages.
    if settings.value("EnableBundledPlugin_gating_model_builder_tab", False, type=bool):
        settings.setValue("EnableBundledPlugin_automated_gating_tab", True)
        settings.remove("EnableBundledPlugin_gating_model_builder_tab")

    # Bundled/approved plugins: available in both the frozen app and source
    # runs, sourced only from bundled_plugins_path — never from the
    # user-writable Experiments/plugins folder.
    for file_path in bundled_plugins_path.glob("*_tab.py"):
        if settings.value(f"EnableBundledPlugin_{file_path.stem}", False, type=bool):
            module_name = f"bundled_plugins.{file_path.stem}"
            module = _load_module(module_name, file_path)
            if module is not None:
                print(f"Successfully loaded bundled plugin module: {module_name}")
                loaded_modules[module_name] = module

    # Arbitrary user-added plugins: the open plugin ecosystem. Deliberately
    # source/dev-only — these are unreviewed files a user can drop into
    # ~/Experiments/plugins/, and we are not executing unreviewed third-party
    # code inside the distributed executable. Unchanged from current behaviour.
    if not meipass:
        for file_path in plugins_path.glob("*_tab.py"):
            if settings.value(f"EnablePlugin_{file_path}", False, type=bool):
                module_name = f"{plugins_path.name}.{file_path.stem}"
                module = _load_module(module_name, file_path)
                if module is not None:
                    print(f"Successfully loaded plugin module: {module_name}")
                    loaded_modules[module_name] = module

    return loaded_modules


def instantiate_plugin_widgets(loaded_modules, bus, controller):
    """Create PluginWidget instances from pre-loaded modules.
    Must be called on the main (Qt) thread."""
    tab_plugins = {}
    for module_name, module in loaded_modules.items():
        # 6. initialise widget to be put in the tab
        widget = module.PluginWidget(bus=bus, controller=controller)
        print(f"Successfully instantiated plugin: {module_name}")
        tab_plugins[module_name] = {'module': module, 'widget': widget}
    return tab_plugins


def load_tabbed_plugins(bus, controller):
    """Original synchronous entry point — kept for compatibility."""
    loaded_modules = load_plugin_modules()
    return instantiate_plugin_widgets(loaded_modules, bus, controller)