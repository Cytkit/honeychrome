# build_optimized.py
import PyInstaller.__main__
import os
import shutil
import subprocess
import sys
import platform

def clean():
    """Clean build directories"""
    for folder in ['dist', 'build']:
        if os.path.exists(folder):
            print(f"Cleaning {folder}...")
            shutil.rmtree(folder)


def get_project_files():
    """Get only essential project files"""
    project_root = os.path.dirname(os.path.abspath(__file__))
    assets = []

    fcs_path = os.path.join(project_root, 'src', 'honeychrome', 'instrument_driver_components', 'data')
    fcs_path_destination = os.path.join('honeychrome', 'instrument_driver_components', 'data')
    assets.append((fcs_path, fcs_path_destination))

    assets_path = os.path.join('src', 'honeychrome', 'view_components', 'assets')
    assets_path_destination = os.path.join('honeychrome', 'view_components', 'assets')
    assets.append((assets_path, assets_path_destination))

    bundled_plugins_path = os.path.join('src', 'honeychrome', 'bundled_plugins')
    bundled_plugins_dest = os.path.join('honeychrome', 'bundled_plugins')
    assets.append((bundled_plugins_path, bundled_plugins_dest))

    label_data_path = os.path.join(project_root, 'src', 'honeychrome', 'data')
    label_data_destination = os.path.join('honeychrome', 'data')
    assets.append((label_data_path, label_data_destination))

    # C kernel extension (compiled before this script runs)
    kernel_dir = os.path.join(
        project_root, 'src', 'honeychrome', 'controller_components'
    )
    kernel_dest = os.path.join('honeychrome', 'controller_components')
    for fname in os.listdir(kernel_dir):
        if fname.startswith('_af_kernel') and fname.split('.')[-1] in ('so', 'pyd', 'dylib'):
            assets.append((os.path.join(kernel_dir, fname), kernel_dest))

    # macOS: bundle libomp.dylib explicitly — Homebrew path is non-standard
    # and PyInstaller will not find it via normal shared-library scanning.
    # On Windows/Linux the OpenMP runtime (vcomp.dll / libgomp.so) is found
    # automatically and needs no explicit entry here.
    if platform.system() == 'Darwin':
        try:
            libomp_prefix = subprocess.check_output(
                ['brew', '--prefix', 'libomp'], stderr=subprocess.DEVNULL
            ).decode().strip()
            libomp_path = os.path.join(libomp_prefix, 'lib', 'libomp.dylib')
            if os.path.exists(libomp_path):
                # Place alongside the kernel extension so @loader_path resolves it
                assets.append((libomp_path, kernel_dest))
        except (subprocess.CalledProcessError, FileNotFoundError):
            print('WARNING: libomp not found via brew — OpenMP will not be '
                  'available in the built app; falling back to NumPy path.')

    # AutoSpectral Optimization kernel (bundled_plugins/) also links OpenMP on
    # macOS, but unlike _af_kernel it is never hidden-imported (loaded
    # dynamically by plugin_loaders.py), so PyInstaller's Analysis phase never
    # sees it as a binary dependency and never applies its automatic
    # @loader_path rewrite. Copying libomp.dylib alongside it is not sufficient
    # on its own — the compiled extension's own load command still points at
    # the absolute Homebrew path unless rewritten explicitly.
    if platform.system() == 'Darwin':
        try:
            libomp_prefix = subprocess.check_output(
                ['brew', '--prefix', 'libomp'], stderr=subprocess.DEVNULL
            ).decode().strip()
            libomp_src = os.path.join(libomp_prefix, 'lib', 'libomp.dylib')
            opt_kernel_dir = os.path.join(
                project_root, 'src', 'honeychrome', 'bundled_plugins'
            )
            for fname in os.listdir(opt_kernel_dir):
                if fname.startswith('_autospectral_opt_kernel') and fname.endswith('.dylib'):
                    opt_kernel_path = os.path.join(opt_kernel_dir, fname)
                    if os.path.exists(libomp_src):
                        libomp_dest = os.path.join(opt_kernel_dir, 'libomp.dylib')
                        shutil.copy2(libomp_src, libomp_dest)
                        subprocess.run([
                            'install_name_tool', '-change',
                            libomp_src, '@loader_path/libomp.dylib',
                            opt_kernel_path,
                        ], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print('WARNING: could not rewrite libomp dependency for the '
                  'AutoSpectral Optimization kernel — it may fail to load '
                  'OpenMP on machines without Homebrew libomp installed.')

    # SOM kernel (bundled_plugins/) -- same situation as
    # _autospectral_opt_kernel just above: lives in the wholesale-copied
    # bundled_plugins/ folder (so the file itself needs no extra collection
    # step) but is loaded dynamically by som_kernel_wrapper.py rather than
    # hidden-imported, so it also never goes through PyInstaller's
    # automatic @loader_path rewrite and needs the same manual fix.
    if platform.system() == 'Darwin':
        try:
            libomp_prefix = subprocess.check_output(
                ['brew', '--prefix', 'libomp'], stderr=subprocess.DEVNULL
            ).decode().strip()
            libomp_src = os.path.join(libomp_prefix, 'lib', 'libomp.dylib')
            som_kernel_dir = os.path.join(
                project_root, 'src', 'honeychrome', 'bundled_plugins'
            )
            for fname in os.listdir(som_kernel_dir):
                if fname.startswith('_som_kernel') and fname.endswith('.dylib'):
                    som_kernel_path = os.path.join(som_kernel_dir, fname)
                    if os.path.exists(libomp_src):
                        libomp_dest = os.path.join(som_kernel_dir, 'libomp.dylib')
                        shutil.copy2(libomp_src, libomp_dest)
                        subprocess.run([
                            'install_name_tool', '-change',
                            libomp_src, '@loader_path/libomp.dylib',
                            som_kernel_path,
                        ], check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print('WARNING: could not rewrite libomp dependency for the '
                  'SOM kernel — it may fail to load OpenMP on machines '
                  'without Homebrew libomp installed.')

    return assets

def main():
    clean()
    icon_path = os.path.join('src', 'honeychrome', 'view_components', 'assets', 'cytkit_web_logo.ico')

    args = ['src/honeychrome/main.py',
            '--name=honeychrome',
            '--noconsole',
            '--clean',
            '--noconfirm',
            '--onedir',
            '--copy-metadata=pandas',
            '--copy-metadata=matplotlib',
            f'--icon={icon_path}'
            ]

    if platform.system() == "Darwin":
        args.append('--windowed')

    # Make the honeychrome package importable by external plugin scripts
    args.append('--hidden-import=honeychrome')
    args.append('--hidden-import=honeychrome.settings')
    args.append('--hidden-import=honeychrome.plugin_loaders')
    args.append('--hidden-import=honeychrome.controller_components.af_kernel_wrapper')
    args.append('--runtime-hook=hooks/runtime-patch-syspath.py')

    # Add project files
    sep = os.pathsep
    for source, dest in get_project_files():
        args.append(f'--add-data={source}{sep}{dest}')

    # Add hooks directory
    args.append('--additional-hooks-dir=hooks')
    args.append('--runtime-hook=hooks/runtime-optional-deps.py')

    # Essential hidden imports
    essential_hidden = ['numpy._core._exceptions',
                        'numpy._core._multiarray_umath',
                        'numpy.fft._pocketfft_internal',
                        'pytz',
                        'scipy.special.cython_special',
                        'pyqtgraph.opengl',
                        ]

    for imp in essential_hidden:
        args.append(f'--hidden-import={imp}')

    # OpenGL uses ctypes-based dynamic platform loading (OpenGL.platform.*)
    # that PyInstaller's static analysis can't see — collect everything since
    # plot_3d_tab.py is only ever loaded dynamically at runtime, never
    # statically imported from main.py.
    args.append('--collect-all=OpenGL')

    # Exclude heavy packages
    # heavy_packages = ['IPython', 'jedi', 'parso', 'bokeh', 'sklearn', 'matplotlib', 'pyqtgraph', 'scipy.sparse', 'scipy.optimize', 'scipy.special', 'scipy.linalg', 'scipy.stats', 'lxml', 'jinja2', 'docx', 'pytest', 'pandas.plotting', 'tkinter', '_tkinter', 'PIL', 'pyarrow', 'urllib3', 'certifi', ]
    heavy_packages = ['IPython', 'jedi', 'parso', 'pytest', 'tkinter', '_tkinter', 'bokeh', 'bokeh.plotting', 'bokeh.models', 'bokeh.layouts', 'narwhals']

    for pkg in heavy_packages:
        args.append(f'--exclude-module={pkg}')

    PyInstaller.__main__.run(args)

    # Show size
    exe_path = os.path.join('dist', 'honeychrome')
    if os.path.exists(exe_path):
        size = os.path.getsize(exe_path) / (1024 * 1024)  # MB
        print(f"Build complete! Executable size: {size:.2f} MB")


if __name__ == "__main__":
    main()