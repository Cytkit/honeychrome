# build_optimized.py
import PyInstaller.__main__
import os
import shutil
import subprocess
import sys
import platform

from PyInstaller.utils.hooks import collect_submodules


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

    if platform.system() == 'Darwin':
        link_kernels_to_bundled_libomp(os.path.join(project_root, 'src', 'honeychrome',
                                                    'bundled_plugins'))

    return assets

# --- One OpenMP runtime on macOS -------------------------------------------
# The SOM and AutoSpectral optimisation kernels (bundled_plugins/) and openTSNE
# (built by install_opentsne.py) all link Homebrew's libomp.dylib.  The app
# must hold exactly one copy: two copies at different paths start two OpenMP
# runtimes when both are used in one session, which can abort the app
# ("OMP: Error #15").  openTSNE goes through PyInstaller's analysis, which
# collects its libomp into Contents/Frameworks; the kernels are copied as data
# and never analysed.  So libomp is added once at the top level, and the
# kernels find it through @rpath (Contents/Frameworks in the app, Homebrew in
# a source checkout).  check_single_libomp() verifies the built app.
# scikit-learn ships its own, older libomp in sklearn/.dylibs; it cannot be
# shared (it lacks entry points Apple clang now calls) and is left alone.

KERNEL_PREFIXES = ('_som_kernel', '_autospectral_opt_kernel', '_gls_m2p_kernel')
APP_FRAMEWORKS_RPATH = '@executable_path/../Frameworks'
LIBOMP_RPATH_REF = '@rpath/libomp.dylib'


def homebrew_libomp():
    """Path of Homebrew's libomp.dylib, or None."""
    try:
        prefix = subprocess.check_output(['brew', '--prefix', 'libomp'],
                                         stderr=subprocess.DEVNULL).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    path = os.path.join(prefix, 'lib', 'libomp.dylib')
    return path if os.path.exists(path) else None


def macho_libomp_refs(binary):
    """Load commands of a Mach-O file that name libomp.dylib."""
    out = subprocess.check_output(['otool', '-L', binary]).decode().splitlines()[1:]
    paths = [line.strip().split(' (')[0] for line in out]
    return [p for p in paths if os.path.basename(p) == 'libomp.dylib']


def macho_rpaths(binary):
    """LC_RPATH entries of a Mach-O file."""
    out = subprocess.check_output(['otool', '-l', binary]).decode().splitlines()
    rpaths = []
    for i, line in enumerate(out):
        if line.strip() == 'cmd LC_RPATH':
            for follow in out[i + 1:i + 4]:
                follow = follow.strip()
                if follow.startswith('path '):
                    rpaths.append(follow[len('path '):].split(' (offset')[0])
                    break
    return rpaths


def link_kernels_to_bundled_libomp(kernel_dir):
    """Point the kernels' libomp load command at @rpath/libomp.dylib, with
    rpaths to the app's Frameworks folder first and Homebrew second."""
    libomp = homebrew_libomp()
    if libomp is None:
        print('WARNING: Homebrew libomp not found; the SOM and optimisation kernels '
              'will not find OpenMP in the built app.')
        return
    stale = os.path.join(kernel_dir, 'libomp.dylib')
    if os.path.exists(stale):
        # Left by earlier versions of this script; a second copy in the app
        os.remove(stale)
        print(f'Removed {stale}')
    for fname in sorted(os.listdir(kernel_dir)):
        if not (fname.startswith(KERNEL_PREFIXES) and fname.endswith(('.so', '.dylib'))):
            continue
        kernel = os.path.join(kernel_dir, fname)
        refs = macho_libomp_refs(kernel)
        if not refs:
            continue
        changed = False
        for ref in refs:
            if ref != LIBOMP_RPATH_REF:
                subprocess.run(['install_name_tool', '-change', ref, LIBOMP_RPATH_REF, kernel],
                               check=True)
                changed = True
        rpaths = macho_rpaths(kernel)
        for rpath in (APP_FRAMEWORKS_RPATH, os.path.dirname(libomp)):
            if rpath not in rpaths:
                subprocess.run(['install_name_tool', '-add_rpath', rpath, kernel], check=True)
                changed = True
        if changed:
            subprocess.run(['codesign', '--force', '--sign', '-', kernel], check=True)
        print(f'{fname}: libomp via {LIBOMP_RPATH_REF}, rpaths {macho_rpaths(kernel)}')


def resolve_macho_ref(ref, binary, executable_dir):
    """File a load command resolves to inside the app, or None."""
    loader_dir = os.path.dirname(binary)

    def expand(path):
        return os.path.normpath(path.replace('@loader_path', loader_dir)
                                .replace('@executable_path', executable_dir))
    if ref.startswith('@rpath/'):
        for rpath in macho_rpaths(binary):
            candidate = expand(os.path.join(rpath, ref[len('@rpath/'):]))
            if os.path.exists(candidate):
                return candidate
        return None
    candidate = expand(ref)
    return candidate if os.path.exists(candidate) else None


def check_single_libomp(app):
    """Fail the build unless the app holds one libomp (besides scikit-learn's
    own) and every kernel and openTSNE extension that links libomp loads it."""
    contents = os.path.join(app, 'Contents')
    executable_dir = os.path.join(contents, 'MacOS')
    copies = set()
    binaries = []
    for root, _dirs, files in os.walk(contents):
        for fname in files:
            path = os.path.join(root, fname)
            if os.path.islink(path) or 'sklearn' in path.split(os.sep):
                continue
            if fname == 'libomp.dylib':
                copies.add(os.path.realpath(path))
            elif fname.endswith(('.so', '.dylib')):
                binaries.append(path)
    errors = []
    if len(copies) != 1:
        errors.append(f'{len(copies)} copies of libomp.dylib: {sorted(copies)}')
    for binary in binaries:
        refs = macho_libomp_refs(binary)
        if not refs:
            continue
        ours = os.path.basename(binary).startswith(KERNEL_PREFIXES) or 'openTSNE' in binary
        for ref in refs:
            target = resolve_macho_ref(ref, binary, executable_dir)
            if target is None:
                message = f'{os.path.relpath(binary, app)}: {ref} does not resolve in the app'
                if ours:
                    errors.append(message)
                else:
                    print(f'WARNING: {message}')
            elif os.path.realpath(target) not in copies:
                errors.append(f'{os.path.relpath(binary, app)}: {ref} -> {target}, '
                              f'not the bundled libomp')
            else:
                print(f'{os.path.relpath(binary, app)}: libomp -> '
                      f'{os.path.relpath(os.path.realpath(target), app)}')
    if errors:
        print('libomp check FAILED:')
        for error in errors:
            print('   ', error)
        sys.exit(1)
    print(f'libomp check passed: one copy at {os.path.relpath(next(iter(copies)), app)}')

def collect_all_submodules(package_root, package_name):
    """Walk the package directory and return every importable module name."""
    modules = []
    for dirpath, _dirnames, filenames in os.walk(package_root):
        # Skip __pycache__ and other non-package dirs
        if '__pycache__' in dirpath:
            continue
        for fname in filenames:
            if not fname.endswith('.py'):
                continue
            if fname == '__init__.py':
                # The directory itself is a package
                rel = os.path.relpath(dirpath, package_root)
                if rel == '.':
                    modules.append(package_name)
                else:
                    modules.append(f"{package_name}.{rel.replace(os.sep, '.')}")
            else:
                rel = os.path.relpath(dirpath, package_root)
                stem = fname[:-3]
                if rel == '.':
                    modules.append(f"{package_name}.{stem}")
                else:
                    modules.append(f"{package_name}.{rel.replace(os.sep, '.')}.{stem}")
    return modules

def main():
    clean()
    icon_path = os.path.join('src', 'honeychrome', 'view_components', 'assets', 'cytkit_web_logo.ico')
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

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
    honeychrome_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'honeychrome')
    for mod in collect_all_submodules(honeychrome_root, 'honeychrome'):
        args.append(f'--hidden-import={mod}')

    hidden_imports = collect_all_submodules(honeychrome_root, 'honeychrome')
    print("Collected:", hidden_imports)
    for mod in hidden_imports:
        args.append(f'--hidden-import={mod}')
    args.append('--runtime-hook=hooks/runtime-patch-syspath.py')

    # Add project files
    sep = os.pathsep
    for source, dest in get_project_files():
        args.append(f'--add-data={source}{sep}{dest}')

    # One libomp for the kernels and openTSNE, at the top level of the app
    # (Contents/Frameworks); see link_kernels_to_bundled_libomp()
    libomp = homebrew_libomp() if platform.system() == 'Darwin' else None
    if libomp is not None:
        args.append(f'--add-binary={libomp}{sep}.')

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

    app = os.path.join('dist', 'honeychrome.app')
    if libomp is not None and os.path.isdir(app):
        check_single_libomp(app)

    # Show size
    exe_name = 'honeychrome.exe' if platform.system() == 'Windows' else 'honeychrome'
    exe_path = os.path.join('dist', 'honeychrome', exe_name)
    if os.path.exists(exe_path):
        size = os.path.getsize(exe_path) / (1024 * 1024)  # MB
        print(f"Build complete! Executable size: {size:.2f} MB")


if __name__ == "__main__":
    main()