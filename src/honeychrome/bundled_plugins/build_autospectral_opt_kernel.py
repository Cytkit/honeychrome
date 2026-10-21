"""
build_autospectral_opt_kernel.py
----------------------------------
Configure + build the _autospectral_opt_kernel pybind11/Armadillo extension
and the _gls_m2p_kernel pybind11/BLAS extension (CMakeLists.txt), and copy
the shared libraries next to this script, so the wrappers'
(autospectral_opt_kernel_wrapper.py, gls_m2p_kernel_wrapper.py)
same-directory imports can find them.

Prerequisites (macOS, Apple Silicon, local dev):
    brew install armadillo libomp cmake
    python -m pip install -r requirements.txt       # from the repository root; includes pybind11

Usage:
    python build_autospectral_opt_kernel.py

Scope note
----------
Used both as a local dev build helper (macOS, Apple Silicon) and directly by
build.yml's CI steps on macOS and Windows — see build.yml for the exact
invocation and required brew/vcpkg/pip prerequisites on those runners. This
is a plain CMake configure+build, not a wheel build.
"""

import os
import shutil
import subprocess
import sys
import sysconfig

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = os.path.join(HERE, 'build')


KERNELS = ('_autospectral_opt_kernel', '_gls_m2p_kernel')


def _find_built_extension(name):
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX') or '.so'
    for root, _dirs, files in os.walk(BUILD_DIR):
        for f in files:
            if f.startswith(name) and f.endswith(ext_suffix):
                return os.path.join(root, f)
    # Fallback in case EXT_SUFFIX detection differs from the actual build output.
    for root, _dirs, files in os.walk(BUILD_DIR):
        for f in files:
            if f.startswith(name) and f.endswith(('.so', '.pyd', '.dylib')):
                return os.path.join(root, f)
    return None


def main():
    use_openmp = os.environ.get('HONEYCHROME_OPENMP', '1').strip() not in ('0', 'false', 'False', '')

    configure_cmd = [
        'cmake', '-S', HERE, '-B', BUILD_DIR,
        f'-DPython3_EXECUTABLE={sys.executable}',
        f'-DHONEYCHROME_OPENMP={"ON" if use_openmp else "OFF"}',
        '-DCMAKE_BUILD_TYPE=Release',
    ]
    print(f'[build] configuring: {" ".join(configure_cmd)}')
    subprocess.run(configure_cmd, check=True)

    build_cmd = ['cmake', '--build', BUILD_DIR, '--config', 'Release', '-j']
    print(f'[build] building: {" ".join(build_cmd)}')
    subprocess.run(build_cmd, check=True)

    for name in KERNELS:
        built = _find_built_extension(name)
        if built is None:
            raise RuntimeError(
                f'build_autospectral_opt_kernel: could not locate the built '
                f'{name} extension under {BUILD_DIR} — check the CMake output above.'
            )

        dest = os.path.join(HERE, os.path.basename(built))
        if os.path.exists(dest):
            os.remove(dest)  # new inode; macOS kills on in-place overwrite of a signed binary
        shutil.copy2(built, dest)
        print(f'[build] copied {built} -> {dest}')
        print(f'[build] {name} extension built successfully.')


if __name__ == '__main__':
    main()
