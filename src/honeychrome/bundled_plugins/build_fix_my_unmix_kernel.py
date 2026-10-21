"""
build_fix_my_unmix_kernel.py
----------------------------
Compile fix_my_unmix_kernel.cpp into a cffi extension (_fmu_kernel) beside
this script (honeychrome/bundled_plugins/), where
controller_components/fix_my_unmix.py looks for it.

Run once in the Python environment Honeychrome runs in:
    python build_fix_my_unmix_kernel.py

Needs a C++17 compiler (Xcode command-line tools on macOS) and cffi, which
Honeychrome already requires. Without the extension Fix My Unmix uses its
numpy implementation, with the same estimator and results but slower.
Rebuild after updating fix_my_unmix_kernel.cpp: an extension built from an
older version no longer matches and the numpy path is used.

Rebuilding is safe over an earlier build. The new extension is compiled in a
clean scratch folder, earlier ``_fmu_kernel`` binaries (including other Python
versions) are removed, and the new file is moved into place as a new file
rather than written over the old one. On macOS it is also ad-hoc signed,
because overwriting a previously loaded binary in place invalidates its
signature and the OS then kills any process that imports it. Quit Honeychrome
before rebuilding so the running session does not keep the old copy loaded.

Threads
-------
Targets are split across ``std::thread`` workers, so no OpenMP runtime
(libomp) is needed on any platform.
"""

import os
import shutil
import subprocess
import sys

import cffi

HERE = os.path.dirname(os.path.abspath(__file__))

CDEF = """
int fmu_max_threads(void);
void fmu_anchored_batch(const double *xb, const double *Yb, const double *Tb, const double *Lb, int nb,
                        const double *xk, const double *Yk, const double *Tk, int nk,
                        const double *start, int m, double max_coefficient, int max_mask_passes,
                        double mask_tolerance, int min_count, int band, double k, int max_iter, double tol,
                        int n_threads, double *slope_out, int *n_out, double *se_out);
"""


def _flags():
    if sys.platform == 'win32':
        return ['/O2', '/std:c++17', '/EHsc'], []
    link_args = [] if sys.platform == 'darwin' else ['-pthread']
    return ['-O3', '-std=c++17', '-pthread'], link_args


def _remove_previous():
    """Delete earlier _fmu_kernel binaries beside this script."""
    for name in os.listdir(HERE):
        if name.startswith('_fmu_kernel.') and name.endswith(('.so', '.pyd', '.dylib')):
            os.remove(os.path.join(HERE, name))


def _install(built: str) -> str:
    """Place the new binary as a fresh file, replacing any earlier build."""
    target = os.path.join(HERE, os.path.basename(built))
    staged = target + '.new'
    shutil.copy2(built, staged)
    _remove_previous()
    os.replace(staged, target)
    if sys.platform == 'darwin':
        subprocess.run(['codesign', '--force', '--sign', '-', target],
                       check=False, capture_output=True)
    return target


def build(verbose: bool = True) -> str:
    """Compile the extension into this folder; returns its path."""
    compile_args, link_args = _flags()
    ffi = cffi.FFI()
    ffi.cdef(CDEF)
    with open(os.path.join(HERE, 'fix_my_unmix_kernel.cpp'), 'r', encoding='utf-8') as handle:
        source = handle.read()
    ffi.set_source('_fmu_kernel', source, source_extension='.cpp',
                   extra_compile_args=compile_args, extra_link_args=link_args)

    tmp = os.path.join(HERE, '_fmu_kernel_build')
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp)
    try:
        built = ffi.compile(tmpdir=tmp, verbose=verbose)
        target = _install(built)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f'[build] {target}')
    return target


if __name__ == '__main__':
    build()
