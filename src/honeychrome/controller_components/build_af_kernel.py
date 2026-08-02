"""
build_af_kernel.py
------------------
Compile af_kernel.c into a cffi extension module (_af_kernel).

Run once at build time (or during development setup):
    python build_af_kernel.py

The output is _af_kernel.<platform>.so (Linux) or
_af_kernel.<platform>.dylib (macOS), placed alongside this script.

OpenMP
------
Enabled by default on Linux (gcc -fopenmp).
On macOS, requires libomp:
    brew install libomp
then set the environment variable:
    HONEYCHROME_OPENMP=1 python build_af_kernel.py

Without OpenMP the kernel still compiles and runs correctly,
single-threaded.  The Python wrapper (_af_kernel_wrapper.py)
works identically either way.
"""

import os
import sys
import cffi

HERE = os.path.dirname(os.path.abspath(__file__))
ffi = cffi.FFI()

ffi.cdef("""
void joint_cov_l1_argmin(
    const double *init_fluor,
    const double *K,
    const double *v_library,
    const double *w,
    const double *base_e_fluor,
    const double *e_resid,
    const double *base_e_resid,
    int32_t      *best_j,
    int B,
    int n_fluors,
    int n_af
);
""")

with open(os.path.join(HERE, 'af_kernel.c'), 'r') as f:
    source = f.read()

use_openmp = os.environ.get('HONEYCHROME_OPENMP', '').strip() not in ('', '0', 'false', 'False')

# Default to OpenMP on for both platforms unless explicitly disabled
# (HONEYCHROME_OPENMP=0). The libomp check below already falls back to
# single-threaded with a warning if libomp isn't installed, so there's no
# need to gate the attempt behind an explicit opt-in.
if 'HONEYCHROME_OPENMP' not in os.environ:
    use_openmp = True

if sys.platform == 'win32':
    libraries = []
    extra_compile_args = ['/O2', '/fp:fast']
    extra_link_args = []
    if use_openmp:
        extra_compile_args.append('/openmp')
else:
    libraries = ['m']
    extra_compile_args = ['-O3', '-ffast-math']
    extra_link_args = []
    if use_openmp:
        if sys.platform == 'darwin':
            brew_prefix = os.popen('brew --prefix libomp 2>/dev/null').read().strip()
            if brew_prefix:
                extra_compile_args += ['-Xpreprocessor', '-fopenmp', f'-I{brew_prefix}/include']
                extra_link_args += [f'-L{brew_prefix}/lib', '-lomp']
                print(f'[build] macOS OpenMP via libomp at {brew_prefix}')
            else:
                print('[build] WARNING: libomp not found. Building single-threaded.')
                use_openmp = False
        else:  # Linux
            extra_compile_args.append('-fopenmp')
            extra_link_args.append('-fopenmp')

if not use_openmp:
    print('[build] Building single-threaded (no OpenMP).')

ffi.set_source(
    '_af_kernel',
    source,
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    libraries=libraries,
)

if __name__ == '__main__':
    ffi.compile(tmpdir=HERE, verbose=True)
    print('[build] _af_kernel extension built successfully.')
