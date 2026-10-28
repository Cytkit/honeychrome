"""
Diagnose why openTSNE's OpenMP build does not speed up on the CI runner.

Run after install_opentsne.py has built and installed openTSNE (its own
self-test may fail; the build is installed before the test runs).

Prints:
  * the runner's CPU layout and the Homebrew libomp version;
  * which libomp images are loaded after `import openTSNE`;
  * for 5,000 and 20,000 events and n_jobs 1, 2 and all cores: wall time
    and CPU time / wall time, split into the affinity step (Annoy kNN and
    perplexity calibration) and the Barnes-Hut optimisation;
  * the all-core runs again with KMP_BLOCKTIME=0 (workers sleep at once
    instead of spinning between parallel regions).

Each configuration runs in its own process, so OpenMP environment settings
take effect and runs do not share a warmed-up thread pool.

Usage:
    python diag_opentsne_threads.py
"""
import json
import os
import platform
import statistics
import subprocess
import sys

REPEATS = 2
SIZES = (5000, 20000)

ONE_RUN = r'''
import json, sys, time
import numpy as np
import openTSNE
from openTSNE import affinity, initialization, TSNEEmbedding

n, n_jobs = int(sys.argv[1]), int(sys.argv[2])
rng = np.random.default_rng(0)
centres = rng.normal(scale=4.0, size=(8, 7))
x = (centres[rng.integers(0, 8, n)] + rng.normal(size=(n, 7))).astype(np.float32)

def timed(step):
    w, c = time.perf_counter(), time.process_time()
    out = step()
    return out, time.perf_counter() - w, time.process_time() - c

# Warm-up on a small subset: one-off costs (first load of the extensions,
# thread-pool start-up) are kept out of the timings
warm = affinity.PerplexityBasedNN(x[:1000], perplexity=30, n_jobs=n_jobs, random_state=42)
TSNEEmbedding(initialization.pca(x[:1000], random_state=42), warm, n_jobs=n_jobs,
              negative_gradient_method='bh', random_state=42).optimize(n_iter=5)

aff, aff_wall, aff_cpu = timed(lambda: affinity.PerplexityBasedNN(
    x, perplexity=30, n_jobs=n_jobs, random_state=42))
init = initialization.pca(x, random_state=42)

def optimise():
    emb = TSNEEmbedding(init, aff, n_jobs=n_jobs, negative_gradient_method='bh',
                        random_state=42)
    emb = emb.optimize(n_iter=50, exaggeration=12)
    return emb.optimize(n_iter=100)

_, opt_wall, opt_cpu = timed(optimise)
print(json.dumps(dict(aff_wall=aff_wall, aff_cpu=aff_cpu, opt_wall=opt_wall, opt_cpu=opt_cpu)))
'''

LOADED = r'''
import ctypes, json, os
import openTSNE
libc = ctypes.CDLL('/usr/lib/libSystem.B.dylib')
libc._dyld_get_image_name.restype = ctypes.c_char_p
images = [libc._dyld_get_image_name(i).decode() for i in range(libc._dyld_image_count())]
print('libomp images loaded after import openTSNE:')
for p in sorted({os.path.realpath(p) for p in images if 'libomp' in os.path.basename(p)}):
    print('   ', p)
'''


def shell(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    except FileNotFoundError:
        return 'n/a'


def run_config(n, n_jobs, extra_env):
    env = dict(os.environ, **extra_env)
    results = []
    for _ in range(REPEATS):
        out = subprocess.run([sys.executable, '-c', ONE_RUN, str(n), str(n_jobs)], env=env,
                             capture_output=True, text=True, check=True).stdout
        results.append(json.loads(out.strip().splitlines()[-1]))
    return {k: statistics.median(r[k] for r in results) for k in results[0]}


def main():
    cores = os.cpu_count()
    print('Python', sys.version.split()[0], platform.platform(), 'cpu_count', cores)
    if platform.system() == 'Darwin':
        for key in ('machdep.cpu.brand_string', 'hw.ncpu', 'hw.activecpu', 'hw.nperflevels',
                    'hw.perflevel0.logicalcpu', 'hw.perflevel1.logicalcpu', 'kern.hv_vmm_present'):
            print(f'  {key}: {shell(["sysctl", "-n", key]) or "n/a"}')
        print('  Homebrew libomp:', shell(['brew', 'list', '--versions', 'libomp']))
        subprocess.run([sys.executable, '-c', LOADED], check=False)
    for var in ('OMP_NUM_THREADS', 'KMP_BLOCKTIME', 'OMP_WAIT_POLICY', 'KMP_DUPLICATE_LIB_OK'):
        print(f'  {var}={os.environ.get(var, "(unset)")}')

    configs = [(1, {}), (2, {}), (cores, {}), (cores, {'KMP_BLOCKTIME': '0'})]
    print(f'\nMedian of {REPEATS} runs; cpu/wall > 1 means threads ran concurrently')
    print(f'{"events":>7} {"n_jobs":>6} {"env":<15} {"affinity s":>10} {"cpu/wall":>8} '
          f'{"BH opt s":>9} {"cpu/wall":>8} {"opt speed-up":>12}')
    for n in SIZES:
        base = None
        for n_jobs, extra in configs:
            r = run_config(n, n_jobs, extra)
            base = base or r['opt_wall']
            label = ','.join(f'{k}={v}' for k, v in extra.items()) or 'default'
            print(f'{n:>7} {n_jobs:>6} {label:<15} {r["aff_wall"]:>10.2f} '
                  f'{r["aff_cpu"] / r["aff_wall"]:>8.2f} {r["opt_wall"]:>9.2f} '
                  f'{r["opt_cpu"] / r["opt_wall"]:>8.2f} {base / r["opt_wall"]:>11.2f}x',
                  flush=True)


if __name__ == '__main__':
    main()
