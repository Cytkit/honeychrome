from PyInstaller.utils.hooks import collect_submodules

# openTSNE's compiled modules import one another from C (for example _tsne
# imports _matrix_mul.matrix_mul), which PyInstaller's analysis cannot see
hiddenimports = collect_submodules('openTSNE')
