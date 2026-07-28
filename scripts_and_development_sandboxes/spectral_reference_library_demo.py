"""Demo of the curated Spectral Reference Library store (steps 1-2).

Run:  python scripts_and_development_sandboxes/spectral_reference_library_demo.py

Uses a throwaway temp DB, so it does not touch your real library. Shows the
foundation that backs both Spectral Process (exact config match) and Spectral QC
(loose cytometer match): compute_config_key + save / flag / list / delete.
"""
import tempfile
from pathlib import Path

from honeychrome.controller_components.spectral_reference_library import (
    compute_config_key,
    SpectralReferenceLibrary,
)


def line(title):
    print('\n' + '=' * 4, title, '=' * 4)


def main():
    tmp = Path(tempfile.mkdtemp()) / 'demo_library.db'
    lib = SpectralReferenceLibrary(library_path=tmp)
    lib.ensure_schema()

    line('config_key is order-independent (sorted channels)')
    aurora_5l = ['B1-A', 'B2-A', 'V5-A', 'UV6-A', 'R1-A']
    print('ordered   :', compute_config_key('Aurora', aurora_5l))
    print('shuffled  :', compute_config_key('Aurora', list(reversed(aurora_5l))))
    print('3-laser   :', compute_config_key('Aurora', ['B1-A', 'B2-A', 'V5-A']), '  <- different config, different key')

    cfg = compute_config_key('Aurora', aurora_5l)

    line('save two user spectra for BUV805 + one shipped (honeychrome) one')
    p_lot1 = lib.save_profile('BUV805', {c: 0.1 for c in aurora_5l}, 'Aurora', cfg, aurora_5l,
                              display_name='BUV805 (lot 1)', source_sample_name='Tube_003.fcs')
    p_lot2 = lib.save_profile('BUV805', {c: 0.2 for c in aurora_5l}, 'Aurora', cfg, aurora_5l,
                              display_name='BUV805 (lot 2, fresh)', source_sample_name='Tube_009.fcs')
    p_ship = lib.save_profile('BUV805', {c: 0.15 for c in aurora_5l}, 'Aurora', cfg, aurora_5l,
                              display_name='BUV805 (Cytek shipped)', origin='honeychrome')
    print(f'  saved ids: lot1={p_lot1.id}, lot2={p_lot2.id}, shipped={p_ship.id}')

    line('flag the fresh lot as the Reference (Spectral Process) + QC Target')
    lib.set_reference(p_lot2.id)
    lib.set_qc_target(p_lot2.id)

    line('list BUV805 on Aurora')
    for p in lib.list_profiles('Aurora'):
        tags = []
        if p.is_reference:
            tags.append('★REF')
        if p.is_qc_target:
            tags.append('QC')
        if not p.is_deletable:
            tags.append('shipped/undeletable')
        print(f'  [{p.id}] {p.display_name:28} {p.origin:11} {" ".join(tags)}')

    print('\n  get_reference_for_config(BUV805) ->', lib.get_reference_for_config('BUV805', cfg).display_name)
    print('  get_qc_target_for_cytometer(BUV805, Aurora) ->', lib.get_qc_target_for_cytometer('BUV805', 'Aurora').display_name)

    line('shipped profiles cannot be deleted')
    try:
        lib.delete_profile(p_ship.id)
    except ValueError as e:
        print('  delete shipped ->', e)
    lib.delete_profile(p_lot1.id)
    print('  deleted user lot1; remaining BUV805 rows:', len(lib.list_profiles('Aurora')))


if __name__ == '__main__':
    main()
