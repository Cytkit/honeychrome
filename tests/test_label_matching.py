"""
test_label_matching.py
----------------------
Marker and fluorophore name matching (controller_components/label_matching):
matches remembered per database, and each database served its own patterns.

Usage:
    pytest tests/test_label_matching.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from honeychrome.controller_components import label_matching as lm  # noqa: E402


def _db(*markers):
    return [{'marker': m, **{f'synonym{i}': '' for i in range(1, 10)}} for m in markers]


def test_remembered_matches_equal_a_fresh_scan():
    db = lm.get_marker_db()
    names = ['CD4 BUV395', 'anti-CD8a', 'Live Dead Blue', 'FSC-A', 'CD45RA', 'nothing']
    first = [lm.match_marker(n, db) for n in names]
    again = [lm.match_marker(n, db) for n in names]
    fresh = []
    for n in names:
        hit = None
        for pattern, _val, canonical in lm._build_patterns(db, 'marker',
                                                           [f'synonym{i}' for i in range(1, 10)]):
            if pattern.search(n):
                hit = canonical
                break
        fresh.append(hit)
    assert first == again == fresh


def test_each_database_gets_its_own_patterns():
    cd4, cd8 = _db('CD4'), _db('CD8')
    assert lm.match_marker('CD4', cd4) == 'CD4'
    assert lm.match_marker('CD4', cd8) is None
    assert lm.match_marker('CD8', cd8) == 'CD8'
