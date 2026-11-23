"""Default Cells and Singlets gates from single-stain control landmarks
(controller_components/landmark_gates.py)."""
import numpy as np

from honeychrome.controller_components.landmark_gates import (
    LANDMARK_BLOCK_ROWS, LANDMARK_BLOCKS, SINGLET_MIN_HALF_WIDTH, densest_box, landmark_rows,
    robust_band, singlet_geometry)


def test_landmark_rows_small_file_reads_every_event():
    assert np.array_equal(landmark_rows(500), np.arange(500))


def test_landmark_rows_large_file_reads_blocks_from_start_to_end():
    rows = landmark_rows(1_000_000)
    assert len(rows) == LANDMARK_BLOCKS * LANDMARK_BLOCK_ROWS
    assert rows[0] == 0 and rows[-1] == 1_000_000 - 1
    assert np.all(np.diff(rows) > 0)


def test_densest_box_holds_the_dense_cluster_and_ignores_sparse_events():
    rng = np.random.default_rng(0)
    cluster = rng.normal((0.35, 0.18), (0.04, 0.03), size=(450, 2))
    sparse = rng.uniform((0.2, 0.3), (0.9, 1.0), size=(50, 2))
    box = densest_box([np.vstack([cluster, sparse])], (0, 1), (0, 1))
    x_min, x_max, y_min, y_max = box
    assert x_min < 0.30 and x_max > 0.40
    assert y_min < 0.14 and y_max > 0.22
    assert y_max < 0.6


def test_densest_box_weights_controls_equally():
    rng = np.random.default_rng(1)
    many = rng.normal((0.3, 0.2), 0.02, size=(5000, 2))
    few = rng.normal((0.6, 0.6), 0.02, size=(100, 2))
    box = densest_box([many, few], (0, 1), (0, 1))
    assert box[1] > 0.6 and box[3] > 0.6


def test_robust_band_centres_on_the_singlets_and_excludes_doublets():
    rng = np.random.default_rng(2)
    ratios = np.concatenate([rng.normal(0.8, 0.04, 900), rng.normal(0.45, 0.05, 100)])
    low, high = robust_band(ratios)
    assert 0.55 < low < 0.75
    assert high > 0.9


def test_robust_band_needs_enough_events():
    assert robust_band(np.array([0.8, 0.81])) is None


def test_robust_band_has_a_minimum_width_for_tight_populations():
    rng = np.random.default_rng(3)
    low, high = robust_band(rng.normal(0.92, 0.005, 1000))
    assert low < 0.92 * (1 - SINGLET_MIN_HALF_WIDTH) + 0.01
    assert high > 0.92 * (1 + SINGLET_MIN_HALF_WIDTH) - 0.01


def test_singlet_geometry_is_a_ratio_band_inside_the_cells_box():
    rng = np.random.default_rng(4)
    x = rng.uniform(0.2, 0.6, 2000)
    points = np.column_stack([x, rng.uniform(0.1, 0.3, 2000), 0.8 * x * rng.normal(1, 0.02, 2000)])
    vertices = singlet_geometry(points, (0.2, 0.6, 0.1, 0.3), 'FSC-H')
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = vertices
    assert x0 == x3 == 0.2 and x1 == x2 == 0.6
    assert y0 / x0 < 0.8 < y3 / x3
    assert singlet_geometry(points, (0.2, 0.6, 0.1, 0.3), None) is None
