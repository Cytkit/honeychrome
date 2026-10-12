from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from flowio import create_fcs

import honeychrome.settings as settings
from honeychrome.controller import Controller


class PreviewCacheTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cap = patch.object(settings, 'max_display_events', 500)
        self.budget = patch.object(settings, 'sample_cache_mb', 16)
        self.cap.start()
        self.budget.start()
        self.addCleanup(self.cap.stop)
        self.addCleanup(self.budget.stop)
        self.c = Controller()
        self.c.experiment.create(Path(self.directory.name) / 'study.kit')
        self.c.experiment_dir = self.c.experiment.generate_subdirs()
        self.c.initialise_ephemeral_data()
        self.c.set_mode('Raw Data')
        labels = self.c.experiment.settings['raw']['event_channels_pnn']
        for name in ['A', 'B', 'C', 'D', 'E', 'F']:
            events = np.random.default_rng(ord(name)).uniform(100, 2000, (3000, len(labels)))
            events[:, 0] = np.arange(3000)
            with (self.c.experiment_dir / f'Raw/{name}.fcs').open('wb') as stream:
                create_fcs(stream, events.ravel().tolist(), labels)

    def test_return_visit_reuses_events_and_calculations(self):
        self.c.experiment.process['unmixing_matrix'] = [[1.0]]
        unmix = patch.object(self.c, '_apply_unmixing', side_effect=lambda events: events * 2)
        unmix_mock = unmix.start()
        self.addCleanup(unmix.stop)
        with patch('honeychrome.controller.sample_from_fcs', side_effect=AssertionError('Full loader used')):
            self.c.load_sample('Raw/A.fcs')
            expected = self.c.raw_event_data.copy()
            expected_unmixed = self.c.unmixed_event_data.copy()
            hists = [hist.copy() for hist in self.c.data_for_cytometry_plots['histograms']]
            self.c.load_sample('Raw/B.fcs')
            with patch.object(self.c, 'calc_hists_and_stats', wraps=self.c.calc_hists_and_stats) as calc:
                self.c.load_sample('Raw/A.fcs')
                self.assertEqual(calc.call_count, 0)
            self.assertEqual(unmix_mock.call_count, 2)
        np.testing.assert_array_equal(self.c.raw_event_data, expected)
        np.testing.assert_array_equal(self.c.unmixed_event_data, expected_unmixed)
        for actual, expected in zip(self.c.data_for_cytometry_plots['histograms'], hists):
            np.testing.assert_array_equal(actual, expected)

    def test_limit_gate_and_file_changes_invalidate(self):
        self.c.load_sample('Raw/A.fcs')
        self.c.load_sample('Raw/B.fcs')
        settings.max_display_events = 750
        with patch.object(self.c, 'calc_hists_and_stats', wraps=self.c.calc_hists_and_stats) as calc:
            self.c.load_sample('Raw/A.fcs')
            self.assertGreater(calc.call_count, 0)
        self.assertEqual(len(self.c.raw_event_data), 750)
        key = self.c.preview_cache.key('Raw/A.fcs')
        gate = self.c.raw_gating.get_gate('Cells', ('root',))
        gate.dimensions[0].min += 0.01
        self.assertNotEqual(self.c.preview_cache.key('Raw/A.fcs'), key)
        key = self.c.preview_cache.key('Raw/A.fcs')
        path = self.c.experiment_dir / 'Raw/A.fcs'
        with path.open('ab') as stream:
            stream.write(b'\x00')
        self.assertNotEqual(self.c.preview_cache.key('Raw/A.fcs'), key)

    def test_budget_eviction_disable_and_experiment_reset(self):
        for name in ['A', 'B', 'C', 'D', 'E', 'F']:
            self.c.load_sample(f'Raw/{name}.fcs')
            self.assertLessEqual(self.c.preview_cache.bytes_used, 16 * 1024 ** 2)
            self.assertLessEqual(len(self.c.preview_cache.entries), 4)
        settings.sample_cache_mb = 0
        self.c.load_sample('Raw/A.fcs')
        self.assertEqual(self.c.preview_cache.bytes_used, 0)
        settings.sample_cache_mb = 16
        self.c.load_sample('Raw/B.fcs')
        self.c.flush_ephemeral_data()
        self.assertEqual(self.c.preview_cache.bytes_used, 0)


if __name__ == '__main__':
    unittest.main()
