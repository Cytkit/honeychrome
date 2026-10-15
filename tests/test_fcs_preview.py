"""Tests for the streaming FCS preview reader (tests adapted from Joan Escrivà Font's pull request #16)."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from flowio import create_fcs
from flowkit import Sample

from honeychrome.controller_components.fcs_preview import load_fcs_preview, read_preview_metadata, preview_indices


class FCSPreviewTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'sample.fcs'
        self.events = np.arange(3000, dtype=np.float32).reshape(1000, 3)
        with self.path.open('wb') as stream:
            create_fcs(stream, self.events.ravel().tolist(), ['Time', 'FSC-A', 'B1-A'],
                       metadata_dict={'TIMESTEP': '0.001', 'P2G': '2'})

    def test_uniform_preview_matches_flowkit_preprocessing_and_column_order(self):
        reference = Sample(self.path).get_events(source='raw')
        indices = preview_indices(1000, 73)
        sample, events = load_fcs_preview(self.path, 73, ['B1-A', 'Time', 'FSC-A'])
        np.testing.assert_array_equal(events, reference[indices][:, [2, 0, 1]])
        self.assertEqual(sample.event_count, 1000)
        self.assertTrue(sample.is_display_subset)
        _, again = load_fcs_preview(self.path, 73, ['B1-A', 'Time', 'FSC-A'])
        np.testing.assert_array_equal(events, again)

    def test_preview_selects_same_events_as_display_cap(self):
        expected = np.sort(np.random.default_rng(42).choice(1000, 73, replace=False))
        np.testing.assert_array_equal(preview_indices(1000, 73), expected)

    def test_all_events_returned_when_within_limit(self):
        sample, events = load_fcs_preview(self.path, 5000)
        self.assertEqual(events.shape, (1000, 3))
        self.assertFalse(sample.is_display_subset)

    def test_metadata_read_and_preview_never_use_full_sample_loader(self):
        with patch('flowkit.Sample', side_effect=AssertionError('Full loader used')):
            meta = read_preview_metadata(self.path)
            self.assertIsNone(meta.events)
            _, events = load_fcs_preview(self.path, 50, metadata=meta)
        self.assertEqual(events.shape, (50, 3))

    def test_truncated_data_and_unsupported_format_fail_before_allocation(self):
        meta = read_preview_metadata(self.path)
        with self.path.open('r+b') as stream:
            stream.truncate(self.path.stat().st_size - 100)
        with self.assertRaisesRegex(ValueError, 'truncated'):
            load_fcs_preview(self.path, 50, metadata=meta)
        meta.text['datatype'] = 'I'
        with self.assertRaisesRegex(ValueError, 'float32/float64'):
            load_fcs_preview(self.path, 50, metadata=meta)

    def test_float64_big_endian_and_log_preprocessing(self):
        meta = read_preview_metadata(self.path)
        meta.text['datatype'] = 'D'
        meta.text['byteord'] = '8,7,6,5,4,3,2,1'
        for i in range(1, 4):
            meta.text[f'p{i}b'] = '64'
        offset = int(meta.text['begindata'])
        with self.path.open('r+b') as stream:
            stream.seek(offset)
            stream.write(self.events.astype('>f8').tobytes())
        meta.channels[3]['pne'] = (2.0, 1.0)
        meta.channels[3]['pnr'] = 4096.0
        _, events = load_fcs_preview(self.path, 1000, metadata=meta)
        np.testing.assert_array_equal(events[:, 2], 10 ** (2 * self.events[:, 2].astype(float) / 4096))


if __name__ == '__main__':
    unittest.main()
