"""Bounded, session-only cache of preview events and computed plot results."""
from collections import OrderedDict
import hashlib
import logging
import pickle
import numpy as np

import honeychrome.settings as settings
from honeychrome.controller_components.gml_functions_mod_from_flowkit import to_gml

logger = logging.getLogger(__name__)


class PreviewCache:
    def __init__(self, controller):
        self.controller = controller
        self.entries = OrderedDict()
        self.bytes_used = 0
        self.loaded_key = None

    def clear(self):
        self.entries.clear()
        self.bytes_used = 0
        self.loaded_key = None

    def key(self, sample_path):
        c = self.controller
        path = c.experiment_dir / sample_path
        stat = path.stat()
        # Default axis limits vary by sample and are restored with cached data.
        def recipes(transforms):
            return {
                name: {key: value for key, value in vars(transform).items()
                       if key in ('id', 'scale_t', 'linear_a', 'logicle_w',
                                  'logicle_m', 'logicle_a', 'log_m')
                       or (key == 'limits' and transform.id != 'default')}
                for name, transform in (transforms or {}).items()
            }
        dependencies = (
            c.experiment.settings, c.experiment.process,
            c.experiment.samples.get('sample_af_profiles'), c.af_precomputed_cache,
            c.transfer_matrix,
            to_gml(c.raw_gating), to_gml(c.unmixed_gating) if c.unmixed_gating else None,
            recipes(c.raw_transformations), recipes(c.unmixed_transformations),
            [data['plots'] for data in self.plot_data()],
            settings.max_display_events, settings.hist_bins_retrieved,
            settings.density_cutoff_retrieved, settings.hist2dtype_retrieved,
        )
        # Hash values rather than aliases, which can change during restoration.
        hasher = hashlib.sha256()
        def hash_value(value):
            if isinstance(value, dict):
                hasher.update(b'dict')
                for name in sorted(value, key=str):
                    hash_value(name)
                    hash_value(value[name])
            elif isinstance(value, (list, tuple)):
                hasher.update(f'sequence:{len(value)}'.encode())
                for item in value:
                    hash_value(item)
            elif isinstance(value, np.ndarray):
                hasher.update(str((value.shape, value.dtype.str)).encode())
                hasher.update(value.tobytes(order='C'))
            else:
                if isinstance(value, np.generic):
                    value = value.item()
                hasher.update(pickle.dumps(value, protocol=5))
        hash_value(dependencies)
        digest = hasher.digest()
        return (str(path.resolve()), stat.st_size, stat.st_mtime_ns, digest)

    def plot_data(self):
        c = self.controller
        return (c.data_for_cytometry_plots_raw, c.data_for_cytometry_plots_process,
                c.data_for_cytometry_plots_unmixed)

    def remember_loaded(self):
        c = self.controller
        self.loaded_key = self.key(c.current_sample_path) if c.current_sample else None

    def before_load(self, sample_path):
        c = self.controller
        budget = settings.sample_cache_mb * 1024 ** 2
        if not budget:
            self.clear()
            return False
        # Skip partially recalculated results after analysis edits.
        if (self.loaded_key and c.current_sample_path != c.live_sample_path
                and c.raw_event_data is not None
                and self.loaded_key == self.key(c.current_sample_path)):
            payload = {
                'sample': c.current_sample,
                'raw': c.raw_event_data, 'unmixed': c.unmixed_event_data,
                'af_sidecar': c.af_sidecar_data,
                'raw_transforms': c.raw_transformations,
                'unmixed_transforms': c.unmixed_transformations,
                'raw_lookup': c.raw_lookup_tables, 'unmixed_lookup': c.unmixed_lookup_tables,
                'plots': [{key: data[key] for key in ('histograms', 'statistics', 'gate_membership')}
                          for data in self.plot_data()],
            }
            if self.array_bytes(payload) > budget:
                return self.restore(sample_path, budget)
            blob = pickle.dumps(payload, protocol=5)
            previous = self.entries.pop(self.loaded_key, None)
            if previous is not None:
                self.bytes_used -= len(previous)
            if len(blob) <= budget:
                while self.entries and (self.bytes_used + len(blob) > budget or len(self.entries) >= 4):
                    _, oldest = self.entries.popitem(last=False)
                    self.bytes_used -= len(oldest)
                self.entries[self.loaded_key] = blob
                self.bytes_used += len(blob)
        return self.restore(sample_path, budget)

    @staticmethod
    def array_bytes(value, seen=None):
        seen = set() if seen is None else seen
        if id(value) in seen:
            return 0
        seen.add(id(value))
        if isinstance(value, np.ndarray):
            return value.nbytes
        if isinstance(value, dict):
            return sum(PreviewCache.array_bytes(item, seen) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(PreviewCache.array_bytes(item, seen) for item in value)
        if hasattr(value, '__dict__'):
            return PreviewCache.array_bytes(vars(value), seen)
        return 0

    def restore(self, sample_path, budget):
        c = self.controller
        while self.entries and self.bytes_used > budget:
            _, oldest = self.entries.popitem(last=False)
            self.bytes_used -= len(oldest)
        key = self.key(sample_path)
        blob = self.entries.pop(key, None)
        if blob is None:
            return False
        # Remove the serialized copy when restoring its arrays.
        self.bytes_used -= len(blob)
        payload = pickle.loads(blob)
        c.current_sample_path = sample_path
        c.current_sample = payload['sample']
        c.raw_event_data = payload['raw']
        c.unmixed_event_data = payload['unmixed']
        c.af_sidecar_data = payload['af_sidecar']
        c.raw_transformations = payload['raw_transforms']
        c.unmixed_transformations = payload['unmixed_transforms']
        c.raw_lookup_tables = payload['raw_lookup']
        c.unmixed_lookup_tables = payload['unmixed_lookup']
        for index, (data, saved) in enumerate(zip(self.plot_data(), payload['plots'])):
            data.update(saved)
            data['event_data'] = c.raw_event_data if index == 0 else c.unmixed_event_data
            data['transformations'] = c.raw_transformations if index == 0 else c.unmixed_transformations
            data['lookup_tables'] = c.raw_lookup_tables if index == 0 else c.unmixed_lookup_tables
        c.initialise_af_matrices()
        c.initialise_data_for_cytometry_plots()
        if c.bus:
            if c.data_for_cytometry_plots is not None:
                c.bus.histsStatsRecalculated.emit(
                    c.current_mode, list(range(len(c.data_for_cytometry_plots['plots'])))
                )
            c.bus.statusMessage.emit(f'Cached sample: {sample_path}, {len(c.raw_event_data):,} events. Statistics use subset.')
        self.loaded_key = key
        logger.info('Preview cache hit: %s; cache %.1f MB', sample_path, self.bytes_used / 1024 ** 2)
        return True
