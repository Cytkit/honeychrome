# import json
from pathlib import Path
# from unicodedata import category

import numpy as np
from PySide6.QtCore import QObject, Signal, QTimer, QSettings
# from PySide6.QtWidgets import QApplication
from flowkit import Sample
from typing import cast

from honeychrome.controller_components.functions import apply_transfer_matrix, export_unmixed_sample, sample_from_fcs
from honeychrome.controller_components.autospectral_functions import precompute_af_matrices, combine_af_precomputed, apply_af_transfer
import honeychrome.settings as settings
from honeychrome.__init__ import __version__

import logging
logger = logging.getLogger(__name__)

class UnmixedExporter(QObject):
    finished = Signal()

    def __init__(self, folder, subsample_bool, bus, controller):
        super().__init__()

        # connect
        self.folder = folder
        if subsample_bool:
            self.subsample: int | None = cast(int, settings.subsample_retrieved)
        else:
            self.subsample: int | None = None

        self.controller = controller
        self.bus = bus

    def run(self):
        all_samples = self.controller.experiment.samples['all_samples']
        exp_dir = self.controller.experiment_dir
        raw_subdir = self.controller.experiment.settings['raw']['raw_samples_subdirectory']

        folder_abs = (exp_dir / self.folder).resolve()
        raw_subdir_abs = (exp_dir / raw_subdir).resolve()

        def sample_key_to_abs(key):
            """Return the resolved absolute path for a sample key regardless of storage form."""
            p = Path(key)
            if p.is_absolute():
                return p.resolve()
            return (exp_dir / p).resolve()

        def key_is_under(key, abs_folder):
            """True if the sample key lives inside abs_folder, comparing resolved real paths.
            Also checks each ancestor of the key via samefile, to handle the case where
            the key was stored using a different symlink name than the one in abs_folder
            (e.g. settings stored 'Link_to__...' but the UI path is 'Raw/Single stain controls').
            """
            key_abs = sample_key_to_abs(key)
            # Fast path: direct prefix match after resolving both sides
            try:
                key_abs.relative_to(abs_folder)
                return True
            except ValueError:
                pass
            # Slow path: walk the key's parent chain and samefile-compare each ancestor
            # against abs_folder. This catches the case where two different symlink names
            # point to the same real directory.
            for parent in key_abs.parents:
                try:
                    if parent.samefile(abs_folder):
                        return True
                except OSError:
                    pass
            return False

        samples_to_calculate = [s for s in all_samples if key_is_under(s, folder_abs)]

        if self.controller.experiment.process['unmixing_matrix'] is not None:
            unmixing_matrix = np.array(self.controller.experiment.process['unmixing_matrix'])
            spillover = np.array(self.controller.experiment.process['spillover'])
            pnn_raw = self.controller.experiment.settings['raw']['event_channels_pnn']
            pnn_unmixed = self.controller.experiment.settings['unmixed']['event_channels_pnn'] # ssr review: consider setting this to antigen fluorophore self.controller.data_for_cytometry_plots['pnn_unmixed']?

            fl_channel_ids_raw = np.array(self.controller.filtered_raw_fluorescence_channel_ids)
            sc_channel_ids_raw = np.array(self.controller.experiment.settings['raw']['scatter_channel_ids'])
            fl_channel_ids_unmixed = np.array(self.controller.experiment.settings['unmixed']['fluorescence_channel_ids'])
            sc_channel_ids_unmixed = np.array(self.controller.experiment.settings['unmixed']['scatter_channel_ids'])
            n_scatter_channels = self.controller.experiment.settings['unmixed']['n_scatter_channels']

            transfer_matrix = np.zeros((len(pnn_unmixed), len(pnn_raw)))
            transfer_matrix[np.ix_(fl_channel_ids_unmixed, fl_channel_ids_raw)] = unmixing_matrix
            transfer_matrix[np.ix_(sc_channel_ids_unmixed, sc_channel_ids_raw)] = np.eye(n_scatter_channels)
            transfer_matrix[self.controller.experiment.settings['unmixed']['time_channel_id'], self.controller.experiment.settings['raw']['time_channel_id']] = 1

            if self.controller.experiment.settings['raw']['event_id_channel_id'] is not None:
                transfer_matrix[self.controller.experiment.settings['unmixed']['event_id_channel_id'], self.controller.experiment.settings['raw']['event_id_channel_id']] = 1
            # note transfer_matrix is transposed - multiply raw event data @ transfer_matrix to get unmixed event data in same form as raw
            transfer_matrix = transfer_matrix.T


            # --- One-time per-batch setup ---

            # Build fluorophore spectra matrix for the SPECTRA keyword.
            # controller._build_fluor_spectra() reads experiment.process['profiles']
            # and stacks rows in spectral_model order → (n_fluor × n_raw_detectors).
            unmixing_spectra = self.controller._build_fluor_spectra()  # may be None if model incomplete

            unmixing_method = self.controller.experiment.settings.get('unmixing_method', 'OLS')
            unmixing_weights_raw = self.controller.experiment.process.get('unmixing_weights')
            unmixing_weights = np.array(unmixing_weights_raw) if unmixing_weights_raw is not None else None

            # For FACSDiscover: detect imaging channels and extend the transfer matrix.
            # These channels are excluded from unmixing (they are not spectral detectors)
            # but should be preserved in the exported FCS file because they carry
            # per-event imaging data needed for downstream analysis.
            cytometer = self.controller.experiment.settings['raw'].get('cytometer', '')
            imaging_channel_ids_raw = []
            imaging_pnn = []

            if 'FACSDiscover' in cytometer:
                from honeychrome.controller_components.cytometer_whitelist import _CYTOMETER_PARAMS
                import re as _re

                params = _CYTOMETER_PARAMS.get('FACSDiscover')
                if params is not None:
                    EXCLUDE_FROM_IMAGING = {'FSC', 'SSC', 'Time'}
                    imaging_prefixes = [
                        p for p in params.non_spectral_pat
                        if not p.startswith('-')
                        and p not in EXCLUDE_FROM_IMAGING
                    ]
                    imaging_pat = _re.compile(
                        '|'.join(rf'(?:^|\b){_re.escape(p)}' for p in imaging_prefixes)
                    )

                    already_used = (
                        set(fl_channel_ids_raw.tolist())
                        | set(sc_channel_ids_raw.tolist())
                        | {self.controller.experiment.settings['raw']['time_channel_id']}
                    )
                    if self.controller.experiment.settings['raw']['event_id_channel_id'] is not None:
                        already_used.add(self.controller.experiment.settings['raw']['event_id_channel_id'])

                    imaging_channel_ids_raw = [
                        i for i, ch in enumerate(pnn_raw)
                        if _re.search(imaging_pat, ch) and i not in already_used
                    ]
                    imaging_pnn = [pnn_raw[i] for i in imaging_channel_ids_raw]

            # Extend pnn_unmixed and transfer_matrix if imaging channels were found.
            if imaging_channel_ids_raw:
                n_unmixed_old = len(pnn_unmixed)
                pnn_unmixed = pnn_unmixed + imaging_pnn   # extend the name list

                # Current transfer_matrix shape: (n_raw, n_unmixed_old) because it
                # is stored transposed (raw event data @ transfer_matrix → unmixed).
                # We add n_imaging extra output columns via identity rows for those
                # raw input columns.
                n_raw = transfer_matrix.shape[0]
                extra_cols = np.zeros((n_raw, len(imaging_channel_ids_raw)))
                for out_col_offset, in_row in enumerate(imaging_channel_ids_raw):
                    extra_cols[in_row, out_col_offset] = 1.0
                transfer_matrix = np.hstack([transfer_matrix, extra_cols])
                # transfer_matrix is now (n_raw, n_unmixed_old + n_imaging)

            # Update the whitelist in raw_param_lookup so imaging channels get
            # their original metadata carried through (voltage, range, etc.).
            # This is handled automatically inside define_fcs_keywords() because
            # imaging channel names are now present in pnn_unmixed and will be
            # looked up against the raw_param_lookup — but they need to be in the
            # whitelist. We extend unmixed_settings temporarily rather than
            # touching _build_raw_param_lookup's whitelist argument directly;
            # the whitelist is built from raw scatter+time, and imaging channels
            # are added by passing them explicitly via a new key.
            imaging_carry_through_set = set(imaging_pnn)  # consumed in define_fcs_keywords


            for n, sample_path in enumerate(samples_to_calculate):
                logger.info(f'UnmixedExporter: sample {n+1}/{len(samples_to_calculate)}')
                if self.bus:
                    self.bus.progress.emit(n, len(samples_to_calculate))

                sample_name = all_samples[sample_path]
                sample_abs = sample_key_to_abs(sample_path)
                sample_rel_suffix = None
                # Try raw_subdir first (normal case), then folder_abs (symlink case where
                # raw_subdir is empty / points elsewhere but folder_abs resolves correctly).
                for anchor in [raw_subdir_abs, folder_abs]:
                    for parent in [sample_abs] + list(sample_abs.parents):
                        try:
                            if parent.samefile(anchor):
                                sample_rel_suffix = sample_abs.relative_to(parent)
                                break
                        except OSError:
                            pass
                    if sample_rel_suffix is not None:
                        break
                if sample_rel_suffix is None:
                    # Final fallback: strip by component count of whichever anchor is shallower
                    anchor = min(raw_subdir_abs, folder_abs, key=lambda p: len(p.parts))
                    sample_rel_suffix = Path(*sample_abs.parts[len(anchor.parts):])
                unmixed_rel_path = (Path(self.controller.experiment.settings['unmixed']['unmixed_samples_subdirectory']) /
                                    sample_rel_suffix)

                full_sample_path = self.controller.experiment_dir / sample_path
                full_unmixed_sample_path = self.controller.experiment_dir / unmixed_rel_path
                full_unmixed_sample_path.parent.mkdir(parents=True, exist_ok=True)
                sample = sample_from_fcs(full_sample_path, self.bus)
                _all_events = sample.get_events(source='raw')
                _sample_ch_idx = {ch: i for i, ch in enumerate(sample.pnn_labels)}
                if set(pnn_raw) <= set(sample.pnn_labels):
                    # Fast path: all pnn_raw channels present — slice in order
                    raw_event_data = _all_events[:, [_sample_ch_idx[ch] for ch in pnn_raw]]
                else:
                    # Some pnn_raw channels absent in this file — build aligned array with zeros
                    raw_event_data = np.zeros((_all_events.shape[0], len(pnn_raw)), dtype=_all_events.dtype)
                    for dst, ch in enumerate(pnn_raw):
                        if ch in _sample_ch_idx:
                            raw_event_data[:, dst] = _all_events[:, _sample_ch_idx[ch]]
                np.nan_to_num(raw_event_data, copy=False, nan=0.0)
                raw_keywords: dict[str, str] = cast(dict[str, str], sample.get_metadata())
                n_events = sample.event_count

                if n_events > 0:
                    # Check whether this sample has AutoSpectral AF profiles assigned
                    sample_af_profiles = self.controller.experiment.samples.get('sample_af_profiles', {})
                    assigned_profile_names = sample_af_profiles.get(sample_path, [])
                    all_af_profiles = self.controller.experiment.process.get('af_profiles', {})
                    active_profiles = [all_af_profiles[name] for name in assigned_profile_names if name in all_af_profiles]

                    if active_profiles:
                        # Build combined AF precomputed matrices for this sample's assigned profiles
                        fluor_spectra = self.controller._build_fluor_spectra()
                        precomputed_list = [
                            precompute_af_matrices(
                                fluor_spectra,
                                np.array(p['spectra'])
                            )
                            for p in active_profiles
                        ]
                        af_precomputed = combine_af_precomputed(precomputed_list)
                        af_spectra = np.vstack([np.array(p['spectra']) for p in active_profiles])

                        # Remap fluorescence channel indices from full event_channels_pnn
                        # to positions in pnn_raw (the column order of raw_event_data).
                        # Profiles and af_spectra were computed in whitelisted-PNN space;
                        # raw_event_data here is aligned to pnn_raw (event_channels_pnn).
                        # For cytometers where pnn_raw == whitelisted_pnn this is a no-op.
                        _raw_settings = self.controller.experiment.settings['raw']
                        _full_pnn = _raw_settings['event_channels_pnn']
                        _fl_ids_remapped = [
                            pnn_raw.index(_full_pnn[i])
                            for i in self.controller.filtered_raw_fluorescence_channel_ids
                            if _full_pnn[i] in pnn_raw
                        ]

                        af_result = apply_af_transfer(
                            raw_event_data,
                            transfer_matrix,
                            af_precomputed,
                            af_spectra,
                            self.controller.experiment.settings,
                            filtered_fl_ids_raw=_fl_ids_remapped,
                            spillover=None,
                        )
                        # apply_af_transfer fills the AF Abundance and AF Index columns.
                        export_event_data = af_result['unmixed']
                        export_pnn = list(pnn_unmixed)
                        logger.info(f'UnmixedExporter: using AF unmixing for {sample_path} ({len(active_profiles)} profile(s))')
                    else:
                        # Without AF correction the AF channels carry no information,
                        # so they are left out of the exported file.
                        unmixed_all = apply_transfer_matrix(transfer_matrix, raw_event_data)
                        keep = [i for i, ch in enumerate(pnn_unmixed) if ch not in settings.af_channels]
                        export_event_data = unmixed_all[:, keep]
                        export_pnn = [pnn_unmixed[i] for i in keep]

                    # Retrieve the unmixing spectra matrix (n_fluor × n_detectors).
                    # stored in experiment.process after unmixing is computed.
                    unmixing_spectra = np.array(
                        self.controller.experiment.process.get('spectra_matrix')
                    ) if self.controller.experiment.process.get('spectra_matrix') is not None else None

                    af_spectra_export = af_spectra if active_profiles else None
                    extra_null = None

                    export_unmixed_sample(
                        sample_name=sample_name,
                        unmixed_folder=full_unmixed_sample_path.parent,
                        export_event_data=export_event_data,
                        export_pnn=export_pnn,
                        spillover=spillover,
                        raw_keywords=raw_keywords,
                        spectral_model=self.controller.experiment.process.get('spectral_model', []),
                        unmixed_settings=self.controller.experiment.settings['unmixed'],
                        raw_settings=self.controller.experiment.settings['raw'],
                        af_spectra=af_spectra_export,
                        unmixing_spectra=unmixing_spectra,
                        version=__version__,
                        subsample=self.subsample,
                        extra_null_channels=extra_null,
                        unmixing_method=unmixing_method,
                        unmixing_weights=unmixing_weights,
                        extra_whitelist=imaging_carry_through_set,
                    )

            logger.info(f'UnmixedExporter: finished')

            if self.bus:
                self.bus.progress.emit(len(samples_to_calculate), len(samples_to_calculate))
                # self.bus.popupMessage.emit(f'Exported {len(samples_to_calculate)} unmixed samples as FCS files. \n\n'
                #                            f'Open <a href="file:///{self.controller.experiment_dir / self.controller.experiment.settings['unmixed']['unmixed_samples_subdirectory']}">'
                #                            f'{self.controller.experiment.settings['unmixed']['unmixed_samples_subdirectory']}</a> folder.')
                self.bus.popupMessage.emit(f'Exported {len(samples_to_calculate)} unmixed samples as FCS files, to \n'
                                           f'"{self.controller.experiment.settings['unmixed']['unmixed_samples_subdirectory']}" folder in experiment folder')
        self.finished.emit()


if __name__ == "__main__":
    from pathlib import Path
    from honeychrome.controller import Controller
    # from honeychrome.view_components.event_bus import EventBus

    kc = Controller()
    base_directory = Path.home() / 'spectral_cytometry'
    # is this a test set-up?
    experiment_name = base_directory / '20240620 Spectral Symposium-poor cell unmixed'
    experiment_path = experiment_name.with_suffix('.kit')
    kc.load_experiment(experiment_path) # note this loads first sample too and runs calculate all histograms and statistics

    unmixed_exporter = UnmixedExporter('Raw/Samples/Spleen', True, None, kc)
    unmixed_exporter.run()

