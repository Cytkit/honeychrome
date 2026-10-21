"""
method_unmix_worker.py
----------------------
Background unmixing of the displayed sample with the experiment's slow
method (noise model or spectral variants, ``unmix_method``). The controller
shows per-cell AF at once and swaps this result in when it arrives; the
result is written to the display cache here, off the main thread.
"""

from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal

from honeychrome.controller_components import unmix_method

logger = logging.getLogger(__name__)


class MethodUnmixWorker(QObject):
    """Runs ``unmix_method.unmix_fluorescence`` for one display job in a QThread."""
    finished = Signal(object, object)   # job, fluorescence result
    error = Signal(object, str)          # job, message

    def __init__(self, job: dict):
        super().__init__()
        self.job = job

    def run(self):
        job = self.job
        try:
            raw_fl = job['raw'][:, list(job['fl_ids'])]
            result = unmix_method.unmix_fluorescence(raw_fl, job['state'], job['af'], job['sample_key'],
                                                     job['method'])
            if job.get('cache_key'):
                unmix_method.store_display_cache(job['experiment_dir'], job['cache_key'], result)
            self.finished.emit(job, result)
        except Exception as e:
            logger.exception('Background unmixing failed')
            self.error.emit(job, str(e) or type(e).__name__)
