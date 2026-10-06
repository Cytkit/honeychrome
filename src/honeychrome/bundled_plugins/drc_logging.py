"""
drc_logging.py — Logging helpers for the DR / Clustering / Statistics plugin
============================================================================
Companion module to ``dr_clustering_tab.py``.  Not a plugin itself (the
filename deliberately does **not** end in ``_tab.py`` so ``plugin_loaders``
will not try to load it as a separate tab).

Provides:
  • get_logger()          — a configured ``logging.Logger`` shared across the
                            plugin's split modules.
  • log_array()           — log the structure of a numpy feature matrix
                            (shape, dtype, per-column min/median/max).
  • log_channel_map()     — log the channel → column mapping actually used.
  • log_transform_params()— log the transform parameters per channel so we can
                            verify the *user's* selections are being applied.
  • log_stage()           — a short banner marking a pipeline stage.

Why a dedicated logger (not ``print``)?
  The existing plugin only emits ``print()`` lines via ``progress_message``.
  A real logger lets us (a) gate verbosity with a single level, (b) tag every
  line with the originating module/function, and (c) leave the statements in
  place permanently without spamming the status bar.

Usage
-----
    from drc_logging import get_logger, log_array, log_stage
    log = get_logger(__name__)
    log_stage(log, "LOAD TRAINING POOL")
    log_array(log, "training_pool", data, channel_names)
"""

from __future__ import annotations

import logging
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Logger configuration
# ---------------------------------------------------------------------------
#
# Level is controlled by the environment variable HONEYCHROME_DRC_LOGLEVEL
# (DEBUG / INFO / WARNING / ...).  Defaults to INFO, which is what surfaces
# the per-stage data-structure summaries requested for debugging.
#
_DEFAULT_LEVEL = os.environ.get("HONEYCHROME_DRC_LOGLEVEL", "INFO").upper()
_ROOT_NAME = "honeychrome.plugins.dr_clustering"
_configured = False


def _configure_root() -> None:
    global _configured
    if _configured:
        return
    root = logging.getLogger(_ROOT_NAME)
    root.setLevel(getattr(logging, _DEFAULT_LEVEL, logging.INFO))
    # Only add a handler if the host application hasn't already configured one
    # for this branch of the tree — avoids duplicate lines.
    if not root.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(
            logging.Formatter("[DRC %(levelname)s] %(funcName)s: %(message)s")
        )
        root.addHandler(handler)
        root.propagate = False
    _configured = True


def get_logger(module_name: str) -> logging.Logger:
    """
    Return a child logger under the plugin's root logger.

    ``module_name`` is normally ``__name__``; we strip it down to the final
    component so lines read e.g. ``drc_pipeline`` rather than the full path
    importlib assigns to a file-loaded plugin module.
    """
    _configure_root()
    short = module_name.rsplit(".", 1)[-1] if module_name else "plugin"
    return logging.getLogger(f"{_ROOT_NAME}.{short}")


# ---------------------------------------------------------------------------
# Structure-logging helpers
# ---------------------------------------------------------------------------

def log_stage(log: logging.Logger, title: str) -> None:
    """Emit a short banner so stage boundaries are easy to spot in the log."""
    log.info("---- %s ----", title)


def log_array(
    log: logging.Logger,
    name: str,
    arr: np.ndarray | None,
    channel_names: list[str] | None = None,
    max_cols: int = 40,
) -> None:
    """
    Log the structure of a feature matrix: shape, dtype, contiguity, the
    presence of non-finite values, and per-column min / median / max.

    This is the core instrument for verifying that the data flowing into DR /
    clustering is the data we think it is (right channels, right magnitudes,
    right transform applied).
    """
    if arr is None:
        log.warning("%s: <None>", name)
        return

    arr = np.asarray(arr)
    n_rows = arr.shape[0]
    n_cols = arr.shape[1] if arr.ndim > 1 else 1
    n_nonfinite = int(np.count_nonzero(~np.isfinite(arr))) if arr.size else 0

    log.debug(
        "%s: shape=%s dtype=%s C_contig=%s non_finite=%d",
        name, tuple(arr.shape), arr.dtype,
        arr.flags["C_CONTIGUOUS"], n_nonfinite,
    )

    if arr.size == 0 or arr.ndim != 2:
        return

    # Per-column summary (guard against all-NaN columns)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        col_min = np.nanmin(arr, axis=0)
        col_med = np.nanmedian(arr, axis=0)
        col_max = np.nanmax(arr, axis=0)

    names = channel_names if channel_names and len(channel_names) >= n_cols \
        else [f"col{i}" for i in range(n_cols)]

    shown = min(n_cols, max_cols)
    for i in range(shown):
        log.debug(
            "    %-22s  min=% .4g  med=% .4g  max=% .4g",
            names[i], col_min[i], col_med[i], col_max[i],
        )
    if n_cols > shown:
        log.debug("    … (%d more columns suppressed)", n_cols - shown)


def log_channel_map(
    log: logging.Logger,
    full_channels: list[str],
    selected_channels: list[str],
    ch_idx: dict[str, int],
) -> None:
    """
    Log the channel → column index mapping that will be used to slice the
    unmixed event matrix.  Flags any selected channel that cannot be resolved
    to a column (a classic source of silent misalignment).
    """
    log.debug(
        "channel map: %d unmixed columns, %d selected for DR/clustering",
        len(full_channels), len(selected_channels),
    )
    missing = [c for c in selected_channels if c not in ch_idx]
    if missing:
        log.warning("  selected channels NOT found in unmixed columns: %s", missing)
    log.debug("  full unmixed channel order: %s", full_channels)
    log.debug(
        "  selected → column: %s",
        {c: ch_idx.get(c) for c in selected_channels},
    )


def log_transform_params(
    log: logging.Logger,
    channel: str,
    transform_id: int,
    params: dict,
    source: str = "state.channel_transform_params",
) -> None:
    """
    Log the transform actually applied to one channel, so we can confirm the
    user's configured parameters (not a hard-coded default / arcsinh fallback)
    are being used.
    """
    kind = {0: "linear", 1: "logicle", 2: "log"}.get(transform_id, f"id={transform_id}")
    if not params:
        log.warning(
            "  %-22s transform=%s  (NO params found in %s — using defaults!)",
            channel, kind, source,
        )
        return
    log.debug(
        "  %-22s transform=%-7s  T=%g W=%g M=%g A=%g",
        channel, kind,
        params.get("T", float("nan")),
        params.get("W", float("nan")),
        params.get("M", float("nan")),
        params.get("A", float("nan")),
    )


def log_files(log: logging.Logger, label: str, rel_paths: list[str]) -> None:
    """Log the list of sample files entering a pipeline stage."""
    log.info("%s: %d file(s)", label, len(rel_paths))
    for rp in rel_paths:
        log.info("    %s", rp)
