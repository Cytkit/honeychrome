import hashlib
import json
from copy import deepcopy

import numpy as np

from honeychrome.controller_components.functions import apply_gates_in_place, gate_lookup_tables
from honeychrome.controller_components.gml_functions_mod_from_flowkit import gate_to_gml
import logging
logger = logging.getLogger(__name__)

# --- Per-sample custom gates (FlowKit custom sample gates) ---------------
#
# All samples share one gating hierarchy (the template). A sample may
# override the geometry of a single gate; the override is stored as a
# FlowKit *custom sample gate* on the shared GatingStrategy
# (``add_gate(..., sample_id=...)``) and mirrored in ``custom_sample_gates``
# so it survives gating rebuilds (and can be persisted later). The gate name
# and hierarchy are unchanged, so plots keep resolving the same source gate.


def _gate_path_for(gating, gate_name):
    """Gate path tuple for ``gate_name`` in ``gating`` (defaults to root)."""
    for gid, gpath in gating.get_gate_ids():
        if gid == gate_name:
            return gpath
    return ('root',)


def install_custom_gate(gating, gate_name, gate, sample_id):
    """(Re)install ``gate`` as ``sample_id``'s custom gate for ``gate_name``."""
    gate_path = _gate_path_for(gating, gate_name)
    try:
        if gating.is_custom_gate(sample_id, gate_name):
            gating.remove_gate(gate_name, sample_id=sample_id)
    except Exception:
        pass
    gating.add_gate(deepcopy(gate), gate_path=gate_path, sample_id=sample_id)


def copy_gate_geometry(src_gate, dst_gate):
    """Copy geometry (not name/hierarchy) from ``src_gate`` onto ``dst_gate``.

    Covers the gate types Honeychrome edits: Rectangle/Range (``dimensions``)
    and Ellipsoid (``coordinates`` / ``covariance_matrix`` / ``distance_square``).
    """
    for attr in ('dimensions', 'coordinates', 'covariance_matrix', 'distance_square'):
        if hasattr(src_gate, attr) and hasattr(dst_gate, attr):
            setattr(dst_gate, attr, deepcopy(getattr(src_gate, attr)))

def serialize_custom_sample_gates(custom_sample_gates):
    """Serialize this scope's per-sample custom gates to
    ``{sample_path: {gate_name: gml_fragment}}`` for the .kit file. Each gate
    is stored as a standalone GatingML string (only the overridden gates)."""
    out = {}
    for sample_path, gates in custom_sample_gates.items():
        serialized = {}
        for gate_name, gate in gates.items():
            try:
                serialized[gate_name] = gate_to_gml(gate)
            except Exception as e:
                logger.warning(f'_serialize_custom_sample_gates: {gate_name!r} for {sample_path} ({e})')
        if serialized:
            out[sample_path] = serialized
    return out


# --- Gating any sample with its own gates -------------------------------
#
# The controller's lookup tables are built for the sample on display: a gate
# with a per-sample override is rasterised with that sample's geometry, and
# per-sample axes (Time) span that sample's range. Anything that gates other
# samples (the Statistics tab, plugins) must not reuse those tables.


def _digest(value) -> str:
    return hashlib.sha1(json.dumps(value, sort_keys=True, default=str).encode('utf-8')).hexdigest()[:16]


def _transform_params(tr) -> list:
    return [getattr(tr, k, None) for k in ('id', 'limits', 'scale_t', 'linear_a', 'logicle_w', 'logicle_m',
                                           'logicle_a', 'log_m', 'scale_bins')]


def _gate_channels(gate) -> list:
    if gate.gate_type == 'QuadrantGate':
        return [d.dimension_ref for d in gate.dimensions]
    return gate.get_dimension_ids()


def _gml(gate) -> str:
    try:
        return gate_to_gml(gate)
    except Exception:
        return repr(sorted(vars(gate).items(), key=lambda kv: kv[0]))


class SampleGating:
    """
    Gates the events of any sample as the main window gates that sample when
    it is displayed:

    * each gate's effective boundary for the sample (its override if it has
      one, the template otherwise) is rasterised onto the transform bins with
      ``functions.gate_lookup_tables``;
    * per-sample ('default', e.g. Time) axes span 0 to 1.05 × the sample's
      maximum;
    * template tables on fixed axes are computed once and shared by every
      sample.

    Only gate boundaries are used: events are binned by their own values, so
    the same object gates any data in the same channels.

    gating      : FlowKit GatingStrategy (the template hierarchy).
    transforms  : {channel: Honeychrome Transform}.
    overrides   : {sample key: {gate name: gate}} (per-sample custom gates).
    seed_tables : optional {gate name: {table name: lookup table}} known to be
                  the template's tables on these transforms.
    """

    def __init__(self, gating, transforms: dict, overrides: dict | None = None,
                 seed_tables: dict | None = None):
        self.gating = gating
        self.transforms = dict(transforms or {})
        self.overrides = {str(k): dict(v) for k, v in (overrides or {}).items()}
        self._seed = dict(seed_tables or {})
        self._template_tables: dict = {}

    @classmethod
    def from_controller(cls, controller, scope: str = 'unmixed', copy: bool = True) -> 'SampleGating':
        """
        The controller's gating, transforms and overrides for *scope*
        ('unmixed' or 'raw'). The displayed sample's overrides are read from
        the live strategy, where ROI edits land first.

        copy=True (for gating events, possibly on a worker thread) takes deep
        copies and seeds the template cache with the controller's tables for
        gates on fixed axes that the displayed sample does not override.
        copy=False only reads the controller, for ``signature()`` on the main
        thread.
        """
        prefix = 'raw' if scope == 'raw' else 'unmixed'
        gating = getattr(controller, f'{prefix}_gating', None)
        transforms = getattr(controller, f'{prefix}_transformations', None) or {}
        tables = getattr(controller, f'{prefix}_lookup_tables', None) or {}
        overrides = {str(k): dict(v) for k, v in
                     ((getattr(controller, 'custom_sample_gates', None) or {}).get(scope) or {}).items()}

        shown = getattr(controller, 'current_sample_path', None)
        shown = str(shown) if shown is not None else None
        if shown is not None and gating is not None:
            live = dict(overrides.get(shown, {}))
            for gid, _path in gating.get_gate_ids():
                try:
                    if gating.is_custom_gate(shown, gid):
                        live[gid] = gating.get_gate(gid, sample_id=shown)
                except Exception:
                    continue
            if live:
                overrides[shown] = live

        if not copy:
            return cls(gating, transforms, overrides)

        seed = {}
        shown_overrides = set(overrides.get(shown, {})) if shown is not None else set()
        if gating is not None:
            for name, gate in cls._nodes(gating):
                if name in shown_overrides or any(
                        getattr(transforms.get(ch), 'id', None) == 'default' for ch in _gate_channels(gate)):
                    continue
                names = list(gate.quadrants.keys()) if gate.gate_type == 'QuadrantGate' else [name]
                if all(n in tables for n in names):
                    seed[name] = {n: np.array(tables[n], copy=True) for n in names}
        return cls(deepcopy(gating), deepcopy(dict(transforms)), deepcopy(overrides), seed)

    @staticmethod
    def _nodes(gating) -> list:
        """(name, template gate) for every gate with its own lookup table(s)."""
        out = []
        for gid, gpath in gating.get_gate_ids():
            if gating._get_gate_node(gid, gpath).gate_type == 'Quadrant':
                continue
            out.append((gid, gating.get_gate(gid)))
        return out

    def gate_names(self) -> list:
        return [gid for gid, _path in self.gating.get_gate_ids()]

    def signature(self, sample_keys=None) -> str:
        """Identifies the gating of *sample_keys* (every sample when None):
        the template gates, those samples' overrides and the axis transforms.
        Use it to tell when results computed from this gating are stale."""
        keys = sorted(self.overrides) if sample_keys is None else sorted(str(k) for k in sample_keys)
        return _digest({
            'gates': [[gid, list(gpath), _gml(self.gating.get_gate(gid))]
                      for gid, gpath in self.gating.get_gate_ids()
                      if self.gating._get_gate_node(gid, gpath).gate_type != 'Quadrant'],
            'overrides': {k: {g: _gml(gate) for g, gate in self.overrides.get(k, {}).items()} for k in keys},
            'transforms': {ch: _transform_params(tr) for ch, tr in self.transforms.items()},
        })

    def _sample_transforms(self, events: np.ndarray, pnn: list) -> dict:
        out = dict(self.transforms)
        for ch, tr in self.transforms.items():
            if getattr(tr, 'id', None) == 'default' and ch in pnn:
                col = events[:, pnn.index(ch)]
                if col.size:
                    sample_tr = deepcopy(tr)
                    sample_tr.set_transform(limits=[0, float(np.max(col)) * 1.05])
                    out[ch] = sample_tr
        return out

    def _lookup_tables(self, sample_key: str, transforms: dict) -> dict:
        own = self.overrides.get(str(sample_key), {})
        tables = {}
        for name, gate in self._nodes(self.gating):
            channels = _gate_channels(gate)
            if any(ch not in transforms for ch in channels):
                continue
            if name in own:
                tables.update(gate_lookup_tables(own[name], name, transforms))
            elif any(getattr(transforms[ch], 'id', None) == 'default' for ch in channels):
                tables.update(gate_lookup_tables(gate, name, transforms))
            else:
                if name not in self._template_tables:
                    self._template_tables[name] = (self._seed.pop(name, None)
                                                   or gate_lookup_tables(gate, name, transforms))
                tables.update(self._template_tables[name])
        return tables

    def memberships(self, sample_key: str, events: np.ndarray, pnn: list) -> dict:
        """{gate name: bool mask} for every gate that can be evaluated on
        *events* (columns named by *pnn*), plus 'root'."""
        pnn = list(pnn)
        transforms = self._sample_transforms(events, pnn)
        data = {'event_data': events, 'pnn': pnn, 'transformations': transforms,
                'lookup_tables': self._lookup_tables(sample_key, transforms), 'gating': self.gating,
                'gate_membership': {'root': np.ones(len(events), dtype=bool)}}
        apply_gates_in_place(data, gates_to_calculate=self.gate_names())
        return data['gate_membership']

    def population_mask(self, sample_key: str, events: np.ndarray, pnn: list, gate_names: list) -> np.ndarray:
        """Events inside any of *gate_names* (their union) for one sample;
        every event when *gate_names* is empty. Raises ValueError for a gate
        that cannot be evaluated (it does not exist, or one of its or an
        ancestor's channels is not in *pnn*)."""
        if not gate_names:
            return np.ones(len(events), dtype=bool)
        membership = self.memberships(sample_key, events, pnn)
        masks = []
        for name in gate_names:
            if name not in membership:
                raise ValueError(f'Gate "{name}" could not be applied: it does not exist, or one of its '
                                 f'channels (or an ancestor gate\'s) is not in the data.')
            masks.append(np.asarray(membership[name], dtype=bool))
        return np.logical_or.reduce(masks)
