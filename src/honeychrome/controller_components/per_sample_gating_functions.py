from copy import deepcopy
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