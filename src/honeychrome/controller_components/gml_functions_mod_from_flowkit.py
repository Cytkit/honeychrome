import networkx as nx
from copy import deepcopy
from flowkit import GatingStrategy
from flowkit._models.gates import QuadrantGate, BooleanGate
# from flowkit._resources import gml_schema
from flowkit._utils.gml_write import _add_transform_to_gml, _add_matrix_to_gml, _add_gates_from_gate_dict, \
    _add_gate_to_gml
from flowkit._utils.xml_utils import _construct_gates, _construct_transforms, _construct_matrices
from flowkit.exceptions import QuadrantReferenceError
from lxml import etree

import logging
logger = logging.getLogger(__name__)

def to_gml(gating_strategy):
    """
    **** Based on export_gatingml in Flowkit ****
    ** export to string instead of file#
    ** export all sample gates
    *********************************************

    Exports a valid GatingML 2.0 document from given GatingStrategy instance.
    Specify the sample ID to use that sample's custom gates in the exported
    file, otherwise the template gates will be exported.

    :param gating_strategy: A GatingStrategy instance
    :param file_handle: File handle for exported GatingML 2.0 document
    :param sample_id: an optional text string representing a Sample instance
    :return: None
    """
    ns_g = "http://www.isac-net.org/std/Gating-ML/v2.0/gating"
    ns_dt = "http://www.isac-net.org/std/Gating-ML/v2.0/datatypes"
    ns_xform = "http://www.isac-net.org/std/Gating-ML/v2.0/transformations"
    ns_map = {
        'gating': ns_g,
        'data-type': ns_dt,
        'transforms': ns_xform
    }

    root = etree.Element('{%s}Gating-ML' % ns_g, nsmap=ns_map)

    # process gating strategy transformations
    for xform_id, xform in gating_strategy.transformations.items():
        _add_transform_to_gml(root, xform_id, xform, ns_map)

    # process gating strategy compensation matrices
    for matrix_id, matrix in gating_strategy.comp_matrices.items():
        _add_matrix_to_gml(root, matrix_id, matrix, ns_map)

    # get gate hierarchy as a dictionary
    gate_dict = gating_strategy.get_gate_hierarchy('dict')

    # recursively convert all gates to GatingML
    _add_gates_from_gate_dict(gating_strategy, gate_dict, ns_map, root, sample_id=None) #todo export all sample gates

    et = etree.ElementTree(root)

    return etree.tostring(root, encoding="unicode", pretty_print=True)
    # return etree.tostring(root, encoding="utf-8", xml_declaration=True, pretty_print=True)


def _add_gates_from_gate_dict(gating_strategy, gate_dict, ns_map, parent_ml, sample_id=None):
    try:
        # the gate_dict will have keys 'name' and 'children'. top-level 'name' value is 'root'
        for child in gate_dict['children']:
            gate_id = child['name']

            try:
                gate = gating_strategy.get_gate(gate_id, sample_id=sample_id)
            except QuadrantReferenceError:
                # single quadrants will be handled in the owning quadrant gate
                gate = None

            if gate is not None:
                child_ml = _add_gate_to_gml(parent_ml, gate, ns_map)

                if gate_dict['name'] != 'root':
                    # this is a recursion, add the parent reference
                    child_ml.set('{%s}parent_id' % ns_map['gating'], gate_dict['name'])

            if 'children' in child:  # and not isinstance(gate, QuadrantGate):
                _add_gates_from_gate_dict(gating_strategy, child, ns_map, parent_ml, sample_id=sample_id)
    except:
        pass



def _get_xml_type(xml_file_or_path):
    """
    **** Based on parse_gating_xml in Flowkit ****
    ** import from string instead of file
    ** import all sample gates
    *********************************************
    """

    root = etree.fromstring(xml_file_or_path)
    doc_type = 'gatingml'

    gating_ns = None
    data_type_ns = None
    transform_ns = None

    # find GatingML target namespace in the map
    for ns, url in root.nsmap.items():
        if url == 'http://www.isac-net.org/std/Gating-ML/v2.0/gating':
            gating_ns = ns
        elif url == 'http://www.isac-net.org/std/Gating-ML/v2.0/datatypes':
            data_type_ns = ns
        elif url == 'http://www.isac-net.org/std/Gating-ML/v2.0/transformations':
            transform_ns = ns

    if gating_ns is None:
        raise ValueError("GatingML namespace reference is missing from GatingML file")

    return doc_type, root, gating_ns, data_type_ns, transform_ns


def from_gml(xml):
    """
    **** Based on parse_gating_xml in Flowkit ****
    ** import from string instead of file
    ** import all sample gates
    *********************************************

    Parse a GatingML 20 document and return as a GatingStrategy.

    :param xml: file handle or file path to a GatingML 2.0 document
    :return: GatingStrategy instance
    """
    doc_type, root_gml, gating_ns, data_type_ns, xform_ns = _get_xml_type(xml)

    gating_strategy = GatingStrategy()

    if doc_type == 'gatingml':
        gates = _construct_gates(root_gml, gating_ns, data_type_ns)
        transformations = _construct_transforms(root_gml, xform_ns, data_type_ns)
        comp_matrices = _construct_matrices(root_gml, xform_ns, data_type_ns)
    elif doc_type == 'flowjo':
        raise ValueError("File is a FlowJo workspace, use parse_wsp or Session.import_flowjo_workspace.")
    else:
        raise ValueError("Gating file format is not supported.")

    for c_id, c in comp_matrices.items():
        gating_strategy.add_comp_matrix(c_id, c)
    for t_id, t in transformations.items():
        gating_strategy.add_transform(t_id, t)

    deps = []
    quadrants = []
    bool_edges = []

    for g_id, gate in gates.items():
        # GML gates have a parent reference & their gate names are
        # required to be unique, so we can use them to assemble the tree
        if gate.parent is None:
            parent = 'root'
        else:
            parent = gate.parent

        deps.append((parent, g_id))

        if isinstance(gate, QuadrantGate):
            for q_id in gate.quadrants:
                deps.append((g_id, q_id))
                quadrants.append(q_id)

        if isinstance(gate, BooleanGate):
            for g_ref in gate.gate_refs:
                deps.append((g_ref['ref'], g_id))

                bool_edges.append((g_ref['ref'], g_id))

    dag = nx.DiGraph(deps)

    is_acyclic = nx.is_directed_acyclic_graph(dag)

    if not is_acyclic:
        raise ValueError("The given GatingML 2.0 file is invalid, cyclic gate dependencies are not allowed.")

    process_order = list(nx.algorithms.topological_sort(dag))

    for q_id in quadrants:
        try:
            process_order.remove(q_id)
        except ValueError:
            logger.warning(f"from_gml: quadrant id '{q_id}' not found in process_order — skipping")



    # remove boolean edges to create a true ancestor graph
    dag.remove_edges_from(bool_edges)

    for g_id in process_order:
        # skip 'root' node
        if g_id == 'root':
            continue
        gate = gates[g_id]

        # For Boolean gates we need to add gate paths to the
        # referenced gates via 'gate_path' key in the gate_refs dict
        if isinstance(gate, BooleanGate):
            bool_gate_refs = gate.gate_refs
            for gate_ref in bool_gate_refs:
                # since we're parsing GML, all gate IDs must be unique
                # so safe to lookup in our graph
                gate_ref_path = list(nx.all_simple_paths(dag, 'root', gate_ref['ref']))[0]
                gate_ref['path'] = tuple(gate_ref_path[:-1])  # don't repeat the gate name

        # need to get the gate path
        # again, since GML gate IDs must be unique, safe to lookup from graph
        gate_path = tuple(nx.shortest_path(dag, 'root', g_id))[:-1]

        # Convert GML gates to their superclass & add to gating strategy
        gating_strategy.add_gate(gate.convert_to_parent_class(), gate_path)

    return gating_strategy


def gate_to_gml(gate):
    """Serialize a single gate to a standalone GatingML 2.0 string.

    Reuses :func:`to_gml` by placing the gate alone (at root) in a throwaway
    GatingStrategy — so a per-sample custom gate is persisted with the exact
    same serializer as the template, with no bespoke per-gate-type code.
    """
    mini = GatingStrategy()
    mini.add_gate(deepcopy(gate), gate_path=('root',))
    return to_gml(mini)


def gate_from_gml(xml, gate_name):
    """Reconstruct a single gate (by name) from a GatingML string produced by
    :func:`gate_to_gml`."""
    return from_gml(xml).get_gate(gate_name)


# otb: have added this here. if you prefer, move to spectral_model_editor.py
def _rename_channel_in_gml(gml_string: str, old_name: str, new_name: str) -> str:
    """
    Rename a single fluorescence channel in a GatingML 2.0 XML string.

    Channel names appear as the 'data-type:name' attribute on
    'data-type:fcs-dimension' elements, and as the 'id' attribute on
    'transforms:*' transform elements (which are also keyed by channel name).
    Both are renamed by exact attribute-value match, so "PE" never touches
    "PE-Cy7", "PE-CF594", etc.

    Returns the serialised XML string unchanged if old_name is not found.
    """
    from lxml import etree

    NS_DT = 'http://www.isac-net.org/std/Gating-ML/v2.0/datatypes'
    NS_XFORM = 'http://www.isac-net.org/std/Gating-ML/v2.0/transformations'
    DT_NAME = f'{{{NS_DT}}}name'
    XFORM_ID = f'{{{NS_XFORM}}}id'  # transforms:id is namespace-qualified

    root = etree.fromstring(gml_string.encode())
    changed = False

    # 1. fcs-dimension name attributes (gate dimension references)
    for elem in root.iter(f'{{{NS_DT}}}fcs-dimension'):
        if elem.get(DT_NAME) == old_name:
            elem.set(DT_NAME, new_name)
            changed = True

    # 2. transform element id attributes (transform definitions keyed by channel)
    for elem in root.iter():
        if elem.get(XFORM_ID) == old_name and elem.tag.startswith(f'{{{NS_XFORM}}}'):
            elem.set(XFORM_ID, new_name)
            changed = True

    if not changed:
        return gml_string

    return etree.tostring(root, encoding='unicode', pretty_print=True)