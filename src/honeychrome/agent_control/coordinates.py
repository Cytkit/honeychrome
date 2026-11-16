"""
coordinates.py
--------------
Gate geometry for the agent tools. Gates hold plot coordinates: the value
after the channel's axis transform (``Transform.xform``), or the raw value on
an untransformed axis. Agents give and read data units; this module converts
between the two and builds or reshapes FlowKit gates exactly as the plot
widget does (``CytometryPlotWidget.new_*_gate`` and ``update_*``).
"""

from __future__ import annotations

import numpy as np
from flowkit import gates as Gates

from honeychrome.controller_components.functions import (
    define_ellipse_gate, define_polygon_gate, define_quad_gates, define_range_gate, define_rectangle_gate,
)

GATE_TYPES = ('rectangle', 'polygon', 'ellipse', 'range', 'quadrant')
TWO_D_TYPES = ('rectangle', 'polygon', 'ellipse', 'quadrant')
NAME_PREFIX = {'rectangle': 'R', 'polygon': 'P', 'ellipse': 'E', 'range': 'Range', 'quadrant': 'Q'}


class GeometryError(ValueError):
    """Coordinates that cannot describe the requested gate."""


def to_plot(transform, values):
    v = np.atleast_1d(np.asarray(values, dtype=np.float64))
    if transform.xform is None:
        return v
    return np.asarray(transform.xform.apply(v.ravel()), dtype=np.float64).reshape(v.shape)


def to_data(transform, values):
    v = np.atleast_1d(np.asarray(values, dtype=np.float64))
    if transform.xform is None:
        return v
    return np.asarray(transform.xform.inverse(v.ravel()), dtype=np.float64).reshape(v.shape)


def transform_name(transform) -> str:
    return {0: 'linear', 1: 'logicle', 2: 'log'}.get(transform.id, 'none (raw linear)')


def axis_range(transform) -> list:
    """The visible axis range in data units."""
    lo, hi = to_data(transform, transform.limits)
    return [float(lo), float(hi)]


def _number(coords: dict, key: str) -> float:
    if key not in coords:
        raise GeometryError(f'Missing "{key}" in coordinates.')
    try:
        value = float(coords[key])
    except (TypeError, ValueError):
        raise GeometryError(f'"{key}" must be a number.') from None
    if not np.isfinite(value):
        raise GeometryError(f'"{key}" must be finite.')
    return value


def _point(value, label: str) -> tuple:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise GeometryError(f'{label} must be a pair [x, y].')
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        raise GeometryError(f'{label} must contain numbers.') from None
    return x, y


def plot_geometry(gate_type: str, coords: dict, tx, ty=None, units: str = 'data') -> dict:
    """
    The ROI geometry of a gate in plot coordinates, from the agent's
    ``coords`` in ``units`` ('data' or 'plot'). ``tx``/``ty`` are the
    transforms of the plot's x and y channels.
    """
    if units not in ('data', 'plot'):
        raise GeometryError('units must be "data" or "plot".')
    conv_x = (lambda v: to_plot(tx, v)) if units == 'data' else (lambda v: np.atleast_1d(np.asarray(v, float)))
    conv_y = (lambda v: to_plot(ty, v)) if units == 'data' else (lambda v: np.atleast_1d(np.asarray(v, float)))

    if gate_type in ('rectangle', 'ellipse'):
        x0, x1 = sorted(conv_x([_number(coords, 'x_min'), _number(coords, 'x_max')]))
        y0, y1 = sorted(conv_y([_number(coords, 'y_min'), _number(coords, 'y_max')]))
        if x1 <= x0 or y1 <= y0:
            raise GeometryError('The gate has zero width or height.')
        if gate_type == 'rectangle':
            return {'pos': [x0, y0], 'size': [x1 - x0, y1 - y0]}
        angle = float(coords.get('angle', 0.0) or 0.0)
        size = np.array([x1 - x0, y1 - y0])
        centre = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
        theta = np.deg2rad(angle)
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        pos = centre - 0.5 * rot @ size
        return {'pos': pos.tolist(), 'size': size.tolist(), 'angle': angle}

    if gate_type == 'polygon':
        vertices = coords.get('vertices')
        if not isinstance(vertices, list) or len(vertices) < 3:
            raise GeometryError('A polygon needs "vertices": a list of at least 3 [x, y] points.')
        points = [_point(v, f'vertex {i}') for i, v in enumerate(vertices)]
        xs = conv_x([p[0] for p in points])
        ys = conv_y([p[1] for p in points])
        return {'vertices': [(float(x), float(y)) for x, y in zip(xs, ys)]}

    if gate_type == 'range':
        x0, x1 = sorted(conv_x([_number(coords, 'x_min'), _number(coords, 'x_max')]))
        if x1 <= x0:
            raise GeometryError('The range has zero width.')
        return {'x1': float(x0), 'x2': float(x1)}

    if gate_type == 'quadrant':
        return {'x': float(conv_x(_number(coords, 'x'))[0]), 'y': float(conv_y(_number(coords, 'y'))[0])}

    raise GeometryError(f'Unknown gate type "{gate_type}". Use one of: {", ".join(GATE_TYPES)}.')


def outside_axes(gate_type: str, geom: dict, tx, ty=None) -> bool:
    """Whether any part of the gate lies outside the visible axis range."""
    def out(t, values):
        lo, hi = t.limits
        return any(v < lo or v > hi for v in values)

    if gate_type == 'rectangle':
        (x0, y0), (dx, dy) = geom['pos'], geom['size']
        return out(tx, [x0, x0 + dx]) or out(ty, [y0, y0 + dy])
    if gate_type == 'ellipse':
        return False
    if gate_type == 'polygon':
        return out(tx, [v[0] for v in geom['vertices']]) or out(ty, [v[1] for v in geom['vertices']])
    if gate_type == 'range':
        return out(tx, [geom['x1'], geom['x2']])
    return out(tx, [geom['x']]) or out(ty, [geom['y']])


def build_gate(name: str, gate_type: str, geom: dict, channel_x: str, channel_y: str | None, transformations: dict):
    """A new FlowKit gate, as the plot widget's ``new_*_gate`` builds it."""
    if gate_type == 'rectangle':
        dim_x, dim_y = define_rectangle_gate(geom['pos'], geom['size'], channel_x, channel_y, transformations)
        return Gates.RectangleGate(name, dimensions=[dim_x, dim_y])
    if gate_type == 'polygon':
        vertices, dim_x, dim_y = define_polygon_gate(geom['vertices'], channel_x, channel_y, transformations)
        return Gates.PolygonGate(name, [dim_x, dim_y], vertices, use_complement=False)
    if gate_type == 'ellipse':
        dim_x, dim_y, centre, covariance, distance_square = define_ellipse_gate(
            geom['pos'], geom['size'], geom['angle'], channel_x, channel_y, transformations)
        return Gates.EllipsoidGate(name, [dim_x, dim_y], centre, covariance, distance_square)
    if gate_type == 'range':
        dim_x = define_range_gate(geom['x1'], geom['x2'], channel_x, transformations)
        return Gates.RectangleGate(name, dimensions=[dim_x])
    quad_divs, quadrants = define_quad_gates(geom['x'], geom['y'], channel_x, channel_y, transformations)
    return Gates.QuadrantGate(name, dividers=quad_divs, quadrants=quadrants)


def reshape_gate(gate, gate_type: str, geom: dict, channel_x: str, channel_y: str | None, transformations: dict):
    """Give an existing gate new geometry, as the plot widget's ``update_*`` does."""
    if gate_type == 'rectangle':
        gate.dimensions = list(define_rectangle_gate(geom['pos'], geom['size'], channel_x, channel_y, transformations))
    elif gate_type == 'polygon':
        gate.vertices = geom['vertices']
    elif gate_type == 'ellipse':
        _dx, _dy, centre, covariance, distance_square = define_ellipse_gate(
            geom['pos'], geom['size'], geom['angle'], channel_x, channel_y, transformations)
        gate.coordinates = centre
        gate.covariance_matrix = covariance
        gate.distance_square = distance_square
    elif gate_type == 'range':
        gate.dimensions = [define_range_gate(geom['x1'], geom['x2'], channel_x, transformations)]
    else:
        quad_divs, quadrants = define_quad_gates(geom['x'], geom['y'], channel_x, channel_y, transformations)
        gate.quadrants = {q.id: q for q in quadrants}
        gate.dimensions = quad_divs


def gate_kind(gate) -> str | None:
    """The agent-facing type of a FlowKit gate."""
    if gate.gate_type == 'RectangleGate':
        return 'rectangle' if len(gate.dimensions) == 2 else 'range'
    return {'PolygonGate': 'polygon', 'EllipsoidGate': 'ellipse', 'QuadrantGate': 'quadrant'}.get(gate.gate_type)


def describe_gate(gate, transformations: dict) -> dict:
    """The gate's channels and geometry in data units."""
    kind = gate_kind(gate)
    out = {'type': kind}

    def data(channel, values, transformed):
        if not transformed or channel not in transformations:
            return [float(v) for v in values]
        return [float(v) for v in to_data(transformations[channel], values)]

    if kind in ('rectangle', 'range'):
        out['channels'] = [d.id for d in gate.dimensions]
        for axis, d in zip(('x', 'y'), gate.dimensions):
            lo, hi = data(d.id, [d.min, d.max], d.transformation_ref)
            out[f'{axis}_min'], out[f'{axis}_max'] = lo, hi
    elif kind == 'polygon':
        cx, cy = gate.dimensions[0].id, gate.dimensions[1].id
        out['channels'] = [cx, cy]
        xs = data(cx, [v[0] for v in gate.vertices], gate.dimensions[0].transformation_ref)
        ys = data(cy, [v[1] for v in gate.vertices], gate.dimensions[1].transformation_ref)
        out['vertices'] = [[x, y] for x, y in zip(xs, ys)]
    elif kind == 'ellipse':
        cx, cy = gate.dimensions[0].id, gate.dimensions[1].id
        out['channels'] = [cx, cy]
        centre = np.asarray(gate.coordinates, dtype=np.float64)
        out['centre'] = [data(cx, [centre[0]], gate.dimensions[0].transformation_ref)[0],
                         data(cy, [centre[1]], gate.dimensions[1].transformation_ref)[0]]
        eigvals, eigvecs = np.linalg.eigh(np.asarray(gate.covariance_matrix, dtype=np.float64))
        order = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, order]
        out['plot_coordinates'] = {'centre': centre.tolist(), 'axes': eigvals[order].tolist(),
                                   'angle': float(np.rad2deg(np.arctan2(eigvecs[1, 0], eigvecs[0, 0])))}
    elif kind == 'quadrant':
        out['channels'] = [d.dimension_ref for d in gate.dimensions]
        for axis, d in zip(('x', 'y'), gate.dimensions):
            out[axis] = data(d.dimension_ref, [d.values[0]], d.transformation_ref)[0]
        out['quadrants'] = list(gate.quadrants.keys())
    return out
