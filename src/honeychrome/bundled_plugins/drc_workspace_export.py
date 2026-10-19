"""
drc_workspace_export.py — export Workspace pages (PDF, SVG, PNG, TIFF)
======================================================================
Companion module to ``dr_clustering_tab.py`` (not ``*_tab.py``).

Every item on a page is a matplotlib figure at its true size, so a page is
composed from the figures at their page positions:

- PDF  — each figure saved as SVG (text as paths, rasterised layers at the
         chosen dpi) and drawn by QSvgRenderer into its rectangle on a
         QPdfWriter page: vector, one PDF page per Workspace page. No new
         dependency (QtSvg is already used for the app's icons).
- SVG  — one file per page: the item SVGs nested at their positions (ids
         prefixed per item so they stay unique); text can be kept as
         editable text.
- PNG / TIFF — each figure rendered by Agg at the chosen dpi and composited
         with Pillow (a matplotlib dependency); one file per page, with the
         dpi written into the file.

Items are clipped to their page. Rendering for export changes a figure's
dpi, so each item is re-rendered for the screen afterwards.
"""

from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np

import drc_workspace_canvas as wc

MM_PER_PT = 25.4 / 72.0
PDF_RESOLUTION = 1200
DEFAULT_EXPORT_DPI = 300
FORMATS = ('PDF', 'SVG', 'PNG', 'TIFF')


# ----------------------------------------------------------------------
# What is on a page
# ----------------------------------------------------------------------

def page_entries(scene, page: int) -> list[tuple]:
    """[(item, x_mm, y_mm, w_mm, h_mm)] of every figure item whose page is
    *page* (a legend goes with its plot), positions relative to the page,
    bottom of the stacking order first."""
    ox, oy = scene.page_origin(page)
    out = []
    for top in sorted(scene.top_items(), key=lambda it: it.zValue()):
        if top.layout_mm['page'] != page:
            continue
        for item in [top] + [c for c in top.childItems() if isinstance(c, wc.FigureItem)]:
            r = item.scene_rect_mm()
            out.append((item, r.x() - ox, r.y() - oy, r.width(), r.height()))
    return out


def entries_for(items: list) -> tuple[list[tuple], float, float]:
    """Entries for *items* relative to their joint bounding box, and the
    box's (w, h) in mm — for exporting one plot and its legend."""
    rects = [it.scene_rect_mm() for it in items]
    x0 = min(r.x() for r in rects)
    y0 = min(r.y() for r in rects)
    x1 = max(r.right() for r in rects)
    y1 = max(r.bottom() for r in rects)
    entries = [(it, r.x() - x0, r.y() - y0, r.width(), r.height()) for it, r in zip(items, rects)]
    return entries, x1 - x0, y1 - y0


# ----------------------------------------------------------------------
# Figures as SVG / RGBA
# ----------------------------------------------------------------------

def figure_svg(fig, editable_text: bool, dpi: float, salt: str) -> str:
    """The figure as an SVG document at its own size (rasterised layers at
    *dpi*); *salt* keeps matplotlib's generated ids distinct per figure."""
    import matplotlib
    buf = io.StringIO()
    with matplotlib.rc_context({'svg.fonttype': 'none' if editable_text else 'path',
                                'svg.hashsalt': salt}):
        fig.savefig(buf, format='svg', dpi=dpi)
    return buf.getvalue()


def figure_rgba(fig, dpi: float) -> np.ndarray:
    """The figure rendered by Agg at *dpi* (h x w x 4, uint8)."""
    fig.set_dpi(dpi)
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


def _restore_screen_render(entries):
    for item, *_rest in entries:
        item.schedule_render()


# ----------------------------------------------------------------------
# SVG
# ----------------------------------------------------------------------

_ID_RE = re.compile(r'\bid="([^"]+)"')


def _nest_svg(svg: str, prefix: str, x: float, y: float, w: float, h: float) -> str:
    """An item's SVG document as a nested <svg> placed at (x, y), w x h in
    the page's mm units, with every id prefixed."""
    body = svg[svg.index('<svg'):]
    ids = set(_ID_RE.findall(body))
    for old in sorted(ids, key=len, reverse=True):
        new = f'{prefix}{old}'
        body = body.replace(f'id="{old}"', f'id="{new}"')
        body = body.replace(f'url(#{old})', f'url(#{new})')
        body = body.replace(f'href="#{old}"', f'href="#{new}"')
    head_end = body.index('>')
    head = body[:head_end]
    head = re.sub(r'\swidth="[^"]*"', '', head)
    head = re.sub(r'\sheight="[^"]*"', '', head)
    head += f' x="{x:.4f}" y="{y:.4f}" width="{w:.4f}" height="{h:.4f}"'
    return head + body[head_end:]


def page_svg(entries: list[tuple], page_w: float, page_h: float, editable_text: bool,
             dpi: float, white: bool = True) -> str:
    """A page as one SVG document (mm user units)."""
    parts = [
        '<?xml version="1.0" encoding="utf-8" standalone="no"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'version="1.1" width="{page_w:.4f}mm" height="{page_h:.4f}mm" '
        f'viewBox="0 0 {page_w:.4f} {page_h:.4f}">',
        '<defs><clipPath id="page_clip">'
        f'<rect x="0" y="0" width="{page_w:.4f}" height="{page_h:.4f}"/></clipPath></defs>',
    ]
    if white:
        parts.append(f'<rect x="0" y="0" width="{page_w:.4f}" height="{page_h:.4f}" fill="white"/>')
    parts.append('<g clip-path="url(#page_clip)">')
    for n, (item, x, y, w, h) in enumerate(entries):
        svg = figure_svg(item.figure(), editable_text, dpi, salt=f'{item.uid}-{n}')
        parts.append(_nest_svg(svg, f'i{n}_', x, y, w, h))
    parts.append('</g></svg>')
    return '\n'.join(parts)


def export_svg(path_stem: Path, scene, pages: list[int], editable_text: bool = True,
               dpi: float = DEFAULT_EXPORT_DPI) -> list[Path]:
    """One SVG file per page: <stem>.svg for a single page, else
    <stem>_page<n>.svg."""
    written = []
    for page in pages:
        entries = page_entries(scene, page)
        p = scene.pages[page]
        path = _page_path(path_stem, page, len(pages), '.svg')
        path.write_text(page_svg(entries, p['w_mm'], p['h_mm'], editable_text, dpi),
                        encoding='utf-8')
        _restore_screen_render(entries)
        written.append(path)
    return written


# ----------------------------------------------------------------------
# PDF
# ----------------------------------------------------------------------

def export_pdf(path: Path, scene, pages: list[int], dpi: float = DEFAULT_EXPORT_DPI) -> Path:
    """One PDF, one page per Workspace page, each at its own size."""
    from PySide6.QtCore import QByteArray, QMarginsF, QRectF, QSizeF
    from PySide6.QtGui import QPageLayout, QPageSize, QPainter, QPdfWriter
    from PySide6.QtSvg import QSvgRenderer

    writer = QPdfWriter(str(path))
    writer.setResolution(PDF_RESOLUTION)
    writer.setCreator('Honeychrome')

    def set_page(index):
        p = scene.pages[index]
        size = QPageSize(QSizeF(p['w_mm'], p['h_mm']), QPageSize.Unit.Millimeter)
        writer.setPageLayout(QPageLayout(size, QPageLayout.Orientation.Portrait,
                                         QMarginsF(0, 0, 0, 0), QPageLayout.Unit.Millimeter))

    set_page(pages[0])
    painter = QPainter(writer)
    px_per_mm = PDF_RESOLUTION / wc.MM_PER_IN
    try:
        for n, page in enumerate(pages):
            if n:
                set_page(page)
                writer.newPage()
            p = scene.pages[page]
            painter.save()
            painter.scale(px_per_mm, px_per_mm)
            painter.setClipRect(QRectF(0.0, 0.0, p['w_mm'], p['h_mm']))
            entries = page_entries(scene, page)
            for i, (item, x, y, w, h) in enumerate(entries):
                svg = figure_svg(item.figure(), False, dpi, salt=f'{item.uid}-{i}')
                renderer = QSvgRenderer(QByteArray(svg.encode('utf-8')))
                renderer.render(painter, QRectF(x, y, w, h))
            painter.restore()
            _restore_screen_render(entries)
    finally:
        painter.end()
    return path


# ----------------------------------------------------------------------
# PNG / TIFF
# ----------------------------------------------------------------------

def render_entries(entries: list[tuple], w_mm: float, h_mm: float, dpi: float) -> np.ndarray:
    """Composite the entries' figures at *dpi* onto a white w x h mm canvas
    (h x w x 3, uint8), clipping anything outside it."""
    from PIL import Image
    px = dpi / wc.MM_PER_IN
    canvas = Image.new('RGBA', (max(1, round(w_mm * px)), max(1, round(h_mm * px))), 'white')
    for item, x, y, _w, _h in entries:
        tile = Image.fromarray(figure_rgba(item.figure(), dpi), 'RGBA')
        left, top = round(x * px), round(y * px)
        # alpha_composite takes no negative offsets: crop what lies off the page
        crop_l, crop_t = max(0, -left), max(0, -top)
        right = min(tile.width, canvas.width - left)
        bottom = min(tile.height, canvas.height - top)
        if right <= crop_l or bottom <= crop_t:
            continue
        canvas.alpha_composite(tile.crop((crop_l, crop_t, right, bottom)),
                               dest=(left + crop_l, top + crop_t))
    _restore_screen_render(entries)
    return np.asarray(canvas.convert('RGB'))


def save_raster(array: np.ndarray, path: Path, fmt: str, dpi: float):
    from PIL import Image
    image = Image.fromarray(array, 'RGB')
    if fmt == 'TIFF':
        image.save(path, format='TIFF', dpi=(dpi, dpi), compression='tiff_lzw')
    else:
        image.save(path, format='PNG', dpi=(dpi, dpi))


def export_raster(path_stem: Path, scene, pages: list[int], fmt: str,
                  dpi: float = DEFAULT_EXPORT_DPI) -> list[Path]:
    """One PNG or TIFF per page: <stem>.<ext> for one page, else
    <stem>_page<n>.<ext>."""
    ext = '.tif' if fmt == 'TIFF' else '.png'
    written = []
    for page in pages:
        p = scene.pages[page]
        array = render_entries(page_entries(scene, page), p['w_mm'], p['h_mm'], dpi)
        path = _page_path(path_stem, page, len(pages), ext)
        save_raster(array, path, fmt, dpi)
        written.append(path)
    return written


def page_figure(scene, page: int, dpi: float = DEFAULT_EXPORT_DPI):
    """A matplotlib Figure of the page at its size, holding the page
    rendered at *dpi* (for the Report's combined PDF and PNGs)."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    p = scene.pages[page]
    array = render_entries(page_entries(scene, page), p['w_mm'], p['h_mm'], dpi)
    fig = Figure(figsize=(p['w_mm'] / wc.MM_PER_IN, p['h_mm'] / wc.MM_PER_IN), dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.imshow(array, interpolation='none')
    ax.set_axis_off()
    return fig


def _page_path(stem: Path, page: int, n_pages: int, ext: str) -> Path:
    stem = Path(stem)
    if stem.suffix.lower() == ext:
        stem = stem.with_suffix('')
    if n_pages == 1:
        return stem.with_name(stem.name + ext)
    return stem.with_name(f"{stem.name}_page{page + 1}{ext}")
