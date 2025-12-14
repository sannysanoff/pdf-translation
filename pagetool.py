#!/usr/bin/env python3
"""
Page tool using Docling to disassemble and assemble images with text and picture coordinates.
"""

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from functools import lru_cache
from PIL import Image, ImageDraw, ImageFont
from docling.document_converter import DocumentConverter
from claude_text_removal import remove_text_pil

CLEANUP_PAD_PX = 3





def _resolve_disassemble_input(path: str) -> tuple[str, str]:
    """
    Resolve a disassemble input to an image path and an output base name.

    Supports:
    - direct .png files
    - stems (foo -> foo.png)
    - directories (dir -> dir.png if present, else dir/page.png)
    """
    if os.path.isdir(path):
        as_png = path.rstrip("/\\") + ".png"
        if os.path.exists(as_png):
            return as_png, os.path.splitext(os.path.basename(as_png))[0]
        page_png = os.path.join(path, "page.png")
        if os.path.exists(page_png):
            # Use directory name as the output folder base name, not "page".
            return page_png, os.path.basename(os.path.normpath(path))
        raise FileNotFoundError(f"Directory {path} does not contain {as_png} or page.png")

    if os.path.exists(path) and path.lower().endswith(".png"):
        return path, os.path.splitext(os.path.basename(path))[0]

    as_png = path + ".png"
    if os.path.exists(as_png):
        return as_png, os.path.splitext(os.path.basename(as_png))[0]

    raise FileNotFoundError(f"File {path} not found (also tried {as_png})")


def disassemble(png_path: str, *, converter: DocumentConverter | None = None, output_base_name: str | None = None):
    """Disassemble PNG into directory with Docling data, texts, and images."""
    if not os.path.exists(png_path):
        raise FileNotFoundError(f"File {png_path} not found")

    converter = converter or DocumentConverter()
    result = converter.convert(png_path)
    doc = result.document

    base_name = output_base_name or os.path.splitext(os.path.basename(png_path))[0]
    output_dir = os.path.join(os.path.dirname(png_path), base_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save the Docling document as JSON and record the coordinate origin if available.
    doc_dict = doc.model_dump()
    # Docling historically uses a bottom-left origin for PDF coordinates, but some
    # inputs (e.g., PNGs) may already be in top-left pixel space. Persist any
    # origin hint so assembly can mirror the exact convention.
    coord_origin = getattr(doc, "coord_origin", None) or doc_dict.get("coord_origin")
    if coord_origin:
        doc_dict["coord_origin"] = str(coord_origin)

    with open(os.path.join(output_dir, 'docling.json'), 'w') as f:
        json.dump(doc_dict, f, indent=2)

    # Save the original page image
    page_img = Image.open(png_path)
    page_img.save(os.path.join(output_dir, 'page.png'))

    # Extract texts with bboxes
    texts = []
    for text_item in doc.texts:
        prov = text_item.prov[0] if text_item.prov else None
        if prov and hasattr(prov, 'bbox'):
            bbox = prov.bbox
            texts.append({
                'text': text_item.text,
                'bbox': {'l': bbox.l, 't': bbox.t, 'r': bbox.r, 'b': bbox.b}
            })
    with open(os.path.join(output_dir, 'texts.json'), 'w') as f:
        json.dump(texts, f, indent=2)

    # Extract pictures with bboxes and save images
    pictures = []
    for i, pic in enumerate(doc.pictures):
        prov = pic.prov[0] if pic.prov else None
        if prov and hasattr(prov, 'bbox'):
            bbox = prov.bbox
            pictures.append({
                'bbox': {'l': bbox.l, 't': bbox.t, 'r': bbox.r, 'b': bbox.b},
                'img_file': f'img_{i}.png'
            })
            # Save the extracted image if available
            if pic.image and pic.image.uri and os.path.exists(pic.image.uri):
                shutil.copy(pic.image.uri, os.path.join(output_dir, f'img_{i}.png'))
    with open(os.path.join(output_dir, 'pictures.json'), 'w') as f:
        json.dump(pictures, f, indent=2)

    # Extract tables with bboxes and cell data
    tables = []
    for i, table in enumerate(doc.tables):
        prov = table.prov[0] if table.prov else None
        if prov and hasattr(prov, 'bbox'):
            bbox = prov.bbox
            table_data = {
                'bbox': {'l': bbox.l, 't': bbox.t, 'r': bbox.r, 'b': bbox.b},
                'table_cells': []
            }
            
            # Extract table cells
            if hasattr(table, 'data') and table.data and hasattr(table.data, 'table_cells'):
                for cell in table.data.table_cells:
                    cell_bbox = cell.bbox
                    table_data['table_cells'].append({
                        'bbox': {'l': cell_bbox.l, 't': cell_bbox.t, 'r': cell_bbox.r, 'b': cell_bbox.b},
                        'text': cell.text,
                        'row_span': cell.row_span,
                        'col_span': cell.col_span,
                        'column_header': cell.column_header,
                        'row_header': cell.row_header
                    })
            
            tables.append(table_data)
    
    with open(os.path.join(output_dir, 'tables.json'), 'w') as f:
        json.dump(tables, f, indent=2)

    print(f"Disassembled {png_path} into {output_dir}/")


def assemble(
    dir_path,
    *,
    debug: bool = False,
    base_image: str = "page",
    font_family: str = "times",
):
    """Assemble directory back into PNG."""
    if not os.path.isdir(dir_path):
        raise NotADirectoryError(f"Directory {dir_path} not found")

    temp_counter = 1

    # Load Docling data for page size
    with open(os.path.join(dir_path, 'docling.json'), 'r') as f:
        doc_data = json.load(f)

    coord_origin = str(doc_data.get("coord_origin") or "BOTTOMLEFT").upper()

    if not doc_data.get('pages'):
        raise ValueError("No pages found in docling.json")

    # Get the first page's size (pages is a dict keyed by page number)
    first_page = next(iter(doc_data['pages'].values()))
    page_size = first_page['size']
    width = int(page_size['width'])
    height = int(page_size['height'])

    # Start from original page when available so pictures/background are preserved.
    page_path = os.path.join(dir_path, "page.png")
    if base_image == "page" and os.path.exists(page_path):
        img = Image.open(page_path).convert("RGB")
        width, height = img.size
    else:
        img = Image.new("RGB", (width, height), "white")

    draw = ImageDraw.Draw(img)

    regular_font_path, bold_font_path = _pick_font_paths(font_family)

    def convert_bbox(bbox, coord_origin=coord_origin):
        # Convert from BOTTOMLEFT to TOPLEFT coordinates
        if str(coord_origin).upper() == 'BOTTOMLEFT':
            return {
                'l': bbox['l'],
                't': height - bbox['t'],
                'r': bbox['r'],
                'b': height - bbox['b']
            }
        else:  # TOPLEFT coordinates, no conversion needed
            return bbox

    # Load pictures and texts
    pictures = []
    pictures_path = os.path.join(dir_path, "pictures.json")
    if os.path.exists(pictures_path):
        with open(pictures_path, "r") as f:
            pictures = json.load(f)

    texts = []
    texts_path = os.path.join(dir_path, 'texts.json')
    if os.path.exists(texts_path):
        with open(texts_path, 'r') as f:
            texts = json.load(f)

    # When assembling from a blank canvas, paste extracted pictures (if present).
    if base_image == "blank":
        for pic in pictures:
            img_file = os.path.join(dir_path, pic["img_file"])
            if os.path.exists(img_file):
                pic_img = Image.open(img_file).convert("RGB")
                bbox = convert_bbox(pic["bbox"])
                box = _to_int_box(bbox, width, height)
                if box.area <= 0:
                    continue
                pic_img = pic_img.resize((box.width, box.height))
                img.paste(pic_img, (box.l, box.t))

    # Pre-load tables to suppress text rendering in table areas
    table_boxes: list[Box] = []
    tables_path = os.path.join(dir_path, 'tables.json')
    if os.path.exists(tables_path):
        with open(tables_path, 'r') as f:
            _tables_for_suppression = json.load(f)
        for table in _tables_for_suppression:
            table_bbox = convert_bbox(table['bbox'], 'BOTTOMLEFT')
            table_boxes.append(_to_int_box(table_bbox, width, height))

    def should_skip_text_box(box: Box) -> bool:
        if not table_boxes or box.area <= 0:
            return False
        for tb in table_boxes:
            inter = box.intersect(tb)
            if inter.area <= 0:
                continue
            # If most of the text box is inside a table, do not render it here.
            if inter.area / box.area >= 0.35:
                return True
        return False



    # Load and draw tables
    tables_for_render: list[dict] = []
    if os.path.exists(tables_path):
        with open(tables_path, 'r') as f:
            tables = json.load(f)
        for table in tables:
            # Table bbox uses BOTTOMLEFT coordinates
            table_bbox = convert_bbox(table['bbox'], 'BOTTOMLEFT')
            table_box = _to_int_box(table_bbox, width, height)
            if debug:
                draw.rectangle([table_box.l, table_box.t, table_box.r, table_box.b], outline="blue", width=3)

            # Prepare and clean cells:
            # - Clip to table bbox
            # - Drop "container" cells that fully contain multiple other cells (Docling sometimes emits these)
            raw_cells: list[tuple[Box, dict]] = []
            for cell in table["table_cells"]:
                cell_bbox = convert_bbox(cell["bbox"], "TOPLEFT")
                cell_box = _to_int_box(cell_bbox, width, height).intersect(table_box)
                if cell_box.area <= 0:
                    continue
                raw_cells.append((cell_box, cell))

            def contains_ratio(outer: Box, inner: Box) -> float:
                if inner.area <= 0:
                    return 0.0
                return outer.intersect(inner).area / inner.area

            container_idxs: set[int] = set()
            for i, (box_i, cell_i) in enumerate(raw_cells):
                txt_i = (cell_i.get("text") or "").strip()
                if not txt_i or box_i.area <= 0:
                    continue
                contained: list[int] = []
                for j, (box_j, cell_j) in enumerate(raw_cells):
                    if i == j:
                        continue
                    txt_j = (cell_j.get("text") or "").strip()
                    if not txt_j or box_j.area <= 0:
                        continue
                    if contains_ratio(box_i, box_j) >= 0.95:
                        contained.append(j)
                if len(contained) >= 2:
                    container_idxs.add(i)

            cells: list[tuple[Box, dict]] = [(b, c) for idx, (b, c) in enumerate(raw_cells) if idx not in container_idxs]

            # Build a grid from the remaining cells and draw table lines.
            x_edges = [table_box.l, table_box.r]
            y_edges = [table_box.t, table_box.b]
            for cb, _ in cells:
                x_edges.extend([cb.l, cb.r])
                y_edges.extend([cb.t, cb.b])

            x_grid = _cluster_positions(x_edges, tol=3)
            y_grid = _cluster_positions(y_edges, tol=3)

            # Ensure table bbox edges are present.
            if table_box.l not in x_grid:
                x_grid.append(table_box.l)
            if table_box.r not in x_grid:
                x_grid.append(table_box.r)
            if table_box.t not in y_grid:
                y_grid.append(table_box.t)
            if table_box.b not in y_grid:
                y_grid.append(table_box.b)
            x_grid = sorted(set(x_grid))
            y_grid = sorted(set(y_grid))

            # Snap cell boxes to the detected grid (reduces overlaps and misaligned boundaries).
            snapped_cells: list[tuple[Box, dict]] = []
            min_text_cell_h = None
            for cb, cell in cells:
                snapped = Box(
                    _snap_to_positions(cb.l, x_grid),
                    _snap_to_positions(cb.t, y_grid),
                    _snap_to_positions(cb.r, x_grid),
                    _snap_to_positions(cb.b, y_grid),
                ).intersect(table_box)
                if snapped.area <= 0:
                    continue
                snapped_cells.append((snapped, cell))
                txt = (cell.get("text") or "").strip()
                if txt:
                    if min_text_cell_h is None or snapped.height < min_text_cell_h:
                        min_text_cell_h = snapped.height



            min_text_cell_h = min_text_cell_h or 1
            table_max_size = max(8, min(56, int(round(min_text_cell_h * 0.9))))

            tables_for_render.append(
                {
                    "table_box": table_box,
                    "x_grid": x_grid,
                    "y_grid": y_grid,
                    "snapped_cells": snapped_cells,
                    "table_max_size": table_max_size,
                }
            )



    # Render tables (grid + text) after erases so lines are preserved.
    for tdata in tables_for_render:
        table_box: Box = tdata["table_box"]
        x_grid: list[int] = tdata["x_grid"]
        y_grid: list[int] = tdata["y_grid"]
        snapped_cells: list[tuple[Box, dict]] = tdata["snapped_cells"]
        table_max_size: int = tdata["table_max_size"]

        # Draw grid lines (outer border heavier).
        line_color = "#666666"
        for x in x_grid:
            w = 2 if x in (table_box.l, table_box.r) else 1
            draw.line([(x, table_box.t), (x, table_box.b)], fill=line_color, width=w)
        for y in y_grid:
            w = 2 if y in (table_box.t, table_box.b) else 1
            draw.line([(table_box.l, y), (table_box.r, y)], fill=line_color, width=w)

        def table_fits(size: int) -> bool:
            font = _load_font(regular_font_path, size)
            spacing = max(0, int(round(size * 0.05)))
            for cell_box, cell in snapped_cells:
                txt = (cell.get("text") or "").strip()
                if not txt:
                    continue
                pad_x = min(6, max(2, int(round(cell_box.width * 0.02))))
                pad_y = min(6, max(2, int(round(cell_box.height * 0.08))))
                ok, _ = _layout_fits(draw, txt, font, cell_box, pad_x, pad_y, spacing)
                if not ok:
                    return False
            return True

        lo, hi = 8, table_max_size
        best = lo
        while lo <= hi:
            mid = (lo + hi) // 2
            if table_fits(mid):
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1

        table_font_size = best
        table_spacing = max(0, int(round(table_font_size * 0.05)))

        for cell_box, cell in snapped_cells:
            if debug:
                draw.rectangle([cell_box.l, cell_box.t, cell_box.r, cell_box.b], outline="lightblue", width=1)
            txt = (cell.get("text") or "").strip()
            if not txt:
                continue

            # Clean the cell area before drawing text
            expanded_cell_box = _expand_box(cell_box, CLEANUP_PAD_PX, width, height)
            crop = img.crop((expanded_cell_box.l, expanded_cell_box.t, expanded_cell_box.r, expanded_cell_box.b))
            crop.save(os.path.join(dir_path, f'temp-{temp_counter:04d}.png'))
            cleaned_crop = remove_text_pil(crop)
            cleaned_crop.save(os.path.join(dir_path, f'temp-{temp_counter:04d}-cleaned.png'))
            img.paste(cleaned_crop, (expanded_cell_box.l, expanded_cell_box.t))
            temp_counter += 1

            is_header = bool(cell.get("column_header") or cell.get("row_header"))
            font_path = bold_font_path if is_header else regular_font_path
            pad_x = min(6, max(2, int(round(cell_box.width * 0.02))))
            pad_y = min(6, max(2, int(round(cell_box.height * 0.08))))

            _draw_text_in_box(
                draw,
                txt,
                cell_box,
                font_path,
                font_size=table_font_size,
                pad_x=pad_x,
                pad_y=pad_y,
                spacing=table_spacing,
                fill="black",
                align="center" if is_header else "left",
                valign="middle",
            )

    # Clean text regions before drawing new text
    for text in texts:
        bbox = convert_bbox(text["bbox"])
        box = _to_int_box(bbox, width, height)
        if should_skip_text_box(box):
            continue
        # Extract sub-image
        expanded_box = _expand_box(box, CLEANUP_PAD_PX, width, height)
        crop = img.crop((expanded_box.l, expanded_box.t, expanded_box.r, expanded_box.b))
        crop.save(os.path.join(dir_path, f'temp-{temp_counter:04d}.png'))
        # Clean it
        cleaned_crop = remove_text_pil(crop)
        cleaned_crop.save(os.path.join(dir_path, f'temp-{temp_counter:04d}-cleaned.png'))
        # Paste back
        img.paste(cleaned_crop, (expanded_box.l, expanded_box.t))
        temp_counter += 1

    # Draw all (non-table) texts.
    for text in texts:
        bbox = convert_bbox(text["bbox"])
        box = _to_int_box(bbox, width, height)
        if should_skip_text_box(box):
            continue
        if debug:
            draw.rectangle([box.l, box.t, box.r, box.b], outline="red", width=2)

        pad_x = max(2, int(round(box.width * 0.01)))
        pad_y = max(2, int(round(box.height * 0.02)))
        max_size = max(8, min(96, box.height))
        size, _, spacing = _fit_font_size(
            draw,
            text["text"],
            box,
            regular_font_path,
            min_size=8,
            max_size=max_size,
            pad_x=pad_x,
            pad_y=pad_y,
            line_spacing_ratio=0.15,
        )
        _draw_text_in_box(
            draw,
            text["text"],
            box,
            regular_font_path,
            font_size=size,
            pad_x=pad_x,
            pad_y=pad_y,
            spacing=spacing,
            fill="black",
            align="left",
            valign="top",
        )

    # Save the assembled image
    output_path = dir_path.rstrip('/') + '-output.png'
    img.save(output_path)
    print(f"Assembled {dir_path} into {output_path}")
    return output_path


@dataclass(frozen=True)
class Box:
    l: int
    t: int
    r: int
    b: int

    @property
    def width(self) -> int:
        return max(0, self.r - self.l)

    @property
    def height(self) -> int:
        return max(0, self.b - self.t)

    @property
    def area(self) -> int:
        return self.width * self.height

    def inset(self, pad_x: int, pad_y: int) -> "Box":
        return Box(self.l + pad_x, self.t + pad_y, self.r - pad_x, self.b - pad_y)

    def intersect(self, other: "Box") -> "Box":
        return Box(max(self.l, other.l), max(self.t, other.t), min(self.r, other.r), min(self.b, other.b))


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _to_int_box(bbox: dict, width: int, height: int) -> Box:
    l = int(round(bbox["l"]))
    t = int(round(bbox["t"]))
    r = int(round(bbox["r"]))
    b = int(round(bbox["b"]))
    if r < l:
        l, r = r, l
    if b < t:
        t, b = b, t
    l = _clamp(l, 0, width)
    r = _clamp(r, 0, width)
    t = _clamp(t, 0, height)
    b = _clamp(b, 0, height)
    return Box(l, t, r, b)


def _expand_box(box: Box, pad: int, width: int, height: int) -> Box:
    if pad <= 0:
        return box
    return Box(
        _clamp(box.l - pad, 0, width),
        _clamp(box.t - pad, 0, height),
        _clamp(box.r + pad, 0, width),
        _clamp(box.b + pad, 0, height),
    )


def _pick_font_paths(font_family: str | None = None) -> tuple[str | None, str | None]:
    candidates = [
        ("times", "/System/Library/Fonts/Supplemental/Times New Roman.ttf", "/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf"),
        ("georgia", "/System/Library/Fonts/Supplemental/Georgia.ttf", "/System/Library/Fonts/Supplemental/Georgia Bold.ttf"),
        ("arial", "/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    ]

    requested = (font_family or "").strip().lower()
    if requested:
        candidates = [c for c in candidates if c[0] == requested] + [c for c in candidates if c[0] != requested]

    for _, regular, bold in candidates:
        if os.path.exists(regular) and os.path.exists(bold):
            return regular, bold
        if os.path.exists(regular):
            return regular, regular

    return None, None


@lru_cache(maxsize=256)
def _load_font(path: str | None, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if path:
        return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _line_height(font: ImageFont.ImageFont, extra_spacing: int) -> int:
    try:
        ascent, descent = font.getmetrics()
        return ascent + descent + extra_spacing
    except Exception:
        return font.size + extra_spacing  # type: ignore[attr-defined]


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> float:
    if hasattr(draw, "textlength"):
        return float(draw.textlength(text, font=font))
    try:
        return float(font.getlength(text))  # type: ignore[attr-defined]
    except Exception:
        bbox = draw.textbbox((0, 0), text, font=font)
        return float(bbox[2] - bbox[0])


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    if max_width <= 0:
        return [""]

    def break_long_token(token: str) -> list[str]:
        parts: list[str] = []
        cur = ""
        for ch in token:
            trial = cur + ch
            if cur and _text_width(draw, trial, font) > max_width:
                parts.append(cur)
                cur = ch
            else:
                cur = trial
        if cur or not parts:
            parts.append(cur)
        return parts

    lines: list[str] = []
    for para in text.splitlines() or [""]:
        words = para.split()
        if not words:
            lines.append("")
            continue

        current = words[0]
        if _text_width(draw, current, font) > max_width:
            pieces = break_long_token(current)
            lines.extend(pieces[:-1])
            current = pieces[-1]

        for word in words[1:]:
            trial = f"{current} {word}" if current else word
            if _text_width(draw, trial, font) <= max_width:
                current = trial
                continue

            if current:
                lines.append(current)
            current = word
            if _text_width(draw, current, font) > max_width:
                pieces = break_long_token(current)
                lines.extend(pieces[:-1])
                current = pieces[-1]

        lines.append(current)

    return lines


def _layout_fits(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    box: Box,
    pad_x: int,
    pad_y: int,
    line_spacing: int,
) -> tuple[bool, list[str]]:
    inner = box.inset(pad_x, pad_y)
    if inner.width <= 0 or inner.height <= 0:
        return False, []

    lines = _wrap_text(draw, text, font, inner.width)
    lh = _line_height(font, line_spacing)
    total_h = lh * len(lines) - line_spacing if lines else 0
    if total_h > inner.height:
        return False, lines

    for ln in lines:
        if _text_width(draw, ln, font) > inner.width + 0.5:
            return False, lines

    return True, lines


def _fit_font_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: Box,
    regular_path: str | None,
    min_size: int,
    max_size: int,
    pad_x: int,
    pad_y: int,
    line_spacing_ratio: float,
) -> tuple[int, list[str], int]:
    if not text.strip():
        return min_size, [""], 0

    max_size = min(max_size, max(1, box.height))
    best_size = min_size
    best_lines: list[str] = [text]
    best_spacing = 0

    lo, hi = min_size, max_size
    while lo <= hi:
        mid = (lo + hi) // 2
        font = _load_font(regular_path, mid)
        spacing = max(0, int(round(mid * line_spacing_ratio)))
        ok, lines = _layout_fits(draw, text, font, box, pad_x, pad_y, spacing)
        if ok:
            best_size, best_lines, best_spacing = mid, lines, spacing
            lo = mid + 1
        else:
            hi = mid - 1

    return best_size, best_lines, best_spacing


def _draw_text_in_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    box: Box,
    font_path: str | None,
    font_size: int,
    pad_x: int,
    pad_y: int,
    spacing: int,
    fill: str,
    align: str,
    valign: str,
) -> None:
    font = _load_font(font_path, font_size)
    inner = box.inset(pad_x, pad_y)
    lines = _wrap_text(draw, text, font, inner.width)
    lh = _line_height(font, spacing)
    total_h = lh * len(lines) - spacing if lines else 0
    if valign == "middle":
        y = inner.t + max(0, (inner.height - total_h) // 2)
    elif valign == "bottom":
        y = inner.b - total_h
    else:
        y = inner.t

    for ln in lines:
        w = _text_width(draw, ln, font)
        if align == "center":
            x = inner.l + max(0, int(round((inner.width - w) / 2)))
        elif align == "right":
            x = inner.r - int(round(w))
        else:
            x = inner.l
        draw.text((x, y), ln, fill=fill, font=font)
        y += lh


def _cluster_positions(values: list[int], tol: int) -> list[int]:
    if not values:
        return []
    values = sorted(values)
    clusters: list[list[int]] = [[values[0]]]
    for v in values[1:]:
        if abs(v - clusters[-1][-1]) <= tol:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    centers = [int(round(sum(c) / len(c))) for c in clusters]
    return sorted(set(centers))


def _snap_to_positions(v: int, positions: list[int]) -> int:
    if not positions:
        return v
    return min(positions, key=lambda p: abs(p - v))





def main():
    parser = argparse.ArgumentParser(description="Page tool for disassembling and assembling images using Docling")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Disassemble command
    disassemble_parser = subparsers.add_parser('disassemble', help='Disassemble PNG into components')
    disassemble_parser.add_argument('inputs', nargs='+', help='PNG file(s), stems, or directories')

    # Assemble command
    assemble_parser = subparsers.add_parser('assemble', help='Assemble directory back into PNG')
    assemble_parser.add_argument('dir_path', help='Path to the directory containing disassembled components')
    assemble_parser.add_argument('--debug', action='store_true', help='Draw debug bounding boxes')
    assemble_parser.add_argument('--base-image', choices=['blank', 'page'], default='page', help='Start from blank or original page.png')
    assemble_parser.add_argument('--font', choices=['times', 'georgia', 'arial'], default='times', help='Font family for rendered text')

    args = parser.parse_args()

    if args.command == 'disassemble':
        # Reuse a single converter instance to avoid repeated OCR/tesseract startup costs.
        converter = DocumentConverter()
        for inp in args.inputs:
            png_path, base_name = _resolve_disassemble_input(inp)
            disassemble(png_path, converter=converter, output_base_name=base_name)
    elif args.command == 'assemble':
        assemble(
            args.dir_path,
            debug=args.debug,
            base_image=args.base_image,
            font_family=args.font,
        )
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
