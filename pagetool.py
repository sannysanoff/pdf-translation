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


def disassemble(png_path):
    """Disassemble PNG into directory with Docling data, texts, and images."""
    if not os.path.exists(png_path):
        raise FileNotFoundError(f"File {png_path} not found")

    converter = DocumentConverter()
    result = converter.convert(png_path)
    doc = result.document

    base_name = os.path.splitext(os.path.basename(png_path))[0]
    output_dir = os.path.join(os.path.dirname(png_path), base_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save the Docling document as JSON
    with open(os.path.join(output_dir, 'docling.json'), 'w') as f:
        json.dump(doc.model_dump(), f, indent=2)

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


def assemble(dir_path, *, debug: bool = False, base_image: str = "blank", font_family: str = "times"):
    """Assemble directory back into PNG."""
    if not os.path.isdir(dir_path):
        raise NotADirectoryError(f"Directory {dir_path} not found")

    # Load Docling data for page size
    with open(os.path.join(dir_path, 'docling.json'), 'r') as f:
        doc_data = json.load(f)

    if not doc_data.get('pages'):
        raise ValueError("No pages found in docling.json")

    # Get the first page's size (pages is a dict keyed by page number)
    first_page = next(iter(doc_data['pages'].values()))
    page_size = first_page['size']
    width = int(page_size['width'])
    height = int(page_size['height'])

    # Create output image
    if base_image == "page":
        page_path = os.path.join(dir_path, "page.png")
        if os.path.exists(page_path):
            img = Image.open(page_path).convert("RGB")
            width, height = img.size
        else:
            img = Image.new("RGB", (width, height), "white")
    else:
        img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    regular_font_path, bold_font_path = _pick_font_paths(font_family)

    def convert_bbox(bbox, coord_origin='BOTTOMLEFT'):
        # Convert from BOTTOMLEFT to TOPLEFT coordinates
        if coord_origin == 'BOTTOMLEFT':
            return {
                'l': bbox['l'],
                't': height - bbox['t'],
                'r': bbox['r'],
                'b': height - bbox['b']
            }
        else:  # TOPLEFT coordinates, no conversion needed
            return bbox

    # Load and draw texts (fit to each bbox)
    texts_path = os.path.join(dir_path, 'texts.json')
    if os.path.exists(texts_path):
        with open(texts_path, 'r') as f:
            texts = json.load(f)
        for text in texts:
            bbox = convert_bbox(text['bbox'])
            box = _to_int_box(bbox, width, height)
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

    # Load and paste pictures
    pictures_path = os.path.join(dir_path, 'pictures.json')
    if os.path.exists(pictures_path):
        with open(pictures_path, 'r') as f:
            pictures = json.load(f)
        for pic in pictures:
            img_file = os.path.join(dir_path, pic['img_file'])
            if os.path.exists(img_file):
                pic_img = Image.open(img_file)
                bbox = convert_bbox(pic['bbox'])
                # Resize to fit bbox
                pic_img = pic_img.resize((int(bbox['r'] - bbox['l']), int(bbox['b'] - bbox['t'])))
                img.paste(pic_img, (int(bbox['l']), int(bbox['t'])))

    # Load and draw tables
    tables_path = os.path.join(dir_path, 'tables.json')
    if os.path.exists(tables_path):
        with open(tables_path, 'r') as f:
            tables = json.load(f)
        for table in tables:
            # Table bbox uses BOTTOMLEFT coordinates
            table_bbox = convert_bbox(table['bbox'], 'BOTTOMLEFT')
            table_box = _to_int_box(table_bbox, width, height)
            if debug:
                draw.rectangle([table_box.l, table_box.t, table_box.r, table_box.b], outline="blue", width=3)

            # Pick a single font size for the whole table that fits all cells
            cell_boxes: list[tuple[Box, dict]] = []
            min_text_cell_h = None
            for cell in table["table_cells"]:
                cell_bbox = convert_bbox(cell["bbox"], "TOPLEFT")
                cell_box = _to_int_box(cell_bbox, width, height)
                cell_boxes.append((cell_box, cell))
                txt = (cell.get("text") or "").strip()
                if txt:
                    if min_text_cell_h is None or cell_box.height < min_text_cell_h:
                        min_text_cell_h = cell_box.height

            min_text_cell_h = min_text_cell_h or 1
            table_max_size = max(8, min(56, int(round(min_text_cell_h * 0.9))))

            def table_fits(size: int) -> bool:
                font = _load_font(regular_font_path, size)
                spacing = max(0, int(round(size * 0.05)))
                for cell_box, cell in cell_boxes:
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
            
            # Draw table cells
            for cell_box, cell in cell_boxes:
                # Render a light grid to preserve table structure (Docling often provides cell bboxes).
                draw.rectangle([cell_box.l, cell_box.t, cell_box.r, cell_box.b], outline="#c9c9c9", width=1)
                if debug:
                    draw.rectangle([cell_box.l, cell_box.t, cell_box.r, cell_box.b], outline="lightblue", width=1)

                txt = (cell.get("text") or "").strip()
                if not txt:
                    continue

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

    def inset(self, pad_x: int, pad_y: int) -> "Box":
        return Box(self.l + pad_x, self.t + pad_y, self.r - pad_x, self.b - pad_y)


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


def main():
    parser = argparse.ArgumentParser(description="Page tool for disassembling and assembling images using Docling")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Disassemble command
    disassemble_parser = subparsers.add_parser('disassemble', help='Disassemble PNG into components')
    disassemble_parser.add_argument('png_path', help='Path to the PNG file')

    # Assemble command
    assemble_parser = subparsers.add_parser('assemble', help='Assemble directory back into PNG')
    assemble_parser.add_argument('dir_path', help='Path to the directory containing disassembled components')
    assemble_parser.add_argument('--debug', action='store_true', help='Draw debug bounding boxes')
    assemble_parser.add_argument('--base-image', choices=['blank', 'page'], default='blank', help='Start from blank or original page.png')
    assemble_parser.add_argument('--font', choices=['times', 'georgia', 'arial'], default='times', help='Font family for rendered text')

    args = parser.parse_args()

    if args.command == 'disassemble':
        disassemble(args.png_path)
    elif args.command == 'assemble':
        assemble(args.dir_path, debug=args.debug, base_image=args.base_image, font_family=args.font)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
