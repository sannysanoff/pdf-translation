#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2  # type: ignore
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class EraseResult:
    rgba: np.ndarray  # uint8 HxWx4
    mask: np.ndarray  # uint8 HxW (0/255)
    background_rgb: tuple[int, int, int]
    text_rgb: tuple[int, int, int] | None


def _load_rgba(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        rgba = np.array(im.convert("RGBA"), dtype=np.uint8)
    return rgba


def _save_rgba(path: Path, rgba: np.ndarray) -> None:
    Image.fromarray(rgba).save(path)


def _quantized_mode_rgb(rgb: np.ndarray, alpha: np.ndarray, shift_bits: int = 3) -> tuple[int, int, int]:
    mask = alpha > 0
    if not np.any(mask):
        return (255, 255, 255)
    px = rgb[mask]
    q = (px >> shift_bits).astype(np.uint16)
    packed = (q[:, 0] << 10) | (q[:, 1] << 5) | q[:, 2]
    counts = np.bincount(packed, minlength=1 << 15)
    idx = int(np.argmax(counts))
    r = ((idx >> 10) & 31) << shift_bits
    g = ((idx >> 5) & 31) << shift_bits
    b = (idx & 31) << shift_bits
    return (int(r), int(g), int(b))


def _luminance_u8(rgb: np.ndarray) -> np.ndarray:
    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return np.clip(y, 0, 255).astype(np.uint8)


def _estimate_text_color(rgb: np.ndarray, mask: np.ndarray) -> tuple[int, int, int] | None:
    m = mask > 0
    if not np.any(m):
        return None
    px = rgb[m].reshape(-1, 3).astype(np.uint8)
    med = np.median(px, axis=0)
    return (int(med[0]), int(med[1]), int(med[2]))


def _filter_components(
    mask: np.ndarray,
    *,
    min_area: int = 8,
    max_area_ratio: float = 0.85,
    border_margin: int = 0,
) -> np.ndarray:
    h, w = mask.shape
    num, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(num, dtype=bool)
    keep[0] = False

    img_area = float(h * w)
    for i in range(1, num):
        x, y, bw, bh, area = stats[i]
        if area < min_area:
            continue
        if area > img_area * max_area_ratio:
            continue
        if border_margin > 0:
            if x <= border_margin or y <= border_margin or (x + bw) >= (w - border_margin) or (y + bh) >= (h - border_margin):
                continue
        if bh >= int(h * 0.8) or bw >= int(w * 0.95):
            continue
        bbox_area = float(bw * bh)
        fill = (float(area) / bbox_area) if bbox_area > 0 else 0.0
        # Heuristic: suppress large low-fill outlines (borders/frames).
        if (bw >= int(0.55 * w) and bh >= int(0.40 * h) and fill < 0.22) or (bw >= int(0.75 * w) and fill < 0.25):
            continue
        # Heuristic: suppress long thin frame lines (common in callouts/borders).
        if (bh <= 6 and bw >= int(0.6 * w)) or (bw <= 6 and bh >= int(0.6 * h)):
            continue
        keep[i] = True

    out = np.zeros_like(mask)
    out[keep[labels]] = 255
    return out


def _build_text_mask(rgba: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int], tuple[int, int, int] | None]:
    rgb = rgba[:, :, :3]
    alpha = rgba[:, :, 3]
    bg = _quantized_mode_rgb(rgb, alpha)

    h, w = alpha.shape
    valid = alpha > 0
    if int(valid.sum()) < 200:
        return np.zeros((h, w), dtype=np.uint8), bg, None

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Strategy:
    # - estimate local background by median blur
    # - compute delta-to-background (dark and light)
    # - threshold delta with Otsu, then expand mask to cover anti-aliasing
    k = max(15, ((min(h, w) // 6) | 1))
    k = int(min(k, 61))
    if k % 2 == 0:
        k += 1
    bg_gray = cv2.cvtColor(cv2.medianBlur(bgr, k), cv2.COLOR_BGR2GRAY)

    dark_delta = cv2.subtract(bg_gray, img_gray)
    light_delta = cv2.subtract(img_gray, bg_gray)
    dark_delta[~valid] = 0
    light_delta[~valid] = 0

    def _delta_mask(delta: np.ndarray, *, min_thr: int) -> np.ndarray:
        if int(delta.max()) < min_thr:
            return np.zeros((h, w), dtype=np.uint8)
        thr, _ = cv2.threshold(delta, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr_u8 = max(min_thr, int(thr))
        m = ((delta >= thr_u8) & valid).astype(np.uint8) * 255
        m = _filter_components(m, min_area=10, max_area_ratio=0.98, border_margin=0)
        if np.any(m):
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        return m

    m_dark = _delta_mask(dark_delta, min_thr=8)
    m_light = _delta_mask(light_delta, min_thr=8)

    dark_area = int((m_dark > 0).sum())
    light_area = int((m_light > 0).sum())

    if dark_area and not light_area:
        mask = m_dark
        delta_for_color = dark_delta
    elif light_area and not dark_area:
        mask = m_light
        delta_for_color = light_delta
    elif dark_area and light_area:
        dark_strength = float(np.median(dark_delta[m_dark > 0])) if dark_area else 0.0
        light_strength = float(np.median(light_delta[m_light > 0])) if light_area else 0.0
        if dark_strength >= light_strength:
            mask = m_dark
            delta_for_color = dark_delta
        else:
            mask = m_light
            delta_for_color = light_delta
    else:
        return np.zeros((h, w), dtype=np.uint8), bg, None

    # Expand to cover anti-aliasing around detected strokes.
    if np.any(mask):
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
        mask = _filter_components(mask, min_area=10, max_area_ratio=0.995, border_margin=0)

    # Estimate text color from strongest deltas inside mask (more likely core stroke).
    text_rgb = None
    if np.any(mask):
        vals = delta_for_color[mask > 0]
        if vals.size >= 50:
            cut = float(np.percentile(vals, 80))
            core_sel = (mask > 0) & (delta_for_color >= cut)
            if int(core_sel.sum()) >= 50:
                text_rgb = _estimate_text_color(rgb, (core_sel.astype(np.uint8) * 255))
    if text_rgb is None:
        text_rgb = _estimate_text_color(rgb, mask)

    return mask, bg, text_rgb


def erase_text_rgba(
    rgba: np.ndarray,
    inpaint_radius: int = 3,
    post_median_ksize: int = 3,
) -> EraseResult:
    if rgba.dtype != np.uint8 or rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("Expected uint8 RGBA image (HxWx4)")

    mask, bg_rgb, text_rgb = _build_text_mask(rgba)

    rgb = rgba[:, :, :3].copy()
    alpha = rgba[:, :, 3].copy()

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if np.any(mask):
        inpainted_bgr = cv2.inpaint(bgr, mask, inpaintRadius=float(inpaint_radius), flags=cv2.INPAINT_TELEA)
        if post_median_ksize and post_median_ksize >= 3 and post_median_ksize % 2 == 1:
            blurred = cv2.medianBlur(inpainted_bgr, post_median_ksize)
            inpainted_bgr = np.where(mask[:, :, None] > 0, blurred, inpainted_bgr)
    else:
        inpainted_bgr = bgr

    out_rgb = cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB)
    out = np.dstack([out_rgb, alpha]).astype(np.uint8)
    return EraseResult(rgba=out, mask=mask, background_rgb=bg_rgb, text_rgb=text_rgb)


def _output_path(input_path: Path) -> Path:
    suffix = input_path.suffix if input_path.suffix else ".png"
    stem = input_path.name[: -len(suffix)] if input_path.name.endswith(suffix) else input_path.name
    return input_path.with_name(f"{stem}-output{suffix}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Erase text from images and write *-output.png next to inputs.")
    ap.add_argument("images", nargs="+", help="One or more image paths")
    ap.add_argument("--debug-mask", action="store_true", help="Also write *-mask.png alongside outputs")
    ap.add_argument("--inpaint-radius", type=int, default=3, help="Inpaint radius (default: 3)")
    ap.add_argument("--post-median", type=int, default=3, help="Median blur ksize applied only to erased region (odd >=3, default: 3)")
    args = ap.parse_args(argv)

    ok = True
    for p_str in args.images:
        p = Path(p_str)
        if not p.exists():
            print(f"[text-erasure] missing: {p}", file=sys.stderr)
            ok = False
            continue
        if p.is_dir():
            print(f"[text-erasure] is a directory (skip): {p}", file=sys.stderr)
            ok = False
            continue
        if p.stat().st_size == 0:
            print(f"[text-erasure] empty file (skip): {p}", file=sys.stderr)
            ok = False
            continue

        try:
            rgba = _load_rgba(p)
        except Exception as e:
            print(f"[text-erasure] failed to read {p}: {e}", file=sys.stderr)
            ok = False
            continue

        res = erase_text_rgba(rgba, inpaint_radius=args.inpaint_radius, post_median_ksize=args.post_median)
        out_path = _output_path(p)
        _save_rgba(out_path, res.rgba)

        if args.debug_mask:
            mask_path = p.with_name(f"{p.stem}-mask.png")
            Image.fromarray(res.mask).save(mask_path)

        if res.text_rgb is not None:
            print(f"[text-erasure] {p} -> {out_path} (text_rgb={res.text_rgb}, bg_rgb={res.background_rgb})")
        else:
            print(f"[text-erasure] {p} -> {out_path} (no text detected, bg_rgb={res.background_rgb})")

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
