#!/usr/bin/env python3
"""
Text removal from PNG images using advanced inpainting techniques.
Removes text while preserving background colors and patterns.

This module can be used both as a CLI tool and imported as a library:

CLI usage:
    python claude_text_removal.py --input image.png --output result.png

Library usage:
    from claude_text_removal import remove_text_pil, remove_text_from_image
    from PIL import Image

    # With PIL Image
    img = Image.open('input.png')
    result_img = remove_text_pil(img)
    result_img.save('output.png')

    # With file paths
    remove_text_from_image('input.png', 'output.png')
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import sys
from typing import Union

try:
    from PIL import Image
except ImportError:
    Image = None


def detect_text_mask(image):
    """
    Detect text regions in the image and create a mask.

    Args:
        image: Input image (BGR format)

    Returns:
        Binary mask where text regions are white (255)
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate image dimensions and area
    img_height, img_width = image.shape[:2]
    img_area = img_height * img_width

    # Get image statistics to determine background
    mean_intensity = np.mean(gray)

    # Create multiple masks using different techniques
    masks = []

    # Method 1: Stroke Width Transform approximation using morphological operations
    # Detect dark text on light background
    if mean_intensity > 127:
        # Light background - look for dark text
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        masks.append(binary)

        # Also try adaptive threshold
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 15, 8)
        masks.append(adaptive)
    else:
        # Dark background - look for light text
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks.append(binary)

    # Method 2: Color-based detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Detect black/dark text (most common)
    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 100])
    dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
    masks.append(dark_mask)

    # Detect any saturated colored text (any hue with high saturation)
    # This catches blue, red, green, yellow, cyan, magenta text, etc.
    # High saturation with reasonable brightness indicates colored text
    lower_colored = np.array([0, 100, 50])  # Any hue, high saturation, decent brightness
    upper_colored = np.array([180, 255, 255])
    colored_mask = cv2.inRange(hsv, lower_colored, upper_colored)

    # Filter colored mask to only keep text-sized regions
    # Remove large colored regions that are likely decorative elements
    colored_contours, _ = cv2.findContours(colored_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_colored_mask = np.zeros_like(colored_mask)
    for contour in colored_contours:
        area = cv2.contourArea(contour)
        # Only keep small colored regions (text), not large shapes
        if area < img_area * 0.05:  # Less than 5% of image
            cv2.drawContours(filtered_colored_mask, [contour], -1, 255, -1)

    masks.append(filtered_colored_mask)

    # Method 3: Variance-based detection (text has high local variance)
    # Calculate local variance using a sliding window
    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)

    gray_float = gray.astype(np.float32)
    mean_local = cv2.filter2D(gray_float, -1, kernel)
    mean_sq_local = cv2.filter2D(gray_float**2, -1, kernel)
    variance = mean_sq_local - mean_local**2

    # Normalize variance
    variance = cv2.normalize(variance, None, 0, 255, cv2.NORM_MINMAX)
    variance = variance.astype(np.uint8)

    # Threshold high variance regions
    _, variance_mask = cv2.threshold(variance, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masks.append(variance_mask)

    # Combine all masks
    combined_mask = np.zeros_like(gray)
    for mask in masks:
        combined_mask = cv2.bitwise_or(combined_mask, mask)

    # Morphological operations to clean up
    # Remove small noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open)

    # Close small gaps in text
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close)

    # Filter contours by characteristics typical of text
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create refined mask with only text-like regions
    refined_mask = np.zeros_like(combined_mask)

    text_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)

        # Skip very small specks
        if area < 20:
            continue

        # Skip very large regions (probably not text)
        if area > img_area * 0.6:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Skip regions that are too large relative to image
        # Use 0.95 threshold to allow text that spans most of small images
        if w > img_width * 0.95 and h > img_height * 0.95:
            continue

        # Check aspect ratio (text characters/words have reasonable ratios)
        if w > 0 and h > 0:
            aspect_ratio = max(w, h) / min(w, h)

            # Individual characters: roughly square to moderately elongated
            # Text lines: can be quite wide
            if aspect_ratio > 30:  # Too extreme
                continue

        # Check if region has sufficient density (text should have reasonable fill)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            if solidity < 0.2:  # Too sparse, probably not text
                continue

        text_contours.append(contour)

    # Draw all text contours
    cv2.drawContours(refined_mask, text_contours, -1, 255, -1)

    # Final dilation to ensure we cover text edges and any remaining artifacts
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    refined_mask = cv2.dilate(refined_mask, kernel_dilate, iterations=3)

    return refined_mask


def inpaint_text_regions(image, mask):
    """
    Remove text by inpainting the masked regions.

    Args:
        image: Input image (BGR format)
        mask: Binary mask of text regions

    Returns:
        Image with text removed
    """
    # Use Navier-Stokes based inpainting (better for texture preservation)
    result = cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_NS)

    # Also try Telea algorithm and blend
    result_telea = cv2.inpaint(image, mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    # Blend both results for better quality
    result = cv2.addWeighted(result, 0.6, result_telea, 0.4, 0)

    return result


def pil_to_cv2(pil_image):
    """
    Convert PIL Image to OpenCV format (BGR).

    Args:
        pil_image: PIL Image object

    Returns:
        OpenCV image (numpy array in BGR format)
    """
    if Image is None:
        raise ImportError("PIL/Pillow is required for PIL image support. Install with: pip install Pillow")

    # Convert PIL Image to numpy array (RGB)
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')

    image_rgb = np.array(pil_image)

    # Convert RGB to BGR (OpenCV format)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    return image_bgr


def cv2_to_pil(cv2_image):
    """
    Convert OpenCV image (BGR) to PIL Image.

    Args:
        cv2_image: OpenCV image (numpy array in BGR format)

    Returns:
        PIL Image object
    """
    if Image is None:
        raise ImportError("PIL/Pillow is required for PIL image support. Install with: pip install Pillow")

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image
    pil_image = Image.fromarray(image_rgb)

    return pil_image


def remove_text_pil(pil_image, verbose=False):
    """
    Remove text from a PIL Image.

    Args:
        pil_image: PIL Image object
        verbose: If True, print processing information

    Returns:
        PIL Image object with text removed
    """
    if verbose:
        print(f"Processing PIL image...")
        print(f"Image size: {pil_image.width}x{pil_image.height}")

    # Convert PIL to OpenCV format
    cv2_image = pil_to_cv2(pil_image)

    # Detect text regions
    if verbose:
        print("Detecting text regions...")
    mask = detect_text_mask(cv2_image)

    # Count text pixels
    if verbose:
        text_pixels = np.sum(mask > 0)
        total_pixels = mask.shape[0] * mask.shape[1]
        text_percentage = (text_pixels / total_pixels) * 100
        print(f"Text coverage: {text_percentage:.2f}%")

    # Inpaint text regions
    if verbose:
        print("Removing text...")
    result_cv2 = inpaint_text_regions(cv2_image, mask)

    # Convert back to PIL
    result_pil = cv2_to_pil(result_cv2)

    if verbose:
        print("Processing complete!")

    return result_pil


def remove_text_from_image(input_path, output_path, verbose=True):
    """
    Remove text from an image file.

    Args:
        input_path: Path to input image file (string or Path object)
        output_path: Path to save output image file (string or Path object)
        verbose: If True, print processing information

    Returns:
        OpenCV image (BGR format) with text removed
    """
    # Read image
    image = cv2.imread(str(input_path))
    if image is None:
        raise ValueError(f"Could not read image: {input_path}")

    if verbose:
        print(f"Processing image: {input_path}")
        print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    # Detect text regions
    if verbose:
        print("Detecting text regions...")
    mask = detect_text_mask(image)

    # Count text pixels
    if verbose:
        text_pixels = np.sum(mask > 0)
        total_pixels = mask.shape[0] * mask.shape[1]
        text_percentage = (text_pixels / total_pixels) * 100
        print(f"Text coverage: {text_percentage:.2f}%")

    # Inpaint text regions
    if verbose:
        print("Removing text...")
    result = inpaint_text_regions(image, mask)

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save result
    cv2.imwrite(str(output_path), result)
    if verbose:
        print(f"Result saved to: {output_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Remove text from PNG images while preserving background.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --input image.png --output result.png
  %(prog)s --input samples/01.png --output output/01.png
        """
    )

    parser.add_argument(
        "--input",
        required=True,
        type=str,
        help="Path to input PNG image"
    )

    parser.add_argument(
        "--output",
        required=True,
        type=str,
        help="Path to output PNG image"
    )

    args = parser.parse_args()

    # Validate input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file does not exist: {args.input}", file=sys.stderr)
        sys.exit(1)

    if not input_path.suffix.lower() == '.png':
        print(f"Warning: Input file is not a PNG: {args.input}", file=sys.stderr)

    try:
        remove_text_from_image(input_path, args.output)
        print("\nSuccess! Text removed from image.")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
