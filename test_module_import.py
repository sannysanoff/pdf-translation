#!/usr/bin/env python3
"""Test script to verify the module can be imported and used with PIL Images."""

from PIL import Image
from pathlib import Path
from claude_text_removal import remove_text_pil, remove_text_from_image

def test_pil_usage():
    """Test using PIL Image directly."""
    print("=" * 60)
    print("Test 1: Using PIL Image")
    print("=" * 60)

    input_path = Path("erasure-samples/09.png")
    output_path = Path("test/09_pil.png")

    # Load image with PIL
    img = Image.open(input_path)
    print(f"Loaded PIL image: {img.width}x{img.height}, mode={img.mode}")

    # Remove text
    result_img = remove_text_pil(img, verbose=True)

    # Save result
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_img.save(output_path)
    print(f"Saved result to: {output_path}\n")

def test_file_usage():
    """Test using file paths directly."""
    print("=" * 60)
    print("Test 2: Using file paths")
    print("=" * 60)

    remove_text_from_image(
        "erasure-samples/09.png",
        "test/09_file.png",
        verbose=True
    )
    print()

if __name__ == "__main__":
    print("Testing claude_text_removal as an importable module\n")

    test_pil_usage()
    test_file_usage()

    print("All tests completed successfully!")
