#!/usr/bin/env python3
import sys
import os
import subprocess

def main():
    if len(sys.argv) < 2:
        print("Usage: python pdf_to_png.py <pdf_path> [-dpi <dpi>]")
        sys.exit(1)

    pdf_path = sys.argv[1]
    dpi = 300
    if len(sys.argv) > 3 and sys.argv[2] == '-dpi':
        dpi = int(sys.argv[3])

    # Get output directory
    base_dir = os.path.dirname(pdf_path)
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_dir = os.path.join(base_dir, pdf_name)
    os.makedirs(output_dir, exist_ok=True)

    # Ghostscript command
    gs_command = [
        'gs',
        '-dSAFER',
        '-dBATCH',
        '-dNOPAUSE',
        '-sDEVICE=png16m',
        f'-r{dpi}',
        '-dTextAlphaBits=4',
        '-dGraphicsAlphaBits=4',
        f'-sOutputFile={os.path.join(output_dir, "%04d.png")}',
        pdf_path
    ]

    try:
        subprocess.run(gs_command, check=True)
        print(f"Converted PDF pages to PNG in {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting PDF: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()