#!/usr/bin/env python3
"""
Evaluate text removal results by comparing with ground truth images.
Calculates various distance metrics.
"""

import cv2
import numpy as np
from pathlib import Path
import sys


def calculate_metrics(output_img, ground_truth_img):
    """Calculate various distance metrics between output and ground truth."""

    # Ensure same dimensions
    if output_img.shape != ground_truth_img.shape:
        print(f"Warning: Shape mismatch - output: {output_img.shape}, ground truth: {ground_truth_img.shape}")
        return None

    # Mean Squared Error (MSE)
    mse = np.mean((output_img.astype(float) - ground_truth_img.astype(float)) ** 2)

    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)

    # Peak Signal-to-Noise Ratio (PSNR)
    if mse > 0:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    else:
        psnr = float('inf')

    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(output_img.astype(float) - ground_truth_img.astype(float)))

    # Structural Similarity Index (SSIM) approximation
    # Using a simple correlation-based metric
    output_flat = output_img.flatten().astype(float)
    gt_flat = ground_truth_img.flatten().astype(float)

    correlation = np.corrcoef(output_flat, gt_flat)[0, 1]

    return {
        'mse': mse,
        'rmse': rmse,
        'psnr': psnr,
        'mae': mae,
        'correlation': correlation
    }


def main():
    # Define paths
    test_dir = Path("test")
    ground_truth_dir = Path("erasure-samples/ground-truth")

    # Get all PNG files in test directory
    test_files = sorted(test_dir.glob("*.png"))

    if not test_files:
        print("No test files found in test/ directory")
        sys.exit(1)

    print("Evaluating text removal results...")
    print("=" * 80)

    all_metrics = []

    for test_file in test_files:
        # Find corresponding ground truth
        gt_file = ground_truth_dir / test_file.name

        if not gt_file.exists():
            print(f"Skipping {test_file.name} - no ground truth found")
            continue

        # Read images
        output_img = cv2.imread(str(test_file))
        gt_img = cv2.imread(str(gt_file))

        if output_img is None or gt_img is None:
            print(f"Error reading images for {test_file.name}")
            continue

        # Calculate metrics
        metrics = calculate_metrics(output_img, gt_img)

        if metrics:
            all_metrics.append(metrics)

            print(f"\n{test_file.name}:")
            print(f"  MSE:         {metrics['mse']:.2f}")
            print(f"  RMSE:        {metrics['rmse']:.2f}")
            print(f"  MAE:         {metrics['mae']:.2f}")
            print(f"  PSNR:        {metrics['psnr']:.2f} dB")
            print(f"  Correlation: {metrics['correlation']:.4f}")

    if all_metrics:
        print("\n" + "=" * 80)
        print("AVERAGE METRICS:")
        print(f"  MSE:         {np.mean([m['mse'] for m in all_metrics]):.2f}")
        print(f"  RMSE:        {np.mean([m['rmse'] for m in all_metrics]):.2f}")
        print(f"  MAE:         {np.mean([m['mae'] for m in all_metrics]):.2f}")
        print(f"  PSNR:        {np.mean([m['psnr'] for m in all_metrics if m['psnr'] != float('inf')]):.2f} dB")
        print(f"  Correlation: {np.mean([m['correlation'] for m in all_metrics]):.4f}")
        print("=" * 80)

        # Overall assessment
        avg_mae = np.mean([m['mae'] for m in all_metrics])
        avg_correlation = np.mean([m['correlation'] for m in all_metrics])

        print("\nQuality Assessment:")
        if avg_mae < 5:
            print("  Excellent - Very close to ground truth")
        elif avg_mae < 10:
            print("  Good - Minor differences from ground truth")
        elif avg_mae < 20:
            print("  Fair - Noticeable differences from ground truth")
        else:
            print("  Poor - Significant differences from ground truth")

        print(f"\nLower distance metrics (MSE, RMSE, MAE) are better.")
        print(f"Higher PSNR and correlation are better.")


if __name__ == "__main__":
    main()
