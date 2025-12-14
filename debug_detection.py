#!/usr/bin/env python3
import cv2
import numpy as np

img = cv2.imread('erasure-samples/10.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_height, img_width = img.shape[:2]
img_area = img_height * img_width

mean_intensity = np.mean(gray)
masks = []

# Method 1: Thresholding
if mean_intensity > 127:
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    masks.append(binary)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 15, 8)
    masks.append(adaptive)

# Method 2: Color-based
lower_dark = np.array([0, 0, 0])
upper_dark = np.array([180, 255, 100])
dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
masks.append(dark_mask)

lower_colored = np.array([0, 100, 50])
upper_colored = np.array([180, 255, 255])
colored_mask = cv2.inRange(hsv, lower_colored, upper_colored)

colored_contours, _ = cv2.findContours(colored_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
filtered_colored_mask = np.zeros_like(colored_mask)
for contour in colored_contours:
    area = cv2.contourArea(contour)
    if area < img_area * 0.05:
        cv2.drawContours(filtered_colored_mask, [contour], -1, 255, -1)

masks.append(filtered_colored_mask)

# Method 3: Variance
kernel_size = 5
kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
gray_float = gray.astype(np.float32)
mean_local = cv2.filter2D(gray_float, -1, kernel)
mean_sq_local = cv2.filter2D(gray_float**2, -1, kernel)
variance = mean_sq_local - mean_local**2
variance = cv2.normalize(variance, None, 0, 255, cv2.NORM_MINMAX)
variance = variance.astype(np.uint8)
_, variance_mask = cv2.threshold(variance, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
masks.append(variance_mask)

# Combine all masks
combined_mask = np.zeros_like(gray)
for i, mask in enumerate(masks):
    print(f"Mask {i}: {np.sum(mask > 0)} pixels")
    combined_mask = cv2.bitwise_or(combined_mask, mask)

print(f"\nCombined: {np.sum(combined_mask > 0)} pixels")
cv2.imwrite('test/step1_combined.png', combined_mask)

# Morphological operations
kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open)
print(f"After OPEN: {np.sum(combined_mask > 0)} pixels")
cv2.imwrite('test/step2_open.png', combined_mask)

kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close)
print(f"After CLOSE: {np.sum(combined_mask > 0)} pixels")
cv2.imwrite('test/step3_close.png', combined_mask)

# Filter contours
contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print(f"\nContours found: {len(contours)}")

refined_mask = np.zeros_like(combined_mask)
text_contours = []

kept = 0
for contour in contours:
    area = cv2.contourArea(contour)
    if area < 20:
        continue
    if area > img_area * 0.6:
        continue

    x, y, w, h = cv2.boundingRect(contour)
    if w > img_width * 0.9 or h > img_height * 0.9:
        continue

    if w > 0 and h > 0:
        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio > 30:
            continue

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area > 0:
        solidity = area / hull_area
        if solidity < 0.2:
            continue

    text_contours.append(contour)
    kept += 1

print(f"Contours kept: {kept}")

cv2.drawContours(refined_mask, text_contours, -1, 255, -1)
print(f"After drawing contours: {np.sum(refined_mask > 0)} pixels")
cv2.imwrite('test/step4_refined.png', refined_mask)

# Final dilation
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
refined_mask = cv2.dilate(refined_mask, kernel_dilate, iterations=3)
print(f"After final dilation: {np.sum(refined_mask > 0)} pixels")
cv2.imwrite('test/step5_final.png', refined_mask)
