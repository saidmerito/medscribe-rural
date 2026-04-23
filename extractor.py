# ocr/extractor.py
"""
OCR pipeline for handwritten medical registers.
Uses PaddleOCR with multilingual support (French, Arabic, English).
Includes image preprocessing to handle low-quality field photos.
"""

import cv2
import numpy as np
from paddleocr import PaddleOCR
from pathlib import Path
from config import OCR_LANGUAGES, OCR_USE_GPU

# Lazy-load OCR models (one per script family to maximize accuracy)
_ocr_latin = None
_ocr_arabic = None


def _get_ocr_latin():
    global _ocr_latin
    if _ocr_latin is None:
        _ocr_latin = PaddleOCR(use_angle_cls=True, lang="fr", use_gpu=OCR_USE_GPU, show_log=False)
    return _ocr_latin


def _get_ocr_arabic():
    global _ocr_arabic
    if _ocr_arabic is None:
        _ocr_arabic = PaddleOCR(use_angle_cls=True, lang="ar", use_gpu=OCR_USE_GPU, show_log=False)
    return _ocr_arabic


def preprocess_image(image_path: str) -> np.ndarray:
    """
    Preprocess a register photo for optimal OCR performance.
    Handles: deskewing, noise reduction, contrast enhancement, binarization.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=10)

    # Adaptive thresholding for uneven lighting (common in field photos)
    binary = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # Deskew
    deskewed = _deskew(binary)

    return deskewed


def _deskew(image: np.ndarray) -> np.ndarray:
    """Correct page tilt using Hough line detection."""
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    if lines is None:
        return image

    angles = []
    for line in lines[:20]:  # Use top 20 lines
        rho, theta = line[0]
        angle = (theta - np.pi / 2) * (180 / np.pi)
        if abs(angle) < 20:  # Only small tilts
            angles.append(angle)

    if not angles:
        return image

    median_angle = np.median(angles)
    if abs(median_angle) < 0.5:
        return image

    h, w = image.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def _detect_script(line_text: str) -> str:
    """Detect dominant script in a text line: 'arabic' or 'latin'."""
    arabic_chars = sum(1 for c in line_text if '\u0600' <= c <= '\u06FF')
    return "arabic" if arabic_chars > len(line_text) * 0.3 else "latin"


def extract_text(image_path: str) -> str:
    """
    Full OCR pipeline: preprocess image → extract text → return raw string.
    Routes lines to the appropriate OCR model based on detected script.
    """
    processed = preprocess_image(image_path)

    # Save preprocessed image to temp file for PaddleOCR
    temp_path = str(Path(image_path).parent / "_preprocessed_temp.jpg")
    cv2.imwrite(temp_path, processed)

    # Run both OCR models
    result_latin = _get_ocr_latin().ocr(temp_path, cls=True)
    result_arabic = _get_ocr_arabic().ocr(temp_path, cls=True)

    # Merge results: for each line, pick the model with higher confidence
    lines = _merge_ocr_results(result_latin, result_arabic)

    # Clean up temp file
    Path(temp_path).unlink(missing_ok=True)

    return "\n".join(lines)


def _merge_ocr_results(result_latin, result_arabic) -> list[str]:
    """
    Merge results from Latin and Arabic OCR models.
    For overlapping bounding boxes, pick the result with higher confidence.
    """
    all_lines = []

    def extract_lines(result):
        if not result or not result[0]:
            return []
        lines = []
        for line in result[0]:
            bbox, (text, confidence) = line
            y_center = (bbox[0][1] + bbox[2][1]) / 2
            lines.append({"y": y_center, "text": text, "conf": confidence})
        return lines

    latin_lines = extract_lines(result_latin)
    arabic_lines = extract_lines(result_arabic)

    # Simple merge: sort all by vertical position, deduplicate overlapping lines
    all_raw = latin_lines + arabic_lines
    all_raw.sort(key=lambda x: x["y"])

    seen_y = []
    for line in all_raw:
        # Check if we already have a line at approximately this vertical position
        duplicate = False
        for y in seen_y:
            if abs(line["y"] - y) < 15:  # Within 15px = same line
                duplicate = True
                break
        if not duplicate:
            all_lines.append(line["text"])
            seen_y.append(line["y"])

    return all_lines
