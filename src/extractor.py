import os
import cv2
import json
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR, PPStructure

# -------------------------------------------------------------------------
# OCR ENGINES INITIALIZATION
# 1. Standard OCR: For core word-level extraction
# 2. Structural Engine: For Tables and Form Layouts
# -------------------------------------------------------------------------
ocr_standard = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
layout_engine = PPStructure(table=True, ocr=True, layout=True, show_log=False)

def load_image(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def detect_checkboxes(img_gray, text_bboxes):
    """
    Heuristic detection for checkboxes using OpenCV.
    Excludes areas where text was already detected to reduce noise.
    """
    mask = np.zeros_like(img_gray)
    for bbox in text_bboxes:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(mask, (x1-3, y1-3), (x2+3, y2+3), 255, -1)

    _, binary = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY_INV)
    binary[mask == 255] = 0 # Avoid text regions

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    checkboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 15 < w < 45 and 15 < h < 45: # Standard checkbox size at 300 DPI
            roi = binary[y:y+h, x:x+w]
            density = cv2.countNonZero(roi) / (w * h)
            checkboxes.append({
                "type": "checkbox",
                "bbox": [float(x), float(y), float(x+w), float(y+h)],
                "is_checked": bool(density > 0.2) # Higher density means checked
            })
    return checkboxes

def detect_rotation(ocr_results):
    """
    Analyzes OCR results to determine if the page is likely rotated.
    If vertical word count > horizontal, the page might be sideways.
    """
    v_count = 0
    h_count = 0
    if not ocr_results or not ocr_results[0]:
        return False

    for line in ocr_results[0]:
        bbox = line[0]
        xs = [p[0] for p in bbox]
        ys = [p[1] for p in bbox]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        if height > width:
            v_count += 1
        else:
            h_count += 1
    return v_count > h_count

def extract_spatial_json(image_path):
    """
    MODE 1: Raw Spatial JSON (Coordinates + Text).
    Includes auto-rotation detection for high accuracy.
    """
    img = load_image(image_path)
    result = ocr_standard.ocr(img, cls=True)

    # Auto-Rotate Logic
    if detect_rotation(result):
        print(f"[INFO] Detected vertical text in {image_path}. Rotating...")
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        result = ocr_standard.ocr(img, cls=True)

    payload = []
    if result and result[0]:
        for line in result[0]:
            bbox = line[0]
            text = line[1][0]
            conf = float(line[1][1])
            xs, ys = [p[0] for p in bbox], [p[1] for p in bbox]
            payload.append({
                "text": text,
                "bbox": [min(xs), min(ys), max(xs), max(ys)],
                "confidence": round(conf, 4)
            })
    return payload

def extract_markdown_layout(spatial_data, row_threshold=15):
    """
    MODE 2: Virtual Markdown Layout.
    Groups words into lines based on Y-coordinates.
    """
    data = sorted(spatial_data, key=lambda x: x["bbox"][1]) # Sort by y_min
    
    lines = []
    for item in data:
        placed = False
        item_y = item["bbox"][1]
        for line in lines:
            line_y = line[0]["bbox"][1]
            if abs(item_y - line_y) < row_threshold:
                line.append(item)
                placed = True
                break
        if not placed:
            lines.append([item])
    
    md_lines = []
    for line in lines:
        row_sorted = sorted(line, key=lambda x: x["bbox"][0]) # Sort row by x
        row_str = "   |   ".join(c["text"] for c in row_sorted)
        md_lines.append(f"| {row_str} |")
        
    return "\n".join(md_lines)

def extract_production_hybrid(image_path):
    """
    MODE 4: Production Hybrid (Tables + Text + Checkboxes).
    Uses structural analysis for tables and OpenCV for boxes.
    """
    img = load_image(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    structure_res = layout_engine(img)
    spatial_data = extract_spatial_json(image_path)
    
    text_bboxes = [b["bbox"] for b in spatial_data]
    checkboxes = detect_checkboxes(img_gray, text_bboxes)
    
    final_output = {
        "text_blocks": spatial_data,
        "tables": [],
        "checkboxes": checkboxes
    }

    for block in structure_res:
        if block["type"] == "table":
            final_output["tables"].append({
                "bbox": block["bbox"],
                "html": block["res"].get("html", "")
            })

    return final_output

if __name__ == "__main__":
    # Test OCR
    example_img = "../images/page_1.png"
    if os.path.exists(example_img):
        print("[INFO] Running Hybrid OCR...")
        data = extract_production_hybrid(example_img)
        print(f"Extraction complete: Found {len(data['text_blocks'])} words, "
              f"{len(data['tables'])} tables, and {len(data['checkboxes'])} checkboxes.")
    else:
        print("Test image not found.")
