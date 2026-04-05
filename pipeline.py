import os
import json
import argparse
from src.converter import pdf_to_images
from src.extractor import extract_spatial_json, extract_markdown_layout, extract_production_hybrid
from src.parser import parse_invoice_data

def run_invoice_pipeline(pdf_path, mode=4, output_file="output.json"):
    """
    Main Orchestrator for the Invoice Analyzer.
    Step 1: PDF to Images (PyMuPDF)
    Step 2: OCR Extraction (Spatial, Markdown, or Production Hybrid)
    Step 3: LLM Parsing/Routing (Gemini 2.5/3 Flash)
    """
    print(f"--- [INFO] Starting Invoice Analysis Pipeline on {pdf_path} (Mode {mode}) ---")
    
    # 1. Convert PDF to Image
    print("[1/3] Converting PDF to images using PyMuPDF...")
    images = pdf_to_images(pdf_path, output_dir="temp_images")
    if not images:
        print("[ERROR] PDF conversion failed.")
        return

    full_results = {}

    # 2. Extract Data Page by Page
    print(f"[2/3] Extracting data from {len(images)} pages...")
    for idx, img_path in enumerate(images):
        page_num = idx + 1
        print(f"      Processing Page {page_num}...")
        
        if mode == 1:
            # MODE 1: Spatial JSON (Raw Coordinates)
            page_data = extract_spatial_json(img_path)
            
        elif mode == 2:
            # MODE 2: Markdown Reconstruction
            spatial = extract_spatial_json(img_path)
            page_data = extract_markdown_layout(spatial)
            
        elif mode == 4:
            # MODE 4: Production Hybrid (Tables + Checkboxes + Text)
            page_data = extract_production_hybrid(img_path)
            
        else:
            print(f"[ERROR] Invalid mode: {mode}")
            return

        # 3. LLM Parsing / Routing
        print(f"      Parsing Page {page_num} using LLM...")
        parsed_json = parse_invoice_data(page_data)
        
        full_results[f"page_{page_num}"] = {
            "image": img_path,
            "extracted_raw": page_data,
            "parsed_json": parsed_json
        }

    # 4. Save Final Output
    with open(output_file, "w") as f:
        json.dump(full_results, f, indent=2)
    
    print(f"\n✅ [SUCCESS] Pipeline Complete. Final results saved to {output_file}")
    return full_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Invoice Analyzer ML Pipeline")
    parser.add_argument("--pdf", type=str, required=True, help="Path to PDF file")
    parser.add_argument("--mode", type=int, default=4, choices=[1, 2, 4], help="OCR Extraction Mode (1, 2, 4)")
    parser.add_argument("--output", type=str, default="final_analysis.json", help="Output filename")
    
    args = parser.parse_args()
    
    if os.path.exists(args.pdf):
        run_invoice_pipeline(args.pdf, mode=args.mode, output_file=args.output)
    else:
        print(f"File {args.pdf} not found.")
