# Invoice Analyzer ML Pipeline 🚀

A comprehensive, automated PDF-to-Structured-JSON pipeline designed for high-accuracy document parsing. This project utilizes PyMuPDF for conversion, PaddleOCR/OpenCV for hybrid extraction, and Gemini (Google GenAI) for semantic parsing.

## 🌟 Key Features
- **PDF-to-Image Conversion**: Uses `PyMuPDF` (muPDF engine) for lightning-fast, high-DPI image generation.
- **Hybrid OCR Strategy**: 
  - **Mode 1 (Spatial JSON)**: Raw word coordinates for LLM spatial awareness.
  - **Mode 2 (Markdown Reconstruction)**: Re-layers document text into virtual tables and lines.
  - **Mode 4 (Production Hybrid)**: Combines standard OCR with `PPStructure` for tables and OpenCV for checkbox detection.
- **LLM-Powered Parsing**: Deep semantic extraction using Gemini Flash models to convert noisy text into clean JSON.
- **Modular Design**: Completely decoupled modules for easy integration into existing workflows.

## 🛠️ Project Structure
```text
PROJECT_INVOICE_ANALYZER/
├── src/
│   ├── converter.py     # PDF to Image logic (PyMuPDF)
│   ├── extractor.py     # OCR extraction modes (1, 2, 4)
│   ├── parser.py        # LLM parsing/routing logic
├── pipeline.py          # Main entry point (Orchestrator)
├── requirements.txt     # Dependencies
├── .env.example         # Environment template
└── README.md            # You are here
```

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API Keys
Create a `.env` file from the template and add your Google Gemini API Key:
```bash
cp .env.example .env
```

### 3. Run the Pipeline
Execute the full pipeline on any PDF:
```bash
python pipeline.py --pdf data.pdf --mode 4 --output final_analysis.json
```

## 📊 Extraction Modes Explained

- **--mode 1 (Spatial JSON)**:
  Best for complex, unstructured forms where the LLM needs to know the exact location of each word.
- **--mode 2 (Markdown)**:
  Best for invoices with simple tables or lists. Reconstructs columns using whitespace and pipes (`|`).
- **--mode 4 (Hybrid)**:
  The "Production" mode. Detects tables as physical structures and uses heuristic OpenCV checks to find checkboxes.

## 📑 License
MIT License - Feel free to use and contribute!
