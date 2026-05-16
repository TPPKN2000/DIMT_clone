---
title: DIMT - Document Intelligent Machine Translation
emoji: 📄
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.57.0
app_file: src/frontend/app.py
pinned: false
---
# DIMT — Document Intelligent Machine Translation

> Production pipeline for translating academic PDF documents (English → Vietnamese) with layout-preserving PDF reconstruction.

## Architecture

```
Input PDF ──→ MinerU API ──→ ZIP (md + middle.json + images/)
                                    │
                    ┌───────────────┤
                    ▼               ▼
              para_blocks      equation images
              merge lines      (interline_eq)
              into paragraphs
                    │
                    ▼
              NLLB-1.3B (LoRA)
              paragraph-level
              translation
                    │
              ┌─────┴─────┐
              ▼            ▼
        translated.md   translated.pdf
                          (images for equations,
                           translated tables)
                    │
                    ▼
              AI Agent (Gemma-4-31B-it)
              ├── Q4 score verification
              ├── keyword extraction
              └── WikiSearch URLs (Vietnamese)
```

## Setup & Installation
 
1. **Install Dependencies (including Torch CUDA 11.8)**:
```bash
uv sync
```

2. **Configure Environment Variables**:
   Create a `.env` file in the root directory with your API keys:
   ```ini
   MINERU_API_KEY=your_mineru_api_key
   GEMMA_4_API_KEY=your_gemma_api_key
   GEMINI_API_KEY=your_gemini_api_key
   ```

3. **NLLB Model**: Ensure the fine-tuned adapter is located at `nllb-1.3B-final/` (local directory)

## How to Run

1. **Start the FastAPI Backend**:
```bash
uv run uvicorn src.backend.api:app --port 8000
```
*(Lưu ý: Quá trình khởi động sẽ mất vài phút để nạp Model NLLB. Bạn sẽ thấy thông báo trạng thái của MinerU API trước khi NLLB bắt đầu nạp).*

2. **Start the Streamlit Frontend**:
   Mở một terminal mới (nhớ kích hoạt môi trường `.venv` nếu không dùng `uv run`):
```bash
uv run streamlit run src/frontend/app.py
```

3. **Monitor MLflow Metrics** (optional):
```bash
uv run mlflow ui --port 5000
```

## Pipeline Steps

### Step 1: PDF Extraction (MinerU API)
- Upload PDF → MinerU Precision API (`/api/v4/file-urls/batch`)
- Async polling until `state: "done"` → download ZIP
- ZIP contains: `.md`, `layout.json`, `images/` (equation renders)
- Rate limits: 50 files/min, 1000 polls/min, ≤200 pages, ≤200MB
- **Fallback**: local mock data in `data/` for demo

### Step 2: Translation (NLLB-1.3B + LoRA)
- Parses `para_blocks` from `layout.json` (pre-grouped paragraphs)
- Respects `merge_prev` flag to concatenate continuation blocks
- Protects inline equations with `[EQ_n]` placeholders
- Translates at paragraph level (not line-by-line) for better context
- Tables: extracts text from HTML cells, translates, reconstructs

### Step 3: PDF Reconstruction (PyMuPDF)
- Creates new PDF with same page dimensions as original
- **Equations**: rendered as IMAGES from `images/` folder (not LaTeX)
- **Tables**: exception — translated text rendered as table grid
- **Text/title**: Vietnamese text placed at original bounding boxes
- Supports Vietnamese diacritics via system TrueType fonts

### Step 4: AI Agent (Gemma-4-31B-it)
- **Q4 Verification**: Identifies spans with scores in the bottom 25th percentile (Q4 quartile) from `layout.json`, uses LLM to verify OCR quality
- **Keyword Extraction**: Parses `Keywords:` from abstract section
- **WikiSearch**: Generates Vietnamese Wikipedia URLs for each keyword

## Usage

1. In the Streamlit UI, upload a PDF document
2. Click **Convert** — extracts, translates, and renders the PDF
3. Preview the translated markdown in the **Translated Markdown** tab
4. Edit text in the **Editable Text** tab if needed
5. Click **Run Agent Analysis** to see Q4 verification and keyword references
6. Download translated `.md` and `.pdf` from the **Downloads** tab
7. Rate the output and submit feedback (logged to MLflow)

## Key Files

| File | Purpose |
|------|---------|
| `src/backend/mineru_client.py` | MinerU API client (real + mock fallback) |
| `src/backend/nllb_service.py` | NLLB translation with paragraph merging |
| `src/backend/agent.py` | AI Agent (Q4 verify, keywords, WikiSearch) |
| `src/backend/pdf_renderer.py` | PDF reconstruction via PyMuPDF |
| `src/backend/api.py` | FastAPI endpoint orchestrator |
| `src/frontend/app.py` | Streamlit UI |
| `src/backend/evaluation.py` | MLflow metrics logger |

## Data Files (from MinerU)

| File | Purpose |
|------|---------|
| `full.md` | Final Markdown rendering of the entire document (flat text with LaTeX math, tables as HTML) |
| `layout.json` | Rich hierarchical structure with bboxes, spans, images|
| `content_list.json` | Flat content list in reading order |
| `images/` | Equation/table renders (JPG, SHA-256 filenames) |
