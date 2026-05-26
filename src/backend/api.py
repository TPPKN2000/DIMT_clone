"""
FastAPI Backend — Production pipeline API.

Endpoints:
  POST /upload         — Upload PDF → MinerU extraction (auto-split >200 pages)
  POST /translate      — Translate via NLLB (layout.json paragraph-level, batch)
  POST /render-pdf     — Render translated PDF from layout.json + images
  POST /agent/verify   — AI agent: Q4 score verification only
  POST /agent/keywords — AI agent: keyword extraction + WikiSearch URLs
  POST /feedback       — Log user metrics to MLflow
  GET  /download/{id}  — Download rendered PDF
  GET  /stream-pdf/{id} — Stream PDF for preview
  POST /hitl/update    — Human-in-the-Loop: update flagged translation
"""

import json
import re
import uuid
import time
from datetime import datetime
import asyncio
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Form, Header
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from .mineru_client import MinerUClient
from .nllb_service import NLLBService
from .marianmt_service import MarianMTService
from .agent import AIAgent
from .evaluation import Evaluator
from .pdf_renderer import PDFRenderer
from .mongo_store import MongoDocStore

# ── Services ─────────────────────────────────────────────────
mineru = MinerUClient(output_dir="temp")
nllb = NLLBService(lazy_load=True)
marianmt = MarianMTService(lazy_load=True)
agent = AIAgent()
evaluator = Evaluator()

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    print("[API] Shutting down.")

app = FastAPI(title="DIMT — Document Intelligent Machine Translation", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)
# Dual-write store: in-memory cache + MongoDB persistence
doc_store = MongoDocStore()

# GPU concurrency guard (single-user, prevent CUDA OOM)
gpu_semaphore = asyncio.Semaphore(1)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Max pages before auto-split for MinerU (API limit: 15 pages)
MAX_MINERU_PAGES = 10


# ── Models ──────────────────────────────────────────────────
class TranslateRequest(BaseModel):
    doc_id: str
    tgt_lang: str = "fra_Latn"
    model_choice: str = "nllb"

class UserAuthRequest(BaseModel):
    username: str
    password_hash: str

class RenderRequest(BaseModel):
    doc_id: str

class AgentRequest(BaseModel):
    doc_id: str
    llm_provider: str = "gemini"

class FeedbackRequest(BaseModel):
    doc_id: str
    original_md: str
    modified_md: str
    user_rating: int
    downloaded: bool
    time_consumed: float

class HITLUpdateRequest(BaseModel):
    doc_id: str
    page_idx: int
    block_idx: int
    new_text: str

class UpdateMarkdownRequest(BaseModel):
    markdown: str

class CorrectionItem(BaseModel):
    page: int
    original: str
    corrected: str
    bbox: Optional[list] = None

class ApplyCorrectionsRequest(BaseModel):
    doc_id: str
    corrections: list[CorrectionItem]


# ── Helpers ─────────────────────────────────────────────────

def _count_pdf_pages(pdf_path: Path) -> int:
    """Count pages in a PDF using PyMuPDF."""
    import fitz
    doc = fitz.open(str(pdf_path))
    count = len(doc)
    doc.close()
    return count

def _split_pdf_in_memory(pdf_path: Path, chunk_size: int = MAX_MINERU_PAGES) -> list[tuple[bytes, str]]:
    """Split a large PDF into chunks in memory. Returns list of (pdf_bytes, chunk_name)."""
    import fitz
    doc = fitz.open(str(pdf_path))
    total = len(doc)
    if total <= chunk_size:
        doc.close()
        with open(pdf_path, "rb") as f:
            return [(f.read(), pdf_path.name)]

    chunks = []
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size - 1, total - 1)
        chunk_doc = fitz.open()
        chunk_doc.insert_pdf(doc, from_page=start, to_page=end)
        chunk_bytes = chunk_doc.tobytes()
        chunk_doc.close()
        chunk_name = f"{pdf_path.stem}_chunk_{start}_{end}.pdf"
        chunks.append((chunk_bytes, chunk_name))
        print(f"[API] Split chunk in memory: pages {start}-{end} → {chunk_name}")

    doc.close()
    return chunks

def _count_paragraphs(layout_data: dict) -> int:
    """Count translatable blocks for timeout estimation."""
    if not layout_data: return 0
    count = 0
    for page in layout_data.get("pdf_info", []):
        # Heuristic: count blocks in both para_blocks and preproc_blocks
        blocks = page.get("preproc_blocks", page.get("para_blocks", []))
        count += len(blocks)
    return count

def _merge_layout_jsons(layouts: list[dict]) -> dict:
    """Merge multiple layout.json dicts into one, adjusting page indices."""
    merged = {"pdf_info": []}
    page_offset = 0
    for layout in layouts:
        for page in layout.get("pdf_info", []):
            page_copy = dict(page)
            page_copy["page_idx"] = page_offset + page.get("page_idx", 0)
            merged["pdf_info"].append(page_copy)
        page_offset += len(layout.get("pdf_info", []))
    return merged


def reconstruct_markdown_from_layout(layout_data: dict) -> str:
    if not layout_data:
        return ""
    md_blocks = []
    heading_types = {"title", "section_title", "heading", "subheading"}
    
    def process_block(block):
        btype = block.get("type", "text")
        
        def get_block_text(b):
            parts = []
            for line in b.get("lines", []):
                line_parts = []
                for span in line.get("spans", []):
                    span_type = span.get("type", "text")
                    content = span.get("content", "")
                    if not content:
                        continue
                    if span_type in {"inline_equation", "interline_equation"}:
                        line_parts.append(f"${content.strip()}$")
                    elif span_type == "table" and "html" in span:
                        line_parts.append(span["html"])
                    else:
                        line_parts.append(content)
                parts.append("".join(line_parts))
            return "\n".join(parts)
            
        if "blocks" in block:
            sub_texts = []
            for sub in block.get("blocks", []):
                sub_texts.append(process_block(sub))
            text = "\n".join([st for st in sub_texts if st.strip()])
        else:
            text = get_block_text(block)
            
        if not text.strip():
            return ""
            
        if btype == "title":
            return f"# {text.strip()}"
        elif btype == "table":
            return text
        elif btype in {"interline_equation", "equation"}:
            return f"$$\n{text.strip()}\n$$"
        elif btype == "image_caption":
            return f"*{text.strip()}*"
        elif btype == "page_footnote":
            return f"_{text.strip()}_"
        else:
            return text

    for page in layout_data.get("pdf_info", []):
        all_root_blocks = (
            page.get("preproc_blocks", page.get("para_blocks", [])) + 
            page.get("discarded_blocks", [])
        )
        for block in all_root_blocks:
            tb = process_block(block)
            if tb.strip():
                md_blocks.append(tb)
                
    return "\n\n".join(md_blocks)


# ── Endpoints ───────────────────────────────────────────────

@app.post("/upload")
async def upload_file(
    pdf_path: str = Form(...),
    x_user: Optional[str] = Header(None),
    save_to_db: Optional[bool] = Form(True)
):
    """Accept absolute PDF path → extract via MinerU. Auto-splits PDFs >10 pages."""
    doc_id = str(uuid.uuid4())[:8]
    evaluator.start_inference(doc_id)
    print(f"\n{'='*60}")

    save_path = Path(pdf_path)
    if not save_path.exists():
        return {"status": "error", "message": f"File does not exist: {pdf_path}"}
    filename = str(save_path.absolute())
    print(f"[API] /upload — doc_id={doc_id}, pdf_path={filename}, user={x_user}, save_to_db={save_to_db}")

    # Q8: Auto-split >10-page PDFs
    page_count = _count_pdf_pages(save_path)
    print(f"[API] PDF has {page_count} pages")

    if page_count > MAX_MINERU_PAGES:
        print(f"[API] Large PDF detected ({page_count} pages). Auto-splitting in memory into chunks of {MAX_MINERU_PAGES}...")
        chunks = _split_pdf_in_memory(save_path)
        all_layouts = []
        all_markdowns = []
        images_dir = None
        extract_dir = None

        for i, (chunk_bytes, chunk_name) in enumerate(chunks):
            print(f"[API] Extracting chunk {i+1}/{len(chunks)}: {chunk_name}")
            res = await asyncio.to_thread(mineru.extract_from_bytes, chunk_bytes, chunk_name)
            if res["status"] == "success":
                all_markdowns.append(res.get("markdown", ""))
                if res.get("middle_json"):
                    all_layouts.append(res["middle_json"])
                if not images_dir:
                    images_dir = res.get("images_dir")
                if not extract_dir:
                    extract_dir = res.get("extract_dir")
            else:
                print(f"[API] ⚠️ Chunk {i+1} extraction failed: {res.get('message')}")

        merged_md = "\n\n---\n\n".join(all_markdowns)
        merged_layout = _merge_layout_jsons(all_layouts) if all_layouts else None

        doc_store.set(doc_id, {
            "filename": filename,
            "markdown": merged_md,
            "middle_json": merged_layout,
            "images_dir": images_dir,
            "extract_dir": extract_dir,
            "num_pages": page_count,
            "num_paragraphs": _count_paragraphs(merged_layout),
            "user_id": x_user,
            "save_to_db": save_to_db
        })
        print(f"[API] ✅ Large PDF extraction complete. {len(all_layouts)} chunks merged.")
        return {
            "status": "success", 
            "doc_id": doc_id, 
            "num_pages": page_count, 
            "num_paragraphs": _count_paragraphs(merged_layout)
        }

    # Normal extraction for ≤10-page PDFs
    print(f"[API] Calling MinerU API...")
    res = await asyncio.to_thread(mineru.extract_from_file, str(save_path))

    if res["status"] == "success":
        m_json = res.get("middle_json")
        p_count = _count_paragraphs(m_json)
        doc_store.set(doc_id, {
            "filename": filename,
            "markdown": res["markdown"],
            "middle_json": m_json,
            "images_dir": res.get("images_dir"),
            "extract_dir": res.get("extract_dir"),
            "num_pages": page_count,
            "num_paragraphs": p_count,
            "user_id": x_user,
            "save_to_db": save_to_db
        })
        print(f"[API] ✅ Extraction complete. MD length={len(res['markdown'])}, "
              f"has_layout={'middle_json' in res and res['middle_json'] is not None}")
        return {
            "status": "success", 
            "doc_id": doc_id,
            "num_pages": page_count,
            "num_paragraphs": p_count
        }

    print(f"[API] ❌ Extraction failed: {res.get('message')}")
    return {"status": "error", "message": res.get("message", "Extraction failed")}


@app.post("/translate")
async def translate_document(req: TranslateRequest):
    """Translate document using NLLB or MarianMT — paragraph-level from layout.json."""
    print(f"\n{'='*60}")
    print(f"[API] /translate — doc_id={req.doc_id}, tgt_lang={req.tgt_lang}, model_choice={req.model_choice}")
    t_start = time.time()

    doc = doc_store.get(req.doc_id)
    if not doc:
        print(f"[API] ❌ Document {req.doc_id} not found")
        return {"status": "error", "message": "Document not found"}

    middle_data = doc.get("middle_json")
    translated_md = ""
    translated_middle = None

    async with gpu_semaphore:
        print("[API] GPU semaphore acquired. Starting translation...")
        if middle_data:
            page_count = len(middle_data.get("pdf_info", []))
            print(f"[API] Translating layout.json ({page_count} pages) using {req.model_choice}...")
            if req.model_choice == "marianmt":
                marianmt.set_target_lang(req.tgt_lang)
                translated_middle = await asyncio.to_thread(
                    marianmt.translate_middle_json, middle_data
                )
            else:
                nllb.set_target_lang(req.tgt_lang)
                translated_middle = await asyncio.to_thread(
                    nllb.translate_middle_json, middle_data
                )
            doc["translated_middle"] = translated_middle
        else:
            pass

    elapsed = time.time() - t_start
    if translated_middle:
        translated_md = reconstruct_markdown_from_layout(translated_middle)
    doc["translated_md"] = translated_md

    # Track approximate token count for production metrics
    src_md = doc.get("markdown", "")
    evaluator.record_tokens(len(src_md.split()) + len(translated_md.split()))
    doc_store.update(req.doc_id, doc)
    evaluator.end_inference(req.doc_id)

    print(f"[API] ✅ Translation complete in {elapsed:.1f}s. MD length={len(translated_md)}")

    return {
        "status": "success",
        "translated_markdown": translated_md,
        "has_middle_json": middle_data is not None,
    }


@app.post("/render-pdf")
async def render_pdf(req: RenderRequest):
    """Render translated PDF from translated layout.json + images."""
    print(f"\n{'='*60}")
    print(f"[API] /render-pdf — doc_id={req.doc_id}")
    t_start = time.time()

    doc = doc_store.get(req.doc_id)
    if not doc:
        print(f"[API] ❌ Document {req.doc_id} not found")
        return {"status": "error", "message": "Document not found"}

    # Use already-translated layout data (decoupled from translation)
    translated_middle = doc.get("translated_middle")
    middle_data = doc.get("middle_json")
    layout_data = translated_middle or middle_data
    if not layout_data:
        print("[API] ❌ No layout.json found for rendering")
        return {"status": "error", "message": "No layout.json found. Run translation first."}

    # Find origin PDF path
    filename_val = doc["filename"]
    if Path(filename_val).is_absolute():
        origin_pdf_path = Path(filename_val)
    else:
        origin_pdf_path = Path("input_docs") / filename_val

    if not origin_pdf_path.exists():
        print(f"[API] ❌ Origin PDF not found: {origin_pdf_path}")
        return {"status": "error", "message": f"Origin PDF not found: {origin_pdf_path}"}

    images_dir = doc.get("images_dir")
    renderer = PDFRenderer(images_dir=images_dir)

    orig_name = Path(filename_val).stem
    output_path = OUTPUT_DIR / f"{orig_name}_translated.pdf"
    print(f"[API] Rendering PDF → {output_path}")

    try:
        async with gpu_semaphore:
            await asyncio.to_thread(
                renderer.render, layout_data, str(origin_pdf_path), str(output_path)
            )
    except Exception as e:
        print(f"[API] ❌ Render error: {e}")
        traceback.print_exc()
        return {"status": "error", "message": f"Render failed: {e}"}

    elapsed = time.time() - t_start
    doc["pdf_path"] = str(output_path)
    
    # Store rendered PDF bytes in-database
    if output_path.exists():
        try:
            with open(output_path, "rb") as f:
                doc["pdf_bytes"] = f.read()
            print(f"[API] Saved PDF bytes to DB for: {output_path.name}")
        except Exception as e:
            print(f"[API] Warning: could not store PDF bytes: {e}")

    # Delete temporary MinerU folder under data/
    import shutil
    extract_dir = doc.get("extract_dir")
    if extract_dir:
        try:
            shutil.rmtree(extract_dir, ignore_errors=True)
            print(f"[API] Temporary MinerU folder deleted: {extract_dir}")
        except Exception as e:
            print(f"[API] Warning: could not delete temporary folder {extract_dir}: {e}")

    doc_store.update(req.doc_id, doc)
    print(f"[API] ✅ PDF rendered in {elapsed:.1f}s")
    return {"status": "success", "pdf_path": str(output_path)}


@app.get("/stream-pdf/{doc_id}")
async def stream_pdf(doc_id: str):
    """Stream the rendered translated PDF for preview."""
    doc = doc_store.get(doc_id)
    if not doc:
        return JSONResponse({"status": "error", "message": "PDF not found"}, status_code=404)
        
    if "pdf_bytes" in doc and doc["pdf_bytes"]:
        return Response(content=doc["pdf_bytes"], media_type="application/pdf")

    if "pdf_path" in doc:
        pdf_path = Path(doc["pdf_path"])
        if pdf_path.exists():
            return FileResponse(
                str(pdf_path),
                media_type="application/pdf",
            )
    return JSONResponse({"status": "error", "message": "PDF file missing"}, status_code=404)


@app.get("/download/{doc_id}")
async def download_pdf(doc_id: str):
    """Download the rendered translated PDF."""
    doc = doc_store.get(doc_id)
    if not doc:
        return JSONResponse({"status": "error", "message": "PDF not found"}, status_code=404)
        
    filename = f"{Path(doc.get('filename', 'translated')).stem}_translated.pdf"
    
    if "pdf_bytes" in doc and doc["pdf_bytes"]:
        return Response(
            content=doc["pdf_bytes"],
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )

    if "pdf_path" in doc:
        pdf_path = Path(doc["pdf_path"])
        if pdf_path.exists():
            return FileResponse(
                str(pdf_path),
                media_type="application/pdf",
                filename=filename,
            )
    return JSONResponse({"status": "error", "message": "PDF file missing"}, status_code=404)


# ── Agent: Q4 Verification (called right after upload) ─────

@app.post("/agent/verify")
async def agent_verify(req: AgentRequest):
    """Run Q4 score verification on extracted layout data."""
    print(f"\n{'='*60}")
    print(f"[API] /agent/verify — doc_id={req.doc_id}")
    t_start = time.time()

    doc = doc_store.get(req.doc_id)
    if not doc:
        print(f"[API] ❌ Document {req.doc_id} not found")
        return {"status": "error", "message": "Document not found"}

    layout_data = doc.get("middle_json", {})

    try:
        agent.set_llm_provider(req.llm_provider)
        print(f"[API] Running Q4 verification (LLM: {req.llm_provider})...")
        q4_result = await asyncio.to_thread(agent.verify_q4_elements, layout_data)
        doc["agent_result"] = {"q4_verification": q4_result}
        doc_store.update(req.doc_id, doc)
        elapsed = time.time() - t_start
        q4_count = q4_result.get("q4_count", 0)
        print(f"[API] ✅ Q4 verification complete in {elapsed:.1f}s. Q4 elements={q4_count}")
        return {"status": "success", "q4_verification": q4_result}
    except Exception as e:
        print(f"[API] ❌ Agent verify error: {e}")
        traceback.print_exc()
        return {"status": "error", "message": f"Agent verify failed: {e}"}


# ── Agent: Keywords (called in parallel with translate) ─────

@app.post("/agent/keywords")
async def agent_keywords(req: AgentRequest):
    """Extract keywords and generate WikiSearch URLs."""
    print(f"\n{'='*60}")
    print(f"[API] /agent/keywords — doc_id={req.doc_id}")
    t_start = time.time()

    doc = doc_store.get(req.doc_id)
    if not doc:
        print(f"[API] ❌ Document {req.doc_id} not found")
        return {"status": "error", "message": "Document not found"}

    markdown = doc.get("markdown", "")

    try:
        agent.set_llm_provider(req.llm_provider)
        print(f"[API] Extracting keywords (LLM: {req.llm_provider})...")
        keywords = await asyncio.to_thread(agent.extract_keywords, markdown)

        # Merge into existing agent_result
        agent_result = doc.get("agent_result", {})
        agent_result["keywords"] = keywords
        doc["agent_result"] = agent_result
        doc_store.update(req.doc_id, doc)

        elapsed = time.time() - t_start
        print(f"[API] ✅ Keywords complete in {elapsed:.1f}s. keywords={len(keywords)}")
        return {"status": "success", "keywords": keywords}
    except Exception as e:
        print(f"[API] ❌ Agent keywords error: {e}")
        traceback.print_exc()
        return {"status": "error", "message": f"Agent keywords failed: {e}"}


# ── Legacy combined agent endpoint (kept for compatibility) ──

@app.post("/agent")
async def run_agent(req: AgentRequest):
    """AI Agent: Q4 verification + keyword extraction + WikiSearch URLs."""
    print(f"\n{'='*60}")
    print(f"[API] /agent — doc_id={req.doc_id}")
    t_start = time.time()

    doc = doc_store.get(req.doc_id)
    if not doc:
        print(f"[API] ❌ Document {req.doc_id} not found")
        return {"status": "error", "message": "Document not found"}

    middle_data = doc.get("middle_json", {})
    markdown = doc.get("markdown", "")

    try:
        agent.set_llm_provider(req.llm_provider)
        print(f"[API] Running agent analysis (Q4 + keywords, LLM: {req.llm_provider})...")
        result = await asyncio.to_thread(agent.run, middle_data, markdown)
        doc["agent_result"] = result
        doc_store.update(req.doc_id, doc)
        elapsed = time.time() - t_start
        q4_count = result.get("q4_verification", {}).get("q4_count", 0)
        kw_count = len(result.get("keywords", []))
        print(f"[API] ✅ Agent complete in {elapsed:.1f}s. Q4 elements={q4_count}, keywords={kw_count}")
        return result
    except Exception as e:
        print(f"[API] ❌ Agent error: {e}")
        traceback.print_exc()
        return {"status": "error", "message": f"Agent failed: {e}"}


@app.post("/feedback")
async def log_feedback(req: FeedbackRequest):
    """Log user feedback metrics to MLflow."""
    print(f"[API] /feedback — doc_id={req.doc_id}, rating={req.user_rating}")
    try:
        metrics = evaluator.log_metrics(
            doc_id=req.doc_id,
            original_md=req.original_md,
            modified_md=req.modified_md,
            user_rating=req.user_rating,
            download=req.downloaded,
            time_consumed=req.time_consumed,
        )
        return {"status": "success", "metrics": metrics}
    except Exception as e:
        print(f"[API] ❌ Feedback error: {e}")
        return {"status": "error", "message": str(e)}


# ── Q6: HITL — Human-in-the-Loop update ────────────────────

@app.post("/hitl/update")
async def hitl_update(req: HITLUpdateRequest):
    """Update a specific translated block's text (Human-in-the-Loop)."""
    print(f"[API] /hitl/update — doc_id={req.doc_id}, page={req.page_idx}, block={req.block_idx}")
    doc = doc_store.get(req.doc_id)
    if not doc:
        return {"status": "error", "message": "Document not found"}

    translated_middle = doc.get("translated_middle")
    if not translated_middle:
        return {"status": "error", "message": "No translated layout found. Run translation first."}

    try:
        pages = translated_middle.get("pdf_info", [])
        if req.page_idx >= len(pages):
            return {"status": "error", "message": f"Page {req.page_idx} out of range"}

        blocks = pages[req.page_idx].get("para_blocks", [])
        if req.block_idx >= len(blocks):
            return {"status": "error", "message": f"Block {req.block_idx} out of range"}

        block = blocks[req.block_idx]
        # Rewrite the block's content with user-edited text
        block["lines"] = [{
            "bbox": block.get("bbox", [0, 0, 0, 0]),
            "spans": [{
                "bbox": block.get("bbox", [0, 0, 0, 0]),
                "type": "text",
                "content": req.new_text,
                "score": 1.0,
                "translated": True,
                "human_edited": True,
            }]
        }]
        doc_store.update(req.doc_id, doc)
        print(f"[API] ✅ HITL update applied: page {req.page_idx}, block {req.block_idx}")
        return {"status": "success", "message": "Block updated"}
    except Exception as e:
        print(f"[API] ❌ HITL error: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/hitl/blocks/{doc_id}")
async def get_hitl_blocks(doc_id: str):
    """Get all Q4-flagged blocks for HITL editing."""
    doc = doc_store.get(doc_id)
    if not doc:
        return {"status": "error", "message": "Document not found"}

    agent_result = doc.get("agent_result", {})
    q4 = agent_result.get("q4_verification", {})
    translated_middle = doc.get("translated_middle", doc.get("middle_json", {}))

    flagged_blocks = []
    for item in q4.get("results", []):
        if item.get("verdict") == "REVIEW":
            # Find the corresponding block in translated layout
            page_idx = item.get("page", 0)
            pages = translated_middle.get("pdf_info", [])
            if page_idx < len(pages):
                blocks = pages[page_idx].get("para_blocks", [])
                for bi, block in enumerate(blocks):
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            content = span.get("content", "")
                            if content and item.get("content", "")[:30] in content[:50]:
                                flagged_blocks.append({
                                    "page_idx": page_idx,
                                    "block_idx": bi,
                                    "type": block.get("type", "text"),
                                    "original_content": item.get("content", ""),
                                    "current_content": content,
                                    "score": item.get("score", 0),
                                    "suggestion": item.get("suggestion", ""),
                                })
                                break

    return {"status": "success", "flagged_blocks": flagged_blocks, "total": len(flagged_blocks)}


@app.post("/document/{doc_id}/update-markdown")
async def update_document_markdown(doc_id: str, req: UpdateMarkdownRequest):
    print(f"[API] /document/{doc_id}/update-markdown — doc_id={doc_id}")
    doc = doc_store.get(doc_id)
    if not doc:
        return {"status": "error", "message": "Document not found"}
        
    translated_middle = doc.get("translated_middle")
    if not translated_middle:
        translated_middle = doc.get("middle_json")
        
    if not translated_middle:
        return {"status": "error", "message": "No layout data found"}
        
    raw_paras = re.split(r'\n\s*\n', req.markdown.strip())
    paras = [p.strip() for p in raw_paras if p.strip()]
    
    blocks = []
    def _collect(blist):
        for b in blist:
            if "lines" in b:
                blocks.append(b)
            if "blocks" in b:
                _collect(b["blocks"])
                
    for page in translated_middle.get("pdf_info", []):
        all_root_blocks = (
            page.get("preproc_blocks", page.get("para_blocks", [])) + 
            page.get("discarded_blocks", [])
        )
        _collect(all_root_blocks)
        
    for idx, block in enumerate(blocks):
        if idx >= len(paras):
            break
        para_text = paras[idx]
        btype = block.get("type", "text")
        if btype in {"title", "section_title", "heading", "subheading"}:
            para_text = para_text.lstrip("#").strip()
        elif btype in {"interline_equation", "equation"}:
            para_text = para_text.strip("$").strip()
        elif btype == "image_caption":
            para_text = para_text.strip("*").strip()
            
        parts = re.split(r'(\$.*?\$)', para_text)
        spans = []
        bbox = block.get("bbox", [0, 0, 0, 0])
        for part in parts:
            if not part:
                continue
            if part.startswith("$") and part.endswith("$") and len(part) > 2:
                spans.append({
                    "bbox": bbox,
                    "type": "inline_equation",
                    "content": part[1:-1],
                    "score": 1.0,
                    "translated": True,
                    "human_edited": True
                })
            else:
                spans.append({
                    "bbox": bbox,
                    "type": "text",
                    "content": part,
                    "score": 1.0,
                    "translated": True,
                    "human_edited": True
                })
        block["lines"] = [{
            "bbox": bbox,
            "spans": spans
        }]
        
    doc["translated_middle"] = translated_middle
    doc["translated_md"] = req.markdown
    doc_store.update(doc_id, doc)
    return {"status": "success", "message": "Markdown and layout blocks synchronized"}


def _is_quartet_text(obj) -> bool:
    if not isinstance(obj, dict): return False
    return (
        "bbox" in obj and
        obj.get("type") == "text" and
        "content" in obj and
        isinstance(obj.get("score"), (int, float))
    )

def _contains_quartet_recursive(obj) -> bool:
    if _is_quartet_text(obj):
        return True
    if isinstance(obj, dict):
        for v in obj.values():
            if _contains_quartet_recursive(v): return True
    elif isinstance(obj, list):
        for item in obj:
            if _contains_quartet_recursive(item): return True
    return False

def _extract_paragraph_info(block_chain: list) -> tuple:
    parts = []
    eq_map = {}
    eq_idx = 0
    for block in block_chain:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                span_type = span.get("type")
                if span_type == "text" or "score" in span:
                    parts.append(span.get("content", ""))
                elif span_type == "inline_equation":
                    placeholder = f"[EQ_{eq_idx}]"
                    eq_map[placeholder] = span
                    parts.append(placeholder)
                    eq_idx += 1
        parts.append(" ")
    return " ".join(parts).strip(), eq_map

def collect_jobs_for_page(blist):
    jobs = []
    i = 0
    while i < len(blist):
        block = blist[i]
        if block.get("type") == "interline_equation":
            i += 1
            continue
        if _contains_quartet_recursive(block) or block.get("lines"):
            chain = [block]
            j = i + 1
            while j < len(blist) and blist[j].get("merge_prev") is True:
                chain.append(blist[j])
                j += 1
            text, eq_map = _extract_paragraph_info(chain)
            if text.strip():
                jobs.append((chain, text, eq_map))
            if "blocks" in block:
                jobs.extend(collect_jobs_for_page(block["blocks"]))
            i = j
        elif "blocks" in block:
            jobs.extend(collect_jobs_for_page(block["blocks"]))
            i += 1
        else:
            i += 1
    return jobs

@app.post("/agent/apply-corrections")
async def apply_corrections(req: ApplyCorrectionsRequest):
    print(f"[API] /agent/apply-corrections — doc_id={req.doc_id}")
    doc = doc_store.get(req.doc_id)
    if not doc:
        return {"status": "error", "message": "Document not found"}

    middle_data = doc.get("middle_json")
    if not middle_data:
        return {"status": "error", "message": "No layout data found"}

    pages = middle_data.get("pdf_info", [])
    applied_count = 0

    for corr in req.corrections:
        page_idx = corr.page
        orig_text = corr.original.strip()
        corr_text = corr.corrected.strip()
        bbox = corr.bbox

        if page_idx >= len(pages):
            continue

        page = pages[page_idx]
        blocks = page.get("preproc_blocks", page.get("para_blocks", [])) + page.get("discarded_blocks", [])

        # Collect paragraph chains on this page and find matching one
        jobs = collect_jobs_for_page(blocks)
        matched_job = None
        for chain, text, eq_map in jobs:
            if text.strip() == orig_text or orig_text in text.strip() or text.strip() in orig_text:
                matched_job = (chain, text, eq_map)
                break
                
        if matched_job:
            chain, text, eq_map = matched_job
            pieces = []
            for part in re.split(r"(\[EQ_\d+\])", corr_text):
                if not part:
                    continue
                if part in eq_map:
                    pieces.append(eq_map[part])
                else:
                    pieces.append({
                        "bbox": chain[0].get("bbox", [0, 0, 0, 0]),
                        "type": "text",
                        "content": part,
                        "score": 1.0,
                        "human_corrected": True
                    })
            
            # Write back to first block, clear other blocks in the chain
            first_block = chain[0]
            first_block["lines"] = [{
                "bbox": first_block.get("bbox", [0, 0, 0, 0]),
                "spans": pieces
            }]
            for extra_block in chain[1:]:
                extra_block["lines"] = []
                
            applied_count += 1

    doc["middle_json"] = middle_data
    doc_store.update(req.doc_id, doc)
    print(f"[API] Applied {applied_count} human corrections to layout JSON")
    return {"status": "success", "applied_count": applied_count}


@app.post("/auth/register")
async def register(req: UserAuthRequest):
    success = doc_store.register_user(req.username, req.password_hash)
    if success:
        return {"status": "success", "message": "Registered successfully"}
    return JSONResponse({"status": "error", "message": "Username already exists"}, status_code=400)


@app.post("/auth/login")
async def login(req: UserAuthRequest):
    success = doc_store.authenticate_user(req.username, req.password_hash)
    if success:
        return {"status": "success", "message": "Authenticated successfully"}
    return JSONResponse({"status": "error", "message": "Invalid username or password"}, status_code=401)


@app.get("/user/documents")
async def user_documents(x_user: Optional[str] = Header(None)):
    if not x_user:
        return JSONResponse({"status": "error", "message": "User not authenticated"}, status_code=401)
    docs = doc_store.get_user_documents(x_user)
    safe_docs = []
    for d in docs:
        safe_docs.append({
            "doc_id": d.get("doc_id", d.get("_id")),
            "filename": d.get("filename"),
            "num_pages": d.get("num_pages", 0),
            "num_paragraphs": d.get("num_paragraphs", 0),
            "updated_at": d.get("updated_at").isoformat() if isinstance(d.get("updated_at"), datetime) else str(d.get("updated_at")),
            "translated_markdown": d.get("translated_md", "")
        })
    return {"status": "success", "documents": safe_docs}


@app.get("/document/{doc_id}")
async def get_document(doc_id: str):
    doc = doc_store.get(doc_id)
    if not doc:
        return JSONResponse({"status": "error", "message": "Document not found"}, status_code=404)
    safe_doc = {k: v for k, v in doc.items() if k != "pdf_bytes"}
    safe_doc["doc_id"] = doc_id
    return {"status": "success", "document": safe_doc}


@app.post("/document/{doc_id}/save")
async def save_document_to_db(doc_id: str, x_user: Optional[str] = Header(None)):
    doc = doc_store.get(doc_id)
    if not doc:
        return JSONResponse({"status": "error", "message": "Document not found"}, status_code=404)
    doc["save_to_db"] = True
    if x_user:
        doc["user_id"] = x_user
    doc_store.set(doc_id, doc)
    return {"status": "success", "message": "Document saved to database history"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
