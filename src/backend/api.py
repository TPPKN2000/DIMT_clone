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
import uuid
import time
import asyncio
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from .mineru_client import MinerUClient
from .nllb_service import NLLBService
from .agent import AIAgent
from .evaluation import Evaluator
from .pdf_renderer import PDFRenderer
from .mongo_store import MongoDocStore

# ── Services ─────────────────────────────────────────────────
mineru = MinerUClient()
nllb = NLLBService(lazy_load=True)
agent = AIAgent()
evaluator = Evaluator()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start background loading of NLLB model
    print("[API] Starting background NLLB model loading...")
    asyncio.create_task(asyncio.to_thread(nllb.load_model))
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

# Max pages before auto-split for MinerU (API limit: 200 pages)
MAX_MINERU_PAGES = 200


# ── Models ──────────────────────────────────────────────────
class TranslateRequest(BaseModel):
    doc_id: str
    tgt_lang: str = "fra_Latn"

class RenderRequest(BaseModel):
    doc_id: str

class AgentRequest(BaseModel):
    doc_id: str
    llm_provider: str = "deepseek"

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


# ── Helpers ─────────────────────────────────────────────────

def _count_pdf_pages(pdf_path: Path) -> int:
    """Count pages in a PDF using PyMuPDF."""
    import fitz
    doc = fitz.open(str(pdf_path))
    count = len(doc)
    doc.close()
    return count

def _split_pdf(pdf_path: Path, chunk_size: int = MAX_MINERU_PAGES) -> list[Path]:
    """Split a large PDF into chunks of chunk_size pages. Returns list of chunk paths."""
    import fitz
    doc = fitz.open(str(pdf_path))
    total = len(doc)
    if total <= chunk_size:
        doc.close()
        return [pdf_path]

    chunks = []
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size - 1, total - 1)
        chunk_doc = fitz.open()
        chunk_doc.insert_pdf(doc, from_page=start, to_page=end)
        chunk_path = pdf_path.parent / f"{pdf_path.stem}_chunk_{start}_{end}.pdf"
        chunk_doc.save(str(chunk_path))
        chunk_doc.close()
        chunks.append(chunk_path)
        print(f"[API] Split chunk: pages {start}-{end} → {chunk_path.name}")

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


# ── Endpoints ───────────────────────────────────────────────

@app.post("/upload")
async def upload_file(file: UploadFile):
    """Upload PDF → extract via MinerU. Auto-splits PDFs >200 pages."""
    doc_id = str(uuid.uuid4())[:8]
    evaluator.start_inference(doc_id)
    print(f"\n{'='*60}")
    print(f"[API] /upload — doc_id={doc_id}, file={file.filename}")

    # Save uploaded file
    input_dir = Path("input_docs")
    input_dir.mkdir(exist_ok=True)
    save_path = input_dir / file.filename
    content = await file.read()
    save_path.write_bytes(content)
    print(f"[API] Saved {len(content)} bytes → {save_path}")

    # Q8: Auto-split >200-page PDFs
    page_count = _count_pdf_pages(save_path)
    print(f"[API] PDF has {page_count} pages")

    if page_count > MAX_MINERU_PAGES:
        print(f"[API] Large PDF detected ({page_count} pages). Auto-splitting into chunks of {MAX_MINERU_PAGES}...")
        chunks = _split_pdf(save_path)
        all_layouts = []
        all_markdowns = []
        images_dir = None
        extract_dir = None

        for i, chunk_path in enumerate(chunks):
            print(f"[API] Extracting chunk {i+1}/{len(chunks)}: {chunk_path.name}")
            res = await asyncio.to_thread(mineru.extract_from_file, str(chunk_path))
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
            "filename": file.filename,
            "markdown": merged_md,
            "middle_json": merged_layout,
            "images_dir": images_dir,
            "extract_dir": extract_dir,
            "num_pages": page_count,
            "num_paragraphs": _count_paragraphs(merged_layout)
        })
        print(f"[API] ✅ Large PDF extraction complete. {len(all_layouts)} chunks merged.")
        return {
            "status": "success", 
            "doc_id": doc_id, 
            "num_pages": page_count, 
            "num_paragraphs": _count_paragraphs(merged_layout)
        }

    # Normal extraction for ≤200-page PDFs
    print(f"[API] Calling MinerU API...")
    res = await asyncio.to_thread(mineru.extract_from_file, str(save_path))

    if res["status"] == "success":
        m_json = res.get("middle_json")
        p_count = _count_paragraphs(m_json)
        doc_store.set(doc_id, {
            "filename": file.filename,
            "markdown": res["markdown"],
            "middle_json": m_json,
            "images_dir": res.get("images_dir"),
            "extract_dir": res.get("extract_dir"),
            "num_pages": page_count,
            "num_paragraphs": p_count
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
    """Translate document using NLLB — paragraph-level from layout.json with batch translation."""
    print(f"\n{'='*60}")
    print(f"[API] /translate — doc_id={req.doc_id}, tgt_lang={req.tgt_lang}")
    t_start = time.time()

    if not nllb.loaded:
        print("[API] NLLB is still loading. Waiting...")
        await asyncio.to_thread(nllb.wait_for_load)
        if not nllb.loaded:
            print("[API] ⚠️ NLLB failed to load. Translations will be mock.")

    doc = doc_store.get(req.doc_id)
    if not doc:
        print(f"[API] ❌ Document {req.doc_id} not found")
        return {"status": "error", "message": "Document not found"}

    # Set target language
    if req.tgt_lang in NLLBService.SUPPORTED_LANGS:
        nllb.set_target_lang(req.tgt_lang)
        print(f"[API] Target language set to {req.tgt_lang}")

    middle_data = doc.get("middle_json")
    translated_md = ""
    translated_middle = None

    async with gpu_semaphore:
        print("[API] GPU semaphore acquired. Starting translation...")
        if middle_data:
            # Q9: Paragraph-level batch translation from layout.json
            page_count = len(middle_data.get("pdf_info", []))
            print(f"[API] Translating layout.json ({page_count} pages)...")
            translated_middle = await asyncio.to_thread(
                nllb.translate_middle_json, middle_data
            )
            doc["translated_middle"] = translated_middle
#             print(f"[API] Layout translation complete. Now translating markdown...")
#             # Also generate translated markdown
#             translated_md = await asyncio.to_thread(
#                 nllb.translate_markdown, doc["markdown"]
#             )
#             print(f"[API] Markdown translation complete.")
        else:
            pass
#             # Fallback: markdown-only translation
#             print("[API] No layout.json found. Falling back to markdown-only translation...")
#             translated_md = await asyncio.to_thread(
#                 nllb.translate_markdown, doc["markdown"]
#             )

    elapsed = time.time() - t_start
    doc["translated_md"] = translated_md
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
    input_dir = Path("input_docs")
    origin_pdf_path = input_dir / doc["filename"]
    if not origin_pdf_path.exists():
        print(f"[API] ❌ Origin PDF not found: {origin_pdf_path}")
        return {"status": "error", "message": f"Origin PDF not found: {origin_pdf_path}"}

    images_dir = doc.get("images_dir")
    renderer = PDFRenderer(images_dir=images_dir)

    output_path = OUTPUT_DIR / f"{req.doc_id}_translated.pdf"
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
    doc_store.update(req.doc_id, doc)
    print(f"[API] ✅ PDF rendered in {elapsed:.1f}s → {output_path}")
    return {"status": "success", "pdf_path": str(output_path)}


@app.get("/stream-pdf/{doc_id}")
async def stream_pdf(doc_id: str):
    """Stream the rendered translated PDF for preview."""
    doc = doc_store.get(doc_id)
    if not doc or "pdf_path" not in doc:
        return JSONResponse({"status": "error", "message": "PDF not found"}, status_code=404)
    pdf_path = Path(doc["pdf_path"])
    if not pdf_path.exists():
        return JSONResponse({"status": "error", "message": "PDF file missing"}, status_code=404)
    return FileResponse(
        str(pdf_path),
        media_type="application/pdf",
    )


@app.get("/download/{doc_id}")
async def download_pdf(doc_id: str):
    """Download the rendered translated PDF."""
    doc = doc_store.get(doc_id)
    if not doc or "pdf_path" not in doc:
        return JSONResponse({"status": "error", "message": "PDF not found"}, status_code=404)
    return FileResponse(
        doc["pdf_path"],
        media_type="application/pdf",
        filename=f"{Path(doc.get('filename', 'translated')).stem}_translated.pdf",
    )


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
        wiki_urls = agent.get_keyword_wiki_urls(keywords)

        # Merge into existing agent_result
        agent_result = doc.get("agent_result", {})
        agent_result["keywords"] = keywords
        agent_result["wiki_references"] = wiki_urls
        doc["agent_result"] = agent_result
        doc_store.update(req.doc_id, doc)

        elapsed = time.time() - t_start
        print(f"[API] ✅ Keywords complete in {elapsed:.1f}s. keywords={len(keywords)}")
        return {"status": "success", "keywords": keywords, "wiki_references": wiki_urls}
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
