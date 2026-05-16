"""
Streamlit Frontend — DIMT Document Translation Pipeline.

Features:
- Upload PDF → Extract via MinerU → Translate via NLLB
- Optional "Human Check" mode: Q4 verification + HITL editing before translation
- Rendered markdown preview + editable text area (notebook-style blocks)
- Download translated PDF (rendered from layout.json + images)
- References tab: keyword WikiSearch links
- Q6: Human-in-the-Loop (HITL) editable UI for flagged elements
- User feedback with MLflow metrics logging
"""

import subprocess
import sys
import time
import concurrent.futures
import streamlit as st
import requests

st.set_page_config(layout="wide", page_title="DIMT — Document Translation", page_icon="📄")

API_BASE = "http://localhost:8000"

# Generous timeout for long operations (translation can take minutes)
LONG_TIMEOUT = 600  # 10 minutes


# ── Backend startup (HF Spaces: only one process allowed) ───
@st.cache_resource
def start_backend():
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn",
         "src.backend.api:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    # Poll until backend is up (max 180s for NLLB to load)
    for _ in range(180):
        try:
            requests.get(f"{API_BASE}/docs", timeout=1)
            return proc
        except Exception:
            time.sleep(1)
    return proc  # return anyway, let requests fail naturally

start_backend()

# ── rest of file unchanged below ────────────────────────────

st.title("📄 Document Intelligent Machine Translation")
st.caption("PDF → MinerU extraction → NLLB translation → Translated PDF + Markdown")

# ── Session State ───────────────────────────────────────────
for key in ["doc_id", "original_markdown", "translated_markdown",
            "agent_result", "pdf_ready", "has_middle", "hitl_blocks",
            "q4_result", "human_check", "q4_confirmed",
            "keywords_result", "num_pages", "num_paragraphs"]:
    if key not in st.session_state:
        st.session_state[key] = 0 if "num_" in key else None

# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    uploaded_file = st.file_uploader("Upload PDF document", type=["pdf"])
    st.slider("Max convert pages", 1, 200, 200)
    st.selectbox("MinerU Engine", ["vlm", "pipeline"])
    target_lang = st.selectbox("Target Language", [
        "French (fra_Latn)",
        "German (deu_Latn)",
    ])
    tgt_lang_code = "fra_Latn" if "fra" in target_lang else "deu_Latn"
    agent_llm = st.selectbox("Agent LLM", ["DeepSeek", "Gemini"])
    agent_llm_code = "gemini" if agent_llm == "Gemini" else "deepseek"

    st.divider()
    human_check = st.checkbox("🔍 Human Check", value=False,
                              help="Enable Q4 verification & HITL editing before translation")
    col1, col2 = st.columns(2)
    convert_btn = col1.button("🚀 Convert", use_container_width=True)
    clear_btn = col2.button("🗑️ Clear", use_container_width=True)

if clear_btn:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


# ── Helper: run translate + render sequentially ─────────────
def _run_translate_and_render(doc_id, tgt_lang, num_paras=0, num_pages=0):
    """Called in a thread — translate then render with dynamic timeout."""
    # Based on observation: ~18s/paragraph and ~2s/page. 
    # We use 20s and 5s + 600s buffer for safety.
    dynamic_timeout = (num_paras * 20) + (num_pages * 5) + 600
    if dynamic_timeout < LONG_TIMEOUT: 
        dynamic_timeout = LONG_TIMEOUT

    tr = requests.post(
        f"{API_BASE}/translate",
        json={"doc_id": doc_id, "tgt_lang": tgt_lang},
        timeout=dynamic_timeout,
    )
    tr_data = tr.json() if tr.status_code == 200 else {}
    if tr_data.get("status") != "success":
        return {"status": "error", "step": "translate", "data": tr_data}

    # Only render if layout.json exists
    if tr_data.get("has_middle_json"):
        rr = requests.post(
            f"{API_BASE}/render-pdf",
            json={"doc_id": doc_id},
            timeout=dynamic_timeout,
        )
        rr_data = rr.json() if rr.status_code == 200 else {}
        return {"status": "success", "translate": tr_data, "render": rr_data}

    return {"status": "success", "translate": tr_data, "render": None}


def _run_keywords(doc_id, llm_provider="deepseek"):
    """Called in a thread — extract keywords (shorter timeout, not GPU-bound)."""
    res = requests.post(
        f"{API_BASE}/agent/keywords",
        json={"doc_id": doc_id, "llm_provider": llm_provider},
        timeout=120,
    )
    return res.json() if res.status_code == 200 else {}


# ── Pipeline Execution ─────────────────────────────────────
if convert_btn and uploaded_file:
    # Step 1: Upload & Extract
    with st.spinner("📤 Uploading & extracting with MinerU..."):
        try:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            res = requests.post(f"{API_BASE}/upload", files=files, timeout=LONG_TIMEOUT)
            if res.status_code == 200:
                data = res.json()
                if data.get("status") == "success":
                    st.session_state.doc_id = data["doc_id"]
                    st.session_state.num_pages = data.get("num_pages", 0)
                    st.session_state.num_paragraphs = data.get("num_paragraphs", 0)
                    st.session_state.human_check = human_check
                    st.session_state.q4_confirmed = False
                    st.success(f"✅ Extraction complete (ID: {data['doc_id']}) — "
                               f"Found {st.session_state.num_pages} pages, {st.session_state.num_paragraphs} paragraphs")
                else:
                    st.error(f"❌ Extraction failed: {data.get('message')}")
            else:
                st.error(f"❌ API error: {res.status_code} — {res.text[:200]}")
        except requests.exceptions.Timeout:
            st.error("❌ Upload timed out. The PDF may be too large.")
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend. Is the server running?")

    # Step 2: If Human Check → run Q4 verification and stop (wait for confirm)
    if st.session_state.human_check and st.session_state.doc_id:
        with st.spinner("🔍 Running Q4 verification (Agent)..."):
            try:
                res = requests.post(
                    f"{API_BASE}/agent/verify",
                    json={"doc_id": st.session_state.doc_id, "llm_provider": agent_llm_code},
                    timeout=LONG_TIMEOUT,
                )
                if res.status_code == 200:
                    result = res.json()
                    if result.get("status") == "success":
                        st.session_state.q4_result = result.get("q4_verification", {})
                        st.success("✅ Q4 verification complete — review flagged elements below")
                    else:
                        st.error(f"❌ Q4 verification failed: {result.get('message')}")
            except requests.exceptions.Timeout:
                st.error("❌ Q4 verification timed out.")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend.")
        # Pipeline pauses here — user must review HITL and click Confirm

    # Step 2b: If NOT Human Check → run translate+render+keywords in parallel
    if not st.session_state.human_check and st.session_state.doc_id:
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        
        dynamic_timeout = (st.session_state.num_paragraphs * 20) + (st.session_state.num_pages * 5) + 600
        if dynamic_timeout < LONG_TIMEOUT: 
            dynamic_timeout = LONG_TIMEOUT

        fut_pipeline = executor.submit(
            _run_translate_and_render,
            st.session_state.doc_id, tgt_lang_code,
            st.session_state.num_paragraphs, st.session_state.num_pages
        )
        fut_keywords = executor.submit(
            _run_keywords, st.session_state.doc_id, agent_llm_code
        )

        # Pipeline A: Translate + Render (critical path)
        with st.spinner("🔄 Translating & rendering PDF... (this may take several minutes)"):
            try:
                pipeline_result = fut_pipeline.result(timeout=dynamic_timeout)
                if pipeline_result.get("status") == "success":
                    tr_data = pipeline_result.get("translate", {})
                    st.session_state.translated_markdown = tr_data.get("translated_markdown", "")
                    st.session_state.has_middle = tr_data.get("has_middle_json", False)
                    rr_data = pipeline_result.get("render")
                    if rr_data and rr_data.get("status") == "success":
                        st.session_state.pdf_ready = True
                    st.success("✅ Translation & PDF rendering complete")
                else:
                    st.error(f"❌ Pipeline failed at {pipeline_result.get('step', 'unknown')}")
            except concurrent.futures.TimeoutError:
                st.error(f"❌ Translation pipeline timed out after {dynamic_timeout}s. The document might be too large.")
            except Exception as e:
                st.error(f"❌ Translation pipeline error: {e}")

        # Pipeline B: Keywords (independent, shorter timeout)
        with st.spinner("📚 Extracting keywords & references..."):
            try:
                kw_result = fut_keywords.result(timeout=120)
                if kw_result.get("status") == "success":
                    st.session_state.keywords_result = kw_result
                    st.success("✅ Keywords extracted")
            except concurrent.futures.TimeoutError:
                st.info("📚 Keywords extraction timed out. Use the retry button in References tab.")
            except Exception as e:
                st.warning(f"⚠️ Keywords error: {e}")

        executor.shutdown(wait=False)


# ── HITL Review & Confirm (only when Human Check is on) ────
if (st.session_state.human_check
    and st.session_state.q4_result
    and not st.session_state.q4_confirmed):

    st.header("🔍 Q4 Verification — Review Flagged Elements")
    q4 = st.session_state.q4_result

    st.info(f"Found **{q4.get('q4_count', 0)}** elements in the bottom 25th percentile. "
            f"Threshold: {q4.get('threshold', 'N/A')}")

    for i, item in enumerate(q4.get("results", [])[:20]):
        icon = "✅" if item.get("verdict") == "OK" else "⚠️"
        score_val = item.get('score', 'N/A')
        score_str = f"{score_val:.3f}" if isinstance(score_val, (int, float)) else str(score_val)

        with st.expander(
            f"{icon} [{item.get('index')}] score={score_str} → {item.get('verdict', 'N/A')} "
            f"| `{item.get('content', '')[:60]}`",
            expanded=(item.get("verdict") == "REVIEW"),
        ):
            if item.get("suggestion"):
                st.caption(f"💡 {item['suggestion']}")

    st.divider()
    if st.button("✅ Confirm & Continue Pipeline", use_container_width=True, type="primary"):
        st.session_state.q4_confirmed = True
        st.rerun()


# ── After Confirm: run translate+render+keywords in parallel
if (st.session_state.human_check
    and st.session_state.q4_confirmed
    and st.session_state.doc_id
    and st.session_state.translated_markdown is None):

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    
    dynamic_timeout = (st.session_state.num_paragraphs * 20) + (st.session_state.num_pages * 5) + 600
    if dynamic_timeout < LONG_TIMEOUT: 
        dynamic_timeout = LONG_TIMEOUT

    fut_pipeline = executor.submit(
        _run_translate_and_render,
        st.session_state.doc_id, tgt_lang_code,
        st.session_state.num_paragraphs, st.session_state.num_pages
    )
    fut_keywords = executor.submit(
        _run_keywords, st.session_state.doc_id, agent_llm_code
    )

    # Pipeline A: Translate + Render (critical path)
    with st.spinner("🔄 Translating & rendering PDF... (this may take several minutes)"):
        try:
            pipeline_result = fut_pipeline.result(timeout=dynamic_timeout)
            if pipeline_result.get("status") == "success":
                tr_data = pipeline_result.get("translate", {})
                st.session_state.translated_markdown = tr_data.get("translated_markdown", "")
                st.session_state.has_middle = tr_data.get("has_middle_json", False)
                rr_data = pipeline_result.get("render")
                if rr_data and rr_data.get("status") == "success":
                    st.session_state.pdf_ready = True
                st.success("✅ Translation & PDF rendering complete")
            else:
                st.error(f"❌ Pipeline failed at {pipeline_result.get('step', 'unknown')}")
        except concurrent.futures.TimeoutError:
            st.error(f"❌ Translation pipeline timed out after {dynamic_timeout}s. The document might be too large.")
        except Exception as e:
            st.error(f"❌ Translation pipeline error: {e}")

    # Pipeline B: Keywords (independent, shorter timeout)
    with st.spinner("📚 Extracting keywords & references..."):
        try:
            kw_result = fut_keywords.result(timeout=120)
            if kw_result.get("status") == "success":
                st.session_state.keywords_result = kw_result
                st.success("✅ Keywords extracted")
        except concurrent.futures.TimeoutError:
            st.info("📚 Keywords extraction timed out. Use the retry button in References tab.")
        except Exception as e:
            st.warning(f"⚠️ Keywords error: {e}")

    executor.shutdown(wait=False)

    # Also fetch HITL blocks after translation
    if st.session_state.has_middle:
        try:
            hitl_res = requests.get(
                f"{API_BASE}/hitl/blocks/{st.session_state.doc_id}",
                timeout=30,
            )
            if hitl_res.status_code == 200:
                st.session_state.hitl_blocks = hitl_res.json().get("flagged_blocks", [])
        except Exception:
            pass


# ── Results Display ────────────────────────────────────────
if st.session_state.translated_markdown is not None:
    st.header("📊 Results")

    # Build tabs dynamically based on human_check
    if st.session_state.human_check:
        tab_names = ["📝 Translated Markdown", "✏️ Editable Text",
                     "🔧 HITL Verification", "📚 References", "📥 Downloads"]
        tabs = st.tabs(tab_names)
        tab_md, tab_edit, tab_hitl, tab_ref, tab_dl = tabs
    else:
        tab_names = ["📝 Translated Markdown", "✏️ Editable Text",
                     "📚 References", "📥 Downloads"]
        tabs = st.tabs(tab_names)
        tab_md, tab_edit, tab_ref, tab_dl = tabs
        tab_hitl = None

    start_edit_time = time.time()

    with tab_md:
        st.markdown(st.session_state.translated_markdown, unsafe_allow_html=True)

    with tab_edit:
        st.subheader("✏️ Editable Translated Blocks")
        st.caption("Each block corresponds to a paragraph in the translated layout. "
                   "Edit as needed, then submit feedback below.")

        # Show as notebook-style blocks if we have translated_middle data
        edited_text = st.text_area(
            "Edit translated markdown (feedback loop)",
            st.session_state.translated_markdown,
            height=500,
        )

        st.subheader("📊 Submit Evaluation")
        rating = st.slider("Rate the translation quality (1-5)", 1, 5, 3)
        downloaded = st.checkbox("Downloaded generated file?")
        if st.button("📤 Submit Feedback & Save Metrics"):
            time_consumed = time.time() - start_edit_time
            try:
                res = requests.post(f"{API_BASE}/feedback", json={
                    "doc_id": st.session_state.doc_id,
                    "original_md": st.session_state.translated_markdown,
                    "modified_md": edited_text,
                    "user_rating": rating,
                    "downloaded": downloaded,
                    "time_consumed": time_consumed,
                }, timeout=30)
                if res.status_code == 200:
                    st.success(f"✅ Metrics logged: {res.json().get('metrics', {})}")
                else:
                    st.error(f"❌ Feedback error: {res.status_code}")
            except Exception as e:
                st.error(f"❌ Feedback error: {e}")

    # HITL Verification Tab (only if human_check)
    if tab_hitl is not None:
        with tab_hitl:
            st.subheader("🔧 Human-in-the-Loop Verification")
            st.caption("Edit flagged translations below and re-render the PDF.")

            if st.session_state.hitl_blocks:
                blocks = st.session_state.hitl_blocks
                st.info(f"Found **{len(blocks)}** flagged elements that need review.")

                for i, block in enumerate(blocks):
                    with st.expander(
                        f"⚠️ Page {block['page_idx']+1}, Block {block['block_idx']} "
                        f"(score: {block.get('score', 0):.3f}, type: {block.get('type', 'text')})",
                        expanded=(i < 3),
                    ):
                        st.caption(f"💡 Suggestion: {block.get('suggestion', 'Manual review recommended')}")

                        st.text_area(
                            "Original content (read-only)",
                            block.get("original_content", ""),
                            height=80,
                            disabled=True,
                            key=f"hitl_orig_{i}",
                        )

                        new_text = st.text_area(
                            "Current translation (edit below)",
                            block.get("current_content", ""),
                            height=100,
                            key=f"hitl_edit_{i}",
                        )

                        if st.button(f"💾 Save Edit", key=f"hitl_save_{i}"):
                            try:
                                res = requests.post(f"{API_BASE}/hitl/update", json={
                                    "doc_id": st.session_state.doc_id,
                                    "page_idx": block["page_idx"],
                                    "block_idx": block["block_idx"],
                                    "new_text": new_text,
                                }, timeout=10)
                                if res.status_code == 200 and res.json().get("status") == "success":
                                    st.success("✅ Block updated!")
                                else:
                                    st.error(f"❌ Update failed: {res.json().get('message', 'Unknown error')}")
                            except Exception as e:
                                st.error(f"❌ Error: {e}")

                st.divider()
                if st.button("🔄 Re-render PDF with edits", use_container_width=True):
                    with st.spinner("📄 Re-rendering PDF with your edits..."):
                        try:
                            res = requests.post(
                                f"{API_BASE}/render-pdf",
                                json={"doc_id": st.session_state.doc_id},
                                timeout=LONG_TIMEOUT,
                            )
                            if res.status_code == 200 and res.json().get("status") == "success":
                                st.session_state.pdf_ready = True
                                st.success("✅ PDF re-rendered with your edits!")
                            else:
                                st.error(f"❌ Re-render failed: {res.json().get('message')}")
                        except Exception as e:
                            st.error(f"❌ Re-render error: {e}")
            else:
                if st.session_state.q4_result:
                    st.success("✅ No flagged elements — all translations look good!")
                else:
                    st.info("No Q4 verification data available.")

    # References Tab
    with tab_ref:
        st.subheader("📚 Keywords & Wikipedia References")
        kw_result = st.session_state.keywords_result
        if kw_result and kw_result.get("status") == "success":
            keywords = kw_result.get("keywords", [])
            wiki = kw_result.get("wiki_references", [])
            if wiki:
                for ref in wiki:
                    urls = []
                    if ref.get("url"):
                        urls.append(f"[Target Lang]({ref['url']})")
                    if ref.get("en_url"):
                        urls.append(f"[English]({ref['en_url']})")
                    if urls:
                        links = " | ".join(urls)
                        st.markdown(f"- **{ref['keyword']}** → {links}")
            elif keywords:
                st.write(", ".join(keywords))
            else:
                st.info("No keywords extracted")
        else:
            st.info("Keywords not yet available.")
            if st.session_state.doc_id and st.button("🔄 Retry Keywords Extraction"):
                with st.spinner("📚 Extracting keywords..."):
                    try:
                        res = requests.post(
                            f"{API_BASE}/agent/keywords",
                            json={"doc_id": st.session_state.doc_id, "llm_provider": agent_llm_code},
                            timeout=120,
                        )
                        if res.status_code == 200:
                            result = res.json()
                            if result.get("status") == "success":
                                st.session_state.keywords_result = result
                                st.rerun()
                    except Exception as e:
                        st.error(f"❌ Keywords error: {e}")

    # Downloads Tab
    with tab_dl:
        st.subheader("📥 Download & Preview")

        if st.session_state.pdf_ready and st.session_state.doc_id:
            try:
                pdf_res = requests.get(
                    f"{API_BASE}/stream-pdf/{st.session_state.doc_id}",
                    timeout=30,
                )
                if pdf_res.status_code == 200:
                    import base64
                    pdf_bytes = pdf_res.content

                    st.divider()
                    st.subheader("👁️ PDF Preview")

                    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
                    st.markdown(pdf_display, unsafe_allow_html=True)

                    st.divider()
                    st.download_button(
                        "💾 Save Translated PDF",
                        pdf_bytes,
                        file_name=f"{st.session_state.doc_id}_translated.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                else:
                    st.warning("PDF stream not available")
            except Exception as e:
                st.error(f"PDF preview error: {e}")
        else:
            st.info("PDF will be available after conversion with layout.json data")