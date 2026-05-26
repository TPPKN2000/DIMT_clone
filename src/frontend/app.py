"""

Streamlit Frontend — DIMT Document Translation Pipeline.



Chạy riêng biệt với backend:

  Terminal 1: uv run uvicorn src.backend.api:app --port 8000

  Terminal 2: uv run streamlit run src/frontend/app.py



Features:

- Upload PDF → Extract via MinerU → Translate via NLLB

- Optional "Human Check" mode: Q4 verification + HITL editing before translation

- Rendered markdown preview + editable text area

- Download translated PDF

- References tab: keyword WikiSearch links

- User feedback with MLflow metrics logging

"""



import time

import concurrent.futures

import streamlit as st

import requests


def check_cooldown(action_name: str, cooldown_seconds: float = 1.5) -> bool:
    """Check if the action is allowed, enforcing a cooldown to prevent duplicate clicks."""
    now = time.time()
    last_time = st.session_state.get(f"last_time_{action_name}", 0.0)
    if now - last_time < cooldown_seconds:
        return False
    st.session_state[f"last_time_{action_name}"] = now
    return True


st.set_page_config(layout="wide", page_title="DIMT — Document Translation", page_icon="📄")



# ── Backend URL — override via env var if needed ─────────────

import os

API_BASE = os.environ.get("DIMT_API_URL", "http://127.0.0.1:8000")



LONG_TIMEOUT = 600  # 10 minutes



# ── Language options ──────────────────────────────────────────

LANG_OPTIONS = [

    "French (fra_Latn)",

    "German (deu_Latn)",

]



LANG_CODE_MAP = {

    "French (fra_Latn)":     "fra_Latn",

    "German (deu_Latn)":     "deu_Latn",

}





# ── Backend connectivity check ────────────────────────────────

def _check_backend() -> bool:

    try:

        r = requests.get(f"{API_BASE}/docs", timeout=5)

        return r.status_code == 200

    except Exception:

        return False



st.title("📄 Document Intelligent Machine Translation")

st.caption("PDF → MinerU extraction → NLLB translation → Translated PDF + Markdown")



# ── Backend status banner ─────────────────────────────────────
if "backend_ok" not in st.session_state:
    st.session_state["backend_ok"] = False

if not st.session_state["backend_ok"]:
    if _check_backend():
        st.session_state["backend_ok"] = True
    else:
        st.error(
            f"❌ Backend không kết nối được tại `{API_BASE}`. "
            "Hãy chạy backend trước:\n\n"
            "```bash\nuv run uvicorn src.backend.api:app --port 8000\n```"
        )
        st.stop()



# ── Session State ───────────────────────────────────────────
for key in ["doc_id", "original_markdown", "translated_markdown",
            "agent_result", "pdf_ready", "has_middle", "hitl_blocks",
            "q4_result", "human_check", "q4_confirmed",
            "keywords_result", "num_pages", "num_paragraphs", "pdf_path",
            "user", "auth_token", "selected_history_doc", "pdf_version"]:
    if key not in st.session_state:
        st.session_state[key] = 0 if "num_" in key or "version" in key else None

if "save_to_db" not in st.session_state:
    st.session_state["save_to_db"] = False

# FIFO display mode: only one exclusive section visible at a time.
# Modes: "history" | "q4" | "results" | None
if "display_mode" not in st.session_state:
    st.session_state["display_mode"] = None

# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    
    # 🔐 User Account Panel
    st.markdown("### 🔐 User Account")
    if not st.session_state.user:
        auth_mode = st.radio("Authentication Mode", ["Login", "Sign Up"], label_visibility="collapsed")
        user_input = st.text_input("Username", key="auth_username")
        pass_input = st.text_input("Password", type="password", key="auth_password")
        
        # Hash password in frontend using hashlib.sha256
        import hashlib
        pass_hash = hashlib.sha256(pass_input.encode()).hexdigest() if pass_input else ""
        
        if auth_mode == "Login":
            if st.button("🔓 Sign In", use_container_width=True):
                if not user_input or not pass_input:
                    st.error("Please enter username and password.")
                else:
                    try:
                        res = requests.post(f"{API_BASE}/auth/login", json={
                            "username": user_input,
                            "password_hash": pass_hash
                        }, timeout=10)
                        if res.status_code == 200:
                            st.session_state.user = user_input
                            st.session_state.auth_token = user_input
                            st.success("Successfully logged in!")
                            st.rerun()
                        else:
                            st.error(res.json().get("message", "Login failed."))
                    except Exception as e:
                        st.error(f"Error connecting to backend: {e}")
        else:
            if st.button("📝 Register", use_container_width=True):
                if not user_input or not pass_input:
                    st.error("Please enter username and password.")
                else:
                    try:
                        res = requests.post(f"{API_BASE}/auth/register", json={
                            "username": user_input,
                            "password_hash": pass_hash
                        }, timeout=10)
                        if res.status_code == 200:
                            st.success("Registration successful! You can now log in.")
                        else:
                            st.error(res.json().get("message", "Registration failed."))
                    except Exception as e:
                        st.error(f"Error connecting to backend: {e}")
    else:
        st.write(f"Logged in as: **{st.session_state.user}**")
        if st.button("🔒 Sign Out", use_container_width=True):
            st.session_state.user = None
            st.session_state.auth_token = None
            st.session_state.selected_history_doc = None
            st.session_state["display_mode"] = None
            for key in ["doc_id", "original_markdown", "translated_markdown", "agent_result", "pdf_ready", "has_middle", "hitl_blocks", "q4_result", "keywords_result"]:
                st.session_state[key] = None
            st.rerun()
            
    st.divider()
    st.markdown("**PDF Document**")
    
    tkinter_available = True
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        tkinter_available = False

    if tkinter_available:
        col_select, col_clear = st.columns([3, 1])
        if col_select.button("📁 Select PDF File", use_container_width=True):
            try:
                root = tk.Tk()
                root.withdraw()
                root.wm_attributes('-topmost', 1)
                selected = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
                root.destroy()
                if selected:
                    st.session_state.pdf_path = selected
                    st.rerun()
            except Exception as e:
                st.error(f"Failed to open file dialog: {e}. Please enter path manually.")
        if col_clear.button("❌", use_container_width=True):
            st.session_state.pdf_path = None
            st.rerun()

    pdf_path_input = st.text_input(
        "PDF File Path", 
        key="pdf_path", 
        placeholder="/path/to/document.pdf"
    )

    translation_model = st.selectbox(
        "Translation Model",
        ["marianMT", "nllb"],
        index=0,
    )

    target_lang = st.selectbox(
        "Target Language",
        LANG_OPTIONS,
        index=0,
    )
    tgt_lang_code = LANG_CODE_MAP[target_lang]

    agent_llm = st.selectbox("Agent LLM", ["Gemini", "GPT"])
    agent_llm_code = "gemini" if agent_llm == "Gemini" else "gpt"

    st.divider()
    human_check = st.checkbox("🔍 Human Check", value=False,
                               help="Enable Q4 verification & HITL editing before translation")
    col1, col2 = st.columns(2)
    convert_btn = col1.button("🚀 Convert", use_container_width=True)
    clear_btn = col2.button("🗑️ Clear", use_container_width=True)

if clear_btn:
    if not check_cooldown("action_clear", 1.5):
        st.stop()
    keys_to_clear = [
        "doc_id", "original_markdown", "translated_markdown",
        "agent_result", "pdf_ready", "has_middle", "hitl_blocks",
        "q4_result", "q4_confirmed", "keywords_result",
        "num_pages", "num_paragraphs", "selected_history_doc", "pdf_version"
    ]
    for key in keys_to_clear:
        if key in st.session_state:
            st.session_state[key] = 0 if "num_" in key or "version" in key else None
    st.session_state["save_to_db"] = False
    st.session_state["display_mode"] = None  # Reset display mode on clear
    st.rerun()


# ── FIFO Display Mode Logic ───────────────────────────────────
# Determine what *should* be displayed based on current state,
# then update display_mode using FIFO (latest event wins).
is_q4_active = (
    st.session_state.human_check 
    and st.session_state.q4_result is not None 
    and not st.session_state.q4_confirmed
)
is_translating = (
    st.session_state.doc_id is not None 
    and st.session_state.translated_markdown is None 
    and not is_q4_active
)
has_results = st.session_state.translated_markdown is not None

# FIFO priority: most recently triggered state wins.
# Order of checks matters — later checks override earlier ones.
if has_results:
    st.session_state["display_mode"] = "results"
elif is_q4_active:
    st.session_state["display_mode"] = "q4"
elif is_translating:
    st.session_state["display_mode"] = "translating"
elif st.session_state.doc_id is None and st.session_state.user:
    # No active document, show history
    st.session_state["display_mode"] = "history"

current_mode = st.session_state.get("display_mode")

if current_mode == "translating":
    st.info("🔄 Translating document, please wait...")

# ── Render History List (only when display_mode == "history") ──
if current_mode == "history":
    try:
        headers = {"X-User": st.session_state.user}
        res = requests.get(f"{API_BASE}/user/documents", headers=headers, timeout=10)
        if res.status_code == 200:
            history_docs = res.json().get("documents", [])
            if history_docs:
                st.subheader("📚 Saved Documents History (Click to preview)")
                for doc in history_docs[:10]:
                    doc_id = doc.get("doc_id")
                    doc_title = doc.get("filename") or f"Document {doc_id}"
                    if len(doc_title) > 60:
                        doc_title = "..." + doc_title[-57:]
                    updated_time = doc.get("updated_at", "")[:19].replace("T", " ")
                    btn_label = f"📄 {doc_title}  |  Pages: {doc.get('num_pages')}  |  Updated: {updated_time}"
                    
                    if st.button(btn_label, key=f"hist_{doc_id}", use_container_width=True):
                        if check_cooldown(f"action_hist_{doc_id}", 1.5):
                            try:
                                doc_res = requests.get(f"{API_BASE}/document/{doc_id}", timeout=10)
                                if doc_res.status_code == 200:
                                    doc_detail = doc_res.json().get("document", {})
                                    st.session_state.doc_id = doc_id
                                    st.session_state.translated_markdown = doc_detail.get("translated_markdown", "")
                                    st.session_state.original_markdown = doc_detail.get("markdown", "")
                                    st.session_state.pdf_ready = True
                                    st.session_state.selected_history_doc = doc_id
                                    st.session_state.save_to_db = True
                                    # Load saved keywords from agent_result
                                    agent_result = doc_detail.get("agent_result", {})
                                    kw_list = agent_result.get("keywords", [])
                                    if kw_list:
                                        st.session_state.keywords_result = {"status": "success", "keywords": kw_list}
                                    st.session_state["display_mode"] = "results"  # Switch to results mode
                                    st.rerun()
                                else:
                                    st.error("Failed to load document details.")
                            except Exception as e:
                                st.error(f"Error: {e}")
            else:
                st.info("No saved documents found in database. Enable 'Save to Database History' in sidebar and translate a PDF to start.")
        else:
            st.error("Failed to retrieve document history.")
    except Exception as e:
        st.warning(f"Could not connect to fetch history: {e}")





# ── Helper: run translate + render sequentially ─────────────

def _run_translate_and_render(doc_id, tgt_lang, model_choice, num_paras=0, num_pages=0):

    """Called in a thread — translate then render with dynamic timeout."""

    dynamic_timeout = (num_paras * 20) + (num_pages * 5) + 600

    if dynamic_timeout < LONG_TIMEOUT:

        dynamic_timeout = LONG_TIMEOUT



    tr = requests.post(

        f"{API_BASE}/translate",

        json={"doc_id": doc_id, "tgt_lang": tgt_lang, "model_choice": model_choice},

        timeout=dynamic_timeout,

    )

    tr_data = tr.json() if tr.status_code == 200 else {}

    if tr_data.get("status") != "success":

        return {"status": "error", "step": "translate", "data": tr_data}



    if tr_data.get("has_middle_json"):

        rr = requests.post(

            f"{API_BASE}/render-pdf",

            json={"doc_id": doc_id},

            timeout=dynamic_timeout,

        )

        rr_data = rr.json() if rr.status_code == 200 else {}

        return {"status": "success", "translate": tr_data, "render": rr_data}



    return {"status": "success", "translate": tr_data, "render": None}





def _run_keywords(doc_id, llm_provider="gemini"):

    """Called in a thread — extract keywords (shorter timeout, not GPU-bound)."""

    res = requests.post(

        f"{API_BASE}/agent/keywords",

        json={"doc_id": doc_id, "llm_provider": llm_provider},

        timeout=120,

    )

    return res.json() if res.status_code == 200 else {}





def _fetch_rendered_pdf(doc_id):

    """Return rendered PDF bytes for preview/download, rendering once if needed."""

    pdf_res = requests.get(f"{API_BASE}/stream-pdf/{doc_id}", timeout=30)

    if pdf_res.status_code == 200:

        return pdf_res.content, None



    render_res = requests.post(

        f"{API_BASE}/render-pdf",

        json={"doc_id": doc_id},

        timeout=LONG_TIMEOUT,

    )

    render_data = render_res.json() if render_res.status_code == 200 else {}

    if render_data.get("status") != "success":

        return None, render_data.get("message", "PDF stream not available")



    pdf_res = requests.get(f"{API_BASE}/stream-pdf/{doc_id}", timeout=30)

    if pdf_res.status_code == 200:

        return pdf_res.content, None

    return None, "Rendered PDF file is not available for streaming"





# ── Pipeline Execution ─────────────────────────────────────

if convert_btn and st.session_state.pdf_path:
    if not check_cooldown("action_convert", 1.5):
        st.stop()
    # Auto-return from history if viewing a saved doc
    if st.session_state.selected_history_doc:
        st.session_state.doc_id = None
        st.session_state.translated_markdown = None
        st.session_state.original_markdown = None
        st.session_state.pdf_ready = False
        st.session_state.selected_history_doc = None
        st.session_state.keywords_result = None
        st.session_state.q4_result = None
        st.session_state.hitl_blocks = []

    # Step 1: Upload & Extract

    with st.spinner("📤 Uploading & extracting with MinerU..."):

        try:
            pdf_path_to_send = st.session_state.pdf_path
            headers = {"X-User": st.session_state.user} if st.session_state.user else {}
            data = {
                "pdf_path": pdf_path_to_send,
                "save_to_db": st.session_state.save_to_db if st.session_state.user else False
            }
            res = requests.post(f"{API_BASE}/upload", data=data, headers=headers, timeout=LONG_TIMEOUT)

            if res.status_code == 200:
                data = res.json()
                if data.get("status") == "success":
                    st.session_state.doc_id = data["doc_id"]
                    st.session_state.save_to_db = False
                    st.session_state.num_pages = data.get("num_pages", 0)
                    st.session_state.num_paragraphs = data.get("num_paragraphs", 0)
                    st.session_state.human_check = human_check
                    st.session_state.q4_confirmed = False

                    cached = data.get("cached", False)
                    cache_msg = " (từ cache 🗄️)" if cached else ""
                    st.success(
                        f"✅ Extraction complete{cache_msg} (ID: {data['doc_id']}) — "
                        f"Found {st.session_state.num_pages} pages, "
                        f"{st.session_state.num_paragraphs} paragraphs"
                    )
                else:
                    st.error(f"❌ Extraction failed: {data.get('message')}")
            else:
                st.error(f"❌ API error: {res.status_code} — {res.text[:200]}")

        except requests.exceptions.Timeout:

            st.error("❌ Upload timed out. The PDF may be too large.")

        except requests.exceptions.ConnectionError:

            st.error("❌ Cannot connect to backend. Is the server running?")



    # Step 2: Human Check → Q4 verification

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
                        st.rerun()  # Re-evaluate FIFO display mode → switch to "q4"

                    else:

                        st.error(f"❌ Q4 verification failed: {result.get('message')}")

            except requests.exceptions.Timeout:

                st.error("❌ Q4 verification timed out.")

            except requests.exceptions.ConnectionError:

                st.error("❌ Cannot connect to backend.")



    # Step 2b: No Human Check → translate+render+keywords in parallel

    if not st.session_state.human_check and st.session_state.doc_id:

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)



        dynamic_timeout = (st.session_state.num_paragraphs * 20) + (st.session_state.num_pages * 5) + 600

        if dynamic_timeout < LONG_TIMEOUT:

            dynamic_timeout = LONG_TIMEOUT



        fut_pipeline = executor.submit(

            _run_translate_and_render,

            st.session_state.doc_id, tgt_lang_code, translation_model.lower(),

            st.session_state.num_paragraphs, st.session_state.num_pages

        )

        fut_keywords = executor.submit(

            _run_keywords, st.session_state.doc_id, agent_llm_code

        )



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
                        st.session_state.pdf_version = st.session_state.get("pdf_version", 0) + 1

                    st.success("✅ Translation & PDF rendering complete")

                else:

                    st.error(f"❌ Pipeline failed at {pipeline_result.get('step', 'unknown')}")

            except concurrent.futures.TimeoutError:

                st.error(f"❌ Translation pipeline timed out after {dynamic_timeout}s.")

            except Exception as e:

                st.error(f"❌ Translation pipeline error: {e}")



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

        # Rerun so FIFO display mode re-evaluates → switch to "results"
        if st.session_state.translated_markdown is not None:
            st.rerun()





# ── HITL Review & Confirm ────────────────────────────────────
if current_mode == "q4":

    st.header("🔍 Q4 Verification — Review Flagged Elements")
    q4 = st.session_state.q4_result

    st.info(f"Found **{q4.get('q4_count', 0)}** elements in the bottom 25th percentile. "
            f"Threshold: {q4.get('threshold', 'N/A')}")

    if "q4_corrections" not in st.session_state:
        st.session_state.q4_corrections = {}

    flagged_items = [item for item in q4.get("results", []) if item.get("verdict") == "REVIEW"]

    if flagged_items:
        st.subheader("⚠️ Elements Needing Review & OCR Corrections")
        for item in flagged_items:
            idx = item.get("index")
            orig_content = item.get("content", "")
            proposed = item.get("proposed_correction") or orig_content
            
            with st.expander(f"Block [{idx}] - Conf: {item.get('score', 0):.3f} | `{orig_content[:60]}`", expanded=True):
                if item.get("suggestion"):
                    st.caption(f"💡 Suggestion: {item['suggestion']}")
                
                st.text_area("📄 Original content (read-only)", value=orig_content, height=100, disabled=True, key=f"q4_orig_{idx}")
                
                col_checkbox, col_edit = st.columns([1, 4])
                
                apply_key = f"q4_apply_{idx}"
                is_different = proposed.strip() != orig_content.strip()
                apply = col_checkbox.checkbox("Apply", value=is_different, key=apply_key)
                
                edit_key = f"q4_edit_{idx}"
                corrected_text = col_edit.text_area("Correction proposal", value=proposed, key=edit_key)
                
                if apply:
                    st.session_state.q4_corrections[idx] = {
                        "page": item.get("page", 0),
                        "original": orig_content,
                        "corrected": corrected_text,
                        "bbox": item.get("bbox")
                    }
                else:
                    st.session_state.q4_corrections.pop(idx, None)
    else:
        st.success("✅ No low-confidence elements flagged for review. Proceed directly.")

    st.divider()
    if st.button("✅ Confirm & Continue Pipeline", use_container_width=True, type="primary"):
        if not check_cooldown("action_q4_confirm", 1.5):
            st.stop()
        # Apply corrections to backend before translating
        if st.session_state.get("q4_corrections"):
            with st.spinner("Applying OCR corrections to layout JSON..."):
                try:
                    corrections_list = list(st.session_state.q4_corrections.values())
                    res = requests.post(
                        f"{API_BASE}/agent/apply-corrections",
                        json={
                            "doc_id": st.session_state.doc_id,
                            "corrections": corrections_list
                        },
                        timeout=30
                    )
                    if res.status_code == 200 and res.json().get("status") == "success":
                        st.success(f"Successfully applied {res.json().get('applied_count', 0)} OCR corrections!")
                    else:
                        st.error("Failed to apply corrections to backend.")
                except Exception as e:
                    st.error(f"Error applying corrections: {e}")
        st.session_state.q4_confirmed = True
        st.rerun()





# ── After Confirm: translate+render+keywords in parallel ─────

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

        st.session_state.doc_id, tgt_lang_code, translation_model.lower(),

        st.session_state.num_paragraphs, st.session_state.num_pages

    )

    fut_keywords = executor.submit(

        _run_keywords, st.session_state.doc_id, agent_llm_code

    )



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
                    st.session_state.pdf_version = st.session_state.get("pdf_version", 0) + 1

                st.success("✅ Translation & PDF rendering complete")

            else:

                st.error(f"❌ Pipeline failed at {pipeline_result.get('step', 'unknown')}")

        except concurrent.futures.TimeoutError:

            st.error(f"❌ Translation pipeline timed out after {dynamic_timeout}s.")

        except Exception as e:

            st.error(f"❌ Translation pipeline error: {e}")



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





    # Rerun so FIFO display mode re-evaluates → switch to "results"
    if st.session_state.translated_markdown is not None:
        st.rerun()





# ── Results Display ────────────────────────────────────────

if current_mode == "results":

    if st.session_state.selected_history_doc:
        if st.button("🔙 Return to History List", use_container_width=True, type="secondary"):
            if check_cooldown("action_return_history", 1.5):
                st.session_state.doc_id = None
                st.session_state.translated_markdown = None
                st.session_state.original_markdown = None
                st.session_state.pdf_ready = False
                st.session_state.selected_history_doc = None
                st.session_state["display_mode"] = "history"
                st.rerun()

    st.header("📊 Results")



    # History docs only show Downloads + Keywords tabs
    if st.session_state.selected_history_doc:
        tab_names = ["📥 Downloads", "📚 Keywords"]
        tabs = st.tabs(tab_names)
        tab_dl, tab_kw = tabs
        tab_md = None
        tab_edit = None
    else:
        tab_names = ["📝 Translated Markdown", "✏️ Editable Text", "📚 Keywords", "📥 Downloads"]
        tabs = st.tabs(tab_names)
        tab_md, tab_edit, tab_kw, tab_dl = tabs



    start_edit_time = time.time()



    if tab_md is not None:

      with tab_md:

        st.markdown(st.session_state.translated_markdown, unsafe_allow_html=True)



    if tab_edit is not None:

      with tab_edit:

        st.subheader("✏️ Editable Translated Blocks")

        st.caption("Each block corresponds to a paragraph in the translated layout. "

                   "Edit as needed, then submit feedback below.")



        edited_text = st.text_area(
            "Edit translated markdown (feedback loop)",
            st.session_state.translated_markdown,
            height=500,
        )

        if st.button("🔄 Apply Edits & Re-render PDF", use_container_width=True):
            if not check_cooldown("action_apply_edits", 1.5):
                st.stop()
            with st.spinner("Applying edits and re-rendering PDF..."):
                try:
                    res = requests.post(
                        f"{API_BASE}/document/{st.session_state.doc_id}/update-markdown",
                        json={"markdown": edited_text},
                        timeout=30
                    )
                    if res.status_code == 200 and res.json().get("status") == "success":
                        st.session_state.translated_markdown = edited_text
                        
                        render_res = requests.post(
                            f"{API_BASE}/render-pdf",
                            json={"doc_id": st.session_state.doc_id},
                            timeout=LONG_TIMEOUT
                        )
                        if render_res.status_code == 200 and render_res.json().get("status") == "success":
                            st.session_state.pdf_ready = True
                            st.session_state.pdf_version = st.session_state.get("pdf_version", 0) + 1
                            st.success("✅ Edits applied and PDF re-rendered!")
                            st.rerun()
                        else:
                            st.error(f"❌ PDF rendering failed: {render_res.text}")
                    else:
                        st.error(f"❌ Sync failed: {res.text}")
                except Exception as e:
                    st.error(f"❌ Error: {e}")



        st.subheader("📊 Submit Evaluation")

        rating = st.slider("Rate the translation quality (1-5)", 1, 5, 3)

        downloaded = st.checkbox("Downloaded generated file?")

        if st.button("📤 Submit Feedback & Save Metrics"):
            if not check_cooldown("action_feedback", 1.5):
                st.stop()
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







    # Keywords Tab

    with tab_kw:

        st.subheader("📚 Extracted Keywords")

        kw_result = st.session_state.keywords_result

        if kw_result and kw_result.get("status") == "success":

            keywords = kw_result.get("keywords", [])

            if keywords:

                for kw in keywords:

                    st.markdown(f"- **{kw}**")

            else:

                st.info("No keywords extracted")

        else:

            st.info("Keywords not yet available.")

            if st.session_state.doc_id and st.button("🔄 Retry Keywords Extraction"):
                if not check_cooldown("action_retry_kw", 1.5):
                    st.stop()
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

        # 💾 Save to Database History Button
        if st.session_state.user and st.session_state.doc_id:
            if st.session_state.save_to_db:
                st.info("💾 Document is saved in your database history.")
            else:
                if st.button("💾 Save to Database History", use_container_width=True, type="primary"):
                    try:
                        headers = {"X-User": st.session_state.user}
                        res = requests.post(f"{API_BASE}/document/{st.session_state.doc_id}/save", headers=headers, timeout=10)
                        if res.status_code == 200:
                            st.session_state.save_to_db = True
                            st.success("Successfully saved to database history!")
                            st.rerun()
                        else:
                            st.error(res.json().get("message", "Failed to save to database history."))
                    except Exception as e:
                        st.error(f"Error saving to database: {e}")

        if st.session_state.doc_id and (st.session_state.pdf_ready or st.session_state.has_middle):

            try:

                pdf_bytes, pdf_error = _fetch_rendered_pdf(st.session_state.doc_id)

                if pdf_bytes:

                    st.session_state.pdf_ready = True



                    st.divider()
                    col_prev, col_open = st.columns([3, 1])
                    col_prev.subheader("👁️ PDF Preview")
                    
                    version = st.session_state.get("pdf_version", 0)
                    preview_url = (
                        f"{API_BASE}/stream-pdf/{st.session_state.doc_id}"
                        f"?v={version}"
                    )
                    col_open.markdown(f"[🌐 Open PDF in New Tab]({preview_url})")

                    st.iframe(preview_url, height=800)



                    st.divider()

                    st.download_button(

                        "💾 Save Translated PDF",

                        pdf_bytes,

                        file_name=f"{st.session_state.doc_id}_translated.pdf",

                        mime="application/pdf",

                        use_container_width=True

                    )

                else:

                    st.warning(pdf_error or "PDF stream not available")

            except Exception as e:

                st.error(f"PDF preview error: {e}")

        else:

            st.info("PDF will be available after conversion with layout.json data")

