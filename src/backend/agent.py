"""
AI Agent — Q4 score verification, keyword extraction, WikiSearch URLs.

Roles:
1. Verify extracted elements from Q4 (bottom 25th percentile) of "score"
   in layout.json spans. Flag low-confidence OCR/equation extractions.
2. Extract keywords from the .md file (from Abstract Keywords line or
   full content scan) and present WikiSearch URLs in target language.
3. Translate skipped translatable elements (e.g. table cell text that
   the main pipeline might miss).

LLM: deepseek-ai/deepseek-v4-flash via NVIDIA API (langchain_openai)
     or gemini-2.5-flash via Google API (langchain_google_genai)
"""

import os
import re
import json
import time
import uuid
import xml.etree.ElementTree as ET
import numpy as np
import requests
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from .database import EvalKeyword

load_dotenv()

# =====================================================================
# MAIN MODULE: AGENT
# =====================================================================

class AIAgent:
    SUPPORTED_PROVIDERS = ["gpt", "gemini"]

    def __init__(self, target_lang: str = "fra_Latn", db_session=None,
                 llm_provider: str = "gemini"):
        self.db = db_session
        self.target_lang = target_lang
        self.llm_provider = llm_provider
        self.set_llm_provider(llm_provider)

    def _get_text_content(self, message) -> str:
        content = message.content
        if isinstance(content, list):
            return "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in content])
        return str(content)

    def set_llm_provider(self, provider: str):
        """Switch LLM backend between gpt and gemini."""
        self.llm_provider = provider
        try:
            if provider == "gemini":
                from langchain_google_genai import ChatGoogleGenerativeAI
                self.llm = ChatGoogleGenerativeAI(
                    model="gemini-flash-lite-latest",
                    google_api_key=os.environ.get("GEMINI_API_KEY", ""),
                    temperature=0.0,
                )
                print(f"[Agent] LLM set to Gemini 2.5 Flash")
            else:
                class NvidiaGPTClient:
                    def __init__(self, api_key: str):
                        from openai import OpenAI
                        self.client = OpenAI(
                            base_url="https://integrate.api.nvidia.com/v1",
                            api_key=api_key
                        )

                    def invoke(self, prompt: str):
                        completion = self.client.chat.completions.create(
                            model="openai/gpt-oss-120b",
                            messages=[{"role": "user", "content": prompt}],
                            temperature=1,
                            top_p=1,
                            max_tokens=4096,
                            stream=True
                        )
                        content_parts = []
                        for chunk in completion:
                            if not getattr(chunk, "choices", None):
                                continue
                            reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
                            if reasoning:
                                print(reasoning, end="", flush=True)
                            if chunk.choices and chunk.choices[0].delta.content is not None:
                                content_parts.append(chunk.choices[0].delta.content)
                                print(chunk.choices[0].delta.content, end="", flush=True)
                        print()
                        class MockMessage:
                            def __init__(self, content):
                                self.content = content
                        return MockMessage("".join(content_parts))

                api_key = os.environ.get("GPT_API_KEY", "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")
                self.llm = NvidiaGPTClient(api_key=api_key)
                print(f"[Agent] LLM set to GPT OSS 120B (Nvidia OpenAI Client)")
        except Exception as e:
            print(f"[Agent] LLM init failed ({provider}): {e}")
            self.llm = None

        self.verify_prompt = PromptTemplate(
            input_variables=["low_score_elements"],
            template="""You are an AI agent verifying OCR/extraction quality.
The following elements were extracted with low confidence scores (bottom 25%).
For each element, check if there are OCR errors, missing math symbols, incorrect formatting, or text glitches.
If you find any issues, set verdict to "REVIEW" and provide a corrected version of the source content in "proposed_correction".
If the element looks correct and needs no changes, set verdict to "OK" and "proposed_correction" to null.

Elements:
{low_score_elements}

Respond as a JSON array. Each item MUST have these exact keys:
- "index": <int>
- "content": "..."
- "verdict": "OK" or "REVIEW"
- "suggestion": "description of issues found"
- "proposed_correction": "corrected text or math formula, or null if OK"

Only output the JSON array, no markdown fences."""
        )

    # ── Q4 Score Verification ───────────────────────────────────

    def _is_quartet_text(self, obj) -> bool:
        if not isinstance(obj, dict): return False
        return (
            "bbox" in obj and
            obj.get("type") == "text" and
            "content" in obj and
            isinstance(obj.get("score"), (int, float))
        )

    def _contains_quartet_recursive(self, obj) -> bool:
        if self._is_quartet_text(obj):
            return True
        if isinstance(obj, dict):
            for v in obj.values():
                if self._contains_quartet_recursive(v): return True
        elif isinstance(obj, list):
            for item in obj:
                if self._contains_quartet_recursive(item): return True
        return False

    def _extract_paragraph_info(self, block_chain: list) -> tuple:
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

    def _collect_jobs_for_scoring(self, blocks: list) -> list:
        jobs = []
        i = 0
        while i < len(blocks):
            block = blocks[i]
            if block.get("type") == "interline_equation":
                i += 1
                continue

            if self._contains_quartet_recursive(block) or block.get("lines"):
                chain = [block]
                j = i + 1
                while j < len(blocks) and blocks[j].get("merge_prev") is True:
                    chain.append(blocks[j])
                    j += 1

                paragraph, eq_map = self._extract_paragraph_info(chain)
                if paragraph.strip():
                    jobs.append((chain, paragraph, eq_map))

                if "blocks" in block:
                    jobs.extend(self._collect_jobs_for_scoring(block["blocks"]))

                i = j
            elif "blocks" in block:
                jobs.extend(self._collect_jobs_for_scoring(block["blocks"]))
                i += 1
            else:
                i += 1

        return jobs

    def _collect_spans_scores(self, block, scores):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if "score" in span:
                    scores.append(span["score"])
        for sub in block.get("blocks", []):
            self._collect_spans_scores(sub, scores)

    def collect_paragraphs_with_scores(self, layout_data: dict) -> list:
        pages = layout_data.get("pdf_info", [])
        paragraphs_info = []
        
        for page in pages:
            page_idx = page.get("page_idx", 0)
            blocks = page.get("preproc_blocks", page.get("para_blocks", [])) + page.get("discarded_blocks", [])
            
            jobs = self._collect_jobs_for_scoring(blocks)
            for chain, text, eq_map in jobs:
                scores = []
                for block in chain:
                    self._collect_spans_scores(block, scores)
                
                mean_score = float(np.mean(scores)) if scores else 1.0
                paragraphs_info.append({
                    "page": page_idx,
                    "content": text,
                    "score": mean_score,
                    "bbox": chain[0].get("bbox", [0, 0, 0, 0])
                })
        return paragraphs_info

    def verify_q4_elements(self, layout_data: dict) -> dict:
        paragraphs = self.collect_paragraphs_with_scores(layout_data)
        if not paragraphs:
            return {"q4_count": 0, "results": [], "threshold": None}

        scores = [p["score"] for p in paragraphs]
        q1_threshold = float(np.percentile(scores, 25))
        q4 = [p for p in paragraphs if p["score"] <= q1_threshold]

        if not q4:
            return {"q4_count": 0, "threshold": q1_threshold, "results": []}

        threshold = q1_threshold
        if not self.llm:
            return {"q4_count": len(q4), "threshold": threshold, "results": []}

        all_results = []
        for batch_start in range(0, min(len(q4), 20), 10):
            batch = q4[batch_start:batch_start + 10]
            elements_text = "\n".join(
                f"[{i}] score={s['score']:.3f}, content=\"{s['content']}\""
                for i, s in enumerate(batch, start=batch_start)
            )
            try:
                resp = self.llm.invoke(self.verify_prompt.format(low_score_elements=elements_text))
                content = self._get_text_content(resp).strip()

                match = re.search(r"\[.*\]", content, re.DOTALL)
                json_str = match.group(0) if match else content
                llm_items = json.loads(json_str)

                for item in llm_items:
                    idx = item.get("index")
                    if idx is not None and isinstance(idx, int) and 0 <= idx < len(q4):
                        orig = q4[idx]
                        item["score"] = orig["score"]
                        item["page"] = orig["page"]
                        item["bbox"] = orig.get("bbox")
                        if "proposed_correction" not in item:
                            item["proposed_correction"] = orig["content"]
                all_results.extend(llm_items)
            except Exception as e:
                print(f"[Agent] Verification error: {e}")
                all_results.extend([
                    {"index": i, "content": s["content"], "score": s["score"],
                     "page": s["page"], "bbox": s.get("bbox"),
                     "verdict": "REVIEW", "suggestion": f"LLM error: {e}",
                     "proposed_correction": s["content"]}
                    for i, s in enumerate(batch, start=batch_start)
                ])

        return {"q4_count": len(q4), "threshold": threshold, "results": all_results}

    # ── Keyword Extraction ──────────────────────────────────────

    def extract_keywords(self, markdown: str) -> list:
        keywords = []
        kw_match = re.search(
            r"(?:Keywords?|Key\s*words?)\s*[:：]\s*(.+?)(?:\n\n|\n##|\n#|\Z)",
            markdown,
            re.IGNORECASE | re.DOTALL,
        )
        if kw_match:
            raw = kw_match.group(1).strip()
            parts = re.split(r"[,;]\s*", raw)
            keywords = [p.strip().rstrip(".") for p in parts if p.strip() and len(p.strip()) > 2]

        if not keywords and self.llm:
            try:
                resp = self.llm.invoke(
                    f"Extract the main technical keywords from this research paper. "
                    f"Return ONLY a JSON array of strings.\n\n{markdown}"
                )
                content = self._get_text_content(resp).strip()

                # REGEX FILTER FOR JSON
                match = re.search(r"\[.*\]", content, re.DOTALL)
                json_str = match.group(0) if match else content
                keywords = json.loads(json_str)
            except Exception:
                pass

        return keywords

    # ── Orchestrator Agent & Tools ──────────────────────────────

    def _safe_request(self, url: str, params: dict, max_retries: int = 3) -> dict:
        """Hàm gọi API bọc lớp bảo vệ chống 429 Too Many Requests."""
        headers = {
            "User-Agent": "DIMT-Research-Bot/1.0 (mailto:admin@dimt-project.org)",
            "Accept": "application/json"
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, headers=headers, timeout=15)
                
                if response.status_code == 429:
                    wait_time = 2 ** attempt
                    print(f"[Agent] ⚠️ 429 Rate Limit. Đang chờ {wait_time}s để thử lại...")
                    time.sleep(wait_time)
                    continue
                    
                response.raise_for_status()
                return response.json()
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"[Agent] ❌ API Request failed after {max_retries} attempts: {e}")
                    return {}
                time.sleep(1)
        return {}

    def agent_normalize_concept(self, keyword: str) -> str:
        if not self.llm:
            return keyword

        prompt = f"""
        Normalize this technical keyword into a canonical English Wikipedia-style concept.
        Rules:
        - Return ONLY the normalized concept
        - No explanations
        - Expand acronyms if possible
        - Remove metrics/modifiers if needed

        Keyword:
        {keyword}
        """
        try:
            resp = self.llm.invoke(prompt)
            return self._get_text_content(resp).strip()
        except Exception:
            return keyword

    def tool_search_wikidata(self, query: str, limit: int = 5) -> list:
        url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities",
            "search": query,
            "language": "en",
            "format": "json",
            "limit": limit
        }
        
        data = self._safe_request(url, params)
        candidates = []
        for item in data.get("search", []):
            candidates.append({
                "id": item.get("id"),
                "label": item.get("label"),
                "description": item.get("description", "")
            })
        return candidates

    def tool_search_crossref(self, query: str) -> dict:
        """
        Fallback Tool: Tìm kiếm trực tiếp trên Crossref API.
        Lấy trực tiếp link DOI của bài báo chứa từ khóa chuyên ngành.
        """
        url = "https://api.crossref.org/works"
        
        # Bỏ dấu gạch ngang giúp Crossref search chính xác hơn với một số keyword
        safe_query = query.replace("-", " ")
        
        params = {
            "query": safe_query,
            "select": "title,URL", # Chỉ lấy Title và URL để tối ưu băng thông (Crossref docs)
            "rows": 1              # Chỉ lấy kết quả liên quan nhất
        }
        
        # Áp dụng quy tắc "Etiquette" (Polite Pool) của Crossref để có server xịn
        headers = {
            "User-Agent": "DIMT-Research-Bot/1.0 (mailto:admin@dimt-project.org)",
            "Accept": "application/json"
        }
        
        try:
            response = requests.get(url, params=params, headers=headers, timeout=15)
            response.raise_for_status()
            
            data = response.json()
            items = data.get("message", {}).get("items", [])
            
            if items:
                item = items[0]
                # Crossref thường trả về title dưới dạng mảng (list)
                title = item.get("title", [""])[0] if item.get("title") else ""
                paper_url = item.get("URL", "")
                
                if paper_url:
                    return {
                        "title": title,
                        "url": paper_url
                    }
        except Exception as e:
            print(f"[Agent] ❌ Crossref Tool failed for '{query}': {e}")
            
        return {}
    
    def agent_select_candidate(self, keyword: str, candidates: list) -> dict:
        if not candidates:
            return {}

        if len(candidates) == 1 or not self.llm:
            return candidates[0]

        candidate_text = "\n".join([
            f"{i}. {c['label']} - {c['description']}"
            for i, c in enumerate(candidates)
        ])

        prompt = f"""
        Select the BEST matching Wikipedia concept for this keyword.
        If none are perfect, pick the most relevant one.

        Keyword:
        {keyword}

        Candidates:
        {candidate_text}

        Return ONLY the candidate index number.
        """
        try:
            resp = self.llm.invoke(prompt)
            idx_text = self._get_text_content(resp).strip()

            idx_match = re.search(r"\d+", idx_text)
            if idx_match:
                idx = int(idx_match.group(0))
                if 0 <= idx < len(candidates):
                    return candidates[idx]
        except Exception:
            pass

        return candidates[0]

    def tool_get_sitelinks(self, qid: str, lang: str = "fr") -> dict:
        url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbgetentities",
            "ids": qid,
            "props": "sitelinks",
            "format": "json"
        }

        data = self._safe_request(url, params)
        if not data or "entities" not in data or qid not in data["entities"]:
            return {"url": None, "en_url": None, "display_label": None}

        sitelinks = data["entities"][qid].get("sitelinks", {})
        target_key = f"{lang}wiki"
        en_key = "enwiki"

        result = {
            "url": None,
            "en_url": None,
            "display_label": None
        }

        if target_key in sitelinks:
            title = sitelinks[target_key]["title"]
            result["display_label"] = title
            result["url"] = f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"

        if en_key in sitelinks:
            en_title = sitelinks[en_key]["title"]
            result["en_url"] = f"https://en.wikipedia.org/wiki/{en_title.replace(' ', '_')}"
            if not result["display_label"]:
                result["display_label"] = en_title

        return result

    def agentic_keyword_pipeline(self, kw: str, wiki_lang: str) -> dict:
        """Luồng Orchestrator cho một keyword đơn lẻ."""
        try:
            # 1. Agent Reasoning -> Chuẩn hóa concept
            normalized = self.agent_normalize_concept(kw)

            # 2. Deterministic Tool -> Tìm ứng viên Wikidata
            candidates = self.tool_search_wikidata(normalized, limit=7)

            # 3. Dynamic Fallback 1: Thử tìm Wikidata bằng keyword gốc
            if not candidates and normalized != kw:
                print(f"[Agent] Fallback to raw keyword on Wikidata for: {kw}")
                candidates = self.tool_search_wikidata(kw, limit=5)

            # --- KHU VỰC TÌM KIẾM HỌC THUẬT (CROSSREF) KHI WIKI THẤT BẠI ---
            if not candidates:
                print(f"[Agent] Wikidata failed. Fallback to Crossref API for '{kw}'...")
                
                # 4. Fallback duy nhất: Tìm trên Crossref
                academic_data = self.tool_search_crossref(kw)
                
                if academic_data and academic_data.get("url"):
                    print(f"[Agent] 🎯 Found Crossref DOI cho '{kw}': {academic_data['title']}")
                    return {
                        "original_keyword": kw,
                        "normalized_keyword": normalized,
                        "wikidata_id": None,
                        "display_label": kw, 
                        "url": academic_data["url"], 
                        "en_url": academic_data["url"]
                    }
                
                # 5. Graceful Degradation (Cả Wiki và Crossref đều bó tay)
                print(f"[Agent] ℹ️ Không có Wiki/Paper cho '{kw}'. Giữ nguyên text.")
                return {
                    "original_keyword": kw,
                    "normalized_keyword": normalized,
                    "wikidata_id": None,
                    "display_label": normalized if normalized != kw else kw,
                    "url": None,
                    "en_url": None
                }
            # --- KẾT THÚC KHU VỰC TÌM KIẾM HỌC THUẬT ---

            # 6. Nếu Wikidata thành công -> Rerank & Select
            best_candidate = self.agent_select_candidate(
                keyword=kw,
                candidates=candidates
            )
            qid = best_candidate["id"]

            # 7. Trích xuất Sitelinks từ Wikidata
            sitelink_data = self.tool_get_sitelinks(
                qid=qid,
                lang=wiki_lang
            )

            return {
                "original_keyword": kw,
                "normalized_keyword": normalized,
                "wikidata_id": qid,
                "display_label": sitelink_data.get("display_label") or best_candidate["label"],
                "url": sitelink_data.get("url"),
                "en_url": sitelink_data.get("en_url")
            }

        except Exception as e:
            print(f"[Agent] ❌ Pipeline crash for '{kw}': {e}")
            return {
                "original_keyword": kw,
                "normalized_keyword": kw,
                "wikidata_id": None,
                "display_label": kw,
                "url": None,
                "en_url": None
            }

    def run(self, layout_data: dict, markdown: str) -> dict:
        q4_result = self.verify_q4_elements(layout_data)
        keywords = self.extract_keywords(markdown)

        return {
            "q4_verification": q4_result,
            "keywords": keywords,
        }