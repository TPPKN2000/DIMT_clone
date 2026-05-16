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
import uuid
import numpy as np
import time
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from .database import EvalKeyword

load_dotenv()




# =====================================================================
# MAIN MODULE: AGENT
# =====================================================================

class AIAgent:
    SUPPORTED_PROVIDERS = ["deepseek", "gemini"]

    def __init__(self, target_lang: str = "fra_Latn", db_session=None,
                 llm_provider: str = "deepseek"):
        self.db = db_session
        self.target_lang = target_lang
        self.llm_provider = llm_provider
        self.set_llm_provider(llm_provider)

    def _get_text_content(self, message) -> str:
        content = message.content
        if isinstance(content, list):
            # Trích xuất text từ các content parts
            return "".join([part.get("text", "") if isinstance(part, dict) else str(part) for part in content])
        return str(content)

    def set_llm_provider(self, provider: str):
        """Switch LLM backend between deepseek and gemini."""
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
                self.llm = ChatOpenAI(
                    base_url="https://integrate.api.nvidia.com/v1",
                    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
                    model="deepseek-ai/deepseek-v4-flash",
                    temperature=0.0,
                )
                print(f"[Agent] LLM set to DeepSeek V4 Flash")
        except Exception as e:
            print(f"[Agent] LLM init failed ({provider}): {e}")
            self.llm = None

        self.verify_prompt = PromptTemplate(
            input_variables=["low_score_elements"],
            template="""You are an AI agent verifying OCR/extraction quality.
The following elements were extracted with low confidence scores (bottom 25%).
For each, determine if the content looks correct or needs manual review.

Elements:
{low_score_elements}

Respond as JSON array. Each item: {{"index": <int>, "content": "...", "verdict": "OK"|"REVIEW", "suggestion": "..."}}
Only output the JSON array, no markdown fences."""
        )

    # ── Q4 Score Verification ───────────────────────────────────

    def collect_scores(self, layout_data: dict) -> list:
        spans_with_scores = []
        for page in layout_data.get("pdf_info", []):
            page_idx = page.get("page_idx", 0)
            for block in page.get("para_blocks", page.get("preproc_blocks", [])):
                self._collect_from_block(block, page_idx, spans_with_scores)
        return spans_with_scores

    def _collect_from_block(self, block, page_idx, results):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                if "score" in span:
                    results.append({
                        "page": page_idx,
                        "type": span.get("type"),
                        "content": span.get("content", "")[:100],
                        "score": span["score"],
                        "bbox": span.get("bbox"),
                    })
        for sub in block.get("blocks", []):
            self._collect_from_block(sub, page_idx, results)

    def get_q4_elements(self, layout_data: dict) -> list:
        all_spans = self.collect_scores(layout_data)
        if not all_spans:
            return []
        scores = [s["score"] for s in all_spans]
        q1_threshold = float(np.percentile(scores, 25))
        return [s for s in all_spans if s["score"] <= q1_threshold]

    def verify_q4_elements(self, layout_data: dict) -> dict:
        q4 = self.get_q4_elements(layout_data)
        if not q4:
            return {"q4_count": 0, "results": [], "threshold": None}

        threshold = max(s["score"] for s in q4)
        if not self.llm:
            return {"q4_count": len(q4), "threshold": threshold, "results": []}

        all_results = []
        for batch_start in range(0, min(len(q4), 40), 20):
            batch = q4[batch_start:batch_start + 20]
            elements_text = "\n".join(
                f"[{i}] type={s['type']}, score={s['score']:.3f}, content=\"{s['content']}\""
                for i, s in enumerate(batch, start=batch_start)
            )
            try:
                resp = self.llm.invoke(self.verify_prompt.format(low_score_elements=elements_text))
                content = self._get_text_content(resp).strip()

                # REGEX FILTER FOR JSON EXTRACTION
                match = re.search(r"\[.*\]", content, re.DOTALL)
                json_str = match.group(0) if match else content
                llm_items = json.loads(json_str)

                for item in llm_items:
                    idx = item.get("index")
                    if idx is not None and isinstance(idx, int) and 0 <= idx < len(q4):
                        item["score"] = q4[idx]["score"]
                all_results.extend(llm_items)
            except Exception as e:
                print(f"[Agent] Verification error: {e}")
                all_results.extend([
                    {"index": i, "content": s["content"][:50], "score": s["score"],
                     "verdict": "REVIEW", "suggestion": f"LLM error: {e}"}
                    for i, s in enumerate(batch, start=batch_start)
                ])

        return {"q4_count": len(q4), "threshold": threshold, "results": all_results}

    # ── Keyword Extraction & WikiSearch ─────────────────────────

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

    # ── Combined Agent Run ──────────────────────────────────────

    def run(self, layout_data: dict, markdown: str) -> dict:
        q4_result = self.verify_q4_elements(layout_data)
        keywords = self.extract_keywords(markdown)

        return {
            "q4_verification": q4_result,
            "keywords": keywords,
            "wiki_references": wiki_urls,
        }