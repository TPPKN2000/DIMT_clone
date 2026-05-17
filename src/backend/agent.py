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
from urllib.parse import quote
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from .database import EvalKeyword

load_dotenv()


# =====================================================================
# LANG CODE → WIKIPEDIA SUBDOMAIN MAPPING
# =====================================================================

LANG_TO_WIKI = {
    "vie_Latn": "vi",
    "fra_Latn": "fr",
    "deu_Latn": "de",
    "zho_Hans": "zh",
    "jpn_Jpan": "ja",
    "kor_Hang": "ko",
    "spa_Latn": "es",
    "por_Latn": "pt",
    "rus_Cyrl": "ru",
    "ara_Arab": "ar",
    "eng_Latn": "en",
}


# =====================================================================
# MAIN MODULE: AGENT
# =====================================================================

class AIAgent:
    SUPPORTED_PROVIDERS = ["deepseek", "gemini"]

    def __init__(self, target_lang: str = "vie_Latn", db_session=None,
                 llm_provider: str = "deepseek"):
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

    # ── Q4 Score Verification ───────────────────────────────────────

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

    # ── Keyword Extraction ──────────────────────────────────────────

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
                    f"Return ONLY a JSON array of strings, no markdown fences.\n\n{markdown[:3000]}"
                )
                content = self._get_text_content(resp).strip()
                match = re.search(r"\[.*\]", content, re.DOTALL)
                json_str = match.group(0) if match else content
                keywords = json.loads(json_str)
            except Exception as e:
                print(f"[Agent] Keyword extraction error: {e}")

        return keywords

    # ── WikiSearch URLs ─────────────────────────────────────────────

    def get_keyword_wiki_urls(self, keywords: list) -> list:
        """
        Build Wikipedia URLs for each keyword.

        Returns list of dicts:
            {"keyword": str, "url": str (target lang), "en_url": str (English)}

        URL format: https://{lang}.wikipedia.org/wiki/{encoded_keyword}
        Falls back gracefully — always provides the English URL at minimum.
        """
        wiki_lang = LANG_TO_WIKI.get(self.target_lang, "en")
        results = []

        for keyword in keywords:
            if not keyword or not keyword.strip():
                continue

            encoded = quote(keyword.strip().replace(" ", "_"))
            en_url = f"https://en.wikipedia.org/wiki/{encoded}"

            entry = {"keyword": keyword.strip(), "en_url": en_url}

            if wiki_lang != "en":
                entry["url"] = f"https://{wiki_lang}.wikipedia.org/wiki/{encoded}"
            else:
                entry["url"] = en_url

            results.append(entry)

        return results

    # ── Table Recovery Agent (Bước 3) ──────────────────────────────

    def recover_missing_translations(
        self,
        original_middle: dict,
        translated_middle: dict,
    ) -> dict:
        """
        Scan translated_middle for blocks that were NOT translated.

        A block is considered untranslated when:
          - It contains spans with text content, AND
          - None of those spans carry "translated": True

        For each untranslated block we extract the source text from
        original_middle (matched by page_idx + bbox), then translate it
        using the LLM (cheaper than NLLB for small patches) and write
        the result back into translated_middle in-place.

        Returns a summary dict:
          {
            "recovered_count": int,       # blocks successfully translated
            "skipped_count":   int,       # blocks where LLM call failed
            "patches":         [          # detail of each recovered block
              {"page": int, "block_type": str, "bbox": list,
               "source": str, "translation": str}
            ]
          }
        """
        if not self.llm:
            print("[Agent] Table recovery skipped — no LLM available")
            return {"recovered_count": 0, "skipped_count": 0, "patches": []}

        # Index original blocks by (page_idx, bbox_key) for fast lookup
        orig_index: dict[tuple, str] = {}
        for page in original_middle.get("pdf_info", []):
            pi = page.get("page_idx", 0)
            for block in self._iter_all_blocks(page):
                bbox_key = tuple(block.get("bbox", []))
                src_text = self._extract_text_from_block(block)
                if src_text:
                    orig_index[(pi, bbox_key)] = src_text

        recovered, skipped = 0, 0
        patches: list[dict] = []

        for page in translated_middle.get("pdf_info", []):
            pi = page.get("page_idx", 0)
            for block in self._iter_all_blocks(page):
                if not self._needs_translation(block):
                    continue

                bbox = block.get("bbox", [])
                bbox_key = tuple(bbox)
                src_text = orig_index.get((pi, bbox_key), "")

                if not src_text:
                    # Try to extract text from the block itself as fallback
                    src_text = self._extract_text_from_block(block)
                if not src_text:
                    continue

                try:
                    translation = self._llm_translate_snippet(src_text)
                    if translation:
                        self._patch_block_translation(block, translation)
                        patches.append({
                            "page":        pi,
                            "block_type":  block.get("type", "text"),
                            "bbox":        bbox,
                            "source":      src_text,
                            "translation": translation,
                        })
                        recovered += 1
                    else:
                        skipped += 1
                except Exception as e:
                    print(f"[Agent] Table recovery error on page {pi}: {e}")
                    skipped += 1

        print(f"[Agent] Table recovery: {recovered} recovered, {skipped} skipped")
        return {
            "recovered_count": recovered,
            "skipped_count":   skipped,
            "patches":         patches,
        }

    # ── Table Recovery helpers ──────────────────────────────────────

    def _iter_all_blocks(self, page: dict):
        """Yield all leaf blocks (that have lines) from a page dict."""
        root = page.get("preproc_blocks", page.get("para_blocks", []))
        yield from self._walk_blocks_recursive(root)

    def _walk_blocks_recursive(self, blocks: list):
        for block in blocks:
            if block.get("lines"):
                yield block
            if block.get("blocks"):
                yield from self._walk_blocks_recursive(block["blocks"])

    def _extract_text_from_block(self, block: dict) -> str:
        """Return all text content of a block as a single string."""
        parts = []
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                c = span.get("content", "").strip()
                if c:
                    parts.append(c)
        return " ".join(parts).strip()

    def _needs_translation(self, block: dict) -> bool:
        """
        Return True when the block has text spans but none is marked translated.

        Skips blocks that:
          - have no text content at all
          - are already fully translated (all text spans carry translated=True)
          - contain only equations / images
        """
        text_spans = []
        translated_spans = []

        for line in block.get("lines", []):
            for span in line.get("spans", []):
                stype = span.get("type", "")
                content = span.get("content", "").strip()
                if stype == "text" and content:
                    text_spans.append(span)
                    if span.get("translated", False):
                        translated_spans.append(span)

        if not text_spans:
            return False   # nothing translatable

        # All text already translated
        if len(translated_spans) == len(text_spans):
            return False

        return True

    def _llm_translate_snippet(self, text: str) -> str:
        """
        Translate a short snippet using the LLM.

        Used only for small patches (missed table cells, captions, labels)
        where NLLB has already finished and we cannot call it again.
        Returns empty string on failure.
        """
        lang_name = {
            "vie_Latn": "Vietnamese",
            "fra_Latn": "French",
            "deu_Latn": "German",
        }.get(self.target_lang, self.target_lang)

        prompt = (
            f"Translate the following English text to {lang_name}. "
            f"Return ONLY the translation, no explanation.\n\n{text}"
        )
        try:
            resp = self.llm.invoke(prompt)
            return self._get_text_content(resp).strip()
        except Exception as e:
            print(f"[Agent] LLM translate snippet error: {e}")
            return ""

    def _patch_block_translation(self, block: dict, translation: str) -> None:
        """Overwrite the block's lines with translated content, preserving bbox."""
        bbox = block.get("bbox", [0, 0, 0, 0])
        block["lines"] = [{
            "bbox": bbox,
            "spans": [{
                "bbox":       bbox,
                "type":       "text",
                "content":    translation,
                "score":      1.0,
                "translated": True,
                "recovered":  True,   # distinguish from NLLB translations
            }]
        }]

    # ── Combined Agent Run ──────────────────────────────────────────

    def run(self, layout_data: dict, markdown: str) -> dict:
        q4_result = self.verify_q4_elements(layout_data)
        keywords = self.extract_keywords(markdown)
        wiki_urls = self.get_keyword_wiki_urls(keywords)

        return {
            "q4_verification": q4_result,
            "keywords": keywords,
            "wiki_references": wiki_urls,
        }
