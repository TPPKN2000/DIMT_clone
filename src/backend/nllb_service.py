"""
NLLB Translation Service — Paragraph-level translation from layout.json.

Key mechanisms:
- Uses `para_blocks` from layout.json (lines pre-grouped into paragraphs)
- `merge_prev`: when True on a text block, the block is a continuation of the
  previous paragraph — we concatenate them before translation for better context
- Inline equations are protected with placeholders [EQ_n] during translation
- Translates at paragraph level (not line-by-line) for coherent output
- Tables: extracts translatable text from HTML, translates, reconstructs
- LoRA adapter loaded from nllb-1.3B-multilingual-final/ with 4-bit quantization
- Target languages: vie_Latn (Vietnamese), deu_Latn (German), fra_Latn (French)
- Q7: Cross-page stitching — merges cross_page spans and splits back by word ratio
- Q9: Batch translation — groups paragraphs into batches (max 512 tokens) for throughput
"""

import re
import copy
import time
import torch
import threading
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from html.parser import HTMLParser

# ── Protection patterns for markdown translation ───────────────
PROTECTED_PATTERNS = [
    (r"```[\s\S]*?```", "CODE"),
    (r"`[^`\n]+`", "INLINECODE"),
    (r"\$\$[\s\S]*?\$\$", "MATH"),
    (r"\\begin\{[^}]+\}[\s\S]*?\\end\{[^}]+\}", "MATH"),
    (r"\\\[[\s\S]*?\\\]", "MATH"),
    (r"\\\([\s\S]*?\\\)", "MATH"),
    (r"(?<![\w\$])\$(?!\$)(?:[^\$\n\\]|\\.)+\$(?!(?:\w|\$))", "MATH"),
    # Markdown syntax protection
    (r"\[([^\]]+)\]\(([^)]+)\)", "LINK"),
    (r"\!\[([^\]]*)\]\(([^)]+)\)", "IMAGE"),
    (r"\*\*[^*]+\*\*", "BOLD"),
    (r"\*[^*]+\*", "ITALIC"),
]

# Max tokens per batch segment
BATCH_MAX_TOKENS = 512


class NLLBService:
    SUPPORTED_LANGS = ["vie_Latn", "deu_Latn", "fra_Latn"]

    # Human-readable display names for the frontend
    LANG_DISPLAY = {
        "vie_Latn": "Vietnamese (vie_Latn)",
        "fra_Latn": "French (fra_Latn)",
        "deu_Latn": "German (deu_Latn)",
    }

    def __init__(self, device=None, tgt_lang: str = "vie_Latn",
                 adapter_path: str = "nllb-1.3B-multilingual-final",
                 lazy_load: bool = False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded = False
        self.src_lang = "eng_Latn"
        self.tgt_lang = tgt_lang
        self.model_name = "facebook/nllb-200-distilled-1.3B"
        self.adapter_path = adapter_path
        self._load_event = threading.Event()

        if not lazy_load:
            self.load_model()

    def load_model(self):
        if self.loaded:
            return
        try:
            print(f"[NLLB] Loading {self.model_name} + LoRA adapter from {self.adapter_path}...")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.adapter_path, src_lang=self.src_lang
            )

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if self.device == "cuda" else torch.float16,
            )

            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                low_cpu_mem_usage=True,
            )

            padded = (len(self.tokenizer) + 63) // 64 * 64
            base_model.resize_token_embeddings(padded, mean_resizing=False)

            self.model = PeftModel.from_pretrained(base_model, self.adapter_path)
            self.model.eval()

            self.loaded = True
            print("[NLLB] Model + LoRA adapter loaded successfully")
        except Exception:
            import traceback
            print("[NLLB] Load failed:")
            traceback.print_exc()
        finally:
            self._load_event.set()

    def wait_for_load(self):
        self._load_event.wait()

    def set_target_lang(self, lang_code: str):
        """Switch target language. Supports vie_Latn, deu_Latn, fra_Latn."""
        if lang_code not in self.SUPPORTED_LANGS:
            raise ValueError(f"Unsupported language: {lang_code}. Use one of {self.SUPPORTED_LANGS}")
        self.tgt_lang = lang_code

    # ── Core translate ──────────────────────────────────────────────

    def _translate_text(self, text: str) -> str:
        """Translate a single string. Returns mock if model not loaded."""
        if not text.strip():
            return text
        if not self.loaded:
            return f"[{self.tgt_lang}] {text}"

        self.tokenizer.src_lang = self.src_lang
        self.tokenizer.tgt_lang = self.tgt_lang

        forced_bos_id = self.tokenizer.convert_tokens_to_ids(self.tgt_lang)
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_id,
                max_length=512,
                num_beams=4,
            )
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()

    def _translate_batch(self, texts: list[str]) -> list[str]:
        """Q9: Translate a batch of strings for better GPU utilization."""
        if not texts:
            return []
        if not self.loaded:
            return [f"[{self.tgt_lang}] {t}" for t in texts]

        self.tokenizer.src_lang = self.src_lang
        self.tokenizer.tgt_lang = self.tgt_lang
        forced_bos_id = self.tokenizer.convert_tokens_to_ids(self.tgt_lang)

        inputs = self.tokenizer(
            texts, return_tensors="pt", truncation=True,
            max_length=512, padding=True
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_id,
                max_length=512,
                num_beams=4,
            )
        results = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        return [r.strip() for r in results]

    # ── Paragraph extraction from layout.json ───────────────────────

    def translate_middle_json(self, middle_data: dict) -> dict:
        """
        Translate all translatable content in layout.json in-place.

        Uses para_blocks for paragraph-level grouping.
        Respects merge_prev to concatenate continuation blocks.
        Protects inline_equation spans with placeholders.
        Q7: Handles cross_page spans via stitching + word-ratio split.
        Q9: Batches paragraphs for efficient GPU throughput.
        Returns a deep-copied middle_data with translated content.
        """
        t_start = time.time()
        translated = copy.deepcopy(middle_data)
        pages = translated.get("pdf_info", [])
        total_pages = len(pages)

        # Q7: Cross-page stitching
        self._stitch_cross_page(pages)

        all_jobs = []

        for pi, page in enumerate(pages):
            blocks = page.get("preproc_blocks", page.get("para_blocks", []))
            jobs = self._collect_translation_jobs(blocks)
            all_jobs.extend(jobs)
            if (pi + 1) % 5 == 0 or pi == total_pages - 1:
                print(f"[NLLB] Collected jobs from page {pi+1}/{total_pages} (total jobs: {len(all_jobs)})")

        # Q9: Batch translate
        if all_jobs:
            print(f"[NLLB] Translating {len(all_jobs)} paragraphs...")
            texts = [job[1] for job in all_jobs]

            unique_texts = []
            text_to_idx = {}
            for text in texts:
                if text not in text_to_idx:
                    text_to_idx[text] = len(unique_texts)
                    unique_texts.append(text)

            translated_unique = []
            batch = []
            batch_tok_count = 0

            for utext in unique_texts:
                tok_count = len(utext.split())
                if batch and batch_tok_count + tok_count > BATCH_MAX_TOKENS:
                    translated_unique.extend(self._translate_batch(batch))
                    batch = []
                    batch_tok_count = 0
                batch.append(utext)
                batch_tok_count += tok_count

            if batch:
                translated_unique.extend(self._translate_batch(batch))

            translated_texts = [translated_unique[text_to_idx[text]] for text in texts]

            for i, (chain, _, eq_map) in enumerate(all_jobs):
                tr = self._restore_equations(translated_texts[i], eq_map)
                self._write_back_translated(chain, tr)

            if (i + 1) % 50 == 0:
                    print(f"[NLLB] Written back {i+1}/{len(all_jobs)} translations")

        # Translate tables separately
        for pi, page in enumerate(pages):
            blocks = page.get("preproc_blocks", page.get("para_blocks", []))
            for block in blocks:
                if block.get("type") == "table":
                    self._translate_table_block(block)

        elapsed = time.time() - t_start
        print(f"[NLLB] ✅ Layout translation complete in {elapsed:.1f}s ({len(all_jobs)} paragraphs)")
        return translated

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

    def _collect_translation_jobs(self, blocks: list) -> list:
        jobs = []
        i = 0
        while i < len(blocks):
            block = blocks[i]

            if self._contains_quartet_recursive(block) or block.get("lines"):
                chain = [block]
                j = i + 1
                while j < len(blocks) and blocks[j].get("merge_prev") is True:
                    chain.append(blocks[j])
                    j += 1

                paragraph, eq_map = self._extract_paragraph(chain)
                if paragraph.strip():
                    jobs.append((chain, paragraph, eq_map))

                if "blocks" in block:
                    jobs.extend(self._collect_translation_jobs(block["blocks"]))

                i = j
            elif "blocks" in block:
                jobs.extend(self._collect_translation_jobs(block["blocks"]))
                i += 1
            else:
                i += 1

        return jobs

    def _stitch_cross_page(self, pages: list):
        """Q7: Cross-page stitching."""
        for pi in range(len(pages) - 1):
            current_blocks = pages[pi].get("para_blocks", [])
            next_blocks = pages[pi + 1].get("para_blocks", [])
            if not current_blocks or not next_blocks:
                continue

            last_block = current_blocks[-1]
            has_cross_page = False
            for line in last_block.get("lines", []):
                for span in line.get("spans", []):
                    if span.get("cross_page", False):
                        has_cross_page = True
                        break
                if has_cross_page:
                    break

            if has_cross_page:
                first_next = next_blocks[0]
                merged_lines = last_block.get("lines", []) + first_next.get("lines", [])
                first_next["lines"] = merged_lines
                first_next["_cross_page_merged"] = True
                first_next["_original_page_line_count"] = len(last_block.get("lines", []))
                last_block["lines"] = []
                last_block["_merged_to_next"] = True
                print(f"[NLLB] Cross-page stitch: page {pi} → page {pi+1}")

    def _extract_paragraph(self, block_chain: list) -> tuple:
        parts = []
        eq_map = {}
        eq_idx = 0

        for block in block_chain:
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if self._is_quartet_text(span):
                        parts.append(span["content"])
                    elif span.get("type") == "inline_equation":
                        placeholder = f"[EQ_{eq_idx}]"
                        eq_map[placeholder] = span.get("content", "")
                        parts.append(placeholder)
                        eq_idx += 1
                    elif span.get("type") == "interline_equation":
                        placeholder = f"[EQ_{eq_idx}]"
                        eq_map[placeholder] = span.get("content", "")
                        parts.append(placeholder)
                        eq_idx += 1
                parts.append(" ")

        return " ".join(parts).strip(), eq_map

    def _restore_equations(self, text: str, eq_map: dict) -> str:
        for placeholder, original in sorted(eq_map.items(), key=lambda x: len(x[0]), reverse=True):
            text = text.replace(placeholder, original)
        return text

    def _write_back_translated(self, chain: list, translated_text: str):
        translated_words = translated_text.split()
        if not translated_words:
            return

        block_word_counts = []
        total_orig_words = 0

        for b in chain:
            b_words = 0
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    if self._is_quartet_text(span):
                        b_words += len(span.get("content", "").split())

            if b_words == 0 and b.get("merge_prev") is True:
                b_words = 1

            block_word_counts.append(b_words)
            total_orig_words += b_words

        word_idx = 0
        for i, b in enumerate(chain):
            if i == len(chain) - 1:
                block_words = translated_words[word_idx:]
            else:
                ratio = block_word_counts[i] / total_orig_words
                alloc_count = int(len(translated_words) * ratio)
                block_words = translated_words[word_idx : word_idx + alloc_count]
                word_idx += alloc_count

            block_translated_text = " ".join(block_words)

            b.pop("_merged", None)

            if block_translated_text:
                b["lines"] = [{
                    "bbox": b.get("bbox", [0, 0, 0, 0]),
                    "spans": [{
                        "bbox": b.get("bbox", [0, 0, 0, 0]),
                        "type": "text",
                        "content": block_translated_text,
                        "score": 1.0,
                        "translated": True,
                    }]
                }]
            else:
                b["lines"] = [{
                    "bbox": b.get("bbox", [0, 0, 0, 0]),
                    "spans": [{
                        "bbox": b.get("bbox", [0, 0, 0, 0]),
                        "type": "text",
                        "content": " ",
                        "score": 1.0,
                        "translated": True,
                    }]
                }]

    # ── Table translation ────────────────────────────────────────────

    def _translate_table_block(self, block: dict):
        for sub in block.get("blocks", []):
            for line in sub.get("lines", []):
                for span in line.get("spans", []):
                    if span.get("type") == "table" and "html" in span:
                        span["html"] = self._translate_table_html(span["html"])

    def _translate_table_html(self, html: str) -> str:
        def replace_cell(match):
            tag = match.group(1)
            content = match.group(2)
            close = match.group(3)

            eq_map = {}
            eq_idx = 0
            def protect_eq(m):
                nonlocal eq_idx
                key = f"[TEQ_{eq_idx}]"
                eq_map[key] = m.group(0)
                eq_idx += 1
                return key
            protected = re.sub(r"<eq>.*?</eq>", protect_eq, content)

            text_only = re.sub(r"<[^>]+>", "", protected).strip()
            if text_only:
                translated = self._translate_text(protected)
                for k, v in eq_map.items():
                    translated = translated.replace(k, v)
                return f"<{tag}>{translated}</{close}>"
            return match.group(0)

        return re.sub(
            r"<(t[dh][^>]*)>(.*?)</(t[dh])>",
            replace_cell,
            html,
            flags=re.DOTALL,
        )

    # ── Markdown-level translation ───────────────────────────────────

    def translate_markdown(self, md_content: str) -> str:
        lines = md_content.split("\n")
        total_lines = len(lines)
        translated_lines = []
        print(f"[NLLB] Translating markdown ({total_lines} lines)...")

        for li, line in enumerate(lines):
            if not line.strip():
                translated_lines.append("")
                continue
            if line.strip().startswith("\\[") or line.strip().startswith("$$") or line.strip().startswith("```"):
                translated_lines.append(line)
                continue

            protected_text, mapping = self._protect(line)
            translated_text = self._translate_text(protected_text)
            final_text = self._restore(translated_text, mapping)
            translated_lines.append(final_text)

            if (li + 1) % 20 == 0:
                print(f"[NLLB] Markdown: {li+1}/{total_lines} lines translated")

        print(f"[NLLB] ✅ Markdown translation complete ({total_lines} lines)")
        return "\n".join(translated_lines)

    def _protect(self, text: str) -> tuple:
        from collections import defaultdict
        placeholders = {}
        counts = defaultdict(int)

        all_matches = []
        for pattern, tag in PROTECTED_PATTERNS:
            for m in re.finditer(pattern, text):
                all_matches.append((m.start(), m.end(), m.group(), tag))

        all_matches.sort(key=lambda x: (x[0], -x[1]))
        filtered = []
        last_end = -1
        for s, e, c, t in all_matches:
            if s >= last_end:
                filtered.append((s, e, c, t))
                last_end = e

        result = text
        for s, e, c, t in sorted(filtered, key=lambda x: x[0], reverse=True):
            idx = counts[t]
            ph = f"[{t}_{idx}]"
            counts[t] += 1
            placeholders[ph] = c
            result = result[:s] + ph + result[e:]

        return result, placeholders

    def _restore(self, text: str, placeholders: dict) -> str:
        for ph, orig in sorted(placeholders.items(), key=lambda x: len(x[0]), reverse=True):
            text = text.replace(ph, orig)
        return text
