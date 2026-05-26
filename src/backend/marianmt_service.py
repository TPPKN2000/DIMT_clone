"""
MarianMT Translation Service — drop-in model alternative configured via inference.yaml.
"""

import os
import re
import copy
import time
import torch
import threading
import yaml
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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


class MarianMTService:
    SUPPORTED_LANGS = ["fra_Latn", "deu_Latn", "fr", "de"]

    def __init__(self, device=None, tgt_lang: str = "fra_Latn", lazy_load: bool = False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.loaded = False
        self.tgt_lang = tgt_lang
        self._load_event = threading.Event()

        # Load configuration from inference.yaml
        config_path = Path(__file__).resolve().parents[2] / "config" / "inference.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            full_config = yaml.safe_load(f)
            self.config = full_config["marianmt"]

        if not lazy_load:
            self.load_model()

    def load_model(self):
        if self.loaded:
            return
        try:
            # Map tgt_lang to fr/de keys
            lang_key = "fr" if "fr" in self.tgt_lang.lower() else "de"
            checkpoint = self.config["models"].get(lang_key)
            
            print(f"[MarianMT] Loading {checkpoint} on {self.device}...")
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()

            self.loaded = True
            print(f"[MarianMT] Model {checkpoint} loaded successfully")
        except Exception:
            import traceback
            print("[MarianMT] Load failed:")
            traceback.print_exc()
        finally:
            self._load_event.set()

    def wait_for_load(self):
        self._load_event.wait()

    def unload_model(self):
        """Unload tokenizer and model from VRAM and run garbage collection."""
        if not self.loaded:
            return
        print("[MarianMT] Unloading model to free VRAM...")
        try:
            if hasattr(self, "model"):
                del self.model
            if hasattr(self, "tokenizer"):
                del self.tokenizer
            self.model = None
            self.tokenizer = None
            self.loaded = False
            self._load_event.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            print("[MarianMT] Model unloaded and VRAM cache cleared successfully.")
        except Exception as e:
            print(f"[MarianMT] Failed to unload model: {e}")

    def set_target_lang(self, lang_code: str):
        """Switch target language."""
        if lang_code not in self.SUPPORTED_LANGS:
            raise ValueError(f"Unsupported language: {lang_code}. Use one of {self.SUPPORTED_LANGS}")
        self.tgt_lang = lang_code

    # ── Core translate ──────────────────────────────────────────────

    def _translate_text(self, text: str) -> str:
        """Translate a single string using MarianMT."""
        if not text.strip():
            return text
        if not self.loaded:
            return f"[marianMT:{self.tgt_lang}] {text}"

        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True,
            max_length=self.config.get("max_source_length", 256)
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.config.get("max_new_tokens", 256),
                num_beams=self.config.get("num_beams", 4),
            )
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()

    def get_dynamic_batch_size(self) -> int:
        """Calculate dynamic batch size based on currently available VRAM."""
        if not torch.cuda.is_available():
            return 4  # Default CPU batch size

        try:
            device_idx = torch.cuda.current_device()
            free_mem, total_mem = torch.cuda.mem_get_info(device_idx)
            free_mem_mb = free_mem / (1024 * 1024)
            total_mem_mb = total_mem / (1024 * 1024)

            print(f"[MarianMT] GPU Info: Total VRAM = {total_mem_mb:.1f}MB, Available VRAM = {free_mem_mb:.1f}MB")

            # Determine batch size strictly based on AVAILABLE VRAM (free_mem_mb)
            if free_mem_mb < 500:
                batch_size = 1
            elif free_mem_mb < 1000:
                batch_size = 2
            elif free_mem_mb < 1800:
                batch_size = 4
            elif free_mem_mb < 3000:
                batch_size = 6
            else:
                # Reserve 1200MB for model and system base overhead, then 300MB per sample
                usable_vram = free_mem_mb - 1200
                calculated_batch = int(usable_vram / 300)
                batch_size = max(6, calculated_batch)

            # Limit batch size to a safe maximum
            batch_size = min(batch_size, 32)
            print(f"[MarianMT] Hardware-driven batch size selected: {batch_size} (based on {free_mem_mb:.1f}MB available VRAM)")
            return batch_size
        except Exception as e:
            print(f"[MarianMT] Failed to calculate dynamic batch size: {e}. Defaulting to 4.")
            return 4

    def _translate_batch(self, texts: list[str]) -> list[str]:
        """Translate a batch of strings using MarianMT."""
        if not texts:
            return []
        if not self.loaded:
            return [f"[marianMT:{self.tgt_lang}] {t}" for t in texts]

        # Tokenize the batch of texts with padding
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True,
            max_length=self.config.get("max_source_length", 256)
        ).to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.config.get("max_new_tokens", 256),
                num_beams=self.config.get("num_beams", 4),
            )
        decoded = self.tokenizer.batch_decode(out, skip_special_tokens=True)
        return [text.strip() for text in decoded]

    # ── Paragraph extraction from layout.json ───────────────────────

    def translate_middle_json(self, middle_data: dict) -> dict:
        """
        Translate all translatable content in layout.json in-place.
        """
        t_start = time.time()
        if not self.loaded:
            self.load_model()
        try:
            translated = copy.deepcopy(middle_data)
            pages = translated.get("pdf_info", [])
            total_pages = len(pages)

            # Q7: Cross-page stitching
            self._stitch_cross_page(pages)

            all_jobs = []

            for pi, page in enumerate(pages):
                blocks = page.get("preproc_blocks", page.get("para_blocks", [])) + page.get("discarded_blocks", [])
                jobs = self._collect_translation_jobs(blocks)
                all_jobs.extend(jobs)
                if (pi + 1) % 5 == 0 or pi == total_pages - 1:
                    print(f"[MarianMT] Collected jobs from page {pi+1}/{total_pages} (total jobs: {len(all_jobs)})")

            if all_jobs:
                batch_size = self.get_dynamic_batch_size()
                print(f"[MarianMT] Translating {len(all_jobs)} paragraphs in batches of {batch_size}...")
                for i in range(0, len(all_jobs), batch_size):
                    batch_jobs = all_jobs[i:i+batch_size]
                    batch_paragraphs = [job[1] for job in batch_jobs]
                    translated_texts = self._translate_batch(batch_paragraphs)
                    
                    for job_idx, (chain, paragraph, eq_map) in enumerate(batch_jobs):
                        translated_text = translated_texts[job_idx] if job_idx < len(translated_texts) else ""
                        missing = self._validate_placeholders(translated_text, eq_map)
                        if missing:
                            # Step 1: Retry translation once
                            retry_texts = self._translate_batch([paragraph])
                            retry_text = retry_texts[0] if retry_texts else translated_text
                            still_missing = self._validate_placeholders(retry_text, eq_map)
                            if not still_missing:
                                print(
                                    f"[MarianMT] Retry succeeded — recovered placeholders "
                                    f"{missing} on 2nd attempt"
                                )
                                translated_text = retry_text
                            else:
                                # Step 2: Use LLM fallback to insert missing tags
                                print(
                                    f"[MarianMT Warning] Missing equation placeholders {still_missing} "
                                    f"after retry. Source={paragraph[:120]!r} "
                                    f"Output={retry_text[:120]!r}. Trying LLM fix..."
                                )
                                llm_fixed = self._fix_missing_placeholders_with_llm(
                                    paragraph, retry_text, still_missing
                                )
                                if llm_fixed:
                                    final_missing = self._validate_placeholders(llm_fixed, eq_map)
                                    if not final_missing:
                                        print(f"[MarianMT] LLM fix succeeded for {still_missing}")
                                        translated_text = llm_fixed
                                    else:
                                        print(
                                            f"[MarianMT Warning] LLM fix still missing {final_missing}. "
                                            f"Appending to end."
                                        )
                                        translated_text = llm_fixed.rstrip() + " " + " ".join(final_missing)
                                else:
                                    # Final fallback: append missing placeholders
                                    translated_text = retry_text.rstrip() + " " + " ".join(still_missing)

                        self._write_back_translated_with_equations(
                            chain=chain,
                            translated_text=translated_text,
                            eq_map=eq_map,
                        )

                    completed_count = i + len(batch_jobs)
                    if completed_count % 50 == 0 or completed_count >= len(all_jobs):
                        print(f"[MarianMT] Written back {completed_count}/{len(all_jobs)} translations")

            # Translate tables recursively in both standard and discarded blocks
            for pi, page in enumerate(pages):
                blocks = page.get("preproc_blocks", page.get("para_blocks", [])) + page.get("discarded_blocks", [])
                self._translate_tables_recursive(blocks)

            elapsed = time.time() - t_start
            print(f"[MarianMT] ✅ Layout translation complete in {elapsed:.1f}s ({len(all_jobs)} paragraphs)")
            return translated
        finally:
            self.unload_model()

    def _translate_tables_recursive(self, blocks: list):
        for block in blocks:
            if block.get("type") == "table":
                self._translate_table_block(block)
            if "blocks" in block:
                self._translate_tables_recursive(block["blocks"])

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
            if block.get("type") == "interline_equation":
                i += 1
                continue

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
        """Cross-page stitching."""
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
                print(f"[MarianMT] Cross-page stitch: page {pi} → page {pi+1}")

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
                        eq_map[placeholder] = {
                            "type": "inline_equation",
                            "content": span.get("content", ""),
                            "bbox": span.get("bbox"),
                            "score": span.get("score", 1.0),
                        }
                        parts.append(placeholder)
                        eq_idx += 1
                parts.append(" ")

        return " ".join(parts).strip(), eq_map

    def _restore_equations(self, text: str, eq_map: dict) -> str:
        for placeholder, original in sorted(eq_map.items(), key=lambda x: len(x[0]), reverse=True):
            content = original.get("content", "") if isinstance(original, dict) else original
            text = text.replace(placeholder, content)
        return text

    def _validate_placeholders(self, translated_text: str, eq_map: dict) -> list[str]:
        return [placeholder for placeholder in eq_map if placeholder not in translated_text]

    def _fix_missing_placeholders_with_llm(
        self, source_text: str, output_text: str, missing_tags: list[str]
    ) -> str | None:
        """
        Use LLM (Gemini flash-lite) zero-shot to re-insert missing equation
        placeholders into the translated text at appropriate positions.
        Returns the fixed text, or None if LLM call fails.
        """
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            api_key = os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                print("[MarianMT] No GEMINI_API_KEY — skipping LLM fix")
                return None

            llm = ChatGoogleGenerativeAI(
                model="gemini-flash-lite-latest",
                google_api_key=api_key,
                temperature=0.0,
            )

            tags_str = ", ".join(missing_tags)
            prompt = (
                f"## REFERENCE TEXT\n{source_text}\n\n"
                f"## OUTPUT TEXT\n{output_text}\n\n"
                f"## TASK\n"
                f"My machine translation model dropped these tags: {tags_str}\n"
                f"Based on the REFERENCE TEXT, insert the missing tags at the "
                f"correct positions in the OUTPUT TEXT.\n"
                f"Output ONLY the corrected OUTPUT TEXT with all tags in place. "
                f"Do not add any explanation or markdown fences."
            )

            resp = llm.invoke(prompt)
            content = resp.content if hasattr(resp, "content") else str(resp)
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            result = content.strip()
            if not result:
                return None
            return result
        except Exception as e:
            print(f"[MarianMT] LLM placeholder fix failed: {e}")
            return None

    def _split_text_and_equations(self, text: str, eq_map: dict) -> list[dict]:
        if not text:
            return []

        if not eq_map:
            return [{"type": "text", "content": text, "score": 1.0, "translated": True}]

        pieces = []
        for part in re.split(r"(\[EQ_\d+\])", text):
            if not part:
                continue

            if part in eq_map:
                eq = eq_map[part]
                if isinstance(eq, dict):
                    pieces.append({
                        "type": eq.get("type", "inline_equation"),
                        "content": eq.get("content", ""),
                        "bbox": eq.get("bbox"),
                        "score": eq.get("score", 1.0),
                        "translated": False,
                    })
                else:
                    pieces.append({
                        "type": "inline_equation",
                        "content": eq,
                        "score": 1.0,
                        "translated": False,
                    })
                continue

            pieces.append({
                "type": "text",
                "content": part,
                "score": 1.0,
                "translated": True,
            })

        return pieces

    def _write_back_translated_with_equations(
        self,
        chain: list,
        translated_text: str,
        eq_map: dict,
    ):
        pieces = self._split_text_and_equations(translated_text, eq_map)
        if not pieces:
            return

        block_word_counts = []
        total_orig_words = 0

        for block in chain:
            count = 0
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    if self._is_quartet_text(span):
                        count += len(span.get("content", "").split())

            if count == 0 and block.get("merge_prev") is True:
                count = 1

            block_word_counts.append(count)
            total_orig_words += count

        if total_orig_words <= 0:
            total_orig_words = len(chain)
            block_word_counts = [1 for _ in chain]

        total_translated_words = sum(
            len(piece["content"].split())
            for piece in pieces
            if piece["type"] == "text"
        )
        total_translated_words = max(total_translated_words, 1)

        piece_idx = 0
        text_words_seen = 0

        for block_idx, block in enumerate(chain):
            bbox = block.get("bbox", [0, 0, 0, 0])
            block.pop("_merged", None)

            if block_idx == len(chain) - 1:
                target_text_words_until_here = total_translated_words
            else:
                ratio = sum(block_word_counts[: block_idx + 1]) / total_orig_words
                target_text_words_until_here = max(1, int(total_translated_words * ratio))

            block_pieces = []

            while piece_idx < len(pieces):
                piece = pieces[piece_idx]

                if piece["type"] in {"inline_equation", "interline_equation"}:
                    block_pieces.append(piece)
                    piece_idx += 1
                    continue

                words = piece["content"].split()
                if not words:
                    piece_idx += 1
                    continue

                remaining_quota = target_text_words_until_here - text_words_seen

                if block_idx != len(chain) - 1 and remaining_quota <= 0:
                    break

                if block_idx == len(chain) - 1 or len(words) <= remaining_quota:
                    block_pieces.append(piece)
                    text_words_seen += len(words)
                    piece_idx += 1
                    continue

                left_words = words[:remaining_quota]
                right_words = words[remaining_quota:]

                if left_words:
                    block_pieces.append({
                        **piece,
                        "content": " ".join(left_words),
                    })
                    text_words_seen += len(left_words)

                pieces[piece_idx] = {
                    **piece,
                    "content": " ".join(right_words),
                }
                break

            if not block_pieces:
                block_pieces = [{
                    "type": "text",
                    "content": " ",
                    "score": 1.0,
                    "translated": True,
                }]

            spans = []
            for piece in block_pieces:
                span_bbox = piece.get("bbox") or bbox
                spans.append({
                    "bbox": span_bbox,
                    "type": piece["type"],
                    "content": piece["content"],
                    "score": piece.get("score", 1.0),
                    "translated": piece.get("translated", piece["type"] == "text"),
                })

            block["lines"] = [{
                "bbox": bbox,
                "spans": spans,
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
        if not self.loaded:
            self.load_model()
        try:
            lines = md_content.split("\n")
            total_lines = len(lines)
            translated_lines = []
            print(f"[MarianMT] Translating markdown ({total_lines} lines)...")

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
                    print(f"[MarianMT] Markdown: {li+1}/{total_lines} lines translated")

            print(f"[MarianMT] ✅ Markdown translation complete ({total_lines} lines)")
            return "\n".join(translated_lines)
        finally:
            self.unload_model()

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
