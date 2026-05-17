"""
Bilingual Mapping — Bước 6.

Builds a structured paragraph-level source↔translation mapping from:
  - original middle_json  (source text extracted by MinerU)
  - translated_middle     (translated version produced by NLLB)

Each pair carries enough metadata (page, bbox, block type) for downstream
uses: re-translation, diff display, export to TMX/bilingual DOCX, or
future fine-tuning data collection.

Schema of one entry
───────────────────
{
  "pair_id":    str,           # stable hash of src text
  "page":       int,
  "block_type": str,           # "text" | "title" | "table" | ...
  "bbox":       [x0,y0,x1,y1],
  "source":     str,           # original English text
  "translation": str,          # translated text
  "tgt_lang":   str,           # e.g. "vie_Latn"
  "human_edited": bool,        # True if HITL update was applied
}

Storage in MongoDB (via MongoDocStore)
──────────────────────────────────────
Key:  "bilingual:{doc_id}"
Value: JSON-encoded list of pair dicts (≤5 MB per doc limit respected by
       storing only the text, not image data).
"""

import hashlib
import json
from typing import Optional


# ── Text extraction helpers ─────────────────────────────────────────

def _text_from_block(block: dict) -> str:
    """Flatten all text spans inside a block into a single string."""
    parts = []
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            c = span.get("content", "").strip()
            if c:
                parts.append(c)
    return " ".join(parts).strip()


def _iter_leaf_blocks(layout_data: dict):
    """
    Yield (page_idx, block) for every block that contains text lines,
    walking the full nested structure (para_blocks + preproc_blocks).
    """
    for page in layout_data.get("pdf_info", []):
        page_idx = page.get("page_idx", 0)
        root_blocks = page.get("preproc_blocks", page.get("para_blocks", []))
        yield from _walk_blocks(root_blocks, page_idx)


def _walk_blocks(blocks: list, page_idx: int):
    for block in blocks:
        if block.get("lines"):
            yield page_idx, block
        if block.get("blocks"):
            yield from _walk_blocks(block["blocks"], page_idx)


def _pair_id(src_text: str) -> str:
    """Short stable ID derived from source text content."""
    return hashlib.sha1(src_text.encode("utf-8", errors="replace")).hexdigest()[:12]


# ── Core builder ───────────────────────────────────────────────────

def build_bilingual_mapping(
    original_middle: dict,
    translated_middle: dict,
    tgt_lang: str,
) -> list[dict]:
    """
    Walk both layout trees in lock-step and pair up source ↔ translation.

    Strategy:
      Both trees have identical structure (translated_middle is a deep copy
      with only span content replaced).  We iterate both simultaneously by
      page_idx and block position, matching on bbox equality as a sanity
      check.  Mismatched blocks are still included but flagged.
    """
    pairs: list[dict] = []

    # Build a flat list of (page_idx, block) from each tree
    orig_blocks = list(_iter_leaf_blocks(original_middle))
    trans_blocks = list(_iter_leaf_blocks(translated_middle))

    # Zip by position — lengths should match; zip stops at the shorter
    for (o_page, o_block), (t_page, t_block) in zip(orig_blocks, trans_blocks):
        src_text = _text_from_block(o_block)
        tgt_text = _text_from_block(t_block)

        # Skip empty pairs (equations-only blocks, whitespace blocks, etc.)
        if not src_text.strip() and not tgt_text.strip():
            continue

        bbox = o_block.get("bbox") or t_block.get("bbox")
        bbox_match = (o_block.get("bbox") == t_block.get("bbox"))

        pairs.append({
            "pair_id":     _pair_id(src_text),
            "page":        o_page,
            "block_type":  o_block.get("type", "text"),
            "bbox":        bbox,
            "source":      src_text,
            "translation": tgt_text,
            "tgt_lang":    tgt_lang,
            "human_edited": bool(
                # Mark as human-edited if any span carries the flag
                any(
                    span.get("human_edited", False)
                    for line in t_block.get("lines", [])
                    for span in line.get("spans", [])
                )
            ),
            "_bbox_match": bbox_match,   # internal QA flag, not shown in UI
        })

    # Append any leftover orig blocks that had no translation counterpart
    for o_page, o_block in orig_blocks[len(trans_blocks):]:
        src_text = _text_from_block(o_block)
        if src_text.strip():
            pairs.append({
                "pair_id":     _pair_id(src_text),
                "page":        o_page,
                "block_type":  o_block.get("type", "text"),
                "bbox":        o_block.get("bbox"),
                "source":      src_text,
                "translation": "",
                "tgt_lang":    tgt_lang,
                "human_edited": False,
                "_bbox_match": False,
            })

    print(f"[Bilingual] Built {len(pairs)} pairs "
          f"(orig_blocks={len(orig_blocks)}, trans_blocks={len(trans_blocks)})")
    return pairs


# ── Persistence helpers ─────────────────────────────────────────────

BILINGUAL_KEY_PREFIX = "bilingual:"


def save_bilingual_mapping(doc_id: str, pairs: list[dict], doc_store) -> None:
    """Persist pairs to MongoDB under key 'bilingual:{doc_id}'."""
    key = f"{BILINGUAL_KEY_PREFIX}{doc_id}"
    doc_store.set(key, {"pairs": pairs, "doc_id": doc_id, "count": len(pairs)})
    print(f"[Bilingual] Saved {len(pairs)} pairs → key={key}")


def load_bilingual_mapping(doc_id: str, doc_store) -> Optional[list[dict]]:
    """Load and return pairs, or None if not yet generated."""
    key = f"{BILINGUAL_KEY_PREFIX}{doc_id}"
    record = doc_store.get(key)
    if record is None:
        return None
    return record.get("pairs", [])


def apply_hitl_edit_to_mapping(
    doc_id: str,
    page_idx: int,
    block_idx: int,
    new_text: str,
    translated_middle: dict,
    doc_store,
) -> None:
    """
    After a HITL edit, find the matching pair by page+bbox and update it.
    Called from api.py /hitl/update so the bilingual export stays consistent.
    """
    pairs = load_bilingual_mapping(doc_id, doc_store)
    if pairs is None:
        return  # Mapping not built yet — nothing to update

    # Resolve the edited block's bbox from translated_middle
    pages = translated_middle.get("pdf_info", [])
    if page_idx >= len(pages):
        return
    blocks = pages[page_idx].get("para_blocks", [])
    if block_idx >= len(blocks):
        return
    edited_bbox = blocks[block_idx].get("bbox")

    updated = 0
    for pair in pairs:
        if pair.get("page") == page_idx and pair.get("bbox") == edited_bbox:
            pair["translation"] = new_text
            pair["human_edited"] = True
            updated += 1

    if updated:
        save_bilingual_mapping(doc_id, pairs, doc_store)
        print(f"[Bilingual] HITL edit applied to {updated} pair(s) "
              f"— page={page_idx}, block={block_idx}")
    else:
        print(f"[Bilingual] HITL edit: no matching pair found "
              f"for page={page_idx}, block={block_idx}")
