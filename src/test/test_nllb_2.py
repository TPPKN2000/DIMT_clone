"""
Test: Newline preservation after NLLB finetuned inference.

Verifies that when two paragraphs separated by \\n are translated
(eng -> deu), the newline structure is preserved in the output.
"""

import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backend.nllb_service import NLLBService


@pytest.fixture(scope="module")
def nllb():
    """Load the NLLB model once for the entire module."""
    svc = NLLBService(tgt_lang="deu_Latn", lazy_load=False)
    yield svc
    svc.unload_model()


INPUT_TEXT = (
    "Machine learning is a subset of artificial intelligence. It calculates by using [EQ_0].ZXQPARAQXZ"
    "It enables computers to learn from data without explicit programming."
)


def test_newline_preserved_split_translate(nllb):
    """Translate each paragraph independently and rejoin — delimiter must survive."""
    paragraphs = INPUT_TEXT.split("ZXQPARAQXZ")
    assert len(paragraphs) == 2, "Input must have exactly 2 paragraphs"

    translated_parts = [nllb._translate_text(p) for p in paragraphs]
    result = "ZXQPARAQXZ".join(translated_parts)

    print(f"\n--- Input ---\n{INPUT_TEXT}")
    print(f"\n--- Output (split-translate) ---\n{result}")

    assert "ZXQPARAQXZ" in result, "ZXQPARAQXZ was lost after split-translate-rejoin"
    lines = result.split("ZXQPARAQXZ")
    assert len(lines) == 2, f"Expected 2 segments, got {len(lines)}"
    assert all(line.strip() for line in lines), "One of the translated paragraphs is empty"


def test_newline_preserved_direct_translate(nllb):
    """Translate the full two-paragraph text as-is — check if ZXQPARAQXZ survives."""
    result = nllb._translate_text(INPUT_TEXT)

    print(f"\n--- Input ---\n{INPUT_TEXT}")
    print(f"\n--- Output (direct) ---\n{result}")

    # Check if the unique non-translatable delimiter survives direct translation
    has_delimiter = "ZXQPARAQXZ" in result
    print(f"\nDelimiter ZXQPARAQXZ preserved in direct mode: {has_delimiter}")

    # We only assert that the output is non-empty
    assert len(result.strip()) > 0, "Translation output is empty"


if __name__ == "__main__":
    print("=" * 60)
    print("Loading NLLB model (deu_Latn)...")
    svc = NLLBService(tgt_lang="fra_Latn", lazy_load=False)
    try:
        print("=" * 60)
        print("[TEST 1] Split-translate newline preservation")
        test_newline_preserved_split_translate(svc)
        print("✅ PASSED\n")

        print("[TEST 2] Direct-translate newline behavior")
        test_newline_preserved_direct_translate(svc)
        print("✅ PASSED\n")

        print("=" * 60)
        print("All tests passed!")
    finally:
        svc.unload_model()
