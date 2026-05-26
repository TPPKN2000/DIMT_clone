import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backend.nllb_service import NLLBService
from backend.pdf_renderer import PDFRenderer


def test_extract_paragraph_protects_inline_equation():
    svc = NLLBService(lazy_load=True)
    chain = [{
        "bbox": [0, 0, 100, 20],
        "lines": [{
            "bbox": [0, 0, 100, 20],
            "spans": [
                {"bbox": [0, 0, 30, 20], "type": "text", "content": "Let", "score": 1.0},
                {
                    "bbox": [30, 0, 60, 20],
                    "type": "inline_equation",
                    "content": r"\phi=\bar{\phi}+\phi'",
                    "score": 0.99,
                },
                {"bbox": [60, 0, 100, 20], "type": "text", "content": "be valid", "score": 1.0},
            ],
        }],
    }]

    paragraph, eq_map = svc._extract_paragraph(chain)

    assert "[EQ_0]" in paragraph
    assert eq_map["[EQ_0]"]["type"] == "inline_equation"
    assert eq_map["[EQ_0]"]["content"] == r"\phi=\bar{\phi}+\phi'"


def test_write_back_preserves_inline_equation_span():
    svc = NLLBService(lazy_load=True)
    chain = [{
        "bbox": [0, 0, 100, 20],
        "lines": [{
            "bbox": [0, 0, 100, 20],
            "spans": [
                {"bbox": [0, 0, 40, 20], "type": "text", "content": "Formula", "score": 1.0},
                {"bbox": [40, 0, 70, 20], "type": "inline_equation", "content": r"\phi", "score": 0.95},
                {"bbox": [70, 0, 100, 20], "type": "text", "content": "works", "score": 1.0},
            ],
        }],
    }]
    eq_map = {
        "[EQ_0]": {
            "type": "inline_equation",
            "content": r"\phi",
            "bbox": [40, 0, 70, 20],
            "score": 0.95,
        }
    }

    svc._write_back_translated_with_equations(
        chain,
        "La formule [EQ_0] fonctionne",
        eq_map,
    )

    spans = chain[0]["lines"][0]["spans"]
    assert any(s["type"] == "inline_equation" and s["content"] == r"\phi" for s in spans)
    assert any(s["type"] == "text" and "La formule" in s["content"] for s in spans)


def test_placeholder_validator_appends_missing_equation_before_write_back():
    svc = NLLBService(lazy_load=True)
    eq_map = {"[EQ_0]": r"\phi"}

    assert svc._validate_placeholders("La formule fonctionne", eq_map) == ["[EQ_0]"]
    assert svc._validate_placeholders("La formule [EQ_0] fonctionne", eq_map) == []


def test_pdf_renderer_extracts_inline_equation_as_eq_token():
    renderer = PDFRenderer.__new__(PDFRenderer)
    renderer.images_dir = None
    page_data = {
        "para_blocks": [{
            "type": "text",
            "bbox": [0, 0, 100, 30],
            "lines": [{
                "bbox": [0, 0, 100, 30],
                "spans": [
                    {"bbox": [0, 0, 40, 30], "type": "text", "content": "Formula", "score": 1.0},
                    {"bbox": [40, 0, 70, 30], "type": "inline_equation", "content": r"\phi", "score": 1.0},
                ],
            }],
        }]
    }

    blocks = renderer.extract_page_blocks(page_data)

    assert blocks
    tokens = blocks[0]["tokens"]
    assert any(kind == "eq" and isinstance(content, dict) and content["content"] == r"\phi" for kind, content in tokens)
