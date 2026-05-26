import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from backend.nllb_service import NLLBService


def test_rewrite_placeholders_basic():
    svc = NLLBService(lazy_load=True)
    paragraph = "Let [EQ_0] be equal to [EQ_1] and check [EQ_0] again."
    eq_map = {
        "[EQ_0]": {"type": "inline_equation", "content": "x"},
        "[EQ_1]": {"type": "inline_equation", "content": "y"},
    }
    
    # Rewrite placeholders starting from index 5
    rewritten, new_eq_map, next_idx = svc._rewrite_placeholders(paragraph, eq_map, 5)
    
    assert rewritten == "Let [EQ_5] be equal to [EQ_6] and check [EQ_5] again."
    assert next_idx == 7
    assert "[EQ_5]" in new_eq_map
    assert "[EQ_6]" in new_eq_map
    assert new_eq_map["[EQ_5]"]["content"] == "x"
    assert new_eq_map["[EQ_6]"]["content"] == "y"


def test_translate_middle_json_collects_discarded_and_standard_blocks():
    # Mock NLLBService to avoid loading full model weights for checking block extraction
    svc = NLLBService(lazy_load=True)
    
    middle_data = {
        "pdf_info": [
            {
                "page_idx": 0,
                "para_blocks": [
                    {
                        "type": "text",
                        "bbox": [0, 0, 100, 20],
                        "lines": [{
                            "bbox": [0, 0, 100, 20],
                            "spans": [{"bbox": [0, 0, 100, 20], "type": "text", "content": "Standard block", "score": 1.0}]
                        }]
                    }
                ],
                "discarded_blocks": [
                    {
                        "type": "page_footnote",
                        "bbox": [0, 90, 100, 100],
                        "lines": [{
                            "bbox": [0, 90, 100, 100],
                            "spans": [{"bbox": [0, 90, 100, 100], "type": "text", "content": "Discarded block note", "score": 1.0}]
                        }]
                    }
                ]
            }
        ]
    }
    
    # We should run translate_middle_json with a mock load/generate to test routing/logic
    svc.loaded = True
    
    def mock_translate_text(text):
        return f"Translated: {text}"
        
    svc._translate_text = mock_translate_text
    
    # Mock unload_model to do nothing
    svc.unload_model = lambda: None
    
    res = svc.translate_middle_json(middle_data)
    
    page = res["pdf_info"][0]
    
    # Check that standard block got translated
    assert page["para_blocks"][0]["lines"][0]["spans"][0]["content"] == "Translated: Standard block"
    
    # Check that discarded block (footnote) also got translated!
    assert page["discarded_blocks"][0]["lines"][0]["spans"][0]["content"] == "Translated: Discarded block note"

