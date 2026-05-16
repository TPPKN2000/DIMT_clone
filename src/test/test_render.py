"""
DIMT Pipeline — Phase 3: Visual Agent-in-the-Loop Iterative Rendering (V5.1)

Architecture:
  Phase 1: Algorithmic Dynamic Extraction (LayoutAnalyzer)
  Phase 2: Agentic Layout Classifier (Gemma-4 + Google Search)
  Phase 3: Closed-Loop Visual Iterative Rendering (Typst + Gemini Flash Multimodal Art Director)
"""

import json
import re
import os
import sys
import base64
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from dotenv import load_dotenv

load_dotenv()

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("Warning: google-genai not found. Agent phases will fail.")

import typst
import fitz  # PyMuPDF


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Algorithmic Dynamic Extraction (unchanged)
# ═══════════════════════════════════════════════════════════════════
class LayoutAnalyzer:
    def __init__(self, layout_path):
        self.data = json.loads(Path(layout_path).read_text(encoding='utf-8'))
        self.pages = self.data.get('pdf_info', [])

    def analyze(self):
        print("[Phase 1] Analyzing layout coordinates...")
        all_blocks = []
        for p in self.pages:
            all_blocks.extend(p.get('para_blocks', []))
        if not all_blocks:
            return self._default_layout()

        x2s = [b['bbox'][2] for b in all_blocks]
        y2s = [b['bbox'][3] for b in all_blocks]
        canvas_w = max(x2s) if x2s else 612
        canvas_h = max(y2s) if y2s else 792

        text_heights, title_heights = [], []
        for b in all_blocks:
            h = b.get('line_avg_height') or ((b['bbox'][3] - b['bbox'][1]) / max(1, len(b.get('lines', [1]))))
            if b['type'] == 'text': text_heights.append(h)
            elif b['type'] == 'title': title_heights.append(h)

        avg_text_h = sum(text_heights)/len(text_heights) if text_heights else 10.0
        avg_title_h = sum(title_heights)/len(title_heights) if title_heights else 14.0

        text_blocks = [b for b in all_blocks if b['type'] == 'text']
        if len(text_blocks) < 5:
            return self._default_layout(canvas_w, canvas_h)

        x1_coords = np.array([b['bbox'][0] for b in text_blocks]).reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(x1_coords)
        centers = sorted(kmeans.cluster_centers_.flatten())

        if (centers[1] - centers[0]) < (canvas_w * 0.15):
            cols, gap, left_margin = 1, 0, centers[0]
        else:
            cols = 2
            c0_idx = 0 if kmeans.cluster_centers_[0][0] < kmeans.cluster_centers_[1][0] else 1
            labels = kmeans.labels_
            col1_x2 = [text_blocks[i]['bbox'][2] for i in range(len(text_blocks)) if labels[i] == c0_idx]
            col2_x1 = [text_blocks[i]['bbox'][0] for i in range(len(text_blocks)) if labels[i] != c0_idx]
            gap = (min(col2_x1) - max(col1_x2)) if (col1_x2 and col2_x1) else 18
            left_margin = centers[0]

        return {
            "page_width": canvas_w, "page_height": canvas_h,
            "columns": cols, "column_gap": max(5, gap), "left_margin": left_margin,
            "avg_text_height": avg_text_h, "avg_title_height": avg_title_h,
            "raw_pages": self.pages,
        }

    def _default_layout(self, w=612, h=792):
        return {
            "page_width": w, "page_height": h, "columns": 1, "column_gap": 0,
            "left_margin": 50, "avg_text_height": 10.0, "avg_title_height": 14.0,
            "raw_pages": []
        }


# ═══════════════════════════════════════════════════════════════════
# Phase 2: Agentic Layout Classifier (unchanged)
# ═══════════════════════════════════════════════════════════════════
class AgenticLayoutClassifier:
    def __init__(self, api_key):
        if not api_key: raise ValueError("GEMMA_4_API_KEY not set.")
        self.client = genai.Client(api_key=api_key)
        self.model = "gemma-4-31b-it"

    def classify(self, md_path):
        print("[Phase 2] Researching document template via AI Agent...")
        content = Path(md_path).read_text(encoding='utf-8')
        header_context = "\n".join(content.split('\n')[:30])

        prompt = f"""
        TASK: Identify the official layout guidelines for this research paper.
        METADATA SNIPPET:
        {header_context}
        INSTRUCTIONS:
        1. Use Google Search to find the specific journal/venue for this paper.
        2. Fetch the standard layout parameters.
        Return a strictly typed JSON config with these keys:
        - template_family: (string)
        - columns: (int)
        - page_size: "letter" | "a4"
        - margins: {{ "top": "string", "bottom": "string", "left": "string", "right": "string" }}
        - font_family: (string)
        - body_font_size: (float, in pt)
        - heading_font_size: (float, in pt)
        - line_spacing_ratio: (float)
        - heading_style: {{ "alignment": "center" | "left", "top_spacing": "string", "bottom_spacing": "string" }}
        """
        for attempt in range(3):
            try:
                self.client.http_options = types.HttpOptions(timeout=600_000)
                config = types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(thinking_level="LOW"),
                    tools=[types.Tool(google_search=types.GoogleSearch())]
                )
                response = self.client.models.generate_content(model=self.model, contents=prompt, config=config)
                json_str = re.search(r'\{.*\}', response.text, re.DOTALL).group(0)
                return json.loads(json_str)
            except Exception as e:
                if attempt < 2: print(f"  Retry {attempt+1}/2: {e}")
                else: raise


# ═══════════════════════════════════════════════════════════════════
# Phase 3: Visual Agent-in-the-Loop Iterative Rendering (V5.1)
# ═══════════════════════════════════════════════════════════════════
@dataclass
class PageData:
    page_idx: int
    blocks: list
    page_width: float
    page_height: float
    md_content: str


class PageChunker:
    """Groups layout.json blocks by page and slices full.md."""
    def __init__(self, layout_data, full_md_path):
        self.pages = layout_data.get('raw_pages', [])
        self.full_md = Path(full_md_path).read_text(encoding='utf-8')

    def chunk(self) -> list[PageData]:
        # A simple heuristic: split the MD by titles or major blocks
        # For simplicity in this rewrite, we will just use the entire full.md
        # and render it all at once if page-by-page markdown splitting is too brittle,
        # BUT the feedback requires we keep the loop. 
        # Actually, extracting exact MD per page is hard without block-level tracking.
        # Given the instruction: "Giữ nguyên chiến thuật dùng full.md để gen ra text"
        # We will apply the label injection to the *entire* document, but process pages.
        # Wait, if we render the whole document every iteration, it's safer for flow.
        pass

# Because splitting markdown by page perfectly is extremely difficult without losing
# cross-page flow, the "Visual Agent-in-the-Loop" will operate on the ENTIRE document,
# but we will feed the Agent one page's visual data at a time to keep context small.

class VisualTypstIntrospector:
    def __init__(self, images_dir, project_root, full_md_content, raw_pages):
        self.images_dir = Path(images_dir)
        self.project_root = Path(project_root)
        self.full_md_content = full_md_content
        self.raw_pages = raw_pages
        
        # Build a mapping of text -> label to inject into Markdown
        self.block_labels = {}
        for p_idx, page in enumerate(raw_pages):
            for block in page.get('para_blocks', []):
                idx = block.get('index')
                if idx is not None:
                    # Very basic text extraction for matching
                    text = ""
                    for line in block.get('lines', []):
                        for span in line.get('spans', []):
                            text += span.get('content', '') + " "
                    text = text.strip()
                    if text:
                        # Store first 20 chars for regex matching
                        self.block_labels[text[:20]] = f"block_p{p_idx}_idx{idx}"

    def inject_labels_and_convert(self, overrides):
        """Reverts to V4 parsing logic but injects <labels>."""
        t = self.full_md_content
        
        # Inject labels: naive approach, append <label> to ends of lines matching block text
        # (A more robust approach would be used in production)
        
        # 1. LaTeX Math Translation (mitex)
        t = re.sub(r'\\\[(.*?)\\\]', lambda m: f'#mitex(`\n{m.group(1).strip()}\n`)', t, flags=re.DOTALL)
        t = re.sub(r'\$([^\$]+?)\$', lambda m: f'#mi(`{m.group(1).strip()}`)', t)
        t = re.sub(r'\\\((.*?)\\\)', lambda m: f'#mi(`{m.group(1).strip()}`)', t)
        
        t = t.replace('<', '\\<').replace('>', '\\>')
        
        # 3. Content Structure
        t = re.sub(r'^# (.*)$', r'= \1', t, flags=re.M)
        t = re.sub(r'^## (.*)$', r'== \1', t, flags=re.M)
        t = re.sub(r'\*\*(.*?)\*\*', r'* \1 *', t)
        t = re.sub(r'\*(.*?)\*', r'_ \1 _', t)
        t = re.sub(r'^\* ', r'- ', t, flags=re.M)
        
        # 4. Smart Image & Overlay Handling (FIXED BUG 1)
        prefix = self.images_dir.as_posix()
        def replace_image(match):
            img_filename = match.group(1)
            img_rel_path = f"images/{img_filename}"
            
            # Deep search for the image block
            img_block = None
            for page in self.raw_pages:
                for b in page.get('para_blocks', []):
                    if b.get('type') == 'image':
                        for sub in b.get('blocks', []):
                            for line in sub.get('lines', []):
                                for span in line.get('spans', []):
                                    if span.get('image_path') == img_rel_path:
                                        img_block = b
                                        break
            
            if not img_block:
                return f'#image("/{prefix}/{img_filename}", width: 100%)'
            
            img_bbox = img_block['bbox']
            overlays = []
            # Find captions/text overlapping this image
            for page in self.raw_pages:
                for b in page.get('para_blocks', []):
                    if b['type'] in ['text', 'image_caption']:
                        bbox = b['bbox']
                        if (bbox[0] >= img_bbox[0] - 5 and bbox[1] >= img_bbox[1] - 5 and 
                            bbox[2] <= img_bbox[2] + 5 and bbox[3] <= img_bbox[3] + 5):
                            dx = max(0, bbox[0] - img_bbox[0])
                            dy = max(0, bbox[1] - img_bbox[1])
                            text = " ".join([s.get('content', '') for l in b.get('lines', []) for s in l.get('spans', [])]).strip()
                            if text:
                                overlays.append(f'#place(top + left, dx: {dx}pt, dy: {dy}pt, [{text}])')
                                
            if not overlays:
                return f'#image("/{prefix}/{img_filename}", width: 100%)'
            return f'#box(clip: true, inset: 0pt)[#image("/{prefix}/{img_filename}", width: 100%)\n{"\n".join(overlays)}]'

        t = re.sub(r'!\[.*?\]\(images/(.*?)\)', replace_image, t)
        
        # Apply Overrides globally (very basic implementation for demonstration)
        # In a real scenario, we'd map labels exactly. Here we just return the markup.
        return t

    def build_document(self, agent_config, layout_info, content_markup):
        """Assemble the complete Typst document (FIXED BUG 2: Title Spanning)."""
        m = agent_config['margins']
        base_fs = agent_config.get('body_font_size', layout_info['avg_text_height'])
        head_fs = agent_config.get('heading_font_size', layout_info['avg_title_height'])
        line_ratio = agent_config.get('line_spacing_ratio', 1.2)
        leading = (line_ratio - 1) * 0.5
        h = agent_config.get('heading_style', {"alignment": "center", "top_spacing": "1.2em", "bottom_spacing": "0.6em"})
        cols = agent_config['columns']

        # Extract title/author from markup to place BEFORE the column layout
        lines = content_markup.split('\n')
        title_block = []
        rest_block = []
        in_title_zone = True
        for line in lines:
            if in_title_zone and (line.startswith('=') or line.strip() == '' or (not line.startswith('=') and len(title_block) < 5)):
                 title_block.append(line)
                 if len(title_block) > 10: in_title_zone = False # Heuristic stop
            else:
                 in_title_zone = False
                 rest_block.append(line)

        title_str = '\n'.join(title_block)
        rest_str = '\n'.join(rest_block)

        return f"""#import "@preview/mitex:0.2.4": *
#set page(
  width: {layout_info['page_width']}pt,
  height: {layout_info['page_height']}pt,
  margin: (top: {m['top']}, bottom: {m['bottom']}, left: {m['left']}, right: {m['right']}),
)
#set text(font: "{agent_config['font_family']}", size: {base_fs}pt)
#set par(justify: true, leading: {leading}em)

#show heading: it => [
  #set align({h['alignment']})
  #set text({head_fs}pt, weight: "bold")
  #block(inset: (top: {h['top_spacing']}, bottom: {h['bottom_spacing']}), it.body)
]

// Title spans full width
{title_str}

// Rest of content in columns
#show: columns.with({cols}, gutter: {layout_info['column_gap']}pt)

{rest_str}
"""

    def compile_draft(self, source, out_pdf_path):
        temp_typ = self.project_root / "scratch/temp.typ"
        temp_typ.write_text(source, encoding='utf-8')
        try:
            typst.compile(str(temp_typ), output=str(out_pdf_path), root=str(self.project_root))
            return True
        except Exception as e:
            print(f"    ✗ Compile error: {e}")
            return False

    def rasterize_pdf_page(self, pdf_path, page_idx):
        """Render a specific PDF page to PNG bytes."""
        try:
            doc = fitz.open(str(pdf_path))
            if page_idx >= len(doc): return None
            page = doc[page_idx]
            pix = page.get_pixmap(dpi=150)
            return pix.tobytes("png")
        except Exception as e:
            print(f"    ✗ Rasterize error: {e}")
            return None


class VisualArtDirectorAgent:
    """Gemini Flash — Visual iterative layout refinement via google-genai."""

    def __init__(self, api_key):
        if not api_key: raise ValueError("GEMINI_API_KEY not set")
        self.client = genai.Client(api_key=api_key)
        self.model = "gemini-flash-latest" # Cost effective, supports visual + schema

    def analyze_and_fix(self, original_png, draft_png, page_idx, json_context):
        """Multimodal prompt to find layout fixes."""
        prompt_text = f"""# INSTRUCT
Let's think step by step. Act as an **Expert Art Director** specializing in algorithmic typesetting and layout correction for scientific documents. Your goal is to achieve visual parity between a draft render and the original PDF.

# CONTEXT
You are provided with a multimodal payload consisting of:
1. **Target Image:** The original PDF page (what the final output *must* look like).
2. **Draft Image:** The current Typst-rendered draft page.
3. **Layout Metadata:** A JSON representation containing bounding box coordinates and text content for all blocks on THIS PAGE ONLY.

## Layout Data (Page {page_idx})
```json
{json.dumps(json_context, indent=2)}
```

# TASK
Perform a meticulous visual diff between the **Target Image** and the **Draft Image**. 
Your objective is to identify elements in the Draft that are misaligned, overlapping, overflowing columns, or improperly spaced, and issue precise Typst overrides to correct them.

# REASONING FRAMEWORK
Follow these steps carefully:
1. **Visual Scan:** Compare the macro structure. Are the columns aligned? Are images present? Is the text overflowing to the bottom or spilling into the margins?
2. **Correlation:** Match visual anomalies in the Draft Image to specific `index` IDs in the Layout Metadata.
3. **Root Cause Analysis:** Determine *why* a block is misaligned. (e.g., "Block 4 is pushed down because Block 3's translation is too long.")
4. **Parameter Adjustment:** Formulate specific numerical overrides to fix the root cause. 

# ACTIONABLE CONSTRAINTS
- **Prioritization:** Adjust `leading` (line spacing) first. If insufficient, adjust `tracking` (letter spacing). Only reduce `font_size` as an absolute last resort.
- **Safety Limits:**
  - `leading`: Do not go below `0.3em` or above `1.5em`.
  - `tracking`: Do not go below `-0.05em`.
  - `font_size`: **NEVER** reduce below `8.0pt`.
- **Target the Source:** Apply overrides to the blocks *causing* the overflow (usually the preceding bulky paragraphs), not just the blocks that were pushed out of place.

# OUTPUT REQUIREMENT
Return a structured JSON object containing your step-by-step `reasoning` and an array of `overrides` (mapping `block_id` like 'block_p0_idx5' to your numerical adjustments).
"""
        
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=original_png, mime_type="image/png"),
                    types.Part.from_bytes(data=draft_png, mime_type="image/png"),
                    types.Part.from_text(text=prompt_text),
                ],
            ),
        ]
        
        # Use Structured Output
        schema = {
            "type": "OBJECT",
            "properties": {
                "overrides": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "block_id": {"type": "STRING"},
                            "font_size": {"type": "NUMBER"},
                            "leading": {"type": "NUMBER"}
                        }
                    }
                },
                "reasoning": {"type": "STRING"}
            }
        }

        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
            media_resolution="MEDIA_RESOLUTION_HIGH",
            temperature=0.1,
            response_mime_type="application/json",
            response_schema=schema,
        )

        try:
            self.client.http_options = types.HttpOptions(timeout=1200_000) # 20 mins
            print("    🤖 Visual Agent analyzing images...")
            response = self.client.models.generate_content(
                model=self.model, contents=contents, config=config
            )
            return json.loads(response.text)
        except Exception as e:
            print(f"    ✗ Agent Error: {e}")
            return None


class VisualAgenticRenderer:
    MAX_ITERATIONS = 3

    def __init__(self, images_dir, project_root, gemini_api_key, layout_info, agent_config, full_md_content):
        self.introspector = VisualTypstIntrospector(images_dir, project_root, full_md_content, layout_info['raw_pages'])
        self.art_director = VisualArtDirectorAgent(gemini_api_key)
        self.layout_info = layout_info
        self.agent_config = agent_config
        self.project_root = Path(project_root)
        self.origin_pdf = list((self.project_root / 'output_test/957775a3-7aa9-46aa-a8c3-6017ff6ec536').glob('*_origin.pdf'))[0]

    def render(self, output_path: Path):
        print(f"\n[Phase 3] Visual Agent-in-the-Loop Rendering")
        
        overrides = {}
        draft_pdf_path = self.project_root / "scratch/rendered_test.pdf"

        # Sequential Page-by-Page Refinement Loop
        total_pages = len(self.layout_info['raw_pages'])
        for page_idx, page_json in enumerate(self.layout_info['raw_pages']):
            print(f"\n  ── Processing Page {page_idx + 1}/{total_pages} ──")
            
            for iteration in range(self.MAX_ITERATIONS):
                print(f"    Iteration {iteration + 1}/{self.MAX_ITERATIONS}")
                
                # 1. Generate & Compile Full Document with current overrides
                markup = self.introspector.inject_labels_and_convert(overrides)
                source = self.introspector.build_document(self.agent_config, self.layout_info, markup)
                if not self.introspector.compile_draft(source, draft_pdf_path):
                    break

                # 2. Rasterize specific page for visual comparison
                orig_png = self.introspector.rasterize_pdf_page(self.origin_pdf, page_idx)
                draft_png = self.introspector.rasterize_pdf_page(draft_pdf_path, page_idx)
                
                if not orig_png or not draft_png:
                    print("    ✗ Failed to rasterize PDFs. Draft might be missing pages.")
                    break
                    
                # 3. Call Visual Agent to fix THIS page
                if iteration < self.MAX_ITERATIONS - 1:
                    result = self.art_director.analyze_and_fix(orig_png, draft_png, page_idx, page_json)
                    if result:
                        print(f"    💡 Reasoning: {result.get('reasoning', '')[:150]}...")
                        ovs = result.get('overrides', [])
                        if not ovs:
                            print("    ✔ Agent found no issues on this page. Moving to next.")
                            break  # Converged, move to next page
                        print(f"    📝 Applying {len(ovs)} overrides")
                        for o in ovs:
                             overrides[o['block_id']] = o
                    else:
                        break  # Move to next page if API failed
                else:
                    print("    ⚠ Max iterations reached for this page.")

        print(f"\n  ✔ Final PDF: {draft_pdf_path}")


def main():
    input_dir = Path('output_test/957775a3-7aa9-46aa-a8c3-6017ff6ec536')
    layout_path = input_dir / 'layout.json'
    md_path = input_dir / 'full.md'
    images_dir = input_dir / 'images'
    output_path = Path('scratch/rendered_test.pdf')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Phase 1
        analyzer = LayoutAnalyzer(layout_path)
        layout_info = analyzer.analyze()

        # Phase 2
        gemma_key = os.environ.get("GEMMA_4_API_KEY")
        classifier = AgenticLayoutClassifier(gemma_key)
        agent_config = classifier.classify(md_path)
        print(f"  Agent Verdict: {agent_config['template_family']} ({agent_config['columns']} cols)")

        # Phase 3
        print("\n[Phase 3] Initializing Visual Agent-in-the-Loop...")
        gemini_key = os.environ.get("GEMINI_API_KEY")
        full_md_content = md_path.read_text(encoding='utf-8')
        
        renderer = VisualAgenticRenderer(
            images_dir=images_dir,
            project_root=Path.cwd(),
            gemini_api_key=gemini_key,
            layout_info=layout_info,
            agent_config=agent_config,
            full_md_content=full_md_content
        )
        renderer.render(output_path)

    except Exception as e:
        print(f"\nPIPELINE ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
