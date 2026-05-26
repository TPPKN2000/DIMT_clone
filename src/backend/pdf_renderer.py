"""
PDF Renderer — Reconstructs a translated PDF using a word-level overlay pipeline.

Approach (V8-Isolated):
  1. Per-page extraction from layout.json (no cross-page state)
  2. Tokenize blocks into (text, word) + (eq, latex) tokens
  3. Binary-search font size to fit text perfectly in bbox
  4. Render word-by-word: insert_text for text, insert_image for equations
  5. Multi-angle support (0, 90, 180, 270)
  6. Direct insert_text + insert_image (No HTMLBox)
"""

import io
import os
import re
import json
import subprocess
from pathlib import Path
import fitz
from typing import Optional

try:
    from .latex_renderer import TectonicEquationRenderer
except ImportError:
    from latex_renderer import TectonicEquationRenderer

TEXT_TYPE = "notos"
TEXT_TYPE_BOLD = "notosbo"

class EquationRenderer:
    """Renders LaTeX equations using MathJax Node subprocess."""
    def __init__(self):
        self.process = None
        self._cache = {}
        self.js_path = Path(__file__).parent / "math_render.js"
        self._start_process()

    def _start_process(self):
        try:
            self.process = subprocess.Popen(
                ["node", str(self.js_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1
            )
            ready_line = self.process.stdout.readline()
            ready_data = json.loads(ready_line)
            if ready_data.get("status") != "ready":
                raise RuntimeError(f"MathJax initialization failed: {ready_line}")
        except Exception as e:
            print(f"[MathJax] Failed to start Node subprocess: {e}")
            if self.process:
                self.process.kill()
                self.process = None

    def render_and_metrics(self, latex: str) -> dict | None:
        raw_tex = latex.strip()
        if not raw_tex:
            return None
        if raw_tex in self._cache:
            return self._cache[raw_tex]

        if not self.process:
            return None

        math_tex = raw_tex.strip("$")
        req = {"id": id(raw_tex), "tex": math_tex}
        try:
            self.process.stdin.write(json.dumps(req) + "\n")
            self.process.stdin.flush()
            
            res_line = self.process.stdout.readline()
            res = json.loads(res_line)
            if "error" in res:
                print(f"[MathJax Error] {res['error']} for tex: {math_tex}")
                return None
            
            svg = res.get("svg")
            if not svg:
                return None

            width_match = re.search(r'width="([\d\.]+)ex"', svg)
            height_match = re.search(r'height="([\d\.]+)ex"', svg)
            valign_match = re.search(r'vertical-align:\s*(-?[\d\.]+)ex', svg)
            
            width = float(width_match.group(1)) if width_match else 1.0
            height = float(height_match.group(1)) if height_match else 1.0
            valign = float(valign_match.group(1)) if valign_match else 0.0
            
            aspect_ratio = width / height if height > 0 else 1.0
            descent_ratio = 0.0
            if valign < 0:
                descent_ratio = -valign / height

            self._cache[raw_tex] = {
                'svg_bytes': svg.encode('utf-8'),
                'aspect_ratio': aspect_ratio,
                'descent_ratio': descent_ratio
            }
            return self._cache[raw_tex]
        except Exception as e:
            print(f"[MathJax Exception] {e} during render for: {math_tex}")
            self._start_process()
            return None

    def close(self):
        if self.process:
            try:
                self.process.stdin.close()
                self.process.terminate()
                self.process.wait(timeout=2)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None

# -------------------------------------------------------------------
# Helper Constants & Font Objects
# -------------------------------------------------------------------
FONT = fitz.Font(TEXT_TYPE)
FONT_BOLD = fitz.Font(TEXT_TYPE_BOLD)
SPACE_RATIO = 0.3
BOLD_BLOCK_TYPES = {'title', 'section_title', 'heading', 'subheading'}
BOLD_MIN_SQUEEZE = 0.90
TITLE_LINE_HEIGHT_RATIO = 1.12
BODY_LINE_HEIGHT_RATIO = 1.3
global_cross_page_lines = []
HTML_EQ_HEIGHT_EM = 1.2

# -------------------------------------------------------------------
# PDFRenderer Class
# -------------------------------------------------------------------
class PDFRenderer:
    # Cache the built-in math font buffer for Unicode math character fallback
    _MATH_FONT_BUFFER = fitz.Font("math").buffer
    _NOTOS_FONT_BUFFER = fitz.Font("notos").buffer
    _NOTOSBO_FONT_BUFFER = fitz.Font("notosbo").buffer
    _MATH_CSS = (
        '@font-face { font-family: "NotoSans"; src: url("notos.ttf"); font-weight: normal; }\n'
        '@font-face { font-family: "NotoSans"; src: url("notosbo.ttf"); font-weight: bold; }\n'
        '@font-face { font-family: "mymathfont"; src: url("math.ttf"); }'
    )

    def __init__(self, images_dir: Optional[str] = None, latex_backend: Optional[str] = None):
        self.images_dir = Path(images_dir) if images_dir else None
        backend = (latex_backend or os.environ.get("LATEX_BACKEND", "mathtext")).lower()
        self.eq_renderer = self._build_equation_renderer(backend)

    def _build_equation_renderer(self, backend: str):
        if backend in ("mathtext", "mathjax"):
            return EquationRenderer()
        if backend != "tectonic":
            raise ValueError(f"Unsupported latex_backend: {backend}")

        return TectonicEquationRenderer(
            cache_dir=os.environ.get("LATEX_CACHE_DIR", ".cache/latex"),
            dpi=int(os.environ.get("LATEX_DPI", "300")),
            timeout_sec=int(os.environ.get("LATEX_TIMEOUT_SEC", "20")),
        )

    def _bbox_to_list(self, bbox):
        """Return a normalized [x0, y0, x1, y1] bbox, or None if invalid."""
        if not bbox or len(bbox) != 4:
            return None
        rect = fitz.Rect(bbox)
        rect.normalize()
        if not rect.is_valid or rect.is_empty or rect.width < 0.5 or rect.height < 0.5:
            return None
        return [rect.x0, rect.y0, rect.x1, rect.y1]

    def _union_bboxes(self, bboxes):
        """Union many valid bboxes into one bbox."""
        valid = [self._bbox_to_list(b) for b in bboxes]
        valid = [b for b in valid if b is not None]
        if not valid:
            return None
        return [
            min(b[0] for b in valid),
            min(b[1] for b in valid),
            max(b[2] for b in valid),
            max(b[3] for b in valid),
        ]

    def _choose_text_bbox(self, block, line_bboxes):
        """
        Use the paragraph/block bbox for rendering whenever layout.json provides it.

        layout.json can contain bboxes at multiple granularities:
        - block['bbox'] / paragraph bbox: the full available text area
        - line['bbox']: only one physical line

        Rendering a whole paragraph into a line bbox forces binary-search to shrink
        the font size until the entire paragraph fits into one-line height. That is
        the main reason text becomes extremely small.
        """
        block_bbox = self._bbox_to_list(block.get('bbox'))
        if block_bbox is not None:
            return block_bbox
        return self._union_bboxes(line_bboxes)

    def _token_width(self, kind, content, fontsize, font_obj):
        """Measure one token with the same font/rendering assumptions used later."""
        if kind == "word":
            return font_obj.text_length(content, fontsize=fontsize)

        latex = content["content"] if isinstance(content, dict) else content
        metrics = self.eq_renderer.render_and_metrics(latex)
        if not metrics:
            return font_obj.text_length(latex, fontsize=fontsize)

        eq_h = fontsize * HTML_EQ_HEIGHT_EM
        return eq_h * metrics['aspect_ratio']

    def _layout_lines(
        self,
        tokens,
        rect,
        fontsize,
        font_obj=FONT,
        max_lines=None,
        squeeze_min=1.0,
        prefer_squeeze=False,
        space_ratio=SPACE_RATIO,
    ):
        """
        Return greedy visual lines using the same width model used by rendering.

        Important: wrapping is decided against the real rect.width, not an
        artificially enlarged width. If a title would create more visual lines
        than the original layout, fit_fontsize() must reduce the font size.
        Squeeze is only a final tiny correction after the correct line count has
        already been achieved; it must not be used to decide where words wrap.
        """
        if not tokens:
            return []

        lines = []
        current = []
        current_w = 0.0
        space_w = fontsize * space_ratio

        for kind, content in tokens:
            if not content:
                continue
            w = self._token_width(kind, content, fontsize, font_obj)
            add_w = w if not current else space_w + w

            if current and current_w + add_w > rect.width:
                lines.append(current)
                if max_lines is not None and len(lines) >= max_lines:
                    return None
                current = [(kind, content, w)]
                current_w = w
            else:
                current.append((kind, content, w))
                current_w += add_w

        if current:
            lines.append(current)

        if max_lines is not None and len(lines) > max_lines:
            return None
        return lines

    def simulate_layout(
        self,
        tokens,
        rect,
        fontsize,
        font_obj=FONT,
        max_lines=None,
        squeeze_min=1.0,
        prefer_squeeze=False,
        line_height_ratio=BODY_LINE_HEIGHT_RATIO,
        space_ratio=SPACE_RATIO,
    ):
        """Simulate the same wrapping policy that render_block() will use."""
        if not tokens:
            return True

        lines = self._layout_lines(
            tokens,
            rect,
            fontsize,
            font_obj=font_obj,
            max_lines=max_lines,
            squeeze_min=squeeze_min,
            prefer_squeeze=prefer_squeeze,
            space_ratio=space_ratio,
        )
        if lines is None:
            return False

        # Width check after allowed squeeze. A line is valid if either it already
        # fits, or it can fit after horizontal morph >= squeeze_min.
        space_w = fontsize * space_ratio
        for line in lines:
            line_w = sum(t[2] for t in line) + max(0, len(line) - 1) * space_w
            if line_w > rect.width + 1:
                if squeeze_min >= 1.0:
                    return False
                if rect.width / line_w < squeeze_min:
                    return False

        needed_h = fontsize + (len(lines) - 1) * (fontsize * line_height_ratio)
        return needed_h <= rect.height + 1

    def fit_fontsize(
        self,
        tokens,
        rect,
        lo=1.0,
        hi=18.0,
        font_obj=FONT,
        max_lines=None,
        squeeze_min=1.0,
        prefer_squeeze=False,
        line_height_ratio=BODY_LINE_HEIGHT_RATIO,
        space_ratio=SPACE_RATIO,
    ) -> float:
        if not tokens:
            return 10.0
        for _ in range(20):
            mid = (lo + hi) / 2
            if self.simulate_layout(
                tokens,
                rect,
                mid,
                font_obj=font_obj,
                max_lines=max_lines,
                squeeze_min=squeeze_min,
                prefer_squeeze=prefer_squeeze,
                line_height_ratio=line_height_ratio,
                space_ratio=space_ratio,
            ):
                lo = mid
            else:
                hi = mid
        return lo

    def _is_quartet_text(self, obj) -> bool:
        """Strictly identify text via the quartet rule: bbox, type=text, content, score=float."""
        if not isinstance(obj, dict): return False
        return (
            "bbox" in obj and
            obj.get("type") == "text" and
            "content" in obj and
            isinstance(obj.get("score"), (int, float))
        )

    def _redact_quartet_recursive(self, page, obj):
        """Recursively find and redact every single quartet text component."""
        if self._is_quartet_text(obj):
            bbox = obj.get("bbox")
            if bbox:
                # Ensure the bbox is a valid fitz.Rect and normalized
                rect = fitz.Rect(bbox)
                rect.normalize()
                if rect.is_valid and not rect.is_empty:
                    page.add_redact_annot(rect, fill=(1, 1, 1))
        
        if isinstance(obj, dict):
            for v in obj.values():
                self._redact_quartet_recursive(page, v)
        elif isinstance(obj, list):
            for item in obj:
                self._redact_quartet_recursive(page, item)

    def _extract_recursive(self, blocks: list, result: list, next_carry_over: list):
        """Recursively extract translatable blocks from any level of the layout JSON."""
        for block in blocks:
            btype = block.get('type', 'text')
            
            # If the block has lines, it's a leaf block containing text
            if "lines" in block:
                valid_lines = []
                sub_angle = block.get('angle', 0)
                for line in block.get('lines', []):
                    # Q7: Cross-page stitching logic
                    if any(span.get('cross_page', False) for span in line.get('spans', [])):
                        next_carry_over.append(line)
                    else:
                        valid_lines.append(line)
                
                if valid_lines:
                    tokens = []
                    line_bboxes = []
                    for line in valid_lines:
                        for span in line.get('spans', []):
                            content = span.get('content', '').strip()
                            if not content: continue

                            if self._is_quartet_text(span):
                                for w in content.split(): tokens.append(("word", w))
                            elif span.get('type') in {'inline_equation', 'interline_equation'}:
                                tokens.append(("eq", {
                                    "content": content,
                                    "span_bbox": span.get("bbox"),
                                    "line_bbox": line.get("bbox")
                                }))
                        
                        if line.get('bbox'):
                            line_bboxes.append(line['bbox'])
                    
                    render_bbox = self._choose_text_bbox(block, line_bboxes)
                    if tokens and render_bbox:
                        result.append({
                            'bbox': render_bbox,
                            'line_bboxes': line_bboxes,
                            'type': btype,
                            'tokens': tokens,
                            'n_lines': len(line_bboxes),
                            'angle': sub_angle,
                            'merge_prev': block.get('merge_prev', False)
                        })
            
            # Recurse into nested blocks regardless of whether this block has lines
            if "blocks" in block:
                self._extract_recursive(block["blocks"], result, next_carry_over)

    def extract_page_blocks(self, page_data: dict) -> list[dict]:
        global global_cross_page_lines
        result = []
        
        # 1. Handle carry-over from previous page (cross_page stitching)
        if global_cross_page_lines:
            tokens = []
            valid_bboxes = []
            for line in global_cross_page_lines:
                for span in line.get('spans', []):
                    content = span.get('content', '').strip()
                    if not content: continue
                    if self._is_quartet_text(span):
                        for w in content.split(): tokens.append(("word", w))
                    elif span.get('type') in {'inline_equation', 'interline_equation'}:
                        tokens.append(("eq", {
                            "content": content,
                            "span_bbox": span.get("bbox"),
                            "line_bbox": line.get("bbox")
                        }))
                if line.get('bbox'): valid_bboxes.append(line['bbox'])
            
            render_bbox = self._union_bboxes(valid_bboxes)
            if tokens and render_bbox:
                result.append({
                    'bbox': render_bbox,
                    'line_bboxes': valid_bboxes,
                    'type': 'text',
                    'tokens': tokens,
                    'n_lines': len(valid_bboxes)
                })
            global_cross_page_lines = []

        # 2. Extract all blocks recursively
        all_root_blocks = (
            page_data.get('preproc_blocks', page_data.get('para_blocks', [])) + 
            page_data.get('discarded_blocks', [])
        )
        next_carry_over = []
        self._extract_recursive(all_root_blocks, result, next_carry_over)
        global_cross_page_lines.extend(next_carry_over)
        
        # 3. Deduplicate to prevent double rendering
        unique_result = []
        seen_bboxes = set()
        for r in result:
            key = tuple(round(v, 2) for v in r['bbox'])
            if key not in seen_bboxes:
                seen_bboxes.add(key)
                unique_result.append(r)
                
        return unique_result

    def _precompute_chain_fonts(self, blocks: list[dict]):
        # Group blocks into chains where blocks after the first have merge_prev == True
        chains = []
        current_chain = []
        for b in blocks:
            if b.get('merge_prev') and current_chain:
                current_chain.append(b)
            else:
                if current_chain:
                    chains.append(current_chain)
                current_chain = [b]
        if current_chain:
            chains.append(current_chain)

        # For each chain, compute individual font sizes and assign the minimum to all blocks in the chain
        for chain in chains:
            font_sizes = []
            for b in chain:
                raw_angle = b.get('angle', 0)
                angle = int(round(raw_angle / 90) * 90) % 360
                bbox = b['bbox']
                rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
                if angle in [90, 270]:
                    logical_width, logical_height = rect.height, rect.width
                else:
                    logical_width, logical_height = rect.width, rect.height
                logical_rect = fitz.Rect(0, 0, logical_width, logical_height)
                
                btype = b['type']
                is_bold = btype in BOLD_BLOCK_TYPES
                font_obj = FONT_BOLD if is_bold else FONT

                max_lines = b.get('n_lines') if is_bold else None
                if is_bold:
                    max_lines = max(1, int(max_lines or 1))

                line_height_ratio = TITLE_LINE_HEIGHT_RATIO if is_bold else BODY_LINE_HEIGHT_RATIO
                squeeze_min = BOLD_MIN_SQUEEZE if is_bold else 1.0

                fs = self.fit_fontsize(
                    b.get('tokens', []),
                    logical_rect,
                    font_obj=font_obj,
                    max_lines=max_lines,
                    squeeze_min=squeeze_min,
                    prefer_squeeze=False,
                    line_height_ratio=line_height_ratio,
                    space_ratio=SPACE_RATIO,
                )
                fs = min(fs, logical_height * 0.9)
                if btype == 'page_footnote':
                    fs = min(fs, 8.0)
                if btype == 'image_caption':
                    fs = min(fs, 9.0)

                if is_bold:
                    for _ in range(40):
                        test_lines = self._layout_lines(
                            b.get('tokens', []),
                            logical_rect,
                            fs,
                            font_obj=font_obj,
                            max_lines=max_lines,
                            squeeze_min=squeeze_min,
                            prefer_squeeze=False,
                            space_ratio=SPACE_RATIO,
                        )
                        if test_lines is not None:
                            break
                        fs *= 0.94
                font_sizes.append(fs)

            min_fs = min(font_sizes) if font_sizes else 10.0
            for b in chain:
                b['fs'] = min_fs

    def render_block(self, page, block, archive=None, block_idx=0):
        raw_angle = block.get('angle', 0)
        angle = int(round(raw_angle / 90) * 90) % 360
        bbox = block['bbox']
        rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
        if angle in [90, 270]:
            logical_width, logical_height = rect.height, rect.width
        else:
            logical_width, logical_height = rect.width, rect.height
        logical_rect = fitz.Rect(0, 0, logical_width, logical_height)
        if logical_width < 2 or logical_height < 2:
            return

        tokens = block.get('tokens', [])
        if not tokens:
            return

        btype = block['type']
        is_bold = btype in BOLD_BLOCK_TYPES
        font_obj = FONT_BOLD if is_bold else FONT

        max_lines = block.get('n_lines') if is_bold else None
        if is_bold:
            max_lines = max(1, int(max_lines or 1))

        line_height_ratio = TITLE_LINE_HEIGHT_RATIO if is_bold else BODY_LINE_HEIGHT_RATIO
        squeeze_min = BOLD_MIN_SQUEEZE if is_bold else 1.0

        fs = block.get('fs')
        if fs is None:
            fs = self.fit_fontsize(
                tokens,
                logical_rect,
                font_obj=font_obj,
                max_lines=max_lines,
                squeeze_min=squeeze_min,
                prefer_squeeze=False,
                line_height_ratio=line_height_ratio,
                space_ratio=SPACE_RATIO,
            )
            fs = min(fs, logical_height * 0.9)
            if btype == 'page_footnote':
                fs = min(fs, 8.0)
            if btype == 'image_caption':
                fs = min(fs, 9.0)

            # Safety pass for titles/headings to satisfy line limits
            if is_bold:
                for _ in range(40):
                    test_lines = self._layout_lines(
                        tokens,
                        logical_rect,
                        fs,
                        font_obj=font_obj,
                        max_lines=max_lines,
                        squeeze_min=squeeze_min,
                        prefer_squeeze=False,
                        space_ratio=SPACE_RATIO,
                    )
                    if test_lines is not None:
                        break
                    fs *= 0.94

        rot_map = {0: 0, 90: 270, 180: 180, 270: 90}
        pdf_rotate = rot_map.get(angle, 0)

        # Build HTML content
        html_parts = []
        for idx, (kind, content) in enumerate(tokens):
            if kind == "word":
                escaped = content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                html_parts.append(escaped)
            elif kind == "eq":
                latex = content["content"] if isinstance(content, dict) else content
                metrics = self.eq_renderer.render_and_metrics(latex)
                if metrics and 'svg_bytes' in metrics:
                    eq_id = f"eq_{block_idx}_{idx}.svg"
                    if archive is not None:
                        archive.add(metrics['svg_bytes'], eq_id)
                    descent_ratio = metrics.get('descent_ratio', 0.0)
                    valign = descent_ratio * HTML_EQ_HEIGHT_EM
                    html_parts.append(f'<img src="{eq_id}" style="height: {HTML_EQ_HEIGHT_EM}em; vertical-align: -{valign:.4f}em;"/>')
                else:
                    escaped = latex.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                    html_parts.append(escaped)

        html_text = " ".join(html_parts)
        align = "justify"
        if is_bold or btype in {'title', 'section_title', 'heading', 'subheading', 'image_caption', 'page_footnote'}:
            align = "left"

        font_weight = "bold" if is_bold else "normal"
        html_content = f"""
        <p style="font-family: 'NotoSans', 'mymathfont', sans-serif; font-size: {fs:.2f}pt; font-weight: {font_weight}; line-height: {line_height_ratio:.2f}; text-align: {align}; margin: 0; padding: 0;">
            {html_text}
        </p>
        """

        try:
            page.insert_htmlbox(
                rect,
                html_content,
                css=self._MATH_CSS,
                archive=archive,
                rotate=pdf_rotate,
                scale_low=0.1
            )
        except Exception as e:
            print(f"[PDFRenderer] HTMLBox insert error: {e}")

    def _collect_images_recursive(self, obj, collected: list):
        if isinstance(obj, dict):
            if "image_path" in obj and "bbox" in obj:
                collected.append((obj["image_path"], obj["bbox"]))
            for v in obj.values():
                self._collect_images_recursive(v, collected)
        elif isinstance(obj, list):
            for item in obj:
                self._collect_images_recursive(item, collected)

    def _rerender_images(self, page, page_data: dict):
        if not self.images_dir:
            return
        collected = []
        self._collect_images_recursive(page_data, collected)
        
        seen = set()
        for img_path, bbox in collected:
            if not bbox or len(bbox) != 4:
                continue
            key = (img_path, tuple(round(v, 2) for v in bbox))
            if key in seen:
                continue
            seen.add(key)
            
            full_path = self.images_dir / img_path
            if full_path.exists():
                rect = fitz.Rect(bbox)
                rect.normalize()
                if rect.is_valid and not rect.is_empty:
                    try:
                        page.insert_image(rect, filename=str(full_path))
                        print(f"[PDFRenderer] Re-rendered image: {img_path} at {bbox}")
                    except Exception as e:
                        print(f"[PDFRenderer Warning] Failed to insert image {img_path}: {e}")

    def render(self, layout_data: dict, origin_pdf_path: str, output_path: str) -> str:
        global global_cross_page_lines
        global_cross_page_lines = []
        src_doc = fitz.open(origin_pdf_path)
        final_doc = fitz.open()
        try:
            for page_data in layout_data.get('pdf_info', []):
                page_idx = page_data.get('page_idx')
                if page_idx is None or page_idx >= len(src_doc): continue
                
                temp_page_doc = fitz.open()
                temp_page_doc.insert_pdf(src_doc, from_page=page_idx, to_page=page_idx)
                page = temp_page_doc[0]
                
                # 1. Exhaustively redact every quartet text component found in page_data
                self._redact_quartet_recursive(page, page_data)
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)
                
                # 2. Extract and render translated blocks
                blocks = self.extract_page_blocks(page_data)
                self._precompute_chain_fonts(blocks)
                arch = fitz.Archive()
                arch.add(self._MATH_FONT_BUFFER, "math.ttf")
                arch.add(self._NOTOS_FONT_BUFFER, "notos.ttf")
                arch.add(self._NOTOSBO_FONT_BUFFER, "notosbo.ttf")
                for b_idx, b in enumerate(blocks):
                    self.render_block(page, b, archive=arch, block_idx=b_idx)
                
                # 3. Re-render any images/elements with "image_path"
                self._rerender_images(page, page_data)

                final_doc.insert_pdf(temp_page_doc)
                temp_page_doc.close()
            final_doc.save(output_path, garbage=4, deflate=True)
        finally:
            if hasattr(self, "eq_renderer") and hasattr(self.eq_renderer, "close"):
                self.eq_renderer.close()
            final_doc.close()
            src_doc.close()
        return output_path
