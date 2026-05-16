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

import json
import io
import re
from pathlib import Path
import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional

# -------------------------------------------------------------------
# Equation Renderer: LaTeX -> PNG bytes (in-memory)
# -------------------------------------------------------------------
plt.rcParams.update({
    "text.usetex": False,
    "mathtext.fontset": "stix",
    "font.family": "STIXGeneral",
    "mathtext.fallback": "cm"
})

TEXT_TYPE = "notos"
TEXT_TYPE_BOLD = "notosbo"

class EquationRenderer:
    """Renders LaTeX internally using Matplotlib's STIX fonts."""
    _cache: dict[str, dict] = {}

    def _clean_mineru_latex(self, tex: str) -> str:
        tex = tex.strip()
        tex = tex.replace('$', '')
        tex = tex.replace('&', r'\quad ')
        tex = tex.replace(r'\\', r'\quad ')
        tex = re.sub(r'\\begin\s*\{[a-zA-Z*]+\}\s*(\{[^\}]*\})?', '', tex)
        tex = re.sub(r'\\end\s*\{[a-zA-Z*]+\}', '', tex)
        fixes = [
            (r'\\operatorname\*', r'\\operatorname'), 
            (r'\\dotsc|\\dotsb|\\dotsi|\\dotso', r'\\dots'), 
            (r'\\le\b', r'\\leq'),                     
            (r'\\ge\b', r'\\geq'),                     
            (r'\\cal\b', r'\\mathcal'),                
            (r'\\rm\b', r'\\mathrm'),                  
            (r'\\bf\b', r'\\mathbf'),                  
            (r'\\mathbbm\b', r'\\mathbb'),             
            (r'\\stackrel', r'\\overset'),             
            (r'\\textstyle', r''),                     
            (r'\\displaystyle', r''),                  
            (r'\\tag\s*\{[^\}]*\}', r''),
            (r'\\tag\b', r''),
            (r'\\Biggl\b|\\Biggr\b', r'\\Bigg'),
            (r'\\biggl\b|\\biggr\b', r'\\bigg'),
            (r'\\Bigl\b|\\Bigr\b', r'\\Big'),
            (r'\\bigl\b|\\bigr\b', r'\\big'),
        ]
        for pattern, repl in fixes:
            tex = re.sub(pattern, repl, tex)
        for cmd in ['mathcal', 'mathbb', 'mathbf', 'mathrm', 'mathscr', 'mathfrak']:
            tex = re.sub(r'\{\s*\\' + cmd + r'\s+([A-Za-z0-9])\s*\}', r'\\' + cmd + r'{\1}', tex)
            tex = re.sub(r'\\' + cmd + r'\s+([A-Za-z0-9])', r'\\' + cmd + r'{\1}', tex)
            tex = re.sub(r'\\' + cmd + r'\s*\{\s*([A-Za-z0-9])\s*\}', r'\\' + cmd + r'{\1}', tex)
        tex = tex.strip()
        if tex.startswith('{') and tex.endswith('}'):
            open_braces = 0
            is_valid_wrap = True
            for i, char in enumerate(tex):
                if char == '{': open_braces += 1
                elif char == '}': open_braces -= 1
                if open_braces == 0 and i < len(tex) - 1:
                    is_valid_wrap = False
                    break
            if is_valid_wrap:
                tex = tex[1:-1].strip()
        return tex

    def render_and_metrics(self, latex: str, dpi: int = 300) -> dict | None:
        raw_tex = latex.strip()
        if not raw_tex: return None
        if raw_tex in self._cache: return self._cache[raw_tex]
        clean_tex = self._clean_mineru_latex(raw_tex)
        try:
            fig, ax = plt.subplots(figsize=(0.01, 0.01))
            ax.axis('off')
            t = ax.text(0, 0, f"${clean_tex}$", fontsize=40, va='baseline')
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bbox = t.get_window_extent(renderer)
            y_baseline = ax.transData.transform((0, 0))[1]
            descent_px = max(0, y_baseline - bbox.y0)
            fig.set_size_inches(bbox.width / dpi, bbox.height / dpi)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close(fig)
            self._cache[raw_tex] = {
                'png_bytes': buf.getvalue(),
                'aspect_ratio': bbox.width / bbox.height if bbox.height > 0 else 1,
                'descent_ratio': descent_px / bbox.height if bbox.height > 0 else 0
            }
            return self._cache[raw_tex]
        except Exception as e:
            print(f"[MathText Error] Failed on: {clean_tex[:30]}... | Error: {e}")
            plt.close('all')
            return None

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

# -------------------------------------------------------------------
# PDFRenderer Class
# -------------------------------------------------------------------
class PDFRenderer:
    def __init__(self, images_dir: Optional[str] = None):
        self.images_dir = Path(images_dir) if images_dir else None
        self.eq_renderer = EquationRenderer()

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

        metrics = self.eq_renderer.render_and_metrics(content)
        if metrics:
            return fontsize * 1.2 * metrics['aspect_ratio']
        return font_obj.text_length(content, fontsize=fontsize)

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
                            elif span.get('type') == 'inline_equation':
                                tokens.append(("eq", content))
                        
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
                            'angle': sub_angle
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
                    elif span.get('type') == 'inline_equation':
                        tokens.append(("eq", content))
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

    def render_block(self, page, block):
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
        fontname = TEXT_TYPE_BOLD if is_bold else TEXT_TYPE
        font_obj = FONT_BOLD if is_bold else FONT

        # Titles/headings are NOT forced to single-line. They keep the original
        # layout's line count. If the original title is one line, we fit it as one
        # line by font shrink + mild horizontal squeeze. If the original title has
        # two lines, it is allowed to wrap into two lines.
        max_lines = block.get('n_lines') if is_bold else None
        if is_bold:
            max_lines = max(1, int(max_lines or 1))

        line_height_ratio = TITLE_LINE_HEIGHT_RATIO if is_bold else BODY_LINE_HEIGHT_RATIO
        squeeze_min = BOLD_MIN_SQUEEZE if is_bold else 1.0
        prefer_squeeze = False

        fs = self.fit_fontsize(
            tokens,
            logical_rect,
            font_obj=font_obj,
            max_lines=max_lines,
            squeeze_min=squeeze_min,
            prefer_squeeze=prefer_squeeze,
            line_height_ratio=line_height_ratio,
            space_ratio=SPACE_RATIO,
        )
        fs = min(fs, logical_height * 0.9)
        if btype == 'page_footnote':
            fs = min(fs, 8.0)
        if btype == 'image_caption':
            fs = min(fs, 9.0)

        # Safety pass: after any height/category cap, titles must still satisfy
        # the original line count. If they break, shrink the font; do not accept
        # the wrapped title.
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

        line_h = fs * line_height_ratio
        ly = fs
        rot_map = {0: 0, 90: 270, 180: 180, 270: 90}
        pdf_rotate = rot_map.get(angle, 0)

        def to_physical(lx, ly):
            if angle == 90:
                return fitz.Point(rect.x0 + ly, rect.y0 + lx)
            elif angle == 180:
                return fitz.Point(rect.x1 - lx, rect.y1 - ly)
            elif angle == 270:
                return fitz.Point(rect.x1 - ly, rect.y1 - lx)
            return fitz.Point(rect.x0 + lx, rect.y0 + ly)

        if is_bold:
            lines = self._layout_lines(
                tokens,
                logical_rect,
                fs,
                font_obj=font_obj,
                max_lines=max_lines,
                squeeze_min=squeeze_min,
                prefer_squeeze=False,
                space_ratio=SPACE_RATIO,
            )
            if not lines:
                return

            for line_tokens in lines:
                total_content_w = sum(t[2] for t in line_tokens)
                num_spaces = len(line_tokens) - 1
                space_w_at_fs = fs * SPACE_RATIO
                needed_w = total_content_w + max(0, num_spaces) * space_w_at_fs
                squeeze_factor = 1.0
                if needed_w > logical_width and needed_w > 0:
                    squeeze_factor = max(squeeze_min, logical_width / needed_w)

                lx = 0
                for kind, content, w_orig in line_tokens:
                    p = to_physical(lx, ly)
                    morph = (p, fitz.Matrix(squeeze_factor, 1.0)) if squeeze_factor < 1.0 else None
                    if kind == "word":
                        page.insert_text(
                            p,
                            content,
                            fontsize=fs,
                            fontname=fontname,
                            morph=morph,
                            rotate=pdf_rotate,
                        )
                    elif kind == "eq":
                        metrics = self.eq_renderer.render_and_metrics(content)
                        if metrics:
                            disp_h = fs * 1.2
                            descent_offset = disp_h * metrics['descent_ratio']
                            p_bl = to_physical(lx, ly + descent_offset)
                            p_tr = to_physical(lx + (w_orig * squeeze_factor), ly - disp_h + descent_offset)
                            eq_rect = fitz.Rect(p_bl, p_tr)
                            eq_rect.normalize()
                            if eq_rect.is_valid and not eq_rect.is_empty:
                                page.insert_image(eq_rect, stream=metrics['png_bytes'], rotate=pdf_rotate)
                        else:
                            page.insert_text(
                                p,
                                content,
                                fontsize=fs,
                                fontname=fontname,
                                morph=morph,
                                rotate=pdf_rotate,
                            )
                    lx += (w_orig * squeeze_factor) + (space_w_at_fs * squeeze_factor)

                ly += line_h
                if ly > logical_height + line_h:
                    break
            return

        i = 0
        while i < len(tokens):
            line_tokens = []
            while i < len(tokens):
                kind, content = tokens[i]
                if kind == "word":
                    w = font_obj.text_length(content, fontsize=fs)
                else:
                    metrics = self.eq_renderer.render_and_metrics(content)
                    w = (fs * 1.2 * metrics['aspect_ratio']) if metrics else font_obj.text_length(content, fontsize=fs)
                if not line_tokens:
                    line_tokens.append((kind, content, w))
                    i += 1
                    continue
                current_content_w = sum(t[2] for t in line_tokens)
                if current_content_w + w + len(line_tokens) * (fs * 0.1) > logical_width:
                    break
                line_tokens.append((kind, content, w))
                i += 1

            is_last_line = (i == len(tokens)) or (ly + line_h > logical_height + fs * 0.5)
            total_content_w = sum(t[2] for t in line_tokens)
            num_spaces = len(line_tokens) - 1
            space_w_at_fs = fs * 0.15
            needed_w = total_content_w + (num_spaces * space_w_at_fs)
            line_fs, squeeze_factor = fs, 1.0
            if needed_w > logical_width:
                combined_scale = logical_width / needed_w
                if combined_scale >= 0.50:
                    line_fs, dynamic_space_w = fs * combined_scale, space_w_at_fs * combined_scale
                else:
                    line_fs, squeeze_factor = fs * 0.50, combined_scale / 0.50
                    dynamic_space_w = space_w_at_fs * 0.50 * squeeze_factor
            elif num_spaces > 0 and not is_last_line:
                dynamic_space_w = min((logical_width - total_content_w) / num_spaces, fs * 0.6)
            else:
                dynamic_space_w = fs * 0.25

            lx = 0
            for kind, content, w_orig in line_tokens:
                w_scaled = w_orig * (line_fs / fs)
                p = to_physical(lx, ly)
                if kind == "word":
                    morph = (p, fitz.Matrix(squeeze_factor, 1.0)) if squeeze_factor < 1.0 else None
                    page.insert_text(p, content, fontsize=line_fs, fontname=fontname, morph=morph, rotate=pdf_rotate)
                elif kind == "eq":
                    metrics = self.eq_renderer.render_and_metrics(content)
                    if metrics:
                        disp_h = line_fs * 1.2
                        descent_offset = disp_h * metrics['descent_ratio']
                        p_bl = to_physical(lx, ly + descent_offset)
                        p_tr = to_physical(lx + (w_scaled * squeeze_factor), ly - disp_h + descent_offset)
                        eq_rect = fitz.Rect(p_bl, p_tr)
                        eq_rect.normalize()
                        if eq_rect.is_valid and not eq_rect.is_empty:
                            page.insert_image(eq_rect, stream=metrics['png_bytes'], rotate=pdf_rotate)
                    else:
                        morph = (p, fitz.Matrix(squeeze_factor, 1.0)) if squeeze_factor < 1.0 else None
                        page.insert_text(p, content, fontsize=line_fs, fontname=fontname, morph=morph, rotate=pdf_rotate)
                lx += (w_scaled * squeeze_factor) + dynamic_space_w
            ly += line_h
            if ly > logical_height + line_h:
                break

    def render(self, layout_data: dict, origin_pdf_path: str, output_path: str) -> str:
        global global_cross_page_lines
        global_cross_page_lines = []
        src_doc = fitz.open(origin_pdf_path)
        final_doc = fitz.open()
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
            for b in blocks:
                self.render_block(page, b)
            final_doc.insert_pdf(temp_page_doc)
            temp_page_doc.close()
        final_doc.save(output_path, garbage=4, deflate=True)
        final_doc.close()
        src_doc.close()
        return output_path
