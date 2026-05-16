"""
DIMT Pipeline -- Word-Level Overlay PDF Re-Renderer (V7)

Approach:
  1. Per-page extraction from layout.json (no cross-page state)
  2. Tokenize blocks into (text, word) + (eq, latex) tokens
  3. Binary-search font size to fit text perfectly in bbox
  4. Render word-by-word: insert_text for text, insert_image for equations
  5. Adaptive bbox merging using spatial heuristics (same-page only)
  6. No HTML. No insert_htmlbox. Direct insert_text + insert_image.

Dependencies: pymupdf (fitz), matplotlib, numpy
"""

import json
import io
import re
from pathlib import Path
import fitz
# import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# -------------------------------------------------------------------
# Equation Renderer: LaTeX -> PNG bytes (in-memory)
# -------------------------------------------------------------------
plt.rcParams.update({
    "text.usetex": False,            # TUYỆT ĐỐI KHÔNG gọi system LaTeX
    "mathtext.fontset": "stix",      # Bật font STIX (Font khoa học siêu đẹp) CÓ SẴN trong Matplotlib
    "font.family": "STIXGeneral",    # Font chữ thường cũng là STIX
    "mathtext.fallback": "cm"        # Nếu thiếu symbol, tự fallback về Computer Modern
})

TEXT_TYPE = "notos"
TEXT_TYPE_BOLD = "notosbo"

class EquationRenderer:
    """Renders LaTeX internally using Matplotlib's STIX fonts, zero OS dependencies."""
    _cache: dict[str, dict] = {}

    def _clean_mineru_latex(self, tex: str) -> str:
        tex = tex.strip()
        tex = tex.replace('$', '')

        # 2. Xử lý các ký tự đặc biệt của môi trường Array/Align mà Mathtext không hiểu
        # Vì ta đã lột bỏ \begin{array}, ta phải biến & thành khoảng trắng 
        # và \\ thành dấu ngắt quãng để tránh lỗi ParseException
        tex = tex.replace('&', r'\quad ')
        tex = tex.replace(r'\\', r'\quad ')
        
        # 1. TRIỆT TIÊU MÔI TRƯỜNG \begin{...} \end{...}
        tex = re.sub(r'\\begin\s*\{[a-zA-Z*]+\}\s*(\{[^\}]*\})?', '', tex)
        tex = re.sub(r'\\end\s*\{[a-zA-Z*]+\}', '', tex)

        # 2. BỘ TỪ ĐIỂN CHUẨN HÓA LỆNH (Mathtext Fallback Dictionary)
        fixes = [
            # --- CÁC DÒNG CŨ ĐÃ CÓ ---
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
            (r'\\tag\s*\{[^\}]*\}', r''),        # Xóa lệnh \tag{...}
            (r'\\tag\b', r''),                   # Xóa \tag lọt sổ
            (r'\\Biggl\b|\\Biggr\b', r'\\Bigg'), # Chặt đuôi l/r của ngoặc cực lớn
            (r'\\biggl\b|\\biggr\b', r'\\bigg'), # Chặt đuôi l/r của ngoặc lớn
            (r'\\Bigl\b|\\Bigr\b', r'\\Big'),    # Chặt đuôi l/r của ngoặc vừa
            (r'\\bigl\b|\\bigr\b', r'\\big'),    # Chặt đuôi l/r của ngoặc nhỏ
        ]
        for pattern, repl in fixes:
            tex = re.sub(pattern, repl, tex)

        # 3. CHUẨN HÓA CÚ PHÁP FONT (Trị bệnh phá ngoặc của MinerU)
        # Sửa: { \mathcal F } hoặc \mathcal F  --->  \mathcal{F}
        for cmd in ['mathcal', 'mathbb', 'mathbf', 'mathrm', 'mathscr', 'mathfrak']:
            # Dạng 1: { \mathcal F }
            tex = re.sub(r'\{\s*\\' + cmd + r'\s+([A-Za-z0-9])\s*\}', r'\\' + cmd + r'{\1}', tex)
            # Dạng 2: \mathcal F
            tex = re.sub(r'\\' + cmd + r'\s+([A-Za-z0-9])', r'\\' + cmd + r'{\1}', tex)
            # Dạng 3: \mathcal { F }
            tex = re.sub(r'\\' + cmd + r'\s*\{\s*([A-Za-z0-9])\s*\}', r'\\' + cmd + r'{\1}', tex)

        tex = tex.strip()
        
        # 4. LỘT VỎ NGOẶC NHỌN THỪA NGOÀI CÙNG
        # Sửa: { P = 1 } ---> P = 1
        if tex.startswith('{') and tex.endswith('}'):
            open_braces = 0
            is_valid_wrap = True
            for i, char in enumerate(tex):
                if char == '{': open_braces += 1
                elif char == '}': open_braces -= 1
                # Nếu ngoặc đóng về 0 trước khi hết chuỗi -> Không phải lớp bọc ngoài cùng
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
        
        # Tiền xử lý LaTeX trước khi vẽ
        clean_tex = self._clean_mineru_latex(raw_tex)
        
        try:
            fig, ax = plt.subplots(figsize=(0.01, 0.01))
            ax.axis('off')
            
            # Vẽ bằng engine Mathtext nội bộ (Đã ép dùng font STIX)
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
# Block Tokenizer: layout block -> flat token list
# -------------------------------------------------------------------
def tokenize_block(block: dict) -> list[tuple[str, str]]:
    """Convert a block to flat list of (type, content) tokens.

    Types: "word" (single word), "eq" (LaTeX string)
    Words are split so we can measure and wrap individually.
    """
    tokens = []
    for line in block.get('lines', []):
        for span in line.get('spans', []):
            stype = span.get('type', '')
            content = span.get('content', '').strip()
            if not content:
                continue
            if stype == 'text':
                # Split into words, preserving spaces for wrapping
                words = content.split()
                for i, w in enumerate(words):
                    tokens.append(("word", w))
            elif stype == 'inline_equation':
                tokens.append(("eq", content))
    return tokens


# -------------------------------------------------------------------
# Unified bbox
# -------------------------------------------------------------------
def union_bbox(blocks: list[dict]) -> list | None:
    bboxes = [b['bbox'] for b in blocks if 'bbox' in b]
    if not bboxes:
        return None
    return [min(b[0] for b in bboxes), min(b[1] for b in bboxes),
            max(b[2] for b in bboxes), max(b[3] for b in bboxes)]


# -------------------------------------------------------------------
# Font Fitting: binary search for optimal font size
# -------------------------------------------------------------------
FONT = fitz.Font(TEXT_TYPE)
FONT_BOLD = fitz.Font(TEXT_TYPE_BOLD)
SPACE_RATIO = 0.3  # space width as fraction of fontsize


def simulate_layout(tokens, rect, fontsize, eq_renderer):
    """Simulate word-wrap layout. Returns True if all tokens fit in rect."""
    line_h = fontsize * 1.3
    space_w = fontsize * SPACE_RATIO
    x = rect.x0
    y = rect.y0 + fontsize  # first baseline

    for i, (kind, content) in enumerate(tokens):
        if kind == "word":
            w = FONT.text_length(content, fontsize=fontsize)
            if x > rect.x0 and x + w > rect.x1:
                x = rect.x0
                y += line_h
            x += w + space_w
        elif kind == "eq":
            # Gọi hàm mới lấy trực tiếp metrics thay vì phải render rồi dùng measure_png
            metrics = eq_renderer.render_and_metrics(content)
            if metrics:
                # Tính toán kích thước ảnh O(1) dựa vào fontsize và aspect_ratio
                disp_h = fontsize * 1.2
                disp_w = disp_h * metrics['aspect_ratio']
                
                if x > rect.x0 and x + disp_w > rect.x1:
                    x = rect.x0
                    y += line_h
                x += disp_w + space_w
            else:
                # Fallback: treat as text
                w = FONT.text_length(content, fontsize=fontsize)
                if x > rect.x0 and x + w > rect.x1:
                    x = rect.x0
                    y += line_h
                x += w + space_w

    return y <= rect.y1 + 1  # 1pt tolerance


def fit_fontsize(tokens, rect, eq_renderer, lo=1.0, hi=18.0) -> float:
    """Binary search for the largest fontsize that fits tokens in rect."""
    if not tokens:
        return 10.0
    for _ in range(20):
        mid = (lo + hi) / 2
        if simulate_layout(tokens, rect, mid, eq_renderer):
            lo = mid
        else:
            hi = mid
    return lo


# -------------------------------------------------------------------
# Spatial Heuristic: adaptive bbox merging
# -------------------------------------------------------------------
def merge_blocks(blocks: list[dict], eps_y: float = 12.0, eps_x: float = 200.0) -> list[dict]:
    """Sử dụng ML DBSCAN để gom cụm các khối văn bản theo không gian mật độ."""
    if not blocks: return []
    
    text_blocks = [b for b in blocks if b['type'] == 'text']
    other_blocks = [b for b in blocks if b['type'] != 'text']
    
    if len(text_blocks) < 2: return blocks
    
    # Ma trận đặc trưng: Trục Y (Top) và Trục X (Left)
    features = np.array([[b['bbox'][1], b['bbox'][0]] for b in text_blocks], dtype=float)
    
    # Scale không gian: Ép khoảng cách ngang (X) gần lại để cụm chủ yếu nhận diện theo dòng (Y)
    features[:, 0] /= eps_y  
    features[:, 1] /= eps_x  
    
    clustering = DBSCAN(eps=1.5, min_samples=1).fit(features)
    labels = clustering.labels_
    
    merged = []
    for cluster_id in set(labels):
        group = [text_blocks[i] for i in range(len(text_blocks)) if labels[i] == cluster_id]
        group = sorted(group, key=lambda x: x['bbox'][1])  # Sort top-to-bottom
        
        merged_bbox = [
            min(b['bbox'][0] for b in group), min(b['bbox'][1] for b in group),
            max(b['bbox'][2] for b in group), max(b['bbox'][3] for b in group),
        ]
        merged_tokens = sum((b.get('tokens', []) for b in group), [])
        total_lines = sum(b.get('n_lines', 1) for b in group)
        
        merged.append({
            'bbox': merged_bbox,
            'type': 'text',
            'tokens': merged_tokens,
            'n_lines': total_lines
        })
        
    return merged + other_blocks


# -------------------------------------------------------------------
# Per-Page Block Extraction
# -------------------------------------------------------------------
# --- 2 BIẾN TOÀN CỤC ĐỂ TRỊ BỆNH VẮT TRANG ---
# TÚI CHỨA CÁC DÒNG VẮT TRANG
global_cross_page_lines = []

def extract_page_blocks(page_data: dict) -> list[dict]:
    global global_cross_page_lines
    result = []
    
    # 1. ĐỔ TÚI TỪ TRANG TRƯỚC SANG TRANG HIỆN TẠI (Nếu có)
    if global_cross_page_lines:
        tokens = []
        valid_bboxes = []
        for line in global_cross_page_lines:
            for span in line.get('spans', []):
                stype = span.get('type', '')
                content = span.get('content', '').strip()
                if not content: continue
                if stype == 'text':
                    for w in content.split(): tokens.append(("word", w))
                elif stype in ('inline_equation', 'interline_equation'):
                    tokens.append(("eq", content))
            if line.get('bbox'):
                valid_bboxes.append(line['bbox'])
                
        if tokens and valid_bboxes:
            new_bbox = [
                min(b[0] for b in valid_bboxes), min(b[1] for b in valid_bboxes),
                max(b[2] for b in valid_bboxes), max(b[3] for b in valid_bboxes)
            ]
            # Render phần vắt trang này như một block text bình thường ở đầu trang
            result.append({
                'bbox': new_bbox,
                'type': 'text',
                'tokens': tokens,
                'n_lines': len(valid_bboxes)
            })
        
        # Xóa sạch túi sau khi đã render xong ở trang mới
        global_cross_page_lines = []

    # 2. XỬ LÝ TOÀN BỘ TEXT CỦA TRANG (Cả nội dung chính lẫn Header/Footer/Page Number)
    all_blocks = page_data.get('para_blocks', []) + page_data.get('discarded_blocks', [])
    next_carry_over = [] # Túi tạm cho trang tiếp theo

    for block in all_blocks:
        btype = block.get('type', '')
        sub_blocks = block.get('blocks', [block]) if 'blocks' in block else [block]
        
        for sub in sub_blocks:
            valid_lines = []
            sub_angle = sub.get('angle', 0)
            for line in sub.get('lines', []):
                # KIỂM TRA VẮT TRANG
                is_cross = any(span.get('cross_page', False) for span in line.get('spans', []))
                
                if is_cross:
                    next_carry_over.append(line) # Bỏ vào túi tạm, KHÔNG render ở trang này
                else:
                    valid_lines.append(line)     # Giữ lại để render ở trang này
                    
            # 3. TRÍCH XUẤT TOÀN BỘ CHỮ KHÔNG CHỪA LẠI GÌ CHO CÁC DÒNG HỢP LỆ
            if valid_lines:
                tokens = []
                valid_bboxes = []
                for line in valid_lines:
                    for span in line.get('spans', []):
                        stype = span.get('type', '')
                        content = span.get('content', '').strip()
                        if not content: continue
                        if stype == 'text':
                            for w in content.split(): tokens.append(("word", w))
                        elif stype in ('inline_equation'):
                            tokens.append(("eq", content))
                    if line.get('bbox'):
                        valid_bboxes.append(line['bbox'])
                
                if tokens and valid_bboxes:
                    new_bbox = [
                        min(b[0] for b in valid_bboxes), min(b[1] for b in valid_bboxes),
                        max(b[2] for b in valid_bboxes), max(b[3] for b in valid_bboxes)
                    ]
                    result.append({
                        'bbox': new_bbox,
                        'type': btype,
                        'tokens': tokens,
                        'n_lines': len(valid_bboxes),
                        'angle': sub_angle
                    })

    # Đưa túi tạm vào túi toàn cục để vòng lặp trang sau lấy ra xài
    global_cross_page_lines.extend(next_carry_over)
                
    return result


# -------------------------------------------------------------------
# Word-Level Renderer
# -------------------------------------------------------------------
def render_block(page: fitz.Page, block: dict, eq_renderer: EquationRenderer):
    """Render a block word-by-word with Auto-Justify Spacing and bounds clamping."""
    # 0. Chuẩn hóa góc xoay về bội số của 90 gần nhất
    raw_angle = block.get('angle', 0)
    angle = int(round(raw_angle / 90) * 90) % 360
    
    bbox = block['bbox']
    rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])

    # 1. Thiết lập không gian logic (Logical Space)
    if angle in [90, 270]:
        logical_width = rect.height
        logical_height = rect.width
    else:
        logical_width = rect.width
        logical_height = rect.height
    
    logical_rect = fitz.Rect(0, 0, logical_width, logical_height)
    if logical_width < 2 or logical_height < 2: return

    tokens = block.get('tokens', [])
    if not tokens: return

    btype = block['type']
    is_bold = btype == 'title'
    fontname = TEXT_TYPE_BOLD if is_bold else TEXT_TYPE
    font_obj = FONT_BOLD if is_bold else FONT

    # 2. Tìm font size tối ưu trong không gian logic
    fs = fit_fontsize(tokens, logical_rect, eq_renderer)
    fs = min(fs, logical_height * 0.9)
    if btype == 'page_footnote': fs = min(fs, 8.0)
    if btype == 'image_caption': fs = min(fs, 9.0)

    line_h = fs * 1.3
    ly = fs  # Tọa độ y trong không gian logic (baseline)

    # Bảng ánh xạ từ JSON (CCW) sang PyMuPDF (CW)
    rot_map = {0: 0, 90: 270, 180: 180, 270: 90}
    pdf_rotate = rot_map.get(angle, 0)

    # Hàm chuyển đổi từ tọa độ Logic (lx, ly) sang tọa độ PDF vật lý
    def to_physical(lx, ly):
        if angle == 90:      # JSON 90: Từ Trên xuống Dưới
            return fitz.Point(rect.x0 + ly, rect.y0 + lx)
        elif angle == 180:   # JSON 180: Từ Phải sang Trái
            return fitz.Point(rect.x1 - lx, rect.y1 - ly)
        elif angle == 270:   # JSON 270: Từ Dưới lên Trên (chuẩn arXiv)
            return fitz.Point(rect.x1 - ly, rect.y1 - lx)
        return fitz.Point(rect.x0 + lx, rect.y0 + ly)

    # 3. Thuật toán Line Buffering
    i = 0
    while i < len(tokens):
        line_tokens = []
        while i < len(tokens):
            kind, content = tokens[i]
            if kind == "word":
                w = font_obj.text_length(content, fontsize=fs)
            else:
                metrics = eq_renderer.render_and_metrics(content)
                w = (fs * 1.2 * metrics['aspect_ratio']) if metrics else font_obj.text_length(content, fontsize=fs)
            
            if not line_tokens:
                line_tokens.append((kind, content, w))
                i += 1
                continue
            
            current_content_w = sum(t[2] for t in line_tokens)
            new_content_w = current_content_w + w
            current_slack = logical_width - current_content_w - (len(line_tokens) - 1) * (fs * 0.2)
            
            if new_content_w + len(line_tokens) * (fs * 0.1) > logical_width:
                huge_gap_if_break = current_slack > fs * 0.7
                required_total_scale = logical_width / (new_content_w + len(line_tokens) * (fs * 0.05))
                if huge_gap_if_break and required_total_scale > 0.6: pass
                else: break
                    
            line_tokens.append((kind, content, w))
            i += 1

        # 4. Tính toán Justify và Scaling
        is_last_line = (i == len(tokens)) or (ly + line_h > logical_height + fs * 0.5)
        total_content_w = sum(t[2] for t in line_tokens)
        num_spaces = len(line_tokens) - 1
        space_w_at_fs = fs * 0.15
        needed_w = total_content_w + (num_spaces * space_w_at_fs)
        
        line_fs, squeeze_factor = fs, 1.0
        if needed_w > logical_width:
            combined_scale = logical_width / needed_w
            if combined_scale >= 0.75:
                line_fs = fs * combined_scale
                dynamic_space_w = space_w_at_fs * combined_scale
            else:
                line_fs = fs * 0.75
                squeeze_factor = combined_scale / 0.75
                dynamic_space_w = space_w_at_fs * 0.75 * squeeze_factor
        elif num_spaces > 0 and not is_last_line:
            dynamic_space_w = min((logical_width - total_content_w) / num_spaces, fs * 0.6)
        else:
            dynamic_space_w = fs * 0.25

        # 5. Render từng từ lên PDF
        lx = 0
        for kind, content, w_orig in line_tokens:
            w_scaled = w_orig * (line_fs / fs)
            p = to_physical(lx, ly)
            
            if kind == "word":
                morph = (p, fitz.Matrix(squeeze_factor, 1.0)) if squeeze_factor < 1.0 else None
                page.insert_text(p, content, fontsize=line_fs, fontname=fontname, morph=morph, rotate=pdf_rotate)
            elif kind == "eq":
                metrics = eq_renderer.render_and_metrics(content)
                if metrics:
                    disp_h = line_fs * 1.2
                    descent_offset = disp_h * metrics['descent_ratio']
                    # Tính toán Rect vật lý cho ảnh công thức dựa trên các điểm góc logic
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
        if ly > logical_height + line_h: break


# -------------------------------------------------------------------
# Core Overlay Renderer
# -------------------------------------------------------------------
class OverlayRenderer:
    def __init__(self, layout_path: str, origin_pdf_path: str, output_path: str):
        self.layout = json.loads(Path(layout_path).read_text(encoding='utf-8'))
        self.pages = self.layout['pdf_info']
        self.origin_pdf = origin_pdf_path
        self.output_path = output_path
        self.eq_renderer = EquationRenderer()
        
    def process_pipeline(self):
        global global_cross_page_lines
        global_cross_page_lines = []
        # 1. Mở file gốc chỉ để đọc
        src_doc = fitz.open(self.origin_pdf)
        print(f"[V8-ISOLATED] Processing {len(src_doc)} pages independently...")
        
        final_doc = fitz.open() # Document trống để chứa kết quả merge
        markdown_output = []

        for page_data in self.pages:
            page_idx = page_data.get('page_idx')
            if page_idx is None or page_idx >= len(src_doc):
                continue
            
            # --- BƯỚC A: TẠO PDF ĐƠN LẺ ---
            # Tạo một document mới chỉ chứa duy nhất trang hiện tại
            temp_page_doc = fitz.open()
            temp_page_doc.insert_pdf(src_doc, from_page=page_idx, to_page=page_idx)
            page = temp_page_doc[0] # Trang duy nhất của doc tạm
            
            # --- BƯỚC B: XỬ LÝ NỘI DUNG (READING ORDER) ---
            blocks = extract_page_blocks(page_data)
            # Sắp xếp theo trục Y (dung sai 15px) rồi đến trục X để xử lý bài báo 2 cột
            blocks = sorted(blocks, key=lambda b: (int(b['bbox'][1] // 15), b['bbox'][0]))

            if blocks:
                # Ghi nhận Markdown cho trang này
                markdown_output.append(f"\n\n\n\n")
                
                # Redact
                for b in blocks:
                    page.add_redact_annot(fitz.Rect(b['bbox']), fill=(1, 1, 1))
                page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

                # Render & Markdown Gen
                for b in blocks:
                    render_block(page, b, self.eq_renderer)
                    
                    # Ghép text cho Markdown
                    para_text = ""
                    for kind, content in b['tokens']:
                        para_text += (content if kind == 'word' else f"${content}$") + " "
                    markdown_output.append(re.sub(r'\s*\$\s*', '$', para_text.strip()) + "\n\n")

            # --- BƯỚC C: MERGE VÀO FILE TỔNG ---
            final_doc.insert_pdf(temp_page_doc)
            temp_page_doc.close()
            print(f"  Finished Page {page_idx + 1}")

        # 2. Lưu kết quả cuối cùng
        final_doc.save(self.output_path, garbage=4, deflate=True)
        final_doc.close()
        src_doc.close()
        
        # Xuất Markdown
        md_path = self.output_path.replace('.pdf', '.md')
        Path(md_path).write_text("".join(markdown_output), encoding='utf-8')
        print(f"Success: {self.output_path} and {md_path}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    input_dir = Path('output_test/957775a3-7aa9-46aa-a8c3-6017ff6ec536')
    layout_path = input_dir / 'layout.json'
    origin_pdf = input_dir / '1fe4b827-34d4-4f76-a343-81e3da8727b2_origin.pdf'
    output_path = Path('scratch/rendered_overlay.pdf')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    renderer = OverlayRenderer(
        layout_path=str(layout_path),
        origin_pdf_path=str(origin_pdf),
        output_path=str(output_path),
    )
    renderer.process_pipeline()


if __name__ == '__main__':
    main()
