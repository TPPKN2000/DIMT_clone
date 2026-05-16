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
from sklearn.cluster import DBSCAN
import fitz
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# Equation Renderer: LaTeX -> PNG bytes (in-memory)
# -------------------------------------------------------------------
class EquationRenderer:
    """Renders LaTeX once, extracting exact topological metrics for O(1) scaling."""
    _cache: dict[str, dict] = {}

    def render_and_metrics(self, latex: str, dpi: int = 200) -> dict | None:
        tex = latex.strip()
        if not tex: return None
        if tex in self._cache: return self._cache[tex]
        
        try:
            # Render 1 lần duy nhất với kích thước lớn để lấy độ phân giải chuẩn
            fig, ax = plt.subplots(figsize=(0.01, 0.01))
            ax.axis('off')
            t = ax.text(0, 0, f"${tex}$", fontsize=50, va='baseline')
            fig.canvas.draw()
            
            renderer = fig.canvas.get_renderer()
            bbox = t.get_window_extent(renderer)
            
            # Tính toán trục toạ độ Data để lấy chính xác đường Baseline
            y_baseline = ax.transData.transform((0, 0))[1]
            descent_px = max(0, y_baseline - bbox.y0)
            
            fig.set_size_inches(bbox.width / dpi, bbox.height / dpi)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight', pad_inches=0, transparent=True)
            plt.close(fig)
            
            # Trích xuất Scale Invariant Metrics
            self._cache[tex] = {
                'png_bytes': buf.getvalue(),
                'aspect_ratio': bbox.width / bbox.height if bbox.height > 0 else 1,
                'descent_ratio': descent_px / bbox.height if bbox.height > 0 else 0
            }
            return self._cache[tex]
        except Exception:
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
FONT = fitz.Font("helv")
FONT_BOLD = fitz.Font("hebo")
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


def fit_fontsize(tokens, rect, eq_renderer, lo=3.0, hi=24.0) -> float:
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
def extract_page_blocks(page_data: dict) -> list[dict]:
    """Extract renderable blocks from a single page's layout data.

    Returns list of {bbox, type, tokens, n_lines, line_avg_height}
    """
    result = []
    discarded_ids = set()
    for db in page_data.get('discarded_blocks', []):
        idx = db.get('index')
        if idx is not None:
            discarded_ids.add(idx)

    for block in page_data.get('para_blocks', []):
        btype = block.get('type', '')
        bidx = block.get('index')
        if bidx in discarded_ids:
            continue

        if btype in ('text', 'title'):
            tokens = tokenize_block(block)
            if tokens:
                result.append({
                    'bbox': block['bbox'],
                    'type': btype,
                    'tokens': tokens,
                    'n_lines': len(block.get('lines', [])),
                    'line_avg_height': block.get('line_avg_height'),
                })

        elif btype == 'list':
            for sub in block.get('blocks', []):
                tokens = tokenize_block(sub)
                if tokens and sub.get('bbox'):
                    result.append({
                        'bbox': sub['bbox'],
                        'type': 'text',
                        'tokens': tokens,
                        'n_lines': len(sub.get('lines', [])),
                        'line_avg_height': sub.get('line_avg_height'),
                    })

        elif btype == 'image':
            caption_subs = [
                s for s in block.get('blocks', [])
                if s.get('type') in ('image_caption', 'image_footnote')
            ]
            if caption_subs:
                bbox = union_bbox(caption_subs)
                all_tokens = []
                total_lines = 0
                avg_h = None
                for sub in caption_subs:
                    all_tokens.extend(tokenize_block(sub))
                    total_lines += len(sub.get('lines', []))
                    if sub.get('line_avg_height'):
                        avg_h = sub['line_avg_height']
                if all_tokens and bbox:
                    result.append({
                        'bbox': bbox,
                        'type': 'image_caption',
                        'tokens': all_tokens,
                        'n_lines': max(1, total_lines),
                        'line_avg_height': avg_h,
                    })

        elif btype == 'table':
            # Bảo tồn Bbox tổng của Table để render dưới dạng 1 block duy nhất
            if block.get('bbox'):
                table_tokens = []
                for sub in block.get('blocks', []):
                    if sub.get('type') == 'table_body':
                        # Chèn thêm ký tự token Markdown Pipe '|' để đánh dấu cell
                        cell_tokens = tokenize_block(sub)
                        if cell_tokens:
                            table_tokens.append(("word", "|"))
                            table_tokens.extend(cell_tokens)
                
                if table_tokens:
                    table_tokens.append(("word", "|"))
                    result.append({
                        'bbox': block['bbox'],
                        'type': 'text', # Render như đoạn text thuần nhưng có ký tự định dạng Markdown
                        'tokens': table_tokens,
                        'n_lines': len(block.get('lines', [])),
                    })

    # discarded_blocks: page_footnote
    for block in page_data.get('discarded_blocks', []):
        if block.get('type') == 'page_footnote':
            tokens = tokenize_block(block)
            if tokens and block.get('bbox'):
                result.append({
                    'bbox': block['bbox'],
                    'type': 'page_footnote',
                    'tokens': tokens,
                    'n_lines': len(block.get('lines', [])),
                    'line_avg_height': block.get('line_avg_height'),
                })

    return result


# -------------------------------------------------------------------
# Word-Level Renderer
# -------------------------------------------------------------------
def render_block(page: fitz.Page, block: dict, eq_renderer: EquationRenderer):
    """Render a block word-by-word with Auto-Justify Spacing and bounds clamping."""
    bbox = block['bbox']
    rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
    if rect.width < 2 or rect.height < 2: return

    tokens = block.get('tokens', [])
    if not tokens: return

    btype = block['type']
    is_bold = btype == 'title'
    fontname = "hebo" if is_bold else "helv"
    font_obj = FONT_BOLD if is_bold else FONT

    # 1. Tìm font size và Khóa chết giới hạn
    fs = fit_fontsize(tokens, rect, eq_renderer)
    fs = min(fs, rect.height * 0.9) 
    
    if btype == 'page_footnote': fs = min(fs, 8.0)
    if btype == 'image_caption': fs = min(fs, 9.0)

    line_h = fs * 1.3
    y = rect.y0 + fs 

# 2. Thuật toán Line Buffering với Ưu tiên Font Scaling (Max -25%)
    i = 0
    while i < len(tokens):
        line_tokens = []
        
        # --- BẮT ĐẦU GOM DÒNG VÀ QUYẾT ĐỊNH KÉO CHỮ LÊN ---
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
            
            # Ước tính khoảng trống để lại nếu không kéo từ này lên (dùng space chuẩn 0.2fs)
            current_slack = rect.width - current_content_w - (len(line_tokens) - 1) * (fs * 0.2)
            
            # Kiểm tra xem có lố Bbox không
            if new_content_w + len(line_tokens) * (fs * 0.1) > rect.width:
                # Nếu lố, đo xem có đáng để "ép" để kéo chữ lên không?
                # Chấp nhận kéo lên nếu khoảng trống để lại quá lớn (> 0.7 lần cỡ chữ)
                huge_gap_if_break = current_slack > fs * 0.7
                
                # Tính toán xem để nhét vừa từ này thì cần tổng scale là bao nhiêu?
                required_total_scale = rect.width / (new_content_w + len(line_tokens) * (fs * 0.05))
                
                # Cho phép kéo lên nếu tổng scale không quá tàn khốc (tổng hợp cả shrink và squeeze > 60%)
                if huge_gap_if_break and required_total_scale > 0.6:
                    pass # Đồng ý kéo từ này lên dòng hiện tại
                else:
                    break # Từ chối, cho xuống dòng
                    
            line_tokens.append((kind, content, w))
            i += 1

        # 3. Tính toán Font Size và Squeeze cho dòng này
        is_last_line = (i == len(tokens)) or (y + line_h > rect.y1 + fs * 0.5)
        total_content_w = sum(t[2] for t in line_tokens)
        num_spaces = len(line_tokens) - 1
        space_w_at_fs = fs * 0.15 # Khoảng trắng mặc định hẹp
        
        # Tổng bề ngang cần thiết tại size gốc
        needed_w = total_content_w + (num_spaces * space_w_at_fs)
        
        line_fs = fs
        squeeze_factor = 1.0
        
        if needed_w > rect.width:
            # Tỷ lệ cần giảm để vừa khít
            combined_scale = rect.width / needed_w
            
            if combined_scale >= 0.75:
                # CHIẾN THUẬT 1: Chỉ giảm Font Size (tối đa 25%)
                line_fs = fs * combined_scale
                dynamic_space_w = space_w_at_fs * combined_scale
            else:
                # CHIẾN THUẬT 2: Giảm Font Size xuống 75% + Ép dẹp (Squeeze) phần còn lại
                line_fs = fs * 0.75
                squeeze_factor = combined_scale / 0.75
                dynamic_space_w = space_w_at_fs * 0.75 * squeeze_factor
        elif num_spaces > 0 and not is_last_line:
            # Dàn đều (Justify) nếu dòng còn trống
            dynamic_space_w = (rect.width - total_content_w) / num_spaces
            if dynamic_space_w > fs * 0.6: dynamic_space_w = fs * 0.6
        else:
            dynamic_space_w = fs * 0.25

        # 4. Bơm mực lên PDF (Sử dụng line_fs và squeeze_factor)
        x = rect.x0
        for kind, content, w_orig in line_tokens:
            # Chiều rộng thực tế sau khi giảm font size
            w_scaled = w_orig * (line_fs / fs)
            
            if kind == "word":
                # Morph matrix dùng để ép dẹp (squeeze) chiều ngang
                morph = (fitz.Point(x, y), fitz.Matrix(squeeze_factor, 1.0)) if squeeze_factor < 1.0 else None
                page.insert_text(fitz.Point(x, y), content, fontsize=line_fs, fontname=fontname, morph=morph)
            elif kind == "eq":
                metrics = eq_renderer.render_and_metrics(content)
                if metrics:
                    disp_h = line_fs * 1.2
                    descent_offset = disp_h * metrics['descent_ratio']
                    # Squeeze ảnh bằng cách nhân squeeze_factor vào chiều rộng Rect
                    eq_rect = fitz.Rect(x, y - disp_h + descent_offset, x + (w_scaled * squeeze_factor), y + descent_offset)
                    page.insert_image(eq_rect, stream=metrics['png_bytes'])
                else:
                    # FALLBACK: Hiển thị lại chuỗi text gốc
                    morph = (fitz.Point(x, y), fitz.Matrix(squeeze_factor, 1.0)) if squeeze_factor < 1.0 else None
                    page.insert_text(fitz.Point(x, y), content, fontsize=line_fs, fontname=fontname, morph=morph)
            x += (w_scaled * squeeze_factor) + dynamic_space_w
        
        y += line_h
        if y > rect.y1 + line_h: break

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
        doc = fitz.open(self.origin_pdf)
        print(f"[Overlay] PDF: {len(doc)} pages")
        
        markdown_output = [] # Lưu trữ nội dung Markdown

        # DUYỆT ĐỘC LẬP TỪNG TRANG: Tuyệt đối không lọt data trang này sang trang khác
        for page_data in self.pages:
            # SỬ DỤNG KEY 'page_idx' TỪ JSON ĐỂ LÀM VÁCH NGĂN
            page_idx = page_data.get('page_idx')
            if page_idx is None or page_idx >= len(doc):
                continue
                
            page = doc[page_idx]
            original_links = page.get_links()

            # 1. Trích xuất & Gom khối nội bộ (Isolated Box)
            blocks = extract_page_blocks(page_data)
            blocks = merge_blocks(blocks)
            
            # Sắp xếp các block chuẩn xác từ trên xuống dưới, trái sang phải để nối text
            blocks = sorted(blocks, key=lambda b: (b['bbox'][1], b['bbox'][0]))

            if not blocks:
                continue

            # ==============================================================
            # PHASE 1: GENERATE MARKDOWN (Giữ nguyên cấu trúc, không dính chữ)
            # ==============================================================
            markdown_output.append(f"\n\n\n\n\n")
            
            for b in blocks:
                tokens = b.get('tokens', [])
                if not tokens: continue
                
                # Tái tạo câu văn từ tokens: Text bình thường thì cách nhau khoảng trắng, Eq thì bọc $...$
                para_text = ""
                for kind, content in tokens:
                    if kind == 'word':
                        para_text += content + " "
                    elif kind == 'eq':
                        para_text += f"${content}$ "
                        
                # Dọn dẹp khoảng trắng thừa quanh phương trình (VD: " $ \sum $ " -> "$\sum$")
                para_text = re.sub(r'\s*\$\s*', '$', para_text.strip())
                markdown_output.append(para_text + "\n\n")

            # ==============================================================
            # PHASE 2: REDACT & RENDER ĐÈ LÊN PDF (Per-page)
            # ==============================================================
            for b in blocks:
                bb = b['bbox']
                page.add_redact_annot(fitz.Rect(bb[0], bb[1], bb[2], bb[3]), fill=(1, 1, 1))
            page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

            for b in blocks:
                render_block(page, b, self.eq_renderer)

            for link in original_links:
                try: page.insert_link(link)
                except Exception: pass

        # Lưu file PDF
        doc.save(self.output_path, garbage=4, deflate=True)
        doc.close()
        print(f"  Saved PDF: {self.output_path}")
        
        # Lưu file Markdown để kiểm tra (Hoàn thành mục tiêu convert)
        md_path = self.output_path.replace('.pdf', '.md')
        Path(md_path).write_text("\n".join(markdown_output), encoding='utf-8')
        print(f"  Saved Markdown: {md_path}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    input_dir = Path('data')
    layout_path = input_dir / '23-025_layout.json'
    origin_pdf = input_dir / '23-025_origin.pdf'
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
