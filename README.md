# DIMT — Document Intelligent Machine Translation

> **DIMT** là hệ thống dịch thuật tài liệu học thuật và hình ảnh chuyên nghiệp (Anh → Việt, Pháp, Đức,...), bảo toàn hoàn hảo định dạng tài liệu gốc và tái cấu trúc tệp PDF đầu ra với công thức toán học sắc nét nhờ sự kết hợp giữa mô hình dịch máy NLLB, LoRA adapter và tác vụ kiểm định chất lượng bằng AI Agent.

---
## 💻 Yêu cầu Hệ thống (Local Setup)
| Thành phần | Yêu cầu tối thiểu | Khuyến nghị |
|---|---|---|
| **CPU** | 4 cores (Intel Core i5 / AMD Ryzen 5) | 8 cores trở lên |
| **RAM** | 8 GB | 16 GB trở lên |
| **GPU** | Không bắt buộc (Chạy bằng CPU) hoặc GPU NVIDIA 4 GB VRAM | NVIDIA 6 GB - 8 GB VRAM trở lên |
| **Disk** | 15 GB trống | 30 GB SSD |
| **OS** | Windows 10/11 hoặc Ubuntu 20.04/22.04 LTS | Windows 10/11 hoặc Ubuntu |
---

## ⚙️ Hướng dẫn Thiết lập Môi trường (Local)

### Bước 1: Cài đặt Node.js & uv
1. Tải và cài đặt Node.js từ trang chủ [Node.js](https://nodejs.org/).
2. Cài đặt `uv`:
   * **Windows (PowerShell):**
     ```powershell
     powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
     ```
   * **Linux/macOS:**
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     source $HOME/.cargo/env
     ```

### Bước 2: Đồng bộ thư viện Python & Node.js
Di chuyển vào thư mục dự án và thực hiện cài đặt:
```bash
git clone https://github.com/zinhcandoit/DocImg-Translate.git
cd DocImg-Translate

# Đồng bộ thư viện Python (tự động tạo môi trường ảo .venv và tải đúng phiên bản)
uv sync

# Cài đặt thư viện Node.js để chạy công cụ dịch toán học MathJax
npm install
```

### Bước 3: Cấu hình biến môi trường (`.env`)
Tạo tệp `.env` tại thư mục gốc của dự án:
```bash
cp .env.example .env
```
Điền đầy đủ các thông tin cấu hình vào file `.env`:
```ini
# --- MinerU API Key ---
MINERU_API_KEY=your_mineru_api_key_here

# --- LLM Agent Keys ---
GEMINI_API_KEY=AIzaSy_your_gemini_key_here
GPT_API_KEY="your_nvidia_api_key"

# --- MongoDB Configuration ---
# Để trống sẽ tự động fallback sang lưu trữ In-Memory tạm thời trên RAM
MONGODB_URL=mongodb://localhost:27017

# --- Latex Rendering Backend ---
# Hỗ trợ: 'mathtext', 'mathjax', hoặc 'tectonic'
LATEX_BACKEND=mathjax

# --- Evidently AI Cloud (Tùy chọn) ---
EVIDENT_AI_API_KEY=ev_your_key_here
EVIDENTLY_PROJECT_ID=your_project_uuid_here

# --- Hugging Face Cache ---
HF_HOME=./.cache/huggingface
```

---

## 🤖 Cơ chế tải mô hình Dịch máy (NLLB & MarianMT)

Hệ thống sử dụng các mô hình dịch máy từ HuggingFace Hub được cấu hình sẵn trong tệp `config/inference.yaml`.
- **NLLB-200**: Sử dụng mô hình nền `facebook/nllb-200-distilled-1.3B` kết hợp với LoRA Adapter `TQZinh/nllb-1.3B-ge-fr`.
- **MarianMT**: Sử dụng mô hình `TQZinh/MarianMT-en-fr` (dịch tiếng Pháp) và `TQZinh/MarianMT-en-de` (dịch tiếng Đức).
---

## 🚀 Hướng dẫn Vận hành bằng Terminal (Local)

Để chạy hệ thống trên máy cá nhân, hãy mở **3 cửa sổ Terminal** riêng biệt:

### Terminal 1: FastAPI Backend
Chịu trách nhiệm xử lý các tác vụ dịch, OCR, render PDF, kết nối cơ sở dữ liệu và quản lý VRAM.
```bash
uv run uvicorn src.backend.api:app --host 127.0.0.1 --port 8000
```

### Terminal 2: Streamlit Frontend
Cung cấp giao diện đồ họa tương tác cho người dùng.
```bash
uv run streamlit run src/frontend/app.py --server.port 8501
```
*   **Giao diện ứng dụng:** Truy cập tại [http://127.0.0.1:8501](http://127.0.0.1:8501).

### Terminal 3: MLflow Dashboard (Tùy chọn)
Dùng để giám sát hiệu suất dịch máy và đánh giá mô hình.
```bash
uv run mlflow ui --port 5000 --backend-store-uri sqlite:///mlruns.db
```
*   **MLflow UI:** Truy cập tại [http://127.0.0.1:5000](http://127.0.0.1:5000).

---

## 💡 Các Tính năng Nổi bật của DIMT

1.  **Dịch máy Paragraph-level bảo toàn công thức**: Hệ thống tự động gộp các dòng văn bản bị cắt nhỏ thành đoạn văn hoàn chỉnh trước khi dịch, đồng thời cô lập các khối toán học dạng `$ ... $` hoặc `$$ ... $$` dưới dạng token giữ chỗ để tránh làm sai lệch cấu trúc công thức khi dịch.
2.  **Tối ưu hóa VRAM thông minh theo dung lượng trống (Available VRAM)**: Trình dịch MarianMT sẽ liên tục giám sát lượng VRAM trống thực tế trên GPU để tự động điều chỉnh kích thước batch (Batch size từ 1 đến 32), giúp các máy tính cấu hình yếu (ví dụ: GPU 4GB VRAM) vẫn có thể dịch các văn bản lớn mà không bao giờ bị tràn bộ nhớ (Out-Of-Memory).
3.  **Lịch sử Dịch thuật đồng bộ kép (Saved History)**: Người dùng có thể quản lý lịch sử dịch lên đến 10 tài liệu gần nhất trên sidebar dựa theo nguyên tắc FIFO (tự động loại bỏ tài liệu cũ nhất khi vượt ngưỡng 10). Giao diện lịch sử hiển thị 2 tab kết quả đồng bộ 100% với phiên dịch chính gồm: **Tải xuống (Downloads)** và **Từ khóa (Keywords)** nhờ kiến trúc lưu trữ MongoDB tách biệt hai collections (`documents` và `agent_metadata`).
4.  **Tự động chuyển đổi mượt mà**: Nếu người dùng đang xem một tài liệu cũ trong Saved History nhưng bấm nút Convert tài liệu mới, hệ thống sẽ tự động kích hoạt chế độ quay lại danh sách lịch sử (Return to History List) trước khi thực thi tiến trình mới để tránh xung đột giao diện.
5.  **Q4 Verification & Agentic Helpers**:
    *   **Q4 Verification**: Hiển thị song song bản dịch gốc kèm theo nội dung văn bản hoàn chỉnh đã trích xuất giúp người dùng dễ dàng so sánh đối chiếu và tinh chỉnh.
    *   **Keyword Extraction**: Tự động trích xuất các từ khóa chuyên ngành chính từ tài liệu.
    *   **Table Recovery**: Khôi phục lại bố cục bảng biểu phức tạp.
6.  **Tự động dọn dẹp thư mục tạm (`temp/`)**: Dữ liệu tải lên và các tệp trung gian từ MinerU API được lưu trữ trong thư mục `temp/` thay vì `data/`, và sẽ tự động được xóa sạch ngay khi tệp PDF dịch được kết xuất thành công để tiết kiệm không gian đĩa cứng máy chủ.

---

## 📋 Tham chiếu nhanh

### Các tệp cấu hình quan trọng:
| Đường dẫn tệp/thư mục | Vai trò / Nhiệm vụ |
|---|---|
| `.env` | Chứa tất cả thông tin bảo mật, API key và địa chỉ kết nối Database |
| `config/inference.yaml` | Chứa thông tin cấu hình và đường dẫn tải các mô hình dịch máy từ HuggingFace Hub |
| `temp/` | Thư mục lưu trữ tạm thời các file tải lên và kết quả trung gian (sẽ tự động dọn dẹp) |
| `output/` | Nơi lưu trữ các file PDF và tài liệu sau khi dịch hoàn tất |

### Các cổng mạng mặc định:
| Dịch vụ | Cổng mặc định | URL cục bộ |
|---|---|---|
| **FastAPI backend** | `8000` | [http://localhost:8000/docs](http://localhost:8000/docs) |
| **Streamlit frontend** | `8501` | [http://localhost:8501](http://localhost:8501) |
| **MLflow UI** | `5000` | [http://localhost:5000](http://localhost:5000) |
| **MongoDB** | `27017` | `mongodb://localhost:27017` |

---

## 🌐 Triển khai Production & Troubleshooting

Đối với các yêu cầu triển khai hệ thống trên server (Ubuntu/Linux Server), cấu hình các dịch vụ chạy ngầm với **Systemd**, thiết lập Nginx Reverse Proxy (HTTPS / SSL), hoặc xử lý các sự cố thường gặp (CUDA Out of memory, MinerU Rate limits,...), vui lòng xem:

👉 **DEPLOYMENT.md**
