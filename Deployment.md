# DIMT — Hướng dẫn Deploy & Vận hành

## Mục lục

1. [Yêu cầu hệ thống](#1-yêu-cầu-hệ-thống)
2. [Chuẩn bị API Keys](#2-chuẩn-bị-api-keys)
3. [Cấu hình `.env`](#3-cấu-hình-env)
4. [Cài đặt môi trường](#4-cài-đặt-môi-trường)
5. [Chuẩn bị model NLLB](#5-chuẩn-bị-model-nllb)
6. [Khởi động dịch vụ](#6-khởi-động-dịch-vụ)
7. [Kiểm tra pipeline end-to-end](#7-kiểm-tra-pipeline-end-to-end)
8. [Deploy production (Linux server)](#8-deploy-production-linux-server)
9. [Giám sát & troubleshoot](#9-giám-sát--troubleshoot)
10. [Tham chiếu nhanh](#10-tham-chiếu-nhanh)

---

## 1. Yêu cầu hệ thống

### Phần cứng tối thiểu

| Thành phần | Yêu cầu tối thiểu | Khuyến nghị |
|---|---|---|
| CPU | 8 cores | 16 cores |
| RAM | 16 GB | 32 GB |
| GPU | NVIDIA 8 GB VRAM (CUDA 11.8+) | RTX 3090 / A100 |
| Disk | 30 GB | 100 GB SSD |
| OS | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS |

> **Không có GPU:** pipeline chạy được trên CPU nhưng dịch ~10× chậm hơn.
> `nllb_service.py` tự detect `cuda` → fallback sang `cpu` nếu không có GPU.

### Phần mềm

```
Python 3.13.x  (bắt buộc, pyproject.toml yêu cầu >=3.13,<3.14)
uv             (package manager)
git
CUDA 11.8 drivers (nếu dùng GPU)
MongoDB 7.x    (tùy chọn — fallback in-memory nếu không có)
```

---

## 2. Chuẩn bị API Keys

Pipeline cần 3–4 API key từ các dịch vụ bên ngoài:

### 2.1 MinerU API Key — **Bắt buộc**

MinerU là dịch vụ OCR/extraction chính cho PDF.

1. Truy cập: https://mineru.net
2. Đăng ký tài khoản (có free tier)
3. Vào **Dashboard → API Keys → Create Key**
4. Copy key dạng: `eyJ...` hoặc `sk-...`

> **Rate limits free tier:** 50 files/phút, 1000 polls/phút, ≤200 trang/file, ≤200 MB.

### 2.2 Gemini API Key — **Nếu dùng agent LLM = Gemini**

1. Truy cập: https://aistudio.google.com/apikey
2. Click **Create API key**
3. Copy key dạng: `AIza...`

### 2.3 DeepSeek API Key (qua NVIDIA) — **Nếu dùng agent LLM = DeepSeek**

1. Truy cập: https://integrate.api.nvidia.com
2. Đăng ký / đăng nhập
3. Vào **API Keys → Generate**
4. Copy key dạng: `nvapi-...`

> **Lưu ý:** Chỉ cần một trong hai (Gemini hoặc DeepSeek). Mặc định trong code là DeepSeek.
> Nếu không có key nào, agent sẽ fallback sang kết quả rỗng — pipeline vẫn chạy nhưng không có Q4 verification.

### 2.4 MongoDB Connection String — **Tùy chọn nhưng khuyến nghị**

Không có MongoDB, hệ thống dùng in-memory dict (mất data khi restart).

**Cách lấy MongoDB Atlas (miễn phí 512 MB):**
1. Truy cập: https://cloud.mongodb.com
2. Tạo cluster → **Connect → Drivers → Python**
3. Copy connection string dạng:
   `mongodb+srv://user:password@cluster0.xxxxx.mongodb.net/`

**Hoặc self-hosted local:**
```bash
# Cài MongoDB trên Ubuntu
sudo apt install -y mongodb
sudo systemctl start mongodb
# Connection string:
# mongodb://localhost:27017/
```

### 2.5 Evidently AI — **Tùy chọn** (drift monitoring)

1. Truy cập: https://cloud.evidentlyai.com
2. Đăng ký → **Settings → API Keys**
3. Tạo project → copy Project ID

---

## 3. Cấu hình `.env`

Tạo file `.env` ở thư mục **gốc của project** (cùng cấp với `pyproject.toml`):

```bash
cp .env.example .env
```

Nội dung đầy đủ:

```ini
# ─────────────────────────────────────────────
# DIMT — Environment Configuration
# ─────────────────────────────────────────────

# ── MinerU (Bắt buộc) ────────────────────────
# Dùng để OCR và extract layout từ PDF
MINERU_API_KEY=your_mineru_api_key_here

# ── LLM Agent (Chọn một hoặc cả hai) ─────────
# DeepSeek qua NVIDIA API (mặc định)
DEEPSEEK_API_KEY=nvapi-your_nvidia_key_here

# Gemini (dự phòng hoặc thay thế)
GEMINI_API_KEY=AIzaSy_your_gemini_key_here

# ── MongoDB (Khuyến nghị) ─────────────────────
# Bỏ trống → in-memory fallback (mất data khi restart)
MONGODB_URL=mongodb+srv://user:pass@cluster.mongodb.net/

# ── Evidently AI (Tùy chọn) ──────────────────
# Drift monitoring cho translation quality
EVIDENT_AI_API_KEY=ev_your_key_here
EVIDENTLY_PROJECT_ID=your_project_uuid_here

# ── MLflow (Tự động, không cần key) ──────────
# Mặc định lưu local vào mlruns.db
# Để override sang remote server:
# MLFLOW_TRACKING_URI=http://your-mlflow-server:5000
```

### Kiểm tra `.env` đã load đúng

```bash
python -c "
from dotenv import load_dotenv
import os
load_dotenv()
keys = ['MINERU_API_KEY', 'DEEPSEEK_API_KEY', 'GEMINI_API_KEY', 'MONGODB_URL']
for k in keys:
    v = os.environ.get(k, '')
    status = '✅' if v else '⚠️  (chưa set)'
    print(f'{status} {k}: {v[:20]}...' if v else f'{status} {k}')
"
```

---

## 4. Cài đặt môi trường

### Bước 4.1 — Clone repo và cài `uv`

```bash
git clone https://github.com/your-org/dimt.git
cd dimt

# Cài uv (nếu chưa có)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env
```

### Bước 4.2 — Cài dependencies

```bash
# uv tự tạo .venv và cài tất cả dependencies từ pyproject.toml
# Bao gồm torch CUDA 11.8 từ pytorch-cu118 index
uv sync
```

> **Lưu ý GPU:** `pyproject.toml` cấu hình `torch` từ index `pytorch-cu118`.
> Nếu máy bạn dùng CUDA 12.x, sửa `pyproject.toml`:
> ```toml
> [[tool.uv.index]]
> url = "https://download.pytorch.org/whl/cu121"
> ```

### Bước 4.3 — Tạo thư mục runtime

```bash
mkdir -p input_docs output data
```

---

## 5. Chuẩn bị model NLLB

Model NLLB cần được đặt đúng vị trí trước khi khởi động backend.

### Cấu trúc thư mục yêu cầu

```
dimt/
├── nllb-1.3B-multilingual-final/   ← LoRA adapter (do bạn train)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── sentencepiece.bpe.model
│   └── special_tokens_map.json
├── src/
├── pyproject.toml
└── .env
```

### Nếu có adapter sẵn — copy vào đúng chỗ

```bash
# Ví dụ copy từ HuggingFace Hub
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='your-hf-username/nllb-1.3B-multilingual-final',
    local_dir='./nllb-1.3B-multilingual-final'
)
"
```

### Kiểm tra model load thành công

```bash
uv run python -c "
from src.backend.nllb_service import NLLBService
svc = NLLBService(lazy_load=False)
result = svc._translate_text('Hello, this is a test.')
print('Translation:', result)
print('Model loaded:', svc.loaded)
"
```

Output kỳ vọng:
```
[NLLB] Loading facebook/nllb-200-distilled-1.3B + LoRA adapter...
[NLLB] Model + LoRA adapter loaded successfully
Translation: Xin chào, đây là một bài kiểm tra.
Model loaded: True
```

---

## 6. Khởi động dịch vụ

### Terminal 1 — Backend FastAPI

```bash
cd dimt
uv run uvicorn src.backend.api:app --host 0.0.0.0 --port 8000 --reload
```

Log khởi động bình thường:
```
[API] Starting background NLLB model loading...
INFO:     Application startup complete.
[NLLB] Loading facebook/nllb-200-distilled-1.3B + LoRA adapter from nllb-1.3B-multilingual-final...
[MongoDB] Connected successfully to dimt-data      ← nếu có MONGODB_URL
[MinerU] API Key detected. Client initialized and ready.
[NLLB] Model + LoRA adapter loaded successfully    ← sau ~2-5 phút
```

> NLLB load ngầm (background thread). API đã sẵn sàng ngay, nhưng yêu cầu translate đầu tiên sẽ đợi model load xong.

### Terminal 2 — Frontend Streamlit

```bash
cd dimt
uv run streamlit run src/frontend/app.py --server.port 8501
```

Mở trình duyệt: http://localhost:8501

### Terminal 3 — MLflow UI (tùy chọn)

```bash
cd dimt
uv run mlflow ui --port 5000 --backend-store-uri sqlite:///mlruns.db
```

Mở trình duyệt: http://localhost:5000

---

## 7. Kiểm tra pipeline end-to-end

### Test nhanh qua curl

```bash
# 1. Health check
curl http://localhost:8000/docs

# 2. Upload PDF
curl -X POST http://localhost:8000/upload \
  -F "file=@/path/to/your/paper.pdf"
# Response: {"status":"success","doc_id":"abc12345","num_pages":10,"cached":false}

# 3. Translate (thay abc12345 bằng doc_id thực)
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"doc_id":"abc12345","tgt_lang":"vie_Latn"}'

# 4. Render PDF
curl -X POST http://localhost:8000/render-pdf \
  -H "Content-Type: application/json" \
  -d '{"doc_id":"abc12345"}'

# 5. Download PDF
curl -o translated.pdf http://localhost:8000/download/abc12345

# 6. Xem bilingual mapping
curl http://localhost:8000/bilingual/abc12345

# 7. Kiểm tra dedup (upload lại file cũ)
curl -X POST http://localhost:8000/upload \
  -F "file=@/path/to/your/paper.pdf"
# Response: {...,"cached":true}  ← không gọi MinerU lại
```

### Test agent endpoints

```bash
# Q4 verification
curl -X POST http://localhost:8000/agent/verify \
  -H "Content-Type: application/json" \
  -d '{"doc_id":"abc12345","llm_provider":"gemini"}'

# Keywords + WikiSearch
curl -X POST http://localhost:8000/agent/keywords \
  -H "Content-Type: application/json" \
  -d '{"doc_id":"abc12345","llm_provider":"gemini"}'

# Table recovery
curl -X POST http://localhost:8000/agent/table-recovery \
  -H "Content-Type: application/json" \
  -d '{"doc_id":"abc12345","llm_provider":"gemini"}'
```

---

## 8. Deploy production (Linux server)

### 8.1 — Chạy backend bằng systemd

Tạo file `/etc/systemd/system/dimt-backend.service`:

```ini
[Unit]
Description=DIMT FastAPI Backend
After=network.target mongod.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/dimt
EnvironmentFile=/home/ubuntu/dimt/.env
ExecStart=/home/ubuntu/.cargo/bin/uv run uvicorn src.backend.api:app \
          --host 0.0.0.0 --port 8000 --workers 1
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable dimt-backend
sudo systemctl start dimt-backend
sudo journalctl -u dimt-backend -f    # xem log
```

> **`--workers 1`** là bắt buộc vì GPU semaphore trong code giả định single-process.

### 8.2 — Chạy frontend bằng systemd

Tạo file `/etc/systemd/system/dimt-frontend.service`:

```ini
[Unit]
Description=DIMT Streamlit Frontend
After=dimt-backend.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/dimt
EnvironmentFile=/home/ubuntu/dimt/.env
ExecStart=/home/ubuntu/.cargo/bin/uv run streamlit run \
          src/frontend/app.py \
          --server.port 8501 \
          --server.address 0.0.0.0 \
          --server.headless true
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable dimt-frontend
sudo systemctl start dimt-frontend
```

### 8.3 — Nginx reverse proxy (HTTPS)

```nginx
# /etc/nginx/sites-available/dimt
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate     /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;

    # Streamlit frontend
    location / {
        proxy_pass         http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade $http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_set_header   Host $host;
        proxy_read_timeout 600s;    # cho phép translate PDF dài
    }

    # FastAPI backend
    location /api/ {
        rewrite            ^/api/(.*) /$1 break;
        proxy_pass         http://localhost:8000;
        proxy_read_timeout 600s;
        client_max_body_size 200M;  # giới hạn upload PDF
    }
}
```

```bash
sudo ln -s /etc/nginx/sites-available/dimt /etc/nginx/sites-enabled/
sudo certbot --nginx -d your-domain.com
sudo nginx -t && sudo systemctl reload nginx
```

---

## 9. Giám sát & troubleshoot

### Các lỗi thường gặp và cách xử lý

**Lỗi: NLLB adapter không tìm thấy**
```
OSError: nllb-1.3B-multilingual-final/ does not exist
```
→ Kiểm tra thư mục `nllb-1.3B-multilingual-final/` tồn tại ở root của project.
→ Đảm bảo chứa `adapter_config.json` và `adapter_model.safetensors`.

---

**Lỗi: CUDA out of memory**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```
→ Model đang load 4-bit quantization nhưng VRAM vẫn không đủ.
→ Giảm `BATCH_MAX_TOKENS` trong `nllb_service.py` từ 512 xuống 256.
→ Hoặc tăng VRAM bằng cách đóng các process GPU khác.

---

**Lỗi: MinerU rate limit**
```
[MinerU] Rate limit hit (429). Retrying...
```
→ Bình thường, code đã có retry với exponential backoff (tenacity).
→ Nếu liên tục fail: nâng plan MinerU hoặc giảm số file concurrent.

---

**Lỗi: MongoDB connection failed**
```
[MongoDB] Connection failed (in-memory fallback): ...
```
→ Không ảnh hưởng hoạt động — chỉ là mất persistence khi restart.
→ Kiểm tra `MONGODB_URL` đúng format và cluster đang chạy.
→ Với Atlas: whitelist IP server trong **Network Access**.

---

**Lỗi: Agent LLM không trả về JSON**
```
[Agent] Verification error: json.loads(...) JSONDecodeError
```
→ Bình thường với LLM lâu lâu trả về ngoài format.
→ Code đã xử lý fallback: block đó sẽ bị đánh dấu `"verdict": "REVIEW"`.

---

### Kiểm tra health

```bash
# Backend đang chạy không?
curl -s http://localhost:8000/docs | grep -o "DIMT"

# GPU được nhận diện không?
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# MongoDB connected không?
uv run python -c "
from src.backend.mongo_store import MongoDocStore
s = MongoDocStore()
print('MongoDB:', 'connected' if s._collection is not None else 'in-memory fallback')
"

# MLflow có ghi metrics không?
uv run mlflow runs list --experiment-name agentic_pipeline_eval 2>/dev/null | head -5
```

---

## 10. Tham chiếu nhanh

### Sơ đồ luồng dữ liệu

```
User upload PDF
    │
    ▼
STEP 0: SHA-256 → MongoDB lookup
    ├─ Cache hit  → trả về kết quả cũ ngay ──────────────────┐
    └─ Cache miss → tiếp tục pipeline              │
                │                                  │
                ▼                                  │
STEP 1: MinerU API → layout.json + images/ + .md  │
                │                                  │
                ▼                                  │
STEP 2: NLLB-1.3B+LoRA translate layout.json      │
        → translated_middle                        │
        → bilingual_mapping saved to MongoDB       │
                │                                  │
                ▼                                  │
STEP 3: PyMuPDF render translated PDF              │
        (dùng PNG gốc cho công thức)               │
                │                                  │
                ▼                                  │
STEP 4: AI Agent (parallel với STEP 3)             │
        ├── Q4 OCR verification                    │
        ├── Keyword extraction + WikiSearch         │
        └── Table recovery (patch untranslated)    │
                │                                  │
                ▼                                  │
        User download + feedback ←─────────────────┘
```

### Tóm tắt file cấu hình

| File | Mục đích |
|------|----------|
| `.env` | Tất cả API keys và connection strings |
| `nllb-1.3B-multilingual-final/` | LoRA adapter weights |
| `mlruns.db` | SQLite cho MLflow metrics (tự tạo) |
| `dimt_eval.db` | SQLite cho evaluation cases (tự tạo) |
| `input_docs/` | PDF gốc được lưu sau upload |
| `output/` | PDF đã dịch |
| `data/` | Mock data cho dev/demo |

### Ports mặc định

| Service | Port | URL |
|---------|------|-----|
| FastAPI backend | 8000 | http://localhost:8000/docs |
| Streamlit frontend | 8501 | http://localhost:8501 |
| MLflow UI | 5000 | http://localhost:5000 |
| MongoDB | 27017 | mongodb://localhost:27017 |
