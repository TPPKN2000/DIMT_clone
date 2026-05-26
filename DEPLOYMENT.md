# DIMT — Hướng dẫn Triển khai & Vận hành Chi tiết (Production)

> Tài liệu này hướng dẫn chi tiết cách thiết lập các khóa API, cấu hình dịch vụ Systemd, thiết lập Nginx Reverse Proxy (HTTPS) và xử lý các sự cố thường gặp khi triển khai hệ thống **DIMT** trên môi trường Ubuntu/Linux Server.

---

## 🔑 1. Chuẩn bị các API Keys & Dịch vụ Liên kết

Hệ thống cần các API Key từ các dịch vụ bên ngoài để hoạt động đầy đủ chức năng:

### 1.1. MinerU API Key (Bắt buộc)
MinerU là dịch vụ OCR và phân tích bố cục PDF cốt lõi của hệ thống.
1. Đăng ký tài khoản tại: [MinerU Dashboard](https://mineru.net).
2. Truy cập **Dashboard → API Keys → Create Key**.
3. Sao chép khóa API có định dạng bắt đầu bằng `eyJ...` hoặc `sk-...`.
*   *Lưu ý tài khoản miễn phí:* Hạn mức tối đa là 50 tệp/phút, 1000 lượt yêu cầu/phút, tệp không quá 200 trang hoặc 200 MB.

### 1.2. Gemini API Key (Bắt buộc cho Agent LLM)
Sử dụng cho các tác vụ của Agent như Q4 Verification và trích xuất từ khóa.
1. Đăng ký và tạo khóa tại: [Google AI Studio](https://aistudio.google.com/apikey).
2. Tạo khóa API mới và copy chuỗi kí tự có dạng `AIzaSy...`.

### 1.3. MongoDB Connection String (Khuyến nghị)
*   **MongoDB Atlas (Cloud miễn phí 512MB):**
    1. Đăng nhập [MongoDB Cloud](https://cloud.mongodb.com).
    2. Tạo Cluster miễn phí → Click **Connect → Drivers → Python** và sao chép chuỗi kết nối dạng:
       `mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/`
*   **Cài đặt cục bộ (Ubuntu):**
    ```bash
    sudo apt install -y mongodb
    sudo systemctl start mongodb
    # Connection string mặc định: mongodb://localhost:27017/
    ```

### 1.4. Evidently AI API Key (Tùy chọn - Giám sát chất lượng dịch)
1. Đăng ký tại: [Evidently AI Cloud](https://cloud.evidentlyai.com).
2. Truy cập **Settings → API Keys** tạo khóa mới và tạo một dự án mới để lấy **Project ID**.

---

## 🌐 2. Hướng dẫn Triển khai Production (Ubuntu/Linux Server)

### 2.1. Chạy Backend bằng Systemd Service
Tạo file cấu hình dịch vụ `/etc/systemd/system/dimt-backend.service`:
```ini
[Unit]
Description=DIMT FastAPI Backend Service
After=network.target mongod.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/dimt
EnvironmentFile=/home/ubuntu/dimt/.env
ExecStart=/home/ubuntu/.cargo/bin/uv run uvicorn src.backend.api:app --host 127.0.0.1 --port 8000 --workers 1
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Kích hoạt và khởi động dịch vụ Backend:
```bash
sudo systemctl daemon-reload
sudo systemctl enable dimt-backend
sudo systemctl start dimt-backend
sudo journalctl -u dimt-backend -f # Xem log realtime
```

### 2.2. Chạy Frontend bằng Systemd Service
Tạo file cấu hình dịch vụ `/etc/systemd/system/dimt-frontend.service`:
```ini
[Unit]
Description=DIMT Streamlit Frontend Service
After=dimt-backend.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/dimt
EnvironmentFile=/home/ubuntu/dimt/.env
ExecStart=/home/ubuntu/.cargo/bin/uv run streamlit run src/frontend/app.py --server.port 8501 --server.address 127.0.0.1 --server.headless true
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```
Kích hoạt và khởi động dịch vụ Frontend:
```bash
sudo systemctl enable dimt-frontend
sudo systemctl start dimt-frontend
```

### 2.3. Cấu hình Nginx Reverse Proxy & SSL (HTTPS)
Tạo tệp cấu hình ảo Nginx tại `/etc/nginx/sites-available/dimt`:
```nginx
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

    # Cấu hình Streamlit Frontend & WebSocket
    location / {
        proxy_pass         http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade $http_upgrade;
        proxy_set_header   Connection "upgrade";
        proxy_set_header   Host $host;
        proxy_read_timeout 600s; # Cho phép thời gian chờ dịch tài liệu dài
    }

    # Cấu hình FastAPI Backend API Routing
    location /api/ {
        rewrite            ^/api/(.*) /$1 break;
        proxy_pass         http://127.0.0.1:8000;
        proxy_read_timeout 600s;
        client_max_body_size 200M; # Giới hạn kích thước tải lên của tài liệu PDF
    }
}
```
Kích hoạt cấu hình và cài đặt chứng chỉ SSL tự động bằng Certbot:
```bash
sudo ln -s /etc/nginx/sites-available/dimt /etc/nginx/sites-enabled/
sudo certbot --nginx -d your-domain.com
sudo nginx -t && sudo systemctl reload nginx
```

---

## 🛠️ 3. Giám sát & Xử lý Sự cố (Troubleshooting)

### 3.1. Các lỗi thường gặp và giải pháp

*   **Lỗi: `torch.cuda.OutOfMemoryError: CUDA out of memory`**
    *   *Nguyên nhân:* Dung lượng VRAM của GPU bị cạn kiệt trong lúc chạy dịch, đặc biệt đối với NLLB.
    *   *Giải pháp:* Nếu dung lượng VRAM < 8GB, chuyển sang sử dụng MarianMT.
*   **Lỗi: `[MinerU] Rate limit hit (429). Retrying...`**
    *   *Nguyên nhân:* Vượt quá giới hạn gọi API của tài khoản MinerU.
    *   *Giải pháp:* Hệ thống tích hợp sẵn cơ chế thử lại (exponential backoff).
*   **Lỗi: `[MongoDB] Connection failed (in-memory fallback)`**
    *   *Nguyên nhân:* Chuỗi kết nối Database cấu hình sai hoặc dịch vụ MongoDB chưa khởi chạy.
    *   *Giải pháp:* Hệ thống vẫn sẽ hoạt động bình thường bằng cách lưu trữ tạm thời trên bộ nhớ RAM (lưu ý: lịch sử và dữ liệu đã dịch sẽ bị mất khi khởi động lại server). Nếu dùng MongoDB Atlas, hãy chắc chắn đã điền đúng mật khẩu và whitelist IP máy chủ trên trang quản trị.

### 3.2. Các lệnh kiểm tra nhanh trạng thái hệ thống (Health Check)
```bash
# Kiểm tra GPU có được PyTorch nhận diện hay không
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# Kiểm tra trạng thái kết nối Database MongoDB
uv run python -c "from src.backend.mongo_store import MongoDocStore; s = MongoDocStore(); print('MongoDB:', 'connected' if s._collection is not None else 'in-memory fallback')"

# Kiểm tra phản hồi từ Backend API
curl -s http://127.0.0.1:8000/docs | grep -o "DIMT"

# Xem danh sách các lần đánh giá ghi nhận trên MLflow
uv run mlflow runs list --experiment-name agentic_pipeline_eval 2>/dev/null | head -5
```

---

## 🧪 4. Kiểm tra nhanh các Endpoints qua Curl

Bạn có thể test trực tiếp luồng xử lý của Backend bằng các dòng lệnh Terminal:

```bash
# 1. Kiểm tra Health Check
curl http://127.0.0.1:8000/docs

# 2. Tải tài liệu PDF lên
curl -X POST http://127.0.0.1:8000/upload \
  -F "file=@/path/to/your/paper.pdf"
# Kết quả phản hồi mẫu: {"status":"success","doc_id":"abc12345","num_pages":10,"cached":false}

# 3. Tiến hành Dịch thuật (Thay thế abc12345 bằng doc_id thực tế nhận được)
curl -X POST http://127.0.0.1:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"doc_id":"abc12345","tgt_lang":"vie_Latn"}'

# 4. Tái cấu trúc và kết xuất PDF đã dịch
curl -X POST http://127.0.0.1:8000/render-pdf \
  -H "Content-Type: application/json" \
  -d '{"doc_id":"abc12345"}'

# 5. Tải bản dịch về máy
curl -o translated.pdf http://127.0.0.1:8000/download/abc12345
```
