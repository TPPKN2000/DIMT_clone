import os
import random
import fitz  # PyMuPDF
from pathlib import Path
from src.backend.mineru_client import MinerUClient

def run_test():
    # Setup directories
    input_dir = Path("input_test")
    output_dir = Path("output_test")
    
    if not input_dir.exists():
        print(f"Directory {input_dir} does not exist.")
        return
        
    output_dir.mkdir(exist_ok=True)
    
    # 1. Lấy một file PDF ngẫu nhiên trong thư mục input_test/
    pdf_files = list(input_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return
        
    pdf_path = random.choice(pdf_files)
    print(f"Selected PDF for testing: {pdf_path}")
    
    # 2. Cắt 5 trang đầu tiên
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    print(f"Original PDF has {total_pages} pages.")
    
    pages_to_keep = min(5, total_pages)
    if total_pages > 5:
        doc.select(range(pages_to_keep))
        
    test_pdf_path = output_dir / f"test_5pages_{pdf_path.name}"
    doc.save(test_pdf_path)
    doc.close()
    
    print(f"Created {pages_to_keep}-page test PDF: {test_pdf_path}")
    
    # 3. Chạy qua MinerU API
    print("-" * 50)
    print("Initiating MinerU API extraction...")
    print("-" * 50)
    
    # Khởi tạo client lưu output vào output_test
    client = MinerUClient(output_dir=str(output_dir))
    
    if not client.token:
        print("ERROR: MINERU_API_KEY is not set in .env")
        return
        
    result = client.extract_from_file(str(test_pdf_path))
    
    # 4. Xuất và lưu kết quả dictionary
    print("-" * 50)
    print(f"MinerU API Result Status: {result.get('status')}")
    
    # Save the exact dictionary returned by the client
    import json
    result_json_path = output_dir / f"api_result_{pdf_path.stem}.json"
    
    # Create a copy without the full markdown/json string to avoid massive files, or just dump it all
    # We will dump everything for inspection.
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
        
    print(f"Lưu toàn bộ kết quả trả về của API vào file: {result_json_path}")
    
    if result.get("status") == "success":
        print("✅ Extraction successful!")
        print(f"Các file giải nén (md, images, middle_json) nằm tại: {result.get('extract_dir')}")
    else:
        print(f"❌ Extraction failed: {result.get('message')}")

if __name__ == "__main__":
    run_test()
