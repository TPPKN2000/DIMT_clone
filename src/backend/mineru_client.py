"""
MinerU API Client — Production-ready with async polling and ZIP extraction.

MinerU API Flow (Precision API v4):
1. Local files: POST /api/v4/file-urls/batch → signed upload URLs + batch_id
   → PUT file to OSS → system auto-starts parsing
2. Remote URLs: POST /api/v4/extract/task → task_id
3. Poll GET /api/v4/extract/task/{task_id} until state="done"
4. Download ZIP from full_zip_url
5. Extract: <name>.md, <name>_middle.json, images/

Rate limits (from official docs):
- Submission: 50 files/min, max 5000 files/day
- Polling: 1000 requests/min
- File constraints: ≤200MB, ≤200 pages
- Data retention: 24 hours after completion
"""

import os
import time
import json
import zipfile
import requests
from pathlib import Path
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

load_dotenv()


class MinerUClient:
    BASE_URL = "https://mineru.net/api/v4"

    def __init__(self, output_dir: str = "data"):
        self.token = os.environ.get("MINERU_API_KEY", "")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }
        if self.token:
            print("[MinerU] API Key detected. Client initialized and ready.")
        else:
            print("[MinerU] WARNING: No MINERU_API_KEY found in .env!")

    # ── Public API ──────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        retry=retry_if_exception_type(requests.exceptions.RequestException),
        reraise=True,
    )
    def extract_from_file(self, file_path: str, model_version: str = "vlm") -> dict:
        """Upload a local file via signed-URL flow and extract."""
        file_path = Path(file_path)
        if not file_path.exists():
            return {"status": "error", "message": f"File not found: {file_path}"}

        # Step 1 — get signed upload URL
        data = {
            "files": [{"name": file_path.name, "data_id": file_path.stem}],
            "model_version": model_version,
        }
        try:
            res = requests.post(
                f"{self.BASE_URL}/file-urls/batch",
                headers=self.headers,
                json=data,
                timeout=30,
            )
            if res.status_code == 429:
                print("[MinerU] Rate limit hit (429). Retrying...")
                raise requests.exceptions.RequestException("Rate limit exceeded")
            res.raise_for_status()
            result = res.json()
            if result.get("code") != 0:
                return {"status": "error", "message": result.get("msg", "Unknown")}

            batch_id = result["data"]["batch_id"]
            upload_urls = result["data"]["file_urls"]

            # Step 2 — PUT file to OSS
            with open(file_path, "rb") as f:
                put_res = requests.put(upload_urls[0], data=f, timeout=120)
            if put_res.status_code not in (200, 201):
                return {"status": "error", "message": f"Upload failed: HTTP {put_res.status_code}"}

            # Step 3 — poll batch results
            return self._poll_batch(batch_id)

        except requests.RequestException as e:
            return {"status": "error", "message": f"Network error: {e}"}

    def extract_from_url(self, pdf_url: str, model_version: str = "vlm") -> dict:
        """Extract from a publicly-accessible URL."""
        data = {"url": pdf_url, "model_version": model_version}
        try:
            res = requests.post(
                f"{self.BASE_URL}/extract/task",
                headers=self.headers,
                json=data,
                timeout=30,
            )
            res.raise_for_status()
            result = res.json()
            if result.get("code") != 0:
                return {"status": "error", "message": result.get("msg", "Unknown")}
            return self._poll_task(result["data"]["task_id"])
        except requests.RequestException as e:
            return {"status": "error", "message": f"Network error: {e}"}

    def extract_local_mock(self, pdf_filename: str) -> dict:
        """Fallback for dev/demo — reads pre-existing files in data/ or output_test/."""
        base = pdf_filename.replace("_origin.pdf", "").replace(".pdf", "")
        
        # Candidate search paths
        search_dirs = [self.output_dir, Path("output_test")]
        
        # Try finding a directory that contains both layout.json and full.md
        for d in search_dirs:
            if not d.exists(): continue
            
            # Deep search for layout.json and a matching .md file
            for root, dirs, files in os.walk(d):
                root_path = Path(root)
                md_file = None
                middle = None
                images = root_path / "images"
                
                # Check for files in this directory
                for fn in files:
                    fp = root_path / fn
                    # Prioritize full.md or files matching the base name
                    if fn == "full.md" or fn == f"{base}.md":
                        md_file = fp
                    elif fn == "layout.json" or fn == f"{base}_middle.json":
                        middle = fp
                    elif not md_file and fn.endswith(".md") and not fn.startswith("_"):
                        md_file = fp
                    elif not middle and (fn.endswith("_middle.json") or fn.endswith("layout.json")):
                        middle = fp

                # If we found both, verify this is likely the right one 
                # (e.g. check if the PDF base name is in the directory or files)
                if md_file and middle:
                    is_match = (
                        base in root_path.name or
                        base in md_file.name or
                        base in middle.name
                    )
                    
                    if is_match:
                        print(f"[MinerU] Found mock data for {base} in {root}")
                        return self._build_result(md_file, middle, images if images.exists() else None, str(root))

        return {"status": "error", "message": f"Mock data not found for {base}"}

    # ── Polling ─────────────────────────────────────────────────

    def _poll_task(self, task_id: str, timeout: int = 600, interval: int = 5) -> dict:
        start = time.time()
        while time.time() - start < timeout:
            try:
                res = requests.get(
                    f"{self.BASE_URL}/extract/task/{task_id}",
                    headers=self.headers,
                    timeout=30,
                )
                data = res.json().get("data", {})
                state = data.get("state", "unknown")

                if state == "done":
                    return self._download_and_extract(data["full_zip_url"], task_id)
                if state == "failed":
                    return {"status": "error", "message": data.get("err_msg", "Unknown")}

                prog = data.get("extract_progress", {})
                print(f"[MinerU] {task_id}: {state} "
                      f"({prog.get('extracted_pages','?')}/{prog.get('total_pages','?')} pages)")
            except requests.RequestException as e:
                print(f"[MinerU] Poll error (retrying): {e}")
            time.sleep(interval)
        return {"status": "error", "message": f"Timeout after {timeout}s"}

    def _poll_batch(self, batch_id: str, timeout: int = 600, interval: int = 5) -> dict:
        start = time.time()
        while time.time() - start < timeout:
            try:
                res = requests.get(
                    f"{self.BASE_URL}/extract-results/batch/{batch_id}",
                    headers=self.headers,
                    timeout=30,
                )
                tasks = res.json().get("data", {}).get("extract_result", [])
                if tasks:
                    t = tasks[0]
                    if t["state"] == "done":
                        return self._download_and_extract(t["full_zip_url"], batch_id)
                    if t["state"] == "failed":
                        return {"status": "error", "message": t.get("err_msg", "Unknown")}
                    prog = t.get("extract_progress", {})
                    print(f"[MinerU] batch {batch_id}: {t['state']} "
                          f"({prog.get('extracted_pages','?')}/{prog.get('total_pages','?')})")
            except requests.RequestException as e:
                print(f"[MinerU] Poll error (retrying): {e}")
            time.sleep(interval)
        return {"status": "error", "message": f"Timeout after {timeout}s"}

    # ── Download & Extract ──────────────────────────────────────

    def _download_and_extract(self, zip_url: str, job_id: str) -> dict:
        extract_dir = self.output_dir / job_id
        extract_dir.mkdir(parents=True, exist_ok=True)

        res = requests.get(zip_url, timeout=120, stream=True)
        res.raise_for_status()
        zip_path = extract_dir / "result.zip"
        with open(zip_path, "wb") as f:
            for chunk in res.iter_content(8192):
                f.write(chunk)

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)

        # Locate key files
        md_file = middle = images_dir = None
        for root, dirs, files in os.walk(extract_dir):
            for fn in files:
                fp = Path(root) / fn
                if fn == "full.md" or (fn.endswith(".md") and not fn.startswith("_")):
                    md_file = fp
                elif fn == "layout.json" or fn.endswith("_middle.json"):
                    middle = fp
            if "images" in dirs:
                images_dir = Path(root) / "images"

        if not md_file:
            return {"status": "error", "message": "No .md file in ZIP"}
        return self._build_result(md_file, middle, images_dir, str(extract_dir))

    @staticmethod
    def _build_result(md_file, middle, images_dir, extract_dir) -> dict:
        result = {
            "status": "success",
            "md_path": str(md_file),
            "middle_json_path": str(middle) if middle and middle.exists() else None,
            "images_dir": str(images_dir) if images_dir and images_dir.exists() else None,
            "extract_dir": extract_dir,
            "markdown": Path(md_file).read_text(encoding="utf-8"),
        }
        if middle and middle.exists():
            result["middle_json"] = json.loads(middle.read_text(encoding="utf-8"))
        return result
