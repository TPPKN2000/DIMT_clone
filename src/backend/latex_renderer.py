from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import fitz


@dataclass(frozen=True)
class EquationMetrics:
    png_bytes: bytes
    aspect_ratio: float
    descent_ratio: float


class TectonicEquationRenderer:
    """
    Render real LaTeX math fragments using Tectonic.

    The output contract matches PDFRenderer's existing equation renderer:
    transparent PNG bytes plus width/height and baseline metrics.
    """

    def __init__(
        self,
        cache_dir: str | Path = ".cache/latex",
        dpi: int = 300,
        timeout_sec: int = 20,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.timeout_sec = timeout_sec

        if shutil.which("tectonic") is None:
            raise RuntimeError(
                "Tectonic is not installed or not found in PATH. "
                "Install it locally or run in a container that includes Tectonic."
            )

    def render_and_metrics(self, latex: str, dpi: Optional[int] = None) -> dict | None:
        raw = latex.strip()
        if not raw:
            return None

        dpi = dpi or self.dpi
        key = self._cache_key(raw, dpi)
        png_path = self.cache_dir / f"{key}.png"
        meta_path = self.cache_dir / f"{key}.meta"

        cached = self._read_cache(png_path, meta_path)
        if cached is not None:
            return cached

        try:
            tex_source = self._build_tex(raw)
            png_bytes, aspect_ratio = self._compile_to_png(tex_source, dpi=dpi)
            metrics = {
                "png_bytes": png_bytes,
                "aspect_ratio": aspect_ratio,
                # External PDF rasterization does not expose a true math
                # baseline. Keep a stable estimate for existing alignment logic.
                "descent_ratio": 0.18,
            }
            png_path.write_bytes(metrics["png_bytes"])
            meta_path.write_text(
                f"{metrics['aspect_ratio']},{metrics['descent_ratio']}",
                encoding="utf-8",
            )
            return metrics
        except Exception as exc:
            raise RuntimeError(f"Tectonic failed to render LaTeX fragment: {raw[:120]!r}") from exc

    def _read_cache(self, png_path: Path, meta_path: Path) -> dict | None:
        if not png_path.exists() or not meta_path.exists():
            return None
        try:
            aspect_ratio, descent_ratio = meta_path.read_text(encoding="utf-8").split(",")
            return {
                "png_bytes": png_path.read_bytes(),
                "aspect_ratio": float(aspect_ratio),
                "descent_ratio": float(descent_ratio),
            }
        except Exception:
            png_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
            return None

    def _cache_key(self, latex: str, dpi: int) -> str:
        payload = f"tectonic-v1|dpi={dpi}|{latex}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _build_tex(self, latex: str) -> str:
        body = self._normalize_math_body(latex)
        return rf"""
\documentclass[preview,border=0pt]{{standalone}}
\usepackage{{amsmath}}
\usepackage{{amsfonts}}
\usepackage{{amssymb}}
\usepackage{{mathtools}}
\usepackage{{bm}}
\usepackage{{mathrsfs}}
\usepackage{{textcomp}}
\pagestyle{{empty}}
\begin{{document}}
{body}
\end{{document}}
""".strip()

    def _normalize_math_body(self, latex: str) -> str:
        """
        Preserve LaTeX semantics. Do not strip environments or rewrite macros.
        """
        s = latex.strip()
        if not s:
            return s

        if s.startswith("\\[") or s.startswith("$$") or (s.startswith("$") and s.endswith("$")):
            return s

        env_match = re.match(r"\\begin\{([^}]+)\}", s)
        if env_match:
            env_name = env_match.group(1).rstrip("*")
            display_envs = {
                "align",
                "alignat",
                "equation",
                "flalign",
                "gather",
                "multline",
            }
            if env_name in display_envs:
                return s
            return f"\\[\n{s}\n\\]"

        return f"${s}$"

    def _compile_to_png(self, tex_source: str, dpi: int) -> tuple[bytes, float]:
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            tex_path = tmpdir / "equation.tex"
            pdf_path = tmpdir / "equation.pdf"
            tex_path.write_text(tex_source, encoding="utf-8")

            cmd = [
                "tectonic",
                "--outdir",
                str(tmpdir),
                "--keep-logs",
                "--keep-intermediates",
                str(tex_path),
            ]
            subprocess.run(
                cmd,
                check=True,
                cwd=tmpdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.timeout_sec,
            )

            if not pdf_path.exists():
                raise RuntimeError("Tectonic did not produce equation.pdf")

            doc = fitz.open(pdf_path)
            page = doc[0]
            rect = page.rect
            zoom = dpi / 72.0
            pix = page.get_pixmap(
                matrix=fitz.Matrix(zoom, zoom),
                alpha=True,
                clip=rect,
            )
            png_bytes = pix.tobytes("png")
            aspect_ratio = rect.width / rect.height if rect.height > 0 else 1.0
            doc.close()
            return png_bytes, aspect_ratio
