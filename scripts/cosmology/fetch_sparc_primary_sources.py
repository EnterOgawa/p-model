#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_sparc_primary_sources.py

Phase 6 / Step 6.5（SPARC：RAR/BTFR）:
SPARC/RAR の一次論文（arXiv PDF + source）を取得し、offline 再現できる形でキャッシュする。

保存先（入力の正）:
- data/cosmology/sources/arxiv_<id>.pdf
- data/cosmology/sources/arxiv_<id>_src.tar.gz
- data/cosmology/sources/arxiv_<id>/  (extract; optional)
- data/cosmology/sources/sparc_primary_sources_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from scripts.summary import worklog  # type: ignore # noqa: E402
except Exception:  # pragma: no cover
    worklog = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def _read_head(path: Path, n: int = 8) -> bytes:
    with path.open("rb") as f:
        return f.read(n)


def _download(url: str, dst: Path, *, force: bool, max_bytes: int) -> Optional[str]:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 0 and not force:
        return None

    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "waveP-sparc-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        total = r.headers.get("Content-Length")
        total_i = int(total) if total is not None and total.isdigit() else None
        if total_i is not None and total_i > max_bytes:
            raise RuntimeError(f"refusing to download large file: {total_i} bytes > max={max_bytes}: {url}")
        with tmp.open("wb") as f:
            shutil.copyfileobj(r, f, length=1024 * 1024)

    if tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        return "downloaded empty file"
    tmp.replace(dst)
    return None


def _safe_extractall(tf: tarfile.TarFile, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for m in tf.getmembers():
        if not isinstance(m.name, str):
            continue
        p = Path(m.name)
        if p.is_absolute():
            raise RuntimeError(f"unsafe tar member (absolute path): {m.name}")
        if any(part in ("..", "") for part in p.parts):
            raise RuntimeError(f"unsafe tar member (path traversal): {m.name}")
    # Python 3.14 will change tarfile extraction defaults; use an explicit filter when available.
    try:
        tf.extractall(out_dir, filter="data")  # type: ignore[call-arg]
    except TypeError:
        tf.extractall(out_dir)


def _extract_src(tar_path: Path, out_dir: Path, *, force: bool) -> bool:
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        return True

    if force and out_dir.exists():
        for p in out_dir.glob("*"):
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink(missing_ok=True)

    head = _read_head(tar_path, 5)
    if head.startswith(b"%PDF"):
        return False
    if not head.startswith(b"\x1f\x8b"):
        return False

    with tarfile.open(tar_path, "r:gz") as tf:
        _safe_extractall(tf, out_dir)
    return True


def _count_tex_files(dir_path: Path) -> Optional[int]:
    if not (dir_path.exists() and dir_path.is_dir()):
        return None
    return sum(1 for _ in dir_path.rglob("*.tex"))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch SPARC/RAR primary papers (arXiv PDF + source tarball).")
    ap.add_argument("--offline", action="store_true", help="Do not download; only verify existing files.")
    ap.add_argument("--force", action="store_true", help="Redownload/re-extract even if files already exist.")
    ap.add_argument("--max-gib", type=float, default=0.5, help="Refuse single-file downloads larger than this (GiB). Default: 0.5")
    args = ap.parse_args(list(argv) if argv is not None else None)

    src_dir = _ROOT / "data" / "cosmology" / "sources"
    src_dir.mkdir(parents=True, exist_ok=True)

    papers = [
        ("sparc_i_mass_models", "1606.09251", "SPARC I: Mass Models for 175 Disk Galaxies"),
        ("rar_one_law_to_rule_them_all", "1610.08981", "RAR: One Law to Rule Them All"),
    ]

    max_bytes = int(float(args.max_gib) * (1024**3))
    rows: List[Dict[str, Any]] = []
    for key, arxiv_id, label in papers:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        src_url = f"https://arxiv.org/e-print/{arxiv_id}"

        pdf_path = src_dir / f"arxiv_{arxiv_id}.pdf"
        src_path = src_dir / f"arxiv_{arxiv_id}_src.tar.gz"
        extract_dir = src_dir / f"arxiv_{arxiv_id}"

        notes: List[str] = []
        if not args.offline:
            try:
                err = _download(pdf_url, pdf_path, force=bool(args.force), max_bytes=max_bytes)
                if err:
                    notes.append(f"pdf: {err}")
            except Exception as e:
                notes.append(f"pdf download failed: {e}")
            try:
                err = _download(src_url, src_path, force=bool(args.force), max_bytes=max_bytes)
                if err:
                    notes.append(f"src: {err}")
            except Exception as e:
                notes.append(f"src download failed: {e}")

        extracted_ok = False
        try:
            if src_path.exists() and src_path.stat().st_size > 0:
                extracted_ok = _extract_src(src_path, extract_dir, force=bool(args.force))
        except Exception as e:
            notes.append(f"extract failed: {e}")
            extracted_ok = False

        rows.append(
            {
                "key": key,
                "arxiv": arxiv_id,
                "label": label,
                "url_abs": f"https://arxiv.org/abs/{arxiv_id}",
                "local_pdf": str(pdf_path.relative_to(_ROOT)).replace("\\", "/") if pdf_path.exists() else None,
                "local_pdf_sha256": _sha256(pdf_path) if pdf_path.exists() else None,
                "local_src": str(src_path.relative_to(_ROOT)).replace("\\", "/") if src_path.exists() else None,
                "local_src_sha256": _sha256(src_path) if src_path.exists() else None,
                "extract_dir": str(extract_dir.relative_to(_ROOT)).replace("\\", "/") if extract_dir.exists() else None,
                "extract_ok": extracted_ok,
                "tex_files": _count_tex_files(extract_dir),
                "notes": notes,
            }
        )

    manifest = {
        "generated_utc": _utc_now(),
        "dataset": "SPARC / RAR primary sources (arXiv)",
        "mode": {"offline": bool(args.offline), "force": bool(args.force), "max_gib": float(args.max_gib)},
        "rows": rows,
    }
    out_manifest = src_dir / "sparc_primary_sources_manifest.json"
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if worklog is not None:
        try:
            worklog.append_event(
                {
                    "ts_utc": manifest["generated_utc"],
                    "topic": "cosmology_sparc",
                    "action": "fetch_sparc_primary_sources",
                    "outputs": [str(out_manifest.relative_to(_ROOT)).replace("\\", "/")],
                    "metrics": {
                        "papers": len(rows),
                        "pdf_ok": sum(1 for r in rows if r.get("local_pdf_sha256")),
                        "src_ok": sum(1 for r in rows if r.get("local_src_sha256")),
                        "extract_ok": sum(1 for r in rows if r.get("extract_ok") is True),
                    },
                }
            )
        except Exception:
            pass

    print(f"[ok] manifest: {out_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
