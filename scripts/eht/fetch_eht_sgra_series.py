#!/usr/bin/env python3
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
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


def _repo_root() -> Path:
    return _ROOT


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest().upper()


def _read_head(path: Path, n: int = 8) -> bytes:
    with open(path, "rb") as f:
        return f.read(n)


def _download(url: str, dst: Path, *, force: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    # 条件分岐: `dst.exists() and not force` を満たす経路を評価する。
    if dst.exists() and not force:
        print(f"[skip] exists: {dst}")
        return

    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"[dl] {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "waveP-eht-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1024 * 1024)

    tmp.replace(dst)
    print(f"[ok] saved: {dst} ({dst.stat().st_size} bytes)")


def _safe_extractall(tf: tarfile.TarFile, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for m in tf.getmembers():
        # 条件分岐: `not isinstance(m.name, str)` を満たす経路を評価する。
        if not isinstance(m.name, str):
            continue

        p = Path(m.name)
        # 条件分岐: `p.is_absolute()` を満たす経路を評価する。
        if p.is_absolute():
            raise RuntimeError(f"unsafe tar member (absolute path): {m.name}")

        # 条件分岐: `any(part in ("..", "") for part in p.parts)` を満たす経路を評価する。

        if any(part in ("..", "") for part in p.parts):
            raise RuntimeError(f"unsafe tar member (path traversal): {m.name}")

    tf.extractall(out_dir)


def _extract_src(tar_path: Path, out_dir: Path, *, force: bool) -> bool:
    # 条件分岐: `out_dir.exists() and any(out_dir.iterdir()) and not force` を満たす経路を評価する。
    if out_dir.exists() and any(out_dir.iterdir()) and not force:
        print(f"[skip] extracted: {out_dir}")
        return True

    # 条件分岐: `force and out_dir.exists()` を満たす経路を評価する。

    if force and out_dir.exists():
        for p in out_dir.glob("*"):
            # 条件分岐: `p.is_dir()` を満たす経路を評価する。
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink(missing_ok=True)

    head = _read_head(tar_path, 5)
    # 条件分岐: `head.startswith(b"%PDF")` を満たす経路を評価する。
    if head.startswith(b"%PDF"):
        print(f"[warn] e-print looks like PDF (not tar.gz): {tar_path.name} (skipping extract)")
        return False

    # 条件分岐: `not head.startswith(b"\x1f\x8b")` を満たす経路を評価する。

    if not head.startswith(b"\x1f\x8b"):
        print(f"[warn] e-print is not gzip (skipping extract): {tar_path.name}")
        return False

    try:
        with tarfile.open(tar_path, "r:gz") as tf:
            _safe_extractall(tf, out_dir)

        print(f"[ok] extracted: {out_dir}")
        return True
    except tarfile.ReadError as e:
        print(f"[warn] tar read failed: {tar_path.name}: {e} (skipping extract)")
        return False


def _count_tex_files(dir_path: Path) -> Optional[int]:
    # 条件分岐: `not (dir_path.exists() and dir_path.is_dir())` を満たす経路を評価する。
    if not (dir_path.exists() and dir_path.is_dir()):
        return None

    return sum(1 for _ in dir_path.rglob("*.tex"))


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch EHT Sgr A* series (arXiv PDFs + sources) for offline reproducibility.")
    ap.add_argument("--offline", action="store_true", help="Do not download; only verify existing files and write manifest.")
    ap.add_argument("--force", action="store_true", help="Redownload/re-extract even if files already exist.")
    args = ap.parse_args()

    root = _repo_root()
    sources_dir = root / "data" / "eht" / "sources"
    out_dir = root / "output" / "private" / "eht"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sgr A* 2022 (ApJL 930 L12-L17) series as posted on arXiv (2023-11).
    # Paper I is already used elsewhere but included here for completeness.
    papers = [
        ("eht_sgra_paper2_2022", "2311.08679", "EHT Sgr A* Paper II (arXiv:2311.08679)"),
        ("eht_sgra_paper1_2022", "2311.08680", "EHT Sgr A* Paper I (arXiv:2311.08680)"),
        ("eht_sgra_paper4_2022", "2311.08697", "EHT Sgr A* Paper IV (arXiv:2311.08697)"),
        ("eht_sgra_paper5_2022", "2311.09478", "EHT Sgr A* Paper V (arXiv:2311.09478)"),
        ("eht_sgra_paper3_2022", "2311.09479", "EHT Sgr A* Paper III (arXiv:2311.09479)"),
        ("eht_sgra_paper6_2022", "2311.09484", "EHT Sgr A* Paper VI (arXiv:2311.09484)"),
    ]

    rows: List[Dict[str, Any]] = []
    for key, arxiv_id, label in papers:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        src_url = f"https://arxiv.org/e-print/{arxiv_id}"

        pdf_path = sources_dir / f"arxiv_{arxiv_id}.pdf"
        src_path = sources_dir / f"arxiv_{arxiv_id}_src.tar.gz"
        extract_dir = sources_dir / f"arxiv_{arxiv_id}"

        notes: List[str] = []
        # 条件分岐: `not args.offline` を満たす経路を評価する。
        if not args.offline:
            try:
                _download(pdf_url, pdf_path, force=bool(args.force))
            except Exception as e:
                notes.append(f"pdf download failed: {e}")

            try:
                _download(src_url, src_path, force=bool(args.force))
            except Exception as e:
                notes.append(f"src download failed: {e}")

        extracted_ok = False
        try:
            # 条件分岐: `src_path.exists()` を満たす経路を評価する。
            if src_path.exists():
                extracted_ok = _extract_src(src_path, extract_dir, force=bool(args.force))
        except Exception as e:
            notes.append(f"extract failed: {e}")
            extracted_ok = False

        row: Dict[str, Any] = {
            "key": key,
            "arxiv": arxiv_id,
            "label": label,
            "url_abs": f"https://arxiv.org/abs/{arxiv_id}",
            "local_pdf": str(pdf_path.relative_to(root)).replace("\\", "/") if pdf_path.exists() else None,
            "local_pdf_sha256": _sha256(pdf_path) if pdf_path.exists() else None,
            "local_src": str(src_path.relative_to(root)).replace("\\", "/") if src_path.exists() else None,
            "local_src_sha256": _sha256(src_path) if src_path.exists() else None,
            "extract_dir": str(extract_dir.relative_to(root)).replace("\\", "/") if extract_dir.exists() else None,
            "extract_ok": extracted_ok,
            "tex_files": _count_tex_files(extract_dir),
            "notes": notes,
        }
        rows.append(row)

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "mode": {"offline": bool(args.offline), "force": bool(args.force)},
        "output": "output/private/eht/eht_sgra_series_sources_manifest.json",
        "rows": rows,
    }
    out_json = out_dir / "eht_sgra_series_sources_manifest.json"
    out_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    try:
        worklog.append_event(
            {
                "ts_utc": manifest["generated_utc"],
                "topic": "eht",
                "action": "fetch_eht_sgra_series",
                "outputs": [str(out_json.relative_to(root)).replace("\\", "/")],
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

    print(f"[ok] json: {out_json}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
