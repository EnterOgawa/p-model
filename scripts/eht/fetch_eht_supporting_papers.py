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


# 関数: `_repo_root` の入出力契約と処理意図を定義する。
def _repo_root() -> Path:
    return _ROOT


# 関数: `_sha256` の入出力契約と処理意図を定義する。

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest().upper()


# 関数: `_read_head` の入出力契約と処理意図を定義する。

def _read_head(path: Path, n: int = 8) -> bytes:
    with open(path, "rb") as f:
        return f.read(n)


# 関数: `_download` の入出力契約と処理意図を定義する。

def _download(url: str, dst: Path, *, force: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    # 条件分岐: `dst.exists() and not force` を満たす経路を評価する。
    if dst.exists() and not force:
        print(f"[skip] exists: {dst}")
        return

    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"[dl] {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "waveP-eht-supporting-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=240) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1024 * 1024)

    tmp.replace(dst)
    print(f"[ok] saved: {dst} ({dst.stat().st_size} bytes)")


# 関数: `_safe_extractall` の入出力契約と処理意図を定義する。

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


# 関数: `_extract_src` の入出力契約と処理意図を定義する。

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


# 関数: `_count_tex_files` の入出力契約と処理意図を定義する。

def _count_tex_files(dir_path: Path) -> Optional[int]:
    # 条件分岐: `not (dir_path.exists() and dir_path.is_dir())` を満たす経路を評価する。
    if not (dir_path.exists() and dir_path.is_dir()):
        return None

    return sum(1 for _ in dir_path.rglob("*.tex"))


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Fetch EHT-related supporting papers (PDF + arXiv source when available) for offline reproducibility.\n"
            "Includes: M87 2018 (A&A 2024/2025), thick disk/photon ring systematics, and photon-ring GR tests."
        )
    )
    ap.add_argument("--offline", action="store_true", help="Do not download; only verify existing files and write manifest.")
    ap.add_argument("--force", action="store_true", help="Redownload/re-extract even if files already exist.")
    args = ap.parse_args()

    root = _repo_root()
    sources_dir = root / "data" / "eht" / "sources"
    out_dir = root / "output" / "private" / "eht"
    out_dir.mkdir(parents=True, exist_ok=True)

    # NOTE:
    # - For arXiv items, we store both PDF and e-print (source) when possible.
    # - For A&A items, we store the publisher PDF (open access) since arXiv may be absent/ambiguous.
    papers: List[Dict[str, Any]] = [
        {
            "key": "eht_m87_2018_paper1_2024",
            "label": "EHT Collaboration (2024) The persistent shadow of the supermassive black hole of M87. I. Observations, calibration, imaging, and analysis",
            "kind": "direct_pdf",
            "doi": "10.1051/0004-6361/202347932",
            "url_landing": "https://www.aanda.org/articles/aa/abs/2024/01/aa47932-23/aa47932-23.html",
            "url_pdf": "https://www.aanda.org/articles/aa/pdf/2024/01/aa47932-23.pdf",
            "local_pdf_name": "aanda_aa47932-23.pdf",
        },
        {
            "key": "eht_m87_2018_paper2_2025",
            "label": "EHT Collaboration (2025) The persistent shadow of the supermassive black hole of M87. II. Model comparisons and theoretical interpretations",
            "kind": "direct_pdf",
            "doi": "10.1051/0004-6361/202451296",
            "url_landing": "https://www.aanda.org/articles/aa/abs/2025/01/aa51296-24/aa51296-24.html",
            "url_pdf": "https://www.aanda.org/articles/aa/pdf/2025/01/aa51296-24.pdf",
            "local_pdf_name": "aanda_aa51296-24.pdf",
        },
        {
            "key": "vincent_2022_thick_disk_photon_ring",
            "label": "Vincent et al. (2022) Thick accretion discs and photon rings: influence on the shadow/ring correspondence",
            "kind": "arxiv",
            "arxiv": "2206.12066",
            "doi": "10.1051/0004-6361/202244339",
        },
        {
            "key": "johnson_2020_photon_ring_interferometric_signature",
            "label": "Johnson et al. (2020) Universal interferometric signatures of a black hole’s photon ring",
            "kind": "arxiv",
            "arxiv": "1907.04329",
            "doi": "10.1126/sciadv.aaz1310",
        },
        {
            "key": "gralla_2020_photon_ring_shape_test",
            "label": "Gralla et al. (2020) Black hole photon ring shape as a test of general relativity",
            "kind": "arxiv",
            "arxiv": "2008.03879",
            "doi": "10.1103/PhysRevD.102.124004",
        },
    ]

    rows: List[Dict[str, Any]] = []
    for paper in papers:
        key = str(paper.get("key") or "").strip()
        kind = str(paper.get("kind") or "").strip()
        label = str(paper.get("label") or "").strip()
        doi = str(paper.get("doi") or "").strip() or None

        notes: List[str] = []
        # 条件分岐: `kind == "arxiv"` を満たす経路を評価する。
        if kind == "arxiv":
            arxiv_id = str(paper.get("arxiv") or "").strip()
            # 条件分岐: `not arxiv_id` を満たす経路を評価する。
            if not arxiv_id:
                raise RuntimeError(f"missing arxiv id for: {key}")

            pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            src_url = f"https://arxiv.org/e-print/{arxiv_id}"

            pdf_path = sources_dir / f"arxiv_{arxiv_id}.pdf"
            src_path = sources_dir / f"arxiv_{arxiv_id}_src.tar.gz"
            extract_dir = sources_dir / f"arxiv_{arxiv_id}"

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
                "label": label,
                "kind": kind,
                "doi": doi,
                "arxiv": arxiv_id,
                "url_abs": f"https://arxiv.org/abs/{arxiv_id}",
                "url_pdf": pdf_url,
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
            continue

        # 条件分岐: `kind == "direct_pdf"` を満たす経路を評価する。

        if kind == "direct_pdf":
            url_pdf = str(paper.get("url_pdf") or "").strip()
            url_landing = str(paper.get("url_landing") or "").strip() or None
            local_name = str(paper.get("local_pdf_name") or "").strip()
            # 条件分岐: `not (url_pdf and local_name)` を満たす経路を評価する。
            if not (url_pdf and local_name):
                raise RuntimeError(f"missing url_pdf/local_pdf_name for: {key}")

            pdf_path = sources_dir / local_name
            # 条件分岐: `not args.offline` を満たす経路を評価する。
            if not args.offline:
                try:
                    _download(url_pdf, pdf_path, force=bool(args.force))
                except Exception as e:
                    notes.append(f"pdf download failed: {e}")

            row2: Dict[str, Any] = {
                "key": key,
                "label": label,
                "kind": kind,
                "doi": doi,
                "url_landing": url_landing,
                "url_pdf": url_pdf,
                "local_pdf": str(pdf_path.relative_to(root)).replace("\\", "/") if pdf_path.exists() else None,
                "local_pdf_sha256": _sha256(pdf_path) if pdf_path.exists() else None,
                "notes": notes,
            }
            rows.append(row2)
            continue

        raise RuntimeError(f"unknown kind: {kind} ({key})")

    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "mode": {"offline": bool(args.offline), "force": bool(args.force)},
        "output": "output/private/eht/eht_supporting_papers_sources_manifest.json",
        "rows": rows,
    }
    out_json = out_dir / "eht_supporting_papers_sources_manifest.json"
    out_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    try:
        worklog.append_event(
            {
                "ts_utc": manifest["generated_utc"],
                "topic": "eht",
                "action": "fetch_eht_supporting_papers",
                "outputs": [str(out_json.relative_to(root)).replace("\\", "/")],
                "metrics": {
                    "papers": len(rows),
                    "pdf_ok": sum(1 for r in rows if r.get("local_pdf_sha256")),
                    "src_ok": sum(1 for r in rows if r.get("local_src_sha256")),
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
