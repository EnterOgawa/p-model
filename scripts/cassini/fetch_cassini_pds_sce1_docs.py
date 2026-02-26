#!/usr/bin/env python3
"""
fetch_cassini_pds_sce1_docs.py

Cassini SCE1 (CO-SS-RSS-1-SCE1-V1.0) の一次ソース（PDS3）から、
volume 直下の `aareadme.txt` / `errata.txt` と `document/` 配下の資料を取得して
`data/cassini/pds_sce1/` にキャッシュする。

目的:
  - 観測量の定義/単位/符号などを一次ソースで確定し、解析の前提を固定する。
  - 2回目以降はオフラインで再現できるよう、取得物を data/ に固定保存する。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog

DEFAULT_BASE_URL = "https://atmos.nmsu.edu/pdsd/archive/data/co-ss-rss-1-sce1-v10"


# 関数: `_repo_root` の入出力契約と処理意図を定義する。
def _repo_root() -> Path:
    return _ROOT


# 関数: `_sha256` の入出力契約と処理意図を定義する。

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


# 関数: `_download` の入出力契約と処理意図を定義する。

def _download(url: str, dst: Path, *, force: bool, timeout_s: int = 180) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    # 条件分岐: `dst.exists() and not force` を満たす経路を評価する。
    if dst.exists() and not force:
        return

    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "waveP-fetch-cassini-pds-sce1/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1024 * 1024)

    tmp.replace(dst)


# 関数: `_cors_for_doy` の入出力契約と処理意図を定義する。

def _cors_for_doy(doy: int) -> int:
    # SCE1 2002 DOY 157-186 are split into 4-day volumes:
    # 157-160 => 0021, 161-164 => 0022, ... 185-186 => 0028
    if doy < 157 or doy > 186:
        raise ValueError(f"DOY out of supported range for SCE1 (157-186): {doy}")

    return 21 + ((doy - 157) // 4)


# 関数: `_extract_dir_links` の入出力契約と処理意図を定義する。

def _extract_dir_links(html: str, *, base_url: str) -> List[str]:
    # Apache autoindex: <a href="name">name</a>
    links = re.findall(r'href="([^"]+)"', html, flags=re.IGNORECASE)
    out: List[str] = []
    for href in links:
        href = href.strip()
        # 条件分岐: `not href or href.startswith("?")` を満たす経路を評価する。
        if not href or href.startswith("?"):
            continue

        # 条件分岐: `href in ("../", "./")` を満たす経路を評価する。

        if href in ("../", "./"):
            continue

        # 条件分岐: `href.startswith("http://") or href.startswith("https://")` を満たす経路を評価する。

        if href.startswith("http://") or href.startswith("https://"):
            # External assets (PDS app bar, etc.) are irrelevant.
            continue

        # 条件分岐: `href.endswith("/")` を満たす経路を評価する。

        if href.endswith("/"):
            continue
        # Avoid query fragments etc.

        href = href.split("#", 1)[0].split("?", 1)[0]
        # 条件分岐: `not href` を満たす経路を評価する。
        if not href:
            continue

        out.append(urllib.parse.urljoin(base_url, href))
    # Deduplicate, stable order

    seen = set()
    uniq: List[str] = []
    for u in out:
        # 条件分岐: `u in seen` を満たす経路を評価する。
        if u in seen:
            continue

        seen.add(u)
        uniq.append(u)

    return uniq


# クラス: `DownloadedFile` の責務と境界条件を定義する。

@dataclass(frozen=True)
class DownloadedFile:
    cors: int
    rel_path: str
    url: str
    size_bytes: int
    sha256: str


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    root = _repo_root()
    out_root = root / "data" / "cassini" / "pds_sce1"
    out_root.mkdir(parents=True, exist_ok=True)

    ap = argparse.ArgumentParser(description="Fetch Cassini SCE1 docs (aareadme/errata/document/) into data/ for offline use.")
    ap.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL, help="Base URL for the PDS mirror.")
    ap.add_argument("--doy-start", type=int, default=162, help="Start DOY (inclusive). Default: 162")
    ap.add_argument("--doy-stop", type=int, default=182, help="Stop DOY (inclusive). Default: 182")
    ap.add_argument("--offline", action="store_true", help="Do not use network; require cached files.")
    ap.add_argument("--force", action="store_true", help="Re-download cached files.")
    args = ap.parse_args()

    doy_start = int(args.doy_start)
    doy_stop = int(args.doy_stop)
    # 条件分岐: `doy_stop < doy_start` を満たす経路を評価する。
    if doy_stop < doy_start:
        doy_start, doy_stop = doy_stop, doy_start

    cors_set = sorted({_cors_for_doy(d) for d in range(doy_start, doy_stop + 1)})
    base_url = str(args.base_url).rstrip("/")

    downloaded: List[DownloadedFile] = []
    missing: List[str] = []

    for cors in cors_set:
        cors_dir = out_root / f"cors_{cors:04d}"

        # volume root text files
        for name in ("aareadme.txt", "errata.txt"):
            rel = name
            url = f"{base_url}/cors_{cors:04d}/{name}"
            dst = cors_dir / rel
            # 条件分岐: `args.offline` を満たす経路を評価する。
            if args.offline:
                # 条件分岐: `not dst.exists()` を満たす経路を評価する。
                if not dst.exists():
                    missing.append(str(dst))
            else:
                _download(url, dst, force=bool(args.force), timeout_s=180)
                downloaded.append(
                    DownloadedFile(
                        cors=cors,
                        rel_path=str(Path(rel).as_posix()),
                        url=url,
                        size_bytes=int(dst.stat().st_size),
                        sha256=_sha256(dst),
                    )
                )

        # document/ directory listing

        doc_dir_url = f"{base_url}/cors_{cors:04d}/document/"
        doc_dir_dst = cors_dir / "document"
        # 条件分岐: `args.offline` を満たす経路を評価する。
        if args.offline:
            # 条件分岐: `not doc_dir_dst.exists()` を満たす経路を評価する。
            if not doc_dir_dst.exists():
                missing.append(str(doc_dir_dst))
            # do not attempt to validate each file here; they are small and optional per-volume.

            continue

        try:
            req = urllib.request.Request(doc_dir_url, headers={"User-Agent": "waveP-fetch-cassini-pds-sce1/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                html = r.read().decode("utf-8", errors="ignore")
        except Exception as e:
            print(f"[warn] failed to list document/ for cors_{cors:04d}: {e}")
            continue

        urls = _extract_dir_links(html, base_url=doc_dir_url)
        for file_url in urls:
            filename = file_url.rsplit("/", 1)[-1]
            # 条件分岐: `not filename` を満たす経路を評価する。
            if not filename:
                continue

            rel = Path("document") / filename
            dst = cors_dir / rel
            _download(file_url, dst, force=bool(args.force), timeout_s=180)
            downloaded.append(
                DownloadedFile(
                    cors=cors,
                    rel_path=str(rel.as_posix()),
                    url=file_url,
                    size_bytes=int(dst.stat().st_size),
                    sha256=_sha256(dst),
                )
            )

    # 条件分岐: `args.offline and missing` を満たす経路を評価する。

    if args.offline and missing:
        print("[err] offline and missing cached paths:")
        for p in missing[:80]:
            print("  -", p)

        # 条件分岐: `len(missing) > 80` を満たす経路を評価する。

        if len(missing) > 80:
            print(f"  ... and {len(missing)-80} more")

        return 2

    manifest = {
        "source": "PDS3 mirror (Cassini SCE1): CO-SS-RSS-1-SCE1-V1.0",
        "base_url": base_url,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "doy_range": {"start": doy_start, "stop": doy_stop},
        "cors_volumes": cors_set,
        "files": [df.__dict__ for df in downloaded] if not args.offline else None,
        "note": "Docs are stored per-volume to match the mirror's structure.",
    }
    man_path = out_root / "manifest_docs.json"
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # 条件分岐: `args.offline` を満たす経路を評価する。
    if args.offline:
        print(f"[ok] offline cache verified: {man_path}")
    else:
        print(f"[ok] wrote: {man_path} (files={len(downloaded)})")

    try:
        worklog.append_event(
            {
                "event_type": "cassini_fetch_docs",
                "argv": list(getattr(__import__('sys'), 'argv', [])),
                "inputs": {"base_url": base_url, "doy_start": doy_start, "doy_stop": doy_stop},
                "outputs": {"manifest": man_path},
                "counts": {"files": len(downloaded), "cors_volumes": len(cors_set)},
            }
        )
    except Exception:
        pass

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
