#!/usr/bin/env python3
"""
fetch_cassini_pds_sce1_odf.py

Cassini Solar Conjunction Experiment (SCE1) の一次データ（PDS3）から、
ODF（Orbit Data File）と対応する .LBL、そして volume の index を取得して
`data/cassini/pds_sce1/` にキャッシュする。

目的:
  - デジタイズではなく一次ソース（PDS）から Cassini 検証を行う
  - 2回目以降はオフライン再現できるように、取得物を data/ に固定保存する

既定の取得対象:
  - 2002年 DOY 157-186（SCE1）を 4日ごとの volume（CORS_0021..0028）に分割したアーカイブ

出力（固定）:
  - data/cassini/pds_sce1/cors_00xx/index/index.tab, index.lbl
  - data/cassini/pds_sce1/cors_00xx/sce1_ddd/odf/*.odf, *.lbl
  - data/cassini/pds_sce1/manifest_odf.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


DEFAULT_BASE_URL = "https://atmos.nmsu.edu/pdsd/archive/data/co-ss-rss-1-sce1-v10"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dst: Path, *, force: bool, timeout_s: int = 180) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        return
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "waveP-fetch-cassini-pds-sce1/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1024 * 1024)
    tmp.replace(dst)


def _iter_lines(path: Path) -> Iterable[str]:
    # PDS index files are typically CRLF; keep robust.
    txt = path.read_text(encoding="utf-8", errors="replace")
    for ln in txt.splitlines():
        ln = ln.strip()
        if ln:
            yield ln


def _cors_for_doy(doy: int) -> int:
    # SCE1 2002 DOY 157-186 are split into 4-day volumes:
    # 157-160 => 0021, 161-164 => 0022, ... 185-186 => 0028
    if doy < 157 or doy > 186:
        raise ValueError(f"DOY out of supported range for SCE1 (157-186): {doy}")
    return 21 + ((doy - 157) // 4)


@dataclass(frozen=True)
class IndexEntry:
    cors: int
    label_rel: str
    data_rel: str


def _parse_index_tab(index_tab: Path, cors: int) -> List[IndexEntry]:
    out: List[IndexEntry] = []
    with index_tab.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 3:
                continue
            label_rel = str(row[1]).strip()
            data_name = str(row[2]).strip()
            if not label_rel or not data_name:
                continue
            if "/" not in label_rel:
                continue
            data_rel = label_rel.rsplit("/", 1)[0] + "/" + data_name
            out.append(IndexEntry(cors=cors, label_rel=label_rel, data_rel=data_rel))
    return out


def _select_odf(entries: Sequence[IndexEntry], *, doy_start: int, doy_stop: int) -> List[IndexEntry]:
    picked: List[IndexEntry] = []
    pat = re.compile(r"^SCE1_(\d{3})/ODF/", re.IGNORECASE)
    for e in entries:
        m = pat.match(e.label_rel)
        if not m:
            continue
        doy = int(m.group(1))
        if doy_start <= doy <= doy_stop:
            picked.append(e)
    return picked


def main() -> int:
    root = _repo_root()
    out_root = root / "data" / "cassini" / "pds_sce1"
    out_root.mkdir(parents=True, exist_ok=True)

    ap = argparse.ArgumentParser(description="Fetch Cassini SCE1 (PDS) ODF + labels into data/ for offline use.")
    ap.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL, help="Base URL for the PDS mirror.")
    ap.add_argument("--doy-start", type=int, default=162, help="Start DOY (inclusive). Default: 162")
    ap.add_argument("--doy-stop", type=int, default=182, help="Stop DOY (inclusive). Default: 182")
    ap.add_argument("--offline", action="store_true", help="Do not use network; require cached files.")
    ap.add_argument("--force", action="store_true", help="Re-download cached files.")
    args = ap.parse_args()

    doy_start = int(args.doy_start)
    doy_stop = int(args.doy_stop)
    if doy_stop < doy_start:
        doy_start, doy_stop = doy_stop, doy_start

    # Determine needed volumes
    cors_set = sorted({_cors_for_doy(d) for d in range(doy_start, doy_stop + 1)})
    base_url = str(args.base_url).rstrip("/")

    # Download index files per volume (or validate cache in offline mode)
    all_entries: List[IndexEntry] = []
    for cors in cors_set:
        cors_dir = out_root / f"cors_{cors:04d}"
        idx_dir = cors_dir / "index"
        idx_tab = idx_dir / "index.tab"
        idx_lbl = idx_dir / "index.lbl"
        idx_tab_url = f"{base_url}/cors_{cors:04d}/index/index.tab"
        idx_lbl_url = f"{base_url}/cors_{cors:04d}/index/index.lbl"

        if args.offline:
            if not idx_tab.exists():
                print(f"[err] offline and missing: {idx_tab}")
                return 2
            if not idx_lbl.exists():
                print(f"[warn] offline and missing: {idx_lbl} (continuing)")
        else:
            _download(idx_tab_url, idx_tab, force=bool(args.force), timeout_s=120)
            _download(idx_lbl_url, idx_lbl, force=bool(args.force), timeout_s=120)

        all_entries.extend(_parse_index_tab(idx_tab, cors=cors))

    odf_entries = _select_odf(all_entries, doy_start=doy_start, doy_stop=doy_stop)
    if not odf_entries:
        print("[err] no ODF entries found in index.tab for the requested DOY range")
        return 2

    downloaded: List[dict] = []
    missing: List[str] = []

    for e in odf_entries:
        cors_dir = out_root / f"cors_{e.cors:04d}"
        # Mirror server is case-sensitive; store paths in lowercase as served.
        label_rel = e.label_rel.strip().replace("\\", "/").lower()
        data_rel = e.data_rel.strip().replace("\\", "/").lower()
        label_url = f"{base_url}/cors_{e.cors:04d}/{label_rel}"
        data_url = f"{base_url}/cors_{e.cors:04d}/{data_rel}"

        label_dst = cors_dir / Path(label_rel)
        data_dst = cors_dir / Path(data_rel)

        if args.offline:
            if not label_dst.exists():
                missing.append(str(label_dst))
            if not data_dst.exists():
                missing.append(str(data_dst))
            continue

        _download(label_url, label_dst, force=bool(args.force), timeout_s=180)
        _download(data_url, data_dst, force=bool(args.force), timeout_s=180)

        downloaded.append(
            {
                "cors": e.cors,
                "label_rel": label_rel,
                "data_rel": data_rel,
                "label_url": label_url,
                "data_url": data_url,
                "label_bytes": int(label_dst.stat().st_size) if label_dst.exists() else None,
                "data_bytes": int(data_dst.stat().st_size) if data_dst.exists() else None,
                "label_sha256": _sha256(label_dst) if label_dst.exists() else None,
                "data_sha256": _sha256(data_dst) if data_dst.exists() else None,
            }
        )

    if args.offline and missing:
        print("[err] offline and missing cached files:")
        for p in missing[:60]:
            print("  -", p)
        if len(missing) > 60:
            print(f"  ... and {len(missing)-60} more")
        return 2

    manifest = {
        "source": "PDS3 mirror (Cassini SCE1): CO-SS-RSS-1-SCE1-V1.0",
        "base_url": base_url,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "doy_range": {"start": doy_start, "stop": doy_stop},
        "cors_volumes": cors_set,
        "files": downloaded if not args.offline else None,
        "note": "Files are stored in lowercase to match the mirror's case-sensitive paths.",
    }
    man_path = out_root / "manifest_odf.json"
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.offline:
        print(f"[ok] offline cache verified: {man_path}")
    else:
        print(f"[ok] wrote: {man_path} (files={len(downloaded)})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

