#!/usr/bin/env python3
"""
fetch_cassini_pds_sce1_tdf.py

Cassini SCE1 (CO-SS-RSS-1-SCE1-V1.0) の一次データ（PDS3）から、
TDF/ATDF（Tracking Data File）と対応する .LBL を取得して
`data/cassini/pds_sce1/` にキャッシュする。

注意:
  - TDF は 1日あたり数十MBと大きい。必要なDOY範囲に絞って取得すること。
  - 2回目以降はオフライン再現できるよう、取得物は data/ に固定保存する。
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
from typing import List, Optional, Sequence


DEFAULT_BASE_URL = "https://atmos.nmsu.edu/pdsd/archive/data/co-ss-rss-1-sce1-v10"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dst: Path, *, force: bool, timeout_s: int = 300) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        return
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "waveP-fetch-cassini-pds-sce1/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1024 * 1024)
    tmp.replace(dst)


def _cors_for_doy(doy: int) -> int:
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


def _select_tdf(entries: Sequence[IndexEntry], *, doy_start: int, doy_stop: int, band: str) -> List[IndexEntry]:
    picked: List[IndexEntry] = []
    pat = re.compile(r"^SCE1_(\d{3})/TDF/", re.IGNORECASE)
    band = band.lower().strip()
    for e in entries:
        m = pat.match(e.label_rel)
        if not m:
            continue
        doy = int(m.group(1))
        if not (doy_start <= doy <= doy_stop):
            continue

        name = e.data_rel.lower()
        if band == "ka":
            if "k252v0" not in name:
                continue
        elif band == "x":
            if "xmmmv0" not in name:
                continue
        elif band == "both":
            pass
        else:
            raise ValueError(f"Unknown band filter: {band} (expected ka/x/both)")

        picked.append(e)
    return picked


def main() -> int:
    root = _repo_root()
    out_root = root / "data" / "cassini" / "pds_sce1"
    out_root.mkdir(parents=True, exist_ok=True)

    ap = argparse.ArgumentParser(description="Fetch Cassini SCE1 (PDS) TDF + labels into data/ for offline use.")
    ap.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL, help="Base URL for the PDS mirror.")
    ap.add_argument("--doy-start", type=int, default=162, help="Start DOY (inclusive). Default: 162")
    ap.add_argument("--doy-stop", type=int, default=182, help="Stop DOY (inclusive). Default: 182")
    ap.add_argument("--band", choices=["ka", "x", "both"], default="ka", help="Which TDF band to fetch.")
    ap.add_argument("--offline", action="store_true", help="Do not use network; require cached files.")
    ap.add_argument("--force", action="store_true", help="Re-download cached files.")
    args = ap.parse_args()

    doy_start = int(args.doy_start)
    doy_stop = int(args.doy_stop)
    if doy_stop < doy_start:
        doy_start, doy_stop = doy_stop, doy_start

    cors_set = sorted({_cors_for_doy(d) for d in range(doy_start, doy_stop + 1)})
    base_url = str(args.base_url).rstrip("/")

    all_entries: List[IndexEntry] = []
    for cors in cors_set:
        idx_tab = out_root / f"cors_{cors:04d}" / "index" / "index.tab"
        if not idx_tab.exists():
            print(f"[err] missing index cache: {idx_tab}")
            print("      run scripts/cassini/fetch_cassini_pds_sce1_odf.py first (it also caches index.tab).")
            return 2
        all_entries.extend(_parse_index_tab(idx_tab, cors=cors))

    tdf_entries = _select_tdf(all_entries, doy_start=doy_start, doy_stop=doy_stop, band=str(args.band))
    if not tdf_entries:
        print("[err] no TDF entries found for requested range/band")
        return 2

    downloaded: List[dict] = []
    missing: List[str] = []

    for e in tdf_entries:
        cors_dir = out_root / f"cors_{e.cors:04d}"
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
        _download(data_url, data_dst, force=bool(args.force), timeout_s=600)

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
        "band": str(args.band),
        "cors_volumes": cors_set,
        "files": downloaded if not args.offline else None,
        "note": "TDF files are stored in lowercase to match the mirror's case-sensitive paths.",
    }
    man_path = out_root / "manifest_tdf.json"
    man_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.offline:
        print(f"[ok] offline cache verified: {man_path}")
    else:
        print(f"[ok] wrote: {man_path} (files={len(downloaded)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

