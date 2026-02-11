#!/usr/bin/env python3
"""
fetch_station_edc.py

DGFI-TUM EDC の slrlog から、LLR/SLR 観測局の site log を取得して
data/llr/stations/ にキャッシュする。

用途:
  - LLRモデル（観測局の自転を含む幾何）を構築するための局座標/緯度経度を取得
  - 2回目以降はオフライン再現（キャッシュ利用）

出力（固定）
  - data/llr/stations/<station>.json
  - data/llr/stations/<station>_<yyyymmdd>.log
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


BASE = "https://edc.dgfi.tum.de"
LIST_URL = f"{BASE}/pub/slr/slrlog/"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _fetch_text(url: str, *, timeout_s: int = 60) -> str:
    with urllib.request.urlopen(url, timeout=timeout_s) as r:
        b = r.read()
    return b.decode("utf-8", "replace")


def _iter_hrefs(html: str) -> list[str]:
    hrefs = re.findall(r'href=[\'"]([^\'"]+)[\'"]', html)
    return [h for h in hrefs if h]


def _detect_station_from_primary(root: Path) -> Optional[str]:
    data_dir = root / "data" / "llr"
    candidates = [
        data_dir / "llr_primary.np2",
        data_dir / "llr_primary.npt",
        data_dir / "llr_primary.crd",
        data_dir / "llr_primary.np2.gz",
        data_dir / "llr_primary.npt.gz",
        data_dir / "llr_primary.crd.gz",
        data_dir / "demo_llr_like.crd",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            txt = _read_text(p)
        except Exception:
            continue
        for line in txt.splitlines():
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("h2"):
                toks = line.split()
                if len(toks) >= 2:
                    return toks[1].strip()
    return None


@dataclass(frozen=True)
class PickedLog:
    station: str
    yyyymmdd: str
    url: str


def _pick_latest_log(station: str) -> PickedLog:
    station_l = station.lower()
    html = _fetch_text(LIST_URL, timeout_s=60)
    hrefs = _iter_hrefs(html)

    pat = re.compile(rf"^/pub/slr/slrlog/{re.escape(station_l)}_(\d{{8}})\.log$", re.IGNORECASE)
    hits: list[tuple[str, str]] = []
    for h in hrefs:
        m = pat.match(h)
        if not m:
            continue
        hits.append((m.group(1), h))

    if not hits:
        raise FileNotFoundError(f"No site log found for station={station} at {LIST_URL}")

    yyyymmdd, href = max(hits, key=lambda t: t[0])
    return PickedLog(station=station.upper(), yyyymmdd=yyyymmdd, url=f"{BASE}{href}")


def _download(url: str, dst: Path, *, force: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        print(f"[skip] exists: {dst}")
        return
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"[dl] {url}")
    with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1024 * 1024)
    tmp.replace(dst)
    print(f"[ok] saved: {dst} ({dst.stat().st_size} bytes)")


def _parse_coords(txt: str) -> dict:
    # Latitude            [deg]: 32.780361 N
    # Longitude           [deg]: 105.820417 W
    # Elevation             [m]: 2788
    x_m = re.search(r"X coordinate\s*\[m\]\s*:\s*([0-9.+-]+)", txt, re.IGNORECASE)
    y_m = re.search(r"Y coordinate\s*\[m\]\s*:\s*([0-9.+-]+)", txt, re.IGNORECASE)
    z_m = re.search(r"Z coordinate\s*\[m\]\s*:\s*([0-9.+-]+)", txt, re.IGNORECASE)
    lat_m = re.search(r"Latitude\s*\[deg\]\s*:\s*([0-9.]+)\s*([NS])", txt, re.IGNORECASE)
    lon_m = re.search(r"Longitude\s*\[deg\]\s*:\s*([0-9.]+)\s*([EW])", txt, re.IGNORECASE)
    ele_m = re.search(r"Elevation\s*\[m\]\s*:\s*([0-9.]+)", txt, re.IGNORECASE)
    if not (lat_m and lon_m and ele_m):
        raise ValueError("Failed to parse Latitude/Longitude/Elevation from site log.")

    lat = float(lat_m.group(1))
    if lat_m.group(2).upper() == "S":
        lat = -lat

    lon = float(lon_m.group(1))
    if lon_m.group(2).upper() == "W":
        lon = -lon

    elev = float(ele_m.group(1))

    pad_m = re.search(r"CDP Pad ID\s*:\s*(\d+)", txt, re.IGNORECASE)
    pad_id = int(pad_m.group(1)) if pad_m else None

    date_m = re.search(r"Date Prepared\s*:\s*(\d{4}-\d{2}-\d{2})", txt, re.IGNORECASE)
    date_prepared = date_m.group(1) if date_m else None

    name_m = re.search(r"Site Name\s*:\s*([A-Za-z0-9 _\\-]+)", txt)
    site_name = name_m.group(1).strip() if name_m else None

    return {
        "site_name": site_name,
        "cdp_pad_id": pad_id,
        "date_prepared": date_prepared,
        "x_m": float(x_m.group(1)) if x_m else None,
        "y_m": float(y_m.group(1)) if y_m else None,
        "z_m": float(z_m.group(1)) if z_m else None,
        "lat_deg": lat,
        "lon_deg": lon,
        "height_m": elev,
    }


def main() -> int:
    root = _repo_root()
    out_dir = root / "data" / "llr" / "stations"
    out_dir.mkdir(parents=True, exist_ok=True)

    ap = argparse.ArgumentParser(description="Fetch ILRS site log from DGFI-TUM EDC (slrlog) and cache it.")
    ap.add_argument("--station", type=str, default="", help="4-char station code (e.g., APOL). If omitted, auto-detect from data/llr/llr_primary.*")
    ap.add_argument("--offline", action="store_true", help="Do not use network; only succeed if station json exists.")
    ap.add_argument("--force", action="store_true", help="Re-download and overwrite cached files.")
    args = ap.parse_args()

    station = (args.station or "").strip().upper()
    if not station:
        station = _detect_station_from_primary(root) or ""
        station = station.strip().upper()

    if not station:
        print("[err] station not specified and could not be detected from data/llr/llr_primary.*")
        return 2

    json_path = out_dir / f"{station.lower()}.json"
    if args.offline:
        if json_path.exists():
            print(f"[ok] offline: {json_path}")
            return 0
        print(f"[err] offline and missing: {json_path}")
        return 2

    if json_path.exists() and not args.force:
        print(f"[skip] exists: {json_path} (use --force to refresh)")
        return 0

    picked = _pick_latest_log(station)
    log_path = out_dir / f"{station.lower()}_{picked.yyyymmdd}.log"
    _download(picked.url, log_path, force=bool(args.force))

    meta = _parse_coords(_read_text(log_path))
    meta_out = {
        "station": station,
        "source": "DGFI-TUM EDC slrlog",
        "log_url": picked.url,
        "log_filename": log_path.name,
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        **meta,
    }
    json_path.write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] json: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
