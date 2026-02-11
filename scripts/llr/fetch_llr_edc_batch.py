#!/usr/bin/env python3
"""
fetch_llr_edc_batch.py

DGFI-TUM EDC (npt_crd_v2) から LLR（主に月面反射器：Apollo11/14/15, Luna17/21）の
月次（YYYYMM）Normal Point（.np2）をまとめて取得し、観測局コード（H2）を抽出して、
必要な site log（slrlog）も同時にキャッシュする。

狙い:
  - 「複数反射器 × 複数局」を一気に揃え、後続のバッチ検証を再現可能にする
  - 2回目以降は data/ 配下のキャッシュでオフライン再現する

出力（固定）
  - data/llr/edc/<target>/<year>/<filename>.np2
  - data/llr/llr_edc_batch_manifest.json
  - data/llr/stations/<station>.json
  - data/llr/stations/<station>_<yyyymmdd>.log
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


_DEFAULT_EDC_BASES = ["ftp://edc.dgfi.tum.de", "https://edc.dgfi.tum.de"]

NPT_ROOT_PATH_V1 = "/pub/slr/data/npt_crd"
NPT_ROOT_PATH_V2 = "/pub/slr/data/npt_crd_v2"
SLRLOG_ROOT_PATH = "/pub/slr/slrlog/"


DEFAULT_TARGETS = ["apollo11", "apollo14", "apollo15", "luna17", "luna21", "nglr1"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _edc_bases() -> list[str]:
    # Allow overriding endpoints (comma-separated), while defaulting to ftp-first
    # because https occasionally returns 503.
    raw = os.environ.get("EDC_BASES", "").strip()
    if raw:
        bases = [b.strip().rstrip("/") for b in raw.split(",") if b.strip()]
        return bases or _DEFAULT_EDC_BASES
    return _DEFAULT_EDC_BASES


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _fetch_text(url: str, *, timeout_s: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "waveP-fetch-llr-edc-batch/1.0"})
    last_err: Optional[Exception] = None
    for attempt in range(1, 8):
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as r:
                b = r.read()
            return b.decode("utf-8", "replace")
        except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
            last_err = e
            # Retry transient server/network errors (EDC occasionally returns 503).
            if int(getattr(e, "code", 0) or 0) not in (429, 500, 502, 503, 504):
                raise
        except urllib.error.URLError as e:  # type: ignore[attr-defined]
            last_err = e

        sleep_s = min(60.0, (2 ** (attempt - 1)) + random.random())
        print(f"[warn] fetch failed (attempt {attempt}/7): {url} → retry in {sleep_s:.1f}s")
        time.sleep(sleep_s)

    raise RuntimeError(f"fetch failed after retries: {url}: {last_err}")


def _fetch_text_any(path: str, *, timeout_s: int = 60) -> tuple[str, str]:
    last_err: Optional[Exception] = None
    for base in _edc_bases():
        url = f"{base}{path}"
        try:
            return _fetch_text(url, timeout_s=timeout_s), url
        except Exception as e:
            last_err = e
            print(f"[warn] fetch failed: {url}: {e}")
    raise RuntimeError(f"fetch failed for all endpoints: {path}: {last_err}")


def _iter_hrefs(html: str) -> Iterable[str]:
    for href in re.findall(r'href=[\'"]([^\'"]+)[\'"]', html):
        if href:
            yield href


def _iter_dir_names(listing: str) -> Iterable[str]:
    # Handle both Apache-style HTML listings and FTP LIST output.
    if "href=" in listing.lower():
        for href in _iter_hrefs(listing):
            name = href.rstrip("/").rsplit("/", 1)[-1].strip()
            if name:
                yield name
        return

    for raw in listing.splitlines():
        line = raw.strip()
        if not line:
            continue
        # Example:
        # drwxr-xr-x    2 9000     9000         4096 Jun 27  2025 2022
        # -rw-rw-r--    1 9000     121          7858 Apr 02  2015 apollo11_201205.npt
        # lrwxrwxrwx    1 ... name -> target
        if " -> " in line:
            line = line.split(" -> ", 1)[0].rstrip()
        toks = line.split()
        if not toks:
            continue
        name = toks[-1].strip()
        if name in (".", ".."):
            continue
        yield name


def _list_years(target: str, root_path: str) -> list[int]:
    listing, _url = _fetch_text_any(f"{root_path}/{target}/")
    years: list[int] = []
    for name in _iter_dir_names(listing):
        if re.fullmatch(r"\d{4}", name):
            years.append(int(name))
    return sorted(set(years))


def _list_files(target: str, year: int, root_path: str) -> list[str]:
    listing, _url = _fetch_text_any(f"{root_path}/{target}/{year}/")
    files: list[str] = []
    for name in _iter_dir_names(listing):
        if not name.lower().startswith(f"{target.lower()}_"):
            continue
        files.append(name)
    return sorted(set(files))


def _is_monthly(filename: str) -> bool:
    # e.g. apollo11_202212.np2 (YYYYMM) / apollo11_201205.npt
    return bool(re.search(r"_(\d{6})\.(np2|npt)$", filename, re.IGNORECASE))


def _is_daily(filename: str) -> bool:
    return bool(re.search(r"_(\d{8})\.(np2|npt)$", filename, re.IGNORECASE))


def _download_any(path: str, dst: Path, *, force: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        print(f"[skip] exists: {dst}")
        return ""
    tmp = dst.with_suffix(dst.suffix + ".part")
    last_err: Optional[Exception] = None
    used_url: Optional[str] = None

    for base in _edc_bases():
        url = f"{base}{path}"
        print(f"[dl] {url}")
        req = urllib.request.Request(url, headers={"User-Agent": "waveP-fetch-llr-edc-batch/1.0"})
        last_err = None
        for attempt in range(1, 8):
            try:
                with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
                    shutil.copyfileobj(r, f, length=1024 * 1024)
                used_url = url
                break
            except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
                last_err = e
                if int(getattr(e, "code", 0) or 0) not in (429, 500, 502, 503, 504):
                    raise
            except urllib.error.URLError as e:  # type: ignore[attr-defined]
                last_err = e

            sleep_s = min(60.0, (2 ** (attempt - 1)) + random.random())
            print(f"[warn] download failed (attempt {attempt}/7): {url} → retry in {sleep_s:.1f}s")
            time.sleep(sleep_s)

        if used_url:
            break

        print(f"[warn] download failed for endpoint: {url}: {last_err}")

    if not used_url:
        raise RuntimeError(f"download failed for all endpoints: {path}: {last_err}")

    tmp.replace(dst)
    print(f"[ok] saved: {dst} ({dst.stat().st_size} bytes)")
    return used_url


def _extract_station_codes(np2_path: Path) -> list[str]:
    stations: set[str] = set()
    try:
        txt = np2_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return []
    for raw in txt.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.lower().startswith("h2"):
            toks = line.split()
            if len(toks) >= 2:
                st = toks[1].strip()
                if st and st.lower() not in ("na", "nan"):
                    stations.add(st.upper())
    return sorted(stations)


# -----------------------
# Station (slrlog) cache
# -----------------------


@dataclass(frozen=True)
class PickedLog:
    station: str
    yyyymmdd: str
    url: str


def _pick_latest_log(station: str) -> PickedLog:
    station_l = station.lower()
    listing, _url = _fetch_text_any(SLRLOG_ROOT_PATH, timeout_s=60)
    hrefs = list(_iter_dir_names(listing))
    pat = re.compile(rf"^/pub/slr/slrlog/{re.escape(station_l)}_(\d{{8}})\.log$", re.IGNORECASE)
    hits: list[tuple[str, str]] = []
    for h in hrefs:
        m = pat.match(f"{SLRLOG_ROOT_PATH}{h.lstrip('/')}" if not h.startswith("/pub/") else h)
        if m:
            hits.append((m.group(1), h))
    if not hits:
        raise FileNotFoundError(f"No site log found for station={station} at {SLRLOG_ROOT_PATH}")
    yyyymmdd, href = max(hits, key=lambda t: t[0])
    # href may already include the full path; normalize to /pub/...
    if not href.startswith("/"):
        href = f"{SLRLOG_ROOT_PATH}{href}"
    return PickedLog(station=station.upper(), yyyymmdd=yyyymmdd, url=f"{_edc_bases()[0]}{href}")


def _parse_coords_from_site_log(txt: str) -> dict:
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


def _ensure_station_cached(root: Path, station: str, *, force: bool) -> None:
    out_dir = root / "data" / "llr" / "stations"
    out_dir.mkdir(parents=True, exist_ok=True)

    station = station.strip().upper()
    if not station:
        return

    json_path = out_dir / f"{station.lower()}.json"
    if json_path.exists() and not force:
        return

    picked = _pick_latest_log(station)
    log_path = out_dir / f"{station.lower()}_{picked.yyyymmdd}.log"
    # picked.url includes protocol+host; convert into a path for fallback handling
    path = re.sub(r"^https?://[^/]+", "", picked.url, flags=re.IGNORECASE)
    path = re.sub(r"^ftp://[^/]+", "", path, flags=re.IGNORECASE)
    _download_any(path if path.startswith("/") else f"/{path}", log_path, force=force)
    txt = log_path.read_text(encoding="utf-8", errors="replace")
    meta = _parse_coords_from_site_log(txt)

    meta_out = {
        "station": station,
        "source": "DGFI-TUM EDC slrlog",
        "log_url": picked.url,
        "log_filename": log_path.name,
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        **meta,
    }
    json_path.write_text(json.dumps(meta_out, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_years_arg(s: str) -> tuple[Optional[int], Optional[int]]:
    if not s:
        return None, None
    m = re.fullmatch(r"(\d{4})(?:-(\d{4}))?", s.strip())
    if not m:
        raise ValueError(f"Invalid --years: {s} (expected YYYY or YYYY-YYYY)")
    y0 = int(m.group(1))
    y1 = int(m.group(2)) if m.group(2) else y0
    return min(y0, y1), max(y0, y1)


def main() -> int:
    root = _repo_root()
    data_dir = root / "data" / "llr"
    edc_dir = data_dir / "edc"
    manifest_path = data_dir / "llr_edc_batch_manifest.json"

    ap = argparse.ArgumentParser(description="Fetch LLR CRD normal points from EDC and cache station site logs.")
    ap.add_argument(
        "--targets",
        type=str,
        default=",".join(DEFAULT_TARGETS),
        help=f"Comma-separated target directories under EDC (default: {','.join(DEFAULT_TARGETS)})",
    )
    ap.add_argument(
        "--years",
        type=str,
        default="",
        help="Optional year range filter: YYYY or YYYY-YYYY (default: all available years per target)",
    )
    ap.add_argument("--offline", action="store_true", help="Do not use network; require existing manifest + files.")
    ap.add_argument("--force", action="store_true", help="Re-download and overwrite cached files.")
    ap.add_argument(
        "--append",
        action="store_true",
        help="Append new targets/files into an existing manifest (avoids full rescan). Ignored with --offline/--force.",
    )
    ap.add_argument("--include-daily", action="store_true", help="Also download YYYYMMDD daily files (larger set).")
    args = ap.parse_args()

    targets = [t.strip() for t in str(args.targets).split(",") if t.strip()]
    if not targets:
        print("[err] empty targets")
        return 2

    if args.offline:
        if not manifest_path.exists():
            print(f"[err] offline and missing manifest: {manifest_path}")
            return 2
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[err] failed to read manifest: {e}")
            return 2

        missing: list[str] = []
        for rec in manifest.get("files", []):
            rel = rec.get("cached_path")
            if not rel:
                continue
            p = root / Path(str(rel))
            if not p.exists():
                missing.append(str(p))
        if missing:
            print("[err] offline and missing cached files:")
            for p in missing[:50]:
                print("  -", p)
            if len(missing) > 50:
                print(f"  ... and {len(missing)-50} more")
            return 2
        print(f"[ok] offline: {manifest_path}")
        return 0

    y0, y1 = _parse_years_arg(str(args.years))

    all_files: list[dict] = []
    all_stations: set[str] = set()
    existing_keys: set[str] = set()
    base_targets: list[str] = []
    prev_year_filter = None
    prev_include_daily = None

    if bool(args.append) and manifest_path.exists() and not args.force:
        try:
            prev = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[warn] failed to read existing manifest for --append: {e} (fall back to full scan)")
            prev = None

        if isinstance(prev, dict):
            prev_files = prev.get("files") if isinstance(prev.get("files"), list) else []
            all_files = [r for r in prev_files if isinstance(r, dict)]
            prev_year_filter = prev.get("year_filter")
            prev_include_daily = prev.get("include_daily")
            for r in all_files:
                k = str(r.get("edc_path") or "")
                if not k:
                    t = str(r.get("target") or "")
                    y = str(r.get("year") or "")
                    fn = str(r.get("filename") or "")
                    if t and y and fn:
                        k = f"{t}/{y}/{fn}"
                if k:
                    existing_keys.add(k)
                for st in (r.get("stations") or []) if isinstance(r.get("stations"), list) else []:
                    if isinstance(st, str) and st.strip():
                        all_stations.add(st.strip().upper())

            for st in (prev.get("stations_detected") or []) if isinstance(prev.get("stations_detected"), list) else []:
                if isinstance(st, str) and st.strip():
                    all_stations.add(st.strip().upper())

            prev_targets = prev.get("targets") if isinstance(prev.get("targets"), list) else []
            base_targets = [str(t).strip() for t in prev_targets if isinstance(t, str) and str(t).strip()]

    for target in targets:
        years_v2 = _list_years(target, NPT_ROOT_PATH_V2)
        years_v1 = _list_years(target, NPT_ROOT_PATH_V1)
        years = sorted(set(years_v1 + years_v2))
        if y0 is not None:
            years = [y for y in years if y0 <= y <= (y1 or y0)]
        if not years:
            print(f"[warn] no years for target={target}")
            continue

        for year in years:
            # Prefer CRD v2 when available for the same year.
            root_path = NPT_ROOT_PATH_V2 if year in years_v2 else NPT_ROOT_PATH_V1
            files = _list_files(target, year, root_path)
            if not files:
                continue

            selected: list[str] = []
            for fn in files:
                if _is_monthly(fn):
                    selected.append(fn)
                elif bool(args.include_daily) and _is_daily(fn):
                    selected.append(fn)

            for fn in sorted(selected):
                path = f"{root_path}/{target}/{year}/{fn}"
                rec_key = path
                if not rec_key:
                    rec_key = f"{target}/{year}/{fn}"
                if rec_key in existing_keys:
                    continue
                dst = edc_dir / target / str(year) / fn
                used_url = _download_any(path, dst, force=bool(args.force))

                st_codes = _extract_station_codes(dst)
                for s in st_codes:
                    all_stations.add(s)

                all_files.append(
                    {
                        "target": target,
                        "year": year,
                        "filename": fn,
                        "url": used_url or None,
                        "edc_path": path,
                        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
                        "bytes": int(dst.stat().st_size) if dst.exists() else None,
                        "sha256": _sha256(dst) if dst.exists() else None,
                        "cached_path": str(dst.relative_to(root)).replace("\\", "/"),
                        "stations": st_codes,
                    }
                )
                if rec_key:
                    existing_keys.add(rec_key)

    # Fetch station site logs for all detected stations
    stations_sorted = sorted(all_stations)
    for st in stations_sorted:
        try:
            _ensure_station_cached(root, st, force=bool(args.force))
        except Exception as e:
            print(f"[warn] station fetch failed: {st}: {e}")

    include_daily_out = bool(args.include_daily)
    if isinstance(prev_include_daily, bool):
        include_daily_out = bool(prev_include_daily) or include_daily_out

    year_filter_out = {"from": y0, "to": y1} if y0 is not None else None
    if prev_year_filter is not None and bool(args.append) and manifest_path.exists() and not args.force:
        year_filter_out = prev_year_filter

    manifest = {
        "source": "DGFI-TUM EDC (npt_crd + npt_crd_v2 + slrlog)",
        "edc_bases": _edc_bases(),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "targets": sorted(set(base_targets + targets)),
        "year_filter": year_filter_out,
        "include_daily": include_daily_out,
        "files": all_files,
        "stations_detected": stations_sorted,
    }
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] manifest: {manifest_path}")
    print(f"[ok] files: {len(all_files)}  stations: {len(stations_sorted)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
