#!/usr/bin/env python3
"""
fetch_cassini_pds_sce1_media_ancillary.py

Cassini SCE1 (CO-SS-RSS-1-SCE1-V1.0) の PDS3 一次配布から、
`sce1_ancillary/ion` と `sce1_ancillary/tro` を取得し、
媒体補正コマンド（ADJUST/DELETE ... FROM/TO/DSN）を抽出して台帳化する。

目的:
- 8.7.43.1: ION/TRO（媒体補正一次配布）を data/cassini/sources に固定保存する。
- 後段の record-level media join（8.7.43.2）で使う時間窓テーブルを生成する。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_BASE_URL = "https://atmos.nmsu.edu/pdsd/archive/data/co-ss-rss-1-sce1-v10"


# 関数: `_repo_root` の入出力契約と処理意図を定義する。

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# 関数: `_sha256` の入出力契約と処理意図を定義する。

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


# 関数: `_cors_for_doy` の入出力契約と処理意図を定義する。

def _cors_for_doy(doy: int) -> int:
    if doy < 157 or doy > 186:
        raise ValueError(f"DOY out of supported range for SCE1 (157-186): {doy}")

    return 21 + ((doy - 157) // 4)


# 関数: `_download` の入出力契約と処理意図を定義する。

def _download(url: str, dst: Path, *, force: bool, timeout_s: int = 120) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        return

    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "waveP-fetch-cassini-media/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r, tmp.open("wb") as f:
        shutil.copyfileobj(r, f, length=1024 * 1024)

    tmp.replace(dst)


# 関数: `_fetch_listing_html` の入出力契約と処理意図を定義する。

def _fetch_listing_html(url: str, *, timeout_s: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "waveP-fetch-cassini-media/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        return r.read().decode("utf-8", errors="ignore")


# 関数: `_extract_listing_files` の入出力契約と処理意図を定義する。

def _extract_listing_files(listing_html: str, base_url: str) -> List[str]:
    hrefs = re.findall(r'href="([^"]+)"', listing_html, flags=re.IGNORECASE)
    out: List[str] = []
    for href in hrefs:
        href = href.strip()
        if not href or href.startswith("?") or href in ("../", "./"):
            continue

        if href.endswith("/"):
            continue

        full = urllib.parse.urljoin(base_url, href)
        full = full.split("#", 1)[0].split("?", 1)[0]
        out.append(full)

    return sorted(set(out))


# 関数: `_first_match_text` の入出力契約と処理意図を定義する。

def _first_match_text(pattern: str, text: str) -> str:
    m = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    if not m:
        return ""

    return str(m.group(1)).strip()


# 関数: `_parse_csp_cmd_time` の入出力契約と処理意図を定義する。

def _parse_csp_cmd_time(token: str) -> Optional[datetime]:
    s = token.strip()
    # Expected examples:
    # - 02/06/01,06:05
    # - 02/06/01,06:00:00.001
    # - 02/06/01,06:00:00
    patterns = [
        "%y/%m/%d,%H:%M:%S.%f",
        "%y/%m/%d,%H:%M:%S",
        "%y/%m/%d,%H:%M",
    ]
    for p in patterns:
        try:
            return datetime.strptime(s, p).replace(tzinfo=timezone.utc)
        except Exception:
            continue

    return None


# 関数: `_extract_command_windows` の入出力契約と処理意図を定義する。

def _extract_command_windows(text: str, medium: str, filename: str) -> List[Dict[str, object]]:
    normalized = text.replace("\r", "\n")
    pat = re.compile(
        r"(?P<verb>ADJUST|DELETE)\((?P<data_type>[^)]*)\)(?P<body>.*?)"
        r"FROM\((?P<start>[^)]*)\)TO\((?P<stop>[^)]*)\)"
        r"DSN\(C(?P<dsn_complex>\d+)\)SCID\((?P<scid>\d+)\)\.",
        flags=re.IGNORECASE | re.DOTALL,
    )
    rows: List[Dict[str, object]] = []
    for m in pat.finditer(normalized):
        verb = str(m.group("verb")).upper()
        data_type = str(m.group("data_type")).strip().upper()
        body = str(m.group("body"))
        start_token = str(m.group("start")).strip()
        stop_token = str(m.group("stop")).strip()
        dsn_complex = int(m.group("dsn_complex"))
        scid = int(m.group("scid"))
        dt_start = _parse_csp_cmd_time(start_token)
        dt_stop = _parse_csp_cmd_time(stop_token)
        if dt_start is None or dt_stop is None:
            continue

        model_hits = re.findall(r"MODEL\s*\(([^)]+)\)", body, flags=re.IGNORECASE | re.DOTALL)
        if not model_hits:
            model_hits = [""]

        for model in model_hits:
            rows.append(
                {
                    "medium": medium,
                    "filename": filename,
                    "verb": verb,
                    "data_type": data_type,
                    "model": str(model).strip().upper(),
                    "start_utc": dt_start.isoformat(),
                    "stop_utc": dt_stop.isoformat(),
                    "dsn_complex": int(dsn_complex),
                    "scid": int(scid),
                }
            )

    return rows


# 関数: `_read_label_meta` の入出力契約と処理意図を定義する。

def _read_label_meta(path: Path) -> Dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    return {
        "product_id": _first_match_text(r'^\s*PRODUCT_ID\s*=\s*"([^"]+)"\s*$', text),
        "original_product_id": _first_match_text(r'^\s*ORIGINAL_PRODUCT_ID\s*=\s*"([^"]+)"\s*$', text),
        "start_time": _first_match_text(r"^\s*START_TIME\s*=\s*([0-9T:\-\.]+)\s*$", text),
        "stop_time": _first_match_text(r"^\s*STOP_TIME\s*=\s*([0-9T:\-\.]+)\s*$", text),
        "producer_id": _first_match_text(r'^\s*PRODUCER_ID\s*=\s*"([^"]+)"\s*$', text),
    }


# 関数: `_write_windows_csv` の入出力契約と処理意図を定義する。

def _write_windows_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "medium",
                "filename",
                "verb",
                "data_type",
                "model",
                "start_utc",
                "stop_utc",
                "dsn_complex",
                "scid",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k) for k in writer.fieldnames})


# 関数: `_write_availability_csv` の入出力契約と処理意図を定義する。

def _write_availability_csv(path: Path, payload: Dict[str, object]) -> None:
    rows = [
        ("all_required_mediums_local", str(bool(payload.get("all_required_mediums_local"))).lower()),
        ("required_mediums", ",".join([str(v) for v in (payload.get("required_mediums") or [])])),
        ("local_mediums_observed", ",".join([str(v) for v in (payload.get("local_mediums_observed") or [])])),
        ("ion_files_n", str(int(payload.get("ion_files_n") or 0))),
        ("tro_files_n", str(int(payload.get("tro_files_n") or 0))),
        ("plasma_model_chpart_windows_n", str(int(payload.get("plasma_model_chpart_windows_n") or 0))),
        ("command_windows_n", str(int(payload.get("command_windows_n") or 0))),
        ("source_listing_urls_n", str(int(payload.get("source_listing_urls_n") or 0))),
    ]
    medium_map = payload.get("medium_availability") if isinstance(payload.get("medium_availability"), dict) else {}
    for k in ["IONCAL", "TROPCAL", "PLSMCAL"]:
        rows.append((f"medium_availability.{k}", str(bool(medium_map.get(k))).lower()))

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        for k, v in rows:
            writer.writerow([k, v])


# 関数: `_sync_public` の入出力契約と処理意図を定義する。

def _sync_public(root: Path, names: Sequence[str]) -> None:
    src_dir = root / "output" / "cassini"
    dst_dir = root / "output" / "public" / "cassini"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for n in names:
        src = src_dir / n
        if src.exists():
            shutil.copy2(src, dst_dir / n)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    root = _repo_root()
    data_dir = root / "data" / "cassini" / "sources" / "csp_media_ancillary"
    files_dir = data_dir / "files"
    out_dir = root / "output" / "cassini"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    ap = argparse.ArgumentParser(description="Fetch PDS SCE1 ancillary media files (ion/tro) and build command-window manifest.")
    ap.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL, help="Base URL for PDS SCE1 mirror.")
    ap.add_argument("--doy-start", type=int, default=162, help="Start DOY (inclusive).")
    ap.add_argument("--doy-stop", type=int, default=182, help="Stop DOY (inclusive).")
    ap.add_argument("--offline", action="store_true", help="Do not fetch from network; use cached files only.")
    ap.add_argument("--force", action="store_true", help="Re-download cached files.")
    args = ap.parse_args()

    doy_start = int(args.doy_start)
    doy_stop = int(args.doy_stop)
    if doy_stop < doy_start:
        doy_start, doy_stop = doy_stop, doy_start

    cors_set = sorted({_cors_for_doy(d) for d in range(doy_start, doy_stop + 1)})
    base_url = str(args.base_url).rstrip("/")
    media_dirs = ["ion", "tro"]

    discovered: Dict[Tuple[str, str], Dict[str, object]] = {}
    listing_urls: List[str] = []
    missing_cache: List[str] = []
    for cors in cors_set:
        for medium in media_dirs:
            list_url = f"{base_url}/cors_{cors:04d}/sce1_ancillary/{medium}/"
            listing_urls.append(list_url)
            if args.offline:
                # Offline mode relies on already-cached files.
                continue

            try:
                html = _fetch_listing_html(list_url, timeout_s=60)
            except Exception:
                continue

            for file_url in _extract_listing_files(html, list_url):
                name = file_url.rsplit("/", 1)[-1]
                low = name.lower()
                if not low.endswith(f".{medium}") and not low.endswith(".lbl"):
                    continue

                key = (medium, name.lower())
                slot = discovered.setdefault(
                    key,
                    {
                        "medium": medium,
                        "filename": name.lower(),
                        "source_urls": [],
                        "cors_candidates": [],
                    },
                )
                srcs = slot.get("source_urls") if isinstance(slot.get("source_urls"), list) else []
                if file_url not in srcs:
                    srcs.append(file_url)

                slot["source_urls"] = srcs
                cors_candidates = slot.get("cors_candidates") if isinstance(slot.get("cors_candidates"), list) else []
                if int(cors) not in cors_candidates:
                    cors_candidates.append(int(cors))

                slot["cors_candidates"] = sorted(cors_candidates)

    if args.offline:
        for medium in media_dirs:
            medium_dir = files_dir / medium
            if not medium_dir.exists():
                continue

            for p in sorted(medium_dir.glob("*")):
                if not p.is_file():
                    continue

                key = (medium, p.name.lower())
                slot = discovered.setdefault(
                    key,
                    {
                        "medium": medium,
                        "filename": p.name.lower(),
                        "source_urls": [],
                        "cors_candidates": [],
                    },
                )
                slot["source_urls"] = []
                slot["cors_candidates"] = []

    files_meta: List[Dict[str, object]] = []
    windows: List[Dict[str, object]] = []
    label_meta_by_stem: Dict[str, Dict[str, str]] = {}
    for key in sorted(discovered.keys()):
        slot = discovered[key]
        medium = str(slot.get("medium") or "")
        filename = str(slot.get("filename") or "")
        urls = [str(v) for v in (slot.get("source_urls") or [])]
        url = urls[0] if urls else ""
        local = files_dir / medium / filename
        if args.offline:
            if not local.exists():
                missing_cache.append(str(local))
                continue
        else:
            if not url:
                continue

            _download(url, local, force=bool(args.force), timeout_s=180)

        meta: Dict[str, object] = {
            "medium": medium,
            "filename": filename,
            "local_path": str(local),
            "size_bytes": int(local.stat().st_size) if local.exists() else 0,
            "sha256": _sha256(local) if local.exists() else "",
            "source_url_primary": url,
            "source_urls": urls,
            "cors_candidates": slot.get("cors_candidates") or [],
        }
        files_meta.append(meta)

        if filename.endswith(".lbl"):
            stem = filename.rsplit(".", 1)[0]
            label_meta_by_stem[stem] = _read_label_meta(local)
            continue

        if filename.endswith(".ion") or filename.endswith(".tro"):
            content = local.read_text(encoding="utf-8", errors="replace")
            windows.extend(_extract_command_windows(content, medium, filename))

    if args.offline and missing_cache:
        print("[err] offline mode missing cached files:")
        for p in missing_cache[:80]:
            print("  -", p)

        if len(missing_cache) > 80:
            print(f"  ... and {len(missing_cache)-80} more")

        return 2

    for meta in files_meta:
        stem = str(meta.get("filename") or "").rsplit(".", 1)[0]
        if stem in label_meta_by_stem:
            meta["label_meta"] = label_meta_by_stem[stem]

    ion_files_n = sum(1 for m in files_meta if str(m.get("filename") or "").endswith(".ion"))
    tro_files_n = sum(1 for m in files_meta if str(m.get("filename") or "").endswith(".tro"))
    chpart_windows_n = sum(1 for w in windows if str(w.get("model") or "").upper() == "CHPART")

    medium_availability = {
        "IONCAL": bool(ion_files_n > 0),
        "TROPCAL": bool(tro_files_n > 0),
        # In SCE1 ancillary distribution, plasma adjustments are encoded via MODEL(CHPART) in ION command windows.
        "PLSMCAL": bool(chpart_windows_n > 0),
    }
    required_mediums = ["IONCAL", "TROPCAL", "PLSMCAL"]
    local_mediums_observed = [k for k in required_mediums if bool(medium_availability.get(k))]
    all_required = all(bool(medium_availability.get(k)) for k in required_mediums)

    manifest = {
        "source": "PDS3 mirror (Cassini SCE1): CO-SS-RSS-1-SCE1-V1.0",
        "base_url": base_url,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "doy_range": {"start": int(doy_start), "stop": int(doy_stop)},
        "cors_volumes": cors_set,
        "media_dirs": media_dirs,
        "required_mediums": required_mediums,
        "medium_availability": medium_availability,
        "local_mediums_observed": local_mediums_observed,
        "all_required_mediums_local": bool(all_required),
        "ion_files_n": int(ion_files_n),
        "tro_files_n": int(tro_files_n),
        "plasma_model_chpart_windows_n": int(chpart_windows_n),
        "command_windows_n": int(len(windows)),
        "source_listing_urls_n": int(len(listing_urls)),
        "source_listing_urls": sorted(set(listing_urls)),
        "files": files_meta,
        "note": "PLSMCAL is operationally treated as MODEL(CHPART) windows in SCE1 ancillary ION command files.",
    }

    data_manifest = data_dir / "media_ancillary_manifest.json"
    data_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    windows_csv = out_dir / "cassini_csp_media_command_windows.csv"
    availability_json = out_dir / "cassini_csp_media_ancillary_availability.json"
    availability_csv = out_dir / "cassini_csp_media_ancillary_availability.csv"
    _write_windows_csv(windows_csv, windows)
    availability_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_availability_csv(availability_csv, manifest)
    _sync_public(
        root,
        [
            windows_csv.name,
            availability_json.name,
            availability_csv.name,
        ],
    )
    print("Wrote:", data_manifest)
    print("Wrote:", windows_csv)
    print("Wrote:", availability_json)
    print("Wrote:", availability_csv)
    print("Synced:", root / "output" / "public" / "cassini")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
