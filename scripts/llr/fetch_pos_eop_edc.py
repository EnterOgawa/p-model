#!/usr/bin/env python3
"""
fetch_pos_eop_edc.py

DGFI-TUM EDC の pos+eop（SINEX .snx.gz）を取得して data/llr/pos_eop/ にキャッシュする。

目的:
  - LLR（EDC実測）評価で必要な「局座標（ITRF）」と「EOP（PM/LOD）」を一次ソース側で固定する
  - 2回目以降は --offline でネットワーク無しでも再現できる（キャッシュ利用）

入力:
  - 既に取得済みの EDC LLR（.np2）から観測日の集合を作り、必要日の pos+eop をまとめて取る
    - 優先: output/llr/batch/llr_batch_points.csv（存在する場合）
    - 代替: data/llr/edc/ 配下の .np2 を走査（epoch_utc を抽出）

出力（固定）
  - data/llr/pos_eop/snx/<YYYY>/<YYMMDD>/pos_eop_<YYMMDD>.snx.gz
  - data/llr/pos_eop/snx/<YYYY>/<YYMMDD>/meta.json

備考:
  - EDC の pos+eop は日次で提供される。ファイル名は例:
      nsgf.pos+eop.240112.v170.snx.gz
    ここでは「同日内で最大v（最新版）」を選ぶ。
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Optional


BASE = "https://edc.dgfi.tum.de"
POS_EOP_ROOT = "/pub/slr/products/pos+eop"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _fetch_text(url: str, *, timeout_s: int = 60) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "waveP-fetch-pos-eop/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        b = r.read()
    return b.decode("utf-8", "replace")


def _iter_hrefs(html: str) -> Iterable[str]:
    for href in re.findall(r'href=[\'"]([^\'"]+)[\'"]', html):
        if href:
            yield href


def _list_files(year: int, yymmdd: str) -> list[str]:
    url = f"{BASE}{POS_EOP_ROOT}/{year}/{yymmdd}/"
    html = _fetch_text(url, timeout_s=60)
    files: list[str] = []
    for href in _iter_hrefs(html):
        if not href.startswith(f"{POS_EOP_ROOT}/{year}/{yymmdd}/"):
            continue
        name = href.rsplit("/", 1)[-1]
        if name.lower().endswith(".snx.gz"):
            files.append(name)
    return sorted(set(files))


@dataclass(frozen=True)
class Picked:
    year: int
    yymmdd: str  # YYMMDD
    filename: str
    url: str
    center: str
    version: int


_RE_FILE = re.compile(
    r"^(?P<center>[a-z0-9]+)\.pos\+eop\.(?P<yymmdd>\d{6})\.v(?P<ver>\d+)\.snx\.gz$",
    re.IGNORECASE,
)


def _pick_latest(year: int, yymmdd: str, *, preferred_center: str = "nsgf") -> Picked:
    files = _list_files(year, yymmdd)
    if not files:
        raise FileNotFoundError(f"no .snx.gz found: {year}/{yymmdd}")

    cand: list[Picked] = []
    for fn in files:
        m = _RE_FILE.match(fn)
        if not m:
            continue
        center = str(m.group("center")).lower()
        ver = int(m.group("ver"))
        cand.append(
            Picked(
                year=year,
                yymmdd=yymmdd,
                filename=fn,
                url=f"{BASE}{POS_EOP_ROOT}/{year}/{yymmdd}/{fn}",
                center=center,
                version=ver,
            )
        )

    if not cand:
        # Fallback: take the lexicographically last file (best-effort)
        fn = sorted(files)[-1]
        return Picked(
            year=year,
            yymmdd=yymmdd,
            filename=fn,
            url=f"{BASE}{POS_EOP_ROOT}/{year}/{yymmdd}/{fn}",
            center="unknown",
            version=-1,
        )

    preferred = [c for c in cand if c.center == preferred_center.lower()]
    pool = preferred if preferred else cand
    return max(pool, key=lambda x: x.version)


def _dst_snx(repo_root: Path, year: int, yymmdd: str) -> Path:
    return repo_root / "data" / "llr" / "pos_eop" / "snx" / str(year) / str(yymmdd) / f"pos_eop_{yymmdd}.snx.gz"


def _dst_meta(repo_root: Path, year: int, yymmdd: str) -> Path:
    return repo_root / "data" / "llr" / "pos_eop" / "snx" / str(year) / str(yymmdd) / "meta.json"


def _download(url: str, dst: Path, *, force: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not force:
        print(f"[skip] exists: {dst}")
        return
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"[dl] {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "waveP-fetch-pos-eop/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1024 * 1024)
    tmp.replace(dst)
    print(f"[ok] saved: {dst} ({dst.stat().st_size} bytes)")


def _parse_yyyymmdd(s: str) -> date:
    # Accept YYYYMMDD or YYMMDD
    s = str(s).strip()
    if re.fullmatch(r"\d{8}", s):
        return date(int(s[0:4]), int(s[4:6]), int(s[6:8]))
    if re.fullmatch(r"\d{6}", s):
        yy = int(s[0:2])
        yyyy = 2000 + yy if yy < 80 else 1900 + yy
        return date(yyyy, int(s[2:4]), int(s[4:6]))
    raise ValueError(f"invalid date format: {s!r} (expected YYYYMMDD or YYMMDD)")


def _yymmdd(d: date) -> str:
    return f"{d.year % 100:02d}{d.month:02d}{d.day:02d}"


def _collect_dates_from_points_csv(path: Path) -> list[date]:
    if not path.exists():
        return []
    dates: set[date] = set()
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        r = csv.DictReader(f)
        if "epoch_utc" not in (r.fieldnames or []):
            return []
        for row in r:
            v = (row.get("epoch_utc") or "").strip()
            if not v:
                continue
            try:
                dt = datetime.fromisoformat(v)
            except Exception:
                continue
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dates.add(dt.astimezone(timezone.utc).date())
    return sorted(dates)


def _collect_dates_from_np2(repo_root: Path) -> list[date]:
    # Parse epoch_utc from NP2 via the existing CRD parser (no network).
    # Fallback to filename-based month/day if parsing fails.
    from scripts.llr import llr_pmodel_overlay_horizons_noargs as llr  # lazy import

    edc = repo_root / "data" / "llr" / "edc"
    if not edc.exists():
        return []

    np2_files = sorted(edc.glob("*/*/*.np2"))
    if not np2_files:
        return []

    dates: set[date] = set()
    for p in np2_files:
        try:
            df, _, _ = llr.parse_crd_npt11(p)  # type: ignore[attr-defined]
            if "epoch_utc" in df.columns:
                for t in df["epoch_utc"].dropna().tolist():
                    if isinstance(t, datetime):
                        dates.add(t.astimezone(timezone.utc).date())
        except Exception:
            # Best-effort: use filename tokens
            m = re.search(r"_(\d{8})\.np2$", p.name)
            if m:
                try:
                    dates.add(_parse_yyyymmdd(m.group(1)))
                except Exception:
                    pass
            m2 = re.search(r"_(\d{6})\.np2$", p.name)
            if m2:
                # Monthly file: take 15th as representative (will still help for coarse cache priming)
                try:
                    dd = _parse_yyyymmdd(m2.group(1) + "15")
                    dates.add(dd)
                except Exception:
                    pass
            continue

    return sorted(dates)


def main() -> int:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="Fetch EDC pos+eop (SINEX .snx.gz) files and cache them under data/llr/pos_eop/.")
    ap.add_argument("--from", dest="date_from", default="", help="Start date (YYYYMMDD or YYMMDD). If omitted, auto-detect from LLR data.")
    ap.add_argument("--to", dest="date_to", default="", help="End date (YYYYMMDD or YYMMDD). If omitted, auto-detect from LLR data.")
    ap.add_argument(
        "--points-csv",
        type=str,
        default="output/llr/batch/llr_batch_points.csv",
        help="Preferred source for required dates (epoch_utc).",
    )
    ap.add_argument("--center", type=str, default="nsgf", help="Preferred analysis center prefix (default: nsgf).")
    ap.add_argument("--offline", action="store_true", help="Do not use network; only succeed if cache exists.")
    ap.add_argument("--force", action="store_true", help="Re-download and overwrite cached files.")
    args = ap.parse_args()

    # Determine target date set
    date_from = _parse_yyyymmdd(args.date_from) if str(args.date_from).strip() else None
    date_to = _parse_yyyymmdd(args.date_to) if str(args.date_to).strip() else None

    if (date_from is None) != (date_to is None):
        print("[err] --from and --to must be provided together (or both omitted).")
        return 2

    dates: list[date] = []
    if date_from is not None and date_to is not None:
        if date_to < date_from:
            print("[err] --to must be >= --from")
            return 2
        d = date_from
        while d <= date_to:
            dates.append(d)
            d = date.fromordinal(d.toordinal() + 1)
    else:
        dates = _collect_dates_from_points_csv(root / str(args.points_csv))
        if not dates:
            dates = _collect_dates_from_np2(root)

    if not dates:
        print("[err] no dates found (need output/llr/batch/llr_batch_points.csv or data/llr/edc/*.np2)")
        return 2

    print(f"[info] target dates: {len(dates)} (min={dates[0]}, max={dates[-1]})")

    failures = 0
    for d in dates:
        yymmdd = _yymmdd(d)
        year = int(d.year)
        dst = _dst_snx(root, year, yymmdd)
        meta_path = _dst_meta(root, year, yymmdd)

        if args.offline:
            if dst.exists():
                print(f"[ok] offline: {dst}")
                continue
            print(f"[miss] offline: {dst}")
            failures += 1
            continue

        if dst.exists() and meta_path.exists() and not args.force:
            print(f"[skip] cached: {dst}")
            continue

        try:
            picked = _pick_latest(year, yymmdd, preferred_center=str(args.center).strip().lower() or "nsgf")
        except (FileNotFoundError, urllib.error.HTTPError) as e:
            print(f"[warn] missing on EDC: {year}/{yymmdd} ({e})")
            continue
        except Exception as e:
            print(f"[err] failed to list: {year}/{yymmdd} ({e})")
            failures += 1
            continue

        try:
            _download(picked.url, dst, force=bool(args.force))
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            print(f"[err] download failed: {picked.url} ({e})")
            failures += 1
            continue

        meta = {
            "source": "DGFI-TUM EDC pos+eop",
            "year": picked.year,
            "yymmdd": picked.yymmdd,
            "date_utc": d.isoformat(),
            "analysis_center": picked.center,
            "version": picked.version,
            "original_filename": picked.filename,
            "url": picked.url,
            "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        }
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    if failures:
        print(f"[err] failures: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

