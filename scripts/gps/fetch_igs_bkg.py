#!/usr/bin/env python3
"""
fetch_igs_bkg.py

IGS 公開データ（BKG root_ftp）から、GPS（時計残差）検証に必要な入力を取得して
data/gps/ に固定ファイル名で配置する。

目的:
  - ローカルに無いデータはWebから取得し、正しいデータで評価する
  - 2回目以降は data/gps を使ってオフライン再現する

取得対象（既定: 2025-10-01 / DOY=274）:
  - BRDC RINEX NAV: BRDC00IGS_R_YYYYDDD0000_01D_MN.rnx
  - IGS Final CLK:  IGS0OPSFIN_YYYYDDD0000_01D_05M_CLK.CLK
  - IGS Final SP3:  IGS0OPSFIN_YYYYDDD0000_01D_15M_ORB.SP3
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional


BASE = "https://igs.bkg.bund.de/root_ftp/IGS"
GPS_EPOCH = date(1980, 1, 6)  # start of GPS week 0


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def _gps_week(d: date) -> int:
    return (d - GPS_EPOCH).days // 7


def _date_from_year_doy(year: int, doy: int) -> date:
    return date(year, 1, 1) + timedelta(days=doy - 1)


def _download(url: str, gz_path: Path, *, force: bool) -> None:
    gz_path.parent.mkdir(parents=True, exist_ok=True)
    # 条件分岐: `gz_path.exists() and not force` を満たす経路を評価する。
    if gz_path.exists() and not force:
        print(f"[skip] exists: {gz_path}")
        return

    tmp = gz_path.with_suffix(gz_path.suffix + ".part")
    print(f"[dl] {url}")
    with urllib.request.urlopen(url, timeout=180) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1024 * 1024)

    tmp.replace(gz_path)
    print(f"[ok] saved: {gz_path} ({gz_path.stat().st_size} bytes)")


def _gunzip(src_gz: Path, dst: Path, *, force: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    # 条件分岐: `dst.exists() and not force` を満たす経路を評価する。
    if dst.exists() and not force:
        print(f"[skip] exists: {dst}")
        return

    tmp = dst.with_suffix(dst.suffix + ".part")
    with gzip.open(src_gz, "rb") as r, open(tmp, "wb") as w:
        shutil.copyfileobj(r, w, length=1024 * 1024)

    tmp.replace(dst)
    print(f"[ok] unzip: {dst} ({dst.stat().st_size} bytes)")


@dataclass(frozen=True)
class Targets:
    year: int
    doy: int
    gps_week: int
    brdc_gz: str
    clk_gz: str
    sp3_gz: str
    brdc_out: str
    clk_out: str
    sp3_out: str


def _targets_for(year: int, doy: int) -> Targets:
    d = _date_from_year_doy(year, doy)
    week = _gps_week(d)
    yddd = f"{year}{doy:03d}0000"
    brdc = f"BRDC00IGS_R_{yddd}_01D_MN.rnx"
    clk = f"IGS0OPSFIN_{yddd}_01D_05M_CLK.CLK"
    sp3 = f"IGS0OPSFIN_{yddd}_01D_15M_ORB.SP3"
    return Targets(
        year=year,
        doy=doy,
        gps_week=week,
        brdc_gz=f"{brdc}.gz",
        clk_gz=f"{clk}.gz",
        sp3_gz=f"{sp3}.gz",
        brdc_out=brdc,
        clk_out=clk,
        sp3_out=sp3,
    )


def main() -> int:
    root = _repo_root()
    data_dir = root / "data" / "gps"
    meta_path = data_dir / "igs_bkg_sources.json"

    ap = argparse.ArgumentParser(description="Fetch IGS public products from BKG root_ftp and cache to data/gps.")
    ap.add_argument("--year", type=int, default=2025, help="Year (UTC) of the target day")
    ap.add_argument("--doy", type=int, default=274, help="Day of year (1-366)")
    ap.add_argument("--date", type=str, default="", help="Optional: YYYY-MM-DD (overrides --year/--doy)")
    ap.add_argument("--offline", action="store_true", help="Do not use network; only succeed if files exist.")
    ap.add_argument("--force", action="store_true", help="Re-download and overwrite cached files.")
    args = ap.parse_args()

    # 条件分岐: `args.date` を満たす経路を評価する。
    if args.date:
        try:
            y, m, d = (int(x) for x in args.date.split("-"))
            dt = date(y, m, d)
        except Exception:
            print(f"[err] invalid --date: {args.date} (expected YYYY-MM-DD)")
            return 2

        year = dt.year
        doy = int(dt.strftime("%j"))
    else:
        year = int(args.year)
        doy = int(args.doy)

    # 条件分岐: `doy < 1 or doy > 366` を満たす経路を評価する。

    if doy < 1 or doy > 366:
        print(f"[err] invalid --doy: {doy}")
        return 2

    t = _targets_for(year, doy)

    out_brdc = data_dir / t.brdc_out
    out_clk = data_dir / t.clk_out
    out_sp3 = data_dir / t.sp3_out

    # 条件分岐: `args.offline` を満たす経路を評価する。
    if args.offline:
        missing = [p for p in (out_brdc, out_clk, out_sp3) if not p.exists()]
        # 条件分岐: `missing` を満たす経路を評価する。
        if missing:
            print("[err] offline and missing:")
            for p in missing:
                print(f"  - {p}")

            return 2

        print("[ok] offline: inputs already exist.")
        return 0

    # URLs

    url_brdc = f"{BASE}/BRDC/{t.year}/{t.doy:03d}/{t.brdc_gz}"
    url_prod = f"{BASE}/products/{t.gps_week}"
    url_clk = f"{url_prod}/{t.clk_gz}"
    url_sp3 = f"{url_prod}/{t.sp3_gz}"

    cache_dir = data_dir / "_cache_gz"
    gz_brdc = cache_dir / t.brdc_gz
    gz_clk = cache_dir / t.clk_gz
    gz_sp3 = cache_dir / t.sp3_gz

    _download(url_brdc, gz_brdc, force=bool(args.force))
    _download(url_clk, gz_clk, force=bool(args.force))
    _download(url_sp3, gz_sp3, force=bool(args.force))

    _gunzip(gz_brdc, out_brdc, force=bool(args.force))
    _gunzip(gz_clk, out_clk, force=bool(args.force))
    _gunzip(gz_sp3, out_sp3, force=bool(args.force))

    meta = {
        "source": "IGS BKG root_ftp",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "year": t.year,
        "doy": t.doy,
        "gps_week": t.gps_week,
        "urls": {"brdc": url_brdc, "clk": url_clk, "sp3": url_sp3},
        "files": {
            "brdc": {"path": str(out_brdc.relative_to(root)).replace("\\", "/"), "sha256": _sha256(out_brdc)},
            "clk": {"path": str(out_clk.relative_to(root)).replace("\\", "/"), "sha256": _sha256(out_clk)},
            "sp3": {"path": str(out_sp3.relative_to(root)).replace("\\", "/"), "sha256": _sha256(out_sp3)},
        },
    }
    data_dir.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] meta: {meta_path}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())

