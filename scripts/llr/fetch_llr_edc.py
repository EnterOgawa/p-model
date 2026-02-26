#!/usr/bin/env python3
"""
fetch_llr_edc.py

DGFI-TUM EDC 公開データ（npt_crd_v2）から LLR/SLR の CRD Normal Point を取得して
data/llr/ 配下に保存し、オフライン再現できる形にする。

主用途:
  - LLRの実測データをデモ入力ではなく公開データで評価する（インパクト重視）
  - 2回目以降は data/llr にキャッシュされたファイルでオフライン再現

出力（固定）
  - data/llr/edc/<target>/<year>/<filename>
  - data/llr/llr_primary.np2        （解析の既定入力）
  - data/llr/llr_primary_source.json（出典メタ）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Tuple


BASE = "https://edc.dgfi.tum.de"
ROOT_PATH = "/pub/slr/data/npt_crd_v2"


# 関数: `_repo_root` の入出力契約と処理意図を定義する。
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# 関数: `_sha256` の入出力契約と処理意図を定義する。

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


# 関数: `_fetch_text` の入出力契約と処理意図を定義する。

def _fetch_text(url: str, *, timeout_s: int = 60) -> str:
    with urllib.request.urlopen(url, timeout=timeout_s) as r:
        b = r.read()

    return b.decode("utf-8", "replace")


# 関数: `_iter_hrefs` の入出力契約と処理意図を定義する。

def _iter_hrefs(html: str) -> Iterable[str]:
    # EDC listing uses single quotes in href, but accept both.
    for href in re.findall(r'href=[\'"]([^\'"]+)[\'"]', html):
        # 条件分岐: `href` を満たす経路を評価する。
        if href:
            yield href


# 関数: `_list_years` の入出力契約と処理意図を定義する。

def _list_years(target: str) -> list[int]:
    url = f"{BASE}{ROOT_PATH}/{target}/"
    html = _fetch_text(url)
    years: list[int] = []
    for href in _iter_hrefs(html):
        # 条件分岐: `not href.startswith(f"{ROOT_PATH}/{target}/")` を満たす経路を評価する。
        if not href.startswith(f"{ROOT_PATH}/{target}/"):
            continue

        m = re.search(r"/(\d{4})/?$", href)
        # 条件分岐: `not m` を満たす経路を評価する。
        if not m:
            continue

        years.append(int(m.group(1)))

    return sorted(set(years))


# 関数: `_list_np2_files` の入出力契約と処理意図を定義する。

def _list_np2_files(target: str, year: int) -> list[str]:
    url = f"{BASE}{ROOT_PATH}/{target}/{year}/"
    html = _fetch_text(url)
    files: list[str] = []
    for href in _iter_hrefs(html):
        # 条件分岐: `not href.startswith(f"{ROOT_PATH}/{target}/{year}/")` を満たす経路を評価する。
        if not href.startswith(f"{ROOT_PATH}/{target}/{year}/"):
            continue

        name = href.rsplit("/", 1)[-1]
        # 条件分岐: `name.lower().endswith((".np2", ".np2.gz", ".crd", ".crd.gz", ".npt", ".npt.gz"))` を満たす経路を評価する。
        if name.lower().endswith((".np2", ".np2.gz", ".crd", ".crd.gz", ".npt", ".npt.gz")):
            files.append(name)

    return sorted(set(files))


# クラス: `Picked` の責務と境界条件を定義する。

@dataclass(frozen=True)
class Picked:
    target: str
    year: int
    filename: str

    # 関数: `url` の入出力契約と処理意図を定義する。
    @property
    def url(self) -> str:
        return f"{BASE}{ROOT_PATH}/{self.target}/{self.year}/{self.filename}"


# 関数: `_pick_latest_file` の入出力契約と処理意図を定義する。

def _pick_latest_file(target: str) -> Picked:
    years = _list_years(target)
    # 条件分岐: `not years` を満たす経路を評価する。
    if not years:
        raise RuntimeError(f"No year directories found for target={target}")

    year = max(years)
    files = _list_np2_files(target, year)
    # 条件分岐: `not files` を満たす経路を評価する。
    if not files:
        raise RuntimeError(f"No files found for target={target} year={year}")

    # Prefer the newest month, and within that month prefer the monthly aggregate file (YYYYMM).

    parsed: list[Tuple[int, int, str]] = []
    for fn in files:
        m = re.search(r"_(\d{6,8})\.", fn)
        # 条件分岐: `not m` を満たす経路を評価する。
        if not m:
            continue

        digits = m.group(1)
        # 条件分岐: `len(digits) == 6` を満たす経路を評価する。
        if len(digits) == 6:
            ym = int(digits)
            ymd = int(digits) * 100 + 99
        else:
            ym = int(digits[:6])
            ymd = int(digits)

        parsed.append((ym, ymd, fn))

    # 条件分岐: `not parsed` を満たす経路を評価する。

    if not parsed:
        # Fallback: lexical
        return Picked(target=target, year=year, filename=sorted(files)[-1])

    latest_ym = max(p[0] for p in parsed)
    in_month = [p for p in parsed if p[0] == latest_ym]

    # Monthly aggregate: digits length == 6
    monthly = [fn for ym, _ymd, fn in in_month if re.search(r"_(\d{6})\.", fn)]
    # 条件分岐: `monthly` を満たす経路を評価する。
    if monthly:
        return Picked(target=target, year=year, filename=sorted(monthly)[-1])

    # Otherwise newest day in that month

    _, _, best = max(in_month, key=lambda t: t[1])
    return Picked(target=target, year=year, filename=best)


# 関数: `_pick_by_ym_or_date` の入出力契約と処理意図を定義する。

def _pick_by_ym_or_date(*, target: str, digits: str) -> Picked:
    # 条件分岐: `not re.fullmatch(r"\d{6,8}", digits)` を満たす経路を評価する。
    if not re.fullmatch(r"\d{6,8}", digits):
        raise ValueError(f"Invalid digits: {digits} (expected YYYYMM or YYYYMMDD)")

    year = int(digits[:4])
    files = _list_np2_files(target, year)
    # 条件分岐: `not files` を満たす経路を評価する。
    if not files:
        raise RuntimeError(f"No files found for target={target} year={year}")

    # Prefer exact match (YYYYMMDD or YYYYMM)

    exact = [fn for fn in files if re.search(rf"_{re.escape(digits)}\.", fn)]
    # 条件分岐: `exact` を満たす経路を評価する。
    if exact:
        return Picked(target=target, year=year, filename=sorted(exact)[-1])

    # Fallback: month match -> monthly aggregate if exists

    if len(digits) == 8:
        ym = digits[:6]
        month = [fn for fn in files if re.search(rf"_{re.escape(ym)}(\d{{2}})?\.", fn)]
        monthly = [fn for fn in month if re.search(rf"_{re.escape(ym)}\.", fn)]
        # 条件分岐: `monthly` を満たす経路を評価する。
        if monthly:
            return Picked(target=target, year=year, filename=sorted(monthly)[-1])

        # 条件分岐: `month` を満たす経路を評価する。

        if month:
            return Picked(target=target, year=year, filename=sorted(month)[-1])

    raise FileNotFoundError(f"No file matched digits={digits} for target={target} year={year}")


# 関数: `_download` の入出力契約と処理意図を定義する。

def _download(url: str, dst: Path, *, force: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    # 条件分岐: `dst.exists() and not force` を満たす経路を評価する。
    if dst.exists() and not force:
        print(f"[skip] exists: {dst}")
        return

    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"[dl] {url}")
    with urllib.request.urlopen(url, timeout=180) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1024 * 1024)

    tmp.replace(dst)
    print(f"[ok] saved: {dst} ({dst.stat().st_size} bytes)")


# 関数: `_set_primary` の入出力契約と処理意図を定義する。

def _set_primary(*, src: Path, primary_path: Path) -> None:
    primary_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, primary_path)
    print(f"[ok] primary: {primary_path}")


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    root = _repo_root()
    data_dir = root / "data" / "llr"
    edc_dir = data_dir / "edc"

    ap = argparse.ArgumentParser(description="Fetch CRD Normal Point data from DGFI-TUM EDC (npt_crd_v2).")
    ap.add_argument("--target", type=str, default="apollo11", help="EDC target directory (e.g., apollo11/apollo14/apollo15)")

    pick = ap.add_mutually_exclusive_group()
    pick.add_argument("--latest", action="store_true", help="Pick the latest available monthly file (default).")
    pick.add_argument("--ym", type=str, default="", help="YYYYMM to pick (monthly aggregate preferred).")
    pick.add_argument("--date", type=str, default="", help="YYYY-MM-DD to pick (falls back to monthly if needed).")
    pick.add_argument("--digits", type=str, default="", help="YYYYMM or YYYYMMDD (direct selection).")
    pick.add_argument("--filename", type=str, default="", help="Explicit filename under <target>/<year>/ (e.g., apollo11_202510.np2)")

    ap.add_argument("--offline", action="store_true", help="Do not use network; only succeed if primary exists.")
    ap.add_argument("--force", action="store_true", help="Re-download and overwrite cached files.")
    args = ap.parse_args()

    primary = data_dir / "llr_primary.np2"
    meta_path = data_dir / "llr_primary_source.json"

    # 条件分岐: `args.offline` を満たす経路を評価する。
    if args.offline:
        # 条件分岐: `primary.exists()` を満たす経路を評価する。
        if primary.exists():
            print(f"[ok] offline: primary exists: {primary}")
            return 0

        print(f"[err] offline and missing: {primary}")
        return 2

    # 条件分岐: `primary.exists() and not args.force and not (args.ym or args.date or args.dig...` を満たす経路を評価する。

    if primary.exists() and not args.force and not (args.ym or args.date or args.digits or args.filename):
        # Default mode is "latest". If we already have a primary, keep it for reproducibility.
        print(f"[skip] primary already exists (use --force to refresh): {primary}")
        return 0

    target = str(args.target).strip()
    # 条件分岐: `not target` を満たす経路を評価する。
    if not target:
        print("[err] --target is empty")
        return 2

    # 条件分岐: `args.filename` を満たす経路を評価する。

    if args.filename:
        fn = args.filename.strip()
        m = re.search(r"_(\d{4})\d{2,4}\.", fn)
        # 条件分岐: `not m` を満たす経路を評価する。
        if not m:
            print(f"[err] cannot infer year from filename: {fn}")
            return 2

        year = int(m.group(1))
        picked = Picked(target=target, year=year, filename=fn)
    # 条件分岐: 前段条件が不成立で、`args.date` を追加評価する。
    elif args.date:
        m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", args.date.strip())
        # 条件分岐: `not m` を満たす経路を評価する。
        if not m:
            print(f"[err] invalid --date: {args.date} (expected YYYY-MM-DD)")
            return 2

        digits = f"{m.group(1)}{m.group(2)}{m.group(3)}"
        picked = _pick_by_ym_or_date(target=target, digits=digits)
    # 条件分岐: 前段条件が不成立で、`args.digits` を追加評価する。
    elif args.digits:
        picked = _pick_by_ym_or_date(target=target, digits=args.digits.strip())
    # 条件分岐: 前段条件が不成立で、`args.ym` を追加評価する。
    elif args.ym:
        picked = _pick_by_ym_or_date(target=target, digits=args.ym.strip())
    else:
        picked = _pick_latest_file(target)

    dst = edc_dir / picked.target / str(picked.year) / picked.filename
    _download(picked.url, dst, force=bool(args.force))

    _set_primary(src=dst, primary_path=primary)
    meta = {
        "source": "DGFI-TUM EDC (npt_crd_v2)",
        "target": picked.target,
        "year": picked.year,
        "filename": picked.filename,
        "url": picked.url,
        "downloaded_utc": datetime.now(timezone.utc).isoformat(),
        "bytes": int(dst.stat().st_size) if dst.exists() else None,
        "sha256": _sha256(dst) if dst.exists() else None,
        "cached_path": str(dst.relative_to(root)).replace("\\", "/"),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] meta: {meta_path}")

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
