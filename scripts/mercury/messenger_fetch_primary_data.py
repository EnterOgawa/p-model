#!/usr/bin/env python3
"""
messenger_fetch_primary_data.py

Roadmap Step 8.7.48（MESSENGER theory-native β）用の一次データ自動取得。

目的:
- MESSENGER PDS（radio science raw bundle）から ODF/TNF を自動取得し、
  `data/mercury/messenger/` 配下へ再現可能に配置する。
- 取得成否を machine-readable（CSV/JSON）で固定し、Stage A/B へ接続する。

入力:
- リモート: https://pds-geosciences.wustl.edu/messenger/urn-nasa-pds-mess-rs-raw/

出力:
- data/mercury/messenger/{data-odf,data-tnf}/...
- output/private/mercury/messenger_fetch_primary_manifest.csv
- output/private/mercury/messenger_fetch_primary_metrics.json
- 上記を output/public/mercury へ同期
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary.worklog import append_event


# クラス: `DownloadTask` の責務と境界条件を定義する。
@dataclass
class DownloadTask:
    product: str
    year: str
    filename: str
    url: str
    rel_path: str


# クラス: `DownloadResult` の責務と境界条件を定義する。

@dataclass
class DownloadResult:
    product: str
    year: str
    filename: str
    url: str
    rel_path: str
    status: str
    bytes: int
    note: str


# 関数: `_safe_rel` の入出力契約と処理意図を定義する。

def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve()).replace("\\", "/")


# 関数: `_resolve_path` の入出力契約と処理意図を定義する。

def _resolve_path(path_str: str, root: Path) -> Path:
    p = Path(str(path_str))
    if p.is_absolute():
        return p

    return (root / p).resolve()


# 関数: `_parse_csv_arg` の入出力契約と処理意図を定義する。

def _parse_csv_arg(text: str) -> List[str]:
    values = [v.strip() for v in str(text).split(",")]
    return [v for v in values if v]


# 関数: `_normalize_exts` の入出力契約と処理意図を定義する。

def _normalize_exts(exts: Sequence[str]) -> Tuple[str, ...]:
    out: List[str] = []
    for ext in exts:
        x = str(ext).strip().lower()
        if not x:
            continue

        if not x.startswith("."):
            x = "." + x

        out.append(x)

    return tuple(sorted(set(out)))


# 関数: `_fetch_html` の入出力契約と処理意図を定義する。

def _fetch_html(url: str, timeout_sec: float) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "waveP-messenger-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=timeout_sec) as r:
        raw = r.read()

    text = raw.decode("utf-8", errors="ignore")
    return text


# 関数: `_extract_hrefs` の入出力契約と処理意図を定義する。

def _extract_hrefs(html: str) -> List[str]:
    import re

    hits = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
    out: List[str] = []
    for h in hits:
        v = unescape(str(h).strip())
        if not v:
            continue

        out.append(v)

    return out


# 関数: `_list_dir_entries` の入出力契約と処理意図を定義する。

def _list_dir_entries(url: str, timeout_sec: float) -> Tuple[List[str], str]:
    try:
        html = _fetch_html(url, timeout_sec=timeout_sec)
    except urllib.error.HTTPError as e:
        return ([], f"http_error:{int(e.code)}")
    except urllib.error.URLError as e:
        return ([], f"url_error:{e.reason}")
    except Exception as e:
        return ([], f"fetch_error:{type(e).__name__}")

    hrefs = _extract_hrefs(html)
    names: List[str] = []
    for h in hrefs:
        if h.startswith("?") or h.startswith("#"):
            continue

        if h.startswith("mailto:"):
            continue

        parsed = urllib.parse.urlparse(h)
        if parsed.scheme in ("http", "https"):
            joined = h
        else:
            joined = urllib.parse.urljoin(url, h)

        rel = urllib.parse.urlparse(joined).path.rsplit("/", 1)[-1]
        if not rel:
            rel = h.rstrip("/")

        name = rel.strip()
        if not name:
            continue

        if name in (".", ".."):
            continue

        if h.endswith("/"):
            name = name + "/"

        names.append(name)

    uniq = sorted(set(names))
    return (uniq, "ok")


# 関数: `_is_downloadable_file` の入出力契約と処理意図を定義する。

def _is_downloadable_file(name: str, exts: Sequence[str]) -> bool:
    if str(name).endswith("/"):
        return False

    suffix = Path(name).suffix.lower()
    if suffix in tuple(exts):
        return True

    return False


# 関数: `_build_tasks` の入出力契約と処理意図を定義する。

def _build_tasks(
    base_url: str,
    data_root: Path,
    products: Sequence[str],
    years: Sequence[str],
    exts: Sequence[str],
    timeout_sec: float,
    max_files_per_year: int,
    fetch_root_metadata: bool,
) -> Tuple[List[DownloadTask], Dict[str, object]]:
    tasks: List[DownloadTask] = []
    inspect_log: Dict[str, object] = {"products": {}}
    for product in products:
        product_url = urllib.parse.urljoin(base_url.rstrip("/") + "/", product.strip("/") + "/")
        product_key = product.strip("/")
        inspect_log["products"][product_key] = {
            "product_url": product_url,
            "root_status": "",
            "root_files": 0,
            "year_status": {},
        }
        entries, status = _list_dir_entries(product_url, timeout_sec=timeout_sec)
        inspect_log["products"][product_key]["root_status"] = status
        if status != "ok":
            continue

        root_files = [n for n in entries if _is_downloadable_file(n, exts)]
        inspect_log["products"][product_key]["root_files"] = int(len(root_files))
        if fetch_root_metadata:
            for name in sorted(root_files):
                rel_path = f"{product_key}/{name}"
                url = urllib.parse.urljoin(product_url, name)
                tasks.append(
                    DownloadTask(
                        product=product_key,
                        year="root",
                        filename=name,
                        url=url,
                        rel_path=rel_path,
                    )
                )

        for year in years:
            year_url = urllib.parse.urljoin(product_url, str(year).strip("/") + "/")
            year_entries, year_status = _list_dir_entries(year_url, timeout_sec=timeout_sec)
            year_key = str(year)
            inspect_log["products"][product_key]["year_status"][year_key] = {
                "status": year_status,
                "files": 0,
            }
            if year_status != "ok":
                continue

            year_files = [n for n in year_entries if _is_downloadable_file(n, exts)]
            year_files = sorted(year_files)
            if max_files_per_year > 0:
                year_files = year_files[: max_files_per_year]

            inspect_log["products"][product_key]["year_status"][year_key]["files"] = int(len(year_files))
            for name in year_files:
                rel_path = f"{product_key}/{year_key}/{name}"
                url = urllib.parse.urljoin(year_url, name)
                tasks.append(
                    DownloadTask(
                        product=product_key,
                        year=year_key,
                        filename=name,
                        url=url,
                        rel_path=rel_path,
                    )
                )

    return (tasks, inspect_log)


# 関数: `_download_one` の入出力契約と処理意図を定義する。

def _download_one(task: DownloadTask, dst_root: Path, force: bool, timeout_sec: float) -> DownloadResult:
    dst = (dst_root / task.rel_path).resolve()
    if dst.exists() and (not force):
        return DownloadResult(
            product=task.product,
            year=task.year,
            filename=task.filename,
            url=task.url,
            rel_path=task.rel_path,
            status="skip",
            bytes=int(dst.stat().st_size),
            note="exists",
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(task.url, headers={"User-Agent": "waveP-messenger-fetch/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as r, tmp.open("wb") as f:
            shutil.copyfileobj(r, f)
    except urllib.error.HTTPError as e:
        if tmp.exists():
            tmp.unlink()

        return DownloadResult(
            product=task.product,
            year=task.year,
            filename=task.filename,
            url=task.url,
            rel_path=task.rel_path,
            status="reject",
            bytes=0,
            note=f"http_error:{int(e.code)}",
        )
    except urllib.error.URLError as e:
        if tmp.exists():
            tmp.unlink()

        return DownloadResult(
            product=task.product,
            year=task.year,
            filename=task.filename,
            url=task.url,
            rel_path=task.rel_path,
            status="reject",
            bytes=0,
            note=f"url_error:{e.reason}",
        )
    except Exception as e:
        if tmp.exists():
            tmp.unlink()

        return DownloadResult(
            product=task.product,
            year=task.year,
            filename=task.filename,
            url=task.url,
            rel_path=task.rel_path,
            status="reject",
            bytes=0,
            note=f"download_error:{type(e).__name__}",
        )

    tmp.replace(dst)
    size = int(dst.stat().st_size)
    return DownloadResult(
        product=task.product,
        year=task.year,
        filename=task.filename,
        url=task.url,
        rel_path=task.rel_path,
        status="download",
        bytes=size,
        note="ok",
    )


# 関数: `_write_manifest` の入出力契約と処理意図を定義する。

def _write_manifest(path: Path, rows: Sequence[DownloadResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "product",
        "year",
        "filename",
        "url",
        "rel_path",
        "status",
        "bytes",
        "note",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "product": r.product,
                    "year": r.year,
                    "filename": r.filename,
                    "url": r.url,
                    "rel_path": r.rel_path,
                    "status": r.status,
                    "bytes": r.bytes,
                    "note": r.note,
                }
            )


# 関数: `_sync_to_public` の入出力契約と処理意図を定義する。

def _sync_to_public(paths: Iterable[Path], private_root: Path, public_root: Path) -> List[Path]:
    public_root.mkdir(parents=True, exist_ok=True)
    synced: List[Path] = []
    for src in paths:
        try:
            rel = src.resolve().relative_to(private_root.resolve())
        except Exception:
            rel = Path(src.name)

        dst = public_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        synced.append(dst)

    return synced


# 関数: `_status_counts` の入出力契約と処理意図を定義する。

def _status_counts(rows: Sequence[DownloadResult]) -> Dict[str, int]:
    out = {"download": 0, "skip": 0, "reject": 0}
    for r in rows:
        out[r.status] = int(out.get(r.status, 0)) + 1

    return out


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch MESSENGER ODF/TNF primary data from PDS.")
    ap.add_argument(
        "--base-url",
        type=str,
        default="https://pds-geosciences.wustl.edu/messenger/urn-nasa-pds-mess-rs-raw/",
        help="Base URL of MESSENGER radio science raw bundle.",
    )
    ap.add_argument(
        "--data-root",
        type=str,
        default=str(_ROOT / "data" / "mercury" / "messenger"),
        help="Local data root.",
    )
    ap.add_argument(
        "--products",
        type=str,
        default="data-odf,data-tnf",
        help="Comma separated product directories under base URL.",
    )
    ap.add_argument(
        "--years",
        type=str,
        default="2011",
        help="Comma separated year directories to fetch.",
    )
    ap.add_argument(
        "--extensions",
        type=str,
        default=".dat,.xml,.lbl,.txt,.csv,.tab,.fmt",
        help="Comma separated file extensions to download.",
    )
    ap.add_argument(
        "--max-files-per-year",
        type=int,
        default=60,
        help="Max files per product-year (0 for all).",
    )
    ap.add_argument(
        "--fetch-root-metadata",
        action="store_true",
        help="Also download root-level metadata files (collection*.csv/xml etc.).",
    )
    ap.add_argument("--force", action="store_true", help="Re-download even if local files exist.")
    ap.add_argument("--timeout-sec", type=float, default=120.0, help="HTTP timeout seconds.")
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(_ROOT / "output" / "private" / "mercury"),
        help="Private output directory.",
    )
    ap.add_argument(
        "--public-dir",
        type=str,
        default=str(_ROOT / "output" / "public" / "mercury"),
        help="Public sync directory.",
    )
    args = ap.parse_args()

    data_root = _resolve_path(args.data_root, _ROOT)
    out_dir = _resolve_path(args.out_dir, _ROOT)
    public_dir = _resolve_path(args.public_dir, _ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)

    products = _parse_csv_arg(args.products)
    years = _parse_csv_arg(args.years)
    exts = _normalize_exts(_parse_csv_arg(args.extensions))
    max_files_per_year = int(args.max_files_per_year)
    tasks, inspect_log = _build_tasks(
        base_url=str(args.base_url).strip(),
        data_root=data_root,
        products=products,
        years=years,
        exts=exts,
        timeout_sec=float(args.timeout_sec),
        max_files_per_year=max_files_per_year,
        fetch_root_metadata=bool(args.fetch_root_metadata),
    )

    results: List[DownloadResult] = []
    for task in tasks:
        res = _download_one(
            task=task,
            dst_root=data_root,
            force=bool(args.force),
            timeout_sec=float(args.timeout_sec),
        )
        results.append(res)

    counts = _status_counts(results)
    total_bytes = int(sum(r.bytes for r in results if r.status in ("download", "skip")))
    overall_status = "reject"
    if counts.get("download", 0) > 0:
        overall_status = "pass"
    elif counts.get("skip", 0) > 0 and counts.get("reject", 0) == 0:
        overall_status = "pass"
    elif counts.get("skip", 0) > 0:
        overall_status = "watch"

    manifest_csv = out_dir / "messenger_fetch_primary_manifest.csv"
    metrics_json = out_dir / "messenger_fetch_primary_metrics.json"
    _write_manifest(manifest_csv, results)
    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase_step": "8.7.48.1",
        "overall_status": overall_status,
        "base_url": str(args.base_url).strip(),
        "data_root": _safe_rel(data_root, _ROOT),
        "products": products,
        "years": years,
        "extensions": list(exts),
        "max_files_per_year": max_files_per_year,
        "fetch_root_metadata": bool(args.fetch_root_metadata),
        "force": bool(args.force),
        "status_counts": counts,
        "n_tasks": int(len(tasks)),
        "n_results": int(len(results)),
        "total_bytes": total_bytes,
        "inspect_log": inspect_log,
        "outputs_private": [
            _safe_rel(manifest_csv, _ROOT),
            _safe_rel(metrics_json, _ROOT),
        ],
    }
    metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    synced = _sync_to_public([manifest_csv, metrics_json], private_root=out_dir, public_root=public_dir)
    metrics["outputs_public"] = [_safe_rel(p, _ROOT) for p in synced]
    metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    _sync_to_public([metrics_json], private_root=out_dir, public_root=public_dir)

    append_event(
        {
            "event": "run_script",
            "script": "scripts/mercury/messenger_fetch_primary_data.py",
            "phase_step": "8.7.48.1",
            "status": overall_status,
            "input": str(args.base_url).strip(),
            "outputs": [_safe_rel(manifest_csv, _ROOT), _safe_rel(metrics_json, _ROOT)],
            "metrics": {
                "n_tasks": int(len(tasks)),
                "download": int(counts.get("download", 0)),
                "skip": int(counts.get("skip", 0)),
                "reject": int(counts.get("reject", 0)),
                "total_bytes": total_bytes,
            },
        }
    )

    print(f"[ok] overall_status={overall_status}")
    print(f"[ok] n_tasks={len(tasks)}")
    print(f"[ok] download={counts.get('download', 0)} skip={counts.get('skip', 0)} reject={counts.get('reject', 0)}")
    print(f"[ok] wrote: {manifest_csv}")
    print(f"[ok] wrote: {metrics_json}")
    print(f"[ok] synced_to_public={len(synced)}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
