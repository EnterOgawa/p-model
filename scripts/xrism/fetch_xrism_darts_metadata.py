#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_xrism_darts_metadata.py

Phase 4 / Step 4.8（XRISM）:
DARTS（ISAS/JAXA）で公開されている XRISM/Resolve の metadata と public list を取得し、
offline 再現できる形で `data/xrism/sources/darts/` にキャッシュして manifest（sha256）を固定する。

入出力（固定）:
- 入力（online）:
  - public list: https://darts.isas.jaxa.jp/pub/xrism/browse/public_list/
  - metadata CSV: https://data.darts.isas.jaxa.jp/pub/xrism/metadata/xrism_resolve_data.csv
- キャッシュ:
  - data/xrism/sources/darts/public_list.html
  - data/xrism/sources/darts/xrism_resolve_data.csv
  - data/xrism/sources/darts/manifest.json
- サマリ（公開済み観測の抽出）:
  - output/private/xrism/darts_resolve_public_observations.csv

注意:
- DARTS 側の directory listing / HTML は安定ではない可能性があるため、一次の正は metadata CSV とし、
  public_list は参照用として同時に保存する（差分が出た場合の切り分けに使う）。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency for offline/local runs
    requests = None

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402

_REQ_TIMEOUT = (30, 300)  # (connect, read)

DEFAULT_PUBLIC_LIST_URL = "https://darts.isas.jaxa.jp/pub/xrism/browse/public_list/"
DEFAULT_METADATA_URL = "https://data.darts.isas.jaxa.jp/pub/xrism/metadata/xrism_resolve_data.csv"


# 関数: `_utc_now` の入出力契約と処理意図を定義する。
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_sha256_bytes` の入出力契約と処理意図を定義する。

def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# 関数: `_sha256_file` の入出力契約と処理意図を定義する。

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


# 関数: `_rel` の入出力契約と処理意図を定義する。

def _rel(path: Path) -> str:
    try:
        return path.relative_to(_ROOT).as_posix()
    except Exception:
        return path.as_posix()


# 関数: `_write_json` の入出力契約と処理意図を定義する。

def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# 関数: `_http_get_bytes` の入出力契約と処理意図を定義する。

def _http_get_bytes(url: str) -> Tuple[bytes, str]:
    # 条件分岐: `requests is None` を満たす経路を評価する。
    if requests is None:
        raise RuntimeError("requests is required for online fetch")

    r = requests.get(url, timeout=_REQ_TIMEOUT)
    r.raise_for_status()
    content_type = str(r.headers.get("Content-Type") or "")
    return bytes(r.content), content_type


_OBSID_RE = re.compile(r"\b(?P<obsid>[0-9]{9})\b")


# 関数: `_extract_obsids_from_public_list` の入出力契約と処理意図を定義する。
def _extract_obsids_from_public_list(html: str) -> List[str]:
    obsids = sorted({m.group("obsid") for m in _OBSID_RE.finditer(html or "")})
    return obsids


# 関数: `_parse_iso_date` の入出力契約と処理意図を定義する。

def _parse_iso_date(s: str) -> Optional[str]:
    s = (s or "").strip()
    # 条件分岐: `not s` を満たす経路を評価する。
    if not s:
        return None
    # Keep only YYYY-MM-DD for stability.

    m = re.match(r"^(\d{4}-\d{2}-\d{2})", s)
    return m.group(1) if m else None


# 関数: `_is_public_row` の入出力契約と処理意図を定義する。

def _is_public_row(row: Dict[str, str], *, today_ymd: str) -> bool:
    d = _parse_iso_date(row.get("public_date") or "")
    # 条件分岐: `not d` を満たす経路を評価する。
    if not d:
        return False

    return d <= today_ymd


# クラス: `ResolveRow` の責務と境界条件を定義する。

@dataclass(frozen=True)
class ResolveRow:
    obsid: str
    object_name: str
    public_date: str
    resolve_exposure: str
    processing_version: str


# 関数: `_read_resolve_rows` の入出力契約と処理意図を定義する。

def _read_resolve_rows(csv_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # 条件分岐: `not isinstance(r, dict)` を満たす経路を評価する。
            if not isinstance(r, dict):
                continue

            rows.append({str(k): (v or "").strip() for k, v in r.items() if k is not None})

    return rows


# 関数: `_summarize_public_observations` の入出力契約と処理意図を定義する。

def _summarize_public_observations(rows: List[Dict[str, str]]) -> List[ResolveRow]:
    today_ymd = datetime.now(timezone.utc).date().isoformat()
    out: List[ResolveRow] = []
    for r in rows:
        # 条件分岐: `not _is_public_row(r, today_ymd=today_ymd)` を満たす経路を評価する。
        if not _is_public_row(r, today_ymd=today_ymd):
            continue

        obsid = (r.get("observation_id") or "").strip()
        # 条件分岐: `not obsid` を満たす経路を評価する。
        if not obsid:
            continue

        out.append(
            ResolveRow(
                obsid=obsid,
                object_name=(r.get("object_name") or "").strip(),
                public_date=_parse_iso_date(r.get("public_date") or "") or "",
                resolve_exposure=(r.get("resolve_exposure") or "").strip(),
                processing_version=(r.get("processing_version") or "").strip(),
            )
        )

    out.sort(key=lambda x: (x.public_date, x.obsid))
    return out


# 関数: `_write_public_summary_csv` の入出力契約と処理意図を定義する。

def _write_public_summary_csv(path: Path, rows: List[ResolveRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["observation_id", "object_name", "public_date", "resolve_exposure", "processing_version"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "observation_id": r.obsid,
                    "object_name": r.object_name,
                    "public_date": r.public_date,
                    "resolve_exposure": r.resolve_exposure,
                    "processing_version": r.processing_version,
                }
            )


# 関数: `_build_manifest` の入出力契約と処理意図を定義する。

def _build_manifest(
    *,
    out_dir: Path,
    public_list_url: str,
    metadata_url: str,
    public_list_path: Path,
    metadata_path: Path,
    content_types: Dict[str, str],
    public_list_obsids: List[str],
) -> Dict[str, Any]:
    files: List[Dict[str, Any]] = []
    for p in [public_list_path, metadata_path]:
        files.append(
            {
                "path": _rel(p),
                "sha256": _sha256_file(p),
                "bytes": int(p.stat().st_size),
                "content_type": content_types.get(p.name, ""),
            }
        )

    return {
        "generated_utc": _utc_now(),
        "sources": {
            "public_list_url": public_list_url,
            "metadata_url": metadata_url,
        },
        "cached": {
            "dir": _rel(out_dir),
            "files": files,
        },
        "public_list_extracted": {
            "n_obsids": len(public_list_obsids),
            "obsids": public_list_obsids,
        },
        "notes": [
            "一次の正は metadata CSV（Resolve）とし、public_list は参照用として同時保存する。",
        ],
    }


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--public-list-url", default=DEFAULT_PUBLIC_LIST_URL)
    p.add_argument("--metadata-url", default=DEFAULT_METADATA_URL)
    p.add_argument("--out-dir", default=str(_ROOT / "data" / "xrism" / "sources" / "darts"))
    p.add_argument("--summary-csv", default=str(_ROOT / "output" / "private" / "xrism" / "darts_resolve_public_observations.csv"))
    args = p.parse_args(list(argv) if argv is not None else None)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    public_list_path = out_dir / "public_list.html"
    metadata_path = out_dir / "xrism_resolve_data.csv"
    manifest_path = out_dir / "manifest.json"

    content_types: Dict[str, str] = {}

    # Fetch public list (HTML)
    html_bytes, ct = _http_get_bytes(str(args.public_list_url))
    content_types[public_list_path.name] = ct
    public_list_path.write_bytes(html_bytes)
    public_list_text = None
    try:
        public_list_text = html_bytes.decode("utf-8", errors="replace")
    except Exception:
        public_list_text = html_bytes.decode("latin-1", errors="replace")

    public_list_obsids = _extract_obsids_from_public_list(public_list_text)

    # Fetch metadata CSV (Resolve)
    csv_bytes, ct = _http_get_bytes(str(args.metadata_url))
    content_types[metadata_path.name] = ct
    metadata_path.write_bytes(csv_bytes)

    # Build summary CSV from metadata (public_date <= today)
    resolve_rows = _read_resolve_rows(metadata_path)
    summary_rows = _summarize_public_observations(resolve_rows)
    _write_public_summary_csv(Path(args.summary_csv), summary_rows)

    # Manifest
    manifest = _build_manifest(
        out_dir=out_dir,
        public_list_url=str(args.public_list_url),
        metadata_url=str(args.metadata_url),
        public_list_path=public_list_path,
        metadata_path=metadata_path,
        content_types=content_types,
        public_list_obsids=public_list_obsids,
    )
    _write_json(manifest_path, manifest)

    worklog.append_event(
        {
            "task": "xrism_darts_metadata_cache",
            "generated_utc": _utc_now(),
            "inputs": {
                "public_list_url": str(args.public_list_url),
                "metadata_url": str(args.metadata_url),
            },
            "outputs": {
                "manifest": manifest_path,
                "public_list_html": public_list_path,
                "resolve_metadata_csv": metadata_path,
                "public_summary_csv": Path(args.summary_csv),
            },
            "metrics": {
                "n_public_summary_rows": len(summary_rows),
                "n_public_list_obsids_extracted": len(public_list_obsids),
            },
        }
    )

    print(f"[ok] wrote: {manifest_path}")
    print(f"[ok] wrote: {args.summary_csv}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())

