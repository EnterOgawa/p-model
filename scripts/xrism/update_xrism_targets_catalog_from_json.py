#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
update_xrism_targets_catalog_from_json.py

Phase 4 / Step 4.8（XRISM）:
`data/xrism/sources/xrism_target_catalog.json`（DARTS metadata 由来の選定リスト）を、
既存の解析スクリプトが参照している `output/private/xrism/xrism_targets_catalog.csv` に反映する。

方針:
- 既存CSVの行（手動コメント等）は保持しつつ、JSONにある obsid を追加する（不足行の補完）。
- 既存行がある場合は、原則として上書きしない（`--overwrite` 指定時のみ更新）。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


# 関数: `_utc_now` の入出力契約と処理意図を定義する。
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_read_csv_rows` の入出力契約と処理意図を定義する。

def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        return []

    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # 条件分岐: `not isinstance(r, dict)` を満たす経路を評価する。
            if not isinstance(r, dict):
                continue

            rows.append({str(k): (v or "").strip() for k, v in r.items() if k is not None})

    return rows


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["obsid", "target_name", "role", "z_sys", "instrument_prefer", "comment", "remote_cat_hint"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: (r.get(k) or "") for k in fieldnames})


# 関数: `_load_target_catalog_json` の入出力契約と処理意図を定義する。

def _load_target_catalog_json(path: Path) -> List[Dict[str, Any]]:
    try:
        j = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    targets = j.get("targets")
    # 条件分岐: `not isinstance(targets, list)` を満たす経路を評価する。
    if not isinstance(targets, list):
        return []

    out: List[Dict[str, Any]] = []
    for t in targets:
        # 条件分岐: `isinstance(t, dict)` を満たす経路を評価する。
        if isinstance(t, dict):
            out.append(t)

    return out


# 関数: `_default_comment` の入出力契約と処理意図を定義する。

def _default_comment(t: Dict[str, Any]) -> str:
    obj = str(t.get("object_name") or "").strip()
    role = str(t.get("role") or "").strip()
    lines = t.get("expected_lines") if isinstance(t.get("expected_lines"), list) else []
    refs = t.get("arxiv_refs") if isinstance(t.get("arxiv_refs"), list) else []
    notes = str(t.get("notes") or "").strip()
    parts: List[str] = []
    # 条件分岐: `role` を満たす経路を評価する。
    if role:
        parts.append(f"role={role}")

    # 条件分岐: `lines` を満たす経路を評価する。

    if lines:
        parts.append("lines=" + "/".join(str(x) for x in lines if str(x)))

    # 条件分岐: `refs` を満たす経路を評価する。

    if refs:
        parts.append("refs=" + ",".join(str(x) for x in refs if str(x)))

    # 条件分岐: `notes` を満たす経路を評価する。

    if notes:
        parts.append(notes)

    s = "; ".join(parts)
    return f"{obj}: {s}".strip(": ").strip()


# 関数: `update_targets_csv` の入出力契約と処理意図を定義する。

def update_targets_csv(*, catalog_json: Path, targets_csv: Path, overwrite: bool) -> Dict[str, Any]:
    json_targets = _load_target_catalog_json(catalog_json)
    csv_rows = _read_csv_rows(targets_csv)
    by_obsid: Dict[str, Dict[str, str]] = {str(r.get("obsid") or "").strip(): r for r in csv_rows if (r.get("obsid") or "").strip()}

    n_added = 0
    n_updated = 0

    for t in json_targets:
        obsid = str(t.get("obsid") or "").strip()
        # 条件分岐: `not obsid` を満たす経路を評価する。
        if not obsid:
            continue

        existing = by_obsid.get(obsid)

        z_opt = t.get("z_opt")
        z_sys = ""
        # 条件分岐: `isinstance(z_opt, (int, float))` を満たす経路を評価する。
        if isinstance(z_opt, (int, float)):
            z_sys = f"{float(z_opt):.8g}"

        new_row = {
            "obsid": obsid,
            "target_name": str(t.get("object_name") or "").strip(),
            "role": str(t.get("role") or "").strip(),
            "z_sys": z_sys,
            "instrument_prefer": "resolve",
            "comment": _default_comment(t),
            "remote_cat_hint": obsid[0] if obsid and obsid[0].isdigit() else "",
        }

        # 条件分岐: `existing is None` を満たす経路を評価する。
        if existing is None:
            by_obsid[obsid] = new_row
            n_added += 1
            continue

        # 条件分岐: `not overwrite` を満たす経路を評価する。

        if not overwrite:
            # Fill only missing essentials.
            if not (existing.get("role") or "").strip() and new_row["role"]:
                existing["role"] = new_row["role"]
                n_updated += 1

            # 条件分岐: `not (existing.get("z_sys") or "").strip() and new_row["z_sys"]` を満たす経路を評価する。

            if not (existing.get("z_sys") or "").strip() and new_row["z_sys"]:
                existing["z_sys"] = new_row["z_sys"]
                n_updated += 1

            # 条件分岐: `not (existing.get("target_name") or "").strip() and new_row["target_name"]` を満たす経路を評価する。

            if not (existing.get("target_name") or "").strip() and new_row["target_name"]:
                existing["target_name"] = new_row["target_name"]
                n_updated += 1

            # 条件分岐: `not (existing.get("remote_cat_hint") or "").strip() and new_row["remote_cat_h...` を満たす経路を評価する。

            if not (existing.get("remote_cat_hint") or "").strip() and new_row["remote_cat_hint"]:
                existing["remote_cat_hint"] = new_row["remote_cat_hint"]
                n_updated += 1

            continue

        # Overwrite mode: update core columns but keep comment if user wrote something non-empty.

        for k in ("target_name", "role", "z_sys", "remote_cat_hint"):
            # 条件分岐: `(existing.get(k) or "").strip() != (new_row.get(k) or "").strip()` を満たす経路を評価する。
            if (existing.get(k) or "").strip() != (new_row.get(k) or "").strip():
                existing[k] = new_row.get(k) or ""
                n_updated += 1

        # 条件分岐: `not (existing.get("comment") or "").strip()` を満たす経路を評価する。

        if not (existing.get("comment") or "").strip():
            existing["comment"] = new_row["comment"]

    out_rows = [by_obsid[k] for k in sorted(by_obsid.keys())]
    _write_csv(targets_csv, out_rows)

    return {
        "generated_utc": _utc_now(),
        "inputs": {"catalog_json": str(catalog_json)},
        "outputs": {"targets_csv": str(targets_csv)},
        "metrics": {"n_added": n_added, "n_updated": n_updated, "n_total": len(out_rows)},
    }


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[Iterable[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--catalog-json", default=str(_ROOT / "data" / "xrism" / "sources" / "xrism_target_catalog.json"))
    p.add_argument("--targets-csv", default=str(_ROOT / "output" / "private" / "xrism" / "xrism_targets_catalog.csv"))
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)

    result = update_targets_csv(
        catalog_json=Path(args.catalog_json),
        targets_csv=Path(args.targets_csv),
        overwrite=bool(args.overwrite),
    )

    worklog.append_event(
        {
            "task": "xrism_targets_catalog_sync",
            "inputs": {"catalog_json": Path(args.catalog_json)},
            "outputs": {"targets_csv": Path(args.targets_csv)},
            "metrics": dict(result.get("metrics") or {}),
        }
    )

    print(f"[ok] wrote: {args.targets_csv}")
    print(f"[ok] added={result['metrics']['n_added']} updated={result['metrics']['n_updated']} total={result['metrics']['n_total']}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())

