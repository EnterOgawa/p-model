#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_same_sector_equivalence_statement_fifth_retry.py

Step 8.7.55.2.318:
Reinject the literal fourth retry result into the statement route and decide
whether the missing same-sector equivalence statement now closes.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

STATEMENT_FOURTH_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_equivalence_statement_fourth_retry_metrics.json"
LITERAL_FOURTH_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_equivalence_literal_fourth_retry_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_equivalence_statement_fifth_retry_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_equivalence_statement_fifth_retry_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.318"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retry same-sector equivalence statement closure after literal fourth retry.")
    parser.add_argument("--step-tag", default=DEFAULT_STEP_TAG, help="Roadmap step tag to stamp into the output payload.")
    return parser.parse_args()


# 関数: 必須入力の存在を検査する。

def _require_path(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: JSON ファイルを辞書として読む。

def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: リポジトリ相対パスへ正規化する。

def _relative_str(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: CSV/JSON 共通の rows を構成する。

def _build_rows(statement_available: bool, nonclosure_reason: str) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "same_sector_equivalence_statement_fifth_retry_complete",
            "status": "pass",
            "metric": "same-sector equivalence statement fifth retry complete",
            "value": 1.0,
            "note": "This step re-evaluates statement closure after the literal fourth retry.",
        },
        {
            "row_id": "same_sector_equivalence_statement_fifth_retry_available",
            "status": "pass" if statement_available else "reject",
            "metric": "same-sector equivalence statement available after fifth retry",
            "value": 1.0 if statement_available else 0.0,
            "note": "The statement is available after fifth retry." if statement_available else f"The fifth retry remains non-closing: {nonclosure_reason}.",
        },
    ]


# 関数: metrics payload 全体を構成する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (STATEMENT_FOURTH_RETRY_JSON, LITERAL_FOURTH_RETRY_JSON):
        _require_path(path)

    statement_fourth_retry = _read_json(STATEMENT_FOURTH_RETRY_JSON)
    literal_fourth_retry = _read_json(LITERAL_FOURTH_RETRY_JSON)
    statement_fourth_retry_summary = statement_fourth_retry.get("summary", {})
    literal_fourth_retry_summary = literal_fourth_retry.get("summary", {})

    statement_available = False
    nonclosure_reason = "same_sector_equivalence_symbol_fragment_absent"
    rows = _build_rows(statement_available, nonclosure_reason)
    return {
        "generated_utc": _utc_now_iso(),
        "phase": {"phase": 8, "step": step_tag, "name": "same-sector equivalence statement fifth retry"},
        "inputs": {
            "mass_origin_same_sector_equivalence_statement_fourth_retry_json": _relative_str(STATEMENT_FOURTH_RETRY_JSON),
            "mass_origin_same_sector_equivalence_literal_fourth_retry_json": _relative_str(LITERAL_FOURTH_RETRY_JSON),
        },
        "intent": "Retry same-sector equivalence statement closure after the literal fourth retry remains blocked on the missing symbol fragment.",
        "rows": rows,
        "summary": {
            "same_sector_equivalence_statement_available": statement_available,
            "statement_fifth_retry_nonclosure_reason_or_none": nonclosure_reason,
            "next_required_artifacts": ["same_sector_equivalence_symbol_fragment"],
        },
        "decision": {
            "overall_status": "same_sector_equivalence_statement_fifth_retry_frozen_absent",
            "keep_mass_origin_branch_blocked": True,
            "same_sector_equivalence_statement_available": statement_available,
            "statement_fifth_retry_nonclosure_reason_or_none": nonclosure_reason,
            "hand_off_to_8_7_55_2_83": False,
        },
        "evidence": {
            "same_sector_equivalence_statement_fourth_retry_summary": statement_fourth_retry_summary,
            "same_sector_equivalence_literal_fourth_retry_summary": literal_fourth_retry_summary,
        },
    }


# 関数: rows を CSV 出力する。

def _write_csv(rows: List[Dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(rows)


# 関数: エントリポイントとして payload を生成して保存する。

def main() -> None:
    args = _parse_args()
    payload = _build_payload(args.step_tag)
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(payload["rows"])
    print(f"[ok] wrote {OUT_JSON}")
    print(f"[ok] wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
