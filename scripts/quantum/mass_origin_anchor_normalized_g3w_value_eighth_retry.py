#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_anchor_normalized_g3w_value_eighth_retry.py

Step 8.7.55.2.309:
Reinject the equivalence-rule eighth retry result into the preferred g3w route
and determine whether the anchor-normalized public g3w value now closes.

Inputs:
  - output/public/quantum/mass_origin_anchor_normalized_g3w_value_seventh_retry_metrics.json
  - output/public/quantum/mass_origin_same_sector_equivalence_rule_eighth_retry_metrics.json

Outputs:
  - output/public/quantum/mass_origin_anchor_normalized_g3w_value_eighth_retry_metrics.json
  - output/public/quantum/mass_origin_anchor_normalized_g3w_value_eighth_retry_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

G3W_SEVENTH_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_value_seventh_retry_metrics.json"
RULE_EIGHTH_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_equivalence_rule_eighth_retry_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_value_eighth_retry_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_value_eighth_retry_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.309"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retry anchor-normalized public g3w value closure for the eighth time.")
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

def _build_rows(*, public_value_available: bool, nonclosure_reason: str | None) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "anchor_normalized_g3w_value_eighth_retry_complete",
            "status": "pass",
            "metric": "anchor-normalized public g3w value eighth retry complete",
            "value": 1.0,
            "note": "This step retries the preferred g3w route after the equivalence-rule eighth retry.",
        },
        {
            "row_id": "anchor_normalized_g3w_value_eighth_retry_public_value_available",
            "status": "pass" if public_value_available else "reject",
            "metric": "anchor-normalized public g3w value available after eighth retry",
            "value": 1.0 if public_value_available else 0.0,
            "note": (
                "The anchor-normalized public g3w value is available after eighth retry."
                if public_value_available
                else f"The eighth retry remains non-closing: {nonclosure_reason}."
            ),
        },
    ]


# 関数: metrics payload 全体を構成する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (G3W_SEVENTH_RETRY_JSON, RULE_EIGHTH_RETRY_JSON):
        _require_path(path)

    g3w_seventh_retry = _read_json(G3W_SEVENTH_RETRY_JSON)
    rule_eighth_retry = _read_json(RULE_EIGHTH_RETRY_JSON)

    g3w_seventh_retry_summary = g3w_seventh_retry.get("summary", {})
    rule_eighth_retry_summary = rule_eighth_retry.get("summary", {})

    public_value_available = bool(rule_eighth_retry_summary.get("same_sector_equivalence_rule_available", False))
    r3_target_available = public_value_available
    r3_target_value_or_none = None
    nonclosure_reason = "same_sector_equivalence_terminal_glyph_absent"
    rows = _build_rows(public_value_available=public_value_available, nonclosure_reason=nonclosure_reason)

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {"phase": 8, "step": step_tag, "name": "anchor-normalized g3w public value eighth retry"},
        "inputs": {
            "mass_origin_anchor_normalized_g3w_value_seventh_retry_json": _relative_str(
                G3W_SEVENTH_RETRY_JSON
            ),
            "mass_origin_same_sector_equivalence_rule_eighth_retry_json": _relative_str(
                RULE_EIGHTH_RETRY_JSON
            ),
        },
        "intent": "Retry whether the anchor-normalized public g3w value can now close after the equivalence-rule eighth retry.",
        "rows": rows,
        "summary": {
            "anchor_normalized_g3w_public_value_available": public_value_available,
            "r3_target_available": r3_target_available,
            "r3_target_value_or_none": r3_target_value_or_none,
            "g3w_eighth_retry_nonclosure_reason_or_none": nonclosure_reason,
        },
        "decision": {
            "overall_status": "anchor_normalized_g3w_value_eighth_retry_frozen_absent",
            "keep_mass_origin_branch_blocked": True,
            "anchor_normalized_g3w_public_value_available": public_value_available,
            "r3_target_available": r3_target_available,
            "r3_target_value_or_none": r3_target_value_or_none,
            "g3w_eighth_retry_nonclosure_reason_or_none": nonclosure_reason,
            "hand_off_to_8_7_55_2_83": False,
        },
        "evidence": {
            "anchor_normalized_g3w_value_seventh_retry_summary": g3w_seventh_retry_summary,
            "same_sector_equivalence_rule_eighth_retry_summary": rule_eighth_retry_summary,
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
