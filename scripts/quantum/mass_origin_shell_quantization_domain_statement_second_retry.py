#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_shell_quantization_domain_statement_second_retry.py

Step 8.7.55.2.428:
Retry shell-quantization domain-statement closure after the literal wording
audit.

Inputs:
  - output/public/quantum/mass_origin_shell_quantization_domain_statement_wording_audit_metrics.json
  - output/public/quantum/mass_origin_shell_quantization_domain_statement_literal_wording_audit_metrics.json

Outputs:
  - output/public/quantum/mass_origin_shell_quantization_domain_statement_second_retry_metrics.json
  - output/public/quantum/mass_origin_shell_quantization_domain_statement_second_retry_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

STATEMENT_WORDING_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_domain_statement_wording_audit_metrics.json"
LITERAL_WORDING_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_domain_statement_literal_wording_audit_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_domain_statement_second_retry_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_domain_statement_second_retry_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.428"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retry the shell-quantization domain statement after the literal wording audit.")
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


# 関数: rows を構成する。

def _build_rows(
    *,
    statement_available: bool,
    nonclosure_reason: str | None,
) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "shell_quantization_domain_statement_second_retry_complete",
            "status": "pass",
            "metric": "shell-quantization domain statement second retry complete",
            "value": 1.0,
            "note": "This step retries the shell-quantization domain statement after the literal wording audit.",
        },
        {
            "row_id": "shell_quantization_domain_statement_second_retry_available",
            "status": "pass" if statement_available else "reject",
            "metric": "shell-quantization domain statement available after second retry",
            "value": 1.0 if statement_available else 0.0,
            "note": (
                "The shell-quantization domain statement is now public canonical."
                if statement_available
                else f"The retry remains non-closing: {nonclosure_reason}."
            ),
        },
    ]


# 関数: metrics payload 全体を構成する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (STATEMENT_WORDING_JSON, LITERAL_WORDING_JSON):
        _require_path(path)

    statement_wording = _read_json(STATEMENT_WORDING_JSON)
    literal_wording = _read_json(LITERAL_WORDING_JSON)

    statement_wording_summary = statement_wording.get("summary", {})
    literal_wording_summary = literal_wording.get("summary", {})

    literal_available = bool(literal_wording_summary.get("shell_quantization_domain_statement_literal_available", False))
    statement_available = bool(literal_available)
    literal_missing_inputs = [str(item) for item in literal_wording_summary.get("domain_statement_literal_missing_inputs", [])]

    if "shell_quantization_domain_statement_phrase_fragment" in literal_missing_inputs:
        nonclosure_reason = "shell_quantization_domain_statement_phrase_fragment_absent"
    else:
        nonclosure_reason = "shell_quantization_domain_statement_literal_absent"

    rows = _build_rows(statement_available=statement_available, nonclosure_reason=nonclosure_reason)

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "shell-quantization domain-statement second retry",
        },
        "inputs": {
            "mass_origin_shell_quantization_domain_statement_wording_audit_json": _relative_str(STATEMENT_WORDING_JSON),
            "mass_origin_shell_quantization_domain_statement_literal_wording_audit_json": _relative_str(LITERAL_WORDING_JSON),
        },
        "intent": "Retry the shell-quantization domain-statement route after the literal wording audit.",
        "formulas": {
            "retry_rule": "shell_quantization_domain_statement_available iff the literal wording audit promotes a shell-quantization domain-statement literal",
            "current_absence": "the literal wording audit still lacks a shell-quantization domain-statement phrase fragment, so the statement retry remains blocked one layer below the statement itself",
        },
        "rows": rows,
        "summary": {
            "shell_quantization_domain_statement_available": statement_available,
            "domain_statement_kind_or_none": "shell_quantization_domain_statement" if statement_available else None,
            "domain_statement_without_new_free_parameters": statement_available,
            "domain_statement_second_retry_nonclosure_reason_or_none": nonclosure_reason,
        },
        "decision": {
            "overall_status": "shell_quantization_domain_statement_second_retry_frozen_absent",
            "keep_mass_origin_branch_blocked": True,
            "shell_quantization_domain_statement_available": statement_available,
            "domain_statement_kind_or_none": "shell_quantization_domain_statement" if statement_available else None,
            "domain_statement_without_new_free_parameters": statement_available,
            "domain_statement_second_retry_nonclosure_reason_or_none": nonclosure_reason,
            "hand_off_to_8_7_55_2_84": False,
            "next_required_artifacts": [
                "shell_quantization_domain_statement_phrase_fragment",
                "shell_quantization_domain_statement_literal",
                "shell_quantization_domain_statement",
                "shell_quantization_to_domain_relation",
            ],
        },
        "evidence": {
            "shell_quantization_domain_statement_wording_audit_summary": statement_wording_summary,
            "shell_quantization_domain_statement_literal_wording_audit_summary": literal_wording_summary,
        },
    }


# 関数: rows を CSV 出力する。

def _write_csv(rows: List[Dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(rows)


# 関数: JSON を整形出力する。

def _write_json(payload: Dict[str, Any]) -> None:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# 関数: エントリポイントとして step を実行する。

def main() -> None:
    args = _parse_args()
    payload = _build_payload(args.step_tag)
    _write_json(payload)
    _write_csv(payload["rows"])
    print(f"[ok] wrote {OUT_JSON}")
    print(f"[ok] wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
