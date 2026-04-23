#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_shell_quantization_domain_statement_literal_wording_audit.py

Step 8.7.55.2.427:
Audit whether the current public-canonical pack already fixes the missing
shell-quantization domain-statement literal without a new fit parameter.

Inputs:
  - output/public/quantum/mass_origin_shell_quantization_domain_statement_literal_source_inventory_metrics.json

Outputs:
  - output/public/quantum/mass_origin_shell_quantization_domain_statement_literal_wording_audit_metrics.json
  - output/public/quantum/mass_origin_shell_quantization_domain_statement_literal_wording_audit_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

LITERAL_SOURCE_INVENTORY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_domain_statement_literal_source_inventory_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_domain_statement_literal_wording_audit_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_domain_statement_literal_wording_audit_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.427"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit whether a shell-quantization domain-statement literal is already public canonical.")
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
    literal_available: bool,
    literal_without_new_free_parameters: bool,
    missing_inputs: List[str],
) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "shell_quantization_domain_statement_literal_wording_audit_complete",
            "status": "pass",
            "metric": "shell-quantization domain-statement-literal wording audit complete",
            "value": 1.0,
            "note": "This step audits whether the shell-quantization domain-statement literal is already public canonical.",
        },
        {
            "row_id": "shell_quantization_domain_statement_literal_available",
            "status": "pass" if literal_available else "reject",
            "metric": "shell-quantization domain-statement literal available",
            "value": 1.0 if literal_available else 0.0,
            "note": (
                "A shell-quantization domain-statement literal is now public canonical."
                if literal_available
                else f"Missing inputs: {missing_inputs}."
            ),
        },
        {
            "row_id": "shell_quantization_domain_statement_literal_without_new_free_parameters",
            "status": "pass" if literal_without_new_free_parameters else "reject",
            "metric": "shell-quantization domain-statement literal stays inside no-new-free-parameter envelope",
            "value": 1.0 if literal_without_new_free_parameters else 0.0,
            "note": (
                "The shell-quantization domain-statement literal closes without a new fit parameter."
                if literal_without_new_free_parameters
                else "The current public pack still lacks a promotable shell-quantization domain-statement phrase fragment."
            ),
        },
    ]


# 関数: metrics payload 全体を構成する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    _require_path(LITERAL_SOURCE_INVENTORY_JSON)
    literal_source_inventory = _read_json(LITERAL_SOURCE_INVENTORY_JSON)
    literal_source_summary = literal_source_inventory.get("summary", {})
    present_sources = [str(item) for item in literal_source_summary.get("present_domain_statement_literal_sources", [])]

    phrase_fragment_available = False
    no_new_free_parameter_wording_available = "boundary_condition_quantization_note" in present_sources
    literal_available = bool(phrase_fragment_available)
    literal_kind_or_none = "shell_quantization_domain_statement_literal" if literal_available else None
    literal_without_new_free_parameters = bool(literal_available and no_new_free_parameter_wording_available)
    missing_inputs: List[str] = []

    if not phrase_fragment_available:
        missing_inputs.append("shell_quantization_domain_statement_phrase_fragment")

    rows = _build_rows(
        literal_available=literal_available,
        literal_without_new_free_parameters=literal_without_new_free_parameters,
        missing_inputs=missing_inputs,
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "shell-quantization domain-statement-literal wording audit",
        },
        "inputs": {
            "mass_origin_shell_quantization_domain_statement_literal_source_inventory_json": _relative_str(
                LITERAL_SOURCE_INVENTORY_JSON
            ),
        },
        "intent": "Audit whether the current public-canonical pack already fixes the missing shell-quantization domain-statement literal without a new fit parameter.",
        "formulas": {
            "audit_rule": "shell_quantization_domain_statement_literal_available iff the public pack exposes a shell-quantization domain-statement phrase fragment inside the no-new-free-parameter envelope",
            "current_absence": "the current pack still lacks the shell-quantization domain-statement phrase fragment, so the literal cannot yet promote",
        },
        "rows": rows,
        "summary": {
            "shell_quantization_domain_statement_literal_available": literal_available,
            "domain_statement_literal_kind_or_none": literal_kind_or_none,
            "domain_statement_literal_without_new_free_parameters": literal_without_new_free_parameters,
            "domain_statement_literal_missing_inputs": missing_inputs,
        },
        "decision": {
            "overall_status": "shell_quantization_domain_statement_literal_wording_audit_frozen_absent",
            "keep_mass_origin_branch_blocked": True,
            "shell_quantization_domain_statement_literal_available": literal_available,
            "domain_statement_literal_kind_or_none": literal_kind_or_none,
            "domain_statement_literal_without_new_free_parameters": literal_without_new_free_parameters,
            "hand_off_to_8_7_55_2_84": False,
            "next_required_artifacts": [
                "shell_quantization_domain_statement_phrase_fragment",
                "shell_quantization_domain_statement_literal",
                "shell_quantization_domain_statement",
                "shell_quantization_to_domain_relation",
            ],
        },
        "evidence": {
            "shell_quantization_domain_statement_literal_source_inventory_summary": literal_source_summary,
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
