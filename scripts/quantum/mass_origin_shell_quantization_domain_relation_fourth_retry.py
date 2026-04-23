#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_shell_quantization_domain_relation_fourth_retry.py

Step 8.7.55.2.421:
Retry closure of the shell-quantization domain relation after the
domain-statement source inventory and wording audit.

Inputs:
  - output/public/quantum/mass_origin_shell_quantization_domain_statement_route_contract_metrics.json
  - output/public/quantum/mass_origin_shell_quantization_domain_statement_source_inventory_metrics.json
  - output/public/quantum/mass_origin_shell_quantization_domain_statement_wording_audit_metrics.json
  - output/public/quantum/mass_origin_shell_quantization_domain_relation_wording_audit_metrics.json

Outputs:
  - output/public/quantum/mass_origin_shell_quantization_domain_relation_fourth_retry_metrics.json
  - output/public/quantum/mass_origin_shell_quantization_domain_relation_fourth_retry_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

ROUTE_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_domain_statement_route_contract_metrics.json"
STATEMENT_SOURCE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_domain_statement_source_inventory_metrics.json"
STATEMENT_AUDIT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_domain_statement_wording_audit_metrics.json"
PREVIOUS_RELATION_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_domain_relation_wording_audit_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_domain_relation_fourth_retry_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_domain_relation_fourth_retry_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.421"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retry closure of the shell-quantization domain relation a fourth time.")
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
    domain_relation_available: bool,
    domain_relation_kind_or_none: str | None,
    domain_relation_without_new_free_parameters: bool,
    nonclosure_reason_or_none: str | None,
) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "shell_quantization_domain_relation_fourth_retry_complete",
            "status": "pass",
            "metric": "shell-quantization domain relation fourth retry complete",
            "value": 1.0,
            "note": "This step retries closure of the shell-quantization domain relation after the domain-statement branch.",
        },
        {
            "row_id": "shell_quantization_to_domain_relation_available",
            "status": "pass" if domain_relation_available else "reject",
            "metric": "shell-quantization to domain relation available after fourth retry",
            "value": 1.0 if domain_relation_available else 0.0,
            "note": (
                f"The current public pack already supplies the relation kind {domain_relation_kind_or_none}."
                if domain_relation_available
                else "The current public pack still lacks the shell-quantization domain-statement literal needed to define a cavity-domain relation."
            ),
        },
        {
            "row_id": "domain_relation_without_new_free_parameters",
            "status": "pass" if domain_relation_without_new_free_parameters else "reject",
            "metric": "domain relation stays inside no-new-free-parameter envelope after fourth retry",
            "value": 1.0 if domain_relation_without_new_free_parameters else 0.0,
            "note": (
                "The domain relation is already fixed without introducing an extra geometric scale."
                if domain_relation_without_new_free_parameters
                else "Any domain relation would still require a new scale or an unfrozen shell-quantization domain-statement literal."
            ),
        },
        {
            "row_id": "domain_relation_fourth_retry_nonclosure_reason",
            "status": "watch" if nonclosure_reason_or_none else "pass",
            "metric": "domain relation fourth-retry non-closure reason",
            "value": 0.0 if nonclosure_reason_or_none else 1.0,
            "note": (
                f"Current non-closure reason: {nonclosure_reason_or_none}."
                if nonclosure_reason_or_none
                else "The shell-quantization domain relation is fully closed."
            ),
        },
    ]


# 関数: metrics payload 全体を構成する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (ROUTE_CONTRACT_JSON, STATEMENT_SOURCE_JSON, STATEMENT_AUDIT_JSON, PREVIOUS_RELATION_JSON):
        _require_path(path)

    route_contract = _read_json(ROUTE_CONTRACT_JSON)
    statement_source = _read_json(STATEMENT_SOURCE_JSON)
    statement_audit = _read_json(STATEMENT_AUDIT_JSON)
    previous_relation = _read_json(PREVIOUS_RELATION_JSON)

    route_contract_summary = route_contract.get("summary", {})
    statement_source_summary = statement_source.get("summary", {})
    statement_audit_summary = statement_audit.get("summary", {})
    previous_relation_summary = previous_relation.get("summary", {})

    domain_relation_available = bool(statement_audit_summary.get("shell_quantization_domain_statement_available", False))
    domain_relation_kind_or_none = "shell_quantization_to_domain_relation" if domain_relation_available else None
    domain_relation_without_new_free_parameters = domain_relation_available
    nonclosure_reason_or_none = None if domain_relation_available else "shell_quantization_domain_statement_literal_absent"
    rows = _build_rows(
        domain_relation_available=domain_relation_available,
        domain_relation_kind_or_none=domain_relation_kind_or_none,
        domain_relation_without_new_free_parameters=domain_relation_without_new_free_parameters,
        nonclosure_reason_or_none=nonclosure_reason_or_none,
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "shell-quantization domain relation fourth retry",
        },
        "inputs": {
            "mass_origin_shell_quantization_domain_statement_route_contract_json": _relative_str(ROUTE_CONTRACT_JSON),
            "mass_origin_shell_quantization_domain_statement_source_inventory_json": _relative_str(STATEMENT_SOURCE_JSON),
            "mass_origin_shell_quantization_domain_statement_wording_audit_json": _relative_str(STATEMENT_AUDIT_JSON),
            "mass_origin_shell_quantization_domain_relation_wording_audit_json": _relative_str(PREVIOUS_RELATION_JSON),
        },
        "intent": "Retry closure of the shell-quantization domain relation after the domain-statement source inventory and wording audit.",
        "formulas": {
            "closure_rule": "the shell-quantization domain relation closes only if the shell-domain statement literal promotes without introducing a new scale",
            "current_absence": "the public shell-family coefficients still lack the shell-quantization domain-statement literal needed to define a cavity-domain relation",
        },
        "rows": rows,
        "summary": {
            "candidate_binding_route_id": route_contract_summary.get("selected_residual_binding_route_or_none"),
            "shell_quantization_to_domain_relation_available": domain_relation_available,
            "domain_relation_kind_or_none": domain_relation_kind_or_none,
            "domain_relation_without_new_free_parameters": domain_relation_without_new_free_parameters,
            "domain_relation_fourth_retry_nonclosure_reason_or_none": nonclosure_reason_or_none,
        },
        "decision": {
            "overall_status": (
                "shell_quantization_domain_relation_fourth_retry_closed"
                if domain_relation_available
                else "shell_quantization_domain_relation_fourth_retry_still_blocked"
            ),
            "keep_mass_origin_branch_blocked": not domain_relation_available,
            "candidate_binding_route_id": route_contract_summary.get("selected_residual_binding_route_or_none"),
            "shell_quantization_to_domain_relation_available": domain_relation_available,
            "domain_relation_kind_or_none": domain_relation_kind_or_none,
            "domain_relation_without_new_free_parameters": domain_relation_without_new_free_parameters,
            "domain_relation_fourth_retry_nonclosure_reason_or_none": nonclosure_reason_or_none,
            "hand_off_to_8_7_55_2_84": False,
            "next_required_artifacts": (
                ["boundary_radius_or_domain_proxy", "shell_quantization_reflective_cavity_rule"]
                if domain_relation_available
                else ["shell_quantization_domain_statement_literal", "shell_quantization_domain_statement", "shell_quantization_to_domain_relation"]
            ),
        },
        "evidence": {
            "shell_quantization_domain_statement_route_contract_summary": route_contract_summary,
            "shell_quantization_domain_statement_source_inventory_summary": statement_source_summary,
            "shell_quantization_domain_statement_wording_audit_summary": statement_audit_summary,
            "shell_quantization_domain_relation_wording_audit_summary": previous_relation_summary,
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
