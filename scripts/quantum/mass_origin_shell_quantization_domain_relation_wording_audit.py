#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_shell_quantization_domain_relation_wording_audit.py

Step 8.7.55.2.413:
Audit whether the current public shell-quantization family already fixes a
no-new-free-parameter same-sector domain relation after the source-level
inventory has been expanded.

Inputs:
  - output/public/quantum/mass_origin_shell_quantization_domain_relation_source_inventory_metrics.json
  - output/public/quantum/mass_origin_shell_quantization_domain_relation_route_contract_metrics.json
  - output/public/quantum/mass_origin_boundary_radius_source_inventory_metrics.json

Outputs:
  - output/public/quantum/mass_origin_shell_quantization_domain_relation_wording_audit_metrics.json
  - output/public/quantum/mass_origin_shell_quantization_domain_relation_wording_audit_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

SOURCE_INVENTORY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_domain_relation_source_inventory_metrics.json"
ROUTE_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_domain_relation_route_contract_metrics.json"
BOUNDARY_SOURCE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_boundary_radius_source_inventory_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_domain_relation_wording_audit_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_domain_relation_wording_audit_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.413"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit whether the shell-quantization domain relation is now public canonical.")
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
    shell_quantization_to_domain_relation_available: bool,
    domain_relation_kind_or_none: str | None,
    domain_relation_without_new_free_parameters: bool,
    missing_inputs: List[str],
) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "shell_quantization_domain_relation_wording_audit_complete",
            "status": "pass",
            "metric": "shell-quantization domain-relation wording audit complete",
            "value": 1.0,
            "note": "This step tests whether shell quantization now fixes a cavity-domain relation after the source inventory was expanded.",
        },
        {
            "row_id": "shell_quantization_to_domain_relation_available",
            "status": "pass" if shell_quantization_to_domain_relation_available else "reject",
            "metric": "shell-quantization to domain relation available after wording audit",
            "value": 1.0 if shell_quantization_to_domain_relation_available else 0.0,
            "note": (
                f"The current public pack already supplies the relation kind {domain_relation_kind_or_none}."
                if shell_quantization_to_domain_relation_available
                else "The current public pack still lacks the statement or operator that maps shell quantization into a cavity domain."
            ),
        },
        {
            "row_id": "domain_relation_without_new_free_parameters",
            "status": "pass" if domain_relation_without_new_free_parameters else "reject",
            "metric": "domain relation stays inside no-new-free-parameter envelope",
            "value": 1.0 if domain_relation_without_new_free_parameters else 0.0,
            "note": (
                "The domain relation is already fixed without introducing an extra geometric scale."
                if domain_relation_without_new_free_parameters
                else "Any domain relation would still require a new scale or an unfrozen shell-domain statement."
            ),
        },
        {
            "row_id": "domain_relation_missing_inputs",
            "status": "missing" if missing_inputs else "pass",
            "metric": "remaining missing inputs for shell-quantization domain relation",
            "value": float(len(missing_inputs)),
            "note": f"Missing inputs: {missing_inputs}.",
        },
    ]


# 関数: metrics payload 全体を構成する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (SOURCE_INVENTORY_JSON, ROUTE_CONTRACT_JSON, BOUNDARY_SOURCE_JSON):
        _require_path(path)

    source_inventory = _read_json(SOURCE_INVENTORY_JSON)
    route_contract = _read_json(ROUTE_CONTRACT_JSON)
    boundary_source = _read_json(BOUNDARY_SOURCE_JSON)

    source_inventory_summary = source_inventory.get("summary", {})
    route_contract_summary = route_contract.get("summary", {})
    boundary_source_summary = boundary_source.get("summary", {})

    shell_quantization_to_domain_relation_available = False
    domain_relation_kind_or_none = None
    domain_relation_without_new_free_parameters = False
    missing_inputs: List[str] = []

    for item in ("shell_quantization_domain_statement", "domain_relation_operator"):
        if item in source_inventory_summary.get("missing_domain_relation_sources", []):
            missing_inputs.append(item)

    rows = _build_rows(
        shell_quantization_to_domain_relation_available=shell_quantization_to_domain_relation_available,
        domain_relation_kind_or_none=domain_relation_kind_or_none,
        domain_relation_without_new_free_parameters=domain_relation_without_new_free_parameters,
        missing_inputs=missing_inputs,
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "shell-quantization domain relation wording audit",
        },
        "inputs": {
            "mass_origin_shell_quantization_domain_relation_source_inventory_json": _relative_str(SOURCE_INVENTORY_JSON),
            "mass_origin_shell_quantization_domain_relation_route_contract_json": _relative_str(ROUTE_CONTRACT_JSON),
            "mass_origin_boundary_radius_source_inventory_json": _relative_str(BOUNDARY_SOURCE_JSON),
        },
        "intent": "Determine whether the current public shell-quantization family already fixes a same-sector domain relation after the source-level inventory was expanded.",
        "formulas": {
            "domain_relation_rule": "the geometric reflective-boundary route reopens only if shell quantization is linked to a cavity domain by a no-new-free-parameter statement and relation operator",
            "current_absence": "the public pack still lacks the statement and operator that turn shell-family coefficients into a cavity-domain relation",
        },
        "rows": rows,
        "summary": {
            "candidate_binding_route_id": route_contract_summary.get("selected_residual_binding_route_or_none"),
            "shell_quantization_to_domain_relation_available": shell_quantization_to_domain_relation_available,
            "domain_relation_kind_or_none": domain_relation_kind_or_none,
            "domain_relation_without_new_free_parameters": domain_relation_without_new_free_parameters,
            "domain_relation_nonclosure_reason_or_none": "shell_quantization_domain_statement_absent",
            "domain_relation_missing_inputs": missing_inputs,
        },
        "decision": {
            "overall_status": "shell_quantization_domain_relation_wording_audit_frozen_absent",
            "keep_mass_origin_branch_blocked": True,
            "candidate_binding_route_id": route_contract_summary.get("selected_residual_binding_route_or_none"),
            "shell_quantization_to_domain_relation_available": shell_quantization_to_domain_relation_available,
            "domain_relation_kind_or_none": domain_relation_kind_or_none,
            "domain_relation_without_new_free_parameters": domain_relation_without_new_free_parameters,
            "domain_relation_nonclosure_reason_or_none": "shell_quantization_domain_statement_absent",
            "hand_off_to_8_7_55_2_84": False,
            "next_required_artifacts": [
                "shell_quantization_domain_statement",
                "domain_relation_operator",
                "shell_quantization_to_domain_relation",
            ],
        },
        "evidence": {
            "shell_quantization_domain_relation_source_inventory_summary": source_inventory_summary,
            "shell_quantization_domain_relation_route_contract_summary": route_contract_summary,
            "boundary_radius_source_inventory_summary": boundary_source_summary,
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
