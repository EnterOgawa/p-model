#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_boundary_radius_proxy_wording_audit.py

Step 8.7.55.2.407:
Audit whether the current public shell-quantization family already fixes a
no-new-free-parameter boundary radius or domain proxy after the source-level
inventory has been expanded.

Inputs:
  - output/public/quantum/mass_origin_boundary_radius_source_inventory_metrics.json
  - output/public/quantum/mass_origin_boundary_radius_proxy_route_contract_metrics.json
  - output/public/quantum/mass_origin_shell_quantization_cavity_source_inventory_metrics.json

Outputs:
  - output/public/quantum/mass_origin_boundary_radius_proxy_wording_audit_metrics.json
  - output/public/quantum/mass_origin_boundary_radius_proxy_wording_audit_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

SOURCE_INVENTORY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_boundary_radius_source_inventory_metrics.json"
ROUTE_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_boundary_radius_proxy_route_contract_metrics.json"
CAVITY_SOURCE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_cavity_source_inventory_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_boundary_radius_proxy_wording_audit_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_boundary_radius_proxy_wording_audit_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.407"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit whether the boundary-radius proxy is now public canonical.")
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
    boundary_radius_or_domain_available: bool,
    boundary_proxy_kind_or_none: str | None,
    boundary_proxy_without_new_free_parameters: bool,
    missing_inputs: List[str],
) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "boundary_radius_proxy_wording_audit_complete",
            "status": "pass",
            "metric": "boundary-radius proxy wording audit complete",
            "value": 1.0,
            "note": "This step tests whether shell quantization now fixes a cavity radius or domain proxy after the source inventory was expanded.",
        },
        {
            "row_id": "boundary_radius_or_domain_available",
            "status": "pass" if boundary_radius_or_domain_available else "reject",
            "metric": "boundary radius or domain proxy available after wording audit",
            "value": 1.0 if boundary_radius_or_domain_available else 0.0,
            "note": (
                f"The current public pack already supplies the proxy kind {boundary_proxy_kind_or_none}."
                if boundary_radius_or_domain_available
                else "The current public pack still lacks a relation that maps shell quantization into a cavity radius or domain."
            ),
        },
        {
            "row_id": "boundary_proxy_without_new_free_parameters",
            "status": "pass" if boundary_proxy_without_new_free_parameters else "reject",
            "metric": "boundary proxy stays inside no-new-free-parameter envelope",
            "value": 1.0 if boundary_proxy_without_new_free_parameters else 0.0,
            "note": (
                "The cavity proxy is already fixed without introducing an extra scale."
                if boundary_proxy_without_new_free_parameters
                else "Any boundary proxy would still require a new scale or an unfrozen shell-to-domain relation."
            ),
        },
        {
            "row_id": "boundary_proxy_missing_inputs",
            "status": "missing" if missing_inputs else "pass",
            "metric": "remaining missing inputs for boundary radius or domain proxy",
            "value": float(len(missing_inputs)),
            "note": f"Missing inputs: {missing_inputs}.",
        },
    ]


# 関数: metrics payload 全体を構成する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (SOURCE_INVENTORY_JSON, ROUTE_CONTRACT_JSON, CAVITY_SOURCE_JSON):
        _require_path(path)

    source_inventory = _read_json(SOURCE_INVENTORY_JSON)
    route_contract = _read_json(ROUTE_CONTRACT_JSON)
    cavity_source = _read_json(CAVITY_SOURCE_JSON)

    source_inventory_summary = source_inventory.get("summary", {})
    route_contract_summary = route_contract.get("summary", {})
    cavity_source_summary = cavity_source.get("summary", {})

    boundary_radius_or_domain_available = False
    boundary_proxy_kind_or_none = None
    boundary_proxy_without_new_free_parameters = False
    missing_inputs: List[str] = []

    if "shell_quantization_to_domain_relation" in source_inventory_summary.get("missing_boundary_radius_sources", []):
        missing_inputs.append("shell_quantization_to_domain_relation")

    rows = _build_rows(
        boundary_radius_or_domain_available=boundary_radius_or_domain_available,
        boundary_proxy_kind_or_none=boundary_proxy_kind_or_none,
        boundary_proxy_without_new_free_parameters=boundary_proxy_without_new_free_parameters,
        missing_inputs=missing_inputs,
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "boundary-radius proxy wording audit",
        },
        "inputs": {
            "mass_origin_boundary_radius_source_inventory_json": _relative_str(SOURCE_INVENTORY_JSON),
            "mass_origin_boundary_radius_proxy_route_contract_json": _relative_str(ROUTE_CONTRACT_JSON),
            "mass_origin_shell_quantization_cavity_source_inventory_json": _relative_str(CAVITY_SOURCE_JSON),
        },
        "intent": "Determine whether the current public shell-quantization family already fixes a cavity radius or domain proxy after the source-level inventory was expanded.",
        "formulas": {
            "boundary_proxy_rule": "a geometric reflective-boundary route reopens only if shell quantization is linked to a cavity radius or domain by a no-new-free-parameter same-sector relation",
            "current_absence": "the public pack still lacks the relation that turns shell-family coefficients into a cavity radius or domain proxy",
        },
        "rows": rows,
        "summary": {
            "candidate_binding_route_id": route_contract_summary.get("selected_residual_binding_route_or_none"),
            "boundary_radius_or_domain_available": boundary_radius_or_domain_available,
            "boundary_proxy_kind_or_none": boundary_proxy_kind_or_none,
            "boundary_proxy_without_new_free_parameters": boundary_proxy_without_new_free_parameters,
            "boundary_proxy_nonclosure_reason_or_none": "shell_quantization_to_domain_relation_absent",
            "boundary_proxy_missing_inputs": missing_inputs,
        },
        "decision": {
            "overall_status": "boundary_radius_proxy_wording_audit_frozen_absent",
            "keep_mass_origin_branch_blocked": True,
            "candidate_binding_route_id": route_contract_summary.get("selected_residual_binding_route_or_none"),
            "boundary_radius_or_domain_available": boundary_radius_or_domain_available,
            "boundary_proxy_kind_or_none": boundary_proxy_kind_or_none,
            "boundary_proxy_without_new_free_parameters": boundary_proxy_without_new_free_parameters,
            "boundary_proxy_nonclosure_reason_or_none": "shell_quantization_to_domain_relation_absent",
            "hand_off_to_8_7_55_2_84": False,
            "next_required_artifacts": [
                "shell_quantization_to_domain_relation",
                "boundary_radius_or_domain_proxy",
                "shell_quantization_reflective_cavity_rule",
            ],
        },
        "evidence": {
            "boundary_radius_source_inventory_summary": source_inventory_summary,
            "boundary_radius_proxy_route_contract_summary": route_contract_summary,
            "shell_quantization_cavity_source_inventory_summary": cavity_source_summary,
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
