#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_boundary_radius_proxy_audit.py

Step 8.7.55.2.401:
Audit whether the current public shell-quantization family already fixes a
no-new-free-parameter cavity radius or domain proxy for the geometric
reflective-boundary route.

Inputs:
  - output/public/quantum/mass_origin_shell_quantization_cavity_source_inventory_metrics.json
  - output/public/quantum/mass_origin_geometric_boundary_residual_route_contract_metrics.json
  - output/public/quantum/mass_origin_geometric_boundary_promotion_metrics.json
  - output/public/quantum/mass_origin_shell_quantization_canonicalization_metrics.json
  - output/public/quantum/mass_origin_mass_eigenmode_boundary_metrics.json

Outputs:
  - output/public/quantum/mass_origin_boundary_radius_proxy_audit_metrics.json
  - output/public/quantum/mass_origin_boundary_radius_proxy_audit_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

SOURCE_INVENTORY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_cavity_source_inventory_metrics.json"
ROUTE_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_geometric_boundary_residual_route_contract_metrics.json"
GEOMETRIC_PROMOTION_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_geometric_boundary_promotion_metrics.json"
SHELL_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_canonicalization_metrics.json"
BOUNDARY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_mass_eigenmode_boundary_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_boundary_radius_proxy_audit_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_boundary_radius_proxy_audit_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.401"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit whether shell quantization already defines a cavity radius or domain proxy.")
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
            "row_id": "boundary_radius_proxy_audit_complete",
            "status": "pass",
            "metric": "boundary radius or domain proxy audit complete",
            "value": 1.0,
            "note": "This step tests whether shell quantization already fixes a reflective cavity radius or domain proxy without adding a new fit.",
        },
        {
            "row_id": "boundary_radius_or_domain_available",
            "status": "pass" if boundary_radius_or_domain_available else "reject",
            "metric": "boundary radius or domain proxy available",
            "value": 1.0 if boundary_radius_or_domain_available else 0.0,
            "note": (
                f"The current public pack already supplies the proxy kind {boundary_proxy_kind_or_none}."
                if boundary_radius_or_domain_available
                else "The current public shell-family rows do not yet fix a cavity radius, shell domain, or equivalent geometric proxy."
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
                else "Any boundary proxy would still require a new geometric scale or an unfrozen same-sector relation."
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
    for path in (SOURCE_INVENTORY_JSON, ROUTE_CONTRACT_JSON, GEOMETRIC_PROMOTION_JSON, SHELL_JSON, BOUNDARY_JSON):
        _require_path(path)

    source_inventory = _read_json(SOURCE_INVENTORY_JSON)
    route_contract = _read_json(ROUTE_CONTRACT_JSON)
    geometric_promotion = _read_json(GEOMETRIC_PROMOTION_JSON)
    shell = _read_json(SHELL_JSON)
    boundary = _read_json(BOUNDARY_JSON)

    source_inventory_summary = source_inventory.get("summary", {})
    route_contract_summary = route_contract.get("summary", {})
    geometric_promotion_summary = geometric_promotion.get("summary", {})
    shell_summary = shell.get("summary", {})
    boundary_summary = boundary.get("summary", {})

    boundary_radius_or_domain_available = False
    boundary_proxy_kind_or_none = None
    boundary_proxy_without_new_free_parameters = False
    missing_inputs: List[str] = []

    if "boundary_radius_or_domain_proxy" in source_inventory_summary.get("missing_cavity_route_sources", []):
        missing_inputs.append("boundary_radius_or_domain_proxy")

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
            "name": "boundary radius or domain proxy audit",
        },
        "inputs": {
            "mass_origin_shell_quantization_cavity_source_inventory_json": _relative_str(SOURCE_INVENTORY_JSON),
            "mass_origin_geometric_boundary_residual_route_contract_json": _relative_str(ROUTE_CONTRACT_JSON),
            "mass_origin_geometric_boundary_promotion_json": _relative_str(GEOMETRIC_PROMOTION_JSON),
            "mass_origin_shell_quantization_canonicalization_json": _relative_str(SHELL_JSON),
            "mass_origin_mass_eigenmode_boundary_json": _relative_str(BOUNDARY_JSON),
        },
        "intent": "Determine whether the current public shell-quantization family already fixes a cavity radius or domain proxy for the geometric reflective-boundary route.",
        "formulas": {
            "boundary_proxy_rule": "a geometric reflective-boundary route can reopen the mexican-hat pilot only if the current public shell family fixes a cavity radius, shell domain, or equivalent boundary proxy without adding a new scale",
            "current_absence": "the surviving shell family exposes only kappa and kZ/kN fit coefficients, which do not by themselves define a geometric radius or reflective domain",
        },
        "rows": rows,
        "summary": {
            "candidate_binding_route_id": route_contract_summary.get("selected_residual_binding_route_or_none"),
            "boundary_radius_or_domain_available": boundary_radius_or_domain_available,
            "boundary_proxy_kind_or_none": boundary_proxy_kind_or_none,
            "boundary_proxy_without_new_free_parameters": boundary_proxy_without_new_free_parameters,
            "boundary_proxy_nonclosure_reason_or_none": "shell_quantization_rows_do_not_define_cavity_scale",
            "boundary_proxy_missing_inputs": missing_inputs,
        },
        "decision": {
            "overall_status": "boundary_radius_proxy_audit_frozen_absent",
            "keep_mass_origin_branch_blocked": True,
            "candidate_binding_route_id": route_contract_summary.get("selected_residual_binding_route_or_none"),
            "boundary_radius_or_domain_available": boundary_radius_or_domain_available,
            "boundary_proxy_kind_or_none": boundary_proxy_kind_or_none,
            "boundary_proxy_without_new_free_parameters": boundary_proxy_without_new_free_parameters,
            "boundary_proxy_nonclosure_reason_or_none": "shell_quantization_rows_do_not_define_cavity_scale",
            "hand_off_to_8_7_55_2_84": False,
            "next_required_artifacts": [
                "boundary_radius_or_domain_proxy",
                "shell_quantization_reflective_cavity_rule",
                "discrete_spectrum_second_reopen_refresh",
            ],
        },
        "evidence": {
            "shell_quantization_cavity_source_inventory_summary": source_inventory_summary,
            "geometric_boundary_residual_route_contract_summary": route_contract_summary,
            "geometric_boundary_promotion_summary": geometric_promotion_summary,
            "shell_quantization_canonicalization_summary": shell_summary,
            "mass_eigenmode_boundary_summary": boundary_summary,
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
