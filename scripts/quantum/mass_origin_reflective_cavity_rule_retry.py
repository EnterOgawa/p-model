#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_reflective_cavity_rule_retry.py

Step 8.7.55.2.402:
Retry closure of the shell-quantization reflective cavity rule using the
current cavity-source inventory and the boundary-radius proxy audit.

Inputs:
  - output/public/quantum/mass_origin_geometric_boundary_residual_route_contract_metrics.json
  - output/public/quantum/mass_origin_shell_quantization_cavity_source_inventory_metrics.json
  - output/public/quantum/mass_origin_boundary_radius_proxy_audit_metrics.json
  - output/public/quantum/mass_origin_geometric_boundary_promotion_metrics.json

Outputs:
  - output/public/quantum/mass_origin_reflective_cavity_rule_retry_metrics.json
  - output/public/quantum/mass_origin_reflective_cavity_rule_retry_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

ROUTE_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_geometric_boundary_residual_route_contract_metrics.json"
SOURCE_INVENTORY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_cavity_source_inventory_metrics.json"
BOUNDARY_PROXY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_boundary_radius_proxy_audit_metrics.json"
GEOMETRIC_PROMOTION_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_geometric_boundary_promotion_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_reflective_cavity_rule_retry_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_reflective_cavity_rule_retry_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.402"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retry closure of the shell-quantization reflective cavity rule.")
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
    geometric_boundary_promotion_rule_available: bool,
    discrete_shell_cavity_ready: bool,
    cavity_rule_nonclosure_reason_or_none: str | None,
) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "reflective_cavity_rule_retry_complete",
            "status": "pass",
            "metric": "reflective cavity rule retry complete",
            "value": 1.0,
            "note": "This step retries closure of the shell-quantization reflective cavity rule after source inventory and boundary-proxy audit.",
        },
        {
            "row_id": "boundary_radius_or_domain_available",
            "status": "pass" if boundary_radius_or_domain_available else "reject",
            "metric": "boundary radius or domain available for reflective cavity rule",
            "value": 1.0 if boundary_radius_or_domain_available else 0.0,
            "note": (
                "A cavity radius or domain proxy is already available."
                if boundary_radius_or_domain_available
                else "The reflective cavity rule still lacks a public cavity radius or domain proxy."
            ),
        },
        {
            "row_id": "geometric_boundary_promotion_rule_available",
            "status": "pass" if geometric_boundary_promotion_rule_available else "reject",
            "metric": "geometric boundary promotion rule available after retry",
            "value": 1.0 if geometric_boundary_promotion_rule_available else 0.0,
            "note": (
                "The shell-quantization family now promotes into a reflective cavity rule."
                if geometric_boundary_promotion_rule_available
                else "The shell-quantization family still cannot be lifted into a reflective cavity rule."
            ),
        },
        {
            "row_id": "discrete_shell_cavity_ready",
            "status": "pass" if discrete_shell_cavity_ready else "reject",
            "metric": "discrete shell cavity ready after retry",
            "value": 1.0 if discrete_shell_cavity_ready else 0.0,
            "note": (
                "The geometric route can now discretize the mexican-hat pilot."
                if discrete_shell_cavity_ready
                else "The geometric route still cannot discretize the mexican-hat pilot."
            ),
        },
        {
            "row_id": "cavity_rule_nonclosure_reason",
            "status": "watch" if cavity_rule_nonclosure_reason_or_none else "pass",
            "metric": "reflective cavity rule non-closure reason",
            "value": 0.0 if cavity_rule_nonclosure_reason_or_none else 1.0,
            "note": (
                f"Current non-closure reason: {cavity_rule_nonclosure_reason_or_none}."
                if cavity_rule_nonclosure_reason_or_none
                else "The reflective cavity rule is fully closed."
            ),
        },
    ]


# 関数: metrics payload 全体を構成する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (ROUTE_CONTRACT_JSON, SOURCE_INVENTORY_JSON, BOUNDARY_PROXY_JSON, GEOMETRIC_PROMOTION_JSON):
        _require_path(path)

    route_contract = _read_json(ROUTE_CONTRACT_JSON)
    source_inventory = _read_json(SOURCE_INVENTORY_JSON)
    boundary_proxy = _read_json(BOUNDARY_PROXY_JSON)
    geometric_promotion = _read_json(GEOMETRIC_PROMOTION_JSON)

    route_contract_summary = route_contract.get("summary", {})
    source_inventory_summary = source_inventory.get("summary", {})
    boundary_proxy_summary = boundary_proxy.get("summary", {})
    geometric_promotion_summary = geometric_promotion.get("summary", {})

    boundary_radius_or_domain_available = bool(boundary_proxy_summary.get("boundary_radius_or_domain_available", False))
    geometric_boundary_promotion_rule_available = boundary_radius_or_domain_available
    discrete_shell_cavity_ready = geometric_boundary_promotion_rule_available
    cavity_rule_nonclosure_reason_or_none = None

    if not geometric_boundary_promotion_rule_available:
        cavity_rule_nonclosure_reason_or_none = "boundary_radius_or_domain_proxy_absent"

    rows = _build_rows(
        boundary_radius_or_domain_available=boundary_radius_or_domain_available,
        geometric_boundary_promotion_rule_available=geometric_boundary_promotion_rule_available,
        discrete_shell_cavity_ready=discrete_shell_cavity_ready,
        cavity_rule_nonclosure_reason_or_none=cavity_rule_nonclosure_reason_or_none,
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "reflective cavity rule closure retry",
        },
        "inputs": {
            "mass_origin_geometric_boundary_residual_route_contract_json": _relative_str(ROUTE_CONTRACT_JSON),
            "mass_origin_shell_quantization_cavity_source_inventory_json": _relative_str(SOURCE_INVENTORY_JSON),
            "mass_origin_boundary_radius_proxy_audit_json": _relative_str(BOUNDARY_PROXY_JSON),
            "mass_origin_geometric_boundary_promotion_json": _relative_str(GEOMETRIC_PROMOTION_JSON),
        },
        "intent": "Retry closure of the shell-quantization reflective cavity rule using the current source inventory and the cavity-radius proxy audit.",
        "formulas": {
            "closure_rule": "the reflective cavity route closes only if a no-new-free-parameter cavity radius or domain proxy is available and can be injected into the mexican-hat pilot as a reflective wall",
            "current_absence": "the public shell-family coefficients still lack a geometric radius or domain proxy, so the reflective cavity rule remains unavailable",
        },
        "rows": rows,
        "summary": {
            "candidate_binding_route_id": route_contract_summary.get("selected_residual_binding_route_or_none"),
            "boundary_radius_or_domain_available": boundary_radius_or_domain_available,
            "geometric_boundary_promotion_rule_available": geometric_boundary_promotion_rule_available,
            "discrete_shell_cavity_ready": discrete_shell_cavity_ready,
            "cavity_rule_nonclosure_reason_or_none": cavity_rule_nonclosure_reason_or_none,
        },
        "decision": {
            "overall_status": (
                "reflective_cavity_rule_retry_closed"
                if geometric_boundary_promotion_rule_available
                else "reflective_cavity_rule_retry_still_blocked"
            ),
            "keep_mass_origin_branch_blocked": not geometric_boundary_promotion_rule_available,
            "candidate_binding_route_id": route_contract_summary.get("selected_residual_binding_route_or_none"),
            "boundary_radius_or_domain_available": boundary_radius_or_domain_available,
            "geometric_boundary_promotion_rule_available": geometric_boundary_promotion_rule_available,
            "discrete_shell_cavity_ready": discrete_shell_cavity_ready,
            "cavity_rule_nonclosure_reason_or_none": cavity_rule_nonclosure_reason_or_none,
            "hand_off_to_8_7_55_2_84": False,
            "next_required_artifacts": (
                ["geometric_binding_selection_retry", "discrete_spectrum_second_reopen_refresh"]
                if geometric_boundary_promotion_rule_available
                else ["boundary_radius_or_domain_proxy", "shell_quantization_reflective_cavity_rule"]
            ),
        },
        "evidence": {
            "geometric_boundary_residual_route_contract_summary": route_contract_summary,
            "shell_quantization_cavity_source_inventory_summary": source_inventory_summary,
            "boundary_radius_proxy_audit_summary": boundary_proxy_summary,
            "geometric_boundary_promotion_summary": geometric_promotion_summary,
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
