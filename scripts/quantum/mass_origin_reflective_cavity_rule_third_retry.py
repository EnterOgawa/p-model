#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_reflective_cavity_rule_third_retry.py

Step 8.7.55.2.415:
Retry closure of the shell-quantization reflective cavity rule after the
boundary-radius proxy third retry.

Inputs:
  - output/public/quantum/mass_origin_shell_quantization_domain_relation_route_contract_metrics.json
  - output/public/quantum/mass_origin_boundary_radius_proxy_third_retry_metrics.json
  - output/public/quantum/mass_origin_reflective_cavity_rule_second_retry_metrics.json

Outputs:
  - output/public/quantum/mass_origin_reflective_cavity_rule_third_retry_metrics.json
  - output/public/quantum/mass_origin_reflective_cavity_rule_third_retry_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

ROUTE_CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_domain_relation_route_contract_metrics.json"
THIRD_PROXY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_boundary_radius_proxy_third_retry_metrics.json"
SECOND_RETRY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_reflective_cavity_rule_second_retry_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_reflective_cavity_rule_third_retry_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_reflective_cavity_rule_third_retry_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.415"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: CLI 引数を解釈する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retry closure of the shell-quantization reflective cavity rule a third time.")
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
    cavity_rule_third_retry_nonclosure_reason_or_none: str | None,
) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "reflective_cavity_rule_third_retry_complete",
            "status": "pass",
            "metric": "reflective cavity rule third retry complete",
            "value": 1.0,
            "note": "This step retries closure of the shell-quantization reflective cavity rule after the shell-domain branch.",
        },
        {
            "row_id": "boundary_radius_or_domain_available",
            "status": "pass" if boundary_radius_or_domain_available else "reject",
            "metric": "boundary radius or domain available for reflective cavity rule third retry",
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
            "metric": "geometric boundary promotion rule available after third retry",
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
            "metric": "discrete shell cavity ready after third retry",
            "value": 1.0 if discrete_shell_cavity_ready else 0.0,
            "note": (
                "The geometric route can now discretize the mexican-hat pilot."
                if discrete_shell_cavity_ready
                else "The geometric route still cannot discretize the mexican-hat pilot."
            ),
        },
        {
            "row_id": "cavity_rule_third_retry_nonclosure_reason",
            "status": "watch" if cavity_rule_third_retry_nonclosure_reason_or_none else "pass",
            "metric": "reflective cavity rule third-retry non-closure reason",
            "value": 0.0 if cavity_rule_third_retry_nonclosure_reason_or_none else 1.0,
            "note": (
                f"Current non-closure reason: {cavity_rule_third_retry_nonclosure_reason_or_none}."
                if cavity_rule_third_retry_nonclosure_reason_or_none
                else "The reflective cavity rule is fully closed."
            ),
        },
    ]


# 関数: metrics payload 全体を構成する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (ROUTE_CONTRACT_JSON, THIRD_PROXY_JSON, SECOND_RETRY_JSON):
        _require_path(path)

    route_contract = _read_json(ROUTE_CONTRACT_JSON)
    third_proxy = _read_json(THIRD_PROXY_JSON)
    second_retry = _read_json(SECOND_RETRY_JSON)

    route_contract_summary = route_contract.get("summary", {})
    third_proxy_summary = third_proxy.get("summary", {})
    second_retry_summary = second_retry.get("summary", {})

    boundary_radius_or_domain_available = bool(third_proxy_summary.get("boundary_radius_or_domain_available", False))
    geometric_boundary_promotion_rule_available = boundary_radius_or_domain_available
    discrete_shell_cavity_ready = geometric_boundary_promotion_rule_available
    cavity_rule_third_retry_nonclosure_reason_or_none = None

    if not geometric_boundary_promotion_rule_available:
        cavity_rule_third_retry_nonclosure_reason_or_none = "shell_quantization_domain_statement_absent"

    rows = _build_rows(
        boundary_radius_or_domain_available=boundary_radius_or_domain_available,
        geometric_boundary_promotion_rule_available=geometric_boundary_promotion_rule_available,
        discrete_shell_cavity_ready=discrete_shell_cavity_ready,
        cavity_rule_third_retry_nonclosure_reason_or_none=cavity_rule_third_retry_nonclosure_reason_or_none,
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "reflective cavity rule third retry",
        },
        "inputs": {
            "mass_origin_shell_quantization_domain_relation_route_contract_json": _relative_str(ROUTE_CONTRACT_JSON),
            "mass_origin_boundary_radius_proxy_third_retry_json": _relative_str(THIRD_PROXY_JSON),
            "mass_origin_reflective_cavity_rule_second_retry_json": _relative_str(SECOND_RETRY_JSON),
        },
        "intent": "Retry closure of the shell-quantization reflective cavity rule after the shell-domain branch.",
        "formulas": {
            "closure_rule": "the reflective cavity route closes only if the boundary-radius proxy closes without a new scale and can be injected into the mexican-hat pilot as a reflective wall",
            "current_absence": "the public shell-family coefficients still lack the shell-quantization domain statement needed to define a cavity radius or domain proxy",
        },
        "rows": rows,
        "summary": {
            "candidate_binding_route_id": route_contract_summary.get("selected_residual_binding_route_or_none"),
            "boundary_radius_or_domain_available": boundary_radius_or_domain_available,
            "geometric_boundary_promotion_rule_available": geometric_boundary_promotion_rule_available,
            "discrete_shell_cavity_ready": discrete_shell_cavity_ready,
            "cavity_rule_third_retry_nonclosure_reason_or_none": cavity_rule_third_retry_nonclosure_reason_or_none,
        },
        "decision": {
            "overall_status": (
                "reflective_cavity_rule_third_retry_closed"
                if geometric_boundary_promotion_rule_available
                else "reflective_cavity_rule_third_retry_still_blocked"
            ),
            "keep_mass_origin_branch_blocked": not geometric_boundary_promotion_rule_available,
            "candidate_binding_route_id": route_contract_summary.get("selected_residual_binding_route_or_none"),
            "boundary_radius_or_domain_available": boundary_radius_or_domain_available,
            "geometric_boundary_promotion_rule_available": geometric_boundary_promotion_rule_available,
            "discrete_shell_cavity_ready": discrete_shell_cavity_ready,
            "cavity_rule_third_retry_nonclosure_reason_or_none": cavity_rule_third_retry_nonclosure_reason_or_none,
            "hand_off_to_8_7_55_2_84": False,
            "next_required_artifacts": (
                ["geometric_binding_selection_third_retry", "discrete_spectrum_fourth_reopen_refresh"]
                if geometric_boundary_promotion_rule_available
                else ["shell_quantization_domain_statement", "shell_quantization_to_domain_relation", "shell_quantization_reflective_cavity_rule"]
            ),
        },
        "evidence": {
            "shell_quantization_domain_relation_route_contract_summary": route_contract_summary,
            "boundary_radius_proxy_third_retry_summary": third_proxy_summary,
            "reflective_cavity_rule_second_retry_summary": second_retry_summary,
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
