#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_anchor_normalized_g3w_value_retry.py

Step 8.7.55.2.257:
Retry the closure of the anchor-normalized g_3w public value and the resulting
R_3 target after the chi_* proxy audit and the rho_* / P_ref reference-ratio
audit.

Inputs:
  - output/public/quantum/mass_origin_anchor_normalized_g3w_value_closure_metrics.json
  - output/public/quantum/mass_origin_anchor_normalized_g3w_rho_residual_contract_metrics.json
  - output/public/quantum/mass_origin_anchor_normalized_g3w_chi_proxy_audit_metrics.json
  - output/public/quantum/mass_origin_anchor_normalized_g3w_reference_ratio_audit_metrics.json

Outputs:
  - output/public/quantum/mass_origin_anchor_normalized_g3w_value_retry_metrics.json
  - output/public/quantum/mass_origin_anchor_normalized_g3w_value_retry_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

VALUE_CLOSURE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_value_closure_metrics.json"
RHO_RESIDUAL_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_rho_residual_contract_metrics.json"
CHI_PROXY_AUDIT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_chi_proxy_audit_metrics.json"
REFERENCE_RATIO_AUDIT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_reference_ratio_audit_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_value_retry_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_value_retry_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.257"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retry the anchor-normalized g_3w public-value closure after the rho residual audits.",
    )
    parser.add_argument(
        "--step-tag",
        default=DEFAULT_STEP_TAG,
        help="Roadmap step tag to stamp into the output payload.",
    )
    return parser.parse_args()


# 関数: `_require_path` の入出力契約と処理意図を定義する。

def _require_path(path: Path) -> None:
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: `_relative_str` の入出力契約と処理意図を定義する。

def _relative_str(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: `_build_rows` の入出力契約と処理意図を定義する。

def _build_rows(
    *,
    anchor_normalized_g3w_public_value_available: bool,
    r3_target_available: bool,
    nonclosure_reason: str | None,
) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "anchor_normalized_g3w_value_retry_complete",
            "status": "pass",
            "metric": "anchor-normalized g_3w value retry complete",
            "value": 1.0,
            "note": "This step retries the closure of the anchor-normalized g_3w public value after the residual rho audits.",
        },
        {
            "row_id": "anchor_normalized_g3w_value_retry_public_value_available",
            "status": "pass" if anchor_normalized_g3w_public_value_available else "missing",
            "metric": "anchor-normalized public g_3w value available after retry",
            "value": 1.0 if anchor_normalized_g3w_public_value_available else 0.0,
            "note": (
                "A public anchor-normalized g_3w value is now available."
                if anchor_normalized_g3w_public_value_available
                else f"The retry remains non-closing: {nonclosure_reason}."
            ),
        },
        {
            "row_id": "anchor_normalized_g3w_value_retry_r3_target_available",
            "status": "pass" if r3_target_available else "reject",
            "metric": "R_3 target available after retry",
            "value": 1.0 if r3_target_available else 0.0,
            "note": (
                "The retry now promotes a public canonical R_3 target."
                if r3_target_available
                else "R_3 target remains unavailable because the anchor-normalized g_3w public value is still missing."
            ),
        },
    ]


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (VALUE_CLOSURE_JSON, RHO_RESIDUAL_JSON, CHI_PROXY_AUDIT_JSON, REFERENCE_RATIO_AUDIT_JSON):
        _require_path(path)

    value_closure = _read_json(VALUE_CLOSURE_JSON)
    rho_residual = _read_json(RHO_RESIDUAL_JSON)
    chi_proxy_audit = _read_json(CHI_PROXY_AUDIT_JSON)
    reference_ratio_audit = _read_json(REFERENCE_RATIO_AUDIT_JSON)

    value_closure_summary = value_closure.get("summary", {})
    rho_residual_summary = rho_residual.get("summary", {})
    chi_proxy_summary = chi_proxy_audit.get("summary", {})
    reference_ratio_summary = reference_ratio_audit.get("summary", {})

    chi_proxy_rule_available = bool(chi_proxy_summary.get("chi_proxy_rule_available", False))
    reference_ratio_rule_available = bool(reference_ratio_summary.get("rho_star_to_reference_ratio_rule_available", False))
    anchor_normalization_rule_available = bool(rho_residual_summary.get("anchor_normalization_rule_available", False))

    anchor_normalized_g3w_public_value_available = bool(
        anchor_normalization_rule_available and chi_proxy_rule_available and reference_ratio_rule_available
    )
    r3_target_available = anchor_normalized_g3w_public_value_available
    r3_target_value_or_none = None
    nonclosure_reason = None

    # 条件分岐: `not chi_proxy_rule_available` を満たす経路を評価する。
    if not chi_proxy_rule_available:
        nonclosure_reason = "chi_star_or_same_sector_proxy_absent"

    # 条件分岐: `chi_proxy_rule_available and not reference_ratio_rule_available` を満たす経路を評価する。

    if chi_proxy_rule_available and not reference_ratio_rule_available:
        nonclosure_reason = "rho_star_to_reference_ratio_rule_absent"

    rows = _build_rows(
        anchor_normalized_g3w_public_value_available=anchor_normalized_g3w_public_value_available,
        r3_target_available=r3_target_available,
        nonclosure_reason=nonclosure_reason,
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "anchor-normalized g3w public value closure retry",
        },
        "inputs": {
            "mass_origin_anchor_normalized_g3w_value_closure_json": _relative_str(VALUE_CLOSURE_JSON),
            "mass_origin_anchor_normalized_g3w_rho_residual_contract_json": _relative_str(RHO_RESIDUAL_JSON),
            "mass_origin_anchor_normalized_g3w_chi_proxy_audit_json": _relative_str(CHI_PROXY_AUDIT_JSON),
            "mass_origin_anchor_normalized_g3w_reference_ratio_audit_json": _relative_str(REFERENCE_RATIO_AUDIT_JSON),
        },
        "intent": "Retry the public anchor-normalized g_3w value closure after auditing the chi_* proxy and rho_* / P_ref reference-ratio routes.",
        "formulas": {
            "retry_closure_rule": "anchor_normalized_g3w_public_value_available iff anchor normalization, chi_* proxy, and rho_* / P_ref reference-ratio routes are all public canonical",
            "r3_retry_promotion_rule": "r3_target_available iff anchor_normalized_g3w_public_value_available",
        },
        "rows": rows,
        "summary": {
            "anchor_normalized_g3w_public_value_available": anchor_normalized_g3w_public_value_available,
            "r3_target_available": r3_target_available,
            "r3_target_value_or_none": r3_target_value_or_none,
            "g3w_route_retry_nonclosure_reason_or_none": nonclosure_reason,
        },
        "decision": {
            "overall_status": "anchor_normalized_g3w_value_retry_frozen_absent",
            "keep_mass_origin_branch_blocked": True,
            "anchor_normalized_g3w_public_value_available": anchor_normalized_g3w_public_value_available,
            "r3_target_available": r3_target_available,
            "r3_target_value_or_none": r3_target_value_or_none,
            "g3w_route_retry_nonclosure_reason_or_none": nonclosure_reason,
            "hand_off_to_8_7_55_2_83": False,
            "next_required_artifacts": [
                "chi_star_or_same_sector_proxy",
                "anchor_normalized_g3w_public_value",
                "r3_target",
                "single_public_vpp_shape",
                "positive_particle_sector_chi_p_to_vpp_public_artifact",
                "solver_ready_row_promoted_to_pass",
            ],
        },
        "evidence": {
            "value_closure_summary": value_closure_summary,
            "rho_residual_summary": rho_residual_summary,
            "chi_proxy_summary": chi_proxy_summary,
            "reference_ratio_summary": reference_ratio_summary,
        },
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(rows: List[Dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(rows)


# 関数: `main` の入出力契約と処理意図を定義する。

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
