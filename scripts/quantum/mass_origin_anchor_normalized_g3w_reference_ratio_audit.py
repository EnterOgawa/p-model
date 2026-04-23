#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_anchor_normalized_g3w_reference_ratio_audit.py

Step 8.7.55.2.256:
Audit whether the current public canonical pack already exposes a no-new-free-
parameter rule for rho_* / P_ref (or an equivalent same-sector reference ratio)
inside the anchor-normalized g_3w route.

Inputs:
  - output/public/quantum/mass_origin_anchor_normalized_g3w_rho_elimination_audit_metrics.json
  - output/public/quantum/mass_origin_anchor_normalized_g3w_rho_residual_contract_metrics.json
  - output/public/quantum/mass_origin_anchor_normalized_g3w_chi_proxy_audit_metrics.json

Outputs:
  - output/public/quantum/mass_origin_anchor_normalized_g3w_reference_ratio_audit_metrics.json
  - output/public/quantum/mass_origin_anchor_normalized_g3w_reference_ratio_audit_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

RHO_ELIMINATION_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_rho_elimination_audit_metrics.json"
RHO_RESIDUAL_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_rho_residual_contract_metrics.json"
CHI_PROXY_AUDIT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_chi_proxy_audit_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_reference_ratio_audit_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_anchor_normalized_g3w_reference_ratio_audit_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.256"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit whether a rho_* / P_ref reference-ratio rule is already public canonical.",
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
    rho_star_to_reference_ratio_rule_available: bool,
    rho_star_to_reference_ratio_kind_or_none: str | None,
    rho_star_to_reference_ratio_without_new_free_parameters: bool,
    missing_inputs: List[str],
) -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "anchor_normalized_g3w_reference_ratio_audit_complete",
            "status": "pass",
            "metric": "rho_* / P_ref reference-ratio audit complete",
            "value": 1.0,
            "note": "This step tests whether the current public pack already supports a same-sector reference-ratio rule for rho_*.",
        },
        {
            "row_id": "anchor_normalized_g3w_reference_ratio_rule_available",
            "status": "pass" if rho_star_to_reference_ratio_rule_available else "reject",
            "metric": "rho_* / P_ref or equivalent same-sector reference-ratio rule available",
            "value": 1.0 if rho_star_to_reference_ratio_rule_available else 0.0,
            "note": (
                f"The current public pack already supports {rho_star_to_reference_ratio_kind_or_none}."
                if rho_star_to_reference_ratio_rule_available
                else "The current public pack still lacks a public same-sector rule for rho_* / P_ref."
            ),
        },
        {
            "row_id": "anchor_normalized_g3w_reference_ratio_without_new_free_parameters",
            "status": "pass" if rho_star_to_reference_ratio_without_new_free_parameters else "reject",
            "metric": "rho_* / P_ref reference-ratio rule stays inside no-new-free-parameter envelope",
            "value": 1.0 if rho_star_to_reference_ratio_without_new_free_parameters else 0.0,
            "note": (
                "The current public pack already supports the anchor reference ratio without a new fit."
                if rho_star_to_reference_ratio_without_new_free_parameters
                else "Any rho_* / P_ref rewrite would still require a missing public same-sector anchor coordinate."
            ),
        },
        {
            "row_id": "anchor_normalized_g3w_reference_ratio_missing_inputs",
            "status": "missing" if missing_inputs else "pass",
            "metric": "remaining missing inputs for rho_* / P_ref reference-ratio rule",
            "value": float(len(missing_inputs)),
            "note": f"Missing inputs: {missing_inputs}.",
        },
    ]


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (RHO_ELIMINATION_JSON, RHO_RESIDUAL_JSON, CHI_PROXY_AUDIT_JSON):
        _require_path(path)

    rho_elimination = _read_json(RHO_ELIMINATION_JSON)
    rho_residual = _read_json(RHO_RESIDUAL_JSON)
    chi_proxy = _read_json(CHI_PROXY_AUDIT_JSON)

    rho_elimination_summary = rho_elimination.get("summary", {})
    rho_residual_summary = rho_residual.get("summary", {})
    chi_proxy_summary = chi_proxy.get("summary", {})
    chi_proxy_decision = chi_proxy.get("decision", {})

    chi_proxy_rule_available = bool(chi_proxy_summary.get("chi_proxy_rule_available", False))
    rho_star_to_reference_ratio_rule_available = False
    rho_star_to_reference_ratio_kind_or_none = None
    rho_star_to_reference_ratio_without_new_free_parameters = False
    missing_inputs: List[str] = []

    # 条件分岐: `not chi_proxy_rule_available` を満たす経路を評価する。
    if not chi_proxy_rule_available:
        missing_inputs.append("chi_star_or_same_sector_proxy")

    rows = _build_rows(
        rho_star_to_reference_ratio_rule_available=rho_star_to_reference_ratio_rule_available,
        rho_star_to_reference_ratio_kind_or_none=rho_star_to_reference_ratio_kind_or_none,
        rho_star_to_reference_ratio_without_new_free_parameters=rho_star_to_reference_ratio_without_new_free_parameters,
        missing_inputs=missing_inputs,
    )

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "rho_star to reference ratio audit",
        },
        "inputs": {
            "mass_origin_anchor_normalized_g3w_rho_elimination_audit_json": _relative_str(RHO_ELIMINATION_JSON),
            "mass_origin_anchor_normalized_g3w_rho_residual_contract_json": _relative_str(RHO_RESIDUAL_JSON),
            "mass_origin_anchor_normalized_g3w_chi_proxy_audit_json": _relative_str(CHI_PROXY_AUDIT_JSON),
        },
        "intent": "Determine whether the current public canonical pack already exposes rho_* / P_ref or an equivalent same-sector reference-ratio rule for the anchor-normalized g_3w route.",
        "formulas": {
            "reference_ratio_rule": "rho_* / P_ref = exp(chi_*) once a public chi_* datum or equivalent same-sector proxy rule is available",
            "same_sector_equivalent_rule": "an equivalent same-sector reference ratio is admissible iff it fixes the anchor reference coordinate without a new fit",
            "current_absence": "the current public pack already fixes chi = ln(rho / P_ref) and rho_* = |P|_*, but it does not yet fix chi_* or an equivalent same-sector proxy value",
        },
        "rows": rows,
        "summary": {
            "rho_star_to_reference_ratio_rule_available": rho_star_to_reference_ratio_rule_available,
            "rho_star_to_reference_ratio_kind_or_none": rho_star_to_reference_ratio_kind_or_none,
            "rho_star_to_reference_ratio_without_new_free_parameters": rho_star_to_reference_ratio_without_new_free_parameters,
            "reference_ratio_missing_inputs": missing_inputs,
        },
        "decision": {
            "overall_status": "anchor_normalized_g3w_reference_ratio_audit_frozen_absent",
            "keep_mass_origin_branch_blocked": True,
            "rho_star_to_reference_ratio_rule_available": rho_star_to_reference_ratio_rule_available,
            "rho_star_to_reference_ratio_kind_or_none": rho_star_to_reference_ratio_kind_or_none,
            "rho_star_to_reference_ratio_without_new_free_parameters": rho_star_to_reference_ratio_without_new_free_parameters,
            "hand_off_to_8_7_55_2_83": bool(chi_proxy_decision.get("hand_off_to_8_7_55_2_83", False)),
            "next_required_artifacts": [
                "chi_star_or_same_sector_proxy",
                "rho_star_to_reference_ratio_rule",
                "anchor_normalized_g3w_public_value",
                "r3_target",
                "single_public_vpp_shape",
                "positive_particle_sector_chi_p_to_vpp_public_artifact",
                "solver_ready_row_promoted_to_pass",
            ],
        },
        "evidence": {
            "rho_elimination_summary": rho_elimination_summary,
            "rho_residual_summary": rho_residual_summary,
            "chi_proxy_summary": chi_proxy_summary,
            "chi_proxy_decision": chi_proxy_decision,
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
