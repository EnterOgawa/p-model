#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_single_vpp_candidate_derivative_audit.py

Step 8.7.55.2.78:
Audit the same-sector V''(|P|_*) and V'''(|P|_*) slots for the minimal
candidate-family inventory frozen in 8.7.55.2.77.

This step still does not choose a single V(|P|) shape. It freezes which
candidate families can, in principle, provide

  - a same-sector curvature slot V''(|P|_*)
  - a three-wave slot V'''(|P|_*)
  - a representable stability condition V''(|P|_*) > 0

at the already-frozen same-sector reference point |P|_*.

Inputs:
  - output/public/quantum/mass_origin_same_sector_chi_to_vpp_contract_metrics.json
  - output/public/quantum/mass_origin_single_vpp_candidate_inventory_metrics.json

Outputs:
  - output/public/quantum/mass_origin_single_vpp_candidate_derivative_metrics.json
  - output/public/quantum/mass_origin_single_vpp_candidate_derivative_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

CONTRACT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_chi_to_vpp_contract_metrics.json"
INVENTORY_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_single_vpp_candidate_inventory_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_single_vpp_candidate_derivative_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_single_vpp_candidate_derivative_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.78"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit same-sector V'' / V''' derivative slots for minimal V(|P|) candidates.",
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


# 関数: `_candidate_rows` の入出力契約と処理意図を定義する。

def _candidate_rows() -> List[Dict[str, Any]]:
    return [
        {
            "row_id": "candidate_derivative_mexican_hat",
            "status": "pass",
            "family": "mexican_hat",
            "metric": "same-sector derivative slots for mexican_hat family",
            "value": 1.0,
            "vpp_defined_at_absP_star": True,
            "vppp_defined_at_absP_star": True,
            "same_sector_curvature_slot_available": True,
            "three_wave_slot_available": True,
            "stability_condition_representable": True,
            "vpp_formula_at_absP_star": "V''(|P|_*) = 8 lambda |P|_*^2",
            "vppp_formula_at_absP_star": "V'''(|P|_*) = 24 lambda |P|_*",
            "note": "For V(x)=lambda(x^2-v^2)^2 with x=|P| and |P|_*=v>0, both V'' and V''' are explicit and same-sector. Positive curvature reduces to lambda>0.",
        },
        {
            "row_id": "candidate_derivative_logarithmic",
            "status": "pass",
            "family": "logarithmic",
            "metric": "same-sector derivative slots for logarithmic family",
            "value": 1.0,
            "vpp_defined_at_absP_star": True,
            "vppp_defined_at_absP_star": True,
            "same_sector_curvature_slot_available": True,
            "three_wave_slot_available": True,
            "stability_condition_representable": True,
            "vpp_formula_at_absP_star": "V''(|P|_*) = 4 mu^2",
            "vppp_formula_at_absP_star": "V'''(|P|_*) = 4 mu^2 / |P|_*",
            "note": "For V(x)=mu^2 x^2[ln(x^2/v^2)-1] with x=|P| and |P|_*=v>0, both derivatives are explicit provided |P|_*>0. Positive curvature reduces to mu^2>0.",
        },
        {
            "row_id": "candidate_derivative_even_polynomial",
            "status": "pass",
            "family": "even_polynomial",
            "metric": "same-sector derivative slots for even-polynomial family",
            "value": 1.0,
            "vpp_defined_at_absP_star": True,
            "vppp_defined_at_absP_star": True,
            "same_sector_curvature_slot_available": True,
            "three_wave_slot_available": True,
            "stability_condition_representable": True,
            "vpp_formula_at_absP_star": "V''(|P|_*) = sum_{n=2}^{N} 2n(2n-1) c_n |P|_*^(2n-2)",
            "vppp_formula_at_absP_star": "V'''(|P|_*) = sum_{n=2}^{N} 2n(2n-1)(2n-2) c_n |P|_*^(2n-3)",
            "note": "For V(x)=sum_{n=2}^{N} c_n x^{2n} with x=|P| and |P|_*>0, both derivatives exist in the same sector. Stability and cubic support are representable but less constrained than the two minimal two-parameter families.",
        },
    ]


# 関数: `_aggregate_rows` の入出力契約と処理意図を定義する。

def _aggregate_rows(candidate_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    vpp_defined_count = sum(1 for row in candidate_rows if bool(row["vpp_defined_at_absP_star"]))
    vppp_defined_count = sum(1 for row in candidate_rows if bool(row["vppp_defined_at_absP_star"]))
    same_sector_ready = sum(1 for row in candidate_rows if bool(row["same_sector_curvature_slot_available"]))
    three_wave_ready = sum(1 for row in candidate_rows if bool(row["three_wave_slot_available"]))
    stability_ready = sum(1 for row in candidate_rows if bool(row["stability_condition_representable"]))

    return [
        {
            "row_id": "candidate_derivative_vpp_defined_count",
            "status": "pass" if vpp_defined_count == len(candidate_rows) else "reject",
            "family": "aggregate",
            "metric": "candidate count with V''(|P|_*) defined",
            "value": float(vpp_defined_count),
            "vpp_defined_at_absP_star": vpp_defined_count == len(candidate_rows),
            "vppp_defined_at_absP_star": None,
            "same_sector_curvature_slot_available": same_sector_ready == len(candidate_rows),
            "three_wave_slot_available": None,
            "stability_condition_representable": stability_ready == len(candidate_rows),
            "vpp_formula_at_absP_star": "",
            "vppp_formula_at_absP_star": "",
            "note": "All candidate families keep an explicit same-sector curvature slot at |P|_*.",
        },
        {
            "row_id": "candidate_derivative_vppp_defined_count",
            "status": "pass" if vppp_defined_count == len(candidate_rows) else "reject",
            "family": "aggregate",
            "metric": "candidate count with V'''(|P|_*) defined",
            "value": float(vppp_defined_count),
            "vpp_defined_at_absP_star": None,
            "vppp_defined_at_absP_star": vppp_defined_count == len(candidate_rows),
            "same_sector_curvature_slot_available": None,
            "three_wave_slot_available": three_wave_ready == len(candidate_rows),
            "stability_condition_representable": None,
            "vpp_formula_at_absP_star": "",
            "vppp_formula_at_absP_star": "",
            "note": "All candidate families keep a three-wave derivative slot at |P|_* and can therefore enter the cubic admissibility test.",
        },
        {
            "row_id": "candidate_derivative_stability_condition_ready",
            "status": "pass" if stability_ready == len(candidate_rows) else "reject",
            "family": "aggregate",
            "metric": "candidate count with representable V''(|P|_*) > 0 stability condition",
            "value": float(stability_ready),
            "vpp_defined_at_absP_star": None,
            "vppp_defined_at_absP_star": None,
            "same_sector_curvature_slot_available": None,
            "three_wave_slot_available": None,
            "stability_condition_representable": stability_ready == len(candidate_rows),
            "vpp_formula_at_absP_star": "",
            "vppp_formula_at_absP_star": "",
            "note": "Each candidate family admits an explicit positivity condition on V''(|P|_*) and is therefore eligible for the next admissibility gate.",
        },
    ]


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (CONTRACT_JSON, INVENTORY_JSON):
        _require_path(path)

    contract = _read_json(CONTRACT_JSON)
    inventory = _read_json(INVENTORY_JSON)

    contract_summary = contract.get("summary", {})
    contract_decision = contract.get("decision", {})
    inventory_summary = inventory.get("summary", {})

    candidate_rows = _candidate_rows()
    rows = candidate_rows + _aggregate_rows(candidate_rows)
    candidate_row_count = len(candidate_rows)
    vpp_defined_count = sum(1 for row in candidate_rows if bool(row["vpp_defined_at_absP_star"]))
    vppp_defined_count = sum(1 for row in candidate_rows if bool(row["vppp_defined_at_absP_star"]))
    same_sector_ready = sum(1 for row in candidate_rows if bool(row["same_sector_curvature_slot_available"]))
    three_wave_ready = sum(1 for row in candidate_rows if bool(row["three_wave_slot_available"]))
    stability_ready = sum(1 for row in candidate_rows if bool(row["stability_condition_representable"]))

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "candidate V'' / V''' derivative audit",
        },
        "inputs": {
            "mass_origin_same_sector_chi_to_vpp_contract_json": _relative_str(CONTRACT_JSON),
            "mass_origin_single_vpp_candidate_inventory_json": _relative_str(INVENTORY_JSON),
        },
        "intent": "Compare V''(|P|_*) and V'''(|P|_*) slots across the minimal same-sector candidate families and freeze whether they are admissible for the next gate.",
        "formulas": {
            "audit_rule": "each candidate must expose same-sector V''(|P|_*) and V'''(|P|_*) at nonzero |P|_* to remain eligible for chi_P and three-wave compatibility tests",
            "same_sector_curvature_slot": "chi_P must eventually map into the candidate's V''(|P|_*) slot",
            "three_wave_slot": "the candidate's V'''(|P|_*) slot must stay available for cubic / 3-wave consistency checks",
        },
        "rows": rows,
        "summary": {
            "candidate_row_count": candidate_row_count,
            "candidate_family_ids": inventory_summary.get("candidate_family_ids", []),
            "same_sector_only": bool(contract_summary.get("same_particle_sector_only", False)),
            "reference_point_symbol": str(contract_summary.get("reference_point_symbol", "")),
            "nonzero_reference_point_required": inventory_summary.get("reference_point_rule", "") == "nonzero_absP_star_required",
            "vpp_defined_candidate_count": vpp_defined_count,
            "vppp_defined_candidate_count": vppp_defined_count,
            "same_sector_curvature_slot_ready": same_sector_ready == candidate_row_count,
            "three_wave_slot_ready": three_wave_ready == candidate_row_count,
            "stability_condition_representable_count": stability_ready,
        },
        "decision": {
            "overall_status": "single_vpp_candidate_derivative_slots_frozen",
            "keep_mass_origin_branch_blocked": True,
            "candidate_row_count": candidate_row_count,
            "vpp_defined_candidate_count": vpp_defined_count,
            "vppp_defined_candidate_count": vppp_defined_count,
            "same_sector_curvature_slot_ready": same_sector_ready == candidate_row_count,
            "three_wave_slot_ready": three_wave_ready == candidate_row_count,
            "stability_condition_representable_count": stability_ready,
            "single_public_vpp_shape_available": False,
            "blocked_state_detail": str(contract_decision.get("blocked_state_detail", "")),
            "next_required_artifacts": contract_decision.get(
                "next_required_artifacts",
                [
                    "positive_particle_sector_chi_p_to_vpp_public_artifact",
                    "single_public_vpp_shape",
                    "solver_ready_row_promoted_to_pass",
                ],
            ),
        },
        "evidence": {
            "contract_summary": contract_summary,
            "inventory_summary": inventory_summary,
        },
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(rows: List[Dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "row_id",
                "status",
                "family",
                "metric",
                "value",
                "vpp_defined_at_absP_star",
                "vppp_defined_at_absP_star",
                "same_sector_curvature_slot_available",
                "three_wave_slot_available",
                "stability_condition_representable",
                "vpp_formula_at_absP_star",
                "vppp_formula_at_absP_star",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    args = _parse_args()
    payload = _build_payload(str(args.step_tag))
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(payload["rows"])
    print(json.dumps(payload["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
