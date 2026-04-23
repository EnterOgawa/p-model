#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_single_vpp_candidate_inventory.py

Step 8.7.55.2.77:
Freeze the minimal same-sector candidate-family inventory for a future single
public V(|P|) shape.

This step does not choose one shape. It defines the smallest canonical family
set that can be carried into the later derivative audit and admissibility gate:

  - mexican_hat
  - logarithmic
  - even_polynomial

The inventory is constrained by the already-frozen same-sector chi_P -> V''(|P|_*)
contract, the surviving shell family, and the requirement that a future family
must be able to discuss both V'' and V''' at the same-sector reference point.

Inputs:
  - output/public/quantum/mass_origin_same_sector_chi_to_vpp_contract_metrics.json
  - output/public/quantum/mass_origin_curvature_boundary_metrics.json
  - output/public/quantum/nuclear_effective_potential_canonical_metrics.json
  - output/public/quantum/action_principle_el_derivation_audit.json

Outputs:
  - output/public/quantum/mass_origin_single_vpp_candidate_inventory_metrics.json
  - output/public/quantum/mass_origin_single_vpp_candidate_inventory_rows.csv
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
CURVATURE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_curvature_boundary_metrics.json"
EFFECTIVE_JSON = ROOT / "output" / "public" / "quantum" / "nuclear_effective_potential_canonical_metrics.json"
ACTION_JSON = ROOT / "output" / "public" / "quantum" / "action_principle_el_derivation_audit.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_single_vpp_candidate_inventory_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_single_vpp_candidate_inventory_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.77"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the minimal same-sector V(|P|) candidate-family inventory.",
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

def _build_rows(contract: Dict[str, Any], curvature: Dict[str, Any], effective: Dict[str, Any], action: Dict[str, Any]) -> List[Dict[str, Any]]:
    contract_summary = contract.get("summary", {})
    curvature_summary = curvature.get("summary", {})
    effective_model = effective.get("model", {})
    effective_positioning = effective_model.get("positioning", [])
    action_equations = action.get("equations", {})
    lagrangian = str(action_equations.get("lagrangian_density", ""))

    contract_ready = bool(contract_summary.get("chi_to_vpp_mapping_contract_frozen", False))
    same_sector_only = bool(contract_summary.get("same_particle_sector_only", False))
    shell_anchor_ids = contract_summary.get("existing_shell_family_row_ids", [])
    abstract_shape_fixed = bool(curvature_summary.get("same_sector_curvature_fixed", False) is False)
    effective_positioning_text = " ".join(str(item) for item in effective_positioning)
    effective_is_phenomenological = "Phenomenological" in effective_positioning_text or "not a first-principles" in effective_positioning_text

    return [
        {
            "row_id": "candidate_inventory_same_sector_only",
            "status": "pass" if contract_ready and same_sector_only else "reject",
            "family": "inventory_gate",
            "metric": "candidate inventory restricted to same particle sector",
            "value": 1.0 if contract_ready and same_sector_only else 0.0,
            "note": "All candidate families are defined only for the same particle sector and inherit the frozen chi_P -> V''(|P|_*) contract.",
        },
        {
            "row_id": "candidate_inventory_reference_point_nonzero_absP_star",
            "status": "pass",
            "family": "inventory_gate",
            "metric": "candidate inventory evaluated at nonzero |P|_*",
            "value": 1.0,
            "note": "The inventory assumes a nonzero same-sector reference point |P|_* so V'' and V''' can both be discussed without falling back to an origin-only expansion.",
        },
        {
            "row_id": "candidate_family_mexican_hat",
            "status": "candidate_inventory",
            "family": "mexican_hat",
            "metric": "V(x)=lambda (x^2-v^2)^2 with x=|P|",
            "value": 2.0,
            "note": "Minimal parameter pair (lambda,v). At x=v>0, V''=8 lambda v^2 and V'''=24 lambda v, so the family can represent both stability and a nonzero cubic slot in the same sector.",
        },
        {
            "row_id": "candidate_family_logarithmic",
            "status": "candidate_inventory",
            "family": "logarithmic",
            "metric": "V(x)=mu^2 x^2 [ln(x^2/v^2)-1] with x=|P|",
            "value": 2.0,
            "note": "Minimal parameter pair (mu,v). At x=v>0, V''=4 mu^2 and V'''=4 mu^2 / v, so the family can represent positive curvature and a nonzero cubic derivative in the same sector.",
        },
        {
            "row_id": "candidate_family_even_polynomial",
            "status": "candidate_inventory",
            "family": "even_polynomial",
            "metric": "V(x)=sum_{n=2}^{N} c_n x^{2n} with x=|P|",
            "value": 3.0,
            "note": "Generic even polynomial family with a lower-bound coefficient count of 3. At a nonzero x=|P|_*, both V'' and V''' can be nonzero, but the family is less constrained and therefore expected to face stricter admissibility filtering.",
        },
        {
            "row_id": "candidate_inventory_third_derivative_support",
            "status": "pass",
            "family": "inventory_gate",
            "metric": "candidate families can discuss V'''(|P|_*)",
            "value": 3.0,
            "note": "All three candidate families support a meaningful V''' discussion at nonzero |P|_* and can therefore be carried into the three-wave derivative audit.",
        },
        {
            "row_id": "candidate_inventory_stability_filter_ready",
            "status": "pass" if abstract_shape_fixed else "watch",
            "family": "inventory_gate",
            "metric": "candidate families ready for V''>0 stability filtering",
            "value": 1.0 if abstract_shape_fixed else 0.0,
            "note": "Each candidate family admits a same-sector V''(|P|_*) > 0 condition, so the next step can filter them by stability and derivative consistency rather than by family syntax.",
        },
        {
            "row_id": "candidate_inventory_shell_family_binding_frozen",
            "status": "pass" if shell_anchor_ids else "reject",
            "family": "inventory_gate",
            "metric": "candidate families remain tied to surviving shell-family anchors",
            "value": float(len(shell_anchor_ids)),
            "note": (
                "The inventory is tied to the surviving shell-family anchors "
                f"{shell_anchor_ids} and may not introduce a new fit family outside the shell canonical pack."
            ),
        },
        {
            "row_id": "current_public_effective_potential_still_phenomenological",
            "status": "watch" if effective_is_phenomenological else "pass",
            "family": "public_effective_potential",
            "metric": "current public effective-potential branch already solves the single-shape problem",
            "value": 0.0 if effective_is_phenomenological else 1.0,
            "note": (
                "The current public effective-potential branch is not promoted into the candidate inventory as a solved same-sector shape because its own positioning remains: "
                f"{effective_positioning_text}"
            ),
        },
        {
            "row_id": "candidate_inventory_not_yet_single_shape",
            "status": "watch",
            "family": "uniqueness_gate",
            "metric": "single public V(|P|) shape already fixed",
            "value": 0.0,
            "note": "This step only freezes the minimal candidate family inventory. Uniqueness is deferred to the later admissibility and closure steps.",
        },
        {
            "row_id": "candidate_inventory_abstract_action_anchor_available",
            "status": "pass" if bool(lagrangian) else "reject",
            "family": "inventory_gate",
            "metric": "abstract action anchor available for candidate inventory",
            "value": 1.0 if bool(lagrangian) else 0.0,
            "note": f"The candidate inventory remains anchored to the abstract action `{lagrangian}` while concrete same-sector coefficients are still open.",
        },
    ]


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (CONTRACT_JSON, CURVATURE_JSON, EFFECTIVE_JSON, ACTION_JSON):
        _require_path(path)

    contract = _read_json(CONTRACT_JSON)
    curvature = _read_json(CURVATURE_JSON)
    effective = _read_json(EFFECTIVE_JSON)
    action = _read_json(ACTION_JSON)

    rows = _build_rows(contract, curvature, effective, action)
    contract_summary = contract.get("summary", {})
    contract_decision = contract.get("decision", {})
    effective_model = effective.get("model", {})
    effective_positioning = effective_model.get("positioning", [])

    candidate_family_ids = [
        "mexican_hat",
        "logarithmic",
        "even_polynomial",
    ]

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "minimal single V(|P|) candidate inventory",
        },
        "inputs": {
            "mass_origin_same_sector_chi_to_vpp_contract_json": _relative_str(CONTRACT_JSON),
            "mass_origin_curvature_boundary_json": _relative_str(CURVATURE_JSON),
            "nuclear_effective_potential_canonical_json": _relative_str(EFFECTIVE_JSON),
            "action_principle_el_derivation_audit_json": _relative_str(ACTION_JSON),
        },
        "intent": "Freeze the smallest same-sector candidate-family inventory that can feed the later V'' / V''' derivative audit and admissibility gate.",
        "formulas": {
            "mexican_hat_family": "V(x)=lambda (x^2-v^2)^2 with x=|P| and x_*=v>0",
            "logarithmic_family": "V(x)=mu^2 x^2 [ln(x^2/v^2)-1] with x=|P| and x_*=v>0",
            "even_polynomial_family": "V(x)=sum_{n=2}^{N} c_n x^{2n} with x=|P| and x_*>0",
            "inventory_rule": "keep only candidate families that can express same-sector V''(|P|_*) and V'''(|P|_*) without introducing a new solver family",
        },
        "rows": rows,
        "summary": {
            "candidate_family_count": len(candidate_family_ids),
            "candidate_family_ids": candidate_family_ids,
            "same_sector_only": bool(contract_summary.get("same_particle_sector_only", False)),
            "third_derivative_support_count": 3,
            "stability_filter_ready": True,
            "reference_point_rule": "nonzero_absP_star_required",
            "current_public_effective_potential_phenomenological": (
                "Phenomenological" in " ".join(str(item) for item in effective_positioning)
                or "not a first-principles" in " ".join(str(item) for item in effective_positioning)
            ),
        },
        "decision": {
            "overall_status": "single_vpp_candidate_inventory_frozen",
            "keep_mass_origin_branch_blocked": True,
            "candidate_family_count": len(candidate_family_ids),
            "same_sector_only": bool(contract_summary.get("same_particle_sector_only", False)),
            "third_derivative_support_count": 3,
            "stability_filter_ready": True,
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
            "curvature_summary": curvature.get("summary", {}),
            "effective_potential_positioning": effective_positioning,
            "action_lagrangian_density": action.get("equations", {}).get("lagrangian_density", ""),
        },
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(rows: List[Dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_id", "status", "family", "metric", "value", "note"],
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
