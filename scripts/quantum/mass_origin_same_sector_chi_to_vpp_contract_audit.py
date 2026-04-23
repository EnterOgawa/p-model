#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_same_sector_chi_to_vpp_contract_audit.py

Step 8.7.55.2.76:
Freeze the public-canonical contract for the missing same-sector
chi_P -> V''(|P|_*) artifact without pretending that the artifact itself has
already been realized.

The purpose of this step is narrower than the existing blocker gate:
it fixes the contract that any future positive particle-sector mapping must
satisfy. The contract covers

  - same-particle-sector scope only
  - explicit mapping equation requirement
  - explicit sign and unit statements
  - explicit |P|_* reference point
  - reuse of the surviving shell-family numerical rows
  - explicit no-new-free-parameter wording

Inputs:
  - output/public/quantum/mass_origin_same_sector_vpp_shape_gate_metrics.json
  - output/public/quantum/mass_origin_readiness_gate_metrics.json
  - output/public/quantum/mass_origin_shell_quantization_canonicalization_metrics.json
  - output/public/quantum/mass_origin_shell_curvature_bridge_metrics.json
  - output/public/quantum/mass_origin_blocked_state_reopen_metrics.json
  - output/public/quantum/mass_origin_latent_reopen_route_inventory_metrics.json
  - output/public/quantum/action_principle_el_derivation_audit.json

Outputs:
  - output/public/quantum/mass_origin_same_sector_chi_to_vpp_contract_metrics.json
  - output/public/quantum/mass_origin_same_sector_chi_to_vpp_contract_rows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

GATE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_vpp_shape_gate_metrics.json"
READINESS_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_readiness_gate_metrics.json"
SHELL_CANON_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_quantization_canonicalization_metrics.json"
SHELL_BRIDGE_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_shell_curvature_bridge_metrics.json"
REOPEN_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_blocked_state_reopen_metrics.json"
LATENT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_latent_reopen_route_inventory_metrics.json"
ACTION_JSON = ROOT / "output" / "public" / "quantum" / "action_principle_el_derivation_audit.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_chi_to_vpp_contract_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_same_sector_chi_to_vpp_contract_rows.csv"
DEFAULT_STEP_TAG = "8.7.55.2.76"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_parse_args` の入出力契約と処理意図を定義する。

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the same-sector chi_P -> V''(|P|_*) contract for the mass-origin branch.",
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


# 関数: `_find_row_by_id` の入出力契約と処理意図を定義する。

def _find_row_by_id(rows: List[Dict[str, Any]], row_id: str) -> Dict[str, Any]:
    for row in rows:
        # 条件分岐: `str(row.get("row_id")) == row_id` を満たす経路を評価する。
        if str(row.get("row_id")) == row_id:
            return row

    raise KeyError(f"missing row_id: {row_id}")


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload(step_tag: str) -> Dict[str, Any]:
    for path in (
        GATE_JSON,
        READINESS_JSON,
        SHELL_CANON_JSON,
        SHELL_BRIDGE_JSON,
        REOPEN_JSON,
        LATENT_JSON,
        ACTION_JSON,
    ):
        _require_path(path)

    gate = _read_json(GATE_JSON)
    readiness = _read_json(READINESS_JSON)
    shell_canon = _read_json(SHELL_CANON_JSON)
    shell_bridge = _read_json(SHELL_BRIDGE_JSON)
    reopen = _read_json(REOPEN_JSON)
    latent = _read_json(LATENT_JSON)
    action = _read_json(ACTION_JSON)

    gate_rows = gate.get("rows", [])
    readiness_rows = readiness.get("rows", [])
    shell_rows = shell_canon.get("rows", [])
    bridge_rows = shell_bridge.get("rows", [])

    # 条件分岐: `not isinstance(gate_rows, list)` を満たす経路を評価する。
    if not isinstance(gate_rows, list):
        raise SystemExit(f"[fail] invalid rows in {GATE_JSON}")

    # 条件分岐: `not isinstance(readiness_rows, list)` を満たす経路を評価する。

    if not isinstance(readiness_rows, list):
        raise SystemExit(f"[fail] invalid rows in {READINESS_JSON}")

    # 条件分岐: `not isinstance(shell_rows, list)` を満たす経路を評価する。

    if not isinstance(shell_rows, list):
        raise SystemExit(f"[fail] invalid rows in {SHELL_CANON_JSON}")

    # 条件分岐: `not isinstance(bridge_rows, list)` を満たす経路を評価する。

    if not isinstance(bridge_rows, list):
        raise SystemExit(f"[fail] invalid rows in {SHELL_BRIDGE_JSON}")

    readiness_same_sector = _find_row_by_id(readiness_rows, "same_sector_chi_p_to_vpp_mapping")
    gate_same_sector_missing = _find_row_by_id(gate_rows, "same_sector_public_artifact_still_missing")
    shell_family_row = _find_row_by_id(shell_rows, "shell_quantization_family_public_candidate")
    shell_kappa_row = _find_row_by_id(shell_rows, "shell_quantization_fit_kappa")
    shell_kz_row = _find_row_by_id(shell_rows, "shell_quantization_fit_kz_over_kn")
    shell_bridge_row = _find_row_by_id(bridge_rows, "shell_to_curvature_bridge_ready")
    shell_coefficients_row = _find_row_by_id(bridge_rows, "shell_quantization_coefficients_not_vpp")

    gate_summary = gate.get("summary", {})
    reopen_decision = reopen.get("decision", {})
    latent_summary = latent.get("summary", {})
    action_equations = action.get("equations", {})

    minimal_spec_items = set(str(item) for item in gate_summary.get("same_sector_minimal_spec_items", []))
    same_particle_sector_only = "same_particle_sector_only" in minimal_spec_items
    explicit_mapping_equation_required = "explicit_mapping_equation" in minimal_spec_items
    sign_units_reference_required = "sign_units_and_reference_point" in minimal_spec_items
    public_shell_rows_required = "public_shell_family_numerical_rows" in minimal_spec_items
    no_new_free_parameter_note_required = "no_new_free_parameter_note" in minimal_spec_items

    same_sector_available = float(readiness_same_sector.get("value", 0.0)) > 0.0
    shell_family_contract_consistent = (
        str(shell_family_row.get("status", "")) == "candidate_public"
        and bool(gate_summary.get("single_public_boundary_family_fixed", False))
    )
    latent_same_sector_count = float(latent_summary.get("latent_positive_same_sector_public_row_count", 0.0))
    latent_routes_exhausted = bool(latent_summary.get("latent_reopen_routes_exhausted", False))

    required_public_row_fields = [
        "same_particle_sector_only",
        "explicit_mapping_equation",
        "sign_statement",
        "unit_statement",
        "reference_point_symbol_absP_star",
        "public_shell_family_numerical_rows",
    ]
    required_contract_annotations = ["no_new_free_parameter_note"]
    lagrangian = str(action_equations.get("lagrangian_density", ""))

    rows = [
        {
            "row_id": "contract_same_particle_sector_only",
            "status": "pass" if same_particle_sector_only else "reject",
            "metric": "contract frozen to same particle sector only",
            "value": 1.0 if same_particle_sector_only else 0.0,
            "note": "The future artifact must stay in the positive particle sector and may not use cross-sector parity or interface-only proxies as substitutes.",
        },
        {
            "row_id": "contract_observable_symbol_chi_p_frozen",
            "status": "pass",
            "metric": "same-sector observable symbol frozen",
            "value": 1.0,
            "note": "The observable side of the contract is frozen to chi_P, or an explicitly declared same-sector shell-family equivalent that is numerically linked back to chi_P.",
        },
        {
            "row_id": "contract_curvature_symbol_vpp_at_absP_star_frozen",
            "status": "pass",
            "metric": "curvature symbol frozen to V''(|P|_*)",
            "value": 1.0,
            "note": "The curvature side of the contract is frozen to V''(|P|_*) evaluated at the same-sector reference point |P|_*.",
        },
        {
            "row_id": "contract_reference_point_absP_star_frozen",
            "status": "pass" if sign_units_reference_required else "reject",
            "metric": "reference point |P|_* explicitly required",
            "value": 1.0 if sign_units_reference_required else 0.0,
            "note": "The future mapping must name |P|_* explicitly and may not leave the curvature evaluation point implicit.",
        },
        {
            "row_id": "contract_sign_statement_required",
            "status": "pass" if sign_units_reference_required else "reject",
            "metric": "explicit sign statement required",
            "value": 1.0 if sign_units_reference_required else 0.0,
            "note": "The future artifact must state the sign convention directly rather than importing it from a cross-sector or phenomenological proxy.",
        },
        {
            "row_id": "contract_unit_statement_required",
            "status": "pass" if sign_units_reference_required else "reject",
            "metric": "explicit unit statement required",
            "value": 1.0 if sign_units_reference_required else 0.0,
            "note": "The future artifact must state the units on both chi_P and V''(|P|_*) so the mapping is numerically checkable in public canonical form.",
        },
        {
            "row_id": "contract_explicit_mapping_equation_still_missing",
            "status": str(readiness_same_sector.get("status", "missing")),
            "metric": "explicit chi_P -> V''(|P|_*) mapping equation already available",
            "value": float(readiness_same_sector.get("value", 0.0)),
            "note": str(readiness_same_sector.get("note", "")),
        },
        {
            "row_id": "contract_public_shell_family_rows_required",
            "status": "pass" if public_shell_rows_required and shell_family_contract_consistent else "reject",
            "metric": "public shell-family numerical rows required",
            "value": 2.0 if public_shell_rows_required and shell_family_contract_consistent else 0.0,
            "note": (
                "The future artifact must reuse the surviving shell-family numerical anchors, currently "
                f"kappa={shell_kappa_row.get('value')} and kZ_over_kN={shell_kz_row.get('value')}, rather than introducing a new fit family."
            ),
        },
        {
            "row_id": "contract_no_new_free_parameter_note_required",
            "status": "pass" if no_new_free_parameter_note_required else "reject",
            "metric": "no-new-free-parameter wording required",
            "value": 1.0 if no_new_free_parameter_note_required else 0.0,
            "note": "The future artifact must say explicitly that the same-sector curvature map does not add a new free parameter beyond the surviving shell family.",
        },
        {
            "row_id": "contract_shell_family_anchor_pack_consistent",
            "status": "pass" if shell_family_contract_consistent else "reject",
            "metric": "surviving shell-family anchor pack consistent with contract",
            "value": 1.0 if shell_family_contract_consistent else 0.0,
            "note": (
                f"Current anchors are shell family status={shell_family_row.get('status', '')}, "
                f"kappa={shell_kappa_row.get('value')}, kZ_over_kN={shell_kz_row.get('value')}, "
                f"and action `{lagrangian}`."
            ),
        },
        {
            "row_id": "contract_shell_bridge_still_missing",
            "status": str(shell_bridge_row.get("status", "reject")),
            "metric": "same-sector shell-family bridge already available",
            "value": float(shell_bridge_row.get("value", 0.0)),
            "note": (
                "The contract is frozen, but the bridge is still absent. "
                f"Current shell-coefficient note remains: {shell_coefficients_row.get('note', '')}"
            ),
        },
        {
            "row_id": "contract_repo_wide_same_sector_route_still_absent",
            "status": "reject" if latent_same_sector_count == 0.0 else "pass",
            "metric": "repo-wide same-sector public route already present",
            "value": latent_same_sector_count,
            "note": (
                f"Repo-wide positive same-sector public row count is {latent_same_sector_count}; "
                f"latent route exhaustion is {latent_routes_exhausted}."
            ),
        },
        {
            "row_id": "contract_blocked_state_detail_consistent",
            "status": "pass" if str(reopen_decision.get('blocked_state_detail', '')) == "specific_missing_artifacts_fixed" else "reject",
            "metric": "blocked-state detail consistent with contract freeze",
            "value": 1.0 if str(reopen_decision.get('blocked_state_detail', '')) == "specific_missing_artifacts_fixed" else 0.0,
            "note": "The contract freeze refines a named missing artifact; it does not reopen the branch by itself.",
        },
    ]

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": step_tag,
            "name": "same-sector chi_P -> V''(|P|_*) contract freeze",
        },
        "inputs": {
            "mass_origin_same_sector_vpp_shape_gate_json": _relative_str(GATE_JSON),
            "mass_origin_readiness_gate_json": _relative_str(READINESS_JSON),
            "mass_origin_shell_quantization_canonicalization_json": _relative_str(SHELL_CANON_JSON),
            "mass_origin_shell_curvature_bridge_json": _relative_str(SHELL_BRIDGE_JSON),
            "mass_origin_blocked_state_reopen_json": _relative_str(REOPEN_JSON),
            "mass_origin_latent_reopen_route_inventory_json": _relative_str(LATENT_JSON),
            "action_principle_el_derivation_audit_json": _relative_str(ACTION_JSON),
        },
        "intent": "Freeze the public-canonical contract that any future positive particle-sector chi_P -> V''(|P|_*) artifact must satisfy.",
        "formulas": {
            "contract_requirement": "same-sector chi_P or an explicitly declared shell-family equivalent -> V''(|P|_*) with explicit sign, units, and |P|_* reference point",
            "shell_family_binding": "public shell-family numerical rows (kappa, kZ_over_kN) must be reused rather than replaced by a new fit family",
            "reopen_dependency": "contract freeze alone does not reopen the branch; reopen still requires the realized public artifact plus single_public_vpp_shape and solver-ready promotion",
        },
        "rows": rows,
        "summary": {
            "same_particle_sector_only": same_particle_sector_only,
            "chi_symbol": "chi_P",
            "curvature_symbol": "V''(|P|_*)",
            "reference_point_symbol": "|P|_*",
            "chi_to_vpp_mapping_contract_frozen": (
                same_particle_sector_only
                and explicit_mapping_equation_required
                and sign_units_reference_required
                and public_shell_rows_required
            ),
            "sign_convention_frozen": sign_units_reference_required,
            "unit_contract_frozen": sign_units_reference_required,
            "shell_family_contract_consistent": shell_family_contract_consistent,
            "required_public_row_fields": required_public_row_fields,
            "required_public_row_field_count": len(required_public_row_fields),
            "required_contract_annotations": required_contract_annotations,
            "explicit_mapping_equation_available": same_sector_available,
            "named_missing_artifact": "positive_particle_sector_chi_p_to_vpp_public_artifact",
            "existing_shell_family_row_ids": [
                "shell_quantization_fit_kappa",
                "shell_quantization_fit_kz_over_kn",
            ],
        },
        "decision": {
            "overall_status": "same_sector_chi_to_vpp_contract_frozen",
            "keep_mass_origin_branch_blocked": True,
            "same_particle_sector_only": same_particle_sector_only,
            "chi_to_vpp_mapping_contract_frozen": (
                same_particle_sector_only
                and explicit_mapping_equation_required
                and sign_units_reference_required
                and public_shell_rows_required
            ),
            "sign_convention_frozen": sign_units_reference_required,
            "unit_contract_frozen": sign_units_reference_required,
            "shell_family_contract_consistent": shell_family_contract_consistent,
            "explicit_mapping_equation_available": same_sector_available,
            "blocked_state_detail": str(reopen_decision.get("blocked_state_detail", "")),
            "next_required_artifacts": reopen_decision.get(
                "next_required_artifacts",
                [
                    "positive_particle_sector_chi_p_to_vpp_public_artifact",
                    "single_public_vpp_shape",
                    "solver_ready_row_promoted_to_pass",
                ],
            ),
        },
        "evidence": {
            "gate_same_sector_missing_row": gate_same_sector_missing,
            "readiness_same_sector_row": readiness_same_sector,
            "shell_family_row": shell_family_row,
            "shell_kappa_row": shell_kappa_row,
            "shell_kz_over_kn_row": shell_kz_row,
            "shell_bridge_row": shell_bridge_row,
            "shell_coefficients_row": shell_coefficients_row,
            "gate_summary": gate_summary,
            "reopen_decision": reopen_decision,
            "latent_inventory_summary": latent_summary,
            "action_lagrangian_density": lagrangian,
        },
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(rows: List[Dict[str, Any]]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
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
