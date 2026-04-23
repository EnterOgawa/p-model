#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mass_origin_curvature_boundary_audit.py

Step 8.7.55.2.2:
Freeze what is currently known about the particle-sector curvature mapping and
about the admissible boundary / quantization families for the mass-origin
branch. This step does not solve the eigenvalue problem. It formalizes which
families are already present as public artifacts, which remain script-only or
doc-only candidates, and whether a unique no-free-parameter solver can be
specified.

Inputs:
  - output/public/quantum/mass_origin_readiness_gate_metrics.json
  - output/public/quantum/action_principle_el_derivation_audit.json
  - output/public/quantum/gravity_quantum_differential_prediction_table_metrics.json
  - output/public/quantum/nuclear_binding_energy_frequency_mapping_interface_metrics.json
  - output/public/quantum/nuclear_binding_energy_frequency_mapping_deuteron_two_body_metrics.json
  - output/public/quantum/nuclear_effective_potential_canonical_metrics.json
  - output/public/quantum/particle_reflection_demo_metrics.json
  - doc/quantum/18_p_field_action_and_schrodinger_mapping.md

Outputs:
  - output/public/quantum/mass_origin_curvature_boundary_metrics.json
  - output/public/quantum/mass_origin_curvature_boundary_rows.csv

Assumptions:
  - The structural parity proxy (k_B T_env / chi_P)_parity from 8.7.55.1.4 is a
    cross-sector entry proxy only; it is not a particle-sector V''(|P|_*)
    determination.
  - The action-principle audit fixes the abstract Lagrangian shape
    L = |D_mu P|^2 - V(|P|) - 1/4 F^2, but not the concrete V(|P|).
  - Boundary / quantization candidates can be frozen as admissible families
    even when they do not yet define a unique solver.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[2]

READINESS_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_readiness_gate_metrics.json"
ACTION_JSON = ROOT / "output" / "public" / "quantum" / "action_principle_el_derivation_audit.json"
GRAVITY_DIFF_JSON = ROOT / "output" / "public" / "quantum" / "gravity_quantum_differential_prediction_table_metrics.json"
NUCLEAR_INTERFACE_JSON = ROOT / "output" / "public" / "quantum" / "nuclear_binding_energy_frequency_mapping_interface_metrics.json"
DEUTERON_BOUNDARY_JSON = ROOT / "output" / "public" / "quantum" / "nuclear_binding_energy_frequency_mapping_deuteron_two_body_metrics.json"
EFFECTIVE_POTENTIAL_JSON = ROOT / "output" / "public" / "quantum" / "nuclear_effective_potential_canonical_metrics.json"
PARTICLE_REFLECTION_JSON = ROOT / "output" / "public" / "quantum" / "particle_reflection_demo_metrics.json"
MASS_NOTE_MD = ROOT / "doc" / "quantum" / "18_p_field_action_and_schrodinger_mapping.md"
SHELL_SCRIPT = ROOT / "scripts" / "quantum" / "nuclear_a_dependence_mean_field.py"
SHELL_PUBLIC_METRICS = ROOT / "output" / "public" / "quantum" / "nuclear_a_dependence_hf_three_body_shell_quantization_metrics.json"
SHELL_PUBLIC_ASYM_METRICS = ROOT / "output" / "public" / "quantum" / "nuclear_a_dependence_hf_three_body_shell_quantization_asym_metrics.json"
OUT_JSON = ROOT / "output" / "public" / "quantum" / "mass_origin_curvature_boundary_metrics.json"
OUT_CSV = ROOT / "output" / "public" / "quantum" / "mass_origin_curvature_boundary_rows.csv"


# 関数: `_utc_now_iso` の入出力契約と処理意図を定義する。
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_require_path` の入出力契約と処理意図を定義する。

def _require_path(path: Path) -> None:
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: `_read_text` の入出力契約と処理意図を定義する。

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# 関数: `_find_row_by_id` の入出力契約と処理意図を定義する。

def _find_row_by_id(rows: List[Dict[str, Any]], row_id: str) -> Dict[str, Any]:
    for row in rows:
        # 条件分岐: `str(row.get("row_id")) == row_id` を満たす経路を評価する。
        if str(row.get("row_id")) == row_id:
            return row

    raise KeyError(f"missing row_id: {row_id}")


# 関数: `_relative_str` の入出力契約と処理意図を定義する。

def _relative_str(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: `_shell_candidate_state` の入出力契約と処理意図を定義する。

def _shell_candidate_state() -> Dict[str, Any]:
    script_text = _read_text(SHELL_SCRIPT)
    has_step_11 = "Step 7.13.15.11" in script_text and "nuclear_a_dependence_hf_three_body_shell_quantization_metrics.json" in script_text
    has_step_12 = "Step 7.13.15.12" in script_text and "nuclear_a_dependence_hf_three_body_shell_quantization_asym_metrics.json" in script_text

    return {
        "script_exists": SHELL_SCRIPT.exists(),
        "public_metrics_exists": SHELL_PUBLIC_METRICS.exists(),
        "public_asym_metrics_exists": SHELL_PUBLIC_ASYM_METRICS.exists(),
        "step_11_declared": has_step_11,
        "step_12_declared": has_step_12,
    }


# 関数: `_build_rows` の入出力契約と処理意図を定義する。

def _build_rows() -> Dict[str, Any]:
    for path in (
        READINESS_JSON,
        ACTION_JSON,
        GRAVITY_DIFF_JSON,
        NUCLEAR_INTERFACE_JSON,
        DEUTERON_BOUNDARY_JSON,
        EFFECTIVE_POTENTIAL_JSON,
        PARTICLE_REFLECTION_JSON,
        MASS_NOTE_MD,
        SHELL_SCRIPT,
    ):
        _require_path(path)

    readiness = _read_json(READINESS_JSON)
    action = _read_json(ACTION_JSON)
    gravity_diff = _read_json(GRAVITY_DIFF_JSON)
    nuclear_interface = _read_json(NUCLEAR_INTERFACE_JSON)
    deuteron_boundary = _read_json(DEUTERON_BOUNDARY_JSON)
    effective_potential = _read_json(EFFECTIVE_POTENTIAL_JSON)
    particle_reflection = _read_json(PARTICLE_REFLECTION_JSON)
    mass_note = _read_text(MASS_NOTE_MD)
    shell_state = _shell_candidate_state()

    readiness_rows = readiness.get("rows", [])
    gravity_rows = gravity_diff.get("rows", [])
    interface_rows = nuclear_interface.get("rows", [])
    # 条件分岐: `not isinstance(readiness_rows, list)` を満たす経路を評価する。
    if not isinstance(readiness_rows, list):
        raise SystemExit(f"[fail] invalid rows in {READINESS_JSON}")

    # 条件分岐: `not isinstance(gravity_rows, list)` を満たす経路を評価する。

    if not isinstance(gravity_rows, list):
        raise SystemExit(f"[fail] invalid rows in {GRAVITY_DIFF_JSON}")

    # 条件分岐: `not isinstance(interface_rows, list)` を満たす経路を評価する。

    if not isinstance(interface_rows, list):
        raise SystemExit(f"[fail] invalid rows in {NUCLEAR_INTERFACE_JSON}")

    same_sector_row = _find_row_by_id(readiness_rows, "same_sector_chi_p_to_vpp_mapping")
    structural_row = _find_row_by_id(gravity_rows, "decoherence_structural_parity_entry")

    action_gate = str(action.get("decision", {}).get("route_a_el_derivation_gate", "unknown"))
    action_lagrangian = str(action.get("equations", {}).get("lagrangian_density", ""))
    effective_positioning = effective_potential.get("model", {}).get("positioning", [])
    deuteron_notes = deuteron_boundary.get("square_well_example", {}).get("notes", [])
    deuteron_fits = deuteron_boundary.get("square_well_example", {}).get("fits_from_R", [])
    reflection_notes = particle_reflection.get("notes", [])
    note_has_oscillon = "oscillon" in mass_note.lower()
    note_has_qball = "q-ball" in mass_note.lower() or "q ball" in mass_note.lower()
    note_has_complex_field = "複素場" in mass_note or "complex field" in mass_note.lower()

    rows = [
        {
            "row_id": "cross_sector_curvature_proxy_frozen",
            "status": "entry_only",
            "family": "cross_sector_proxy",
            "metric": "(k_B T_env / chi_P)_parity",
            "value": float(structural_row.get("differential_prediction_value", 0.0)),
            "public_artifact": True,
            "note": "8.7.55.1.4 freezes a structural parity proxy, but readiness gate already classifies it as cross-sector only and not a particle-sector V''(|P|_*) mapping.",
        },
        {
            "row_id": "same_sector_curvature_mapping_particle_sector",
            "status": str(same_sector_row.get("status", "missing")),
            "family": "curvature_mapping",
            "metric": "chi_P -> V''(|P|_*) in particle sector",
            "value": float(same_sector_row.get("value", 0.0)),
            "public_artifact": False,
            "note": str(same_sector_row.get("note", "")),
        },
        {
            "row_id": "minimal_action_shape_fixed_but_abstract",
            "status": "pass" if action_gate == "pass" else "watch",
            "family": "potential_shape",
            "metric": "abstract action shape fixed",
            "value": 1.0 if action_gate == "pass" else 0.0,
            "public_artifact": True,
            "note": f"Action audit freezes `{action_lagrangian}`, but not the concrete V(|P|) coefficients or a particle-sector curvature map.",
        },
        {
            "row_id": "reflection_boundary_candidate_public",
            "status": "candidate_public",
            "family": "boundary_reflection",
            "metric": "Dirichlet reflection demo present",
            "value": float(particle_reflection.get('config', {}).get('steps', 0)),
            "public_artifact": True,
            "note": "Reflection / Dirichlet boundary is now a public toy artifact. It shows discrete modes from boundary closure, but it is not yet a unique particle-mass solver.",
        },
        {
            "row_id": "deuteron_two_body_boundary_candidate_public",
            "status": "candidate_public_interface",
            "family": "boundary_two_body",
            "metric": "illustrative two-body boundary family",
            "value": float(len(deuteron_fits)),
            "public_artifact": True,
            "note": "Deuteron two-body boundary condition is public and machine-readable, but its own note freezes it as an illustrative operational I/F, not a first-principles force model.",
        },
        {
            "row_id": "shell_quantization_candidate_script_branch",
            "status": (
                "candidate_public"
                if shell_state["public_metrics_exists"] and shell_state["public_asym_metrics_exists"]
                else ("candidate_script_only" if shell_state["script_exists"] else "missing")
            ),
            "family": "boundary_shell_quantization",
            "metric": "shell quantization branch canonicalized to public",
            "value": 1.0 if shell_state["public_metrics_exists"] and shell_state["public_asym_metrics_exists"] else 0.0,
            "public_artifact": bool(shell_state["public_metrics_exists"] and shell_state["public_asym_metrics_exists"]),
            "note": (
                "Shell quantization family is now a public canonical artifact from Step 7.13.15.11/.12."
                if shell_state["public_metrics_exists"] and shell_state["public_asym_metrics_exists"]
                else "Shell quantization family exists as Step 7.13.15.11/.12 in the source tree, but the expected public metrics are absent, so it cannot yet serve as a canonical mass-origin branch artifact."
            ),
        },
        {
            "row_id": "complex_field_oscillon_qball_candidate_doc_only",
            "status": "candidate_doc_only" if note_has_oscillon and note_has_qball and note_has_complex_field else "missing",
            "family": "boundary_complex_field",
            "metric": "oscillon/Q-ball stabilization note",
            "value": 1.0 if note_has_oscillon and note_has_qball and note_has_complex_field else 0.0,
            "public_artifact": False,
            "note": "The mass-origin note explicitly keeps oscillon/Q-ball and complex-field stabilization as admissible mechanisms, but there is no public artifact that turns them into a unique solver.",
        },
        {
            "row_id": "effective_potential_shape_is_phenomenological",
            "status": "watch",
            "family": "potential_shape",
            "metric": "phenomenological effective-potential branch",
            "value": float(len(effective_potential.get("results_by_dataset", []))),
            "public_artifact": True,
            "note": "The canonical nuclear effective-potential branch is public, but its own positioning says it is phenomenological and not a first-principles derivation of nuclear forces.",
        },
        {
            "row_id": "single_boundary_family_unique",
            "status": "reject",
            "family": "uniqueness_gate",
            "metric": "one admissible boundary / quantization family only",
            "value": 0.0,
            "public_artifact": False,
            "note": "At least four admissible families remain open: reflection, two-body boundary, shell quantization, and complex-field / Q-ball stabilization.",
        },
        {
            "row_id": "single_vpp_shape_unique",
            "status": "reject",
            "family": "uniqueness_gate",
            "metric": "one V(|P|) shape fixed without new free parameters",
            "value": 0.0,
            "public_artifact": False,
            "note": "The abstract action is fixed, but no unique V(|P|) ansatz is frozen in the same particle sector.",
        },
        {
            "row_id": "no_free_parameter_mass_solver_spec_ready",
            "status": "reject",
            "family": "solver_readiness",
            "metric": "ready to specify a unique mass-eigenvalue solver",
            "value": 0.0,
            "public_artifact": False,
            "note": "Observed targets and admissible families are frozen, but same-sector curvature, a unique potential shape, and a unique boundary family are all still open.",
        },
    ]

    return {
        "rows": rows,
        "action_gate": action_gate,
        "effective_positioning": effective_positioning,
        "deuteron_notes": deuteron_notes,
        "reflection_notes": reflection_notes,
        "shell_state": shell_state,
        "light_nuclei_count": len(interface_rows),
    }


# 関数: `_build_payload` の入出力契約と処理意図を定義する。

def _build_payload() -> Dict[str, Any]:
    bundle = _build_rows()
    rows = bundle["rows"]
    candidate_rows = [row for row in rows if str(row.get("status", "")).startswith("candidate")]
    public_candidate_rows = [row for row in candidate_rows if bool(row.get("public_artifact"))]
    missing_curvature = _find_row_by_id(rows, "same_sector_curvature_mapping_particle_sector")

    return {
        "generated_utc": _utc_now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.55.2.2",
            "name": "mass-origin curvature and boundary family audit",
        },
        "inputs": {
            "mass_origin_readiness_gate_json": _relative_str(READINESS_JSON),
            "action_principle_el_derivation_audit_json": _relative_str(ACTION_JSON),
            "gravity_quantum_differential_prediction_table_json": _relative_str(GRAVITY_DIFF_JSON),
            "nuclear_binding_energy_frequency_mapping_interface_json": _relative_str(NUCLEAR_INTERFACE_JSON),
            "nuclear_binding_energy_frequency_mapping_deuteron_two_body_json": _relative_str(DEUTERON_BOUNDARY_JSON),
            "nuclear_effective_potential_canonical_json": _relative_str(EFFECTIVE_POTENTIAL_JSON),
            "particle_reflection_demo_json": _relative_str(PARTICLE_REFLECTION_JSON),
            "mass_origin_note_md": _relative_str(MASS_NOTE_MD),
            "shell_quantization_script": _relative_str(SHELL_SCRIPT),
        },
        "intent": "Freeze which same-sector curvature and boundary / quantization families are actually available for the mass-origin branch, and whether they specify a unique no-free-parameter solver.",
        "formulas": {
            "mass_frequency_mapping": "m_* = ħ ω_* / c^2",
            "solver_need": "same-sector V''(|P|_*) + unique boundary / quantization rule -> discrete omega_* ladder",
            "cross_sector_proxy_only": "(k_B T_env / chi_P)_parity is structural and cross-sector; it is not yet the particle-sector V''(|P|_*) map",
        },
        "rows": rows,
        "evidence": {
            "effective_potential_positioning": bundle["effective_positioning"],
            "deuteron_two_body_notes": bundle["deuteron_notes"],
            "reflection_notes": bundle["reflection_notes"],
            "shell_quantization_state": bundle["shell_state"],
        },
        "summary": {
            "row_count": len(rows),
            "candidate_family_count": len(candidate_rows),
            "public_candidate_family_count": len(public_candidate_rows),
            "same_sector_curvature_fixed": False,
            "unique_boundary_family_fixed": False,
            "unique_potential_shape_fixed": False,
            "shell_quantization_public_canonical": bool(bundle["shell_state"]["public_metrics_exists"] and bundle["shell_state"]["public_asym_metrics_exists"]),
            "light_nuclei_interface_count": int(bundle["light_nuclei_count"]),
            "cross_sector_proxy_value": float(_find_row_by_id(rows, "cross_sector_curvature_proxy_frozen")["value"]),
            "same_sector_curvature_value": float(missing_curvature["value"]),
        },
        "decision": {
            "overall_status": "candidate_families_frozen_uniqueness_not_ready",
            "new_free_parameters_introduced": False,
            "same_sector_curvature_fixed": False,
            "unique_boundary_family_fixed": False,
            "unique_potential_shape_fixed": False,
            "proceed_to_no_free_parameter_mass_solver": False,
            "next_required_steps": [
                "8.7.55.2.3",
                "8.7.54.23",
            ],
        },
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = ["row_id", "status", "family", "metric", "value", "public_artifact", "note"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    payload = _build_payload()
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(OUT_CSV, payload["rows"])
    print(f"[ok] json: {OUT_JSON}")
    print(f"[ok] csv : {OUT_CSV}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
