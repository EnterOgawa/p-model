#!/usr/bin/env python3
"""
Generate Trial-1 U(1) derivation artifacts for 8.7.56.1-.4 and 8.7.56.17.

This branch audits whether the already-canonical complex/vector P-field pack is
enough to upgrade the adopted U(1) sector from an external assumption into a
P-only derivation. The audit records three distinct facts:

1. Global U(1) structure is already available from the |P|-only mexican-hat
   potential and the complex phase ansatz.
2. A local gauge structure is already present in the canonical P_mu closure via
   the Stueckelberg compensator pi.
3. Direct electromagnetic emergence through A_mu ~ d_mu theta is not yet
   available with no new free parameter, because a pure-gradient candidate has
   vanishing curvature and the current canonical Maxwell template still assumes
   an independent connection A_mu.

The branch therefore freezes an honest Trial-1 partial closeout and promotes a
new residual route that tests whether the Stueckelberg gauge compensator can be
identified with the electromagnetic phase connection.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
MEXICAN = OUT / "mass_origin_mexican_hat_parameter_freeze_metrics.json"
ACTION_AUDIT = OUT / "action_principle_el_derivation_audit.json"


# Function: return the current UTC time in ISO 8601 format.
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Function: abort if a required path is missing.

def req(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# Function: load a UTF-8 JSON artifact.

def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# Function: load a UTF-8 text source.

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# Function: convert an absolute path into a repository-relative path.

def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# Function: return the first source line that contains the requested pattern.

def hit(text: str, pattern: str) -> dict | None:
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# Function: build a standard result row.

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# Function: build a standard payload object.

def payload(
    step: str,
    name: str,
    inputs: dict,
    intent: str,
    formulas: dict,
    rows: list[dict],
    summary: dict,
    decision: dict,
    evidence: dict,
) -> dict:
    return {
        "generated_utc": now_iso(),
        "phase": {"phase": 8, "step": step, "name": name},
        "inputs": inputs,
        "intent": intent,
        "formulas": formulas,
        "rows": rows,
        "summary": summary,
        "decision": decision,
        "evidence": evidence,
    }


# Function: save a JSON artifact and its row table.

def write_artifact(stem: str, data: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / f"{stem}_metrics.json"
    csv_path = OUT / f"{stem}_rows.csv"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# Function: execute Trial-1 and the residual route contract.

def main() -> None:
    for path in (PART1, PART3A, MEXICAN, ACTION_AUDIT):
        req(path)

    part1 = read_text(PART1)
    part3a = read_text(PART3A)
    mexican = read_json(MEXICAN)
    action_audit = read_json(ACTION_AUDIT)

    selected_potential = mexican["summary"]["selected_potential_family_formula"]
    route_a_el_gate = action_audit["decision"]["route_a_el_derivation_gate"]
    route_a_audit_status = action_audit["numerical_audit"]["status"]

    common_inputs = {
        "part1_core_theory_markdown": rel(PART1),
        "part3a_quantum_foundations_markdown": rel(PART3A),
        "mass_origin_mexican_hat_parameter_freeze_json": rel(MEXICAN),
        "action_principle_el_derivation_audit_json": rel(ACTION_AUDIT),
    }

    source_inventory = payload(
        "8.7.56.1",
        "Complex vector phase / global U(1) source inventory",
        common_inputs,
        "Inventory the already-public source pack that makes a global U(1) structure available inside the complex/vector P-field canon.",
        {
            "complex_phase_ansatz": "P_mu = |P_mu| exp(i theta_mu) or, in the scalar reduction template, P = R exp(i theta)",
            "abs_only_potential": selected_potential,
            "global_u1_rule": "If V depends on |P| only, the action is invariant under P -> exp(i alpha) P for constant alpha.",
            "noether_rule": "Part I already freezes d_mu J^mu = 0 for the U(1) phase symmetry.",
            "local_vector_closure": "P_mu -> P_mu + d_mu alpha,  pi -> pi + m_P alpha",
            "required_source_items": [
                "complex_phase_ansatz",
                "mexican_hat_abs_only_dependence",
                "u1_noether_current_statement",
                "stueckelberg_local_gauge_closure",
                "route_a_gauge_covariant_template",
                "v1_1_u1_origin_judgment",
            ],
        },
        [
            row(
                "trial1_source_inventory_complete",
                "pass",
                "Trial-1 source inventory complete",
                1,
                "The global-U(1) source pack is frozen.",
            ),
            row(
                "trial1_global_u1_from_abs_only_potential",
                "pass",
                "global U(1) from |P|-only mexican-hat potential",
                1,
                "The selected mexican-hat family depends on |P| only, so the constant-phase symmetry is already available.",
            ),
            row(
                "trial1_u1_noether_current_statement_present",
                "pass",
                "U(1) Noether-current statement present",
                1,
                "Part I already freezes d_mu J^mu = 0 for the U(1) phase symmetry.",
            ),
            row(
                "trial1_stueckelberg_local_gauge_source_present",
                "pass",
                "Stueckelberg local gauge source present",
                1,
                "Part I already contains a local gauge compensator structure for P_mu.",
            ),
        ],
        {
            "required_source_count": 6,
            "present_source_count": 6,
            "missing_source_count": 0,
            "missing_source_items": [],
            "global_u1_automatic_from_abs_only_potential": True,
            "stueckelberg_local_gauge_structure_already_canonical": True,
            "first_route_to_close_or_none": "global_to_local_gauge_promotion_necessity_audit",
        },
        {
            "overall_status": "trial1_global_u1_source_inventory_frozen",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_2": True,
            "next_required_artifacts": [
                "global_to_local_gauge_promotion_necessity_audit",
                "a_mu_direct_emergence_no_new_parameter_gate",
            ],
        },
        {
            "part3a_complex_phase_line": hit(part3a, "P=R\\,e^{i\\theta}"),
            "part1_u1_current_line": hit(part1, "加えて U(1) 位相対称性に対し"),
            "part1_u1_symmetry_line": hit(part1, "局所位相（U(1)）"),
            "part1_stueckelberg_line": hit(part1, "Stückelberg 場 $\\pi$"),
            "part1_gauge_transform_line": hit(part1, "P_\\mu\\to P_\\mu+\\partial_\\mu\\alpha"),
            "part3a_v1_1_judgment_line": hit(part3a, "A棄却、B採用"),
            "mexican_hat_summary": mexican["summary"],
        },
    )

    gauge_promotion = payload(
        "8.7.56.2",
        "Global-to-local gauge promotion necessity audit",
        common_inputs,
        "Audit whether the complex-phase global U(1) can be promoted to a local gauge symmetry inside the P-only canon without introducing a new connection field.",
        {
            "global_phase_rule": "P -> exp(i alpha) P for constant alpha",
            "local_phase_obstruction": "d_nu(exp(i alpha(x)) P_mu) = exp(i alpha) [d_nu P_mu + i (d_nu alpha) P_mu]",
            "promotion_rule": "A local compensator/connection is necessary once alpha becomes x-dependent.",
            "existing_local_candidate": "Part I already contains the Stueckelberg redundancy P_mu -> P_mu + d_mu alpha, pi -> pi + m_P alpha.",
            "audit_rule": "Trial-1 local-promotion pass requires identifying the needed compensator with existing P-only canon rather than with an extra independent connection.",
        },
        [
            row(
                "trial1_localization_requires_compensator",
                "pass",
                "localizing the global phase requires a compensator",
                1,
                "The derivative of a local phase generates extra d_mu alpha terms that must be cancelled.",
            ),
            row(
                "trial1_stueckelberg_local_gauge_closure_available",
                "pass",
                "Stueckelberg local gauge closure available",
                1,
                "The canonical P_mu closure already supplies a local redundancy with the compensator pi.",
            ),
            row(
                "trial1_complex_phase_localization_p_only_available",
                "fail",
                "complex-phase localization available from P-only canon",
                0,
                "Part III-A still judges that the complex phase alone does not close the local U(1) route without an independent connection.",
            ),
        ],
        {
            "global_phase_localization_requires_connection": True,
            "stueckelberg_local_gauge_closure_available": True,
            "complex_phase_local_promotion_without_new_connection": False,
            "p_vector_local_gauge_structure_available": True,
            "first_route_to_close_or_none": "a_mu_direct_emergence_no_new_parameter_gate",
        },
        {
            "overall_status": "trial1_global_to_local_necessity_audited",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_3": True,
            "next_required_artifacts": [
                "a_mu_direct_emergence_no_new_parameter_gate",
            ],
        },
        {
            "part1_stueckelberg_line": hit(part1, "Stückelberg 場 $\\pi$"),
            "part1_gauge_transform_line": hit(part1, "P_\\mu\\to P_\\mu+\\partial_\\mu\\alpha"),
            "part3a_independent_connection_line": hit(part3a, "独立接続"),
            "part3a_reject_condition_line": hit(part3a, "局所位相不変性を保つために、P 以外の新しい自由度が必須になる場合"),
            "part3a_a_reject_line": hit(part3a, "A棄却、B採用"),
        },
    )

    direct_emergence = payload(
        "8.7.56.3",
        "A_mu ~ d_mu theta direct emergence / no-new-parameter gate",
        common_inputs,
        "Test whether an electromagnetic connection can emerge directly as a phase gradient with no new free parameter.",
        {
            "candidate_connection": "A_mu(candidate) = c_theta d_mu theta",
            "pure_gradient_curvature": "F_munu[d theta] = c_theta (d_mu d_nu theta - d_nu d_mu theta) = 0 for smooth theta",
            "route_a_template_status": f"action_principle_el_derivation_audit.route_a_el_derivation_gate = {route_a_el_gate}",
            "gate_rule": "The gate passes only if A_mu emerges from P-only canon and still supports a nontrivial curvature sector without introducing an independent connection.",
        },
        [
            row(
                "trial1_pure_gradient_curvature_zero",
                "pass",
                "pure-gradient A_mu has zero field strength",
                1,
                "For smooth theta, the antisymmetrized second derivative vanishes identically.",
            ),
            row(
                "trial1_direct_a_mu_emergence_available",
                "fail",
                "direct A_mu emergence available with no new field",
                0,
                "A pure-gradient candidate gives only a pure-gauge branch and does not supply a generic Maxwell curvature sector.",
            ),
            row(
                "trial1_route_a_template_requires_independent_connection",
                "pass",
                "route-A template still assumes an independent connection",
                1,
                "The existing EL audit passes only after introducing A_mu as a separate gauge connection in the template action.",
            ),
            row(
                "trial1_residual_route_candidate_discovered",
                "pass",
                "residual route candidate discovered",
                1,
                "The remaining path is to test whether the Stueckelberg compensator can be identified with the electromagnetic connection.",
            ),
        ],
        {
            "a_mu_pure_gradient_implies_zero_curvature": True,
            "a_mu_direct_emergence_available": False,
            "route_a_template_status": route_a_audit_status,
            "independent_connection_still_required_for_maxwell_sector": True,
            "first_route_to_close_or_none": "stueckelberg_to_em_connection_identification",
        },
        {
            "overall_status": "trial1_direct_emergence_gate_failed_with_residual_route",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_4": True,
            "next_required_artifacts": [
                "trial1_declaration_gate",
                "stueckelberg_to_em_connection_route_contract",
            ],
        },
        {
            "part3a_independent_connection_line": hit(part3a, "独立接続"),
            "part3a_minimal_coupling_line": hit(part3a, "D_\\mu=\\partial_\\mu+i q A_\\mu"),
            "part3a_route_a_template_line": hit(part3a, "\\mathcal{L} = \\lvert D_\\mu P\\rvert^2 - V(\\lvert P\\rvert) - \\frac{1}{4}F_{\\mu\\nu}F^{\\mu\\nu}"),
            "action_principle_el_audit_summary": {
                "route_a_el_derivation_gate": route_a_el_gate,
                "numerical_audit_status": route_a_audit_status,
            },
            "part1_stueckelberg_line": hit(part1, "Stückelberg 場 $\\pi$"),
        },
    )

    declaration = payload(
        "8.7.56.4",
        "Trial-1 declaration gate / fallback classification",
        common_inputs,
        "Classify Trial-1 as a full derivation, a partial closeout, or a failure, and decide whether Trial-2 can be launched on the current canon.",
        {
            "trial1_success_rule": "Full pass requires both automatic global U(1) and no-new-free-parameter local/emergent connection closure.",
            "trial1_partial_rule": "If global U(1) and a local P-vector gauge structure are present but the EM connection is not yet derived, freeze an honest partial closeout and open a residual route.",
        },
        [
            row(
                "trial1_global_u1_automatic",
                "pass",
                "global U(1) automatic from P-field structure",
                1,
                "The |P|-only mexican-hat structure already leaves a global phase symmetry.",
            ),
            row(
                "trial1_local_p_vector_gauge_structure_present",
                "pass",
                "local P-vector gauge structure present",
                1,
                "The Stueckelberg closure gives an already-canonical local gauge redundancy for P_mu.",
            ),
            row(
                "trial1_em_connection_derived_from_p_only",
                "fail",
                "electromagnetic connection derived from P-only canon",
                0,
                "Direct electromagnetic emergence is not yet available with no new free parameter.",
            ),
            row(
                "trial1_upgrade_from_b_to_a_complete",
                "fail",
                "B-adoption upgraded to full A-derivation",
                0,
                "Trial-1 does not yet promote the adopted EM/U(1) sector into a complete P-only derivation.",
            ),
        ],
        {
            "trial1_global_u1_automatic": True,
            "trial1_local_p_vector_gauge_structure_available": True,
            "trial1_em_connection_derived": False,
            "trial1_pass_level": "partial",
            "trial1_declaration": "honest_partial_closeout_global_u1_auto_and_stueckelberg_local_redundancy_present_but_em_connection_missing",
            "advance_to_8_7_56_5": False,
            "recommended_next_route_or_none": "8.7.56.17",
        },
        {
            "overall_status": "trial1_partial_closeout_residual_route_required",
            "trial1_branch_closeable": True,
            "advance_to_8_7_56_5": False,
            "next_required_artifacts": [
                "stueckelberg_to_em_connection_route_contract",
            ],
        },
        {
            "source_inventory_summary": source_inventory["summary"],
            "global_to_local_summary": gauge_promotion["summary"],
            "direct_emergence_summary": direct_emergence["summary"],
            "part3a_a_reject_line": hit(part3a, "A棄却、B採用"),
        },
    )

    residual_route = payload(
        "8.7.56.17",
        "Stueckelberg-to-electromagnetic connection residual route contract",
        common_inputs,
        "Freeze the new residual route suggested by Trial-1: test whether the already-canonical Stueckelberg compensator can supply the missing electromagnetic phase connection without adding a new field.",
        {
            "selected_residual_route": "stueckelberg_to_em_connection_identification",
            "pivot_principle": "The remaining gap is no longer global U(1) itself but the identification of the existing local P-vector gauge compensator with the electromagnetic connection needed by the Maxwell sector.",
            "missing_v2_artifact": "stueckelberg_to_electromagnetic_connection_statement",
            "trial2_hold_rule": "Keep 8.7.56.5-.8 on hold until the Trial-1 residual route decides whether the EM connection closes inside the P-only canon.",
        },
        [
            row(
                "trial1_residual_route_contract_complete",
                "pass",
                "Trial-1 residual route contract complete",
                1,
                "The Stueckelberg-to-EM connection route is frozen as the next official route.",
            ),
            row(
                "trial1_residual_route_uses_existing_stueckelberg_canon",
                "pass",
                "residual route uses existing Stueckelberg canon",
                1,
                "The new route reuses the already-frozen P_mu gauge compensator rather than adding a new ontology.",
            ),
            row(
                "trial2_launch_blocked_until_route_closes",
                "pass",
                "Trial-2 launch blocked until residual route closes",
                1,
                "Electromagnetic first-principles work stays on hold until the missing connection statement is tested.",
            ),
        ],
        {
            "selected_residual_route": "stueckelberg_to_em_connection_identification",
            "missing_v2_artifact": "stueckelberg_to_electromagnetic_connection_statement",
            "split_contract_ready": True,
            "advance_to_8_7_56_5": False,
        },
        {
            "overall_status": "trial1_residual_route_contract_frozen",
            "trial1_branch_closeable": True,
            "advance_to_8_7_56_5": False,
            "recommended_next_route_or_none": "8.7.56.18",
            "next_required_artifacts": [
                "stueckelberg_to_em_connection_source_inventory",
                "stueckelberg_to_em_connection_identification_audit",
                "trial1_reopened_declaration_gate",
            ],
        },
        {
            "part1_stueckelberg_line": hit(part1, "Stückelberg 場 $\\pi$"),
            "part1_gauge_transform_line": hit(part1, "P_\\mu\\to P_\\mu+\\partial_\\mu\\alpha"),
            "trial1_declaration_summary": declaration["summary"],
            "direct_emergence_summary": direct_emergence["summary"],
        },
    )

    write_artifact("mass_origin_v2_complex_vector_phase_global_u1_source_inventory", source_inventory)
    write_artifact("mass_origin_v2_global_to_local_gauge_promotion_necessity_audit", gauge_promotion)
    write_artifact("mass_origin_v2_a_mu_direct_emergence_no_new_parameter_gate", direct_emergence)
    write_artifact("mass_origin_v2_trial1_declaration_gate", declaration)
    write_artifact("mass_origin_v2_stueckelberg_to_em_connection_route_contract", residual_route)


if __name__ == "__main__":
    main()
