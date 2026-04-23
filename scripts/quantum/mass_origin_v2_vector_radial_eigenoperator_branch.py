#!/usr/bin/env python3
"""
Generate working-action vector radial-eigenoperator artifacts for 8.7.56.174-.177.

This branch takes the honest non-closure of the working-action vector rebuild and
identifies the next deeper blocker needed before a numeric current-canon vector
mass spectrum can be claimed again.
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
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"

TRIAL1_VEV = OUT / "mass_origin_v2_trial1_breakthrough_modified_vev_decomposition_metrics.json"
TRIAL1_MAXWELL = OUT / "mass_origin_v2_trial1_breakthrough_maxwell_coupling_audit_metrics.json"
TRIAL2_DECLARATION = OUT / "mass_origin_v2_trial2_declaration_gate_metrics.json"
QBALL_RADIAL = OUT / "mass_origin_qball_radial_equation_derivation_metrics.json"
VECTOR_SEPARATION = OUT / "mass_origin_vector_qball_radial_angular_separation_metrics.json"
VECTOR_SOLVER_SPEC = OUT / "mass_origin_vector_qball_solver_spec_metrics.json"
VECTOR_TRIAL_STATES = OUT / "mass_origin_vector_qball_trial_state_inventory_metrics.json"
VECTOR_FULL_COUPLED = OUT / "mass_origin_vector_qball_full_coupled_solver_pilot_metrics.json"
VECTOR_REBUILD_SOURCE = OUT / "mass_origin_v2_vector_rebuild_source_inventory_metrics.json"
VECTOR_REDUCED_SOLVER = OUT / "mass_origin_v2_vector_reduced_solver_pilot_metrics.json"
VECTOR_ANCHOR = OUT / "mass_origin_v2_vector_anchor_refresh_metrics.json"
VECTOR_GATE = OUT / "mass_origin_v2_vector_rebuild_declaration_gate_metrics.json"
VECTOR_ROUTE = OUT / "mass_origin_v2_vector_radial_eigenoperator_route_contract_metrics.json"
TRIAL3_FALLBACK_ROUTE = OUT / "mass_origin_v2_trial3_explicit_k_positive_extension_route_contract_metrics.json"


# Function: return the current UTC timestamp in ISO 8601 format.
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Function: stop execution if a required path is missing.

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


# Function: build a standard inventory target record.

def target_record(file_key: str, path: Path, text: str, pattern: str, note: str) -> dict:
    target_hit = hit(text, pattern)
    return {
        "file_key": file_key,
        "file": rel(path),
        "pattern": pattern,
        "present": target_hit is not None,
        "note": note,
        "evidence": target_hit,
    }


# Function: execute the working-action vector radial-eigenoperator residual branch.

def main() -> None:
    for path in (
        PART1,
        PART3A,
        PART5,
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        TRIAL1_VEV,
        TRIAL1_MAXWELL,
        TRIAL2_DECLARATION,
        QBALL_RADIAL,
        VECTOR_SEPARATION,
        VECTOR_SOLVER_SPEC,
        VECTOR_TRIAL_STATES,
        VECTOR_FULL_COUPLED,
        VECTOR_REBUILD_SOURCE,
        VECTOR_REDUCED_SOLVER,
        VECTOR_ANCHOR,
        VECTOR_GATE,
        VECTOR_ROUTE,
        TRIAL3_FALLBACK_ROUTE,
    ):
        req(path)

    part1 = read_text(PART1)
    part3a = read_text(PART3A)
    part5 = read_text(PART5)
    status = read_text(STATUS)
    roadmap = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)

    trial1_vev = read_json(TRIAL1_VEV)
    trial1_maxwell = read_json(TRIAL1_MAXWELL)
    trial2_declaration = read_json(TRIAL2_DECLARATION)
    qball_radial = read_json(QBALL_RADIAL)
    vector_separation = read_json(VECTOR_SEPARATION)
    vector_solver_spec = read_json(VECTOR_SOLVER_SPEC)
    vector_trial_states = read_json(VECTOR_TRIAL_STATES)
    vector_full_coupled = read_json(VECTOR_FULL_COUPLED)
    vector_rebuild_source = read_json(VECTOR_REBUILD_SOURCE)
    vector_reduced_solver = read_json(VECTOR_REDUCED_SOLVER)
    vector_anchor = read_json(VECTOR_ANCHOR)
    vector_gate = read_json(VECTOR_GATE)
    vector_route = read_json(VECTOR_ROUTE)
    trial3_fallback_route = read_json(TRIAL3_FALLBACK_ROUTE)

    common_inputs = {
        "part1_markdown": rel(PART1),
        "part3a_markdown": rel(PART3A),
        "part5_markdown": rel(PART5),
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "mass_origin_v2_trial1_breakthrough_modified_vev_decomposition_json": rel(TRIAL1_VEV),
        "mass_origin_v2_trial1_breakthrough_maxwell_coupling_audit_json": rel(TRIAL1_MAXWELL),
        "mass_origin_v2_trial2_declaration_gate_json": rel(TRIAL2_DECLARATION),
        "mass_origin_qball_radial_equation_derivation_json": rel(QBALL_RADIAL),
        "mass_origin_vector_qball_radial_angular_separation_json": rel(VECTOR_SEPARATION),
        "mass_origin_vector_qball_solver_spec_json": rel(VECTOR_SOLVER_SPEC),
        "mass_origin_vector_qball_trial_state_inventory_json": rel(VECTOR_TRIAL_STATES),
        "mass_origin_vector_qball_full_coupled_solver_pilot_json": rel(VECTOR_FULL_COUPLED),
        "mass_origin_v2_vector_rebuild_source_inventory_json": rel(VECTOR_REBUILD_SOURCE),
        "mass_origin_v2_vector_reduced_solver_pilot_json": rel(VECTOR_REDUCED_SOLVER),
        "mass_origin_v2_vector_anchor_refresh_json": rel(VECTOR_ANCHOR),
        "mass_origin_v2_vector_rebuild_declaration_gate_json": rel(VECTOR_GATE),
        "mass_origin_v2_vector_radial_eigenoperator_route_contract_json": rel(VECTOR_ROUTE),
        "mass_origin_v2_trial3_explicit_k_positive_extension_route_contract_json": rel(TRIAL3_FALLBACK_ROUTE),
    }

    working_action_mass_formula_ready = bool(
        trial1_vev["summary"]["working_action_uses_mexican_hat_only_mass_source"]
    ) and bool(trial1_vev["summary"]["transverse_mode_massless_under_breakthrough_action"])
    photon_split_ready = bool(trial1_maxwell["summary"]["transverse_maxwell_reduction_available"])
    scalar_boundary_pack_ready = bool(qball_radial["summary"]["qball_radial_equation_ready"]) and bool(
        qball_radial["summary"]["finite_energy_boundary_conditions_ready"]
    )
    vector_separation_pack_ready = bool(vector_separation["summary"]["vector_qball_ansatz_ready"]) and (
        int(vector_separation["summary"]["quantum_number_axis_count"]) == 4
    )
    vector_bookkeeping_pack_ready = bool(vector_rebuild_source["summary"]["trial_state_bookkeeping_pack_present"]) and (
        int(vector_trial_states["summary"]["trial_state_count"]) > 0
    )
    historic_full_coupled_pack_ready = bool(vector_full_coupled["summary"]["exact_full_coupled_vector_ladder_available"])

    source_targets = [
        target_record(
            "part1_total_action_anchor",
            PART1,
            part1,
            "\\mathcal{L}_{P,\\mathrm{full}}",
            "Part I still anchors the canonical full P-sector action that the rebuild must reinterpret under the breakthrough working action.",
        ),
        target_record(
            "status_next_step_anchor",
            STATUS,
            status,
            "current official next step は `8.7.56.174`",
            "STATUS must already expose the radial-eigenoperator residual branch as the next official route.",
        ),
        target_record(
            "roadmap_radial_eigenoperator_branch",
            ROADMAP,
            roadmap,
            "`8.7.56.173-.176`",
            "ROADMAP must already expose the radial-eigenoperator residual branch.",
        ),
    ]
    source_pack_ready = all(item["present"] for item in source_targets)

    source_inventory = payload(
        "8.7.56.174",
        "Working-action vector radial eigenoperator source inventory",
        common_inputs,
        "Freeze the source pack needed to decide whether a current-canon radial eigenoperator can be identified for the working-action vector rebuild.",
        {
            "operator_requirement": "a current-canon vector radial eigenoperator needs the working-action mass source, the massless transverse-photon split, scalar radial boundary conditions, vector quantum-number bookkeeping, and a historic benchmark template",
            "projection_requirement": "because the transverse mode is now the photon candidate, the rebuild must identify a nontransverse massive-sector projector before any radial operator can be written",
            "template_rule": "historic full-coupled and exact-ladder packs may be reused only as benchmark templates, not as already-preserved current-canon operators",
        },
        [
            row("vector_radial_operator_source_inventory_complete", "pass", "working-action vector radial eigenoperator source inventory complete", 1, "The source pack for the radial-eigenoperator residual branch is frozen."),
            row("vector_radial_operator_source_pack_ready", "pass" if source_pack_ready else "reject", "working-action vector radial eigenoperator source pack ready", 1 if source_pack_ready else 0, "The residual branch has the required control-doc anchors."),
            row("vector_radial_operator_working_action_mass_formula_ready", "pass" if working_action_mass_formula_ready else "reject", "working-action mass formula ready", 1 if working_action_mass_formula_ready else 0, "The breakthrough working action supplies the mexican-hat-only mass source and the massless transverse split."),
            row("vector_radial_operator_scalar_boundary_pack_ready", "pass" if scalar_boundary_pack_ready else "reject", "scalar radial boundary pack ready", 1 if scalar_boundary_pack_ready else 0, "The scalar Q-ball branch still supplies radial ODE and finite-energy boundary conditions."),
            row("vector_radial_operator_vector_separation_pack_ready", "pass" if vector_separation_pack_ready else "reject", "vector radial-angular separation pack ready", 1 if vector_separation_pack_ready else 0, "The old vector branch still supplies the `(n,k,ell,s)` separation skeleton."),
            row("vector_radial_operator_bookkeeping_pack_ready", "pass" if vector_bookkeeping_pack_ready else "reject", "vector bookkeeping pack ready", 1 if vector_bookkeeping_pack_ready else 0, "The rebuild may still reuse the trial-state bookkeeping as an index set."),
            row("vector_radial_operator_historic_full_coupled_template_ready", "pass" if historic_full_coupled_pack_ready else "reject", "historic full-coupled template ready", 1 if historic_full_coupled_pack_ready else 0, "The old full-coupled ladder survives only as a benchmark template."),
        ],
        {
            "working_action_vector_radial_eigenoperator_source_pack_ready": source_pack_ready,
            "working_action_mass_formula_ready": working_action_mass_formula_ready,
            "working_action_transverse_photon_split_ready": photon_split_ready,
            "scalar_radial_boundary_pack_ready": scalar_boundary_pack_ready,
            "vector_radial_angular_separation_pack_ready": vector_separation_pack_ready,
            "vector_bookkeeping_pack_ready": vector_bookkeeping_pack_ready,
            "historic_full_coupled_template_ready": historic_full_coupled_pack_ready,
            "first_route_to_close_or_none": "working_action_vector_radial_eigenoperator_identification_audit",
        },
        {
            "overall_status": "working_action_vector_radial_eigenoperator_source_inventory_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_175": True,
            "next_required_artifacts": ["working_action_vector_radial_eigenoperator_identification_audit"],
        },
        {
            "inventory_targets": source_targets,
            "trial1_vev_summary": trial1_vev["summary"],
            "trial1_maxwell_summary": trial1_maxwell["summary"],
            "qball_radial_summary": qball_radial["summary"],
            "vector_separation_summary": vector_separation["summary"],
            "vector_solver_spec_summary": vector_solver_spec["summary"],
            "vector_trial_states_summary": vector_trial_states["summary"],
            "vector_full_coupled_summary": vector_full_coupled["summary"],
            "vector_route_summary": vector_route["summary"],
            "ai_context_current_step": ai_context["current_step"],
        },
    )

    massive_sector_projector_available = False
    radial_eigenoperator_identification_available = False
    mass_proxy_numeric_ready = False
    transverse_photon_subtraction_ready = False

    identification_audit = payload(
        "8.7.56.175",
        "Working-action vector radial eigenoperator identification audit",
        common_inputs,
        "Audit whether the frozen source pack is already sufficient to identify a current-canon radial eigenoperator, or whether a deeper missing artifact still blocks numeric rebuild.",
        {
            "projector_rule": "once A_mu = delta P_mu^T / sqrt(Z_P) is adopted as the photon candidate, the rebuild needs an explicit projector from delta P_mu to the remaining massive nontransverse sector",
            "operator_rule": "a radial eigenoperator can be claimed only if that massive-sector projector and its boundary/regularity statement are both current-canon visible",
            "benchmark_rule": "historic full-coupled templates do not themselves count as a current-canon operator because they were built before the breakthrough working action removed the separate Proca mass source",
        },
        [
            row("vector_radial_operator_scalar_boundary_pack_reusable", "pass" if scalar_boundary_pack_ready else "reject", "scalar radial boundary pack reusable", 1 if scalar_boundary_pack_ready else 0, "The scalar branch still supplies the base radial ODE and finite-energy boundary conditions."),
            row("vector_radial_operator_vector_bookkeeping_reusable", "pass" if vector_bookkeeping_pack_ready else "reject", "vector bookkeeping reusable", 1 if vector_bookkeeping_pack_ready else 0, "The old `(n,k,ell,s)` bookkeeping remains available as an index set."),
            row("vector_radial_operator_historic_template_reusable_as_benchmark", "pass" if historic_full_coupled_pack_ready else "reject", "historic full-coupled template reusable as benchmark", 1 if historic_full_coupled_pack_ready else 0, "The old full-coupled ladder remains usable only as a benchmark template."),
            row("vector_radial_operator_transverse_photon_subtraction_ready", "reject", "transverse photon subtraction ready", 0, "Current canon does not yet expose the explicit subtraction/projector that removes the photon branch from the rebuild operator."),
            row("vector_radial_operator_massive_sector_projector_available", "reject", "massive nontransverse sector projector available", 0, "No current-canon statement identifies the remaining massive sector after the transverse photon is split off."),
            row("vector_radial_operator_identification_available", "reject", "working-action vector radial eigenoperator identification available", 0, "Without the massive-sector projector, no current-canon radial eigenoperator can be claimed."),
            row("vector_radial_operator_mass_proxy_numeric_ready", "reject", "working-action vector mass proxy numeric ready after audit", 0, "Numeric mass proxies remain unavailable while the operator itself is missing."),
        ],
        {
            "transverse_photon_subtraction_ready": transverse_photon_subtraction_ready,
            "working_action_massive_sector_projector_available": massive_sector_projector_available,
            "working_action_vector_radial_eigenoperator_identification_available": radial_eigenoperator_identification_available,
            "working_action_vector_mass_proxy_numeric_ready": mass_proxy_numeric_ready,
            "identification_nonclosure_reason_or_none": "working_action_massive_nontransverse_sector_projector_absent",
            "first_route_to_close_or_none": "working_action_vector_rebuild_reopen_gate",
        },
        {
            "overall_status": "working_action_vector_radial_eigenoperator_identification_nonclosure_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_176": True,
            "next_required_artifacts": ["working_action_vector_rebuild_reopen_gate"],
        },
        {
            "source_inventory_summary": source_inventory["summary"],
            "vector_reduced_solver_summary": vector_reduced_solver["summary"],
            "vector_anchor_summary": vector_anchor["summary"],
            "vector_gate_summary": vector_gate["summary"],
            "vector_route_summary": vector_route["summary"],
            "qball_radial_formulas": qball_radial["formulas"],
            "vector_separation_formulas": vector_separation["formulas"],
            "vector_full_coupled_formulas": vector_full_coupled["formulas"],
        },
    )

    reopen_gate = payload(
        "8.7.56.176",
        "Working-action vector rebuild reopen gate / Trial-3 fallback third refresh",
        common_inputs,
        "Integrate the radial-eigenoperator audit and decide whether the working-action vector rebuild can reopen numerically or whether a deeper projector residual route must be selected.",
        {
            "reopen_rule": "the vector rebuild reopens only if a current-canon radial eigenoperator becomes identifiable",
            "anchor_rule": "anchor refresh can reopen only after the same operator supplies numeric mass proxies",
            "fallback_rule": "Trial-3 remains on fallback hold while the vector branch still lacks the massive-sector projector behind the radial eigenoperator",
        },
        [
            row("vector_rebuild_reopen_gate_source_inventory_ready", "pass" if source_inventory["summary"]["working_action_vector_radial_eigenoperator_source_pack_ready"] else "reject", "working-action vector radial eigenoperator source inventory ready at gate", 1 if source_inventory["summary"]["working_action_vector_radial_eigenoperator_source_pack_ready"] else 0, "The radial-eigenoperator residual branch has its source pack frozen."),
            row("vector_rebuild_reopen_gate_identification_ready", "reject", "working-action vector radial eigenoperator identification ready at gate", 0, "The current canon still lacks the massive-sector projector needed by the operator."),
            row("vector_rebuild_reopen_gate_anchor_refresh_ready", "reject", "working-action vector anchor refresh ready at third gate", 0, "Anchor refresh remains blocked until the operator yields numeric mass proxies."),
            row("vector_rebuild_reopen_gate_trial3_hold_retained", "pass", "Trial-3 fallback hold retained at third gate", 1, "The weak-sector branch remains downstream of the unresolved projector/operator route."),
        ],
        {
            "working_action_vector_rebuild_reopen_ready": False,
            "working_action_vector_mass_spectrum_reduced_solver_numeric_ready": False,
            "working_action_vector_mass_spectrum_anchor_refresh_ready": False,
            "trial2_paper_side_sync_deferred_until_vector_anchor_refresh": True,
            "trial3_fallback_hold_retained": True,
            "trial3_fallback_hold_release_ready": False,
            "recommended_next_route_or_none": "8.7.56.177",
        },
        {
            "overall_status": "working_action_vector_rebuild_reopen_gate_nonclosure_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_177": True,
            "next_required_artifacts": ["working_action_massive_sector_projector_identification"],
        },
        {
            "source_inventory_summary": source_inventory["summary"],
            "identification_audit_summary": identification_audit["summary"],
            "vector_anchor_summary": vector_anchor["summary"],
            "trial2_declaration_summary": trial2_declaration["summary"],
            "trial3_fallback_route_summary": trial3_fallback_route["summary"],
        },
    )

    route_contract = payload(
        "8.7.56.177",
        "Working-action massive-sector projector route contract",
        common_inputs,
        "Freeze the deeper residual route suggested by the radial-eigenoperator audit: identify the current-canon projector that removes the photon branch and isolates the remaining massive sector.",
        {
            "selected_residual_route": "working_action_massive_sector_projector_identification",
            "blocking_rule": "without an explicit projector from delta P_mu to the remaining massive nontransverse sector, no current-canon radial eigenoperator can be written",
            "dependency_rule": "Trial-3 and Trial-2 paper-side sync remain downstream of the same unresolved massive-sector projector",
        },
        [
            row("vector_massive_sector_projector_route_contract_complete", "pass", "working-action massive-sector projector route contract complete", 1, "The next residual route is frozen after the radial-eigenoperator non-closure."),
            row("vector_massive_sector_projector_missing", "pass", "working-action massive-sector projector missing", 1, "The missing artifact is the current-canon projector that isolates the massive nontransverse sector."),
            row("vector_massive_sector_projector_trial3_dependency_blocked", "pass", "Trial-3 dependency still blocked by massive-sector projector", 1, "The weak-sector branch remains blocked by the unresolved projector."),
            row("vector_massive_sector_projector_paper_sync_deferred", "pass", "Trial-2 paper-side sync still deferred by massive-sector projector", 1, "Paper-side sync remains deferred until the vector anchor refresh becomes current-canon ready."),
        ],
        {
            "selected_residual_route": "working_action_massive_sector_projector_identification",
            "missing_v2_artifact": "working_action_massive_nontransverse_sector_projector",
            "trial3_dependency_state": "blocked_by_working_action_massive_sector_projector",
            "trial2_paper_side_sync_state": "deferred_until_working_action_vector_anchor_refresh",
            "split_contract_ready": True,
            "recommended_next_route_or_none": "8.7.56.178",
        },
        {
            "overall_status": "working_action_massive_sector_projector_route_contract_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_178": True,
            "next_required_artifacts": [
                "working_action_massive_sector_projector_source_inventory",
                "working_action_massive_sector_projector_identification_audit",
                "working_action_vector_rebuild_reopen_fourth_gate",
            ],
        },
        {
            "reopen_gate_summary": reopen_gate["summary"],
            "identification_audit_summary": identification_audit["summary"],
            "vector_route_summary": vector_route["summary"],
            "trial3_fallback_route_summary": trial3_fallback_route["summary"],
            "trial1_maxwell_summary": trial1_maxwell["summary"],
        },
    )

    write_artifact("mass_origin_v2_vector_radial_eigenoperator_source_inventory", source_inventory)
    write_artifact("mass_origin_v2_vector_radial_eigenoperator_identification_audit", identification_audit)
    write_artifact("mass_origin_v2_vector_rebuild_reopen_gate", reopen_gate)
    write_artifact("mass_origin_v2_working_action_massive_sector_projector_route_contract", route_contract)

    print("[ok] wrote:")
    print(" - mass_origin_v2_vector_radial_eigenoperator_source_inventory_metrics.json")
    print(" - mass_origin_v2_vector_radial_eigenoperator_identification_audit_metrics.json")
    print(" - mass_origin_v2_vector_rebuild_reopen_gate_metrics.json")
    print(" - mass_origin_v2_working_action_massive_sector_projector_route_contract_metrics.json")


# Function: run the working-action vector radial-eigenoperator branch from the CLI.

if __name__ == "__main__":
    main()
