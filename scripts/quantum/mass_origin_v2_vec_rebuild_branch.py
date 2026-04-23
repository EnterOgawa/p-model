#!/usr/bin/env python3
"""
Generate working-action vector mass-spectrum rebuild artifacts for 8.7.56.169-.173.

This branch takes the output of the vector re-audit execution branch and
formalizes the next honest step under the breakthrough working action.
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

MEXICAN_HAT = OUT / "mass_origin_mexican_hat_parameter_freeze_metrics.json"
TRIAL1_VEV = OUT / "mass_origin_v2_trial1_breakthrough_modified_vev_decomposition_metrics.json"
TRIAL1_MAXWELL = OUT / "mass_origin_v2_trial1_breakthrough_maxwell_coupling_audit_metrics.json"
TRIAL2_DECLARATION = OUT / "mass_origin_v2_trial2_declaration_gate_metrics.json"
VECTOR_REAUDIT_DECLARATION = OUT / "mass_origin_v2_post_breakthrough_vector_mass_spectrum_declaration_gate_metrics.json"
VECTOR_REAUDIT_ROUTE = OUT / "mass_origin_v2_post_breakthrough_vector_mass_spectrum_reaudit_route_contract_metrics.json"
VECTOR_SOURCE_INVENTORY = OUT / "mass_origin_v2_post_breakthrough_vector_mass_spectrum_action_sensitive_source_inventory_metrics.json"
VECTOR_PRESERVATION_AUDIT = OUT / "mass_origin_v2_post_breakthrough_vector_mass_spectrum_preservation_rebuild_audit_metrics.json"
VECTOR_SOLVER_SPEC = OUT / "mass_origin_vector_qball_solver_spec_metrics.json"
VECTOR_TRIAL_STATES = OUT / "mass_origin_vector_qball_trial_state_inventory_metrics.json"
VECTOR_EXACT = OUT / "mass_origin_vector_qball_exact_mass_table_handoff_retry_metrics.json"
VECTOR_HEAVY = OUT / "mass_origin_vector_qball_baryon_tau_neutron_fit_table_metrics.json"
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


# Function: execute the working-action vector rebuild branch.

def main() -> None:
    for path in (
        PART1,
        PART3A,
        PART5,
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        MEXICAN_HAT,
        TRIAL1_VEV,
        TRIAL1_MAXWELL,
        TRIAL2_DECLARATION,
        VECTOR_REAUDIT_DECLARATION,
        VECTOR_REAUDIT_ROUTE,
        VECTOR_SOURCE_INVENTORY,
        VECTOR_PRESERVATION_AUDIT,
        VECTOR_SOLVER_SPEC,
        VECTOR_TRIAL_STATES,
        VECTOR_EXACT,
        VECTOR_HEAVY,
        TRIAL3_FALLBACK_ROUTE,
    ):
        req(path)

    part1 = read_text(PART1)
    part3a = read_text(PART3A)
    part5 = read_text(PART5)
    status = read_text(STATUS)
    roadmap = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)

    mexican_hat = read_json(MEXICAN_HAT)
    trial1_vev = read_json(TRIAL1_VEV)
    trial1_maxwell = read_json(TRIAL1_MAXWELL)
    trial2_declaration = read_json(TRIAL2_DECLARATION)
    vector_reaudit_declaration = read_json(VECTOR_REAUDIT_DECLARATION)
    vector_reaudit_route = read_json(VECTOR_REAUDIT_ROUTE)
    vector_source_inventory = read_json(VECTOR_SOURCE_INVENTORY)
    vector_preservation_audit = read_json(VECTOR_PRESERVATION_AUDIT)
    vector_solver_spec = read_json(VECTOR_SOLVER_SPEC)
    vector_trial_states = read_json(VECTOR_TRIAL_STATES)
    vector_exact = read_json(VECTOR_EXACT)
    vector_heavy = read_json(VECTOR_HEAVY)
    trial3_fallback_route = read_json(TRIAL3_FALLBACK_ROUTE)

    common_inputs = {
        "part1_markdown": rel(PART1),
        "part3a_markdown": rel(PART3A),
        "part5_markdown": rel(PART5),
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "mass_origin_mexican_hat_parameter_freeze_json": rel(MEXICAN_HAT),
        "mass_origin_v2_trial1_breakthrough_modified_vev_decomposition_json": rel(TRIAL1_VEV),
        "mass_origin_v2_trial1_breakthrough_maxwell_coupling_audit_json": rel(TRIAL1_MAXWELL),
        "mass_origin_v2_trial2_declaration_gate_json": rel(TRIAL2_DECLARATION),
        "mass_origin_v2_post_breakthrough_vector_mass_spectrum_declaration_gate_json": rel(VECTOR_REAUDIT_DECLARATION),
        "mass_origin_v2_post_breakthrough_vector_mass_spectrum_reaudit_route_contract_json": rel(VECTOR_REAUDIT_ROUTE),
        "mass_origin_v2_post_breakthrough_vector_mass_spectrum_action_sensitive_source_inventory_json": rel(VECTOR_SOURCE_INVENTORY),
        "mass_origin_v2_post_breakthrough_vector_mass_spectrum_preservation_rebuild_audit_json": rel(VECTOR_PRESERVATION_AUDIT),
        "mass_origin_vector_qball_solver_spec_json": rel(VECTOR_SOLVER_SPEC),
        "mass_origin_vector_qball_trial_state_inventory_json": rel(VECTOR_TRIAL_STATES),
        "mass_origin_vector_qball_exact_mass_table_handoff_retry_json": rel(VECTOR_EXACT),
        "mass_origin_vector_qball_baryon_tau_neutron_fit_table_json": rel(VECTOR_HEAVY),
        "mass_origin_v2_trial3_explicit_k_positive_extension_route_contract_json": rel(TRIAL3_FALLBACK_ROUTE),
    }

    trial_sector_count = int(vector_trial_states["summary"]["trial_sector_count"])
    trial_state_count = int(vector_trial_states["summary"]["trial_state_count"])
    exact_candidate_count = int(vector_exact["summary"]["exact_ratio_candidate_count"])
    historic_best_exact = vector_exact["summary"]["best_exact_match_or_none"]
    historic_best_proton = vector_heavy["summary"]["best_proton_row_or_none"]
    historic_best_tau = vector_heavy["summary"]["best_tau_row_or_none"]
    historic_best_np = vector_heavy["summary"]["best_neutron_proton_pair_or_none"]

    source_targets = [
        target_record(
            "part1_old_explicit_mass_term",
            PART1,
            part1,
            "+\\frac{m_P^2}{2}P_\\mu P^\\mu",
            "Part I still contains the old explicit mass term that the working-action rebuild must not reuse naively.",
        ),
        target_record(
            "part1_stueckelberg_completed_mass_term",
            PART1,
            part1,
            "+\\frac{m_P^2}{2}\\left(P_\\mu-\\frac{1}{m_P}\\partial_\\mu\\pi\\right)",
            "Part I still records the old completed mass term that motivated the re-audit.",
        ),
        target_record(
            "part3a_exact_vector_hierarchy_line",
            PART3A,
            part3a,
            "mass-origin route で固定した exact vector hierarchy",
            "Part III-A still points to the exact vector hierarchy as the old benchmark family.",
        ),
        target_record(
            "part3a_case_b_line",
            PART3A,
            part3a,
            "Part I の explicit Proca/Stückelberg term が残るため",
            "Part III-A still records the Case-B reason that motivated the breakthrough pivot.",
        ),
        target_record(
            "part5_trial2_hold_line",
            PART5,
            part5,
            "Trial-2 hold",
            "Part V still carries the hold wording that will remain deferred until the rebuild closes.",
        ),
        target_record(
            "status_next_step_anchor",
            STATUS,
            status,
            "current official next step は `8.7.56.169`",
            "STATUS must already point to the working-action vector rebuild branch.",
        ),
        target_record(
            "roadmap_working_action_vector_rebuild_branch",
            ROADMAP,
            roadmap,
            "`8.7.56.169-.172`",
            "ROADMAP must expose the working-action vector rebuild branch.",
        ),
    ]
    source_pack_ready = all(item["present"] for item in source_targets)
    working_action_ready = bool(trial1_vev["summary"]["working_action_uses_mexican_hat_only_mass_source"])
    transverse_split_ready = bool(trial1_vev["summary"]["transverse_mode_massless_under_breakthrough_action"]) and bool(
        trial1_maxwell["summary"]["transverse_maxwell_reduction_available"]
    )
    old_benchmark_pack_retained = bool(vector_reaudit_declaration["summary"]["historic_benchmark_pack_retained"])

    source_inventory = payload(
        "8.7.56.169",
        "Working-action vector mass-spectrum rebuild source inventory",
        common_inputs,
        "Freeze the source pack that combines the breakthrough working action, the transverse-photon split, and the retained old vector benchmark pack before any rebuild pilot is attempted.",
        {
            "working_action_rule": "rebuild uses the mexican-hat-only working action and must not silently inherit the old explicit Proca/Stueckelberg mass term as a physical current-canon source",
            "sector_bookkeeping_rule": "old trial-state and sector tables may be reused as bookkeeping indices even when old exact masses remain benchmark-only",
            "rebuild_source_rule": "the rebuild branch needs the working action, photon split, old benchmark pack, and current hold/defer state visible at the same time",
        },
        [
            row("vector_rebuild_source_inventory_complete", "pass", "working-action vector rebuild source inventory complete", 1, "The rebuild source pack is frozen."),
            row("vector_rebuild_source_pack_ready", "pass" if source_pack_ready else "reject", "working-action vector rebuild source pack ready", 1 if source_pack_ready else 0, "The branch needs the working-action formulas, old benchmark pack, and roadmap/status anchors together."),
            row("vector_rebuild_mass_source_unified", "pass" if working_action_ready else "reject", "working-action mass source unified", 1 if working_action_ready else 0, "The rebuild branch starts only from the mexican-hat-only mass source frozen by the breakthrough pivot."),
            row("vector_rebuild_transverse_photon_split_formula_ready", "pass" if transverse_split_ready else "reject", "transverse photon split formula ready", 1 if transverse_split_ready else 0, "The rebuild source pack reuses the transverse-photon split A_mu = delta P_mu^T / sqrt(Z_P)."),
            row("vector_rebuild_old_benchmark_pack_retained", "pass" if old_benchmark_pack_retained else "reject", "old vector benchmark pack retained", 1 if old_benchmark_pack_retained else 0, "The old exact ladder survives only as benchmark bookkeeping for the rebuild."),
        ],
        {
            "working_action_vector_rebuild_source_pack_ready": source_pack_ready,
            "working_action_uses_mexican_hat_only_mass_source": working_action_ready,
            "transverse_photon_split_formula_ready": transverse_split_ready,
            "old_vector_benchmark_pack_retained": old_benchmark_pack_retained,
            "trial_state_bookkeeping_pack_present": True,
            "first_route_to_close_or_none": "working_action_vector_mass_spectrum_reduced_solver_pilot",
        },
        {
            "overall_status": "working_action_vector_rebuild_source_inventory_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_170": True,
            "next_required_artifacts": ["working_action_vector_mass_spectrum_reduced_solver_pilot"],
        },
        {
            "inventory_targets": source_targets,
            "mexican_hat_summary": mexican_hat["summary"],
            "trial1_vev_summary": trial1_vev["summary"],
            "trial1_maxwell_summary": trial1_maxwell["summary"],
            "vector_reaudit_declaration_summary": vector_reaudit_declaration["summary"],
            "vector_source_inventory_summary": vector_source_inventory["summary"],
            "vector_preservation_audit_summary": vector_preservation_audit["summary"],
            "vector_solver_spec_summary": vector_solver_spec["summary"],
            "vector_trial_states_summary": vector_trial_states["summary"],
            "ai_context_current_step": ai_context["current_step"],
        },
    )

    bookkeeping_projection_ready = (
        source_inventory["summary"]["trial_state_bookkeeping_pack_present"]
        and source_inventory["summary"]["transverse_photon_split_formula_ready"]
    )

    reduced_solver_pilot = payload(
        "8.7.56.170",
        "Working-action vector mass-spectrum reduced solver pilot",
        common_inputs,
        "Attempt the first rebuild pilot by reusing only the old sector bookkeeping and the breakthrough working-action formulas, while refusing to overclaim a numeric ladder before a new radial eigenoperator is frozen.",
        {
            "reduced_pilot_rule": "reuse the old (n,k,ell,s) bookkeeping table as an index set, but do not carry over old exact mass ratios as current-canon outputs without a working-action radial eigenoperator",
            "transverse_sector_rule": "the massless transverse split supplies the kinetic normalization, not a complete discrete mass ladder",
            "numeric_closure_rule": "numeric reduced-solver closure requires an explicit working-action vector radial eigenoperator",
        },
        [
            row("vector_reduced_solver_bookkeeping_projection_ready", "pass" if bookkeeping_projection_ready else "reject", "working-action vector reduced-solver bookkeeping projection ready", 1 if bookkeeping_projection_ready else 0, "The old trial-state table can be reused as bookkeeping under the working action."),
            row("vector_reduced_solver_trial_sector_count", "pass", "working-action vector trial sector count", trial_sector_count, "The reduced pilot inherits the old sector bookkeeping as a starting index set."),
            row("vector_reduced_solver_trial_state_count", "pass", "working-action vector trial state count", trial_state_count, "The reduced pilot inherits the old lower-bound state count as bookkeeping only."),
            row("vector_reduced_solver_radial_eigenoperator_available", "reject", "working-action vector radial eigenoperator available", 0, "No new radial eigenoperator has yet been frozen for the working-action rebuild."),
            row("vector_reduced_solver_mass_proxy_numeric_ready", "reject", "working-action vector mass proxy numeric ready", 0, "Without the new radial eigenoperator, the reduced solver cannot produce numeric current-canon mass proxies."),
        ],
        {
            "bookkeeping_projection_ready": bookkeeping_projection_ready,
            "reduced_trial_sector_count": trial_sector_count,
            "reduced_trial_state_count_lower_bound": trial_state_count,
            "historic_exact_ratio_candidate_count_reference": exact_candidate_count,
            "working_action_vector_radial_eigenoperator_available": False,
            "working_action_vector_mass_proxy_numeric_ready": False,
            "first_route_to_close_or_none": "working_action_vector_mass_spectrum_anchor_refresh",
        },
        {
            "overall_status": "working_action_vector_reduced_solver_pilot_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_171": True,
            "next_required_artifacts": ["working_action_vector_mass_spectrum_anchor_refresh"],
        },
        {
            "trial1_vev_formulas": trial1_vev["formulas"],
            "trial1_maxwell_formulas": trial1_maxwell["formulas"],
            "vector_solver_spec_summary": vector_solver_spec["summary"],
            "vector_trial_states_summary": vector_trial_states["summary"],
            "vector_exact_summary": vector_exact["summary"],
        },
    )

    anchor_refresh = payload(
        "8.7.56.171",
        "Working-action vector mass-spectrum anchor refresh",
        common_inputs,
        "Re-evaluate the anchor pack under the reduced pilot and separate historic benchmark rows from any current-canon refresh claims.",
        {
            "anchor_refresh_rule": "historic anchor rows remain reference targets, but current-canon refresh requires numeric mass proxies from the reduced solver",
            "muon_rule": "the muon benchmark is the lightest anchor to revisit first once a working-action mass proxy exists",
            "heavy_rule": "proton, tau, and neutron/proton pair refreshes depend on the same missing radial eigenoperator",
        },
        [
            row("vector_anchor_refresh_historic_reference_pack_ready", "pass", "historic anchor reference pack ready", 1, "The old benchmark rows remain available as reference targets for the rebuild."),
            row("vector_anchor_refresh_historic_best_exact_muon_relative_error_reference", "pass", "historic best exact muon relative error reference", float(historic_best_exact["relative_error"]), "The old exact muon benchmark is retained only as rebuild reference data."),
            row("vector_anchor_refresh_muon_ready", "reject", "working-action muon anchor refresh ready", 0, "No working-action numeric mass proxy exists yet for refreshing the muon anchor."),
            row("vector_anchor_refresh_heavy_ready", "reject", "working-action heavy anchor refresh ready", 0, "The proton and tau anchors cannot be refreshed before the reduced solver closes numerically."),
            row("vector_anchor_refresh_neutron_proton_pair_ready", "reject", "working-action neutron/proton pair refresh ready", 0, "The baryon doublet pair remains blocked by the same missing working-action mass proxy."),
        ],
        {
            "historic_anchor_reference_pack_ready": True,
            "working_action_muon_anchor_refresh_ready": False,
            "working_action_heavy_anchor_refresh_ready": False,
            "working_action_neutron_proton_pair_refresh_ready": False,
            "anchor_refresh_blocker_or_none": "working_action_vector_mass_proxy_numeric_absent",
            "first_route_to_close_or_none": "working_action_vector_mass_spectrum_declaration_gate",
        },
        {
            "overall_status": "working_action_vector_anchor_refresh_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_172": True,
            "next_required_artifacts": ["working_action_vector_mass_spectrum_declaration_gate"],
        },
        {
            "historic_best_exact_match_row": historic_best_exact,
            "historic_best_proton_row": historic_best_proton,
            "historic_best_tau_row": historic_best_tau,
            "historic_best_neutron_proton_pair": historic_best_np,
            "vector_preservation_audit_summary": vector_preservation_audit["summary"],
            "reduced_solver_pilot_summary": reduced_solver_pilot["summary"],
        },
    )

    declaration_gate = payload(
        "8.7.56.172",
        "Working-action vector mass-spectrum declaration gate / Trial-3 fallback second refresh",
        common_inputs,
        "Integrate the rebuild source inventory, reduced solver pilot, and anchor refresh and decide whether the vector rebuild branch closes or contracts to a deeper residual route.",
        {
            "gate_rule": "the branch closes only if the reduced solver and anchor refresh both become numeric under the working action",
            "fallback_rule": "Trial-3 remains on fallback hold while the vector rebuild lacks a current-canon radial eigenoperator",
            "paper_sync_rule": "Trial-2 paper-side sync remains deferred until the vector rebuild reaches a current-canon anchor status",
        },
        [
            row("vector_rebuild_gate_source_inventory_ready", "pass" if source_inventory["summary"]["working_action_vector_rebuild_source_pack_ready"] else "reject", "working-action vector rebuild source inventory ready", 1 if source_inventory["summary"]["working_action_vector_rebuild_source_pack_ready"] else 0, "The rebuild branch now has its source pack frozen."),
            row("vector_rebuild_gate_reduced_solver_numeric_ready", "reject", "working-action vector reduced solver numeric ready at gate", 0, "The reduced solver still lacks a working-action radial eigenoperator."),
            row("vector_rebuild_gate_anchor_refresh_ready", "reject", "working-action vector anchor refresh ready at gate", 0, "Anchor refresh remains blocked while numeric mass proxies are absent."),
            row("vector_rebuild_gate_trial3_hold_retained", "pass", "Trial-3 fallback hold retained at second gate", 1, "The weak-sector route remains blocked by the unresolved vector rebuild."),
        ],
        {
            "working_action_vector_mass_spectrum_rebuild_source_inventory_ready": source_inventory["summary"]["working_action_vector_rebuild_source_pack_ready"],
            "working_action_vector_mass_spectrum_reduced_solver_numeric_ready": reduced_solver_pilot["summary"]["working_action_vector_mass_proxy_numeric_ready"],
            "working_action_vector_mass_spectrum_anchor_refresh_ready": False,
            "trial2_paper_side_sync_deferred_until_vector_rebuild": True,
            "trial3_fallback_hold_retained": True,
            "trial3_fallback_hold_release_ready": False,
            "recommended_next_route_or_none": "8.7.56.173",
        },
        {
            "overall_status": "working_action_vector_rebuild_branch_nonclosure_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_173": True,
            "next_required_artifacts": ["working_action_vector_radial_eigenoperator_identification"],
        },
        {
            "source_inventory_summary": source_inventory["summary"],
            "reduced_solver_pilot_summary": reduced_solver_pilot["summary"],
            "anchor_refresh_summary": anchor_refresh["summary"],
            "vector_reaudit_route_summary": vector_reaudit_route["summary"],
            "trial2_declaration_summary": trial2_declaration["summary"],
        },
    )

    route_contract = payload(
        "8.7.56.173",
        "Working-action vector radial eigenoperator route contract",
        common_inputs,
        "Freeze the next official residual route after the non-closing rebuild branch: identify the working-action radial eigenoperator that would permit a numeric current-canon vector ladder.",
        {
            "selected_residual_route": "working_action_vector_radial_eigenoperator_identification",
            "blocking_rule": "without a working-action radial eigenoperator, reduced solver masses and anchor refreshes remain benchmark-only",
            "dependency_rule": "Trial-3 and Trial-2 paper-side sync both stay downstream of the same unresolved vector eigenoperator",
        },
        [
            row("vector_radial_eigenoperator_route_contract_complete", "pass", "working-action vector radial eigenoperator route contract complete", 1, "The next residual route is frozen after the rebuild branch non-closure."),
            row("vector_radial_eigenoperator_missing", "pass", "working-action vector radial eigenoperator missing", 1, "The missing artifact is the working-action radial eigenoperator needed for numeric rebuild closure."),
            row("vector_radial_eigenoperator_trial3_dependency_blocked", "pass", "Trial-3 dependency still blocked by vector eigenoperator", 1, "The weak-sector branch remains blocked by the unresolved vector eigenoperator."),
            row("vector_radial_eigenoperator_paper_sync_deferred", "pass", "Trial-2 paper-side sync still deferred by vector eigenoperator", 1, "Paper-side sync remains deferred until the rebuild route settles current-canon vector claims."),
        ],
        {
            "selected_residual_route": "working_action_vector_radial_eigenoperator_identification",
            "missing_v2_artifact": "working_action_vector_mass_spectrum_radial_eigenoperator",
            "trial3_dependency_state": "blocked_by_working_action_vector_radial_eigenoperator",
            "trial2_paper_side_sync_state": "deferred_until_working_action_vector_anchor_refresh",
            "split_contract_ready": True,
            "recommended_next_route_or_none": "8.7.56.174",
        },
        {
            "overall_status": "working_action_vector_radial_eigenoperator_route_contract_frozen",
            "trial2_branch_closeable": False,
            "advance_to_8_7_56_174": True,
            "next_required_artifacts": [
                "working_action_vector_radial_eigenoperator_source_inventory",
                "working_action_vector_radial_eigenoperator_identification_audit",
                "working_action_vector_rebuild_reopen_gate",
            ],
        },
        {
            "declaration_gate_summary": declaration_gate["summary"],
            "trial3_fallback_route_summary": trial3_fallback_route["summary"],
            "vector_reaudit_route_summary": vector_reaudit_route["summary"],
            "trial1_vev_summary": trial1_vev["summary"],
        },
    )

    write_artifact("mass_origin_v2_vector_rebuild_source_inventory", source_inventory)
    write_artifact("mass_origin_v2_vector_reduced_solver_pilot", reduced_solver_pilot)
    write_artifact("mass_origin_v2_vector_anchor_refresh", anchor_refresh)
    write_artifact("mass_origin_v2_vector_rebuild_declaration_gate", declaration_gate)
    write_artifact("mass_origin_v2_vector_radial_eigenoperator_route_contract", route_contract)

    print("[ok] wrote:")
    print(" - mass_origin_v2_vector_rebuild_source_inventory_metrics.json")
    print(" - mass_origin_v2_vector_reduced_solver_pilot_metrics.json")
    print(" - mass_origin_v2_vector_anchor_refresh_metrics.json")
    print(" - mass_origin_v2_vector_rebuild_declaration_gate_metrics.json")
    print(" - mass_origin_v2_vector_radial_eigenoperator_route_contract_metrics.json")


# Function: run the working-action vector rebuild branch from the CLI.

if __name__ == "__main__":
    main()
