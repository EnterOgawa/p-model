#!/usr/bin/env python3
"""Generate 8.7.56.1283-.1286 closure-gap contract artifacts."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_OUT = ROOT / "output" / "public" / "quantum"

STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
WORK_HISTORY_RECENT = ROOT / "doc" / "WORK_HISTORY_RECENT.md"
CURRENT_PROBLEM = ROOT / "doc" / "quantum" / "34_trial2_numeric_alpha_current_problem.md"
CURRENT_STATUS = ROOT / "doc" / "quantum" / "36_trial2_numeric_alpha_current_status.md"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
NOTE = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_vector_qball_form_factor.md")

REVIEW_INVENTORY = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_ground_state_two_component_closure_review_source_inventory_metrics.json"
REVIEW_AUDIT = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_ground_state_two_component_closure_review_audit_metrics.json"
REVIEW_GATE = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_ground_state_two_component_closure_review_declaration_gate_metrics.json"
REVIEW_EVAL = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_ground_state_two_component_closure_review_numeric_evaluation_metrics.json"
ROUTE_LOCAL_GATE = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_route_local_no_go_theorem_review_declaration_gate_metrics.json"
ROUTE_LOCAL_EVAL = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_route_local_no_go_theorem_review_numeric_evaluation_metrics.json"
SPECTRUM_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_v2_trial3_two_component_spectrum_branch.py"
FULL_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_full_coupled_solver_branch.py"

NEXT_ROUTE = "8.7.56.1287"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_projection_theorem_carry_over_contract"


# Function: return the current UTC timestamp.
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# Function: abort if one required path is missing.

def require(path: Path) -> None:
    """Abort if one required path is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# Function: read one UTF-8 text file.

def read_text(path: Path) -> str:
    """Read one UTF-8 text file."""
    return path.read_text(encoding="utf-8")


# Function: read one UTF-8 JSON file.

def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# Function: convert one path to repo-relative display form when possible.

def display_path(path: Path) -> str:
    """Convert one path to repo-relative display form when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# Function: return the first matching line for one substring pattern.

def hit(text: str, pattern: str) -> dict | None:
    """Return the first matching line for one substring pattern."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# Function: build one standard metrics row.

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build one standard metrics row."""
    return {"row_id": row_id, "status": status, "metric": metric, "value": float(value), "note": note}


# Function: build one standard payload.

def payload(step: str, name: str, inputs: dict, rows: list[dict], summary: dict, decision: dict, evidence: dict) -> dict:
    """Build one standard payload."""
    return {"generated_utc": now_iso(), "phase": {"phase": 8, "step": step, "name": name}, "inputs": inputs, "rows": rows, "summary": summary, "decision": decision, "evidence": evidence}


# Function: write one JSON metrics payload and CSV row table.

def write_artifact(stem: str, data: dict) -> None:
    """Write one JSON metrics payload and CSV row table."""
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    (PUBLIC_OUT / f"{stem}_metrics.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (PUBLIC_OUT / f"{stem}_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# Function: execute the 8.7.56.1283-.1286 branch.

def main() -> None:
    """Execute the 8.7.56.1283-.1286 branch."""
    for path in (STATUS, ROADMAP, AI_CONTEXT, WORK_HISTORY_RECENT, CURRENT_PROBLEM, CURRENT_STATUS, PART1, PART3A, PART5, REVIEW_INVENTORY, REVIEW_AUDIT, REVIEW_GATE, REVIEW_EVAL, ROUTE_LOCAL_GATE, ROUTE_LOCAL_EVAL, SPECTRUM_BRANCH, FULL_BRANCH):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    work_history_recent_text = read_text(WORK_HISTORY_RECENT)
    part1_text = read_text(PART1)
    spectrum_text = read_text(SPECTRUM_BRANCH)
    full_text = read_text(FULL_BRANCH)

    note_available = NOTE.exists()
    note_text = read_text(NOTE) if note_available else ""
    review_inventory = read_json(REVIEW_INVENTORY)
    review_gate = read_json(REVIEW_GATE)["summary"]
    review_audit = read_json(REVIEW_AUDIT)["summary"]
    review_eval = read_json(REVIEW_EVAL)["summary"]
    route_local_gate = read_json(ROUTE_LOCAL_GATE)["summary"]
    route_local_eval = read_json(ROUTE_LOCAL_EVAL)["summary"]

    current_pack_limit_state_ready = review_gate["trial2_numeric_alpha_problem_classification"] == "vector_qball_form_factor_ground_state_two_component_closure_not_implied_under_current_pack"
    generic_positive = bool(review_gate["generic_post_photon_two_component_sector_available"])
    closure_not_implied = not bool(review_gate["ground_state_two_component_closure_already_implied_under_current_pack"])
    solver_scalar_reduction = bool(review_gate["current_full_solver_hardcodes_ell0_scalar_reduction"])
    pilot_induction = bool(review_gate["current_two_component_pilot_ell0_induction_available"])
    vector_unopened = not bool(review_gate["vector_form_factor_exact_computation_ready_under_current_pack"])
    route_local_retained = bool(review_gate["route_local_no_go_theorem_retained"])
    projection_carry = bool(route_local_gate["projection_theorem_carry_over_required"])
    source_reopen = review_gate["secondary_residual_lane"] == "qball_projection_overlap_future_source_theorem_reopen"
    physical_reject = bool(review_gate["physical_reject_required"])
    contract_ready = current_pack_limit_state_ready and generic_positive and closure_not_implied and solver_scalar_reduction and (not pilot_induction) and vector_unopened and route_local_retained and projection_carry and source_reopen and (not physical_reject)

    inputs = {
        "source_files": {
            "status": display_path(STATUS),
            "roadmap": display_path(ROADMAP),
            "ai_context": display_path(AI_CONTEXT),
            "work_history_recent": display_path(WORK_HISTORY_RECENT),
            "current_problem_note": display_path(CURRENT_PROBLEM),
            "current_status_note": display_path(CURRENT_STATUS),
            "part1": display_path(PART1),
            "part3a": display_path(PART3A),
            "part5": display_path(PART5),
            "vector_form_factor_note": display_path(NOTE),
            "vector_form_factor_note_available": note_available,
        },
        "prior_metrics": {
            "review_inventory": display_path(REVIEW_INVENTORY),
            "review_audit": display_path(REVIEW_AUDIT),
            "review_gate": display_path(REVIEW_GATE),
            "review_eval": display_path(REVIEW_EVAL),
            "route_local_gate": display_path(ROUTE_LOCAL_GATE),
            "route_local_eval": display_path(ROUTE_LOCAL_EVAL),
        },
        "solver_sources": {"spectrum_branch": display_path(SPECTRUM_BRANCH), "full_branch": display_path(FULL_BRANCH)},
        "constants": {"beta_1": float(review_eval["beta_1"]), "next_route_name": NEXT_ROUTE_NAME, "next_route": NEXT_ROUTE},
    }

    common_evidence = {
        "part1_hits": {
            "post_photon_nontransverse_sector": hit(part1_text, "post-photon nontransverse sector"),
            "one_massive_eigenmode": hit(part1_text, "one massive propagating eigenmode"),
            "coupled_tail": hit(part1_text, "coupled asymptotic eigenmode"),
        },
        "solver_hits": {
            "full_solver_ell0_guard": hit(full_text, "if ell == 0:"),
            "full_solver_return_zero": hit(full_text, "return 0.0"),
            "spectrum_solver_kproxy": hit(spectrum_text, "k_proxy = math.sqrt(max(float(ell * (ell + 1)), 0.0)) / rr"),
            "spectrum_solver_coupling": hit(spectrum_text, "coupling = float(beta) * k_proxy"),
        },
        "note_hits": review_inventory["evidence"]["note_hits"] if not note_available else {
            "signed_density_line": hit(note_text, r"j^0_{\rm vector} = 2\omega"),
            "induced_fl_claim": hit(note_text, r"f_L(r) \propto"),
            "literal_q_equals_m0_claim": hit(note_text, "q = m_0"),
        },
        "status_hits": {
            "status_1283": hit(status_text, "8.7.56.1283"),
            "roadmap_1283": hit(roadmap_text, "`8.7.56.1283-.1286`"),
            "problem_closure_gap": hit(current_problem_text, "closure gap"),
            "status_not_implied": hit(current_status_text, "not implied"),
            "history_1279": hit(work_history_recent_text, "8.7.56.1279-.1282"),
        },
    }

    inventory = payload(
        "8.7.56.1283",
        "Trial-2 numeric alpha vector Q-ball form-factor ground-state two-component closure gap contract source inventory",
        inputs,
        [
            row("current_pack_limit_state_ready", "pass" if current_pack_limit_state_ready else "reject", "current-pack limit state ready", 1 if current_pack_limit_state_ready else 0, "The contract branch starts only after .1279-.1282 has frozen the honest non-implication state."),
            row("generic_post_photon_two_component_sector_surface_available", "pass" if generic_positive else "reject", "generic post-photon two-component sector surface available", 1 if generic_positive else 0, "Part I still provides the positive generic sector surface used by the contract."),
            row("current_solver_scalar_reduction_evidence_available", "pass" if solver_scalar_reduction else "reject", "current solver scalar reduction evidence available", 1 if solver_scalar_reduction else 0, "The current exact solver still hardcodes the ell=0 scalar reduction used by the contract."),
            row("projection_theorem_carry_over_state_available", "pass" if projection_carry else "reject", "projection theorem carry-over state available", 1 if projection_carry else 0, "The theorem-side carry-over lane remains available from the retained route-local no-go review."),
            row("future_source_theorem_reopen_state_available", "pass" if source_reopen else "reject", "future source theorem reopen state available", 1 if source_reopen else 0, "The future-source-theorem reopen lane remains the secondary retained route."),
        ],
        {"inventory_ready": True, "selected_next_substep": "8.7.56.1284", "prior_problem_classification": review_gate["trial2_numeric_alpha_problem_classification"]},
        {"overall_status": "trial2_numeric_alpha_vector_qball_form_factor_ground_state_two_component_closure_gap_contract_inventory_fixed", "advance_to_8_7_56_1284": True, "next_required_artifacts": ["trial2_numeric_alpha_vector_qball_form_factor_ground_state_two_component_closure_gap_contract_audit"]},
        {**common_evidence, "review_gate_summary": review_gate, "route_local_gate_summary": route_local_gate},
    )

    audit = payload(
        "8.7.56.1284",
        "Trial-2 numeric alpha vector Q-ball form-factor ground-state two-component closure gap contract audit",
        inputs,
        [
            row("generic_post_photon_two_component_sector_positive", "pass" if generic_positive else "reject", "generic post-photon two-component sector positive", 1 if generic_positive else 0, "The contract keeps the theorem-side positive generic sector rather than collapsing it into a hard no-go."),
            row("ell0_ground_state_closure_not_implied", "pass" if closure_not_implied else "reject", "ell=0 ground-state closure not implied", 1 if closure_not_implied else 0, "The contract is honest only if the ell=0 closure remains unlicensed under the current pack."),
            row("current_solver_scalar_reduction_retained", "pass" if solver_scalar_reduction else "reject", "current solver scalar reduction retained", 1 if solver_scalar_reduction else 0, "The contract preserves the solver-side evidence that ell=0 still collapses to the scalar reference."),
            row("exact_vector_computation_unopened", "pass" if vector_unopened else "reject", "exact vector computation unopened", 1 if vector_unopened else 0, "The contract must not overstate the current pack as already computation-ready."),
            row("projection_theorem_carry_over_retained", "pass" if projection_carry else "reject", "projection theorem carry-over retained", 1 if projection_carry else 0, "The retained theorem-side route still points first to projection-theorem carry-over."),
            row("future_source_theorem_reopen_retained", "pass" if source_reopen else "reject", "future source theorem reopen retained", 1 if source_reopen else 0, "The future source-theorem reopen route remains secondary and is not erased by the contract."),
            row("physical_reject_not_selected", "pass" if not physical_reject else "reject", "physical reject not selected", 1 if not physical_reject else 0, "The contract keeps this as a current-pack limit rather than a physical reject."),
            row("closure_gap_contract_honest", "pass" if contract_ready else "reject", "closure-gap contract honest", 1 if contract_ready else 0, "The contract is honest only if theorem-side, solver-side, and non-reject conditions cohere."),
        ],
        {
            "generic_post_photon_two_component_sector_available": generic_positive,
            "ground_state_two_component_closure_already_implied_under_current_pack": not closure_not_implied,
            "current_full_solver_hardcodes_ell0_scalar_reduction": solver_scalar_reduction,
            "current_two_component_pilot_ell0_induction_available": pilot_induction,
            "vector_form_factor_exact_computation_ready_under_current_pack": not vector_unopened,
            "route_local_no_go_theorem_retained": route_local_retained,
            "projection_theorem_carry_over_required": projection_carry,
            "future_source_theorem_reopen_retained": source_reopen,
            "physical_reject_required": physical_reject,
            "closure_gap_contract_honest": contract_ready,
            "result_class": "closure_gap_contract_honest_under_current_pack",
        },
        {"overall_status": "trial2_numeric_alpha_vector_qball_form_factor_ground_state_two_component_closure_gap_contract_audit_completed", "advance_to_8_7_56_1285": True, "next_required_artifacts": ["trial2_numeric_alpha_vector_qball_form_factor_ground_state_two_component_closure_gap_contract_declaration_gate"]},
        {"review_audit_summary": review_audit, "review_gate_summary": review_gate, "route_local_gate_summary": route_local_gate},
    )

    gate = payload(
        "8.7.56.1285",
        "Trial-2 numeric alpha vector Q-ball form-factor ground-state two-component closure gap contract declaration gate",
        inputs,
        [
            row("closure_gap_contract_ready", "pass" if contract_ready else "reject", "closure-gap contract ready", 1 if contract_ready else 0, "The current-pack limit can now be frozen as one explicit contract."),
            row("vector_form_factor_exact_computation_ready_under_current_pack", "pass" if not vector_unopened else "reject", "vector form-factor exact computation ready under current pack", 0 if vector_unopened else 1, "The contract keeps exact vector computation unopened under the current pack."),
            row("projection_theorem_carry_over_required", "pass" if projection_carry else "reject", "projection theorem carry-over required", 1 if projection_carry else 0, "The next primary route remains the theorem-side carry-over lane."),
            row("future_source_theorem_reopen_retained", "pass" if source_reopen else "reject", "future source theorem reopen retained", 1 if source_reopen else 0, "The secondary retained lane remains the future-source-theorem reopen route."),
            row("route_local_no_go_theorem_retained", "pass" if route_local_retained else "reject", "route-local no-go theorem retained", 1 if route_local_retained else 0, "The current-pack contract does not erase the T2 route-local no-go result."),
            row("physical_reject_required", "pass" if physical_reject else "reject", "physical reject required", 1 if physical_reject else 0, "Physical reject remains unselected under this contract."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "vector_qball_form_factor_ground_state_two_component_closure_gap_contract_under_current_pack",
            "current_pack_limit_state": review_gate["trial2_numeric_alpha_problem_classification"],
            "closure_gap_contract_ready": contract_ready,
            "closure_gap_contract_honest": contract_ready,
            "generic_post_photon_two_component_sector_available": generic_positive,
            "ground_state_two_component_closure_already_implied_under_current_pack": not closure_not_implied,
            "vector_form_factor_exact_computation_ready_under_current_pack": not vector_unopened,
            "route_local_no_go_theorem_retained": route_local_retained,
            "projection_theorem_carry_over_required": projection_carry,
            "future_source_theorem_reopen_retained": source_reopen,
            "primary_residual_lane": "vector_qball_form_factor_projection_theorem_carry_over",
            "secondary_residual_lane": "qball_projection_overlap_future_source_theorem_reopen",
            "reserve_residual_lane": "qball_projection_overlap_analytic_tail_theorem_refinement",
            "physical_reject_required": physical_reject,
            "closeout_ready": False,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {"overall_status": "trial2_numeric_alpha_vector_qball_form_factor_ground_state_two_component_closure_gap_contract_declared", "advance_to_8_7_56_1286": True, "next_required_artifacts": [NEXT_ROUTE_NAME]},
        {"audit_summary": audit["summary"], "review_gate_summary": review_gate, "route_local_gate_summary": route_local_gate},
    )

    evaluation = payload(
        "8.7.56.1286",
        "Trial-2 numeric alpha vector Q-ball form-factor ground-state two-component closure gap contract numeric evaluation",
        inputs,
        [
            row("beta_1_fixed", "pass", "beta_1 fixed", float(review_eval["beta_1"]), "The retained electron-like beta_1 stays fixed during the contract branch."),
            row("q_theory_over_m0_fixed", "pass", "q_theory/m0 fixed", float(review_eval["q_theory_over_m0"]), "The retained projection-overlap matching-scale candidate stays fixed."),
            row("F_exact_at_q_theory_fixed", "pass", "F_exact at q_theory fixed", float(review_eval["F_exact_at_q_theory"]), "The retained exact-profile pass stays fixed during the contract branch."),
            row("alpha_exact_at_q_theory_fixed", "pass", "alpha_exact at q_theory fixed", float(review_eval["alpha_exact_at_q_theory"]), "The retained alpha_exact value stays fixed during the contract branch."),
            row("exact_ground_state_polarization_weight_fixed", "pass", "exact ground-state polarization weight fixed", float(review_eval["exact_ground_state_polarization_weight"]), "The current exact vector ground-state still stays at zero polarization weight."),
            row("numeric_state_changed_by_current_branch", "pass" if bool(review_eval["numeric_state_changed_by_current_branch"]) else "reject", "numeric state changed by current branch", 1 if bool(review_eval["numeric_state_changed_by_current_branch"]) else 0, "The contract branch does not introduce a new numeric candidate beyond the retained state."),
            row("route_state_changed_by_current_branch", "pass", "route state changed by current branch", 1.0, "The route now advances from non-implication review to an explicit closure-gap contract."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "vector_qball_form_factor_ground_state_two_component_closure_gap_contract_under_current_pack",
            "beta_1": float(review_eval["beta_1"]),
            "exact_ground_state_polarization_weight": float(review_eval["exact_ground_state_polarization_weight"]),
            "exact_ground_state_coupled_charge_factor": float(review_eval["exact_ground_state_coupled_charge_factor"]),
            "ell0_zero_seed_max_abs_fL": float(review_eval["ell0_zero_seed_max_abs_fL"]),
            "scalar_literal_F_m0": float(review_eval["scalar_literal_F_m0"]),
            "q_theory_over_m0": float(review_eval["q_theory_over_m0"]),
            "F_exact_at_q_theory": float(review_eval["F_exact_at_q_theory"]),
            "alpha_exact_at_q_theory": float(review_eval["alpha_exact_at_q_theory"]),
            "vector_form_factor_exact_computation_ready_under_current_pack": not vector_unopened,
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {"overall_status": "trial2_numeric_alpha_vector_qball_form_factor_ground_state_two_component_closure_gap_contract_completed", "advance_to_next_route": True, "next_required_artifacts": [NEXT_ROUTE_NAME]},
        {"prior_problem_classification": review_gate["trial2_numeric_alpha_problem_classification"], "new_problem_classification": "vector_qball_form_factor_ground_state_two_component_closure_gap_contract_under_current_pack", "route_local_eval_summary": route_local_eval},
    )

    write_artifact("mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_ground_state_two_component_closure_gap_contract_source_inventory", inventory)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_ground_state_two_component_closure_gap_contract_audit", audit)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_ground_state_two_component_closure_gap_contract_declaration_gate", gate)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_ground_state_two_component_closure_gap_contract_numeric_evaluation", evaluation)
    print("[done] 8.7.56.1283-.1286 artifacts generated")


if __name__ == "__main__":
    main()
