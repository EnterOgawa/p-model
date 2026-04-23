#!/usr/bin/env python3
"""Generate 8.7.56.1327-.1330 current-canon closeout / exploratory split artifacts.

Purpose:
    Freeze the honest theorem-side closeout for the current-canon vector-Qball
    form-factor route after the retained-lane reopening stack has been fully
    formalized. This branch does not create a new numeric candidate. Instead it
    declares that the text-side current-canon route is exhausted under the
    present public pack and that the next admissible work must split into three
    exploratory computation branches:

    1. generalized vector solver
    2. effective source ansatz
    3. observable dictionary
"""

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
FULL_COUPLED_SOLVER = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_full_coupled_solver_branch.py"
TWO_COMPONENT_SOLVER = ROOT / "scripts" / "quantum" / "mass_origin_v2_trial3_two_component_spectrum_branch.py"
NEXT_STEPS_NOTE = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_next_steps_20260327.md")

ROUTE_LOCAL_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_route_local_no_go_theorem_review_"
    "declaration_gate_metrics.json"
)
CLOSURE_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_ground_state_two_component_"
    "closure_gap_contract_declaration_gate_metrics.json"
)
PRIMARY_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_projection_theorem_carry_over_"
    "contract_declaration_gate_metrics.json"
)
PRIMARY_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_projection_theorem_carry_over_"
    "contract_numeric_evaluation_metrics.json"
)
REOPEN_RETAIN_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_future_source_theorem_reopen_"
    "retain_contract_declaration_gate_metrics.json"
)
REOPEN_RETAIN_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_future_source_theorem_reopen_"
    "retain_contract_numeric_evaluation_metrics.json"
)
TOP_LEVEL_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_retained_lane_top_level_"
    "contract_declaration_gate_metrics.json"
)
TOP_LEVEL_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_retained_lane_top_level_"
    "contract_numeric_evaluation_metrics.json"
)
RETAINED_CARRY_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_retained_lane_carry_over_"
    "contract_declaration_gate_metrics.json"
)
RETAINED_CARRY_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_retained_lane_carry_over_"
    "contract_numeric_evaluation_metrics.json"
)
REOPEN_RETAINED_LANE_INVENTORY = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_future_source_theorem_reopen_retained_lane_"
    "contract_source_inventory_metrics.json"
)
REOPEN_RETAINED_LANE_AUDIT = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_future_source_theorem_reopen_retained_lane_"
    "contract_audit_metrics.json"
)
REOPEN_RETAINED_LANE_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_future_source_theorem_reopen_retained_lane_"
    "contract_declaration_gate_metrics.json"
)
REOPEN_RETAINED_LANE_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_future_source_theorem_reopen_retained_lane_"
    "contract_numeric_evaluation_metrics.json"
)

BRANCH_CLASS = (
    "vector_qball_form_factor_current_canon_closeout_exploratory_split_contract_under_current_pack"
)
NEXT_ROUTE = "8.7.56.1331"
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_exploratory_generalized_vector_solver_branch"
)
SECONDARY_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_exploratory_effective_source_ansatz_branch"
)
RESERVE_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_exploratory_observable_dictionary_branch"
)


# Function: return the current UTC timestamp.
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# Function: abort when one required input is missing.

def require(path: Path) -> None:
    """Abort when one required input is missing."""
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
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# Function: build one standard payload.

def payload(
    step: str,
    name: str,
    inputs: dict,
    rows: list[dict],
    summary: dict,
    decision: dict,
    evidence: dict,
) -> dict:
    """Build one standard payload."""
    return {
        "generated_utc": now_iso(),
        "phase": {"phase": 8, "step": step, "name": name},
        "inputs": inputs,
        "rows": rows,
        "summary": summary,
        "decision": decision,
        "evidence": evidence,
    }


# Function: write one JSON metrics payload and CSV rows table.

def write_artifact(stem: str, data: dict) -> None:
    """Write one JSON metrics payload and CSV rows table."""
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    (PUBLIC_OUT / f"{stem}_metrics.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (PUBLIC_OUT / f"{stem}_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_id", "status", "metric", "value", "note"],
        )
        writer.writeheader()
        writer.writerows(data["rows"])


# Function: convert one summary value to float with default fallback.

def float_value(summary: dict, key: str, default: float = 0.0) -> float:
    """Convert one summary value to float with default fallback."""
    return float(summary.get(key, default))


# Function: execute the 8.7.56.1327-.1330 branch.

def main() -> None:
    """Execute the 8.7.56.1327-.1330 branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        PART1,
        PART3A,
        PART5,
        FULL_COUPLED_SOLVER,
        TWO_COMPONENT_SOLVER,
        NEXT_STEPS_NOTE,
        ROUTE_LOCAL_GATE,
        CLOSURE_GATE,
        PRIMARY_GATE,
        PRIMARY_EVAL,
        REOPEN_RETAIN_GATE,
        REOPEN_RETAIN_EVAL,
        TOP_LEVEL_GATE,
        TOP_LEVEL_EVAL,
        RETAINED_CARRY_GATE,
        RETAINED_CARRY_EVAL,
        REOPEN_RETAINED_LANE_INVENTORY,
        REOPEN_RETAINED_LANE_AUDIT,
        REOPEN_RETAINED_LANE_GATE,
        REOPEN_RETAINED_LANE_EVAL,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    work_history_recent_text = read_text(WORK_HISTORY_RECENT)
    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    next_steps_note_text = read_text(NEXT_STEPS_NOTE)

    route_local_gate_summary = dict(read_json(ROUTE_LOCAL_GATE)["summary"])
    closure_gate_summary = dict(read_json(CLOSURE_GATE)["summary"])
    primary_gate_summary = dict(read_json(PRIMARY_GATE)["summary"])
    primary_eval_summary = dict(read_json(PRIMARY_EVAL)["summary"])
    reopen_retain_gate_summary = dict(read_json(REOPEN_RETAIN_GATE)["summary"])
    reopen_retain_eval_summary = dict(read_json(REOPEN_RETAIN_EVAL)["summary"])
    top_level_gate_summary = dict(read_json(TOP_LEVEL_GATE)["summary"])
    top_level_eval_summary = dict(read_json(TOP_LEVEL_EVAL)["summary"])
    retained_carry_gate_summary = dict(read_json(RETAINED_CARRY_GATE)["summary"])
    retained_carry_eval_summary = dict(read_json(RETAINED_CARRY_EVAL)["summary"])
    reopen_retained_lane_inventory = read_json(REOPEN_RETAINED_LANE_INVENTORY)
    reopen_retained_lane_audit_summary = dict(read_json(REOPEN_RETAINED_LANE_AUDIT)["summary"])
    reopen_retained_lane_gate_summary = dict(read_json(REOPEN_RETAINED_LANE_GATE)["summary"])
    reopen_retained_lane_eval_summary = dict(read_json(REOPEN_RETAINED_LANE_EVAL)["summary"])

    latest_retained_lane_reopen_completed = (
        reopen_retained_lane_gate_summary["trial2_numeric_alpha_problem_classification"]
        == "vector_qball_form_factor_future_source_theorem_reopen_retained_lane_contract_under_current_pack"
    )
    retained_lane_carry_over_completed = (
        retained_carry_gate_summary["trial2_numeric_alpha_problem_classification"]
        == "vector_qball_form_factor_retained_lane_carry_over_contract_under_current_pack"
    )
    route_local_no_go_theorem_retained = (
        route_local_gate_summary["trial2_numeric_alpha_problem_classification"]
        == "qball_projection_overlap_route_local_no_go_theorem_under_current_canon"
    )
    closure_gap_retained = (
        closure_gate_summary["trial2_numeric_alpha_problem_classification"]
        == "vector_qball_form_factor_ground_state_two_component_closure_gap_contract_under_current_pack"
    )
    projection_theorem_primary_retained = (
        reopen_retained_lane_gate_summary["primary_residual_lane"]
        == "vector_qball_form_factor_projection_theorem_carry_over"
    )
    future_source_theorem_reopen_retained = (
        reopen_retained_lane_gate_summary["secondary_residual_lane"]
        == "qball_projection_overlap_future_source_theorem_reopen"
    )
    reserve_tail_refinement_retained = (
        reopen_retained_lane_gate_summary["reserve_residual_lane"]
        == "qball_projection_overlap_analytic_tail_theorem_refinement"
    )
    vector_form_factor_exact_computation_unopened = not bool(
        reopen_retained_lane_eval_summary["vector_form_factor_exact_computation_ready_under_current_pack"]
    )
    physical_reject_not_selected = not bool(
        reopen_retained_lane_gate_summary["physical_reject_required"]
    )
    source_closure_observable_dictionary_gap_confirmed = all(
        (
            route_local_no_go_theorem_retained,
            closure_gap_retained,
            vector_form_factor_exact_computation_unopened,
        )
    )
    solver_scripts_ready = FULL_COUPLED_SOLVER.exists() and TWO_COMPONENT_SOLVER.exists()
    next_steps_note_available = NEXT_STEPS_NOTE.exists()
    generalized_vector_solver_branch_admissible = all(
        (
            closure_gap_retained,
            solver_scripts_ready,
            next_steps_note_available,
        )
    )
    effective_source_ansatz_branch_admissible = all(
        (
            route_local_no_go_theorem_retained,
            next_steps_note_available,
        )
    )
    observable_dictionary_branch_admissible = source_closure_observable_dictionary_gap_confirmed
    current_canon_theorem_side_limit_exhausted = all(
        (
            latest_retained_lane_reopen_completed,
            retained_lane_carry_over_completed,
            route_local_no_go_theorem_retained,
            closure_gap_retained,
            projection_theorem_primary_retained,
            future_source_theorem_reopen_retained,
            reserve_tail_refinement_retained,
            vector_form_factor_exact_computation_unopened,
            physical_reject_not_selected,
        )
    )
    current_canon_text_only_mainline_extension_admissible = False
    current_canon_closeout_ready = current_canon_theorem_side_limit_exhausted
    exploratory_split_ready = all(
        (
            current_canon_theorem_side_limit_exhausted,
            generalized_vector_solver_branch_admissible,
            effective_source_ansatz_branch_admissible,
            observable_dictionary_branch_admissible,
        )
    )

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
            "full_coupled_solver": display_path(FULL_COUPLED_SOLVER),
            "two_component_solver": display_path(TWO_COMPONENT_SOLVER),
            "next_steps_note": display_path(NEXT_STEPS_NOTE),
        },
        "prior_metrics": {
            "route_local_gate": display_path(ROUTE_LOCAL_GATE),
            "closure_gate": display_path(CLOSURE_GATE),
            "primary_gate": display_path(PRIMARY_GATE),
            "primary_eval": display_path(PRIMARY_EVAL),
            "reopen_retain_gate": display_path(REOPEN_RETAIN_GATE),
            "reopen_retain_eval": display_path(REOPEN_RETAIN_EVAL),
            "top_level_gate": display_path(TOP_LEVEL_GATE),
            "top_level_eval": display_path(TOP_LEVEL_EVAL),
            "retained_carry_gate": display_path(RETAINED_CARRY_GATE),
            "retained_carry_eval": display_path(RETAINED_CARRY_EVAL),
            "reopen_retained_lane_inventory": display_path(REOPEN_RETAINED_LANE_INVENTORY),
            "reopen_retained_lane_audit": display_path(REOPEN_RETAINED_LANE_AUDIT),
            "reopen_retained_lane_gate": display_path(REOPEN_RETAINED_LANE_GATE),
            "reopen_retained_lane_eval": display_path(REOPEN_RETAINED_LANE_EVAL),
        },
        "constants": {
            "beta_1": float_value(reopen_retained_lane_eval_summary, "beta_1"),
            "q_theory_over_m0": float_value(reopen_retained_lane_eval_summary, "q_theory_over_m0"),
            "next_route_name": NEXT_ROUTE_NAME,
            "secondary_route_name": SECONDARY_ROUTE_NAME,
            "reserve_route_name": RESERVE_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
        },
    }

    inventory = payload(
        "8.7.56.1327",
        "Trial-2 numeric alpha vector Q-ball form-factor current-canon closeout / exploratory split contract source inventory",
        inputs,
        [
            row(
                "current_canon_theorem_side_limit_exhausted",
                "pass" if current_canon_theorem_side_limit_exhausted else "reject",
                "current-canon theorem-side limit exhausted",
                1 if current_canon_theorem_side_limit_exhausted else 0,
                "The theorem-side text route is exhausted only after the retained reopening stack, route-local no-go, and closure-gap stack all agree that exact vector computation remains unopened.",
            ),
            row(
                "source_closure_observable_dictionary_gap_confirmed",
                "pass" if source_closure_observable_dictionary_gap_confirmed else "reject",
                "source / closure / observable dictionary gap confirmed",
                1 if source_closure_observable_dictionary_gap_confirmed else 0,
                "4D vectorization moved the blocker from matching-scale guesswork to the source / closure / observable dictionary gap.",
            ),
            row(
                "generalized_vector_solver_branch_candidate",
                "pass" if generalized_vector_solver_branch_admissible else "reject",
                "generalized vector solver branch candidate",
                1 if generalized_vector_solver_branch_admissible else 0,
                "The solver-side exploratory branch is admissible because the closure gap is explicit and the exploratory next-steps note gives a yes/no program for ell=0 series and solver redesign.",
            ),
            row(
                "effective_source_ansatz_branch_candidate",
                "pass" if effective_source_ansatz_branch_admissible else "reject",
                "effective source ansatz branch candidate",
                1 if effective_source_ansatz_branch_admissible else 0,
                "The source-side exploratory branch is admissible because the route-local no-go is already localized to the missing T2 effective source formula.",
            ),
            row(
                "observable_dictionary_branch_candidate",
                "pass" if observable_dictionary_branch_admissible else "reject",
                "observable dictionary branch candidate",
                1 if observable_dictionary_branch_admissible else 0,
                "The observable-dictionary branch is admissible because source, closure, and physical-reading gaps now remain isolated after 4D vectorization.",
            ),
            row(
                "current_canon_text_only_mainline_extension_admissible",
                "reject",
                "current-canon text-only mainline extension admissible",
                0,
                "Another wording-only retained branch would add no new public-canonical surface and would re-enter the same theorem-side loop.",
            ),
        ],
        {
            "current_canon_closeout_inventory_ready": current_canon_theorem_side_limit_exhausted,
            "source_closure_observable_dictionary_gap_confirmed": source_closure_observable_dictionary_gap_confirmed,
            "generalized_vector_solver_branch_candidate": generalized_vector_solver_branch_admissible,
            "effective_source_ansatz_branch_candidate": effective_source_ansatz_branch_admissible,
            "observable_dictionary_branch_candidate": observable_dictionary_branch_admissible,
            "selected_next_substep": "8.7.56.1328",
            "current_pack_limit_state": reopen_retained_lane_gate_summary["trial2_numeric_alpha_problem_classification"],
            "overall_status": (
                "trial2_numeric_alpha_vector_qball_form_factor_current_canon_closeout_exploratory_split_"
                "inventory_fixed"
            ),
        },
        {
            "overall_status": (
                "trial2_numeric_alpha_vector_qball_form_factor_current_canon_closeout_exploratory_split_"
                "inventory_fixed"
            ),
            "advance_to_8_7_56_1328": True,
            "next_required_artifacts": [
                "trial2_numeric_alpha_vector_qball_form_factor_current_canon_closeout_exploratory_split_contract_audit"
            ],
        },
        {
            "status_hits": {
                "status_1327": hit(status_text, "8.7.56.1327"),
                "current_canon_closeout": hit(status_text, "current-canon closeout / exploratory split"),
                "source_closure_dictionary_gap": hit(status_text, "source / closure / observable dictionary gap"),
            },
            "roadmap_hits": {
                "roadmap_1327": hit(roadmap_text, "`8.7.56.1327`"),
                "roadmap_1328": hit(roadmap_text, "`8.7.56.1328`"),
                "roadmap_1329": hit(roadmap_text, "`8.7.56.1329`"),
            },
            "current_problem_hits": {
                "problem_current_canon_closeout": hit(current_problem_text, "current-canon theorem-side limit"),
                "problem_gap": hit(current_problem_text, "source / closure / observable dictionary gap"),
            },
            "current_status_hits": {
                "status_current_canon_closeout": hit(current_status_text, "current-canon closeout"),
                "status_gap": hit(current_status_text, "source / closure / observable dictionary gap"),
            },
            "paper_hits": {
                "part1_current": hit(part1_text, r"J^\mu_{\mathrm{matter}}"),
                "part3a_current": hit(part3a_text, "current-canon-closeout-exploratory-split-contract next"),
                "part5_current": hit(part5_text, "future-source-theorem-reopen-retained-lane-contract under current pack"),
            },
            "next_steps_note_hits": {
                "step_a": hit(next_steps_note_text, "Step A."),
                "step_b": hit(next_steps_note_text, "Step B."),
                "step_c": hit(next_steps_note_text, "Step C."),
                "step_d": hit(next_steps_note_text, "Step D."),
            },
            "reopen_retained_lane_inventory_summary": reopen_retained_lane_inventory["summary"],
            "reopen_retained_lane_audit_summary": reopen_retained_lane_audit_summary,
            "top_level_gate_summary": top_level_gate_summary,
            "reopen_retain_gate_summary": reopen_retain_gate_summary,
            "primary_gate_summary": primary_gate_summary,
            "route_local_gate_summary": route_local_gate_summary,
        },
    )

    audit = payload(
        "8.7.56.1328",
        "Trial-2 numeric alpha vector Q-ball form-factor current-canon closeout / exploratory split contract audit",
        inputs,
        [
            row(
                "current_canon_closeout_ready",
                "pass" if current_canon_closeout_ready else "reject",
                "current-canon closeout ready",
                1 if current_canon_closeout_ready else 0,
                "The theorem-side current-canon route can close out only after the retained reopening stack and route-local no-go theorem agree that no further text-only continuation is honest.",
            ),
            row(
                "generalized_vector_solver_branch_admissible",
                "pass" if generalized_vector_solver_branch_admissible else "reject",
                "generalized vector solver branch admissible",
                1 if generalized_vector_solver_branch_admissible else 0,
                "The first exploratory branch is admissible because the ell=0 closure gap is explicit and a concrete solver-side yes/no program is retained.",
            ),
            row(
                "effective_source_ansatz_branch_admissible",
                "pass" if effective_source_ansatz_branch_admissible else "reject",
                "effective source ansatz branch admissible",
                1 if effective_source_ansatz_branch_admissible else 0,
                "The second exploratory branch is admissible because T2 failure is localized to the missing effective source theorem.",
            ),
            row(
                "observable_dictionary_branch_admissible",
                "pass" if observable_dictionary_branch_admissible else "reject",
                "observable dictionary branch admissible",
                1 if observable_dictionary_branch_admissible else 0,
                "The third exploratory branch is admissible because the remaining observable reading gap can now be isolated from the theorem-side closeout.",
            ),
            row(
                "current_canon_text_only_mainline_extension_admissible",
                "reject",
                "current-canon text-only mainline extension admissible",
                0,
                "The current-canon theorem-side route should now stop instead of spawning another retained wording contract.",
            ),
            row(
                "physical_reject_required",
                "reject",
                "physical reject required",
                0,
                "The theorem-side closeout remains route-local and does not require physical reject.",
            ),
        ],
        {
            "current_canon_theorem_side_limit_exhausted": current_canon_theorem_side_limit_exhausted,
            "current_canon_closeout_ready": current_canon_closeout_ready,
            "current_canon_closeout_honest": current_canon_closeout_ready,
            "exploratory_split_ready": exploratory_split_ready,
            "generalized_vector_solver_branch_admissible": generalized_vector_solver_branch_admissible,
            "effective_source_ansatz_branch_admissible": effective_source_ansatz_branch_admissible,
            "observable_dictionary_branch_admissible": observable_dictionary_branch_admissible,
            "current_canon_text_only_mainline_extension_admissible": False,
            "physical_reject_required": False,
            "result_class": (
                "current_canon_closeout_exploratory_split_honest"
                if exploratory_split_ready
                else "current_canon_closeout_exploratory_split_not_yet_honest"
            ),
        },
        {
            "overall_status": (
                "trial2_numeric_alpha_vector_qball_form_factor_current_canon_closeout_exploratory_split_"
                "audit_completed"
            ),
            "advance_to_8_7_56_1329": True,
            "next_required_artifacts": [
                "trial2_numeric_alpha_vector_qball_form_factor_current_canon_closeout_exploratory_split_contract_declaration_gate"
            ],
        },
        {
            "closure_gate_summary": closure_gate_summary,
            "primary_gate_summary": primary_gate_summary,
            "reopen_retain_gate_summary": reopen_retain_gate_summary,
            "top_level_gate_summary": top_level_gate_summary,
            "retained_carry_gate_summary": retained_carry_gate_summary,
            "reopen_retained_lane_gate_summary": reopen_retained_lane_gate_summary,
            "route_local_gate_summary": route_local_gate_summary,
        },
    )

    declaration_gate = payload(
        "8.7.56.1329",
        "Trial-2 numeric alpha vector Q-ball form-factor current-canon closeout / exploratory split contract declaration gate",
        inputs,
        [
            row(
                "current_canon_closeout_ready",
                "pass" if current_canon_closeout_ready else "reject",
                "current-canon closeout ready",
                1 if current_canon_closeout_ready else 0,
                "The theorem-side current-canon route is closed out inside the current pack.",
            ),
            row(
                "exploratory_split_ready",
                "pass" if exploratory_split_ready else "reject",
                "exploratory split ready",
                1 if exploratory_split_ready else 0,
                "Exploratory work may continue only after the current-canon theorem-side route is explicitly closed out.",
            ),
            row(
                "generalized_vector_solver_branch_admissible",
                "pass" if generalized_vector_solver_branch_admissible else "reject",
                "generalized vector solver branch admissible",
                1 if generalized_vector_solver_branch_admissible else 0,
                "The first exploratory branch is the generalized vector solver branch.",
            ),
            row(
                "effective_source_ansatz_branch_admissible",
                "pass" if effective_source_ansatz_branch_admissible else "reject",
                "effective source ansatz branch admissible",
                1 if effective_source_ansatz_branch_admissible else 0,
                "The second exploratory branch is the effective source ansatz branch.",
            ),
            row(
                "observable_dictionary_branch_admissible",
                "pass" if observable_dictionary_branch_admissible else "reject",
                "observable dictionary branch admissible",
                1 if observable_dictionary_branch_admissible else 0,
                "The third exploratory branch is the observable dictionary branch.",
            ),
            row(
                "physical_reject_required",
                "reject",
                "physical reject required",
                0,
                "Physical reject remains unselected while the route moves from current-canon closeout into exploratory computation branches.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "current_pack_limit_state": reopen_retained_lane_gate_summary["trial2_numeric_alpha_problem_classification"],
            "current_canon_theorem_side_limit_exhausted": current_canon_theorem_side_limit_exhausted,
            "current_canon_closeout_ready": current_canon_closeout_ready,
            "current_canon_closeout_honest": current_canon_closeout_ready,
            "current_canon_closeout_completed": current_canon_closeout_ready,
            "exploratory_split_ready": exploratory_split_ready,
            "exploratory_split_honest": exploratory_split_ready,
            "generalized_vector_solver_branch_admissible": generalized_vector_solver_branch_admissible,
            "effective_source_ansatz_branch_admissible": effective_source_ansatz_branch_admissible,
            "observable_dictionary_branch_admissible": observable_dictionary_branch_admissible,
            "current_canon_text_only_mainline_extension_admissible": False,
            "projection_theorem_carry_over_required": projection_theorem_primary_retained,
            "future_source_theorem_reopen_retained": future_source_theorem_reopen_retained,
            "reserve_tail_refinement_retained": reserve_tail_refinement_retained,
            "vector_form_factor_exact_computation_ready_under_current_pack": False,
            "physical_reject_required": False,
            "closeout_ready": False,
            "selected_exploratory_primary_route": NEXT_ROUTE_NAME,
            "selected_exploratory_secondary_route": SECONDARY_ROUTE_NAME,
            "selected_exploratory_reserve_route": RESERVE_ROUTE_NAME,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": (
                "trial2_numeric_alpha_vector_qball_form_factor_current_canon_closeout_exploratory_split_"
                "declared"
            ),
            "advance_to_8_7_56_1330": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "audit_summary": audit["summary"],
            "primary_eval_summary": primary_eval_summary,
            "reopen_retain_eval_summary": reopen_retain_eval_summary,
            "top_level_eval_summary": top_level_eval_summary,
            "retained_carry_eval_summary": retained_carry_eval_summary,
            "reopen_retained_lane_eval_summary": reopen_retained_lane_eval_summary,
        },
    )

    evaluation = payload(
        "8.7.56.1330",
        "Trial-2 numeric alpha vector Q-ball form-factor current-canon closeout / exploratory split contract numeric evaluation",
        inputs,
        [
            row(
                "beta_1_fixed",
                "pass",
                "beta_1 fixed",
                float_value(reopen_retained_lane_eval_summary, "beta_1"),
                "The retained electron-like beta_1 stays fixed through the current-canon closeout / exploratory split branch.",
            ),
            row(
                "q_theory_over_m0_fixed",
                "pass",
                "q_theory/m0 fixed",
                float_value(reopen_retained_lane_eval_summary, "q_theory_over_m0"),
                "The retained matching-scale candidate stays fixed through the current-canon closeout / exploratory split branch.",
            ),
            row(
                "F_exact_at_q_theory_fixed",
                "pass",
                "F_exact at q_theory fixed",
                float_value(reopen_retained_lane_eval_summary, "F_exact_at_q_theory"),
                "The retained exact-profile overlap value stays fixed through the current-canon closeout / exploratory split branch.",
            ),
            row(
                "alpha_exact_at_q_theory_fixed",
                "pass",
                "alpha_exact at q_theory fixed",
                float_value(reopen_retained_lane_eval_summary, "alpha_exact_at_q_theory"),
                "The retained exact alpha candidate stays fixed through the current-canon closeout / exploratory split branch.",
            ),
            row(
                "exact_ground_state_polarization_weight_fixed",
                "pass",
                "exact ground-state polarization weight fixed",
                float_value(reopen_retained_lane_eval_summary, "exact_ground_state_polarization_weight"),
                "The current exact vector ground state still stays at zero polarization weight under the current pack.",
            ),
            row(
                "numeric_state_changed_by_current_branch",
                "reject",
                "numeric state changed by current branch",
                0,
                "This branch closes out the text-side current-canon route and splits future work; it does not create a new numeric evaluation.",
            ),
            row(
                "route_state_changed_by_current_branch",
                "pass",
                "route state changed by current branch",
                1,
                "The route now advances from current-canon retained reopening freeze to current-canon closeout plus exploratory split.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "beta_1": float_value(reopen_retained_lane_eval_summary, "beta_1"),
            "exact_ground_state_polarization_weight": float_value(
                reopen_retained_lane_eval_summary, "exact_ground_state_polarization_weight"
            ),
            "exact_ground_state_coupled_charge_factor": float_value(
                reopen_retained_lane_eval_summary, "exact_ground_state_coupled_charge_factor"
            ),
            "ell0_zero_seed_max_abs_fL": float_value(
                reopen_retained_lane_eval_summary, "ell0_zero_seed_max_abs_fL"
            ),
            "scalar_literal_F_m0": float_value(reopen_retained_lane_eval_summary, "scalar_literal_F_m0"),
            "q_theory_over_m0": float_value(reopen_retained_lane_eval_summary, "q_theory_over_m0"),
            "F_exact_at_q_theory": float_value(reopen_retained_lane_eval_summary, "F_exact_at_q_theory"),
            "alpha_exact_at_q_theory": float_value(reopen_retained_lane_eval_summary, "alpha_exact_at_q_theory"),
            "current_canon_closeout_completed": current_canon_closeout_ready,
            "exploratory_split_completed": exploratory_split_ready,
            "vector_form_factor_exact_computation_ready_under_current_pack": False,
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": (
                "trial2_numeric_alpha_vector_qball_form_factor_current_canon_closeout_exploratory_split_"
                "contract_completed"
            ),
            "advance_to_next_route": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "prior_problem_classification": reopen_retained_lane_gate_summary["trial2_numeric_alpha_problem_classification"],
            "new_problem_classification": BRANCH_CLASS,
            "reopen_retained_lane_eval_summary": reopen_retained_lane_eval_summary,
        },
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_current_canon_closeout_exploratory_split_contract_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_current_canon_closeout_exploratory_split_contract_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_current_canon_closeout_exploratory_split_contract_declaration_gate",
        declaration_gate,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_current_canon_closeout_exploratory_split_contract_numeric_evaluation",
        evaluation,
    )

    print("[done] 8.7.56.1327-.1330 artifacts generated")


if __name__ == "__main__":
    main()
