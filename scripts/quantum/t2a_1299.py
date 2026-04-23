#!/usr/bin/env python3
"""Generate 8.7.56.1299-.1302 retained-lane hold contract artifacts.

Purpose:
    Freeze the full primary/secondary/reserve retained ordering as the
    current-pack top-level hold contract after the primary projection theorem,
    secondary future-source theorem reopen, and reserve tail-refinement
    contracts have already been fixed.
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
SECONDARY_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_future_source_theorem_reopen_"
    "secondary_contract_declaration_gate_metrics.json"
)
SECONDARY_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_future_source_theorem_reopen_"
    "secondary_contract_numeric_evaluation_metrics.json"
)
RESERVE_INVENTORY = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_reserve_tail_refinement_"
    "contract_source_inventory_metrics.json"
)
RESERVE_AUDIT = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_reserve_tail_refinement_"
    "contract_audit_metrics.json"
)
RESERVE_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_reserve_tail_refinement_"
    "contract_declaration_gate_metrics.json"
)
RESERVE_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_reserve_tail_refinement_"
    "contract_numeric_evaluation_metrics.json"
)
ROUTE_LOCAL_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_route_local_no_go_theorem_review_"
    "declaration_gate_metrics.json"
)

NEXT_ROUTE = "8.7.56.1303"
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_current_pack_retained_lane_carry_over_contract"
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


# Function: write one JSON metrics payload and CSV row table.

def write_artifact(stem: str, data: dict) -> None:
    """Write one JSON metrics payload and CSV row table."""
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


# Function: execute the 8.7.56.1299-.1302 branch.

def main() -> None:
    """Execute the 8.7.56.1299-.1302 branch."""
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
        CLOSURE_GATE,
        PRIMARY_GATE,
        PRIMARY_EVAL,
        SECONDARY_GATE,
        SECONDARY_EVAL,
        RESERVE_INVENTORY,
        RESERVE_AUDIT,
        RESERVE_GATE,
        RESERVE_EVAL,
        ROUTE_LOCAL_GATE,
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

    closure_gate_summary = dict(read_json(CLOSURE_GATE)["summary"])
    primary_gate_summary = dict(read_json(PRIMARY_GATE)["summary"])
    primary_eval_summary = dict(read_json(PRIMARY_EVAL)["summary"])
    secondary_gate_summary = dict(read_json(SECONDARY_GATE)["summary"])
    secondary_eval_summary = dict(read_json(SECONDARY_EVAL)["summary"])
    reserve_inventory = read_json(RESERVE_INVENTORY)
    reserve_audit_summary = dict(read_json(RESERVE_AUDIT)["summary"])
    reserve_gate_summary = dict(read_json(RESERVE_GATE)["summary"])
    reserve_eval_summary = dict(read_json(RESERVE_EVAL)["summary"])
    route_local_gate_summary = dict(read_json(ROUTE_LOCAL_GATE)["summary"])

    reserve_contract_completed = (
        reserve_gate_summary["trial2_numeric_alpha_problem_classification"]
        == "vector_qball_form_factor_reserve_tail_refinement_contract_under_current_pack"
    )
    projection_theorem_primary_retained = (
        reserve_gate_summary["primary_residual_lane"]
        == "vector_qball_form_factor_projection_theorem_carry_over"
    )
    future_source_theorem_secondary_retained = (
        reserve_gate_summary["secondary_residual_lane"]
        == "qball_projection_overlap_future_source_theorem_reopen"
    )
    reserve_tail_refinement_retained = (
        reserve_gate_summary["reserve_residual_lane"]
        == "qball_projection_overlap_analytic_tail_theorem_refinement"
    )
    route_local_no_go_theorem_retained = bool(
        reserve_gate_summary["route_local_no_go_theorem_retained"]
    )
    vector_form_factor_exact_computation_unopened = not bool(
        reserve_eval_summary["vector_form_factor_exact_computation_ready_under_current_pack"]
    )
    numeric_state_unchanged = not bool(
        reserve_eval_summary["numeric_state_changed_by_current_branch"]
    )
    physical_reject_not_selected = not bool(
        reserve_gate_summary["physical_reject_required"]
    )
    retained_lane_hold_contract_ready = all(
        (
            reserve_contract_completed,
            projection_theorem_primary_retained,
            future_source_theorem_secondary_retained,
            reserve_tail_refinement_retained,
            route_local_no_go_theorem_retained,
            vector_form_factor_exact_computation_unopened,
            physical_reject_not_selected,
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
        },
        "prior_metrics": {
            "closure_gate": display_path(CLOSURE_GATE),
            "primary_gate": display_path(PRIMARY_GATE),
            "primary_eval": display_path(PRIMARY_EVAL),
            "secondary_gate": display_path(SECONDARY_GATE),
            "secondary_eval": display_path(SECONDARY_EVAL),
            "reserve_inventory": display_path(RESERVE_INVENTORY),
            "reserve_audit": display_path(RESERVE_AUDIT),
            "reserve_gate": display_path(RESERVE_GATE),
            "reserve_eval": display_path(RESERVE_EVAL),
            "route_local_gate": display_path(ROUTE_LOCAL_GATE),
        },
        "constants": {
            "beta_1": float(reserve_eval_summary["beta_1"]),
            "q_theory_over_m0": float(reserve_eval_summary["q_theory_over_m0"]),
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
        },
    }

    inventory = payload(
        "8.7.56.1299",
        "Trial-2 numeric alpha vector Q-ball form-factor current-pack retained-lane hold contract source inventory",
        inputs,
        [
            row(
                "reserve_contract_completed",
                "pass" if reserve_contract_completed else "reject",
                "reserve tail-refinement contract completed",
                1 if reserve_contract_completed else 0,
                "The top-level hold contract starts only after the reserve retained-lane contract is already fixed.",
            ),
            row(
                "projection_theorem_primary_retained",
                "pass" if projection_theorem_primary_retained else "reject",
                "projection-theorem primary retained",
                1 if projection_theorem_primary_retained else 0,
                "The theorem-side projection carry-over lane stays primary before top-level hold review.",
            ),
            row(
                "future_source_theorem_secondary_retained",
                "pass" if future_source_theorem_secondary_retained else "reject",
                "future-source-theorem secondary retained",
                1 if future_source_theorem_secondary_retained else 0,
                "The future source-theorem reopen lane stays secondary before top-level hold review.",
            ),
            row(
                "reserve_tail_refinement_retained",
                "pass" if reserve_tail_refinement_retained else "reject",
                "reserve tail-refinement retained",
                1 if reserve_tail_refinement_retained else 0,
                "The analytic tail refinement lane stays reserve before top-level hold review.",
            ),
            row(
                "vector_form_factor_exact_computation_unopened",
                "pass" if vector_form_factor_exact_computation_unopened else "reject",
                "vector form-factor exact computation unopened",
                1 if vector_form_factor_exact_computation_unopened else 0,
                "Current-pack exact vector computation remains unopened and must stay unopened in the hold contract.",
            ),
            row(
                "physical_reject_not_selected",
                "pass" if physical_reject_not_selected else "reject",
                "physical reject not selected",
                1 if physical_reject_not_selected else 0,
                "The hold contract must keep physical reject unselected.",
            ),
        ],
        {
            "reserve_contract_completed": reserve_contract_completed,
            "projection_theorem_primary_retained": projection_theorem_primary_retained,
            "future_source_theorem_secondary_retained": future_source_theorem_secondary_retained,
            "reserve_tail_refinement_retained": reserve_tail_refinement_retained,
            "vector_form_factor_exact_computation_unopened": vector_form_factor_exact_computation_unopened,
            "physical_reject_not_selected": physical_reject_not_selected,
            "selected_next_substep": "8.7.56.1300",
            "current_pack_limit_state": reserve_gate_summary[
                "trial2_numeric_alpha_problem_classification"
            ],
            "overall_status": (
                "trial2_numeric_alpha_vector_qball_form_factor_current_pack_retained_lane_hold_"
                "contract_inventory_fixed"
            ),
        },
        {
            "overall_status": (
                "trial2_numeric_alpha_vector_qball_form_factor_current_pack_retained_lane_hold_"
                "contract_inventory_fixed"
            ),
            "advance_to_8_7_56_1300": True,
            "next_required_artifacts": [
                "trial2_numeric_alpha_vector_qball_form_factor_current_pack_retained_lane_hold_contract_audit"
            ],
        },
        {
            "status_hits": {
                "status_1299": hit(status_text, "8.7.56.1299"),
                "hold_contract": hit(status_text, "current-pack retained-lane hold contract"),
            },
            "roadmap_hits": {
                "roadmap_1299": hit(roadmap_text, "`8.7.56.1299-.1302`"),
                "reserve_tail_refinement": hit(
                    roadmap_text, "qball_projection_overlap_analytic_tail_theorem_refinement"
                ),
            },
            "current_problem_hits": {
                "reserve_problem_class": hit(
                    current_problem_text,
                    "vector_qball_form_factor_reserve_tail_refinement_contract_under_current_pack",
                ),
                "next_hold": hit(current_problem_text, "current-pack hold contract"),
            },
            "current_status_hits": {
                "reserve_contract": hit(current_status_text, "reserve-tail-refinement contract"),
                "next_hold": hit(current_status_text, "current-pack hold contract"),
                "physical_reject": hit(current_status_text, "`physical_reject_required = false`"),
            },
            "paper_hits": {
                "part1_current": hit(part1_text, r"J^\mu_{\mathrm{matter}}"),
                "part3a_hold": hit(part3a_text, "current-pack retained-lane hold-contract next"),
                "part5_reserve_contract": hit(
                    part5_text, "reserve-tail-refinement contract under current pack"
                ),
            },
            "reserve_inventory_summary": reserve_inventory["summary"],
            "work_history_recent_hits": {
                "reserve_contract_entry": hit(
                    work_history_recent_text, "8.7.56.1295-.1298"
                ),
            },
        },
    )

    audit = payload(
        "8.7.56.1300",
        "Trial-2 numeric alpha vector Q-ball form-factor current-pack retained-lane hold contract audit",
        inputs,
        [
            row(
                "retained_lane_hold_contract_ready",
                "pass" if retained_lane_hold_contract_ready else "reject",
                "current-pack retained-lane hold contract ready",
                1 if retained_lane_hold_contract_ready else 0,
                "The top-level hold contract is honest only if the frozen primary/secondary/reserve ordering and non-reject guard all survive unchanged.",
            ),
            row(
                "projection_theorem_primary_retained",
                "pass" if projection_theorem_primary_retained else "reject",
                "projection-theorem primary retained",
                1 if projection_theorem_primary_retained else 0,
                "The projection theorem must remain the primary retained residual.",
            ),
            row(
                "future_source_theorem_secondary_retained",
                "pass" if future_source_theorem_secondary_retained else "reject",
                "future-source-theorem secondary retained",
                1 if future_source_theorem_secondary_retained else 0,
                "The future source-theorem reopen lane must remain secondary during top-level hold freeze.",
            ),
            row(
                "reserve_tail_refinement_retained",
                "pass" if reserve_tail_refinement_retained else "reject",
                "reserve tail-refinement retained",
                1 if reserve_tail_refinement_retained else 0,
                "The analytic tail refinement lane must remain reserve during top-level hold freeze.",
            ),
            row(
                "vector_form_factor_exact_computation_ready_under_current_pack",
                "reject",
                "vector form-factor exact computation ready under current pack",
                0,
                "Exact vector computation remains unopened under the current pack.",
            ),
            row(
                "physical_reject_required",
                "reject",
                "physical reject required",
                0,
                "Physical reject remains unselected.",
            ),
        ],
        {
            "projection_theorem_primary_retained": projection_theorem_primary_retained,
            "future_source_theorem_secondary_retained": future_source_theorem_secondary_retained,
            "reserve_tail_refinement_retained": reserve_tail_refinement_retained,
            "route_local_no_go_theorem_retained": route_local_no_go_theorem_retained,
            "vector_form_factor_exact_computation_ready_under_current_pack": False,
            "physical_reject_required": False,
            "retained_lane_hold_contract_ready": retained_lane_hold_contract_ready,
            "result_class": (
                "current_pack_retained_lane_hold_contract_honest"
                if retained_lane_hold_contract_ready
                else "current_pack_retained_lane_hold_contract_not_yet_honest"
            ),
        },
        {
            "overall_status": (
                "trial2_numeric_alpha_vector_qball_form_factor_current_pack_retained_lane_hold_"
                "contract_audit_completed"
            ),
            "advance_to_8_7_56_1301": True,
            "next_required_artifacts": [
                "trial2_numeric_alpha_vector_qball_form_factor_current_pack_retained_lane_hold_contract_declaration_gate"
            ],
        },
        {
            "reserve_audit_summary": reserve_audit_summary,
            "reserve_gate_summary": reserve_gate_summary,
            "route_local_gate_summary": route_local_gate_summary,
            "secondary_gate_summary": secondary_gate_summary,
            "primary_gate_summary": primary_gate_summary,
            "closure_gate_summary": closure_gate_summary,
        },
    )

    declaration_gate = payload(
        "8.7.56.1301",
        "Trial-2 numeric alpha vector Q-ball form-factor current-pack retained-lane hold contract declaration gate",
        inputs,
        [
            row(
                "retained_lane_hold_contract_ready",
                "pass" if retained_lane_hold_contract_ready else "reject",
                "current-pack retained-lane hold contract ready",
                1 if retained_lane_hold_contract_ready else 0,
                "The hold contract can be declared only after primary/secondary/reserve ordering has already been frozen.",
            ),
            row(
                "projection_theorem_primary_retained",
                "pass" if projection_theorem_primary_retained else "reject",
                "projection-theorem primary retained",
                1 if projection_theorem_primary_retained else 0,
                "The theorem-side projection lane remains the main retained residual.",
            ),
            row(
                "future_source_theorem_reopen_retained",
                "pass" if future_source_theorem_secondary_retained else "reject",
                "future-source-theorem reopen retained",
                1 if future_source_theorem_secondary_retained else 0,
                "The future source-theorem reopen lane remains the secondary retained residual.",
            ),
            row(
                "reserve_tail_refinement_retained",
                "pass" if reserve_tail_refinement_retained else "reject",
                "reserve tail-refinement retained",
                1 if reserve_tail_refinement_retained else 0,
                "The analytic tail refinement lane remains reserve.",
            ),
            row(
                "vector_form_factor_exact_computation_ready_under_current_pack",
                "reject",
                "vector form-factor exact computation ready under current pack",
                0,
                "Exact vector computation is still unopened under the current pack.",
            ),
            row(
                "physical_reject_required",
                "reject",
                "physical reject required",
                0,
                "Physical reject remains unselected.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": (
                "vector_qball_form_factor_current_pack_retained_lane_hold_contract_under_current_pack"
            ),
            "current_pack_limit_state": reserve_gate_summary[
                "trial2_numeric_alpha_problem_classification"
            ],
            "current_pack_retained_lane_hold_contract_ready": retained_lane_hold_contract_ready,
            "current_pack_retained_lane_hold_contract_honest": retained_lane_hold_contract_ready,
            "route_local_no_go_theorem_retained": route_local_no_go_theorem_retained,
            "projection_theorem_carry_over_required": projection_theorem_primary_retained,
            "future_source_theorem_reopen_retained": future_source_theorem_secondary_retained,
            "reserve_tail_refinement_retained": reserve_tail_refinement_retained,
            "primary_residual_lane": "vector_qball_form_factor_projection_theorem_carry_over",
            "secondary_residual_lane": "qball_projection_overlap_future_source_theorem_reopen",
            "reserve_residual_lane": "qball_projection_overlap_analytic_tail_theorem_refinement",
            "vector_form_factor_exact_computation_ready_under_current_pack": False,
            "physical_reject_required": False,
            "closeout_ready": False,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": (
                "trial2_numeric_alpha_vector_qball_form_factor_current_pack_retained_lane_hold_"
                "contract_declared"
            ),
            "advance_to_8_7_56_1302": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "audit_summary": audit["summary"],
            "reserve_eval_summary": reserve_eval_summary,
            "secondary_eval_summary": secondary_eval_summary,
            "primary_eval_summary": primary_eval_summary,
        },
    )

    evaluation = payload(
        "8.7.56.1302",
        "Trial-2 numeric alpha vector Q-ball form-factor current-pack retained-lane hold contract numeric evaluation",
        inputs,
        [
            row(
                "beta_1_fixed",
                "pass",
                "beta_1 fixed",
                float(reserve_eval_summary["beta_1"]),
                "The retained electron-like beta_1 stays fixed through the top-level hold contract branch.",
            ),
            row(
                "q_theory_over_m0_fixed",
                "pass",
                "q_theory/m0 fixed",
                float(reserve_eval_summary["q_theory_over_m0"]),
                "The retained matching-scale candidate stays fixed through the top-level hold contract branch.",
            ),
            row(
                "F_exact_at_q_theory_fixed",
                "pass",
                "F_exact at q_theory fixed",
                float(reserve_eval_summary["F_exact_at_q_theory"]),
                "The retained exact-profile overlap value stays fixed through the top-level hold contract branch.",
            ),
            row(
                "alpha_exact_at_q_theory_fixed",
                "pass",
                "alpha_exact at q_theory fixed",
                float(reserve_eval_summary["alpha_exact_at_q_theory"]),
                "The retained exact alpha candidate stays fixed through the top-level hold contract branch.",
            ),
            row(
                "exact_ground_state_polarization_weight_fixed",
                "pass",
                "exact ground-state polarization weight fixed",
                float(reserve_eval_summary["exact_ground_state_polarization_weight"]),
                "The current exact vector ground-state still stays at zero polarization weight.",
            ),
            row(
                "numeric_state_changed_by_current_branch",
                "reject" if numeric_state_unchanged else "pass",
                "numeric state changed by current branch",
                0 if numeric_state_unchanged else 1,
                "This branch only freezes the top-level hold order and does not create a new numeric evaluation.",
            ),
            row(
                "route_state_changed_by_current_branch",
                "pass",
                "route state changed by current branch",
                1,
                "The route now advances from the reserve contract freeze to an explicit current-pack retained-lane hold contract.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": (
                "vector_qball_form_factor_current_pack_retained_lane_hold_contract_under_current_pack"
            ),
            "beta_1": float(reserve_eval_summary["beta_1"]),
            "exact_ground_state_polarization_weight": float(
                reserve_eval_summary["exact_ground_state_polarization_weight"]
            ),
            "exact_ground_state_coupled_charge_factor": float(
                reserve_eval_summary["exact_ground_state_coupled_charge_factor"]
            ),
            "ell0_zero_seed_max_abs_fL": float(reserve_eval_summary["ell0_zero_seed_max_abs_fL"]),
            "scalar_literal_F_m0": float(reserve_eval_summary["scalar_literal_F_m0"]),
            "q_theory_over_m0": float(reserve_eval_summary["q_theory_over_m0"]),
            "F_exact_at_q_theory": float(reserve_eval_summary["F_exact_at_q_theory"]),
            "alpha_exact_at_q_theory": float(reserve_eval_summary["alpha_exact_at_q_theory"]),
            "vector_form_factor_exact_computation_ready_under_current_pack": False,
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": (
                "trial2_numeric_alpha_vector_qball_form_factor_current_pack_retained_lane_hold_"
                "contract_completed"
            ),
            "advance_to_next_route": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "prior_problem_classification": reserve_gate_summary[
                "trial2_numeric_alpha_problem_classification"
            ],
            "new_problem_classification": (
                "vector_qball_form_factor_current_pack_retained_lane_hold_contract_under_current_pack"
            ),
            "reserve_eval_summary": reserve_eval_summary,
        },
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_current_pack_retained_lane_hold_contract_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_current_pack_retained_lane_hold_contract_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_current_pack_retained_lane_hold_contract_declaration_gate",
        declaration_gate,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_current_pack_retained_lane_hold_contract_numeric_evaluation",
        evaluation,
    )

    print("[done] 8.7.56.1299-.1302 artifacts generated")


if __name__ == "__main__":
    main()
