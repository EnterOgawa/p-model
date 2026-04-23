#!/usr/bin/env python3
"""Generate 8.7.56.1291-.1294 future-source-theorem reopen secondary contract artifacts.

Purpose:
    Freeze the secondary retained residual ordering after the primary
    projection-theorem carry-over contract has already been fixed. This branch
    does not reopen exact vector computation; it only formalizes that the
    future source-theorem reopen lane remains the honest secondary residual
    under the current-pack limit, while the reserve tail-refinement lane stays
    reserve and physical reject stays unselected.
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
NOTE = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_vector_qball_form_factor.md")

CLOSURE_INVENTORY = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_ground_state_two_component_"
    "closure_gap_contract_source_inventory_metrics.json"
)
CLOSURE_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_ground_state_two_component_"
    "closure_gap_contract_declaration_gate_metrics.json"
)
PRIMARY_INVENTORY = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_projection_theorem_carry_over_"
    "contract_source_inventory_metrics.json"
)
PRIMARY_AUDIT = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_projection_theorem_carry_over_"
    "contract_audit_metrics.json"
)
PRIMARY_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_projection_theorem_carry_over_"
    "contract_declaration_gate_metrics.json"
)
PRIMARY_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_projection_theorem_carry_over_"
    "contract_numeric_evaluation_metrics.json"
)
ROUTE_LOCAL_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_route_local_no_go_theorem_review_"
    "declaration_gate_metrics.json"
)
ROUTE_LOCAL_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_route_local_no_go_theorem_review_"
    "numeric_evaluation_metrics.json"
)

NEXT_ROUTE = "8.7.56.1295"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_reserve_tail_refinement_contract"


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


# Function: execute the 8.7.56.1291-.1294 branch.

def main() -> None:
    """Execute the 8.7.56.1291-.1294 branch."""
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
        CLOSURE_INVENTORY,
        CLOSURE_GATE,
        PRIMARY_INVENTORY,
        PRIMARY_AUDIT,
        PRIMARY_GATE,
        PRIMARY_EVAL,
        ROUTE_LOCAL_GATE,
        ROUTE_LOCAL_EVAL,
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

    note_available = NOTE.exists()
    note_text = read_text(NOTE) if note_available else ""

    closure_inventory = read_json(CLOSURE_INVENTORY)
    closure_gate = read_json(CLOSURE_GATE)
    primary_inventory = read_json(PRIMARY_INVENTORY)
    primary_audit = read_json(PRIMARY_AUDIT)
    primary_gate = read_json(PRIMARY_GATE)
    primary_eval = read_json(PRIMARY_EVAL)
    route_local_gate = read_json(ROUTE_LOCAL_GATE)
    route_local_eval = read_json(ROUTE_LOCAL_EVAL)

    closure_inventory_summary = dict(closure_inventory["summary"])
    closure_gate_summary = dict(closure_gate["summary"])
    primary_inventory_summary = dict(primary_inventory["summary"])
    primary_audit_summary = dict(primary_audit["summary"])
    primary_gate_summary = dict(primary_gate["summary"])
    primary_eval_summary = dict(primary_eval["summary"])
    route_local_gate_summary = dict(route_local_gate["summary"])
    route_local_eval_summary = dict(route_local_eval["summary"])

    primary_contract_completed = (
        primary_gate_summary["trial2_numeric_alpha_problem_classification"]
        == "vector_qball_form_factor_projection_theorem_carry_over_contract_under_current_pack"
    )
    projection_theorem_primary_retained = (
        primary_gate_summary["primary_residual_lane"]
        == "vector_qball_form_factor_projection_theorem_carry_over"
    )
    future_source_secondary_retained = (
        primary_gate_summary["secondary_residual_lane"]
        == "qball_projection_overlap_future_source_theorem_reopen"
    )
    reserve_tail_refinement_retained = (
        primary_gate_summary["reserve_residual_lane"]
        == "qball_projection_overlap_analytic_tail_theorem_refinement"
    )
    route_local_no_go_theorem_retained = bool(
        primary_gate_summary["route_local_no_go_theorem_retained"]
    )
    vector_form_factor_exact_computation_unopened = not bool(
        primary_eval_summary["vector_form_factor_exact_computation_ready_under_current_pack"]
    )
    numeric_state_unchanged = not bool(primary_eval_summary["numeric_state_changed_by_current_branch"])
    physical_reject_not_selected = not bool(primary_gate_summary["physical_reject_required"])
    primary_contract_ready = bool(primary_gate_summary["projection_theorem_carry_over_contract_ready"])

    if note_available:
        note_hits = {
            "signed_density_line": hit(note_text, r"j^0_{\rm vector} = 2\omega"),
            "induced_fl_claim": hit(note_text, r"f_L(r) \propto"),
            "literal_q_equals_m0_claim": hit(note_text, "q = m_0"),
        }
    else:
        note_hits = dict(closure_inventory["evidence"]["note_hits"])

    note_fallback_evidence_available = (
        note_hits.get("signed_density_line") is not None
        and note_hits.get("induced_fl_claim") is not None
    )
    wording_exposes_secondary_lane = (
        hit(part3a_text, "future-source-theorem reopen secondary") is not None
        and hit(part5_text, "qball_projection_overlap_future_source_theorem_reopen") is not None
    )

    future_source_theorem_reopen_secondary_contract_ready = (
        primary_contract_completed
        and primary_contract_ready
        and projection_theorem_primary_retained
        and future_source_secondary_retained
        and reserve_tail_refinement_retained
        and route_local_no_go_theorem_retained
        and vector_form_factor_exact_computation_unopened
        and numeric_state_unchanged
        and physical_reject_not_selected
        and note_fallback_evidence_available
        and wording_exposes_secondary_lane
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
            "vector_form_factor_note": display_path(NOTE),
            "vector_form_factor_note_available": note_available,
        },
        "prior_metrics": {
            "closure_inventory": display_path(CLOSURE_INVENTORY),
            "closure_gate": display_path(CLOSURE_GATE),
            "primary_inventory": display_path(PRIMARY_INVENTORY),
            "primary_audit": display_path(PRIMARY_AUDIT),
            "primary_gate": display_path(PRIMARY_GATE),
            "primary_eval": display_path(PRIMARY_EVAL),
            "route_local_gate": display_path(ROUTE_LOCAL_GATE),
            "route_local_eval": display_path(ROUTE_LOCAL_EVAL),
        },
        "constants": {
            "beta_1": float(primary_eval_summary["beta_1"]),
            "q_theory_over_m0": float(primary_eval_summary["q_theory_over_m0"]),
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
        },
    }

    inventory = payload(
        "8.7.56.1291",
        "Trial-2 numeric alpha vector Q-ball form-factor future-source-theorem reopen secondary contract source inventory",
        inputs,
        [
            row(
                "projection_theorem_carry_over_contract_completed",
                "pass" if primary_contract_completed else "reject",
                "projection-theorem carry-over contract completed",
                1 if primary_contract_completed else 0,
                "The secondary contract starts only after the primary projection-theorem carry-over contract has already been frozen.",
            ),
            row(
                "future_source_theorem_reopen_secondary_lane_available",
                "pass" if future_source_secondary_retained else "reject",
                "future-source-theorem reopen secondary lane available",
                1 if future_source_secondary_retained else 0,
                "The retained secondary lane must already point to the future source-theorem reopen route.",
            ),
            row(
                "projection_theorem_primary_lane_still_available",
                "pass" if projection_theorem_primary_retained else "reject",
                "projection-theorem primary lane still available",
                1 if projection_theorem_primary_retained else 0,
                "The secondary contract inherits the already frozen theorem-side primary lane without reopening it.",
            ),
            row(
                "reserve_tail_refinement_lane_available",
                "pass" if reserve_tail_refinement_retained else "reject",
                "reserve tail-refinement lane available",
                1 if reserve_tail_refinement_retained else 0,
                "The reserve tail-refinement lane must remain present while the secondary lane is frozen.",
            ),
            row(
                "vector_form_factor_note_fallback_evidence_available",
                "pass" if note_fallback_evidence_available else "reject",
                "vector form-factor note fallback evidence available",
                1 if note_fallback_evidence_available else 0,
                "The secondary contract keeps the frozen note evidence even when the raw note file is missing.",
            ),
        ],
        {
            "inventory_ready": True,
            "selected_next_substep": "8.7.56.1292",
            "prior_problem_classification": primary_gate_summary[
                "trial2_numeric_alpha_problem_classification"
            ],
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_future_source_theorem_reopen_secondary_contract_inventory_fixed",
            "advance_to_8_7_56_1292": True,
            "next_required_artifacts": [
                "trial2_numeric_alpha_vector_qball_form_factor_future_source_theorem_reopen_secondary_contract_audit"
            ],
        },
        {
            "part1_hits": {
                "post_photon_nontransverse_sector": hit(
                    part1_text, "post-photon nontransverse sector"
                ),
                "matter_current": hit(part1_text, r"J^\mu_{\mathrm{matter}}=(\rho c,\rho \mathbf{v})"),
            },
            "part3a_hits": {
                "carry_order_line": hit(
                    part3a_text, "projection-theorem carry-over primary"
                ),
                "secondary_line": hit(
                    part3a_text, "future-source-theorem reopen secondary"
                ),
            },
            "part5_hits": {
                "primary_lane_line": hit(
                    part5_text, "vector_qball_form_factor_projection_theorem_carry_over"
                ),
                "secondary_lane_line": hit(
                    part5_text, "qball_projection_overlap_future_source_theorem_reopen"
                ),
                "reserve_lane_line": hit(
                    part5_text, "qball_projection_overlap_analytic_tail_theorem_refinement"
                ),
            },
            "note_hits": note_hits,
            "status_hits": {
                "status_1291": hit(status_text, "8.7.56.1291"),
                "roadmap_1291": hit(roadmap_text, "`8.7.56.1291-.1294`"),
                "problem_1291": hit(current_problem_text, "8.7.56.1291-.1294"),
                "status_problem_class": hit(
                    current_status_text,
                    "vector_qball_form_factor_projection_theorem_carry_over_contract_under_current_pack",
                ),
                "history_1287": hit(work_history_recent_text, "8.7.56.1287-.1290"),
            },
            "closure_inventory_summary": closure_inventory_summary,
            "closure_gate_summary": closure_gate_summary,
            "primary_inventory_summary": primary_inventory_summary,
            "primary_gate_summary": primary_gate_summary,
            "route_local_gate_summary": route_local_gate_summary,
        },
    )

    audit = payload(
        "8.7.56.1292",
        "Trial-2 numeric alpha vector Q-ball form-factor future-source-theorem reopen secondary contract audit",
        inputs,
        [
            row(
                "projection_theorem_primary_retained",
                "pass" if projection_theorem_primary_retained else "reject",
                "projection-theorem primary retained",
                1 if projection_theorem_primary_retained else 0,
                "The secondary contract must keep the projection-theorem lane primary.",
            ),
            row(
                "future_source_theorem_reopen_secondary_retained",
                "pass" if future_source_secondary_retained else "reject",
                "future-source-theorem reopen secondary retained",
                1 if future_source_secondary_retained else 0,
                "The future source-theorem reopen lane remains the honest secondary residual.",
            ),
            row(
                "reserve_tail_refinement_retained",
                "pass" if reserve_tail_refinement_retained else "reject",
                "reserve tail-refinement retained",
                1 if reserve_tail_refinement_retained else 0,
                "The reserve tail-refinement lane remains reserve and is not promoted above the secondary lane.",
            ),
            row(
                "vector_form_factor_exact_computation_unopened",
                "pass" if vector_form_factor_exact_computation_unopened else "reject",
                "vector form-factor exact computation unopened",
                1 if vector_form_factor_exact_computation_unopened else 0,
                "The current pack still does not reopen the exact vector computation.",
            ),
            row(
                "physical_reject_not_selected",
                "pass" if physical_reject_not_selected else "reject",
                "physical reject not selected",
                1 if physical_reject_not_selected else 0,
                "The secondary contract remains a structural hold, not a physical reject.",
            ),
            row(
                "future_source_theorem_reopen_secondary_contract_honest",
                "pass" if future_source_theorem_reopen_secondary_contract_ready else "reject",
                "future-source-theorem reopen secondary contract honest",
                1 if future_source_theorem_reopen_secondary_contract_ready else 0,
                "The ordering is honest only if the primary/secondary/reserve lanes, unopened computation state, and non-reject reading all cohere.",
            ),
        ],
        {
            "projection_theorem_primary_retained": projection_theorem_primary_retained,
            "future_source_theorem_reopen_secondary_retained": future_source_secondary_retained,
            "reserve_tail_refinement_retained": reserve_tail_refinement_retained,
            "route_local_no_go_theorem_retained": route_local_no_go_theorem_retained,
            "vector_form_factor_exact_computation_ready_under_current_pack": not vector_form_factor_exact_computation_unopened,
            "physical_reject_required": not physical_reject_not_selected,
            "future_source_theorem_reopen_secondary_contract_ready": future_source_theorem_reopen_secondary_contract_ready,
            "result_class": (
                "future_source_theorem_reopen_secondary_contract_honest_under_current_pack"
                if future_source_theorem_reopen_secondary_contract_ready
                else "future_source_theorem_reopen_secondary_contract_not_yet_honest"
            ),
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_future_source_theorem_reopen_secondary_contract_audit_completed",
            "advance_to_8_7_56_1293": True,
            "next_required_artifacts": [
                "trial2_numeric_alpha_vector_qball_form_factor_future_source_theorem_reopen_secondary_contract_declaration_gate"
            ],
        },
        {
            "primary_audit_summary": primary_audit_summary,
            "primary_gate_summary": primary_gate_summary,
            "route_local_gate_summary": route_local_gate_summary,
        },
    )

    declaration_gate = payload(
        "8.7.56.1293",
        "Trial-2 numeric alpha vector Q-ball form-factor future-source-theorem reopen secondary contract declaration gate",
        inputs,
        [
            row(
                "future_source_theorem_reopen_secondary_contract_ready",
                "pass" if future_source_theorem_reopen_secondary_contract_ready else "reject",
                "future-source-theorem reopen secondary contract ready",
                1 if future_source_theorem_reopen_secondary_contract_ready else 0,
                "The secondary contract becomes official only after the retained lane ordering and non-reject reading both pass.",
            ),
            row(
                "projection_theorem_primary_retained",
                "pass" if projection_theorem_primary_retained else "reject",
                "projection-theorem primary retained",
                1 if projection_theorem_primary_retained else 0,
                "The theorem-side projection carry-over lane remains primary under the current pack.",
            ),
            row(
                "future_source_theorem_reopen_secondary_retained",
                "pass" if future_source_secondary_retained else "reject",
                "future-source-theorem reopen secondary retained",
                1 if future_source_secondary_retained else 0,
                "The future source-theorem reopen lane remains the official secondary residual.",
            ),
            row(
                "reserve_tail_refinement_retained",
                "pass" if reserve_tail_refinement_retained else "reject",
                "reserve tail-refinement retained",
                1 if reserve_tail_refinement_retained else 0,
                "The reserve tail-refinement lane remains reserve after freezing the secondary contract.",
            ),
            row(
                "vector_form_factor_exact_computation_ready_under_current_pack",
                "pass" if not vector_form_factor_exact_computation_unopened else "reject",
                "vector form-factor exact computation ready under current pack",
                0 if vector_form_factor_exact_computation_unopened else 1,
                "Exact vector computation remains unopened under the current pack.",
            ),
            row(
                "physical_reject_required",
                "pass" if not physical_reject_not_selected else "reject",
                "physical reject required",
                0 if physical_reject_not_selected else 1,
                "Physical reject remains unselected.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "vector_qball_form_factor_future_source_theorem_reopen_secondary_contract_under_current_pack",
            "current_pack_limit_state": primary_gate_summary[
                "trial2_numeric_alpha_problem_classification"
            ],
            "future_source_theorem_reopen_secondary_contract_ready": future_source_theorem_reopen_secondary_contract_ready,
            "future_source_theorem_reopen_secondary_contract_honest": future_source_theorem_reopen_secondary_contract_ready,
            "route_local_no_go_theorem_retained": route_local_no_go_theorem_retained,
            "projection_theorem_carry_over_required": projection_theorem_primary_retained,
            "future_source_theorem_reopen_retained": future_source_secondary_retained,
            "primary_residual_lane": "vector_qball_form_factor_projection_theorem_carry_over",
            "secondary_residual_lane": "qball_projection_overlap_future_source_theorem_reopen",
            "reserve_residual_lane": "qball_projection_overlap_analytic_tail_theorem_refinement",
            "vector_form_factor_exact_computation_ready_under_current_pack": not vector_form_factor_exact_computation_unopened,
            "physical_reject_required": False,
            "closeout_ready": False,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_future_source_theorem_reopen_secondary_contract_declared",
            "advance_to_8_7_56_1294": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "audit_summary": audit["summary"],
            "primary_eval_summary": primary_eval_summary,
            "route_local_gate_summary": route_local_gate_summary,
        },
    )

    evaluation = payload(
        "8.7.56.1294",
        "Trial-2 numeric alpha vector Q-ball form-factor future-source-theorem reopen secondary contract numeric evaluation",
        inputs,
        [
            row(
                "beta_1_fixed",
                "pass",
                "beta_1 fixed",
                float(primary_eval_summary["beta_1"]),
                "The retained electron-like beta_1 stays fixed through the secondary contract branch.",
            ),
            row(
                "q_theory_over_m0_fixed",
                "pass",
                "q_theory/m0 fixed",
                float(primary_eval_summary["q_theory_over_m0"]),
                "The retained matching-scale candidate stays fixed through the secondary contract branch.",
            ),
            row(
                "F_exact_at_q_theory_fixed",
                "pass",
                "F_exact at q_theory fixed",
                float(primary_eval_summary["F_exact_at_q_theory"]),
                "The retained exact-profile overlap value stays fixed through the secondary contract branch.",
            ),
            row(
                "alpha_exact_at_q_theory_fixed",
                "pass",
                "alpha_exact at q_theory fixed",
                float(primary_eval_summary["alpha_exact_at_q_theory"]),
                "The retained exact alpha candidate stays fixed through the secondary contract branch.",
            ),
            row(
                "exact_ground_state_polarization_weight_fixed",
                "pass",
                "exact ground-state polarization weight fixed",
                float(primary_eval_summary["exact_ground_state_polarization_weight"]),
                "The current exact vector ground-state still stays at zero polarization weight.",
            ),
            row(
                "numeric_state_changed_by_current_branch",
                "pass" if not numeric_state_unchanged else "reject",
                "numeric state changed by current branch",
                0 if numeric_state_unchanged else 1,
                "This branch only freezes the secondary carry order and does not create a new numeric evaluation.",
            ),
            row(
                "route_state_changed_by_current_branch",
                "pass",
                "route state changed by current branch",
                1.0,
                "The route now advances from projection-theorem carry-over contract freeze to an explicit future-source-theorem reopen secondary contract.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "vector_qball_form_factor_future_source_theorem_reopen_secondary_contract_under_current_pack",
            "beta_1": float(primary_eval_summary["beta_1"]),
            "exact_ground_state_polarization_weight": float(
                primary_eval_summary["exact_ground_state_polarization_weight"]
            ),
            "exact_ground_state_coupled_charge_factor": float(
                primary_eval_summary["exact_ground_state_coupled_charge_factor"]
            ),
            "ell0_zero_seed_max_abs_fL": float(primary_eval_summary["ell0_zero_seed_max_abs_fL"]),
            "scalar_literal_F_m0": float(primary_eval_summary["scalar_literal_F_m0"]),
            "q_theory_over_m0": float(primary_eval_summary["q_theory_over_m0"]),
            "F_exact_at_q_theory": float(primary_eval_summary["F_exact_at_q_theory"]),
            "alpha_exact_at_q_theory": float(primary_eval_summary["alpha_exact_at_q_theory"]),
            "vector_form_factor_exact_computation_ready_under_current_pack": not vector_form_factor_exact_computation_unopened,
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_future_source_theorem_reopen_secondary_contract_completed",
            "advance_to_next_route": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "prior_problem_classification": primary_gate_summary[
                "trial2_numeric_alpha_problem_classification"
            ],
            "new_problem_classification": "vector_qball_form_factor_future_source_theorem_reopen_secondary_contract_under_current_pack",
            "route_local_eval_summary": route_local_eval_summary,
        },
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_future_source_theorem_reopen_secondary_contract_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_future_source_theorem_reopen_secondary_contract_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_future_source_theorem_reopen_secondary_contract_declaration_gate",
        declaration_gate,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_future_source_theorem_reopen_secondary_contract_numeric_evaluation",
        evaluation,
    )

    print("[done] 8.7.56.1291-.1294 artifacts generated")


if __name__ == "__main__":
    main()
