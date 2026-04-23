#!/usr/bin/env python3
"""Generate 8.7.56.1323-.1326 future-source-theorem reopen retained-lane contract artifacts.

Purpose:
    Freeze the future-source-theorem reopen lane itself as the last retained
    current-canon branch after the fully frozen retained-lane carry-over
    contract has already been fixed. This branch still does not reopen exact
    vector computation; it formalizes that the theorem-side projection lane
    stays primary, the future-source-theorem reopen lane stays retained
    secondary, the analytic tail-refinement lane stays retained reserve, and
    the next official route shifts from current-canon retention toward an
    explicit current-canon closeout / exploratory split contract.
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
HOLD_INVENTORY = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_current_pack_retained_lane_hold_"
    "contract_source_inventory_metrics.json"
)
HOLD_AUDIT = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_current_pack_retained_lane_hold_"
    "contract_audit_metrics.json"
)
HOLD_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_current_pack_retained_lane_hold_"
    "contract_declaration_gate_metrics.json"
)
HOLD_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_current_pack_retained_lane_hold_"
    "contract_numeric_evaluation_metrics.json"
)
CARRY_INVENTORY = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_current_pack_retained_lane_carry_over_"
    "contract_source_inventory_metrics.json"
)
CARRY_AUDIT = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_current_pack_retained_lane_carry_over_"
    "contract_audit_metrics.json"
)
CARRY_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_current_pack_retained_lane_carry_over_"
    "contract_declaration_gate_metrics.json"
)
CARRY_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_current_pack_retained_lane_carry_over_"
    "contract_numeric_evaluation_metrics.json"
)
REOPEN_INVENTORY = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_future_source_theorem_reopen_"
    "retain_contract_source_inventory_metrics.json"
)
REOPEN_AUDIT = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_future_source_theorem_reopen_"
    "retain_contract_audit_metrics.json"
)
REOPEN_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_future_source_theorem_reopen_"
    "retain_contract_declaration_gate_metrics.json"
)
REOPEN_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_future_source_theorem_reopen_"
    "retain_contract_numeric_evaluation_metrics.json"
)
TOP_LEVEL_INVENTORY = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_retained_lane_top_level_"
    "contract_source_inventory_metrics.json"
)
TOP_LEVEL_AUDIT = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_retained_lane_top_level_"
    "contract_audit_metrics.json"
)
TOP_LEVEL_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_retained_lane_top_level_"
    "contract_declaration_gate_metrics.json"
)
TOP_LEVEL_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_retained_lane_top_level_"
    "contract_numeric_evaluation_metrics.json"
)
RETAINED_CARRY_INVENTORY = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_retained_lane_carry_over_"
    "contract_source_inventory_metrics.json"
)
RETAINED_CARRY_AUDIT = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_retained_lane_carry_over_"
    "contract_audit_metrics.json"
)
RETAINED_CARRY_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_retained_lane_carry_over_"
    "contract_declaration_gate_metrics.json"
)
RETAINED_CARRY_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_retained_lane_carry_over_"
    "contract_numeric_evaluation_metrics.json"
)
ROUTE_LOCAL_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_route_local_no_go_theorem_review_"
    "declaration_gate_metrics.json"
)

NEXT_ROUTE = "8.7.56.1327"
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_current_canon_closeout_exploratory_split_contract"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_future_source_theorem_reopen_retained_lane_contract_under_current_pack"
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


# Function: convert one summary value to float with default fallback.

def float_value(summary: dict, key: str, default: float = 0.0) -> float:
    """Convert one summary value to float with default fallback."""
    return float(summary.get(key, default))


# Function: execute the 8.7.56.1323-.1326 branch.

def main() -> None:
    """Execute the 8.7.56.1323-.1326 branch."""
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
        RESERVE_INVENTORY,
        RESERVE_AUDIT,
        RESERVE_GATE,
        RESERVE_EVAL,
        HOLD_INVENTORY,
        HOLD_AUDIT,
        HOLD_GATE,
        HOLD_EVAL,
        CARRY_INVENTORY,
        CARRY_AUDIT,
        CARRY_GATE,
        CARRY_EVAL,
        REOPEN_INVENTORY,
        REOPEN_AUDIT,
        REOPEN_GATE,
        REOPEN_EVAL,
        TOP_LEVEL_INVENTORY,
        TOP_LEVEL_AUDIT,
        TOP_LEVEL_GATE,
        TOP_LEVEL_EVAL,
        RETAINED_CARRY_INVENTORY,
        RETAINED_CARRY_AUDIT,
        RETAINED_CARRY_GATE,
        RETAINED_CARRY_EVAL,
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
    reserve_inventory = read_json(RESERVE_INVENTORY)
    reserve_audit_summary = dict(read_json(RESERVE_AUDIT)["summary"])
    reserve_gate_summary = dict(read_json(RESERVE_GATE)["summary"])
    reserve_eval_summary = dict(read_json(RESERVE_EVAL)["summary"])
    hold_inventory = read_json(HOLD_INVENTORY)
    hold_audit_summary = dict(read_json(HOLD_AUDIT)["summary"])
    hold_gate_summary = dict(read_json(HOLD_GATE)["summary"])
    hold_eval_summary = dict(read_json(HOLD_EVAL)["summary"])
    carry_inventory = read_json(CARRY_INVENTORY)
    carry_audit_summary = dict(read_json(CARRY_AUDIT)["summary"])
    carry_gate_summary = dict(read_json(CARRY_GATE)["summary"])
    carry_eval_summary = dict(read_json(CARRY_EVAL)["summary"])
    reopen_inventory = read_json(REOPEN_INVENTORY)
    reopen_audit_summary = dict(read_json(REOPEN_AUDIT)["summary"])
    reopen_gate_summary = dict(read_json(REOPEN_GATE)["summary"])
    reopen_eval_summary = dict(read_json(REOPEN_EVAL)["summary"])
    top_level_inventory = read_json(TOP_LEVEL_INVENTORY)
    top_level_audit_summary = dict(read_json(TOP_LEVEL_AUDIT)["summary"])
    top_level_gate_summary = dict(read_json(TOP_LEVEL_GATE)["summary"])
    top_level_eval_summary = dict(read_json(TOP_LEVEL_EVAL)["summary"])
    retained_carry_inventory = read_json(RETAINED_CARRY_INVENTORY)
    retained_carry_audit_summary = dict(read_json(RETAINED_CARRY_AUDIT)["summary"])
    retained_carry_gate_summary = dict(read_json(RETAINED_CARRY_GATE)["summary"])
    retained_carry_eval_summary = dict(read_json(RETAINED_CARRY_EVAL)["summary"])
    route_local_gate_summary = dict(read_json(ROUTE_LOCAL_GATE)["summary"])

    carry_over_contract_completed = (
        retained_carry_gate_summary["trial2_numeric_alpha_problem_classification"]
        == "vector_qball_form_factor_retained_lane_carry_over_contract_under_current_pack"
    )
    projection_theorem_primary_retained = (
        retained_carry_gate_summary["primary_residual_lane"]
        == "vector_qball_form_factor_projection_theorem_carry_over"
    )
    future_source_theorem_reopen_retained = (
        retained_carry_gate_summary["secondary_residual_lane"]
        == "qball_projection_overlap_future_source_theorem_reopen"
    )
    reserve_tail_refinement_retained = (
        retained_carry_gate_summary["reserve_residual_lane"]
        == "qball_projection_overlap_analytic_tail_theorem_refinement"
    )
    route_local_no_go_theorem_retained = bool(
        retained_carry_gate_summary["route_local_no_go_theorem_retained"]
    )
    vector_form_factor_exact_computation_unopened = not bool(
        retained_carry_eval_summary["vector_form_factor_exact_computation_ready_under_current_pack"]
    )
    numeric_state_unchanged = not bool(
        retained_carry_eval_summary["numeric_state_changed_by_current_branch"]
    )
    physical_reject_not_selected = not bool(
        retained_carry_gate_summary["physical_reject_required"]
    )
    future_source_theorem_reopen_retained_lane_contract_ready = all(
        (
            carry_over_contract_completed,
            projection_theorem_primary_retained,
            future_source_theorem_reopen_retained,
            reserve_tail_refinement_retained,
            route_local_no_go_theorem_retained,
            vector_form_factor_exact_computation_unopened,
            numeric_state_unchanged,
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
            "reserve_inventory": display_path(RESERVE_INVENTORY),
            "reserve_audit": display_path(RESERVE_AUDIT),
            "reserve_gate": display_path(RESERVE_GATE),
            "reserve_eval": display_path(RESERVE_EVAL),
            "hold_inventory": display_path(HOLD_INVENTORY),
            "hold_audit": display_path(HOLD_AUDIT),
            "hold_gate": display_path(HOLD_GATE),
            "hold_eval": display_path(HOLD_EVAL),
            "carry_inventory": display_path(CARRY_INVENTORY),
            "carry_audit": display_path(CARRY_AUDIT),
            "carry_gate": display_path(CARRY_GATE),
            "carry_eval": display_path(CARRY_EVAL),
            "reopen_inventory": display_path(REOPEN_INVENTORY),
            "reopen_audit": display_path(REOPEN_AUDIT),
            "reopen_gate": display_path(REOPEN_GATE),
            "reopen_eval": display_path(REOPEN_EVAL),
            "top_level_inventory": display_path(TOP_LEVEL_INVENTORY),
            "top_level_audit": display_path(TOP_LEVEL_AUDIT),
            "top_level_gate": display_path(TOP_LEVEL_GATE),
            "top_level_eval": display_path(TOP_LEVEL_EVAL),
            "retained_carry_inventory": display_path(RETAINED_CARRY_INVENTORY),
            "retained_carry_audit": display_path(RETAINED_CARRY_AUDIT),
            "retained_carry_gate": display_path(RETAINED_CARRY_GATE),
            "retained_carry_eval": display_path(RETAINED_CARRY_EVAL),
            "route_local_gate": display_path(ROUTE_LOCAL_GATE),
        },
        "constants": {
            "beta_1": float_value(retained_carry_eval_summary, "beta_1"),
            "q_theory_over_m0": float_value(retained_carry_eval_summary, "q_theory_over_m0"),
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
        },
    }

    inventory = payload(
        "8.7.56.1323",
        "Trial-2 numeric alpha vector Q-ball form-factor future-source-theorem reopen retained-lane contract source inventory",
        inputs,
        [
            row(
                "carry_over_contract_completed",
                "pass" if carry_over_contract_completed else "reject",
                "retained-lane carry-over contract completed",
                1 if carry_over_contract_completed else 0,
                "The future-source-theorem reopen retained-lane contract starts only after the retained-lane carry-over contract has already been fixed.",
            ),
            row(
                "projection_theorem_primary_retained",
                "pass" if projection_theorem_primary_retained else "reject",
                "projection-theorem primary retained",
                1 if projection_theorem_primary_retained else 0,
                "The theorem-side projection lane remains primary before future-source-theorem reopen retained-lane formalization.",
            ),
            row(
                "future_source_theorem_reopen_retained",
                "pass" if future_source_theorem_reopen_retained else "reject",
                "future-source-theorem reopen retained",
                1 if future_source_theorem_reopen_retained else 0,
                "The future source-theorem reopen lane remains the retained secondary lane before its retained-lane contract is formalized.",
            ),
            row(
                "reserve_tail_refinement_retained",
                "pass" if reserve_tail_refinement_retained else "reject",
                "reserve tail-refinement retained",
                1 if reserve_tail_refinement_retained else 0,
                "The analytic tail-refinement lane remains the retained reserve lane before the retained-lane reopen contract is formalized.",
            ),
            row(
                "vector_form_factor_exact_computation_unopened",
                "pass" if vector_form_factor_exact_computation_unopened else "reject",
                "vector form-factor exact computation unopened",
                1 if vector_form_factor_exact_computation_unopened else 0,
                "Current-pack exact vector computation remains unopened inside the future-source-theorem reopen retained-lane branch.",
            ),
            row(
                "physical_reject_not_selected",
                "pass" if physical_reject_not_selected else "reject",
                "physical reject not selected",
                1 if physical_reject_not_selected else 0,
                "The retained-lane reopen branch must keep physical reject unselected.",
            ),
        ],
        {
            "carry_over_contract_completed": carry_over_contract_completed,
            "projection_theorem_primary_retained": projection_theorem_primary_retained,
            "future_source_theorem_reopen_retained": future_source_theorem_reopen_retained,
            "reserve_tail_refinement_retained": reserve_tail_refinement_retained,
            "vector_form_factor_exact_computation_unopened": vector_form_factor_exact_computation_unopened,
            "physical_reject_not_selected": physical_reject_not_selected,
            "selected_next_substep": "8.7.56.1324",
            "current_pack_limit_state": retained_carry_gate_summary["trial2_numeric_alpha_problem_classification"],
            "overall_status": (
                "trial2_numeric_alpha_vector_qball_form_factor_future_source_theorem_reopen_retained_lane_"
                "contract_inventory_fixed"
            ),
        },
        {
            "overall_status": (
                "trial2_numeric_alpha_vector_qball_form_factor_future_source_theorem_reopen_retained_lane_"
                "contract_inventory_fixed"
            ),
            "advance_to_8_7_56_1324": True,
            "next_required_artifacts": [
                "trial2_numeric_alpha_vector_qball_form_factor_future_source_theorem_reopen_retained_lane_contract_audit"
            ],
        },
        {
            "status_hits": {
                "status_1323": hit(status_text, "8.7.56.1323"),
                "carry_over_contract": hit(status_text, "retained-lane carry-over contract"),
                "future_source_reopen_next": hit(status_text, "future-source-theorem reopen retained-lane contract"),
            },
            "roadmap_hits": {
                "roadmap_1323": hit(roadmap_text, "`8.7.56.1323-.1326`"),
                "carry_over_contract": hit(roadmap_text, "retained-lane carry-over contract"),
            },
            "current_problem_hits": {
                "carry_over_problem_class": hit(
                    current_problem_text,
                    "vector_qball_form_factor_retained_lane_carry_over_contract_under_current_pack",
                ),
                "future_source_reopen": hit(current_problem_text, "future-source-theorem reopen"),
            },
            "current_status_hits": {
                "carry_over_contract": hit(
                    current_status_text,
                    "retained-lane carry-over contract",
                ),
                "future_source_reopen": hit(current_status_text, "future-source-theorem reopen retained-lane contract"),
                "physical_reject": hit(current_status_text, "`physical_reject_required = false`"),
            },
            "paper_hits": {
                "part1_current": hit(part1_text, r"J^\mu_{\mathrm{matter}}"),
                "part3a_future_source_reopen_next": hit(
                    part3a_text, "future-source-theorem-reopen-retained-lane-contract next"
                ),
                "part5_carry_over": hit(part5_text, "retained-lane carry-over contract under current pack"),
            },
            "reserve_inventory_summary": reserve_inventory["summary"],
            "hold_inventory_summary": hold_inventory["summary"],
            "carry_inventory_summary": carry_inventory["summary"],
            "reopen_inventory_summary": reopen_inventory["summary"],
            "top_level_inventory_summary": top_level_inventory["summary"],
            "retained_carry_inventory_summary": retained_carry_inventory["summary"],
            "work_history_recent_hits": {
                "carry_over_contract_entry": hit(work_history_recent_text, "8.7.56.1319-.1322"),
            },
        },
    )

    audit = payload(
        "8.7.56.1324",
        "Trial-2 numeric alpha vector Q-ball form-factor future-source-theorem reopen retained-lane contract audit",
        inputs,
        [
            row(
                "future_source_theorem_reopen_retained_lane_contract_ready",
                "pass" if future_source_theorem_reopen_retained_lane_contract_ready else "reject",
                "future-source-theorem reopen retained-lane contract ready",
                1 if future_source_theorem_reopen_retained_lane_contract_ready else 0,
                "The retained-lane reopen contract is honest only if the frozen carry-over ordering survives unchanged and no exact vector computation is reopened.",
            ),
            row(
                "projection_theorem_primary_retained",
                "pass" if projection_theorem_primary_retained else "reject",
                "projection-theorem primary retained",
                1 if projection_theorem_primary_retained else 0,
                "The projection theorem must remain the primary retained residual.",
            ),
            row(
                "future_source_theorem_reopen_retained",
                "pass" if future_source_theorem_reopen_retained else "reject",
                "future-source-theorem reopen retained",
                1 if future_source_theorem_reopen_retained else 0,
                "The future source-theorem reopen lane must remain the retained secondary lane.",
            ),
            row(
                "reserve_tail_refinement_retained",
                "pass" if reserve_tail_refinement_retained else "reject",
                "reserve tail-refinement retained",
                1 if reserve_tail_refinement_retained else 0,
                "The analytic tail-refinement lane must remain the retained reserve lane.",
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
            "future_source_theorem_reopen_retained": future_source_theorem_reopen_retained,
            "reserve_tail_refinement_retained": reserve_tail_refinement_retained,
            "route_local_no_go_theorem_retained": route_local_no_go_theorem_retained,
            "vector_form_factor_exact_computation_ready_under_current_pack": False,
            "physical_reject_required": False,
            "future_source_theorem_reopen_retained_lane_contract_ready": (
                future_source_theorem_reopen_retained_lane_contract_ready
            ),
            "result_class": (
                "future_source_theorem_reopen_retained_lane_contract_honest"
                if future_source_theorem_reopen_retained_lane_contract_ready
                else "future_source_theorem_reopen_retained_lane_contract_not_yet_honest"
            ),
        },
        {
            "overall_status": (
                "trial2_numeric_alpha_vector_qball_form_factor_future_source_theorem_reopen_retained_lane_"
                "contract_audit_completed"
            ),
            "advance_to_8_7_56_1325": True,
            "next_required_artifacts": [
                "trial2_numeric_alpha_vector_qball_form_factor_future_source_theorem_reopen_retained_lane_contract_declaration_gate"
            ],
        },
        {
            "reserve_audit_summary": reserve_audit_summary,
            "reserve_gate_summary": reserve_gate_summary,
            "hold_audit_summary": hold_audit_summary,
            "hold_gate_summary": hold_gate_summary,
            "carry_audit_summary": carry_audit_summary,
            "current_pack_carry_gate_summary": carry_gate_summary,
            "retained_carry_audit_summary": retained_carry_audit_summary,
            "retained_carry_gate_summary": retained_carry_gate_summary,
            "reopen_audit_summary": reopen_audit_summary,
            "reopen_gate_summary": reopen_gate_summary,
            "top_level_audit_summary": top_level_audit_summary,
            "top_level_gate_summary": top_level_gate_summary,
            "route_local_gate_summary": route_local_gate_summary,
            "primary_gate_summary": primary_gate_summary,
            "closure_gate_summary": closure_gate_summary,
        },
    )

    declaration_gate = payload(
        "8.7.56.1325",
        "Trial-2 numeric alpha vector Q-ball form-factor future-source-theorem reopen retained-lane contract declaration gate",
        inputs,
        [
            row(
                "future_source_theorem_reopen_retained_lane_contract_ready",
                "pass" if future_source_theorem_reopen_retained_lane_contract_ready else "reject",
                "future-source-theorem reopen retained-lane contract ready",
                1 if future_source_theorem_reopen_retained_lane_contract_ready else 0,
                "The retained-lane reopen contract can be declared only after the retained-lane carry-over ordering has already been fixed.",
            ),
            row(
                "projection_theorem_carry_over_required",
                "pass" if projection_theorem_primary_retained else "reject",
                "projection-theorem carry-over required",
                1 if projection_theorem_primary_retained else 0,
                "The theorem-side projection lane remains the main retained residual.",
            ),
            row(
                "future_source_theorem_reopen_retained",
                "pass" if future_source_theorem_reopen_retained else "reject",
                "future-source-theorem reopen retained",
                1 if future_source_theorem_reopen_retained else 0,
                "The future source-theorem reopen lane remains the retained secondary residual.",
            ),
            row(
                "reserve_tail_refinement_retained",
                "pass" if reserve_tail_refinement_retained else "reject",
                "reserve tail-refinement retained",
                1 if reserve_tail_refinement_retained else 0,
                "The analytic tail-refinement lane remains the retained reserve residual.",
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
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "current_pack_limit_state": retained_carry_gate_summary["trial2_numeric_alpha_problem_classification"],
            "future_source_theorem_reopen_retained_lane_contract_ready": (
                future_source_theorem_reopen_retained_lane_contract_ready
            ),
            "future_source_theorem_reopen_retained_lane_contract_honest": (
                future_source_theorem_reopen_retained_lane_contract_ready
            ),
            "route_local_no_go_theorem_retained": route_local_no_go_theorem_retained,
            "projection_theorem_carry_over_required": projection_theorem_primary_retained,
            "future_source_theorem_reopen_retained": future_source_theorem_reopen_retained,
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
                "trial2_numeric_alpha_vector_qball_form_factor_future_source_theorem_reopen_retained_lane_"
                "contract_declared"
            ),
            "advance_to_8_7_56_1326": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "audit_summary": audit["summary"],
            "reopen_eval_summary": reopen_eval_summary,
            "reserve_eval_summary": reserve_eval_summary,
            "retained_carry_eval_summary": retained_carry_eval_summary,
            "primary_eval_summary": primary_eval_summary,
            "top_level_eval_summary": top_level_eval_summary,
        },
    )

    evaluation = payload(
        "8.7.56.1326",
        "Trial-2 numeric alpha vector Q-ball form-factor future-source-theorem reopen retained-lane contract numeric evaluation",
        inputs,
        [
            row(
                "beta_1_fixed",
                "pass",
                "beta_1 fixed",
                float_value(retained_carry_eval_summary, "beta_1"),
                "The retained electron-like beta_1 stays fixed through the retained-lane reopen contract branch.",
            ),
            row(
                "q_theory_over_m0_fixed",
                "pass",
                "q_theory/m0 fixed",
                float_value(retained_carry_eval_summary, "q_theory_over_m0"),
                "The retained matching-scale candidate stays fixed through the retained-lane reopen contract branch.",
            ),
            row(
                "F_exact_at_q_theory_fixed",
                "pass",
                "F_exact at q_theory fixed",
                float_value(retained_carry_eval_summary, "F_exact_at_q_theory"),
                "The retained exact-profile overlap value stays fixed through the retained-lane reopen contract branch.",
            ),
            row(
                "alpha_exact_at_q_theory_fixed",
                "pass",
                "alpha_exact at q_theory fixed",
                float_value(retained_carry_eval_summary, "alpha_exact_at_q_theory"),
                "The retained exact alpha candidate stays fixed through the retained-lane reopen contract branch.",
            ),
            row(
                "exact_ground_state_polarization_weight_fixed",
                "pass",
                "exact ground-state polarization weight fixed",
                float_value(retained_carry_eval_summary, "exact_ground_state_polarization_weight"),
                "The current exact vector ground-state still stays at zero polarization weight.",
            ),
            row(
                "numeric_state_changed_by_current_branch",
                "reject" if numeric_state_unchanged else "pass",
                "numeric state changed by current branch",
                0 if numeric_state_unchanged else 1,
                "This branch only freezes the retained future-source reopen lane and does not create a new numeric evaluation.",
            ),
            row(
                "route_state_changed_by_current_branch",
                "pass",
                "route state changed by current branch",
                1,
                "The route now advances from retained-lane carry-over freeze to an explicit retained-lane reopen contract.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "beta_1": float_value(retained_carry_eval_summary, "beta_1"),
            "exact_ground_state_polarization_weight": float_value(
                retained_carry_eval_summary, "exact_ground_state_polarization_weight"
            ),
            "exact_ground_state_coupled_charge_factor": float_value(
                retained_carry_eval_summary, "exact_ground_state_coupled_charge_factor"
            ),
            "ell0_zero_seed_max_abs_fL": float_value(
                retained_carry_eval_summary, "ell0_zero_seed_max_abs_fL"
            ),
            "scalar_literal_F_m0": float_value(retained_carry_eval_summary, "scalar_literal_F_m0"),
            "q_theory_over_m0": float_value(retained_carry_eval_summary, "q_theory_over_m0"),
            "F_exact_at_q_theory": float_value(retained_carry_eval_summary, "F_exact_at_q_theory"),
            "alpha_exact_at_q_theory": float_value(retained_carry_eval_summary, "alpha_exact_at_q_theory"),
            "vector_form_factor_exact_computation_ready_under_current_pack": False,
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": (
                "trial2_numeric_alpha_vector_qball_form_factor_future_source_theorem_reopen_retained_lane_"
                "contract_completed"
            ),
            "advance_to_next_route": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "prior_problem_classification": retained_carry_gate_summary["trial2_numeric_alpha_problem_classification"],
            "new_problem_classification": BRANCH_CLASS,
            "retained_carry_eval_summary": retained_carry_eval_summary,
        },
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_future_source_theorem_reopen_retained_lane_contract_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_future_source_theorem_reopen_retained_lane_contract_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_future_source_theorem_reopen_retained_lane_contract_declaration_gate",
        declaration_gate,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_future_source_theorem_reopen_retained_lane_contract_numeric_evaluation",
        evaluation,
    )

    print("[done] 8.7.56.1323-.1326 artifacts generated")


if __name__ == "__main__":
    main()
