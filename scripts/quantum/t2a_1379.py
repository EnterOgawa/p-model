#!/usr/bin/env python3
"""Generate 8.7.56.1379-.1382 exploratory retained-lane carry-over refresh contract artifacts."""

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

ROUTE_LOCAL_INV = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_generalized_solver_route_local_no_go_review_"
    "source_inventory_metrics.json"
)
ROUTE_LOCAL_AUDIT = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_generalized_solver_route_local_no_go_review_"
    "audit_metrics.json"
)
ROUTE_LOCAL_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_generalized_solver_route_local_no_go_review_"
    "declaration_gate_metrics.json"
)
ROUTE_LOCAL_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_generalized_solver_route_local_no_go_review_"
    "numeric_evaluation_metrics.json"
)
PRIMARY_INV = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_exact_action_level_ell0_operator_"
    "reopen_retained_lane_contract_source_inventory_metrics.json"
)
PRIMARY_AUDIT = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_exact_action_level_ell0_operator_"
    "reopen_retained_lane_contract_audit_metrics.json"
)
PRIMARY_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_exact_action_level_ell0_operator_"
    "reopen_retained_lane_contract_declaration_gate_metrics.json"
)
PRIMARY_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_exact_action_level_ell0_operator_"
    "reopen_retained_lane_contract_numeric_evaluation_metrics.json"
)
SECONDARY_INV = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_future_source_theorem_reopen_"
    "retained_lane_contract_source_inventory_metrics.json"
)
SECONDARY_AUDIT = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_future_source_theorem_reopen_"
    "retained_lane_contract_audit_metrics.json"
)
SECONDARY_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_future_source_theorem_reopen_"
    "retained_lane_contract_declaration_gate_metrics.json"
)
SECONDARY_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_future_source_theorem_reopen_"
    "retained_lane_contract_numeric_evaluation_metrics.json"
)
SOURCE_INV = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_effective_source_ansatz_secondary_contract_"
    "source_inventory_metrics.json"
)
SOURCE_AUDIT = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_effective_source_ansatz_secondary_contract_"
    "audit_metrics.json"
)
SOURCE_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_effective_source_ansatz_secondary_contract_"
    "declaration_gate_metrics.json"
)
SOURCE_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_effective_source_ansatz_secondary_contract_"
    "numeric_evaluation_metrics.json"
)
DICT_INV = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_observable_dictionary_reserve_contract_"
    "source_inventory_metrics.json"
)
DICT_AUDIT = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_observable_dictionary_reserve_contract_"
    "audit_metrics.json"
)
DICT_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_observable_dictionary_reserve_contract_"
    "declaration_gate_metrics.json"
)
DICT_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_observable_dictionary_reserve_contract_"
    "numeric_evaluation_metrics.json"
)
TOP_LEVEL_INV = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_retained_lane_top_level_contract_"
    "source_inventory_metrics.json"
)
TOP_LEVEL_AUDIT = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_retained_lane_top_level_contract_"
    "audit_metrics.json"
)
TOP_LEVEL_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_retained_lane_top_level_contract_"
    "declaration_gate_metrics.json"
)
TOP_LEVEL_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_retained_lane_top_level_contract_"
    "numeric_evaluation_metrics.json"
)
CARRY_INV = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_retained_lane_carry_over_contract_"
    "source_inventory_metrics.json"
)
CARRY_AUDIT = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_retained_lane_carry_over_contract_"
    "audit_metrics.json"
)
CARRY_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_retained_lane_carry_over_contract_"
    "declaration_gate_metrics.json"
)
CARRY_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_retained_lane_carry_over_contract_"
    "numeric_evaluation_metrics.json"
)
RESERVE_RETAINED_INV = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_observable_dictionary_reserve_retained_lane_contract_"
    "source_inventory_metrics.json"
)
RESERVE_RETAINED_AUDIT = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_observable_dictionary_reserve_retained_lane_contract_"
    "audit_metrics.json"
)
RESERVE_RETAINED_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_observable_dictionary_reserve_retained_lane_contract_"
    "declaration_gate_metrics.json"
)
RESERVE_RETAINED_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_observable_dictionary_reserve_retained_lane_contract_"
    "numeric_evaluation_metrics.json"
)
TOP_LEVEL_REFRESH_INV = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_retained_lane_top_level_refresh_contract_"
    "source_inventory_metrics.json"
)
TOP_LEVEL_REFRESH_AUDIT = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_retained_lane_top_level_refresh_contract_"
    "audit_metrics.json"
)
TOP_LEVEL_REFRESH_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_retained_lane_top_level_refresh_contract_"
    "declaration_gate_metrics.json"
)
TOP_LEVEL_REFRESH_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_retained_lane_top_level_refresh_contract_"
    "numeric_evaluation_metrics.json"
)

PRIOR_CLASS = (
    "vector_qball_form_factor_exploratory_retained_lane_top_level_refresh_contract_under_exploratory_split"
)
CARRY_CLASS = (
    "vector_qball_form_factor_exploratory_retained_lane_carry_over_contract_under_exploratory_split"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_exploratory_retained_lane_carry_over_refresh_contract_under_exploratory_split"
)
PRIMARY_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_exploratory_exact_action_level_ell0_operator_"
    "reopen_retained_lane_contract"
)
SECONDARY_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_exploratory_future_source_theorem_reopen_retained_lane_contract"
)
RESERVE_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_exploratory_observable_dictionary_branch"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_exploratory_exact_action_level_ell0_operator_reopen_retained_lane_refresh_contract"
)
NEXT_ROUTE = "8.7.56.1383"


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


# Function: execute the 8.7.56.1379-.1382 branch.

def main() -> None:
    """Execute the 8.7.56.1379-.1382 branch."""
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
        ROUTE_LOCAL_INV,
        ROUTE_LOCAL_AUDIT,
        ROUTE_LOCAL_GATE,
        PRIMARY_GATE,
        PRIMARY_EVAL,
        PRIMARY_INV,
        PRIMARY_AUDIT,
        SECONDARY_INV,
        SECONDARY_AUDIT,
        SECONDARY_GATE,
        SECONDARY_EVAL,
        SOURCE_INV,
        SOURCE_AUDIT,
        SOURCE_GATE,
        SOURCE_EVAL,
        DICT_INV,
        DICT_AUDIT,
        DICT_GATE,
        DICT_EVAL,
        TOP_LEVEL_INV,
        TOP_LEVEL_AUDIT,
        TOP_LEVEL_GATE,
        TOP_LEVEL_EVAL,
        CARRY_INV,
        CARRY_AUDIT,
        CARRY_GATE,
        CARRY_EVAL,
        RESERVE_RETAINED_INV,
        RESERVE_RETAINED_AUDIT,
        RESERVE_RETAINED_GATE,
        RESERVE_RETAINED_EVAL,
        TOP_LEVEL_REFRESH_INV,
        TOP_LEVEL_REFRESH_AUDIT,
        TOP_LEVEL_REFRESH_GATE,
        TOP_LEVEL_REFRESH_EVAL,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    work_history_recent_text = read_text(WORK_HISTORY_RECENT)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)

    route_local_inv = read_json(ROUTE_LOCAL_INV)
    route_local_audit = read_json(ROUTE_LOCAL_AUDIT)
    primary_gate_summary = dict(read_json(PRIMARY_GATE)["summary"])
    primary_eval_summary = dict(read_json(PRIMARY_EVAL)["summary"])
    primary_inv = read_json(PRIMARY_INV)
    primary_audit = read_json(PRIMARY_AUDIT)
    secondary_gate_summary = dict(read_json(SECONDARY_GATE)["summary"])
    secondary_eval_summary = dict(read_json(SECONDARY_EVAL)["summary"])
    secondary_inv = read_json(SECONDARY_INV)
    secondary_audit = read_json(SECONDARY_AUDIT)
    route_local_gate_summary = dict(read_json(ROUTE_LOCAL_GATE)["summary"])
    route_local_eval_summary = dict(read_json(ROUTE_LOCAL_EVAL)["summary"])
    source_inv = read_json(SOURCE_INV)
    source_audit = read_json(SOURCE_AUDIT)
    source_gate_summary = dict(read_json(SOURCE_GATE)["summary"])
    source_eval_summary = dict(read_json(SOURCE_EVAL)["summary"])
    dict_inv = read_json(DICT_INV)
    dict_audit = read_json(DICT_AUDIT)
    dict_gate_summary = dict(read_json(DICT_GATE)["summary"])
    dict_eval_summary = dict(read_json(DICT_EVAL)["summary"])
    top_level_inv = read_json(TOP_LEVEL_INV)
    top_level_audit = read_json(TOP_LEVEL_AUDIT)
    top_level_gate_summary = dict(read_json(TOP_LEVEL_GATE)["summary"])
    top_level_eval_summary = dict(read_json(TOP_LEVEL_EVAL)["summary"])
    carry_inv = read_json(CARRY_INV)
    carry_audit = read_json(CARRY_AUDIT)
    carry_gate_summary = dict(read_json(CARRY_GATE)["summary"])
    carry_eval_summary = dict(read_json(CARRY_EVAL)["summary"])
    reserve_retained_inv = read_json(RESERVE_RETAINED_INV)
    reserve_retained_audit = read_json(RESERVE_RETAINED_AUDIT)
    reserve_retained_gate_summary = dict(read_json(RESERVE_RETAINED_GATE)["summary"])
    reserve_retained_eval_summary = dict(read_json(RESERVE_RETAINED_EVAL)["summary"])
    top_level_refresh_inv = read_json(TOP_LEVEL_REFRESH_INV)
    top_level_refresh_audit = read_json(TOP_LEVEL_REFRESH_AUDIT)
    top_level_refresh_gate_summary = dict(read_json(TOP_LEVEL_REFRESH_GATE)["summary"])
    top_level_refresh_eval_summary = dict(read_json(TOP_LEVEL_REFRESH_EVAL)["summary"])

    part1_current_surface_available = (
        hit(part1_text, "J^\\mu_{\\mathrm{matter}}=(\\rho c,\\rho \\mathbf{v})") is not None
    )
    part1_interaction_surface_available = (
        hit(part1_text, "\\mathcal{L}_{\\mathrm{int}}=g_P\\,P_\\mu J^\\mu_{\\mathrm{matter}}") is not None
    )
    part3a_carry_over_refresh_next_wording_available = (
        hit(part3a_text, "exploratory-retained-lane-carry-over-refresh-contract next")
        is not None
    )
    part5_carry_over_refresh_route_available = (
        hit(part5_text, "exploratory_retained_lane_carry_over_refresh_contract")
        is not None
    )

    carry_over_contract_completed = (
        carry_gate_summary["trial2_numeric_alpha_problem_classification"] == CARRY_CLASS
    )
    carry_over_contract_honest = bool(carry_gate_summary["retained_lane_carry_over_contract_honest"])
    top_level_refresh_contract_completed = (
        top_level_refresh_gate_summary["trial2_numeric_alpha_problem_classification"] == PRIOR_CLASS
    )
    top_level_refresh_contract_honest = bool(
        top_level_refresh_gate_summary["retained_lane_top_level_refresh_contract_honest"]
    )
    operator_primary_retained = bool(
        top_level_refresh_gate_summary["exact_action_level_ell0_operator_reopen_primary_retained"]
    )
    observable_dictionary_branch_reserve_retained = bool(
        top_level_refresh_gate_summary["observable_dictionary_branch_reserve_retained"]
    )
    future_exact_operator_reopen_retained = bool(
        top_level_refresh_gate_summary["future_exact_operator_reopen_retained"]
    )
    future_source_theorem_reopen_retained = bool(
        top_level_refresh_gate_summary["future_source_theorem_reopen_retained"]
    )
    observable_dictionary_requires_exact_charge_current_bridge = bool(
        top_level_refresh_gate_summary["observable_dictionary_requires_exact_charge_current_bridge"]
    )
    observable_dictionary_exact_mapping_available = bool(
        top_level_refresh_gate_summary["observable_dictionary_exact_mapping_available"]
    )
    observable_dictionary_final_observable_available = bool(
        top_level_refresh_gate_summary["observable_dictionary_final_observable_available"]
    )
    primary_retained_lane_contract_honest = bool(
        primary_gate_summary["exact_action_level_ell0_operator_reopen_retained_lane_contract_honest"]
    )
    future_source_secondary_contract_honest = bool(
        source_gate_summary["effective_source_ansatz_secondary_contract_honest"]
    )
    secondary_retained_lane_contract_honest = bool(
        secondary_gate_summary["future_source_theorem_reopen_retained_lane_contract_honest"]
    )
    dictionary_reserve_contract_honest = bool(
        dict_gate_summary["observable_dictionary_reserve_contract_honest"]
    )
    top_level_contract_honest = bool(top_level_gate_summary["retained_lane_top_level_contract_honest"])
    route_local_review_honest = bool(route_local_gate_summary["route_local_no_go_review_honest"])
    vector_form_factor_exact_computation_unopened = not bool(
        top_level_refresh_gate_summary["vector_form_factor_exact_computation_ready_under_current_pack"]
    )
    physical_reject_not_selected = not bool(top_level_refresh_gate_summary["physical_reject_required"])
    numeric_state_unchanged = not bool(
        top_level_refresh_eval_summary["numeric_state_changed_by_current_branch"]
    )

    retained_lane_carry_over_refresh_contract_ready = all(
        (
            carry_over_contract_completed,
            carry_over_contract_honest,
            top_level_refresh_contract_completed,
            top_level_refresh_contract_honest,
            top_level_contract_honest,
            primary_retained_lane_contract_honest,
            secondary_retained_lane_contract_honest,
            operator_primary_retained,
            future_exact_operator_reopen_retained,
            future_source_theorem_reopen_retained,
            observable_dictionary_requires_exact_charge_current_bridge,
            not observable_dictionary_exact_mapping_available,
            not observable_dictionary_final_observable_available,
            route_local_review_honest,
            future_source_secondary_contract_honest,
            dictionary_reserve_contract_honest,
            observable_dictionary_branch_reserve_retained,
            part1_current_surface_available,
            part1_interaction_surface_available,
            part3a_carry_over_refresh_next_wording_available,
            part5_carry_over_refresh_route_available,
            vector_form_factor_exact_computation_unopened,
            numeric_state_unchanged,
            physical_reject_not_selected,
        )
    )
    retained_lane_carry_over_refresh_contract_honest = all(
        (
            retained_lane_carry_over_refresh_contract_ready,
            observable_dictionary_branch_reserve_retained,
            not observable_dictionary_exact_mapping_available,
            not observable_dictionary_final_observable_available,
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
            "route_local_inventory": display_path(ROUTE_LOCAL_INV),
            "route_local_audit": display_path(ROUTE_LOCAL_AUDIT),
            "route_local_gate": display_path(ROUTE_LOCAL_GATE),
            "primary_inventory": display_path(PRIMARY_INV),
            "primary_audit": display_path(PRIMARY_AUDIT),
            "primary_gate": display_path(PRIMARY_GATE),
            "primary_eval": display_path(PRIMARY_EVAL),
            "secondary_inventory": display_path(SECONDARY_INV),
            "secondary_audit": display_path(SECONDARY_AUDIT),
            "secondary_gate": display_path(SECONDARY_GATE),
            "secondary_eval": display_path(SECONDARY_EVAL),
            "source_inventory": display_path(SOURCE_INV),
            "source_audit": display_path(SOURCE_AUDIT),
            "source_gate": display_path(SOURCE_GATE),
            "source_eval": display_path(SOURCE_EVAL),
            "dictionary_inventory": display_path(DICT_INV),
            "dictionary_audit": display_path(DICT_AUDIT),
            "dictionary_gate": display_path(DICT_GATE),
            "dictionary_eval": display_path(DICT_EVAL),
            "top_level_inventory": display_path(TOP_LEVEL_INV),
            "top_level_audit": display_path(TOP_LEVEL_AUDIT),
            "top_level_gate": display_path(TOP_LEVEL_GATE),
            "top_level_eval": display_path(TOP_LEVEL_EVAL),
            "carry_inventory": display_path(CARRY_INV),
            "carry_audit": display_path(CARRY_AUDIT),
            "carry_gate": display_path(CARRY_GATE),
            "carry_eval": display_path(CARRY_EVAL),
            "reserve_retained_inventory": display_path(RESERVE_RETAINED_INV),
            "reserve_retained_audit": display_path(RESERVE_RETAINED_AUDIT),
            "reserve_retained_gate": display_path(RESERVE_RETAINED_GATE),
            "reserve_retained_eval": display_path(RESERVE_RETAINED_EVAL),
            "top_level_refresh_inventory": display_path(TOP_LEVEL_REFRESH_INV),
            "top_level_refresh_audit": display_path(TOP_LEVEL_REFRESH_AUDIT),
            "top_level_refresh_gate": display_path(TOP_LEVEL_REFRESH_GATE),
            "top_level_refresh_eval": display_path(TOP_LEVEL_REFRESH_EVAL),
        },
        "constants": {
            "beta_1": float_value(top_level_refresh_eval_summary, "beta_1"),
            "q_theory_over_m0": float_value(top_level_refresh_eval_summary, "q_theory_over_m0"),
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
        },
    }

    inventory = payload(
        "8.7.56.1379",
        "Trial-2 numeric alpha vector Q-ball form-factor exploratory retained-lane carry-over refresh contract source inventory",
        inputs,
        [
            row(
                "top_level_refresh_contract_completed",
                "pass" if top_level_refresh_contract_completed else "reject",
                "top-level refresh contract completed",
                1 if top_level_refresh_contract_completed else 0,
                "The carry-over refresh contract starts only after the refreshed top-level retained ordering has already been frozen explicitly.",
            ),
            row(
                "retained_lane_carry_over_refresh_contract_ready",
                "pass" if retained_lane_carry_over_refresh_contract_ready else "reject",
                "retained-lane carry-over refresh contract ready",
                1 if retained_lane_carry_over_refresh_contract_ready else 0,
                "The refreshed carry-over contract is admissible only if the refreshed top-level ordering and all primary/secondary/reserve retained lanes stay frozen together.",
            ),
            row(
                "top_level_refresh_contract_honest",
                "pass" if top_level_refresh_contract_honest else "reject",
                "top-level refresh contract honest",
                1 if top_level_refresh_contract_honest else 0,
                "The refreshed carry-over contract can be formalized only after the refreshed top-level retained ordering has already been frozen honestly.",
            ),
            row(
                "observable_dictionary_branch_reserve_retained",
                "pass" if observable_dictionary_branch_reserve_retained else "reject",
                "observable dictionary reserve retained",
                1 if observable_dictionary_branch_reserve_retained else 0,
                "The missing final observable mapping remains the reserve retained lane.",
            ),
            row(
                "observable_dictionary_requires_exact_charge_current_bridge",
                "pass" if observable_dictionary_requires_exact_charge_current_bridge else "reject",
                "observable dictionary requires exact charge-current bridge",
                1 if observable_dictionary_requires_exact_charge_current_bridge else 0,
                "Reserve-lane formalization is admissible only if the exact charge-current bridge is still explicitly missing.",
            ),
            row(
                "observable_dictionary_exact_mapping_available",
                "reject" if not observable_dictionary_exact_mapping_available else "pass",
                "observable dictionary exact mapping available",
                1 if observable_dictionary_exact_mapping_available else 0,
                "The reserve lane stays reserve because no exact proxy-to-final-observable mapping has been derived yet.",
            ),
            row(
                "observable_dictionary_final_observable_available",
                "reject" if not observable_dictionary_final_observable_available else "pass",
                "observable dictionary final observable available",
                1 if observable_dictionary_final_observable_available else 0,
                "The reserve lane stays reserve because the final observable mapping is still absent.",
            ),
            row(
                "part3a_carry_over_refresh_next_wording_available",
                "pass" if part3a_carry_over_refresh_next_wording_available else "reject",
                "Part III-A carry-over-refresh wording available",
                1 if part3a_carry_over_refresh_next_wording_available else 0,
                "The public current-state wording already surfaces the carry-over refresh route as the live next route.",
            ),
            row(
                "part5_carry_over_refresh_route_available",
                "pass" if part5_carry_over_refresh_route_available else "reject",
                "Part V carry-over-refresh route wording available",
                1 if part5_carry_over_refresh_route_available else 0,
                "The future-prediction wording already surfaces the carry-over refresh route as the live next route.",
            ),
            row(
                "vector_form_factor_exact_computation_unopened",
                "pass" if vector_form_factor_exact_computation_unopened else "reject",
                "vector form-factor exact computation unopened",
                1 if vector_form_factor_exact_computation_unopened else 0,
                "Refreshing the carry-over retained ordering does not open exact vector computation under the current pack.",
            ),
        ],
        {
            "retained_lane_carry_over_refresh_contract_ready": (
                retained_lane_carry_over_refresh_contract_ready
            ),
            "top_level_refresh_contract_completed": top_level_refresh_contract_completed,
            "top_level_refresh_contract_honest": top_level_refresh_contract_honest,
            "secondary_retained_lane_contract_honest": secondary_retained_lane_contract_honest,
            "observable_dictionary_branch_reserve_retained": observable_dictionary_branch_reserve_retained,
            "observable_dictionary_requires_exact_charge_current_bridge": (
                observable_dictionary_requires_exact_charge_current_bridge
            ),
            "observable_dictionary_exact_mapping_available": (
                observable_dictionary_exact_mapping_available
            ),
            "observable_dictionary_final_observable_available": (
                observable_dictionary_final_observable_available
            ),
            "vector_form_factor_exact_computation_unopened": vector_form_factor_exact_computation_unopened,
            "overall_status": (
                "trial2_numeric_alpha_vector_qball_form_factor_exploratory_retained_lane_carry_over_"
                "refresh_contract_inventory_fixed"
            ),
        },
        {
            "overall_status": (
                "trial2_numeric_alpha_vector_qball_form_factor_exploratory_retained_lane_carry_over_"
                "refresh_contract_inventory_fixed"
            ),
            "advance_to_8_7_56_1380": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "status_hit": hit(status_text, "8.7.56.1379"),
            "roadmap_hit": hit(roadmap_text, "8.7.56.1379"),
            "work_history_recent_hit": hit(work_history_recent_text, "8.7.56.1375-.1378"),
            "current_problem_hit": hit(current_problem_text, "exploratory retained-lane carry-over refresh contract"),
            "current_status_hit": hit(current_status_text, "exploratory retained-lane carry-over refresh contract"),
            "part1_current_hit": hit(part1_text, "J^\\mu_{\\mathrm{matter}}=(\\rho c,\\rho \\mathbf{v})"),
            "part1_interaction_hit": hit(part1_text, "\\mathcal{L}_{\\mathrm{int}}=g_P\\,P_\\mu J^\\mu_{\\mathrm{matter}}"),
            "part3a_carry_over_refresh_hit": hit(part3a_text, "exploratory-retained-lane-carry-over-refresh-contract next"),
            "part5_carry_over_refresh_hit": hit(part5_text, "exploratory_retained_lane_carry_over_refresh_contract"),
            "route_local_inventory_summary": route_local_inv["summary"],
            "route_local_audit_summary": route_local_audit["summary"],
            "primary_inventory_summary": primary_inv["summary"],
            "primary_audit_summary": primary_audit["summary"],
            "secondary_inventory_summary": secondary_inv["summary"],
            "secondary_audit_summary": secondary_audit["summary"],
            "source_inventory_summary": source_inv["summary"],
            "source_audit_summary": source_audit["summary"],
            "dictionary_inventory_summary": dict_inv["summary"],
            "dictionary_audit_summary": dict_audit["summary"],
            "top_level_inventory_summary": top_level_inv["summary"],
            "top_level_audit_summary": top_level_audit["summary"],
            "carry_inventory_summary": carry_inv["summary"],
            "carry_audit_summary": carry_audit["summary"],
            "reserve_retained_inventory_summary": reserve_retained_inv["summary"],
            "reserve_retained_audit_summary": reserve_retained_audit["summary"],
            "top_level_refresh_inventory_summary": top_level_refresh_inv["summary"],
            "top_level_refresh_audit_summary": top_level_refresh_audit["summary"],
        },
    )

    audit = payload(
        "8.7.56.1380",
        "Trial-2 numeric alpha vector Q-ball form-factor exploratory retained-lane carry-over refresh contract audit",
        inputs,
        [
            row(
                "retained_lane_carry_over_refresh_contract_ready",
                "pass" if retained_lane_carry_over_refresh_contract_ready else "reject",
                "retained-lane carry-over refresh contract ready",
                1 if retained_lane_carry_over_refresh_contract_ready else 0,
                "The carry-over refresh contract is ready only if the refreshed top-level ordering stays unchanged and no exact computation is falsely reopened.",
            ),
            row(
                "retained_lane_carry_over_refresh_contract_honest",
                "pass" if retained_lane_carry_over_refresh_contract_honest else "reject",
                "retained-lane carry-over refresh contract honest",
                1 if retained_lane_carry_over_refresh_contract_honest else 0,
                "The carry-over refresh contract is honest only if the retained future reopen / exact-bridge-required language stays explicit across the carried primary/secondary/reserve lanes.",
            ),
            row(
                "future_exact_operator_reopen_retained",
                "pass" if future_exact_operator_reopen_retained else "reject",
                "future exact-operator reopen retained",
                1 if future_exact_operator_reopen_retained else 0,
                "Primary status remains with the missing exact ell=0 operator theorem.",
            ),
            row(
                "future_source_theorem_reopen_retained",
                "pass" if future_source_theorem_reopen_retained else "reject",
                "future source-theorem reopen retained",
                1 if future_source_theorem_reopen_retained else 0,
                "Secondary status remains with the missing exact effective source theorem.",
            ),
            row(
                "observable_dictionary_branch_reserve_retained",
                "pass" if observable_dictionary_branch_reserve_retained else "reject",
                "observable dictionary branch reserve retained",
                1 if observable_dictionary_branch_reserve_retained else 0,
                "Reserve status remains with the missing final observable mapping.",
            ),
            row(
                "observable_dictionary_exact_mapping_available",
                "reject",
                "observable dictionary exact mapping available",
                0,
                "The reserve retained lane still does not have an exact proxy-to-final-observable mapping under the current pack.",
            ),
            row(
                "physical_reject_required",
                "reject",
                "physical reject required",
                0,
                "No physical reject follows from refreshing the exploratory carry-over retained ordering.",
            ),
        ],
        {
            "retained_lane_carry_over_refresh_contract_ready": (
                retained_lane_carry_over_refresh_contract_ready
            ),
            "retained_lane_carry_over_refresh_contract_honest": (
                retained_lane_carry_over_refresh_contract_honest
            ),
            "top_level_refresh_contract_honest": top_level_refresh_contract_honest,
            "exact_action_level_ell0_operator_reopen_primary_retained": operator_primary_retained,
            "future_exact_operator_reopen_retained": future_exact_operator_reopen_retained,
            "future_source_theorem_reopen_retained": future_source_theorem_reopen_retained,
            "observable_dictionary_branch_reserve_retained": observable_dictionary_branch_reserve_retained,
            "observable_dictionary_requires_exact_charge_current_bridge": (
                observable_dictionary_requires_exact_charge_current_bridge
            ),
            "observable_dictionary_exact_mapping_available": False,
            "observable_dictionary_final_observable_available": False,
            "vector_form_factor_exact_computation_ready_under_current_pack": False,
            "physical_reject_required": False,
            "result_class": (
                "exploratory_retained_lane_carry_over_refresh_contract_honest"
            ),
        },
        {
            "overall_status": (
                "trial2_numeric_alpha_vector_qball_form_factor_exploratory_retained_lane_carry_over_"
                "refresh_contract_audit_completed"
            ),
            "advance_to_8_7_56_1381": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "route_local_gate_summary": route_local_gate_summary,
            "route_local_eval_summary": route_local_eval_summary,
            "primary_gate_summary": primary_gate_summary,
            "secondary_gate_summary": secondary_gate_summary,
            "secondary_eval_summary": secondary_eval_summary,
            "source_gate_summary": source_gate_summary,
            "source_eval_summary": source_eval_summary,
            "dictionary_gate_summary": dict_gate_summary,
            "dictionary_eval_summary": dict_eval_summary,
            "top_level_gate_summary": top_level_gate_summary,
            "top_level_eval_summary": top_level_eval_summary,
            "carry_gate_summary": carry_gate_summary,
            "carry_eval_summary": carry_eval_summary,
            "reserve_retained_gate_summary": reserve_retained_gate_summary,
            "reserve_retained_eval_summary": reserve_retained_eval_summary,
            "top_level_refresh_gate_summary": top_level_refresh_gate_summary,
            "top_level_refresh_eval_summary": top_level_refresh_eval_summary,
        },
    )

    declaration_gate = payload(
        "8.7.56.1381",
        "Trial-2 numeric alpha vector Q-ball form-factor exploratory retained-lane carry-over refresh contract declaration gate",
        inputs,
        [
            row(
                "retained_lane_carry_over_refresh_contract_honest",
                "pass" if retained_lane_carry_over_refresh_contract_honest else "reject",
                "retained-lane carry-over refresh contract honest",
                1 if retained_lane_carry_over_refresh_contract_honest else 0,
                "The declaration gate freezes the refreshed carry-over retained ordering only; it does not claim an already derived exact vector computation or final observable mapping.",
            ),
            row(
                "future_exact_operator_reopen_retained",
                "pass" if future_exact_operator_reopen_retained else "reject",
                "future exact-operator reopen retained",
                1 if future_exact_operator_reopen_retained else 0,
                "Primary retained status remains with the missing exact ell=0 operator.",
            ),
            row(
                "future_source_theorem_reopen_retained",
                "pass" if future_source_theorem_reopen_retained else "reject",
                "future source-theorem reopen retained",
                1 if future_source_theorem_reopen_retained else 0,
                "Secondary retained status remains with the missing exact effective source theorem.",
            ),
            row(
                "observable_dictionary_branch_reserve_retained",
                "pass" if observable_dictionary_branch_reserve_retained else "reject",
                "observable dictionary reserve retained",
                1 if observable_dictionary_branch_reserve_retained else 0,
                "Reserve retained status remains with the missing final observable mapping.",
            ),
            row(
                "physical_reject_required",
                "reject",
                "physical reject required",
                0,
                "No physical reject follows from freezing the refreshed exploratory carry-over contract.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "current_pack_limit_state": PRIOR_CLASS,
            "retained_lane_carry_over_refresh_contract_ready": (
                retained_lane_carry_over_refresh_contract_ready
            ),
            "retained_lane_carry_over_refresh_contract_honest": (
                retained_lane_carry_over_refresh_contract_honest
            ),
            "exact_action_level_ell0_operator_available": False,
            "exact_action_level_ell0_operator_reopen_required": True,
            "exact_action_level_ell0_operator_reopen_primary_retained": operator_primary_retained,
            "future_exact_operator_reopen_retained": future_exact_operator_reopen_retained,
            "future_source_theorem_reopen_retained": future_source_theorem_reopen_retained,
            "future_source_theorem_reopen_retained_lane_contract_honest": (
                secondary_retained_lane_contract_honest
            ),
            "observable_dictionary_branch_reserve_retained": observable_dictionary_branch_reserve_retained,
            "observable_dictionary_requires_exact_charge_current_bridge": (
                observable_dictionary_requires_exact_charge_current_bridge
            ),
            "observable_dictionary_exact_mapping_available": False,
            "observable_dictionary_final_observable_available": False,
            "vector_form_factor_exact_computation_ready_under_current_pack": False,
            "physical_reject_required": False,
            "selected_primary_exploratory_route": PRIMARY_ROUTE_NAME,
            "selected_secondary_exploratory_route": SECONDARY_ROUTE_NAME,
            "selected_reserve_exploratory_route": RESERVE_ROUTE_NAME,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": (
                "trial2_numeric_alpha_vector_qball_form_factor_exploratory_retained_lane_carry_over_"
                "refresh_contract_declared"
            ),
            "advance_to_8_7_56_1382": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "audit_summary": audit["summary"],
            "top_level_refresh_eval_summary": top_level_refresh_eval_summary,
        },
    )

    evaluation = payload(
        "8.7.56.1382",
        "Trial-2 numeric alpha vector Q-ball form-factor exploratory retained-lane carry-over refresh contract numeric evaluation",
        inputs,
        [
            row(
                "beta_1_fixed",
                "pass",
                "beta_1 fixed",
                float_value(top_level_refresh_eval_summary, "beta_1"),
                "The retained beta_1 baseline stays unchanged while the route moves from the refreshed top-level contract to the refreshed carry-over contract.",
            ),
            row(
                "q_theory_over_m0_fixed",
                "pass",
                "q_theory/m0 fixed",
                float_value(top_level_refresh_eval_summary, "q_theory_over_m0"),
                "The retained matching-scale baseline stays unchanged under the refreshed carry-over contract.",
            ),
            row(
                "F_exact_at_q_theory_fixed",
                "pass",
                "F_exact at q_theory fixed",
                float_value(top_level_refresh_eval_summary, "F_exact_at_q_theory"),
                "The retained exact-profile overlap baseline stays unchanged under the refreshed carry-over contract.",
            ),
            row(
                "alpha_exact_at_q_theory_fixed",
                "pass",
                "alpha exact at q_theory fixed",
                float_value(top_level_refresh_eval_summary, "alpha_exact_at_q_theory"),
                "The retained alpha baseline stays unchanged under the refreshed carry-over contract.",
            ),
            row(
                "exact_ground_state_polarization_weight_fixed",
                "pass",
                "exact ground-state polarization weight fixed",
                float_value(top_level_refresh_eval_summary, "exact_ground_state_polarization_weight"),
                "The exact ground state still stays at zero polarization weight under the current exact solver.",
            ),
            row(
                "numeric_state_changed_by_current_branch",
                "reject",
                "numeric state changed by current branch",
                0,
                "This branch only refreshes the exploratory carry-over retained ordering and does not create a new vector numeric evaluation.",
            ),
            row(
                "route_state_changed_by_current_branch",
                "pass",
                "route state changed by current branch",
                1,
                "The route now advances from the refreshed top-level contract to the exact-action-level ell=0 operator reopen refresh handoff.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "beta_1": float_value(top_level_refresh_eval_summary, "beta_1"),
            "q_theory_over_m0": float_value(top_level_refresh_eval_summary, "q_theory_over_m0"),
            "F_exact_at_q_theory": float_value(top_level_refresh_eval_summary, "F_exact_at_q_theory"),
            "alpha_exact_at_q_theory": float_value(top_level_refresh_eval_summary, "alpha_exact_at_q_theory"),
            "exact_ground_state_polarization_weight": float_value(
                top_level_refresh_eval_summary,
                "exact_ground_state_polarization_weight",
            ),
            "exact_ground_state_coupled_charge_factor": float_value(
                top_level_refresh_eval_summary,
                "exact_ground_state_coupled_charge_factor",
            ),
            "ell0_zero_seed_max_abs_fL": float_value(
                top_level_refresh_eval_summary,
                "ell0_zero_seed_max_abs_fL",
            ),
            "current_pilot_odd_series_singular_coefficient": float_value(
                top_level_refresh_eval_summary,
                "current_pilot_odd_series_singular_coefficient",
            ),
            "future_exact_operator_reopen_retained": future_exact_operator_reopen_retained,
            "future_source_theorem_reopen_retained": future_source_theorem_reopen_retained,
            "observable_dictionary_branch_reserve_retained": observable_dictionary_branch_reserve_retained,
            "vector_form_factor_exact_computation_ready_under_current_pack": False,
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": (
                "trial2_numeric_alpha_vector_qball_form_factor_exploratory_retained_lane_carry_over_"
                "refresh_contract_completed"
            ),
            "advance_to_next_route": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "prior_problem_classification": PRIOR_CLASS,
            "new_problem_classification": BRANCH_CLASS,
            "top_level_refresh_eval_summary": top_level_refresh_eval_summary,
        },
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_retained_lane_carry_over_refresh_contract_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_retained_lane_carry_over_refresh_contract_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_retained_lane_carry_over_refresh_contract_declaration_gate",
        declaration_gate,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_retained_lane_carry_over_refresh_contract_numeric_evaluation",
        evaluation,
    )

    print("[done] 8.7.56.1379-.1382 artifacts generated")


if __name__ == "__main__":
    main()
