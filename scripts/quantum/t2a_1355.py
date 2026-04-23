#!/usr/bin/env python3
"""Generate 8.7.56.1355-.1358 exploratory retained-lane top-level contract artifacts."""

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
OPERATOR_INV = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_exact_action_level_ell0_operator_"
    "reopen_retain_contract_source_inventory_metrics.json"
)
OPERATOR_AUDIT = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_exact_action_level_ell0_operator_"
    "reopen_retain_contract_audit_metrics.json"
)
OPERATOR_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_exact_action_level_ell0_operator_"
    "reopen_retain_contract_declaration_gate_metrics.json"
)
OPERATOR_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_exact_action_level_ell0_operator_"
    "reopen_retain_contract_numeric_evaluation_metrics.json"
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

PRIOR_CLASS = (
    "vector_qball_form_factor_exploratory_observable_dictionary_reserve_contract_under_exploratory_split"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_exploratory_retained_lane_top_level_contract_under_exploratory_split"
)
PRIMARY_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_exploratory_exact_action_level_ell0_operator_"
    "reopen_retain_contract"
)
SECONDARY_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_exploratory_effective_source_ansatz_secondary_contract"
)
RESERVE_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_exploratory_observable_dictionary_branch"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_exploratory_retained_lane_carry_over_contract"
)
NEXT_ROUTE = "8.7.56.1359"


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


# Function: execute the 8.7.56.1355-.1358 branch.

def main() -> None:
    """Execute the 8.7.56.1355-.1358 branch."""
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
        OPERATOR_GATE,
        OPERATOR_EVAL,
        OPERATOR_INV,
        OPERATOR_AUDIT,
        ROUTE_LOCAL_GATE,
        ROUTE_LOCAL_EVAL,
        SOURCE_INV,
        SOURCE_AUDIT,
        SOURCE_GATE,
        SOURCE_EVAL,
        DICT_INV,
        DICT_AUDIT,
        DICT_GATE,
        DICT_EVAL,
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
    operator_gate_summary = dict(read_json(OPERATOR_GATE)["summary"])
    operator_eval_summary = dict(read_json(OPERATOR_EVAL)["summary"])
    operator_inv = read_json(OPERATOR_INV)
    operator_audit = read_json(OPERATOR_AUDIT)
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

    part1_current_surface_available = (
        hit(part1_text, "J^\\mu_{\\mathrm{matter}}=(\\rho c,\\rho \\mathbf{v})") is not None
    )
    part1_interaction_surface_available = (
        hit(part1_text, "\\mathcal{L}_{\\mathrm{int}}=g_P\\,P_\\mu J^\\mu_{\\mathrm{matter}}") is not None
    )
    part3a_operator_reopen_wording_available = (
        hit(part3a_text, "exploratory exact-action-level `ell=0` operator reopen retain contract")
        is not None
    )
    part3a_source_wording_available = (
        hit(part3a_text, "exploratory effective-source-ansatz secondary contract") is not None
    )
    part3a_dict_wording_available = (
        hit(part3a_text, "exploratory observable-dictionary reserve contract") is not None
    )
    part5_operator_reopen_wording_available = (
        hit(part5_text, "exploratory exact-action-level `ell=0` operator reopen retain contract")
        is not None
    )
    part5_source_wording_available = (
        hit(part5_text, "exploratory effective-source-ansatz secondary contract") is not None
    )
    part5_dict_wording_available = (
        hit(part5_text, "exploratory observable-dictionary reserve contract") is not None
    )

    operator_primary_retained = bool(
        operator_gate_summary["exact_action_level_ell0_operator_reopen_primary_retained"]
    )
    source_secondary_retained = bool(
        source_gate_summary["effective_source_ansatz_branch_secondary_retained"]
    )
    dictionary_reserve_retained = bool(
        dict_gate_summary["observable_dictionary_branch_reserve_retained"]
    )
    operator_contract_honest = bool(
        operator_gate_summary["exact_action_level_ell0_operator_reopen_retain_honest"]
    )
    source_contract_honest = bool(
        source_gate_summary["effective_source_ansatz_secondary_contract_honest"]
    )
    dictionary_contract_honest = bool(
        dict_gate_summary["observable_dictionary_reserve_contract_honest"]
    )
    route_local_review_honest = bool(route_local_gate_summary["route_local_no_go_review_honest"])
    future_exact_operator_reopen_retained = bool(
        operator_gate_summary["future_exact_operator_reopen_retained"]
    )
    future_source_theorem_reopen_retained = bool(
        source_gate_summary["effective_source_ansatz_theorem_required"]
    )
    observable_dictionary_requires_exact_charge_current_bridge = bool(
        dict_gate_summary["observable_dictionary_requires_exact_charge_current_bridge"]
    )
    vector_form_factor_exact_computation_unopened = not bool(
        dict_gate_summary["vector_form_factor_exact_computation_ready_under_current_pack"]
    )
    physical_reject_not_selected = not bool(dict_gate_summary["physical_reject_required"])
    numeric_state_unchanged = not bool(dict_eval_summary["numeric_state_changed_by_current_branch"])

    retained_lane_top_level_contract_ready = all(
        (
            operator_primary_retained,
            source_secondary_retained,
            dictionary_reserve_retained,
            operator_contract_honest,
            source_contract_honest,
            dictionary_contract_honest,
            route_local_review_honest,
            part1_current_surface_available,
            part1_interaction_surface_available,
            part3a_operator_reopen_wording_available,
            part3a_source_wording_available,
            part3a_dict_wording_available,
            part5_operator_reopen_wording_available,
            part5_source_wording_available,
            part5_dict_wording_available,
            vector_form_factor_exact_computation_unopened,
            numeric_state_unchanged,
            physical_reject_not_selected,
        )
    )
    retained_lane_top_level_contract_honest = all(
        (
            retained_lane_top_level_contract_ready,
            future_exact_operator_reopen_retained,
            future_source_theorem_reopen_retained,
            observable_dictionary_requires_exact_charge_current_bridge,
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
            "route_local_eval": display_path(ROUTE_LOCAL_EVAL),
            "operator_inventory": display_path(OPERATOR_INV),
            "operator_audit": display_path(OPERATOR_AUDIT),
            "operator_gate": display_path(OPERATOR_GATE),
            "operator_eval": display_path(OPERATOR_EVAL),
            "source_inventory": display_path(SOURCE_INV),
            "source_audit": display_path(SOURCE_AUDIT),
            "source_gate": display_path(SOURCE_GATE),
            "source_eval": display_path(SOURCE_EVAL),
            "dictionary_inventory": display_path(DICT_INV),
            "dictionary_audit": display_path(DICT_AUDIT),
            "dictionary_gate": display_path(DICT_GATE),
            "dictionary_eval": display_path(DICT_EVAL),
        },
        "constants": {
            "beta_1": float_value(dict_eval_summary, "beta_1"),
            "q_theory_over_m0": float_value(dict_eval_summary, "q_theory_over_m0"),
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
        },
    }

    inventory = payload(
        "8.7.56.1355",
        "Trial-2 numeric alpha vector Q-ball form-factor exploratory retained-lane top-level contract source inventory",
        inputs,
        [
            row(
                "route_local_review_honest",
                "pass" if route_local_review_honest else "reject",
                "route-local no-go review honest",
                1 if route_local_review_honest else 0,
                "The top-level contract still rests on the earlier route-local scope cut.",
            ),
            row(
                "operator_primary_retained",
                "pass" if operator_primary_retained else "reject",
                "exact-action-level ell=0 operator reopen primary retained",
                1 if operator_primary_retained else 0,
                "The exact operator reopen lane remains the primary retained exploratory item.",
            ),
            row(
                "source_secondary_retained",
                "pass" if source_secondary_retained else "reject",
                "effective-source ansatz secondary retained",
                1 if source_secondary_retained else 0,
                "The source/current theorem lane remains secondary under the exploratory split.",
            ),
            row(
                "dictionary_reserve_retained",
                "pass" if dictionary_reserve_retained else "reject",
                "observable dictionary reserve retained",
                1 if dictionary_reserve_retained else 0,
                "The observable-dictionary gap remains the reserve retained lane.",
            ),
            row(
                "part1_current_surface_available",
                "pass" if part1_current_surface_available else "reject",
                "Part I matter current surface available",
                1 if part1_current_surface_available else 0,
                "The exploratory retained ordering still anchors to the explicit Part I matter current surface.",
            ),
            row(
                "part1_interaction_surface_available",
                "pass" if part1_interaction_surface_available else "reject",
                "Part I interaction surface available",
                1 if part1_interaction_surface_available else 0,
                "The exploratory retained ordering still anchors to the explicit interaction term.",
            ),
            row(
                "vector_form_factor_exact_computation_unopened",
                "pass" if vector_form_factor_exact_computation_unopened else "reject",
                "vector form-factor exact computation unopened",
                1 if vector_form_factor_exact_computation_unopened else 0,
                "Freezing the top-level retained ordering still does not open exact vector computation under the current pack.",
            ),
            row(
                "physical_reject_not_selected",
                "pass" if physical_reject_not_selected else "reject",
                "physical reject not selected",
                1 if physical_reject_not_selected else 0,
                "The exploratory split remains a retain-and-carry state rather than a physical reject.",
            ),
        ],
        {
            "retained_lane_top_level_contract_ready": retained_lane_top_level_contract_ready,
            "route_local_review_honest": route_local_review_honest,
            "operator_primary_retained": operator_primary_retained,
            "source_secondary_retained": source_secondary_retained,
            "dictionary_reserve_retained": dictionary_reserve_retained,
            "part1_current_surface_available": part1_current_surface_available,
            "part1_interaction_surface_available": part1_interaction_surface_available,
            "part3a_operator_reopen_wording_available": part3a_operator_reopen_wording_available,
            "part3a_source_wording_available": part3a_source_wording_available,
            "part3a_dict_wording_available": part3a_dict_wording_available,
            "part5_operator_reopen_wording_available": part5_operator_reopen_wording_available,
            "part5_source_wording_available": part5_source_wording_available,
            "part5_dict_wording_available": part5_dict_wording_available,
            "overall_status": (
                "trial2_numeric_alpha_vector_qball_form_factor_exploratory_retained_lane_top_level_"
                "contract_inventory_fixed"
            ),
        },
        {
            "overall_status": (
                "trial2_numeric_alpha_vector_qball_form_factor_exploratory_retained_lane_top_level_"
                "contract_inventory_fixed"
            ),
            "advance_to_8_7_56_1356": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "status_hit": hit(status_text, "8.7.56.1355"),
            "roadmap_hit": hit(roadmap_text, "8.7.56.1355"),
            "work_history_recent_hit": hit(work_history_recent_text, "8.7.56.1351-.1354"),
            "current_problem_hit": hit(current_problem_text, "top-level retained contract"),
            "current_status_hit": hit(current_status_text, "top-level retained contract"),
            "part1_current_hit": hit(part1_text, "J^\\mu_{\\mathrm{matter}}=(\\rho c,\\rho \\mathbf{v})"),
            "part1_interaction_hit": hit(part1_text, "\\mathcal{L}_{\\mathrm{int}}=g_P\\,P_\\mu J^\\mu_{\\mathrm{matter}}"),
            "route_local_inventory_summary": route_local_inv["summary"],
            "route_local_audit_summary": route_local_audit["summary"],
            "operator_inventory_summary": operator_inv["summary"],
            "operator_audit_summary": operator_audit["summary"],
            "source_inventory_summary": source_inv["summary"],
            "source_audit_summary": source_audit["summary"],
            "dictionary_inventory_summary": dict_inv["summary"],
            "dictionary_audit_summary": dict_audit["summary"],
        },
    )

    audit = payload(
        "8.7.56.1356",
        "Trial-2 numeric alpha vector Q-ball form-factor exploratory retained-lane top-level contract audit",
        inputs,
        [
            row(
                "retained_lane_top_level_contract_ready",
                "pass" if retained_lane_top_level_contract_ready else "reject",
                "retained-lane top-level contract ready",
                1 if retained_lane_top_level_contract_ready else 0,
                "The top-level contract is ready only if the operator/source/dictionary ordering stays unchanged and no exact computation is falsely reopened.",
            ),
            row(
                "retained_lane_top_level_contract_honest",
                "pass" if retained_lane_top_level_contract_honest else "reject",
                "retained-lane top-level contract honest",
                1 if retained_lane_top_level_contract_honest else 0,
                "The top-level contract is honest only if the retained future reopen / theorem-required language stays explicit.",
            ),
            row(
                "future_exact_operator_reopen_retained",
                "pass" if future_exact_operator_reopen_retained else "reject",
                "future exact-operator reopen retained",
                1 if future_exact_operator_reopen_retained else 0,
                "The primary lane remains a future exact-operator reopen requirement.",
            ),
            row(
                "future_source_theorem_reopen_retained",
                "pass" if future_source_theorem_reopen_retained else "reject",
                "future source-theorem reopen retained",
                1 if future_source_theorem_reopen_retained else 0,
                "The secondary lane remains a future source/current theorem requirement.",
            ),
            row(
                "observable_dictionary_requires_exact_charge_current_bridge",
                "pass" if observable_dictionary_requires_exact_charge_current_bridge else "reject",
                "observable dictionary requires exact charge-current bridge",
                1 if observable_dictionary_requires_exact_charge_current_bridge else 0,
                "The reserve lane remains a true final-observable mapping gap rather than a completed bridge.",
            ),
            row(
                "vector_form_factor_exact_computation_ready_under_current_pack",
                "reject",
                "vector form-factor exact computation ready under current pack",
                0,
                "Even after freezing the top-level ordering, exact vector computation remains unopened.",
            ),
            row(
                "physical_reject_required",
                "reject",
                "physical reject required",
                0,
                "Freezing the top-level retained ordering does not imply physical reject.",
            ),
        ],
        {
            "retained_lane_top_level_contract_ready": retained_lane_top_level_contract_ready,
            "retained_lane_top_level_contract_honest": retained_lane_top_level_contract_honest,
            "exact_action_level_ell0_operator_reopen_primary_retained": operator_primary_retained,
            "effective_source_ansatz_branch_secondary_retained": source_secondary_retained,
            "observable_dictionary_branch_reserve_retained": dictionary_reserve_retained,
            "future_exact_operator_reopen_retained": future_exact_operator_reopen_retained,
            "future_source_theorem_reopen_retained": future_source_theorem_reopen_retained,
            "observable_dictionary_requires_exact_charge_current_bridge": observable_dictionary_requires_exact_charge_current_bridge,
            "vector_form_factor_exact_computation_ready_under_current_pack": False,
            "physical_reject_required": False,
            "result_class": "exploratory_retained_lane_top_level_contract_honest",
        },
        {
            "overall_status": (
                "trial2_numeric_alpha_vector_qball_form_factor_exploratory_retained_lane_top_level_"
                "contract_audit_completed"
            ),
            "advance_to_8_7_56_1357": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "route_local_gate_summary": route_local_gate_summary,
            "route_local_eval_summary": route_local_eval_summary,
            "operator_gate_summary": operator_gate_summary,
            "operator_eval_summary": operator_eval_summary,
            "source_gate_summary": source_gate_summary,
            "source_eval_summary": source_eval_summary,
            "dictionary_gate_summary": dict_gate_summary,
            "dictionary_eval_summary": dict_eval_summary,
        },
    )

    declaration_gate = payload(
        "8.7.56.1357",
        "Trial-2 numeric alpha vector Q-ball form-factor exploratory retained-lane top-level contract declaration gate",
        inputs,
        [
            row(
                "retained_lane_top_level_contract_honest",
                "pass" if retained_lane_top_level_contract_honest else "reject",
                "retained-lane top-level contract honest",
                1 if retained_lane_top_level_contract_honest else 0,
                "The declaration gate freezes the exploratory ordering only; it does not claim any new theorem success.",
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
                "pass" if dictionary_reserve_retained else "reject",
                "observable dictionary reserve retained",
                1 if dictionary_reserve_retained else 0,
                "Reserve retained status remains with the missing final observable mapping.",
            ),
            row(
                "physical_reject_required",
                "reject",
                "physical reject required",
                0,
                "No physical reject follows from freezing the exploratory top-level contract.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "current_pack_limit_state": PRIOR_CLASS,
            "retained_lane_top_level_contract_ready": retained_lane_top_level_contract_ready,
            "retained_lane_top_level_contract_honest": retained_lane_top_level_contract_honest,
            "exact_action_level_ell0_operator_reopen_primary_retained": operator_primary_retained,
            "effective_source_ansatz_branch_secondary_retained": source_secondary_retained,
            "observable_dictionary_branch_reserve_retained": dictionary_reserve_retained,
            "future_exact_operator_reopen_retained": future_exact_operator_reopen_retained,
            "future_source_theorem_reopen_retained": future_source_theorem_reopen_retained,
            "observable_dictionary_requires_exact_charge_current_bridge": observable_dictionary_requires_exact_charge_current_bridge,
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
                "trial2_numeric_alpha_vector_qball_form_factor_exploratory_retained_lane_top_level_"
                "contract_declared"
            ),
            "advance_to_8_7_56_1358": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "audit_summary": audit["summary"],
            "dictionary_eval_summary": dict_eval_summary,
        },
    )

    evaluation = payload(
        "8.7.56.1358",
        "Trial-2 numeric alpha vector Q-ball form-factor exploratory retained-lane top-level contract numeric evaluation",
        inputs,
        [
            row(
                "beta_1_fixed",
                "pass",
                "beta_1 fixed",
                float_value(dict_eval_summary, "beta_1"),
                "The retained beta_1 baseline stays unchanged through the top-level exploratory contract freeze.",
            ),
            row(
                "q_theory_over_m0_fixed",
                "pass",
                "q_theory/m0 fixed",
                float_value(dict_eval_summary, "q_theory_over_m0"),
                "The retained matching-scale baseline stays unchanged through the top-level exploratory contract freeze.",
            ),
            row(
                "F_exact_at_q_theory_fixed",
                "pass",
                "F_exact at q_theory fixed",
                float_value(dict_eval_summary, "F_exact_at_q_theory"),
                "The retained exact-profile overlap baseline stays unchanged through the top-level exploratory contract freeze.",
            ),
            row(
                "alpha_exact_at_q_theory_fixed",
                "pass",
                "alpha exact at q_theory fixed",
                float_value(dict_eval_summary, "alpha_exact_at_q_theory"),
                "The retained alpha baseline stays unchanged through the top-level exploratory contract freeze.",
            ),
            row(
                "exact_ground_state_polarization_weight_fixed",
                "pass",
                "exact ground-state polarization weight fixed",
                float_value(dict_eval_summary, "exact_ground_state_polarization_weight"),
                "The exact ground state still stays at zero polarization weight under the current exact solver.",
            ),
            row(
                "numeric_state_changed_by_current_branch",
                "reject",
                "numeric state changed by current branch",
                0,
                "This branch only freezes the exploratory ordering and does not create a new numeric evaluation.",
            ),
            row(
                "route_state_changed_by_current_branch",
                "pass",
                "route state changed by current branch",
                1,
                "The route now advances from observable-dictionary reserve contract to exploratory retained-lane top-level contract.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "beta_1": float_value(dict_eval_summary, "beta_1"),
            "q_theory_over_m0": float_value(dict_eval_summary, "q_theory_over_m0"),
            "F_exact_at_q_theory": float_value(dict_eval_summary, "F_exact_at_q_theory"),
            "alpha_exact_at_q_theory": float_value(dict_eval_summary, "alpha_exact_at_q_theory"),
            "exact_ground_state_polarization_weight": float_value(
                dict_eval_summary,
                "exact_ground_state_polarization_weight",
            ),
            "exact_ground_state_coupled_charge_factor": float_value(
                dict_eval_summary,
                "exact_ground_state_coupled_charge_factor",
            ),
            "ell0_zero_seed_max_abs_fL": float_value(dict_eval_summary, "ell0_zero_seed_max_abs_fL"),
            "current_pilot_odd_series_singular_coefficient": float_value(
                dict_eval_summary,
                "current_pilot_odd_series_singular_coefficient",
            ),
            "exact_action_level_ell0_operator_reopen_primary_retained": operator_primary_retained,
            "effective_source_ansatz_branch_secondary_retained": source_secondary_retained,
            "observable_dictionary_branch_reserve_retained": dictionary_reserve_retained,
            "vector_form_factor_exact_computation_ready_under_current_pack": False,
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": (
                "trial2_numeric_alpha_vector_qball_form_factor_exploratory_retained_lane_top_level_"
                "contract_completed"
            ),
            "advance_to_next_route": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "prior_problem_classification": PRIOR_CLASS,
            "new_problem_classification": BRANCH_CLASS,
            "dictionary_eval_summary": dict_eval_summary,
        },
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_retained_lane_top_level_contract_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_retained_lane_top_level_contract_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_retained_lane_top_level_contract_declaration_gate",
        declaration_gate,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_retained_lane_top_level_contract_numeric_evaluation",
        evaluation,
    )

    print("[done] 8.7.56.1355-.1358 artifacts generated")


if __name__ == "__main__":
    main()
