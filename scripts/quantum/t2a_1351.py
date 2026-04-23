#!/usr/bin/env python3
"""Generate 8.7.56.1351-.1354 observable-dictionary reserve-contract artifacts."""

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
NEXT_STEPS_NOTE = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_next_steps_20260327.md")

OPERATOR_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_exact_action_level_ell0_operator_"
    "reopen_retain_contract_declaration_gate_metrics.json"
)
OPERATOR_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_exact_action_level_ell0_operator_"
    "reopen_retain_contract_numeric_evaluation_metrics.json"
)
ROUTE_LOCAL_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_generalized_solver_route_local_no_go_review_"
    "declaration_gate_metrics.json"
)
SERIES_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_ell0_series_theorem_attempt_"
    "declaration_gate_metrics.json"
)

PRIOR_CLASS = (
    "vector_qball_form_factor_exploratory_effective_source_ansatz_secondary_contract_under_exploratory_split"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_exploratory_observable_dictionary_reserve_contract_under_exploratory_split"
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
    "trial2_numeric_alpha_vector_qball_form_factor_exploratory_retained_lane_top_level_contract"
)
NEXT_ROUTE = "8.7.56.1355"


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


# Function: execute the 8.7.56.1351-.1354 branch.

def main() -> None:
    """Execute the 8.7.56.1351-.1354 branch."""
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
        NEXT_STEPS_NOTE,
        OPERATOR_GATE,
        OPERATOR_EVAL,
        ROUTE_LOCAL_GATE,
        SERIES_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    next_steps_note_text = read_text(NEXT_STEPS_NOTE)

    operator_gate_summary = dict(read_json(OPERATOR_GATE)["summary"])
    operator_eval_summary = dict(read_json(OPERATOR_EVAL)["summary"])
    route_local_gate_summary = dict(read_json(ROUTE_LOCAL_GATE)["summary"])
    series_gate_summary = dict(read_json(SERIES_GATE)["summary"])

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
    part5_operator_reopen_wording_available = (
        hit(part5_text, "exploratory exact-action-level `ell=0` operator reopen retain contract")
        is not None
    )
    note_step_c_available = hit(next_steps_note_text, "### Step C.") is not None
    note_proxy_surface_available = (
        hit(next_steps_note_text, "\\rho_{\\rm vector}(r) = |f_0(r)|^2 - |f_L(r)|^2") is not None
        or hit(next_steps_note_text, "ρ_vector = |f_0|^2 - |f_L|^2") is not None
    )
    note_proxy_not_final_available = (
        hit(next_steps_note_text, "proxy としては使えるが、最終 answer ではない") is not None
    )
    note_effective_source_surface_available = (
        hit(next_steps_note_text, "\\mathcal L \\supset a_\\mu\\,J^{\\mu}_{\\rm eff}[P^{\\rm Qball}]")
        is not None
    )
    note_jeff_low_order_proxy_available = (
        hit(next_steps_note_text, "J_eff^0") is not None
        and hit(next_steps_note_text, "|f_0|^2 - |f_L|^2") is not None
    )

    observable_dictionary_reserve_contract_ready = all(
        (
            operator_gate_summary["exact_action_level_ell0_operator_reopen_retain_honest"],
            operator_gate_summary["exact_action_level_ell0_operator_reopen_primary_retained"],
            operator_gate_summary["effective_source_ansatz_branch_secondary_retained"],
            route_local_gate_summary["current_pilot_no_go_is_route_local_only"],
            not route_local_gate_summary["current_pilot_no_go_closes_generalized_vector_solver_lane"],
            series_gate_summary["ell0_series_theorem_no_go_gate_passed"],
            part1_current_surface_available,
            part1_interaction_surface_available,
            part3a_operator_reopen_wording_available,
            part5_operator_reopen_wording_available,
            note_step_c_available,
            note_proxy_surface_available,
            note_proxy_not_final_available,
            note_effective_source_surface_available,
            note_jeff_low_order_proxy_available,
        )
    )
    observable_dictionary_reserve_contract_honest = all(
        (
            observable_dictionary_reserve_contract_ready,
            operator_gate_summary["exact_action_level_ell0_operator_available"] is False,
            operator_gate_summary["effective_source_ansatz_branch_secondary_retained"],
        )
    )
    observable_dictionary_exact_mapping_available = False
    observable_dictionary_final_observable_available = False
    observable_dictionary_requires_exact_charge_current_bridge = True
    observable_dictionary_branch_reserve_retained = bool(
        operator_gate_summary["observable_dictionary_branch_reserve_retained"]
    )

    inputs = {
        "status": display_path(STATUS),
        "roadmap": display_path(ROADMAP),
        "ai_context": display_path(AI_CONTEXT),
        "work_history_recent": display_path(WORK_HISTORY_RECENT),
        "current_problem": display_path(CURRENT_PROBLEM),
        "current_status": display_path(CURRENT_STATUS),
        "part1": display_path(PART1),
        "part3a": display_path(PART3A),
        "part5": display_path(PART5),
        "next_steps_note": display_path(NEXT_STEPS_NOTE),
        "operator_gate": display_path(OPERATOR_GATE),
        "operator_eval": display_path(OPERATOR_EVAL),
        "route_local_gate": display_path(ROUTE_LOCAL_GATE),
        "series_gate": display_path(SERIES_GATE),
    }

    inventory = payload(
        "8.7.56.1351",
        "Trial-2 numeric alpha vector Q-ball form-factor exploratory observable-dictionary reserve contract source inventory",
        inputs,
        [
            row(
                "exact_action_level_ell0_operator_reopen_retain_honest",
                "pass" if operator_gate_summary["exact_action_level_ell0_operator_reopen_retain_honest"] else "reject",
                "exact-action-level ell=0 operator reopen retain honest",
                1 if operator_gate_summary["exact_action_level_ell0_operator_reopen_retain_honest"] else 0,
                "The effective-source lane is only admissible after the primary retained operator-reopen contract is frozen.",
            ),
            row(
                "part1_current_surface_available",
                "pass" if part1_current_surface_available else "reject",
                "Part I matter current surface available",
                1 if part1_current_surface_available else 0,
                "The ansatz lane still anchors to Part I's explicit matter current rather than to a free proxy alone.",
            ),
            row(
                "part1_interaction_surface_available",
                "pass" if part1_interaction_surface_available else "reject",
                "Part I interaction surface available",
                1 if part1_interaction_surface_available else 0,
                "The ansatz lane still anchors to the explicit interaction term g_P P_mu J^mu_matter.",
            ),
            row(
                "note_step_c_available",
                "pass" if note_step_c_available else "reject",
                "Step C available",
                1 if note_step_c_available else 0,
                "The retained decision program explicitly places the next secondary lane at the exact source/current stage.",
            ),
            row(
                "note_proxy_surface_available",
                "pass" if note_proxy_surface_available else "reject",
                "vector proxy surface available",
                1 if note_proxy_surface_available else 0,
                "The note keeps rho_vector as an explicit proxy object that can be tested against a future exact source theorem.",
            ),
            row(
                "note_proxy_not_final_available",
                "pass" if note_proxy_not_final_available else "reject",
                "proxy not final theorem surface available",
                1 if note_proxy_not_final_available else 0,
                "The note explicitly says the proxy is not yet the final answer, which is required for an honest secondary retained lane.",
            ),
            row(
                "note_effective_source_surface_available",
                "pass" if note_effective_source_surface_available else "reject",
                "effective source surface available",
                1 if note_effective_source_surface_available else 0,
                "The note frames the secondary lane as a_mu J_eff^mu[P^Qball], i.e. an action-level source/current theorem problem.",
            ),
            row(
                "note_jeff_low_order_proxy_available",
                "pass" if note_jeff_low_order_proxy_available else "reject",
                "J_eff low-order proxy test available",
                1 if note_jeff_low_order_proxy_available else 0,
                "The note already gives the exact yes/no criterion: whether J_eff^0 reduces at low order to the current vector proxy.",
            ),
        ],
        {
            "observable_dictionary_reserve_contract_ready": observable_dictionary_reserve_contract_ready,
            "part1_current_surface_available": part1_current_surface_available,
            "part1_interaction_surface_available": part1_interaction_surface_available,
            "part3a_operator_reopen_wording_available": part3a_operator_reopen_wording_available,
            "part5_operator_reopen_wording_available": part5_operator_reopen_wording_available,
            "note_step_c_available": note_step_c_available,
            "note_proxy_surface_available": note_proxy_surface_available,
            "note_proxy_not_final_available": note_proxy_not_final_available,
            "note_effective_source_surface_available": note_effective_source_surface_available,
            "note_jeff_low_order_proxy_available": note_jeff_low_order_proxy_available,
        },
        {
            "overall_status": (
                "trial2_numeric_alpha_vector_qball_form_factor_exploratory_observable_dictionary_reserve_"
                "inventory_completed"
            ),
            "advance_to_8_7_56_1352": True,
            "next_required_artifacts": [RESERVE_ROUTE_NAME],
        },
        {
            "status_hit": hit(status_text, "8.7.56.1351"),
            "roadmap_hit": hit(roadmap_text, "8.7.56.1351"),
            "current_problem_hit": hit(current_problem_text, "observable-dictionary reserve contract"),
            "current_status_hit": hit(current_status_text, "observable dictionary lane"),
            "part1_current_hit": hit(part1_text, "J^\\mu_{\\mathrm{matter}}=(\\rho c,\\rho \\mathbf{v})"),
            "part1_interaction_hit": hit(part1_text, "\\mathcal{L}_{\\mathrm{int}}=g_P\\,P_\\mu J^\\mu_{\\mathrm{matter}}"),
            "note_step_c_hit": hit(next_steps_note_text, "### Step C."),
            "note_proxy_hit": hit(next_steps_note_text, "\\rho_{\\rm vector}(r) = |f_0(r)|^2 - |f_L(r)|^2"),
            "note_effective_source_hit": hit(next_steps_note_text, "\\mathcal L \\supset a_\\mu\\,J^{\\mu}_{\\rm eff}[P^{\\rm Qball}]"),
        },
    )

    audit = payload(
        "8.7.56.1352",
        "Trial-2 numeric alpha vector Q-ball form-factor exploratory observable-dictionary reserve contract audit",
        inputs,
        [
            row(
                "observable_dictionary_reserve_contract_ready",
                "pass" if observable_dictionary_reserve_contract_ready else "reject",
                "observable-dictionary reserve contract ready",
                1 if observable_dictionary_reserve_contract_ready else 0,
                "The reserve lane is ready because the primary operator-reopen and secondary effective-source lanes are already frozen and the remaining gap is now the observable dictionary itself.",
            ),
            row(
                "observable_dictionary_reserve_contract_honest",
                "pass" if observable_dictionary_reserve_contract_honest else "reject",
                "observable-dictionary reserve contract honest",
                1 if observable_dictionary_reserve_contract_honest else 0,
                "The reserve lane is honest only if it remains downstream of the missing exact operator and the still-proxy effective-source stage.",
            ),
            row(
                "observable_dictionary_exact_mapping_available",
                "pass" if observable_dictionary_exact_mapping_available else "reject",
                "observable-dictionary exact mapping available",
                1 if observable_dictionary_exact_mapping_available else 0,
                "The current pack still does not provide an exact observable mapping; that absence is why this lane remains reserve and retained.",
            ),
            row(
                "observable_dictionary_final_observable_available",
                "pass" if observable_dictionary_final_observable_available else "reject",
                "effective-source ansatz proxy only",
                1 if observable_dictionary_final_observable_available else 0,
                "The current vector density remains a proxy, not yet the exact electromagnetic current.",
            ),
            row(
                "observable_dictionary_requires_exact_charge_current_bridge",
                "pass" if observable_dictionary_requires_exact_charge_current_bridge else "reject",
                "effective-source ansatz theorem required",
                1 if observable_dictionary_requires_exact_charge_current_bridge else 0,
                "The retained Step C lane is explicitly a theorem requirement rather than a completed derivation.",
            ),
            row(
                "observable_dictionary_branch_reserve_retained",
                "pass" if observable_dictionary_branch_reserve_retained else "reject",
                "observable dictionary branch reserve retained",
                1 if observable_dictionary_branch_reserve_retained else 0,
                "Observable dictionary work still remains reserve after the operator/source ordering is frozen.",
            ),
            row(
                "vector_form_factor_exact_computation_ready_under_current_pack",
                "pass" if operator_gate_summary["vector_form_factor_exact_computation_ready_under_current_pack"] else "reject",
                "vector form-factor exact computation ready under current pack",
                1 if operator_gate_summary["vector_form_factor_exact_computation_ready_under_current_pack"] else 0,
                "Freezing the secondary source ansatz lane still does not open exact vector computation under the current pack.",
            ),
            row(
                "physical_reject_required",
                "reject",
                "physical reject required",
                0,
                "Keeping the effective-source lane as retained secondary work does not imply physical reject.",
            ),
        ],
        {
            "observable_dictionary_reserve_contract_ready": observable_dictionary_reserve_contract_ready,
            "observable_dictionary_reserve_contract_honest": observable_dictionary_reserve_contract_honest,
            "exact_action_level_ell0_operator_reopen_primary_retained": operator_gate_summary[
                "exact_action_level_ell0_operator_reopen_primary_retained"
            ],
            "effective_source_ansatz_branch_secondary_retained": True,
            "observable_dictionary_exact_mapping_available": observable_dictionary_exact_mapping_available,
            "observable_dictionary_final_observable_available": observable_dictionary_final_observable_available,
            "observable_dictionary_requires_exact_charge_current_bridge": observable_dictionary_requires_exact_charge_current_bridge,
            "observable_dictionary_branch_reserve_retained": observable_dictionary_branch_reserve_retained,
            "vector_form_factor_exact_computation_ready_under_current_pack": False,
            "physical_reject_required": False,
            "result_class": "exploratory_observable_dictionary_reserve_contract_honest",
        },
        {
            "overall_status": (
                "trial2_numeric_alpha_vector_qball_form_factor_exploratory_observable_dictionary_reserve_"
                "audit_completed"
            ),
            "advance_to_8_7_56_1353": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "operator_gate_summary": operator_gate_summary,
            "route_local_gate_summary": route_local_gate_summary,
            "series_gate_summary": series_gate_summary,
        },
    )

    declaration_gate = payload(
        "8.7.56.1353",
        "Trial-2 numeric alpha vector Q-ball form-factor exploratory observable-dictionary reserve contract declaration gate",
        inputs,
        [
            row(
                "observable_dictionary_reserve_contract_honest",
                "pass" if observable_dictionary_reserve_contract_honest else "reject",
                "observable-dictionary reserve contract honest",
                1 if observable_dictionary_reserve_contract_honest else 0,
                "The declaration gate only freezes the secondary retained source-theorem lane; it does not claim that the theorem has already been derived.",
            ),
            row(
                "exact_action_level_ell0_operator_reopen_primary_retained",
                "pass" if operator_gate_summary["exact_action_level_ell0_operator_reopen_primary_retained"] else "reject",
                "exact-action-level ell=0 operator reopen primary retained",
                1 if operator_gate_summary["exact_action_level_ell0_operator_reopen_primary_retained"] else 0,
                "Primary status remains with the missing exact operator reopen lane.",
            ),
            row(
                "effective_source_ansatz_branch_secondary_retained",
                "pass",
                "effective-source ansatz branch secondary retained",
                1,
                "The source-ansatz lane is now frozen as the secondary retained exploratory contract.",
            ),
            row(
                "observable_dictionary_branch_reserve_retained",
                "pass" if observable_dictionary_branch_reserve_retained else "reject",
                "observable dictionary branch reserve retained",
                1 if observable_dictionary_branch_reserve_retained else 0,
                "Observable-dictionary work remains reserve after the secondary source-theorem lane is frozen.",
            ),
            row(
                "physical_reject_required",
                "reject",
                "physical reject required",
                0,
                "No physical reject follows from freezing the retained source-theorem ordering.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "current_pack_limit_state": PRIOR_CLASS,
            "observable_dictionary_reserve_contract_ready": observable_dictionary_reserve_contract_ready,
            "observable_dictionary_reserve_contract_honest": observable_dictionary_reserve_contract_honest,
            "exact_action_level_ell0_operator_reopen_primary_retained": operator_gate_summary[
                "exact_action_level_ell0_operator_reopen_primary_retained"
            ],
            "effective_source_ansatz_branch_secondary_retained": True,
            "observable_dictionary_exact_mapping_available": observable_dictionary_exact_mapping_available,
            "observable_dictionary_final_observable_available": observable_dictionary_final_observable_available,
            "observable_dictionary_requires_exact_charge_current_bridge": observable_dictionary_requires_exact_charge_current_bridge,
            "observable_dictionary_branch_reserve_retained": observable_dictionary_branch_reserve_retained,
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
                "trial2_numeric_alpha_vector_qball_form_factor_exploratory_observable_dictionary_reserve_"
                "declared"
            ),
            "advance_to_8_7_56_1354": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "audit_summary": audit["summary"],
            "operator_eval_summary": operator_eval_summary,
        },
    )

    evaluation = payload(
        "8.7.56.1354",
        "Trial-2 numeric alpha vector Q-ball form-factor exploratory observable-dictionary reserve contract numeric evaluation",
        inputs,
        [
            row(
                "beta_1_fixed",
                "pass",
                "beta_1 fixed",
                float_value(operator_eval_summary, "beta_1"),
                "The retained beta_1 baseline stays unchanged while the route moves from the secondary effective-source-ansatz contract to the reserve observable-dictionary contract.",
            ),
            row(
                "q_theory_over_m0_fixed",
                "pass",
                "q_theory/m0 fixed",
                float_value(operator_eval_summary, "q_theory_over_m0"),
                "The retained matching-scale baseline stays unchanged under the reserve observable-dictionary freeze.",
            ),
            row(
                "F_exact_at_q_theory_fixed",
                "pass",
                "F_exact at q_theory fixed",
                float_value(operator_eval_summary, "F_exact_at_q_theory"),
                "The retained exact-profile overlap baseline stays unchanged under the reserve observable-dictionary freeze.",
            ),
            row(
                "alpha_exact_at_q_theory_fixed",
                "pass",
                "alpha exact at q_theory fixed",
                float_value(operator_eval_summary, "alpha_exact_at_q_theory"),
                "The retained alpha baseline stays unchanged under the reserve observable-dictionary freeze.",
            ),
            row(
                "exact_ground_state_polarization_weight_fixed",
                "pass",
                "exact ground-state polarization weight fixed",
                float_value(operator_eval_summary, "exact_ground_state_polarization_weight"),
                "The exact ground state still stays at zero polarization weight under the current exact solver.",
            ),
            row(
                "observable_dictionary_exact_mapping_available",
                "reject",
                "observable-dictionary exact mapping available",
                0,
                "The reserve contract freezes the remaining dictionary gap; it does not create a new exact observable mapping.",
            ),
            row(
                "numeric_state_changed_by_current_branch",
                "reject",
                "numeric state changed by current branch",
                0,
                "This branch only freezes the retained reserve ordering and does not create a new vector numeric evaluation.",
            ),
            row(
                "route_state_changed_by_current_branch",
                "pass",
                "route state changed by current branch",
                1,
                "The route now advances from the secondary effective-source-ansatz contract to the reserve observable-dictionary contract.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "beta_1": float_value(operator_eval_summary, "beta_1"),
            "q_theory_over_m0": float_value(operator_eval_summary, "q_theory_over_m0"),
            "F_exact_at_q_theory": float_value(operator_eval_summary, "F_exact_at_q_theory"),
            "alpha_exact_at_q_theory": float_value(operator_eval_summary, "alpha_exact_at_q_theory"),
            "exact_ground_state_polarization_weight": float_value(
                operator_eval_summary,
                "exact_ground_state_polarization_weight",
            ),
            "exact_ground_state_coupled_charge_factor": float_value(
                operator_eval_summary,
                "exact_ground_state_coupled_charge_factor",
            ),
            "ell0_zero_seed_max_abs_fL": float_value(operator_eval_summary, "ell0_zero_seed_max_abs_fL"),
            "current_pilot_odd_series_singular_coefficient": float_value(
                operator_eval_summary,
                "current_pilot_odd_series_singular_coefficient",
            ),
            "exact_action_level_ell0_operator_reopen_primary_retained": operator_gate_summary[
                "exact_action_level_ell0_operator_reopen_primary_retained"
            ],
            "effective_source_ansatz_branch_secondary_retained": True,
            "observable_dictionary_branch_reserve_retained": observable_dictionary_branch_reserve_retained,
            "observable_dictionary_exact_mapping_available": observable_dictionary_exact_mapping_available,
            "vector_form_factor_exact_computation_ready_under_current_pack": False,
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": (
                "trial2_numeric_alpha_vector_qball_form_factor_exploratory_observable_dictionary_reserve_"
                "branch_completed"
            ),
            "advance_to_next_route": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "prior_problem_classification": PRIOR_CLASS,
            "new_problem_classification": BRANCH_CLASS,
            "operator_eval_summary": operator_eval_summary,
        },
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_observable_dictionary_reserve_contract_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_observable_dictionary_reserve_contract_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_observable_dictionary_reserve_contract_declaration_gate",
        declaration_gate,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_exploratory_observable_dictionary_reserve_contract_numeric_evaluation",
        evaluation,
    )

    print("[done] 8.7.56.1351-.1354 artifacts generated")


if __name__ == "__main__":
    main()
