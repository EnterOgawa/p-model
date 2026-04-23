#!/usr/bin/env python3
"""Generate 8.7.56.5379-.5382 scalar-proxy alpha(q) gate / route refresh artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5375-5378",
        "updated_pack_scalar_proxy_alpha_q_curve_diagnosis_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5379-5382"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor scalar-proxy alpha(q) "
    "curve gate / route refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_scalar_proxy_alpha_q_curve_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "scalar_proxy_alpha_q_curve_diagnosed_matching_scale_redrive_primary_"
    "source_materialization_secondary_gate"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "scalar_proxy_alpha_q_curve_audited_matching_scale_redrive_primary_"
    "source_materialization_secondary_next"
)


# Function: write one metrics payload as JSON and CSV.
def write_artifact(kind: str, data: dict) -> dict[str, str]:
    """Write one JSON payload and one rows CSV."""
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    paths = build_metrics_paths(PUBLIC_OUT, STEM, kind)
    paths["json"].write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with paths["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_id", "status", "metric", "value", "note"],
        )
        writer.writeheader()
        writer.writerows(data["rows"])

    return {"json": sign_base.display_path(paths["json"])}


# Function: return formulas used by the route refresh.

def build_formulae() -> dict[str, str]:
    """Return formulas used in the scalar-proxy alpha(q) gate."""
    return {
        "gate_a": "Gate A = scalar-proxy alpha(q) diagnosis available now",
        "gate_b": "Gate B = scalar-proxy matching-scale redrive promoted next",
        "gate_c": "Gate C = farther hybrid continuation reopen required now",
    }


# Function: execute `.5379-.5382`.

def main() -> None:
    """Execute the scalar-proxy alpha(q) gate / route refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary["exact_scalar_proxy_alpha_q_curve_formula_available_now"]
        and prior_summary["exact_scalar_proxy_q_exact_exists_on_retained_interval_now"]
        and prior_summary["exact_scalar_proxy_q_exact_unique_on_retained_interval_now"]
        and not prior_summary["exact_scalar_proxy_formula_failure_now"]
        and prior_summary["exact_scalar_proxy_matching_scale_primary_verdict_available_now"]
    )
    gate_b = bool(gate_a)
    gate_c = False
    retry_mode = bool(prior_summary["retry_gate_computation_mode_selected"])
    non_surrogate_guard = bool(prior_summary["failure_matrix_non_surrogate_guard_preserved"])
    computation_gate_pivot_selected_now = bool(prior_summary["computation_gate_pivot_selected_now"])
    source_materialization_numeric_rerun_demoted_to_secondary_now = bool(
        prior_summary["selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_demoted_to_secondary_now"]
    )
    pack_update_required_now = bool(gate_b)

    rows = [
        sign_base.row(
            "gate_a_updated_pack_exact_scalar_proxy_alpha_q_curve_diagnosis_available_now",
            "pass" if gate_a else "reject",
            "gate A updated-pack exact scalar-proxy alpha(q) diagnosis available now",
            sign_base.truth(gate_a),
            "The retained scalar-proxy curve now gives one honest verdict: q_exact exists, is unique, and keeps the formula alive.",
        ),
        sign_base.row(
            "gate_b_updated_pack_scalar_proxy_matching_scale_redrive_promoted_next",
            "pass" if gate_b else "reject",
            "gate B updated-pack scalar-proxy matching-scale redrive promoted next",
            sign_base.truth(gate_b),
            "The next honest blocker is now the re-derivation of the matching scale, not one more extra-q source-materialization replay.",
        ),
        sign_base.row(
            "gate_c_farther_hybrid_continuation_reopen_required_now",
            "pass" if gate_c else "reject",
            "gate C farther hybrid continuation reopen required now",
            sign_base.truth(gate_c),
            "Farther hybrid continuation stays reserve-only because the scalar-proxy computation already provides a higher-value blocker reduction.",
        ),
        sign_base.row(
            "retry_gate_computation_mode_selected",
            "pass" if retry_mode else "reject",
            "retry gate computation mode selected",
            sign_base.truth(retry_mode),
            "The route refresh keeps the computation-first policy active after repeated theory-extension replay exhausted its value.",
        ),
        sign_base.row(
            "failure_matrix_non_surrogate_guard_preserved",
            "pass" if non_surrogate_guard else "reject",
            "failure-matrix non-surrogate guard preserved",
            sign_base.truth(non_surrogate_guard),
            "The scalar-proxy pivot does not reopen exhausted surrogate or farther-hybrid retries.",
        ),
        sign_base.row(
            "computation_gate_pivot_selected_now",
            "pass" if computation_gate_pivot_selected_now else "reject",
            "computation-gate pivot selected now",
            sign_base.truth(computation_gate_pivot_selected_now),
            "The current blocker is reclassified by an actual alpha(q) computation rather than by another theorem-family inventory pass.",
        ),
        sign_base.row(
            "selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_demoted_to_secondary_now",
            "pass" if source_materialization_numeric_rerun_demoted_to_secondary_now else "reject",
            "selected-extension independent extra-q-range source-materialization numeric rerun demoted to secondary now",
            sign_base.truth(source_materialization_numeric_rerun_demoted_to_secondary_now),
            "The extra-q source-materialization lane remains available but no longer deserves primary priority once q_exact is known.",
        ),
        sign_base.row(
            "pack_update_required_now",
            "pass" if pack_update_required_now else "reject",
            "updated-pack substantive pack update required now",
            sign_base.truth(pack_update_required_now),
            "A substantive route shift happened here: the new primary lane is scalar-proxy matching-scale redrive.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "case_label": prior_summary["case_label"],
        "alpha_target": float(prior_summary["alpha_target"]),
        "beta1": float(prior_summary["beta1"]),
        "q_star_over_m0": float(prior_summary["q_star_over_m0"]),
        "q_exact_list": prior_summary["q_exact_list"],
        "primary_q_exact_over_m0": float(prior_summary["primary_q_exact_over_m0"]),
        "delta_q_over_m0": float(prior_summary["delta_q_over_m0"]),
        "delta_q_over_q_star": float(prior_summary["delta_q_over_q_star"]),
        "alpha_at_q_star": float(prior_summary["alpha_at_q_star"]),
        "relative_residual_at_q_star": float(prior_summary["relative_residual_at_q_star"]),
        "alpha_max": float(prior_summary["alpha_max"]),
        "alpha_max_over_target": float(prior_summary["alpha_max_over_target"]),
        "gate_a_updated_pack_exact_scalar_proxy_alpha_q_curve_diagnosis_available_now": gate_a,
        "gate_b_updated_pack_scalar_proxy_matching_scale_redrive_promoted_next": gate_b,
        "gate_c_farther_hybrid_continuation_reopen_required_now": gate_c,
        "retry_gate_computation_mode_selected": retry_mode,
        "failure_matrix_non_surrogate_guard_preserved": non_surrogate_guard,
        "computation_gate_pivot_selected_now": computation_gate_pivot_selected_now,
        "selected_extension_independent_extra_q_range_source_materialization_numeric_rerun_demoted_to_secondary_now": source_materialization_numeric_rerun_demoted_to_secondary_now,
        "pack_update_required_now": pack_update_required_now,
        "selected_primary_completion_lane": "updated_pack_scalar_proxy_matching_scale_redrive_audit",
        "selected_secondary_completion_lane": "updated_pack_selected_extension_independent_extra_q_range_source_materialization_numeric_rerun",
        "selected_reserve_completion_lane": "farther_hybrid_reserve_only_until_independent_need_reappears",
        "selected_next_generation_route": "trial2_numeric_alpha_scalar_proxy_matching_scale_redrive_audit",
        "recommended_next_route_or_none": "8.7.56.5383",
        "selected_followup_route": "trial2_numeric_alpha_scalar_proxy_matching_scale_redrive_gate",
        "selected_followup_route_or_none": "8.7.56.5387",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5381",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5383",
                "followup_route": "8.7.56.5387",
            },
        },
        rows,
        summary,
        {
            "overall_status": "scalar_proxy_alpha_q_curve_gate_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} scalar-proxy alpha(q) gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# Function: run the gate refresh when invoked as one CLI script.

if __name__ == "__main__":
    main()
