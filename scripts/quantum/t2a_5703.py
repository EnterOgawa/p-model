#!/usr/bin/env python3
"""Generate 8.7.56.5703-.5706 Trial-2 final closure followup audit artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_beta_sensitivity_final_closure_followup_backend import (
    build_trial2_beta_sensitivity_final_closure_followup_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5699-5702",
        "updated_pack_trial2_beta_sensitivity_uniqueness_anchor_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
AUDIT_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "94_trial2_numeric_alpha_vector_qball_beta_sensitivity_final_closure_followup_audit.md"
)

STEP_TAG = "8.7.56.5703-5706"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "beta-sensitivity final closure followup audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_beta_sensitivity_final_closure_followup_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_beta_sensitivity_uniqueness_anchor_sign_support_completed_"
    "final_closure_followup_primary_conditional_hold_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_beta_sensitivity_final_closure_audited_"
    "first_principles_direct_alpha_gate_next"
)


# 関数: JSON/CSV artifact を書き出す。
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


# 関数: audit note が expected claims を含むか確認する。

def note_contains_audit(text: str) -> bool:
    """Return whether the final-closure note carries the expected claims."""
    patterns = (
        "first-principles direct-alpha closure",
        "pure analytic operator-level continuum refinement",
        "Delta_common",
        "beta_*",
    )
    return all(pattern in text for pattern in patterns)


# 関数: audit で使う式 bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the final-closure audit."""
    return {
        "delta_common": "Delta_common(beta) = alpha_qstar(beta) - R8(beta)",
        "selector": "select beta_* from Delta_common(beta_*) = 0",
        "alpha_star": "alpha_* = alpha_qstar(beta_*) = R8(beta_*)",
    }


# 関数: `.5703-.5706` を実行する。

def main() -> None:
    """Execute the Trial-2 final closure followup audit."""
    sign_base.require(PRIOR_GATE)
    sign_base.require(AUDIT_NOTE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    note_text = sign_base.read_text(AUDIT_NOTE)
    pack = build_trial2_beta_sensitivity_final_closure_followup_pack()

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    note_available = note_contains_audit(note_text)

    rows = [
        sign_base.row(
            "updated_pack_trial2_beta_sensitivity_final_closure_followup_route_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 beta-sensitivity final closure followup route selected now",
            sign_base.truth(route_selected),
            "The followup starts only from the synchronized uniqueness-anchor support state where the remaining blocker is the final closure verdict itself.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_final_closure_followup_note_available_now",
            "pass" if note_available else "reject",
            "exact Trial-2 beta-sensitivity final closure followup note available now",
            sign_base.truth(note_available),
            "The note must explicitly record first-principles direct-alpha closure and pure analytic continuum refinement as distinct layers.",
        ),
        sign_base.row(
            "exact_trial2_target_free_common_root_selector_available_now",
            "pass"
            if pack["target_free_common_root_selector_available_now"]
            else "reject",
            "exact Trial-2 target-free common-root selector available now",
            sign_base.truth(pack["target_free_common_root_selector_available_now"]),
            "The final verdict only makes sense once beta is selected by equality of two independent frozen-action readouts.",
        ),
        sign_base.row(
            "exact_trial2_practical_direct_alpha_closeout_available_now",
            "pass"
            if pack["practical_direct_alpha_closeout_available_now"]
            else "reject",
            "exact Trial-2 practical direct-alpha closeout available now",
            sign_base.truth(pack["practical_direct_alpha_closeout_available_now"]),
            "The closure statement requires the retained alpha_common readout to already exist on the target-free selector.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_discrete_pointwise_dominance_theorem_available_now",
            "pass"
            if pack["discrete_pointwise_dominance_theorem_available_now"]
            else "reject",
            "exact Trial-2 beta-sensitivity discrete pointwise-dominance theorem available now",
            sign_base.truth(pack["discrete_pointwise_dominance_theorem_available_now"]),
            "The closure chain requires the already-synchronized discrete negativity theorem from spectral projection.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_continuum_open_interval_support_available_now",
            "pass"
            if pack["continuum_open_interval_support_available_now"]
            else "reject",
            "exact Trial-2 beta-sensitivity continuum open-interval support available now",
            sign_base.truth(pack["continuum_open_interval_support_available_now"]),
            "The closure chain requires interior continuum stability, with the boundary-layer collapse treated as a box artifact rather than a physics sign loss.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_weighted_integral_sign_support_available_now",
            "pass"
            if pack["weighted_integral_sign_support_available_now"]
            else "reject",
            "exact Trial-2 beta-sensitivity weighted-integral sign support available now",
            sign_base.truth(pack["weighted_integral_sign_support_available_now"]),
            "The closure chain requires the weighted integral signs that feed the derivative chain and make the boundary complement non-blocking.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_derivative_chain_sign_support_available_now",
            "pass"
            if pack["derivative_chain_sign_support_available_now"]
            else "reject",
            "exact Trial-2 beta-sensitivity derivative-chain sign support available now",
            sign_base.truth(pack["derivative_chain_sign_support_available_now"]),
            "The closure chain requires opposite monotonicity of alpha_qstar(beta) and R8(beta), hence positive Delta_common'(beta).",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_uniqueness_anchor_support_available_now",
            "pass"
            if pack["uniqueness_anchor_support_available_now"]
            else "reject",
            "exact Trial-2 beta-sensitivity uniqueness-anchor support available now",
            sign_base.truth(pack["uniqueness_anchor_support_available_now"]),
            "The closure chain requires lower / upper sign anchors, the retained common root, sampled non-ambiguity, and local transversality on one synchronized surface.",
        ),
        sign_base.row(
            "exact_trial2_first_principles_direct_alpha_closure_completed_now",
            "pass"
            if pack["exact_trial2_first_principles_direct_alpha_closure_completed_now"]
            else "reject",
            "exact Trial-2 first-principles direct-alpha closure completed now",
            sign_base.truth(
                pack["exact_trial2_first_principles_direct_alpha_closure_completed_now"]
            ),
            "Pass means the retained frozen-action chain is now strong enough to treat alpha_* = alpha_qstar(beta_*) = R8(beta_*) as completed at the first-principles level.",
        ),
        sign_base.row(
            "exact_trial2_pure_analytic_operator_level_continuum_refinement_deferred_to_v3_now",
            "pass"
            if pack[
                "exact_trial2_pure_analytic_operator_level_continuum_refinement_deferred_to_v3_now"
            ]
            else "reject",
            "exact Trial-2 pure analytic operator-level continuum refinement deferred to v3 now",
            sign_base.truth(
                pack[
                    "exact_trial2_pure_analytic_operator_level_continuum_refinement_deferred_to_v3_now"
                ]
            ),
            "The remaining operator-level continuum sharpening is treated as mathematical refinement rather than a blocker for the direct-alpha closure claim.",
        ),
        sign_base.row(
            "updated_pack_trial2_final_closure_gate_required_now",
            "pass"
            if pack["updated_pack_trial2_final_closure_gate_required_now"]
            else "reject",
            "updated-pack Trial-2 final closure gate required now",
            sign_base.truth(pack["updated_pack_trial2_final_closure_gate_required_now"]),
            "Once the closure verdict is positive, the next honest step is the final official declaration gate rather than another support-level replay.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "beta_common_root": float(pack["beta_common_root"]),
        "alpha_common_value": float(pack["alpha_common_value"]),
        "alpha_common_rel_error_vs_target": float(
            pack["alpha_common_rel_error_vs_target"]
        ),
        "delta_common_lower_anchor": float(pack["delta_common_lower_anchor"]),
        "delta_common_upper_anchor": float(pack["delta_common_upper_anchor"]),
        "derivative_transversality_min": float(
            pack["derivative_transversality_min"]
        ),
        "derivative_transversality_max": float(
            pack["derivative_transversality_max"]
        ),
        "boundary_complement_abs_fraction_max_n2": float(
            pack["boundary_complement_abs_fraction_max_n2"]
        ),
        "boundary_complement_abs_fraction_max_n3": float(
            pack["boundary_complement_abs_fraction_max_n3"]
        ),
        "boundary_complement_abs_fraction_max_n4": float(
            pack["boundary_complement_abs_fraction_max_n4"]
        ),
        "continuum_smallest_window_margin": float(
            pack["continuum_smallest_window_margin"]
        ),
        "continuum_largest_window_margin": float(
            pack["continuum_largest_window_margin"]
        ),
        "exact_trial2_first_principles_direct_alpha_closure_completed_now": bool(
            pack["exact_trial2_first_principles_direct_alpha_closure_completed_now"]
        ),
        "exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now": bool(
            pack["exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now"]
        ),
        "exact_trial2_pure_analytic_operator_level_continuum_refinement_deferred_to_v3_now": bool(
            pack[
                "exact_trial2_pure_analytic_operator_level_continuum_refinement_deferred_to_v3_now"
            ]
        ),
        "updated_pack_trial2_final_closure_gate_required_now": bool(
            pack["updated_pack_trial2_final_closure_gate_required_now"]
        ),
    }

    payload = sign_base.payload(
        "8.7.56.5705",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "audit_note": sign_base.display_path(AUDIT_NOTE),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5707",
                "followup_route": "trial2_beta_sensitivity_final_closure_gate",
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_beta_sensitivity_final_closure_followup_audited",
            "branch_completed": True,
            "breakthrough_passed_now": bool(
                pack["exact_trial2_first_principles_direct_alpha_closure_completed_now"]
            ),
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 final closure followup audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
