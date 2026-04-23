#!/usr/bin/env python3
"""Audit the final closure verdict for the Trial-2 direct-alpha theorem route.

Purpose:
    The current pack already carries the following synchronized surfaces:

    1. one target-free common-root selector for beta,
    2. one practical direct-alpha closeout alpha_common,
    3. one discrete pointwise-dominance theorem,
    4. one continuum open-interval support layer,
    5. one weighted-integral sign-support layer,
    6. one explicit derivative-chain sign-support layer,
    7. one uniqueness-anchor support layer.

    This backend does not replay heavy solves. It reads those already-fixed
    public artifacts and decides whether they are now strong enough to support
    the final official verdict:

        first-principles direct-alpha closure is complete,
        pure analytic operator-level continuum refinement is deferred to v3.0.

Inputs:
    - The synchronized public declaration artifacts from `.5623-.5626`,
      `.5663-.5666`, `.5671-.5674`, `.5679-.5682`, `.5687-.5690`,
      and `.5695-.5698`

Outputs:
    - One in-memory final-closure pack consumed by `.5703-.5710`

Assumptions:
    - No new parameter is introduced
    - No new heavy replay is performed
    - The closure statement is first-principles / target-free
    - Pure analytic operator-level continuum refinement may remain open
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
COMMON_ROOT_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5623-5626",
        "updated_pack_trial2_interaction_total_over_harmonic_sq_beta_root_followup_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
DISCRETE_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5663-5666",
        "updated_pack_trial2_beta_sensitivity_spectral_projection_followup_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
CONTINUUM_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5671-5674",
        "updated_pack_trial2_beta_sensitivity_continuum_spectral_projection_followup_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
WEIGHTED_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5679-5682",
        "updated_pack_trial2_beta_sensitivity_operator_level_spectral_projection_followup_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
DERIVATIVE_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5687-5690",
        "updated_pack_trial2_beta_sensitivity_derivative_chain_followup_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
UNIQUENESS_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5695-5698",
        "updated_pack_trial2_beta_sensitivity_uniqueness_anchor_followup_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]


# 関数: declaration artifact の summary object を返す。
def read_summary(path: Path) -> dict:
    """Return the summary object from one synchronized declaration artifact."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload["summary"])


# 関数: final closure verdict pack を返す。

def build_trial2_beta_sensitivity_final_closure_followup_pack() -> dict:
    """Return the final closure verdict pack for the Trial-2 theorem route."""
    common_root = read_summary(COMMON_ROOT_AUDIT)
    discrete = read_summary(DISCRETE_AUDIT)
    continuum = read_summary(CONTINUUM_AUDIT)
    weighted = read_summary(WEIGHTED_AUDIT)
    derivative = read_summary(DERIVATIVE_AUDIT)
    uniqueness = read_summary(UNIQUENESS_AUDIT)

    target_free_common_root_selector_available_now = bool(
        common_root[
            "exact_trial2_interaction_total_over_harmonic_sq_target_free_beta_selector_available_now"
        ]
    )
    practical_direct_alpha_closeout_available_now = bool(
        common_root[
            "exact_trial2_interaction_total_over_harmonic_sq_practical_direct_alpha_closeout_available_now"
        ]
    )
    discrete_pointwise_dominance_theorem_available_now = bool(
        discrete["exact_trial2_beta_sensitivity_discrete_negativity_theorem_available_now"]
    )
    continuum_open_interval_support_available_now = bool(
        continuum["exact_trial2_beta_sensitivity_continuum_open_interval_support_available_now"]
    )
    weighted_integral_sign_support_available_now = bool(
        weighted["exact_trial2_beta_sensitivity_weighted_integral_sign_support_available_now"]
    )
    derivative_chain_sign_support_available_now = bool(
        derivative[
            "exact_trial2_delta_common_derivative_chain_positive_local_support_available_now"
        ]
    )
    uniqueness_anchor_support_available_now = bool(
        uniqueness["exact_trial2_beta_sensitivity_uniqueness_anchor_support_available_now"]
    )

    # 物理 closeout の判定は、既に同期済みの support theorem layers を束ねて行う。
    first_principles_direct_alpha_closure_completed_now = bool(
        target_free_common_root_selector_available_now
        and practical_direct_alpha_closeout_available_now
        and discrete_pointwise_dominance_theorem_available_now
        and continuum_open_interval_support_available_now
        and weighted_integral_sign_support_available_now
        and derivative_chain_sign_support_available_now
        and uniqueness_anchor_support_available_now
    )
    pure_analytic_operator_level_continuum_refinement_available_now = False
    pure_analytic_operator_level_continuum_refinement_deferred_to_v3_now = bool(
        first_principles_direct_alpha_closure_completed_now
        and not pure_analytic_operator_level_continuum_refinement_available_now
    )
    updated_pack_trial2_final_closure_gate_required_now = bool(
        first_principles_direct_alpha_closure_completed_now
    )

    return {
        "beta_common_root": float(
            common_root["interaction_total_over_harmonic_sq_beta_common_root"]
        ),
        "alpha_common_value": float(
            common_root["interaction_total_over_harmonic_sq_alpha_common_value"]
        ),
        "alpha_common_rel_error_vs_target": float(
            common_root["interaction_total_over_harmonic_sq_alpha_common_rel_error_vs_target"]
        ),
        "delta_common_lower_anchor": float(uniqueness["delta_common_lower_anchor"]),
        "delta_common_upper_anchor": float(uniqueness["delta_common_upper_anchor"]),
        "derivative_transversality_min": float(
            uniqueness["derivative_transversality_min"]
        ),
        "derivative_transversality_max": float(
            uniqueness["derivative_transversality_max"]
        ),
        "boundary_complement_abs_fraction_max_n2": float(
            weighted["boundary_complement_abs_fraction_max_n2"]
        ),
        "boundary_complement_abs_fraction_max_n3": float(
            weighted["boundary_complement_abs_fraction_max_n3"]
        ),
        "boundary_complement_abs_fraction_max_n4": float(
            weighted["boundary_complement_abs_fraction_max_n4"]
        ),
        "continuum_smallest_window_margin": float(
            continuum["smallest_interior_window_continuum_margin_estimate"]
        ),
        "continuum_largest_window_margin": float(
            continuum["largest_interior_window_continuum_margin_estimate"]
        ),
        "target_free_common_root_selector_available_now": (
            target_free_common_root_selector_available_now
        ),
        "practical_direct_alpha_closeout_available_now": (
            practical_direct_alpha_closeout_available_now
        ),
        "discrete_pointwise_dominance_theorem_available_now": (
            discrete_pointwise_dominance_theorem_available_now
        ),
        "continuum_open_interval_support_available_now": (
            continuum_open_interval_support_available_now
        ),
        "weighted_integral_sign_support_available_now": (
            weighted_integral_sign_support_available_now
        ),
        "derivative_chain_sign_support_available_now": (
            derivative_chain_sign_support_available_now
        ),
        "uniqueness_anchor_support_available_now": (
            uniqueness_anchor_support_available_now
        ),
        "exact_trial2_first_principles_direct_alpha_closure_completed_now": (
            first_principles_direct_alpha_closure_completed_now
        ),
        "exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now": (
            pure_analytic_operator_level_continuum_refinement_available_now
        ),
        "exact_trial2_pure_analytic_operator_level_continuum_refinement_deferred_to_v3_now": (
            pure_analytic_operator_level_continuum_refinement_deferred_to_v3_now
        ),
        "updated_pack_trial2_final_closure_gate_required_now": (
            updated_pack_trial2_final_closure_gate_required_now
        ),
    }


# 関数: backend 単体実行時に retained metrics を表示する。

def main() -> None:
    """Run the final-closure backend directly and print retained verdicts."""
    pack = build_trial2_beta_sensitivity_final_closure_followup_pack()
    print("[trial2-beta-final-closure-followup]")
    print(f"beta_common_root = {pack['beta_common_root']:.16f}")
    print(f"alpha_common_value = {pack['alpha_common_value']:.16f}")
    print(
        "first_principles_direct_alpha_closure_completed = "
        f"{pack['exact_trial2_first_principles_direct_alpha_closure_completed_now']}"
    )
    print(
        "pure_analytic_operator_level_continuum_refinement_deferred_to_v3 = "
        f"{pack['exact_trial2_pure_analytic_operator_level_continuum_refinement_deferred_to_v3_now']}"
    )


if __name__ == "__main__":
    main()
