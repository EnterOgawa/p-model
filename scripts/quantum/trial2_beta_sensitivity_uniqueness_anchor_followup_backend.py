#!/usr/bin/env python3
"""Audit uniqueness-anchor support for the Trial-2 common-root selector.

Purpose:
    The derivative-chain followup already fixed one local transversality layer
    around the retained common root:

        d alpha_qstar / d beta > 0
        d R8 / d beta < 0
        d Delta_common / d beta > 0

    The remaining blocker is narrower: can the retained lower / upper sign
    anchors and the already synchronized common-root scan be bundled into one
    uniqueness-anchor support surface?

    This backend stays honest about the theorem level. It does not replay the
    heavy scalar solves. Instead, it reads the already synchronized public
    artifacts from the target-free common-root audit and the derivative-chain
    audit, then tests whether they jointly supply:

    1. one lower negative anchor,
    2. one upper positive anchor,
    3. one retained common root between those anchors,
    4. one sampled non-ambiguous selector on the localized family,
    5. one local positive transversality support layer.

    The output is therefore a support theorem surface, not yet the final
    strict theorem itself.

Inputs:
    - output/public/quantum/q_8_7_56_5623_5626_..._declaration_gate_metrics.json
    - output/public/quantum/q_8_7_56_5687_5690_..._declaration_gate_metrics.json

Outputs:
    - One in-memory audit pack consumed by `.5695-.5702` wrappers

Assumptions:
    - No new parameter is introduced
    - No new heavy replay is performed
    - The final strict theorem remains a separate declaration layer
"""

from __future__ import annotations

import json
import math
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
DERIVATIVE_CHAIN_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5687-5690",
        "updated_pack_trial2_beta_sensitivity_derivative_chain_followup_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]


# 関数: declaration artifact の summary section を返す。
def read_summary(path: Path) -> dict:
    """Return the summary object stored in one synchronized declaration artifact."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload["summary"])


# 関数: uniqueness-anchor support pack を返す。

def build_trial2_beta_sensitivity_uniqueness_anchor_followup_pack() -> dict:
    """Return one uniqueness-anchor support pack for the strict-theorem route."""
    common_root_summary = read_summary(COMMON_ROOT_AUDIT)
    derivative_summary = read_summary(DERIVATIVE_CHAIN_AUDIT)

    beta_lower = float(common_root_summary["common_root_scan_beta_min"])
    beta_upper = float(common_root_summary["common_root_scan_beta_max"])
    delta_lower = float(common_root_summary["common_root_difference_first"])
    delta_upper = float(common_root_summary["common_root_difference_last"])
    beta_common_root = float(
        common_root_summary["interaction_total_over_harmonic_sq_beta_common_root"]
    )
    alpha_common_value = float(
        common_root_summary["interaction_total_over_harmonic_sq_alpha_common_value"]
    )
    alpha_common_rel_error_vs_target = float(
        common_root_summary["interaction_total_over_harmonic_sq_alpha_common_rel_error_vs_target"]
    )

    lower_anchor_negative_now = bool(delta_lower < 0.0)
    upper_anchor_positive_now = bool(delta_upper > 0.0)
    common_root_inside_anchor_interval_now = bool(
        beta_lower < beta_common_root < beta_upper
    )
    sampled_selector_monotone_now = bool(
        common_root_summary["common_root_difference_monotone_increasing_now"]
    )
    sampled_selector_single_sign_change_now = bool(
        int(common_root_summary["common_root_difference_sign_change_count"]) == 1
    )
    local_delta_derivative_positive_now = bool(
        derivative_summary[
            "exact_trial2_delta_common_derivative_chain_positive_local_support_available_now"
        ]
        and float(derivative_summary["delta_common_derivative_min"]) > 0.0
    )
    sampled_anchor_gap_span = float(delta_upper - delta_lower)
    lower_anchor_abs_margin = float(abs(delta_lower))
    upper_anchor_abs_margin = float(abs(delta_upper))
    derivative_transversality_min = float(
        derivative_summary["delta_common_derivative_min"]
    )
    derivative_transversality_max = float(
        derivative_summary["delta_common_derivative_max"]
    )

    uniqueness_anchor_support_available_now = bool(
        lower_anchor_negative_now
        and upper_anchor_positive_now
        and common_root_inside_anchor_interval_now
        and sampled_selector_monotone_now
        and sampled_selector_single_sign_change_now
        and local_delta_derivative_positive_now
    )
    exact_trial2_beta_sensitivity_uniqueness_anchor_theorem_available_now = False
    updated_pack_trial2_beta_sensitivity_final_closure_followup_required_now = bool(
        uniqueness_anchor_support_available_now
        and not exact_trial2_beta_sensitivity_uniqueness_anchor_theorem_available_now
    )

    return {
        "beta_anchor_lower": beta_lower,
        "beta_anchor_upper": beta_upper,
        "delta_common_lower_anchor": delta_lower,
        "delta_common_upper_anchor": delta_upper,
        "beta_common_root": beta_common_root,
        "alpha_common_value": alpha_common_value,
        "alpha_common_rel_error_vs_target": alpha_common_rel_error_vs_target,
        "lower_anchor_negative_now": lower_anchor_negative_now,
        "upper_anchor_positive_now": upper_anchor_positive_now,
        "common_root_inside_anchor_interval_now": (
            common_root_inside_anchor_interval_now
        ),
        "sampled_selector_monotone_now": sampled_selector_monotone_now,
        "sampled_selector_single_sign_change_now": (
            sampled_selector_single_sign_change_now
        ),
        "local_delta_derivative_positive_now": local_delta_derivative_positive_now,
        "sampled_anchor_gap_span": sampled_anchor_gap_span,
        "lower_anchor_abs_margin": lower_anchor_abs_margin,
        "upper_anchor_abs_margin": upper_anchor_abs_margin,
        "derivative_transversality_min": derivative_transversality_min,
        "derivative_transversality_max": derivative_transversality_max,
        "uniqueness_anchor_support_available_now": (
            uniqueness_anchor_support_available_now
        ),
        "exact_trial2_beta_sensitivity_uniqueness_anchor_theorem_available_now": (
            exact_trial2_beta_sensitivity_uniqueness_anchor_theorem_available_now
        ),
        "updated_pack_trial2_beta_sensitivity_final_closure_followup_required_now": (
            updated_pack_trial2_beta_sensitivity_final_closure_followup_required_now
        ),
    }


# 関数: backend 単体実行時に retained metrics を表示する。

def main() -> None:
    """Run the uniqueness-anchor backend directly and print key retained values."""
    pack = build_trial2_beta_sensitivity_uniqueness_anchor_followup_pack()
    print("[trial2-beta-uniqueness-anchor-followup]")
    print(f"beta_anchor_lower = {pack['beta_anchor_lower']:.16f}")
    print(f"beta_anchor_upper = {pack['beta_anchor_upper']:.16f}")
    print(f"delta_common_lower_anchor = {pack['delta_common_lower_anchor']:.16f}")
    print(f"delta_common_upper_anchor = {pack['delta_common_upper_anchor']:.16f}")
    print(f"beta_common_root = {pack['beta_common_root']:.16f}")
    print(
        "uniqueness_anchor_support_available = "
        f"{pack['uniqueness_anchor_support_available_now']}"
    )


if __name__ == "__main__":
    main()
