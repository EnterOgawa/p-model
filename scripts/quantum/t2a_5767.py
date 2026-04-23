#!/usr/bin/env python3
"""Generate 8.7.56.5767-.5770 source-weighted operator-level continuum audit artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_beta_sensitivity_source_weighted_operator_level_continuum_followup_backend import (
    build_trial2_beta_sensitivity_source_weighted_operator_level_continuum_followup_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5763-5766",
        "updated_pack_trial2_beta_sensitivity_source_weighted_comparison_pure_continuum_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
AUDIT_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "102_trial2_numeric_alpha_vector_qball_source_weighted_operator_level_continuum_followup_audit.md"
)

STEP_TAG = "8.7.56.5767-5770"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "source-weighted operator-level continuum followup audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_beta_sensitivity_source_weighted_operator_level_continuum_followup_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_source_weighted_comparison_pure_continuum_support_completed_"
    "operator_level_refinement_deferred_v3_conditional_reopen_only_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_source_weighted_operator_level_control_window_continuum_audited_"
    "gate_sync_next"
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
    """Return whether the operator-level note carries the required claims."""
    patterns = (
        "operator-level",
        "control window",
        "omitted dangerous tail",
        "global one-sign kernel theorem",
    )
    return all(pattern in text for pattern in patterns)


# 関数: audit で使う式 bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the operator-level control-window audit."""
    return {
        "comparison_identity": "w_beta(x) = N_beta^(S)(x) - P_beta^(S)(x)",
        "uniform_tail_bound": (
            "T_neg^(omitted)(x) <= C_neg,max * (1 - C_X)^(-1) * beta * A_match / kappa * exp(-kappa * (X - x_match))"
        ),
        "control_window_margin": (
            "m_inf(x) >= m_X(x) - T_neg^(omitted)(x) >= min_x m_X(x) - T_neg,max^(omitted)"
        ),
    }


# 関数: `.5767-.5770` を実行する。

def main() -> None:
    """Execute the source-weighted operator-level continuum audit."""
    sign_base.require(PRIOR_GATE)
    sign_base.require(AUDIT_NOTE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    note_text = sign_base.read_text(AUDIT_NOTE)
    pack = (
        build_trial2_beta_sensitivity_source_weighted_operator_level_continuum_followup_pack()
    )

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    note_available = note_contains_audit(note_text)
    closure_available = bool(
        pack[
            "exact_trial2_source_weighted_operator_level_control_window_continuum_closure_available_now"
        ]
    )
    global_kernel_available = bool(
        pack["exact_trial2_pure_analytic_global_one_sign_kernel_theorem_available_now"]
    )
    gate_required_now = bool(
        pack["updated_pack_trial2_source_weighted_operator_level_continuum_gate_required_now"]
    )

    rows = [
        sign_base.row(
            "updated_pack_trial2_source_weighted_operator_level_continuum_followup_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 source-weighted operator-level continuum followup selected now",
            sign_base.truth(route_selected),
            "This branch starts only after the source-weighted pure-continuum support layer is already official and the remaining gap has narrowed to one stronger operator-level control-window closure.",
        ),
        sign_base.row(
            "exact_trial2_source_weighted_operator_level_continuum_note_available_now",
            "pass" if note_available else "reject",
            "exact Trial-2 source-weighted operator-level continuum note available now",
            sign_base.truth(note_available),
            "The note must state the control-window theorem, the omitted dangerous tail bound, and the distinction from the still-open global kernel theorem.",
        ),
        sign_base.row(
            "exact_trial2_source_weighted_operator_level_family_lower_bounds_positive_now",
            "pass"
            if pack["family_comparison_margin_lower_bound_min"] > 0.0
            else "reject",
            "exact Trial-2 source-weighted operator-level family lower bounds positive now",
            sign_base.truth(pack["family_comparison_margin_lower_bound_min"] > 0.0),
            "The tested cutoff family keeps a positive comparison-margin lower bound after the omitted dangerous tail is subtracted.",
        ),
        sign_base.row(
            "exact_trial2_source_weighted_operator_level_retained_lower_bound_positive_now",
            "pass"
            if pack["retained_comparison_margin_lower_bound"] > 0.0
            else "reject",
            "exact Trial-2 source-weighted operator-level retained lower bound positive now",
            sign_base.truth(pack["retained_comparison_margin_lower_bound"] > 0.0),
            "At the retained theorem cutoff X = 140, the omitted dangerous tail remains strictly below the retained comparison margin.",
        ),
        sign_base.row(
            "exact_trial2_source_weighted_operator_level_control_window_closure_available_now",
            "pass" if closure_available else "reject",
            "exact Trial-2 source-weighted operator-level control-window closure available now",
            sign_base.truth(closure_available),
            "Pass means the exact source-weighted comparison identity and the explicit omitted-tail bound combine into one honest operator-level control-window continuum theorem.",
        ),
        sign_base.row(
            "exact_trial2_pure_analytic_global_one_sign_kernel_theorem_available_now",
            "pass" if global_kernel_available else "reject",
            "exact Trial-2 pure analytic global one-sign kernel theorem available now",
            sign_base.truth(global_kernel_available),
            "This branch intentionally stops below the global one-sign kernel theorem and keeps that stronger refinement out of the v2 wording.",
        ),
        sign_base.row(
            "updated_pack_trial2_source_weighted_operator_level_continuum_gate_required_now",
            "pass" if gate_required_now else "reject",
            "updated-pack Trial-2 source-weighted operator-level continuum gate required now",
            sign_base.truth(gate_required_now),
            "Once the control-window operator-level continuum theorem is fixed, the next honest task is to sync the wording and return to conditional reopen only.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "beta_common_root": float(pack["beta_common_root"]),
        "alpha_common_value": float(prior_summary["alpha_common_value"]),
        "alpha_common_rel_error_vs_target": float(
            prior_summary["alpha_common_rel_error_vs_target"]
        ),
        "retained_x_cutoff": float(pack["retained_x_cutoff"]),
        "retained_min_comparison_margin": float(pack["retained_min_comparison_margin"]),
        "retained_negative_control_coeff_max": float(
            pack["retained_negative_control_coeff_max"]
        ),
        "retained_tail_contraction_upper_bound": float(
            pack["retained_tail_contraction_upper_bound"]
        ),
        "retained_source_tail_integral_upper_bound": float(
            pack["retained_source_tail_integral_upper_bound"]
        ),
        "retained_omitted_negative_tail_upper_bound": float(
            pack["retained_omitted_negative_tail_upper_bound"]
        ),
        "retained_comparison_margin_lower_bound": float(
            pack["retained_comparison_margin_lower_bound"]
        ),
        "retained_omitted_negative_tail_over_retained_margin": float(
            pack["retained_omitted_negative_tail_over_retained_margin"]
        ),
        "family_comparison_margin_lower_bound_min": float(
            pack["family_comparison_margin_lower_bound_min"]
        ),
        "family_comparison_margin_lower_bound_max": float(
            pack["family_comparison_margin_lower_bound_max"]
        ),
        "family_omitted_negative_tail_ratio_max": float(
            pack["family_omitted_negative_tail_ratio_max"]
        ),
        "source_weighted_operator_level_control_window_continuum_support_available_now": bool(
            pack["source_weighted_operator_level_control_window_continuum_support_available_now"]
        ),
        "exact_trial2_source_weighted_operator_level_control_window_continuum_closure_available_now": bool(
            pack[
                "exact_trial2_source_weighted_operator_level_control_window_continuum_closure_available_now"
            ]
        ),
        "exact_trial2_pure_analytic_global_one_sign_kernel_theorem_available_now": bool(
            pack["exact_trial2_pure_analytic_global_one_sign_kernel_theorem_available_now"]
        ),
        "updated_pack_trial2_source_weighted_operator_level_continuum_gate_required_now": bool(
            pack[
                "updated_pack_trial2_source_weighted_operator_level_continuum_gate_required_now"
            ]
        ),
    }

    payload = sign_base.payload(
        "8.7.56.5769",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
        },
        rows,
        summary,
        {
            "overall_status": (
                "trial2_source_weighted_operator_level_continuum_audit_completed"
            ),
            "branch_completed": True,
            "breakthrough_passed_now": closure_available,
            "physical_reject_required": False,
        },
        {
            "retained_x_cutoff": float(pack["retained_x_cutoff"]),
            "retained_omitted_negative_tail_upper_bound": float(
                pack["retained_omitted_negative_tail_upper_bound"]
            ),
            "retained_comparison_margin_lower_bound": float(
                pack["retained_comparison_margin_lower_bound"]
            ),
            "family_comparison_margin_lower_bound_min": float(
                pack["family_comparison_margin_lower_bound_min"]
            ),
        },
    )
    outputs = write_artifact("declaration_gate", payload)
    print("[done] 8.7.56.5767-5770 Trial-2 source-weighted operator-level continuum audit completed")
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
