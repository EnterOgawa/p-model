#!/usr/bin/env python3
"""Generate 8.7.56.5743-.5746 patched half-line Green-kernel audit artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_beta_sensitivity_halfline_green_kernel_followup_backend import (
    build_trial2_beta_sensitivity_halfline_green_kernel_followup_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5739-5742",
        "updated_pack_trial2_beta_sensitivity_patched_tail_pure_continuum_closure_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
AUDIT_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "99_trial2_numeric_alpha_vector_qball_halfline_green_kernel_followup_audit.md"
)

STEP_TAG = "8.7.56.5743-5746"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "patched half-line Green-kernel followup audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_beta_sensitivity_halfline_green_kernel_followup_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_first_principles_direct_alpha_closure_completed_patched_tail_"
    "pure_continuum_closure_synced_operator_level_refinement_deferred_v3_"
    "conditional_reopen_only_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_patched_halfline_green_kernel_audited_"
    "source_weighted_comparison_gate_next"
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
    """Return whether the half-line note carries the expected claims."""
    patterns = (
        "half-line",
        "Green-kernel",
        "source-weighted comparison theorem",
        "Yukawa",
    )
    return all(pattern in text for pattern in patterns)


# 関数: audit で使う式 bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the half-line Green-kernel audit."""
    return {
        "halfline_equation": (
            "w_beta'' + V_beta(x) w_beta = -2 beta x y_beta(x), "
            "V_beta = beta^2 - 1 + 6 y_beta + 3 y_beta^2"
        ),
        "halfline_bc": "w_beta(x_min)=0, w_beta'(X) + kappa w_beta(X) = 0",
        "green_representation": (
            "w_beta(x) = - Integral_xmin^X G_beta^(halfline)(x, xi) [2 beta xi y_beta(xi)] dxi"
        ),
    }


# 関数: `.5743-.5746` を実行する。

def main() -> None:
    """Execute the patched half-line Green-kernel followup audit."""
    sign_base.require(PRIOR_GATE)
    sign_base.require(AUDIT_NOTE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    note_text = sign_base.read_text(AUDIT_NOTE)
    pack = build_trial2_beta_sensitivity_halfline_green_kernel_followup_pack()

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    note_available = note_contains_audit(note_text)
    halfline_green_kernel_available_now = bool(
        pack["exact_trial2_beta_sensitivity_halfline_green_kernel_available_now"]
    )
    halfline_green_kernel_one_sign_available_now = bool(
        pack["exact_trial2_beta_sensitivity_halfline_green_kernel_one_sign_available_now"]
    )
    halfline_source_weighted_bvp_negative_control_window_now = bool(
        pack[
            "exact_trial2_beta_sensitivity_halfline_source_weighted_bvp_negative_control_window_now"
        ]
    )
    halfline_yukawa_contraction_available_now = bool(
        pack["exact_trial2_beta_sensitivity_halfline_yukawa_contraction_available_now"]
    )
    halfline_green_kernel_negative_closeout_available_now = bool(
        pack["exact_trial2_beta_sensitivity_halfline_green_kernel_negative_closeout_available_now"]
    )
    source_weighted_comparison_followup_required_now = bool(
        pack["updated_pack_trial2_source_weighted_comparison_followup_required_now"]
    )
    exact_pure_analytic_operator_level_continuum_refinement_available_now = bool(
        pack["exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now"]
    )

    rows = [
        sign_base.row(
            "updated_pack_trial2_halfline_green_kernel_followup_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 patched half-line Green-kernel followup selected now",
            sign_base.truth(route_selected),
            "This branch reopens pure analytic refinement only after the patched-tail pure-continuum layer is already synchronized and no unconditional route remains inside the old pack.",
        ),
        sign_base.row(
            "exact_trial2_halfline_green_kernel_note_available_now",
            "pass" if note_available else "reject",
            "exact Trial-2 patched half-line Green-kernel note available now",
            sign_base.truth(note_available),
            "The note must record the half-line kernel sign loss, the source-weighted control-window negativity, and the source-weighted comparison followup requirement.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_halfline_green_kernel_available_now",
            "pass" if halfline_green_kernel_available_now else "reject",
            "exact Trial-2 beta-sensitivity half-line Green-kernel available now",
            sign_base.truth(halfline_green_kernel_available_now),
            "The admissible patched tail now supports a genuine half-line decaying-boundary operator surface distinct from the old finite Dirichlet-box route.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_halfline_green_kernel_one_sign_available_now",
            "pass" if halfline_green_kernel_one_sign_available_now else "reject",
            "exact Trial-2 beta-sensitivity half-line Green kernel one-sign available now",
            sign_base.truth(halfline_green_kernel_one_sign_available_now),
            "A positive result would have closed the operator-level refinement directly; the retained audit instead fixes mixed-sign sampled columns even on the patched half-line.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_halfline_source_weighted_bvp_negative_control_window_now",
            "pass" if halfline_source_weighted_bvp_negative_control_window_now else "reject",
            "exact Trial-2 beta-sensitivity half-line source-weighted BVP negative on control window now",
            sign_base.truth(halfline_source_weighted_bvp_negative_control_window_now),
            "The physical control window still keeps the actual source-weighted half-line solution strictly negative across all tested truncations.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_halfline_yukawa_contraction_available_now",
            "pass" if halfline_yukawa_contraction_available_now else "reject",
            "exact Trial-2 beta-sensitivity half-line Yukawa contraction available now",
            sign_base.truth(halfline_yukawa_contraction_available_now),
            "A simple positive-Yukawa Neumann contraction would have provided one easy continuum theorem; the retained contraction supremum instead stays well above one.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_halfline_green_kernel_negative_closeout_available_now",
            "pass" if halfline_green_kernel_negative_closeout_available_now else "reject",
            "exact Trial-2 beta-sensitivity half-line Green-kernel negative closeout available now",
            sign_base.truth(halfline_green_kernel_negative_closeout_available_now),
            "The honest verdict is that one-sign half-line kernel control fails, while the source-weighted control-window solution remains coherent and negative.",
        ),
        sign_base.row(
            "updated_pack_trial2_source_weighted_comparison_followup_required_now",
            "pass" if source_weighted_comparison_followup_required_now else "reject",
            "updated-pack Trial-2 source-weighted comparison followup required now",
            sign_base.truth(source_weighted_comparison_followup_required_now),
            "Once one-sign kernel and naive Yukawa contraction both fail, the next honest blocker is a source-weighted comparison theorem rather than another kernel replay.",
        ),
        sign_base.row(
            "exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now",
            "pass" if exact_pure_analytic_operator_level_continuum_refinement_available_now else "reject",
            "exact Trial-2 pure analytic operator-level continuum refinement available now",
            sign_base.truth(exact_pure_analytic_operator_level_continuum_refinement_available_now),
            "This audit intentionally does not overclaim the final theorem; it only isolates the next sharper operator-level blocker.",
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
        "tail_match_x": float(pack["tail_match_x"]),
        "retained_x_max": float(pack["retained_x_max"]),
        "exact_trial2_beta_sensitivity_halfline_green_kernel_available_now": (
            halfline_green_kernel_available_now
        ),
        "exact_trial2_beta_sensitivity_halfline_green_kernel_one_sign_available_now": (
            halfline_green_kernel_one_sign_available_now
        ),
        "exact_trial2_beta_sensitivity_halfline_source_weighted_bvp_negative_control_window_now": (
            halfline_source_weighted_bvp_negative_control_window_now
        ),
        "exact_trial2_beta_sensitivity_halfline_yukawa_contraction_available_now": (
            halfline_yukawa_contraction_available_now
        ),
        "exact_trial2_beta_sensitivity_halfline_green_kernel_negative_closeout_available_now": (
            halfline_green_kernel_negative_closeout_available_now
        ),
        "updated_pack_trial2_source_weighted_comparison_followup_required_now": (
            source_weighted_comparison_followup_required_now
        ),
        "retained_left_zero_crossing_x": float(pack["retained_left_zero_crossing_x"]),
        "retained_right_zero_crossing_x": float(pack["retained_right_zero_crossing_x"]),
        "retained_wronskian_min": float(pack["retained_wronskian_min"]),
        "retained_wronskian_max": float(pack["retained_wronskian_max"]),
        "retained_probe_075_negative_fraction": float(
            pack["retained_probe_075_negative_fraction"]
        ),
        "retained_probe_075_positive_fraction": float(
            pack["retained_probe_075_positive_fraction"]
        ),
        "retained_probe_075_kernel_min": float(pack["retained_probe_075_kernel_min"]),
        "retained_probe_075_kernel_max": float(pack["retained_probe_075_kernel_max"]),
        "retained_bvp_control_solution_min": float(
            pack["retained_bvp_control_solution_min"]
        ),
        "retained_bvp_control_solution_max": float(
            pack["retained_bvp_control_solution_max"]
        ),
        "retained_bvp_tail_positive_fraction": float(
            pack["retained_bvp_tail_positive_fraction"]
        ),
        "retained_green_control_solution_min": float(
            pack["retained_green_control_solution_min"]
        ),
        "retained_green_control_solution_max": float(
            pack["retained_green_control_solution_max"]
        ),
        "retained_green_bvp_control_rel_linf": float(
            pack["retained_green_bvp_control_rel_linf"]
        ),
        "retained_green_bvp_control_corrcoef": float(
            pack["retained_green_bvp_control_corrcoef"]
        ),
        "retained_yukawa_contraction_sup": float(pack["retained_yukawa_contraction_sup"]),
        "retained_yukawa_contraction_argmax_x": float(
            pack["retained_yukawa_contraction_argmax_x"]
        ),
        "selected_next_generation_route": (
            "trial2_beta_sensitivity_source_weighted_comparison_followup"
        ),
        "recommended_next_route_or_none": (
            "trial2_beta_sensitivity_source_weighted_comparison_followup"
        ),
        "selected_followup_route": (
            "trial2_beta_sensitivity_source_weighted_comparison_followup"
        ),
        "selected_followup_route_or_none": (
            "trial2_beta_sensitivity_source_weighted_comparison_followup"
        ),
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5745",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "audit_note": sign_base.display_path(AUDIT_NOTE),
            },
            "formulae": build_formulae(),
        },
        rows,
        summary,
        {
            "overall_status": "trial2_patched_halfline_green_kernel_followup_audit_completed",
            "branch_completed": True,
            "breakthrough_passed_now": halfline_green_kernel_negative_closeout_available_now,
            "physical_reject_required": False,
        },
        {
            "route_rows": pack["route_rows"],
        },
    )
    outputs = write_artifact("declaration_gate", payload)
    print("[done] 8.7.56.5743-5746 Trial-2 patched half-line Green-kernel audit completed")
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
