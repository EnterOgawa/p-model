#!/usr/bin/env python3
"""Generate 8.7.56.5775-.5778 source-weighted full operator-level audit artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_beta_sensitivity_source_weighted_full_operator_level_followup_backend import (
    build_trial2_beta_sensitivity_source_weighted_full_operator_level_followup_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5771-5774",
        "updated_pack_trial2_beta_sensitivity_source_weighted_operator_level_continuum_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
AUDIT_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "103_trial2_numeric_alpha_vector_qball_source_weighted_full_operator_level_followup_audit.md"
)

STEP_TAG = "8.7.56.5775-5778"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "source-weighted full operator-level followup audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_beta_sensitivity_source_weighted_full_operator_level_followup_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_source_weighted_operator_level_control_window_continuum_closure_completed_"
    "global_kernel_refinement_deferred_v3_conditional_reopen_only_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_source_weighted_full_operator_level_weighted_integral_audited_"
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


# 関数: audit note が required claims を含むか確認する。

def note_contains_audit(text: str) -> bool:
    """Return whether the full operator-level note carries the required claims."""
    patterns = (
        "weighted-integral",
        "global one-sign kernel",
        "compact complement",
        "operator-level continuum refinement",
    )
    return all(pattern in text for pattern in patterns)


# 関数: audit で使う式 bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the full operator-level audit."""
    return {
        "weighted_integral": "dI_n / d beta = n * int_0^inf y_beta^(n-1) w_beta(x) x dx",
        "full_lower_bound": (
            "control_negative_integral - compact_complement_abs_integral - analytic_tail_upper_bound"
        ),
        "closure_rule": (
            "If the full lower bound stays positive for n = 2, 3, 4, then the v2 operator-level continuum chain is complete without the stronger auxiliary global one-sign kernel theorem."
        ),
    }


# 関数: `.5775-.5778` を実行する。

def main() -> None:
    """Execute the source-weighted full operator-level audit."""
    sign_base.require(PRIOR_GATE)
    sign_base.require(AUDIT_NOTE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    note_text = sign_base.read_text(AUDIT_NOTE)
    pack = build_trial2_beta_sensitivity_source_weighted_full_operator_level_followup_pack()

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    note_available = note_contains_audit(note_text)
    control_window_theorem_available = bool(
        pack["source_weighted_operator_level_control_window_theorem_available_now"]
    )
    retained_n2_positive = bool(pack["retained_full_lower_bounds"]["2"] > 0.0)
    retained_n3_positive = bool(pack["retained_full_lower_bounds"]["3"] > 0.0)
    retained_n4_positive = bool(pack["retained_full_lower_bounds"]["4"] > 0.0)
    family_n2_positive = bool(pack["family_full_lower_bound_min_n2"] > 0.0)
    family_n3_positive = bool(pack["family_full_lower_bound_min_n3"] > 0.0)
    family_n4_positive = bool(pack["family_full_lower_bound_min_n4"] > 0.0)
    closure_available = bool(
        pack[
            "exact_trial2_source_weighted_full_halfline_weighted_integral_closure_available_now"
        ]
    )
    refinement_completed = bool(
        pack["exact_trial2_pure_analytic_operator_level_continuum_refinement_completed_now"]
    )
    gate_required_now = bool(
        pack["updated_pack_trial2_source_weighted_full_operator_level_gate_required_now"]
    )

    rows = [
        sign_base.row(
            "updated_pack_trial2_source_weighted_full_operator_level_followup_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 source-weighted full operator-level followup selected now",
            sign_base.truth(route_selected),
            "This branch starts only after the control-window operator-level theorem is already official and the only remaining wording still refers to a stronger auxiliary global-kernel refinement.",
        ),
        sign_base.row(
            "exact_trial2_source_weighted_full_operator_level_note_available_now",
            "pass" if note_available else "reject",
            "exact Trial-2 source-weighted full operator-level note available now",
            sign_base.truth(note_available),
            "The note must explain why the v2 theorem needs full weighted-integral closure rather than a stronger global one-sign kernel auxiliary theorem.",
        ),
        sign_base.row(
            "exact_trial2_source_weighted_operator_level_control_window_theorem_available_now",
            "pass" if control_window_theorem_available else "reject",
            "exact Trial-2 source-weighted operator-level control-window theorem available now",
            sign_base.truth(control_window_theorem_available),
            "The new full-halfline route only becomes honest once the previously synchronized control-window theorem remains intact.",
        ),
        sign_base.row(
            "exact_trial2_source_weighted_full_halfline_lower_bound_positive_n2_now",
            "pass" if retained_n2_positive and family_n2_positive else "reject",
            "exact Trial-2 source-weighted full-halfline lower bound positive for n=2 now",
            sign_base.truth(retained_n2_positive and family_n2_positive),
            "For n=2 the control-window negative integral still dominates both the compact complement and the analytic omitted tail on the retained theorem row and across the tested cutoff family.",
        ),
        sign_base.row(
            "exact_trial2_source_weighted_full_halfline_lower_bound_positive_n3_now",
            "pass" if retained_n3_positive and family_n3_positive else "reject",
            "exact Trial-2 source-weighted full-halfline lower bound positive for n=3 now",
            sign_base.truth(retained_n3_positive and family_n3_positive),
            "For n=3 the same domination pattern persists with an even smaller complement-to-control ratio.",
        ),
        sign_base.row(
            "exact_trial2_source_weighted_full_halfline_lower_bound_positive_n4_now",
            "pass" if retained_n4_positive and family_n4_positive else "reject",
            "exact Trial-2 source-weighted full-halfline lower bound positive for n=4 now",
            sign_base.truth(retained_n4_positive and family_n4_positive),
            "For n=4 the compact complement and analytic tail are completely subdominant to the retained control-window theorem.",
        ),
        sign_base.row(
            "exact_trial2_source_weighted_full_halfline_weighted_integral_closure_available_now",
            "pass" if closure_available else "reject",
            "exact Trial-2 source-weighted full-halfline weighted-integral closure available now",
            sign_base.truth(closure_available),
            "Pass means the full half-line weighted-integral signs are fixed directly from the exact source-weighted operator solution, compact-complement control, and analytic tail bounds.",
        ),
        sign_base.row(
            "exact_trial2_pure_analytic_operator_level_continuum_refinement_completed_now",
            "pass" if refinement_completed else "reject",
            "exact Trial-2 pure analytic operator-level continuum refinement completed now",
            sign_base.truth(refinement_completed),
            "Once the full weighted-integral theorem closes, the v2 operator-level continuum chain no longer needs to defer a stronger auxiliary global-kernel refinement.",
        ),
        sign_base.row(
            "updated_pack_trial2_source_weighted_full_operator_level_gate_required_now",
            "pass" if gate_required_now else "reject",
            "updated-pack Trial-2 source-weighted full operator-level gate required now",
            sign_base.truth(gate_required_now),
            "After the full operator-level theorem is fixed, the only honest next step is to sync the final v2 wording and remove the deferred label.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "beta_common_root": float(pack["beta_common_root"]),
        "retained_x_cutoff": float(pack["retained_x_cutoff"]),
        "retained_control_negative_integral_n2": float(
            pack["retained_control_negative_integrals"]["2"]
        ),
        "retained_control_negative_integral_n3": float(
            pack["retained_control_negative_integrals"]["3"]
        ),
        "retained_control_negative_integral_n4": float(
            pack["retained_control_negative_integrals"]["4"]
        ),
        "retained_compact_complement_abs_integral_n2": float(
            pack["retained_compact_complement_abs_integrals"]["2"]
        ),
        "retained_compact_complement_abs_integral_n3": float(
            pack["retained_compact_complement_abs_integrals"]["3"]
        ),
        "retained_compact_complement_abs_integral_n4": float(
            pack["retained_compact_complement_abs_integrals"]["4"]
        ),
        "retained_analytic_tail_upper_bound_n2": float(
            pack["retained_analytic_tail_upper_bounds"]["2"]
        ),
        "retained_analytic_tail_upper_bound_n3": float(
            pack["retained_analytic_tail_upper_bounds"]["3"]
        ),
        "retained_analytic_tail_upper_bound_n4": float(
            pack["retained_analytic_tail_upper_bounds"]["4"]
        ),
        "retained_full_lower_bound_n2": float(pack["retained_full_lower_bounds"]["2"]),
        "retained_full_lower_bound_n3": float(pack["retained_full_lower_bounds"]["3"]),
        "retained_full_lower_bound_n4": float(pack["retained_full_lower_bounds"]["4"]),
        "retained_complement_and_tail_over_control_ratio_n2": float(
            pack["retained_complement_and_tail_over_control_ratio"]["2"]
        ),
        "retained_complement_and_tail_over_control_ratio_n3": float(
            pack["retained_complement_and_tail_over_control_ratio"]["3"]
        ),
        "retained_complement_and_tail_over_control_ratio_n4": float(
            pack["retained_complement_and_tail_over_control_ratio"]["4"]
        ),
        "family_full_lower_bound_min_n2": float(pack["family_full_lower_bound_min_n2"]),
        "family_full_lower_bound_min_n3": float(pack["family_full_lower_bound_min_n3"]),
        "family_full_lower_bound_min_n4": float(pack["family_full_lower_bound_min_n4"]),
        "source_weighted_operator_level_control_window_theorem_available_now": bool(
            pack["source_weighted_operator_level_control_window_theorem_available_now"]
        ),
        "exact_trial2_source_weighted_full_halfline_weighted_integral_closure_available_now": bool(
            pack[
                "exact_trial2_source_weighted_full_halfline_weighted_integral_closure_available_now"
            ]
        ),
        "exact_trial2_pure_analytic_operator_level_continuum_refinement_completed_now": bool(
            pack["exact_trial2_pure_analytic_operator_level_continuum_refinement_completed_now"]
        ),
        "exact_trial2_pure_analytic_global_one_sign_kernel_theorem_needed_now": bool(
            pack["exact_trial2_pure_analytic_global_one_sign_kernel_theorem_needed_now"]
        ),
        "updated_pack_trial2_source_weighted_full_operator_level_gate_required_now": bool(
            pack["updated_pack_trial2_source_weighted_full_operator_level_gate_required_now"]
        ),
    }

    payload = sign_base.payload(
        "8.7.56.5777",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
        },
        rows,
        summary,
        {
            "overall_status": (
                "trial2_source_weighted_full_operator_level_audit_completed"
            ),
            "branch_completed": True,
            "breakthrough_passed_now": closure_available,
            "physical_reject_required": False,
        },
        {
            "retained_full_lower_bound_n2": float(pack["retained_full_lower_bounds"]["2"]),
            "retained_full_lower_bound_n3": float(pack["retained_full_lower_bounds"]["3"]),
            "retained_full_lower_bound_n4": float(pack["retained_full_lower_bounds"]["4"]),
            "family_full_lower_bound_min_n2": float(pack["family_full_lower_bound_min_n2"]),
            "family_full_lower_bound_min_n3": float(pack["family_full_lower_bound_min_n3"]),
            "family_full_lower_bound_min_n4": float(pack["family_full_lower_bound_min_n4"]),
        },
    )
    outputs = write_artifact("declaration_gate", payload)
    print("[done] 8.7.56.5775-5778 Trial-2 source-weighted full operator-level audit completed")
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
