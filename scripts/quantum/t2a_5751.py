#!/usr/bin/env python3
"""Generate 8.7.56.5751-.5754 source-weighted comparison audit artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_beta_sensitivity_source_weighted_comparison_followup_backend import (
    build_trial2_beta_sensitivity_source_weighted_comparison_followup_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5747-5750",
        "updated_pack_trial2_beta_sensitivity_halfline_green_kernel_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
AUDIT_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "100_trial2_numeric_alpha_vector_qball_source_weighted_comparison_followup_audit.md"
)

STEP_TAG = "8.7.56.5751-5754"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "source-weighted comparison followup audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_beta_sensitivity_source_weighted_comparison_followup_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_patched_halfline_green_kernel_negative_closeout_completed_"
    "source_weighted_comparison_followup_primary_conditional_reopen_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_source_weighted_comparison_audited_"
    "pure_continuum_gate_next"
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
    """Return whether the source-weighted comparison note carries the claims."""
    patterns = (
        "source-weighted comparison",
        "positive weighted contribution",
        "negative weighted contribution",
        "pure-continuum",
    )
    return all(pattern in text for pattern in patterns)


# 関数: audit で使う式 bundle を返す。
def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the source-weighted comparison audit."""
    return {
        "kernel_split": "G_beta^(halfline)(x, xi) = G_beta,+ - G_beta,-",
        "source_split": (
            "w_beta(x) = -∫ G_beta S_beta = N_beta^(S)(x) - P_beta^(S)(x), "
            "P_beta^(S)=∫ G_beta,+ S_beta, N_beta^(S)=∫ G_beta,- S_beta"
        ),
        "comparison_rule": "w_beta(x) < 0 iff P_beta^(S)(x) > N_beta^(S)(x)",
    }


# 関数: `.5751-.5754` を実行する。
def main() -> None:
    """Execute the source-weighted comparison followup audit."""
    sign_base.require(PRIOR_GATE)
    sign_base.require(AUDIT_NOTE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    note_text = sign_base.read_text(AUDIT_NOTE)
    pack = build_trial2_beta_sensitivity_source_weighted_comparison_followup_pack()

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    note_available = note_contains_audit(note_text)
    surface_available = bool(pack["source_weighted_comparison_surface_available_now"])
    positive_dominance = bool(
        pack[
            "exact_trial2_beta_sensitivity_source_weighted_positive_dominance_control_window_now"
        ]
    )
    identity_available = bool(
        pack[
            "exact_trial2_beta_sensitivity_source_weighted_comparison_identity_available_now"
        ]
    )
    green_bvp_coherent = bool(
        pack[
            "exact_trial2_beta_sensitivity_source_weighted_comparison_green_bvp_coherent_now"
        ]
    )
    stable_now = bool(
        pack["exact_trial2_beta_sensitivity_source_weighted_comparison_stable_now"]
    )
    support_available = bool(
        pack[
            "exact_trial2_beta_sensitivity_source_weighted_comparison_support_available_now"
        ]
    )
    pure_continuum_followup_required_now = bool(
        pack[
            "updated_pack_trial2_source_weighted_comparison_pure_continuum_followup_required_now"
        ]
    )
    exact_pure_analytic_operator_level_continuum_refinement_available_now = bool(
        pack["exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now"]
    )

    rows = [
        sign_base.row(
            "updated_pack_trial2_source_weighted_comparison_followup_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 source-weighted comparison followup selected now",
            sign_base.truth(route_selected),
            "This branch starts only after the patched half-line kernel route has closed negatively and promoted source-weighted comparison as the next honest blocker.",
        ),
        sign_base.row(
            "exact_trial2_source_weighted_comparison_note_available_now",
            "pass" if note_available else "reject",
            "exact Trial-2 source-weighted comparison note available now",
            sign_base.truth(note_available),
            "The note must record the exact positive/negative source-weighted split and explain why this is not another one-sign kernel replay.",
        ),
        sign_base.row(
            "exact_trial2_source_weighted_comparison_surface_available_now",
            "pass" if surface_available else "reject",
            "exact Trial-2 source-weighted comparison surface available now",
            sign_base.truth(surface_available),
            "The half-line Green representation is already available from `.5743-.5750`, so the next route may work directly at the source-weighted comparison level.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_source_weighted_positive_dominance_control_window_now",
            "pass" if positive_dominance else "reject",
            "exact Trial-2 beta-sensitivity source-weighted positive dominance on control window now",
            sign_base.truth(positive_dominance),
            "Pass means the positive source-weighted part of the mixed-sign kernel dominates the negative part pointwise on the retained physical control window.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_source_weighted_comparison_identity_available_now",
            "pass" if identity_available else "reject",
            "exact Trial-2 beta-sensitivity source-weighted comparison identity available now",
            sign_base.truth(identity_available),
            "The split P_beta^(S)-N_beta^(S) must reconstruct the retained Green-convolution solution to numerical precision before it can carry theorem weight.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_source_weighted_comparison_green_bvp_coherent_now",
            "pass" if green_bvp_coherent else "reject",
            "exact Trial-2 beta-sensitivity source-weighted comparison Green/BVP coherent now",
            sign_base.truth(green_bvp_coherent),
            "The exact comparison identity must remain attached to the actual half-line solution rather than drifting away from the retained BVP surface.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_source_weighted_comparison_stable_now",
            "pass" if stable_now else "reject",
            "exact Trial-2 beta-sensitivity source-weighted comparison stable now",
            sign_base.truth(stable_now),
            "The retained dominance margins must stay stable across X in {80,100,140}; otherwise the route would still be a truncation accident.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_source_weighted_comparison_support_available_now",
            "pass" if support_available else "reject",
            "exact Trial-2 beta-sensitivity source-weighted comparison support available now",
            sign_base.truth(support_available),
            "This is the strongest honest result of the branch: negativity is now rewritten as one exact source-weighted comparison identity with stable positive dominance on the physical window.",
        ),
        sign_base.row(
            "updated_pack_trial2_source_weighted_comparison_pure_continuum_followup_required_now",
            "pass" if pure_continuum_followup_required_now else "reject",
            "updated-pack Trial-2 source-weighted comparison pure-continuum followup required now",
            sign_base.truth(pure_continuum_followup_required_now),
            "Once control-window comparison support is fixed, the next blocker is no longer source-weighted sign itself but its honest promotion to the full pure-continuum statement.",
        ),
        sign_base.row(
            "exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now",
            "pass" if exact_pure_analytic_operator_level_continuum_refinement_available_now else "reject",
            "exact Trial-2 pure analytic operator-level continuum refinement available now",
            sign_base.truth(exact_pure_analytic_operator_level_continuum_refinement_available_now),
            "This audit intentionally stops one layer below the full operator-level theorem; it isolates the next continuum blocker instead of overclaiming final closure.",
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
        "source_weighted_comparison_surface_available_now": surface_available,
        "exact_trial2_beta_sensitivity_source_weighted_positive_dominance_control_window_now": (
            positive_dominance
        ),
        "exact_trial2_beta_sensitivity_source_weighted_comparison_identity_available_now": (
            identity_available
        ),
        "exact_trial2_beta_sensitivity_source_weighted_comparison_green_bvp_coherent_now": (
            green_bvp_coherent
        ),
        "exact_trial2_beta_sensitivity_source_weighted_comparison_stable_now": (
            stable_now
        ),
        "exact_trial2_beta_sensitivity_source_weighted_comparison_support_available_now": (
            support_available
        ),
        "updated_pack_trial2_source_weighted_comparison_pure_continuum_followup_required_now": (
            pure_continuum_followup_required_now
        ),
        "retained_x_max": float(pack["retained_x_max"]),
        "retained_min_comparison_ratio": float(pack["retained_min_comparison_ratio"]),
        "retained_min_comparison_ratio_x": float(
            pack["retained_min_comparison_ratio_x"]
        ),
        "retained_min_comparison_relative_gap": float(
            pack["retained_min_comparison_relative_gap"]
        ),
        "retained_min_comparison_relative_gap_x": float(
            pack["retained_min_comparison_relative_gap_x"]
        ),
        "retained_min_comparison_margin": float(pack["retained_min_comparison_margin"]),
        "retained_min_comparison_margin_x": float(
            pack["retained_min_comparison_margin_x"]
        ),
        "retained_max_identity_rel_error": float(pack["retained_max_identity_rel_error"]),
        "retained_max_identity_rel_error_x": float(
            pack["retained_max_identity_rel_error_x"]
        ),
        "retained_green_bvp_control_rel_linf": float(
            pack["retained_green_bvp_control_rel_linf"]
        ),
        "comparison_ratio_rel_spread": float(pack["comparison_ratio_rel_spread"]),
        "comparison_relative_gap_rel_spread": float(
            pack["comparison_relative_gap_rel_spread"]
        ),
        "selected_next_generation_route": (
            "trial2_beta_sensitivity_source_weighted_comparison_pure_continuum_followup"
        ),
        "recommended_next_route_or_none": (
            "trial2_beta_sensitivity_source_weighted_comparison_pure_continuum_followup"
        ),
        "selected_followup_route": (
            "trial2_beta_sensitivity_source_weighted_comparison_pure_continuum_followup"
        ),
        "selected_followup_route_or_none": (
            "trial2_beta_sensitivity_source_weighted_comparison_pure_continuum_followup"
        ),
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5753",
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
            "overall_status": "trial2_source_weighted_comparison_followup_audit_completed",
            "branch_completed": True,
            "breakthrough_passed_now": support_available,
            "physical_reject_required": False,
        },
        {"route_rows": pack["route_rows"]},
    )
    outputs = write_artifact("declaration_gate", payload)
    print("[done] 8.7.56.5751-5754 Trial-2 source-weighted comparison audit completed")
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
