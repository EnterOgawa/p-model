#!/usr/bin/env python3
"""Generate 8.7.56.5759-.5762 source-weighted pure-continuum audit artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_beta_sensitivity_source_weighted_comparison_pure_continuum_followup_backend import (
    build_trial2_beta_sensitivity_source_weighted_comparison_pure_continuum_followup_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5755-5758",
        "updated_pack_trial2_beta_sensitivity_source_weighted_comparison_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
AUDIT_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "101_trial2_numeric_alpha_vector_qball_source_weighted_pure_continuum_followup_audit.md"
)

STEP_TAG = "8.7.56.5759-5762"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "source-weighted comparison pure-continuum followup audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_beta_sensitivity_source_weighted_comparison_pure_continuum_followup_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_source_weighted_comparison_sign_support_completed_"
    "pure_continuum_followup_primary_conditional_reopen_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_source_weighted_comparison_pure_continuum_audited_"
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
    """Return whether the pure-continuum note carries the required claims."""
    patterns = (
        "source-weighted comparison",
        "pure-continuum",
        "tail contraction",
        "omitted negative tail",
    )
    return all(pattern in text for pattern in patterns)


# 関数: audit で使う式 bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the pure-continuum comparison audit."""
    return {
        "tail_profile": (
            "y_tail(x) = Y_match * x_match / x * exp(-kappa * (x - x_match))"
        ),
        "tail_source": (
            "S_tail(x) = 2 * beta * x_match * Y_match * exp(-kappa * (x - x_match))"
        ),
        "dangerous_tail_bound": (
            "T_neg^(omitted) <= C_neg,max * (1 - C_X)^(-1) * beta * A_match / kappa * exp(-kappa * (X - x_match))"
        ),
    }


# 関数: `.5759-.5762` を実行する。

def main() -> None:
    """Execute the source-weighted comparison pure-continuum audit."""
    sign_base.require(PRIOR_GATE)
    sign_base.require(AUDIT_NOTE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    note_text = sign_base.read_text(AUDIT_NOTE)
    pack = (
        build_trial2_beta_sensitivity_source_weighted_comparison_pure_continuum_followup_pack()
    )

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    note_available = note_contains_audit(note_text)
    pure_continuum_support_available = bool(
        pack["source_weighted_comparison_pure_continuum_support_available_now"]
    )
    operator_level_available = bool(
        pack["exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now"]
    )
    gate_required_now = bool(
        pack["updated_pack_trial2_source_weighted_comparison_pure_continuum_gate_required_now"]
    )

    rows = [
        sign_base.row(
            "updated_pack_trial2_source_weighted_comparison_pure_continuum_followup_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 source-weighted comparison pure-continuum followup selected now",
            sign_base.truth(route_selected),
            "This branch starts only after exact source-weighted sign support is already official and the live blocker has reduced to continuum promotion.",
        ),
        sign_base.row(
            "exact_trial2_source_weighted_comparison_pure_continuum_note_available_now",
            "pass" if note_available else "reject",
            "exact Trial-2 source-weighted comparison pure-continuum note available now",
            sign_base.truth(note_available),
            "The note must state the explicit tail contraction bound and the omitted dangerous tail estimate before continuum promotion can carry theorem weight.",
        ),
        sign_base.row(
            "exact_trial2_source_weighted_comparison_tail_contraction_admissible_now",
            "pass" if pack["tail_contraction_upper_bound"] < 1.0 else "reject",
            "exact Trial-2 source-weighted comparison tail contraction admissible now",
            sign_base.truth(pack["tail_contraction_upper_bound"] < 1.0),
            "The far-tail comparison route only makes sense if the tail-only Yukawa contraction constant stays below one.",
        ),
        sign_base.row(
            "exact_trial2_source_weighted_comparison_omitted_negative_tail_nonreversing_now",
            "pass" if pack["comparison_margin_lower_bound"] > 0.0 else "reject",
            "exact Trial-2 source-weighted comparison omitted negative tail nonreversing now",
            sign_base.truth(pack["comparison_margin_lower_bound"] > 0.0),
            "Pass means the explicit omitted negative tail bound is strictly smaller than the retained comparison margin, so sign reversal cannot occur past X=140.",
        ),
        sign_base.row(
            "exact_trial2_source_weighted_comparison_pure_continuum_support_available_now",
            "pass" if pure_continuum_support_available else "reject",
            "exact Trial-2 source-weighted comparison pure-continuum support available now",
            sign_base.truth(pure_continuum_support_available),
            "This is the strongest honest result of the branch: the retained comparison sign support survives one explicit X -> +inf tail promotion.",
        ),
        sign_base.row(
            "exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now",
            "pass" if operator_level_available else "reject",
            "exact Trial-2 pure analytic operator-level continuum refinement available now",
            sign_base.truth(operator_level_available),
            "This audit intentionally stops one layer below the full operator-level theorem and isolates the remaining refinement without overclaiming it.",
        ),
        sign_base.row(
            "updated_pack_trial2_source_weighted_comparison_pure_continuum_gate_required_now",
            "pass" if gate_required_now else "reject",
            "updated-pack Trial-2 source-weighted comparison pure-continuum gate required now",
            sign_base.truth(gate_required_now),
            "Once the continuum support layer is fixed, the next honest task is to sync it into the official theorem wording and return to conditional reopen only.",
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
        "retained_min_comparison_margin_x": float(
            pack["retained_min_comparison_margin_x"]
        ),
        "retained_negative_control_coeff_max": float(
            pack["retained_negative_control_coeff_max"]
        ),
        "retained_negative_control_coeff_max_x": float(
            pack["retained_negative_control_coeff_max_x"]
        ),
        "tail_contraction_upper_bound": float(pack["tail_contraction_upper_bound"]),
        "tail_resolvent_multiplier_upper_bound": float(
            pack["tail_resolvent_multiplier_upper_bound"]
        ),
        "source_tail_integral_upper_bound": float(
            pack["source_tail_integral_upper_bound"]
        ),
        "omitted_negative_tail_upper_bound": float(
            pack["omitted_negative_tail_upper_bound"]
        ),
        "comparison_margin_lower_bound": float(pack["comparison_margin_lower_bound"]),
        "omitted_negative_tail_over_retained_margin": float(
            pack["omitted_negative_tail_over_retained_margin"]
        ),
        "source_weighted_comparison_pure_continuum_support_available_now": (
            pure_continuum_support_available
        ),
        "exact_trial2_pure_analytic_operator_level_continuum_refinement_available_now": (
            operator_level_available
        ),
        "updated_pack_trial2_source_weighted_comparison_pure_continuum_gate_required_now": (
            gate_required_now
        ),
        "selected_next_generation_route": (
            "trial2_beta_sensitivity_source_weighted_comparison_pure_continuum_gate"
        ),
        "recommended_next_route_or_none": (
            "trial2_beta_sensitivity_source_weighted_comparison_pure_continuum_gate"
        ),
        "selected_followup_route": (
            "trial2_beta_sensitivity_source_weighted_comparison_pure_continuum_gate"
        ),
        "selected_followup_route_or_none": (
            "trial2_beta_sensitivity_source_weighted_comparison_pure_continuum_gate"
        ),
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5761",
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
            "overall_status": (
                "trial2_source_weighted_comparison_pure_continuum_audit_completed"
            ),
            "branch_completed": True,
            "breakthrough_passed_now": pure_continuum_support_available,
            "physical_reject_required": False,
        },
        {"retained_pack": pack},
    )
    outputs = write_artifact("declaration_gate", payload)
    print("[done] 8.7.56.5759-5762 Trial-2 source-weighted pure-continuum audit completed")
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
