#!/usr/bin/env python3
"""Generate 8.7.56.5663-.5666 Trial-2 beta-sensitivity spectral-projection artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_beta_sensitivity_spectral_projection_followup_backend import (
    build_trial2_beta_sensitivity_spectral_projection_followup_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5659-5662",
        "updated_pack_trial2_beta_sensitivity_green_kernel_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
AUDIT_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "89_trial2_numeric_alpha_vector_qball_beta_sensitivity_spectral_projection_followup_audit.md"
)

STEP_TAG = "8.7.56.5663-5666"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "beta-sensitivity spectral-projection followup audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_beta_sensitivity_spectral_projection_followup_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_beta_sensitivity_green_kernel_negative_closeout_completed_"
    "spectral_projection_followup_primary_conditional_hold_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_beta_sensitivity_spectral_projection_audited_"
    "discrete_pointwise_dominance_gate_next"
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
    """Return whether the spectral-projection note carries the expected claims."""
    patterns = (
        "spectral projection",
        "pointwise dominance",
        "continuum followup",
    )
    return all(pattern in text for pattern in patterns)


# 関数: audit で使う式 bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the spectral-projection followup audit."""
    return {
        "spectral_projection": (
            "w_beta = Sum_n <phi_n, 2 beta x y_beta> / lambda_n * phi_n"
        ),
        "principal_component": "w_beta^(1) = <phi_1, 2 beta x y_beta> / lambda_1 * phi_1",
        "pointwise_margin": (
            "M(x) = |w_beta^(1)(x)| - Sum_{n>=2} |<phi_n, 2 beta x y_beta> / lambda_n| "
            "|phi_n(x)|"
        ),
    }


# 関数: `.5663-.5666` を実行する。

def main() -> None:
    """Execute the Trial-2 beta-sensitivity spectral-projection followup audit."""
    sign_base.require(PRIOR_GATE)
    sign_base.require(AUDIT_NOTE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    note_text = sign_base.read_text(AUDIT_NOTE)
    pack = build_trial2_beta_sensitivity_spectral_projection_followup_pack()

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    note_available = note_contains_audit(note_text)
    discrete_spectral_projection_available_now = bool(
        pack["exact_trial2_beta_sensitivity_discrete_spectral_projection_available_now"]
    )
    principal_mode_one_sign_now = bool(
        pack["exact_trial2_beta_sensitivity_principal_mode_one_sign_now"]
    )
    principal_component_negative_now = bool(
        pack["exact_trial2_beta_sensitivity_principal_component_negative_now"]
    )
    discrete_pointwise_dominance_now = bool(
        pack["exact_trial2_beta_sensitivity_discrete_pointwise_dominance_now"]
    )
    discrete_negativity_theorem_available_now = bool(
        pack["exact_trial2_beta_sensitivity_discrete_negativity_theorem_available_now"]
    )
    exact_common_root_monotonicity_theorem_available_now = bool(
        pack["exact_trial2_common_root_monotonicity_theorem_available_now"]
    )
    continuum_spectral_projection_followup_required_now = bool(
        pack[
            "updated_pack_trial2_beta_sensitivity_continuum_spectral_projection_followup_required_now"
        ]
    )

    rows = [
        sign_base.row(
            "updated_pack_trial2_beta_sensitivity_spectral_projection_followup_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 beta-sensitivity spectral-projection followup selected now",
            sign_base.truth(route_selected),
            "This branch starts only after the naive one-sign Green-kernel theorem has already closed negatively on the canonical window.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_spectral_projection_note_available_now",
            "pass" if note_available else "reject",
            "exact Trial-2 beta-sensitivity spectral-projection note available now",
            sign_base.truth(note_available),
            "The note must record the full-spectrum decomposition, the pointwise dominance verdict, and the still-open continuum followup.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_discrete_spectral_projection_available_now",
            "pass" if discrete_spectral_projection_available_now else "reject",
            "exact Trial-2 beta-sensitivity discrete spectral projection available now",
            sign_base.truth(discrete_spectral_projection_available_now),
            "The transformed Dirichlet operator now carries one exact finite spectral decomposition on the canonical window.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_principal_mode_one_sign_now",
            "pass" if principal_mode_one_sign_now else "reject",
            "exact Trial-2 beta-sensitivity principal mode one-sign now",
            sign_base.truth(principal_mode_one_sign_now),
            "The principal Dirichlet eigenfunction must stay one-sign before any pointwise dominance theorem can be attempted.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_principal_component_negative_now",
            "pass" if principal_component_negative_now else "reject",
            "exact Trial-2 beta-sensitivity principal component negative now",
            sign_base.truth(principal_component_negative_now),
            "Because lambda_1 < 0 and the source is positive, the first spectral component must stay negative on the canonical window.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_discrete_pointwise_dominance_now",
            "pass" if discrete_pointwise_dominance_now else "reject",
            "exact Trial-2 beta-sensitivity discrete pointwise dominance now",
            sign_base.truth(discrete_pointwise_dominance_now),
            "Pass means the absolute principal component exceeds the absolute remainder sum at every sampled point on every retained canonical-window discretization.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_discrete_negativity_theorem_available_now",
            "pass" if discrete_negativity_theorem_available_now else "reject",
            "exact Trial-2 beta-sensitivity discrete negativity theorem available now",
            sign_base.truth(discrete_negativity_theorem_available_now),
            "This is the strongest honest result in the current branch: the full finite spectral expansion reconstructs the source-weighted resolvent exactly and proves its negativity by pointwise dominance.",
        ),
        sign_base.row(
            "updated_pack_trial2_beta_sensitivity_continuum_spectral_projection_followup_required_now",
            "pass" if continuum_spectral_projection_followup_required_now else "reject",
            "updated-pack Trial-2 beta-sensitivity continuum spectral-projection followup required now",
            sign_base.truth(continuum_spectral_projection_followup_required_now),
            "The route remains unfinished because the current theorem is still discrete; the next blocker is continuum / operator-level promotion rather than another discrete replay.",
        ),
        sign_base.row(
            "exact_trial2_common_root_monotonicity_theorem_available_now",
            "pass" if exact_common_root_monotonicity_theorem_available_now else "reject",
            "exact Trial-2 common-root monotonicity theorem available now",
            sign_base.truth(exact_common_root_monotonicity_theorem_available_now),
            "The current audit still does not promote the final strict common-root theorem; it only closes the discrete spectral-projection support layer.",
        ),
    ]

    fine_row = next(row for row in pack["spectral_rows"] if row["point_count"] == 1200)

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "interaction_total_over_harmonic_sq_beta_common_root": float(
            prior_summary["interaction_total_over_harmonic_sq_beta_common_root"]
        ),
        "interaction_total_over_harmonic_sq_alpha_common_value": float(
            prior_summary["interaction_total_over_harmonic_sq_alpha_common_value"]
        ),
        "interaction_total_over_harmonic_sq_alpha_common_rel_error_vs_target": float(
            prior_summary["interaction_total_over_harmonic_sq_alpha_common_rel_error_vs_target"]
        ),
        "exact_trial2_beta_sensitivity_discrete_spectral_projection_available_now": (
            discrete_spectral_projection_available_now
        ),
        "exact_trial2_beta_sensitivity_principal_mode_one_sign_now": (
            principal_mode_one_sign_now
        ),
        "exact_trial2_beta_sensitivity_principal_component_negative_now": (
            principal_component_negative_now
        ),
        "exact_trial2_beta_sensitivity_discrete_pointwise_dominance_now": (
            discrete_pointwise_dominance_now
        ),
        "exact_trial2_beta_sensitivity_discrete_negativity_theorem_available_now": (
            discrete_negativity_theorem_available_now
        ),
        "updated_pack_trial2_beta_sensitivity_continuum_spectral_projection_followup_required_now": (
            continuum_spectral_projection_followup_required_now
        ),
        "discrete_pointwise_margin_min_global": float(
            pack["discrete_pointwise_margin_min_global"]
        ),
        "discrete_pointwise_margin_max_global": float(
            pack["discrete_pointwise_margin_max_global"]
        ),
        "reconstruction_rel_linf_max": float(pack["reconstruction_rel_linf_max"]),
        "spectral_lambda_1_min": float(pack["spectral_lambda_1_min"]),
        "spectral_lambda_2_min": float(pack["spectral_lambda_2_min"]),
        "fine_pointwise_margin_min": float(fine_row["pointwise_margin_min"]),
        "fine_reconstructed_solution_max": float(fine_row["reconstructed_solution_max"]),
        "selected_next_generation_route": (
            "trial2_beta_sensitivity_continuum_spectral_projection_followup"
        ),
        "recommended_next_route_or_none": (
            "trial2_beta_sensitivity_continuum_spectral_projection_followup"
        ),
        "selected_followup_route": (
            "trial2_beta_sensitivity_continuum_spectral_projection_followup"
        ),
        "selected_followup_route_or_none": (
            "trial2_beta_sensitivity_continuum_spectral_projection_followup"
        ),
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5665",
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
            "overall_status": "trial2_beta_sensitivity_spectral_projection_followup_audit_completed",
            "branch_completed": True,
            "breakthrough_passed_now": discrete_negativity_theorem_available_now,
            "physical_reject_required": False,
        },
        {"fine_spectral_row": fine_row},
    )
    outputs = write_artifact("declaration_gate", payload)
    print(
        "[done] 8.7.56.5663-5666 Trial-2 beta-sensitivity spectral-projection audit completed"
    )
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
