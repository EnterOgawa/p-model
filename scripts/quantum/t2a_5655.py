#!/usr/bin/env python3
"""Generate 8.7.56.5655-.5658 Trial-2 beta-sensitivity Green-kernel audit artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_beta_sensitivity_green_kernel_followup_backend import (
    build_trial2_beta_sensitivity_green_kernel_followup_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5651-5654",
        "updated_pack_trial2_beta_sensitivity_maximum_principle_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
AUDIT_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "88_trial2_numeric_alpha_vector_qball_beta_sensitivity_green_kernel_followup_audit.md"
)

STEP_TAG = "8.7.56.5655-5658"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "beta-sensitivity Green-kernel followup audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_beta_sensitivity_green_kernel_followup_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_beta_sensitivity_maximum_principle_negative_closeout_completed_"
    "green_kernel_followup_primary_conditional_hold_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_beta_sensitivity_green_kernel_audited_spectral_projection_gate_next"
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
    """Return whether the Green-kernel note carries the expected claims."""
    patterns = (
        "Green kernel",
        "source-weighted resolvent",
        "spectral projection",
    )
    return all(pattern in text for pattern in patterns)


# 関数: audit で使う式 bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the Green-kernel followup audit."""
    return {
        "transformed_equation": (
            "H_beta w_beta = 2 beta x y_beta, "
            "H_beta = -d^2/dx^2 - (beta^2 - 1 + 6 y_beta + 3 y_beta^2)"
        ),
        "green_kernel_representation": (
            "w_beta(x) = Integral_a^b G_beta(x, xi) [2 beta xi y_beta(xi)] dxi"
        ),
        "spectral_projection": (
            "w_beta = Sum_n <phi_n, 2 beta x y_beta> / lambda_n * phi_n"
        ),
    }


# 関数: `.5655-.5658` を実行する。

def main() -> None:
    """Execute the Trial-2 beta-sensitivity Green-kernel followup audit."""
    sign_base.require(PRIOR_GATE)
    sign_base.require(AUDIT_NOTE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    note_text = sign_base.read_text(AUDIT_NOTE)
    pack = build_trial2_beta_sensitivity_green_kernel_followup_pack()

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    note_available = note_contains_audit(note_text)
    green_kernel_available_now = bool(
        pack["exact_trial2_beta_sensitivity_green_kernel_available_now"]
    )
    green_kernel_one_sign_available_now = bool(
        pack["exact_trial2_beta_sensitivity_green_kernel_one_sign_available_now"]
    )
    source_weighted_resolvent_negative_now = bool(
        pack["exact_trial2_beta_sensitivity_source_weighted_resolvent_negative_now"]
    )
    single_negative_mode_dominance_support_available_now = bool(
        pack[
            "exact_trial2_beta_sensitivity_single_negative_mode_dominance_support_available_now"
        ]
    )
    green_kernel_negative_closeout_available_now = bool(
        pack["exact_trial2_beta_sensitivity_green_kernel_negative_closeout_available_now"]
    )
    spectral_projection_followup_required_now = bool(
        pack[
            "updated_pack_trial2_beta_sensitivity_spectral_projection_followup_required_now"
        ]
    )
    exact_common_root_monotonicity_theorem_available_now = bool(
        pack["exact_trial2_common_root_monotonicity_theorem_available_now"]
    )

    rows = [
        sign_base.row(
            "updated_pack_trial2_beta_sensitivity_green_kernel_followup_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 beta-sensitivity Green-kernel followup selected now",
            sign_base.truth(route_selected),
            "This branch starts only after the naive maximum-principle proof path has already closed negatively on the canonical window.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_green_kernel_note_available_now",
            "pass" if note_available else "reject",
            "exact Trial-2 beta-sensitivity Green-kernel note available now",
            sign_base.truth(note_available),
            "The note must record the mixed-sign Green kernel, the negative source-weighted resolvent solution, and the spectral-projection followup requirement.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_green_kernel_available_now",
            "pass" if green_kernel_available_now else "reject",
            "exact Trial-2 beta-sensitivity Green kernel available now",
            sign_base.truth(green_kernel_available_now),
            "The transformed operator now has an actual Dirichlet Green-kernel / resolvent surface on the canonical window.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_green_kernel_one_sign_available_now",
            "pass" if green_kernel_one_sign_available_now else "reject",
            "exact Trial-2 beta-sensitivity Green kernel one-sign available now",
            sign_base.truth(green_kernel_one_sign_available_now),
            "A strict one-sign kernel would have closed the route directly; the retained audit instead finds mixed-sign columns and a mixed coarse full inverse.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_source_weighted_resolvent_negative_now",
            "pass" if source_weighted_resolvent_negative_now else "reject",
            "exact Trial-2 beta-sensitivity source-weighted resolvent negative now",
            sign_base.truth(source_weighted_resolvent_negative_now),
            "Even though the kernel is not globally one-sign, the actual solution of H_beta w_beta = 2 beta x y_beta remains strictly negative on the canonical window.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_single_negative_mode_dominance_support_available_now",
            "pass" if single_negative_mode_dominance_support_available_now else "reject",
            "exact Trial-2 beta-sensitivity single-negative-mode dominance support available now",
            sign_base.truth(single_negative_mode_dominance_support_available_now),
            "The canonical Dirichlet spectrum satisfies lambda_1 < 0 < lambda_2 and the first spectral coefficient dominates the retained positive-mode tail.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_green_kernel_negative_closeout_available_now",
            "pass" if green_kernel_negative_closeout_available_now else "reject",
            "exact Trial-2 beta-sensitivity Green-kernel negative closeout available now",
            sign_base.truth(green_kernel_negative_closeout_available_now),
            "The naive one-sign Green-kernel proof path is honestly closed once mixed-sign kernel support coexists with a still-negative source-weighted resolvent solution.",
        ),
        sign_base.row(
            "updated_pack_trial2_beta_sensitivity_spectral_projection_followup_required_now",
            "pass" if spectral_projection_followup_required_now else "reject",
            "updated-pack Trial-2 beta-sensitivity spectral-projection followup required now",
            sign_base.truth(spectral_projection_followup_required_now),
            "The honest next theorem route is now source-weighted spectral projection / principal-mode dominance rather than global Green-kernel one-sign control.",
        ),
        sign_base.row(
            "exact_trial2_common_root_monotonicity_theorem_available_now",
            "pass" if exact_common_root_monotonicity_theorem_available_now else "reject",
            "exact Trial-2 common-root monotonicity theorem available now",
            sign_base.truth(exact_common_root_monotonicity_theorem_available_now),
            "This audit still does not promote a strict common-root theorem; it only closes the naive Green-kernel proof path and isolates the next spectral blocker.",
        ),
    ]

    coarse_inverse = pack["coarse_full_inverse_row"]
    fine_source_solution = next(
        row for row in pack["source_solution_rows"] if row["point_count"] == 1200
    )
    fine_probe_075 = next(
        column
        for row in pack["sampled_column_rows"]
        if row["point_count"] == 1200
        for column in row["columns"]
        if abs(column["probe_fraction"] - 0.75) < 1.0e-12
    )
    spectral_row = pack["spectral_projection_row"]

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
        "exact_trial2_beta_sensitivity_green_kernel_available_now": (
            green_kernel_available_now
        ),
        "exact_trial2_beta_sensitivity_green_kernel_one_sign_available_now": (
            green_kernel_one_sign_available_now
        ),
        "exact_trial2_beta_sensitivity_source_weighted_resolvent_negative_now": (
            source_weighted_resolvent_negative_now
        ),
        "exact_trial2_beta_sensitivity_single_negative_mode_dominance_support_available_now": (
            single_negative_mode_dominance_support_available_now
        ),
        "exact_trial2_beta_sensitivity_green_kernel_negative_closeout_available_now": (
            green_kernel_negative_closeout_available_now
        ),
        "updated_pack_trial2_beta_sensitivity_spectral_projection_followup_required_now": (
            spectral_projection_followup_required_now
        ),
        "coarse_full_inverse_negative_fraction": float(
            coarse_inverse["full_inverse_negative_fraction"]
        ),
        "coarse_full_inverse_positive_fraction": float(
            coarse_inverse["full_inverse_positive_fraction"]
        ),
        "coarse_full_inverse_max": float(coarse_inverse["full_inverse_max"]),
        "fine_probe_075_negative_fraction": float(
            fine_probe_075["column_negative_fraction"]
        ),
        "fine_probe_075_positive_fraction": float(
            fine_probe_075["column_positive_fraction"]
        ),
        "fine_probe_075_column_max": float(fine_probe_075["column_max"]),
        "fine_source_solution_min": float(fine_source_solution["solution_min"]),
        "fine_source_solution_max": float(fine_source_solution["solution_max"]),
        "spectral_lambda_1": float(spectral_row["lambda_1"]),
        "spectral_lambda_2": float(spectral_row["lambda_2"]),
        "principal_mode_dominance_ratio": float(
            spectral_row["principal_mode_dominance_ratio"]
        ),
        "selected_next_generation_route": (
            "trial2_beta_sensitivity_spectral_projection_followup"
        ),
        "recommended_next_route_or_none": (
            "trial2_beta_sensitivity_spectral_projection_followup"
        ),
        "selected_followup_route": (
            "trial2_beta_sensitivity_spectral_projection_followup"
        ),
        "selected_followup_route_or_none": (
            "trial2_beta_sensitivity_spectral_projection_followup"
        ),
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5657",
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
            "overall_status": "trial2_beta_sensitivity_green_kernel_followup_audit_completed",
            "branch_completed": True,
            "breakthrough_passed_now": green_kernel_negative_closeout_available_now,
            "physical_reject_required": False,
        },
        {
            "coarse_full_inverse": coarse_inverse,
            "fine_probe_075": fine_probe_075,
            "fine_source_solution": fine_source_solution,
            "spectral_projection": spectral_row,
        },
    )
    outputs = write_artifact("declaration_gate", payload)
    print("[done] 8.7.56.5655-5658 Trial-2 beta-sensitivity Green-kernel audit completed")
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
