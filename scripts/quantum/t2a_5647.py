#!/usr/bin/env python3
"""Generate 8.7.56.5647-.5650 Trial-2 beta-sensitivity monotonicity audit artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_beta_sensitivity_monotonicity_followup_backend import (
    build_trial2_beta_sensitivity_monotonicity_followup_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5643-5646",
        "updated_pack_trial2_beta_sensitivity_equation_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
AUDIT_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "87_trial2_numeric_alpha_vector_qball_beta_sensitivity_monotonicity_followup_audit.md"
)

STEP_TAG = "8.7.56.5647-5650"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "beta-sensitivity monotonicity followup audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_beta_sensitivity_monotonicity_followup_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_beta_sensitivity_equation_audited_sign_support_monotonicity_"
    "followup_primary_conditional_hold_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_beta_sensitivity_maximum_principle_audited_green_kernel_gate_next"
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
    """Return whether the monotonicity note carries the expected claims."""
    patterns = (
        "maximum-principle",
        "green-kernel",
        "principal Dirichlet eigenvalue",
    )
    return all(pattern in text for pattern in patterns)


# 関数: audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the monotonicity followup audit."""
    return {
        "beta_sensitivity_equation": (
            "u_beta'' + (2/x)u_beta' + (beta^2 - 1 + 6 y_beta + 3 y_beta^2)u_beta = -2 beta y_beta"
        ),
        "transformed_operator": (
            "w_beta = x u_beta, H_beta w_beta = 2 beta x y_beta, "
            "H_beta = -d^2/dx^2 - (beta^2 - 1 + 6 y_beta + 3 y_beta^2)"
        ),
        "maximum_principle_condition": "lambda_1(H_beta; [x_min, x_max]) > 0",
    }


# 関数: `.5647-.5650` を実行する。

def main() -> None:
    """Execute the Trial-2 beta-sensitivity monotonicity followup audit."""
    sign_base.require(PRIOR_GATE)
    sign_base.require(AUDIT_NOTE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    note_text = sign_base.read_text(AUDIT_NOTE)
    pack = build_trial2_beta_sensitivity_monotonicity_followup_pack()

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    note_available = note_contains_audit(note_text)
    transformed_operator_available_now = bool(
        pack["exact_trial2_beta_sensitivity_transformed_operator_available_now"]
    )
    source_positive_on_canonical_window_now = bool(
        pack["transformed_source_positive_on_canonical_window_now"]
    )
    potential_positive_on_canonical_window_now = bool(
        pack["transformed_potential_positive_on_canonical_window_now"]
    )
    maximum_principle_available_on_inner_window_now = bool(
        pack["maximum_principle_available_on_inner_window_now"]
    )
    maximum_principle_available_on_canonical_window_now = bool(
        pack["maximum_principle_available_on_canonical_window_now"]
    )
    principal_eigenvalue_sign_flip_available_now = bool(
        pack["principal_eigenvalue_sign_flip_available_now"]
    )
    maximum_principle_negative_closeout_available_now = bool(
        pack["maximum_principle_negative_closeout_available_now"]
    )
    green_kernel_followup_required_now = bool(
        pack["green_kernel_followup_required_now"]
    )
    exact_common_root_monotonicity_theorem_available_now = bool(
        pack["exact_trial2_common_root_monotonicity_theorem_available_now"]
    )

    rows = [
        sign_base.row(
            "updated_pack_trial2_beta_sensitivity_monotonicity_followup_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 beta-sensitivity monotonicity followup selected now",
            sign_base.truth(route_selected),
            "This branch starts only after the exact beta-sensitivity equation and local sign-support surface are already available.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_monotonicity_note_available_now",
            "pass" if note_available else "reject",
            "exact Trial-2 beta-sensitivity monotonicity note available now",
            sign_base.truth(note_available),
            "The note must record the transformed operator, principal-eigenvalue sign flip, and green-kernel followup requirement.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_transformed_operator_available_now",
            "pass" if transformed_operator_available_now else "reject",
            "exact Trial-2 beta-sensitivity transformed operator available now",
            sign_base.truth(transformed_operator_available_now),
            "This is the Dirichlet operator H_beta = -d^2/dx^2 - V_beta acting on w_beta = x u_beta.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_source_positive_on_canonical_window_now",
            "pass" if source_positive_on_canonical_window_now else "reject",
            "exact Trial-2 beta-sensitivity source positive on canonical window now",
            sign_base.truth(source_positive_on_canonical_window_now),
            "The transformed source 2 beta x y_beta stays positive on the canonical window, so failure of the maximum-principle path is not caused by source sign loss.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_potential_positive_on_canonical_window_now",
            "pass" if potential_positive_on_canonical_window_now else "reject",
            "exact Trial-2 beta-sensitivity potential positive on canonical window now",
            sign_base.truth(potential_positive_on_canonical_window_now),
            "The transformed potential V_beta remains positive on the canonical window, so the blocker is not a simple potential-sign flip either.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_maximum_principle_available_on_inner_window_now",
            "pass" if maximum_principle_available_on_inner_window_now else "reject",
            "exact Trial-2 beta-sensitivity maximum principle available on inner window now",
            sign_base.truth(maximum_principle_available_on_inner_window_now),
            "Inner windows such as [0.05, 10] still have positive principal Dirichlet eigenvalue, so the route is locally plausible before the canonical window is imposed.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_maximum_principle_available_on_canonical_window_now",
            "pass" if maximum_principle_available_on_canonical_window_now else "reject",
            "exact Trial-2 beta-sensitivity maximum principle available on canonical window now",
            sign_base.truth(maximum_principle_available_on_canonical_window_now),
            "The canonical theorem path would require lambda_1(H_beta; [0.05, 20]) > 0, which does not hold in the retained audit.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_principal_eigenvalue_sign_flip_available_now",
            "pass" if principal_eigenvalue_sign_flip_available_now else "reject",
            "exact Trial-2 beta-sensitivity principal-eigenvalue sign flip available now",
            sign_base.truth(principal_eigenvalue_sign_flip_available_now),
            "The principal Dirichlet eigenvalue changes sign between the inner and canonical windows, so the loss of inverse-positivity is an actual spectral event rather than noise.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_maximum_principle_negative_closeout_available_now",
            "pass" if maximum_principle_negative_closeout_available_now else "reject",
            "exact Trial-2 beta-sensitivity maximum-principle negative closeout available now",
            sign_base.truth(maximum_principle_negative_closeout_available_now),
            "The classical maximum-principle proof path is honestly closed once the canonical-window principal eigenvalue turns negative while source and potential signs remain favorable.",
        ),
        sign_base.row(
            "updated_pack_trial2_beta_sensitivity_green_kernel_followup_required_now",
            "pass" if green_kernel_followup_required_now else "reject",
            "updated-pack Trial-2 beta-sensitivity green-kernel followup required now",
            sign_base.truth(green_kernel_followup_required_now),
            "The honest next theorem route is no longer a naive maximum principle, but a Green-kernel / resolvent sign route on the transformed operator.",
        ),
        sign_base.row(
            "exact_trial2_common_root_monotonicity_theorem_available_now",
            "pass" if exact_common_root_monotonicity_theorem_available_now else "reject",
            "exact Trial-2 common-root monotonicity theorem available now",
            sign_base.truth(exact_common_root_monotonicity_theorem_available_now),
            "This audit still does not promote an analytic monotonicity / uniqueness theorem; it only closes the most naive maximum-principle proof path.",
        ),
    ]

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
        "exact_trial2_beta_sensitivity_transformed_operator_available_now": (
            transformed_operator_available_now
        ),
        "exact_trial2_beta_sensitivity_source_positive_on_canonical_window_now": (
            source_positive_on_canonical_window_now
        ),
        "exact_trial2_beta_sensitivity_potential_positive_on_canonical_window_now": (
            potential_positive_on_canonical_window_now
        ),
        "exact_trial2_beta_sensitivity_maximum_principle_available_on_inner_window_now": (
            maximum_principle_available_on_inner_window_now
        ),
        "exact_trial2_beta_sensitivity_maximum_principle_available_on_canonical_window_now": (
            maximum_principle_available_on_canonical_window_now
        ),
        "exact_trial2_beta_sensitivity_principal_eigenvalue_sign_flip_available_now": (
            principal_eigenvalue_sign_flip_available_now
        ),
        "exact_trial2_beta_sensitivity_maximum_principle_negative_closeout_available_now": (
            maximum_principle_negative_closeout_available_now
        ),
        "updated_pack_trial2_beta_sensitivity_green_kernel_followup_required_now": (
            green_kernel_followup_required_now
        ),
        "exact_trial2_common_root_monotonicity_theorem_available_now": (
            exact_common_root_monotonicity_theorem_available_now
        ),
        "window5_principal_dirichlet_eigenvalue": float(
            pack["window5_principal_dirichlet_eigenvalue"]
        ),
        "window10_principal_dirichlet_eigenvalue": float(
            pack["window10_principal_dirichlet_eigenvalue"]
        ),
        "window12_principal_dirichlet_eigenvalue": float(
            pack["window12_principal_dirichlet_eigenvalue"]
        ),
        "window15_principal_dirichlet_eigenvalue": float(
            pack["window15_principal_dirichlet_eigenvalue"]
        ),
        "window20_principal_dirichlet_eigenvalue": float(
            pack["window20_principal_dirichlet_eigenvalue"]
        ),
        "principal_dirichlet_sign_flip_root_x_max": float(
            pack["principal_dirichlet_sign_flip_root_x_max"]
        ),
        "transformed_potential_zero_crossing_x": float(
            pack["transformed_potential_zero_crossing_x"]
        ),
        "canonical_window_potential_min": float(pack["canonical_window_potential_min"]),
        "canonical_window_potential_max": float(pack["canonical_window_potential_max"]),
        "canonical_window_source_min": float(pack["canonical_window_source_min"]),
        "canonical_window_source_max": float(pack["canonical_window_source_max"]),
        "selected_next_generation_route": "trial2_beta_sensitivity_green_kernel_followup",
        "recommended_next_route_or_none": "trial2_beta_sensitivity_green_kernel_followup",
        "selected_followup_route": "trial2_beta_sensitivity_green_kernel_followup",
        "selected_followup_route_or_none": "trial2_beta_sensitivity_green_kernel_followup",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5649",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "audit_note": sign_base.display_path(AUDIT_NOTE),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "trial2_beta_sensitivity_green_kernel_followup",
                "followup_route": "conditional_hold_only",
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_beta_sensitivity_monotonicity_audit_completed",
            "branch_completed": True,
            "breakthrough_passed_now": maximum_principle_negative_closeout_available_now,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 beta-sensitivity monotonicity audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
