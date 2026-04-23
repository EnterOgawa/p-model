#!/usr/bin/env python3
"""Generate 8.7.56.5639-.5642 Trial-2 beta-sensitivity equation audit artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_beta_sensitivity_equation_backend import (
    build_trial2_beta_sensitivity_equation_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5635-5638",
        "updated_pack_trial2_target_free_common_root_strict_theorem_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
AUDIT_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "86_trial2_numeric_alpha_vector_qball_beta_sensitivity_equation_audit.md"
)

STEP_TAG = "8.7.56.5639-5642"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "beta-sensitivity equation audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_beta_sensitivity_equation_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_target_free_common_root_practical_closeout_completed_"
    "strict_theorem_negative_closeout_conditional_hold_only_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_beta_sensitivity_equation_audited_sign_support_gate_next"
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
    """Return whether the beta-sensitivity note carries the expected claims."""
    patterns = (
        "beta-sensitivity",
        "u_beta",
        "monotonicity",
    )
    return all(pattern in text for pattern in patterns)


# 関数: audit で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the beta-sensitivity audit."""
    return {
        "family_equation": "y'' + (2/x) y' + (beta^2 - 1) y + 3 y^2 + y^3 = 0",
        "sensitivity_equation": (
            "u_beta'' + (2/x) u_beta' + (beta^2 - 1 + 6 y_beta + 3 y_beta^2) u_beta = -2 beta y_beta"
        ),
        "selector": "Delta_common(beta) = alpha_qstar(beta) - alpha_R8(beta)",
    }


# 関数: `.5639-.5642` を実行する。

def main() -> None:
    """Execute the Trial-2 beta-sensitivity equation audit."""
    sign_base.require(PRIOR_GATE)
    sign_base.require(AUDIT_NOTE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    note_text = sign_base.read_text(AUDIT_NOTE)
    pack = build_trial2_beta_sensitivity_equation_pack()

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    note_available = note_contains_audit(note_text)
    exact_beta_sensitivity_equation_available_now = bool(
        pack["exact_beta_sensitivity_equation_available_now"]
    )
    local_beta_sensitivity_support_available_now = bool(
        pack["local_beta_sensitivity_support_available_now"]
    )
    u_beta_negative_support_available_now = bool(
        pack["u_beta_negative_support_available_now"]
    )
    alpha_qstar_derivative_positive_now = bool(
        pack["alpha_qstar_derivative_positive_now"]
    )
    alpha_r8_derivative_negative_now = bool(
        pack["alpha_r8_derivative_negative_now"]
    )
    i2_derivative_negative_now = bool(pack["i2_derivative_negative_now"])
    ig_derivative_negative_now = bool(pack["ig_derivative_negative_now"])
    i4_derivative_negative_now = bool(pack["i4_derivative_negative_now"])
    boundary_derivative_negative_now = bool(pack["boundary_derivative_negative_now"])
    exact_common_root_monotonicity_theorem_available_now = bool(
        pack["exact_common_root_monotonicity_theorem_available_now"]
    )
    beta_sensitivity_monotonicity_followup_required_now = bool(
        pack["beta_sensitivity_monotonicity_followup_required_now"]
    )

    rows = [
        sign_base.row(
            "updated_pack_trial2_beta_sensitivity_equation_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 beta-sensitivity equation selected now",
            sign_base.truth(route_selected),
            "This route starts only after the strict-theorem followup closes negatively and the common-root selector remains the best practical object.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_equation_note_available_now",
            "pass" if note_available else "reject",
            "exact Trial-2 beta-sensitivity equation note available now",
            sign_base.truth(note_available),
            "The note must record the sensitivity equation, u_beta sign support, and the monotonicity followup target.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_equation_available_now",
            "pass" if exact_beta_sensitivity_equation_available_now else "reject",
            "exact Trial-2 beta-sensitivity equation available now",
            sign_base.truth(exact_beta_sensitivity_equation_available_now),
            "This is the exact beta-differentiated ground-state equation for u_beta = partial_beta y_beta on the retained localized branch.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_local_support_available_now",
            "pass" if local_beta_sensitivity_support_available_now else "reject",
            "exact Trial-2 beta-sensitivity local support available now",
            sign_base.truth(local_beta_sensitivity_support_available_now),
            "Pass means the finite-difference u_beta satisfies the linearized equation with stable small residual across h = 1e-4 .. 1e-6.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_u_beta_negative_support_available_now",
            "pass" if u_beta_negative_support_available_now else "reject",
            "exact Trial-2 beta-sensitivity u_beta negative support available now",
            sign_base.truth(u_beta_negative_support_available_now),
            "Pass means u_beta(x) stays negative on the retained interior window for every tested h, giving one strong sign-support hint for future monotonicity proofs.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_alpha_qstar_derivative_positive_now",
            "pass" if alpha_qstar_derivative_positive_now else "reject",
            "exact Trial-2 beta-sensitivity alpha_qstar derivative positive now",
            sign_base.truth(alpha_qstar_derivative_positive_now),
            "The common-root selector continues to show d alpha_qstar / d beta > 0 across the tested local h-family.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_alpha_r8_derivative_negative_now",
            "pass" if alpha_r8_derivative_negative_now else "reject",
            "exact Trial-2 beta-sensitivity alpha_R8 derivative negative now",
            sign_base.truth(alpha_r8_derivative_negative_now),
            "The exact R8 readout continues to show d alpha_R8 / d beta < 0 across the tested local h-family.",
        ),
        sign_base.row(
            "exact_trial2_beta_sensitivity_integral_derivative_sign_support_now",
            "pass"
            if (
                i2_derivative_negative_now
                and ig_derivative_negative_now
                and i4_derivative_negative_now
                and boundary_derivative_negative_now
            )
            else "reject",
            "exact Trial-2 beta-sensitivity integral-derivative sign support now",
            sign_base.truth(
                i2_derivative_negative_now
                and ig_derivative_negative_now
                and i4_derivative_negative_now
                and boundary_derivative_negative_now
            ),
            "The tested local branch keeps dI2/dbeta, dIg/dbeta, dI4/dbeta, and dB/dbeta all negative, giving one concrete sign-support surface beneath R8(beta).",
        ),
        sign_base.row(
            "exact_trial2_common_root_monotonicity_theorem_available_now",
            "pass" if exact_common_root_monotonicity_theorem_available_now else "reject",
            "exact Trial-2 common-root monotonicity theorem available now",
            sign_base.truth(exact_common_root_monotonicity_theorem_available_now),
            "The current route has not yet promoted the sign-support surface into one strict analytic monotonicity / uniqueness theorem.",
        ),
        sign_base.row(
            "updated_pack_trial2_beta_sensitivity_monotonicity_followup_required_now",
            "pass" if beta_sensitivity_monotonicity_followup_required_now else "reject",
            "updated-pack Trial-2 beta-sensitivity monotonicity followup required now",
            sign_base.truth(beta_sensitivity_monotonicity_followup_required_now),
            "The honest next blocker is no longer route existence; it is the monotonicity / uniqueness theorem built on top of the beta-sensitivity equation.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "interaction_total_over_harmonic_sq_beta_common_root": float(
            pack["beta_common_root"]
        ),
        "interaction_total_over_harmonic_sq_alpha_common_value": float(
            pack["alpha_common_value"]
        ),
        "interaction_total_over_harmonic_sq_alpha_common_rel_error_vs_target": float(
            pack["alpha_common_rel_error_vs_target"]
        ),
        "exact_trial2_beta_sensitivity_equation_available_now": (
            exact_beta_sensitivity_equation_available_now
        ),
        "exact_trial2_beta_sensitivity_local_support_available_now": (
            local_beta_sensitivity_support_available_now
        ),
        "exact_trial2_beta_sensitivity_u_beta_negative_support_available_now": (
            u_beta_negative_support_available_now
        ),
        "exact_trial2_beta_sensitivity_alpha_qstar_derivative_positive_now": (
            alpha_qstar_derivative_positive_now
        ),
        "exact_trial2_beta_sensitivity_alpha_r8_derivative_negative_now": (
            alpha_r8_derivative_negative_now
        ),
        "exact_trial2_beta_sensitivity_i2_derivative_negative_now": (
            i2_derivative_negative_now
        ),
        "exact_trial2_beta_sensitivity_ig_derivative_negative_now": (
            ig_derivative_negative_now
        ),
        "exact_trial2_beta_sensitivity_i4_derivative_negative_now": (
            i4_derivative_negative_now
        ),
        "exact_trial2_beta_sensitivity_boundary_derivative_negative_now": (
            boundary_derivative_negative_now
        ),
        "exact_trial2_common_root_monotonicity_theorem_available_now": (
            exact_common_root_monotonicity_theorem_available_now
        ),
        "updated_pack_trial2_beta_sensitivity_monotonicity_followup_required_now": (
            beta_sensitivity_monotonicity_followup_required_now
        ),
        "u_beta_min_global": float(pack["u_beta_min_global"]),
        "u_beta_max_global": float(pack["u_beta_max_global"]),
        "linearized_residual_rel_rms_min": float(
            pack["linearized_residual_rel_rms_min"]
        ),
        "linearized_residual_rel_rms_max": float(
            pack["linearized_residual_rel_rms_max"]
        ),
        "d_alpha_qstar_dbeta_min": float(pack["d_alpha_qstar_dbeta_min"]),
        "d_alpha_qstar_dbeta_max": float(pack["d_alpha_qstar_dbeta_max"]),
        "d_alpha_r8_dbeta_min": float(pack["d_alpha_r8_dbeta_min"]),
        "d_alpha_r8_dbeta_max": float(pack["d_alpha_r8_dbeta_max"]),
        "d_i2_dbeta_min": float(pack["d_i2_dbeta_min"]),
        "d_i2_dbeta_max": float(pack["d_i2_dbeta_max"]),
        "d_ig_dbeta_min": float(pack["d_ig_dbeta_min"]),
        "d_ig_dbeta_max": float(pack["d_ig_dbeta_max"]),
        "d_i4_dbeta_min": float(pack["d_i4_dbeta_min"]),
        "d_i4_dbeta_max": float(pack["d_i4_dbeta_max"]),
        "d_boundary_dbeta_min": float(pack["d_boundary_dbeta_min"]),
        "d_boundary_dbeta_max": float(pack["d_boundary_dbeta_max"]),
    }

    payload = sign_base.payload(
        "8.7.56.5641",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "audit_note": sign_base.display_path(AUDIT_NOTE),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_beta_sensitivity_equation_audit_completed",
            "branch_completed": True,
            "breakthrough_passed_now": local_beta_sensitivity_support_available_now,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 beta-sensitivity equation audit completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
