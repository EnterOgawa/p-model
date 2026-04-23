#!/usr/bin/env python3
"""Generate 8.7.56.5815-.5818 Trial-2 b-elimination artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_b_elimination_backend import (
    build_trial2_b_elimination_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5811-5814",
        "updated_pack_trial2_j_over_i2_exact_normalization_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5815-5818"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "b elimination audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_b_elimination_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_j_over_i2_exact_normalization_negative_closeout_completed_"
    "b_elimination_primary_q_reserve_4d_hold_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_b_elimination_negative_closeout_completed_"
    "q_primary_4d_hold_secondary_next"
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


# 関数: route で固定する式 bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the b-elimination audit."""
    return {
        "exact_alpha_formula": (
            "alpha(beta) = [4(g + eps - b) - q][2(5 + beta^2) + 10 g - q - 4 b] "
            "/ [36 (1 + beta^2)^2]"
        ),
        "exact_goal_condition": "alpha(beta_*) = 1 / 137",
        "b_quadratic": (
            "16 b^2 - 4(A0 + T0) b + A0 T0 - 36 (1 + beta^2)^2 / 137 = 0, "
            "A0 = 4(g + eps) - q, T0 = 2(5 + beta^2) + 10 g - q"
        ),
        "route_reading": (
            "If exact-goal substitution leaves a near root and a far root for b "
            "without a selector supplied by the current pack, b elimination closes "
            "negatively and q elimination becomes the next honest route."
        ),
    }


# 関数: `.5815-.5818` を実行する。

def main() -> None:
    """Execute the b-elimination audit."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_b_elimination_pack()

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    b_selector_unavailable = not bool(pack["exact_trial2_b_root_selector_available_now"])
    b_elimination_unavailable = not bool(
        pack["exact_trial2_b_elimination_theorem_available_now"]
    )
    q_followup_required = bool(pack["exact_trial2_q_elimination_followup_required_now"])
    fourd_hold_retained = bool(pack["exact_trial2_fourd_time_component_hold_retained_now"])

    rows = [
        sign_base.row(
            "updated_pack_trial2_j_over_i2_negative_closeout_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 J over I2 negative closeout selected now",
            sign_base.truth(route_selected),
            "The b-elimination audit starts only after the J/I2 route has already closed negatively.",
        ),
        sign_base.row(
            "exact_trial2_b_goal_quadratic_two_root_ambiguity_now",
            "pass" if b_selector_unavailable else "reject",
            "exact Trial-2 b goal quadratic two-root ambiguity now",
            sign_base.truth(b_selector_unavailable),
            "Substituting the exact-goal constant into the current 3D algebra yields a near b root and one far b root, but the current pack does not provide a selector between them.",
        ),
        sign_base.row(
            "exact_trial2_b_zero_naive_route_rejected_now",
            "pass" if pack["alpha_if_b_zero_rel_error_vs_goal"] > 0.1 else "reject",
            "exact Trial-2 b equals zero naive route rejected now",
            sign_base.truth(pack["alpha_if_b_zero_rel_error_vs_goal"] > 0.1),
            "Naively dropping the boundary-weighted invariant sends alpha to a much larger value, so b elimination is not equivalent to setting b = 0.",
        ),
        sign_base.row(
            "exact_trial2_b_elimination_negative_closeout_now",
            "pass" if b_elimination_unavailable else "reject",
            "exact Trial-2 b elimination negative closeout now",
            sign_base.truth(b_elimination_unavailable),
            "Therefore the current 3D algebra does not yet supply a new theorem object that eliminates b and still selects the exact-goal constant uniquely.",
        ),
        sign_base.row(
            "updated_pack_trial2_q_elimination_followup_required_now",
            "pass" if q_followup_required else "reject",
            "updated-pack Trial-2 q elimination followup required now",
            sign_base.truth(q_followup_required),
            "The next honest residual-closing route therefore moves to q = I4 / I2 elimination.",
        ),
        sign_base.row(
            "exact_trial2_fourd_time_component_hold_retained_now",
            "pass" if fourd_hold_retained else "reject",
            "exact Trial-2 4D time-component hold retained now",
            sign_base.truth(fourd_hold_retained),
            "4D augmentation remains held until the remaining honest 3D route is actually exhausted.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "beta_symbolic_root": float(pack["beta_symbolic_root"]),
        "b_beta": float(pack["b_beta"]),
        "b_root_near": float(pack["b_root_near"]),
        "b_root_far": float(pack["b_root_far"]),
        "b_root_near_rel_shift": float(pack["b_root_near_rel_shift"]),
        "b_root_far_rel_shift": float(pack["b_root_far_rel_shift"]),
        "alpha_if_b_zero": float(pack["alpha_if_b_zero"]),
        "alpha_if_b_zero_rel_error_vs_goal": float(
            pack["alpha_if_b_zero_rel_error_vs_goal"]
        ),
        "exact_trial2_b_root_selector_available_now": bool(
            pack["exact_trial2_b_root_selector_available_now"]
        ),
        "exact_trial2_b_elimination_theorem_available_now": bool(
            pack["exact_trial2_b_elimination_theorem_available_now"]
        ),
        "exact_trial2_q_elimination_followup_required_now": bool(
            q_followup_required
        ),
        "exact_trial2_fourd_time_component_hold_retained_now": bool(
            fourd_hold_retained
        ),
        "selected_next_generation_route": "trial2_q_elimination_audit",
        "recommended_next_route_or_none": ".5819-.5822",
        "selected_followup_route": "trial2_fourd_time_component_augmentation_audit",
        "selected_followup_route_or_none": ".5823-.5826",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5817",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
        },
        rows,
        summary,
        {
            "overall_status": "trial2_b_elimination_audited",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {
            "b_root_near_rel_shift": float(pack["b_root_near_rel_shift"]),
            "alpha_if_b_zero_rel_error_vs_goal": float(
                pack["alpha_if_b_zero_rel_error_vs_goal"]
            ),
        },
    )
    outputs = write_artifact("declaration_gate", payload)
    print("[done] 8.7.56.5815-5818 Trial-2 b elimination audit completed")
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
