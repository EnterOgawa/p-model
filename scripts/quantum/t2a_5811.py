#!/usr/bin/env python3
"""Generate 8.7.56.5811-.5814 Trial-2 J over I2 exact-normalization artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_j_over_i2_exact_normalization_backend import (
    build_trial2_j_over_i2_exact_normalization_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5807-5810",
        "updated_pack_trial2_exact_constant_route_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5811-5814"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "J over I2 exact normalization audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_j_over_i2_exact_normalization_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_exact_constant_route_inventory_audited_"
    "j_over_i2_primary_b_secondary_q_reserve_4d_hold_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_j_over_i2_exact_normalization_negative_closeout_completed_"
    "b_elimination_primary_q_reserve_4d_hold_next"
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
    """Return formulas fixed by the J/I2 exact-normalization audit."""
    return {
        "exact_alpha_formula": "alpha = f(beta)^2 / (4 pi), f(beta) = J(beta) / I2(beta)",
        "exact_goal_f_condition": "f(beta_*)^2 = 4 pi / 137",
        "current_finite_invariant_alpha": (
            "alpha(beta) = [4(g + eps - b) - q][2(5 + beta^2) + 10 g - q - 4 b] "
            "/ [36 (1 + beta^2)^2]"
        ),
        "route_reading": (
            "If no new target-constant identity for f(beta) is materialized inside the "
            "current 3D algebra, J/I2 exact normalization is a negative closeout and "
            "b elimination becomes the next honest route."
        ),
    }


# 関数: `.5811-.5814` を実行する。

def main() -> None:
    """Execute the J/I2 exact-normalization audit."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_j_over_i2_exact_normalization_pack()

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    same_3d_replay_only = bool(pack["same_3d_algebra_replay_only_now"])
    exact_identity_unavailable = not bool(
        pack["exact_trial2_j_over_i2_target_constant_identity_available_now"]
    )
    exact_theorem_unavailable = not bool(
        pack["exact_trial2_j_over_i2_exact_normalization_theorem_available_now"]
    )
    b_followup_required = bool(pack["exact_trial2_b_elimination_followup_required_now"])
    q_reserve_retained = bool(pack["exact_trial2_q_elimination_reserve_retained_now"])
    fourd_hold_retained = bool(
        pack["exact_trial2_fourd_time_component_hold_retained_now"]
    )

    rows = [
        sign_base.row(
            "updated_pack_trial2_exact_constant_route_inventory_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 exact-constant route inventory selected now",
            sign_base.truth(route_selected),
            "The J/I2 audit starts only after the route inventory has already promoted J/I2 exact normalization as the primary internal exact-goal route.",
        ),
        sign_base.row(
            "exact_trial2_same_3d_algebra_replay_only_now",
            "pass" if same_3d_replay_only else "reject",
            "exact Trial-2 same 3D algebra replay only now",
            sign_base.truth(same_3d_replay_only),
            "The current pack already exactifies alpha(beta) in finite-invariant form, so replaying the same algebra does not create a new normalization theorem by itself.",
        ),
        sign_base.row(
            "exact_trial2_j_over_i2_target_constant_identity_unavailable_now",
            "pass" if exact_identity_unavailable else "reject",
            "exact Trial-2 J over I2 target constant identity unavailable now",
            sign_base.truth(exact_identity_unavailable),
            "The current 3D algebra contains alpha = f^2/(4 pi) with f = J/I2, but it does not yet materialize a new identity that forces f(beta_*)^2 = 4 pi / 137 exactly.",
        ),
        sign_base.row(
            "exact_trial2_j_over_i2_exact_normalization_negative_closeout_now",
            "pass" if exact_theorem_unavailable else "reject",
            "exact Trial-2 J over I2 exact normalization negative closeout now",
            sign_base.truth(exact_theorem_unavailable),
            "Therefore the primary J/I2 route is now exhausted inside the current 3D algebra and closes negatively as an exact-constant extraction route.",
        ),
        sign_base.row(
            "updated_pack_trial2_b_elimination_followup_required_now",
            "pass" if b_followup_required else "reject",
            "updated-pack Trial-2 b elimination followup required now",
            sign_base.truth(b_followup_required),
            "The honest next blocker moves from J/I2 normalization to b = B/I2 elimination.",
        ),
        sign_base.row(
            "exact_trial2_q_elimination_reserve_retained_now",
            "pass" if q_reserve_retained else "reject",
            "exact Trial-2 q elimination reserve retained now",
            sign_base.truth(q_reserve_retained),
            "Quartic q elimination stays reserve because the current residual would still require an order-one relative q shift.",
        ),
        sign_base.row(
            "exact_trial2_fourd_time_component_hold_retained_now",
            "pass" if fourd_hold_retained else "reject",
            "exact Trial-2 4D time-component hold retained now",
            sign_base.truth(fourd_hold_retained),
            "4D augmentation remains held until the remaining honest 3D routes are actually exhausted.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "beta_symbolic_root": float(pack["beta_symbolic_root"]),
        "f_beta": float(pack["f_beta"]),
        "f_goal_exact_one_over_137": float(pack["f_goal_exact_one_over_137"]),
        "f_goal_gap": float(pack["f_goal_gap"]),
        "f_goal_gap_rel": float(pack["f_goal_gap_rel"]),
        "alpha_exact_symbolic": float(pack["alpha_exact_symbolic"]),
        "alpha_goal_exact_one_over_137": float(pack["alpha_goal_exact_one_over_137"]),
        "alpha_exact_symbolic_rel_error_vs_exact_goal": float(
            pack["alpha_exact_symbolic_rel_error_vs_exact_goal"]
        ),
        "delta_f_needed_rel_linearized": float(pack["delta_f_needed_rel_linearized"]),
        "exact_trial2_j_over_i2_target_constant_identity_available_now": bool(
            pack["exact_trial2_j_over_i2_target_constant_identity_available_now"]
        ),
        "exact_trial2_j_over_i2_exact_normalization_theorem_available_now": bool(
            pack["exact_trial2_j_over_i2_exact_normalization_theorem_available_now"]
        ),
        "exact_trial2_b_elimination_followup_required_now": bool(
            b_followup_required
        ),
        "exact_trial2_q_elimination_reserve_retained_now": bool(q_reserve_retained),
        "exact_trial2_fourd_time_component_hold_retained_now": bool(
            fourd_hold_retained
        ),
        "selected_next_generation_route": "trial2_b_elimination_audit",
        "recommended_next_route_or_none": ".5815-.5818",
        "selected_followup_route": "trial2_q_elimination_audit",
        "selected_followup_route_or_none": ".5819-.5822",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5813",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
        },
        rows,
        summary,
        {
            "overall_status": "trial2_j_over_i2_exact_normalization_audited",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {
            "f_goal_gap_rel": float(pack["f_goal_gap_rel"]),
            "alpha_exact_symbolic_rel_error_vs_exact_goal": float(
                pack["alpha_exact_symbolic_rel_error_vs_exact_goal"]
            ),
            "delta_f_needed_rel_linearized": float(pack["delta_f_needed_rel_linearized"]),
        },
    )
    outputs = write_artifact("declaration_gate", payload)
    print("[done] 8.7.56.5811-5814 Trial-2 J over I2 exact normalization audit completed")
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
