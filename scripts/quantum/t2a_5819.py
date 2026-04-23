#!/usr/bin/env python3
"""Generate 8.7.56.5819-.5822 Trial-2 q-elimination artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_q_elimination_backend import (
    build_trial2_q_elimination_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5815-5818",
        "updated_pack_trial2_b_elimination_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5819-5822"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "q elimination audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_q_elimination_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_b_elimination_negative_closeout_completed_"
    "q_primary_4d_hold_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_q_elimination_negative_closeout_completed_"
    "three_d_internal_exactification_exhausted_4d_time_component_primary_next"
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
    """Return formulas fixed by the q-elimination audit."""
    return {
        "exact_alpha_formula": (
            "alpha(beta) = [4(g + eps - b) - q][2(5 + beta^2) + 10 g - q - 4 b] "
            "/ [36 (1 + beta^2)^2]"
        ),
        "exact_goal_condition": "alpha(beta_*) = 1 / 137",
        "q_quadratic": (
            "q^2 - (A0 + T0) q + A0 T0 - 36 (1 + beta^2)^2 / 137 = 0, "
            "A0 = 4(g + eps - b), T0 = 2(5 + beta^2) + 10 g - 4 b"
        ),
        "route_reading": (
            "If exact-goal substitution leaves a near root and a far root for q, "
            "and the near root already needs an order-one relative shift, the last "
            "remaining honest 3D route closes negatively and 4D augmentation becomes primary."
        ),
    }


# 関数: `.5819-.5822` を実行する。

def main() -> None:
    """Execute the q-elimination audit."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_q_elimination_pack()

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    q_selector_unavailable = not bool(pack["exact_trial2_q_root_selector_available_now"])
    q_order_one_shift = bool(pack["exact_trial2_q_order_one_shift_required_now"])
    q_elimination_unavailable = not bool(
        pack["exact_trial2_q_elimination_theorem_available_now"]
    )
    fourd_required = bool(pack["exact_trial2_fourd_time_component_augmentation_required_now"])

    rows = [
        sign_base.row(
            "updated_pack_trial2_b_negative_closeout_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 b negative closeout selected now",
            sign_base.truth(route_selected),
            "The q-elimination audit starts only after the b-elimination route has already closed negatively.",
        ),
        sign_base.row(
            "exact_trial2_q_goal_quadratic_two_root_ambiguity_now",
            "pass" if q_selector_unavailable else "reject",
            "exact Trial-2 q goal quadratic two-root ambiguity now",
            sign_base.truth(q_selector_unavailable),
            "Substituting the exact-goal constant into the current 3D algebra yields a near q root and one far q root, but the current pack does not provide a selector between them.",
        ),
        sign_base.row(
            "exact_trial2_q_order_one_shift_required_now",
            "pass" if q_order_one_shift else "reject",
            "exact Trial-2 q order-one shift required now",
            sign_base.truth(q_order_one_shift),
            "The near q root already requires an order-one relative shift, so q elimination is not a small internal correction of the current reduced algebra.",
        ),
        sign_base.row(
            "exact_trial2_q_elimination_negative_closeout_now",
            "pass" if q_elimination_unavailable else "reject",
            "exact Trial-2 q elimination negative closeout now",
            sign_base.truth(q_elimination_unavailable),
            "Therefore the last remaining honest 3D internal route is exhausted without producing a zero-residual exact constant.",
        ),
        sign_base.row(
            "updated_pack_trial2_fourd_time_component_augmentation_required_now",
            "pass" if fourd_required else "reject",
            "updated-pack Trial-2 4D time-component augmentation required now",
            sign_base.truth(fourd_required),
            "With J/I2, b, and q all closed negatively inside the current 3D algebra, the next honest route is 4D time-component augmentation.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "beta_symbolic_root": float(pack["beta_symbolic_root"]),
        "q_beta": float(pack["q_beta"]),
        "q_root_near": float(pack["q_root_near"]),
        "q_root_far": float(pack["q_root_far"]),
        "q_root_near_rel_shift": float(pack["q_root_near_rel_shift"]),
        "q_root_far_rel_shift": float(pack["q_root_far_rel_shift"]),
        "exact_trial2_q_root_selector_available_now": bool(
            pack["exact_trial2_q_root_selector_available_now"]
        ),
        "exact_trial2_q_order_one_shift_required_now": bool(q_order_one_shift),
        "exact_trial2_q_elimination_theorem_available_now": bool(
            pack["exact_trial2_q_elimination_theorem_available_now"]
        ),
        "exact_trial2_fourd_time_component_augmentation_required_now": bool(
            fourd_required
        ),
        "selected_next_generation_route": "trial2_fourd_time_component_augmentation_audit",
        "recommended_next_route_or_none": ".5823-.5826",
        "selected_followup_route": "none",
        "selected_followup_route_or_none": "none",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5821",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
        },
        rows,
        summary,
        {
            "overall_status": "trial2_q_elimination_audited",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {
            "q_root_near_rel_shift": float(pack["q_root_near_rel_shift"]),
            "q_root_far_rel_shift": float(pack["q_root_far_rel_shift"]),
        },
    )
    outputs = write_artifact("declaration_gate", payload)
    print("[done] 8.7.56.5819-5822 Trial-2 q elimination audit completed")
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
