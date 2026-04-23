#!/usr/bin/env python3
"""Generate 8.7.56.5807-.5810 exact-constant route gate artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5803-5806",
        "updated_pack_trial2_exact_constant_route_inventory_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5807-5810"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "exact-constant route gate"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_exact_constant_route_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_exact_constant_route_inventory_audited_"
    "j_over_i2_primary_b_secondary_q_reserve_4d_hold_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_exact_constant_route_inventory_audited_"
    "j_over_i2_primary_b_secondary_q_reserve_4d_hold_next"
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


# 関数: `.5807-.5810` を実行する。
def main() -> None:
    """Execute the exact-constant route gate."""
    sign_base.require(PRIOR_AUDIT)

    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    j_over_i2_primary = bool(
        prior_summary["exact_trial2_j_over_i2_normalization_primary_now"]
    )
    b_secondary = bool(prior_summary["exact_trial2_b_elimination_secondary_now"])
    q_reserve = bool(prior_summary["exact_trial2_q_elimination_reserve_now"])
    fourd_hold = bool(prior_summary["exact_trial2_fourd_time_component_augmentation_hold_now"])

    rows = [
        sign_base.row(
            "gate_a_trial2_exact_constant_route_inventory_selected_now",
            "pass" if route_selected else "reject",
            "Gate A Trial-2 exact-constant route inventory selected now",
            sign_base.truth(route_selected),
            "The route gate starts only after the route inventory has been honestly completed and classified.",
        ),
        sign_base.row(
            "gate_b_trial2_j_over_i2_primary_now",
            "pass" if j_over_i2_primary else "reject",
            "Gate B Trial-2 J over I2 normalization primary now",
            sign_base.truth(j_over_i2_primary),
            "The first exact-constant blocker is promoted as J/I2 exact normalization rather than a replay of selector or exact-alpha extraction.",
        ),
        sign_base.row(
            "gate_c_trial2_b_secondary_now",
            "pass" if b_secondary else "reject",
            "Gate C Trial-2 b elimination secondary now",
            sign_base.truth(b_secondary),
            "Boundary-weighted invariant elimination is kept immediately downstream of the J/I2 normalization lane.",
        ),
        sign_base.row(
            "gate_d_trial2_q_reserve_now",
            "pass" if q_reserve else "reject",
            "Gate D Trial-2 q elimination reserve now",
            sign_base.truth(q_reserve),
            "Quartic elimination stays reserve because the current local sensitivity requires an order-one relative change.",
        ),
        sign_base.row(
            "gate_e_trial2_fourd_hold_now",
            "pass" if fourd_hold else "reject",
            "Gate E Trial-2 4D time-component augmentation hold now",
            sign_base.truth(fourd_hold),
            "The 4D augmentation lane remains admissible but is held behind the 3D internal exactification routes.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "alpha_goal_gap_rel": float(prior_summary["alpha_goal_gap_rel"]),
        "delta_f_needed_rel": float(prior_summary["delta_f_needed_rel"]),
        "delta_b_needed_rel": float(prior_summary["delta_b_needed_rel"]),
        "delta_q_needed_rel": float(prior_summary["delta_q_needed_rel"]),
        "exact_trial2_j_over_i2_normalization_primary_now": bool(j_over_i2_primary),
        "exact_trial2_b_elimination_secondary_now": bool(b_secondary),
        "exact_trial2_q_elimination_reserve_now": bool(q_reserve),
        "exact_trial2_fourd_time_component_augmentation_hold_now": bool(fourd_hold),
        "selected_next_generation_route": "trial2_j_over_i2_exact_normalization_audit",
        "recommended_next_route_or_none": ".5811-.5814",
        "selected_followup_route": "trial2_b_elimination_audit",
        "selected_followup_route_or_none": ".5815-.5818",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5809",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "formulae": {
                "route_order": (
                    "primary = J/I2 exact normalization; "
                    "secondary = b elimination; reserve = q elimination; "
                    "hold = 4D time-component augmentation"
                )
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_exact_constant_route_gate_completed",
            "branch_completed": True,
            "breakthrough_passed_now": j_over_i2_primary,
            "physical_reject_required": False,
        },
        {
            "alpha_goal_gap_rel": float(prior_summary["alpha_goal_gap_rel"]),
            "delta_f_needed_rel": float(prior_summary["delta_f_needed_rel"]),
        },
    )
    outputs = write_artifact("declaration_gate", payload)
    print("[done] 8.7.56.5807-5810 Trial-2 exact-constant route gate completed")
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
