#!/usr/bin/env python3
"""Generate 8.7.56.5803-.5806 exact-constant route-inventory artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_exact_constant_route_priority_backend import (
    build_trial2_exact_constant_route_priority_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5799-5802",
        "updated_pack_trial2_zero_residual_final_theorem_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5803-5806"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "exact-constant route inventory audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_exact_constant_route_inventory_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_exact_alpha_finite_invariant_form_completed_"
    "zero_residual_exact_constant_unavailable_current_pack_"
    "conditional_reopen_only_next"
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


# 関数: route inventory の式 bundle を返す。
def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the exact-constant route inventory."""
    return {
        "exact_alpha_formula": (
            "alpha(beta) = [4(g + eps - b) - q] "
            "[2(5 + beta^2) + 10 g - q - 4 b] / [36 (1 + beta^2)^2]"
        ),
        "exact_goal": "alpha_goal = 1 / 137",
        "normalization_identity": "alpha = f(beta)^2 / (4 pi), f = J / I2",
        "route_order": (
            "primary = J/I2 exact normalization; secondary = b elimination; "
            "reserve = q elimination; hold = 4D time-component augmentation"
        ),
    }


# 関数: `.5803-.5806` を実行する。
def main() -> None:
    """Execute the exact-constant route inventory audit."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_exact_constant_route_priority_pack()
    row = pack["symbolic_row"]

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    same_3d_zero_residual_unavailable = not bool(
        pack["exact_trial2_same_3d_invariant_algebra_zero_residual_available_now"]
    )
    j_over_i2_primary = bool(pack["exact_trial2_j_over_i2_normalization_primary_now"])
    b_secondary = bool(pack["exact_trial2_b_elimination_secondary_now"])
    q_reserve = bool(pack["exact_trial2_q_elimination_reserve_now"])
    fourd_hold = bool(pack["exact_trial2_fourd_time_component_augmentation_hold_now"])
    j_over_i2_required = bool(
        pack["updated_pack_trial2_j_over_i2_normalization_audit_required_now"]
    )

    rows = [
        sign_base.row(
            "updated_pack_trial2_exact_constant_route_inventory_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 exact-constant route inventory selected now",
            sign_base.truth(route_selected),
            "This audit starts only after the current pack has been honestly classified as exact-form available but zero-residual exact constant unavailable.",
        ),
        sign_base.row(
            "exact_trial2_same_3d_invariant_algebra_zero_residual_unavailable_now",
            "pass" if same_3d_zero_residual_unavailable else "reject",
            "exact Trial-2 same 3D invariant algebra zero residual unavailable now",
            sign_base.truth(same_3d_zero_residual_unavailable),
            "The current finite-invariant 3D algebra is already exact enough to evaluate alpha(beta_symbolic_root), so replaying the same algebra without a new theorem object is not an honest zero-residual route.",
        ),
        sign_base.row(
            "exact_trial2_j_over_i2_normalization_primary_now",
            "pass" if j_over_i2_primary else "reject",
            "exact Trial-2 J over I2 normalization primary now",
            sign_base.truth(j_over_i2_primary),
            "Because alpha = f^2/(4 pi) with f = J/I2, the shortest internal route is an exact normalization theorem for J/I2 itself.",
        ),
        sign_base.row(
            "exact_trial2_b_elimination_secondary_now",
            "pass" if b_secondary else "reject",
            "exact Trial-2 b elimination secondary now",
            sign_base.truth(b_secondary),
            "The boundary-weighted invariant b = B/I2 remains the cleanest secondary elimination target inside the current reduced algebra.",
        ),
        sign_base.row(
            "exact_trial2_q_elimination_reserve_now",
            "pass" if q_reserve else "reject",
            "exact Trial-2 q elimination reserve now",
            sign_base.truth(q_reserve),
            "The quartic invariant q = I4/I2 would need an order-one relative shift to absorb the residual, so it stays reserve rather than primary.",
        ),
        sign_base.row(
            "exact_trial2_fourd_time_component_augmentation_hold_now",
            "pass" if fourd_hold else "reject",
            "exact Trial-2 4D time-component augmentation hold now",
            sign_base.truth(fourd_hold),
            "A 4D augmentation route remains admissible only after the 3D internal exactification routes have been honestly exhausted.",
        ),
        sign_base.row(
            "updated_pack_trial2_j_over_i2_normalization_audit_required_now",
            "pass" if j_over_i2_required else "reject",
            "updated-pack Trial-2 J over I2 normalization audit required now",
            sign_base.truth(j_over_i2_required),
            "The next honest blocker is no longer route discovery but the actual J/I2 exact-normalization audit.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "beta_symbolic_root": float(row["beta_symbolic_root"]),
        "alpha_exact_symbolic": float(row["alpha_exact_symbolic"]),
        "alpha_goal_exact_one_over_137": float(row["alpha_goal_exact_one_over_137"]),
        "alpha_goal_gap": float(row["alpha_goal_gap"]),
        "alpha_goal_gap_rel": float(row["alpha_goal_gap_rel"]),
        "d_alpha_df": float(row["d_alpha_df"]),
        "d_alpha_dg": float(row["d_alpha_dg"]),
        "d_alpha_dq": float(row["d_alpha_dq"]),
        "d_alpha_db": float(row["d_alpha_db"]),
        "delta_f_needed_rel": float(row["delta_f_needed_rel"]),
        "delta_g_needed_rel": float(row["delta_g_needed_rel"]),
        "delta_q_needed_rel": float(row["delta_q_needed_rel"]),
        "delta_b_needed_rel": float(row["delta_b_needed_rel"]),
        "exact_trial2_j_over_i2_normalization_primary_now": bool(j_over_i2_primary),
        "exact_trial2_b_elimination_secondary_now": bool(b_secondary),
        "exact_trial2_q_elimination_reserve_now": bool(q_reserve),
        "exact_trial2_fourd_time_component_augmentation_hold_now": bool(fourd_hold),
        "updated_pack_trial2_j_over_i2_normalization_audit_required_now": bool(
            j_over_i2_required
        ),
        "selected_next_generation_route": "trial2_j_over_i2_exact_normalization_audit",
        "recommended_next_route_or_none": ".5811-.5814",
        "selected_followup_route": "trial2_b_elimination_audit",
        "selected_followup_route_or_none": ".5815-.5818",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5805",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
        },
        rows,
        summary,
        {
            "overall_status": "trial2_exact_constant_route_inventory_audited",
            "branch_completed": True,
            "breakthrough_passed_now": j_over_i2_primary,
            "physical_reject_required": False,
        },
        {
            "beta_symbolic_root": float(row["beta_symbolic_root"]),
            "alpha_goal_gap_rel": float(row["alpha_goal_gap_rel"]),
            "delta_f_needed_rel": float(row["delta_f_needed_rel"]),
            "delta_b_needed_rel": float(row["delta_b_needed_rel"]),
            "delta_q_needed_rel": float(row["delta_q_needed_rel"]),
        },
    )
    outputs = write_artifact("declaration_gate", payload)
    print("[done] 8.7.56.5803-5806 Trial-2 exact-constant route inventory audit completed")
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
