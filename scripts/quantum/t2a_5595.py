#!/usr/bin/env python3
"""Generate 8.7.56.5595-.5598 Trial-2 interaction-over-harmonic gate artifacts."""

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
        "8.7.56.5591-5594",
        "updated_pack_trial2_interaction_harmonic_exact_relation_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5595-5598"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "interaction-over-harmonic gate / entropy refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_interaction_harmonic_exact_relation_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_interaction_harmonic_exact_relation_available_boundary_remainder_"
    "non_negligible_entropy_primary_gate"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_interaction_harmonic_exact_relation_negative_closeout_completed_"
    "entropy_primary_next"
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


# 関数: gate で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used by the exact-relation gate."""
    return {
        "gate_a": "Gate A = exact decomposition exists but one-term exact route remains unavailable",
        "gate_b": "Gate B = interaction-over-harmonic negative closeout is available now",
        "gate_c": "Gate C = entropy route is promoted primary now",
    }


# 関数: `.5595-.5598` を実行する。

def main() -> None:
    """Execute the Trial-2 interaction-over-harmonic gate / entropy refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary["exact_trial2_interaction_harmonic_exact_relation_available_now"]
        and not prior_summary["exact_trial2_interaction_harmonic_exact_route_available_now"]
    )
    gate_b = bool(
        gate_a and prior_summary["exact_trial2_interaction_harmonic_negative_closeout_available_now"]
    )
    gate_c = bool(gate_b and prior_summary["trial2_entropy_promoted_primary_now"])
    trial2_interaction_harmonic_exact_relation_completed_now = bool(gate_b)
    trial2_entropy_primary_next_now = bool(gate_c)

    rows = [
        sign_base.row(
            "gate_a_updated_pack_trial2_interaction_harmonic_exact_relation_negative_now",
            "pass" if gate_a else "reject",
            "gate A updated-pack Trial-2 interaction-over-harmonic exact relation negative now",
            sign_base.truth(gate_a),
            "The front runner now has one exact decomposition, but it still does not collapse into one simple target-free law.",
        ),
        sign_base.row(
            "gate_b_updated_pack_trial2_interaction_harmonic_negative_closeout_available_now",
            "pass" if gate_b else "reject",
            "gate B updated-pack Trial-2 interaction-over-harmonic negative closeout available now",
            sign_base.truth(gate_b),
            "This route closes honestly once the exact decomposition is validated and the one-term target-free collapse remains unavailable.",
        ),
        sign_base.row(
            "gate_c_updated_pack_trial2_entropy_primary_next_now",
            "pass" if gate_c else "reject",
            "gate C updated-pack Trial-2 entropy primary next now",
            sign_base.truth(gate_c),
            "With the interaction-over-harmonic route exhausted, entropy becomes the next honest low-cost direct-alpha branch.",
        ),
        sign_base.row(
            "trial2_interaction_harmonic_exact_relation_completed_now",
            "pass" if trial2_interaction_harmonic_exact_relation_completed_now else "reject",
            "Trial-2 interaction-over-harmonic exact relation completed now",
            sign_base.truth(trial2_interaction_harmonic_exact_relation_completed_now),
            "The exact-relation lane is complete once its negative closeout is fixed and synced.",
        ),
        sign_base.row(
            "trial2_entropy_primary_next_now",
            "pass" if trial2_entropy_primary_next_now else "reject",
            "Trial-2 entropy primary next now",
            sign_base.truth(trial2_entropy_primary_next_now),
            "Entropy is the next admissible followup only after the interaction-over-harmonic exact-law route dead-ends honestly.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_interaction_over_harmonic": float(
            prior_summary["retained_interaction_over_harmonic"]
        ),
        "retained_boundary_share_of_front_runner": float(
            prior_summary["retained_boundary_share_of_front_runner"]
        ),
        "retained_beta_plus_gradient_rel_error_vs_front_runner": float(
            prior_summary["retained_beta_plus_gradient_rel_error_vs_front_runner"]
        ),
        "nearest_boundary_share_of_front_runner": float(
            prior_summary["nearest_boundary_share_of_front_runner"]
        ),
        "gate_a_updated_pack_trial2_interaction_harmonic_exact_relation_negative_now": gate_a,
        "gate_b_updated_pack_trial2_interaction_harmonic_negative_closeout_available_now": gate_b,
        "gate_c_updated_pack_trial2_entropy_primary_next_now": gate_c,
        "trial2_interaction_harmonic_exact_relation_completed_now": (
            trial2_interaction_harmonic_exact_relation_completed_now
        ),
        "trial2_entropy_primary_next_now": trial2_entropy_primary_next_now,
        "selected_primary_completion_lane": "trial2_entropy_route",
        "selected_secondary_completion_lane": "conditional_reopen_only",
        "selected_reserve_completion_lane": "conditional_reopen_only",
        "selected_next_generation_route": "trial2_entropy_route",
        "recommended_next_route_or_none": "8.7.56.5599",
        "selected_followup_route": "trial2_entropy_route",
        "selected_followup_route_or_none": None,
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5597",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5599",
                "followup_route": "trial2_entropy_route",
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_interaction_harmonic_exact_relation_gate_completed",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 interaction-over-harmonic gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
