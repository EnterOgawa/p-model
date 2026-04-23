#!/usr/bin/env python3
"""Generate 8.7.56.5579-.5582 Trial-2 alpha(beta) gate artifacts."""

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
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5575-5578",
        "updated_pack_trial2_alpha_beta_curve_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5579-5582"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "alpha(beta) gate / energy-partition refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_alpha_beta_curve_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_alpha_beta_family_global_nonunique_local_microshift_"
    "energy_partition_followup_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_alpha_beta_curve_audited_local_beta_microshift_completed_"
    "energy_partition_primary_entropy_secondary_next"
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
    """Return formulas used by the alpha(beta) gate."""
    return {
        "gate_a": "Gate A = alpha(beta) local branch unique now",
        "gate_b": "Gate B = alpha(beta) exact global theorem unavailable now",
        "gate_c": "Gate C = energy partition promoted as the next primary route",
    }


# 関数: `.5579-.5582` を実行する。

def main() -> None:
    """Execute the Trial-2 alpha(beta) gate."""
    sign_base.require(PRIOR_GATE)
    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]

    gate_a = bool(prior_summary["exact_trial2_alpha_beta_local_branch_unique_now"])
    gate_b = bool(not prior_summary["exact_trial2_alpha_beta_exact_route_available_now"])
    gate_c = bool(prior_summary["updated_pack_trial2_energy_partition_primary_followup_required_now"])
    trial2_alpha_beta_curve_completed_now = bool(gate_a and gate_b)
    trial2_energy_partition_primary_next_now = bool(trial2_alpha_beta_curve_completed_now and gate_c)
    trial2_entropy_route_secondary_retained_now = True

    rows = [
        sign_base.row(
            "gate_a_updated_pack_trial2_alpha_beta_local_branch_unique_now",
            "pass" if gate_a else "reject",
            "gate A updated-pack Trial-2 alpha(beta) local branch unique now",
            sign_base.truth(gate_a),
            "The retained high-beta branch only compresses the blocker honestly if it carries one local target crossing.",
        ),
        sign_base.row(
            "gate_b_updated_pack_trial2_alpha_beta_exact_route_unavailable_now",
            "pass" if gate_b else "reject",
            "gate B updated-pack Trial-2 alpha(beta) exact route unavailable now",
            sign_base.truth(gate_b),
            "The global family remains nonunique, so alpha(beta) does not close as one strict theorem route.",
        ),
        sign_base.row(
            "gate_c_updated_pack_trial2_energy_partition_promoted_primary_now",
            "pass" if gate_c else "reject",
            "gate C updated-pack Trial-2 energy partition promoted primary now",
            sign_base.truth(gate_c),
            "Once alpha(beta) reduces the blocker to one beta microshift, energy partition becomes the next honest route.",
        ),
        sign_base.row(
            "trial2_alpha_beta_curve_completed_now",
            "pass" if trial2_alpha_beta_curve_completed_now else "reject",
            "Trial-2 alpha(beta) curve completed now",
            sign_base.truth(trial2_alpha_beta_curve_completed_now),
            "The route is complete once the local microshift is fixed and the exact global theorem remains unavailable.",
        ),
        sign_base.row(
            "trial2_energy_partition_primary_next_now",
            "pass" if trial2_energy_partition_primary_next_now else "reject",
            "Trial-2 energy partition primary next now",
            sign_base.truth(trial2_energy_partition_primary_next_now),
            "The next honest blocker is whether the retained beta microshift is encoded by one simple energy ratio.",
        ),
        sign_base.row(
            "trial2_entropy_route_secondary_retained_now",
            "pass" if trial2_entropy_route_secondary_retained_now else "reject",
            "Trial-2 entropy route secondary retained now",
            sign_base.truth(trial2_entropy_route_secondary_retained_now),
            "Entropy remains secondary until energy partition dead-ends honestly.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_beta1": float(prior_summary["retained_beta1"]),
        "retained_alpha_at_q_star": float(prior_summary["retained_alpha_at_q_star"]),
        "retained_alpha_rel_error_vs_target": float(prior_summary["retained_alpha_rel_error_vs_target"]),
        "alpha_beta_global_root_list": prior_summary["alpha_beta_global_root_list"],
        "nearest_alpha_beta_root_to_retained": float(prior_summary["nearest_alpha_beta_root_to_retained"]),
        "nearest_alpha_beta_root_rel_shift_vs_retained": float(
            prior_summary["nearest_alpha_beta_root_rel_shift_vs_retained"]
        ),
        "nearest_alpha_beta_root_charge_rel_error_vs_retained": float(
            prior_summary["nearest_alpha_beta_root_charge_rel_error_vs_retained"]
        ),
        "nearest_alpha_beta_root_energy_rel_error_vs_retained": float(
            prior_summary["nearest_alpha_beta_root_energy_rel_error_vs_retained"]
        ),
        "gate_a_updated_pack_trial2_alpha_beta_local_branch_unique_now": gate_a,
        "gate_b_updated_pack_trial2_alpha_beta_exact_route_unavailable_now": gate_b,
        "gate_c_updated_pack_trial2_energy_partition_promoted_primary_now": gate_c,
        "trial2_alpha_beta_curve_completed_now": trial2_alpha_beta_curve_completed_now,
        "trial2_energy_partition_primary_next_now": trial2_energy_partition_primary_next_now,
        "trial2_entropy_route_secondary_retained_now": trial2_entropy_route_secondary_retained_now,
        "selected_primary_completion_lane": "trial2_energy_partition_ratio",
        "selected_secondary_completion_lane": "trial2_entropy_route",
        "selected_reserve_completion_lane": "trial2_entropy_route",
        "selected_next_generation_route": "trial2_energy_partition_ratio",
        "recommended_next_route_or_none": "8.7.56.5583",
        "selected_followup_route": "trial2_energy_partition_ratio",
        "selected_followup_route_or_none": None,
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5581",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5583",
                "followup_route": "trial2_energy_partition_ratio",
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_alpha_beta_curve_gate_completed",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 alpha(beta) gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
