#!/usr/bin/env python3
"""Generate 8.7.56.5571-.5574 Trial-2 direct-alpha self-consistent gate artifacts."""

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
        "8.7.56.5567-5570",
        "updated_pack_trial2_direct_alpha_self_consistent_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5571-5574"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "direct-alpha self-consistent gate / alpha-beta refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_direct_alpha_self_consistent_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_direct_alpha_self_consistent_target_free_mismatch_"
    "alpha_beta_followup_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_direct_alpha_self_consistent_negative_closeout_completed_"
    "alpha_beta_primary_energy_partition_secondary_entropy_reserve_next"
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
    """Return formulas used by the direct-alpha self-consistent gate."""
    return {
        "gate_a": "Gate A = self-consistent root exists uniquely now",
        "gate_b": "Gate B = self-consistent route closes negatively now",
        "gate_c": "Gate C = alpha(beta) promoted as the next primary direct-alpha route",
    }


# 関数: `.5571-.5574` を実行する。

def main() -> None:
    """Execute the Trial-2 direct-alpha self-consistent gate."""
    sign_base.require(PRIOR_GATE)
    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]

    gate_a = bool(
        prior_summary["exact_trial2_direct_alpha_self_consistent_root_exists_now"]
        and prior_summary["exact_trial2_direct_alpha_self_consistent_root_unique_now"]
    )
    gate_b = bool(
        prior_summary[
            "exact_trial2_direct_alpha_self_consistent_route_negative_closeout_available_now"
        ]
    )
    gate_c = bool(
        prior_summary["updated_pack_trial2_alpha_beta_curve_primary_followup_required_now"]
    )
    trial2_direct_alpha_self_consistent_negative_closeout_completed_now = bool(
        gate_a and gate_b
    )
    trial2_alpha_beta_curve_primary_next_now = bool(
        trial2_direct_alpha_self_consistent_negative_closeout_completed_now and gate_c
    )
    trial2_energy_partition_secondary_retained_now = True
    trial2_entropy_route_reserve_retained_now = True

    rows = [
        sign_base.row(
            "gate_a_updated_pack_trial2_direct_alpha_self_consistent_root_unique_now",
            "pass" if gate_a else "reject",
            "gate A updated-pack Trial-2 direct-alpha self-consistent root unique now",
            sign_base.truth(gate_a),
            "The direct-alpha fixed-point equation is only worth closing if it produces one unique retained root.",
        ),
        sign_base.row(
            "gate_b_updated_pack_trial2_direct_alpha_self_consistent_negative_closeout_completed_now",
            "pass" if gate_b else "reject",
            "gate B updated-pack Trial-2 direct-alpha self-consistent negative closeout completed now",
            sign_base.truth(gate_b),
            "The retained root exists but does not select q_exact, so the self-consistent route closes negatively.",
        ),
        sign_base.row(
            "gate_c_updated_pack_trial2_alpha_beta_curve_promoted_primary_now",
            "pass" if gate_c else "reject",
            "gate C updated-pack Trial-2 alpha(beta) curve promoted primary now",
            sign_base.truth(gate_c),
            "Once self-consistent direct-alpha fails, alpha(beta) becomes the next honest direct-alpha followup.",
        ),
        sign_base.row(
            "trial2_direct_alpha_self_consistent_negative_closeout_completed_now",
            "pass" if trial2_direct_alpha_self_consistent_negative_closeout_completed_now else "reject",
            "Trial-2 direct-alpha self-consistent negative closeout completed now",
            sign_base.truth(trial2_direct_alpha_self_consistent_negative_closeout_completed_now),
            "The route is now closed in the official machine-readable chain.",
        ),
        sign_base.row(
            "trial2_alpha_beta_curve_primary_next_now",
            "pass" if trial2_alpha_beta_curve_primary_next_now else "reject",
            "Trial-2 alpha(beta) curve primary next now",
            sign_base.truth(trial2_alpha_beta_curve_primary_next_now),
            "The next honest blocker is to determine whether alpha itself can be read as one dimensionless function of beta.",
        ),
        sign_base.row(
            "trial2_energy_partition_secondary_retained_now",
            "pass" if trial2_energy_partition_secondary_retained_now else "reject",
            "Trial-2 energy-partition secondary retained now",
            sign_base.truth(trial2_energy_partition_secondary_retained_now),
            "Energy-ratio reading remains a low-cost secondary route if alpha(beta) dead-ends honestly.",
        ),
        sign_base.row(
            "trial2_entropy_route_reserve_retained_now",
            "pass" if trial2_entropy_route_reserve_retained_now else "reject",
            "Trial-2 entropy route reserve retained now",
            sign_base.truth(trial2_entropy_route_reserve_retained_now),
            "Entropy remains reserve-only until the lower-cost direct-alpha candidates are exhausted.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_exact_over_m0": float(prior_summary["q_exact_over_m0"]),
        "q_blind_over_m0": float(prior_summary["q_blind_over_m0"]),
        "q_star_over_m0": float(prior_summary["q_star_over_m0"]),
        "primary_q_self_consistent_over_m0": float(prior_summary["primary_q_self_consistent_over_m0"]),
        "q_self_consistent_rel_error_vs_q_exact": float(prior_summary["q_self_consistent_rel_error_vs_q_exact"]),
        "gate_a_updated_pack_trial2_direct_alpha_self_consistent_root_unique_now": gate_a,
        "gate_b_updated_pack_trial2_direct_alpha_self_consistent_negative_closeout_completed_now": gate_b,
        "gate_c_updated_pack_trial2_alpha_beta_curve_promoted_primary_now": gate_c,
        "trial2_direct_alpha_self_consistent_negative_closeout_completed_now": (
            trial2_direct_alpha_self_consistent_negative_closeout_completed_now
        ),
        "trial2_alpha_beta_curve_primary_next_now": trial2_alpha_beta_curve_primary_next_now,
        "trial2_energy_partition_secondary_retained_now": trial2_energy_partition_secondary_retained_now,
        "trial2_entropy_route_reserve_retained_now": trial2_entropy_route_reserve_retained_now,
        "selected_primary_completion_lane": "trial2_alpha_beta_curve",
        "selected_secondary_completion_lane": "trial2_energy_partition_ratio",
        "selected_reserve_completion_lane": "trial2_entropy_route",
        "selected_next_generation_route": "trial2_alpha_beta_curve",
        "recommended_next_route_or_none": "8.7.56.5575",
        "selected_followup_route": "trial2_alpha_beta_curve",
        "selected_followup_route_or_none": None,
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5573",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5575",
                "followup_route": "trial2_alpha_beta_curve",
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_direct_alpha_self_consistent_gate_completed",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 direct-alpha self-consistent gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


if __name__ == "__main__":
    main()
