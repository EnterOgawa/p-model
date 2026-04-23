#!/usr/bin/env python3
"""Generate 8.7.56.5467-.5470 Trial-2 numerical closeout gate artifacts."""

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
        "8.7.56.5463-5466",
        "updated_pack_trial2_numerical_closeout_inventory_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5467-5470"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "numerical closeout gate / paper-sync refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_numerical_closeout_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_numerical_closeout_inventory_audited_target_free_blind_overlap_"
    "practical_close_primary_paper_sync_gate"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_numerical_closeout_practical_blind_overlap_numeric_close_"
    "paper_sync_completed_expert_share_primary_next"
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


# 関数: numerical closeout gate で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used by the Trial-2 numerical closeout gate."""
    return {
        "gate_a": "Gate A = Trial-2 numerical closeout inventory available now",
        "gate_b": "Gate B = practical blind-overlap numerical closeout paper-sync refreshed now",
        "gate_c": "Gate C = farther hybrid continuation reopen required now",
    }


# 関数: `.5467-.5470` を実行する。

def main() -> None:
    """Execute the Trial-2 numerical closeout gate / paper-sync refresh."""
    sign_base.require(PRIOR_AUDIT)
    prior_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]

    gate_a = bool(
        prior_summary["exact_trial2_numerical_closeout_inventory_available_now"]
        and prior_summary["trial2_practical_blind_overlap_numerical_closeout_available_now"]
    )
    gate_b = bool(
        gate_a
        and prior_summary[
            "updated_pack_trial2_numerical_closeout_paper_sync_followup_required_now"
        ]
    )
    gate_c = False
    trial2_exact_theorem_closeout_still_missing_now = not bool(
        prior_summary["trial2_exact_theorem_closeout_available_now"]
    )
    selected_extension_source_materialization_reserve_only_retained_now = True
    pack_update_required_now = bool(gate_b)

    rows = [
        sign_base.row(
            "gate_a_updated_pack_exact_trial2_numerical_closeout_inventory_available_now",
            "pass" if gate_a else "reject",
            "gate A updated-pack exact Trial-2 numerical closeout inventory available now",
            sign_base.truth(gate_a),
            "The inventory now fixes one honest numerical reading that keeps scalar-proxy success and theorem-side failure in the same pack.",
        ),
        sign_base.row(
            "gate_b_updated_pack_trial2_practical_blind_overlap_numerical_closeout_paper_sync_refreshed_now",
            "pass" if gate_b else "reject",
            "gate B updated-pack Trial-2 practical blind-overlap numerical closeout paper-sync refreshed now",
            sign_base.truth(gate_b),
            "The official reading now says Trial-2 closes numerically through the blind-overlap matching scale, while the exact target-free theorem remains open.",
        ),
        sign_base.row(
            "gate_c_farther_hybrid_continuation_reopen_required_now",
            "reject",
            "gate C farther hybrid continuation reopen required now",
            0.0,
            "Farther hybrid continuation still stays reserve-only because no genuinely new source or computation branch has appeared.",
        ),
        sign_base.row(
            "trial2_exact_theorem_closeout_still_missing_now",
            "pass" if trial2_exact_theorem_closeout_still_missing_now else "reject",
            "Trial-2 exact theorem closeout still missing now",
            sign_base.truth(trial2_exact_theorem_closeout_still_missing_now),
            "Paper sync should preserve the distinction between practical numerical closeout and a still-missing exact target-free derivation.",
        ),
        sign_base.row(
            "selected_extension_source_materialization_reserve_only_retained_now",
            "pass"
            if selected_extension_source_materialization_reserve_only_retained_now
            else "reject",
            "selected-extension source-materialization reserve only retained now",
            sign_base.truth(
                selected_extension_source_materialization_reserve_only_retained_now
            ),
            "The extra-q source-materialization lane remains exhausted and reserve-only after the numerical closeout reading is fixed.",
        ),
        sign_base.row(
            "pack_update_required_now",
            "pass" if pack_update_required_now else "reject",
            "updated-pack substantive route update required now",
            sign_base.truth(pack_update_required_now),
            "A substantive route update has happened: the live blocker is no longer whether Trial-2 has one honest numerical reading, but how to sync that reading into final share and declaration artifacts.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_blind_over_m0": float(prior_summary["q_blind_over_m0"]),
        "q_exact_over_m0": float(prior_summary["q_exact_over_m0"]),
        "q_star_over_m0": float(prior_summary["q_star_over_m0"]),
        "q_theory_over_m0": float(prior_summary["q_theory_over_m0"]),
        "q_exact_matches_prior_blind_crossing_abs_error": float(
            prior_summary["q_exact_matches_prior_blind_crossing_abs_error"]
        ),
        "delta_q_over_q_star": float(prior_summary["delta_q_over_q_star"]),
        "alpha_target": float(prior_summary["alpha_target"]),
        "alpha_at_q_star": float(prior_summary["alpha_at_q_star"]),
        "relative_residual_at_q_star": float(
            prior_summary["relative_residual_at_q_star"]
        ),
        "alpha_exact_at_q_theory": float(prior_summary["alpha_exact_at_q_theory"]),
        "q_theory_diagnostic": prior_summary["q_theory_diagnostic"],
        "best_extra_label_vs_alpha_target": prior_summary[
            "best_extra_label_vs_alpha_target"
        ],
        "best_extra_alpha_target_residual": float(
            prior_summary["best_extra_alpha_target_residual"]
        ),
        "best_extra_q_exact_gap": float(prior_summary["best_extra_q_exact_gap"]),
        "gate_a_updated_pack_exact_trial2_numerical_closeout_inventory_available_now": gate_a,
        "gate_b_updated_pack_trial2_practical_blind_overlap_numerical_closeout_paper_sync_refreshed_now": gate_b,
        "gate_c_farther_hybrid_continuation_reopen_required_now": gate_c,
        "trial2_practical_blind_overlap_numerical_closeout_available_now": bool(
            prior_summary["trial2_practical_blind_overlap_numerical_closeout_available_now"]
        ),
        "trial2_exact_theorem_closeout_still_missing_now": trial2_exact_theorem_closeout_still_missing_now,
        "selected_extension_source_materialization_reserve_only_retained_now": selected_extension_source_materialization_reserve_only_retained_now,
        "selected_primary_completion_lane": (
            "updated_pack_trial2_numerical_closeout_expert_share_sync"
        ),
        "selected_secondary_completion_lane": (
            "updated_pack_trial2_numerical_closeout_final_declaration_gate"
        ),
        "selected_reserve_completion_lane": (
            "farther_hybrid_reserve_only_until_new_independent_source_exists"
        ),
        "selected_next_generation_route": (
            "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_"
            "trial2_numerical_closeout_expert_share_sync"
        ),
        "recommended_next_route_or_none": "8.7.56.5471",
        "selected_followup_route": (
            "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_"
            "trial2_numerical_closeout_final_declaration_gate"
        ),
        "selected_followup_route_or_none": "8.7.56.5475",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5469",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_audit": sign_base.display_path(PRIOR_AUDIT)},
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5471",
                "followup_route": "8.7.56.5475",
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_numerical_closeout_gate_declared",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 numerical closeout gate completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# 関数: CLI entrypoint から gate refresh を実行する。

if __name__ == "__main__":
    main()
