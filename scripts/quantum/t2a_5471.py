#!/usr/bin/env python3
"""Generate 8.7.56.5471-.5474 Trial-2 numerical closeout expert-share sync artifacts."""

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
        "8.7.56.5467-5470",
        "updated_pack_trial2_numerical_closeout_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
EXPERT_SHARE = (
    ROOT
    / "doc"
    / "quantum"
    / "62_trial2_numeric_alpha_vector_qball_practical_closeout_expert_share.md"
)
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
CURRENT_PROBLEM = ROOT / "doc" / "quantum" / "34_trial2_numeric_alpha_current_problem.md"
CURRENT_STATUS = ROOT / "doc" / "quantum" / "36_trial2_numeric_alpha_current_status.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

STEP_TAG = "8.7.56.5471-5474"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "numerical closeout expert-share sync"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_numerical_closeout_expert_share_sync",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_numerical_closeout_practical_blind_overlap_numeric_close_"
    "paper_sync_completed_expert_share_primary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_numerical_closeout_expert_share_sync_completed_final_"
    "declaration_primary_next"
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


# 関数: expert-share sync で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used by the Trial-2 numerical closeout expert-share sync."""
    return {
        "limited_claim": (
            "practical blind-overlap numerical closeout available now AND "
            "exact target-free theorem closeout unavailable now"
        ),
        "share_scope": (
            "sync note/doc surfaces to the limited claim without promoting it "
            "into an exact theorem closeout"
        ),
        "next_gate": "final declaration = conditional reopen only",
    }


# 関数: 期待される limited-claim 表記が含まれるかを確認する。

def has_limited_claim(text: str) -> bool:
    """Return whether one text carries the practical-vs-exact split."""
    return (
        "practical blind-overlap numerical closeout" in text
        and "exact target-free theorem closeout" in text
    )


# 関数: `.5471-.5474` を実行する。

def main() -> None:
    """Execute the Trial-2 numerical closeout expert-share sync."""
    for path in (
        PRIOR_GATE,
        EXPERT_SHARE,
        STATUS,
        ROADMAP,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        PART5,
    ):
        sign_base.require(path)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    share_text = sign_base.read_text(EXPERT_SHARE)
    status_text = sign_base.read_text(STATUS)
    roadmap_text = sign_base.read_text(ROADMAP)
    problem_text = sign_base.read_text(CURRENT_PROBLEM)
    current_status_text = sign_base.read_text(CURRENT_STATUS)
    part5_text = sign_base.read_text(PART5)

    note_available = bool(
        has_limited_claim(share_text)
        and "What Is Not Claimed" in share_text
        and "Reopen Conditions" in share_text
    )
    doc_surfaces_synced = all(
        has_limited_claim(text)
        for text in (
            status_text,
            roadmap_text,
            problem_text,
            current_status_text,
            part5_text,
        )
    )
    expert_share_sync_available = bool(
        prior_summary["trial2_practical_blind_overlap_numerical_closeout_available_now"]
        and prior_summary["trial2_exact_theorem_closeout_still_missing_now"]
        and note_available
        and doc_surfaces_synced
    )
    limited_claim_synced = bool(expert_share_sync_available)
    exact_target_free_theorem_closeout_still_unavailable_now = bool(
        prior_summary["trial2_exact_theorem_closeout_still_missing_now"]
    )
    updated_pack_trial2_numerical_closeout_final_declaration_followup_required_now = (
        bool(limited_claim_synced)
    )
    same_schema_replay_detected_now = False

    rows = [
        sign_base.row(
            "exact_trial2_numerical_closeout_expert_share_note_available_now",
            "pass" if note_available else "reject",
            "exact Trial-2 numerical closeout expert-share note available now",
            sign_base.truth(note_available),
            "The dedicated expert-share note exists and explicitly separates practical numerical closeout from still-missing exact theorem closeout.",
        ),
        sign_base.row(
            "exact_trial2_numerical_closeout_expert_share_sync_available_now",
            "pass" if expert_share_sync_available else "reject",
            "exact Trial-2 numerical closeout expert-share sync available now",
            sign_base.truth(expert_share_sync_available),
            "Status, roadmap, current-problem, current-status, Part V, and the expert-share note all carry the same limited claim.",
        ),
        sign_base.row(
            "trial2_practical_blind_overlap_numerical_closeout_limited_claim_synced_now",
            "pass" if limited_claim_synced else "reject",
            "Trial-2 practical blind-overlap numerical closeout limited claim synced now",
            sign_base.truth(limited_claim_synced),
            "The public-canonical reading is now synced as a limited practical closeout rather than an exact theorem claim.",
        ),
        sign_base.row(
            "trial2_exact_target_free_theorem_closeout_still_unavailable_now",
            "pass"
            if exact_target_free_theorem_closeout_still_unavailable_now
            else "reject",
            "Trial-2 exact target-free theorem closeout still unavailable now",
            sign_base.truth(
                exact_target_free_theorem_closeout_still_unavailable_now
            ),
            "The expert-share sync preserves the honest theorem-side boundary instead of weakening it.",
        ),
        sign_base.row(
            "updated_pack_trial2_numerical_closeout_final_declaration_followup_required_now",
            "pass"
            if updated_pack_trial2_numerical_closeout_final_declaration_followup_required_now
            else "reject",
            "updated-pack Trial-2 numerical closeout final declaration followup required now",
            sign_base.truth(
                updated_pack_trial2_numerical_closeout_final_declaration_followup_required_now
            ),
            "Once the limited claim is synced, the only remaining unconditional task is to declare the lane complete with conditional reopen only.",
        ),
        sign_base.row(
            "updated_pack_same_schema_trial2_numerical_closeout_replay_detected_now",
            "pass" if same_schema_replay_detected_now else "reject",
            "updated-pack same-schema Trial-2 numerical closeout replay detected now",
            sign_base.truth(same_schema_replay_detected_now),
            "False means this branch did not reopen exhausted computation routes and only synced the limited closeout statement.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_blind_over_m0": float(prior_summary["q_blind_over_m0"]),
        "q_exact_over_m0": float(prior_summary["q_exact_over_m0"]),
        "q_star_over_m0": float(prior_summary["q_star_over_m0"]),
        "delta_q_over_q_star": float(prior_summary["delta_q_over_q_star"]),
        "alpha_target": float(prior_summary["alpha_target"]),
        "alpha_at_q_star": float(prior_summary["alpha_at_q_star"]),
        "relative_residual_at_q_star": float(
            prior_summary["relative_residual_at_q_star"]
        ),
        "exact_trial2_numerical_closeout_expert_share_note_available_now": (
            note_available
        ),
        "exact_trial2_numerical_closeout_expert_share_sync_available_now": (
            expert_share_sync_available
        ),
        "trial2_practical_blind_overlap_numerical_closeout_limited_claim_synced_now": (
            limited_claim_synced
        ),
        "trial2_exact_target_free_theorem_closeout_still_unavailable_now": (
            exact_target_free_theorem_closeout_still_unavailable_now
        ),
        "updated_pack_trial2_numerical_closeout_final_declaration_followup_required_now": (
            updated_pack_trial2_numerical_closeout_final_declaration_followup_required_now
        ),
        "updated_pack_same_schema_trial2_numerical_closeout_replay_detected_now": (
            same_schema_replay_detected_now
        ),
        "selected_primary_completion_lane": (
            "updated_pack_trial2_numerical_closeout_final_declaration_gate"
        ),
        "selected_secondary_completion_lane": "conditional_reopen_only_after_final_declaration",
        "selected_reserve_completion_lane": (
            "farther_hybrid_reserve_only_until_new_independent_source_exists"
        ),
        "selected_next_generation_route": (
            "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_"
            "trial2_numerical_closeout_final_declaration_gate"
        ),
        "recommended_next_route_or_none": "8.7.56.5475",
        "selected_followup_route": "conditional_reopen_only",
        "selected_followup_route_or_none": None,
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5473",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "expert_share": sign_base.display_path(EXPERT_SHARE),
                "status": sign_base.display_path(STATUS),
                "roadmap": sign_base.display_path(ROADMAP),
                "current_problem": sign_base.display_path(CURRENT_PROBLEM),
                "current_status": sign_base.display_path(CURRENT_STATUS),
                "part5": sign_base.display_path(PART5),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5475",
                "followup_route": None,
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_numerical_closeout_expert_share_synced",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 numerical closeout expert-share sync completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# 関数: CLI entrypoint から expert-share sync を実行する。

if __name__ == "__main__":
    main()
