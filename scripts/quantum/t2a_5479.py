#!/usr/bin/env python3
"""Generate 8.7.56.5479-.5482 Trial-2 conditional reopen inventory artifacts."""

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
        "8.7.56.5475-5478",
        "updated_pack_trial2_practical_numeric_closeout_final_declaration",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
SOURCE_MATERIALIZATION_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5459-5462",
        "updated_pack_selected_extension_independent_extra_q_range_source_materialization_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
INVENTORY_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "63_trial2_numeric_alpha_vector_qball_conditional_reopen_inventory.md"
)

STEP_TAG = "8.7.56.5479-5482"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "conditional reopen inventory audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_conditional_reopen_inventory_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_practical_numeric_closeout_final_declaration_completed_"
    "conditional_reopen_only_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_conditional_reopen_inventory_audited_no_current_trigger_"
    "hold_gate_next"
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


# 関数: inventory で使う式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used by the conditional reopen inventory audit."""
    return {
        "candidate_a": "reopen trigger A = genuinely new target-free theorem route",
        "candidate_b": "reopen trigger B = genuinely new selected-extension-native source / computation branch",
        "inadmissible_c": "legacy Phase-3 sideband carry-over is not an admissible trigger",
        "inadmissible_d": "farther hybrid continuation is reserve-only and not an unconditional trigger",
    }


# 関数: note が expected inventory claims を含むかを確認する。

def note_contains_inventory(text: str) -> bool:
    """Return whether one note carries the expected conditional inventory."""
    patterns = (
        "genuinely new target-free theorem route",
        "genuinely new selected-extension-native source / computation branch",
        "legacy Phase-3 sideband carry-over",
        "farther hybrid continuation",
        "current pack has no currently admissible reopen trigger",
    )
    return all(pattern in text for pattern in patterns)


# 関数: `.5479-.5482` を実行する。

def main() -> None:
    """Execute the Trial-2 conditional reopen inventory audit."""
    for path in (PRIOR_GATE, SOURCE_MATERIALIZATION_GATE, INVENTORY_NOTE):
        sign_base.require(path)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    source_summary = sign_base.read_json(SOURCE_MATERIALIZATION_GATE)["summary"]
    note_text = sign_base.read_text(INVENTORY_NOTE)

    note_available = note_contains_inventory(note_text)
    target_free_route_currently_materialized_now = False
    source_or_computation_branch_currently_materialized_now = False
    legacy_phase3_sideband_reopen_trigger_admissible_now = False
    farther_hybrid_unconditional_reopen_admissible_now = False
    conditional_reopen_inventory_available_now = bool(
        prior_summary["no_unconditional_next_route_now"]
        and prior_summary[
            "future_reopen_requires_new_target_free_theorem_route_or_new_independent_source_now"
        ]
        and source_summary[
            "gate_a_updated_pack_exact_selected_extension_independent_extra_q_range_source_materialization_negative_closeout_available_now"
        ]
        and note_available
    )
    no_current_trigger_detected_now = bool(
        conditional_reopen_inventory_available_now
        and not target_free_route_currently_materialized_now
        and not source_or_computation_branch_currently_materialized_now
        and not legacy_phase3_sideband_reopen_trigger_admissible_now
        and not farther_hybrid_unconditional_reopen_admissible_now
    )
    hold_gate_followup_required_now = bool(no_current_trigger_detected_now)

    rows = [
        sign_base.row(
            "exact_trial2_conditional_reopen_inventory_note_available_now",
            "pass" if note_available else "reject",
            "exact Trial-2 conditional reopen inventory note available now",
            sign_base.truth(note_available),
            "The dedicated inventory note exists and enumerates admissible versus inadmissible reopen triggers.",
        ),
        sign_base.row(
            "exact_trial2_conditional_reopen_inventory_available_now",
            "pass" if conditional_reopen_inventory_available_now else "reject",
            "exact Trial-2 conditional reopen inventory available now",
            sign_base.truth(conditional_reopen_inventory_available_now),
            "The final declaration, exhausted source-materialization lane, and explicit inventory note are now bundled into one reopen audit.",
        ),
        sign_base.row(
            "trial2_new_target_free_theorem_route_currently_materialized_now",
            "pass" if target_free_route_currently_materialized_now else "reject",
            "Trial-2 new target-free theorem route currently materialized now",
            sign_base.truth(target_free_route_currently_materialized_now),
            "Reject means no new theorem route has appeared beyond the already exhausted Route B/A/D/C stack.",
        ),
        sign_base.row(
            "trial2_new_selected_extension_native_source_or_computation_branch_currently_materialized_now",
            "pass" if source_or_computation_branch_currently_materialized_now else "reject",
            "Trial-2 new selected-extension-native source or computation branch currently materialized now",
            sign_base.truth(source_or_computation_branch_currently_materialized_now),
            "Reject means no new helper-backed selected-extension-native rescue exists beyond the exhausted extra-q/source-materialization stack.",
        ),
        sign_base.row(
            "trial2_legacy_phase3_sideband_reopen_trigger_admissible_now",
            "pass" if legacy_phase3_sideband_reopen_trigger_admissible_now else "reject",
            "Trial-2 legacy Phase-3 sideband reopen trigger admissible now",
            sign_base.truth(legacy_phase3_sideband_reopen_trigger_admissible_now),
            "Reject means near-target legacy sidebands remain inadmissible as reopen triggers.",
        ),
        sign_base.row(
            "trial2_farther_hybrid_unconditional_reopen_admissible_now",
            "pass" if farther_hybrid_unconditional_reopen_admissible_now else "reject",
            "Trial-2 farther hybrid unconditional reopen admissible now",
            sign_base.truth(farther_hybrid_unconditional_reopen_admissible_now),
            "Reject means farther hybrid remains reserve-only and cannot justify an unconditional reopen.",
        ),
        sign_base.row(
            "trial2_conditional_reopen_no_current_trigger_detected_now",
            "pass" if no_current_trigger_detected_now else "reject",
            "Trial-2 conditional reopen no current trigger detected now",
            sign_base.truth(no_current_trigger_detected_now),
            "The honest current-pack reading is that no admissible reopen trigger is presently materialized.",
        ),
        sign_base.row(
            "updated_pack_trial2_conditional_hold_gate_followup_required_now",
            "pass" if hold_gate_followup_required_now else "reject",
            "updated-pack Trial-2 conditional hold gate followup required now",
            sign_base.truth(hold_gate_followup_required_now),
            "The only remaining task is to promote the hold reading itself into one final gate without inventing a new branch of physics.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "q_blind_over_m0": float(prior_summary["q_blind_over_m0"]),
        "q_exact_over_m0": float(prior_summary["q_exact_over_m0"]),
        "q_star_over_m0": float(prior_summary["q_star_over_m0"]),
        "delta_q_over_q_star": float(prior_summary["delta_q_over_q_star"]),
        "exact_trial2_conditional_reopen_inventory_note_available_now": note_available,
        "exact_trial2_conditional_reopen_inventory_available_now": (
            conditional_reopen_inventory_available_now
        ),
        "trial2_new_target_free_theorem_route_currently_materialized_now": (
            target_free_route_currently_materialized_now
        ),
        "trial2_new_selected_extension_native_source_or_computation_branch_currently_materialized_now": (
            source_or_computation_branch_currently_materialized_now
        ),
        "trial2_legacy_phase3_sideband_reopen_trigger_admissible_now": (
            legacy_phase3_sideband_reopen_trigger_admissible_now
        ),
        "trial2_farther_hybrid_unconditional_reopen_admissible_now": (
            farther_hybrid_unconditional_reopen_admissible_now
        ),
        "trial2_conditional_reopen_no_current_trigger_detected_now": (
            no_current_trigger_detected_now
        ),
        "updated_pack_trial2_conditional_hold_gate_followup_required_now": (
            hold_gate_followup_required_now
        ),
        "selected_primary_completion_lane": (
            "updated_pack_trial2_conditional_reopen_hold_gate"
        ),
        "selected_secondary_completion_lane": "conditional_reopen_only",
        "selected_reserve_completion_lane": (
            "farther_hybrid_reserve_only_until_genuinely_new_route_exists"
        ),
        "selected_next_generation_route": (
            "trial2_numeric_alpha_vector_qball_form_factor_updated_pack_"
            "trial2_conditional_reopen_hold_gate"
        ),
        "recommended_next_route_or_none": "8.7.56.5483",
        "selected_followup_route": "conditional_reopen_only",
        "selected_followup_route_or_none": None,
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5481",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "source_materialization_gate": sign_base.display_path(
                    SOURCE_MATERIALIZATION_GATE
                ),
                "inventory_note": sign_base.display_path(INVENTORY_NOTE),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route": "8.7.56.5483",
                "followup_route": None,
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_conditional_reopen_inventory_audited",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {"formulae": build_formulae()},
    )
    declaration_paths = write_artifact("declaration_gate", payload)
    write_artifact("route_sync", payload)
    print(f"[done] {STEP_TAG} Trial-2 conditional reopen inventory completed")
    print(f"[done] declaration: {declaration_paths['json']}")


# 関数: CLI entrypoint から conditional reopen inventory を実行する。

if __name__ == "__main__":
    main()
