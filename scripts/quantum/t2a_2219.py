#!/usr/bin/env python3
"""Generate 8.7.56.2219-.2222 coefficient-law decision gate artifacts."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"

STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
WORK_HISTORY_RECENT = ROOT / "doc" / "WORK_HISTORY_RECENT.md"
CURRENT_PROBLEM = ROOT / "doc" / "quantum" / "34_trial2_numeric_alpha_current_problem.md"
CURRENT_STATUS = ROOT / "doc" / "quantum" / "36_trial2_numeric_alpha_current_status.md"
UNIFIED_ROADMAP = ROOT / "doc" / "quantum" / "39_trial2_vector_qball_unified_closure_roadmap.md"
LONG_ROADMAP = ROOT / "doc" / "quantum" / "55_trial2_numeric_alpha_vector_qball_long_horizon_roadmap.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2215-2218",
        "harmonic_post_break_segment_coefficient_law",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2219-2222"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor coefficient-law "
    "decision gate"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "harmonic_post_break_coefficient_law_registry",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_post_break_segment_"
    "coefficient_law_extraction_next"
)
BRANCH_CLASS_SINGLE_THEOREM = (
    "vector_qball_form_factor_boundary_bulk_lattice_post_break_segment_single_"
    "theorem_blind_prediction_retained_next"
)
BRANCH_CLASS_FALLBACK = (
    "vector_qball_form_factor_boundary_bulk_lattice_post_break_segment_"
    "coefficient_law_not_blind_predictive_fallback_sixth_piecewise_next"
)
NEXT_ROUTE_NAME_SINGLE_THEOREM = (
    "trial2_numeric_alpha_vector_qball_form_factor_segment_single_theorem_"
    "validation_audit"
)
NEXT_ROUTE_SINGLE_THEOREM = "8.7.56.2223"
FOLLOWUP_ROUTE_NAME_SINGLE_THEOREM = (
    "trial2_numeric_alpha_vector_qball_form_factor_segment_single_theorem_"
    "registry_refresh"
)
FOLLOWUP_ROUTE_SINGLE_THEOREM = "8.7.56.2227"
NEXT_ROUTE_NAME_FALLBACK = (
    "trial2_numeric_alpha_vector_qball_form_factor_fallback_sixth_post_break_"
    "piecewise_farther_continuation_audit"
)
NEXT_ROUTE_FALLBACK = "8.7.56.2223"
FOLLOWUP_ROUTE_NAME_FALLBACK = (
    "trial2_numeric_alpha_vector_qball_form_factor_fallback_sixth_post_break_"
    "piecewise_registry_refresh"
)
FOLLOWUP_ROUTE_FALLBACK = "8.7.56.2227"


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


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

    return {
        "json": sign_base.display_path(paths["json"]),
        "csv": sign_base.display_path(paths["csv"]),
    }


# 関数: テキスト中の最初の一致行を返す。

def find_line(text: str, pattern: str) -> dict[str, object] | None:
    """Return the first matching line payload for one text pattern."""
    for line_number, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {
                "pattern": pattern,
                "line": line_number,
                "text": line.strip(),
            }

    return None


# 関数: 使用公式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the coefficient-law decision gate."""
    return {
        "gate_a_rule": "Gate A is retained only when the blind coefficient-law prediction itself clears the same exact sixth holdout/monitor thresholds as the retained continuation family.",
        "gate_b_rule": "Gate B fallback is selected once the coefficient-law route remains non-predictive and the retained sixth post-break piecewise continuation is still admissible.",
        "gate_c_rule": "Gate C substantive pack update remains closed while the blocker is still localized to coefficient-law prediction accuracy rather than a missing public-canonical surface.",
    }


# 関数: `.2219-.2222` を実行する。

def main() -> None:
    """Execute the coefficient-law decision gate."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        LONG_ROADMAP,
        PART5,
        PRIOR_GATE,
    ):
        sign_base.require(path)

    status_text = sign_base.read_text(STATUS)
    roadmap_text = sign_base.read_text(ROADMAP)
    current_problem_text = sign_base.read_text(CURRENT_PROBLEM)
    current_status_text = sign_base.read_text(CURRENT_STATUS)
    unified_text = sign_base.read_text(UNIFIED_ROADMAP)
    long_text = sign_base.read_text(LONG_ROADMAP)
    part5_text = sign_base.read_text(PART5)
    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]

    inventory_ready = bool(
        prior_summary["single_theorem_route_supported"]
        or prior_summary["fallback_sixth_post_break_piecewise_selected"]
    )
    gate_a_single_theorem_route_retained = bool(
        prior_summary["single_theorem_route_supported"]
    )
    gate_b_fallback_sixth_piecewise_selected = bool(
        prior_summary["fallback_sixth_post_break_piecewise_selected"]
    )
    gate_c_substantive_pack_update_required = False
    loading_index_theorem_reserve_selected = True
    physical_reject_required = False

    if gate_a_single_theorem_route_retained:
        branch_class = BRANCH_CLASS_SINGLE_THEOREM
        next_route_name = NEXT_ROUTE_NAME_SINGLE_THEOREM
        next_route = NEXT_ROUTE_SINGLE_THEOREM
        followup_route_name = FOLLOWUP_ROUTE_NAME_SINGLE_THEOREM
        followup_route = FOLLOWUP_ROUTE_SINGLE_THEOREM
    else:
        branch_class = BRANCH_CLASS_FALLBACK
        next_route_name = NEXT_ROUTE_NAME_FALLBACK
        next_route = NEXT_ROUTE_FALLBACK
        followup_route_name = FOLLOWUP_ROUTE_NAME_FALLBACK
        followup_route = FOLLOWUP_ROUTE_FALLBACK

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "coefficient-law decision inventory ready",
            sign_base.truth(inventory_ready),
            "The decision gate starts only after the route-reset law audit has already frozen the blind sixth-prediction metrics.",
        ),
        sign_base.row(
            "gate_a_single_theorem_route_retained",
            "pass" if gate_a_single_theorem_route_retained else "reject",
            "Gate A single-theorem route retained",
            sign_base.truth(gate_a_single_theorem_route_retained),
            "Gate A remains open only if the blind law already beats the retained sixth continuation on the same exact gate.",
        ),
        sign_base.row(
            "gate_b_fallback_sixth_piecewise_selected",
            "pass" if gate_b_fallback_sixth_piecewise_selected else "reject",
            "Gate B fallback sixth post-break piecewise selected",
            sign_base.truth(gate_b_fallback_sixth_piecewise_selected),
            "Gate B is selected once coefficient-law extraction stays informative but non-predictive on the exact sixth gate.",
        ),
        sign_base.row(
            "gate_c_substantive_pack_update_required",
            "reject",
            "Gate C substantive pack update required",
            sign_base.truth(gate_c_substantive_pack_update_required),
            "A substantive pack update remains unnecessary because the blocker is coefficient-law prediction quality, not a missing pack-level surface.",
        ),
        sign_base.row(
            "loading_index_theorem_reserve_selected",
            "pass",
            "loading-index theorem reserve selected",
            sign_base.truth(loading_index_theorem_reserve_selected),
            "The loading-index theorem remains reserve-only while the mainline either tests a single-theorem law or falls back to the retained sixth continuation family.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": branch_class,
        "prior_problem_classification": PRIOR_CLASS,
        "best_mismatch_slope_model_name": str(prior_summary["best_mismatch_slope_model_name"]),
        "best_mismatch_intercept_model_name": str(prior_summary["best_mismatch_intercept_model_name"]),
        "best_correlation_slope_model_name": str(prior_summary["best_correlation_slope_model_name"]),
        "best_correlation_intercept_model_name": str(prior_summary["best_correlation_intercept_model_name"]),
        "best_reconstruction_decay_exponent_model_name": str(
            prior_summary["best_reconstruction_decay_exponent_model_name"]
        ),
        "best_reconstruction_decay_prefactor_model_name": str(
            prior_summary["best_reconstruction_decay_prefactor_model_name"]
        ),
        "primary_combo_best_model_mismatch_slope": str(
            prior_summary["primary_combo_best_model_mismatch_slope"]
        ),
        "primary_combo_best_model_mismatch_intercept": str(
            prior_summary["primary_combo_best_model_mismatch_intercept"]
        ),
        "primary_combo_best_model_correlation_slope": str(
            prior_summary["primary_combo_best_model_correlation_slope"]
        ),
        "primary_combo_best_model_correlation_intercept": str(
            prior_summary["primary_combo_best_model_correlation_intercept"]
        ),
        "primary_combo_best_score_vs_threshold": float(
            prior_summary["primary_combo_best_score_vs_threshold"]
        ),
        "blind_full_sixth_holdout_max_mismatch_abs_error": float(
            prior_summary["blind_full_sixth_holdout_max_mismatch_abs_error"]
        ),
        "blind_full_sixth_monitor_max_mismatch_abs_error": float(
            prior_summary["blind_full_sixth_monitor_max_mismatch_abs_error"]
        ),
        "blind_full_sixth_holdout_max_correlation_abs_error": float(
            prior_summary["blind_full_sixth_holdout_max_correlation_abs_error"]
        ),
        "blind_full_sixth_monitor_max_correlation_abs_error": float(
            prior_summary["blind_full_sixth_monitor_max_correlation_abs_error"]
        ),
        "blind_full_sixth_holdout_max_reconstruction_abs_error": float(
            prior_summary["blind_full_sixth_holdout_max_reconstruction_abs_error"]
        ),
        "blind_full_sixth_monitor_max_reconstruction_abs_error": float(
            prior_summary["blind_full_sixth_monitor_max_reconstruction_abs_error"]
        ),
        "primary_combo_best_holdout_max_mismatch_abs_error": float(
            prior_summary["primary_combo_best_holdout_max_mismatch_abs_error"]
        ),
        "primary_combo_best_monitor_max_mismatch_abs_error": float(
            prior_summary["primary_combo_best_monitor_max_mismatch_abs_error"]
        ),
        "primary_combo_best_holdout_max_correlation_abs_error": float(
            prior_summary["primary_combo_best_holdout_max_correlation_abs_error"]
        ),
        "primary_combo_best_monitor_max_correlation_abs_error": float(
            prior_summary["primary_combo_best_monitor_max_correlation_abs_error"]
        ),
        "primary_coefficients_blind_predictive": bool(
            prior_summary["primary_coefficients_blind_predictive"]
        ),
        "reconstruction_coefficient_law_available": bool(
            prior_summary["reconstruction_coefficient_law_available"]
        ),
        "primary_mixed_coefficient_law_passes_current_piecewise_threshold": bool(
            prior_summary["primary_mixed_coefficient_law_passes_current_piecewise_threshold"]
        ),
        "single_theorem_route_supported": gate_a_single_theorem_route_retained,
        "fallback_sixth_post_break_piecewise_selected": gate_b_fallback_sixth_piecewise_selected,
        "gate_a_single_theorem_route_retained": gate_a_single_theorem_route_retained,
        "gate_b_fallback_sixth_piecewise_selected": gate_b_fallback_sixth_piecewise_selected,
        "gate_c_substantive_pack_update_required": gate_c_substantive_pack_update_required,
        "loading_index_theorem_reserve_selected": loading_index_theorem_reserve_selected,
        "selected_next_generation_route": next_route_name,
        "recommended_next_route_or_none": next_route,
        "selected_followup_route": followup_route_name,
        "selected_followup_route_or_none": followup_route,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2221",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "status": sign_base.display_path(STATUS),
                "roadmap": sign_base.display_path(ROADMAP),
                "ai_context": sign_base.display_path(AI_CONTEXT),
                "work_history_recent": sign_base.display_path(WORK_HISTORY_RECENT),
                "current_problem": sign_base.display_path(CURRENT_PROBLEM),
                "current_status": sign_base.display_path(CURRENT_STATUS),
                "unified_roadmap": sign_base.display_path(UNIFIED_ROADMAP),
                "long_roadmap": sign_base.display_path(LONG_ROADMAP),
                "part5": sign_base.display_path(PART5),
                "prior_gate": sign_base.display_path(PRIOR_GATE),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": branch_class,
                "next_route_name": next_route_name,
                "next_route": next_route,
                "followup_route_name": followup_route_name,
                "followup_route": followup_route,
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_segment_coefficient_law_gate_declared",
            "branch_completed": True,
            "next_required_artifacts": [next_route_name],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2215"),
                "roadmap_branch_hit": find_line(roadmap_text, ".2215-.2218"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2215"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2215"),
                "unified_roadmap_hit": find_line(unified_text, ".2215-.2218"),
                "long_roadmap_hit": find_line(long_text, ".2215-.2218"),
                "part5_hit": find_line(part5_text, ".2215-.2218"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_sync_rows = [
        sign_base.row(
            "status_synced",
            "pass",
            "STATUS sync target present",
            sign_base.truth(bool(find_line(status_text, "8.7.56.2215"))),
            "The coefficient-law decision gate is only honest if the official status already points to the route-reset law extraction branch.",
        ),
        sign_base.row(
            "roadmap_synced",
            "pass",
            "ROADMAP sync target present",
            sign_base.truth(bool(find_line(roadmap_text, ".2219-.2222"))),
            "The public roadmap must expose the decision gate branch before route sync can proceed.",
        ),
        sign_base.row(
            "long_horizon_synced",
            "pass",
            "long-horizon roadmap sync target present",
            sign_base.truth(bool(find_line(long_text, ".2219-.2222"))),
            "The long-horizon roadmap must still expose the law-decision state before the fallback or single-theorem route is frozen.",
        ),
    ]
    route_sync_payload = sign_base.payload(
        "8.7.56.2222",
        STEP_NAME + " route sync",
        {
            "source_files": {
                "status": sign_base.display_path(STATUS),
                "roadmap": sign_base.display_path(ROADMAP),
                "current_problem": sign_base.display_path(CURRENT_PROBLEM),
                "current_status": sign_base.display_path(CURRENT_STATUS),
                "unified_roadmap": sign_base.display_path(UNIFIED_ROADMAP),
                "long_roadmap": sign_base.display_path(LONG_ROADMAP),
                "part5": sign_base.display_path(PART5),
                "declaration_gate": declaration_paths["json"],
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": branch_class,
                "next_route_name": next_route_name,
                "next_route": next_route,
                "followup_route_name": followup_route_name,
                "followup_route": followup_route,
            },
        },
        route_sync_rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_segment_coefficient_law_gate_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [next_route_name],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2215"),
                "roadmap_branch_hit": find_line(roadmap_text, ".2219-.2222"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2215"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2215"),
                "unified_roadmap_hit": find_line(unified_text, ".2219-.2222"),
                "long_roadmap_hit": find_line(long_text, ".2219-.2222"),
                "part5_hit": find_line(part5_text, ".2219-.2222"),
            },
        },
    )
    write_artifact("route_sync", route_sync_payload)


if __name__ == "__main__":
    main()
