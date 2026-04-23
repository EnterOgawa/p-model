#!/usr/bin/env python3
"""Generate 8.7.56.2179-.2182 fourth post-break registry refresh artifacts."""

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
        "8.7.56.2175-2178",
        "harmonic_fourth_post_break_piecewise_farther",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2179-2182"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor fourth post-break "
    "registry refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "harmonic_fourth_post_break_piecewise_registry_refresh",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_fourth_post_break_"
    "piecewise_validation_to_589824_farther_continuation_next"
)
BRANCH_CLASS_RETAIN = (
    "vector_qball_form_factor_boundary_bulk_lattice_fourth_post_break_"
    "piecewise_validation_to_688128_farther_continuation_next"
)
BRANCH_CLASS_RESET = (
    "vector_qball_form_factor_boundary_bulk_lattice_fifth_post_break_"
    "piecewise_validation_to_688128_farther_continuation_next"
)
NEXT_ROUTE_NAME_RETAIN = (
    "trial2_numeric_alpha_vector_qball_form_factor_fourth_post_break_"
    "piecewise_ultra_farther_continuation_audit"
)
NEXT_ROUTE_RETAIN = "8.7.56.2183"
FOLLOWUP_ROUTE_NAME_RETAIN = (
    "trial2_numeric_alpha_vector_qball_form_factor_fourth_post_break_"
    "piecewise_registry_refresh"
)
FOLLOWUP_ROUTE_RETAIN = "8.7.56.2187"
NEXT_ROUTE_NAME_RESET = (
    "trial2_numeric_alpha_vector_qball_form_factor_fifth_post_break_"
    "piecewise_farther_continuation_audit"
)
NEXT_ROUTE_RESET = "8.7.56.2183"
FOLLOWUP_ROUTE_NAME_RESET = (
    "trial2_numeric_alpha_vector_qball_form_factor_fifth_post_break_"
    "piecewise_registry_refresh"
)
FOLLOWUP_ROUTE_RESET = "8.7.56.2187"


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
    """Return formulas used in the fourth post-break registry refresh."""
    return {
        "gate_a_rule": "Gate A retains the same fourth post-break piecewise continuation once it survives farther holdout and monitor windows through harmonic 688128.",
        "gate_b_rule": "Gate B promotes the reserve fifth post-break piecewise surface once it passes farther holdout and monitor windows after the inherited fourth segment fails.",
        "gate_c_rule": "Gate C substantive pack update stays closed because the blocker remains inside the retained fourth post-break continuation family.",
    }


# 関数: `.2179-.2182` を実行する。

def main() -> None:
    """Execute the fourth post-break registry refresh."""
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
        prior_summary["same_fourth_piecewise_farther_continuation_supported"]
        or prior_summary["fifth_post_break_piecewise_surface_admissible_now"]
    )
    gate_a_same_fourth_piecewise_validation_to_688128_retained = bool(
        prior_summary["same_fourth_piecewise_farther_continuation_supported"]
    )
    gate_b_fifth_piecewise_reactivation_selected = bool(
        prior_summary["fifth_post_break_piecewise_surface_admissible_now"]
    )
    gate_c_substantive_pack_update_required = False
    loading_index_theorem_reserve_selected = True
    physical_reject_required = False

    if gate_a_same_fourth_piecewise_validation_to_688128_retained:
        branch_class = BRANCH_CLASS_RETAIN
        next_route_name = NEXT_ROUTE_NAME_RETAIN
        next_route = NEXT_ROUTE_RETAIN
        followup_route_name = FOLLOWUP_ROUTE_NAME_RETAIN
        followup_route = FOLLOWUP_ROUTE_RETAIN
    else:
        branch_class = BRANCH_CLASS_RESET
        next_route_name = NEXT_ROUTE_NAME_RESET
        next_route = NEXT_ROUTE_RESET
        followup_route_name = FOLLOWUP_ROUTE_NAME_RESET
        followup_route = FOLLOWUP_ROUTE_RESET

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "fourth post-break registry inventory ready",
            sign_base.truth(inventory_ready),
            "The registry refresh starts only after the farther audit has already shown whether the same fourth segment survives through harmonic 688128.",
        ),
        sign_base.row(
            "gate_a_same_fourth_piecewise_validation_to_688128_retained",
            "pass" if gate_a_same_fourth_piecewise_validation_to_688128_retained else "reject",
            "Gate A same fourth post-break piecewise validation through harmonic 688128 retained",
            sign_base.truth(gate_a_same_fourth_piecewise_validation_to_688128_retained),
            "Gate A is retained only if the inherited fourth segment passes both farther holdout and monitor windows.",
        ),
        sign_base.row(
            "gate_b_fifth_piecewise_reactivation_selected",
            "reject" if not gate_b_fifth_piecewise_reactivation_selected else "pass",
            "Gate B fifth post-break piecewise reactivation selected",
            sign_base.truth(gate_b_fifth_piecewise_reactivation_selected),
            "Gate B is selected only after the inherited fourth segment fails and the reserve fifth segment passes the same farther windows.",
        ),
        sign_base.row(
            "gate_c_substantive_pack_update_required",
            "reject",
            "Gate C substantive pack update required",
            sign_base.truth(gate_c_substantive_pack_update_required),
            "A substantive pack update remains unnecessary because the blocker stays inside the retained fourth post-break continuation family.",
        ),
        sign_base.row(
            "loading_index_theorem_reserve_selected",
            "pass",
            "loading-index theorem reserve selected",
            sign_base.truth(loading_index_theorem_reserve_selected),
            "The loading-index theorem remains reserve-only while the mainline continues the post-break piecewise family.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": branch_class,
        "prior_problem_classification": PRIOR_CLASS,
        "fourth_post_break_piecewise_mismatch_slope": float(
            prior_summary["fourth_post_break_piecewise_mismatch_slope"]
        ),
        "fourth_post_break_piecewise_mismatch_intercept": float(
            prior_summary["fourth_post_break_piecewise_mismatch_intercept"]
        ),
        "fourth_post_break_piecewise_correlation_slope": float(
            prior_summary["fourth_post_break_piecewise_correlation_slope"]
        ),
        "fourth_post_break_piecewise_correlation_intercept": float(
            prior_summary["fourth_post_break_piecewise_correlation_intercept"]
        ),
        "fourth_post_break_reconstruction_decay_exponent": float(
            prior_summary["fourth_post_break_reconstruction_decay_exponent"]
        ),
        "fourth_post_break_reconstruction_decay_prefactor": float(
            prior_summary["fourth_post_break_reconstruction_decay_prefactor"]
        ),
        "fourth_farther_holdout_max_mismatch_abs_error": float(
            prior_summary["fourth_farther_holdout_max_mismatch_abs_error"]
        ),
        "fourth_farther_holdout_max_correlation_abs_error": float(
            prior_summary["fourth_farther_holdout_max_correlation_abs_error"]
        ),
        "fourth_farther_holdout_max_reconstruction_abs_error": float(
            prior_summary["fourth_farther_holdout_max_reconstruction_abs_error"]
        ),
        "fourth_farther_monitor_max_mismatch_abs_error": float(
            prior_summary["fourth_farther_monitor_max_mismatch_abs_error"]
        ),
        "fourth_farther_monitor_max_correlation_abs_error": float(
            prior_summary["fourth_farther_monitor_max_correlation_abs_error"]
        ),
        "fourth_farther_monitor_max_reconstruction_abs_error": float(
            prior_summary["fourth_farther_monitor_max_reconstruction_abs_error"]
        ),
        "same_fourth_piecewise_farther_continuation_supported": gate_a_same_fourth_piecewise_validation_to_688128_retained,
        "fourth_post_break_piecewise_validation_to_688128_supported": bool(
            prior_summary["fourth_post_break_piecewise_validation_to_688128_supported"]
        ),
        "fifth_post_break_piecewise_mismatch_slope": float(
            prior_summary["fifth_post_break_piecewise_mismatch_slope"]
        ),
        "fifth_post_break_piecewise_mismatch_intercept": float(
            prior_summary["fifth_post_break_piecewise_mismatch_intercept"]
        ),
        "fifth_post_break_piecewise_correlation_slope": float(
            prior_summary["fifth_post_break_piecewise_correlation_slope"]
        ),
        "fifth_post_break_piecewise_correlation_intercept": float(
            prior_summary["fifth_post_break_piecewise_correlation_intercept"]
        ),
        "fifth_post_break_reconstruction_decay_exponent": float(
            prior_summary["fifth_post_break_reconstruction_decay_exponent"]
        ),
        "fifth_post_break_reconstruction_decay_prefactor": float(
            prior_summary["fifth_post_break_reconstruction_decay_prefactor"]
        ),
        "fifth_holdout_max_mismatch_abs_error": float(
            prior_summary["fifth_holdout_max_mismatch_abs_error"]
        ),
        "fifth_holdout_max_correlation_abs_error": float(
            prior_summary["fifth_holdout_max_correlation_abs_error"]
        ),
        "fifth_holdout_max_reconstruction_abs_error": float(
            prior_summary["fifth_holdout_max_reconstruction_abs_error"]
        ),
        "fifth_monitor_max_mismatch_abs_error": float(
            prior_summary["fifth_monitor_max_mismatch_abs_error"]
        ),
        "fifth_monitor_max_correlation_abs_error": float(
            prior_summary["fifth_monitor_max_correlation_abs_error"]
        ),
        "fifth_monitor_max_reconstruction_abs_error": float(
            prior_summary["fifth_monitor_max_reconstruction_abs_error"]
        ),
        "fifth_post_break_piecewise_validation_to_688128_supported": bool(
            prior_summary["fifth_post_break_piecewise_validation_to_688128_supported"]
        ),
        "fifth_post_break_piecewise_surface_admissible_now": gate_b_fifth_piecewise_reactivation_selected,
        "gate_a_same_fourth_piecewise_validation_to_688128_retained": gate_a_same_fourth_piecewise_validation_to_688128_retained,
        "gate_b_fifth_piecewise_reactivation_selected": gate_b_fifth_piecewise_reactivation_selected,
        "gate_c_substantive_pack_update_required": gate_c_substantive_pack_update_required,
        "loading_index_theorem_reserve_selected": loading_index_theorem_reserve_selected,
        "selected_next_generation_route": next_route_name,
        "recommended_next_route_or_none": next_route,
        "selected_followup_route": followup_route_name,
        "selected_followup_route_or_none": followup_route,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2181",
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
            "overall_status": "vector_qball_form_factor_fourth_post_break_registry_refresh_declared",
            "branch_completed": True,
            "next_required_artifacts": [next_route_name],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2179"),
                "roadmap_branch_hit": find_line(roadmap_text, "8.7.56.2179-.2182"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2179"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2179"),
                "unified_roadmap_hit": find_line(unified_text, ".2175-.2178"),
                "long_roadmap_hit": find_line(long_text, ".2175-.2178"),
                "part5_hit": find_line(part5_text, ".2171-.2178"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_sync_rows = [
        sign_base.row(
            "status_synced",
            "pass",
            "STATUS sync target present",
            sign_base.truth(bool(find_line(status_text, "8.7.56.2179"))),
            "The fourth post-break registry refresh is only honest if the official status already points to the same fourth-segment route.",
        ),
        sign_base.row(
            "roadmap_synced",
            "pass",
            "ROADMAP sync target present",
            sign_base.truth(bool(find_line(roadmap_text, "8.7.56.2179-.2182"))),
            "The public roadmap must expose the same fourth post-break registry branch before route sync can proceed.",
        ),
        sign_base.row(
            "long_horizon_synced",
            "pass",
            "long-horizon roadmap sync target present",
            sign_base.truth(bool(find_line(long_text, ".2175-.2178"))),
            "The long-horizon roadmap must still expose the prior farther continuation state before the new registry is frozen.",
        ),
    ]
    route_sync_payload = sign_base.payload(
        "8.7.56.2182",
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
            "overall_status": "vector_qball_form_factor_fourth_post_break_registry_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [next_route_name],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2179"),
                "roadmap_branch_hit": find_line(roadmap_text, "8.7.56.2179-.2182"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2179"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2179"),
                "unified_roadmap_hit": find_line(unified_text, ".2175-.2178"),
                "long_roadmap_hit": find_line(long_text, ".2175-.2178"),
                "part5_hit": find_line(part5_text, ".2171-.2178"),
            },
        },
    )
    write_artifact("route_sync", route_sync_payload)


if __name__ == "__main__":
    main()
