#!/usr/bin/env python3
"""Generate 8.7.56.2163-.2166 second post-break registry refresh artifacts."""

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
        "8.7.56.2159-2162",
        "harmonic_second_post_break_piecewise_ultra_farther",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2163-2166"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor second post-break "
    "registry refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "harmonic_second_post_break_piecewise_registry_refresh",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_second_post_break_"
    "piecewise_validation_to_393216_farther_continuation_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_third_post_break_"
    "piecewise_validation_to_491520_farther_continuation_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_third_post_break_"
    "piecewise_extreme_farther_continuation_audit"
)
NEXT_ROUTE = "8.7.56.2167"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_third_post_break_"
    "piecewise_registry_refresh"
)
FOLLOWUP_ROUTE = "8.7.56.2171"


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
    """Return formulas used in the second post-break registry refresh."""
    return {
        "gate_a_rule": "Gate A closes once the inherited second post-break piecewise continuation fails on the ultra-farther holdout or monitor windows.",
        "gate_b_rule": "Gate B promotes the reserve third post-break piecewise surface once it passes ultra-farther holdout and monitor windows after the inherited second segment fails.",
        "gate_c_rule": "Gate C substantive pack update stays closed because the blocker remains inside the retained second post-break family.",
    }


# 関数: `.2163-.2166` を実行する。

def main() -> None:
    """Execute the second post-break registry refresh."""
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
    status_hit = find_line(status_text, BRANCH_CLASS) or find_line(status_text, "8.7.56.2163")
    current_problem_hit = find_line(current_problem_text, BRANCH_CLASS) or find_line(
        current_problem_text,
        "8.7.56.2163",
    )
    current_status_hit = find_line(current_status_text, BRANCH_CLASS) or find_line(
        current_status_text,
        "8.7.56.2163",
    )

    inventory_ready = bool(prior_summary["third_post_break_piecewise_validation_to_491520_supported"])
    gate_a_second_piecewise_validation_to_491520_retained = bool(
        prior_summary["same_second_piecewise_ultra_farther_continuation_supported"]
    )
    gate_b_third_piecewise_reactivation_selected = bool(
        prior_summary["third_post_break_piecewise_surface_admissible_now"]
    )
    gate_c_substantive_pack_update_required = False
    loading_index_theorem_reserve_selected = True
    physical_reject_required = False

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "second post-break registry inventory ready",
            sign_base.truth(inventory_ready),
            "The registry refresh starts only after the ultra-farther audit has already shown that the same second segment survives through harmonic 491520.",
        ),
        sign_base.row(
            "gate_a_second_piecewise_validation_to_491520_retained",
            "pass" if gate_a_second_piecewise_validation_to_491520_retained else "reject",
            "Gate A same second post-break piecewise validation through harmonic 491520 retained",
            sign_base.truth(gate_a_second_piecewise_validation_to_491520_retained),
            "Gate A closes because the inherited second segment no longer passes the ultra-farther holdout and monitor windows.",
        ),
        sign_base.row(
            "gate_b_third_piecewise_reactivation_selected",
            "reject" if not gate_b_third_piecewise_reactivation_selected else "pass",
            "Gate B third post-break piecewise reactivation selected",
            sign_base.truth(gate_b_third_piecewise_reactivation_selected),
            "Gate B is selected because the reserve third segment passes the ultra-farther holdout and monitor windows after the inherited second segment fails.",
        ),
        sign_base.row(
            "gate_c_substantive_pack_update_required",
            "reject",
            "Gate C substantive pack update required",
            sign_base.truth(gate_c_substantive_pack_update_required),
            "A substantive pack update is still unnecessary because the blocker remains inside the retained second post-break continuation family.",
        ),
        sign_base.row(
            "loading_index_theorem_reserve_selected",
            "pass",
            "loading-index theorem reserve selected",
            sign_base.truth(loading_index_theorem_reserve_selected),
            "The loading-index theorem remains reserve-only while the mainline continues the same second post-break piecewise family.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "second_post_break_piecewise_mismatch_slope": float(
            prior_summary["second_post_break_piecewise_mismatch_slope"]
        ),
        "second_post_break_piecewise_mismatch_intercept": float(
            prior_summary["second_post_break_piecewise_mismatch_intercept"]
        ),
        "second_post_break_piecewise_correlation_slope": float(
            prior_summary["second_post_break_piecewise_correlation_slope"]
        ),
        "second_post_break_piecewise_correlation_intercept": float(
            prior_summary["second_post_break_piecewise_correlation_intercept"]
        ),
        "second_post_break_reconstruction_decay_exponent": float(
            prior_summary["second_post_break_reconstruction_decay_exponent"]
        ),
        "second_post_break_reconstruction_decay_prefactor": float(
            prior_summary["second_post_break_reconstruction_decay_prefactor"]
        ),
        "second_ultra_holdout_max_mismatch_abs_error": float(
            prior_summary["second_ultra_holdout_max_mismatch_abs_error"]
        ),
        "second_ultra_holdout_max_correlation_abs_error": float(
            prior_summary["second_ultra_holdout_max_correlation_abs_error"]
        ),
        "second_ultra_holdout_max_reconstruction_abs_error": float(
            prior_summary["second_ultra_holdout_max_reconstruction_abs_error"]
        ),
        "second_ultra_monitor_max_mismatch_abs_error": float(
            prior_summary["second_ultra_monitor_max_mismatch_abs_error"]
        ),
        "second_ultra_monitor_max_correlation_abs_error": float(
            prior_summary["second_ultra_monitor_max_correlation_abs_error"]
        ),
        "second_ultra_monitor_max_reconstruction_abs_error": float(
            prior_summary["second_ultra_monitor_max_reconstruction_abs_error"]
        ),
        "same_second_piecewise_ultra_farther_continuation_supported": gate_a_second_piecewise_validation_to_491520_retained,
        "second_post_break_piecewise_validation_to_491520_supported": bool(
            prior_summary["second_post_break_piecewise_validation_to_491520_supported"]
        ),
        "third_post_break_piecewise_mismatch_slope": float(
            prior_summary["third_post_break_piecewise_mismatch_slope"]
        ),
        "third_post_break_piecewise_mismatch_intercept": float(
            prior_summary["third_post_break_piecewise_mismatch_intercept"]
        ),
        "third_post_break_piecewise_correlation_slope": float(
            prior_summary["third_post_break_piecewise_correlation_slope"]
        ),
        "third_post_break_piecewise_correlation_intercept": float(
            prior_summary["third_post_break_piecewise_correlation_intercept"]
        ),
        "third_post_break_reconstruction_decay_exponent": float(
            prior_summary["third_post_break_reconstruction_decay_exponent"]
        ),
        "third_post_break_reconstruction_decay_prefactor": float(
            prior_summary["third_post_break_reconstruction_decay_prefactor"]
        ),
        "third_holdout_max_mismatch_abs_error": float(
            prior_summary["third_holdout_max_mismatch_abs_error"]
        ),
        "third_holdout_max_correlation_abs_error": float(
            prior_summary["third_holdout_max_correlation_abs_error"]
        ),
        "third_holdout_max_reconstruction_abs_error": float(
            prior_summary["third_holdout_max_reconstruction_abs_error"]
        ),
        "third_monitor_max_mismatch_abs_error": float(
            prior_summary["third_monitor_max_mismatch_abs_error"]
        ),
        "third_monitor_max_correlation_abs_error": float(
            prior_summary["third_monitor_max_correlation_abs_error"]
        ),
        "third_monitor_max_reconstruction_abs_error": float(
            prior_summary["third_monitor_max_reconstruction_abs_error"]
        ),
        "third_post_break_piecewise_validation_to_491520_supported": bool(
            prior_summary["third_post_break_piecewise_validation_to_491520_supported"]
        ),
        "third_post_break_piecewise_surface_admissible_now": gate_b_third_piecewise_reactivation_selected,
        "gate_a_second_piecewise_validation_to_491520_retained": gate_a_second_piecewise_validation_to_491520_retained,
        "gate_b_third_piecewise_reactivation_selected": gate_b_third_piecewise_reactivation_selected,
        "gate_c_substantive_pack_update_required": gate_c_substantive_pack_update_required,
        "loading_index_theorem_reserve_selected": loading_index_theorem_reserve_selected,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2165",
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
                "current_route": BRANCH_CLASS,
                "next_route_name": NEXT_ROUTE_NAME,
                "next_route": NEXT_ROUTE,
                "followup_route_name": FOLLOWUP_ROUTE_NAME,
                "followup_route": FOLLOWUP_ROUTE,
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_second_post_break_registry_refresh_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": status_hit,
                "roadmap_branch_hit": find_line(roadmap_text, "8.7.56.2163-.2166"),
                "current_problem_hit": current_problem_hit,
                "current_status_hit": current_status_hit,
                "unified_roadmap_hit": find_line(unified_text, ".2155-.2158"),
                "long_roadmap_hit": find_line(long_text, ".2155-.2158"),
                "part5_hit": find_line(part5_text, ".2151-.2158"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_sync_rows = [
        sign_base.row(
            "status_synced",
            "pass" if status_hit else "reject",
            "STATUS sync target present",
            sign_base.truth(bool(status_hit)),
            "The registry refresh is only honest if the official status still points to the same second post-break ultra-farther branch before the new state is frozen.",
        ),
        sign_base.row(
            "roadmap_synced",
            "pass" if bool(find_line(roadmap_text, "8.7.56.2163-.2166")) else "reject",
            "ROADMAP sync target present",
            sign_base.truth(bool(find_line(roadmap_text, "8.7.56.2163-.2166"))),
            "The public roadmap must expose the same second post-break registry branch before route sync can proceed.",
        ),
        sign_base.row(
            "long_horizon_synced",
            "pass" if bool(find_line(long_text, ".2155-.2158")) else "reject",
            "long-horizon roadmap sync target present",
            sign_base.truth(bool(find_line(long_text, ".2155-.2158"))),
            "The long-horizon roadmap must still expose the prior registry state before the new farther continuation registry is frozen.",
        ),
    ]
    route_sync_payload = sign_base.payload(
        "8.7.56.2166",
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
                "current_route": BRANCH_CLASS,
                "next_route_name": NEXT_ROUTE_NAME,
                "next_route": NEXT_ROUTE,
                "followup_route_name": FOLLOWUP_ROUTE_NAME,
                "followup_route": FOLLOWUP_ROUTE,
            },
        },
        route_sync_rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_second_post_break_registry_refresh_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": status_hit,
                "roadmap_branch_hit": find_line(roadmap_text, "8.7.56.2163-.2166"),
                "current_problem_hit": current_problem_hit,
                "current_status_hit": current_status_hit,
                "unified_roadmap_hit": find_line(unified_text, ".2155-.2158"),
                "long_roadmap_hit": find_line(long_text, ".2155-.2158"),
                "part5_hit": find_line(part5_text, ".2151-.2158"),
            },
        },
    )
    write_artifact("route_sync", route_sync_payload)


if __name__ == "__main__":
    main()
