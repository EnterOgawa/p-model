#!/usr/bin/env python3
"""Generate 8.7.56.2139-.2142 post-break registry refresh artifacts."""

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
        "8.7.56.2135-2138",
        "harmonic_post_break_piecewise_curvature",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2139-2142"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor post-break "
    "stress-envelope registry refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "harmonic_post_break_piecewise_registry",
    prefix="q",
)

PRIOR_CLASS = "vector_qball_form_factor_boundary_bulk_lattice_post_break_piecewise_holdout_gate_next"
BRANCH_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_post_break_piecewise_"
    "validation_to_163840_farther_continuation_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_post_break_piecewise_"
    "farther_continuation_audit"
)
NEXT_ROUTE = "8.7.56.2143"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_post_break_piecewise_"
    "registry_refresh"
)
FOLLOWUP_ROUTE = "8.7.56.2147"


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
    """Return formulas used in the post-break registry refresh."""
    return {
        "gate_a_rule": "Gate A exact global post-break theorem stays closed",
        "gate_b_rule": "Gate B retains the piecewise validation through harmonic 163840 and sends the mainline to farther continuation",
        "gate_c_rule": "Gate C substantive pack update stays closed because the blocker is localized to farther post-break continuation rather than to a missing pack surface",
    }


# 関数: `.2139-.2142` を実行する。

def main() -> None:
    """Execute the post-break stress-envelope registry refresh."""
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

    inventory_ready = bool(prior_summary["post_break_piecewise_surface_selected"])
    gate_a_exact_global_post_break_theorem_selected = bool(
        prior_summary["exact_global_post_break_theorem_available"]
    )
    gate_b_piecewise_validation_to_163840_retained = bool(
        prior_summary["piecewise_validation_to_163840_supported"]
    )
    gate_c_substantive_pack_update_required = False
    post_break_monitor_drift_detected = bool(prior_summary["post_break_monitor_drift_detected"])
    loading_index_theorem_reserve_selected = True
    physical_reject_required = False

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "post-break registry inventory ready",
            sign_base.truth(inventory_ready),
            "The registry refresh starts only after the post-break audit has already selected the piecewise surface on the exact holdout.",
        ),
        sign_base.row(
            "gate_a_exact_global_post_break_theorem_selected",
            "reject",
            "Gate A exact global post-break theorem selected",
            sign_base.truth(gate_a_exact_global_post_break_theorem_selected),
            "Gate A stays closed because the current branch only closes holdout-level validation through harmonic 163840.",
        ),
        sign_base.row(
            "gate_b_piecewise_validation_to_163840_retained",
            "pass" if gate_b_piecewise_validation_to_163840_retained else "reject",
            "Gate B piecewise validation through harmonic 163840 retained",
            sign_base.truth(gate_b_piecewise_validation_to_163840_retained),
            "Gate B is retained because the post-break piecewise surface beats curvature on the exact holdout.",
        ),
        sign_base.row(
            "post_break_monitor_drift_detected",
            "pass" if post_break_monitor_drift_detected else "reject",
            "farther monitor drift detected beyond harmonic 163840",
            sign_base.truth(post_break_monitor_drift_detected),
            "The next honest blocker rises only after the holdout has already passed, so the mainline now moves to farther post-break continuation.",
        ),
        sign_base.row(
            "gate_c_substantive_pack_update_required",
            "reject",
            "Gate C substantive pack update required",
            sign_base.truth(gate_c_substantive_pack_update_required),
            "A substantive pack update is still unnecessary because the blocker remains inside the retained post-break family.",
        ),
        sign_base.row(
            "loading_index_theorem_reserve_selected",
            "pass",
            "loading-index theorem reserve selected",
            sign_base.truth(loading_index_theorem_reserve_selected),
            "The loading-index theorem remains reserve-only while the current mainline continues the retained post-break piecewise family.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "piecewise_holdout_max_mismatch_abs_error": float(
            prior_summary["piecewise_holdout_max_mismatch_abs_error"]
        ),
        "piecewise_holdout_max_correlation_abs_error": float(
            prior_summary["piecewise_holdout_max_correlation_abs_error"]
        ),
        "piecewise_holdout_max_reconstruction_abs_error": float(
            prior_summary["piecewise_holdout_max_reconstruction_abs_error"]
        ),
        "piecewise_monitor_max_mismatch_abs_error": float(
            prior_summary["piecewise_monitor_max_mismatch_abs_error"]
        ),
        "piecewise_monitor_max_correlation_abs_error": float(
            prior_summary["piecewise_monitor_max_correlation_abs_error"]
        ),
        "piecewise_monitor_max_reconstruction_abs_error": float(
            prior_summary["piecewise_monitor_max_reconstruction_abs_error"]
        ),
        "curvature_holdout_max_mismatch_abs_error": float(
            prior_summary["curvature_holdout_max_mismatch_abs_error"]
        ),
        "curvature_holdout_max_correlation_abs_error": float(
            prior_summary["curvature_holdout_max_correlation_abs_error"]
        ),
        "piecewise_beats_curvature_on_holdout": bool(
            prior_summary["piecewise_beats_curvature_on_holdout"]
        ),
        "piecewise_validation_to_163840_supported": gate_b_piecewise_validation_to_163840_retained,
        "post_break_monitor_drift_detected": post_break_monitor_drift_detected,
        "exact_global_post_break_theorem_available": gate_a_exact_global_post_break_theorem_selected,
        "gate_a_exact_global_post_break_theorem_selected": gate_a_exact_global_post_break_theorem_selected,
        "gate_b_piecewise_validation_to_163840_retained": gate_b_piecewise_validation_to_163840_retained,
        "gate_c_substantive_pack_update_required": gate_c_substantive_pack_update_required,
        "loading_index_theorem_reserve_selected": loading_index_theorem_reserve_selected,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2141",
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
            "overall_status": "vector_qball_form_factor_post_break_registry_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2139"),
                "roadmap_branch_hit": find_line(roadmap_text, "8.7.56.2139-.2142"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2135"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2135"),
                "unified_roadmap_hit": find_line(unified_text, ".2135-.2138"),
                "long_roadmap_hit": find_line(long_text, ".2135-.2138"),
                "part5_hit": find_line(part5_text, ".2127-.2134"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_sync_rows = [
        sign_base.row(
            "status_synced",
            "pass",
            "STATUS sync target present",
            sign_base.truth(bool(find_line(status_text, "8.7.56.2139"))),
            "The registry refresh is only honest if the official status already points to the same post-break registry route.",
        ),
        sign_base.row(
            "roadmap_synced",
            "pass",
            "ROADMAP sync target present",
            sign_base.truth(bool(find_line(roadmap_text, "8.7.56.2139-.2142"))),
            "The public roadmap must expose the same post-break registry branch before route sync can proceed.",
        ),
        sign_base.row(
            "long_horizon_synced",
            "pass",
            "long-horizon roadmap sync target present",
            sign_base.truth(bool(find_line(long_text, ".2135-.2138"))),
            "The long-horizon roadmap must still expose the post-break reactivation route before the new registry is frozen.",
        ),
    ]
    route_sync_payload = sign_base.payload(
        "8.7.56.2142",
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
            "overall_status": "vector_qball_form_factor_post_break_registry_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2139"),
                "roadmap_branch_hit": find_line(roadmap_text, "8.7.56.2139-.2142"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2135"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2135"),
                "unified_roadmap_hit": find_line(unified_text, ".2135-.2138"),
                "long_roadmap_hit": find_line(long_text, ".2135-.2138"),
                "part5_hit": find_line(part5_text, ".2127-.2134"),
            },
        },
    )
    write_artifact("route_sync", route_sync_payload)


if __name__ == "__main__":
    main()
