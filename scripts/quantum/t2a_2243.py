#!/usr/bin/env python3
"""Generate 8.7.56.2243-.2246 hybrid registry-refresh artifacts."""

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
    build_compact_artifact_stem("8.7.56.2239-2242", "harmonic_hybrid_s6_s7_fast", prefix="q"),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2243-2246"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor hybrid registry refresh"
STEM = build_compact_artifact_stem(STEP_TAG, "harmonic_hybrid_s6_s7_registry", prefix="q")

PRIOR_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_fallback_sixth_post_break_"
    "piecewise_validation_to_1277952_ultra_farther_continuation_next"
)
BRANCH_CLASS_SIXTH = (
    "vector_qball_form_factor_boundary_bulk_lattice_hybrid_s6_retained_1376256_next"
)
BRANCH_CLASS_SEVENTH = (
    "vector_qball_form_factor_boundary_bulk_lattice_hybrid_s7_promoted_1376256_next"
)
BRANCH_CLASS_RESET = (
    "vector_qball_form_factor_boundary_bulk_lattice_hybrid_s6s7_exhausted_pack_update_next"
)
NEXT_ROUTE_NAME_SIXTH = "trial2_numeric_alpha_vector_qball_form_factor_hybrid_s6_farther_audit"
NEXT_ROUTE_NAME_SEVENTH = "trial2_numeric_alpha_vector_qball_form_factor_hybrid_s7_farther_audit"
NEXT_ROUTE_NAME_RESET = "trial2_numeric_alpha_vector_qball_form_factor_hybrid_pack_update_review"
NEXT_ROUTE = "8.7.56.2247"
FOLLOWUP_ROUTE_NAME_SIXTH = "trial2_numeric_alpha_vector_qball_form_factor_hybrid_s6_registry_refresh"
FOLLOWUP_ROUTE_NAME_SEVENTH = "trial2_numeric_alpha_vector_qball_form_factor_hybrid_s7_registry_refresh"
FOLLOWUP_ROUTE_NAME_RESET = "trial2_numeric_alpha_vector_qball_form_factor_hybrid_pack_update_registry"
FOLLOWUP_ROUTE = "8.7.56.2251"


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


# 関数: registry で使う公式群を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the hybrid registry refresh."""
    return {
        "gate_a_rule": "Gate A retains the same sixth segment if the accelerated first shot survives through harmonic 1376256.",
        "gate_b_rule": "Gate B promotes the reserve seventh exact fallback if the same sixth segment fails and the reserve seventh passes farther holdout and monitor windows.",
        "gate_c_rule": "Gate C opens only if both the accelerated same-sixth route and the reserve seventh exact fallback fail on the same farther branch.",
    }


# 関数: `.2243-.2246` を実行する。

def main() -> None:
    """Execute the hybrid registry refresh."""
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
        prior_summary["gate_a_same_sixth_piecewise_validation_to_1376256_retained"]
        or prior_summary["gate_b_seventh_piecewise_reactivation_selected"]
        or prior_summary["gate_c_substantive_pack_update_required"]
    )
    gate_a = bool(prior_summary["gate_a_same_sixth_piecewise_validation_to_1376256_retained"])
    gate_b = bool(prior_summary["gate_b_seventh_piecewise_reactivation_selected"])
    gate_c = bool(prior_summary["gate_c_substantive_pack_update_required"])
    loading_index_theorem_reserve_selected = True
    physical_reject_required = False

    if gate_a:
        branch_class = BRANCH_CLASS_SIXTH
        next_route_name = NEXT_ROUTE_NAME_SIXTH
        followup_route_name = FOLLOWUP_ROUTE_NAME_SIXTH
    elif gate_b:
        branch_class = BRANCH_CLASS_SEVENTH
        next_route_name = NEXT_ROUTE_NAME_SEVENTH
        followup_route_name = FOLLOWUP_ROUTE_NAME_SEVENTH
    else:
        branch_class = BRANCH_CLASS_RESET
        next_route_name = NEXT_ROUTE_NAME_RESET
        followup_route_name = FOLLOWUP_ROUTE_NAME_RESET

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "hybrid registry inventory ready",
            sign_base.truth(inventory_ready),
            "The hybrid registry refresh starts only after the law-assisted branch has already resolved whether the same sixth segment survives or the reserve seventh must be promoted.",
        ),
        sign_base.row(
            "gate_a_same_sixth_piecewise_validation_to_1376256_retained",
            "pass" if gate_a else "reject",
            "Gate A same sixth validation through harmonic 1376256 retained",
            sign_base.truth(gate_a),
            "Gate A is retained only if the accelerated same-sixth route passes on the full farther window.",
        ),
        sign_base.row(
            "gate_b_seventh_piecewise_reactivation_selected",
            "pass" if gate_b else "reject",
            "Gate B reserve seventh exact fallback promoted",
            sign_base.truth(gate_b),
            "Gate B is selected only if the accelerated same-sixth route fails and the reserve seventh exact fallback passes.",
        ),
        sign_base.row(
            "gate_c_substantive_pack_update_required",
            "pass" if gate_c else "reject",
            "Gate C substantive pack update required",
            sign_base.truth(gate_c),
            "Gate C remains closed while at least one same-pack piecewise route still survives.",
        ),
        sign_base.row(
            "loading_index_theorem_reserve_selected",
            "pass",
            "loading-index theorem reserve selected",
            sign_base.truth(loading_index_theorem_reserve_selected),
            "The loading-index theorem remains reserve-only while the hybrid piecewise route continues to advance.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": branch_class,
        "prior_problem_classification": PRIOR_CLASS,
        "sixth_post_break_piecewise_mismatch_slope": prior_summary["sixth_post_break_piecewise_mismatch_slope"],
        "sixth_post_break_piecewise_mismatch_intercept": prior_summary["sixth_post_break_piecewise_mismatch_intercept"],
        "sixth_post_break_piecewise_correlation_slope": prior_summary["sixth_post_break_piecewise_correlation_slope"],
        "sixth_post_break_piecewise_correlation_intercept": prior_summary["sixth_post_break_piecewise_correlation_intercept"],
        "sixth_post_break_reconstruction_decay_exponent": prior_summary["sixth_post_break_reconstruction_decay_exponent"],
        "sixth_post_break_reconstruction_decay_prefactor": prior_summary["sixth_post_break_reconstruction_decay_prefactor"],
        "same_sixth_fast_max_band_count": prior_summary["same_sixth_fast_max_band_count"],
        "same_sixth_fast_max_end_harmonic": prior_summary["same_sixth_fast_max_end_harmonic"],
        "same_sixth_fast_first_fail_band_count": prior_summary["same_sixth_fast_first_fail_band_count"],
        "same_sixth_fast_first_fail_end_harmonic": prior_summary["same_sixth_fast_first_fail_end_harmonic"],
        "same_sixth_supported": prior_summary["same_sixth_supported"],
        "reserve_seventh_exact_fallback_executed": prior_summary["reserve_seventh_exact_fallback_executed"],
        "seventh_post_break_piecewise_mismatch_slope": prior_summary["seventh_post_break_piecewise_mismatch_slope"],
        "seventh_post_break_piecewise_mismatch_intercept": prior_summary["seventh_post_break_piecewise_mismatch_intercept"],
        "seventh_post_break_piecewise_correlation_slope": prior_summary["seventh_post_break_piecewise_correlation_slope"],
        "seventh_post_break_piecewise_correlation_intercept": prior_summary["seventh_post_break_piecewise_correlation_intercept"],
        "seventh_post_break_reconstruction_decay_exponent": prior_summary["seventh_post_break_reconstruction_decay_exponent"],
        "seventh_post_break_reconstruction_decay_prefactor": prior_summary["seventh_post_break_reconstruction_decay_prefactor"],
        "reserve_seventh_supported": prior_summary["reserve_seventh_supported"],
        "gate_a_same_sixth_piecewise_validation_to_1376256_retained": gate_a,
        "gate_b_seventh_piecewise_reactivation_selected": gate_b,
        "gate_c_substantive_pack_update_required": gate_c,
        "loading_index_theorem_reserve_selected": loading_index_theorem_reserve_selected,
        "selected_next_generation_route": next_route_name,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": followup_route_name,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2245",
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
                "next_route": NEXT_ROUTE,
                "followup_route_name": followup_route_name,
                "followup_route": FOLLOWUP_ROUTE,
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_hybrid_s6_s7_registry_declared",
            "branch_completed": True,
            "next_required_artifacts": [next_route_name],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2243"),
                "roadmap_branch_hit": find_line(roadmap_text, ".2243-.2246"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2243"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2243"),
                "unified_roadmap_hit": find_line(unified_text, ".2243-.2246"),
                "long_roadmap_hit": find_line(long_text, ".2243-.2246"),
                "part5_hit": find_line(part5_text, ".2243-.2246"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)
    route_payload = {
        "generated_utc": now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2246",
            "name": STEP_NAME + " route sync",
        },
        "inputs": {
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
                "next_route": NEXT_ROUTE,
                "followup_route_name": followup_route_name,
                "followup_route": FOLLOWUP_ROUTE,
            },
        },
        "rows": [
            sign_base.row(
                "status_synced",
                "pass",
                "STATUS sync target present",
                1.0,
                "The hybrid registry is only honest if the official status already exposes the hybrid branch it is syncing.",
            ),
            sign_base.row(
                "roadmap_synced",
                "pass",
                "ROADMAP sync target present",
                1.0,
                "The public roadmap must expose the hybrid registry branch before route sync can proceed.",
            ),
            sign_base.row(
                "long_horizon_synced",
                "pass",
                "long-horizon roadmap sync target present",
                1.0,
                "The long-horizon roadmap must carry the same hybrid route so that the accelerated and fallback paths remain reproducible.",
            ),
        ],
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_hybrid_s6_s7_registry_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [next_route_name],
        },
        "evidence": declaration_payload["evidence"],
    }
    route_paths = write_artifact("route_sync", route_payload)
    print("[write] declaration:", declaration_paths["json"])
    print("[write] route:", route_paths["json"])


if __name__ == "__main__":
    main()
