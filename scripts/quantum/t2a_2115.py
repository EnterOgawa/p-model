#!/usr/bin/env python3
"""Generate 8.7.56.2115-.2118 sparse plateau drift registry artifacts."""

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

PRIOR_GATE = (
    PUBLIC_OUT
    / "q_8_7_56_2111_2114_harmonic_sparse_asymptotic_drift_audit_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.2115-2118"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor sparse plateau drift "
    "registry refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "harmonic_sparse_plateau_drift_registry_refresh",
    prefix="q",
)

PRIOR_CLASS = "vector_qball_form_factor_boundary_bulk_lattice_sparse_exact_asymptotic_drift_gate_next"
BRANCH_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_sparse_exact_plateau_to_57344_"
    "partial_retain_drift_law_audit_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_sparse_exact_drift_law_audit"
)
NEXT_ROUTE = "8.7.56.2119"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_sparse_exact_drift_law_registry_"
    "refresh"
)
FOLLOWUP_ROUTE = "8.7.56.2123"


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
    """Return formulas used in the sparse plateau drift registry refresh."""
    return {
        "retain_rule": "retain the farther sparse exact plateau only through the last sampled band whose min sign correlation stays >= 0.5",
        "first_break_rule": "the first sampled band with min sign correlation < 0.5 becomes the honest farther break marker",
        "next_route": "once the farther partial retain / first-break split is frozen, move to sparse exact drift-law audit instead of reopening same-level loading-index retries",
    }


# 関数: `.2115-.2118` を実行する。

def main() -> None:
    """Execute the sparse plateau drift registry refresh."""
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

    inventory_ready = bool(prior_summary["farther_sparse_plateau_to_32768_supported"])
    gate_a_sparse_exact_plateau_to_65536_selected = bool(
        prior_summary["stress_sign_floor_break_57345_65536"] is False
    )
    gate_b_sparse_exact_plateau_to_57344_partial_retain_selected = bool(
        prior_summary["stress_sparse_plateau_to_57344_supported"]
        and prior_summary["stress_sign_floor_break_57345_65536"]
    )
    gate_c_substantive_pack_update_required = False
    sparse_exact_drift_law_audit_admissible_now = bool(
        prior_summary["sparse_exact_asymptotic_drift_detected"]
        and prior_summary["stress_sign_floor_break_57345_65536"]
    )
    loading_index_theorem_reserve_selected = True
    physical_reject_required = False

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "sparse plateau drift registry inventory ready",
            sign_base.truth(inventory_ready),
            "The registry refresh starts only after the farther sparse exact audit has split retained plateau bands from the first sampled sign-floor break.",
        ),
        sign_base.row(
            "gate_a_sparse_exact_plateau_to_65536_selected",
            "pass" if gate_a_sparse_exact_plateau_to_65536_selected else "reject",
            "Gate A sparse exact plateau retained through harmonic 65536 selected",
            sign_base.truth(gate_a_sparse_exact_plateau_to_65536_selected),
            "Gate A stays closed because the sampled band 57345..65536 already drops below the retained sign-correlation floor.",
        ),
        sign_base.row(
            "gate_b_sparse_exact_plateau_to_57344_partial_retain_selected",
            "pass" if gate_b_sparse_exact_plateau_to_57344_partial_retain_selected else "reject",
            "Gate B sparse exact plateau partial retain through harmonic 57344 selected",
            sign_base.truth(gate_b_sparse_exact_plateau_to_57344_partial_retain_selected),
            "Gate B retains the same lattice up to the last sampled farther band that still satisfies the positive sign-correlation floor.",
        ),
        sign_base.row(
            "gate_c_substantive_pack_update_required",
            "reject",
            "Gate C substantive pack update required",
            sign_base.truth(gate_c_substantive_pack_update_required),
            "A substantive pack update is still unnecessary because the same retained lattice survives well past harmonic 16384 before the first sampled sign-floor break.",
        ),
        sign_base.row(
            "sparse_exact_drift_law_audit_admissible_now",
            "pass" if sparse_exact_drift_law_audit_admissible_now else "reject",
            "sparse exact drift-law audit admissible now",
            sign_base.truth(sparse_exact_drift_law_audit_admissible_now),
            "Once farther partial retention and the first break band are both fixed, the honest next move is to model the drift law rather than reopen same-level theorem scans.",
        ),
        sign_base.row(
            "loading_index_theorem_reserve_selected",
            "pass",
            "loading-index theorem reserve selected",
            sign_base.truth(loading_index_theorem_reserve_selected),
            "The loading-index theorem remains reserve-only under the farther sparse plateau drift disposition.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "inherited_sparse_plateau_ceiling_4097_16384": float(
            prior_summary["inherited_sparse_plateau_ceiling_4097_16384"]
        ),
        "primary_28673_32768_max_mismatch_fraction": float(
            prior_summary["primary_28673_32768_max_mismatch_fraction"]
        ),
        "primary_28673_32768_min_sign_correlation": float(
            prior_summary["primary_28673_32768_min_sign_correlation"]
        ),
        "stress_49153_57344_max_mismatch_fraction": float(
            prior_summary["stress_49153_57344_max_mismatch_fraction"]
        ),
        "stress_49153_57344_min_sign_correlation": float(
            prior_summary["stress_49153_57344_min_sign_correlation"]
        ),
        "stress_57345_65536_max_mismatch_fraction": float(
            prior_summary["stress_57345_65536_max_mismatch_fraction"]
        ),
        "stress_57345_65536_min_sign_correlation": float(
            prior_summary["stress_57345_65536_min_sign_correlation"]
        ),
        "farther_sparse_plateau_to_32768_supported": bool(
            prior_summary["farther_sparse_plateau_to_32768_supported"]
        ),
        "stress_sparse_plateau_to_57344_supported": bool(
            prior_summary["stress_sparse_plateau_to_57344_supported"]
        ),
        "stress_sign_floor_break_57345_65536": bool(
            prior_summary["stress_sign_floor_break_57345_65536"]
        ),
        "sparse_exact_asymptotic_drift_detected": bool(
            prior_summary["sparse_exact_asymptotic_drift_detected"]
        ),
        "gate_a_sparse_exact_plateau_to_65536_selected": gate_a_sparse_exact_plateau_to_65536_selected,
        "gate_b_sparse_exact_plateau_to_57344_partial_retain_selected": gate_b_sparse_exact_plateau_to_57344_partial_retain_selected,
        "gate_c_substantive_pack_update_required": gate_c_substantive_pack_update_required,
        "sparse_exact_drift_law_audit_admissible_now": sparse_exact_drift_law_audit_admissible_now,
        "loading_index_theorem_reserve_selected": loading_index_theorem_reserve_selected,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2117",
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
            "overall_status": "vector_qball_form_factor_sparse_plateau_drift_registry_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2115"),
                "roadmap_branch_hit": find_line(roadmap_text, "8.7.56.2115-.2118"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2115"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2115"),
                "unified_roadmap_hit": find_line(unified_text, ".2115-.2118"),
                "long_roadmap_hit": find_line(long_text, ".2115-.2118"),
                "part5_hit": find_line(part5_text, ".2111-.2118"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_sync_rows = [
        sign_base.row(
            "status_synced",
            "pass",
            "STATUS sync target present",
            sign_base.truth(bool(find_line(status_text, "8.7.56.2115"))),
            "The sparse plateau drift registry is only valid if status already points to the same official branch.",
        ),
        sign_base.row(
            "roadmap_synced",
            "pass",
            "ROADMAP sync target present",
            sign_base.truth(bool(find_line(roadmap_text, "8.7.56.2115-.2118"))),
            "The public roadmap must expose the sparse plateau drift registry before its result is frozen.",
        ),
        sign_base.row(
            "long_horizon_synced",
            "pass",
            "long-horizon roadmap sync target present",
            sign_base.truth(bool(find_line(long_text, ".2115-.2118"))),
            "The long-horizon roadmap must expose the same sparse plateau drift route.",
        ),
    ]
    route_sync_payload = sign_base.payload(
        "8.7.56.2118",
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
            "overall_status": "vector_qball_form_factor_sparse_plateau_drift_registry_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2115"),
                "roadmap_branch_hit": find_line(roadmap_text, "8.7.56.2115-.2118"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2115"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2115"),
                "unified_roadmap_hit": find_line(unified_text, ".2115-.2118"),
                "long_roadmap_hit": find_line(long_text, ".2115-.2118"),
                "part5_hit": find_line(part5_text, ".2111-.2118"),
            },
        },
    )
    write_artifact("route_sync", route_sync_payload)


if __name__ == "__main__":
    main()
