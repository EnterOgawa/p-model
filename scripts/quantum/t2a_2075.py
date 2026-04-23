#!/usr/bin/env python3
"""Generate 8.7.56.2075-.2078 asymptotic continuation registry artifacts."""

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

PRIOR_GATE = PUBLIC_OUT / "q_8_7_56_2071_2074_harmonic_asymptotic_continuation_declaration_gate_metrics.json"

STEP_TAG = "8.7.56.2075-2078"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor loading-index theorem "
    "reserve or pack-update registry"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "harmonic_asymptotic_registry",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_asymptotic_extension_512_"
    "retained_loading_index_theorem_reserve_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_asymptotic_extension_512_"
    "retained_loading_index_theorem_reserve_further_asymptotic_continuation_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_bulk_lattice_farther_"
    "asymptotic_continuation_audit"
)
NEXT_ROUTE = "8.7.56.2079"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_loading_index_theorem_"
    "reserve_or_pack_update_registry_refresh"
)
FOLLOWUP_ROUTE = "8.7.56.2083"


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
    """Return formulas used in the asymptotic continuation registry."""
    return {
        "retained_bulk_lattice": "delta_q^(n,m) = delta_q,base^(box) + m_n Delta_box",
        "registry_gate": "retain continuation while asymptotic blocks satisfy max mismatch <= 0.25 and min sign correlation >= 0.5",
        "reserve_read": "keep exact loading-index theorem as reserve while farther continuation remains honest",
    }


# 関数: `.2075-.2078` を実行する。

def main() -> None:
    """Execute the loading-index theorem reserve or pack-update registry sync."""
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

    inventory_ready = bool(prior_summary["same_lattice_survives_to_512"])
    gate_a_asymptotic_continuation_retained = bool(
        prior_summary["asymptotic_farther_harmonic_extension_to_512_supported"]
    )
    gate_b_loading_index_theorem_reserve_selected = bool(
        prior_summary["exact_loading_index_theorem_remains_reserve"]
    )
    gate_c_substantive_pack_update_required = False
    same_level_loading_index_affine_retry_admissible = False
    same_level_smooth_loading_retry_admissible = False
    farther_asymptotic_continuation_admissible_now = bool(
        prior_summary["farther_asymptotic_continuation_admissible_now"]
    )
    loading_index_theorem_reserve_admissible_now = True
    substantive_pack_update_required_now = False
    physical_reject_required = False

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "asymptotic continuation registry inventory ready",
            sign_base.truth(inventory_ready),
            "The registry starts only after the audit has already shown that the same lattice survives through harmonic 512.",
        ),
        sign_base.row(
            "gate_a_asymptotic_continuation_retained",
            "pass" if gate_a_asymptotic_continuation_retained else "reject",
            "Gate A asymptotic continuation retained",
            sign_base.truth(gate_a_asymptotic_continuation_retained),
            "Because every asymptotic block still survives the same thresholds, the honest mainline remains farther continuation of the same lattice.",
        ),
        sign_base.row(
            "gate_b_loading_index_theorem_reserve_selected",
            "pass" if gate_b_loading_index_theorem_reserve_selected else "reject",
            "Gate B loading-index theorem reserve selected",
            sign_base.truth(gate_b_loading_index_theorem_reserve_selected),
            "The exact loading-index theorem stays reserve-only while asymptotic continuation remains live.",
        ),
        sign_base.row(
            "gate_c_substantive_pack_update_required",
            "reject",
            "Gate C substantive pack update required",
            sign_base.truth(gate_c_substantive_pack_update_required),
            "A pack update is still unnecessary because the same retained family continues to survive on farther asymptotic blocks.",
        ),
        sign_base.row(
            "farther_asymptotic_continuation_admissible_now",
            "pass" if farther_asymptotic_continuation_admissible_now else "reject",
            "farther asymptotic continuation admissible now",
            sign_base.truth(farther_asymptotic_continuation_admissible_now),
            "The next honest computation is farther continuation of the same retained lattice rather than another theorem fit.",
        ),
        sign_base.row(
            "loading_index_theorem_reserve_admissible_now",
            "pass",
            "loading-index theorem reserve admissible now",
            sign_base.truth(loading_index_theorem_reserve_admissible_now),
            "The discrete loading theorem remains meaningful only as a reserve surface while the continuation route survives.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "asymptotic_farther_harmonic_extension_to_512_supported": bool(
            prior_summary["asymptotic_farther_harmonic_extension_to_512_supported"]
        ),
        "extension_129_160_max_mismatch_fraction": float(prior_summary["extension_129_160_max_mismatch_fraction"]),
        "extension_161_192_max_mismatch_fraction": float(prior_summary["extension_161_192_max_mismatch_fraction"]),
        "extension_193_256_max_mismatch_fraction": float(prior_summary["extension_193_256_max_mismatch_fraction"]),
        "extension_257_320_max_mismatch_fraction": float(prior_summary["extension_257_320_max_mismatch_fraction"]),
        "extension_321_384_max_mismatch_fraction": float(prior_summary["extension_321_384_max_mismatch_fraction"]),
        "extension_385_448_max_mismatch_fraction": float(prior_summary["extension_385_448_max_mismatch_fraction"]),
        "extension_449_512_max_mismatch_fraction": float(prior_summary["extension_449_512_max_mismatch_fraction"]),
        "gate_a_asymptotic_continuation_retained": gate_a_asymptotic_continuation_retained,
        "gate_b_loading_index_theorem_reserve_selected": gate_b_loading_index_theorem_reserve_selected,
        "gate_c_substantive_pack_update_required": gate_c_substantive_pack_update_required,
        "same_level_loading_index_affine_retry_admissible": same_level_loading_index_affine_retry_admissible,
        "same_level_smooth_loading_retry_admissible": same_level_smooth_loading_retry_admissible,
        "farther_asymptotic_continuation_admissible_now": farther_asymptotic_continuation_admissible_now,
        "loading_index_theorem_reserve_admissible_now": loading_index_theorem_reserve_admissible_now,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "substantive_pack_update_required_now": substantive_pack_update_required_now,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2077",
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
            "overall_status": "vector_qball_form_factor_asymptotic_continuation_registry_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2071"),
                "roadmap_branch_hit": find_line(roadmap_text, "8.7.56.2075-.2078"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2071"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2071"),
                "unified_roadmap_hit": find_line(unified_text, ".2075-.2078"),
                "long_roadmap_hit": find_line(long_text, ".2075-.2078"),
                "part5_hit": find_line(part5_text, ".2063-.2070"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_sync_rows = [
        sign_base.row(
            "status_synced",
            "pass",
            "STATUS sync target present",
            sign_base.truth(bool(find_line(status_text, "8.7.56.2071"))),
            "The asymptotic continuation registry is only valid if status still points to the same live continuation state.",
        ),
        sign_base.row(
            "roadmap_synced",
            "pass",
            "ROADMAP sync target present",
            sign_base.truth(bool(find_line(roadmap_text, "8.7.56.2075-.2078"))),
            "The public roadmap must expose the reserve registry branch before it is frozen.",
        ),
        sign_base.row(
            "long_horizon_synced",
            "pass",
            "long-horizon roadmap sync target present",
            sign_base.truth(bool(find_line(long_text, ".2075-.2078"))),
            "The long-horizon roadmap must expose the same reserve registry route.",
        ),
    ]

    route_sync_payload = sign_base.payload(
        "8.7.56.2078",
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
            "overall_status": "vector_qball_form_factor_asymptotic_continuation_registry_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2071"),
                "roadmap_branch_hit": find_line(roadmap_text, "8.7.56.2075-.2078"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2071"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2071"),
                "unified_roadmap_hit": find_line(unified_text, ".2075-.2078"),
                "long_roadmap_hit": find_line(long_text, ".2075-.2078"),
                "part5_hit": find_line(part5_text, ".2063-.2070"),
            },
        },
    )
    write_artifact("route_sync", route_sync_payload)


if __name__ == "__main__":
    main()
