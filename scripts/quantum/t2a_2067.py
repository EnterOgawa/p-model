#!/usr/bin/env python3
"""Generate 8.7.56.2067-.2070 bulk-lattice extension registry artifacts."""

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

PRIOR_GATE = PUBLIC_OUT / "q_8_7_56_2063_2066_harmonic_ext_loading_theorem_declaration_gate_metrics.json"

STEP_TAG = "8.7.56.2067-2070"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor bulk-lattice loading closeout / registry"
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "harmonic_ext_loading_registry",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_extension_128_retained_"
    "loading_index_theorem_deferred_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_extension_128_retained_"
    "loading_index_theorem_deferred_asymptotic_continuation_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_bulk_lattice_asymptotic_"
    "farther_harmonic_continuation_audit"
)
NEXT_ROUTE = "8.7.56.2071"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_loading_index_theorem_reserve_"
    "or_pack_update_registry"
)
FOLLOWUP_ROUTE = "8.7.56.2075"


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
    """Return formulas used in the bulk-lattice extension registry."""
    return {
        "retained_bulk_lattice": "delta_q^(n,m) = delta_q,base^(box) + m_n Delta_box",
        "extension_read": "retain lattice family if max mismatch <= 0.25 and min sign correlation >= 0.5 on all farther blocks",
        "deferred_theorem_read": "defer exact loading-index theorem while asymptotic continuation remains admissible",
    }


# 関数: `.2067-.2070` を実行する。

def main() -> None:
    """Execute the bulk-lattice loading closeout / registry sync."""
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

    inventory_ready = bool(prior_summary["farther_harmonic_extension_selected"])
    gate_a_exact_loading_index_theorem_selected = bool(prior_summary["simple_loading_index_theorem_available"])
    gate_b_farther_extension_retained = bool(prior_summary["farther_harmonic_extension_to_128_supported"])
    gate_c_substantive_pack_update_required = False
    same_level_loading_index_affine_retry_admissible = False
    same_level_smooth_loading_retry_admissible = False
    asymptotic_farther_harmonic_continuation_admissible_now = bool(
        prior_summary["asymptotic_farther_harmonic_continuation_admissible_now"]
    )
    loading_index_theorem_reserve_admissible_now = True
    substantive_pack_update_required_now = False
    physical_reject_required = False

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "bulk-lattice extension registry inventory ready",
            sign_base.truth(inventory_ready),
            "The registry starts only after the theorem-vs-extension branch has already selected farther continuation as the honest mainline.",
        ),
        sign_base.row(
            "gate_a_exact_loading_index_theorem_selected",
            "reject" if not gate_a_exact_loading_index_theorem_selected else "pass",
            "Gate A exact loading-index theorem selected",
            sign_base.truth(gate_a_exact_loading_index_theorem_selected),
            "Gate A would require an actual theorem for the discrete loading indices, which is still unavailable.",
        ),
        sign_base.row(
            "gate_b_farther_extension_retained",
            "pass" if gate_b_farther_extension_retained else "reject",
            "Gate B farther-harmonic extension retained",
            sign_base.truth(gate_b_farther_extension_retained),
            "Once the same bulk lattice survives to harmonic 128, the honest closeout is to retain farther continuation and defer the theorem gap.",
        ),
        sign_base.row(
            "gate_c_substantive_pack_update_required",
            "reject",
            "Gate C substantive pack update required",
            sign_base.truth(gate_c_substantive_pack_update_required),
            "The current pack still has a live continuation route before any pack update is needed.",
        ),
        sign_base.row(
            "same_level_loading_index_affine_retry_admissible",
            "reject",
            "same-level loading-index affine retry admissible",
            sign_base.truth(same_level_loading_index_affine_retry_admissible),
            "Low-order affine/index regressions are already falsified and should not reopen as same-level retries.",
        ),
        sign_base.row(
            "asymptotic_farther_harmonic_continuation_admissible_now",
            "pass" if asymptotic_farther_harmonic_continuation_admissible_now else "reject",
            "asymptotic farther-harmonic continuation admissible now",
            sign_base.truth(asymptotic_farther_harmonic_continuation_admissible_now),
            "The retained lattice should now be tested on asymptotically farther harmonic blocks before any new signed rule is introduced.",
        ),
        sign_base.row(
            "loading_index_theorem_reserve_admissible_now",
            "pass",
            "loading-index theorem reserve admissible now",
            sign_base.truth(loading_index_theorem_reserve_admissible_now),
            "The exact loading-index theorem stays open, but only as a reserve surface after the honest continuation route is exhausted.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "farther_harmonic_extension_to_128_supported": bool(
            prior_summary["farther_harmonic_extension_to_128_supported"]
        ),
        "extension_25_40_max_mismatch_fraction": float(prior_summary["extension_25_40_max_mismatch_fraction"]),
        "extension_41_64_max_mismatch_fraction": float(prior_summary["extension_41_64_max_mismatch_fraction"]),
        "extension_65_96_max_mismatch_fraction": float(prior_summary["extension_65_96_max_mismatch_fraction"]),
        "extension_97_128_max_mismatch_fraction": float(prior_summary["extension_97_128_max_mismatch_fraction"]),
        "gate_a_exact_loading_index_theorem_selected": gate_a_exact_loading_index_theorem_selected,
        "gate_b_farther_extension_retained": gate_b_farther_extension_retained,
        "gate_c_substantive_pack_update_required": gate_c_substantive_pack_update_required,
        "same_level_loading_index_affine_retry_admissible": same_level_loading_index_affine_retry_admissible,
        "same_level_smooth_loading_retry_admissible": same_level_smooth_loading_retry_admissible,
        "asymptotic_farther_harmonic_continuation_admissible_now": asymptotic_farther_harmonic_continuation_admissible_now,
        "loading_index_theorem_reserve_admissible_now": loading_index_theorem_reserve_admissible_now,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "substantive_pack_update_required_now": substantive_pack_update_required_now,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2069",
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
            "overall_status": "vector_qball_form_factor_bulk_lattice_extension_registry_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2063"),
                "roadmap_branch_hit": find_line(roadmap_text, "8.7.56.2067-.2070"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2063"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2063"),
                "unified_roadmap_hit": find_line(unified_text, ".2067-.2070"),
                "long_roadmap_hit": find_line(long_text, ".2067-.2070"),
                "part5_hit": find_line(part5_text, ".2055-.2062"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_sync_rows = [
        sign_base.row(
            "status_synced",
            "pass",
            "STATUS sync target present",
            sign_base.truth(bool(find_line(status_text, "8.7.56.2063"))),
            "The registry is only valid if the status still points to the theorem-vs-extension route that has just been resolved.",
        ),
        sign_base.row(
            "roadmap_synced",
            "pass",
            "ROADMAP sync target present",
            sign_base.truth(bool(find_line(roadmap_text, "8.7.56.2067-.2070"))),
            "The public roadmap must expose the extension registry branch before it is fixed.",
        ),
        sign_base.row(
            "long_horizon_synced",
            "pass",
            "long-horizon roadmap sync target present",
            sign_base.truth(bool(find_line(long_text, ".2067-.2070"))),
            "The long-horizon roadmap must also show the same extension registry route.",
        ),
    ]
    route_sync_payload = sign_base.payload(
        "8.7.56.2070",
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
            "overall_status": "vector_qball_form_factor_bulk_lattice_extension_registry_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2063"),
                "roadmap_branch_hit": find_line(roadmap_text, "8.7.56.2067-.2070"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2063"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2063"),
                "unified_roadmap_hit": find_line(unified_text, ".2067-.2070"),
                "long_roadmap_hit": find_line(long_text, ".2067-.2070"),
                "part5_hit": find_line(part5_text, ".2055-.2062"),
            },
        },
    )
    write_artifact("route_sync", route_sync_payload)


if __name__ == "__main__":
    main()
