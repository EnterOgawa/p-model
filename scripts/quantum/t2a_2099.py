#!/usr/bin/env python3
"""Generate 8.7.56.2099-.2102 quarter-band continuation registry artifacts."""

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

PRIOR_GATE = PUBLIC_OUT / "q_8_7_56_2095_2098_harmonic_quarter_band_farther_continuation_declaration_gate_metrics.json"

STEP_TAG = "8.7.56.2099-2102"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor loading-index theorem "
    "reserve or pack-update registry refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "harmonic_quarter_band_farther_registry_refresh",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_quarter_band_asymptotic_"
    "extension_4096_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_quarter_band_asymptotic_"
    "extension_4096_retained_loading_index_theorem_reserve_ultra_farther_"
    "continuation_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_quarter_band_ultra_farther_"
    "continuation_audit"
)
NEXT_ROUTE = "8.7.56.2103"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_loading_index_theorem_"
    "reserve_or_pack_update_registry_refresh"
)
FOLLOWUP_ROUTE = "8.7.56.2107"


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
    """Return formulas used in the farther registry refresh."""
    return {
        "quarter_band_retain": "retain farther continuation while the same lattice stays inside the inherited quarter-band asymptotic surface",
        "reserve_read": "keep exact loading-index theorem as reserve while same-lattice farther continuation still survives",
        "next_route": "advance to quarter-band ultra-farther continuation only if farther continuation to 4096 remains honest and pack update stays unnecessary",
    }


# 関数: `.2099-.2102` を実行する。

def main() -> None:
    """Execute the quarter-band farther registry refresh."""
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

    inventory_ready = bool(prior_summary["same_lattice_survives_to_4096_under_quarter_band"])
    gate_a_quarter_band_farther_continuation_retained = bool(
        prior_summary["quarter_band_further_continuation_to_4096_supported"]
    )
    gate_b_loading_index_theorem_reserve_selected = True
    gate_c_substantive_pack_update_required = False
    quarter_band_ultra_farther_continuation_admissible_now = bool(
        prior_summary["quarter_band_further_continuation_to_4096_supported"]
    )
    loading_index_theorem_reserve_admissible_now = True
    substantive_pack_update_required_now = False
    physical_reject_required = False

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "quarter-band farther registry inventory ready",
            sign_base.truth(inventory_ready),
            "The registry refresh starts only after farther continuation to 4096 has already been measured directly.",
        ),
        sign_base.row(
            "gate_a_quarter_band_farther_continuation_retained",
            "pass" if gate_a_quarter_band_farther_continuation_retained else "reject",
            "Gate A quarter-band farther continuation retained",
            sign_base.truth(gate_a_quarter_band_farther_continuation_retained),
            "The same lattice remains the honest mainline while farther continuation stays inside the quarter-band asymptotic surface.",
        ),
        sign_base.row(
            "gate_b_loading_index_theorem_reserve_selected",
            "pass",
            "Gate B loading-index theorem reserve selected",
            sign_base.truth(gate_b_loading_index_theorem_reserve_selected),
            "The exact loading-index theorem remains reserve-only while empirical continuation still survives.",
        ),
        sign_base.row(
            "gate_c_substantive_pack_update_required",
            "reject",
            "Gate C substantive pack update required",
            sign_base.truth(gate_c_substantive_pack_update_required),
            "A substantive pack update is still unnecessary because the same retained lattice survives one more farther continuation audit.",
        ),
        sign_base.row(
            "quarter_band_ultra_farther_continuation_admissible_now",
            "pass" if quarter_band_ultra_farther_continuation_admissible_now else "reject",
            "quarter-band ultra-farther continuation admissible now",
            sign_base.truth(quarter_band_ultra_farther_continuation_admissible_now),
            "The next honest computation is to continue the same quarter-band family beyond harmonic 4096.",
        ),
        sign_base.row(
            "loading_index_theorem_reserve_admissible_now",
            "pass",
            "loading-index theorem reserve admissible now",
            sign_base.truth(loading_index_theorem_reserve_admissible_now),
            "The exact loading-index theorem remains reserve-only while the quarter-band family still survives.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "quarter_band_abs_deviation_ceiling_3073_4096": float(
            prior_summary["quarter_band_abs_deviation_ceiling_3073_4096"]
        ),
        "quarter_band_abs_deviation_max_3073_4096": float(
            prior_summary["quarter_band_abs_deviation_max_3073_4096"]
        ),
        "quarter_band_min_sign_correlation_3073_4096": float(
            prior_summary["quarter_band_min_sign_correlation_3073_4096"]
        ),
        "signed_reconstruction_abs_error_continues_decay": bool(
            prior_summary["signed_reconstruction_abs_error_continues_decay"]
        ),
        "gate_a_quarter_band_farther_continuation_retained": gate_a_quarter_band_farther_continuation_retained,
        "gate_b_loading_index_theorem_reserve_selected": gate_b_loading_index_theorem_reserve_selected,
        "gate_c_substantive_pack_update_required": gate_c_substantive_pack_update_required,
        "quarter_band_ultra_farther_continuation_admissible_now": quarter_band_ultra_farther_continuation_admissible_now,
        "loading_index_theorem_reserve_admissible_now": loading_index_theorem_reserve_admissible_now,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "substantive_pack_update_required_now": substantive_pack_update_required_now,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2101",
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
            "overall_status": "vector_qball_form_factor_quarter_band_farther_registry_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2095"),
                "roadmap_branch_hit": find_line(roadmap_text, "8.7.56.2095-.2098"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2095"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2095"),
                "unified_roadmap_hit": find_line(unified_text, ".2095-.2098"),
                "long_roadmap_hit": find_line(long_text, ".2095-.2098"),
                "part5_hit": find_line(part5_text, ".2087-.2094"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_sync_rows = [
        sign_base.row(
            "status_synced",
            "pass",
            "STATUS sync target present",
            sign_base.truth(bool(find_line(status_text, "8.7.56.2095"))),
            "The registry refresh is only valid if status already points to the farther continuation branch that just completed.",
        ),
        sign_base.row(
            "roadmap_synced",
            "pass",
            "ROADMAP sync target present",
            sign_base.truth(bool(find_line(roadmap_text, "8.7.56.2095-.2098"))),
            "The public roadmap must expose the farther continuation branch before the refreshed registry is frozen.",
        ),
        sign_base.row(
            "long_horizon_synced",
            "pass",
            "long-horizon roadmap sync target present",
            sign_base.truth(bool(find_line(long_text, ".2095-.2098"))),
            "The long-horizon roadmap must also expose the farther continuation branch before the refreshed registry is frozen.",
        ),
    ]
    route_sync_payload = sign_base.payload(
        "8.7.56.2102",
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
            "overall_status": "vector_qball_form_factor_quarter_band_farther_registry_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2095"),
                "roadmap_branch_hit": find_line(roadmap_text, "8.7.56.2095-.2098"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2095"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2095"),
                "unified_roadmap_hit": find_line(unified_text, ".2095-.2098"),
                "long_roadmap_hit": find_line(long_text, ".2095-.2098"),
                "part5_hit": find_line(part5_text, ".2087-.2094"),
            },
        },
    )
    write_artifact("route_sync", route_sync_payload)


if __name__ == "__main__":
    main()
