#!/usr/bin/env python3
"""Generate 8.7.56.5895-.5898 primary observable rerun gate artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_alpha_observable_sensitivity_inventory_backend import (
    build_trial2_alpha_observable_sensitivity_inventory_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5891-5894",
        "updated_pack_trial2_independent_observable_filter_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5895-5898"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "primary observable rerun gate"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_primary_observable_rerun_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_alpha_observable_inventory_filter_completed_"
    "primary_rerun_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_primary_observable_rerun_unavailable_"
    "qed_vacuum_materialization_primary_weak_sector_secondary_next"
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


# 関数: rerun gate で固定する rule bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas / rules fixed by the rerun gate."""
    return {
        "primary_rerun_rule": (
            "promote rerun only when a surface is independent, alpha-explicit, and "
            "current-pack rerun-ready"
        ),
        "materialization_rule": (
            "when current primary-ready count is zero, next branch is observable-map "
            "materialization rather than replay of excluded or alpha-inactive surfaces"
        ),
        "priority_order": (
            "primary materialization = QED vacuum baseline pack; "
            "secondary materialization = weak beta-decay Route A/B; "
            "reserve diagnostic = de Broglie recoil-vs-g-2 alpha consistency"
        ),
    }


# 関数: `.5895-.5898` を実行する。

def main() -> None:
    """Execute the primary observable rerun gate."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_alpha_observable_sensitivity_inventory_pack()
    summary_pack = pack["summary"]

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    primary_rerun_unavailable = int(summary_pack["primary_ready_surface_count"]) == 0
    rerun_ready_independent_unavailable = int(
        summary_pack["rerun_ready_independent_surface_count"]
    ) == 0
    qed_materialization_primary = bool(pack["trial2_alpha_primary_materialization_qed_vacuum_now"])
    weak_materialization_secondary = bool(
        pack["trial2_alpha_secondary_materialization_weak_sector_now"]
    )
    de_broglie_reserve = bool(pack["trial2_alpha_reserve_diagnostic_de_broglie_now"])

    rows = [
        sign_base.row(
            "updated_pack_trial2_independent_observable_filter_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 independent observable filter selected now",
            sign_base.truth(route_selected),
            "The rerun gate starts only after the inventory and exclusion gate agree that direct CODATA-input observables are excluded from the primary score.",
        ),
        sign_base.row(
            "trial2_primary_observable_rerun_unavailable_now",
            "pass" if primary_rerun_unavailable else "reject",
            "Trial-2 primary observable rerun unavailable now",
            sign_base.truth(primary_rerun_unavailable),
            "No current surface satisfies independence plus alpha-explicit rerun readiness, so the next mainline cannot be a blind observable rerun.",
        ),
        sign_base.row(
            "trial2_rerun_ready_independent_surface_unavailable_now",
            "pass" if rerun_ready_independent_unavailable else "reject",
            "Trial-2 rerun-ready independent surface unavailable now",
            sign_base.truth(rerun_ready_independent_unavailable),
            "The blocker is observable-map materialization itself rather than prioritization among already-runnable independent targets.",
        ),
        sign_base.row(
            "trial2_qed_vacuum_observable_map_materialization_primary_now",
            "pass" if qed_materialization_primary else "reject",
            "Trial-2 QED-vacuum observable-map materialization primary now",
            sign_base.truth(qed_materialization_primary),
            "Casimir / Lamb / H 1S-2S have the best retained source pack and the cleanest path to an explicit alpha-carrying observable map.",
        ),
        sign_base.row(
            "trial2_weak_sector_alpha_materialization_secondary_now",
            "pass" if weak_materialization_secondary else "reject",
            "Trial-2 weak-sector alpha materialization secondary now",
            sign_base.truth(weak_materialization_secondary),
            "Weak-sector observables remain important independent targets, but alpha is more hidden there than in the QED-vacuum baseline pack.",
        ),
        sign_base.row(
            "trial2_de_broglie_alpha_consistency_reserve_now",
            "pass" if de_broglie_reserve else "reject",
            "Trial-2 de Broglie alpha consistency reserve now",
            sign_base.truth(de_broglie_reserve),
            "The recoil-vs-g-2 surface is retained for cross-checking only; promoting it would reintroduce a circular benchmark into the primary score.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "primary_ready_surface_count": int(summary_pack["primary_ready_surface_count"]),
        "rerun_ready_independent_surface_count": int(
            summary_pack["rerun_ready_independent_surface_count"]
        ),
        "selected_primary_materialization_observable_id": str(
            summary_pack["selected_primary_materialization_observable_id"]
        ),
        "selected_secondary_materialization_observable_id": str(
            summary_pack["selected_secondary_materialization_observable_id"]
        ),
        "selected_reserve_diagnostic_observable_id": str(
            summary_pack["selected_reserve_diagnostic_observable_id"]
        ),
        "selected_next_generation_route": "trial2_qed_vacuum_alpha_observable_map_materialization_audit",
        "recommended_next_route_or_none": ".5899-.5902",
        "selected_followup_route": "trial2_weak_sector_alpha_dependency_materialization_audit",
        "selected_followup_route_or_none": ".5903-.5906",
        "selected_third_route": "trial2_first_independent_observable_rerun_gate",
        "selected_third_route_or_none": ".5907-.5910",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5897",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
            "top_priority": {
                "primary_materialization": str(
                    summary_pack["selected_primary_materialization_observable_id"]
                ),
                "secondary_materialization": str(
                    summary_pack["selected_secondary_materialization_observable_id"]
                ),
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_primary_observable_rerun_gate_closed_materialization_required",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {
            "primary_ready_surface_count": int(summary_pack["primary_ready_surface_count"]),
            "rerun_ready_independent_surface_count": int(
                summary_pack["rerun_ready_independent_surface_count"]
            ),
        },
    )
    artifacts = write_artifact("declaration_gate", payload)
    print("[ok] rerun gate:", artifacts["json"])


if __name__ == "__main__":
    main()
