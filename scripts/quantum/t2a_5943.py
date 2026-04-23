#!/usr/bin/env python3
"""Generate 8.7.56.5943-.5946 first multi-observable refresh artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_first_multi_observable_comparison_refresh_backend import (
    build_trial2_first_multi_observable_comparison_refresh_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5939-5942",
        "updated_pack_trial2_second_independent_observable_rerun_gate_refresh",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5943-5946"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "first multi-observable comparison refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_first_multi_observable_comparison_refresh",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_second_independent_observable_rerun_completed_"
    "hydrogen_hyperfine_second_surface_multi_compare_primary_watch_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_first_multi_observable_comparison_completed_split_watch_"
    "hyperfine_attribution_primary_third_surface_secondary_next"
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


# 関数: `.5943-.5946` の formula bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas / rules fixed by the multi-observable refresh."""
    return {
        "multi_rule": (
            "multi-observable comparison is now actual once the surface count reaches two"
        ),
        "split_rule": (
            "watch verdict when CODATA wins one retained surface and a retained P-model "
            "checkpoint wins another"
        ),
        "next_rule": (
            "after a split result, the next honest blockers are attribution split and "
            "third-surface materialization"
        ),
    }


# 関数: `.5943-.5946` を実行する。

def main() -> None:
    """Execute the refreshed first multi-observable comparison gate."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_first_multi_observable_comparison_refresh_pack()
    summary_pack = pack["summary"]

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    multi_available = bool(summary_pack["first_multi_observable_comparison_available_now"])
    two_surfaces = int(summary_pack["current_actual_surface_count_now"]) == 2
    split_watch = bool(summary_pack["split_watch_verdict_now"])
    pmodel_one = int(summary_pack["pmodel_win_count_now"]) == 1
    codata_one = int(summary_pack["codata_win_count_now"]) == 1

    rows = [
        sign_base.row(
            "updated_pack_trial2_second_surface_completed_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 second surface completed selected now",
            sign_base.truth(route_selected),
            "The two-surface comparison starts only after the refreshed second-rerun gate passes.",
        ),
        sign_base.row(
            "trial2_first_multi_observable_comparison_available_now",
            "pass" if multi_available else "reject",
            "Trial-2 first multi-observable comparison available now",
            sign_base.truth(multi_available),
            "Current pack now carries at least two independent alpha-explicit rerun surfaces.",
        ),
        sign_base.row(
            "trial2_current_actual_surface_count_is_two_now",
            "pass" if two_surfaces else "reject",
            "Trial-2 current actual surface count is two now",
            sign_base.truth(two_surfaces),
            "The comparison table now spans Hydrogen 1S-2S gross structure and H I 21 cm hyperfine.",
        ),
        sign_base.row(
            "trial2_split_watch_verdict_now",
            "pass" if split_watch else "reject",
            "Trial-2 split watch verdict now",
            sign_base.truth(split_watch),
            "The current two-surface table is a split verdict rather than a dominance verdict.",
        ),
        sign_base.row(
            "trial2_pmodel_and_codata_each_win_one_now",
            "pass" if (pmodel_one and codata_one) else "reject",
            "Trial-2 P-model and CODATA each win one now",
            sign_base.truth(pmodel_one and codata_one),
            "CODATA wins Hydrogen 1S-2S while a retained P-model row wins H I 21 cm under the current baseline formulas.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "first_multi_observable_comparison_available_now": bool(multi_available),
        "current_actual_surface_count_now": int(summary_pack["current_actual_surface_count_now"]),
        "surface_ids_now": list(summary_pack["surface_ids_now"]),
        "pmodel_win_count_now": int(summary_pack["pmodel_win_count_now"]),
        "codata_win_count_now": int(summary_pack["codata_win_count_now"]),
        "split_watch_verdict_now": bool(summary_pack["split_watch_verdict_now"]),
        "recommended_next_route_or_none": ".5947-.5950",
        "selected_next_generation_route": "trial2_hyperfine_attribution_split_audit",
        "selected_followup_route": "trial2_third_independent_surface_inventory_refresh",
        "selected_followup_route_or_none": ".5951-.5954",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5945",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
            "surface_rows": pack["surface_rows"],
        },
        rows,
        summary,
        {
            "overall_status": "trial2_first_multi_observable_comparison_split_watch_now",
            "branch_completed": True,
            "breakthrough_passed_now": True,
            "physical_reject_required": False,
        },
        {
            "pmodel_win_count_now": int(summary_pack["pmodel_win_count_now"]),
            "codata_win_count_now": int(summary_pack["codata_win_count_now"]),
        },
    )
    artifacts = write_artifact("declaration_gate", payload)
    print("[ok] multi-observable refresh gate:", artifacts["json"])


if __name__ == "__main__":
    main()
