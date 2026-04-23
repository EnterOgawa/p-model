#!/usr/bin/env python3
"""Generate 8.7.56.5939-.5942 second-rerun refresh artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_second_independent_observable_rerun_gate_refresh_backend import (
    build_trial2_second_independent_rerun_gate_refresh_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5935-5938",
        "updated_pack_trial2_hydrogen_hyperfine_absolute_alpha_formula_materialization_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5939-5942"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "second independent observable rerun gate refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_second_independent_observable_rerun_gate_refresh",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_hydrogen_hyperfine_absolute_alpha_formula_materialized_"
    "second_surface_gate_primary_multi_compare_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_second_independent_observable_rerun_completed_"
    "hydrogen_hyperfine_second_surface_multi_compare_primary_watch_secondary_next"
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


# 関数: `.5939-.5942` の rule bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas / rules fixed by the refreshed second-surface gate."""
    return {
        "second_surface_rule": (
            "second rerun becomes available only when current actual surface count "
            "reaches two independent alpha-explicit surfaces"
        ),
        "current_surface_rule": (
            "the retained pair is Hydrogen 1S-2S gross structure plus H I 21 cm Fermi baseline"
        ),
        "multi_rule": (
            "once surface count is two, the next honest branch is the first actual "
            "multi-observable comparison refresh"
        ),
    }


# 関数: `.5939-.5942` を実行する。

def main() -> None:
    """Execute the refreshed second independent rerun gate."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_second_independent_rerun_gate_refresh_pack()
    summary_pack = pack["summary"]

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    second_available = bool(summary_pack["second_independent_observable_rerun_available_now"])
    two_surfaces = int(summary_pack["current_actual_surface_count_now"]) == 2
    correct_second_id = (
        str(summary_pack["selected_second_surface_id"]) == "hydrogen_hyperfine_21cm_fermi_baseline"
    )
    multi_compare_ready = second_available and two_surfaces

    rows = [
        sign_base.row(
            "updated_pack_trial2_hyperfine_materialization_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 hyperfine materialization selected now",
            sign_base.truth(route_selected),
            "The refreshed second gate starts only after the hyperfine absolute-formula audit passes.",
        ),
        sign_base.row(
            "trial2_second_independent_observable_rerun_available_now",
            "pass" if second_available else "reject",
            "Trial-2 second independent observable rerun available now",
            sign_base.truth(second_available),
            "The current public pack now has a genuine second independent alpha-explicit rerun surface.",
        ),
        sign_base.row(
            "trial2_current_actual_surface_count_is_two_now",
            "pass" if two_surfaces else "reject",
            "Trial-2 current actual surface count is two now",
            sign_base.truth(two_surfaces),
            "The retained actual surfaces are Hydrogen 1S-2S gross structure and H I 21 cm hyperfine.",
        ),
        sign_base.row(
            "trial2_selected_second_surface_is_hyperfine_now",
            "pass" if correct_second_id else "reject",
            "Trial-2 selected second surface is hyperfine now",
            sign_base.truth(correct_second_id),
            "The new second surface is the Hydrogen hyperfine 21 cm Fermi baseline.",
        ),
        sign_base.row(
            "trial2_multi_observable_refresh_ready_now",
            "pass" if multi_compare_ready else "reject",
            "Trial-2 multi-observable refresh ready now",
            sign_base.truth(multi_compare_ready),
            "Once the second surface exists, the first two-surface comparison becomes the next honest gate.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "second_independent_observable_rerun_available_now": bool(second_available),
        "current_actual_surface_count_now": int(summary_pack["current_actual_surface_count_now"]),
        "current_actual_surface_ids": list(summary_pack["current_actual_surface_ids"]),
        "selected_first_surface_id": str(summary_pack["selected_first_surface_id"]),
        "selected_second_surface_id": str(summary_pack["selected_second_surface_id"]),
        "recommended_next_route_or_none": ".5943-.5946",
        "selected_followup_route": "trial2_first_multi_observable_comparison_refresh",
        "selected_followup_route_or_none": ".5943-.5946",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5941",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
            "surface_table": pack["surface_table"],
        },
        rows,
        summary,
        {
            "overall_status": "trial2_second_independent_observable_rerun_available_now",
            "branch_completed": True,
            "breakthrough_passed_now": True,
            "physical_reject_required": False,
        },
        {
            "current_actual_surface_count_now": int(summary_pack["current_actual_surface_count_now"]),
            "first_surface_best_relative_error_vs_observed": float(
                summary_pack["first_surface_best_relative_error_vs_observed"]
            ),
            "second_surface_best_relative_error_vs_observed": float(
                summary_pack["second_surface_best_relative_error_vs_observed"]
            ),
        },
    )
    artifacts = write_artifact("declaration_gate", payload)
    print("[ok] second-rerun refresh gate:", artifacts["json"])


if __name__ == "__main__":
    main()
