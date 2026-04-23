#!/usr/bin/env python3
"""Generate 8.7.56.5931-.5934 first multi-observable gate artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_first_multi_observable_comparison_gate_backend import (
    build_trial2_first_multi_observable_comparison_gate_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5927-5930",
        "updated_pack_trial2_second_independent_observable_rerun_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5931-5934"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "first multi-observable comparison gate"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_first_multi_observable_comparison_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_second_independent_observable_rerun_unavailable_completed_"
    "multi_observable_gate_primary_conditional_reopen_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_first_multi_observable_comparison_unavailable_second_surface_missing_"
    "conditional_reopen_only_next"
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


# 関数: `.5931-.5934` の rule bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas / rules fixed by the multi-observable gate."""
    return {
        "multi_rule": (
            "multi-observable comparison requires at least two independent, "
            "alpha-explicit, rerun-ready surfaces"
        ),
        "current_rule": (
            "the current pack retains exactly one actual surface: Hydrogen 1S-2S gross structure"
        ),
        "reopen_rule": (
            "reopen only when a genuinely new second independent alpha-explicit "
            "surface is materialized"
        ),
    }


# 関数: `.5931-.5934` を実行する。

def main() -> None:
    """Execute the first multi-observable comparison gate."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_first_multi_observable_comparison_gate_pack()
    summary_pack = pack["summary"]

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    multi_unavailable = not bool(summary_pack["first_multi_observable_comparison_available_now"])
    only_one_surface = int(summary_pack["current_actual_surface_count_now"]) == 1
    one_surface_missing = int(summary_pack["missing_surface_count_now"]) == 1
    retained_hydrogen = (
        str(summary_pack["retained_only_surface_id"]) == "hydrogen_1s2s_gross_structure_baseline"
    )

    rows = [
        sign_base.row(
            "updated_pack_trial2_second_rerun_unavailable_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 second rerun unavailable selected now",
            sign_base.truth(route_selected),
            "The multi-observable gate starts only after the second independent rerun gate closes negatively.",
        ),
        sign_base.row(
            "trial2_first_multi_observable_comparison_unavailable_now",
            "pass" if multi_unavailable else "reject",
            "Trial-2 first multi-observable comparison unavailable now",
            sign_base.truth(multi_unavailable),
            "Current pack cannot form a two-surface comparison table yet.",
        ),
        sign_base.row(
            "trial2_current_actual_surface_count_is_one_now",
            "pass" if only_one_surface else "reject",
            "Trial-2 current actual surface count is one now",
            sign_base.truth(only_one_surface),
            "Only Hydrogen 1S-2S gross structure is rerun-ready under the current public pack.",
        ),
        sign_base.row(
            "trial2_one_more_surface_required_for_multi_now",
            "pass" if one_surface_missing else "reject",
            "Trial-2 one more surface required for multi now",
            sign_base.truth(one_surface_missing),
            "Exactly one additional independent alpha-explicit surface is still missing.",
        ),
        sign_base.row(
            "trial2_retained_only_surface_is_hydrogen_now",
            "pass" if retained_hydrogen else "reject",
            "Trial-2 retained only surface is hydrogen now",
            sign_base.truth(retained_hydrogen),
            "The retained only actual surface is the Hydrogen 1S-2S gross-structure baseline.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "first_multi_observable_comparison_available_now": False,
        "current_actual_surface_count_now": int(summary_pack["current_actual_surface_count_now"]),
        "missing_surface_count_now": int(summary_pack["missing_surface_count_now"]),
        "retained_only_surface_id": str(summary_pack["retained_only_surface_id"]),
        "recommended_next_route_or_none": "No unconditional next official branch",
        "selected_reopen_condition": (
            "genuinely new second independent alpha-explicit rerun surface actually materializes"
        ),
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5933",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
            "retained_only_surface": pack["retained_only_surface"],
            "prediction_table": pack["prediction_table"],
        },
        rows,
        summary,
        {
            "overall_status": "trial2_first_multi_observable_comparison_unavailable_current_pack",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {
            "current_actual_surface_count_now": int(summary_pack["current_actual_surface_count_now"]),
            "missing_surface_count_now": int(summary_pack["missing_surface_count_now"]),
        },
    )
    artifacts = write_artifact("declaration_gate", payload)
    print("[ok] multi-observable gate:", artifacts["json"])


if __name__ == "__main__":
    main()
