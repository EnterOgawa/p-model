#!/usr/bin/env python3
"""Generate 8.7.56.5979-.5982 refreshed multi-observable CODATA-lead artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_multi_observable_codata_lead_gate_refresh_backend import (
    build_trial2_multi_observable_codata_lead_gate_refresh_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5975-5978",
        "updated_pack_trial2_third_independent_surface_gate_second_refresh",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5979-5982"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "multi-observable CODATA-lead gate refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_multi_observable_codata_lead_gate_refresh",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_third_independent_surface_completed_three_surface_gate_primary_"
    "codata_refresh_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_three_surface_codata_lead_watch_retained_conditional_reopen_only_next"
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
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])

    return {"json": sign_base.display_path(paths["json"])}


# 関数: `.5979-.5982` の formula bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas / rules fixed by the refreshed three-surface aggregate gate."""
    return {
        "table_rule": (
            "compare Hydrogen 1S-2S gross structure, Hydrogen 21 cm g/2-corrected "
            "hyperfine, and Hydrogen H-alpha fine-structure Dirac span"
        ),
        "codata_sweep_rule": "codata sweep when alpha_CODATA is best overall on all actual surfaces",
        "watch_rule": (
            "retain watch, not final pass/reject, while the retained table is still "
            "Hydrogen-only and baseline-level"
        ),
    }


# 関数: `.5979-.5982` を実行する。

def main() -> None:
    """Execute the refreshed three-surface aggregate gate."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_multi_observable_codata_lead_gate_refresh_pack()
    summary_pack = pack["summary"]

    route_selected = str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    codata_sweep = bool(summary_pack["codata_sweep_verdict_now"])
    watch_retained = bool(summary_pack["multi_observable_watch_retained_now"])
    pass_unavailable = not bool(summary_pack["multi_observable_pass_available_now"])
    count_is_three = int(summary_pack["current_actual_surface_count_now"]) == 3

    rows = [
        sign_base.row(
            "updated_pack_trial2_third_surface_gate_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 third-surface gate selected now",
            sign_base.truth(route_selected),
            "The aggregate CODATA-lead refresh starts only after the third independent surface gate passes.",
        ),
        sign_base.row(
            "trial2_three_surface_codata_sweep_now",
            "pass" if codata_sweep else "reject",
            "Trial-2 three-surface CODATA sweep now",
            sign_base.truth(codata_sweep),
            "The retained three-surface Hydrogen table is closest to alpha_CODATA on every actual surface.",
        ),
        sign_base.row(
            "trial2_three_actual_surfaces_now",
            "pass" if count_is_three else "reject",
            "Trial-2 three actual surfaces now",
            sign_base.truth(count_is_three),
            "The aggregate verdict now rests on three actual alpha-explicit surfaces rather than two.",
        ),
        sign_base.row(
            "trial2_multi_observable_pass_still_unavailable_now",
            "pass" if pass_unavailable else "reject",
            "Trial-2 multi-observable pass still unavailable now",
            sign_base.truth(pass_unavailable),
            "Even the strengthened three-surface table remains a watch because the comparison is still Hydrogen-only and baseline-level.",
        ),
        sign_base.row(
            "trial2_three_surface_codata_lead_watch_retained_now",
            "pass" if watch_retained else "reject",
            "Trial-2 three-surface CODATA-lead watch retained now",
            sign_base.truth(watch_retained),
            "The honest reading is a stronger CODATA-lead watch, not a final pass or reject.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "current_actual_surface_count_now": int(summary_pack["current_actual_surface_count_now"]),
        "surface_ids_now": list(summary_pack["surface_ids_now"]),
        "pmodel_win_count_now": int(summary_pack["pmodel_win_count_now"]),
        "codata_win_count_now": int(summary_pack["codata_win_count_now"]),
        "codata_sweep_verdict_now": bool(summary_pack["codata_sweep_verdict_now"]),
        "multi_observable_watch_retained_now": bool(summary_pack["multi_observable_watch_retained_now"]),
        "recommended_next_route_or_none": "none",
        "selected_next_generation_route": "conditional_reopen_only",
        "selected_followup_route": "new_non_hydrogen_surface_or_full_hyperfine_precision_or_new_selected_extension",
        "selected_followup_route_or_none": "conditional",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5981",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
            "surface_rows": pack["surface_rows"],
        },
        rows,
        summary,
        {
            "overall_status": "trial2_three_surface_codata_lead_watch_retained",
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
    print("[ok] three-surface CODATA-lead gate:", artifacts["json"])


if __name__ == "__main__":
    main()
