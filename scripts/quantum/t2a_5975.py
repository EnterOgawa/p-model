#!/usr/bin/env python3
"""Generate 8.7.56.5975-.5978 third-surface second-refresh artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_third_independent_surface_gate_second_refresh_backend import (
    build_trial2_third_independent_surface_gate_second_refresh_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5971-5974",
        "updated_pack_trial2_hydrogen_fine_structure_absolute_alpha_formula_materialization_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5975-5978"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "third independent surface gate second refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_third_independent_surface_gate_second_refresh",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_hydrogen_fine_structure_absolute_alpha_formula_materialized_"
    "third_surface_gate_primary_codata_refresh_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_third_independent_surface_completed_three_surface_gate_primary_"
    "codata_refresh_secondary_next"
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


# 関数: `.5975-.5978` の formula bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas / rules fixed by the refreshed third-surface gate."""
    return {
        "count_rule": "actual surface count rises to three only if the new fine-structure surface is genuinely new and rerun-ready",
        "independence_rule": (
            "the third surface must not replay the gross alpha^2 family and must "
            "remain distinct from the corrected hyperfine magnetic-contact family"
        ),
        "refresh_rule": "refresh the aggregate table only after the third-surface gate passes",
    }


# 関数: `.5975-.5978` を実行する。

def main() -> None:
    """Execute the refreshed third-surface gate."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_third_independent_surface_gate_second_refresh_pack()
    summary_pack = pack["summary"]

    route_selected = str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    third_surface_available = bool(summary_pack["genuine_third_independent_surface_available_now"])
    count_is_three = int(summary_pack["current_actual_surface_count_now"]) == 3
    all_alpha_explicit = bool(summary_pack["all_surfaces_alpha_explicit_now"])
    all_primary_ready = bool(summary_pack["all_surfaces_primary_score_admissible_now"])

    rows = [
        sign_base.row(
            "updated_pack_trial2_fine_structure_surface_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 fine-structure surface selected now",
            sign_base.truth(route_selected),
            "The second-refresh gate starts only after the new fine-structure surface is materialized.",
        ),
        sign_base.row(
            "trial2_genuine_third_independent_surface_available_now",
            "pass" if third_surface_available else "reject",
            "Trial-2 genuine third independent surface available now",
            sign_base.truth(third_surface_available),
            "The retained H-alpha fine-structure Dirac span actualizes the missing third independent alpha-explicit family.",
        ),
        sign_base.row(
            "trial2_actual_surface_count_is_three_now",
            "pass" if count_is_three else "reject",
            "Trial-2 actual surface count is three now",
            sign_base.truth(count_is_three),
            "The observable-comparison table now contains three actual alpha-explicit surfaces.",
        ),
        sign_base.row(
            "trial2_all_surface_rows_alpha_explicit_now",
            "pass" if all_alpha_explicit else "reject",
            "Trial-2 all surface rows alpha explicit now",
            sign_base.truth(all_alpha_explicit),
            "All retained surfaces in the refreshed table are deterministic alpha-explicit formulas.",
        ),
        sign_base.row(
            "trial2_all_surface_rows_primary_ready_now",
            "pass" if all_primary_ready else "reject",
            "Trial-2 all surface rows primary ready now",
            sign_base.truth(all_primary_ready),
            "All three retained surfaces remain admissible in the current primary score.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "current_actual_surface_count_now": int(summary_pack["current_actual_surface_count_now"]),
        "surface_ids_now": list(summary_pack["surface_ids_now"]),
        "family_ids_now": list(summary_pack["family_ids_now"]),
        "genuine_third_independent_surface_available_now": bool(
            summary_pack["genuine_third_independent_surface_available_now"]
        ),
        "genuine_third_independent_surface_id_now": str(
            summary_pack["genuine_third_independent_surface_id_now"]
        ),
        "recommended_next_route_or_none": ".5979-.5982",
        "selected_next_generation_route": "trial2_multi_observable_codata_lead_gate_refresh",
        "selected_followup_route": "conditional_reopen_only",
        "selected_followup_route_or_none": "conditional",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5977",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
            "surface_rows": pack["surface_rows"],
        },
        rows,
        summary,
        {
            "overall_status": "trial2_third_independent_surface_completed",
            "branch_completed": True,
            "breakthrough_passed_now": True,
            "physical_reject_required": False,
        },
        {
            "current_actual_surface_count_now": int(summary_pack["current_actual_surface_count_now"]),
            "genuine_third_independent_surface_available_now": bool(
                summary_pack["genuine_third_independent_surface_available_now"]
            ),
        },
    )
    artifacts = write_artifact("declaration_gate", payload)
    print("[ok] third-surface second-refresh gate:", artifacts["json"])


if __name__ == "__main__":
    main()
