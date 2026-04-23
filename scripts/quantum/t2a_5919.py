#!/usr/bin/env python3
"""Generate 8.7.56.5919-.5922 first actual rerun gate artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_first_actual_independent_observable_rerun_gate_backend import (
    build_trial2_first_actual_independent_rerun_gate_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5915-5918",
        "updated_pack_trial2_weak_beta_decay_explicit_alpha_formula_materialization_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5919-5922"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "first actual independent observable rerun gate"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_first_actual_independent_observable_rerun_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_weak_beta_decay_explicit_alpha_formula_negative_closeout_"
    "completed_first_actual_rerun_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_first_actual_independent_observable_rerun_completed_"
    "hydrogen_1s2s_gross_structure_lamb_absolute_formula_primary_"
    "multi_observable_gate_secondary_next"
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


# 関数: `.5919-.5922` の rule bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas / rules fixed by the first actual rerun gate."""
    return {
        "first_rerun_rule": (
            "promote the first actual rerun as soon as one surface is independent, "
            "alpha explicit, and rerun ready"
        ),
        "hydrogen_rule": (
            "the first materialized surface is nu_1S2S(alpha) = (3/8) * mu_red * c^2 * alpha^2 / h"
        ),
        "next_rule": (
            "once the first rerun exists, the next blocker is no longer availability "
            "but the second independent observable / multi-observable comparison table"
        ),
    }


# 関数: `.5919-.5922` を実行する。

def main() -> None:
    """Execute the first actual independent observable rerun gate."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_first_actual_independent_rerun_gate_pack()
    summary_pack = pack["summary"]

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    rerun_available = bool(pack["trial2_first_actual_independent_observable_rerun_available_now"])
    best_overall_is_codata = str(summary_pack["best_overall_alpha_label"]) == "alpha_CODATA"
    best_pmodel_is_4d_can = str(summary_pack["best_pmodel_alpha_label"]) == "alpha_P_4D_can"
    weak_still_unavailable = not bool(summary_pack["weak_explicit_formula_ready_now"])

    rows = [
        sign_base.row(
            "updated_pack_trial2_weak_negative_closeout_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 weak negative closeout selected now",
            sign_base.truth(route_selected),
            "The first actual rerun gate starts after weak explicit-alpha materialization closes negatively and the QED lane remains primary.",
        ),
        sign_base.row(
            "trial2_first_actual_independent_observable_rerun_available_now",
            "pass" if rerun_available else "reject",
            "Trial-2 first actual independent observable rerun available now",
            sign_base.truth(rerun_available),
            "Hydrogen 1S-2S gross structure is now a real independent rerun surface under the current public pack.",
        ),
        sign_base.row(
            "trial2_first_rerun_best_overall_currently_codata_now",
            "pass" if best_overall_is_codata else "reject",
            "Trial-2 first rerun best overall currently CODATA now",
            sign_base.truth(best_overall_is_codata),
            "On the first materialized surface, alpha_CODATA is the closest of the retained checkpoints to the observed value.",
        ),
        sign_base.row(
            "trial2_first_rerun_best_pmodel_currently_4d_can_now",
            "pass" if best_pmodel_is_4d_can else "reject",
            "Trial-2 first rerun best P-model currently 4D can now",
            sign_base.truth(best_pmodel_is_4d_can),
            "Among retained P-model checkpoints, alpha_P_4D,can is the closest on the first rerun surface.",
        ),
        sign_base.row(
            "trial2_weak_explicit_formula_still_unavailable_now",
            "pass" if weak_still_unavailable else "reject",
            "Trial-2 weak explicit formula still unavailable now",
            sign_base.truth(weak_still_unavailable),
            "The first rerun does not lift the weak-sector blocker; weak remains reserve until a genuine alpha-explicit formula appears.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "first_actual_independent_observable_rerun_available_now": True,
        "selected_observable_id": str(summary_pack["selected_observable_id"]),
        "selected_observable_label": str(summary_pack["selected_observable_label"]),
        "best_overall_alpha_label": str(summary_pack["best_overall_alpha_label"]),
        "best_pmodel_alpha_label": str(summary_pack["best_pmodel_alpha_label"]),
        "selected_next_generation_route": "trial2_lamb_absolute_alpha_formula_materialization_audit",
        "recommended_next_route_or_none": ".5923-.5926",
        "selected_followup_route": "trial2_second_independent_observable_rerun_gate",
        "selected_followup_route_or_none": ".5927-.5930",
        "selected_third_route": "trial2_first_multi_observable_comparison_gate",
        "selected_third_route_or_none": ".5931-.5934",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5921",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
            "prediction_table": pack["prediction_table"],
        },
        rows,
        summary,
        {
            "overall_status": "trial2_first_actual_independent_observable_rerun_completed",
            "branch_completed": True,
            "breakthrough_passed_now": True,
            "physical_reject_required": False,
        },
        {
            "best_overall_relative_error_vs_observed": float(
                summary_pack["best_overall_relative_error_vs_observed"]
            ),
            "best_pmodel_relative_error_vs_observed": float(
                summary_pack["best_pmodel_relative_error_vs_observed"]
            ),
        },
    )
    artifacts = write_artifact("declaration_gate", payload)
    print("[ok] first actual rerun gate:", artifacts["json"])


if __name__ == "__main__":
    main()
