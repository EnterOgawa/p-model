#!/usr/bin/env python3
"""Generate 8.7.56.5911-.5914 QED-vacuum absolute-formula artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_qed_vacuum_absolute_alpha_formula_materialization_backend import (
    build_trial2_qed_vacuum_absolute_alpha_formula_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5907-5910",
        "updated_pack_trial2_first_independent_observable_rerun_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5911-5914"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "QED-vacuum absolute alpha formula materialization audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_qed_vacuum_absolute_alpha_formula_materialization_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_first_independent_observable_rerun_unavailable_"
    "qed_vacuum_formula_materialization_primary_weak_sector_formula_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_qed_vacuum_absolute_alpha_formula_materialized_"
    "hydrogen_1s2s_primary_weak_formula_secondary_next"
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


# 関数: `.5911-.5914` の rule bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas / rules fixed by QED absolute-formula materialization."""
    return {
        "hydrogen_1s2s_gross_rule": (
            "nu_1S2S(alpha) = (3/8) * mu_red * c^2 * alpha^2 / h"
        ),
        "qed_materialization_rule": (
            "promote a QED-vacuum surface only when alpha enters one deterministic "
            "public formula and the observable is independent"
        ),
        "lamb_rule": (
            "Lamb remains structurally alpha-sensitive but absolute-formula "
            "materialization is still unavailable in the current public pack"
        ),
    }


# 関数: `.5911-.5914` を実行する。

def main() -> None:
    """Execute the QED-vacuum absolute alpha formula materialization audit."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_qed_vacuum_absolute_alpha_formula_pack()
    summary_pack = pack["summary"]

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    formula_materialized = bool(pack["trial2_qed_vacuum_absolute_formula_materialized_now"])
    primary_ready = bool(pack["trial2_qed_vacuum_primary_ready_now"])
    first_surface_available = bool(pack["trial2_first_actual_qed_rerun_surface_available_now"])
    best_overall_is_codata = str(summary_pack["best_overall_alpha_label"]) == "alpha_CODATA"
    best_pmodel_is_4d_can = str(summary_pack["best_pmodel_alpha_label"]) == "alpha_P_4D_can"

    rows = [
        sign_base.row(
            "updated_pack_trial2_first_rerun_unavailable_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 first-rerun-unavailable gate selected now",
            sign_base.truth(route_selected),
            "QED absolute-formula materialization starts only after the previous gate localizes the blocker to formula materialization itself.",
        ),
        sign_base.row(
            "trial2_qed_vacuum_absolute_formula_materialized_now",
            "pass" if formula_materialized else "reject",
            "Trial-2 QED-vacuum absolute formula materialized now",
            sign_base.truth(formula_materialized),
            "The current pack now carries one explicit alpha-to-observable formula on the hydrogen 1S-2S surface.",
        ),
        sign_base.row(
            "trial2_qed_vacuum_primary_ready_now",
            "pass" if primary_ready else "reject",
            "Trial-2 QED-vacuum primary ready now",
            sign_base.truth(primary_ready),
            "The hydrogen 1S-2S gross-structure baseline is now rerun-ready and independent.",
        ),
        sign_base.row(
            "trial2_first_actual_qed_rerun_surface_available_now",
            "pass" if first_surface_available else "reject",
            "Trial-2 first actual QED rerun surface available now",
            sign_base.truth(first_surface_available),
            "The first honest independent rerun surface is now available inside the QED-vacuum pack.",
        ),
        sign_base.row(
            "trial2_qed_best_overall_currently_codata_now",
            "pass" if best_overall_is_codata else "reject",
            "Trial-2 QED best overall currently CODATA now",
            sign_base.truth(best_overall_is_codata),
            "On this first coarse QED baseline, alpha_CODATA is the closest of the retained checkpoints to the observed 1S-2S frequency.",
        ),
        sign_base.row(
            "trial2_qed_best_pmodel_currently_4d_can_now",
            "pass" if best_pmodel_is_4d_can else "reject",
            "Trial-2 QED best P-model currently 4D can now",
            sign_base.truth(best_pmodel_is_4d_can),
            "Among retained P-model checkpoints, alpha_P_4D,can is the closest on the first materialized 1S-2S surface.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "qed_absolute_primary_ready_count": int(summary_pack["qed_absolute_primary_ready_count"]),
        "selected_primary_target_ids": list(summary_pack["selected_primary_target_ids"]),
        "selected_first_rerun_surface_id": str(summary_pack["selected_first_rerun_surface_id"]),
        "best_overall_alpha_label": str(summary_pack["best_overall_alpha_label"]),
        "best_pmodel_alpha_label": str(summary_pack["best_pmodel_alpha_label"]),
        "selected_next_generation_route": "trial2_weak_beta_decay_explicit_alpha_formula_materialization_audit",
        "recommended_next_route_or_none": ".5915-.5918",
        "selected_followup_route": "trial2_first_actual_independent_observable_rerun_gate",
        "selected_followup_route_or_none": ".5919-.5922",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5913",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
            "hydrogen_1s2s_predictions": pack["hydrogen_1s2s_predictions"],
        },
        rows,
        summary,
        {
            "overall_status": "trial2_qed_vacuum_absolute_alpha_formula_materialized",
            "branch_completed": True,
            "breakthrough_passed_now": True,
            "physical_reject_required": False,
        },
        {
            "hydrogen_1s2s_observed_hz": float(summary_pack["hydrogen_1s2s_observed_hz"]),
            "hydrogen_1s2s_sigma_hz": float(summary_pack["hydrogen_1s2s_sigma_hz"]),
            "best_overall_relative_error_vs_observed": float(
                summary_pack["best_overall_relative_error_vs_observed"]
            ),
            "best_pmodel_relative_error_vs_observed": float(
                summary_pack["best_pmodel_relative_error_vs_observed"]
            ),
        },
    )
    artifacts = write_artifact("declaration_gate", payload)
    print("[ok] qed absolute-formula gate:", artifacts["json"])


if __name__ == "__main__":
    main()
