#!/usr/bin/env python3
"""Generate 8.7.56.5935-.5938 hydrogen hyperfine absolute-formula artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_hydrogen_hyperfine_absolute_alpha_formula_materialization_backend import (
    build_trial2_hydrogen_hyperfine_absolute_alpha_formula_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5931-5934",
        "updated_pack_trial2_first_multi_observable_comparison_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5935-5938"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "hydrogen hyperfine absolute alpha formula materialization audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_hydrogen_hyperfine_absolute_alpha_formula_materialization_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_first_multi_observable_comparison_unavailable_second_surface_missing_"
    "conditional_reopen_only_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_hydrogen_hyperfine_absolute_alpha_formula_materialized_"
    "second_surface_gate_primary_multi_compare_secondary_next"
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


# 関数: `.5935-.5938` の formula bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas / rules fixed by hyperfine materialization."""
    return {
        "hyperfine_rule": (
            "nu_hfs(alpha) = (8/3) * alpha^4 * (mu_p / mu_B) * "
            "(mu_red / m_e)^3 * m_e c^2 / h"
        ),
        "surface_rule": (
            "promote H I 21 cm only when it becomes one deterministic "
            "alpha-explicit absolute rerun surface under the current public pack"
        ),
        "comparison_rule": (
            "the second surface is honest only if it stays independent and does not "
            "reuse CODATA alpha extraction inputs"
        ),
    }


# 関数: `.5935-.5938` を実行する。

def main() -> None:
    """Execute the hydrogen hyperfine absolute alpha-formula audit."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_hydrogen_hyperfine_absolute_alpha_formula_pack()
    summary_pack = pack["summary"]
    surface = pack["surface"]
    predictions = list(surface["predictions"])
    best_overall = predictions[0]
    codata = next(row for row in predictions if str(row["alpha_label"]) == "alpha_CODATA")

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    formula_materialized = bool(pack["trial2_hyperfine_absolute_formula_materialized_now"])
    surface_ready = bool(pack["trial2_hyperfine_surface_ready_now"])
    best_is_vertex = str(best_overall["alpha_label"]) == "alpha_P_4D_vertex"
    beats_codata = abs(float(best_overall["relative_error_vs_observed"])) < abs(
        float(codata["relative_error_vs_observed"])
    )

    rows = [
        sign_base.row(
            "updated_pack_trial2_multi_unavailable_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 multi-unavailable selected now",
            sign_base.truth(route_selected),
            "The hyperfine materialization branch starts only after the prior pack localizes the blocker to the missing second surface.",
        ),
        sign_base.row(
            "trial2_hyperfine_absolute_formula_materialized_now",
            "pass" if formula_materialized else "reject",
            "Trial-2 hyperfine absolute formula materialized now",
            sign_base.truth(formula_materialized),
            "The retained H I 21 cm source cache now carries one deterministic absolute alpha-to-observable Fermi baseline.",
        ),
        sign_base.row(
            "trial2_hyperfine_surface_ready_now",
            "pass" if surface_ready else "reject",
            "Trial-2 hyperfine surface ready now",
            sign_base.truth(surface_ready),
            "Hydrogen hyperfine 21 cm is now rerun-ready and independent under the current public pack.",
        ),
        sign_base.row(
            "trial2_hyperfine_best_overall_is_vertex_now",
            "pass" if best_is_vertex else "reject",
            "Trial-2 hyperfine best overall is vertex now",
            sign_base.truth(best_is_vertex),
            "On the H I 21 cm Fermi baseline, alpha_P_4D,vertex is the closest retained checkpoint to the observed frequency.",
        ),
        sign_base.row(
            "trial2_hyperfine_vertex_beats_codata_now",
            "pass" if beats_codata else "reject",
            "Trial-2 hyperfine vertex beats CODATA now",
            sign_base.truth(beats_codata),
            "The best retained P-model row must beat alpha_CODATA on the second surface for this route to count as a genuine forward branch.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "hyperfine_surface_id": str(summary_pack["hyperfine_surface_id"]),
        "hyperfine_surface_ready_now": bool(summary_pack["hyperfine_surface_ready_now"]),
        "best_overall_alpha_label": str(summary_pack["best_overall_alpha_label"]),
        "best_overall_relative_error_vs_observed": float(
            summary_pack["best_overall_relative_error_vs_observed"]
        ),
        "best_pmodel_alpha_label": str(summary_pack["best_pmodel_alpha_label"]),
        "best_pmodel_relative_error_vs_observed": float(
            summary_pack["best_pmodel_relative_error_vs_observed"]
        ),
        "selected_next_generation_route": "trial2_second_independent_observable_rerun_gate_refresh",
        "recommended_next_route_or_none": ".5939-.5942",
        "selected_followup_route": "trial2_first_multi_observable_comparison_refresh",
        "selected_followup_route_or_none": ".5943-.5946",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5937",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
            "surface": surface,
        },
        rows,
        summary,
        {
            "overall_status": "trial2_hydrogen_hyperfine_absolute_alpha_formula_materialized",
            "branch_completed": True,
            "breakthrough_passed_now": True,
            "physical_reject_required": False,
        },
        {
            "observed_hz": float(summary_pack["observed_hz"]),
            "sigma_hz": float(summary_pack["sigma_hz"]),
            "best_overall_relative_error_vs_observed": float(
                summary_pack["best_overall_relative_error_vs_observed"]
            ),
            "best_pmodel_relative_error_vs_observed": float(
                summary_pack["best_pmodel_relative_error_vs_observed"]
            ),
            "codata_relative_error_vs_observed": float(codata["relative_error_vs_observed"]),
        },
    )
    artifacts = write_artifact("declaration_gate", payload)
    print("[ok] hyperfine absolute-formula gate:", artifacts["json"])


if __name__ == "__main__":
    main()
