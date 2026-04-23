#!/usr/bin/env python3
"""Generate 8.7.56.5923-.5926 Lamb absolute-formula artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_lamb_absolute_alpha_formula_materialization_backend import (
    build_trial2_lamb_absolute_alpha_formula_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5919-5922",
        "updated_pack_trial2_first_actual_independent_observable_rerun_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5923-5926"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "Lamb absolute alpha formula materialization audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_lamb_absolute_alpha_formula_materialization_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_first_actual_independent_observable_rerun_completed_"
    "hydrogen_1s2s_gross_structure_lamb_absolute_formula_primary_"
    "multi_observable_gate_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_lamb_absolute_formula_negative_closeout_completed_"
    "second_observable_primary_multi_observable_secondary_next"
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


# 関数: `.5923-.5926` の rule bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas / rules fixed by the Lamb audit."""
    return {
        "lamb_rule": (
            "retain Lamb only when the public pack carries one deterministic "
            "absolute alpha-to-observable map"
        ),
        "scaling_rule": (
            "Z^4 and Z^6 scaling plus nuclear tables are structural evidence, "
            "not yet an absolute rerun formula"
        ),
        "gate_rule": (
            "without one absolute Lamb formula, Hydrogen 1S-2S remains the only "
            "actual independent rerun surface"
        ),
    }


# 関数: `.5923-.5926` を実行する。

def main() -> None:
    """Execute the Lamb absolute alpha formula materialization audit."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_lamb_absolute_alpha_formula_pack()
    summary_pack = pack["summary"]

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    formula_unavailable = not bool(summary_pack["lamb_absolute_formula_materialized_now"])
    structural_retained = bool(summary_pack["lamb_structural_alpha_sensitivity_retained_now"])
    hydrogen_only_actual = bool(summary_pack["hydrogen_surface_still_only_actual_rerun_surface_now"])

    rows = [
        sign_base.row(
            "updated_pack_trial2_first_actual_rerun_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 first actual rerun selected now",
            sign_base.truth(route_selected),
            "Lamb materialization starts only after the first actual independent rerun surface is fixed.",
        ),
        sign_base.row(
            "trial2_lamb_absolute_formula_unavailable_now",
            "pass" if formula_unavailable else "reject",
            "Trial-2 Lamb absolute formula unavailable now",
            sign_base.truth(formula_unavailable),
            "The current public Lamb pack still lacks one deterministic absolute alpha-to-observable formula.",
        ),
        sign_base.row(
            "trial2_lamb_structural_alpha_sensitivity_retained_now",
            "pass" if structural_retained else "reject",
            "Trial-2 Lamb structural alpha sensitivity retained now",
            sign_base.truth(structural_retained),
            "Z^4 / Z^6 scaling and retained nuclear tables keep Lamb alive as a structural candidate even while the absolute formula is missing.",
        ),
        sign_base.row(
            "trial2_hydrogen_surface_still_only_actual_rerun_surface_now",
            "pass" if hydrogen_only_actual else "reject",
            "Trial-2 hydrogen surface still only actual rerun surface now",
            sign_base.truth(hydrogen_only_actual),
            "Without Lamb absolute-formula materialization, Hydrogen 1S-2S remains the only actual rerun-ready independent surface.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "lamb_absolute_formula_materialized_now": False,
        "qed_actual_rerun_surface_count_now": int(summary_pack["qed_actual_rerun_surface_count_now"]),
        "lamb_z_grid_count": int(summary_pack["lamb_z_grid_count"]),
        "lamb_nuclear_table_count": int(summary_pack["lamb_nuclear_table_count"]),
        "selected_next_generation_route": "trial2_second_independent_observable_rerun_gate",
        "recommended_next_route_or_none": ".5927-.5930",
        "selected_followup_route": "trial2_first_multi_observable_comparison_gate",
        "selected_followup_route_or_none": ".5931-.5934",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5925",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
            "lamb_surface": pack["lamb_surface"],
        },
        rows,
        summary,
        {
            "overall_status": "trial2_lamb_absolute_formula_negative_closeout_completed",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {
            "qed_actual_rerun_surface_count_now": int(summary_pack["qed_actual_rerun_surface_count_now"]),
            "lamb_z_grid_count": int(summary_pack["lamb_z_grid_count"]),
            "lamb_nuclear_table_count": int(summary_pack["lamb_nuclear_table_count"]),
        },
    )
    artifacts = write_artifact("declaration_gate", payload)
    print("[ok] lamb absolute-formula gate:", artifacts["json"])


if __name__ == "__main__":
    main()
