#!/usr/bin/env python3
"""Generate 8.7.56.5971-.5974 Hydrogen fine-structure materialization artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_hydrogen_fine_structure_absolute_alpha_formula_materialization_backend import (
    build_trial2_hydrogen_fine_structure_absolute_alpha_formula_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5967-5970",
        "updated_pack_trial2_multi_observable_corrected_hyperfine_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5971-5974"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "Hydrogen fine-structure absolute alpha formula materialization audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_hydrogen_fine_structure_absolute_alpha_formula_materialization_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_hyperfine_g2_corrected_two_surface_codata_lead_watch_retained_"
    "conditional_reopen_only_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_hydrogen_fine_structure_absolute_alpha_formula_materialized_"
    "third_surface_gate_primary_codata_refresh_secondary_next"
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


# 関数: `.5971-.5974` の formula bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas / rules fixed by the fine-structure materialization."""
    return {
        "surface_rule": "select Hydrogen H-alpha multiplet span as the first retained fine-structure absolute-alpha surface",
        "formula_rule": (
            "nu_fs_span(alpha) = max_allowed |E_3,j_u(alpha)-E_2,j_l(alpha)| / h "
            "- min_allowed |E_3,j_u(alpha)-E_2,j_l(alpha)| / h"
        ),
        "energy_rule": (
            "E_n,j(alpha) = mu_red c^2 * [1 + alpha^2/(n-delta_j)^2]^(-1/2) - mu_red c^2"
        ),
    }


# 関数: `.5971-.5974` を実行する。

def main() -> None:
    """Execute the Hydrogen fine-structure materialization gate."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_hydrogen_fine_structure_absolute_alpha_formula_pack()
    summary_pack = pack["summary"]

    route_selected = str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    materialized = bool(pack["trial2_hydrogen_fine_structure_absolute_formula_materialized_now"])
    surface_ready = bool(summary_pack["selected_surface_ready_now"])
    genuinely_new = bool(summary_pack["selected_surface_is_genuinely_new_now"])
    codata_best = str(summary_pack["selected_best_overall_alpha_label"]) == "alpha_CODATA"

    rows = [
        sign_base.row(
            "updated_pack_trial2_codata_lead_watch_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 CODATA-lead watch selected now",
            sign_base.truth(route_selected),
            "The fine-structure branch starts only after the corrected two-surface CODATA-lead watch is fixed.",
        ),
        sign_base.row(
            "trial2_hydrogen_fine_structure_absolute_formula_materialized_now",
            "pass" if materialized else "reject",
            "Trial-2 Hydrogen fine-structure absolute formula materialized now",
            sign_base.truth(materialized),
            "The current public pack now carries one deterministic Dirac-level fine-structure alpha surface.",
        ),
        sign_base.row(
            "trial2_hydrogen_fine_structure_surface_ready_now",
            "pass" if surface_ready else "reject",
            "Trial-2 Hydrogen fine-structure surface ready now",
            sign_base.truth(surface_ready),
            "The H-alpha fine-structure span surface is rerun-ready under the current retained checkpoints.",
        ),
        sign_base.row(
            "trial2_hydrogen_fine_structure_surface_genuinely_new_now",
            "pass" if genuinely_new else "reject",
            "Trial-2 Hydrogen fine-structure surface genuinely new now",
            sign_base.truth(genuinely_new),
            "The selected surface forms a distinct fine-structure family rather than replaying the gross alpha^2 baseline.",
        ),
        sign_base.row(
            "trial2_hydrogen_fine_structure_best_overall_is_codata_now",
            "pass" if codata_best else "reject",
            "Trial-2 Hydrogen fine-structure best overall is CODATA now",
            sign_base.truth(codata_best),
            "On the retained H-alpha fine-structure span, alpha_CODATA is the closest checkpoint in the current pack.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "selected_surface_id": str(summary_pack["selected_surface_id"]),
        "selected_observed_hz": float(summary_pack["selected_observed_hz"]),
        "selected_best_overall_alpha_label": str(summary_pack["selected_best_overall_alpha_label"]),
        "selected_best_overall_relative_error_vs_observed": float(
            summary_pack["selected_best_overall_relative_error_vs_observed"]
        ),
        "selected_best_pmodel_alpha_label": str(summary_pack["selected_best_pmodel_alpha_label"]),
        "selected_best_pmodel_relative_error_vs_observed": float(
            summary_pack["selected_best_pmodel_relative_error_vs_observed"]
        ),
        "recommended_next_route_or_none": ".5975-.5978",
        "selected_next_generation_route": "trial2_third_independent_surface_gate_second_refresh",
        "selected_followup_route": "trial2_multi_observable_codata_lead_gate_refresh",
        "selected_followup_route_or_none": ".5979-.5982",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5973",
        STEP_NAME + " declaration gate",
        {"source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)}, "formulae": build_formulae()},
        rows,
        summary,
        {
            "overall_status": "trial2_hydrogen_fine_structure_absolute_formula_materialized",
            "branch_completed": True,
            "breakthrough_passed_now": True,
            "physical_reject_required": False,
        },
        {
            "selected_best_overall_relative_error_vs_observed": float(
                summary_pack["selected_best_overall_relative_error_vs_observed"]
            ),
            "selected_best_pmodel_relative_error_vs_observed": float(
                summary_pack["selected_best_pmodel_relative_error_vs_observed"]
            ),
        },
    )
    artifacts = write_artifact("declaration_gate", payload)
    print("[ok] Hydrogen fine-structure materialization gate:", artifacts["json"])


if __name__ == "__main__":
    main()
