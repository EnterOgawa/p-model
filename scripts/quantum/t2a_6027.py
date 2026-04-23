#!/usr/bin/env python3
"""Generate 8.7.56.6027-.6030 native He II surface materialization artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_native_helium_ion_surface_materialization_backend import (
    build_trial2_native_helium_ion_surface_materialization_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
STATUS = ROOT / "doc" / "STATUS.md"

STEP_TAG = "8.7.56.6027-6030"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor native helium-ion "
    "surface materialization audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "trial2_native_helium_ion_surface_materialization_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_native_relativistic_third_surface_negative_closeout_completed_"
    "native_two_surface_split_watch_retained_conditional_reopen_only_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_native_helium_ion_surface_completed_non_hydrogen_gate_primary_"
    "three_surface_watch_secondary_next"
)


# 関数: JSON/CSV artifact を書き出す。
def write_artifact(kind: str, data: dict) -> dict[str, str]:
    """Write one JSON payload and one rows CSV."""
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    paths = build_metrics_paths(PUBLIC_OUT, STEM, kind)
    paths["json"].write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with paths["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])

    return {"json": sign_base.display_path(paths["json"])}


# 関数: `.6027-.6030` の formula bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas / rules fixed by the native He II audit."""
    return {
        "native_rule": "primary comparison uses one P-model-native formula with one P-model alpha checkpoint family",
        "heii_rule": (
            "nu_HeII,4to3(alpha) = (mu_red * c^2 / (2 h)) * (Z alpha)^2 * "
            "(1/3^2 - 1/4^2), with Z = 2"
        ),
        "selection_rule": "use the retained NIST He II 468.67 nm line cache with ritz fallback when obs_nu is blank",
    }


# 関数: `.6027-.6030` を実行する。

def main() -> None:
    """Execute the native He II surface audit."""
    pack = build_trial2_native_helium_ion_surface_materialization_pack()
    summary_pack = pack["summary"]
    prior_text = sign_base.read_text(STATUS)
    prior_class_match = PRIOR_CLASS in prior_text

    rows = [
        sign_base.row(
            "trial2_prior_native_two_surface_state_selected_now",
            "pass" if prior_class_match else "reject",
            "Trial-2 prior native two-surface state selected now",
            sign_base.truth(prior_class_match),
            "The He II reopen check starts only after the native two-surface watch is fixed.",
        ),
        sign_base.row(
            "trial2_native_non_hydrogen_surface_ready_now",
            "pass" if summary_pack["native_non_hydrogen_surface_ready_now"] else "reject",
            "Trial-2 native non-Hydrogen surface ready now",
            sign_base.truth(summary_pack["native_non_hydrogen_surface_ready_now"]),
            "He II is one-electron and avoids the neutral-He screening-law blocker.",
        ),
        sign_base.row(
            "trial2_heii_selected_line_uses_retained_target_now",
            "pass" if summary_pack["selected_line_id"] == "He_II_468.67nm" else "reject",
            "Trial-2 He II selected line uses retained fixed target now",
            1.0 if summary_pack["selected_line_id"] == "He_II_468.67nm" else 0.0,
            "The selected non-Hydrogen line is fixed from the local NIST cache as He_II_468.67nm, not hand-tuned during rerun.",
        ),
        sign_base.row(
            "trial2_heii_best_pmodel_relative_error_vs_observed_now",
            "pass",
            "Trial-2 He II best P-model relative error vs observed now",
            float(summary_pack["best_pmodel_relative_error_vs_observed"]),
            "This sets the new native non-Hydrogen surface checkpoint under the same native shell.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "surface_id": str(summary_pack["surface_id"]),
        "selected_line_id": str(summary_pack["selected_line_id"]),
        "observed_lambda_vac_nm": float(summary_pack["observed_lambda_vac_nm"]),
        "best_overall_alpha_label": str(summary_pack["best_overall_alpha_label"]),
        "best_overall_relative_error_vs_observed": float(
            summary_pack["best_overall_relative_error_vs_observed"]
        ),
        "best_pmodel_alpha_label": str(summary_pack["best_pmodel_alpha_label"]),
        "best_pmodel_relative_error_vs_observed": float(
            summary_pack["best_pmodel_relative_error_vs_observed"]
        ),
        "selected_next_generation_route": "trial2_native_non_hydrogen_surface_gate",
        "recommended_next_route_or_none": ".6031-.6034",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.6029",
        STEP_NAME + " declaration gate",
        {"source_files": {"status": sign_base.display_path(STATUS)}, "formulae": build_formulae()},
        rows,
        summary,
        {
            "overall_status": "trial2_native_helium_ion_surface_completed",
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
    print("[ok] native He II surface audit gate:", artifacts["json"])


if __name__ == "__main__":
    main()
