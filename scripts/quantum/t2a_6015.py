#!/usr/bin/env python3
"""Generate 8.7.56.6015-.6018 native relativistic Halpha audit artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_native_relativistic_halpha_surface_materialization_backend import (
    build_trial2_native_relativistic_halpha_surface_materialization_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
STATUS = ROOT / "doc" / "STATUS.md"

STEP_TAG = "8.7.56.6015-6018"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor native relativistic "
    "Halpha surface materialization audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "trial2_native_relativistic_halpha_surface_materialization_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_pmodel_native_formula_and_alpha_absolute_condition_fixed_"
    "relativistic_third_surface_primary_diagnostic_tables_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_native_relativistic_halpha_surface_negative_closeout_"
    "third_surface_gate_primary_watch_gate_secondary_next"
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


# 関数: `.6015-.6018` の formula bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas / rules fixed by the native Halpha audit."""
    return {
        "fine_structure_rule": (
            "retain nu_fs_span(alpha) from the reduced-mass Dirac-Coulomb baseline "
            "only as diagnostic until one public-canonical relativistic bridge exists"
        ),
        "part3a_rule": (
            "Part III-A currently publishes the positive-frequency KG -> Schr "
            "envelope, not one literal relativistic bound-state spectrum"
        ),
        "adopted_u1_rule": (
            "the adopted-U(1) sector fixes Coulomb structurally, but does not yet "
            "publish one relative-relativistic bound-state bridge"
        ),
    }


# 関数: `.6015-.6018` を実行する。

def main() -> None:
    """Execute the native relativistic Halpha audit."""
    pack = build_trial2_native_relativistic_halpha_surface_materialization_pack()
    summary_pack = pack["summary"]
    bridge = pack["bridge_checkpoints"]
    prior_text = sign_base.read_text(STATUS)
    prior_class_match = PRIOR_CLASS in prior_text

    rows = [
        sign_base.row(
            "trial2_prior_absolute_condition_state_selected_now",
            "pass" if prior_class_match else "reject",
            "Trial-2 prior absolute-condition state selected now",
            sign_base.truth(prior_class_match),
            "The native Halpha audit starts only after the absolute primary-comparison condition is fixed.",
        ),
        sign_base.row(
            "part3a_positive_frequency_kg_public_now",
            "pass" if bridge["part3a_positive_frequency_kg_public_now"] else "reject",
            "Part III-A positive-frequency KG public now",
            sign_base.truth(bridge["part3a_positive_frequency_kg_public_now"]),
            "Part III-A already publishes the positive-frequency KG starting point.",
        ),
        sign_base.row(
            "part3a_relativistic_bound_state_bridge_public_now",
            "pass" if bridge["part3a_relativistic_bound_state_bridge_public_now"] else "reject",
            "Part III-A relativistic bound-state bridge public now",
            sign_base.truth(bridge["part3a_relativistic_bound_state_bridge_public_now"]),
            "A native Halpha surface needs one public-canonical relativistic bridge, not only the nonrelativistic Schr envelope.",
        ),
        sign_base.row(
            "adopted_u1_relativistic_bound_state_bridge_public_now",
            "pass" if bridge["adopted_u1_relativistic_bound_state_bridge_public_now"] else "reject",
            "Adopted-U(1) relativistic bound-state bridge public now",
            sign_base.truth(bridge["adopted_u1_relativistic_bound_state_bridge_public_now"]),
            "The adopted-U(1) sector must connect the relativistic bound-state formula honestly to the retained Coulomb route.",
        ),
        sign_base.row(
            "trial2_native_relativistic_halpha_surface_ready_now",
            "pass" if summary_pack["native_relativistic_surface_ready_now"] else "reject",
            "Trial-2 native relativistic Halpha surface ready now",
            sign_base.truth(summary_pack["native_relativistic_surface_ready_now"]),
            "Without both relativistic bridges, Halpha remains diagnostic only and cannot enter the native primary table.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "selected_surface_id": str(summary_pack["selected_surface_id"]),
        "native_relativistic_surface_ready_now": bool(summary_pack["native_relativistic_surface_ready_now"]),
        "best_overall_alpha_label": str(summary_pack["best_overall_alpha_label"]),
        "best_overall_relative_error_vs_observed": float(
            summary_pack["best_overall_relative_error_vs_observed"]
        ),
        "best_pmodel_alpha_label": str(summary_pack["best_pmodel_alpha_label"]),
        "best_pmodel_relative_error_vs_observed": float(
            summary_pack["best_pmodel_relative_error_vs_observed"]
        ),
        "selected_next_generation_route": "trial2_native_third_surface_gate",
        "recommended_next_route_or_none": ".6019-.6022",
        "selected_followup_route": "trial2_native_multi_observable_watch_pass_gate",
        "selected_followup_route_or_none": ".6023-.6026",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.6017",
        STEP_NAME + " declaration gate",
        {"source_files": {"status": sign_base.display_path(STATUS)}, "formulae": build_formulae()},
        rows,
        summary,
        {
            "overall_status": "trial2_native_relativistic_halpha_surface_negative_closeout",
            "branch_completed": True,
            "breakthrough_passed_now": False,
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
    print("[ok] native relativistic Halpha audit gate:", artifacts["json"])


if __name__ == "__main__":
    main()
