#!/usr/bin/env python3
"""Generate 8.7.56.5867-.5870 exact-goal closeout followup artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_exact_goal_closeout_followup_backend import (
    build_trial2_exact_goal_closeout_followup_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5863-5866",
        "updated_pack_trial2_sign_flip_interpolation_selector_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5867-5870"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "exact-goal closeout followup"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_exact_goal_closeout_followup",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_4d_full_integral_external_probe_current_vertex_positive_partial_"
    "exact_goal_followup_primary_sign_flip_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_4d_external_probe_exact_goal_closeout_followup_audited_"
    "sign_flip_primary_weight_source_secondary_gate"
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


# 関数: gate で固定する式 bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the exact-goal closeout followup."""
    return {
        "mixed_family": (
            "alpha_4D,mix(eta) = alpha_3D / (C_4D^eta M_4D^(2-eta))"
        ),
        "selector_gap": "delta_eta = eta_vertex - eta_*",
        "linearized_gap": (
            "delta alpha_lin = alpha_goal * ln(M_4D / C_4D) * delta_eta"
        ),
    }


# 関数: `.5867-.5870` を実行する。

def main() -> None:
    """Execute the exact-goal closeout followup."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_exact_goal_closeout_followup_pack()

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    eta_gap_localized = bool(
        pack["exact_trial2_4d_vertex_gap_localized_as_eta_gap_now"]
    )
    not_numerical_noise = bool(
        pack["exact_trial2_4d_vertex_gap_not_numerical_noise_now"]
    )
    zero_residual_unavailable = not bool(
        pack["exact_trial2_4d_zero_residual_exact_goal_available_now"]
    )
    closeout_unavailable = bool(
        pack["exact_trial2_4d_exact_goal_closeout_unavailable_current_pack_now"]
    )

    rows = [
        sign_base.row(
            "updated_pack_trial2_4d_external_probe_positive_partial_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 4D external-probe positive partial selected now",
            sign_base.truth(route_selected),
            "The closeout followup starts only after the external-probe current-vertex candidate is fixed as the strongest live deterministic route.",
        ),
        sign_base.row(
            "exact_trial2_4d_vertex_gap_localized_as_eta_gap_now",
            "pass" if eta_gap_localized else "reject",
            "exact Trial-2 4D vertex gap localized as eta gap now",
            sign_base.truth(eta_gap_localized),
            "The remaining exact-goal gap is explained by the selector-weight mismatch delta_eta rather than by a missing order-one effect.",
        ),
        sign_base.row(
            "exact_trial2_4d_vertex_gap_not_numerical_noise_now",
            "pass" if not_numerical_noise else "reject",
            "exact Trial-2 4D vertex gap not numerical noise now",
            sign_base.truth(not_numerical_noise),
            "The residual stays far above the numerical-noise floor, so zero residual cannot be claimed as a rounding artifact.",
        ),
        sign_base.row(
            "exact_trial2_4d_zero_residual_exact_goal_unavailable_now",
            "pass" if zero_residual_unavailable else "reject",
            "exact Trial-2 4D zero-residual exact goal unavailable now",
            sign_base.truth(zero_residual_unavailable),
            "The deterministic external-probe candidate remains near exact but does not collapse to the exact goal.",
        ),
        sign_base.row(
            "updated_pack_trial2_exact_goal_closeout_unavailable_current_pack_now",
            "pass" if closeout_unavailable else "reject",
            "updated-pack Trial-2 exact-goal closeout unavailable current pack now",
            sign_base.truth(closeout_unavailable),
            "The remaining blocker is theorem-level source selection for the deterministic external-probe weight, not a missing large correction.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "alpha_vertex_candidate_abs_gap_vs_exact_goal": float(
            pack["alpha_vertex_candidate_abs_gap_vs_exact_goal"]
        ),
        "alpha_vertex_candidate_rel_gap_vs_exact_goal": float(
            pack["alpha_vertex_candidate_rel_gap_vs_exact_goal"]
        ),
        "delta_eta_vertex_minus_exact_goal_interpolant": float(
            pack["delta_eta_vertex_minus_exact_goal_interpolant"]
        ),
        "linearization_ratio_actual_over_linearized": float(
            pack["linearization_ratio_actual_over_linearized"]
        ),
        "required_denominator_multiplier_for_exact_goal": float(
            pack["required_denominator_multiplier_for_exact_goal"]
        ),
        "selected_next_generation_route": "trial2_sign_flip_interpolation_diagnostic",
        "recommended_next_route_or_none": ".5871-.5874",
        "selected_followup_route": "trial2_external_probe_weight_theorem_source_audit",
        "selected_followup_route_or_none": ".5875-.5878",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5869",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
        },
        rows,
        summary,
        {
            "overall_status": "trial2_exact_goal_closeout_followup_audited",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {
            "pack": pack,
        },
    )
    outputs = write_artifact("declaration_gate", payload)
    print("[done] 8.7.56.5867-5870 exact-goal closeout followup completed")
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
