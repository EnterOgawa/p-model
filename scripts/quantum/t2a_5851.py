#!/usr/bin/env python3
"""Generate 8.7.56.5851-.5854 selector 4D mixed-normalization audit artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_selector_4d_mixed_normalization_exactification_backend import (
    build_trial2_selector_4d_mixed_normalization_exactification_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5847-5850",
        "updated_pack_trial2_4d_full_mode_summation_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5851-5854"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "selector 4D version mixed-normalization exactification audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_selector_4d_mixed_normalization_exactification_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_4d_full_mode_summation_directional_negative_closeout_completed_"
    "selector_4d_primary_full_integral_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_selector_4d_mixed_normalization_exactification_audited_gate_primary_next"
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


# 関数: route で固定する式 bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas fixed by the mixed-normalization audit."""
    return {
        "mixed_family": (
            "alpha_4D,mix(eta) = alpha_3D / (C_4D^eta M_4D^(2-eta))"
        ),
        "exact_goal_interpolant": (
            "eta_* solves alpha_4D,mix(eta_*) = 1/137 inside the fixed selector family."
        ),
        "deterministic_candidate": (
            "eta_w = normalized polarization weight of the leading selector inside the nonzero-time family."
        ),
    }


# 関数: `.5851-.5854` を実行する。

def main() -> None:
    """Execute the selector-level mixed-normalization exactification audit."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_selector_4d_mixed_normalization_exactification_pack()

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    unique_interpolant = bool(pack["exact_trial2_selector_4d_unique_exact_goal_interpolant_now"])
    weighted_improves_canonical = bool(
        pack["exact_trial2_selector_4d_weighted_candidate_improves_canonical_now"]
    )
    weighted_near_goal = bool(
        pack["exact_trial2_selector_4d_weighted_candidate_near_exact_goal_now"]
    )
    positive_partial = bool(
        pack["exact_trial2_selector_4d_mixed_normalization_positive_partial_now"]
    )
    zero_residual_available = bool(
        pack["exact_trial2_selector_4d_zero_residual_theorem_available_now"]
    )

    rows = [
        sign_base.row(
            "updated_pack_trial2_4d_full_mode_summation_gate_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 4D full mode summation gate selected now",
            sign_base.truth(route_selected),
            "The mixed-normalization audit starts only after the full-mode directional gate has closed negatively.",
        ),
        sign_base.row(
            "exact_trial2_selector_4d_unique_exact_goal_interpolant_now",
            "pass" if unique_interpolant else "reject",
            "exact Trial-2 selector 4D unique exact-goal interpolant now",
            sign_base.truth(unique_interpolant),
            "Inside the mixed selector family there is one unique eta_* in (0,1) where the exact goal is reached.",
        ),
        sign_base.row(
            "exact_trial2_selector_4d_weighted_candidate_improves_canonical_now",
            "pass" if weighted_improves_canonical else "reject",
            "exact Trial-2 selector 4D weighted candidate improves canonical now",
            sign_base.truth(weighted_improves_canonical),
            "The deterministic candidate eta_w derived from the selector-family weights improves on the canonical single-row correction.",
        ),
        sign_base.row(
            "exact_trial2_selector_4d_weighted_candidate_near_exact_goal_now",
            "pass" if weighted_near_goal else "reject",
            "exact Trial-2 selector 4D weighted candidate near exact goal now",
            sign_base.truth(weighted_near_goal),
            "The deterministic weighted candidate is already within 1e-5 relative error of the exact goal.",
        ),
        sign_base.row(
            "exact_trial2_selector_4d_mixed_normalization_positive_partial_now",
            "pass" if positive_partial else "reject",
            "exact Trial-2 selector 4D mixed-normalization positive partial now",
            sign_base.truth(positive_partial),
            "The same selector family now carries a stronger residual absorber than the canonical row, even though the exact selector theorem for eta is still open.",
        ),
        sign_base.row(
            "exact_trial2_selector_4d_zero_residual_theorem_available_now",
            "pass" if zero_residual_available else "reject",
            "exact Trial-2 selector 4D zero-residual theorem available now",
            sign_base.truth(zero_residual_available),
            "The audit distinguishes near-exact deterministic improvement from an actual zero-residual theorem.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "eta_exact_goal_interpolant": float(pack["eta_exact_goal_interpolant"]),
        "eta_weighted_candidate": float(pack["eta_weighted_candidate"]),
        "eta_weighted_rel_gap_vs_exact_goal_interpolant": float(
            pack["eta_weighted_rel_gap_vs_exact_goal_interpolant"]
        ),
        "alpha_weighted_candidate": float(pack["alpha_weighted_candidate"]),
        "alpha_weighted_candidate_rel_error_vs_exact_goal": float(
            pack["alpha_weighted_candidate_rel_error_vs_exact_goal"]
        ),
        "weighted_candidate_improvement_factor_vs_canonical": float(
            pack["weighted_candidate_improvement_factor_vs_canonical"]
        ),
        "weighted_candidate_improvement_factor_vs_3d": float(
            pack["weighted_candidate_improvement_factor_vs_3d"]
        ),
        "selected_next_generation_route": "trial2_selector_4d_mixed_normalization_gate",
        "recommended_next_route_or_none": ".5855-.5858",
        "selected_followup_route": "trial2_4d_full_integral_external_probe_current_vertex_exactification_audit",
        "selected_followup_route_or_none": ".5859-.5862",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5853",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
        },
        rows,
        summary,
        {
            "overall_status": "trial2_selector_4d_mixed_normalization_exactification_audited",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {
            "pack": pack,
        },
    )
    outputs = write_artifact("declaration_gate", payload)
    print("[done] 8.7.56.5851-5854 selector 4D mixed-normalization audit completed")
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
