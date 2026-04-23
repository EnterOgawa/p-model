#!/usr/bin/env python3
"""Generate 8.7.56.5859-.5862 4D full-integral external-probe audit artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_4d_full_integral_external_probe_current_vertex_exactification_backend import (
    build_trial2_4d_full_integral_external_probe_current_vertex_exactification_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5855-5858",
        "updated_pack_trial2_selector_4d_mixed_normalization_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5859-5862"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "4D full integral external-probe current-vertex exactification audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_4d_full_integral_external_probe_current_vertex_exactification_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_selector_4d_mixed_normalization_weight_candidate_positive_partial_"
    "full_integral_primary_sign_flip_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_4d_full_integral_external_probe_current_vertex_exactification_"
    "audited_exact_goal_primary_sign_flip_secondary_gate"
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
    """Return formulas fixed by the 4D full-integral audit."""
    return {
        "external_probe_current": (
            "J_ext^mu[Q](x) := delta S_frozen[Q;A] / delta A_mu(x) |_(A=0)"
        ),
        "full_integral_weight_rule": (
            "eta_vertex := normalized leading nonzero-time selector weight "
            "inside the retained full-integral family"
        ),
        "vertex_candidate_rule": (
            "alpha_4D,vertex = alpha_3D / (C_4D^eta_vertex M_4D^(2-eta_vertex))"
        ),
    }


# 関数: `.5859-.5862` を実行する。

def main() -> None:
    """Execute the 4D full-integral / external-probe current-vertex audit."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_4d_full_integral_external_probe_current_vertex_exactification_pack()

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    target_surface_explicit = bool(
        pack["exact_trial2_4d_external_probe_current_vertex_target_surface_explicit_now"]
    )
    weight_candidate_available = bool(
        pack["exact_trial2_4d_external_probe_weight_candidate_available_now"]
    )
    positive_partial = bool(
        pack["exact_trial2_4d_external_probe_current_vertex_positive_partial_now"]
    )
    near_exact_goal = bool(
        pack["exact_trial2_4d_external_probe_current_vertex_near_exact_goal_now"]
    )
    zero_residual_unavailable = not bool(
        pack["exact_trial2_4d_zero_residual_exact_goal_available_now"]
    )
    exact_goal_followup_required = bool(
        pack["exact_trial2_4d_exact_goal_closeout_followup_required_now"]
    )

    rows = [
        sign_base.row(
            "updated_pack_trial2_selector_4d_mixed_normalization_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 selector 4D mixed-normalization selected now",
            sign_base.truth(route_selected),
            "The full-integral route starts only after selector-level mixed normalization has been fixed as the strongest current family.",
        ),
        sign_base.row(
            "exact_trial2_4d_external_probe_current_vertex_target_surface_explicit_now",
            "pass" if target_surface_explicit else "reject",
            "exact Trial-2 4D external-probe current-vertex target surface explicit now",
            sign_base.truth(target_surface_explicit),
            "The retained 4D current-vertex lesson is now connected to the current exact-goal pack as a computation-first target surface.",
        ),
        sign_base.row(
            "exact_trial2_4d_external_probe_weight_candidate_available_now",
            "pass" if weight_candidate_available else "reject",
            "exact Trial-2 4D external-probe weight candidate available now",
            sign_base.truth(weight_candidate_available),
            "The normalized leading nonzero-time selector weight gives one deterministic external-probe participation fraction with no new free parameter.",
        ),
        sign_base.row(
            "exact_trial2_4d_external_probe_current_vertex_positive_partial_now",
            "pass" if positive_partial else "reject",
            "exact Trial-2 4D external-probe current-vertex positive partial now",
            sign_base.truth(positive_partial),
            "The resulting full-integral current-vertex candidate improves the canonical 4D row and therefore keeps the 4D route alive.",
        ),
        sign_base.row(
            "exact_trial2_4d_external_probe_current_vertex_near_exact_goal_now",
            "pass" if near_exact_goal else "reject",
            "exact Trial-2 4D external-probe current-vertex near exact goal now",
            sign_base.truth(near_exact_goal),
            "The deterministic full-integral candidate is already within 1e-5 relative error of the exact goal.",
        ),
        sign_base.row(
            "exact_trial2_4d_zero_residual_exact_goal_unavailable_now",
            "pass" if zero_residual_unavailable else "reject",
            "exact Trial-2 4D zero-residual exact goal unavailable now",
            sign_base.truth(zero_residual_unavailable),
            "The route is still computation-backed exactification, not a zero-residual theorem.",
        ),
        sign_base.row(
            "updated_pack_trial2_exact_goal_closeout_followup_required_now",
            "pass" if exact_goal_followup_required else "reject",
            "updated-pack Trial-2 exact-goal closeout followup required now",
            sign_base.truth(exact_goal_followup_required),
            "The next honest blocker is the exact-goal closeout verdict for the external-probe candidate itself.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "eta_exact_goal_interpolant": float(pack["eta_exact_goal_interpolant"]),
        "eta_vertex_weight_candidate": float(pack["eta_vertex_weight_candidate"]),
        "eta_vertex_rel_gap_vs_exact_goal_interpolant": float(
            pack["eta_vertex_rel_gap_vs_exact_goal_interpolant"]
        ),
        "alpha_vertex_candidate": float(pack["alpha_vertex_candidate"]),
        "alpha_vertex_candidate_rel_error_vs_exact_goal": float(
            pack["alpha_vertex_candidate_rel_error_vs_exact_goal"]
        ),
        "vertex_candidate_improvement_factor_vs_canonical": float(
            pack["vertex_candidate_improvement_factor_vs_canonical"]
        ),
        "vertex_candidate_improvement_factor_vs_3d": float(
            pack["vertex_candidate_improvement_factor_vs_3d"]
        ),
        "vertex_candidate_improvement_factor_vs_best_full_mode": float(
            pack["vertex_candidate_improvement_factor_vs_best_full_mode"]
        ),
        "selected_next_generation_route": "trial2_sign_flip_interpolation_selector_gate",
        "recommended_next_route_or_none": ".5863-.5866",
        "selected_followup_route": "trial2_exact_goal_closeout_followup_audit",
        "selected_followup_route_or_none": ".5867-.5870",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5861",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
        },
        rows,
        summary,
        {
            "overall_status": "trial2_4d_full_integral_external_probe_current_vertex_exactification_audited",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {
            "pack": pack,
        },
    )
    outputs = write_artifact("declaration_gate", payload)
    print(
        "[done] 8.7.56.5859-5862 4D full-integral external-probe current-vertex audit completed"
    )
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
