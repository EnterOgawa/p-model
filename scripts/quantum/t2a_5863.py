#!/usr/bin/env python3
"""Generate 8.7.56.5863-.5866 sign-flip interpolation selector gate artifacts."""

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
        "8.7.56.5859-5862",
        "updated_pack_trial2_4d_full_integral_external_probe_current_vertex_exactification_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5863-5866"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "sign-flip interpolation selector gate"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_sign_flip_interpolation_selector_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_4d_full_integral_external_probe_current_vertex_exactification_"
    "audited_exact_goal_primary_sign_flip_secondary_gate"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_4d_full_integral_external_probe_current_vertex_positive_partial_"
    "exact_goal_followup_primary_sign_flip_secondary_next"
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
    """Return formulas fixed by the sign-flip selector gate."""
    return {
        "mixed_family": (
            "alpha_4D,mix(eta) = alpha_3D / (C_4D^eta M_4D^(2-eta))"
        ),
        "sign_flip_bracket": (
            "alpha_4D,mix(0) = alpha_4D,can < 1/137 < alpha_4D,mix(1)"
        ),
        "deterministic_candidate": (
            "eta_vertex := normalized leading nonzero-time selector weight "
            "from the retained full-integral family"
        ),
    }


# 関数: `.5863-.5866` を実行する。

def main() -> None:
    """Execute the sign-flip interpolation selector gate."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_4d_full_integral_external_probe_current_vertex_exactification_pack()

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    sign_flip_bracket = bool(pack["exact_trial2_4d_sign_flip_bracket_retained_now"])
    positive_partial = bool(
        pack["exact_trial2_4d_external_probe_current_vertex_positive_partial_now"]
    )
    zero_residual_unavailable = not bool(
        pack["exact_trial2_4d_zero_residual_exact_goal_available_now"]
    )
    eta_vertex_above_eta_star = bool(
        float(pack["eta_vertex_weight_candidate"])
        > float(pack["eta_exact_goal_interpolant"])
    )
    exact_goal_followup_primary = bool(positive_partial and zero_residual_unavailable)
    sign_flip_secondary = bool(sign_flip_bracket)

    rows = [
        sign_base.row(
            "updated_pack_trial2_4d_full_integral_external_probe_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 4D full-integral external-probe selected now",
            sign_base.truth(route_selected),
            "The selector gate starts only after the external-probe current-vertex audit has been fixed as the strongest live 4D route.",
        ),
        sign_base.row(
            "exact_trial2_4d_sign_flip_bracket_retained_now",
            "pass" if sign_flip_bracket else "reject",
            "exact Trial-2 4D sign-flip bracket retained now",
            sign_base.truth(sign_flip_bracket),
            "The mixed 4D family still brackets the exact goal from opposite sides, so interpolation remains a valid diagnostic surface.",
        ),
        sign_base.row(
            "exact_trial2_eta_vertex_above_eta_star_now",
            "pass" if eta_vertex_above_eta_star else "reject",
            "exact Trial-2 eta_vertex above eta_star now",
            sign_base.truth(eta_vertex_above_eta_star),
            "The deterministic full-integral weight sits on the overshoot side of the unique interpolant, so the remaining gap is localized rather than random.",
        ),
        sign_base.row(
            "exact_trial2_4d_external_probe_current_vertex_positive_partial_now",
            "pass" if positive_partial else "reject",
            "exact Trial-2 4D external-probe current-vertex positive partial now",
            sign_base.truth(positive_partial),
            "The external-probe reinterpretation improves the canonical 4D row and therefore keeps the route genuinely alive.",
        ),
        sign_base.row(
            "exact_trial2_4d_zero_residual_exact_goal_unavailable_now",
            "pass" if zero_residual_unavailable else "reject",
            "exact Trial-2 4D zero-residual exact goal unavailable now",
            sign_base.truth(zero_residual_unavailable),
            "Even with the deterministic full-integral weight candidate, the current pack still stops short of zero residual.",
        ),
        sign_base.row(
            "updated_pack_trial2_exact_goal_closeout_followup_primary_now",
            "pass" if exact_goal_followup_primary else "reject",
            "updated-pack Trial-2 exact-goal closeout followup primary now",
            sign_base.truth(exact_goal_followup_primary),
            "The most honest next blocker is now the exact-goal closeout verdict for the external-probe candidate itself.",
        ),
        sign_base.row(
            "updated_pack_trial2_sign_flip_secondary_now",
            "pass" if sign_flip_secondary else "reject",
            "updated-pack Trial-2 sign-flip secondary now",
            sign_base.truth(sign_flip_secondary),
            "Sign-flip interpolation remains useful as a diagnostic consistency check, but not as the primary theorem route.",
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
        "selected_next_generation_route": "trial2_exact_goal_closeout_followup_audit",
        "recommended_next_route_or_none": ".5867-.5870",
        "selected_followup_route": "trial2_sign_flip_interpolation_diagnostic",
        "selected_followup_route_or_none": ".5871-.5874",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5865",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
        },
        rows,
        summary,
        {
            "overall_status": "trial2_4d_external_probe_positive_partial_sign_flip_gated",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {
            "pack": pack,
        },
    )
    outputs = write_artifact("declaration_gate", payload)
    print("[done] 8.7.56.5863-5866 sign-flip interpolation selector gate completed")
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
