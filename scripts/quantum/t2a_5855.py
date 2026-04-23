#!/usr/bin/env python3
"""Generate 8.7.56.5855-.5858 selector 4D mixed-normalization gate artifacts."""

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
        "8.7.56.5851-5854",
        "updated_pack_trial2_selector_4d_mixed_normalization_exactification_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5855-5858"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "selector 4D mixed-normalization gate"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_selector_4d_mixed_normalization_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_selector_4d_mixed_normalization_exactification_audited_gate_primary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_selector_4d_mixed_normalization_weight_candidate_positive_partial_"
    "full_integral_primary_sign_flip_secondary_next"
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
    """Return formulas fixed by the mixed-normalization gate."""
    return {
        "candidate_rule": (
            "alpha_4D,mix,w = alpha_3D / (C_4D^eta_w M_4D^(2-eta_w))"
        ),
        "deterministic_weight": (
            "eta_w = normalized polarization weight of leading_nontrivial_time_component inside the nonzero-time selector family"
        ),
        "gate_reading": (
            "If alpha_4D,mix,w improves the canonical row but still does not collapse to zero residual, the honest next blocker is theorem-level derivation of eta_w or its exact replacement from the 4D full integral."
        ),
    }


# 関数: `.5855-.5858` を実行する。

def main() -> None:
    """Execute the selector-level mixed-normalization gate."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_selector_4d_mixed_normalization_exactification_pack()

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    positive_partial = bool(
        pack["exact_trial2_selector_4d_mixed_normalization_positive_partial_now"]
    )
    zero_residual_unavailable = not bool(
        pack["exact_trial2_selector_4d_zero_residual_theorem_available_now"]
    )
    full_integral_primary = bool(positive_partial and zero_residual_unavailable)
    sign_flip_secondary = bool(full_integral_primary)

    rows = [
        sign_base.row(
            "updated_pack_trial2_selector_4d_mixed_normalization_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 selector 4D mixed-normalization selected now",
            sign_base.truth(route_selected),
            "The gate starts only after the selector-level mixed-normalization audit has fixed one deterministic candidate family.",
        ),
        sign_base.row(
            "exact_trial2_selector_4d_mixed_normalization_positive_partial_now",
            "pass" if positive_partial else "reject",
            "exact Trial-2 selector 4D mixed-normalization positive partial now",
            sign_base.truth(positive_partial),
            "The weighted selector candidate significantly outperforms the canonical 4D row and therefore keeps the mixed-normalization route alive.",
        ),
        sign_base.row(
            "exact_trial2_selector_4d_zero_residual_theorem_unavailable_now",
            "pass" if zero_residual_unavailable else "reject",
            "exact Trial-2 selector 4D zero-residual theorem unavailable now",
            sign_base.truth(zero_residual_unavailable),
            "The current pack still lacks one theorem selecting eta_w exactly enough to force alpha = 1/137 with zero residual.",
        ),
        sign_base.row(
            "updated_pack_trial2_4d_full_integral_primary_now",
            "pass" if full_integral_primary else "reject",
            "updated-pack Trial-2 4D full integral primary now",
            sign_base.truth(full_integral_primary),
            "The strongest next computation route is now the 4D full integral / external-probe current-vertex exactification that could derive the missing mixed-normalization law.",
        ),
        sign_base.row(
            "updated_pack_trial2_sign_flip_secondary_now",
            "pass" if sign_flip_secondary else "reject",
            "updated-pack Trial-2 sign-flip secondary now",
            sign_base.truth(sign_flip_secondary),
            "Sign-flip interpolation remains useful as a diagnostic, but not as the primary theorem route.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "eta_weighted_candidate": float(pack["eta_weighted_candidate"]),
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
        "selected_next_generation_route": (
            "trial2_4d_full_integral_external_probe_current_vertex_exactification_audit"
        ),
        "recommended_next_route_or_none": ".5859-.5862",
        "selected_followup_route": "trial2_sign_flip_interpolation_diagnostic_or_gate",
        "selected_followup_route_or_none": ".5863-.5866",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5857",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
        },
        rows,
        summary,
        {
            "overall_status": "trial2_selector_4d_mixed_normalization_positive_partial",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {
            "pack": pack,
        },
    )
    outputs = write_artifact("declaration_gate", payload)
    print("[done] 8.7.56.5855-5858 selector 4D mixed-normalization gate completed")
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
