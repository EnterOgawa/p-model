#!/usr/bin/env python3
"""Generate 8.7.56.5843-.5846 Trial-2 4D full mode summation directional artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_4d_full_mode_summation_directional_check_backend import (
    build_trial2_4d_full_mode_summation_directional_check_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5839-5842",
        "updated_pack_trial2_4d_exact_goal_closeout_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5843-5846"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "4D full mode summation directional check audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_4d_full_mode_summation_directional_check_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_4d_canonical_exact_alpha_correction_completed_"
    "zero_residual_exact_goal_unavailable_current_pack_conditional_reopen_only_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_4d_full_mode_summation_directional_audited_gate_primary_next"
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
    """Return formulas fixed by the full-mode directional check."""
    return {
        "full_mode_mass_sq_rule": (
            "alpha_4D,full[M^2] = alpha_3D / sum_i w_hat_i M_4D(i)^2"
        ),
        "full_mode_charge_mass_rule": (
            "alpha_4D,full[CM] = alpha_3D / sum_i w_hat_i C_4D(i) M_4D(i)"
        ),
        "directional_reading": (
            "If the best full-mode aggregate improves the 3D baseline but still "
            "fails to beat the canonical single-row 4D correction, the route "
            "closes negatively as a main exact-goal extractor and instead "
            "supports selector-4D mixed-normalization exactification."
        ),
    }


# 関数: `.5843-.5846` を実行する。

def main() -> None:
    """Execute the 4D full mode summation directional check audit."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_4d_full_mode_summation_directional_check_pack()
    best_row = pack["best_row"]

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    full_mode_family_available = bool(len(pack["aggregated_rows"]) >= 2)
    best_improves_baseline = bool(pack["best_improves_baseline_now"])
    best_beats_canonical = bool(pack["best_beats_canonical_now"])
    directional_negative_closeout = bool(
        pack["exact_trial2_4d_full_mode_directional_negative_closeout_now"]
    )
    selector_mixed_normalization_supported = bool(
        pack["exact_trial2_4d_selector_mixed_normalization_route_supported_now"]
    )

    rows = [
        sign_base.row(
            "updated_pack_trial2_4d_exact_goal_closeout_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 4D exact-goal closeout selected now",
            sign_base.truth(route_selected),
            "The full mode summation directional check starts only after the canonical 4D exact-goal closeout gate has been fixed.",
        ),
        sign_base.row(
            "exact_trial2_4d_full_mode_family_machine_readable_now",
            "pass" if full_mode_family_available else "reject",
            "exact Trial-2 4D full mode family machine readable now",
            sign_base.truth(full_mode_family_available),
            "The current deterministic selector family can now be aggregated into explicit weighted full-mode candidates.",
        ),
        sign_base.row(
            "exact_trial2_4d_full_mode_best_row_improves_baseline_now",
            "pass" if best_improves_baseline else "reject",
            "exact Trial-2 4D full mode best row improves baseline now",
            sign_base.truth(best_improves_baseline),
            "The best summation candidate does shrink the 3D exact-goal residual relative to the uncorrrected symbolic value.",
        ),
        sign_base.row(
            "exact_trial2_4d_full_mode_best_row_beats_canonical_now",
            "pass" if best_beats_canonical else "reject",
            "exact Trial-2 4D full mode best row beats canonical now",
            sign_base.truth(best_beats_canonical),
            "This is the key question: whether family-level accumulation can outperform the already canonized single-row 4D correction.",
        ),
        sign_base.row(
            "exact_trial2_4d_full_mode_directional_negative_closeout_now",
            "pass" if directional_negative_closeout else "reject",
            "exact Trial-2 4D full mode directional negative closeout now",
            sign_base.truth(directional_negative_closeout),
            "Because the best summation candidate improves the 3D baseline but still loses to the canonical row, full mode accumulation is a directional diagnostic rather than a main residual-closing route.",
        ),
        sign_base.row(
            "exact_trial2_4d_selector_mixed_normalization_route_supported_now",
            "pass" if selector_mixed_normalization_supported else "reject",
            "exact Trial-2 4D selector mixed-normalization route supported now",
            sign_base.truth(selector_mixed_normalization_supported),
            "The sign structure overshoot(full sum) / undershoot(canonical) supports a missing selector-level mixed-normalization law rather than simple family averaging.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "best_selector_set": str(best_row["selector_set"]),
        "best_formula_label": str(best_row["formula_label"]),
        "best_aggregated_alpha": float(best_row["corrected_alpha"]),
        "best_aggregated_rel_error_vs_exact_goal": float(
            best_row["corrected_alpha_rel_error_vs_exact_goal"]
        ),
        "best_aggregated_reduction_factor": float(
            best_row["exact_goal_residual_reduction_factor"]
        ),
        "canonical_alpha": float(pack["canonical_row"]["corrected_alpha"]),
        "canonical_rel_error_vs_exact_goal": float(
            pack["canonical_rel_error_vs_exact_goal"]
        ),
        "canonical_advantage_factor": float(pack["canonical_advantage_factor"]),
        "best_row_margin_abs": float(pack["best_row_margin_abs"]),
        "best_row_ratio_vs_second": float(pack["best_row_ratio_vs_second"]),
        "selected_next_generation_route": "trial2_4d_full_mode_summation_gate",
        "recommended_next_route_or_none": ".5847-.5850",
        "selected_followup_route": "trial2_selector_4d_version_mixed_normalization_exactification",
        "selected_followup_route_or_none": ".5851-.5854",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5845",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
        },
        rows,
        summary,
        {
            "overall_status": "trial2_4d_full_mode_summation_directional_audited",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {
            "best_row": best_row,
            "second_row": pack["second_row"],
            "canonical_row": pack["canonical_row"],
            "aggregated_rows": pack["aggregated_rows"],
        },
    )
    outputs = write_artifact("declaration_gate", payload)
    print("[done] 8.7.56.5843-5846 Trial-2 4D full mode summation directional audit completed")
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
