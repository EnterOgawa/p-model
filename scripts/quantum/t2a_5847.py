#!/usr/bin/env python3
"""Generate 8.7.56.5847-.5850 Trial-2 4D full mode summation gate artifacts."""

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
        "8.7.56.5843-5846",
        "updated_pack_trial2_4d_full_mode_summation_directional_check_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5847-5850"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "4D full mode summation gate"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_4d_full_mode_summation_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_4d_full_mode_summation_directional_audited_gate_primary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_4d_full_mode_summation_directional_negative_closeout_completed_"
    "selector_4d_primary_full_integral_secondary_next"
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
    """Return formulas fixed by the full-mode gate."""
    return {
        "gate_reading": (
            "If the best full-mode aggregate does not beat the canonical 4D row, "
            "full mode summation closes negatively as a main exact-goal route."
        ),
        "promoted_next_order": (
            "selector 4D version / mixed-normalization exactification -> "
            "4D full integral / external-probe current-vertex exactification"
        ),
    }


# 関数: `.5847-.5850` を実行する。

def main() -> None:
    """Execute the 4D full mode summation gate."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_4d_full_mode_summation_directional_check_pack()
    best_row = pack["best_row"]

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    directional_negative_closeout = bool(
        pack["exact_trial2_4d_full_mode_directional_negative_closeout_now"]
    )
    selector_4d_primary = bool(
        pack["exact_trial2_4d_selector_mixed_normalization_route_supported_now"]
    )
    full_integral_secondary = bool(directional_negative_closeout)

    rows = [
        sign_base.row(
            "updated_pack_trial2_4d_full_mode_summation_directional_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 4D full mode summation directional selected now",
            sign_base.truth(route_selected),
            "The gate starts only after the directional audit has already fixed the best full-mode aggregate.",
        ),
        sign_base.row(
            "exact_trial2_4d_full_mode_directional_negative_closeout_now",
            "pass" if directional_negative_closeout else "reject",
            "exact Trial-2 4D full mode directional negative closeout now",
            sign_base.truth(directional_negative_closeout),
            "The best full-mode aggregate is not a better exact-goal extractor than the canonical single-row 4D correction.",
        ),
        sign_base.row(
            "updated_pack_trial2_selector_4d_version_primary_now",
            "pass" if selector_4d_primary else "reject",
            "updated-pack Trial-2 selector 4D version primary now",
            sign_base.truth(selector_4d_primary),
            "The honest next mainline is selector 4D version / mixed-normalization exactification.",
        ),
        sign_base.row(
            "updated_pack_trial2_4d_full_integral_secondary_now",
            "pass" if full_integral_secondary else "reject",
            "updated-pack Trial-2 4D full integral secondary now",
            sign_base.truth(full_integral_secondary),
            "The strongest direct-computation fallback after selector 4D version is the external-probe current-vertex / full-integral route.",
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
        "canonical_alpha": float(pack["canonical_row"]["corrected_alpha"]),
        "canonical_rel_error_vs_exact_goal": float(
            pack["canonical_rel_error_vs_exact_goal"]
        ),
        "canonical_advantage_factor": float(pack["canonical_advantage_factor"]),
        "selected_next_generation_route": (
            "trial2_selector_4d_version_mixed_normalization_exactification_audit"
        ),
        "recommended_next_route_or_none": ".5851-.5854",
        "selected_followup_route": (
            "trial2_4d_full_integral_external_probe_current_vertex_exactification_audit"
        ),
        "selected_followup_route_or_none": ".5855-.5858",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5849",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
        },
        rows,
        summary,
        {
            "overall_status": "trial2_4d_full_mode_summation_negative_closeout",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {
            "best_row": best_row,
            "second_row": pack["second_row"],
            "canonical_row": pack["canonical_row"],
        },
    )
    outputs = write_artifact("declaration_gate", payload)
    print("[done] 8.7.56.5847-5850 Trial-2 4D full mode summation gate completed")
    print(f"[done] declaration: {outputs['json']}")


if __name__ == "__main__":
    main()
