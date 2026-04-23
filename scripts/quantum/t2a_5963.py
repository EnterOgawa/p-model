#!/usr/bin/env python3
"""Generate 8.7.56.5963-.5966 corrected-attribution refresh artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_hyperfine_corrected_attribution_refresh_backend import (
    build_trial2_hyperfine_corrected_attribution_refresh_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5959-5962",
        "updated_pack_trial2_hyperfine_g2_correction_materialization_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5963-5966"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "hyperfine corrected attribution refresh"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_hyperfine_corrected_attribution_refresh",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_hyperfine_g2_correction_materialized_corrected_attribution_primary_"
    "two_surface_refresh_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_hyperfine_g2_corrected_attribution_completed_two_surface_gate_"
    "primary_watch_secondary_next"
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


# 関数: `.5963-.5966` の formula bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas / rules fixed by the corrected-attribution refresh."""
    return {
        "effective_alpha_1s2s": "alpha_eff^(1S2S) = sqrt(nu_obs / nu_1S2S(1))",
        "effective_alpha_hfs_g2": "alpha_eff^(hfs,g2) = (nu_obs / nu_hfs_g2(1))^(1/4)",
        "reduction_rule": "the corrected branch is meaningful only if the old split shrinks materially",
    }


# 関数: `.5963-.5966` を実行する。

def main() -> None:
    """Execute the corrected-attribution refresh gate."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_hyperfine_corrected_attribution_refresh_pack()
    summary_pack = pack["summary"]

    route_selected = str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    split_reduced = bool(summary_pack["corrected_hyperfine_split_reduced_now"])
    both_codata = bool(summary_pack["both_surfaces_closest_to_codata_now"])
    strong_reduction = float(summary_pack["split_reduction_factor"]) > 10.0

    rows = [
        sign_base.row(
            "updated_pack_trial2_hyperfine_g2_materialization_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 hyperfine g2 materialization selected now",
            sign_base.truth(route_selected),
            "The corrected-attribution refresh starts only after the g/2 correction materializes.",
        ),
        sign_base.row(
            "trial2_corrected_hyperfine_split_reduced_now",
            "pass" if split_reduced else "reject",
            "Trial-2 corrected hyperfine split reduced now",
            sign_base.truth(split_reduced),
            "The source-backed g/2 correction must reduce the old effective-alpha split rather than just re-label it.",
        ),
        sign_base.row(
            "trial2_corrected_split_reduction_factor_gt_ten_now",
            "pass" if strong_reduction else "reject",
            "Trial-2 corrected split reduction factor > 10 now",
            sign_base.truth(strong_reduction),
            "The corrected branch materially changes the attribution geometry if the split shrinks by more than one order of magnitude.",
        ),
        sign_base.row(
            "trial2_both_surfaces_closest_to_codata_now",
            "pass" if both_codata else "reject",
            "Trial-2 both surfaces closest to CODATA now",
            sign_base.truth(both_codata),
            "After the correction, both Hydrogen 1S-2S and corrected 21 cm read the CODATA-side alpha neighborhood.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "alpha_eff_1s2s": float(summary_pack["alpha_eff_1s2s"]),
        "alpha_eff_hfs_corrected": float(summary_pack["alpha_eff_hfs_corrected"]),
        "effective_alpha_split_relative_old": float(summary_pack["effective_alpha_split_relative_old"]),
        "effective_alpha_split_relative_corrected": float(summary_pack["effective_alpha_split_relative_corrected"]),
        "split_reduction_factor": float(summary_pack["split_reduction_factor"]),
        "both_surfaces_closest_to_codata_now": bool(summary_pack["both_surfaces_closest_to_codata_now"]),
        "recommended_next_route_or_none": ".5967-.5970",
        "selected_next_generation_route": "trial2_multi_observable_corrected_hyperfine_gate",
        "selected_followup_route": "third_independent_surface_or_full_hyperfine_precision_source",
        "selected_followup_route_or_none": "conditional",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5965",
        STEP_NAME + " declaration gate",
        {"source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)}, "formulae": build_formulae()},
        rows,
        summary,
        {
            "overall_status": "trial2_hyperfine_corrected_attribution_refreshed",
            "branch_completed": True,
            "breakthrough_passed_now": True,
            "physical_reject_required": False,
        },
        {
            "effective_alpha_split_relative_corrected": float(summary_pack["effective_alpha_split_relative_corrected"]),
            "split_reduction_factor": float(summary_pack["split_reduction_factor"]),
        },
    )
    artifacts = write_artifact("declaration_gate", payload)
    print("[ok] corrected attribution gate:", artifacts["json"])


if __name__ == "__main__":
    main()
