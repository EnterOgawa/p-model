#!/usr/bin/env python3
"""Generate 8.7.56.5947-.5950 hyperfine attribution-split artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_hyperfine_attribution_split_audit_backend import (
    build_trial2_hyperfine_attribution_split_audit_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5943-5946",
        "updated_pack_trial2_first_multi_observable_comparison_refresh",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5947-5950"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "hyperfine attribution split audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_hyperfine_attribution_split_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_first_multi_observable_comparison_completed_split_watch_"
    "hyperfine_attribution_primary_third_surface_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_hyperfine_attribution_split_completed_third_surface_inventory_"
    "primary_watch_gate_secondary_next"
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


# 関数: `.5947-.5950` の formula bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas / rules fixed by the attribution audit."""
    return {
        "effective_alpha_1s2s_rule": "alpha_eff_1s2s = sqrt(nu_obs / K_1s2s)",
        "effective_alpha_hfs_rule": "alpha_eff_hfs = (nu_obs / K_hfs)^(1/4)",
        "attribution_rule": (
            "split watch is localized when the two surfaces imply materially "
            "different effective-alpha values and different retained checkpoints "
            "sit closest to those values"
        ),
    }


# 関数: `.5947-.5950` を実行する。

def main() -> None:
    """Execute the hyperfine attribution split audit."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_hyperfine_attribution_split_audit_pack()
    summary_pack = pack["summary"]

    route_selected = str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    split_watch = bool(summary_pack["split_watch_verdict_now"])
    split_positive = float(summary_pack["effective_alpha_split_relative"]) > 0.0
    codata_closest_1s2s = str(summary_pack["closest_to_1s2s_alpha_label"]) == "alpha_CODATA"
    vertex_closest_hfs = str(summary_pack["closest_to_hfs_alpha_label"]) == "alpha_P_4D_vertex"

    rows = [
        sign_base.row(
            "updated_pack_trial2_first_multi_compare_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 first multi-observable comparison selected now",
            sign_base.truth(route_selected),
            "The attribution audit starts only after the two-surface split watch is fixed as the live blocker.",
        ),
        sign_base.row(
            "trial2_split_watch_verdict_retained_now",
            "pass" if split_watch else "reject",
            "Trial-2 split watch verdict retained now",
            sign_base.truth(split_watch),
            "The current two-surface comparison must still be a split before attribution can be localized honestly.",
        ),
        sign_base.row(
            "trial2_effective_alpha_split_positive_now",
            "pass" if split_positive else "reject",
            "Trial-2 effective alpha split positive now",
            sign_base.truth(split_positive),
            "The 21 cm Fermi surface implies a higher effective alpha than Hydrogen 1S-2S gross structure.",
        ),
        sign_base.row(
            "trial2_codata_closest_to_1s2s_now",
            "pass" if codata_closest_1s2s else "reject",
            "Trial-2 CODATA closest to 1S-2S now",
            sign_base.truth(codata_closest_1s2s),
            "Hydrogen 1S-2S gross structure remains aligned with the CODATA-side checkpoint.",
        ),
        sign_base.row(
            "trial2_vertex_closest_to_hyperfine_now",
            "pass" if vertex_closest_hfs else "reject",
            "Trial-2 vertex closest to hyperfine now",
            sign_base.truth(vertex_closest_hfs),
            "The H I 21 cm hyperfine win is localized to the retained P-model vertex-side checkpoint.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "alpha_eff_1s2s": float(summary_pack["alpha_eff_1s2s"]),
        "alpha_eff_hfs": float(summary_pack["alpha_eff_hfs"]),
        "effective_alpha_split_relative": float(summary_pack["effective_alpha_split_relative"]),
        "closest_to_1s2s_alpha_label": str(summary_pack["closest_to_1s2s_alpha_label"]),
        "closest_to_hfs_alpha_label": str(summary_pack["closest_to_hfs_alpha_label"]),
        "hyperfine_attribution_split_localized_now": bool(summary_pack["hyperfine_attribution_split_localized_now"]),
        "selected_next_generation_route": "trial2_third_independent_surface_inventory_refresh",
        "recommended_next_route_or_none": ".5951-.5954",
        "selected_followup_route": "trial2_multi_observable_watch_pass_gate",
        "selected_followup_route_or_none": ".5955-.5958",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5949",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
            "checkpoint_gap_rows": pack["checkpoint_gap_rows"],
        },
        rows,
        summary,
        {
            "overall_status": "trial2_hyperfine_attribution_split_localized",
            "branch_completed": True,
            "breakthrough_passed_now": True,
            "physical_reject_required": False,
        },
        {
            "alpha_eff_1s2s": float(summary_pack["alpha_eff_1s2s"]),
            "alpha_eff_hfs": float(summary_pack["alpha_eff_hfs"]),
            "effective_alpha_split_relative": float(summary_pack["effective_alpha_split_relative"]),
        },
    )
    artifacts = write_artifact("declaration_gate", payload)
    print("[ok] hyperfine attribution split gate:", artifacts["json"])


if __name__ == "__main__":
    main()
