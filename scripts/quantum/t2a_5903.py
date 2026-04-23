#!/usr/bin/env python3
"""Generate 8.7.56.5903-.5906 weak-sector materialization artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_weak_sector_alpha_dependency_materialization_backend import (
    build_trial2_weak_sector_alpha_materialization_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5899-5902",
        "updated_pack_trial2_qed_vacuum_alpha_observable_map_materialization_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5903-5906"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "weak-sector alpha dependency materialization audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_weak_sector_alpha_dependency_materialization_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_qed_vacuum_alpha_observable_map_positive_partial_"
    "weak_sector_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_qed_vacuum_positive_partial_weak_sector_materialization_completed_"
    "first_rerun_gate_next"
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


# 関数: `.5903-.5906` の rule bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas / rules fixed by weak-sector materialization."""
    return {
        "weak_route_ab_rule": (
            "Route A/B is the only retained weak-sector surface with a plausible "
            "future alpha lever, but that lever is not public or explicit yet"
        ),
        "ckm_pmns_rule": (
            "CKM and PMNS currently act as alpha-inactive closure surfaces, not as "
            "observable rerun maps"
        ),
        "weak_materialization_rule": (
            "weak-sector stays secondary until one explicit alpha dependency is "
            "materialized in the public pack"
        ),
    }


# 関数: `.5903-.5906` を実行する。

def main() -> None:
    """Execute the weak-sector alpha dependency materialization audit."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_weak_sector_alpha_materialization_pack()
    summary_pack = pack["summary"]

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    materialization_complete = bool(pack["trial2_weak_sector_materialization_complete_now"])
    positive_partial = bool(pack["trial2_weak_sector_positive_partial_now"])
    secondary_target_selected = int(summary_pack["weak_selected_secondary_target_count"]) > 0
    primary_ready_unavailable = not bool(pack["trial2_weak_sector_primary_ready_now"])
    rerun_ready_unavailable = not bool(pack["trial2_weak_sector_rerun_ready_now"])

    rows = [
        sign_base.row(
            "updated_pack_trial2_qed_vacuum_materialization_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 QED-vacuum materialization selected now",
            sign_base.truth(route_selected),
            "The weak-sector audit starts only after the QED-vacuum branch fixes the primary materialization target family.",
        ),
        sign_base.row(
            "trial2_weak_sector_materialization_complete_now",
            "pass" if materialization_complete else "reject",
            "Trial-2 weak-sector materialization complete now",
            sign_base.truth(materialization_complete),
            "The retained weak-sector surfaces are now classified by whether they already expose an explicit alpha dependency in the public pack.",
        ),
        sign_base.row(
            "trial2_weak_sector_positive_partial_now",
            "pass" if positive_partial else "reject",
            "Trial-2 weak-sector positive partial now",
            sign_base.truth(positive_partial),
            "The weak sector still contributes independent comparison targets, even though it is not yet the first observable rerun surface.",
        ),
        sign_base.row(
            "trial2_weak_sector_secondary_target_selected_now",
            "pass" if secondary_target_selected else "reject",
            "Trial-2 weak-sector secondary target selected now",
            sign_base.truth(secondary_target_selected),
            "Weak beta-decay Route A/B remains the first secondary materialization candidate inside the weak sector.",
        ),
        sign_base.row(
            "trial2_weak_sector_primary_ready_still_unavailable_now",
            "pass" if primary_ready_unavailable else "reject",
            "Trial-2 weak-sector primary-ready still unavailable now",
            sign_base.truth(primary_ready_unavailable),
            "No weak-sector surface currently satisfies explicit alpha leverage plus rerun readiness in the public pack.",
        ),
        sign_base.row(
            "trial2_weak_sector_rerun_ready_unavailable_now",
            "pass" if rerun_ready_unavailable else "reject",
            "Trial-2 weak-sector rerun-ready unavailable now",
            sign_base.truth(rerun_ready_unavailable),
            "The weak sector is still a future formula-materialization task rather than an immediately runnable alpha comparison lane.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "weak_surface_count": int(summary_pack["weak_surface_count"]),
        "weak_selected_secondary_target_count": int(
            summary_pack["weak_selected_secondary_target_count"]
        ),
        "weak_primary_ready_count": int(summary_pack["weak_primary_ready_count"]),
        "weak_rerun_ready_count": int(summary_pack["weak_rerun_ready_count"]),
        "selected_secondary_target_ids": list(summary_pack["selected_secondary_target_ids"]),
        "selected_next_generation_route": "trial2_first_independent_observable_rerun_gate",
        "recommended_next_route_or_none": ".5907-.5910",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5905",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
            "selected_secondary_target_ids": list(summary_pack["selected_secondary_target_ids"]),
        },
        rows,
        summary,
        {
            "overall_status": "trial2_weak_sector_alpha_materialization_completed",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {
            "weak_surface_count": int(summary_pack["weak_surface_count"]),
            "weak_primary_ready_count": int(summary_pack["weak_primary_ready_count"]),
            "weak_rerun_ready_count": int(summary_pack["weak_rerun_ready_count"]),
        },
    )
    artifacts = write_artifact("declaration_gate", payload)
    print("[ok] weak sector materialization gate:", artifacts["json"])


if __name__ == "__main__":
    main()
