#!/usr/bin/env python3
"""Generate 8.7.56.5899-.5902 QED-vacuum materialization artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_qed_vacuum_alpha_observable_map_materialization_backend import (
    build_trial2_qed_vacuum_alpha_materialization_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5895-5898",
        "updated_pack_trial2_primary_observable_rerun_gate",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5899-5902"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "QED-vacuum alpha observable-map materialization audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_qed_vacuum_alpha_observable_map_materialization_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_primary_observable_rerun_unavailable_"
    "qed_vacuum_materialization_primary_weak_sector_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_qed_vacuum_alpha_observable_map_positive_partial_"
    "weak_sector_gate_next"
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


# 関数: `.5899-.5902` の rule bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas / rules fixed by QED-vacuum materialization."""
    return {
        "casimir_rule": (
            "the current ideal-conductor Casimir baseline uses hbar and c only; "
            "it is alpha inactive under the present implementation"
        ),
        "hydrogen_lamb_rule": (
            "Lamb / H 1S-2S are structurally alpha sensitive, but the current pack "
            "still lacks an explicit alpha-to-observable prediction map"
        ),
        "qed_materialization_rule": (
            "QED-vacuum is positive partial only when it yields an honest "
            "independent target family, even if current rerun readiness remains absent"
        ),
    }


# 関数: `.5899-.5902` を実行する。

def main() -> None:
    """Execute the QED-vacuum alpha observable-map materialization audit."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_qed_vacuum_alpha_materialization_pack()
    summary_pack = pack["summary"]

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    materialization_complete = bool(pack["trial2_qed_vacuum_materialization_complete_now"])
    positive_partial = bool(pack["trial2_qed_vacuum_positive_partial_now"])
    structurally_sensitive_available = int(
        summary_pack["qed_structurally_alpha_sensitive_count"]
    ) > 0
    primary_target_selected = int(summary_pack["qed_selected_primary_target_count"]) > 0
    primary_ready_unavailable = not bool(pack["trial2_qed_vacuum_primary_ready_now"])
    excluded_explicit_retained = int(summary_pack["qed_rerun_ready_count"]) > 0

    rows = [
        sign_base.row(
            "updated_pack_trial2_primary_observable_rerun_gate_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 primary observable rerun gate selected now",
            sign_base.truth(route_selected),
            "The QED-vacuum audit starts only after the inventory/filter/rerun gate agrees that no independent rerun-ready surface exists yet.",
        ),
        sign_base.row(
            "trial2_qed_vacuum_materialization_complete_now",
            "pass" if materialization_complete else "reject",
            "Trial-2 QED-vacuum materialization complete now",
            sign_base.truth(materialization_complete),
            "The retained Casimir / Lamb / H 1S-2S subsurfaces are now classified at the alpha-observable-map level.",
        ),
        sign_base.row(
            "trial2_qed_vacuum_positive_partial_now",
            "pass" if positive_partial else "reject",
            "Trial-2 QED-vacuum positive partial now",
            sign_base.truth(positive_partial),
            "The current pack does contain an honest independent high-sensitivity target family inside QED-vacuum, even though rerun readiness is still missing.",
        ),
        sign_base.row(
            "trial2_qed_vacuum_structurally_alpha_sensitive_subsurface_available_now",
            "pass" if structurally_sensitive_available else "reject",
            "Trial-2 QED-vacuum structurally alpha-sensitive subsurface available now",
            sign_base.truth(structurally_sensitive_available),
            "Lamb-shift and H 1S-2S surfaces remain the cleanest current path to an explicit alpha observable map.",
        ),
        sign_base.row(
            "trial2_qed_vacuum_primary_target_selected_now",
            "pass" if primary_target_selected else "reject",
            "Trial-2 QED-vacuum primary target selected now",
            sign_base.truth(primary_target_selected),
            "Hydrogen / Lamb sub-surfaces are retained as the first honest materialization target inside the QED-vacuum pack.",
        ),
        sign_base.row(
            "trial2_qed_vacuum_primary_ready_still_unavailable_now",
            "pass" if primary_ready_unavailable else "reject",
            "Trial-2 QED-vacuum primary-ready still unavailable now",
            sign_base.truth(primary_ready_unavailable),
            "The current public QED-vacuum script still lacks an explicit alpha-dependent prediction formula, so blind observable rerun would still be premature.",
        ),
        sign_base.row(
            "trial2_qed_vacuum_excluded_explicit_alpha_crosscheck_retained_now",
            "pass" if excluded_explicit_retained else "reject",
            "Trial-2 QED-vacuum excluded explicit alpha cross-check retained now",
            sign_base.truth(excluded_explicit_retained),
            "The recoil-vs-g-2 surface remains available as an excluded explicit-alpha diagnostic inside the same pack.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "qed_subsurface_count": int(summary_pack["qed_subsurface_count"]),
        "qed_structurally_alpha_sensitive_count": int(
            summary_pack["qed_structurally_alpha_sensitive_count"]
        ),
        "qed_selected_primary_target_count": int(
            summary_pack["qed_selected_primary_target_count"]
        ),
        "qed_primary_ready_count": int(summary_pack["qed_primary_ready_count"]),
        "selected_qed_primary_target_ids": list(summary_pack["selected_qed_primary_target_ids"]),
        "selected_next_generation_route": "trial2_weak_sector_alpha_dependency_materialization_audit",
        "recommended_next_route_or_none": ".5903-.5906",
        "selected_followup_route": "trial2_first_independent_observable_rerun_gate",
        "selected_followup_route_or_none": ".5907-.5910",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5901",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
            "selected_qed_primary_target_ids": list(summary_pack["selected_qed_primary_target_ids"]),
        },
        rows,
        summary,
        {
            "overall_status": "trial2_qed_vacuum_alpha_observable_map_positive_partial",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {
            "qed_subsurface_count": int(summary_pack["qed_subsurface_count"]),
            "qed_structurally_alpha_sensitive_count": int(
                summary_pack["qed_structurally_alpha_sensitive_count"]
            ),
            "qed_primary_ready_count": int(summary_pack["qed_primary_ready_count"]),
        },
    )
    artifacts = write_artifact("declaration_gate", payload)
    print("[ok] qed vacuum materialization gate:", artifacts["json"])


if __name__ == "__main__":
    main()
