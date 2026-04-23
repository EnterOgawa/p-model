#!/usr/bin/env python3
"""Generate 8.7.56.5907-.5910 first independent observable rerun gate artifacts."""

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
from scripts.quantum.trial2_weak_sector_alpha_dependency_materialization_backend import (
    build_trial2_weak_sector_alpha_materialization_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE_QED = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5899-5902",
        "updated_pack_trial2_qed_vacuum_alpha_observable_map_materialization_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_GATE_WEAK = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5903-5906",
        "updated_pack_trial2_weak_sector_alpha_dependency_materialization_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5907-5910"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "first independent observable rerun gate"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_first_independent_observable_rerun_gate",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_qed_vacuum_positive_partial_weak_sector_materialization_completed_"
    "first_rerun_gate_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_first_independent_observable_rerun_unavailable_"
    "qed_vacuum_formula_materialization_primary_weak_sector_formula_secondary_next"
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


# 関数: `.5907-.5910` の rule bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas / rules fixed by the first rerun gate."""
    return {
        "rerun_gate_rule": (
            "promote the first independent observable rerun only when one surface is "
            "simultaneously independent, alpha explicit, and actually rerun ready"
        ),
        "qed_next_rule": (
            "QED-vacuum now needs absolute alpha-to-observable formula materialization "
            "rather than further inventory replay"
        ),
        "weak_next_rule": (
            "weak-sector remains secondary until one explicit alpha-dependent observable "
            "formula exists in the public pack"
        ),
    }


# 関数: `.5907-.5910` を実行する。

def main() -> None:
    """Execute the first independent observable rerun gate."""
    sign_base.require(PRIOR_GATE_QED)
    sign_base.require(PRIOR_GATE_WEAK)

    prior_summary = sign_base.read_json(PRIOR_GATE_WEAK)["summary"]
    qed_pack = build_trial2_qed_vacuum_alpha_materialization_pack()
    weak_pack = build_trial2_weak_sector_alpha_materialization_pack()

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    first_rerun_unavailable = not bool(
        qed_pack["trial2_qed_vacuum_primary_ready_now"]
        or weak_pack["trial2_weak_sector_primary_ready_now"]
    )
    qed_formula_materialization_primary = bool(qed_pack["trial2_qed_vacuum_positive_partial_now"])
    weak_formula_materialization_secondary = bool(
        weak_pack["trial2_weak_sector_positive_partial_now"]
    )

    rows = [
        sign_base.row(
            "updated_pack_trial2_qed_weak_materialization_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 QED + weak materialization selected now",
            sign_base.truth(route_selected),
            "The first rerun gate starts only after both primary and secondary materialization audits are fixed.",
        ),
        sign_base.row(
            "trial2_first_independent_observable_rerun_unavailable_now",
            "pass" if first_rerun_unavailable else "reject",
            "Trial-2 first independent observable rerun unavailable now",
            sign_base.truth(first_rerun_unavailable),
            "No current independent observable is yet both alpha explicit and rerun ready, so actual comparison must wait for formula materialization.",
        ),
        sign_base.row(
            "trial2_qed_vacuum_formula_materialization_primary_now",
            "pass" if qed_formula_materialization_primary else "reject",
            "Trial-2 QED-vacuum formula materialization primary now",
            sign_base.truth(qed_formula_materialization_primary),
            "QED-vacuum remains the first honest place to build an absolute alpha observable map.",
        ),
        sign_base.row(
            "trial2_weak_sector_formula_materialization_secondary_now",
            "pass" if weak_formula_materialization_secondary else "reject",
            "Trial-2 weak-sector formula materialization secondary now",
            sign_base.truth(weak_formula_materialization_secondary),
            "Weak beta-decay Route A/B remains the secondary formula-materialization target after QED-vacuum.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "first_independent_observable_rerun_available_now": not first_rerun_unavailable,
        "selected_next_generation_route": "trial2_qed_vacuum_absolute_alpha_formula_materialization_audit",
        "recommended_next_route_or_none": ".5911-.5914",
        "selected_followup_route": "trial2_weak_beta_decay_explicit_alpha_formula_materialization_audit",
        "selected_followup_route_or_none": ".5915-.5918",
        "selected_third_route": "trial2_first_actual_independent_observable_rerun_gate",
        "selected_third_route_or_none": ".5919-.5922",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5909",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "prior_gate_qed": sign_base.display_path(PRIOR_GATE_QED),
                "prior_gate_weak": sign_base.display_path(PRIOR_GATE_WEAK),
            },
            "formulae": build_formulae(),
            "selected_current_targets": {
                "qed_primary": "hydrogen_lamb_pack",
                "weak_secondary": "weak_beta_decay_route_ab",
            },
        },
        rows,
        summary,
        {
            "overall_status": "trial2_first_independent_observable_rerun_unavailable",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {
            "qed_primary_ready_now": bool(qed_pack["trial2_qed_vacuum_primary_ready_now"]),
            "weak_primary_ready_now": bool(weak_pack["trial2_weak_sector_primary_ready_now"]),
        },
    )
    artifacts = write_artifact("declaration_gate", payload)
    print("[ok] first rerun gate:", artifacts["json"])


if __name__ == "__main__":
    main()
