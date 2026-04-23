#!/usr/bin/env python3
"""Generate 8.7.56.5915-.5918 weak beta-decay explicit-formula artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.quantum.trial2_weak_beta_decay_explicit_alpha_formula_materialization_backend import (
    build_trial2_weak_beta_decay_explicit_alpha_formula_pack,
)
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.5911-5914",
        "updated_pack_trial2_qed_vacuum_absolute_alpha_formula_materialization_audit",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.5915-5918"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor updated-pack Trial-2 "
    "weak beta-decay explicit alpha formula materialization audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "updated_pack_trial2_weak_beta_decay_explicit_alpha_formula_materialization_audit",
    prefix="q",
)
PRIOR_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_qed_vacuum_absolute_alpha_formula_materialized_"
    "hydrogen_1s2s_primary_weak_formula_secondary_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_residual_origin_missing_action_updated_pack_"
    "trial2_weak_beta_decay_explicit_alpha_formula_negative_closeout_"
    "completed_first_actual_rerun_gate_next"
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


# 関数: `.5915-.5918` の rule bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas / rules fixed by weak explicit-alpha audit."""
    return {
        "route_ab_rule": (
            "Route A/B keeps Q-value surrogates and closure gates, but fine-structure "
            "alpha is not a public deterministic input in the current pack"
        ),
        "route_b_symbol_rule": (
            "the symbol alpha inside Route-B standalone is an internal blend weight, "
            "not the fine-structure constant"
        ),
        "weak_formula_rule": (
            "without one explicit fine-structure-alpha observable map, the weak route "
            "cannot be promoted to an actual rerun surface"
        ),
    }


# 関数: `.5915-.5918` を実行する。

def main() -> None:
    """Execute the weak beta-decay explicit alpha formula materialization audit."""
    sign_base.require(PRIOR_GATE)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    pack = build_trial2_weak_beta_decay_explicit_alpha_formula_pack()
    summary_pack = pack["summary"]

    route_selected = (
        str(prior_summary["trial2_numeric_alpha_problem_classification"]) == PRIOR_CLASS
    )
    formula_unavailable = not bool(pack["trial2_weak_beta_decay_explicit_formula_materialized_now"])
    primary_ready_unavailable = not bool(pack["trial2_weak_beta_decay_primary_ready_now"])
    negative_closeout = bool(pack["trial2_weak_beta_decay_negative_closeout_now"])
    symbol_collision = bool(summary_pack["route_b_internal_alpha_symbol_collision_now"])

    rows = [
        sign_base.row(
            "updated_pack_trial2_qed_absolute_formula_selected_now",
            "pass" if route_selected else "reject",
            "updated-pack Trial-2 QED absolute-formula selected now",
            sign_base.truth(route_selected),
            "Weak explicit-alpha audit starts only after the first QED absolute formula is materialized.",
        ),
        sign_base.row(
            "trial2_weak_beta_decay_explicit_formula_unavailable_now",
            "pass" if formula_unavailable else "reject",
            "Trial-2 weak beta-decay explicit formula unavailable now",
            sign_base.truth(formula_unavailable),
            "Current weak beta-decay public artifacts do not expose fine-structure alpha as a deterministic observable input.",
        ),
        sign_base.row(
            "trial2_weak_beta_decay_primary_ready_unavailable_now",
            "pass" if primary_ready_unavailable else "reject",
            "Trial-2 weak beta-decay primary ready unavailable now",
            sign_base.truth(primary_ready_unavailable),
            "Weak beta-decay cannot yet enter the first actual rerun table as an alpha-explicit surface.",
        ),
        sign_base.row(
            "trial2_weak_route_b_internal_alpha_symbol_collision_now",
            "pass" if symbol_collision else "reject",
            "Trial-2 weak Route-B internal alpha symbol collision now",
            sign_base.truth(symbol_collision),
            "The current Route-B standalone script uses `alpha` as an internal blend weight, not as fine-structure alpha.",
        ),
        sign_base.row(
            "trial2_weak_beta_decay_negative_closeout_now",
            "pass" if negative_closeout else "reject",
            "Trial-2 weak beta-decay negative closeout now",
            sign_base.truth(negative_closeout),
            "The weak formula branch closes negatively inside the current pack and drops to reserve until a genuine fine-structure-alpha surface appears.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "weak_explicit_formula_ready_count": int(summary_pack["weak_explicit_formula_ready_count"]),
        "selected_secondary_target_ids": list(summary_pack["selected_secondary_target_ids"]),
        "selected_next_generation_route": "trial2_first_actual_independent_observable_rerun_gate",
        "recommended_next_route_or_none": ".5919-.5922",
        "physical_reject_required": False,
    }

    payload = sign_base.payload(
        "8.7.56.5917",
        STEP_NAME + " declaration gate",
        {
            "source_files": {"prior_gate": sign_base.display_path(PRIOR_GATE)},
            "formulae": build_formulae(),
            "weak_surfaces": pack["surfaces"],
        },
        rows,
        summary,
        {
            "overall_status": "trial2_weak_beta_decay_explicit_alpha_formula_negative_closeout",
            "branch_completed": True,
            "breakthrough_passed_now": False,
            "physical_reject_required": False,
        },
        {
            "route_ab_transition": str(summary_pack["route_ab_transition"]),
            "route_b_internal_alpha_symbol_collision_now": bool(
                summary_pack["route_b_internal_alpha_symbol_collision_now"]
            ),
        },
    )
    artifacts = write_artifact("declaration_gate", payload)
    print("[ok] weak explicit-formula gate:", artifacts["json"])


if __name__ == "__main__":
    main()
