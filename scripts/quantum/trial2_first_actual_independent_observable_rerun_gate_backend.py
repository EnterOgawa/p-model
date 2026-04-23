#!/usr/bin/env python3
"""Cut the first actual independent-observable rerun gate for Trial-2.

Purpose:
    After QED-vacuum formula materialization succeeds, Trial-2 can finally run
    one honest independent observable comparison. This backend fixes that first
    gate and reports the retained ranking on the selected surface.

Inputs:
    - scripts/quantum/trial2_qed_vacuum_absolute_alpha_formula_materialization_backend.py
    - scripts/quantum/trial2_weak_beta_decay_explicit_alpha_formula_materialization_backend.py

Outputs:
    - One in-memory gate pack consumed by `.5919-.5922` wrappers
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_qed_vacuum_absolute_alpha_formula_materialization_backend import (
    build_trial2_qed_vacuum_absolute_alpha_formula_pack,
)
from scripts.quantum.trial2_weak_beta_decay_explicit_alpha_formula_materialization_backend import (
    build_trial2_weak_beta_decay_explicit_alpha_formula_pack,
)


# 関数: `.5919-.5922` 用の first-rerun gate pack を返す。
def build_trial2_first_actual_independent_rerun_gate_pack() -> dict:
    """Return the retained first actual independent-observable rerun gate pack."""
    qed_pack = build_trial2_qed_vacuum_absolute_alpha_formula_pack()
    weak_pack = build_trial2_weak_beta_decay_explicit_alpha_formula_pack()

    first_surface = next(
        row
        for row in qed_pack["surfaces"]
        if str(row["surface_id"]) == "hydrogen_1s2s_gross_structure_baseline"
    )
    predictions = list(first_surface["predictions"])
    best_overall = min(predictions, key=lambda row: abs(float(row["relative_error_vs_observed"])))
    pmodel_predictions = [
        row
        for row in predictions
        if str(row["alpha_label"]).startswith("alpha_P_") or str(row["alpha_label"]) == "alpha_common"
    ]
    best_pmodel = min(
        pmodel_predictions,
        key=lambda row: abs(float(row["relative_error_vs_observed"])),
    )

    first_rerun_available = bool(qed_pack["trial2_qed_vacuum_primary_ready_now"])

    return {
        "qed_pack_summary": qed_pack["summary"],
        "weak_pack_summary": weak_pack["summary"],
        "selected_observable": first_surface,
        "prediction_table": predictions,
        "summary": {
            "first_actual_independent_observable_rerun_available_now": first_rerun_available,
            "selected_observable_id": str(first_surface["surface_id"]),
            "selected_observable_label": str(first_surface["label"]),
            "best_overall_alpha_label": str(best_overall["alpha_label"]),
            "best_overall_relative_error_vs_observed": float(
                best_overall["relative_error_vs_observed"]
            ),
            "best_pmodel_alpha_label": str(best_pmodel["alpha_label"]),
            "best_pmodel_relative_error_vs_observed": float(
                best_pmodel["relative_error_vs_observed"]
            ),
            "weak_explicit_formula_ready_now": bool(
                weak_pack["trial2_weak_beta_decay_primary_ready_now"]
            ),
        },
        "trial2_first_actual_independent_observable_rerun_available_now": first_rerun_available,
    }


# 関数: backend 単体実行時の compact summary を返す。

def main() -> None:
    """Run the first actual independent-observable rerun gate directly."""
    pack = build_trial2_first_actual_independent_rerun_gate_pack()
    summary = pack["summary"]
    print("[trial2_first_actual_independent_observable_rerun_gate_backend]")
    print(
        "  first_actual_independent_observable_rerun_available_now = "
        f"{summary['first_actual_independent_observable_rerun_available_now']}"
    )
    print(f"  selected_observable_id = {summary['selected_observable_id']}")
    print(f"  best_overall_alpha_label = {summary['best_overall_alpha_label']}")
    print(f"  best_pmodel_alpha_label = {summary['best_pmodel_alpha_label']}")


if __name__ == "__main__":
    main()
