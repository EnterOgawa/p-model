#!/usr/bin/env python3
"""Refresh the first multi-observable comparison after H I 21 cm materializes.

Purpose:
    Once two independent alpha-explicit rerun surfaces exist, Trial-2 can cut
    the first actual multi-observable comparison table. This backend bundles
    Hydrogen 1S-2S gross structure and H I 21 cm Fermi baseline into one
    comparison object and fixes the current split verdict.

Inputs:
    - scripts/quantum/trial2_second_independent_observable_rerun_gate_refresh_backend.py

Outputs:
    - One in-memory pack consumed by `.5943-.5946` wrappers
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_second_independent_observable_rerun_gate_refresh_backend import (
    build_trial2_second_independent_rerun_gate_refresh_pack,
)


# 関数: `.5943-.5946` 用の multi-observable refresh pack を返す。
def build_trial2_first_multi_observable_comparison_refresh_pack() -> dict:
    """Return the refreshed first multi-observable comparison pack."""
    second_pack = build_trial2_second_independent_rerun_gate_refresh_pack()
    surfaces = list(second_pack["surface_table"])
    multi_available = bool(second_pack["summary"]["second_independent_observable_rerun_available_now"])

    pmodel_labels = {"alpha_P_frozen", "alpha_common", "alpha_P_4D_can", "alpha_P_4D_vertex"}
    per_surface_rows = []
    pmodel_win_count = 0
    codata_win_count = 0

    for surface in surfaces:
        predictions = list(surface["predictions"])
        best_overall = min(predictions, key=lambda row: abs(float(row["relative_error_vs_observed"])))
        pmodel_rows = [row for row in predictions if str(row["alpha_label"]) in pmodel_labels]
        best_pmodel = min(pmodel_rows, key=lambda row: abs(float(row["relative_error_vs_observed"])))
        pmodel_wins = abs(float(best_pmodel["relative_error_vs_observed"])) <= abs(
            float(best_overall["relative_error_vs_observed"])
        ) and str(best_overall["alpha_label"]) in pmodel_labels
        codata_wins = str(best_overall["alpha_label"]) == "alpha_CODATA"

        if pmodel_wins:
            pmodel_win_count += 1

        if codata_wins:
            codata_win_count += 1

        per_surface_rows.append(
            {
                "surface_id": str(surface["surface_id"]),
                "surface_label": str(surface["label"]),
                "best_overall_alpha_label": str(best_overall["alpha_label"]),
                "best_overall_relative_error_vs_observed": float(
                    best_overall["relative_error_vs_observed"]
                ),
                "best_pmodel_alpha_label": str(best_pmodel["alpha_label"]),
                "best_pmodel_relative_error_vs_observed": float(
                    best_pmodel["relative_error_vs_observed"]
                ),
                "pmodel_wins_now": bool(pmodel_wins),
                "codata_wins_now": bool(codata_wins),
            }
        )

    split_watch = multi_available and pmodel_win_count > 0 and codata_win_count > 0

    return {
        "surface_rows": per_surface_rows,
        "summary": {
            "first_multi_observable_comparison_available_now": bool(multi_available),
            "current_actual_surface_count_now": int(second_pack["summary"]["current_actual_surface_count_now"]),
            "surface_ids_now": list(second_pack["summary"]["current_actual_surface_ids"]),
            "pmodel_win_count_now": int(pmodel_win_count),
            "codata_win_count_now": int(codata_win_count),
            "split_watch_verdict_now": bool(split_watch),
        },
        "trial2_first_multi_observable_comparison_available_now": bool(multi_available),
    }


# 関数: backend 単体実行時の compact summary を返す。

def main() -> None:
    """Run the refreshed first multi-observable comparison backend directly."""
    pack = build_trial2_first_multi_observable_comparison_refresh_pack()
    summary = pack["summary"]
    print("[trial2_first_multi_observable_comparison_refresh_backend]")
    print(
        "  first_multi_observable_comparison_available_now = "
        f"{summary['first_multi_observable_comparison_available_now']}"
    )
    print(f"  pmodel_win_count_now = {summary['pmodel_win_count_now']}")
    print(f"  codata_win_count_now = {summary['codata_win_count_now']}")
    print(f"  split_watch_verdict_now = {summary['split_watch_verdict_now']}")


if __name__ == "__main__":
    main()
