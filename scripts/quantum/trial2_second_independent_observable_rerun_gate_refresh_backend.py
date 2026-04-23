#!/usr/bin/env python3
"""Refresh the second independent observable rerun gate with H I 21 cm.

Purpose:
    The prior second-surface gate closed negatively because only Hydrogen 1S-2S
    gross structure was alpha-explicit and rerun-ready. After materializing the
    H I 21 cm Fermi baseline, this backend recomputes the actual surface count
    and fixes whether the second independent surface now exists.

Inputs:
    - scripts/quantum/trial2_qed_vacuum_absolute_alpha_formula_materialization_backend.py
    - scripts/quantum/trial2_hydrogen_hyperfine_absolute_alpha_formula_materialization_backend.py

Outputs:
    - One in-memory gate pack consumed by `.5939-.5942` wrappers
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_hydrogen_hyperfine_absolute_alpha_formula_materialization_backend import (
    build_trial2_hydrogen_hyperfine_absolute_alpha_formula_pack,
)
from scripts.quantum.trial2_qed_vacuum_absolute_alpha_formula_materialization_backend import (
    build_trial2_qed_vacuum_absolute_alpha_formula_pack,
)


# 関数: `.5939-.5942` 用の refreshed second gate pack を返す。
def build_trial2_second_independent_rerun_gate_refresh_pack() -> dict:
    """Return the refreshed second independent observable rerun gate pack."""
    first_pack = build_trial2_qed_vacuum_absolute_alpha_formula_pack()
    hyperfine_pack = build_trial2_hydrogen_hyperfine_absolute_alpha_formula_pack()

    first_surface = next(
        row
        for row in first_pack["surfaces"]
        if str(row["surface_id"]) == "hydrogen_1s2s_gross_structure_baseline"
    )
    second_surface = dict(hyperfine_pack["surface"])
    surfaces = [first_surface, second_surface]

    actual_surfaces = [
        row
        for row in surfaces
        if bool(row["current_alpha_rerun_ready_now"]) and bool(row["independent_observable_now"])
    ]
    second_available = len(actual_surfaces) >= 2

    return {
        "surface_table": surfaces,
        "summary": {
            "second_independent_observable_rerun_available_now": bool(second_available),
            "current_actual_surface_count_now": int(len(actual_surfaces)),
            "current_actual_surface_ids": [str(row["surface_id"]) for row in actual_surfaces],
            "selected_first_surface_id": str(first_surface["surface_id"]),
            "selected_second_surface_id": str(second_surface["surface_id"]),
            "first_surface_best_alpha_label": str(first_surface["predictions"][0]["alpha_label"]),
            "first_surface_best_relative_error_vs_observed": float(
                first_surface["predictions"][0]["relative_error_vs_observed"]
            ),
            "second_surface_best_alpha_label": str(second_surface["predictions"][0]["alpha_label"]),
            "second_surface_best_relative_error_vs_observed": float(
                second_surface["predictions"][0]["relative_error_vs_observed"]
            ),
        },
        "trial2_second_independent_observable_rerun_available_now": bool(second_available),
    }


# 関数: backend 単体実行時の compact summary を返す。

def main() -> None:
    """Run the refreshed second independent observable rerun gate directly."""
    pack = build_trial2_second_independent_rerun_gate_refresh_pack()
    summary = pack["summary"]
    print("[trial2_second_independent_observable_rerun_gate_refresh_backend]")
    print(
        "  second_independent_observable_rerun_available_now = "
        f"{summary['second_independent_observable_rerun_available_now']}"
    )
    print(f"  current_actual_surface_count_now = {summary['current_actual_surface_count_now']}")
    print(f"  current_actual_surface_ids = {summary['current_actual_surface_ids']}")


if __name__ == "__main__":
    main()
