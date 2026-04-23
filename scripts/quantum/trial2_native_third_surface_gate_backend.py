#!/usr/bin/env python3
"""Cut the native third-surface gate under the absolute P-model rule.

Purpose:
    Once Trial-2 primary comparison is restricted to "P-model formula x
    P-model alpha", the retained Hydrogen table must be rebuilt around native
    surfaces only. The next honest question is whether one native relativistic
    third surface is actually available or whether the pack still remains on
    the two-surface split-watch table.

Inputs:
    - scripts/quantum/trial2_qed_vacuum_absolute_alpha_formula_materialization_backend.py
    - scripts/quantum/trial2_hydrogen_hyperfine_absolute_alpha_formula_materialization_backend.py
    - scripts/quantum/trial2_native_relativistic_halpha_surface_materialization_backend.py

Outputs:
    - One in-memory gate pack consumed by `.6019-.6022` wrappers
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
from scripts.quantum.trial2_native_relativistic_halpha_surface_materialization_backend import (
    build_trial2_native_relativistic_halpha_surface_materialization_pack,
)
from scripts.quantum.trial2_qed_vacuum_absolute_alpha_formula_materialization_backend import (
    build_trial2_qed_vacuum_absolute_alpha_formula_pack,
)


PMODEL_LABELS = {"alpha_P_frozen", "alpha_common", "alpha_P_4D_can", "alpha_P_4D_vertex"}


# 関数: 1 surface row を compact score row に圧縮する。
def build_surface_row(surface: dict) -> dict:
    """Return one compact score row for one native actual surface."""
    predictions = list(surface["predictions"])
    best_overall = min(predictions, key=lambda row: abs(float(row["relative_error_vs_observed"])))
    pmodel_rows = [row for row in predictions if str(row["alpha_label"]) in PMODEL_LABELS]
    best_pmodel = min(pmodel_rows, key=lambda row: abs(float(row["relative_error_vs_observed"])))
    family_id = str(surface.get("family_id") or f"{surface['surface_id']}_family")
    return {
        "surface_id": str(surface["surface_id"]),
        "surface_label": str(surface["label"]),
        "family_id": family_id,
        "best_overall_alpha_label": str(best_overall["alpha_label"]),
        "best_overall_relative_error_vs_observed": float(best_overall["relative_error_vs_observed"]),
        "best_pmodel_alpha_label": str(best_pmodel["alpha_label"]),
        "best_pmodel_relative_error_vs_observed": float(best_pmodel["relative_error_vs_observed"]),
        "pmodel_wins_now": str(best_overall["alpha_label"]) in PMODEL_LABELS,
        "codata_wins_now": str(best_overall["alpha_label"]) == "alpha_CODATA",
    }


# 関数: `.6019-.6022` 用の native third-surface gate pack を返す。

def build_trial2_native_third_surface_gate_pack() -> dict:
    """Return the native third-surface gate pack."""
    gross_pack = build_trial2_qed_vacuum_absolute_alpha_formula_pack()
    hyperfine_pack = build_trial2_hydrogen_hyperfine_absolute_alpha_formula_pack()
    halpha_pack = build_trial2_native_relativistic_halpha_surface_materialization_pack()

    gross_surface = next(
        surface
        for surface in gross_pack["surfaces"]
        if str(surface["surface_id"]) == "hydrogen_1s2s_gross_structure_baseline"
    )
    native_surface_rows = [
        build_surface_row(gross_surface),
        build_surface_row(hyperfine_pack["surface"]),
    ]
    native_pmodel_wins = int(sum(1 for row in native_surface_rows if bool(row["pmodel_wins_now"])))
    native_codata_wins = int(sum(1 for row in native_surface_rows if bool(row["codata_wins_now"])))

    return {
        "native_surface_rows": native_surface_rows,
        "diagnostic_retained_surface": dict(halpha_pack["surface"]),
        "summary": {
            "native_actual_surface_count_now": int(len(native_surface_rows)),
            "native_surface_ids_now": [str(row["surface_id"]) for row in native_surface_rows],
            "native_pmodel_win_count_now": native_pmodel_wins,
            "native_codata_win_count_now": native_codata_wins,
            "native_split_watch_retained_now": bool(
                native_pmodel_wins == 1 and native_codata_wins == 1
            ),
            "native_genuine_third_surface_available_now": bool(
                halpha_pack["summary"]["native_relativistic_surface_ready_now"]
            ),
            "diagnostic_retained_surface_id": str(halpha_pack["summary"]["selected_surface_id"]),
            "current_honest_reading": (
                "Current primary native table still contains only the 1S-2S "
                "gross-structure baseline and the tree-level 21 cm hyperfine "
                "baseline. The retained Halpha fine-structure surface stays "
                "diagnostic because the relativistic envelope bridge is still absent."
            ),
        },
        "trial2_native_third_surface_gate_completed_now": True,
    }


# 関数: backend 単体実行時の compact summary を返す。

def main() -> None:
    """Run the native third-surface gate backend directly."""
    pack = build_trial2_native_third_surface_gate_pack()
    summary = pack["summary"]
    print("[trial2_native_third_surface_gate_backend]")
    print(f"  native_actual_surface_count_now = {summary['native_actual_surface_count_now']}")
    print(
        "  native_genuine_third_surface_available_now = "
        f"{summary['native_genuine_third_surface_available_now']}"
    )
    print(
        "  native_split_watch_retained_now = "
        f"{summary['native_split_watch_retained_now']}"
    )


if __name__ == "__main__":
    main()
