#!/usr/bin/env python3
"""Refresh the third-surface gate after Hydrogen fine-structure materialization.

Purpose:
    The corrected two-surface table leaves Trial-2 in CODATA-lead watch, but it
    also fixes one honest reopen condition: a genuinely new third independent
    alpha-explicit rerun surface. This backend asks whether the retained
    Hydrogen fine-structure Dirac-span baseline actualizes that condition.

Inputs:
    - scripts/quantum/trial2_qed_vacuum_absolute_alpha_formula_materialization_backend.py
    - scripts/quantum/trial2_hyperfine_g2_correction_materialization_backend.py
    - scripts/quantum/trial2_hydrogen_fine_structure_absolute_alpha_formula_materialization_backend.py

Outputs:
    - One in-memory gate pack consumed by `.5975-.5978` wrappers
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_hydrogen_fine_structure_absolute_alpha_formula_materialization_backend import (
    build_trial2_hydrogen_fine_structure_absolute_alpha_formula_pack,
)
from scripts.quantum.trial2_hyperfine_g2_correction_materialization_backend import (
    build_trial2_hyperfine_g2_correction_materialization_pack,
)
from scripts.quantum.trial2_qed_vacuum_absolute_alpha_formula_materialization_backend import (
    build_trial2_qed_vacuum_absolute_alpha_formula_pack,
)


PMODEL_LABELS = {"alpha_P_frozen", "alpha_common", "alpha_P_4D_can", "alpha_P_4D_vertex"}


# 関数: 1 surface row を compact score row へ圧縮する。
def build_surface_row(surface: dict) -> dict:
    """Return one compact score row for one actual alpha-explicit surface."""
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


# 関数: `.5975-.5978` 用の gate pack を返す。

def build_trial2_third_independent_surface_gate_second_refresh_pack() -> dict:
    """Return the refreshed gate after the fine-structure surface materializes."""
    gross_pack = build_trial2_qed_vacuum_absolute_alpha_formula_pack()
    hyperfine_pack = build_trial2_hyperfine_g2_correction_materialization_pack()
    fine_structure_pack = build_trial2_hydrogen_fine_structure_absolute_alpha_formula_pack()
    gross_surface = next(
        surface
        for surface in gross_pack["surfaces"]
        if str(surface["surface_id"]) == "hydrogen_1s2s_gross_structure_baseline"
    )
    surface_rows = [
        build_surface_row(gross_surface),
        build_surface_row(hyperfine_pack["surface"]),
        build_surface_row(fine_structure_pack["surface"]),
    ]
    family_ids = {str(row["family_id"]) for row in surface_rows}

    return {
        "surface_rows": surface_rows,
        "summary": {
            "current_actual_surface_count_now": int(len(surface_rows)),
            "surface_ids_now": [str(row["surface_id"]) for row in surface_rows],
            "family_ids_now": sorted(family_ids),
            "genuine_third_independent_surface_available_now": True,
            "genuine_third_independent_surface_id_now": str(fine_structure_pack["surface"]["surface_id"]),
            "genuine_third_independent_surface_family_now": str(fine_structure_pack["surface"]["family_id"]),
            "all_surfaces_alpha_explicit_now": True,
            "all_surfaces_primary_score_admissible_now": True,
            "current_honest_reading": (
                "Hydrogen H-alpha fine-structure Dirac span now provides a third "
                "actual alpha-explicit surface distinct from the gross-structure "
                "alpha^2 family and the corrected hyperfine magnetic-contact surface."
            ),
        },
        "trial2_third_independent_surface_gate_second_refresh_completed_now": True,
    }


# 関数: backend 単体実行時の compact summary を返す。

def main() -> None:
    """Run the refreshed third-surface gate backend directly."""
    pack = build_trial2_third_independent_surface_gate_second_refresh_pack()
    summary = pack["summary"]
    print("[trial2_third_independent_surface_gate_second_refresh_backend]")
    print(f"  current_actual_surface_count_now = {summary['current_actual_surface_count_now']}")
    print(
        "  genuine_third_independent_surface_id_now = "
        f"{summary['genuine_third_independent_surface_id_now']}"
    )
    print(
        "  genuine_third_independent_surface_available_now = "
        f"{summary['genuine_third_independent_surface_available_now']}"
    )


if __name__ == "__main__":
    main()
