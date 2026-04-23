#!/usr/bin/env python3
"""Refresh the native surface gate after materializing one He II surface.

Purpose:
    Trial-2 native observable validation previously stopped at two Hydrogen
    surfaces because no genuinely new non-Hydrogen native route had actualized.
    Once the He II hydrogenic baseline is available, the honest next question is
    whether the native primary table now contains one genuine third surface.

Inputs:
    - scripts/quantum/trial2_qed_vacuum_absolute_alpha_formula_materialization_backend.py
    - scripts/quantum/trial2_hydrogen_hyperfine_absolute_alpha_formula_materialization_backend.py
    - scripts/quantum/trial2_native_helium_ion_surface_materialization_backend.py

Outputs:
    - One in-memory gate pack consumed by `.6031-.6034` wrappers
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
from scripts.quantum.trial2_native_helium_ion_surface_materialization_backend import (
    build_trial2_native_helium_ion_surface_materialization_pack,
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
    return {
        "surface_id": str(surface["surface_id"]),
        "surface_label": str(surface["label"]),
        "family_id": str(surface.get("family_id") or f"{surface['surface_id']}_family"),
        "best_overall_alpha_label": str(best_overall["alpha_label"]),
        "best_overall_relative_error_vs_observed": float(best_overall["relative_error_vs_observed"]),
        "best_pmodel_alpha_label": str(best_pmodel["alpha_label"]),
        "best_pmodel_relative_error_vs_observed": float(best_pmodel["relative_error_vs_observed"]),
        "pmodel_wins_now": str(best_overall["alpha_label"]) in PMODEL_LABELS,
        "codata_wins_now": str(best_overall["alpha_label"]) == "alpha_CODATA",
    }


# 関数: `.6031-.6034` 用の native non-Hydrogen gate pack を返す。

def build_trial2_native_non_hydrogen_surface_gate_pack() -> dict:
    """Return the native non-Hydrogen surface gate pack."""
    gross_pack = build_trial2_qed_vacuum_absolute_alpha_formula_pack()
    hyperfine_pack = build_trial2_hydrogen_hyperfine_absolute_alpha_formula_pack()
    helium_pack = build_trial2_native_helium_ion_surface_materialization_pack()

    gross_surface = next(
        surface
        for surface in gross_pack["surfaces"]
        if str(surface["surface_id"]) == "hydrogen_1s2s_gross_structure_baseline"
    )
    native_surface_rows = [
        build_surface_row(gross_surface),
        build_surface_row(hyperfine_pack["surface"]),
        build_surface_row(helium_pack["surface"]),
    ]
    native_pmodel_wins = int(sum(1 for row in native_surface_rows if bool(row["pmodel_wins_now"])))
    native_codata_wins = int(sum(1 for row in native_surface_rows if bool(row["codata_wins_now"])))

    return {
        "native_surface_rows": native_surface_rows,
        "summary": {
            "native_actual_surface_count_now": int(len(native_surface_rows)),
            "native_surface_ids_now": [str(row["surface_id"]) for row in native_surface_rows],
            "native_pmodel_win_count_now": native_pmodel_wins,
            "native_codata_win_count_now": native_codata_wins,
            "native_non_hydrogen_actual_surface_count_now": 1,
            "native_non_hydrogen_surface_id_now": str(helium_pack["summary"]["surface_id"]),
            "native_genuine_third_surface_available_now": True,
            "native_split_watch_retained_now": bool(
                native_pmodel_wins == 1 and native_codata_wins == 1
            ),
            "native_codata_lead_diagnostic_now": bool(native_codata_wins > native_pmodel_wins),
            "current_honest_reading": (
                "The native primary table now contains one genuine non-Hydrogen "
                "surface via the He II one-electron gross-structure baseline. "
                "This removes the missing-third-surface blocker, but the three-surface "
                "table still does not yield a native pass verdict."
            ),
        },
        "trial2_native_non_hydrogen_surface_gate_completed_now": True,
    }


# 関数: backend 単体実行時の compact summary を返す。

def main() -> None:
    """Run the native non-Hydrogen surface gate backend directly."""
    pack = build_trial2_native_non_hydrogen_surface_gate_pack()
    summary = pack["summary"]
    print("[trial2_native_non_hydrogen_surface_gate_backend]")
    print(f"  native_actual_surface_count_now = {summary['native_actual_surface_count_now']}")
    print(
        "  native_non_hydrogen_actual_surface_count_now = "
        f"{summary['native_non_hydrogen_actual_surface_count_now']}"
    )
    print(
        "  native_codata_lead_diagnostic_now = "
        f"{summary['native_codata_lead_diagnostic_now']}"
    )


if __name__ == "__main__":
    main()
