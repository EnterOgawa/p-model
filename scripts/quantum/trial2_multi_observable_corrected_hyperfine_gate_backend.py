#!/usr/bin/env python3
"""Cut the corrected two-surface comparison after the `g/2` hyperfine update.

Purpose:
    Once the source-backed `g/2` correction is materialized and the corrected
    attribution picture is known, Trial-2 needs one honest gate asking whether
    the current two-surface table still splits or now leans one way.

Inputs:
    - scripts/quantum/trial2_qed_vacuum_absolute_alpha_formula_materialization_backend.py
    - scripts/quantum/trial2_hyperfine_g2_correction_materialization_backend.py
    - scripts/quantum/trial2_hyperfine_corrected_attribution_refresh_backend.py

Outputs:
    - One in-memory pack consumed by `.5967-.5970` wrappers
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_hyperfine_corrected_attribution_refresh_backend import (
    build_trial2_hyperfine_corrected_attribution_refresh_pack,
)
from scripts.quantum.trial2_hyperfine_g2_correction_materialization_backend import (
    build_trial2_hyperfine_g2_correction_materialization_pack,
)
from scripts.quantum.trial2_qed_vacuum_absolute_alpha_formula_materialization_backend import (
    build_trial2_qed_vacuum_absolute_alpha_formula_pack,
)


PMODEL_LABELS = {"alpha_P_frozen", "alpha_common", "alpha_P_4D_can", "alpha_P_4D_vertex"}


# 関数: 1 surface row を summary row へ圧縮する。
def build_surface_row(surface: dict) -> dict:
    """Return one compact score row for one alpha-explicit surface."""
    predictions = list(surface["predictions"])
    best_overall = min(predictions, key=lambda row: abs(float(row["relative_error_vs_observed"])))
    pmodel_rows = [row for row in predictions if str(row["alpha_label"]) in PMODEL_LABELS]
    best_pmodel = min(pmodel_rows, key=lambda row: abs(float(row["relative_error_vs_observed"])))
    pmodel_wins = abs(float(best_pmodel["relative_error_vs_observed"])) <= abs(
        float(best_overall["relative_error_vs_observed"])
    ) and str(best_overall["alpha_label"]) in PMODEL_LABELS
    codata_wins = str(best_overall["alpha_label"]) == "alpha_CODATA"
    return {
        "surface_id": str(surface["surface_id"]),
        "surface_label": str(surface["label"]),
        "best_overall_alpha_label": str(best_overall["alpha_label"]),
        "best_overall_relative_error_vs_observed": float(best_overall["relative_error_vs_observed"]),
        "best_pmodel_alpha_label": str(best_pmodel["alpha_label"]),
        "best_pmodel_relative_error_vs_observed": float(best_pmodel["relative_error_vs_observed"]),
        "pmodel_wins_now": bool(pmodel_wins),
        "codata_wins_now": bool(codata_wins),
    }


# 関数: `.5967-.5970` 用の corrected gate pack を返す。

def build_trial2_multi_observable_corrected_hyperfine_gate_pack() -> dict:
    """Return the corrected two-surface gate pack."""
    first_pack = build_trial2_qed_vacuum_absolute_alpha_formula_pack()
    corrected_pack = build_trial2_hyperfine_g2_correction_materialization_pack()
    attribution_pack = build_trial2_hyperfine_corrected_attribution_refresh_pack()
    first_surface = next(
        surface for surface in first_pack["surfaces"] if str(surface["surface_id"]) == "hydrogen_1s2s_gross_structure_baseline"
    )

    surface_rows = [
        build_surface_row(first_surface),
        build_surface_row(corrected_pack["surface"]),
    ]

    pmodel_win_count = sum(1 for row in surface_rows if bool(row["pmodel_wins_now"]))
    codata_win_count = sum(1 for row in surface_rows if bool(row["codata_wins_now"]))
    split_watch = pmodel_win_count > 0 and codata_win_count > 0
    codata_sweep = codata_win_count == len(surface_rows) and pmodel_win_count == 0

    return {
        "surface_rows": surface_rows,
        "summary": {
            "current_actual_surface_count_now": len(surface_rows),
            "surface_ids_now": [str(row["surface_id"]) for row in surface_rows],
            "pmodel_win_count_now": int(pmodel_win_count),
            "codata_win_count_now": int(codata_win_count),
            "split_watch_verdict_now": bool(split_watch),
            "codata_sweep_verdict_now": bool(codata_sweep),
            "multi_observable_pass_available_now": False,
            "multi_observable_watch_retained_now": True,
            "multi_observable_codata_lead_watch_retained_now": bool(codata_sweep),
            "both_surfaces_closest_to_codata_now": bool(
                attribution_pack["summary"]["both_surfaces_closest_to_codata_now"]
            ),
            "current_honest_reading": (
                "the corrected two-surface table now leans to CODATA on both "
                "actual surfaces, but a third genuinely independent surface is "
                "still missing so the verdict remains watch rather than final reject"
            ),
        },
        "trial2_multi_observable_corrected_hyperfine_gate_completed_now": True,
    }


# 関数: backend 単体実行時の compact summary を返す。

def main() -> None:
    """Run the corrected two-surface gate backend directly."""
    pack = build_trial2_multi_observable_corrected_hyperfine_gate_pack()
    summary = pack["summary"]
    print("[trial2_multi_observable_corrected_hyperfine_gate_backend]")
    print(f"  pmodel_win_count_now = {summary['pmodel_win_count_now']}")
    print(f"  codata_win_count_now = {summary['codata_win_count_now']}")
    print(f"  split_watch_verdict_now = {summary['split_watch_verdict_now']}")
    print(f"  codata_sweep_verdict_now = {summary['codata_sweep_verdict_now']}")


if __name__ == "__main__":
    main()
