#!/usr/bin/env python3
"""Materialize one absolute alpha-to-observable formula in the QED-vacuum pack.

Purpose:
    Trial-2 observable comparison is blocked not by target selection but by the
    absence of one public, deterministic `alpha -> observable` formula on an
    independent primary surface. This backend promotes the cleanest current
    candidate to that role:

        Hydrogen 1S-2S gross-structure baseline

    under the reduced-mass Coulomb rule

        nu_1S2S(alpha) = (3/8) * mu_red * c^2 * alpha^2 / h.

    This is not claimed as a full QED precision formula. It is the first honest
    absolute alpha surface already supported by the public pack.

Inputs:
    - output/public/quantum/qed_vacuum_precision_metrics.json
    - scripts/quantum/qed_vacuum_precision.py

Outputs:
    - One in-memory audit pack consumed by `.5911-.5914` wrappers
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_qed_vacuum_alpha_observable_map_materialization_backend import (
    build_trial2_qed_vacuum_alpha_materialization_pack,
)


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"

ALPHA_P_FROZEN = 0.007302943961943229
ALPHA_COMMON = 0.00730293811658175
ALPHA_P_4D_CAN = 0.0072988143426522215
ALPHA_P_4D_VERTEX = 0.007299279720153683
ALPHA_CODATA = 0.0072973525643

C_M_PER_S = 299_792_458.0
H_J_S = 6.626_070_15e-34
M_E_KG = 9.109_383_701_5e-31
M_P_KG = 1.672_621_923_69e-27


# 関数: JSON payload を 1 本読む。
def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON payload."""
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: 水素 1S-2S gross-structure baseline を返す。

def hydrogen_1s2s_gross_frequency_hz(alpha_value: float) -> float:
    """Return the reduced-mass Coulomb 1S-2S baseline frequency."""
    mu_red = (M_E_KG * M_P_KG) / (M_E_KG + M_P_KG)
    return float((3.0 / 8.0) * mu_red * (C_M_PER_S**2) * (alpha_value**2) / H_J_S)


# 関数: 1 候補 alpha の prediction row を返す。

def build_prediction_row(*, alpha_label: str, alpha_value: float, observed_hz: float, sigma_hz: float) -> dict:
    """Return one prediction row for the 1S-2S baseline."""
    predicted_hz = hydrogen_1s2s_gross_frequency_hz(alpha_value)
    delta_hz = float(predicted_hz - observed_hz)
    rel_error = float(delta_hz / observed_hz)
    z_sigma = float(delta_hz / sigma_hz) if sigma_hz > 0.0 else math.nan
    return {
        "alpha_label": alpha_label,
        "alpha_value": float(alpha_value),
        "predicted_hz": predicted_hz,
        "observed_hz": float(observed_hz),
        "delta_hz": delta_hz,
        "relative_error_vs_observed": rel_error,
        "sigma_units_vs_observed": z_sigma,
    }


# 関数: `.5911-.5914` 用の audit pack を返す。

def build_trial2_qed_vacuum_absolute_alpha_formula_pack() -> dict:
    """Return the retained QED-vacuum absolute-formula materialization pack."""
    prior_pack = build_trial2_qed_vacuum_alpha_materialization_pack()
    metrics = read_json(PUBLIC_OUT / "qed_vacuum_precision_metrics.json")
    hydrogen = metrics["hydrogen_1s2s"]
    observed_hz = float(hydrogen["f_hz"])
    sigma_hz = float(hydrogen["sigma_hz"])

    predictions = [
        build_prediction_row(
            alpha_label="alpha_P_frozen",
            alpha_value=ALPHA_P_FROZEN,
            observed_hz=observed_hz,
            sigma_hz=sigma_hz,
        ),
        build_prediction_row(
            alpha_label="alpha_common",
            alpha_value=ALPHA_COMMON,
            observed_hz=observed_hz,
            sigma_hz=sigma_hz,
        ),
        build_prediction_row(
            alpha_label="alpha_P_4D_can",
            alpha_value=ALPHA_P_4D_CAN,
            observed_hz=observed_hz,
            sigma_hz=sigma_hz,
        ),
        build_prediction_row(
            alpha_label="alpha_P_4D_vertex",
            alpha_value=ALPHA_P_4D_VERTEX,
            observed_hz=observed_hz,
            sigma_hz=sigma_hz,
        ),
        build_prediction_row(
            alpha_label="alpha_CODATA",
            alpha_value=ALPHA_CODATA,
            observed_hz=observed_hz,
            sigma_hz=sigma_hz,
        ),
    ]
    predictions_sorted = sorted(
        predictions,
        key=lambda row: abs(float(row["relative_error_vs_observed"])),
    )
    pmodel_sorted = [
        row for row in predictions_sorted if str(row["alpha_label"]).startswith("alpha_P_") or str(row["alpha_label"]) == "alpha_common"
    ]

    surfaces = [
        {
            "surface_id": "hydrogen_1s2s_gross_structure_baseline",
            "label": "Hydrogen 1S-2S gross-structure baseline",
            "formula": "nu_1S2S(alpha) = (3/8) * mu_red * c^2 * alpha^2 / h",
            "alpha_dependency_kind": "explicit_absolute_alpha_formula",
            "current_alpha_rerun_ready_now": True,
            "independent_observable_now": True,
            "primary_score_admissible_now": True,
            "selected_primary_target_now": True,
            "notes": (
                "This is the first honest public alpha-explicit rerun surface in the "
                "current QED-vacuum pack. It is a reduced-mass Coulomb baseline, not "
                "a full QED precision formula."
            ),
            "observed_hz": observed_hz,
            "sigma_hz": sigma_hz,
            "predictions": predictions_sorted,
        },
        {
            "surface_id": "lamb_shift_absolute_formula",
            "label": "Lamb-shift absolute formula",
            "formula": None,
            "alpha_dependency_kind": "structurally_alpha_sensitive_but_absolute_formula_unavailable",
            "current_alpha_rerun_ready_now": False,
            "independent_observable_now": True,
            "primary_score_admissible_now": False,
            "selected_primary_target_now": False,
            "notes": (
                "Lamb scaling is retained, but the current public pack still lacks one "
                "deterministic absolute alpha-to-observable formula for the Lamb sector."
            ),
            "observed_hz": None,
            "sigma_hz": None,
            "predictions": [],
        },
    ]

    primary_ready_rows = [row for row in surfaces if row["primary_score_admissible_now"]]

    return {
        "alpha_constants": {
            "alpha_P_frozen": ALPHA_P_FROZEN,
            "alpha_common": ALPHA_COMMON,
            "alpha_P_4D_can": ALPHA_P_4D_CAN,
            "alpha_P_4D_vertex": ALPHA_P_4D_VERTEX,
            "alpha_CODATA": ALPHA_CODATA,
        },
        "prior_qed_pack_summary": prior_pack["summary"],
        "surfaces": surfaces,
        "summary": {
            "qed_absolute_formula_surface_count": len(surfaces),
            "qed_absolute_primary_ready_count": len(primary_ready_rows),
            "selected_primary_target_ids": [
                str(row["surface_id"]) for row in surfaces if row["selected_primary_target_now"]
            ],
            "selected_first_rerun_surface_id": "hydrogen_1s2s_gross_structure_baseline",
            "best_overall_alpha_label": str(predictions_sorted[0]["alpha_label"]),
            "best_overall_relative_error_vs_observed": float(
                predictions_sorted[0]["relative_error_vs_observed"]
            ),
            "best_pmodel_alpha_label": str(pmodel_sorted[0]["alpha_label"]),
            "best_pmodel_relative_error_vs_observed": float(
                pmodel_sorted[0]["relative_error_vs_observed"]
            ),
            "hydrogen_1s2s_observed_hz": observed_hz,
            "hydrogen_1s2s_sigma_hz": sigma_hz,
        },
        "hydrogen_1s2s_predictions": predictions_sorted,
        "trial2_qed_vacuum_absolute_formula_materialized_now": True,
        "trial2_qed_vacuum_primary_ready_now": bool(len(primary_ready_rows) > 0),
        "trial2_first_actual_qed_rerun_surface_available_now": True,
    }


# 関数: backend 単体実行時の compact summary を返す。

def main() -> None:
    """Run the absolute alpha-formula materialization backend directly."""
    pack = build_trial2_qed_vacuum_absolute_alpha_formula_pack()
    summary = pack["summary"]
    print("[trial2_qed_vacuum_absolute_alpha_formula_materialization_backend]")
    print(
        "  selected_first_rerun_surface_id = "
        f"{summary['selected_first_rerun_surface_id']}"
    )
    print(
        "  best_overall_alpha_label = "
        f"{summary['best_overall_alpha_label']}"
    )
    print(
        "  best_overall_relative_error_vs_observed = "
        f"{summary['best_overall_relative_error_vs_observed']}"
    )


if __name__ == "__main__":
    main()
