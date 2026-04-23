#!/usr/bin/env python3
"""Materialize one absolute alpha-to-observable formula on H I hyperfine 21 cm.

Purpose:
    Trial-2 observable comparison needs a genuinely new second independent
    alpha-explicit rerun surface. The retained Step 7.12 hydrogen hyperfine
    baseline already fixes the observed 21 cm frequency from NIST AtSpec.
    This backend promotes the minimal nonrelativistic Fermi-contact baseline
    to one public deterministic alpha-to-observable map:

        nu_hfs(alpha)
        = (8/3) * alpha^4 * (mu_p / mu_B) * (mu_red / m_e)^3 * m_e c^2 / h

    This is not a full hydrogen hyperfine precision formula. It is a tree-level
    absolute alpha baseline that stays within the current public pack and
    supplies the second independent rerun surface.

Inputs:
    - data/quantum/sources/nist_atspec_handbook/extracted_values.json
    - scripts/quantum/atomic_hydrogen_hyperfine_baseline.py

Outputs:
    - One in-memory audit pack consumed by `.5935-.5938` wrappers
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


ALPHA_P_FROZEN = 0.007302943961943229
ALPHA_COMMON = 0.00730293811658175
ALPHA_P_4D_CAN = 0.0072988143426522215
ALPHA_P_4D_VERTEX = 0.007299279720153683
ALPHA_CODATA = 0.0072973525643

C_M_PER_S = 299_792_458.0
H_J_S = 6.626_070_15e-34
M_E_KG = 9.109_383_701_5e-31
M_P_KG = 1.672_621_923_69e-27

# CODATA 2022 / NIST all-constants listing:
#   proton magnetic moment to Bohr magneton ratio mu_p / mu_B
MU_P_OVER_MU_B = 1.521_032_202_30e-3

ATSPEC_EXTRACTED = ROOT / "data" / "quantum" / "sources" / "nist_atspec_handbook" / "extracted_values.json"


# 関数: JSON payload を 1 本読む。
def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON payload."""
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: H I 21 cm Fermi baseline frequency を返す。

def hydrogen_hyperfine_fermi_frequency_hz(alpha_value: float) -> float:
    """Return the hydrogen 21 cm Fermi-contact baseline frequency."""
    mu_red = (M_E_KG * M_P_KG) / (M_E_KG + M_P_KG)
    return float(
        (8.0 / 3.0)
        * (alpha_value**4)
        * MU_P_OVER_MU_B
        * ((mu_red / M_E_KG) ** 3)
        * M_E_KG
        * (C_M_PER_S**2)
        / H_J_S
    )


# 関数: 1 候補 alpha の prediction row を返す。

def build_prediction_row(*, alpha_label: str, alpha_value: float, observed_hz: float, sigma_hz: float) -> dict:
    """Return one prediction row for the 21 cm Fermi baseline."""
    predicted_hz = hydrogen_hyperfine_fermi_frequency_hz(alpha_value)
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


# 関数: `.5935-.5938` 用の audit pack を返す。

def build_trial2_hydrogen_hyperfine_absolute_alpha_formula_pack() -> dict:
    """Return the retained hydrogen hyperfine absolute-formula materialization pack."""
    extracted = read_json(ATSPEC_EXTRACTED)
    hyperfine = extracted["hydrogen_hyperfine_21cm"]
    observed_hz = float(hyperfine["f_hz"])
    sigma_hz = float(hyperfine["sigma_hz"])

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
        row
        for row in predictions_sorted
        if str(row["alpha_label"]).startswith("alpha_P_") or str(row["alpha_label"]) == "alpha_common"
    ]

    surface = {
        "surface_id": "hydrogen_hyperfine_21cm_fermi_baseline",
        "label": "Hydrogen hyperfine 21 cm Fermi baseline",
        "formula": (
            "nu_hfs(alpha) = (8/3) * alpha^4 * (mu_p / mu_B) * "
            "(mu_red / m_e)^3 * m_e c^2 / h"
        ),
        "formula_equivalent": (
            "nu_hfs(alpha) = (16/3) * alpha^2 * c * R_infty(alpha) * "
            "(mu_p / mu_B) * (mu_red / m_e)^3"
        ),
        "alpha_dependency_kind": "explicit_absolute_alpha_formula",
        "current_alpha_rerun_ready_now": True,
        "independent_observable_now": True,
        "primary_score_admissible_now": True,
        "selected_primary_target_now": True,
        "observed_hz": observed_hz,
        "sigma_hz": sigma_hz,
        "source_token_mhz": str(hyperfine["token"]),
        "predictions": predictions_sorted,
        "notes": (
            "This is a nonrelativistic Fermi-contact baseline for the H I 21 cm "
            "hyperfine transition. It is not claimed as the full hydrogen "
            "hyperfine precision theory."
        ),
    }

    return {
        "alpha_constants": {
            "alpha_P_frozen": ALPHA_P_FROZEN,
            "alpha_common": ALPHA_COMMON,
            "alpha_P_4D_can": ALPHA_P_4D_CAN,
            "alpha_P_4D_vertex": ALPHA_P_4D_VERTEX,
            "alpha_CODATA": ALPHA_CODATA,
        },
        "source_constants": {
            "mu_p_over_mu_B": MU_P_OVER_MU_B,
            "m_e_kg": M_E_KG,
            "m_p_kg": M_P_KG,
            "c_m_per_s": C_M_PER_S,
            "h_j_s": H_J_S,
        },
        "surface": surface,
        "summary": {
            "hyperfine_surface_id": str(surface["surface_id"]),
            "hyperfine_surface_ready_now": True,
            "best_overall_alpha_label": str(predictions_sorted[0]["alpha_label"]),
            "best_overall_relative_error_vs_observed": float(
                predictions_sorted[0]["relative_error_vs_observed"]
            ),
            "best_pmodel_alpha_label": str(pmodel_sorted[0]["alpha_label"]),
            "best_pmodel_relative_error_vs_observed": float(
                pmodel_sorted[0]["relative_error_vs_observed"]
            ),
            "observed_hz": observed_hz,
            "sigma_hz": sigma_hz,
        },
        "trial2_hyperfine_absolute_formula_materialized_now": True,
        "trial2_hyperfine_surface_ready_now": True,
    }


# 関数: backend 単体実行時の compact summary を返す。

def main() -> None:
    """Run the hydrogen hyperfine absolute alpha-formula backend directly."""
    pack = build_trial2_hydrogen_hyperfine_absolute_alpha_formula_pack()
    summary = pack["summary"]
    print("[trial2_hydrogen_hyperfine_absolute_alpha_formula_materialization_backend]")
    print(f"  hyperfine_surface_id = {summary['hyperfine_surface_id']}")
    print(f"  best_overall_alpha_label = {summary['best_overall_alpha_label']}")
    print(
        "  best_overall_relative_error_vs_observed = "
        f"{summary['best_overall_relative_error_vs_observed']}"
    )


if __name__ == "__main__":
    main()
