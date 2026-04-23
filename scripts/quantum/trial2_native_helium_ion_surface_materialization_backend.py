#!/usr/bin/env python3
"""Materialize one native non-Hydrogen alpha-explicit surface from He II.

Purpose:
    Trial-2 native observable validation currently stops at a two-surface
    Hydrogen table because the retained Halpha fine-structure route still lacks
    one public-canonical relativistic bridge. The cleanest honest reopen
    candidate is a one-electron non-Hydrogen ion:

        He II (He+) 468.67 nm hydrogenic gross-structure baseline.

    This stays inside the already-public nonrelativistic P-model canon:

    - Part III-A positive-frequency KG -> Schr envelope
    - Trial-1 / Trial-2 Coulomb route

    and avoids the screening-law blocker that closed neutral He I negatively.

Inputs:
    - data/quantum/sources/nist_asd_he_ii_lines/extracted_values.json

Outputs:
    - One in-memory audit pack consumed by `.6027-.6030` wrappers
"""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

ALPHA_P_FROZEN = 0.007302943961943229
ALPHA_COMMON = 0.00730293811658175
ALPHA_P_4D_CAN = 0.0072988143426522215
ALPHA_P_4D_VERTEX = 0.007299279720153683
ALPHA_CODATA = 0.0072973525643

C_M_PER_S = 299_792_458.0
H_J_S = 6.626_070_15e-34
M_E_KG = 9.109_383_701_5e-31
M_ALPHA_KG = 6.644_657_335_7e-27

HE_II_EXTRACTED = (
    ROOT
    / "data"
    / "quantum"
    / "sources"
    / "nist_asd_he_ii_lines"
    / "extracted_values.json"
)

HE_II_TARGET_ID = "He_II_468.67nm"
HE_II_SURFACE_ID = "helium_ii_468_67nm_gross_structure_baseline"
HE_II_Z = 2.0
HE_II_N_LOWER = 3
HE_II_N_UPPER = 4


# 関数: JSON payload を 1 本読む。
def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON payload."""
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: He II reduced mass を返す。

def helium_ion_reduced_mass_kg() -> float:
    """Return the reduced mass for one-electron helium ion."""
    return float((M_E_KG * M_ALPHA_KG) / (M_E_KG + M_ALPHA_KG))


# 関数: He II 4->3 hydrogenic gross-structure baseline frequency を返す。

def helium_ii_468_frequency_hz(alpha_value: float) -> float:
    """Return the He II 4->3 hydrogenic gross-structure baseline frequency."""
    mu_red = helium_ion_reduced_mass_kg()
    line_factor = 0.5 * ((1.0 / (HE_II_N_LOWER**2)) - (1.0 / (HE_II_N_UPPER**2)))
    return float(
        mu_red
        * (C_M_PER_S**2)
        * ((HE_II_Z * alpha_value) ** 2)
        * line_factor
        / H_J_S
    )


# 関数: 1 候補 alpha の prediction row を返す。

def build_prediction_row(*, alpha_label: str, alpha_value: float, observed_hz: float, sigma_hz: float) -> dict:
    """Return one prediction row for the He II 468.67 nm baseline."""
    predicted_hz = helium_ii_468_frequency_hz(alpha_value)
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


# 関数: `.6027-.6030` 用の He II surface pack を返す。

def build_trial2_native_helium_ion_surface_materialization_pack() -> dict:
    """Return the native He II surface materialization pack."""
    extracted = read_json(HE_II_EXTRACTED)
    selected_lines = list(extracted["selected_lines"])
    selected = next(row for row in selected_lines if str(row["id"]) == HE_II_TARGET_ID)
    selected_line = dict(selected["selected"])
    observed_hz = float(selected_line["frequency_hz"])
    sigma_hz = float(selected_line["frequency_unc_hz"])

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
    best_overall = predictions_sorted[0]
    best_pmodel = pmodel_sorted[0]

    surface = {
        "surface_id": HE_II_SURFACE_ID,
        "label": "He II 468.67 nm hydrogenic gross-structure baseline",
        "family_id": "helium_ii_hydrogenic_gross_structure_family",
        "formula": (
            "nu_HeII,4to3(alpha) = (mu_red * c^2 / (2 h)) * (Z alpha)^2 * "
            "(1/3^2 - 1/4^2), with Z = 2"
        ),
        "alpha_dependency_kind": "explicit_absolute_alpha_formula",
        "current_alpha_rerun_ready_now": True,
        "independent_observable_now": True,
        "primary_score_admissible_now": True,
        "selected_primary_target_now": True,
        "spectra_token": str(extracted["spectra"]),
        "selected_line_id": HE_II_TARGET_ID,
        "selected_line_source": str(selected_line["selected_nu_source"]),
        "observed_lambda_vac_nm": float(selected_line["lambda_vac_nm"]),
        "observed_hz": observed_hz,
        "sigma_hz": sigma_hz,
        "predictions": predictions_sorted,
        "notes": (
            "This is a one-electron He+ surface, so the neutral-He screening-law "
            "blocker does not apply. The current public canon already supplies the "
            "nonrelativistic envelope plus Coulomb route needed for this gross-structure baseline."
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
            "m_e_kg": M_E_KG,
            "m_alpha_kg": M_ALPHA_KG,
            "c_m_per_s": C_M_PER_S,
            "h_j_s": H_J_S,
            "he_ii_z": HE_II_Z,
            "he_ii_n_lower": HE_II_N_LOWER,
            "he_ii_n_upper": HE_II_N_UPPER,
        },
        "surface": surface,
        "summary": {
            "surface_id": HE_II_SURFACE_ID,
            "selected_line_id": HE_II_TARGET_ID,
            "observed_lambda_vac_nm": float(selected_line["lambda_vac_nm"]),
            "best_overall_alpha_label": str(best_overall["alpha_label"]),
            "best_overall_relative_error_vs_observed": float(
                best_overall["relative_error_vs_observed"]
            ),
            "best_pmodel_alpha_label": str(best_pmodel["alpha_label"]),
            "best_pmodel_relative_error_vs_observed": float(
                best_pmodel["relative_error_vs_observed"]
            ),
            "native_non_hydrogen_surface_ready_now": True,
        },
        "trial2_native_helium_ion_surface_materialized_now": True,
        "trial2_native_non_hydrogen_surface_ready_now": True,
    }


# 関数: backend 単体実行時の compact summary を返す。

def main() -> None:
    """Run the native He II surface backend directly."""
    pack = build_trial2_native_helium_ion_surface_materialization_pack()
    summary = pack["summary"]
    print("[trial2_native_helium_ion_surface_materialization_backend]")
    print(f"  surface_id = {summary['surface_id']}")
    print(f"  best_overall_alpha_label = {summary['best_overall_alpha_label']}")
    print(
        "  best_pmodel_relative_error_vs_observed = "
        f"{summary['best_pmodel_relative_error_vs_observed']}"
    )


if __name__ == "__main__":
    main()
