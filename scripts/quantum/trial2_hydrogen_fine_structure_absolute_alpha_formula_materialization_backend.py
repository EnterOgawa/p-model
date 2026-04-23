#!/usr/bin/env python3
"""Materialize one Hydrogen fine-structure absolute alpha-to-observable surface.

Purpose:
    Trial-2 observable comparison is still blocked by the absence of one
    genuinely new third independent alpha-explicit rerun surface. The retained
    Hydrogen gross-structure and corrected hyperfine surfaces are already
    materialized, but the current pack still lacks one public deterministic
    fine-structure baseline.

    This backend promotes the cleanest available candidate to that role:

        Hydrogen H-alpha multiplet fine-structure span

    under a reduced-mass Dirac-Coulomb baseline. The observable is the span
    between the maximum and minimum allowed 3->2 E1 transition frequencies.

Inputs:
    - output/public/quantum/atomic_hydrogen_baseline_metrics.json

Outputs:
    - One in-memory audit pack consumed by `.5971-.5974` wrappers
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.trial2_qed_vacuum_absolute_alpha_formula_materialization_backend import (
    ALPHA_CODATA,
    ALPHA_COMMON,
    ALPHA_P_4D_CAN,
    ALPHA_P_4D_VERTEX,
    ALPHA_P_FROZEN,
    C_M_PER_S,
    H_J_S,
    M_E_KG,
    M_P_KG,
)


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
ATOMIC_HYDROGEN = PUBLIC_OUT / "atomic_hydrogen_baseline_metrics.json"

ALPHA_ROWS = {
    "alpha_P_frozen": ALPHA_P_FROZEN,
    "alpha_common": ALPHA_COMMON,
    "alpha_P_4D_can": ALPHA_P_4D_CAN,
    "alpha_P_4D_vertex": ALPHA_P_4D_VERTEX,
    "alpha_CODATA": ALPHA_CODATA,
}
HALPHA_SURFACE_ID = "hydrogen_halpha_fine_structure_dirac_span_baseline"


# 関数: JSON payload を 1 本読む。
def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON payload."""
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: H-alpha multiplet row を baseline metrics から返す。

def find_halpha_multiplet(metrics: dict) -> dict:
    """Return the retained H-alpha multiplet dictionary."""
    for row in metrics["multiplets"]:
        if str(row["id"]) == "H_I_Hα":
            return row

    raise KeyError("H_I_Hα multiplet is missing from atomic_hydrogen_baseline_metrics.json")


# 関数: Dirac-Coulomb の delta_j を返す。

def dirac_delta_j(*, j_value: float, alpha_value: float) -> float:
    """Return the reduced Dirac-Coulomb delta_j(alpha)."""
    kappa_abs = float(j_value + 0.5)
    return float(kappa_abs - math.sqrt((kappa_abs**2) - (alpha_value**2)))


# 関数: reduced-mass Dirac bound energy を返す。

def dirac_hydrogen_bound_energy_joule(*, principal_n: int, j_value: float, alpha_value: float) -> float:
    """Return one reduced-mass Dirac-Coulomb bound energy in joule."""
    mu_red = (M_E_KG * M_P_KG) / (M_E_KG + M_P_KG)
    delta_value = dirac_delta_j(j_value=j_value, alpha_value=alpha_value)
    denominator = (principal_n - delta_value) ** 2
    return float(
        mu_red
        * (C_M_PER_S**2)
        * ((1.0 + ((alpha_value**2) / denominator)) ** (-0.5) - 1.0)
    )


# 関数: H-alpha allowed Dirac transition list を返す。

def build_halpha_allowed_transition_rows(*, alpha_value: float) -> list[dict]:
    """Return all retained Dirac-Coulomb allowed H-alpha transition rows."""
    upper_levels = [
        {"term": "3S1/2", "n": 3, "l": 0, "j": 0.5},
        {"term": "3P1/2", "n": 3, "l": 1, "j": 0.5},
        {"term": "3P3/2", "n": 3, "l": 1, "j": 1.5},
        {"term": "3D3/2", "n": 3, "l": 2, "j": 1.5},
        {"term": "3D5/2", "n": 3, "l": 2, "j": 2.5},
    ]
    lower_levels = [
        {"term": "2S1/2", "n": 2, "l": 0, "j": 0.5},
        {"term": "2P1/2", "n": 2, "l": 1, "j": 0.5},
        {"term": "2P3/2", "n": 2, "l": 1, "j": 1.5},
    ]
    rows: list[dict] = []

    for upper in upper_levels:
        for lower in lower_levels:
            if abs(int(upper["l"]) - int(lower["l"])) != 1:
                continue

            if abs(float(upper["j"]) - float(lower["j"])) > 1.0:
                continue

            upper_energy = dirac_hydrogen_bound_energy_joule(
                principal_n=int(upper["n"]),
                j_value=float(upper["j"]),
                alpha_value=alpha_value,
            )
            lower_energy = dirac_hydrogen_bound_energy_joule(
                principal_n=int(lower["n"]),
                j_value=float(lower["j"]),
                alpha_value=alpha_value,
            )
            frequency_hz = abs(upper_energy - lower_energy) / H_J_S
            rows.append(
                {
                    "upper_term": str(upper["term"]),
                    "lower_term": str(lower["term"]),
                    "transition_id": f"{upper['term']}->{lower['term']}",
                    "frequency_hz": float(frequency_hz),
                }
            )

    rows.sort(key=lambda row: float(row["frequency_hz"]))
    return rows


# 関数: H-alpha Dirac fine-structure span を返す。

def hydrogen_halpha_fine_structure_dirac_span_hz(alpha_value: float) -> float:
    """Return the retained H-alpha fine-structure span under the Dirac baseline."""
    rows = build_halpha_allowed_transition_rows(alpha_value=alpha_value)
    return float(rows[-1]["frequency_hz"] - rows[0]["frequency_hz"])


# 関数: 1 候補 alpha の prediction row を返す。

def build_prediction_row(*, alpha_label: str, alpha_value: float, observed_hz: float, sigma_hz: float) -> dict:
    """Return one prediction row for the retained H-alpha fine-structure span."""
    predicted_hz = hydrogen_halpha_fine_structure_dirac_span_hz(alpha_value)
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


# 関数: `.5971-.5974` 用の audit pack を返す。

def build_trial2_hydrogen_fine_structure_absolute_alpha_formula_pack() -> dict:
    """Return the retained Hydrogen fine-structure absolute-formula pack."""
    metrics = read_json(ATOMIC_HYDROGEN)
    halpha = find_halpha_multiplet(metrics)
    observed_components = [
        float(C_M_PER_S / (float(row["lambda_vac_nm"]) * 1.0e-9))
        for row in halpha["components"]
    ]
    observed_hz = float(max(observed_components) - min(observed_components))
    sigma_hz = 0.0
    transition_rows = build_halpha_allowed_transition_rows(alpha_value=ALPHA_CODATA)

    predictions = [
        build_prediction_row(
            alpha_label=alpha_label,
            alpha_value=alpha_value,
            observed_hz=observed_hz,
            sigma_hz=sigma_hz,
        )
        for alpha_label, alpha_value in ALPHA_ROWS.items()
    ]
    predictions_sorted = sorted(predictions, key=lambda row: abs(float(row["relative_error_vs_observed"])))
    pmodel_sorted = [
        row
        for row in predictions_sorted
        if str(row["alpha_label"]).startswith("alpha_P_") or str(row["alpha_label"]) == "alpha_common"
    ]

    return {
        "surface": {
            "surface_id": HALPHA_SURFACE_ID,
            "label": "Hydrogen H-alpha fine-structure Dirac span baseline",
            "formula": (
                "nu_fs_span(alpha) = max_allowed |E_3,j_u(alpha) - E_2,j_l(alpha)| / h "
                "- min_allowed |E_3,j_u(alpha) - E_2,j_l(alpha)| / h"
            ),
            "energy_formula": (
                "E_n,j(alpha) = mu_red c^2 * [1 + alpha^2 / (n - delta_j)^2]^(-1/2) - mu_red c^2, "
                "delta_j = j + 1/2 - sqrt((j + 1/2)^2 - alpha^2)"
            ),
            "alpha_dependency_kind": "explicit_absolute_alpha_formula",
            "family_id": "hydrogen_dirac_fine_structure_span_family",
            "current_alpha_rerun_ready_now": True,
            "independent_observable_now": True,
            "primary_score_admissible_now": True,
            "selected_primary_target_now": True,
            "genuinely_new_independent_surface_now": True,
            "observed_hz": observed_hz,
            "sigma_hz": sigma_hz,
            "n_components_observed": int(halpha["n_components"]),
            "transition_rows_codata": transition_rows,
            "predictions": predictions_sorted,
            "notes": (
                "This is a reduced-mass Dirac-Coulomb fine-structure baseline for "
                "the retained H-alpha multiplet span. It is a new alpha-explicit "
                "family relative to the gross alpha^2 baseline and the hyperfine "
                "magnetic-contact surface, but it is not claimed as a full QED "
                "spectroscopy precision formula."
            ),
        },
        "summary": {
            "selected_surface_id": HALPHA_SURFACE_ID,
            "selected_observed_hz": observed_hz,
            "selected_best_overall_alpha_label": str(predictions_sorted[0]["alpha_label"]),
            "selected_best_overall_relative_error_vs_observed": float(
                predictions_sorted[0]["relative_error_vs_observed"]
            ),
            "selected_best_pmodel_alpha_label": str(pmodel_sorted[0]["alpha_label"]),
            "selected_best_pmodel_relative_error_vs_observed": float(
                pmodel_sorted[0]["relative_error_vs_observed"]
            ),
            "selected_surface_ready_now": True,
            "selected_surface_is_genuinely_new_now": True,
        },
        "trial2_hydrogen_fine_structure_absolute_formula_materialized_now": True,
        "trial2_hydrogen_fine_structure_surface_ready_now": True,
    }


# 関数: backend 単体実行時の compact summary を返す。

def main() -> None:
    """Run the Hydrogen fine-structure absolute alpha-formula backend directly."""
    pack = build_trial2_hydrogen_fine_structure_absolute_alpha_formula_pack()
    summary = pack["summary"]
    print("[trial2_hydrogen_fine_structure_absolute_alpha_formula_materialization_backend]")
    print(f"  selected_surface_id = {summary['selected_surface_id']}")
    print(f"  selected_best_overall_alpha_label = {summary['selected_best_overall_alpha_label']}")
    print(
        "  selected_best_overall_relative_error_vs_observed = "
        f"{summary['selected_best_overall_relative_error_vs_observed']}"
    )


if __name__ == "__main__":
    main()
