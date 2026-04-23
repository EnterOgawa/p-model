#!/usr/bin/env python3
"""Audit whether the retained He II family contains a stronger native route.

Purpose:
    The He II 468.67 nm one-electron baseline already actualized one genuine
    non-Hydrogen native surface. The next honest question is whether other
    retained He II hydrogenic lines can overturn the current watch verdict or
    whether they are merely same-family replays.

Inputs:
    - data/quantum/sources/nist_asd_he_ii_lines/extracted_values.json

Outputs:
    - One in-memory audit pack consumed by `.6039-.6042` wrappers
"""

from __future__ import annotations

import json
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


# 関数: JSON payload を 1 本読む。
def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON payload."""
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: He II reduced mass を返す。

def helium_ion_reduced_mass_kg() -> float:
    """Return the reduced mass for one-electron helium ion."""
    return float((M_E_KG * M_ALPHA_KG) / (M_E_KG + M_ALPHA_KG))


# 関数: 候補 n-pair の frequency を返す。

def helium_line_frequency_hz(alpha_value: float, *, n_lower: int, n_upper: int) -> float:
    """Return one hydrogenic He II transition frequency."""
    mu_red = helium_ion_reduced_mass_kg()
    line_factor = 0.5 * ((1.0 / (n_lower**2)) - (1.0 / (n_upper**2)))
    return float(mu_red * (C_M_PER_S**2) * ((2.0 * alpha_value) ** 2) * line_factor / H_J_S)


# 関数: retained line に最も近い hydrogenic n-pair を返す。

def infer_best_n_pair(observed_hz: float) -> tuple[int, int]:
    """Infer the best hydrogenic pair by the CODATA row."""
    best = None
    for n_lower in range(1, 8):
        for n_upper in range(n_lower + 1, 12):
            predicted = helium_line_frequency_hz(ALPHA_CODATA, n_lower=n_lower, n_upper=n_upper)
            rel = abs((predicted - observed_hz) / observed_hz)
            candidate = (rel, n_lower, n_upper)
            if best is None or candidate < best:
                best = candidate

    assert best is not None
    return int(best[1]), int(best[2])


# 関数: 1 候補 alpha の relative residual を返す。

def build_prediction_row(*, alpha_label: str, alpha_value: float, observed_hz: float, n_lower: int, n_upper: int) -> dict:
    """Return one relative-residual row for one He II line."""
    predicted_hz = helium_line_frequency_hz(alpha_value, n_lower=n_lower, n_upper=n_upper)
    rel_error = float((predicted_hz - observed_hz) / observed_hz)
    return {
        "alpha_label": alpha_label,
        "alpha_value": float(alpha_value),
        "predicted_hz": predicted_hz,
        "relative_error_vs_observed": rel_error,
    }


# 関数: `.6039-.6042` 用の He II family-strength audit pack を返す。

def build_trial2_native_helium_ion_family_strength_audit_pack() -> dict:
    """Return the retained He II family-strength audit pack."""
    extracted = read_json(HE_II_EXTRACTED)
    line_rows = []
    for selected in extracted["selected_lines"]:
        selected_line = dict(selected["selected"])
        observed_hz = float(selected_line["frequency_hz"])
        n_lower, n_upper = infer_best_n_pair(observed_hz)
        predictions = [
            build_prediction_row(
                alpha_label="alpha_P_frozen",
                alpha_value=ALPHA_P_FROZEN,
                observed_hz=observed_hz,
                n_lower=n_lower,
                n_upper=n_upper,
            ),
            build_prediction_row(
                alpha_label="alpha_common",
                alpha_value=ALPHA_COMMON,
                observed_hz=observed_hz,
                n_lower=n_lower,
                n_upper=n_upper,
            ),
            build_prediction_row(
                alpha_label="alpha_P_4D_can",
                alpha_value=ALPHA_P_4D_CAN,
                observed_hz=observed_hz,
                n_lower=n_lower,
                n_upper=n_upper,
            ),
            build_prediction_row(
                alpha_label="alpha_P_4D_vertex",
                alpha_value=ALPHA_P_4D_VERTEX,
                observed_hz=observed_hz,
                n_lower=n_lower,
                n_upper=n_upper,
            ),
            build_prediction_row(
                alpha_label="alpha_CODATA",
                alpha_value=ALPHA_CODATA,
                observed_hz=observed_hz,
                n_lower=n_lower,
                n_upper=n_upper,
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
        line_rows.append(
            {
                "line_id": str(selected["id"]),
                "observed_lambda_vac_nm": float(selected_line["lambda_vac_nm"]),
                "n_lower": n_lower,
                "n_upper": n_upper,
                "best_overall_alpha_label": str(predictions_sorted[0]["alpha_label"]),
                "best_overall_relative_error_vs_observed": float(
                    predictions_sorted[0]["relative_error_vs_observed"]
                ),
                "best_pmodel_alpha_label": str(pmodel_sorted[0]["alpha_label"]),
                "best_pmodel_relative_error_vs_observed": float(
                    pmodel_sorted[0]["relative_error_vs_observed"]
                ),
            }
        )

    strongest_pmodel_row = min(
        line_rows,
        key=lambda row: abs(float(row["best_pmodel_relative_error_vs_observed"])),
    )
    codata_win_count = int(sum(1 for row in line_rows if row["best_overall_alpha_label"] == "alpha_CODATA"))

    return {
        "line_rows": line_rows,
        "summary": {
            "heii_family_line_count_now": int(len(line_rows)),
            "heii_family_codata_win_count_now": codata_win_count,
            "heii_family_pmodel_win_count_now": int(len(line_rows) - codata_win_count),
            "heii_family_strongest_pmodel_line_id_now": str(strongest_pmodel_row["line_id"]),
            "heii_family_strongest_pmodel_relative_error_vs_observed_now": float(
                strongest_pmodel_row["best_pmodel_relative_error_vs_observed"]
            ),
            "heii_family_stronger_than_46867_route_available_now": bool(
                strongest_pmodel_row["line_id"] != "He_II_468.67nm"
            ),
        },
        "trial2_native_helium_ion_same_family_negative_closeout_now": bool(
            strongest_pmodel_row["line_id"] == "He_II_468.67nm"
        ),
    }


# 関数: backend 単体実行時の compact summary を返す。

def main() -> None:
    """Run the native He II family-strength audit directly."""
    pack = build_trial2_native_helium_ion_family_strength_audit_pack()
    summary = pack["summary"]
    print("[trial2_native_helium_ion_family_strength_audit_backend]")
    print(f"  heii_family_line_count_now = {summary['heii_family_line_count_now']}")
    print(
        "  heii_family_strongest_pmodel_line_id_now = "
        f"{summary['heii_family_strongest_pmodel_line_id_now']}"
    )
    print(
        "  heii_family_stronger_than_46867_route_available_now = "
        f"{summary['heii_family_stronger_than_46867_route_available_now']}"
    )


if __name__ == "__main__":
    main()
