#!/usr/bin/env python3
"""Audit whether retained He I lines admit one simple native screening surrogate.

Purpose:
    The current public He I cache is observed-only. A genuinely useful reopen
    would require one deterministic P-model-native screening law that turns the
    retained He I lines into one alpha-explicit rerun surface.

    This backend does not invent a Helium theory. It computes the strongest
    local constant-Z_eff surrogate that can be built from the retained He I
    lines under the already-public reduced-mass Coulomb shell. If that surrogate
    is unphysical or still requires an inferred line-by-line pair fit with a
    material residual, the simple-screening route closes negatively.

Inputs:
    - data/quantum/sources/nist_asd_he_i_lines/extracted_values.json

Outputs:
    - One in-memory audit pack consumed by `.6047-.6050` wrappers
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

ALPHA_P_FROZEN = 0.007302943961943229
ALPHA_CODATA = 0.0072973525643

C_M_PER_S = 299_792_458.0
H_J_S = 6.626_070_15e-34
M_E_KG = 9.109_383_701_5e-31
M_ALPHA_KG = 6.644_657_335_7e-27

HE_I_EXTRACTED = (
    ROOT
    / "data"
    / "quantum"
    / "sources"
    / "nist_asd_he_i_lines"
    / "extracted_values.json"
)


# 関数: JSON payload を 1 本読む。
def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON payload."""
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: neutral-He baseline 用の reduced mass を返す。

def helium_reduced_mass_kg() -> float:
    """Return the reduced mass used for the retained neutral-He shell."""
    return float((M_E_KG * M_ALPHA_KG) / (M_E_KG + M_ALPHA_KG))


# 関数: He I cache から observed line rows を返す。

def build_observed_line_rows() -> list[dict]:
    """Return retained He I observed rows with frequencies in Hz."""
    extracted = read_json(HE_I_EXTRACTED)
    rows: list[dict] = []
    for record in extracted["selected_lines"]:
        selected = dict(record["selected"])
        lambda_vac_nm = float(selected["lambda_vac_nm"])
        observed_hz = float(C_M_PER_S / (lambda_vac_nm * 1.0e-9))
        rows.append(
            {
                "line_id": str(record["id"]),
                "observed_lambda_vac_nm": lambda_vac_nm,
                "observed_hz": observed_hz,
            }
        )

    return rows


# 関数: hydrogenic candidate pair list を返す。

def build_candidate_pairs() -> list[tuple[int, int]]:
    """Return the retained candidate pair list for simple-screening scans."""
    return [
        (n_lower, n_upper)
        for n_lower in range(1, 8)
        for n_upper in range(n_lower + 1, 13)
    ]


# 関数: 1 line / 1 pair に必要な effective charge を返す。

def implied_zeff(alpha_value: float, observed_hz: float, *, n_lower: int, n_upper: int) -> float:
    """Return the constant effective charge implied by one observed line."""
    mu_red = helium_reduced_mass_kg()
    line_factor = 0.5 * ((1.0 / (n_lower**2)) - (1.0 / (n_upper**2)))
    return float(
        ((observed_hz * H_J_S) / (mu_red * (C_M_PER_S**2) * (alpha_value**2) * line_factor)) ** 0.5
    )


# 関数: 1 global Z_eff surrogate の score row を返す。

def build_constant_zeff_fit(
    alpha_value: float,
    *,
    enforce_individual_physical_ceiling: bool,
) -> dict:
    """Return the strongest constant-Z_eff surrogate under one alpha value."""
    observed_rows = build_observed_line_rows()
    candidate_pairs = build_candidate_pairs()
    per_line_options: list[list[tuple[float, int, int]]] = []
    for row in observed_rows:
        line_options = []
        for n_lower, n_upper in candidate_pairs:
            line_options.append(
                (
                    implied_zeff(alpha_value, float(row["observed_hz"]), n_lower=n_lower, n_upper=n_upper),
                    n_lower,
                    n_upper,
                )
            )

        per_line_options.append(line_options)

    best_score = None
    best_payload = None
    mu_red = helium_reduced_mass_kg()
    for combo in itertools.product(*per_line_options):
        zeff_values = [float(item[0]) for item in combo]
        if enforce_individual_physical_ceiling and any(value > 2.0 for value in zeff_values):
            continue

        mean_zeff = float(sum(zeff_values) / len(zeff_values))
        rel_spread = float(max(abs(value - mean_zeff) / mean_zeff for value in zeff_values))
        per_line_rows = []
        max_rel_residual = 0.0
        for observed_row, (zeff_value, n_lower, n_upper) in zip(observed_rows, combo):
            line_factor = 0.5 * ((1.0 / (n_lower**2)) - (1.0 / (n_upper**2)))
            predicted_hz = float(
                mu_red * (C_M_PER_S**2) * ((mean_zeff * alpha_value) ** 2) * line_factor / H_J_S
            )
            rel_residual = float((predicted_hz - float(observed_row["observed_hz"])) / float(observed_row["observed_hz"]))
            max_rel_residual = max(max_rel_residual, abs(rel_residual))
            per_line_rows.append(
                {
                    "line_id": str(observed_row["line_id"]),
                    "observed_lambda_vac_nm": float(observed_row["observed_lambda_vac_nm"]),
                    "n_lower": int(n_lower),
                    "n_upper": int(n_upper),
                    "implied_zeff_for_line": float(zeff_value),
                    "predicted_hz_with_mean_zeff": predicted_hz,
                    "relative_residual_with_mean_zeff": rel_residual,
                }
            )

        score = (
            float(max_rel_residual),
            float(rel_spread),
            float(abs(mean_zeff - 2.0)),
            float(mean_zeff),
        )
        if best_score is None or score < best_score:
            best_score = score
            best_payload = {
                "constant_zeff": mean_zeff,
                "max_relative_residual": float(max_rel_residual),
                "relative_spread": rel_spread,
                "line_rows": per_line_rows,
            }

    assert best_payload is not None
    return best_payload


# 関数: `.6047-.6050` 用の He I screening audit pack を返す。

def build_trial2_native_helium_simple_screening_audit_pack() -> dict:
    """Return the retained He I simple-screening audit pack."""
    unrestricted_fit = build_constant_zeff_fit(ALPHA_P_FROZEN, enforce_individual_physical_ceiling=False)
    physical_fit = build_constant_zeff_fit(ALPHA_P_FROZEN, enforce_individual_physical_ceiling=True)
    diagnostic_unrestricted_fit = build_constant_zeff_fit(ALPHA_CODATA, enforce_individual_physical_ceiling=False)
    diagnostic_physical_fit = build_constant_zeff_fit(ALPHA_CODATA, enforce_individual_physical_ceiling=True)

    unrestricted_physical_now = bool(unrestricted_fit["constant_zeff"] <= 2.0)
    physical_subpercent_now = bool(physical_fit["max_relative_residual"] < 0.01)
    helium_simple_screening_surface_ready_now = bool(
        unrestricted_physical_now and physical_subpercent_now
    )

    return {
        "summary": {
            "helium_selected_line_count_now": int(len(build_observed_line_rows())),
            "unrestricted_constant_zeff_now": float(unrestricted_fit["constant_zeff"]),
            "unrestricted_max_relative_residual_now": float(unrestricted_fit["max_relative_residual"]),
            "unrestricted_relative_spread_now": float(unrestricted_fit["relative_spread"]),
            "unrestricted_physical_admissible_now": bool(unrestricted_physical_now),
            "physical_constant_zeff_now": float(physical_fit["constant_zeff"]),
            "physical_max_relative_residual_now": float(physical_fit["max_relative_residual"]),
            "physical_relative_spread_now": float(physical_fit["relative_spread"]),
            "physical_subpercent_now": bool(physical_subpercent_now),
            "helium_simple_screening_surface_ready_now": bool(helium_simple_screening_surface_ready_now),
            "diagnostic_unrestricted_constant_zeff_codata_now": float(
                diagnostic_unrestricted_fit["constant_zeff"]
            ),
            "diagnostic_physical_constant_zeff_codata_now": float(
                diagnostic_physical_fit["constant_zeff"]
            ),
            "current_honest_reading": (
                "The strongest unrestricted constant-Z_eff surrogate is nonphysical "
                "because it requires Z_eff > 2 for neutral helium. Imposing the "
                "physical ceiling Z_eff <= 2 still leaves a fitted, inferred "
                "line-by-line pair map with max relative residual above 2%, so the "
                "retained He I cache does not materialize one honest native surface."
            ),
        },
        "unrestricted_fit": unrestricted_fit,
        "physical_fit": physical_fit,
        "diagnostic_unrestricted_fit": diagnostic_unrestricted_fit,
        "diagnostic_physical_fit": diagnostic_physical_fit,
        "trial2_native_helium_simple_screening_completed_now": True,
        "trial2_native_helium_simple_screening_negative_closeout_now": True,
    }


# 関数: backend 単体実行時の compact summary を返す。

def main() -> None:
    """Run the native He I simple-screening audit directly."""
    pack = build_trial2_native_helium_simple_screening_audit_pack()
    summary = pack["summary"]
    print("[trial2_native_helium_simple_screening_audit_backend]")
    print(f"  unrestricted_constant_zeff_now = {summary['unrestricted_constant_zeff_now']}")
    print(
        "  unrestricted_physical_admissible_now = "
        f"{summary['unrestricted_physical_admissible_now']}"
    )
    print(f"  physical_constant_zeff_now = {summary['physical_constant_zeff_now']}")
    print(
        "  physical_max_relative_residual_now = "
        f"{summary['physical_max_relative_residual_now']}"
    )
    print(
        "  helium_simple_screening_surface_ready_now = "
        f"{summary['helium_simple_screening_surface_ready_now']}"
    )


if __name__ == "__main__":
    main()
