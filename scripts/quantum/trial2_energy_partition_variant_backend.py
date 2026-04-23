#!/usr/bin/env python3
"""Audit variant energy-partition ratios on the retained scalar Q-ball branch.

Purpose:
    After the direct-alpha followup family returned to conditional hold, an
    expert suggested reopening the energy-partition branch with one stricter
    blind check: keep the same retained scalar profile, separate cubic and
    quartic interaction energies, and compare a fixed list of dimensionless
    variants against the practical alpha target without introducing any new
    fit parameter.

    The goal is not to tune alpha, but to test whether one physically
    interpretable variant becomes a materially better front runner than the
    previous baseline

        R_int_harm = E_int / E_harm.

Inputs:
    - scripts/quantum/trial2_energy_partition_ratio_backend.py
    - scripts/quantum/scalar_proxy_alpha_q_curve_backend.py

Outputs:
    - One in-memory audit pack consumed by `.5607-.5614` wrappers

Assumptions:
    - No new parameter is introduced
    - The retained beta and the nearest high-beta row stay fixed
    - Variants are screened blindly from the existing on-shell energy split
"""

from __future__ import annotations

import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.scalar_proxy_alpha_q_curve_backend import ALPHA_TARGET
from scripts.quantum.trial2_energy_partition_ratio_backend import (
    build_energy_partition_row,
)


# 関数: retained / near beta rows から one candidate value pair を構成する。
def build_candidate_row(
    candidate_name: str,
    formula: str,
    retained_value: float,
    near_value: float,
    alias_of: str | None = None,
) -> dict:
    """Return one screened candidate row with retained / near diagnostics."""
    retained_value = float(retained_value)
    near_value = float(near_value)
    retained_rel_error = float((retained_value - ALPHA_TARGET) / ALPHA_TARGET)
    near_rel_error = float((near_value - ALPHA_TARGET) / ALPHA_TARGET)
    near_rel_shift = float((near_value - retained_value) / retained_value)
    return {
        "candidate_name": candidate_name,
        "formula": formula,
        "alias_of": alias_of,
        "retained_value": retained_value,
        "retained_rel_error_vs_target": retained_rel_error,
        "retained_abs_rel_error_vs_target": float(abs(retained_rel_error)),
        "near_value": near_value,
        "near_rel_error_vs_target": near_rel_error,
        "near_rel_shift_vs_retained": near_rel_shift,
    }


# 関数: 1 つの energy row から variant family を評価する。

def build_variant_family(row: dict) -> dict[str, float]:
    """Return the fixed variant family on one on-shell energy row."""
    beta = float(row["beta"])
    energy_kinetic = float(row["energy_kinetic"])
    energy_mass = float(row["energy_mass"])
    energy_gradient = float(row["energy_gradient"])
    energy_cubic = float(row["energy_cubic"])
    energy_quartic = float(row["energy_quartic"])
    energy_interaction = float(row["energy_interaction"])
    energy_harmonic = float(row["energy_harmonic"])
    energy_total = float(row["energy_total"])
    epsilon_beta = float(1.0 - beta * beta)

    return {
        "baseline_interaction_over_harmonic": float(energy_interaction / energy_harmonic),
        "variant_1_cubic_over_harmonic": float(energy_cubic / energy_harmonic),
        "variant_2_quartic_over_harmonic": float(energy_quartic / energy_harmonic),
        "variant_3_cubic_over_total": float(energy_cubic / energy_total),
        "variant_4_nonharmonic_over_total": float(
            (energy_interaction + energy_gradient) / energy_total
        ),
        "variant_5_interaction_over_harmonic_plus_gradient": float(
            energy_interaction / (energy_harmonic + energy_gradient)
        ),
        "variant_6_cubic_over_abs_kinetic_mass_gap": float(
            energy_cubic / abs(energy_kinetic - energy_mass)
        ),
        "variant_7_abs_cubic_times_epsilon_beta": float(abs(energy_cubic) * epsilon_beta),
        "variant_8_interaction_total_over_harmonic_sq": float(
            (energy_interaction * energy_total) / (energy_harmonic * energy_harmonic)
        ),
    }


# 関数: retained / near rows から blind variant screening table を組み立てる。

def build_candidate_rows(retained_row: dict, near_row: dict) -> list[dict]:
    """Return the ranked candidate table for the expert-suggested variants."""
    retained_family = build_variant_family(retained_row)
    near_family = build_variant_family(near_row)
    formulas = {
        "baseline_interaction_over_harmonic": "R_base = E_int / E_harm",
        "variant_1_cubic_over_harmonic": "R1 = E_cubic / E_harm",
        "variant_2_quartic_over_harmonic": "R2 = E_quartic / E_harm",
        "variant_3_cubic_over_total": "R3 = E_cubic / E_total",
        "variant_4_nonharmonic_over_total": "R4 = (E_int + E_grad) / E_total",
        "variant_5_interaction_over_harmonic_plus_gradient": (
            "R5 = E_int / (E_harm + E_grad)"
        ),
        "variant_6_cubic_over_abs_kinetic_mass_gap": (
            "R6 = E_cubic / abs(E_kin - E_mass)"
        ),
        "variant_7_abs_cubic_times_epsilon_beta": "R7 = abs(E_cubic) * (1 - beta^2)",
        "variant_8_interaction_total_over_harmonic_sq": (
            "R8 = (E_int / E_harm) * (E_total / E_harm)"
        ),
        "variant_9_interaction_total_over_harmonic_sq_alias": (
            "R9 = E_int * E_total / E_harm^2"
        ),
    }

    candidate_rows = [
        build_candidate_row(
            candidate_name=name,
            formula=formulas[name],
            retained_value=retained_family[name],
            near_value=near_family[name],
        )
        for name in retained_family
    ]
    candidate_rows.append(
        build_candidate_row(
            candidate_name="variant_9_interaction_total_over_harmonic_sq_alias",
            formula=formulas["variant_9_interaction_total_over_harmonic_sq_alias"],
            retained_value=retained_family["variant_8_interaction_total_over_harmonic_sq"],
            near_value=near_family["variant_8_interaction_total_over_harmonic_sq"],
            alias_of="variant_8_interaction_total_over_harmonic_sq",
        )
    )

    return sorted(
        candidate_rows,
        key=lambda row: (
            float(row["retained_abs_rel_error_vs_target"]),
            abs(float(row["near_rel_shift_vs_retained"])),
            str(row["candidate_name"]),
        ),
    )


# 関数: duplicate alias を除いた front-runner / second-runner pair を返す。

def select_unique_front_runners(candidate_rows: list[dict]) -> tuple[dict, dict]:
    """Return the best and second-best unique candidate rows."""
    unique_rows: list[dict] = []
    seen_formula_keys: set[str] = set()
    for row in candidate_rows:
        formula_key = str(row["alias_of"] or row["candidate_name"])
        if formula_key in seen_formula_keys:
            continue

        seen_formula_keys.add(formula_key)
        unique_rows.append(row)

    if len(unique_rows) < 2:
        raise SystemExit("[fail] unique variant front-runner table is too small")

    return dict(unique_rows[0]), dict(unique_rows[1])


# 関数: energy-partition variant screening 全体を official pack に束ねる。

def build_trial2_energy_partition_variant_pack(
    retained_beta: float,
    nearest_beta: float,
) -> dict:
    """Return one blind screening pack for the expert-suggested variants."""
    retained_row = build_energy_partition_row(float(retained_beta))
    near_row = build_energy_partition_row(float(nearest_beta))
    if retained_row is None or near_row is None:
        raise SystemExit("[fail] retained or nearest energy row is unavailable")

    candidate_rows = build_candidate_rows(retained_row, near_row)
    front_runner, second_runner = select_unique_front_runners(candidate_rows)
    baseline_row = next(
        row for row in candidate_rows if row["candidate_name"] == "baseline_interaction_over_harmonic"
    )
    variant_8_row = next(
        row
        for row in candidate_rows
        if row["candidate_name"] == "variant_8_interaction_total_over_harmonic_sq"
    )
    variant_9_row = next(
        row
        for row in candidate_rows
        if row["candidate_name"] == "variant_9_interaction_total_over_harmonic_sq_alias"
    )

    front_runner_improves_baseline_now = bool(
        float(front_runner["retained_abs_rel_error_vs_target"])
        < float(baseline_row["retained_abs_rel_error_vs_target"])
    )
    front_runner_exact_route_available_now = bool(
        float(front_runner["retained_abs_rel_error_vs_target"]) <= 1.0e-12
        and abs(float(front_runner["near_rel_shift_vs_retained"])) <= 1.0e-12
    )
    front_runner_exact_relation_primary_next_now = bool(
        front_runner_improves_baseline_now and not front_runner_exact_route_available_now
    )

    return {
        "alpha_target": float(ALPHA_TARGET),
        "retained_beta1": float(retained_beta),
        "nearest_alpha_beta_root_to_retained": float(nearest_beta),
        "retained_energy_row": retained_row,
        "nearest_energy_row": near_row,
        "candidate_rows": candidate_rows,
        "baseline_row": baseline_row,
        "front_runner": front_runner,
        "second_runner": second_runner,
        "variant_8_row": variant_8_row,
        "variant_9_row": variant_9_row,
        "front_runner_improves_baseline_now": front_runner_improves_baseline_now,
        "front_runner_exact_route_available_now": front_runner_exact_route_available_now,
        "front_runner_exact_relation_primary_next_now": (
            front_runner_exact_relation_primary_next_now
        ),
    }


# 関数: backend 単体実行時に screening summary を表示する。

def main() -> None:
    """Run the energy-partition variant screening directly."""
    pack = build_trial2_energy_partition_variant_pack(
        retained_beta=0.9982557379261291,
        nearest_beta=0.9982996989044647,
    )
    front_runner = dict(pack["front_runner"])
    baseline_row = dict(pack["baseline_row"])
    print("[trial2_energy_partition_variant_backend] front runner:")
    print(f"  {front_runner['candidate_name']} = {front_runner['retained_value']:.15f}")
    print(
        "  rel error vs alpha_target = "
        f"{front_runner['retained_rel_error_vs_target']:.15f}"
    )
    print(
        "  baseline rel error vs alpha_target = "
        f"{baseline_row['retained_rel_error_vs_target']:.15f}"
    )
    print(
        "  near-root rel error vs alpha_target = "
        f"{front_runner['near_rel_error_vs_target']:.15f}"
    )


if __name__ == "__main__":
    main()
