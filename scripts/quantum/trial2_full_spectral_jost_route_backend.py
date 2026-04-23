#!/usr/bin/env python3
"""Audit the genuinely new full spectral / Jost Trial-2 route.

Purpose:
    Build the unique s-wave linearized radial operator implied by the retained
    Mexican-hat Q-ball profile and test whether canonical spectral objects
    select `q_exact` target-free. This route is intentionally distinct from the
    exhausted support-band / distinguished-scale replay:

        1. the operator is the linearized frozen-action radial operator
        2. the exact object is the s-wave phase shift of that operator
        3. the companion Jost-sector object is a Born-level Jost root

Inputs:
    - scripts/quantum/trial2_fresh_pattern_round1_backend.py
    - scripts/quantum/scalar_proxy_alpha_q_curve_backend.py

Outputs:
    - one in-memory diagnostic pack consumed by `.5543-.5550` wrappers

Assumptions:
    - q_exact is already fixed numerically
    - alpha_target is not used as an input parameter
    - the route must remain distinct from the exhausted blind-overlap theorem,
      support-band spectral selector, effective coupling / residue, and Bohr /
      Compton branches
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from scipy.optimize import minimize_scalar


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quantum.scalar_proxy_alpha_q_curve_backend import build_scalar_proxy_alpha_q_curve_pack
from scripts.quantum.trial2_fresh_pattern_round1_backend import load_retained_profile_data


PHASE_Q_MAX = 0.5
BORN_Q_MAX = 0.5


# 関数: retained profile から shifted s-wave potential を返す。
def build_linearized_sw_operator_pack() -> dict:
    """Return the unique retained s-wave operator implied by the frozen EOM."""
    profile_data = load_retained_profile_data()
    alpha_pack = build_scalar_proxy_alpha_q_curve_pack(q_min=0.0, q_max=5.0, q_count=50001)

    radius = np.asarray(profile_data["radius"], dtype=float)
    profile = np.asarray(profile_data["profile"], dtype=float)
    beta1 = float(alpha_pack["scalar_ground_state"]["beta_n"])
    epsilon_beta = float(1.0 - beta1**2)
    threshold_q = float(math.sqrt(epsilon_beta))
    shifted_potential = -(6.0 * profile + 3.0 * np.square(profile))
    full_potential = epsilon_beta + shifted_potential
    negative_part = np.minimum(shifted_potential, 0.0)
    return {
        "radius": radius,
        "profile": profile,
        "beta1": beta1,
        "epsilon_beta": epsilon_beta,
        "threshold_q": threshold_q,
        "shifted_potential": shifted_potential,
        "full_potential": full_potential,
        "potential_min": float(np.min(full_potential)),
        "potential_max": float(np.max(full_potential)),
        "negative_area": float(np.trapezoid(-negative_part, radius)),
        "turning_count": int(np.count_nonzero(np.diff(np.sign(full_potential)) != 0)),
    }


# 関数: shifted operator の Born Jost proxy を返す。

def born_jost_proxy(q_ratio: float, radius: np.ndarray, potential: np.ndarray, epsilon_beta: float) -> complex:
    """Return the Born-level Jost proxy for one physical q value."""
    energy = float(q_ratio**2 - epsilon_beta)
    if energy <= 1.0e-12:
        return complex(np.nan, np.nan)

    wave_number = float(math.sqrt(energy))
    oscillatory_integral = np.trapezoid(
        (np.exp(2.0j * wave_number * radius) - 1.0) * potential,
        radius,
    )
    return 1.0 + oscillatory_integral / (2.0j * wave_number)


# 関数: exact s-wave phase shift を返す。

def phase_shift_exact(
    q_ratio: float,
    radius_start: float,
    radius_end: float,
    potential_of_radius,
    epsilon_beta: float,
) -> float:
    """Return the exact retained s-wave phase shift for one physical q value."""
    energy = float(q_ratio**2 - epsilon_beta)
    if energy <= 1.0e-10:
        return math.nan

    wave_number = float(math.sqrt(energy))

    # 関数: `ode` の入出力契約と処理意図を定義する。
    def ode(radius_value: float, state: list[float]) -> list[float]:
        u_value, up_value = state
        return [
            up_value,
            -(wave_number**2 - float(potential_of_radius(radius_value))) * u_value,
        ]

    solution = solve_ivp(
        ode,
        (float(radius_start), float(radius_end)),
        [float(radius_start), 1.0],
        t_eval=[float(radius_end)],
        rtol=1.0e-8,
        atol=1.0e-10,
        max_step=0.05,
    )
    u_end = float(solution.y[0, -1])
    up_end = float(solution.y[1, -1])
    theta = float(math.atan2(wave_number * u_end, up_end))
    delta = float(theta - wave_number * radius_end)
    while delta <= -math.pi / 2.0:
        delta += math.pi

    while delta > math.pi / 2.0:
        delta -= math.pi

    return delta


# 関数: canonical Born/Jost landmarks を返す。

def build_born_jost_landmarks(operator_pack: dict, q_exact: float) -> dict:
    """Return canonical Born/Jost landmarks for the retained operator."""
    radius = np.asarray(operator_pack["radius"], dtype=float)
    potential = np.asarray(operator_pack["shifted_potential"], dtype=float)
    epsilon_beta = float(operator_pack["epsilon_beta"])
    threshold_q = float(operator_pack["threshold_q"])
    q_values = np.linspace(threshold_q + 1.0e-4, BORN_Q_MAX, 4001)
    jost_values = np.array(
        [
            born_jost_proxy(float(q_value), radius, potential, epsilon_beta)
            for q_value in q_values
        ]
    )
    real_values = np.real(jost_values)
    roots: list[float] = []
    for index in range(len(q_values) - 1):
        left_q = float(q_values[index])
        right_q = float(q_values[index + 1])
        left_value = float(real_values[index])
        right_value = float(real_values[index + 1])
        if math.isnan(left_value) or math.isnan(right_value):
            continue

        if left_value * right_value < 0.0:
            roots.append(
                float(
                    brentq(
                        lambda q_value: float(
                            np.real(
                                born_jost_proxy(
                                    float(q_value),
                                    radius,
                                    potential,
                                    epsilon_beta,
                                )
                            )
                        ),
                        left_q,
                        right_q,
                    )
                )
            )

    born_re_root = float(roots[0]) if roots else math.nan
    born_re_root_rel_error = (
        float((born_re_root - q_exact) / q_exact) if roots else math.nan
    )
    return {
        "born_re_jost_zero_exists_now": bool(roots),
        "born_re_jost_zero_q_over_m0": born_re_root,
        "born_re_jost_zero_rel_error_vs_q_exact": born_re_root_rel_error,
        "born_re_jost_zero_count": int(len(roots)),
    }


# 関数: canonical exact phase landmarks を返す。

def build_exact_phase_landmarks(operator_pack: dict, q_exact: float) -> dict:
    """Return canonical exact phase-shift landmarks for the retained operator."""
    radius = np.asarray(operator_pack["radius"], dtype=float)
    potential = np.asarray(operator_pack["shifted_potential"], dtype=float)
    epsilon_beta = float(operator_pack["epsilon_beta"])
    threshold_q = float(operator_pack["threshold_q"])
    potential_of_radius = interp1d(
        radius,
        potential,
        kind="cubic",
        bounds_error=False,
        fill_value=0.0,
    )

    q_scan = np.linspace(threshold_q + 1.0e-4, PHASE_Q_MAX, 121)
    phase_scan = np.array(
        [
            phase_shift_exact(
                float(q_value),
                float(radius[0]),
                float(radius[-1]),
                potential_of_radius,
                epsilon_beta,
            )
            for q_value in q_scan
        ]
    )
    peak_index = int(np.nanargmax(phase_scan))
    left_q = float(q_scan[max(peak_index - 2, 0)])
    right_q = float(q_scan[min(peak_index + 2, len(q_scan) - 1)])
    peak_optimum = minimize_scalar(
        lambda q_value: -phase_shift_exact(
            float(q_value),
            float(radius[0]),
            float(radius[-1]),
            potential_of_radius,
            epsilon_beta,
        ),
        bounds=(left_q, right_q),
        method="bounded",
        options={"xatol": 1.0e-7},
    )
    phase_peak_q = float(peak_optimum.x)
    phase_peak_value = float(
        phase_shift_exact(
            phase_peak_q,
            float(radius[0]),
            float(radius[-1]),
            potential_of_radius,
            epsilon_beta,
        )
    )
    phase_peak_rel_error = float((phase_peak_q - q_exact) / q_exact)
    phase_derivative = np.gradient(phase_scan, q_scan)
    derivative_peak_index = int(np.nanargmax(np.abs(phase_derivative)))
    derivative_peak_q = float(q_scan[derivative_peak_index])
    derivative_peak_rel_error = float((derivative_peak_q - q_exact) / q_exact)
    return {
        "exact_phase_surface_available_now": True,
        "exact_phase_peak_q_over_m0": phase_peak_q,
        "exact_phase_peak_value": phase_peak_value,
        "exact_phase_peak_rel_error_vs_q_exact": phase_peak_rel_error,
        "exact_phase_derivative_peak_q_over_m0": derivative_peak_q,
        "exact_phase_derivative_peak_rel_error_vs_q_exact": derivative_peak_rel_error,
    }


# 関数: full spectral / Jost audit pack を構築する。

def build_trial2_full_spectral_jost_pack() -> dict:
    """Return the full spectral / Jost diagnostic pack."""
    alpha_pack = build_scalar_proxy_alpha_q_curve_pack(q_min=0.0, q_max=5.0, q_count=50001)
    q_exact = float(alpha_pack["primary_q_exact_over_m0"])
    q_star = float(alpha_pack["q_star_over_m0"])
    q_star_rel_error = float((q_star - q_exact) / q_exact)
    operator_pack = build_linearized_sw_operator_pack()
    born_pack = build_born_jost_landmarks(operator_pack, q_exact)
    phase_pack = build_exact_phase_landmarks(operator_pack, q_exact)

    born_exists = bool(born_pack["born_re_jost_zero_exists_now"])
    exact_phase_surface_available_now = bool(phase_pack["exact_phase_surface_available_now"])
    exact_jost_function_materialized_now = False
    phase_peak_beats_q_star_now = bool(
        abs(float(phase_pack["exact_phase_peak_rel_error_vs_q_exact"]))
        < abs(q_star_rel_error)
    )
    spectral_landmark_nonuniqueness_now = bool(
        born_exists
        and abs(
            float(phase_pack["exact_phase_peak_q_over_m0"])
            - float(born_pack["born_re_jost_zero_q_over_m0"])
        )
        / q_exact
        > 1.0e-2
    )
    target_free_selector_available_now = False
    heuristic_phase_front_runner_only_now = bool(
        exact_phase_surface_available_now
        and not phase_peak_beats_q_star_now
        and not target_free_selector_available_now
    )
    negative_closeout_available_now = bool(
        exact_phase_surface_available_now
        and not target_free_selector_available_now
        and (spectral_landmark_nonuniqueness_now or not born_exists or not exact_jost_function_materialized_now)
    )

    return {
        "q_exact_over_m0": q_exact,
        "q_star_over_m0": q_star,
        "q_star_rel_error_vs_q_exact": q_star_rel_error,
        "beta1": float(operator_pack["beta1"]),
        "epsilon_beta": float(operator_pack["epsilon_beta"]),
        "threshold_q_over_m0": float(operator_pack["threshold_q"]),
        "s_wave_operator_available_now": True,
        "s_wave_potential_min": float(operator_pack["potential_min"]),
        "s_wave_potential_max": float(operator_pack["potential_max"]),
        "s_wave_negative_area": float(operator_pack["negative_area"]),
        "s_wave_turning_count": int(operator_pack["turning_count"]),
        "exact_jost_function_materialized_now": exact_jost_function_materialized_now,
        **born_pack,
        **phase_pack,
        "phase_peak_beats_q_star_now": phase_peak_beats_q_star_now,
        "spectral_landmark_nonuniqueness_now": spectral_landmark_nonuniqueness_now,
        "target_free_selector_available_now": target_free_selector_available_now,
        "heuristic_phase_front_runner_only_now": heuristic_phase_front_runner_only_now,
        "negative_closeout_available_now": negative_closeout_available_now,
    }


# 関数: CLI 実行時に compact summary を返す。

def main() -> None:
    """Run the full spectral / Jost helper directly."""
    import json

    pack = build_trial2_full_spectral_jost_pack()
    summary = {
        "born_re_jost_zero_q_over_m0": pack["born_re_jost_zero_q_over_m0"],
        "born_re_jost_zero_rel_error_vs_q_exact": pack["born_re_jost_zero_rel_error_vs_q_exact"],
        "exact_phase_peak_q_over_m0": pack["exact_phase_peak_q_over_m0"],
        "exact_phase_peak_rel_error_vs_q_exact": pack["exact_phase_peak_rel_error_vs_q_exact"],
        "spectral_landmark_nonuniqueness_now": pack["spectral_landmark_nonuniqueness_now"],
        "negative_closeout_available_now": pack["negative_closeout_available_now"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


# 関数: CLI entrypoint から helper を実行する。

if __name__ == "__main__":
    main()
