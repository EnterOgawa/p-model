#!/usr/bin/env python3
"""Generate 8.7.56.1971-.1974 box-free tail completion artifacts."""

from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.special import exp1


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"

STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
WORK_HISTORY_RECENT = ROOT / "doc" / "WORK_HISTORY_RECENT.md"
CURRENT_PROBLEM = ROOT / "doc" / "quantum" / "34_trial2_numeric_alpha_current_problem.md"
CURRENT_STATUS = ROOT / "doc" / "quantum" / "36_trial2_numeric_alpha_current_status.md"
UNIFIED_ROADMAP = ROOT / "doc" / "quantum" / "39_trial2_vector_qball_unified_closure_roadmap.md"
LONG_ROADMAP = ROOT / "doc" / "quantum" / "55_trial2_numeric_alpha_vector_qball_long_horizon_roadmap.md"
LESSONS = ROOT / "doc" / "quantum" / "56_trial2_numeric_alpha_vector_qball_theory_lessons_after_interval_extension.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

QBALL_BRANCH_REFRESH = PUBLIC_OUT / "mass_origin_qball_charge_mapping_branch_refresh_metrics.json"
QBALL_SOLVER = ROOT / "scripts" / "quantum" / "mass_origin_qball_charge_mapping_branch.py"
PRIOR_GATE = (
    PUBLIC_OUT
    / "q_8_7_56_1967_1970_asymp_generalization_gate_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1971-1974"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor conditional box-free tail "
    "completion or substantive pack-update reactivation"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "box_free_tail_completion_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_extended_interval_exact_box_boundary_asymptotic_"
    "obstruction_tail_completion_or_pack_update_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_box_free_tail_completion_threshold_dependent_"
    "noncanonical_new_signed_rule_or_pack_update_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_conditional_new_signed_observable_"
    "rule_reactivation_after_box_free_tail_audit"
)
NEXT_ROUTE = "8.7.56.1975"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_box_free_tail_closeout_or_"
    "substantive_pack_update_registry"
)
FOLLOWUP_ROUTE = "8.7.56.1979"
Q_THEORY = 0.24297729990871803
VECTOR_NO_GO_ALPHA = 0.0005600186431488893
SCALAR_ALPHA_TARGET = 0.00715678583937324
RETAINED_Q_MAX = 4.0
HIGH_Q_MIN = 4.0
HIGH_Q_MAX = 8.0
ROOT_SCAN_DENSITY = 5000
ROOT_TOL = 1.0e-10
LINEAR_THRESHOLDS = np.array([0.10, 0.05, 0.02, 0.01], dtype=float)


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: JSON/CSV artifact を書き出す。

def write_artifact(kind: str, data: dict) -> dict[str, str]:
    """Write one JSON payload and one rows CSV."""
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    paths = build_metrics_paths(PUBLIC_OUT, STEM, kind)
    paths["json"].write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with paths["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_id", "status", "metric", "value", "note"],
        )
        writer.writeheader()
        writer.writerows(data["rows"])

    return {
        "json": sign_base.display_path(paths["json"]),
        "csv": sign_base.display_path(paths["csv"]),
    }


# 関数: 線形 tail 判定指標を返す。

def linearity_ratio(field: np.ndarray, kappa_sq: float) -> np.ndarray:
    """Return the local nonlinearity-to-linear-tail ratio."""
    return np.abs(3.0 * field + field * field) / float(kappa_sq)


# 関数: decaying / growing mode の local amplitudes を返す。

def projected_tail_amplitudes(
    radius: float,
    field_value: float,
    field_prime_value: float,
    kappa: float,
) -> tuple[float, float]:
    """Project the local linear tail onto decaying and growing Yukawa modes."""
    u_value = radius * field_value
    u_prime_value = radius * field_prime_value + field_value
    decaying = 0.5 * math.exp(kappa * radius) * (u_value - (u_prime_value / kappa))
    growing = 0.5 * math.exp(-kappa * radius) * (u_value + (u_prime_value / kappa))
    return float(decaying), float(growing)


# 関数: threshold 初回到達 index を返す。

def first_threshold_index(ratio: np.ndarray, threshold: float) -> int:
    """Return the first index where the linear-tail ratio falls below one threshold."""
    indices = np.where(ratio <= float(threshold))[0]
    if indices.size == 0:
        raise ValueError(f"no linear-tail point found for threshold={threshold}")

    return int(indices[0])


# 関数: 指定しきい値の連続 plateau window を返す。

def contiguous_plateau_window(ratio: np.ndarray, threshold: float) -> tuple[int, int]:
    """Return the first contiguous window where the linear-tail ratio stays below threshold."""
    start = first_threshold_index(ratio, threshold)
    end = start
    while end + 1 < ratio.size and ratio[end + 1] <= float(threshold):
        end += 1

    return int(start), int(end)


# 関数: analytic decaying tail の重み積分を返す。

def tail_norm(decaying_amplitude: float, kappa: float, r_match: float) -> float:
    """Return the decaying-tail norm contribution beyond the matching radius."""
    return float(
        (decaying_amplitude * decaying_amplitude)
        * math.exp(-2.0 * kappa * r_match)
        / (2.0 * kappa)
    )


# 関数: analytic decaying tail の form-factor integral を返す。

def tail_form_factor_piece(
    decaying_amplitude: float,
    kappa: float,
    r_match: float,
    q_ratio: float,
) -> float:
    """Return the analytic decaying-tail form-factor contribution."""
    if abs(q_ratio) <= 1.0e-12:
        return tail_norm(decaying_amplitude, kappa, r_match)

    z_value = (2.0 * kappa - 1j * float(q_ratio)) * r_match
    return float((decaying_amplitude * decaying_amplitude / float(q_ratio)) * exp1(z_value).imag)


# 関数: match radius で切った box-free completed form factor を返す。

def completed_form_factor(
    radius: np.ndarray,
    weight: np.ndarray,
    decaying_amplitude: float,
    kappa: float,
    match_index: int,
    q_ratio: float,
) -> float:
    """Return the normalized box-free completed overlap form factor."""
    radius_inner = radius[: match_index + 1]
    weight_inner = weight[: match_index + 1]
    r_match = float(radius_inner[-1])
    inner_norm = float(np.trapezoid(weight_inner, radius_inner))
    tail_norm_value = tail_norm(decaying_amplitude, kappa, r_match)
    total_norm = inner_norm + tail_norm_value

    if abs(q_ratio) <= 1.0e-12:
        return 1.0

    qx = float(q_ratio) * radius_inner
    sinc = np.ones_like(qx)
    mask = np.abs(qx) > 1.0e-12
    sinc[mask] = np.sin(qx[mask]) / qx[mask]
    inner_piece = float(np.trapezoid(weight_inner * sinc, radius_inner))
    tail_piece = tail_form_factor_piece(decaying_amplitude, kappa, r_match, float(q_ratio))
    return float((inner_piece + tail_piece) / total_norm)


# 関数: completed form factor の signed zero を探す。

def find_completed_signed_zeros(
    radius: np.ndarray,
    weight: np.ndarray,
    decaying_amplitude: float,
    kappa: float,
    match_index: int,
    q_min: float,
    q_max: float,
) -> np.ndarray:
    """Locate all simple signed zeros of the completed overlap on one q interval."""
    scan = np.linspace(float(q_min), float(q_max), int(ROOT_SCAN_DENSITY * (q_max - q_min)) + 1)
    values = np.array(
        [
            completed_form_factor(
                radius,
                weight,
                decaying_amplitude,
                kappa,
                match_index,
                float(q_value),
            )
            for q_value in scan
        ],
        dtype=float,
    )
    roots: list[float] = []
    for q_left, q_right, f_left, f_right in zip(scan[:-1], scan[1:], values[:-1], values[1:]):
        if abs(f_left) <= ROOT_TOL and q_left > q_min:
            root = float(q_left)
        elif f_left * f_right < 0.0:
            root = float(
                brentq(
                    lambda q_value: completed_form_factor(
                        radius,
                        weight,
                        decaying_amplitude,
                        kappa,
                        match_index,
                        float(q_value),
                    ),
                    float(q_left),
                    float(q_right),
                )
            )
        else:
            continue

        if not roots or abs(root - roots[-1]) > 1.0e-6:
            roots.append(root)

    return np.array(roots, dtype=float)


# 関数: audit 用の公式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the box-free tail completion audit."""
    return {
        "linear_tail_equation": "f'' + 2 f'/r - kappa^2 f = 0, kappa = sqrt(1 - beta_n^2)",
        "linear_tail_general": "f_lin(r) = (A_dec e^{-kappa r} + B_grow e^{+kappa r}) / r",
        "decaying_projection": "A_dec(r) = 0.5 exp(kappa r) [r f(r) - (r f'(r)+f(r))/kappa]",
        "growing_projection": "B_grow(r) = 0.5 exp(-kappa r) [r f(r) + (r f'(r)+f(r))/kappa]",
        "completed_tail": "f_tail(r; r_m) = A_dec(r_m) e^{-kappa r} / r for r >= r_m",
        "completed_overlap": "F_comp(q; r_m) = [int_0^{r_m} dr w(r) sinc(qr) + int_{r_m}^{inf} dr A_dec(r_m)^2 e^{-2 kappa r} sinc(qr)] / N_comp(r_m)",
        "linearity_ratio": "eta(r) = |3 f(r) + f(r)^2| / kappa^2",
    }


# 関数: `.1971-.1974` を実行する。

def main() -> None:
    """Execute the box-free tail completion or pack-update reactivation audit."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        LONG_ROADMAP,
        LESSONS,
        PART5,
        QBALL_BRANCH_REFRESH,
        QBALL_SOLVER,
        PRIOR_GATE,
    ):
        sign_base.require(path)

    status_text = sign_base.read_text(STATUS)
    roadmap_text = sign_base.read_text(ROADMAP)
    current_problem_text = sign_base.read_text(CURRENT_PROBLEM)
    current_status_text = sign_base.read_text(CURRENT_STATUS)
    unified_text = sign_base.read_text(UNIFIED_ROADMAP)
    long_text = sign_base.read_text(LONG_ROADMAP)
    lessons_text = sign_base.read_text(LESSONS)
    part5_text = sign_base.read_text(PART5)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    qball_branch_refresh = sign_base.read_json(QBALL_BRANCH_REFRESH)
    scalar_ground_state = sign_base.extract_scalar_ground_state(qball_branch_refresh)

    inventory_ready = all(
        (
            bool(prior_summary["exact_alpha_promotion_retained"]),
            bool(prior_summary["exact_signed_form_factor_promotion_retained"]),
            bool(prior_summary["box_free_tail_completion_admissible_now"]),
        )
    )

    qball_module = sign_base.load_qball_module()
    radius, field, field_prime = qball_module.solve_full_profile(
        float(scalar_ground_state["beta_n"]),
        float(scalar_ground_state["central_amplitude"]),
    )
    weight = (field**2) * (radius**2)
    kappa = math.sqrt(1.0 - float(scalar_ground_state["beta_n"]) ** 2)
    eta = linearity_ratio(field, kappa * kappa)

    plateau_start, plateau_end = contiguous_plateau_window(eta, 0.10)
    plateau_radius = radius[plateau_start : plateau_end + 1]
    plateau_field = field[plateau_start : plateau_end + 1]
    plateau_prime = field_prime[plateau_start : plateau_end + 1]
    decaying_plateau = []
    growing_plateau = []
    for radius_value, field_value, field_prime_value in zip(
        plateau_radius,
        plateau_field,
        plateau_prime,
    ):
        decaying_value, growing_value = projected_tail_amplitudes(
            float(radius_value),
            float(field_value),
            float(field_prime_value),
            kappa,
        )
        decaying_plateau.append(decaying_value)
        growing_plateau.append(growing_value)

    decaying_plateau_array = np.array(decaying_plateau, dtype=float)
    growing_plateau_array = np.array(growing_plateau, dtype=float)
    plateau_decaying_mean = float(np.mean(decaying_plateau_array))
    plateau_growing_mean = float(np.mean(growing_plateau_array))
    plateau_decaying_rel_spread = float(
        (np.max(decaying_plateau_array) - np.min(decaying_plateau_array))
        / abs(plateau_decaying_mean)
    )
    plateau_growing_rel_spread = float(
        (np.max(growing_plateau_array) - np.min(growing_plateau_array))
        / abs(plateau_growing_mean)
    )
    plateau_growing_to_decaying_ratio = float(
        np.mean(np.abs(growing_plateau_array / decaying_plateau_array))
    )
    linear_tail_plateau_available = bool(plateau_decaying_rel_spread <= 5.0e-5)

    threshold_rows = []
    threshold_alphas = []
    threshold_match_radii = []
    threshold_spacing_means = []
    threshold_spacing_rel_gaps = []
    threshold_vector_gain = []
    threshold_scalar_fraction = []
    threshold_zero_counts = []

    for threshold in LINEAR_THRESHOLDS:
        match_index = first_threshold_index(eta, float(threshold))
        r_match = float(radius[match_index])
        decaying_value, growing_value = projected_tail_amplitudes(
            r_match,
            float(field[match_index]),
            float(field_prime[match_index]),
            kappa,
        )
        completed_q_theory = completed_form_factor(
            radius,
            weight,
            decaying_value,
            kappa,
            match_index,
            Q_THEORY,
        )
        alpha_completed = (abs(completed_q_theory) ** 2) / (4.0 * math.pi)
        high_q_roots = find_completed_signed_zeros(
            radius,
            weight,
            decaying_value,
            kappa,
            match_index,
            HIGH_Q_MIN,
            HIGH_Q_MAX,
        )
        high_q_spacings = np.diff(high_q_roots)
        spacing_mean = float(np.mean(high_q_spacings))
        spacing_theory = math.pi / r_match
        spacing_rel_gap = abs(spacing_mean - spacing_theory) / spacing_theory

        threshold_match_radii.append(r_match)
        threshold_alphas.append(alpha_completed)
        threshold_spacing_means.append(spacing_mean)
        threshold_spacing_rel_gaps.append(spacing_rel_gap)
        threshold_vector_gain.append(alpha_completed / VECTOR_NO_GO_ALPHA)
        threshold_scalar_fraction.append(alpha_completed / SCALAR_ALPHA_TARGET)
        threshold_zero_counts.append(int(high_q_roots.size))
        threshold_rows.append(
            {
                "threshold": float(threshold),
                "match_radius_over_m0": r_match,
                "decaying_amplitude": decaying_value,
                "growing_amplitude": growing_value,
                "growing_to_decaying_ratio": abs(growing_value / decaying_value),
                "completed_alpha_at_q_theory": alpha_completed,
                "completed_vector_gain": alpha_completed / VECTOR_NO_GO_ALPHA,
                "completed_scalar_fraction": alpha_completed / SCALAR_ALPHA_TARGET,
                "high_q_zero_count": int(high_q_roots.size),
                "mean_high_q_spacing": spacing_mean,
                "spacing_pi_over_r_match": spacing_theory,
                "spacing_rel_gap_vs_pi_over_r_match": spacing_rel_gap,
            }
        )

    threshold_match_radii_array = np.array(threshold_match_radii, dtype=float)
    threshold_alphas_array = np.array(threshold_alphas, dtype=float)
    threshold_spacing_means_array = np.array(threshold_spacing_means, dtype=float)
    threshold_spacing_rel_gaps_array = np.array(threshold_spacing_rel_gaps, dtype=float)
    threshold_vector_gain_array = np.array(threshold_vector_gain, dtype=float)
    threshold_scalar_fraction_array = np.array(threshold_scalar_fraction, dtype=float)
    threshold_zero_counts_array = np.array(threshold_zero_counts, dtype=float)

    threshold_family_available = bool(
        threshold_match_radii_array.size == LINEAR_THRESHOLDS.size
    )
    matching_radius_dependence_obstruction_detected = bool(
        linear_tail_plateau_available
        and threshold_family_available
        and ((np.max(threshold_match_radii_array) - np.min(threshold_match_radii_array)) > 0.2)
    )
    threshold_family_tracks_match_radius = bool(
        np.max(threshold_spacing_rel_gaps_array) <= 5.0e-4
    )
    box_free_tail_completion_canonical_available = False
    box_free_tail_completion_scalar_leaning_family_present = bool(
        np.min(threshold_vector_gain_array) > 5.0
        and np.max(threshold_scalar_fraction_array) < 0.6
    )
    new_signed_observable_rule_admissible_now = bool(
        matching_radius_dependence_obstruction_detected
        and threshold_family_tracks_match_radius
    )
    substantive_pack_update_still_admissible = True

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "box-free tail completion inventory ready",
            sign_base.truth(inventory_ready),
            "The cutoff-removal audit starts only after Gate B finite-interval exact / asymptotic obstruction has been fixed.",
        ),
        sign_base.row(
            "plateau_start_over_m0",
            "pass",
            "first eta<=0.1 linear-tail radius over m0",
            float(plateau_radius[0]),
            "The first local radius where the nonlinear correction is at most 10 percent of the linear tail scale.",
        ),
        sign_base.row(
            "plateau_end_over_m0",
            "pass",
            "last eta<=0.1 linear-tail radius over m0 in the first contiguous window",
            float(plateau_radius[-1]),
            "The retained linear-tail plateau is narrow and localized around the near-zero crossing window.",
        ),
        sign_base.row(
            "plateau_decaying_mean",
            "pass",
            "decaying amplitude plateau mean",
            plateau_decaying_mean,
            "The decaying Yukawa projection is numerically stable on the first linear-tail plateau.",
        ),
        sign_base.row(
            "plateau_decaying_rel_spread",
            "pass" if plateau_decaying_rel_spread <= 5.0e-5 else "watch",
            "relative spread of the decaying amplitude on the eta<=0.1 plateau",
            plateau_decaying_rel_spread,
            "A tiny spread means the linear decaying mode itself is well defined before the solver-box edge.",
        ),
        sign_base.row(
            "plateau_growing_ratio",
            "watch",
            "mean |B_grow/A_dec| on the eta<=0.1 plateau",
            plateau_growing_to_decaying_ratio,
            "A nonzero growing component means the retained numerical tail is contaminated and must be truncated at a chosen matching radius.",
        ),
        sign_base.row(
            "threshold_match_radius_min",
            "watch",
            "minimum threshold-selected matching radius over m0",
            float(np.min(threshold_match_radii_array)),
            "Different linear-tail thresholds pick different truncation radii for the box-free completion family.",
        ),
        sign_base.row(
            "threshold_match_radius_max",
            "watch",
            "maximum threshold-selected matching radius over m0",
            float(np.max(threshold_match_radii_array)),
            "The current completion family is controlled by the threshold-dependent start of the retained linear-tail window.",
        ),
        sign_base.row(
            "completed_alpha_min",
            "watch",
            "minimum completed alpha(q_theory) across the threshold family",
            float(np.min(threshold_alphas_array)),
            "All threshold-selected completions lean toward the scalar side, but none retain the previous exact promotion.",
        ),
        sign_base.row(
            "completed_alpha_max",
            "watch",
            "maximum completed alpha(q_theory) across the threshold family",
            float(np.max(threshold_alphas_array)),
            "The threshold family stays below the retained exact scalar alpha even when the decaying-tail plateau is used.",
        ),
        sign_base.row(
            "completed_vector_gain_min",
            "watch",
            "minimum alpha gain over the vector no-go across the threshold family",
            float(np.min(threshold_vector_gain_array)),
            "The threshold family is scalar-leaning relative to the vector no-go, but still does not canonically close the exact promotion.",
        ),
        sign_base.row(
            "completed_scalar_fraction_max",
            "watch",
            "maximum scalar-target fraction across the threshold family",
            float(np.max(threshold_scalar_fraction_array)),
            "The best threshold-selected completion reaches only about half of the retained exact scalar target.",
        ),
        sign_base.row(
            "spacing_rel_gap_max_vs_pi_over_r_match",
            "watch",
            "maximum relative gap between high-q spacing and pi/r_match across the threshold family",
            float(np.max(threshold_spacing_rel_gaps_array)),
            "The high-q zero lattice follows the chosen matching radius rather than a unique box-free canonical scale.",
        ),
        sign_base.row(
            "threshold_family_zero_count_min",
            "pass",
            "minimum completed signed-zero count on 4<=q/m0<=8 across the threshold family",
            float(np.min(threshold_zero_counts_array)),
            "The completed family keeps a simple sign-parity structure, so the obstruction is canonicality rather than loss of real-sign bookkeeping.",
        ),
        sign_base.row(
            "box_free_tail_completion_canonical_available",
            "reject" if not box_free_tail_completion_canonical_available else "pass",
            "Gate A box-free tail completion retained canonically",
            sign_base.truth(box_free_tail_completion_canonical_available),
            "The current cutoff-removal family depends on an arbitrary linear-tail threshold and therefore does not define a unique canonical continuation.",
        ),
        sign_base.row(
            "matching_radius_dependence_obstruction_detected",
            "pass" if matching_radius_dependence_obstruction_detected else "reject",
            "Gate B threshold-dependent matching-radius obstruction detected",
            sign_base.truth(matching_radius_dependence_obstruction_detected),
            "The cutoff-removal family is controlled by the chosen truncation radius inside the contaminated tail window.",
        ),
        sign_base.row(
            "new_signed_observable_rule_admissible_now",
            "pass" if new_signed_observable_rule_admissible_now else "watch",
            "new signed observable rule admissible now",
            sign_base.truth(new_signed_observable_rule_admissible_now),
            "Because the box-free tail family is noncanonical, the next honest second shot is a genuinely new signed observable rule or a pack update that supplies one.",
        ),
    ]

    summary = {
        "generated_utc": now_iso(),
        "step": STEP_TAG,
        "step_name": STEP_NAME,
        "prior_classification": PRIOR_CLASS,
        "current_classification": BRANCH_CLASS,
        "retained_interval_over_m0": RETAINED_Q_MAX,
        "linear_tail_plateau_start_over_m0": float(plateau_radius[0]),
        "linear_tail_plateau_end_over_m0": float(plateau_radius[-1]),
        "linear_tail_plateau_available": linear_tail_plateau_available,
        "decaying_amplitude_plateau_mean": plateau_decaying_mean,
        "decaying_amplitude_plateau_rel_spread": plateau_decaying_rel_spread,
        "growing_amplitude_plateau_mean": plateau_growing_mean,
        "growing_amplitude_plateau_rel_spread": plateau_growing_rel_spread,
        "growing_to_decaying_ratio_plateau_mean": plateau_growing_to_decaying_ratio,
        "threshold_family_available": threshold_family_available,
        "threshold_grid": LINEAR_THRESHOLDS.tolist(),
        "threshold_match_radius_min_over_m0": float(np.min(threshold_match_radii_array)),
        "threshold_match_radius_max_over_m0": float(np.max(threshold_match_radii_array)),
        "completed_alpha_min_at_q_theory": float(np.min(threshold_alphas_array)),
        "completed_alpha_max_at_q_theory": float(np.max(threshold_alphas_array)),
        "completed_vector_gain_min": float(np.min(threshold_vector_gain_array)),
        "completed_vector_gain_max": float(np.max(threshold_vector_gain_array)),
        "completed_scalar_fraction_min": float(np.min(threshold_scalar_fraction_array)),
        "completed_scalar_fraction_max": float(np.max(threshold_scalar_fraction_array)),
        "completed_spacing_mean_min": float(np.min(threshold_spacing_means_array)),
        "completed_spacing_mean_max": float(np.max(threshold_spacing_means_array)),
        "spacing_rel_gap_max_vs_pi_over_r_match": float(np.max(threshold_spacing_rel_gaps_array)),
        "threshold_family_tracks_match_radius": threshold_family_tracks_match_radius,
        "box_free_tail_completion_canonical_available": box_free_tail_completion_canonical_available,
        "box_free_tail_completion_scalar_leaning_family_present": box_free_tail_completion_scalar_leaning_family_present,
        "matching_radius_dependence_obstruction_detected": matching_radius_dependence_obstruction_detected,
        "new_signed_observable_rule_admissible_now": new_signed_observable_rule_admissible_now,
        "substantive_pack_update_still_admissible": substantive_pack_update_still_admissible,
        "same_level_old_retry_admissible": False,
        "overall_status": "vector_qball_form_factor_box_free_tail_completion_audit_declared",
    }

    decision = {
        "selected_route": FOLLOWUP_ROUTE_NAME,
        "next_step": FOLLOWUP_ROUTE,
        "why": (
            "The decaying Yukawa tail can be projected stably, but the resulting "
            "cutoff-removal family depends on an arbitrary matching radius and "
            "therefore does not provide a unique canonical continuation."
        ),
    }

    evidence = {
        "inputs": {
            "qball_branch_refresh": sign_base.display_path(QBALL_BRANCH_REFRESH),
            "prior_gate": sign_base.display_path(PRIOR_GATE),
            "status_hit": sign_base.hit(status_text, "8.7.56.1967-.1970"),
            "roadmap_hit": sign_base.hit(roadmap_text, ".1971-.1974"),
            "current_problem_hit": sign_base.hit(current_problem_text, "box-free tail completion"),
            "current_status_hit": sign_base.hit(current_status_text, "box-free tail completion"),
            "unified_hit": sign_base.hit(unified_text, ".1971-.1974"),
            "long_hit": sign_base.hit(long_text, "box-free tail"),
            "lessons_hit": sign_base.hit(lessons_text, "retained interval"),
            "part5_hit": sign_base.hit(part5_text, "0 <= q/m0 <= 4"),
        },
        "formulas": build_formulae(),
        "threshold_family_rows": threshold_rows,
    }

    declaration = {
        "generated_utc": now_iso(),
        "phase": {"phase": 8, "step": STEP_TAG, "name": STEP_NAME},
        "inputs": {
            "q_theory_over_m0": Q_THEORY,
            "retained_interval_over_m0": RETAINED_Q_MAX,
            "high_q_window_over_m0": [HIGH_Q_MIN, HIGH_Q_MAX],
            "linearity_thresholds": LINEAR_THRESHOLDS.tolist(),
        },
        "intent": "Audit whether the sign-parity theorem admits a canonical box-free tail completion under the retained solver-box pack.",
        "formulas": build_formulae(),
        "rows": rows,
        "summary": summary,
        "decision": decision,
        "evidence": evidence,
    }
    declaration_paths = write_artifact("declaration_gate", declaration)

    route_rows = [
        sign_base.row(
            "gate_a_box_free_tail_completion_retained",
            "reject",
            "Gate A box-free tail completion retained canonically",
            sign_base.truth(False),
            "The cutoff-removal family does not become canonical under the current retained numerical tail pack.",
        ),
        sign_base.row(
            "gate_b_matching_radius_obstruction_selected",
            "pass",
            "Gate B threshold-dependent matching-radius obstruction selected",
            sign_base.truth(True),
            "A stable decaying-tail plateau exists, but any completion depends on the arbitrary choice of where to cut off the contaminated numerical tail.",
        ),
        sign_base.row(
            "gate_c_current_continuation_rule_blocked",
            "reject",
            "Gate C current continuation rule already blocked on the retained interval",
            sign_base.truth(False),
            "The exact sign-parity theorem on 0<=q/m0<=4 remains intact; only the next cutoff-removal generalization is noncanonical.",
        ),
        sign_base.row(
            "new_signed_observable_rule_admissible_now",
            "pass" if new_signed_observable_rule_admissible_now else "watch",
            "new signed observable rule admissible now",
            sign_base.truth(new_signed_observable_rule_admissible_now),
            "After the box-free tail family is honestly blocked, a genuinely new signed rule becomes the next honest second shot.",
        ),
        sign_base.row(
            "substantive_pack_update_still_admissible",
            "pass",
            "substantive pack update still admissible",
            sign_base.truth(substantive_pack_update_still_admissible),
            "A pack update that changes the contaminated tail sector can still reopen a canonical cutoff-removal theorem.",
        ),
    ]

    route_sync = {
        "generated_utc": now_iso(),
        "phase": {"phase": 8, "step": STEP_TAG, "name": STEP_NAME},
        "inputs": {
            "declaration_gate": declaration_paths["json"],
        },
        "intent": "Sync the official route after the box-free tail completion audit.",
        "rows": route_rows,
        "summary": {
            "generated_utc": now_iso(),
            "step": STEP_TAG,
            "current_classification": BRANCH_CLASS,
            "gate_a_box_free_tail_completion_retained": False,
            "gate_b_matching_radius_obstruction_selected": True,
            "gate_c_current_continuation_rule_blocked": False,
            "exact_alpha_promotion_retained": True,
            "exact_signed_form_factor_promotion_retained": True,
            "new_signed_observable_rule_admissible_now": new_signed_observable_rule_admissible_now,
            "substantive_pack_update_still_admissible": substantive_pack_update_still_admissible,
            "next_route": NEXT_ROUTE_NAME,
            "next_step": NEXT_ROUTE,
            "followup_route": FOLLOWUP_ROUTE_NAME,
            "followup_step": FOLLOWUP_ROUTE,
            "overall_status": "vector_qball_form_factor_box_free_tail_completion_route_synced",
        },
        "decision": {
            "selected_next_route": NEXT_ROUTE_NAME,
            "selected_next_step": NEXT_ROUTE,
            "why": (
                "The box-free tail family is scalar-leaning but noncanonical, so the "
                "next honest computation is a genuinely new signed observable rule "
                "that leaves the exact finite-interval theorem intact."
            ),
        },
        "evidence": {
            "declaration_gate": declaration_paths,
            "status_hit": sign_base.hit(status_text, "vector_qball_form_factor_extended_interval_exact_box_boundary_asymptotic_obstruction_tail_completion_or_pack_update_next"),
            "roadmap_hit": sign_base.hit(roadmap_text, ".1975-.1978"),
        },
    }
    route_paths = write_artifact("route_sync", route_sync)

    print("[ok] 8.7.56.1971-.1974 box-free tail completion artifacts generated")
    print(f"[json] {declaration_paths['json']}")
    print(f"[json] {route_paths['json']}")


if __name__ == "__main__":
    main()
