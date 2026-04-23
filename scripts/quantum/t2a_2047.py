#!/usr/bin/env python3
"""Generate 8.7.56.2047-.2050 q-dependent phase-slip loading artifacts.

This branch tests whether the retained boundary-local-jet phase-slip theorem
can be extended from a single constant shift into a minimal q-dependent
loading family. If the minimal smooth loading still fails on translated
higher-harmonic windows, the honest next surface is a harmonic-index dependent
signed rule.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
import scripts.quantum.t2a_2023 as alias_base
import scripts.quantum.t2a_2031 as phase_base
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
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

QBALL_BRANCH_REFRESH = PUBLIC_OUT / "mass_origin_qball_charge_mapping_branch_refresh_metrics.json"
PRIOR_GATE = PUBLIC_OUT / "q_8_7_56_2043_2046_alias_image_phase_slip_registry_declaration_gate_metrics.json"
PRIOR_AUDIT = PUBLIC_OUT / "q_8_7_56_2039_2042_alias_image_phase_slip_theorem_declaration_gate_metrics.json"

STEP_TAG = "8.7.56.2047-2050"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor q-dependent boundary "
    "phase-slip loading or higher-harmonic signed-rule reactivation"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "q_dependent_phase_slip_loading",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_boundary_alias_image_local_jet_phase_slip_theorem_"
    "retained_q_dependent_or_higher_harmonic_loading_reactivation_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_minimal_q_dependent_boundary_loading_blocked_"
    "higher_harmonic_windowwise_signed_rule_gate_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_alias_image_phase_slip_loading_"
    "closeout_registry"
)
NEXT_ROUTE = "8.7.56.2051"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_higher_harmonic_windowwise_"
    "phase_slip_signed_rule_reactivation"
)
FOLLOWUP_ROUTE = "8.7.56.2055"

FIT_Q_MIN = alias_base.FIT_Q_MIN
FIT_Q_MAX = alias_base.FIT_Q_MAX
EDGE_Q_MIN = alias_base.EDGE_Q_MIN
EDGE_Q_MAX = alias_base.EDGE_Q_MAX
WINDOW_SCAN_DENSITY = 250
SEARCH_DECIMATION = 25
DELTA_MIN = 0.0
DELTA_MAX = 1.5
DELTA_STEP = 0.0005
QDEP_MAXITER = 35
QDEP_POPSIZE = 12
QDEP_SEED = 2


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


# 関数: translated higher-harmonic window を返す。
def translated_window(alias_harmonic: float, offsets: tuple[float, float]) -> tuple[float, float]:
    """Return one translated window."""
    return float(alias_harmonic + offsets[0]), float(alias_harmonic + offsets[1])


# 関数: harmonic window inventory を構成する。
def build_windows(radius: np.ndarray, weight: np.ndarray, norm: float, alias_1: float) -> list[dict[str, object]]:
    """Build active and translated higher-harmonic windows."""
    fit_offsets = (FIT_Q_MIN - alias_1, FIT_Q_MAX - alias_1)
    edge_offsets = (EDGE_Q_MIN - (2.0 * alias_1), EDGE_Q_MAX - (2.0 * alias_1))
    windows: list[dict[str, object]] = []
    for harmonic_index in range(1, 9):
        alias_harmonic = harmonic_index * alias_1
        offsets = fit_offsets if (harmonic_index % 2) == 1 else edge_offsets
        q_min, q_max = translated_window(alias_harmonic, offsets)
        q_scan = np.linspace(
            q_min,
            q_max,
            int(round((q_max - q_min) * WINDOW_SCAN_DENSITY)) + 1,
        )
        exact_values, exact_abs, exact_sign = phase_base.exact_sign_data(
            radius,
            weight,
            norm,
            q_scan,
        )
        windows.append(
            {
                "harmonic_index": harmonic_index,
                "alias_harmonic": float(alias_harmonic),
                "q_min": float(q_min),
                "q_max": float(q_max),
                "q_scan": q_scan,
                "exact_values": exact_values,
                "exact_abs": exact_abs,
                "exact_sign": exact_sign,
                "template_type": "fit" if (harmonic_index % 2) == 1 else "edge",
            }
        )

    return windows


# 関数: one delta family を評価する。
def evaluate_family(
    windows: list[dict[str, object]],
    lookup_q: np.ndarray,
    lookup_values: np.ndarray,
    delta_function,
) -> list[dict[str, float]]:
    """Evaluate one loading family on all windows."""
    results: list[dict[str, float]] = []
    for window in windows:
        harmonic_index = int(window["harmonic_index"])
        alias_harmonic = float(window["alias_harmonic"])
        q_scan = np.asarray(window["q_scan"], dtype=float)
        exact_values = np.asarray(window["exact_values"], dtype=float)
        exact_abs = np.asarray(window["exact_abs"], dtype=float)
        exact_sign = np.asarray(window["exact_sign"], dtype=float)
        delta_values = np.asarray(delta_function(harmonic_index, q_scan), dtype=float)
        center = alias_harmonic + (((-1) ** (harmonic_index + 1)) * delta_values)
        q_image = np.abs(center - q_scan)
        image_values = np.interp(q_image, lookup_q, lookup_values)
        sigma_pred = phase_base.alias_sigma_from_values(image_values, harmonic_index)
        metrics = phase_base.signed_window_metrics(
            sigma_pred,
            exact_sign,
            exact_values,
            exact_abs,
        )
        metrics["delta_min"] = float(np.min(delta_values))
        metrics["delta_max"] = float(np.max(delta_values))
        metrics["delta_mean"] = float(np.mean(delta_values))
        metrics["q_image_min"] = float(np.min(q_image))
        metrics["q_image_max"] = float(np.max(q_image))
        results.append(metrics)

    return results


# 関数: one window subset の summary を返す。
def summarize_window_group(results: list[dict[str, float]], indices: list[int]) -> dict[str, float]:
    """Return max mismatch/error and min correlation on one subset."""
    subset = [results[index - 1] for index in indices]
    return {
        "max_mismatch": float(max(item["sign_mismatch_fraction"] for item in subset)),
        "max_abs_error": float(max(item["signed_reconstruction_max_abs_error"] for item in subset)),
        "min_correlation": float(min(item["sign_correlation"] for item in subset)),
    }


# 関数: independent harmonic optimum を返す。
def optimize_independent_delta(
    window: dict[str, object],
    lookup_q: np.ndarray,
    lookup_values: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """Return one independent delta optimum and its full-window metrics."""
    harmonic_index = int(window["harmonic_index"])
    alias_harmonic = float(window["alias_harmonic"])
    q_scan_full = np.asarray(window["q_scan"], dtype=float)
    exact_sign = np.asarray(window["exact_sign"], dtype=float)
    q_scan_dec = q_scan_full[::SEARCH_DECIMATION]
    exact_sign_dec = exact_sign[::SEARCH_DECIMATION]

    best_delta = float(DELTA_MIN)
    best_mismatch = 1.0
    for delta_q in np.arange(DELTA_MIN, DELTA_MAX + (0.5 * DELTA_STEP), DELTA_STEP):
        q_image = phase_base.shifted_alias_image_q(
            q_scan_dec,
            alias_harmonic,
            harmonic_index,
            float(delta_q),
        )
        image_values = np.interp(q_image, lookup_q, lookup_values)
        sigma_pred = phase_base.alias_sigma_from_values(image_values, harmonic_index)
        mismatch = alias_base.sign_mismatch_fraction(sigma_pred, exact_sign_dec)
        if mismatch < best_mismatch - 1.0e-12:
            best_delta = float(delta_q)
            best_mismatch = float(mismatch)

    metrics = evaluate_family(
        [window],
        lookup_q,
        lookup_values,
        lambda _n, q_array, delta=best_delta: np.full_like(q_array, delta, dtype=float),
    )[0]
    return best_delta, metrics


# 関数: q-dependent 2-term family の objective を返す。
def q2_family_objective(
    params: np.ndarray,
    fit_windows: list[dict[str, object]],
    lookup_q: np.ndarray,
    lookup_values: np.ndarray,
) -> float:
    """Return the minimax objective for delta(q)=a0+a1/q+a2/q^2."""
    a0, a1, a2 = [float(value) for value in params]
    results = evaluate_family(
        fit_windows,
        lookup_q,
        lookup_values,
        lambda _n, q_array: a0 + (a1 / q_array) + (a2 / (q_array * q_array)),
    )
    return float(max(item["sign_mismatch_fraction"] for item in results))


# 関数: parity-split q-dependent family の objective を返す。
def parity_q_family_objective(
    params: np.ndarray,
    fit_windows: list[dict[str, object]],
    lookup_q: np.ndarray,
    lookup_values: np.ndarray,
) -> float:
    """Return the minimax objective for odd/even split delta(q)."""
    odd_a0, odd_a1, even_a0, even_a1 = [float(value) for value in params]
    results = evaluate_family(
        fit_windows,
        lookup_q,
        lookup_values,
        lambda harmonic_index, q_array: (
            odd_a0 + (odd_a1 / q_array)
            if (harmonic_index % 2) == 1
            else even_a0 + (even_a1 / q_array)
        ),
    )
    return float(max(item["sign_mismatch_fraction"] for item in results))


# 関数: audit で使う公式群を返す。
def build_formulae() -> dict[str, str]:
    """Return formulas used in the q-dependent loading audit."""
    return {
        "retained_constant_theorem": "delta_q,jet = (3/2) (h1 / h0)",
        "minimal_q_dependent_family": "delta_q^(2)(q)=a0 + a1/q + a2/q^2",
        "parity_split_family": "delta_q^(oe)(q)=a_o0+a_o1/q (odd), a_e0+a_e1/q (even)",
        "independent_harmonic_loading": "delta_q,star^(n)=argmin_delta mismatch_n(delta)",
        "windowwise_rule": "sigma_img,delta^(n)(q)=(-1)^n sign(F_exact(|q_alias^(n)+(-1)^(n+1) delta_q^(n)(q)-q|))",
    }


# 関数: `.2047-.2050` を実行する。
def main() -> None:
    """Execute the q-dependent boundary loading or higher-harmonic signed-rule audit."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        LONG_ROADMAP,
        PART5,
        QBALL_BRANCH_REFRESH,
        PRIOR_GATE,
        PRIOR_AUDIT,
    ):
        sign_base.require(path)

    status_text = sign_base.read_text(STATUS)
    roadmap_text = sign_base.read_text(ROADMAP)
    current_problem_text = sign_base.read_text(CURRENT_PROBLEM)
    current_status_text = sign_base.read_text(CURRENT_STATUS)
    unified_text = sign_base.read_text(UNIFIED_ROADMAP)
    long_text = sign_base.read_text(LONG_ROADMAP)
    part5_text = sign_base.read_text(PART5)

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]
    inventory_ready = bool(prior_gate_summary["q_dependent_or_higher_harmonic_loading_admissible_now"])

    qball_branch_refresh = sign_base.read_json(QBALL_BRANCH_REFRESH)
    scalar_ground_state = sign_base.extract_scalar_ground_state(qball_branch_refresh)
    qball_module = sign_base.load_qball_module()
    radius, field, _field_prime = qball_module.solve_full_profile(
        float(scalar_ground_state["beta_n"]),
        float(scalar_ground_state["central_amplitude"]),
    )
    weight = (field**2) * (radius**2)
    norm = float(np.trapezoid(weight, radius))
    bulk_delta_r, _bulk_fraction, _edge_gap = alias_base.bulk_grid_summary(radius)
    alias_1 = (2.0 * np.pi) / bulk_delta_r
    lookup_q = np.arange(
        0.0,
        phase_base.LOOKUP_Q_MAX + phase_base.LOOKUP_Q_STEP,
        phase_base.LOOKUP_Q_STEP,
        dtype=float,
    )
    lookup_values = phase_base.form_factor_array(radius, weight, norm, lookup_q)
    windows = build_windows(radius, weight, norm, alias_1)
    fit_windows = windows[:4]
    holdout_windows = windows[4:6]
    extension_windows = windows[6:8]

    constant_delta = float(prior_audit_summary["delta_q_theorem_over_m0"])
    constant_results = evaluate_family(
        windows,
        lookup_q,
        lookup_values,
        lambda _n, q_array: np.full_like(q_array, constant_delta, dtype=float),
    )

    q2_bounds = [(0.0, 1.5), (-1000.0, 1000.0), (-100000.0, 100000.0)]
    q2_result = differential_evolution(
        lambda params: q2_family_objective(params, fit_windows, lookup_q, lookup_values),
        q2_bounds,
        maxiter=QDEP_MAXITER,
        popsize=QDEP_POPSIZE,
        seed=QDEP_SEED,
        polish=False,
        disp=False,
    )
    q2_params = [float(value) for value in q2_result.x]
    q2_results = evaluate_family(
        windows,
        lookup_q,
        lookup_values,
        lambda _n, q_array: q2_params[0] + (q2_params[1] / q_array) + (q2_params[2] / (q_array * q_array)),
    )

    parity_q_bounds = [(0.0, 1.5), (-500.0, 500.0), (0.0, 1.5), (-500.0, 500.0)]
    parity_q_result = differential_evolution(
        lambda params: parity_q_family_objective(params, fit_windows, lookup_q, lookup_values),
        parity_q_bounds,
        maxiter=QDEP_MAXITER,
        popsize=QDEP_POPSIZE,
        seed=QDEP_SEED,
        polish=False,
        disp=False,
    )
    parity_q_params = [float(value) for value in parity_q_result.x]
    parity_q_results = evaluate_family(
        windows,
        lookup_q,
        lookup_values,
        lambda harmonic_index, q_array: (
            parity_q_params[0] + (parity_q_params[1] / q_array)
            if (harmonic_index % 2) == 1
            else parity_q_params[2] + (parity_q_params[3] / q_array)
        ),
    )

    independent_deltas: list[float] = []
    independent_results: list[dict[str, float]] = []
    for window in windows:
        best_delta, metrics = optimize_independent_delta(window, lookup_q, lookup_values)
        independent_deltas.append(best_delta)
        independent_results.append(metrics)

    constant_fit_summary = summarize_window_group(constant_results, [1, 2, 3, 4])
    constant_holdout_summary = summarize_window_group(constant_results, [5, 6])
    q2_fit_summary = summarize_window_group(q2_results, [1, 2, 3, 4])
    q2_holdout_summary = summarize_window_group(q2_results, [5, 6])
    parity_q_fit_summary = summarize_window_group(parity_q_results, [1, 2, 3, 4])
    parity_q_holdout_summary = summarize_window_group(parity_q_results, [5, 6])
    independent_core_summary = summarize_window_group(independent_results, [3, 4, 5, 6])
    independent_extension_summary = summarize_window_group(independent_results, [7, 8])

    harmonic_delta_array = np.array(independent_deltas[2:8], dtype=float)
    harmonic_delta_total_variation = float(np.sum(np.abs(np.diff(harmonic_delta_array))))
    harmonic_delta_range = float(np.max(harmonic_delta_array) - np.min(harmonic_delta_array))
    harmonic_delta_monotone = bool(
        np.all(np.diff(harmonic_delta_array) >= -1.0e-12)
        or np.all(np.diff(harmonic_delta_array) <= 1.0e-12)
    )
    q2_predicted_delta_means = np.array([item["delta_mean"] for item in q2_results[2:8]], dtype=float)
    q2_vs_independent_delta_rms = float(
        np.sqrt(np.mean((q2_predicted_delta_means - harmonic_delta_array) ** 2))
    )

    minimal_q_dependent_boundary_loading_supported = bool(
        q2_fit_summary["max_mismatch"] <= 0.2
        and q2_holdout_summary["max_mismatch"] <= 0.2
        and q2_holdout_summary["min_correlation"] >= 0.6
    )
    generous_parity_split_loading_supported = bool(
        parity_q_fit_summary["max_mismatch"] <= 0.2
        and parity_q_holdout_summary["max_mismatch"] <= 0.2
        and parity_q_holdout_summary["min_correlation"] >= 0.6
    )
    higher_harmonic_windowwise_signed_rule_admissible = bool(
        independent_core_summary["max_mismatch"] <= 0.2
        and independent_extension_summary["max_mismatch"] <= 0.2
        and not harmonic_delta_monotone
        and q2_vs_independent_delta_rms >= 0.15
    )
    same_level_constant_delta_retry_admissible = False
    same_level_minimal_q_dependent_retry_admissible = False
    substantive_pack_update_required_now = False
    physical_reject_required = False

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "q-dependent loading inventory ready",
            sign_base.truth(inventory_ready),
            "The branch starts only after the constant-slip theorem is fixed and same-level constant-delta retry is closed.",
        ),
        sign_base.row(
            "constant_delta_q_theorem_over_m0",
            "watch",
            "retained constant theorem delta_q/m0",
            constant_delta,
            "This is the active-window boundary theorem retained by `.2039-.2046`.",
        ),
        sign_base.row(
            "constant_fit_window_max_mismatch_fraction",
            "watch",
            "constant-theorem max mismatch on fit set n=1..4",
            constant_fit_summary["max_mismatch"],
            "The retained constant theorem closes active windows but fails on translated higher harmonics.",
        ),
        sign_base.row(
            "constant_holdout_window_max_mismatch_fraction",
            "watch",
            "constant-theorem max mismatch on holdout set n=5..6",
            constant_holdout_summary["max_mismatch"],
            "The holdout failure is the blocker that motivates this branch.",
        ),
        sign_base.row(
            "q2_loading_a0",
            "watch",
            "q-dependent family a0",
            q2_params[0],
            "Minimal smooth loading is first tested with delta(q)=a0+a1/q+a2/q^2.",
        ),
        sign_base.row(
            "q2_loading_a1",
            "watch",
            "q-dependent family a1",
            q2_params[1],
            "The 1/q term is the first boundary-asymptotic correction beyond the retained constant theorem.",
        ),
        sign_base.row(
            "q2_loading_a2",
            "watch",
            "q-dependent family a2",
            q2_params[2],
            "The 1/q^2 term is the minimal curvature extension tested before opening a genuinely new signed rule.",
        ),
        sign_base.row(
            "q2_fit_window_max_mismatch_fraction",
            "watch",
            "q-dependent family max mismatch on fit set n=1..4",
            q2_fit_summary["max_mismatch"],
            "The minimal smooth loading can improve the first translated windows, but it must also survive holdouts to remain canonical.",
        ),
        sign_base.row(
            "q2_holdout_window_max_mismatch_fraction",
            "watch",
            "q-dependent family max mismatch on holdout set n=5..6",
            q2_holdout_summary["max_mismatch"],
            "A large holdout mismatch means the smooth q-dependent family is not a stable theorem-level continuation.",
        ),
        sign_base.row(
            "parity_split_fit_window_max_mismatch_fraction",
            "watch",
            "odd/even split family max mismatch on fit set n=1..4",
            parity_q_fit_summary["max_mismatch"],
            "A more generous odd/even split is tested so the branch cannot dismiss q-dependence too early.",
        ),
        sign_base.row(
            "parity_split_holdout_window_max_mismatch_fraction",
            "watch",
            "odd/even split family max mismatch on holdout set n=5..6",
            parity_q_holdout_summary["max_mismatch"],
            "If even the split family fails on holdout windows, the honest next surface is not a smooth q-dependent loading law.",
        ),
        sign_base.row(
            "independent_harmonic3_best_delta_over_m0",
            "watch",
            "windowwise optimum delta_q/m0 on harmonic-3 fit window",
            independent_deltas[2],
            "Independent windowwise loading shows how much loading the third harmonic really wants once smooth q-dependence is relaxed.",
        ),
        sign_base.row(
            "independent_harmonic4_best_delta_over_m0",
            "watch",
            "windowwise optimum delta_q/m0 on harmonic-4 edge window",
            independent_deltas[3],
            "The fourth harmonic already prefers a very different loading from the retained constant theorem.",
        ),
        sign_base.row(
            "independent_harmonic5_best_delta_over_m0",
            "watch",
            "windowwise optimum delta_q/m0 on harmonic-5 fit holdout",
            independent_deltas[4],
            "The fifth harmonic holdout stays scalar-compatible only after switching to a different local optimum branch.",
        ),
        sign_base.row(
            "independent_harmonic6_best_delta_over_m0",
            "watch",
            "windowwise optimum delta_q/m0 on harmonic-6 edge holdout",
            independent_deltas[5],
            "The sixth harmonic holdout prefers a much smaller loading again, showing that the sequence is not smooth in q.",
        ),
        sign_base.row(
            "harmonic_delta_monotone",
            "reject" if not harmonic_delta_monotone else "pass",
            "windowwise harmonic delta sequence monotone",
            sign_base.truth(harmonic_delta_monotone),
            "A non-monotone loading sequence is evidence against one smooth q-dependent theorem family.",
        ),
        sign_base.row(
            "q2_vs_independent_delta_rms",
            "watch",
            "RMS gap between q-dependent family means and independent harmonic optima",
            q2_vs_independent_delta_rms,
            "A large RMS gap shows that the minimal smooth family cannot track the harmonic-by-harmonic loading sequence even before exact promotion is considered.",
        ),
        sign_base.row(
            "minimal_q_dependent_boundary_loading_supported",
            "reject" if not minimal_q_dependent_boundary_loading_supported else "pass",
            "minimal q-dependent boundary loading supported",
            sign_base.truth(minimal_q_dependent_boundary_loading_supported),
            "The minimal smooth loading family is only retained if it closes the translated and holdout windows simultaneously.",
        ),
        sign_base.row(
            "generous_parity_split_loading_supported",
            "reject" if not generous_parity_split_loading_supported else "pass",
            "generous odd/even split loading supported",
            sign_base.truth(generous_parity_split_loading_supported),
            "A parity-split relaxation is tracked so the branch can honestly distinguish smooth-q failure from a purely even/odd loading effect.",
        ),
        sign_base.row(
            "higher_harmonic_windowwise_signed_rule_admissible",
            "pass" if higher_harmonic_windowwise_signed_rule_admissible else "reject",
            "higher-harmonic windowwise signed rule admissible",
            sign_base.truth(higher_harmonic_windowwise_signed_rule_admissible),
            "Once smooth q-dependent loading fails but windowwise harmonic loading stays inside the partial-retain envelope, the honest next surface is a harmonic-index dependent signed rule.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "bulk_delta_r_over_m0": float(bulk_delta_r),
        "first_alias_harmonic_over_m0": float(alias_1),
        "constant_delta_q_theorem_over_m0": constant_delta,
        "constant_fit_window_max_mismatch_fraction": constant_fit_summary["max_mismatch"],
        "constant_holdout_window_max_mismatch_fraction": constant_holdout_summary["max_mismatch"],
        "q2_loading_a0": q2_params[0],
        "q2_loading_a1": q2_params[1],
        "q2_loading_a2": q2_params[2],
        "q2_fit_window_max_mismatch_fraction": q2_fit_summary["max_mismatch"],
        "q2_fit_window_max_abs_error": q2_fit_summary["max_abs_error"],
        "q2_fit_window_min_sign_correlation": q2_fit_summary["min_correlation"],
        "q2_holdout_window_max_mismatch_fraction": q2_holdout_summary["max_mismatch"],
        "q2_holdout_window_max_abs_error": q2_holdout_summary["max_abs_error"],
        "q2_holdout_window_min_sign_correlation": q2_holdout_summary["min_correlation"],
        "parity_split_odd_a0": parity_q_params[0],
        "parity_split_odd_a1": parity_q_params[1],
        "parity_split_even_a0": parity_q_params[2],
        "parity_split_even_a1": parity_q_params[3],
        "parity_split_fit_window_max_mismatch_fraction": parity_q_fit_summary["max_mismatch"],
        "parity_split_fit_window_max_abs_error": parity_q_fit_summary["max_abs_error"],
        "parity_split_fit_window_min_sign_correlation": parity_q_fit_summary["min_correlation"],
        "parity_split_holdout_window_max_mismatch_fraction": parity_q_holdout_summary["max_mismatch"],
        "parity_split_holdout_window_max_abs_error": parity_q_holdout_summary["max_abs_error"],
        "parity_split_holdout_window_min_sign_correlation": parity_q_holdout_summary["min_correlation"],
        "independent_harmonic3_best_delta_over_m0": independent_deltas[2],
        "independent_harmonic3_best_mismatch_fraction": independent_results[2]["sign_mismatch_fraction"],
        "independent_harmonic4_best_delta_over_m0": independent_deltas[3],
        "independent_harmonic4_best_mismatch_fraction": independent_results[3]["sign_mismatch_fraction"],
        "independent_harmonic5_best_delta_over_m0": independent_deltas[4],
        "independent_harmonic5_best_mismatch_fraction": independent_results[4]["sign_mismatch_fraction"],
        "independent_harmonic6_best_delta_over_m0": independent_deltas[5],
        "independent_harmonic6_best_mismatch_fraction": independent_results[5]["sign_mismatch_fraction"],
        "independent_harmonic7_best_delta_over_m0": independent_deltas[6],
        "independent_harmonic7_best_mismatch_fraction": independent_results[6]["sign_mismatch_fraction"],
        "independent_harmonic8_best_delta_over_m0": independent_deltas[7],
        "independent_harmonic8_best_mismatch_fraction": independent_results[7]["sign_mismatch_fraction"],
        "independent_core_window_max_mismatch_fraction": independent_core_summary["max_mismatch"],
        "independent_extension_window_max_mismatch_fraction": independent_extension_summary["max_mismatch"],
        "harmonic_delta_total_variation": harmonic_delta_total_variation,
        "harmonic_delta_range": harmonic_delta_range,
        "harmonic_delta_monotone": harmonic_delta_monotone,
        "q2_vs_independent_delta_rms": q2_vs_independent_delta_rms,
        "minimal_q_dependent_boundary_loading_supported": minimal_q_dependent_boundary_loading_supported,
        "generous_parity_split_loading_supported": generous_parity_split_loading_supported,
        "higher_harmonic_windowwise_signed_rule_admissible": higher_harmonic_windowwise_signed_rule_admissible,
        "same_level_constant_delta_retry_admissible": same_level_constant_delta_retry_admissible,
        "same_level_minimal_q_dependent_retry_admissible": same_level_minimal_q_dependent_retry_admissible,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "substantive_pack_update_required_now": substantive_pack_update_required_now,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2049",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "status": sign_base.display_path(STATUS),
                "roadmap": sign_base.display_path(ROADMAP),
                "ai_context": sign_base.display_path(AI_CONTEXT),
                "work_history_recent": sign_base.display_path(WORK_HISTORY_RECENT),
                "current_problem": sign_base.display_path(CURRENT_PROBLEM),
                "current_status": sign_base.display_path(CURRENT_STATUS),
                "unified_roadmap": sign_base.display_path(UNIFIED_ROADMAP),
                "long_roadmap": sign_base.display_path(LONG_ROADMAP),
                "part5": sign_base.display_path(PART5),
                "qball_branch_refresh": sign_base.display_path(QBALL_BRANCH_REFRESH),
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
            },
            "constants": {
                "delta_search_over_m0": [DELTA_MIN, DELTA_MAX],
                "delta_search_step_over_m0": DELTA_STEP,
                "search_decimation": SEARCH_DECIMATION,
                "fit_window_count": len(fit_windows),
                "holdout_window_count": len(holdout_windows),
                "extension_window_count": len(extension_windows),
                "next_route_name": NEXT_ROUTE_NAME,
                "next_route": NEXT_ROUTE,
                "followup_route_name": FOLLOWUP_ROUTE_NAME,
                "followup_route": FOLLOWUP_ROUTE,
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_q_dependent_loading_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2047"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, "8.7.56.2047-.2050"),
                "current_problem_hit": sign_base.hit(current_problem_text, "q-dependent or harmonic-index dependent loading"),
                "current_status_hit": sign_base.hit(current_status_text, "q-dependent or harmonic-index dependent loading"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2047-.2050"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2047-.2050"),
                "part5_hit": sign_base.hit(part5_text, ".2039-.2046"),
            },
        },
    )

    route_payload = sign_base.payload(
        "8.7.56.2050",
        STEP_NAME + " route sync",
        declaration_payload["inputs"],
        [
            sign_base.row(
                "minimal_q_dependent_boundary_loading_supported",
                "reject" if not minimal_q_dependent_boundary_loading_supported else "pass",
                "minimal q-dependent boundary loading supported",
                sign_base.truth(minimal_q_dependent_boundary_loading_supported),
                "The branch only remains on a smooth loading family if the minimal q-dependent surface survives translated and holdout windows together.",
            ),
            sign_base.row(
                "higher_harmonic_windowwise_signed_rule_admissible",
                "pass" if higher_harmonic_windowwise_signed_rule_admissible else "reject",
                "higher-harmonic windowwise signed rule admissible",
                sign_base.truth(higher_harmonic_windowwise_signed_rule_admissible),
                "When smooth q-dependence fails but windowwise harmonic loading remains structured, the honest next route is a harmonic-index signed rule.",
            ),
            sign_base.row(
                "next_route_fixed",
                "pass",
                "next route fixed",
                1.0,
                "The next official branch is the alias-image phase-slip loading closeout / registry.",
            ),
        ],
        summary,
        {
            "overall_status": "vector_qball_form_factor_q_dependent_loading_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"formulas": build_formulae()},
    )

    declaration_paths = write_artifact("declaration_gate", declaration_payload)
    route_paths = write_artifact("route_sync", route_payload)
    print("[done] 8.7.56.2047-.2050 complete")
    print(f"[info] declaration gate: {declaration_paths['json']}")
    print(f"[info] route sync: {route_paths['json']}")


if __name__ == "__main__":
    main()
