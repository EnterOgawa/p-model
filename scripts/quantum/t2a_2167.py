#!/usr/bin/env python3
"""Generate 8.7.56.2167-.2170 third post-break extreme-farther artifacts."""

from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
import scripts.quantum.t2a_2023 as alias_base
import scripts.quantum.t2a_2031 as phase_base
import scripts.quantum.t2a_2055 as lattice_base
import scripts.quantum.t2a_2111 as sparse_base
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
PRIOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2159-2162",
        "harmonic_second_post_break_piecewise_ultra_farther",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_REGISTRY = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2163-2166",
        "harmonic_second_post_break_piecewise_registry_refresh",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2167-2170"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor third post-break "
    "piecewise extreme-farther continuation audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "harmonic_third_post_break_piecewise_extreme_farther",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_third_post_break_"
    "piecewise_validation_to_491520_farther_continuation_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_third_post_break_"
    "piecewise_extreme_farther_reactivation_gate_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_third_post_break_"
    "piecewise_registry_refresh"
)
NEXT_ROUTE = "8.7.56.2171"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_fourth_post_break_"
    "piecewise_farther_continuation_audit"
)
FOLLOWUP_ROUTE = "8.7.56.2175"

EXTREME_FARTHER_BANDS = [
    (491521, 499712),
    (499713, 507904),
    (507905, 516096),
    (516097, 524288),
    (524289, 532480),
    (532481, 540672),
    (540673, 548864),
    (548865, 557056),
    (557057, 565248),
    (565249, 573440),
    (573441, 581632),
    (581633, 589824),
]
THIRD_HOLDOUT = EXTREME_FARTHER_BANDS[:4]
THIRD_MONITOR = EXTREME_FARTHER_BANDS[4:]
FOURTH_FIT = EXTREME_FARTHER_BANDS[:4]
FOURTH_HOLDOUT = EXTREME_FARTHER_BANDS[4:8]
FOURTH_MONITOR = EXTREME_FARTHER_BANDS[8:]
EXTREME_FARTHER_SAMPLE_HARMONIC_STRIDE = 512
Q_REF_OVER_M0 = 425984.0
MISMATCH_TOL = 0.02
CORRELATION_TOL = 0.02
RECON_TOL = 6.0e-10


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


# 関数: テキスト中の最初の一致行を返す。

def find_line(text: str, pattern: str) -> dict[str, object] | None:
    """Return the first matching line payload for one text pattern."""
    for line_number, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {
                "pattern": pattern,
                "line": line_number,
                "text": line.strip(),
            }

    return None


# 関数: stress coordinate を返す。

def stress_log_coordinate(q_center_over_m0: np.ndarray) -> np.ndarray:
    """Return the stress log coordinate relative to the retained reference."""
    return np.log2(q_center_over_m0 / Q_REF_OVER_M0)


# 関数: affine law を最小二乗で fit する。

def fit_affine(x_values: np.ndarray, y_values: np.ndarray) -> tuple[float, float]:
    """Return slope and intercept for one affine fit."""
    design = np.vstack([x_values, np.ones_like(x_values)]).T
    slope, intercept = np.linalg.lstsq(design, y_values, rcond=None)[0]
    return float(slope), float(intercept)


# 関数: power-law decay を最小二乗で fit する。

def fit_power_law(
    q_center_over_m0: np.ndarray,
    y_values: np.ndarray,
) -> tuple[float, float]:
    """Return exponent and prefactor for one power law."""
    log_q = np.log(q_center_over_m0)
    log_y = np.log(y_values)
    slope, intercept = fit_affine(log_q, log_y)
    exponent = -slope
    prefactor = math.exp(intercept)
    return float(exponent), float(prefactor)


# 関数: sampled summaries から系列を構成する。

def build_series(
    summaries: dict[str, dict[str, float]],
    bands: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return centers, mismatches, correlations, and reconstruction errors."""
    centers: list[float] = []
    mismatches: list[float] = []
    correlations: list[float] = []
    recon_errors: list[float] = []
    for band_start, band_end in bands:
        key = f"{band_start}_{band_end}"
        band_summary = summaries[key]
        centers.append(0.5 * (band_start + band_end))
        mismatches.append(float(band_summary["max_mismatch"]))
        correlations.append(float(band_summary["min_correlation"]))
        recon_errors.append(float(band_summary["max_abs_error"]))

    return (
        np.asarray(centers, dtype=float),
        np.asarray(mismatches, dtype=float),
        np.asarray(correlations, dtype=float),
        np.asarray(recon_errors, dtype=float),
    )


# 関数: 最大 absolute error を返す。

def max_abs_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Return one max absolute error."""
    return float(np.max(np.abs(actual - predicted)))


# 関数: audit で使う公式群を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the extreme-farther continuation audit."""
    return {
        "retained_lattice": "delta_q^(n,m) = delta_q,base^(box) + m_n Delta_box",
        "same_third_piecewise": "M_3(x)=a_3 x+b_3, C_3(x)=c_3 x+d_3, E_3(q)=A_3 q^{-nu_3} inherited from 393217..491520",
        "fourth_piecewise_reserve": "M_4(x)=a_4 x+b_4, C_4(x)=c_4 x+d_4, E_4(q)=A_4 q^{-nu_4} fitted on 491521..524288 only as a reserve diagnostic",
        "selection_rule": "A fourth post-break surface becomes admissible only if the inherited third segment fails and the reserve fourth segment passes extreme-farther holdout and monitor windows.",
    }


# 関数: `.2167-.2170` を実行する。

def main() -> None:
    """Execute the third post-break extreme-farther continuation audit."""
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
        PRIOR_AUDIT,
        PRIOR_REGISTRY,
    ):
        sign_base.require(path)

    status_text = sign_base.read_text(STATUS)
    roadmap_text = sign_base.read_text(ROADMAP)
    current_problem_text = sign_base.read_text(CURRENT_PROBLEM)
    current_status_text = sign_base.read_text(CURRENT_STATUS)
    unified_text = sign_base.read_text(UNIFIED_ROADMAP)
    long_text = sign_base.read_text(LONG_ROADMAP)
    part5_text = sign_base.read_text(PART5)
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]
    prior_registry_summary = sign_base.read_json(PRIOR_REGISTRY)["summary"]
    inventory_ready = bool(
        prior_registry_summary["gate_b_third_piecewise_reactivation_selected"]
    )

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

    theorem_lattice_base = float(prior_audit_summary["theorem_lattice_base_over_m0"])
    theorem_lattice_step = float(prior_audit_summary["bulk_delta_r_over_m0"])
    windows = sparse_base.build_sampled_windows(
        radius,
        weight,
        norm,
        alias_1,
        EXTREME_FARTHER_BANDS,
        EXTREME_FARTHER_SAMPLE_HARMONIC_STRIDE,
    )
    results = lattice_base.evaluate_lattice_family(
        windows,
        lookup_q,
        lookup_values,
        theorem_lattice_base,
        theorem_lattice_step,
    )
    band_summaries = {
        f"{band_start}_{band_end}": sparse_base.summarize_sampled_band(
            windows,
            results,
            band_start,
            band_end,
        )
        for band_start, band_end in EXTREME_FARTHER_BANDS
    }

    centers, mismatches, correlations, recon_errors = build_series(
        band_summaries,
        EXTREME_FARTHER_BANDS,
    )
    x_all = stress_log_coordinate(centers)

    third_m_slope = float(prior_registry_summary["third_post_break_piecewise_mismatch_slope"])
    third_m_intercept = float(
        prior_registry_summary["third_post_break_piecewise_mismatch_intercept"]
    )
    third_c_slope = float(prior_registry_summary["third_post_break_piecewise_correlation_slope"])
    third_c_intercept = float(
        prior_registry_summary["third_post_break_piecewise_correlation_intercept"]
    )
    third_rec_exp = float(
        prior_registry_summary["third_post_break_reconstruction_decay_exponent"]
    )
    third_rec_pref = float(
        prior_registry_summary["third_post_break_reconstruction_decay_prefactor"]
    )
    third_m_pred = (third_m_slope * x_all) + third_m_intercept
    third_c_pred = (third_c_slope * x_all) + third_c_intercept
    third_r_pred = third_rec_pref * np.power(centers, -third_rec_exp)

    third_holdout_slice = slice(0, len(THIRD_HOLDOUT))
    third_monitor_slice = slice(len(THIRD_HOLDOUT), len(EXTREME_FARTHER_BANDS))
    third_extreme_holdout_max_mismatch_abs_error = max_abs_error(
        mismatches[third_holdout_slice],
        third_m_pred[third_holdout_slice],
    )
    third_extreme_holdout_max_correlation_abs_error = max_abs_error(
        correlations[third_holdout_slice],
        third_c_pred[third_holdout_slice],
    )
    third_extreme_holdout_max_reconstruction_abs_error = max_abs_error(
        recon_errors[third_holdout_slice],
        third_r_pred[third_holdout_slice],
    )
    third_extreme_monitor_max_mismatch_abs_error = max_abs_error(
        mismatches[third_monitor_slice],
        third_m_pred[third_monitor_slice],
    )
    third_extreme_monitor_max_correlation_abs_error = max_abs_error(
        correlations[third_monitor_slice],
        third_c_pred[third_monitor_slice],
    )
    third_extreme_monitor_max_reconstruction_abs_error = max_abs_error(
        recon_errors[third_monitor_slice],
        third_r_pred[third_monitor_slice],
    )
    same_third_piecewise_extreme_farther_continuation_supported = bool(
        third_extreme_holdout_max_mismatch_abs_error <= MISMATCH_TOL
        and third_extreme_holdout_max_correlation_abs_error <= CORRELATION_TOL
        and third_extreme_holdout_max_reconstruction_abs_error <= RECON_TOL
        and third_extreme_monitor_max_mismatch_abs_error <= MISMATCH_TOL
        and third_extreme_monitor_max_correlation_abs_error <= CORRELATION_TOL
        and third_extreme_monitor_max_reconstruction_abs_error <= RECON_TOL
    )
    third_post_break_piecewise_validation_to_589824_supported = bool(
        same_third_piecewise_extreme_farther_continuation_supported
    )

    fourth_fit_slice = slice(0, len(FOURTH_FIT))
    fourth_holdout_slice = slice(
        len(FOURTH_FIT),
        len(FOURTH_FIT) + len(FOURTH_HOLDOUT),
    )
    fourth_monitor_slice = slice(
        len(FOURTH_FIT) + len(FOURTH_HOLDOUT),
        len(EXTREME_FARTHER_BANDS),
    )
    fourth_m_slope, fourth_m_intercept = fit_affine(
        x_all[fourth_fit_slice],
        mismatches[fourth_fit_slice],
    )
    fourth_c_slope, fourth_c_intercept = fit_affine(
        x_all[fourth_fit_slice],
        correlations[fourth_fit_slice],
    )
    fourth_rec_exp, fourth_rec_pref = fit_power_law(
        centers[fourth_fit_slice],
        recon_errors[fourth_fit_slice],
    )
    fourth_m_pred = (fourth_m_slope * x_all) + fourth_m_intercept
    fourth_c_pred = (fourth_c_slope * x_all) + fourth_c_intercept
    fourth_r_pred = fourth_rec_pref * np.power(centers, -fourth_rec_exp)
    fourth_holdout_max_mismatch_abs_error = max_abs_error(
        mismatches[fourth_holdout_slice],
        fourth_m_pred[fourth_holdout_slice],
    )
    fourth_holdout_max_correlation_abs_error = max_abs_error(
        correlations[fourth_holdout_slice],
        fourth_c_pred[fourth_holdout_slice],
    )
    fourth_holdout_max_reconstruction_abs_error = max_abs_error(
        recon_errors[fourth_holdout_slice],
        fourth_r_pred[fourth_holdout_slice],
    )
    fourth_monitor_max_mismatch_abs_error = max_abs_error(
        mismatches[fourth_monitor_slice],
        fourth_m_pred[fourth_monitor_slice],
    )
    fourth_monitor_max_correlation_abs_error = max_abs_error(
        correlations[fourth_monitor_slice],
        fourth_c_pred[fourth_monitor_slice],
    )
    fourth_monitor_max_reconstruction_abs_error = max_abs_error(
        recon_errors[fourth_monitor_slice],
        fourth_r_pred[fourth_monitor_slice],
    )
    fourth_post_break_piecewise_validation_to_589824_supported = bool(
        fourth_holdout_max_mismatch_abs_error <= MISMATCH_TOL
        and fourth_holdout_max_correlation_abs_error <= CORRELATION_TOL
        and fourth_holdout_max_reconstruction_abs_error <= RECON_TOL
        and fourth_monitor_max_mismatch_abs_error <= MISMATCH_TOL
        and fourth_monitor_max_correlation_abs_error <= CORRELATION_TOL
        and fourth_monitor_max_reconstruction_abs_error <= RECON_TOL
    )
    fourth_post_break_piecewise_surface_admissible_now = bool(
        (not same_third_piecewise_extreme_farther_continuation_supported)
        and fourth_post_break_piecewise_validation_to_589824_supported
    )
    exact_global_extreme_farther_third_post_break_theorem_available = False
    physical_reject_required = False

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "third post-break extreme-farther inventory ready",
            sign_base.truth(inventory_ready),
            "The extreme-farther continuation starts only after the reserve third segment has already been promoted through harmonic 491520.",
        ),
        sign_base.row(
            "third_extreme_holdout_max_mismatch_abs_error",
            "pass" if third_extreme_holdout_max_mismatch_abs_error <= MISMATCH_TOL else "reject",
            "same third piecewise extreme-farther holdout max mismatch abs error through harmonic 524288",
            third_extreme_holdout_max_mismatch_abs_error,
            "The inherited third segment survives only if the next quartet stays inside the retained mismatch tolerance.",
        ),
        sign_base.row(
            "third_extreme_holdout_max_correlation_abs_error",
            "pass" if third_extreme_holdout_max_correlation_abs_error <= CORRELATION_TOL else "reject",
            "same third piecewise extreme-farther holdout max correlation abs error through harmonic 524288",
            third_extreme_holdout_max_correlation_abs_error,
            "The sign-floor channel must confirm the same extreme-farther survival for the inherited third segment.",
        ),
        sign_base.row(
            "third_extreme_monitor_max_mismatch_abs_error",
            "pass" if third_extreme_monitor_max_mismatch_abs_error <= MISMATCH_TOL else "reject",
            "same third piecewise extreme-farther monitor max mismatch abs error through harmonic 589824",
            third_extreme_monitor_max_mismatch_abs_error,
            "The extreme-farther monitor checks that the same third segment does not collapse immediately after the first quartet.",
        ),
        sign_base.row(
            "third_extreme_monitor_max_correlation_abs_error",
            "pass" if third_extreme_monitor_max_correlation_abs_error <= CORRELATION_TOL else "reject",
            "same third piecewise extreme-farther monitor max correlation abs error through harmonic 589824",
            third_extreme_monitor_max_correlation_abs_error,
            "The monitor condition must also hold on the sign-floor channel.",
        ),
        sign_base.row(
            "same_third_piecewise_extreme_farther_continuation_supported",
            "pass" if same_third_piecewise_extreme_farther_continuation_supported else "reject",
            "same third post-break piecewise extreme-farther continuation supported",
            sign_base.truth(same_third_piecewise_extreme_farther_continuation_supported),
            "No new surface is admissible while the inherited third segment still survives extreme-farther holdout and monitor windows.",
        ),
        sign_base.row(
            "fourth_post_break_piecewise_mismatch_slope",
            "watch",
            "fourth post-break reserve mismatch slope",
            fourth_m_slope,
            "A fourth segment is computed only as a reserve diagnostic after the same third segment has already been tested on the extreme-farther window.",
        ),
        sign_base.row(
            "fourth_holdout_max_mismatch_abs_error",
            "pass" if fourth_holdout_max_mismatch_abs_error <= MISMATCH_TOL else "reject",
            "fourth post-break holdout max mismatch abs error through harmonic 557056",
            fourth_holdout_max_mismatch_abs_error,
            "The reserve fourth segment would only become admissible if the inherited third segment failed first.",
        ),
        sign_base.row(
            "fourth_holdout_max_correlation_abs_error",
            "pass" if fourth_holdout_max_correlation_abs_error <= CORRELATION_TOL else "reject",
            "fourth post-break holdout max correlation abs error through harmonic 557056",
            fourth_holdout_max_correlation_abs_error,
            "The reserve fourth segment is monitored on the sign-floor channel for completeness.",
        ),
        sign_base.row(
            "fourth_monitor_max_mismatch_abs_error",
            "pass" if fourth_monitor_max_mismatch_abs_error <= MISMATCH_TOL else "reject",
            "fourth post-break monitor max mismatch abs error through harmonic 589824",
            fourth_monitor_max_mismatch_abs_error,
            "Even a passing reserve fourth segment remains non-admissible when the inherited third segment already survives.",
        ),
        sign_base.row(
            "fourth_monitor_max_correlation_abs_error",
            "pass" if fourth_monitor_max_correlation_abs_error <= CORRELATION_TOL else "reject",
            "fourth post-break monitor max correlation abs error through harmonic 589824",
            fourth_monitor_max_correlation_abs_error,
            "The reserve route is kept only as a diagnostic and not as the official mainline.",
        ),
        sign_base.row(
            "fourth_post_break_piecewise_surface_admissible_now",
            "reject" if not fourth_post_break_piecewise_surface_admissible_now else "pass",
            "fourth post-break piecewise surface admissible now",
            sign_base.truth(fourth_post_break_piecewise_surface_admissible_now),
            "The retry gate opens the fourth segment only after the inherited third segment has honestly failed on the extreme-farther continuation audit.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "theorem_lattice_base_over_m0": theorem_lattice_base,
        "bulk_delta_r_over_m0": theorem_lattice_step,
        "extreme_farther_sample_harmonic_stride": EXTREME_FARTHER_SAMPLE_HARMONIC_STRIDE,
        "third_post_break_piecewise_mismatch_slope": third_m_slope,
        "third_post_break_piecewise_mismatch_intercept": third_m_intercept,
        "third_post_break_piecewise_correlation_slope": third_c_slope,
        "third_post_break_piecewise_correlation_intercept": third_c_intercept,
        "third_post_break_reconstruction_decay_exponent": third_rec_exp,
        "third_post_break_reconstruction_decay_prefactor": third_rec_pref,
        "third_extreme_holdout_max_mismatch_abs_error": third_extreme_holdout_max_mismatch_abs_error,
        "third_extreme_holdout_max_correlation_abs_error": third_extreme_holdout_max_correlation_abs_error,
        "third_extreme_holdout_max_reconstruction_abs_error": third_extreme_holdout_max_reconstruction_abs_error,
        "third_extreme_monitor_max_mismatch_abs_error": third_extreme_monitor_max_mismatch_abs_error,
        "third_extreme_monitor_max_correlation_abs_error": third_extreme_monitor_max_correlation_abs_error,
        "third_extreme_monitor_max_reconstruction_abs_error": third_extreme_monitor_max_reconstruction_abs_error,
        "same_third_piecewise_extreme_farther_continuation_supported": same_third_piecewise_extreme_farther_continuation_supported,
        "third_post_break_piecewise_validation_to_589824_supported": third_post_break_piecewise_validation_to_589824_supported,
        "fourth_post_break_piecewise_mismatch_slope": fourth_m_slope,
        "fourth_post_break_piecewise_mismatch_intercept": fourth_m_intercept,
        "fourth_post_break_piecewise_correlation_slope": fourth_c_slope,
        "fourth_post_break_piecewise_correlation_intercept": fourth_c_intercept,
        "fourth_post_break_reconstruction_decay_exponent": fourth_rec_exp,
        "fourth_post_break_reconstruction_decay_prefactor": fourth_rec_pref,
        "fourth_holdout_max_mismatch_abs_error": fourth_holdout_max_mismatch_abs_error,
        "fourth_holdout_max_correlation_abs_error": fourth_holdout_max_correlation_abs_error,
        "fourth_holdout_max_reconstruction_abs_error": fourth_holdout_max_reconstruction_abs_error,
        "fourth_monitor_max_mismatch_abs_error": fourth_monitor_max_mismatch_abs_error,
        "fourth_monitor_max_correlation_abs_error": fourth_monitor_max_correlation_abs_error,
        "fourth_monitor_max_reconstruction_abs_error": fourth_monitor_max_reconstruction_abs_error,
        "fourth_post_break_piecewise_validation_to_589824_supported": fourth_post_break_piecewise_validation_to_589824_supported,
        "fourth_post_break_piecewise_surface_admissible_now": fourth_post_break_piecewise_surface_admissible_now,
        "exact_global_extreme_farther_third_post_break_theorem_available": exact_global_extreme_farther_third_post_break_theorem_available,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2169",
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
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
                "prior_registry": sign_base.display_path(PRIOR_REGISTRY),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route_name": NEXT_ROUTE_NAME,
                "next_route": NEXT_ROUTE,
                "followup_route_name": FOLLOWUP_ROUTE_NAME,
                "followup_route": FOLLOWUP_ROUTE,
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_third_post_break_extreme_farther_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2167"),
                "roadmap_branch_hit": find_line(roadmap_text, "8.7.56.2167-.2170"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2167"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2167"),
                "unified_roadmap_hit": find_line(unified_text, ".2163-.2166"),
                "long_roadmap_hit": find_line(long_text, ".2163-.2166"),
                "part5_hit": find_line(part5_text, ".2159-.2166"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_sync_rows = [
        sign_base.row(
            "status_synced",
            "pass",
            "STATUS sync target present",
            sign_base.truth(bool(find_line(status_text, "8.7.56.2167"))),
            "The extreme-farther continuation audit is only honest if the official status already points to the same third-segment route.",
        ),
        sign_base.row(
            "roadmap_synced",
            "pass",
            "ROADMAP sync target present",
            sign_base.truth(bool(find_line(roadmap_text, "8.7.56.2167-.2170"))),
            "The public roadmap must expose the same third post-break extreme-farther branch before route sync can proceed.",
        ),
        sign_base.row(
            "long_horizon_synced",
            "pass",
            "long-horizon roadmap sync target present",
            sign_base.truth(bool(find_line(long_text, ".2163-.2166"))),
            "The long-horizon roadmap must still expose the prior registry state before the extreme-farther continuation result is frozen.",
        ),
    ]
    route_sync_payload = sign_base.payload(
        "8.7.56.2170",
        STEP_NAME + " route sync",
        {
            "source_files": {
                "status": sign_base.display_path(STATUS),
                "roadmap": sign_base.display_path(ROADMAP),
                "current_problem": sign_base.display_path(CURRENT_PROBLEM),
                "current_status": sign_base.display_path(CURRENT_STATUS),
                "unified_roadmap": sign_base.display_path(UNIFIED_ROADMAP),
                "long_roadmap": sign_base.display_path(LONG_ROADMAP),
                "part5": sign_base.display_path(PART5),
                "declaration_gate": declaration_paths["json"],
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": BRANCH_CLASS,
                "next_route_name": NEXT_ROUTE_NAME,
                "next_route": NEXT_ROUTE,
                "followup_route_name": FOLLOWUP_ROUTE_NAME,
                "followup_route": FOLLOWUP_ROUTE,
            },
        },
        route_sync_rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_third_post_break_extreme_farther_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2167"),
                "roadmap_branch_hit": find_line(roadmap_text, "8.7.56.2167-.2170"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2167"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2167"),
                "unified_roadmap_hit": find_line(unified_text, ".2163-.2166"),
                "long_roadmap_hit": find_line(long_text, ".2163-.2166"),
                "part5_hit": find_line(part5_text, ".2159-.2166"),
            },
        },
    )
    write_artifact("route_sync", route_sync_payload)


if __name__ == "__main__":
    main()
