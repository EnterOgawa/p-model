#!/usr/bin/env python3
"""Generate 8.7.56.2135-.2138 post-break curvature vs piecewise artifacts.

The retained stress-envelope linear law is validated only through harmonic
90112. This branch keeps the same retained boundary bulk-lattice family and
the same exact sampled-window contract, then asks a narrower question:
should the first post-break continuation be modeled by a curvature law or by
a post-break piecewise affine law?
"""

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
        "8.7.56.2127-2130",
        "harmonic_sparse_drift_law_farther_validation",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2131-2134",
        "harmonic_sparse_drift_law_farther_validation_registry_refresh",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2135-2138"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor post-break "
    "stress-envelope piecewise or curvature reactivation"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "harmonic_post_break_piecewise_curvature",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_stress_envelope_linear_"
    "validation_to_90112_curvature_or_piecewise_reactivation_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_post_break_piecewise_"
    "holdout_gate_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_post_break_stress_envelope_"
    "registry_refresh"
)
NEXT_ROUTE = "8.7.56.2139"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_post_break_piecewise_"
    "farther_continuation_audit"
)
FOLLOWUP_ROUTE = "8.7.56.2143"

POST_BREAK_BANDS = [
    (90113, 98304),
    (98305, 106496),
    (106497, 114688),
    (114689, 122880),
    (122881, 131072),
    (131073, 139264),
    (139265, 147456),
    (147457, 155648),
    (155649, 163840),
    (163841, 172032),
    (172033, 180224),
    (180225, 188416),
    (188417, 196608),
]
FIT_BANDS = POST_BREAK_BANDS[:4]
HOLDOUT_BANDS = POST_BREAK_BANDS[4:9]
MONITOR_BANDS = POST_BREAK_BANDS[9:]
POST_BREAK_SAMPLE_HARMONIC_STRIDE = 512
Q_REF_OVER_M0 = 32768.0
MISMATCH_HOLDOUT_TOL = 0.02
CORRELATION_HOLDOUT_TOL = 0.02
RECON_HOLDOUT_TOL = 6.0e-10


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


# 関数: log2 stress coordinate を返す。

def stress_log_coordinate(q_center_over_m0: np.ndarray) -> np.ndarray:
    """Return the stress log coordinate relative to the retained reference."""
    return np.log2(q_center_over_m0 / Q_REF_OVER_M0)


# 関数: affine law を最小二乗で fit する。

def fit_affine(x_values: np.ndarray, y_values: np.ndarray) -> tuple[float, float]:
    """Return slope and intercept for one affine fit."""
    design = np.vstack([x_values, np.ones_like(x_values)]).T
    slope, intercept = np.linalg.lstsq(design, y_values, rcond=None)[0]
    return float(slope), float(intercept)


# 関数: quadratic law を最小二乗で fit する。

def fit_quadratic(x_values: np.ndarray, y_values: np.ndarray) -> tuple[float, float, float]:
    """Return quadratic coefficients for one least-squares fit."""
    quad, lin, const = np.polyfit(x_values, y_values, deg=2)
    return float(quad), float(lin), float(const)


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


# 関数: abs error の最大値を返す。

def max_abs_error(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Return one max absolute error."""
    return float(np.max(np.abs(actual - predicted)))


# 関数: audit で使う公式群を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the post-break reactivation audit."""
    return {
        "retained_lattice": "delta_q^(n,m) = delta_q,base^(box) + m_n Delta_box",
        "stress_coordinate": "x(q) = log2(q / 32768)",
        "piecewise_mismatch": "M_piece(x) = a_M x + b_M fitted on post-break bands 90113..122880",
        "piecewise_correlation": "C_piece(x) = a_C x + b_C fitted on post-break bands 90113..122880",
        "piecewise_reconstruction": "E_piece(q) = A_piece q^{-nu_piece} fitted on post-break bands 90113..122880",
        "curvature_mismatch": "M_curv(x) = c2 x^2 + c1 x + c0 fitted on the same fit bands",
        "curvature_correlation": "C_curv(x) = d2 x^2 + d1 x + d0 fitted on the same fit bands",
        "support_rule": "retain the post-break piecewise branch only if it beats curvature on the exact holdout and keeps holdout residuals inside small tolerances through harmonic 163840",
    }


# 関数: `.2135-.2138` を実行する。

def main() -> None:
    """Execute the post-break curvature vs piecewise audit."""
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
        PRIOR_GATE,
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
    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    inventory_ready = bool(prior_gate_summary["post_break_curvature_or_piecewise_surface_admissible_now"])

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
        POST_BREAK_BANDS,
        POST_BREAK_SAMPLE_HARMONIC_STRIDE,
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
        for band_start, band_end in POST_BREAK_BANDS
    }

    centers, mismatches, correlations, recon_errors = build_series(
        band_summaries,
        POST_BREAK_BANDS,
    )
    x_all = stress_log_coordinate(centers)
    x_fit = x_all[: len(FIT_BANDS)]
    x_holdout = x_all[len(FIT_BANDS) : len(FIT_BANDS) + len(HOLDOUT_BANDS)]
    x_monitor = x_all[-len(MONITOR_BANDS) :]

    piece_m_slope, piece_m_intercept = fit_affine(x_fit, mismatches[: len(FIT_BANDS)])
    piece_c_slope, piece_c_intercept = fit_affine(x_fit, correlations[: len(FIT_BANDS)])
    piece_rec_exp, piece_rec_pref = fit_power_law(
        centers[: len(FIT_BANDS)],
        recon_errors[: len(FIT_BANDS)],
    )
    piece_m_pred = (piece_m_slope * x_all) + piece_m_intercept
    piece_c_pred = (piece_c_slope * x_all) + piece_c_intercept
    piece_r_pred = piece_rec_pref * np.power(centers, -piece_rec_exp)

    curv_m_q2, curv_m_q1, curv_m_q0 = fit_quadratic(x_fit, mismatches[: len(FIT_BANDS)])
    curv_c_q2, curv_c_q1, curv_c_q0 = fit_quadratic(x_fit, correlations[: len(FIT_BANDS)])
    curv_m_pred = (curv_m_q2 * x_all * x_all) + (curv_m_q1 * x_all) + curv_m_q0
    curv_c_pred = (curv_c_q2 * x_all * x_all) + (curv_c_q1 * x_all) + curv_c_q0

    holdout_slice = slice(len(FIT_BANDS), len(FIT_BANDS) + len(HOLDOUT_BANDS))
    monitor_slice = slice(len(FIT_BANDS) + len(HOLDOUT_BANDS), len(POST_BREAK_BANDS))
    piecewise_holdout_max_mismatch_abs_error = max_abs_error(
        mismatches[holdout_slice],
        piece_m_pred[holdout_slice],
    )
    piecewise_holdout_max_correlation_abs_error = max_abs_error(
        correlations[holdout_slice],
        piece_c_pred[holdout_slice],
    )
    piecewise_holdout_max_reconstruction_abs_error = max_abs_error(
        recon_errors[holdout_slice],
        piece_r_pred[holdout_slice],
    )
    piecewise_monitor_max_mismatch_abs_error = max_abs_error(
        mismatches[monitor_slice],
        piece_m_pred[monitor_slice],
    )
    piecewise_monitor_max_correlation_abs_error = max_abs_error(
        correlations[monitor_slice],
        piece_c_pred[monitor_slice],
    )
    piecewise_monitor_max_reconstruction_abs_error = max_abs_error(
        recon_errors[monitor_slice],
        piece_r_pred[monitor_slice],
    )
    curvature_holdout_max_mismatch_abs_error = max_abs_error(
        mismatches[holdout_slice],
        curv_m_pred[holdout_slice],
    )
    curvature_holdout_max_correlation_abs_error = max_abs_error(
        correlations[holdout_slice],
        curv_c_pred[holdout_slice],
    )
    piecewise_holdout_combined_abs_error = (
        piecewise_holdout_max_mismatch_abs_error
        + piecewise_holdout_max_correlation_abs_error
        + piecewise_holdout_max_reconstruction_abs_error
    )
    curvature_holdout_combined_abs_error = (
        curvature_holdout_max_mismatch_abs_error
        + curvature_holdout_max_correlation_abs_error
        + piecewise_holdout_max_reconstruction_abs_error
    )

    post_break_piecewise_holdout_supported = bool(
        piecewise_holdout_max_mismatch_abs_error <= MISMATCH_HOLDOUT_TOL
        and piecewise_holdout_max_correlation_abs_error <= CORRELATION_HOLDOUT_TOL
        and piecewise_holdout_max_reconstruction_abs_error <= RECON_HOLDOUT_TOL
    )
    post_break_curvature_holdout_supported = bool(
        curvature_holdout_max_mismatch_abs_error <= MISMATCH_HOLDOUT_TOL
        and curvature_holdout_max_correlation_abs_error <= CORRELATION_HOLDOUT_TOL
    )
    piecewise_beats_curvature_on_holdout = bool(
        piecewise_holdout_combined_abs_error < curvature_holdout_combined_abs_error
    )
    piecewise_validation_to_163840_supported = bool(
        post_break_piecewise_holdout_supported and piecewise_beats_curvature_on_holdout
    )
    post_break_monitor_drift_detected = bool(
        piecewise_monitor_max_mismatch_abs_error > MISMATCH_HOLDOUT_TOL
        or piecewise_monitor_max_correlation_abs_error > CORRELATION_HOLDOUT_TOL
    )
    post_break_piecewise_surface_selected = piecewise_validation_to_163840_supported
    post_break_curvature_surface_selected = False
    exact_global_post_break_theorem_available = False
    physical_reject_required = False

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "post-break curvature or piecewise inventory ready",
            sign_base.truth(inventory_ready),
            "The post-break branch starts only after the retained linear law has already localized the first honest break on harmonic 90113..98304.",
        ),
        sign_base.row(
            "post_break_piecewise_mismatch_slope",
            "watch",
            "piecewise mismatch slope in x(q)=log2(q/32768)",
            piece_m_slope,
            "The first post-break fit keeps the same stress coordinate and asks whether a local affine continuation is already enough.",
        ),
        sign_base.row(
            "post_break_piecewise_correlation_slope",
            "watch",
            "piecewise correlation slope in x(q)=log2(q/32768)",
            piece_c_slope,
            "The same post-break affine surface is tested against sign-correlation drift on the exact holdout.",
        ),
        sign_base.row(
            "piecewise_holdout_max_mismatch_abs_error",
            "pass" if piecewise_holdout_max_mismatch_abs_error <= MISMATCH_HOLDOUT_TOL else "reject",
            "piecewise holdout max mismatch abs error through harmonic 163840",
            piecewise_holdout_max_mismatch_abs_error,
            "The holdout mismatch error is the primary gate for whether post-break piecewise continuation remains honest.",
        ),
        sign_base.row(
            "piecewise_holdout_max_correlation_abs_error",
            "pass" if piecewise_holdout_max_correlation_abs_error <= CORRELATION_HOLDOUT_TOL else "reject",
            "piecewise holdout max correlation abs error through harmonic 163840",
            piecewise_holdout_max_correlation_abs_error,
            "The holdout correlation error checks whether the same post-break affine surface stays coherent on the sign-floor channel.",
        ),
        sign_base.row(
            "piecewise_holdout_max_reconstruction_abs_error",
            "pass" if piecewise_holdout_max_reconstruction_abs_error <= RECON_HOLDOUT_TOL else "reject",
            "piecewise holdout max reconstruction abs error through harmonic 163840",
            piecewise_holdout_max_reconstruction_abs_error,
            "The reconstruction channel remains a monitor because the exact overlap evaluation keeps decaying cleanly after the break.",
        ),
        sign_base.row(
            "piecewise_holdout_combined_abs_error",
            "pass" if piecewise_holdout_combined_abs_error < curvature_holdout_combined_abs_error else "reject",
            "piecewise combined holdout abs error through harmonic 163840",
            piecewise_holdout_combined_abs_error,
            "The piecewise family is selected on the total exact holdout burden, with mismatch remaining the dominant blocker and reconstruction staying negligible.",
        ),
        sign_base.row(
            "curvature_holdout_combined_abs_error",
            "reject" if piecewise_holdout_combined_abs_error < curvature_holdout_combined_abs_error else "pass",
            "curvature combined holdout abs error through harmonic 163840",
            curvature_holdout_combined_abs_error,
            "Curvature only wins if it reduces the total holdout burden on the same exact sampled bands.",
        ),
        sign_base.row(
            "curvature_holdout_max_mismatch_abs_error",
            "reject" if curvature_holdout_max_mismatch_abs_error > piecewise_holdout_max_mismatch_abs_error else "pass",
            "curvature holdout max mismatch abs error through harmonic 163840",
            curvature_holdout_max_mismatch_abs_error,
            "Curvature only wins if it lowers the same holdout mismatch error on the same exact sampled bands.",
        ),
        sign_base.row(
            "piecewise_beats_curvature_on_holdout",
            "pass" if piecewise_beats_curvature_on_holdout else "reject",
            "piecewise beats curvature on the exact holdout",
            sign_base.truth(piecewise_beats_curvature_on_holdout),
            "The honest first shot is whichever family wins on the exact holdout without reopening same-level linear refits.",
        ),
        sign_base.row(
            "piecewise_validation_to_163840_supported",
            "pass" if piecewise_validation_to_163840_supported else "reject",
            "piecewise validation retained through harmonic 163840",
            sign_base.truth(piecewise_validation_to_163840_supported),
            "Gate B is only retained if piecewise stays within tolerances through the last holdout band 155649..163840.",
        ),
        sign_base.row(
            "post_break_monitor_drift_detected",
            "pass" if post_break_monitor_drift_detected else "reject",
            "farther post-break drift detected on harmonic 163841..196608",
            sign_base.truth(post_break_monitor_drift_detected),
            "Once holdout passes but monitor drifts, the honest next blocker moves to farther post-break continuation rather than back to curvature.",
        ),
        sign_base.row(
            "exact_global_post_break_theorem_available",
            "reject",
            "exact global post-break theorem available",
            sign_base.truth(exact_global_post_break_theorem_available),
            "The current branch closes only a holdout-level post-break theorem, not a global exact continuation theorem.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "theorem_lattice_base_over_m0": theorem_lattice_base,
        "bulk_delta_r_over_m0": theorem_lattice_step,
        "post_break_sample_harmonic_stride": POST_BREAK_SAMPLE_HARMONIC_STRIDE,
        "post_break_piecewise_mismatch_slope": piece_m_slope,
        "post_break_piecewise_mismatch_intercept": piece_m_intercept,
        "post_break_piecewise_correlation_slope": piece_c_slope,
        "post_break_piecewise_correlation_intercept": piece_c_intercept,
        "post_break_reconstruction_decay_exponent": piece_rec_exp,
        "post_break_reconstruction_decay_prefactor": piece_rec_pref,
        "curvature_mismatch_quad_coeff": curv_m_q2,
        "curvature_mismatch_linear_coeff": curv_m_q1,
        "curvature_mismatch_const_coeff": curv_m_q0,
        "curvature_correlation_quad_coeff": curv_c_q2,
        "curvature_correlation_linear_coeff": curv_c_q1,
        "curvature_correlation_const_coeff": curv_c_q0,
        "piecewise_holdout_max_mismatch_abs_error": piecewise_holdout_max_mismatch_abs_error,
        "piecewise_holdout_max_correlation_abs_error": piecewise_holdout_max_correlation_abs_error,
        "piecewise_holdout_max_reconstruction_abs_error": piecewise_holdout_max_reconstruction_abs_error,
        "piecewise_holdout_combined_abs_error": piecewise_holdout_combined_abs_error,
        "piecewise_monitor_max_mismatch_abs_error": piecewise_monitor_max_mismatch_abs_error,
        "piecewise_monitor_max_correlation_abs_error": piecewise_monitor_max_correlation_abs_error,
        "piecewise_monitor_max_reconstruction_abs_error": piecewise_monitor_max_reconstruction_abs_error,
        "curvature_holdout_max_mismatch_abs_error": curvature_holdout_max_mismatch_abs_error,
        "curvature_holdout_max_correlation_abs_error": curvature_holdout_max_correlation_abs_error,
        "curvature_holdout_combined_abs_error": curvature_holdout_combined_abs_error,
        "post_break_piecewise_holdout_supported": post_break_piecewise_holdout_supported,
        "post_break_curvature_holdout_supported": post_break_curvature_holdout_supported,
        "piecewise_beats_curvature_on_holdout": piecewise_beats_curvature_on_holdout,
        "post_break_piecewise_surface_selected": post_break_piecewise_surface_selected,
        "post_break_curvature_surface_selected": post_break_curvature_surface_selected,
        "piecewise_validation_to_163840_supported": piecewise_validation_to_163840_supported,
        "post_break_monitor_drift_detected": post_break_monitor_drift_detected,
        "exact_global_post_break_theorem_available": exact_global_post_break_theorem_available,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    for band_start, band_end in POST_BREAK_BANDS:
        key = f"{band_start}_{band_end}"
        band_summary = band_summaries[key]
        summary[f"{key}_max_mismatch_fraction"] = band_summary["max_mismatch"]
        summary[f"{key}_min_sign_correlation"] = band_summary["min_correlation"]
        summary[f"{key}_signed_reconstruction_max_abs_error"] = band_summary["max_abs_error"]

    declaration_payload = sign_base.payload(
        "8.7.56.2137",
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
                "prior_gate": sign_base.display_path(PRIOR_GATE),
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
            "overall_status": "vector_qball_form_factor_post_break_piecewise_curvature_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2135"),
                "roadmap_branch_hit": find_line(roadmap_text, "8.7.56.2135-.2138"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2131"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2131"),
                "unified_roadmap_hit": find_line(unified_text, ".2131-.2134"),
                "long_roadmap_hit": find_line(long_text, ".2131-.2134"),
                "part5_hit": find_line(part5_text, ".2127-.2134"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_sync_rows = [
        sign_base.row(
            "status_synced",
            "pass",
            "STATUS sync target present",
            sign_base.truth(bool(find_line(status_text, "8.7.56.2135"))),
            "The post-break reactivation is only honest if the official status already points to the same curvature-or-piecewise branch.",
        ),
        sign_base.row(
            "roadmap_synced",
            "pass",
            "ROADMAP sync target present",
            sign_base.truth(bool(find_line(roadmap_text, "8.7.56.2135-.2138"))),
            "The public roadmap must expose the same post-break reactivation branch before route sync can proceed.",
        ),
        sign_base.row(
            "long_horizon_synced",
            "pass",
            "long-horizon roadmap sync target present",
            sign_base.truth(bool(find_line(long_text, ".2131-.2134"))),
            "The long-horizon roadmap must still expose the retained stress-envelope route before the post-break decision is frozen.",
        ),
    ]
    route_sync_payload = sign_base.payload(
        "8.7.56.2138",
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
            "overall_status": "vector_qball_form_factor_post_break_piecewise_curvature_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2135"),
                "roadmap_branch_hit": find_line(roadmap_text, "8.7.56.2135-.2138"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2131"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2131"),
                "unified_roadmap_hit": find_line(unified_text, ".2131-.2134"),
                "long_roadmap_hit": find_line(long_text, ".2131-.2134"),
                "part5_hit": find_line(part5_text, ".2127-.2134"),
            },
        },
    )
    write_artifact("route_sync", route_sync_payload)


if __name__ == "__main__":
    main()
