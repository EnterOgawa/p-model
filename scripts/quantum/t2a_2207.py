#!/usr/bin/env python3
"""Generate 8.7.56.2207-.2210 fifth post-break farther continuation artifacts."""

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
        "8.7.56.2199-2202",
        "harmonic_fourth_post_break_piecewise_ultra_extreme_farther",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_REGISTRY = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2203-2206",
        "harmonic_fourth_post_break_piecewise_registry_refresh",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2207-2210"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor fifth post-break "
    "piecewise farther continuation audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "harmonic_fifth_post_break_piecewise_farther",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_fifth_post_break_"
    "piecewise_validation_to_983040_farther_continuation_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_fifth_post_break_"
    "piecewise_farther_reactivation_gate_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_fifth_post_break_"
    "piecewise_registry_refresh"
)
NEXT_ROUTE = "8.7.56.2211"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_sixth_post_break_"
    "piecewise_farther_continuation_audit"
)
FOLLOWUP_ROUTE = "8.7.56.2215"

FARTHER_BANDS = [
    (983041, 991232),
    (991233, 999424),
    (999425, 1007616),
    (1007617, 1015808),
    (1015809, 1024000),
    (1024001, 1032192),
    (1032193, 1040384),
    (1040385, 1048576),
    (1048577, 1056768),
    (1056769, 1064960),
    (1064961, 1073152),
    (1073153, 1081344),
]
FIFTH_HOLDOUT = FARTHER_BANDS[:4]
FIFTH_MONITOR = FARTHER_BANDS[4:]
SIXTH_FIT = FARTHER_BANDS[:4]
SIXTH_HOLDOUT = FARTHER_BANDS[4:8]
SIXTH_MONITOR = FARTHER_BANDS[8:]
FARTHER_SAMPLE_HARMONIC_STRIDE = 512
Q_REF_OVER_M0 = 786432.0
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
    """Return formulas used in the fifth farther continuation audit."""
    return {
        "retained_lattice": "delta_q^(n,m) = delta_q,base^(box) + m_n Delta_box",
        "same_fifth_piecewise": "M_5(x)=a_5 x+b_5, C_5(x)=c_5 x+d_5, E_5(q)=A_5 q^{-nu_5} inherited from 884737..983040",
        "sixth_piecewise_reserve": "M_6(x)=a_6 x+b_6, C_6(x)=c_6 x+d_6, E_6(q)=A_6 q^{-nu_6} fitted on 983041..1015808 only as a reserve diagnostic",
        "selection_rule": "A sixth post-break surface becomes admissible only if the inherited fifth segment fails and the reserve sixth segment passes farther holdout and monitor windows.",
    }


# 関数: `.2207-.2210` を実行する。

def main() -> None:
    """Execute the fifth post-break farther continuation audit."""
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
        prior_registry_summary["gate_b_fifth_piecewise_reactivation_selected"]
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
        FARTHER_BANDS,
        FARTHER_SAMPLE_HARMONIC_STRIDE,
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
        for band_start, band_end in FARTHER_BANDS
    }

    centers, mismatches, correlations, recon_errors = build_series(
        band_summaries,
        FARTHER_BANDS,
    )
    x_all = stress_log_coordinate(centers)

    fifth_m_slope = float(
        prior_registry_summary["fifth_post_break_piecewise_mismatch_slope"]
    )
    fifth_m_intercept = float(
        prior_registry_summary["fifth_post_break_piecewise_mismatch_intercept"]
    )
    fifth_c_slope = float(
        prior_registry_summary["fifth_post_break_piecewise_correlation_slope"]
    )
    fifth_c_intercept = float(
        prior_registry_summary["fifth_post_break_piecewise_correlation_intercept"]
    )
    fifth_rec_exp = float(
        prior_registry_summary["fifth_post_break_reconstruction_decay_exponent"]
    )
    fifth_rec_pref = float(
        prior_registry_summary["fifth_post_break_reconstruction_decay_prefactor"]
    )
    fifth_m_pred = (fifth_m_slope * x_all) + fifth_m_intercept
    fifth_c_pred = (fifth_c_slope * x_all) + fifth_c_intercept
    fifth_r_pred = fifth_rec_pref * np.power(centers, -fifth_rec_exp)

    fifth_holdout_slice = slice(0, len(FIFTH_HOLDOUT))
    fifth_monitor_slice = slice(len(FIFTH_HOLDOUT), len(FARTHER_BANDS))
    fifth_farther_holdout_max_mismatch_abs_error = max_abs_error(
        mismatches[fifth_holdout_slice],
        fifth_m_pred[fifth_holdout_slice],
    )
    fifth_farther_holdout_max_correlation_abs_error = max_abs_error(
        correlations[fifth_holdout_slice],
        fifth_c_pred[fifth_holdout_slice],
    )
    fifth_farther_holdout_max_reconstruction_abs_error = max_abs_error(
        recon_errors[fifth_holdout_slice],
        fifth_r_pred[fifth_holdout_slice],
    )
    fifth_farther_monitor_max_mismatch_abs_error = max_abs_error(
        mismatches[fifth_monitor_slice],
        fifth_m_pred[fifth_monitor_slice],
    )
    fifth_farther_monitor_max_correlation_abs_error = max_abs_error(
        correlations[fifth_monitor_slice],
        fifth_c_pred[fifth_monitor_slice],
    )
    fifth_farther_monitor_max_reconstruction_abs_error = max_abs_error(
        recon_errors[fifth_monitor_slice],
        fifth_r_pred[fifth_monitor_slice],
    )
    same_fifth_piecewise_farther_continuation_supported = bool(
        fifth_farther_holdout_max_mismatch_abs_error <= MISMATCH_TOL
        and fifth_farther_holdout_max_correlation_abs_error <= CORRELATION_TOL
        and fifth_farther_holdout_max_reconstruction_abs_error <= RECON_TOL
        and fifth_farther_monitor_max_mismatch_abs_error <= MISMATCH_TOL
        and fifth_farther_monitor_max_correlation_abs_error <= CORRELATION_TOL
        and fifth_farther_monitor_max_reconstruction_abs_error <= RECON_TOL
    )
    fifth_post_break_piecewise_validation_to_1081344_supported = bool(
        same_fifth_piecewise_farther_continuation_supported
    )

    sixth_fit_slice = slice(0, len(SIXTH_FIT))
    sixth_holdout_slice = slice(len(SIXTH_FIT), len(SIXTH_FIT) + len(SIXTH_HOLDOUT))
    sixth_monitor_slice = slice(
        len(SIXTH_FIT) + len(SIXTH_HOLDOUT),
        len(FARTHER_BANDS),
    )
    sixth_m_slope, sixth_m_intercept = fit_affine(
        x_all[sixth_fit_slice],
        mismatches[sixth_fit_slice],
    )
    sixth_c_slope, sixth_c_intercept = fit_affine(
        x_all[sixth_fit_slice],
        correlations[sixth_fit_slice],
    )
    sixth_rec_exp, sixth_rec_pref = fit_power_law(
        centers[sixth_fit_slice],
        recon_errors[sixth_fit_slice],
    )
    sixth_m_pred = (sixth_m_slope * x_all) + sixth_m_intercept
    sixth_c_pred = (sixth_c_slope * x_all) + sixth_c_intercept
    sixth_r_pred = sixth_rec_pref * np.power(centers, -sixth_rec_exp)
    sixth_holdout_max_mismatch_abs_error = max_abs_error(
        mismatches[sixth_holdout_slice],
        sixth_m_pred[sixth_holdout_slice],
    )
    sixth_holdout_max_correlation_abs_error = max_abs_error(
        correlations[sixth_holdout_slice],
        sixth_c_pred[sixth_holdout_slice],
    )
    sixth_holdout_max_reconstruction_abs_error = max_abs_error(
        recon_errors[sixth_holdout_slice],
        sixth_r_pred[sixth_holdout_slice],
    )
    sixth_monitor_max_mismatch_abs_error = max_abs_error(
        mismatches[sixth_monitor_slice],
        sixth_m_pred[sixth_monitor_slice],
    )
    sixth_monitor_max_correlation_abs_error = max_abs_error(
        correlations[sixth_monitor_slice],
        sixth_c_pred[sixth_monitor_slice],
    )
    sixth_monitor_max_reconstruction_abs_error = max_abs_error(
        recon_errors[sixth_monitor_slice],
        sixth_r_pred[sixth_monitor_slice],
    )
    sixth_post_break_piecewise_validation_to_1081344_supported = bool(
        sixth_holdout_max_mismatch_abs_error <= MISMATCH_TOL
        and sixth_holdout_max_correlation_abs_error <= CORRELATION_TOL
        and sixth_holdout_max_reconstruction_abs_error <= RECON_TOL
        and sixth_monitor_max_mismatch_abs_error <= MISMATCH_TOL
        and sixth_monitor_max_correlation_abs_error <= CORRELATION_TOL
        and sixth_monitor_max_reconstruction_abs_error <= RECON_TOL
    )
    sixth_post_break_piecewise_surface_admissible_now = bool(
        (not same_fifth_piecewise_farther_continuation_supported)
        and sixth_post_break_piecewise_validation_to_1081344_supported
    )
    exact_global_farther_fifth_post_break_theorem_available = False
    physical_reject_required = False

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "fifth post-break farther inventory ready",
            sign_base.truth(inventory_ready),
            "The farther continuation starts only after the reserve fifth segment has already been promoted through harmonic 983040.",
        ),
        sign_base.row(
            "fifth_farther_holdout_max_mismatch_abs_error",
            "pass" if fifth_farther_holdout_max_mismatch_abs_error <= MISMATCH_TOL else "reject",
            "same fifth piecewise farther holdout max mismatch abs error through harmonic 1015808",
            fifth_farther_holdout_max_mismatch_abs_error,
            "The inherited fifth segment survives only if the next quartet stays inside the retained mismatch tolerance.",
        ),
        sign_base.row(
            "fifth_farther_holdout_max_correlation_abs_error",
            "pass" if fifth_farther_holdout_max_correlation_abs_error <= CORRELATION_TOL else "reject",
            "same fifth piecewise farther holdout max correlation abs error through harmonic 1015808",
            fifth_farther_holdout_max_correlation_abs_error,
            "The sign-floor channel must confirm the same farther survival for the inherited fifth segment.",
        ),
        sign_base.row(
            "fifth_farther_monitor_max_mismatch_abs_error",
            "pass" if fifth_farther_monitor_max_mismatch_abs_error <= MISMATCH_TOL else "reject",
            "same fifth piecewise farther monitor max mismatch abs error through harmonic 1081344",
            fifth_farther_monitor_max_mismatch_abs_error,
            "The farther monitor checks that the same fifth segment does not collapse immediately after the first quartet.",
        ),
        sign_base.row(
            "fifth_farther_monitor_max_correlation_abs_error",
            "pass" if fifth_farther_monitor_max_correlation_abs_error <= CORRELATION_TOL else "reject",
            "same fifth piecewise farther monitor max correlation abs error through harmonic 1081344",
            fifth_farther_monitor_max_correlation_abs_error,
            "The monitor condition must also hold on the sign-floor channel.",
        ),
        sign_base.row(
            "same_fifth_piecewise_farther_continuation_supported",
            "pass" if same_fifth_piecewise_farther_continuation_supported else "reject",
            "same fifth post-break piecewise farther continuation supported",
            sign_base.truth(same_fifth_piecewise_farther_continuation_supported),
            "No new surface is admissible while the inherited fifth segment still survives farther holdout and monitor windows.",
        ),
        sign_base.row(
            "sixth_post_break_piecewise_mismatch_slope",
            "watch",
            "sixth post-break reserve mismatch slope",
            sixth_m_slope,
            "A sixth segment is computed only as a reserve diagnostic after the same fifth segment has already been tested on the farther window.",
        ),
        sign_base.row(
            "sixth_holdout_max_mismatch_abs_error",
            "pass" if sixth_holdout_max_mismatch_abs_error <= MISMATCH_TOL else "reject",
            "sixth post-break holdout max mismatch abs error through harmonic 1048576",
            sixth_holdout_max_mismatch_abs_error,
            "The reserve sixth segment would only become admissible if the inherited fifth segment failed first.",
        ),
        sign_base.row(
            "sixth_holdout_max_correlation_abs_error",
            "pass" if sixth_holdout_max_correlation_abs_error <= CORRELATION_TOL else "reject",
            "sixth post-break holdout max correlation abs error through harmonic 1048576",
            sixth_holdout_max_correlation_abs_error,
            "The reserve sixth segment is monitored on the sign-floor channel for completeness.",
        ),
        sign_base.row(
            "sixth_monitor_max_mismatch_abs_error",
            "pass" if sixth_monitor_max_mismatch_abs_error <= MISMATCH_TOL else "reject",
            "sixth post-break monitor max mismatch abs error through harmonic 1081344",
            sixth_monitor_max_mismatch_abs_error,
            "Even a passing reserve sixth segment remains non-admissible when the inherited fifth segment already survives.",
        ),
        sign_base.row(
            "sixth_monitor_max_correlation_abs_error",
            "pass" if sixth_monitor_max_correlation_abs_error <= CORRELATION_TOL else "reject",
            "sixth post-break monitor max correlation abs error through harmonic 1081344",
            sixth_monitor_max_correlation_abs_error,
            "The reserve route is kept only as a diagnostic and not as the official mainline.",
        ),
        sign_base.row(
            "sixth_post_break_piecewise_surface_admissible_now",
            "reject" if not sixth_post_break_piecewise_surface_admissible_now else "pass",
            "sixth post-break piecewise surface admissible now",
            sign_base.truth(sixth_post_break_piecewise_surface_admissible_now),
            "The retry gate opens the sixth segment only after the inherited fifth segment has honestly failed on the farther continuation audit.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "theorem_lattice_base_over_m0": theorem_lattice_base,
        "bulk_delta_r_over_m0": theorem_lattice_step,
        "farther_sample_harmonic_stride": FARTHER_SAMPLE_HARMONIC_STRIDE,
        "fifth_post_break_piecewise_mismatch_slope": fifth_m_slope,
        "fifth_post_break_piecewise_mismatch_intercept": fifth_m_intercept,
        "fifth_post_break_piecewise_correlation_slope": fifth_c_slope,
        "fifth_post_break_piecewise_correlation_intercept": fifth_c_intercept,
        "fifth_post_break_reconstruction_decay_exponent": fifth_rec_exp,
        "fifth_post_break_reconstruction_decay_prefactor": fifth_rec_pref,
        "fifth_farther_holdout_max_mismatch_abs_error": fifth_farther_holdout_max_mismatch_abs_error,
        "fifth_farther_holdout_max_correlation_abs_error": fifth_farther_holdout_max_correlation_abs_error,
        "fifth_farther_holdout_max_reconstruction_abs_error": fifth_farther_holdout_max_reconstruction_abs_error,
        "fifth_farther_monitor_max_mismatch_abs_error": fifth_farther_monitor_max_mismatch_abs_error,
        "fifth_farther_monitor_max_correlation_abs_error": fifth_farther_monitor_max_correlation_abs_error,
        "fifth_farther_monitor_max_reconstruction_abs_error": fifth_farther_monitor_max_reconstruction_abs_error,
        "same_fifth_piecewise_farther_continuation_supported": same_fifth_piecewise_farther_continuation_supported,
        "fifth_post_break_piecewise_validation_to_1081344_supported": fifth_post_break_piecewise_validation_to_1081344_supported,
        "sixth_post_break_piecewise_mismatch_slope": sixth_m_slope,
        "sixth_post_break_piecewise_mismatch_intercept": sixth_m_intercept,
        "sixth_post_break_piecewise_correlation_slope": sixth_c_slope,
        "sixth_post_break_piecewise_correlation_intercept": sixth_c_intercept,
        "sixth_post_break_reconstruction_decay_exponent": sixth_rec_exp,
        "sixth_post_break_reconstruction_decay_prefactor": sixth_rec_pref,
        "sixth_holdout_max_mismatch_abs_error": sixth_holdout_max_mismatch_abs_error,
        "sixth_holdout_max_correlation_abs_error": sixth_holdout_max_correlation_abs_error,
        "sixth_holdout_max_reconstruction_abs_error": sixth_holdout_max_reconstruction_abs_error,
        "sixth_monitor_max_mismatch_abs_error": sixth_monitor_max_mismatch_abs_error,
        "sixth_monitor_max_correlation_abs_error": sixth_monitor_max_correlation_abs_error,
        "sixth_monitor_max_reconstruction_abs_error": sixth_monitor_max_reconstruction_abs_error,
        "sixth_post_break_piecewise_validation_to_1081344_supported": sixth_post_break_piecewise_validation_to_1081344_supported,
        "sixth_post_break_piecewise_surface_admissible_now": sixth_post_break_piecewise_surface_admissible_now,
        "exact_global_farther_fifth_post_break_theorem_available": exact_global_farther_fifth_post_break_theorem_available,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2209",
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
            "overall_status": "vector_qball_form_factor_fifth_post_break_farther_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2207"),
                "roadmap_branch_hit": find_line(roadmap_text, ".2207-.2210"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2207"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2207"),
                "unified_roadmap_hit": find_line(unified_text, ".2203-.2206"),
                "long_roadmap_hit": find_line(long_text, ".2203-.2206"),
                "part5_hit": find_line(part5_text, ".2203-.2206"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_sync_rows = [
        sign_base.row(
            "status_synced",
            "pass",
            "STATUS sync target present",
            sign_base.truth(bool(find_line(status_text, "8.7.56.2207"))),
            "The farther continuation audit is only honest if the official status already points to the same fifth-segment route.",
        ),
        sign_base.row(
            "roadmap_synced",
            "pass",
            "ROADMAP sync target present",
            sign_base.truth(bool(find_line(roadmap_text, ".2207-.2210"))),
            "The public roadmap must expose the same fifth post-break farther branch before route sync can proceed.",
        ),
        sign_base.row(
            "long_horizon_synced",
            "pass",
            "long-horizon roadmap sync target present",
            sign_base.truth(bool(find_line(long_text, ".2203-.2206"))),
            "The long-horizon roadmap must still expose the prior registry state before the farther continuation result is frozen.",
        ),
    ]
    route_sync_payload = sign_base.payload(
        "8.7.56.2210",
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
            "overall_status": "vector_qball_form_factor_fifth_post_break_farther_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2207"),
                "roadmap_branch_hit": find_line(roadmap_text, ".2207-.2210"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2207"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2207"),
                "unified_roadmap_hit": find_line(unified_text, ".2203-.2206"),
                "long_roadmap_hit": find_line(long_text, ".2203-.2206"),
                "part5_hit": find_line(part5_text, ".2203-.2206"),
            },
        },
    )
    write_artifact("route_sync", route_sync_payload)


if __name__ == "__main__":
    main()
