#!/usr/bin/env python3
"""Generate 8.7.56.2143-.2146 farther post-break piecewise artifacts."""

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
        "8.7.56.2135-2138",
        "harmonic_post_break_piecewise_curvature",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2139-2142",
        "harmonic_post_break_piecewise_registry",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2143-2146"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor post-break piecewise "
    "farther continuation audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "harmonic_post_break_piecewise_farther",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_post_break_piecewise_"
    "validation_to_163840_farther_continuation_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_second_post_break_"
    "piecewise_reactivation_gate_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_post_break_piecewise_"
    "registry_refresh"
)
NEXT_ROUTE = "8.7.56.2147"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_second_post_break_piecewise_"
    "farther_continuation_audit"
)
FOLLOWUP_ROUTE = "8.7.56.2151"

FARTHER_BANDS = [
    (196609, 204800),
    (204801, 212992),
    (212993, 221184),
    (221185, 229376),
    (229377, 237568),
    (237569, 245760),
    (245761, 253952),
    (253953, 262144),
    (262145, 270336),
    (270337, 278528),
    (278529, 286720),
    (286721, 294912),
]
INHERITED_HOLDOUT = FARTHER_BANDS[:4]
SECOND_FIT = FARTHER_BANDS[:4]
SECOND_HOLDOUT = FARTHER_BANDS[4:8]
SECOND_MONITOR = FARTHER_BANDS[8:]
FARTHER_SAMPLE_HARMONIC_STRIDE = 512
Q_REF_OVER_M0 = 32768.0
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
    """Return formulas used in the farther continuation audit."""
    return {
        "retained_lattice": "delta_q^(n,m) = delta_q,base^(box) + m_n Delta_box",
        "inherited_piecewise": "M_1(x)=a_1 x+b_1, C_1(x)=c_1 x+d_1, E_1(q)=A_1 q^{-nu_1} inherited from 90113..163840",
        "second_piecewise": "M_2(x)=a_2 x+b_2, C_2(x)=c_2 x+d_2, E_2(q)=A_2 q^{-nu_2} fitted on 196609..229376",
        "selection_rule": "route reset is honest only when inherited continuation fails on the first farther holdout and the second piecewise segment passes both farther holdout and monitor windows",
    }


# 関数: `.2143-.2146` を実行する。

def main() -> None:
    """Execute the farther post-break piecewise continuation audit."""
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
    inventory_ready = bool(prior_gate_summary["gate_b_piecewise_validation_to_163840_retained"])

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
    inherited_m_slope = float(prior_audit_summary["post_break_piecewise_mismatch_slope"])
    inherited_m_intercept = float(prior_audit_summary["post_break_piecewise_mismatch_intercept"])
    inherited_c_slope = float(prior_audit_summary["post_break_piecewise_correlation_slope"])
    inherited_c_intercept = float(prior_audit_summary["post_break_piecewise_correlation_intercept"])
    inherited_rec_exp = float(prior_audit_summary["post_break_reconstruction_decay_exponent"])
    inherited_rec_pref = float(prior_audit_summary["post_break_reconstruction_decay_prefactor"])
    inherited_m_pred = (inherited_m_slope * x_all) + inherited_m_intercept
    inherited_c_pred = (inherited_c_slope * x_all) + inherited_c_intercept
    inherited_r_pred = inherited_rec_pref * np.power(centers, -inherited_rec_exp)

    holdout_slice = slice(0, len(INHERITED_HOLDOUT))
    inherited_holdout_max_mismatch_abs_error = max_abs_error(
        mismatches[holdout_slice],
        inherited_m_pred[holdout_slice],
    )
    inherited_holdout_max_correlation_abs_error = max_abs_error(
        correlations[holdout_slice],
        inherited_c_pred[holdout_slice],
    )
    inherited_holdout_max_reconstruction_abs_error = max_abs_error(
        recon_errors[holdout_slice],
        inherited_r_pred[holdout_slice],
    )
    inherited_piecewise_farther_continuation_supported = bool(
        inherited_holdout_max_mismatch_abs_error <= MISMATCH_TOL
        and inherited_holdout_max_correlation_abs_error <= CORRELATION_TOL
        and inherited_holdout_max_reconstruction_abs_error <= RECON_TOL
    )

    second_fit_slice = slice(0, len(SECOND_FIT))
    second_holdout_slice = slice(len(SECOND_FIT), len(SECOND_FIT) + len(SECOND_HOLDOUT))
    second_monitor_slice = slice(
        len(SECOND_FIT) + len(SECOND_HOLDOUT),
        len(FARTHER_BANDS),
    )
    second_m_slope, second_m_intercept = fit_affine(
        x_all[second_fit_slice],
        mismatches[second_fit_slice],
    )
    second_c_slope, second_c_intercept = fit_affine(
        x_all[second_fit_slice],
        correlations[second_fit_slice],
    )
    second_rec_exp, second_rec_pref = fit_power_law(
        centers[second_fit_slice],
        recon_errors[second_fit_slice],
    )
    second_m_pred = (second_m_slope * x_all) + second_m_intercept
    second_c_pred = (second_c_slope * x_all) + second_c_intercept
    second_r_pred = second_rec_pref * np.power(centers, -second_rec_exp)
    second_holdout_max_mismatch_abs_error = max_abs_error(
        mismatches[second_holdout_slice],
        second_m_pred[second_holdout_slice],
    )
    second_holdout_max_correlation_abs_error = max_abs_error(
        correlations[second_holdout_slice],
        second_c_pred[second_holdout_slice],
    )
    second_holdout_max_reconstruction_abs_error = max_abs_error(
        recon_errors[second_holdout_slice],
        second_r_pred[second_holdout_slice],
    )
    second_monitor_max_mismatch_abs_error = max_abs_error(
        mismatches[second_monitor_slice],
        second_m_pred[second_monitor_slice],
    )
    second_monitor_max_correlation_abs_error = max_abs_error(
        correlations[second_monitor_slice],
        second_c_pred[second_monitor_slice],
    )
    second_monitor_max_reconstruction_abs_error = max_abs_error(
        recon_errors[second_monitor_slice],
        second_r_pred[second_monitor_slice],
    )
    second_post_break_piecewise_validation_to_294912_supported = bool(
        second_holdout_max_mismatch_abs_error <= MISMATCH_TOL
        and second_holdout_max_correlation_abs_error <= CORRELATION_TOL
        and second_holdout_max_reconstruction_abs_error <= RECON_TOL
        and second_monitor_max_mismatch_abs_error <= MISMATCH_TOL
        and second_monitor_max_correlation_abs_error <= CORRELATION_TOL
        and second_monitor_max_reconstruction_abs_error <= RECON_TOL
    )
    second_post_break_piecewise_surface_admissible_now = bool(
        (not inherited_piecewise_farther_continuation_supported)
        and second_post_break_piecewise_validation_to_294912_supported
    )
    exact_global_farther_post_break_theorem_available = False
    physical_reject_required = False

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "farther post-break piecewise inventory ready",
            sign_base.truth(inventory_ready),
            "The farther continuation branch starts only after the first post-break piecewise theorem has already been retained through harmonic 163840.",
        ),
        sign_base.row(
            "inherited_holdout_max_mismatch_abs_error",
            "reject" if inherited_holdout_max_mismatch_abs_error > MISMATCH_TOL else "pass",
            "inherited piecewise farther holdout max mismatch abs error",
            inherited_holdout_max_mismatch_abs_error,
            "The inherited piecewise continuation is blocked as soon as the first farther quartet exceeds the retained holdout tolerance.",
        ),
        sign_base.row(
            "inherited_holdout_max_correlation_abs_error",
            "reject" if inherited_holdout_max_correlation_abs_error > CORRELATION_TOL else "pass",
            "inherited piecewise farther holdout max correlation abs error",
            inherited_holdout_max_correlation_abs_error,
            "The sign-floor channel confirms the same farther-blocker on the inherited piecewise surface.",
        ),
        sign_base.row(
            "inherited_piecewise_farther_continuation_supported",
            "reject" if not inherited_piecewise_farther_continuation_supported else "pass",
            "inherited piecewise farther continuation supported",
            sign_base.truth(inherited_piecewise_farther_continuation_supported),
            "The same post-break piecewise surface is retained only if the first farther quartet still stays inside the inherited tolerances.",
        ),
        sign_base.row(
            "second_post_break_piecewise_mismatch_slope",
            "watch",
            "second post-break piecewise mismatch slope",
            second_m_slope,
            "Once inherited continuation fails, the honest new first shot is a second piecewise segment fitted on the first blocked quartet.",
        ),
        sign_base.row(
            "second_post_break_piecewise_correlation_slope",
            "watch",
            "second post-break piecewise correlation slope",
            second_c_slope,
            "The second piecewise segment must fit both mismatch and sign-floor drift on the same blocked quartet.",
        ),
        sign_base.row(
            "second_holdout_max_mismatch_abs_error",
            "pass" if second_holdout_max_mismatch_abs_error <= MISMATCH_TOL else "reject",
            "second piecewise holdout max mismatch abs error through harmonic 262144",
            second_holdout_max_mismatch_abs_error,
            "The new segment only counts as honest if it rescues the next quartet rather than merely refitting the blocked quartet.",
        ),
        sign_base.row(
            "second_holdout_max_correlation_abs_error",
            "pass" if second_holdout_max_correlation_abs_error <= CORRELATION_TOL else "reject",
            "second piecewise holdout max correlation abs error through harmonic 262144",
            second_holdout_max_correlation_abs_error,
            "The correlation holdout confirms whether the second segment is coherent on the sign-floor channel.",
        ),
        sign_base.row(
            "second_monitor_max_mismatch_abs_error",
            "pass" if second_monitor_max_mismatch_abs_error <= MISMATCH_TOL else "reject",
            "second piecewise monitor max mismatch abs error through harmonic 294912",
            second_monitor_max_mismatch_abs_error,
            "The farther monitor checks that the new segment does not collapse immediately after the first rescued quartet.",
        ),
        sign_base.row(
            "second_monitor_max_correlation_abs_error",
            "pass" if second_monitor_max_correlation_abs_error <= CORRELATION_TOL else "reject",
            "second piecewise monitor max correlation abs error through harmonic 294912",
            second_monitor_max_correlation_abs_error,
            "The same monitor condition must hold on the sign-floor channel as well.",
        ),
        sign_base.row(
            "second_post_break_piecewise_validation_to_294912_supported",
            "pass" if second_post_break_piecewise_validation_to_294912_supported else "reject",
            "second post-break piecewise validation retained through harmonic 294912",
            sign_base.truth(second_post_break_piecewise_validation_to_294912_supported),
            "A route reset is honest only if the second segment survives both holdout and monitor windows.",
        ),
        sign_base.row(
            "second_post_break_piecewise_surface_admissible_now",
            "pass" if second_post_break_piecewise_surface_admissible_now else "reject",
            "second post-break piecewise surface admissible now",
            sign_base.truth(second_post_break_piecewise_surface_admissible_now),
            "Once inherited continuation fails and the second segment passes, the official mainline can move to the second post-break piecewise family.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "theorem_lattice_base_over_m0": theorem_lattice_base,
        "bulk_delta_r_over_m0": theorem_lattice_step,
        "farther_sample_harmonic_stride": FARTHER_SAMPLE_HARMONIC_STRIDE,
        "inherited_piecewise_mismatch_slope": inherited_m_slope,
        "inherited_piecewise_mismatch_intercept": inherited_m_intercept,
        "inherited_piecewise_correlation_slope": inherited_c_slope,
        "inherited_piecewise_correlation_intercept": inherited_c_intercept,
        "inherited_piecewise_reconstruction_decay_exponent": inherited_rec_exp,
        "inherited_piecewise_reconstruction_decay_prefactor": inherited_rec_pref,
        "inherited_holdout_max_mismatch_abs_error": inherited_holdout_max_mismatch_abs_error,
        "inherited_holdout_max_correlation_abs_error": inherited_holdout_max_correlation_abs_error,
        "inherited_holdout_max_reconstruction_abs_error": inherited_holdout_max_reconstruction_abs_error,
        "inherited_piecewise_farther_continuation_supported": inherited_piecewise_farther_continuation_supported,
        "second_post_break_piecewise_mismatch_slope": second_m_slope,
        "second_post_break_piecewise_mismatch_intercept": second_m_intercept,
        "second_post_break_piecewise_correlation_slope": second_c_slope,
        "second_post_break_piecewise_correlation_intercept": second_c_intercept,
        "second_post_break_reconstruction_decay_exponent": second_rec_exp,
        "second_post_break_reconstruction_decay_prefactor": second_rec_pref,
        "second_holdout_max_mismatch_abs_error": second_holdout_max_mismatch_abs_error,
        "second_holdout_max_correlation_abs_error": second_holdout_max_correlation_abs_error,
        "second_holdout_max_reconstruction_abs_error": second_holdout_max_reconstruction_abs_error,
        "second_monitor_max_mismatch_abs_error": second_monitor_max_mismatch_abs_error,
        "second_monitor_max_correlation_abs_error": second_monitor_max_correlation_abs_error,
        "second_monitor_max_reconstruction_abs_error": second_monitor_max_reconstruction_abs_error,
        "second_post_break_piecewise_validation_to_294912_supported": second_post_break_piecewise_validation_to_294912_supported,
        "second_post_break_piecewise_surface_admissible_now": second_post_break_piecewise_surface_admissible_now,
        "exact_global_farther_post_break_theorem_available": exact_global_farther_post_break_theorem_available,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2145",
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
            "overall_status": "vector_qball_form_factor_post_break_farther_piecewise_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2143"),
                "roadmap_branch_hit": find_line(roadmap_text, "8.7.56.2143-.2146"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2143"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2143"),
                "unified_roadmap_hit": find_line(unified_text, ".2139-.2142"),
                "long_roadmap_hit": find_line(long_text, ".2139-.2142"),
                "part5_hit": find_line(part5_text, ".2135-.2142"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_sync_rows = [
        sign_base.row(
            "status_synced",
            "pass",
            "STATUS sync target present",
            sign_base.truth(bool(find_line(status_text, "8.7.56.2143"))),
            "The farther continuation audit is only honest if the official status already points to the same inherited piecewise route.",
        ),
        sign_base.row(
            "roadmap_synced",
            "pass",
            "ROADMAP sync target present",
            sign_base.truth(bool(find_line(roadmap_text, "8.7.56.2143-.2146"))),
            "The public roadmap must expose the same farther continuation branch before route sync can proceed.",
        ),
        sign_base.row(
            "long_horizon_synced",
            "pass",
            "long-horizon roadmap sync target present",
            sign_base.truth(bool(find_line(long_text, ".2139-.2142"))),
            "The long-horizon roadmap must still expose the first post-break piecewise route before the new route reset is frozen.",
        ),
    ]
    route_sync_payload = sign_base.payload(
        "8.7.56.2146",
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
            "overall_status": "vector_qball_form_factor_post_break_farther_piecewise_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2143"),
                "roadmap_branch_hit": find_line(roadmap_text, "8.7.56.2143-.2146"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2143"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2143"),
                "unified_roadmap_hit": find_line(unified_text, ".2139-.2142"),
                "long_roadmap_hit": find_line(long_text, ".2139-.2142"),
                "part5_hit": find_line(part5_text, ".2135-.2142"),
            },
        },
    )
    write_artifact("route_sync", route_sync_payload)


if __name__ == "__main__":
    main()
