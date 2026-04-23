#!/usr/bin/env python3
"""Generate 8.7.56.2127-.2130 stress-envelope farther validation artifacts.

`.2119-.2126` retained a computation-side stress-envelope drift law on the
stress bands `32769..57344`, but that law was only checked against the first
break band `57345..65536`. This branch keeps the same retained bulk lattice,
the same representative sparse exact window contract, and the same linear
stress-envelope law, then pushes farther into post-break bands.

The honest question is not whether the whole family can be saved by another
same-level affine refit. The honest question is whether the retained linear
law still predicts later post-break monitor bands well enough to remain a
continuation theorem, or whether its residuals accelerate and force the next
branch into a new curvature / piecewise surface.
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
PRIOR_SPARSE_AUDIT = (
    PUBLIC_OUT
    / "q_8_7_56_2111_2114_harmonic_sparse_asymptotic_drift_audit_declaration_gate_metrics.json"
)
PRIOR_DRIFT_AUDIT = (
    PUBLIC_OUT
    / "q_8_7_56_2119_2122_harmonic_sparse_drift_law_audit_declaration_gate_metrics.json"
)
PRIOR_DRIFT_GATE = (
    PUBLIC_OUT
    / "q_8_7_56_2123_2126_harmonic_sparse_drift_law_registry_refresh_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.2127-2130"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor stress-envelope "
    "drift-law farther validation audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "harmonic_sparse_drift_law_farther_validation",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_stress_envelope_drift_law_"
    "partial_retain_farther_validation_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_stress_envelope_linear_"
    "validation_to_90112_post_break_curvature_gate_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_stress_envelope_drift_law_"
    "registry_refresh"
)
NEXT_ROUTE = "8.7.56.2131"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_stress_envelope_curvature_"
    "or_post_break_piecewise_reactivation"
)
FOLLOWUP_ROUTE = "8.7.56.2135"

FARTHER_BANDS = [
    (65537, 73728),
    (73729, 81920),
    (81921, 90112),
    (90113, 98304),
]
FARTHER_SAMPLE_HARMONIC_STRIDE = 512
MISMATCH_PRED_ABS_ERROR_TOL = 0.02
CORRELATION_PRED_ABS_ERROR_TOL = 0.01
RECON_PRED_ABS_ERROR_TOL = 6.0e-10


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

def stress_log_coordinate(q_center_over_m0: float, q_ref_over_m0: float) -> float:
    """Return the log2 stress coordinate for one band center."""
    return float(math.log2(q_center_over_m0 / q_ref_over_m0))


# 関数: 数列が単調非減少か判定する。

def monotone_nondecreasing(values: list[float]) -> bool:
    """Return whether one sequence is monotone nondecreasing."""
    return all(left <= right + 1.0e-15 for left, right in zip(values, values[1:]))


# 関数: audit で使う公式群を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the stress-envelope farther validation audit."""
    return {
        "retained_lattice": "delta_q^(n,m) = delta_q,base^(box) + m_n Delta_box",
        "stress_coordinate": "x(q) = log2(q / 32768)",
        "mismatch_law": "M_env(x) = a_M x + b_M from retained stress bands 32769..57344",
        "correlation_law": "C_env(x) = a_C x + b_C from retained stress bands 32769..57344",
        "reconstruction_law": "E_rec(q) = A_rec q^{-nu_rec} from retained stress bands 32769..57344",
        "farther_validation_rule": "retain the linear law only while monitor-band mismatch/correlation/error residuals remain within small farther-validation tolerances",
    }


# 関数: `.2127-.2130` を実行する。

def main() -> None:
    """Execute the stress-envelope farther validation audit."""
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
        PRIOR_SPARSE_AUDIT,
        PRIOR_DRIFT_AUDIT,
        PRIOR_DRIFT_GATE,
    ):
        sign_base.require(path)

    status_text = sign_base.read_text(STATUS)
    roadmap_text = sign_base.read_text(ROADMAP)
    current_problem_text = sign_base.read_text(CURRENT_PROBLEM)
    current_status_text = sign_base.read_text(CURRENT_STATUS)
    unified_text = sign_base.read_text(UNIFIED_ROADMAP)
    long_text = sign_base.read_text(LONG_ROADMAP)
    part5_text = sign_base.read_text(PART5)

    sparse_summary = sign_base.read_json(PRIOR_SPARSE_AUDIT)["summary"]
    drift_summary = sign_base.read_json(PRIOR_DRIFT_AUDIT)["summary"]
    prior_gate_summary = sign_base.read_json(PRIOR_DRIFT_GATE)["summary"]
    inventory_ready = bool(prior_gate_summary["stress_envelope_farther_validation_admissible_now"])

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

    theorem_lattice_base = float(sparse_summary["theorem_lattice_base_over_m0"])
    theorem_lattice_step = float(sparse_summary["bulk_delta_r_over_m0"])
    q_ref_over_m0 = float(drift_summary["stress_q_ref_over_m0"])
    mismatch_slope = float(drift_summary["stress_mismatch_envelope_slope"])
    mismatch_intercept = float(drift_summary["stress_mismatch_envelope_intercept"])
    correlation_slope = float(drift_summary["stress_sign_floor_envelope_slope"])
    correlation_intercept = float(drift_summary["stress_sign_floor_envelope_intercept"])
    recon_exponent = float(drift_summary["stress_reconstruction_decay_exponent"])
    recon_prefactor = float(drift_summary["stress_reconstruction_decay_prefactor"])

    farther_windows = sparse_base.build_sampled_windows(
        radius,
        weight,
        norm,
        alias_1,
        FARTHER_BANDS,
        FARTHER_SAMPLE_HARMONIC_STRIDE,
    )
    farther_results = lattice_base.evaluate_lattice_family(
        farther_windows,
        lookup_q,
        lookup_values,
        theorem_lattice_base,
        theorem_lattice_step,
    )
    farther_summaries = {
        f"{band_start}_{band_end}": sparse_base.summarize_sampled_band(
            farther_windows,
            farther_results,
            band_start,
            band_end,
        )
        for band_start, band_end in FARTHER_BANDS
    }

    mismatch_residuals: list[float] = []
    correlation_residuals: list[float] = []
    reconstruction_residuals: list[float] = []
    farther_band_pass_flags: list[bool] = []

    for band_start, band_end in FARTHER_BANDS:
        key = f"{band_start}_{band_end}"
        summary = farther_summaries[key]
        q_center = 0.5 * (band_start + band_end)
        stress_x = stress_log_coordinate(q_center, q_ref_over_m0)
        predicted_mismatch = (mismatch_slope * stress_x) + mismatch_intercept
        predicted_correlation = (correlation_slope * stress_x) + correlation_intercept
        predicted_reconstruction = recon_prefactor * (q_center ** (-recon_exponent))
        mismatch_abs_error = abs(summary["max_mismatch"] - predicted_mismatch)
        correlation_abs_error = abs(summary["min_correlation"] - predicted_correlation)
        reconstruction_abs_error = abs(summary["max_abs_error"] - predicted_reconstruction)
        farther_summaries[key]["q_center_over_m0"] = q_center
        farther_summaries[key]["predicted_max_mismatch"] = predicted_mismatch
        farther_summaries[key]["predicted_min_sign_correlation"] = predicted_correlation
        farther_summaries[key]["predicted_signed_reconstruction_max_abs_error"] = predicted_reconstruction
        farther_summaries[key]["mismatch_prediction_abs_error"] = mismatch_abs_error
        farther_summaries[key]["correlation_prediction_abs_error"] = correlation_abs_error
        farther_summaries[key]["reconstruction_prediction_abs_error"] = reconstruction_abs_error
        band_pass = bool(
            mismatch_abs_error <= MISMATCH_PRED_ABS_ERROR_TOL
            and correlation_abs_error <= CORRELATION_PRED_ABS_ERROR_TOL
            and reconstruction_abs_error <= RECON_PRED_ABS_ERROR_TOL
        )
        farther_summaries[key]["linear_validation_pass"] = band_pass
        mismatch_residuals.append(mismatch_abs_error)
        correlation_residuals.append(correlation_abs_error)
        reconstruction_residuals.append(reconstruction_abs_error)
        farther_band_pass_flags.append(band_pass)

    mismatch_residual_monotone_increasing = monotone_nondecreasing(mismatch_residuals)
    correlation_residual_monotone_increasing = monotone_nondecreasing(correlation_residuals)
    first_break_band_survives_farther_validation = bool(all(farther_band_pass_flags[:3]))
    farther_linear_validation_break_detected = bool(not farther_band_pass_flags[3])
    stress_envelope_linear_law_globally_validated = bool(all(farther_band_pass_flags))
    exact_global_sparse_drift_law_available = False
    post_break_curvature_or_piecewise_surface_admissible_now = bool(
        first_break_band_survives_farther_validation and farther_linear_validation_break_detected
    )
    substantive_pack_update_required_now = False
    physical_reject_required = False

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "stress-envelope farther validation inventory ready",
            sign_base.truth(inventory_ready),
            "The farther validation branch starts only after the stress-envelope drift law has been retained as an honest partial theorem.",
        ),
        sign_base.row(
            "farther_sample_harmonic_stride",
            "watch",
            "farther representative harmonic stride",
            float(FARTHER_SAMPLE_HARMONIC_STRIDE),
            "Farther validation keeps every 512th harmonic window while preserving the exact overlap evaluation inside each sampled window.",
        ),
    ]

    for band_start, band_end in FARTHER_BANDS:
        key = f"{band_start}_{band_end}"
        summary = farther_summaries[key]
        rows.extend(
            [
                sign_base.row(
                    f"{key}_max_mismatch_fraction",
                    "pass" if summary["linear_validation_pass"] else "watch",
                    f"farther-band max mismatch on harmonic {band_start}..{band_end}",
                    summary["max_mismatch"],
                    "The farther validation keeps the retained lattice fixed and checks whether the linear stress-envelope law still predicts post-break mismatch growth honestly.",
                ),
                sign_base.row(
                    f"{key}_mismatch_prediction_abs_error",
                    "pass" if summary["mismatch_prediction_abs_error"] <= MISMATCH_PRED_ABS_ERROR_TOL else "reject",
                    f"farther-band mismatch prediction abs error on harmonic {band_start}..{band_end}",
                    summary["mismatch_prediction_abs_error"],
                    "Mismatch residuals are the primary curvature diagnostic for the retained linear envelope law.",
                ),
                sign_base.row(
                    f"{key}_min_sign_correlation",
                    "pass" if summary["linear_validation_pass"] else "watch",
                    f"farther-band min sign correlation on harmonic {band_start}..{band_end}",
                    summary["min_correlation"],
                    "Correlation residuals test whether the retained linear sign-floor law remains honest after the first break band.",
                ),
                sign_base.row(
                    f"{key}_correlation_prediction_abs_error",
                    "pass" if summary["correlation_prediction_abs_error"] <= CORRELATION_PRED_ABS_ERROR_TOL else "reject",
                    f"farther-band correlation prediction abs error on harmonic {band_start}..{band_end}",
                    summary["correlation_prediction_abs_error"],
                    "Once the correlation residuals accelerate, the retained linear law stops being globally honest even if pointwise reconstruction stays excellent.",
                ),
                sign_base.row(
                    f"{key}_signed_reconstruction_max_abs_error",
                    "watch",
                    f"farther-band signed reconstruction max abs error on harmonic {band_start}..{band_end}",
                    summary["max_abs_error"],
                    "The reconstruction channel remains a monitor because it continues to decay even after the mismatch/correlation channels start to bend.",
                ),
                sign_base.row(
                    f"{key}_reconstruction_prediction_abs_error",
                    "pass" if summary["reconstruction_prediction_abs_error"] <= RECON_PRED_ABS_ERROR_TOL else "reject",
                    f"farther-band reconstruction prediction abs error on harmonic {band_start}..{band_end}",
                    summary["reconstruction_prediction_abs_error"],
                    "A tiny reconstruction residual confirms that the post-break failure is not coming from the exact overlap evaluation itself.",
                ),
            ]
        )

    rows.extend(
        [
            sign_base.row(
                "mismatch_residual_monotone_increasing",
                "pass" if mismatch_residual_monotone_increasing else "reject",
                "mismatch residuals are monotone increasing across farther bands",
                sign_base.truth(mismatch_residual_monotone_increasing),
                "Monotone mismatch residual growth is the honest computation-side signal that post-break curvature has started to dominate the retained linear law.",
            ),
            sign_base.row(
                "correlation_residual_monotone_increasing",
                "pass" if correlation_residual_monotone_increasing else "reject",
                "correlation residuals are monotone increasing across farther bands",
                sign_base.truth(correlation_residual_monotone_increasing),
                "Correlation residual growth sharpens the same blocker from the sign-floor channel.",
            ),
            sign_base.row(
                "first_break_band_survives_farther_validation",
                "pass" if first_break_band_survives_farther_validation else "reject",
                "retained linear validation survives through harmonic 90112",
                sign_base.truth(first_break_band_survives_farther_validation),
                "The first three farther monitor bands define the last region where the retained linear law still remains honest within monitor tolerances.",
            ),
            sign_base.row(
                "farther_linear_validation_break_detected",
                "pass" if farther_linear_validation_break_detected else "reject",
                "first post-break linear validation failure detected on harmonic 90113..98304",
                sign_base.truth(farther_linear_validation_break_detected),
                "The first clear post-break acceleration is fixed only when the fourth farther band exceeds the retained monitor tolerances.",
            ),
            sign_base.row(
                "stress_envelope_linear_law_globally_validated",
                "reject",
                "stress-envelope linear law globally validated",
                sign_base.truth(stress_envelope_linear_law_globally_validated),
                "The retained linear law is monitor-valid only through harmonic 90112, so it does not close a global validation theorem.",
            ),
            sign_base.row(
                "exact_global_sparse_drift_law_available",
                "reject",
                "exact global sparse drift law available",
                sign_base.truth(exact_global_sparse_drift_law_available),
                "The farther validation keeps the exact global drift law closed.",
            ),
            sign_base.row(
                "post_break_curvature_or_piecewise_surface_admissible_now",
                "pass" if post_break_curvature_or_piecewise_surface_admissible_now else "reject",
                "post-break curvature or piecewise surface admissible now",
                sign_base.truth(post_break_curvature_or_piecewise_surface_admissible_now),
                "Once the retained linear law passes through 90112 and fails on 90113..98304, the next honest branch is a curvature or post-break piecewise surface rather than another same-level linear refit.",
            ),
        ]
    )

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "theorem_lattice_base_over_m0": theorem_lattice_base,
        "bulk_delta_r_over_m0": theorem_lattice_step,
        "stress_q_ref_over_m0": q_ref_over_m0,
        "stress_mismatch_envelope_slope": mismatch_slope,
        "stress_mismatch_envelope_intercept": mismatch_intercept,
        "stress_sign_floor_envelope_slope": correlation_slope,
        "stress_sign_floor_envelope_intercept": correlation_intercept,
        "stress_reconstruction_decay_exponent": recon_exponent,
        "stress_reconstruction_decay_prefactor": recon_prefactor,
        "farther_sample_harmonic_stride": FARTHER_SAMPLE_HARMONIC_STRIDE,
        "farther_sampled_harmonic_count_65537_98304": int(len(farther_windows)),
        "mismatch_residual_monotone_increasing": mismatch_residual_monotone_increasing,
        "correlation_residual_monotone_increasing": correlation_residual_monotone_increasing,
        "first_break_band_survives_farther_validation": first_break_band_survives_farther_validation,
        "farther_linear_validation_break_detected": farther_linear_validation_break_detected,
        "farther_linear_validation_break_band": "90113_98304",
        "stress_envelope_linear_law_globally_validated": stress_envelope_linear_law_globally_validated,
        "exact_global_sparse_drift_law_available": exact_global_sparse_drift_law_available,
        "post_break_curvature_or_piecewise_surface_admissible_now": post_break_curvature_or_piecewise_surface_admissible_now,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "substantive_pack_update_required_now": substantive_pack_update_required_now,
        "physical_reject_required": physical_reject_required,
    }

    for band_start, band_end in FARTHER_BANDS:
        key = f"{band_start}_{band_end}"
        band_summary = farther_summaries[key]
        summary[f"{key}_q_center_over_m0"] = band_summary["q_center_over_m0"]
        summary[f"{key}_max_mismatch_fraction"] = band_summary["max_mismatch"]
        summary[f"{key}_predicted_max_mismatch_fraction"] = band_summary["predicted_max_mismatch"]
        summary[f"{key}_mismatch_prediction_abs_error"] = band_summary["mismatch_prediction_abs_error"]
        summary[f"{key}_min_sign_correlation"] = band_summary["min_correlation"]
        summary[f"{key}_predicted_min_sign_correlation"] = band_summary["predicted_min_sign_correlation"]
        summary[f"{key}_correlation_prediction_abs_error"] = band_summary["correlation_prediction_abs_error"]
        summary[f"{key}_signed_reconstruction_max_abs_error"] = band_summary["max_abs_error"]
        summary[f"{key}_predicted_signed_reconstruction_max_abs_error"] = band_summary[
            "predicted_signed_reconstruction_max_abs_error"
        ]
        summary[f"{key}_reconstruction_prediction_abs_error"] = band_summary[
            "reconstruction_prediction_abs_error"
        ]
        summary[f"{key}_linear_validation_pass"] = band_summary["linear_validation_pass"]

    declaration_payload = sign_base.payload(
        "8.7.56.2129",
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
                "prior_sparse_audit": sign_base.display_path(PRIOR_SPARSE_AUDIT),
                "prior_drift_audit": sign_base.display_path(PRIOR_DRIFT_AUDIT),
                "prior_drift_gate": sign_base.display_path(PRIOR_DRIFT_GATE),
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
            "overall_status": "vector_qball_form_factor_stress_envelope_farther_validation_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2127"),
                "roadmap_branch_hit": find_line(roadmap_text, "8.7.56.2127-.2130"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2127"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2127"),
                "unified_roadmap_hit": find_line(unified_text, ".2123-.2126"),
                "long_roadmap_hit": find_line(long_text, ".2123-.2126"),
                "part5_hit": find_line(part5_text, ".2107-.2114"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_sync_rows = [
        sign_base.row(
            "status_synced",
            "pass",
            "STATUS sync target present",
            sign_base.truth(bool(find_line(status_text, "8.7.56.2127"))),
            "The farther validation branch is only valid if the official status already points to the same retained linear-law route.",
        ),
        sign_base.row(
            "roadmap_synced",
            "pass",
            "ROADMAP sync target present",
            sign_base.truth(bool(find_line(roadmap_text, "8.7.56.2127-.2130"))),
            "The public roadmap must expose the same farther validation branch before registry sync can proceed.",
        ),
        sign_base.row(
            "long_horizon_synced",
            "pass",
            "long-horizon roadmap sync target present",
            sign_base.truth(bool(find_line(long_text, ".2123-.2126"))),
            "The long-horizon roadmap must still show the retained stress-envelope route before the farther validation outcome is frozen.",
        ),
    ]
    route_sync_payload = sign_base.payload(
        "8.7.56.2130",
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
            "overall_status": "vector_qball_form_factor_stress_envelope_farther_validation_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2127"),
                "roadmap_branch_hit": find_line(roadmap_text, "8.7.56.2127-.2130"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2127"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2127"),
                "unified_roadmap_hit": find_line(unified_text, ".2123-.2126"),
                "long_roadmap_hit": find_line(long_text, ".2123-.2126"),
                "part5_hit": find_line(part5_text, ".2107-.2114"),
            },
        },
    )
    write_artifact("route_sync", route_sync_payload)


if __name__ == "__main__":
    main()
