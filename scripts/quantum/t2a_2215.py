#!/usr/bin/env python3
"""Generate 8.7.56.2215-.2218 post-break segment coefficient-law artifacts.

The post-break continuation has already realized six piecewise segments. The
route reset prioritizes extracting a law from those realized coefficients
before continuing with another farther continuation.

This branch evaluates a small library of index-laws on the realized
first-sixth coefficient series and then applies the blind sixth-step prediction
back onto the exact sixth holdout/monitor bands. If the predicted sixth
coefficients do not pass the same continuation thresholds already used in
`.2207-.2210`, then the coefficient-law route is treated as informative but
not yet theorem-level and the roadmap must fall back to the retained sixth
post-break continuation.
"""

from __future__ import annotations

import csv
import itertools
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
import scripts.quantum.t2a_2207 as sixth_base
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

FIRST_SEGMENT_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2135-2138",
        "harmonic_post_break_piecewise_curvature",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
SECOND_SEGMENT_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2143-2146",
        "harmonic_post_break_piecewise_farther",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
THIRD_SEGMENT_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2167-2170",
        "harmonic_third_post_break_piecewise_extreme_farther",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
FOURTH_SEGMENT_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2199-2202",
        "harmonic_fourth_post_break_piecewise_ultra_extreme_farther",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
FIFTH_TO_SIXTH_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2207-2210",
        "harmonic_fifth_post_break_piecewise_farther",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2215-2218"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor post-break segment "
    "coefficient-law extraction audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "harmonic_post_break_segment_coefficient_law",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_post_break_segment_"
    "coefficient_law_extraction_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_post_break_segment_"
    "coefficient_law_blind_prediction_gate_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_segment_coefficient_law_"
    "decision_gate"
)
NEXT_ROUTE = "8.7.56.2219"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_fallback_sixth_post_break_"
    "piecewise_farther_continuation_audit"
)
FOLLOWUP_ROUTE = "8.7.56.2223"

MODEL_NAMES = ("affine", "log", "inverse", "power")
COEFFICIENT_LAW_FIELD_NORM_FLOOR = 1.0e-9
COEFFICIENT_LAW_GOOD_LAST_NORM = 0.05


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


# 関数: affine model の one-step prediction を返す。

def predict_affine(x_values: np.ndarray, y_values: np.ndarray, x_target: float) -> float:
    """Return one affine least-squares prediction."""
    design = np.vstack([x_values, np.ones_like(x_values)]).T
    slope, intercept = np.linalg.lstsq(design, y_values, rcond=None)[0]
    return float((slope * x_target) + intercept)


# 関数: log-affine model の one-step prediction を返す。

def predict_log_affine(
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_target: float,
) -> float | None:
    """Return one log-affine prediction when the index grid is admissible."""
    if np.any(x_values <= 0.0) or x_target <= 0.0:
        return None

    return predict_affine(np.log(x_values), y_values, float(math.log(x_target)))


# 関数: inverse-affine model の one-step prediction を返す。

def predict_inverse_affine(
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_target: float,
) -> float | None:
    """Return one inverse-index affine prediction when admissible."""
    if np.any(x_values == 0.0) or x_target == 0.0:
        return None

    return predict_affine(1.0 / x_values, y_values, float(1.0 / x_target))


# 関数: power-law model の one-step prediction を返す。

def predict_power_law(
    x_values: np.ndarray,
    y_values: np.ndarray,
    x_target: float,
) -> float | None:
    """Return one power-law prediction for positive data only."""
    if np.any(x_values <= 0.0) or x_target <= 0.0 or np.any(y_values <= 0.0):
        return None

    log_prediction = predict_affine(
        np.log(x_values),
        np.log(y_values),
        float(math.log(x_target)),
    )
    return float(math.exp(log_prediction))


MODEL_LIBRARY = {
    "affine": predict_affine,
    "log": predict_log_affine,
    "inverse": predict_inverse_affine,
    "power": predict_power_law,
}


# 関数: sixth blind validation で使う coefficient summary を返す。
def coefficient_source_map() -> list[tuple[Path, dict[str, str]]]:
    """Return the realized first-sixth segment summary-key mapping."""
    return [
        (
            FIRST_SEGMENT_AUDIT,
            {
                "mismatch_slope": "post_break_piecewise_mismatch_slope",
                "mismatch_intercept": "post_break_piecewise_mismatch_intercept",
                "correlation_slope": "post_break_piecewise_correlation_slope",
                "correlation_intercept": "post_break_piecewise_correlation_intercept",
                "reconstruction_decay_exponent": "post_break_reconstruction_decay_exponent",
                "reconstruction_decay_prefactor": "post_break_reconstruction_decay_prefactor",
            },
        ),
        (
            SECOND_SEGMENT_AUDIT,
            {
                "mismatch_slope": "second_post_break_piecewise_mismatch_slope",
                "mismatch_intercept": "second_post_break_piecewise_mismatch_intercept",
                "correlation_slope": "second_post_break_piecewise_correlation_slope",
                "correlation_intercept": "second_post_break_piecewise_correlation_intercept",
                "reconstruction_decay_exponent": "second_post_break_reconstruction_decay_exponent",
                "reconstruction_decay_prefactor": "second_post_break_reconstruction_decay_prefactor",
            },
        ),
        (
            THIRD_SEGMENT_AUDIT,
            {
                "mismatch_slope": "third_post_break_piecewise_mismatch_slope",
                "mismatch_intercept": "third_post_break_piecewise_mismatch_intercept",
                "correlation_slope": "third_post_break_piecewise_correlation_slope",
                "correlation_intercept": "third_post_break_piecewise_correlation_intercept",
                "reconstruction_decay_exponent": "third_post_break_reconstruction_decay_exponent",
                "reconstruction_decay_prefactor": "third_post_break_reconstruction_decay_prefactor",
            },
        ),
        (
            FOURTH_SEGMENT_AUDIT,
            {
                "mismatch_slope": "fourth_post_break_piecewise_mismatch_slope",
                "mismatch_intercept": "fourth_post_break_piecewise_mismatch_intercept",
                "correlation_slope": "fourth_post_break_piecewise_correlation_slope",
                "correlation_intercept": "fourth_post_break_piecewise_correlation_intercept",
                "reconstruction_decay_exponent": "fourth_post_break_reconstruction_decay_exponent",
                "reconstruction_decay_prefactor": "fourth_post_break_reconstruction_decay_prefactor",
            },
        ),
        (
            FIFTH_TO_SIXTH_AUDIT,
            {
                "mismatch_slope": "fifth_post_break_piecewise_mismatch_slope",
                "mismatch_intercept": "fifth_post_break_piecewise_mismatch_intercept",
                "correlation_slope": "fifth_post_break_piecewise_correlation_slope",
                "correlation_intercept": "fifth_post_break_piecewise_correlation_intercept",
                "reconstruction_decay_exponent": "fifth_post_break_reconstruction_decay_exponent",
                "reconstruction_decay_prefactor": "fifth_post_break_reconstruction_decay_prefactor",
            },
        ),
        (
            FIFTH_TO_SIXTH_AUDIT,
            {
                "mismatch_slope": "sixth_post_break_piecewise_mismatch_slope",
                "mismatch_intercept": "sixth_post_break_piecewise_mismatch_intercept",
                "correlation_slope": "sixth_post_break_piecewise_correlation_slope",
                "correlation_intercept": "sixth_post_break_piecewise_correlation_intercept",
                "reconstruction_decay_exponent": "sixth_post_break_reconstruction_decay_exponent",
                "reconstruction_decay_prefactor": "sixth_post_break_reconstruction_decay_prefactor",
            },
        ),
    ]


# 関数: realized coefficient series を読み出す。

def load_coefficient_series() -> dict[str, np.ndarray]:
    """Return realized first-sixth coefficient series."""
    series = {
        "mismatch_slope": [],
        "mismatch_intercept": [],
        "correlation_slope": [],
        "correlation_intercept": [],
        "reconstruction_decay_exponent": [],
        "reconstruction_decay_prefactor": [],
    }
    for source_path, key_map in coefficient_source_map():
        summary = sign_base.read_json(source_path)["summary"]
        for field_name, summary_key in key_map.items():
            series[field_name].append(float(summary[summary_key]))

    return {field_name: np.asarray(values, dtype=float) for field_name, values in series.items()}


# 関数: one series に対する rolling blind residual を返す。

def rolling_blind_residuals(
    series: np.ndarray,
    predictor_name: str,
) -> tuple[list[float], list[float], list[float], float] | None:
    """Return predictions, residuals, normalized residuals, and the scale."""
    predictor = MODEL_LIBRARY[predictor_name]
    scale = max(float(np.max(np.abs(series))), COEFFICIENT_LAW_FIELD_NORM_FLOOR)
    predictions: list[float] = []
    residuals: list[float] = []
    normalized_residuals: list[float] = []
    segment_indices = np.arange(1.0, float(len(series)) + 1.0, dtype=float)
    for segment_count in range(3, len(series) + 1):
        prediction = predictor(
            segment_indices[: segment_count - 1],
            series[: segment_count - 1],
            float(segment_indices[segment_count - 1]),
        )
        if prediction is None or not math.isfinite(prediction):
            return None

        predictions.append(float(prediction))
        residual = float(abs(prediction - series[segment_count - 1]))
        residuals.append(residual)
        normalized_residuals.append(residual / scale)

    return predictions, residuals, normalized_residuals, scale


# 関数: one coefficient field の最良 model を返す。

def select_best_model(field_name: str, series: np.ndarray) -> dict[str, object]:
    """Return the best model payload for one coefficient field."""
    candidates: list[dict[str, object]] = []
    for predictor_name in MODEL_NAMES:
        metrics = rolling_blind_residuals(series, predictor_name)
        if metrics is None:
            continue

        predictions, residuals, normalized_residuals, scale = metrics
        candidates.append(
            {
                "field_name": field_name,
                "model_name": predictor_name,
                "predictions": predictions,
                "residuals": residuals,
                "normalized_residuals": normalized_residuals,
                "scale": scale,
                "last_prediction": float(predictions[-1]),
                "last_residual": float(residuals[-1]),
                "last_normalized_residual": float(normalized_residuals[-1]),
                "max_normalized_residual": float(max(normalized_residuals)),
            }
        )

    if not candidates:
        raise RuntimeError(f"no admissible model for field: {field_name}")

    candidates.sort(
        key=lambda item: (
            float(item["last_normalized_residual"]),
            float(item["max_normalized_residual"]),
            str(item["model_name"]),
        )
    )
    return candidates[0]


# 関数: sixth exact sampled band summaries を再構成する。

def build_sixth_band_data() -> dict[str, np.ndarray]:
    """Return exact sixth band arrays used in `.2207-.2210`."""
    qball_branch_refresh = sign_base.read_json(sixth_base.QBALL_BRANCH_REFRESH)
    scalar_ground_state = sign_base.extract_scalar_ground_state(qball_branch_refresh)
    qball_module = sign_base.load_qball_module()
    radius, field, _field_prime = qball_module.solve_full_profile(
        float(scalar_ground_state["beta_n"]),
        float(scalar_ground_state["central_amplitude"]),
    )
    weight = (field**2) * (radius**2)
    norm = float(np.trapezoid(weight, radius))
    bulk_delta_r, _bulk_fraction, _edge_gap = sixth_base.alias_base.bulk_grid_summary(radius)
    alias_1 = (2.0 * np.pi) / bulk_delta_r
    lookup_q = np.arange(
        0.0,
        sixth_base.phase_base.LOOKUP_Q_MAX + sixth_base.phase_base.LOOKUP_Q_STEP,
        sixth_base.phase_base.LOOKUP_Q_STEP,
        dtype=float,
    )
    lookup_values = sixth_base.phase_base.form_factor_array(radius, weight, norm, lookup_q)
    prior_audit_summary = sign_base.read_json(sixth_base.PRIOR_AUDIT)["summary"]
    theorem_lattice_base = float(prior_audit_summary["theorem_lattice_base_over_m0"])
    theorem_lattice_step = float(prior_audit_summary["bulk_delta_r_over_m0"])
    windows = sixth_base.sparse_base.build_sampled_windows(
        radius,
        weight,
        norm,
        alias_1,
        sixth_base.FARTHER_BANDS,
        sixth_base.FARTHER_SAMPLE_HARMONIC_STRIDE,
    )
    results = sixth_base.lattice_base.evaluate_lattice_family(
        windows,
        lookup_q,
        lookup_values,
        theorem_lattice_base,
        theorem_lattice_step,
    )
    band_summaries = {
        f"{band_start}_{band_end}": sixth_base.sparse_base.summarize_sampled_band(
            windows,
            results,
            band_start,
            band_end,
        )
        for band_start, band_end in sixth_base.FARTHER_BANDS
    }
    centers, mismatches, correlations, recon_errors = sixth_base.build_series(
        band_summaries,
        sixth_base.FARTHER_BANDS,
    )
    x_all = sixth_base.stress_log_coordinate(centers)
    return {
        "centers": centers,
        "x_all": x_all,
        "mismatches": mismatches,
        "correlations": correlations,
        "reconstruction_errors": recon_errors,
    }


# 関数: predicted sixth coefficients を current continuation gate で採点する。

def evaluate_predicted_sixth(
    band_data: dict[str, np.ndarray],
    predicted_coefficients: dict[str, float],
) -> dict[str, float]:
    """Return exact sixth holdout/monitor errors for one predicted coefficient set."""
    x_all = np.asarray(band_data["x_all"], dtype=float)
    centers = np.asarray(band_data["centers"], dtype=float)
    mismatches = np.asarray(band_data["mismatches"], dtype=float)
    correlations = np.asarray(band_data["correlations"], dtype=float)
    recon_errors = np.asarray(band_data["reconstruction_errors"], dtype=float)
    holdout_slice = slice(
        len(sixth_base.SIXTH_FIT),
        len(sixth_base.SIXTH_FIT) + len(sixth_base.SIXTH_HOLDOUT),
    )
    monitor_slice = slice(
        len(sixth_base.SIXTH_FIT) + len(sixth_base.SIXTH_HOLDOUT),
        len(sixth_base.FARTHER_BANDS),
    )
    predicted_mismatch = (
        predicted_coefficients["mismatch_slope"] * x_all
    ) + predicted_coefficients["mismatch_intercept"]
    predicted_correlation = (
        predicted_coefficients["correlation_slope"] * x_all
    ) + predicted_coefficients["correlation_intercept"]
    predicted_reconstruction = predicted_coefficients["reconstruction_decay_prefactor"] * np.power(
        centers,
        -predicted_coefficients["reconstruction_decay_exponent"],
    )
    return {
        "holdout_max_mismatch_abs_error": sixth_base.max_abs_error(
            mismatches[holdout_slice],
            predicted_mismatch[holdout_slice],
        ),
        "holdout_max_correlation_abs_error": sixth_base.max_abs_error(
            correlations[holdout_slice],
            predicted_correlation[holdout_slice],
        ),
        "holdout_max_reconstruction_abs_error": sixth_base.max_abs_error(
            recon_errors[holdout_slice],
            predicted_reconstruction[holdout_slice],
        ),
        "monitor_max_mismatch_abs_error": sixth_base.max_abs_error(
            mismatches[monitor_slice],
            predicted_mismatch[monitor_slice],
        ),
        "monitor_max_correlation_abs_error": sixth_base.max_abs_error(
            correlations[monitor_slice],
            predicted_correlation[monitor_slice],
        ),
        "monitor_max_reconstruction_abs_error": sixth_base.max_abs_error(
            recon_errors[monitor_slice],
            predicted_reconstruction[monitor_slice],
        ),
    }


# 関数: primary field 用 mixed model combo を探索する。

def search_primary_combo(
    coefficient_series: dict[str, np.ndarray],
    band_data: dict[str, np.ndarray],
    inherited_reconstruction: dict[str, float],
) -> dict[str, object]:
    """Return the best primary mixed-law combo under the sixth continuation gate."""
    segment_indices = np.arange(1.0, 6.0, dtype=float)
    best: dict[str, object] | None = None
    for mismatch_slope_model, mismatch_intercept_model, correlation_slope_model, correlation_intercept_model in itertools.product(
        MODEL_NAMES,
        repeat=4,
    ):
        predicted_coefficients = {
            "mismatch_slope": MODEL_LIBRARY[mismatch_slope_model](
                segment_indices,
                coefficient_series["mismatch_slope"][:5],
                6.0,
            ),
            "mismatch_intercept": MODEL_LIBRARY[mismatch_intercept_model](
                segment_indices,
                coefficient_series["mismatch_intercept"][:5],
                6.0,
            ),
            "correlation_slope": MODEL_LIBRARY[correlation_slope_model](
                segment_indices,
                coefficient_series["correlation_slope"][:5],
                6.0,
            ),
            "correlation_intercept": MODEL_LIBRARY[correlation_intercept_model](
                segment_indices,
                coefficient_series["correlation_intercept"][:5],
                6.0,
            ),
            "reconstruction_decay_exponent": inherited_reconstruction[
                "reconstruction_decay_exponent"
            ],
            "reconstruction_decay_prefactor": inherited_reconstruction[
                "reconstruction_decay_prefactor"
            ],
        }
        if not all(
            prediction is not None and math.isfinite(float(prediction))
            for prediction in predicted_coefficients.values()
        ):
            continue

        gate_errors = evaluate_predicted_sixth(
            band_data,
            {
                field_name: float(prediction)
                for field_name, prediction in predicted_coefficients.items()
            },
        )
        score = max(
            gate_errors["holdout_max_mismatch_abs_error"] / sixth_base.MISMATCH_TOL,
            gate_errors["holdout_max_correlation_abs_error"] / sixth_base.CORRELATION_TOL,
            gate_errors["monitor_max_mismatch_abs_error"] / sixth_base.MISMATCH_TOL,
            gate_errors["monitor_max_correlation_abs_error"] / sixth_base.CORRELATION_TOL,
        )
        payload = {
            "score": float(score),
            "model_combo": {
                "mismatch_slope": mismatch_slope_model,
                "mismatch_intercept": mismatch_intercept_model,
                "correlation_slope": correlation_slope_model,
                "correlation_intercept": correlation_intercept_model,
            },
            "predicted_coefficients": {
                field_name: float(prediction)
                for field_name, prediction in predicted_coefficients.items()
            },
            "gate_errors": gate_errors,
        }
        if best is None or payload["score"] < float(best["score"]):
            best = payload

    if best is None:
        raise RuntimeError("no admissible primary combo")

    return best


# 関数: audit で使う公式群を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the coefficient-law extraction audit."""
    return {
        "segment_family": "M_k(x)=a_M(k) x + b_M(k), C_k(x)=a_C(k) x + b_C(k), E_k(q)=A_E(k) q^{-nu_E(k)}",
        "candidate_models": "candidate index-laws = {affine in k, affine in log(k), affine in 1/k, positive power law in k}",
        "blind_rule": "fit candidate laws on realized first-five segment coefficients and predict the sixth coefficients blind before scoring on the exact sixth holdout/monitor bands",
        "primary_combo_rule": "search the best mixed law on (a_M, b_M, a_C, b_C) and keep sixth reconstruction inherited only to test whether the primary channels alone already clear the retained piecewise thresholds",
        "decision_rule": "single-theorem route is supported only if blind sixth prediction clears the same mismatch/correlation/reconstruction thresholds already used for the retained sixth post-break continuation",
    }


# 関数: `.2215-.2218` を実行する。

def main() -> None:
    """Execute the post-break segment coefficient-law extraction audit."""
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
        FIRST_SEGMENT_AUDIT,
        SECOND_SEGMENT_AUDIT,
        THIRD_SEGMENT_AUDIT,
        FOURTH_SEGMENT_AUDIT,
        FIFTH_TO_SIXTH_AUDIT,
        sixth_base.QBALL_BRANCH_REFRESH,
        sixth_base.PRIOR_AUDIT,
    ):
        sign_base.require(path)

    status_text = sign_base.read_text(STATUS)
    roadmap_text = sign_base.read_text(ROADMAP)
    current_problem_text = sign_base.read_text(CURRENT_PROBLEM)
    current_status_text = sign_base.read_text(CURRENT_STATUS)
    unified_text = sign_base.read_text(UNIFIED_ROADMAP)
    long_text = sign_base.read_text(LONG_ROADMAP)
    part5_text = sign_base.read_text(PART5)

    coefficient_series = load_coefficient_series()
    best_models = {
        field_name: select_best_model(field_name, series)
        for field_name, series in coefficient_series.items()
    }
    band_data = build_sixth_band_data()
    sixth_summary = sign_base.read_json(FIFTH_TO_SIXTH_AUDIT)["summary"]

    blind_full_prediction = {
        "mismatch_slope": float(best_models["mismatch_slope"]["last_prediction"]),
        "mismatch_intercept": float(best_models["mismatch_intercept"]["last_prediction"]),
        "correlation_slope": float(best_models["correlation_slope"]["last_prediction"]),
        "correlation_intercept": float(best_models["correlation_intercept"]["last_prediction"]),
        "reconstruction_decay_exponent": float(
            best_models["reconstruction_decay_exponent"]["last_prediction"]
        ),
        "reconstruction_decay_prefactor": float(
            best_models["reconstruction_decay_prefactor"]["last_prediction"]
        ),
    }
    blind_full_gate_errors = evaluate_predicted_sixth(band_data, blind_full_prediction)
    full_blind_single_theorem_supported = bool(
        blind_full_gate_errors["holdout_max_mismatch_abs_error"] <= sixth_base.MISMATCH_TOL
        and blind_full_gate_errors["holdout_max_correlation_abs_error"] <= sixth_base.CORRELATION_TOL
        and blind_full_gate_errors["holdout_max_reconstruction_abs_error"] <= sixth_base.RECON_TOL
        and blind_full_gate_errors["monitor_max_mismatch_abs_error"] <= sixth_base.MISMATCH_TOL
        and blind_full_gate_errors["monitor_max_correlation_abs_error"] <= sixth_base.CORRELATION_TOL
        and blind_full_gate_errors["monitor_max_reconstruction_abs_error"] <= sixth_base.RECON_TOL
    )

    inherited_reconstruction = {
        "reconstruction_decay_exponent": float(
            sixth_summary["sixth_post_break_reconstruction_decay_exponent"]
        ),
        "reconstruction_decay_prefactor": float(
            sixth_summary["sixth_post_break_reconstruction_decay_prefactor"]
        ),
    }
    best_primary_combo = search_primary_combo(
        coefficient_series,
        band_data,
        inherited_reconstruction,
    )
    primary_combo_passes_current_threshold = bool(
        best_primary_combo["gate_errors"]["holdout_max_mismatch_abs_error"]
        <= sixth_base.MISMATCH_TOL
        and best_primary_combo["gate_errors"]["holdout_max_correlation_abs_error"]
        <= sixth_base.CORRELATION_TOL
        and best_primary_combo["gate_errors"]["monitor_max_mismatch_abs_error"]
        <= sixth_base.MISMATCH_TOL
        and best_primary_combo["gate_errors"]["monitor_max_correlation_abs_error"]
        <= sixth_base.CORRELATION_TOL
    )

    reconstruction_coefficient_law_available = bool(
        best_models["reconstruction_decay_exponent"]["last_normalized_residual"]
        <= COEFFICIENT_LAW_GOOD_LAST_NORM
        and best_models["reconstruction_decay_prefactor"]["last_normalized_residual"]
        <= COEFFICIENT_LAW_GOOD_LAST_NORM
    )
    primary_coefficients_blind_predictive = bool(
        best_models["mismatch_slope"]["last_normalized_residual"] <= COEFFICIENT_LAW_GOOD_LAST_NORM
        and best_models["mismatch_intercept"]["last_normalized_residual"] <= COEFFICIENT_LAW_GOOD_LAST_NORM
        and best_models["correlation_slope"]["last_normalized_residual"] <= COEFFICIENT_LAW_GOOD_LAST_NORM
        and best_models["correlation_intercept"]["last_normalized_residual"] <= COEFFICIENT_LAW_GOOD_LAST_NORM
    )
    single_theorem_route_supported = bool(
        full_blind_single_theorem_supported
        and primary_coefficients_blind_predictive
        and reconstruction_coefficient_law_available
    )
    fallback_sixth_post_break_piecewise_selected = not single_theorem_route_supported
    physical_reject_required = False

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass",
            "post-break coefficient-law inventory ready",
            1.0,
            "The law-extraction route starts only after six realized post-break segments have already been retained and logged in public metrics artifacts.",
        ),
        sign_base.row(
            "best_mismatch_slope_model_last_norm_residual",
            "pass" if best_models["mismatch_slope"]["last_normalized_residual"] <= COEFFICIENT_LAW_GOOD_LAST_NORM else "watch",
            "best mismatch-slope model last normalized blind residual",
            float(best_models["mismatch_slope"]["last_normalized_residual"]),
            "The mismatch-slope channel is the first blocker candidate because it directly controls sixth-segment envelope drift on the retained stress coordinate.",
        ),
        sign_base.row(
            "best_mismatch_intercept_model_last_norm_residual",
            "pass" if best_models["mismatch_intercept"]["last_normalized_residual"] <= COEFFICIENT_LAW_GOOD_LAST_NORM else "watch",
            "best mismatch-intercept model last normalized blind residual",
            float(best_models["mismatch_intercept"]["last_normalized_residual"]),
            "The mismatch-intercept channel measures whether the post-break offset can be predicted rather than refit segment by segment.",
        ),
        sign_base.row(
            "best_correlation_slope_model_last_norm_residual",
            "pass" if best_models["correlation_slope"]["last_normalized_residual"] <= COEFFICIENT_LAW_GOOD_LAST_NORM else "watch",
            "best correlation-slope model last normalized blind residual",
            float(best_models["correlation_slope"]["last_normalized_residual"]),
            "The correlation-slope channel checks whether sign-floor drift is law-like on the realized segment index.",
        ),
        sign_base.row(
            "best_correlation_intercept_model_last_norm_residual",
            "pass" if best_models["correlation_intercept"]["last_normalized_residual"] <= COEFFICIENT_LAW_GOOD_LAST_NORM else "watch",
            "best correlation-intercept model last normalized blind residual",
            float(best_models["correlation_intercept"]["last_normalized_residual"]),
            "The correlation-intercept channel is the cleanest near-zero drift test because the realized sequence is already close to sign-parity closure.",
        ),
        sign_base.row(
            "reconstruction_coefficient_law_available",
            "pass" if reconstruction_coefficient_law_available else "reject",
            "reconstruction-coefficient law available",
            sign_base.truth(reconstruction_coefficient_law_available),
            "The reconstruction channel only closes if both decay exponent and prefactor are themselves blind-predictive rather than inherited monitors.",
        ),
        sign_base.row(
            "blind_full_sixth_holdout_max_mismatch_abs_error",
            "pass" if blind_full_gate_errors["holdout_max_mismatch_abs_error"] <= sixth_base.MISMATCH_TOL else "reject",
            "blind full-law sixth holdout max mismatch abs error",
            blind_full_gate_errors["holdout_max_mismatch_abs_error"],
            "The full blind law is tested on the same exact sixth holdout windows already used for the retained sixth continuation gate.",
        ),
        sign_base.row(
            "blind_full_sixth_monitor_max_mismatch_abs_error",
            "pass" if blind_full_gate_errors["monitor_max_mismatch_abs_error"] <= sixth_base.MISMATCH_TOL else "reject",
            "blind full-law sixth monitor max mismatch abs error",
            blind_full_gate_errors["monitor_max_mismatch_abs_error"],
            "The full blind law must also survive the same sixth monitor windows rather than only matching the first rescued quartet.",
        ),
        sign_base.row(
            "primary_mixed_best_holdout_max_mismatch_abs_error",
            "pass" if best_primary_combo["gate_errors"]["holdout_max_mismatch_abs_error"] <= sixth_base.MISMATCH_TOL else "reject",
            "best primary mixed-law sixth holdout max mismatch abs error",
            best_primary_combo["gate_errors"]["holdout_max_mismatch_abs_error"],
            "This row asks whether even the best primary-only mixed coefficient law would already clear the sixth holdout mismatch threshold once reconstruction is inherited rather than blind-predicted.",
        ),
        sign_base.row(
            "primary_mixed_best_monitor_max_mismatch_abs_error",
            "pass" if best_primary_combo["gate_errors"]["monitor_max_mismatch_abs_error"] <= sixth_base.MISMATCH_TOL else "reject",
            "best primary mixed-law sixth monitor max mismatch abs error",
            best_primary_combo["gate_errors"]["monitor_max_mismatch_abs_error"],
            "If this row fails, the route is not blocked by reconstruction monitors; it is blocked directly by the mismatch channel itself.",
        ),
        sign_base.row(
            "primary_mixed_coefficient_law_passes_current_piecewise_threshold",
            "pass" if primary_combo_passes_current_threshold else "reject",
            "best primary mixed coefficient law passes current piecewise threshold",
            sign_base.truth(primary_combo_passes_current_threshold),
            "This gate is the strongest honest test of whether a single coefficient-law route already beats the retained sixth post-break continuation on its own current criteria.",
        ),
        sign_base.row(
            "single_theorem_route_supported",
            "pass" if single_theorem_route_supported else "reject",
            "single-theorem route supported now",
            sign_base.truth(single_theorem_route_supported),
            "A single-theorem route is supported only if the blind sixth prediction clears the same exact holdout/monitor thresholds used by the retained continuation branch.",
        ),
        sign_base.row(
            "fallback_sixth_post_break_piecewise_selected",
            "pass" if fallback_sixth_post_break_piecewise_selected else "reject",
            "fallback sixth post-break piecewise continuation selected",
            sign_base.truth(fallback_sixth_post_break_piecewise_selected),
            "If the coefficient-law route stays non-predictive on the exact sixth gate, the roadmap must honestly fall back to the retained sixth post-break continuation family.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "segment_index_count": 6.0,
        "best_mismatch_slope_model_name": str(best_models["mismatch_slope"]["model_name"]),
        "best_mismatch_slope_last_prediction": float(best_models["mismatch_slope"]["last_prediction"]),
        "best_mismatch_slope_last_residual": float(best_models["mismatch_slope"]["last_residual"]),
        "best_mismatch_slope_last_normalized_residual": float(best_models["mismatch_slope"]["last_normalized_residual"]),
        "best_mismatch_intercept_model_name": str(best_models["mismatch_intercept"]["model_name"]),
        "best_mismatch_intercept_last_prediction": float(best_models["mismatch_intercept"]["last_prediction"]),
        "best_mismatch_intercept_last_residual": float(best_models["mismatch_intercept"]["last_residual"]),
        "best_mismatch_intercept_last_normalized_residual": float(best_models["mismatch_intercept"]["last_normalized_residual"]),
        "best_correlation_slope_model_name": str(best_models["correlation_slope"]["model_name"]),
        "best_correlation_slope_last_prediction": float(best_models["correlation_slope"]["last_prediction"]),
        "best_correlation_slope_last_residual": float(best_models["correlation_slope"]["last_residual"]),
        "best_correlation_slope_last_normalized_residual": float(best_models["correlation_slope"]["last_normalized_residual"]),
        "best_correlation_intercept_model_name": str(best_models["correlation_intercept"]["model_name"]),
        "best_correlation_intercept_last_prediction": float(best_models["correlation_intercept"]["last_prediction"]),
        "best_correlation_intercept_last_residual": float(best_models["correlation_intercept"]["last_residual"]),
        "best_correlation_intercept_last_normalized_residual": float(best_models["correlation_intercept"]["last_normalized_residual"]),
        "best_reconstruction_decay_exponent_model_name": str(best_models["reconstruction_decay_exponent"]["model_name"]),
        "best_reconstruction_decay_exponent_last_prediction": float(best_models["reconstruction_decay_exponent"]["last_prediction"]),
        "best_reconstruction_decay_exponent_last_normalized_residual": float(best_models["reconstruction_decay_exponent"]["last_normalized_residual"]),
        "best_reconstruction_decay_prefactor_model_name": str(best_models["reconstruction_decay_prefactor"]["model_name"]),
        "best_reconstruction_decay_prefactor_last_prediction": float(best_models["reconstruction_decay_prefactor"]["last_prediction"]),
        "best_reconstruction_decay_prefactor_last_normalized_residual": float(best_models["reconstruction_decay_prefactor"]["last_normalized_residual"]),
        "blind_full_prediction_mismatch_slope": blind_full_prediction["mismatch_slope"],
        "blind_full_prediction_mismatch_intercept": blind_full_prediction["mismatch_intercept"],
        "blind_full_prediction_correlation_slope": blind_full_prediction["correlation_slope"],
        "blind_full_prediction_correlation_intercept": blind_full_prediction["correlation_intercept"],
        "blind_full_prediction_reconstruction_decay_exponent": blind_full_prediction["reconstruction_decay_exponent"],
        "blind_full_prediction_reconstruction_decay_prefactor": blind_full_prediction["reconstruction_decay_prefactor"],
        "blind_full_sixth_holdout_max_mismatch_abs_error": blind_full_gate_errors["holdout_max_mismatch_abs_error"],
        "blind_full_sixth_holdout_max_correlation_abs_error": blind_full_gate_errors["holdout_max_correlation_abs_error"],
        "blind_full_sixth_holdout_max_reconstruction_abs_error": blind_full_gate_errors["holdout_max_reconstruction_abs_error"],
        "blind_full_sixth_monitor_max_mismatch_abs_error": blind_full_gate_errors["monitor_max_mismatch_abs_error"],
        "blind_full_sixth_monitor_max_correlation_abs_error": blind_full_gate_errors["monitor_max_correlation_abs_error"],
        "blind_full_sixth_monitor_max_reconstruction_abs_error": blind_full_gate_errors["monitor_max_reconstruction_abs_error"],
        "primary_combo_best_model_mismatch_slope": str(best_primary_combo["model_combo"]["mismatch_slope"]),
        "primary_combo_best_model_mismatch_intercept": str(best_primary_combo["model_combo"]["mismatch_intercept"]),
        "primary_combo_best_model_correlation_slope": str(best_primary_combo["model_combo"]["correlation_slope"]),
        "primary_combo_best_model_correlation_intercept": str(best_primary_combo["model_combo"]["correlation_intercept"]),
        "primary_combo_best_score_vs_threshold": float(best_primary_combo["score"]),
        "primary_combo_best_predicted_mismatch_slope": float(best_primary_combo["predicted_coefficients"]["mismatch_slope"]),
        "primary_combo_best_predicted_mismatch_intercept": float(best_primary_combo["predicted_coefficients"]["mismatch_intercept"]),
        "primary_combo_best_predicted_correlation_slope": float(best_primary_combo["predicted_coefficients"]["correlation_slope"]),
        "primary_combo_best_predicted_correlation_intercept": float(best_primary_combo["predicted_coefficients"]["correlation_intercept"]),
        "primary_combo_best_holdout_max_mismatch_abs_error": float(best_primary_combo["gate_errors"]["holdout_max_mismatch_abs_error"]),
        "primary_combo_best_holdout_max_correlation_abs_error": float(best_primary_combo["gate_errors"]["holdout_max_correlation_abs_error"]),
        "primary_combo_best_holdout_max_reconstruction_abs_error": float(best_primary_combo["gate_errors"]["holdout_max_reconstruction_abs_error"]),
        "primary_combo_best_monitor_max_mismatch_abs_error": float(best_primary_combo["gate_errors"]["monitor_max_mismatch_abs_error"]),
        "primary_combo_best_monitor_max_correlation_abs_error": float(best_primary_combo["gate_errors"]["monitor_max_correlation_abs_error"]),
        "primary_combo_best_monitor_max_reconstruction_abs_error": float(best_primary_combo["gate_errors"]["monitor_max_reconstruction_abs_error"]),
        "primary_coefficients_blind_predictive": primary_coefficients_blind_predictive,
        "reconstruction_coefficient_law_available": reconstruction_coefficient_law_available,
        "primary_mixed_coefficient_law_passes_current_piecewise_threshold": primary_combo_passes_current_threshold,
        "single_theorem_route_supported": single_theorem_route_supported,
        "fallback_sixth_post_break_piecewise_selected": fallback_sixth_post_break_piecewise_selected,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2217",
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
                "first_segment_audit": sign_base.display_path(FIRST_SEGMENT_AUDIT),
                "second_segment_audit": sign_base.display_path(SECOND_SEGMENT_AUDIT),
                "third_segment_audit": sign_base.display_path(THIRD_SEGMENT_AUDIT),
                "fourth_segment_audit": sign_base.display_path(FOURTH_SEGMENT_AUDIT),
                "fifth_to_sixth_audit": sign_base.display_path(FIFTH_TO_SIXTH_AUDIT),
                "qball_branch_refresh": sign_base.display_path(sixth_base.QBALL_BRANCH_REFRESH),
                "sixth_prior_audit": sign_base.display_path(sixth_base.PRIOR_AUDIT),
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
            "overall_status": "vector_qball_form_factor_segment_coefficient_law_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2215"),
                "roadmap_branch_hit": find_line(roadmap_text, ".2215-.2218"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2215"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2215"),
                "unified_roadmap_hit": find_line(unified_text, ".2215-.2218"),
                "long_roadmap_hit": find_line(long_text, ".2215-.2218"),
                "part5_hit": find_line(part5_text, ".2215-.2218"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_sync_rows = [
        sign_base.row(
            "status_synced",
            "pass",
            "STATUS sync target present",
            sign_base.truth(bool(find_line(status_text, "8.7.56.2215"))),
            "The coefficient-law extraction audit is only honest if the official status already points to the route-reset law branch.",
        ),
        sign_base.row(
            "roadmap_synced",
            "pass",
            "ROADMAP sync target present",
            sign_base.truth(bool(find_line(roadmap_text, ".2215-.2218"))),
            "The public roadmap must expose the coefficient-law extraction branch before route sync can proceed.",
        ),
        sign_base.row(
            "long_horizon_synced",
            "pass",
            "long-horizon roadmap sync target present",
            sign_base.truth(bool(find_line(long_text, ".2215-.2218"))),
            "The long-horizon roadmap must still expose the route-reset law extraction state before the decision gate is frozen.",
        ),
    ]
    route_sync_payload = sign_base.payload(
        "8.7.56.2218",
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
            "overall_status": "vector_qball_form_factor_segment_coefficient_law_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2215"),
                "roadmap_branch_hit": find_line(roadmap_text, ".2215-.2218"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2215"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2215"),
                "unified_roadmap_hit": find_line(unified_text, ".2215-.2218"),
                "long_roadmap_hit": find_line(long_text, ".2215-.2218"),
                "part5_hit": find_line(part5_text, ".2215-.2218"),
            },
        },
    )
    write_artifact("route_sync", route_sync_payload)


if __name__ == "__main__":
    main()
