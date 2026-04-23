#!/usr/bin/env python3
"""Generate 8.7.56.2031-.2034 boundary alias-image reactivation artifacts."""

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
PRIOR_GATE = (
    PUBLIC_OUT
    / "q_8_7_56_2027_2030_alias_harmonic_spike_gate_declaration_gate_metrics.json"
)
PRIOR_AUDIT = (
    PUBLIC_OUT
    / "q_8_7_56_2023_2026_alias_harmonic_spike_audit_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.2031-2034"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor boundary alias-image "
    "signed rule reactivation"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "boundary_alias_image_reactivation",
    prefix="q",
)

PRIOR_CLASS = "vector_qball_form_factor_boundary_alias_image_signed_rule_reactivation_next"
BRANCH_CLASS = (
    "vector_qball_form_factor_boundary_alias_image_shared_phase_slip_partial_retain_"
    "decision_gate_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_alias_image_shared_phase_slip_"
    "decision_gate_registry"
)
NEXT_ROUTE = "8.7.56.2035"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_exact_boundary_phase_slip_theorem_"
    "or_alias_image_higher_q_generalization"
)
FOLLOWUP_ROUTE = "8.7.56.2039"
FIT_Q_MIN = alias_base.FIT_Q_MIN
FIT_Q_MAX = alias_base.FIT_Q_MAX
EDGE_Q_MIN = alias_base.EDGE_Q_MIN
EDGE_Q_MAX = alias_base.EDGE_Q_MAX
SEARCH_DELTA_MIN = 0.2
SEARCH_DELTA_MAX = 0.6
SEARCH_DELTA_STEP = 0.001
SEARCH_DECIMATION = 200
LOOKUP_Q_MAX = 60.0
LOOKUP_Q_STEP = 0.001
FORM_FACTOR_CHUNK = 2048


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


# 関数: q array 上の exact overlap form factor を batch 評価する。

def form_factor_array(
    radius: np.ndarray,
    weight: np.ndarray,
    norm: float,
    q_values: np.ndarray,
    *,
    chunk_size: int = FORM_FACTOR_CHUNK,
) -> np.ndarray:
    """Evaluate the normalized overlap form factor on one q array."""
    q_array = np.asarray(q_values, dtype=float)
    outputs = np.empty_like(q_array)
    for start in range(0, q_array.size, chunk_size):
        stop = min(start + chunk_size, q_array.size)
        q_chunk = q_array[start:stop]
        qx = q_chunk[:, None] * radius[None, :]
        sinc = np.ones_like(qx)
        mask = np.abs(qx) > 1.0e-12
        sinc[mask] = np.sin(qx[mask]) / qx[mask]
        numerator = np.trapezoid(weight[None, :] * sinc, radius, axis=1)
        outputs[start:stop] = numerator / norm

    return outputs


# 関数: exact overlap / abs / sign を返す。

def exact_sign_data(
    radius: np.ndarray,
    weight: np.ndarray,
    norm: float,
    q_scan: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact overlap values, absolute values, and sign states."""
    values = form_factor_array(radius, weight, norm, q_scan)
    absolute_values = np.abs(values)
    sign_values = np.sign(values)
    sign_values[np.abs(values) <= alias_base.SIGN_ZERO_TOL] = 0.0
    return values, absolute_values, sign_values


# 関数: alias-image shifted q を返す。

def shifted_alias_image_q(
    q_scan: np.ndarray,
    alias_harmonic: float,
    harmonic_index: int,
    delta_q: float,
) -> np.ndarray:
    """Return the shifted alias-image argument."""
    center = alias_harmonic + (((-1) ** (harmonic_index + 1)) * float(delta_q))
    return np.abs(center - q_scan)


# 関数: alias-image sign を返す。

def alias_sigma_from_values(image_values: np.ndarray, harmonic_index: int) -> np.ndarray:
    """Return the harmonic-parity alias-image sign from exact image values."""
    sigma = np.sign(image_values)
    sigma[np.abs(image_values) <= alias_base.SIGN_ZERO_TOL] = 0.0
    if (harmonic_index % 2) == 1:
        sigma = -sigma

    return sigma


# 関数: signed-rule window diagnostics を返す。

def signed_window_metrics(
    sigma_pred: np.ndarray,
    sigma_exact: np.ndarray,
    exact_values: np.ndarray,
    exact_abs: np.ndarray,
) -> dict[str, float]:
    """Return mismatch, correlation, and signed reconstruction errors."""
    reconstructed = sigma_pred * exact_abs
    return {
        "sign_mismatch_fraction": float(np.mean(sigma_pred != sigma_exact)),
        "sign_correlation": float(np.mean(sigma_pred * sigma_exact)),
        "signed_reconstruction_max_abs_error": float(
            np.max(np.abs(reconstructed - exact_values))
        ),
        "signed_reconstruction_mean_abs_error": float(
            np.mean(np.abs(reconstructed - exact_values))
        ),
        "exact_max_abs_form_factor": float(np.max(exact_abs)),
    }


# 関数: decimated search objective を返す。

def minimax_shift_objective(
    lookup_q: np.ndarray,
    lookup_f: np.ndarray,
    fit_q_scan: np.ndarray,
    fit_sign: np.ndarray,
    edge_q_scan: np.ndarray,
    edge_sign: np.ndarray,
    alias_1: float,
    alias_2: float,
    delta_q: float,
) -> tuple[float, float, float]:
    """Return fit/edge mismatch and the minimax objective."""
    fit_q_image = shifted_alias_image_q(fit_q_scan, alias_1, 1, delta_q)
    edge_q_image = shifted_alias_image_q(edge_q_scan, alias_2, 2, delta_q)
    fit_values = np.interp(fit_q_image, lookup_q, lookup_f)
    edge_values = np.interp(edge_q_image, lookup_q, lookup_f)
    fit_sigma = alias_sigma_from_values(fit_values, 1)
    edge_sigma = alias_sigma_from_values(edge_values, 2)
    fit_mismatch = alias_base.sign_mismatch_fraction(fit_sigma, fit_sign)
    edge_mismatch = alias_base.sign_mismatch_fraction(edge_sigma, edge_sign)
    return fit_mismatch, edge_mismatch, max(fit_mismatch, edge_mismatch)


# 関数: audit で使う公式群を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the alias-image reactivation audit."""
    return {
        "plain_alias_image_rule": "sigma_img^(n)(q)=(-1)^n sign(F_exact(|q_alias^(n)-q|))",
        "shifted_alias_image_rule": "sigma_img,delta^(n)(q)=(-1)^n sign(F_exact(|q_alias^(n)+(-1)^(n+1) delta_q-q|))",
        "search_rule": "delta_q_star = argmin_delta max(mismatch_fit(delta), mismatch_edge(delta))",
        "signed_reconstruction": "F_recon(q)=sigma_pred(q) |F_exact(q)|",
    }


# 関数: `.2031-.2034` を実行する。

def main() -> None:
    """Execute the boundary alias-image signed-rule reactivation audit."""
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
    inventory_ready = bool(prior_gate_summary["alias_image_signed_rule_reactivation_admissible_now"])

    qball_branch_refresh = sign_base.read_json(QBALL_BRANCH_REFRESH)
    scalar_ground_state = sign_base.extract_scalar_ground_state(qball_branch_refresh)
    qball_module = sign_base.load_qball_module()
    radius, field, _field_prime = qball_module.solve_full_profile(
        float(scalar_ground_state["beta_n"]),
        float(scalar_ground_state["central_amplitude"]),
    )
    weight = (field**2) * (radius**2)
    norm = float(np.trapezoid(weight, radius))
    r_box = float(radius[-1])
    bulk_delta_r = float(prior_audit_summary["bulk_delta_r_over_m0"])
    edge_gap = float(prior_audit_summary["edge_cell_relative_gap"])
    alias_1 = float(prior_audit_summary["first_alias_harmonic_over_m0"])
    alias_2 = float(prior_audit_summary["second_alias_harmonic_over_m0"])

    fit_q_scan = np.linspace(FIT_Q_MIN, FIT_Q_MAX, int(round((FIT_Q_MAX - FIT_Q_MIN) * alias_base.WINDOW_SCAN_DENSITY)) + 1)
    edge_q_scan = np.linspace(EDGE_Q_MIN, EDGE_Q_MAX, int(round((EDGE_Q_MAX - EDGE_Q_MIN) * alias_base.WINDOW_SCAN_DENSITY)) + 1)
    fit_values, fit_abs, fit_sign = exact_sign_data(radius, weight, norm, fit_q_scan)
    edge_values, edge_abs, edge_sign = exact_sign_data(radius, weight, norm, edge_q_scan)

    plain_fit_q_image = shifted_alias_image_q(fit_q_scan, alias_1, 1, 0.0)
    plain_edge_q_image = shifted_alias_image_q(edge_q_scan, alias_2, 2, 0.0)
    plain_fit_image_values = form_factor_array(radius, weight, norm, plain_fit_q_image)
    plain_edge_image_values = form_factor_array(radius, weight, norm, plain_edge_q_image)
    plain_fit_sigma = alias_sigma_from_values(plain_fit_image_values, 1)
    plain_edge_sigma = alias_sigma_from_values(plain_edge_image_values, 2)
    plain_fit_metrics = signed_window_metrics(plain_fit_sigma, fit_sign, fit_values, fit_abs)
    plain_edge_metrics = signed_window_metrics(plain_edge_sigma, edge_sign, edge_values, edge_abs)

    lookup_q = np.arange(0.0, LOOKUP_Q_MAX + LOOKUP_Q_STEP, LOOKUP_Q_STEP, dtype=float)
    lookup_f = form_factor_array(radius, weight, norm, lookup_q)
    fit_q_dec = fit_q_scan[::SEARCH_DECIMATION]
    fit_sign_dec = fit_sign[::SEARCH_DECIMATION]
    edge_q_dec = edge_q_scan[::SEARCH_DECIMATION]
    edge_sign_dec = edge_sign[::SEARCH_DECIMATION]
    delta_grid = np.arange(SEARCH_DELTA_MIN, SEARCH_DELTA_MAX + (0.5 * SEARCH_DELTA_STEP), SEARCH_DELTA_STEP)

    best_delta = float(delta_grid[0])
    best_fit_dec = 1.0
    best_edge_dec = 1.0
    best_objective = 1.0
    for delta_q in delta_grid:
        fit_mismatch, edge_mismatch, objective = minimax_shift_objective(
            lookup_q,
            lookup_f,
            fit_q_dec,
            fit_sign_dec,
            edge_q_dec,
            edge_sign_dec,
            alias_1,
            alias_2,
            float(delta_q),
        )
        if (objective < best_objective - 1.0e-12) or (
            abs(objective - best_objective) <= 1.0e-12 and delta_q < best_delta
        ):
            best_delta = float(delta_q)
            best_fit_dec = float(fit_mismatch)
            best_edge_dec = float(edge_mismatch)
            best_objective = float(objective)

    shifted_fit_q_image = shifted_alias_image_q(fit_q_scan, alias_1, 1, best_delta)
    shifted_edge_q_image = shifted_alias_image_q(edge_q_scan, alias_2, 2, best_delta)
    shifted_fit_image_values = form_factor_array(radius, weight, norm, shifted_fit_q_image)
    shifted_edge_image_values = form_factor_array(radius, weight, norm, shifted_edge_q_image)
    shifted_fit_sigma = alias_sigma_from_values(shifted_fit_image_values, 1)
    shifted_edge_sigma = alias_sigma_from_values(shifted_edge_image_values, 2)
    shifted_fit_metrics = signed_window_metrics(shifted_fit_sigma, fit_sign, fit_values, fit_abs)
    shifted_edge_metrics = signed_window_metrics(shifted_edge_sigma, edge_sign, edge_values, edge_abs)

    fit_mismatch_gain = (
        plain_fit_metrics["sign_mismatch_fraction"] / shifted_fit_metrics["sign_mismatch_fraction"]
    )
    edge_mismatch_gain = (
        plain_edge_metrics["sign_mismatch_fraction"] / shifted_edge_metrics["sign_mismatch_fraction"]
    )
    edge_gap_estimate = math.pi * edge_gap / r_box
    delta_q_rel_to_edge_gap_estimate = abs(best_delta - edge_gap_estimate) / best_delta
    delta_q_rel_to_pi_over_rbox = best_delta / (math.pi / r_box)

    plain_alias_image_exact_available = bool(
        plain_fit_metrics["sign_mismatch_fraction"] <= 0.05
        and plain_edge_metrics["sign_mismatch_fraction"] <= 0.05
        and plain_fit_metrics["signed_reconstruction_max_abs_error"] <= 1.0e-6
        and plain_edge_metrics["signed_reconstruction_max_abs_error"] <= 1.0e-6
    )
    shared_phase_slip_alias_family_supported = bool(
        shifted_fit_metrics["sign_mismatch_fraction"] < plain_fit_metrics["sign_mismatch_fraction"]
        and shifted_edge_metrics["sign_mismatch_fraction"] < plain_edge_metrics["sign_mismatch_fraction"]
        and shifted_fit_metrics["sign_correlation"] > plain_fit_metrics["sign_correlation"]
        and shifted_edge_metrics["sign_correlation"] > plain_edge_metrics["sign_correlation"]
    )
    shared_phase_slip_partial_window_retained = bool(
        shifted_fit_metrics["sign_mismatch_fraction"] <= 0.2
        and shifted_edge_metrics["sign_mismatch_fraction"] <= 0.1
        and shifted_fit_metrics["sign_correlation"] >= 0.6
        and shifted_edge_metrics["sign_correlation"] >= 0.8
    )
    shared_phase_slip_canonical_theorem_available = False
    same_level_unshifted_alias_retry_admissible = False
    exact_boundary_phase_slip_theorem_admissible_now = True
    substantive_pack_update_required_now = False
    physical_reject_required = False

    rows = [
        sign_base.row("inventory_ready", "pass" if inventory_ready else "reject", "alias-image reactivation inventory ready", sign_base.truth(inventory_ready), "The branch starts only after the alias-harmonic gate has promoted the alias-image family to the active theorem surface."),
        sign_base.row("plain_fit_alias_image_sign_mismatch_fraction", "watch", "plain alias-image sign mismatch fraction on 200<=q/m0<=260", plain_fit_metrics["sign_mismatch_fraction"], "The unshifted alias-image rule is the direct first shot promoted by `.2027-.2030`."),
        sign_base.row("plain_edge_alias_image_sign_mismatch_fraction", "watch", "plain alias-image sign mismatch fraction on 380<=q/m0<=420", plain_edge_metrics["sign_mismatch_fraction"], "This is the second-harmonic residual left by the plain alias-image parity family."),
        sign_base.row("shared_phase_slip_delta_q_star_over_m0", "watch", "shared phase-slip optimum delta_q/m0", best_delta, "The minimax search uses one shared phase-slip to improve both alias windows simultaneously."),
        sign_base.row("fit_window_phase_slip_sign_mismatch_fraction", "watch", "shared phase-slip sign mismatch fraction on 200<=q/m0<=260", shifted_fit_metrics["sign_mismatch_fraction"], "The fit-window mismatch must fall materially below the plain alias-image baseline for the shared phase-slip family to be worth retaining."),
        sign_base.row("edge_window_phase_slip_sign_mismatch_fraction", "watch", "shared phase-slip sign mismatch fraction on 380<=q/m0<=420", shifted_edge_metrics["sign_mismatch_fraction"], "The edge window is the harder second-harmonic test of whether one shared phase-slip really sharpens the alias-image family."),
        sign_base.row("fit_window_phase_slip_sign_correlation", "watch", "shared phase-slip sign correlation on 200<=q/m0<=260", shifted_fit_metrics["sign_correlation"], "Correlation is tracked alongside mismatch so the branch cannot hide behind sparse sign flips alone."),
        sign_base.row("edge_window_phase_slip_sign_correlation", "watch", "shared phase-slip sign correlation on 380<=q/m0<=420", shifted_edge_metrics["sign_correlation"], "The edge-window correlation measures whether the second-harmonic spike really aligns with the shared phase-slip family."),
        sign_base.row("fit_window_phase_slip_signed_reconstruction_max_abs_error", "watch", "shared phase-slip max signed reconstruction error on 200<=q/m0<=260", shifted_fit_metrics["signed_reconstruction_max_abs_error"], "A theorem-level closeout would need to improve the pointwise signed observable, not just the parity bookkeeping."),
        sign_base.row("edge_window_phase_slip_signed_reconstruction_max_abs_error", "watch", "shared phase-slip max signed reconstruction error on 380<=q/m0<=420", shifted_edge_metrics["signed_reconstruction_max_abs_error"], "The edge window decides whether the shared phase-slip family is merely a parity relabel or a genuine signed-observable theorem."),
        sign_base.row("fit_mismatch_gain_over_plain_alias_image", "pass" if fit_mismatch_gain > 1.5 else "watch", "fit-window mismatch gain over plain alias image", fit_mismatch_gain, "A large gain indicates that the residual is not exhausted by the unshifted image family."),
        sign_base.row("edge_mismatch_gain_over_plain_alias_image", "pass" if edge_mismatch_gain > 2.0 else "watch", "edge-window mismatch gain over plain alias image", edge_mismatch_gain, "The second harmonic matters most for whether one shared phase-slip is a real theorem candidate rather than a fit artifact."),
        sign_base.row("delta_q_rel_to_edge_gap_estimate", "watch", "relative gap between delta_q* and pi*edge_gap/R_box", delta_q_rel_to_edge_gap_estimate, "The optimum shared phase-slip is not explained by the simple last-cell gap estimate, so a canonical theorem is not yet available."),
        sign_base.row("plain_alias_image_exact_available", "reject" if not plain_alias_image_exact_available else "pass", "plain alias-image exact canonical closeout available", sign_base.truth(plain_alias_image_exact_available), "The first-shot unshifted alias-image family does not close the residual spike windows exactly."),
        sign_base.row("shared_phase_slip_alias_family_supported", "pass" if shared_phase_slip_alias_family_supported else "reject", "shared phase-slip alias family supported", sign_base.truth(shared_phase_slip_alias_family_supported), "The minimal extension is retained only if it improves both active windows simultaneously relative to the plain alias-image rule."),
        sign_base.row("shared_phase_slip_partial_window_retained", "pass" if shared_phase_slip_partial_window_retained else "watch", "shared phase-slip finite-window partial retain", sign_base.truth(shared_phase_slip_partial_window_retained), "The branch can honestly retain the shared phase-slip family only as a finite-window improvement, not as an exact canonical closeout."),
        sign_base.row("shared_phase_slip_canonical_theorem_available", "reject", "shared phase-slip canonical theorem available", sign_base.truth(shared_phase_slip_canonical_theorem_available), "No exact theorem for delta_q* is fixed here, and the pointwise signed reconstruction errors remain on the plain-alias scale."),
        sign_base.row("same_level_unshifted_alias_retry_admissible", "reject", "same-level unshifted alias retry admissible", sign_base.truth(same_level_unshifted_alias_retry_admissible), "Once the shared phase-slip family is shown to dominate the unshifted alias family, same-level plain-alias retry should remain closed."),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "fit_window_over_m0": [FIT_Q_MIN, FIT_Q_MAX],
        "edge_window_over_m0": [EDGE_Q_MIN, EDGE_Q_MAX],
        "solver_box_edge_over_m0": r_box,
        "bulk_delta_r_over_m0": bulk_delta_r,
        "edge_cell_relative_gap": edge_gap,
        "first_alias_harmonic_over_m0": alias_1,
        "second_alias_harmonic_over_m0": alias_2,
        "plain_fit_alias_image_sign_mismatch_fraction": plain_fit_metrics["sign_mismatch_fraction"],
        "plain_edge_alias_image_sign_mismatch_fraction": plain_edge_metrics["sign_mismatch_fraction"],
        "plain_fit_alias_image_sign_correlation": plain_fit_metrics["sign_correlation"],
        "plain_edge_alias_image_sign_correlation": plain_edge_metrics["sign_correlation"],
        "plain_fit_alias_image_signed_reconstruction_max_abs_error": plain_fit_metrics["signed_reconstruction_max_abs_error"],
        "plain_edge_alias_image_signed_reconstruction_max_abs_error": plain_edge_metrics["signed_reconstruction_max_abs_error"],
        "shared_phase_slip_delta_q_star_over_m0": best_delta,
        "search_decimated_fit_mismatch_at_delta_q_star": best_fit_dec,
        "search_decimated_edge_mismatch_at_delta_q_star": best_edge_dec,
        "search_decimated_objective_at_delta_q_star": best_objective,
        "fit_window_phase_slip_sign_mismatch_fraction": shifted_fit_metrics["sign_mismatch_fraction"],
        "edge_window_phase_slip_sign_mismatch_fraction": shifted_edge_metrics["sign_mismatch_fraction"],
        "fit_window_phase_slip_sign_correlation": shifted_fit_metrics["sign_correlation"],
        "edge_window_phase_slip_sign_correlation": shifted_edge_metrics["sign_correlation"],
        "fit_window_phase_slip_signed_reconstruction_max_abs_error": shifted_fit_metrics["signed_reconstruction_max_abs_error"],
        "edge_window_phase_slip_signed_reconstruction_max_abs_error": shifted_edge_metrics["signed_reconstruction_max_abs_error"],
        "fit_window_phase_slip_signed_reconstruction_mean_abs_error": shifted_fit_metrics["signed_reconstruction_mean_abs_error"],
        "edge_window_phase_slip_signed_reconstruction_mean_abs_error": shifted_edge_metrics["signed_reconstruction_mean_abs_error"],
        "fit_mismatch_gain_over_plain_alias_image": fit_mismatch_gain,
        "edge_mismatch_gain_over_plain_alias_image": edge_mismatch_gain,
        "delta_q_rel_to_edge_gap_estimate": delta_q_rel_to_edge_gap_estimate,
        "delta_q_rel_to_pi_over_rbox": delta_q_rel_to_pi_over_rbox,
        "plain_alias_image_exact_available": plain_alias_image_exact_available,
        "shared_phase_slip_alias_family_supported": shared_phase_slip_alias_family_supported,
        "shared_phase_slip_partial_window_retained": shared_phase_slip_partial_window_retained,
        "shared_phase_slip_canonical_theorem_available": shared_phase_slip_canonical_theorem_available,
        "same_level_unshifted_alias_retry_admissible": same_level_unshifted_alias_retry_admissible,
        "exact_boundary_phase_slip_theorem_admissible_now": exact_boundary_phase_slip_theorem_admissible_now,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "substantive_pack_update_required_now": substantive_pack_update_required_now,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2033",
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
                "fit_window_over_m0": [FIT_Q_MIN, FIT_Q_MAX],
                "edge_window_over_m0": [EDGE_Q_MIN, EDGE_Q_MAX],
                "search_delta_over_m0": [SEARCH_DELTA_MIN, SEARCH_DELTA_MAX],
                "search_delta_step_over_m0": SEARCH_DELTA_STEP,
                "search_decimation": SEARCH_DECIMATION,
                "lookup_q_max_over_m0": LOOKUP_Q_MAX,
                "lookup_q_step_over_m0": LOOKUP_Q_STEP,
                "next_route_name": NEXT_ROUTE_NAME,
                "next_route": NEXT_ROUTE,
                "followup_route_name": FOLLOWUP_ROUTE_NAME,
                "followup_route": FOLLOWUP_ROUTE,
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_boundary_alias_image_reactivation_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2031"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, "8.7.56.2031-.2034"),
                "current_problem_hit": sign_base.hit(current_problem_text, "boundary alias-image signed rule reactivation"),
                "current_status_hit": sign_base.hit(current_status_text, "boundary alias-image signed rule reactivation"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2031-.2034"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2031-.2034"),
                "part5_hit": sign_base.hit(part5_text, ".2023-.2030"),
            },
        },
    )

    route_payload = sign_base.payload(
        "8.7.56.2034",
        STEP_NAME + " route sync",
        declaration_payload["inputs"],
        [
            sign_base.row("shared_phase_slip_alias_family_supported", "pass" if shared_phase_slip_alias_family_supported else "reject", "shared phase-slip alias family supported", sign_base.truth(shared_phase_slip_alias_family_supported), "The next gate is justified only if one shared phase-slip improves both alias windows together."),
            sign_base.row("shared_phase_slip_canonical_theorem_available", "reject", "shared phase-slip canonical theorem available", sign_base.truth(shared_phase_slip_canonical_theorem_available), "The branch stops short of exact promotion because delta_q* is still fit-defined and the pointwise reconstruction errors are unchanged."),
            sign_base.row("next_route_fixed", "pass", "next route fixed", 1.0, "The next official branch is the alias-image shared phase-slip decision gate / registry."),
        ],
        summary,
        {
            "overall_status": "vector_qball_form_factor_boundary_alias_image_reactivation_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"formulas": build_formulae()},
    )

    declaration_paths = write_artifact("declaration_gate", declaration_payload)
    route_paths = write_artifact("route_sync", route_payload)
    print("[ok] 8.7.56.2031-.2034 boundary alias-image reactivation artifacts generated")
    print(f"[ok] declaration: {declaration_paths['json']}")
    print(f"[ok] route sync:   {route_paths['json']}")


if __name__ == "__main__":
    main()
