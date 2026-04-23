#!/usr/bin/env python3
"""Generate 8.7.56.2007-.2010 boundary phase-curvature higher-q extension artifacts."""

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


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
import scripts.quantum.t2a_1963 as asymp_base
import scripts.quantum.t2a_1975 as local_jet_base
import scripts.quantum.t2a_1983 as ext_base
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
    / "q_8_7_56_2003_2006_boundary_phase_curvature_gate_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.2007-2010"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor boundary phase-curvature "
    "higher-q extension audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "boundary_phase_curvature_higher_q_ext_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_boundary_phase_curvature_window_120_to_200_"
    "large_coefficient_partial_retain_higher_q_extension_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_boundary_phase_curvature_farther_high_q_"
    "unresolved_sign_root_floor_gate_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_farther_high_q_sign_root_"
    "decision_gate_registry"
)
NEXT_ROUTE = "8.7.56.2011"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_resolved_high_q_sign_root_"
    "floor_envelope_microphase_audit"
)
FOLLOWUP_ROUTE = "8.7.56.2015"
FIT_Q_MIN = 200.0
FIT_Q_MAX = 260.0
FLOOR_Q_MAX = 320.0
MICRO_Q_MAX = 380.0
EDGE_Q_MAX = 420.0


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


# 関数: retained 4-term carrier の zero equation を返す。

def shifted_zero_equation(
    q_ratio: float,
    h0: float,
    h1: float,
    h2: float,
    r_box: float,
    phi0: float,
    phi_inv: float,
    delta_r: float,
    phi_inv2: float,
) -> float:
    """Return the boundary phase-curvature zero equation."""
    phase = (
        q_ratio * r_box
        + phi0
        + (phi_inv / q_ratio)
        + (delta_r * q_ratio)
        + (phi_inv2 / (q_ratio * q_ratio))
    )
    return (
        (-h0 * q_ratio * q_ratio + h2) * math.cos(phase)
        + (h1 * q_ratio) * math.sin(phase)
    )


# 関数: one q interval で custom carrier の root lattice を返す。

def find_custom_zeros(
    q_min: float,
    q_max: float,
    equation,
    r_box: float,
) -> np.ndarray:
    """Locate all custom zero roots on one q interval."""
    roots: list[float] = []
    max_mode = int(math.ceil(q_max * r_box / math.pi)) + 4
    for mode_index in range(max_mode):
        intervals = [
            (
                (mode_index * math.pi / r_box) + 1.0e-6,
                ((mode_index + 0.5) * math.pi / r_box) - 1.0e-6,
            ),
            (
                ((mode_index + 0.5) * math.pi / r_box) + 1.0e-6,
                ((mode_index + 1.0) * math.pi / r_box) - 1.0e-6,
            ),
        ]
        for left, right in intervals:
            if right <= q_min or left >= q_max:
                continue

            f_left = equation(left)
            f_right = equation(right)
            if not (math.isfinite(f_left) and math.isfinite(f_right)):
                continue

            if f_left * f_right >= 0.0:
                continue

            root = float(brentq(equation, left, right))
            if not roots or abs(root - roots[-1]) > 1.0e-6:
                roots.append(root)

    return np.array(roots, dtype=float)


# 関数: root spacing の統計を返す。

def spacing_stats(roots: np.ndarray, spacing_ref: float) -> dict[str, float]:
    """Return spacing diagnostics for one zero lattice."""
    if roots.size < 2:
        return {
            "spacing_count": 0.0,
            "mean_spacing": math.nan,
            "max_spacing_rel_gap_vs_pi_over_rbox": math.nan,
        }

    spacing = np.diff(roots)
    rel_gap = np.abs((spacing / spacing_ref) - 1.0)
    return {
        "spacing_count": float(spacing.size),
        "mean_spacing": float(np.mean(spacing)),
        "max_spacing_rel_gap_vs_pi_over_rbox": float(np.max(rel_gap)),
    }


# 関数: one q window の rule metrics を返す。

def evaluate_window(
    radius: np.ndarray,
    weight: np.ndarray,
    norm: float,
    exact_roots_all: np.ndarray,
    predicted_roots_all: np.ndarray,
    q_min: float,
    q_max: float,
    spacing_ref: float,
) -> dict[str, float]:
    """Return root, sign, spacing, and amplitude diagnostics on one q window."""
    exact_window = exact_roots_all[(exact_roots_all >= q_min) & (exact_roots_all <= q_max)]
    predicted_window = predicted_roots_all[
        (predicted_roots_all >= q_min) & (predicted_roots_all <= q_max)
    ]
    root_stats = ext_base.nearest_neighbor_stats(exact_window, predicted_window)
    exact_spacing = spacing_stats(exact_window, spacing_ref)
    q_scan = np.linspace(q_min, q_max, 120001)
    form_factor_scan = np.array(
        [sign_base.form_factor(radius, weight, norm, float(value)) for value in q_scan],
        dtype=float,
    )
    absolute_scan = np.abs(form_factor_scan)
    prior_zero_count = int(np.count_nonzero(exact_roots_all < q_min))
    reconstruction = local_jet_base.evaluate_rule_window(
        q_scan,
        form_factor_scan,
        absolute_scan,
        prior_zero_count,
        predicted_window,
    )
    return {
        "exact_zero_count": float(exact_window.size),
        "predicted_zero_count": float(predicted_window.size),
        **root_stats,
        **exact_spacing,
        "sign_mismatch_fraction": reconstruction["sign_mismatch_fraction"],
        "signed_reconstruction_max_abs_error": reconstruction["max_abs_error"],
        "signed_reconstruction_mean_abs_error": reconstruction["mean_abs_error"],
        "max_abs_form_factor": float(np.max(absolute_scan)),
        "mean_abs_form_factor": float(np.mean(absolute_scan)),
    }


# 関数: audit 用の公式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the farther high-q extension audit."""
    return {
        "retained_phase_curvature_rule": "phi_4(q)=phi0 + phi_-1/q + dR q + phi_-2/q^2",
        "boundary_zero_equation": "(-h0 q^2 + h2) cos(q R_box + phi_4(q)) + h1 q sin(q R_box + phi_4(q)) = 0",
        "sign_root_floor_read": "if sign mismatches become O(1) while |F_exact(q)| stays tiny, then the blocker is not a smooth carrier amplitude mismatch but an unresolved sign-root floor",
        "envelope_microphase_read": "small envelope with dense sign flips implies envelope/microphase decoupling rather than a same-level smooth phase-carvature failure",
    }


# 関数: `.2007-.2010` を実行する。

def main() -> None:
    """Execute the boundary phase-curvature higher-q extension audit."""
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
    ):
        sign_base.require(path)

    status_text = sign_base.read_text(STATUS)
    roadmap_text = sign_base.read_text(ROADMAP)
    current_problem_text = sign_base.read_text(CURRENT_PROBLEM)
    current_status_text = sign_base.read_text(CURRENT_STATUS)
    unified_text = sign_base.read_text(UNIFIED_ROADMAP)
    long_text = sign_base.read_text(LONG_ROADMAP)
    part5_text = sign_base.read_text(PART5)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    inventory_ready = bool(prior_summary["higher_q_phase_curvature_generalization_admissible_now"])

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
    spacing_ref = math.pi / r_box
    h0, h1, h2 = local_jet_base.boundary_local_jet(radius, field)
    exact_roots_all = asymp_base.find_signed_zeros_interval(
        radius,
        weight,
        norm,
        EDGE_Q_MAX,
    )
    predicted_roots_all = find_custom_zeros(
        FIT_Q_MIN,
        EDGE_Q_MAX,
        lambda q_value: shifted_zero_equation(
            q_value,
            h0,
            h1,
            h2,
            r_box,
            float(prior_summary["phase_curvature_phi0"]),
            float(prior_summary["phase_curvature_phi_inv"]),
            float(prior_summary["phase_curvature_delta_r"]),
            float(prior_summary["phase_curvature_phi_inv2"]),
        ),
        r_box,
    )

    fit_window = evaluate_window(
        radius,
        weight,
        norm,
        exact_roots_all,
        predicted_roots_all,
        FIT_Q_MIN,
        FIT_Q_MAX,
        spacing_ref,
    )
    floor_window = evaluate_window(
        radius,
        weight,
        norm,
        exact_roots_all,
        predicted_roots_all,
        FIT_Q_MAX,
        FLOOR_Q_MAX,
        spacing_ref,
    )
    micro_window = evaluate_window(
        radius,
        weight,
        norm,
        exact_roots_all,
        predicted_roots_all,
        FLOOR_Q_MAX,
        MICRO_Q_MAX,
        spacing_ref,
    )
    edge_window = evaluate_window(
        radius,
        weight,
        norm,
        exact_roots_all,
        predicted_roots_all,
        MICRO_Q_MAX,
        EDGE_Q_MAX,
        spacing_ref,
    )

    farther_high_q_extension_supported = bool(
        (floor_window["sign_mismatch_fraction"] <= 0.05)
        and (micro_window["sign_mismatch_fraction"] <= 0.05)
        and (edge_window["sign_mismatch_fraction"] <= 0.05)
    )
    unresolved_sign_root_floor_detected = bool(
        (floor_window["sign_mismatch_fraction"] >= 0.5)
        and (floor_window["signed_reconstruction_max_abs_error"] <= 1.0e-5)
        and (floor_window["exact_zero_count"] >= 5.0 * floor_window["predicted_zero_count"])
    )
    envelope_microphase_decoupling_detected = bool(
        (floor_window["sign_mismatch_fraction"] >= 0.5)
        and (micro_window["sign_mismatch_fraction"] >= 0.5)
        and (floor_window["max_abs_form_factor"] <= 1.0e-6)
        and (micro_window["max_abs_form_factor"] <= 1.0e-6)
    )
    edge_spike_window_detected = bool(
        (edge_window["sign_mismatch_fraction"] <= 0.2)
        and (edge_window["max_abs_form_factor"] >= 1000.0 * micro_window["max_abs_form_factor"])
    )
    smooth_phase_curvature_family_exhausted = bool(
        (not farther_high_q_extension_supported)
        and unresolved_sign_root_floor_detected
        and envelope_microphase_decoupling_detected
    )
    physical_reject_required = False

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "phase-curvature farther higher-q inventory ready",
            sign_base.truth(inventory_ready),
            "The farther-window audit starts only after the 120<=q/m0<=200 phase-curvature partial-retain theorem has been formally fixed.",
        ),
        sign_base.row(
            "fit_window_sign_mismatch_fraction",
            "watch",
            "4-term phase-curvature sign mismatch fraction on 200<=q/m0<=260",
            fit_window["sign_mismatch_fraction"],
            "This is the first farther higher-q window directly downstream of the retained partial theorem.",
        ),
        sign_base.row(
            "floor_window_sign_mismatch_fraction",
            "watch",
            "4-term phase-curvature sign mismatch fraction on 260<=q/m0<=320",
            floor_window["sign_mismatch_fraction"],
            "This window tests whether the later failure is a smooth continuation problem or a catastrophic sign-root floor.",
        ),
        sign_base.row(
            "micro_window_sign_mismatch_fraction",
            "watch",
            "4-term phase-curvature sign mismatch fraction on 320<=q/m0<=380",
            micro_window["sign_mismatch_fraction"],
            "This window isolates the microphase regime where the envelope is tiny but sign flips can still proliferate.",
        ),
        sign_base.row(
            "edge_window_sign_mismatch_fraction",
            "watch",
            "4-term phase-curvature sign mismatch fraction on 380<=q/m0<=420",
            edge_window["sign_mismatch_fraction"],
            "This edge window checks whether a late amplitude spike partially reopens the sign pattern after the microphase regime.",
        ),
        sign_base.row(
            "farther_high_q_extension_supported",
            "reject" if not farther_high_q_extension_supported else "pass",
            "4-term phase-curvature farther high-q extension supported",
            sign_base.truth(farther_high_q_extension_supported),
            "The smooth carrier is only retained as a farther high-q theorem if all later windows keep the sign mismatch under control.",
        ),
        sign_base.row(
            "unresolved_sign_root_floor_detected",
            "watch" if unresolved_sign_root_floor_detected else "pass",
            "unresolved high-q sign-root floor detected",
            sign_base.truth(unresolved_sign_root_floor_detected),
            "Dense root proliferation with tiny reconstruction error indicates a sign-root floor rather than a simple smooth carrier miss.",
        ),
        sign_base.row(
            "envelope_microphase_decoupling_detected",
            "watch" if envelope_microphase_decoupling_detected else "pass",
            "envelope/microphase decoupling detected",
            sign_base.truth(envelope_microphase_decoupling_detected),
            "Tiny absolute envelope together with O(1) sign mismatch means the blocker has moved from amplitude fitting to microphase control.",
        ),
        sign_base.row(
            "edge_spike_window_detected",
            "watch" if edge_spike_window_detected else "pass",
            "edge spike window detected",
            sign_base.truth(edge_spike_window_detected),
            "A late edge spike suggests the failure is not monotone in q and should be split from the microphase floor regime.",
        ),
        sign_base.row(
            "smooth_phase_curvature_family_exhausted",
            "watch" if smooth_phase_curvature_family_exhausted else "pass",
            "smooth phase-curvature family exhausted",
            sign_base.truth(smooth_phase_curvature_family_exhausted),
            "Once the blocker is sign-root floor plus microphase decoupling, more same-level smooth carrier refits are no longer the honest next move.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_phase_curvature_phi0": float(prior_summary["phase_curvature_phi0"]),
        "retained_phase_curvature_phi_inv": float(prior_summary["phase_curvature_phi_inv"]),
        "retained_phase_curvature_delta_r": float(prior_summary["phase_curvature_delta_r"]),
        "retained_phase_curvature_phi_inv2": float(prior_summary["phase_curvature_phi_inv2"]),
        "spacing_ref_pi_over_rbox": spacing_ref,
        "fit_window_exact_zero_count": fit_window["exact_zero_count"],
        "fit_window_predicted_zero_count": fit_window["predicted_zero_count"],
        "fit_window_root_nn_max_abs_error": fit_window["exact_to_pred_max_abs_error"],
        "fit_window_sign_mismatch_fraction": fit_window["sign_mismatch_fraction"],
        "fit_window_signed_reconstruction_max_abs_error": fit_window["signed_reconstruction_max_abs_error"],
        "fit_window_mean_spacing": fit_window["mean_spacing"],
        "fit_window_max_spacing_rel_gap_vs_pi_over_rbox": fit_window["max_spacing_rel_gap_vs_pi_over_rbox"],
        "fit_window_max_abs_form_factor": fit_window["max_abs_form_factor"],
        "floor_window_exact_zero_count": floor_window["exact_zero_count"],
        "floor_window_predicted_zero_count": floor_window["predicted_zero_count"],
        "floor_window_root_nn_max_abs_error": floor_window["exact_to_pred_max_abs_error"],
        "floor_window_sign_mismatch_fraction": floor_window["sign_mismatch_fraction"],
        "floor_window_signed_reconstruction_max_abs_error": floor_window["signed_reconstruction_max_abs_error"],
        "floor_window_mean_spacing": floor_window["mean_spacing"],
        "floor_window_max_spacing_rel_gap_vs_pi_over_rbox": floor_window["max_spacing_rel_gap_vs_pi_over_rbox"],
        "floor_window_max_abs_form_factor": floor_window["max_abs_form_factor"],
        "micro_window_exact_zero_count": micro_window["exact_zero_count"],
        "micro_window_predicted_zero_count": micro_window["predicted_zero_count"],
        "micro_window_root_nn_max_abs_error": micro_window["exact_to_pred_max_abs_error"],
        "micro_window_sign_mismatch_fraction": micro_window["sign_mismatch_fraction"],
        "micro_window_signed_reconstruction_max_abs_error": micro_window["signed_reconstruction_max_abs_error"],
        "micro_window_mean_spacing": micro_window["mean_spacing"],
        "micro_window_max_spacing_rel_gap_vs_pi_over_rbox": micro_window["max_spacing_rel_gap_vs_pi_over_rbox"],
        "micro_window_max_abs_form_factor": micro_window["max_abs_form_factor"],
        "edge_window_exact_zero_count": edge_window["exact_zero_count"],
        "edge_window_predicted_zero_count": edge_window["predicted_zero_count"],
        "edge_window_root_nn_max_abs_error": edge_window["exact_to_pred_max_abs_error"],
        "edge_window_sign_mismatch_fraction": edge_window["sign_mismatch_fraction"],
        "edge_window_signed_reconstruction_max_abs_error": edge_window["signed_reconstruction_max_abs_error"],
        "edge_window_mean_spacing": edge_window["mean_spacing"],
        "edge_window_max_spacing_rel_gap_vs_pi_over_rbox": edge_window["max_spacing_rel_gap_vs_pi_over_rbox"],
        "edge_window_max_abs_form_factor": edge_window["max_abs_form_factor"],
        "farther_high_q_extension_supported": farther_high_q_extension_supported,
        "unresolved_sign_root_floor_detected": unresolved_sign_root_floor_detected,
        "envelope_microphase_decoupling_detected": envelope_microphase_decoupling_detected,
        "edge_spike_window_detected": edge_spike_window_detected,
        "smooth_phase_curvature_family_exhausted": smooth_phase_curvature_family_exhausted,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2009",
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
            },
            "constants": {
                "fit_window_over_m0": [FIT_Q_MIN, FIT_Q_MAX],
                "floor_window_upper_over_m0": FLOOR_Q_MAX,
                "micro_window_upper_over_m0": MICRO_Q_MAX,
                "edge_window_upper_over_m0": EDGE_Q_MAX,
                "next_route_name": NEXT_ROUTE_NAME,
                "next_route": NEXT_ROUTE,
                "followup_route_name": FOLLOWUP_ROUTE_NAME,
                "followup_route": FOLLOWUP_ROUTE,
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_boundary_phase_curvature_farther_high_q_audited",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2007"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, "8.7.56.2007-.2010"),
                "current_problem_hit": sign_base.hit(
                    current_problem_text,
                    "higher_q_phase_curvature_generalization_admissible_now",
                ),
                "current_status_hit": sign_base.hit(
                    current_status_text,
                    "boundary phase-curvature decision gate",
                ),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2007-.2010"),
                "long_roadmap_hit": sign_base.hit(
                    long_text,
                    "boundary phase-curvature higher-q extension audit",
                ),
                "part5_hit": sign_base.hit(part5_text, ".1999-.2006"),
            },
        },
    )

    route_rows = [
        sign_base.row(
            "smooth_phase_curvature_family_exhausted",
            "watch" if smooth_phase_curvature_family_exhausted else "pass",
            "smooth phase-curvature family exhausted",
            sign_base.truth(smooth_phase_curvature_family_exhausted),
            "The next route should move to sign-root-floor structure only if the same smooth family is no longer the honest variable to tune.",
        ),
        sign_base.row(
            "unresolved_sign_root_floor_detected",
            "watch" if unresolved_sign_root_floor_detected else "pass",
            "unresolved high-q sign-root floor detected",
            sign_base.truth(unresolved_sign_root_floor_detected),
            "The next official branch is justified only if the failure can be reclassified as a sign-root floor instead of a same-level smooth carrier miss.",
        ),
        sign_base.row(
            "envelope_microphase_decoupling_detected",
            "watch" if envelope_microphase_decoupling_detected else "pass",
            "envelope/microphase decoupling detected",
            sign_base.truth(envelope_microphase_decoupling_detected),
            "Once the envelope stays tiny while sign flips proliferate, the next branch should isolate sign-root logic from envelope size.",
        ),
        sign_base.row(
            "next_route_fixed",
            "pass",
            "next route fixed",
            1.0,
            "The next official branch is the farther high-q sign-root decision gate / registry.",
        ),
    ]

    route_payload = sign_base.payload(
        "8.7.56.2010",
        STEP_NAME + " route sync",
        {
            "declaration_source": sign_base.display_path(
                build_metrics_paths(PUBLIC_OUT, STEM, "declaration_gate")["json"]
            ),
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "selected_next_generation_route_or_none": NEXT_ROUTE,
            "selected_followup_route": FOLLOWUP_ROUTE_NAME,
            "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        },
        route_rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_boundary_phase_curvature_farther_high_q_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_next_hit": sign_base.hit(status_text, "8.7.56.2007"),
                "roadmap_next_hit": sign_base.hit(roadmap_text, "8.7.56.2011-.2014"),
                "current_problem_next_hit": sign_base.hit(
                    current_problem_text,
                    "phase_curvature_higher_q_holdout_failed",
                ),
                "current_status_next_hit": sign_base.hit(
                    current_status_text,
                    "large-coefficient 4-term phase-curvature family",
                ),
                "unified_roadmap_next_hit": sign_base.hit(unified_text, ".2011-.2014"),
                "long_roadmap_next_hit": sign_base.hit(
                    long_text,
                    "phase-curvature generalization decision gate / registry",
                ),
                "part5_next_hit": sign_base.hit(part5_text, ".1999-.2006"),
            },
        },
    )

    declaration_paths = write_artifact("declaration_gate", declaration_payload)
    route_paths = write_artifact("route_sync", route_payload)
    print("[ok] 8.7.56.2007-.2010 boundary phase-curvature farther high-q artifacts generated")
    print(f"[ok] declaration: {declaration_paths['json']}")
    print(f"[ok] route sync:   {route_paths['json']}")


if __name__ == "__main__":
    main()
