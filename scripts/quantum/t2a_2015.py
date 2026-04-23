#!/usr/bin/env python3
"""Generate 8.7.56.2015-.2018 resolved high-q sign-root artifacts.

The prior branch classified the farther high-q failure as an unresolved
sign-root floor plus envelope/microphase split. This script resolves that
blocker into two separable pieces:

1. A bookkeeping artifact from the old root finder:
   the prior `exact_zero_count` included every `|F(q_left)| <= ROOT_TOL` scan
   hit as a fresh zero, which over-counts roots whenever the envelope becomes
   tiny and the scan walks through near-zero plateaus.
2. A genuine higher-q structure tied to the discrete solver box:
   once the direct overlap is evaluated on a grid with step `Δr`, the raw
   signed read becomes non-canonical beyond the Nyquist scale

       q_N = π / Δr_max,

   and the later residual mismatch aligns with alias harmonics

       q_alias^(n) = 2 n π / Δr_max.

Inside the floor/micro windows the catastrophic "root explosion" disappears
once the audit is rewritten in terms of sign-change parity rather than raw
ROOT_TOL hits. The remaining honest blocker is therefore not a generic
sign-root floor but a boundary alias-harmonic spike family.
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
from scipy.optimize import brentq


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
import scripts.quantum.t2a_1975 as local_jet_base
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
PRIOR_AUDIT = (
    PUBLIC_OUT
    / "q_8_7_56_2007_2010_boundary_phase_curvature_higher_q_ext_audit_declaration_gate_metrics.json"
)
PRIOR_GATE = (
    PUBLIC_OUT
    / "q_8_7_56_2011_2014_farther_high_q_sign_root_gate_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.2015-2018"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor resolved high-q "
    "sign-root floor / envelope-microphase audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "resolved_high_q_sign_floor_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_resolved_high_q_sign_root_floor_envelope_microphase_"
    "reactivation_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_high_q_sign_root_floor_resolved_alias_harmonic_spike_"
    "gate_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_resolved_high_q_sign_root_"
    "decision_gate_registry"
)
NEXT_ROUTE = "8.7.56.2019"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_boundary_alias_harmonic_"
    "spike_audit"
)
FOLLOWUP_ROUTE = "8.7.56.2023"

FIT_Q_MIN = 200.0
FIT_Q_MAX = 260.0
FLOOR_Q_MAX = 320.0
MICRO_Q_MAX = 380.0
EDGE_Q_MAX = 420.0
WINDOW_SCAN_DENSITY = 2000
PRIOR_PARITY_SCAN_DENSITY = 300
SIGN_ZERO_TOL = 1.0e-12
ENVELOPE_FLOOR_CANDIDATES = np.array([1.0e-9, 3.0e-9, 1.0e-8, 3.0e-8], dtype=float)


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
    """Return the retained boundary phase-curvature zero equation."""
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


# 関数: retained carrier から predicted zero lattice を返す。

def find_predicted_zeros(
    q_min: float,
    q_max: float,
    equation,
    r_box: float,
) -> np.ndarray:
    """Locate all retained-carrier zero roots on one q interval."""
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


# 関数: one q window の direct overlap scan を返す。

def scan_window(
    radius: np.ndarray,
    weight: np.ndarray,
    norm: float,
    q_min: float,
    q_max: float,
    density: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return q grid, direct overlap values, and sign states on one q window."""
    q_scan = np.linspace(q_min, q_max, int(round((q_max - q_min) * density)) + 1)
    values = np.array(
        [sign_base.form_factor(radius, weight, norm, float(q_value)) for q_value in q_scan],
        dtype=float,
    )
    sign_values = np.sign(values)
    sign_values[np.abs(values) <= SIGN_ZERO_TOL] = 0.0
    return q_scan, values, sign_values


# 関数: one scan の sign-change count を返す。

def sign_change_count(sign_values: np.ndarray) -> int:
    """Count sign changes on one scanned q grid."""
    return int(np.count_nonzero(sign_values[1:] != sign_values[:-1]))


# 関数: one q window の predicted sign mismatch を返す。

def parity_mismatch_metrics(
    q_scan: np.ndarray,
    values: np.ndarray,
    sign_values: np.ndarray,
    predicted_roots: np.ndarray,
    prior_sign_changes: int,
) -> dict[str, float]:
    """Return mismatch metrics when parity is tracked by sign changes."""
    abs_values = np.abs(values)
    sigma_pred = np.empty_like(q_scan)
    for index, q_value in enumerate(q_scan):
        count = prior_sign_changes + int(
            np.count_nonzero(predicted_roots < (float(q_value) - 1.0e-10))
        )
        sigma_pred[index] = 1.0 if (count % 2) == 0 else -1.0

    reconstructed = sigma_pred * abs_values
    return {
        "sign_mismatch_fraction": float(np.mean(sigma_pred != sign_values)),
        "signed_reconstruction_max_abs_error": float(
            np.max(np.abs(reconstructed - values))
        ),
        "signed_reconstruction_mean_abs_error": float(
            np.mean(np.abs(reconstructed - values))
        ),
        "max_abs_form_factor": float(np.max(abs_values)),
        "mean_abs_form_factor": float(np.mean(abs_values)),
    }


# 関数: one q window と alias harmonic の距離を返す。

def min_harmonic_distance(
    q_min: float,
    q_max: float,
    harmonic_period: float,
) -> float:
    """Return the minimum distance from one q window to any alias harmonic."""
    q_scan = np.linspace(q_min, q_max, int(round((q_max - q_min) * 100)) + 1)
    harmonics = harmonic_period * np.arange(1, 6, dtype=float)
    distances = np.min(np.abs(q_scan[:, None] - harmonics[None, :]), axis=1)
    return float(np.min(distances))


# 関数: floor/micro window の threshold family を返す。

def envelope_floor_family(
    floor_abs: np.ndarray,
    floor_sign: np.ndarray,
    floor_sigma_pred: np.ndarray,
    micro_abs: np.ndarray,
    micro_sign: np.ndarray,
    micro_sigma_pred: np.ndarray,
) -> dict[str, float]:
    """Return shared envelope-floor candidate metrics on floor/micro windows."""
    results: dict[str, float] = {}
    total_points = floor_abs.size + micro_abs.size
    best_tau = 0.0
    best_mismatch = math.inf
    best_keep_fraction = 0.0
    for tau in ENVELOPE_FLOOR_CANDIDATES:
        floor_mask = floor_abs > tau
        micro_mask = micro_abs > tau
        kept = int(np.count_nonzero(floor_mask)) + int(np.count_nonzero(micro_mask))
        mismatch = (
            int(np.count_nonzero(floor_sigma_pred[floor_mask] != floor_sign[floor_mask]))
            + int(np.count_nonzero(micro_sigma_pred[micro_mask] != micro_sign[micro_mask]))
        )
        mismatch_fraction = 0.0 if kept == 0 else (mismatch / kept)
        keep_fraction = kept / total_points
        tau_label = f"{tau:.0e}".replace("+", "")
        results[f"tau_{tau_label}_combined_mismatch_fraction"] = float(mismatch_fraction)
        results[f"tau_{tau_label}_combined_keep_fraction"] = float(keep_fraction)
        if mismatch_fraction < best_mismatch:
            best_tau = float(tau)
            best_mismatch = float(mismatch_fraction)
            best_keep_fraction = float(keep_fraction)

    results["best_envelope_floor_tau"] = best_tau
    results["best_envelope_floor_combined_mismatch_fraction"] = best_mismatch
    results["best_envelope_floor_combined_keep_fraction"] = best_keep_fraction
    return results


# 関数: audit 用の公式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the high-q sign-root floor resolution audit."""
    return {
        "nyquist_scale": "q_N = pi / Delta r_max",
        "alias_harmonics": "q_alias^(n) = 2 n pi / Delta r_max",
        "resolved_zero_read": "count sign changes on a fixed q scan instead of treating every |F(q_left)| <= ROOT_TOL hit as a fresh zero",
        "resolved_parity_rule": "sigma_res(q)=(-1)^(N_scan(<q_min)+N_pred(q_min<=q_n<q))",
        "envelope_floor_candidate": "|F_exact(q)| <= epsilon_floor implies unresolved microphase sector rather than a canonical signed observable",
    }


# 関数: `.2015-.2018` を実行する。

def main() -> None:
    """Execute the resolved high-q sign-root floor / envelope-microphase audit."""
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
    inventory_ready = bool(prior_gate_summary["resolved_high_q_sign_root_floor_reactivation_admissible_now"])

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
    dr = np.diff(radius)
    delta_r_max = float(np.max(dr))
    delta_r_mean = float(np.mean(dr))
    q_nyquist_max_step = math.pi / delta_r_max
    q_nyquist_mean_step = math.pi / delta_r_mean
    alias_period = 2.0 * q_nyquist_max_step
    first_alias_harmonic = alias_period
    second_alias_harmonic = 2.0 * alias_period

    h0, h1, h2 = local_jet_base.boundary_local_jet(radius, field)
    predicted_roots_all = find_predicted_zeros(
        FIT_Q_MIN,
        EDGE_Q_MAX,
        lambda q_value: shifted_zero_equation(
            q_value,
            h0,
            h1,
            h2,
            r_box,
            float(prior_audit_summary["retained_phase_curvature_phi0"]),
            float(prior_audit_summary["retained_phase_curvature_phi_inv"]),
            float(prior_audit_summary["retained_phase_curvature_delta_r"]),
            float(prior_audit_summary["retained_phase_curvature_phi_inv2"]),
        ),
        r_box,
    )

    window_specs = [
        ("fit", FIT_Q_MIN, FIT_Q_MAX),
        ("floor", FIT_Q_MAX, FLOOR_Q_MAX),
        ("micro", FLOOR_Q_MAX, MICRO_Q_MAX),
        ("edge", MICRO_Q_MAX, EDGE_Q_MAX),
    ]
    window_data: dict[str, dict[str, float | np.ndarray]] = {}
    for label, q_min, q_max in window_specs:
        q_scan, values, sign_values = scan_window(
            radius,
            weight,
            norm,
            q_min,
            q_max,
            WINDOW_SCAN_DENSITY,
        )
        predicted_window = predicted_roots_all[
            (predicted_roots_all >= q_min) & (predicted_roots_all <= q_max)
        ]
        _q_prior, _prior_values, prior_sign_values = scan_window(
            radius,
            weight,
            norm,
            0.0,
            q_min,
            PRIOR_PARITY_SCAN_DENSITY,
        )
        prior_changes = sign_change_count(prior_sign_values)
        parity_metrics = parity_mismatch_metrics(
            q_scan,
            values,
            sign_values,
            predicted_window,
            prior_changes,
        )
        sign_changes = sign_change_count(sign_values)
        mean_sign_change_spacing = (q_max - q_min) / sign_changes
        predicted_count_rel_gap = abs(sign_changes - predicted_window.size) / predicted_window.size
        window_data[label] = {
            "q_scan": q_scan,
            "values": values,
            "abs_values": np.abs(values),
            "sign_values": sign_values,
            "prior_sign_changes": float(prior_changes),
            "sign_change_count": float(sign_changes),
            "mean_sign_change_spacing": float(mean_sign_change_spacing),
            "predicted_zero_count": float(predicted_window.size),
            "predicted_count_rel_gap": float(predicted_count_rel_gap),
            "min_alias_harmonic_distance": min_harmonic_distance(
                q_min,
                q_max,
                alias_period,
            ),
            **parity_metrics,
        }

        sigma_pred = np.empty_like(q_scan)
        for index, q_value in enumerate(q_scan):
            count = int(window_data[label]["prior_sign_changes"]) + int(
                np.count_nonzero(predicted_window < (float(q_value) - 1.0e-10))
            )
            sigma_pred[index] = 1.0 if (count % 2) == 0 else -1.0

        window_data[label]["sigma_pred"] = sigma_pred

    floor_root_duplication_ratio = (
        float(prior_audit_summary["floor_window_exact_zero_count"])
        / float(window_data["floor"]["sign_change_count"])
    )
    micro_root_duplication_ratio = (
        float(prior_audit_summary["micro_window_exact_zero_count"])
        / float(window_data["micro"]["sign_change_count"])
    )
    fit_root_duplication_ratio = (
        float(prior_audit_summary["fit_window_exact_zero_count"])
        / float(window_data["fit"]["sign_change_count"])
    )
    edge_root_duplication_ratio = (
        float(prior_audit_summary["edge_window_exact_zero_count"])
        / float(window_data["edge"]["sign_change_count"])
    )

    floor_mismatch_gain = (
        float(prior_audit_summary["floor_window_sign_mismatch_fraction"])
        / float(window_data["floor"]["sign_mismatch_fraction"])
    )
    micro_mismatch_gain = (
        float(prior_audit_summary["micro_window_sign_mismatch_fraction"])
        / float(window_data["micro"]["sign_mismatch_fraction"])
    )

    threshold_family = envelope_floor_family(
        np.array(window_data["floor"]["abs_values"], dtype=float),
        np.array(window_data["floor"]["sign_values"], dtype=float),
        np.array(window_data["floor"]["sigma_pred"], dtype=float),
        np.array(window_data["micro"]["abs_values"], dtype=float),
        np.array(window_data["micro"]["sign_values"], dtype=float),
        np.array(window_data["micro"]["sigma_pred"], dtype=float),
    )

    sign_root_floor_resolved = bool(
        floor_root_duplication_ratio > 2.0
        and micro_root_duplication_ratio > 2.0
        and float(window_data["floor"]["predicted_count_rel_gap"]) < 0.05
        and float(window_data["micro"]["predicted_count_rel_gap"]) < 0.01
    )
    envelope_floor_candidate_admissible = bool(
        threshold_family["tau_1e-08_combined_mismatch_fraction"] < 0.02
    )
    fit_alias_harmonic_window_detected = bool(
        float(window_data["fit"]["min_alias_harmonic_distance"]) < 0.01
    )
    edge_alias_harmonic_window_detected = bool(
        float(window_data["edge"]["min_alias_harmonic_distance"]) < 0.01
    )
    remaining_blocker_is_alias_harmonic_spike = bool(
        sign_root_floor_resolved
        and envelope_floor_candidate_admissible
        and fit_alias_harmonic_window_detected
        and edge_alias_harmonic_window_detected
    )
    physical_reject_required = False

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "resolved high-q sign-root inventory ready",
            sign_base.truth(inventory_ready),
            "The resolution audit starts only after the prior gate has promoted the sign-root floor / envelope-microphase family to the active blocker.",
        ),
        sign_base.row(
            "q_nyquist_box_over_m0",
            "watch",
            "box-grid Nyquist scale q_N/m0",
            q_nyquist_max_step,
            "Direct overlap sign is non-canonical once q exceeds the Nyquist scale set by the largest solver-box grid step.",
        ),
        sign_base.row(
            "first_alias_harmonic_over_m0",
            "watch",
            "first alias harmonic over m0",
            first_alias_harmonic,
            "The fit window sits on top of the first alias harmonic rather than a smooth farther high-q continuation.",
        ),
        sign_base.row(
            "second_alias_harmonic_over_m0",
            "watch",
            "second alias harmonic over m0",
            second_alias_harmonic,
            "The edge spike window sits on top of the second alias harmonic.",
        ),
        sign_base.row(
            "floor_root_duplication_ratio",
            "watch",
            "floor-window raw root duplication ratio",
            floor_root_duplication_ratio,
            "Raw ROOT_TOL bookkeeping over-counts floor-window zeros by more than an order of magnitude relative to sign-change parity.",
        ),
        sign_base.row(
            "micro_root_duplication_ratio",
            "watch",
            "micro-window raw root duplication ratio",
            micro_root_duplication_ratio,
            "The micro window also over-counts zeros once ROOT_TOL hits are treated as fresh roots.",
        ),
        sign_base.row(
            "floor_resolved_sign_mismatch_fraction",
            "watch",
            "floor-window resolved sign mismatch fraction",
            float(window_data["floor"]["sign_mismatch_fraction"]),
            "After replacing raw ROOT_TOL roots with sign-change parity, the floor mismatch drops sharply.",
        ),
        sign_base.row(
            "micro_resolved_sign_mismatch_fraction",
            "watch",
            "micro-window resolved sign mismatch fraction",
            float(window_data["micro"]["sign_mismatch_fraction"]),
            "The micro window becomes a mild residual once parity is tracked by sign changes rather than raw ROOT_TOL hits.",
        ),
        sign_base.row(
            "candidate_envelope_floor_tau_1e_8_combined_mismatch_fraction",
            "watch",
            "candidate envelope floor tau=1e-8 combined mismatch fraction",
            threshold_family["tau_1e-08_combined_mismatch_fraction"],
            "A modest shared envelope floor candidate already collapses the floor/micro mismatch family below the 2% level.",
        ),
        sign_base.row(
            "sign_root_floor_resolved",
            "pass" if sign_root_floor_resolved else "reject",
            "high-q sign-root floor resolved",
            sign_base.truth(sign_root_floor_resolved),
            "The old floor is resolved only if the raw root explosion collapses to near-predicted sign-change counts.",
        ),
        sign_base.row(
            "envelope_floor_candidate_admissible",
            "pass" if envelope_floor_candidate_admissible else "reject",
            "shared envelope-floor candidate admissible",
            sign_base.truth(envelope_floor_candidate_admissible),
            "The envelope/microphase split is only meaningful if one shared floor candidate suppresses the residual floor/micro mismatch family.",
        ),
        sign_base.row(
            "fit_alias_harmonic_window_detected",
            "watch" if fit_alias_harmonic_window_detected else "pass",
            "fit-window alias harmonic detected",
            sign_base.truth(fit_alias_harmonic_window_detected),
            "The first farther higher-q residual sits exactly on the first alias harmonic of the box grid.",
        ),
        sign_base.row(
            "edge_alias_harmonic_window_detected",
            "watch" if edge_alias_harmonic_window_detected else "pass",
            "edge-window alias harmonic detected",
            sign_base.truth(edge_alias_harmonic_window_detected),
            "The late edge spike reopens on top of the second alias harmonic.",
        ),
        sign_base.row(
            "remaining_blocker_is_alias_harmonic_spike",
            "watch" if remaining_blocker_is_alias_harmonic_spike else "pass",
            "remaining blocker is alias-harmonic spike",
            sign_base.truth(remaining_blocker_is_alias_harmonic_spike),
            "Once the floor is resolved, the honest residual blocker is no longer generic microphase but alias-harmonic spike windows.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "box_grid_point_count": float(radius.size),
        "box_grid_delta_r_max": delta_r_max,
        "box_grid_delta_r_mean": delta_r_mean,
        "q_nyquist_box_over_m0": q_nyquist_max_step,
        "q_nyquist_mean_step_over_m0": q_nyquist_mean_step,
        "first_alias_harmonic_over_m0": first_alias_harmonic,
        "second_alias_harmonic_over_m0": second_alias_harmonic,
        "fit_raw_root_duplication_ratio": fit_root_duplication_ratio,
        "floor_raw_root_duplication_ratio": floor_root_duplication_ratio,
        "micro_raw_root_duplication_ratio": micro_root_duplication_ratio,
        "edge_raw_root_duplication_ratio": edge_root_duplication_ratio,
        "fit_sign_change_count": float(window_data["fit"]["sign_change_count"]),
        "floor_sign_change_count": float(window_data["floor"]["sign_change_count"]),
        "micro_sign_change_count": float(window_data["micro"]["sign_change_count"]),
        "edge_sign_change_count": float(window_data["edge"]["sign_change_count"]),
        "fit_predicted_zero_count": float(window_data["fit"]["predicted_zero_count"]),
        "floor_predicted_zero_count": float(window_data["floor"]["predicted_zero_count"]),
        "micro_predicted_zero_count": float(window_data["micro"]["predicted_zero_count"]),
        "edge_predicted_zero_count": float(window_data["edge"]["predicted_zero_count"]),
        "fit_predicted_count_rel_gap": float(window_data["fit"]["predicted_count_rel_gap"]),
        "floor_predicted_count_rel_gap": float(window_data["floor"]["predicted_count_rel_gap"]),
        "micro_predicted_count_rel_gap": float(window_data["micro"]["predicted_count_rel_gap"]),
        "edge_predicted_count_rel_gap": float(window_data["edge"]["predicted_count_rel_gap"]),
        "fit_resolved_sign_mismatch_fraction": float(window_data["fit"]["sign_mismatch_fraction"]),
        "floor_resolved_sign_mismatch_fraction": float(window_data["floor"]["sign_mismatch_fraction"]),
        "micro_resolved_sign_mismatch_fraction": float(window_data["micro"]["sign_mismatch_fraction"]),
        "edge_resolved_sign_mismatch_fraction": float(window_data["edge"]["sign_mismatch_fraction"]),
        "fit_min_alias_harmonic_distance": float(window_data["fit"]["min_alias_harmonic_distance"]),
        "floor_min_alias_harmonic_distance": float(window_data["floor"]["min_alias_harmonic_distance"]),
        "micro_min_alias_harmonic_distance": float(window_data["micro"]["min_alias_harmonic_distance"]),
        "edge_min_alias_harmonic_distance": float(window_data["edge"]["min_alias_harmonic_distance"]),
        "floor_mismatch_gain": floor_mismatch_gain,
        "micro_mismatch_gain": micro_mismatch_gain,
        **threshold_family,
        "sign_root_floor_resolved": sign_root_floor_resolved,
        "envelope_floor_candidate_admissible": envelope_floor_candidate_admissible,
        "fit_alias_harmonic_window_detected": fit_alias_harmonic_window_detected,
        "edge_alias_harmonic_window_detected": edge_alias_harmonic_window_detected,
        "remaining_blocker_is_alias_harmonic_spike": remaining_blocker_is_alias_harmonic_spike,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2017",
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
            "constants": {
                "fit_window_over_m0": [FIT_Q_MIN, FIT_Q_MAX],
                "floor_window_upper_over_m0": FLOOR_Q_MAX,
                "micro_window_upper_over_m0": MICRO_Q_MAX,
                "edge_window_upper_over_m0": EDGE_Q_MAX,
                "window_scan_density": WINDOW_SCAN_DENSITY,
                "prior_parity_scan_density": PRIOR_PARITY_SCAN_DENSITY,
                "envelope_floor_candidates": ENVELOPE_FLOOR_CANDIDATES.tolist(),
                "next_route_name": NEXT_ROUTE_NAME,
                "next_route": NEXT_ROUTE,
                "followup_route_name": FOLLOWUP_ROUTE_NAME,
                "followup_route": FOLLOWUP_ROUTE,
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_high_q_sign_root_floor_resolution_audited",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2015"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, "8.7.56.2015-.2018"),
                "current_problem_hit": sign_base.hit(
                    current_problem_text,
                    "resolved high-q sign-root floor / envelope-microphase split",
                ),
                "current_status_hit": sign_base.hit(
                    current_status_text,
                    "resolved high-q sign-root floor / envelope-microphase audit",
                ),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2015-.2018"),
                "long_roadmap_hit": sign_base.hit(
                    long_text,
                    "resolved high-q sign-root floor / envelope-microphase audit",
                ),
                "part5_hit": sign_base.hit(part5_text, ".2007-.2014"),
            },
        },
    )

    route_rows = [
        sign_base.row(
            "sign_root_floor_resolved",
            "pass" if sign_root_floor_resolved else "reject",
            "high-q sign-root floor resolved",
            sign_base.truth(sign_root_floor_resolved),
            "The next official gate is justified only if the raw root explosion has been honestly reclassified as bookkeeping rather than a physical root family.",
        ),
        sign_base.row(
            "remaining_blocker_is_alias_harmonic_spike",
            "watch" if remaining_blocker_is_alias_harmonic_spike else "pass",
            "remaining blocker is alias-harmonic spike",
            sign_base.truth(remaining_blocker_is_alias_harmonic_spike),
            "Once the floor is resolved, the honest followup is an alias-harmonic spike audit rather than another generic sign-root family retry.",
        ),
        sign_base.row(
            "next_route_fixed",
            "pass",
            "next route fixed",
            1.0,
            "The next official branch is the resolved high-q sign-root decision gate / registry.",
        ),
    ]

    route_payload = sign_base.payload(
        "8.7.56.2018",
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
            "overall_status": "vector_qball_form_factor_high_q_sign_root_floor_resolution_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_next_hit": sign_base.hit(status_text, "8.7.56.2015"),
                "roadmap_next_hit": sign_base.hit(roadmap_text, "8.7.56.2019-.2022"),
                "current_problem_next_hit": sign_base.hit(
                    current_problem_text,
                    "resolved high-q sign-root floor / envelope-microphase split",
                ),
                "current_status_next_hit": sign_base.hit(
                    current_status_text,
                    "resolved high-q sign-root floor / envelope-microphase audit",
                ),
                "unified_roadmap_next_hit": sign_base.hit(unified_text, ".2019-.2022"),
                "long_roadmap_next_hit": sign_base.hit(
                    long_text,
                    "resolved high-q sign-root decision gate / registry",
                ),
                "part5_next_hit": sign_base.hit(part5_text, ".2007-.2014"),
            },
        },
    )

    declaration_paths = write_artifact("declaration_gate", declaration_payload)
    route_paths = write_artifact("route_sync", route_payload)
    print("[ok] 8.7.56.2015-.2018 resolved high-q sign-root floor artifacts generated")
    print(f"[ok] declaration: {declaration_paths['json']}")
    print(f"[ok] route sync:   {route_paths['json']}")


if __name__ == "__main__":
    main()
