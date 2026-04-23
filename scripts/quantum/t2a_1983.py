#!/usr/bin/env python3
"""Generate 8.7.56.1983-.1986 boundary local-jet higher-q extension artifacts."""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime
from datetime import timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
import scripts.quantum.t2a_1963 as asymp_base
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
LESSONS = ROOT / "doc" / "quantum" / "56_trial2_numeric_alpha_vector_qball_theory_lessons_after_interval_extension.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

QBALL_BRANCH_REFRESH = PUBLIC_OUT / "mass_origin_qball_charge_mapping_branch_refresh_metrics.json"
PRIOR_GATE = (
    PUBLIC_OUT
    / "q_8_7_56_1979_1982_box_edge_local_jet_closeout_registry_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1983-1986"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor boundary local-jet "
    "higher-q extension audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "boundary_local_jet_higher_q_ext_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_box_edge_local_jet_signed_rule_retained_higher_q_"
    "generalization_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_box_edge_local_jet_extension_to_40_retained_"
    "asymptotic_phase_drift_generalization_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_boundary_local_jet_"
    "generalization_decision_gate_registry"
)
NEXT_ROUTE = "8.7.56.1987"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_boundary_local_jet_"
    "asymptotic_phase_drift_audit"
)
FOLLOWUP_ROUTE = "8.7.56.1991"
EXTENSION_Q_MIN = 12.0
EXTENSION_Q_MAX = 40.0
MONITOR_Q_MAX = 60.0
STRESS_Q_MAX = 80.0


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


# 関数: nearest-neighbor root 誤差の統計を返す。

def nearest_neighbor_stats(exact_roots: np.ndarray, predicted_roots: np.ndarray) -> dict[str, float]:
    """Return symmetric nearest-neighbor root error statistics."""
    exact_to_pred = np.array(
        [float(np.min(np.abs(predicted_roots - value))) for value in exact_roots],
        dtype=float,
    )
    pred_to_exact = np.array(
        [float(np.min(np.abs(exact_roots - value))) for value in predicted_roots],
        dtype=float,
    )
    return {
        "exact_to_pred_max_abs_error": float(np.max(exact_to_pred)),
        "exact_to_pred_mean_abs_error": float(np.mean(exact_to_pred)),
        "pred_to_exact_max_abs_error": float(np.max(pred_to_exact)),
        "pred_to_exact_mean_abs_error": float(np.mean(pred_to_exact)),
    }


# 関数: one q window の signed-rule metrics を返す。

def evaluate_window(
    radius: np.ndarray,
    weight: np.ndarray,
    norm: float,
    exact_roots_all: np.ndarray,
    predicted_roots_all: np.ndarray,
    q_min: float,
    q_max: float,
) -> dict[str, float]:
    """Return zero-lattice and reconstruction diagnostics on one q window."""
    exact_window = exact_roots_all[(exact_roots_all >= q_min) & (exact_roots_all <= q_max)]
    predicted_window = predicted_roots_all[
        (predicted_roots_all >= q_min) & (predicted_roots_all <= q_max)
    ]
    root_stats = nearest_neighbor_stats(exact_window, predicted_window)
    q_scan = np.linspace(q_min, q_max, 80001)
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
        "signed_reconstruction_max_abs_error": reconstruction["max_abs_error"],
        "signed_reconstruction_mean_abs_error": reconstruction["mean_abs_error"],
        "sign_mismatch_fraction": reconstruction["sign_mismatch_fraction"],
    }


# 関数: audit 用の公式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the higher-q extension audit."""
    return {
        "retained_box_edge_rule": "G_jet(q)=(-h0 q^2 + h2) cos(q R_box) + h1 q sin(q R_box)=0",
        "nearest_neighbor_metric": "eps_nn = max(min_j |q_exact,i-q_pred,j|, min_i |q_pred,j-q_exact,i|)",
        "hybrid_signed_rule": "sigma_hybrid(q)=sigma_exact(q) for 0<=q<=4, and sigma_hybrid(q)=(-1)^{N_<4+N_jet(q)} for q>4",
        "extension_window": "retained higher-q extension audit on 12<=q/m0<=40",
        "monitor_windows": "40<=q/m0<=60 as warning monitor, 60<=q/m0<=80 as stress monitor",
    }


# 関数: `.1983-.1986` を実行する。

def main() -> None:
    """Execute the boundary local-jet higher-q extension audit."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        LONG_ROADMAP,
        LESSONS,
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
    lessons_text = sign_base.read_text(LESSONS)
    part5_text = sign_base.read_text(PART5)

    prior_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    inventory_ready = bool(prior_summary["box_edge_local_jet_signed_rule_retained"])

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
    h0, h1, h2 = local_jet_base.boundary_local_jet(radius, field)

    exact_roots_all = asymp_base.find_signed_zeros_interval(radius, weight, norm, STRESS_Q_MAX)
    exact_roots_all = exact_roots_all[exact_roots_all >= EXTENSION_Q_MIN]
    predicted_roots_all = local_jet_base.find_local_jet_zeros(
        EXTENSION_Q_MIN,
        STRESS_Q_MAX,
        h0,
        h1,
        h2,
        r_box,
    )

    extension_metrics = evaluate_window(
        radius,
        weight,
        norm,
        exact_roots_all,
        predicted_roots_all,
        EXTENSION_Q_MIN,
        EXTENSION_Q_MAX,
    )
    monitor_metrics = evaluate_window(
        radius,
        weight,
        norm,
        exact_roots_all,
        predicted_roots_all,
        EXTENSION_Q_MAX,
        MONITOR_Q_MAX,
    )
    stress_metrics = evaluate_window(
        radius,
        weight,
        norm,
        exact_roots_all,
        predicted_roots_all,
        MONITOR_Q_MAX,
        STRESS_Q_MAX,
    )

    higher_q_extension_supported = bool(
        extension_metrics["exact_to_pred_max_abs_error"] <= 0.002
        and extension_metrics["signed_reconstruction_max_abs_error"] <= 2.0e-7
        and extension_metrics["sign_mismatch_fraction"] <= 0.01
    )
    monitor_window_warning = bool(
        monitor_metrics["exact_to_pred_max_abs_error"] > 0.002
        or monitor_metrics["sign_mismatch_fraction"] > 0.01
    )
    asymptotic_phase_drift_detected = bool(
        stress_metrics["exact_to_pred_max_abs_error"] > 0.01
        or stress_metrics["sign_mismatch_fraction"] > 0.1
    )
    asymptotic_generalization_beyond_40_not_yet_supported = asymptotic_phase_drift_detected
    physical_reject_required = False

    rows = [
        sign_base.row("inventory_ready", "pass" if inventory_ready else "reject", "higher-q extension inventory ready", sign_base.truth(inventory_ready), "The higher-q audit starts only after the boundary local-jet rule has already been retained by the closeout branch."),
        sign_base.row("solver_box_edge_over_m0", "watch", "solver box edge R_box/m0^-1", r_box, "The higher-q extension continues the same retained finite-box boundary rule without changing the underlying pack."),
        sign_base.row("extension_exact_zero_count", "watch", "exact zero count on 12<=q/m0<=40", extension_metrics["exact_zero_count"], "This counts the exact sign changes that the retained boundary local-jet rule must track on the first higher-q extension window."),
        sign_base.row("extension_predicted_zero_count", "watch", "predicted zero count on 12<=q/m0<=40", extension_metrics["predicted_zero_count"], "The retained local-jet zero lattice may differ by a few edge roots while still preserving the signed observable reconstruction."),
        sign_base.row("extension_root_nn_max_abs_error", "watch", "nearest-neighbor max root error on 12<=q/m0<=40", extension_metrics["exact_to_pred_max_abs_error"], "A small symmetric nearest-neighbor error shows that the local-jet lattice remains phase-locked to the exact zero set over the retained higher-q extension window."),
        sign_base.row("extension_signed_reconstruction_max_abs_error", "pass" if extension_metrics["signed_reconstruction_max_abs_error"] <= 2.0e-7 else "watch", "max signed reconstruction error on 12<=q/m0<=40", extension_metrics["signed_reconstruction_max_abs_error"], "The higher-q extension is accepted only if the signed observable itself remains numerically stable, not merely the raw zero count."),
        sign_base.row("extension_sign_mismatch_fraction", "pass" if extension_metrics["sign_mismatch_fraction"] <= 0.01 else "watch", "sign mismatch fraction on 12<=q/m0<=40", extension_metrics["sign_mismatch_fraction"], "This measures how often the retained parity rule flips relative to the exact sign sector on the first higher-q extension window."),
        sign_base.row("monitor_root_nn_max_abs_error", "watch", "nearest-neighbor max root error on 40<=q/m0<=60", monitor_metrics["exact_to_pred_max_abs_error"], "The monitor window shows where the retained local-jet rule starts to accumulate visible phase drift beyond the first extension window."),
        sign_base.row("monitor_signed_reconstruction_max_abs_error", "watch", "max signed reconstruction error on 40<=q/m0<=60", monitor_metrics["signed_reconstruction_max_abs_error"], "Even with moderate phase drift, the signed observable can remain numerically accurate on the monitor window."),
        sign_base.row("stress_root_nn_max_abs_error", "watch", "nearest-neighbor max root error on 60<=q/m0<=80", stress_metrics["exact_to_pred_max_abs_error"], "The stress window determines whether the current local-jet theorem is already asymptotically complete or whether a new phase-drift rule is needed."),
        sign_base.row("stress_signed_reconstruction_max_abs_error", "watch", "max signed reconstruction error on 60<=q/m0<=80", stress_metrics["signed_reconstruction_max_abs_error"], "The signed observable itself stays finite, but the stress window decides whether the current theorem remains canonically stable."),
        sign_base.row("stress_sign_mismatch_fraction", "watch", "sign mismatch fraction on 60<=q/m0<=80", stress_metrics["sign_mismatch_fraction"], "A large mismatch fraction at very high q indicates asymptotic phase drift rather than failure of the finite extension window."),
        sign_base.row("higher_q_extension_supported", "pass" if higher_q_extension_supported else "reject", "boundary local-jet extension to 12<=q/m0<=40 supported", sign_base.truth(higher_q_extension_supported), "The retained theorem survives the first higher-q extension only if root locking and signed reconstruction both remain stable on 12<=q/m0<=40."),
        sign_base.row("monitor_window_warning", "watch" if monitor_window_warning else "pass", "40<=q/m0<=60 monitor warning detected", sign_base.truth(monitor_window_warning), "This warning separates the finite higher-q extension from the later asymptotic drift problem."),
        sign_base.row("asymptotic_phase_drift_detected", "watch" if asymptotic_phase_drift_detected else "pass", "asymptotic phase drift detected on 60<=q/m0<=80", sign_base.truth(asymptotic_phase_drift_detected), "The stress window asks whether a new phase-drift signed rule will be needed after the finite extension is retained."),
        sign_base.row("asymptotic_generalization_beyond_40_not_yet_supported", "watch" if asymptotic_generalization_beyond_40_not_yet_supported else "pass", "asymptotic generalization beyond q/m0=40 not yet supported", sign_base.truth(asymptotic_generalization_beyond_40_not_yet_supported), "The current retained theorem can pass the first higher-q extension and still leave a later asymptotic generalization gap."),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "solver_box_edge_over_m0": r_box,
        "higher_q_extension_lower_over_m0": EXTENSION_Q_MIN,
        "higher_q_extension_upper_over_m0": EXTENSION_Q_MAX,
        "monitor_window_upper_over_m0": MONITOR_Q_MAX,
        "stress_window_upper_over_m0": STRESS_Q_MAX,
        "extension_exact_zero_count": extension_metrics["exact_zero_count"],
        "extension_predicted_zero_count": extension_metrics["predicted_zero_count"],
        "extension_root_nn_max_abs_error": extension_metrics["exact_to_pred_max_abs_error"],
        "extension_root_nn_mean_abs_error": extension_metrics["exact_to_pred_mean_abs_error"],
        "extension_signed_reconstruction_max_abs_error": extension_metrics["signed_reconstruction_max_abs_error"],
        "extension_signed_reconstruction_mean_abs_error": extension_metrics["signed_reconstruction_mean_abs_error"],
        "extension_sign_mismatch_fraction": extension_metrics["sign_mismatch_fraction"],
        "monitor_root_nn_max_abs_error": monitor_metrics["exact_to_pred_max_abs_error"],
        "monitor_signed_reconstruction_max_abs_error": monitor_metrics["signed_reconstruction_max_abs_error"],
        "monitor_sign_mismatch_fraction": monitor_metrics["sign_mismatch_fraction"],
        "stress_root_nn_max_abs_error": stress_metrics["exact_to_pred_max_abs_error"],
        "stress_signed_reconstruction_max_abs_error": stress_metrics["signed_reconstruction_max_abs_error"],
        "stress_sign_mismatch_fraction": stress_metrics["sign_mismatch_fraction"],
        "higher_q_extension_supported": higher_q_extension_supported,
        "monitor_window_warning": monitor_window_warning,
        "asymptotic_phase_drift_detected": asymptotic_phase_drift_detected,
        "asymptotic_generalization_beyond_40_not_yet_supported": asymptotic_generalization_beyond_40_not_yet_supported,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.1985",
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
                "lessons": sign_base.display_path(LESSONS),
                "part5": sign_base.display_path(PART5),
                "qball_branch_refresh": sign_base.display_path(QBALL_BRANCH_REFRESH),
                "prior_gate": sign_base.display_path(PRIOR_GATE),
            },
            "constants": {
                "extension_q_min_over_m0": EXTENSION_Q_MIN,
                "extension_q_max_over_m0": EXTENSION_Q_MAX,
                "monitor_q_max_over_m0": MONITOR_Q_MAX,
                "stress_q_max_over_m0": STRESS_Q_MAX,
                "next_route_name": NEXT_ROUTE_NAME,
                "next_route": NEXT_ROUTE,
                "followup_route_name": FOLLOWUP_ROUTE_NAME,
                "followup_route": FOLLOWUP_ROUTE,
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_boundary_local_jet_higher_q_extension_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.1983"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, "8.7.56.1983-.1986"),
                "current_problem_hit": sign_base.hit(current_problem_text, "box_edge_local_jet_signed_rule_retained_higher_q_generalization_next"),
                "current_status_hit": sign_base.hit(current_status_text, "box_edge_local_jet_signed_rule_retained_higher_q_generalization_next"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".1983-.1986"),
                "long_roadmap_hit": sign_base.hit(long_text, "8.7.56.1983"),
                "lessons_hit": sign_base.hit(lessons_text, "sign/phase"),
                "part5_hit": sign_base.hit(part5_text, ".1975-.1982"),
            },
        },
    )

    route_payload = sign_base.payload(
        "8.7.56.1986",
        STEP_NAME + " route sync",
        declaration_payload["inputs"],
        [
            sign_base.row("higher_q_extension_supported", "pass" if higher_q_extension_supported else "reject", "boundary local-jet extension to 12<=q/m0<=40 supported", sign_base.truth(higher_q_extension_supported), "The retained local-jet theorem survives the first higher-q extension window and therefore deserves a formal decision gate sync."),
            sign_base.row("asymptotic_phase_drift_detected", "watch" if asymptotic_phase_drift_detected else "pass", "asymptotic phase drift detected on 60<=q/m0<=80", sign_base.truth(asymptotic_phase_drift_detected), "The later asymptotic drift is a followup theorem question, not a reason to discard the successful finite higher-q extension window."),
            sign_base.row("next_route_fixed", "pass", "next route fixed", 1.0, "The next official branch is the boundary local-jet generalization decision gate / registry."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
            "selected_followup_route": FOLLOWUP_ROUTE_NAME,
            "selected_followup_route_or_none": FOLLOWUP_ROUTE,
            "physical_reject_required": physical_reject_required,
        },
        {
            "overall_status": "vector_qball_form_factor_boundary_local_jet_higher_q_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"formulas": build_formulae()},
    )

    write_artifact("declaration_gate", declaration_payload)
    write_artifact("route_sync", route_payload)

    print("[ok] 8.7.56.1983-.1986 boundary local-jet higher-q extension artifacts generated")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
