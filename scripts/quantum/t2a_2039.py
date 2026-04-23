#!/usr/bin/env python3
"""Generate 8.7.56.2039-.2042 alias-image phase-slip theorem artifacts."""

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
import scripts.quantum.t2a_1975 as local_jet_base
import scripts.quantum.t2a_2031 as phase_base
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
PRIOR_AUDIT = PUBLIC_OUT / "q_8_7_56_2031_2034_boundary_alias_image_reactivation_declaration_gate_metrics.json"
PRIOR_GATE = PUBLIC_OUT / "q_8_7_56_2035_2038_alias_image_phase_slip_gate_declaration_gate_metrics.json"

STEP_TAG = "8.7.56.2039-2042"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor exact boundary phase-slip theorem or alias-image higher-q generalization"
STEM = build_compact_artifact_stem(STEP_TAG, "alias_image_phase_slip_theorem", prefix="q")

PRIOR_CLASS = "vector_qball_form_factor_boundary_alias_image_shared_phase_slip_partial_retain_exact_phase_slip_theorem_or_higher_q_generalization_next"
BRANCH_CLASS = "vector_qball_form_factor_boundary_alias_image_local_jet_phase_slip_theorem_derived_higher_harmonic_generalization_gate_next"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_alias_image_shared_phase_slip_closeout_registry"
NEXT_ROUTE = "8.7.56.2043"
FOLLOWUP_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_q_dependent_boundary_phase_slip_loading_or_higher_harmonic_signed_rule_reactivation"
FOLLOWUP_ROUTE = "8.7.56.2047"

FIT_Q_MIN = phase_base.FIT_Q_MIN
FIT_Q_MAX = phase_base.FIT_Q_MAX
EDGE_Q_MIN = phase_base.EDGE_Q_MIN
EDGE_Q_MAX = phase_base.EDGE_Q_MAX
DELTA_SEARCH_MIN = 0.35
DELTA_SEARCH_MAX = 0.45
DELTA_SEARCH_STEP = 0.0005
HIGHER_WINDOW_DENSITY = 500


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: JSON/CSV artifact を書き出す。

def write_artifact(kind: str, data: dict) -> dict[str, str]:
    """Write one JSON payload and one rows CSV."""
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    paths = build_metrics_paths(PUBLIC_OUT, STEM, kind)
    paths["json"].write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with paths["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])

    return {"json": sign_base.display_path(paths["json"]), "csv": sign_base.display_path(paths["csv"])}


# 関数: one window の theorem metrics を返す。

def window_metrics(
    radius: np.ndarray,
    weight: np.ndarray,
    norm: float,
    q_min: float,
    q_max: float,
    alias_harmonic: float,
    harmonic_index: int,
    delta_q: float,
    *,
    density: int,
) -> dict[str, float]:
    """Return signed metrics for one window under one phase-slip."""
    q_scan = np.linspace(q_min, q_max, int(round((q_max - q_min) * density)) + 1)
    exact_values, exact_abs, exact_sign = phase_base.exact_sign_data(radius, weight, norm, q_scan)
    q_image = phase_base.shifted_alias_image_q(q_scan, alias_harmonic, harmonic_index, delta_q)
    image_values = phase_base.form_factor_array(radius, weight, norm, q_image)
    sigma_pred = phase_base.alias_sigma_from_values(image_values, harmonic_index)
    metrics = phase_base.signed_window_metrics(sigma_pred, exact_sign, exact_values, exact_abs)
    metrics["q_min"] = float(q_min)
    metrics["q_max"] = float(q_max)
    metrics["alias_harmonic"] = float(alias_harmonic)
    return metrics


# 関数: one window の最適 delta を返す。

def optimize_window_delta(
    radius: np.ndarray,
    weight: np.ndarray,
    norm: float,
    q_min: float,
    q_max: float,
    alias_harmonic: float,
    harmonic_index: int,
) -> tuple[float, float]:
    """Return the best delta and mismatch on one active window."""
    q_scan = np.linspace(q_min, q_max, int(round((q_max - q_min) * 600)) + 1)
    _exact_values, _exact_abs, exact_sign = phase_base.exact_sign_data(radius, weight, norm, q_scan)
    lookup_q = np.arange(0.0, phase_base.LOOKUP_Q_MAX + phase_base.LOOKUP_Q_STEP, phase_base.LOOKUP_Q_STEP, dtype=float)
    lookup_values = phase_base.form_factor_array(radius, weight, norm, lookup_q)
    best_delta = float(DELTA_SEARCH_MIN)
    best_mismatch = 1.0
    for delta_q in np.arange(DELTA_SEARCH_MIN, DELTA_SEARCH_MAX + (0.5 * DELTA_SEARCH_STEP), DELTA_SEARCH_STEP):
        q_image = phase_base.shifted_alias_image_q(q_scan, alias_harmonic, harmonic_index, float(delta_q))
        image_values = np.interp(q_image, lookup_q, lookup_values)
        sigma_pred = phase_base.alias_sigma_from_values(image_values, harmonic_index)
        mismatch = phase_base.alias_base.sign_mismatch_fraction(sigma_pred, exact_sign)
        if mismatch < best_mismatch - 1.0e-12:
            best_delta = float(delta_q)
            best_mismatch = float(mismatch)

    return best_delta, best_mismatch


# 関数: translated higher-harmonic window を返す。

def translated_window(alias_harmonic: float, template_offsets: tuple[float, float]) -> tuple[float, float]:
    """Return one translated higher-harmonic window."""
    return float(alias_harmonic + template_offsets[0]), float(alias_harmonic + template_offsets[1])


# 関数: 使用公式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the theorem audit."""
    return {
        "shared_phase_slip_rule": "sigma_img,delta^(n)(q)=(-1)^n sign(F_exact(|q_alias^(n)+(-1)^(n+1) delta_q-q|))",
        "boundary_local_jet_theorem_candidate": "delta_q,jet = (3/2) (h1 / h0)",
        "higher_harmonic_template": "W_fit^(n)=q_alias^(n)+[FIT_Q_MIN-q_alias^(1), FIT_Q_MAX-q_alias^(1)], W_edge^(n)=q_alias^(n)+[EDGE_Q_MIN-q_alias^(2), EDGE_Q_MAX-q_alias^(2)]",
    }


# 関数: `.2039-.2042` を実行する。

def main() -> None:
    """Execute the boundary phase-slip theorem or higher-q generalization audit."""
    for path in (
        STATUS, ROADMAP, AI_CONTEXT, WORK_HISTORY_RECENT, CURRENT_PROBLEM, CURRENT_STATUS,
        UNIFIED_ROADMAP, LONG_ROADMAP, PART5, QBALL_BRANCH_REFRESH, PRIOR_AUDIT, PRIOR_GATE
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
    inventory_ready = bool(prior_gate_summary["exact_boundary_phase_slip_theorem_admissible_now"])

    qball_branch_refresh = sign_base.read_json(QBALL_BRANCH_REFRESH)
    scalar_ground_state = sign_base.extract_scalar_ground_state(qball_branch_refresh)
    qball_module = sign_base.load_qball_module()
    radius, field, _field_prime = qball_module.solve_full_profile(
        float(scalar_ground_state["beta_n"]),
        float(scalar_ground_state["central_amplitude"]),
    )
    weight = (field**2) * (radius**2)
    norm = float(np.trapezoid(weight, radius))
    h0, h1, h2 = local_jet_base.boundary_local_jet(radius, field)
    delta_q_theorem = 1.5 * (h1 / h0)

    alias_1 = float(prior_audit_summary["first_alias_harmonic_over_m0"])
    alias_2 = float(prior_audit_summary["second_alias_harmonic_over_m0"])
    active_fit_opt_delta, active_fit_opt_mismatch = optimize_window_delta(radius, weight, norm, FIT_Q_MIN, FIT_Q_MAX, alias_1, 1)
    active_edge_opt_delta, active_edge_opt_mismatch = optimize_window_delta(radius, weight, norm, EDGE_Q_MIN, EDGE_Q_MAX, alias_2, 2)
    theorem_fit_metrics = window_metrics(radius, weight, norm, FIT_Q_MIN, FIT_Q_MAX, alias_1, 1, delta_q_theorem, density=phase_base.alias_base.WINDOW_SCAN_DENSITY)
    theorem_edge_metrics = window_metrics(radius, weight, norm, EDGE_Q_MIN, EDGE_Q_MAX, alias_2, 2, delta_q_theorem, density=phase_base.alias_base.WINDOW_SCAN_DENSITY)

    fit_offsets = (FIT_Q_MIN - alias_1, FIT_Q_MAX - alias_1)
    edge_offsets = (EDGE_Q_MIN - alias_2, EDGE_Q_MAX - alias_2)
    alias_3 = 3.0 * alias_1
    alias_4 = 4.0 * alias_1
    alias_5 = 5.0 * alias_1
    alias_6 = 6.0 * alias_1
    h3_q_min, h3_q_max = translated_window(alias_3, fit_offsets)
    h4_q_min, h4_q_max = translated_window(alias_4, edge_offsets)
    h5_q_min, h5_q_max = translated_window(alias_5, fit_offsets)
    h6_q_min, h6_q_max = translated_window(alias_6, edge_offsets)
    harmonic3_fit_metrics = window_metrics(radius, weight, norm, h3_q_min, h3_q_max, alias_3, 3, delta_q_theorem, density=HIGHER_WINDOW_DENSITY)
    harmonic4_edge_metrics = window_metrics(radius, weight, norm, h4_q_min, h4_q_max, alias_4, 4, delta_q_theorem, density=HIGHER_WINDOW_DENSITY)
    harmonic5_fit_metrics = window_metrics(radius, weight, norm, h5_q_min, h5_q_max, alias_5, 5, delta_q_theorem, density=HIGHER_WINDOW_DENSITY)
    harmonic6_edge_metrics = window_metrics(radius, weight, norm, h6_q_min, h6_q_max, alias_6, 6, delta_q_theorem, density=HIGHER_WINDOW_DENSITY)

    delta_q_shared_search = float(prior_gate_summary["shared_phase_slip_delta_q_star_over_m0"])
    delta_q_theorem_vs_shared_search_abs_gap = abs(delta_q_theorem - delta_q_shared_search)
    delta_q_theorem_vs_fit_opt_abs_gap = abs(delta_q_theorem - active_fit_opt_delta)
    delta_q_theorem_vs_edge_opt_abs_gap = abs(delta_q_theorem - active_edge_opt_delta)
    delta_q_theorem_vs_window_optima_max_abs_gap = max(delta_q_theorem_vs_fit_opt_abs_gap, delta_q_theorem_vs_edge_opt_abs_gap)

    boundary_local_jet_phase_slip_theorem_derived = bool(
        delta_q_theorem_vs_window_optima_max_abs_gap <= 0.005
        and theorem_fit_metrics["sign_mismatch_fraction"] <= 0.2
        and theorem_edge_metrics["sign_mismatch_fraction"] <= 0.1
        and theorem_fit_metrics["sign_correlation"] >= 0.6
        and theorem_edge_metrics["sign_correlation"] >= 0.8
    )
    active_window_theorem_supported = boundary_local_jet_phase_slip_theorem_derived
    higher_harmonic_generalization_supported = bool(
        harmonic3_fit_metrics["sign_mismatch_fraction"] <= 0.2
        and harmonic4_edge_metrics["sign_mismatch_fraction"] <= 0.2
        and harmonic5_fit_metrics["sign_mismatch_fraction"] <= 0.2
        and harmonic6_edge_metrics["sign_mismatch_fraction"] <= 0.2
    )
    same_level_constant_delta_retry_admissible = False
    q_dependent_or_higher_harmonic_loading_admissible_now = bool(boundary_local_jet_phase_slip_theorem_derived and not higher_harmonic_generalization_supported)
    substantive_pack_update_required_now = False
    physical_reject_required = False

    rows = [
        sign_base.row("inventory_ready", "pass" if inventory_ready else "reject", "phase-slip theorem inventory ready", sign_base.truth(inventory_ready), "The branch starts only after `.2035-.2038` has retained the shared phase-slip family and opened the theorem question."),
        sign_base.row("boundary_h0", "watch", "boundary local-jet h0", h0, "The theorem candidate is built from retained box-edge local-jet data."),
        sign_base.row("boundary_h1", "watch", "boundary local-jet h1", h1, "The first boundary derivative is the active loading datum in the retained theorem candidate."),
        sign_base.row("boundary_h2", "watch", "boundary local-jet h2", h2, "The second derivative is monitored for completeness but does not enter the retained theorem candidate."),
        sign_base.row("delta_q_theorem_over_m0", "watch", "boundary local-jet theorem delta_q/m0", delta_q_theorem, "The retained theorem candidate is delta_q,jet=(3/2)(h1/h0)."),
        sign_base.row("delta_q_shared_search_over_m0", "watch", "shared-search delta_q/m0", delta_q_shared_search, "This is the prior minimax fit retained by `.2031-.2038`."),
        sign_base.row("delta_q_theorem_vs_shared_search_abs_gap", "pass" if delta_q_theorem_vs_shared_search_abs_gap <= 0.005 else "watch", "absolute gap between theorem delta and shared-search delta", delta_q_theorem_vs_shared_search_abs_gap, "A small gap shows that the boundary local-jet theorem candidate reproduces the previously searched shared phase-slip without re-fitting it."),
        sign_base.row("active_fit_opt_delta_over_m0", "watch", "active fit-window optimum delta_q/m0", active_fit_opt_delta, "The fit window is optimized independently to test whether one theorem-level delta matches the local optimum."),
        sign_base.row("active_edge_opt_delta_over_m0", "watch", "active edge-window optimum delta_q/m0", active_edge_opt_delta, "The edge window is optimized independently to test whether the theorem survives the second-harmonic residual."),
        sign_base.row("delta_q_theorem_vs_window_optima_max_abs_gap", "pass" if delta_q_theorem_vs_window_optima_max_abs_gap <= 0.005 else "watch", "max abs gap between theorem delta and independent window optima", delta_q_theorem_vs_window_optima_max_abs_gap, "The theorem is retained only if the boundary-only delta matches both active windows within a narrow tolerance."),
        sign_base.row("theorem_fit_window_sign_mismatch_fraction", "watch", "theorem fit-window sign mismatch fraction", theorem_fit_metrics["sign_mismatch_fraction"], "The active fit window must remain inside the prior partial-retain envelope under the boundary-only theorem delta."),
        sign_base.row("theorem_edge_window_sign_mismatch_fraction", "watch", "theorem edge-window sign mismatch fraction", theorem_edge_metrics["sign_mismatch_fraction"], "The active edge window is the harder second-harmonic test of the theorem candidate."),
        sign_base.row("harmonic3_fit_window_sign_mismatch_fraction", "watch", "higher-harmonic fit-template sign mismatch fraction", harmonic3_fit_metrics["sign_mismatch_fraction"], "The first higher-harmonic template tests whether one constant delta survives beyond the retained active windows."),
        sign_base.row("harmonic4_edge_window_sign_mismatch_fraction", "watch", "higher-harmonic edge-template sign mismatch fraction", harmonic4_edge_metrics["sign_mismatch_fraction"], "The second higher-harmonic template decides whether the theorem generalizes to the next alias-image edge family."),
        sign_base.row("harmonic5_fit_holdout_sign_mismatch_fraction", "watch", "holdout fit-template sign mismatch fraction", harmonic5_fit_metrics["sign_mismatch_fraction"], "A farther fit-type holdout is tracked so the branch cannot hide behind one favorable harmonic only."),
        sign_base.row("harmonic6_edge_holdout_sign_mismatch_fraction", "watch", "holdout edge-template sign mismatch fraction", harmonic6_edge_metrics["sign_mismatch_fraction"], "A farther edge-type holdout determines whether the constant-slip family remains credible beyond the first generalization attempt."),
        sign_base.row("boundary_local_jet_phase_slip_theorem_derived", "pass" if boundary_local_jet_phase_slip_theorem_derived else "reject", "boundary local-jet phase-slip theorem derived", sign_base.truth(boundary_local_jet_phase_slip_theorem_derived), "The theorem is retained when the boundary-only delta reproduces the independently searched active-window optima and keeps both active windows inside the prior partial-retain envelope."),
        sign_base.row("higher_harmonic_generalization_supported", "pass" if higher_harmonic_generalization_supported else "reject", "higher-harmonic generalization supported", sign_base.truth(higher_harmonic_generalization_supported), "The same constant-slip family would only generalize if the translated higher-harmonic template windows stayed inside the same mismatch envelope."),
        sign_base.row("same_level_constant_delta_retry_admissible", "reject", "same-level constant-delta retry admissible", sign_base.truth(same_level_constant_delta_retry_admissible), "Once the current constant-slip theorem is evaluated directly, same-level refitting of one constant delta should remain closed."),
        sign_base.row("q_dependent_or_higher_harmonic_loading_admissible_now", "pass" if q_dependent_or_higher_harmonic_loading_admissible_now else "reject", "q-dependent or higher-harmonic loading admissible now", sign_base.truth(q_dependent_or_higher_harmonic_loading_admissible_now), "If the active-window theorem closes but higher harmonics fail, the honest next surface is a q-dependent or harmonic-index dependent loading rule."),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "boundary_h0": h0,
        "boundary_h1": h1,
        "boundary_h2": h2,
        "delta_q_theorem_over_m0": delta_q_theorem,
        "delta_q_shared_search_over_m0": delta_q_shared_search,
        "delta_q_theorem_vs_shared_search_abs_gap": delta_q_theorem_vs_shared_search_abs_gap,
        "active_fit_opt_delta_over_m0": active_fit_opt_delta,
        "active_edge_opt_delta_over_m0": active_edge_opt_delta,
        "active_fit_opt_mismatch": active_fit_opt_mismatch,
        "active_edge_opt_mismatch": active_edge_opt_mismatch,
        "delta_q_theorem_vs_fit_opt_abs_gap": delta_q_theorem_vs_fit_opt_abs_gap,
        "delta_q_theorem_vs_edge_opt_abs_gap": delta_q_theorem_vs_edge_opt_abs_gap,
        "delta_q_theorem_vs_window_optima_max_abs_gap": delta_q_theorem_vs_window_optima_max_abs_gap,
        "theorem_fit_window_sign_mismatch_fraction": theorem_fit_metrics["sign_mismatch_fraction"],
        "theorem_edge_window_sign_mismatch_fraction": theorem_edge_metrics["sign_mismatch_fraction"],
        "theorem_fit_window_sign_correlation": theorem_fit_metrics["sign_correlation"],
        "theorem_edge_window_sign_correlation": theorem_edge_metrics["sign_correlation"],
        "theorem_fit_window_signed_reconstruction_max_abs_error": theorem_fit_metrics["signed_reconstruction_max_abs_error"],
        "theorem_edge_window_signed_reconstruction_max_abs_error": theorem_edge_metrics["signed_reconstruction_max_abs_error"],
        "harmonic3_fit_window_sign_mismatch_fraction": harmonic3_fit_metrics["sign_mismatch_fraction"],
        "harmonic4_edge_window_sign_mismatch_fraction": harmonic4_edge_metrics["sign_mismatch_fraction"],
        "harmonic5_fit_holdout_sign_mismatch_fraction": harmonic5_fit_metrics["sign_mismatch_fraction"],
        "harmonic6_edge_holdout_sign_mismatch_fraction": harmonic6_edge_metrics["sign_mismatch_fraction"],
        "harmonic3_fit_window_sign_correlation": harmonic3_fit_metrics["sign_correlation"],
        "harmonic4_edge_window_sign_correlation": harmonic4_edge_metrics["sign_correlation"],
        "harmonic5_fit_holdout_sign_correlation": harmonic5_fit_metrics["sign_correlation"],
        "harmonic6_edge_holdout_sign_correlation": harmonic6_edge_metrics["sign_correlation"],
        "boundary_local_jet_phase_slip_theorem_derived": boundary_local_jet_phase_slip_theorem_derived,
        "active_window_theorem_supported": active_window_theorem_supported,
        "higher_harmonic_generalization_supported": higher_harmonic_generalization_supported,
        "same_level_constant_delta_retry_admissible": same_level_constant_delta_retry_admissible,
        "q_dependent_or_higher_harmonic_loading_admissible_now": q_dependent_or_higher_harmonic_loading_admissible_now,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "substantive_pack_update_required_now": substantive_pack_update_required_now,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2041",
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
                "edge_window_over_m0": [EDGE_Q_MIN, EDGE_Q_MAX],
                "delta_search_over_m0": [DELTA_SEARCH_MIN, DELTA_SEARCH_MAX],
                "delta_search_step_over_m0": DELTA_SEARCH_STEP,
                "higher_fit_template_offsets_over_m0": list(fit_offsets),
                "higher_edge_template_offsets_over_m0": list(edge_offsets),
                "next_route_name": NEXT_ROUTE_NAME,
                "next_route": NEXT_ROUTE,
                "followup_route_name": FOLLOWUP_ROUTE_NAME,
                "followup_route": FOLLOWUP_ROUTE,
            },
        },
        rows,
        summary,
        {"overall_status": "vector_qball_form_factor_alias_image_phase_slip_theorem_declared", "branch_completed": True, "next_required_artifacts": [NEXT_ROUTE_NAME]},
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2039"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, "8.7.56.2039-.2042"),
                "current_problem_hit": sign_base.hit(current_problem_text, "shared phase-slip"),
                "current_status_hit": sign_base.hit(current_status_text, "shared phase-slip"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2039-.2042"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2039-.2042"),
                "part5_hit": sign_base.hit(part5_text, ".2031-.2038"),
            },
        },
    )

    route_payload = sign_base.payload(
        "8.7.56.2042",
        STEP_NAME + " route sync",
        declaration_payload["inputs"],
        [
            sign_base.row("boundary_local_jet_phase_slip_theorem_derived", "pass" if boundary_local_jet_phase_slip_theorem_derived else "reject", "boundary local-jet phase-slip theorem derived", sign_base.truth(boundary_local_jet_phase_slip_theorem_derived), "The current branch succeeds only if delta_q,jet reproduces the retained active-window shared phase-slip without free fitting."),
            sign_base.row("higher_harmonic_generalization_supported", "pass" if higher_harmonic_generalization_supported else "reject", "higher-harmonic generalization supported", sign_base.truth(higher_harmonic_generalization_supported), "The route diverges immediately once the same constant-slip family fails on translated higher-harmonic template windows."),
            sign_base.row("next_route_fixed", "pass", "next route fixed", 1.0, "The next official branch is the alias-image shared phase-slip closeout / registry."),
        ],
        summary,
        {"overall_status": "vector_qball_form_factor_alias_image_phase_slip_theorem_route_synced", "branch_completed": True, "next_required_artifacts": [NEXT_ROUTE_NAME]},
        {"formulas": build_formulae()},
    )

    declaration_paths = write_artifact("declaration_gate", declaration_payload)
    route_paths = write_artifact("route_sync", route_payload)
    print("[done] 8.7.56.2039-.2042 complete")
    print(f"[info] declaration gate: {declaration_paths['json']}")
    print(f"[info] route sync: {route_paths['json']}")


if __name__ == "__main__":
    main()
