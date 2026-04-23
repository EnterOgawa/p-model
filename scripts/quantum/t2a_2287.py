#!/usr/bin/env python3
"""Generate 8.7.56.2287-.2290 hybrid seventh/eighth farther artifacts."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_2207 as base
from scripts.utils.windows_length_policy import build_compact_artifact_stem


PRIOR_AUDIT = (
    base.PUBLIC_OUT
    / "q_8_7_56_2279_2282_harmonic_hybrid_s7_s8_hyper_extreme_ultra_fast_declaration_gate_metrics.json"
)
PRIOR_REGISTRY = (
    base.PUBLIC_OUT
    / "q_8_7_56_2283_2286_harmonic_hybrid_s7_s8_hyper_extreme_ultra_registry_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.2287-2290"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor hybrid seventh/eighth "
    "farther audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "harmonic_hybrid_s7_s8_farther_fast",
    prefix="q",
)
base.STEM = STEM

PRIOR_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_hybrid_s7_retained_1867776_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_hybrid_s7_s8_farther_fast_gate"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_hybrid_s7s8_farther_registry"
)
NEXT_ROUTE = "8.7.56.2291"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_hybrid_selected_farther_continuation_audit"
)
FOLLOWUP_ROUTE = "8.7.56.2295"

FARTHER_BANDS = [
    (1867777, 1875968),
    (1875969, 1884160),
    (1884161, 1892352),
    (1892353, 1900544),
    (1900545, 1908736),
    (1908737, 1916928),
    (1916929, 1925120),
    (1925121, 1933312),
    (1933313, 1941504),
    (1941505, 1949696),
    (1949697, 1957888),
    (1957889, 1966080),
]
FIRST_HOLDOUT = FARTHER_BANDS[:4]
FIRST_MONITOR = FARTHER_BANDS[4:]
RESERVE_FIT = FARTHER_BANDS[:4]
RESERVE_HOLDOUT = FARTHER_BANDS[4:8]
RESERVE_MONITOR = FARTHER_BANDS[8:]


# 関数: 監査で使う公式群を返す。
def build_formulae() -> dict[str, str]:
    """Return formulas used in the hybrid farther audit."""
    return {
        "retained_lattice": "delta_q^(n,m)=delta_q,base^(box)+m_n Delta_box",
        "first_shot": "M_7(x)=a_7 x+b_7, C_7(x)=c_7 x+d_7, E_7(q)=A_7 q^{-nu_7}",
        "fastlane_scan": "Extend the retained seventh prefix until monitor error first exceeds the retained tolerances on the farther branch.",
        "reserve_fallback": "M_8(x)=a_8 x+b_8, C_8(x)=c_8 x+d_8, E_8(q)=A_8 q^{-nu_8} fitted only after the retained seventh first shot fails on the full farther window.",
    }


# 関数: `.2287-.2290` を実行する。

def main() -> None:
    """Execute the hybrid seventh/eighth farther continuation audit."""
    for path in (
        base.STATUS,
        base.ROADMAP,
        base.AI_CONTEXT,
        base.WORK_HISTORY_RECENT,
        base.CURRENT_PROBLEM,
        base.CURRENT_STATUS,
        base.UNIFIED_ROADMAP,
        base.LONG_ROADMAP,
        base.PART5,
        base.QBALL_BRANCH_REFRESH,
        PRIOR_AUDIT,
        PRIOR_REGISTRY,
    ):
        base.sign_base.require(path)

    status_text = base.sign_base.read_text(base.STATUS)
    roadmap_text = base.sign_base.read_text(base.ROADMAP)
    current_problem_text = base.sign_base.read_text(base.CURRENT_PROBLEM)
    current_status_text = base.sign_base.read_text(base.CURRENT_STATUS)
    unified_text = base.sign_base.read_text(base.UNIFIED_ROADMAP)
    long_text = base.sign_base.read_text(base.LONG_ROADMAP)
    part5_text = base.sign_base.read_text(base.PART5)
    prior_audit_summary = base.sign_base.read_json(PRIOR_AUDIT)["summary"]
    prior_registry_summary = base.sign_base.read_json(PRIOR_REGISTRY)["summary"]
    prior_gate_a_retained = bool(
        prior_registry_summary.get(
            "gate_a_same_seventh_piecewise_validation_to_1867776_retained",
            False,
        )
    )
    inventory_ready = bool(prior_gate_a_retained)

    qball_branch_refresh = base.sign_base.read_json(base.QBALL_BRANCH_REFRESH)
    scalar_ground_state = base.sign_base.extract_scalar_ground_state(qball_branch_refresh)
    qball_module = base.sign_base.load_qball_module()
    radius, field, _field_prime = qball_module.solve_full_profile(
        float(scalar_ground_state["beta_n"]),
        float(scalar_ground_state["central_amplitude"]),
    )
    weight = (field**2) * (radius**2)
    norm = float(base.np.trapezoid(weight, radius))
    bulk_delta_r, _bulk_fraction, _edge_gap = base.alias_base.bulk_grid_summary(radius)
    alias_1 = (2.0 * base.np.pi) / bulk_delta_r
    lookup_q = base.np.arange(
        0.0,
        base.phase_base.LOOKUP_Q_MAX + base.phase_base.LOOKUP_Q_STEP,
        base.phase_base.LOOKUP_Q_STEP,
        dtype=float,
    )
    lookup_values = base.phase_base.form_factor_array(radius, weight, norm, lookup_q)

    theorem_lattice_base = float(prior_audit_summary["theorem_lattice_base_over_m0"])
    theorem_lattice_step = float(prior_audit_summary["bulk_delta_r_over_m0"])
    windows = base.sparse_base.build_sampled_windows(
        radius,
        weight,
        norm,
        alias_1,
        FARTHER_BANDS,
        base.FARTHER_SAMPLE_HARMONIC_STRIDE,
    )
    results = base.lattice_base.evaluate_lattice_family(
        windows,
        lookup_q,
        lookup_values,
        theorem_lattice_base,
        theorem_lattice_step,
    )
    band_summaries = {
        f"{band_start}_{band_end}": base.sparse_base.summarize_sampled_band(
            windows,
            results,
            band_start,
            band_end,
        )
        for band_start, band_end in FARTHER_BANDS
    }
    centers, mismatches, correlations, recon_errors = base.build_series(
        band_summaries,
        FARTHER_BANDS,
    )
    x_all = base.stress_log_coordinate(centers)

    seventh_m_slope = float(prior_audit_summary["seventh_post_break_piecewise_mismatch_slope"])
    seventh_m_intercept = float(
        prior_audit_summary["seventh_post_break_piecewise_mismatch_intercept"]
    )
    seventh_c_slope = float(
        prior_audit_summary["seventh_post_break_piecewise_correlation_slope"]
    )
    seventh_c_intercept = float(
        prior_audit_summary["seventh_post_break_piecewise_correlation_intercept"]
    )
    seventh_rec_exp = float(
        prior_audit_summary["seventh_post_break_reconstruction_decay_exponent"]
    )
    seventh_rec_pref = float(
        prior_audit_summary["seventh_post_break_reconstruction_decay_prefactor"]
    )
    seventh_m_pred = (seventh_m_slope * x_all) + seventh_m_intercept
    seventh_c_pred = (seventh_c_slope * x_all) + seventh_c_intercept
    seventh_r_pred = seventh_rec_pref * base.np.power(centers, -seventh_rec_exp)

    seventh_holdout_slice = slice(0, len(FIRST_HOLDOUT))
    seventh_monitor_slice = slice(len(FIRST_HOLDOUT), len(FARTHER_BANDS))
    seventh_holdout_mismatch = base.max_abs_error(
        mismatches[seventh_holdout_slice],
        seventh_m_pred[seventh_holdout_slice],
    )
    seventh_holdout_corr = base.max_abs_error(
        correlations[seventh_holdout_slice],
        seventh_c_pred[seventh_holdout_slice],
    )
    seventh_holdout_recon = base.max_abs_error(
        recon_errors[seventh_holdout_slice],
        seventh_r_pred[seventh_holdout_slice],
    )
    seventh_monitor_mismatch = base.max_abs_error(
        mismatches[seventh_monitor_slice],
        seventh_m_pred[seventh_monitor_slice],
    )
    seventh_monitor_corr = base.max_abs_error(
        correlations[seventh_monitor_slice],
        seventh_c_pred[seventh_monitor_slice],
    )
    seventh_monitor_recon = base.max_abs_error(
        recon_errors[seventh_monitor_slice],
        seventh_r_pred[seventh_monitor_slice],
    )
    same_seventh_supported = bool(
        seventh_holdout_mismatch <= base.MISMATCH_TOL
        and seventh_holdout_corr <= base.CORRELATION_TOL
        and seventh_holdout_recon <= base.RECON_TOL
        and seventh_monitor_mismatch <= base.MISMATCH_TOL
        and seventh_monitor_corr <= base.CORRELATION_TOL
        and seventh_monitor_recon <= base.RECON_TOL
    )

    fast_max_count = len(FIRST_HOLDOUT)
    fast_first_fail_count: int | None = None
    fast_prefix_metrics: list[dict[str, float | int | bool | None]] = []
    for prefix_count in range(len(FIRST_HOLDOUT) + 1, len(FARTHER_BANDS) + 1):
        prefix_slice = slice(len(FIRST_HOLDOUT), prefix_count)
        prefix_mismatch = base.max_abs_error(
            mismatches[prefix_slice],
            seventh_m_pred[prefix_slice],
        )
        prefix_corr = base.max_abs_error(
            correlations[prefix_slice],
            seventh_c_pred[prefix_slice],
        )
        prefix_recon = base.max_abs_error(
            recon_errors[prefix_slice],
            seventh_r_pred[prefix_slice],
        )
        prefix_ok = bool(
            seventh_holdout_mismatch <= base.MISMATCH_TOL
            and seventh_holdout_corr <= base.CORRELATION_TOL
            and seventh_holdout_recon <= base.RECON_TOL
            and prefix_mismatch <= base.MISMATCH_TOL
            and prefix_corr <= base.CORRELATION_TOL
            and prefix_recon <= base.RECON_TOL
        )
        fast_prefix_metrics.append(
            {
                "band_count": prefix_count,
                "band_end_harmonic": FARTHER_BANDS[prefix_count - 1][1],
                "monitor_max_mismatch_abs_error": prefix_mismatch,
                "monitor_max_correlation_abs_error": prefix_corr,
                "monitor_max_reconstruction_abs_error": prefix_recon,
                "supported": prefix_ok,
            }
        )
        if prefix_ok:
            fast_max_count = prefix_count
        elif fast_first_fail_count is None:
            fast_first_fail_count = prefix_count

    fast_end_harmonic = FARTHER_BANDS[fast_max_count - 1][1]
    fast_fail_end_harmonic = (
        FARTHER_BANDS[fast_first_fail_count - 1][1]
        if fast_first_fail_count is not None
        else None
    )

    gate_a = same_seventh_supported
    gate_b = False
    gate_c = False
    reserve_executed = False
    eighth_m_slope: float | None = None
    eighth_m_intercept: float | None = None
    eighth_c_slope: float | None = None
    eighth_c_intercept: float | None = None
    eighth_rec_exp: float | None = None
    eighth_rec_pref: float | None = None
    eighth_holdout_mismatch: float | None = None
    eighth_holdout_corr: float | None = None
    eighth_holdout_recon: float | None = None
    eighth_monitor_mismatch: float | None = None
    eighth_monitor_corr: float | None = None
    eighth_monitor_recon: float | None = None
    eighth_supported = False
    loading_index_theorem_reserve_selected = True
    physical_reject_required = False

    if not same_seventh_supported:
        reserve_executed = True
        reserve_summaries = {
            f"{band_start}_{band_end}": band_summaries[f"{band_start}_{band_end}"]
            for band_start, band_end in RESERVE_FIT + RESERVE_HOLDOUT + RESERVE_MONITOR
        }
        reserve_centers, reserve_mismatches, reserve_correlations, reserve_recon_errors = (
            base.build_series(
                reserve_summaries,
                RESERVE_FIT + RESERVE_HOLDOUT + RESERVE_MONITOR,
            )
        )
        reserve_x = base.stress_log_coordinate(reserve_centers)
        fit_count = len(RESERVE_FIT)
        eighth_m_slope, eighth_m_intercept = base.fit_affine(
            reserve_x[:fit_count],
            reserve_mismatches[:fit_count],
        )
        eighth_c_slope, eighth_c_intercept = base.fit_affine(
            reserve_x[:fit_count],
            reserve_correlations[:fit_count],
        )
        eighth_rec_exp, eighth_rec_pref = base.fit_power_law(
            reserve_centers[:fit_count],
            reserve_recon_errors[:fit_count],
        )

        eighth_m_pred = (eighth_m_slope * reserve_x) + eighth_m_intercept
        eighth_c_pred = (eighth_c_slope * reserve_x) + eighth_c_intercept
        eighth_r_pred = eighth_rec_pref * base.np.power(
            reserve_centers,
            -eighth_rec_exp,
        )
        eighth_holdout_slice = slice(fit_count, fit_count + len(RESERVE_HOLDOUT))
        eighth_monitor_slice = slice(fit_count + len(RESERVE_HOLDOUT), len(reserve_centers))
        eighth_holdout_mismatch = base.max_abs_error(
            reserve_mismatches[eighth_holdout_slice],
            eighth_m_pred[eighth_holdout_slice],
        )
        eighth_holdout_corr = base.max_abs_error(
            reserve_correlations[eighth_holdout_slice],
            eighth_c_pred[eighth_holdout_slice],
        )
        eighth_holdout_recon = base.max_abs_error(
            reserve_recon_errors[eighth_holdout_slice],
            eighth_r_pred[eighth_holdout_slice],
        )
        eighth_monitor_mismatch = base.max_abs_error(
            reserve_mismatches[eighth_monitor_slice],
            eighth_m_pred[eighth_monitor_slice],
        )
        eighth_monitor_corr = base.max_abs_error(
            reserve_correlations[eighth_monitor_slice],
            eighth_c_pred[eighth_monitor_slice],
        )
        eighth_monitor_recon = base.max_abs_error(
            reserve_recon_errors[eighth_monitor_slice],
            eighth_r_pred[eighth_monitor_slice],
        )
        eighth_supported = bool(
            eighth_holdout_mismatch <= base.MISMATCH_TOL
            and eighth_holdout_corr <= base.CORRELATION_TOL
            and eighth_holdout_recon <= base.RECON_TOL
            and eighth_monitor_mismatch <= base.MISMATCH_TOL
            and eighth_monitor_corr <= base.CORRELATION_TOL
            and eighth_monitor_recon <= base.RECON_TOL
        )
        gate_b = eighth_supported
        gate_c = not eighth_supported

    rows = [
        base.sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "hybrid farther inventory ready",
            base.sign_base.truth(inventory_ready),
            "The hybrid farther branch starts only after the seventh route has already been retained through harmonic 1867776.",
        ),
        base.sign_base.row(
            "same_seventh_holdout_max_mismatch_abs_error",
            "pass" if seventh_holdout_mismatch <= base.MISMATCH_TOL else "watch",
            "same seventh first-shot holdout max mismatch abs error through harmonic 1900544",
            seventh_holdout_mismatch,
            "The accelerated first shot trusts the retained seventh segment only if the initial quartet stays inside the retained mismatch tolerance.",
        ),
        base.sign_base.row(
            "same_seventh_holdout_max_correlation_abs_error",
            "pass" if seventh_holdout_corr <= base.CORRELATION_TOL else "watch",
            "same seventh first-shot holdout max correlation abs error through harmonic 1900544",
            seventh_holdout_corr,
            "The sign-floor channel must stay inside tolerance before the accelerated seventh first shot can be trusted.",
        ),
        base.sign_base.row(
            "same_seventh_fast_max_band_count",
            "pass",
            "same seventh accelerated first-shot maximum supported band count",
            float(fast_max_count),
            "This is the farthest prefix length that survives before the first honest tolerance break.",
        ),
        base.sign_base.row(
            "same_seventh_fast_max_end_harmonic",
            "pass",
            "same seventh accelerated first-shot maximum supported end harmonic",
            float(fast_end_harmonic),
            "This is the farthest harmonic reached by the accelerated same-seventh route without exact fallback.",
        ),
        base.sign_base.row(
            "same_seventh_fast_first_fail_band_count",
            "watch" if fast_first_fail_count is not None else "pass",
            "same seventh accelerated first-shot first fail band count",
            float(fast_first_fail_count) if fast_first_fail_count is not None else -1.0,
            "The first failing prefix marks the natural ceiling of the accelerated same-seventh route on the current farther branch.",
        ),
        base.sign_base.row(
            "same_seventh_full_monitor_max_mismatch_abs_error",
            "pass" if seventh_monitor_mismatch <= base.MISMATCH_TOL else "watch",
            "same seventh full farther-window monitor max mismatch abs error through harmonic 1966080",
            seventh_monitor_mismatch,
            "The full farther window is the honest first-shot gate; once it fails, the exact reserve fallback is allowed to run.",
        ),
        base.sign_base.row(
            "same_seventh_full_monitor_max_correlation_abs_error",
            "pass" if seventh_monitor_corr <= base.CORRELATION_TOL else "watch",
            "same seventh full farther-window monitor max correlation abs error through harmonic 1966080",
            seventh_monitor_corr,
            "This correlation error is the current blocker for pushing the retained seventh segment blindly across the entire farther branch.",
        ),
        base.sign_base.row(
            "same_seventh_supported",
            "pass" if same_seventh_supported else "reject",
            "same seventh first-shot continuation through harmonic 1966080 supported",
            base.sign_base.truth(same_seventh_supported),
            "The accelerated route keeps the same seventh segment only if the entire farther window stays inside tolerance.",
        ),
        base.sign_base.row(
            "reserve_eighth_exact_fallback_executed",
            "watch",
            "reserve eighth exact fallback executed",
            base.sign_base.truth(reserve_executed),
            "The exact fallback is triggered only after the accelerated same-seventh first shot fails on the full farther window.",
        ),
        base.sign_base.row(
            "reserve_eighth_holdout_max_mismatch_abs_error",
            (
                "pass"
                if eighth_holdout_mismatch is not None
                and eighth_holdout_mismatch <= base.MISMATCH_TOL
                else "watch"
            ),
            "reserve eighth holdout max mismatch abs error through harmonic 1933312",
            float(eighth_holdout_mismatch) if eighth_holdout_mismatch is not None else -1.0,
            "The reserve eighth segment becomes admissible only if the exact fallback passes its own holdout quartet.",
        ),
        base.sign_base.row(
            "reserve_eighth_holdout_max_correlation_abs_error",
            (
                "pass"
                if eighth_holdout_corr is not None
                and eighth_holdout_corr <= base.CORRELATION_TOL
                else "watch"
            ),
            "reserve eighth holdout max correlation abs error through harmonic 1933312",
            float(eighth_holdout_corr) if eighth_holdout_corr is not None else -1.0,
            "The reserve eighth exact fallback is also monitored on the sign-floor channel.",
        ),
        base.sign_base.row(
            "reserve_eighth_monitor_max_mismatch_abs_error",
            (
                "pass"
                if eighth_monitor_mismatch is not None
                and eighth_monitor_mismatch <= base.MISMATCH_TOL
                else "watch"
            ),
            "reserve eighth monitor max mismatch abs error through harmonic 1966080",
            float(eighth_monitor_mismatch) if eighth_monitor_mismatch is not None else -1.0,
            "This is the decisive farther monitor for the reserve eighth exact fallback.",
        ),
        base.sign_base.row(
            "reserve_eighth_monitor_max_correlation_abs_error",
            (
                "pass"
                if eighth_monitor_corr is not None
                and eighth_monitor_corr <= base.CORRELATION_TOL
                else "watch"
            ),
            "reserve eighth monitor max correlation abs error through harmonic 1966080",
            float(eighth_monitor_corr) if eighth_monitor_corr is not None else -1.0,
            "The reserve eighth fallback must also hold on the sign-floor channel before promotion is allowed.",
        ),
        base.sign_base.row(
            "reserve_eighth_supported",
            "pass" if eighth_supported else "reject",
            "reserve eighth exact fallback admissible now",
            base.sign_base.truth(eighth_supported),
            "After the accelerated same-seventh first shot fails, the reserve eighth exact fallback becomes the honest next surface only if it passes holdout and monitor windows on the farther branch.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "theorem_lattice_base_over_m0": theorem_lattice_base,
        "bulk_delta_r_over_m0": theorem_lattice_step,
        "farther_sample_harmonic_stride": base.FARTHER_SAMPLE_HARMONIC_STRIDE,
        "seventh_post_break_piecewise_mismatch_slope": seventh_m_slope,
        "seventh_post_break_piecewise_mismatch_intercept": seventh_m_intercept,
        "seventh_post_break_piecewise_correlation_slope": seventh_c_slope,
        "seventh_post_break_piecewise_correlation_intercept": seventh_c_intercept,
        "seventh_post_break_reconstruction_decay_exponent": seventh_rec_exp,
        "seventh_post_break_reconstruction_decay_prefactor": seventh_rec_pref,
        "same_seventh_holdout_max_mismatch_abs_error": seventh_holdout_mismatch,
        "same_seventh_holdout_max_correlation_abs_error": seventh_holdout_corr,
        "same_seventh_holdout_max_reconstruction_abs_error": seventh_holdout_recon,
        "same_seventh_full_monitor_max_mismatch_abs_error": seventh_monitor_mismatch,
        "same_seventh_full_monitor_max_correlation_abs_error": seventh_monitor_corr,
        "same_seventh_full_monitor_max_reconstruction_abs_error": seventh_monitor_recon,
        "same_seventh_fast_max_band_count": fast_max_count,
        "same_seventh_fast_max_end_harmonic": fast_end_harmonic,
        "same_seventh_fast_first_fail_band_count": fast_first_fail_count,
        "same_seventh_fast_first_fail_end_harmonic": fast_fail_end_harmonic,
        "same_seventh_fast_prefix_metrics": fast_prefix_metrics,
        "same_seventh_supported": same_seventh_supported,
        "reserve_eighth_exact_fallback_executed": reserve_executed,
        "eighth_post_break_piecewise_mismatch_slope": eighth_m_slope,
        "eighth_post_break_piecewise_mismatch_intercept": eighth_m_intercept,
        "eighth_post_break_piecewise_correlation_slope": eighth_c_slope,
        "eighth_post_break_piecewise_correlation_intercept": eighth_c_intercept,
        "eighth_post_break_reconstruction_decay_exponent": eighth_rec_exp,
        "eighth_post_break_reconstruction_decay_prefactor": eighth_rec_pref,
        "reserve_eighth_holdout_max_mismatch_abs_error": eighth_holdout_mismatch,
        "reserve_eighth_holdout_max_correlation_abs_error": eighth_holdout_corr,
        "reserve_eighth_holdout_max_reconstruction_abs_error": eighth_holdout_recon,
        "reserve_eighth_monitor_max_mismatch_abs_error": eighth_monitor_mismatch,
        "reserve_eighth_monitor_max_correlation_abs_error": eighth_monitor_corr,
        "reserve_eighth_monitor_max_reconstruction_abs_error": eighth_monitor_recon,
        "reserve_eighth_supported": eighth_supported,
        "gate_a_same_seventh_piecewise_validation_to_1966080_retained": gate_a,
        "gate_b_eighth_piecewise_reactivation_selected": gate_b,
        "gate_c_substantive_pack_update_required": gate_c,
        "loading_index_theorem_reserve_selected": loading_index_theorem_reserve_selected,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = base.sign_base.payload(
        "8.7.56.2289",
        STEP_NAME + " declaration gate",
        {
            "source_files": {
                "status": base.sign_base.display_path(base.STATUS),
                "roadmap": base.sign_base.display_path(base.ROADMAP),
                "ai_context": base.sign_base.display_path(base.AI_CONTEXT),
                "work_history_recent": base.sign_base.display_path(base.WORK_HISTORY_RECENT),
                "current_problem": base.sign_base.display_path(base.CURRENT_PROBLEM),
                "current_status": base.sign_base.display_path(base.CURRENT_STATUS),
                "unified_roadmap": base.sign_base.display_path(base.UNIFIED_ROADMAP),
                "long_roadmap": base.sign_base.display_path(base.LONG_ROADMAP),
                "part5": base.sign_base.display_path(base.PART5),
                "qball_branch_refresh": base.sign_base.display_path(base.QBALL_BRANCH_REFRESH),
                "prior_audit": base.sign_base.display_path(PRIOR_AUDIT),
                "prior_registry": base.sign_base.display_path(PRIOR_REGISTRY),
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
            "overall_status": "vector_qball_form_factor_hybrid_s7_s8_farther_fast_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": base.find_line(status_text, "8.7.56.2287"),
                "roadmap_branch_hit": base.find_line(roadmap_text, ".2287-.2290"),
                "current_problem_hit": base.find_line(current_problem_text, "8.7.56.2287"),
                "current_status_hit": base.find_line(current_status_text, "8.7.56.2287"),
                "unified_roadmap_hit": base.find_line(unified_text, ".2287-.2290"),
                "long_roadmap_hit": base.find_line(long_text, ".2287-.2290"),
                "part5_hit": base.find_line(part5_text, ".2287-.2290"),
            },
        },
    )
    declaration_paths = base.write_artifact("declaration_gate", declaration_payload)
    route_payload = {
        "generated_utc": base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2290",
            "name": STEP_NAME + " route sync",
        },
        "inputs": {
            "source_files": {
                "status": base.sign_base.display_path(base.STATUS),
                "roadmap": base.sign_base.display_path(base.ROADMAP),
                "current_problem": base.sign_base.display_path(base.CURRENT_PROBLEM),
                "current_status": base.sign_base.display_path(base.CURRENT_STATUS),
                "unified_roadmap": base.sign_base.display_path(base.UNIFIED_ROADMAP),
                "long_roadmap": base.sign_base.display_path(base.LONG_ROADMAP),
                "part5": base.sign_base.display_path(base.PART5),
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
        "rows": [
            base.sign_base.row(
                "status_synced",
                "pass",
                "STATUS sync target present",
                1.0,
                "The hybrid farther audit is only honest if the official status already points to the retained seventh branch that it accelerates.",
            ),
            base.sign_base.row(
                "roadmap_synced",
                "pass",
                "ROADMAP sync target present",
                1.0,
                "The public roadmap must expose the hybrid seventh branch before route sync can proceed.",
            ),
            base.sign_base.row(
                "long_horizon_synced",
                "pass",
                "long-horizon roadmap sync target present",
                1.0,
                "The long-horizon roadmap must carry the same hybrid route so that the accelerated first shot and exact fallback remain reproducible.",
            ),
        ],
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_hybrid_s7_s8_farther_fast_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        "evidence": declaration_payload["evidence"],
    }
    route_paths = base.write_artifact("route_sync", route_payload)
    print("[write] declaration:", declaration_paths["json"])
    print("[write] route:", route_paths["json"])


if __name__ == "__main__":
    main()
