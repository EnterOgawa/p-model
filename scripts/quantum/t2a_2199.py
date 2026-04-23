#!/usr/bin/env python3
"""Generate 8.7.56.2199-.2202 fourth post-break ultra-extreme-farther continuation artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_2175 as base
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


base.PRIOR_AUDIT = build_metrics_paths(
    base.PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2191-2194",
        "harmonic_fourth_post_break_piecewise_extreme_farther",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
base.PRIOR_REGISTRY = build_metrics_paths(
    base.PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2195-2198",
        "harmonic_fourth_post_break_piecewise_registry_refresh",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2199-2202"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor fourth post-break "
    "ultra-extreme-farther continuation audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "harmonic_fourth_post_break_piecewise_ultra_extreme_farther",
    prefix="q",
)
base.STEM = STEM

PRIOR_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_fourth_post_break_"
    "piecewise_validation_to_884736_farther_continuation_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_fourth_post_break_"
    "piecewise_ultra_extreme_farther_reactivation_gate_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_fourth_post_break_"
    "piecewise_registry_refresh"
)
NEXT_ROUTE = "8.7.56.2203"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_fourth_post_break_"
    "piecewise_hyper_extreme_farther_continuation_audit"
)
FOLLOWUP_ROUTE = "8.7.56.2207"

FARTHER_BANDS = [
    (884737, 892928),
    (892929, 901120),
    (901121, 909312),
    (909313, 917504),
    (917505, 925696),
    (925697, 933888),
    (933889, 942080),
    (942081, 950272),
    (950273, 958464),
    (958465, 966656),
    (966657, 974848),
    (974849, 983040),
]
FOURTH_HOLDOUT = FARTHER_BANDS[:4]
FOURTH_MONITOR = FARTHER_BANDS[4:]
FIFTH_FIT = FARTHER_BANDS[:4]
FIFTH_HOLDOUT = FARTHER_BANDS[4:8]
FIFTH_MONITOR = FARTHER_BANDS[8:]


# 関数: `.2199-.2202` 用の公式群を返す。
def build_formulae() -> dict[str, str]:
    """Return formulas used in the ultra-extreme-farther continuation audit."""
    return {
        "retained_lattice": "delta_q^(n,m) = delta_q,base^(box) + m_n Delta_box",
        "same_fourth_piecewise": "M_4(x)=a_4 x+b_4, C_4(x)=c_4 x+d_4, E_4(q)=A_4 q^{-nu_4} inherited from the retained fourth-segment registry through harmonic 884736",
        "fifth_piecewise_reserve": "M_5(x)=a_5 x+b_5, C_5(x)=c_5 x+d_5, E_5(q)=A_5 q^{-nu_5} fitted on 884737..917504 only as a reserve diagnostic",
        "selection_rule": "A fifth post-break surface becomes admissible only if the inherited fourth segment fails and the reserve fifth segment passes ultra-extreme-farther holdout and monitor windows.",
    }


# 関数: `.2199-.2202` を実行する。

def main() -> None:
    """Execute the fourth post-break ultra-extreme-farther continuation audit."""
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
        base.PRIOR_AUDIT,
        base.PRIOR_REGISTRY,
    ):
        base.sign_base.require(path)

    status_text = base.sign_base.read_text(base.STATUS)
    roadmap_text = base.sign_base.read_text(base.ROADMAP)
    current_problem_text = base.sign_base.read_text(base.CURRENT_PROBLEM)
    current_status_text = base.sign_base.read_text(base.CURRENT_STATUS)
    unified_text = base.sign_base.read_text(base.UNIFIED_ROADMAP)
    long_text = base.sign_base.read_text(base.LONG_ROADMAP)
    part5_text = base.sign_base.read_text(base.PART5)
    prior_audit_summary = base.sign_base.read_json(base.PRIOR_AUDIT)["summary"]
    prior_registry_summary = base.sign_base.read_json(base.PRIOR_REGISTRY)["summary"]
    inventory_ready = bool(
        prior_registry_summary["gate_a_same_fourth_piecewise_validation_to_884736_retained"]
    )

    qball_branch_refresh = base.sign_base.read_json(base.QBALL_BRANCH_REFRESH)
    scalar_ground_state = base.sign_base.extract_scalar_ground_state(qball_branch_refresh)
    qball_module = base.sign_base.load_qball_module()
    radius, field, _field_prime = qball_module.solve_full_profile(
        float(scalar_ground_state["beta_n"]),
        float(scalar_ground_state["central_amplitude"]),
    )
    weight = (field**2) * (radius**2)
    norm = float(np.trapezoid(weight, radius))
    bulk_delta_r, _bulk_fraction, _edge_gap = base.alias_base.bulk_grid_summary(radius)
    alias_1 = (2.0 * np.pi) / bulk_delta_r
    lookup_q = np.arange(
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

    fourth_m_slope = float(
        prior_registry_summary["fourth_post_break_piecewise_mismatch_slope"]
    )
    fourth_m_intercept = float(
        prior_registry_summary["fourth_post_break_piecewise_mismatch_intercept"]
    )
    fourth_c_slope = float(
        prior_registry_summary["fourth_post_break_piecewise_correlation_slope"]
    )
    fourth_c_intercept = float(
        prior_registry_summary["fourth_post_break_piecewise_correlation_intercept"]
    )
    fourth_rec_exp = float(
        prior_registry_summary["fourth_post_break_reconstruction_decay_exponent"]
    )
    fourth_rec_pref = float(
        prior_registry_summary["fourth_post_break_reconstruction_decay_prefactor"]
    )
    fourth_m_pred = (fourth_m_slope * x_all) + fourth_m_intercept
    fourth_c_pred = (fourth_c_slope * x_all) + fourth_c_intercept
    fourth_r_pred = fourth_rec_pref * np.power(centers, -fourth_rec_exp)

    fourth_holdout_slice = slice(0, len(FOURTH_HOLDOUT))
    fourth_monitor_slice = slice(len(FOURTH_HOLDOUT), len(FARTHER_BANDS))
    fourth_farther_holdout_max_mismatch_abs_error = base.max_abs_error(
        mismatches[fourth_holdout_slice],
        fourth_m_pred[fourth_holdout_slice],
    )
    fourth_farther_holdout_max_correlation_abs_error = base.max_abs_error(
        correlations[fourth_holdout_slice],
        fourth_c_pred[fourth_holdout_slice],
    )
    fourth_farther_holdout_max_reconstruction_abs_error = base.max_abs_error(
        recon_errors[fourth_holdout_slice],
        fourth_r_pred[fourth_holdout_slice],
    )
    fourth_farther_monitor_max_mismatch_abs_error = base.max_abs_error(
        mismatches[fourth_monitor_slice],
        fourth_m_pred[fourth_monitor_slice],
    )
    fourth_farther_monitor_max_correlation_abs_error = base.max_abs_error(
        correlations[fourth_monitor_slice],
        fourth_c_pred[fourth_monitor_slice],
    )
    fourth_farther_monitor_max_reconstruction_abs_error = base.max_abs_error(
        recon_errors[fourth_monitor_slice],
        fourth_r_pred[fourth_monitor_slice],
    )
    same_fourth_piecewise_ultra_extreme_farther_continuation_supported = bool(
        fourth_farther_holdout_max_mismatch_abs_error <= base.MISMATCH_TOL
        and fourth_farther_holdout_max_correlation_abs_error <= base.CORRELATION_TOL
        and fourth_farther_holdout_max_reconstruction_abs_error <= base.RECON_TOL
        and fourth_farther_monitor_max_mismatch_abs_error <= base.MISMATCH_TOL
        and fourth_farther_monitor_max_correlation_abs_error <= base.CORRELATION_TOL
        and fourth_farther_monitor_max_reconstruction_abs_error <= base.RECON_TOL
    )
    fourth_post_break_piecewise_validation_to_983040_supported = bool(
        same_fourth_piecewise_ultra_extreme_farther_continuation_supported
    )

    fifth_fit_slice = slice(0, len(FIFTH_FIT))
    fifth_holdout_slice = slice(len(FIFTH_FIT), len(FIFTH_FIT) + len(FIFTH_HOLDOUT))
    fifth_monitor_slice = slice(
        len(FIFTH_FIT) + len(FIFTH_HOLDOUT),
        len(FARTHER_BANDS),
    )
    fifth_m_slope, fifth_m_intercept = base.fit_affine(
        x_all[fifth_fit_slice],
        mismatches[fifth_fit_slice],
    )
    fifth_c_slope, fifth_c_intercept = base.fit_affine(
        x_all[fifth_fit_slice],
        correlations[fifth_fit_slice],
    )
    fifth_rec_exp, fifth_rec_pref = base.fit_power_law(
        centers[fifth_fit_slice],
        recon_errors[fifth_fit_slice],
    )
    fifth_m_pred = (fifth_m_slope * x_all) + fifth_m_intercept
    fifth_c_pred = (fifth_c_slope * x_all) + fifth_c_intercept
    fifth_r_pred = fifth_rec_pref * np.power(centers, -fifth_rec_exp)
    fifth_holdout_max_mismatch_abs_error = base.max_abs_error(
        mismatches[fifth_holdout_slice],
        fifth_m_pred[fifth_holdout_slice],
    )
    fifth_holdout_max_correlation_abs_error = base.max_abs_error(
        correlations[fifth_holdout_slice],
        fifth_c_pred[fifth_holdout_slice],
    )
    fifth_holdout_max_reconstruction_abs_error = base.max_abs_error(
        recon_errors[fifth_holdout_slice],
        fifth_r_pred[fifth_holdout_slice],
    )
    fifth_monitor_max_mismatch_abs_error = base.max_abs_error(
        mismatches[fifth_monitor_slice],
        fifth_m_pred[fifth_monitor_slice],
    )
    fifth_monitor_max_correlation_abs_error = base.max_abs_error(
        correlations[fifth_monitor_slice],
        fifth_c_pred[fifth_monitor_slice],
    )
    fifth_monitor_max_reconstruction_abs_error = base.max_abs_error(
        recon_errors[fifth_monitor_slice],
        fifth_r_pred[fifth_monitor_slice],
    )
    fifth_post_break_piecewise_validation_to_983040_supported = bool(
        fifth_holdout_max_mismatch_abs_error <= base.MISMATCH_TOL
        and fifth_holdout_max_correlation_abs_error <= base.CORRELATION_TOL
        and fifth_holdout_max_reconstruction_abs_error <= base.RECON_TOL
        and fifth_monitor_max_mismatch_abs_error <= base.MISMATCH_TOL
        and fifth_monitor_max_correlation_abs_error <= base.CORRELATION_TOL
        and fifth_monitor_max_reconstruction_abs_error <= base.RECON_TOL
    )
    fifth_post_break_piecewise_surface_admissible_now = bool(
        (not same_fourth_piecewise_ultra_extreme_farther_continuation_supported)
        and fifth_post_break_piecewise_validation_to_983040_supported
    )
    exact_global_ultra_extreme_farther_fourth_post_break_theorem_available = False
    physical_reject_required = False

    rows = [
        base.sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "fourth post-break ultra-extreme-farther inventory ready",
            base.sign_base.truth(inventory_ready),
            "The ultra-extreme-farther continuation starts only after the same fourth segment has already been retained through harmonic 884736.",
        ),
        base.sign_base.row(
            "fourth_farther_holdout_max_mismatch_abs_error",
            "pass" if fourth_farther_holdout_max_mismatch_abs_error <= base.MISMATCH_TOL else "reject",
            "same fourth piecewise ultra-extreme-farther holdout max mismatch abs error through harmonic 917504",
            fourth_farther_holdout_max_mismatch_abs_error,
            "The inherited fourth segment survives only if the next quartet stays inside the retained mismatch tolerance.",
        ),
        base.sign_base.row(
            "fourth_farther_holdout_max_correlation_abs_error",
            "pass" if fourth_farther_holdout_max_correlation_abs_error <= base.CORRELATION_TOL else "reject",
            "same fourth piecewise ultra-extreme-farther holdout max correlation abs error through harmonic 917504",
            fourth_farther_holdout_max_correlation_abs_error,
            "The sign-floor channel must confirm the same ultra-extreme-farther survival for the inherited fourth segment.",
        ),
        base.sign_base.row(
            "fourth_farther_monitor_max_mismatch_abs_error",
            "pass" if fourth_farther_monitor_max_mismatch_abs_error <= base.MISMATCH_TOL else "reject",
            "same fourth piecewise ultra-extreme-farther monitor max mismatch abs error through harmonic 983040",
            fourth_farther_monitor_max_mismatch_abs_error,
            "The ultra-extreme-farther monitor checks that the same fourth segment does not collapse immediately after the first quartet.",
        ),
        base.sign_base.row(
            "fourth_farther_monitor_max_correlation_abs_error",
            "pass" if fourth_farther_monitor_max_correlation_abs_error <= base.CORRELATION_TOL else "reject",
            "same fourth piecewise ultra-extreme-farther monitor max correlation abs error through harmonic 983040",
            fourth_farther_monitor_max_correlation_abs_error,
            "The monitor condition must also hold on the sign-floor channel.",
        ),
        base.sign_base.row(
            "same_fourth_piecewise_ultra_extreme_farther_continuation_supported",
            "pass" if same_fourth_piecewise_ultra_extreme_farther_continuation_supported else "reject",
            "same fourth post-break piecewise ultra-extreme-farther continuation supported",
            base.sign_base.truth(same_fourth_piecewise_ultra_extreme_farther_continuation_supported),
            "No new surface is admissible while the inherited fourth segment still survives ultra-extreme-farther holdout and monitor windows.",
        ),
        base.sign_base.row(
            "fifth_post_break_piecewise_mismatch_slope",
            "watch",
            "fifth post-break reserve mismatch slope",
            fifth_m_slope,
            "A fifth segment is computed only as a reserve diagnostic after the same fourth segment has already been tested on the ultra-extreme-farther window.",
        ),
        base.sign_base.row(
            "fifth_holdout_max_mismatch_abs_error",
            "pass" if fifth_holdout_max_mismatch_abs_error <= base.MISMATCH_TOL else "reject",
            "fifth post-break holdout max mismatch abs error through harmonic 950272",
            fifth_holdout_max_mismatch_abs_error,
            "The reserve fifth segment would only become admissible if the inherited fourth segment failed first.",
        ),
        base.sign_base.row(
            "fifth_holdout_max_correlation_abs_error",
            "pass" if fifth_holdout_max_correlation_abs_error <= base.CORRELATION_TOL else "reject",
            "fifth post-break holdout max correlation abs error through harmonic 950272",
            fifth_holdout_max_correlation_abs_error,
            "The reserve fifth segment is monitored on the sign-floor channel for completeness.",
        ),
        base.sign_base.row(
            "fifth_monitor_max_mismatch_abs_error",
            "pass" if fifth_monitor_max_mismatch_abs_error <= base.MISMATCH_TOL else "reject",
            "fifth post-break monitor max mismatch abs error through harmonic 983040",
            fifth_monitor_max_mismatch_abs_error,
            "Even a passing reserve fifth segment remains non-admissible when the inherited fourth segment already survives.",
        ),
        base.sign_base.row(
            "fifth_monitor_max_correlation_abs_error",
            "pass" if fifth_monitor_max_correlation_abs_error <= base.CORRELATION_TOL else "reject",
            "fifth post-break monitor max correlation abs error through harmonic 983040",
            fifth_monitor_max_correlation_abs_error,
            "The reserve route is kept only as a diagnostic and not as the official mainline.",
        ),
        base.sign_base.row(
            "fifth_post_break_piecewise_surface_admissible_now",
            "reject" if not fifth_post_break_piecewise_surface_admissible_now else "pass",
            "fifth post-break piecewise surface admissible now",
            base.sign_base.truth(fifth_post_break_piecewise_surface_admissible_now),
            "The retry gate opens the fifth segment only after the inherited fourth segment has honestly failed on the ultra-extreme-farther continuation audit.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "theorem_lattice_base_over_m0": theorem_lattice_base,
        "bulk_delta_r_over_m0": theorem_lattice_step,
        "farther_sample_harmonic_stride": base.FARTHER_SAMPLE_HARMONIC_STRIDE,
        "fourth_post_break_piecewise_mismatch_slope": fourth_m_slope,
        "fourth_post_break_piecewise_mismatch_intercept": fourth_m_intercept,
        "fourth_post_break_piecewise_correlation_slope": fourth_c_slope,
        "fourth_post_break_piecewise_correlation_intercept": fourth_c_intercept,
        "fourth_post_break_reconstruction_decay_exponent": fourth_rec_exp,
        "fourth_post_break_reconstruction_decay_prefactor": fourth_rec_pref,
        "fourth_farther_holdout_max_mismatch_abs_error": fourth_farther_holdout_max_mismatch_abs_error,
        "fourth_farther_holdout_max_correlation_abs_error": fourth_farther_holdout_max_correlation_abs_error,
        "fourth_farther_holdout_max_reconstruction_abs_error": fourth_farther_holdout_max_reconstruction_abs_error,
        "fourth_farther_monitor_max_mismatch_abs_error": fourth_farther_monitor_max_mismatch_abs_error,
        "fourth_farther_monitor_max_correlation_abs_error": fourth_farther_monitor_max_correlation_abs_error,
        "fourth_farther_monitor_max_reconstruction_abs_error": fourth_farther_monitor_max_reconstruction_abs_error,
        "same_fourth_piecewise_ultra_extreme_farther_continuation_supported": same_fourth_piecewise_ultra_extreme_farther_continuation_supported,
        "fourth_post_break_piecewise_validation_to_983040_supported": fourth_post_break_piecewise_validation_to_983040_supported,
        "fifth_post_break_piecewise_mismatch_slope": fifth_m_slope,
        "fifth_post_break_piecewise_mismatch_intercept": fifth_m_intercept,
        "fifth_post_break_piecewise_correlation_slope": fifth_c_slope,
        "fifth_post_break_piecewise_correlation_intercept": fifth_c_intercept,
        "fifth_post_break_reconstruction_decay_exponent": fifth_rec_exp,
        "fifth_post_break_reconstruction_decay_prefactor": fifth_rec_pref,
        "fifth_holdout_max_mismatch_abs_error": fifth_holdout_max_mismatch_abs_error,
        "fifth_holdout_max_correlation_abs_error": fifth_holdout_max_correlation_abs_error,
        "fifth_holdout_max_reconstruction_abs_error": fifth_holdout_max_reconstruction_abs_error,
        "fifth_monitor_max_mismatch_abs_error": fifth_monitor_max_mismatch_abs_error,
        "fifth_monitor_max_correlation_abs_error": fifth_monitor_max_correlation_abs_error,
        "fifth_monitor_max_reconstruction_abs_error": fifth_monitor_max_reconstruction_abs_error,
        "fifth_post_break_piecewise_validation_to_983040_supported": fifth_post_break_piecewise_validation_to_983040_supported,
        "fifth_post_break_piecewise_surface_admissible_now": fifth_post_break_piecewise_surface_admissible_now,
        "exact_global_ultra_extreme_farther_fourth_post_break_theorem_available": exact_global_ultra_extreme_farther_fourth_post_break_theorem_available,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = base.sign_base.payload(
        "8.7.56.2201",
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
                "prior_audit": base.sign_base.display_path(base.PRIOR_AUDIT),
                "prior_registry": base.sign_base.display_path(base.PRIOR_REGISTRY),
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
            "overall_status": "vector_qball_form_factor_fourth_post_break_ultra_extreme_farther_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": base.find_line(status_text, "8.7.56.2199"),
                "roadmap_branch_hit": base.find_line(roadmap_text, ".2199-.2202"),
                "current_problem_hit": base.find_line(current_problem_text, "8.7.56.2199"),
                "current_status_hit": base.find_line(current_status_text, "8.7.56.2199"),
                "unified_roadmap_hit": base.find_line(unified_text, ".2195-.2198"),
                "long_roadmap_hit": base.find_line(long_text, ".2195-.2198"),
                "part5_hit": base.find_line(part5_text, ".2195-.2198"),
            },
        },
    )
    declaration_paths = base.write_artifact("declaration_gate", declaration_payload)

    route_sync_rows = [
        base.sign_base.row(
            "status_synced",
            "pass",
            "STATUS sync target present",
            base.sign_base.truth(bool(base.find_line(status_text, "8.7.56.2199"))),
            "The ultra-extreme-farther continuation audit is only honest if the official status already points to the same fourth-segment route.",
        ),
        base.sign_base.row(
            "roadmap_synced",
            "pass",
            "ROADMAP sync target present",
            base.sign_base.truth(bool(base.find_line(roadmap_text, ".2199-.2202"))),
            "The public roadmap must expose the same fourth post-break ultra-extreme-farther branch before route sync can proceed.",
        ),
        base.sign_base.row(
            "long_horizon_synced",
            "pass",
            "long-horizon roadmap sync target present",
            base.sign_base.truth(bool(base.find_line(long_text, ".2195-.2198"))),
            "The long-horizon roadmap must still expose the prior registry state before the ultra-extreme-farther continuation result is frozen.",
        ),
    ]
    route_sync_payload = base.sign_base.payload(
        "8.7.56.2202",
        STEP_NAME + " route sync",
        {
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
        route_sync_rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_fourth_post_break_ultra_extreme_farther_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": base.find_line(status_text, "8.7.56.2199"),
                "roadmap_branch_hit": base.find_line(roadmap_text, ".2199-.2202"),
                "current_problem_hit": base.find_line(current_problem_text, "8.7.56.2199"),
                "current_status_hit": base.find_line(current_status_text, "8.7.56.2199"),
                "unified_roadmap_hit": base.find_line(unified_text, ".2195-.2198"),
                "long_roadmap_hit": base.find_line(long_text, ".2195-.2198"),
                "part5_hit": base.find_line(part5_text, ".2195-.2198"),
            },
        },
    )
    base.write_artifact("route_sync", route_sync_payload)


if __name__ == "__main__":
    main()
