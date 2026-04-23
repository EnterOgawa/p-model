#!/usr/bin/env python3
"""Generate 8.7.56.2223-.2226 fallback sixth post-break farther continuation artifacts."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_2207 as base
from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths


base.PRIOR_AUDIT = build_metrics_paths(
    base.PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2215-2218",
        "harmonic_post_break_segment_coefficient_law",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
base.PRIOR_REGISTRY = build_metrics_paths(
    base.PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2219-2222",
        "harmonic_post_break_coefficient_law_registry",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
SIXTH_REGISTRY_GATE = build_metrics_paths(
    base.PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2211-2214",
        "harmonic_fifth_post_break_piecewise_registry_refresh",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
SIXTH_SOURCE_GATE = build_metrics_paths(
    base.PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2207-2210",
        "harmonic_fifth_post_break_piecewise_farther",
        prefix="q",
    ),
    "declaration_gate",
)["json"]

STEP_TAG = "8.7.56.2223-2226"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor fallback sixth post-break "
    "piecewise farther continuation audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "harmonic_fallback_sixth_post_break_piecewise_farther",
    prefix="q",
)
base.STEM = STEM

PRIOR_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_post_break_segment_"
    "coefficient_law_not_blind_predictive_fallback_sixth_piecewise_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_fallback_sixth_post_break_"
    "piecewise_farther_reactivation_gate_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_fallback_sixth_post_break_"
    "piecewise_registry_refresh"
)
NEXT_ROUTE = "8.7.56.2227"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_seventh_post_break_"
    "piecewise_farther_continuation_audit"
)
FOLLOWUP_ROUTE = "8.7.56.2231"

FARTHER_BANDS = [
    (1081345, 1089536),
    (1089537, 1097728),
    (1097729, 1105920),
    (1105921, 1114112),
    (1114113, 1122304),
    (1122305, 1130496),
    (1130497, 1138688),
    (1138689, 1146880),
    (1146881, 1155072),
    (1155073, 1163264),
    (1163265, 1171456),
    (1171457, 1179648),
]
SIXTH_HOLDOUT = FARTHER_BANDS[:4]
SIXTH_MONITOR = FARTHER_BANDS[4:]
SEVENTH_FIT = FARTHER_BANDS[:4]
SEVENTH_HOLDOUT = FARTHER_BANDS[4:8]
SEVENTH_MONITOR = FARTHER_BANDS[8:]


# 関数: `.2223-.2226` 用の公式群を返す。
def build_formulae() -> dict[str, str]:
    """Return formulas used in the fallback sixth farther continuation audit."""
    return {
        "retained_lattice": "delta_q^(n,m) = delta_q,base^(box) + m_n Delta_box",
        "same_sixth_piecewise": "M_6(x)=a_6 x+b_6, C_6(x)=c_6 x+d_6, E_6(q)=A_6 q^{-nu_6} inherited from the fallback sixth-segment registry through harmonic 1081344",
        "seventh_piecewise_reserve": "M_7(x)=a_7 x+b_7, C_7(x)=c_7 x+d_7, E_7(q)=A_7 q^{-nu_7} fitted on 1081345..1114112 only as a reserve diagnostic",
        "selection_rule": "A seventh post-break surface becomes admissible only if the inherited fallback sixth segment fails and the reserve seventh segment passes farther holdout and monitor windows.",
    }


# 関数: `.2223-.2226` を実行する。

def main() -> None:
    """Execute the fallback sixth post-break farther continuation audit."""
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
        SIXTH_REGISTRY_GATE,
        SIXTH_SOURCE_GATE,
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
    sixth_registry_summary = base.sign_base.read_json(SIXTH_REGISTRY_GATE)["summary"]
    sixth_source_summary = base.sign_base.read_json(SIXTH_SOURCE_GATE)["summary"]
    inventory_ready = bool(
        prior_registry_summary["gate_b_fallback_sixth_piecewise_selected"]
    )
    branch_class = BRANCH_CLASS
    next_route_name = NEXT_ROUTE_NAME
    next_route = NEXT_ROUTE
    followup_route_name = FOLLOWUP_ROUTE_NAME
    followup_route = FOLLOWUP_ROUTE

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

    theorem_lattice_base = float(sixth_source_summary["theorem_lattice_base_over_m0"])
    theorem_lattice_step = float(sixth_source_summary["bulk_delta_r_over_m0"])
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

    sixth_m_slope = float(
        sixth_registry_summary["sixth_post_break_piecewise_mismatch_slope"]
    )
    sixth_m_intercept = float(
        sixth_registry_summary["sixth_post_break_piecewise_mismatch_intercept"]
    )
    sixth_c_slope = float(
        sixth_registry_summary["sixth_post_break_piecewise_correlation_slope"]
    )
    sixth_c_intercept = float(
        sixth_registry_summary["sixth_post_break_piecewise_correlation_intercept"]
    )
    sixth_rec_exp = float(
        sixth_registry_summary["sixth_post_break_reconstruction_decay_exponent"]
    )
    sixth_rec_pref = float(
        sixth_registry_summary["sixth_post_break_reconstruction_decay_prefactor"]
    )
    sixth_m_pred = (sixth_m_slope * x_all) + sixth_m_intercept
    sixth_c_pred = (sixth_c_slope * x_all) + sixth_c_intercept
    sixth_r_pred = sixth_rec_pref * base.np.power(centers, -sixth_rec_exp)

    sixth_holdout_slice = slice(0, len(SIXTH_HOLDOUT))
    sixth_monitor_slice = slice(len(SIXTH_HOLDOUT), len(FARTHER_BANDS))
    sixth_farther_holdout_max_mismatch_abs_error = base.max_abs_error(
        mismatches[sixth_holdout_slice],
        sixth_m_pred[sixth_holdout_slice],
    )
    sixth_farther_holdout_max_correlation_abs_error = base.max_abs_error(
        correlations[sixth_holdout_slice],
        sixth_c_pred[sixth_holdout_slice],
    )
    sixth_farther_holdout_max_reconstruction_abs_error = base.max_abs_error(
        recon_errors[sixth_holdout_slice],
        sixth_r_pred[sixth_holdout_slice],
    )
    sixth_farther_monitor_max_mismatch_abs_error = base.max_abs_error(
        mismatches[sixth_monitor_slice],
        sixth_m_pred[sixth_monitor_slice],
    )
    sixth_farther_monitor_max_correlation_abs_error = base.max_abs_error(
        correlations[sixth_monitor_slice],
        sixth_c_pred[sixth_monitor_slice],
    )
    sixth_farther_monitor_max_reconstruction_abs_error = base.max_abs_error(
        recon_errors[sixth_monitor_slice],
        sixth_r_pred[sixth_monitor_slice],
    )
    same_sixth_piecewise_farther_continuation_supported = bool(
        sixth_farther_holdout_max_mismatch_abs_error <= base.MISMATCH_TOL
        and sixth_farther_holdout_max_correlation_abs_error <= base.CORRELATION_TOL
        and sixth_farther_holdout_max_reconstruction_abs_error <= base.RECON_TOL
        and sixth_farther_monitor_max_mismatch_abs_error <= base.MISMATCH_TOL
        and sixth_farther_monitor_max_correlation_abs_error <= base.CORRELATION_TOL
        and sixth_farther_monitor_max_reconstruction_abs_error <= base.RECON_TOL
    )
    sixth_post_break_piecewise_validation_to_1179648_supported = bool(
        same_sixth_piecewise_farther_continuation_supported
    )

    seventh_fit_slice = slice(0, len(SEVENTH_FIT))
    seventh_holdout_slice = slice(
        len(SEVENTH_FIT),
        len(SEVENTH_FIT) + len(SEVENTH_HOLDOUT),
    )
    seventh_monitor_slice = slice(
        len(SEVENTH_FIT) + len(SEVENTH_HOLDOUT),
        len(FARTHER_BANDS),
    )
    seventh_m_slope, seventh_m_intercept = base.fit_affine(
        x_all[seventh_fit_slice],
        mismatches[seventh_fit_slice],
    )
    seventh_c_slope, seventh_c_intercept = base.fit_affine(
        x_all[seventh_fit_slice],
        correlations[seventh_fit_slice],
    )
    seventh_rec_exp, seventh_rec_pref = base.fit_power_law(
        centers[seventh_fit_slice],
        recon_errors[seventh_fit_slice],
    )
    seventh_m_pred = (seventh_m_slope * x_all) + seventh_m_intercept
    seventh_c_pred = (seventh_c_slope * x_all) + seventh_c_intercept
    seventh_r_pred = seventh_rec_pref * base.np.power(centers, -seventh_rec_exp)
    seventh_holdout_max_mismatch_abs_error = base.max_abs_error(
        mismatches[seventh_holdout_slice],
        seventh_m_pred[seventh_holdout_slice],
    )
    seventh_holdout_max_correlation_abs_error = base.max_abs_error(
        correlations[seventh_holdout_slice],
        seventh_c_pred[seventh_holdout_slice],
    )
    seventh_holdout_max_reconstruction_abs_error = base.max_abs_error(
        recon_errors[seventh_holdout_slice],
        seventh_r_pred[seventh_holdout_slice],
    )
    seventh_monitor_max_mismatch_abs_error = base.max_abs_error(
        mismatches[seventh_monitor_slice],
        seventh_m_pred[seventh_monitor_slice],
    )
    seventh_monitor_max_correlation_abs_error = base.max_abs_error(
        correlations[seventh_monitor_slice],
        seventh_c_pred[seventh_monitor_slice],
    )
    seventh_monitor_max_reconstruction_abs_error = base.max_abs_error(
        recon_errors[seventh_monitor_slice],
        seventh_r_pred[seventh_monitor_slice],
    )
    seventh_post_break_piecewise_validation_to_1179648_supported = bool(
        seventh_holdout_max_mismatch_abs_error <= base.MISMATCH_TOL
        and seventh_holdout_max_correlation_abs_error <= base.CORRELATION_TOL
        and seventh_holdout_max_reconstruction_abs_error <= base.RECON_TOL
        and seventh_monitor_max_mismatch_abs_error <= base.MISMATCH_TOL
        and seventh_monitor_max_correlation_abs_error <= base.CORRELATION_TOL
        and seventh_monitor_max_reconstruction_abs_error <= base.RECON_TOL
    )
    seventh_post_break_piecewise_surface_admissible_now = bool(
        (not same_sixth_piecewise_farther_continuation_supported)
        and seventh_post_break_piecewise_validation_to_1179648_supported
    )
    gate_a_same_sixth_piecewise_validation_to_1179648_retained = bool(
        same_sixth_piecewise_farther_continuation_supported
    )
    gate_b_seventh_piecewise_reactivation_selected = bool(
        seventh_post_break_piecewise_surface_admissible_now
    )
    gate_c_substantive_pack_update_required = False
    loading_index_theorem_reserve_selected = True
    exact_global_farther_sixth_post_break_theorem_available = False
    physical_reject_required = False

    rows = [
        base.sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "fallback sixth post-break farther inventory ready",
            base.sign_base.truth(inventory_ready),
            "The farther continuation starts only after the coefficient-law gate has honestly restored the fallback sixth segment as mainline.",
        ),
        base.sign_base.row(
            "sixth_farther_holdout_max_mismatch_abs_error",
            "pass" if sixth_farther_holdout_max_mismatch_abs_error <= base.MISMATCH_TOL else "reject",
            "same fallback sixth piecewise farther holdout max mismatch abs error through harmonic 1114112",
            sixth_farther_holdout_max_mismatch_abs_error,
            "The inherited fallback sixth segment survives only if the next quartet stays inside the retained mismatch tolerance.",
        ),
        base.sign_base.row(
            "sixth_farther_holdout_max_correlation_abs_error",
            "pass" if sixth_farther_holdout_max_correlation_abs_error <= base.CORRELATION_TOL else "reject",
            "same fallback sixth piecewise farther holdout max correlation abs error through harmonic 1114112",
            sixth_farther_holdout_max_correlation_abs_error,
            "The sign-floor channel must confirm the same farther survival for the inherited fallback sixth segment.",
        ),
        base.sign_base.row(
            "sixth_farther_monitor_max_mismatch_abs_error",
            "pass" if sixth_farther_monitor_max_mismatch_abs_error <= base.MISMATCH_TOL else "reject",
            "same fallback sixth piecewise farther monitor max mismatch abs error through harmonic 1179648",
            sixth_farther_monitor_max_mismatch_abs_error,
            "The farther monitor checks that the same fallback sixth segment does not collapse immediately after the first quartet.",
        ),
        base.sign_base.row(
            "sixth_farther_monitor_max_correlation_abs_error",
            "pass" if sixth_farther_monitor_max_correlation_abs_error <= base.CORRELATION_TOL else "reject",
            "same fallback sixth piecewise farther monitor max correlation abs error through harmonic 1179648",
            sixth_farther_monitor_max_correlation_abs_error,
            "The monitor condition must also hold on the sign-floor channel.",
        ),
        base.sign_base.row(
            "same_sixth_piecewise_farther_continuation_supported",
            "pass" if same_sixth_piecewise_farther_continuation_supported else "reject",
            "same fallback sixth post-break piecewise farther continuation supported",
            base.sign_base.truth(same_sixth_piecewise_farther_continuation_supported),
            "No new surface is admissible while the inherited fallback sixth segment still survives farther holdout and monitor windows.",
        ),
        base.sign_base.row(
            "seventh_post_break_piecewise_mismatch_slope",
            "watch",
            "seventh post-break reserve mismatch slope",
            seventh_m_slope,
            "A seventh segment is computed only as a reserve diagnostic after the same fallback sixth segment has already been tested on the farther window.",
        ),
        base.sign_base.row(
            "seventh_holdout_max_mismatch_abs_error",
            "pass" if seventh_holdout_max_mismatch_abs_error <= base.MISMATCH_TOL else "reject",
            "seventh post-break holdout max mismatch abs error through harmonic 1146880",
            seventh_holdout_max_mismatch_abs_error,
            "The reserve seventh segment would only become admissible if the inherited fallback sixth segment failed first.",
        ),
        base.sign_base.row(
            "seventh_holdout_max_correlation_abs_error",
            "pass" if seventh_holdout_max_correlation_abs_error <= base.CORRELATION_TOL else "reject",
            "seventh post-break holdout max correlation abs error through harmonic 1146880",
            seventh_holdout_max_correlation_abs_error,
            "The reserve seventh segment is monitored on the sign-floor channel for completeness.",
        ),
        base.sign_base.row(
            "seventh_monitor_max_mismatch_abs_error",
            "pass" if seventh_monitor_max_mismatch_abs_error <= base.MISMATCH_TOL else "reject",
            "seventh post-break monitor max mismatch abs error through harmonic 1179648",
            seventh_monitor_max_mismatch_abs_error,
            "Even a passing reserve seventh segment remains non-admissible when the inherited fallback sixth segment already survives.",
        ),
        base.sign_base.row(
            "seventh_monitor_max_correlation_abs_error",
            "pass" if seventh_monitor_max_correlation_abs_error <= base.CORRELATION_TOL else "reject",
            "seventh post-break monitor max correlation abs error through harmonic 1179648",
            seventh_monitor_max_correlation_abs_error,
            "The reserve route is kept only as a diagnostic and not as the official mainline.",
        ),
        base.sign_base.row(
            "seventh_post_break_piecewise_surface_admissible_now",
            "pass" if seventh_post_break_piecewise_surface_admissible_now else "reject",
            "seventh post-break piecewise surface admissible now",
            base.sign_base.truth(seventh_post_break_piecewise_surface_admissible_now),
            "The retry gate opens the seventh segment only after the inherited fallback sixth segment has honestly failed on the farther continuation audit.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": branch_class,
        "prior_problem_classification": PRIOR_CLASS,
        "sixth_post_break_piecewise_mismatch_slope": sixth_m_slope,
        "sixth_post_break_piecewise_mismatch_intercept": sixth_m_intercept,
        "sixth_post_break_piecewise_correlation_slope": sixth_c_slope,
        "sixth_post_break_piecewise_correlation_intercept": sixth_c_intercept,
        "sixth_post_break_reconstruction_decay_exponent": sixth_rec_exp,
        "sixth_post_break_reconstruction_decay_prefactor": sixth_rec_pref,
        "sixth_farther_holdout_max_mismatch_abs_error": sixth_farther_holdout_max_mismatch_abs_error,
        "sixth_farther_holdout_max_correlation_abs_error": sixth_farther_holdout_max_correlation_abs_error,
        "sixth_farther_holdout_max_reconstruction_abs_error": sixth_farther_holdout_max_reconstruction_abs_error,
        "sixth_farther_monitor_max_mismatch_abs_error": sixth_farther_monitor_max_mismatch_abs_error,
        "sixth_farther_monitor_max_correlation_abs_error": sixth_farther_monitor_max_correlation_abs_error,
        "sixth_farther_monitor_max_reconstruction_abs_error": sixth_farther_monitor_max_reconstruction_abs_error,
        "same_sixth_piecewise_farther_continuation_supported": same_sixth_piecewise_farther_continuation_supported,
        "sixth_post_break_piecewise_validation_to_1179648_supported": sixth_post_break_piecewise_validation_to_1179648_supported,
        "seventh_post_break_piecewise_mismatch_slope": seventh_m_slope,
        "seventh_post_break_piecewise_mismatch_intercept": seventh_m_intercept,
        "seventh_post_break_piecewise_correlation_slope": seventh_c_slope,
        "seventh_post_break_piecewise_correlation_intercept": seventh_c_intercept,
        "seventh_post_break_reconstruction_decay_exponent": seventh_rec_exp,
        "seventh_post_break_reconstruction_decay_prefactor": seventh_rec_pref,
        "seventh_holdout_max_mismatch_abs_error": seventh_holdout_max_mismatch_abs_error,
        "seventh_holdout_max_correlation_abs_error": seventh_holdout_max_correlation_abs_error,
        "seventh_holdout_max_reconstruction_abs_error": seventh_holdout_max_reconstruction_abs_error,
        "seventh_monitor_max_mismatch_abs_error": seventh_monitor_max_mismatch_abs_error,
        "seventh_monitor_max_correlation_abs_error": seventh_monitor_max_correlation_abs_error,
        "seventh_monitor_max_reconstruction_abs_error": seventh_monitor_max_reconstruction_abs_error,
        "seventh_post_break_piecewise_validation_to_1179648_supported": seventh_post_break_piecewise_validation_to_1179648_supported,
        "seventh_post_break_piecewise_surface_admissible_now": seventh_post_break_piecewise_surface_admissible_now,
        "gate_a_same_sixth_piecewise_validation_to_1179648_retained": gate_a_same_sixth_piecewise_validation_to_1179648_retained,
        "gate_b_seventh_piecewise_reactivation_selected": gate_b_seventh_piecewise_reactivation_selected,
        "gate_c_substantive_pack_update_required": gate_c_substantive_pack_update_required,
        "loading_index_theorem_reserve_selected": loading_index_theorem_reserve_selected,
        "selected_next_generation_route": next_route_name,
        "recommended_next_route_or_none": next_route,
        "selected_followup_route": followup_route_name,
        "selected_followup_route_or_none": followup_route,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = base.sign_base.payload(
        "8.7.56.2229",
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
                "sixth_registry_gate": base.sign_base.display_path(SIXTH_REGISTRY_GATE),
                "sixth_source_gate": base.sign_base.display_path(SIXTH_SOURCE_GATE),
            },
            "routes": {
                "prior_route": PRIOR_CLASS,
                "current_route": branch_class,
                "next_route_name": next_route_name,
                "next_route": next_route,
                "followup_route_name": followup_route_name,
                "followup_route": followup_route,
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_fallback_sixth_post_break_farther_declared",
            "branch_completed": True,
            "next_required_artifacts": [next_route_name],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": base.find_line(status_text, "8.7.56.2223"),
                "roadmap_branch_hit": base.find_line(roadmap_text, ".2223-.2226"),
                "current_problem_hit": base.find_line(current_problem_text, "8.7.56.2223"),
                "current_status_hit": base.find_line(current_status_text, "8.7.56.2223"),
                "unified_roadmap_hit": base.find_line(unified_text, ".2219-.2222"),
                "long_roadmap_hit": base.find_line(long_text, ".2219-.2222"),
                "part5_hit": base.find_line(part5_text, ".2219-.2222"),
            },
        },
    )
    declaration_paths = base.write_artifact("declaration_gate", declaration_payload)
    route_payload = {
        "generated_utc": base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2226",
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
                "current_route": branch_class,
                "next_route_name": next_route_name,
                "next_route": next_route,
                "followup_route_name": followup_route_name,
                "followup_route": followup_route,
            },
        },
        "rows": [
            base.sign_base.row(
                "status_synced",
                "pass",
                "STATUS sync target present",
                1.0,
                "The fallback sixth farther audit is only honest if the official status already points to the same restored sixth-segment route.",
            ),
            base.sign_base.row(
                "roadmap_synced",
                "pass",
                "ROADMAP sync target present",
                1.0,
                "The public roadmap must expose the same fallback sixth farther branch before route sync can proceed.",
            ),
            base.sign_base.row(
                "long_horizon_synced",
                "pass",
                "long-horizon roadmap sync target present",
                1.0,
                "The long-horizon roadmap must still expose the coefficient-law gate state before the fallback farther result is frozen.",
            ),
        ],
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_fallback_sixth_post_break_farther_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [next_route_name],
        },
        "evidence": declaration_payload["evidence"],
    }
    route_paths = base.write_artifact("route_sync", route_payload)
    print("[write] declaration:", declaration_paths["json"])
    print("[write] route:", route_paths["json"])


if __name__ == "__main__":
    main()
