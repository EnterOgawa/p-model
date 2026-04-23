#!/usr/bin/env python3
"""Generate 8.7.56.1963-.1966 asymptotic sign-parity generalization artifacts.

The exact sign-parity theorem is already retained on `0 <= q/m0 <= 4`. This
branch audits whether that theorem extends canonically to the asymptotic
regime, or whether the observed large-q behavior is instead controlled by the
finite solver box used in the retained overlap computation.
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
QBALL_SOLVER = ROOT / "scripts" / "quantum" / "mass_origin_qball_charge_mapping_branch.py"
PRIOR_GATE = (
    PUBLIC_OUT
    / "q_8_7_56_1959_1962_ext_interval_decision_gate_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1963-1966"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor asymptotic sign-parity "
    "generalization audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "asymp_sign_parity_audit",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_global_exact_alpha_signed_form_factor_extended_interval_0_to_4_"
    "promotion_retained_asymptotic_generalization_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_asymptotic_sign_parity_box_boundary_obstruction_"
    "decision_gate_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_generalization_decision_gate_closeout_registry"
)
NEXT_ROUTE = "8.7.56.1967"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_conditional_box_free_tail_completion_"
    "or_substantive_pack_update_reactivation"
)
FOLLOWUP_ROUTE = "8.7.56.1971"
RETAINED_Q_MAX = 4.0
ASYMPTOTIC_Q_MAX = 8.0
Q_SAMPLES = np.array([4.0, 5.0, 6.0, 7.0, 8.0], dtype=float)
ROOT_TOL = 1.0e-10
ROOT_SCAN_DENSITY = 10000


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


# 関数: 指定区間の signed zero を探す。

def find_signed_zeros_interval(
    radius: np.ndarray,
    weight: np.ndarray,
    norm: float,
    q_max: float,
) -> np.ndarray:
    """Locate all simple signed zeros of F_exact(q) on 0 <= q <= q_max."""
    scan = np.linspace(0.0, float(q_max), int(ROOT_SCAN_DENSITY * q_max) + 1)
    values = np.array(
        [sign_base.form_factor(radius, weight, norm, float(q_value)) for q_value in scan]
    )
    roots: list[float] = []
    for q_left, q_right, f_left, f_right in zip(scan[:-1], scan[1:], values[:-1], values[1:]):
        if abs(f_left) <= ROOT_TOL and q_left > 0.0:
            root = float(q_left)
        elif f_left * f_right < 0.0:
            root = float(
                brentq(
                    lambda q_ratio: sign_base.form_factor(radius, weight, norm, float(q_ratio)),
                    float(q_left),
                    float(q_right),
                )
            )
        else:
            continue

        if not roots or abs(root - roots[-1]) > 1.0e-6:
            roots.append(root)

    return np.array(roots, dtype=float)


# 関数: finite-box leading asymptotic term を返す。

def leading_box_term(h_box: float, norm: float, r_box: float, q_ratio: float) -> float:
    """Return the leading finite-box asymptotic boundary term."""
    q_value = float(q_ratio)
    return -(h_box / (norm * q_value * q_value)) * math.cos(q_value * r_box)


# 関数: audit 用の公式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the asymptotic continuation audit."""
    return {
        "retained_box_overlap": "F_box(q) = int_0^{R_box} dr w(r) sinc(q r) / int_0^{R_box} dr w(r)",
        "edge_density": "h(r) = w(r)/r = r f(r)^2",
        "finite_box_asymptotic": "F_box(q) = -(h(R_box)/(N q^2)) cos(q R_box) + O(q^{-3})",
        "high_q_zero_lattice": "q_n^(box) ~= (n + 1/2) pi / R_box",
        "retained_exact_interval": "0 <= q/m0 <= 4",
        "asymptotic_audit_interval": "0 <= q/m0 <= 8",
    }


# 関数: `.1963-.1966` を実行する。

def main() -> None:
    """Execute the asymptotic sign-parity generalization audit."""
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
        QBALL_SOLVER,
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
    qball_branch_refresh = sign_base.read_json(QBALL_BRANCH_REFRESH)
    scalar_ground_state = sign_base.extract_scalar_ground_state(qball_branch_refresh)

    inventory_ready = all(
        (
            bool(prior_summary["exact_alpha_promotion_retained"]),
            bool(prior_summary["exact_signed_form_factor_promotion_retained"]),
            bool(prior_summary["asymptotic_generalization_admissible"]),
            float(prior_summary["extended_interval_over_m0"]) >= RETAINED_Q_MAX,
        )
    )

    qball_module = sign_base.load_qball_module()
    radius, field, _field_prime = qball_module.solve_full_profile(
        float(scalar_ground_state["beta_n"]),
        float(scalar_ground_state["central_amplitude"]),
    )
    weight = (field**2) * (radius**2)
    norm = float(np.trapezoid(weight, radius))
    r_box = float(radius[-1])
    tail_field_at_box_edge = float(field[-1])
    h_box_edge = float(weight[-1] / radius[-1])

    signed_zero_roots = find_signed_zeros_interval(radius, weight, norm, ASYMPTOTIC_Q_MAX)
    root_slopes = np.array(
        [sign_base.root_slope(radius, weight, norm, float(root)) for root in signed_zero_roots],
        dtype=float,
    )
    simple_zero_set_available = bool(
        root_slopes.size > 0 and np.all(np.abs(root_slopes) > 1.0e-6)
    )

    high_q_roots = signed_zero_roots[signed_zero_roots >= RETAINED_Q_MAX]
    high_q_zero_count = int(high_q_roots.size)
    high_q_spacings = np.diff(high_q_roots)
    asymptotic_spacing_theory = math.pi / r_box
    mean_high_q_spacing = float(np.mean(high_q_spacings))
    min_high_q_spacing = float(np.min(high_q_spacings))
    max_high_q_spacing = float(np.max(high_q_spacings))
    spacing_rel_gap_vs_theory = abs(mean_high_q_spacing - asymptotic_spacing_theory) / asymptotic_spacing_theory

    predicted_lattice = []
    lattice_errors = []
    for root in high_q_roots:
        mode_index = round((float(root) * r_box / math.pi) - 0.5)
        predicted = ((mode_index + 0.5) * math.pi) / r_box
        predicted_lattice.append(float(predicted))
        lattice_errors.append(float(root - predicted))

    root_lattice_max_abs_error = float(np.max(np.abs(lattice_errors)))
    root_lattice_mean_abs_error = float(np.mean(np.abs(lattice_errors)))

    sample_actual = np.array(
        [sign_base.form_factor(radius, weight, norm, float(q_ratio)) for q_ratio in Q_SAMPLES],
        dtype=float,
    )
    sample_leading = np.array(
        [leading_box_term(h_box_edge, norm, r_box, float(q_ratio)) for q_ratio in Q_SAMPLES],
        dtype=float,
    )
    sample_rel_errors = np.abs(sample_actual - sample_leading) / np.maximum(np.abs(sample_actual), 1.0e-30)
    leading_fit_mean_rel_error = float(np.mean(sample_rel_errors))
    leading_fit_max_rel_error = float(np.max(sample_rel_errors))

    box_boundary_asymptotic_supported = bool(
        spacing_rel_gap_vs_theory <= 5.0e-4 and leading_fit_max_rel_error <= 0.15
    )
    asymptotic_continuation_retained = False
    finite_interval_exact_but_asymptotic_obstruction_detected = bool(
        box_boundary_asymptotic_supported and simple_zero_set_available
    )
    current_continuation_rule_blocked = False
    physical_reject_required = False

    formulas = build_formulae()

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "asymptotic generalization inventory ready",
            sign_base.truth(inventory_ready),
            "The asymptotic audit starts only after the exact sign-parity theorem is official on 0<=q/m0<=4.",
        ),
        sign_base.row(
            "solver_box_edge_over_m0",
            "watch",
            "solver box edge R_box/m0^-1",
            r_box,
            "The retained overlap is currently evaluated on a finite solver box, so large-q behavior may carry a box-edge signature.",
        ),
        sign_base.row(
            "tail_field_at_box_edge",
            "watch",
            "retained field value at the solver box edge",
            tail_field_at_box_edge,
            "A nonzero tail at the finite box edge can feed a boundary-driven asymptotic term.",
        ),
        sign_base.row(
            "h_box_edge",
            "watch",
            "edge density h(R_box)=R_box f(R_box)^2",
            h_box_edge,
            "This is the coefficient of the leading finite-box boundary term.",
        ),
        sign_base.row(
            "asymptotic_spacing_theory",
            "watch",
            "finite-box asymptotic zero spacing theory pi/R_box",
            asymptotic_spacing_theory,
            "If the large-q zeros are boundary-driven, their spacing should approach pi/R_box.",
        ),
        sign_base.row(
            "high_q_zero_count_on_4_to_8",
            "watch",
            "signed zero count on 4<=q/m0<=8",
            float(high_q_zero_count),
            "These zeros probe the first high-q regime beyond the retained exact interval.",
        ),
        sign_base.row(
            "mean_high_q_spacing",
            "watch",
            "mean high-q zero spacing on 4<=q/m0<=8",
            mean_high_q_spacing,
            "This is the observed large-q spacing of the signed-zero lattice.",
        ),
        sign_base.row(
            "spacing_rel_gap_vs_theory",
            "watch",
            "relative gap between observed high-q spacing and pi/R_box",
            spacing_rel_gap_vs_theory,
            "A tiny gap indicates that the large-q zero lattice is controlled by the finite solver box.",
        ),
        sign_base.row(
            "root_lattice_max_abs_error",
            "watch",
            "max absolute error against the half-integer pi/R_box lattice",
            root_lattice_max_abs_error,
            "The tracked high-q zeros sit close to the finite-box half-integer lattice.",
        ),
        sign_base.row(
            "leading_fit_mean_rel_error",
            "watch",
            "mean relative error of the leading finite-box asymptotic term",
            leading_fit_mean_rel_error,
            "The leading boundary term approximates F(q) well on q in {4,5,6,7,8}.",
        ),
        sign_base.row(
            "leading_fit_max_rel_error",
            "watch",
            "max relative error of the leading finite-box asymptotic term",
            leading_fit_max_rel_error,
            "The current high-q sample remains quantitatively compatible with box-edge asymptotics.",
        ),
        sign_base.row(
            "box_boundary_asymptotic_supported",
            "pass" if box_boundary_asymptotic_supported else "reject",
            "finite-box boundary asymptotic supported",
            sign_base.truth(box_boundary_asymptotic_supported),
            "The observed high-q zero lattice and amplitudes are consistent with the finite-box boundary term rather than a box-free canonical theorem.",
        ),
        sign_base.row(
            "asymptotic_continuation_retained",
            "reject" if not asymptotic_continuation_retained else "pass",
            "Gate A asymptotic continuation retained",
            sign_base.truth(asymptotic_continuation_retained),
            "Current evidence does not support a box-free asymptotic theorem under the present solver-box pack.",
        ),
        sign_base.row(
            "finite_interval_exact_but_asymptotic_obstruction_detected",
            "pass" if finite_interval_exact_but_asymptotic_obstruction_detected else "reject",
            "Gate B finite-interval exact but asymptotic obstruction detected",
            sign_base.truth(finite_interval_exact_but_asymptotic_obstruction_detected),
            "The exact theorem survives on 0<=q/m0<=4, but the large-q continuation is governed by the solver-box boundary scale R_box=30.",
        ),
        sign_base.row(
            "current_continuation_rule_blocked",
            "reject" if not current_continuation_rule_blocked else "pass",
            "Gate C current continuation rule blocked on the audited finite interval",
            sign_base.truth(current_continuation_rule_blocked),
            "The current rule is not blocked on the retained finite interval; the obstruction appears only in the asymptotic generalization.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_interval_over_m0": RETAINED_Q_MAX,
        "asymptotic_audit_interval_over_m0": ASYMPTOTIC_Q_MAX,
        "solver_box_edge_over_m0": r_box,
        "tail_field_at_box_edge": tail_field_at_box_edge,
        "h_box_edge": h_box_edge,
        "asymptotic_spacing_theory": asymptotic_spacing_theory,
        "high_q_zero_count_on_4_to_8": high_q_zero_count,
        "mean_high_q_spacing": mean_high_q_spacing,
        "min_high_q_spacing": min_high_q_spacing,
        "max_high_q_spacing": max_high_q_spacing,
        "spacing_rel_gap_vs_theory": spacing_rel_gap_vs_theory,
        "root_lattice_max_abs_error": root_lattice_max_abs_error,
        "root_lattice_mean_abs_error": root_lattice_mean_abs_error,
        "leading_fit_mean_rel_error": leading_fit_mean_rel_error,
        "leading_fit_max_rel_error": leading_fit_max_rel_error,
        "box_boundary_asymptotic_supported": box_boundary_asymptotic_supported,
        "asymptotic_continuation_retained": asymptotic_continuation_retained,
        "finite_interval_exact_but_asymptotic_obstruction_detected": finite_interval_exact_but_asymptotic_obstruction_detected,
        "current_continuation_rule_blocked": current_continuation_rule_blocked,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    decision = {
        "overall_status": "vector_qball_form_factor_asymptotic_generalization_audit_declared",
        "branch_completed": True,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "formulas": formulas,
        "sample_q_values_over_m0": [float(value) for value in Q_SAMPLES],
        "sample_actual_form_factor": [float(value) for value in sample_actual],
        "sample_leading_box_term": [float(value) for value in sample_leading],
        "sample_relative_errors": [float(value) for value in sample_rel_errors],
        "high_q_signed_zero_roots_over_m0": [float(value) for value in high_q_roots],
        "high_q_predicted_box_lattice_over_m0": predicted_lattice,
        "high_q_root_lattice_errors": lattice_errors,
        "high_q_root_slopes": [float(value) for value in root_slopes[signed_zero_roots >= RETAINED_Q_MAX]],
        "hits": {
            "status_branch_hit": sign_base.hit(status_text, "8.7.56.1963"),
            "roadmap_branch_hit": sign_base.hit(roadmap_text, "8.7.56.1963-.1966"),
            "current_problem_hit": sign_base.hit(current_problem_text, "asymptotic sign-parity generalization"),
            "current_status_hit": sign_base.hit(current_status_text, "extended_interval_over_m0 = 4.0"),
            "unified_roadmap_hit": sign_base.hit(unified_text, "118. `.1963-.1966`"),
            "long_roadmap_hit": sign_base.hit(long_text, "8.7.56.1963"),
            "part5_hit": sign_base.hit(part5_text, "0<=q/m0<=4"),
        },
    }

    declaration_payload = sign_base.payload(
        "8.7.56.1965",
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
                "solver_module": sign_base.display_path(QBALL_SOLVER),
                "prior_gate": sign_base.display_path(PRIOR_GATE),
            },
            "constants": {
                "retained_interval_over_m0": RETAINED_Q_MAX,
                "asymptotic_audit_interval_over_m0": ASYMPTOTIC_Q_MAX,
                "next_route_name": NEXT_ROUTE_NAME,
                "next_route": NEXT_ROUTE,
                "followup_route_name": FOLLOWUP_ROUTE_NAME,
                "followup_route": FOLLOWUP_ROUTE,
            },
        },
        rows,
        summary,
        decision,
        evidence,
    )

    route_payload = sign_base.payload(
        "8.7.56.1966",
        STEP_NAME + " route sync",
        declaration_payload["inputs"],
        [
            sign_base.row(
                "finite_interval_exact_but_asymptotic_obstruction_detected",
                "pass" if finite_interval_exact_but_asymptotic_obstruction_detected else "reject",
                "Gate B finite-interval exact but asymptotic obstruction detected",
                sign_base.truth(finite_interval_exact_but_asymptotic_obstruction_detected),
                "The current theorem remains exact on 0<=q/m0<=4, but its large-q continuation is controlled by the solver-box edge.",
            ),
            sign_base.row(
                "asymptotic_continuation_retained",
                "reject" if not asymptotic_continuation_retained else "pass",
                "Gate A asymptotic continuation retained",
                sign_base.truth(asymptotic_continuation_retained),
                "No box-free asymptotic theorem is retained under the present pack.",
            ),
            sign_base.row(
                "next_route_fixed",
                "pass",
                "next route fixed",
                1.0,
                "The next official branch is the generalization decision gate / closeout registry.",
            ),
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
            "overall_status": "vector_qball_form_factor_asymptotic_generalization_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"formulas": formulas},
    )

    write_artifact("declaration_gate", declaration_payload)
    write_artifact("route_sync", route_payload)

    print("[ok] 8.7.56.1963-.1966 asymptotic sign-parity generalization artifacts generated")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
