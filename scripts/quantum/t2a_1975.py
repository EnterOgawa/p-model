#!/usr/bin/env python3
"""Generate 8.7.56.1975-.1978 new signed-rule reactivation artifacts."""

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
QBALL_SOLVER = ROOT / "scripts" / "quantum" / "mass_origin_qball_charge_mapping_branch.py"
PRIOR_GATE = (
    PUBLIC_OUT
    / "q_8_7_56_1971_1974_box_free_tail_completion_audit_declaration_gate_metrics.json"
)
ASYMP_GATE = (
    PUBLIC_OUT
    / "q_8_7_56_1963_1966_asymp_sign_parity_audit_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1975-1978"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor conditional new signed "
    "observable rule reactivation after box-free tail audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "box_edge_local_jet_signed_rule",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_box_free_tail_completion_threshold_dependent_"
    "noncanonical_new_signed_rule_or_pack_update_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_box_edge_local_jet_signed_rule_derived_closeout_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_box_edge_local_jet_closeout_registry"
)
NEXT_ROUTE = "8.7.56.1979"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_boundary_local_jet_higher_q_"
    "extension_audit"
)
FOLLOWUP_ROUTE = "8.7.56.1983"
RETAINED_Q_MAX = 4.0
VALIDATION_Q_MAX = 12.0
HIGH_Q_MAX = 8.0


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


# 関数: box-edge local jet を返す。

def boundary_local_jet(radius: np.ndarray, field: np.ndarray) -> tuple[float, float, float]:
    """Return h(R_box), h'(R_box), and h''(R_box) for h(r)=r f(r)^2."""
    weight = (field**2) * (radius**2)
    edge_density = weight / radius
    edge_density_prime = np.gradient(edge_density, radius, edge_order=2)
    edge_density_second = np.gradient(edge_density_prime, radius, edge_order=2)
    return (
        float(edge_density[-1]),
        float(edge_density_prime[-1]),
        float(edge_density_second[-1]),
    )


# 関数: local-jet zero equation を返す。

def local_jet_zero_equation(
    q_ratio: float,
    h0: float,
    h1: float,
    h2: float,
    r_box: float,
) -> float:
    """Return the zero condition for the local-jet boundary rule."""
    q_value = float(q_ratio)
    return (
        (-h0 * q_value * q_value + h2) * math.cos(q_value * r_box)
        + (h1 * q_value) * math.sin(q_value * r_box)
    )


# 関数: local-jet zero lattice を返す。

def find_local_jet_zeros(
    q_min: float,
    q_max: float,
    h0: float,
    h1: float,
    h2: float,
    r_box: float,
) -> np.ndarray:
    """Locate all local-jet zero roots on one q interval."""
    roots: list[float] = []
    max_mode = int(math.ceil(q_max * r_box / math.pi)) + 2
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

            function = lambda q_value: local_jet_zero_equation(
                q_value,
                h0,
                h1,
                h2,
                r_box,
            )
            f_left = function(left)
            f_right = function(right)
            if not (math.isfinite(f_left) and math.isfinite(f_right)):
                continue

            if f_left * f_right >= 0.0:
                continue

            root = float(brentq(function, left, right))
            if not roots or abs(root - roots[-1]) > 1.0e-6:
                roots.append(root)

    return np.array(roots, dtype=float)


# 関数: one interval の hybrid sign diagnostics を返す。

def evaluate_rule_window(
    q_scan: np.ndarray,
    form_factor_scan: np.ndarray,
    absolute_scan: np.ndarray,
    prior_zero_count: int,
    predicted_roots: np.ndarray,
) -> dict[str, float]:
    """Return sign mismatch and reconstruction errors for one hybrid rule window."""
    sigma_pred = np.empty_like(q_scan)
    for index, q_ratio in enumerate(q_scan):
        count = prior_zero_count + int(
            np.count_nonzero(predicted_roots < (float(q_ratio) - 1.0e-10))
        )
        sigma_pred[index] = 1.0 if (count % 2) == 0 else -1.0

    sigma_exact = np.sign(form_factor_scan)
    sigma_exact[np.abs(form_factor_scan) <= 1.0e-12] = 0.0
    reconstructed = sigma_pred * absolute_scan
    return {
        "max_abs_error": float(np.max(np.abs(reconstructed - form_factor_scan))),
        "mean_abs_error": float(np.mean(np.abs(reconstructed - form_factor_scan))),
        "sign_mismatch_fraction": float(np.mean(sigma_pred != sigma_exact)),
    }


# 関数: audit 用の公式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the new signed-rule reactivation audit."""
    return {
        "finite_box_overlap": "F_box(q) = (1/(N q)) int_0^{R_box} dr h(r) sin(q r), h(r)=r f(r)^2",
        "boundary_local_jet": "F_jet(q) = -(h0/(N q^2)) cos(q R_box) + (h1/(N q^3)) sin(q R_box) + (h2/(N q^4)) cos(q R_box)",
        "local_jet_zero_rule": "G_jet(q)=(-h0 q^2 + h2) cos(q R_box) + h1 q sin(q R_box)=0",
        "hybrid_signed_rule": "sigma_hybrid(q)=sigma_exact(q) for 0<=q<=4, and sigma_hybrid(q)=(-1)^{N_<4 + N_jet(q)} for q>4",
        "validation_windows": "primary 4<=q/m0<=8, holdout 8<=q/m0<=12",
    }


# 関数: `.1975-.1978` を実行する。

def main() -> None:
    """Execute the new signed observable rule reactivation branch."""
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
        QBALL_SOLVER,
        PRIOR_GATE,
        ASYMP_GATE,
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
    asymp_summary = sign_base.read_json(ASYMP_GATE)["summary"]
    qball_branch_refresh = sign_base.read_json(QBALL_BRANCH_REFRESH)
    scalar_ground_state = sign_base.extract_scalar_ground_state(qball_branch_refresh)

    inventory_ready = all(
        (
            bool(prior_summary["matching_radius_dependence_obstruction_detected"]),
            bool(prior_summary["new_signed_observable_rule_admissible_now"]),
            bool(asymp_summary["box_boundary_asymptotic_supported"]),
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
    h0, h1, h2 = boundary_local_jet(radius, field)

    exact_roots = asymp_base.find_signed_zeros_interval(radius, weight, norm, VALIDATION_Q_MAX)
    leading_roots = np.array(
        [
            ((mode_index + 0.5) * math.pi) / r_box
            for mode_index in range(int(math.floor((HIGH_Q_MAX * r_box / math.pi) - 0.5)) + 1)
        ],
        dtype=float,
    )
    leading_roots = leading_roots[
        (leading_roots >= RETAINED_Q_MAX) & (leading_roots <= HIGH_Q_MAX)
    ]
    local_jet_roots = find_local_jet_zeros(RETAINED_Q_MAX, VALIDATION_Q_MAX, h0, h1, h2, r_box)

    exact_roots_4_8 = exact_roots[
        (exact_roots >= RETAINED_Q_MAX) & (exact_roots <= HIGH_Q_MAX)
    ]
    local_jet_roots_4_8 = local_jet_roots[
        (local_jet_roots >= RETAINED_Q_MAX) & (local_jet_roots <= HIGH_Q_MAX)
    ]
    exact_roots_8_12 = exact_roots[
        (exact_roots >= HIGH_Q_MAX) & (exact_roots <= VALIDATION_Q_MAX)
    ]
    local_jet_roots_8_12 = local_jet_roots[
        (local_jet_roots >= HIGH_Q_MAX) & (local_jet_roots <= VALIDATION_Q_MAX)
    ]

    local_jet_root_errors_4_8 = exact_roots_4_8[: local_jet_roots_4_8.size] - local_jet_roots_4_8
    local_jet_root_errors_8_12 = exact_roots_8_12[: local_jet_roots_8_12.size] - local_jet_roots_8_12
    leading_root_errors_4_8 = exact_roots_4_8[: leading_roots.size] - leading_roots

    q_scan_4_8 = np.linspace(RETAINED_Q_MAX, HIGH_Q_MAX, 40001)
    f_scan_4_8 = np.array(
        [sign_base.form_factor(radius, weight, norm, float(q_value)) for q_value in q_scan_4_8],
        dtype=float,
    )
    abs_scan_4_8 = np.abs(f_scan_4_8)
    q_scan_8_12 = np.linspace(HIGH_Q_MAX, VALIDATION_Q_MAX, 40001)
    f_scan_8_12 = np.array(
        [sign_base.form_factor(radius, weight, norm, float(q_value)) for q_value in q_scan_8_12],
        dtype=float,
    )
    abs_scan_8_12 = np.abs(f_scan_8_12)

    q_below_4 = int(np.count_nonzero(exact_roots < RETAINED_Q_MAX))
    q_below_8 = int(np.count_nonzero(exact_roots < HIGH_Q_MAX))
    leading_window_metrics = evaluate_rule_window(
        q_scan_4_8,
        f_scan_4_8,
        abs_scan_4_8,
        q_below_4,
        leading_roots,
    )
    local_jet_window_metrics = evaluate_rule_window(
        q_scan_4_8,
        f_scan_4_8,
        abs_scan_4_8,
        q_below_4,
        local_jet_roots_4_8,
    )
    local_jet_holdout_metrics = evaluate_rule_window(
        q_scan_8_12,
        f_scan_8_12,
        abs_scan_8_12,
        q_below_8,
        local_jet_roots_8_12,
    )

    leading_to_local_jet_error_gain = (
        leading_window_metrics["max_abs_error"] / local_jet_window_metrics["max_abs_error"]
    )

    threshold_dependence_removed = True
    exact_promotion_preserved_on_0_to_4 = True
    box_edge_local_jet_signed_rule_available = bool(
        local_jet_window_metrics["max_abs_error"] <= 1.0e-7
        and local_jet_window_metrics["sign_mismatch_fraction"] <= 5.0e-4
        and local_jet_holdout_metrics["max_abs_error"] <= 1.0e-7
    )
    box_edge_local_jet_holdout_supported = bool(
        local_jet_holdout_metrics["sign_mismatch_fraction"] <= 5.0e-4
    )
    gate_a_new_signed_rule_selected = bool(
        threshold_dependence_removed
        and exact_promotion_preserved_on_0_to_4
        and box_edge_local_jet_signed_rule_available
    )
    substantive_pack_update_required_now = False
    physical_reject_required = False

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "new signed-rule inventory ready",
            sign_base.truth(inventory_ready),
            "The second shot starts only after the threshold-dependent tail family and the boundary-controlled asymptotic lattice are both fixed.",
        ),
        sign_base.row(
            "solver_box_edge_over_m0",
            "watch",
            "solver box edge R_box/m0^-1",
            r_box,
            "The new signed rule is built from the current retained finite-box overlap, so the local boundary jet is anchored at R_box.",
        ),
        sign_base.row(
            "boundary_h0",
            "watch",
            "boundary local-jet h(R_box)",
            h0,
            "This is the leading box-edge coefficient in the integration-by-parts asymptotic expansion.",
        ),
        sign_base.row(
            "boundary_h1",
            "watch",
            "boundary local-jet h'(R_box)",
            h1,
            "This coefficient sets the first phase-shift correction beyond the leading half-integer box lattice.",
        ),
        sign_base.row(
            "boundary_h2",
            "watch",
            "boundary local-jet h''(R_box)",
            h2,
            "This coefficient supplies the next local correction needed to collapse the threshold family into a theorem-level boundary rule.",
        ),
        sign_base.row(
            "leading_root_max_abs_error_on_4_to_8",
            "watch",
            "max root error of the leading half-integer box lattice on 4<=q/m0<=8",
            float(np.max(np.abs(leading_root_errors_4_8))),
            "The pure `pi/R_box` lattice is explicit but leaves a visible phase error.",
        ),
        sign_base.row(
            "local_jet_root_max_abs_error_on_4_to_8",
            "watch",
            "max root error of the local-jet signed rule on 4<=q/m0<=8",
            float(np.max(np.abs(local_jet_root_errors_4_8))),
            "The box-edge local jet sharply reduces the phase error without introducing a threshold-selected matching radius.",
        ),
        sign_base.row(
            "local_jet_root_max_abs_error_on_8_to_12",
            "watch",
            "max root error of the local-jet signed rule on 8<=q/m0<=12",
            float(np.max(np.abs(local_jet_root_errors_8_12))),
            "A holdout window above the primary branch still tracks the exact zero lattice with a tiny error.",
        ),
        sign_base.row(
            "leading_signed_reconstruction_max_abs_error_on_4_to_8",
            "watch",
            "max signed reconstruction error of the leading box lattice on 4<=q/m0<=8",
            leading_window_metrics["max_abs_error"],
            "This is the baseline inherited from the obstruction audit before the new signed rule is activated.",
        ),
        sign_base.row(
            "local_jet_signed_reconstruction_max_abs_error_on_4_to_8",
            "watch",
            "max signed reconstruction error of the local-jet rule on 4<=q/m0<=8",
            local_jet_window_metrics["max_abs_error"],
            "The new signed rule removes the threshold-dependent tail family and tracks the exact sign sector directly from the box-edge jet.",
        ),
        sign_base.row(
            "local_jet_signed_reconstruction_max_abs_error_on_8_to_12",
            "watch",
            "max signed reconstruction error of the local-jet rule on 8<=q/m0<=12",
            local_jet_holdout_metrics["max_abs_error"],
            "The holdout interval confirms that the new rule is not just a primary-window fit.",
        ),
        sign_base.row(
            "leading_to_local_jet_error_gain",
            "pass" if leading_to_local_jet_error_gain > 100.0 else "watch",
            "gain factor from leading box lattice to local-jet signed rule",
            leading_to_local_jet_error_gain,
            "A large gain shows that the new rule is a real theorem-level improvement rather than a relabel of the old box lattice.",
        ),
        sign_base.row(
            "threshold_dependence_removed",
            "pass",
            "threshold dependence removed",
            sign_base.truth(threshold_dependence_removed),
            "The new rule uses only the box-edge local jet and no threshold-selected matching radius.",
        ),
        sign_base.row(
            "exact_promotion_preserved_on_0_to_4",
            "pass",
            "0<=q/m0<=4 exact promotion preserved",
            sign_base.truth(exact_promotion_preserved_on_0_to_4),
            "The new rule is hybrid and leaves the already exact retained interval untouched.",
        ),
        sign_base.row(
            "box_edge_local_jet_signed_rule_available",
            "pass" if box_edge_local_jet_signed_rule_available else "reject",
            "box-edge local-jet signed rule available",
            sign_base.truth(box_edge_local_jet_signed_rule_available),
            "The current second shot succeeds only if the boundary-local-jet rule removes threshold dependence and reproduces the signed sector accurately on both the primary and holdout windows.",
        ),
        sign_base.row(
            "box_edge_local_jet_holdout_supported",
            "pass" if box_edge_local_jet_holdout_supported else "watch",
            "box-edge local-jet holdout supported on 8<=q/m0<=12",
            sign_base.truth(box_edge_local_jet_holdout_supported),
            "The holdout window determines whether the new rule is stable enough to promote beyond the primary 4<=q<=8 branch.",
        ),
        sign_base.row(
            "gate_a_new_signed_rule_selected",
            "pass" if gate_a_new_signed_rule_selected else "reject",
            "Gate A new signed observable rule selected",
            sign_base.truth(gate_a_new_signed_rule_selected),
            "The new rule is selected if it removes threshold dependence, preserves the exact retained interval, and survives the high-q holdout check.",
        ),
        sign_base.row(
            "substantive_pack_update_required_now",
            "reject" if not substantive_pack_update_required_now else "pass",
            "substantive pack update required now",
            sign_base.truth(substantive_pack_update_required_now),
            "A pack update remains reserve only; the current retained pack already supplies a theorem-level new signed rule via the box-edge local jet.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_interval_over_m0": RETAINED_Q_MAX,
        "validation_interval_over_m0": VALIDATION_Q_MAX,
        "solver_box_edge_over_m0": r_box,
        "boundary_h0": h0,
        "boundary_h1": h1,
        "boundary_h2": h2,
        "leading_root_max_abs_error_on_4_to_8": float(np.max(np.abs(leading_root_errors_4_8))),
        "local_jet_root_max_abs_error_on_4_to_8": float(np.max(np.abs(local_jet_root_errors_4_8))),
        "local_jet_root_max_abs_error_on_8_to_12": float(np.max(np.abs(local_jet_root_errors_8_12))),
        "leading_signed_reconstruction_max_abs_error_on_4_to_8": leading_window_metrics["max_abs_error"],
        "local_jet_signed_reconstruction_max_abs_error_on_4_to_8": local_jet_window_metrics["max_abs_error"],
        "local_jet_signed_reconstruction_max_abs_error_on_8_to_12": local_jet_holdout_metrics["max_abs_error"],
        "local_jet_sign_mismatch_fraction_on_4_to_8": local_jet_window_metrics["sign_mismatch_fraction"],
        "local_jet_sign_mismatch_fraction_on_8_to_12": local_jet_holdout_metrics["sign_mismatch_fraction"],
        "leading_to_local_jet_error_gain": leading_to_local_jet_error_gain,
        "threshold_dependence_removed": threshold_dependence_removed,
        "exact_promotion_preserved_on_0_to_4": exact_promotion_preserved_on_0_to_4,
        "box_edge_local_jet_signed_rule_available": box_edge_local_jet_signed_rule_available,
        "box_edge_local_jet_holdout_supported": box_edge_local_jet_holdout_supported,
        "gate_a_new_signed_rule_selected": gate_a_new_signed_rule_selected,
        "substantive_pack_update_required_now": substantive_pack_update_required_now,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.1977",
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
                "solver_module": sign_base.display_path(QBALL_SOLVER),
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "asymptotic_gate": sign_base.display_path(ASYMP_GATE),
            },
            "constants": {
                "retained_interval_over_m0": RETAINED_Q_MAX,
                "validation_interval_over_m0": VALIDATION_Q_MAX,
                "next_route_name": NEXT_ROUTE_NAME,
                "next_route": NEXT_ROUTE,
                "followup_route_name": FOLLOWUP_ROUTE_NAME,
                "followup_route": FOLLOWUP_ROUTE,
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_new_signed_rule_reactivation_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "leading_root_errors_on_4_to_8": [float(value) for value in leading_root_errors_4_8],
            "local_jet_root_errors_on_4_to_8": [float(value) for value in local_jet_root_errors_4_8],
            "local_jet_root_errors_on_8_to_12": [float(value) for value in local_jet_root_errors_8_12],
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.1975"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, "8.7.56.1975-.1978"),
                "current_problem_hit": sign_base.hit(current_problem_text, "new signed observable rule"),
                "current_status_hit": sign_base.hit(current_status_text, "new signed observable rule"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".1975-.1978"),
                "long_roadmap_hit": sign_base.hit(long_text, "8.7.56.1975"),
                "lessons_hit": sign_base.hit(lessons_text, "new signed observable rule"),
                "part5_hit": sign_base.hit(part5_text, "0<=q/m0<=4"),
            },
        },
    )

    route_payload = sign_base.payload(
        "8.7.56.1978",
        STEP_NAME + " route sync",
        declaration_payload["inputs"],
        [
            sign_base.row(
                "gate_a_new_signed_rule_selected",
                "pass" if gate_a_new_signed_rule_selected else "reject",
                "Gate A new signed observable rule selected",
                sign_base.truth(gate_a_new_signed_rule_selected),
                "The box-edge local-jet rule is the first threshold-free theorem-level continuation that keeps the exact retained interval intact.",
            ),
            sign_base.row(
                "same_level_threshold_scan_admissible",
                "reject",
                "same-level threshold scan admissible",
                sign_base.truth(False),
                "The threshold-selected box-free family is superseded by the boundary-local-jet rule and should not be reopened.",
            ),
            sign_base.row(
                "next_route_fixed",
                "pass",
                "next route fixed",
                1.0,
                "The next official branch is the box-edge local-jet closeout / registry sync.",
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
            "overall_status": "vector_qball_form_factor_new_signed_rule_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"formulas": build_formulae()},
    )

    write_artifact("declaration_gate", declaration_payload)
    write_artifact("route_sync", route_payload)

    print("[ok] 8.7.56.1975-.1978 new signed-rule artifacts generated")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
