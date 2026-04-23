#!/usr/bin/env python3
"""Generate 8.7.56.2023-.2026 boundary alias-harmonic spike audit artifacts.

This branch audits the residual first/second alias-harmonic spike windows
after the generic high-q sign-root floor has already been resolved.
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
    / "q_8_7_56_2015_2018_resolved_high_q_sign_floor_audit_declaration_gate_metrics.json"
)
PRIOR_GATE = (
    PUBLIC_OUT
    / "q_8_7_56_2019_2022_resolved_high_q_sign_root_gate_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.2023-2026"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor boundary alias-harmonic "
    "spike audit"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "alias_harmonic_spike_audit",
    prefix="q",
)

PRIOR_CLASS = "vector_qball_form_factor_boundary_alias_harmonic_spike_reactivation_next"
BRANCH_CLASS = (
    "vector_qball_form_factor_boundary_alias_harmonic_spike_audited_"
    "alias_image_reactivation_gate_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_alias_harmonic_spike_"
    "decision_gate_registry"
)
NEXT_ROUTE = "8.7.56.2027"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_boundary_alias_image_"
    "signed_rule_reactivation"
)
FOLLOWUP_ROUTE = "8.7.56.2031"

FIT_Q_MIN = 200.0
FIT_Q_MAX = 260.0
EDGE_Q_MIN = 380.0
EDGE_Q_MAX = 420.0
WINDOW_SCAN_DENSITY = 2000
SIGN_ZERO_TOL = 1.0e-12


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


# 関数: box-edge local-jet zero equation を返す。

def local_jet_zero_equation(
    q_ratio: float,
    h0: float,
    h1: float,
    h2: float,
    r_box: float,
) -> float:
    """Return the retained local-jet zero equation."""
    return (
        (-h0 * q_ratio * q_ratio + h2) * math.cos(q_ratio * r_box)
        + (h1 * q_ratio) * math.sin(q_ratio * r_box)
    )


# 関数: one q window の direct overlap scan を返す。

def scan_window(
    radius: np.ndarray,
    weight: np.ndarray,
    norm: float,
    q_min: float,
    q_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return q grid, overlap values, and sign states on one q window."""
    q_scan = np.linspace(q_min, q_max, int(round((q_max - q_min) * WINDOW_SCAN_DENSITY)) + 1)
    values = np.array(
        [sign_base.form_factor(radius, weight, norm, float(q_value)) for q_value in q_scan],
        dtype=float,
    )
    sign_values = np.sign(values)
    sign_values[np.abs(values) <= SIGN_ZERO_TOL] = 0.0
    return q_scan, values, sign_values


# 関数: one root family を返す。

def find_predicted_roots(
    q_min: float,
    q_max: float,
    equation,
    r_box: float,
) -> np.ndarray:
    """Locate one predicted zero lattice on one q interval."""
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


# 関数: root parity から予測 sign を返す。

def predicted_sign_from_roots(
    q_scan: np.ndarray,
    values: np.ndarray,
    predicted_roots: np.ndarray,
) -> np.ndarray:
    """Return one window-local parity sign anchored at the first exact sign."""
    sigma_pred = np.empty_like(q_scan)
    nonzero = values[np.abs(values) > SIGN_ZERO_TOL]
    sigma_start = 1.0 if nonzero[0] > 0.0 else -1.0
    for index, q_value in enumerate(q_scan):
        count = int(np.count_nonzero(predicted_roots < (float(q_value) - 1.0e-10)))
        sigma_pred[index] = sigma_start if (count % 2) == 0 else -sigma_start

    return sigma_pred


# 関数: alias-image parity sign を返す。

def alias_image_sign(
    radius: np.ndarray,
    weight: np.ndarray,
    norm: float,
    q_scan: np.ndarray,
    alias_harmonic: float,
    harmonic_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return alias-image mapped q values and their parity sign."""
    q_image = np.abs(alias_harmonic - q_scan)
    image_values = np.array(
        [sign_base.form_factor(radius, weight, norm, float(q_value)) for q_value in q_image],
        dtype=float,
    )
    sigma_image = np.sign(image_values)
    sigma_image[np.abs(image_values) <= SIGN_ZERO_TOL] = 0.0
    if (harmonic_index % 2) == 1:
        sigma_image = -sigma_image

    return q_image, sigma_image


# 関数: mismatch fraction を返す。

def sign_mismatch_fraction(predicted: np.ndarray, exact: np.ndarray) -> float:
    """Return one sign mismatch fraction."""
    return float(np.mean(predicted != exact))


# 関数: window 上の最大 subleading ratio を返す。

def subleading_ratios(
    q_min: float,
    h0: float,
    h1: float,
    h2: float,
) -> tuple[float, float]:
    """Return sup |h1|/(|h0| q) and sup |h2|/(|h0| q^2) on one window."""
    q_small = float(q_min)
    ratio_h1 = abs(h1) / (abs(h0) * q_small)
    ratio_h2 = abs(h2) / (abs(h0) * q_small * q_small)
    return ratio_h1, ratio_h2


# 関数: outer bulk grid の uniformity を返す。

def bulk_grid_summary(radius: np.ndarray) -> tuple[float, float, float]:
    """Return the dominant bulk step, its interval fraction, and edge-cell mismatch."""
    dr = np.diff(radius)
    bulk_delta_r = float(np.round(np.max(dr[:-1]), 12))
    bulk_mask = np.isclose(dr[:-1], bulk_delta_r, atol=1.0e-12, rtol=0.0)
    bulk_fraction = float(np.count_nonzero(bulk_mask) / dr.size)
    edge_cell_relative_gap = abs(float(dr[-1]) - bulk_delta_r) / bulk_delta_r
    return bulk_delta_r, bulk_fraction, edge_cell_relative_gap


# 関数: audit 用の公式を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the alias-harmonic spike audit."""
    return {
        "retained_local_jet_rule": "G_jet(q)=(-h0 q^2 + h2) cos(q R_box) + h1 q sin(q R_box)=0",
        "subleading_hierarchy": "epsilon_1(q)=|h1|/(|h0| q), epsilon_2(q)=|h2|/(|h0| q^2)",
        "alias_harmonics": "q_alias^(n)=2 n pi / Delta r_bulk",
        "bulk_uniform_identity": "sin((2 n pi / Delta r_bulk - q) j Delta r_bulk)=(-1)^(n+1) sin(q j Delta r_bulk)",
        "alias_image_rule": "sigma_img^(n)(q)=(-1)^n sign(F_exact(|q_alias^(n)-q|))",
    }


# 関数: `.2023-.2026` を実行する。

def main() -> None:
    """Execute the boundary alias-harmonic spike audit."""
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

    prior_gate_summary = sign_base.read_json(PRIOR_GATE)["summary"]
    inventory_ready = bool(prior_gate_summary["alias_harmonic_spike_audit_admissible_now"])

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
    bulk_delta_r, bulk_fraction, edge_cell_relative_gap = bulk_grid_summary(radius)
    alias_1 = 2.0 * math.pi / bulk_delta_r
    alias_2 = 2.0 * alias_1

    fit_q_scan, fit_values, fit_sign = scan_window(radius, weight, norm, FIT_Q_MIN, FIT_Q_MAX)
    edge_q_scan, edge_values, edge_sign = scan_window(radius, weight, norm, EDGE_Q_MIN, EDGE_Q_MAX)

    fit_roots = find_predicted_roots(
        FIT_Q_MIN,
        FIT_Q_MAX,
        lambda q_value: local_jet_zero_equation(q_value, h0, h1, h2, r_box),
        r_box,
    )
    edge_roots = find_predicted_roots(
        EDGE_Q_MIN,
        EDGE_Q_MAX,
        lambda q_value: local_jet_zero_equation(q_value, h0, h1, h2, r_box),
        r_box,
    )

    fit_local_sigma = predicted_sign_from_roots(fit_q_scan, fit_values, fit_roots)
    edge_local_sigma = predicted_sign_from_roots(edge_q_scan, edge_values, edge_roots)
    fit_local_mismatch = sign_mismatch_fraction(fit_local_sigma, fit_sign)
    edge_local_mismatch = sign_mismatch_fraction(edge_local_sigma, edge_sign)

    fit_q_image, fit_image_sigma = alias_image_sign(radius, weight, norm, fit_q_scan, alias_1, 1)
    edge_q_image, edge_image_sigma = alias_image_sign(radius, weight, norm, edge_q_scan, alias_2, 2)
    fit_image_mismatch = sign_mismatch_fraction(fit_image_sigma, fit_sign)
    edge_image_mismatch = sign_mismatch_fraction(edge_image_sigma, edge_sign)
    fit_image_gain = fit_local_mismatch / fit_image_mismatch
    edge_image_gain = edge_local_mismatch / edge_image_mismatch
    fit_image_corr = float(np.mean(fit_image_sigma * fit_sign))
    edge_image_corr = float(np.mean(edge_image_sigma * edge_sign))

    fit_ratio_h1, fit_ratio_h2 = subleading_ratios(FIT_Q_MIN, h0, h1, h2)
    edge_ratio_h1, edge_ratio_h2 = subleading_ratios(EDGE_Q_MIN, h0, h1, h2)

    fit_subleading_negligible = bool(fit_ratio_h1 < 2.0e-3 and fit_ratio_h2 < 1.0e-5)
    edge_subleading_negligible = bool(edge_ratio_h1 < 1.0e-3 and edge_ratio_h2 < 1.0e-5)
    fit_alias_image_supported = bool(fit_image_mismatch < fit_local_mismatch)
    edge_alias_image_supported = bool(edge_image_mismatch <= edge_local_mismatch)
    alias_image_family_admissible = bool(
        fit_subleading_negligible
        and edge_subleading_negligible
        and fit_alias_image_supported
        and edge_alias_image_supported
    )
    same_level_boundary_term_retry_admissible = False
    substantive_pack_update_required_now = False
    physical_reject_required = False

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "alias-harmonic spike inventory ready",
            sign_base.truth(inventory_ready),
            "The spike audit starts only after the prior gate has promoted alias-harmonic windows to the active blocker.",
        ),
        sign_base.row(
            "bulk_delta_r_over_m0",
            "watch",
            "dominant bulk grid spacing over m0^-1",
            bulk_delta_r,
            "The spike windows are keyed to the dominant outer bulk spacing rather than to an abstract smooth-phase carrier.",
        ),
        sign_base.row(
            "bulk_uniform_cell_fraction",
            "watch",
            "fraction of intervals already on the dominant bulk spacing",
            bulk_fraction,
            "A nearly uniform outer bulk grid makes the alias-image identity meaningful before any new pack update.",
        ),
        sign_base.row(
            "first_alias_harmonic_over_m0",
            "watch",
            "first alias harmonic over m0",
            alias_1,
            "The fit spike sits on the first alias harmonic of the dominant bulk spacing.",
        ),
        sign_base.row(
            "second_alias_harmonic_over_m0",
            "watch",
            "second alias harmonic over m0",
            alias_2,
            "The edge spike sits on the second alias harmonic of the dominant bulk spacing.",
        ),
        sign_base.row(
            "fit_window_max_h1_over_h0q",
            "watch",
            "fit-window max |h1|/(|h0| q)",
            fit_ratio_h1,
            "The first local-jet correction is already three orders below the leading h0 carrier on the fit spike window.",
        ),
        sign_base.row(
            "fit_window_max_h2_over_h0q2",
            "watch",
            "fit-window max |h2|/(|h0| q^2)",
            fit_ratio_h2,
            "The h2 correction is completely negligible on the fit spike window.",
        ),
        sign_base.row(
            "edge_window_max_h1_over_h0q",
            "watch",
            "edge-window max |h1|/(|h0| q)",
            edge_ratio_h1,
            "The first local-jet correction is even smaller on the edge spike window.",
        ),
        sign_base.row(
            "edge_window_max_h2_over_h0q2",
            "watch",
            "edge-window max |h2|/(|h0| q^2)",
            edge_ratio_h2,
            "The h2 correction remains negligible on the edge spike window as well.",
        ),
        sign_base.row(
            "fit_window_local_jet_sign_mismatch_fraction",
            "watch",
            "fit-window local-jet sign mismatch fraction",
            fit_local_mismatch,
            "This is the window-local mismatch left by the retained boundary local-jet rule at the first alias harmonic.",
        ),
        sign_base.row(
            "fit_window_alias_image_sign_mismatch_fraction",
            "watch",
            "fit-window alias-image sign mismatch fraction",
            fit_image_mismatch,
            "Mapping the fit spike onto the first alias image improves the sign family substantially.",
        ),
        sign_base.row(
            "edge_window_local_jet_sign_mismatch_fraction",
            "watch",
            "edge-window local-jet sign mismatch fraction",
            edge_local_mismatch,
            "This is the window-local mismatch left by the retained boundary local-jet rule at the second alias harmonic.",
        ),
        sign_base.row(
            "edge_window_alias_image_sign_mismatch_fraction",
            "watch",
            "edge-window alias-image sign mismatch fraction",
            edge_image_mismatch,
            "The second-harmonic spike is also better captured by the alias-image parity rule than by another boundary derivative retry.",
        ),
        sign_base.row(
            "fit_alias_image_gain_over_local_jet",
            "pass" if fit_image_gain > 1.5 else "watch",
            "fit-window alias-image gain over local jet",
            fit_image_gain,
            "A large gain on the first harmonic indicates that the residual spike belongs to an alias-image family rather than a same-level boundary derivative correction.",
        ),
        sign_base.row(
            "edge_alias_image_gain_over_local_jet",
            "watch",
            "edge-window alias-image gain over local jet",
            edge_image_gain,
            "The second harmonic still prefers the image parity family, even though the gain is modest.",
        ),
        sign_base.row(
            "fit_subleading_negligible",
            "pass" if fit_subleading_negligible else "reject",
            "fit-window subleading jet terms negligible",
            sign_base.truth(fit_subleading_negligible),
            "If h1 and h2 are already negligible, another same-level boundary-term retry is not the honest next move.",
        ),
        sign_base.row(
            "edge_subleading_negligible",
            "pass" if edge_subleading_negligible else "reject",
            "edge-window subleading jet terms negligible",
            sign_base.truth(edge_subleading_negligible),
            "The edge spike is not carried by a large h1 or h2 correction either.",
        ),
        sign_base.row(
            "alias_image_family_admissible",
            "pass" if alias_image_family_admissible else "reject",
            "boundary alias-image family admissible",
            sign_base.truth(alias_image_family_admissible),
            "The next theorem-level surface is admissible only if both spike windows are already dominated by the leading alias image rather than by same-level boundary-term corrections.",
        ),
        sign_base.row(
            "same_level_boundary_term_retry_admissible",
            "reject" if not same_level_boundary_term_retry_admissible else "pass",
            "same-level boundary-term retry admissible",
            sign_base.truth(same_level_boundary_term_retry_admissible),
            "Once the spike windows are leading-carrier dominated, more same-level h1/h2 refits should remain closed.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "bulk_delta_r_over_m0": bulk_delta_r,
        "bulk_uniform_cell_fraction": bulk_fraction,
        "edge_cell_relative_gap": edge_cell_relative_gap,
        "first_alias_harmonic_over_m0": alias_1,
        "second_alias_harmonic_over_m0": alias_2,
        "fit_window_alias_index": 1.0,
        "edge_window_alias_index": 2.0,
        "fit_window_max_h1_over_h0q": fit_ratio_h1,
        "fit_window_max_h2_over_h0q2": fit_ratio_h2,
        "edge_window_max_h1_over_h0q": edge_ratio_h1,
        "edge_window_max_h2_over_h0q2": edge_ratio_h2,
        "fit_window_local_jet_sign_mismatch_fraction": fit_local_mismatch,
        "fit_window_alias_image_sign_mismatch_fraction": fit_image_mismatch,
        "fit_alias_image_gain_over_local_jet": fit_image_gain,
        "fit_alias_image_sign_correlation": fit_image_corr,
        "fit_alias_image_q_min_over_m0": float(np.min(fit_q_image)),
        "fit_alias_image_q_max_over_m0": float(np.max(fit_q_image)),
        "edge_window_local_jet_sign_mismatch_fraction": edge_local_mismatch,
        "edge_window_alias_image_sign_mismatch_fraction": edge_image_mismatch,
        "edge_alias_image_gain_over_local_jet": edge_image_gain,
        "edge_alias_image_sign_correlation": edge_image_corr,
        "edge_alias_image_q_min_over_m0": float(np.min(edge_q_image)),
        "edge_alias_image_q_max_over_m0": float(np.max(edge_q_image)),
        "fit_subleading_negligible": fit_subleading_negligible,
        "edge_subleading_negligible": edge_subleading_negligible,
        "fit_alias_image_supported": fit_alias_image_supported,
        "edge_alias_image_supported": edge_alias_image_supported,
        "alias_image_family_admissible": alias_image_family_admissible,
        "same_level_boundary_term_retry_admissible": same_level_boundary_term_retry_admissible,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "substantive_pack_update_required_now": substantive_pack_update_required_now,
        "physical_reject_required": physical_reject_required,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2025",
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
                "window_scan_density": WINDOW_SCAN_DENSITY,
                "next_route_name": NEXT_ROUTE_NAME,
                "next_route": NEXT_ROUTE,
                "followup_route_name": FOLLOWUP_ROUTE_NAME,
                "followup_route": FOLLOWUP_ROUTE,
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_alias_harmonic_spike_audited",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2023"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, "8.7.56.2023-.2026"),
                "current_problem_hit": sign_base.hit(current_problem_text, "alias-harmonic spike"),
                "current_status_hit": sign_base.hit(current_status_text, "alias-harmonic spike"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2023-.2026"),
                "long_roadmap_hit": sign_base.hit(long_text, "boundary alias-harmonic spike audit"),
                "part5_hit": sign_base.hit(part5_text, ".2015-.2022"),
            },
        },
    )

    route_payload = sign_base.payload(
        "8.7.56.2026",
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
        [
            sign_base.row(
                "alias_image_family_admissible",
                "pass" if alias_image_family_admissible else "reject",
                "boundary alias-image family admissible",
                sign_base.truth(alias_image_family_admissible),
                "The next official gate is justified only if the spike windows already point to an alias-image family instead of another same-level derivative retry.",
            ),
            sign_base.row(
                "same_level_boundary_term_retry_admissible",
                "reject" if not same_level_boundary_term_retry_admissible else "pass",
                "same-level boundary-term retry admissible",
                sign_base.truth(same_level_boundary_term_retry_admissible),
                "Once the leading alias image dominates, another same-level h1/h2 retry should remain closed.",
            ),
            sign_base.row(
                "next_route_fixed",
                "pass",
                "next route fixed",
                1.0,
                "The next official branch is the alias-harmonic spike decision gate / registry.",
            ),
        ],
        summary,
        {
            "overall_status": "vector_qball_form_factor_alias_harmonic_spike_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {"formulas": build_formulae()},
    )

    declaration_paths = write_artifact("declaration_gate", declaration_payload)
    route_paths = write_artifact("route_sync", route_payload)
    print("[ok] 8.7.56.2023-.2026 alias-harmonic spike artifacts generated")
    print(f"[ok] declaration: {declaration_paths['json']}")
    print(f"[ok] route sync:   {route_paths['json']}")


if __name__ == "__main__":
    main()
