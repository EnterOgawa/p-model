#!/usr/bin/env python3
"""Generate 8.7.56.2055-.2058 higher-harmonic lattice loading artifacts.

This branch reactivates the higher-harmonic windowwise signed-rule surface
after smooth q-dependent loading has been closed. The new first shot is a
discrete boundary bulk-lattice rule built from the retained constant-slip
theorem and the finite-box bulk spacing. If that lattice survives across
core, extension, and farther harmonic windows, the unresolved gap shrinks to
the loading-index theorem rather than to another smooth loading retry.
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


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.quantum.t2a_1843 as sign_base
import scripts.quantum.t2a_2023 as alias_base
import scripts.quantum.t2a_2031 as phase_base
import scripts.quantum.t2a_2047 as loading_base
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
PRIOR_GATE = PUBLIC_OUT / "q_8_7_56_2051_2054_alias_image_phase_slip_loading_registry_declaration_gate_metrics.json"
PRIOR_AUDIT = PUBLIC_OUT / "q_8_7_56_2039_2042_alias_image_phase_slip_theorem_declaration_gate_metrics.json"

STEP_TAG = "8.7.56.2055-2058"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor higher-harmonic "
    "windowwise phase-slip signed-rule reactivation"
)
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "higher_harmonic_lattice_loading",
    prefix="q",
)

PRIOR_CLASS = (
    "vector_qball_form_factor_q_dependent_boundary_loading_closed_"
    "higher_harmonic_windowwise_signed_rule_reactivation_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_boundary_bulk_lattice_signed_rule_retained_"
    "loading_index_theorem_gate_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_higher_harmonic_loading_"
    "decision_gate_registry"
)
NEXT_ROUTE = "8.7.56.2059"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_exact_loading_index_theorem_"
    "or_farther_harmonic_extension_reactivation"
)
FOLLOWUP_ROUTE = "8.7.56.2063"

HARMONIC_MIN = 3
HARMONIC_FIT_MAX = 8
HARMONIC_EXTENSION_MAX = 16
HARMONIC_FARTHER_MAX = 24
MAX_LOADING_INDEX = 80
BASE_SCAN_STEPS = 121


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


# 関数: テキスト中の最初の一致行を返す。

def find_line(text: str, pattern: str) -> dict[str, object] | None:
    """Return the first matching line payload for one text pattern."""
    for line_number, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {
                "pattern": pattern,
                "line": line_number,
                "text": line.strip(),
            }

    return None


# 関数: higher-harmonic window inventory を構成する。

def build_higher_harmonic_windows(
    radius: np.ndarray,
    weight: np.ndarray,
    norm: float,
    alias_1: float,
) -> list[dict[str, object]]:
    """Build harmonic windows from n=3 through n=24."""
    fit_offsets = (loading_base.FIT_Q_MIN - alias_1, loading_base.FIT_Q_MAX - alias_1)
    edge_offsets = (
        loading_base.EDGE_Q_MIN - (2.0 * alias_1),
        loading_base.EDGE_Q_MAX - (2.0 * alias_1),
    )
    windows: list[dict[str, object]] = []
    for harmonic_index in range(HARMONIC_MIN, HARMONIC_FARTHER_MAX + 1):
        alias_harmonic = harmonic_index * alias_1
        offsets = fit_offsets if (harmonic_index % 2) == 1 else edge_offsets
        q_min, q_max = loading_base.translated_window(alias_harmonic, offsets)
        q_scan = np.linspace(
            q_min,
            q_max,
            int(round((q_max - q_min) * loading_base.WINDOW_SCAN_DENSITY)) + 1,
        )
        exact_values, exact_abs, exact_sign = phase_base.exact_sign_data(
            radius,
            weight,
            norm,
            q_scan,
        )
        windows.append(
            {
                "harmonic_index": harmonic_index,
                "alias_harmonic": float(alias_harmonic),
                "q_min": float(q_min),
                "q_max": float(q_max),
                "q_scan": q_scan,
                "exact_values": exact_values,
                "exact_abs": exact_abs,
                "exact_sign": exact_sign,
                "template_type": "fit" if (harmonic_index % 2) == 1 else "edge",
            }
        )

    return windows


# 関数: one window 上で lattice loading index を最適化する。

def optimize_window_lattice_index(
    window: dict[str, object],
    lookup_q: np.ndarray,
    lookup_values: np.ndarray,
    lattice_base: float,
    lattice_step: float,
) -> dict[str, float]:
    """Return the best lattice index and metrics on one harmonic window."""
    best_metrics: dict[str, float] | None = None
    best_loading_index = 0
    best_delta = lattice_base
    for loading_index in range(MAX_LOADING_INDEX + 1):
        delta_q = float(lattice_base + (loading_index * lattice_step))
        if delta_q > 1.5:
            break

        metrics = loading_base.evaluate_family(
            [window],
            lookup_q,
            lookup_values,
            lambda _n, q_array, delta=delta_q: np.full_like(q_array, delta, dtype=float),
        )[0]
        mismatch = float(metrics["sign_mismatch_fraction"])
        if best_metrics is None or mismatch < float(best_metrics["sign_mismatch_fraction"]):
            best_metrics = dict(metrics)
            best_loading_index = loading_index
            best_delta = delta_q

    assert best_metrics is not None
    best_metrics["loading_index"] = float(best_loading_index)
    best_metrics["delta_selected"] = float(best_delta)
    return best_metrics


# 関数: lattice family を全 window に評価する。

def evaluate_lattice_family(
    windows: list[dict[str, object]],
    lookup_q: np.ndarray,
    lookup_values: np.ndarray,
    lattice_base: float,
    lattice_step: float,
) -> list[dict[str, float]]:
    """Evaluate the harmonic loading lattice on all windows."""
    return [
        optimize_window_lattice_index(window, lookup_q, lookup_values, lattice_base, lattice_step)
        for window in windows
    ]


# 関数: one harmonic subset の summary を返す。

def summarize_harmonic_group(
    windows: list[dict[str, object]],
    results: list[dict[str, float]],
    harmonic_indices: list[int],
) -> dict[str, float]:
    """Return max mismatch/error and min correlation on one harmonic subset."""
    paired = [
        (window, result)
        for window, result in zip(windows, results, strict=True)
        if int(window["harmonic_index"]) in harmonic_indices
    ]
    return {
        "max_mismatch": float(max(result["sign_mismatch_fraction"] for _window, result in paired)),
        "max_abs_error": float(max(result["signed_reconstruction_max_abs_error"] for _window, result in paired)),
        "min_correlation": float(min(result["sign_correlation"] for _window, result in paired)),
        "max_loading_index": float(max(result["loading_index"] for _window, result in paired)),
        "min_loading_index": float(min(result["loading_index"] for _window, result in paired)),
    }


# 関数: bulk lattice の best base を direct search する。

def search_best_bulk_lattice_base(
    windows: list[dict[str, object]],
    lookup_q: np.ndarray,
    lookup_values: np.ndarray,
    lattice_step: float,
    fit_harmonics: list[int],
) -> tuple[float, list[dict[str, float]]]:
    """Search the best base in [0, lattice_step] on the core windows."""
    best_base = 0.0
    best_results: list[dict[str, float]] = []
    best_score = math.inf
    for trial_base in np.linspace(0.0, lattice_step, BASE_SCAN_STEPS):
        trial_results = evaluate_lattice_family(
            windows,
            lookup_q,
            lookup_values,
            float(trial_base),
            lattice_step,
        )
        trial_summary = summarize_harmonic_group(windows, trial_results, fit_harmonics)
        trial_score = float(trial_summary["max_mismatch"])
        if trial_score < best_score - 1.0e-12:
            best_score = trial_score
            best_base = float(trial_base)
            best_results = trial_results

    return best_base, best_results


# 関数: integer sequence の monotonicity を判定する。

def is_monotone(values: list[int]) -> bool:
    """Return whether one integer sequence is monotone."""
    diffs = np.diff(np.asarray(values, dtype=float))
    return bool(np.all(diffs >= -1.0e-12) or np.all(diffs <= 1.0e-12))


# 関数: audit で使う公式群を返す。

def build_formulae() -> dict[str, str]:
    """Return formulas used in the higher-harmonic lattice audit."""
    return {
        "retained_constant_theorem": "delta_q,jet = (3/2) (h1 / h0)",
        "bulk_lattice_step": "Delta_box = Delta r_bulk",
        "theorem_base": "delta_q,base^(box) = (Delta_box - (delta_q,jet mod Delta_box)) mod Delta_box",
        "harmonic_lattice_rule": "delta_q^(n,m) = delta_q,base^(box) + m_n Delta_box",
        "loading_index_rule": "m_n = argmin_{m in Z_{>=0}} mismatch_n(delta_q,base^(box) + m Delta_box)",
    }


# 関数: `.2055-.2058` を実行する。

def main() -> None:
    """Execute the higher-harmonic windowwise phase-slip signed-rule audit."""
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
        PRIOR_AUDIT,
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
    prior_audit_summary = sign_base.read_json(PRIOR_AUDIT)["summary"]
    inventory_ready = bool(prior_gate_summary["higher_harmonic_windowwise_loading_admissible_now"])

    qball_branch_refresh = sign_base.read_json(QBALL_BRANCH_REFRESH)
    scalar_ground_state = sign_base.extract_scalar_ground_state(qball_branch_refresh)
    qball_module = sign_base.load_qball_module()
    radius, field, _field_prime = qball_module.solve_full_profile(
        float(scalar_ground_state["beta_n"]),
        float(scalar_ground_state["central_amplitude"]),
    )
    weight = (field**2) * (radius**2)
    norm = float(np.trapezoid(weight, radius))
    bulk_delta_r, _bulk_fraction, _edge_gap = alias_base.bulk_grid_summary(radius)
    alias_1 = (2.0 * np.pi) / bulk_delta_r
    lookup_q = np.arange(
        0.0,
        phase_base.LOOKUP_Q_MAX + phase_base.LOOKUP_Q_STEP,
        phase_base.LOOKUP_Q_STEP,
        dtype=float,
    )
    lookup_values = phase_base.form_factor_array(radius, weight, norm, lookup_q)

    windows = build_higher_harmonic_windows(radius, weight, norm, alias_1)
    independent_deltas: list[float] = []
    for window in windows:
        best_delta, _metrics = loading_base.optimize_independent_delta(window, lookup_q, lookup_values)
        independent_deltas.append(float(best_delta))

    theorem_delta_jet = float(prior_audit_summary["delta_q_theorem_over_m0"])
    theorem_lattice_step = float(bulk_delta_r)
    theorem_lattice_base = float((theorem_lattice_step - np.mod(theorem_delta_jet, theorem_lattice_step)) % theorem_lattice_step)

    fit_harmonics = list(range(HARMONIC_MIN, HARMONIC_FIT_MAX + 1))
    extension_harmonics = list(range(HARMONIC_FIT_MAX + 1, HARMONIC_EXTENSION_MAX + 1))
    farther_harmonics = list(range(HARMONIC_EXTENSION_MAX + 1, HARMONIC_FARTHER_MAX + 1))

    searched_best_base, searched_best_results = search_best_bulk_lattice_base(
        windows,
        lookup_q,
        lookup_values,
        theorem_lattice_step,
        fit_harmonics,
    )
    theorem_results = evaluate_lattice_family(
        windows,
        lookup_q,
        lookup_values,
        theorem_lattice_base,
        theorem_lattice_step,
    )

    searched_fit_summary = summarize_harmonic_group(windows, searched_best_results, fit_harmonics)
    searched_extension_summary = summarize_harmonic_group(windows, searched_best_results, extension_harmonics)
    searched_farther_summary = summarize_harmonic_group(windows, searched_best_results, farther_harmonics)
    theorem_fit_summary = summarize_harmonic_group(windows, theorem_results, fit_harmonics)
    theorem_extension_summary = summarize_harmonic_group(windows, theorem_results, extension_harmonics)
    theorem_farther_summary = summarize_harmonic_group(windows, theorem_results, farther_harmonics)

    theorem_loading_indices = [int(round(result["loading_index"])) for result in theorem_results]
    theorem_delta_sequence = [float(result["delta_selected"]) for result in theorem_results]
    loading_index_monotone = is_monotone(theorem_loading_indices)
    delta_sequence_monotone = is_monotone([int(round(value * 10000.0)) for value in theorem_delta_sequence])
    theorem_vs_searched_base_gap = float(abs(theorem_lattice_base - searched_best_base))
    theorem_quantization_max_abs_gap = float(
        max(abs(independent_delta - delta_selected) for independent_delta, delta_selected in zip(independent_deltas, theorem_delta_sequence, strict=True))
    )
    theorem_quantization_rms_gap = float(
        np.sqrt(
            np.mean(
                [
                    (independent_delta - delta_selected) ** 2
                    for independent_delta, delta_selected in zip(independent_deltas, theorem_delta_sequence, strict=True)
                ]
            )
        )
    )

    boundary_bulk_lattice_signed_rule_supported = bool(
        theorem_fit_summary["max_mismatch"] <= 0.25
        and theorem_extension_summary["max_mismatch"] <= 0.25
        and theorem_farther_summary["max_mismatch"] <= 0.25
        and theorem_fit_summary["min_correlation"] >= 0.5
        and theorem_extension_summary["min_correlation"] >= 0.5
        and theorem_farther_summary["min_correlation"] >= 0.5
    )
    theorem_base_saturates_scanned_base = bool(theorem_vs_searched_base_gap <= 0.001)
    exact_loading_index_theorem_available = False
    farther_harmonic_extension_admissible_now = boundary_bulk_lattice_signed_rule_supported
    same_level_smooth_loading_retry_admissible = False
    same_level_base_scan_retry_admissible = False
    substantive_pack_update_required_now = False
    physical_reject_required = False

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "higher-harmonic loading inventory ready",
            sign_base.truth(inventory_ready),
            "The branch starts only after smooth q-dependent loading has been closed and the next honest surface is the higher-harmonic windowwise family.",
        ),
        sign_base.row(
            "bulk_delta_r_over_m0",
            "watch",
            "bulk lattice step Delta_box/m0",
            theorem_lattice_step,
            "The bulk radial box spacing is the first natural discrete loading step to test before introducing any new smooth q-dependent surface.",
        ),
        sign_base.row(
            "delta_q_theorem_over_m0",
            "watch",
            "retained local-jet theorem delta_q/m0",
            theorem_delta_jet,
            "The retained constant-slip theorem seeds the harmonic loading lattice through its complementary box residue.",
        ),
        sign_base.row(
            "theorem_lattice_base_over_m0",
            "watch",
            "boundary-theorem lattice base/m0",
            theorem_lattice_base,
            "The theorem base is the complementary residue of the retained constant theorem modulo the box bulk spacing.",
        ),
        sign_base.row(
            "searched_best_lattice_base_over_m0",
            "watch",
            "searched best bulk-lattice base/m0",
            searched_best_base,
            "A direct base scan is tracked so the theorem base can be compared against the numerically best bulk-lattice continuation.",
        ),
        sign_base.row(
            "theorem_vs_searched_base_gap_over_m0",
            "pass" if theorem_base_saturates_scanned_base else "watch",
            "gap between theorem base and searched best base/m0",
            theorem_vs_searched_base_gap,
            "A small gap means the boundary-only theorem already saturates the best bulk-lattice continuation without another same-level base refit.",
        ),
        sign_base.row(
            "theorem_fit_window_max_mismatch_fraction",
            "pass" if theorem_fit_summary["max_mismatch"] <= 0.25 else "reject",
            "theorem bulk-lattice max mismatch on harmonic 3..8",
            theorem_fit_summary["max_mismatch"],
            "The lattice family must survive the first translated harmonics before it can replace the smooth loading family.",
        ),
        sign_base.row(
            "theorem_extension_window_max_mismatch_fraction",
            "pass" if theorem_extension_summary["max_mismatch"] <= 0.25 else "reject",
            "theorem bulk-lattice max mismatch on harmonic 9..16",
            theorem_extension_summary["max_mismatch"],
            "The same lattice family must also survive the next unseen translated windows, not only the first reactivation block.",
        ),
        sign_base.row(
            "theorem_farther_window_max_mismatch_fraction",
            "pass" if theorem_farther_summary["max_mismatch"] <= 0.25 else "reject",
            "theorem bulk-lattice max mismatch on harmonic 17..24",
            theorem_farther_summary["max_mismatch"],
            "The farther harmonic block separates a true discrete lattice from a one-window overfit.",
        ),
        sign_base.row(
            "theorem_quantization_max_abs_gap_over_m0",
            "watch",
            "max quantization gap between independent delta and theorem lattice delta",
            theorem_quantization_max_abs_gap,
            "This gap measures how much continuous harmonic optima must move before they land on the boundary bulk lattice.",
        ),
        sign_base.row(
            "loading_index_monotone",
            "reject" if not loading_index_monotone else "pass",
            "loading-index sequence monotone",
            sign_base.truth(loading_index_monotone),
            "A non-monotone loading-index sequence means the lattice can be retained while the exact loading-index theorem is still missing.",
        ),
        sign_base.row(
            "boundary_bulk_lattice_signed_rule_supported",
            "pass" if boundary_bulk_lattice_signed_rule_supported else "reject",
            "boundary bulk-lattice signed rule supported",
            sign_base.truth(boundary_bulk_lattice_signed_rule_supported),
            "The current branch retains the bulk-lattice family only if it survives core, extension, and farther harmonic windows together.",
        ),
        sign_base.row(
            "exact_loading_index_theorem_available",
            "reject",
            "exact loading-index theorem available",
            sign_base.truth(exact_loading_index_theorem_available),
            "Even if the lattice survives, an exact theorem for the harmonic loading indices must still be derived separately.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "bulk_delta_r_over_m0": theorem_lattice_step,
        "delta_q_theorem_over_m0": theorem_delta_jet,
        "theorem_lattice_base_over_m0": theorem_lattice_base,
        "searched_best_lattice_base_over_m0": searched_best_base,
        "theorem_vs_searched_base_gap_over_m0": theorem_vs_searched_base_gap,
        "theorem_fit_window_max_mismatch_fraction": theorem_fit_summary["max_mismatch"],
        "theorem_fit_window_min_sign_correlation": theorem_fit_summary["min_correlation"],
        "theorem_extension_window_max_mismatch_fraction": theorem_extension_summary["max_mismatch"],
        "theorem_extension_window_min_sign_correlation": theorem_extension_summary["min_correlation"],
        "theorem_farther_window_max_mismatch_fraction": theorem_farther_summary["max_mismatch"],
        "theorem_farther_window_min_sign_correlation": theorem_farther_summary["min_correlation"],
        "searched_fit_window_max_mismatch_fraction": searched_fit_summary["max_mismatch"],
        "searched_extension_window_max_mismatch_fraction": searched_extension_summary["max_mismatch"],
        "searched_farther_window_max_mismatch_fraction": searched_farther_summary["max_mismatch"],
        "theorem_quantization_max_abs_gap_over_m0": theorem_quantization_max_abs_gap,
        "theorem_quantization_rms_gap_over_m0": theorem_quantization_rms_gap,
        "loading_index_monotone": loading_index_monotone,
        "delta_sequence_monotone": delta_sequence_monotone,
        "boundary_bulk_lattice_signed_rule_supported": boundary_bulk_lattice_signed_rule_supported,
        "theorem_base_saturates_scanned_base": theorem_base_saturates_scanned_base,
        "exact_loading_index_theorem_available": exact_loading_index_theorem_available,
        "same_level_smooth_loading_retry_admissible": same_level_smooth_loading_retry_admissible,
        "same_level_base_scan_retry_admissible": same_level_base_scan_retry_admissible,
        "farther_harmonic_extension_admissible_now": farther_harmonic_extension_admissible_now,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "substantive_pack_update_required_now": substantive_pack_update_required_now,
        "physical_reject_required": physical_reject_required,
    }
    for window, result, independent_delta in zip(windows, theorem_results, independent_deltas, strict=True):
        harmonic_index = int(window["harmonic_index"])
        summary[f"harmonic{harmonic_index}_loading_index"] = int(round(result["loading_index"]))
        summary[f"harmonic{harmonic_index}_delta_selected_over_m0"] = float(result["delta_selected"])
        summary[f"harmonic{harmonic_index}_independent_delta_over_m0"] = float(independent_delta)
        summary[f"harmonic{harmonic_index}_mismatch_fraction"] = float(result["sign_mismatch_fraction"])

    declaration_payload = sign_base.payload(
        "8.7.56.2057",
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
                "prior_audit": sign_base.display_path(PRIOR_AUDIT),
            },
            "constants": {
                "harmonic_range": [HARMONIC_MIN, HARMONIC_FARTHER_MAX],
                "fit_harmonics": fit_harmonics,
                "extension_harmonics": extension_harmonics,
                "farther_harmonics": farther_harmonics,
                "max_loading_index": MAX_LOADING_INDEX,
                "base_scan_steps": BASE_SCAN_STEPS,
                "next_route_name": NEXT_ROUTE_NAME,
                "next_route": NEXT_ROUTE,
                "followup_route_name": FOLLOWUP_ROUTE_NAME,
                "followup_route": FOLLOWUP_ROUTE,
            },
        },
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_higher_harmonic_lattice_loading_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2055"),
                "roadmap_branch_hit": find_line(roadmap_text, "8.7.56.2055-.2058"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2055"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2055"),
                "unified_roadmap_hit": find_line(unified_text, ".2055-.2058"),
                "long_roadmap_hit": find_line(long_text, ".2055-.2058"),
                "part5_hit": find_line(part5_text, ".2047-.2054"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_sync_rows = [
        sign_base.row(
            "status_synced",
            "pass",
            "STATUS sync target present",
            sign_base.truth(bool(find_line(status_text, "8.7.56.2055"))),
            "The branch is only valid if the official status already points to the harmonic-windowwise route.",
        ),
        sign_base.row(
            "roadmap_synced",
            "pass",
            "ROADMAP sync target present",
            sign_base.truth(bool(find_line(roadmap_text, "8.7.56.2055-.2058"))),
            "The public roadmap must expose the same windowwise reactivation branch before registry sync can proceed.",
        ),
        sign_base.row(
            "long_horizon_synced",
            "pass",
            "long-horizon roadmap sync target present",
            sign_base.truth(bool(find_line(long_text, ".2055-.2058"))),
            "The long-horizon roadmap must also show the harmonic-windowwise reactivation route.",
        ),
    ]
    route_sync_payload = sign_base.payload(
        "8.7.56.2058",
        STEP_NAME + " route sync",
        {
            "source_files": {
                "status": sign_base.display_path(STATUS),
                "roadmap": sign_base.display_path(ROADMAP),
                "current_problem": sign_base.display_path(CURRENT_PROBLEM),
                "current_status": sign_base.display_path(CURRENT_STATUS),
                "unified_roadmap": sign_base.display_path(UNIFIED_ROADMAP),
                "long_roadmap": sign_base.display_path(LONG_ROADMAP),
                "part5": sign_base.display_path(PART5),
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
            "overall_status": "vector_qball_form_factor_higher_harmonic_lattice_loading_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "hits": {
                "status_branch_hit": find_line(status_text, "8.7.56.2055"),
                "roadmap_branch_hit": find_line(roadmap_text, "8.7.56.2055-.2058"),
                "current_problem_hit": find_line(current_problem_text, "8.7.56.2055"),
                "current_status_hit": find_line(current_status_text, "8.7.56.2055"),
                "unified_roadmap_hit": find_line(unified_text, ".2055-.2058"),
                "long_roadmap_hit": find_line(long_text, ".2055-.2058"),
                "part5_hit": find_line(part5_text, ".2047-.2054"),
            },
        },
    )
    write_artifact("route_sync", route_sync_payload)


if __name__ == "__main__":
    main()
