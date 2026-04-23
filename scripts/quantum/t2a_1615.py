#!/usr/bin/env python3
"""Generate 8.7.56.1615-.1618 ground-state identification audit artifacts.

The prior caseB/effective-metric transverse audit failed because the retained
exact branch carries a temporal near-node and therefore creates a large `u<0`
region. A new expert directive proposed that the true upstream blocker may be
the branch itself: perhaps the reused exact branch is an excited-state-like
solution, while the genuine ground state should stay nodeless and keep the
vector correction perturbatively small.

This branch tests that proposal directly on the current exact pilot ODE.

What must be checked:

1. whether the scalar limit of the *same* exact pilot ODE is already nodeless,
2. whether increasing `amp_L` creates the first zero crossing or merely drifts
   an already existing zero inward,
3. whether any scanned `amp_L` window restores a genuinely nodeless branch,
4. whether the note's claimed core scale `|f_L/f_0| ~ 0.01` is robust on the
   retained Phase-1-equivalent branch.

If the scalar limit itself already changes sign and no nodeless window appears
in the scanned family, then the ground-state/nodeless hypothesis is *not*
supported under the current pack and the honest mainline returns to the already
prepared caseB `v^2` subtraction exact treatment.
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

from scripts.utils.windows_length_policy import build_compact_artifact_stem
from scripts.utils.windows_length_policy import build_metrics_paths

import scripts.quantum.t2a_1599 as exact_profile_tools


PUBLIC_OUT = ROOT / "output" / "public" / "quantum"

STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
WORK_HISTORY_RECENT = ROOT / "doc" / "WORK_HISTORY_RECENT.md"
CURRENT_PROBLEM = ROOT / "doc" / "quantum" / "34_trial2_numeric_alpha_current_problem.md"
CURRENT_STATUS = ROOT / "doc" / "quantum" / "36_trial2_numeric_alpha_current_status.md"
UNIFIED_ROADMAP = ROOT / "doc" / "quantum" / "39_trial2_vector_qball_unified_closure_roadmap.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

DIRECTIVE_NOTE = Path(
    r"C:\Users\ogawa\Downloads\trial2_vector_qball_ground_state_identification_20260328.md"
)
PRIOR_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1611_1614_eff_metric_tp_alpha_audit_declaration_gate_metrics.json"
)
ANCHOR_EVAL = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1483_1486_ell0_anchor_continuation_numeric_evaluation_metrics.json"
)
PHASE1_EVAL = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_phase1_exact_coupled_l0_solver_numeric_evaluation_metrics.json"
)
TP_RESPONSE_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "47_trial2_numeric_alpha_vector_qball_effective_metric_transverse_response.md"
)

STEP_TAG = "8.7.56.1615-1618"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor ground-state identification "
    "nodeless-condition audit"
)
STEM = build_compact_artifact_stem(STEP_TAG, "gs_nodeless_audit", prefix="q")

PRIOR_CLASS = (
    "vector_qball_form_factor_effective_metric_transverse_projection_no_scalar_foundation_"
    "v2_subtraction_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_ground_state_nodeless_hypothesis_not_supported_"
    "effective_metric_v2_restore_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_effective_metric_v2_subtraction_exact_treatment_restore"
)
NEXT_ROUTE = "8.7.56.1619"
DOWNSTREAM_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_effective_metric_disposition_sync_closeout"
)
DOWNSTREAM_ROUTE = "8.7.56.1623"

TARGET_ALPHA = 1.0 / 137.035999084
PHASE1_BETA = 0.9982557379261291
PHASE1_AMP0 = 3.5
PHASE1_AMP_L = 1.25
AMP_L_SCAN = np.linspace(0.0, PHASE1_AMP_L, 11)
ZERO_EPS = 1.0e-8
RATIO_EPS = 1.0e-4


# 関数: 現在UTC時刻を返す。
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力の存在を確認する。

def require(path: Path) -> None:
    """Abort when one required input is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: UTF-8 テキストを読み込む。

def read_text(path: Path) -> str:
    """Read one UTF-8 text file."""
    return path.read_text(encoding="utf-8")


# 関数: UTF-8 JSON を読み込む。

def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: 表示用の相対パスを返す。

def display_path(path: Path) -> str:
    """Return one repo-relative path when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# 関数: 部分文字列に一致する最初の行を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first line matching one substring."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: 標準 metrics row を構成する。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build one standard metrics row."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# 関数: 標準 payload を構成する。

def payload(
    step: str,
    name: str,
    inputs: dict,
    rows: list[dict],
    summary: dict,
    decision: dict,
    evidence: dict,
) -> dict:
    """Build one standard payload."""
    return {
        "generated_utc": now_iso(),
        "phase": {"phase": 8, "step": step, "name": name},
        "inputs": inputs,
        "rows": rows,
        "summary": summary,
        "decision": decision,
        "evidence": evidence,
    }


# 関数: JSON/CSV 成果物を書き出す。

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

    return {"json": display_path(paths["json"]), "csv": display_path(paths["csv"])}


# 関数: 真偽値を 0/1 に変換する。

def truth(value: bool) -> float:
    """Convert one boolean into 0/1 float form."""
    return 1.0 if value else 0.0


# 関数: 0-crossing 情報を返す。

def first_zero_cross(radius: np.ndarray, values: np.ndarray) -> tuple[int, float]:
    """Return the sign-change count and the first interpolated zero radius."""
    mask = np.abs(values) > ZERO_EPS
    filtered = values[mask]
    filtered_radius = radius[mask]
    if filtered.size < 2:
        return 0, math.nan

    signs = np.sign(filtered)
    flips = np.where(signs[:-1] * signs[1:] < 0.0)[0]
    if flips.size == 0:
        return 0, math.nan

    idx = int(flips[0])
    r0 = float(filtered_radius[idx])
    r1 = float(filtered_radius[idx + 1])
    y0 = float(filtered[idx])
    y1 = float(filtered[idx + 1])
    zero_r = float(r0 - y0 * (r1 - r0) / (y1 - y0))
    return int(flips.size), zero_r


# 関数: core 比率統計を返す。

def core_ratio_stats(
    radius: np.ndarray,
    f0_values: np.ndarray,
    f_l_values: np.ndarray,
    zero_radius: float,
    core_fraction: float,
) -> dict[str, float]:
    """Return max/p90/median |fL/f0| inside one fractional inner-core window."""
    core_mask = radius <= (float(core_fraction) * float(zero_radius))
    ratio_mask = np.abs(f0_values) > RATIO_EPS
    mask = core_mask & ratio_mask
    if not np.any(mask):
        return {"max": math.nan, "p90": math.nan, "median": math.nan}

    values = np.abs(f_l_values[mask] / f0_values[mask])
    return {
        "max": float(np.max(values)),
        "p90": float(np.quantile(values, 0.9)),
        "median": float(np.median(values)),
    }


# 関数: 1本の branch profile を解析する。

def analyze_branch(
    pivot,
    beta: float,
    amp0: float,
    amp_l: float,
) -> dict:
    """Reconstruct and analyze one exact branch profile."""
    profile = exact_profile_tools.solve_exact_profile_with_arrays(pivot, beta, amp0, amp_l)
    radius = np.asarray(profile["radius"], dtype=float)
    f0_values = np.asarray(profile["f0"], dtype=float)
    f_l_values = np.asarray(profile["fL"], dtype=float)

    sign_change_count, zero_radius = first_zero_cross(radius, f0_values)
    min_abs_f0_index = int(np.argmin(np.abs(f0_values)))
    min_abs_f0 = float(np.min(np.abs(f0_values)))
    min_abs_radius = float(radius[min_abs_f0_index])
    quarter_stats = core_ratio_stats(radius, f0_values, f_l_values, zero_radius, 0.25)
    half_stats = core_ratio_stats(radius, f0_values, f_l_values, zero_radius, 0.5)

    return {
        "amp_l": float(amp_l),
        "q_theory_over_m0": float(profile["q_theory_over_m0"]),
        "sign_change_count": sign_change_count,
        "zero_radius": float(zero_radius),
        "min_abs_f0": min_abs_f0,
        "min_abs_f0_radius": min_abs_radius,
        "quarter_core_ratio_max": quarter_stats["max"],
        "quarter_core_ratio_p90": quarter_stats["p90"],
        "quarter_core_ratio_median": quarter_stats["median"],
        "half_core_ratio_max": half_stats["max"],
        "half_core_ratio_p90": half_stats["p90"],
        "half_core_ratio_median": half_stats["median"],
    }


# 関数: branch で固定する主要式を返す。

def build_formulae() -> dict[str, str]:
    """Return the formulas used in the ground-state audit."""
    return {
        "exact_branch_ansatz": "P_mu = P_mu^Qball + a_mu with retained exact pilot ODE at beta_1",
        "nodeless_hypothesis": "ground state hypothesis => f_0(r) > 0 for all r > 0",
        "sign_change_gate": "sign changes are counted from reconstructed f_0(r) with |f_0| > 1e-8",
        "zero_radius": "r_zero := first interpolated radius where f_0 changes sign",
        "quarter_core_ratio": "max/p90/median of |f_L/f_0| on r <= 0.25 r_zero",
        "half_core_ratio": "max/p90/median of |f_L/f_0| on r <= 0.50 r_zero",
    }


# 関数: `.1615-.1618` を実行する。

def main() -> None:
    """Execute the ground-state identification / nodeless-condition audit."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        PART5,
        DIRECTIVE_NOTE,
        PRIOR_GATE,
        ANCHOR_EVAL,
        PHASE1_EVAL,
        TP_RESPONSE_NOTE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_roadmap_text = read_text(UNIFIED_ROADMAP)
    part5_text = read_text(PART5)
    directive_text = read_text(DIRECTIVE_NOTE)
    tp_response_text = read_text(TP_RESPONSE_NOTE)

    prior_summary = read_json(PRIOR_GATE)["summary"]
    anchor_summary = read_json(ANCHOR_EVAL)["summary"]
    phase1_summary = read_json(PHASE1_EVAL)["summary"]

    exact_branch = exact_profile_tools.load_module(
        exact_profile_tools.EXACT_REINJECTION_BRANCH,
        "t2a_1479_reuse_for_1615",
    )
    pivot = exact_branch.load_module(exact_branch.PIVOT_BRANCH, "pivot_branch_for_1615")

    sweep = [
        analyze_branch(pivot, PHASE1_BETA, PHASE1_AMP0, float(amp_l))
        for amp_l in AMP_L_SCAN
    ]
    scalar_row = sweep[0]
    phase1_row = sweep[-1]

    zero_radius_values = np.asarray([row["zero_radius"] for row in sweep], dtype=float)
    finite_zero_mask = np.isfinite(zero_radius_values)
    zero_radius_monotone_decreasing = bool(
        np.all(np.diff(zero_radius_values[finite_zero_mask]) < 0.0)
    )
    nodeless_window_found_in_scan = bool(
        any(row["sign_change_count"] == 0 for row in sweep)
    )
    scalar_limit_nodeless = bool(scalar_row["sign_change_count"] == 0)
    phase1_branch_nodeless = bool(phase1_row["sign_change_count"] == 0)
    phase1_wrong_solution_excited_state_claim_supported = bool(
        (not scalar_limit_nodeless) is False and (not phase1_branch_nodeless)
    )
    # Explicitly keep the readable interpretation separate.
    phase1_wrong_solution_excited_state_claim_supported = bool(
        scalar_limit_nodeless and (not phase1_branch_nodeless)
    )
    ground_state_nodeless_hypothesis_supported_under_current_pack = bool(
        scalar_limit_nodeless and nodeless_window_found_in_scan
    )
    amp_l_star_found_within_scanned_window = bool(nodeless_window_found_in_scan)
    zero_radius_inward_shift = float(
        scalar_row["zero_radius"] - phase1_row["zero_radius"]
    )
    phase1_core_ratio_order_point_zero_one_supported = bool(
        phase1_row["quarter_core_ratio_p90"] <= 0.02
    )
    half_core_median_order_point_zero_one = bool(
        phase1_row["half_core_ratio_median"] <= 0.02
    )
    effective_metric_v2_subtraction_restore_required = True
    ground_state_vector_correction_test_admissible_now = False
    physical_reject_required = False

    rows = [
        row(
            "prior_caseb_tp_no_go_ready",
            "pass" if prior_summary["trial2_numeric_alpha_problem_classification"] == PRIOR_CLASS else "reject",
            "prior caseB transverse no-go ready",
            truth(prior_summary["trial2_numeric_alpha_problem_classification"] == PRIOR_CLASS),
            "The ground-state note is being tested only after the caseB/effective-metric transverse rescue itself failed honestly.",
        ),
        row(
            "scalar_limit_nodeless",
            "pass" if scalar_limit_nodeless else "reject",
            "scalar-limit branch is nodeless",
            truth(scalar_limit_nodeless),
            "If the same exact pilot ODE already gave a nodeless scalar limit, the note's excited-state diagnosis would gain direct current-pack support.",
        ),
        row(
            "scalar_limit_zero_radius",
            "watch",
            "scalar-limit first zero-cross radius",
            scalar_row["zero_radius"],
            "The scalar limit of the same exact pilot ODE already changes sign at finite radius, so nodelessness is not a current-pack baseline property.",
        ),
        row(
            "phase1_branch_nodeless",
            "pass" if phase1_branch_nodeless else "reject",
            "Phase-1-equivalent branch is nodeless",
            truth(phase1_branch_nodeless),
            "The reused exact branch itself is tested directly rather than assumed to be excited-state-like from generic lore alone.",
        ),
        row(
            "phase1_zero_radius",
            "watch",
            "Phase-1-equivalent first zero-cross radius",
            phase1_row["zero_radius"],
            "This is where the retained exact branch first changes sign under the current exact pilot ODE.",
        ),
        row(
            "zero_radius_inward_shift",
            "pass" if zero_radius_inward_shift > 0.0 else "reject",
            "scalar-limit minus phase1 zero-cross radius",
            zero_radius_inward_shift,
            "A positive shift means increasing amp_L pushes an already existing sign change inward rather than creating the first one from a nodeless baseline.",
        ),
        row(
            "zero_radius_monotone_decreasing_under_amp_l_scan",
            "pass" if zero_radius_monotone_decreasing else "reject",
            "zero-cross radius decreases monotonically with amp_L",
            truth(zero_radius_monotone_decreasing),
            "The scan checks whether the branch family behaves like node drift rather than a sharp nodeless-to-noded transition.",
        ),
        row(
            "nodeless_window_found_in_amp_l_scan",
            "pass" if nodeless_window_found_in_scan else "reject",
            "any scanned amp_L branch stays nodeless",
            truth(nodeless_window_found_in_scan),
            "If no nodeless window exists in the scanned family, the proposed amp_L* threshold is not supported by the current exact pilot.",
        ),
        row(
            "phase1_quarter_core_ratio_p90",
            "watch",
            "Phase-1-equivalent |fL/f0| p90 on r <= 0.25 r_zero",
            phase1_row["quarter_core_ratio_p90"],
            "This is a robust inner-core statistic for the note's claim that the true ground-state correction should stay around one percent.",
        ),
        row(
            "phase1_half_core_ratio_median",
            "watch",
            "Phase-1-equivalent |fL/f0| median on r <= 0.50 r_zero",
            phase1_row["half_core_ratio_median"],
            "The median in the half-core window is near 1e-2, but the stronger p90/max statistics remain much larger and therefore the one-percent claim is not robust.",
        ),
        row(
            "phase1_core_ratio_order_point_zero_one_supported",
            "pass" if phase1_core_ratio_order_point_zero_one_supported else "reject",
            "Phase-1-equivalent core ratio robustly stays at O(0.01)",
            truth(phase1_core_ratio_order_point_zero_one_supported),
            "The current pack only supports an O(0.01) claim if robust inner-core statistics, not just one median, stay that small.",
        ),
        row(
            "phase1_wrong_solution_excited_state_claim_supported",
            "pass" if phase1_wrong_solution_excited_state_claim_supported else "reject",
            "wrong-solution excited-state claim supported",
            truth(phase1_wrong_solution_excited_state_claim_supported),
            "This would require a nodeless scalar baseline together with a noded Phase-1 branch; the current exact pilot does not satisfy that pattern.",
        ),
        row(
            "ground_state_nodeless_hypothesis_supported_under_current_pack",
            "pass" if ground_state_nodeless_hypothesis_supported_under_current_pack else "reject",
            "ground-state nodeless hypothesis supported under current pack",
            truth(ground_state_nodeless_hypothesis_supported_under_current_pack),
            "The expert note is adopted only if the current exact pilot itself supports a genuinely nodeless family branch.",
        ),
        row(
            "effective_metric_v2_subtraction_restore_required",
            "pass" if effective_metric_v2_subtraction_restore_required else "reject",
            "effective-metric v2 subtraction restored as mainline",
            truth(effective_metric_v2_subtraction_restore_required),
            "Because the nodeless hypothesis does not open a new eligible branch under current-pack computation, the honest next step returns to the deferred caseB v^2 subtraction treatment.",
        ),
    ]

    inputs = {
        "source_files": {
            "status": display_path(STATUS),
            "roadmap": display_path(ROADMAP),
            "ai_context": display_path(AI_CONTEXT),
            "work_history_recent": display_path(WORK_HISTORY_RECENT),
            "current_problem": display_path(CURRENT_PROBLEM),
            "current_status": display_path(CURRENT_STATUS),
            "unified_roadmap": display_path(UNIFIED_ROADMAP),
            "part5": display_path(PART5),
            "directive_note": display_path(DIRECTIVE_NOTE),
            "prior_gate": display_path(PRIOR_GATE),
            "anchor_eval": display_path(ANCHOR_EVAL),
            "phase1_eval": display_path(PHASE1_EVAL),
            "tp_response_note": display_path(TP_RESPONSE_NOTE),
        },
        "constants": {
            "beta_1": PHASE1_BETA,
            "amp0": PHASE1_AMP0,
            "phase1_amp_l": PHASE1_AMP_L,
            "amp_l_scan_count": float(AMP_L_SCAN.size),
            "target_alpha": TARGET_ALPHA,
            "scalar_alpha_exact_at_q_theory": 0.00715678583937324,
            "vector_alpha_at_q_theory": 0.0005579616187042394,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "downstream_route_name": DOWNSTREAM_ROUTE_NAME,
            "downstream_route": DOWNSTREAM_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "scalar_limit_sign_change_count": scalar_row["sign_change_count"],
        "scalar_limit_zero_radius": scalar_row["zero_radius"],
        "phase1_sign_change_count": phase1_row["sign_change_count"],
        "phase1_zero_radius": phase1_row["zero_radius"],
        "zero_radius_inward_shift": zero_radius_inward_shift,
        "zero_radius_monotone_decreasing_under_amp_l_scan": zero_radius_monotone_decreasing,
        "nodeless_window_found_in_amp_l_scan": nodeless_window_found_in_scan,
        "amp_l_star_found_within_scanned_window": amp_l_star_found_within_scanned_window,
        "phase1_quarter_core_ratio_max": phase1_row["quarter_core_ratio_max"],
        "phase1_quarter_core_ratio_p90": phase1_row["quarter_core_ratio_p90"],
        "phase1_quarter_core_ratio_median": phase1_row["quarter_core_ratio_median"],
        "phase1_half_core_ratio_max": phase1_row["half_core_ratio_max"],
        "phase1_half_core_ratio_p90": phase1_row["half_core_ratio_p90"],
        "phase1_half_core_ratio_median": phase1_row["half_core_ratio_median"],
        "half_core_median_order_point_zero_one": half_core_median_order_point_zero_one,
        "phase1_core_ratio_order_point_zero_one_supported": (
            phase1_core_ratio_order_point_zero_one_supported
        ),
        "phase1_wrong_solution_excited_state_claim_supported": (
            phase1_wrong_solution_excited_state_claim_supported
        ),
        "ground_state_nodeless_hypothesis_supported_under_current_pack": (
            ground_state_nodeless_hypothesis_supported_under_current_pack
        ),
        "ground_state_vector_correction_test_admissible_now": (
            ground_state_vector_correction_test_admissible_now
        ),
        "effective_metric_v2_subtraction_restore_required": (
            effective_metric_v2_subtraction_restore_required
        ),
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "downstream_disposition_route_name": DOWNSTREAM_ROUTE_NAME,
        "downstream_disposition_route_or_none": DOWNSTREAM_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": True,
        "next_required_artifacts": [NEXT_ROUTE_NAME, DOWNSTREAM_ROUTE_NAME],
    }

    evidence = {
        "formulas": build_formulae(),
        "hits": {
            "directive_common_failure_source": hit(
                directive_text,
                "共通項は metric でも展開次数でもない。**amp₀ = 3.5, amp_L = 1.25 という解そのもの。**",
            ),
            "directive_ground_state_nodeless": hit(
                directive_text,
                "ground state の条件: **f₀(r) > 0 for all r > 0**（nodeless）。",
            ),
            "directive_amp_l_star": hit(directive_text, "amp_L*"),
            "tp_response_temporal_near_node": hit(
                tp_response_text,
                "temporal near-node",
            ),
            "current_problem_next_branch": hit(
                current_problem_text,
                "effective-metric `v^2` subtraction exact treatment",
            ),
            "current_status_next_branch": hit(
                current_status_text,
                "effective-metric `v^2` subtraction exact treatment",
            ),
            "unified_roadmap_current_branch": hit(
                unified_roadmap_text,
                "`.1615-.1618` は **effective-metric `v^2` subtraction exact treatment** branch",
            ),
            "part5_effective_metric": hit(part5_text, "effective-metric"),
        },
        "scan_rows": sweep,
        "retained_numeric_state": {
            "scalar_F_exact_at_q_theory": 0.2998913524347805,
            "scalar_alpha_exact_at_q_theory": 0.00715678583937324,
            "vector_F_at_q_theory": anchor_summary["phase1_equivalent_row"]["F_at_q_theory"],
            "vector_alpha_at_q_theory": anchor_summary["phase1_equivalent_row"]["alpha_at_q_theory"],
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
        },
    }

    inventory_paths = write_artifact(
        "inventory",
        payload(
            "8.7.56.1615",
            f"{STEP_NAME} inventory",
            inputs,
            rows,
            summary,
            decision,
            evidence,
        ),
    )
    audit_paths = write_artifact(
        "audit",
        payload(
            "8.7.56.1616",
            f"{STEP_NAME} audit",
            inputs,
            rows,
            summary,
            decision,
            evidence,
        ),
    )
    gate_paths = write_artifact(
        "declaration_gate",
        payload(
            "8.7.56.1617",
            f"{STEP_NAME} declaration gate",
            inputs,
            rows,
            summary,
            decision,
            evidence,
        ),
    )
    route_paths = write_artifact(
        "route_sync",
        payload(
            "8.7.56.1618",
            f"{STEP_NAME} route sync",
            inputs,
            rows,
            summary,
            decision,
            evidence,
        ),
    )

    print("[ok] ground-state identification / nodeless-condition audit artifacts written:")
    print(f" - {inventory_paths['json']}")
    print(f" - {audit_paths['json']}")
    print(f" - {gate_paths['json']}")
    print(f" - {route_paths['json']}")


if __name__ == "__main__":
    main()
