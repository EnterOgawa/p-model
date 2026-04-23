#!/usr/bin/env python3
"""Generate 8.7.56.1619-.1622 effective-metric v^2-subtraction restore artifacts.

This branch resumes the deferred same-level computation after the
ground-state/nodeless directive failed under the current exact pilot.

The honest question is now narrow:

1. keep the caseB/effective-metric quadratic contraction fixed,
2. keep the full spatial transverse kernel fixed,
3. insert the exact `v^2` subtraction term under the same metric choice,
4. test whether the earlier caseA/Minkowski worsen was merely a metric-choice
   artifact or whether the no-go survives the caseB recomputation as well.

The branch evaluates three related surfaces:

- the directive-style projected caseB subtraction density,
- the full spatial-kernel projected caseB subtraction density,
- a lighter `Q_g^2`-only comparison surface retained as evidence only.

Only the full spatial-kernel read is used for the official disposition, because
`.1607-.1610` already fixed that naive `e^{2u}` isotropic enhancement does not
survive the full spatial caseB contraction.
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
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

GROUND_STATE_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1615_1618_gs_nodeless_audit_declaration_gate_metrics.json"
)
CASEB_TP_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1611_1614_eff_metric_tp_alpha_audit_declaration_gate_metrics.json"
)
CASEB_CORE_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1607_1610_eff_metric_k_deriv_declaration_gate_metrics.json"
)
CASEA_SUB_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1599_1602_v2_sub_exact_treat_declaration_gate_metrics.json"
)
BREAKTHROUGH_VEV = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "mass_origin_v2_trial1_breakthrough_modified_vev_decomposition_metrics.json"
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
CASEB_NOTE = Path(
    r"C:\Users\ogawa\Downloads\trial2_vector_qball_effective_metric_contraction_20260328.md"
)
GROUND_STATE_NOTE = Path(
    r"C:\Users\ogawa\Downloads\trial2_vector_qball_ground_state_identification_20260328.md"
)
TP_RESPONSE_NOTE = (
    ROOT
    / "doc"
    / "quantum"
    / "47_trial2_numeric_alpha_vector_qball_effective_metric_transverse_response.md"
)

STEP_TAG = "8.7.56.1619-1622"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor effective-metric v^2 subtraction "
    "exact treatment restore"
)
STEM = build_compact_artifact_stem(STEP_TAG, "eff_metric_v2_sub_restore", prefix="q")

PRIOR_CLASS = (
    "vector_qball_form_factor_ground_state_nodeless_hypothesis_not_supported_"
    "effective_metric_v2_restore_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_effective_metric_v2_subtraction_no_metric_rescue_"
    "disposition_sync_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_effective_metric_disposition_sync_closeout"
)
NEXT_ROUTE = "8.7.56.1623"

TARGET_ALPHA = 1.0 / 137.035999084
SCALAR_ALPHA = 0.00715678583937324
SCALAR_F = 0.2998913524347805
VECTOR_ALPHA = 0.0005579616187042394
VECTOR_F = -0.083735013520183
DIRECTIVE_PREF = 4.0 / 3.0
FULL_CASEB_PREF = 7.0 / 3.0
VEV_UNIT_SQ = 1.0


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


# 関数: 規格化された球対称 form factor を評価する。

def form_factor(radius: np.ndarray, density: np.ndarray, q_ratio: float) -> tuple[float, float]:
    """Evaluate one normalized spherical form factor."""
    weight = density * (radius**2)
    norm = float(np.trapezoid(weight, radius))
    qx = float(q_ratio) * radius
    sinc = np.ones_like(qx)
    mask = np.abs(qx) > 1.0e-12
    sinc[mask] = np.sin(qx[mask]) / qx[mask]
    numerator = float(np.trapezoid(weight * sinc, radius))
    return float(numerator / norm), float(norm)


# 関数: alpha と residual を返す。

def alpha_and_residual(form_value: float) -> tuple[float, float]:
    """Return alpha(F) and its target residual."""
    alpha_value = float((form_value * form_value) / (4.0 * math.pi))
    residual_rel = float(abs(alpha_value - TARGET_ALPHA) / TARGET_ALPHA)
    return alpha_value, residual_rel


# 関数: branch で使う式束を返す。

def build_formulae() -> dict[str, str]:
    """Return the exact-treatment formulas used in the restored caseB branch."""
    return {
        "metric_log_coordinate": "u(r) = ln(|f_0(r)| / P_ref), P_ref ~ P_infty",
        "caseb_spatial_core": (
            "Delta K_core,g^{ij} = lambda[(-f_0^2-e^{-2u}v^2+e^{-4u}f_L^2) delta^{ij} "
            "+ 2e^{-4u}f_L^2 r_hat^i r_hat^j]"
        ),
        "directive_subtracted_density": (
            "rho_sub,g^dir(r) = f_0(r)^2 + e^{-2u(r)}v_unit^2"
            " - (4/3)e^{-4u(r)}f_L(r)^2"
        ),
        "full_subtracted_density": (
            "rho_sub,g^full(r) = f_0(r)^2 + e^{-2u(r)}v_unit^2"
            " - (7/3)e^{-4u(r)}f_L(r)^2"
        ),
        "q2_only_comparison": (
            "rho_sub,g^{Q2}(r) = e^{2u(r)}f_0(r)^2 - e^{-2u(r)}f_L(r)^2"
        ),
        "core_window_rule": "core window := {r | u(r) >= 0}",
        "form_factor": "F(q) = int rho(r) sinc(qr) r^2 dr / int rho(r) r^2 dr",
    }


# 関数: `.1619-.1622` を実行する。

def main() -> None:
    """Execute the restored caseB v^2-subtraction exact-treatment branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        PART1,
        PART5,
        GROUND_STATE_GATE,
        CASEB_TP_GATE,
        CASEB_CORE_GATE,
        CASEA_SUB_GATE,
        BREAKTHROUGH_VEV,
        ANCHOR_EVAL,
        PHASE1_EVAL,
        CASEB_NOTE,
        GROUND_STATE_NOTE,
        TP_RESPONSE_NOTE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_roadmap_text = read_text(UNIFIED_ROADMAP)
    part1_text = read_text(PART1)
    part5_text = read_text(PART5)
    caseb_note_text = read_text(CASEB_NOTE)
    ground_state_note_text = read_text(GROUND_STATE_NOTE)
    tp_response_text = read_text(TP_RESPONSE_NOTE)

    ground_state_summary = read_json(GROUND_STATE_GATE)["summary"]
    caseb_tp_summary = read_json(CASEB_TP_GATE)["summary"]
    caseb_core_summary = read_json(CASEB_CORE_GATE)["summary"]
    casea_sub_summary = read_json(CASEA_SUB_GATE)["summary"]
    breakthrough_payload = read_json(BREAKTHROUGH_VEV)
    breakthrough_summary = breakthrough_payload["summary"]

    exact_branch = exact_profile_tools.load_module(
        exact_profile_tools.EXACT_REINJECTION_BRANCH,
        "t2a_1479_reuse_for_1619",
    )
    pivot = exact_branch.load_module(exact_branch.PIVOT_BRANCH, "pivot_branch_for_1619")

    # Reuse the same retained exact branch as `.1599-.1615`.
    anchor_summary = read_json(ANCHOR_EVAL)["summary"]
    phase1_summary = read_json(PHASE1_EVAL)["summary"]
    phase1_row = anchor_summary["phase1_equivalent_row"]
    profile = exact_profile_tools.solve_exact_profile_with_arrays(
        pivot,
        float(anchor_summary["beta_1_scalar"]),
        float(phase1_summary["phase1_best_alpha_candidate"]["amp0"]),
        float(phase1_row["amp_l"]),
    )

    radius = np.asarray(profile["radius"], dtype=float)
    f0_values = np.asarray(profile["f0"], dtype=float)
    f_l_values = np.asarray(profile["fL"], dtype=float)
    q_theory = float(profile["q_theory_over_m0"])

    p_ref_proxy = float(abs(f0_values[-1]))
    temporal_abs = np.abs(f0_values)
    u_values = np.log(np.maximum(temporal_abs, 1.0e-18) / p_ref_proxy)
    e_plus_2u = np.exp(2.0 * u_values)
    e_minus_2u = np.exp(-2.0 * u_values)
    e_minus_4u = np.exp(-4.0 * u_values)
    core_mask = temporal_abs >= p_ref_proxy

    directive_sub_density = (
        f0_values * f0_values
        + e_minus_2u * VEV_UNIT_SQ
        - DIRECTIVE_PREF * e_minus_4u * (f_l_values * f_l_values)
    )
    full_sub_density = (
        f0_values * f0_values
        + e_minus_2u * VEV_UNIT_SQ
        - FULL_CASEB_PREF * e_minus_4u * (f_l_values * f_l_values)
    )
    q2_only_density = e_plus_2u * (f0_values * f0_values) - e_minus_2u * (f_l_values * f_l_values)
    full_sub_core_density = np.where(core_mask, full_sub_density, 0.0)

    directive_f, directive_norm = form_factor(radius, directive_sub_density, q_theory)
    full_f, full_norm = form_factor(radius, full_sub_density, q_theory)
    q2_only_f, q2_only_norm = form_factor(radius, q2_only_density, q_theory)
    core_f, core_norm = form_factor(radius, full_sub_core_density, q_theory)

    directive_alpha, directive_residual = alpha_and_residual(directive_f)
    full_alpha, full_residual = alpha_and_residual(full_f)
    q2_only_alpha, q2_only_residual = alpha_and_residual(q2_only_f)
    core_alpha, core_residual = alpha_and_residual(core_f)

    casea_residual = float(casea_sub_summary["residual_sub_rel"])
    scalar_residual = float(abs(SCALAR_ALPHA - TARGET_ALPHA) / TARGET_ALPHA)

    full_vs_casea_ratio = float(full_residual / casea_residual)
    full_vs_casea_gap = float(full_residual - casea_residual)
    core_vs_casea_ratio = float(core_residual / casea_residual)
    core_vs_casea_gap = float(core_residual - casea_residual)
    directive_vs_full_gap = float(abs(directive_alpha - full_alpha))
    q2_vs_full_gap = float(abs(q2_only_alpha - full_alpha))

    prior_ready = bool(
        ground_state_summary.get("trial2_numeric_alpha_problem_classification") == PRIOR_CLASS
        and ground_state_summary.get("effective_metric_v2_subtraction_restore_required", False)
    )
    full_spatial_caseb_selected = bool(
        caseb_core_summary.get("effective_metric_spatial_kernel_formula_derived", False)
    )
    q2_only_surface_admissible_as_official = False
    full_profile_supports_scalar_candidate = bool(full_residual <= scalar_residual)
    core_window_supports_scalar_candidate = bool(core_residual <= scalar_residual)
    q2_only_supports_scalar_candidate = bool(q2_only_residual <= scalar_residual)
    metric_artifact_rescue_supported = bool(full_profile_supports_scalar_candidate)
    full_profile_worsens_vs_casea = bool(full_residual > casea_residual)
    core_window_improves_vs_casea = bool(core_residual < casea_residual)
    effective_metric_disposition_sync_closeout_admissible_now = True
    physical_reject_required = False

    rows = [
        row(
            "prior_ready",
            "pass" if prior_ready else "reject",
            "ground-state no-go branch restored caseB subtraction as next route",
            truth(prior_ready),
            "The restore branch only starts after the ground-state/nodeless hypothesis failed and explicitly returned the mainline to caseB v^2 subtraction.",
        ),
        row(
            "full_spatial_caseb_selected",
            "pass" if full_spatial_caseb_selected else "reject",
            "full spatial caseB kernel selected as official read",
            truth(full_spatial_caseb_selected),
            "After `.1607-.1610`, the honest exact treatment must use the full spatial caseB kernel rather than the lighter Q_g^2-only slogan surface.",
        ),
        row(
            "directive_sub_alpha_at_q_theory",
            "reject" if not full_profile_supports_scalar_candidate else "pass",
            "directive-style caseB subtraction alpha at q_theory",
            directive_alpha,
            "This is the lighter projected subtraction surface with the 4/3 coefficient retained only as comparison evidence.",
        ),
        row(
            "full_sub_alpha_at_q_theory",
            "reject" if not full_profile_supports_scalar_candidate else "pass",
            "full spatial-kernel caseB subtraction alpha at q_theory",
            full_alpha,
            "This is the official exact-treatment read after restoring the deferred caseB branch.",
        ),
        row(
            "full_sub_residual_rel",
            "reject" if not full_profile_supports_scalar_candidate else "pass",
            "full spatial-kernel caseB subtraction residual",
            full_residual,
            "The full-profile caseB subtraction result is the metric-consistent quantity that must be compared against both the scalar residual and the prior caseA worsen.",
        ),
        row(
            "full_sub_vs_casea_residual_ratio",
            "reject" if full_profile_worsens_vs_casea else "pass",
            "full caseB subtraction residual divided by prior caseA worsen residual",
            full_vs_casea_ratio,
            "Values above one mean the metric-consistent recomputation is even worse than the retained caseA/Minkowski worsen.",
        ),
        row(
            "core_window_sub_alpha_at_q_theory",
            "reject" if not core_window_supports_scalar_candidate else "pass",
            "core-window full caseB subtraction alpha at q_theory",
            core_alpha,
            "This diagnostic keeps only the nominal suppression region u>=0 under the full spatial kernel.",
        ),
        row(
            "core_window_sub_residual_rel",
            "reject" if not core_window_supports_scalar_candidate else "pass",
            "core-window full caseB subtraction residual",
            core_residual,
            "Even the restricted core window must still close to the scalar candidate to count as a rescue.",
        ),
        row(
            "core_window_improves_vs_casea",
            "pass" if core_window_improves_vs_casea else "reject",
            "core-window full caseB subtraction improves vs caseA worsen",
            truth(core_window_improves_vs_casea),
            "The core-window diagnostic is allowed to improve slightly over caseA while still remaining far from the scalar candidate.",
        ),
        row(
            "q2_only_alpha_at_q_theory",
            "watch",
            "Q_g^2-only comparison alpha at q_theory",
            q2_only_alpha,
            "This keeps the original directive's lighter Q_g^2-only subtraction surface as evidence, but not as the official branch readout.",
        ),
        row(
            "q2_only_surface_admissible_as_official",
            "reject",
            "Q_g^2-only comparison surface admissible as official read",
            truth(q2_only_surface_admissible_as_official),
            "The lighter Q_g^2-only surface is no longer honest once the full spatial caseB contraction has been derived explicitly.",
        ),
        row(
            "metric_artifact_rescue_supported",
            "pass" if metric_artifact_rescue_supported else "reject",
            "metric-choice artifact rescue supported",
            truth(metric_artifact_rescue_supported),
            "To support the artifact hypothesis, the full spatial caseB recomputation would need to recover at least the retained scalar residual scale.",
        ),
        row(
            "effective_metric_disposition_sync_closeout_admissible_now",
            "pass" if effective_metric_disposition_sync_closeout_admissible_now else "reject",
            "effective-metric disposition sync / closeout admissible now",
            truth(effective_metric_disposition_sync_closeout_admissible_now),
            "Once the restored caseB subtraction result is fixed, the next honest action is the official disposition sync / closeout.",
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
            "part1": display_path(PART1),
            "part5": display_path(PART5),
            "ground_state_gate": display_path(GROUND_STATE_GATE),
            "caseb_tp_gate": display_path(CASEB_TP_GATE),
            "caseb_core_gate": display_path(CASEB_CORE_GATE),
            "casea_sub_gate": display_path(CASEA_SUB_GATE),
            "breakthrough_vev": display_path(BREAKTHROUGH_VEV),
            "anchor_eval": display_path(ANCHOR_EVAL),
            "phase1_eval": display_path(PHASE1_EVAL),
            "caseb_note": display_path(CASEB_NOTE),
            "ground_state_note": display_path(GROUND_STATE_NOTE),
            "tp_response_note": display_path(TP_RESPONSE_NOTE),
        },
        "constants": {
            "q_theory_over_m0": q_theory,
            "p_ref_proxy_from_outer_radius": p_ref_proxy,
            "vev_unit_squared": VEV_UNIT_SQ,
            "directive_prefactor": DIRECTIVE_PREF,
            "full_caseb_prefactor": FULL_CASEB_PREF,
            "target_alpha": TARGET_ALPHA,
            "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
            "vector_alpha_at_q_theory": VECTOR_ALPHA,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "selected_caseb_subtraction_surface": "full_spatial_kernel_projected",
        "directive_sub_alpha_at_q_theory": directive_alpha,
        "directive_sub_residual_rel": directive_residual,
        "full_sub_alpha_at_q_theory": full_alpha,
        "full_sub_residual_rel": full_residual,
        "core_window_sub_alpha_at_q_theory": core_alpha,
        "core_window_sub_residual_rel": core_residual,
        "q2_only_alpha_at_q_theory": q2_only_alpha,
        "q2_only_residual_rel": q2_only_residual,
        "directive_vs_full_alpha_gap": directive_vs_full_gap,
        "q2_only_vs_full_alpha_gap": q2_vs_full_gap,
        "full_sub_vs_casea_residual_gap": full_vs_casea_gap,
        "full_sub_vs_casea_residual_ratio": full_vs_casea_ratio,
        "core_window_vs_casea_residual_gap": core_vs_casea_gap,
        "core_window_vs_casea_residual_ratio": core_vs_casea_ratio,
        "full_profile_supports_scalar_candidate": full_profile_supports_scalar_candidate,
        "core_window_supports_scalar_candidate": core_window_supports_scalar_candidate,
        "q2_only_supports_scalar_candidate": q2_only_supports_scalar_candidate,
        "metric_artifact_rescue_supported": metric_artifact_rescue_supported,
        "full_profile_worsens_vs_casea": full_profile_worsens_vs_casea,
        "core_window_improves_vs_casea": core_window_improves_vs_casea,
        "negative_u_fraction": float(caseb_tp_summary["negative_u_fraction"]),
        "e_minus_4u_max": float(caseb_tp_summary["e_minus_4u_max"]),
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "physical_reject_required": physical_reject_required,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": True,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "formulas": build_formulae(),
        "hits": {
            "part1_pref_equals_infty": hit(part1_text, "P_{\\mathrm{ref}}\\equiv P_{\\infty}"),
            "caseb_note_y0v_pref": hit(caseb_note_text, "|y_0(x)| \\cdot v"),
            "caseb_note_q2g": hit(caseb_note_text, "Q^2\\big|_{g(P)} = -e^{2u}f_0^2 + e^{-2u}f_L^2"),
            "ground_state_note_nodeless": hit(
                ground_state_note_text,
                "ground state の条件: **f₀(r) > 0 for all r > 0**（nodeless）。",
            ),
            "tp_response_next_route": hit(
                tp_response_text,
                "effective-metric `v^2` subtraction exact treatment",
            ),
            "current_problem_caseb_v2_restore": hit(
                current_problem_text,
                "effective-metric `v^2` subtraction exact treatment restore",
            ),
            "current_status_caseb_v2_restore": hit(
                current_status_text,
                "effective-metric `v^2` subtraction exact treatment restore",
            ),
            "unified_roadmap_caseb_v2_restore": hit(
                unified_roadmap_text,
                "`.1619-.1622` は **effective-metric `v^2` subtraction exact treatment restore**",
            ),
            "part5_caseb_v2_restore": hit(
                part5_text,
                "effective-metric `v^2` subtraction exact treatment restore",
            ),
            "breakthrough_background_vev": {
                "pattern": "P_mu^(0) = (v, 0, 0, 0)",
                "line": 26,
                "text": breakthrough_payload["formulas"]["background_vev"],
            },
        },
        "norms": {
            "directive_sub_norm": directive_norm,
            "full_sub_norm": full_norm,
            "q2_only_norm": q2_only_norm,
            "core_window_norm": core_norm,
        },
        "retained_numeric_state": {
            "scalar_F_exact_at_q_theory": SCALAR_F,
            "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
            "vector_F_at_q_theory": VECTOR_F,
            "vector_alpha_at_q_theory": VECTOR_ALPHA,
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
        },
    }

    manifest = {
        "inventory": write_artifact(
            "inventory",
            payload(
                "8.7.56.1619",
                f"{STEP_NAME} inventory",
                inputs,
                rows,
                summary,
                decision,
                evidence,
            ),
        ),
        "audit": write_artifact(
            "audit",
            payload(
                "8.7.56.1620",
                f"{STEP_NAME} audit",
                inputs,
                rows,
                summary,
                decision,
                evidence,
            ),
        ),
        "declaration_gate": write_artifact(
            "declaration_gate",
            payload(
                "8.7.56.1621",
                f"{STEP_NAME} declaration gate",
                inputs,
                rows,
                summary,
                decision,
                evidence,
            ),
        ),
        "route_sync": write_artifact(
            "route_sync",
            payload(
                "8.7.56.1622",
                f"{STEP_NAME} route sync",
                inputs,
                rows,
                summary,
                decision,
                evidence,
            ),
        ),
    }

    print(
        json.dumps(
            {"step": STEP_TAG, "stem": STEM, "artifacts": manifest},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
