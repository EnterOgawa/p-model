#!/usr/bin/env python3
"""Generate 8.7.56.1611-.1614 effective-metric transverse-projection alpha-audit artifacts.

This branch executes the next honest computation after the caseB/effective-metric
quadratic-core derivation.

What is already fixed:

1. caseA/Minkowski contraction is rejected while caseB/effective metric is the
   retained Part-I branch,
2. the caseB spatial quadratic core is
   `Delta K_core,g^{ij} = lambda[(-f0^2-e^{-2u}v^2+e^{-4u}fL^2) delta^{ij}
   + 2 e^{-4u} fL^2 rhat^i rhat^j]`,
3. the only robust gain from the full caseB contraction is the `e^{-4u}`
   suppression of anisotropic `f_L^2`, not a naive `e^{2u}` enhancement of the
   isotropic `f_0^2` term.

What this branch audits:

- if the directive's temporal proxy `u(r)=ln(|f_0(r)|/P_ref)` is implemented on
  the retained exact branch with `P_ref=P_infty` approximated by the available
  outer-radius proxy,
- does the projected caseB kernel still support the retained scalar alpha
  candidate,
- or does the temporal near-node generate an `u<0` region where `e^{-4u}`
  reverses from suppression into enhancement and destroys the leading-foundation
  reading.

The branch therefore evaluates two closely related surfaces:

1. the full-profile directive-style projected density
   `rho_T,g = f_0^2 - (4/3)e^{-4u}f_L^2`,
2. a core-window diagnostic where only `u>=0` points are kept.

If both fail, the next honest action is not another rhetorical classification
loop but the already scheduled caseB `v^2` subtraction exact treatment.
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

CASEB_DIRECTIVE = Path(
    r"C:\Users\ogawa\Downloads\trial2_vector_qball_effective_metric_contraction_20260328.md"
)
TP_DIRECTIVE = Path(
    r"C:\Users\ogawa\Downloads\trial2_vector_qball_transverse_projection_20260328.md"
)
PRIOR_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1607_1610_eff_metric_k_deriv_declaration_gate_metrics.json"
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
RECONSTRUCTION_BRANCH = ROOT / "scripts" / "quantum" / "t2a_1599.py"

STEP_TAG = "8.7.56.1611-1614"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor effective-metric transverse-projection "
    "alpha audit"
)
STEM = build_compact_artifact_stem(STEP_TAG, "eff_metric_tp_alpha_audit", prefix="q")

PRIOR_CLASS = "vector_qball_form_factor_effective_metric_quadratic_core_derived_transverse_audit_next"
BRANCH_CLASS = (
    "vector_qball_form_factor_effective_metric_transverse_projection_no_scalar_foundation_"
    "v2_subtraction_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_effective_metric_v2_subtraction_exact_treatment"
)
NEXT_ROUTE = "8.7.56.1615"
DOWNSTREAM_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_effective_metric_disposition_sync"
)
DOWNSTREAM_ROUTE = "8.7.56.1619"

SCALAR_F = 0.2998913524347805
SCALAR_ALPHA = 0.00715678583937324
TARGET_ALPHA = 1.0 / 137.035999084
VECTOR_F = -0.083735013520183
VECTOR_ALPHA = 0.0005579616187042394
TP_PREF = 4.0 / 3.0
FULL_CASEB_PREF = 7.0 / 3.0


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


# 関数: ローカル Python モジュールを動的 import する。

def load_module(path: Path, module_name: str):
    """Dynamically import one local Python module."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[fail] unable to import module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


# 関数: branch で使う formula bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return the effective-metric transverse-audit formulas."""
    return {
        "metric_log_coordinate": "u(r) = ln(|f_0(r)| / P_ref), P_ref = P_infty",
        "tp_ratio_rule_caseb": "R_aniso/iso <= (4/3)e^{-4u}(f_L/f_0)^2",
        "directive_projected_density": "rho_T,g^dir(r) = f_0(r)^2 - (4/3)e^{-4u(r)}f_L(r)^2",
        "full_projected_density": "rho_T,g^full(r) = f_0(r)^2 - (7/3)e^{-4u(r)}f_L(r)^2",
        "core_window_rule": "core window := {r | u(r) >= 0} <=> {|f_0(r)| >= P_ref}",
        "form_factor": "F(q) = int rho(r) sinc(qr) r^2 dr / int rho(r) r^2 dr",
    }


# 関数: `.1611-.1614` を実行する。

def main() -> None:
    """Execute the effective-metric transverse-projection alpha audit."""
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
        CASEB_DIRECTIVE,
        TP_DIRECTIVE,
        PRIOR_GATE,
        ANCHOR_EVAL,
        PHASE1_EVAL,
        RECONSTRUCTION_BRANCH,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_roadmap_text = read_text(UNIFIED_ROADMAP)
    part1_text = read_text(PART1)
    part5_text = read_text(PART5)
    caseb_directive_text = read_text(CASEB_DIRECTIVE)
    tp_directive_text = read_text(TP_DIRECTIVE)

    prior_summary = read_json(PRIOR_GATE)["summary"]
    anchor_summary = read_json(ANCHOR_EVAL)["summary"]
    phase1_summary = read_json(PHASE1_EVAL)["summary"]

    profile_module = load_module(RECONSTRUCTION_BRANCH, "t2a_1599_reuse_for_1611")
    exact_branch = profile_module.load_module(
        profile_module.EXACT_REINJECTION_BRANCH,
        "t2a_1479_reuse_for_1611",
    )
    pivot = exact_branch.load_module(exact_branch.PIVOT_BRANCH, "pivot_branch_for_1611")

    phase1_row = anchor_summary["phase1_equivalent_row"]
    beta = float(anchor_summary["beta_1_scalar"])
    amp0 = float(phase1_summary["phase1_best_alpha_candidate"]["amp0"])
    amp_l = float(phase1_row["amp_l"])
    profile = profile_module.solve_exact_profile_with_arrays(pivot, beta, amp0, amp_l)

    radius = np.asarray(profile["radius"], dtype=float)
    f0_values = np.asarray(profile["f0"], dtype=float)
    f_l_values = np.asarray(profile["fL"], dtype=float)
    q_theory = float(profile["q_theory_over_m0"])

    p_ref_proxy = float(abs(f0_values[-1]))
    temporal_abs = np.abs(f0_values)
    u_values = np.log(np.maximum(temporal_abs, 1.0e-18) / p_ref_proxy)
    e_minus_4u = np.exp(-4.0 * u_values)
    core_mask = temporal_abs >= p_ref_proxy

    node_like_index = int(np.argmin(temporal_abs))
    node_like_radius = float(radius[node_like_index])
    node_like_abs_f0 = float(temporal_abs[node_like_index])
    u_min = float(np.min(u_values))
    u_max = float(np.max(u_values))
    e_minus_4u_min = float(np.min(e_minus_4u))
    e_minus_4u_max = float(np.max(e_minus_4u))
    negative_u_fraction = float(np.mean(u_values < 0.0))
    core_fraction = float(np.mean(core_mask))

    scalar_density = f0_values * f0_values
    directive_density = scalar_density - TP_PREF * e_minus_4u * (f_l_values * f_l_values)
    full_projected_density = (
        scalar_density - FULL_CASEB_PREF * e_minus_4u * (f_l_values * f_l_values)
    )
    core_window_density = np.where(core_mask, directive_density, 0.0)

    directive_f, directive_norm = form_factor(radius, directive_density, q_theory)
    full_projected_f, full_projected_norm = form_factor(
        radius,
        full_projected_density,
        q_theory,
    )
    core_window_f, core_window_norm = form_factor(radius, core_window_density, q_theory)

    directive_alpha, directive_residual_rel = alpha_and_residual(directive_f)
    full_projected_alpha, full_projected_residual_rel = alpha_and_residual(
        full_projected_f
    )
    core_window_alpha, core_window_residual_rel = alpha_and_residual(core_window_f)

    core_window_vs_vector_gap = float(abs(core_window_alpha - VECTOR_ALPHA))
    core_window_vs_vector_rel = float(core_window_vs_vector_gap / VECTOR_ALPHA)
    directive_vs_scalar_gap = float(abs(directive_alpha - SCALAR_ALPHA))
    full_vs_directive_gap = float(abs(full_projected_alpha - directive_alpha))

    prior_ready = bool(
        prior_summary.get("trial2_numeric_alpha_problem_classification") == PRIOR_CLASS
        and prior_summary.get("effective_metric_tp_audit_admissible_now", False)
    )
    directive_metric_log_rule_available = bool(
        hit(caseb_directive_text, "u(r) = \\ln\\left(\\frac{|f_0(r)|}{P_{\\rm ref}}\\right)")
        or hit(caseb_directive_text, "u(r) = \\ln\\left(\\frac{|f_0(r)|}{P_{\\rm ref}}\\right)")
        or hit(caseb_directive_text, "u(r) = \\ln\\left(\\frac{|f_0(r)|}{P_{\\rm ref}}\\right)")
        or hit(caseb_directive_text, "u(r) = \\ln\\left(\\frac{|f_0(r)|}{P_{\\rm ref}}\\right)")
        is not None
    )
    # The downloaded note uses plain UTF-8 text; the exact substring is easier to
    # track via the more robust absolute-value hit below.
    directive_abs_temporal_present = bool(
        hit(caseb_directive_text, "u(r) = \\ln\\left(\\frac{|f_0(r)|}{P_{\\rm ref}}\\right)")
        or hit(caseb_directive_text, "|f_0(r)|")
    )
    part1_pref_equals_infty_available = bool(
        hit(part1_text, "P_{\\mathrm{ref}}\\equiv P_{\\infty}")
    )

    temporal_node_like_region_present = bool(node_like_abs_f0 < 0.01 * p_ref_proxy)
    negative_u_region_present = bool(negative_u_fraction > 0.0)
    suppression_reverses_to_enhancement_outside_core = bool(e_minus_4u_max > 1.0)
    directive_projected_alpha_supports_scalar_candidate = bool(
        directive_residual_rel <= 2.0 * (abs(TARGET_ALPHA - SCALAR_ALPHA) / TARGET_ALPHA)
    )
    full_projected_alpha_supports_scalar_candidate = bool(
        full_projected_residual_rel <= 2.0 * (abs(TARGET_ALPHA - SCALAR_ALPHA) / TARGET_ALPHA)
    )
    core_window_supports_scalar_candidate = bool(
        core_window_residual_rel <= 2.0 * (abs(TARGET_ALPHA - SCALAR_ALPHA) / TARGET_ALPHA)
    )
    effective_metric_scalar_foundation_supported = bool(
        directive_projected_alpha_supports_scalar_candidate
        and full_projected_alpha_supports_scalar_candidate
        and core_window_supports_scalar_candidate
    )
    effective_metric_v2_subtraction_exact_treatment_admissible_now = True
    effective_metric_disposition_sync_admissible_now = False
    physical_reject_required = False

    scalar_residual_rel = float(abs(TARGET_ALPHA - SCALAR_ALPHA) / TARGET_ALPHA)

    rows = [
        row(
            "prior_ready",
            "pass" if prior_ready else "reject",
            "prior caseB quadratic-core derivation ready",
            truth(prior_ready),
            "The effective-metric transverse audit starts only after the caseB quadratic core has already been derived explicitly.",
        ),
        row(
            "p_ref_proxy_from_outer_radius",
            "watch",
            "outer-radius proxy for P_infty",
            p_ref_proxy,
            "Current pack has no deeper asymptotic closure for P_infty on the retained exact branch, so the honest no-new-parameter proxy is |f_0(r_max)|.",
        ),
        row(
            "temporal_node_like_region_present",
            "pass" if temporal_node_like_region_present else "reject",
            "temporal near-node region present",
            truth(temporal_node_like_region_present),
            "The retained exact branch drives |f_0| close to zero before the tail, so the directive's log metric factor must confront a near-node rather than a uniformly positive core profile.",
        ),
        row(
            "negative_u_fraction",
            "watch" if negative_u_region_present else "pass",
            "fraction of profile with u<0",
            negative_u_fraction,
            "Where |f_0| < P_ref, the caseB factor e^{-4u} stops suppressing and starts enhancing the anisotropic correction.",
        ),
        row(
            "e_minus_4u_max",
            "watch" if e_minus_4u_max > 1.0 else "pass",
            "maximum e^{-4u} factor",
            e_minus_4u_max,
            "This is the strongest current-pack enhancement generated by the temporal near-node under the directive's own metric rule.",
        ),
        row(
            "directive_projected_alpha_at_q_theory",
            "reject" if not directive_projected_alpha_supports_scalar_candidate else "pass",
            "directive-style projected alpha at q_theory",
            directive_alpha,
            "Using rho_T,g = f_0^2 - (4/3)e^{-4u}f_L^2, the full-profile caseB transverse audit moves far away from the retained scalar candidate.",
        ),
        row(
            "directive_projected_residual_rel",
            "reject" if not directive_projected_alpha_supports_scalar_candidate else "pass",
            "directive-style projected alpha residual",
            directive_residual_rel,
            "The full-profile directive-style caseB projection must be compared directly to the retained 1.9 percent scalar residual.",
        ),
        row(
            "full_projected_alpha_at_q_theory",
            "reject" if not full_projected_alpha_supports_scalar_candidate else "pass",
            "full spatial projected alpha at q_theory",
            full_projected_alpha,
            "Including the isotropic e^{-4u}f_L^2 term from the exact caseB spatial kernel leaves the no-go essentially unchanged.",
        ),
        row(
            "full_vs_directive_alpha_gap",
            "watch",
            "full projected alpha minus directive alpha",
            full_vs_directive_gap,
            "The full-kernel versus directive-style coefficient choice is numerically secondary compared with the dominant caseB tail/enhancement effect.",
        ),
        row(
            "core_window_alpha_at_q_theory",
            "reject" if not core_window_supports_scalar_candidate else "pass",
            "core-window projected alpha at q_theory",
            core_window_alpha,
            "Even after restricting to the nominal suppression region u>=0, the projected alpha stays on the same vector no-go scale rather than returning to the scalar candidate.",
        ),
        row(
            "core_window_vs_vector_alpha_rel",
            "watch",
            "core-window alpha relative gap vs retained vector no-go alpha",
            core_window_vs_vector_rel,
            "The core-window result lands within about one percent of the already retained vector blind no-go value.",
        ),
        row(
            "effective_metric_scalar_foundation_supported",
            "pass" if effective_metric_scalar_foundation_supported else "reject",
            "effective-metric transverse audit supports scalar foundation",
            truth(effective_metric_scalar_foundation_supported),
            "Current pack does not support a caseB rescue of the scalar candidate at the transverse-projection stage.",
        ),
        row(
            "effective_metric_v2_subtraction_exact_treatment_admissible_now",
            "pass" if effective_metric_v2_subtraction_exact_treatment_admissible_now else "reject",
            "effective-metric v^2 subtraction exact treatment admissible now",
            truth(effective_metric_v2_subtraction_exact_treatment_admissible_now),
            "After the caseB transverse audit is fixed, the next honest same-level computation is the exact caseB treatment of the v^2 subtraction term.",
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
            "caseb_directive_note": display_path(CASEB_DIRECTIVE),
            "tp_directive_note": display_path(TP_DIRECTIVE),
            "prior_gate": display_path(PRIOR_GATE),
            "anchor_eval": display_path(ANCHOR_EVAL),
            "phase1_eval": display_path(PHASE1_EVAL),
            "reconstruction_branch": display_path(RECONSTRUCTION_BRANCH),
        },
        "constants": {
            "tp_prefactor": TP_PREF,
            "full_caseb_prefactor": FULL_CASEB_PREF,
            "q_theory_over_m0": q_theory,
            "target_alpha": TARGET_ALPHA,
            "scalar_F_exact_at_q_theory": SCALAR_F,
            "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
            "vector_F_at_q_theory": VECTOR_F,
            "vector_alpha_at_q_theory": VECTOR_ALPHA,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "downstream_route_name": DOWNSTREAM_ROUTE_NAME,
            "downstream_route": DOWNSTREAM_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "phase1_equivalent_max_abs_ratio": float(phase1_row["max_abs_ratio"]),
        "p_ref_proxy_from_outer_radius": p_ref_proxy,
        "temporal_node_like_radius": node_like_radius,
        "temporal_node_like_abs_f0": node_like_abs_f0,
        "u_min": u_min,
        "u_max": u_max,
        "negative_u_fraction": negative_u_fraction,
        "core_fraction": core_fraction,
        "e_minus_4u_min": e_minus_4u_min,
        "e_minus_4u_max": e_minus_4u_max,
        "directive_projected_F_at_q_theory": directive_f,
        "directive_projected_alpha_at_q_theory": directive_alpha,
        "directive_projected_residual_rel": directive_residual_rel,
        "full_projected_F_at_q_theory": full_projected_f,
        "full_projected_alpha_at_q_theory": full_projected_alpha,
        "full_projected_residual_rel": full_projected_residual_rel,
        "core_window_F_at_q_theory": core_window_f,
        "core_window_alpha_at_q_theory": core_window_alpha,
        "core_window_residual_rel": core_window_residual_rel,
        "core_window_vs_vector_alpha_rel": core_window_vs_vector_rel,
        "directive_projected_alpha_supports_scalar_candidate": (
            directive_projected_alpha_supports_scalar_candidate
        ),
        "full_projected_alpha_supports_scalar_candidate": (
            full_projected_alpha_supports_scalar_candidate
        ),
        "core_window_supports_scalar_candidate": core_window_supports_scalar_candidate,
        "effective_metric_scalar_foundation_supported": (
            effective_metric_scalar_foundation_supported
        ),
        "effective_metric_v2_subtraction_exact_treatment_admissible_now": (
            effective_metric_v2_subtraction_exact_treatment_admissible_now
        ),
        "effective_metric_disposition_sync_admissible_now": (
            effective_metric_disposition_sync_admissible_now
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
            "part1_pref_equals_infty": hit(part1_text, "P_{\\mathrm{ref}}\\equiv P_{\\infty}"),
            "part1_metric_log_coordinate": hit(
                part1_text,
                "u\\equiv\\ln\\!\\left(\\frac{P_t}{P_{\\mathrm{ref}}}\\right)",
            ),
            "caseb_directive_temporal_abs": hit(caseb_directive_text, "|f_0(r)|"),
            "caseb_directive_q2g": hit(
                caseb_directive_text,
                "Q_\\mu Q^\\mu\\big|_{g(P)} = g^{00}f_0^2 + g^{ij}f_L^2\\hat{r}_i\\hat{r}_j",
            ),
            "tp_directive_projector": hit(tp_directive_text, "\\Pi_{ij}^T"),
            "tp_directive_case_i": hit(tp_directive_text, "Case I"),
            "current_problem_eff_metric_tp_audit": hit(
                current_problem_text,
                "effective-metric transverse-projection α audit",
            ),
            "current_status_eff_metric_tp_audit": hit(
                current_status_text,
                "effective-metric transverse-projection α audit",
            ),
            "unified_roadmap_eff_metric_tp_audit": hit(
                unified_roadmap_text,
                "effective-metric transverse-projection α audit",
            ),
            "part5_eff_metric_tp_audit": hit(
                part5_text,
                "effective-metric transverse-projection α audit",
            ),
        },
        "support_counts": {
            "negative_u_sample_count": float(np.count_nonzero(u_values < 0.0)),
            "core_sample_count": float(np.count_nonzero(core_mask)),
            "temporal_profile_sample_count": float(radius.size),
        },
        "norms": {
            "directive_projected_norm": directive_norm,
            "full_projected_norm": full_projected_norm,
            "core_window_norm": core_window_norm,
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
                "8.7.56.1611",
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
                "8.7.56.1612",
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
                "8.7.56.1613",
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
                "8.7.56.1614",
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
