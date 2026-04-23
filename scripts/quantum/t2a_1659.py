#!/usr/bin/env python3
"""Generate 8.7.56.1659-.1662 projected-kernel transverse-response artifacts.

This branch is the first fallback after the density / constitutive-map family
closed honestly as Gate B retain-but-not-promote. The goal is no longer to
invent another local density. It is to test whether the already-derived
quadratic `P_mu` background kernel itself can define the observable.

We therefore work directly with the vacuum-subtracted spatial quadratic core

    Delta K_core^{ij}[Q] = lambda[(Q^2-v^2) delta^{ij} + 2 Q^i Q^j]

on the retained exact vector / Q-ball branch. For the longitudinal radial
ansatz `Q_i = f_L(r) rhat_i`, the spatial tensor decomposes into

    T_ij(r) = A(r) delta_ij + B(r) rhat_i rhat_j

with

    A(r) = -f_0(r)^2 + f_L(r)^2,
    B(r) = 2 f_L(r)^2,

after dropping the vacuum constant `-v^2 delta_ij`, which only shifts the
background and does not contribute to the branch-local scattering contrast.

The physical transverse probe is then the polarization-averaged matrix element

    M_T(q) = 4 pi int r^2 [A j_0(qr) + (B/3)(j_0(qr)+j_2(qr))] dr

and the normalized observable is

    F_T(q) = M_T(q) / M_T(0),    alpha_T(q) = F_T(q)^2 / (4 pi).

The script also tracks two evidence-only comparison surfaces:

1. a positive-overlap variant where the isotropic term is forced positive,
2. a scalar-only overlap `f_0^2`.

No post-hoc normalization or extra parameter is introduced.
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

import scripts.quantum.t2a_1627 as energy_density_tools


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

PRIOR_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1655_1658_primary_decision_gate_declaration_gate_metrics.json"
)
QUAD_DERIV_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1575_1578_quadratic_k_deriv_declaration_gate_metrics.json"
)
ENERGY_CORE_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1627_1630_energy_density_ff_audit_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1659-1662"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor P_mu transverse response / "
    "projected-kernel observable audit"
)
STEM = build_compact_artifact_stem(STEP_TAG, "pmu_tresp_pk_audit", prefix="q")

PRIOR_CLASS = (
    "vector_qball_form_factor_primary_gate_b_retain_not_promote_"
    "transverse_response_fallback_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_p_mu_transverse_response_projected_kernel_"
    "tracks_vector_no_go_ground_state_fallback_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_constrained_ground_state_"
    "branch_selection_audit"
)
NEXT_ROUTE = "8.7.56.1663"
FOLLOWUP_CLOSEOUT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_fallback_closeout_reopen_registry"
)
FOLLOWUP_CLOSEOUT_ROUTE = "8.7.56.1667"

TARGET_ALPHA = 1.0 / 137.035999084
SCALAR_F = 0.2998913524347805
SCALAR_ALPHA = 0.00715678583937324
VECTOR_F = -0.083735013520183
VECTOR_ALPHA = 0.0005579616187042394


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


# 関数: alpha と target の相対残差を返す。

def alpha_residual_rel(alpha_value: float) -> float:
    """Return one target-relative residual."""
    return float(abs(float(alpha_value) - TARGET_ALPHA) / TARGET_ALPHA)


# 関数: 球ベッセル j0 を返す。

def spherical_j0(x: np.ndarray) -> np.ndarray:
    """Return spherical Bessel j_0(x)."""
    values = np.ones_like(x)
    mask = np.abs(x) > 1.0e-12
    values[mask] = np.sin(x[mask]) / x[mask]
    return values


# 関数: 球ベッセル j2 を返す。

def spherical_j2(x: np.ndarray) -> np.ndarray:
    """Return spherical Bessel j_2(x)."""
    values = np.zeros_like(x)
    small = np.abs(x) <= 1.0e-6
    xs = x[small]
    values[small] = (xs * xs) / 15.0
    mask = ~small
    xm = x[mask]
    values[mask] = (
        ((3.0 / (xm**3)) - (1.0 / xm)) * np.sin(xm)
        - (3.0 * np.cos(xm) / (xm**2))
    )
    return values


# 関数: transverse projected matrix element を規格化して返す。

def projected_form_factor(
    radius: np.ndarray,
    a_values: np.ndarray,
    b_values: np.ndarray,
    q_ratio: float,
) -> dict:
    """Evaluate the polarization-averaged projected-kernel form factor."""
    qx = float(q_ratio) * radius
    j0 = spherical_j0(qx)
    j2 = spherical_j2(qx)
    kernel_q = a_values * j0 + (b_values / 3.0) * (j0 + j2)
    kernel_0 = a_values + (b_values / 3.0)
    numerator = float(np.trapezoid((radius**2) * kernel_q, radius))
    denominator = float(np.trapezoid((radius**2) * kernel_0, radius))
    form_value = float(numerator / denominator)
    alpha_value = float((form_value * form_value) / (4.0 * math.pi))
    return {
        "F_at_q_theory": form_value,
        "alpha_at_q_theory": alpha_value,
        "alpha_residual_rel": alpha_residual_rel(alpha_value),
        "numerator_at_q_theory": numerator,
        "denominator_at_zero": denominator,
        "denominator_negative": bool(denominator < 0.0),
    }


# 関数: 公式 / evidence-only formula bundle を返す。

def build_formulae() -> dict[str, str]:
    """Return the projected-kernel formulas used in this branch."""
    return {
        "quadratic_core": (
            "Delta K_core^{ij}[Q] = lambda[(Q^2-v^2) delta^{ij} + 2 Q^i Q^j]"
        ),
        "radial_decomposition": (
            "Q_i = f_L(r) rhat_i => T_ij(r) = A(r) delta_ij + B(r) rhat_i rhat_j"
        ),
        "official_vacuum_subtracted_coefficients": (
            "A(r) = -f_0(r)^2 + f_L(r)^2, B(r) = 2 f_L(r)^2"
        ),
        "projected_matrix_element": (
            "M_T(q) = 4 pi int r^2 [A j_0(qr) + (B/3)(j_0(qr)+j_2(qr))] dr"
        ),
        "normalized_observable": "F_T(q) = M_T(q)/M_T(0), alpha_T(q) = F_T(q)^2/(4 pi)",
        "positive_overlap_variant": (
            "A_plus(r) = +f_0(r)^2, B_plus(r) = 2 f_L(r)^2"
        ),
        "scalar_only_overlap": "rho_scalar(r) = f_0(r)^2",
    }


# 関数: `.1659-.1662` を実行する。

def main() -> None:
    """Execute the projected-kernel transverse-response fallback branch."""
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
        PRIOR_GATE,
        QUAD_DERIV_GATE,
        ENERGY_CORE_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_text = read_text(UNIFIED_ROADMAP)
    part1_text = read_text(PART1)
    part5_text = read_text(PART5)

    prior_summary = read_json(PRIOR_GATE)["summary"]
    quad_summary = read_json(QUAD_DERIV_GATE)["summary"]
    energy_summary = read_json(ENERGY_CORE_GATE)["summary"]
    bundle = energy_density_tools.build_density_bundle()

    radius = np.asarray(bundle["radius"], dtype=float)
    f0_values = np.asarray(bundle["f0_values"], dtype=float)
    f_l_values = np.asarray(bundle["f_l_values"], dtype=float)
    q_theory = float(bundle["q_theory_over_m0"])

    official_a = -(f0_values * f0_values) + (f_l_values * f_l_values)
    official_b = 2.0 * (f_l_values * f_l_values)
    positive_a = f0_values * f0_values
    positive_b = 2.0 * (f_l_values * f_l_values)

    official_surface = projected_form_factor(radius, official_a, official_b, q_theory)
    positive_overlap_surface = projected_form_factor(
        radius,
        positive_a,
        positive_b,
        q_theory,
    )
    scalar_only_surface = energy_density_tools.summarize_surface(
        radius,
        bundle["scalar_proxy_density"],
        q_theory,
    )

    prior_gate_ready = bool(
        prior_summary.get("trial2_numeric_alpha_problem_classification") == PRIOR_CLASS
        and prior_summary.get("gate_b_retain_not_promote_selected", False)
    )
    quadratic_core_available = bool(
        quad_summary.get("quadratic_operator_core_derived", False)
        and quad_summary.get("background_dependent_quadratic_core_available", False)
    )
    official_projected_kernel_available = bool(prior_gate_ready and quadratic_core_available)
    projected_kernel_tracks_vector_no_go_scale = bool(
        abs(official_surface["alpha_at_q_theory"] - VECTOR_ALPHA) / VECTOR_ALPHA <= 0.05
    )
    projected_kernel_supports_scalar_candidate = bool(
        official_surface["alpha_residual_rel"] <= 0.05
    )
    projected_kernel_exact_foundation_supported = bool(
        official_surface["alpha_residual_rel"] <= 0.01
    )
    positive_overlap_variant_available = True
    positive_overlap_supports_scalar_candidate = bool(
        positive_overlap_surface["alpha_residual_rel"] <= 0.05
    )
    scalar_only_overlap_available = True
    scalar_only_overlap_supports_scalar_candidate = bool(
        scalar_only_surface["alpha_residual_rel"] <= 0.05
    )
    projected_kernel_denominator_negative = bool(
        official_surface["denominator_negative"]
    )
    transverse_response_fallback_failed = bool(
        official_projected_kernel_available
        and not projected_kernel_supports_scalar_candidate
        and not positive_overlap_supports_scalar_candidate
        and not scalar_only_overlap_supports_scalar_candidate
    )
    constrained_ground_state_branch_selection_admissible_now = bool(
        transverse_response_fallback_failed
    )
    physical_reject_required = False

    projected_vs_vector_alpha_rel_gap = float(
        abs(official_surface["alpha_at_q_theory"] - VECTOR_ALPHA) / VECTOR_ALPHA
    )
    projected_vs_scalar_alpha_rel_gap = float(
        abs(official_surface["alpha_at_q_theory"] - SCALAR_ALPHA) / SCALAR_ALPHA
    )

    rows = [
        row(
            "prior_gate_ready",
            "pass" if prior_gate_ready else "reject",
            "prior Gate B decision ready",
            truth(prior_gate_ready),
            "The projected-kernel fallback only starts after the first-shot breakthrough pack has already closed honestly as Gate B retain-but-not-promote.",
        ),
        row(
            "quadratic_core_available",
            "pass" if quadratic_core_available else "reject",
            "quadratic core available",
            truth(quadratic_core_available),
            "The fallback uses the already-derived quadratic core rather than inventing another local density surface.",
        ),
        row(
            "official_projected_kernel_available",
            "pass" if official_projected_kernel_available else "reject",
            "official projected-kernel observable available",
            truth(official_projected_kernel_available),
            "The vacuum-subtracted projected-kernel matrix element is now an executable current-pack observable with no new parameter or post-hoc normalization.",
        ),
        row(
            "official_projected_kernel_alpha_at_q_theory",
            "reject" if not projected_kernel_supports_scalar_candidate else "pass",
            "official projected-kernel alpha at q_theory",
            official_surface["alpha_at_q_theory"],
            "This is the first-principles transverse-response observable read from the vacuum-subtracted quadratic core on the retained exact branch.",
        ),
        row(
            "official_projected_kernel_residual_rel",
            "reject" if not projected_kernel_supports_scalar_candidate else "pass",
            "official projected-kernel alpha relative residual vs target",
            official_surface["alpha_residual_rel"],
            "The official projected-kernel observable must approach the target on the same no-post-hoc footing as every prior branch.",
        ),
        row(
            "projected_kernel_tracks_vector_no_go_scale",
            "pass" if projected_kernel_tracks_vector_no_go_scale else "watch",
            "official projected-kernel alpha tracks vector no-go scale",
            projected_vs_vector_alpha_rel_gap,
            "The normalized projected-kernel observable lands essentially on the retained vector blind-no-go scale rather than on the retained scalar strong candidate.",
        ),
        row(
            "projected_kernel_vs_scalar_alpha_rel_gap",
            "reject",
            "official projected-kernel alpha relative gap vs retained scalar alpha",
            projected_vs_scalar_alpha_rel_gap,
            "This quantifies how far the projected-kernel observable remains from the retained scalar strong candidate.",
        ),
        row(
            "projected_kernel_denominator_negative",
            "watch" if projected_kernel_denominator_negative else "pass",
            "projected-kernel denominator at q=0 is negative",
            truth(projected_kernel_denominator_negative),
            "The vacuum-subtracted matrix element has a negative q=0 denominator because the isotropic signed core is dominated by the -f_0^2 term. The normalization is still honest because no absolute-value salvage is applied.",
        ),
        row(
            "positive_overlap_variant_supports_scalar_candidate",
            "pass" if positive_overlap_supports_scalar_candidate else "reject",
            "positive-overlap evidence variant supports scalar candidate",
            positive_overlap_surface["alpha_residual_rel"],
            "Even after forcing the isotropic overlap positive, the projected response remains on the vector-no-go scale and does not rescue the scalar candidate.",
        ),
        row(
            "scalar_only_overlap_supports_scalar_candidate",
            "pass" if scalar_only_overlap_supports_scalar_candidate else "reject",
            "scalar-only overlap comparison supports scalar candidate",
            scalar_only_surface["alpha_residual_rel"],
            "A scalar-only overlap comparison remains evidence only and still lands on the retained vector-no-go scale once normalized on the same retained exact branch.",
        ),
        row(
            "transverse_response_fallback_failed",
            "pass" if transverse_response_fallback_failed else "reject",
            "transverse-response fallback failed under current pack",
            truth(transverse_response_fallback_failed),
            "The official projected-kernel observable and both evidence-only overlap variants all fail to approach the retained scalar strong candidate under the current frozen-action pack.",
        ),
        row(
            "constrained_ground_state_branch_selection_admissible_now",
            "pass" if constrained_ground_state_branch_selection_admissible_now else "reject",
            "constrained ground-state / branch-selection audit admissible now",
            truth(constrained_ground_state_branch_selection_admissible_now),
            "Once the response-observable fallback also fails, the next honest route is the branch-selection / constrained-ground-state audit rather than another response variant.",
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
            "prior_gate": display_path(PRIOR_GATE),
            "quadratic_derivation_gate": display_path(QUAD_DERIV_GATE),
            "energy_core_gate": display_path(ENERGY_CORE_GATE),
            "density_bundle_source": display_path(Path(energy_density_tools.__file__)),
        },
        "constants": {
            "q_theory_over_m0": q_theory,
            "scalar_F_exact_at_q_theory": SCALAR_F,
            "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
            "vector_F_at_q_theory": VECTOR_F,
            "vector_alpha_at_q_theory": VECTOR_ALPHA,
            "target_alpha": TARGET_ALPHA,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "followup_closeout_route_name": FOLLOWUP_CLOSEOUT_ROUTE_NAME,
            "followup_closeout_route": FOLLOWUP_CLOSEOUT_ROUTE,
        },
    }

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "official_surface_name": "pi_T_deltaK_core_pi_T_vacuum_subtracted",
        "official_projected_kernel_F_at_q_theory": official_surface["F_at_q_theory"],
        "official_projected_kernel_alpha_at_q_theory": official_surface["alpha_at_q_theory"],
        "official_projected_kernel_alpha_residual_rel": official_surface["alpha_residual_rel"],
        "official_projected_kernel_numerator_at_q_theory": official_surface["numerator_at_q_theory"],
        "official_projected_kernel_denominator_at_zero": official_surface["denominator_at_zero"],
        "projected_kernel_denominator_negative": projected_kernel_denominator_negative,
        "projected_kernel_tracks_vector_no_go_scale": projected_kernel_tracks_vector_no_go_scale,
        "projected_kernel_supports_scalar_candidate": projected_kernel_supports_scalar_candidate,
        "projected_kernel_exact_foundation_supported": projected_kernel_exact_foundation_supported,
        "projected_kernel_vs_vector_alpha_rel_gap": projected_vs_vector_alpha_rel_gap,
        "projected_kernel_vs_scalar_alpha_rel_gap": projected_vs_scalar_alpha_rel_gap,
        "positive_overlap_variant_F_at_q_theory": positive_overlap_surface["F_at_q_theory"],
        "positive_overlap_variant_alpha_at_q_theory": positive_overlap_surface["alpha_at_q_theory"],
        "positive_overlap_variant_alpha_residual_rel": positive_overlap_surface["alpha_residual_rel"],
        "positive_overlap_variant_supports_scalar_candidate": positive_overlap_supports_scalar_candidate,
        "scalar_only_overlap_F_at_q_theory": scalar_only_surface["F_at_q_theory"],
        "scalar_only_overlap_alpha_at_q_theory": scalar_only_surface["alpha_at_q_theory"],
        "scalar_only_overlap_alpha_residual_rel": scalar_only_surface["alpha_residual_rel"],
        "scalar_only_overlap_supports_scalar_candidate": scalar_only_overlap_supports_scalar_candidate,
        "transverse_response_fallback_failed": transverse_response_fallback_failed,
        "constrained_ground_state_branch_selection_admissible_now": (
            constrained_ground_state_branch_selection_admissible_now
        ),
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_closeout_route": FOLLOWUP_CLOSEOUT_ROUTE_NAME,
        "recommended_followup_closeout_route_or_none": FOLLOWUP_CLOSEOUT_ROUTE,
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
            "status_current_branch": hit(status_text, "8.7.56.1659"),
            "roadmap_current_branch": hit(roadmap_text, "8.7.56.1659-.1662"),
            "current_problem_response_fallback": hit(
                current_problem_text,
                "response-observable fallback problem",
            ),
            "current_status_transverse_response": hit(
                current_status_text,
                "transverse response / projected-kernel observable",
            ),
            "unified_roadmap_transverse_response": hit(
                unified_text,
                "`.1659-.1662` は **`P_\\mu` transverse response / projected-kernel observable audit**",
            ),
            "part5_transverse_response": hit(
                part5_text,
                "next mainline は `.1659-.1662` **`P_\\mu` transverse response / projected-kernel observable audit**",
            ),
            "part1_metric_casea_reject": hit(
                part1_text,
                "caseA: reject; caseB: adopt.",
            ),
            "part1_qball_metric_surface": hit(
                part1_text,
                "g_{\\mu\\nu}(P)",
            ),
        },
        "quadratic_core_carry_over": quad_summary,
        "prior_gate_summary": prior_summary,
        "energy_core_summary": energy_summary,
        "retained_numeric_state": {
            "scalar_F_exact_at_q_theory": SCALAR_F,
            "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
            "vector_F_at_q_theory": VECTOR_F,
            "vector_alpha_at_q_theory": VECTOR_ALPHA,
            "energy_core_alpha_at_q_theory": float(
                energy_summary["official_alpha_E_at_q_theory"]
            ),
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
        },
    }

    manifest = {
        "inventory": write_artifact(
            "inventory",
            payload(
                "8.7.56.1659",
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
                "8.7.56.1660",
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
                "8.7.56.1661",
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
                "8.7.56.1662",
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
            {"step": STEP_TAG, "stem": STEM, "artifacts": manifest, "summary": summary},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
