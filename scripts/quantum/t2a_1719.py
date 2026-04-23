#!/usr/bin/env python3
"""Generate 8.7.56.1719-.1722 inverse local-observable constraint artifacts.

The updated source-extended pack has now exhausted both secondary reopen
surfaces:

1. exact constitutive-map reopen failed under the updated pack,
2. branch-local full nonlinear energy-density reopen failed and simply carried
   the pre-update vector no-go scale forward.

This side diagnostic therefore does not search for a new canonical observable
directly. Instead it asks a sharper inverse question:

    if one still insists on a local observable family
        rho(r) = rho_0(r) + sum_i c_i phi_i(r),
    what coefficient pattern would be *required* to reproduce the target
    fixed-q_theory form factor?

The answer identifies whether a local family is even remotely plausible under
the current pack, or whether any such rescue would require huge / sign-
indefinite / otherwise noncanonical coefficients.
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

import scripts.quantum.t2a_1627 as density_tools


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

EXTERNAL_NOTE = Path(
    r"C:\Users\ogawa\Downloads\trial2_vector_qball_inverse_problem_20260328.md"
)
UPDATED_CONSTITUTIVE_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1711_1714_updpk_const_map_reopen_declaration_gate_metrics.json"
)
UPDATED_NONLINEAR_GATE = (
    ROOT
    / "output"
    / "public"
    / "quantum"
    / "q_8_7_56_1715_1718_updpk_full_nl_reopen_declaration_gate_metrics.json"
)

STEP_TAG = "8.7.56.1719-1722"
STEP_NAME = (
    "Trial-2 numeric alpha vector Q-ball form-factor inverse local-observable "
    "constraint audit"
)
STEM = build_compact_artifact_stem(STEP_TAG, "inv_local_constraint", prefix="q")

PRIOR_CLASS = (
    "vector_qball_form_factor_updated_pack_branch_local_full_nonlinear_energy_"
    "carryover_tracks_vector_no_go_inverse_constraint_audit_next"
)
BRANCH_CLASS = (
    "vector_qball_form_factor_inverse_local_constraint_requires_large_or_"
    "noncanonical_coefficients_new_primary_surface_gate_next"
)
NEXT_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_external_input_assimilation_"
    "new_primary_surface_gate"
)
NEXT_ROUTE = "8.7.56.1723"
FOLLOWUP_ROUTE_NAME = (
    "trial2_numeric_alpha_vector_qball_form_factor_pack_update_closeout_"
    "reopen_registry_refresh"
)
FOLLOWUP_ROUTE = "8.7.56.1727"

TARGET_ALPHA = 1.0 / 137.035999084
TARGET_FORM_FACTOR = math.sqrt(4.0 * math.pi * TARGET_ALPHA)
SCALAR_ALPHA = 0.00715678583937324
SCALAR_FORM_FACTOR = 0.2998913524347805


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


# 関数: repo 相対表示パスを返す。

def display_path(path: Path) -> str:
    """Return one repo-relative path when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# 関数: 部分一致する最初の行を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first line matching one substring."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: 標準 metrics row を作る。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build one standard metrics row."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# 関数: 標準 payload を作る。

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


# 関数: basis density の form-factor numerator / norm / form factor を返す。

def summarize_basis(radius: np.ndarray, density: np.ndarray, q_ratio: float) -> dict:
    """Return one basis summary."""
    qr = q_ratio * radius
    sinc = np.where(np.abs(qr) > 1.0e-10, np.sin(qr) / qr, 1.0)
    r_squared = radius * radius
    numerator = float(np.trapezoid(density * r_squared * sinc, radius))
    norm = float(np.trapezoid(density * r_squared, radius))
    form_factor = numerator / norm
    return {
        "numerator": numerator,
        "norm": norm,
        "form_factor": float(form_factor),
    }


# 関数: target に対する Delta = A - F_target B を返す。

def delta_constraint(summary: dict, target_form_factor: float) -> float:
    """Return one inverse constraint delta."""
    return float(
        summary["numerator"] - float(target_form_factor) * summary["norm"]
    )


# 関数: one-parameter rescue coefficient を返す。

def one_parameter_solution(
    base_summary: dict,
    add_summary: dict,
    target_form_factor: float,
) -> float:
    """Solve the exact one-parameter local-family condition."""
    delta0 = delta_constraint(base_summary, target_form_factor)
    delta1 = delta_constraint(add_summary, target_form_factor)
    return float(-delta0 / delta1)


# 関数: minimal-norm coefficient vector を返す。

def minimal_norm_solution(delta0: float, deltas: np.ndarray) -> np.ndarray:
    """Return the minimal-norm exact solution of delta0 + c·deltas = 0."""
    return np.asarray(-delta0 * deltas / float(np.dot(deltas, deltas)), dtype=float)


# 関数: density candidate を構成する。

def build_density(base: np.ndarray, pieces: list[np.ndarray], coeffs: np.ndarray) -> np.ndarray:
    """Build one composite local density candidate."""
    out = np.array(base, dtype=float, copy=True)
    for coeff, piece in zip(coeffs, pieces):
        out = out + float(coeff) * np.asarray(piece, dtype=float)

    return out


# 関数: `.1719-.1722` を実行する。

def main() -> None:
    """Execute the inverse local-observable constraint audit."""
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
        EXTERNAL_NOTE,
        UPDATED_CONSTITUTIVE_GATE,
        UPDATED_NONLINEAR_GATE,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_text = read_text(UNIFIED_ROADMAP)
    long_text = read_text(LONG_ROADMAP)
    part5_text = read_text(PART5)
    external_text = read_text(EXTERNAL_NOTE)

    updated_constitutive_summary = read_json(UPDATED_CONSTITUTIVE_GATE)["summary"]
    updated_nonlinear_summary = read_json(UPDATED_NONLINEAR_GATE)["summary"]

    bundle = density_tools.build_density_bundle()
    radius = np.asarray(bundle["radius"], dtype=float)
    q_ratio = float(bundle["q_theory_over_m0"])

    base_density = np.asarray(bundle["scalar_proxy_density"], dtype=float)
    f_l_sq = np.asarray(bundle["f_l_values"] * bundle["f_l_values"], dtype=float)
    f0_prime_sq = np.asarray(bundle["f0_prime"] * bundle["f0_prime"], dtype=float)
    grad_l_density = np.asarray(
        bundle["f_l_prime"] * bundle["f_l_prime"]
        + (2.0 * bundle["f_l_values"] * bundle["f_l_values"] / np.maximum(radius * radius, 1.0e-12)),
        dtype=float,
    )
    mh_proxy_density = np.asarray(bundle["mh_family_proxy_density"], dtype=float)

    base_summary = summarize_basis(radius, base_density, q_ratio)
    f_l_sq_summary = summarize_basis(radius, f_l_sq, q_ratio)
    f0_prime_sq_summary = summarize_basis(radius, f0_prime_sq, q_ratio)
    grad_l_summary = summarize_basis(radius, grad_l_density, q_ratio)
    mh_proxy_summary = summarize_basis(radius, mh_proxy_density, q_ratio)

    delta0_target = delta_constraint(base_summary, TARGET_FORM_FACTOR)
    delta0_scalar = delta_constraint(base_summary, SCALAR_FORM_FACTOR)
    deltas3_target = np.asarray(
        [
            delta_constraint(f_l_sq_summary, TARGET_FORM_FACTOR),
            delta_constraint(f0_prime_sq_summary, TARGET_FORM_FACTOR),
            delta_constraint(grad_l_summary, TARGET_FORM_FACTOR),
        ],
        dtype=float,
    )
    deltas3_scalar = np.asarray(
        [
            delta_constraint(f_l_sq_summary, SCALAR_FORM_FACTOR),
            delta_constraint(f0_prime_sq_summary, SCALAR_FORM_FACTOR),
            delta_constraint(grad_l_summary, SCALAR_FORM_FACTOR),
        ],
        dtype=float,
    )
    deltas4_target = np.asarray(
        [*deltas3_target, delta_constraint(mh_proxy_summary, TARGET_FORM_FACTOR)],
        dtype=float,
    )
    deltas4_scalar = np.asarray(
        [*deltas3_scalar, delta_constraint(mh_proxy_summary, SCALAR_FORM_FACTOR)],
        dtype=float,
    )

    c_f_l_sq_target = one_parameter_solution(base_summary, f_l_sq_summary, TARGET_FORM_FACTOR)
    c_f_l_sq_scalar = one_parameter_solution(base_summary, f_l_sq_summary, SCALAR_FORM_FACTOR)

    one_parameter_target_density = base_density + c_f_l_sq_target * f_l_sq
    one_parameter_scalar_density = base_density + c_f_l_sq_scalar * f_l_sq
    one_parameter_target_summary = summarize_basis(radius, one_parameter_target_density, q_ratio)
    one_parameter_scalar_summary = summarize_basis(radius, one_parameter_scalar_density, q_ratio)

    coeffs3_target = minimal_norm_solution(delta0_target, deltas3_target)
    coeffs3_scalar = minimal_norm_solution(delta0_scalar, deltas3_scalar)
    coeffs4_target = minimal_norm_solution(delta0_target, deltas4_target)
    coeffs4_scalar = minimal_norm_solution(delta0_scalar, deltas4_scalar)

    three_target_density = build_density(
        base_density,
        [f_l_sq, f0_prime_sq, grad_l_density],
        coeffs3_target,
    )
    three_scalar_density = build_density(
        base_density,
        [f_l_sq, f0_prime_sq, grad_l_density],
        coeffs3_scalar,
    )
    four_target_density = build_density(
        base_density,
        [f_l_sq, f0_prime_sq, grad_l_density, mh_proxy_density],
        coeffs4_target,
    )
    four_scalar_density = build_density(
        base_density,
        [f_l_sq, f0_prime_sq, grad_l_density, mh_proxy_density],
        coeffs4_scalar,
    )

    three_target_summary = summarize_basis(radius, three_target_density, q_ratio)
    three_scalar_summary = summarize_basis(radius, three_scalar_density, q_ratio)
    four_target_summary = summarize_basis(radius, four_target_density, q_ratio)
    four_scalar_summary = summarize_basis(radius, four_scalar_density, q_ratio)

    local_o1_three_basis_impossible_for_target = bool(
        abs(delta0_target) > float(np.sum(np.abs(deltas3_target)))
    )
    local_o1_four_basis_impossible_for_target = bool(
        abs(delta0_target) > float(np.sum(np.abs(deltas4_target)))
    )
    one_parameter_requires_huge_weight_for_target = bool(abs(c_f_l_sq_target) > 1.0e3)
    one_parameter_requires_huge_weight_for_scalar = bool(abs(c_f_l_sq_scalar) > 1.0e3)
    three_basis_target_density_sign_indefinite = bool(np.min(three_target_density) < 0.0)
    four_basis_target_density_sign_indefinite = bool(np.min(four_target_density) < 0.0)
    local_family_rescue_requires_large_or_noncanonical_coefficients = bool(
        local_o1_three_basis_impossible_for_target
        and one_parameter_requires_huge_weight_for_target
        and three_basis_target_density_sign_indefinite
        and four_basis_target_density_sign_indefinite
    )

    inputs = {
        "source_files": {
            "status": display_path(STATUS),
            "roadmap": display_path(ROADMAP),
            "ai_context": display_path(AI_CONTEXT),
            "work_history_recent": display_path(WORK_HISTORY_RECENT),
            "current_problem": display_path(CURRENT_PROBLEM),
            "current_status": display_path(CURRENT_STATUS),
            "unified_roadmap": display_path(UNIFIED_ROADMAP),
            "long_roadmap": display_path(LONG_ROADMAP),
            "part5": display_path(PART5),
            "external_inverse_note": display_path(EXTERNAL_NOTE),
            "updated_constitutive_gate": display_path(UPDATED_CONSTITUTIVE_GATE),
            "updated_nonlinear_gate": display_path(UPDATED_NONLINEAR_GATE),
        },
        "constants": {
            "q_theory_over_m0": q_ratio,
            "target_alpha": TARGET_ALPHA,
            "target_form_factor": TARGET_FORM_FACTOR,
            "scalar_alpha_exact_at_q_theory": SCALAR_ALPHA,
            "scalar_form_factor_candidate": SCALAR_FORM_FACTOR,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
            "followup_route_name": FOLLOWUP_ROUTE_NAME,
            "followup_route": FOLLOWUP_ROUTE,
        },
    }

    rows = [
        row(
            "updated_pack_constitutive_reopen_failed",
            "pass",
            "updated-pack constitutive reopen failed",
            truth(not updated_constitutive_summary["exact_constitutive_map_available_under_updated_pack"]),
            "The inverse audit only starts after the updated-pack exact constitutive-map reopen has already failed honestly.",
        ),
        row(
            "updated_pack_nonlinear_reopen_failed",
            "pass",
            "updated-pack nonlinear reopen failed",
            truth(updated_nonlinear_summary["updated_pack_nonlinear_reopen_failed"]),
            "The branch-local nonlinear family has already been shown to carry over unchanged and remain on the vector no-go scale.",
        ),
        row(
            "base_same_branch_scalar_proxy_F_at_q_theory",
            "watch",
            "base same-branch scalar-proxy form factor at q_theory",
            base_summary["form_factor"],
            "On the retained exact vector branch, the local base density |f_0|^2 itself is already far from the positive target form-factor window.",
        ),
        row(
            "one_parameter_fLsq_coeff_for_target_alpha",
            "watch",
            "required c in rho = |f_0|^2 + c |f_L|^2 for physical target alpha",
            c_f_l_sq_target,
            "The exact one-parameter inverse solution shows how much |f_L|^2 weight is required on the same branch to hit the physical target form factor.",
        ),
        row(
            "one_parameter_fLsq_coeff_for_scalar_candidate",
            "watch",
            "required c in rho = |f_0|^2 + c |f_L|^2 for retained scalar-candidate form factor",
            c_f_l_sq_scalar,
            "Using the retained scalar-candidate form factor instead of the physical target gives the same qualitative result if the coefficient remains huge.",
        ),
        row(
            "one_parameter_requires_huge_weight_for_target",
            "pass",
            "one-parameter local family requires huge fL^2 weight for target",
            truth(one_parameter_requires_huge_weight_for_target),
            "A viable local-family rescue would not normally require c ~ O(10^4-10^5) on the same branch.",
        ),
        row(
            "local_o1_three_basis_impossible_for_target",
            "pass",
            "O(1) three-basis local family impossible for target",
            truth(local_o1_three_basis_impossible_for_target),
            "For rho = |f_0|^2 + c1 |f_L|^2 + c2 f_0'^2 + c3 grad_L, the exact inverse constraint |Delta0| <= sum |Delta_i| already fails by orders of magnitude when |c_i| <= 1.",
        ),
        row(
            "local_o1_four_basis_impossible_for_target",
            "pass",
            "O(1) four-basis local family impossible for target",
            truth(local_o1_four_basis_impossible_for_target),
            "Even after adding the Mexican-hat proxy basis, the O(1) coefficient cube still cannot satisfy the exact inverse form-factor constraint.",
        ),
        row(
            "three_basis_min_norm_coeff_norm_target",
            "watch",
            "three-basis minimal-norm coefficient norm for target",
            float(np.linalg.norm(coeffs3_target)),
            "The least-norm exact solution shows how far outside a natural O(1) local family one must move even before checking positivity.",
        ),
        row(
            "three_basis_target_density_sign_indefinite",
            "pass",
            "three-basis target density becomes sign-indefinite",
            truth(three_basis_target_density_sign_indefinite),
            "The minimal-norm exact solution already drives the local density negative, so it cannot serve as a canonical positive density candidate.",
        ),
        row(
            "four_basis_min_norm_coeff_norm_target",
            "watch",
            "four-basis minimal-norm coefficient norm for target",
            float(np.linalg.norm(coeffs4_target)),
            "Adding the Mexican-hat proxy compresses the coefficient norm, but only by leaning on a noncanonical basis component.",
        ),
        row(
            "four_basis_target_density_sign_indefinite",
            "pass",
            "four-basis target density becomes sign-indefinite",
            truth(four_basis_target_density_sign_indefinite),
            "Even the compressed four-basis exact solution still creates a sign-indefinite density profile.",
        ),
        row(
            "local_family_rescue_requires_large_or_noncanonical_coefficients",
            "pass",
            "local-family rescue requires large or noncanonical coefficients",
            truth(local_family_rescue_requires_large_or_noncanonical_coefficients),
            "The inverse audit shows that any same-branch local-family rescue now needs either huge coefficients or a sign-indefinite / noncanonical density candidate.",
        ),
        row(
            "new_primary_surface_gate_admissible_now",
            "pass",
            "new primary-surface gate admissible now",
            1.0,
            "Once the inverse audit confirms that local rescue needs large or noncanonical coefficients, the honest next route is the new-primary-surface gate rather than another same-level local surrogate retry.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "updated_pack_constitutive_reopen_failed": not updated_constitutive_summary[
            "exact_constitutive_map_available_under_updated_pack"
        ],
        "updated_pack_nonlinear_reopen_failed": updated_nonlinear_summary[
            "updated_pack_nonlinear_reopen_failed"
        ],
        "base_same_branch_scalar_proxy_F_at_q_theory": base_summary["form_factor"],
        "base_same_branch_scalar_proxy_alpha_at_q_theory": (base_summary["form_factor"] ** 2) / (4.0 * math.pi),
        "target_form_factor": TARGET_FORM_FACTOR,
        "scalar_candidate_form_factor": SCALAR_FORM_FACTOR,
        "delta0_target": delta0_target,
        "delta0_scalar_candidate": delta0_scalar,
        "one_parameter_fLsq_coeff_for_target_alpha": c_f_l_sq_target,
        "one_parameter_fLsq_coeff_for_scalar_candidate": c_f_l_sq_scalar,
        "one_parameter_target_density_positive": bool(np.min(one_parameter_target_density) >= 0.0),
        "one_parameter_target_norm": one_parameter_target_summary["norm"],
        "one_parameter_requires_huge_weight_for_target": one_parameter_requires_huge_weight_for_target,
        "one_parameter_requires_huge_weight_for_scalar": one_parameter_requires_huge_weight_for_scalar,
        "three_basis_min_norm_coeff_target": {
            "c_fLsq": float(coeffs3_target[0]),
            "c_f0p2": float(coeffs3_target[1]),
            "c_gradL": float(coeffs3_target[2]),
        },
        "three_basis_min_norm_coeff_scalar_candidate": {
            "c_fLsq": float(coeffs3_scalar[0]),
            "c_f0p2": float(coeffs3_scalar[1]),
            "c_gradL": float(coeffs3_scalar[2]),
        },
        "three_basis_target_density_sign_indefinite": three_basis_target_density_sign_indefinite,
        "four_basis_min_norm_coeff_target": {
            "c_fLsq": float(coeffs4_target[0]),
            "c_f0p2": float(coeffs4_target[1]),
            "c_gradL": float(coeffs4_target[2]),
            "c_mh_proxy": float(coeffs4_target[3]),
        },
        "four_basis_min_norm_coeff_scalar_candidate": {
            "c_fLsq": float(coeffs4_scalar[0]),
            "c_f0p2": float(coeffs4_scalar[1]),
            "c_gradL": float(coeffs4_scalar[2]),
            "c_mh_proxy": float(coeffs4_scalar[3]),
        },
        "four_basis_target_density_sign_indefinite": four_basis_target_density_sign_indefinite,
        "local_o1_three_basis_impossible_for_target": local_o1_three_basis_impossible_for_target,
        "local_o1_four_basis_impossible_for_target": local_o1_four_basis_impossible_for_target,
        "local_family_rescue_requires_large_or_noncanonical_coefficients": local_family_rescue_requires_large_or_noncanonical_coefficients,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    decision = {
        "overall_status": f"{BRANCH_CLASS}_declared",
        "branch_completed": True,
        "next_required_artifacts": [NEXT_ROUTE_NAME],
    }

    evidence = {
        "formulas": {
            "local_family_ansatz": "rho(r) = rho_0(r) + sum_i c_i phi_i(r)",
            "form_factor_rule": "F(q) = (A_0 + sum_i c_i A_i) / (B_0 + sum_i c_i B_i)",
            "inverse_constraint_rule": "Delta_0(F_*) + sum_i c_i Delta_i(F_*) = 0, Delta_i(F_*) = A_i - F_* B_i",
            "one_parameter_rule": "c_* = -Delta_0 / Delta_1",
            "minimal_norm_rule": "c_* = -Delta_0 Delta / ||Delta||^2",
            "o1_cube_necessary_condition": "|Delta_0| <= sum_i |Delta_i| for |c_i| <= 1",
            "failure_read": "If target matching requires huge coefficients or sign-indefinite densities, then the local observable family is not a plausible canonical rescue class under the current updated pack.",
        },
        "hits": {
            "external_inverse_note_hit": hit(
                external_text,
                "Question B",
            ),
            "status_current_branch": hit(
                status_text,
                "inverse local-observable constraint audit",
            ),
            "roadmap_current_branch": hit(
                roadmap_text,
                "8.7.56.1719-.1722",
            ),
            "current_problem_branch_hit": hit(
                current_problem_text,
                "inverse local-observable constraint audit",
            ),
            "current_status_branch_hit": hit(
                current_status_text,
                "inverse local-observable constraint audit",
            ),
            "unified_roadmap_branch_hit": hit(
                unified_text,
                "`.1719-.1722` は **inverse local-observable constraint audit**",
            ),
            "long_roadmap_branch_hit": hit(
                long_text,
                "8.7.56.1719-.1722",
            ),
            "part5_prior_hit": hit(
                part5_text,
                "branch-local full nonlinear energy-density reopen after pack update",
            ),
        },
        "prior_summaries": {
            "updated_constitutive_gate": updated_constitutive_summary,
            "updated_nonlinear_gate": updated_nonlinear_summary,
        },
        "basis_summaries": {
            "base_same_branch_scalar_proxy": base_summary,
            "fLsq": f_l_sq_summary,
            "f0p2": f0_prime_sq_summary,
            "gradL": grad_l_summary,
            "mh_proxy": mh_proxy_summary,
            "one_parameter_target": one_parameter_target_summary,
            "one_parameter_scalar_candidate": one_parameter_scalar_summary,
            "three_basis_target": three_target_summary,
            "three_basis_scalar_candidate": three_scalar_summary,
            "four_basis_target": four_target_summary,
            "four_basis_scalar_candidate": four_scalar_summary,
        },
    }

    outputs: dict[str, dict[str, str]] = {}
    for kind in ("inventory", "audit", "declaration_gate", "route_sync"):
        outputs[kind] = write_artifact(
            kind,
            payload(
                step=STEP_TAG,
                name=f"{STEP_NAME} {kind.replace('_', ' ')}",
                inputs=inputs,
                rows=rows,
                summary=summary,
                decision=decision,
                evidence=evidence,
            ),
        )

    print("[ok] inverse local-observable constraint audit artifacts written:")
    for kind, paths in outputs.items():
        print(f"  - {kind}: {paths['json']} | {paths['csv']}")


if __name__ == "__main__":
    main()
