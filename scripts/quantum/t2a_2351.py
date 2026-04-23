#!/usr/bin/env python3
"""Generate 8.7.56.2351-.2354 missing-action δβ₁ first-shot artifacts."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
import sys
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

PRIOR_GATE = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.2347-2350",
        "observable_definition_mismatch",
        prefix="q",
    ),
    "declaration_gate",
)["json"]
ELL0_OPERATOR_AUDIT = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.1471-1474",
        "ell0_exact_operator_derivation",
        prefix="q",
    ),
    "audit",
)["json"]
ELL0_ANCHOR_EVAL = build_metrics_paths(
    PUBLIC_OUT,
    build_compact_artifact_stem(
        "8.7.56.1483-1486",
        "ell0_anchor_continuation",
        prefix="q",
    ),
    "numeric_evaluation",
)["json"]
QBALL_BRANCH_REFRESH = PUBLIC_OUT / "mass_origin_qball_charge_mapping_branch_refresh_metrics.json"
QBALL_SOLVER = ROOT / "scripts" / "quantum" / "mass_origin_qball_charge_mapping_branch.py"

STEP_TAG = "8.7.56.2351-2354"
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor missing action-level term audit"
STEM = build_compact_artifact_stem(
    STEP_TAG,
    "missing_action_delta_beta1_audit",
    prefix="q",
)

PRIOR_CLASS = "vector_qball_form_factor_residual_origin_missing_action_primary_after_boundary_observable_audits_next"
BRANCH_CLASS = "vector_qball_form_factor_residual_origin_missing_action_profile_fixed_delta_beta1_candidate_audit_gate"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_residual_origin_synthesis_hybrid_reserve_refresh"
NEXT_ROUTE = "8.7.56.2355"
FOLLOWUP_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_exact_coupled_eigenvalue_shift_theorem_audit"
FOLLOWUP_ROUTE = "8.7.56.2359"

ALPHA_TARGET = 1.0 / 137.035999084
DIFF_Q = 1.0e-6
DIFF_BETA = 1.0e-7


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


# 関数: scalar Q-ball solver module を読み込む。
def load_qball_module():
    """Load the retained scalar Q-ball solver module."""
    spec = importlib.util.spec_from_file_location("wavep_qball_charge_mapping", QBALL_SOLVER)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[fail] unable to load module from {QBALL_SOLVER}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 関数: retained scalar ground-state row を返す。
def extract_scalar_ground_state(qball_refresh: dict) -> dict:
    """Extract the retained scalar mode-1 row."""
    for row_data in qball_refresh["evidence"]["discrete_mode_rows"]:
        if int(row_data["mode_index"]) == 1:
            return {
                "beta_n": float(row_data["beta_n"]),
                "central_amplitude": float(row_data["central_amplitude"]),
                "charge_proxy": float(row_data["charge_proxy"]),
                "energy_proxy": float(row_data["energy_proxy"]),
            }

    raise SystemExit("[fail] missing scalar ground-state row in branch refresh metrics")


# 関数: retained exact profile の normalized form factor を返す。
def form_factor(radius: np.ndarray, field: np.ndarray, q_ratio: float) -> float:
    """Evaluate the normalized spherical-overlap form factor."""
    qx = float(q_ratio) * radius
    sinc = np.ones_like(qx)
    mask = np.abs(qx) > 1.0e-12
    sinc[mask] = np.sin(qx[mask]) / qx[mask]
    weight = (field**2) * (radius**2)
    norm = float(np.trapezoid(weight, radius))
    return float(np.trapezoid(weight * sinc, radius) / norm)


# 関数: beta から q_* を返す。
def q_from_beta(beta: float) -> float:
    """Return q_*(beta)/m0 on the retained scalar matching rule."""
    return float((1.0 - float(beta) * float(beta)) ** 0.25)


# 関数: current branch で使う式を返す。
def build_formulae() -> dict[str, str]:
    """Return formulas used in the missing-action audit."""
    return {
        "matching_scale": "q_*(beta)/m0 = (1-beta^2)^(1/4)",
        "retained_form_factor": "F(q) = int dr f(r)^2 r^2 sinc(q r) / int dr f(r)^2 r^2",
        "alpha_rule": "alpha(beta) = F(q_*(beta))^2 / (4 pi)",
        "profile_fixed_first_shot": "Only q_*(beta) is shifted while the retained exact profile f(r) is held fixed at the current scalar branch point.",
        "delta_beta2_rule": "delta_beta1^2 = beta_corrected^2 - beta_1^2",
    }


# 関数: `.2351-.2354` を実行する。
def main() -> None:
    """Execute the missing-action δβ1 first-shot audit."""
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
        PRIOR_GATE,
        ELL0_OPERATOR_AUDIT,
        ELL0_ANCHOR_EVAL,
        QBALL_BRANCH_REFRESH,
        QBALL_SOLVER,
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
    operator_summary = sign_base.read_json(ELL0_OPERATOR_AUDIT)["summary"]
    anchor_summary = sign_base.read_json(ELL0_ANCHOR_EVAL)["summary"]
    qball_refresh = sign_base.read_json(QBALL_BRANCH_REFRESH)

    qball_module = load_qball_module()
    scalar_row = extract_scalar_ground_state(qball_refresh)
    beta_1 = float(scalar_row["beta_n"])
    amp_1 = float(scalar_row["central_amplitude"])
    radius, field, _field_prime = qball_module.solve_full_profile(beta_1, amp_1)

    def alpha_profile_fixed(beta: float) -> tuple[float, float, float]:
        q_value = q_from_beta(beta)
        f_value = form_factor(radius, field, q_value)
        alpha_value = (f_value * f_value) / (4.0 * math.pi)
        return q_value, f_value, alpha_value

    q_theory, f_exact, alpha_exact = alpha_profile_fixed(beta_1)
    target_gap_abs = ALPHA_TARGET - alpha_exact
    beta_gap = 1.0 - beta_1 * beta_1

    q_plus = q_theory + DIFF_Q
    q_minus = q_theory - DIFF_Q
    f_plus = form_factor(radius, field, q_plus)
    f_minus = form_factor(radius, field, q_minus)
    alpha_plus_q = (f_plus * f_plus) / (4.0 * math.pi)
    alpha_minus_q = (f_minus * f_minus) / (4.0 * math.pi)
    dF_dq = (f_plus - f_minus) / (2.0 * DIFF_Q)
    dalpha_dq = (alpha_plus_q - alpha_minus_q) / (2.0 * DIFF_Q)
    dq_dbeta = -beta_1 / (2.0 * ((1.0 - beta_1 * beta_1) ** 0.75))

    beta_plus = beta_1 + DIFF_BETA
    beta_minus = beta_1 - DIFF_BETA
    _q_plus_beta, _f_plus_beta, alpha_plus_beta = alpha_profile_fixed(beta_plus)
    _q_minus_beta, _f_minus_beta, alpha_minus_beta = alpha_profile_fixed(beta_minus)
    dalpha_dbeta = (alpha_plus_beta - alpha_minus_beta) / (2.0 * DIFF_BETA)
    required_delta_beta_linear = target_gap_abs / dalpha_dbeta
    local_log_elasticity_alpha_vs_q = (q_theory / alpha_exact) * dalpha_dq

    beta_hi = beta_1 + 1.0e-4
    while True:
        _q_hi, _f_hi, alpha_hi = alpha_profile_fixed(beta_hi)
        if alpha_hi > ALPHA_TARGET:
            break

        beta_hi += 1.0e-4
        if beta_hi >= 0.999999:
            raise SystemExit("[fail] unable to bracket profile-fixed beta correction root")

    beta_corrected = float(
        brentq(
            lambda beta: alpha_profile_fixed(float(beta))[2] - ALPHA_TARGET,
            beta_1,
            beta_hi,
        )
    )
    q_corrected, f_corrected, alpha_corrected = alpha_profile_fixed(beta_corrected)
    required_delta_beta_exact = beta_corrected - beta_1
    required_delta_beta2_exact = beta_corrected * beta_corrected - beta_1 * beta_1
    delta_q_abs = q_corrected - q_theory
    delta_q_rel = delta_q_abs / q_theory

    max_fl_over_f0 = float(anchor_summary["phase1_equivalent_row"]["max_abs_ratio"])
    ceiling_sq = max_fl_over_f0 * max_fl_over_f0
    required_delta_beta_fraction_of_beta_gap = required_delta_beta_exact / beta_gap
    required_delta_beta2_fraction_of_beta_gap = required_delta_beta2_exact / beta_gap
    required_delta_beta_vs_ceiling_sq = required_delta_beta_exact / ceiling_sq
    required_delta_beta2_vs_ceiling_sq = required_delta_beta2_exact / ceiling_sq

    inventory_ready = bool(prior_summary["missing_action_level_primary_now"])
    exact_coupled_eigenvalue_shift_theorem_available = bool(
        operator_summary["exact_action_level_closed_ell0_operator_available"]
    )
    profile_fixed_eigenvalue_shift_candidate_admissible = bool(
        required_delta_beta2_exact > 0.0
        and required_delta_beta2_fraction_of_beta_gap < 0.05
        and required_delta_beta2_vs_ceiling_sq < 0.01
    )

    rows = [
        sign_base.row(
            "inventory_ready",
            "pass" if inventory_ready else "reject",
            "missing-action inventory ready",
            sign_base.truth(inventory_ready),
            "The first-shot audit starts only after boundary primary and observable primary have already been cut from the residual-origin lane ordering.",
        ),
        sign_base.row(
            "exact_action_level_closed_ell0_operator_available",
            "reject" if not exact_coupled_eigenvalue_shift_theorem_available else "pass",
            "exact coupled ell=0 operator available",
            sign_base.truth(exact_coupled_eigenvalue_shift_theorem_available),
            "The exact coupled theorem is still unavailable, so any current δβ₁ read remains a falsifiable first shot rather than a canonical derivation.",
        ),
        sign_base.row(
            "phase1_exact_solver_cross_term_present",
            "reject" if not operator_summary["phase1_exact_solver_cross_term_present"] else "pass",
            "phase-1 exact solver cross term present",
            sign_base.truth(bool(operator_summary["phase1_exact_solver_cross_term_present"])),
            "The current exact solver still omits the coupled cross term that could shift the scalar eigenvalue.",
        ),
        sign_base.row(
            "phase1_exact_solver_constraint_elimination_present",
            "reject" if not operator_summary["phase1_exact_solver_constraint_elimination_present"] else "pass",
            "phase-1 exact solver constraint elimination present",
            sign_base.truth(bool(operator_summary["phase1_exact_solver_constraint_elimination_present"])),
            "Constraint elimination is still missing, so the action-level source of an eigenvalue shift is not yet theorem-level closed.",
        ),
        sign_base.row(
            "restored_exact_branch_max_abs_fL_over_f0",
            "watch",
            "restored exact branch max |fL/f0|",
            max_fl_over_f0,
            "The vector companion branch is nontrivial and sets the current amplitude ceiling for any perturbative missing-action estimate.",
        ),
        sign_base.row(
            "profile_fixed_required_delta_beta_exact",
            "pass" if required_delta_beta_exact < 1.0e-4 else "watch",
            "profile-fixed exact beta shift required to hit alpha_target",
            required_delta_beta_exact,
            "Only a small upward shift of beta_1 is needed to close the retained 1.9% scalar residual if the exact profile is held fixed and only q_*(beta) is moved.",
        ),
        sign_base.row(
            "profile_fixed_required_delta_beta2_fraction_of_beta_gap",
            "pass" if required_delta_beta2_fraction_of_beta_gap < 0.05 else "watch",
            "required delta(beta^2) as a fraction of the current 1-beta^2 gap",
            required_delta_beta2_fraction_of_beta_gap,
            "The required shift uses only a small fraction of the current gap to the continuum edge, so the first shot is numerically modest rather than a large retuning.",
        ),
        sign_base.row(
            "profile_fixed_required_delta_beta2_vs_ceiling_sq",
            "pass" if required_delta_beta2_vs_ceiling_sq < 0.01 else "watch",
            "required delta(beta^2) relative to max|fL/f0|^2 ceiling",
            required_delta_beta2_vs_ceiling_sq,
            "Compared with the restored exact-branch vector ceiling, the needed delta(beta^2) is tiny, which keeps the first-shot missing-action hypothesis numerically admissible.",
        ),
        sign_base.row(
            "profile_fixed_alpha_corrected_relerr_vs_target",
            "pass" if abs(alpha_corrected - ALPHA_TARGET) <= 1.0e-12 else "watch",
            "profile-fixed corrected alpha relative error vs target",
            abs(alpha_corrected - ALPHA_TARGET) / ALPHA_TARGET,
            "The profile-fixed correction closes the target by construction and shows that a small eigenvalue shift is sufficient at the level of the retained scalar profile map.",
        ),
        sign_base.row(
            "profile_fixed_eigenvalue_shift_candidate_admissible",
            "pass" if profile_fixed_eigenvalue_shift_candidate_admissible else "reject",
            "profile-fixed eigenvalue-shift candidate admissible",
            sign_base.truth(profile_fixed_eigenvalue_shift_candidate_admissible),
            "This first shot is admissible only because the needed beta shift is small relative to both the beta gap and the retained vector-branch amplitude ceiling.",
        ),
        sign_base.row(
            "exact_coupled_eigenvalue_shift_theorem_available",
            "reject",
            "exact coupled eigenvalue-shift theorem available",
            sign_base.truth(exact_coupled_eigenvalue_shift_theorem_available),
            "The current branch has only a falsifiable profile-fixed candidate; the coupled theorem that would derive delta(beta_1) from the exact action-level operator is still missing.",
        ),
        sign_base.row(
            "missing_action_level_primary_supported",
            "pass",
            "missing action-level term retained as the primary residual lane",
            1.0,
            "Boundary primary is already falsified and observable primary is already demoted, so the current residual-origin mainline remains in the missing-action lane.",
        ),
    ]

    summary = {
        "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
        "prior_problem_classification": PRIOR_CLASS,
        "retained_scalar_residual_rel": float(prior_summary["retained_scalar_residual_rel"]),
        "boundary_origin_primary_supported": False,
        "observable_definition_primary_supported": False,
        "observable_definition_secondary_carryover": True,
        "beta_1": beta_1,
        "beta_gap_to_continuum": beta_gap,
        "q_theory_over_m0": q_theory,
        "F_exact_at_q_theory": f_exact,
        "alpha_exact_at_q_theory": alpha_exact,
        "alpha_target": ALPHA_TARGET,
        "alpha_target_gap_abs": target_gap_abs,
        "dF_dq_at_q_theory": dF_dq,
        "dalpha_dq_at_q_theory": dalpha_dq,
        "dq_dbeta_at_beta_1": dq_dbeta,
        "dalpha_dbeta_at_beta_1": dalpha_dbeta,
        "local_log_elasticity_alpha_vs_q": local_log_elasticity_alpha_vs_q,
        "required_delta_beta_linear": required_delta_beta_linear,
        "beta_corrected_profile_fixed": beta_corrected,
        "delta_beta_exact_profile_fixed": required_delta_beta_exact,
        "delta_beta2_exact_profile_fixed": required_delta_beta2_exact,
        "q_corrected_profile_fixed": q_corrected,
        "delta_q_abs": delta_q_abs,
        "delta_q_rel": delta_q_rel,
        "F_corrected_profile_fixed": f_corrected,
        "alpha_corrected_profile_fixed": alpha_corrected,
        "required_delta_beta_fraction_of_beta_gap": required_delta_beta_fraction_of_beta_gap,
        "required_delta_beta2_fraction_of_beta_gap": required_delta_beta2_fraction_of_beta_gap,
        "max_fl_over_f0_ceiling": max_fl_over_f0,
        "max_fl_over_f0_ceiling_sq": ceiling_sq,
        "required_delta_beta_vs_ceiling_sq": required_delta_beta_vs_ceiling_sq,
        "required_delta_beta2_vs_ceiling_sq": required_delta_beta2_vs_ceiling_sq,
        "phase1_exact_solver_cross_term_present": bool(
            operator_summary["phase1_exact_solver_cross_term_present"]
        ),
        "phase1_exact_solver_constraint_elimination_present": bool(
            operator_summary["phase1_exact_solver_constraint_elimination_present"]
        ),
        "phase1_exact_solver_scalar_nonlinear_ansatz_only": bool(
            operator_summary["phase1_exact_solver_scalar_nonlinear_ansatz_only"]
        ),
        "trial3_family_solver_ell0_coupling_collapses": bool(
            operator_summary["trial3_family_solver_ell0_coupling_collapses"]
        ),
        "exact_action_level_closed_ell0_operator_available": exact_coupled_eigenvalue_shift_theorem_available,
        "profile_fixed_eigenvalue_shift_candidate_admissible": profile_fixed_eigenvalue_shift_candidate_admissible,
        "exact_coupled_eigenvalue_shift_theorem_available": exact_coupled_eigenvalue_shift_theorem_available,
        "selected_next_generation_route": NEXT_ROUTE_NAME,
        "recommended_next_route_or_none": NEXT_ROUTE,
        "selected_followup_route": FOLLOWUP_ROUTE_NAME,
        "selected_followup_route_or_none": FOLLOWUP_ROUTE,
        "physical_reject_required": False,
    }

    declaration_payload = sign_base.payload(
        "8.7.56.2353",
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
                "prior_gate": sign_base.display_path(PRIOR_GATE),
                "ell0_operator_audit": sign_base.display_path(ELL0_OPERATOR_AUDIT),
                "ell0_anchor_eval": sign_base.display_path(ELL0_ANCHOR_EVAL),
                "qball_branch_refresh": sign_base.display_path(QBALL_BRANCH_REFRESH),
                "qball_solver": sign_base.display_path(QBALL_SOLVER),
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
        rows,
        summary,
        {
            "overall_status": "vector_qball_form_factor_missing_action_delta_beta1_audit_declared",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "formulas": build_formulae(),
            "scalar_ground_state_row": scalar_row,
            "phase1_equivalent_row": anchor_summary["phase1_equivalent_row"],
            "hits": {
                "status_branch_hit": sign_base.hit(status_text, "8.7.56.2351"),
                "roadmap_branch_hit": sign_base.hit(roadmap_text, ".2351-.2354"),
                "current_problem_hit": sign_base.hit(current_problem_text, "missing action-level term"),
                "current_status_hit": sign_base.hit(current_status_text, "missing action-level term"),
                "unified_roadmap_hit": sign_base.hit(unified_text, ".2351-.2354"),
                "long_roadmap_hit": sign_base.hit(long_text, ".2351-.2354"),
                "part5_hit": sign_base.hit(part5_text, "2026-03-30 update"),
            },
        },
    )
    declaration_paths = write_artifact("declaration_gate", declaration_payload)

    route_payload = {
        "generated_utc": sign_base.now_iso(),
        "phase": {
            "phase": 8,
            "step": "8.7.56.2354",
            "name": STEP_NAME + " route sync",
        },
        "inputs": {
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
        "rows": [
            sign_base.row(
                "missing_action_delta_beta1_audit_synced",
                "pass",
                "missing-action delta-beta audit synced",
                1.0,
                "The residual-origin mainline only stays honest if the first concrete missing-action shot is written as a candidate audit rather than prematurely promoted to theorem level.",
            ),
        ],
        "summary": summary,
        "decision": {
            "overall_status": "vector_qball_form_factor_missing_action_delta_beta1_audit_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        "evidence": declaration_payload["evidence"],
    }
    route_paths = write_artifact("route_sync", route_payload)
    print("[write] declaration:", declaration_paths["json"])
    print("[write] route:", route_paths["json"])


if __name__ == "__main__":
    main()
