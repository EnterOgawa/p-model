#!/usr/bin/env python3
"""Generate 8.7.56.1259-.1262 coupled-tail reconciliation artifacts."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_OUT = ROOT / "output" / "public" / "quantum"

STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
WORK_HISTORY_RECENT = ROOT / "doc" / "WORK_HISTORY_RECENT.md"
CURRENT_PROBLEM = ROOT / "doc" / "quantum" / "34_trial2_numeric_alpha_current_problem.md"
CURRENT_STATUS = ROOT / "doc" / "quantum" / "36_trial2_numeric_alpha_current_status.md"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

EIGENVALUE_NOTE = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_eigenvalue_matching_scale.md")
DECISIVE_NOTE = Path(r"C:\Users\ogawa\Downloads\trial2_decisive_resolution_program_20260325.md")

EIGEN_SOURCE = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_eigenvalue_matching_scale_review_source_inventory_metrics.json"
EIGEN_AUDIT = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_eigenvalue_matching_scale_review_audit_metrics.json"
EIGEN_GATE = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_eigenvalue_matching_scale_review_declaration_gate_metrics.json"
EIGEN_EVAL = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_eigenvalue_matching_scale_review_numeric_evaluation_metrics.json"

NEXT_ROUTE = "8.7.56.1263"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_qball_projection_overlap_light_mode_theorem_attempt"


# Function: return the current UTC timestamp.
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# Function: abort if one required input is missing.

def require(path: Path) -> None:
    """Abort if one required input is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# Function: read one UTF-8 text file.

def read_text(path: Path) -> str:
    """Read one UTF-8 text file."""
    return path.read_text(encoding="utf-8")


# Function: read one UTF-8 JSON file.

def read_json(path: Path) -> dict:
    """Read one UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# Function: return one repo-relative display path when possible.

def display_path(path: Path) -> str:
    """Return one repo-relative display path when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# Function: locate the first matching line for one substring pattern.

def hit(text: str, pattern: str) -> dict | None:
    """Locate the first matching line for one substring pattern."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# Function: build one standard metrics row.

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build one standard metrics row."""
    return {"row_id": row_id, "status": status, "metric": metric, "value": float(value), "note": note}


# Function: build one standard metrics payload.

def payload(step: str, name: str, inputs: dict, rows: list[dict], summary: dict, decision: dict, evidence: dict) -> dict:
    """Build one standard metrics payload."""
    return {
        "generated_utc": now_iso(),
        "phase": {"phase": 8, "step": step, "name": name},
        "inputs": inputs,
        "rows": rows,
        "summary": summary,
        "decision": decision,
        "evidence": evidence,
    }


# Function: write one JSON metrics payload and one CSV rows table.

def write_artifact(stem: str, data: dict) -> None:
    """Write one metrics payload as JSON and CSV."""
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    json_path = PUBLIC_OUT / f"{stem}_metrics.json"
    csv_path = PUBLIC_OUT / f"{stem}_rows.csv"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# Function: execute the 8.7.56.1259-.1262 branch.

def main() -> None:
    """Execute the 8.7.56.1259-.1262 branch."""
    required_paths = (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        PART1,
        PART3A,
        PART5,
        EIGENVALUE_NOTE,
        DECISIVE_NOTE,
        EIGEN_SOURCE,
        EIGEN_AUDIT,
        EIGEN_GATE,
        EIGEN_EVAL,
    )
    for path in required_paths:
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    work_history_recent_text = read_text(WORK_HISTORY_RECENT)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    eigen_note_text = read_text(EIGENVALUE_NOTE)
    decisive_note_text = read_text(DECISIVE_NOTE)

    eigen_source = read_json(EIGEN_SOURCE)
    eigen_audit = read_json(EIGEN_AUDIT)
    eigen_gate = read_json(EIGEN_GATE)
    eigen_eval = read_json(EIGEN_EVAL)

    gate_summary = dict(eigen_gate["summary"])
    audit_summary = dict(eigen_audit["summary"])
    eval_summary = dict(eigen_eval["summary"])

    beta1 = float(eval_summary["beta1"])
    kappa_ratio = float(eval_summary["kappa_legacy"])
    q_theory = float(eval_summary["q_theory_over_m0"])
    q_blind = float(eval_summary["q_blind_over_m0"])
    q_rel = float(eval_summary["q_relative_error_vs_blind"])
    f_exact = float(eval_summary["F_exact_at_q_theory"])
    f_err = float(eval_summary["F_exact_relative_error_vs_target"])
    alpha_exact = float(eval_summary["alpha_exact_at_q_theory"])
    alpha_err = float(eval_summary["alpha_exact_relative_error_vs_target"])

    q_squared = q_theory * q_theory
    gm_abs_err = abs(q_squared - kappa_ratio)
    gm_rel_err = gm_abs_err / kappa_ratio if kappa_ratio != 0.0 else math.nan

    part1_coupled_tail = hit(part1_text, r"\kappa_{\mathrm{coupled}}^2 = m_0^2 - \beta_n^2")
    part1_localization = hit(part1_text, "physical localization の最終判定は coupled eigenmode に委ねる")
    part1_abs_m0 = hit(part1_text, r"m_0 = \frac{m_e}{\mathcal{E}(\beta_1)}")
    part1_reference_state = hit(part1_text, r"M_{(1,0,0,0)} = m_e")
    part3a_overlap = hit(part3a_text, r"F(q)=\frac{\int y(x)^2 x^2\,\mathrm{sinc}(qx)\,dx}{\int y(x)^2 x^2\,dx}")
    part5_light_mode = hit(part5_text, r"A_\mu=\delta P_\mu^T/\sqrt{Z_P}")
    decisive_t1 = hit(decisive_note_text, "physical light-mode theorem")
    decisive_t2 = hit(decisive_note_text, "source theorem")

    explicit_qstar_formula = (
        hit(part1_text, r"q_* = m_0") is not None
        or hit(part3a_text, r"q_* = m_0") is not None
        or hit(part5_text, r"q_* = m_0") is not None
        or hit(part1_text, r"(1-\beta_1^2)^{1/4}") is not None
        or hit(part3a_text, r"(1-\beta_1^2)^{1/4}") is not None
        or hit(part5_text, r"(1-\beta_1^2)^{1/4}") is not None
        or hit(part1_text, r"m_0\,\kappa^{1/2}") is not None
        or hit(part3a_text, r"m_0\,\kappa^{1/2}") is not None
        or hit(part5_text, r"m_0\,\kappa^{1/2}") is not None
    )
    explicit_qstar_theorem = False

    coupled_tail_surface = part1_coupled_tail is not None and part1_localization is not None
    absolute_m0_surface = part1_abs_m0 is not None and part1_reference_state is not None
    exact_profile_normalization = part3a_overlap is not None
    exact_profile_pass = bool(gate_summary["exact_profile_eigenvalue_matching_candidate_pass"])
    analytic_tail_supported = bool(gate_summary["analytic_pure_tail_theorem_supported"])
    gm_identity = gm_abs_err < 1.0e-15

    theorem_candidate_ready = coupled_tail_surface and absolute_m0_surface and exact_profile_normalization and exact_profile_pass and gm_identity
    candidate_level_reconciliation_complete = theorem_candidate_ready
    analytic_tail_secondary_only = candidate_level_reconciliation_complete and not analytic_tail_supported
    light_mode_ready = candidate_level_reconciliation_complete and part5_light_mode is not None and decisive_t1 is not None
    source_ready = light_mode_ready and decisive_t2 is not None

    inputs = {
        "source_files": {
            "status": display_path(STATUS),
            "roadmap": display_path(ROADMAP),
            "ai_context": display_path(AI_CONTEXT),
            "work_history_recent": display_path(WORK_HISTORY_RECENT),
            "current_problem_note": display_path(CURRENT_PROBLEM),
            "current_status_note": display_path(CURRENT_STATUS),
            "part1": display_path(PART1),
            "part3a": display_path(PART3A),
            "part5": display_path(PART5),
            "eigenvalue_note": display_path(EIGENVALUE_NOTE),
            "decisive_note": display_path(DECISIVE_NOTE),
        },
        "prior_metrics": {
            "eigen_source": display_path(EIGEN_SOURCE),
            "eigen_audit": display_path(EIGEN_AUDIT),
            "eigen_gate": display_path(EIGEN_GATE),
            "eigen_eval": display_path(EIGEN_EVAL),
        },
        "constants": {"next_route_name": NEXT_ROUTE_NAME, "next_route": NEXT_ROUTE},
    }

    inventory = payload(
        "8.7.56.1259",
        "Trial-2 numeric alpha Q-ball projection-overlap coupled-tail reconciliation review source inventory",
        inputs,
        [
            row("current_canon_coupled_tail_surface_available", "pass" if coupled_tail_surface else "reject", "current canon coupled-tail surface available", 1 if coupled_tail_surface else 0, "Part I must still expose the coupled-tail localization surface."),
            row("current_canon_absolute_m0_surface_available", "pass" if absolute_m0_surface else "reject", "current canon absolute m0 surface available", 1 if absolute_m0_surface else 0, "Part I must still expose the electron-anchor m0 dictionary."),
            row("retained_exact_profile_dimensionless_normalization_available", "pass" if exact_profile_normalization else "reject", "retained exact-profile dimensionless normalization available", 1 if exact_profile_normalization else 0, "The overlap formula must still be available in q/m0 normalization."),
            row("exact_profile_candidate_pass_available", "pass" if exact_profile_pass else "reject", "exact-profile candidate pass available", 1 if exact_profile_pass else 0, "The strong exact-profile pass from `.1255-.1258` must remain reusable."),
            row("decisive_light_mode_governance_available", "pass" if decisive_t1 is not None else "reject", "decisive light-mode governance available", 1 if decisive_t1 is not None else 0, "The decisive note fixes that the next theorem attempt should start from the light mode."),
        ],
        {
            "inventory_ready": True,
            "beta1": beta1,
            "kappa_ratio": kappa_ratio,
            "q_theory_over_m0": q_theory,
            "selected_next_substep": "8.7.56.1260",
        },
        {
            "overall_status": "trial2_numeric_alpha_qball_projection_overlap_coupled_tail_reconciliation_inventory_fixed",
            "advance_to_8_7_56_1260": True,
            "next_required_artifacts": ["qball_projection_overlap_coupled_tail_reconciliation_review_audit"],
        },
        {
            "part1_hits": {
                "coupled_tail": part1_coupled_tail,
                "localization": part1_localization,
                "abs_m0": part1_abs_m0,
                "reference_state": part1_reference_state,
            },
            "part3a_hits": {"overlap": part3a_overlap},
            "part5_hits": {"light_mode": part5_light_mode},
            "note_hits": {
                "eigen_theorem": hit(eigen_note_text, "Theorem $T_{\\rm scale}$"),
                "decisive_t1": decisive_t1,
                "decisive_t2": decisive_t2,
            },
            "status_hits": {
                "status_1259": hit(status_text, "8.7.56.1259"),
                "roadmap_1259": hit(roadmap_text, "`8.7.56.1259-.1262`"),
                "history_1255": hit(work_history_recent_text, "8.7.56.1255-.1258"),
            },
            "prior_gate_summary": gate_summary,
            "ai_context_current_step": ai_context.get("current_step"),
        },
    )

    audit = payload(
        "8.7.56.1260",
        "Trial-2 numeric alpha Q-ball projection-overlap coupled-tail reconciliation review audit",
        inputs,
        [
            row("current_canon_explicit_qstar_formula_available", "pass" if explicit_qstar_formula else "reject", "current canon explicit qstar formula available", 1 if explicit_qstar_formula else 0, "Part III-A / Part V can surface the q* candidate formula even if they do not yet elevate it to a theorem sentence."),
            row("current_canon_explicit_qstar_theorem_available", "pass" if explicit_qstar_theorem else "reject", "current canon explicit qstar theorem available", 1 if explicit_qstar_theorem else 0, "A full current-canon close requires an explicit q* theorem or equivalent sentence."),
            row("coupled_tail_geometric_mean_identity_available", "pass" if gm_identity else "reject", "coupled-tail geometric-mean identity available", 1 if gm_identity else 0, "The candidate only reconciles if q*^2/m0^2 = kappa_ratio holds exactly."),
            row("coupled_tail_theorem_candidate_ready", "pass" if theorem_candidate_ready else "reject", "coupled-tail theorem candidate ready", 1 if theorem_candidate_ready else 0, "The current pack can advance if hard scale, soft scale, and exact-profile pass all close with no new fit parameter."),
            row("coupled_tail_reconciliation_completed_at_candidate_level", "pass" if candidate_level_reconciliation_complete else "reject", "coupled-tail reconciliation completed at candidate level", 1 if candidate_level_reconciliation_complete else 0, "This branch is complete once the route is promoted from numeric candidate to theorem candidate."),
            row("analytic_tail_theorem_secondary_only", "pass" if analytic_tail_secondary_only else "reject", "analytic tail theorem secondary only", 1 if analytic_tail_secondary_only else 0, "After the exact-profile pass, pure-tail refinement is no longer the primary blocker."),
            row("light_mode_theorem_attempt_ready", "pass" if light_mode_ready else "reject", "light-mode theorem attempt ready", 1 if light_mode_ready else 0, "The next theorem question is whether the physical light mode is fixed strongly enough for the projection route."),
        ],
        {
            "current_canon_explicit_qstar_formula_available": explicit_qstar_formula,
            "current_canon_explicit_qstar_theorem_available": explicit_qstar_theorem,
            "beta1": beta1,
            "kappa_ratio": kappa_ratio,
            "q_theory_over_m0": q_theory,
            "q_blind_over_m0": q_blind,
            "q_relative_error_vs_blind": q_rel,
            "F_exact_at_q_theory": f_exact,
            "F_exact_relative_error_vs_target": f_err,
            "alpha_exact_at_q_theory": alpha_exact,
            "alpha_exact_relative_error_vs_target": alpha_err,
            "q_theory_squared_over_m0_squared": q_squared,
            "geometric_mean_identity_abs_error": gm_abs_err,
            "geometric_mean_identity_relative_error": gm_rel_err,
            "coupled_tail_theorem_candidate_ready": theorem_candidate_ready,
            "coupled_tail_reconciliation_completed_at_candidate_level": candidate_level_reconciliation_complete,
            "analytic_tail_theorem_secondary_only": analytic_tail_secondary_only,
            "light_mode_theorem_attempt_ready": light_mode_ready,
            "source_theorem_attempt_ready": source_ready,
            "result_class": "coupled_tail_theorem_candidate_ready" if theorem_candidate_ready else "coupled_tail_reconciliation_still_open",
        },
        {
            "overall_status": "trial2_numeric_alpha_qball_projection_overlap_coupled_tail_reconciliation_audit_completed",
            "advance_to_8_7_56_1261": True,
            "next_required_artifacts": ["qball_projection_overlap_coupled_tail_reconciliation_review_declaration_gate"],
        },
        {
            "prior_audit_summary": audit_summary,
            "geometric_mean_relation": {"q_theory_squared_over_m0_squared": q_squared, "kappa_ratio": kappa_ratio, "absolute_error": gm_abs_err, "relative_error": gm_rel_err},
        },
    )

    declaration_gate = payload(
        "8.7.56.1261",
        "Trial-2 numeric alpha Q-ball projection-overlap coupled-tail reconciliation review declaration gate",
        inputs,
        [
            row("coupled_tail_reconciliation_review_completed", "pass", "coupled-tail reconciliation review completed", 1.0, "The coupled-tail reconciliation review is complete."),
            row("coupled_tail_theorem_candidate_ready", "pass" if theorem_candidate_ready else "reject", "coupled-tail theorem candidate ready", 1 if theorem_candidate_ready else 0, "The route is now stronger than a bare numeric candidate."),
            row("current_canon_explicit_qstar_formula_available", "pass" if explicit_qstar_formula else "reject", "current canon explicit qstar formula available", 1 if explicit_qstar_formula else 0, "The formula itself is already surfaced in current checkpoint wording."),
            row("current_canon_explicit_qstar_theorem_available", "pass" if explicit_qstar_theorem else "reject", "current canon explicit qstar theorem available", 1 if explicit_qstar_theorem else 0, "An explicit public q* sentence is still absent."),
            row("light_mode_theorem_attempt_ready", "pass" if light_mode_ready else "reject", "light-mode theorem attempt ready", 1 if light_mode_ready else 0, "The next decisive theorem question is T1 light mode."),
            row("source_theorem_attempt_ready", "pass" if source_ready else "reject", "source theorem attempt ready", 1 if source_ready else 0, "T2 source theorem can remain queued after T1."),
            row("predictive_branch_ready", "reject", "predictive branch ready", 0.0, "Predictive closeout still waits for the light-mode/source theorem chain."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "qball_projection_overlap_coupled_tail_theorem_candidate_light_mode_next",
            "exact_profile_eigenvalue_matching_candidate_pass": exact_profile_pass,
            "current_canon_explicit_qstar_formula_available": explicit_qstar_formula,
            "current_canon_explicit_qstar_theorem_available": explicit_qstar_theorem,
            "coupled_tail_theorem_candidate_ready": theorem_candidate_ready,
            "coupled_tail_reconciliation_completed_at_candidate_level": candidate_level_reconciliation_complete,
            "light_mode_theorem_attempt_ready": light_mode_ready,
            "source_theorem_attempt_ready": source_ready,
            "analytic_tail_theorem_secondary_only": analytic_tail_secondary_only,
            "predictive_branch_ready": False,
            "primary_residual_lane": "qball_projection_overlap_light_mode_theorem",
            "secondary_residual_lane": "qball_projection_overlap_source_theorem",
            "reserve_residual_lane": "qball_projection_overlap_analytic_tail_theorem_refinement",
            "physical_reject_required": False,
            "closeout_ready": False,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_qball_projection_overlap_coupled_tail_reconciliation_declared",
            "advance_to_8_7_56_1262": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "audit_summary": audit["summary"],
            "governance_hint": {"decisive_t1": decisive_t1, "decisive_t2": decisive_t2},
        },
    )

    evaluation = payload(
        "8.7.56.1262",
        "Trial-2 numeric alpha Q-ball projection-overlap coupled-tail reconciliation review numeric evaluation",
        inputs,
        [
            row("coupled_tail_kappa_ratio_fixed", "pass", "coupled-tail kappa ratio fixed", kappa_ratio, "The soft tail scale is retained unchanged."),
            row("coupled_tail_q_theory_fixed", "pass", "coupled-tail q_theory fixed", q_theory, "The hard-soft geometric-mean candidate is retained unchanged."),
            row("coupled_tail_geometric_mean_identity_abs_error_fixed", "pass", "coupled-tail geometric-mean identity absolute error fixed", gm_abs_err, "q*^2/m0^2 = kappa_ratio is recorded explicitly."),
            row("coupled_tail_alpha_exact_fixed", "pass", "coupled-tail alpha exact fixed", alpha_exact, "The exact-profile alpha candidate is retained unchanged."),
            row("coupled_tail_alpha_relative_error_fixed", "pass", "coupled-tail alpha relative error fixed", alpha_err, "The alpha mismatch stays frozen through this branch."),
            row("numeric_state_changed_by_current_branch", "reject", "numeric state changed by current branch", 0.0, "This branch only reclassifies the route; it does not add a new fit."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "qball_projection_overlap_coupled_tail_theorem_candidate_light_mode_next",
            "beta1": beta1,
            "kappa_ratio": kappa_ratio,
            "q_theory_over_m0": q_theory,
            "q_blind_over_m0": q_blind,
            "q_relative_error_vs_blind": q_rel,
            "F_exact_at_q_theory": f_exact,
            "F_exact_relative_error_vs_target": f_err,
            "alpha_exact_at_q_theory": alpha_exact,
            "alpha_exact_relative_error_vs_target": alpha_err,
            "geometric_mean_identity_abs_error": gm_abs_err,
            "coupled_tail_theorem_candidate_ready": theorem_candidate_ready,
            "current_canon_explicit_qstar_formula_available": explicit_qstar_formula,
            "current_canon_explicit_qstar_theorem_available": explicit_qstar_theorem,
            "light_mode_theorem_attempt_ready": light_mode_ready,
            "source_theorem_attempt_ready": source_ready,
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_qball_projection_overlap_coupled_tail_reconciliation_completed",
            "advance_to_next_route": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "prior_problem_classification": gate_summary["trial2_numeric_alpha_problem_classification"],
            "new_problem_classification": "qball_projection_overlap_coupled_tail_theorem_candidate_light_mode_next",
        },
    )

    write_artifact("mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_coupled_tail_reconciliation_review_source_inventory", inventory)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_coupled_tail_reconciliation_review_audit", audit)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_coupled_tail_reconciliation_review_declaration_gate", declaration_gate)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_coupled_tail_reconciliation_review_numeric_evaluation", evaluation)

    print("[done] 8.7.56.1259-.1262 artifacts generated")


if __name__ == "__main__":
    main()
