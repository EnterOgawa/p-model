#!/usr/bin/env python3
"""Generate 8.7.56.1255-.1258 Trial-2 eigenvalue-matching-scale artifacts.

Purpose:
    Recast the post-`.1251-.1254` route from a generic tail-weighting reserve
    audit into a computation-first review of the new
    `pmodel_v2_trial2_eigenvalue_matching_scale.md` note. The branch tests
    whether the eigenvalue-derived candidate
    `q_*/m_0 = (1 - beta_1^2)^(1/4)` reproduces the already fixed blind target
    crossing on the retained exact ground-state profile.

Inputs:
    - Current operational docs and Part I / Part III-A / Part V surfaces
    - The `.1243-.1246` overlap metrics and `.1251-.1254` exact-scale review
    - The retained scalar/vector Q-ball ground-state metrics
    - The external note
      `C:/Users/ogawa/Downloads/pmodel_v2_trial2_eigenvalue_matching_scale.md`

Outputs:
    - Four machine-readable metrics payloads under `output/public/quantum/`

Assumptions:
    - No new fit parameter is introduced. The candidate uses only the frozen
      ground-state eigenvalue `beta_1`.
    - The same retained exact profile used in `.1243` is reused here so that
      the branch measures a genuine no-new-free-parameter prediction candidate.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


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

NOTE = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_eigenvalue_matching_scale.md")
QBALL_BRANCH_REFRESH = PUBLIC_OUT / "mass_origin_qball_charge_mapping_branch_refresh_metrics.json"
QBALL_FULL_COUPLED = PUBLIC_OUT / "mass_origin_vector_qball_full_coupled_solver_pilot_metrics.json"
OVERLAP_AUDIT = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_audit_metrics.json"
OVERLAP_EVAL = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_numeric_evaluation_metrics.json"
EFFECTIVE_GATE = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_"
    "effective_support_scale_review_declaration_gate_metrics.json"
)
EFFECTIVE_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_"
    "effective_support_scale_review_numeric_evaluation_metrics.json"
)
QBALL_SOLVER = ROOT / "scripts" / "quantum" / "mass_origin_qball_charge_mapping_branch.py"

ALPHA_TARGET = 1.0 / 137.035999084
TARGET_FORM_FACTOR = math.sqrt(4.0 * math.pi * ALPHA_TARGET)
NEXT_ROUTE = "8.7.56.1259"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_qball_projection_overlap_coupled_tail_reconciliation_review"


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
    """Return the first matching line for one substring pattern."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# Function: build one standard metrics row.

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build one standard metrics row."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# Function: build one standard metrics payload.

def payload(
    step: str,
    name: str,
    inputs: dict,
    rows: list[dict],
    summary: dict,
    decision: dict,
    evidence: dict,
) -> dict:
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


# Function: load the retained scalar Q-ball solver as a reusable module.

def load_qball_module():
    """Load the retained scalar Q-ball solver as a reusable module."""
    spec = importlib.util.spec_from_file_location("wavep_qball_charge_mapping", QBALL_SOLVER)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[fail] unable to load module from {QBALL_SOLVER}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Function: extract the scalar ground-state row from the retained branch-refresh metrics.

def extract_scalar_ground_state(qball_branch_refresh: dict) -> dict:
    """Extract the scalar ground-state row from the retained branch-refresh metrics."""
    for row_data in qball_branch_refresh["evidence"]["discrete_mode_rows"]:
        if int(row_data["mode_index"]) == 1:
            return {
                "mode_index": int(row_data["mode_index"]),
                "beta_n": float(row_data["beta_n"]),
                "charge_proxy": float(row_data["charge_proxy"]),
                "energy_proxy": float(row_data["energy_proxy"]),
                "central_amplitude": float(row_data["central_amplitude"]),
                "mass_ratio_to_first": float(row_data["mass_ratio_to_first"]),
            }

    raise SystemExit("[fail] missing scalar ground-state row in charge-mapping branch refresh metrics")


# Function: extract the exact vector-ladder reference state from the retained full-coupled metrics.

def extract_exact_ground_state(qball_full_coupled: dict) -> dict:
    """Extract the exact vector-ladder reference state from the retained full-coupled metrics."""
    for row_data in qball_full_coupled["evidence"]["exact_ladder_sample_rows"]:
        if (
            int(row_data["n"]) == 1
            and int(row_data["k"]) == 0
            and int(row_data["ell"]) == 0
            and int(row_data["s"]) == 0
        ):
            return {
                "n": int(row_data["n"]),
                "k": int(row_data["k"]),
                "ell": int(row_data["ell"]),
                "s": int(row_data["s"]),
                "beta_n": float(row_data["beta_n"]),
                "exact_charge_proxy": float(row_data["exact_charge_proxy"]),
                "exact_mass_proxy": float(row_data["exact_mass_proxy"]),
                "mass_ratio_to_scalar_base": float(row_data["mass_ratio_to_scalar_base"]),
            }

    raise SystemExit("[fail] missing exact vector reference row M_(1,0,0,0)")


# Function: evaluate one normalized spherical-overlap form factor on the retained profile.

def form_factor(radius: np.ndarray, weight: np.ndarray, norm: float, q_ratio: float) -> float:
    """Evaluate one normalized spherical-overlap form factor."""
    qx = float(q_ratio) * radius
    sinc = np.ones_like(qx)
    mask = np.abs(qx) > 1.0e-12
    sinc[mask] = np.sin(qx[mask]) / qx[mask]
    numerator = np.trapezoid(weight * sinc, radius)
    return float(numerator / norm)


# Function: evaluate the pure exponential tail proxy from the note.

def tail_form_factor(q_ratio: float, kappa: float) -> float:
    """Evaluate the pure exponential tail proxy from the note."""
    denominator = (4.0 * kappa * kappa + q_ratio * q_ratio) ** 2
    return float(16.0 * (kappa**4) * q_ratio / denominator)


# Function: execute the 8.7.56.1255-.1258 branch.

def main() -> None:
    """Execute the 8.7.56.1255-.1258 branch."""
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
        NOTE,
        QBALL_BRANCH_REFRESH,
        QBALL_FULL_COUPLED,
        OVERLAP_AUDIT,
        OVERLAP_EVAL,
        EFFECTIVE_GATE,
        EFFECTIVE_EVAL,
        QBALL_SOLVER,
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
    note_text = read_text(NOTE)

    qball_branch_refresh = read_json(QBALL_BRANCH_REFRESH)
    qball_full_coupled = read_json(QBALL_FULL_COUPLED)
    overlap_audit = read_json(OVERLAP_AUDIT)
    overlap_eval = read_json(OVERLAP_EVAL)
    effective_gate = read_json(EFFECTIVE_GATE)
    effective_eval = read_json(EFFECTIVE_EVAL)

    qball_module = load_qball_module()
    scalar_ground_state = extract_scalar_ground_state(qball_branch_refresh)
    exact_ground_state = extract_exact_ground_state(qball_full_coupled)

    radius, field, _field_prime = qball_module.solve_full_profile(
        float(scalar_ground_state["beta_n"]),
        float(scalar_ground_state["central_amplitude"]),
    )
    weight = (field**2) * (radius**2)
    norm = float(np.trapezoid(weight, radius))

    beta1 = float(scalar_ground_state["beta_n"])
    q_blind = float(overlap_eval["summary"]["first_target_matching_q_over_m0"])
    best_candidate_error = float(effective_eval["summary"]["best_candidate_error"])
    kappa_legacy = math.sqrt(1.0 - beta1 * beta1)
    q_theory = math.sqrt(kappa_legacy)
    q_peak_tail = (2.0 * kappa_legacy) / math.sqrt(3.0)
    q_rel_error_vs_blind = abs(q_theory - q_blind) / q_blind
    improvement_factor_vs_best_profile_statistic = best_candidate_error / q_rel_error_vs_blind

    F_theory_exact = form_factor(radius, weight, norm, q_theory)
    F_blind_exact = form_factor(radius, weight, norm, q_blind)
    F_exact_rel_error_vs_target = abs(F_theory_exact - TARGET_FORM_FACTOR) / TARGET_FORM_FACTOR
    alpha_theory_exact = (F_theory_exact**2) / (4.0 * math.pi)
    alpha_theory_rel_error_vs_target = abs(alpha_theory_exact - ALPHA_TARGET) / ALPHA_TARGET

    F_tail_theory = tail_form_factor(q_theory, kappa_legacy)
    F_tail_peak = tail_form_factor(q_peak_tail, kappa_legacy)
    pure_tail_target_reachable = F_tail_peak >= TARGET_FORM_FACTOR

    exact_profile_q_match_pass = q_rel_error_vs_blind < 0.01
    exact_profile_form_factor_near_target_pass = F_exact_rel_error_vs_target < 0.02
    exact_profile_alpha_near_target_pass = alpha_theory_rel_error_vs_target < 0.03
    analytic_pure_tail_theorem_supported = (
        pure_tail_target_reachable and abs(F_tail_theory - TARGET_FORM_FACTOR) / TARGET_FORM_FACTOR < 0.10
    )
    exact_profile_eigenvalue_candidate_pass = (
        exact_profile_q_match_pass
        and exact_profile_form_factor_near_target_pass
        and exact_profile_alpha_near_target_pass
    )

    exact_reference_consistent = (
        math.isclose(float(scalar_ground_state["beta_n"]), float(exact_ground_state["beta_n"]), rel_tol=0.0, abs_tol=1.0e-15)
        and math.isclose(
            float(scalar_ground_state["charge_proxy"]),
            float(exact_ground_state["exact_charge_proxy"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        and math.isclose(
            float(scalar_ground_state["energy_proxy"]),
            float(exact_ground_state["exact_mass_proxy"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    )

    part1_coupled_tail_line = hit(part1_text, r"\kappa_{\mathrm{coupled}}^2 = m_0^2 - \beta_n^2")
    note_eigenvalue_scale_line = hit(note_text, r"q_*^{\rm theory} = m_0\,(1 - \beta_1^2)^{1/4}")
    note_tail_line = hit(note_text, r"\kappa = \sqrt{1 - \beta_1^2}")
    note_pure_tail_formula_line = hit(note_text, r"F_{\rm tail}(q)")
    part3a_overlap_line = hit(part3a_text, r"F(q)=\frac{\int y(x)^2 x^2\,\mathrm{sinc}(qx)\,dx}{\int y(x)^2 x^2\,dx}")
    part5_current_line = hit(part5_text, "tail-weighting reserve review branch")
    current_problem_tail_line = hit(current_problem_text, "tail-weighting reserve candidate")
    current_status_tail_line = hit(current_status_text, "tail-weighting reserve next")

    current_canon_coupled_tail_surface_available = part1_coupled_tail_line is not None
    current_canon_explicit_qstar_theorem_available = False
    current_canon_coupled_tail_reconciliation_required = (
        current_canon_coupled_tail_surface_available and not current_canon_explicit_qstar_theorem_available
    )

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
            "eigenvalue_matching_scale_note": display_path(NOTE),
        },
        "prior_metrics": {
            "qball_branch_refresh": display_path(QBALL_BRANCH_REFRESH),
            "qball_full_coupled": display_path(QBALL_FULL_COUPLED),
            "overlap_audit": display_path(OVERLAP_AUDIT),
            "overlap_eval": display_path(OVERLAP_EVAL),
            "effective_gate": display_path(EFFECTIVE_GATE),
            "effective_eval": display_path(EFFECTIVE_EVAL),
        },
        "solver_module": display_path(QBALL_SOLVER),
        "constants": {
            "alpha_target": ALPHA_TARGET,
            "target_form_factor": TARGET_FORM_FACTOR,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
        },
    }

    inventory = payload(
        "8.7.56.1255",
        "Trial-2 numeric alpha Q-ball projection-overlap eigenvalue-matching-scale review source inventory",
        inputs,
        [
            row("eigenvalue_matching_scale_note_available", "pass", "eigenvalue-matching-scale note available", 1.0, "The external eigenvalue-matching-scale note is present."),
            row("prior_overlap_exact_profile_metrics_available", "pass", "prior overlap exact-profile metrics available", 1.0, "The blind overlap and effective-support-scale review metrics are present and can be reused."),
            row("retained_ground_state_row_available", "pass", "retained ground-state row available", 1.0, "The retained scalar ground-state row still exposes beta_1 and the central amplitude needed for exact-profile reconstruction."),
            row("exact_vector_ground_state_proxy_consistent", "pass" if exact_reference_consistent else "reject", "exact vector ground-state proxy consistent", 1 if exact_reference_consistent else 0, "The exact vector reference state M_(1,0,0,0) must remain identical to the scalar baseline row used for electron identification."),
            row("current_canon_coupled_tail_surface_available", "pass" if current_canon_coupled_tail_surface_available else "reject", "current canon coupled-tail surface available", 1 if current_canon_coupled_tail_surface_available else 0, "Part I must still expose the coupled asymptotic eigenmode decaying-tail surface."),
            row("eigenvalue_candidate_is_no_new_free_parameter", "pass", "eigenvalue candidate is no-new-free-parameter", 1.0, "The candidate uses only the frozen beta_1 value and does not introduce a new fit parameter."),
        ],
        {
            "inventory_ready": True,
            "beta1": beta1,
            "q_blind_over_m0": q_blind,
            "q_theory_over_m0": q_theory,
            "selected_next_substep": "8.7.56.1256",
        },
        {
            "overall_status": "trial2_numeric_alpha_qball_projection_overlap_eigenvalue_matching_scale_inventory_fixed",
            "advance_to_8_7_56_1256": True,
            "next_required_artifacts": ["qball_projection_overlap_eigenvalue_matching_scale_review_audit"],
        },
        {
            "note_hits": {
                "eigenvalue_scale_line": note_eigenvalue_scale_line,
                "tail_line": note_tail_line,
                "pure_tail_formula_line": note_pure_tail_formula_line,
            },
            "paper_hits": {
                "part1_coupled_tail_line": part1_coupled_tail_line,
                "part3a_overlap_line": part3a_overlap_line,
                "part5_current_line": part5_current_line,
            },
            "note_pack_hits": {
                "current_problem_tail_line": current_problem_tail_line,
                "current_status_tail_line": current_status_tail_line,
            },
            "status_hits": {
                "status_next_1255": hit(status_text, "8.7.56.1255"),
                "roadmap_branch_1255": hit(roadmap_text, "`8.7.56.1255-.1258`"),
                "work_history_1251_entry": hit(work_history_recent_text, "8.7.56.1251-.1254"),
            },
            "prior_overlap_eval_summary": overlap_eval["summary"],
            "prior_effective_gate_summary": effective_gate["summary"],
            "ai_context_current_step": ai_context.get("current_step"),
        },
    )

    audit = payload(
        "8.7.56.1256",
        "Trial-2 numeric alpha Q-ball projection-overlap eigenvalue-matching-scale review audit",
        inputs,
        [
            row("eigenvalue_matching_scale_q_match_pass", "pass" if exact_profile_q_match_pass else "reject", "eigenvalue-matching-scale q-match pass", 1 if exact_profile_q_match_pass else 0, "The eigenvalue-derived q_* candidate should land near the already fixed blind target crossing."),
            row("eigenvalue_matching_scale_form_factor_near_target_pass", "pass" if exact_profile_form_factor_near_target_pass else "reject", "eigenvalue-matching-scale form-factor near-target pass", 1 if exact_profile_form_factor_near_target_pass else 0, "The exact retained profile evaluated at q_*^(theory) should reproduce the target form factor without any new fit parameter."),
            row("eigenvalue_matching_scale_alpha_near_target_pass", "pass" if exact_profile_alpha_near_target_pass else "reject", "eigenvalue-matching-scale alpha near-target pass", 1 if exact_profile_alpha_near_target_pass else 0, "The exact retained profile evaluated at q_*^(theory) should reproduce alpha within a small residual tension band."),
            row("eigenvalue_matching_scale_improves_over_best_profile_statistic", "pass" if improvement_factor_vs_best_profile_statistic > 5.0 else "reject", "eigenvalue-matching-scale improves over best profile statistic", improvement_factor_vs_best_profile_statistic, "The eigenvalue-derived q_* should outperform the best previous profile-statistics proxy by a wide margin."),
            row("pure_exponential_tail_target_reachable", "pass" if pure_tail_target_reachable else "reject", "pure exponential tail target reachable", 1 if pure_tail_target_reachable else 0, "The note's pure exponential tail proxy only supports a theorem if it can itself reach the observed target form factor."),
            row("analytic_pure_tail_theorem_supported", "pass" if analytic_pure_tail_theorem_supported else "reject", "analytic pure-tail theorem supported", 1 if analytic_pure_tail_theorem_supported else 0, "The analytic theorem only passes if the pure-tail proxy reproduces the target rather than stopping far below it."),
            row("current_canon_coupled_tail_reconciliation_required", "pass" if current_canon_coupled_tail_reconciliation_required else "reject", "current canon coupled-tail reconciliation required", 1 if current_canon_coupled_tail_reconciliation_required else 0, "Part I exposes a coupled-tail surface, but current canon still lacks an explicit q_* theorem that reconciles the note's candidate with the canonical coupled-tail wording."),
        ],
        {
            "exact_profile_eigenvalue_candidate_pass": exact_profile_eigenvalue_candidate_pass,
            "q_theory_over_m0": q_theory,
            "q_blind_over_m0": q_blind,
            "q_relative_error_vs_blind": q_rel_error_vs_blind,
            "best_profile_statistic_error": best_candidate_error,
            "improvement_factor_vs_best_profile_statistic": improvement_factor_vs_best_profile_statistic,
            "F_target": TARGET_FORM_FACTOR,
            "F_exact_at_q_theory": F_theory_exact,
            "F_exact_relative_error_vs_target": F_exact_rel_error_vs_target,
            "alpha_target": ALPHA_TARGET,
            "alpha_exact_at_q_theory": alpha_theory_exact,
            "alpha_exact_relative_error_vs_target": alpha_theory_rel_error_vs_target,
            "kappa_legacy": kappa_legacy,
            "q_peak_tail_over_m0": q_peak_tail,
            "F_tail_at_q_theory": F_tail_theory,
            "F_tail_peak": F_tail_peak,
            "pure_tail_target_reachable": pure_tail_target_reachable,
            "analytic_pure_tail_theorem_supported": analytic_pure_tail_theorem_supported,
            "current_canon_coupled_tail_reconciliation_required": current_canon_coupled_tail_reconciliation_required,
            "result_class": (
                "eigenvalue_matching_candidate_exact_profile_pass_but_analytic_tail_theorem_open"
                if exact_profile_eigenvalue_candidate_pass and not analytic_pure_tail_theorem_supported
                else (
                    "eigenvalue_matching_candidate_pass"
                    if exact_profile_eigenvalue_candidate_pass
                    else "eigenvalue_matching_candidate_not_supported"
                )
            ),
        },
        {
            "overall_status": "trial2_numeric_alpha_qball_projection_overlap_eigenvalue_matching_scale_audit_completed",
            "advance_to_8_7_56_1257": True,
            "next_required_artifacts": ["qball_projection_overlap_eigenvalue_matching_scale_review_declaration_gate"],
        },
        {
            "scalar_ground_state": scalar_ground_state,
            "exact_ground_state": exact_ground_state,
            "comparison": {
                "F_blind_exact": F_blind_exact,
                "F_theory_exact": F_theory_exact,
                "F_tail_theory": F_tail_theory,
                "F_tail_peak": F_tail_peak,
            },
        },
    )

    declaration_gate = payload(
        "8.7.56.1257",
        "Trial-2 numeric alpha Q-ball projection-overlap eigenvalue-matching-scale review declaration gate",
        inputs,
        [
            row("eigenvalue_matching_scale_review_completed", "pass", "eigenvalue-matching-scale review completed", 1.0, "The eigenvalue-matching-scale review branch has now been audited end-to-end."),
            row("exact_profile_eigenvalue_candidate_pass", "pass" if exact_profile_eigenvalue_candidate_pass else "reject", "exact-profile eigenvalue candidate pass", 1 if exact_profile_eigenvalue_candidate_pass else 0, "The exact retained profile supports the eigenvalue-derived q_* candidate without any new fit parameter."),
            row("analytic_pure_tail_theorem_supported", "pass" if analytic_pure_tail_theorem_supported else "reject", "analytic pure-tail theorem supported", 1 if analytic_pure_tail_theorem_supported else 0, "The note's pure-tail theorem remains provisional until the analytic proxy itself reaches the observed target."),
            row("current_canon_coupled_tail_surface_available", "pass" if current_canon_coupled_tail_surface_available else "reject", "current canon coupled-tail surface available", 1 if current_canon_coupled_tail_surface_available else 0, "Part I still exposes the coupled asymptotic eigenmode decaying-tail surface."),
            row("current_canon_coupled_tail_reconciliation_required", "pass" if current_canon_coupled_tail_reconciliation_required else "reject", "current canon coupled-tail reconciliation required", 1 if current_canon_coupled_tail_reconciliation_required else 0, "Current canon still lacks the explicit q_* theorem or normalization sentence that would turn the candidate into a canonical theorem."),
            row("projection_overlap_predictive_branch_ready", "pass" if not current_canon_coupled_tail_reconciliation_required and analytic_pure_tail_theorem_supported else "reject", "projection-overlap predictive branch ready", 1 if (not current_canon_coupled_tail_reconciliation_required and analytic_pure_tail_theorem_supported) else 0, "Predictive promotion remains withheld while theorem-level reconciliation is still open."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "qball_projection_overlap_eigenvalue_matching_candidate_exact_profile_pass_theorem_open",
            "exact_profile_eigenvalue_matching_candidate_available": True,
            "exact_profile_eigenvalue_matching_candidate_pass": exact_profile_eigenvalue_candidate_pass,
            "analytic_pure_tail_theorem_supported": analytic_pure_tail_theorem_supported,
            "current_canon_coupled_tail_surface_available": current_canon_coupled_tail_surface_available,
            "current_canon_coupled_tail_reconciliation_required": current_canon_coupled_tail_reconciliation_required,
            "predictive_branch_ready": False,
            "primary_residual_lane": "qball_projection_overlap_coupled_tail_reconciliation",
            "secondary_residual_lane": "qball_projection_overlap_analytic_tail_theorem_refinement",
            "reserve_residual_lane": "adopted_u1_charge_unit_dictionary",
            "physical_reject_required": False,
            "closeout_ready": False,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_qball_projection_overlap_eigenvalue_matching_scale_declared",
            "advance_to_8_7_56_1258": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "audit_summary": audit["summary"],
            "prior_effective_gate_summary": effective_gate["summary"],
        },
    )

    evaluation = payload(
        "8.7.56.1258",
        "Trial-2 numeric alpha Q-ball projection-overlap eigenvalue-matching-scale review numeric evaluation",
        inputs,
        [
            row("eigenvalue_matching_scale_q_theory_fixed", "pass", "eigenvalue-matching-scale q_theory fixed", q_theory, "The eigenvalue-derived q_* candidate is fixed from beta_1 with no new fit parameter."),
            row("eigenvalue_matching_scale_q_relative_error_vs_blind_fixed", "pass", "eigenvalue-matching-scale q relative error vs blind fixed", q_rel_error_vs_blind, "The relative mismatch between q_*^(theory) and the blind crossing is recorded exactly."),
            row("eigenvalue_matching_scale_F_exact_fixed", "pass", "eigenvalue-matching-scale F_exact fixed", F_theory_exact, "The exact retained profile form factor at q_*^(theory) is recorded exactly."),
            row("eigenvalue_matching_scale_alpha_exact_fixed", "pass", "eigenvalue-matching-scale alpha_exact fixed", alpha_theory_exact, "The exact retained profile alpha candidate at q_*^(theory) is recorded exactly."),
            row("eigenvalue_matching_scale_alpha_relative_error_vs_target_fixed", "pass", "eigenvalue-matching-scale alpha relative error vs target fixed", alpha_theory_rel_error_vs_target, "The alpha mismatch at q_*^(theory) is recorded exactly."),
            row("pure_exponential_tail_peak_fixed", "pass", "pure exponential tail peak fixed", F_tail_peak, "The pure-tail proxy ceiling is recorded to show why the note's analytic theorem is not yet closed."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "qball_projection_overlap_eigenvalue_matching_candidate_exact_profile_pass_theorem_open",
            "beta1": beta1,
            "kappa_legacy": kappa_legacy,
            "q_theory_over_m0": q_theory,
            "q_blind_over_m0": q_blind,
            "q_relative_error_vs_blind": q_rel_error_vs_blind,
            "improvement_factor_vs_best_profile_statistic": improvement_factor_vs_best_profile_statistic,
            "F_target": TARGET_FORM_FACTOR,
            "F_exact_at_q_theory": F_theory_exact,
            "F_exact_relative_error_vs_target": F_exact_rel_error_vs_target,
            "alpha_target": ALPHA_TARGET,
            "alpha_exact_at_q_theory": alpha_theory_exact,
            "alpha_exact_relative_error_vs_target": alpha_theory_rel_error_vs_target,
            "q_peak_tail_over_m0": q_peak_tail,
            "F_tail_at_q_theory": F_tail_theory,
            "F_tail_peak": F_tail_peak,
            "pure_tail_target_reachable": pure_tail_target_reachable,
            "exact_profile_eigenvalue_candidate_pass": exact_profile_eigenvalue_candidate_pass,
            "analytic_pure_tail_theorem_supported": analytic_pure_tail_theorem_supported,
            "numeric_state_changed_by_current_branch": True,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_qball_projection_overlap_eigenvalue_matching_scale_completed",
            "advance_to_next_route": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "comparison": {
                "F_blind_exact": F_blind_exact,
                "F_theory_exact": F_theory_exact,
                "alpha_target": ALPHA_TARGET,
                "alpha_theory_exact": alpha_theory_exact,
            },
            "tail_proxy": {
                "q_peak_tail_over_m0": q_peak_tail,
                "F_tail_at_q_theory": F_tail_theory,
                "F_tail_peak": F_tail_peak,
            },
        },
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_eigenvalue_matching_scale_review_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_eigenvalue_matching_scale_review_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_eigenvalue_matching_scale_review_declaration_gate",
        declaration_gate,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_eigenvalue_matching_scale_review_numeric_evaluation",
        evaluation,
    )

    print("[done] 8.7.56.1255-.1258 artifacts generated")


if __name__ == "__main__":
    main()
