#!/usr/bin/env python3
"""Generate exact-action-level ell=0 operator-derivation artifacts for 8.7.56.1471-.1474.

This branch does not assume that the Phase 1 pilot ODE is already the exact
action-level ell=0 operator. Instead, it audits what is explicitly frozen by:

- Part I post-photon nontransverse wording
- the public 2x2 nontransverse quadratic-form freeze
- the public diagonalization / basis-statement freeze
- the current Phase 1 exact-pilot script
- the retained Trial-3 two-component family solver

The goal is to decide whether current canon plus frozen public artifacts already
support a closed exact ell=0 operator, or only a partial free quadratic
backbone that still requires a corrected operator/bootstrap phase.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


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
CASE_GAMMA_ADVICE = ROOT / "doc" / "quantum" / "42_trial2_numeric_alpha_vector_qball_case_gamma_advice_request.md"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

UNIFIED_PLAN = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_unified_closure_plan_20260327.md")
NEXT_STEPS = Path(r"C:\Users\ogawa\Downloads\trial2_vector_qball_next_steps_20260327.md")
SOLVER_FIX = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_solver_fix_final.md")
PERTURBATIVE_NOTE = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_perturbative_fL_correction.md")

POST_PHOTON_QFORM = PUBLIC_OUT / "mass_origin_v2_post_photon_nontransverse_two_by_two_quadratic_form_metrics.json"
POST_PHOTON_DIAG = PUBLIC_OUT / "mass_origin_v2_post_photon_nontransverse_diagonalization_basis_statement_metrics.json"
PHASE1_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_phase1_exact_coupled_l0_solver_"
    "numeric_evaluation_metrics.json"
)
CASE_GAMMA_EVAL = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_vector_qball_form_factor_unified_closure_perturbative_fl_driven_ode_"
    "diagnostic_reopen_review_numeric_evaluation_metrics.json"
)

PIVOT_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_v2_trial3_two_component_pivot_branch.py"
TRIAL3_FAMILY_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_v2_trial3_two_component_spectrum_branch.py"
PHASE1_SCRIPT = ROOT / "scripts" / "quantum" / "t2a_1419.py"

STEP_TAG = "8.7.56.1471-1474"
STEM = build_compact_artifact_stem(STEP_TAG, "ell0_exact_operator_derivation", prefix="q")
STEP_NAME = "Trial-2 numeric alpha vector Q-ball form-factor exact-action-level ell=0 operator derivation"

PRIOR_CLASS = "vector_qball_form_factor_unified_closure_future_reopen_trigger_handoff_archive_registry_restore_completed"
BRANCH_CLASS = (
    "vector_qball_form_factor_exact_action_level_ell0_operator_derivation_partial_free_backbone_bootstrap_required"
)
NEXT_ROUTE_NAME = "trial2_numeric_alpha_vector_qball_form_factor_corrected_exact_action_level_ell0_family_map_bootstrap"
NEXT_ROUTE = "8.7.56.1475"


# Function: return the current UTC timestamp.
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# Function: abort immediately when one required path is missing.

def require(path: Path) -> None:
    """Abort immediately when one required path is missing."""
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


# Function: convert one absolute path into repo-relative display text when possible.

def display_path(path: Path) -> str:
    """Convert one absolute path into repo-relative display text when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# Function: return the first matching line for one substring pattern.

def hit(text: str, pattern: str) -> dict | None:
    """Return the first matching line for one substring pattern."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# Function: extract the text between two markers when both exist.

def slice_between(text: str, start: str, end: str) -> str:
    """Extract the text between two markers when both exist."""
    start_index = text.find(start)
    if start_index < 0:
        return ""

    end_index = text.find(end, start_index)
    if end_index < 0:
        return text[start_index:]

    return text[start_index:end_index]


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


# Function: build one standard payload object.

def payload(
    step: str,
    name: str,
    inputs: dict,
    rows: list[dict],
    summary: dict,
    decision: dict,
    evidence: dict,
) -> dict:
    """Build one standard payload object."""
    return {
        "generated_utc": now_iso(),
        "phase": {"phase": 8, "step": step, "name": name},
        "inputs": inputs,
        "rows": rows,
        "summary": summary,
        "decision": decision,
        "evidence": evidence,
    }


# Function: write one JSON payload and its rows CSV with Windows-safe paths.

def write_artifact(kind: str, data: dict) -> dict[str, str]:
    """Write one JSON payload and its rows CSV with Windows-safe paths."""
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    paths = build_metrics_paths(PUBLIC_OUT, STEM, kind)
    paths["json"].write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with paths["csv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])

    return {"json": display_path(paths["json"]), "csv": display_path(paths["csv"])}


# Function: dynamically load one local Python module.

def load_module(path: Path, module_name: str):
    """Dynamically load one local Python module."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[fail] unable to import module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Function: execute the exact-action-level ell=0 operator derivation branch.

def main() -> None:
    """Execute the exact-action-level ell=0 operator derivation branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        CURRENT_STATUS,
        UNIFIED_ROADMAP,
        CASE_GAMMA_ADVICE,
        PART1,
        PART5,
        UNIFIED_PLAN,
        NEXT_STEPS,
        SOLVER_FIX,
        PERTURBATIVE_NOTE,
        POST_PHOTON_QFORM,
        POST_PHOTON_DIAG,
        PHASE1_EVAL,
        CASE_GAMMA_EVAL,
        PIVOT_BRANCH,
        TRIAL3_FAMILY_BRANCH,
        PHASE1_SCRIPT,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    part1_text = read_text(PART1)
    part5_text = read_text(PART5)
    current_problem_text = read_text(CURRENT_PROBLEM)
    current_status_text = read_text(CURRENT_STATUS)
    unified_roadmap_text = read_text(UNIFIED_ROADMAP)
    solver_fix_text = read_text(SOLVER_FIX)
    perturbative_note_text = read_text(PERTURBATIVE_NOTE)
    next_steps_text = read_text(NEXT_STEPS)
    phase1_script_text = read_text(PHASE1_SCRIPT)
    family_script_text = read_text(TRIAL3_FAMILY_BRANCH)

    qform = read_json(POST_PHOTON_QFORM)
    diag = read_json(POST_PHOTON_DIAG)
    phase1_eval = read_json(PHASE1_EVAL)
    case_gamma_eval = read_json(CASE_GAMMA_EVAL)
    pivot = load_module(PIVOT_BRANCH, "trial3_pivot_branch")

    qform_summary = qform["summary"]
    diag_summary = diag["summary"]
    phase1_summary = phase1_eval["summary"]
    case_gamma_summary = case_gamma_eval["summary"]

    phase1_exact_slice = slice_between(
        phase1_script_text,
        "def solve_exact_profile(",
        "def run_exact_scan(",
    )
    family_solver_slice = slice_between(
        family_script_text,
        "def solve_two_component_profile(",
        "def main(",
    )

    part1_post_photon_available = hit(part1_text, "post-photon nontransverse sector") is not None
    part1_constraint_branch_available = hit(part1_text, "one constraint branch") is not None
    part1_free_action_available = hit(part1_text, "\\mathcal{L}_{P_\\mu}^{\\mathrm{free}}") is not None
    part1_field_strength_available = hit(part1_text, "F^{(P)}_{\\mu\\nu}") is not None

    qform_available = bool(qform_summary["working_action_nontransverse_two_by_two_quadratic_form_available"])
    diagonalization_available = bool(diag_summary["working_action_nontransverse_quadratic_diagonalization_available"])
    radial_mass_formula_available = qform_summary["radial_mass_squared_formula"] == "m_0^2 = 4 lambda v^2 / Z_P"
    longitudinal_direct_mass_zero_available = (
        qform_summary["longitudinal_direct_mass_squared_formula"] == "m_L,dir^2 = 0"
    )
    offdiag_omega_k_available = "-omega k" in str(qform["formulas"]["quadratic_form_matrix"])
    one_propagating_mode_available = int(diag_summary["post_photon_nontransverse_propagating_dof_count"]) == 1
    one_constraint_mode_available = int(diag_summary["post_photon_nontransverse_constraint_mode_count"]) == 1

    phase1_mass_terms_reused = (
        float(pivot.RADIAL_MASS_SQUARED) == 4.0 and float(pivot.LONGITUDINAL_DIRECT_MASS_SQUARED) == 0.0
    )
    phase1_exact_solver_cross_term_present = (
        "- coupling * f_l" in phase1_exact_slice and "- coupling * f0" in phase1_exact_slice
    )
    phase1_exact_solver_constraint_elimination_present = (
        "constraint" in phase1_exact_slice.lower() or "stueckelberg" in phase1_exact_slice.lower()
    )
    phase1_exact_solver_scalar_nonlinear_ansatz_only = "3.0 * rho + rho * rho" in phase1_exact_slice

    family_solver_ell_dependent_coupling_only = "math.sqrt(max(float(ell * (ell + 1)), 0.0)) / rr" in family_solver_slice
    family_solver_ell0_coupling_collapses = family_solver_ell_dependent_coupling_only

    exact_action_level_linear_backbone_available = all(
        (
            part1_post_photon_available,
            part1_constraint_branch_available,
            part1_free_action_available,
            part1_field_strength_available,
            qform_available,
            diagonalization_available,
            radial_mass_formula_available,
            longitudinal_direct_mass_zero_available,
            offdiag_omega_k_available,
            one_propagating_mode_available,
            one_constraint_mode_available,
        )
    )
    exact_action_level_closed_ell0_operator_available = bool(
        exact_action_level_linear_backbone_available
        and phase1_exact_solver_cross_term_present
        and phase1_exact_solver_constraint_elimination_present
        and not phase1_exact_solver_scalar_nonlinear_ansatz_only
    )
    exact_action_level_operator_derivation_partial = bool(
        exact_action_level_linear_backbone_available and not exact_action_level_closed_ell0_operator_available
    )
    corrected_operator_bootstrap_required = bool(exact_action_level_operator_derivation_partial)
    old_family_map_on_current_pilot_admissible = bool(
        exact_action_level_closed_ell0_operator_available and not family_solver_ell0_coupling_collapses
    )

    inventory_rows = [
        row(
            "part1_post_photon_nontransverse_sector_available",
            "pass" if part1_post_photon_available else "reject",
            "Part I post-photon nontransverse sector available",
            1 if part1_post_photon_available else 0,
            "The operator derivation inventory starts from the Part I post-photon nontransverse sector wording.",
        ),
        row(
            "post_photon_two_by_two_quadratic_form_available",
            "pass" if qform_available else "reject",
            "post-photon 2x2 quadratic form available",
            1 if qform_available else 0,
            "The public quadratic-form freeze must already expose the nontransverse 2x2 backbone.",
        ),
        row(
            "post_photon_diagonalization_available",
            "pass" if diagonalization_available else "reject",
            "post-photon diagonalization available",
            1 if diagonalization_available else 0,
            "The public diagonalization freeze must already expose one propagating mode plus one constraint branch.",
        ),
        row(
            "phase1_exact_pilot_available",
            "pass",
            "Phase 1 exact pilot available",
            1,
            "The current pilot script is inventoried as the implementation to compare against the public operator backbone.",
        ),
        row(
            "case_gamma_diagnostic_available",
            "pass",
            "Case gamma diagnostic available",
            1,
            "The Case gamma diagnostic remains part of the operator-derivation pack because it constrains what not to retry.",
        ),
    ]

    inventory_payload = payload(
        "8.7.56.1471",
        f"{STEP_NAME} inventory",
        {
            "source_files": {
                "status": display_path(STATUS),
                "roadmap": display_path(ROADMAP),
                "ai_context": display_path(AI_CONTEXT),
                "work_history_recent": display_path(WORK_HISTORY_RECENT),
                "current_problem_note": display_path(CURRENT_PROBLEM),
                "current_status_note": display_path(CURRENT_STATUS),
                "unified_closure_roadmap_note": display_path(UNIFIED_ROADMAP),
                "case_gamma_advice_note": display_path(CASE_GAMMA_ADVICE),
                "part1": display_path(PART1),
                "part5": display_path(PART5),
                "unified_plan_note": display_path(UNIFIED_PLAN),
                "next_steps_note": display_path(NEXT_STEPS),
                "solver_fix_note": display_path(SOLVER_FIX),
                "perturbative_note": display_path(PERTURBATIVE_NOTE),
            },
            "source_metrics": {
                "post_photon_qform": display_path(POST_PHOTON_QFORM),
                "post_photon_diag": display_path(POST_PHOTON_DIAG),
                "phase1_eval": display_path(PHASE1_EVAL),
                "case_gamma_eval": display_path(CASE_GAMMA_EVAL),
            },
            "solver_modules": {
                "pivot_branch": display_path(PIVOT_BRANCH),
                "trial3_family_branch": display_path(TRIAL3_FAMILY_BRANCH),
                "phase1_script": display_path(PHASE1_SCRIPT),
            },
        },
        inventory_rows,
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "current_pack_limit_state": PRIOR_CLASS,
            "inventory_ready": True,
            "exact_action_level_linear_backbone_available": exact_action_level_linear_backbone_available,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_exact_action_level_ell0_operator_derivation_inventory_completed",
            "branch_completed": False,
            "next_required_artifacts": [f"{STEM}_audit_metrics.json"],
        },
        {
            "part1_post_photon_hit": hit(part1_text, "post-photon nontransverse sector"),
            "part1_constraint_hit": hit(part1_text, "one constraint branch"),
            "qform_matrix": qform["formulas"]["quadratic_form_matrix"],
            "diag_basis_statement": diag["formulas"]["basis_statement"],
            "solver_fix_hit": hit(solver_fix_text, "F_{0r}^{(P)} = i\\omega f_L - f_0'"),
            "perturbative_note_hit": hit(perturbative_note_text, "F_{0r}^{(P)} = i\\omega f_L - f_0'"),
            "phase1_script_hit": hit(phase1_script_text, "def solve_exact_profile("),
            "part5_hit": hit(part5_text, "exact-action-level `ell=0` operator derivation"),
            "problem_note_hit": hit(current_problem_text, "exact-action-level `ell=0` operator derivation"),
            "status_note_hit": hit(current_status_text, "exact-action-level `ell=0` operator derivation"),
        },
    )
    inventory_paths = write_artifact("inventory", inventory_payload)

    audit_rows = [
        row(
            "exact_action_level_linear_backbone_available",
            "pass" if exact_action_level_linear_backbone_available else "reject",
            "exact action-level linear backbone available",
            1 if exact_action_level_linear_backbone_available else 0,
            "Current canon plus frozen public artifacts do expose the post-photon free 2x2 backbone, its diagonalization, and the propagating/constraint split.",
        ),
        row(
            "offdiag_omega_k_available",
            "pass" if offdiag_omega_k_available else "reject",
            "free quadratic backbone off-diagonal omega-k mixing available",
            1 if offdiag_omega_k_available else 0,
            "The public quadratic-form freeze explicitly retains the off-diagonal -omega k term.",
        ),
        row(
            "phase1_exact_solver_cross_term_present",
            "pass" if phase1_exact_solver_cross_term_present else "reject",
            "Phase 1 exact pilot cross term present",
            1 if phase1_exact_solver_cross_term_present else 0,
            "The current Phase 1 pilot would count as exact only if it explicitly carried the off-diagonal ell=0 mixing term.",
        ),
        row(
            "phase1_exact_solver_constraint_elimination_present",
            "pass" if phase1_exact_solver_constraint_elimination_present else "reject",
            "Phase 1 exact pilot constraint elimination present",
            1 if phase1_exact_solver_constraint_elimination_present else 0,
            "A closed exact ell=0 operator needs an explicit constraint/Stueckelberg elimination step or equivalent statement.",
        ),
        row(
            "phase1_exact_solver_scalar_nonlinear_ansatz_only",
            "watch" if phase1_exact_solver_scalar_nonlinear_ansatz_only else "pass",
            "Phase 1 exact pilot uses scalar-style nonlinear ansatz only",
            1 if phase1_exact_solver_scalar_nonlinear_ansatz_only else 0,
            "The current pilot still inserts the shared 3 rho + rho^2 ansatz rather than a fully action-derived two-component nonlinear closure.",
        ),
        row(
            "family_solver_ell0_coupling_collapses",
            "watch" if family_solver_ell0_coupling_collapses else "pass",
            "Trial-3 family solver ell-dependent coupling collapses at ell=0",
            1 if family_solver_ell0_coupling_collapses else 0,
            "The retained family solver uses sqrt(ell(ell+1))/r coupling, so its coupling vanishes at ell=0 and cannot stand in for the action-level ell=0 backbone.",
        ),
        row(
            "exact_action_level_closed_ell0_operator_available",
            "pass" if exact_action_level_closed_ell0_operator_available else "reject",
            "exact action-level closed ell=0 operator available",
            1 if exact_action_level_closed_ell0_operator_available else 0,
            "Full success requires the public backbone plus implemented mixing, constraint elimination, and non-heuristic nonlinear closure.",
        ),
        row(
            "exact_action_level_operator_derivation_partial",
            "pass" if exact_action_level_operator_derivation_partial else "reject",
            "exact operator derivation is partial free-backbone only",
            1 if exact_action_level_operator_derivation_partial else 0,
            "The honest current result is a partial derivation when the linear backbone is fixed but the closed nonlinear ell=0 operator still remains open.",
        ),
        row(
            "corrected_operator_bootstrap_required",
            "pass" if corrected_operator_bootstrap_required else "reject",
            "corrected exact-operator bootstrap required",
            1 if corrected_operator_bootstrap_required else 0,
            "The next computation step must bootstrap a corrected ell=0 family from the fixed backbone instead of reusing the old pilot unchanged.",
        ),
        row(
            "old_family_map_on_current_pilot_admissible",
            "pass" if old_family_map_on_current_pilot_admissible else "reject",
            "old family map on the current pilot remains admissible",
            1 if old_family_map_on_current_pilot_admissible else 0,
            "If this fails, the roadmap must change before the family-map phase starts.",
        ),
    ]

    audit_payload = payload(
        "8.7.56.1472",
        f"{STEP_NAME} audit",
        {
            "inventory_json": inventory_paths["json"],
            "part1": display_path(PART1),
            "post_photon_qform_json": display_path(POST_PHOTON_QFORM),
            "post_photon_diag_json": display_path(POST_PHOTON_DIAG),
            "phase1_script": display_path(PHASE1_SCRIPT),
            "trial3_family_branch": display_path(TRIAL3_FAMILY_BRANCH),
            "phase1_eval_json": display_path(PHASE1_EVAL),
            "case_gamma_eval_json": display_path(CASE_GAMMA_EVAL),
        },
        audit_rows,
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "current_pack_limit_state": PRIOR_CLASS,
            "part1_post_photon_nontransverse_sector_available": part1_post_photon_available,
            "part1_constraint_branch_available": part1_constraint_branch_available,
            "exact_action_level_linear_backbone_available": exact_action_level_linear_backbone_available,
            "exact_action_level_closed_ell0_operator_available": exact_action_level_closed_ell0_operator_available,
            "exact_action_level_operator_derivation_partial": exact_action_level_operator_derivation_partial,
            "phase1_exact_solver_cross_term_present": phase1_exact_solver_cross_term_present,
            "phase1_exact_solver_constraint_elimination_present": phase1_exact_solver_constraint_elimination_present,
            "phase1_exact_solver_scalar_nonlinear_ansatz_only": phase1_exact_solver_scalar_nonlinear_ansatz_only,
            "trial3_family_solver_ell0_coupling_collapses": family_solver_ell0_coupling_collapses,
            "corrected_operator_bootstrap_required": corrected_operator_bootstrap_required,
            "old_family_map_on_current_pilot_admissible": old_family_map_on_current_pilot_admissible,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_exact_action_level_ell0_operator_derivation_partial_backbone_completed",
            "branch_completed": False,
            "next_required_artifacts": [f"{STEM}_declaration_gate_metrics.json"],
        },
        {
            "part1_post_photon_hit": hit(part1_text, "post-photon nontransverse sector"),
            "part1_constraint_hit": hit(part1_text, "one constraint branch"),
            "part1_free_action_hit": hit(part1_text, "\\mathcal{L}_{P_\\mu}^{\\mathrm{free}}"),
            "qform_matrix": qform["formulas"]["quadratic_form_matrix"],
            "diag_basis_statement": diag["formulas"]["basis_statement"],
            "phase1_exact_solver_f0_line": hit(phase1_exact_slice, "f0_double_prime = -(2.0 / safe_r) * f0_prime"),
            "phase1_exact_solver_fL_line": hit(phase1_exact_slice, "f_l_double_prime = -(2.0 / safe_r) * f_l_prime"),
            "trial3_family_solver_coupling_line": hit(
                family_solver_slice,
                "k_proxy = math.sqrt(max(float(ell * (ell + 1)), 0.0)) / rr",
            ),
            "solver_fix_offdiag_hit": hit(solver_fix_text, "F_{0r}^{(P)} = i\\omega f_L - f_0'"),
            "next_steps_hit": hit(next_steps_text, "longitudinal operator"),
        },
    )
    audit_paths = write_artifact("audit", audit_payload)

    gate_rows = [
        row(
            "partial_free_backbone_only_selected",
            "pass" if exact_action_level_operator_derivation_partial else "reject",
            "partial free-backbone-only disposition selected",
            1 if exact_action_level_operator_derivation_partial else 0,
            "The declaration gate should only select the partial-backbone disposition when the public linear operator exists but the closed ell=0 operator still does not.",
        ),
        row(
            "corrected_operator_bootstrap_required",
            "pass" if corrected_operator_bootstrap_required else "reject",
            "corrected exact-operator bootstrap required",
            1 if corrected_operator_bootstrap_required else 0,
            "The next branch changes from generic family continuation to corrected-family bootstrap only if the audit confirmed the operator gap.",
        ),
        row(
            "advance_to_corrected_family_map_bootstrap",
            "pass" if corrected_operator_bootstrap_required else "reject",
            "advance to corrected exact-operator family-map bootstrap",
            1 if corrected_operator_bootstrap_required else 0,
            "Once the exact operator is only partial, the next mainline must rebuild the family map on the corrected backbone.",
        ),
    ]

    gate_payload = payload(
        "8.7.56.1473",
        f"{STEP_NAME} declaration gate",
        {
            "inventory_json": inventory_paths["json"],
            "audit_json": audit_paths["json"],
            "roadmap": display_path(ROADMAP),
            "status": display_path(STATUS),
        },
        gate_rows,
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "prior_problem_classification": PRIOR_CLASS,
            "exact_action_level_linear_backbone_available": exact_action_level_linear_backbone_available,
            "exact_action_level_closed_ell0_operator_available": exact_action_level_closed_ell0_operator_available,
            "exact_action_level_operator_derivation_partial": exact_action_level_operator_derivation_partial,
            "corrected_operator_bootstrap_required": corrected_operator_bootstrap_required,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
            "physical_reject_required": False,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_exact_action_level_ell0_operator_derivation_completed",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "roadmap_next_step_hit": hit(roadmap_text, "8.7.56.1475-.1478"),
            "status_next_step_hit": hit(status_text, "8.7.56.1471"),
            "unified_roadmap_hit": hit(unified_roadmap_text, "exact-action-level `ell=0` operator derivation"),
        },
    )
    gate_paths = write_artifact("declaration_gate", gate_payload)

    eval_rows = [
        row(
            "radial_mass_squared_over_lambda_v2_over_zp",
            "pass",
            "radial mass squared coefficient in units of lambda v^2 / Z_P",
            float(pivot.RADIAL_MASS_SQUARED),
            "The public free backbone and the pivot constants agree on m_0^2 = 4 lambda v^2 / Z_P.",
        ),
        row(
            "longitudinal_direct_mass_squared",
            "pass",
            "direct longitudinal mass squared coefficient",
            float(pivot.LONGITUDINAL_DIRECT_MASS_SQUARED),
            "The public free backbone and the pivot constants agree that the direct longitudinal mexican-hat mass is zero.",
        ),
        row(
            "phase1_best_exact_f_at_q_theory",
            "watch",
            "Phase 1 best exact pilot F(q_theory)",
            float(phase1_summary["phase1_best_alpha_candidate"]["F_at_q_theory"]),
            "The old exact pilot result is retained only as a numeric baseline, not as proof that the exact action-level operator has already been implemented.",
        ),
        row(
            "phase1_best_exact_alpha_at_q_theory",
            "watch",
            "Phase 1 best exact pilot alpha(q_theory)",
            float(phase1_summary["phase1_best_alpha_candidate"]["alpha_at_q_theory"]),
            "The old exact pilot remains far from the scalar strong candidate and cannot settle the operator question alone.",
        ),
        row(
            "diagnostic_case_gamma_ratio",
            "watch",
            "Case gamma diagnostic max|f_L|/max|f_0|",
            float(case_gamma_summary["diagnostic_max_abs_ratio"]),
            "Case gamma remains frozen as a diagnostic result and constrains the next action not to retry perturbative rescue.",
        ),
        row(
            "phase1_best_seed_amp_ratio",
            "watch",
            "Phase 1 best seed amp_l/amp0 ratio",
            float(phase1_summary["phase1_best_alpha_candidate"]["amp_l"])
            / float(phase1_summary["phase1_best_alpha_candidate"]["amp0"]),
            "The old best pilot seed ratio is recorded because the corrected bootstrap will need a new family map rather than a reuse of this isolated seed.",
        ),
        row(
            "exact_action_level_linear_backbone_available",
            "pass" if exact_action_level_linear_backbone_available else "reject",
            "exact action-level linear backbone available",
            1 if exact_action_level_linear_backbone_available else 0,
            "This row keeps the key positive result machine-readable for the downstream corrected-family bootstrap.",
        ),
        row(
            "exact_action_level_closed_ell0_operator_available",
            "pass" if exact_action_level_closed_ell0_operator_available else "reject",
            "exact action-level closed ell=0 operator available",
            1 if exact_action_level_closed_ell0_operator_available else 0,
            "This row keeps the key negative result machine-readable for the downstream corrected-family bootstrap.",
        ),
    ]

    eval_payload = payload(
        "8.7.56.1474",
        f"{STEP_NAME} numeric evaluation",
        {
            "inventory_json": inventory_paths["json"],
            "audit_json": audit_paths["json"],
            "declaration_gate_json": gate_paths["json"],
            "post_photon_qform_json": display_path(POST_PHOTON_QFORM),
            "phase1_eval_json": display_path(PHASE1_EVAL),
            "case_gamma_eval_json": display_path(CASE_GAMMA_EVAL),
        },
        eval_rows,
        {
            "trial2_numeric_alpha_problem_classification": BRANCH_CLASS,
            "prior_problem_classification": PRIOR_CLASS,
            "beta_1_scalar": float(phase1_summary["beta_1_scalar"]),
            "q_theory_over_m0_scalar": float(phase1_summary["q_theory_over_m0_scalar"]),
            "F_exact_at_q_theory_scalar": float(phase1_summary["F_exact_at_q_theory_scalar"]),
            "alpha_exact_at_q_theory_scalar": float(phase1_summary["alpha_exact_at_q_theory_scalar"]),
            "phase1_best_exact_F_at_q_theory": float(phase1_summary["phase1_best_alpha_candidate"]["F_at_q_theory"]),
            "phase1_best_exact_alpha_at_q_theory": float(
                phase1_summary["phase1_best_alpha_candidate"]["alpha_at_q_theory"]
            ),
            "case_gamma_diagnostic_ratio": float(case_gamma_summary["diagnostic_max_abs_ratio"]),
            "exact_action_level_linear_backbone_available": exact_action_level_linear_backbone_available,
            "exact_action_level_closed_ell0_operator_available": exact_action_level_closed_ell0_operator_available,
            "exact_action_level_operator_derivation_partial": exact_action_level_operator_derivation_partial,
            "corrected_operator_bootstrap_required": corrected_operator_bootstrap_required,
            "old_family_map_on_current_pilot_admissible": old_family_map_on_current_pilot_admissible,
            "physical_reject_required": False,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_vector_qball_form_factor_exact_action_level_ell0_operator_derivation_route_synced",
            "branch_completed": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "phase1_best_alpha_candidate": phase1_summary["phase1_best_alpha_candidate"],
            "case_gamma_summary": case_gamma_summary,
            "qform_summary": qform_summary,
            "diag_summary": diag_summary,
        },
    )
    eval_paths = write_artifact("numeric_evaluation", eval_payload)

    print("[ok] exact-action-level ell=0 operator derivation artifacts written:")
    print(f" - {inventory_paths['json']}")
    print(f" - {audit_paths['json']}")
    print(f" - {gate_paths['json']}")
    print(f" - {eval_paths['json']}")


if __name__ == "__main__":
    main()
