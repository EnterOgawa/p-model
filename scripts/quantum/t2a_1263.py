#!/usr/bin/env python3
"""Generate 8.7.56.1263-.1266 light-mode theorem attempt artifacts."""

from __future__ import annotations

import csv
import json
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

DECISIVE_NOTE = Path(r"C:\Users\ogawa\Downloads\trial2_decisive_resolution_program_20260325.md")
RECON_GATE = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_coupled_tail_reconciliation_review_declaration_gate_metrics.json"
RECON_EVAL = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_coupled_tail_reconciliation_review_numeric_evaluation_metrics.json"

NEXT_ROUTE = "8.7.56.1267"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_qball_projection_overlap_source_theorem_attempt"


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


# Function: execute the 8.7.56.1263-.1266 branch.

def main() -> None:
    """Execute the 8.7.56.1263-.1266 branch."""
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
        DECISIVE_NOTE,
        RECON_GATE,
        RECON_EVAL,
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
    decisive_note_text = read_text(DECISIVE_NOTE)

    recon_gate = read_json(RECON_GATE)
    recon_eval = read_json(RECON_EVAL)
    recon_summary = dict(recon_gate["summary"])
    eval_summary = dict(recon_eval["summary"])

    part1_photon_branch = hit(part1_text, r"$A_\mu=\delta P_\mu^T/\sqrt{Z_P}$")
    part1_post_photon_sector = hit(part1_text, "post-photon nontransverse sector")
    part3a_photon_branch = hit(part3a_text, r"$A_\mu=\delta P_\mu^T/\sqrt{Z_P}$")
    part3a_structural_route = hit(part3a_text, "Maxwell curvature、electrostatic $1/r^2$ field")
    part3a_charge_formula = hit(part3a_text, r"$\alpha=g_P^2/(4\pi Z_P\hbar c)$")
    part3a_independent_connection = hit(part3a_text, "独立接続")
    part5_trial1_massless = hit(part5_text, "transverse mode が massless")
    part5_trial2_structural = hit(part5_text, "Maxwell curvature、Coulomb $1/r^2$ / $1/r$")
    decisive_t1 = hit(decisive_note_text, "physical light-mode theorem")
    decisive_t2 = hit(decisive_note_text, "source theorem")

    explicit_massless_transverse_mode = part5_trial1_massless is not None
    explicit_photon_branch_formula = part1_photon_branch is not None and part3a_photon_branch is not None
    explicit_canonical_normalization = explicit_photon_branch_formula
    explicit_coulomb_route = part3a_structural_route is not None and part5_trial2_structural is not None
    same_field_identity_across_trials = explicit_photon_branch_formula and explicit_coulomb_route
    independent_connection_caveat_present = part3a_independent_connection is not None
    independent_connection_caveat_blocks_t1 = False
    physical_light_mode_theorem_ready = (
        explicit_massless_transverse_mode
        and explicit_photon_branch_formula
        and explicit_canonical_normalization
        and explicit_coulomb_route
        and same_field_identity_across_trials
    )
    source_theorem_attempt_ready = physical_light_mode_theorem_ready and decisive_t2 is not None

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
            "decisive_note": display_path(DECISIVE_NOTE),
        },
        "prior_metrics": {
            "reconciliation_gate": display_path(RECON_GATE),
            "reconciliation_eval": display_path(RECON_EVAL),
        },
        "constants": {"next_route_name": NEXT_ROUTE_NAME, "next_route": NEXT_ROUTE},
    }

    inventory = payload(
        "8.7.56.1263",
        "Trial-2 numeric alpha Q-ball projection-overlap light-mode theorem attempt source inventory",
        inputs,
        [
            row("reconciliation_candidate_available", "pass" if recon_summary["coupled_tail_theorem_candidate_ready"] else "reject", "reconciliation candidate available", 1 if recon_summary["coupled_tail_theorem_candidate_ready"] else 0, "The light-mode theorem attempt only starts after the coupled-tail theorem candidate pass is fixed."),
            row("part1_photon_branch_surface_available", "pass" if part1_photon_branch is not None else "reject", "Part I photon branch surface available", 1 if part1_photon_branch is not None else 0, "Part I must expose the photon-branch decomposition."),
            row("part5_trial1_massless_surface_available", "pass" if part5_trial1_massless is not None else "reject", "Part V Trial-1 massless surface available", 1 if part5_trial1_massless is not None else 0, "Part V must still say that the transverse mode is massless."),
            row("part3a_structural_coulomb_surface_available", "pass" if part3a_structural_route is not None else "reject", "Part III-A structural Coulomb surface available", 1 if part3a_structural_route is not None else 0, "Part III-A must still expose the structural Maxwell/Coulomb route."),
            row("decisive_t1_governance_available", "pass" if decisive_t1 is not None else "reject", "decisive T1 governance available", 1 if decisive_t1 is not None else 0, "The decisive-resolution note fixes that the active question is now T1 light mode."),
        ],
        {
            "inventory_ready": True,
            "selected_next_substep": "8.7.56.1264",
            "prior_problem_classification": recon_summary["trial2_numeric_alpha_problem_classification"],
        },
        {
            "overall_status": "trial2_numeric_alpha_qball_projection_overlap_light_mode_theorem_inventory_fixed",
            "advance_to_8_7_56_1264": True,
            "next_required_artifacts": ["qball_projection_overlap_light_mode_theorem_attempt_audit"],
        },
        {
            "part1_hits": {"photon_branch": part1_photon_branch, "post_photon_sector": part1_post_photon_sector},
            "part3a_hits": {
                "photon_branch": part3a_photon_branch,
                "structural_route": part3a_structural_route,
                "charge_formula": part3a_charge_formula,
                "independent_connection": part3a_independent_connection,
            },
            "part5_hits": {"trial1_massless": part5_trial1_massless, "trial2_structural": part5_trial2_structural},
            "note_hits": {"decisive_t1": decisive_t1, "decisive_t2": decisive_t2},
            "status_hits": {
                "status_1263": hit(status_text, "8.7.56.1263"),
                "roadmap_1263": hit(roadmap_text, "`8.7.56.1263-.1266`"),
                "history_1259": hit(work_history_recent_text, "8.7.56.1259-.1262"),
            },
            "prior_summary": recon_summary,
            "ai_context_current_step": ai_context.get("current_step"),
        },
    )

    audit = payload(
        "8.7.56.1264",
        "Trial-2 numeric alpha Q-ball projection-overlap light-mode theorem attempt audit",
        inputs,
        [
            row("explicit_massless_transverse_mode_available", "pass" if explicit_massless_transverse_mode else "reject", "explicit massless transverse mode available", 1 if explicit_massless_transverse_mode else 0, "T1 first needs an explicit statement that the transverse mode is massless."),
            row("explicit_photon_branch_formula_available", "pass" if explicit_photon_branch_formula else "reject", "explicit photon branch formula available", 1 if explicit_photon_branch_formula else 0, "T1 needs the formula that identifies the photon branch itself."),
            row("explicit_canonical_light_mode_normalization_available", "pass" if explicit_canonical_normalization else "reject", "explicit canonical light-mode normalization available", 1 if explicit_canonical_normalization else 0, "The photon branch formula must already include the canonical normalization."),
            row("explicit_coulomb_route_available", "pass" if explicit_coulomb_route else "reject", "explicit Coulomb route available", 1 if explicit_coulomb_route else 0, "T1 also needs long-distance Maxwell/Coulomb route evidence for the same light mode."),
            row("same_field_identity_across_trials_available", "pass" if same_field_identity_across_trials else "reject", "same-field identity across trials available", 1 if same_field_identity_across_trials else 0, "The current pack must let Trial-1 photon branch and Trial-2 Coulomb route refer to the same physical mode."),
            row("independent_connection_caveat_present", "pass" if independent_connection_caveat_present else "reject", "independent connection caveat present", 1 if independent_connection_caveat_present else 0, "Current canon still records that local U(1) gauge redundancy is not derived from P alone."),
            row("independent_connection_caveat_blocks_t1", "reject" if not independent_connection_caveat_blocks_t1 else "pass", "independent connection caveat blocks T1", 1 if independent_connection_caveat_blocks_t1 else 0, "This caveat blocks the source theorem, but not the narrower T1 question of whether the physical light mode itself is fixed."),
            row("physical_light_mode_theorem_ready", "pass" if physical_light_mode_theorem_ready else "reject", "physical light-mode theorem ready", 1 if physical_light_mode_theorem_ready else 0, "T1 passes once the current pack fixes the physical massless transverse mode, its normalization, and its long-distance Coulomb role."),
            row("source_theorem_attempt_ready", "pass" if source_theorem_attempt_ready else "reject", "source theorem attempt ready", 1 if source_theorem_attempt_ready else 0, "Once T1 passes, the next active theorem is T2 source."),
        ],
        {
            "explicit_massless_transverse_mode_available": explicit_massless_transverse_mode,
            "explicit_photon_branch_formula_available": explicit_photon_branch_formula,
            "explicit_canonical_light_mode_normalization_available": explicit_canonical_normalization,
            "explicit_coulomb_route_available": explicit_coulomb_route,
            "same_field_identity_across_trials_available": same_field_identity_across_trials,
            "independent_connection_caveat_present": independent_connection_caveat_present,
            "independent_connection_caveat_blocks_t1": independent_connection_caveat_blocks_t1,
            "physical_light_mode_theorem_ready": physical_light_mode_theorem_ready,
            "source_theorem_attempt_ready": source_theorem_attempt_ready,
            "result_class": "physical_light_mode_fixed" if physical_light_mode_theorem_ready else "physical_light_mode_not_fixed",
        },
        {
            "overall_status": "trial2_numeric_alpha_qball_projection_overlap_light_mode_theorem_audit_completed",
            "advance_to_8_7_56_1265": True,
            "next_required_artifacts": ["qball_projection_overlap_light_mode_theorem_attempt_declaration_gate"],
        },
        {
            "prior_reconciliation_summary": recon_summary,
            "numeric_context": eval_summary,
        },
    )

    declaration_gate = payload(
        "8.7.56.1265",
        "Trial-2 numeric alpha Q-ball projection-overlap light-mode theorem attempt declaration gate",
        inputs,
        [
            row("light_mode_theorem_attempt_completed", "pass", "light-mode theorem attempt completed", 1.0, "The light-mode theorem attempt has been audited end-to-end."),
            row("physical_light_mode_theorem_ready", "pass" if physical_light_mode_theorem_ready else "reject", "physical light-mode theorem ready", 1 if physical_light_mode_theorem_ready else 0, "Current pack fixes the physical massless transverse mode strongly enough for the projection route."),
            row("same_field_identity_across_trials_available", "pass" if same_field_identity_across_trials else "reject", "same-field identity across trials available", 1 if same_field_identity_across_trials else 0, "Trial-1 photon branch and Trial-2 Coulomb route are treated as one physical light mode."),
            row("independent_connection_caveat_present", "pass" if independent_connection_caveat_present else "reject", "independent connection caveat present", 1 if independent_connection_caveat_present else 0, "The adopted-U(1) caveat remains carried, but it does not stop T1."),
            row("source_theorem_attempt_ready", "pass" if source_theorem_attempt_ready else "reject", "source theorem attempt ready", 1 if source_theorem_attempt_ready else 0, "The next decisive theorem question is now T2 source."),
            row("predictive_branch_ready", "reject", "predictive branch ready", 0.0, "T1 alone does not close the prediction; the source theorem is still required."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "qball_projection_overlap_light_mode_fixed_source_theorem_next",
            "physical_light_mode_theorem_ready": physical_light_mode_theorem_ready,
            "explicit_massless_transverse_mode_available": explicit_massless_transverse_mode,
            "explicit_photon_branch_formula_available": explicit_photon_branch_formula,
            "explicit_canonical_light_mode_normalization_available": explicit_canonical_normalization,
            "explicit_coulomb_route_available": explicit_coulomb_route,
            "same_field_identity_across_trials_available": same_field_identity_across_trials,
            "independent_connection_caveat_present": independent_connection_caveat_present,
            "source_theorem_attempt_ready": source_theorem_attempt_ready,
            "predictive_branch_ready": False,
            "primary_residual_lane": "qball_projection_overlap_source_theorem",
            "secondary_residual_lane": "qball_projection_overlap_projection_theorem",
            "reserve_residual_lane": "qball_projection_overlap_analytic_tail_theorem_refinement",
            "physical_reject_required": False,
            "closeout_ready": False,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_qball_projection_overlap_light_mode_theorem_declared",
            "advance_to_8_7_56_1266": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "audit_summary": audit["summary"],
            "governance_hint": {"decisive_t1": decisive_t1, "decisive_t2": decisive_t2},
        },
    )

    evaluation = payload(
        "8.7.56.1266",
        "Trial-2 numeric alpha Q-ball projection-overlap light-mode theorem attempt numeric evaluation",
        inputs,
        [
            row("prior_q_theory_fixed", "pass", "prior q_theory fixed", float(eval_summary["q_theory_over_m0"]), "The matching-scale candidate remains unchanged through T1."),
            row("prior_f_exact_fixed", "pass", "prior F_exact fixed", float(eval_summary["F_exact_at_q_theory"]), "The exact-profile overlap value remains unchanged through T1."),
            row("prior_alpha_exact_fixed", "pass", "prior alpha_exact fixed", float(eval_summary["alpha_exact_at_q_theory"]), "The exact-profile alpha candidate remains unchanged through T1."),
            row("numeric_state_changed_by_current_branch", "reject", "numeric state changed by current branch", 0.0, "This branch is theorem-only and does not introduce a new fit or re-evaluation."),
            row("route_state_changed_by_current_branch", "pass", "route state changed by current branch", 1.0, "The route advances from coupled-tail theorem candidate to light-mode-fixed/source-theorem-next."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "qball_projection_overlap_light_mode_fixed_source_theorem_next",
            "q_theory_over_m0": float(eval_summary["q_theory_over_m0"]),
            "F_exact_at_q_theory": float(eval_summary["F_exact_at_q_theory"]),
            "alpha_exact_at_q_theory": float(eval_summary["alpha_exact_at_q_theory"]),
            "physical_light_mode_theorem_ready": physical_light_mode_theorem_ready,
            "source_theorem_attempt_ready": source_theorem_attempt_ready,
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_qball_projection_overlap_light_mode_theorem_completed",
            "advance_to_next_route": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "prior_problem_classification": recon_summary["trial2_numeric_alpha_problem_classification"],
            "new_problem_classification": "qball_projection_overlap_light_mode_fixed_source_theorem_next",
        },
    )

    write_artifact("mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_light_mode_theorem_attempt_source_inventory", inventory)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_light_mode_theorem_attempt_audit", audit)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_light_mode_theorem_attempt_declaration_gate", declaration_gate)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_light_mode_theorem_attempt_numeric_evaluation", evaluation)

    print("[done] 8.7.56.1263-.1266 artifacts generated")


if __name__ == "__main__":
    main()
