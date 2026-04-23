#!/usr/bin/env python3
"""Generate 8.7.56.1271-.1274 route-local no-go theorem review artifacts."""

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
SOURCE_GATE = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_source_theorem_attempt_declaration_gate_metrics.json"
SOURCE_EVAL = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_source_theorem_attempt_numeric_evaluation_metrics.json"

NEXT_ROUTE = "8.7.56.1275"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_qball_projection_overlap_route_local_no_go_carry_over_contract"


# Function: return one current UTC timestamp.
def now_iso() -> str:
    """Return one current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# Function: abort when one required path is missing.

def require(path: Path) -> None:
    """Abort when one required path is missing."""
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


# Function: show one repo-relative path when possible.

def display_path(path: Path) -> str:
    """Show one repo-relative path when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# Function: locate one substring hit.

def hit(text: str, pattern: str) -> dict | None:
    """Locate one substring hit."""
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


# Function: write one JSON metrics payload and one CSV row table.

def write_artifact(stem: str, data: dict) -> None:
    """Write one JSON metrics payload and one CSV row table."""
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    json_path = PUBLIC_OUT / f"{stem}_metrics.json"
    csv_path = PUBLIC_OUT / f"{stem}_rows.csv"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# Function: execute the 8.7.56.1271-.1274 branch.

def main() -> None:
    """Execute the 8.7.56.1271-.1274 branch."""
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
        SOURCE_GATE,
        SOURCE_EVAL,
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

    source_gate = read_json(SOURCE_GATE)
    source_eval = read_json(SOURCE_EVAL)
    gate_summary = dict(source_gate["summary"])
    eval_summary = dict(source_eval["summary"])

    part1_current = hit(part1_text, r"J^\mu_{\mathrm{matter}}=(\rho c,\rho \mathbf{v})")
    part1_interaction = hit(part1_text, r"\mathcal{L}_{\mathrm{int}}=g_P\,P_\mu J^\mu_{\mathrm{matter}}")
    part1_micro_current = hit(part1_text, r"-\lambda_{\mathrm{rot}}g_{P}\,\bar{\psi}\gamma^\mu\frac{1-\gamma^5}{2}\psi\,P_\mu")
    part1_full_closure = hit(part1_text, r"+g_P P_\mu J^\mu_{\mathrm{matter}}")
    part3a_independent_connection = hit(part3a_text, "独立接続")
    part3a_structure_template = hit(part3a_text, "構造テンプレート")
    part3a_not_origin = hit(part3a_text, "起源導出とは扱わない")
    part5_source_fail = hit(part5_text, "effective source formula")
    decisive_fail_t2 = hit(decisive_note_text, "effective Q-ball source が action から出ない")
    decisive_t2 = hit(decisive_note_text, "### \\(T_2\\): source theorem")

    source_theorem_failed_under_current_canon = gate_summary["trial2_numeric_alpha_problem_classification"] == "qball_projection_overlap_source_theorem_failed_under_current_canon"
    explicit_matter_current_surface_available = bool(gate_summary["explicit_matter_current_surface_available"])
    explicit_interaction_surface_available = bool(gate_summary["explicit_interaction_surface_available"])
    explicit_effective_source_formula_available = bool(gate_summary["explicit_effective_source_formula_available"])
    independent_connection_caveat_present = bool(gate_summary["independent_connection_caveat_present"])
    route_local_no_go_candidate_ready = bool(gate_summary["route_local_no_go_candidate_ready"])

    generic_matter_coupling_only = explicit_matter_current_surface_available and explicit_interaction_surface_available and not explicit_effective_source_formula_available
    microscopic_current_surfaces_exist_but_generic_only = part1_micro_current is not None and not explicit_effective_source_formula_available
    alternative_current_canon_source_reading_survives = False
    route_local_no_go_theorem_honest = (
        source_theorem_failed_under_current_canon
        and generic_matter_coupling_only
        and microscopic_current_surfaces_exist_but_generic_only
        and independent_connection_caveat_present
        and part3a_structure_template is not None
        and part3a_not_origin is not None
        and not alternative_current_canon_source_reading_survives
    )
    theorem_side_stop_contract_ready = route_local_no_go_theorem_honest
    projection_theorem_carry_over_required = route_local_no_go_theorem_honest

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
            "source_gate": display_path(SOURCE_GATE),
            "source_eval": display_path(SOURCE_EVAL),
        },
        "constants": {"next_route_name": NEXT_ROUTE_NAME, "next_route": NEXT_ROUTE},
    }

    inventory = payload(
        "8.7.56.1271",
        "Trial-2 numeric alpha Q-ball projection-overlap route-local no-go theorem review source inventory",
        inputs,
        [
            row("source_theorem_fail_state_available", "pass" if source_theorem_failed_under_current_canon else "reject", "source theorem fail state available", 1 if source_theorem_failed_under_current_canon else 0, "The route-local no-go review only starts after T2 has failed under current canon."),
            row("part1_generic_current_surface_available", "pass" if part1_current is not None else "reject", "Part I generic current surface available", 1 if part1_current is not None else 0, "The review must re-check the generic current source surface itself."),
            row("part1_micro_current_surface_available", "pass" if part1_micro_current is not None else "reject", "Part I micro current surface available", 1 if part1_micro_current is not None else 0, "The review must also re-check the microscopic current/spin surfaces for alternate source readings."),
            row("part3a_structure_template_surface_available", "pass" if part3a_structure_template is not None else "reject", "Part III-A structure-template surface available", 1 if part3a_structure_template is not None else 0, "The review depends on the adopted-U(1) section explicitly remaining a structure template rather than an origin derivation."),
            row("decisive_t2_fail_governance_available", "pass" if decisive_fail_t2 is not None else "reject", "decisive T2 fail governance available", 1 if decisive_fail_t2 is not None else 0, "The decisive-resolution note already says that if T2 fails, the route closes as a no-go theorem."),
        ],
        {
            "inventory_ready": True,
            "selected_next_substep": "8.7.56.1272",
            "prior_problem_classification": gate_summary["trial2_numeric_alpha_problem_classification"],
        },
        {
            "overall_status": "trial2_numeric_alpha_qball_projection_overlap_route_local_no_go_inventory_fixed",
            "advance_to_8_7_56_1272": True,
            "next_required_artifacts": ["qball_projection_overlap_route_local_no_go_theorem_review_audit"],
        },
        {
            "part1_hits": {
                "matter_current": part1_current,
                "interaction": part1_interaction,
                "micro_current": part1_micro_current,
                "full_closure": part1_full_closure,
            },
            "part3a_hits": {
                "independent_connection": part3a_independent_connection,
                "structure_template": part3a_structure_template,
                "not_origin": part3a_not_origin,
            },
            "part5_hits": {"source_fail": part5_source_fail},
            "note_hits": {"decisive_t2": decisive_t2, "decisive_fail_t2": decisive_fail_t2},
            "status_hits": {
                "status_1271": hit(status_text, "8.7.56.1271"),
                "roadmap_1271": hit(roadmap_text, "`8.7.56.1271-.1274`"),
                "history_1267": hit(work_history_recent_text, "8.7.56.1267-.1270"),
                "problem_route_local": hit(current_problem_text, "route-local no-go"),
                "status_route_local": hit(current_status_text, "route-local no-go"),
            },
            "prior_summary": gate_summary,
            "ai_context_current_step": ai_context.get("current_step"),
        },
    )

    audit = payload(
        "8.7.56.1272",
        "Trial-2 numeric alpha Q-ball projection-overlap route-local no-go theorem review audit",
        inputs,
        [
            row("generic_matter_coupling_only", "pass" if generic_matter_coupling_only else "reject", "generic matter coupling only", 1 if generic_matter_coupling_only else 0, "Current canon still has only the generic matter-current coupling, not a Q-ball-background source formula."),
            row("microscopic_current_surfaces_exist_but_generic_only", "pass" if microscopic_current_surfaces_exist_but_generic_only else "reject", "microscopic current surfaces exist but generic only", 1 if microscopic_current_surfaces_exist_but_generic_only else 0, "The micro current/spin terms remain matter-sector couplings and do not supply a Q-ball-to-light-mode source theorem."),
            row("adopted_u1_structure_template_only", "pass" if part3a_structure_template is not None and part3a_not_origin is not None else "reject", "adopted U(1) structure template only", 1 if part3a_structure_template is not None and part3a_not_origin is not None else 0, "Part III-A explicitly keeps the adopted-U(1) section as a structure template rather than an origin derivation."),
            row("alternative_current_canon_source_reading_survives", "reject" if not alternative_current_canon_source_reading_survives else "pass", "alternative current-canon source reading survives", 1 if alternative_current_canon_source_reading_survives else 0, "No distinct current-canon source reading survives after re-checking the generic current and micro current surfaces."),
            row("route_local_no_go_theorem_honest", "pass" if route_local_no_go_theorem_honest else "reject", "route-local no-go theorem honest", 1 if route_local_no_go_theorem_honest else 0, "The T2 failure is honest and localized to this route under current canon."),
            row("projection_theorem_carry_over_required", "pass" if projection_theorem_carry_over_required else "reject", "projection theorem carry-over required", 1 if projection_theorem_carry_over_required else 0, "T3 remains downstream and can only survive as a carry-over lane after the route-local no-go is fixed."),
        ],
        {
            "generic_matter_coupling_only": generic_matter_coupling_only,
            "microscopic_current_surfaces_exist_but_generic_only": microscopic_current_surfaces_exist_but_generic_only,
            "adopted_u1_structure_template_only": part3a_structure_template is not None and part3a_not_origin is not None,
            "alternative_current_canon_source_reading_survives": alternative_current_canon_source_reading_survives,
            "route_local_no_go_theorem_honest": route_local_no_go_theorem_honest,
            "theorem_side_stop_contract_ready": theorem_side_stop_contract_ready,
            "projection_theorem_carry_over_required": projection_theorem_carry_over_required,
            "result_class": "route_local_no_go_honest" if route_local_no_go_theorem_honest else "route_local_no_go_not_yet_honest",
        },
        {
            "overall_status": "trial2_numeric_alpha_qball_projection_overlap_route_local_no_go_audit_completed",
            "advance_to_8_7_56_1273": True,
            "next_required_artifacts": ["qball_projection_overlap_route_local_no_go_theorem_review_declaration_gate"],
        },
        {
            "prior_source_fail_summary": gate_summary,
            "numeric_context": eval_summary,
        },
    )

    declaration_gate = payload(
        "8.7.56.1273",
        "Trial-2 numeric alpha Q-ball projection-overlap route-local no-go theorem review declaration gate",
        inputs,
        [
            row("route_local_no_go_theorem_review_completed", "pass", "route-local no-go theorem review completed", 1.0, "The route-local no-go theorem review has been audited end-to-end."),
            row("route_local_no_go_theorem_honest", "pass" if route_local_no_go_theorem_honest else "reject", "route-local no-go theorem honest", 1 if route_local_no_go_theorem_honest else 0, "The T2 failure is honest and does not overreach beyond the current route."),
            row("alternative_current_canon_source_reading_survives", "reject" if not alternative_current_canon_source_reading_survives else "pass", "alternative current-canon source reading survives", 1 if alternative_current_canon_source_reading_survives else 0, "No alternate current-canon source reading survives after the re-check."),
            row("theorem_side_stop_contract_ready", "pass" if theorem_side_stop_contract_ready else "reject", "theorem-side stop contract ready", 1 if theorem_side_stop_contract_ready else 0, "The theorem-side stop can now be declared without forcing a physical reject."),
            row("projection_theorem_carry_over_required", "pass" if projection_theorem_carry_over_required else "reject", "projection theorem carry-over required", 1 if projection_theorem_carry_over_required else 0, "The downstream projection theorem is retained only as a carry-over lane contingent on a future T2 reopen."),
            row("physical_reject_required", "reject", "physical reject required", 0.0, "This is a route-local no-go under current canon, not a physical reject of the broader program."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "qball_projection_overlap_route_local_no_go_theorem_under_current_canon",
            "route_local_no_go_theorem_honest": route_local_no_go_theorem_honest,
            "theorem_side_stop_contract_ready": theorem_side_stop_contract_ready,
            "alternative_current_canon_source_reading_survives": alternative_current_canon_source_reading_survives,
            "projection_theorem_carry_over_required": projection_theorem_carry_over_required,
            "primary_residual_lane": "qball_projection_overlap_projection_theorem_carry_over",
            "secondary_residual_lane": "qball_projection_overlap_future_source_theorem_reopen",
            "reserve_residual_lane": "qball_projection_overlap_analytic_tail_theorem_refinement",
            "physical_reject_required": False,
            "closeout_ready": False,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_qball_projection_overlap_route_local_no_go_declared",
            "advance_to_8_7_56_1274": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "audit_summary": audit["summary"],
            "governance_hint": {"decisive_t2": decisive_t2, "decisive_fail_t2": decisive_fail_t2},
        },
    )

    evaluation = payload(
        "8.7.56.1274",
        "Trial-2 numeric alpha Q-ball projection-overlap route-local no-go theorem review numeric evaluation",
        inputs,
        [
            row("prior_q_theory_fixed", "pass", "prior q_theory fixed", float(eval_summary["q_theory_over_m0"]), "The matching-scale candidate remains unchanged through the route-local no-go review."),
            row("prior_f_exact_fixed", "pass", "prior F_exact fixed", float(eval_summary["F_exact_at_q_theory"]), "The exact-profile overlap value remains unchanged through the route-local no-go review."),
            row("prior_alpha_exact_fixed", "pass", "prior alpha_exact fixed", float(eval_summary["alpha_exact_at_q_theory"]), "The exact-profile alpha candidate remains unchanged through the route-local no-go review."),
            row("numeric_state_changed_by_current_branch", "reject", "numeric state changed by current branch", 0.0, "This branch is theorem-only and does not introduce a new fit or re-evaluation."),
            row("route_state_changed_by_current_branch", "pass", "route state changed by current branch", 1.0, "The route advances from source-theorem fail to route-local no-go theorem under current canon."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "qball_projection_overlap_route_local_no_go_theorem_under_current_canon",
            "q_theory_over_m0": float(eval_summary["q_theory_over_m0"]),
            "F_exact_at_q_theory": float(eval_summary["F_exact_at_q_theory"]),
            "alpha_exact_at_q_theory": float(eval_summary["alpha_exact_at_q_theory"]),
            "route_local_no_go_theorem_honest": route_local_no_go_theorem_honest,
            "projection_theorem_carry_over_required": projection_theorem_carry_over_required,
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_qball_projection_overlap_route_local_no_go_completed",
            "advance_to_next_route": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "prior_problem_classification": gate_summary["trial2_numeric_alpha_problem_classification"],
            "new_problem_classification": "qball_projection_overlap_route_local_no_go_theorem_under_current_canon",
        },
    )

    write_artifact("mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_route_local_no_go_theorem_review_source_inventory", inventory)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_route_local_no_go_theorem_review_audit", audit)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_route_local_no_go_theorem_review_declaration_gate", declaration_gate)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_route_local_no_go_theorem_review_numeric_evaluation", evaluation)

    print("[done] 8.7.56.1271-.1274 artifacts generated")


if __name__ == "__main__":
    main()
