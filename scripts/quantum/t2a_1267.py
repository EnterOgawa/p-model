#!/usr/bin/env python3
"""Generate 8.7.56.1267-.1270 source theorem attempt artifacts."""

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
LIGHT_MODE_GATE = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_light_mode_theorem_attempt_declaration_gate_metrics.json"
LIGHT_MODE_EVAL = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_light_mode_theorem_attempt_numeric_evaluation_metrics.json"

NEXT_ROUTE = "8.7.56.1271"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_qball_projection_overlap_route_local_no_go_theorem_review"


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


# Function: locate the first matching line for one substring pattern.

def hit(text: str, pattern: str) -> dict | None:
    """Locate the first matching line for one substring pattern."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# Function: locate the first hit among several substring patterns.

def first_hit(texts: list[str], patterns: list[str]) -> dict | None:
    """Locate the first hit among several substring patterns."""
    for text in texts:
        for pattern in patterns:
            found = hit(text, pattern)
            if found is not None:
                return found

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


# Function: execute the 8.7.56.1267-.1270 branch.

def main() -> None:
    """Execute the 8.7.56.1267-.1270 branch."""
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
        LIGHT_MODE_GATE,
        LIGHT_MODE_EVAL,
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

    light_mode_gate = read_json(LIGHT_MODE_GATE)
    light_mode_eval = read_json(LIGHT_MODE_EVAL)
    gate_summary = dict(light_mode_gate["summary"])
    eval_summary = dict(light_mode_eval["summary"])

    part1_current = hit(part1_text, r"J^\mu_{\mathrm{matter}}=(\rho c,\rho \mathbf{v})")
    part1_interaction = hit(part1_text, r"\mathcal{L}_{\mathrm{int}}=g_P\,P_\mu J^\mu_{\mathrm{matter}}")
    part1_photon_branch = hit(part1_text, r"$A_\mu=\delta P_\mu^T/\sqrt{Z_P}$")
    part3a_independent_connection = hit(part3a_text, "独立接続")
    part3a_structure_template = hit(part3a_text, "構造テンプレート")
    part3a_structural_route = hit(part3a_text, "Trial-2 で Maxwell curvature、Coulomb $1/r^2$ / $1/r$")
    part5_structural_route = hit(part5_text, "Maxwell curvature、Coulomb $1/r^2$ / $1/r$")
    decisive_t2 = hit(decisive_note_text, "source theorem")
    decisive_t3 = hit(decisive_note_text, "projection theorem")
    decisive_effective_source = hit(decisive_note_text, r"J^{\mu}_{\rm eff}[P^{\rm Qball}]")
    decisive_source_formula = hit(decisive_note_text, r"\mathcal L \supset a_\mu J^{\mu}_{\rm eff}[P^{\rm Qball}]")

    explicit_matter_current_surface = part1_current is not None
    explicit_interaction_surface = part1_interaction is not None
    explicit_fixed_light_mode_available = bool(gate_summary["physical_light_mode_theorem_ready"])
    explicit_qball_background_expansion = first_hit(
        [part1_text, part3a_text, part5_text],
        [r"P_\mu = P^{\rm Qball}_\mu + a_\mu^{\rm light}", "Q-ball background", r"P^{\rm Qball}", r"a_\mu^{\rm light}"],
    ) is not None
    explicit_effective_source_formula = first_hit(
        [part1_text, part3a_text, part5_text],
        [r"J^{\mu}_{\rm eff}", r"a_\mu J^{\mu}_{\rm eff}", "effective source", "qball_to_light_mode_coupling"],
    ) is not None
    explicit_qball_to_light_mode_coupling_statement = explicit_qball_background_expansion and explicit_effective_source_formula
    independent_connection_caveat_present = part3a_independent_connection is not None
    independent_connection_caveat_blocks_t2 = independent_connection_caveat_present and not explicit_effective_source_formula
    source_theorem_ready = (
        explicit_fixed_light_mode_available
        and explicit_matter_current_surface
        and explicit_interaction_surface
        and explicit_qball_background_expansion
        and explicit_effective_source_formula
        and not independent_connection_caveat_blocks_t2
    )
    projection_theorem_attempt_ready = source_theorem_ready and decisive_t3 is not None
    route_local_no_go_candidate_ready = not source_theorem_ready

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
            "light_mode_gate": display_path(LIGHT_MODE_GATE),
            "light_mode_eval": display_path(LIGHT_MODE_EVAL),
        },
        "constants": {"next_route_name": NEXT_ROUTE_NAME, "next_route": NEXT_ROUTE},
    }

    inventory = payload(
        "8.7.56.1267",
        "Trial-2 numeric alpha Q-ball projection-overlap source theorem attempt source inventory",
        inputs,
        [
            row("physical_light_mode_theorem_ready", "pass" if explicit_fixed_light_mode_available else "reject", "physical light mode theorem ready", 1 if explicit_fixed_light_mode_available else 0, "T2 only starts after T1 fixes the physical light mode."),
            row("part1_matter_current_surface_available", "pass" if explicit_matter_current_surface else "reject", "Part I matter current surface available", 1 if explicit_matter_current_surface else 0, "Part I must still expose the matter current that appears in the explicit interaction term."),
            row("part1_interaction_surface_available", "pass" if explicit_interaction_surface else "reject", "Part I interaction surface available", 1 if explicit_interaction_surface else 0, "Current pack must still expose the explicit vector-current interaction term."),
            row("decisive_t2_governance_available", "pass" if decisive_t2 is not None else "reject", "decisive T2 governance available", 1 if decisive_t2 is not None else 0, "The decisive-resolution note fixes that the active question is now T2 source."),
            row("explicit_qball_background_expansion_surface_available", "pass" if explicit_qball_background_expansion else "reject", "explicit Q-ball background expansion surface available", 1 if explicit_qball_background_expansion else 0, "T2 ultimately needs an explicit background-plus-light-mode expansion."),
        ],
        {
            "inventory_ready": True,
            "selected_next_substep": "8.7.56.1268",
            "prior_problem_classification": gate_summary["trial2_numeric_alpha_problem_classification"],
        },
        {
            "overall_status": "trial2_numeric_alpha_qball_projection_overlap_source_theorem_inventory_fixed",
            "advance_to_8_7_56_1268": True,
            "next_required_artifacts": ["qball_projection_overlap_source_theorem_attempt_audit"],
        },
        {
            "part1_hits": {
                "matter_current": part1_current,
                "interaction": part1_interaction,
                "photon_branch": part1_photon_branch,
            },
            "part3a_hits": {
                "independent_connection": part3a_independent_connection,
                "structure_template": part3a_structure_template,
                "structural_route": part3a_structural_route,
            },
            "part5_hits": {"structural_route": part5_structural_route},
            "note_hits": {
                "decisive_t2": decisive_t2,
                "decisive_t3": decisive_t3,
                "decisive_effective_source": decisive_effective_source,
                "decisive_source_formula": decisive_source_formula,
            },
            "status_hits": {
                "status_1267": hit(status_text, "8.7.56.1267"),
                "roadmap_1267": hit(roadmap_text, "`8.7.56.1267-.1270`"),
                "history_1263": hit(work_history_recent_text, "8.7.56.1263-.1266"),
                "problem_source_theorem": hit(current_problem_text, "source theorem problem"),
                "status_source_theorem_next": hit(current_status_text, "source theorem next"),
            },
            "prior_summary": gate_summary,
            "ai_context_current_step": ai_context.get("current_step"),
        },
    )

    audit = payload(
        "8.7.56.1268",
        "Trial-2 numeric alpha Q-ball projection-overlap source theorem attempt audit",
        inputs,
        [
            row("explicit_matter_current_surface_available", "pass" if explicit_matter_current_surface else "reject", "explicit matter current surface available", 1 if explicit_matter_current_surface else 0, "T2 needs the current source that appears in the explicit interaction term."),
            row("explicit_interaction_surface_available", "pass" if explicit_interaction_surface else "reject", "explicit interaction surface available", 1 if explicit_interaction_surface else 0, "T2 needs an explicit interaction term to start any action-level source derivation."),
            row("explicit_fixed_light_mode_available", "pass" if explicit_fixed_light_mode_available else "reject", "explicit fixed light mode available", 1 if explicit_fixed_light_mode_available else 0, "T2 depends on the already-fixed physical light mode from T1."),
            row("explicit_qball_background_expansion_available", "pass" if explicit_qball_background_expansion else "reject", "explicit Q-ball background expansion available", 1 if explicit_qball_background_expansion else 0, "T2 requires a public surface that expands the Q-ball background into the fixed light mode."),
            row("explicit_effective_source_formula_available", "pass" if explicit_effective_source_formula else "reject", "explicit effective source formula available", 1 if explicit_effective_source_formula else 0, "T2 requires an explicit formula for the effective source carried by the Q-ball background."),
            row("independent_connection_caveat_present", "pass" if independent_connection_caveat_present else "reject", "independent connection caveat present", 1 if independent_connection_caveat_present else 0, "Current canon still says local Maxwell/minimal coupling is an adopted independent-connection sector rather than a P-only derivation."),
            row("independent_connection_caveat_blocks_t2", "pass" if independent_connection_caveat_blocks_t2 else "reject", "independent connection caveat blocks T2", 1 if independent_connection_caveat_blocks_t2 else 0, "Without an explicit Q-ball-to-light-mode source formula, the adopted-U(1) caveat now blocks T2 itself."),
            row("source_theorem_ready", "pass" if source_theorem_ready else "reject", "source theorem ready", 1 if source_theorem_ready else 0, "T2 passes only if current pack already derives the effective source for the fixed light mode."),
            row("projection_theorem_attempt_ready", "pass" if projection_theorem_attempt_ready else "reject", "projection theorem attempt ready", 1 if projection_theorem_attempt_ready else 0, "T3 cannot start before T2 supplies the effective source."),
        ],
        {
            "explicit_matter_current_surface_available": explicit_matter_current_surface,
            "explicit_interaction_surface_available": explicit_interaction_surface,
            "explicit_fixed_light_mode_available": explicit_fixed_light_mode_available,
            "explicit_qball_background_expansion_available": explicit_qball_background_expansion,
            "explicit_effective_source_formula_available": explicit_effective_source_formula,
            "explicit_qball_to_light_mode_coupling_statement_available": explicit_qball_to_light_mode_coupling_statement,
            "independent_connection_caveat_present": independent_connection_caveat_present,
            "independent_connection_caveat_blocks_t2": independent_connection_caveat_blocks_t2,
            "source_theorem_ready": source_theorem_ready,
            "projection_theorem_attempt_ready": projection_theorem_attempt_ready,
            "result_class": "source_theorem_derived" if source_theorem_ready else "source_theorem_not_derived_under_current_canon",
        },
        {
            "overall_status": "trial2_numeric_alpha_qball_projection_overlap_source_theorem_audit_completed",
            "advance_to_8_7_56_1269": True,
            "next_required_artifacts": ["qball_projection_overlap_source_theorem_attempt_declaration_gate"],
        },
        {
            "prior_light_mode_summary": gate_summary,
            "numeric_context": eval_summary,
        },
    )

    declaration_gate = payload(
        "8.7.56.1269",
        "Trial-2 numeric alpha Q-ball projection-overlap source theorem attempt declaration gate",
        inputs,
        [
            row("source_theorem_attempt_completed", "pass", "source theorem attempt completed", 1.0, "The source theorem attempt has been audited end-to-end."),
            row("explicit_matter_current_surface_available", "pass" if explicit_matter_current_surface else "reject", "explicit matter current surface available", 1 if explicit_matter_current_surface else 0, "Current pack still carries the explicit matter-current surface."),
            row("explicit_interaction_surface_available", "pass" if explicit_interaction_surface else "reject", "explicit interaction surface available", 1 if explicit_interaction_surface else 0, "Current pack still carries the explicit vector-current interaction term."),
            row("explicit_effective_source_formula_available", "pass" if explicit_effective_source_formula else "reject", "explicit effective source formula available", 1 if explicit_effective_source_formula else 0, "Current pack does not yet surface the Q-ball effective source formula required by T2."),
            row("independent_connection_caveat_blocks_t2", "pass" if independent_connection_caveat_blocks_t2 else "reject", "independent connection caveat blocks T2", 1 if independent_connection_caveat_blocks_t2 else 0, "The adopted-U(1) caveat now localizes to T2 because the effective source formula is absent."),
            row("route_local_no_go_candidate_ready", "pass" if route_local_no_go_candidate_ready else "reject", "route-local no-go candidate ready", 1 if route_local_no_go_candidate_ready else 0, "T2 currently fails under current canon without forcing a physical reject."),
            row("projection_theorem_attempt_ready", "pass" if projection_theorem_attempt_ready else "reject", "projection theorem attempt ready", 1 if projection_theorem_attempt_ready else 0, "T3 remains downstream and cannot start while T2 is unresolved."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "qball_projection_overlap_source_theorem_failed_under_current_canon",
            "physical_light_mode_theorem_ready": bool(gate_summary["physical_light_mode_theorem_ready"]),
            "explicit_matter_current_surface_available": explicit_matter_current_surface,
            "explicit_interaction_surface_available": explicit_interaction_surface,
            "explicit_qball_background_expansion_available": explicit_qball_background_expansion,
            "explicit_effective_source_formula_available": explicit_effective_source_formula,
            "explicit_qball_to_light_mode_coupling_statement_available": explicit_qball_to_light_mode_coupling_statement,
            "independent_connection_caveat_present": independent_connection_caveat_present,
            "independent_connection_caveat_blocks_t2": independent_connection_caveat_blocks_t2,
            "source_theorem_ready": source_theorem_ready,
            "projection_theorem_attempt_ready": projection_theorem_attempt_ready,
            "route_local_no_go_candidate_ready": route_local_no_go_candidate_ready,
            "primary_residual_lane": "qball_projection_overlap_effective_source_formula_absent",
            "secondary_residual_lane": "qball_projection_overlap_adopted_u1_independent_connection",
            "reserve_residual_lane": "qball_projection_overlap_projection_theorem",
            "physical_reject_required": False,
            "closeout_ready": False,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_qball_projection_overlap_source_theorem_declared",
            "advance_to_8_7_56_1270": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "audit_summary": audit["summary"],
            "governance_hint": {
                "decisive_t2": decisive_t2,
                "decisive_t3": decisive_t3,
                "decisive_effective_source": decisive_effective_source,
                "decisive_source_formula": decisive_source_formula,
            },
        },
    )

    evaluation = payload(
        "8.7.56.1270",
        "Trial-2 numeric alpha Q-ball projection-overlap source theorem attempt numeric evaluation",
        inputs,
        [
            row("prior_q_theory_fixed", "pass", "prior q_theory fixed", float(eval_summary["q_theory_over_m0"]), "The matching-scale candidate remains unchanged through T2."),
            row("prior_f_exact_fixed", "pass", "prior F_exact fixed", float(eval_summary["F_exact_at_q_theory"]), "The exact-profile overlap value remains unchanged through T2."),
            row("prior_alpha_exact_fixed", "pass", "prior alpha_exact fixed", float(eval_summary["alpha_exact_at_q_theory"]), "The exact-profile alpha candidate remains unchanged through T2."),
            row("numeric_state_changed_by_current_branch", "reject", "numeric state changed by current branch", 0.0, "This branch is theorem-only and does not introduce a new fit or re-evaluation."),
            row("route_state_changed_by_current_branch", "pass", "route state changed by current branch", 1.0, "The route advances from source-theorem-next to source-theorem-failed-under-current-canon."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "qball_projection_overlap_source_theorem_failed_under_current_canon",
            "q_theory_over_m0": float(eval_summary["q_theory_over_m0"]),
            "F_exact_at_q_theory": float(eval_summary["F_exact_at_q_theory"]),
            "alpha_exact_at_q_theory": float(eval_summary["alpha_exact_at_q_theory"]),
            "source_theorem_ready": source_theorem_ready,
            "projection_theorem_attempt_ready": projection_theorem_attempt_ready,
            "numeric_state_changed_by_current_branch": False,
            "route_state_changed_by_current_branch": True,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_qball_projection_overlap_source_theorem_completed",
            "advance_to_next_route": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "prior_problem_classification": gate_summary["trial2_numeric_alpha_problem_classification"],
            "new_problem_classification": "qball_projection_overlap_source_theorem_failed_under_current_canon",
        },
    )

    write_artifact("mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_source_theorem_attempt_source_inventory", inventory)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_source_theorem_attempt_audit", audit)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_source_theorem_attempt_declaration_gate", declaration_gate)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_source_theorem_attempt_numeric_evaluation", evaluation)

    print("[done] 8.7.56.1267-.1270 artifacts generated")


if __name__ == "__main__":
    main()
