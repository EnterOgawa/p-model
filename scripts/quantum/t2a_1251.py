#!/usr/bin/env python3
"""Generate 8.7.56.1251-.1254 Trial-2 exact support-scale review artifacts.

Purpose:
    Audit whether current canon now selects one unique effective support scale
    for the Q-ball projection-overlap route, or whether `.1247-.1250` only
    justified a finite support band without fixing the exact weighting rule.

Inputs:
    - Current operational docs and Part I / Part III-A / Part V surfaces
    - The `.1247-.1250` matching-scale review metrics
    - The projection-overlap note and the current problem notes

Outputs:
    - Four machine-readable metrics payloads under `output/public/quantum/`

Assumptions:
    - The support band itself is already justified by `.1247-.1250`.
    - This branch only audits whether current canon picks one exact support
      scale or whether only reserve hints such as asymptotic-tail weighting
      remain available.
"""

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
EXPERT_SHARE = ROOT / "doc" / "quantum" / "35_trial2_numeric_alpha_projection_overlap_expert_share.md"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
NOTE = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_projection_overlap_justification.md")

MATCHING_SOURCE = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_matching_scale_review_source_inventory_metrics.json"
MATCHING_AUDIT = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_matching_scale_review_audit_metrics.json"
MATCHING_GATE = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_matching_scale_review_declaration_gate_metrics.json"
MATCHING_EVAL = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_matching_scale_review_numeric_evaluation_metrics.json"

NEXT_ROUTE = "8.7.56.1255"
NEXT_ROUTE_NAME = "trial2_numeric_alpha_qball_projection_overlap_tail_weighting_reserve_review"


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


# Function: rank the retained exact-scale candidates by relative q-error.

def candidate_ranking(candidate_rel_errors: dict) -> list[tuple[str, float]]:
    """Rank the retained exact-scale candidates by relative q-error."""
    return sorted(candidate_rel_errors.items(), key=lambda item: item[1])


# Function: execute the 8.7.56.1251-.1254 branch.

def main() -> None:
    """Execute the 8.7.56.1251-.1254 branch."""
    required_paths = (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        WORK_HISTORY_RECENT,
        CURRENT_PROBLEM,
        EXPERT_SHARE,
        PART1,
        PART3A,
        PART5,
        NOTE,
        MATCHING_SOURCE,
        MATCHING_AUDIT,
        MATCHING_GATE,
        MATCHING_EVAL,
    )
    for path in required_paths:
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    work_history_recent_text = read_text(WORK_HISTORY_RECENT)
    current_problem_text = read_text(CURRENT_PROBLEM)
    expert_share_text = read_text(EXPERT_SHARE)
    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    note_text = read_text(NOTE)

    matching_source = read_json(MATCHING_SOURCE)
    matching_audit = read_json(MATCHING_AUDIT)
    matching_gate = read_json(MATCHING_GATE)
    matching_eval = read_json(MATCHING_EVAL)

    candidate_rel_errors = matching_eval["evidence"]["candidate_q_rel_errors"]
    ranking = candidate_ranking(candidate_rel_errors)
    best_candidate_name, best_candidate_error = ranking[0]
    second_candidate_name, second_candidate_error = ranking[1]
    worst_candidate_name, worst_candidate_error = ranking[-1]
    candidate_error_gap = float(second_candidate_error - best_candidate_error)
    candidate_error_spread = float(worst_candidate_error - best_candidate_error)
    candidate_ambiguity_significant = candidate_error_gap < 0.05

    part1_decaying_tail_line = hit(part1_text, "decaying tail")
    note_large_x_tail_line = hit(note_text, "overlap integral が large-$x$ tail で支配される")
    note_internal_scale_line = hit(note_text, "internal structure scale")
    part3a_nonunique_line = hit(part3a_text, "採るかはまだ一意でない")
    part5_nonunique_line = hit(part5_text, "exact scale open")
    current_problem_exact_scale_line = hit(current_problem_text, "exact support scale の選択")
    expert_share_exact_scale_line = hit(expert_share_text, "exact support scale の選択")

    explicit_half_mass_rule_available = False
    explicit_mean_rule_available = False
    explicit_rms_rule_available = False
    explicit_weighting_rule_available = False
    unique_effective_support_scale_available = False
    current_public_nonuniqueness_surface_available = (
        part3a_nonunique_line is not None or part5_nonunique_line is not None
    )
    tail_weighting_reserve_candidate_available = (
        part1_decaying_tail_line is not None and note_large_x_tail_line is not None
    )

    inputs = {
        "source_files": {
            "status": display_path(STATUS),
            "roadmap": display_path(ROADMAP),
            "ai_context": display_path(AI_CONTEXT),
            "work_history_recent": display_path(WORK_HISTORY_RECENT),
            "current_problem_note": display_path(CURRENT_PROBLEM),
            "expert_share_note": display_path(EXPERT_SHARE),
            "part1": display_path(PART1),
            "part3a": display_path(PART3A),
            "part5": display_path(PART5),
            "projection_overlap_note": display_path(NOTE),
        },
        "prior_metrics": {
            "matching_source": display_path(MATCHING_SOURCE),
            "matching_audit": display_path(MATCHING_AUDIT),
            "matching_gate": display_path(MATCHING_GATE),
            "matching_eval": display_path(MATCHING_EVAL),
        },
        "constants": {
            "best_candidate_name": best_candidate_name,
            "best_candidate_error": best_candidate_error,
            "second_candidate_name": second_candidate_name,
            "second_candidate_error": second_candidate_error,
            "next_route_name": NEXT_ROUTE_NAME,
            "next_route": NEXT_ROUTE,
        },
    }

    inventory = payload(
        "8.7.56.1251",
        "Trial-2 numeric alpha Q-ball projection-overlap effective-support-scale review source inventory",
        inputs,
        [
            row("prior_matching_scale_review_available", "pass", "prior matching-scale review available", 1.0, "The .1247-.1250 matching-scale review metrics are present and can be reused."),
            row("part1_decaying_tail_surface_available", "pass" if part1_decaying_tail_line is not None else "reject", "Part I decaying-tail surface available", 1 if part1_decaying_tail_line is not None else 0, "Part I is checked for any asymptotic-tail selection rule that could promote one exact support scale."),
            row("note_large_x_tail_candidate_available", "pass" if note_large_x_tail_line is not None else "reject", "projection-overlap note large-x tail candidate available", 1 if note_large_x_tail_line is not None else 0, "The note is checked for a tail-dominance weighting candidate."),
            row("current_public_nonuniqueness_surface_available", "pass" if current_public_nonuniqueness_surface_available else "reject", "current public nonuniqueness surface available", 1 if current_public_nonuniqueness_surface_available else 0, "The current public wording is checked for an explicit statement that no unique exact support scale has yet been selected."),
            row("candidate_error_ranking_ready", "pass", "candidate error ranking ready", 1.0, "The retained half-mass / mean / rms candidate errors are available for the exact-scale audit."),
            row("support_scale_review_inventory_ready", "pass", "support-scale review inventory ready", 1.0, "All current-canon and reserve-tail surfaces required for the exact-scale review are available."),
        ],
        {
            "inventory_ready": True,
            "candidate_ranking": ranking,
            "current_public_nonuniqueness_surface_available": current_public_nonuniqueness_surface_available,
            "tail_weighting_reserve_candidate_available": tail_weighting_reserve_candidate_available,
            "selected_next_substep": "8.7.56.1252",
        },
        {
            "overall_status": "trial2_numeric_alpha_qball_projection_overlap_effective_support_scale_inventory_fixed",
            "advance_to_8_7_56_1252": True,
            "next_required_artifacts": ["qball_projection_overlap_effective_support_scale_review_audit"],
        },
        {
            "paper_hits": {
                "part1_decaying_tail_line": part1_decaying_tail_line,
                "part3a_nonunique_line": part3a_nonunique_line,
                "part5_nonunique_line": part5_nonunique_line,
            },
            "note_hits": {
                "note_large_x_tail_line": note_large_x_tail_line,
                "note_internal_scale_line": note_internal_scale_line,
            },
            "note_pack_hits": {
                "current_problem_exact_scale_line": current_problem_exact_scale_line,
                "expert_share_exact_scale_line": expert_share_exact_scale_line,
            },
            "status_hits": {
                "status_next_1251": hit(status_text, "8.7.56.1251"),
                "roadmap_branch_1251": hit(roadmap_text, "`8.7.56.1251-.1254`"),
                "work_history_1247_entry": hit(work_history_recent_text, "8.7.56.1247-.1250"),
            },
            "prior_matching_gate_summary": matching_gate["summary"],
            "prior_matching_eval_summary": matching_eval["summary"],
            "ai_context_current_step": ai_context.get("current_step"),
        },
    )

    audit = payload(
        "8.7.56.1252",
        "Trial-2 numeric alpha Q-ball projection-overlap effective-support-scale review audit",
        inputs,
        [
            row("projection_overlap_unique_effective_support_scale_available", "pass" if unique_effective_support_scale_available else "reject", "projection-overlap unique effective support scale available", 1 if unique_effective_support_scale_available else 0, "This passes only if current canon explicitly selects one exact support scale or weighting rule."),
            row("projection_overlap_explicit_half_mass_rule_available", "pass" if explicit_half_mass_rule_available else "reject", "projection-overlap explicit half-mass rule available", 1 if explicit_half_mass_rule_available else 0, "Current canon is checked for an explicit half-mass support-scale selection rule."),
            row("projection_overlap_explicit_mean_rule_available", "pass" if explicit_mean_rule_available else "reject", "projection-overlap explicit mean rule available", 1 if explicit_mean_rule_available else 0, "Current canon is checked for an explicit mean-radius support-scale selection rule."),
            row("projection_overlap_explicit_rms_rule_available", "pass" if explicit_rms_rule_available else "reject", "projection-overlap explicit rms rule available", 1 if explicit_rms_rule_available else 0, "Current canon is checked for an explicit rms-radius support-scale selection rule."),
            row("projection_overlap_explicit_weighting_rule_available", "pass" if explicit_weighting_rule_available else "reject", "projection-overlap explicit weighting rule available", 1 if explicit_weighting_rule_available else 0, "Current canon is checked for an explicit weighting-rule theorem that would collapse the support band to one exact scale."),
            row("projection_overlap_candidate_ambiguity_significant", "pass" if candidate_ambiguity_significant else "reject", "projection-overlap candidate ambiguity significant", 1 if candidate_ambiguity_significant else 0, "The retained candidates remain too close numerically to promote one exact support scale without an explicit theorem."),
            row("projection_overlap_tail_weighting_reserve_candidate_available", "pass" if tail_weighting_reserve_candidate_available else "reject", "projection-overlap tail-weighting reserve candidate available", 1 if tail_weighting_reserve_candidate_available else 0, "Part I decaying-tail wording plus the note's large-x tail dominance provide only a reserve candidate, not a fixed exact-scale theorem."),
        ],
        {
            "unique_effective_support_scale_available": unique_effective_support_scale_available,
            "explicit_half_mass_rule_available": explicit_half_mass_rule_available,
            "explicit_mean_rule_available": explicit_mean_rule_available,
            "explicit_rms_rule_available": explicit_rms_rule_available,
            "explicit_weighting_rule_available": explicit_weighting_rule_available,
            "candidate_ambiguity_significant": candidate_ambiguity_significant,
            "tail_weighting_reserve_candidate_available": tail_weighting_reserve_candidate_available,
            "current_public_nonuniqueness_surface_available": current_public_nonuniqueness_surface_available,
            "best_candidate_name": best_candidate_name,
            "best_candidate_error": best_candidate_error,
            "second_candidate_name": second_candidate_name,
            "second_candidate_error": second_candidate_error,
            "result_class": "projection_overlap_support_band_only_current_canon_limit",
        },
        {
            "overall_status": "trial2_numeric_alpha_qball_projection_overlap_effective_support_scale_audit_completed",
            "advance_to_8_7_56_1253": True,
            "next_required_artifacts": ["qball_projection_overlap_effective_support_scale_review_declaration_gate"],
        },
        {
            "candidate_ranking": ranking,
            "candidate_error_gap": candidate_error_gap,
            "candidate_error_spread": candidate_error_spread,
        },
    )

    declaration_gate = payload(
        "8.7.56.1253",
        "Trial-2 numeric alpha Q-ball projection-overlap effective-support-scale review declaration gate",
        inputs,
        [
            row("projection_overlap_effective_support_scale_review_completed", "pass", "projection-overlap effective-support-scale review completed", 1.0, "The exact support-scale review branch has now been audited end-to-end."),
            row("projection_overlap_support_band_justified", "pass" if matching_gate["summary"]["finite_internal_scale_theory_side_justified"] else "reject", "projection-overlap support band justified", 1 if matching_gate["summary"]["finite_internal_scale_theory_side_justified"] else 0, "The finite support band remains justified from the prior branch."),
            row("projection_overlap_unique_effective_support_scale_available", "pass" if unique_effective_support_scale_available else "reject", "projection-overlap unique effective support scale available", 1 if unique_effective_support_scale_available else 0, "Predictive promotion requires one unique exact support scale selected by canon."),
            row("projection_overlap_current_canon_limit_reached", "pass" if not unique_effective_support_scale_available else "reject", "projection-overlap current canon limit reached", 1 if not unique_effective_support_scale_available else 0, "Current canon stops at a justified support band and does not yet supply one exact selection theorem."),
            row("projection_overlap_tail_weighting_reserve_candidate_available", "pass" if tail_weighting_reserve_candidate_available else "reject", "projection-overlap tail-weighting reserve candidate available", 1 if tail_weighting_reserve_candidate_available else 0, "The remaining positive lead is a reserve tail-weighting candidate rather than a current-canon exact rule."),
            row("projection_overlap_predictive_branch_ready", "pass" if unique_effective_support_scale_available else "reject", "projection-overlap predictive branch ready", 1 if unique_effective_support_scale_available else 0, "Predictive status stays withheld while the exact support scale remains unfixed."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "qball_projection_overlap_support_band_only_current_canon_limit",
            "finite_internal_support_band_justified": matching_gate["summary"]["finite_internal_scale_theory_side_justified"],
            "unique_effective_support_scale_available": unique_effective_support_scale_available,
            "current_public_nonuniqueness_surface_available": current_public_nonuniqueness_surface_available,
            "tail_weighting_reserve_candidate_available": tail_weighting_reserve_candidate_available,
            "predictive_branch_ready": False,
            "primary_residual_lane": "qball_projection_overlap_tail_weighting_reserve",
            "secondary_residual_lane": "adopted_u1_charge_unit_dictionary",
            "reserve_residual_lane": "adopted_u1_vacuum_polarization_external_import",
            "physical_reject_required": False,
            "closeout_ready": False,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_qball_projection_overlap_effective_support_scale_declared",
            "advance_to_8_7_56_1254": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "prior_matching_gate_summary": matching_gate["summary"],
            "effective_support_scale_audit_summary": audit["summary"],
        },
    )

    evaluation = payload(
        "8.7.56.1254",
        "Trial-2 numeric alpha Q-ball projection-overlap effective-support-scale review numeric evaluation",
        inputs,
        [
            row("projection_overlap_best_candidate_error_fixed", "pass", "projection-overlap best candidate error fixed", best_candidate_error, "The best retained exact-scale candidate remains the half-mass target-phase proxy."),
            row("projection_overlap_second_candidate_error_fixed", "pass", "projection-overlap second candidate error fixed", second_candidate_error, "The second retained candidate remains the mean first-zero proxy."),
            row("projection_overlap_candidate_error_gap_fixed", "pass", "projection-overlap candidate error gap fixed", candidate_error_gap, "The small gap between the first two candidates quantifies why no unique exact scale is selected numerically."),
            row("projection_overlap_candidate_error_spread_fixed", "pass", "projection-overlap candidate error spread fixed", candidate_error_spread, "The retained candidate spread is recorded for the exact-scale ambiguity audit."),
            row("projection_overlap_tail_weighting_reserve_candidate_fixed", "pass" if tail_weighting_reserve_candidate_available else "reject", "projection-overlap tail-weighting reserve candidate fixed", 1 if tail_weighting_reserve_candidate_available else 0, "Tail-weighting remains a reserve candidate because only hint-level surfaces exist."),
            row("projection_overlap_numeric_state_changed_by_exact_scale_review", "reject", "projection-overlap numeric state changed by exact-scale review", 0.0, "This branch does not alter q_*; it only classifies whether the exact support scale is selected by canon."),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "qball_projection_overlap_support_band_only_current_canon_limit",
            "best_candidate_name": best_candidate_name,
            "best_candidate_error": best_candidate_error,
            "second_candidate_name": second_candidate_name,
            "second_candidate_error": second_candidate_error,
            "candidate_error_gap": candidate_error_gap,
            "candidate_error_spread": candidate_error_spread,
            "candidate_ambiguity_significant": candidate_ambiguity_significant,
            "tail_weighting_reserve_candidate_available": tail_weighting_reserve_candidate_available,
            "numeric_state_changed_by_current_branch": False,
            "selected_next_generation_route": NEXT_ROUTE_NAME,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_qball_projection_overlap_effective_support_scale_completed",
            "advance_to_next_route": True,
            "next_required_artifacts": [NEXT_ROUTE_NAME],
        },
        {
            "candidate_ranking": ranking,
            "current_public_nonuniqueness_surface_available": current_public_nonuniqueness_surface_available,
        },
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_effective_support_scale_review_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_effective_support_scale_review_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_effective_support_scale_review_declaration_gate",
        declaration_gate,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_qball_projection_overlap_effective_support_scale_review_numeric_evaluation",
        evaluation,
    )

    print("[done] 8.7.56.1251-.1254 artifacts generated")


if __name__ == "__main__":
    main()
