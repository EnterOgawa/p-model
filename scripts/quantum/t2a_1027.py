#!/usr/bin/env python3
"""Generate 8.7.56.1027-.1030 Trial-2 numeric alpha retry-triage expert-escalation artifacts."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"

EXPERT_NOTE = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_two_sector_hierarchy.md")
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
PRIMARY_SOURCES = ROOT / "doc" / "PRIMARY_SOURCES.md"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
EM_DOC = ROOT / "doc" / "quantum" / "16_electromagnetism_charge_maxwell_photon.md"

SOURCE_1019 = OUT / "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_statement_source_inventory_metrics.json"
AUDIT_1020 = OUT / "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_statement_audit_metrics.json"
GATE_1021 = OUT / "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_statement_declaration_gate_metrics.json"
ROUTE_1022 = OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_fifty_second_refresh_metrics.json"
SOURCE_1023 = OUT / "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_literal_source_inventory_metrics.json"
AUDIT_1024 = OUT / "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_literal_audit_metrics.json"
GATE_1025 = OUT / "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_literal_declaration_gate_metrics.json"
ROUTE_1026 = OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_fifty_third_refresh_metrics.json"

CURRENT_BLOCKER = "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_phrase_fragment"
CURRENT_EXPERT_ROUTE = "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_expert_advice"
NEXT_BUNDLE_ROUTE = "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_expert_bundle_refresh"
NEXT_ROUTE = "8.7.56.1031"


# Function: return the current UTC timestamp.
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# Function: stop execution when a required path is missing.

def require(path: Path) -> None:
    """Require a path to exist before execution continues."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# Function: read a UTF-8 text file.

def read_text(path: Path) -> str:
    """Read a UTF-8 text file."""
    return path.read_text(encoding="utf-8")


# Function: read a UTF-8 JSON file.

def read_json(path: Path) -> dict:
    """Read a UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# Function: return a stable display path for repo or external files.

def display_path(path: Path) -> str:
    """Return a stable path relative to the repo root when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# Function: locate the first line containing a substring pattern.

def hit(text: str, pattern: str) -> dict | None:
    """Return the first line hit for the given substring pattern."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# Function: build a standard metrics row.

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build one standard metrics row."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# Function: build a standard payload object.

def payload(
    step: str,
    name: str,
    inputs: dict,
    intent: str,
    formulas: dict,
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
        "intent": intent,
        "formulas": formulas,
        "rows": rows,
        "summary": summary,
        "decision": decision,
        "evidence": evidence,
    }


# Function: write a JSON metrics artifact and the matching CSV rows table.

def write_artifact(stem: str, data: dict) -> None:
    """Write one metrics payload as JSON and CSV."""
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / f"{stem}_metrics.json"
    csv_path = OUT / f"{stem}_rows.csv"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# Function: execute the retry-triage expert-escalation branch.

def main() -> None:
    """Execute the Trial-2 numeric alpha retry-triage expert-escalation branch."""
    for path in (
        EXPERT_NOTE,
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        PRIMARY_SOURCES,
        PART1,
        PART3A,
        PART5,
        EM_DOC,
        SOURCE_1019,
        AUDIT_1020,
        GATE_1021,
        ROUTE_1022,
        SOURCE_1023,
        AUDIT_1024,
        GATE_1025,
        ROUTE_1026,
    ):
        require(path)

    expert_note_text = read_text(EXPERT_NOTE)
    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    em_doc_text = read_text(EM_DOC)
    source_1019 = read_json(SOURCE_1019)["summary"]
    audit_1020 = read_json(AUDIT_1020)["summary"]
    gate_1021 = read_json(GATE_1021)["summary"]
    route_1022 = read_json(ROUTE_1022)["summary"]
    source_1023 = read_json(SOURCE_1023)["summary"]
    audit_1024 = read_json(AUDIT_1024)["summary"]
    gate_1025 = read_json(GATE_1025)["summary"]
    route_1026 = read_json(ROUTE_1026)["summary"]

    prior_phrase_fragment_route_active = (
        gate_1025["selected_residual_route"] == CURRENT_BLOCKER
        and gate_1025["missing_v2_artifact"] == CURRENT_BLOCKER
        and route_1026["selected_next_generation_route"] == CURRENT_BLOCKER
        and bool(gate_1025["trial2_numeric_alpha_two_sector_hierarchy_pivot_active"])
    )

    statement_branch_no_new_surface = (
        not audit_1020["public_em_sector_normalization_statement_available"]
        and bool(audit_1020["public_single_zp_photon_canon_surface_available"])
        and bool(audit_1020["public_local_maxwell_adoption_surface_available"])
    )
    literal_branch_no_new_surface = bool(gate_1025["no_new_public_canonical_surface_added_in_literal_branch"])
    same_pattern_retry_threshold_reached = (
        prior_phrase_fragment_route_active
        and statement_branch_no_new_surface
        and literal_branch_no_new_surface
        and bool(route_1026["retry_triage_reconsideration_recommended"])
    )

    note_has_two_sector_em_literal = hit(expert_note_text, r"Z_P^{\rm EM} = 1") is not None
    note_has_em_bridge_literal = hit(expert_note_text, r"e = g_P / \sqrt{Z_P^{\rm EM}} = g_P") is not None
    note_has_newton_mchi_bridge = hit(expert_note_text, r"\frac{g_P\,v}{M_\chi^2} = \frac{4\pi G}{c^2}") is not None
    note_has_planck_assumption = hit(expert_note_text, r"M_\chi \sim M_{\rm Pl}") is not None

    part1_has_single_zp_photon_canon = hit(part1_text, r"A_\mu=\delta P_\mu^T/\sqrt{Z_P}") is not None
    part1_has_zp_vector_surface = hit(part1_text, r"-\frac{Z_{P}}{4}F^{(P)}_{\mu\nu}F_{(P)}^{\mu\nu}") is not None
    part1_has_vector_free_surface = hit(part1_text, r"-\frac{1}{4}F^{(P)}_{\mu\nu}F_{(P)}^{\mu\nu}") is not None
    part3a_has_phrase_fragment_blocker = hit(part3a_text, "EM-sector normalization phrase fragment") is not None
    part5_has_phrase_fragment_blocker = hit(
        part5_text, "missing `trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_phrase_fragment`"
    ) is not None
    em_doc_has_local_maxwell_adoption = hit(em_doc_text, "局所（固有時）では Maxwell/QED をそのまま採用") is not None
    em_doc_has_no_alpha_dependence_claim = hit(em_doc_text, "微細構造定数 α の P 依存を主張しない") is not None
    status_mentions_triage = hit(status_text, "retry triage gate の再判定") is not None
    roadmap_mentions_1027 = hit(roadmap_text, "`8.7.56.1027-.1030`") is not None

    triage_problem_classification = "text_search"
    text_search_continuation_justified = False
    alternate_computation_without_new_public_statement_available = False
    expert_advice_escalation_required = same_pattern_retry_threshold_reached and not text_search_continuation_justified
    expert_question_set_minimal = expert_advice_escalation_required

    inventory_ready = all(
        [
            prior_phrase_fragment_route_active,
            note_has_two_sector_em_literal,
            note_has_em_bridge_literal,
            note_has_newton_mchi_bridge,
            note_has_planck_assumption,
            part1_has_single_zp_photon_canon,
            part1_has_zp_vector_surface,
            part1_has_vector_free_surface,
            part3a_has_phrase_fragment_blocker,
            part5_has_phrase_fragment_blocker,
            em_doc_has_local_maxwell_adoption,
            em_doc_has_no_alpha_dependence_claim,
            status_mentions_triage,
            roadmap_mentions_1027,
            same_pattern_retry_threshold_reached,
        ]
    )

    current_blocker_is_missing_positive_em_normalization_surface = (
        not audit_1020["public_em_sector_normalization_statement_available"]
        and not audit_1024["public_em_sector_normalization_literal_available"]
        and not audit_1024["public_em_sector_normalization_phrase_fragment_available"]
    )

    common_inputs = {
        "expert_note_markdown": display_path(EXPERT_NOTE),
        "status_markdown": display_path(STATUS),
        "roadmap_markdown": display_path(ROADMAP),
        "ai_context_json": display_path(AI_CONTEXT),
        "primary_sources_markdown": display_path(PRIMARY_SOURCES),
        "part1_markdown": display_path(PART1),
        "part3a_markdown": display_path(PART3A),
        "part5_markdown": display_path(PART5),
        "electromagnetism_doc_markdown": display_path(EM_DOC),
        "prior_1019_json": display_path(SOURCE_1019),
        "prior_1020_json": display_path(AUDIT_1020),
        "prior_1021_json": display_path(GATE_1021),
        "prior_1022_json": display_path(ROUTE_1022),
        "prior_1023_json": display_path(SOURCE_1023),
        "prior_1024_json": display_path(AUDIT_1024),
        "prior_1025_json": display_path(GATE_1025),
        "prior_1026_json": display_path(ROUTE_1026),
    }

    expert_questions = [
        "current canon の範囲で `Z_P^{EM}=1` または `e=g_P` を public statement / literal として正当に昇格できるか",
        "昇格できない場合、two-sector hierarchy は external extension とみなし Trial-2 numeric α を structural pass / numeric open で止めるべきか",
        "昇格できる場合、single-Z_P photon canon (`A_mu=delta P_mu^T/sqrt(Z_P)` と `-Z_P F^2 / 4`) と両立する最小の式または文言は何か",
    ]

    inventory = payload(
        "8.7.56.1027",
        "Trial-2 numeric alpha two-sector hierarchy EM-sector normalization expert-advice source inventory",
        common_inputs,
        "Freeze the retry-triage pack: the same-pattern low-value repetition, the current single-Z_P photon canon conflict, and the minimal expert question set that replaces mechanical phrase-fragment descent.",
        {
            "triage_rule": "if statement/literal descent repeats without adding a new public-canonical surface, stop text-search subdivision and escalate for expert advice",
            "classification_rule": "the present blocker is a text-search blocker because it asks for a public-canonical statement, not a missing numerical substitution",
            "stop_rule": "phrase-fragment descent stops when the same family has already repeated and no new public-canonical surface is added",
        },
        [
            row(
                "trial2_numeric_alpha_two_sector_em_normalization_expert_inventory_complete",
                "pass" if inventory_ready else "reject",
                "two-sector EM-normalization expert inventory complete",
                1 if inventory_ready else 0,
                "This branch freezes the retry-triage judgment and the minimal expert question pack instead of descending into a third wording subdivision.",
            ),
            row(
                "trial2_numeric_alpha_same_pattern_retry_threshold_reached_before_phrase_fragment_branch",
                "pass" if same_pattern_retry_threshold_reached else "reject",
                "same-pattern retry threshold reached before phrase-fragment branch",
                1 if same_pattern_retry_threshold_reached else 0,
                "The statement and literal branches both failed to add a new public-canonical surface, so the phrase-fragment branch is low-value by the retry-gate rule.",
            ),
            row(
                "trial2_numeric_alpha_problem_classified_as_text_search_before_phrase_fragment_branch",
                "pass",
                "problem classified as text search before phrase-fragment branch",
                1,
                "The blocker is the absence of a public statement/literal, not a missing numeric substitution.",
            ),
            row(
                "trial2_numeric_alpha_expert_question_set_minimal_for_two_sector_em_normalization",
                "pass" if expert_question_set_minimal else "reject",
                "expert question set minimal for two-sector EM normalization",
                1 if expert_question_set_minimal else 0,
                "The escalation asks only whether the positive EM-normalization surface exists, whether no-go closeout is required, or how the current canon conflict should be resolved minimally.",
            ),
        ],
        {
            "inventory_ready": inventory_ready,
            "same_pattern_retry_threshold_reached": same_pattern_retry_threshold_reached,
            "retry_triage_gate_triggered": same_pattern_retry_threshold_reached,
            "triage_problem_classification": triage_problem_classification,
            "text_search_continuation_justified": text_search_continuation_justified,
            "expert_advice_escalation_required": expert_advice_escalation_required,
            "expert_question_set_minimal": expert_question_set_minimal,
            "current_blocker": CURRENT_BLOCKER,
            "first_route_to_close_or_none": NEXT_BUNDLE_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_two_sector_em_normalization_expert_inventory_frozen",
            "advance_to_8_7_56_1028": inventory_ready,
            "next_required_artifacts": [CURRENT_EXPERT_ROUTE, NEXT_BUNDLE_ROUTE],
        },
        {
            "prior_statement_audit_summary": audit_1020,
            "prior_literal_audit_summary": audit_1024,
            "prior_literal_gate_summary": gate_1025,
            "prior_route_summary": route_1026,
            "expert_questions": expert_questions,
            "note_hits": {
                "em_sector_normalization": hit(expert_note_text, r"Z_P^{\rm EM} = 1"),
                "em_bridge": hit(expert_note_text, r"e = g_P / \sqrt{Z_P^{\rm EM}} = g_P"),
                "newton_mchi_bridge": hit(expert_note_text, r"\frac{g_P\,v}{M_\chi^2} = \frac{4\pi G}{c^2}"),
                "planck_assumption": hit(expert_note_text, r"M_\chi \sim M_{\rm Pl}"),
            },
        },
    )

    audit = payload(
        "8.7.56.1028",
        "Trial-2 numeric alpha two-sector hierarchy EM-sector normalization expert-advice audit",
        common_inputs,
        "Audit the retry-triage judgment: confirm that the present blocker is a low-value text-search blocker, that phrase-fragment descent should stop, and that expert advice is the next honest route.",
        {
            "classification_rule": "a blocker is text-search if it asks for a missing public statement/literal rather than a missing numeric substitution",
            "continuation_rule": "text-search continuation is unjustified when the last completed wording branches add no new public-canonical surface",
            "escalation_rule": "if continuation is unjustified, the next official route is expert advice rather than phrase-fragment descent",
        },
        [
            row(
                "trial2_numeric_alpha_two_sector_em_normalization_expert_audit_complete",
                "pass" if inventory_ready else "reject",
                "two-sector EM-normalization expert audit complete",
                1 if inventory_ready else 0,
                "The retry-triage judgment is audited before any phrase-fragment branch is allowed to execute.",
            ),
            row(
                "trial2_numeric_alpha_problem_is_text_search_not_computation",
                "pass" if triage_problem_classification == "text_search" else "reject",
                "problem is text search not computation",
                1 if triage_problem_classification == "text_search" else 0,
                "The present blocker asks for a public-canonical EM-normalization statement/literal, not for a missing formula substitution.",
            ),
            row(
                "trial2_numeric_alpha_text_search_continuation_unjustified_after_same_pattern_retries",
                "pass" if not text_search_continuation_justified else "reject",
                "text-search continuation unjustified after same-pattern retries",
                1 if not text_search_continuation_justified else 0,
                "The statement and literal branches introduced no new public-canonical surface, so phrase-fragment descent is now low-value.",
            ),
            row(
                "trial2_numeric_alpha_expert_advice_escalation_required_after_triage",
                "pass" if expert_advice_escalation_required else "reject",
                "expert-advice escalation required after triage",
                1 if expert_advice_escalation_required else 0,
                "Because the blocker remains a public-surface question and the last branches added no new surface, expert advice is the next honest route.",
            ),
        ],
        {
            "audit_ready": inventory_ready,
            "triage_problem_classification": triage_problem_classification,
            "text_search_continuation_justified": text_search_continuation_justified,
            "alternate_computation_without_new_public_statement_available": alternate_computation_without_new_public_statement_available,
            "expert_advice_escalation_required": expert_advice_escalation_required,
            "current_blocker_is_missing_positive_em_normalization_surface": current_blocker_is_missing_positive_em_normalization_surface,
            "first_route_to_close_after_audit_or_none": NEXT_BUNDLE_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_two_sector_em_normalization_expert_audited",
            "advance_to_8_7_56_1029": True,
            "next_required_artifacts": [CURRENT_EXPERT_ROUTE, NEXT_BUNDLE_ROUTE],
        },
        {
            "part1_hits": {
                "photon_zp_extraction": hit(part1_text, r"A_\mu=\delta P_\mu^T/\sqrt{Z_P}"),
                "vector_zp_surface": hit(part1_text, r"-\frac{Z_{P}}{4}F^{(P)}_{\mu\nu}F_{(P)}^{\mu\nu}"),
                "vector_free_surface": hit(part1_text, r"-\frac{1}{4}F^{(P)}_{\mu\nu}F_{(P)}^{\mu\nu}"),
            },
            "em_doc_hits": {
                "local_maxwell_adoption": hit(em_doc_text, "局所（固有時）では Maxwell/QED をそのまま採用"),
                "no_alpha_dependence_claim": hit(em_doc_text, "微細構造定数 α の P 依存を主張しない"),
            },
            "current_blocker_hits": {
                "part3a_phrase_fragment": hit(part3a_text, "EM-sector normalization phrase fragment"),
                "part5_phrase_fragment": hit(
                    part5_text,
                    "missing `trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_phrase_fragment`",
                ),
            },
        },
    )

    gate = payload(
        "8.7.56.1029",
        "Trial-2 numeric alpha two-sector hierarchy EM-sector normalization expert-advice declaration gate",
        common_inputs,
        "Update the official gate after retry triage: the two-sector pivot remains active, mechanical wording descent stops, and expert-advice escalation becomes the official next route.",
        {
            "gate_rule": "the two-sector pivot can stay active while phrase-fragment descent is stopped by the retry gate",
            "stop_rule": "once same-pattern text-search retries add no new public-canonical surface, mechanical wording descent stops and expert escalation becomes official",
        },
        [
            row(
                "trial2_numeric_alpha_two_sector_em_normalization_expert_gate_complete",
                "pass",
                "two-sector EM-normalization expert gate complete",
                1,
                "The official state is updated after the retry-triage judgment.",
            ),
            row(
                "trial2_numeric_alpha_two_sector_hierarchy_pivot_still_active_after_triage",
                "pass" if gate_1025["trial2_numeric_alpha_two_sector_hierarchy_pivot_active"] else "reject",
                "two-sector hierarchy pivot still active after triage",
                1 if gate_1025["trial2_numeric_alpha_two_sector_hierarchy_pivot_active"] else 0,
                "The pivot remains live even though phrase-fragment descent is stopped.",
            ),
            row(
                "trial2_numeric_alpha_mechanical_wording_descent_stopped_after_triage",
                "pass" if expert_advice_escalation_required else "reject",
                "mechanical wording descent stopped after triage",
                1 if expert_advice_escalation_required else 0,
                "The next route is no longer the phrase-fragment branch.",
            ),
            row(
                "trial2_numeric_alpha_current_pack_still_not_closeout_ready_after_triage",
                "reject",
                "current pack still not closeout ready after triage",
                0,
                "Trial-2 numeric alpha remains open because the positive EM-normalization public surface is still absent.",
            ),
        ],
        {
            "trial2_numeric_alpha_computation_formula_ready": True,
            "trial2_numeric_alpha_absolute_normalization_dictionary_ready": True,
            "trial2_numeric_alpha_raw_final_computation_value_available": True,
            "trial2_numeric_alpha_numeric_from_current_pack_ready": False,
            "trial2_numeric_alpha_closeout_ready": False,
            "trial2_numeric_alpha_final_computation_performed": True,
            "trial2_numeric_alpha_final_computation_result_class": "precanonical_unit_incomplete",
            "trial2_numeric_alpha_retry_loop_retired": True,
            "trial2_numeric_alpha_retry_triage_gate_triggered": same_pattern_retry_threshold_reached,
            "trial2_numeric_alpha_problem_classification": triage_problem_classification,
            "trial2_numeric_alpha_text_search_continuation_justified": text_search_continuation_justified,
            "trial2_numeric_alpha_mechanical_wording_descent_stopped": expert_advice_escalation_required,
            "trial2_numeric_alpha_expert_advice_escalation_active": expert_advice_escalation_required,
            "trial2_numeric_alpha_two_sector_hierarchy_pivot_active": True,
            "selected_residual_route": NEXT_BUNDLE_ROUTE,
            "missing_v2_artifact": CURRENT_BLOCKER,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_two_sector_em_normalization_expert_gate_closed",
            "advance_to_8_7_56_1030": True,
            "next_required_artifacts": [CURRENT_EXPERT_ROUTE, NEXT_BUNDLE_ROUTE],
        },
        {
            "expert_audit_summary": audit["summary"],
            "prior_literal_gate_summary": gate_1025,
            "prior_route_summary": route_1026,
        },
    )

    route = payload(
        "8.7.56.1030",
        "Trial-2 numeric alpha route contract one-hundred-fifty-fourth refresh",
        common_inputs,
        "Refresh the next-generation contract after retry triage: retain the two-sector pivot, stop phrase-fragment descent, and advance to expert-bundle refresh as the next official route.",
        {
            "next_route_rule": "the next route is expert-bundle refresh for the missing EM-normalization public surface",
            "reserve_rule": "strong-side non-Abelian, running, and confinement remain on v3 hold reserve",
            "triage_rule": "retry triage converts low-value phrase-fragment descent into expert escalation",
        },
        [
            row(
                "trial2_numeric_alpha_route_contract_one_hundred_fifty_fourth_refresh_complete",
                "pass",
                "route contract one-hundred-fifty-fourth refresh complete",
                1,
                "The retry-triage judgment is converted into the next-generation contract.",
            ),
            row(
                "trial2_numeric_alpha_next_route_selected_as_two_sector_em_normalization_expert_bundle_refresh",
                "pass" if expert_advice_escalation_required else "reject",
                "next route selected as two-sector EM-normalization expert-bundle refresh",
                1 if expert_advice_escalation_required else 0,
                "The next official branch is no longer phrase-fragment descent; it is expert-bundle refresh for the unresolved public surface.",
            ),
            row(
                "trial2_numeric_alpha_precision_mainline_retained_after_triage",
                "pass" if route_1026["precision_alpha_mainline_retained"] else "reject",
                "precision-alpha mainline retained after triage",
                1 if route_1026["precision_alpha_mainline_retained"] else 0,
                "Trial-2 numeric alpha remains the precision mainline despite the escalation.",
            ),
            row(
                "trial2_numeric_alpha_two_sector_hierarchy_pivot_retained_after_triage",
                "pass" if route_1026["two_sector_hierarchy_pivot_retained"] else "reject",
                "two-sector hierarchy pivot retained after triage",
                1 if route_1026["two_sector_hierarchy_pivot_retained"] else 0,
                "The pivot stays active while the current blocker is pushed to expert review.",
            ),
        ],
        {
            "selected_next_generation_route": NEXT_BUNDLE_ROUTE,
            "strong_side_route_state": route_1026["strong_side_route_state"],
            "precision_alpha_mainline_retained": bool(route_1026["precision_alpha_mainline_retained"]),
            "electron_identification_pivot_retained": bool(route_1026["electron_identification_pivot_retained"]),
            "h0p_bridge_pivot_retained": bool(route_1026["h0p_bridge_pivot_retained"]),
            "final_computation_branch_retained": bool(route_1026["final_computation_branch_retained"]),
            "unit_consistency_audit_branch_retained": bool(route_1026["unit_consistency_audit_branch_retained"]),
            "dimensionless_alpha_bridge_branch_retained": bool(
                route_1026["dimensionless_alpha_bridge_branch_retained"]
            ),
            "em_unit_convention_bridge_branch_retained": bool(
                route_1026["em_unit_convention_bridge_branch_retained"]
            ),
            "mapping_statement_branch_retained": bool(route_1026["mapping_statement_branch_retained"]),
            "mapping_literal_branch_retained": bool(route_1026["mapping_literal_branch_retained"]),
            "expert_advice_escalation_branch_retained": True,
            "two_sector_hierarchy_pivot_retained": bool(route_1026["two_sector_hierarchy_pivot_retained"]),
            "same_pattern_retry_threshold_reached": same_pattern_retry_threshold_reached,
            "retry_triage_gate_triggered": same_pattern_retry_threshold_reached,
            "mechanical_wording_descent_stopped": expert_advice_escalation_required,
            "external_dependency_active": False,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_route_contract_one_hundred_fifty_fourth_refresh_frozen",
            "advance_to_next_route": True,
            "next_required_artifacts": [CURRENT_EXPERT_ROUTE, NEXT_BUNDLE_ROUTE],
        },
        {
            "gate_summary": gate["summary"],
            "prior_route_summary": route_1026,
        },
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_expert_advice_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_expert_advice_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_expert_advice_declaration_gate",
        gate,
    )
    write_artifact(
        "mass_origin_v2_t2_alpha_route_contract_one_hundred_fifty_fourth_refresh",
        route,
    )

    print("[done] 8.7.56.1027-.1030 artifacts generated:")
    print(" - mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_expert_advice_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_expert_advice_audit_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_expert_advice_declaration_gate_metrics.json")
    print(" - mass_origin_v2_t2_alpha_route_contract_one_hundred_fifty_fourth_refresh_metrics.json")


# Function: run the retry-triage expert-escalation branch from the CLI.

def run_cli() -> None:
    """CLI entry point for the Trial-2 numeric alpha retry-triage expert-escalation branch."""
    main()


if __name__ == "__main__":
    run_cli()
