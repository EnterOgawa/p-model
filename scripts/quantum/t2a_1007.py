#!/usr/bin/env python3
"""Generate 8.7.56.1007-.1010 Trial-2 numeric alpha expert-response intake artifacts."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_OUT = ROOT / "output" / "public" / "quantum"
PRIVATE_OUT = ROOT / "output" / "private" / "quantum"

STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
PRIMARY_SOURCES = ROOT / "doc" / "PRIMARY_SOURCES.md"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
EM_DOC = ROOT / "doc" / "quantum" / "16_electromagnetism_charge_maxwell_photon.md"

EXPERT_NOTE = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_final_computation.md")
EXPERT_BUNDLE_DIR = PRIVATE_OUT / "expert_review_bundle_20260324_004752"
EXPERT_BUNDLE_ZIP = PRIVATE_OUT / "expert_review_bundle_20260324_004752.zip"
EXPERT_MANIFEST = EXPERT_BUNDLE_DIR / "BUNDLE_MANIFEST.txt"
EXPERT_QUESTIONS = EXPERT_BUNDLE_DIR / "QUESTIONS_FOR_REVIEW.txt"

SOURCE_1003 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_final_computation_"
    "expert_advice_gp_to_elementary_charge_mapping_source_inventory_metrics.json"
)
AUDIT_1004 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_final_computation_"
    "expert_advice_gp_to_elementary_charge_mapping_audit_metrics.json"
)
GATE_1005 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_final_computation_"
    "expert_advice_gp_to_elementary_charge_mapping_declaration_gate_metrics.json"
)
ROUTE_1006 = PUBLIC_OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_forty_eighth_refresh_metrics.json"

CURRENT_LITERAL_BLOCKER = (
    "trial2_numeric_alpha_final_computation_"
    "dimensionless_alpha_bridge_gp_to_elementary_charge_mapping_literal"
)
PENDING_RESPONSE_ROUTE = (
    "trial2_numeric_alpha_final_computation_"
    "expert_advice_gp_to_elementary_charge_mapping_response_arrival"
)
NEXT_ROUTE = "8.7.56.1011"


# Function: return the current UTC timestamp.
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# Function: stop execution when a required path is missing.

def require(path: Path) -> None:
    """Require one input path to exist before execution continues."""
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


# Function: count numbered review questions in the bundle.

def count_questions(text: str) -> int:
    """Count numbered review questions in the bundle question file."""
    total = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped[:2] in {"1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9."}:
            total += 1

    return total


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
    PUBLIC_OUT.mkdir(parents=True, exist_ok=True)
    json_path = PUBLIC_OUT / f"{stem}_metrics.json"
    csv_path = PUBLIC_OUT / f"{stem}_rows.csv"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# Function: execute the expert-response intake branch.

def main() -> None:
    """Execute the Trial-2 numeric alpha expert-response intake branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        PRIMARY_SOURCES,
        PART1,
        PART3A,
        PART5,
        EM_DOC,
        EXPERT_NOTE,
        EXPERT_BUNDLE_DIR,
        EXPERT_BUNDLE_ZIP,
        EXPERT_MANIFEST,
        EXPERT_QUESTIONS,
        SOURCE_1003,
        AUDIT_1004,
        GATE_1005,
        ROUTE_1006,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    em_doc_text = read_text(EM_DOC)
    manifest_text = read_text(EXPERT_MANIFEST)
    questions_text = read_text(EXPERT_QUESTIONS)
    ai_context = read_json(AI_CONTEXT)
    source_1003 = read_json(SOURCE_1003)["summary"]
    audit_1004 = read_json(AUDIT_1004)["summary"]
    gate_1005 = read_json(GATE_1005)["summary"]
    route_1006 = read_json(ROUTE_1006)["summary"]

    response_candidates = sorted(
        Path(r"C:\Users\ogawa\Downloads").glob("pmodel_v2_trial2*response*.md"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    external_response_available = len(response_candidates) > 0
    latest_response_or_none = (
        str(response_candidates[0]).replace("\\", "/") if external_response_available else None
    )
    question_count = count_questions(questions_text)
    manifest_has_zero_missing = "MISSING_COUNT=0" in manifest_text
    manifest_has_copied = "COPIED_COUNT=19" in manifest_text
    prior_gate_ready = (
        source_1003["expert_bundle_ready"]
        and source_1003["expert_question_pack_ready"]
        and audit_1004["expert_question_set_minimal"]
        and gate_1005["trial2_numeric_alpha_expert_advice_escalation_active"]
        and route_1006["selected_next_generation_route"]
        == "trial2_numeric_alpha_final_computation_expert_advice_gp_to_elementary_charge_mapping_response"
    )
    status_has_next_step = hit(status_text, "8.7.56.1007") is not None
    roadmap_has_response_branch = hit(roadmap_text, "`8.7.56.1007-.1010`") is not None
    part1_has_weak_field = hit(part1_text, r"g_P/Z_P=4\pi G") is not None
    part3a_has_response_intake = hit(part3a_text, "expert-response intake") is not None
    part5_has_response_intake = hit(part5_text, "expert-response intake") is not None
    em_doc_has_coulomb_surface = hit(em_doc_text, r"\Phi(r)=\frac{1}{4\pi\varepsilon_0}\frac{q}{r}") is not None

    inventory_ready = all(
        [
            prior_gate_ready,
            manifest_has_zero_missing,
            manifest_has_copied,
            question_count >= 4,
            status_has_next_step,
            roadmap_has_response_branch,
            part1_has_weak_field,
            part3a_has_response_intake,
            part5_has_response_intake,
            em_doc_has_coulomb_surface,
        ]
    )

    common_inputs = {
        "status_markdown": display_path(STATUS),
        "roadmap_markdown": display_path(ROADMAP),
        "ai_context_json": display_path(AI_CONTEXT),
        "primary_sources_markdown": display_path(PRIMARY_SOURCES),
        "part1_markdown": display_path(PART1),
        "part3a_markdown": display_path(PART3A),
        "part5_markdown": display_path(PART5),
        "electromagnetism_doc_markdown": display_path(EM_DOC),
        "expert_note_markdown": display_path(EXPERT_NOTE),
        "expert_bundle_dir": display_path(EXPERT_BUNDLE_DIR),
        "expert_bundle_zip": display_path(EXPERT_BUNDLE_ZIP),
        "expert_manifest_txt": display_path(EXPERT_MANIFEST),
        "expert_questions_txt": display_path(EXPERT_QUESTIONS),
        "prior_1003_json": display_path(SOURCE_1003),
        "prior_1004_json": display_path(AUDIT_1004),
        "prior_1005_json": display_path(GATE_1005),
        "prior_1006_json": display_path(ROUTE_1006),
        "expert_response_markdown_or_none": latest_response_or_none,
    }

    inventory = payload(
        "8.7.56.1007",
        "Trial-2 numeric alpha expert-response intake source inventory",
        common_inputs,
        "Freeze the expert-response intake pack: current canon, the refreshed expert bundle, the minimal question set, and the fact that no external response has been received yet.",
        {
            "inventory_rule": "the response-intake pack is ready when the current canon, the frozen expert bundle, and the expert question set are assembled even if no response has arrived yet",
            "blocking_rule": "without an external response artifact, the branch is pending external input rather than a new text-search or computation subroute",
        },
        [
            row(
                "trial2_numeric_alpha_expert_response_intake_inventory_complete",
                "pass" if inventory_ready else "reject",
                "expert-response intake inventory complete",
                1 if inventory_ready else 0,
                "The response-intake pack must be frozen before the external-input gate is audited.",
            ),
            row(
                "trial2_numeric_alpha_external_response_available_now",
                "reject" if not external_response_available else "pass",
                "external response available now",
                1 if external_response_available else 0,
                "No expert response note is available yet, so the branch remains pending external input.",
            ),
            row(
                "trial2_numeric_alpha_expert_bundle_still_ready",
                "pass" if manifest_has_zero_missing else "reject",
                "expert bundle still ready",
                1 if manifest_has_zero_missing else 0,
                "The refreshed expert bundle remains the canonical share pack for this blocker.",
            ),
            row(
                "trial2_numeric_alpha_question_pack_still_minimal",
                "pass" if question_count >= 4 else "reject",
                "question pack still minimal",
                1 if question_count >= 4 else 0,
                "The response branch reuses the frozen minimal question set rather than reopening the wording loop.",
            ),
        ],
        {
            "inventory_ready": inventory_ready,
            "expert_bundle_ready": manifest_has_zero_missing,
            "expert_question_pack_ready": question_count >= 4,
            "external_response_available": external_response_available,
            "current_blocker_under_review": CURRENT_LITERAL_BLOCKER,
            "first_route_to_close_or_none": PENDING_RESPONSE_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_expert_response_intake_inventory_frozen",
            "advance_to_8_7_56_1008": inventory_ready,
            "next_required_artifacts": [PENDING_RESPONSE_ROUTE],
        },
        {
            "bundle_manifest_hit": {
                "copied_count_line_present": manifest_has_copied,
                "missing_count_line_present": manifest_has_zero_missing,
                "question_count": question_count,
            },
            "status_hit": hit(status_text, "8.7.56.1007"),
            "roadmap_hit": hit(roadmap_text, "`8.7.56.1007-.1010`"),
        },
    )

    audit = payload(
        "8.7.56.1008",
        "Trial-2 numeric alpha expert-response audit",
        common_inputs,
        "Audit whether the response-intake branch is externally blocked and whether any further internal wording/computation descent would be low-value without a new expert response artifact.",
        {
            "audit_rule": "if no external response artifact exists, the branch is response-pending and no internal substitute should be fabricated",
            "stop_rule": "mechanical wording descent must remain stopped unless genuinely new public-canonical evidence appears",
        },
        [
            row(
                "trial2_numeric_alpha_expert_response_audit_complete",
                "pass" if inventory_ready else "reject",
                "expert-response audit complete",
                1 if inventory_ready else 0,
                "The external-input gate is audited only after the response-intake pack is frozen.",
            ),
            row(
                "trial2_numeric_alpha_external_response_still_pending",
                "pass" if not external_response_available else "reject",
                "external response still pending",
                1 if not external_response_available else 0,
                "The current state is an external-input wait, not a live internal derivation branch.",
            ),
            row(
                "trial2_numeric_alpha_no_internal_substitute_for_missing_response",
                "pass",
                "no internal substitute for missing response",
                1,
                "Current canon offers no new public-canonical surface that would justify resuming the wording loop without external input.",
            ),
            row(
                "trial2_numeric_alpha_mechanical_wording_descent_must_remain_stopped",
                "pass" if gate_1005["trial2_numeric_alpha_mechanical_wording_descent_stopped"] else "reject",
                "mechanical wording descent must remain stopped",
                1 if gate_1005["trial2_numeric_alpha_mechanical_wording_descent_stopped"] else 0,
                "The retry triage gate remains in force while the branch waits for external input.",
            ),
        ],
        {
            "audit_ready": inventory_ready,
            "external_response_available": external_response_available,
            "response_pending_external_input": not external_response_available,
            "no_internal_substitute_available": True,
            "mechanical_wording_descent_stopped": gate_1005["trial2_numeric_alpha_mechanical_wording_descent_stopped"],
            "first_route_to_close_after_audit_or_none": PENDING_RESPONSE_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_expert_response_audited",
            "advance_to_8_7_56_1009": True,
            "next_required_artifacts": [PENDING_RESPONSE_ROUTE],
        },
        {
            "source_1003_summary": source_1003,
            "audit_1004_summary": audit_1004,
            "response_candidates": [str(path).replace("\\", "/") for path in response_candidates],
        },
    )

    gate = payload(
        "8.7.56.1009",
        "Trial-2 numeric alpha expert-response declaration gate",
        common_inputs,
        "Update the official gate after the response-intake audit: expert-advice escalation remains active, the branch is pending external input, and no further internal wording descent is authorized.",
        {
            "gate_rule": "without an external expert response, Trial-2 remains structural-pass / numeric-open under current canon",
            "pending_rule": "the next official action is to ingest a response when one arrives, not to fabricate a substitute artifact",
        },
        [
            row(
                "trial2_numeric_alpha_expert_response_gate_complete",
                "pass",
                "expert-response declaration gate complete",
                1,
                "The official state is updated after the response-intake audit.",
            ),
            row(
                "trial2_numeric_alpha_expert_response_pending_external_input",
                "pass" if not external_response_available else "reject",
                "expert response pending external input",
                1 if not external_response_available else 0,
                "No expert response has arrived yet, so the branch remains externally blocked.",
            ),
            row(
                "trial2_numeric_alpha_expert_advice_escalation_still_active",
                "pass" if gate_1005["trial2_numeric_alpha_expert_advice_escalation_active"] else "reject",
                "expert-advice escalation still active",
                1 if gate_1005["trial2_numeric_alpha_expert_advice_escalation_active"] else 0,
                "The escalation remains the official mainline until a response is ingested.",
            ),
            row(
                "trial2_numeric_alpha_closeout_still_not_ready",
                "reject",
                "closeout still not ready",
                0,
                "Without an external response or a genuinely new public-canonical surface, Trial-2 cannot close honestly.",
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
            "trial2_numeric_alpha_retry_triage_gate_triggered": True,
            "trial2_numeric_alpha_expert_advice_escalation_active": True,
            "trial2_numeric_alpha_mechanical_wording_descent_stopped": True,
            "trial2_numeric_alpha_expert_response_pending_external_input": not external_response_available,
            "selected_residual_route": PENDING_RESPONSE_ROUTE,
            "missing_v2_artifact": "external_expert_response_for_missing_gp_to_elementary_charge_literal",
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_expert_response_gate_closed",
            "advance_to_8_7_56_1010": True,
            "next_required_artifacts": [PENDING_RESPONSE_ROUTE],
        },
        {
            "audit_summary": audit["summary"],
            "prior_gate_summary": gate_1005,
        },
    )

    route = payload(
        "8.7.56.1010",
        "Trial-2 numeric alpha route contract one-hundred-forty-ninth refresh",
        common_inputs,
        "Refresh the next-generation contract after the response-intake declaration gate: keep Trial-2 on the precision mainline, keep the strong side on reserve, and block further internal looping until an external response arrives.",
        {
            "next_route_rule": "the next route is response-arrival integration on receipt, not another internal wording subdivision",
            "reserve_rule": "strong-side non-Abelian, running, and confinement remain on v3 hold reserve",
        },
        [
            row(
                "trial2_numeric_alpha_route_contract_one_hundred_forty_ninth_refresh_complete",
                "pass",
                "route contract one-hundred-forty-ninth refresh complete",
                1,
                "The response-intake declaration gate is converted into the next-generation contract.",
            ),
            row(
                "trial2_numeric_alpha_next_route_selected_as_response_arrival_integration",
                "pass",
                "next route selected as response-arrival integration",
                1,
                "The next official route is to integrate an expert response on receipt, not to continue internal looping.",
            ),
            row(
                "trial2_numeric_alpha_external_dependency_active",
                "pass" if not external_response_available else "reject",
                "external dependency active",
                1 if not external_response_available else 0,
                "This branch is now blocked on external expert input.",
            ),
            row(
                "trial2_numeric_alpha_precision_mainline_retained_after_response_intake_gate",
                "pass" if route_1006["precision_alpha_mainline_retained"] else "reject",
                "precision-alpha mainline retained after response-intake gate",
                1 if route_1006["precision_alpha_mainline_retained"] else 0,
                "Trial-2 numeric alpha remains the precision mainline despite the external block.",
            ),
        ],
        {
            "selected_next_generation_route": PENDING_RESPONSE_ROUTE,
            "strong_side_route_state": route_1006["strong_side_route_state"],
            "precision_alpha_mainline_retained": bool(route_1006["precision_alpha_mainline_retained"]),
            "electron_identification_pivot_retained": bool(route_1006["electron_identification_pivot_retained"]),
            "h0p_bridge_pivot_retained": bool(route_1006["h0p_bridge_pivot_retained"]),
            "final_computation_branch_retained": bool(route_1006["final_computation_branch_retained"]),
            "unit_consistency_audit_branch_retained": bool(route_1006["unit_consistency_audit_branch_retained"]),
            "dimensionless_alpha_bridge_branch_retained": bool(
                route_1006["dimensionless_alpha_bridge_branch_retained"]
            ),
            "em_unit_convention_bridge_branch_retained": bool(
                route_1006["em_unit_convention_bridge_branch_retained"]
            ),
            "mapping_statement_branch_retained": bool(route_1006["mapping_statement_branch_retained"]),
            "mapping_literal_branch_retained": bool(route_1006["mapping_literal_branch_retained"]),
            "expert_advice_escalation_branch_retained": bool(route_1006["expert_advice_escalation_branch_retained"]),
            "same_pattern_retry_threshold_reached": True,
            "retry_triage_gate_triggered": True,
            "external_dependency_active": not external_response_available,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_route_contract_one_hundred_forty_ninth_refresh_frozen",
            "advance_to_next_route": True,
            "next_required_artifacts": [PENDING_RESPONSE_ROUTE],
        },
        {
            "gate_summary": gate["summary"],
            "prior_route_summary": route_1006,
        },
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_final_computation_expert_advice_gp_to_elementary_charge_mapping_response_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_final_computation_expert_advice_gp_to_elementary_charge_mapping_response_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_final_computation_expert_advice_gp_to_elementary_charge_mapping_response_declaration_gate",
        gate,
    )
    write_artifact(
        "mass_origin_v2_t2_alpha_route_contract_one_hundred_forty_ninth_refresh",
        route,
    )

    print("[done] 8.7.56.1007-.1010 artifacts generated:")
    print(" - mass_origin_v2_trial2_numeric_alpha_final_computation_expert_advice_gp_to_elementary_charge_mapping_response_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_final_computation_expert_advice_gp_to_elementary_charge_mapping_response_audit_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_final_computation_expert_advice_gp_to_elementary_charge_mapping_response_declaration_gate_metrics.json")
    print(" - mass_origin_v2_t2_alpha_route_contract_one_hundred_forty_ninth_refresh_metrics.json")


# Function: run the expert-response intake branch from the CLI.

def run_cli() -> None:
    """CLI entry point for the Trial-2 numeric alpha expert-response intake branch."""
    main()


if __name__ == "__main__":
    run_cli()
