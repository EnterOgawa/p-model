#!/usr/bin/env python3
"""Generate 8.7.56.1035-.1038 Trial-2 numeric alpha expert-response intake artifacts."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PUBLIC_OUT = ROOT / "output" / "public" / "quantum"

EXPERT_RESPONSE = Path(r"C:\Users\ogawa\Downloads\pmodel_v2_trial2_zp_em_equals_one.md")
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
PRIMARY_SOURCES = ROOT / "doc" / "PRIMARY_SOURCES.md"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
EM_DOC = ROOT / "doc" / "quantum" / "16_electromagnetism_charge_maxwell_photon.md"

SOURCE_1031 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "expert_bundle_refresh_source_inventory_metrics.json"
)
AUDIT_1032 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "expert_bundle_refresh_audit_metrics.json"
)
GATE_1033 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "expert_bundle_refresh_declaration_gate_metrics.json"
)
ROUTE_1034 = PUBLIC_OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_fifty_fifth_refresh_metrics.json"

CURRENT_RESPONSE_ROUTE = "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_expert_response_intake"
NEXT_RECONCILIATION_ROUTE = (
    "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_current_canon_reconciliation"
)
NEXT_RECONCILIATION_ARTIFACT = (
    "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_current_canon_reconciliation_note"
)
NEXT_ROUTE = "8.7.56.1039"


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


# Function: convert an AI-context path value into a Path object.

def as_path(path_text: str) -> Path:
    """Return an absolute Path for an AI-context path value."""
    raw = Path(path_text)
    if raw.is_absolute():
        return raw

    return ROOT / raw


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
        EXPERT_RESPONSE,
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        PRIMARY_SOURCES,
        PART1,
        PART3A,
        PART5,
        EM_DOC,
        SOURCE_1031,
        AUDIT_1032,
        GATE_1033,
        ROUTE_1034,
    ):
        require(path)

    response_text = read_text(EXPERT_RESPONSE)
    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    em_doc_text = read_text(EM_DOC)
    ai_context = read_json(AI_CONTEXT)
    source_1031 = read_json(SOURCE_1031)["summary"]
    audit_1032 = read_json(AUDIT_1032)["summary"]
    gate_1033 = read_json(GATE_1033)["summary"]
    route_1034 = read_json(ROUTE_1034)["summary"]

    latest_bundle_zip = as_path(ai_context["latest_expert_bundle"])
    latest_bundle_dir = as_path(ai_context["latest_expert_bundle_dir"])
    require(latest_bundle_zip)
    require(latest_bundle_dir)

    prior_response_wait_active = (
        gate_1033["selected_residual_route"] == CURRENT_RESPONSE_ROUTE
        and bool(gate_1033["trial2_numeric_alpha_expert_response_pending_external_input"])
        and route_1034["selected_next_generation_route"] == CURRENT_RESPONSE_ROUTE
        and bool(route_1034["external_dependency_active"])
    )

    response_has_yes_statement = hit(response_text, "Yes。Part I §2.7.0") is not None
    response_has_em_literal = hit(response_text, r"Z_P^{\rm EM} = 1") is not None
    response_has_gravity_literal = hit(response_text, r"Z_P^{\rm grav} = M_\chi^2/v^2") is not None
    response_has_bare_vector_rule = hit(response_text, r"-\frac{1}{4}F_{(P)}^{\mu\nu}") is not None
    response_has_scalar_rule = hit(response_text, r"\frac{M_\chi^2}{2}\partial_\mu\chi") is not None
    response_has_no_conflict_claim = hit(response_text, "conflict はない") is not None
    response_has_stale_vector_line_reference = hit(response_text, "Part I L1048-1052") is not None
    response_has_stale_scalar_line_reference = hit(response_text, "Part I L1040-1043") is not None

    part1_has_scalar_kinetic_surface = hit(part1_text, r"\frac{M_\chi^2}{2}\partial_\mu\chi") is not None
    part1_has_bare_vector_surface = hit(part1_text, r"-\frac{1}{4}F^{(P)}_{\mu\nu}F_{(P)}^{\mu\nu}") is not None
    part1_has_photon_zp_surface = hit(part1_text, r"A_\mu=\delta P_\mu^T/\sqrt{Z_P}") is not None
    part1_has_later_vector_zp_surface = hit(
        part1_text, r"-\frac{Z_{P}}{4}F^{(P)}_{\mu\nu}F_{(P)}^{\mu\nu}"
    ) is not None
    part1_has_current_newton_surface = hit(part1_text, r"g_P/Z_P=4\pi G") is not None

    part3a_has_structural_charge_rule = hit(part3a_text, r"e=g_P/\sqrt{Z_P}") is not None
    part3a_has_expert_response_wait_state = hit(part3a_text, "expert-response pending external input") is not None
    part5_has_structural_charge_rule = hit(part5_text, r"e=g_P/\sqrt{Z_P}") is not None
    part5_has_expert_response_wait_state = hit(part5_text, "expert-response pending external input") is not None
    em_doc_has_local_maxwell_adoption = hit(em_doc_text, "局所（固有時）では Maxwell/QED をそのまま採用") is not None
    em_doc_has_no_alpha_dependence_claim = hit(em_doc_text, "微細構造定数 α の P 依存を主張しない") is not None
    status_has_1035_next_step = hit(status_text, "8.7.56.1035") is not None
    roadmap_has_1035_branch = hit(roadmap_text, "`8.7.56.1035-.1038`") is not None

    positive_public_statement_candidate_detected = (
        response_has_yes_statement
        and response_has_em_literal
        and response_has_gravity_literal
        and part1_has_scalar_kinetic_surface
        and part1_has_bare_vector_surface
    )
    current_canon_conflict_present = (
        part1_has_photon_zp_surface
        and part1_has_later_vector_zp_surface
        and part3a_has_structural_charge_rule
        and part5_has_structural_charge_rule
        and response_has_no_conflict_claim
    )
    response_line_reference_stale_against_current_part1 = (
        response_has_stale_vector_line_reference and response_has_stale_scalar_line_reference
    )
    minimal_conflict_resolution_candidate = (
        positive_public_statement_candidate_detected and current_canon_conflict_present
    )
    selected_response_classification = (
        "minimal_conflict_resolution_candidate"
        if minimal_conflict_resolution_candidate
        else "positive_public_statement_candidate"
    )
    no_go_closeout_candidate = False
    new_public_canonical_surface_added_to_current_blocker_pack = positive_public_statement_candidate_detected

    inventory_ready = all(
        [
            prior_response_wait_active,
            response_has_yes_statement,
            response_has_em_literal,
            response_has_gravity_literal,
            response_has_bare_vector_rule,
            response_has_scalar_rule,
            part1_has_scalar_kinetic_surface,
            part1_has_bare_vector_surface,
            part1_has_photon_zp_surface,
            part1_has_later_vector_zp_surface,
            part1_has_current_newton_surface,
            part3a_has_structural_charge_rule,
            part3a_has_expert_response_wait_state,
            part5_has_structural_charge_rule,
            part5_has_expert_response_wait_state,
            em_doc_has_local_maxwell_adoption,
            em_doc_has_no_alpha_dependence_claim,
            status_has_1035_next_step,
            roadmap_has_1035_branch,
        ]
    )

    common_inputs = {
        "expert_response_markdown": display_path(EXPERT_RESPONSE),
        "status_markdown": display_path(STATUS),
        "roadmap_markdown": display_path(ROADMAP),
        "ai_context_json": display_path(AI_CONTEXT),
        "primary_sources_markdown": display_path(PRIMARY_SOURCES),
        "part1_markdown": display_path(PART1),
        "part3a_markdown": display_path(PART3A),
        "part5_markdown": display_path(PART5),
        "electromagnetism_doc_markdown": display_path(EM_DOC),
        "expert_bundle_dir": display_path(latest_bundle_dir),
        "expert_bundle_zip": display_path(latest_bundle_zip),
        "prior_1031_json": display_path(SOURCE_1031),
        "prior_1032_json": display_path(AUDIT_1032),
        "prior_1033_json": display_path(GATE_1033),
        "prior_1034_json": display_path(ROUTE_1034),
    }

    inventory = payload(
        "8.7.56.1035",
        "Trial-2 numeric alpha two-sector hierarchy EM-sector normalization expert-response intake source inventory",
        common_inputs,
        "Freeze the expert-response intake pack for the new note: the response itself, the Part I scalar and bare-vector kinetic surfaces it points to, the later single-Z_P photon canon that remains live, and the previously refreshed expert bundle.",
        {
            "inventory_rule": "the response-intake pack is ready when the response note, the current public canon surfaces it cites, and the current expert-share bundle are assembled together",
            "classification_rule": "the response may contain a positive public-statement candidate, but if the current canon still carries later conflicting single-Z_P photon surfaces it must be treated as a conflict-resolution candidate instead of an immediate closeout",
        },
        [
            row(
                "trial2_numeric_alpha_expert_response_intake_inventory_complete",
                "pass" if inventory_ready else "reject",
                "expert-response intake inventory complete",
                1 if inventory_ready else 0,
                "The response note, current canon surfaces, and current expert bundle are assembled into one intake pack.",
            ),
            row(
                "trial2_numeric_alpha_expert_response_available_now",
                "pass",
                "expert response available now",
                1,
                "The new note replaces the previous pending-external-input state for step 8.7.56.1035.",
            ),
            row(
                "trial2_numeric_alpha_positive_public_statement_candidate_detected_in_response",
                "pass" if positive_public_statement_candidate_detected else "reject",
                "positive public statement candidate detected in response",
                1 if positive_public_statement_candidate_detected else 0,
                "The response points to Part I 2.7.0 scalar and bare-vector kinetic terms as the proposed Z_P^EM = 1 / Z_P^grav = M_chi^2/v^2 surface.",
            ),
            row(
                "trial2_numeric_alpha_current_single_zp_photon_canon_still_present",
                "pass" if current_canon_conflict_present else "reject",
                "current single-Z_P photon canon still present",
                1 if current_canon_conflict_present else 0,
                "The later photon extraction and -Z_P F^2 / 4 surface remain live, so the response cannot be promoted blindly.",
            ),
        ],
        {
            "inventory_ready": inventory_ready,
            "expert_response_available": True,
            "expert_response_classification_candidate": selected_response_classification,
            "positive_public_statement_candidate_detected": positive_public_statement_candidate_detected,
            "current_canon_conflict_present": current_canon_conflict_present,
            "new_public_canonical_surface_added_to_current_blocker_pack": new_public_canonical_surface_added_to_current_blocker_pack,
            "first_route_to_close_or_none": NEXT_RECONCILIATION_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_expert_response_intake_inventory_frozen",
            "advance_to_8_7_56_1036": inventory_ready,
            "next_required_artifacts": [NEXT_RECONCILIATION_ROUTE],
        },
        {
            "response_hits": {
                "yes_statement": hit(response_text, "Yes。Part I §2.7.0"),
                "em_literal": hit(response_text, r"Z_P^{\rm EM} = 1"),
                "gravity_literal": hit(response_text, r"Z_P^{\rm grav} = M_\chi^2/v^2"),
                "no_conflict_claim": hit(response_text, "conflict はない"),
            },
            "part1_hits": {
                "scalar_kinetic": hit(part1_text, r"\frac{M_\chi^2}{2}\partial_\mu\chi"),
                "bare_vector_free": hit(part1_text, r"-\frac{1}{4}F^{(P)}_{\mu\nu}F_{(P)}^{\mu\nu}"),
                "photon_zp": hit(part1_text, r"A_\mu=\delta P_\mu^T/\sqrt{Z_P}"),
                "later_vector_zp": hit(part1_text, r"-\frac{Z_{P}}{4}F^{(P)}_{\mu\nu}F_{(P)}^{\mu\nu}"),
            },
        },
    )

    audit = payload(
        "8.7.56.1036",
        "Trial-2 numeric alpha two-sector hierarchy EM-sector normalization expert-response classification audit",
        common_inputs,
        "Classify the new response against the current canon: determine whether it is a clean positive public statement, a no-go closeout, or a minimal conflict-resolution candidate that surfaces the bare-vector kinetic route but still conflicts with the later photon-Z_P canon.",
        {
            "audit_rule": "if the response identifies a real public-canonical formula pair but the current canon still retains a later conflicting normalization surface, the honest classification is minimal conflict-resolution candidate",
            "retirement_rule": "once the response adds a new public-canonical surface to the blocker pack, the historical text-search blocker is retired and the next route becomes canon reconciliation rather than more wording descent",
        },
        [
            row(
                "trial2_numeric_alpha_expert_response_classification_audit_complete",
                "pass" if inventory_ready else "reject",
                "expert-response classification audit complete",
                1 if inventory_ready else 0,
                "The response is classified against the current canon rather than accepted or rejected by note text alone.",
            ),
            row(
                "trial2_numeric_alpha_response_classified_as_minimal_conflict_resolution_candidate",
                "pass" if minimal_conflict_resolution_candidate else "reject",
                "response classified as minimal conflict-resolution candidate",
                1 if minimal_conflict_resolution_candidate else 0,
                "The response surfaces Part I 2.7.0 as a real candidate but still conflicts with the later single-Z_P photon canon.",
            ),
            row(
                "trial2_numeric_alpha_response_not_classified_as_no_go_closeout",
                "pass" if not no_go_closeout_candidate else "reject",
                "response not classified as no-go closeout",
                1 if not no_go_closeout_candidate else 0,
                "The response found genuine public-canonical formulas, so the route does not close as a no-go.",
            ),
            row(
                "trial2_numeric_alpha_historical_text_search_blocker_retired",
                "pass" if new_public_canonical_surface_added_to_current_blocker_pack else "reject",
                "historical text-search blocker retired",
                1 if new_public_canonical_surface_added_to_current_blocker_pack else 0,
                "The response adds a new public-canonical surface to the blocker pack, so the old phrase/literal absence state is no longer the active blocker.",
            ),
        ],
        {
            "audit_ready": inventory_ready,
            "response_contains_positive_public_statement_candidate": positive_public_statement_candidate_detected,
            "response_contains_no_go_closeout_candidate": no_go_closeout_candidate,
            "response_contains_minimal_conflict_resolution_candidate": minimal_conflict_resolution_candidate,
            "selected_response_classification": selected_response_classification,
            "current_canon_conflict_present": current_canon_conflict_present,
            "response_line_reference_stale_against_current_part1": response_line_reference_stale_against_current_part1,
            "new_public_canonical_surface_added_to_current_blocker_pack": new_public_canonical_surface_added_to_current_blocker_pack,
            "historical_text_search_blocker_retired": new_public_canonical_surface_added_to_current_blocker_pack,
            "first_route_to_close_after_audit_or_none": NEXT_RECONCILIATION_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_expert_response_classified",
            "advance_to_8_7_56_1037": True,
            "next_required_artifacts": [NEXT_RECONCILIATION_ROUTE],
        },
        {
            "response_hits": {
                "stale_vector_line_reference": hit(response_text, "Part I L1048-1052"),
                "stale_scalar_line_reference": hit(response_text, "Part I L1040-1043"),
            },
            "part3a_hits": {
                "structural_charge_rule": hit(part3a_text, r"e=g_P/\sqrt{Z_P}"),
                "expert_response_wait_state": hit(part3a_text, "expert-response pending external input"),
            },
            "part5_hits": {
                "structural_charge_rule": hit(part5_text, r"e=g_P/\sqrt{Z_P}"),
                "expert_response_wait_state": hit(part5_text, "expert-response pending external input"),
            },
        },
    )

    gate = payload(
        "8.7.56.1037",
        "Trial-2 numeric alpha two-sector hierarchy EM-sector normalization expert-response declaration gate",
        common_inputs,
        "Update the official gate after the response classification: external-response pending is retired, the response is frozen as a minimal conflict-resolution candidate, and the residual blocker becomes current-canon reconciliation rather than EM statement absence.",
        {
            "gate_rule": "a minimal conflict-resolution candidate can retire the external wait without making the current pack closeout-ready",
            "residual_rule": "the residual blocker is now the precedence or translation between Part I 2.7.0 bare-vector normalization and the later photon-Z_P canon",
        },
        [
            row(
                "trial2_numeric_alpha_expert_response_gate_complete",
                "pass",
                "expert-response declaration gate complete",
                1,
                "The official state is updated after the response classification audit.",
            ),
            row(
                "trial2_numeric_alpha_expert_response_pending_external_input_retired",
                "pass",
                "expert response pending external input retired",
                1,
                "A real expert response artifact has now been ingested, so the external wait is over.",
            ),
            row(
                "trial2_numeric_alpha_selected_residual_route_is_current_canon_reconciliation",
                "pass" if minimal_conflict_resolution_candidate else "reject",
                "selected residual route is current-canon reconciliation",
                1 if minimal_conflict_resolution_candidate else 0,
                "The next blocker is the reconciliation between the bare-vector candidate and the later single-Z_P photon canon.",
            ),
            row(
                "trial2_numeric_alpha_closeout_still_not_ready_after_response_intake",
                "reject",
                "closeout still not ready after response intake",
                0,
                "The response advances the blocker pack but does not yet make numeric alpha closeout honest.",
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
            "trial2_numeric_alpha_problem_classification": "conflict_resolution",
            "trial2_numeric_alpha_text_search_continuation_justified": False,
            "trial2_numeric_alpha_mechanical_wording_descent_stopped": True,
            "trial2_numeric_alpha_expert_response_pending_external_input": False,
            "trial2_numeric_alpha_expert_response_intake_completed": True,
            "trial2_numeric_alpha_expert_response_classification": selected_response_classification,
            "trial2_numeric_alpha_positive_public_statement_candidate_present": positive_public_statement_candidate_detected,
            "trial2_numeric_alpha_no_go_closeout_candidate_selected": no_go_closeout_candidate,
            "trial2_numeric_alpha_minimal_conflict_resolution_candidate_selected": minimal_conflict_resolution_candidate,
            "trial2_numeric_alpha_new_public_canonical_surface_added_to_current_blocker_pack": new_public_canonical_surface_added_to_current_blocker_pack,
            "trial2_numeric_alpha_historical_text_search_blocker_retired": new_public_canonical_surface_added_to_current_blocker_pack,
            "trial2_numeric_alpha_two_sector_hierarchy_pivot_active": True,
            "selected_residual_route": NEXT_RECONCILIATION_ROUTE,
            "missing_v2_artifact": NEXT_RECONCILIATION_ARTIFACT,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_expert_response_gate_closed",
            "advance_to_8_7_56_1038": True,
            "next_required_artifacts": [NEXT_RECONCILIATION_ROUTE],
        },
        {
            "audit_summary": audit["summary"],
            "prior_wait_gate_summary": gate_1033,
        },
    )

    route = payload(
        "8.7.56.1038",
        "Trial-2 numeric alpha route contract one-hundred-fifty-sixth refresh",
        common_inputs,
        "Refresh the next-generation contract after expert-response intake: retain the precision-alpha mainline, retire the external dependency, and advance to current-canon reconciliation as the next official route.",
        {
            "next_route_rule": "the next route is current-canon reconciliation between the Part I 2.7.0 bare-vector candidate and the later photon-Z_P canon",
            "reserve_rule": "strong-side non-Abelian, running, and confinement remain on v3 hold reserve",
        },
        [
            row(
                "trial2_numeric_alpha_route_contract_one_hundred_fifty_sixth_refresh_complete",
                "pass",
                "route contract one-hundred-fifty-sixth refresh complete",
                1,
                "The expert-response intake gate is converted into the new next-generation contract.",
            ),
            row(
                "trial2_numeric_alpha_next_route_selected_as_current_canon_reconciliation",
                "pass" if minimal_conflict_resolution_candidate else "reject",
                "next route selected as current-canon reconciliation",
                1 if minimal_conflict_resolution_candidate else 0,
                "The next official branch is reconciliation, not more waiting and not a raw closeout jump.",
            ),
            row(
                "trial2_numeric_alpha_external_dependency_retired",
                "pass",
                "external dependency retired",
                1,
                "A response artifact is now in hand, so the mainline is no longer blocked on outside input.",
            ),
            row(
                "trial2_numeric_alpha_precision_mainline_retained_after_response_intake",
                "pass" if bool(route_1034.get("precision_alpha_mainline_retained", False)) else "reject",
                "precision-alpha mainline retained after response intake",
                1 if bool(route_1034.get("precision_alpha_mainline_retained", False)) else 0,
                "Trial-2 numeric alpha remains the precision mainline after the new intake.",
            ),
        ],
        {
            "selected_next_generation_route": NEXT_RECONCILIATION_ROUTE,
            "strong_side_route_state": route_1034.get("strong_side_route_state"),
            "precision_alpha_mainline_retained": bool(route_1034.get("precision_alpha_mainline_retained", False)),
            "electron_identification_pivot_retained": bool(route_1034.get("electron_identification_pivot_retained", False)),
            "h0p_bridge_pivot_retained": bool(route_1034.get("h0p_bridge_pivot_retained", False)),
            "final_computation_branch_retained": bool(route_1034.get("final_computation_branch_retained", False)),
            "unit_consistency_audit_branch_retained": bool(
                route_1034.get("unit_consistency_audit_branch_retained", False)
            ),
            "dimensionless_alpha_bridge_branch_retained": bool(
                route_1034.get("dimensionless_alpha_bridge_branch_retained", False)
            ),
            "em_unit_convention_bridge_branch_retained": bool(
                route_1034.get("em_unit_convention_bridge_branch_retained", False)
            ),
            "mapping_statement_branch_retained": bool(route_1034.get("mapping_statement_branch_retained", False)),
            "mapping_literal_branch_retained": bool(route_1034.get("mapping_literal_branch_retained", False)),
            "expert_advice_escalation_branch_retained": True,
            "two_sector_hierarchy_pivot_retained": True,
            "expert_response_intake_branch_completed": True,
            "external_dependency_active": False,
            "historical_text_search_blocker_retired": new_public_canonical_surface_added_to_current_blocker_pack,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_route_contract_one_hundred_fifty_sixth_refresh_frozen",
            "advance_to_next_route": True,
            "next_required_artifacts": [NEXT_RECONCILIATION_ROUTE],
        },
        {
            "gate_summary": gate["summary"],
            "prior_route_summary": route_1034,
        },
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_expert_response_intake_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_expert_response_intake_classification_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_expert_response_intake_declaration_gate",
        gate,
    )
    write_artifact(
        "mass_origin_v2_t2_alpha_route_contract_one_hundred_fifty_sixth_refresh",
        route,
    )

    print("[done] 8.7.56.1035-.1038 artifacts generated:")
    print(" - mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_expert_response_intake_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_expert_response_intake_classification_audit_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_expert_response_intake_declaration_gate_metrics.json")
    print(" - mass_origin_v2_t2_alpha_route_contract_one_hundred_fifty_sixth_refresh_metrics.json")


# Function: run the expert-response intake branch from the CLI.

def run_cli() -> None:
    """CLI entry point for the Trial-2 numeric alpha expert-response intake branch."""
    main()


if __name__ == "__main__":
    run_cli()
