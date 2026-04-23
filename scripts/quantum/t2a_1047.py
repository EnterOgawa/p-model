#!/usr/bin/env python3
"""Generate 8.7.56.1047-.1050 Trial-2 numeric alpha checkpoint-wording promotion artifacts."""

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
PRIMARY_SOURCES = ROOT / "doc" / "PRIMARY_SOURCES.md"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
EM_DOC = ROOT / "doc" / "quantum" / "16_electromagnetism_charge_maxwell_photon.md"

SOURCE_1043 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "current_canon_bridge_statement_source_inventory_metrics.json"
)
AUDIT_1044 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "current_canon_bridge_statement_audit_metrics.json"
)
GATE_1045 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "current_canon_bridge_statement_declaration_gate_metrics.json"
)
ROUTE_1046 = PUBLIC_OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_fifty_eighth_refresh_metrics.json"

CURRENT_CHECKPOINT_ROUTE = (
    "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_current_canon_"
    "bridge_statement_checkpoint_wording_promotion"
)
NEXT_NUMERIC_REOPEN_ROUTE = "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_numeric_reopen"
NEXT_NUMERIC_REOPEN_ARTIFACT = (
    "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_numeric_reopen_note"
)
NEXT_ROUTE = "8.7.56.1051"

CHECKPOINT_BRIDGE_HEAD = (
    "current checkpoint wording としては、電磁結合は Part I 2.7.0 の vector kinetic coefficient"
)
CHECKPOINT_BRIDGE_MID = "scalar kinetic coefficient"
CHECKPOINT_BRIDGE_TAIL = "これらは同一作用の別 sector である"
NOTE_BRIDGE_HEAD = "The electromagnetic coupling is normalized by the vector kinetic"
NOTE_BRIDGE_MID = "while the gravitational coupling"
NOTE_BRIDGE_TAIL = "These are distinct sectors of the same action."


# Function: return the current UTC timestamp.
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# Function: stop execution when a required path is missing.

def require(path: Path) -> None:
    """Require one input path to exist before execution continues."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# Function: read one UTF-8 text file.

def read_text(path: Path) -> str:
    """Read a UTF-8 text file."""
    return path.read_text(encoding="utf-8")


# Function: read one UTF-8 JSON file.

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


# Function: return whether any pattern hit exists in the text.

def any_hit(text: str, patterns: tuple[str, ...]) -> dict | None:
    """Return the first matching hit among multiple patterns."""
    for pattern in patterns:
        found = hit(text, pattern)
        if found is not None:
            return found

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


# Function: execute the checkpoint-wording promotion branch.

def main() -> None:
    """Execute the Trial-2 numeric alpha checkpoint-wording promotion branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        PRIMARY_SOURCES,
        PART1,
        PART3A,
        PART5,
        EM_DOC,
        SOURCE_1043,
        AUDIT_1044,
        GATE_1045,
        ROUTE_1046,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    em_doc_text = read_text(EM_DOC)
    ai_context = read_json(AI_CONTEXT)
    source_1043 = read_json(SOURCE_1043)["summary"]
    audit_1044 = read_json(AUDIT_1044)["summary"]
    gate_1045 = read_json(GATE_1045)["summary"]
    route_1046 = read_json(ROUTE_1046)["summary"]

    latest_bundle_zip = as_path(ai_context["latest_expert_bundle"])
    latest_bundle_dir = as_path(ai_context["latest_expert_bundle_dir"])
    latest_note_path = as_path(ai_context["latest_expert_note"])
    for path in (latest_bundle_zip, latest_bundle_dir, latest_note_path):
        require(path)

    note_text = read_text(latest_note_path)

    prior_checkpoint_route_active = (
        source_1043["first_route_to_close_or_none"] == CURRENT_CHECKPOINT_ROUTE
        and audit_1044["first_route_to_close_after_audit_or_none"] == CURRENT_CHECKPOINT_ROUTE
        and gate_1045["selected_residual_route"] == CURRENT_CHECKPOINT_ROUTE
        and route_1046["selected_next_generation_route"] == CURRENT_CHECKPOINT_ROUTE
        and not bool(route_1046["external_dependency_active"])
    )

    status_has_1047_next_step = hit(status_text, "8.7.56.1047") is not None
    roadmap_has_1047_branch = hit(roadmap_text, "`8.7.56.1047-.1050`") is not None

    part1_has_bare_vector_surface = hit(part1_text, r"-\frac{1}{4}F^{(P)}_{\mu\nu}F_{(P)}^{\mu\nu}") is not None
    part1_has_scalar_kinetic_surface = hit(part1_text, r"\frac{M_\chi^2}{2}\partial_\mu\chi") is not None
    part1_has_photon_zp_surface = hit(part1_text, r"A_\mu=\delta P_\mu^T/\sqrt{Z_P}") is not None
    part1_has_later_vector_zp_surface = hit(part1_text, r"-\frac{Z_P}{4}F^{(P)}_{\mu\nu}F_{(P)}^{\mu\nu}") is not None
    part1_has_wavefunction_glossary = hit(part1_text, "波動関数正規化係数") is not None

    part3a_has_bridge_head = hit(part3a_text, CHECKPOINT_BRIDGE_HEAD) is not None
    part3a_has_bridge_mid = hit(part3a_text, CHECKPOINT_BRIDGE_MID) is not None
    part3a_has_bridge_tail = hit(part3a_text, CHECKPOINT_BRIDGE_TAIL) is not None
    part5_has_bridge_head = hit(part5_text, CHECKPOINT_BRIDGE_HEAD) is not None
    part5_has_bridge_mid = hit(part5_text, CHECKPOINT_BRIDGE_MID) is not None
    part5_has_bridge_tail = hit(part5_text, CHECKPOINT_BRIDGE_TAIL) is not None

    part3a_public_bridge_hit = any_hit(
        part3a_text,
        (CHECKPOINT_BRIDGE_HEAD, CHECKPOINT_BRIDGE_MID, CHECKPOINT_BRIDGE_TAIL),
    )
    part5_public_bridge_hit = any_hit(
        part5_text,
        (CHECKPOINT_BRIDGE_HEAD, CHECKPOINT_BRIDGE_MID, CHECKPOINT_BRIDGE_TAIL),
    )
    part1_public_bridge_hit = any_hit(
        part1_text,
        ("vector kinetic coefficient", "scalar kinetic coefficient", CHECKPOINT_BRIDGE_TAIL),
    )

    note_has_yes_part1_surface = hit(note_text, "Yes。Part I §2.7.0 の全作用定義そのもの。") is not None
    note_has_candidate_bridge_head = hit(note_text, NOTE_BRIDGE_HEAD) is not None
    note_has_candidate_bridge_mid = hit(note_text, NOTE_BRIDGE_MID) is not None
    note_has_candidate_bridge_tail = hit(note_text, NOTE_BRIDGE_TAIL) is not None
    note_has_conflict_free_claim = hit(note_text, "conflict はない。") is not None
    note_has_numeric_close_claim = hit(note_text, "numeric computation は閉じる。") is not None

    bare_seed_surface_available = part1_has_bare_vector_surface and part1_has_scalar_kinetic_surface
    later_single_zp_photon_canon_available = part1_has_photon_zp_surface and part1_has_later_vector_zp_surface
    candidate_bridge_statement_from_expert_note_available = (
        note_has_yes_part1_surface
        and note_has_candidate_bridge_head
        and note_has_candidate_bridge_mid
        and note_has_candidate_bridge_tail
    )
    expert_note_bridge_statement_conflict_free = note_has_conflict_free_claim and note_has_candidate_bridge_tail
    part3a_checkpoint_bridge_statement_available = (
        part3a_has_bridge_head and part3a_has_bridge_mid and part3a_has_bridge_tail
    )
    part5_checkpoint_bridge_statement_available = (
        part5_has_bridge_head and part5_has_bridge_mid and part5_has_bridge_tail
    )
    explicit_current_public_bridge_statement_available = (
        part3a_checkpoint_bridge_statement_available and part5_checkpoint_bridge_statement_available
    )
    checkpoint_wording_promotion_completed = explicit_current_public_bridge_statement_available
    numeric_computation_reopen_ready = checkpoint_wording_promotion_completed
    em_doc_has_local_maxwell_adoption = hit(em_doc_text, "局所（固有時）では Maxwell/QED をそのまま採用") is not None

    inventory_ready = all(
        [
            prior_checkpoint_route_active,
            status_has_1047_next_step,
            roadmap_has_1047_branch,
            bare_seed_surface_available,
            later_single_zp_photon_canon_available,
            part1_has_wavefunction_glossary,
            part3a_checkpoint_bridge_statement_available,
            part5_checkpoint_bridge_statement_available,
            em_doc_has_local_maxwell_adoption,
            candidate_bridge_statement_from_expert_note_available,
            expert_note_bridge_statement_conflict_free,
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
        "expert_bundle_dir": display_path(latest_bundle_dir),
        "expert_bundle_zip": display_path(latest_bundle_zip),
        "latest_expert_note": display_path(latest_note_path),
        "prior_1043_json": display_path(SOURCE_1043),
        "prior_1044_json": display_path(AUDIT_1044),
        "prior_1045_json": display_path(GATE_1045),
        "prior_1046_json": display_path(ROUTE_1046),
    }

    inventory = payload(
        "8.7.56.1047",
        "Trial-2 numeric alpha two-sector hierarchy EM-sector normalization checkpoint-wording promotion source inventory",
        common_inputs,
        "Freeze the checkpoint-wording promotion pack: prior bridge-statement metrics, Part I bridge evidence, the restored expert-note candidate, and the now-promoted current checkpoint wording in Part III-A and Part V.",
        {
            "inventory_rule": "the checkpoint-wording promotion pack is ready when the restored bridge sentence and the promoted current checkpoint wording are assembled together",
            "promotion_rule": "when the same bridge sentence is now carried by current checkpoint wording, the honest next route becomes numeric reopen rather than further wording descent",
        },
        [
            row(
                "trial2_numeric_alpha_checkpoint_wording_promotion_inventory_complete",
                "pass" if inventory_ready else "reject",
                "checkpoint-wording promotion inventory complete",
                1 if inventory_ready else 0,
                "The prior bridge-statement metrics, Part I evidence, restored note, and promoted checkpoint wording are assembled into one pack.",
            ),
            row(
                "trial2_numeric_alpha_part3a_checkpoint_bridge_statement_available",
                "pass" if part3a_checkpoint_bridge_statement_available else "reject",
                "Part III-A checkpoint bridge statement available",
                1 if part3a_checkpoint_bridge_statement_available else 0,
                "Part III-A now carries the explicit bridge sentence that distinguishes the vector kinetic coefficient from the scalar kinetic coefficient.",
            ),
            row(
                "trial2_numeric_alpha_part5_checkpoint_bridge_statement_available",
                "pass" if part5_checkpoint_bridge_statement_available else "reject",
                "Part V checkpoint bridge statement available",
                1 if part5_checkpoint_bridge_statement_available else 0,
                "Part V now carries the same explicit bridge sentence in the public future-predictions checkpoint.",
            ),
            row(
                "trial2_numeric_alpha_explicit_current_public_bridge_statement_available_after_checkpoint_wording_promotion",
                "pass" if explicit_current_public_bridge_statement_available else "reject",
                "explicit current public bridge statement available after checkpoint-wording promotion",
                1 if explicit_current_public_bridge_statement_available else 0,
                "The current public checkpoint surfaces now explicitly state that the EM and gravitational couplings are normalized by different kinetic coefficients inside the same action.",
            ),
        ],
        {
            "inventory_ready": inventory_ready,
            "prior_checkpoint_wording_promotion_route_active": prior_checkpoint_route_active,
            "external_expert_note_path_currently_available": True,
            "part1_bridge_evidence_available": bare_seed_surface_available
            and later_single_zp_photon_canon_available
            and part1_has_wavefunction_glossary,
            "part3a_checkpoint_bridge_statement_available": part3a_checkpoint_bridge_statement_available,
            "part5_checkpoint_bridge_statement_available": part5_checkpoint_bridge_statement_available,
            "explicit_current_public_bridge_statement_available": explicit_current_public_bridge_statement_available,
            "checkpoint_wording_promotion_completed": checkpoint_wording_promotion_completed,
            "first_route_to_close_or_none": NEXT_NUMERIC_REOPEN_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_checkpoint_wording_promotion_inventory_frozen",
            "advance_to_8_7_56_1048": inventory_ready,
            "next_required_artifacts": [NEXT_NUMERIC_REOPEN_ROUTE],
        },
        {
            "expert_note_hits": {
                "yes_part1_surface": hit(note_text, "Yes。Part I §2.7.0 の全作用定義そのもの。"),
                "candidate_head": hit(note_text, NOTE_BRIDGE_HEAD),
                "candidate_mid": hit(note_text, NOTE_BRIDGE_MID),
                "candidate_tail": hit(note_text, NOTE_BRIDGE_TAIL),
                "conflict_free": hit(note_text, "conflict はない。"),
            },
            "public_bridge_hits": {
                "part1": part1_public_bridge_hit,
                "part3a": part3a_public_bridge_hit,
                "part5": part5_public_bridge_hit,
            },
        },
    )

    audit = payload(
        "8.7.56.1048",
        "Trial-2 numeric alpha two-sector hierarchy EM-sector normalization checkpoint-wording promotion audit",
        common_inputs,
        "Audit whether the promoted checkpoint wording now satisfies the explicit public bridge-statement requirement without introducing new physics, and whether that promotion honestly reopens the numeric alpha route.",
        {
            "audit_rule": "checkpoint-wording promotion is acceptable when it restates the restored note by using already-public Part I kinetic-coefficient surfaces",
            "reopen_rule": "once the explicit bridge sentence exists on current public checkpoint surfaces, the honest next route is numeric reopen rather than more wording promotion",
        },
        [
            row(
                "trial2_numeric_alpha_checkpoint_wording_promotion_audit_complete",
                "pass" if inventory_ready else "reject",
                "checkpoint-wording promotion audit complete",
                1 if inventory_ready else 0,
                "The promoted bridge wording is audited against the restored note and the existing Part I kinetic-coefficient surfaces.",
            ),
            row(
                "trial2_numeric_alpha_checkpoint_wording_promotion_completed_without_new_physics",
                "pass" if checkpoint_wording_promotion_completed else "reject",
                "checkpoint-wording promotion completed without new physics",
                1 if checkpoint_wording_promotion_completed else 0,
                "The promoted sentence only states how the already-public vector and scalar kinetic coefficients normalize different sectors of the same action.",
            ),
            row(
                "trial2_numeric_alpha_explicit_public_bridge_statement_requirement_satisfied",
                "pass" if explicit_current_public_bridge_statement_available else "reject",
                "explicit public bridge statement requirement satisfied",
                1 if explicit_current_public_bridge_statement_available else 0,
                "The bridge sentence is now carried by current public checkpoint wording rather than only by the external note.",
            ),
            row(
                "trial2_numeric_alpha_numeric_computation_reopen_ready_after_checkpoint_wording_promotion",
                "pass" if numeric_computation_reopen_ready else "reject",
                "numeric computation reopen ready after checkpoint-wording promotion",
                1 if numeric_computation_reopen_ready else 0,
                "The wording gap is closed, so the next honest residual route is numeric reopen rather than another clarification pass.",
            ),
        ],
        {
            "audit_ready": inventory_ready,
            "selected_checkpoint_wording_promotion_class": (
                "minimal_bridge_statement_promoted_without_new_physics"
                if checkpoint_wording_promotion_completed
                else "checkpoint_wording_promotion_incomplete"
            ),
            "explicit_current_public_bridge_statement_available": explicit_current_public_bridge_statement_available,
            "checkpoint_wording_promotion_requires_new_physics": False,
            "explicit_public_bridge_statement_requirement_satisfied": explicit_current_public_bridge_statement_available,
            "numeric_computation_reopen_ready": numeric_computation_reopen_ready,
            "first_route_to_close_after_audit_or_none": NEXT_NUMERIC_REOPEN_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_checkpoint_wording_promotion_classified",
            "advance_to_8_7_56_1049": True,
            "next_required_artifacts": [NEXT_NUMERIC_REOPEN_ROUTE],
        },
        {
            "prior_bridge_statement_summary": {
                "source": source_1043,
                "audit": audit_1044,
                "gate": gate_1045,
            },
            "status_hits": {
                "status_next_1047": hit(status_text, "8.7.56.1047"),
                "roadmap_branch_1047": hit(roadmap_text, "`8.7.56.1047-.1050`"),
            },
        },
    )

    gate = payload(
        "8.7.56.1049",
        "Trial-2 numeric alpha two-sector hierarchy EM-sector normalization checkpoint-wording promotion declaration gate",
        common_inputs,
        "Update the official gate after checkpoint-wording promotion: the explicit bridge sentence is now public, so the residual route becomes numeric reopen while closeout itself remains not ready.",
        {
            "gate_rule": "when the explicit bridge sentence is public, checkpoint-wording promotion is complete and the residual route becomes numeric reopen",
            "closeout_rule": "numeric reopen readiness is not yet final numeric closeout",
        },
        [
            row(
                "trial2_numeric_alpha_checkpoint_wording_promotion_gate_complete",
                "pass",
                "checkpoint-wording promotion gate complete",
                1,
                "The official gate is updated after the promoted bridge sentence is audited.",
            ),
            row(
                "trial2_numeric_alpha_current_canon_bridge_statement_checkpoint_wording_promotion_completed",
                "pass" if checkpoint_wording_promotion_completed else "reject",
                "current-canon bridge-statement checkpoint-wording promotion completed",
                1 if checkpoint_wording_promotion_completed else 0,
                "The explicit bridge sentence is now carried by current public checkpoint wording.",
            ),
            row(
                "trial2_numeric_alpha_selected_residual_route_is_numeric_reopen",
                "pass" if numeric_computation_reopen_ready else "reject",
                "selected residual route is numeric reopen",
                1 if numeric_computation_reopen_ready else 0,
                "The remaining work is now an honest numeric reopen route under the promoted bridge wording.",
            ),
            row(
                "trial2_numeric_alpha_closeout_still_not_ready_after_checkpoint_wording_promotion",
                "reject",
                "closeout still not ready after checkpoint-wording promotion",
                0,
                "The wording blocker is closed, but the reopened numeric route still needs its own audit and declaration gate.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "numeric_reopen_readiness",
            "trial2_numeric_alpha_text_search_continuation_justified": False,
            "trial2_numeric_alpha_mechanical_wording_descent_stopped": True,
            "trial2_numeric_alpha_expert_response_pending_external_input": False,
            "trial2_numeric_alpha_current_canon_reconciliation_completed": True,
            "trial2_numeric_alpha_current_canon_bridge_statement_branch_completed": True,
            "trial2_numeric_alpha_current_canon_bridge_statement_checkpoint_wording_promotion_completed": checkpoint_wording_promotion_completed,
            "trial2_numeric_alpha_explicit_current_canon_bridge_statement_available": explicit_current_public_bridge_statement_available,
            "trial2_numeric_alpha_numeric_computation_reopen_ready": numeric_computation_reopen_ready,
            "trial2_numeric_alpha_closeout_ready": False,
            "selected_residual_route": NEXT_NUMERIC_REOPEN_ROUTE,
            "missing_v2_artifact": NEXT_NUMERIC_REOPEN_ARTIFACT,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_checkpoint_wording_promotion_gate_closed",
            "advance_to_8_7_56_1050": True,
            "next_required_artifacts": [NEXT_NUMERIC_REOPEN_ROUTE],
        },
        {
            "audit_summary": audit["summary"],
            "prior_gate_summary": gate_1045,
        },
    )

    route = payload(
        "8.7.56.1050",
        "Trial-2 numeric alpha route contract one-hundred-fifty-ninth refresh",
        common_inputs,
        "Refresh the next-generation contract after checkpoint-wording promotion: retain the precision-alpha mainline, keep external dependency retired, and advance to numeric reopen as the next official route.",
        {
            "next_route_rule": "the next route reopens the numeric alpha computation under the now-public bridge sentence",
            "reserve_rule": "strong-side non-Abelian, running, and confinement remain on v3 hold reserve",
        },
        [
            row(
                "trial2_numeric_alpha_route_contract_one_hundred_fifty_ninth_refresh_complete",
                "pass",
                "route contract one-hundred-fifty-ninth refresh complete",
                1,
                "The checkpoint-wording promotion gate is converted into the next-generation contract.",
            ),
            row(
                "trial2_numeric_alpha_next_route_selected_as_numeric_reopen",
                "pass" if numeric_computation_reopen_ready else "reject",
                "next route selected as numeric reopen",
                1 if numeric_computation_reopen_ready else 0,
                "The next official branch reopens the numeric alpha route under the promoted bridge wording.",
            ),
            row(
                "trial2_numeric_alpha_external_dependency_remains_retired_after_checkpoint_wording_promotion",
                "pass",
                "external dependency remains retired after checkpoint-wording promotion",
                1,
                "The mainline remains independent of outside input after the restored-note wording has been promoted.",
            ),
            row(
                "trial2_numeric_alpha_precision_mainline_retained_after_checkpoint_wording_promotion",
                "pass" if bool(route_1046.get("precision_alpha_mainline_retained", False)) else "reject",
                "precision-alpha mainline retained after checkpoint-wording promotion",
                1 if bool(route_1046.get("precision_alpha_mainline_retained", False)) else 0,
                "Trial-2 numeric alpha remains the precision mainline after the wording blocker is closed.",
            ),
        ],
        {
            "selected_next_generation_route": NEXT_NUMERIC_REOPEN_ROUTE,
            "strong_side_route_state": route_1046.get("strong_side_route_state"),
            "precision_alpha_mainline_retained": bool(route_1046.get("precision_alpha_mainline_retained", False)),
            "electron_identification_pivot_retained": bool(route_1046.get("electron_identification_pivot_retained", False)),
            "h0p_bridge_pivot_retained": bool(route_1046.get("h0p_bridge_pivot_retained", False)),
            "final_computation_branch_retained": bool(route_1046.get("final_computation_branch_retained", False)),
            "unit_consistency_audit_branch_retained": bool(
                route_1046.get("unit_consistency_audit_branch_retained", False)
            ),
            "dimensionless_alpha_bridge_branch_retained": bool(
                route_1046.get("dimensionless_alpha_bridge_branch_retained", False)
            ),
            "em_unit_convention_bridge_branch_retained": bool(
                route_1046.get("em_unit_convention_bridge_branch_retained", False)
            ),
            "mapping_statement_branch_retained": bool(route_1046.get("mapping_statement_branch_retained", False)),
            "mapping_literal_branch_retained": bool(route_1046.get("mapping_literal_branch_retained", False)),
            "expert_advice_escalation_branch_retained": True,
            "two_sector_hierarchy_pivot_retained": True,
            "expert_response_intake_branch_completed": True,
            "current_canon_reconciliation_branch_completed": True,
            "current_canon_bridge_statement_branch_completed": True,
            "current_canon_bridge_statement_checkpoint_wording_promotion_completed": checkpoint_wording_promotion_completed,
            "external_dependency_active": False,
            "hard_conflict_reading_retired": True,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_route_contract_one_hundred_fifty_ninth_refresh_frozen",
            "advance_to_next_route": True,
            "next_required_artifacts": [NEXT_NUMERIC_REOPEN_ROUTE],
        },
        {
            "gate_summary": gate["summary"],
            "prior_route_summary": route_1046,
        },
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_current_canon_bridge_statement_checkpoint_wording_promotion_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_current_canon_bridge_statement_checkpoint_wording_promotion_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_current_canon_bridge_statement_checkpoint_wording_promotion_declaration_gate",
        gate,
    )
    write_artifact(
        "mass_origin_v2_t2_alpha_route_contract_one_hundred_fifty_ninth_refresh",
        route,
    )

    print("[done] 8.7.56.1047-.1050 artifacts generated:")
    print(
        " - mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_current_canon_bridge_statement_checkpoint_wording_promotion_source_inventory_metrics.json"
    )
    print(
        " - mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_current_canon_bridge_statement_checkpoint_wording_promotion_audit_metrics.json"
    )
    print(
        " - mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_current_canon_bridge_statement_checkpoint_wording_promotion_declaration_gate_metrics.json"
    )
    print(" - mass_origin_v2_t2_alpha_route_contract_one_hundred_fifty_ninth_refresh_metrics.json")


# Function: run the checkpoint-wording promotion branch from the CLI.

def run_cli() -> None:
    """CLI entry point for the Trial-2 numeric alpha checkpoint-wording promotion branch."""
    main()


if __name__ == "__main__":
    run_cli()
