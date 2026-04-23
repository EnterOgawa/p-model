#!/usr/bin/env python3
"""Generate 8.7.56.1051-.1054 Trial-2 numeric alpha numeric-reopen artifacts."""

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

SOURCE_1047 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "current_canon_bridge_statement_checkpoint_wording_promotion_source_inventory_metrics.json"
)
AUDIT_1048 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "current_canon_bridge_statement_checkpoint_wording_promotion_audit_metrics.json"
)
GATE_1049 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "current_canon_bridge_statement_checkpoint_wording_promotion_declaration_gate_metrics.json"
)
ROUTE_1050 = PUBLIC_OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_fifty_ninth_refresh_metrics.json"

FINAL_SOURCE_979 = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_final_computation_source_inventory_metrics.json"
FINAL_GATE_981 = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_final_computation_declaration_gate_metrics.json"
UNIT_GATE_985 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_final_computation_unit_consistency_declaration_gate_metrics.json"
)
DIM_AUDIT_988 = PUBLIC_OUT / "mass_origin_v2_trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_audit_metrics.json"
EM_AUDIT_992 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_em_unit_convention_audit_metrics.json"
)
MAP_STMT_AUDIT_996 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_"
    "gp_to_elementary_charge_mapping_statement_audit_metrics.json"
)
MAP_LIT_AUDIT_1000 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_"
    "gp_to_elementary_charge_mapping_literal_audit_metrics.json"
)

CURRENT_ROUTE = "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_numeric_reopen"
NEXT_DIMENSIONLESS_ROUTE = (
    "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_dimensionless_alpha_bridge_reopen"
)
NEXT_DIMENSIONLESS_ARTIFACT = (
    "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_dimensionless_alpha_bridge_reopen_note"
)
NEXT_ROUTE = "8.7.56.1055"

CHECKPOINT_BRIDGE_HEAD = (
    "current checkpoint wording としては、電磁結合は Part I 2.7.0 の vector kinetic coefficient"
)
CHECKPOINT_BRIDGE_TAIL = "これらは同一作用の別 sector である"


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


# Function: return a stable display path for repo files.

def display_path(path: Path) -> str:
    """Return a stable path relative to the repo root when possible."""
    return str(path.relative_to(ROOT)).replace("\\", "/")


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


# Function: build a standard metrics payload.

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


# Function: execute the numeric-reopen branch.

def main() -> None:
    """Execute the Trial-2 numeric alpha numeric-reopen branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        PRIMARY_SOURCES,
        PART1,
        PART3A,
        PART5,
        EM_DOC,
        SOURCE_1047,
        AUDIT_1048,
        GATE_1049,
        ROUTE_1050,
        FINAL_SOURCE_979,
        FINAL_GATE_981,
        UNIT_GATE_985,
        DIM_AUDIT_988,
        EM_AUDIT_992,
        MAP_STMT_AUDIT_996,
        MAP_LIT_AUDIT_1000,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    em_doc_text = read_text(EM_DOC)

    source_1047 = read_json(SOURCE_1047)["summary"]
    audit_1048 = read_json(AUDIT_1048)["summary"]
    gate_1049 = read_json(GATE_1049)["summary"]
    route_1050 = read_json(ROUTE_1050)["summary"]
    final_source_979 = read_json(FINAL_SOURCE_979)["summary"]
    final_gate_981 = read_json(FINAL_GATE_981)["summary"]
    unit_gate_985 = read_json(UNIT_GATE_985)["summary"]
    dim_audit_988 = read_json(DIM_AUDIT_988)["summary"]
    em_audit_992 = read_json(EM_AUDIT_992)["summary"]
    map_stmt_audit_996 = read_json(MAP_STMT_AUDIT_996)["summary"]
    map_lit_audit_1000 = read_json(MAP_LIT_AUDIT_1000)["summary"]

    prior_numeric_reopen_route_active = (
        source_1047["first_route_to_close_or_none"] == CURRENT_ROUTE
        and audit_1048["first_route_to_close_after_audit_or_none"] == CURRENT_ROUTE
        and gate_1049["selected_residual_route"] == CURRENT_ROUTE
        and route_1050["selected_next_generation_route"] == CURRENT_ROUTE
        and not bool(route_1050["external_dependency_active"])
    )

    status_has_1051_next_step = hit(status_text, "8.7.56.1051") is not None
    roadmap_has_1051_branch = hit(roadmap_text, "`8.7.56.1051-.1054`") is not None

    part1_has_weak_field_normalization = hit(part1_text, r"g_P/Z_P=4\pi G") is not None
    part3a_has_bridge_head = hit(part3a_text, CHECKPOINT_BRIDGE_HEAD) is not None
    part3a_has_bridge_tail = hit(part3a_text, CHECKPOINT_BRIDGE_TAIL) is not None
    part5_has_bridge_head = hit(part5_text, CHECKPOINT_BRIDGE_HEAD) is not None
    part5_has_bridge_tail = hit(part5_text, CHECKPOINT_BRIDGE_TAIL) is not None
    part3a_has_final_formula = hit(part3a_text, r"\alpha=\frac{4\pi G^2 Z_P}{\hbar c}") is not None
    part3a_has_raw_alpha_candidate = hit(part3a_text, "2.748672883601193") is not None
    part3a_has_dimension_vector = hit(part3a_text, "kg^{-1}m^3") is not None
    part3a_has_dimensionless_alpha_gap = (
        hit(part3a_text, "dimensionless fine-structure constant になっていない") is not None
    )
    part3a_has_mapping_statement_absent = (
        hit(part3a_text, r"explicit $g_P\leftrightarrow e$ charge-normalization statement") is not None
    )
    part5_has_numeric_reopen_next = hit(part5_text, "numeric-reopen next / closeout not ready") is not None
    em_doc_has_local_maxwell_adoption = hit(em_doc_text, "局所（固有時）では Maxwell/QED をそのまま採用") is not None

    explicit_current_public_bridge_statement_available = (
        bool(source_1047["explicit_current_public_bridge_statement_available"])
        and bool(audit_1048["explicit_public_bridge_statement_requirement_satisfied"])
        and bool(gate_1049["trial2_numeric_alpha_explicit_current_canon_bridge_statement_available"])
    )
    checkpoint_wording_promotion_completed = bool(
        gate_1049["trial2_numeric_alpha_current_canon_bridge_statement_checkpoint_wording_promotion_completed"]
    )

    retained_final_computation_input_pack_ready = bool(final_source_979["final_computation_input_pack_ready"])
    retained_computation_formula_ready = bool(unit_gate_985["trial2_numeric_alpha_computation_formula_ready"])
    retained_absolute_normalization_dictionary_ready = bool(
        unit_gate_985["trial2_numeric_alpha_absolute_normalization_dictionary_ready"]
    )
    retained_raw_final_computation_value_available = bool(
        unit_gate_985["trial2_numeric_alpha_raw_final_computation_value_available"]
    )
    retained_raw_final_numeric_emittable_before_unit_audit = bool(
        final_gate_981["trial2_numeric_alpha_numeric_from_current_pack_ready"]
    )
    retained_unit_consistency_boundary_active = (
        unit_gate_985["trial2_numeric_alpha_final_computation_result_class"] == "precanonical_unit_incomplete"
        and unit_gate_985["selected_residual_route"] == "trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge"
    )
    retained_dimensionless_alpha_bridge_active = (
        not bool(dim_audit_988["explicit_dimensionless_alpha_bridge_available"])
        and dim_audit_988["first_route_to_close_after_audit_or_none"]
        == "trial2_numeric_alpha_final_computation_dimensionless_alpha_bridge_em_unit_convention"
    )
    retained_charge_mapping_statement_absent = (
        not bool(em_audit_992["explicit_gp_to_elementary_charge_mapping_available"])
        and not bool(map_stmt_audit_996["explicit_mapping_statement_available"])
    )
    retained_charge_mapping_literal_absent = (
        not bool(map_lit_audit_1000["explicit_mapping_literal_available"])
        and not bool(map_lit_audit_1000["new_public_canonical_surface_added_in_literal_branch"])
    )

    raw_final_computation_reopen_ready = (
        explicit_current_public_bridge_statement_available
        and checkpoint_wording_promotion_completed
        and retained_final_computation_input_pack_ready
        and retained_computation_formula_ready
        and retained_absolute_normalization_dictionary_ready
        and retained_raw_final_computation_value_available
        and retained_raw_final_numeric_emittable_before_unit_audit
    )
    dimensionless_alpha_bridge_still_missing = retained_dimensionless_alpha_bridge_active
    explicit_gp_to_elementary_charge_mapping_available = not retained_charge_mapping_statement_absent
    numeric_reopen_scope_fixed = (
        raw_final_computation_reopen_ready
        and retained_unit_consistency_boundary_active
        and dimensionless_alpha_bridge_still_missing
        and retained_charge_mapping_literal_absent
    )

    inventory_ready = all(
        [
            prior_numeric_reopen_route_active,
            status_has_1051_next_step,
            roadmap_has_1051_branch,
            part1_has_weak_field_normalization,
            part3a_has_bridge_head,
            part3a_has_bridge_tail,
            part5_has_bridge_head,
            part5_has_bridge_tail,
            part3a_has_final_formula,
            part3a_has_raw_alpha_candidate,
            part3a_has_dimension_vector,
            part3a_has_dimensionless_alpha_gap,
            part3a_has_mapping_statement_absent,
            part5_has_numeric_reopen_next,
            em_doc_has_local_maxwell_adoption,
            raw_final_computation_reopen_ready,
            retained_unit_consistency_boundary_active,
            dimensionless_alpha_bridge_still_missing,
            retained_charge_mapping_statement_absent,
            retained_charge_mapping_literal_absent,
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
        "prior_1047_json": display_path(SOURCE_1047),
        "prior_1048_json": display_path(AUDIT_1048),
        "prior_1049_json": display_path(GATE_1049),
        "prior_1050_json": display_path(ROUTE_1050),
        "retained_979_json": display_path(FINAL_SOURCE_979),
        "retained_981_json": display_path(FINAL_GATE_981),
        "retained_985_json": display_path(UNIT_GATE_985),
        "retained_988_json": display_path(DIM_AUDIT_988),
        "retained_992_json": display_path(EM_AUDIT_992),
        "retained_996_json": display_path(MAP_STMT_AUDIT_996),
        "retained_1000_json": display_path(MAP_LIT_AUDIT_1000),
    }

    inventory = payload(
        "8.7.56.1051",
        "Trial-2 numeric alpha two-sector hierarchy EM-sector normalization numeric-reopen source inventory",
        common_inputs,
        "Freeze the numeric-reopen pack: promoted current checkpoint wording, Part I bridge evidence, the retained raw final-computation stack, and the retained unit / dimensionless-alpha / charge-mapping audits that bound the honest reopen scope.",
        {
            "inventory_rule": "the numeric-reopen pack is ready when the promoted bridge sentence and the retained final-computation-to-dimensionless-bridge family are assembled together",
            "reopen_rule": "checkpoint-wording promotion reopens the retained raw final computation stack, but it does not by itself solve the older dimensionless-alpha bridge family",
        },
        [
            row(
                "trial2_numeric_alpha_numeric_reopen_inventory_complete",
                "pass" if inventory_ready else "reject",
                "numeric-reopen inventory complete",
                1 if inventory_ready else 0,
                "The promoted bridge wording and the retained final-computation / dimensionless-bridge family are assembled into one numeric-reopen pack.",
            ),
            row(
                "trial2_numeric_alpha_explicit_current_public_bridge_statement_available_for_numeric_reopen",
                "pass" if explicit_current_public_bridge_statement_available else "reject",
                "explicit current public bridge statement available for numeric reopen",
                1 if explicit_current_public_bridge_statement_available else 0,
                "The EM-sector normalization blocker is retired because the bridge sentence is now public checkpoint wording.",
            ),
            row(
                "trial2_numeric_alpha_raw_final_computation_reopens_after_checkpoint_wording_promotion",
                "pass" if raw_final_computation_reopen_ready else "reject",
                "raw final computation reopens after checkpoint-wording promotion",
                1 if raw_final_computation_reopen_ready else 0,
                "The retained final-computation stack can now be treated as the honest reopened numeric scope under the public bridge sentence.",
            ),
            row(
                "trial2_numeric_alpha_dimensionless_alpha_bridge_family_remains_active_after_numeric_reopen",
                "pass" if dimensionless_alpha_bridge_still_missing else "reject",
                "dimensionless-alpha bridge family remains active after numeric reopen",
                1 if dimensionless_alpha_bridge_still_missing else 0,
                "The reopened scope still lands on the retained dimensionless-alpha / charge-mapping blocker family rather than on closeout.",
            ),
        ],
        {
            "inventory_ready": inventory_ready,
            "prior_numeric_reopen_route_active": prior_numeric_reopen_route_active,
            "explicit_current_public_bridge_statement_available": explicit_current_public_bridge_statement_available,
            "checkpoint_wording_promotion_completed": checkpoint_wording_promotion_completed,
            "retained_final_computation_input_pack_ready": retained_final_computation_input_pack_ready,
            "retained_raw_final_computation_value_available": retained_raw_final_computation_value_available,
            "retained_unit_consistency_boundary_active": retained_unit_consistency_boundary_active,
            "retained_dimensionless_alpha_bridge_active": retained_dimensionless_alpha_bridge_active,
            "retained_charge_mapping_statement_absent": retained_charge_mapping_statement_absent,
            "retained_charge_mapping_literal_absent": retained_charge_mapping_literal_absent,
            "first_route_to_close_or_none": NEXT_DIMENSIONLESS_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_numeric_reopen_inventory_frozen",
            "advance_to_8_7_56_1052": inventory_ready,
            "next_required_artifacts": [NEXT_DIMENSIONLESS_ROUTE],
        },
        {
            "checkpoint_wording_hits": {
                "part3a_bridge_head": hit(part3a_text, CHECKPOINT_BRIDGE_HEAD),
                "part3a_bridge_tail": hit(part3a_text, CHECKPOINT_BRIDGE_TAIL),
                "part5_bridge_head": hit(part5_text, CHECKPOINT_BRIDGE_HEAD),
                "part5_bridge_tail": hit(part5_text, CHECKPOINT_BRIDGE_TAIL),
            },
            "retained_numeric_hits": {
                "part1_weak_field_normalization": hit(part1_text, r"g_P/Z_P=4\pi G"),
                "part3a_final_formula": hit(part3a_text, r"\alpha=\frac{4\pi G^2 Z_P}{\hbar c}"),
                "part3a_raw_alpha_candidate": hit(part3a_text, "2.748672883601193"),
                "part3a_dimension_vector": hit(part3a_text, "kg^{-1}m^3"),
                "part3a_mapping_statement_absent": hit(
                    part3a_text, r"explicit $g_P\leftrightarrow e$ charge-normalization statement"
                ),
            },
        },
    )

    audit = payload(
        "8.7.56.1052",
        "Trial-2 numeric alpha two-sector hierarchy EM-sector normalization numeric-reopen audit",
        common_inputs,
        "Audit what the public bridge sentence now honestly reopens: the retained raw final-computation stack is back on the table, but the retained dimensionless-alpha / charge-mapping blocker family still prevents closeout.",
        {
            "audit_rule": "once the EM-sector normalization bridge is public, the honest reopen scope is the retained raw final-computation stack up to the pre-existing unit-consistency boundary",
            "closeout_rule": "closeout still requires a dimensionless-alpha bridge and an explicit charge-normalization mapping or equivalent SI alpha formula",
        },
        [
            row(
                "trial2_numeric_alpha_numeric_reopen_audit_complete",
                "pass" if inventory_ready else "reject",
                "numeric-reopen audit complete",
                1 if inventory_ready else 0,
                "The reopened numeric scope is audited against the promoted bridge wording and the retained final-computation / dimensionless-bridge family.",
            ),
            row(
                "trial2_numeric_alpha_em_sector_normalization_blocker_retired_after_numeric_reopen_audit",
                "pass" if explicit_current_public_bridge_statement_available else "reject",
                "EM-sector normalization blocker retired after numeric-reopen audit",
                1 if explicit_current_public_bridge_statement_available else 0,
                "The public bridge sentence closes the specific EM-vs-gravity normalization blocker that forced the detour.",
            ),
            row(
                "trial2_numeric_alpha_raw_final_computation_is_the_honest_reopened_numeric_scope",
                "pass" if raw_final_computation_reopen_ready else "reject",
                "raw final computation is the honest reopened numeric scope",
                1 if raw_final_computation_reopen_ready else 0,
                "The retained raw alpha candidate can be carried forward again, but only as the reopened precursor to the retained dimensionless-alpha audits.",
            ),
            row(
                "trial2_numeric_alpha_dimensionless_alpha_bridge_still_missing_after_numeric_reopen_audit",
                "pass" if dimensionless_alpha_bridge_still_missing else "reject",
                "dimensionless-alpha bridge still missing after numeric-reopen audit",
                1 if dimensionless_alpha_bridge_still_missing else 0,
                "The promoted bridge wording does not itself supply the missing SI / charge-normalization bridge that would make alpha dimensionless.",
            ),
            row(
                "trial2_numeric_alpha_closeout_ready_after_numeric_reopen_audit",
                "reject",
                "closeout ready after numeric-reopen audit",
                0,
                "The reopened scope remains pre-closeout because the direct SI readout still carries units and the charge-mapping family is still unresolved.",
            ),
        ],
        {
            "audit_ready": inventory_ready,
            "selected_numeric_reopen_class": (
                "raw_final_computation_reopened_but_dimensionless_alpha_bridge_still_missing"
                if numeric_reopen_scope_fixed
                else "numeric_reopen_scope_incomplete"
            ),
            "explicit_current_public_bridge_statement_available": explicit_current_public_bridge_statement_available,
            "raw_final_computation_reopen_ready": raw_final_computation_reopen_ready,
            "retained_raw_final_computation_result_class": unit_gate_985[
                "trial2_numeric_alpha_final_computation_result_class"
            ],
            "dimensionless_alpha_bridge_still_missing": dimensionless_alpha_bridge_still_missing,
            "explicit_gp_to_elementary_charge_mapping_available": explicit_gp_to_elementary_charge_mapping_available,
            "numeric_closeout_ready": False,
            "first_route_to_close_after_audit_or_none": NEXT_DIMENSIONLESS_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_numeric_reopen_classified",
            "advance_to_8_7_56_1053": True,
            "next_required_artifacts": [NEXT_DIMENSIONLESS_ROUTE],
        },
        {
            "prior_checkpoint_wording_summary": {
                "source": source_1047,
                "audit": audit_1048,
                "gate": gate_1049,
            },
            "retained_numeric_summary": {
                "final_source": final_source_979,
                "final_gate": final_gate_981,
                "unit_gate": unit_gate_985,
                "dimensionless_audit": dim_audit_988,
                "em_unit_audit": em_audit_992,
                "mapping_statement_audit": map_stmt_audit_996,
                "mapping_literal_audit": map_lit_audit_1000,
            },
        },
    )

    gate = payload(
        "8.7.56.1053",
        "Trial-2 numeric alpha two-sector hierarchy EM-sector normalization numeric-reopen declaration gate",
        common_inputs,
        "Update the official gate after numeric reopen: raw final computation is the honest reopened scope, but the residual blocker family is still the reopened dimensionless-alpha bridge rather than closeout.",
        {
            "gate_rule": "numeric reopen can be honest even when canonical numeric readiness remains false, as long as the reopened scope is fixed at the retained pre-closeout boundary",
            "residual_rule": "the next residual route is the reopened dimensionless-alpha bridge family",
        },
        [
            row(
                "trial2_numeric_alpha_numeric_reopen_gate_complete",
                "pass",
                "numeric-reopen gate complete",
                1,
                "The official gate is updated after the reopened numeric scope is audited.",
            ),
            row(
                "trial2_numeric_alpha_raw_final_computation_reopen_completed",
                "pass" if raw_final_computation_reopen_ready else "reject",
                "raw final computation reopen completed",
                1 if raw_final_computation_reopen_ready else 0,
                "The retained raw final-computation stack is now officially back on the mainline as the reopened numeric precursor.",
            ),
            row(
                "trial2_numeric_alpha_dimensionless_alpha_bridge_reopen_required",
                "pass" if dimensionless_alpha_bridge_still_missing else "reject",
                "dimensionless-alpha bridge reopen required",
                1 if dimensionless_alpha_bridge_still_missing else 0,
                "The reopened numeric path now hands off to the retained dimensionless-alpha bridge family.",
            ),
            row(
                "trial2_numeric_alpha_closeout_still_not_ready_after_numeric_reopen",
                "reject",
                "closeout still not ready after numeric reopen",
                0,
                "The bridge sentence is public, but the direct SI readout is still pre-canonical and the charge-mapping family remains unresolved.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "reopened_dimensionless_alpha_bridge",
            "trial2_numeric_alpha_text_search_continuation_justified": False,
            "trial2_numeric_alpha_mechanical_wording_descent_stopped": True,
            "trial2_numeric_alpha_current_canon_bridge_statement_checkpoint_wording_promotion_completed": checkpoint_wording_promotion_completed,
            "trial2_numeric_alpha_raw_final_computation_reopen_ready": raw_final_computation_reopen_ready,
            "trial2_numeric_alpha_numeric_from_current_pack_ready": False,
            "trial2_numeric_alpha_closeout_ready": False,
            "trial2_numeric_alpha_dimensionless_alpha_bridge_reopen_required": dimensionless_alpha_bridge_still_missing,
            "trial2_numeric_alpha_explicit_gp_to_elementary_charge_mapping_available": explicit_gp_to_elementary_charge_mapping_available,
            "selected_residual_route": NEXT_DIMENSIONLESS_ROUTE,
            "missing_v2_artifact": NEXT_DIMENSIONLESS_ARTIFACT,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_numeric_reopen_gate_closed",
            "advance_to_8_7_56_1054": True,
            "next_required_artifacts": [NEXT_DIMENSIONLESS_ROUTE],
        },
        {
            "audit_summary": audit["summary"],
            "retained_unit_gate_summary": unit_gate_985,
        },
    )

    route = payload(
        "8.7.56.1054",
        "Trial-2 numeric alpha route contract one-hundred-sixtieth refresh",
        common_inputs,
        "Refresh the next-generation contract after numeric reopen: keep the precision-alpha mainline active, keep the strong side on reserve, and hand the reopened scope to the dimensionless-alpha bridge family.",
        {
            "next_route_rule": "the next route re-audits the retained dimensionless-alpha bridge family under the now-public bridge sentence",
            "reserve_rule": "strong-side non-Abelian, running, and confinement remain on v3 hold reserve",
        },
        [
            row(
                "trial2_numeric_alpha_route_contract_one_hundred_sixtieth_refresh_complete",
                "pass",
                "route contract one-hundred-sixtieth refresh complete",
                1,
                "The numeric-reopen gate is converted into the next-generation contract.",
            ),
            row(
                "trial2_numeric_alpha_next_route_selected_as_dimensionless_alpha_bridge_reopen",
                "pass" if dimensionless_alpha_bridge_still_missing else "reject",
                "next route selected as dimensionless-alpha bridge reopen",
                1 if dimensionless_alpha_bridge_still_missing else 0,
                "The next official branch reopens the retained dimensionless-alpha bridge family under the public bridge sentence.",
            ),
            row(
                "trial2_numeric_alpha_external_dependency_remains_retired_after_numeric_reopen",
                "pass",
                "external dependency remains retired after numeric reopen",
                1,
                "The mainline remains independent of outside input after the checkpoint-wording promotion and numeric-reopen scope fix.",
            ),
            row(
                "trial2_numeric_alpha_precision_mainline_retained_after_numeric_reopen",
                "pass" if bool(route_1050.get("precision_alpha_mainline_retained", False)) else "reject",
                "precision-alpha mainline retained after numeric reopen",
                1 if bool(route_1050.get("precision_alpha_mainline_retained", False)) else 0,
                "Trial-2 numeric alpha remains the precision mainline while the reopened dimensionless bridge is still pending.",
            ),
        ],
        {
            "selected_next_generation_route": NEXT_DIMENSIONLESS_ROUTE,
            "strong_side_route_state": route_1050.get("strong_side_route_state"),
            "precision_alpha_mainline_retained": bool(route_1050.get("precision_alpha_mainline_retained", False)),
            "electron_identification_pivot_retained": bool(route_1050.get("electron_identification_pivot_retained", False)),
            "h0p_bridge_pivot_retained": bool(route_1050.get("h0p_bridge_pivot_retained", False)),
            "final_computation_branch_retained": True,
            "unit_consistency_audit_branch_retained": True,
            "dimensionless_alpha_bridge_branch_retained": True,
            "em_unit_convention_bridge_branch_retained": True,
            "mapping_statement_branch_retained": True,
            "mapping_literal_branch_retained": True,
            "two_sector_hierarchy_pivot_retained": True,
            "current_canon_bridge_statement_checkpoint_wording_promotion_completed": checkpoint_wording_promotion_completed,
            "numeric_reopen_scope_fixed": numeric_reopen_scope_fixed,
            "external_dependency_active": False,
            "hard_conflict_reading_retired": True,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_route_contract_one_hundred_sixtieth_refresh_frozen",
            "advance_to_next_route": True,
            "next_required_artifacts": [NEXT_DIMENSIONLESS_ROUTE],
        },
        {
            "gate_summary": gate["summary"],
            "prior_route_summary": route_1050,
        },
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_numeric_reopen_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_numeric_reopen_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_numeric_reopen_declaration_gate",
        gate,
    )
    write_artifact("mass_origin_v2_t2_alpha_route_contract_one_hundred_sixtieth_refresh", route)

    print("[done] 8.7.56.1051-.1054 artifacts generated:")
    print(" - mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_numeric_reopen_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_numeric_reopen_audit_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_numeric_reopen_declaration_gate_metrics.json")
    print(" - mass_origin_v2_t2_alpha_route_contract_one_hundred_sixtieth_refresh_metrics.json")


# Function: run the numeric-reopen branch from the CLI.

def run_cli() -> None:
    """CLI entry point for the Trial-2 numeric alpha numeric-reopen branch."""
    main()


if __name__ == "__main__":
    run_cli()
