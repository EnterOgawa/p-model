#!/usr/bin/env python3
"""Generate 8.7.56.1055-.1058 Trial-2 numeric alpha dimensionless-bridge reopen artifacts."""

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

SOURCE_1051 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "numeric_reopen_source_inventory_metrics.json"
)
AUDIT_1052 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "numeric_reopen_audit_metrics.json"
)
GATE_1053 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "numeric_reopen_declaration_gate_metrics.json"
)
ROUTE_1054 = PUBLIC_OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_sixtieth_refresh_metrics.json"

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
EXPERT_BUNDLE_AUDIT_1032 = PUBLIC_OUT / (
    "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "expert_bundle_refresh_audit_metrics.json"
)

CURRENT_ROUTE = (
    "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_dimensionless_alpha_bridge_reopen"
)
NEXT_NO_GO_ROUTE = (
    "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "dimensionless_alpha_bridge_no_go_closeout"
)
NEXT_NO_GO_ARTIFACT = (
    "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
    "dimensionless_alpha_bridge_no_go_closeout_note"
)
NEXT_ROUTE = "8.7.56.1059"

CHECKPOINT_BRIDGE_HEAD = (
    "current checkpoint wording としては、電磁結合は Part I 2.7.0 の vector kinetic coefficient"
)
RAW_DIRECT_SI_FORMULA = r"\alpha=4\pi G^2Z_P/(\hbar c)"
DIMENSION_VECTOR = "kg^{-1}m^3"
DIMENSIONLESS_GAP = "dimensionless fine-structure constant になっていない"
EXPLICIT_BRIDGE_GAP = r"explicit SI alpha formula / explicit $g_P\leftrightarrow e$ mapping /"
MAPPING_STATEMENT_GAP = r"explicit $g_P\leftrightarrow e$ charge-normalization statement"
LOCAL_MAXWELL_ADOPTION = "局所（固有時）では Maxwell/QED をそのまま採用"
REOPEN_NEXT_STATE = "dimensionless-alpha-bridge reopen next / closeout not ready"
REOPEN_SCOPE_BLOCKER = "dimensionless-alpha-bridge reopen scope"


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


# Function: execute the dimensionless-alpha-bridge reopen branch.

def main() -> None:
    """Execute the Trial-2 numeric alpha dimensionless-alpha-bridge reopen branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        PRIMARY_SOURCES,
        PART1,
        PART3A,
        PART5,
        EM_DOC,
        SOURCE_1051,
        AUDIT_1052,
        GATE_1053,
        ROUTE_1054,
        DIM_AUDIT_988,
        EM_AUDIT_992,
        MAP_STMT_AUDIT_996,
        MAP_LIT_AUDIT_1000,
        EXPERT_BUNDLE_AUDIT_1032,
    ):
        require(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    em_doc_text = read_text(EM_DOC)

    source_1051 = read_json(SOURCE_1051)["summary"]
    audit_1052 = read_json(AUDIT_1052)["summary"]
    gate_1053 = read_json(GATE_1053)["summary"]
    route_1054 = read_json(ROUTE_1054)["summary"]
    dim_audit_988 = read_json(DIM_AUDIT_988)["summary"]
    em_audit_992 = read_json(EM_AUDIT_992)["summary"]
    map_stmt_audit_996 = read_json(MAP_STMT_AUDIT_996)["summary"]
    map_lit_audit_1000 = read_json(MAP_LIT_AUDIT_1000)["summary"]
    expert_bundle_audit_1032 = read_json(EXPERT_BUNDLE_AUDIT_1032)["summary"]

    prior_dimensionless_bridge_reopen_route_active = (
        source_1051["first_route_to_close_or_none"] == CURRENT_ROUTE
        and audit_1052["first_route_to_close_after_audit_or_none"] == CURRENT_ROUTE
        and gate_1053["selected_residual_route"] == CURRENT_ROUTE
        and route_1054["selected_next_generation_route"] == CURRENT_ROUTE
        and bool(route_1054["numeric_reopen_scope_fixed"])
        and not bool(route_1054["external_dependency_active"])
    )

    status_has_1055_next_step = hit(status_text, "8.7.56.1055") is not None
    roadmap_has_1055_branch = hit(roadmap_text, "`8.7.56.1055-.1058`") is not None

    part1_has_weak_field_normalization = hit(part1_text, r"g_P/Z_P=4\pi G") is not None
    part3a_has_bridge_head = hit(part3a_text, CHECKPOINT_BRIDGE_HEAD) is not None
    part3a_has_raw_direct_si_formula = hit(part3a_text, RAW_DIRECT_SI_FORMULA) is not None
    part3a_has_dimension_vector = hit(part3a_text, DIMENSION_VECTOR) is not None
    part3a_has_dimensionless_gap = hit(part3a_text, DIMENSIONLESS_GAP) is not None
    part3a_has_explicit_bridge_gap = hit(part3a_text, EXPLICIT_BRIDGE_GAP) is not None
    part3a_has_mapping_statement_gap = hit(part3a_text, MAPPING_STATEMENT_GAP) is not None
    part5_has_reopen_next_state = hit(part5_text, REOPEN_NEXT_STATE) is not None
    part5_has_reopen_scope_blocker = hit(part5_text, REOPEN_SCOPE_BLOCKER) is not None
    em_doc_has_local_maxwell_adoption = hit(em_doc_text, LOCAL_MAXWELL_ADOPTION) is not None

    explicit_current_public_bridge_statement_available = bool(
        source_1051["explicit_current_public_bridge_statement_available"]
    ) and bool(audit_1052["explicit_current_public_bridge_statement_available"])
    numeric_reopen_scope_fixed = bool(route_1054["numeric_reopen_scope_fixed"])
    raw_final_computation_reopen_ready = bool(audit_1052["raw_final_computation_reopen_ready"])

    retained_dimensionless_alpha_bridge_active = not bool(
        dim_audit_988["explicit_dimensionless_alpha_bridge_available"]
    )
    retained_explicit_si_alpha_formula_absent = (
        not bool(dim_audit_988["explicit_si_alpha_formula_available"])
        and not bool(em_audit_992["explicit_si_alpha_formula_available"])
    )
    retained_em_unit_convention_bridge_active = not bool(
        em_audit_992["explicit_em_unit_convention_bridge_available"]
    )
    retained_gp_to_elementary_charge_mapping_absent = (
        not bool(em_audit_992["explicit_gp_to_elementary_charge_mapping_available"])
        and not bool(map_stmt_audit_996["explicit_mapping_statement_available"])
        and not bool(map_lit_audit_1000["explicit_mapping_literal_available"])
    )
    retained_mapping_literal_branch_added_no_new_surface = not bool(
        map_lit_audit_1000["new_public_canonical_surface_added_in_literal_branch"]
    )
    no_go_closeout_response_type_acceptable = "no_go_closeout" in expert_bundle_audit_1032[
        "acceptable_response_types"
    ]

    dimensionless_alpha_bridge_blocker_confirmed = (
        explicit_current_public_bridge_statement_available
        and numeric_reopen_scope_fixed
        and raw_final_computation_reopen_ready
        and retained_dimensionless_alpha_bridge_active
        and retained_explicit_si_alpha_formula_absent
        and retained_em_unit_convention_bridge_active
        and retained_gp_to_elementary_charge_mapping_absent
    )

    inventory_ready = all(
        [
            prior_dimensionless_bridge_reopen_route_active,
            status_has_1055_next_step,
            roadmap_has_1055_branch,
            part1_has_weak_field_normalization,
            part3a_has_bridge_head,
            part3a_has_raw_direct_si_formula,
            part3a_has_dimension_vector,
            part3a_has_dimensionless_gap,
            part3a_has_explicit_bridge_gap,
            part3a_has_mapping_statement_gap,
            part5_has_reopen_next_state,
            part5_has_reopen_scope_blocker,
            em_doc_has_local_maxwell_adoption,
            no_go_closeout_response_type_acceptable,
            dimensionless_alpha_bridge_blocker_confirmed,
            retained_mapping_literal_branch_added_no_new_surface,
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
        "prior_1051_json": display_path(SOURCE_1051),
        "prior_1052_json": display_path(AUDIT_1052),
        "prior_1053_json": display_path(GATE_1053),
        "prior_1054_json": display_path(ROUTE_1054),
        "retained_988_json": display_path(DIM_AUDIT_988),
        "retained_992_json": display_path(EM_AUDIT_992),
        "retained_996_json": display_path(MAP_STMT_AUDIT_996),
        "retained_1000_json": display_path(MAP_LIT_AUDIT_1000),
        "retained_1032_json": display_path(EXPERT_BUNDLE_AUDIT_1032),
    }

    inventory = payload(
        "8.7.56.1055",
        "Trial-2 numeric alpha two-sector hierarchy EM-sector normalization dimensionless-alpha-bridge reopen source inventory",
        common_inputs,
        "Freeze the reopened dimensionless-alpha-bridge pack: public bridge sentence, retained raw final-computation boundary, retained dimensionless-alpha / EM unit-convention / charge-mapping audits, and the admissible no-go-closeout response class.",
        {
            "inventory_rule": "the reopened pack is ready when the public bridge sentence and all retained dimensionless-alpha blocker surfaces are assembled together",
            "residual_rule": "if the bridge sentence is public but explicit SI alpha and explicit g_P-to-elementary-charge mapping remain absent, the honest next route contracts to current-canon no-go closeout",
        },
        [
            row(
                "trial2_numeric_alpha_dimensionless_bridge_reopen_inventory_complete",
                "pass" if inventory_ready else "reject",
                "dimensionless-alpha-bridge reopen inventory complete",
                1 if inventory_ready else 0,
                "The reopened blocker pack combines the bridge sentence, raw final-computation boundary, retained bridge-family audits, and the admissible no-go-closeout response class.",
            ),
            row(
                "trial2_numeric_alpha_public_bridge_sentence_retained_for_dimensionless_reopen",
                "pass" if explicit_current_public_bridge_statement_available else "reject",
                "public bridge sentence retained for dimensionless-alpha reopen",
                1 if explicit_current_public_bridge_statement_available else 0,
                "The upstream EM-vs-gravity normalization blocker remains retired during the reopened deeper audit.",
            ),
            row(
                "trial2_numeric_alpha_explicit_si_alpha_formula_still_absent_after_reopen",
                "pass" if retained_explicit_si_alpha_formula_absent else "reject",
                "explicit SI alpha formula still absent after reopen",
                1 if retained_explicit_si_alpha_formula_absent else 0,
                "Even after the public bridge sentence, current canon still does not provide the positive SI alpha formula that would make the direct readout dimensionless.",
            ),
            row(
                "trial2_numeric_alpha_explicit_gp_to_elementary_charge_mapping_still_absent_after_reopen",
                "pass" if retained_gp_to_elementary_charge_mapping_absent else "reject",
                "explicit g_P-to-elementary-charge mapping still absent after reopen",
                1 if retained_gp_to_elementary_charge_mapping_absent else 0,
                "The reopened pack still lacks the positive statement or literal that identifies structural e with physical elementary charge.",
            ),
        ],
        {
            "inventory_ready": inventory_ready,
            "prior_dimensionless_alpha_bridge_reopen_route_active": prior_dimensionless_bridge_reopen_route_active,
            "explicit_current_public_bridge_statement_available": explicit_current_public_bridge_statement_available,
            "numeric_reopen_scope_fixed": numeric_reopen_scope_fixed,
            "raw_final_computation_reopen_ready": raw_final_computation_reopen_ready,
            "retained_dimensionless_alpha_bridge_active": retained_dimensionless_alpha_bridge_active,
            "retained_explicit_si_alpha_formula_absent": retained_explicit_si_alpha_formula_absent,
            "retained_em_unit_convention_bridge_active": retained_em_unit_convention_bridge_active,
            "retained_gp_to_elementary_charge_mapping_absent": retained_gp_to_elementary_charge_mapping_absent,
            "retained_mapping_literal_branch_added_no_new_surface": retained_mapping_literal_branch_added_no_new_surface,
            "no_go_closeout_response_type_acceptable": no_go_closeout_response_type_acceptable,
            "first_route_to_close_or_none": NEXT_NO_GO_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_dimensionless_bridge_reopen_inventory_frozen",
            "advance_to_8_7_56_1056": inventory_ready,
            "next_required_artifacts": [NEXT_NO_GO_ROUTE],
        },
        {
            "checkpoint_and_reopen_hits": {
                "part1_weak_field_normalization": hit(part1_text, r"g_P/Z_P=4\pi G"),
                "part3a_bridge_head": hit(part3a_text, CHECKPOINT_BRIDGE_HEAD),
                "part3a_direct_si_formula": hit(part3a_text, RAW_DIRECT_SI_FORMULA),
                "part3a_dimension_vector": hit(part3a_text, DIMENSION_VECTOR),
                "part3a_dimensionless_gap": hit(part3a_text, DIMENSIONLESS_GAP),
                "part3a_explicit_bridge_gap": hit(part3a_text, EXPLICIT_BRIDGE_GAP),
                "part3a_mapping_statement_gap": hit(part3a_text, MAPPING_STATEMENT_GAP),
                "part5_reopen_next_state": hit(part5_text, REOPEN_NEXT_STATE),
                "part5_reopen_scope_blocker": hit(part5_text, REOPEN_SCOPE_BLOCKER),
                "em_doc_local_maxwell_adoption": hit(em_doc_text, LOCAL_MAXWELL_ADOPTION),
            },
            "retained_bridge_family_summary": {
                "dimensionless_bridge": dim_audit_988,
                "em_unit_bridge": em_audit_992,
                "mapping_statement": map_stmt_audit_996,
                "mapping_literal": map_lit_audit_1000,
                "expert_bundle_question_audit": expert_bundle_audit_1032,
            },
        },
    )

    audit = payload(
        "8.7.56.1056",
        "Trial-2 numeric alpha two-sector hierarchy EM-sector normalization dimensionless-alpha-bridge reopen audit",
        common_inputs,
        "Audit whether the now-public bridge sentence changes the deeper blocker family. It does not: explicit SI alpha and explicit g_P-to-elementary-charge mapping remain absent, so the reopened residual contracts to current-canon no-go closeout rather than renewed wording search.",
        {
            "audit_rule": "a reopened deeper audit closes only if public canon now contains either an explicit SI alpha formula or an explicit g_P-to-elementary-charge mapping",
            "closeout_rule": "if neither surface appears after reopen, the honest outcome is current-canon no-go closeout for this route, not numeric closeout",
        },
        [
            row(
                "trial2_numeric_alpha_dimensionless_bridge_reopen_audit_complete",
                "pass" if inventory_ready else "reject",
                "dimensionless-alpha-bridge reopen audit complete",
                1 if inventory_ready else 0,
                "The reopened bridge-family pack is audited under the public bridge sentence.",
            ),
            row(
                "trial2_numeric_alpha_public_bridge_sentence_does_not_supply_explicit_si_alpha_formula",
                "pass" if retained_explicit_si_alpha_formula_absent else "reject",
                "public bridge sentence does not supply explicit SI alpha formula",
                1 if retained_explicit_si_alpha_formula_absent else 0,
                "The new bridge sentence retires the normalization conflict but does not add the positive SI alpha identity needed for dimensionless closeout.",
            ),
            row(
                "trial2_numeric_alpha_public_bridge_sentence_does_not_supply_explicit_gp_to_elementary_charge_mapping",
                "pass" if retained_gp_to_elementary_charge_mapping_absent else "reject",
                "public bridge sentence does not supply explicit g_P-to-elementary-charge mapping",
                1 if retained_gp_to_elementary_charge_mapping_absent else 0,
                "The deeper charge-normalization bridge remains absent after reopen.",
            ),
            row(
                "trial2_numeric_alpha_current_canon_no_go_closeout_candidate_selected_after_reopen",
                "pass" if dimensionless_alpha_bridge_blocker_confirmed else "reject",
                "current-canon no-go closeout candidate selected after reopen",
                1 if dimensionless_alpha_bridge_blocker_confirmed else 0,
                "Because the public bridge sentence did not resolve the deeper bridge family, the honest next route is current-canon no-go closeout for this numeric path.",
            ),
            row(
                "trial2_numeric_alpha_numeric_closeout_ready_after_dimensionless_reopen_audit",
                "reject",
                "numeric closeout ready after dimensionless-alpha reopen audit",
                0,
                "The reopened branch confirms blocker persistence, not numeric alpha closeout.",
            ),
        ],
        {
            "audit_ready": inventory_ready,
            "selected_dimensionless_alpha_bridge_reopen_class": (
                "public_bridge_sentence_present_but_dimensionless_alpha_bridge_still_absent"
                if dimensionless_alpha_bridge_blocker_confirmed
                else "dimensionless_alpha_bridge_reopen_incomplete"
            ),
            "explicit_current_public_bridge_statement_available": explicit_current_public_bridge_statement_available,
            "raw_final_computation_reopen_ready": raw_final_computation_reopen_ready,
            "explicit_si_alpha_formula_available": False,
            "explicit_gp_to_elementary_charge_mapping_available": False,
            "explicit_dimensionless_alpha_bridge_available": False,
            "dimensionless_alpha_bridge_blocker_confirmed": dimensionless_alpha_bridge_blocker_confirmed,
            "current_canon_no_go_closeout_candidate_selected": dimensionless_alpha_bridge_blocker_confirmed,
            "numeric_closeout_ready": False,
            "first_route_to_close_after_audit_or_none": NEXT_NO_GO_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_dimensionless_bridge_reopen_audited",
            "advance_to_8_7_56_1057": True,
            "next_required_artifacts": [NEXT_NO_GO_ROUTE],
        },
        {
            "prior_numeric_reopen_summary": {
                "source": source_1051,
                "audit": audit_1052,
                "gate": gate_1053,
                "route": route_1054,
            },
            "retained_blocker_summary": {
                "dimensionless_bridge": dim_audit_988,
                "em_unit_bridge": em_audit_992,
                "mapping_statement": map_stmt_audit_996,
                "mapping_literal": map_lit_audit_1000,
            },
        },
    )

    gate = payload(
        "8.7.56.1057",
        "Trial-2 numeric alpha two-sector hierarchy EM-sector normalization dimensionless-alpha-bridge reopen declaration gate",
        common_inputs,
        "Update the official gate after the reopened deeper audit: the blocker family is confirmed under current canon, numeric closeout remains unavailable, and the next residual route becomes current-canon no-go closeout.",
        {
            "gate_rule": "a reopened deeper blocker can close the reopen branch even when numeric closeout remains unavailable",
            "residual_rule": "once the blocker is confirmed under the public bridge sentence, the next route is current-canon no-go closeout rather than more wording descent",
        },
        [
            row(
                "trial2_numeric_alpha_dimensionless_bridge_reopen_gate_complete",
                "pass",
                "dimensionless-alpha-bridge reopen gate complete",
                1,
                "The official gate is updated after the reopened deeper audit.",
            ),
            row(
                "trial2_numeric_alpha_dimensionless_bridge_blocker_confirmed_under_current_canon",
                "pass" if dimensionless_alpha_bridge_blocker_confirmed else "reject",
                "dimensionless-alpha bridge blocker confirmed under current canon",
                1 if dimensionless_alpha_bridge_blocker_confirmed else 0,
                "The public bridge sentence does not eliminate the explicit SI alpha / explicit g_P-to-e mapping gap.",
            ),
            row(
                "trial2_numeric_alpha_no_go_closeout_candidate_selected",
                "pass" if dimensionless_alpha_bridge_blocker_confirmed else "reject",
                "no-go closeout candidate selected",
                1 if dimensionless_alpha_bridge_blocker_confirmed else 0,
                "The next official route is the current-canon no-go closeout branch for this reopened numeric path.",
            ),
            row(
                "trial2_numeric_alpha_closeout_still_not_ready_after_dimensionless_reopen_gate",
                "reject",
                "closeout still not ready after dimensionless-alpha reopen gate",
                0,
                "The reopened deeper audit confirms blocker persistence rather than full numeric alpha closeout.",
            ),
        ],
        {
            "trial2_numeric_alpha_problem_classification": "current_canon_no_go_closeout_candidate",
            "trial2_numeric_alpha_text_search_continuation_justified": False,
            "trial2_numeric_alpha_mechanical_wording_descent_stopped": True,
            "trial2_numeric_alpha_current_canon_bridge_statement_checkpoint_wording_promotion_completed": True,
            "trial2_numeric_alpha_raw_final_computation_reopen_ready": raw_final_computation_reopen_ready,
            "trial2_numeric_alpha_dimensionless_alpha_bridge_reopen_completed": True,
            "trial2_numeric_alpha_dimensionless_alpha_bridge_blocker_confirmed": dimensionless_alpha_bridge_blocker_confirmed,
            "trial2_numeric_alpha_explicit_si_alpha_formula_available": False,
            "trial2_numeric_alpha_explicit_gp_to_elementary_charge_mapping_available": False,
            "trial2_numeric_alpha_numeric_from_current_pack_ready": False,
            "trial2_numeric_alpha_closeout_ready": False,
            "trial2_numeric_alpha_no_go_closeout_candidate_selected": dimensionless_alpha_bridge_blocker_confirmed,
            "selected_residual_route": NEXT_NO_GO_ROUTE,
            "missing_v2_artifact": NEXT_NO_GO_ARTIFACT,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_dimensionless_bridge_reopen_gate_closed",
            "advance_to_8_7_56_1058": True,
            "next_required_artifacts": [NEXT_NO_GO_ROUTE],
        },
        {
            "audit_summary": audit["summary"],
            "question_audit_summary": expert_bundle_audit_1032,
        },
    )

    route = payload(
        "8.7.56.1058",
        "Trial-2 numeric alpha route contract one-hundred-sixty-first refresh",
        common_inputs,
        "Refresh the next-generation contract after the reopened deeper audit: keep the precision-alpha mainline frozen, keep strong-side work on reserve, and advance to current-canon no-go closeout for the unresolved dimensionless-alpha bridge family.",
        {
            "next_route_rule": "the next route formalizes current-canon no-go closeout for the reopened dimensionless-alpha bridge family",
            "reserve_rule": "strong-side non-Abelian, running, and confinement remain on v3 hold reserve",
        },
        [
            row(
                "trial2_numeric_alpha_route_contract_one_hundred_sixty_first_refresh_complete",
                "pass",
                "route contract one-hundred-sixty-first refresh complete",
                1,
                "The reopened deeper-audit gate is converted into the next-generation contract.",
            ),
            row(
                "trial2_numeric_alpha_next_route_selected_as_current_canon_no_go_closeout",
                "pass" if dimensionless_alpha_bridge_blocker_confirmed else "reject",
                "next route selected as current-canon no-go closeout",
                1 if dimensionless_alpha_bridge_blocker_confirmed else 0,
                "The next official branch formalizes blocker-confirmed no-go closeout for the reopened numeric path.",
            ),
            row(
                "trial2_numeric_alpha_external_dependency_remains_retired_after_dimensionless_reopen",
                "pass",
                "external dependency remains retired after dimensionless-alpha reopen",
                1,
                "The mainline stays independent of outside input after the public bridge sentence and reopened deeper audit.",
            ),
            row(
                "trial2_numeric_alpha_precision_mainline_retained_after_dimensionless_reopen",
                "pass" if bool(route_1054.get("precision_alpha_mainline_retained", False)) else "reject",
                "precision-alpha mainline retained after dimensionless-alpha reopen",
                1 if bool(route_1054.get("precision_alpha_mainline_retained", False)) else 0,
                "Trial-2 numeric alpha remains the precision mainline even though the current-canon route is moving toward no-go closeout.",
            ),
        ],
        {
            "selected_next_generation_route": NEXT_NO_GO_ROUTE,
            "strong_side_route_state": route_1054.get("strong_side_route_state"),
            "precision_alpha_mainline_retained": bool(route_1054.get("precision_alpha_mainline_retained", False)),
            "electron_identification_pivot_retained": bool(route_1054.get("electron_identification_pivot_retained", False)),
            "h0p_bridge_pivot_retained": bool(route_1054.get("h0p_bridge_pivot_retained", False)),
            "final_computation_branch_retained": True,
            "unit_consistency_audit_branch_retained": True,
            "dimensionless_alpha_bridge_branch_retained": True,
            "em_unit_convention_bridge_branch_retained": True,
            "mapping_statement_branch_retained": True,
            "mapping_literal_branch_retained": True,
            "two_sector_hierarchy_pivot_retained": True,
            "current_canon_bridge_statement_checkpoint_wording_promotion_completed": True,
            "numeric_reopen_scope_fixed": numeric_reopen_scope_fixed,
            "dimensionless_alpha_bridge_reopen_completed": True,
            "current_canon_no_go_closeout_candidate_selected": dimensionless_alpha_bridge_blocker_confirmed,
            "external_dependency_active": False,
            "hard_conflict_reading_retired": True,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_route_contract_one_hundred_sixty_first_refresh_frozen",
            "advance_to_next_route": True,
            "next_required_artifacts": [NEXT_NO_GO_ROUTE],
        },
        {
            "gate_summary": gate["summary"],
            "prior_route_summary": route_1054,
        },
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
        "dimensionless_alpha_bridge_reopen_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
        "dimensionless_alpha_bridge_reopen_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
        "dimensionless_alpha_bridge_reopen_declaration_gate",
        gate,
    )
    write_artifact("mass_origin_v2_t2_alpha_route_contract_one_hundred_sixty_first_refresh", route)

    print("[done] 8.7.56.1055-.1058 artifacts generated:")
    print(
        " - mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
        "dimensionless_alpha_bridge_reopen_source_inventory_metrics.json"
    )
    print(
        " - mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
        "dimensionless_alpha_bridge_reopen_audit_metrics.json"
    )
    print(
        " - mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_"
        "dimensionless_alpha_bridge_reopen_declaration_gate_metrics.json"
    )
    print(" - mass_origin_v2_t2_alpha_route_contract_one_hundred_sixty_first_refresh_metrics.json")


# Function: run the dimensionless-alpha-bridge reopen branch from the CLI.

def run_cli() -> None:
    """CLI entry point for the Trial-2 numeric alpha dimensionless-alpha-bridge reopen branch."""
    main()


if __name__ == "__main__":
    run_cli()
