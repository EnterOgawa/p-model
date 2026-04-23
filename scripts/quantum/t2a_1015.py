#!/usr/bin/env python3
"""Generate 8.7.56.1015-.1018 Trial-2 numeric alpha two-sector hierarchy statement artifacts."""

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

SOURCE_1011 = OUT / "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_source_inventory_metrics.json"
AUDIT_1012 = OUT / "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_audit_metrics.json"
GATE_1013 = OUT / "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_declaration_gate_metrics.json"
ROUTE_1014 = OUT / "mass_origin_v2_t2_alpha_route_contract_one_hundred_fiftieth_refresh_metrics.json"

CURRENT_ROUTE = "trial2_numeric_alpha_two_sector_hierarchy_statement"
NEXT_RESIDUAL_ROUTE = "trial2_numeric_alpha_two_sector_hierarchy_em_sector_normalization_statement"
NEXT_ROUTE = "8.7.56.1019"


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


# Function: execute the two-sector hierarchy statement branch.

def main() -> None:
    """Execute the Trial-2 numeric alpha two-sector hierarchy statement branch."""
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
        SOURCE_1011,
        AUDIT_1012,
        GATE_1013,
        ROUTE_1014,
    ):
        require(path)

    expert_note_text = read_text(EXPERT_NOTE)
    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    em_doc_text = read_text(EM_DOC)
    ai_context = read_json(AI_CONTEXT)
    source_1011 = read_json(SOURCE_1011)["summary"]
    audit_1012 = read_json(AUDIT_1012)["summary"]
    gate_1013 = read_json(GATE_1013)["summary"]
    route_1014 = read_json(ROUTE_1014)["summary"]

    prior_two_sector_pivot_active = (
        gate_1013["selected_residual_route"] == CURRENT_ROUTE
        and gate_1013["missing_v2_artifact"] == CURRENT_ROUTE
        and bool(gate_1013["trial2_numeric_alpha_two_sector_hierarchy_pivot_active"])
        and route_1014["selected_next_generation_route"] == CURRENT_ROUTE
    )

    note_has_two_sector_table = hit(expert_note_text, "2つの sector の kinetic 係数は異なる") is not None
    note_has_em_sector_normalization = hit(expert_note_text, r"Z_P^{\rm EM} = 1") is not None
    note_has_newton_bridge = hit(expert_note_text, r"\frac{g_P\,v}{M_\chi^2} = \frac{4\pi G}{c^2}") is not None
    note_has_em_bridge = hit(expert_note_text, r"e = g_P / \sqrt{Z_P^{\rm EM}} = g_P") is not None

    part1_has_gravity_kinetic_surface = hit(part1_text, r"\frac{M_\chi^2}{2}\partial_\mu\chi") is not None
    part1_has_vector_free_surface = hit(part1_text, r"-\frac{1}{4}F^{(P)}_{\mu\nu}F_{(P)}^{\mu\nu}") is not None
    part1_has_photon_zp_extraction = hit(part1_text, r"A_\mu=\delta P_\mu^T/\sqrt{Z_P}") is not None
    part1_has_vector_zp_surface = hit(part1_text, r"-\frac{Z_{P}}{4}F^{(P)}_{\mu\nu}F_{(P)}^{\mu\nu}") is not None
    part1_has_current_newton_surface = hit(part1_text, r"g_P/Z_P=4\pi G") is not None

    part3a_has_current_computation_formula = hit(
        part3a_text, r"\alpha=16\pi G^2\lambda v^2/(m_0^2\hbar c)"
    ) is not None
    part3a_has_two_sector_wording = hit(part3a_text, "two-sector hierarchy") is not None
    part5_has_two_sector_wording = hit(part5_text, "two-sector hierarchy pivot") is not None
    status_has_1015_next_step = hit(status_text, "8.7.56.1015") is not None
    roadmap_has_1015_branch = hit(roadmap_text, "`8.7.56.1015-.1018`") is not None
    em_doc_has_coulomb_surface = hit(em_doc_text, r"\Phi(r)=\frac{1}{4\pi\varepsilon_0}\frac{q}{r}") is not None

    public_two_sector_hierarchy_statement_available = any(
        candidate is not None
        for candidate in (
            hit(part1_text, "gravity と EM が異なる kinetic normalization"),
            hit(part3a_text, "gravity と EM が異なる kinetic normalization"),
            hit(part5_text, "gravity と EM が異なる kinetic normalization"),
            hit(part1_text, "two-sector hierarchy"),
            hit(part3a_text, "two-sector hierarchy"),
            hit(part5_text, "two-sector hierarchy"),
        )
    )
    public_em_sector_normalization_statement_available = any(
        candidate is not None
        for candidate in (
            hit(part1_text, r"Z_P^{\rm EM} = 1"),
            hit(part3a_text, r"Z_P^{\rm EM} = 1"),
            hit(part5_text, r"Z_P^{\rm EM} = 1"),
            hit(part1_text, r"e = g_P"),
            hit(part3a_text, r"e = g_P"),
            hit(part5_text, r"e = g_P"),
        )
    )
    public_newton_mchi_bridge_statement_available = any(
        candidate is not None
        for candidate in (
            hit(part1_text, r"\frac{g_P\,v}{M_\chi^2} = \frac{4\pi G}{c^2}"),
            hit(part3a_text, r"\frac{g_P\,v}{M_\chi^2} = \frac{4\pi G}{c^2}"),
            hit(part5_text, r"\frac{g_P\,v}{M_\chi^2} = \frac{4\pi G}{c^2}"),
        )
    )
    public_mchi_frozen_value_available = any(
        candidate is not None
        for candidate in (
            hit(part1_text, r"M_\chi = M_{\rm Pl}"),
            hit(part3a_text, r"M_\chi = M_{\rm Pl}"),
            hit(part5_text, r"M_\chi = M_{\rm Pl}"),
            hit(part1_text, r"M_\chi \sim M_{\rm Pl}"),
            hit(part3a_text, r"M_\chi \sim M_{\rm Pl}"),
            hit(part5_text, r"M_\chi \sim M_{\rm Pl}"),
        )
    )

    inventory_ready = all(
        [
            prior_two_sector_pivot_active,
            note_has_two_sector_table,
            note_has_em_sector_normalization,
            note_has_newton_bridge,
            note_has_em_bridge,
            part1_has_gravity_kinetic_surface,
            part1_has_vector_free_surface,
            part1_has_photon_zp_extraction,
            part1_has_vector_zp_surface,
            part1_has_current_newton_surface,
            part3a_has_current_computation_formula,
            part5_has_two_sector_wording,
            status_has_1015_next_step,
            roadmap_has_1015_branch,
            em_doc_has_coulomb_surface,
        ]
    )

    dominant_blocker_is_missing_em_sector_normalization_statement = (
        not public_em_sector_normalization_statement_available
        and not public_newton_mchi_bridge_statement_available
        and not public_mchi_frozen_value_available
        and part1_has_photon_zp_extraction
        and part1_has_vector_zp_surface
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
        "prior_1011_json": display_path(SOURCE_1011),
        "prior_1012_json": display_path(AUDIT_1012),
        "prior_1013_json": display_path(GATE_1013),
        "prior_1014_json": display_path(ROUTE_1014),
    }

    inventory = payload(
        "8.7.56.1015",
        "Trial-2 numeric alpha two-sector hierarchy statement source inventory",
        common_inputs,
        "Freeze the mechanism-level statement pack for the two-sector hierarchy pivot: note surfaces, current canon surfaces, and the current route contract.",
        {
            "inventory_rule": "the statement pack is ready when the note, the current canon kinetic surfaces, and the current route contract are assembled together",
            "statement_rule": "the generic two-sector hierarchy statement is audited before splitting into lower-level EM-normalization or Newton-M_chi subroutes",
        },
        [
            row(
                "trial2_numeric_alpha_two_sector_hierarchy_statement_inventory_complete",
                "pass" if inventory_ready else "reject",
                "two-sector hierarchy statement inventory complete",
                1 if inventory_ready else 0,
                "The mechanism-level statement branch can only be audited after the note and the current canon surfaces are assembled together.",
            ),
            row(
                "trial2_numeric_alpha_note_contains_em_sector_normalization_claim",
                "pass" if note_has_em_sector_normalization else "reject",
                "note contains EM-sector normalization claim",
                1 if note_has_em_sector_normalization else 0,
                "The note explicitly proposes Z_P^EM = 1, making this a mechanism-level statement branch rather than a generic wording loop.",
            ),
            row(
                "trial2_numeric_alpha_current_canon_contains_single_zp_photon_surface",
                "pass" if part1_has_photon_zp_extraction and part1_has_vector_zp_surface else "reject",
                "current canon contains single-ZP photon surface",
                1 if part1_has_photon_zp_extraction and part1_has_vector_zp_surface else 0,
                "The first audit target is the conflict between the note's EM split and the current single-ZP photon normalization surfaces.",
            ),
            row(
                "trial2_numeric_alpha_public_two_sector_hierarchy_statement_surface_check",
                "pass" if public_two_sector_hierarchy_statement_available else "reject",
                "public two-sector hierarchy statement surface check",
                1 if public_two_sector_hierarchy_statement_available else 0,
                "Current checkpoint wording now surfaces the generic two-sector hierarchy, so the branch can shrink to the first concrete missing substatement."
                if public_two_sector_hierarchy_statement_available
                else "Current canon still lacks an explicit mechanism-level statement adopting the split hierarchy.",
            ),
        ],
        {
            "inventory_ready": inventory_ready,
            "two_sector_hierarchy_pivot_active": bool(gate_1013["trial2_numeric_alpha_two_sector_hierarchy_pivot_active"]),
            "public_two_sector_hierarchy_statement_available": public_two_sector_hierarchy_statement_available,
            "public_em_sector_normalization_statement_available": public_em_sector_normalization_statement_available,
            "public_newton_mchi_bridge_statement_available": public_newton_mchi_bridge_statement_available,
            "public_mchi_frozen_value_available": public_mchi_frozen_value_available,
            "first_route_to_close_or_none": NEXT_RESIDUAL_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_two_sector_hierarchy_statement_inventory_frozen",
            "advance_to_8_7_56_1016": inventory_ready,
            "next_required_artifacts": [NEXT_RESIDUAL_ROUTE],
        },
        {
            "prior_pivot_summary": gate_1013,
            "note_hits": {
                "em_sector_normalization": hit(expert_note_text, r"Z_P^{\rm EM} = 1"),
                "newton_bridge": hit(expert_note_text, r"\frac{g_P\,v}{M_\chi^2} = \frac{4\pi G}{c^2}"),
            },
        },
    )

    audit = payload(
        "8.7.56.1016",
        "Trial-2 numeric alpha two-sector hierarchy statement audit",
        common_inputs,
        "Audit whether the mechanism-level two-sector hierarchy statement already exists in current canon, or whether the branch shrinks to the first missing EM-sector normalization statement.",
        {
            "audit_rule": "if the generic mechanism statement is absent, the branch must shrink to the first concrete public statement that would make the split operational",
            "shrink_rule": "the first blocker is the EM-sector normalization statement because it directly conflicts with the current A_mu = delta P_mu^T / sqrt(Z_P) and -Z_P F^2 / 4 surfaces",
        },
        [
            row(
                "trial2_numeric_alpha_two_sector_hierarchy_statement_audit_complete",
                "pass" if inventory_ready else "reject",
                "two-sector hierarchy statement audit complete",
                1 if inventory_ready else 0,
                "The mechanism statement is audited against current canon rather than accepted from the note alone.",
            ),
            row(
                "trial2_numeric_alpha_public_two_sector_hierarchy_statement_available_after_statement_audit",
                "pass" if public_two_sector_hierarchy_statement_available else "reject",
                "public two-sector hierarchy statement available after statement audit",
                1 if public_two_sector_hierarchy_statement_available else 0,
                "Current checkpoint wording already carries the generic split statement, so the blocker can shrink to the first concrete EM-side statement."
                if public_two_sector_hierarchy_statement_available
                else "Current canon still lacks an explicit public statement that gravity and EM use distinct kinetic normalizations in the Trial-2 route.",
            ),
            row(
                "trial2_numeric_alpha_public_em_sector_normalization_statement_available_after_statement_audit",
                "pass" if public_em_sector_normalization_statement_available else "reject",
                "public EM-sector normalization statement available after statement audit",
                1 if public_em_sector_normalization_statement_available else 0,
                "Current canon does not yet contain a public statement equivalent to Z_P^EM = 1 or e = g_P for the photon branch.",
            ),
            row(
                "trial2_numeric_alpha_dominant_blocker_is_missing_em_sector_normalization_statement",
                "pass" if dominant_blocker_is_missing_em_sector_normalization_statement else "reject",
                "dominant blocker is missing EM-sector normalization statement",
                1 if dominant_blocker_is_missing_em_sector_normalization_statement else 0,
                "The first concrete blocker is the EM-sector statement because it is the earliest point where the note diverges from the current single-ZP photon canon.",
            ),
        ],
        {
            "audit_ready": inventory_ready,
            "public_two_sector_hierarchy_statement_available": public_two_sector_hierarchy_statement_available,
            "public_em_sector_normalization_statement_available": public_em_sector_normalization_statement_available,
            "public_newton_mchi_bridge_statement_available": public_newton_mchi_bridge_statement_available,
            "public_mchi_frozen_value_available": public_mchi_frozen_value_available,
            "dominant_blocker_is_missing_em_sector_normalization_statement": dominant_blocker_is_missing_em_sector_normalization_statement,
            "first_route_to_close_after_audit_or_none": NEXT_RESIDUAL_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_two_sector_hierarchy_statement_audited",
            "advance_to_8_7_56_1017": True,
            "next_required_artifacts": [NEXT_RESIDUAL_ROUTE],
        },
        {
            "part1_hits": {
                "photon_zp_extraction": hit(part1_text, r"A_\mu=\delta P_\mu^T/\sqrt{Z_P}"),
                "vector_zp_surface": hit(part1_text, r"-\frac{Z_{P}}{4}F^{(P)}_{\mu\nu}F_{(P)}^{\mu\nu}"),
                "vector_free_surface": hit(part1_text, r"-\frac{1}{4}F^{(P)}_{\mu\nu}F_{(P)}^{\mu\nu}"),
            },
            "note_hits": {
                "em_sector_normalization": hit(expert_note_text, r"Z_P^{\rm EM} = 1"),
                "em_bridge": hit(expert_note_text, r"e = g_P / \sqrt{Z_P^{\rm EM}} = g_P"),
                "planck_assumption": hit(expert_note_text, r"M_\chi \sim M_{\rm Pl}"),
            },
        },
    )

    gate = payload(
        "8.7.56.1017",
        "Trial-2 numeric alpha two-sector hierarchy statement declaration gate",
        common_inputs,
        "Update the official gate after the statement audit: the generic mechanism statement shrinks to the first missing EM-sector normalization statement while the pivot remains active.",
        {
            "gate_rule": "the pivot remains live even when the generic mechanism statement shrinks to a more concrete first missing public statement",
            "blocker_rule": "the first missing public artifact is now the EM-sector normalization statement inside the two-sector hierarchy route",
        },
        [
            row(
                "trial2_numeric_alpha_two_sector_hierarchy_statement_gate_complete",
                "pass",
                "two-sector hierarchy statement declaration gate complete",
                1,
                "The official state is updated after the statement audit.",
            ),
            row(
                "trial2_numeric_alpha_two_sector_hierarchy_pivot_still_active_after_statement_audit",
                "pass" if gate_1013["trial2_numeric_alpha_two_sector_hierarchy_pivot_active"] else "reject",
                "two-sector hierarchy pivot still active after statement audit",
                1 if gate_1013["trial2_numeric_alpha_two_sector_hierarchy_pivot_active"] else 0,
                "The pivot stays live while the first concrete EM-sector statement is audited next.",
            ),
            row(
                "trial2_numeric_alpha_selected_residual_route_is_em_sector_normalization_statement",
                "pass" if dominant_blocker_is_missing_em_sector_normalization_statement else "reject",
                "selected residual route is EM-sector normalization statement",
                1 if dominant_blocker_is_missing_em_sector_normalization_statement else 0,
                "The branch now points to the first concrete public statement required to make the split operational.",
            ),
            row(
                "trial2_numeric_alpha_current_pack_still_not_closeout_ready_after_statement_audit",
                "reject",
                "current pack still not closeout ready after statement audit",
                0,
                "Even after the mechanism audit, Trial-2 still lacks the public surface needed for an honest numerical closeout.",
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
            "trial2_numeric_alpha_two_sector_hierarchy_pivot_active": True,
            "selected_residual_route": NEXT_RESIDUAL_ROUTE,
            "missing_v2_artifact": NEXT_RESIDUAL_ROUTE,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_two_sector_hierarchy_statement_gate_closed",
            "advance_to_8_7_56_1018": True,
            "next_required_artifacts": [NEXT_RESIDUAL_ROUTE],
        },
        {
            "audit_summary": audit["summary"],
            "prior_gate_summary": gate_1013,
            "ai_context_prior_step": ai_context["current_step"],
        },
    )

    route = payload(
        "8.7.56.1018",
        "Trial-2 numeric alpha route contract one-hundred-fifty-first refresh",
        common_inputs,
        "Refresh the next-generation contract after the two-sector hierarchy statement audit: keep Trial-2 on the precision mainline and advance to the EM-sector normalization statement branch.",
        {
            "next_route_rule": "the next route is the EM-sector normalization statement branch inside the two-sector hierarchy pivot",
            "reserve_rule": "strong-side non-Abelian, running, and confinement remain on v3 hold reserve",
        },
        [
            row(
                "trial2_numeric_alpha_route_contract_one_hundred_fifty_first_refresh_complete",
                "pass",
                "route contract one-hundred-fifty-first refresh complete",
                1,
                "The statement audit is converted into the next-generation contract.",
            ),
            row(
                "trial2_numeric_alpha_next_route_selected_as_em_sector_normalization_statement",
                "pass" if dominant_blocker_is_missing_em_sector_normalization_statement else "reject",
                "next route selected as EM-sector normalization statement",
                1 if dominant_blocker_is_missing_em_sector_normalization_statement else 0,
                "The next official branch is the first concrete substatement inside the two-sector hierarchy route.",
            ),
            row(
                "trial2_numeric_alpha_precision_mainline_retained_after_statement_audit",
                "pass" if route_1014["precision_alpha_mainline_retained"] else "reject",
                "precision-alpha mainline retained after statement audit",
                1 if route_1014["precision_alpha_mainline_retained"] else 0,
                "Trial-2 numeric alpha remains the precision mainline despite the mechanism-level shrink.",
            ),
            row(
                "trial2_numeric_alpha_two_sector_hierarchy_pivot_retained_after_statement_audit",
                "pass" if route_1014["two_sector_hierarchy_pivot_retained"] else "reject",
                "two-sector hierarchy pivot retained after statement audit",
                1 if route_1014["two_sector_hierarchy_pivot_retained"] else 0,
                "The pivot remains active while the EM-sector normalization statement is audited next.",
            ),
        ],
        {
            "selected_next_generation_route": NEXT_RESIDUAL_ROUTE,
            "strong_side_route_state": route_1014["strong_side_route_state"],
            "precision_alpha_mainline_retained": bool(route_1014["precision_alpha_mainline_retained"]),
            "electron_identification_pivot_retained": bool(route_1014["electron_identification_pivot_retained"]),
            "h0p_bridge_pivot_retained": bool(route_1014["h0p_bridge_pivot_retained"]),
            "final_computation_branch_retained": bool(route_1014["final_computation_branch_retained"]),
            "unit_consistency_audit_branch_retained": bool(route_1014["unit_consistency_audit_branch_retained"]),
            "dimensionless_alpha_bridge_branch_retained": bool(
                route_1014["dimensionless_alpha_bridge_branch_retained"]
            ),
            "em_unit_convention_bridge_branch_retained": bool(
                route_1014["em_unit_convention_bridge_branch_retained"]
            ),
            "mapping_statement_branch_retained": bool(route_1014["mapping_statement_branch_retained"]),
            "mapping_literal_branch_retained": bool(route_1014["mapping_literal_branch_retained"]),
            "expert_advice_escalation_branch_retained": bool(route_1014["expert_advice_escalation_branch_retained"]),
            "two_sector_hierarchy_pivot_retained": bool(route_1014["two_sector_hierarchy_pivot_retained"]),
            "external_dependency_active": bool(route_1014["external_dependency_active"]),
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_route_contract_one_hundred_fifty_first_refresh_frozen",
            "advance_to_next_route": True,
            "next_required_artifacts": [NEXT_RESIDUAL_ROUTE],
        },
        {
            "gate_summary": gate["summary"],
            "prior_route_summary": route_1014,
        },
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_statement_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_statement_audit",
        audit,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_statement_declaration_gate",
        gate,
    )
    write_artifact(
        "mass_origin_v2_t2_alpha_route_contract_one_hundred_fifty_first_refresh",
        route,
    )

    print("[done] 8.7.56.1015-.1018 artifacts generated:")
    print(" - mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_statement_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_statement_audit_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_two_sector_hierarchy_statement_declaration_gate_metrics.json")
    print(" - mass_origin_v2_t2_alpha_route_contract_one_hundred_fifty_first_refresh_metrics.json")


# Function: run the two-sector hierarchy statement branch from the CLI.

def run_cli() -> None:
    """CLI entry point for the Trial-2 numeric alpha two-sector hierarchy statement branch."""
    main()


if __name__ == "__main__":
    run_cli()
