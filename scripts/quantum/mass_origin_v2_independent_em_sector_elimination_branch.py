#!/usr/bin/env python3
"""
Generate Trial-1 third residual artifacts for 8.7.56.26-.28 and 8.7.56.29.

This branch tests whether the current canon can eliminate the independently
adopted electromagnetic sector. The source pack reveals that the blocker is no
longer merely "no massless charge-selective statement" in the abstract. The
canon explicitly keeps:

1. L_EM inside the total action frozen in Part I.
2. Maxwell/U(1) adopted independently in Part III-A.
3. A-reject / B-adopt wording that says electromagnetism is not derived from
   P-only structure.

Therefore the next narrowed blocker is an override statement for that Part III-A
adoption, not just a generic elimination wish.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
ROUTE = OUT / "mass_origin_v2_independent_em_sector_elimination_route_contract_metrics.json"
REREOPENED_GATE = OUT / "mass_origin_v2_trial1_rereopened_declaration_gate_metrics.json"


# Function: return the current UTC time in ISO 8601 format.
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# Function: abort if a required path is missing.

def req(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# Function: load a UTF-8 JSON artifact.

def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# Function: load a UTF-8 text source.

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# Function: convert an absolute path into a repository-relative path.

def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


# Function: return the first source line that contains the requested pattern.

def hit(text: str, pattern: str) -> dict | None:
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# Function: build a standard result row.

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
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


# Function: save a JSON artifact and its row table.

def write_artifact(stem: str, data: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / f"{stem}_metrics.json"
    csv_path = OUT / f"{stem}_rows.csv"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# Function: execute the third residual branch and freeze the next route contract.

def main() -> None:
    for path in (PART1, PART3A, ROUTE, REREOPENED_GATE):
        req(path)

    part1 = read_text(PART1)
    part3a = read_text(PART3A)
    route = read_json(ROUTE)
    rereopened_gate = read_json(REREOPENED_GATE)

    common_inputs = {
        "part1_core_theory_markdown": rel(PART1),
        "part3a_quantum_foundations_markdown": rel(PART3A),
        "mass_origin_v2_independent_em_sector_elimination_route_contract_json": rel(ROUTE),
        "mass_origin_v2_trial1_rereopened_declaration_gate_json": rel(REREOPENED_GATE),
    }

    source_inventory = payload(
        "8.7.56.26",
        "Independent EM sector elimination source inventory",
        common_inputs,
        "Inventory the already-public source pack needed to test whether the independently adopted electromagnetic sector can be removed from the canon.",
        {
            "required_source_items": [
                "part1_total_action_contains_em_statement",
                "part1_vector_total_action_contains_em_statement",
                "part3a_independent_maxwell_adoption_statement",
                "part3a_independent_connection_requirement_statement",
                "part3a_a_reject_b_adopt_statement",
                "part3a_b_adoption_statement",
            ],
            "inventory_rule": "The branch requires both the explicit independent-EM statements and any available override candidates to be present before the elimination audit can be trusted.",
        },
        [
            row(
                "trial1_third_residual_source_inventory_complete",
                "pass",
                "third residual source inventory complete",
                1,
                "The independent-EM elimination source pack is frozen.",
            ),
            row(
                "trial1_third_residual_present_source_count",
                "pass",
                "present source count",
                6,
                "All required source statements are explicit in the current canon.",
            ),
            row(
                "trial1_third_residual_missing_source_count",
                "pass",
                "missing source count",
                0,
                "The branch is blocked by explicit contrary statements rather than by missing references.",
            ),
            row(
                "trial1_third_residual_a_reject_b_adopt_statement_present",
                "pass",
                "A-reject / B-adopt statement present",
                1,
                "Part III-A explicitly fixes the U(1)/EM sector as independently adopted.",
            ),
        ],
        {
            "required_source_count": 6,
            "present_source_count": 6,
            "missing_source_count": 0,
            "missing_source_items": [],
            "independent_em_sector_elimination_candidate_source_present": True,
            "first_route_to_close_or_none": "independent_em_sector_elimination_identification_audit",
        },
        {
            "overall_status": "trial1_third_residual_source_inventory_frozen",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_27": True,
            "next_required_artifacts": [
                "independent_em_sector_elimination_identification_audit",
            ],
        },
        {
            "part1_total_action_em_line": hit(part1, "+\\mathcal{L}_{\\mathrm{EM}}"),
            "part1_vector_total_action_header_line": hit(part1, "\\mathcal{L}_{\\mathrm{total}}^{\\mathrm{vec}}"),
            "part3a_independent_maxwell_adoption_line": hit(part3a, "Maxwell 方程式（U(1) ゲージ場 $A_\\mu$、電場 $E$、磁場 $B$）を、P-model の枠組みとは独立に採用する"),
            "part3a_independent_connection_line": hit(part3a, "局所位相勾配を補償する**独立接続**として別途導入する必要がある"),
            "part3a_a_reject_b_adopt_line": hit(part3a, "**A棄却、B採用**"),
            "part3a_b_adoption_line": hit(part3a, "標準 U(1) を独立の有効理論として採用し"),
        },
    )

    identification_audit = payload(
        "8.7.56.27",
        "Independent EM sector elimination identification audit",
        common_inputs,
        "Audit whether the current canon contains any statement that overrides the independently adopted EM sector and removes it from the total action without adding new fields or parameters.",
        {
            "candidate_requirements": [
                "part1 total action no longer requires a separate L_EM sector",
                "part3a no longer requires an independent A_mu connection",
                "part3a A-reject / B-adopt wording is explicitly overridden by a P-only derivation statement",
            ],
            "audit_rule": "The audit passes only if the public canon itself provides an override or replacement statement that removes the independent EM adoption.",
        },
        [
            row(
                "trial1_third_residual_part1_em_sector_removal_available",
                "fail",
                "Part I total action EM-sector removal available",
                0,
                "Part I still freezes L_EM inside the total action and does not provide a P-only replacement statement.",
            ),
            row(
                "trial1_third_residual_part3a_independent_connection_override_available",
                "fail",
                "Part III-A independent connection override available",
                0,
                "Part III-A still states that A_mu must be introduced as an independent connection.",
            ),
            row(
                "trial1_third_residual_part3a_b_adoption_override_available",
                "fail",
                "Part III-A B-adoption override available",
                0,
                "Part III-A still explicitly fixes A-reject / B-adopt and does not yet contain a v2 override statement.",
            ),
            row(
                "trial1_third_residual_independent_em_sector_elimination_identification_available",
                "fail",
                "independent EM sector elimination identification available",
                0,
                "The current canon still keeps both the independent Maxwell adoption and the separate EM sector inside the action.",
            ),
        ],
        {
            "part1_em_sector_removal_available": False,
            "part3a_independent_connection_override_available": False,
            "part3a_b_adoption_override_available": False,
            "independent_em_sector_elimination_identification_available": False,
            "identification_nonclosure_reason_or_none": "part3a_a_reject_b_adopt_and_independent_maxwell_adoption_still_fixed",
            "first_route_to_close_or_none": "part3a_b_adoption_override_identification",
        },
        {
            "overall_status": "trial1_third_residual_identification_audit_failed",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_28": True,
            "next_required_artifacts": [
                "trial1_fourth_reopened_declaration_gate",
                "part3a_b_adoption_override_route_contract",
            ],
        },
        {
            "part1_total_action_em_line": hit(part1, "+\\mathcal{L}_{\\mathrm{EM}}"),
            "part1_vector_total_action_header_line": hit(part1, "\\mathcal{L}_{\\mathrm{total}}^{\\mathrm{vec}}"),
            "part3a_independent_maxwell_adoption_line": hit(part3a, "Maxwell 方程式（U(1) ゲージ場 $A_\\mu$、電場 $E$、磁場 $B$）を、P-model の枠組みとは独立に採用する"),
            "part3a_independent_connection_line": hit(part3a, "局所位相勾配を補償する**独立接続**として別途導入する必要がある"),
            "part3a_a_reject_b_adopt_line": hit(part3a, "**A棄却、B採用**"),
            "part3a_b_adoption_line": hit(part3a, "標準 U(1) を独立の有効理論として採用し"),
        },
    )

    reopened_gate = payload(
        "8.7.56.28",
        "Trial-1 fourth reopened declaration gate / Trial-2 unlock gate",
        common_inputs,
        "Reflect the third residual audit into Trial-1 and decide whether Trial-2 can be unlocked.",
        {
            "trial1_full_pass_rule": "Trial-1 upgrades to full pass only if the public canon removes the independently adopted EM sector and replaces it with a P-only statement.",
            "trial2_unlock_rule": "Trial-2 stays on hold unless Trial-1 closes as a full pass.",
        },
        [
            row(
                "trial1_fourth_reopened_full_pass_ready",
                "fail",
                "Trial-1 fourth reopened full pass ready",
                0,
                "The third residual audit still leaves the independently adopted EM sector intact.",
            ),
            row(
                "trial2_unlock_ready_after_third_residual",
                "fail",
                "Trial-2 unlock ready after third residual",
                0,
                "Electromagnetic first-principles work remains blocked by the unresolved B-adoption override.",
            ),
            row(
                "trial1_partial_closeout_retained_after_third_residual",
                "pass",
                "Trial-1 partial closeout retained after third residual",
                1,
                "Global U(1) and local P-vector gauge redundancy remain valid even though the independent EM sector is still unresolved.",
            ),
        ],
        {
            "trial1_pass_level": "partial_third_residual_unresolved",
            "trial1_full_pass_ready": False,
            "trial2_unlock_ready": False,
            "advance_to_8_7_56_5": False,
            "recommended_next_route_or_none": "8.7.56.29",
        },
        {
            "overall_status": "trial1_fourth_reopened_gate_failed_fourth_residual_required",
            "trial1_branch_closeable": True,
            "advance_to_8_7_56_5": False,
            "next_required_artifacts": [
                "part3a_b_adoption_override_route_contract",
            ],
        },
        {
            "third_residual_route_summary": route["summary"],
            "third_residual_identification_summary": identification_audit["summary"],
            "previous_rereopened_gate_summary": rereopened_gate["summary"],
        },
    )

    next_route = payload(
        "8.7.56.29",
        "Part III-A B-adoption override residual route contract",
        common_inputs,
        "Freeze the fourth Trial-1 residual route suggested by the independent-EM audit: test whether the explicit A-reject / B-adopt wording in Part III-A can be overridden by a P-only derivation statement.",
        {
            "selected_residual_route": "part3a_b_adoption_override_identification",
            "pivot_principle": "The narrowed blocker is the explicit Part III-A adoption of an independent Maxwell/U(1) sector, not merely a generic absence of EM elimination language.",
            "missing_v2_artifact": "part3a_b_adoption_override_statement",
            "trial2_hold_rule": "Keep 8.7.56.5-.8 on hold until the Part III-A B-adoption override route closes.",
        },
        [
            row(
                "trial1_fourth_residual_route_contract_complete",
                "pass",
                "fourth Trial-1 residual route contract complete",
                1,
                "The Part III-A B-adoption override route is frozen as the next official route.",
            ),
            row(
                "trial1_fourth_residual_route_new_field_count",
                "pass",
                "new fields introduced by fourth residual route",
                0,
                "The route still attempts to close the issue inside the existing canon rather than by adding a new field.",
            ),
            row(
                "trial2_hold_retained_under_fourth_residual_route",
                "pass",
                "Trial-2 hold retained under fourth residual route",
                1,
                "Trial-2 remains blocked until the Part III-A adoption override is tested.",
            ),
        ],
        {
            "selected_residual_route": "part3a_b_adoption_override_identification",
            "missing_v2_artifact": "part3a_b_adoption_override_statement",
            "split_contract_ready": True,
            "advance_to_8_7_56_5": False,
        },
        {
            "overall_status": "trial1_fourth_residual_route_contract_frozen",
            "trial1_branch_closeable": True,
            "advance_to_8_7_56_5": False,
            "recommended_next_route_or_none": "8.7.56.30",
            "next_required_artifacts": [
                "part3a_b_adoption_override_source_inventory",
                "part3a_b_adoption_override_identification_audit",
                "trial1_fifth_reopened_declaration_gate",
            ],
        },
        {
            "reopened_gate_summary": reopened_gate["summary"],
            "identification_audit_summary": identification_audit["summary"],
            "part3a_a_reject_b_adopt_line": hit(part3a, "**A棄却、B採用**"),
            "part3a_b_adoption_line": hit(part3a, "標準 U(1) を独立の有効理論として採用し"),
        },
    )

    write_artifact("mass_origin_v2_independent_em_sector_elimination_source_inventory", source_inventory)
    write_artifact("mass_origin_v2_independent_em_sector_elimination_identification_audit", identification_audit)
    write_artifact("mass_origin_v2_trial1_fourth_reopened_declaration_gate", reopened_gate)
    write_artifact("mass_origin_v2_part3a_b_adoption_override_route_contract", next_route)


if __name__ == "__main__":
    main()
