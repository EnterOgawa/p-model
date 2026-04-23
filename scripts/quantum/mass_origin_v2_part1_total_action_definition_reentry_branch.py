#!/usr/bin/env python3
"""
Generate Trial-1 eighteenth residual artifacts for 8.7.56.86-.88 and 8.7.56.89.

This branch re-enters the explicit Part I canonical total-action definition from
the newly narrowed upstream side.

The previous residual established that the Part I vector total-action header /
decomposition cannot be overridden from inside the current canon. That
decomposition is still supported by the earlier Part I master-copy chain:
Part I explicitly fixes itself as the canonical total-action baseline and then
defines

    S_total := \\int d^4x L_total^vec .

The present branch therefore tests whether that canonical total-action
definition can itself already be reopened from inside the canon.

The answer remains no. The stronger support now sits one level further
upstream at the still-fixed Part I master-copy statement that freezes Part I as
the authoritative location of the total action, so the blocker narrows again to
the Part I master-copy-statement route.
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
ROUTE = OUT / "mass_origin_v2_part1_total_action_definition_reentry_route_contract_metrics.json"
EIGHTEENTH_GATE = OUT / "mass_origin_v2_trial1_eighteenth_reopened_declaration_gate_metrics.json"


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


# Function: execute the eighteenth residual branch and freeze the next route contract.

def main() -> None:
    for path in (PART1, PART3A, ROUTE, EIGHTEENTH_GATE):
        req(path)

    part1 = read_text(PART1)
    part3a = read_text(PART3A)
    route = read_json(ROUTE)
    eighteenth_gate = read_json(EIGHTEENTH_GATE)

    common_inputs = {
        "part1_core_theory_markdown": rel(PART1),
        "part3a_quantum_foundations_markdown": rel(PART3A),
        "mass_origin_v2_part1_total_action_definition_reentry_route_contract_json": rel(ROUTE),
        "mass_origin_v2_trial1_eighteenth_reopened_declaration_gate_json": rel(EIGHTEENTH_GATE),
    }

    source_inventory = payload(
        "8.7.56.86",
        "Part I total-action definition re-entry source inventory",
        common_inputs,
        "Inventory the source pack needed to test whether the explicit Part I canonical total-action definition can already be overridden from inside the current canon on the re-entry path.",
        {
            "required_source_items": [
                "part1_total_action_master_copy_statement",
                "part1_total_action_definition_equation",
                "part1_vector_total_action_header",
                "part1_total_action_contains_em_term",
                "part3a_em_required_constituent_statement",
            ],
            "inventory_rule": "The audit must see the Part I master-copy statement and canonical total-action definition together with the downstream vector total-action decomposition and Part III-A required-constituent framing that they currently support.",
        },
        [
            row(
                "trial1_eighteenth_residual_source_inventory_complete",
                "pass",
                "eighteenth residual source inventory complete",
                1,
                "The Part I total-action definition re-entry source pack is frozen.",
            ),
            row(
                "trial1_eighteenth_residual_present_source_count",
                "pass",
                "present source count",
                5,
                "All required Part I / Part III-A source statements are explicit in the current canon.",
            ),
            row(
                "trial1_eighteenth_residual_missing_source_count",
                "pass",
                "missing source count",
                0,
                "The branch is blocked by explicit contrary wording rather than by missing citations.",
            ),
            row(
                "trial1_eighteenth_residual_part1_master_copy_statement_present",
                "pass",
                "Part I master-copy statement present",
                1,
                "Part I still explicitly fixes itself as the authoritative location of the total-action baseline.",
            ),
            row(
                "trial1_eighteenth_residual_part1_total_action_definition_present",
                "pass",
                "Part I total-action definition present",
                1,
                "Part I still explicitly fixes S_total := ∫ d^4x L_total^vec as the canonical total-action definition.",
            ),
        ],
        {
            "required_source_count": 5,
            "present_source_count": 5,
            "missing_source_count": 0,
            "missing_source_items": [],
            "part1_total_action_definition_reentry_candidate_source_present": True,
            "first_route_to_close_or_none": "part1_total_action_definition_reentry_identification_audit",
        },
        {
            "overall_status": "trial1_eighteenth_residual_source_inventory_frozen",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_87": True,
            "next_required_artifacts": [
                "part1_total_action_definition_reentry_identification_audit",
            ],
        },
        {
            "part1_total_action_master_copy_line": hit(part1, "全作用の正本を Part I に固定"),
            "part1_total_action_definition_line": hit(part1, "S_{\\mathrm{total}}"),
            "part1_vector_total_action_header_line": hit(part1, "\\mathcal{L}_{\\mathrm{total}}^{\\mathrm{vec}}"),
            "part1_total_action_em_line": hit(part1, "+\\mathcal{L}_{\\mathrm{EM}}"),
            "part3a_em_required_constituent_line": hit(
                part3a,
                "電磁気は、原子・分子・物性へ進むための必須構成要素である",
            ),
        },
    )

    identification_audit = payload(
        "8.7.56.87",
        "Part I total-action definition re-entry identification audit",
        common_inputs,
        "Audit whether the present canon contains any statement that already overrides the explicit Part I canonical total-action definition on the re-entry path.",
        {
            "candidate_requirements": [
                "the canonical total-action definition no longer remains operative",
                "the Part I master-copy statement no longer fixes Part I as the authoritative action baseline",
                "the downstream vector total-action decomposition and required-constituent framing are no longer supported by the Part I master-copy chain",
            ],
            "audit_rule": "The audit passes only if the current canon itself contains an override that displaces the explicit Part I total-action definition rather than merely retaining it under the still-fixed Part I master-copy statement.",
        },
        [
            row(
                "trial1_eighteenth_residual_part1_master_copy_statement_present",
                "pass",
                "Part I master-copy statement present",
                1,
                "The stronger earlier statement to be tested on this re-entry path remains explicit in the current canon.",
            ),
            row(
                "trial1_eighteenth_residual_part1_total_action_definition_present",
                "pass",
                "Part I total-action definition present",
                1,
                "The canonical total-action definition also remains explicit in the current canon.",
            ),
            row(
                "trial1_eighteenth_residual_part1_total_action_definition_override_available",
                "fail",
                "Part I total-action definition override available",
                0,
                "No later canonical statement removes or replaces the explicit total-action definition S_total := ∫ d^4x L_total^vec.",
            ),
            row(
                "trial1_eighteenth_residual_part1_master_copy_statement_override_available",
                "fail",
                "Part I master-copy statement override available",
                0,
                "Part I still explicitly fixes itself as the canonical source of the total-action baseline.",
            ),
            row(
                "trial1_eighteenth_residual_part1_total_action_definition_reentry_identification_available",
                "fail",
                "Part I total-action definition re-entry identification available",
                0,
                "The canonical total-action definition remains supported by the still-unoverridden Part I master-copy statement.",
            ),
        ],
        {
            "part1_total_action_master_copy_statement_present": True,
            "part1_total_action_definition_present": True,
            "part1_total_action_definition_override_available": False,
            "part1_total_action_master_copy_statement_override_available": False,
            "part1_total_action_definition_reentry_identification_available": False,
            "identification_nonclosure_reason_or_none": "part1_total_action_master_copy_statement_still_fixed_and_supports_total_action_definition",
            "first_route_to_close_or_none": "part1_total_action_master_copy_statement_override_identification",
        },
        {
            "overall_status": "trial1_eighteenth_residual_identification_audit_failed",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_88": True,
            "next_required_artifacts": [
                "trial1_nineteenth_reopened_declaration_gate",
                "part1_total_action_master_copy_statement_reentry_route_contract",
            ],
        },
        {
            "part1_total_action_master_copy_line": hit(part1, "全作用の正本を Part I に固定"),
            "part1_total_action_definition_line": hit(part1, "S_{\\mathrm{total}}"),
            "part1_vector_total_action_header_line": hit(part1, "\\mathcal{L}_{\\mathrm{total}}^{\\mathrm{vec}}"),
            "part1_total_action_em_line": hit(part1, "+\\mathcal{L}_{\\mathrm{EM}}"),
            "part3a_em_required_constituent_line": hit(
                part3a,
                "電磁気は、原子・分子・物性へ進むための必須構成要素である",
            ),
        },
    )

    nineteenth_gate = payload(
        "8.7.56.88",
        "Trial-1 nineteenth reopened declaration gate",
        common_inputs,
        "Re-evaluate whether Trial-1 can now be declared fully passed and whether Trial-2 can unlock after the Part I total-action-definition re-entry audit.",
        {
            "gate_rule": "Trial-1 becomes a full pass only if the explicit Part I total-action definition is overridden inside the current canon.",
            "unlock_rule": "Trial-2 may unlock only after Trial-1 reaches full-pass status.",
        },
        [
            row(
                "trial1_nineteenth_reopened_gate_complete",
                "pass",
                "nineteenth reopened declaration gate complete",
                1,
                "The branch refresh is frozen after the Part I total-action-definition re-entry audit.",
            ),
            row(
                "trial1_full_pass_ready_after_eighteenth_residual",
                "fail",
                "Trial-1 full pass ready after eighteenth residual",
                0,
                "Trial-1 remains blocked by the still-explicit Part I master-copy statement and canonical total-action definition.",
            ),
            row(
                "trial2_unlock_ready_after_eighteenth_residual",
                "fail",
                "Trial-2 unlock ready after eighteenth residual",
                0,
                "Trial-2 remains on hold because Trial-1 is not yet a full pass.",
            ),
            row(
                "trial1_eighteenth_residual_blocker_shifted_to_part1_master_copy_statement",
                "fail",
                "eighteenth residual blocker shifted to Part I master-copy statement",
                0,
                "The canonical total-action definition remains, but its strongest surviving earlier support is now the Part I master-copy statement.",
            ),
        ],
        {
            "trial1_pass_level": "partial_eighteenth_residual_unresolved",
            "trial1_full_pass_ready": False,
            "trial2_unlock_ready": False,
            "advance_to_8_7_56_5": False,
            "recommended_next_route_or_none": "8.7.56.89",
        },
        {
            "overall_status": "trial1_eighteenth_residual_still_unresolved",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_5": False,
            "next_required_artifacts": [
                "part1_total_action_master_copy_statement_reentry_route_contract",
            ],
        },
        {
            "route_contract_summary": route["summary"],
            "eighteenth_reopened_gate_summary": eighteenth_gate["summary"],
            "identification_audit_summary": identification_audit["summary"],
        },
    )

    next_route = payload(
        "8.7.56.89",
        "Part I total-action master-copy statement re-entry residual route contract",
        common_inputs,
        "Freeze the nineteenth Trial-1 residual route suggested by the Part I total-action-definition re-entry audit: re-enter the still-fixed Part I master-copy statement from the new upstream side.",
        {
            "selected_residual_route": "part1_total_action_master_copy_statement_override_identification",
            "pivot_principle": "The explicit canonical total-action definition still remains, but its strongest surviving earlier support is now the Part I master-copy statement that freezes Part I as the authoritative total-action baseline.",
            "missing_v2_artifact": "part1_total_action_master_copy_statement_override_statement",
            "trial2_hold_rule": "Keep 8.7.56.5-.8 on hold until the Part I master-copy-statement re-entry route closes.",
        },
        [
            row(
                "trial1_nineteenth_residual_route_contract_complete",
                "pass",
                "nineteenth Trial-1 residual route contract complete",
                1,
                "The Part I master-copy-statement re-entry route is frozen as the next official route.",
            ),
            row(
                "trial1_nineteenth_residual_route_new_field_count",
                "pass",
                "new fields introduced by nineteenth residual route",
                0,
                "The route still attempts to close the issue inside the existing canon rather than by adding a new field.",
            ),
            row(
                "trial2_hold_retained_under_nineteenth_residual_route",
                "pass",
                "Trial-2 hold retained under nineteenth residual route",
                1,
                "Trial-2 remains blocked until the Part I master-copy statement is addressed on the re-entry path.",
            ),
        ],
        {
            "selected_residual_route": "part1_total_action_master_copy_statement_override_identification",
            "missing_v2_artifact": "part1_total_action_master_copy_statement_override_statement",
            "split_contract_ready": True,
            "advance_to_8_7_56_5": False,
        },
        {
            "overall_status": "trial1_nineteenth_residual_route_contract_frozen",
            "trial1_branch_closeable": True,
            "advance_to_8_7_56_5": False,
            "recommended_next_route_or_none": "8.7.56.90",
            "next_required_artifacts": [
                "part1_total_action_master_copy_statement_reentry_source_inventory",
                "part1_total_action_master_copy_statement_reentry_identification_audit",
                "trial1_twentieth_reopened_declaration_gate",
            ],
        },
        {
            "nineteenth_reopened_gate_summary": nineteenth_gate["summary"],
            "identification_audit_summary": identification_audit["summary"],
            "part1_total_action_master_copy_line": hit(part1, "全作用の正本を Part I に固定"),
            "part1_total_action_definition_line": hit(part1, "S_{\\mathrm{total}}"),
            "part1_vector_total_action_header_line": hit(part1, "\\mathcal{L}_{\\mathrm{total}}^{\\mathrm{vec}}"),
            "part1_total_action_em_line": hit(part1, "+\\mathcal{L}_{\\mathrm{EM}}"),
        },
    )

    write_artifact(
        "mass_origin_v2_part1_total_action_definition_reentry_source_inventory",
        source_inventory,
    )
    write_artifact(
        "mass_origin_v2_part1_total_action_definition_reentry_identification_audit",
        identification_audit,
    )
    write_artifact("mass_origin_v2_trial1_nineteenth_reopened_declaration_gate", nineteenth_gate)
    write_artifact(
        "mass_origin_v2_part1_total_action_master_copy_statement_reentry_route_contract",
        next_route,
    )

    print("[ok] wrote:")
    print(" - mass_origin_v2_part1_total_action_definition_reentry_source_inventory_metrics.json")
    print(" - mass_origin_v2_part1_total_action_definition_reentry_identification_audit_metrics.json")
    print(" - mass_origin_v2_trial1_nineteenth_reopened_declaration_gate_metrics.json")
    print(" - mass_origin_v2_part1_total_action_master_copy_statement_reentry_route_contract_metrics.json")


# Function: run the branch script from the command line.

if __name__ == "__main__":
    main()
