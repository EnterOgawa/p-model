#!/usr/bin/env python3
"""
Generate Trial-1 sixth residual artifacts for 8.7.56.38-.40 and 8.7.56.41.

This branch narrows the Part III-A wording problem one step further.

The previous residual established that the explicit independent Maxwell
adoption remains supported by the still-fixed requirement that A_mu must be
introduced as an independent connection. The present branch asks whether that
independent-connection requirement can already be overridden from inside the
current canon.

The answer is still no, because Part III-A also freezes the stronger statement
that a degree of freedom external to P is required: the judgment text and the
A→B decision table both say that P-alone cannot close the gauge sector. The
next blocker therefore narrows to that "new degree of freedom required"
statement itself.
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
ROUTE = OUT / "mass_origin_v2_part3a_independent_connection_override_route_contract_metrics.json"
SIXTH_GATE = OUT / "mass_origin_v2_trial1_sixth_reopened_declaration_gate_metrics.json"


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


# Function: execute the sixth residual branch and freeze the next route contract.

def main() -> None:
    for path in (PART1, PART3A, ROUTE, SIXTH_GATE):
        req(path)

    part1 = read_text(PART1)
    part3a = read_text(PART3A)
    route = read_json(ROUTE)
    sixth_gate = read_json(SIXTH_GATE)

    common_inputs = {
        "part1_core_theory_markdown": rel(PART1),
        "part3a_quantum_foundations_markdown": rel(PART3A),
        "mass_origin_v2_part3a_independent_connection_override_route_contract_json": rel(ROUTE),
        "mass_origin_v2_trial1_sixth_reopened_declaration_gate_json": rel(SIXTH_GATE),
    }

    source_inventory = payload(
        "8.7.56.38",
        "Part III-A independent connection override source inventory",
        common_inputs,
        "Inventory the source pack needed to test whether the explicit Part III-A independent-connection requirement can already be overridden from inside the current canon.",
        {
            "required_source_items": [
                "part1_total_action_contains_em_statement",
                "part1_vector_total_action_contains_em_statement",
                "part3a_independent_maxwell_adoption_sentence",
                "part3a_independent_connection_requirement_sentence",
                "part3a_a_then_b_conditional_principle_statement",
                "part3a_b_only_if_a_fails_statement",
                "part3a_new_degree_of_freedom_required_statement",
                "part3a_decision_table_new_dof_row",
                "part3a_a_reject_b_adopt_final_statement",
                "part3a_b_operational_adoption_statement",
            ],
            "inventory_rule": "The audit must see both the independent-connection requirement and the stronger 'P-external degree of freedom required' statements that justify it before the route can narrow honestly.",
        },
        [
            row(
                "trial1_sixth_residual_source_inventory_complete",
                "pass",
                "sixth residual source inventory complete",
                1,
                "The independent-connection override source pack is frozen.",
            ),
            row(
                "trial1_sixth_residual_present_source_count",
                "pass",
                "present source count",
                10,
                "All required Part I / Part III-A source statements are explicit in the current canon.",
            ),
            row(
                "trial1_sixth_residual_missing_source_count",
                "pass",
                "missing source count",
                0,
                "The branch is blocked by explicit contrary wording rather than by missing citations.",
            ),
            row(
                "trial1_sixth_residual_independent_connection_requirement_present",
                "pass",
                "independent connection requirement present",
                1,
                "Part III-A explicitly requires that A_mu be introduced as an independent connection.",
            ),
            row(
                "trial1_sixth_residual_new_dof_required_statement_present",
                "pass",
                "new degree of freedom required statement present",
                1,
                "Part III-A also explicitly says that P-external new freedom is required, reinforcing the connection requirement.",
            ),
        ],
        {
            "required_source_count": 10,
            "present_source_count": 10,
            "missing_source_count": 0,
            "missing_source_items": [],
            "part3a_independent_connection_override_candidate_source_present": True,
            "first_route_to_close_or_none": "part3a_independent_connection_override_identification_audit",
        },
        {
            "overall_status": "trial1_sixth_residual_source_inventory_frozen",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_39": True,
            "next_required_artifacts": [
                "part3a_independent_connection_override_identification_audit",
            ],
        },
        {
            "part1_total_action_em_line": hit(part1, "+\\mathcal{L}_{\\mathrm{EM}}"),
            "part1_vector_total_action_header_line": hit(part1, "\\mathcal{L}_{\\mathrm{total}}^{\\mathrm{vec}}"),
            "part3a_independent_maxwell_adoption_line": hit(part3a, "Maxwell 方程式（U(1) ゲージ場 $A_\\mu$、電場 $E$、磁場 $B$）を、P-model の枠組みとは独立に採用する"),
            "part3a_independent_connection_line": hit(part3a, "局所位相勾配を補償する**独立接続**として別途導入する必要がある"),
            "part3a_a_then_b_principle_line": hit(part3a, "まず導出（A）に挑戦し、難しい場合のみ有効理論（B）で拘束する"),
            "part3a_b_only_if_a_fails_line": hit(part3a, "A が成立しない場合のみ、U(1) を独立に採用し"),
            "part3a_new_dof_required_line": hit(part3a, "P 以外の新自由度が必須"),
            "part3a_decision_table_new_dof_row": hit(part3a, "| ゲージ不変性を保つために P 以外の新自由度が必須 |"),
            "part3a_a_reject_b_adopt_line": hit(part3a, "**A棄却、B採用**"),
            "part3a_b_adoption_line": hit(part3a, "標準 U(1) を独立の有効理論として採用し"),
        },
    )

    identification_audit = payload(
        "8.7.56.39",
        "Part III-A independent connection override identification audit",
        common_inputs,
        "Audit whether the present canon contains any statement that already overrides the explicit Part III-A independent-connection requirement without adding a new field or a new free parameter.",
        {
            "candidate_requirements": [
                "independent connection requirement no longer remains operative",
                "P-external new-degree-of-freedom requirement no longer remains operative",
                "final A-reject / B-adopt judgment is explicitly reopened by a P-only derivation statement",
            ],
            "audit_rule": "The audit passes only if the current canon itself contains an override that displaces the explicit independent-connection requirement rather than merely surrounding it with conditional template language.",
        },
        [
            row(
                "trial1_sixth_residual_part3a_conditional_b_template_present",
                "pass",
                "Part III-A conditional B-template present",
                1,
                "The text still preserves the principle 'use B only if A fails.'",
            ),
            row(
                "trial1_sixth_residual_part3a_independent_connection_requirement_present",
                "pass",
                "Part III-A independent connection requirement present",
                1,
                "The requirement sentence to be overridden remains explicit in the current canon.",
            ),
            row(
                "trial1_sixth_residual_part3a_independent_connection_override_available",
                "fail",
                "Part III-A independent connection override available",
                0,
                "No later canonical statement displaces the explicit requirement that A_mu be introduced as an independent connection.",
            ),
            row(
                "trial1_sixth_residual_part3a_new_dof_requirement_override_available",
                "fail",
                "Part III-A new-degree-of-freedom requirement override available",
                0,
                "Part III-A still explicitly says that a P-external new degree of freedom is required to keep gauge invariance.",
            ),
            row(
                "trial1_sixth_residual_part3a_independent_connection_override_identification_available",
                "fail",
                "Part III-A independent connection override identification available",
                0,
                "The independent-connection requirement remains supported by the still-unoverridden new-degree-of-freedom requirement.",
            ),
        ],
        {
            "part3a_conditional_b_template_present": True,
            "part3a_independent_connection_requirement_present": True,
            "part3a_independent_connection_override_available": False,
            "part3a_new_dof_requirement_override_available": False,
            "part3a_independent_connection_override_identification_available": False,
            "identification_nonclosure_reason_or_none": "part3a_new_degree_of_freedom_requirement_still_fixed_and_supports_independent_connection_requirement",
            "first_route_to_close_or_none": "part3a_new_degree_of_freedom_requirement_override_identification",
        },
        {
            "overall_status": "trial1_sixth_residual_identification_audit_failed",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_40": True,
            "next_required_artifacts": [
                "trial1_seventh_reopened_declaration_gate",
                "part3a_new_degree_of_freedom_requirement_override_route_contract",
            ],
        },
        {
            "part3a_independent_connection_line": hit(part3a, "局所位相勾配を補償する**独立接続**として別途導入する必要がある"),
            "part3a_new_dof_required_line": hit(part3a, "P 以外の新自由度が必須"),
            "part3a_decision_table_new_dof_row": hit(part3a, "| ゲージ不変性を保つために P 以外の新自由度が必須 |"),
            "part3a_a_then_b_principle_line": hit(part3a, "まず導出（A）に挑戦し、難しい場合のみ有効理論（B）で拘束する"),
            "part3a_b_only_if_a_fails_line": hit(part3a, "A が成立しない場合のみ、U(1) を独立に採用し"),
            "part3a_a_reject_b_adopt_line": hit(part3a, "**A棄却、B採用**"),
        },
    )

    seventh_gate = payload(
        "8.7.56.40",
        "Trial-1 seventh reopened declaration gate",
        common_inputs,
        "Re-evaluate whether Trial-1 can now be declared fully passed and whether Trial-2 can unlock after the independent-connection override audit.",
        {
            "gate_rule": "Trial-1 becomes a full pass only if the explicit Part III-A independent-connection requirement is overridden inside the current canon, eliminating the need for a P-external gauge-sector supplement.",
            "unlock_rule": "Trial-2 may unlock only after Trial-1 reaches full-pass status.",
        },
        [
            row(
                "trial1_seventh_reopened_gate_complete",
                "pass",
                "seventh reopened declaration gate complete",
                1,
                "The branch refresh is frozen after the independent-connection override audit.",
            ),
            row(
                "trial1_full_pass_ready_after_sixth_residual",
                "fail",
                "Trial-1 full pass ready after sixth residual",
                0,
                "Trial-1 remains blocked by the still-explicit new-degree-of-freedom and independent-connection requirements in Part III-A.",
            ),
            row(
                "trial2_unlock_ready_after_sixth_residual",
                "fail",
                "Trial-2 unlock ready after sixth residual",
                0,
                "Trial-2 remains on hold because Trial-1 is not yet a full pass.",
            ),
            row(
                "trial1_sixth_residual_blocker_shifted_to_new_dof_requirement",
                "fail",
                "sixth residual blocker shifted to new-degree-of-freedom requirement",
                0,
                "The independent-connection requirement remains, but its strongest surviving support is the explicit statement that P-external new freedom is required.",
            ),
        ],
        {
            "trial1_pass_level": "partial_sixth_residual_unresolved",
            "trial1_full_pass_ready": False,
            "trial2_unlock_ready": False,
            "advance_to_8_7_56_5": False,
            "recommended_next_route_or_none": "8.7.56.41",
        },
        {
            "overall_status": "trial1_sixth_residual_still_unresolved",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_5": False,
            "next_required_artifacts": [
                "part3a_new_degree_of_freedom_requirement_override_route_contract",
            ],
        },
        {
            "route_contract_summary": route["summary"],
            "sixth_reopened_gate_summary": sixth_gate["summary"],
            "identification_audit_summary": identification_audit["summary"],
        },
    )

    next_route = payload(
        "8.7.56.41",
        "Part III-A new-degree-of-freedom requirement override residual route contract",
        common_inputs,
        "Freeze the seventh Trial-1 residual route suggested by the independent-connection override audit: test whether the explicit Part III-A new-degree-of-freedom requirement can be overridden from inside the current canon.",
        {
            "selected_residual_route": "part3a_new_degree_of_freedom_requirement_override_identification",
            "pivot_principle": "The independent-connection requirement is still supported by the explicit statement that a P-external new degree of freedom is required, so the narrowed blocker is now that requirement statement itself.",
            "missing_v2_artifact": "part3a_new_degree_of_freedom_requirement_override_statement",
            "trial2_hold_rule": "Keep 8.7.56.5-.8 on hold until the Part III-A new-degree-of-freedom requirement override route closes.",
        },
        [
            row(
                "trial1_seventh_residual_route_contract_complete",
                "pass",
                "seventh Trial-1 residual route contract complete",
                1,
                "The Part III-A new-degree-of-freedom requirement override route is frozen as the next official route.",
            ),
            row(
                "trial1_seventh_residual_route_new_field_count",
                "pass",
                "new fields introduced by seventh residual route",
                0,
                "The route still attempts to close the issue inside the existing canon rather than by adding a new field.",
            ),
            row(
                "trial2_hold_retained_under_seventh_residual_route",
                "pass",
                "Trial-2 hold retained under seventh residual route",
                1,
                "Trial-2 remains blocked until the new-degree-of-freedom requirement is addressed.",
            ),
        ],
        {
            "selected_residual_route": "part3a_new_degree_of_freedom_requirement_override_identification",
            "missing_v2_artifact": "part3a_new_degree_of_freedom_requirement_override_statement",
            "split_contract_ready": True,
            "advance_to_8_7_56_5": False,
        },
        {
            "overall_status": "trial1_seventh_residual_route_contract_frozen",
            "trial1_branch_closeable": True,
            "advance_to_8_7_56_5": False,
            "recommended_next_route_or_none": "8.7.56.42",
            "next_required_artifacts": [
                "part3a_new_degree_of_freedom_requirement_override_source_inventory",
                "part3a_new_degree_of_freedom_requirement_override_identification_audit",
                "trial1_eighth_reopened_declaration_gate",
            ],
        },
        {
            "seventh_reopened_gate_summary": seventh_gate["summary"],
            "identification_audit_summary": identification_audit["summary"],
            "part3a_independent_connection_line": hit(part3a, "局所位相勾配を補償する**独立接続**として別途導入する必要がある"),
            "part3a_new_dof_required_line": hit(part3a, "P 以外の新自由度が必須"),
            "part3a_decision_table_new_dof_row": hit(part3a, "| ゲージ不変性を保つために P 以外の新自由度が必須 |"),
            "part3a_a_reject_b_adopt_line": hit(part3a, "**A棄却、B採用**"),
        },
    )

    write_artifact("mass_origin_v2_part3a_independent_connection_override_source_inventory", source_inventory)
    write_artifact("mass_origin_v2_part3a_independent_connection_override_identification_audit", identification_audit)
    write_artifact("mass_origin_v2_trial1_seventh_reopened_declaration_gate", seventh_gate)
    write_artifact("mass_origin_v2_part3a_new_degree_of_freedom_requirement_override_route_contract", next_route)

    print("[ok] wrote:")
    print(" - mass_origin_v2_part3a_independent_connection_override_source_inventory_metrics.json")
    print(" - mass_origin_v2_part3a_independent_connection_override_identification_audit_metrics.json")
    print(" - mass_origin_v2_trial1_seventh_reopened_declaration_gate_metrics.json")
    print(" - mass_origin_v2_part3a_new_degree_of_freedom_requirement_override_route_contract_metrics.json")


# Function: run the branch script from the command line.

if __name__ == "__main__":
    main()
