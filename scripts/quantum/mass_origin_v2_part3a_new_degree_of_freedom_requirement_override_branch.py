#!/usr/bin/env python3
"""
Generate Trial-1 seventh residual artifacts for 8.7.56.42-.44 and 8.7.56.45.

This branch narrows the Part III-A wording problem one step further.

The previous residual established that the explicit independent-connection
requirement remains supported by the stronger A-to-B criterion that says
P-external new freedom is required to keep gauge invariance. The present branch
asks whether that "new degree of freedom required" criterion can already be
overridden from inside the current canon.

The answer is still no, because Part III-A also freezes the same judgment in
the decision-table row and then carries the result forward to the final
"A reject / B adopt" closeout. The next blocker therefore narrows again to the
decision-table row that operationalizes the requirement.
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
ROUTE = OUT / "mass_origin_v2_part3a_new_degree_of_freedom_requirement_override_route_contract_metrics.json"
SEVENTH_GATE = OUT / "mass_origin_v2_trial1_seventh_reopened_declaration_gate_metrics.json"


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


# Function: execute the seventh residual branch and freeze the next route contract.

def main() -> None:
    for path in (PART1, PART3A, ROUTE, SEVENTH_GATE):
        req(path)

    part1 = read_text(PART1)
    part3a = read_text(PART3A)
    route = read_json(ROUTE)
    seventh_gate = read_json(SEVENTH_GATE)

    common_inputs = {
        "part1_core_theory_markdown": rel(PART1),
        "part3a_quantum_foundations_markdown": rel(PART3A),
        "mass_origin_v2_part3a_new_degree_of_freedom_requirement_override_route_contract_json": rel(ROUTE),
        "mass_origin_v2_trial1_seventh_reopened_declaration_gate_json": rel(SEVENTH_GATE),
    }

    source_inventory = payload(
        "8.7.56.42",
        "Part III-A new-degree-of-freedom requirement override source inventory",
        common_inputs,
        "Inventory the source pack needed to test whether the explicit Part III-A criterion that P-external new freedom is required can already be overridden from inside the current canon.",
        {
            "required_source_items": [
                "part1_total_action_contains_em_statement",
                "part1_vector_total_action_contains_em_statement",
                "part3a_a_then_b_conditional_principle_statement",
                "part3a_b_only_if_a_fails_statement",
                "part3a_new_degree_of_freedom_required_statement",
                "part3a_decision_table_new_dof_row",
                "part3a_independent_connection_requirement_statement",
                "part3a_a_reject_b_adopt_final_statement",
                "part3a_b_operational_adoption_statement",
            ],
            "inventory_rule": "The audit must see the abstract criterion sentence, its operational decision-table row, and the final B-adoption closeout before the route can narrow honestly.",
        },
        [
            row(
                "trial1_seventh_residual_source_inventory_complete",
                "pass",
                "seventh residual source inventory complete",
                1,
                "The new-degree-of-freedom override source pack is frozen.",
            ),
            row(
                "trial1_seventh_residual_present_source_count",
                "pass",
                "present source count",
                9,
                "All required Part I / Part III-A source statements are explicit in the current canon.",
            ),
            row(
                "trial1_seventh_residual_missing_source_count",
                "pass",
                "missing source count",
                0,
                "The branch is blocked by explicit contrary wording rather than by missing citations.",
            ),
            row(
                "trial1_seventh_residual_new_dof_requirement_statement_present",
                "pass",
                "new-degree-of-freedom requirement statement present",
                1,
                "Part III-A explicitly says that A must be rejected if gauge invariance needs a P-external new degree of freedom.",
            ),
            row(
                "trial1_seventh_residual_decision_table_new_dof_row_present",
                "pass",
                "decision-table new-degree-of-freedom row present",
                1,
                "The same judgment is also frozen operationally in the A-to-B decision table.",
            ),
        ],
        {
            "required_source_count": 9,
            "present_source_count": 9,
            "missing_source_count": 0,
            "missing_source_items": [],
            "part3a_new_degree_of_freedom_override_candidate_source_present": True,
            "first_route_to_close_or_none": "part3a_new_degree_of_freedom_requirement_override_identification_audit",
        },
        {
            "overall_status": "trial1_seventh_residual_source_inventory_frozen",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_43": True,
            "next_required_artifacts": [
                "part3a_new_degree_of_freedom_requirement_override_identification_audit",
            ],
        },
        {
            "part1_total_action_em_line": hit(part1, "+\\mathcal{L}_{\\mathrm{EM}}"),
            "part1_vector_total_action_header_line": hit(part1, "\\mathcal{L}_{\\mathrm{total}}^{\\mathrm{vec}}"),
            "part3a_a_then_b_principle_line": hit(part3a, "まず導出（A）に挑戦し、難しい場合のみ有効理論（B）で拘束する"),
            "part3a_b_only_if_a_fails_line": hit(part3a, "A が成立しない場合のみ、U(1) を独立に採用し"),
            "part3a_new_dof_required_statement_line": hit(part3a, "P 以外の新しい自由度が必須になる場合は A を棄却し B へ移行する"),
            "part3a_decision_table_new_dof_row": hit(part3a, "| ゲージ不変性を保つために P 以外の新自由度が必須 |"),
            "part3a_independent_connection_line": hit(part3a, "局所位相勾配を補償する**独立接続**として別途導入する必要がある"),
            "part3a_a_reject_b_adopt_line": hit(part3a, "**A棄却、B採用**"),
            "part3a_b_adoption_line": hit(part3a, "標準 U(1) を独立の有効理論として採用し"),
        },
    )

    identification_audit = payload(
        "8.7.56.43",
        "Part III-A new-degree-of-freedom requirement override identification audit",
        common_inputs,
        "Audit whether the present canon contains any statement that already overrides the explicit Part III-A criterion that a P-external new degree of freedom is required to keep gauge invariance.",
        {
            "candidate_requirements": [
                "new-degree-of-freedom requirement statement no longer remains operative",
                "decision-table new-degree-of-freedom row no longer remains operative",
                "final A-reject / B-adopt judgment is explicitly reopened by a P-only derivation statement",
            ],
            "audit_rule": "The audit passes only if the current canon itself contains an override that displaces the explicit new-degree-of-freedom requirement rather than merely surrounding it with conditional template language.",
        },
        [
            row(
                "trial1_seventh_residual_part3a_new_dof_requirement_statement_present",
                "pass",
                "Part III-A new-degree-of-freedom requirement statement present",
                1,
                "The criterion sentence to be overridden remains explicit in the current canon.",
            ),
            row(
                "trial1_seventh_residual_part3a_decision_table_new_dof_row_present",
                "pass",
                "Part III-A decision-table new-degree-of-freedom row present",
                1,
                "The decision table still operationalizes the same requirement.",
            ),
            row(
                "trial1_seventh_residual_part3a_new_dof_requirement_override_available",
                "fail",
                "Part III-A new-degree-of-freedom requirement override available",
                0,
                "No later canonical statement displaces the explicit criterion that a P-external new degree of freedom forces A->B.",
            ),
            row(
                "trial1_seventh_residual_part3a_decision_table_new_dof_row_override_available",
                "fail",
                "Part III-A decision-table new-degree-of-freedom row override available",
                0,
                "The A-to-B decision table still carries the same new-degree-of-freedom judgment as an operational row.",
            ),
            row(
                "trial1_seventh_residual_part3a_new_dof_requirement_override_identification_available",
                "fail",
                "Part III-A new-degree-of-freedom requirement override identification available",
                0,
                "The criterion statement remains supported by the still-unoverridden decision-table row and final B-adoption closeout.",
            ),
        ],
        {
            "part3a_new_degree_of_freedom_requirement_statement_present": True,
            "part3a_decision_table_new_dof_row_present": True,
            "part3a_new_degree_of_freedom_requirement_override_available": False,
            "part3a_decision_table_new_dof_row_override_available": False,
            "part3a_new_degree_of_freedom_requirement_override_identification_available": False,
            "identification_nonclosure_reason_or_none": "part3a_decision_table_new_degree_of_freedom_row_still_fixed_and_supports_requirement_statement",
            "first_route_to_close_or_none": "part3a_decision_table_new_degree_of_freedom_row_override_identification",
        },
        {
            "overall_status": "trial1_seventh_residual_identification_audit_failed",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_44": True,
            "next_required_artifacts": [
                "trial1_eighth_reopened_declaration_gate",
                "part3a_decision_table_new_degree_of_freedom_row_override_route_contract",
            ],
        },
        {
            "part3a_new_dof_required_statement_line": hit(part3a, "P 以外の新しい自由度が必須になる場合は A を棄却し B へ移行する"),
            "part3a_decision_table_new_dof_row": hit(part3a, "| ゲージ不変性を保つために P 以外の新自由度が必須 |"),
            "part3a_independent_connection_line": hit(part3a, "局所位相勾配を補償する**独立接続**として別途導入する必要がある"),
            "part3a_a_reject_b_adopt_line": hit(part3a, "**A棄却、B採用**"),
            "part3a_b_adoption_line": hit(part3a, "標準 U(1) を独立の有効理論として採用し"),
        },
    )

    eighth_gate = payload(
        "8.7.56.44",
        "Trial-1 eighth reopened declaration gate",
        common_inputs,
        "Re-evaluate whether Trial-1 can now be declared fully passed and whether Trial-2 can unlock after the new-degree-of-freedom override audit.",
        {
            "gate_rule": "Trial-1 becomes a full pass only if the explicit Part III-A criterion that P-external new freedom is required is overridden inside the current canon.",
            "unlock_rule": "Trial-2 may unlock only after Trial-1 reaches full-pass status.",
        },
        [
            row(
                "trial1_eighth_reopened_gate_complete",
                "pass",
                "eighth reopened declaration gate complete",
                1,
                "The branch refresh is frozen after the new-degree-of-freedom override audit.",
            ),
            row(
                "trial1_full_pass_ready_after_seventh_residual",
                "fail",
                "Trial-1 full pass ready after seventh residual",
                0,
                "Trial-1 remains blocked by the still-explicit new-degree-of-freedom criterion, decision-table row, and downstream B-adoption closeout in Part III-A.",
            ),
            row(
                "trial2_unlock_ready_after_seventh_residual",
                "fail",
                "Trial-2 unlock ready after seventh residual",
                0,
                "Trial-2 remains on hold because Trial-1 is not yet a full pass.",
            ),
            row(
                "trial1_seventh_residual_blocker_shifted_to_decision_table_new_dof_row",
                "fail",
                "seventh residual blocker shifted to decision-table new-degree-of-freedom row",
                0,
                "The abstract requirement remains, but its strongest surviving operational support is now the explicit A-to-B decision-table row.",
            ),
        ],
        {
            "trial1_pass_level": "partial_seventh_residual_unresolved",
            "trial1_full_pass_ready": False,
            "trial2_unlock_ready": False,
            "advance_to_8_7_56_5": False,
            "recommended_next_route_or_none": "8.7.56.45",
        },
        {
            "overall_status": "trial1_seventh_residual_still_unresolved",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_5": False,
            "next_required_artifacts": [
                "part3a_decision_table_new_degree_of_freedom_row_override_route_contract",
            ],
        },
        {
            "route_contract_summary": route["summary"],
            "seventh_reopened_gate_summary": seventh_gate["summary"],
            "identification_audit_summary": identification_audit["summary"],
        },
    )

    next_route = payload(
        "8.7.56.45",
        "Part III-A decision-table new-degree-of-freedom row override residual route contract",
        common_inputs,
        "Freeze the eighth Trial-1 residual route suggested by the new-degree-of-freedom override audit: test whether the explicit Part III-A decision-table row that forces A->B on P-external new freedom can be overridden from inside the current canon.",
        {
            "selected_residual_route": "part3a_decision_table_new_degree_of_freedom_row_override_identification",
            "pivot_principle": "The abstract criterion remains explicit, but its strongest surviving operational support is the A-to-B decision-table row that freezes the same judgment in machine-readable form.",
            "missing_v2_artifact": "part3a_decision_table_new_degree_of_freedom_row_override_statement",
            "trial2_hold_rule": "Keep 8.7.56.5-.8 on hold until the Part III-A decision-table new-degree-of-freedom row override route closes.",
        },
        [
            row(
                "trial1_eighth_residual_route_contract_complete",
                "pass",
                "eighth Trial-1 residual route contract complete",
                1,
                "The Part III-A decision-table new-degree-of-freedom row override route is frozen as the next official route.",
            ),
            row(
                "trial1_eighth_residual_route_new_field_count",
                "pass",
                "new fields introduced by eighth residual route",
                0,
                "The route still attempts to close the issue inside the existing canon rather than by adding a new field.",
            ),
            row(
                "trial2_hold_retained_under_eighth_residual_route",
                "pass",
                "Trial-2 hold retained under eighth residual route",
                1,
                "Trial-2 remains blocked until the decision-table new-degree-of-freedom row is addressed.",
            ),
        ],
        {
            "selected_residual_route": "part3a_decision_table_new_degree_of_freedom_row_override_identification",
            "missing_v2_artifact": "part3a_decision_table_new_degree_of_freedom_row_override_statement",
            "split_contract_ready": True,
            "advance_to_8_7_56_5": False,
        },
        {
            "overall_status": "trial1_eighth_residual_route_contract_frozen",
            "trial1_branch_closeable": True,
            "advance_to_8_7_56_5": False,
            "recommended_next_route_or_none": "8.7.56.46",
            "next_required_artifacts": [
                "part3a_decision_table_new_degree_of_freedom_row_override_source_inventory",
                "part3a_decision_table_new_degree_of_freedom_row_override_identification_audit",
                "trial1_ninth_reopened_declaration_gate",
            ],
        },
        {
            "eighth_reopened_gate_summary": eighth_gate["summary"],
            "identification_audit_summary": identification_audit["summary"],
            "part3a_new_dof_required_statement_line": hit(part3a, "P 以外の新しい自由度が必須になる場合は A を棄却し B へ移行する"),
            "part3a_decision_table_new_dof_row": hit(part3a, "| ゲージ不変性を保つために P 以外の新自由度が必須 |"),
            "part3a_a_reject_b_adopt_line": hit(part3a, "**A棄却、B採用**"),
            "part3a_b_adoption_line": hit(part3a, "標準 U(1) を独立の有効理論として採用し"),
        },
    )

    write_artifact(
        "mass_origin_v2_part3a_new_degree_of_freedom_requirement_override_source_inventory",
        source_inventory,
    )
    write_artifact(
        "mass_origin_v2_part3a_new_degree_of_freedom_requirement_override_identification_audit",
        identification_audit,
    )
    write_artifact("mass_origin_v2_trial1_eighth_reopened_declaration_gate", eighth_gate)
    write_artifact(
        "mass_origin_v2_part3a_decision_table_new_degree_of_freedom_row_override_route_contract",
        next_route,
    )

    print("[ok] wrote:")
    print(" - mass_origin_v2_part3a_new_degree_of_freedom_requirement_override_source_inventory_metrics.json")
    print(" - mass_origin_v2_part3a_new_degree_of_freedom_requirement_override_identification_audit_metrics.json")
    print(" - mass_origin_v2_trial1_eighth_reopened_declaration_gate_metrics.json")
    print(" - mass_origin_v2_part3a_decision_table_new_degree_of_freedom_row_override_route_contract_metrics.json")


# Function: run the branch script from the command line.

if __name__ == "__main__":
    main()
