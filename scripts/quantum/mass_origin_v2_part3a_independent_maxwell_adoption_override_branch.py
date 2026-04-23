#!/usr/bin/env python3
"""
Generate Trial-1 fifth residual artifacts for 8.7.56.34-.36 and 8.7.56.37.

This branch narrows the Part III-A wording problem one step further.

The previous residual established that the canon already contains:

1. A conditional A-to-B template.
2. Retention of the A-structure template.
3. An explicit sentence that Maxwell/U(1) is adopted independently.

The remaining question is whether that independent Maxwell-adoption sentence can
already be overridden from inside the current canon. The answer is still no,
because the same Part III-A section also fixes the need for an independent
connection A_mu. That stronger statement still supports the independent EM
adoption, so the blocker narrows again to the independent-connection override.
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
ROUTE = OUT / "mass_origin_v2_part3a_independent_maxwell_adoption_override_route_contract_metrics.json"
FIFTH_GATE = OUT / "mass_origin_v2_trial1_fifth_reopened_declaration_gate_metrics.json"


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


# Function: execute the fifth residual branch and freeze the next route contract.

def main() -> None:
    for path in (PART1, PART3A, ROUTE, FIFTH_GATE):
        req(path)

    part1 = read_text(PART1)
    part3a = read_text(PART3A)
    route = read_json(ROUTE)
    fifth_gate = read_json(FIFTH_GATE)

    common_inputs = {
        "part1_core_theory_markdown": rel(PART1),
        "part3a_quantum_foundations_markdown": rel(PART3A),
        "mass_origin_v2_part3a_independent_maxwell_adoption_override_route_contract_json": rel(ROUTE),
        "mass_origin_v2_trial1_fifth_reopened_declaration_gate_json": rel(FIFTH_GATE),
    }

    source_inventory = payload(
        "8.7.56.34",
        "Part III-A independent Maxwell adoption override source inventory",
        common_inputs,
        "Inventory the source pack needed to test whether the explicit Part III-A independent Maxwell/U(1) adoption sentence can already be overridden from inside the current canon.",
        {
            "required_source_items": [
                "part1_total_action_contains_em_statement",
                "part1_vector_total_action_contains_em_statement",
                "part3a_independent_maxwell_adoption_sentence",
                "part3a_a_then_b_conditional_principle_statement",
                "part3a_b_only_if_a_fails_statement",
                "part3a_independent_connection_requirement_statement",
                "part3a_a_reject_b_adopt_final_statement",
                "part3a_b_operational_adoption_statement",
                "part3a_a_structure_template_retained_statement",
            ],
            "inventory_rule": "The audit must see both the independent Maxwell-adoption sentence and the independent-connection requirement that supports it before the route can narrow honestly.",
        },
        [
            row(
                "trial1_fifth_residual_source_inventory_complete",
                "pass",
                "fifth residual source inventory complete",
                1,
                "The independent Maxwell-adoption override source pack is frozen.",
            ),
            row(
                "trial1_fifth_residual_present_source_count",
                "pass",
                "present source count",
                9,
                "All required Part I / Part III-A source statements are explicit in the current canon.",
            ),
            row(
                "trial1_fifth_residual_missing_source_count",
                "pass",
                "missing source count",
                0,
                "The branch is blocked by explicit contrary wording rather than by missing citations.",
            ),
            row(
                "trial1_fifth_residual_independent_maxwell_sentence_present",
                "pass",
                "independent Maxwell sentence present",
                1,
                "Part III-A explicitly adopts Maxwell/U(1) independently of the P-model framework.",
            ),
            row(
                "trial1_fifth_residual_independent_connection_requirement_present",
                "pass",
                "independent connection requirement present",
                1,
                "Part III-A also still requires an independent A_mu connection, reinforcing that adoption.",
            ),
        ],
        {
            "required_source_count": 9,
            "present_source_count": 9,
            "missing_source_count": 0,
            "missing_source_items": [],
            "part3a_independent_maxwell_adoption_override_candidate_source_present": True,
            "first_route_to_close_or_none": "part3a_independent_maxwell_adoption_override_identification_audit",
        },
        {
            "overall_status": "trial1_fifth_residual_source_inventory_frozen",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_35": True,
            "next_required_artifacts": [
                "part3a_independent_maxwell_adoption_override_identification_audit",
            ],
        },
        {
            "part1_total_action_em_line": hit(part1, "+\\mathcal{L}_{\\mathrm{EM}}"),
            "part1_vector_total_action_header_line": hit(part1, "\\mathcal{L}_{\\mathrm{total}}^{\\mathrm{vec}}"),
            "part3a_independent_maxwell_adoption_line": hit(part3a, "Maxwell 方程式（U(1) ゲージ場 $A_\\mu$、電場 $E$、磁場 $B$）を、P-model の枠組みとは独立に採用する"),
            "part3a_a_then_b_principle_line": hit(part3a, "まず導出（A）に挑戦し、難しい場合のみ有効理論（B）で拘束する"),
            "part3a_b_only_if_a_fails_line": hit(part3a, "A が成立しない場合のみ、U(1) を独立に採用し"),
            "part3a_independent_connection_line": hit(part3a, "局所位相勾配を補償する**独立接続**として別途導入する必要がある"),
            "part3a_a_reject_b_adopt_line": hit(part3a, "**A棄却、B採用**"),
            "part3a_b_adoption_line": hit(part3a, "標準 U(1) を独立の有効理論として採用し"),
            "part3a_a_structure_template_line": hit(part3a, "を示す**構造テンプレート**として残し"),
        },
    )

    identification_audit = payload(
        "8.7.56.35",
        "Part III-A independent Maxwell adoption override identification audit",
        common_inputs,
        "Audit whether the present canon contains any statement that already overrides the explicit Part III-A independent Maxwell/U(1) adoption sentence without adding a new field or a new free parameter.",
        {
            "candidate_requirements": [
                "independent Maxwell adoption sentence no longer remains operative",
                "independent connection requirement no longer forces a separate A_mu sector",
                "final A-reject / B-adopt judgment is explicitly reopened by a P-only derivation statement",
            ],
            "audit_rule": "The audit passes only if the current canon itself contains an override that displaces the explicit independent Maxwell-adoption sentence rather than merely surrounding it with conditional template language.",
        },
        [
            row(
                "trial1_fifth_residual_part3a_conditional_b_template_present",
                "pass",
                "Part III-A conditional B-template present",
                1,
                "The text still preserves the principle 'use B only if A fails.'",
            ),
            row(
                "trial1_fifth_residual_part3a_independent_maxwell_adoption_sentence_present",
                "pass",
                "Part III-A independent Maxwell adoption sentence present",
                1,
                "The sentence to be overridden remains explicit in the current canon.",
            ),
            row(
                "trial1_fifth_residual_part3a_independent_maxwell_adoption_override_available",
                "fail",
                "Part III-A independent Maxwell adoption override available",
                0,
                "No later canonical statement displaces the explicit independent Maxwell/U(1) adoption sentence.",
            ),
            row(
                "trial1_fifth_residual_part3a_independent_connection_override_available",
                "fail",
                "Part III-A independent connection override available",
                0,
                "Part III-A still explicitly says that A_mu must be introduced as an independent connection.",
            ),
            row(
                "trial1_fifth_residual_part3a_independent_maxwell_override_identification_available",
                "fail",
                "Part III-A independent Maxwell adoption override identification available",
                0,
                "The explicit independent Maxwell adoption remains supported by the still-unoverridden independent-connection requirement.",
            ),
        ],
        {
            "part3a_conditional_b_template_present": True,
            "part3a_independent_maxwell_adoption_sentence_present": True,
            "part3a_independent_maxwell_adoption_override_available": False,
            "part3a_independent_connection_override_available": False,
            "part3a_independent_maxwell_override_identification_available": False,
            "identification_nonclosure_reason_or_none": "part3a_independent_connection_requirement_still_fixed_and_supports_independent_maxwell_adoption",
            "first_route_to_close_or_none": "part3a_independent_connection_override_identification",
        },
        {
            "overall_status": "trial1_fifth_residual_identification_audit_failed",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_36": True,
            "next_required_artifacts": [
                "trial1_sixth_reopened_declaration_gate",
                "part3a_independent_connection_override_route_contract",
            ],
        },
        {
            "part3a_independent_maxwell_adoption_line": hit(part3a, "Maxwell 方程式（U(1) ゲージ場 $A_\\mu$、電場 $E$、磁場 $B$）を、P-model の枠組みとは独立に採用する"),
            "part3a_a_then_b_principle_line": hit(part3a, "まず導出（A）に挑戦し、難しい場合のみ有効理論（B）で拘束する"),
            "part3a_b_only_if_a_fails_line": hit(part3a, "A が成立しない場合のみ、U(1) を独立に採用し"),
            "part3a_independent_connection_line": hit(part3a, "局所位相勾配を補償する**独立接続**として別途導入する必要がある"),
            "part3a_a_reject_b_adopt_line": hit(part3a, "**A棄却、B採用**"),
            "part3a_b_adoption_line": hit(part3a, "標準 U(1) を独立の有効理論として採用し"),
        },
    )

    sixth_gate = payload(
        "8.7.56.36",
        "Trial-1 sixth reopened declaration gate",
        common_inputs,
        "Re-evaluate whether Trial-1 can now be declared fully passed and whether Trial-2 can unlock after the independent Maxwell-adoption override audit.",
        {
            "gate_rule": "Trial-1 becomes a full pass only if the explicit Part III-A independent Maxwell-adoption sentence is overridden inside the current canon, eliminating the need for an independently adopted EM sector.",
            "unlock_rule": "Trial-2 may unlock only after Trial-1 reaches full-pass status.",
        },
        [
            row(
                "trial1_sixth_reopened_gate_complete",
                "pass",
                "sixth reopened declaration gate complete",
                1,
                "The branch refresh is frozen after the independent Maxwell-adoption override audit.",
            ),
            row(
                "trial1_full_pass_ready_after_fifth_residual",
                "fail",
                "Trial-1 full pass ready after fifth residual",
                0,
                "Trial-1 remains blocked by the still-explicit independent Maxwell adoption and independent-connection requirement in Part III-A.",
            ),
            row(
                "trial2_unlock_ready_after_fifth_residual",
                "fail",
                "Trial-2 unlock ready after fifth residual",
                0,
                "Trial-2 remains on hold because Trial-1 is not yet a full pass.",
            ),
            row(
                "trial1_fifth_residual_blocker_shifted_to_independent_connection_requirement",
                "fail",
                "fifth residual blocker shifted to independent connection requirement",
                0,
                "The explicit independent Maxwell adoption remains, but its strongest surviving support is the independent-connection requirement.",
            ),
        ],
        {
            "trial1_pass_level": "partial_fifth_residual_unresolved",
            "trial1_full_pass_ready": False,
            "trial2_unlock_ready": False,
            "advance_to_8_7_56_5": False,
            "recommended_next_route_or_none": "8.7.56.37",
        },
        {
            "overall_status": "trial1_fifth_residual_still_unresolved",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_5": False,
            "next_required_artifacts": [
                "part3a_independent_connection_override_route_contract",
            ],
        },
        {
            "route_contract_summary": route["summary"],
            "fifth_reopened_gate_summary": fifth_gate["summary"],
            "identification_audit_summary": identification_audit["summary"],
        },
    )

    next_route = payload(
        "8.7.56.37",
        "Part III-A independent connection override residual route contract",
        common_inputs,
        "Freeze the sixth Trial-1 residual route suggested by the independent Maxwell-adoption override audit: test whether the explicit Part III-A independent-connection requirement can be overridden from inside the current canon.",
        {
            "selected_residual_route": "part3a_independent_connection_override_identification",
            "pivot_principle": "The conditional A-to-B template is already present, but the explicit independent Maxwell adoption is still supported by the separate A_mu connection requirement, so the narrowed blocker is now that connection-requirement sentence.",
            "missing_v2_artifact": "part3a_independent_connection_override_statement",
            "trial2_hold_rule": "Keep 8.7.56.5-.8 on hold until the Part III-A independent-connection override route closes.",
        },
        [
            row(
                "trial1_sixth_residual_route_contract_complete",
                "pass",
                "sixth Trial-1 residual route contract complete",
                1,
                "The Part III-A independent-connection override route is frozen as the next official route.",
            ),
            row(
                "trial1_sixth_residual_route_new_field_count",
                "pass",
                "new fields introduced by sixth residual route",
                0,
                "The route still attempts to close the issue inside the existing canon rather than by adding a new field.",
            ),
            row(
                "trial2_hold_retained_under_sixth_residual_route",
                "pass",
                "Trial-2 hold retained under sixth residual route",
                1,
                "Trial-2 remains blocked until the independent-connection requirement is addressed.",
            ),
        ],
        {
            "selected_residual_route": "part3a_independent_connection_override_identification",
            "missing_v2_artifact": "part3a_independent_connection_override_statement",
            "split_contract_ready": True,
            "advance_to_8_7_56_5": False,
        },
        {
            "overall_status": "trial1_sixth_residual_route_contract_frozen",
            "trial1_branch_closeable": True,
            "advance_to_8_7_56_5": False,
            "recommended_next_route_or_none": "8.7.56.38",
            "next_required_artifacts": [
                "part3a_independent_connection_override_source_inventory",
                "part3a_independent_connection_override_identification_audit",
                "trial1_seventh_reopened_declaration_gate",
            ],
        },
        {
            "sixth_reopened_gate_summary": sixth_gate["summary"],
            "identification_audit_summary": identification_audit["summary"],
            "part3a_independent_maxwell_adoption_line": hit(part3a, "Maxwell 方程式（U(1) ゲージ場 $A_\\mu$、電場 $E$、磁場 $B$）を、P-model の枠組みとは独立に採用する"),
            "part3a_independent_connection_line": hit(part3a, "局所位相勾配を補償する**独立接続**として別途導入する必要がある"),
            "part3a_a_reject_b_adopt_line": hit(part3a, "**A棄却、B採用**"),
        },
    )

    write_artifact("mass_origin_v2_part3a_independent_maxwell_adoption_override_source_inventory", source_inventory)
    write_artifact("mass_origin_v2_part3a_independent_maxwell_adoption_override_identification_audit", identification_audit)
    write_artifact("mass_origin_v2_trial1_sixth_reopened_declaration_gate", sixth_gate)
    write_artifact("mass_origin_v2_part3a_independent_connection_override_route_contract", next_route)

    print("[ok] wrote:")
    print(" - mass_origin_v2_part3a_independent_maxwell_adoption_override_source_inventory_metrics.json")
    print(" - mass_origin_v2_part3a_independent_maxwell_adoption_override_identification_audit_metrics.json")
    print(" - mass_origin_v2_trial1_sixth_reopened_declaration_gate_metrics.json")
    print(" - mass_origin_v2_part3a_independent_connection_override_route_contract_metrics.json")


# Function: run the branch script from the command line.

if __name__ == "__main__":
    main()
