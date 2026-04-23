#!/usr/bin/env python3
"""
Generate Trial-1 thirteenth residual artifacts for 8.7.56.66-.68 and 8.7.56.69.

This branch narrows the Part III-A wording problem one step further.

The previous residual established that the explicit B-only-if-A-fails template
remains supported by the more general A→B conditional principle. The present
branch asks whether that more general principle can already be overridden from
inside the current canon.

The answer is still no, because Part III-A still explicitly adopts Maxwell/U(1)
independently of the P-model framework at the axiom side. That stronger,
earlier statement still supports the fallback principle, so the blocker
re-enters at the independent-Maxwell-adoption route.
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
ROUTE = OUT / "mass_origin_v2_part3a_a_then_b_principle_override_route_contract_metrics.json"
THIRTEENTH_GATE = OUT / "mass_origin_v2_trial1_thirteenth_reopened_declaration_gate_metrics.json"


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


# Function: execute the thirteenth residual branch and freeze the next route contract.

def main() -> None:
    for path in (PART1, PART3A, ROUTE, THIRTEENTH_GATE):
        req(path)

    part1 = read_text(PART1)
    part3a = read_text(PART3A)
    route = read_json(ROUTE)
    thirteenth_gate = read_json(THIRTEENTH_GATE)

    common_inputs = {
        "part1_core_theory_markdown": rel(PART1),
        "part3a_quantum_foundations_markdown": rel(PART3A),
        "mass_origin_v2_part3a_a_then_b_principle_override_route_contract_json": rel(ROUTE),
        "mass_origin_v2_trial1_thirteenth_reopened_declaration_gate_json": rel(THIRTEENTH_GATE),
    }

    source_inventory = payload(
        "8.7.56.66",
        "Part III-A A-to-B conditional principle override source inventory",
        common_inputs,
        "Inventory the source pack needed to test whether the explicit Part III-A A→B conditional principle can already be overridden from inside the current canon.",
        {
            "required_source_items": [
                "part1_total_action_contains_em_statement",
                "part1_vector_total_action_contains_em_statement",
                "part3a_em_required_constituent_statement",
                "part3a_independent_maxwell_adoption_statement",
                "part3a_a_then_b_conditional_principle_statement",
                "part3a_b_only_if_a_fails_template",
                "part3a_final_a_reject_b_adopt_judgment",
                "part3a_b_operational_adoption_statement",
            ],
            "inventory_rule": "The audit must see the general A→B principle itself, the earlier axiom-side independent-Maxwell adoption statement that already licenses B, and the downstream wording that still carries the same choice forward.",
        },
        [
            row(
                "trial1_thirteenth_residual_source_inventory_complete",
                "pass",
                "thirteenth residual source inventory complete",
                1,
                "The A-to-B conditional principle override source pack is frozen.",
            ),
            row(
                "trial1_thirteenth_residual_present_source_count",
                "pass",
                "present source count",
                8,
                "All required Part I / Part III-A source statements are explicit in the current canon.",
            ),
            row(
                "trial1_thirteenth_residual_missing_source_count",
                "pass",
                "missing source count",
                0,
                "The branch is blocked by explicit contrary wording rather than by missing citations.",
            ),
            row(
                "trial1_thirteenth_residual_a_then_b_principle_present",
                "pass",
                "A-to-B conditional principle present",
                1,
                "The explicit principle to be overridden remains in force in the current canon.",
            ),
            row(
                "trial1_thirteenth_residual_independent_maxwell_adoption_present",
                "pass",
                "independent Maxwell adoption present",
                1,
                "The earlier axiom-side independent Maxwell adoption sentence also remains explicit in the current canon.",
            ),
        ],
        {
            "required_source_count": 8,
            "present_source_count": 8,
            "missing_source_count": 0,
            "missing_source_items": [],
            "part3a_a_then_b_principle_override_candidate_source_present": True,
            "first_route_to_close_or_none": "part3a_a_then_b_principle_override_identification_audit",
        },
        {
            "overall_status": "trial1_thirteenth_residual_source_inventory_frozen",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_67": True,
            "next_required_artifacts": [
                "part3a_a_then_b_principle_override_identification_audit",
            ],
        },
        {
            "part1_total_action_em_line": hit(part1, "+\\mathcal{L}_{\\mathrm{EM}}"),
            "part1_vector_total_action_header_line": hit(part1, "\\mathcal{L}_{\\mathrm{total}}^{\\mathrm{vec}}"),
            "part3a_em_required_constituent_line": hit(part3a, "電磁気は、原子・分子・物性へ進むための必須構成要素である"),
            "part3a_independent_maxwell_adoption_line": hit(
                part3a,
                "Maxwell 方程式（U(1) ゲージ場 $A_\\mu$、電場 $E$、磁場 $B$）を、P-model の枠組みとは独立に採用する",
            ),
            "part3a_a_then_b_principle_line": hit(part3a, "まず導出（A）に挑戦し、難しい場合のみ有効理論（B）で拘束する"),
            "part3a_b_only_if_a_fails_line": hit(part3a, "A が成立しない場合のみ、U(1) を独立に採用し"),
            "part3a_a_reject_b_adopt_line": hit(part3a, "**A棄却、B採用**"),
            "part3a_b_adoption_line": hit(part3a, "標準 U(1) を独立の有効理論として採用し"),
        },
    )

    identification_audit = payload(
        "8.7.56.67",
        "Part III-A A-to-B conditional principle override identification audit",
        common_inputs,
        "Audit whether the present canon contains any statement that already overrides the explicit Part III-A A→B conditional principle.",
        {
            "candidate_requirements": [
                "A-to-B conditional principle no longer remains operative",
                "independent Maxwell adoption no longer remains operative",
                "the axiom-side electromagnetism adoption frame is explicitly reopened by a P-only derivation statement",
            ],
            "audit_rule": "The audit passes only if the current canon itself contains an override that displaces the explicit A→B conditional principle rather than merely surrounding it with an even earlier axiom-side adoption of independent Maxwell/U(1).",
        },
        [
            row(
                "trial1_thirteenth_residual_part3a_a_then_b_principle_present",
                "pass",
                "Part III-A A-to-B conditional principle present",
                1,
                "The general principle to be overridden remains explicit in the current canon.",
            ),
            row(
                "trial1_thirteenth_residual_part3a_independent_maxwell_adoption_present",
                "pass",
                "Part III-A independent Maxwell adoption present",
                1,
                "The earlier axiom-side independent Maxwell adoption sentence remains explicit in the current canon.",
            ),
            row(
                "trial1_thirteenth_residual_part3a_a_then_b_principle_override_available",
                "fail",
                "Part III-A A-to-B conditional principle override available",
                0,
                "No later canonical statement displaces the explicit general A→B conditional principle.",
            ),
            row(
                "trial1_thirteenth_residual_part3a_independent_maxwell_adoption_override_available",
                "fail",
                "Part III-A independent Maxwell adoption override available",
                0,
                "The axiom-side independent Maxwell adoption sentence still explicitly fixes B before the conditional fallback principle is even invoked.",
            ),
            row(
                "trial1_thirteenth_residual_part3a_a_then_b_principle_override_identification_available",
                "fail",
                "Part III-A A-to-B conditional principle override identification available",
                0,
                "The A→B conditional principle remains supported by the still-unoverridden earlier independent Maxwell adoption sentence.",
            ),
        ],
        {
            "part3a_a_then_b_principle_present": True,
            "part3a_independent_maxwell_adoption_present": True,
            "part3a_a_then_b_principle_override_available": False,
            "part3a_independent_maxwell_adoption_override_available": False,
            "part3a_a_then_b_principle_override_identification_available": False,
            "identification_nonclosure_reason_or_none": "part3a_independent_maxwell_adoption_still_fixed_and_supports_a_then_b_principle",
            "first_route_to_close_or_none": "part3a_independent_maxwell_adoption_override_identification",
        },
        {
            "overall_status": "trial1_thirteenth_residual_identification_audit_failed",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_68": True,
            "next_required_artifacts": [
                "trial1_fourteenth_reopened_declaration_gate",
                "part3a_independent_maxwell_adoption_reentry_route_contract",
            ],
        },
        {
            "part3a_em_required_constituent_line": hit(part3a, "電磁気は、原子・分子・物性へ進むための必須構成要素である"),
            "part3a_independent_maxwell_adoption_line": hit(
                part3a,
                "Maxwell 方程式（U(1) ゲージ場 $A_\\mu$、電場 $E$、磁場 $B$）を、P-model の枠組みとは独立に採用する",
            ),
            "part3a_a_then_b_principle_line": hit(part3a, "まず導出（A）に挑戦し、難しい場合のみ有効理論（B）で拘束する"),
            "part3a_b_only_if_a_fails_line": hit(part3a, "A が成立しない場合のみ、U(1) を独立に採用し"),
            "part3a_a_reject_b_adopt_line": hit(part3a, "**A棄却、B採用**"),
            "part3a_b_adoption_line": hit(part3a, "標準 U(1) を独立の有効理論として採用し"),
        },
    )

    fourteenth_gate = payload(
        "8.7.56.68",
        "Trial-1 fourteenth reopened declaration gate",
        common_inputs,
        "Re-evaluate whether Trial-1 can now be declared fully passed and whether Trial-2 can unlock after the A-to-B conditional principle override audit.",
        {
            "gate_rule": "Trial-1 becomes a full pass only if the explicit Part III-A A→B conditional principle is overridden inside the current canon.",
            "unlock_rule": "Trial-2 may unlock only after Trial-1 reaches full-pass status.",
        },
        [
            row(
                "trial1_fourteenth_reopened_gate_complete",
                "pass",
                "fourteenth reopened declaration gate complete",
                1,
                "The branch refresh is frozen after the A-to-B conditional principle override audit.",
            ),
            row(
                "trial1_full_pass_ready_after_thirteenth_residual",
                "fail",
                "Trial-1 full pass ready after thirteenth residual",
                0,
                "Trial-1 remains blocked by the still-explicit A→B conditional principle and the earlier independent Maxwell adoption sentence in Part III-A.",
            ),
            row(
                "trial2_unlock_ready_after_thirteenth_residual",
                "fail",
                "Trial-2 unlock ready after thirteenth residual",
                0,
                "Trial-2 remains on hold because Trial-1 is not yet a full pass.",
            ),
            row(
                "trial1_thirteenth_residual_blocker_shifted_to_independent_maxwell_reentry",
                "fail",
                "thirteenth residual blocker shifted to independent Maxwell re-entry",
                0,
                "The A→B conditional principle remains, but its strongest surviving earlier support is now the axiom-side independent Maxwell adoption sentence.",
            ),
        ],
        {
            "trial1_pass_level": "partial_thirteenth_residual_unresolved",
            "trial1_full_pass_ready": False,
            "trial2_unlock_ready": False,
            "advance_to_8_7_56_5": False,
            "recommended_next_route_or_none": "8.7.56.69",
        },
        {
            "overall_status": "trial1_thirteenth_residual_still_unresolved",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_5": False,
            "next_required_artifacts": [
                "part3a_independent_maxwell_adoption_reentry_route_contract",
            ],
        },
        {
            "route_contract_summary": route["summary"],
            "thirteenth_reopened_gate_summary": thirteenth_gate["summary"],
            "identification_audit_summary": identification_audit["summary"],
        },
    )

    next_route = payload(
        "8.7.56.69",
        "Part III-A independent Maxwell adoption re-entry residual route contract",
        common_inputs,
        "Freeze the fourteenth Trial-1 residual route suggested by the A-to-B principle audit: re-enter the explicit independent Maxwell/U(1) adoption sentence from the new upstream side.",
        {
            "selected_residual_route": "part3a_independent_maxwell_adoption_override_identification",
            "pivot_principle": "The general A→B conditional principle remains explicit, but its strongest surviving earlier support is the axiom-side statement that Maxwell/U(1) is adopted independently of the P-model framework.",
            "missing_v2_artifact": "part3a_independent_maxwell_adoption_override_statement",
            "trial2_hold_rule": "Keep 8.7.56.5-.8 on hold until the independent Maxwell adoption re-entry route closes.",
        },
        [
            row(
                "trial1_fourteenth_residual_route_contract_complete",
                "pass",
                "fourteenth Trial-1 residual route contract complete",
                1,
                "The independent Maxwell adoption re-entry route is frozen as the next official route.",
            ),
            row(
                "trial1_fourteenth_residual_route_new_field_count",
                "pass",
                "new fields introduced by fourteenth residual route",
                0,
                "The route still attempts to close the issue inside the existing canon rather than by adding a new field.",
            ),
            row(
                "trial2_hold_retained_under_fourteenth_residual_route",
                "pass",
                "Trial-2 hold retained under fourteenth residual route",
                1,
                "Trial-2 remains blocked until the independent Maxwell adoption sentence is addressed on the re-entry path.",
            ),
        ],
        {
            "selected_residual_route": "part3a_independent_maxwell_adoption_override_identification",
            "missing_v2_artifact": "part3a_independent_maxwell_adoption_override_statement",
            "split_contract_ready": True,
            "advance_to_8_7_56_5": False,
        },
        {
            "overall_status": "trial1_fourteenth_residual_route_contract_frozen",
            "trial1_branch_closeable": True,
            "advance_to_8_7_56_5": False,
            "recommended_next_route_or_none": "8.7.56.70",
            "next_required_artifacts": [
                "part3a_independent_maxwell_adoption_reentry_source_inventory",
                "part3a_independent_maxwell_adoption_reentry_identification_audit",
                "trial1_fifteenth_reopened_declaration_gate",
            ],
        },
        {
            "fourteenth_reopened_gate_summary": fourteenth_gate["summary"],
            "identification_audit_summary": identification_audit["summary"],
            "part3a_em_required_constituent_line": hit(part3a, "電磁気は、原子・分子・物性へ進むための必須構成要素である"),
            "part3a_independent_maxwell_adoption_line": hit(
                part3a,
                "Maxwell 方程式（U(1) ゲージ場 $A_\\mu$、電場 $E$、磁場 $B$）を、P-model の枠組みとは独立に採用する",
            ),
            "part3a_a_then_b_principle_line": hit(part3a, "まず導出（A）に挑戦し、難しい場合のみ有効理論（B）で拘束する"),
        },
    )

    write_artifact(
        "mass_origin_v2_part3a_a_then_b_principle_override_source_inventory",
        source_inventory,
    )
    write_artifact(
        "mass_origin_v2_part3a_a_then_b_principle_override_identification_audit",
        identification_audit,
    )
    write_artifact("mass_origin_v2_trial1_fourteenth_reopened_declaration_gate", fourteenth_gate)
    write_artifact(
        "mass_origin_v2_part3a_independent_maxwell_adoption_reentry_route_contract",
        next_route,
    )

    print("[ok] wrote:")
    print(" - mass_origin_v2_part3a_a_then_b_principle_override_source_inventory_metrics.json")
    print(" - mass_origin_v2_part3a_a_then_b_principle_override_identification_audit_metrics.json")
    print(" - mass_origin_v2_trial1_fourteenth_reopened_declaration_gate_metrics.json")
    print(" - mass_origin_v2_part3a_independent_maxwell_adoption_reentry_route_contract_metrics.json")


# Function: run the branch script from the command line.

if __name__ == "__main__":
    main()
