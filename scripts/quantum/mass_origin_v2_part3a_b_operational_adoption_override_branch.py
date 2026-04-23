#!/usr/bin/env python3
"""
Generate Trial-1 tenth residual artifacts for 8.7.56.54-.56 and 8.7.56.57.

This branch narrows the Part III-A wording problem one step further.

The previous residual established that the explicit final "A reject / B adopt"
judgment remains supported by the downstream B-side operational adoption
wording. The present branch asks whether that B-side operational adoption can
already be overridden from inside the current canon.

The answer is still no, because Part III-A also carries the same choice into
the adjacent constraint sentence that says the P-to-U(1) coupling is fixed only
by the existing constraint section. The next blocker therefore narrows again to
that B-side operational constraint statement.
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
ROUTE = OUT / "mass_origin_v2_part3a_b_operational_adoption_override_route_contract_metrics.json"
TENTH_GATE = OUT / "mass_origin_v2_trial1_tenth_reopened_declaration_gate_metrics.json"


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


# Function: execute the tenth residual branch and freeze the next route contract.

def main() -> None:
    for path in (PART1, PART3A, ROUTE, TENTH_GATE):
        req(path)

    part1 = read_text(PART1)
    part3a = read_text(PART3A)
    route = read_json(ROUTE)
    tenth_gate = read_json(TENTH_GATE)

    common_inputs = {
        "part1_core_theory_markdown": rel(PART1),
        "part3a_quantum_foundations_markdown": rel(PART3A),
        "mass_origin_v2_part3a_b_operational_adoption_override_route_contract_json": rel(ROUTE),
        "mass_origin_v2_trial1_tenth_reopened_declaration_gate_json": rel(TENTH_GATE),
    }

    source_inventory = payload(
        "8.7.56.54",
        "Part III-A B-side operational adoption override source inventory",
        common_inputs,
        "Inventory the source pack needed to test whether the explicit Part III-A B-side operational adoption wording can already be overridden from inside the current canon.",
        {
            "required_source_items": [
                "part1_total_action_contains_em_statement",
                "part1_vector_total_action_contains_em_statement",
                "part3a_b_only_if_a_fails_template",
                "part3a_final_a_reject_b_adopt_judgment",
                "part3a_b_operational_adoption_statement",
                "part3a_b_operational_constraint_statement",
            ],
            "inventory_rule": "The audit must see the B-side operational adoption sentence itself, the adjacent constraint sentence that carries the same adoption forward, and the earlier final / conditional wording that frames the operational branch.",
        },
        [
            row(
                "trial1_tenth_residual_source_inventory_complete",
                "pass",
                "tenth residual source inventory complete",
                1,
                "The B-side operational adoption override source pack is frozen.",
            ),
            row(
                "trial1_tenth_residual_present_source_count",
                "pass",
                "present source count",
                6,
                "All required Part I / Part III-A source statements are explicit in the current canon.",
            ),
            row(
                "trial1_tenth_residual_missing_source_count",
                "pass",
                "missing source count",
                0,
                "The branch is blocked by explicit contrary wording rather than by missing citations.",
            ),
            row(
                "trial1_tenth_residual_b_operational_adoption_present",
                "pass",
                "B-side operational adoption present",
                1,
                "The explicit downstream B-side operational adoption sentence remains in force in the current canon.",
            ),
            row(
                "trial1_tenth_residual_b_operational_constraint_present",
                "pass",
                "B-side operational constraint present",
                1,
                "The adjacent constraint sentence still carries the same operational B-side choice forward.",
            ),
        ],
        {
            "required_source_count": 6,
            "present_source_count": 6,
            "missing_source_count": 0,
            "missing_source_items": [],
            "part3a_b_operational_adoption_override_candidate_source_present": True,
            "first_route_to_close_or_none": "part3a_b_operational_adoption_override_identification_audit",
        },
        {
            "overall_status": "trial1_tenth_residual_source_inventory_frozen",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_55": True,
            "next_required_artifacts": [
                "part3a_b_operational_adoption_override_identification_audit",
            ],
        },
        {
            "part1_total_action_em_line": hit(part1, "+\\mathcal{L}_{\\mathrm{EM}}"),
            "part1_vector_total_action_header_line": hit(part1, "\\mathcal{L}_{\\mathrm{total}}^{\\mathrm{vec}}"),
            "part3a_b_only_if_a_fails_line": hit(part3a, "A が成立しない場合のみ、U(1) を独立に採用し"),
            "part3a_final_a_reject_b_adopt_line": hit(part3a, "**A棄却、B採用**"),
            "part3a_b_adoption_line": hit(part3a, "標準 U(1) を独立の有効理論として採用し"),
            "part3a_b_constraint_line": hit(part3a, "P との結合は §\\ref{sec:2-6-3-s2-6-3} の制約で拘束する"),
        },
    )

    identification_audit = payload(
        "8.7.56.55",
        "Part III-A B-side operational adoption override identification audit",
        common_inputs,
        "Audit whether the present canon contains any statement that already overrides the explicit Part III-A B-side operational adoption wording.",
        {
            "candidate_requirements": [
                "B-side operational adoption wording no longer remains operative",
                "B-side operational constraint wording no longer remains operative",
                "the earlier conditional A-to-B template explicitly reopens the downstream B-side wording",
            ],
            "audit_rule": "The audit passes only if the current canon itself contains an override that displaces the explicit B-side operational adoption sentence rather than merely surrounding it with softer conditional language.",
        },
        [
            row(
                "trial1_tenth_residual_part3a_b_operational_adoption_present",
                "pass",
                "Part III-A B-side operational adoption present",
                1,
                "The downstream B-side operational adoption sentence to be overridden remains explicit in the current canon.",
            ),
            row(
                "trial1_tenth_residual_part3a_b_operational_constraint_present",
                "pass",
                "Part III-A B-side operational constraint present",
                1,
                "The adjacent B-side constraint sentence still operationalizes the same adoption choice.",
            ),
            row(
                "trial1_tenth_residual_part3a_b_operational_adoption_override_available",
                "fail",
                "Part III-A B-side operational adoption override available",
                0,
                "No later canonical statement displaces the explicit B-side operational adoption sentence.",
            ),
            row(
                "trial1_tenth_residual_part3a_b_operational_constraint_override_available",
                "fail",
                "Part III-A B-side operational constraint override available",
                0,
                "The adjacent constraint sentence still says that the P-to-U(1) coupling is fixed only by the existing constraint section.",
            ),
            row(
                "trial1_tenth_residual_part3a_b_operational_adoption_override_identification_available",
                "fail",
                "Part III-A B-side operational adoption override identification available",
                0,
                "The B-side operational adoption remains supported by the still-unoverridden adjacent B-side constraint sentence.",
            ),
        ],
        {
            "part3a_b_operational_adoption_present": True,
            "part3a_b_operational_constraint_present": True,
            "part3a_b_operational_adoption_override_available": False,
            "part3a_b_operational_constraint_override_available": False,
            "part3a_b_operational_adoption_override_identification_available": False,
            "identification_nonclosure_reason_or_none": "part3a_b_operational_constraint_still_fixed_and_supports_b_operational_adoption",
            "first_route_to_close_or_none": "part3a_b_operational_constraint_override_identification",
        },
        {
            "overall_status": "trial1_tenth_residual_identification_audit_failed",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_56": True,
            "next_required_artifacts": [
                "trial1_eleventh_reopened_declaration_gate",
                "part3a_b_operational_constraint_override_route_contract",
            ],
        },
        {
            "part3a_b_only_if_a_fails_line": hit(part3a, "A が成立しない場合のみ、U(1) を独立に採用し"),
            "part3a_final_a_reject_b_adopt_line": hit(part3a, "**A棄却、B採用**"),
            "part3a_b_adoption_line": hit(part3a, "標準 U(1) を独立の有効理論として採用し"),
            "part3a_b_constraint_line": hit(part3a, "P との結合は §\\ref{sec:2-6-3-s2-6-3} の制約で拘束する"),
        },
    )

    eleventh_gate = payload(
        "8.7.56.56",
        "Trial-1 eleventh reopened declaration gate",
        common_inputs,
        "Re-evaluate whether Trial-1 can now be declared fully passed and whether Trial-2 can unlock after the B-side operational adoption override audit.",
        {
            "gate_rule": "Trial-1 becomes a full pass only if the explicit Part III-A B-side operational adoption wording is overridden inside the current canon.",
            "unlock_rule": "Trial-2 may unlock only after Trial-1 reaches full-pass status.",
        },
        [
            row(
                "trial1_eleventh_reopened_gate_complete",
                "pass",
                "eleventh reopened declaration gate complete",
                1,
                "The branch refresh is frozen after the B-side operational adoption override audit.",
            ),
            row(
                "trial1_full_pass_ready_after_tenth_residual",
                "fail",
                "Trial-1 full pass ready after tenth residual",
                0,
                "Trial-1 remains blocked by the still-explicit B-side operational adoption and adjacent B-side constraint wording in Part III-A.",
            ),
            row(
                "trial2_unlock_ready_after_tenth_residual",
                "fail",
                "Trial-2 unlock ready after tenth residual",
                0,
                "Trial-2 remains on hold because Trial-1 is not yet a full pass.",
            ),
            row(
                "trial1_tenth_residual_blocker_shifted_to_b_operational_constraint",
                "fail",
                "tenth residual blocker shifted to B-side operational constraint",
                0,
                "The B-side operational adoption remains, but its strongest surviving downstream support is now the adjacent B-side constraint sentence.",
            ),
        ],
        {
            "trial1_pass_level": "partial_tenth_residual_unresolved",
            "trial1_full_pass_ready": False,
            "trial2_unlock_ready": False,
            "advance_to_8_7_56_5": False,
            "recommended_next_route_or_none": "8.7.56.57",
        },
        {
            "overall_status": "trial1_tenth_residual_still_unresolved",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_5": False,
            "next_required_artifacts": [
                "part3a_b_operational_constraint_override_route_contract",
            ],
        },
        {
            "route_contract_summary": route["summary"],
            "tenth_reopened_gate_summary": tenth_gate["summary"],
            "identification_audit_summary": identification_audit["summary"],
        },
    )

    next_route = payload(
        "8.7.56.57",
        "Part III-A B-side operational constraint override residual route contract",
        common_inputs,
        "Freeze the eleventh Trial-1 residual route suggested by the B-side operational adoption audit: test whether the explicit Part III-A B-side constraint wording can be overridden from inside the current canon.",
        {
            "selected_residual_route": "part3a_b_operational_constraint_override_identification",
            "pivot_principle": "The B-side operational adoption remains explicit, but its strongest surviving downstream support is the adjacent constraint sentence that still fixes the P-to-U(1) coupling only by the existing constraint section.",
            "missing_v2_artifact": "part3a_b_operational_constraint_override_statement",
            "trial2_hold_rule": "Keep 8.7.56.5-.8 on hold until the Part III-A B-side operational constraint override route closes.",
        },
        [
            row(
                "trial1_eleventh_residual_route_contract_complete",
                "pass",
                "eleventh Trial-1 residual route contract complete",
                1,
                "The Part III-A B-side operational constraint override route is frozen as the next official route.",
            ),
            row(
                "trial1_eleventh_residual_route_new_field_count",
                "pass",
                "new fields introduced by eleventh residual route",
                0,
                "The route still attempts to close the issue inside the existing canon rather than by adding a new field.",
            ),
            row(
                "trial2_hold_retained_under_eleventh_residual_route",
                "pass",
                "Trial-2 hold retained under eleventh residual route",
                1,
                "Trial-2 remains blocked until the B-side operational constraint wording is addressed.",
            ),
        ],
        {
            "selected_residual_route": "part3a_b_operational_constraint_override_identification",
            "missing_v2_artifact": "part3a_b_operational_constraint_override_statement",
            "split_contract_ready": True,
            "advance_to_8_7_56_5": False,
        },
        {
            "overall_status": "trial1_eleventh_residual_route_contract_frozen",
            "trial1_branch_closeable": True,
            "advance_to_8_7_56_5": False,
            "recommended_next_route_or_none": "8.7.56.58",
            "next_required_artifacts": [
                "part3a_b_operational_constraint_override_source_inventory",
                "part3a_b_operational_constraint_override_identification_audit",
                "trial1_twelfth_reopened_declaration_gate",
            ],
        },
        {
            "eleventh_reopened_gate_summary": eleventh_gate["summary"],
            "identification_audit_summary": identification_audit["summary"],
            "part3a_b_only_if_a_fails_line": hit(part3a, "A が成立しない場合のみ、U(1) を独立に採用し"),
            "part3a_b_adoption_line": hit(part3a, "標準 U(1) を独立の有効理論として採用し"),
            "part3a_b_constraint_line": hit(part3a, "P との結合は §\\ref{sec:2-6-3-s2-6-3} の制約で拘束する"),
        },
    )

    write_artifact(
        "mass_origin_v2_part3a_b_operational_adoption_override_source_inventory",
        source_inventory,
    )
    write_artifact(
        "mass_origin_v2_part3a_b_operational_adoption_override_identification_audit",
        identification_audit,
    )
    write_artifact("mass_origin_v2_trial1_eleventh_reopened_declaration_gate", eleventh_gate)
    write_artifact(
        "mass_origin_v2_part3a_b_operational_constraint_override_route_contract",
        next_route,
    )

    print("[ok] wrote:")
    print(" - mass_origin_v2_part3a_b_operational_adoption_override_source_inventory_metrics.json")
    print(" - mass_origin_v2_part3a_b_operational_adoption_override_identification_audit_metrics.json")
    print(" - mass_origin_v2_trial1_eleventh_reopened_declaration_gate_metrics.json")
    print(" - mass_origin_v2_part3a_b_operational_constraint_override_route_contract_metrics.json")


# Function: run the branch script from the command line.

if __name__ == "__main__":
    main()
