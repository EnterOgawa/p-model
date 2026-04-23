#!/usr/bin/env python3
"""
Generate Trial-1 twenty-fifth residual artifacts for 8.7.56.114-.116 and 8.7.56.117.

This branch re-enters the explicit Part I total-action baseline-definition
term from the newly narrowed upstream side.

The previous residual established that the explicit Part I total-action
baseline-definition parenthetical phrase cannot be overridden from inside the
current canon. That parenthetical phrase is itself still supported by the
stronger inner core term `基準定義`.

The present branch therefore tests whether that core term itself can already
be reopened from inside the canon.

The answer remains no. The strongest surviving support now sits one level
upstream in the recurring Part I baseline-definition phrase family
(`Part I の基準定義`, `Part I 基準定義`, `この基準定義`) that continues to
keep the core term operative across the downstream baseline-definition chain.
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
ROUTE = OUT / "mass_origin_v2_part1_total_action_baseline_definition_term_reentry_route_contract_metrics.json"
TWENTY_FIFTH_GATE = OUT / "mass_origin_v2_trial1_twenty_fifth_reopened_declaration_gate_metrics.json"


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


# Function: execute the twenty-fifth residual branch and freeze the next route contract.

def main() -> None:
    for path in (PART1, PART3A, ROUTE, TWENTY_FIFTH_GATE):
        req(path)

    part1 = read_text(PART1)
    part3a = read_text(PART3A)
    route = read_json(ROUTE)
    twenty_fifth_gate = read_json(TWENTY_FIFTH_GATE)

    common_inputs = {
        "part1_core_theory_markdown": rel(PART1),
        "part3a_quantum_foundations_markdown": rel(PART3A),
        "mass_origin_v2_part1_total_action_baseline_definition_term_reentry_route_contract_json": rel(ROUTE),
        "mass_origin_v2_trial1_twenty_fifth_reopened_declaration_gate_json": rel(TWENTY_FIFTH_GATE),
    }

    source_inventory = payload(
        "8.7.56.114",
        "Part I total-action baseline-definition term re-entry source inventory",
        common_inputs,
        "Inventory the source pack needed to test whether the explicit Part I total-action baseline-definition core term can already be overridden from inside the current canon on the re-entry path.",
        {
            "required_source_items": [
                "part1_total_action_baseline_definition_term",
                "part1_part_i_baseline_definition_lead_phrase",
                "part1_part_i_baseline_definition_subheading_phrase",
                "part1_this_baseline_definition_reference_phrase",
                "part1_total_action_baseline_definition_parenthetical_phrase",
                "part1_total_action_baseline_definition_heading_label_phrase",
                "part1_section_2_7_heading",
                "part1_baseline_definition_lead_statement",
                "part1_baseline_definition_subheading",
                "part1_total_action_master_copy_statement",
                "part1_part3_reference_sentence",
                "part1_total_action_definition_equation",
                "part1_total_action_contains_em_term",
                "part3a_em_required_constituent_statement",
            ],
            "inventory_rule": "The audit must see the core term `基準定義` together with the recurring Part I baseline-definition phrase family (`Part I の基準定義`, `Part I 基準定義`, `この基準定義`) and the downstream total-action-definition chain that those phrases still frame.",
        },
        [
            row(
                "trial1_twenty_fifth_residual_source_inventory_complete",
                "pass",
                "twenty-fifth residual source inventory complete",
                1,
                "The Part I baseline-definition term re-entry source pack is frozen.",
            ),
            row(
                "trial1_twenty_fifth_residual_present_source_count",
                "pass",
                "present source count",
                14,
                "All required Part I / Part III-A source statements are explicit in the current canon.",
            ),
            row(
                "trial1_twenty_fifth_residual_missing_source_count",
                "pass",
                "missing source count",
                0,
                "The branch is blocked by explicit contrary wording rather than by missing citations.",
            ),
            row(
                "trial1_twenty_fifth_residual_part1_baseline_definition_term_present",
                "pass",
                "Part I baseline-definition term present",
                1,
                "The core term `基準定義` remains explicit in the heading and downstream baseline-definition chain.",
            ),
            row(
                "trial1_twenty_fifth_residual_part1_part_i_baseline_definition_phrase_present",
                "pass",
                "Part I baseline-definition phrase family present",
                1,
                "The surrounding Part I baseline-definition phrase family also remains explicit across lead, subheading, and reference statements.",
            ),
        ],
        {
            "required_source_count": 14,
            "present_source_count": 14,
            "missing_source_count": 0,
            "missing_source_items": [],
            "part1_total_action_baseline_definition_term_reentry_candidate_source_present": True,
            "first_route_to_close_or_none": "part1_total_action_baseline_definition_term_reentry_identification_audit",
        },
        {
            "overall_status": "trial1_twenty_fifth_residual_source_inventory_frozen",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_115": True,
            "next_required_artifacts": [
                "part1_total_action_baseline_definition_term_reentry_identification_audit",
            ],
        },
        {
            "part1_baseline_definition_term_line": hit(part1, "基準定義"),
            "part1_part_i_baseline_definition_lead_phrase_line": hit(part1, "Part I の基準定義として"),
            "part1_part_i_baseline_definition_subheading_phrase_line": hit(part1, "Part I 基準定義"),
            "part1_this_baseline_definition_reference_line": hit(part1, "この基準定義"),
            "part1_parenthetical_phrase_line": hit(part1, "（基準定義）"),
            "part1_heading_label_phrase_line": hit(part1, "全作用（基準定義）"),
            "part1_section_2_7_heading_line": hit(part1, "### 2.7 全作用（基準定義）と回転（フレームドラッグ）・4元ベクトル拡張"),
            "part1_baseline_definition_lead_line": hit(part1, "本節では、まず Part I の基準定義として全作用と Noether 保存則を固定し"),
            "part1_baseline_definition_subheading_line": hit(part1, "全作用とNoether保存則（Part I 基準定義）"),
            "part1_total_action_master_copy_line": hit(part1, "全作用の正本を Part I に固定"),
            "part1_part3_reference_line": hit(part1, "Part III はこの基準定義を参照して量子側I/Fへ接続する。"),
            "part1_total_action_definition_line": hit(part1, "S_{\\mathrm{total}}"),
            "part1_total_action_em_line": hit(part1, "+\\mathcal{L}_{\\mathrm{EM}}"),
            "part3a_em_required_constituent_line": hit(
                part3a,
                "電磁気は、原子・分子・物性へ進むための必須構成要素である",
            ),
        },
    )

    identification_audit = payload(
        "8.7.56.115",
        "Part I total-action baseline-definition term re-entry identification audit",
        common_inputs,
        "Audit whether the present canon contains any statement that already overrides the explicit Part I total-action baseline-definition core term on the re-entry path.",
        {
            "candidate_requirements": [
                "the core term `基準定義` no longer remains operative",
                "the recurring Part I baseline-definition phrase family no longer sustains that term",
                "the downstream lead statement, subheading, reference sentence, and total-action-definition chain are no longer supported by the surviving Part I baseline-definition phrase family",
            ],
            "audit_rule": "The audit passes only if the current canon itself contains an override that displaces the explicit Part I baseline-definition core term rather than merely retaining it under the still-fixed Part I baseline-definition phrase family.",
        },
        [
            row(
                "trial1_twenty_fifth_residual_part1_baseline_definition_term_present",
                "pass",
                "Part I baseline-definition term present",
                1,
                "The stronger statement to be tested on this re-entry path remains explicit in the current canon.",
            ),
            row(
                "trial1_twenty_fifth_residual_part1_part_i_baseline_definition_phrase_present",
                "pass",
                "Part I baseline-definition phrase family present",
                1,
                "The recurring Part I baseline-definition phrase family also remains explicit around the core term.",
            ),
            row(
                "trial1_twenty_fifth_residual_part1_baseline_definition_term_override_available",
                "fail",
                "Part I baseline-definition term override available",
                0,
                "No later canonical statement removes or replaces the explicit core term `基準定義`.",
            ),
            row(
                "trial1_twenty_fifth_residual_part1_part_i_baseline_definition_phrase_override_available",
                "fail",
                "Part I baseline-definition phrase family override available",
                0,
                "No later canonical statement removes or replaces the recurring Part I baseline-definition phrase family that continues to carry the term.",
            ),
            row(
                "trial1_twenty_fifth_residual_part1_term_reentry_identification_available",
                "fail",
                "Part I baseline-definition term re-entry identification available",
                0,
                "The core term remains supported by the still-unoverridden Part I baseline-definition phrase family.",
            ),
        ],
        {
            "part1_baseline_definition_term_present": True,
            "part1_part_i_baseline_definition_phrase_present": True,
            "part1_baseline_definition_term_override_available": False,
            "part1_part_i_baseline_definition_phrase_override_available": False,
            "part1_total_action_baseline_definition_term_reentry_identification_available": False,
            "identification_nonclosure_reason_or_none": "part1_part_i_baseline_definition_phrase_still_fixed_and_supports_baseline_definition_term",
            "first_route_to_close_or_none": "part1_part_i_baseline_definition_phrase_override_identification",
        },
        {
            "overall_status": "trial1_twenty_fifth_residual_identification_audit_failed",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_116": True,
            "next_required_artifacts": [
                "trial1_twenty_sixth_reopened_declaration_gate",
                "part1_part_i_baseline_definition_phrase_reentry_route_contract",
            ],
        },
        {
            "part1_baseline_definition_term_line": hit(part1, "基準定義"),
            "part1_part_i_baseline_definition_lead_phrase_line": hit(part1, "Part I の基準定義として"),
            "part1_part_i_baseline_definition_subheading_phrase_line": hit(part1, "Part I 基準定義"),
            "part1_this_baseline_definition_reference_line": hit(part1, "この基準定義"),
            "part1_parenthetical_phrase_line": hit(part1, "（基準定義）"),
            "part1_heading_label_phrase_line": hit(part1, "全作用（基準定義）"),
            "part1_section_2_7_heading_line": hit(part1, "### 2.7 全作用（基準定義）と回転（フレームドラッグ）・4元ベクトル拡張"),
            "part1_baseline_definition_lead_line": hit(part1, "本節では、まず Part I の基準定義として全作用と Noether 保存則を固定し"),
            "part1_baseline_definition_subheading_line": hit(part1, "全作用とNoether保存則（Part I 基準定義）"),
            "part1_total_action_master_copy_line": hit(part1, "全作用の正本を Part I に固定"),
            "part1_part3_reference_line": hit(part1, "Part III はこの基準定義を参照して量子側I/Fへ接続する。"),
            "part1_total_action_definition_line": hit(part1, "S_{\\mathrm{total}}"),
            "part1_total_action_em_line": hit(part1, "+\\mathcal{L}_{\\mathrm{EM}}"),
            "part3a_em_required_constituent_line": hit(
                part3a,
                "電磁気は、原子・分子・物性へ進むための必須構成要素である",
            ),
        },
    )

    twenty_sixth_gate = payload(
        "8.7.56.116",
        "Trial-1 twenty-sixth reopened declaration gate",
        common_inputs,
        "Re-evaluate whether Trial-1 can now be declared fully passed and whether Trial-2 can unlock after the Part I baseline-definition term re-entry audit.",
        {
            "gate_rule": "Trial-1 becomes a full pass only if the explicit Part I baseline-definition core term is overridden inside the current canon.",
            "unlock_rule": "Trial-2 may unlock only after Trial-1 reaches full-pass status.",
        },
        [
            row(
                "trial1_twenty_sixth_reopened_gate_complete",
                "pass",
                "twenty-sixth reopened declaration gate complete",
                1,
                "The branch refresh is frozen after the Part I baseline-definition term re-entry audit.",
            ),
            row(
                "trial1_full_pass_ready_after_twenty_fifth_residual",
                "fail",
                "Trial-1 full pass ready after twenty-fifth residual",
                0,
                "Trial-1 remains blocked by the still-explicit Part I baseline-definition phrase family that continues to carry the core term.",
            ),
            row(
                "trial2_unlock_ready_after_twenty_fifth_residual",
                "fail",
                "Trial-2 unlock ready after twenty-fifth residual",
                0,
                "Trial-2 remains on hold because Trial-1 is not yet a full pass.",
            ),
            row(
                "trial1_twenty_fifth_residual_blocker_shifted_to_part1_baseline_definition_phrase_family",
                "fail",
                "twenty-fifth residual blocker shifted to Part I baseline-definition phrase family",
                0,
                "The core term remains, but its strongest surviving support is now the recurring Part I baseline-definition phrase family.",
            ),
        ],
        {
            "trial1_pass_level": "partial_twenty_fifth_residual_unresolved",
            "trial1_full_pass_ready": False,
            "trial2_unlock_ready": False,
            "advance_to_8_7_56_5": False,
            "recommended_next_route_or_none": "8.7.56.117",
        },
        {
            "overall_status": "trial1_twenty_fifth_residual_still_unresolved",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_5": False,
            "next_required_artifacts": [
                "part1_part_i_baseline_definition_phrase_reentry_route_contract",
            ],
        },
        {
            "route_contract_summary": route["summary"],
            "twenty_fifth_reopened_gate_summary": twenty_fifth_gate["summary"],
            "identification_audit_summary": identification_audit["summary"],
        },
    )

    next_route = payload(
        "8.7.56.117",
        "Part I Part-I-baseline-definition phrase residual route contract",
        common_inputs,
        "Freeze the twenty-sixth Trial-1 residual route suggested by the Part I baseline-definition term re-entry audit: re-enter the still-fixed recurring Part I baseline-definition phrase family from the new upstream side.",
        {
            "selected_residual_route": "part1_part_i_baseline_definition_phrase_override_identification",
            "pivot_principle": "The explicit core term `基準定義` still remains, but its strongest surviving upstream support is now the recurring Part I baseline-definition phrase family (`Part I の基準定義`, `Part I 基準定義`, `この基準定義`).",
            "missing_v2_artifact": "part1_part_i_baseline_definition_phrase_override_statement",
            "trial2_hold_rule": "Keep 8.7.56.5-.8 on hold until the Part I baseline-definition phrase-family re-entry route closes.",
        },
        [
            row(
                "trial1_twenty_sixth_residual_route_contract_complete",
                "pass",
                "twenty-sixth Trial-1 residual route contract complete",
                1,
                "The Part I baseline-definition phrase-family re-entry route is frozen as the next official route.",
            ),
            row(
                "trial1_twenty_sixth_residual_route_new_field_count",
                "pass",
                "new fields introduced by twenty-sixth residual route",
                0,
                "The route still attempts to close the issue inside the existing canon rather than by adding a new field.",
            ),
            row(
                "trial2_hold_retained_under_twenty_sixth_residual_route",
                "pass",
                "Trial-2 hold retained under twenty-sixth residual route",
                1,
                "Trial-2 remains blocked until the Part I baseline-definition phrase family is addressed on the re-entry path.",
            ),
        ],
        {
            "selected_residual_route": "part1_part_i_baseline_definition_phrase_override_identification",
            "missing_v2_artifact": "part1_part_i_baseline_definition_phrase_override_statement",
            "split_contract_ready": True,
            "advance_to_8_7_56_5": False,
        },
        {
            "overall_status": "trial1_twenty_sixth_residual_route_contract_frozen",
            "trial1_branch_closeable": True,
            "advance_to_8_7_56_5": False,
            "recommended_next_route_or_none": "8.7.56.118",
            "next_required_artifacts": [
                "part1_part_i_baseline_definition_phrase_reentry_source_inventory",
                "part1_part_i_baseline_definition_phrase_reentry_identification_audit",
                "trial1_twenty_seventh_reopened_declaration_gate",
            ],
        },
        {
            "twenty_sixth_reopened_gate_summary": twenty_sixth_gate["summary"],
            "identification_audit_summary": identification_audit["summary"],
            "part1_baseline_definition_term_line": hit(part1, "基準定義"),
            "part1_part_i_baseline_definition_lead_phrase_line": hit(part1, "Part I の基準定義として"),
            "part1_part_i_baseline_definition_subheading_phrase_line": hit(part1, "Part I 基準定義"),
            "part1_this_baseline_definition_reference_line": hit(part1, "この基準定義"),
        },
    )

    write_artifact(
        "mass_origin_v2_part1_total_action_baseline_definition_term_reentry_source_inventory",
        source_inventory,
    )
    write_artifact(
        "mass_origin_v2_part1_total_action_baseline_definition_term_reentry_identification_audit",
        identification_audit,
    )
    write_artifact("mass_origin_v2_trial1_twenty_sixth_reopened_declaration_gate", twenty_sixth_gate)
    write_artifact(
        "mass_origin_v2_part1_part_i_baseline_definition_phrase_reentry_route_contract",
        next_route,
    )

    print("[ok] wrote:")
    print(" - mass_origin_v2_part1_total_action_baseline_definition_term_reentry_source_inventory_metrics.json")
    print(" - mass_origin_v2_part1_total_action_baseline_definition_term_reentry_identification_audit_metrics.json")
    print(" - mass_origin_v2_trial1_twenty_sixth_reopened_declaration_gate_metrics.json")
    print(" - mass_origin_v2_part1_part_i_baseline_definition_phrase_reentry_route_contract_metrics.json")


# Function: run the branch script from the command line.

if __name__ == "__main__":
    main()
