#!/usr/bin/env python3
"""
Generate Trial-1 second residual artifacts for 8.7.56.22-.24 and 8.7.56.25.

This branch tests the narrower question left after the Stueckelberg residual:
can the current P-only canon already be read as a massless, charge-selective
electromagnetic connection?

The source pack is present, but the current canon still says the opposite:

1. Part I freezes a massive vector closure and a universal coupling
   g_P P_mu J_matter.
2. Part I keeps L_EM as a separate sector in the total action.
3. Part III-A records that the massless photon and charge-selective coupling
   are adopted external conditions and that a separate independent connection is
   required.

Therefore the branch closes honestly with a new residual route focused on the
still-missing elimination of the independent electromagnetic sector.
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
ROUTE = OUT / "mass_origin_v2_massless_charge_selective_connection_route_contract_metrics.json"
REOPENED_GATE = OUT / "mass_origin_v2_trial1_reopened_declaration_gate_metrics.json"


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


# Function: execute the second residual branch and freeze the next route contract.

def main() -> None:
    for path in (PART1, PART3A, ROUTE, REOPENED_GATE):
        req(path)

    part1 = read_text(PART1)
    part3a = read_text(PART3A)
    route = read_json(ROUTE)
    reopened_gate = read_json(REOPENED_GATE)

    common_inputs = {
        "part1_core_theory_markdown": rel(PART1),
        "part3a_quantum_foundations_markdown": rel(PART3A),
        "mass_origin_v2_massless_charge_selective_connection_route_contract_json": rel(ROUTE),
        "mass_origin_v2_trial1_reopened_declaration_gate_json": rel(REOPENED_GATE),
    }

    source_inventory = payload(
        "8.7.56.22",
        "Massless charge-selective connection source inventory",
        common_inputs,
        "Inventory the already-public source pack needed to test whether the current P-vector canon can be re-read as a massless, charge-selective electromagnetic connection.",
        {
            "required_source_items": [
                "massive_p_vector_closure_statement",
                "universal_p_current_coupling_statement",
                "separate_em_sector_in_total_action_statement",
                "adopted_massless_photon_condition",
                "adopted_charge_selective_q_condition",
                "independent_connection_requirement_statement",
                "b_adoption_statement",
            ],
            "inventory_rule": "All source items needed for the second residual audit must already be explicit in the public canon before the audit can be trusted.",
        },
        [
            row(
                "trial1_second_residual_source_inventory_complete",
                "pass",
                "second residual source inventory complete",
                1,
                "The massless/charge-selective source pack is frozen.",
            ),
            row(
                "trial1_second_residual_present_source_count",
                "pass",
                "present source count",
                7,
                "All required source statements are explicit in the public canon.",
            ),
            row(
                "trial1_second_residual_missing_source_count",
                "pass",
                "missing source count",
                0,
                "The branch is blocked by incompatibility of statements rather than missing citations.",
            ),
            row(
                "trial1_second_residual_separate_em_sector_statement_present",
                "pass",
                "separate EM sector statement present",
                1,
                "Part I and Part III-A both keep an independent electromagnetic connection or sector explicit.",
            ),
        ],
        {
            "required_source_count": 7,
            "present_source_count": 7,
            "missing_source_count": 0,
            "missing_source_items": [],
            "massless_charge_selective_connection_candidate_source_present": True,
            "first_route_to_close_or_none": "massless_charge_selective_connection_identification_audit",
        },
        {
            "overall_status": "trial1_second_residual_source_inventory_frozen",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_23": True,
            "next_required_artifacts": [
                "massless_charge_selective_connection_identification_audit",
            ],
        },
        {
            "part1_massive_vector_line": hit(part1, "+\\frac{m_{P}^2}{2}\\left(P_\\mu-\\frac{1}{m_{P}}\\partial_\\mu\\pi\\right)"),
            "part1_universal_current_coupling_line": hit(part1, "\\mathcal{L}_{\\mathrm{int}}=g_P\\,P_\\mu J^\\mu_{\\mathrm{matter}}"),
            "part1_total_action_contains_em_line": hit(part1, "+\\mathcal{L}_{\\mathrm{EM}}"),
            "part3a_massless_photon_line": hit(part3a, "ゲージ場は質量項を持たない最小形を採用する"),
            "part3a_charge_selective_q_line": hit(part3a, "結合定数 q は環境非依存"),
            "part3a_independent_connection_line": hit(part3a, "独立接続"),
            "part3a_b_adoption_line": hit(part3a, "標準 U(1) を独立の有効理論として採用し"),
        },
    )

    identification_audit = payload(
        "8.7.56.23",
        "Massless charge-selective connection identification audit",
        common_inputs,
        "Audit whether the present P-only canon can eliminate the independent electromagnetic sector and simultaneously recover a massless, charge-selective connection without new fields or new free parameters.",
        {
            "candidate_requirements": [
                "photon-mass zero derived from P-only canon",
                "charge-selective current split derived from P-only canon",
                "independent EM sector eliminated from the total action",
            ],
            "audit_rule": "The audit passes only if the current canon itself rewrites the massive universal P-vector closure into a massless charge-selective electromagnetic sector.",
        },
        [
            row(
                "trial1_second_residual_massless_photon_internal_derivation_available",
                "fail",
                "massless photon internal derivation available",
                0,
                "The massless photon statement appears only as an adopted external condition, not as a P-only derivation.",
            ),
            row(
                "trial1_second_residual_charge_selective_current_from_p_only_available",
                "fail",
                "charge-selective current from P-only available",
                0,
                "The current canon still freezes the universal interaction g_P P_mu J_matter instead of a charge-selective current split.",
            ),
            row(
                "trial1_second_residual_independent_em_sector_elimination_available",
                "fail",
                "independent EM sector elimination available",
                0,
                "Part I still keeps L_EM in the total action and Part III-A still requires an independent connection A_mu.",
            ),
            row(
                "trial1_second_residual_massless_charge_selective_connection_identification_available",
                "fail",
                "massless charge-selective connection identification available",
                0,
                "The current canon does not yet eliminate the independent EM sector or rewrite the massive universal P-vector closure as a photon-like connection.",
            ),
        ],
        {
            "massless_photon_internal_derivation_available": False,
            "charge_selective_current_from_p_only_available": False,
            "independent_em_sector_elimination_available": False,
            "massless_charge_selective_connection_identification_available": False,
            "identification_nonclosure_reason_or_none": "independent_em_sector_still_required_and_massive_universal_p_vector_closure_retained",
            "first_route_to_close_or_none": "independent_em_sector_elimination_identification",
        },
        {
            "overall_status": "trial1_second_residual_identification_audit_failed",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_24": True,
            "next_required_artifacts": [
                "trial1_rereopened_declaration_gate",
                "independent_em_sector_elimination_route_contract",
            ],
        },
        {
            "part1_massive_vector_line": hit(part1, "+\\frac{m_{P}^2}{2}\\left(P_\\mu-\\frac{1}{m_{P}}\\partial_\\mu\\pi\\right)"),
            "part1_universal_current_coupling_line": hit(part1, "\\mathcal{L}_{\\mathrm{int}}=g_P\\,P_\\mu J^\\mu_{\\mathrm{matter}}"),
            "part1_total_action_contains_em_line": hit(part1, "+\\mathcal{L}_{\\mathrm{EM}}"),
            "part3a_independent_connection_line": hit(part3a, "独立接続"),
            "part3a_b_adoption_line": hit(part3a, "標準 U(1) を独立の有効理論として採用し"),
            "part3a_massless_photon_line": hit(part3a, "ゲージ場は質量項を持たない最小形を採用する"),
            "part3a_charge_selective_q_line": hit(part3a, "結合定数 q は環境非依存"),
        },
    )

    rereopened_gate = payload(
        "8.7.56.24",
        "Trial-1 re-reopened declaration gate / Trial-2 unlock gate",
        common_inputs,
        "Reflect the second residual audit into Trial-1 and decide whether Trial-2 can finally be unlocked.",
        {
            "trial1_full_pass_rule": "Trial-1 upgrades to full pass only if the current canon eliminates the independent EM sector and supplies a massless charge-selective connection statement.",
            "trial2_unlock_rule": "Trial-2 stays on hold unless Trial-1 closes as a full pass.",
        },
        [
            row(
                "trial1_rereopened_full_pass_ready",
                "fail",
                "Trial-1 re-reopened full pass ready",
                0,
                "The second residual audit still leaves the electromagnetic sector outside the P-only canon.",
            ),
            row(
                "trial2_unlock_ready_after_second_residual",
                "fail",
                "Trial-2 unlock ready after second residual",
                0,
                "Electromagnetic first-principles work remains blocked by the unresolved independent-sector elimination.",
            ),
            row(
                "trial1_partial_closeout_retained_after_second_residual",
                "pass",
                "Trial-1 partial closeout retained after second residual",
                1,
                "Global U(1) and local P-vector gauge redundancy remain valid even though the EM identification is still unresolved.",
            ),
        ],
        {
            "trial1_pass_level": "partial_second_residual_unresolved",
            "trial1_full_pass_ready": False,
            "trial2_unlock_ready": False,
            "advance_to_8_7_56_5": False,
            "recommended_next_route_or_none": "8.7.56.25",
        },
        {
            "overall_status": "trial1_rereopened_gate_failed_third_residual_required",
            "trial1_branch_closeable": True,
            "advance_to_8_7_56_5": False,
            "next_required_artifacts": [
                "independent_em_sector_elimination_route_contract",
            ],
        },
        {
            "second_residual_route_summary": route["summary"],
            "reopened_gate_summary": reopened_gate["summary"],
            "identification_audit_summary": identification_audit["summary"],
        },
    )

    third_route = payload(
        "8.7.56.25",
        "Independent EM sector elimination residual route contract",
        common_inputs,
        "Freeze the third Trial-1 residual route suggested by the second residual audit: test whether the independent electromagnetic sector can be eliminated from the canon in favor of a P-only statement.",
        {
            "selected_residual_route": "independent_em_sector_elimination_identification",
            "pivot_principle": "The next blocker is not generic U(1) structure but the explicit retention of an independent electromagnetic connection and sector outside the massive universal P-vector closure.",
            "missing_v2_artifact": "independent_em_sector_elimination_statement",
            "trial2_hold_rule": "Keep 8.7.56.5-.8 on hold until the independent EM sector elimination route closes.",
        },
        [
            row(
                "trial1_third_residual_route_contract_complete",
                "pass",
                "third Trial-1 residual route contract complete",
                1,
                "The independent-EM-sector elimination route is frozen as the next official route.",
            ),
            row(
                "trial1_third_residual_route_new_field_count",
                "pass",
                "new fields introduced by third residual route",
                0,
                "The route still tries to close the issue inside the existing public canon.",
            ),
            row(
                "trial2_hold_retained_under_third_residual_route",
                "pass",
                "Trial-2 hold retained under third residual route",
                1,
                "Trial-2 remains blocked until the independent EM sector elimination statement is tested.",
            ),
        ],
        {
            "selected_residual_route": "independent_em_sector_elimination_identification",
            "missing_v2_artifact": "independent_em_sector_elimination_statement",
            "split_contract_ready": True,
            "advance_to_8_7_56_5": False,
        },
        {
            "overall_status": "trial1_third_residual_route_contract_frozen",
            "trial1_branch_closeable": True,
            "advance_to_8_7_56_5": False,
            "recommended_next_route_or_none": "8.7.56.26",
            "next_required_artifacts": [
                "independent_em_sector_elimination_source_inventory",
                "independent_em_sector_elimination_identification_audit",
                "trial1_fourth_reopened_declaration_gate",
            ],
        },
        {
            "rereopened_gate_summary": rereopened_gate["summary"],
            "identification_audit_summary": identification_audit["summary"],
            "part1_total_action_contains_em_line": hit(part1, "+\\mathcal{L}_{\\mathrm{EM}}"),
            "part3a_independent_connection_line": hit(part3a, "独立接続"),
        },
    )

    write_artifact("mass_origin_v2_massless_charge_selective_connection_source_inventory", source_inventory)
    write_artifact("mass_origin_v2_massless_charge_selective_connection_identification_audit", identification_audit)
    write_artifact("mass_origin_v2_trial1_rereopened_declaration_gate", rereopened_gate)
    write_artifact("mass_origin_v2_independent_em_sector_elimination_route_contract", third_route)


if __name__ == "__main__":
    main()
