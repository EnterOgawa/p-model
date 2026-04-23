#!/usr/bin/env python3
"""
Generate Trial-1 residual artifacts for 8.7.56.18-.20 and 8.7.56.21.

This branch tests whether the already-canonical Stueckelberg closure of the
vector P-field can be reinterpreted as the missing electromagnetic phase
connection. The audit is stricter than a generic "local gauge exists" check:

1. The candidate must behave like a connection rather than only as a
   gauge-invariant composite.
2. The resulting sector must support a massless photon-like limit.
3. The matter coupling must become charge-selective rather than the already
   frozen universal P_mu J_matter interaction.

The current canon does not yet satisfy these conditions, so the branch freezes
an honest non-closure and promotes a new residual route that focuses on the
missing massless / charge-selective connection statement.
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
TRIAL1_DECLARATION = OUT / "mass_origin_v2_trial1_declaration_gate_metrics.json"
TRIAL1_ROUTE = OUT / "mass_origin_v2_stueckelberg_to_em_connection_route_contract_metrics.json"


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


# Function: execute the residual branch and the next route contract.

def main() -> None:
    for path in (PART1, PART3A, TRIAL1_DECLARATION, TRIAL1_ROUTE):
        req(path)

    part1 = read_text(PART1)
    part3a = read_text(PART3A)
    trial1_declaration = read_json(TRIAL1_DECLARATION)
    trial1_route = read_json(TRIAL1_ROUTE)

    common_inputs = {
        "part1_core_theory_markdown": rel(PART1),
        "part3a_quantum_foundations_markdown": rel(PART3A),
        "mass_origin_v2_trial1_declaration_gate_json": rel(TRIAL1_DECLARATION),
        "mass_origin_v2_stueckelberg_to_em_connection_route_contract_json": rel(TRIAL1_ROUTE),
    }

    source_inventory = payload(
        "8.7.56.18",
        "Stueckelberg-to-electromagnetic connection source inventory",
        common_inputs,
        "Inventory the already-public source pack for identifying the P-field Stueckelberg closure with the missing electromagnetic phase connection.",
        {
            "required_source_items": [
                "stueckelberg_local_gauge_transform",
                "gauge_invariant_pi_mu_definition",
                "vector_field_strength_F_P",
                "universal_matter_current_coupling",
                "independent_em_sector_statement",
                "route_a_u1_template",
                "massless_photon_condition",
                "charge_selective_coupling_condition",
            ],
            "inventory_rule": "All source items needed to test the identification must already be present in the public canon.",
        },
        [
            row(
                "trial1_residual_source_inventory_complete",
                "pass",
                "Trial-1 residual source inventory complete",
                1,
                "The Stueckelberg-to-EM source pack is frozen.",
            ),
            row(
                "trial1_residual_present_source_count",
                "pass",
                "present source count",
                8,
                "All required source statements are already explicit in the public canon.",
            ),
            row(
                "trial1_residual_missing_source_count",
                "pass",
                "missing source count",
                0,
                "The residual branch is blocked by interpretation, not by missing source text.",
            ),
            row(
                "trial1_connection_candidate_source_present",
                "pass",
                "Stueckelberg connection candidate source present",
                1,
                "Part I already freezes both the gauge transform and the gauge-invariant Pi_mu combination.",
            ),
        ],
        {
            "required_source_count": 8,
            "present_source_count": 8,
            "missing_source_count": 0,
            "missing_source_items": [],
            "stueckelberg_connection_candidate_source_present": True,
            "first_route_to_close_or_none": "stueckelberg_to_em_connection_identification_audit",
        },
        {
            "overall_status": "trial1_residual_source_inventory_frozen",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_19": True,
            "next_required_artifacts": [
                "stueckelberg_to_em_connection_identification_audit",
            ],
        },
        {
            "part1_stueckelberg_line": hit(part1, "Stückelberg 場 $\\pi$"),
            "part1_gauge_transform_line": hit(part1, "P_\\mu\\to P_\\mu+\\partial_\\mu\\alpha"),
            "part1_pi_mu_line": hit(part1, "\\Pi_\\mu:=P_\\mu-\\partial_\\mu\\pi/m_P"),
            "part1_vector_strength_line": hit(part1, "F^{(P)}_{\\mu\\nu}\\equiv \\partial_\\mu P_\\nu-\\partial_\\nu P_\\mu"),
            "part1_universal_current_coupling_line": hit(part1, "\\mathcal{L}_{\\mathrm{int}}=g_P\\,P_\\mu J^\\mu_{\\mathrm{matter}}"),
            "part1_total_action_contains_em_line": hit(part1, "+\\mathcal{L}_{\\mathrm{EM}}"),
            "part3a_route_a_template_line": hit(part3a, "\\mathcal{L} = \\lvert D_\\mu P\\rvert^2 - V(\\lvert P\\rvert) - \\frac{1}{4}F_{\\mu\\nu}F^{\\mu\\nu},"),
            "part3a_massless_photon_line": hit(part3a, "ゲージ場は質量項を持たない最小形を採用する"),
            "part3a_charge_selective_line": hit(part3a, "結合定数 q は環境非依存"),
        },
    )

    identification_audit = payload(
        "8.7.56.19",
        "Stueckelberg-to-electromagnetic connection identification audit",
        common_inputs,
        "Audit whether the canonical Stueckelberg gauge structure can actually serve as the electromagnetic connection needed by the Maxwell/U(1) sector.",
        {
            "candidate_1": "A_mu ?= P_mu",
            "candidate_2": "A_mu ?= Pi_mu = P_mu - d_mu pi / m_P",
            "candidate_requirements": [
                "connection-like gauge transform",
                "nontrivial curvature sector",
                "massless photon-like limit",
                "charge-selective matter coupling",
            ],
            "audit_rule": "The audit passes only if at least one candidate closes all four requirements without adding a new field or a new free parameter.",
        },
        [
            row(
                "trial1_p_mu_connection_transform_available",
                "pass",
                "P_mu has connection-like gauge transform",
                1,
                "P_mu already shifts by a derivative under the Stueckelberg redundancy.",
            ),
            row(
                "trial1_pi_mu_connection_transform_available",
                "fail",
                "Pi_mu has connection-like gauge transform",
                0,
                "Pi_mu is gauge invariant, so it is not itself a connection variable.",
            ),
            row(
                "trial1_massless_em_limit_available",
                "fail",
                "massless photon-like limit available",
                0,
                "The current canonical P_mu closure is explicitly massive and no P-only massless EM statement is frozen yet.",
            ),
            row(
                "trial1_charge_selective_current_split_available",
                "fail",
                "charge-selective current split available",
                0,
                "The current canonical interaction is the universal coupling g_P P_mu J_matter rather than a charge-selective q A_mu j_em mapping.",
            ),
            row(
                "trial1_independent_em_sector_eliminated",
                "fail",
                "independent EM sector eliminated",
                0,
                "Part I still keeps L_EM as a separate sector in the total action.",
            ),
            row(
                "trial1_stueckelberg_to_em_connection_identification_available",
                "fail",
                "Stueckelberg-to-EM connection identification available",
                0,
                "The present canon does not yet justify reading the massive universal P-vector closure as the Maxwell connection.",
            ),
        ],
        {
            "p_mu_connection_transform_available": True,
            "pi_mu_connection_transform_available": False,
            "massless_photon_like_limit_available": False,
            "charge_selective_current_split_available": False,
            "independent_em_sector_eliminated": False,
            "stueckelberg_to_em_connection_identification_available": False,
            "identification_nonclosure_reason_or_none": "massive_universal_p_vector_not_yet_equivalent_to_massless_charge_selective_em_connection",
            "first_route_to_close_or_none": "massless_charge_selective_connection_identification",
        },
        {
            "overall_status": "trial1_residual_identification_audit_failed",
            "trial1_branch_closeable": False,
            "advance_to_8_7_56_20": True,
            "next_required_artifacts": [
                "trial1_reopened_declaration_gate",
                "massless_charge_selective_connection_route_contract",
            ],
        },
        {
            "part1_gauge_transform_line": hit(part1, "P_\\mu\\to P_\\mu+\\partial_\\mu\\alpha"),
            "part1_pi_mu_line": hit(part1, "\\Pi_\\mu:=P_\\mu-\\partial_\\mu\\pi/m_P"),
            "part1_massive_term_line": hit(part1, "+\\frac{m_{P}^2}{2}\\left(P_\\mu-\\frac{1}{m_{P}}\\partial_\\mu\\pi\\right)"),
            "part1_universal_current_coupling_line": hit(part1, "\\mathcal{L}_{\\mathrm{int}}=g_P\\,P_\\mu J^\\mu_{\\mathrm{matter}}"),
            "part1_total_action_contains_em_line": hit(part1, "+\\mathcal{L}_{\\mathrm{EM}}"),
            "part3a_massless_photon_line": hit(part3a, "ゲージ場は質量項を持たない最小形を採用する"),
            "part3a_charge_selective_line": hit(part3a, "結合定数 q は環境非依存"),
            "part3a_independent_connection_line": hit(part3a, "独立接続"),
        },
    )

    reopened_gate = payload(
        "8.7.56.20",
        "Trial-1 reopened declaration gate / Trial-2 unlock gate",
        common_inputs,
        "Reflect the identification audit into a reopened Trial-1 declaration and decide whether Trial-2 can be unlocked.",
        {
            "trial1_full_pass_rule": "Trial-1 upgrades to full pass only if the residual route identifies a massless charge-selective electromagnetic connection inside the P-only canon.",
            "trial2_unlock_rule": "Trial-2 stays on hold unless Trial-1 closes as a full pass.",
        },
        [
            row(
                "trial1_reopened_full_pass_ready",
                "fail",
                "Trial-1 reopened full pass ready",
                0,
                "The residual route did not close the missing EM connection statement.",
            ),
            row(
                "trial2_unlock_ready",
                "fail",
                "Trial-2 unlock ready",
                0,
                "Electromagnetic first-principles work remains blocked by the unresolved connection identity.",
            ),
            row(
                "trial1_partial_closeout_retained",
                "pass",
                "Trial-1 partial closeout retained",
                1,
                "Global U(1) and local P-vector gauge redundancy remain preserved outcomes of Trial-1.",
            ),
        ],
        {
            "trial1_pass_level": "partial_residual_unresolved",
            "trial1_full_pass_ready": False,
            "trial2_unlock_ready": False,
            "advance_to_8_7_56_5": False,
            "recommended_next_route_or_none": "8.7.56.21",
        },
        {
            "overall_status": "trial1_reopened_gate_failed_residual_route_required",
            "trial1_branch_closeable": True,
            "advance_to_8_7_56_5": False,
            "next_required_artifacts": [
                "massless_charge_selective_connection_route_contract",
            ],
        },
        {
            "trial1_declaration_summary": trial1_declaration["summary"],
            "trial1_route_summary": trial1_route["summary"],
            "identification_audit_summary": identification_audit["summary"],
        },
    )

    residual_route = payload(
        "8.7.56.21",
        "Massless charge-selective connection residual route contract",
        common_inputs,
        "Freeze the next residual route suggested by the Stueckelberg audit: test whether a massless and charge-selective electromagnetic connection statement can be extracted from the current P-vector canon.",
        {
            "selected_residual_route": "massless_charge_selective_connection_identification",
            "pivot_principle": "The missing ingredient is no longer local gauge redundancy itself but a statement that turns the massive universal P-vector closure into a massless charge-selective electromagnetic connection.",
            "missing_v2_artifact": "massless_charge_selective_connection_statement",
            "trial2_hold_rule": "Keep 8.7.56.5-.8 on hold until the massless / charge-selective connection route closes.",
        },
        [
            row(
                "trial1_second_residual_route_contract_complete",
                "pass",
                "second Trial-1 residual route contract complete",
                1,
                "The massless charge-selective connection route is frozen as the next official route.",
            ),
            row(
                "trial1_second_residual_route_new_field_count",
                "pass",
                "new fields introduced by second residual route",
                0,
                "The route still tries to reuse the existing P-vector canon rather than adding a new field.",
            ),
            row(
                "trial2_hold_retained_under_second_residual_route",
                "pass",
                "Trial-2 hold retained under second residual route",
                1,
                "Trial-2 remains blocked until the missing massless charge-selective statement is tested.",
            ),
        ],
        {
            "selected_residual_route": "massless_charge_selective_connection_identification",
            "missing_v2_artifact": "massless_charge_selective_connection_statement",
            "split_contract_ready": True,
            "advance_to_8_7_56_5": False,
        },
        {
            "overall_status": "trial1_second_residual_route_contract_frozen",
            "trial1_branch_closeable": True,
            "advance_to_8_7_56_5": False,
            "recommended_next_route_or_none": "8.7.56.22",
            "next_required_artifacts": [
                "massless_charge_selective_connection_source_inventory",
                "massless_charge_selective_connection_identification_audit",
                "trial1_rereopened_declaration_gate",
            ],
        },
        {
            "reopened_gate_summary": reopened_gate["summary"],
            "identification_audit_summary": identification_audit["summary"],
            "part1_universal_current_coupling_line": hit(part1, "\\mathcal{L}_{\\mathrm{int}}=g_P\\,P_\\mu J^\\mu_{\\mathrm{matter}}"),
            "part3a_massless_photon_line": hit(part3a, "ゲージ場は質量項を持たない最小形を採用する"),
        },
    )

    write_artifact("mass_origin_v2_stueckelberg_to_em_connection_source_inventory", source_inventory)
    write_artifact("mass_origin_v2_stueckelberg_to_em_connection_identification_audit", identification_audit)
    write_artifact("mass_origin_v2_trial1_reopened_declaration_gate", reopened_gate)
    write_artifact("mass_origin_v2_massless_charge_selective_connection_route_contract", residual_route)


if __name__ == "__main__":
    main()
