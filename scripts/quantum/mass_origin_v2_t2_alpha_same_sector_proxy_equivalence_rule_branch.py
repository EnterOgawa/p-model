#!/usr/bin/env python3
"""Generate 8.7.56.739-.742 Trial-2 numeric alpha same-sector-proxy-equivalence-rule artifacts."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"

PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"

PRIOR_INVENTORY = OUT / "mass_origin_v2_trial2_numeric_alpha_chi_star_or_same_sector_proxy_numeric_value_source_inventory_metrics.json"
PRIOR_AUDIT = OUT / "mass_origin_v2_trial2_numeric_alpha_chi_star_or_same_sector_proxy_numeric_value_audit_metrics.json"
PRIOR_GATE = OUT / "mass_origin_v2_trial2_numeric_alpha_chi_star_or_same_sector_proxy_numeric_value_declaration_gate_metrics.json"
PRIOR_ROUTE = OUT / "mass_origin_v2_t2_alpha_route_contract_eighty_first_refresh_metrics.json"
SAME_SECTOR_PROXY_EQUIVALENCE_AUDIT = OUT / "mass_origin_same_sector_proxy_equivalence_audit_metrics.json"
SAME_SECTOR_EQUIVALENCE_SOURCE_INVENTORY = OUT / "mass_origin_same_sector_equivalence_source_inventory_metrics.json"
SAME_SECTOR_EQUIVALENCE_WORDING_AUDIT = OUT / "mass_origin_same_sector_equivalence_wording_audit_metrics.json"
SAME_SECTOR_EQUIVALENCE_STATEMENT_ROUTE_CONTRACT = (
    OUT / "mass_origin_same_sector_equivalence_statement_route_contract_metrics.json"
)
QBALL_SPIN_ORBIT = OUT / "mass_origin_vector_qball_spin_orbit_mass_ratio_table_metrics.json"

CURRENT_ROUTE = "trial2_numeric_alpha_newton_limit_same_sector_proxy_equivalence_rule_identification"
NEXT_ROUTE = "8.7.56.743"
NEXT_BRANCH = "8.7.56.743-.746"
NEXT_RESIDUAL_ROUTE = "trial2_numeric_alpha_newton_limit_same_sector_proxy_equivalence_statement_identification"
NEXT_MISSING_ARTIFACT = "trial2_numeric_alpha_same_sector_proxy_equivalence_statement"


# Function: return the current UTC timestamp.
def now_iso() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


# Function: stop execution when a required path is missing.

def require(path: Path) -> None:
    """Require an input path to exist before execution continues."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# Function: read a UTF-8 text file.

def read_text(path: Path) -> str:
    """Read a UTF-8 text file."""
    return path.read_text(encoding="utf-8")


# Function: read a UTF-8 JSON file.

def read_json(path: Path) -> dict:
    """Read a UTF-8 JSON file."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# Function: return a stable display path.

def display_path(path: Path) -> str:
    """Return a stable path relative to the repository root when possible."""
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


# Function: locate the first line containing a substring pattern.

def hit(text: str, pattern: str) -> dict | None:
    """Return the first line hit for the given substring pattern."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# Function: build a standard metrics row.

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build a standard metrics row."""
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
    """Build a standard metrics payload."""
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


# Function: write a JSON metrics artifact and the matching CSV rows table.

def write_artifact(stem: str, data: dict) -> None:
    """Write a metrics payload as JSON and CSV."""
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / f"{stem}_metrics.json"
    csv_path = OUT / f"{stem}_rows.csv"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# Function: build a standard inventory target record.

def target_record(file_key: str, path: Path, text: str, pattern: str, note: str) -> dict:
    """Build a standard inventory target record."""
    target_hit = hit(text, pattern)
    return {
        "file_key": file_key,
        "file": display_path(path),
        "pattern": pattern,
        "present": target_hit is not None,
        "note": note,
        "evidence": target_hit,
    }


# Function: normalize the historical first-close target into the current proxy naming.

def normalized_first_route(raw_first_route: str) -> str | None:
    """Normalize the historical same-sector first-close target into proxy naming."""
    if raw_first_route == "same_sector_equivalence_statement":
        return "same_sector_proxy_equivalence_statement"

    return raw_first_route or None


# Function: execute the same-sector-proxy-equivalence-rule residual branch.

def main() -> None:
    """Execute the Trial-2 numeric alpha same-sector-proxy-equivalence-rule residual branch."""
    for path in (
        PART1,
        PART3A,
        PART5,
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        PRIOR_INVENTORY,
        PRIOR_AUDIT,
        PRIOR_GATE,
        PRIOR_ROUTE,
        SAME_SECTOR_PROXY_EQUIVALENCE_AUDIT,
        SAME_SECTOR_EQUIVALENCE_SOURCE_INVENTORY,
        SAME_SECTOR_EQUIVALENCE_WORDING_AUDIT,
        SAME_SECTOR_EQUIVALENCE_STATEMENT_ROUTE_CONTRACT,
        QBALL_SPIN_ORBIT,
    ):
        require(path)

    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    prior_inventory = read_json(PRIOR_INVENTORY)
    prior_audit = read_json(PRIOR_AUDIT)
    prior_gate = read_json(PRIOR_GATE)
    prior_route = read_json(PRIOR_ROUTE)
    same_sector_proxy_equivalence_audit = read_json(SAME_SECTOR_PROXY_EQUIVALENCE_AUDIT)
    same_sector_equivalence_source_inventory = read_json(SAME_SECTOR_EQUIVALENCE_SOURCE_INVENTORY)
    same_sector_equivalence_wording_audit = read_json(SAME_SECTOR_EQUIVALENCE_WORDING_AUDIT)
    same_sector_equivalence_statement_route_contract = read_json(
        SAME_SECTOR_EQUIVALENCE_STATEMENT_ROUTE_CONTRACT
    )
    qball_spin_orbit = read_json(QBALL_SPIN_ORBIT)

    prior_inventory_summary = prior_inventory["summary"]
    prior_audit_summary = prior_audit["summary"]
    prior_gate_summary = prior_gate["summary"]
    prior_route_summary = prior_route["summary"]
    same_sector_proxy_equivalence_summary = same_sector_proxy_equivalence_audit["summary"]
    same_sector_equivalence_source_inventory_summary = same_sector_equivalence_source_inventory["summary"]
    same_sector_equivalence_wording_audit_summary = same_sector_equivalence_wording_audit["summary"]
    same_sector_equivalence_statement_route_contract_summary = (
        same_sector_equivalence_statement_route_contract["summary"]
    )
    qball_spin_orbit_summary = qball_spin_orbit["summary"]

    electron_identification_pivot_retained = bool(
        prior_inventory_summary["electron_identification_pivot_retained"]
    ) and bool(prior_route_summary["electron_identification_pivot_retained"])
    absolute_normalization_dictionary_ready = bool(
        prior_inventory_summary["absolute_normalization_dictionary_ready"]
    ) and bool(prior_gate_summary["trial2_numeric_alpha_absolute_normalization_dictionary_ready"])
    m0_numeric_from_electron_identification_ready = bool(
        prior_inventory_summary["m0_numeric_from_electron_identification_ready"]
    )
    qball_ground_state_proxy_evidence_retained = bool(
        prior_inventory_summary["qball_ground_state_proxy_evidence_retained"]
    ) and float(qball_spin_orbit_summary["reference_state_mass_proxy"]) > 0.0
    chi_proxy_inventory_ready = bool(prior_inventory_summary["chi_proxy_inventory_ready"])
    chi_star_proxy_source_inventory_ready = bool(prior_inventory_summary["chi_star_proxy_source_inventory_ready"])
    same_sector_proxy_equivalence_rule_available = bool(
        prior_audit_summary["same_sector_proxy_equivalence_rule_available"]
    ) and bool(same_sector_proxy_equivalence_summary["same_sector_proxy_rule_available"])
    proxy_route_nonclosure_reason_or_none = prior_audit_summary["proxy_route_nonclosure_reason_or_none"]
    same_sector_equivalence_source_inventory_ready = bool(
        same_sector_equivalence_source_inventory_summary["equivalence_source_inventory_ready"]
    )
    raw_first_route_to_close_or_none = str(
        same_sector_equivalence_source_inventory_summary["first_route_to_close_or_none"]
    )
    first_route_to_close_or_none = normalized_first_route(raw_first_route_to_close_or_none)
    same_sector_proxy_equivalence_statement_available = (
        "same_sector_equivalence_statement"
        not in same_sector_equivalence_wording_audit_summary["equivalence_rule_missing_inputs"]
    )
    equivalence_relation_operator_available = (
        "equivalence_relation_operator"
        not in same_sector_equivalence_wording_audit_summary["equivalence_rule_missing_inputs"]
    )
    statement_route_contract_ready = (
        same_sector_equivalence_statement_route_contract_summary[
            "missing_same_sector_equivalence_statement_artifact"
        ]
        == "same_sector_equivalence_statement"
    ) and bool(same_sector_equivalence_statement_route_contract_summary["split_contract_ready"])
    declaration_gate_consistent = prior_gate_summary["selected_residual_route"] == CURRENT_ROUTE
    route_contract_consistent = prior_route_summary["selected_next_generation_route"] == CURRENT_ROUTE
    same_sector_proxy_equivalence_lineage_ready = (
        same_sector_equivalence_source_inventory_ready and statement_route_contract_ready
    )
    dominant_blocker_is_same_sector_proxy_equivalence_statement_absence = (
        absolute_normalization_dictionary_ready
        and not same_sector_proxy_equivalence_rule_available
        and proxy_route_nonclosure_reason_or_none == "same_sector_equivalence_rule_absent"
        and first_route_to_close_or_none == "same_sector_proxy_equivalence_statement"
        and not same_sector_proxy_equivalence_statement_available
    )

    common_inputs = {
        "part1_markdown": display_path(PART1),
        "part3a_markdown": display_path(PART3A),
        "part5_markdown": display_path(PART5),
        "status_markdown": display_path(STATUS),
        "roadmap_markdown": display_path(ROADMAP),
        "ai_context_json": display_path(AI_CONTEXT),
        "mass_origin_v2_trial2_numeric_alpha_chi_star_or_same_sector_proxy_numeric_value_source_inventory_json": display_path(
            PRIOR_INVENTORY
        ),
        "mass_origin_v2_trial2_numeric_alpha_chi_star_or_same_sector_proxy_numeric_value_audit_json": display_path(
            PRIOR_AUDIT
        ),
        "mass_origin_v2_trial2_numeric_alpha_chi_star_or_same_sector_proxy_numeric_value_declaration_gate_json": display_path(
            PRIOR_GATE
        ),
        "mass_origin_v2_t2_alpha_route_contract_eighty_first_refresh_json": display_path(PRIOR_ROUTE),
        "mass_origin_same_sector_proxy_equivalence_audit_json": display_path(
            SAME_SECTOR_PROXY_EQUIVALENCE_AUDIT
        ),
        "mass_origin_same_sector_equivalence_source_inventory_json": display_path(
            SAME_SECTOR_EQUIVALENCE_SOURCE_INVENTORY
        ),
        "mass_origin_same_sector_equivalence_wording_audit_json": display_path(
            SAME_SECTOR_EQUIVALENCE_WORDING_AUDIT
        ),
        "mass_origin_same_sector_equivalence_statement_route_contract_json": display_path(
            SAME_SECTOR_EQUIVALENCE_STATEMENT_ROUTE_CONTRACT
        ),
        "mass_origin_vector_qball_spin_orbit_mass_ratio_table_json": display_path(QBALL_SPIN_ORBIT),
    }

    inventory_targets = [
        target_record(
            "part1_electron_identification_statement",
            PART1,
            part1_text,
            r"M_{(1,0,0,0)} = m_e",
            "Part I still carries the electron-identification absolute-normalization statement.",
        ),
        target_record(
            "part3a_alpha_computation_formula",
            PART3A,
            part3a_text,
            r"\alpha=16\pi G^2\lambda v^2/(m_0^2\hbar c)",
            "Part III-A still carries the current computation-side alpha formula.",
        ),
        target_record(
            "part3a_same_sector_proxy_equivalence_rule_wording",
            PART3A,
            part3a_text,
            "same_sector_proxy_equivalence_rule",
            "Part III-A still names the current same-sector proxy equivalence-rule blocker before this branch closes.",
        ),
        target_record(
            "part5_same_sector_proxy_equivalence_rule_branch_wording",
            PART5,
            part5_text,
            "same_sector_proxy_equivalence_rule",
            "Part V still names the current official same-sector proxy equivalence-rule residual branch before this branch closes.",
        ),
        target_record(
            "status_current_same_sector_proxy_equivalence_rule_branch",
            STATUS,
            status_text,
            "8.7.56.739-.742",
            "STATUS still names the current official branch before this shrink step closes.",
        ),
        target_record(
            "roadmap_current_same_sector_proxy_equivalence_rule_branch",
            ROADMAP,
            roadmap_text,
            "8.7.56.739-.742",
            "ROADMAP still names the current official branch before this shrink step closes.",
        ),
    ]
    inventory_ready = (
        all(target["present"] for target in inventory_targets)
        and electron_identification_pivot_retained
        and absolute_normalization_dictionary_ready
        and m0_numeric_from_electron_identification_ready
        and qball_ground_state_proxy_evidence_retained
        and chi_proxy_inventory_ready
        and chi_star_proxy_source_inventory_ready
        and same_sector_proxy_equivalence_lineage_ready
        and declaration_gate_consistent
        and route_contract_consistent
    )

    inventory_payload = payload(
        "8.7.56.739",
        "Trial-2 numeric alpha same-sector-proxy-equivalence-rule source inventory",
        common_inputs,
        "Freeze the same-sector-proxy-equivalence-rule residual pack and show that the next honest closure attempt starts from the missing same-sector-proxy-equivalence statement inside the inherited same-sector equivalence lineage.",
        {
            "inventory_rule": "the same-sector-proxy-equivalence-rule route stays open until the public pack exposes an explicit same-sector equivalence statement together with an equivalence relation operator inside the no-new-free-parameter envelope",
            "lineage_rule": "the proxy-rule route inherits its internal closure order from the public same-sector-equivalence source inventory lineage, so the first internal close target can be normalized from same_sector_equivalence_statement to same_sector_proxy_equivalence_statement",
            "continuity_rule": "the prior declaration gate and prior route contract must still point at the same-sector-proxy-equivalence-rule route before this branch can shrink it",
        },
        [
            row(
                "trial2_numeric_alpha_same_sector_proxy_rule_inventory_complete",
                "pass" if inventory_ready else "reject",
                "same-sector-proxy-equivalence-rule inventory complete",
                1 if inventory_ready else 0,
                "The electron-identification dictionary, computation formula, proxy-rule audit, historical same-sector lineage, and prior contracts are frozen as one pack.",
            ),
            row(
                "trial2_numeric_alpha_electron_identification_pivot_retained_during_proxy_rule_inventory",
                "pass" if electron_identification_pivot_retained else "reject",
                "electron-identification pivot retained during proxy-rule inventory",
                1 if electron_identification_pivot_retained else 0,
                "The current residual still sits on the electron-identification absolute-normalization dictionary.",
            ),
            row(
                "trial2_numeric_alpha_same_sector_proxy_equivalence_lineage_ready",
                "pass" if same_sector_proxy_equivalence_lineage_ready else "reject",
                "same-sector-proxy-equivalence lineage ready",
                1 if same_sector_proxy_equivalence_lineage_ready else 0,
                "The underlying same-sector-equivalence source inventory and statement route contract are already frozen and can be reused as lineage evidence.",
            ),
            row(
                "trial2_numeric_alpha_same_sector_proxy_equivalence_statement_is_first_route_to_close",
                "pass" if first_route_to_close_or_none == "same_sector_proxy_equivalence_statement" else "reject",
                "same-sector-proxy-equivalence statement is first route to close",
                1 if first_route_to_close_or_none == "same_sector_proxy_equivalence_statement" else 0,
                "The proxy-rule route now points to a statement-level internal closure target rather than a generic rule-level blocker.",
            ),
            row(
                "trial2_numeric_alpha_prior_declaration_gate_consistent_with_proxy_rule",
                "pass" if declaration_gate_consistent else "reject",
                "prior declaration gate consistent with same-sector-proxy-equivalence-rule route",
                1 if declaration_gate_consistent else 0,
                "The eighty-first branch gate must still point at the same-sector-proxy-equivalence-rule route before this branch can shrink it.",
            ),
            row(
                "trial2_numeric_alpha_prior_route_contract_consistent_with_proxy_rule",
                "pass" if route_contract_consistent else "reject",
                "prior route contract consistent with same-sector-proxy-equivalence-rule route",
                1 if route_contract_consistent else 0,
                "The eighty-first refresh must still point at the same-sector-proxy-equivalence-rule route before this branch can shrink it.",
            ),
        ],
        {
            "inventory_ready": inventory_ready,
            "electron_identification_pivot_retained": electron_identification_pivot_retained,
            "absolute_normalization_dictionary_ready": absolute_normalization_dictionary_ready,
            "m0_numeric_from_electron_identification_ready": m0_numeric_from_electron_identification_ready,
            "qball_ground_state_proxy_evidence_retained": qball_ground_state_proxy_evidence_retained,
            "chi_proxy_inventory_ready": chi_proxy_inventory_ready,
            "chi_star_proxy_source_inventory_ready": chi_star_proxy_source_inventory_ready,
            "same_sector_proxy_equivalence_lineage_ready": same_sector_proxy_equivalence_lineage_ready,
            "same_sector_proxy_equivalence_rule_available": same_sector_proxy_equivalence_rule_available,
            "same_sector_proxy_equivalence_statement_available": same_sector_proxy_equivalence_statement_available,
            "equivalence_relation_operator_available": equivalence_relation_operator_available,
            "first_route_to_close_or_none": first_route_to_close_or_none,
            "declaration_gate_consistent": declaration_gate_consistent,
            "route_contract_consistent": route_contract_consistent,
        },
        {
            "overall_status": "trial2_numeric_alpha_same_sector_proxy_rule_inventory_frozen",
            "advance_to_8_7_56_740": inventory_ready,
            "next_required_artifacts": [],
        },
        {
            "inventory_targets": inventory_targets,
            "prior_inventory_summary": prior_inventory_summary,
            "prior_audit_summary": prior_audit_summary,
            "prior_gate_summary": prior_gate_summary,
            "prior_route_summary": prior_route_summary,
            "same_sector_proxy_equivalence_summary": same_sector_proxy_equivalence_summary,
            "same_sector_equivalence_source_inventory_summary": same_sector_equivalence_source_inventory_summary,
            "same_sector_equivalence_wording_audit_summary": same_sector_equivalence_wording_audit_summary,
            "same_sector_equivalence_statement_route_contract_summary": same_sector_equivalence_statement_route_contract_summary,
            "qball_spin_orbit_summary": qball_spin_orbit_summary,
            "ai_context_current_step": ai_context["current_step"],
        },
    )

    audit_payload = payload(
        "8.7.56.740",
        "Trial-2 numeric alpha same-sector-proxy-equivalence-rule audit",
        common_inputs,
        "Audit whether the current proxy-rule blocker remains generic or whether the honest next blocker has already shrunk to the missing same-sector-proxy-equivalence statement.",
        {
            "proxy_rule_rule": "the same-sector-proxy-equivalence-rule route remains open while the public pack lacks an explicit same-sector equivalence statement together with an equivalence relation operator",
            "shrink_rule": "if the inherited same-sector-equivalence source inventory names same_sector_equivalence_statement as the first route to close and the wording audit still returns that statement absent, the dominant proxy-rule blocker shrinks to same_sector_proxy_equivalence_statement itself",
            "numeric_rule": "numeric alpha stays open until the same-sector proxy can be promoted without a new fit parameter",
        },
        [
            row(
                "trial2_numeric_alpha_same_sector_proxy_rule_audit_complete",
                "pass",
                "same-sector-proxy-equivalence-rule audit complete",
                1,
                "This step tests whether the current public pack still needs a generic proxy-rule blocker or a more specific proxy-statement blocker.",
            ),
            row(
                "trial2_numeric_alpha_same_sector_proxy_equivalence_rule_available",
                "pass" if same_sector_proxy_equivalence_rule_available else "reject",
                "same-sector-proxy-equivalence rule available",
                1 if same_sector_proxy_equivalence_rule_available else 0,
                "The current pack still lacks the public same-sector proxy equivalence rule needed for numeric alpha.",
            ),
            row(
                "trial2_numeric_alpha_same_sector_proxy_equivalence_statement_available",
                "pass" if same_sector_proxy_equivalence_statement_available else "reject",
                "same-sector-proxy-equivalence statement available",
                1 if same_sector_proxy_equivalence_statement_available else 0,
                "The inherited same-sector-equivalence wording audit still shows that the statement surface itself is absent.",
            ),
            row(
                "trial2_numeric_alpha_equivalence_relation_operator_available_under_proxy_rule",
                "pass" if equivalence_relation_operator_available else "reject",
                "equivalence relation operator available under same-sector-proxy-equivalence-rule route",
                1 if equivalence_relation_operator_available else 0,
                "The inherited same-sector-equivalence wording audit still shows that the equivalence operator is absent together with the statement.",
            ),
            row(
                "trial2_numeric_alpha_same_sector_proxy_equivalence_statement_is_first_route_to_close_in_audit",
                "pass" if first_route_to_close_or_none == "same_sector_proxy_equivalence_statement" else "reject",
                "same-sector-proxy-equivalence statement is first route to close in audit",
                1 if first_route_to_close_or_none == "same_sector_proxy_equivalence_statement" else 0,
                "The proxy-rule lineage already isolates the next honest blocker as the statement-level surface.",
            ),
            row(
                "trial2_numeric_alpha_dominant_blocker_is_same_sector_proxy_equivalence_statement_absence",
                "pass" if dominant_blocker_is_same_sector_proxy_equivalence_statement_absence else "reject",
                "dominant blocker is same-sector-proxy-equivalence statement absence",
                1 if dominant_blocker_is_same_sector_proxy_equivalence_statement_absence else 0,
                "The generic proxy-rule blocker can now shrink to the missing statement surface itself.",
            ),
        ],
        {
            "audit_ready": inventory_ready,
            "same_sector_proxy_equivalence_lineage_ready": same_sector_proxy_equivalence_lineage_ready,
            "same_sector_proxy_equivalence_rule_available": same_sector_proxy_equivalence_rule_available,
            "same_sector_proxy_equivalence_statement_available": same_sector_proxy_equivalence_statement_available,
            "equivalence_relation_operator_available": equivalence_relation_operator_available,
            "proxy_route_nonclosure_reason_or_none": proxy_route_nonclosure_reason_or_none,
            "first_route_to_close_or_none": first_route_to_close_or_none,
            "dominant_blocker_is_same_sector_proxy_equivalence_statement_absence": dominant_blocker_is_same_sector_proxy_equivalence_statement_absence,
            "first_route_to_close_after_audit_or_none": "trial2_numeric_alpha_same_sector_proxy_equivalence_rule_declaration_gate",
        },
        {
            "overall_status": "trial2_numeric_alpha_same_sector_proxy_rule_audit_complete",
            "advance_to_8_7_56_741": True,
            "next_required_artifacts": [],
        },
        {
            "inventory_summary": inventory_payload["summary"],
            "same_sector_proxy_equivalence_summary": same_sector_proxy_equivalence_summary,
            "same_sector_equivalence_source_inventory_summary": same_sector_equivalence_source_inventory_summary,
            "same_sector_equivalence_wording_audit_summary": same_sector_equivalence_wording_audit_summary,
        },
    )

    gate_payload = payload(
        "8.7.56.741",
        "Trial-2 numeric alpha same-sector-proxy-equivalence-rule declaration gate",
        common_inputs,
        "Close the same-sector-proxy-equivalence-rule branch honestly: keep numeric alpha open, preserve the computation route and electron-identification dictionary, and freeze same_sector_proxy_equivalence_statement as the next official blocker.",
        {
            "closure_rule": "the branch closes as numeric-open whenever the current public pack lacks the same-sector equivalence statement that the proxy-rule lineage already marks as the first closure target",
            "shrink_rule": "a same-sector-proxy-equivalence-rule blocker becomes a same-sector-proxy-equivalence-statement blocker once the first closure target and its absence are both public-canonical inside the inherited same-sector-equivalence lineage",
            "mainline_rule": "precision-alpha remains on the mainline while strong-side gaps stay on v3 hold reserve",
        },
        [
            row(
                "trial2_numeric_alpha_same_sector_proxy_rule_gate_complete",
                "pass",
                "same-sector-proxy-equivalence-rule declaration gate complete",
                1,
                "The branch now closes the generic proxy-rule route and freezes the concrete next blocker.",
            ),
            row(
                "trial2_numeric_alpha_computation_formula_retained_after_same_sector_proxy_rule_gate",
                "pass" if prior_gate_summary["trial2_numeric_alpha_computation_formula_ready"] else "reject",
                "computation formula retained after same-sector-proxy-equivalence-rule gate",
                1 if prior_gate_summary["trial2_numeric_alpha_computation_formula_ready"] else 0,
                "The Newton-limit alpha computation formula remains the official route basis.",
            ),
            row(
                "trial2_numeric_alpha_absolute_normalization_dictionary_retained_after_same_sector_proxy_rule_gate",
                "pass" if absolute_normalization_dictionary_ready else "reject",
                "absolute-normalization dictionary retained after same-sector-proxy-equivalence-rule gate",
                1 if absolute_normalization_dictionary_ready else 0,
                "Electron identification remains part of the official mainline while the proxy-rule route stays open.",
            ),
            row(
                "trial2_numeric_alpha_numeric_from_current_pack_ready_after_same_sector_proxy_rule_gate",
                "pass" if same_sector_proxy_equivalence_rule_available else "reject",
                "numeric alpha from current pack ready after same-sector-proxy-equivalence-rule gate",
                1 if same_sector_proxy_equivalence_rule_available else 0,
                "Numeric alpha stays open because the statement-level same-sector equivalence surface is still absent.",
            ),
            row(
                "trial2_numeric_alpha_blocker_shrunk_to_same_sector_proxy_equivalence_statement",
                "pass" if dominant_blocker_is_same_sector_proxy_equivalence_statement_absence else "reject",
                "blocker shrunk to same-sector-proxy-equivalence statement",
                1 if dominant_blocker_is_same_sector_proxy_equivalence_statement_absence else 0,
                "The next route no longer targets the generic proxy rule; it targets the missing proxy-statement surface directly.",
            ),
        ],
        {
            "trial2_numeric_alpha_computation_formula_ready": bool(
                prior_gate_summary["trial2_numeric_alpha_computation_formula_ready"]
            ),
            "trial2_numeric_alpha_absolute_normalization_dictionary_ready": absolute_normalization_dictionary_ready,
            "trial2_numeric_alpha_numeric_from_current_pack_ready": False,
            "trial2_numeric_alpha_closeout_ready": False,
            "dominant_blocker_shrunk_from_same_sector_proxy_equivalence_rule_to_same_sector_proxy_equivalence_statement": dominant_blocker_is_same_sector_proxy_equivalence_statement_absence,
            "selected_residual_route": NEXT_RESIDUAL_ROUTE,
            "missing_v2_artifact": NEXT_MISSING_ARTIFACT,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_same_sector_proxy_rule_gate_closed_statement_open",
            "advance_to_8_7_56_742": True,
            "next_required_artifacts": [NEXT_MISSING_ARTIFACT],
        },
        {
            "audit_summary": audit_payload["summary"],
            "prior_gate_summary": prior_gate_summary,
            "prior_route_summary": prior_route_summary,
        },
    )

    route_payload = payload(
        "8.7.56.742",
        "Trial-2 numeric alpha next-generation route contract eighty-second refresh",
        common_inputs,
        "Refresh the next-generation contract after the same-sector-proxy-equivalence-rule shrink: keep precision-alpha on the mainline, keep the strong side on reserve, and promote same_sector_proxy_equivalence_statement as the next official blocker.",
        {
            "selected_route_rule": "the next official route is the missing same-sector-proxy-equivalence statement inside the computation-side proxy rule",
            "mainline_rule": "precision-alpha remains the next-generation mainline after the proxy-statement shrink",
            "reserve_rule": "strong-side non-Abelian, running, and confinement gaps remain on v3 hold reserve",
        },
        [
            row(
                "trial2_numeric_alpha_same_sector_proxy_rule_gate_closed_before_refresh",
                "pass",
                "same-sector-proxy-equivalence-rule declaration gate closed before route refresh",
                1,
                "The next-generation contract is only refreshed after the branch closes its declaration gate.",
            ),
            row(
                "trial2_numeric_alpha_next_route_selected_as_same_sector_proxy_equivalence_statement",
                "pass",
                "same-sector-proxy-equivalence statement route selected",
                1,
                "The next route now targets the concrete same-sector-proxy-equivalence statement blocker.",
            ),
            row(
                "trial2_numeric_alpha_precision_mainline_retained_after_same_sector_proxy_statement_shrink",
                "pass" if prior_route_summary["precision_alpha_mainline_retained"] else "reject",
                "precision-alpha mainline retained after same-sector-proxy-equivalence-statement shrink",
                1 if prior_route_summary["precision_alpha_mainline_retained"] else 0,
                "The mainline remains Trial-2 numeric alpha, not the strong-side reserve.",
            ),
            row(
                "trial2_numeric_alpha_strong_side_route_state_retained_after_same_sector_proxy_statement_shrink",
                "pass" if prior_route_summary["strong_side_route_state"] == "v3_hold_reserve" else "reject",
                "strong-side route state retained after same-sector-proxy-equivalence-statement shrink",
                1 if prior_route_summary["strong_side_route_state"] == "v3_hold_reserve" else 0,
                "The strong side remains exploratory and is not promoted by the current alpha residual shrink.",
            ),
        ],
        {
            "selected_next_generation_route": NEXT_RESIDUAL_ROUTE,
            "strong_side_route_state": prior_route_summary["strong_side_route_state"],
            "precision_alpha_mainline_retained": bool(prior_route_summary["precision_alpha_mainline_retained"]),
            "electron_identification_pivot_retained": electron_identification_pivot_retained,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_route_contract_eighty_second_refresh_frozen",
            "advance_to_next_route": True,
            "next_required_artifacts": [
                "trial2_numeric_alpha_same_sector_proxy_equivalence_statement_source_inventory",
                "trial2_numeric_alpha_same_sector_proxy_equivalence_statement_audit",
            ],
        },
        {
            "gate_summary": gate_payload["summary"],
            "prior_route_summary": prior_route_summary,
        },
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_same_sector_proxy_equivalence_rule_source_inventory",
        inventory_payload,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_same_sector_proxy_equivalence_rule_audit",
        audit_payload,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_same_sector_proxy_equivalence_rule_declaration_gate",
        gate_payload,
    )
    write_artifact("mass_origin_v2_t2_alpha_route_contract_eighty_second_refresh", route_payload)

    print("[done] 8.7.56.739-.742 artifacts generated:")
    print(" - mass_origin_v2_trial2_numeric_alpha_same_sector_proxy_equivalence_rule_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_same_sector_proxy_equivalence_rule_audit_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_same_sector_proxy_equivalence_rule_declaration_gate_metrics.json")
    print(" - mass_origin_v2_t2_alpha_route_contract_eighty_second_refresh_metrics.json")
    print(f" - next official branch should move to {NEXT_BRANCH}")


# Function: run the same-sector-proxy-equivalence-rule residual branch from the CLI.

def run_cli() -> None:
    """CLI entry point for the same-sector-proxy-equivalence-rule residual branch."""
    main()


if __name__ == "__main__":
    run_cli()
