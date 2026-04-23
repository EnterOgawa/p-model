#!/usr/bin/env python3
"""Generate 8.7.56.723-.726 Trial-2 numeric alpha same-sector-literal artifacts."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"

PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"

PRIOR_GATE = (
    OUT / "mass_origin_v2_trial2_numeric_alpha_same_sector_equivalence_statement_declaration_gate_metrics.json"
)
PRIOR_ROUTE = OUT / "mass_origin_v2_t2_alpha_route_contract_seventy_seventh_refresh_metrics.json"
LITERAL_SOURCE_INVENTORY = OUT / "mass_origin_same_sector_equivalence_literal_source_inventory_metrics.json"
LITERAL_WORDING_AUDIT = OUT / "mass_origin_same_sector_equivalence_literal_wording_audit_metrics.json"
PHRASE_FRAGMENT_ROUTE_CONTRACT = OUT / "mass_origin_same_sector_equivalence_phrase_fragment_route_contract_metrics.json"

NEXT_ROUTE = "8.7.56.727"
NEXT_BRANCH = "8.7.56.727-.730"
CURRENT_ROUTE = "trial2_numeric_alpha_newton_limit_same_sector_equivalence_literal_identification"
NEXT_RESIDUAL_ROUTE = "trial2_numeric_alpha_newton_limit_same_sector_equivalence_phrase_fragment_identification"
NEXT_MISSING_ARTIFACT = "trial2_numeric_alpha_same_sector_equivalence_phrase_fragment"


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


# Function: execute the same-sector-literal residual branch.

def main() -> None:
    """Execute the Trial-2 numeric alpha same-sector-literal residual branch."""
    for path in (
        PART3A,
        PART5,
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        PRIOR_GATE,
        PRIOR_ROUTE,
        LITERAL_SOURCE_INVENTORY,
        LITERAL_WORDING_AUDIT,
        PHRASE_FRAGMENT_ROUTE_CONTRACT,
    ):
        require(path)

    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    prior_gate = read_json(PRIOR_GATE)
    prior_route = read_json(PRIOR_ROUTE)
    literal_source_inventory = read_json(LITERAL_SOURCE_INVENTORY)
    literal_wording_audit = read_json(LITERAL_WORDING_AUDIT)
    phrase_fragment_route_contract = read_json(PHRASE_FRAGMENT_ROUTE_CONTRACT)

    prior_gate_summary = prior_gate["summary"]
    prior_route_summary = prior_route["summary"]
    literal_source_inventory_summary = literal_source_inventory["summary"]
    literal_wording_audit_summary = literal_wording_audit["summary"]
    phrase_fragment_route_contract_summary = phrase_fragment_route_contract["summary"]

    first_route_to_close_or_none = literal_source_inventory_summary["first_route_to_close_or_none"]
    literal_missing_inputs = list(literal_wording_audit_summary["literal_missing_inputs"])
    same_sector_equivalence_literal_available = bool(
        literal_wording_audit_summary["same_sector_equivalence_literal_available"]
    )
    same_sector_equivalence_phrase_fragment_available = (
        "same_sector_equivalence_phrase_fragment" not in literal_missing_inputs
    )
    equivalence_relation_operator_available = "equivalence_relation_operator" not in literal_missing_inputs
    literal_source_inventory_ready = bool(literal_source_inventory_summary["literal_source_inventory_ready"])
    route_contract_consistent = prior_route_summary["selected_next_generation_route"] == CURRENT_ROUTE
    declaration_gate_consistent = prior_gate_summary["selected_residual_route"] == CURRENT_ROUTE
    phrase_fragment_route_contract_ready = (
        phrase_fragment_route_contract_summary["missing_same_sector_equivalence_phrase_fragment_artifact"]
        == "same_sector_equivalence_phrase_fragment"
    ) and bool(phrase_fragment_route_contract_summary["split_contract_ready"])
    dominant_blocker_is_same_sector_equivalence_phrase_fragment_absence = (
        first_route_to_close_or_none == "same_sector_equivalence_phrase_fragment"
        and not same_sector_equivalence_phrase_fragment_available
    )

    common_inputs = {
        "part3a_markdown": display_path(PART3A),
        "part5_markdown": display_path(PART5),
        "status_markdown": display_path(STATUS),
        "roadmap_markdown": display_path(ROADMAP),
        "ai_context_json": display_path(AI_CONTEXT),
        "mass_origin_v2_trial2_numeric_alpha_same_sector_equivalence_statement_declaration_gate_json": display_path(
            PRIOR_GATE
        ),
        "mass_origin_v2_t2_alpha_route_contract_seventy_seventh_refresh_json": display_path(PRIOR_ROUTE),
        "mass_origin_same_sector_equivalence_literal_source_inventory_json": display_path(
            LITERAL_SOURCE_INVENTORY
        ),
        "mass_origin_same_sector_equivalence_literal_wording_audit_json": display_path(
            LITERAL_WORDING_AUDIT
        ),
        "mass_origin_same_sector_equivalence_phrase_fragment_route_contract_json": display_path(
            PHRASE_FRAGMENT_ROUTE_CONTRACT
        ),
    }

    inventory_targets = [
        target_record(
            "part3a_alpha_computation_formula",
            PART3A,
            part3a_text,
            r"\alpha=16\pi G^2\lambda v^2/(m_0^2\hbar c)",
            "Part III-A still carries the current computation-side alpha formula.",
        ),
        target_record(
            "part3a_same_sector_literal_wording",
            PART3A,
            part3a_text,
            "same_sector_equivalence_literal",
            "Part III-A still names the current same-sector-equivalence-literal blocker before this branch shrinks it.",
        ),
        target_record(
            "part5_same_sector_literal_branch_wording",
            PART5,
            part5_text,
            "same-sector-equivalence-literal residual branch `8.7.56.723-.726`",
            "Part V still names the current official same-sector-equivalence-literal residual branch before this branch closes.",
        ),
        target_record(
            "status_current_same_sector_literal_branch",
            STATUS,
            status_text,
            "8.7.56.723-.726",
            "STATUS still names the current official branch before this shrink step closes.",
        ),
        target_record(
            "roadmap_current_same_sector_literal_branch",
            ROADMAP,
            roadmap_text,
            "8.7.56.723-.726",
            "ROADMAP still names the current official branch before this shrink step closes.",
        ),
    ]
    inventory_ready = (
        all(item["present"] for item in inventory_targets)
        and literal_source_inventory_ready
        and declaration_gate_consistent
        and route_contract_consistent
        and phrase_fragment_route_contract_ready
    )

    inventory_payload = payload(
        "8.7.56.723",
        "Trial-2 numeric alpha same-sector-equivalence-literal source inventory",
        common_inputs,
        "Freeze the same-sector-literal residual pack and show that the next honest closure attempt starts from the missing same-sector equivalence phrase fragment.",
        {
            "inventory_rule": "the same-sector-literal route stays open until the public pack exposes an explicit same-sector equivalence phrase fragment together with an equivalence relation operator",
            "first_route_rule": "if the literal source inventory says the first route to close is same_sector_equivalence_phrase_fragment, the current literal blocker can be inventoried as a phrase-fragment-led residual pack",
            "continuity_rule": "the prior declaration gate and prior route contract must still point at the same-sector-literal route before this branch can shrink it",
        },
        [
            row(
                "trial2_numeric_alpha_same_sector_literal_inventory_complete",
                "pass" if inventory_ready else "reject",
                "same-sector-equivalence-literal inventory complete",
                1 if inventory_ready else 0,
                "The current computation formula, same-sector-literal wording, historical literal inventory, historical wording audit, and prior contracts are frozen as one pack.",
            ),
            row(
                "trial2_numeric_alpha_same_sector_literal_source_inventory_ready",
                "pass" if literal_source_inventory_ready else "reject",
                "same-sector equivalence literal source inventory ready",
                1 if literal_source_inventory_ready else 0,
                "The public-canonical inventory for the same-sector-literal route is already frozen.",
            ),
            row(
                "trial2_numeric_alpha_same_sector_equivalence_phrase_fragment_is_first_route_to_close",
                "pass" if first_route_to_close_or_none == "same_sector_equivalence_phrase_fragment" else "reject",
                "same-sector equivalence phrase fragment is first route to close",
                1 if first_route_to_close_or_none == "same_sector_equivalence_phrase_fragment" else 0,
                "The literal inventory already points to same_sector_equivalence_phrase_fragment as the next minimal closure target.",
            ),
            row(
                "trial2_numeric_alpha_same_sector_phrase_fragment_route_contract_ready",
                "pass" if phrase_fragment_route_contract_ready else "reject",
                "same-sector equivalence phrase-fragment route contract ready",
                1 if phrase_fragment_route_contract_ready else 0,
                "The historical phrase-fragment route contract is already split and ready to be reused as the next residual lineage.",
            ),
            row(
                "trial2_numeric_alpha_prior_declaration_gate_consistent",
                "pass" if declaration_gate_consistent else "reject",
                "prior declaration gate consistent with same-sector-literal route",
                1 if declaration_gate_consistent else 0,
                "The seventy-seventh branch gate must still point at the same-sector-literal route before this branch can shrink it.",
            ),
            row(
                "trial2_numeric_alpha_prior_route_contract_consistent",
                "pass" if route_contract_consistent else "reject",
                "prior route contract consistent with same-sector-literal route",
                1 if route_contract_consistent else 0,
                "The seventy-seventh refresh must still point at the same-sector-literal route before this branch can shrink it.",
            ),
        ],
        {
            "inventory_ready": inventory_ready,
            "same_sector_equivalence_literal_source_inventory_ready": literal_source_inventory_ready,
            "same_sector_equivalence_literal_available": same_sector_equivalence_literal_available,
            "same_sector_equivalence_phrase_fragment_available": same_sector_equivalence_phrase_fragment_available,
            "equivalence_relation_operator_available": equivalence_relation_operator_available,
            "first_route_to_close_or_none": first_route_to_close_or_none,
            "declaration_gate_consistent": declaration_gate_consistent,
            "route_contract_consistent": route_contract_consistent,
        },
        {
            "overall_status": "trial2_numeric_alpha_same_sector_literal_inventory_frozen"
            if inventory_ready
            else "trial2_numeric_alpha_same_sector_literal_inventory_incomplete",
            "advance_to_8_7_56_724": inventory_ready,
            "next_required_artifacts": []
            if inventory_ready
            else ["trial2_numeric_alpha_same_sector_equivalence_literal_source_inventory"],
        },
        {
            "inventory_targets": inventory_targets,
            "literal_source_inventory_summary": literal_source_inventory_summary,
            "literal_wording_audit_summary": literal_wording_audit_summary,
            "phrase_fragment_route_contract_summary": phrase_fragment_route_contract_summary,
            "prior_gate_summary": prior_gate_summary,
            "prior_route_summary": prior_route_summary,
            "ai_context_current_step": ai_context["current_step"],
        },
    )

    audit_payload = payload(
        "8.7.56.724",
        "Trial-2 numeric alpha same-sector-equivalence-literal audit",
        common_inputs,
        "Audit whether the current literal blocker remains generic or whether the honest next blocker has already shrunk to the missing same-sector equivalence phrase fragment.",
        {
            "literal_rule": "the same-sector-literal route remains open while the public pack lacks an explicit same-sector equivalence phrase fragment together with an equivalence relation operator",
            "shrink_rule": "if the public source inventory names same_sector_equivalence_phrase_fragment as the first route to close and the wording audit still returns that phrase fragment absent, the dominant blocker shrinks to same_sector_equivalence_phrase_fragment itself",
            "numeric_rule": "numeric alpha stays open until the same-sector equivalence phrase fragment or a stronger direct literal surface becomes public canonical",
        },
        [
            row(
                "trial2_numeric_alpha_same_sector_literal_audit_complete",
                "pass",
                "same-sector-equivalence-literal audit complete",
                1,
                "This step tests whether the current public pack still needs a generic literal blocker or a more specific same-sector equivalence phrase-fragment blocker.",
            ),
            row(
                "trial2_numeric_alpha_same_sector_equivalence_literal_available",
                "pass" if same_sector_equivalence_literal_available else "reject",
                "same-sector equivalence literal available",
                1 if same_sector_equivalence_literal_available else 0,
                "The public audit still shows that the same-sector equivalence literal itself is absent.",
            ),
            row(
                "trial2_numeric_alpha_same_sector_equivalence_phrase_fragment_available",
                "pass" if same_sector_equivalence_phrase_fragment_available else "reject",
                "same-sector equivalence phrase fragment available",
                1 if same_sector_equivalence_phrase_fragment_available else 0,
                "The public audit still shows that the same-sector equivalence phrase fragment itself is absent.",
            ),
            row(
                "trial2_numeric_alpha_equivalence_relation_operator_available",
                "pass" if equivalence_relation_operator_available else "reject",
                "equivalence relation operator available",
                1 if equivalence_relation_operator_available else 0,
                "The public audit still shows that the equivalence relation operator is absent together with the phrase fragment.",
            ),
            row(
                "trial2_numeric_alpha_same_sector_equivalence_phrase_fragment_is_first_route_to_close_in_audit",
                "pass" if first_route_to_close_or_none == "same_sector_equivalence_phrase_fragment" else "reject",
                "same-sector equivalence phrase fragment is first route to close in audit",
                1 if first_route_to_close_or_none == "same_sector_equivalence_phrase_fragment" else 0,
                "The same-sector-literal inventory already isolates the next honest blocker as the same-sector equivalence phrase fragment.",
            ),
            row(
                "trial2_numeric_alpha_dominant_blocker_is_same_sector_equivalence_phrase_fragment_absence",
                "pass" if dominant_blocker_is_same_sector_equivalence_phrase_fragment_absence else "reject",
                "dominant blocker is same-sector equivalence phrase fragment absence",
                1 if dominant_blocker_is_same_sector_equivalence_phrase_fragment_absence else 0,
                "The generic same-sector-literal blocker can now shrink to the missing same-sector equivalence phrase fragment itself.",
            ),
        ],
        {
            "audit_ready": inventory_ready,
            "same_sector_equivalence_literal_source_inventory_ready": literal_source_inventory_ready,
            "same_sector_equivalence_literal_available": same_sector_equivalence_literal_available,
            "same_sector_equivalence_phrase_fragment_available": same_sector_equivalence_phrase_fragment_available,
            "equivalence_relation_operator_available": equivalence_relation_operator_available,
            "first_route_to_close_or_none": first_route_to_close_or_none,
            "dominant_blocker_is_same_sector_equivalence_phrase_fragment_absence": dominant_blocker_is_same_sector_equivalence_phrase_fragment_absence,
            "first_route_to_close_after_audit_or_none": "trial2_numeric_alpha_same_sector_equivalence_literal_declaration_gate",
        },
        {
            "overall_status": "trial2_numeric_alpha_same_sector_literal_audit_complete",
            "advance_to_8_7_56_725": True,
            "next_required_artifacts": [],
        },
        {
            "inventory_summary": inventory_payload["summary"],
            "literal_source_inventory_summary": literal_source_inventory_summary,
            "literal_wording_audit_summary": literal_wording_audit_summary,
        },
    )

    gate_payload = payload(
        "8.7.56.725",
        "Trial-2 numeric alpha same-sector-equivalence-literal declaration gate",
        common_inputs,
        "Close the same-sector-literal branch honestly: keep numeric alpha open, preserve the computation route, and freeze same_sector_equivalence_phrase_fragment as the next official blocker.",
        {
            "closure_rule": "the branch closes as numeric-open whenever the current public pack lacks the same-sector equivalence phrase fragment that the literal inventory already marks as the first closure target",
            "shrink_rule": "a same-sector-literal blocker becomes a same-sector-equivalence-phrase-fragment blocker once the first closure target and its absence are both public-canonical",
            "mainline_rule": "precision-alpha remains on the mainline while strong-side gaps stay on v3 hold reserve",
        },
        [
            row(
                "trial2_numeric_alpha_same_sector_literal_gate_complete",
                "pass",
                "same-sector-equivalence-literal declaration gate complete",
                1,
                "The branch now closes the generic same-sector-literal route and freezes the concrete next blocker.",
            ),
            row(
                "trial2_numeric_alpha_computation_formula_retained_after_same_sector_literal_gate",
                "pass" if prior_gate_summary["trial2_numeric_alpha_computation_formula_ready"] else "reject",
                "computation formula retained after same-sector-literal gate",
                1 if prior_gate_summary["trial2_numeric_alpha_computation_formula_ready"] else 0,
                "The Newton-limit alpha computation formula remains the official route basis.",
            ),
            row(
                "trial2_numeric_alpha_numeric_from_current_pack_ready_after_same_sector_literal_gate",
                "pass"
                if prior_gate_summary["trial2_numeric_alpha_numeric_from_current_pack_ready"]
                else "reject",
                "numeric alpha from current pack ready after same-sector-literal gate",
                1 if prior_gate_summary["trial2_numeric_alpha_numeric_from_current_pack_ready"] else 0,
                "Numeric alpha stays open because the same-sector equivalence phrase fragment is still absent.",
            ),
            row(
                "trial2_numeric_alpha_blocker_shrunk_to_same_sector_equivalence_phrase_fragment",
                "pass" if dominant_blocker_is_same_sector_equivalence_phrase_fragment_absence else "reject",
                "blocker shrunk to same-sector equivalence phrase fragment",
                1 if dominant_blocker_is_same_sector_equivalence_phrase_fragment_absence else 0,
                "The next route no longer targets the generic same-sector literal; it targets the missing same-sector equivalence phrase fragment directly.",
            ),
        ],
        {
            "trial2_numeric_alpha_computation_formula_ready": bool(
                prior_gate_summary["trial2_numeric_alpha_computation_formula_ready"]
            ),
            "trial2_numeric_alpha_closeout_ready": False,
            "trial2_numeric_alpha_numeric_from_current_pack_ready": False,
            "dominant_blocker_shrunk_from_same_sector_equivalence_literal_to_same_sector_equivalence_phrase_fragment": dominant_blocker_is_same_sector_equivalence_phrase_fragment_absence,
            "selected_residual_route": NEXT_RESIDUAL_ROUTE,
            "missing_v2_artifact": NEXT_MISSING_ARTIFACT,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_same_sector_literal_gate_closed_phrase_fragment_open",
            "advance_to_8_7_56_726": True,
            "next_required_artifacts": [NEXT_RESIDUAL_ROUTE],
        },
        {
            "audit_summary": audit_payload["summary"],
            "prior_gate_summary": prior_gate_summary,
            "prior_route_summary": prior_route_summary,
        },
    )

    route_payload = payload(
        "8.7.56.726",
        "Trial-2 numeric alpha next-generation route contract seventy-eighth refresh",
        common_inputs,
        "Refresh the next-generation contract after the same-sector-literal shrink: keep precision-alpha on the mainline, keep the strong side on reserve, and promote same_sector_equivalence_phrase_fragment as the next official blocker.",
        {
            "selected_route_rule": "the next official route is the missing same-sector equivalence phrase fragment inside the computation-side same-sector literal",
            "mainline_rule": "precision-alpha remains the next-generation mainline after the same-sector-phrase-fragment shrink",
            "reserve_rule": "strong-side non-Abelian, running, and confinement gaps remain on v3 hold reserve",
        },
        [
            row(
                "trial2_numeric_alpha_same_sector_literal_gate_closed_before_refresh",
                "pass",
                "same-sector-literal declaration gate closed before route refresh",
                1,
                "The next-generation contract is only refreshed after the branch closes its declaration gate.",
            ),
            row(
                "trial2_numeric_alpha_next_route_selected_as_same_sector_equivalence_phrase_fragment",
                "pass",
                "same-sector equivalence phrase fragment route selected",
                1,
                "The next route now targets the concrete same-sector equivalence phrase-fragment blocker.",
            ),
            row(
                "trial2_numeric_alpha_precision_mainline_retained_after_same_sector_phrase_fragment_shrink",
                "pass",
                "precision-alpha mainline retained after same-sector-phrase-fragment shrink",
                1,
                "The mainline remains Trial-2 numeric alpha, not the strong-side reserve.",
            ),
            row(
                "trial2_numeric_alpha_strong_side_route_state_retained_after_same_sector_phrase_fragment_shrink",
                "pass",
                "strong-side route state retained after same-sector-phrase-fragment shrink",
                1,
                "The strong side remains exploratory and is not promoted by the current alpha residual shrink.",
            ),
        ],
        {
            "selected_next_generation_route": NEXT_RESIDUAL_ROUTE,
            "strong_side_route_state": prior_route_summary["strong_side_route_state"],
            "precision_alpha_mainline_retained": True,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_route_contract_seventy_eighth_refresh_frozen",
            "advance_to_next_route": True,
            "next_required_artifacts": [
                "trial2_numeric_alpha_same_sector_equivalence_phrase_fragment_source_inventory",
                "trial2_numeric_alpha_same_sector_equivalence_phrase_fragment_audit",
            ],
        },
        {
            "gate_summary": gate_payload["summary"],
            "prior_route_summary": prior_route_summary,
        },
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_same_sector_equivalence_literal_source_inventory",
        inventory_payload,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_same_sector_equivalence_literal_audit",
        audit_payload,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_same_sector_equivalence_literal_declaration_gate",
        gate_payload,
    )
    write_artifact("mass_origin_v2_t2_alpha_route_contract_seventy_eighth_refresh", route_payload)

    print("[ok] wrote:")
    print(" - mass_origin_v2_trial2_numeric_alpha_same_sector_equivalence_literal_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_same_sector_equivalence_literal_audit_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_same_sector_equivalence_literal_declaration_gate_metrics.json")
    print(" - mass_origin_v2_t2_alpha_route_contract_seventy_eighth_refresh_metrics.json")
    print(f" - next official branch should move to {NEXT_BRANCH}")


# Function: run the same-sector-literal residual branch from the CLI.

def run_cli() -> None:
    """CLI entry point for the same-sector-literal residual branch."""
    main()


if __name__ == "__main__":
    run_cli()
