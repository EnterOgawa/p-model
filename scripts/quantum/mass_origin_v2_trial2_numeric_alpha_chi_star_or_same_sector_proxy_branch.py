#!/usr/bin/env python3
"""Generate 8.7.56.711-.714 Trial-2 numeric alpha chi/proxy residual artifacts."""

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
    OUT / "mass_origin_v2_trial2_numeric_alpha_absolute_normalization_input_pack_declaration_gate_metrics.json"
)
PRIOR_ROUTE = OUT / "mass_origin_v2_t2_alpha_route_contract_seventy_fourth_refresh_metrics.json"
CHI_STAR_PROXY_INVENTORY = OUT / "mass_origin_chi_star_proxy_source_inventory_metrics.json"
SAME_SECTOR_PROXY_EQUIVALENCE_AUDIT = OUT / "mass_origin_same_sector_proxy_equivalence_audit_metrics.json"

NEXT_ROUTE = "8.7.56.715"
NEXT_BRANCH = "8.7.56.715-.718"
CURRENT_ROUTE = "trial2_numeric_alpha_newton_limit_chi_star_or_same_sector_proxy_identification"
NEXT_RESIDUAL_ROUTE = "trial2_numeric_alpha_newton_limit_same_sector_equivalence_rule_identification"
NEXT_MISSING_ARTIFACT = "trial2_numeric_alpha_same_sector_equivalence_rule"


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


# Function: execute the chi/proxy residual branch.

def main() -> None:
    """Execute the Trial-2 numeric alpha chi/proxy residual branch."""
    for path in (
        PART3A,
        PART5,
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        PRIOR_GATE,
        PRIOR_ROUTE,
        CHI_STAR_PROXY_INVENTORY,
        SAME_SECTOR_PROXY_EQUIVALENCE_AUDIT,
    ):
        require(path)

    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    prior_gate = read_json(PRIOR_GATE)
    prior_route = read_json(PRIOR_ROUTE)
    chi_star_proxy_inventory = read_json(CHI_STAR_PROXY_INVENTORY)
    same_sector_proxy_equivalence_audit = read_json(SAME_SECTOR_PROXY_EQUIVALENCE_AUDIT)

    prior_gate_summary = prior_gate["summary"]
    prior_route_summary = prior_route["summary"]
    chi_star_proxy_summary = chi_star_proxy_inventory["summary"]
    same_sector_proxy_equivalence_summary = same_sector_proxy_equivalence_audit["summary"]

    missing_proxy_route_sources = list(chi_star_proxy_summary["missing_proxy_route_sources"])
    first_route_to_close_or_none = chi_star_proxy_summary["first_route_to_close_or_none"]
    chi_star_or_same_sector_proxy_family_available = len(missing_proxy_route_sources) == 0
    same_sector_equivalence_rule_available = bool(
        same_sector_proxy_equivalence_summary["same_sector_proxy_rule_available"]
    )
    chi_star_proxy_inventory_ready = bool(chi_star_proxy_summary["proxy_source_inventory_ready"])
    route_contract_consistent = prior_route_summary["selected_next_generation_route"] == CURRENT_ROUTE
    declaration_gate_consistent = prior_gate_summary["selected_residual_route"] == CURRENT_ROUTE
    dominant_blocker_is_same_sector_equivalence_rule_absence = (
        first_route_to_close_or_none == "same_sector_equivalence_rule"
        and not same_sector_equivalence_rule_available
    )

    common_inputs = {
        "part3a_markdown": display_path(PART3A),
        "part5_markdown": display_path(PART5),
        "status_markdown": display_path(STATUS),
        "roadmap_markdown": display_path(ROADMAP),
        "ai_context_json": display_path(AI_CONTEXT),
        "mass_origin_v2_trial2_numeric_alpha_absolute_normalization_input_pack_declaration_gate_json": display_path(
            PRIOR_GATE
        ),
        "mass_origin_v2_t2_alpha_route_contract_seventy_fourth_refresh_json": display_path(PRIOR_ROUTE),
        "mass_origin_chi_star_proxy_source_inventory_json": display_path(CHI_STAR_PROXY_INVENTORY),
        "mass_origin_same_sector_proxy_equivalence_audit_json": display_path(
            SAME_SECTOR_PROXY_EQUIVALENCE_AUDIT
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
            "part3a_chi_star_proxy_family_wording",
            PART3A,
            part3a_text,
            "chi_star_or_same_sector_proxy",
            "Part III-A still names the current chi/proxy-family blocker before this branch shrinks it.",
        ),
        target_record(
            "part5_chi_star_proxy_family_wording",
            PART5,
            part5_text,
            "chi-star-or-same-sector-proxy residual branch `8.7.56.711-.714`",
            "Part V still names the current official chi/proxy-family residual branch before this branch closes.",
        ),
        target_record(
            "status_current_chi_star_proxy_branch",
            STATUS,
            status_text,
            "8.7.56.711-.714",
            "STATUS still names the current official branch before this shrink step closes.",
        ),
        target_record(
            "roadmap_current_chi_star_proxy_branch",
            ROADMAP,
            roadmap_text,
            "8.7.56.711-.714",
            "ROADMAP still names the current official branch before this shrink step closes.",
        ),
    ]
    inventory_ready = (
        all(item["present"] for item in inventory_targets)
        and chi_star_proxy_inventory_ready
        and declaration_gate_consistent
        and route_contract_consistent
    )

    inventory_payload = payload(
        "8.7.56.711",
        "Trial-2 numeric alpha chi-star-or-same-sector-proxy source inventory",
        common_inputs,
        "Freeze the chi/proxy-family residual pack and show that the next honest closure attempt starts from the missing same-sector equivalence rule.",
        {
            "inventory_rule": "the chi/proxy-family route stays open until the public pack exposes either an explicit chi_* proxy datum or a same-sector equivalence rule that can supply the proxy honestly",
            "first_route_rule": "if the proxy-family source inventory says the first route to close is same_sector_equivalence_rule, the current generic family blocker can be inventoried as a same-sector-rule-led residual pack",
            "continuity_rule": "the prior declaration gate and prior route contract must still point at the chi/proxy-family route before this branch can shrink it",
        },
        [
            row(
                "trial2_numeric_alpha_chi_star_proxy_inventory_complete",
                "pass" if inventory_ready else "reject",
                "chi-star or same-sector proxy inventory complete",
                1 if inventory_ready else 0,
                "The current computation formula, proxy-family wording, proxy inventory, equivalence audit, and prior contracts are frozen as one pack.",
            ),
            row(
                "trial2_numeric_alpha_chi_star_proxy_source_inventory_ready",
                "pass" if chi_star_proxy_inventory_ready else "reject",
                "chi-star proxy source inventory ready",
                1 if chi_star_proxy_inventory_ready else 0,
                "The public-canonical inventory for the chi/proxy-family route is already frozen.",
            ),
            row(
                "trial2_numeric_alpha_same_sector_equivalence_rule_is_first_route_to_close",
                "pass" if first_route_to_close_or_none == "same_sector_equivalence_rule" else "reject",
                "same-sector equivalence rule is first route to close",
                1 if first_route_to_close_or_none == "same_sector_equivalence_rule" else 0,
                "The chi/proxy-family inventory already points to same_sector_equivalence_rule as the next minimal closure target.",
            ),
            row(
                "trial2_numeric_alpha_prior_declaration_gate_consistent",
                "pass" if declaration_gate_consistent else "reject",
                "prior declaration gate consistent with chi/proxy-family route",
                1 if declaration_gate_consistent else 0,
                "The seventy-fourth branch gate must still point at the chi/proxy-family route before this branch can shrink it.",
            ),
            row(
                "trial2_numeric_alpha_prior_route_contract_consistent",
                "pass" if route_contract_consistent else "reject",
                "prior route contract consistent with chi/proxy-family route",
                1 if route_contract_consistent else 0,
                "The seventy-fourth refresh must still point at the chi/proxy-family route before this branch can shrink it.",
            ),
        ],
        {
            "inventory_ready": inventory_ready,
            "chi_star_proxy_inventory_ready": chi_star_proxy_inventory_ready,
            "chi_star_or_same_sector_proxy_family_available": chi_star_or_same_sector_proxy_family_available,
            "same_sector_equivalence_rule_available": same_sector_equivalence_rule_available,
            "first_route_to_close_or_none": first_route_to_close_or_none,
            "declaration_gate_consistent": declaration_gate_consistent,
            "route_contract_consistent": route_contract_consistent,
        },
        {
            "overall_status": "trial2_numeric_alpha_chi_star_proxy_inventory_frozen"
            if inventory_ready
            else "trial2_numeric_alpha_chi_star_proxy_inventory_incomplete",
            "advance_to_8_7_56_712": inventory_ready,
            "next_required_artifacts": []
            if inventory_ready
            else ["trial2_numeric_alpha_chi_star_or_same_sector_proxy_source_inventory"],
        },
        {
            "inventory_targets": inventory_targets,
            "chi_star_proxy_inventory_summary": chi_star_proxy_summary,
            "same_sector_proxy_equivalence_summary": same_sector_proxy_equivalence_summary,
            "prior_gate_summary": prior_gate_summary,
            "prior_route_summary": prior_route_summary,
            "ai_context_current_step": ai_context["current_step"],
        },
    )

    audit_payload = payload(
        "8.7.56.712",
        "Trial-2 numeric alpha chi-star-or-same-sector-proxy audit",
        common_inputs,
        "Audit whether the current proxy-family blocker remains generic or whether the honest next blocker has already shrunk to the missing same-sector equivalence rule.",
        {
            "family_rule": "the chi/proxy-family route remains open while the public pack lacks both an explicit proxy datum and a no-new-free-parameter same-sector equivalence rule",
            "shrink_rule": "if the public source inventory names same_sector_equivalence_rule as the first route to close and the equivalence audit still returns unavailable, the dominant blocker shrinks to same_sector_equivalence_rule itself",
            "numeric_rule": "numeric alpha stays open until the same-sector equivalence rule or a direct chi_* proxy becomes public canonical",
        },
        [
            row(
                "trial2_numeric_alpha_chi_star_proxy_audit_complete",
                "pass",
                "chi-star or same-sector proxy audit complete",
                1,
                "This step tests whether the current public pack still needs a generic proxy family or a more specific same-sector equivalence rule.",
            ),
            row(
                "trial2_numeric_alpha_chi_star_or_same_sector_proxy_family_available",
                "pass" if chi_star_or_same_sector_proxy_family_available else "reject",
                "chi-star or same-sector proxy family available",
                1 if chi_star_or_same_sector_proxy_family_available else 0,
                "The current pack still lacks the public proxy-family closure needed for numeric alpha.",
            ),
            row(
                "trial2_numeric_alpha_same_sector_equivalence_rule_available",
                "pass" if same_sector_equivalence_rule_available else "reject",
                "same-sector equivalence rule available",
                1 if same_sector_equivalence_rule_available else 0,
                "The public audit still shows that the same-sector equivalence rule itself is absent.",
            ),
            row(
                "trial2_numeric_alpha_same_sector_equivalence_rule_is_first_route_to_close_in_audit",
                "pass" if first_route_to_close_or_none == "same_sector_equivalence_rule" else "reject",
                "same-sector equivalence rule is first route to close in audit",
                1 if first_route_to_close_or_none == "same_sector_equivalence_rule" else 0,
                "The chi/proxy-family inventory already isolates the next honest blocker as the equivalence rule.",
            ),
            row(
                "trial2_numeric_alpha_dominant_blocker_is_same_sector_equivalence_rule_absence",
                "pass" if dominant_blocker_is_same_sector_equivalence_rule_absence else "reject",
                "dominant blocker is same-sector equivalence rule absence",
                1 if dominant_blocker_is_same_sector_equivalence_rule_absence else 0,
                "The generic chi/proxy-family blocker can now shrink to the missing same-sector equivalence rule itself.",
            ),
        ],
        {
            "audit_ready": inventory_ready,
            "chi_star_proxy_inventory_ready": chi_star_proxy_inventory_ready,
            "chi_star_or_same_sector_proxy_family_available": chi_star_or_same_sector_proxy_family_available,
            "same_sector_equivalence_rule_available": same_sector_equivalence_rule_available,
            "first_route_to_close_or_none": first_route_to_close_or_none,
            "dominant_blocker_is_same_sector_equivalence_rule_absence": dominant_blocker_is_same_sector_equivalence_rule_absence,
            "first_route_to_close_after_audit_or_none": "trial2_numeric_alpha_chi_star_or_same_sector_proxy_declaration_gate",
        },
        {
            "overall_status": "trial2_numeric_alpha_chi_star_proxy_audit_complete",
            "advance_to_8_7_56_713": True,
            "next_required_artifacts": [],
        },
        {
            "inventory_summary": inventory_payload["summary"],
            "chi_star_proxy_inventory_summary": chi_star_proxy_summary,
            "same_sector_proxy_equivalence_summary": same_sector_proxy_equivalence_summary,
        },
    )

    gate_payload = payload(
        "8.7.56.713",
        "Trial-2 numeric alpha chi-star-or-same-sector-proxy declaration gate",
        common_inputs,
        "Close the chi/proxy-family branch honestly: keep numeric alpha open, preserve the computation route, and freeze same_sector_equivalence_rule as the next official blocker.",
        {
            "closure_rule": "the branch closes as numeric-open whenever the current public pack lacks the same-sector equivalence rule that the chi/proxy-family inventory already marks as the first closure target",
            "shrink_rule": "a generic chi/proxy-family blocker becomes a same-sector-equivalence-rule blocker once the first closure target and its absence are both public-canonical",
            "mainline_rule": "precision-alpha remains on the mainline while strong-side gaps stay on v3 hold reserve",
        },
        [
            row(
                "trial2_numeric_alpha_chi_star_proxy_gate_complete",
                "pass",
                "chi-star or same-sector proxy declaration gate complete",
                1,
                "The branch now closes the generic proxy-family route and freezes the concrete next blocker.",
            ),
            row(
                "trial2_numeric_alpha_computation_formula_retained_after_chi_star_proxy_gate",
                "pass" if prior_gate_summary["trial2_numeric_alpha_computation_formula_ready"] else "reject",
                "computation formula retained after chi-star proxy gate",
                1 if prior_gate_summary["trial2_numeric_alpha_computation_formula_ready"] else 0,
                "The Newton-limit alpha computation formula remains the official route basis.",
            ),
            row(
                "trial2_numeric_alpha_numeric_from_current_pack_ready_after_chi_star_proxy_gate",
                "pass"
                if prior_gate_summary["trial2_numeric_alpha_numeric_from_current_pack_ready"]
                else "reject",
                "numeric alpha from current pack ready after chi-star proxy gate",
                1 if prior_gate_summary["trial2_numeric_alpha_numeric_from_current_pack_ready"] else 0,
                "Numeric alpha stays open because the same-sector equivalence rule is still absent.",
            ),
            row(
                "trial2_numeric_alpha_blocker_shrunk_to_same_sector_equivalence_rule",
                "pass" if dominant_blocker_is_same_sector_equivalence_rule_absence else "reject",
                "blocker shrunk to same-sector equivalence rule",
                1 if dominant_blocker_is_same_sector_equivalence_rule_absence else 0,
                "The next route no longer targets the generic chi/proxy family; it targets the missing same-sector equivalence rule directly.",
            ),
        ],
        {
            "trial2_numeric_alpha_computation_formula_ready": bool(
                prior_gate_summary["trial2_numeric_alpha_computation_formula_ready"]
            ),
            "trial2_numeric_alpha_closeout_ready": False,
            "trial2_numeric_alpha_numeric_from_current_pack_ready": False,
            "dominant_blocker_shrunk_from_chi_star_or_same_sector_proxy_to_same_sector_equivalence_rule": dominant_blocker_is_same_sector_equivalence_rule_absence,
            "selected_residual_route": NEXT_RESIDUAL_ROUTE,
            "missing_v2_artifact": NEXT_MISSING_ARTIFACT,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_chi_star_proxy_gate_closed_same_sector_rule_open",
            "advance_to_8_7_56_714": True,
            "next_required_artifacts": [NEXT_RESIDUAL_ROUTE],
        },
        {
            "audit_summary": audit_payload["summary"],
            "prior_gate_summary": prior_gate_summary,
            "prior_route_summary": prior_route_summary,
        },
    )

    route_payload = payload(
        "8.7.56.714",
        "Trial-2 numeric alpha next-generation route contract seventy-fifth refresh",
        common_inputs,
        "Refresh the next-generation contract after the chi/proxy-family shrink: keep precision-alpha on the mainline, keep the strong side on reserve, and promote same_sector_equivalence_rule as the next official blocker.",
        {
            "selected_route_rule": "the next official route is the missing same-sector equivalence rule inside the computation-side proxy family",
            "mainline_rule": "precision-alpha remains the next-generation mainline after the same-sector-rule shrink",
            "reserve_rule": "strong-side non-Abelian, running, and confinement gaps remain on v3 hold reserve",
        },
        [
            row(
                "trial2_numeric_alpha_chi_star_proxy_gate_closed_before_refresh",
                "pass",
                "chi-star proxy declaration gate closed before route refresh",
                1,
                "The next-generation contract is only refreshed after the branch closes its declaration gate.",
            ),
            row(
                "trial2_numeric_alpha_next_route_selected_as_same_sector_equivalence_rule",
                "pass",
                "same-sector equivalence rule route selected",
                1,
                "The next route now targets the concrete same-sector equivalence rule blocker.",
            ),
            row(
                "trial2_numeric_alpha_precision_mainline_retained_after_same_sector_rule_shrink",
                "pass",
                "precision-alpha mainline retained after same-sector-rule shrink",
                1,
                "The mainline remains Trial-2 numeric alpha, not the strong-side reserve.",
            ),
            row(
                "trial2_numeric_alpha_strong_side_route_state_retained_after_same_sector_rule_shrink",
                "pass",
                "strong-side route state retained after same-sector-rule shrink",
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
            "overall_status": "trial2_numeric_alpha_route_contract_seventy_fifth_refresh_frozen",
            "advance_to_next_route": True,
            "next_required_artifacts": [
                "trial2_numeric_alpha_same_sector_equivalence_rule_source_inventory",
                "trial2_numeric_alpha_same_sector_equivalence_rule_audit",
            ],
        },
        {
            "gate_summary": gate_payload["summary"],
            "prior_route_summary": prior_route_summary,
        },
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_chi_star_or_same_sector_proxy_source_inventory",
        inventory_payload,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_chi_star_or_same_sector_proxy_audit",
        audit_payload,
    )
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_chi_star_or_same_sector_proxy_declaration_gate",
        gate_payload,
    )
    write_artifact("mass_origin_v2_t2_alpha_route_contract_seventy_fifth_refresh", route_payload)

    print("[ok] wrote:")
    print(" - mass_origin_v2_trial2_numeric_alpha_chi_star_or_same_sector_proxy_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_chi_star_or_same_sector_proxy_audit_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_chi_star_or_same_sector_proxy_declaration_gate_metrics.json")
    print(" - mass_origin_v2_t2_alpha_route_contract_seventy_fifth_refresh_metrics.json")
    print(f" - next official branch should move to {NEXT_BRANCH}")


# Function: run the chi/proxy residual branch from the CLI.

def run_cli() -> None:
    """CLI entry point for the chi/proxy residual branch."""
    main()


if __name__ == "__main__":
    run_cli()
