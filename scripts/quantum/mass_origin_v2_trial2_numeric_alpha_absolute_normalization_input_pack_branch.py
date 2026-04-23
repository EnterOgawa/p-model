#!/usr/bin/env python3
"""Generate 8.7.56.707-.710 Trial-2 numeric alpha absolute-input-pack artifacts."""

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

NEWTON_AUDIT = OUT / "mass_origin_v2_trial2_numeric_alpha_newton_limit_audit_metrics.json"
COMPUTATION_GATE = OUT / "mass_origin_v2_trial2_numeric_alpha_computation_declaration_gate_metrics.json"
PRIOR_ROUTE = OUT / "mass_origin_v2_t2_alpha_route_contract_seventy_third_refresh_metrics.json"
CHI_PROXY_INVENTORY = OUT / "mass_origin_anchor_normalized_g3w_chi_proxy_inventory_metrics.json"
SAME_SECTOR_PROXY_EQUIVALENCE_AUDIT = OUT / "mass_origin_same_sector_proxy_equivalence_audit_metrics.json"

NEXT_ROUTE = "8.7.56.711"
NEXT_BRANCH = "8.7.56.711-.714"
NEXT_RESIDUAL_ROUTE = "trial2_numeric_alpha_newton_limit_chi_star_or_same_sector_proxy_identification"
NEXT_MISSING_ARTIFACT = "trial2_numeric_alpha_chi_star_or_same_sector_proxy"
CURRENT_ROUTE = "trial2_numeric_alpha_newton_limit_absolute_normalization_input_pack_identification"


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


# Function: execute the absolute-normalization-input-pack residual branch.

def main() -> None:
    """Execute the Trial-2 numeric alpha absolute-input-pack residual branch."""
    for path in (
        PART1,
        PART3A,
        PART5,
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        NEWTON_AUDIT,
        COMPUTATION_GATE,
        PRIOR_ROUTE,
        CHI_PROXY_INVENTORY,
        SAME_SECTOR_PROXY_EQUIVALENCE_AUDIT,
    ):
        require(path)

    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)
    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    newton_audit = read_json(NEWTON_AUDIT)
    computation_gate = read_json(COMPUTATION_GATE)
    prior_route = read_json(PRIOR_ROUTE)
    chi_proxy_inventory = read_json(CHI_PROXY_INVENTORY)
    same_sector_proxy_equivalence_audit = read_json(SAME_SECTOR_PROXY_EQUIVALENCE_AUDIT)

    newton_summary = newton_audit["summary"]
    gate_summary = computation_gate["summary"]
    route_summary = prior_route["summary"]
    chi_proxy_summary = chi_proxy_inventory["summary"]
    proxy_equivalence_summary = same_sector_proxy_equivalence_audit["summary"]

    same_sector_symbolic_bridge_ready = bool(newton_summary["same_sector_symbolic_bridge_ready"])
    chi_proxy_inventory_ready = bool(chi_proxy_summary["chi_proxy_inventory_ready"])
    chi_proxy_missing_sources = list(chi_proxy_summary["missing_chi_proxy_sources"])
    chi_star_or_same_sector_proxy_available = "chi_star_or_same_sector_proxy" not in chi_proxy_missing_sources
    same_sector_proxy_equivalence_rule_available = bool(
        proxy_equivalence_summary["same_sector_proxy_rule_available"]
    )
    absolute_numeric_input_pack_ready = bool(newton_summary["absolute_numeric_input_pack_ready"])
    route_contract_consistent = route_summary["selected_next_generation_route"] == CURRENT_ROUTE

    common_inputs = {
        "part1_markdown": display_path(PART1),
        "part3a_markdown": display_path(PART3A),
        "part5_markdown": display_path(PART5),
        "status_markdown": display_path(STATUS),
        "roadmap_markdown": display_path(ROADMAP),
        "ai_context_json": display_path(AI_CONTEXT),
        "mass_origin_v2_trial2_numeric_alpha_newton_limit_audit_json": display_path(NEWTON_AUDIT),
        "mass_origin_v2_trial2_numeric_alpha_computation_declaration_gate_json": display_path(COMPUTATION_GATE),
        "mass_origin_v2_t2_alpha_route_contract_seventy_third_refresh_json": display_path(PRIOR_ROUTE),
        "mass_origin_anchor_normalized_g3w_chi_proxy_inventory_json": display_path(CHI_PROXY_INVENTORY),
        "mass_origin_same_sector_proxy_equivalence_audit_json": display_path(
            SAME_SECTOR_PROXY_EQUIVALENCE_AUDIT
        ),
    }

    inventory_targets = [
        target_record(
            "part1_weak_field_normalization",
            PART1,
            part1_text,
            "current canon では $g_P/Z_P=4\\pi G$",
            "Part I carries the weak-field normalization that anchors the computation route.",
        ),
        target_record(
            "part3a_alpha_computation_formula",
            PART3A,
            part3a_text,
            r"\alpha=16\pi G^2\lambda v^2/(m_0^2\hbar c)",
            "Part III-A carries the current Newton-limit alpha computation formula.",
        ),
        target_record(
            "part3a_absolute_input_pack_wording",
            PART3A,
            part3a_text,
            "absolute same-sector proxy / normalization input",
            "Part III-A already states that the numeric blocker is the missing absolute same-sector proxy input.",
        ),
        target_record(
            "part5_chi_star_blocker_wording",
            PART5,
            part5_text,
            "chi_star_or_same_sector_proxy",
            "Part V checkpoint wording already names the concrete missing proxy item.",
        ),
        target_record(
            "status_absolute_input_pack_branch",
            STATUS,
            status_text,
            "absolute-normalization-input-pack residual branch",
            "STATUS still names the current official branch before this shrink step closes.",
        ),
        target_record(
            "roadmap_absolute_input_pack_branch",
            ROADMAP,
            roadmap_text,
            "8.7.56.707-.710",
            "ROADMAP still names the current official branch before this shrink step closes.",
        ),
    ]
    inventory_ready = all(item["present"] for item in inventory_targets) and route_contract_consistent

    inventory_payload = payload(
        "8.7.56.707",
        "Trial-2 numeric alpha absolute-normalization-input-pack source inventory",
        common_inputs,
        "Freeze the computation-side source pack and show that the current absolute-input-pack blocker lives in the missing chi_star_or_same_sector_proxy family.",
        {
            "inventory_rule": "the absolute input pack stays open until the computation route has a public same-sector proxy that can supply an honest absolute normalization input",
            "current_route_rule": "the current official route remains the absolute-normalization-input-pack residual until this branch freezes the more specific blocker",
            "shrink_rule": "if the symbolic bridge is ready and the current pack names one concrete missing proxy family, the generic input-pack blocker can shrink to that family",
        },
        [
            row(
                "trial2_numeric_alpha_absolute_input_pack_inventory_complete",
                "pass" if inventory_ready else "reject",
                "absolute-normalization-input-pack inventory complete",
                1 if inventory_ready else 0,
                "The weak-field normalization, computation formula, missing proxy evidence, and prior route contract are frozen as one pack.",
            ),
            row(
                "trial2_numeric_alpha_same_sector_symbolic_bridge_pack_ready",
                "pass" if same_sector_symbolic_bridge_ready else "reject",
                "same-sector symbolic bridge pack ready",
                1 if same_sector_symbolic_bridge_ready else 0,
                "The symbolic chi_P bridge is already closed before the absolute proxy step begins.",
            ),
            row(
                "trial2_numeric_alpha_chi_proxy_inventory_ready",
                "pass" if chi_proxy_inventory_ready else "reject",
                "chi-star or same-sector proxy inventory ready",
                1 if chi_proxy_inventory_ready else 0,
                "The current public source inventory for the proxy route is already frozen.",
            ),
            row(
                "trial2_numeric_alpha_missing_chi_star_or_same_sector_proxy_count",
                "watch",
                "missing chi_star_or_same_sector_proxy count",
                float(len(chi_proxy_missing_sources)),
                f"Missing proxy-family sources: {chi_proxy_missing_sources}.",
            ),
            row(
                "trial2_numeric_alpha_prior_route_contract_consistent",
                "pass" if route_contract_consistent else "reject",
                "prior route contract consistent with absolute-input-pack route",
                1 if route_contract_consistent else 0,
                "The seventy-third refresh must still point at the absolute-input-pack route before this branch can shrink it.",
            ),
        ],
        {
            "inventory_ready": inventory_ready,
            "same_sector_symbolic_bridge_ready": same_sector_symbolic_bridge_ready,
            "chi_proxy_inventory_ready": chi_proxy_inventory_ready,
            "chi_star_or_same_sector_proxy_available": chi_star_or_same_sector_proxy_available,
            "same_sector_proxy_equivalence_rule_available": same_sector_proxy_equivalence_rule_available,
            "route_contract_consistent": route_contract_consistent,
            "first_route_to_close_or_none": "trial2_numeric_alpha_absolute_normalization_input_pack_audit",
        },
        {
            "overall_status": "trial2_numeric_alpha_absolute_input_pack_inventory_frozen"
            if inventory_ready
            else "trial2_numeric_alpha_absolute_input_pack_inventory_incomplete",
            "advance_to_8_7_56_708": inventory_ready,
            "next_required_artifacts": [] if inventory_ready else ["trial2_numeric_alpha_absolute_normalization_input_pack_source_inventory"],
        },
        {
            "inventory_targets": inventory_targets,
            "newton_audit_summary": newton_summary,
            "chi_proxy_inventory_summary": chi_proxy_summary,
            "same_sector_proxy_equivalence_summary": proxy_equivalence_summary,
            "prior_route_summary": route_summary,
            "ai_context_current_step": ai_context["current_step"],
        },
    )

    audit_payload = payload(
        "8.7.56.708",
        "Trial-2 numeric alpha absolute-normalization-input-pack audit",
        common_inputs,
        "Audit whether the current computation pack can supply an honest absolute same-sector proxy and determine whether the generic input-pack blocker shrinks to the missing chi_star_or_same_sector_proxy family.",
        {
            "absolute_input_rule": "absolute_numeric_input_pack_ready iff the current pack supplies both the symbolic bridge and a public absolute same-sector proxy input",
            "proxy_family_rule": "chi_star_or_same_sector_proxy remains the dominant blocker when the symbolic bridge is ready but the concrete proxy family stays absent",
            "equivalence_rule": "a same-sector proxy equivalence audit can explain why the proxy family remains absent without changing the fact that the missing family is chi_star_or_same_sector_proxy",
        },
        [
            row(
                "trial2_numeric_alpha_absolute_input_pack_audit_complete",
                "pass",
                "absolute-normalization-input-pack audit complete",
                1,
                "This step tests whether the current public pack can honestly evaluate the computation-side alpha formula numerically.",
            ),
            row(
                "trial2_numeric_alpha_same_sector_symbolic_bridge_ready_in_audit",
                "pass" if same_sector_symbolic_bridge_ready else "reject",
                "same-sector symbolic bridge ready in audit",
                1 if same_sector_symbolic_bridge_ready else 0,
                "The symbolic bridge itself is no longer the blocker.",
            ),
            row(
                "trial2_numeric_alpha_chi_star_or_same_sector_proxy_available",
                "pass" if chi_star_or_same_sector_proxy_available else "reject",
                "chi_star_or_same_sector_proxy available",
                1 if chi_star_or_same_sector_proxy_available else 0,
                "The current public pack still lacks the concrete same-sector proxy family needed to normalize alpha absolutely.",
            ),
            row(
                "trial2_numeric_alpha_same_sector_proxy_equivalence_rule_available",
                "pass" if same_sector_proxy_equivalence_rule_available else "reject",
                "same-sector proxy equivalence rule available",
                1 if same_sector_proxy_equivalence_rule_available else 0,
                "The older proxy audit still shows that the equivalence rule behind the proxy family is absent.",
            ),
            row(
                "trial2_numeric_alpha_absolute_numeric_input_pack_ready",
                "pass" if absolute_numeric_input_pack_ready else "reject",
                "absolute numeric input pack ready",
                1 if absolute_numeric_input_pack_ready else 0,
                "The computation formula remains numeric-open until the proxy family becomes public canonical.",
            ),
            row(
                "trial2_numeric_alpha_dominant_blocker_is_chi_star_or_same_sector_proxy_absence",
                "pass" if not chi_star_or_same_sector_proxy_available else "reject",
                "dominant blocker is chi_star_or_same_sector_proxy absence",
                1 if not chi_star_or_same_sector_proxy_available else 0,
                "The generic absolute-input-pack blocker now shrinks to the concrete missing proxy family named in the current public pack.",
            ),
        ],
        {
            "audit_ready": inventory_ready,
            "same_sector_symbolic_bridge_ready": same_sector_symbolic_bridge_ready,
            "chi_proxy_inventory_ready": chi_proxy_inventory_ready,
            "chi_star_or_same_sector_proxy_available": chi_star_or_same_sector_proxy_available,
            "same_sector_proxy_equivalence_rule_available": same_sector_proxy_equivalence_rule_available,
            "absolute_numeric_input_pack_ready": absolute_numeric_input_pack_ready,
            "dominant_blocker_is_chi_star_or_same_sector_proxy_absence": not chi_star_or_same_sector_proxy_available,
            "first_route_to_close_or_none": "trial2_numeric_alpha_absolute_normalization_input_pack_declaration_gate",
        },
        {
            "overall_status": "trial2_numeric_alpha_absolute_input_pack_audit_complete",
            "advance_to_8_7_56_709": True,
            "next_required_artifacts": [],
        },
        {
            "inventory_summary": inventory_payload["summary"],
            "newton_audit_summary": newton_summary,
            "chi_proxy_inventory_summary": chi_proxy_summary,
            "same_sector_proxy_equivalence_summary": proxy_equivalence_summary,
        },
    )

    gate_payload = payload(
        "8.7.56.709",
        "Trial-2 numeric alpha absolute-normalization-input-pack declaration gate",
        common_inputs,
        "Close the absolute-input-pack branch honestly: keep the computation formula, keep numeric alpha open, and freeze the next residual as the missing chi_star_or_same_sector_proxy family.",
        {
            "closure_rule": "the branch closes as formula-ready but numeric-open whenever the symbolic bridge is ready and the concrete proxy family remains absent",
            "shrink_rule": "a generic absolute-input-pack blocker becomes a chi_star_or_same_sector_proxy blocker once the missing family is explicitly identified",
            "mainline_rule": "precision-alpha remains on the mainline while strong-side gaps stay on v3 hold reserve",
        },
        [
            row(
                "trial2_numeric_alpha_absolute_input_pack_gate_complete",
                "pass",
                "absolute-normalization-input-pack declaration gate complete",
                1,
                "The branch now closes the generic input-pack route and freezes the concrete next blocker.",
            ),
            row(
                "trial2_numeric_alpha_computation_formula_retained",
                "pass",
                "computation formula retained through declaration gate",
                1,
                "The Newton-limit alpha computation formula remains the official route basis.",
            ),
            row(
                "trial2_numeric_alpha_numeric_from_current_pack_ready_after_absolute_pack_gate",
                "pass" if absolute_numeric_input_pack_ready else "reject",
                "numeric alpha from current pack ready after absolute-input-pack gate",
                1 if absolute_numeric_input_pack_ready else 0,
                "The current pack still cannot emit an honest number because the concrete proxy family is absent.",
            ),
            row(
                "trial2_numeric_alpha_blocker_shrunk_to_chi_star_or_same_sector_proxy",
                "pass" if not chi_star_or_same_sector_proxy_available else "reject",
                "blocker shrunk to chi_star_or_same_sector_proxy family",
                1 if not chi_star_or_same_sector_proxy_available else 0,
                "The next route no longer targets a generic normalization pack; it targets the concrete missing proxy family.",
            ),
        ],
        {
            "trial2_numeric_alpha_computation_formula_ready": True,
            "trial2_numeric_alpha_closeout_ready": absolute_numeric_input_pack_ready,
            "trial2_numeric_alpha_numeric_from_current_pack_ready": absolute_numeric_input_pack_ready,
            "dominant_blocker_shrunk_from_absolute_normalization_input_pack_to_chi_star_or_same_sector_proxy": not chi_star_or_same_sector_proxy_available,
            "selected_residual_route": NEXT_RESIDUAL_ROUTE,
            "missing_v2_artifact": NEXT_MISSING_ARTIFACT,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_absolute_input_pack_gate_closed_proxy_family_open",
            "advance_to_8_7_56_710": True,
            "next_required_artifacts": [NEXT_RESIDUAL_ROUTE],
        },
        {
            "audit_summary": audit_payload["summary"],
            "prior_gate_summary": gate_summary,
            "prior_route_summary": route_summary,
        },
    )

    route_payload = payload(
        "8.7.56.710",
        "Trial-2 numeric alpha next-generation route contract seventy-fourth refresh",
        common_inputs,
        "Refresh the next-generation contract after the absolute-input-pack shrink: keep the precision-alpha mainline, keep the strong side on reserve, and promote the chi_star_or_same_sector_proxy family as the next official blocker.",
        {
            "selected_route_rule": "the next official route is the missing chi_star_or_same_sector_proxy family inside the computation-side normalization pack",
            "mainline_rule": "precision-alpha remains the next-generation mainline after the computation pivot",
            "reserve_rule": "strong-side non-Abelian, running, and confinement gaps remain on v3 hold reserve",
        },
        [
            row(
                "trial2_numeric_alpha_absolute_input_pack_gate_closed_before_refresh",
                "pass",
                "absolute-input-pack declaration gate closed before route refresh",
                1,
                "The next-generation contract is only refreshed after the branch closes its declaration gate.",
            ),
            row(
                "trial2_numeric_alpha_next_route_selected_as_chi_star_or_same_sector_proxy",
                "pass",
                "chi_star_or_same_sector_proxy route selected",
                1,
                "The next route now targets the concrete missing proxy family.",
            ),
            row(
                "trial2_numeric_alpha_precision_mainline_retained_after_absolute_pack_shrink",
                "pass",
                "precision-alpha mainline retained after absolute-input-pack shrink",
                1,
                "The mainline remains Trial-2 numeric alpha, not the strong-side reserve.",
            ),
            row(
                "trial2_numeric_alpha_strong_side_route_state_retained_after_absolute_pack_shrink",
                "pass",
                "strong-side route state retained after absolute-input-pack shrink",
                1,
                "The strong side remains exploratory and is not promoted by the current alpha residual shrink.",
            ),
        ],
        {
            "selected_next_generation_route": NEXT_RESIDUAL_ROUTE,
            "strong_side_route_state": route_summary["strong_side_route_state"],
            "precision_alpha_mainline_retained": True,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_route_contract_seventy_fourth_refresh_frozen",
            "advance_to_next_route": True,
            "next_required_artifacts": [
                "trial2_numeric_alpha_chi_star_or_same_sector_proxy_source_inventory",
                "trial2_numeric_alpha_chi_star_or_same_sector_proxy_audit",
            ],
        },
        {
            "gate_summary": gate_payload["summary"],
            "prior_route_summary": route_summary,
        },
    )

    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_absolute_normalization_input_pack_source_inventory",
        inventory_payload,
    )
    write_artifact("mass_origin_v2_trial2_numeric_alpha_absolute_normalization_input_pack_audit", audit_payload)
    write_artifact(
        "mass_origin_v2_trial2_numeric_alpha_absolute_normalization_input_pack_declaration_gate",
        gate_payload,
    )
    write_artifact("mass_origin_v2_t2_alpha_route_contract_seventy_fourth_refresh", route_payload)

    print("[ok] wrote:")
    print(" - mass_origin_v2_trial2_numeric_alpha_absolute_normalization_input_pack_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_absolute_normalization_input_pack_audit_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_absolute_normalization_input_pack_declaration_gate_metrics.json")
    print(" - mass_origin_v2_t2_alpha_route_contract_seventy_fourth_refresh_metrics.json")
    print(f" - next official branch should move to {NEXT_BRANCH}")


# Function: run the absolute-input-pack branch from the CLI.

def run_cli() -> None:
    """CLI entry point for the absolute-input-pack residual branch."""
    main()


if __name__ == "__main__":
    run_cli()
