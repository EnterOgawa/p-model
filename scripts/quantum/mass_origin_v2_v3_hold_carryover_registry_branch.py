#!/usr/bin/env python3
"""Generate 8.7.56.411-.414 v3.0 hold carry-over registry artifacts.

The integrated v2.0 checkpoint is already frozen. The next official work is not
another residual closure inside v2.0, but an explicit registry of what is being
carried into v3.0. This branch therefore:

1. inventories the precision-alpha and strong-side carry-over pack,
2. audits whether the carry-over classification is honest and complete,
3. freezes the v3.0 hold declaration gate, and
4. selects the next-generation route after the registry is fixed.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

TRIAL2_ALPHA_AUDIT = OUT / "mass_origin_v2_trial2_fine_structure_constant_coupling_mapping_audit_metrics.json"
TRIAL2_DECL = OUT / "mass_origin_v2_trial2_declaration_gate_metrics.json"
TRIAL4_STRUCT = OUT / "mass_origin_v2_trial4_su3_analogy_structural_audit_metrics.json"
TRIAL4_PILOT = OUT / "mass_origin_v2_trial4_running_confinement_qualitative_pilot_metrics.json"
TRIAL4_GATE = OUT / "mass_origin_v2_trial4_exploratory_declaration_v3_hold_gate_metrics.json"
INTEGRATED_AUDIT = OUT / "mass_origin_v2_integrated_closeout_audit_metrics.json"
INTEGRATED_GATE = OUT / "mass_origin_v2_integrated_declaration_gate_metrics.json"
V3_HOLD_CONTRACT = OUT / "mass_origin_v2_v3_hold_route_contract_metrics.json"

NEXT_ROUTE = "8.7.56.415"
NEXT_ROUTE_LABEL = "trial2_numeric_alpha_precision_carryover_resolution"

PART5_CURRENT_STATE = "mainline は numeric $\\alpha$ precision と strong-side missing structure を v3.0 carry-over registry"
PART5_NEXT_STEP = "8.7.56.411-.414"
STATUS_NEXT_STEP = "current official next step は `8.7.56.411`"
ROADMAP_NEXT_BRANCH = "8.7.56.411-.414"


# 関数: UTC 現在時刻を ISO 8601 形式で返す。
def now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力 path の存在を確認する。

def req(path: Path) -> None:
    """Abort immediately when a required input path is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: UTF-8 JSON artifact を読む。

def read_json(path: Path) -> dict:
    """Load a UTF-8 JSON artifact into a dictionary."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: UTF-8 text source を読む。

def read_text(path: Path) -> str:
    """Read a UTF-8 text source into memory."""
    return path.read_text(encoding="utf-8")


# 関数: repo 相対 POSIX path を返す。

def rel(path: Path) -> str:
    """Return a repository-relative POSIX path string."""
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: 指定した部分文字列の最初の hit 行を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first line hit for a substring pattern, if any."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: 共通 schema の row を組み立てる。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build a standard metrics row payload."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# 関数: 共通 schema の payload を組み立てる。

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
    """Build the standard JSON metrics payload used across the roadmap."""
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


# 関数: JSON artifact と rows CSV を side-by-side で保存する。

def write_artifact(stem: str, data: dict) -> None:
    """Write the metrics payload as JSON and as a rows CSV sidecar."""
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / f"{stem}_metrics.json"
    csv_path = OUT / f"{stem}_rows.csv"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# 関数: wording target の present/absent を監査する。

def audit_target(
    file_key: str,
    path: Path,
    text: str,
    pattern: str,
    note: str,
    expected_present: bool = True,
) -> dict:
    """Audit whether a wording target is present or intentionally absent."""
    target_hit = hit(text, pattern)
    present = target_hit is not None
    return {
        "file_key": file_key,
        "file": rel(path),
        "pattern": pattern,
        "expected_present": expected_present,
        "present": present,
        "matched_expectation": present is expected_present,
        "note": note,
        "evidence": target_hit,
    }


# 関数: carry-over source inventory を構築する。

def build_inventory(
    common_inputs: dict,
    trial2_alpha_audit: dict,
    trial2_decl: dict,
    trial4_struct: dict,
    trial4_pilot: dict,
    trial4_gate: dict,
    integrated_audit: dict,
    integrated_gate: dict,
    v3_hold_contract: dict,
    status_text: str,
    roadmap_text: str,
    part5_text: str,
) -> dict:
    """Freeze the v3.0 hold carry-over source pack."""
    inventory_targets = [
        audit_target(
            "status_next_step",
            STATUS,
            status_text,
            STATUS_NEXT_STEP,
            "STATUS must point to 8.7.56.411 as the current next official step.",
        ),
        audit_target(
            "roadmap_next_branch",
            ROADMAP,
            roadmap_text,
            ROADMAP_NEXT_BRANCH,
            "ROADMAP must still advertise 8.7.56.411-.414 as the current official branch.",
        ),
        audit_target(
            "part5_current_state",
            PART5,
            part5_text,
            PART5_CURRENT_STATE,
            "Part V must already say that the program moved from the integrated checkpoint into the v3 carry-over registry stage.",
        ),
        audit_target(
            "part5_next_step",
            PART5,
            part5_text,
            PART5_NEXT_STEP,
            "Part V must point to 8.7.56.411-.414 as the current official next step.",
        ),
    ]
    inventory_ready = all(item["matched_expectation"] for item in inventory_targets)
    integrated_checkpoint_frozen = bool(v3_hold_contract["summary"]["v2_integrated_checkpoint_frozen"])
    trial2_alpha_open = bool(not trial2_alpha_audit["summary"]["alpha_numeric_from_current_pack_ready"])
    trial4_strong_side_open = bool(
        not trial4_struct["summary"]["su3_analogy_structural_pass"]
        and not trial4_pilot["summary"]["running_qualitative_foothold_available"]
        and not trial4_pilot["summary"]["confinement_qualitative_foothold_available"]
    )

    return payload(
        "8.7.56.411",
        "v3_hold_carryover_source_inventory",
        common_inputs,
        "Inventory the carry-over pack that survives after the integrated v2.0 checkpoint freezes: numeric alpha precision on the EM side, strong-side missing non-Abelian/running/confinement structure, and the current registry wording across control docs and Part V.",
        {
            "registry_rule": "only items explicitly classified as carry-over after the integrated v2.0 closeout enter the v3.0 hold registry",
            "precision_rule": "numeric alpha precision remains open even though the Trial-2 structural route already passed",
            "strong_side_rule": "strong-side carry-over remains open while explicit SU(3)-like closure plus honest running/confinement are still absent",
        },
        [
            row(
                "v3_hold_inventory_targets_present",
                "pass" if inventory_ready else "reject",
                "carry-over wording targets present",
                sum(1 for item in inventory_targets if item["present"]),
                "All current wording surfaces must acknowledge the carry-over registry stage.",
            ),
            row(
                "v3_hold_integrated_checkpoint_frozen",
                "pass" if integrated_checkpoint_frozen else "reject",
                "integrated v2.0 checkpoint frozen",
                1 if integrated_checkpoint_frozen else 0,
                "The carry-over registry only starts after the integrated checkpoint is already frozen.",
            ),
            row(
                "v3_hold_trial2_numeric_alpha_open",
                "pass" if trial2_alpha_open else "watch",
                "Trial-2 numeric alpha precision open",
                1 if trial2_alpha_open else 0,
                "The symbolic alpha route is closed, but the numeric normalization remains unresolved.",
            ),
            row(
                "v3_hold_trial4_strong_side_open",
                "pass" if trial4_strong_side_open else "watch",
                "Trial-4 strong-side missing structure open",
                1 if trial4_strong_side_open else 0,
                "Strong-side non-Abelian closure, running, and confinement remain carry-over items.",
            ),
        ],
        {
            "inventory_ready": inventory_ready,
            "integrated_checkpoint_frozen": integrated_checkpoint_frozen,
            "trial2_numeric_alpha_carryover_open": trial2_alpha_open,
            "trial4_strong_side_carryover_open": trial4_strong_side_open,
            "required_target_count": len(inventory_targets),
            "present_target_count": sum(1 for item in inventory_targets if item["present"]),
            "first_route_to_close_or_none": "v3_hold_carryover_audit",
        },
        {
            "overall_status": "v3_hold_carryover_inventory_frozen" if inventory_ready else "v3_hold_carryover_inventory_incomplete",
            "advance_to_8_7_56_412": inventory_ready,
            "next_required_artifacts": [] if inventory_ready else ["v3_hold_carryover_source_inventory"],
        },
        {
            "inventory_targets": inventory_targets,
            "trial2_alpha_audit_summary": trial2_alpha_audit["summary"],
            "trial2_declaration_summary": trial2_decl["summary"],
            "trial4_structural_summary": trial4_struct["summary"],
            "trial4_pilot_summary": trial4_pilot["summary"],
            "trial4_gate_summary": trial4_gate["summary"],
            "integrated_audit_summary": integrated_audit["summary"],
            "integrated_gate_summary": integrated_gate["summary"],
            "v3_hold_contract_summary": v3_hold_contract["summary"],
        },
    )


# 関数: carry-over audit を構築する。

def build_audit(
    common_inputs: dict,
    inventory: dict,
    trial2_alpha_audit: dict,
    trial2_decl: dict,
    trial4_struct: dict,
    trial4_pilot: dict,
    trial4_gate: dict,
    integrated_audit: dict,
    v3_hold_contract: dict,
) -> dict:
    """Audit whether the v3.0 carry-over classification is coherent and useful."""
    inventory_ready = bool(inventory["summary"]["inventory_ready"])
    precision_alpha_open = bool(trial2_alpha_audit["summary"]["alpha_numeric_from_current_pack_ready"] is False)
    strong_side_open = bool(
        not trial4_struct["summary"]["su3_analogy_structural_pass"]
        or not trial4_pilot["summary"]["running_qualitative_foothold_available"]
        or not trial4_pilot["summary"]["confinement_qualitative_foothold_available"]
    )
    checkpoint_consistent = bool(integrated_audit["summary"]["v2_integrated_closeout_ready"])
    precision_route_preferred = bool(
        precision_alpha_open
        and trial2_decl["summary"]["v2_minimum_condition_satisfied_under_breakthrough_working_action"]
        and trial4_gate["summary"]["trial4_v3_mainline_promotion_ready"] is False
    )
    strong_side_route_direct_promotion_ready = bool(trial4_gate["summary"]["trial4_v3_mainline_promotion_ready"])
    carryover_registry_ready = bool(
        inventory_ready
        and checkpoint_consistent
        and precision_alpha_open
        and strong_side_open
        and v3_hold_contract["summary"]["trial2_numeric_alpha_carryover_required"]
        and v3_hold_contract["summary"]["trial4_strong_side_carryover_required"]
    )

    return payload(
        "8.7.56.412",
        "v3_hold_carryover_audit",
        common_inputs,
        "Audit whether the remaining open items are honestly classified as v3.0 carry-over, and decide which carry-over route should be treated as the first next-generation mainline candidate.",
        {
            "classification_rule": "a carry-over item is honest only if the current v2.0 checkpoint already closes without it",
            "priority_rule": "prefer the carry-over route that is narrower and already rests on a structurally closed branch",
            "promotion_rule": "strong-side work stays on reserve unless an honest non-Abelian/running/confinement promotion surface exists",
        },
        [
            row(
                "v3_hold_checkpoint_consistent",
                "pass" if checkpoint_consistent else "reject",
                "integrated checkpoint remains consistent",
                1 if checkpoint_consistent else 0,
                "Carry-over classification is only meaningful if the integrated checkpoint stays coherent.",
            ),
            row(
                "v3_hold_precision_alpha_open",
                "pass" if precision_alpha_open else "watch",
                "numeric alpha precision remains open",
                1 if precision_alpha_open else 0,
                "The precision-alpha problem survives as an explicit carry-over item.",
            ),
            row(
                "v3_hold_strong_side_open",
                "pass" if strong_side_open else "watch",
                "strong-side non-Abelian/running/confinement gaps remain open",
                1 if strong_side_open else 0,
                "The strong-side branch remains exploratory only.",
            ),
            row(
                "v3_hold_precision_route_preferred",
                "pass" if precision_route_preferred else "watch",
                "precision alpha carry-over route preferred",
                1 if precision_route_preferred else 0,
                "Numeric alpha is the narrower next-generation route because the EM structure is already closed while the strong side is not.",
            ),
            row(
                "v3_hold_strong_side_direct_promotion_ready",
                "reject" if not strong_side_route_direct_promotion_ready else "pass",
                "strong-side direct promotion ready",
                1 if strong_side_route_direct_promotion_ready else 0,
                "Strong-side direct promotion would require explicit SU(3)-like closure plus honest running/confinement footholds.",
            ),
        ],
        {
            "checkpoint_consistent": checkpoint_consistent,
            "precision_alpha_open": precision_alpha_open,
            "strong_side_open": strong_side_open,
            "precision_alpha_followthrough_preferred": precision_route_preferred,
            "strong_side_direct_mainline_promotion_ready": strong_side_route_direct_promotion_ready,
            "carryover_registry_ready": carryover_registry_ready,
            "first_route_to_close_or_none": "v3_hold_declaration_gate",
        },
        {
            "overall_status": "v3_hold_carryover_audit_complete" if carryover_registry_ready else "v3_hold_carryover_audit_incomplete",
            "advance_to_8_7_56_413": carryover_registry_ready,
            "next_required_artifacts": [] if carryover_registry_ready else ["v3_hold_carryover_audit"],
        },
        {
            "inventory_summary": inventory["summary"],
            "trial2_alpha_audit_summary": trial2_alpha_audit["summary"],
            "trial2_declaration_summary": trial2_decl["summary"],
            "trial4_structural_summary": trial4_struct["summary"],
            "trial4_pilot_summary": trial4_pilot["summary"],
            "trial4_gate_summary": trial4_gate["summary"],
            "integrated_audit_summary": integrated_audit["summary"],
            "v3_hold_contract_summary": v3_hold_contract["summary"],
        },
    )


# 関数: v3 hold declaration gate を構築する。

def build_gate(common_inputs: dict, audit: dict, trial4_gate: dict) -> dict:
    """Freeze the v3.0 hold declaration gate."""
    carryover_ready = bool(audit["summary"]["carryover_registry_ready"])
    precision_route_preferred = bool(audit["summary"]["precision_alpha_followthrough_preferred"])

    return payload(
        "8.7.56.413",
        "v3_hold_declaration_gate",
        common_inputs,
        "Freeze the v3.0 hold registry as the official post-v2 boundary, with numeric alpha precision selected as the first next-generation follow-through route and the strong side retained on reserve.",
        {
            "gate_rule": "close the v3 hold declaration once the carry-over classification is complete and the next-generation priority order is explicit",
            "selection_rule": "precision alpha becomes the first next-generation route when the EM structure is already closed and only numeric normalization remains open",
        },
        [
            row(
                "v3_hold_declaration_gate_complete",
                "pass" if carryover_ready else "reject",
                "v3 hold declaration gate complete",
                1 if carryover_ready else 0,
                "The v3 hold declaration becomes official only after the carry-over audit passes.",
            ),
            row(
                "v3_hold_registry_frozen",
                "pass" if carryover_ready else "reject",
                "v3 hold registry frozen",
                1 if carryover_ready else 0,
                "The program boundary after v2.0 is now frozen as a v3 hold registry rather than another v2 checkpoint.",
            ),
            row(
                "v3_hold_precision_route_selected_first",
                "pass" if precision_route_preferred else "watch",
                "precision alpha route selected first",
                1 if precision_route_preferred else 0,
                "Numeric alpha precision is selected as the first next-generation route.",
            ),
            row(
                "v3_hold_strong_side_reserve_retained",
                "pass",
                "strong-side reserve retained",
                1,
                "Strong-side missing structure remains on reserve while the next-generation precision route runs first.",
            ),
        ],
        {
            "v3_hold_registry_complete": carryover_ready,
            "v3_hold_program_boundary_frozen": carryover_ready,
            "first_next_generation_route_selected": NEXT_ROUTE_LABEL if precision_route_preferred else None,
            "strong_side_route_state": "v3_hold_reserve",
            "recommended_next_route_or_none": "8.7.56.414",
        },
        {
            "overall_status": "v3_hold_declaration_closed" if carryover_ready else "v3_hold_declaration_open",
            "advance_to_8_7_56_414": carryover_ready,
            "next_required_artifacts": [] if carryover_ready else ["v3_hold_declaration_gate"],
        },
        {
            "audit_summary": audit["summary"],
            "trial4_gate_summary": trial4_gate["summary"],
        },
    )


# 関数: next-generation route contract を構築する。

def build_contract(common_inputs: dict, gate: dict, audit: dict, trial4_gate: dict) -> dict:
    """Freeze the next-generation route contract after the v3 hold registry closes."""
    gate_closed = bool(gate["summary"]["v3_hold_registry_complete"])
    precision_route_preferred = bool(audit["summary"]["precision_alpha_followthrough_preferred"])

    return payload(
        "8.7.56.414",
        "v3_next_generation_route_contract",
        common_inputs,
        "Formalize the first next-generation route after the v3 hold registry closes: pursue numeric alpha precision first, retain strong-side work on reserve, and keep the integrated v2.0 checkpoint frozen.",
        {
            "contract_rule": "once the v3 hold registry closes, the next-generation route is the highest-readiness carry-over path rather than a reopened v2 loop",
            "reserve_rule": "strong-side missing structure remains on v3 reserve until a stronger canon surface exists",
        },
        [
            row(
                "v3_hold_registry_closed",
                "pass" if gate_closed else "reject",
                "v3 hold registry closed",
                1 if gate_closed else 0,
                "The next-generation route contract depends on the registry gate being closed first.",
            ),
            row(
                "v3_next_generation_precision_alpha_route_selected",
                "pass" if precision_route_preferred else "watch",
                "precision alpha next-generation route selected",
                1 if precision_route_preferred else 0,
                "The first next-generation route is the numeric alpha precision carry-over resolution branch.",
            ),
            row(
                "v3_next_generation_strong_side_reserve_retained",
                "pass",
                "strong-side reserve retained",
                1,
                "Strong-side gaps remain active but do not outrank the precision-alpha route.",
            ),
            row(
                "v2_integrated_checkpoint_remains_frozen",
                "pass" if gate_closed else "reject",
                "v2 integrated checkpoint remains frozen",
                1 if gate_closed else 0,
                "The post-v2 next-generation work must not reopen the frozen integrated checkpoint.",
            ),
        ],
        {
            "selected_next_generation_route": NEXT_ROUTE_LABEL if precision_route_preferred else None,
            "strong_side_route_state": "v3_hold_reserve",
            "v2_integrated_checkpoint_still_frozen": gate_closed,
            "trial4_direct_promotion_ready": trial4_gate["summary"]["trial4_v3_mainline_promotion_ready"],
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "v3_next_generation_route_contract_frozen" if gate_closed else "v3_next_generation_route_contract_pending",
            "advance_to_next_route": gate_closed,
            "next_required_artifacts": [
                "trial2_numeric_alpha_precision_carryover_source_inventory",
                "trial2_numeric_alpha_precision_carryover_audit",
            ]
            if gate_closed
            else ["v3_next_generation_route_contract"],
        },
        {
            "audit_summary": audit["summary"],
            "gate_summary": gate["summary"],
            "trial4_gate_summary": trial4_gate["summary"],
        },
    )


# 関数: current branch を実行する。

def main() -> None:
    """Execute the v3.0 hold carry-over registry branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        PART5,
        TRIAL2_ALPHA_AUDIT,
        TRIAL2_DECL,
        TRIAL4_STRUCT,
        TRIAL4_PILOT,
        TRIAL4_GATE,
        INTEGRATED_AUDIT,
        INTEGRATED_GATE,
        V3_HOLD_CONTRACT,
    ):
        req(path)

    common_inputs = {
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "part5_future_predictions_markdown": rel(PART5),
        "mass_origin_v2_trial2_fine_structure_constant_coupling_mapping_audit_json": rel(TRIAL2_ALPHA_AUDIT),
        "mass_origin_v2_trial2_declaration_gate_json": rel(TRIAL2_DECL),
        "mass_origin_v2_trial4_su3_analogy_structural_audit_json": rel(TRIAL4_STRUCT),
        "mass_origin_v2_trial4_running_confinement_qualitative_pilot_json": rel(TRIAL4_PILOT),
        "mass_origin_v2_trial4_exploratory_declaration_v3_hold_gate_json": rel(TRIAL4_GATE),
        "mass_origin_v2_integrated_closeout_audit_json": rel(INTEGRATED_AUDIT),
        "mass_origin_v2_integrated_declaration_gate_json": rel(INTEGRATED_GATE),
        "mass_origin_v2_v3_hold_route_contract_json": rel(V3_HOLD_CONTRACT),
    }

    trial2_alpha_audit = read_json(TRIAL2_ALPHA_AUDIT)
    trial2_decl = read_json(TRIAL2_DECL)
    trial4_struct = read_json(TRIAL4_STRUCT)
    trial4_pilot = read_json(TRIAL4_PILOT)
    trial4_gate = read_json(TRIAL4_GATE)
    integrated_audit = read_json(INTEGRATED_AUDIT)
    integrated_gate = read_json(INTEGRATED_GATE)
    v3_hold_contract = read_json(V3_HOLD_CONTRACT)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    part5_text = read_text(PART5)

    inventory = build_inventory(
        common_inputs,
        trial2_alpha_audit,
        trial2_decl,
        trial4_struct,
        trial4_pilot,
        trial4_gate,
        integrated_audit,
        integrated_gate,
        v3_hold_contract,
        status_text,
        roadmap_text,
        part5_text,
    )
    audit = build_audit(
        common_inputs,
        inventory,
        trial2_alpha_audit,
        trial2_decl,
        trial4_struct,
        trial4_pilot,
        trial4_gate,
        integrated_audit,
        v3_hold_contract,
    )
    gate = build_gate(common_inputs, audit, trial4_gate)
    contract = build_contract(common_inputs, gate, audit, trial4_gate)

    write_artifact("mass_origin_v2_v3_hold_carryover_source_inventory", inventory)
    write_artifact("mass_origin_v2_v3_hold_carryover_audit", audit)
    write_artifact("mass_origin_v2_v3_hold_declaration_gate", gate)
    write_artifact("mass_origin_v3_next_generation_route_contract", contract)

    print("[ok] generated v3 hold carry-over registry artifacts:")
    print(" - mass_origin_v2_v3_hold_carryover_source_inventory_metrics.json")
    print(" - mass_origin_v2_v3_hold_carryover_audit_metrics.json")
    print(" - mass_origin_v2_v3_hold_declaration_gate_metrics.json")
    print(" - mass_origin_v3_next_generation_route_contract_metrics.json")


# 関数: CLI 直実行時に branch main を起動する。

if __name__ == "__main__":
    main()
