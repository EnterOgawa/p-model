#!/usr/bin/env python3
"""Generate 8.7.56.415-.418 Trial-2 numeric alpha carry-over artifacts.

The v3.0 hold registry selected the Trial-2 numeric-alpha problem as the first
next-generation route. This branch does not reopen the structural EM closure.
Instead it:

1. inventories the surviving numeric-alpha precision pack,
2. audits the remaining normalization blocker,
3. freezes the carry-over declaration gate, and
4. refreshes the strong-side reserve while selecting the next residual route.
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
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

TRIAL2_COULOMB = OUT / "mass_origin_v2_trial2_curvature_coulomb_pilot_metrics.json"
TRIAL2_ALPHA_AUDIT = OUT / "mass_origin_v2_trial2_fine_structure_constant_coupling_mapping_audit_metrics.json"
TRIAL2_DECL = OUT / "mass_origin_v2_trial2_declaration_gate_metrics.json"
TRIAL2_PAPER_SYNC = OUT / "mass_origin_v2_trial2_paper_side_sync_reopened_declaration_gate_metrics.json"
V3_HOLD_AUDIT = OUT / "mass_origin_v2_v3_hold_carryover_audit_metrics.json"
V3_HOLD_GATE = OUT / "mass_origin_v2_v3_hold_declaration_gate_metrics.json"
V3_NEXT_ROUTE = OUT / "mass_origin_v3_next_generation_route_contract_metrics.json"
QED_PRECISION = OUT / "qed_vacuum_precision_metrics.json"

NEXT_ROUTE = "8.7.56.419"
NEXT_ROUTE_LABEL = "trial2_numeric_alpha_independent_normalization_source_identification"

STATUS_NEXT_STEP = "current official next step は `8.7.56.415`"
ROADMAP_NEXT_BRANCH = "8.7.56.415-.418"
PART3A_E_FORMULA = "$e=g_P/\\sqrt{Z_P}$"
PART3A_ALPHA_FORMULA = "$\\alpha=g_P^2/(4\\pi Z_P\\hbar c)$"
PART3A_STRUCTURAL_PASS = "foundational / structural pass (numeric α open)"
PART5_NEXT_STEP = "8.7.56.415-.418"


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


# 関数: numeric alpha carry-over source inventory を構築する。

def build_inventory(
    common_inputs: dict,
    trial2_coulomb: dict,
    trial2_alpha_audit: dict,
    trial2_decl: dict,
    trial2_paper_sync: dict,
    v3_hold_audit: dict,
    v3_hold_gate: dict,
    v3_next_route: dict,
    qed_precision: dict,
    status_text: str,
    roadmap_text: str,
    part3a_text: str,
    part5_text: str,
) -> dict:
    """Freeze the source inventory for the numeric-alpha carry-over route."""
    inventory_targets = [
        audit_target(
            "status_next_step",
            STATUS,
            status_text,
            STATUS_NEXT_STEP,
            "STATUS must point to 8.7.56.415 as the current next official step.",
        ),
        audit_target(
            "roadmap_next_branch",
            ROADMAP,
            roadmap_text,
            ROADMAP_NEXT_BRANCH,
            "ROADMAP must advertise 8.7.56.415-.418 as the current official branch.",
        ),
        audit_target(
            "part3a_e_formula",
            PART3A,
            part3a_text,
            PART3A_E_FORMULA,
            "Part III-A must keep the structural electric-charge formula.",
        ),
        audit_target(
            "part3a_alpha_formula",
            PART3A,
            part3a_text,
            PART3A_ALPHA_FORMULA,
            "Part III-A must keep the structural alpha formula.",
        ),
        audit_target(
            "part3a_structural_pass",
            PART3A,
            part3a_text,
            PART3A_STRUCTURAL_PASS,
            "Part III-A must still classify Trial-2 as a structural pass with numeric alpha open.",
        ),
        audit_target(
            "part5_next_step",
            PART5,
            part5_text,
            PART5_NEXT_STEP,
            "Part V must point to the numeric-alpha carry-over branch as the current next step.",
        ),
    ]
    inventory_ready = all(item["matched_expectation"] for item in inventory_targets)
    precision_target_pack_available = bool(
        trial2_alpha_audit["summary"]["alpha_target_inverse_value"] == qed_precision["sources"][4]["extracted_value"]["alpha_inv"]
    )
    current_numeric_gap_open = bool(
        not trial2_alpha_audit["summary"]["alpha_numeric_from_current_pack_ready"]
        and not trial2_coulomb["summary"]["coulomb_normalization_numeric_ready"]
    )

    return payload(
        "8.7.56.415",
        "trial2_numeric_alpha_precision_carryover_source_inventory",
        common_inputs,
        "Inventory the surviving numeric-alpha carry-over pack: structural formulas, precision target, current normalization gap, paper-side synced wording, and the v3 hold route selection that chose this route first.",
        {
            "formula_rule": "retain the structural formulas e = g_P / sqrt(Z_P) and alpha = g_P^2 / (4 pi Z_P hbar c) without reopening the structural EM pass",
            "precision_rule": "reuse the cached QED alpha target as the precision benchmark for the carry-over route",
            "gap_rule": "the current carry-over problem is the missing independent numeric normalization, not the symbolic alpha formula",
        },
        [
            row(
                "trial2_numeric_alpha_inventory_targets_present",
                "pass" if inventory_ready else "reject",
                "numeric-alpha carry-over wording targets present",
                sum(1 for item in inventory_targets if item["present"]),
                "All current wording surfaces must acknowledge the numeric-alpha carry-over stage.",
            ),
            row(
                "trial2_numeric_alpha_structural_formula_pack_ready",
                "pass"
                if trial2_alpha_audit["summary"]["electric_charge_formula_ready"]
                and trial2_alpha_audit["summary"]["alpha_formula_ready"]
                else "reject",
                "structural formula pack ready",
                1
                if trial2_alpha_audit["summary"]["electric_charge_formula_ready"]
                and trial2_alpha_audit["summary"]["alpha_formula_ready"]
                else 0,
                "The numeric carry-over route starts from a structurally closed formula pack.",
            ),
            row(
                "trial2_numeric_alpha_precision_target_pack_available",
                "pass" if precision_target_pack_available else "reject",
                "precision target pack available",
                1 if precision_target_pack_available else 0,
                "The route already has a cached precision alpha benchmark.",
            ),
            row(
                "trial2_numeric_alpha_current_normalization_gap_open",
                "pass" if current_numeric_gap_open else "watch",
                "current normalization gap open",
                1 if current_numeric_gap_open else 0,
                "The structural route still lacks an independent numeric normalization source.",
            ),
        ],
        {
            "inventory_ready": inventory_ready,
            "structural_formula_pack_ready": bool(
                trial2_alpha_audit["summary"]["electric_charge_formula_ready"]
                and trial2_alpha_audit["summary"]["alpha_formula_ready"]
            ),
            "precision_target_pack_available": precision_target_pack_available,
            "current_numeric_normalization_gap_open": current_numeric_gap_open,
            "trial2_paper_state_synced": trial2_paper_sync["summary"]["trial2_current_paper_state_synced"],
            "first_route_to_close_or_none": "trial2_numeric_alpha_precision_carryover_audit",
        },
        {
            "overall_status": "trial2_numeric_alpha_precision_inventory_frozen" if inventory_ready else "trial2_numeric_alpha_precision_inventory_incomplete",
            "advance_to_8_7_56_416": inventory_ready,
            "next_required_artifacts": [] if inventory_ready else ["trial2_numeric_alpha_precision_carryover_source_inventory"],
        },
        {
            "inventory_targets": inventory_targets,
            "trial2_coulomb_summary": trial2_coulomb["summary"],
            "trial2_alpha_audit_summary": trial2_alpha_audit["summary"],
            "trial2_declaration_summary": trial2_decl["summary"],
            "trial2_paper_sync_summary": trial2_paper_sync["summary"],
            "v3_hold_audit_summary": v3_hold_audit["summary"],
            "v3_hold_gate_summary": v3_hold_gate["summary"],
            "v3_next_route_summary": v3_next_route["summary"],
            "qed_precision_alpha_target": qed_precision["sources"][4]["extracted_value"],
        },
    )


# 関数: numeric alpha carry-over audit を構築する。

def build_audit(
    common_inputs: dict,
    inventory: dict,
    trial2_coulomb: dict,
    trial2_alpha_audit: dict,
    trial2_decl: dict,
    v3_next_route: dict,
) -> dict:
    """Audit the remaining numeric-alpha normalization blocker."""
    inventory_ready = bool(inventory["summary"]["inventory_ready"])
    structural_formula_ready = bool(inventory["summary"]["structural_formula_pack_ready"])
    precision_target_pack_available = bool(inventory["summary"]["precision_target_pack_available"])
    current_numeric_prediction_available = bool(trial2_alpha_audit["summary"]["alpha_numeric_from_current_pack_ready"])
    coulomb_normalization_ready = bool(trial2_coulomb["summary"]["coulomb_normalization_numeric_ready"])
    independent_normalization_source_available = bool(current_numeric_prediction_available or coulomb_normalization_ready)
    dominant_blocker_is_independent_normalization_source = bool(
        structural_formula_ready and precision_target_pack_available and not independent_normalization_source_available
    )
    precision_route_still_selected = bool(
        v3_next_route["summary"]["selected_next_generation_route"] == "trial2_numeric_alpha_precision_carryover_resolution"
    )
    audit_ready = bool(inventory_ready and dominant_blocker_is_independent_normalization_source and precision_route_still_selected)

    return payload(
        "8.7.56.416",
        "trial2_numeric_alpha_precision_carryover_audit",
        common_inputs,
        "Audit whether the numeric-alpha carry-over problem has now been reduced to a single independent normalization-source blocker under the current canon.",
        {
            "blocker_rule": "once the symbolic formula and the precision benchmark are both fixed, the remaining blocker is the missing independent normalization source for g_P / sqrt(Z_P)",
            "nonreopen_rule": "the audit must not reopen the already-passed structural EM branch",
            "reserve_rule": "strong-side work remains on reserve regardless of the precision-alpha blocker state",
        },
        [
            row(
                "trial2_numeric_alpha_structural_formula_still_ready",
                "pass" if structural_formula_ready else "reject",
                "structural alpha formula still ready",
                1 if structural_formula_ready else 0,
                "The current branch starts from the already-passed structural EM route.",
            ),
            row(
                "trial2_numeric_alpha_precision_target_available",
                "pass" if precision_target_pack_available else "reject",
                "precision target available",
                1 if precision_target_pack_available else 0,
                "The QED alpha benchmark remains available as the numeric target.",
            ),
            row(
                "trial2_numeric_alpha_independent_normalization_source_available",
                "reject" if not independent_normalization_source_available else "pass",
                "independent normalization source available",
                1 if independent_normalization_source_available else 0,
                "Current canon still lacks an independent numeric normalization source for the structural charge formula.",
            ),
            row(
                "trial2_numeric_alpha_dominant_blocker_is_independent_normalization_source",
                "pass" if dominant_blocker_is_independent_normalization_source else "watch",
                "dominant blocker is independent normalization source",
                1 if dominant_blocker_is_independent_normalization_source else 0,
                "The residual problem is no longer the formula or the target pack but the missing normalization source.",
            ),
            row(
                "trial2_numeric_alpha_precision_route_still_selected",
                "pass" if precision_route_still_selected else "reject",
                "precision route still selected",
                1 if precision_route_still_selected else 0,
                "The current mainline remains the numeric-alpha carry-over route.",
            ),
        ],
        {
            "audit_ready": audit_ready,
            "structural_formula_ready": structural_formula_ready,
            "precision_target_pack_available": precision_target_pack_available,
            "current_numeric_prediction_available": current_numeric_prediction_available,
            "coulomb_normalization_numeric_ready": coulomb_normalization_ready,
            "independent_normalization_source_available_under_current_canon": independent_normalization_source_available,
            "dominant_blocker_is_independent_normalization_source": dominant_blocker_is_independent_normalization_source,
            "first_route_to_close_or_none": "trial2_numeric_alpha_precision_carryover_declaration_gate",
        },
        {
            "overall_status": "trial2_numeric_alpha_precision_audit_complete" if audit_ready else "trial2_numeric_alpha_precision_audit_incomplete",
            "advance_to_8_7_56_417": audit_ready,
            "next_required_artifacts": [] if audit_ready else ["trial2_numeric_alpha_precision_carryover_audit"],
        },
        {
            "inventory_summary": inventory["summary"],
            "trial2_coulomb_summary": trial2_coulomb["summary"],
            "trial2_alpha_audit_summary": trial2_alpha_audit["summary"],
            "trial2_declaration_summary": trial2_decl["summary"],
            "v3_next_route_summary": v3_next_route["summary"],
        },
    )


# 関数: numeric alpha carry-over declaration gate を構築する。

def build_gate(common_inputs: dict, audit: dict) -> dict:
    """Freeze the declaration gate for the numeric-alpha carry-over route."""
    audit_ready = bool(audit["summary"]["audit_ready"])
    closeout_ready = bool(audit["summary"]["independent_normalization_source_available_under_current_canon"])

    return payload(
        "8.7.56.417",
        "trial2_numeric_alpha_precision_carryover_declaration_gate",
        common_inputs,
        "Freeze the declaration gate for the numeric-alpha carry-over route: isolate the remaining blocker, keep the structural EM pass frozen, and point to the next residual route.",
        {
            "gate_rule": "close the current carry-over branch once the blocker is isolated even if the numeric-alpha closeout itself is still pending",
            "residual_rule": "the next residual route identifies the missing independent normalization source rather than reopening structural EM derivations",
        },
        [
            row(
                "trial2_numeric_alpha_precision_gate_complete",
                "pass" if audit_ready else "reject",
                "numeric-alpha carry-over gate complete",
                1 if audit_ready else 0,
                "The gate closes once the residual blocker is isolated.",
            ),
            row(
                "trial2_numeric_alpha_closeout_ready",
                "pass" if closeout_ready else "reject",
                "numeric-alpha closeout ready",
                1 if closeout_ready else 0,
                "Numeric-alpha closeout would require an explicit independent normalization source under the current canon.",
            ),
            row(
                "trial2_numeric_alpha_independent_normalization_source_missing",
                "pass" if not closeout_ready else "watch",
                "independent normalization source missing",
                1 if not closeout_ready else 0,
                "The residual blocker is the missing independent normalization source.",
            ),
            row(
                "trial2_numeric_alpha_structural_em_pass_preserved",
                "pass",
                "structural EM pass preserved",
                1,
                "The branch does not reopen the structural EM pass while chasing the numeric blocker.",
            ),
        ],
        {
            "trial2_numeric_alpha_precision_branch_closeable": audit_ready,
            "trial2_numeric_alpha_closeout_ready": closeout_ready,
            "selected_residual_route": NEXT_ROUTE_LABEL,
            "missing_v2_artifact": "trial2_numeric_alpha_independent_normalization_source_pack",
            "recommended_next_route_or_none": "8.7.56.418",
        },
        {
            "overall_status": "trial2_numeric_alpha_precision_gate_closed" if audit_ready else "trial2_numeric_alpha_precision_gate_open",
            "advance_to_8_7_56_418": audit_ready,
            "next_required_artifacts": [] if audit_ready else ["trial2_numeric_alpha_precision_carryover_declaration_gate"],
        },
        {
            "audit_summary": audit["summary"],
        },
    )


# 関数: strong-side reserve refresh / next-generation route contract を構築する。

def build_contract(common_inputs: dict, gate: dict, v3_hold_gate: dict, v3_next_route: dict) -> dict:
    """Refresh the strong-side reserve and freeze the next route contract."""
    gate_closed = bool(gate["summary"]["trial2_numeric_alpha_precision_branch_closeable"])
    residual_route_selected = gate["summary"]["selected_residual_route"] == NEXT_ROUTE_LABEL

    return payload(
        "8.7.56.418",
        "trial2_numeric_alpha_next_generation_route_contract",
        common_inputs,
        "Refresh the strong-side reserve after the precision-alpha branch and freeze the next residual route inside the selected next-generation EM carry-over program.",
        {
            "contract_rule": "the next official route remains inside the numeric-alpha carry-over program until the independent normalization source is identified",
            "reserve_rule": "strong-side work remains on reserve while the precision-alpha route continues",
        },
        [
            row(
                "trial2_numeric_alpha_precision_branch_gate_closed",
                "pass" if gate_closed else "reject",
                "numeric-alpha branch gate closed",
                1 if gate_closed else 0,
                "The next route contract depends on the declaration gate being frozen first.",
            ),
            row(
                "trial2_numeric_alpha_residual_route_selected",
                "pass" if residual_route_selected else "reject",
                "numeric-alpha residual route selected",
                1 if residual_route_selected else 0,
                "The next official route stays inside the numeric-alpha program.",
            ),
            row(
                "trial2_numeric_alpha_strong_side_reserve_retained",
                "pass",
                "strong-side reserve retained",
                1,
                "Strong-side non-Abelian/running/confinement gaps remain on reserve and do not outrank the EM precision route.",
            ),
            row(
                "trial2_numeric_alpha_first_next_generation_route_preserved",
                "pass"
                if v3_next_route["summary"]["selected_next_generation_route"] == "trial2_numeric_alpha_precision_carryover_resolution"
                else "reject",
                "first next-generation route preserved",
                1
                if v3_next_route["summary"]["selected_next_generation_route"] == "trial2_numeric_alpha_precision_carryover_resolution"
                else 0,
                "The precision-alpha program remains the selected next-generation mainline.",
            ),
        ],
        {
            "selected_next_generation_route": NEXT_ROUTE_LABEL if residual_route_selected else None,
            "strong_side_route_state": v3_hold_gate["summary"]["strong_side_route_state"],
            "precision_alpha_mainline_retained": True,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_next_route_contract_frozen" if gate_closed else "trial2_numeric_alpha_next_route_contract_pending",
            "advance_to_next_route": gate_closed,
            "next_required_artifacts": [
                "trial2_numeric_alpha_independent_normalization_source_inventory",
                "trial2_numeric_alpha_independent_normalization_source_audit",
            ]
            if gate_closed
            else ["trial2_numeric_alpha_next_generation_route_contract"],
        },
        {
            "gate_summary": gate["summary"],
            "v3_hold_gate_summary": v3_hold_gate["summary"],
            "v3_next_route_summary": v3_next_route["summary"],
        },
    )


# 関数: current branch を実行する。

def main() -> None:
    """Execute the Trial-2 numeric-alpha carry-over branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        PART3A,
        PART5,
        TRIAL2_COULOMB,
        TRIAL2_ALPHA_AUDIT,
        TRIAL2_DECL,
        TRIAL2_PAPER_SYNC,
        V3_HOLD_AUDIT,
        V3_HOLD_GATE,
        V3_NEXT_ROUTE,
        QED_PRECISION,
    ):
        req(path)

    common_inputs = {
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "part3a_quantum_foundations_markdown": rel(PART3A),
        "part5_future_predictions_markdown": rel(PART5),
        "mass_origin_v2_trial2_curvature_coulomb_pilot_json": rel(TRIAL2_COULOMB),
        "mass_origin_v2_trial2_fine_structure_constant_coupling_mapping_audit_json": rel(TRIAL2_ALPHA_AUDIT),
        "mass_origin_v2_trial2_declaration_gate_json": rel(TRIAL2_DECL),
        "mass_origin_v2_trial2_paper_side_sync_reopened_declaration_gate_json": rel(TRIAL2_PAPER_SYNC),
        "mass_origin_v2_v3_hold_carryover_audit_json": rel(V3_HOLD_AUDIT),
        "mass_origin_v2_v3_hold_declaration_gate_json": rel(V3_HOLD_GATE),
        "mass_origin_v3_next_generation_route_contract_json": rel(V3_NEXT_ROUTE),
        "qed_vacuum_precision_metrics_json": rel(QED_PRECISION),
    }

    trial2_coulomb = read_json(TRIAL2_COULOMB)
    trial2_alpha_audit = read_json(TRIAL2_ALPHA_AUDIT)
    trial2_decl = read_json(TRIAL2_DECL)
    trial2_paper_sync = read_json(TRIAL2_PAPER_SYNC)
    v3_hold_audit = read_json(V3_HOLD_AUDIT)
    v3_hold_gate = read_json(V3_HOLD_GATE)
    v3_next_route = read_json(V3_NEXT_ROUTE)
    qed_precision = read_json(QED_PRECISION)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)

    inventory = build_inventory(
        common_inputs,
        trial2_coulomb,
        trial2_alpha_audit,
        trial2_decl,
        trial2_paper_sync,
        v3_hold_audit,
        v3_hold_gate,
        v3_next_route,
        qed_precision,
        status_text,
        roadmap_text,
        part3a_text,
        part5_text,
    )
    audit = build_audit(
        common_inputs,
        inventory,
        trial2_coulomb,
        trial2_alpha_audit,
        trial2_decl,
        v3_next_route,
    )
    gate = build_gate(common_inputs, audit)
    contract = build_contract(common_inputs, gate, v3_hold_gate, v3_next_route)

    write_artifact("mass_origin_v2_trial2_numeric_alpha_precision_carryover_source_inventory", inventory)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_precision_carryover_audit", audit)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_precision_carryover_declaration_gate", gate)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_next_generation_route_contract", contract)

    print("[ok] generated Trial-2 numeric alpha precision carry-over artifacts:")
    print(" - mass_origin_v2_trial2_numeric_alpha_precision_carryover_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_precision_carryover_audit_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_precision_carryover_declaration_gate_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_next_generation_route_contract_metrics.json")


# 関数: CLI 直実行時に branch main を起動する。

if __name__ == "__main__":
    main()
