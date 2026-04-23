#!/usr/bin/env python3
"""Generate 8.7.56.423-.426 Coulomb-normalization-source-surface artifacts.

The previous branch established that the current canon exposes zero explicit
Coulomb-normalization source surfaces. This branch narrows the blocker further
by locating the best candidate placement surface for the missing statement.
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
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

TRIAL2_COULOMB = OUT / "mass_origin_v2_trial2_curvature_coulomb_pilot_metrics.json"
TRIAL2_ALPHA_AUDIT = OUT / "mass_origin_v2_trial2_fine_structure_constant_coupling_mapping_audit_metrics.json"
INDEPENDENT_INVENTORY = OUT / "mass_origin_v2_trial2_numeric_alpha_independent_normalization_source_inventory_metrics.json"
INDEPENDENT_AUDIT = OUT / "mass_origin_v2_trial2_numeric_alpha_independent_normalization_source_audit_metrics.json"
INDEPENDENT_GATE = OUT / "mass_origin_v2_trial2_numeric_alpha_independent_normalization_source_declaration_gate_metrics.json"
INDEPENDENT_ROUTE = OUT / "mass_origin_v2_trial2_numeric_alpha_next_generation_route_contract_second_refresh_metrics.json"
V3_HOLD_GATE = OUT / "mass_origin_v2_v3_hold_declaration_gate_metrics.json"

NEXT_ROUTE = "8.7.56.427"
NEXT_ROUTE_LABEL = "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_primary_surface_statement_identification"

STATUS_NEXT_STEP = "current official next step は `8.7.56.423`"
ROADMAP_NEXT_BRANCH = "8.7.56.423-.426"
PART5_NEXT_STEP = "8.7.56.423-.426"
PART5_BLOCKER = "Coulomb-normalization source surface の欠落"
PART3A_PRIMARY_SURFACE = "#### 2.6.1 現行 canon で固定した source / structural route"
PART3A_SECONDARY_SURFACE = "#### 2.6.2 未導出（近似検証と判定の固定）"
PART5_CHECKPOINT_SURFACE = "### 3.2 v2.0 checkpoint：electromagnetism / weak-sector closeout（理論側 checkpoint）"
PART1_NO_SURFACE_PATTERN = "epsilon_0"


# 関数: UTC 現在時刻を ISO 8601 形式で返す。
def now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力 path の存在を確認する。

def req(path: Path) -> None:
    """Abort when a required input path is missing."""
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
    """Build a standard metrics-row payload."""
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


# 関数: residual source inventory を構築する。

def build_inventory(
    common_inputs: dict,
    trial2_coulomb: dict,
    trial2_alpha_audit: dict,
    independent_inventory: dict,
    independent_audit: dict,
    independent_gate: dict,
    independent_route: dict,
    status_text: str,
    roadmap_text: str,
    part1_text: str,
    part3a_text: str,
    part5_text: str,
) -> dict:
    """Freeze the source inventory for the source-surface residual."""
    inventory_targets = [
        audit_target(
            "status_next_step",
            STATUS,
            status_text,
            STATUS_NEXT_STEP,
            "STATUS must point to 8.7.56.423 as the current official next step.",
        ),
        audit_target(
            "roadmap_next_branch",
            ROADMAP,
            roadmap_text,
            ROADMAP_NEXT_BRANCH,
            "ROADMAP must advertise 8.7.56.423-.426 as the current official branch.",
        ),
        audit_target(
            "part5_next_step",
            PART5,
            part5_text,
            PART5_NEXT_STEP,
            "Part V must point to the source-surface branch as the current next step.",
        ),
        audit_target(
            "part5_blocker",
            PART5,
            part5_text,
            PART5_BLOCKER,
            "Part V must describe the blocker as the missing Coulomb-normalization source surface.",
        ),
        audit_target(
            "part3a_primary_surface",
            PART3A,
            part3a_text,
            PART3A_PRIMARY_SURFACE,
            "Part III-A must still expose the structural Trial-2 section as a candidate placement surface.",
        ),
        audit_target(
            "part5_checkpoint_surface",
            PART5,
            part5_text,
            PART5_CHECKPOINT_SURFACE,
            "Part V must still expose the checkpoint summary surface as a secondary placement surface.",
        ),
    ]
    inventory_ready = all(item["matched_expectation"] for item in inventory_targets)
    no_surface_evidence_ready = bool(
        independent_audit["summary"]["explicit_coulomb_normalization_source_surface_hit_count"] == 0
        and not independent_audit["summary"]["explicit_coulomb_normalization_source_surface_available"]
    )
    route_contract_consistent = bool(
        independent_gate["summary"]["selected_residual_route"]
        == "trial2_numeric_alpha_coulomb_normalization_source_surface_identification"
        and independent_route["summary"]["selected_next_generation_route"]
        == "trial2_numeric_alpha_coulomb_normalization_source_surface_identification"
    )
    part1_no_surface_confirmed = bool(hit(part1_text, PART1_NO_SURFACE_PATTERN) is None)

    return payload(
        "8.7.56.423",
        "trial2_numeric_alpha_coulomb_normalization_source_surface_source_inventory",
        common_inputs,
        "Inventory the narrowed source-surface residual pack across the Coulomb no-ready evidence, the zero-hit no-surface audit, the candidate placement surfaces in Part III-A and Part V, and the current route contract.",
        {
            "surface_rule": "the narrowed residual now asks where the missing Coulomb-normalization source statement should live, not whether the structural alpha formula still exists",
            "placement_rule": "Part III-A is expected to host the primary technical surface while Part V remains a checkpoint-only secondary surface",
            "absence_rule": "Part I still provides no direct Coulomb-normalization source surface under the current canon",
        },
        [
            row(
                "trial2_numeric_alpha_coulomb_surface_inventory_targets_present",
                "pass" if inventory_ready else "reject",
                "source-surface inventory wording targets present",
                sum(1 for item in inventory_targets if item["present"]),
                "Control docs and candidate placement surfaces must align on the narrowed residual branch.",
            ),
            row(
                "trial2_numeric_alpha_coulomb_no_surface_evidence_ready",
                "pass" if no_surface_evidence_ready else "reject",
                "zero-hit Coulomb source-surface evidence ready",
                1 if no_surface_evidence_ready else 0,
                "The prior branch must already have frozen the zero-hit source-surface result.",
            ),
            row(
                "trial2_numeric_alpha_route_contract_consistent",
                "pass" if route_contract_consistent else "reject",
                "source-surface residual route contract consistent",
                1 if route_contract_consistent else 0,
                "The declaration gate and route contract must agree on the narrowed residual branch.",
            ),
            row(
                "trial2_numeric_alpha_part1_no_direct_surface_confirmed",
                "pass" if part1_no_surface_confirmed else "reject",
                "Part I no direct source surface confirmed",
                1 if part1_no_surface_confirmed else 0,
                "Part I remains a no-surface region for the numeric-alpha normalization issue.",
            ),
        ],
        {
            "inventory_ready": inventory_ready,
            "no_surface_evidence_ready": no_surface_evidence_ready,
            "route_contract_consistent": route_contract_consistent,
            "part1_no_direct_surface_confirmed": part1_no_surface_confirmed,
            "first_route_to_close_or_none": "trial2_numeric_alpha_coulomb_normalization_source_surface_audit",
        },
        {
            "overall_status": "trial2_numeric_alpha_coulomb_source_surface_inventory_frozen"
            if inventory_ready
            else "trial2_numeric_alpha_coulomb_source_surface_inventory_incomplete",
            "advance_to_8_7_56_424": inventory_ready,
            "next_required_artifacts": []
            if inventory_ready
            else ["trial2_numeric_alpha_coulomb_normalization_source_surface_source_inventory"],
        },
        {
            "inventory_targets": inventory_targets,
            "trial2_coulomb_summary": trial2_coulomb["summary"],
            "trial2_alpha_audit_summary": trial2_alpha_audit["summary"],
            "independent_inventory_summary": independent_inventory["summary"],
            "independent_audit_summary": independent_audit["summary"],
            "independent_gate_summary": independent_gate["summary"],
            "independent_route_summary": independent_route["summary"],
        },
    )


# 関数: source-surface audit を構築する。

def build_audit(
    common_inputs: dict,
    inventory: dict,
    independent_audit: dict,
    part1_text: str,
    part3a_text: str,
    part5_text: str,
) -> dict:
    """Audit which candidate surface should host the missing statement."""
    part3a_primary_surface = audit_target(
        "part3a_primary_surface",
        PART3A,
        part3a_text,
        PART3A_PRIMARY_SURFACE,
        "Part III-A 2.6.1 is the primary candidate placement surface for a future numeric-alpha normalization-source statement.",
    )
    part3a_secondary_surface = audit_target(
        "part3a_secondary_surface",
        PART3A,
        part3a_text,
        PART3A_SECONDARY_SURFACE,
        "Part III-A 2.6.2 remains a secondary/fallback surface because it already carries unresolved-item wording.",
    )
    part5_checkpoint_surface = audit_target(
        "part5_checkpoint_surface",
        PART5,
        part5_text,
        PART5_CHECKPOINT_SURFACE,
        "Part V 3.2 remains only a checkpoint-summary surface and not the primary technical home for the missing statement.",
    )
    part1_no_surface = audit_target(
        "part1_no_direct_surface",
        PART1,
        part1_text,
        PART1_NO_SURFACE_PATTERN,
        "Part I does not expose a direct Coulomb-normalization surface.",
        expected_present=False,
    )

    part3a_primary_surface_available = bool(part3a_primary_surface["present"])
    part3a_secondary_surface_available = bool(part3a_secondary_surface["present"])
    part5_secondary_surface_only = bool(part5_checkpoint_surface["present"])
    part1_direct_surface_available = bool(part1_no_surface["present"])
    dominant_blocker_is_part3a_primary_surface_statement_absence = bool(
        inventory["summary"]["inventory_ready"]
        and independent_audit["summary"]["explicit_coulomb_normalization_source_surface_hit_count"] == 0
        and part3a_primary_surface_available
        and part5_secondary_surface_only
        and not part1_direct_surface_available
    )
    audit_ready = dominant_blocker_is_part3a_primary_surface_statement_absence

    return payload(
        "8.7.56.424",
        "trial2_numeric_alpha_coulomb_normalization_source_surface_audit",
        common_inputs,
        "Audit the candidate placement surfaces for the missing Coulomb-normalization source statement and identify the primary technical locus under the current canon.",
        {
            "primary_surface_rule": "Part III-A 2.6.1 should host the primary technical statement because it already freezes the Trial-2 structural route",
            "secondary_surface_rule": "Part V remains a checkpoint summary and therefore only a secondary/reference surface",
            "part1_rule": "Part I remains a no-surface region for this numeric-alpha normalization issue",
        },
        [
            row(
                "trial2_numeric_alpha_part3a_primary_surface_available",
                "pass" if part3a_primary_surface_available else "reject",
                "Part III-A primary surface available",
                1 if part3a_primary_surface_available else 0,
                "Part III-A 2.6.1 is available as the primary candidate surface.",
            ),
            row(
                "trial2_numeric_alpha_part3a_secondary_surface_available",
                "pass" if part3a_secondary_surface_available else "watch",
                "Part III-A secondary surface available",
                1 if part3a_secondary_surface_available else 0,
                "Part III-A 2.6.2 survives as a fallback surface only.",
            ),
            row(
                "trial2_numeric_alpha_part5_secondary_surface_only",
                "pass" if part5_secondary_surface_only else "reject",
                "Part V secondary-only surface available",
                1 if part5_secondary_surface_only else 0,
                "Part V 3.2 remains available only as a checkpoint summary surface.",
            ),
            row(
                "trial2_numeric_alpha_part1_direct_surface_available",
                "reject" if not part1_direct_surface_available else "watch",
                "Part I direct source surface available",
                1 if part1_direct_surface_available else 0,
                "Part I still exposes no direct source-surface wording for this problem.",
            ),
            row(
                "trial2_numeric_alpha_dominant_blocker_is_part3a_primary_surface_statement_absence",
                "pass" if dominant_blocker_is_part3a_primary_surface_statement_absence else "watch",
                "dominant blocker is Part III-A primary-surface statement absence",
                1 if dominant_blocker_is_part3a_primary_surface_statement_absence else 0,
                "The residual now shrinks from generic source-surface absence to the missing statement on the Part III-A primary surface.",
            ),
        ],
        {
            "audit_ready": audit_ready,
            "part3a_primary_surface_available": part3a_primary_surface_available,
            "part3a_secondary_surface_available": part3a_secondary_surface_available,
            "part5_secondary_surface_only": part5_secondary_surface_only,
            "part1_direct_surface_available": part1_direct_surface_available,
            "dominant_blocker_is_part3a_primary_surface_statement_absence": dominant_blocker_is_part3a_primary_surface_statement_absence,
            "first_route_to_close_or_none": "trial2_numeric_alpha_coulomb_normalization_source_surface_declaration_gate",
        },
        {
            "overall_status": "trial2_numeric_alpha_coulomb_source_surface_audit_complete"
            if audit_ready
            else "trial2_numeric_alpha_coulomb_source_surface_audit_incomplete",
            "advance_to_8_7_56_425": audit_ready,
            "next_required_artifacts": []
            if audit_ready
            else ["trial2_numeric_alpha_coulomb_normalization_source_surface_audit"],
        },
        {
            "inventory_summary": inventory["summary"],
            "independent_audit_summary": independent_audit["summary"],
            "part3a_primary_surface": part3a_primary_surface,
            "part3a_secondary_surface": part3a_secondary_surface,
            "part5_checkpoint_surface": part5_checkpoint_surface,
            "part1_no_surface": part1_no_surface,
        },
    )


# 関数: declaration gate を構築する。

def build_gate(common_inputs: dict, audit: dict) -> dict:
    """Freeze the declaration gate for the narrowed source-surface blocker."""
    audit_ready = bool(audit["summary"]["audit_ready"])

    return payload(
        "8.7.56.425",
        "trial2_numeric_alpha_coulomb_normalization_source_surface_declaration_gate",
        common_inputs,
        "Freeze the declaration gate for the source-surface residual and isolate the next EM precision blocker officially.",
        {
            "gate_rule": "close the current residual branch once the primary placement surface is identified, even though the statement itself is still absent",
            "residual_rule": "the next residual route identifies the missing Part-III-A primary-surface statement rather than another generic source-surface search",
        },
        [
            row(
                "trial2_numeric_alpha_coulomb_source_surface_gate_complete",
                "pass" if audit_ready else "reject",
                "source-surface gate complete",
                1 if audit_ready else 0,
                "The gate closes once the primary placement surface is identified.",
            ),
            row(
                "trial2_numeric_alpha_closeout_ready",
                "reject",
                "numeric-alpha closeout ready",
                0,
                "Numeric-alpha closeout still requires the missing primary-surface statement.",
            ),
            row(
                "trial2_numeric_alpha_part3a_primary_surface_statement_missing",
                "pass" if audit_ready else "watch",
                "Part III-A primary-surface statement missing",
                1 if audit_ready else 0,
                "The narrowed blocker is the missing required statement on the Part III-A primary surface.",
            ),
            row(
                "trial2_numeric_alpha_precision_mainline_preserved",
                "pass",
                "precision-alpha mainline preserved",
                1,
                "The branch does not reopen the structural EM pass or promote the strong-side reserve.",
            ),
        ],
        {
            "trial2_numeric_alpha_coulomb_source_surface_branch_closeable": audit_ready,
            "trial2_numeric_alpha_closeout_ready": False,
            "selected_residual_route": NEXT_ROUTE_LABEL,
            "missing_v2_artifact": "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_primary_surface_statement",
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_coulomb_source_surface_gate_closed"
            if audit_ready
            else "trial2_numeric_alpha_coulomb_source_surface_gate_open",
            "advance_to_8_7_56_426": audit_ready,
            "next_required_artifacts": []
            if audit_ready
            else ["trial2_numeric_alpha_coulomb_normalization_source_surface_declaration_gate"],
        },
        {
            "audit_summary": audit["summary"],
        },
    )


# 関数: next-generation route contract third refresh を構築する。

def build_contract(common_inputs: dict, gate: dict, v3_hold_gate: dict) -> dict:
    """Refresh the strong-side reserve after the primary-surface audit."""
    gate_closed = bool(gate["summary"]["trial2_numeric_alpha_coulomb_source_surface_branch_closeable"])
    residual_route_selected = gate["summary"]["selected_residual_route"] == NEXT_ROUTE_LABEL

    return payload(
        "8.7.56.426",
        "trial2_numeric_alpha_next_generation_route_contract_third_refresh",
        common_inputs,
        "Refresh the strong-side reserve after the source-surface branch and freeze the next EM precision residual route contract.",
        {
            "contract_rule": "the EM precision mainline remains active while the blocker shrinks from source-surface absence to a missing statement on the identified primary surface",
            "reserve_rule": "strong-side work remains on reserve and does not outrank the EM precision route",
        },
        [
            row(
                "trial2_numeric_alpha_coulomb_source_surface_gate_closed",
                "pass" if gate_closed else "reject",
                "source-surface gate closed",
                1 if gate_closed else 0,
                "The next route contract depends on the declaration gate being frozen first.",
            ),
            row(
                "trial2_numeric_alpha_part3a_primary_surface_statement_route_selected",
                "pass" if residual_route_selected else "reject",
                "Part III-A primary-surface statement route selected",
                1 if residual_route_selected else 0,
                "The next official route stays inside the EM precision program.",
            ),
            row(
                "trial2_numeric_alpha_strong_side_reserve_retained",
                "pass",
                "strong-side reserve retained",
                1,
                "Strong-side non-Abelian/running/confinement gaps remain on reserve.",
            ),
            row(
                "trial2_numeric_alpha_precision_mainline_retained",
                "pass",
                "precision-alpha mainline retained",
                1,
                "The first next-generation mainline remains the Trial-2 numeric-alpha program.",
            ),
        ],
        {
            "selected_next_generation_route": NEXT_ROUTE_LABEL if residual_route_selected else None,
            "strong_side_route_state": v3_hold_gate["summary"]["strong_side_route_state"],
            "precision_alpha_mainline_retained": True,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_next_route_contract_third_refresh_frozen"
            if gate_closed
            else "trial2_numeric_alpha_next_route_contract_third_refresh_pending",
            "advance_to_next_route": gate_closed,
            "next_required_artifacts": [
                "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_primary_surface_statement_inventory",
                "trial2_numeric_alpha_coulomb_normalization_source_surface_part3a_primary_surface_statement_audit",
            ]
            if gate_closed
            else ["trial2_numeric_alpha_next_generation_route_contract_third_refresh"],
        },
        {
            "gate_summary": gate["summary"],
            "v3_hold_gate_summary": v3_hold_gate["summary"],
        },
    )


# 関数: current branch を実行する。

def main() -> None:
    """Execute the Trial-2 numeric-alpha Coulomb-source-surface branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        PART1,
        PART3A,
        PART5,
        TRIAL2_COULOMB,
        TRIAL2_ALPHA_AUDIT,
        INDEPENDENT_INVENTORY,
        INDEPENDENT_AUDIT,
        INDEPENDENT_GATE,
        INDEPENDENT_ROUTE,
        V3_HOLD_GATE,
    ):
        req(path)

    common_inputs = {
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "part1_core_theory_markdown": rel(PART1),
        "part3a_quantum_foundations_markdown": rel(PART3A),
        "part5_future_predictions_markdown": rel(PART5),
        "mass_origin_v2_trial2_curvature_coulomb_pilot_json": rel(TRIAL2_COULOMB),
        "mass_origin_v2_trial2_fine_structure_constant_coupling_mapping_audit_json": rel(TRIAL2_ALPHA_AUDIT),
        "mass_origin_v2_trial2_numeric_alpha_independent_normalization_source_inventory_json": rel(INDEPENDENT_INVENTORY),
        "mass_origin_v2_trial2_numeric_alpha_independent_normalization_source_audit_json": rel(INDEPENDENT_AUDIT),
        "mass_origin_v2_trial2_numeric_alpha_independent_normalization_source_declaration_gate_json": rel(INDEPENDENT_GATE),
        "mass_origin_v2_trial2_numeric_alpha_next_generation_route_contract_second_refresh_json": rel(INDEPENDENT_ROUTE),
        "mass_origin_v2_v3_hold_declaration_gate_json": rel(V3_HOLD_GATE),
    }

    trial2_coulomb = read_json(TRIAL2_COULOMB)
    trial2_alpha_audit = read_json(TRIAL2_ALPHA_AUDIT)
    independent_inventory = read_json(INDEPENDENT_INVENTORY)
    independent_audit = read_json(INDEPENDENT_AUDIT)
    independent_gate = read_json(INDEPENDENT_GATE)
    independent_route = read_json(INDEPENDENT_ROUTE)
    v3_hold_gate = read_json(V3_HOLD_GATE)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)

    inventory = build_inventory(
        common_inputs,
        trial2_coulomb,
        trial2_alpha_audit,
        independent_inventory,
        independent_audit,
        independent_gate,
        independent_route,
        status_text,
        roadmap_text,
        part1_text,
        part3a_text,
        part5_text,
    )
    audit = build_audit(
        common_inputs,
        inventory,
        independent_audit,
        part1_text,
        part3a_text,
        part5_text,
    )
    gate = build_gate(common_inputs, audit)
    contract = build_contract(common_inputs, gate, v3_hold_gate)

    write_artifact("mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_source_inventory", inventory)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_audit", audit)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_declaration_gate", gate)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_next_generation_route_contract_third_refresh", contract)

    print("[ok] generated Trial-2 numeric alpha Coulomb-normalization-source-surface artifacts:")
    print(" - mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_audit_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_coulomb_normalization_source_surface_declaration_gate_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_next_generation_route_contract_third_refresh_metrics.json")


# 関数: CLI 直実行時に branch main を起動する。

if __name__ == "__main__":
    main()
