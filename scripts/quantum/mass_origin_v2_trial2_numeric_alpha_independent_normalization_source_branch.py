#!/usr/bin/env python3
"""Generate 8.7.56.419-.422 Trial-2 numeric-alpha normalization-source artifacts.

This branch follows the precision carry-over route. The symbolic Trial-2
electromagnetic formulas and the QED precision target are already frozen, so
the current residual problem is locating an independent numeric normalization
source under the current canon.

The branch:

1. inventories the surviving structural/precision pack,
2. audits whether any independent normalization source surface exists,
3. freezes the declaration gate for the narrowed blocker, and
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
PART1 = ROOT / "doc" / "paper" / "10_part1_core_theory.md"
PART3A = ROOT / "doc" / "paper" / "12_part3a_quantum_foundations.md"
PART5 = ROOT / "doc" / "paper" / "14_part5_future_predictions.md"

TRIAL2_COULOMB = OUT / "mass_origin_v2_trial2_curvature_coulomb_pilot_metrics.json"
TRIAL2_ALPHA_AUDIT = OUT / "mass_origin_v2_trial2_fine_structure_constant_coupling_mapping_audit_metrics.json"
PRECISION_INVENTORY = OUT / "mass_origin_v2_trial2_numeric_alpha_precision_carryover_source_inventory_metrics.json"
PRECISION_AUDIT = OUT / "mass_origin_v2_trial2_numeric_alpha_precision_carryover_audit_metrics.json"
PRECISION_GATE = OUT / "mass_origin_v2_trial2_numeric_alpha_precision_carryover_declaration_gate_metrics.json"
PRECISION_ROUTE = OUT / "mass_origin_v2_trial2_numeric_alpha_next_generation_route_contract_metrics.json"
V3_HOLD_GATE = OUT / "mass_origin_v2_v3_hold_declaration_gate_metrics.json"
QED_PRECISION = OUT / "qed_vacuum_precision_metrics.json"

NEXT_ROUTE = "8.7.56.423"
NEXT_ROUTE_LABEL = "trial2_numeric_alpha_coulomb_normalization_source_surface_identification"

STATUS_NEXT_STEP = "current official next step は `8.7.56.419`"
ROADMAP_NEXT_BRANCH = "8.7.56.419-.422"
PART3A_FORMULA = "$\\alpha=g_P^2/(4\\pi Z_P\\hbar c)$"
PART3A_STATE = "foundational / structural pass (numeric α open)"
PART5_NEXT_STEP = "8.7.56.419-.422"
PART5_BLOCKER = "independent normalization source の欠落"
EPSILON0_PATTERN = "epsilon_0"
ALPHA_TARGET_PATTERN = "137.035999084"
BOHR_TARGET_PATTERN = "27.21138624593"


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


# 関数: 複数ファイルにまたがる surface hit 数を数える。

def surface_hits(targets: list[dict]) -> int:
    """Count how many audit targets are present."""
    return sum(1 for item in targets if item["present"])


# 関数: residual source inventory を構築する。

def build_inventory(
    common_inputs: dict,
    trial2_coulomb: dict,
    trial2_alpha_audit: dict,
    precision_inventory: dict,
    precision_audit: dict,
    precision_gate: dict,
    precision_route: dict,
    qed_precision: dict,
    status_text: str,
    roadmap_text: str,
    part3a_text: str,
    part5_text: str,
) -> dict:
    """Freeze the source inventory for the independent-normalization-source route."""
    inventory_targets = [
        audit_target(
            "status_next_step",
            STATUS,
            status_text,
            STATUS_NEXT_STEP,
            "STATUS must point to 8.7.56.419 as the current official next step.",
        ),
        audit_target(
            "roadmap_next_branch",
            ROADMAP,
            roadmap_text,
            ROADMAP_NEXT_BRANCH,
            "ROADMAP must advertise 8.7.56.419-.422 as the current official branch.",
        ),
        audit_target(
            "part3a_alpha_formula",
            PART3A,
            part3a_text,
            PART3A_FORMULA,
            "Part III-A must keep the structural alpha formula.",
        ),
        audit_target(
            "part3a_numeric_open_state",
            PART3A,
            part3a_text,
            PART3A_STATE,
            "Part III-A must still classify Trial-2 as structural pass with numeric alpha open.",
        ),
        audit_target(
            "part5_next_step",
            PART5,
            part5_text,
            PART5_NEXT_STEP,
            "Part V must point to the independent-normalization-source branch as the current next step.",
        ),
        audit_target(
            "part5_blocker",
            PART5,
            part5_text,
            PART5_BLOCKER,
            "Part V must describe the blocker as the missing independent normalization source.",
        ),
    ]
    inventory_ready = all(item["matched_expectation"] for item in inventory_targets)
    structural_formula_pack_ready = bool(
        trial2_alpha_audit["summary"]["electric_charge_formula_ready"]
        and trial2_alpha_audit["summary"]["alpha_formula_ready"]
    )
    precision_target_pack_available = bool(
        trial2_alpha_audit["summary"]["alpha_target_inverse_value"]
        == qed_precision["sources"][4]["extracted_value"]["alpha_inv"]
    )
    current_numeric_normalization_gap_open = bool(
        not trial2_alpha_audit["summary"]["alpha_numeric_from_current_pack_ready"]
        and not trial2_coulomb["summary"]["coulomb_normalization_numeric_ready"]
    )
    route_contract_consistent = bool(
        precision_gate["summary"]["selected_residual_route"]
        == "trial2_numeric_alpha_independent_normalization_source_identification"
        and precision_route["summary"]["selected_next_generation_route"]
        == "trial2_numeric_alpha_independent_normalization_source_identification"
    )

    return payload(
        "8.7.56.419",
        "trial2_numeric_alpha_independent_normalization_source_source_inventory",
        common_inputs,
        "Inventory the independent-normalization-source residual pack across the structural alpha formula, the precision target, the current Coulomb no-ready evidence, and the route contract that selected this blocker as the current mainline.",
        {
            "formula_rule": "retain e = g_P / sqrt(Z_P) and alpha = g_P^2 / (4 pi Z_P hbar c) as already-frozen structural formulas",
            "precision_rule": "reuse the cached QED alpha target pack as the required numeric benchmark",
            "residual_rule": "the current route isolates the missing independent normalization source rather than reopening the structural EM pass",
        },
        [
            row(
                "trial2_numeric_alpha_independent_normalization_inventory_targets_present",
                "pass" if inventory_ready else "reject",
                "independent-normalization-source wording targets present",
                sum(1 for item in inventory_targets if item["present"]),
                "Control docs and Part V must all advertise the current residual branch consistently.",
            ),
            row(
                "trial2_numeric_alpha_structural_formula_pack_ready",
                "pass" if structural_formula_pack_ready else "reject",
                "structural formula pack ready",
                1 if structural_formula_pack_ready else 0,
                "The current route still starts from an already-passed structural formula pack.",
            ),
            row(
                "trial2_numeric_alpha_precision_target_pack_available",
                "pass" if precision_target_pack_available else "reject",
                "precision target pack available",
                1 if precision_target_pack_available else 0,
                "The QED precision alpha target remains available as the numeric benchmark.",
            ),
            row(
                "trial2_numeric_alpha_current_numeric_normalization_gap_open",
                "pass" if current_numeric_normalization_gap_open else "watch",
                "current numeric normalization gap open",
                1 if current_numeric_normalization_gap_open else 0,
                "Both alpha-from-current-pack and Coulomb normalization remain numerically open.",
            ),
            row(
                "trial2_numeric_alpha_route_contract_consistent",
                "pass" if route_contract_consistent else "reject",
                "residual route contract consistent",
                1 if route_contract_consistent else 0,
                "The declaration gate and the route contract must agree on the current residual branch.",
            ),
        ],
        {
            "inventory_ready": inventory_ready,
            "structural_formula_pack_ready": structural_formula_pack_ready,
            "precision_target_pack_available": precision_target_pack_available,
            "current_numeric_normalization_gap_open": current_numeric_normalization_gap_open,
            "coulomb_normalization_numeric_ready": trial2_coulomb["summary"]["coulomb_normalization_numeric_ready"],
            "residual_route_contract_consistent": route_contract_consistent,
            "first_route_to_close_or_none": "trial2_numeric_alpha_independent_normalization_source_audit",
        },
        {
            "overall_status": "trial2_numeric_alpha_independent_normalization_inventory_frozen"
            if inventory_ready
            else "trial2_numeric_alpha_independent_normalization_inventory_incomplete",
            "advance_to_8_7_56_420": inventory_ready,
            "next_required_artifacts": []
            if inventory_ready
            else ["trial2_numeric_alpha_independent_normalization_source_source_inventory"],
        },
        {
            "inventory_targets": inventory_targets,
            "trial2_coulomb_summary": trial2_coulomb["summary"],
            "trial2_alpha_audit_summary": trial2_alpha_audit["summary"],
            "precision_inventory_summary": precision_inventory["summary"],
            "precision_audit_summary": precision_audit["summary"],
            "precision_gate_summary": precision_gate["summary"],
            "precision_route_summary": precision_route["summary"],
            "qed_precision_alpha_target": qed_precision["sources"][4]["extracted_value"],
        },
    )


# 関数: residual audit を構築する。

def build_audit(
    common_inputs: dict,
    inventory: dict,
    trial2_coulomb: dict,
    trial2_alpha_audit: dict,
    precision_route: dict,
    part1_text: str,
    part3a_text: str,
    part5_text: str,
) -> dict:
    """Audit whether any independent normalization source surface exists."""
    structural_formula_ready = bool(inventory["summary"]["structural_formula_pack_ready"])
    precision_target_pack_available = bool(inventory["summary"]["precision_target_pack_available"])
    current_numeric_prediction_available = bool(trial2_alpha_audit["summary"]["alpha_numeric_from_current_pack_ready"])
    coulomb_normalization_numeric_ready = bool(trial2_coulomb["summary"]["coulomb_normalization_numeric_ready"])

    source_surface_targets = [
        audit_target(
            "part1_epsilon0_surface",
            PART1,
            part1_text,
            EPSILON0_PATTERN,
            "Part I would expose an explicit epsilon_0-like normalization surface if the current canon already had one.",
            expected_present=False,
        ),
        audit_target(
            "part3a_epsilon0_surface",
            PART3A,
            part3a_text,
            EPSILON0_PATTERN,
            "Part III-A would expose an explicit epsilon_0-like normalization surface if the current canon already had one.",
            expected_present=False,
        ),
        audit_target(
            "part5_alpha_target_surface",
            PART5,
            part5_text,
            ALPHA_TARGET_PATTERN,
            "Part V would cite the precision alpha value directly if the current canon already carried the numeric normalization source.",
            expected_present=False,
        ),
        audit_target(
            "part3a_bohr_target_surface",
            PART3A,
            part3a_text,
            BOHR_TARGET_PATTERN,
            "Part III-A would carry the Bohr-scale Coulomb target directly if the current canon already exposed a numeric Coulomb normalization source.",
            expected_present=False,
        ),
    ]
    source_surface_hit_count = surface_hits(source_surface_targets)
    explicit_coulomb_normalization_source_surface_available = bool(source_surface_hit_count > 0)
    independent_normalization_source_available = bool(
        current_numeric_prediction_available
        or coulomb_normalization_numeric_ready
        or explicit_coulomb_normalization_source_surface_available
    )
    dominant_blocker_is_coulomb_normalization_source_surface = bool(
        structural_formula_ready
        and precision_target_pack_available
        and not independent_normalization_source_available
    )
    precision_route_still_selected = bool(
        precision_route["summary"]["selected_next_generation_route"]
        == "trial2_numeric_alpha_independent_normalization_source_identification"
    )
    audit_ready = bool(
        inventory["summary"]["inventory_ready"]
        and dominant_blocker_is_coulomb_normalization_source_surface
        and precision_route_still_selected
    )

    return payload(
        "8.7.56.420",
        "trial2_numeric_alpha_independent_normalization_source_audit",
        common_inputs,
        "Audit whether the independent-normalization-source blocker now shrinks to the absence of an explicit Coulomb/absolute-charge normalization source surface under the current canon.",
        {
            "surface_rule": "an honest numeric alpha closeout needs either a numeric Coulomb normalization closure or an explicit independent source surface under the current canon",
            "absence_rule": "if both the numeric closure and the source-surface hits remain absent, the blocker shrinks to the missing Coulomb-normalization-source surface",
            "reserve_rule": "strong-side work stays on reserve while the EM precision route continues",
        },
        [
            row(
                "trial2_numeric_alpha_current_numeric_prediction_available",
                "pass" if current_numeric_prediction_available else "reject",
                "current numeric alpha prediction available",
                1 if current_numeric_prediction_available else 0,
                "The structural pack still does not produce a numeric alpha value directly.",
            ),
            row(
                "trial2_numeric_alpha_coulomb_normalization_numeric_ready",
                "pass" if coulomb_normalization_numeric_ready else "reject",
                "Coulomb normalization numeric ready",
                1 if coulomb_normalization_numeric_ready else 0,
                "The Coulomb normalization branch remains numerically open.",
            ),
            row(
                "trial2_numeric_alpha_coulomb_normalization_source_surface_hits",
                "reject" if source_surface_hit_count == 0 else "pass",
                "explicit Coulomb-normalization source surface hit count",
                source_surface_hit_count,
                "No explicit epsilon_0-like or numeric-target normalization surface is exposed in the current canon sources.",
            ),
            row(
                "trial2_numeric_alpha_dominant_blocker_is_coulomb_normalization_source_surface",
                "pass" if dominant_blocker_is_coulomb_normalization_source_surface else "watch",
                "dominant blocker is Coulomb-normalization source surface",
                1 if dominant_blocker_is_coulomb_normalization_source_surface else 0,
                "The residual problem shrinks from generic normalization absence to the missing source surface itself.",
            ),
            row(
                "trial2_numeric_alpha_precision_route_still_selected",
                "pass" if precision_route_still_selected else "reject",
                "precision-alpha route still selected",
                1 if precision_route_still_selected else 0,
                "The current mainline remains the EM precision route rather than the strong-side reserve.",
            ),
        ],
        {
            "audit_ready": audit_ready,
            "structural_formula_ready": structural_formula_ready,
            "precision_target_pack_available": precision_target_pack_available,
            "current_numeric_prediction_available": current_numeric_prediction_available,
            "coulomb_normalization_numeric_ready": coulomb_normalization_numeric_ready,
            "explicit_coulomb_normalization_source_surface_hit_count": source_surface_hit_count,
            "explicit_coulomb_normalization_source_surface_available": explicit_coulomb_normalization_source_surface_available,
            "independent_normalization_source_available_under_current_canon": independent_normalization_source_available,
            "dominant_blocker_is_coulomb_normalization_source_surface": dominant_blocker_is_coulomb_normalization_source_surface,
            "first_route_to_close_or_none": "trial2_numeric_alpha_independent_normalization_source_declaration_gate",
        },
        {
            "overall_status": "trial2_numeric_alpha_independent_normalization_audit_complete"
            if audit_ready
            else "trial2_numeric_alpha_independent_normalization_audit_incomplete",
            "advance_to_8_7_56_421": audit_ready,
            "next_required_artifacts": []
            if audit_ready
            else ["trial2_numeric_alpha_independent_normalization_source_audit"],
        },
        {
            "inventory_summary": inventory["summary"],
            "trial2_coulomb_summary": trial2_coulomb["summary"],
            "trial2_alpha_audit_summary": trial2_alpha_audit["summary"],
            "precision_route_summary": precision_route["summary"],
            "source_surface_targets": source_surface_targets,
        },
    )


# 関数: declaration gate を構築する。

def build_gate(common_inputs: dict, audit: dict) -> dict:
    """Freeze the declaration gate for the narrowed EM precision blocker."""
    audit_ready = bool(audit["summary"]["audit_ready"])
    closeout_ready = bool(audit["summary"]["independent_normalization_source_available_under_current_canon"])

    return payload(
        "8.7.56.421",
        "trial2_numeric_alpha_independent_normalization_source_declaration_gate",
        common_inputs,
        "Freeze the declaration gate for the independent-normalization-source residual route and isolate the next EM precision blocker officially.",
        {
            "gate_rule": "close the current residual branch once the source-surface blocker is isolated, even though numeric alpha still remains open",
            "residual_rule": "the next residual route identifies the missing Coulomb-normalization source surface rather than reopening the formula pack",
        },
        [
            row(
                "trial2_numeric_alpha_independent_normalization_gate_complete",
                "pass" if audit_ready else "reject",
                "independent-normalization-source gate complete",
                1 if audit_ready else 0,
                "The gate closes once the narrowed blocker is isolated.",
            ),
            row(
                "trial2_numeric_alpha_closeout_ready",
                "pass" if closeout_ready else "reject",
                "numeric-alpha closeout ready",
                1 if closeout_ready else 0,
                "Numeric-alpha closeout still requires an explicit normalization source surface under the current canon.",
            ),
            row(
                "trial2_numeric_alpha_coulomb_normalization_source_surface_missing",
                "pass" if not closeout_ready else "watch",
                "Coulomb-normalization source surface missing",
                1 if not closeout_ready else 0,
                "The narrowed blocker is the missing Coulomb-normalization source surface.",
            ),
            row(
                "trial2_numeric_alpha_structural_em_pass_preserved",
                "pass",
                "structural EM pass preserved",
                1,
                "The branch does not reopen the already-frozen structural EM pass.",
            ),
        ],
        {
            "trial2_numeric_alpha_independent_normalization_branch_closeable": audit_ready,
            "trial2_numeric_alpha_closeout_ready": closeout_ready,
            "selected_residual_route": NEXT_ROUTE_LABEL,
            "missing_v2_artifact": "trial2_numeric_alpha_coulomb_normalization_source_surface_pack",
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial2_numeric_alpha_independent_normalization_gate_closed"
            if audit_ready
            else "trial2_numeric_alpha_independent_normalization_gate_open",
            "advance_to_8_7_56_422": audit_ready,
            "next_required_artifacts": []
            if audit_ready
            else ["trial2_numeric_alpha_independent_normalization_source_declaration_gate"],
        },
        {
            "audit_summary": audit["summary"],
        },
    )


# 関数: next-generation route contract second refresh を構築する。

def build_contract(common_inputs: dict, gate: dict, v3_hold_gate: dict) -> dict:
    """Refresh the strong-side reserve after the narrowed EM precision residual."""
    gate_closed = bool(gate["summary"]["trial2_numeric_alpha_independent_normalization_branch_closeable"])
    residual_route_selected = gate["summary"]["selected_residual_route"] == NEXT_ROUTE_LABEL

    return payload(
        "8.7.56.422",
        "trial2_numeric_alpha_next_generation_route_contract_second_refresh",
        common_inputs,
        "Refresh the strong-side reserve after the independent-normalization-source branch and freeze the next EM precision residual route contract.",
        {
            "contract_rule": "the EM precision mainline remains active while the blocker shrinks from generic normalization absence to the missing source surface",
            "reserve_rule": "strong-side work remains on reserve and does not outrank the EM precision route",
        },
        [
            row(
                "trial2_numeric_alpha_independent_normalization_gate_closed",
                "pass" if gate_closed else "reject",
                "independent-normalization-source gate closed",
                1 if gate_closed else 0,
                "The next route contract depends on the declaration gate being frozen first.",
            ),
            row(
                "trial2_numeric_alpha_coulomb_source_surface_route_selected",
                "pass" if residual_route_selected else "reject",
                "Coulomb-normalization source-surface route selected",
                1 if residual_route_selected else 0,
                "The next official route stays inside the EM precision program.",
            ),
            row(
                "trial2_numeric_alpha_strong_side_reserve_retained",
                "pass",
                "strong-side reserve retained",
                1,
                "Strong-side non-Abelian/running/confinement gaps remain on reserve and do not outrank the EM precision route.",
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
            "overall_status": "trial2_numeric_alpha_next_route_contract_second_refresh_frozen"
            if gate_closed
            else "trial2_numeric_alpha_next_route_contract_second_refresh_pending",
            "advance_to_next_route": gate_closed,
            "next_required_artifacts": [
                "trial2_numeric_alpha_coulomb_normalization_source_surface_inventory",
                "trial2_numeric_alpha_coulomb_normalization_source_surface_audit",
            ]
            if gate_closed
            else ["trial2_numeric_alpha_next_generation_route_contract_second_refresh"],
        },
        {
            "gate_summary": gate["summary"],
            "v3_hold_gate_summary": v3_hold_gate["summary"],
        },
    )


# 関数: current branch を実行する。

def main() -> None:
    """Execute the Trial-2 numeric-alpha independent-normalization-source branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        PART1,
        PART3A,
        PART5,
        TRIAL2_COULOMB,
        TRIAL2_ALPHA_AUDIT,
        PRECISION_INVENTORY,
        PRECISION_AUDIT,
        PRECISION_GATE,
        PRECISION_ROUTE,
        V3_HOLD_GATE,
        QED_PRECISION,
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
        "mass_origin_v2_trial2_numeric_alpha_precision_carryover_source_inventory_json": rel(PRECISION_INVENTORY),
        "mass_origin_v2_trial2_numeric_alpha_precision_carryover_audit_json": rel(PRECISION_AUDIT),
        "mass_origin_v2_trial2_numeric_alpha_precision_carryover_declaration_gate_json": rel(PRECISION_GATE),
        "mass_origin_v2_trial2_numeric_alpha_next_generation_route_contract_json": rel(PRECISION_ROUTE),
        "mass_origin_v2_v3_hold_declaration_gate_json": rel(V3_HOLD_GATE),
        "qed_vacuum_precision_metrics_json": rel(QED_PRECISION),
    }

    trial2_coulomb = read_json(TRIAL2_COULOMB)
    trial2_alpha_audit = read_json(TRIAL2_ALPHA_AUDIT)
    precision_inventory = read_json(PRECISION_INVENTORY)
    precision_audit = read_json(PRECISION_AUDIT)
    precision_gate = read_json(PRECISION_GATE)
    precision_route = read_json(PRECISION_ROUTE)
    v3_hold_gate = read_json(V3_HOLD_GATE)
    qed_precision = read_json(QED_PRECISION)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    part1_text = read_text(PART1)
    part3a_text = read_text(PART3A)
    part5_text = read_text(PART5)

    inventory = build_inventory(
        common_inputs,
        trial2_coulomb,
        trial2_alpha_audit,
        precision_inventory,
        precision_audit,
        precision_gate,
        precision_route,
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
        precision_route,
        part1_text,
        part3a_text,
        part5_text,
    )
    gate = build_gate(common_inputs, audit)
    contract = build_contract(common_inputs, gate, v3_hold_gate)

    write_artifact("mass_origin_v2_trial2_numeric_alpha_independent_normalization_source_inventory", inventory)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_independent_normalization_source_audit", audit)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_independent_normalization_source_declaration_gate", gate)
    write_artifact("mass_origin_v2_trial2_numeric_alpha_next_generation_route_contract_second_refresh", contract)

    print("[ok] generated Trial-2 numeric alpha independent-normalization-source artifacts:")
    print(" - mass_origin_v2_trial2_numeric_alpha_independent_normalization_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_independent_normalization_source_audit_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_independent_normalization_source_declaration_gate_metrics.json")
    print(" - mass_origin_v2_trial2_numeric_alpha_next_generation_route_contract_second_refresh_metrics.json")


# 関数: CLI 直実行時に branch main を起動する。

if __name__ == "__main__":
    main()
