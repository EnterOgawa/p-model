#!/usr/bin/env python3
"""
Generate relaunched Trial-3 explicit k-positive interpolation-signature artifacts.

This branch executes roadmap steps 8.7.56.241-.244.

The previous residual showed that the primary blocker is no longer the whole
node-resolved interpolation builder but the missing explicit k-positive
signature itself. This branch freezes the signature-level source pack and
formalizes that the current callers still pass only `ell`, so the next honest
route is the missing callsite-side k-axis propagation artifact.
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

PRIOR_SOURCE = (
    OUT
    / "mass_origin_v2_trial3_relaunched_explicit_k_positive_node_resolved_interpolation_builder_source_inventory_metrics.json"
)
PRIOR_AUDIT = (
    OUT
    / "mass_origin_v2_trial3_relaunched_explicit_k_positive_node_resolved_interpolation_builder_identification_audit_metrics.json"
)
PRIOR_DECLARATION = OUT / "mass_origin_v2_trial3_relaunched_declaration_fourth_gate_metrics.json"
PRIOR_DISPOSITION = (
    OUT / "mass_origin_v2_trial3_relaunched_paper_sync_trial4_disposition_fourth_refresh_metrics.json"
)
NUMERICAL_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_numerical_solver_branch.py"
FULL_COUPLED_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_full_coupled_solver_branch.py"

NEXT_ROUTE = "8.7.56.245"
RESIDUAL_ROUTE = "trial3_relaunched_explicit_k_positive_interpolation_callsite_dependency_identification"
MISSING_ARTIFACT = "trial3_relaunched_explicit_k_positive_interpolation_callsite_k_axis_propagation"


# 関数: 現在UTC時刻を ISO 8601 文字列で返す。
def now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力が欠けていれば即時停止する。

def req(path: Path) -> None:
    """Abort when a required input path is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: UTF-8 JSON を読み込む。

def read_json(path: Path) -> dict:
    """Read a UTF-8 JSON artifact."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: UTF-8 テキストを読み込む。

def read_text(path: Path) -> str:
    """Read a UTF-8 text source."""
    return path.read_text(encoding="utf-8")


# 関数: 絶対パスをリポジトリ相対表記へ変換する。

def rel(path: Path) -> str:
    """Convert an absolute path into a repo-relative string."""
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: パターンを含む最初の行を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first source line that contains the requested pattern."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: 標準 row オブジェクトを構築する。

def row(row_id: str, status: str, metric: str, value: float, note: str) -> dict:
    """Build a standard result row."""
    return {
        "row_id": row_id,
        "status": status,
        "metric": metric,
        "value": float(value),
        "note": note,
    }


# 関数: 標準 payload オブジェクトを構築する。

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
    """Build a standard payload object."""
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


# 関数: JSON と rows CSV を同時に保存する。

def write_artifact(stem: str, data: dict) -> None:
    """Save a JSON artifact and its row CSV."""
    OUT.mkdir(parents=True, exist_ok=True)
    json_path = OUT / f"{stem}_metrics.json"
    csv_path = OUT / f"{stem}_rows.csv"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row_id", "status", "metric", "value", "note"])
        writer.writeheader()
        writer.writerows(data["rows"])


# 関数: source inventory 用の target record を返す。

def target_record(file_key: str, path: Path, text: str, pattern: str, note: str) -> dict:
    """Build a source-inventory target record."""
    target_hit = hit(text, pattern)
    return {
        "file_key": file_key,
        "file": rel(path),
        "pattern": pattern,
        "present": target_hit is not None,
        "note": note,
        "evidence": target_hit,
    }


# 関数: `.241-.244` branch を実行して signature residual を固定する。

def main() -> None:
    """Execute the relaunched Trial-3 interpolation-signature residual branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        PRIOR_SOURCE,
        PRIOR_AUDIT,
        PRIOR_DECLARATION,
        PRIOR_DISPOSITION,
        NUMERICAL_BRANCH,
        FULL_COUPLED_BRANCH,
    ):
        req(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    numerical_text = read_text(NUMERICAL_BRANCH)
    full_text = read_text(FULL_COUPLED_BRANCH)

    prior_source = read_json(PRIOR_SOURCE)
    prior_audit = read_json(PRIOR_AUDIT)
    prior_declaration = read_json(PRIOR_DECLARATION)
    prior_disposition = read_json(PRIOR_DISPOSITION)

    interpolation_signature = hit(
        numerical_text, "def interpolate_integer_modes(scan_rows: list[dict], ell: int)"
    )
    interpolation_signature_text = interpolation_signature["text"] if interpolation_signature else ""
    interpolation_signature_has_k_argument = ", k" in interpolation_signature_text
    numerical_ell_scan_call = hit(
        numerical_text, "ell_scan_rows = {ell: scan_ell_sector(ell) for ell in (1, 2, 3)}"
    )
    numerical_interpolation_call = hit(
        numerical_text, "base_modes_by_ell = {ell: interpolate_integer_modes(rows, ell) for ell, rows in ell_scan_rows.items()}"
    )
    full_ell_scan_call = hit(
        full_text, "ell_scan_rows = {ell: prev.scan_ell_sector(ell) for ell in (1, 2, 3)}"
    )
    full_interpolation_call = hit(
        full_text, "base_modes_by_ell = {ell: prev.interpolate_integer_modes(rows, ell) for ell, rows in ell_scan_rows.items()}"
    )
    zero_node_output_row = hit(numerical_text, '"k": 0,')
    trial_state_zero_id = hit(numerical_text, 'trial_state_id": f"M_({n},0,{ell},{s})"')

    numerical_callsite_ell_only_present = bool(
        numerical_ell_scan_call is not None and numerical_interpolation_call is not None
    )
    full_callsite_ell_only_present = bool(
        full_ell_scan_call is not None and full_interpolation_call is not None
    )
    explicit_k_callsite_present = False

    inventory_targets = [
        target_record(
            "status_next_step_8_7_56_241",
            STATUS,
            status_text,
            "current official next step は `8.7.56.241`",
            "STATUS must already point to the interpolation-signature branch.",
        ),
        target_record(
            "roadmap_branch_8_7_56_241_244",
            ROADMAP,
            roadmap_text,
            "`8.7.56.241-.244` 試練3 relaunched explicit `k>0` interpolation-signature residual branch",
            "ROADMAP must already freeze the interpolation-signature branch.",
        ),
        {
            "file_key": "interpolation_signature",
            "file": rel(NUMERICAL_BRANCH),
            "pattern": "def interpolate_integer_modes(scan_rows: list[dict], ell: int)",
            "present": interpolation_signature is not None,
            "note": "The current interpolation signature must remain visible as the primary residual surface.",
            "evidence": interpolation_signature,
        },
        {
            "file_key": "interpolation_signature_without_k_argument",
            "file": rel(NUMERICAL_BRANCH),
            "pattern": "signature excludes explicit k argument",
            "present": not interpolation_signature_has_k_argument,
            "note": "The interpolation signature still lacks an explicit node-axis argument.",
            "evidence": interpolation_signature,
        },
        {
            "file_key": "numerical_ell_only_callsite",
            "file": rel(NUMERICAL_BRANCH),
            "pattern": "base_modes_by_ell = {ell: interpolate_integer_modes(rows, ell) for ell, rows in ell_scan_rows.items()}",
            "present": numerical_callsite_ell_only_present,
            "note": "The numerical branch still calls the interpolator through an ell-only dictionary comprehension.",
            "evidence": {"ell_scan_rows": numerical_ell_scan_call, "interpolation_call": numerical_interpolation_call},
        },
        {
            "file_key": "full_coupled_ell_only_callsite",
            "file": rel(FULL_COUPLED_BRANCH),
            "pattern": "base_modes_by_ell = {ell: prev.interpolate_integer_modes(rows, ell) for ell, rows in ell_scan_rows.items()}",
            "present": full_callsite_ell_only_present,
            "note": "The full-coupled branch still mirrors the same ell-only interpolation callsite.",
            "evidence": {"ell_scan_rows": full_ell_scan_call, "interpolation_call": full_interpolation_call},
        },
        {
            "file_key": "zero_node_output_row",
            "file": rel(NUMERICAL_BRANCH),
            "pattern": "\"k\": 0,",
            "present": zero_node_output_row is not None,
            "note": "The current interpolation output still freezes zero-node rows.",
            "evidence": zero_node_output_row,
        },
        {
            "file_key": "trial_state_zero_node_identifier",
            "file": rel(NUMERICAL_BRANCH),
            "pattern": "M_(n,0,ell,s)",
            "present": trial_state_zero_id is not None,
            "note": "The downstream trial-state id still bakes the zero-node label.",
            "evidence": trial_state_zero_id,
        },
    ]
    inventory_ready = all(bool(item["present"]) for item in inventory_targets)

    interpolation_signature_extension_available = bool(
        interpolation_signature is not None and interpolation_signature_has_k_argument
    )
    interpolation_callsite_k_axis_propagation_available = bool(
        explicit_k_callsite_present and interpolation_signature_extension_available
    )
    interpolation_signature_branch_available = bool(
        interpolation_signature_extension_available and interpolation_callsite_k_axis_propagation_available
    )

    nonclosure_reason = (
        "interpolation_callsites_still_pass_only_ell_and_support_signature_excluding_explicit_k_argument"
        if not interpolation_callsite_k_axis_propagation_available
        else "interpolate_integer_modes_signature_still_excludes_explicit_k_argument"
        if not interpolation_signature_extension_available
        else None
    )

    common_inputs = {
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "prior_node_resolved_source_inventory_json": rel(PRIOR_SOURCE),
        "prior_node_resolved_identification_audit_json": rel(PRIOR_AUDIT),
        "prior_declaration_fourth_gate_json": rel(PRIOR_DECLARATION),
        "prior_paper_sync_trial4_disposition_fourth_refresh_json": rel(PRIOR_DISPOSITION),
        "mass_origin_vector_qball_numerical_solver_branch_py": rel(NUMERICAL_BRANCH),
        "mass_origin_vector_qball_full_coupled_solver_branch_py": rel(FULL_COUPLED_BRANCH),
    }

    inventory = payload(
        "8.7.56.241",
        "Trial-3 relaunched explicit k-positive interpolation-signature source inventory",
        common_inputs,
        "Freeze the signature-level solver-side blocker by inventorying the interpolation signature itself, the ell-only caller dependency, and the downstream zero-node propagation that still supports it.",
        {
            "inventory_rule": "collect the interpolation signature, both ell-only callers, and the downstream zero-node propagation in one machine-readable pack",
            "signature_rule": "the source pack must say explicitly whether interpolate_integer_modes already exposes a k-axis argument",
            "callsite_rule": "the source pack must say explicitly whether any current caller already propagates a k axis into the interpolation layer",
        },
        [
            row(
                "trial3_relaunched_interpolation_signature_source_inventory_complete",
                "pass",
                "Trial-3 relaunched interpolation-signature source inventory complete",
                1,
                "The interpolation-signature source inventory is frozen.",
            ),
            row(
                "trial3_relaunched_interpolation_signature_has_k_argument",
                "pass" if interpolation_signature_has_k_argument else "reject",
                "interpolation signature has explicit k argument",
                1 if interpolation_signature_has_k_argument else 0,
                "The current signature still exposes only scan_rows and ell.",
            ),
            row(
                "trial3_relaunched_numerical_callsite_ell_only_present",
                "pass" if numerical_callsite_ell_only_present else "reject",
                "numerical interpolation callsite remains ell-only",
                1 if numerical_callsite_ell_only_present else 0,
                "The numerical branch still constructs base_modes_by_ell with an ell-only interpolation call.",
            ),
            row(
                "trial3_relaunched_full_coupled_callsite_ell_only_present",
                "pass" if full_callsite_ell_only_present else "reject",
                "full-coupled interpolation callsite remains ell-only",
                1 if full_callsite_ell_only_present else 0,
                "The full-coupled branch still mirrors the same ell-only interpolation call.",
            ),
        ],
        {
            "inventory_ready": inventory_ready,
            "interpolation_signature_present": interpolation_signature is not None,
            "interpolation_signature_has_k_argument": interpolation_signature_has_k_argument,
            "numerical_interpolation_callsite_ell_only_present": numerical_callsite_ell_only_present,
            "full_coupled_interpolation_callsite_ell_only_present": full_callsite_ell_only_present,
            "explicit_k_callsite_present": explicit_k_callsite_present,
            "first_route_to_close_or_none": "trial3_relaunched_explicit_k_positive_interpolation_signature_identification_audit",
        },
        {
            "overall_status": "trial3_relaunched_interpolation_signature_source_inventory_frozen",
            "trial3_branch_closeable": False,
            "advance_to_8_7_56_242": True,
            "next_required_artifacts": [
                "trial3_relaunched_explicit_k_positive_interpolation_signature_identification_audit"
            ],
        },
        {
            "inventory_targets": inventory_targets,
            "prior_node_resolved_source_summary": prior_source["summary"],
            "prior_node_resolved_audit_summary": prior_audit["summary"],
            "prior_declaration_summary": prior_declaration["summary"],
            "prior_disposition_summary": prior_disposition["summary"],
        },
    )

    audit = payload(
        "8.7.56.242",
        "Trial-3 relaunched explicit k-positive interpolation-signature identification audit",
        common_inputs,
        "Audit whether the missing explicit k-positive interpolation signature is blocked only by the function signature itself or more upstream by the ell-only caller dependency that still defines the current solver surface.",
        {
            "signature_audit_rule": "the signature layer passes only if interpolate_integer_modes exposes an explicit k-axis argument",
            "callsite_audit_rule": "the callsite layer passes only if at least one current caller propagates a k axis into the interpolation layer",
            "branch_audit_rule": "the interpolation-signature branch passes only if both the signature and the callsite propagation layers pass together",
        },
        [
            row(
                "trial3_relaunched_interpolation_signature_extension_available",
                "pass" if interpolation_signature_extension_available else "reject",
                "explicit k-positive interpolation signature extension available",
                1 if interpolation_signature_extension_available else 0,
                "The current function signature still excludes the node axis.",
            ),
            row(
                "trial3_relaunched_interpolation_callsite_k_axis_propagation_available",
                "pass" if interpolation_callsite_k_axis_propagation_available else "reject",
                "interpolation callsite k-axis propagation available",
                1 if interpolation_callsite_k_axis_propagation_available else 0,
                "Current callers still pass only ell and never propagate a k axis into the interpolation layer.",
            ),
            row(
                "trial3_relaunched_interpolation_signature_branch_available",
                "pass" if interpolation_signature_branch_available else "reject",
                "explicit k-positive interpolation-signature branch available",
                1 if interpolation_signature_branch_available else 0,
                "The current solver surface still cannot support an honest k-positive interpolation signature.",
            ),
        ],
        {
            "trial3_relaunched_explicit_k_positive_interpolation_signature_extension_available": interpolation_signature_extension_available,
            "trial3_relaunched_explicit_k_positive_interpolation_callsite_k_axis_propagation_available": interpolation_callsite_k_axis_propagation_available,
            "trial3_relaunched_explicit_k_positive_interpolation_signature_branch_available": interpolation_signature_branch_available,
            "identification_nonclosure_reason_or_none": nonclosure_reason,
            "first_route_to_close_or_none": "trial3_relaunched_declaration_fifth_gate",
        },
        {
            "overall_status": "trial3_relaunched_interpolation_signature_identification_audited",
            "trial3_branch_closeable": False,
            "advance_to_8_7_56_243": True,
            "next_required_artifacts": ["trial3_relaunched_declaration_fifth_gate"],
        },
        {
            "prior_node_resolved_audit_summary": prior_audit["summary"],
            "interpolation_signature": interpolation_signature,
            "numerical_ell_scan_call": numerical_ell_scan_call,
            "numerical_interpolation_call": numerical_interpolation_call,
            "full_ell_scan_call": full_ell_scan_call,
            "full_interpolation_call": full_interpolation_call,
            "zero_node_output_row": zero_node_output_row,
            "trial_state_zero_id": trial_state_zero_id,
        },
    )

    declaration = payload(
        "8.7.56.243",
        "Trial-3 relaunched declaration fifth gate",
        common_inputs,
        "Freeze whether the interpolation-signature residual is already closeable or whether the next official route must shrink again to the missing callsite-side k-axis propagation artifact.",
        {
            "gate_rule": "the branch closes only if the explicit k-positive interpolation signature and its caller-side k-axis propagation both exist",
            "reserve_rule": "Trial-2 paper-side sync remains unlocked reserve work while the scientific Trial-3 residual stays open",
        },
        [
            row(
                "trial3_relaunched_fifth_declaration_gate_complete",
                "pass",
                "Trial-3 relaunched fifth declaration gate complete",
                1,
                "The fifth declaration gate is frozen.",
            ),
            row(
                "trial3_relaunched_interpolation_signature_branch_closeable",
                "pass" if interpolation_signature_branch_available else "reject",
                "interpolation-signature branch closeable",
                1 if interpolation_signature_branch_available else 0,
                "The branch remains non-closeable while the ell-only caller dependency still blocks k-axis propagation.",
            ),
            row(
                "trial3_relaunched_interpolation_signature_residual_route_required",
                "pass" if not interpolation_signature_branch_available else "reject",
                "interpolation-signature residual route required",
                1 if not interpolation_signature_branch_available else 0,
                "A narrower residual route remains necessary after the interpolation-signature audit.",
            ),
            row(
                "trial3_relaunched_trial2_paper_side_sync_execute_now",
                "reject",
                "execute Trial-2 paper-side sync now",
                0,
                "Trial-2 paper sync stays unlocked reserve work while the scientific Trial-3 residual remains open.",
            ),
        ],
        {
            "trial3_relaunched_branch_closeable": interpolation_signature_branch_available,
            "trial3_relaunched_residual_route_required": not interpolation_signature_branch_available,
            "trial2_paper_side_sync_execute_now": False,
            "trial4_deferred": True,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial3_relaunched_fifth_declaration_gate_frozen",
            "trial3_branch_closeable": interpolation_signature_branch_available,
            "advance_to_8_7_56_244": True,
            "next_required_artifacts": ["trial3_relaunched_paper_sync_trial4_disposition_fifth_refresh"],
        },
        {
            "source_inventory_summary": inventory["summary"],
            "signature_identification_summary": audit["summary"],
            "prior_declaration_summary": prior_declaration["summary"],
        },
    )

    disposition = payload(
        "8.7.56.244",
        "Trial-2 paper-side sync / Trial-4 disposition fifth refresh",
        common_inputs,
        "Refresh the reserve/deferred ordering after the interpolation-signature audit and freeze the next residual route for the relaunched weak-sector mainline.",
        {
            "trial2_rule": "retain Trial-2 paper-side sync as unlocked reserve work while the scientific Trial-3 residual is still open",
            "trial4_rule": "keep Trial-4 deferred until the relaunched Trial-3 branch loses all honest current-canon solver routes",
            "selected_residual_route": RESIDUAL_ROUTE,
            "missing_v2_artifact": MISSING_ARTIFACT,
        },
        [
            row(
                "trial3_relaunched_fifth_disposition_gate_complete",
                "pass",
                "Trial-3 relaunched fifth disposition gate complete",
                1,
                "The reserve/deferred ordering after the interpolation-signature audit is frozen.",
            ),
            row(
                "trial3_relaunched_fifth_trial2_paper_side_sync_reserve_retained",
                "pass",
                "Trial-2 paper-side sync retained as unlocked reserve",
                1,
                "Trial-2 paper-side sync remains available but not promoted ahead of the solver-side residual.",
            ),
            row(
                "trial3_relaunched_fifth_trial4_deferred_retained",
                "pass",
                "Trial-4 deferred disposition retained",
                1,
                "Trial-4 remains deferred while the Trial-3 solver residual remains honest.",
            ),
            row(
                "trial3_relaunched_fifth_next_residual_route_frozen",
                "pass",
                "next residual route frozen",
                1,
                "The next blocker is the missing callsite-side k-axis propagation required by the interpolation signature extension.",
            ),
        ],
        {
            "selected_residual_route": RESIDUAL_ROUTE,
            "missing_v2_artifact": MISSING_ARTIFACT,
            "trial2_paper_side_sync_state": "unlocked_reserve_retained",
            "trial4_deferred": True,
            "split_contract_ready": True,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial3_relaunched_fifth_disposition_gate_frozen",
            "trial3_branch_closeable": interpolation_signature_branch_available,
            "advance_to_8_7_56_245": True,
            "next_required_artifacts": [
                "trial3_relaunched_explicit_k_positive_interpolation_callsite_dependency_source_inventory",
                "trial3_relaunched_explicit_k_positive_interpolation_callsite_dependency_identification_audit",
            ],
        },
        {
            "source_inventory_summary": inventory["summary"],
            "signature_identification_summary": audit["summary"],
            "declaration_summary": declaration["summary"],
            "prior_disposition_summary": prior_disposition["summary"],
            "ai_context_current_step": ai_context["current_step"],
        },
    )

    write_artifact(
        "mass_origin_v2_trial3_relaunched_explicit_k_positive_interpolation_signature_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial3_relaunched_explicit_k_positive_interpolation_signature_identification_audit",
        audit,
    )
    write_artifact("mass_origin_v2_trial3_relaunched_declaration_fifth_gate", declaration)
    write_artifact(
        "mass_origin_v2_trial3_relaunched_paper_sync_trial4_disposition_fifth_refresh",
        disposition,
    )

    print("[ok] wrote:")
    print(" - mass_origin_v2_trial3_relaunched_explicit_k_positive_interpolation_signature_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial3_relaunched_explicit_k_positive_interpolation_signature_identification_audit_metrics.json")
    print(" - mass_origin_v2_trial3_relaunched_declaration_fifth_gate_metrics.json")
    print(" - mass_origin_v2_trial3_relaunched_paper_sync_trial4_disposition_fifth_refresh_metrics.json")


# 関数: CLI から `.241-.244` branch を実行する。

if __name__ == "__main__":
    main()
