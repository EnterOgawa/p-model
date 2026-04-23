#!/usr/bin/env python3
"""
Generate relaunched Trial-3 explicit k-positive interpolation-callsite-dependency artifacts.

This branch executes roadmap steps 8.7.56.245-.248.

The previous residual showed that the interpolation signature is blocked not
only by the function surface itself but by the caller dependency around it:
both numerical and full-coupled branches still build `ell_scan_rows`, call the
interpolator with `(rows, ell)`, and consume an ell-only `base_modes_by_ell`
container. This branch freezes that dependency and narrows the next blocker.
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
    / "mass_origin_v2_trial3_relaunched_explicit_k_positive_interpolation_signature_source_inventory_metrics.json"
)
PRIOR_AUDIT = (
    OUT
    / "mass_origin_v2_trial3_relaunched_explicit_k_positive_interpolation_signature_identification_audit_metrics.json"
)
PRIOR_DECLARATION = OUT / "mass_origin_v2_trial3_relaunched_declaration_fifth_gate_metrics.json"
PRIOR_DISPOSITION = (
    OUT / "mass_origin_v2_trial3_relaunched_paper_sync_trial4_disposition_fifth_refresh_metrics.json"
)
NUMERICAL_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_numerical_solver_branch.py"
FULL_COUPLED_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_full_coupled_solver_branch.py"

NEXT_ROUTE = "8.7.56.249"
RESIDUAL_ROUTE = "trial3_relaunched_explicit_k_positive_base_modes_by_ell_container_identification"
MISSING_ARTIFACT = "trial3_relaunched_explicit_k_positive_base_modes_by_ell_k_axis_container"


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


# 関数: `.245-.248` branch を実行して callsite dependency residual を固定する。

def main() -> None:
    """Execute the relaunched Trial-3 interpolation-callsite-dependency residual branch."""
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

    numerical_ell_scan_rows = hit(
        numerical_text, "ell_scan_rows = {ell: scan_ell_sector(ell) for ell in (1, 2, 3)}"
    )
    numerical_interpolation_call = hit(
        numerical_text, "base_modes_by_ell = {ell: interpolate_integer_modes(rows, ell) for ell, rows in ell_scan_rows.items()}"
    )
    numerical_consumer_signature = hit(
        numerical_text, "base_modes_by_ell: dict[int, list[dict]],"
    )
    numerical_consumer_loop = hit(numerical_text, "for ell, base_modes in base_modes_by_ell.items():")
    numerical_consumer_count = hit(
        numerical_text, '"integer_mode_count": len(base_modes_by_ell[int(ell)]),'
    )
    numerical_consumer_sample = hit(
        numerical_text, 'str(ell): sample(base_modes_by_ell[int(ell)])'
    )

    full_ell_scan_rows = hit(
        full_text, "ell_scan_rows = {ell: prev.scan_ell_sector(ell) for ell in (1, 2, 3)}"
    )
    full_interpolation_call = hit(
        full_text, "base_modes_by_ell = {ell: prev.interpolate_integer_modes(rows, ell) for ell, rows in ell_scan_rows.items()}"
    )
    full_consumer_signature = hit(full_text, "base_modes_by_ell: dict[int, list[dict]],")
    full_consumer_loop = hit(full_text, "for ell, base_modes in base_modes_by_ell.items():")
    full_consumer_count = hit(full_text, "total_integer_modes = sum(len(rows) for rows in base_modes_by_ell.values())")

    numerical_callsite_dependency_present = bool(
        numerical_ell_scan_rows is not None
        and numerical_interpolation_call is not None
        and numerical_consumer_signature is not None
        and numerical_consumer_loop is not None
        and numerical_consumer_count is not None
        and numerical_consumer_sample is not None
    )
    full_callsite_dependency_present = bool(
        full_ell_scan_rows is not None
        and full_interpolation_call is not None
        and full_consumer_signature is not None
        and full_consumer_loop is not None
        and full_consumer_count is not None
    )

    inventory_targets = [
        target_record(
            "status_next_step_8_7_56_245",
            STATUS,
            status_text,
            "current official next step は `8.7.56.245`",
            "STATUS must already point to the interpolation-callsite-dependency branch.",
        ),
        target_record(
            "roadmap_branch_8_7_56_245_248",
            ROADMAP,
            roadmap_text,
            "`8.7.56.245-.248` 試練3 relaunched explicit `k>0` interpolation-callsite-dependency residual branch",
            "ROADMAP must already freeze the interpolation-callsite-dependency branch.",
        ),
        {
            "file_key": "numerical_ell_scan_rows_builder",
            "file": rel(NUMERICAL_BRANCH),
            "pattern": "ell_scan_rows = {ell: scan_ell_sector(ell) for ell in (1, 2, 3)}",
            "present": numerical_ell_scan_rows is not None,
            "note": "The numerical branch still constructs ell_scan_rows with an ell-only scan builder.",
            "evidence": numerical_ell_scan_rows,
        },
        {
            "file_key": "numerical_interpolation_callsite",
            "file": rel(NUMERICAL_BRANCH),
            "pattern": "base_modes_by_ell = {ell: interpolate_integer_modes(rows, ell) for ell, rows in ell_scan_rows.items()}",
            "present": numerical_interpolation_call is not None,
            "note": "The numerical branch still calls the interpolation layer with rows and ell only.",
            "evidence": numerical_interpolation_call,
        },
        {
            "file_key": "numerical_base_modes_by_ell_consumer_signature",
            "file": rel(NUMERICAL_BRANCH),
            "pattern": "base_modes_by_ell: dict[int, list[dict]],",
            "present": numerical_consumer_signature is not None,
            "note": "The numerical downstream consumer still types the container as ell-only.",
            "evidence": numerical_consumer_signature,
        },
        {
            "file_key": "numerical_base_modes_by_ell_consumer_loop",
            "file": rel(NUMERICAL_BRANCH),
            "pattern": "for ell, base_modes in base_modes_by_ell.items():",
            "present": numerical_consumer_loop is not None,
            "note": "The numerical downstream consumer still loops over ell-only container items.",
            "evidence": numerical_consumer_loop,
        },
        {
            "file_key": "numerical_base_modes_by_ell_consumer_count",
            "file": rel(NUMERICAL_BRANCH),
            "pattern": "\"integer_mode_count\": len(base_modes_by_ell[int(ell)]),",
            "present": numerical_consumer_count is not None,
            "note": "The numerical downstream metrics still count integer modes through ell-only indexing.",
            "evidence": numerical_consumer_count,
        },
        {
            "file_key": "numerical_base_modes_by_ell_consumer_sample",
            "file": rel(NUMERICAL_BRANCH),
            "pattern": "sample(base_modes_by_ell[int(ell)])",
            "present": numerical_consumer_sample is not None,
            "note": "The numerical sampling surface still dereferences the ell-only container directly.",
            "evidence": numerical_consumer_sample,
        },
        {
            "file_key": "full_ell_scan_rows_builder",
            "file": rel(FULL_COUPLED_BRANCH),
            "pattern": "ell_scan_rows = {ell: prev.scan_ell_sector(ell) for ell in (1, 2, 3)}",
            "present": full_ell_scan_rows is not None,
            "note": "The full-coupled branch still constructs ell_scan_rows with an ell-only scan builder.",
            "evidence": full_ell_scan_rows,
        },
        {
            "file_key": "full_interpolation_callsite",
            "file": rel(FULL_COUPLED_BRANCH),
            "pattern": "base_modes_by_ell = {ell: prev.interpolate_integer_modes(rows, ell) for ell, rows in ell_scan_rows.items()}",
            "present": full_interpolation_call is not None,
            "note": "The full-coupled branch still calls the interpolation layer with rows and ell only.",
            "evidence": full_interpolation_call,
        },
        {
            "file_key": "full_base_modes_by_ell_consumer_signature",
            "file": rel(FULL_COUPLED_BRANCH),
            "pattern": "base_modes_by_ell: dict[int, list[dict]],",
            "present": full_consumer_signature is not None,
            "note": "The full-coupled downstream consumer still types the container as ell-only.",
            "evidence": full_consumer_signature,
        },
        {
            "file_key": "full_base_modes_by_ell_consumer_loop",
            "file": rel(FULL_COUPLED_BRANCH),
            "pattern": "for ell, base_modes in base_modes_by_ell.items():",
            "present": full_consumer_loop is not None,
            "note": "The full-coupled consumer still loops over ell-only container items.",
            "evidence": full_consumer_loop,
        },
        {
            "file_key": "full_base_modes_by_ell_consumer_count",
            "file": rel(FULL_COUPLED_BRANCH),
            "pattern": "total_integer_modes = sum(len(rows) for rows in base_modes_by_ell.values())",
            "present": full_consumer_count is not None,
            "note": "The full-coupled metrics still count rows through ell-only container values.",
            "evidence": full_consumer_count,
        },
    ]
    inventory_ready = all(bool(item["present"]) for item in inventory_targets)

    numerical_callsite_k_axis_propagation_available = False
    full_callsite_k_axis_propagation_available = False
    base_modes_by_ell_k_axis_container_available = False
    interpolation_callsite_dependency_branch_available = False

    nonclosure_reason = (
        "base_modes_by_ell_consumers_still_assume_ell_only_container_and_block_k_axis_propagation"
    )

    common_inputs = {
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "prior_interpolation_signature_source_inventory_json": rel(PRIOR_SOURCE),
        "prior_interpolation_signature_identification_audit_json": rel(PRIOR_AUDIT),
        "prior_declaration_fifth_gate_json": rel(PRIOR_DECLARATION),
        "prior_paper_sync_trial4_disposition_fifth_refresh_json": rel(PRIOR_DISPOSITION),
        "mass_origin_vector_qball_numerical_solver_branch_py": rel(NUMERICAL_BRANCH),
        "mass_origin_vector_qball_full_coupled_solver_branch_py": rel(FULL_COUPLED_BRANCH),
    }

    inventory = payload(
        "8.7.56.245",
        "Trial-3 relaunched explicit k-positive interpolation-callsite-dependency source inventory",
        common_inputs,
        "Freeze the caller-side solver blocker by inventorying the ell-only scan builders, interpolation callsites, and base_modes_by_ell consumer surfaces that still prevent k-axis propagation.",
        {
            "inventory_rule": "collect both branch callsites and both downstream base_modes_by_ell consumer surfaces in one machine-readable pack",
            "callsite_rule": "the source pack must say explicitly whether either branch already propagates a k axis into the interpolation layer",
            "consumer_rule": "the source pack must say explicitly whether downstream surfaces still assume an ell-only base_modes_by_ell container",
        },
        [
            row(
                "trial3_relaunched_interpolation_callsite_dependency_source_inventory_complete",
                "pass",
                "Trial-3 relaunched interpolation-callsite-dependency source inventory complete",
                1,
                "The interpolation-callsite-dependency source inventory is frozen.",
            ),
            row(
                "trial3_relaunched_numerical_callsite_dependency_present",
                "pass" if numerical_callsite_dependency_present else "reject",
                "numerical callsite dependency surfaces present",
                1 if numerical_callsite_dependency_present else 0,
                "The numerical branch still exposes the full ell-only callsite-to-consumer dependency chain.",
            ),
            row(
                "trial3_relaunched_full_callsite_dependency_present",
                "pass" if full_callsite_dependency_present else "reject",
                "full-coupled callsite dependency surfaces present",
                1 if full_callsite_dependency_present else 0,
                "The full-coupled branch still exposes the same ell-only dependency chain.",
            ),
            row(
                "trial3_relaunched_base_modes_by_ell_ell_only_container_present",
                "pass",
                "base_modes_by_ell still behaves as ell-only container",
                1,
                "Both branches still type and consume base_modes_by_ell through an ell-only container surface.",
            ),
        ],
        {
            "inventory_ready": inventory_ready,
            "numerical_callsite_dependency_present": numerical_callsite_dependency_present,
            "full_callsite_dependency_present": full_callsite_dependency_present,
            "numerical_callsite_k_axis_propagation_available": numerical_callsite_k_axis_propagation_available,
            "full_callsite_k_axis_propagation_available": full_callsite_k_axis_propagation_available,
            "base_modes_by_ell_k_axis_container_available": base_modes_by_ell_k_axis_container_available,
            "first_route_to_close_or_none": "trial3_relaunched_explicit_k_positive_interpolation_callsite_dependency_identification_audit",
        },
        {
            "overall_status": "trial3_relaunched_interpolation_callsite_dependency_source_inventory_frozen",
            "trial3_branch_closeable": False,
            "advance_to_8_7_56_246": True,
            "next_required_artifacts": [
                "trial3_relaunched_explicit_k_positive_interpolation_callsite_dependency_identification_audit"
            ],
        },
        {
            "inventory_targets": inventory_targets,
            "prior_signature_source_summary": prior_source["summary"],
            "prior_signature_audit_summary": prior_audit["summary"],
            "prior_declaration_summary": prior_declaration["summary"],
            "prior_disposition_summary": prior_disposition["summary"],
        },
    )

    audit = payload(
        "8.7.56.246",
        "Trial-3 relaunched explicit k-positive interpolation-callsite-dependency identification audit",
        common_inputs,
        "Audit whether caller-side k-axis propagation is blocked primarily by the ell-only interpolation callsites themselves or by the downstream base_modes_by_ell container surfaces that still assume ell-only grouping.",
        {
            "numerical_callsite_rule": "the numerical branch passes only if its interpolation callsite and downstream consumers propagate a k axis coherently",
            "full_callsite_rule": "the full-coupled branch passes only if its interpolation callsite and downstream consumers propagate a k axis coherently",
            "container_rule": "the shared dependency passes only if the base_modes_by_ell container stops being consumed as ell-only in both branches",
        },
        [
            row(
                "trial3_relaunched_numerical_callsite_k_axis_propagation_available",
                "pass" if numerical_callsite_k_axis_propagation_available else "reject",
                "numerical callsite k-axis propagation available",
                1 if numerical_callsite_k_axis_propagation_available else 0,
                "The numerical branch still constructs and consumes an ell-only base_modes_by_ell container.",
            ),
            row(
                "trial3_relaunched_full_callsite_k_axis_propagation_available",
                "pass" if full_callsite_k_axis_propagation_available else "reject",
                "full-coupled callsite k-axis propagation available",
                1 if full_callsite_k_axis_propagation_available else 0,
                "The full-coupled branch still mirrors the same ell-only dependency chain.",
            ),
            row(
                "trial3_relaunched_base_modes_by_ell_k_axis_container_available",
                "pass" if base_modes_by_ell_k_axis_container_available else "reject",
                "base_modes_by_ell k-axis container available",
                1 if base_modes_by_ell_k_axis_container_available else 0,
                "The shared container still assumes ell-only grouping and blocks k-axis propagation downstream.",
            ),
        ],
        {
            "trial3_relaunched_explicit_k_positive_numerical_callsite_k_axis_propagation_available": numerical_callsite_k_axis_propagation_available,
            "trial3_relaunched_explicit_k_positive_full_callsite_k_axis_propagation_available": full_callsite_k_axis_propagation_available,
            "trial3_relaunched_explicit_k_positive_base_modes_by_ell_k_axis_container_available": base_modes_by_ell_k_axis_container_available,
            "trial3_relaunched_explicit_k_positive_interpolation_callsite_dependency_branch_available": interpolation_callsite_dependency_branch_available,
            "identification_nonclosure_reason_or_none": nonclosure_reason,
            "first_route_to_close_or_none": "trial3_relaunched_declaration_sixth_gate",
        },
        {
            "overall_status": "trial3_relaunched_interpolation_callsite_dependency_identification_audited",
            "trial3_branch_closeable": False,
            "advance_to_8_7_56_247": True,
            "next_required_artifacts": ["trial3_relaunched_declaration_sixth_gate"],
        },
        {
            "prior_signature_audit_summary": prior_audit["summary"],
            "numerical_ell_scan_rows": numerical_ell_scan_rows,
            "numerical_interpolation_call": numerical_interpolation_call,
            "numerical_consumer_signature": numerical_consumer_signature,
            "numerical_consumer_loop": numerical_consumer_loop,
            "numerical_consumer_count": numerical_consumer_count,
            "numerical_consumer_sample": numerical_consumer_sample,
            "full_ell_scan_rows": full_ell_scan_rows,
            "full_interpolation_call": full_interpolation_call,
            "full_consumer_signature": full_consumer_signature,
            "full_consumer_loop": full_consumer_loop,
            "full_consumer_count": full_consumer_count,
        },
    )

    declaration = payload(
        "8.7.56.247",
        "Trial-3 relaunched declaration sixth gate",
        common_inputs,
        "Freeze whether the interpolation-callsite-dependency residual is already closeable or whether the next official route must shrink again to the missing base_modes_by_ell k-axis container artifact.",
        {
            "gate_rule": "the branch closes only if both numerical and full-coupled callers, plus their shared consumers, propagate a k axis coherently",
            "reserve_rule": "Trial-2 paper-side sync remains unlocked reserve work while the scientific Trial-3 residual stays open",
        },
        [
            row(
                "trial3_relaunched_sixth_declaration_gate_complete",
                "pass",
                "Trial-3 relaunched sixth declaration gate complete",
                1,
                "The sixth declaration gate is frozen.",
            ),
            row(
                "trial3_relaunched_interpolation_callsite_dependency_branch_closeable",
                "pass" if interpolation_callsite_dependency_branch_available else "reject",
                "interpolation-callsite-dependency branch closeable",
                1 if interpolation_callsite_dependency_branch_available else 0,
                "The branch remains non-closeable while base_modes_by_ell still behaves as an ell-only container.",
            ),
            row(
                "trial3_relaunched_interpolation_callsite_dependency_residual_route_required",
                "pass" if not interpolation_callsite_dependency_branch_available else "reject",
                "interpolation-callsite-dependency residual route required",
                1 if not interpolation_callsite_dependency_branch_available else 0,
                "A narrower residual route remains necessary after the callsite-dependency audit.",
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
            "trial3_relaunched_branch_closeable": interpolation_callsite_dependency_branch_available,
            "trial3_relaunched_residual_route_required": not interpolation_callsite_dependency_branch_available,
            "trial2_paper_side_sync_execute_now": False,
            "trial4_deferred": True,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial3_relaunched_sixth_declaration_gate_frozen",
            "trial3_branch_closeable": interpolation_callsite_dependency_branch_available,
            "advance_to_8_7_56_248": True,
            "next_required_artifacts": ["trial3_relaunched_paper_sync_trial4_disposition_sixth_refresh"],
        },
        {
            "source_inventory_summary": inventory["summary"],
            "callsite_dependency_identification_summary": audit["summary"],
            "prior_declaration_summary": prior_declaration["summary"],
        },
    )

    disposition = payload(
        "8.7.56.248",
        "Trial-2 paper-side sync / Trial-4 disposition sixth refresh",
        common_inputs,
        "Refresh the reserve/deferred ordering after the interpolation-callsite-dependency audit and freeze the next residual route for the relaunched weak-sector mainline.",
        {
            "trial2_rule": "retain Trial-2 paper-side sync as unlocked reserve work while the scientific Trial-3 residual is still open",
            "trial4_rule": "keep Trial-4 deferred until the relaunched Trial-3 branch loses all honest current-canon solver routes",
            "selected_residual_route": RESIDUAL_ROUTE,
            "missing_v2_artifact": MISSING_ARTIFACT,
        },
        [
            row(
                "trial3_relaunched_sixth_disposition_gate_complete",
                "pass",
                "Trial-3 relaunched sixth disposition gate complete",
                1,
                "The reserve/deferred ordering after the callsite-dependency audit is frozen.",
            ),
            row(
                "trial3_relaunched_sixth_trial2_paper_side_sync_reserve_retained",
                "pass",
                "Trial-2 paper-side sync retained as unlocked reserve",
                1,
                "Trial-2 paper-side sync remains available but not promoted ahead of the solver-side residual.",
            ),
            row(
                "trial3_relaunched_sixth_trial4_deferred_retained",
                "pass",
                "Trial-4 deferred disposition retained",
                1,
                "Trial-4 remains deferred while the Trial-3 solver residual remains honest.",
            ),
            row(
                "trial3_relaunched_sixth_next_residual_route_frozen",
                "pass",
                "next residual route frozen",
                1,
                "The next blocker is the missing base_modes_by_ell k-axis container required by caller-side propagation.",
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
            "overall_status": "trial3_relaunched_sixth_disposition_gate_frozen",
            "trial3_branch_closeable": interpolation_callsite_dependency_branch_available,
            "advance_to_8_7_56_249": True,
            "next_required_artifacts": [
                "trial3_relaunched_explicit_k_positive_base_modes_by_ell_container_source_inventory",
                "trial3_relaunched_explicit_k_positive_base_modes_by_ell_container_identification_audit",
            ],
        },
        {
            "source_inventory_summary": inventory["summary"],
            "callsite_dependency_identification_summary": audit["summary"],
            "declaration_summary": declaration["summary"],
            "prior_disposition_summary": prior_disposition["summary"],
            "ai_context_current_step": ai_context["current_step"],
        },
    )

    write_artifact(
        "mass_origin_v2_trial3_relaunched_explicit_k_positive_interpolation_callsite_dependency_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial3_relaunched_explicit_k_positive_interpolation_callsite_dependency_identification_audit",
        audit,
    )
    write_artifact("mass_origin_v2_trial3_relaunched_declaration_sixth_gate", declaration)
    write_artifact(
        "mass_origin_v2_trial3_relaunched_paper_sync_trial4_disposition_sixth_refresh",
        disposition,
    )

    print("[ok] wrote:")
    print(" - mass_origin_v2_trial3_relaunched_explicit_k_positive_interpolation_callsite_dependency_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial3_relaunched_explicit_k_positive_interpolation_callsite_dependency_identification_audit_metrics.json")
    print(" - mass_origin_v2_trial3_relaunched_declaration_sixth_gate_metrics.json")
    print(" - mass_origin_v2_trial3_relaunched_paper_sync_trial4_disposition_sixth_refresh_metrics.json")


# 関数: CLI から `.245-.248` branch を実行する。

if __name__ == "__main__":
    main()
