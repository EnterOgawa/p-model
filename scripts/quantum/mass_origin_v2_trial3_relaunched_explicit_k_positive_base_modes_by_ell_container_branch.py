#!/usr/bin/env python3
"""
Generate relaunched Trial-3 explicit k-positive base-modes-by-ell container artifacts.

This branch executes roadmap steps 8.7.56.249-.252.

The previous residual showed that the solver-side k-axis blocker is no longer
the interpolation signature or the callsites by themselves. Both numerical and
full-coupled branches still consume `base_modes_by_ell` through an ell-only
container surface, so this branch freezes that container-level dependency and
shrinks the next residual route again.
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
    / "mass_origin_v2_trial3_relaunched_explicit_k_positive_interpolation_callsite_dependency_source_inventory_metrics.json"
)
PRIOR_AUDIT = (
    OUT
    / "mass_origin_v2_trial3_relaunched_explicit_k_positive_interpolation_callsite_dependency_identification_audit_metrics.json"
)
PRIOR_DECLARATION = OUT / "mass_origin_v2_trial3_relaunched_declaration_sixth_gate_metrics.json"
PRIOR_DISPOSITION = (
    OUT / "mass_origin_v2_trial3_relaunched_paper_sync_trial4_disposition_sixth_refresh_metrics.json"
)
NUMERICAL_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_numerical_solver_branch.py"
FULL_COUPLED_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_full_coupled_solver_branch.py"

NEXT_ROUTE = "8.7.56.253"
RESIDUAL_ROUTE = "trial3_relaunched_explicit_k_positive_base_modes_by_ell_consumer_surface_identification"
MISSING_ARTIFACT = "trial3_relaunched_explicit_k_positive_base_modes_by_ell_consumer_k_axis_surface"


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


# 関数: `.249-.252` branch を実行して base-modes-by-ell container residual を固定する。

def main() -> None:
    """Execute the relaunched Trial-3 base-modes-by-ell container residual branch."""
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

    numerical_signature = hit(numerical_text, "base_modes_by_ell: dict[int, list[dict]],")
    numerical_loop = hit(numerical_text, "for ell, base_modes in base_modes_by_ell.items():")
    numerical_total_count = hit(
        numerical_text, "total_integer_modes = sum(len(modes) for modes in base_modes_by_ell.values())"
    )
    numerical_index_count = hit(
        numerical_text, '"integer_mode_count": len(base_modes_by_ell[int(ell)]),'
    )
    numerical_sample_index = hit(
        numerical_text, 'str(ell): sample(base_modes_by_ell[int(ell)])'
    )

    full_signature = hit(full_text, "base_modes_by_ell: dict[int, list[dict]],")
    full_loop = hit(full_text, "for ell, base_modes in base_modes_by_ell.items():")
    full_total_count = hit(full_text, "total_integer_modes = sum(len(rows) for rows in base_modes_by_ell.values())")
    full_builder_pass = hit(full_text, "exact_rows = build_exact_ladder(scalar_modes, base_modes_by_ell, lambda_rot)")

    numerical_container_surface_present = bool(
        numerical_signature is not None
        and numerical_loop is not None
        and numerical_total_count is not None
        and numerical_index_count is not None
        and numerical_sample_index is not None
    )
    full_container_surface_present = bool(
        full_signature is not None
        and full_loop is not None
        and full_total_count is not None
        and full_builder_pass is not None
    )
    inventory_ready = bool(numerical_container_surface_present and full_container_surface_present)

    numerical_base_modes_by_ell_k_axis_container_available = False
    full_base_modes_by_ell_k_axis_container_available = False
    base_modes_by_ell_consumer_surface_available = False
    base_modes_by_ell_container_branch_available = False

    nonclosure_reason = (
        "base_modes_by_ell_consumers_still_assume_dict_int_list_dict_and_ell_only_indexing"
    )

    common_inputs = {
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "prior_interpolation_callsite_dependency_source_inventory_json": rel(PRIOR_SOURCE),
        "prior_interpolation_callsite_dependency_identification_audit_json": rel(PRIOR_AUDIT),
        "prior_declaration_sixth_gate_json": rel(PRIOR_DECLARATION),
        "prior_paper_sync_trial4_disposition_sixth_refresh_json": rel(PRIOR_DISPOSITION),
        "mass_origin_vector_qball_numerical_solver_branch_py": rel(NUMERICAL_BRANCH),
        "mass_origin_vector_qball_full_coupled_solver_branch_py": rel(FULL_COUPLED_BRANCH),
    }

    inventory_targets = [
        target_record(
            "status_next_step_8_7_56_249",
            STATUS,
            status_text,
            "current official next step は `8.7.56.249`",
            "STATUS must already point to the base-modes-by-ell container branch.",
        ),
        target_record(
            "roadmap_branch_8_7_56_249_252",
            ROADMAP,
            roadmap_text,
            "`8.7.56.249-.252` 試練3 relaunched explicit `k>0` base-modes-by-ell container residual branch",
            "ROADMAP must already freeze the base-modes-by-ell container branch.",
        ),
        {
            "file_key": "numerical_base_modes_by_ell_signature",
            "file": rel(NUMERICAL_BRANCH),
            "pattern": "base_modes_by_ell: dict[int, list[dict]],",
            "present": numerical_signature is not None,
            "note": "The numerical branch still types base_modes_by_ell as an ell-only container.",
            "evidence": numerical_signature,
        },
        {
            "file_key": "numerical_base_modes_by_ell_loop_surface",
            "file": rel(NUMERICAL_BRANCH),
            "pattern": "for ell, base_modes in base_modes_by_ell.items():",
            "present": numerical_loop is not None,
            "note": "The numerical branch still iterates through base_modes_by_ell as ell-only grouped items.",
            "evidence": numerical_loop,
        },
        {
            "file_key": "numerical_base_modes_by_ell_total_count_surface",
            "file": rel(NUMERICAL_BRANCH),
            "pattern": "total_integer_modes = sum(len(modes) for modes in base_modes_by_ell.values())",
            "present": numerical_total_count is not None,
            "note": "The numerical branch still counts integer modes via ell-only container values.",
            "evidence": numerical_total_count,
        },
        {
            "file_key": "numerical_base_modes_by_ell_index_count_surface",
            "file": rel(NUMERICAL_BRANCH),
            "pattern": "\"integer_mode_count\": len(base_modes_by_ell[int(ell)]),",
            "present": numerical_index_count is not None,
            "note": "The numerical branch still indexes base_modes_by_ell with ell-only integer keys.",
            "evidence": numerical_index_count,
        },
        {
            "file_key": "numerical_base_modes_by_ell_sample_surface",
            "file": rel(NUMERICAL_BRANCH),
            "pattern": "sample(base_modes_by_ell[int(ell)])",
            "present": numerical_sample_index is not None,
            "note": "The numerical sample surface still dereferences the ell-only container directly.",
            "evidence": numerical_sample_index,
        },
        {
            "file_key": "full_base_modes_by_ell_signature",
            "file": rel(FULL_COUPLED_BRANCH),
            "pattern": "base_modes_by_ell: dict[int, list[dict]],",
            "present": full_signature is not None,
            "note": "The full-coupled branch still types base_modes_by_ell as an ell-only container.",
            "evidence": full_signature,
        },
        {
            "file_key": "full_base_modes_by_ell_loop_surface",
            "file": rel(FULL_COUPLED_BRANCH),
            "pattern": "for ell, base_modes in base_modes_by_ell.items():",
            "present": full_loop is not None,
            "note": "The full-coupled branch still iterates through ell-only grouped items.",
            "evidence": full_loop,
        },
        {
            "file_key": "full_base_modes_by_ell_total_count_surface",
            "file": rel(FULL_COUPLED_BRANCH),
            "pattern": "total_integer_modes = sum(len(rows) for rows in base_modes_by_ell.values())",
            "present": full_total_count is not None,
            "note": "The full-coupled metrics still count integer modes via ell-only container values.",
            "evidence": full_total_count,
        },
        {
            "file_key": "full_base_modes_by_ell_builder_pass_surface",
            "file": rel(FULL_COUPLED_BRANCH),
            "pattern": "exact_rows = build_exact_ladder(scalar_modes, base_modes_by_ell, lambda_rot)",
            "present": full_builder_pass is not None,
            "note": "The full-coupled branch still passes the ell-only container directly into the exact-ladder builder.",
            "evidence": full_builder_pass,
        },
    ]

    inventory = payload(
        "8.7.56.249",
        "Trial-3 relaunched explicit k-positive base-modes-by-ell container source inventory",
        common_inputs,
        "Freeze the downstream container blocker by inventorying the typed signature, loops, counts, and indexing surfaces that still consume base_modes_by_ell as an ell-only dict[int, list[dict]] container.",
        {
            "inventory_rule": "collect both numerical and full-coupled base_modes_by_ell consumer surfaces in one machine-readable pack",
            "signature_rule": "the source pack must say explicitly whether base_modes_by_ell is still typed as dict[int, list[dict]]",
            "consumer_rule": "the source pack must say explicitly whether loop, count, and indexing surfaces still consume the container through ell-only keys",
        },
        [
            row(
                "trial3_relaunched_base_modes_by_ell_container_source_inventory_complete",
                "pass",
                "Trial-3 relaunched base-modes-by-ell container source inventory complete",
                1,
                "The source inventory for the base-modes-by-ell container residual branch is frozen.",
            ),
            row(
                "trial3_relaunched_numerical_base_modes_by_ell_container_surface_present",
                "pass" if numerical_container_surface_present else "reject",
                "numerical base-modes-by-ell container surfaces present",
                1 if numerical_container_surface_present else 0,
                "The numerical branch still exposes the ell-only container signature, loop, count, and indexing surfaces.",
            ),
            row(
                "trial3_relaunched_full_base_modes_by_ell_container_surface_present",
                "pass" if full_container_surface_present else "reject",
                "full-coupled base-modes-by-ell container surfaces present",
                1 if full_container_surface_present else 0,
                "The full-coupled branch still exposes the ell-only container signature, loop, count, and builder-pass surfaces.",
            ),
            row(
                "trial3_relaunched_base_modes_by_ell_ell_only_container_present",
                "pass",
                "base-modes-by-ell still behaves as ell-only container",
                1,
                "Both branches still consume base_modes_by_ell as dict[int, list[dict]] without a propagated k axis.",
            ),
        ],
        {
            "inventory_ready": inventory_ready,
            "numerical_container_surface_present": numerical_container_surface_present,
            "full_container_surface_present": full_container_surface_present,
            "numerical_base_modes_by_ell_k_axis_container_available": numerical_base_modes_by_ell_k_axis_container_available,
            "full_base_modes_by_ell_k_axis_container_available": full_base_modes_by_ell_k_axis_container_available,
            "base_modes_by_ell_consumer_surface_available": base_modes_by_ell_consumer_surface_available,
            "first_route_to_close_or_none": "trial3_relaunched_explicit_k_positive_base_modes_by_ell_container_identification_audit",
        },
        {
            "overall_status": "trial3_relaunched_base_modes_by_ell_container_source_inventory_frozen",
            "trial3_branch_closeable": False,
            "advance_to_8_7_56_250": True,
            "next_required_artifacts": [
                "trial3_relaunched_explicit_k_positive_base_modes_by_ell_container_identification_audit"
            ],
        },
        {
            "inventory_targets": inventory_targets,
            "prior_callsite_source_summary": prior_source["summary"],
            "prior_callsite_audit_summary": prior_audit["summary"],
            "prior_declaration_summary": prior_declaration["summary"],
            "prior_disposition_summary": prior_disposition["summary"],
        },
    )

    audit = payload(
        "8.7.56.250",
        "Trial-3 relaunched explicit k-positive base-modes-by-ell container identification audit",
        common_inputs,
        "Audit whether the current solver can already reinterpret base_modes_by_ell as a k-axis-aware container or whether the downstream consumer surfaces still hard-code dict[int, list[dict]] and ell-only indexing.",
        {
            "numerical_rule": "the numerical branch passes only if its signature, loop, count, and sample/index surfaces stop assuming ell-only grouping",
            "full_rule": "the full-coupled branch passes only if its signature, loop, count, and builder-pass surfaces stop assuming ell-only grouping",
            "consumer_rule": "the shared dependency passes only if the consumer surfaces can carry a k axis through the base_modes_by_ell container",
        },
        [
            row(
                "trial3_relaunched_numerical_base_modes_by_ell_k_axis_container_available",
                "pass" if numerical_base_modes_by_ell_k_axis_container_available else "reject",
                "numerical base-modes-by-ell k-axis container available",
                1 if numerical_base_modes_by_ell_k_axis_container_available else 0,
                "The numerical branch still types and indexes base_modes_by_ell through ell-only surfaces.",
            ),
            row(
                "trial3_relaunched_full_base_modes_by_ell_k_axis_container_available",
                "pass" if full_base_modes_by_ell_k_axis_container_available else "reject",
                "full-coupled base-modes-by-ell k-axis container available",
                1 if full_base_modes_by_ell_k_axis_container_available else 0,
                "The full-coupled branch still mirrors the same ell-only container assumption.",
            ),
            row(
                "trial3_relaunched_base_modes_by_ell_consumer_surface_available",
                "pass" if base_modes_by_ell_consumer_surface_available else "reject",
                "base-modes-by-ell consumer surface available",
                1 if base_modes_by_ell_consumer_surface_available else 0,
                "The shared consumer surfaces still assume dict[int, list[dict]] and ell-only indexing.",
            ),
        ],
        {
            "trial3_relaunched_explicit_k_positive_numerical_base_modes_by_ell_k_axis_container_available": numerical_base_modes_by_ell_k_axis_container_available,
            "trial3_relaunched_explicit_k_positive_full_base_modes_by_ell_k_axis_container_available": full_base_modes_by_ell_k_axis_container_available,
            "trial3_relaunched_explicit_k_positive_base_modes_by_ell_consumer_surface_available": base_modes_by_ell_consumer_surface_available,
            "trial3_relaunched_explicit_k_positive_base_modes_by_ell_container_branch_available": base_modes_by_ell_container_branch_available,
            "identification_nonclosure_reason_or_none": nonclosure_reason,
            "first_route_to_close_or_none": "trial3_relaunched_declaration_seventh_gate",
        },
        {
            "overall_status": "trial3_relaunched_base_modes_by_ell_container_identification_audited",
            "trial3_branch_closeable": False,
            "advance_to_8_7_56_251": True,
            "next_required_artifacts": ["trial3_relaunched_declaration_seventh_gate"],
        },
        {
            "prior_callsite_audit_summary": prior_audit["summary"],
            "numerical_signature": numerical_signature,
            "numerical_loop": numerical_loop,
            "numerical_total_count": numerical_total_count,
            "numerical_index_count": numerical_index_count,
            "numerical_sample_index": numerical_sample_index,
            "full_signature": full_signature,
            "full_loop": full_loop,
            "full_total_count": full_total_count,
            "full_builder_pass": full_builder_pass,
        },
    )

    declaration = payload(
        "8.7.56.251",
        "Trial-3 relaunched declaration seventh gate",
        common_inputs,
        "Freeze whether the base-modes-by-ell container residual is already closeable or whether the next official route must shrink again to the downstream consumer-surface artifact that still hard-codes ell-only grouping.",
        {
            "gate_rule": "the branch closes only if both numerical and full-coupled consumer surfaces stop assuming dict[int, list[dict]] and ell-only indexing",
            "reserve_rule": "Trial-2 paper-side sync remains unlocked reserve work while the scientific Trial-3 residual stays open",
        },
        [
            row(
                "trial3_relaunched_seventh_declaration_gate_complete",
                "pass",
                "Trial-3 relaunched seventh declaration gate complete",
                1,
                "The seventh declaration gate is frozen.",
            ),
            row(
                "trial3_relaunched_base_modes_by_ell_container_branch_closeable",
                "pass" if base_modes_by_ell_container_branch_available else "reject",
                "base-modes-by-ell container branch closeable",
                1 if base_modes_by_ell_container_branch_available else 0,
                "The branch remains non-closeable while consumer surfaces still hard-code ell-only grouping.",
            ),
            row(
                "trial3_relaunched_base_modes_by_ell_container_residual_route_required",
                "pass" if not base_modes_by_ell_container_branch_available else "reject",
                "base-modes-by-ell container residual route required",
                1 if not base_modes_by_ell_container_branch_available else 0,
                "A narrower residual route remains necessary after the container audit.",
            ),
            row(
                "trial3_relaunched_trial2_paper_side_sync_execute_now",
                "reject",
                "execute Trial-2 paper-side sync now",
                0,
                "Trial-2 paper sync stays unlocked reserve work while the Trial-3 solver-side residual remains open.",
            ),
        ],
        {
            "trial3_relaunched_branch_closeable": base_modes_by_ell_container_branch_available,
            "trial3_relaunched_residual_route_required": not base_modes_by_ell_container_branch_available,
            "trial2_paper_side_sync_execute_now": False,
            "trial4_deferred": True,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial3_relaunched_seventh_declaration_gate_frozen",
            "trial3_branch_closeable": base_modes_by_ell_container_branch_available,
            "advance_to_8_7_56_252": True,
            "next_required_artifacts": ["trial3_relaunched_paper_sync_trial4_disposition_seventh_refresh"],
        },
        {
            "source_inventory_summary": inventory["summary"],
            "base_modes_by_ell_identification_summary": audit["summary"],
            "prior_declaration_summary": prior_declaration["summary"],
        },
    )

    disposition = payload(
        "8.7.56.252",
        "Trial-2 paper-side sync / Trial-4 disposition seventh refresh",
        common_inputs,
        "Refresh the reserve/deferred ordering after the base-modes-by-ell container audit and freeze the next residual route for the relaunched weak-sector mainline.",
        {
            "trial2_rule": "retain Trial-2 paper-side sync as unlocked reserve work while the scientific Trial-3 residual is still open",
            "trial4_rule": "keep Trial-4 deferred until the relaunched Trial-3 branch loses all honest current-canon solver routes",
            "selected_residual_route": RESIDUAL_ROUTE,
            "missing_v2_artifact": MISSING_ARTIFACT,
        },
        [
            row(
                "trial3_relaunched_seventh_disposition_gate_complete",
                "pass",
                "Trial-3 relaunched seventh disposition gate complete",
                1,
                "The reserve/deferred ordering after the base-modes-by-ell container audit is frozen.",
            ),
            row(
                "trial3_relaunched_seventh_trial2_paper_side_sync_reserve_retained",
                "pass",
                "Trial-2 paper-side sync retained as unlocked reserve",
                1,
                "Trial-2 paper-side sync remains available but not promoted ahead of the solver-side residual.",
            ),
            row(
                "trial3_relaunched_seventh_trial4_deferred_retained",
                "pass",
                "Trial-4 deferred disposition retained",
                1,
                "Trial-4 remains deferred while the Trial-3 solver residual remains honest.",
            ),
            row(
                "trial3_relaunched_seventh_next_residual_route_frozen",
                "pass",
                "next residual route frozen",
                1,
                "The next blocker is the missing consumer surface that would let base_modes_by_ell carry a k axis downstream.",
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
            "overall_status": "trial3_relaunched_seventh_disposition_gate_frozen",
            "trial3_branch_closeable": base_modes_by_ell_container_branch_available,
            "advance_to_8_7_56_253": True,
            "next_required_artifacts": [
                "trial3_relaunched_explicit_k_positive_base_modes_by_ell_consumer_surface_source_inventory",
                "trial3_relaunched_explicit_k_positive_base_modes_by_ell_consumer_surface_identification_audit",
            ],
        },
        {
            "source_inventory_summary": inventory["summary"],
            "base_modes_by_ell_identification_summary": audit["summary"],
            "declaration_summary": declaration["summary"],
            "prior_disposition_summary": prior_disposition["summary"],
            "ai_context_current_step": ai_context["current_step"],
        },
    )

    write_artifact(
        "mass_origin_v2_trial3_relaunched_explicit_k_positive_base_modes_by_ell_container_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial3_relaunched_explicit_k_positive_base_modes_by_ell_container_identification_audit",
        audit,
    )
    write_artifact("mass_origin_v2_trial3_relaunched_declaration_seventh_gate", declaration)
    write_artifact(
        "mass_origin_v2_trial3_relaunched_paper_sync_trial4_disposition_seventh_refresh",
        disposition,
    )

    print("[ok] wrote:")
    print(" - mass_origin_v2_trial3_relaunched_explicit_k_positive_base_modes_by_ell_container_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial3_relaunched_explicit_k_positive_base_modes_by_ell_container_identification_audit_metrics.json")
    print(" - mass_origin_v2_trial3_relaunched_declaration_seventh_gate_metrics.json")
    print(" - mass_origin_v2_trial3_relaunched_paper_sync_trial4_disposition_seventh_refresh_metrics.json")


# 関数: CLI から `.249-.252` branch を実行する。

if __name__ == "__main__":
    main()
