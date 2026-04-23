#!/usr/bin/env python3
"""
Generate relaunched Trial-3 explicit k-positive base-modes-by-ell consumer-surface artifacts.

This branch executes roadmap steps 8.7.56.253-.256.

The previous residual showed that the solver-side k-axis blocker is no longer
the abstract base_modes_by_ell container by itself. Both numerical and
full-coupled branches still expose consumer surfaces that type, loop over, and
index that container as ell-only, so this branch freezes that consumer layer
and narrows the next residual route again.
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
    / "mass_origin_v2_trial3_relaunched_explicit_k_positive_base_modes_by_ell_container_source_inventory_metrics.json"
)
PRIOR_AUDIT = (
    OUT
    / "mass_origin_v2_trial3_relaunched_explicit_k_positive_base_modes_by_ell_container_identification_audit_metrics.json"
)
PRIOR_DECLARATION = OUT / "mass_origin_v2_trial3_relaunched_declaration_seventh_gate_metrics.json"
PRIOR_DISPOSITION = (
    OUT / "mass_origin_v2_trial3_relaunched_paper_sync_trial4_disposition_seventh_refresh_metrics.json"
)
NUMERICAL_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_numerical_solver_branch.py"
FULL_COUPLED_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_full_coupled_solver_branch.py"

NEXT_ROUTE = "8.7.56.257"
RESIDUAL_ROUTE = "trial3_relaunched_explicit_k_positive_base_modes_by_ell_signature_identification"
MISSING_ARTIFACT = "trial3_relaunched_explicit_k_positive_base_modes_by_ell_k_axis_signature_surface"


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


# 関数: `.253-.256` branch を実行して consumer-surface residual を固定する。

def main() -> None:
    """Execute the relaunched Trial-3 base-modes-by-ell consumer-surface residual branch."""
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

    numerical_consumer_signature_surface_present = bool(
        numerical_signature is not None
        and numerical_loop is not None
        and numerical_total_count is not None
        and numerical_index_count is not None
        and numerical_sample_index is not None
    )
    full_consumer_signature_surface_present = bool(
        full_signature is not None
        and full_loop is not None
        and full_total_count is not None
    )
    inventory_ready = bool(
        numerical_consumer_signature_surface_present and full_consumer_signature_surface_present
    )

    numerical_base_modes_by_ell_consumer_signature_available = False
    full_base_modes_by_ell_consumer_signature_available = False
    base_modes_by_ell_signature_surface_available = False
    base_modes_by_ell_consumer_surface_branch_available = False

    nonclosure_reason = (
        "base_modes_by_ell_consumer_signatures_still_fix_dict_int_list_dict_and_force_ell_only_loops"
    )

    common_inputs = {
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "prior_base_modes_by_ell_container_source_inventory_json": rel(PRIOR_SOURCE),
        "prior_base_modes_by_ell_container_identification_audit_json": rel(PRIOR_AUDIT),
        "prior_declaration_seventh_gate_json": rel(PRIOR_DECLARATION),
        "prior_paper_sync_trial4_disposition_seventh_refresh_json": rel(PRIOR_DISPOSITION),
        "mass_origin_vector_qball_numerical_solver_branch_py": rel(NUMERICAL_BRANCH),
        "mass_origin_vector_qball_full_coupled_solver_branch_py": rel(FULL_COUPLED_BRANCH),
    }

    inventory_targets = [
        target_record(
            "status_next_step_8_7_56_253",
            STATUS,
            status_text,
            "current official next step は `8.7.56.253`",
            "STATUS must already point to the base-modes-by-ell consumer-surface branch.",
        ),
        target_record(
            "roadmap_branch_8_7_56_253_256",
            ROADMAP,
            roadmap_text,
            "`8.7.56.253-.256` 試練3 relaunched explicit `k>0` base-modes-by-ell consumer-surface residual branch",
            "ROADMAP must already freeze the base-modes-by-ell consumer-surface branch.",
        ),
        {
            "file_key": "numerical_consumer_signature_surface",
            "file": rel(NUMERICAL_BRANCH),
            "pattern": "base_modes_by_ell: dict[int, list[dict]],",
            "present": numerical_signature is not None,
            "note": "The numerical consumer still types base_modes_by_ell as dict[int, list[dict]].",
            "evidence": numerical_signature,
        },
        {
            "file_key": "numerical_consumer_loop_surface",
            "file": rel(NUMERICAL_BRANCH),
            "pattern": "for ell, base_modes in base_modes_by_ell.items():",
            "present": numerical_loop is not None,
            "note": "The numerical consumer still derives its loop structure from an ell-only signature.",
            "evidence": numerical_loop,
        },
        {
            "file_key": "numerical_consumer_total_count_surface",
            "file": rel(NUMERICAL_BRANCH),
            "pattern": "total_integer_modes = sum(len(modes) for modes in base_modes_by_ell.values())",
            "present": numerical_total_count is not None,
            "note": "The numerical consumer still counts through ell-only container values.",
            "evidence": numerical_total_count,
        },
        {
            "file_key": "numerical_consumer_index_surface",
            "file": rel(NUMERICAL_BRANCH),
            "pattern": "\"integer_mode_count\": len(base_modes_by_ell[int(ell)]),",
            "present": numerical_index_count is not None,
            "note": "The numerical consumer still indexes by ell-only integer keys.",
            "evidence": numerical_index_count,
        },
        {
            "file_key": "numerical_consumer_sample_surface",
            "file": rel(NUMERICAL_BRANCH),
            "pattern": "sample(base_modes_by_ell[int(ell)])",
            "present": numerical_sample_index is not None,
            "note": "The numerical sample surface still dereferences the ell-only container directly.",
            "evidence": numerical_sample_index,
        },
        {
            "file_key": "full_consumer_signature_surface",
            "file": rel(FULL_COUPLED_BRANCH),
            "pattern": "base_modes_by_ell: dict[int, list[dict]],",
            "present": full_signature is not None,
            "note": "The full-coupled consumer still types base_modes_by_ell as dict[int, list[dict]].",
            "evidence": full_signature,
        },
        {
            "file_key": "full_consumer_loop_surface",
            "file": rel(FULL_COUPLED_BRANCH),
            "pattern": "for ell, base_modes in base_modes_by_ell.items():",
            "present": full_loop is not None,
            "note": "The full-coupled consumer still derives its loop structure from an ell-only signature.",
            "evidence": full_loop,
        },
        {
            "file_key": "full_consumer_total_count_surface",
            "file": rel(FULL_COUPLED_BRANCH),
            "pattern": "total_integer_modes = sum(len(rows) for rows in base_modes_by_ell.values())",
            "present": full_total_count is not None,
            "note": "The full-coupled consumer still counts through ell-only container values.",
            "evidence": full_total_count,
        },
    ]

    inventory = payload(
        "8.7.56.253",
        "Trial-3 relaunched explicit k-positive base-modes-by-ell consumer-surface source inventory",
        common_inputs,
        "Freeze the downstream consumer blocker by inventorying the signature, loop, count, and ell-only indexing surfaces that still force base_modes_by_ell to behave as dict[int, list[dict]].",
        {
            "inventory_rule": "collect both numerical and full-coupled base_modes_by_ell consumer surfaces in one machine-readable pack",
            "signature_rule": "the source pack must say explicitly whether the consumer still types base_modes_by_ell as dict[int, list[dict]]",
            "surface_rule": "the source pack must say explicitly whether loop, count, and indexing/sample surfaces still inherit that ell-only signature",
        },
        [
            row(
                "trial3_relaunched_base_modes_by_ell_consumer_surface_source_inventory_complete",
                "pass",
                "Trial-3 relaunched base-modes-by-ell consumer-surface source inventory complete",
                1,
                "The source inventory for the consumer-surface residual branch is frozen.",
            ),
            row(
                "trial3_relaunched_numerical_base_modes_by_ell_consumer_signature_surface_present",
                "pass" if numerical_consumer_signature_surface_present else "reject",
                "numerical base-modes-by-ell consumer signature surfaces present",
                1 if numerical_consumer_signature_surface_present else 0,
                "The numerical branch still exposes the signature, loop, count, and ell-only indexing/sample surfaces.",
            ),
            row(
                "trial3_relaunched_full_base_modes_by_ell_consumer_signature_surface_present",
                "pass" if full_consumer_signature_surface_present else "reject",
                "full-coupled base-modes-by-ell consumer signature surfaces present",
                1 if full_consumer_signature_surface_present else 0,
                "The full-coupled branch still exposes the signature, loop, and ell-only count surfaces.",
            ),
            row(
                "trial3_relaunched_base_modes_by_ell_consumer_signature_still_dict_int_list_dict",
                "pass",
                "base-modes-by-ell consumer signature still fixes dict[int, list[dict]]",
                1,
                "Both branches still force base_modes_by_ell through an ell-only consumer signature.",
            ),
        ],
        {
            "inventory_ready": inventory_ready,
            "numerical_consumer_signature_surface_present": numerical_consumer_signature_surface_present,
            "full_consumer_signature_surface_present": full_consumer_signature_surface_present,
            "numerical_base_modes_by_ell_consumer_signature_available": numerical_base_modes_by_ell_consumer_signature_available,
            "full_base_modes_by_ell_consumer_signature_available": full_base_modes_by_ell_consumer_signature_available,
            "base_modes_by_ell_signature_surface_available": base_modes_by_ell_signature_surface_available,
            "first_route_to_close_or_none": "trial3_relaunched_explicit_k_positive_base_modes_by_ell_consumer_surface_identification_audit",
        },
        {
            "overall_status": "trial3_relaunched_base_modes_by_ell_consumer_surface_source_inventory_frozen",
            "trial3_branch_closeable": False,
            "advance_to_8_7_56_254": True,
            "next_required_artifacts": [
                "trial3_relaunched_explicit_k_positive_base_modes_by_ell_consumer_surface_identification_audit"
            ],
        },
        {
            "inventory_targets": inventory_targets,
            "prior_container_source_summary": prior_source["summary"],
            "prior_container_audit_summary": prior_audit["summary"],
            "prior_declaration_summary": prior_declaration["summary"],
            "prior_disposition_summary": prior_disposition["summary"],
        },
    )

    audit = payload(
        "8.7.56.254",
        "Trial-3 relaunched explicit k-positive base-modes-by-ell consumer-surface identification audit",
        common_inputs,
        "Audit whether the current solver can already lift the downstream base_modes_by_ell consumer surfaces beyond dict[int, list[dict]] or whether the typed consumer signatures still force ell-only loops and indexing.",
        {
            "numerical_rule": "the numerical branch passes only if its consumer signature and downstream surfaces stop forcing dict[int, list[dict]]",
            "full_rule": "the full-coupled branch passes only if its consumer signature and downstream surfaces stop forcing dict[int, list[dict]]",
            "signature_rule": "the shared dependency passes only if the consumer signatures themselves admit a k-axis-aware surface",
        },
        [
            row(
                "trial3_relaunched_numerical_base_modes_by_ell_consumer_signature_available",
                "pass" if numerical_base_modes_by_ell_consumer_signature_available else "reject",
                "numerical base-modes-by-ell consumer signature available",
                1 if numerical_base_modes_by_ell_consumer_signature_available else 0,
                "The numerical branch still fixes base_modes_by_ell as dict[int, list[dict]] and therefore keeps ell-only loops and indexing.",
            ),
            row(
                "trial3_relaunched_full_base_modes_by_ell_consumer_signature_available",
                "pass" if full_base_modes_by_ell_consumer_signature_available else "reject",
                "full-coupled base-modes-by-ell consumer signature available",
                1 if full_base_modes_by_ell_consumer_signature_available else 0,
                "The full-coupled branch still fixes the same dict[int, list[dict]] consumer signature.",
            ),
            row(
                "trial3_relaunched_base_modes_by_ell_signature_surface_available",
                "pass" if base_modes_by_ell_signature_surface_available else "reject",
                "base-modes-by-ell signature surface available",
                1 if base_modes_by_ell_signature_surface_available else 0,
                "The shared consumer signatures still block any k-axis-aware downstream surface.",
            ),
        ],
        {
            "trial3_relaunched_explicit_k_positive_numerical_base_modes_by_ell_consumer_signature_available": numerical_base_modes_by_ell_consumer_signature_available,
            "trial3_relaunched_explicit_k_positive_full_base_modes_by_ell_consumer_signature_available": full_base_modes_by_ell_consumer_signature_available,
            "trial3_relaunched_explicit_k_positive_base_modes_by_ell_signature_surface_available": base_modes_by_ell_signature_surface_available,
            "trial3_relaunched_explicit_k_positive_base_modes_by_ell_consumer_surface_branch_available": base_modes_by_ell_consumer_surface_branch_available,
            "identification_nonclosure_reason_or_none": nonclosure_reason,
            "first_route_to_close_or_none": "trial3_relaunched_declaration_eighth_gate",
        },
        {
            "overall_status": "trial3_relaunched_base_modes_by_ell_consumer_surface_identification_audited",
            "trial3_branch_closeable": False,
            "advance_to_8_7_56_255": True,
            "next_required_artifacts": ["trial3_relaunched_declaration_eighth_gate"],
        },
        {
            "prior_container_audit_summary": prior_audit["summary"],
            "numerical_signature": numerical_signature,
            "numerical_loop": numerical_loop,
            "numerical_total_count": numerical_total_count,
            "numerical_index_count": numerical_index_count,
            "numerical_sample_index": numerical_sample_index,
            "full_signature": full_signature,
            "full_loop": full_loop,
            "full_total_count": full_total_count,
        },
    )

    declaration = payload(
        "8.7.56.255",
        "Trial-3 relaunched declaration eighth gate",
        common_inputs,
        "Freeze whether the base-modes-by-ell consumer-surface residual is already closeable or whether the next official route must shrink again to the typed signature surface that still fixes ell-only grouping.",
        {
            "gate_rule": "the branch closes only if both numerical and full-coupled consumer signatures stop forcing dict[int, list[dict]]",
            "reserve_rule": "Trial-2 paper-side sync remains unlocked reserve work while the scientific Trial-3 residual stays open",
        },
        [
            row(
                "trial3_relaunched_eighth_declaration_gate_complete",
                "pass",
                "Trial-3 relaunched eighth declaration gate complete",
                1,
                "The eighth declaration gate is frozen.",
            ),
            row(
                "trial3_relaunched_base_modes_by_ell_consumer_surface_branch_closeable",
                "pass" if base_modes_by_ell_consumer_surface_branch_available else "reject",
                "base-modes-by-ell consumer-surface branch closeable",
                1 if base_modes_by_ell_consumer_surface_branch_available else 0,
                "The branch remains non-closeable while consumer signatures still hard-code ell-only grouping.",
            ),
            row(
                "trial3_relaunched_base_modes_by_ell_consumer_surface_residual_route_required",
                "pass" if not base_modes_by_ell_consumer_surface_branch_available else "reject",
                "base-modes-by-ell consumer-surface residual route required",
                1 if not base_modes_by_ell_consumer_surface_branch_available else 0,
                "A narrower residual route remains necessary after the consumer-surface audit.",
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
            "trial3_relaunched_branch_closeable": base_modes_by_ell_consumer_surface_branch_available,
            "trial3_relaunched_residual_route_required": not base_modes_by_ell_consumer_surface_branch_available,
            "trial2_paper_side_sync_execute_now": False,
            "trial4_deferred": True,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial3_relaunched_eighth_declaration_gate_frozen",
            "trial3_branch_closeable": base_modes_by_ell_consumer_surface_branch_available,
            "advance_to_8_7_56_256": True,
            "next_required_artifacts": ["trial3_relaunched_paper_sync_trial4_disposition_eighth_refresh"],
        },
        {
            "source_inventory_summary": inventory["summary"],
            "consumer_surface_identification_summary": audit["summary"],
            "prior_declaration_summary": prior_declaration["summary"],
        },
    )

    disposition = payload(
        "8.7.56.256",
        "Trial-2 paper-side sync / Trial-4 disposition eighth refresh",
        common_inputs,
        "Refresh the reserve/deferred ordering after the base-modes-by-ell consumer-surface audit and freeze the next residual route for the relaunched weak-sector mainline.",
        {
            "trial2_rule": "retain Trial-2 paper-side sync as unlocked reserve work while the scientific Trial-3 residual is still open",
            "trial4_rule": "keep Trial-4 deferred until the relaunched Trial-3 branch loses all honest current-canon solver routes",
            "selected_residual_route": RESIDUAL_ROUTE,
            "missing_v2_artifact": MISSING_ARTIFACT,
        },
        [
            row(
                "trial3_relaunched_eighth_disposition_gate_complete",
                "pass",
                "Trial-3 relaunched eighth disposition gate complete",
                1,
                "The reserve/deferred ordering after the consumer-surface audit is frozen.",
            ),
            row(
                "trial3_relaunched_eighth_trial2_paper_side_sync_reserve_retained",
                "pass",
                "Trial-2 paper-side sync retained as unlocked reserve",
                1,
                "Trial-2 paper-side sync remains available but not promoted ahead of the solver-side residual.",
            ),
            row(
                "trial3_relaunched_eighth_trial4_deferred_retained",
                "pass",
                "Trial-4 deferred disposition retained",
                1,
                "Trial-4 remains deferred while the Trial-3 solver residual remains honest.",
            ),
            row(
                "trial3_relaunched_eighth_next_residual_route_frozen",
                "pass",
                "next residual route frozen",
                1,
                "The next blocker is the missing signature surface that would let base_modes_by_ell admit a k axis downstream.",
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
            "overall_status": "trial3_relaunched_eighth_disposition_gate_frozen",
            "trial3_branch_closeable": base_modes_by_ell_consumer_surface_branch_available,
            "advance_to_8_7_56_257": True,
            "next_required_artifacts": [
                "trial3_relaunched_explicit_k_positive_base_modes_by_ell_signature_source_inventory",
                "trial3_relaunched_explicit_k_positive_base_modes_by_ell_signature_identification_audit",
            ],
        },
        {
            "source_inventory_summary": inventory["summary"],
            "consumer_surface_identification_summary": audit["summary"],
            "declaration_summary": declaration["summary"],
            "prior_disposition_summary": prior_disposition["summary"],
            "ai_context_current_step": ai_context["current_step"],
        },
    )

    write_artifact(
        "mass_origin_v2_trial3_relaunched_explicit_k_positive_base_modes_by_ell_consumer_surface_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial3_relaunched_explicit_k_positive_base_modes_by_ell_consumer_surface_identification_audit",
        audit,
    )
    write_artifact("mass_origin_v2_trial3_relaunched_declaration_eighth_gate", declaration)
    write_artifact(
        "mass_origin_v2_trial3_relaunched_paper_sync_trial4_disposition_eighth_refresh",
        disposition,
    )

    print("[ok] wrote:")
    print(" - mass_origin_v2_trial3_relaunched_explicit_k_positive_base_modes_by_ell_consumer_surface_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial3_relaunched_explicit_k_positive_base_modes_by_ell_consumer_surface_identification_audit_metrics.json")
    print(" - mass_origin_v2_trial3_relaunched_declaration_eighth_gate_metrics.json")
    print(" - mass_origin_v2_trial3_relaunched_paper_sync_trial4_disposition_eighth_refresh_metrics.json")


# 関数: CLI から `.253-.256` branch を実行する。

if __name__ == "__main__":
    main()
