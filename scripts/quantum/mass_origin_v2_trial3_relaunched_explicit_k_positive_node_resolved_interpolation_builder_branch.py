#!/usr/bin/env python3
"""
Generate relaunched Trial-3 explicit k-positive node-resolved interpolation-builder artifacts.

This branch executes roadmap steps 8.7.56.237-.240.

The previous residual showed that the solver-side blocker is no longer the
whole integer-mode builder, but the narrower interpolation layer: the current
interpolator still accepts only `(scan_rows, ell)`, emits rows with `k=0`, and
the downstream trial-state rows still bake `k=0` into both row content and ids.
This branch freezes that narrower blocker and chooses the next residual route.
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
    / "mass_origin_v2_trial3_relaunched_explicit_k_positive_integer_mode_builder_source_inventory_metrics.json"
)
PRIOR_AUDIT = (
    OUT
    / "mass_origin_v2_trial3_relaunched_explicit_k_positive_integer_mode_builder_identification_audit_metrics.json"
)
PRIOR_DECLARATION = OUT / "mass_origin_v2_trial3_relaunched_declaration_third_gate_metrics.json"
PRIOR_DISPOSITION = (
    OUT / "mass_origin_v2_trial3_relaunched_paper_sync_trial4_disposition_third_refresh_metrics.json"
)
NUMERICAL_BRANCH = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_numerical_solver_branch.py"

NEXT_ROUTE = "8.7.56.241"
RESIDUAL_ROUTE = "trial3_relaunched_explicit_k_positive_interpolation_signature_identification"
MISSING_ARTIFACT = "trial3_relaunched_explicit_k_positive_interpolation_signature_extension"


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


# 関数: パターンを含む全行を返す。

def hits(text: str, pattern: str) -> list[dict]:
    """Return all source lines that contain the requested pattern."""
    found = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            found.append({"pattern": pattern, "line": line_no, "text": line.strip()})

    return found


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


# 関数: `.237-.240` branch を実行して machine-readable residual を固定する。

def main() -> None:
    """Execute the relaunched Trial-3 node-resolved interpolation-builder residual branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        PRIOR_SOURCE,
        PRIOR_AUDIT,
        PRIOR_DECLARATION,
        PRIOR_DISPOSITION,
        NUMERICAL_BRANCH,
    ):
        req(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    numerical_text = read_text(NUMERICAL_BRANCH)

    prior_source = read_json(PRIOR_SOURCE)
    prior_audit = read_json(PRIOR_AUDIT)
    prior_declaration = read_json(PRIOR_DECLARATION)
    prior_disposition = read_json(PRIOR_DISPOSITION)

    interpolation_signature = hit(
        numerical_text, "def interpolate_integer_modes(scan_rows: list[dict], ell: int)"
    )
    interpolation_signature_text = interpolation_signature["text"] if interpolation_signature else ""
    interpolation_signature_has_k_argument = ", k" in interpolation_signature_text
    interpolation_k_zero_hits = [
        entry for entry in hits(numerical_text, '"k": 0,') if 276 <= int(entry["line"]) <= 320
    ]
    trial_state_signature = hit(
        numerical_text, "def build_trial_state_rows(scalar_modes: list[dict], sector_rows: list[dict])"
    )
    trial_state_k_zero_hits = [
        entry for entry in hits(numerical_text, '"k": 0,') if 325 <= int(entry["line"]) <= 345
    ]
    zero_node_identifier_hit = hit(numerical_text, 'trial_state_id": f"M_({n},0,{ell},{s})"')

    inventory_targets = [
        target_record(
            "status_next_step_8_7_56_237",
            STATUS,
            status_text,
            "current official next step は `8.7.56.237`",
            "STATUS must already point to the node-resolved interpolation-builder branch.",
        ),
        target_record(
            "roadmap_branch_8_7_56_237_240",
            ROADMAP,
            roadmap_text,
            "`8.7.56.237-.240` 試練3 relaunched explicit `k>0` node-resolved interpolation-builder residual branch",
            "ROADMAP must already freeze the node-resolved interpolation-builder branch.",
        ),
        {
            "file_key": "interpolation_signature",
            "file": rel(NUMERICAL_BRANCH),
            "pattern": "def interpolate_integer_modes(scan_rows: list[dict], ell: int)",
            "present": interpolation_signature is not None,
            "note": "The current interpolation entry point must remain visible in the numerical branch.",
            "evidence": interpolation_signature,
        },
        {
            "file_key": "interpolation_signature_without_k_argument",
            "file": rel(NUMERICAL_BRANCH),
            "pattern": "signature excludes explicit k argument",
            "present": not interpolation_signature_has_k_argument,
            "note": "The signature still carries only scan_rows and ell, with no node-axis argument.",
            "evidence": interpolation_signature,
        },
        {
            "file_key": "interpolation_output_rows_k_zero_hardcode",
            "file": rel(NUMERICAL_BRANCH),
            "pattern": "\"k\": 0,",
            "present": len(interpolation_k_zero_hits) > 0,
            "note": "The interpolated base-mode rows still hardcode k=0.",
            "evidence": interpolation_k_zero_hits,
        },
        {
            "file_key": "trial_state_builder_signature",
            "file": rel(NUMERICAL_BRANCH),
            "pattern": "def build_trial_state_rows(scalar_modes: list[dict], sector_rows: list[dict])",
            "present": trial_state_signature is not None,
            "note": "The trial-state row builder must remain visible because it propagates the node label downstream.",
            "evidence": trial_state_signature,
        },
        {
            "file_key": "trial_state_rows_k_zero_hardcode",
            "file": rel(NUMERICAL_BRANCH),
            "pattern": "\"k\": 0,",
            "present": len(trial_state_k_zero_hits) > 0,
            "note": "The downstream trial-state rows still hardcode k=0.",
            "evidence": trial_state_k_zero_hits,
        },
        {
            "file_key": "trial_state_identifier_zero_node_pattern",
            "file": rel(NUMERICAL_BRANCH),
            "pattern": "M_(n,0,ell,s)",
            "present": zero_node_identifier_hit is not None,
            "note": "The trial-state id still bakes the zero-node label into the canonical identifier.",
            "evidence": zero_node_identifier_hit,
        },
    ]
    inventory_ready = all(bool(item["present"]) for item in inventory_targets)

    interpolation_signature_extension_available = bool(
        interpolation_signature is not None and interpolation_signature_has_k_argument
    )
    node_resolved_row_emitter_available = bool(
        not interpolation_k_zero_hits
        and not trial_state_k_zero_hits
        and zero_node_identifier_hit is None
    )
    node_resolved_interpolation_builder_available = bool(
        interpolation_signature_extension_available and node_resolved_row_emitter_available
    )

    nonclosure_reason = (
        "interpolate_integer_modes_signature_still_excludes_explicit_k_argument"
        if not interpolation_signature_extension_available
        else "interpolate_integer_modes_and_trial_state_rows_still_emit_only_zero_node_labels"
        if not node_resolved_row_emitter_available
        else None
    )

    common_inputs = {
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "prior_builder_source_inventory_json": rel(PRIOR_SOURCE),
        "prior_builder_identification_audit_json": rel(PRIOR_AUDIT),
        "prior_declaration_third_gate_json": rel(PRIOR_DECLARATION),
        "prior_paper_sync_trial4_disposition_third_refresh_json": rel(PRIOR_DISPOSITION),
        "mass_origin_vector_qball_numerical_solver_branch_py": rel(NUMERICAL_BRANCH),
    }

    inventory = payload(
        "8.7.56.237",
        "Trial-3 relaunched explicit k-positive node-resolved interpolation-builder source inventory",
        common_inputs,
        "Freeze the narrowed solver-side blocker by inventorying the interpolation signature, emitted mode rows, and downstream trial-state labels that still assume k=0.",
        {
            "inventory_rule": "collect the interpolation signature, its emitted rows, and the downstream trial-state identifiers in one machine-readable pack",
            "signature_rule": "the source pack must say explicitly whether interpolate_integer_modes already exposes a node-axis argument",
            "row_emitter_rule": "the source pack must say explicitly whether interpolated rows and downstream trial-state rows still hardcode k=0",
        },
        [
            row(
                "trial3_relaunched_node_resolved_interpolation_builder_source_inventory_complete",
                "pass",
                "Trial-3 relaunched node-resolved interpolation-builder source inventory complete",
                1,
                "The narrowed interpolation-builder inventory is frozen.",
            ),
            row(
                "trial3_relaunched_interpolation_signature_has_k_argument",
                "pass" if interpolation_signature_has_k_argument else "reject",
                "interpolation signature has explicit k argument",
                1 if interpolation_signature_has_k_argument else 0,
                "The current signature still lacks the node-axis argument required by k-positive mode construction.",
            ),
            row(
                "trial3_relaunched_interpolation_output_rows_k_zero_hardcode_present",
                "pass" if len(interpolation_k_zero_hits) > 0 else "reject",
                "interpolation output rows still hardcode k=0",
                1 if len(interpolation_k_zero_hits) > 0 else 0,
                "The emitted base-mode rows still freeze k at zero.",
            ),
            row(
                "trial3_relaunched_trial_state_identifier_zero_node_pattern_present",
                "pass" if zero_node_identifier_hit is not None else "reject",
                "trial-state identifiers still bake the zero-node label",
                1 if zero_node_identifier_hit is not None else 0,
                "The downstream row id still uses the M_(n,0,ell,s) pattern.",
            ),
        ],
        {
            "inventory_ready": inventory_ready,
            "interpolation_signature_present": interpolation_signature is not None,
            "interpolation_signature_has_k_argument": interpolation_signature_has_k_argument,
            "interpolation_output_rows_k_zero_hardcode_present": bool(interpolation_k_zero_hits),
            "trial_state_row_k_zero_hardcode_present": bool(trial_state_k_zero_hits),
            "trial_state_identifier_zero_node_pattern_present": zero_node_identifier_hit is not None,
            "first_route_to_close_or_none": "trial3_relaunched_explicit_k_positive_node_resolved_interpolation_builder_identification_audit",
        },
        {
            "overall_status": "trial3_relaunched_node_resolved_interpolation_builder_source_inventory_frozen",
            "trial3_branch_closeable": False,
            "advance_to_8_7_56_238": True,
            "next_required_artifacts": [
                "trial3_relaunched_explicit_k_positive_node_resolved_interpolation_builder_identification_audit"
            ],
        },
        {
            "inventory_targets": inventory_targets,
            "prior_builder_source_summary": prior_source["summary"],
            "prior_builder_audit_summary": prior_audit["summary"],
            "prior_declaration_summary": prior_declaration["summary"],
            "prior_disposition_summary": prior_disposition["summary"],
        },
    )

    audit = payload(
        "8.7.56.238",
        "Trial-3 relaunched explicit k-positive node-resolved interpolation-builder identification audit",
        common_inputs,
        "Audit whether the narrowed interpolation builder is blocked primarily by the missing k-axis signature or by the downstream zero-node row emitter that still propagates k=0 labels.",
        {
            "signature_audit_rule": "the interpolation-signature layer passes only if interpolate_integer_modes exposes an explicit node-axis argument",
            "row_emitter_audit_rule": "the row-emitter layer passes only if both interpolated rows and downstream trial-state rows stop emitting k=0 labels",
            "builder_audit_rule": "the node-resolved interpolation builder passes only if both the signature layer and the row-emitter layer pass together",
        },
        [
            row(
                "trial3_relaunched_interpolation_signature_extension_available",
                "pass" if interpolation_signature_extension_available else "reject",
                "explicit k-positive interpolation signature extension available",
                1 if interpolation_signature_extension_available else 0,
                "The current signature still excludes the explicit node-axis argument.",
            ),
            row(
                "trial3_relaunched_node_resolved_row_emitter_available",
                "pass" if node_resolved_row_emitter_available else "reject",
                "node-resolved row emitter available",
                1 if node_resolved_row_emitter_available else 0,
                "The current row emitters still output zero-node rows and identifiers.",
            ),
            row(
                "trial3_relaunched_node_resolved_interpolation_builder_available",
                "pass" if node_resolved_interpolation_builder_available else "reject",
                "node-resolved k-positive interpolation builder available",
                1 if node_resolved_interpolation_builder_available else 0,
                "The current solver still cannot emit honest k-positive node-resolved mode rows.",
            ),
        ],
        {
            "trial3_relaunched_explicit_k_positive_node_resolved_interpolation_builder_available": node_resolved_interpolation_builder_available,
            "trial3_relaunched_explicit_k_positive_interpolation_signature_extension_available": interpolation_signature_extension_available,
            "trial3_relaunched_explicit_k_positive_node_resolved_row_emitter_available": node_resolved_row_emitter_available,
            "identification_nonclosure_reason_or_none": nonclosure_reason,
            "first_route_to_close_or_none": "trial3_relaunched_declaration_fourth_gate",
        },
        {
            "overall_status": "trial3_relaunched_node_resolved_interpolation_builder_identification_audited",
            "trial3_branch_closeable": False,
            "advance_to_8_7_56_239": True,
            "next_required_artifacts": ["trial3_relaunched_declaration_fourth_gate"],
        },
        {
            "prior_builder_audit_summary": prior_audit["summary"],
            "interpolation_signature": interpolation_signature,
            "interpolation_k_zero_hits": interpolation_k_zero_hits,
            "trial_state_signature": trial_state_signature,
            "trial_state_k_zero_hits": trial_state_k_zero_hits,
            "zero_node_identifier_hit": zero_node_identifier_hit,
        },
    )

    declaration = payload(
        "8.7.56.239",
        "Trial-3 relaunched declaration fourth gate",
        common_inputs,
        "Freeze whether the narrowed interpolation-builder residual is already closeable or whether the next official route must shrink again to the missing interpolation-signature artifact.",
        {
            "gate_rule": "the branch closes only if the explicit k-positive interpolation signature and the node-resolved row emitter both exist",
            "reserve_rule": "Trial-2 paper-side sync remains unlocked reserve work while the scientific Trial-3 residual stays open",
        },
        [
            row(
                "trial3_relaunched_fourth_declaration_gate_complete",
                "pass",
                "Trial-3 relaunched fourth declaration gate complete",
                1,
                "The fourth declaration gate is frozen.",
            ),
            row(
                "trial3_relaunched_node_resolved_interpolation_builder_branch_closeable",
                "pass" if node_resolved_interpolation_builder_available else "reject",
                "node-resolved interpolation-builder branch closeable",
                1 if node_resolved_interpolation_builder_available else 0,
                "The branch remains non-closeable while the explicit k-positive interpolation signature is absent.",
            ),
            row(
                "trial3_relaunched_node_resolved_interpolation_builder_residual_route_required",
                "pass" if not node_resolved_interpolation_builder_available else "reject",
                "node-resolved interpolation-builder residual route required",
                1 if not node_resolved_interpolation_builder_available else 0,
                "A narrower residual route remains necessary after the identification audit.",
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
            "trial3_relaunched_branch_closeable": node_resolved_interpolation_builder_available,
            "trial3_relaunched_residual_route_required": not node_resolved_interpolation_builder_available,
            "trial2_paper_side_sync_execute_now": False,
            "trial4_deferred": True,
            "recommended_next_route_or_none": NEXT_ROUTE,
        },
        {
            "overall_status": "trial3_relaunched_fourth_declaration_gate_frozen",
            "trial3_branch_closeable": node_resolved_interpolation_builder_available,
            "advance_to_8_7_56_240": True,
            "next_required_artifacts": ["trial3_relaunched_paper_sync_trial4_disposition_fourth_refresh"],
        },
        {
            "source_inventory_summary": inventory["summary"],
            "builder_identification_summary": audit["summary"],
            "prior_declaration_summary": prior_declaration["summary"],
        },
    )

    disposition = payload(
        "8.7.56.240",
        "Trial-2 paper-side sync / Trial-4 disposition fourth refresh",
        common_inputs,
        "Refresh the reserve/deferred ordering after the narrowed interpolation-builder audit and freeze the next residual route for the relaunched weak-sector mainline.",
        {
            "trial2_rule": "retain Trial-2 paper-side sync as unlocked reserve work while the scientific Trial-3 residual is still open",
            "trial4_rule": "keep Trial-4 deferred until the relaunched Trial-3 branch loses all honest current-canon solver routes",
            "selected_residual_route": RESIDUAL_ROUTE,
            "missing_v2_artifact": MISSING_ARTIFACT,
        },
        [
            row(
                "trial3_relaunched_fourth_disposition_gate_complete",
                "pass",
                "Trial-3 relaunched fourth disposition gate complete",
                1,
                "The reserve/deferred ordering after the narrowed audit is frozen.",
            ),
            row(
                "trial3_relaunched_fourth_trial2_paper_side_sync_reserve_retained",
                "pass",
                "Trial-2 paper-side sync retained as unlocked reserve",
                1,
                "Trial-2 paper-side sync remains available but not promoted ahead of the solver-side residual.",
            ),
            row(
                "trial3_relaunched_fourth_trial4_deferred_retained",
                "pass",
                "Trial-4 deferred disposition retained",
                1,
                "Trial-4 remains deferred while the Trial-3 solver residual remains honest.",
            ),
            row(
                "trial3_relaunched_fourth_next_residual_route_frozen",
                "pass",
                "next residual route frozen",
                1,
                "The next blocker is the missing interpolation-signature extension required by k-positive mode construction.",
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
            "overall_status": "trial3_relaunched_fourth_disposition_gate_frozen",
            "trial3_branch_closeable": node_resolved_interpolation_builder_available,
            "advance_to_8_7_56_241": True,
            "next_required_artifacts": [
                "trial3_relaunched_explicit_k_positive_interpolation_signature_source_inventory",
                "trial3_relaunched_explicit_k_positive_interpolation_signature_identification_audit",
            ],
        },
        {
            "source_inventory_summary": inventory["summary"],
            "builder_identification_summary": audit["summary"],
            "declaration_summary": declaration["summary"],
            "prior_disposition_summary": prior_disposition["summary"],
            "ai_context_current_step": ai_context["current_step"],
        },
    )

    write_artifact(
        "mass_origin_v2_trial3_relaunched_explicit_k_positive_node_resolved_interpolation_builder_source_inventory",
        inventory,
    )
    write_artifact(
        "mass_origin_v2_trial3_relaunched_explicit_k_positive_node_resolved_interpolation_builder_identification_audit",
        audit,
    )
    write_artifact("mass_origin_v2_trial3_relaunched_declaration_fourth_gate", declaration)
    write_artifact(
        "mass_origin_v2_trial3_relaunched_paper_sync_trial4_disposition_fourth_refresh",
        disposition,
    )

    print("[ok] wrote:")
    print(" - mass_origin_v2_trial3_relaunched_explicit_k_positive_node_resolved_interpolation_builder_source_inventory_metrics.json")
    print(" - mass_origin_v2_trial3_relaunched_explicit_k_positive_node_resolved_interpolation_builder_identification_audit_metrics.json")
    print(" - mass_origin_v2_trial3_relaunched_declaration_fourth_gate_metrics.json")
    print(" - mass_origin_v2_trial3_relaunched_paper_sync_trial4_disposition_fourth_refresh_metrics.json")


# 関数: CLI から `.237-.240` branch を実行する。

if __name__ == "__main__":
    main()
