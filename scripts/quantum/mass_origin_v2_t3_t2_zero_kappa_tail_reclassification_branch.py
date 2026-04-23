#!/usr/bin/env python3
"""
Generate Trial-3 two-component beta-above-unity zero-kappa tail-reclassification artifacts for 8.7.56.371-.374.

This branch narrows the post-`beta_n > 1` tail blocker one step beyond the
previous decaying-tail continuation audit. The exact same-family W/Z anchors are
still numerically realized only after the full-coupled builder clips
`sqrt(1-beta_n^2)` to zero, which forces the winning states onto a branch with
`polarization_weight = 0`, `coupled_charge_factor = 1`, and
`coupled_mass_factor = 1`. The remaining question is therefore no longer a
generic tail continuation issue, but whether current canon contains any
explicit statement that reclassifies this zero-kappa clip branch as a physical
beta-above-unity tail family.
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

PRIOR_SOURCE = OUT / "mass_origin_v2_t3_t2_beta_tail_continuation_source_inventory_metrics.json"
PRIOR_AUDIT = OUT / "mass_origin_v2_t3_t2_beta_tail_continuation_audit_metrics.json"
PRIOR_GATE = OUT / "mass_origin_v2_t3_t2_beta_tail_continuation_declaration_gate_metrics.json"
PRIOR_DISPOSITION = OUT / "mass_origin_v2_t3_t2_paper_sync_trial4_disp_35th_refresh_metrics.json"
COUPLED_FREEZE = OUT / "mass_origin_vector_qball_coupled_constraint_freeze_audit_metrics.json"
FULL_SOLVER = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_full_coupled_solver_branch.py"

ANCHOR_FAMILY = {"k": 17, "ell": 1, "s": 1}
TRIAL2_RESERVE_STATE = "unlocked_reserve_retained"
NEXT_ROUTE = "8.7.56.375"


# 関数: 現在の UTC 時刻を ISO 8601 形式で返す。
def now_iso() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


# 関数: 必須入力 artifact の存在を確認する。

def req(path: Path) -> None:
    """Abort when a required input artifact is missing."""
    if not path.exists():
        raise SystemExit(f"[fail] missing required input: {path}")


# 関数: UTF-8 JSON artifact を辞書として読む。

def read_json(path: Path) -> dict:
    """Load a UTF-8 JSON artifact into a dictionary."""
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


# 関数: UTF-8 テキスト source を文字列として読む。

def read_text(path: Path) -> str:
    """Read a UTF-8 text source into memory."""
    return path.read_text(encoding="utf-8")


# 関数: 絶対パスを repo 相対表記へ変換する。

def rel(path: Path) -> str:
    """Return a repo-relative POSIX-style path string."""
    return str(path.relative_to(ROOT)).replace("\\", "/")


# 関数: source 内で最初に一致した pattern の行情報を返す。

def hit(text: str, pattern: str) -> dict | None:
    """Return the first line hit for a substring pattern, if any."""
    for line_no, line in enumerate(text.splitlines(), start=1):
        if pattern in line:
            return {"pattern": pattern, "line": line_no, "text": line.strip()}

    return None


# 関数: 共通 schema の metrics row を組み立てる。

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


# 関数: compact state payload を読みやすい形へ整える。

def compact_state(state: dict | None) -> dict | None:
    """Return a compact subset of a state dictionary for evidence payloads."""
    if state is None:
        return None

    fields = (
        "n",
        "k",
        "ell",
        "s",
        "ratio_value",
        "relative_error",
        "passes_threshold",
        "beta_n",
        "polarization_weight",
        "coupled_charge_factor",
        "coupled_mass_factor",
        "mass_ratio_to_scalar_base",
    )
    return {field: state[field] for field in fields if field in state}


# 関数: exact-anchor state が zero-kappa clip signature を満たすかを判定する。

def has_zero_kappa_clip_signature(state: dict) -> bool:
    """Return True when a state sits on the clipped zero-kappa exact-anchor branch."""
    return bool(
        float(state["beta_n"]) > 1.0
        and float(state["polarization_weight"]) == 0.0
        and float(state["coupled_charge_factor"]) == 1.0
        and float(state["coupled_mass_factor"]) == 1.0
    )


# 関数: zero-kappa tail-reclassification branch を実行する。

def main() -> None:
    """Execute the zero-kappa tail-reclassification branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        PRIOR_SOURCE,
        PRIOR_AUDIT,
        PRIOR_GATE,
        PRIOR_DISPOSITION,
        COUPLED_FREEZE,
        FULL_SOLVER,
    ):
        req(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    prior_source = read_json(PRIOR_SOURCE)
    prior_audit = read_json(PRIOR_AUDIT)
    prior_gate = read_json(PRIOR_GATE)
    prior_disposition = read_json(PRIOR_DISPOSITION)
    coupled_freeze = read_json(COUPLED_FREEZE)
    full_text = read_text(FULL_SOLVER)

    exact_best_w = prior_source["evidence"]["exact_best_w_or_none"]
    exact_best_z = prior_source["evidence"]["exact_best_z_or_none"]
    subunity_best_w = prior_source["evidence"]["subunity_best_w_or_none"]
    subunity_best_z = prior_source["evidence"]["subunity_best_z_or_none"]
    subunity_best_pair = prior_source["evidence"]["subunity_best_pair_or_none"]

    exact_w_zero_kappa_signature_confirmed = has_zero_kappa_clip_signature(exact_best_w)
    exact_z_zero_kappa_signature_confirmed = has_zero_kappa_clip_signature(exact_best_z)
    exact_anchor_zero_kappa_clip_signature_confirmed = bool(
        exact_w_zero_kappa_signature_confirmed and exact_z_zero_kappa_signature_confirmed
    )
    current_canon_has_explicit_zero_kappa_tail_statement = False
    current_canon_has_explicit_clip_branch_physical_statement = False
    zero_kappa_clip_branch_physically_admissible_under_current_canon = bool(
        exact_anchor_zero_kappa_clip_signature_confirmed
        and current_canon_has_explicit_zero_kappa_tail_statement
        and current_canon_has_explicit_clip_branch_physical_statement
    )
    branch_closeable = bool(
        zero_kappa_clip_branch_physically_admissible_under_current_canon
        and prior_audit["summary"]["zero_kappa_clip_branch_physically_reclassified_under_current_canon"]
    )

    common_inputs = {
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "mass_origin_v2_t3_t2_beta_tail_continuation_source_inventory_json": rel(PRIOR_SOURCE),
        "mass_origin_v2_t3_t2_beta_tail_continuation_audit_json": rel(PRIOR_AUDIT),
        "mass_origin_v2_t3_t2_beta_tail_continuation_declaration_gate_json": rel(PRIOR_GATE),
        "mass_origin_v2_t3_t2_paper_sync_trial4_disp_35th_refresh_json": rel(PRIOR_DISPOSITION),
        "mass_origin_vector_qball_coupled_constraint_freeze_audit_json": rel(COUPLED_FREEZE),
        "mass_origin_vector_qball_full_coupled_solver_branch_py": rel(FULL_SOLVER),
    }

    source_inventory = payload(
        "8.7.56.371",
        "Trial-3 two-component beta-above-unity zero-kappa tail-reclassification source inventory",
        common_inputs,
        "Freeze the exact-anchor zero-kappa clip signature, the absence of current-canon zero-kappa statements, and the surviving beta<=1 evidence in one source pack.",
        {
            "signature_rule": "the current exact-anchor branch is numerically defined by beta_n > 1 together with polarization_weight = 0, coupled_charge_factor = 1, and coupled_mass_factor = 1",
            "classification_rule": "if current canon contains no explicit zero-kappa tail statement, the next honest blocker is the missing statement itself rather than the numerical signature",
        },
        [
            row(
                "trial3_t2_zero_kappa_tail_reclassification_source_inventory_complete",
                "pass",
                "Trial-3 two-component beta-above-unity zero-kappa tail-reclassification source inventory complete",
                1,
                "The zero-kappa tail-reclassification source pack is frozen.",
            ),
            row(
                "trial3_t2_exact_anchor_zero_kappa_clip_signature_confirmed_in_inventory",
                "pass" if exact_anchor_zero_kappa_clip_signature_confirmed else "reject",
                "exact same-family anchor zero-kappa clip signature confirmed",
                1 if exact_anchor_zero_kappa_clip_signature_confirmed else 0,
                "The current exact anchors are both carried by the same beta>1 zero-kappa clip signature.",
            ),
            row(
                "trial3_t2_current_canon_has_explicit_zero_kappa_tail_statement_in_inventory",
                "pass" if current_canon_has_explicit_zero_kappa_tail_statement else "reject",
                "current canon has explicit zero-kappa tail statement",
                1 if current_canon_has_explicit_zero_kappa_tail_statement else 0,
                "A direct zero-kappa tail statement would be the first ingredient of an honest reclassification.",
            ),
            row(
                "trial3_t2_current_canon_has_explicit_clip_branch_physical_statement_in_inventory",
                "pass" if current_canon_has_explicit_clip_branch_physical_statement else "reject",
                "current canon has explicit clip-branch physical statement",
                1 if current_canon_has_explicit_clip_branch_physical_statement else 0,
                "A direct statement that the clipped branch is physical would also be required for honest closeout.",
            ),
            row(
                "trial3_t2_subunity_pair_preserved_in_zero_kappa_inventory",
                "pass" if subunity_best_pair["passes_threshold"] else "reject",
                "same-family beta<=1 pair preserved in zero-kappa inventory",
                1 if subunity_best_pair["passes_threshold"] else 0,
                "The near-exact beta<=1 pair remains preserved and is not the current blocker.",
            ),
            row(
                "trial3_t2_subunity_z_anchor_pass_preserved_in_zero_kappa_inventory",
                "pass" if subunity_best_z["passes_threshold"] else "reject",
                "same-family beta<=1 Z anchor pass preserved in zero-kappa inventory",
                1 if subunity_best_z["passes_threshold"] else 0,
                "The beta<=1 subset still preserves Z, so the open issue stays on the exact-anchor continuation branch.",
            ),
            row(
                "trial3_t2_subunity_w_anchor_miss_preserved_in_zero_kappa_inventory",
                "reject" if not subunity_best_w["passes_threshold"] else "pass",
                "same-family beta<=1 W miss preserved in zero-kappa inventory",
                1 if not subunity_best_w["passes_threshold"] else 0,
                "The branch still cannot close honestly inside beta<=1, so the zero-kappa exact-anchor route remains active.",
            ),
        ],
        {
            "anchor_family_or_none": dict(ANCHOR_FAMILY),
            "exact_anchor_zero_kappa_clip_signature_confirmed": exact_anchor_zero_kappa_clip_signature_confirmed,
            "current_canon_has_explicit_zero_kappa_tail_statement": current_canon_has_explicit_zero_kappa_tail_statement,
            "current_canon_has_explicit_clip_branch_physical_statement": current_canon_has_explicit_clip_branch_physical_statement,
            "subunity_pair_preserved": bool(subunity_best_pair["passes_threshold"]),
            "subunity_z_anchor_pass": bool(subunity_best_z["passes_threshold"]),
            "subunity_w_anchor_pass": bool(subunity_best_w["passes_threshold"]),
            "next_required_route": "trial3_t2_zero_kappa_tail_reclassification_audit",
        },
        {
            "overall_status": "trial3_t2_zero_kappa_tail_reclassification_inventory_frozen",
            "advance_to_8_7_56_372": True,
            "next_required_artifacts": ["trial3_t2_zero_kappa_tail_reclassification_audit"],
        },
        {
            "status_next_step_line": hit(status_text, "current official next step は `8.7.56.371`"),
            "roadmap_branch_line": hit(
                roadmap_text,
                "`8.7.56.371-.374` 試練3 two-component beta-above-unity zero-kappa tail-reclassification residual branch",
            ),
            "clip_rule_line": hit(full_text, "localization = math.sqrt(max(0.0, 1.0 - beta_n * beta_n))"),
            "zero_kappa_statement_line": hit(full_text, "zero-kappa"),
            "physical_clip_branch_statement_line": hit(full_text, "clip branch"),
            "coupled_freeze_formulas": coupled_freeze["formulas"],
            "exact_best_w_or_none": compact_state(exact_best_w),
            "exact_best_z_or_none": compact_state(exact_best_z),
            "subunity_best_w_or_none": compact_state(subunity_best_w),
            "subunity_best_z_or_none": compact_state(subunity_best_z),
            "subunity_best_pair_or_none": subunity_best_pair,
        },
    )

    audit = payload(
        "8.7.56.372",
        "Trial-3 two-component beta-above-unity zero-kappa tail-reclassification audit",
        common_inputs,
        "Audit whether current canon explicitly licenses the clipped zero-kappa branch used by the exact same-family anchors as a physical beta-above-unity tail family.",
        {
            "admissibility_rule": "the exact-anchor branch is honest only if current canon explicitly states that the zero-kappa clipped branch is a physical tail continuation",
            "residual_rule": "if that statement is absent, the blocker narrows from generic reclassification to the missing zero-kappa tail statement itself",
        },
        [
            row(
                "trial3_t2_zero_kappa_tail_reclassification_audit_complete",
                "pass",
                "Trial-3 two-component beta-above-unity zero-kappa tail-reclassification audit complete",
                1,
                "The zero-kappa tail-reclassification audit is frozen.",
            ),
            row(
                "trial3_t2_exact_anchor_zero_kappa_clip_signature_confirmed_audit",
                "pass" if exact_anchor_zero_kappa_clip_signature_confirmed else "reject",
                "exact same-family anchor zero-kappa clip signature confirmed",
                1 if exact_anchor_zero_kappa_clip_signature_confirmed else 0,
                "The numerical signature itself is not in doubt in the current branch.",
            ),
            row(
                "trial3_t2_current_canon_has_explicit_zero_kappa_tail_statement",
                "pass" if current_canon_has_explicit_zero_kappa_tail_statement else "reject",
                "current canon has explicit zero-kappa tail statement",
                1 if current_canon_has_explicit_zero_kappa_tail_statement else 0,
                "This is the direct textual prerequisite for an honest zero-kappa reclassification.",
            ),
            row(
                "trial3_t2_current_canon_has_explicit_clip_branch_physical_statement",
                "pass" if current_canon_has_explicit_clip_branch_physical_statement else "reject",
                "current canon has explicit clip-branch physical statement",
                1 if current_canon_has_explicit_clip_branch_physical_statement else 0,
                "The current numerics are only physically admissible if current canon explicitly treats the clip branch as physical.",
            ),
            row(
                "trial3_t2_zero_kappa_clip_branch_physically_admissible_under_current_canon",
                "pass" if zero_kappa_clip_branch_physically_admissible_under_current_canon else "reject",
                "zero-kappa clip branch physically admissible under current canon",
                1 if zero_kappa_clip_branch_physically_admissible_under_current_canon else 0,
                "Without this condition, the exact-anchor gain remains numerical only.",
            ),
        ],
        {
            "anchor_family_or_none": dict(ANCHOR_FAMILY),
            "exact_anchor_zero_kappa_clip_signature_confirmed": exact_anchor_zero_kappa_clip_signature_confirmed,
            "current_canon_has_explicit_zero_kappa_tail_statement": current_canon_has_explicit_zero_kappa_tail_statement,
            "current_canon_has_explicit_clip_branch_physical_statement": current_canon_has_explicit_clip_branch_physical_statement,
            "zero_kappa_clip_branch_physically_admissible_under_current_canon": zero_kappa_clip_branch_physically_admissible_under_current_canon,
            "next_required_route": "trial3_t2_zero_kappa_tail_reclassification_declaration_eleventh_gate",
        },
        {
            "overall_status": "trial3_t2_zero_kappa_tail_reclassification_audited",
            "advance_to_8_7_56_373": True,
            "next_required_artifacts": ["trial3_t2_zero_kappa_tail_reclassification_declaration_eleventh_gate"],
        },
        {
            "prior_audit_summary": prior_audit["summary"],
            "exact_anchor_signature": {
                "exact_w_zero_kappa_signature_confirmed": exact_w_zero_kappa_signature_confirmed,
                "exact_z_zero_kappa_signature_confirmed": exact_z_zero_kappa_signature_confirmed,
            },
            "exact_best_w_or_none": compact_state(exact_best_w),
            "exact_best_z_or_none": compact_state(exact_best_z),
        },
    )

    declaration_gate = payload(
        "8.7.56.373",
        "Trial-3 two-component declaration eleventh gate",
        common_inputs,
        "Freeze whether the current exact-anchor gain closes Trial-3, or whether the honest next blocker is the missing zero-kappa tail statement for the clipped beta>1 branch.",
        {
            "closeout_rule": "close Trial-3 only if current canon explicitly licenses the clipped zero-kappa exact-anchor branch physically",
            "residual_rule": "if the signature is confirmed but the statement is absent, the next blocker is the missing zero-kappa tail statement itself",
        },
        [
            row(
                "trial3_t2_declaration_eleventh_gate_complete",
                "pass",
                "Trial-3 two-component declaration eleventh gate complete",
                1,
                "The eleventh gate is frozen.",
            ),
            row(
                "trial3_t2_branch_closeable_eleventh_gate",
                "pass" if branch_closeable else "reject",
                "two-component branch closeable after zero-kappa tail-reclassification audit",
                1 if branch_closeable else 0,
                "The branch closes only if current canon explicitly licenses the clipped zero-kappa branch physically.",
            ),
            row(
                "trial3_t2_residual_route_required_eleventh_gate",
                "reject" if branch_closeable else "pass",
                "two-component residual route still required after zero-kappa tail-reclassification audit",
                0 if branch_closeable else 1,
                "A residual route remains required while the zero-kappa tail statement is absent.",
            ),
        ],
        {
            "solver_range_blocker_removed": True,
            "exact_anchor_zero_kappa_clip_signature_confirmed": exact_anchor_zero_kappa_clip_signature_confirmed,
            "zero_kappa_clip_branch_physically_admissible_under_current_canon": zero_kappa_clip_branch_physically_admissible_under_current_canon,
            "trial3_current_branch_closeable": branch_closeable,
            "selected_residual_route": (
                None
                if branch_closeable
                else "trial3_two_component_beta_above_unity_zero_kappa_tail_statement_identification"
            ),
            "missing_v2_artifact": (
                None
                if branch_closeable
                else "trial3_two_component_beta_above_unity_zero_kappa_tail_statement_pack"
            ),
            "recommended_next_route_or_none": None if branch_closeable else NEXT_ROUTE,
        },
        {
            "overall_status": "trial3_t2_declaration_eleventh_gate_frozen",
            "trial3_branch_closeable": branch_closeable,
            "advance_to_8_7_56_374": True,
            "next_required_artifacts": [] if branch_closeable else [NEXT_ROUTE],
        },
        {
            "audit_summary": audit["summary"],
            "prior_gate_summary": prior_gate["summary"],
            "current_ai_context_step": ai_context["current_step"],
        },
    )

    disposition = payload(
        "8.7.56.374",
        "Trial-2 paper-side sync / Trial-4 disposition thirty-sixth refresh",
        common_inputs,
        "Refresh the reserve/deferred ordering after the zero-kappa tail-reclassification audit narrows the blocker to the missing zero-kappa tail statement for the clipped beta>1 branch.",
        {
            "trial2_rule": "Trial-2 paper-side sync remains unlocked reserve retained while Trial-3 still has an honest current-canon residual route",
            "trial4_rule": "Trial-4 remains deferred while the two-component Trial-3 route is still scientifically live",
        },
        [
            row(
                "trial3_t2_trial2_trial4_thirty_sixth_refresh_complete",
                "pass",
                "Trial-2 paper-side sync / Trial-4 disposition thirty-sixth refresh complete",
                1,
                "The reserve/deferred ordering is refreshed after the zero-kappa tail-reclassification audit.",
            ),
            row(
                "trial3_t2_trial2_reserve_retained_thirty_sixth_refresh",
                "pass",
                "Trial-2 paper-side sync reserve retained",
                1,
                "Trial-2 paper sync remains reserve work while the Trial-3 zero-kappa-tail statement route is still open.",
            ),
            row(
                "trial3_t2_trial4_deferred_retained_thirty_sixth_refresh",
                "pass",
                "Trial-4 deferred retained",
                1,
                "Trial-4 stays deferred while the two-component Trial-3 route remains live.",
            ),
        ],
        {
            "selected_residual_route": declaration_gate["summary"]["selected_residual_route"],
            "missing_v2_artifact": declaration_gate["summary"]["missing_v2_artifact"],
            "trial2_paper_side_sync_state": TRIAL2_RESERVE_STATE,
            "trial4_deferred": True,
            "recommended_next_route_or_none": declaration_gate["summary"]["recommended_next_route_or_none"],
        },
        {
            "overall_status": "trial3_t2_trial2_trial4_thirty_sixth_refresh_frozen",
            "trial3_branch_closeable": branch_closeable,
            "advance_to_next_branch": not branch_closeable,
            "next_required_artifacts": [] if branch_closeable else [NEXT_ROUTE],
        },
        {
            "declaration_summary": declaration_gate["summary"],
            "prior_disposition_summary": prior_disposition["summary"],
        },
    )

    write_artifact("mass_origin_v2_t3_t2_zero_kappa_tail_reclassification_source_inventory", source_inventory)
    write_artifact("mass_origin_v2_t3_t2_zero_kappa_tail_reclassification_audit", audit)
    write_artifact("mass_origin_v2_t3_t2_zero_kappa_tail_reclassification_declaration_gate", declaration_gate)
    write_artifact("mass_origin_v2_t3_t2_paper_sync_trial4_disp_36th_refresh", disposition)

    print("[done] Trial-3 two-component zero-kappa tail-reclassification artifacts written:")
    print(" - mass_origin_v2_t3_t2_zero_kappa_tail_reclassification_source_inventory_metrics.json")
    print(" - mass_origin_v2_t3_t2_zero_kappa_tail_reclassification_audit_metrics.json")
    print(" - mass_origin_v2_t3_t2_zero_kappa_tail_reclassification_declaration_gate_metrics.json")
    print(" - mass_origin_v2_t3_t2_paper_sync_trial4_disp_36th_refresh_metrics.json")


# 関数: CLI から branch を実行する。

def run_cli() -> None:
    """CLI entry point for the Trial-3 zero-kappa tail-reclassification branch."""
    main()


if __name__ == "__main__":
    run_cli()
