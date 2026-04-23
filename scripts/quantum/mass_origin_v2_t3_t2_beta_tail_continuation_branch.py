#!/usr/bin/env python3
"""
Generate Trial-3 two-component beta-above-unity decaying-tail continuation artifacts for 8.7.56.367-.370.

This branch narrows the post-`beta_n > 1` tail blocker one level deeper than
the previous audit. The frozen boundary rule still demands an imaginary
localization scale `kappa` once the exact same-family W/Z anchors cross unity,
but the current builder does not implement such a continuation explicitly.
Instead, it clips `sqrt(1-beta_n^2)` to zero, which forces the exact-anchor
states onto a zero-polarization, unit-factor branch. The remaining question is
therefore no longer a generic decaying-tail continuation, but whether current
canon explicitly reclassifies that clipped zero-kappa branch as a physical
beta-above-unity tail continuation.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"

BETA_TAIL_SOURCE = OUT / "mass_origin_v2_t3_t2_beta_tail_source_inventory_metrics.json"
BETA_TAIL_AUDIT = OUT / "mass_origin_v2_t3_t2_beta_tail_audit_metrics.json"
BETA_TAIL_GATE = OUT / "mass_origin_v2_t3_t2_beta_tail_declaration_gate_metrics.json"
BETA_TAIL_DISPOSITION = OUT / "mass_origin_v2_t3_t2_paper_sync_trial4_disp_34th_refresh_metrics.json"
COUPLED_FREEZE = OUT / "mass_origin_vector_qball_coupled_constraint_freeze_audit_metrics.json"
FULL_SOLVER = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_full_coupled_solver_branch.py"

ANCHOR_FAMILY = {"k": 17, "ell": 1, "s": 1}
TRIAL2_RESERVE_STATE = "unlocked_reserve_retained"
NEXT_ROUTE = "8.7.56.371"


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


# 関数: local Python module を動的 import する。

def load_module(path: Path, module_name: str):
    """Load a local Python module from a filesystem path."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[fail] unable to import module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


# 関数: exact-anchor state が clip-equal zero-kappa branch かを判定する。

def is_zero_kappa_clip_equivalent(state: dict) -> bool:
    """Return True when the state sits exactly on the builder's clipped zero-kappa branch."""
    return bool(
        float(state["polarization_weight"]) == 0.0
        and float(state["coupled_charge_factor"]) == 1.0
        and float(state["coupled_mass_factor"]) == 1.0
        and float(state["beta_n"]) > 1.0
    )


# 関数: beta-tail continuation branch を実行する。

def main() -> None:
    """Execute the beta-above-unity decaying-tail continuation branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        BETA_TAIL_SOURCE,
        BETA_TAIL_AUDIT,
        BETA_TAIL_GATE,
        BETA_TAIL_DISPOSITION,
        COUPLED_FREEZE,
        FULL_SOLVER,
    ):
        req(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    prior_source = read_json(BETA_TAIL_SOURCE)
    prior_audit = read_json(BETA_TAIL_AUDIT)
    prior_gate = read_json(BETA_TAIL_GATE)
    prior_disposition = read_json(BETA_TAIL_DISPOSITION)
    coupled_freeze = read_json(COUPLED_FREEZE)
    full_text = read_text(FULL_SOLVER)
    load_module(FULL_SOLVER, "trial3_t2_beta_tail_continuation_full_solver")

    exact_best_w = prior_source["evidence"]["exact_best_w_or_none"]
    exact_best_z = prior_source["evidence"]["exact_best_z_or_none"]
    subunity_best_w = prior_source["evidence"]["subunity_best_w_or_none"]
    subunity_best_z = prior_source["evidence"]["subunity_best_z_or_none"]
    subunity_best_pair = prior_source["evidence"]["subunity_best_pair_or_none"]
    subunity_pair_preserved = bool(subunity_best_pair["passes_threshold"])
    subunity_w_anchor_pass = bool(subunity_best_w["passes_threshold"])

    exact_w_zero_kappa_clip_equivalent = is_zero_kappa_clip_equivalent(exact_best_w)
    exact_z_zero_kappa_clip_equivalent = is_zero_kappa_clip_equivalent(exact_best_z)
    exact_anchor_support_currently_realized_only_as_zero_kappa_clip_branch = bool(
        exact_w_zero_kappa_clip_equivalent and exact_z_zero_kappa_clip_equivalent
    )
    current_canon_has_explicit_zero_kappa_tail_reclassification = False
    current_canon_has_explicit_imaginary_kappa_tail_interpretation = False
    beta_above_unity_tail_continuation_available_under_current_canon = False
    zero_kappa_clip_branch_physically_reclassified_under_current_canon = bool(
        exact_anchor_support_currently_realized_only_as_zero_kappa_clip_branch
        and current_canon_has_explicit_zero_kappa_tail_reclassification
    )
    branch_closeable = bool(
        beta_above_unity_tail_continuation_available_under_current_canon
        and prior_audit["summary"]["beta_above_unity_decaying_tail_available_under_current_canon"]
    )

    common_inputs = {
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "mass_origin_v2_t3_t2_beta_tail_source_inventory_json": rel(BETA_TAIL_SOURCE),
        "mass_origin_v2_t3_t2_beta_tail_audit_json": rel(BETA_TAIL_AUDIT),
        "mass_origin_v2_t3_t2_beta_tail_declaration_gate_json": rel(BETA_TAIL_GATE),
        "mass_origin_v2_t3_t2_paper_sync_trial4_disp_34th_refresh_json": rel(BETA_TAIL_DISPOSITION),
        "mass_origin_vector_qball_coupled_constraint_freeze_audit_json": rel(COUPLED_FREEZE),
        "mass_origin_vector_qball_full_coupled_solver_branch_py": rel(FULL_SOLVER),
    }

    source_inventory = payload(
        "8.7.56.367",
        "Trial-3 two-component beta-above-unity decaying-tail continuation source inventory",
        common_inputs,
        "Freeze the imaginary-kappa requirement, the clipped zero-kappa realization of the exact anchors, the absence of explicit current-canon tail statements, and the surviving beta<=1 evidence in one continuation source pack.",
        {
            "frozen_rule": "the frozen canon defines localized support by a real decaying tail with kappa = sqrt(1-beta^2)",
            "clip_rule": "the current builder realizes beta>1 states only through the clipped substitution sqrt(max(0,1-beta^2)) = 0, which zeroes polarization-weight corrections",
        },
        [
            row(
                "trial3_t2_beta_tail_continuation_source_inventory_complete",
                "pass",
                "Trial-3 two-component beta-above-unity decaying-tail continuation source inventory complete",
                1,
                "The decaying-tail continuation source pack is frozen.",
            ),
            row(
                "trial3_t2_exact_anchor_support_requires_imaginary_kappa_in_inventory",
                "reject" if prior_audit["summary"]["exact_anchor_support_requires_imaginary_kappa"] else "pass",
                "exact same-family anchor support requires imaginary kappa",
                1 if prior_audit["summary"]["exact_anchor_support_requires_imaginary_kappa"] else 0,
                "The frozen canonical boundary still sends the exact anchors into an imaginary-kappa regime.",
            ),
            row(
                "trial3_t2_exact_anchor_support_realized_as_zero_kappa_clip_branch_in_inventory",
                "reject" if exact_anchor_support_currently_realized_only_as_zero_kappa_clip_branch else "pass",
                "exact same-family anchor support currently realized as zero-kappa clip branch",
                1 if exact_anchor_support_currently_realized_only_as_zero_kappa_clip_branch else 0,
                "The current numerics reach the exact anchors only after clipping the tail scale to zero and removing polarization corrections.",
            ),
            row(
                "trial3_t2_current_canon_has_explicit_imaginary_kappa_tail_interpretation_in_inventory",
                "pass" if current_canon_has_explicit_imaginary_kappa_tail_interpretation else "reject",
                "current canon has explicit imaginary-kappa tail interpretation",
                1 if current_canon_has_explicit_imaginary_kappa_tail_interpretation else 0,
                "No current-canon statement explains what the beta>1 imaginary-kappa regime means physically.",
            ),
            row(
                "trial3_t2_current_canon_has_explicit_zero_kappa_tail_reclassification_in_inventory",
                "pass" if current_canon_has_explicit_zero_kappa_tail_reclassification else "reject",
                "current canon has explicit zero-kappa tail reclassification",
                1 if current_canon_has_explicit_zero_kappa_tail_reclassification else 0,
                "No current-canon statement yet promotes the clipped zero-kappa branch into a physical tail class.",
            ),
            row(
                "trial3_t2_subunity_pair_preserved_in_tail_continuation_inventory",
                "pass" if subunity_pair_preserved else "reject",
                "same-family beta<=1 pair preserved in continuation inventory",
                1 if subunity_pair_preserved else 0,
                "The ratio-compatible beta<=1 pair remains preserved and is not the current blocker.",
            ),
            row(
                "trial3_t2_subunity_w_anchor_miss_preserved_in_tail_continuation_inventory",
                "reject" if not subunity_w_anchor_pass else "pass",
                "same-family beta<=1 W miss preserved in continuation inventory",
                1 if not subunity_w_anchor_pass else 0,
                "The branch still cannot close honestly inside beta<=1, so the beta>1 continuation question remains active.",
            ),
        ],
        {
            "anchor_family_or_none": dict(ANCHOR_FAMILY),
            "exact_anchor_support_requires_imaginary_kappa": prior_audit["summary"]["exact_anchor_support_requires_imaginary_kappa"],
            "exact_anchor_support_currently_realized_only_as_zero_kappa_clip_branch": exact_anchor_support_currently_realized_only_as_zero_kappa_clip_branch,
            "current_canon_has_explicit_imaginary_kappa_tail_interpretation": current_canon_has_explicit_imaginary_kappa_tail_interpretation,
            "current_canon_has_explicit_zero_kappa_tail_reclassification": current_canon_has_explicit_zero_kappa_tail_reclassification,
            "subunity_pair_preserved": subunity_pair_preserved,
            "subunity_w_anchor_pass": subunity_w_anchor_pass,
            "next_required_route": "trial3_t2_beta_tail_continuation_audit",
        },
        {
            "overall_status": "trial3_t2_beta_tail_continuation_inventory_frozen",
            "advance_to_8_7_56_368": True,
            "next_required_artifacts": ["trial3_t2_beta_tail_continuation_audit"],
        },
        {
            "status_next_step_line": hit(status_text, "current official next step は `8.7.56.367`"),
            "roadmap_branch_line": hit(
                roadmap_text,
                "`8.7.56.367-.370` 試練3 two-component beta-above-unity decaying-tail continuation residual branch",
            ),
            "localized_boundary_rule_line": hit(
                full_text,
                'localization = math.sqrt(max(0.0, 1.0 - beta_n * beta_n))',
            ),
            "imaginary_kappa_statement_line": hit(full_text, "imaginary_kappa"),
            "zero_kappa_tail_statement_line": hit(full_text, "zero_kappa"),
            "coupled_freeze_formulas": coupled_freeze["formulas"],
            "exact_best_w_or_none": compact_state(exact_best_w),
            "exact_best_z_or_none": compact_state(exact_best_z),
            "subunity_best_w_or_none": compact_state(subunity_best_w),
            "subunity_best_z_or_none": compact_state(subunity_best_z),
            "subunity_best_pair_or_none": subunity_best_pair,
        },
    )

    audit = payload(
        "8.7.56.368",
        "Trial-3 two-component beta-above-unity decaying-tail continuation audit",
        common_inputs,
        "Audit whether the current exact-anchor gain is already canonically supported, or whether it survives only as an unlicensed zero-kappa clip branch with no current-canon tail reclassification.",
        {
            "classification_rule": "if exact-anchor support mathematically requires imaginary kappa but the current numerics realize it only by clipping kappa to zero, the immediate blocker is whether current canon reclassifies that clipped zero-kappa branch physically",
            "residual_rule": "if there is no explicit zero-kappa tail reclassification, the blocker narrows from generic decaying-tail continuation to zero-kappa tail reclassification itself",
        },
        [
            row(
                "trial3_t2_beta_tail_continuation_audit_complete",
                "pass",
                "Trial-3 two-component beta-above-unity decaying-tail continuation audit complete",
                1,
                "The decaying-tail continuation audit is frozen.",
            ),
            row(
                "trial3_t2_exact_anchor_support_requires_imaginary_kappa_audit_tenth_branch",
                "reject" if prior_audit["summary"]["exact_anchor_support_requires_imaginary_kappa"] else "pass",
                "exact same-family anchor support requires imaginary kappa",
                1 if prior_audit["summary"]["exact_anchor_support_requires_imaginary_kappa"] else 0,
                "The frozen boundary rule still pushes the exact anchors outside the real-kappa regime.",
            ),
            row(
                "trial3_t2_exact_anchor_support_realized_only_as_zero_kappa_clip_branch",
                "reject" if exact_anchor_support_currently_realized_only_as_zero_kappa_clip_branch else "pass",
                "exact same-family anchor support realized only as zero-kappa clip branch",
                1 if exact_anchor_support_currently_realized_only_as_zero_kappa_clip_branch else 0,
                "The current solver reaches the exact anchors only after clipping the boundary scale to zero and zeroing polarization corrections.",
            ),
            row(
                "trial3_t2_current_canon_has_explicit_zero_kappa_tail_reclassification",
                "pass" if current_canon_has_explicit_zero_kappa_tail_reclassification else "reject",
                "current canon has explicit zero-kappa tail reclassification",
                1 if current_canon_has_explicit_zero_kappa_tail_reclassification else 0,
                "This is the direct admissibility condition for the current numerical branch.",
            ),
            row(
                "trial3_t2_current_canon_has_explicit_imaginary_kappa_tail_interpretation",
                "pass" if current_canon_has_explicit_imaginary_kappa_tail_interpretation else "reject",
                "current canon has explicit imaginary-kappa tail interpretation",
                1 if current_canon_has_explicit_imaginary_kappa_tail_interpretation else 0,
                "An explicit imaginary-kappa interpretation is still absent, so the frozen rule itself does not close the branch.",
            ),
            row(
                "trial3_t2_zero_kappa_clip_branch_physically_reclassified_under_current_canon",
                "pass" if zero_kappa_clip_branch_physically_reclassified_under_current_canon else "reject",
                "zero-kappa clip branch physically reclassified under current canon",
                1 if zero_kappa_clip_branch_physically_reclassified_under_current_canon else 0,
                "Without this reclassification, the current exact-anchor gain remains numerical only.",
            ),
            row(
                "trial3_t2_beta_above_unity_tail_continuation_available_under_current_canon",
                "pass" if beta_above_unity_tail_continuation_available_under_current_canon else "reject",
                "beta-above-unity tail continuation available under current canon",
                1 if beta_above_unity_tail_continuation_available_under_current_canon else 0,
                "This is the direct closeout criterion for the current residual branch.",
            ),
        ],
        {
            "anchor_family_or_none": dict(ANCHOR_FAMILY),
            "exact_anchor_support_requires_imaginary_kappa": prior_audit["summary"]["exact_anchor_support_requires_imaginary_kappa"],
            "exact_anchor_support_currently_realized_only_as_zero_kappa_clip_branch": exact_anchor_support_currently_realized_only_as_zero_kappa_clip_branch,
            "current_canon_has_explicit_zero_kappa_tail_reclassification": current_canon_has_explicit_zero_kappa_tail_reclassification,
            "current_canon_has_explicit_imaginary_kappa_tail_interpretation": current_canon_has_explicit_imaginary_kappa_tail_interpretation,
            "zero_kappa_clip_branch_physically_reclassified_under_current_canon": zero_kappa_clip_branch_physically_reclassified_under_current_canon,
            "beta_above_unity_tail_continuation_available_under_current_canon": beta_above_unity_tail_continuation_available_under_current_canon,
            "next_required_route": "trial3_t2_beta_tail_continuation_declaration_tenth_gate",
        },
        {
            "overall_status": "trial3_t2_beta_tail_continuation_audited",
            "advance_to_8_7_56_369": True,
            "next_required_artifacts": ["trial3_t2_beta_tail_continuation_declaration_tenth_gate"],
        },
        {
            "prior_tail_audit_summary": prior_audit["summary"],
            "exact_anchor_zero_kappa_clip_signature": {
                "exact_w_zero_kappa_clip_equivalent": exact_w_zero_kappa_clip_equivalent,
                "exact_z_zero_kappa_clip_equivalent": exact_z_zero_kappa_clip_equivalent,
            },
            "exact_best_w_or_none": compact_state(exact_best_w),
            "exact_best_z_or_none": compact_state(exact_best_z),
        },
    )

    declaration_gate = payload(
        "8.7.56.369",
        "Trial-3 two-component declaration tenth gate",
        common_inputs,
        "Freeze whether the current exact-anchor gain closes Trial-3, or whether the honest next blocker is the missing zero-kappa tail reclassification for the clipped beta>1 branch.",
        {
            "closeout_rule": "close Trial-3 only if the current canon explicitly licenses the beta>1 tail continuation that the exact-anchor numerics actually use",
            "residual_rule": "if the exact anchors survive only on a clipped zero-kappa branch and current canon does not reclassify that branch physically, the next blocker is the missing zero-kappa tail reclassification itself",
        },
        [
            row(
                "trial3_t2_declaration_tenth_gate_complete",
                "pass",
                "Trial-3 two-component declaration tenth gate complete",
                1,
                "The tenth gate is frozen.",
            ),
            row(
                "trial3_t2_branch_closeable_tenth_gate",
                "pass" if branch_closeable else "reject",
                "two-component branch closeable after decaying-tail continuation audit",
                1 if branch_closeable else 0,
                "The branch closes only if the current canon honestly licenses the beta>1 tail continuation actually used by the numerics.",
            ),
            row(
                "trial3_t2_residual_route_required_tenth_gate",
                "reject" if branch_closeable else "pass",
                "two-component residual route still required after decaying-tail continuation audit",
                0 if branch_closeable else 1,
                "A residual route remains required while the clipped zero-kappa branch lacks a current-canon reclassification.",
            ),
        ],
        {
            "solver_range_blocker_removed": True,
            "exact_anchor_support_currently_realized_only_as_zero_kappa_clip_branch": exact_anchor_support_currently_realized_only_as_zero_kappa_clip_branch,
            "zero_kappa_clip_branch_physically_reclassified_under_current_canon": zero_kappa_clip_branch_physically_reclassified_under_current_canon,
            "trial3_current_branch_closeable": branch_closeable,
            "selected_residual_route": (
                None
                if branch_closeable
                else "trial3_two_component_beta_above_unity_zero_kappa_tail_reclassification_identification"
            ),
            "missing_v2_artifact": (
                None
                if branch_closeable
                else "trial3_two_component_beta_above_unity_zero_kappa_tail_reclassification_pack"
            ),
            "recommended_next_route_or_none": None if branch_closeable else NEXT_ROUTE,
        },
        {
            "overall_status": "trial3_t2_declaration_tenth_gate_frozen",
            "trial3_branch_closeable": branch_closeable,
            "advance_to_8_7_56_370": True,
            "next_required_artifacts": [] if branch_closeable else [NEXT_ROUTE],
        },
        {
            "audit_summary": audit["summary"],
            "prior_gate_summary": prior_gate["summary"],
            "current_ai_context_step": ai_context["current_step"],
        },
    )

    disposition = payload(
        "8.7.56.370",
        "Trial-2 paper-side sync / Trial-4 disposition thirty-fifth refresh",
        common_inputs,
        "Refresh the reserve/deferred ordering after the decaying-tail continuation audit narrows the blocker to the missing zero-kappa tail reclassification for the clipped beta>1 branch.",
        {
            "trial2_rule": "Trial-2 paper-side sync remains unlocked reserve retained while Trial-3 still has an honest current-canon residual route",
            "trial4_rule": "Trial-4 remains deferred while the two-component Trial-3 route is still scientifically live",
        },
        [
            row(
                "trial3_t2_trial2_trial4_thirty_fifth_refresh_complete",
                "pass",
                "Trial-2 paper-side sync / Trial-4 disposition thirty-fifth refresh complete",
                1,
                "The reserve/deferred ordering is refreshed after the decaying-tail continuation audit.",
            ),
            row(
                "trial3_t2_trial2_reserve_retained_thirty_fifth_refresh",
                "pass",
                "Trial-2 paper-side sync reserve retained",
                1,
                "Trial-2 paper sync remains reserve work while the Trial-3 zero-kappa-tail reclassification route is still open.",
            ),
            row(
                "trial3_t2_trial4_deferred_retained_thirty_fifth_refresh",
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
            "overall_status": "trial3_t2_trial2_trial4_thirty_fifth_refresh_frozen",
            "trial3_branch_closeable": branch_closeable,
            "advance_to_next_branch": not branch_closeable,
            "next_required_artifacts": [] if branch_closeable else [NEXT_ROUTE],
        },
        {
            "declaration_summary": declaration_gate["summary"],
            "prior_disposition_summary": prior_disposition["summary"],
        },
    )

    write_artifact("mass_origin_v2_t3_t2_beta_tail_continuation_source_inventory", source_inventory)
    write_artifact("mass_origin_v2_t3_t2_beta_tail_continuation_audit", audit)
    write_artifact("mass_origin_v2_t3_t2_beta_tail_continuation_declaration_gate", declaration_gate)
    write_artifact("mass_origin_v2_t3_t2_paper_sync_trial4_disp_35th_refresh", disposition)

    print("[done] Trial-3 two-component beta-above-unity decaying-tail continuation artifacts written:")
    print(" - mass_origin_v2_t3_t2_beta_tail_continuation_source_inventory_metrics.json")
    print(" - mass_origin_v2_t3_t2_beta_tail_continuation_audit_metrics.json")
    print(" - mass_origin_v2_t3_t2_beta_tail_continuation_declaration_gate_metrics.json")
    print(" - mass_origin_v2_t3_t2_paper_sync_trial4_disp_35th_refresh_metrics.json")


# 関数: CLI から branch を実行する。

def run_cli() -> None:
    """CLI entry point for the Trial-3 beta-above-unity decaying-tail continuation branch."""
    main()


if __name__ == "__main__":
    run_cli()
