#!/usr/bin/env python3
"""
Generate Trial-3 two-component beta-above-unity anchor-support admissibility artifacts for 8.7.56.359-.362.

This branch freezes the next honest blocker after the charge-window-extension pivot.
The range blocker is gone and the ratio-compatible family `(k, ell, s) = (17, 1, 1)`
now hits exact same-family W/Z anchors numerically, but those anchors sit at
`beta_n > 1`. Under the currently frozen full-coupled canon, the same route still
uses `alpha_(ell,s)(beta) ~ sqrt(1 - beta^2)` and the localized boundary
`kappa = sqrt(1 - beta^2)`, while the implementation clips the square root input
to zero. The branch therefore audits whether current canon already licenses that
beta-above-unity support, or whether the blocker has collapsed to a missing
localized-boundary continuation statement.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "public" / "quantum"
STATUS = ROOT / "doc" / "STATUS.md"
ROADMAP = ROOT / "doc" / "ROADMAP.md"
AI_CONTEXT = ROOT / "doc" / "AI_CONTEXT_MIN.json"

PIVOT_SOURCE = OUT / "mass_origin_v2_t3_t2_charge_window_pivot_source_inventory_metrics.json"
PIVOT_AUDIT = OUT / "mass_origin_v2_t3_t2_charge_window_pivot_execution_audit_metrics.json"
PIVOT_GATE = OUT / "mass_origin_v2_t3_t2_charge_window_pivot_declaration_gate_metrics.json"
PIVOT_DISPOSITION = OUT / "mass_origin_v2_t3_t2_paper_sync_trial4_disp_32nd_refresh_metrics.json"
COUPLED_FREEZE = OUT / "mass_origin_vector_qball_coupled_constraint_freeze_audit_metrics.json"
SOLVER_SPEC = OUT / "mass_origin_vector_qball_solver_spec_metrics.json"
FULL_SOLVER = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_full_coupled_solver_branch.py"

ANCHOR_FAMILY = {"k": 17, "ell": 1, "s": 1}
TRIAL2_RESERVE_STATE = "unlocked_reserve_retained"
NEXT_ROUTE = "8.7.56.363"


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


# 関数: long row table から要約サンプルだけを返す。

def sample(rows: list[dict], count: int = 12) -> list[dict]:
    """Return a sparse sample of long tables for compact evidence payloads."""
    if len(rows) <= count:
        return rows

    step = max(1, len(rows) // count)
    sampled = [rows[index] for index in range(0, len(rows), step)]
    return sampled[:count]


# 関数: local Python module を動的 import する。

def load_module(path: Path, module_name: str):
    """Load a local Python module from a filesystem path."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"[fail] unable to import module: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 関数: beta から localization argument と clipped weight の状態を評価する。

def evaluate_beta_state(full_solver, beta_n: float, ell: int, s: int) -> dict:
    """Evaluate the current full-coupled implementation at a fixed beta value."""
    localization_argument = 1.0 - float(beta_n) * float(beta_n)
    clipped_localization = math.sqrt(max(0.0, localization_argument))
    return {
        "beta_n": float(beta_n),
        "ell": int(ell),
        "s": int(s),
        "localization_argument": float(localization_argument),
        "localized_boundary_defined_without_clip": bool(localization_argument >= 0.0),
        "clipped_localization_value": float(clipped_localization),
        "polarization_weight_impl": float(full_solver.polarization_weight(float(beta_n), int(ell), int(s))),
        "coupled_charge_factor_impl": float(full_solver.coupled_charge_factor(float(beta_n), int(ell), int(s))),
        "coupled_mass_factor_impl": float(full_solver.coupled_mass_factor(float(beta_n), int(ell), int(s))),
    }


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


# 関数: current beta-above-unity admissibility branch を実行する。

def main() -> None:
    """Execute the beta-above-unity anchor-support admissibility branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        PIVOT_SOURCE,
        PIVOT_AUDIT,
        PIVOT_GATE,
        PIVOT_DISPOSITION,
        COUPLED_FREEZE,
        SOLVER_SPEC,
        FULL_SOLVER,
    ):
        req(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    pivot_source = read_json(PIVOT_SOURCE)
    pivot_audit = read_json(PIVOT_AUDIT)
    pivot_gate = read_json(PIVOT_GATE)
    pivot_disposition = read_json(PIVOT_DISPOSITION)
    coupled_freeze = read_json(COUPLED_FREEZE)
    solver_spec = read_json(SOLVER_SPEC)
    full_text = read_text(FULL_SOLVER)
    full_solver = load_module(FULL_SOLVER, "trial3_t2_beta_admissibility_full_solver")

    exact_best_w = pivot_audit["evidence"]["exact_best_w_or_none"]
    exact_best_z = pivot_audit["evidence"]["exact_best_z_or_none"]
    subunity_best_w = pivot_audit["evidence"]["subunity_best_w_or_none"]
    subunity_best_z = pivot_audit["evidence"]["subunity_best_z_or_none"]
    subunity_best_pair = pivot_audit["evidence"]["subunity_best_pair_or_none"]
    continuation_rows = list(pivot_audit["evidence"]["continuation_row_sample"])

    exact_w_eval = evaluate_beta_state(full_solver, float(exact_best_w["beta_n"]), 1, 1)
    exact_z_eval = evaluate_beta_state(full_solver, float(exact_best_z["beta_n"]), 1, 1)
    subunity_w_eval = evaluate_beta_state(full_solver, float(subunity_best_w["beta_n"]), 1, 1)
    subunity_z_eval = evaluate_beta_state(full_solver, float(subunity_best_z["beta_n"]), 1, 1)

    exact_w_beta_excess = float(exact_best_w["beta_n"]) - 1.0
    exact_z_beta_excess = float(exact_best_z["beta_n"]) - 1.0
    exact_anchor_support_requires_beta_above_unity = bool(
        exact_w_beta_excess > 0.0 or exact_z_beta_excess > 0.0
    )
    current_builder_uses_clipped_localization = bool(
        hit(full_text, "math.sqrt(max(0.0, 1.0 - beta_n * beta_n))") is not None
    )
    current_canon_has_explicit_beta_above_unity_continuation = False
    beta_above_unity_localized_boundary_defined_under_current_canon = bool(
        exact_w_eval["localized_boundary_defined_without_clip"]
        and exact_z_eval["localized_boundary_defined_without_clip"]
    )
    clipped_branch_only_reason_for_exact_anchor_support = bool(
        current_builder_uses_clipped_localization
        and exact_anchor_support_requires_beta_above_unity
        and not beta_above_unity_localized_boundary_defined_under_current_canon
        and float(exact_w_eval["polarization_weight_impl"]) == 0.0
        and float(exact_z_eval["polarization_weight_impl"]) == 0.0
    )
    subunity_anchor_closeout_available = bool(
        pivot_audit["summary"]["same_family_subunity_pair_preserved"]
        and pivot_audit["summary"]["same_family_subunity_z_anchor_pass"]
        and pivot_audit["summary"]["same_family_subunity_w_anchor_pass"]
    )
    beta_above_unity_anchor_support_admissible_under_current_canon = bool(
        exact_anchor_support_requires_beta_above_unity
        and current_canon_has_explicit_beta_above_unity_continuation
        and beta_above_unity_localized_boundary_defined_under_current_canon
    )
    branch_closeable = bool(
        pivot_audit["summary"]["solver_range_blocker_removed"]
        and (subunity_anchor_closeout_available or beta_above_unity_anchor_support_admissible_under_current_canon)
    )

    common_inputs = {
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "mass_origin_v2_t3_t2_charge_window_pivot_source_inventory_json": rel(PIVOT_SOURCE),
        "mass_origin_v2_t3_t2_charge_window_pivot_execution_audit_json": rel(PIVOT_AUDIT),
        "mass_origin_v2_t3_t2_charge_window_pivot_declaration_gate_json": rel(PIVOT_GATE),
        "mass_origin_v2_t3_t2_paper_sync_trial4_disp_32nd_refresh_json": rel(PIVOT_DISPOSITION),
        "mass_origin_vector_qball_coupled_constraint_freeze_audit_json": rel(COUPLED_FREEZE),
        "mass_origin_vector_qball_solver_spec_json": rel(SOLVER_SPEC),
        "mass_origin_vector_qball_full_coupled_solver_branch_py": rel(FULL_SOLVER),
    }

    source_inventory = payload(
        "8.7.56.359",
        "Trial-3 two-component beta-above-unity anchor-support admissibility source inventory",
        common_inputs,
        "Freeze the exact beta-above-unity W/Z anchor hits, the current clipped polarization rule, the localized-boundary rule, and the surviving beta<=1 evidence in one admissibility source pack.",
        {
            "exact_anchor_rule": "the charge-window-extension pivot is only informative for closeout if the exact same-family anchors it finds are admissible under the current full-coupled canon",
            "current_canon_rule": "the frozen solver canon uses alpha_(ell,s)(beta) ~ sqrt(1-beta^2) and localized boundary kappa = sqrt(1-beta^2), while the implementation clips negative arguments to zero",
        },
        [
            row(
                "trial3_t2_beta_above_unity_admissibility_source_inventory_complete",
                "pass",
                "Trial-3 two-component beta-above-unity anchor-support admissibility source inventory complete",
                1,
                "The admissibility source pack is frozen.",
            ),
            row(
                "trial3_t2_exact_same_family_w_anchor_present",
                "pass" if exact_best_w is not None else "reject",
                "exact same-family W anchor present after charge-window extension",
                1 if exact_best_w is not None else 0,
                "The admissibility audit starts only because the extended family now reaches an exact W anchor numerically.",
            ),
            row(
                "trial3_t2_exact_same_family_z_anchor_present",
                "pass" if exact_best_z is not None else "reject",
                "exact same-family Z anchor present after charge-window extension",
                1 if exact_best_z is not None else 0,
                "The extended family now reaches an exact Z anchor numerically as well.",
            ),
            row(
                "trial3_t2_subunity_pair_preserved_in_inventory",
                "pass" if pivot_audit["summary"]["same_family_subunity_pair_preserved"] else "reject",
                "same-family beta<=1 pair preserved",
                1 if pivot_audit["summary"]["same_family_subunity_pair_preserved"] else 0,
                "The admissibility question is not about the ratio pair, which remains preserved inside beta<=1.",
            ),
            row(
                "trial3_t2_subunity_w_anchor_still_missed_in_inventory",
                "reject" if not pivot_audit["summary"]["same_family_subunity_w_anchor_pass"] else "pass",
                "same-family beta<=1 W anchor still missed",
                1 if not pivot_audit["summary"]["same_family_subunity_w_anchor_pass"] else 0,
                "The branch cannot close inside beta<=1 because the W anchor remains open there.",
            ),
            row(
                "trial3_t2_current_clip_rule_present_in_inventory",
                "pass" if current_builder_uses_clipped_localization else "reject",
                "current clipped localization rule present",
                1 if current_builder_uses_clipped_localization else 0,
                "The current builder-side implementation clips sqrt(1-beta^2) to zero above unity.",
            ),
            row(
                "trial3_t2_localized_boundary_rule_present_in_inventory",
                "pass",
                "localized boundary rule present in frozen canon",
                1,
                "The frozen coupled-constraint audit still defines the localized boundary as kappa = sqrt(1-beta^2).",
            ),
        ],
        {
            "anchor_family_or_none": dict(ANCHOR_FAMILY),
            "exact_w_beta_excess_over_unity": exact_w_beta_excess,
            "exact_z_beta_excess_over_unity": exact_z_beta_excess,
            "current_builder_uses_clipped_localization": current_builder_uses_clipped_localization,
            "current_canon_has_explicit_beta_above_unity_continuation": current_canon_has_explicit_beta_above_unity_continuation,
            "subunity_pair_preserved": pivot_audit["summary"]["same_family_subunity_pair_preserved"],
            "subunity_z_anchor_pass": pivot_audit["summary"]["same_family_subunity_z_anchor_pass"],
            "subunity_w_anchor_pass": pivot_audit["summary"]["same_family_subunity_w_anchor_pass"],
            "next_required_route": "trial3_t2_beta_above_unity_admissibility_audit",
        },
        {
            "overall_status": "trial3_t2_beta_above_unity_admissibility_inventory_frozen",
            "advance_to_8_7_56_360": True,
            "next_required_artifacts": ["trial3_t2_beta_above_unity_admissibility_audit"],
        },
        {
            "status_next_step_line": hit(status_text, "current official next step は `8.7.56.359`"),
            "roadmap_branch_line": hit(
                roadmap_text,
                "`8.7.56.359-.362` Trial-3 two-component beta-above-unity anchor-support admissibility residual branch",
            ),
            "polarization_weight_line": hit(full_text, "def polarization_weight(beta_n: float, ell: int, s: int) -> float:"),
            "clip_rule_line": hit(full_text, "localization = math.sqrt(max(0.0, 1.0 - beta_n * beta_n))"),
            "localized_boundary_rule_line": hit(full_text, '"localized_boundary_rule": "all components decay with the same localization scale kappa = sqrt(1 - beta^2)"'),
            "coupled_freeze_formulas": coupled_freeze["formulas"],
            "solver_spec_formulas": solver_spec["formulas"],
            "exact_best_w_or_none": compact_state(exact_best_w),
            "exact_best_z_or_none": compact_state(exact_best_z),
            "subunity_best_w_or_none": compact_state(subunity_best_w),
            "subunity_best_z_or_none": compact_state(subunity_best_z),
            "subunity_best_pair_or_none": subunity_best_pair,
        },
    )

    audit = payload(
        "8.7.56.360",
        "Trial-3 two-component beta-above-unity anchor-support admissibility audit",
        common_inputs,
        "Audit whether the exact same-family W/Z anchors found beyond beta_n = 1 are already licensed by the current full-coupled canon, or whether they only survive because the current implementation clips the localized-boundary square root to zero.",
        {
            "admissibility_rule": "beta-above-unity anchor support is admissible only if the current canon defines a physical continuation for the localized boundary and polarization weight beyond beta = 1",
            "residual_rule": "if the exact anchors need beta_n > 1, the beta<=1 W anchor still fails, and the current canon has only the clipped implementation but no continuation statement, the blocker collapses to beta-above-unity localized-boundary continuation",
        },
        [
            row(
                "trial3_t2_beta_above_unity_admissibility_audit_complete",
                "pass",
                "Trial-3 two-component beta-above-unity anchor-support admissibility audit complete",
                1,
                "The admissibility audit is frozen.",
            ),
            row(
                "trial3_t2_exact_anchor_support_requires_beta_above_unity_audit",
                "reject" if exact_anchor_support_requires_beta_above_unity else "pass",
                "exact same-family anchor support requires beta_n > 1",
                1 if exact_anchor_support_requires_beta_above_unity else 0,
                "The exact same-family anchor support is only a live closeout candidate if the current canon can justify continuing past beta = 1.",
            ),
            row(
                "trial3_t2_beta_above_unity_localized_boundary_defined_under_current_canon",
                "pass" if beta_above_unity_localized_boundary_defined_under_current_canon else "reject",
                "beta-above-unity localized boundary defined under current canon",
                1 if beta_above_unity_localized_boundary_defined_under_current_canon else 0,
                "The frozen localized-boundary rule must remain defined beyond unity if the exact anchors are to count as an honest closeout.",
            ),
            row(
                "trial3_t2_current_canon_has_explicit_beta_above_unity_continuation",
                "pass" if current_canon_has_explicit_beta_above_unity_continuation else "reject",
                "current canon has explicit beta-above-unity continuation statement",
                1 if current_canon_has_explicit_beta_above_unity_continuation else 0,
                "An honest closeout would require a frozen continuation statement, not only a clipped numerical implementation.",
            ),
            row(
                "trial3_t2_clipped_branch_only_reason_for_exact_anchor_support",
                "reject" if clipped_branch_only_reason_for_exact_anchor_support else "pass",
                "clipped branch is the only reason exact anchor support survives",
                1 if clipped_branch_only_reason_for_exact_anchor_support else 0,
                "If true, the exact anchors do not yet count as a current-canon physical closeout.",
            ),
            row(
                "trial3_t2_subunity_anchor_closeout_available",
                "pass" if subunity_anchor_closeout_available else "reject",
                "same-family beta<=1 anchor closeout available",
                1 if subunity_anchor_closeout_available else 0,
                "If beta<=1 already closed both anchors, no beta-above-unity admissibility route would be needed.",
            ),
            row(
                "trial3_t2_beta_above_unity_anchor_support_admissible_under_current_canon",
                "pass" if beta_above_unity_anchor_support_admissible_under_current_canon else "reject",
                "beta-above-unity anchor support admissible under current canon",
                1 if beta_above_unity_anchor_support_admissible_under_current_canon else 0,
                "This is the honest decision point for whether the current Trial-3 branch can be closed.",
            ),
        ],
        {
            "anchor_family_or_none": dict(ANCHOR_FAMILY),
            "exact_w_beta_excess_over_unity": exact_w_beta_excess,
            "exact_z_beta_excess_over_unity": exact_z_beta_excess,
            "exact_w_localization_argument": exact_w_eval["localization_argument"],
            "exact_z_localization_argument": exact_z_eval["localization_argument"],
            "subunity_w_localization_argument": subunity_w_eval["localization_argument"],
            "subunity_z_localization_argument": subunity_z_eval["localization_argument"],
            "subunity_anchor_closeout_available": subunity_anchor_closeout_available,
            "current_builder_uses_clipped_localization": current_builder_uses_clipped_localization,
            "current_canon_has_explicit_beta_above_unity_continuation": current_canon_has_explicit_beta_above_unity_continuation,
            "beta_above_unity_localized_boundary_defined_under_current_canon": beta_above_unity_localized_boundary_defined_under_current_canon,
            "clipped_branch_only_reason_for_exact_anchor_support": clipped_branch_only_reason_for_exact_anchor_support,
            "beta_above_unity_anchor_support_admissible_under_current_canon": beta_above_unity_anchor_support_admissible_under_current_canon,
            "next_required_route": "trial3_t2_beta_above_unity_declaration_eighth_gate",
        },
        {
            "overall_status": "trial3_t2_beta_above_unity_admissibility_audited",
            "advance_to_8_7_56_361": True,
            "next_required_artifacts": ["trial3_t2_beta_above_unity_declaration_eighth_gate"],
        },
        {
            "pivot_summary": pivot_audit["summary"],
            "exact_w_eval": exact_w_eval,
            "exact_z_eval": exact_z_eval,
            "subunity_w_eval": subunity_w_eval,
            "subunity_z_eval": subunity_z_eval,
            "continuation_row_sample": sample(continuation_rows, 12),
        },
    )

    declaration_gate = payload(
        "8.7.56.361",
        "Trial-3 two-component declaration eighth gate",
        common_inputs,
        "Freeze whether the charge-window-extension gain now closes Trial-3, or whether the exact-anchor route remains blocked by a missing beta-above-unity localized-boundary continuation under the current canon.",
        {
            "closeout_rule": "close Trial-3 only if the exact-anchor support is either already closed inside beta<=1 or explicitly admissible above unity under the frozen canon",
            "residual_rule": "if exact-anchor support needs beta>1 while the frozen localized-boundary rule remains sqrt(1-beta^2) with only a clipped implementation, the next blocker is the missing localized-boundary continuation itself",
        },
        [
            row(
                "trial3_t2_declaration_eighth_gate_complete",
                "pass",
                "Trial-3 two-component declaration eighth gate complete",
                1,
                "The beta-above-unity admissibility gate is frozen.",
            ),
            row(
                "trial3_t2_branch_closeable_eighth_gate",
                "pass" if branch_closeable else "reject",
                "two-component branch closeable after beta-above-unity admissibility audit",
                1 if branch_closeable else 0,
                "The branch closes only if the current canon honestly licenses the exact-anchor support.",
            ),
            row(
                "trial3_t2_residual_route_required_eighth_gate",
                "reject" if branch_closeable else "pass",
                "two-component residual route still required after beta-above-unity admissibility audit",
                0 if branch_closeable else 1,
                "A residual route remains required while the beta-above-unity localized-boundary continuation is absent.",
            ),
        ],
        {
            "solver_range_blocker_removed": pivot_audit["summary"]["solver_range_blocker_removed"],
            "same_family_subunity_pair_preserved": pivot_audit["summary"]["same_family_subunity_pair_preserved"],
            "same_family_subunity_w_anchor_pass": pivot_audit["summary"]["same_family_subunity_w_anchor_pass"],
            "same_family_subunity_z_anchor_pass": pivot_audit["summary"]["same_family_subunity_z_anchor_pass"],
            "beta_above_unity_anchor_support_admissible_under_current_canon": beta_above_unity_anchor_support_admissible_under_current_canon,
            "trial3_current_branch_closeable": branch_closeable,
            "selected_residual_route": (
                None
                if branch_closeable
                else "trial3_two_component_beta_above_unity_localized_boundary_continuation_identification"
            ),
            "missing_v2_artifact": (
                None
                if branch_closeable
                else "trial3_two_component_beta_above_unity_localized_boundary_continuation_pack"
            ),
            "recommended_next_route_or_none": None if branch_closeable else NEXT_ROUTE,
        },
        {
            "overall_status": "trial3_t2_declaration_eighth_gate_frozen",
            "trial3_branch_closeable": branch_closeable,
            "advance_to_8_7_56_362": True,
            "next_required_artifacts": [] if branch_closeable else [NEXT_ROUTE],
        },
        {
            "audit_summary": audit["summary"],
            "pivot_gate_summary": pivot_gate["summary"],
            "current_ai_context_step": ai_context["current_step"],
        },
    )

    disposition = payload(
        "8.7.56.362",
        "Trial-2 paper-side sync / Trial-4 disposition thirty-third refresh",
        common_inputs,
        "Refresh the reserve/deferred ordering after the beta-above-unity admissibility audit narrows the blocker to a missing localized-boundary continuation statement.",
        {
            "trial2_rule": "Trial-2 paper-side sync remains unlocked reserve retained while Trial-3 still has an honest current-canon residual route",
            "trial4_rule": "Trial-4 remains deferred while the two-component Trial-3 route is still scientifically live",
        },
        [
            row(
                "trial3_t2_trial2_trial4_thirty_third_refresh_complete",
                "pass",
                "Trial-2 paper-side sync / Trial-4 disposition thirty-third refresh complete",
                1,
                "The reserve/deferred ordering is refreshed after the admissibility audit.",
            ),
            row(
                "trial3_t2_trial2_reserve_retained_thirty_third_refresh",
                "pass",
                "Trial-2 paper-side sync reserve retained",
                1,
                "Trial-2 paper sync remains reserve work while the Trial-3 localized-boundary continuation route is still open.",
            ),
            row(
                "trial3_t2_trial4_deferred_retained_thirty_third_refresh",
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
            "overall_status": "trial3_t2_trial2_trial4_thirty_third_refresh_frozen",
            "trial3_branch_closeable": branch_closeable,
            "advance_to_next_branch": not branch_closeable,
            "next_required_artifacts": [] if branch_closeable else [NEXT_ROUTE],
        },
        {
            "declaration_summary": declaration_gate["summary"],
            "pivot_disposition_summary": pivot_disposition["summary"],
        },
    )

    write_artifact("mass_origin_v2_t3_t2_beta_admissibility_source_inventory", source_inventory)
    write_artifact("mass_origin_v2_t3_t2_beta_admissibility_audit", audit)
    write_artifact("mass_origin_v2_t3_t2_beta_admissibility_declaration_gate", declaration_gate)
    write_artifact("mass_origin_v2_t3_t2_paper_sync_trial4_disp_33rd_refresh", disposition)

    print("[done] Trial-3 two-component beta-above-unity admissibility artifacts written:")
    print(" - mass_origin_v2_t3_t2_beta_admissibility_source_inventory_metrics.json")
    print(" - mass_origin_v2_t3_t2_beta_admissibility_audit_metrics.json")
    print(" - mass_origin_v2_t3_t2_beta_admissibility_declaration_gate_metrics.json")
    print(" - mass_origin_v2_t3_t2_paper_sync_trial4_disp_33rd_refresh_metrics.json")


# 関数: CLI から branch を実行する。

def run_cli() -> None:
    """CLI entry point for the Trial-3 beta-above-unity admissibility branch."""
    main()


if __name__ == "__main__":
    run_cli()
