#!/usr/bin/env python3
"""
Generate Trial-3 two-component beta-above-unity decaying-tail continuation artifacts for 8.7.56.363-.366.

This branch narrows the current blocker one level deeper than the previous
beta-above-unity admissibility audit. The exact same-family W/Z anchors are still
reachable only for `beta_n > 1`, and the frozen canonical boundary rule
`kappa = sqrt(1 - beta^2)` therefore becomes imaginary. The current builder clips
that quantity to zero, but no current-canon statement yet explains how the same
family should remain a localized, decaying-tail branch once `beta_n` crosses unity.
The branch therefore freezes the decaying-tail continuation question itself.
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

BETA_SOURCE = OUT / "mass_origin_v2_t3_t2_beta_admissibility_source_inventory_metrics.json"
BETA_AUDIT = OUT / "mass_origin_v2_t3_t2_beta_admissibility_audit_metrics.json"
BETA_GATE = OUT / "mass_origin_v2_t3_t2_beta_admissibility_declaration_gate_metrics.json"
BETA_DISPOSITION = OUT / "mass_origin_v2_t3_t2_paper_sync_trial4_disp_33rd_refresh_metrics.json"
COUPLED_FREEZE = OUT / "mass_origin_vector_qball_coupled_constraint_freeze_audit_metrics.json"
FULL_SOLVER = ROOT / "scripts" / "quantum" / "mass_origin_vector_qball_full_coupled_solver_branch.py"

ANCHOR_FAMILY = {"k": 17, "ell": 1, "s": 1}
TRIAL2_RESERVE_STATE = "unlocked_reserve_retained"
NEXT_ROUTE = "8.7.56.367"


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


# 関数: beta と ell,s から current builder の tail-related state を評価する。

def evaluate_tail_state(full_solver, beta_n: float, ell: int, s: int) -> dict:
    """Evaluate the current tail-related quantities for one state."""
    localization_argument = 1.0 - float(beta_n) * float(beta_n)
    imaginary_kappa_magnitude = math.sqrt(max(0.0, -localization_argument))
    clipped_kappa_value = math.sqrt(max(0.0, localization_argument))
    return {
        "beta_n": float(beta_n),
        "ell": int(ell),
        "s": int(s),
        "localization_argument": float(localization_argument),
        "requires_imaginary_kappa": bool(localization_argument < 0.0),
        "imaginary_kappa_magnitude": float(imaginary_kappa_magnitude),
        "clipped_kappa_value": float(clipped_kappa_value),
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


# 関数: beta-above-unity decaying-tail continuation branch を実行する。

def main() -> None:
    """Execute the beta-above-unity decaying-tail continuation branch."""
    for path in (
        STATUS,
        ROADMAP,
        AI_CONTEXT,
        BETA_SOURCE,
        BETA_AUDIT,
        BETA_GATE,
        BETA_DISPOSITION,
        COUPLED_FREEZE,
        FULL_SOLVER,
    ):
        req(path)

    status_text = read_text(STATUS)
    roadmap_text = read_text(ROADMAP)
    ai_context = read_json(AI_CONTEXT)
    beta_source = read_json(BETA_SOURCE)
    beta_audit = read_json(BETA_AUDIT)
    beta_gate = read_json(BETA_GATE)
    beta_disposition = read_json(BETA_DISPOSITION)
    coupled_freeze = read_json(COUPLED_FREEZE)
    full_text = read_text(FULL_SOLVER)
    full_solver = load_module(FULL_SOLVER, "trial3_t2_beta_tail_full_solver")

    exact_best_w = beta_source["evidence"]["exact_best_w_or_none"]
    exact_best_z = beta_source["evidence"]["exact_best_z_or_none"]
    subunity_best_w = beta_source["evidence"]["subunity_best_w_or_none"]
    subunity_best_z = beta_source["evidence"]["subunity_best_z_or_none"]
    subunity_best_pair = beta_source["evidence"]["subunity_best_pair_or_none"]

    exact_w_tail = evaluate_tail_state(full_solver, float(exact_best_w["beta_n"]), 1, 1)
    exact_z_tail = evaluate_tail_state(full_solver, float(exact_best_z["beta_n"]), 1, 1)
    subunity_w_tail = evaluate_tail_state(full_solver, float(subunity_best_w["beta_n"]), 1, 1)
    subunity_z_tail = evaluate_tail_state(full_solver, float(subunity_best_z["beta_n"]), 1, 1)

    beta_above_unity_decaying_tail_available_under_current_canon = False
    current_canon_has_explicit_imaginary_kappa_interpretation = False
    current_canon_has_explicit_zero_kappa_tail_reclassification = False
    exact_anchor_support_requires_imaginary_kappa = bool(
        exact_w_tail["requires_imaginary_kappa"] and exact_z_tail["requires_imaginary_kappa"]
    )
    clipped_zero_kappa_is_only_available_implementation = bool(
        exact_anchor_support_requires_imaginary_kappa
        and float(exact_w_tail["clipped_kappa_value"]) == 0.0
        and float(exact_z_tail["clipped_kappa_value"]) == 0.0
        and float(exact_w_tail["polarization_weight_impl"]) == 0.0
        and float(exact_z_tail["polarization_weight_impl"]) == 0.0
    )
    subunity_decay_rule_still_valid = bool(
        not subunity_w_tail["requires_imaginary_kappa"] and not subunity_z_tail["requires_imaginary_kappa"]
    )
    branch_closeable = bool(
        beta_above_unity_decaying_tail_available_under_current_canon
        and beta_audit["summary"]["beta_above_unity_anchor_support_admissible_under_current_canon"]
    )

    common_inputs = {
        "status_markdown": rel(STATUS),
        "roadmap_markdown": rel(ROADMAP),
        "ai_context_json": rel(AI_CONTEXT),
        "mass_origin_v2_t3_t2_beta_admissibility_source_inventory_json": rel(BETA_SOURCE),
        "mass_origin_v2_t3_t2_beta_admissibility_audit_json": rel(BETA_AUDIT),
        "mass_origin_v2_t3_t2_beta_admissibility_declaration_gate_json": rel(BETA_GATE),
        "mass_origin_v2_t3_t2_paper_sync_trial4_disp_33rd_refresh_json": rel(BETA_DISPOSITION),
        "mass_origin_vector_qball_coupled_constraint_freeze_audit_json": rel(COUPLED_FREEZE),
        "mass_origin_vector_qball_full_coupled_solver_branch_py": rel(FULL_SOLVER),
    }

    source_inventory = payload(
        "8.7.56.363",
        "Trial-3 two-component beta-above-unity localized-boundary continuation source inventory",
        common_inputs,
        "Freeze the exact same-family beta>1 anchor states, their negative localization arguments, the decaying-tail rule, the clipped-kappa implementation, and the surviving beta<=1 evidence in one continuation source pack.",
        {
            "boundary_rule": "the frozen canon still defines a localized state by a real decaying tail with kappa = sqrt(1-beta^2)",
            "continuation_rule": "if exact-anchor support sits at beta>1, the next honest question is whether current canon supplies a decaying-tail continuation instead of merely clipping kappa to zero numerically",
        },
        [
            row(
                "trial3_t2_beta_tail_source_inventory_complete",
                "pass",
                "Trial-3 two-component beta-above-unity localized-boundary continuation source inventory complete",
                1,
                "The localized-boundary continuation source pack is frozen.",
            ),
            row(
                "trial3_t2_exact_anchor_requires_imaginary_kappa_in_inventory",
                "reject" if exact_anchor_support_requires_imaginary_kappa else "pass",
                "exact same-family anchor support requires imaginary kappa",
                1 if exact_anchor_support_requires_imaginary_kappa else 0,
                "The exact anchors now force the boundary question into the beta>1 regime where kappa is no longer real.",
            ),
            row(
                "trial3_t2_current_decaying_tail_rule_present_in_inventory",
                "pass",
                "current decaying-tail rule present in frozen canon",
                1,
                "The coupled freeze still defines localized solutions through a real decaying kappa.",
            ),
            row(
                "trial3_t2_clipped_zero_kappa_implementation_present_in_inventory",
                "pass" if hit(full_text, "math.sqrt(max(0.0, 1.0 - beta_n * beta_n))") else "reject",
                "clipped zero-kappa implementation present",
                1 if hit(full_text, "math.sqrt(max(0.0, 1.0 - beta_n * beta_n))") else 0,
                "The current builder-side implementation clips negative localization arguments to zero.",
            ),
            row(
                "trial3_t2_subunity_pair_preserved_in_tail_inventory",
                "pass" if beta_source["summary"]["subunity_pair_preserved"] else "reject",
                "same-family beta<=1 pair preserved in tail inventory",
                1 if beta_source["summary"]["subunity_pair_preserved"] else 0,
                "The ratio-compatible beta<=1 pair remains available and therefore is not the current blocker.",
            ),
            row(
                "trial3_t2_subunity_w_anchor_miss_preserved_in_tail_inventory",
                "reject" if not beta_source["summary"]["subunity_w_anchor_pass"] else "pass",
                "same-family beta<=1 W miss preserved in tail inventory",
                1 if not beta_source["summary"]["subunity_w_anchor_pass"] else 0,
                "The branch still cannot close inside beta<=1, so the beta>1 tail question remains active.",
            ),
        ],
        {
            "anchor_family_or_none": dict(ANCHOR_FAMILY),
            "exact_anchor_support_requires_imaginary_kappa": exact_anchor_support_requires_imaginary_kappa,
            "subunity_decay_rule_still_valid": subunity_decay_rule_still_valid,
            "current_canon_has_explicit_imaginary_kappa_interpretation": current_canon_has_explicit_imaginary_kappa_interpretation,
            "current_canon_has_explicit_zero_kappa_tail_reclassification": current_canon_has_explicit_zero_kappa_tail_reclassification,
            "next_required_route": "trial3_t2_beta_tail_continuation_audit",
        },
        {
            "overall_status": "trial3_t2_beta_tail_inventory_frozen",
            "advance_to_8_7_56_364": True,
            "next_required_artifacts": ["trial3_t2_beta_tail_continuation_audit"],
        },
        {
            "status_next_step_line": hit(status_text, "current official next step は `8.7.56.363`"),
            "roadmap_branch_line": hit(
                roadmap_text,
                "`8.7.56.363-.366` Trial-3 two-component beta-above-unity localized-boundary continuation residual branch",
            ),
            "localized_boundary_rule_line": hit(full_text, '"localized_boundary_rule": "all components decay with the same localization scale kappa = sqrt(1 - beta^2)"'),
            "clip_rule_line": hit(full_text, "localization = math.sqrt(max(0.0, 1.0 - beta_n * beta_n))"),
            "coupled_freeze_formulas": coupled_freeze["formulas"],
            "exact_best_w_or_none": compact_state(exact_best_w),
            "exact_best_z_or_none": compact_state(exact_best_z),
            "subunity_best_w_or_none": compact_state(subunity_best_w),
            "subunity_best_z_or_none": compact_state(subunity_best_z),
            "subunity_best_pair_or_none": subunity_best_pair,
        },
    )

    audit = payload(
        "8.7.56.364",
        "Trial-3 two-component beta-above-unity localized-boundary continuation audit",
        common_inputs,
        "Audit whether the current canon already provides a decaying-tail continuation beyond beta = 1, or whether exact-anchor support lives only on an unlicensed clipped-zero-kappa branch.",
        {
            "decaying_tail_rule": "localized states remain canonically admissible only while kappa is real and the tail decays at infinity",
            "residual_rule": "if exact-anchor support requires imaginary kappa, no current-canon continuation statement exists, and the implementation only clips to zero, the blocker collapses to the missing decaying-tail continuation itself",
        },
        [
            row(
                "trial3_t2_beta_tail_audit_complete",
                "pass",
                "Trial-3 two-component beta-above-unity localized-boundary continuation audit complete",
                1,
                "The localized-boundary continuation audit is frozen.",
            ),
            row(
                "trial3_t2_exact_anchor_support_requires_imaginary_kappa_audit",
                "reject" if exact_anchor_support_requires_imaginary_kappa else "pass",
                "exact same-family anchor support requires imaginary kappa",
                1 if exact_anchor_support_requires_imaginary_kappa else 0,
                "A decaying-tail continuation is only needed because the exact-anchor states leave the real-kappa regime.",
            ),
            row(
                "trial3_t2_current_canon_has_explicit_imaginary_kappa_interpretation",
                "pass" if current_canon_has_explicit_imaginary_kappa_interpretation else "reject",
                "current canon has explicit imaginary-kappa interpretation",
                1 if current_canon_has_explicit_imaginary_kappa_interpretation else 0,
                "An honest continuation would require an explicit statement for what beta>1 means once kappa becomes imaginary.",
            ),
            row(
                "trial3_t2_current_canon_has_explicit_zero_kappa_tail_reclassification",
                "pass" if current_canon_has_explicit_zero_kappa_tail_reclassification else "reject",
                "current canon has explicit zero-kappa tail reclassification",
                1 if current_canon_has_explicit_zero_kappa_tail_reclassification else 0,
                "If the clipped zero-kappa branch is to count physically, current canon must say so explicitly.",
            ),
            row(
                "trial3_t2_clipped_zero_kappa_is_only_available_implementation",
                "reject" if clipped_zero_kappa_is_only_available_implementation else "pass",
                "clipped zero-kappa is the only available implementation",
                1 if clipped_zero_kappa_is_only_available_implementation else 0,
                "If true, the current exact-anchor gain still lacks an honest physical continuation statement.",
            ),
            row(
                "trial3_t2_beta_above_unity_decaying_tail_available_under_current_canon",
                "pass" if beta_above_unity_decaying_tail_available_under_current_canon else "reject",
                "beta-above-unity decaying-tail continuation available under current canon",
                1 if beta_above_unity_decaying_tail_available_under_current_canon else 0,
                "This is the direct closeout criterion for the current residual branch.",
            ),
        ],
        {
            "anchor_family_or_none": dict(ANCHOR_FAMILY),
            "exact_anchor_support_requires_imaginary_kappa": exact_anchor_support_requires_imaginary_kappa,
            "exact_w_imaginary_kappa_magnitude": exact_w_tail["imaginary_kappa_magnitude"],
            "exact_z_imaginary_kappa_magnitude": exact_z_tail["imaginary_kappa_magnitude"],
            "subunity_decay_rule_still_valid": subunity_decay_rule_still_valid,
            "current_canon_has_explicit_imaginary_kappa_interpretation": current_canon_has_explicit_imaginary_kappa_interpretation,
            "current_canon_has_explicit_zero_kappa_tail_reclassification": current_canon_has_explicit_zero_kappa_tail_reclassification,
            "clipped_zero_kappa_is_only_available_implementation": clipped_zero_kappa_is_only_available_implementation,
            "beta_above_unity_decaying_tail_available_under_current_canon": beta_above_unity_decaying_tail_available_under_current_canon,
            "next_required_route": "trial3_t2_beta_tail_declaration_ninth_gate",
        },
        {
            "overall_status": "trial3_t2_beta_tail_audited",
            "advance_to_8_7_56_365": True,
            "next_required_artifacts": ["trial3_t2_beta_tail_declaration_ninth_gate"],
        },
        {
            "beta_admissibility_summary": beta_audit["summary"],
            "exact_w_tail": exact_w_tail,
            "exact_z_tail": exact_z_tail,
            "subunity_w_tail": subunity_w_tail,
            "subunity_z_tail": subunity_z_tail,
        },
    )

    declaration_gate = payload(
        "8.7.56.365",
        "Trial-3 two-component declaration ninth gate",
        common_inputs,
        "Freeze whether the current exact-anchor gain closes Trial-3, or whether the honest next blocker is the missing beta-above-unity decaying-tail continuation itself.",
        {
            "closeout_rule": "close Trial-3 only if the current canon already supplies a physical decaying-tail continuation for beta>1 exact-anchor support",
            "residual_rule": "if exact-anchor support needs imaginary kappa and current canon still offers only the clipped zero-kappa implementation, the next blocker is the missing decaying-tail continuation statement",
        },
        [
            row(
                "trial3_t2_declaration_ninth_gate_complete",
                "pass",
                "Trial-3 two-component declaration ninth gate complete",
                1,
                "The decaying-tail continuation gate is frozen.",
            ),
            row(
                "trial3_t2_branch_closeable_ninth_gate",
                "pass" if branch_closeable else "reject",
                "two-component branch closeable after decaying-tail continuation audit",
                1 if branch_closeable else 0,
                "The branch closes only if the current canon honestly licenses the beta>1 tail continuation.",
            ),
            row(
                "trial3_t2_residual_route_required_ninth_gate",
                "reject" if branch_closeable else "pass",
                "two-component residual route still required after decaying-tail continuation audit",
                0 if branch_closeable else 1,
                "A residual route remains required while the beta>1 decaying-tail continuation is absent.",
            ),
        ],
        {
            "solver_range_blocker_removed": True,
            "beta_above_unity_anchor_support_admissible_under_current_canon": beta_audit["summary"]["beta_above_unity_anchor_support_admissible_under_current_canon"],
            "beta_above_unity_decaying_tail_available_under_current_canon": beta_above_unity_decaying_tail_available_under_current_canon,
            "trial3_current_branch_closeable": branch_closeable,
            "selected_residual_route": (
                None
                if branch_closeable
                else "trial3_two_component_beta_above_unity_decaying_tail_continuation_identification"
            ),
            "missing_v2_artifact": (
                None
                if branch_closeable
                else "trial3_two_component_beta_above_unity_decaying_tail_continuation_pack"
            ),
            "recommended_next_route_or_none": None if branch_closeable else NEXT_ROUTE,
        },
        {
            "overall_status": "trial3_t2_declaration_ninth_gate_frozen",
            "trial3_branch_closeable": branch_closeable,
            "advance_to_8_7_56_366": True,
            "next_required_artifacts": [] if branch_closeable else [NEXT_ROUTE],
        },
        {
            "audit_summary": audit["summary"],
            "beta_gate_summary": beta_gate["summary"],
            "current_ai_context_step": ai_context["current_step"],
        },
    )

    disposition = payload(
        "8.7.56.366",
        "Trial-2 paper-side sync / Trial-4 disposition thirty-fourth refresh",
        common_inputs,
        "Refresh the reserve/deferred ordering after the decaying-tail continuation audit narrows the blocker to the missing beta-above-unity tail statement.",
        {
            "trial2_rule": "Trial-2 paper-side sync remains unlocked reserve retained while Trial-3 still has an honest current-canon residual route",
            "trial4_rule": "Trial-4 remains deferred while the two-component Trial-3 route is still scientifically live",
        },
        [
            row(
                "trial3_t2_trial2_trial4_thirty_fourth_refresh_complete",
                "pass",
                "Trial-2 paper-side sync / Trial-4 disposition thirty-fourth refresh complete",
                1,
                "The reserve/deferred ordering is refreshed after the decaying-tail continuation audit.",
            ),
            row(
                "trial3_t2_trial2_reserve_retained_thirty_fourth_refresh",
                "pass",
                "Trial-2 paper-side sync reserve retained",
                1,
                "Trial-2 paper sync remains reserve work while the Trial-3 decaying-tail continuation route is still open.",
            ),
            row(
                "trial3_t2_trial4_deferred_retained_thirty_fourth_refresh",
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
            "overall_status": "trial3_t2_trial2_trial4_thirty_fourth_refresh_frozen",
            "trial3_branch_closeable": branch_closeable,
            "advance_to_next_branch": not branch_closeable,
            "next_required_artifacts": [] if branch_closeable else [NEXT_ROUTE],
        },
        {
            "declaration_summary": declaration_gate["summary"],
            "beta_disposition_summary": beta_disposition["summary"],
        },
    )

    write_artifact("mass_origin_v2_t3_t2_beta_tail_source_inventory", source_inventory)
    write_artifact("mass_origin_v2_t3_t2_beta_tail_audit", audit)
    write_artifact("mass_origin_v2_t3_t2_beta_tail_declaration_gate", declaration_gate)
    write_artifact("mass_origin_v2_t3_t2_paper_sync_trial4_disp_34th_refresh", disposition)

    print("[done] Trial-3 two-component beta-above-unity decaying-tail continuation artifacts written:")
    print(" - mass_origin_v2_t3_t2_beta_tail_source_inventory_metrics.json")
    print(" - mass_origin_v2_t3_t2_beta_tail_audit_metrics.json")
    print(" - mass_origin_v2_t3_t2_beta_tail_declaration_gate_metrics.json")
    print(" - mass_origin_v2_t3_t2_paper_sync_trial4_disp_34th_refresh_metrics.json")


# 関数: CLI から branch を実行する。

def run_cli() -> None:
    """CLI entry point for the Trial-3 beta-above-unity decaying-tail continuation branch."""
    main()


if __name__ == "__main__":
    run_cli()
