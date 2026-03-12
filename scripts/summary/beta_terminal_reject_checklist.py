#!/usr/bin/env python3
"""
beta_terminal_reject_checklist.py

Roadmap Step 8.7.47.22 + 8.7.48.11:
- Freeze a machine-readable checklist for the beta-terminal reject factors,
  including MESSENGER readiness rows.
- Include additional primary-data requirements for next-stage resolution.

Inputs (public-first fallback):
- output/public/summary/beta_cross_channel_registry.json
- output/public/summary/beta_terminal_comparator_policy_sensitivity.json
- output/public/vlbi/vlbi_allsky_beta_consistency_metrics.json
- output/public/vlbi/vlbi_beta_source_session_matrix_metrics.json

Outputs (default: output/private/summary and synced to output/public/summary):
- beta_terminal_reject_checklist.json
- beta_terminal_reject_checklist.csv
- beta_terminal_reject_checklist.pdf
- beta_terminal_reject_checklist.png
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


_ROOT = Path(__file__).resolve().parents[2]


# 関数: `_safe_rel` の入出力契約と処理意図を定義する。
def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve()).replace("\\", "/")


# 関数: `_first_existing` の入出力契約と処理意図を定義する。

def _first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for p in paths:
        # 条件分岐: `p.exists()` を満たす経路を評価する。
        if p.exists():
            return p

    return None


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_normalize_status` の入出力契約と処理意図を定義する。

def _normalize_status(value: Any) -> str:
    s = str(value or "").strip().lower()
    # 条件分岐: `s in {"pass", "ok"}` を満たす経路を評価する。
    if s in {"pass", "ok"}:
        return "pass"

    # 条件分岐: `s in {"watch", "mixed", "pending"}` を満たす経路を評価する。

    if s in {"watch", "mixed", "pending"}:
        return "watch"

    # 条件分岐: `s in {"reject", "ng", "fail", "failed", "hard_reject"}` を満たす経路を評価する。

    if s in {"reject", "ng", "fail", "failed", "hard_reject"}:
        return "reject"

    return "reject"


# 関数: `_status_to_score` の入出力契約と処理意図を定義する。

def _status_to_score(status: str) -> float:
    s = _normalize_status(status)
    # 条件分岐: `s == "pass"` を満たす経路を評価する。
    if s == "pass":
        return 0.5

    # 条件分岐: `s == "watch"` を満たす経路を評価する。

    if s == "watch":
        return 1.5

    return 2.8


# 関数: `_status_color` の入出力契約と処理意図を定義する。

def _status_color(status: str) -> str:
    s = _normalize_status(status)
    # 条件分岐: `s == "pass"` を満たす経路を評価する。
    if s == "pass":
        return "#2ca02c"

    # 条件分岐: `s == "watch"` を満たす経路を評価する。

    if s == "watch":
        return "#f1c232"

    return "#d62728"


# 関数: `_to_float` の入出力契約と処理意図を定義する。

def _to_float(value: Any) -> Optional[float]:
    try:
        x = float(value)
    except Exception:
        return None

    # 条件分岐: `not np.isfinite(x)` を満たす経路を評価する。

    if not np.isfinite(x):
        return None

    return x


# 関数: `_find_source_entry` の入出力契約と処理意図を定義する。

def _find_source_entry(source_summary: Sequence[Dict[str, Any]], source_name: str) -> Dict[str, Any]:
    key = str(source_name).strip().lower()
    for row in source_summary:
        # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
        if not isinstance(row, dict):
            continue

        # 条件分岐: `str(row.get("source", "")).strip().lower() == key` を満たす経路を評価する。

        if str(row.get("source", "")).strip().lower() == key:
            return row

    return {}


# 関数: `_find_policy_row` の入出力契約と処理意図を定義する。

def _find_policy_row(policy_rows: Sequence[Dict[str, Any]], policy_id: str) -> Dict[str, Any]:
    key = str(policy_id).strip().lower()
    for row in policy_rows:
        # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
        if not isinstance(row, dict):
            continue

        # 条件分岐: `str(row.get("policy_id", "")).strip().lower() == key` を満たす経路を評価する。

        if str(row.get("policy_id", "")).strip().lower() == key:
            return row

    return {}


# 関数: `_build_rows` の入出力契約と処理意図を定義する。

def _build_rows(
    registry: Dict[str, Any],
    policy: Dict[str, Any],
    vlbi_allsky: Dict[str, Any],
    source_matrix: Dict[str, Any],
) -> List[Dict[str, Any]]:
    vlbi = registry.get("vlbi") if isinstance(registry.get("vlbi"), dict) else {}
    llr = registry.get("llr") if isinstance(registry.get("llr"), dict) else {}
    messenger = registry.get("messenger") if isinstance(registry.get("messenger"), dict) else {}
    cross = registry.get("cross_channel") if isinstance(registry.get("cross_channel"), dict) else {}
    beta_terminal = registry.get("beta_terminal") if isinstance(registry.get("beta_terminal"), dict) else {}

    policy_summary = policy.get("summary") if isinstance(policy.get("summary"), dict) else {}
    policy_rows = policy.get("rows") if isinstance(policy.get("rows"), list) else []
    policy_d_row = _find_policy_row(policy_rows, "policy_D_exclude_plus_messenger_fusion")
    allsky_cons = vlbi_allsky.get("consistency") if isinstance(vlbi_allsky.get("consistency"), dict) else {}
    source_summary = source_matrix.get("source_summary") if isinstance(source_matrix.get("source_summary"), list) else []
    source_proxy = (
        source_matrix.get("session_consistency_proxy")
        if isinstance(source_matrix.get("session_consistency_proxy"), dict)
        else {}
    )

    near_sun_0235 = _find_source_entry(source_summary, "0235+164")
    near_sun_0229 = _find_source_entry(source_summary, "0229+131")
    far_source = _find_source_entry(source_summary, "0955+476")
    active_policy_id = str(beta_terminal.get("active_policy_id") or "policy_A_hard_reject_keep")
    active_policy_label = str(beta_terminal.get("active_policy_label") or "A: hard reject keep")
    policy_governance = (
        beta_terminal.get("policy_governance")
        if isinstance(beta_terminal.get("policy_governance"), dict)
        else {}
    )
    policy_b_hold_status = _normalize_status(policy_governance.get("policy_b_hold_status"))
    policy_d_promotion_status = _normalize_status(policy_governance.get("policy_d_promotion_status"))
    policy_d_promotion_ready = bool(policy_governance.get("policy_d_promotion_ready", False))
    policy_d_promotion_blockers = (
        policy_governance.get("policy_d_promotion_blockers")
        if isinstance(policy_governance.get("policy_d_promotion_blockers"), list)
        else []
    )
    policy_d_blocker_priority = (
        policy_governance.get("policy_d_promotion_blocker_priority")
        if isinstance(policy_governance.get("policy_d_promotion_blocker_priority"), list)
        else []
    )
    policy_d_blocker_resolution_order = (
        policy_governance.get("policy_d_promotion_blocker_resolution_order")
        if isinstance(policy_governance.get("policy_d_promotion_blocker_resolution_order"), list)
        else []
    )
    policy_d_blocker_order_status = _normalize_status(policy_governance.get("policy_d_promotion_blocker_order_status"))
    recommended_active_policy_id = str(policy_governance.get("recommended_active_policy_id") or "")
    llr_gate_execution_order = (
        policy_governance.get("llr_gate_execution_order")
        if isinstance(policy_governance.get("llr_gate_execution_order"), list)
        else []
    )
    llr_gate_actions_ordered = (
        policy_governance.get("llr_gate_actions_ordered")
        if isinstance(policy_governance.get("llr_gate_actions_ordered"), list)
        else []
    )
    llr_gate_order_status = _normalize_status(policy_governance.get("llr_gate_order_status"))
    llr_gate_repro_commands_min = (
        policy_governance.get("llr_gate_repro_commands_min")
        if isinstance(policy_governance.get("llr_gate_repro_commands_min"), list)
        else []
    )
    policy_switch_decision = (
        policy_governance.get("policy_switch_decision")
        if isinstance(policy_governance.get("policy_switch_decision"), dict)
        else {}
    )
    policy_terminal_watch_statement = (
        policy_governance.get("policy_terminal_watch_statement")
        if isinstance(policy_governance.get("policy_terminal_watch_statement"), dict)
        else {}
    )
    policy_switch_status = _normalize_status(policy_switch_decision.get("status"))
    policy_switch_decision_id = str(policy_switch_decision.get("decision_id") or "")
    policy_switch_required_now = bool(policy_switch_decision.get("switch_required_now", False))
    policy_switch_allowed_now = bool(policy_switch_decision.get("switch_allowed_now", False))
    policy_switch_hold_reason = str(policy_switch_decision.get("hold_reason") or "")
    policy_terminal_watch_status = _normalize_status(policy_terminal_watch_statement.get("status"))
    policy_terminal_watch_statement_id = str(policy_terminal_watch_statement.get("statement_id") or "")
    policy_terminal_watch_statement_text = str(policy_terminal_watch_statement.get("statement_text") or "")
    policy_d_reassessment = (
        policy_governance.get("policy_d_promotion_reassessment")
        if isinstance(policy_governance.get("policy_d_promotion_reassessment"), dict)
        else {}
    )
    policy_terminal_watch_statement = (
        policy_governance.get("policy_terminal_watch_statement")
        if isinstance(policy_governance.get("policy_terminal_watch_statement"), dict)
        else {}
    )
    policy_d_reassessment_status = _normalize_status(policy_d_reassessment.get("status"))
    policy_d_reassessment_order_match = bool(policy_d_reassessment.get("blocker_order_alignment", False))
    policy_d_reassessment_count_delta = int(policy_d_reassessment.get("blocker_count_delta", 0) or 0)
    policy_d_reassessment_policy_alignment = bool(policy_d_reassessment.get("policy_d_status_alignment", False))
    messenger_beta_definition = str(messenger.get("beta_primary_definition") or "unknown")
    messenger_beta = _to_float(messenger.get("beta_primary_est"))
    messenger_sigma = _to_float(messenger.get("beta_primary_sigma"))
    messenger_beta_dyn = _to_float(messenger.get("beta_dyn_diagnostic_est"))
    messenger_sigma_dyn = _to_float(messenger.get("beta_dyn_diagnostic_sigma"))
    stage_i_priority = (
        messenger.get("stage_i_nuisance_priority")
        if isinstance(messenger.get("stage_i_nuisance_priority"), dict)
        else {}
    )
    stage_i_top_nuisance = (
        stage_i_priority.get("mitigation_priority")[0]
        if isinstance(stage_i_priority.get("mitigation_priority"), list) and stage_i_priority.get("mitigation_priority")
        else {}
    )
    policy_d_pair_status = _normalize_status(policy_d_row.get("pair_status_llr_messenger"))
    policy_d_pair_watch_status = _normalize_status(policy_d_row.get("pair_watch_gate_status"))
    policy_d_pair_pass_status = _normalize_status(policy_d_row.get("pair_pass_gate_status"))
    policy_d_pair_abs_delta = _to_float(policy_d_row.get("pair_abs_delta_beta_llr_messenger"))
    policy_d_pair_watch_limit = _to_float(policy_d_row.get("pair_required_abs_delta_watch"))
    stage_i_odf_baseline_abs_z = _to_float(messenger.get("stage_i_odf_baseline_abs_z_beta_minus_1"))
    stage_i_odf_best_abs_z = _to_float(messenger.get("stage_i_odf_best_abs_z_beta_minus_1"))
    stage_i_odf_max_shift_delta_z = _to_float(messenger.get("stage_i_odf_max_shift_delta_z"))
    stage_i_tnf_baseline_abs_z = _to_float(messenger.get("stage_i_tnf_baseline_abs_z_beta_minus_1"))
    stage_i_tnf_best_abs_z = _to_float(messenger.get("stage_i_tnf_best_abs_z_beta_minus_1"))
    stage_i_tnf_max_shift_delta_z = _to_float(messenger.get("stage_i_tnf_max_shift_delta_z"))
    messenger_abs_z = None
    messenger_beta_minus_1_status = "reject"
    if (
        messenger_beta is not None
        and messenger_sigma is not None
        and float(messenger_sigma) > 0.0
    ):
        messenger_abs_z = abs((float(messenger_beta) - 1.0) / max(float(messenger_sigma), 1e-30))
        messenger_beta_minus_1_status = "pass" if messenger_abs_z <= 2.0 else "reject"

    messenger_dyn_abs_z = None
    messenger_dyn_beta_minus_1_status = "reject"
    if (
        messenger_beta_dyn is not None
        and messenger_sigma_dyn is not None
        and float(messenger_sigma_dyn) > 0.0
    ):
        messenger_dyn_abs_z = abs((float(messenger_beta_dyn) - 1.0) / max(float(messenger_sigma_dyn), 1e-30))
        messenger_dyn_beta_minus_1_status = "pass" if messenger_dyn_abs_z <= 2.0 else "reject"

    rows: List[Dict[str, Any]] = []
    rows.append(
        {
            "id": "terminal_policy_current",
            "category": "terminal_policy",
            "status": _normalize_status(beta_terminal.get("status")),
            "metric": "current_policy",
            "value": f"{active_policy_id}:{active_policy_label}",
            "evidence": str(policy_summary.get("status_shift_baseline_to_exclude") or "NA"),
            "next_action": "keep_active_policy_until_gates_change",
        }
    )
    rows.append(
        {
            "id": "cross_consistency_gate",
            "category": "cross_channel",
            "status": _normalize_status(beta_terminal.get("cross_consistency_status")),
            "metric": "abs_z(beta_vlbi-beta_llr)",
            "value": _to_float(beta_terminal.get("beta_consistency_abs_z")),
            "evidence": str(cross.get("beta_consistency_status") or ""),
            "next_action": "require_independent_vlbi_or_llr_extension",
        }
    )
    rows.append(
        {
            "id": "policy_b_hold_governance",
            "category": "policy_governance",
            "status": policy_b_hold_status,
            "metric": "policy_B_hold_status",
            "value": str(policy_b_hold_status),
            "evidence": str(policy_governance.get("policy_b_hold_conditions") or ""),
            "next_action": (
                "keep_policy_B_until_policy_D_promotion_pass"
                if policy_b_hold_status == "pass"
                else "recompute_policy_selection_inputs"
            ),
        }
    )
    rows.append(
        {
            "id": "llr_gate_execution_order",
            "category": "policy_governance",
            "status": llr_gate_order_status,
            "metric": "llr_gate_execution_order",
            "value": "->".join(str(x) for x in llr_gate_execution_order),
            "evidence": "actions=" + "->".join(str(x) for x in llr_gate_actions_ordered),
            "next_action": (
                str(llr_gate_actions_ordered[0])
                if len(llr_gate_actions_ordered) > 0
                else "keep_llr_gate_order_frozen"
            ),
        }
    )
    rows.append(
        {
            "id": "llr_gate_repro_min_chain",
            "category": "policy_governance",
            "status": ("pass" if len(llr_gate_repro_commands_min) >= 4 else "watch"),
            "metric": "llr_gate_repro_commands_min",
            "value": " | ".join(str(x) for x in llr_gate_repro_commands_min),
            "evidence": f"n={len(llr_gate_repro_commands_min)}",
            "next_action": "run_llr_gate_chain_in_order",
        }
    )
    rows.append(
        {
            "id": "policy_d_promotion_governance",
            "category": "policy_governance",
            "status": policy_d_promotion_status,
            "metric": "policy_D_promotion_status",
            "value": str(policy_d_promotion_status),
            "evidence": (
                f"ready={policy_d_promotion_ready}, blockers={','.join(str(x) for x in policy_d_promotion_blockers)}"
            ),
            "next_action": (
                "policy_D_promotion_ready_use_llr_messenger_fusion"
                if policy_d_promotion_ready
                else "resolve_policy_D_promotion_blockers"
            ),
        }
    )
    rows.append(
        {
            "id": "policy_d_promotion_reassessment",
            "category": "policy_governance",
            "status": policy_d_reassessment_status,
            "metric": "policy_D_reassessment_status",
            "value": str(policy_d_reassessment_status),
            "evidence": (
                f"order_match={policy_d_reassessment_order_match}, "
                f"count_delta={policy_d_reassessment_count_delta}, "
                f"policy_alignment={policy_d_reassessment_policy_alignment}"
            ),
            "next_action": (
                "keep_policy_D_reassessment_snapshot"
                if policy_d_reassessment_status == "pass"
                else "recompute_policy_D_reassessment_after_llr_updates"
            ),
        }
    )
    top_blocker = policy_d_blocker_priority[0] if len(policy_d_blocker_priority) > 0 else {}
    rows.append(
        {
            "id": "policy_d_blocker_resolution_order",
            "category": "policy_governance",
            "status": policy_d_blocker_order_status,
            "metric": "policy_D_blocker_resolution_order",
            "value": ",".join(str(x) for x in policy_d_blocker_resolution_order),
            "evidence": (
                f"top={str(top_blocker.get('blocker_id') or '')}, "
                f"action={str(top_blocker.get('recommended_action') or '')}"
            ),
            "next_action": str(top_blocker.get("recommended_action") or "resolve_policy_D_promotion_blockers"),
        }
    )
    for blocker in policy_d_blocker_priority:
        # 条件分岐: `not isinstance(blocker, dict)` を満たす経路を評価する。
        if not isinstance(blocker, dict):
            continue

        blocker_id = str(blocker.get("blocker_id") or "").strip()
        # 条件分岐: `not blocker_id` を満たす経路を評価する。
        if not blocker_id:
            continue

        depends_on = blocker.get("depends_on") if isinstance(blocker.get("depends_on"), list) else []
        rank = blocker.get("priority_rank")
        rows.append(
            {
                "id": f"policy_d_blocker_priority_{blocker_id}",
                "category": "policy_governance",
                "status": _normalize_status(blocker.get("status")),
                "metric": "policy_D_blocker_priority",
                "value": blocker_id,
                "evidence": (
                    f"rank={rank}, depends_on={','.join(str(x) for x in depends_on)}, "
                    f"resolution={str(blocker.get('resolution_condition') or '')}"
                ),
                "next_action": str(blocker.get("recommended_action") or "resolve_policy_D_promotion_blockers"),
            }
        )

    rows.append(
        {
            "id": "policy_recommended_active_policy",
            "category": "policy_governance",
            "status": ("pass" if recommended_active_policy_id == active_policy_id else "watch"),
            "metric": "recommended_active_policy_id",
            "value": recommended_active_policy_id,
            "evidence": str(policy_governance.get("recommended_active_policy_reason") or ""),
            "next_action": (
                "active_policy_aligned_with_governance"
                if recommended_active_policy_id == active_policy_id
                else "align_active_policy_with_governance_recommendation"
            ),
        }
    )
    rows.append(
        {
            "id": "policy_switch_redecision",
            "category": "policy_governance",
            "status": policy_switch_status,
            "metric": "policy_switch_decision_id",
            "value": policy_switch_decision_id,
            "evidence": (
                f"switch_required_now={policy_switch_required_now}, "
                f"switch_allowed_now={policy_switch_allowed_now}, "
                f"hold_reason={policy_switch_hold_reason}"
            ),
            "next_action": (
                "switch_policy_B_to_policy_D_now"
                if policy_switch_required_now and policy_switch_allowed_now
                else "hold_policy_B_until_policy_D_ready"
            ),
        }
    )
    rows.append(
        {
            "id": "policy_terminal_watch_statement",
            "category": "policy_governance",
            "status": policy_terminal_watch_status,
            "metric": "policy_terminal_watch_statement_id",
            "value": policy_terminal_watch_statement_id,
            "evidence": policy_terminal_watch_statement_text,
            "next_action": (
                "keep_canonical_watch_statement_until_policy_D_ready"
                if policy_terminal_watch_status == "pass"
                else "refresh_watch_statement_after_policy_switch_redecision"
            ),
        }
    )
    active_policy_update_required = bool(
        recommended_active_policy_id and active_policy_id and (recommended_active_policy_id != active_policy_id)
    )
    rows.append(
        {
            "id": "active_policy_update_decision",
            "category": "terminal_policy",
            "status": ("pass" if not active_policy_update_required else "watch"),
            "metric": "active_policy_update_required",
            "value": active_policy_update_required,
            "evidence": f"active={active_policy_id}, recommended={recommended_active_policy_id}",
            "next_action": (
                "keep_active_policy_as_recommended"
                if not active_policy_update_required
                else "update_active_policy_to_recommended_id"
            ),
        }
    )
    rows.append(
        {
            "id": "vlbi_comparator_eligibility",
            "category": "vlbi",
            "status": _normalize_status(vlbi.get("subset_refit_status")),
            "metric": "comparator_eligible",
            "value": bool(vlbi.get("subset_refit_eligible", True)),
            "evidence": str(vlbi.get("subset_refit_reason") or ""),
            "next_action": "keep_comparator_ineligible",
        }
    )
    rows.append(
        {
            "id": "vlbi_allsky_session_consistency",
            "category": "vlbi",
            "status": _normalize_status(allsky_cons.get("status")),
            "metric": "allsky_chi2_dof",
            "value": _to_float(allsky_cons.get("chi2_dof")),
            "evidence": f"n_valid={int(allsky_cons.get('n_valid', 0) or 0)}",
            "next_action": "expand_high_sensitivity_sessions",
        }
    )
    rows.append(
        {
            "id": "vlbi_source_session_proxy",
            "category": "vlbi",
            "status": _normalize_status("reject"),
            "metric": "source_session_chi2_dof",
            "value": _to_float(source_proxy.get("chi2_dof")),
            "evidence": f"rows={int(source_proxy.get('dof', 0) or 0)+1}",
            "next_action": "split_corona_vs_structure_systematics",
        }
    )
    rows.append(
        {
            "id": "vlbi_near_sun_source_0235",
            "category": "vlbi_source",
            "status": _normalize_status(near_sun_0235.get("status")),
            "metric": "0235+164_chi2_dof",
            "value": _to_float(near_sun_0235.get("chi2_dof")),
            "evidence": "near_sun_target",
            "next_action": "require_multiband_corona_control",
        }
    )
    rows.append(
        {
            "id": "vlbi_near_sun_source_0229",
            "category": "vlbi_source",
            "status": _normalize_status(near_sun_0229.get("status")),
            "metric": "0229+131_chi2_dof",
            "value": _to_float(near_sun_0229.get("chi2_dof")),
            "evidence": "near_sun_target",
            "next_action": "require_multiband_corona_control",
        }
    )
    rows.append(
        {
            "id": "vlbi_far_source_reference",
            "category": "vlbi_source",
            "status": _normalize_status(far_source.get("status")),
            "metric": "0955+476_chi2_dof",
            "value": _to_float(far_source.get("chi2_dof")),
            "evidence": "far_source_reference",
            "next_action": "keep_as_structure_stability_reference",
        }
    )
    rows.append(
        {
            "id": "policy_d_llr_messenger_pair_z",
            "category": "policy_d_pair",
            "status": policy_d_pair_status,
            "metric": "abs_z(beta_llr-beta_messenger)",
            "value": _to_float(policy_d_row.get("pair_abs_z_llr_messenger")),
            "evidence": (
                f"policy_D fusion pair gate (watch={policy_d_pair_watch_status}, "
                f"pass={policy_d_pair_pass_status})"
            ),
            "next_action": (
                "maintain_pair_gate_and_focus_bias_audit"
                if policy_d_pair_status == "pass"
                else "reduce_pair_abs_z_to_watch_or_better"
            ),
        }
    )
    rows.append(
        {
            "id": "policy_d_pair_delta_gap",
            "category": "policy_d_pair",
            "status": policy_d_pair_status,
            "metric": "abs_delta_vs_watch_threshold",
            "value": (
                (policy_d_pair_abs_delta or 0.0)
                - (policy_d_pair_watch_limit or 0.0)
            ),
            "evidence": (
                f"abs_delta={policy_d_pair_abs_delta}, watch_limit={policy_d_pair_watch_limit}"
            ),
            "next_action": (
                "pair_gate_pass_keep_bias_as_remaining_gate"
                if policy_d_pair_status == "pass"
                else "move_messenger_beta_into_watch_band_or_increase_sigma_if_justified"
            ),
        }
    )
    rows.append(
        {
            "id": "policy_d_messenger_watch_band",
            "category": "policy_d_pair",
            "status": policy_d_pair_status,
            "metric": "required_messenger_beta_watch_range",
            "value": (
                f"[{_to_float(policy_d_row.get('pair_required_messenger_beta_min_watch'))}, "
                f"{_to_float(policy_d_row.get('pair_required_messenger_beta_max_watch'))}]"
            ),
            "evidence": f"current_beta_messenger={messenger_beta}",
            "next_action": (
                "watch_band_satisfied_focus_bias_and_policy_governance"
                if policy_d_pair_status == "pass"
                else "verify_if_stage_i_nuisance_can_move_beta_into_range"
            ),
        }
    )
    rows.append(
        {
            "id": "messenger_stage_i_odf_decomposition",
            "category": "messenger_diagnostic",
            "status": _normalize_status("reject"),
            "metric": "odf_abs_z_baseline_to_best",
            "value": (
                (stage_i_odf_baseline_abs_z or 0.0) - (stage_i_odf_best_abs_z or 0.0)
                if stage_i_odf_baseline_abs_z is not None and stage_i_odf_best_abs_z is not None
                else None
            ),
            "evidence": f"baseline={stage_i_odf_baseline_abs_z}, best={stage_i_odf_best_abs_z}, max_shift_z={stage_i_odf_max_shift_delta_z}",
            "next_action": "prioritize_plasma_transponder_nuisance_control_for_odf",
        }
    )
    rows.append(
        {
            "id": "messenger_stage_i_tnf_decomposition",
            "category": "messenger_diagnostic",
            "status": _normalize_status("watch"),
            "metric": "tnf_abs_z_baseline_to_best",
            "value": (
                (stage_i_tnf_baseline_abs_z or 0.0) - (stage_i_tnf_best_abs_z or 0.0)
                if stage_i_tnf_baseline_abs_z is not None and stage_i_tnf_best_abs_z is not None
                else None
            ),
            "evidence": f"baseline={stage_i_tnf_baseline_abs_z}, best={stage_i_tnf_best_abs_z}, max_shift_z={stage_i_tnf_max_shift_delta_z}",
            "next_action": "keep_tnf_replay_stable_while_aligning_odf_primary",
        }
    )
    rows.append(
        {
            "id": "messenger_stage_i_nuisance_priority_top1",
            "category": "messenger_diagnostic",
            "status": _normalize_status(stage_i_priority.get("priority_status")),
            "metric": "top_nuisance_scenario",
            "value": str(stage_i_top_nuisance.get("scenario_id") or ""),
            "evidence": (
                f"group={stage_i_top_nuisance.get('scenario_group')}, "
                f"max_abs_z_delta={stage_i_top_nuisance.get('max_abs_z_delta')}"
            ),
            "next_action": str(stage_i_top_nuisance.get("recommended_action") or "register_stage_i_nuisance_priority"),
        }
    )
    rows.append(
        {
            "id": "messenger_stage_i_srp_proxy_registration",
            "category": "messenger_diagnostic",
            "status": ("pass" if bool(stage_i_priority.get("srp_proxy_registered", False)) else "watch"),
            "metric": "srp_proxy_registered",
            "value": bool(stage_i_priority.get("srp_proxy_registered", False)),
            "evidence": (
                f"{str(stage_i_priority.get('srp_proxy_note') or '')}"
                f"; scenarios={','.join([str(x) for x in (stage_i_priority.get('srp_proxy_scenarios') or [])])}"
            ),
            "next_action": (
                "calibrate_srp_proxy_against_spacecraft_force_model"
                if bool(stage_i_priority.get("srp_proxy_registered", False))
                else "add_or_map_srp_like_proxy_if_physically_justified"
            ),
        }
    )
    rows.append(
        {
            "id": "llr_beta_minus_1",
            "category": "llr",
            "status": _normalize_status(beta_terminal.get("beta_minus_1_status")),
            "metric": "abs_z(beta_llr-1)",
            "value": _to_float(llr.get("abs_z")),
            "evidence": str(llr.get("selected_mode") or ""),
            "next_action": "maintain_decontaminated_pipeline",
        }
    )
    rows.append(
        {
            "id": "llr_bias_audit",
            "category": "llr",
            "status": _normalize_status(llr.get("bias_audit_status")),
            "metric": "bias_audit_status",
            "value": str(llr.get("bias_audit_status") or ""),
            "evidence": str(llr.get("promotion_status") or ""),
            "next_action": "continue_station_target_hardware_balance",
        }
    )
    rows.append(
        {
            "id": "messenger_stage_j_final_gate",
            "category": "messenger",
            "status": _normalize_status(messenger.get("stage_j_status")),
            "metric": "stage_j_status",
            "value": str(messenger.get("stage_j_status") or ""),
            "evidence": str(messenger.get("eligibility_reason") or ""),
            "next_action": "keep_stage_j_pass_while_expanding_years",
        }
    )
    rows.append(
        {
            "id": "messenger_replay_consistency",
            "category": "messenger",
            "status": _normalize_status(messenger.get("stage_e_replay_status")),
            "metric": "abs_z(beta_tnf-beta_odf)",
            "value": _to_float(messenger.get("stage_e_replay_z_delta_beta")),
            "evidence": str(messenger.get("stage_e_replay_status") or ""),
            "next_action": "maintain_replay_gate_under_kernel_updates",
        }
    )
    rows.append(
        {
            "id": "messenger_bias_audit",
            "category": "messenger",
            "status": _normalize_status(messenger.get("bias_audit_status")),
            "metric": "bias_audit_status",
            "value": str(messenger.get("bias_audit_status") or ""),
            "evidence": str(messenger.get("policy_note") or ""),
            "next_action": "keep_core_policy_pass_and_track_diagnostic_scenarios",
        }
    )
    rows.append(
        {
            "id": "messenger_beta_lt_minus_1_primary",
            "category": "messenger",
            "status": _normalize_status(messenger_beta_minus_1_status),
            "metric": "abs_z(beta_lt-1)",
            "value": messenger_abs_z,
            "evidence": f"beta_primary_definition={messenger_beta_definition}",
            "next_action": "keep_beta_lt_as_primary_beta_channel",
        }
    )
    rows.append(
        {
            "id": "messenger_beta_dyn_diagnostic",
            "category": "messenger_diagnostic",
            "status": _normalize_status(messenger_dyn_beta_minus_1_status),
            "metric": "abs_z(beta_dyn-1)",
            "value": messenger_dyn_abs_z,
            "evidence": "non_gravitational_acceleration_sensitive_diagnostic",
            "next_action": "treat_beta_dyn_as_nongrav_model_deficit_indicator",
        }
    )
    rows.append(
        {
            "id": "next_primary_data_vlbi",
            "category": "next_data",
            "status": "watch",
            "metric": "primary_data_requirement",
            "value": "vgos_or_ka_multiband_near_far_pair",
            "evidence": "corona_residual_plus_source_structure",
            "next_action": "add_high_sensitivity_sessions_with_multiband_control",
        }
    )
    rows.append(
        {
            "id": "next_primary_data_llr",
            "category": "next_data",
            "status": "watch",
            "metric": "primary_data_requirement",
            "value": "independent_llr_channel_expansion",
            "evidence": "llr_bias_audit_not_pass",
            "next_action": "add_balanced_station_hardware_and_independent_templates",
        }
    )
    rows.append(
        {
            "id": "next_primary_data_messenger",
            "category": "next_data",
            "status": "watch" if messenger_dyn_beta_minus_1_status == "reject" else "pass",
            "metric": "primary_data_requirement",
            "value": "messenger_nongrav_model_upgrade",
            "evidence": "beta_dyn_diagnostic_gap",
            "next_action": "improve_nongrav_model_and_monitor_beta_dyn_convergence",
        }
    )
    return rows


# 関数: `_write_csv_rows` の入出力契約と処理意図を定義する。

def _write_csv_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "category", "status", "metric", "value", "evidence", "next_action"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


# 関数: `_write_plot` の入出力契約と処理意図を定義する。

def _write_plot(rows: Sequence[Dict[str, Any]], out_pdf: Path, out_png: Path) -> None:
    labels = [str(r.get("id") or "") for r in rows]
    statuses = [_normalize_status(r.get("status")) for r in rows]
    scores = [_status_to_score(s) for s in statuses]
    colors = [_status_color(s) for s in statuses]
    notes = [f"{r.get('metric')}={r.get('value')}" for r in rows]

    fig, ax = plt.subplots(figsize=(12.2, 6.6))
    y = np.arange(len(labels), dtype=float)
    ax.barh(y, scores, color=colors, alpha=0.92)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.0)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 3.0)
    ax.axvline(1.0, color="#999999", linestyle="--", linewidth=1.0)
    ax.axvline(2.0, color="#999999", linestyle="--", linewidth=1.0)
    ax.grid(axis="x", alpha=0.2)
    ax.set_xlabel("status score (pass=0.5, watch=1.5, reject=2.8)")
    ax.set_title("Step 8.7.47.22+8.7.48.11: beta terminal reject checklist")

    for i, note in enumerate(notes):
        ax.text(0.02, i, note, va="center", ha="left", fontsize=8.7)

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


# 関数: `_sync_outputs_to_public` の入出力契約と処理意図を定義する。

def _sync_outputs_to_public(paths: Iterable[Path], private_root: Path, public_root: Path) -> List[str]:
    public_root.mkdir(parents=True, exist_ok=True)
    synced: List[str] = []
    for p in paths:
        try:
            rel = p.resolve().relative_to(private_root.resolve())
        except Exception:
            rel = Path(p.name)

        dst = public_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        synced.append(_safe_rel(dst, _ROOT))

    return synced


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    ap = argparse.ArgumentParser(description="Build beta terminal reject checklist (Roadmap 8.7.47.22 + 8.7.48.30).")
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(_ROOT / "output" / "private" / "summary"),
    )
    ap.add_argument(
        "--public-dir",
        type=str,
        default=str(_ROOT / "output" / "public" / "summary"),
    )
    args = ap.parse_args()

    out_dir = Path(str(args.out_dir))
    public_dir = Path(str(args.public_dir))
    # 条件分岐: `not out_dir.is_absolute()` を満たす経路を評価する。
    if not out_dir.is_absolute():
        out_dir = (_ROOT / out_dir).resolve()

    # 条件分岐: `not public_dir.is_absolute()` を満たす経路を評価する。

    if not public_dir.is_absolute():
        public_dir = (_ROOT / public_dir).resolve()

    registry_path = _first_existing(
        [
            _ROOT / "output" / "public" / "summary" / "beta_cross_channel_registry.json",
            _ROOT / "output" / "private" / "summary" / "beta_cross_channel_registry.json",
        ]
    )
    policy_path = _first_existing(
        [
            _ROOT / "output" / "public" / "summary" / "beta_terminal_comparator_policy_sensitivity.json",
            _ROOT / "output" / "private" / "summary" / "beta_terminal_comparator_policy_sensitivity.json",
        ]
    )
    vlbi_allsky_path = _first_existing(
        [
            _ROOT / "output" / "public" / "vlbi" / "vlbi_allsky_beta_consistency_metrics.json",
            _ROOT / "output" / "vlbi" / "vlbi_allsky_beta_consistency_metrics.json",
        ]
    )
    source_matrix_path = _first_existing(
        [
            _ROOT / "output" / "public" / "vlbi" / "vlbi_beta_source_session_matrix_metrics.json",
            _ROOT / "output" / "vlbi" / "vlbi_beta_source_session_matrix_metrics.json",
        ]
    )

    # 条件分岐: `registry_path is None or policy_path is None or vlbi_allsky_path is None or source_matrix_path is None` を満たす経路を評価する。

    if registry_path is None or policy_path is None or vlbi_allsky_path is None or source_matrix_path is None:
        raise FileNotFoundError("required inputs are missing for beta terminal reject checklist.")

    registry = _read_json(registry_path)
    policy = _read_json(policy_path)
    vlbi_allsky = _read_json(vlbi_allsky_path)
    source_matrix = _read_json(source_matrix_path)

    rows = _build_rows(registry=registry, policy=policy, vlbi_allsky=vlbi_allsky, source_matrix=source_matrix)
    policy_governance = (
        (registry.get("beta_terminal") or {}).get("policy_governance")
        if isinstance((registry.get("beta_terminal") or {}).get("policy_governance"), dict)
        else {}
    )
    blocker_priority = (
        policy_governance.get("policy_d_promotion_blocker_priority")
        if isinstance(policy_governance.get("policy_d_promotion_blocker_priority"), list)
        else []
    )
    blocker_resolution_order = (
        policy_governance.get("policy_d_promotion_blocker_resolution_order")
        if isinstance(policy_governance.get("policy_d_promotion_blocker_resolution_order"), list)
        else []
    )
    policy_d_reassessment = (
        policy_governance.get("policy_d_promotion_reassessment")
        if isinstance(policy_governance.get("policy_d_promotion_reassessment"), dict)
        else {}
    )
    policy_switch_decision = (
        policy_governance.get("policy_switch_decision")
        if isinstance(policy_governance.get("policy_switch_decision"), dict)
        else {}
    )
    policy_terminal_watch_statement = (
        policy_governance.get("policy_terminal_watch_statement")
        if isinstance(policy_governance.get("policy_terminal_watch_statement"), dict)
        else {}
    )
    llr_gate_execution_order_main = (
        policy_governance.get("llr_gate_execution_order")
        if isinstance(policy_governance.get("llr_gate_execution_order"), list)
        else []
    )
    llr_gate_actions_ordered_main = (
        policy_governance.get("llr_gate_actions_ordered")
        if isinstance(policy_governance.get("llr_gate_actions_ordered"), list)
        else []
    )
    llr_gate_repro_commands_min_main = (
        policy_governance.get("llr_gate_repro_commands_min")
        if isinstance(policy_governance.get("llr_gate_repro_commands_min"), list)
        else []
    )
    blocker_actions_ordered: List[str] = []
    for row in blocker_priority:
        # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
        if not isinstance(row, dict):
            continue

        action = str(row.get("recommended_action") or "").strip()
        # 条件分岐: `not action` を満たす経路を評価する。
        if not action:
            continue

        blocker_actions_ordered.append(action)

    beta_terminal = registry.get("beta_terminal") if isinstance(registry.get("beta_terminal"), dict) else {}
    active_policy_current = str(beta_terminal.get("active_policy_id") or "")
    recommended_policy_id = str(policy_governance.get("recommended_active_policy_id") or "")
    active_policy_update_required = bool(
        active_policy_current and recommended_policy_id and (active_policy_current != recommended_policy_id)
    )

    status_counts = {
        "pass": int(sum(1 for r in rows if _normalize_status(r.get("status")) == "pass")),
        "watch": int(sum(1 for r in rows if _normalize_status(r.get("status")) == "watch")),
        "reject": int(sum(1 for r in rows if _normalize_status(r.get("status")) == "reject")),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "beta_terminal_reject_checklist.json"
    out_csv = out_dir / "beta_terminal_reject_checklist.csv"
    out_pdf = out_dir / "beta_terminal_reject_checklist.pdf"
    out_png = out_dir / "beta_terminal_reject_checklist.png"

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase": {"step": "8.7.47.22+8.7.48.11+8.7.48.30"},
        "inputs": {
            "beta_cross_channel_registry_json": _safe_rel(registry_path, _ROOT),
            "beta_terminal_comparator_policy_sensitivity_json": _safe_rel(policy_path, _ROOT),
            "vlbi_allsky_beta_consistency_metrics_json": _safe_rel(vlbi_allsky_path, _ROOT),
            "vlbi_beta_source_session_matrix_metrics_json": _safe_rel(source_matrix_path, _ROOT),
        },
        "rows": rows,
        "summary": {
            "terminal_status_current": _normalize_status(((registry.get("beta_terminal") or {}).get("status"))),
            "status_counts": status_counts,
            "active_policy_current": active_policy_current,
            "active_policy_recommended": recommended_policy_id,
            "active_policy_update_required": active_policy_update_required,
            "policy_d_reassessment_status": _normalize_status(policy_d_reassessment.get("status")),
            "policy_d_reassessment_order_match": bool(policy_d_reassessment.get("blocker_order_alignment", False)),
            "policy_d_reassessment_blocker_delta": int(policy_d_reassessment.get("blocker_count_delta", 0) or 0),
            "policy_switch_decision_id": str(policy_switch_decision.get("decision_id") or ""),
            "policy_switch_required_now": bool(policy_switch_decision.get("switch_required_now", False)),
            "policy_switch_allowed_now": bool(policy_switch_decision.get("switch_allowed_now", False)),
            "policy_switch_hold_reason": str(policy_switch_decision.get("hold_reason") or ""),
            "policy_terminal_watch_statement_id": str(policy_terminal_watch_statement.get("statement_id") or ""),
            "policy_terminal_watch_statement_text": str(policy_terminal_watch_statement.get("statement_text") or ""),
            "llr_gate_order_status": _normalize_status(policy_governance.get("llr_gate_order_status")),
            "llr_gate_execution_order": [str(x) for x in llr_gate_execution_order_main],
            "llr_gate_actions_ordered": [str(x) for x in llr_gate_actions_ordered_main],
            "llr_gate_repro_commands_min": [str(x) for x in llr_gate_repro_commands_min_main],
            "policy_d_blocker_resolution_order": [str(x) for x in blocker_resolution_order],
            "policy_d_blocker_actions_ordered": blocker_actions_ordered,
            "next_primary_data_requirements": [
                "vgos_or_ka_multiband_near_far_pair",
                "independent_llr_channel_expansion",
                "messenger_dyn_lt_parameter_separation",
            ],
        },
        "outputs": {
            "json": _safe_rel(out_json, _ROOT),
            "csv": _safe_rel(out_csv, _ROOT),
            "pdf": _safe_rel(out_pdf, _ROOT),
            "png": _safe_rel(out_png, _ROOT),
        },
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv_rows(out_csv, rows)
    _write_plot(rows, out_pdf=out_pdf, out_png=out_png)

    produced = [out_json, out_csv, out_pdf, out_png]
    synced = _sync_outputs_to_public(produced, private_root=out_dir, public_root=public_dir)
    print(f"[ok] wrote: {out_json}")
    print(f"[ok] wrote: {out_csv}")
    print(f"[ok] wrote: {out_pdf}")
    print(f"[ok] wrote: {out_png}")
    print(f"[ok] synced_to_public: {len(synced)} files")
    print(
        "[summary] "
        f"pass={status_counts['pass']} watch={status_counts['watch']} reject={status_counts['reject']} "
        f"terminal={_normalize_status(((registry.get('beta_terminal') or {}).get('status')))}"
    )
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
