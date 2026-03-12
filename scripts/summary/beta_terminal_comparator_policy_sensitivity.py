#!/usr/bin/env python3
"""
beta_terminal_comparator_policy_sensitivity.py

Roadmap Step 8.7.47.21 + 8.7.48.11:
- Audit terminal-policy sensitivity when VLBI comparator is ineligible,
  including MESSENGER-assisted fallback branches.
  (`subset_refit_eligible=false`).
- Freeze machine-readable comparison between:
  1) hard reject keep (current),
  2) comparator-excluded fallback,
  3) comparator-excluded + LLR promotion-pass requirement,
  4) comparator-excluded + LLR+MESSENGER fusion,
  5) comparator-excluded + MESSENGER-only.

Input:
- output/public/summary/beta_cross_channel_registry.json
  (fallback: output/private/summary/beta_cross_channel_registry.json)

Outputs (default: output/private/summary, then synced to output/public/summary):
- beta_terminal_comparator_policy_sensitivity.json
- beta_terminal_comparator_policy_sensitivity.csv
- beta_terminal_comparator_policy_sensitivity.pdf
- beta_terminal_comparator_policy_sensitivity.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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


# 関数: `_combine_status` の入出力契約と処理意図を定義する。

def _combine_status(statuses: Sequence[str]) -> str:
    norm = [_normalize_status(v) for v in statuses if str(v or "").strip()]
    # 条件分岐: `not norm` を満たす経路を評価する。
    if not norm:
        return "reject"

    # 条件分岐: `any(v == "reject" for v in norm)` を満たす経路を評価する。

    if any(v == "reject" for v in norm):
        return "reject"

    # 条件分岐: `all(v == "pass" for v in norm)` を満たす経路を評価する。

    if all(v == "pass" for v in norm):
        return "pass"

    return "watch"


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


# 関数: `_status_from_abs_z` の入出力契約と処理意図を定義する。

def _status_from_abs_z(abs_z: Optional[float]) -> str:
    # 条件分岐: `abs_z is None or not np.isfinite(abs_z)` を満たす経路を評価する。
    if abs_z is None or not np.isfinite(abs_z):
        return "reject"

    # 条件分岐: `abs_z <= 2.0` を満たす経路を評価する。

    if abs_z <= 2.0:
        return "pass"

    # 条件分岐: `abs_z <= 3.0` を満たす経路を評価する。

    if abs_z <= 3.0:
        return "watch"

    return "reject"


# 関数: `_abs_z_minus_1` の入出力契約と処理意図を定義する。

def _abs_z_minus_1(beta_est: Optional[float], beta_sigma: Optional[float]) -> Optional[float]:
    # 条件分岐: `beta_est is None or beta_sigma is None` を満たす経路を評価する。
    if beta_est is None or beta_sigma is None:
        return None

    # 条件分岐: `not np.isfinite(beta_est) or not np.isfinite(beta_sigma) or beta_sigma <= 0.0` を満たす経路を評価する。

    if not np.isfinite(beta_est) or not np.isfinite(beta_sigma) or beta_sigma <= 0.0:
        return None

    denom = max(float(beta_sigma), 1e-30)
    return abs((float(beta_est) - 1.0) / denom)


# 関数: `_pair_abs_z_status` の入出力契約と処理意図を定義する。

def _pair_abs_z_status(
    beta_a: Optional[float],
    sigma_a: Optional[float],
    beta_b: Optional[float],
    sigma_b: Optional[float],
) -> tuple[Optional[float], str]:
    # 条件分岐: `any(v is None for v in (beta_a, sigma_a, beta_b, sigma_b))` を満たす経路を評価する。
    if any(v is None for v in (beta_a, sigma_a, beta_b, sigma_b)):
        return None, "reject"

    s1 = float(sigma_a or 0.0)
    s2 = float(sigma_b or 0.0)
    # 条件分岐: `s1 <= 0.0 or s2 <= 0.0` を満たす経路を評価する。
    if s1 <= 0.0 or s2 <= 0.0:
        return None, "reject"

    denom = math.sqrt(max((s1 * s1) + (s2 * s2), 1e-30))
    # 条件分岐: `denom <= 0.0 or not np.isfinite(denom)` を満たす経路を評価する。
    if denom <= 0.0 or not np.isfinite(denom):
        return None, "reject"

    abs_z = abs((float(beta_a) - float(beta_b)) / denom)
    return abs_z, _status_from_abs_z(abs_z)


# 関数: `_build_pair_decomposition_requirements` の入出力契約と処理意図を定義する。

def _build_pair_decomposition_requirements(
    beta_anchor: Optional[float],
    sigma_anchor: Optional[float],
    beta_candidate: Optional[float],
    sigma_candidate: Optional[float],
) -> Dict[str, Optional[float]]:
    delta_beta = None
    abs_delta_beta = None
    sigma_combined = None
    required_abs_delta_pass = None
    required_abs_delta_watch = None
    required_candidate_beta_min_pass = None
    required_candidate_beta_max_pass = None
    required_candidate_beta_min_watch = None
    required_candidate_beta_max_watch = None
    required_candidate_sigma_pass = None
    required_candidate_sigma_watch = None

    # 条件分岐: `all(v is not None for v in (beta_anchor, sigma_anchor, beta_candidate, sigma_candidate))` を満たす経路を評価する。
    if all(v is not None for v in (beta_anchor, sigma_anchor, beta_candidate, sigma_candidate)):
        b_a = float(beta_anchor or 0.0)
        s_a = float(sigma_anchor or 0.0)
        b_c = float(beta_candidate or 0.0)
        s_c = float(sigma_candidate or 0.0)
        # 条件分岐: `s_a > 0.0 and s_c > 0.0 and np.isfinite(s_a) and np.isfinite(s_c)` を満たす経路を評価する。
        if s_a > 0.0 and s_c > 0.0 and np.isfinite(s_a) and np.isfinite(s_c):
            delta_beta = float(b_a - b_c)
            abs_delta_beta = abs(delta_beta)
            sigma_combined = float(math.sqrt(max((s_a * s_a) + (s_c * s_c), 1e-30)))
            # 条件分岐: `sigma_combined > 0.0 and np.isfinite(sigma_combined)` を満たす経路を評価する。
            if sigma_combined > 0.0 and np.isfinite(sigma_combined):
                required_abs_delta_pass = 2.0 * sigma_combined
                required_abs_delta_watch = 3.0 * sigma_combined
                required_candidate_beta_min_pass = b_a - required_abs_delta_pass
                required_candidate_beta_max_pass = b_a + required_abs_delta_pass
                required_candidate_beta_min_watch = b_a - required_abs_delta_watch
                required_candidate_beta_max_watch = b_a + required_abs_delta_watch
                required_candidate_sigma_pass = float(
                    math.sqrt(max((abs_delta_beta / 2.0) ** 2 - (s_a * s_a), 0.0))
                )
                required_candidate_sigma_watch = float(
                    math.sqrt(max((abs_delta_beta / 3.0) ** 2 - (s_a * s_a), 0.0))
                )

    return {
        "delta_beta_anchor_minus_candidate": delta_beta,
        "abs_delta_beta": abs_delta_beta,
        "sigma_combined": sigma_combined,
        "required_abs_delta_pass": required_abs_delta_pass,
        "required_abs_delta_watch": required_abs_delta_watch,
        "required_candidate_beta_min_pass": required_candidate_beta_min_pass,
        "required_candidate_beta_max_pass": required_candidate_beta_max_pass,
        "required_candidate_beta_min_watch": required_candidate_beta_min_watch,
        "required_candidate_beta_max_watch": required_candidate_beta_max_watch,
        "required_candidate_sigma_pass": required_candidate_sigma_pass,
        "required_candidate_sigma_watch": required_candidate_sigma_watch,
    }


# 関数: `_fuse_beta` の入出力契約と処理意図を定義する。

def _fuse_beta(components: Sequence[tuple[str, Optional[float], Optional[float]]]) -> tuple[Optional[float], Optional[float], List[str]]:
    weighted_terms: List[tuple[float, float]] = []
    used_labels: List[str] = []
    for label, beta_val, sigma_val in components:
        # 条件分岐: `beta_val is None or sigma_val is None` を満たす経路を評価する。
        if beta_val is None or sigma_val is None:
            continue

        s = float(sigma_val)
        # 条件分岐: `s <= 0.0 or not np.isfinite(s)` を満たす経路を評価する。
        if s <= 0.0 or not np.isfinite(s):
            continue

        w = 1.0 / (s * s)
        weighted_terms.append((float(beta_val), w))
        used_labels.append(str(label))

    # 条件分岐: `not weighted_terms` を満たす経路を評価する。

    if not weighted_terms:
        return None, None, used_labels

    w_sum = float(sum(w for _, w in weighted_terms))
    # 条件分岐: `w_sum <= 0.0 or not np.isfinite(w_sum)` を満たす経路を評価する。
    if w_sum <= 0.0 or not np.isfinite(w_sum):
        return None, None, used_labels

    beta_est = float(sum(beta * w for beta, w in weighted_terms) / w_sum)
    beta_sigma = float(math.sqrt(1.0 / w_sum))
    return beta_est, beta_sigma, used_labels


# 関数: `_build_policy_rows` の入出力契約と処理意図を定義する。

def _build_policy_rows(registry: Dict[str, Any]) -> List[Dict[str, Any]]:
    vlbi = registry.get("vlbi") if isinstance(registry.get("vlbi"), dict) else {}
    llr = registry.get("llr") if isinstance(registry.get("llr"), dict) else {}
    messenger = registry.get("messenger") if isinstance(registry.get("messenger"), dict) else {}
    cross = registry.get("cross_channel") if isinstance(registry.get("cross_channel"), dict) else {}
    beta_terminal = registry.get("beta_terminal") if isinstance(registry.get("beta_terminal"), dict) else {}
    policy_matrix = beta_terminal.get("policy_matrix") if isinstance(beta_terminal.get("policy_matrix"), dict) else {}
    policy_d_registry = (
        policy_matrix.get("policy_D_exclude_plus_messenger_fusion")
        if isinstance(policy_matrix.get("policy_D_exclude_plus_messenger_fusion"), dict)
        else {}
    )
    policy_d_pair_registry = (
        policy_d_registry.get("pair_decomposition") if isinstance(policy_d_registry.get("pair_decomposition"), dict) else {}
    )

    subset_refit_eligible = bool(vlbi.get("subset_refit_eligible", True))
    subset_refit_reason = str(vlbi.get("subset_refit_reason") or "")

    llr_beta = _to_float(beta_terminal.get("beta_llr_est"))
    llr_sigma = _to_float(beta_terminal.get("beta_llr_sigma"))
    messenger_beta = _to_float(messenger.get("beta_primary_est"))
    messenger_sigma = _to_float(messenger.get("beta_primary_sigma"))
    combined_beta = _to_float(beta_terminal.get("beta_combined_est"))
    combined_sigma = _to_float(beta_terminal.get("beta_combined_sigma"))

    llr_abs_z_minus_1 = _abs_z_minus_1(llr_beta, llr_sigma)
    messenger_abs_z_minus_1 = _abs_z_minus_1(messenger_beta, messenger_sigma)
    combined_abs_z_minus_1 = _abs_z_minus_1(combined_beta, combined_sigma)

    llr_bias_status = _normalize_status(llr.get("bias_audit_status"))
    vlbi_bias_status = _normalize_status(vlbi.get("bias_audit_status"))
    messenger_bias_status = _normalize_status(messenger.get("bias_audit_status"))
    llr_promotion_status = _normalize_status(llr.get("promotion_status"))
    cross_consistency_raw = _normalize_status(cross.get("beta_consistency_status"))
    messenger_stage_j_status = _normalize_status(messenger.get("stage_j_status"))
    messenger_replay_status = _normalize_status(messenger.get("stage_e_replay_status"))
    messenger_eligible = bool(messenger.get("comparator_eligible", False))

    has_llr = llr_beta is not None and llr_sigma is not None and float(llr_sigma) > 0.0
    has_messenger = messenger_beta is not None and messenger_sigma is not None and float(messenger_sigma) > 0.0
    has_vlbi = (
        _to_float(beta_terminal.get("beta_vlbi_est")) is not None
        and _to_float(beta_terminal.get("beta_vlbi_sigma")) is not None
        and float(_to_float(beta_terminal.get("beta_vlbi_sigma")) or 0.0) > 0.0
    )

    policies = [
        {
            "policy_id": "policy_A_hard_reject_keep",
            "policy_label": "A: hard reject keep",
            "exclude_comparator_if_ineligible": False,
            "require_llr_promotion_pass": False,
            "use_messenger_fusion": False,
            "messenger_only": False,
        },
        {
            "policy_id": "policy_B_exclude_comparator_when_ineligible",
            "policy_label": "B: exclude comparator",
            "exclude_comparator_if_ineligible": True,
            "require_llr_promotion_pass": False,
            "use_messenger_fusion": False,
            "messenger_only": False,
        },
        {
            "policy_id": "policy_C_exclude_plus_promotion_pass",
            "policy_label": "C: exclude + promotion-pass",
            "exclude_comparator_if_ineligible": True,
            "require_llr_promotion_pass": True,
            "use_messenger_fusion": False,
            "messenger_only": False,
        },
        {
            "policy_id": "policy_D_exclude_plus_messenger_fusion",
            "policy_label": "D: exclude + LLR+MESSENGER",
            "exclude_comparator_if_ineligible": True,
            "require_llr_promotion_pass": False,
            "use_messenger_fusion": True,
            "messenger_only": False,
        },
        {
            "policy_id": "policy_E_exclude_messenger_only",
            "policy_label": "E: exclude + MESSENGER-only",
            "exclude_comparator_if_ineligible": True,
            "require_llr_promotion_pass": False,
            "use_messenger_fusion": False,
            "messenger_only": True,
        },
    ]

    rows: List[Dict[str, Any]] = []
    for policy in policies:
        exclude_comparator = bool(policy.get("exclude_comparator_if_ineligible"))
        require_promotion = bool(policy.get("require_llr_promotion_pass"))
        use_messenger_fusion = bool(policy.get("use_messenger_fusion"))
        messenger_only = bool(policy.get("messenger_only"))
        comparator_excluded = subset_refit_eligible is False and exclude_comparator

        # 条件分岐: `messenger_only` を満たす経路を評価する。
        if messenger_only:
            availability_status = "pass" if (has_messenger and messenger_eligible) else "reject"
        # 条件分岐: 前段条件が不成立で、`exclude_comparator and use_messenger_fusion` を追加評価する。
        elif exclude_comparator and use_messenger_fusion:
            llr_or_messenger_ready = has_llr or (has_messenger and messenger_eligible)
            availability_status = "pass" if llr_or_messenger_ready else "reject"
        # 条件分岐: 前段条件が不成立で、`exclude_comparator` を追加評価する。
        elif exclude_comparator:
            availability_status = "pass" if has_llr else "reject"
        else:
            availability_status = "pass" if (has_llr and has_vlbi) else "reject"

        # 条件分岐: `subset_refit_eligible is False and not exclude_comparator` を満たす経路を評価する。

        if subset_refit_eligible is False and not exclude_comparator:
            cross_consistency_status = "reject"
            cross_consistency_mode = "hard_reject_ineligible"
        # 条件分岐: 前段条件が不成立で、`comparator_excluded` を追加評価する。
        elif comparator_excluded:
            cross_consistency_status = "pass"
            cross_consistency_mode = "excluded_due_to_ineligible"
        else:
            cross_consistency_status = cross_consistency_raw
            cross_consistency_mode = "active"

        channels_used: List[str] = []
        pairwise_abs_z_l_m: Optional[float] = None
        pairwise_status_l_m = "not_applicable"
        pair_abs_delta_beta_l_m: Optional[float] = None
        pair_sigma_combined_l_m: Optional[float] = None
        pair_required_abs_delta_pass: Optional[float] = None
        pair_required_abs_delta_watch: Optional[float] = None
        pair_required_messenger_beta_min_pass: Optional[float] = None
        pair_required_messenger_beta_max_pass: Optional[float] = None
        pair_required_messenger_beta_min_watch: Optional[float] = None
        pair_required_messenger_beta_max_watch: Optional[float] = None
        pair_required_messenger_sigma_pass: Optional[float] = None
        pair_required_messenger_sigma_watch: Optional[float] = None
        pair_watch_gate_status = "not_applicable"
        pair_pass_gate_status = "not_applicable"
        pair_gap_to_watch_z: Optional[float] = None
        pair_gap_to_pass_z: Optional[float] = None
        pair_gap_abs_delta_watch: Optional[float] = None
        pair_gap_abs_delta_pass: Optional[float] = None
        abs_z_minus_1: Optional[float]

        # 条件分岐: `comparator_excluded and messenger_only` を満たす経路を評価する。
        if comparator_excluded and messenger_only:
            beta_source = "messenger_only"
            beta_est = messenger_beta
            beta_sigma = messenger_sigma
            abs_z_minus_1 = messenger_abs_z_minus_1
            beta_minus_1_status = _status_from_abs_z(abs_z_minus_1)
            bias_scope = "messenger_only"
            bias_audit_status = _combine_status([messenger_bias_status, messenger_stage_j_status, messenger_replay_status])
            channels_used = ["messenger"] if availability_status == "pass" else []
        # 条件分岐: 前段条件が不成立で、`comparator_excluded and use_messenger_fusion` を追加評価する。
        elif comparator_excluded and use_messenger_fusion:
            beta_source = "llr_messenger_fused"
            beta_est, beta_sigma, channels_used = _fuse_beta(
                [
                    ("llr", llr_beta, llr_sigma),
                    ("messenger", messenger_beta if messenger_eligible else None, messenger_sigma if messenger_eligible else None),
                ]
            )
            abs_z_minus_1 = _abs_z_minus_1(beta_est, beta_sigma)
            beta_minus_1_status = _status_from_abs_z(abs_z_minus_1)
            pairwise_abs_z_l_m, pairwise_status_l_m = _pair_abs_z_status(
                llr_beta,
                llr_sigma,
                messenger_beta if messenger_eligible else None,
                messenger_sigma if messenger_eligible else None,
            )
            pair_req = (
                policy_d_pair_registry
                if policy_d_pair_registry and policy.get("policy_id") == "policy_D_exclude_plus_messenger_fusion"
                else _build_pair_decomposition_requirements(
                    beta_anchor=llr_beta,
                    sigma_anchor=llr_sigma,
                    beta_candidate=(messenger_beta if messenger_eligible else None),
                    sigma_candidate=(messenger_sigma if messenger_eligible else None),
                )
            )
            pair_abs_delta_beta_l_m = _to_float(pair_req.get("abs_delta_beta"))
            pair_sigma_combined_l_m = _to_float(pair_req.get("sigma_combined"))
            pair_required_abs_delta_pass = _to_float(pair_req.get("required_abs_delta_pass"))
            pair_required_abs_delta_watch = _to_float(pair_req.get("required_abs_delta_watch"))
            pair_required_messenger_beta_min_pass = _to_float(pair_req.get("required_candidate_beta_min_pass"))
            pair_required_messenger_beta_max_pass = _to_float(pair_req.get("required_candidate_beta_max_pass"))
            pair_required_messenger_beta_min_watch = _to_float(pair_req.get("required_candidate_beta_min_watch"))
            pair_required_messenger_beta_max_watch = _to_float(pair_req.get("required_candidate_beta_max_watch"))
            pair_required_messenger_sigma_pass = _to_float(pair_req.get("required_candidate_sigma_pass"))
            pair_required_messenger_sigma_watch = _to_float(pair_req.get("required_candidate_sigma_watch"))
            pair_watch_gate_status = _normalize_status(pair_req.get("watch_gate_status"))
            pair_pass_gate_status = _normalize_status(pair_req.get("pass_gate_status"))
            pair_gap_to_watch_z = _to_float(pair_req.get("gap_to_watch_z"))
            pair_gap_to_pass_z = _to_float(pair_req.get("gap_to_pass_z"))
            pair_gap_abs_delta_watch = _to_float(pair_req.get("gap_abs_delta_watch"))
            pair_gap_abs_delta_pass = _to_float(pair_req.get("gap_abs_delta_pass"))
            # 条件分岐: `"llr" in channels_used and "messenger" in channels_used` を満たす経路を評価する。
            if "llr" in channels_used and "messenger" in channels_used:
                cross_consistency_status = _combine_status([cross_consistency_status, pairwise_status_l_m])
                cross_consistency_mode = f"{cross_consistency_mode}+llr_messenger_pair"

            bias_scope = "llr_plus_messenger"
            bias_audit_status = _combine_status(
                [
                    llr_bias_status if "llr" in channels_used else "pass",
                    messenger_bias_status if "messenger" in channels_used else "pass",
                    messenger_stage_j_status if "messenger" in channels_used else "pass",
                    messenger_replay_status if "messenger" in channels_used else "pass",
                ]
            )
            # 条件分岐: `beta_est is None or beta_sigma is None` を満たす経路を評価する。
            if beta_est is None or beta_sigma is None:
                beta_minus_1_status = "reject"
        # 条件分岐: 前段条件が不成立で、`comparator_excluded` を追加評価する。
        elif comparator_excluded:
            beta_source = "llr_only"
            beta_est = llr_beta
            beta_sigma = llr_sigma
            abs_z_minus_1 = llr_abs_z_minus_1
            beta_minus_1_status = _status_from_abs_z(abs_z_minus_1)
            bias_scope = "llr_only"
            bias_audit_status = llr_bias_status
            channels_used = ["llr"] if availability_status == "pass" else []
        else:
            beta_source = "combined"
            beta_est = combined_beta
            beta_sigma = combined_sigma
            abs_z_minus_1 = combined_abs_z_minus_1
            beta_minus_1_status = _status_from_abs_z(abs_z_minus_1)
            bias_scope = "vlbi_plus_llr"
            bias_audit_status = _combine_status([vlbi_bias_status, llr_bias_status])
            channels_used = ["vlbi", "llr"] if availability_status == "pass" else []

        # 条件分岐: `require_promotion and beta_source != "messenger_only"` を満たす経路を評価する。

        if require_promotion and beta_source != "messenger_only":
            promotion_status = llr_promotion_status
            gate_statuses = [
                availability_status,
                cross_consistency_status,
                beta_minus_1_status,
                bias_audit_status,
                promotion_status,
            ]
        else:
            promotion_status = "not_applicable"
            gate_statuses = [
                availability_status,
                cross_consistency_status,
                beta_minus_1_status,
                bias_audit_status,
            ]

        terminal_status = _combine_status(gate_statuses)
        notes: List[str] = []
        # 条件分岐: `subset_refit_eligible is False` を満たす経路を評価する。
        if subset_refit_eligible is False:
            notes.append(f"subset_refit_eligible=false ({subset_refit_reason})")

        # 条件分岐: `comparator_excluded` を満たす経路を評価する。

        if comparator_excluded:
            notes.append("cross comparator excluded in terminal gate")

        # 条件分岐: `require_promotion` を満たす経路を評価する。

        if require_promotion:
            notes.append("LLR promotion pass required")

        # 条件分岐: `use_messenger_fusion` を満たす経路を評価する。

        if use_messenger_fusion:
            notes.append("messenger fusion enabled")

        # 条件分岐: `messenger_only` を満たす経路を評価する。

        if messenger_only:
            notes.append("messenger-only fallback")

        # 条件分岐: `pairwise_abs_z_l_m is not None` を満たす経路を評価する。

        if pairwise_abs_z_l_m is not None:
            notes.append(f"abs_z(llr-messenger)={pairwise_abs_z_l_m:.3f}")

        # 条件分岐: `pair_required_abs_delta_watch is not None and pair_abs_delta_beta_l_m is not...` を満たす経路を評価する。

        if pair_required_abs_delta_watch is not None and pair_abs_delta_beta_l_m is not None:
            notes.append(f"abs_delta={pair_abs_delta_beta_l_m:.3f}, watch<= {pair_required_abs_delta_watch:.3f}")

        rows.append(
            {
                "policy_id": policy["policy_id"],
                "policy_label": policy["policy_label"],
                "subset_refit_eligible": subset_refit_eligible,
                "comparator_excluded": comparator_excluded,
                "comparator_mode": cross_consistency_mode,
                "beta_source": beta_source,
                "availability_status": availability_status,
                "cross_consistency_status": cross_consistency_status,
                "beta_minus_1_status": beta_minus_1_status,
                "bias_audit_scope": bias_scope,
                "bias_audit_status": bias_audit_status,
                "promotion_requirement": "required" if require_promotion else "none",
                "promotion_status": promotion_status,
                "terminal_status": terminal_status,
                "beta_est": beta_est,
                "beta_sigma": beta_sigma,
                "abs_z_beta_minus_1": abs_z_minus_1,
                "channels_used": ",".join(channels_used),
                "pair_abs_z_llr_messenger": pairwise_abs_z_l_m,
                "pair_status_llr_messenger": pairwise_status_l_m,
                "pair_abs_delta_beta_llr_messenger": pair_abs_delta_beta_l_m,
                "pair_sigma_combined_llr_messenger": pair_sigma_combined_l_m,
                "pair_required_abs_delta_pass": pair_required_abs_delta_pass,
                "pair_required_abs_delta_watch": pair_required_abs_delta_watch,
                "pair_required_messenger_beta_min_pass": pair_required_messenger_beta_min_pass,
                "pair_required_messenger_beta_max_pass": pair_required_messenger_beta_max_pass,
                "pair_required_messenger_beta_min_watch": pair_required_messenger_beta_min_watch,
                "pair_required_messenger_beta_max_watch": pair_required_messenger_beta_max_watch,
                "pair_required_messenger_sigma_pass": pair_required_messenger_sigma_pass,
                "pair_required_messenger_sigma_watch": pair_required_messenger_sigma_watch,
                "pair_watch_gate_status": pair_watch_gate_status,
                "pair_pass_gate_status": pair_pass_gate_status,
                "pair_gap_to_watch_z": pair_gap_to_watch_z,
                "pair_gap_to_pass_z": pair_gap_to_pass_z,
                "pair_gap_abs_delta_watch": pair_gap_abs_delta_watch,
                "pair_gap_abs_delta_pass": pair_gap_abs_delta_pass,
                "note": " / ".join(notes),
            }
        )

    return rows


# 関数: `_write_csv_rows` の入出力契約と処理意図を定義する。

def _write_csv_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "policy_id",
        "policy_label",
        "subset_refit_eligible",
        "comparator_excluded",
        "comparator_mode",
        "beta_source",
        "availability_status",
        "cross_consistency_status",
        "beta_minus_1_status",
        "bias_audit_scope",
        "bias_audit_status",
        "promotion_requirement",
        "promotion_status",
        "terminal_status",
        "beta_est",
        "beta_sigma",
        "abs_z_beta_minus_1",
        "channels_used",
        "pair_abs_z_llr_messenger",
        "pair_status_llr_messenger",
        "pair_abs_delta_beta_llr_messenger",
        "pair_sigma_combined_llr_messenger",
        "pair_required_abs_delta_pass",
        "pair_required_abs_delta_watch",
        "pair_required_messenger_beta_min_pass",
        "pair_required_messenger_beta_max_pass",
        "pair_required_messenger_beta_min_watch",
        "pair_required_messenger_beta_max_watch",
        "pair_required_messenger_sigma_pass",
        "pair_required_messenger_sigma_watch",
        "pair_watch_gate_status",
        "pair_pass_gate_status",
        "pair_gap_to_watch_z",
        "pair_gap_to_pass_z",
        "pair_gap_abs_delta_watch",
        "pair_gap_abs_delta_pass",
        "note",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


# 関数: `_write_plot` の入出力契約と処理意図を定義する。

def _write_plot(rows: Sequence[Dict[str, Any]], out_pdf: Path, out_png: Path) -> None:
    labels = [str(r.get("policy_label") or r.get("policy_id") or "") for r in rows]
    statuses = [_normalize_status(r.get("terminal_status")) for r in rows]
    scores = [_status_to_score(s) for s in statuses]
    colors = [_status_color(s) for s in statuses]
    notes = [
        f"{r.get('terminal_status')} | beta_source={r.get('beta_source')} | "
        f"cross={r.get('cross_consistency_status')} | bias={r.get('bias_audit_status')}"
        for r in rows
    ]

    fig, ax = plt.subplots(figsize=(11.8, 4.8))
    y = np.arange(len(labels), dtype=float)
    ax.barh(y, scores, color=colors, alpha=0.92)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 3.0)
    ax.axvline(1.0, color="#999999", linestyle="--", linewidth=1.0)
    ax.axvline(2.0, color="#999999", linestyle="--", linewidth=1.0)
    ax.grid(axis="x", alpha=0.2)
    ax.set_xlabel("status score (pass=0.5, watch=1.5, reject=2.8)")
    ax.set_title("Step 8.7.47.21+8.7.48.11: beta terminal policy sensitivity")

    for i, note in enumerate(notes):
        ax.text(0.02, i, note, va="center", ha="left", fontsize=9.2)

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
    ap = argparse.ArgumentParser(
        description=(
            "Audit beta-terminal policy sensitivity for comparator ineligible "
            "(Roadmap 8.7.47.21 + 8.7.48.11)."
        )
    )
    ap.add_argument(
        "--registry-json",
        type=str,
        default="",
        help="Path to beta_cross_channel_registry.json (optional; defaults to public/private search).",
    )
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

    registry_path: Optional[Path] = None
    # 条件分岐: `str(args.registry_json or "").strip()` を満たす経路を評価する。
    if str(args.registry_json or "").strip():
        p = Path(str(args.registry_json))
        registry_path = p if p.is_absolute() else (_ROOT / p).resolve()
    else:
        registry_path = _first_existing(
            [
                _ROOT / "output" / "public" / "summary" / "beta_cross_channel_registry.json",
                _ROOT / "output" / "private" / "summary" / "beta_cross_channel_registry.json",
            ]
        )

    # 条件分岐: `registry_path is None or not registry_path.exists()` を満たす経路を評価する。

    if registry_path is None or not registry_path.exists():
        raise FileNotFoundError("beta_cross_channel_registry.json was not found.")

    registry = _read_json(registry_path)
    rows = _build_policy_rows(registry)
    rows_sorted = sorted(rows, key=lambda r: str(r.get("policy_id", "")))

    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "beta_terminal_comparator_policy_sensitivity.json"
    out_csv = out_dir / "beta_terminal_comparator_policy_sensitivity.csv"
    out_pdf = out_dir / "beta_terminal_comparator_policy_sensitivity.pdf"
    out_png = out_dir / "beta_terminal_comparator_policy_sensitivity.png"

    by_id = {str(r.get("policy_id")): r for r in rows_sorted}
    baseline = by_id.get("policy_A_hard_reject_keep")
    comparator_excluded = by_id.get("policy_B_exclude_comparator_when_ineligible")
    promoted = by_id.get("policy_C_exclude_plus_promotion_pass")
    messenger_fused = by_id.get("policy_D_exclude_plus_messenger_fusion")
    messenger_only = by_id.get("policy_E_exclude_messenger_only")

    summary = {
        "baseline_terminal_status": (baseline or {}).get("terminal_status"),
        "exclude_terminal_status": (comparator_excluded or {}).get("terminal_status"),
        "exclude_plus_promotion_terminal_status": (promoted or {}).get("terminal_status"),
        "exclude_plus_messenger_fusion_terminal_status": (messenger_fused or {}).get("terminal_status"),
        "exclude_plus_messenger_only_terminal_status": (messenger_only or {}).get("terminal_status"),
        "status_shift_baseline_to_exclude": (
            f"{(baseline or {}).get('terminal_status')} -> {(comparator_excluded or {}).get('terminal_status')}"
        ),
        "status_shift_exclude_to_promotion": (
            f"{(comparator_excluded or {}).get('terminal_status')} -> {(promoted or {}).get('terminal_status')}"
        ),
        "status_shift_exclude_to_messenger_fusion": (
            f"{(comparator_excluded or {}).get('terminal_status')} -> {(messenger_fused or {}).get('terminal_status')}"
        ),
        "status_shift_messenger_fusion_to_messenger_only": (
            f"{(messenger_fused or {}).get('terminal_status')} -> {(messenger_only or {}).get('terminal_status')}"
        ),
        "policy_d_pair_abs_z_llr_messenger": (messenger_fused or {}).get("pair_abs_z_llr_messenger"),
        "policy_d_pair_status_llr_messenger": (messenger_fused or {}).get("pair_status_llr_messenger"),
        "policy_d_pair_abs_delta_beta": (messenger_fused or {}).get("pair_abs_delta_beta_llr_messenger"),
        "policy_d_pair_required_abs_delta_watch": (messenger_fused or {}).get("pair_required_abs_delta_watch"),
        "policy_d_required_messenger_beta_watch_range": [
            (messenger_fused or {}).get("pair_required_messenger_beta_min_watch"),
            (messenger_fused or {}).get("pair_required_messenger_beta_max_watch"),
        ],
        "policy_d_required_messenger_sigma_watch": (messenger_fused or {}).get("pair_required_messenger_sigma_watch"),
    }

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase": {"step": "8.7.47.21+8.7.48.11"},
        "inputs": {
            "registry_json": _safe_rel(registry_path, _ROOT),
        },
        "policy_scope": {
            "subset_refit_eligible": bool((registry.get("vlbi") or {}).get("subset_refit_eligible", True)),
            "subset_refit_reason": str((registry.get("vlbi") or {}).get("subset_refit_reason") or ""),
            "messenger_comparator_eligible": bool((registry.get("messenger") or {}).get("comparator_eligible", False)),
        },
        "rows": rows_sorted,
        "summary": summary,
        "outputs": {
            "json": _safe_rel(out_json, _ROOT),
            "csv": _safe_rel(out_csv, _ROOT),
            "pdf": _safe_rel(out_pdf, _ROOT),
            "png": _safe_rel(out_png, _ROOT),
        },
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv_rows(out_csv, rows_sorted)
    _write_plot(rows_sorted, out_pdf=out_pdf, out_png=out_png)

    produced = [out_json, out_csv, out_pdf, out_png]
    synced = _sync_outputs_to_public(produced, private_root=out_dir, public_root=public_dir)
    print(f"[ok] wrote: {out_json}")
    print(f"[ok] wrote: {out_csv}")
    print(f"[ok] wrote: {out_pdf}")
    print(f"[ok] wrote: {out_png}")
    print(f"[ok] synced_to_public: {len(synced)} files")
    print(
        "[summary] "
        f"baseline={(baseline or {}).get('terminal_status')} "
        f"exclude={(comparator_excluded or {}).get('terminal_status')} "
        f"exclude_plus_promotion={(promoted or {}).get('terminal_status')} "
        f"exclude_plus_messenger_fusion={(messenger_fused or {}).get('terminal_status')} "
        f"exclude_plus_messenger_only={(messenger_only or {}).get('terminal_status')}"
    )
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
