#!/usr/bin/env python3
"""
beta_cross_channel_registry.py

Roadmap Step 8.7.47.7-8.7.47.8, 8.7.47.20, 8.7.48.12:
- Connect LLR/VLBI beta-channel outcomes to Part IV scoreboard interface.
- Freeze a machine-readable cross-channel terminal decision.
 - 8.7.47.11: Separate VLBI/LLR bias-audit components into machine-readable gates.
 - 8.7.48: Add MESSENGER channel row and policy notes to the terminal registry.
 - 8.7.48.12: Fix terminal policy selection when VLBI comparator is ineligible.

Inputs (public-first fallback):
- output/public/vlbi/vlbi_high_sensitivity_threshold_sweep_metrics.json
- output/public/vlbi/vlbi_beta_stable_source_refit_metrics.json
- output/public/vlbi/vlbi_beta_timeband_stratified_refit_metrics.json
- output/public/vlbi/vlbi_beta_watchpack_apply_chain_metrics.json
- output/public/vlbi/vlbi_beta_cross_consistency_subset_refit_metrics.json
- output/public/llr/llr_kappa_llr_metrics.json
- output/public/mercury/messenger_beta_stage_j_final_gate_metrics.json
- output/public/mercury/messenger_beta_stage_i_nuisance_sensitivity_metrics.json
- output/public/mercury/messenger_beta_stage_h_segmentation_metrics.json
- output/public/mercury/messenger_beta_stage_e_tnf_replay_metrics.json
- output/public/mercury/messenger_beta_stage_d_joint_metrics.json

Outputs (default: output/private/summary and synced to output/public/summary):
- beta_cross_channel_registry.json
- beta_cross_channel_registry.csv
- beta_cross_channel_registry.pdf
- beta_cross_channel_registry.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


_ROOT = Path(__file__).resolve().parents[2]


# 関数: `_safe_rel` の入出力契約と処理意図を定義する。
def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve()).replace("\\", "/")


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_first_existing` の入出力契約と処理意図を定義する。

def _first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for p in paths:
        # 条件分岐: `p.exists()` を満たす経路を評価する。
        if p.exists():
            return p

    return None


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


# 関数: `_extract_stage_i_beta_decomposition` の入出力契約と処理意図を定義する。

def _extract_stage_i_beta_decomposition(stage_i: Dict[str, Any]) -> Dict[str, Any]:
    branch_summary_rows = stage_i.get("branch_summary") if isinstance(stage_i.get("branch_summary"), list) else []
    decomposition_rows = (
        stage_i.get("beta_dyn_decomposition") if isinstance(stage_i.get("beta_dyn_decomposition"), list) else []
    )
    branch_summary_by_branch: Dict[str, Dict[str, Any]] = {}
    for row in branch_summary_rows:
        # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
        if not isinstance(row, dict):
            continue

        branch = str(row.get("branch") or "").strip().lower()
        # 条件分岐: `not branch` を満たす経路を評価する。
        if not branch:
            continue

        branch_summary_by_branch[branch] = {
            "status": _normalize_status(row.get("status")),
            "max_abs_z_delta_core": _to_float(row.get("max_abs_z_delta_core")),
            "max_abs_z_delta_all": _to_float(row.get("max_abs_z_delta_all")),
            "beta_base": _to_float(row.get("beta_base")),
            "beta_sigma_base": _to_float(row.get("beta_sigma_base")),
            "n_scenarios": int(row.get("n_scenarios") or 0),
            "n_core_scenarios": int(row.get("n_core_scenarios") or 0),
            "n_diagnostic_scenarios": int(row.get("n_diagnostic_scenarios") or 0),
        }

    decomposition_by_branch: Dict[str, Dict[str, Any]] = {}
    for row in decomposition_rows:
        # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
        if not isinstance(row, dict):
            continue

        branch = str(row.get("branch") or "").strip().lower()
        # 条件分岐: `not branch` を満たす経路を評価する。
        if not branch:
            continue

        baseline = row.get("baseline") if isinstance(row.get("baseline"), dict) else {}
        best_abs_z = row.get("best_abs_z") if isinstance(row.get("best_abs_z"), dict) else {}
        max_shift = row.get("max_shift_vs_baseline") if isinstance(row.get("max_shift_vs_baseline"), dict) else {}
        decomposition_by_branch[branch] = {
            "baseline_abs_z_beta_minus_1": _to_float(baseline.get("abs_z_beta_minus_1")),
            "baseline_beta_dyn": _to_float(baseline.get("beta_dyn")),
            "baseline_beta_sigma": _to_float(baseline.get("beta_sigma")),
            "best_abs_z_beta_minus_1": _to_float(best_abs_z.get("abs_z_beta_minus_1")),
            "best_abs_z_beta_dyn": _to_float(best_abs_z.get("beta_dyn")),
            "best_abs_z_beta_sigma": _to_float(best_abs_z.get("beta_sigma")),
            "best_abs_z_scenario_id": str(best_abs_z.get("scenario_id") or ""),
            "best_abs_z_scenario_group": str(best_abs_z.get("scenario_group") or ""),
            "best_abs_z_delta_vs_baseline_z": _to_float(best_abs_z.get("delta_vs_baseline_z")),
            "max_shift_scenario_id": str(max_shift.get("scenario_id") or ""),
            "max_shift_scenario_group": str(max_shift.get("scenario_group") or ""),
            "max_shift_abs_z_beta_minus_1": _to_float(max_shift.get("abs_z_beta_minus_1")),
            "max_shift_beta_dyn": _to_float(max_shift.get("beta_dyn")),
            "max_shift_beta_sigma": _to_float(max_shift.get("beta_sigma")),
            "max_shift_delta_vs_baseline_z": _to_float(max_shift.get("delta_vs_baseline_z")),
        }

    return {
        "branch_summary_by_branch": branch_summary_by_branch,
        "decomposition_by_branch": decomposition_by_branch,
        "raw_decomposition_rows": decomposition_rows,
    }


# 関数: `_build_stage_i_nuisance_priority` の入出力契約と処理意図を定義する。

def _build_stage_i_nuisance_priority(
    stage_i_policy: Dict[str, Any],
    stage_i_decomposition_by_branch: Dict[str, Any],
) -> Dict[str, Any]:
    scenarios = stage_i_policy.get("scenarios") if isinstance(stage_i_policy.get("scenarios"), list) else []
    scenario_meta: Dict[str, Dict[str, Any]] = {}
    for scenario in scenarios:
        # 条件分岐: `not isinstance(scenario, dict)` を満たす経路を評価する。
        if not isinstance(scenario, dict):
            continue

        sid = str(scenario.get("scenario_id") or "").strip()
        # 条件分岐: `not sid` を満たす経路を評価する。
        if not sid:
            continue

        scenario_meta[sid] = {
            "scenario_group": str(scenario.get("scenario_group") or ""),
            "note": str(scenario.get("note") or ""),
            "use_sun_quad_proxy": bool(scenario.get("use_sun_quad_proxy", False)),
            "use_transponder_quad": bool(scenario.get("use_transponder_quad", False)),
            "use_plasma_proxy": bool(scenario.get("use_plasma_proxy", False)),
            "use_srp_proxy": bool(scenario.get("use_srp_proxy", False)),
        }

    # 関数: `_recommended_action` の入出力契約と処理意図を定義する。

    def _recommended_action(scenario_id: str, scenario_group: str, meta: Dict[str, Any]) -> str:
        sid = str(scenario_id or "").lower()
        grp = str(scenario_group or "").lower()
        note = str(meta.get("note") or "").lower()
        use_plasma = bool(meta.get("use_plasma_proxy", False))
        use_transponder = bool(meta.get("use_transponder_quad", False))
        use_sunq = bool(meta.get("use_sun_quad_proxy", False))
        use_srp = bool(meta.get("use_srp_proxy", False))
        # 条件分岐: `use_plasma or "plasma" in sid or "plasma" in grp` を満たす経路を評価する。
        if use_plasma or "plasma" in sid or "plasma" in grp:
            return "prioritize_plasma_residual_modeling_and_media_controls"

        # 条件分岐: 前段条件が不成立で、`use_transponder or "transponder" in sid or "transponder" in grp` を追加評価する。

        if use_transponder or "transponder" in sid or "transponder" in grp:
            return "prioritize_transponder_drift_model_upgrade"

        # 条件分岐: 前段条件が不成立で、`"station_bias" in sid or "station_bias" in grp` を追加評価する。

        if "station_bias" in sid or "station_bias" in grp:
            return "tighten_station_bias_cap_and_station_level_calibration"

        # 条件分岐: 前段条件が不成立で、`use_sunq or "sunq" in sid or "sun_quadrupole" in grp` を追加評価する。

        if use_sunq or "sunq" in sid or "sun_quadrupole" in grp:
            return "audit_sun_quadrupole_like_proxy_vs_physical_solar_terms"

        # 条件分岐: 前段条件が不成立で、`use_srp or "srp" in sid or "srp" in grp` を追加評価する。

        if use_srp or "srp" in sid or "srp" in grp:
            return "audit_srp_like_proxy_vs_spacecraft_solar_radiation_pressure_model"

        # 条件分岐: 前段条件が不成立で、`"combined" in grp or "all_proxies" in sid or "combined" in note` を追加評価する。

        if "combined" in grp or "all_proxies" in sid or "combined" in note:
            return "run_joint_high_correlation_nuisance_stress_tests"

        return "review_branch_specific_nuisance_controls"

    mitigation_by_scenario: Dict[str, Dict[str, Any]] = {}
    stabilization_by_scenario: Dict[str, Dict[str, Any]] = {}
    for branch, row in stage_i_decomposition_by_branch.items():
        # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
        if not isinstance(row, dict):
            continue

        branch_name = str(branch or "").strip().lower()
        max_shift_sid = str(row.get("max_shift_scenario_id") or "").strip()
        max_shift_group = str(row.get("max_shift_scenario_group") or "").strip()
        max_shift_delta = _to_float(row.get("max_shift_delta_vs_baseline_z"))
        # 条件分岐: `max_shift_sid and max_shift_delta is not None` を満たす経路を評価する。
        if max_shift_sid and max_shift_delta is not None:
            meta = scenario_meta.get(max_shift_sid, {})
            group = max_shift_group or str(meta.get("scenario_group") or "")
            item = mitigation_by_scenario.get(max_shift_sid, {})
            # 条件分岐: `not item` を満たす経路を評価する。
            if not item:
                item = {
                    "scenario_id": max_shift_sid,
                    "scenario_group": group,
                    "branches": [],
                    "max_abs_z_delta": 0.0,
                    "sum_abs_z_delta": 0.0,
                }

            delta_abs = abs(float(max_shift_delta))
            item["max_abs_z_delta"] = float(max(float(item.get("max_abs_z_delta") or 0.0), delta_abs))
            item["sum_abs_z_delta"] = float((float(item.get("sum_abs_z_delta") or 0.0)) + delta_abs)
            branches = list(item.get("branches") or [])
            # 条件分岐: `branch_name and branch_name not in branches` を満たす経路を評価する。
            if branch_name and branch_name not in branches:
                branches.append(branch_name)

            item["branches"] = branches
            item["recommended_action"] = _recommended_action(max_shift_sid, group, meta)
            mitigation_by_scenario[max_shift_sid] = item

        best_sid = str(row.get("best_abs_z_scenario_id") or "").strip()
        best_group = str(row.get("best_abs_z_scenario_group") or "").strip()
        best_delta = _to_float(row.get("best_abs_z_delta_vs_baseline_z"))
        # 条件分岐: `best_sid and best_delta is not None` を満たす経路を評価する。
        if best_sid and best_delta is not None:
            meta = scenario_meta.get(best_sid, {})
            group = best_group or str(meta.get("scenario_group") or "")
            item = stabilization_by_scenario.get(best_sid, {})
            # 条件分岐: `not item` を満たす経路を評価する。
            if not item:
                item = {
                    "scenario_id": best_sid,
                    "scenario_group": group,
                    "branches": [],
                    "max_abs_z_improvement": 0.0,
                    "sum_abs_z_improvement": 0.0,
                }

            delta_abs = abs(float(best_delta))
            item["max_abs_z_improvement"] = float(max(float(item.get("max_abs_z_improvement") or 0.0), delta_abs))
            item["sum_abs_z_improvement"] = float((float(item.get("sum_abs_z_improvement") or 0.0)) + delta_abs)
            branches = list(item.get("branches") or [])
            # 条件分岐: `branch_name and branch_name not in branches` を満たす経路を評価する。
            if branch_name and branch_name not in branches:
                branches.append(branch_name)

            item["branches"] = branches
            item["recommended_action"] = _recommended_action(best_sid, group, meta)
            stabilization_by_scenario[best_sid] = item

    mitigation_priority = sorted(
        list(mitigation_by_scenario.values()),
        key=lambda x: (
            float(x.get("max_abs_z_delta") or 0.0),
            float(x.get("sum_abs_z_delta") or 0.0),
            len(list(x.get("branches") or [])),
        ),
        reverse=True,
    )
    stabilization_priority = sorted(
        list(stabilization_by_scenario.values()),
        key=lambda x: (
            float(x.get("max_abs_z_improvement") or 0.0),
            float(x.get("sum_abs_z_improvement") or 0.0),
            len(list(x.get("branches") or [])),
        ),
        reverse=True,
    )
    for idx, row in enumerate(mitigation_priority, start=1):
        row["priority_rank"] = idx

    for idx, row in enumerate(stabilization_priority, start=1):
        row["priority_rank"] = idx

    srp_proxy_registered = False
    srp_proxy_scenarios: List[str] = []
    for sid, meta in scenario_meta.items():
        sid_l = str(sid).lower()
        grp_l = str(meta.get("scenario_group") or "").lower()
        note_l = str(meta.get("note") or "").lower()
        use_srp = bool(meta.get("use_srp_proxy", False))
        # 条件分岐: `use_srp or "srp" in sid_l or "srp" in grp_l or "srp" in note_l` を満たす経路を評価する。
        if use_srp or "srp" in sid_l or "srp" in grp_l or "srp" in note_l:
            srp_proxy_registered = True
            srp_proxy_scenarios.append(str(sid))

    top_mitigation = mitigation_priority[0] if mitigation_priority else {}
    top_stabilization = stabilization_priority[0] if stabilization_priority else {}
    return {
        "method": "rank_by_stage_i_abs_z_shift_across_branches",
        "priority_status": ("pass" if mitigation_priority else "watch"),
        "mitigation_priority": mitigation_priority,
        "stabilization_priority": stabilization_priority,
        "top_mitigation_scenario_id": str(top_mitigation.get("scenario_id") or ""),
        "top_mitigation_recommended_action": str(top_mitigation.get("recommended_action") or ""),
        "top_stabilization_scenario_id": str(top_stabilization.get("scenario_id") or ""),
        "srp_proxy_registered": srp_proxy_registered,
        "srp_proxy_scenarios": sorted(set([s for s in srp_proxy_scenarios if str(s or "").strip()])),
        "srp_proxy_note": (
            "explicit_srp_proxy_registered_in_stage_i_policy"
            if srp_proxy_registered
            else "explicit_srp_proxy_not_found_in_stage_i_policy"
        ),
    }


# 関数: `_signed_shift_to_interval` の入出力契約と処理意図を定義する。

def _signed_shift_to_interval(value: Optional[float], lower: Optional[float], upper: Optional[float]) -> Optional[float]:
    # 条件分岐: `value is None or lower is None or upper is None` を満たす経路を評価する。
    if value is None or lower is None or upper is None:
        return None

    v = float(value)
    lo = float(lower)
    hi = float(upper)
    # 条件分岐: `not (np.isfinite(v) and np.isfinite(lo) and np.isfinite(hi))` を満たす経路を評価する。
    if not (np.isfinite(v) and np.isfinite(lo) and np.isfinite(hi)):
        return None

    # 条件分岐: `v < lo` を満たす経路を評価する。

    if v < lo:
        return lo - v

    # 条件分岐: `v > hi` を満たす経路を評価する。

    if v > hi:
        return hi - v

    return 0.0


# 関数: `_build_pair_decomposition` の入出力契約と処理意図を定義する。

def _build_pair_decomposition(
    beta_anchor: Optional[float],
    sigma_anchor: Optional[float],
    beta_candidate: Optional[float],
    sigma_candidate: Optional[float],
) -> Dict[str, Any]:
    delta_beta = None
    abs_delta_beta = None
    sigma_combined = None
    abs_z = None
    status = "reject"
    required_abs_delta_pass = None
    required_abs_delta_watch = None
    required_sigma_candidate_pass = None
    required_sigma_candidate_watch = None
    required_candidate_beta_min_pass = None
    required_candidate_beta_max_pass = None
    required_candidate_beta_min_watch = None
    required_candidate_beta_max_watch = None
    gap_to_watch_z = None
    gap_to_pass_z = None
    gap_abs_delta_watch = None
    gap_abs_delta_pass = None
    watch_gate_status = "reject"
    pass_gate_status = "reject"

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
                abs_z = abs_delta_beta / sigma_combined
                status = _status_from_abs_z(abs_z)
                required_abs_delta_pass = 2.0 * sigma_combined
                required_abs_delta_watch = 3.0 * sigma_combined
                required_candidate_beta_min_pass = b_a - required_abs_delta_pass
                required_candidate_beta_max_pass = b_a + required_abs_delta_pass
                required_candidate_beta_min_watch = b_a - required_abs_delta_watch
                required_candidate_beta_max_watch = b_a + required_abs_delta_watch
                required_sigma_candidate_pass = float(
                    math.sqrt(max((abs_delta_beta / 2.0) ** 2 - (s_a * s_a), 0.0))
                )
                required_sigma_candidate_watch = float(
                    math.sqrt(max((abs_delta_beta / 3.0) ** 2 - (s_a * s_a), 0.0))
                )
                gap_to_watch_z = abs_z - 3.0
                gap_to_pass_z = abs_z - 2.0
                gap_abs_delta_watch = abs_delta_beta - required_abs_delta_watch
                gap_abs_delta_pass = abs_delta_beta - required_abs_delta_pass
                watch_gate_status = "pass" if abs_z <= 3.0 else "reject"
                pass_gate_status = "pass" if abs_z <= 2.0 else "reject"

    return {
        "status": status,
        "delta_beta_anchor_minus_candidate": delta_beta,
        "abs_delta_beta": abs_delta_beta,
        "sigma_anchor": _to_float(sigma_anchor),
        "sigma_candidate": _to_float(sigma_candidate),
        "sigma_combined": sigma_combined,
        "abs_z_pair": abs_z,
        "required_abs_delta_pass": required_abs_delta_pass,
        "required_abs_delta_watch": required_abs_delta_watch,
        "required_candidate_beta_min_pass": required_candidate_beta_min_pass,
        "required_candidate_beta_max_pass": required_candidate_beta_max_pass,
        "required_candidate_beta_min_watch": required_candidate_beta_min_watch,
        "required_candidate_beta_max_watch": required_candidate_beta_max_watch,
        "required_candidate_sigma_pass": required_sigma_candidate_pass,
        "required_candidate_sigma_watch": required_sigma_candidate_watch,
        "watch_gate_status": watch_gate_status,
        "pass_gate_status": pass_gate_status,
        "gap_to_watch_z": gap_to_watch_z,
        "gap_to_pass_z": gap_to_pass_z,
        "gap_abs_delta_watch": gap_abs_delta_watch,
        "gap_abs_delta_pass": gap_abs_delta_pass,
    }


# 関数: `_build_policy_d_stage_i_mapping` の入出力契約と処理意図を定義する。

def _build_policy_d_stage_i_mapping(
    messenger: Dict[str, Any],
    pair_decomposition: Dict[str, Any],
) -> Dict[str, Any]:
    watch_min = _to_float(pair_decomposition.get("required_candidate_beta_min_watch"))
    watch_max = _to_float(pair_decomposition.get("required_candidate_beta_max_watch"))
    current_beta = _to_float(messenger.get("beta_dyn_diagnostic_est"))
    stage_i_decomp = (
        messenger.get("stage_i_beta_dyn_decomposition")
        if isinstance(messenger.get("stage_i_beta_dyn_decomposition"), dict)
        else {}
    )
    branch_rows: Dict[str, Any] = {}
    best_branch = ""
    best_shift_abs = None
    best_shift_signed = None
    for branch in ("odf", "tnf"):
        row = stage_i_decomp.get(branch) if isinstance(stage_i_decomp.get(branch), dict) else {}
        baseline_beta = _to_float(row.get("baseline_beta_dyn"))
        best_beta = _to_float(row.get("best_abs_z_beta_dyn"))
        max_shift_beta = _to_float(row.get("max_shift_beta_dyn"))
        shift_current = _signed_shift_to_interval(current_beta, watch_min, watch_max)
        shift_baseline = _signed_shift_to_interval(baseline_beta, watch_min, watch_max)
        shift_best = _signed_shift_to_interval(best_beta, watch_min, watch_max)
        shift_max_shift = _signed_shift_to_interval(max_shift_beta, watch_min, watch_max)
        baseline_to_best_shift = (
            (best_beta - baseline_beta)
            if baseline_beta is not None and best_beta is not None
            else None
        )
        baseline_to_max_shift = (
            (max_shift_beta - baseline_beta)
            if baseline_beta is not None and max_shift_beta is not None
            else None
        )
        branch_rows[branch] = {
            "baseline_beta_dyn": baseline_beta,
            "best_beta_dyn": best_beta,
            "max_shift_beta_dyn": max_shift_beta,
            "shift_needed_current_to_watch_range": shift_current,
            "shift_needed_baseline_to_watch_range": shift_baseline,
            "shift_needed_best_to_watch_range": shift_best,
            "shift_needed_max_shift_to_watch_range": shift_max_shift,
            "baseline_to_best_shift_beta": baseline_to_best_shift,
            "baseline_to_max_shift_shift_beta": baseline_to_max_shift,
            "baseline_abs_z_beta_minus_1": _to_float(row.get("baseline_abs_z_beta_minus_1")),
            "best_abs_z_beta_minus_1": _to_float(row.get("best_abs_z_beta_minus_1")),
            "max_shift_abs_z_beta_minus_1": _to_float(row.get("max_shift_abs_z_beta_minus_1")),
            "max_shift_delta_vs_baseline_z": _to_float(row.get("max_shift_delta_vs_baseline_z")),
            "best_abs_z_scenario_id": str(row.get("best_abs_z_scenario_id") or ""),
            "max_shift_scenario_id": str(row.get("max_shift_scenario_id") or ""),
        }
        # 条件分岐: `shift_best is not None` を満たす経路を評価する。
        if shift_best is not None:
            shift_abs = abs(float(shift_best))
            # 条件分岐: `best_shift_abs is None or shift_abs < best_shift_abs` を満たす経路を評価する。
            if best_shift_abs is None or shift_abs < best_shift_abs:
                best_shift_abs = shift_abs
                best_shift_signed = float(shift_best)
                best_branch = branch

    watch_reachable = bool(best_shift_abs is not None and best_shift_abs <= 1e-12)
    return {
        "mapping_beta_definition": "beta_dyn_diagnostic",
        "watch_range_min": watch_min,
        "watch_range_max": watch_max,
        "current_messenger_beta": current_beta,
        "shift_needed_current_to_watch_range": _signed_shift_to_interval(current_beta, watch_min, watch_max),
        "branch_candidates": branch_rows,
        "best_candidate_branch": best_branch,
        "best_candidate_shift_needed_to_watch_range": best_shift_signed,
        "watch_gate_reachable_with_stage_i_candidates": watch_reachable,
        "watch_gate_reachable_status": ("pass" if watch_reachable else "reject"),
    }


# 関数: `_extract_vlbi_channel` の入出力契約と処理意図を定義する。

def _extract_vlbi_channel(root: Path) -> Tuple[Dict[str, Any], List[str]]:
    threshold_path = _first_existing(
        [
            root / "output" / "public" / "vlbi" / "vlbi_high_sensitivity_threshold_sweep_metrics.json",
            root / "output" / "vlbi" / "vlbi_high_sensitivity_threshold_sweep_metrics.json",
        ]
    )
    stable_path = _first_existing(
        [
            root / "output" / "public" / "vlbi" / "vlbi_beta_stable_source_refit_metrics.json",
            root / "output" / "vlbi" / "vlbi_beta_stable_source_refit_metrics.json",
        ]
    )
    timeband_path = _first_existing(
        [
            root / "output" / "public" / "vlbi" / "vlbi_beta_timeband_stratified_refit_metrics.json",
            root / "output" / "vlbi" / "vlbi_beta_timeband_stratified_refit_metrics.json",
        ]
    )
    chain_path = _first_existing(
        [
            root / "output" / "public" / "vlbi" / "vlbi_beta_watchpack_apply_chain_metrics.json",
            root / "output" / "vlbi" / "vlbi_beta_watchpack_apply_chain_metrics.json",
        ]
    )
    subset_refit_path = _first_existing(
        [
            root / "output" / "public" / "vlbi" / "vlbi_beta_cross_consistency_subset_refit_metrics.json",
            root / "output" / "private" / "vlbi" / "vlbi_beta_cross_consistency_subset_refit_metrics.json",
            root / "output" / "vlbi" / "vlbi_beta_cross_consistency_subset_refit_metrics.json",
        ]
    )
    paths = [p for p in (threshold_path, stable_path, timeband_path, chain_path, subset_refit_path) if p is not None]
    # 条件分岐: `not paths` を満たす経路を評価する。
    if not paths:
        return {}, []

    threshold = _read_json(threshold_path) if threshold_path is not None else {}
    stable = _read_json(stable_path) if stable_path is not None else {}
    timeband = _read_json(timeband_path) if timeband_path is not None else {}
    chain = _read_json(chain_path) if chain_path is not None else {}
    subset_refit = _read_json(subset_refit_path) if subset_refit_path is not None else {}

    rec = threshold.get("recommendation") if isinstance(threshold.get("recommendation"), dict) else {}
    threshold_status = _normalize_status(rec.get("recommended_status"))
    threshold_ns = rec.get("recommended_threshold_ns")
    threshold_chi2_dof = rec.get("recommended_chi2_dof")
    min_sessions_operational = rec.get("min_sessions_operational")

    sweep_rows = threshold.get("sweep_rows") if isinstance(threshold.get("sweep_rows"), list) else []
    beta_mean = None
    beta_sigma = None
    # 条件分岐: `sweep_rows` を満たす経路を評価する。
    if sweep_rows:
        sel = None
        for r in sweep_rows:
            # 条件分岐: `not isinstance(r, dict)` を満たす経路を評価する。
            if not isinstance(r, dict):
                continue

            # 条件分岐: `threshold_ns is not None and float(r.get("threshold_ns", float("nan"))) == fl...` を満たす経路を評価する。

            if threshold_ns is not None and float(r.get("threshold_ns", float("nan"))) == float(threshold_ns):
                sel = r
                break

        # 条件分岐: `sel is None` を満たす経路を評価する。

        if sel is None:
            sel = sweep_rows[0] if isinstance(sweep_rows[0], dict) else None

        # 条件分岐: `isinstance(sel, dict)` を満たす経路を評価する。

        if isinstance(sel, dict):
            beta_mean = sel.get("beta_weighted_mean")
            beta_sigma = sel.get("beta_weighted_sigma")

    subset_baseline = subset_refit.get("baseline_scenario") if isinstance(subset_refit.get("baseline_scenario"), dict) else {}
    subset_best_any = subset_refit.get("best_any_scenario") if isinstance(subset_refit.get("best_any_scenario"), dict) else {}
    subset_best_operational = (
        subset_refit.get("best_operational_scenario")
        if isinstance(subset_refit.get("best_operational_scenario"), dict)
        else {}
    )
    subset_eligibility = (
        subset_refit.get("vlbi_beta_comparator_eligibility")
        if isinstance(subset_refit.get("vlbi_beta_comparator_eligibility"), dict)
        else {}
    )
    subset_refit_status = _normalize_status(subset_eligibility.get("status"))
    subset_refit_reason = str(subset_eligibility.get("reason") or "")
    subset_refit_eligible = bool(subset_eligibility.get("eligible"))
    subset_beta_mean = _to_float(subset_baseline.get("beta_weighted_mean"))
    subset_beta_sigma = _to_float(subset_baseline.get("beta_weighted_sigma"))
    # 条件分岐: `subset_beta_mean is not None and subset_beta_sigma is not None and subset_bet...` を満たす経路を評価する。
    if subset_beta_mean is not None and subset_beta_sigma is not None and subset_beta_sigma > 0.0:
        beta_mean = float(subset_beta_mean)
        beta_sigma = float(subset_beta_sigma)

    stable_cons = stable.get("consistency") if isinstance(stable.get("consistency"), dict) else {}
    stable_status = _normalize_status(stable_cons.get("status"))
    stable_chi2_dof = stable_cons.get("chi2_dof")

    session_cons = (
        timeband.get("session_consistency_stable") if isinstance(timeband.get("session_consistency_stable"), dict) else {}
    )
    timeband_status = _normalize_status(session_cons.get("status"))
    timeband_chi2_dof = session_cons.get("chi2_dof")
    sep_diag = timeband.get("separation_diagnostics") if isinstance(timeband.get("separation_diagnostics"), dict) else {}
    separation_hint = sep_diag.get("separation_hint")

    bias_components = {
        "threshold_gate": threshold_status,
        "stable_gate": stable_status,
        "timeband_gate": timeband_status,
        "subset_refit_gate": subset_refit_status,
    }
    vlbi_bias_status = _combine_status(list(bias_components.values()))
    vlbi_status = _combine_status([threshold_status, stable_status, timeband_status, subset_refit_status])
    return (
        {
            "status": vlbi_status,
            "threshold_status": threshold_status,
            "threshold_ns": threshold_ns,
            "threshold_chi2_dof": threshold_chi2_dof,
            "min_sessions_operational": min_sessions_operational,
            "stable_status": stable_status,
            "stable_chi2_dof": stable_chi2_dof,
            "timeband_status": timeband_status,
            "timeband_chi2_dof": timeband_chi2_dof,
            "separation_hint": separation_hint,
            "selected_scenario": chain.get("selected_scenario"),
            "selected_policy": chain.get("selected_policy"),
            "beta_weighted_mean": beta_mean,
            "beta_weighted_sigma": beta_sigma,
            "subset_refit_status": subset_refit_status,
            "subset_refit_reason": subset_refit_reason,
            "subset_refit_eligible": subset_refit_eligible,
            "subset_refit_baseline_abs_z_vs_llr": _to_float(subset_baseline.get("abs_z_vs_llr")),
            "subset_refit_best_any_abs_z_vs_llr": _to_float(subset_best_any.get("abs_z_vs_llr")),
            "subset_refit_best_operational_abs_z_vs_llr": _to_float(subset_best_operational.get("abs_z_vs_llr")),
            "bias_audit_status": vlbi_bias_status,
            "bias_audit_components": bias_components,
        },
        [_safe_rel(p, root) for p in paths],
    )


# 関数: `_extract_llr_channel` の入出力契約と処理意図を定義する。

def _extract_llr_channel(root: Path) -> Tuple[Dict[str, Any], List[str]]:
    llr_path = _first_existing(
        [
            root / "output" / "public" / "llr" / "llr_kappa_llr_metrics.json",
            root / "output" / "private" / "llr" / "llr_kappa_llr_metrics.json",
            root / "output" / "llr" / "llr_kappa_llr_metrics.json",
        ]
    )
    # 条件分岐: `llr_path is None` を満たす経路を評価する。
    if llr_path is None:
        return {}, []

    promotion_path = _first_existing(
        [
            root / "output" / "public" / "llr" / "llr_kappa_llr_beta_promotion_gate_metrics.json",
            root / "output" / "private" / "llr" / "llr_kappa_llr_beta_promotion_gate_metrics.json",
            root / "output" / "llr" / "llr_kappa_llr_beta_promotion_gate_metrics.json",
        ]
    )
    j = _read_json(llr_path)
    promotion = _read_json(promotion_path) if promotion_path is not None else {}
    overall = _normalize_status(j.get("overall_status"))
    gates = j.get("gate") if isinstance(j.get("gate"), dict) else {}
    gate_status_map: Dict[str, str] = {}
    gate_statuses: List[str] = []
    for key, g in gates.items():
        # 条件分岐: `not isinstance(g, dict)` を満たす経路を評価する。
        if not isinstance(g, dict):
            continue

        status = _normalize_status(g.get("status"))
        gate_status_map[str(key)] = status
        gate_statuses.append(status)

    llr_status = _combine_status([overall] + gate_statuses)
    fit = j.get("fit") if isinstance(j.get("fit"), dict) else {}
    beta_mapping = fit.get("beta_mapping") if isinstance(fit.get("beta_mapping"), dict) else {}
    kappa_est = beta_mapping.get("beta_est", fit.get("selected_kappa_est"))
    kappa_sigma = beta_mapping.get("beta_sigma", fit.get("selected_kappa_sigma"))
    abs_z = beta_mapping.get("abs_z_beta_minus_1", fit.get("selected_abs_z"))
    selected_mode = fit.get("selected_mode")
    # 条件分岐: `isinstance(beta_mapping, dict) and beta_mapping.get("source")` を満たす経路を評価する。
    if isinstance(beta_mapping, dict) and beta_mapping.get("source"):
        selected_mode = f"{selected_mode}|{beta_mapping.get('source')}"

    bias_audit = j.get("bias_audit") if isinstance(j.get("bias_audit"), dict) else {}
    bias_components_raw = bias_audit.get("components") if isinstance(bias_audit.get("components"), dict) else {}
    bias_components: Dict[str, str] = {}
    # 条件分岐: `bias_components_raw` を満たす経路を評価する。
    if bias_components_raw:
        for key, value in bias_components_raw.items():
            bias_components[str(key)] = _normalize_status(value)
    else:
        for key in (
            "imbalance_policy_gate",
            "station_stratified_gate",
            "target_stratified_gate",
            "template_null_gate",
            "template_decontamination_gate",
        ):
            # 条件分岐: `key in gate_status_map` を満たす経路を評価する。
            if key in gate_status_map:
                bias_components[key] = gate_status_map[key]

    llr_bias_status = _combine_status(list(bias_components.values())) if bias_components else _normalize_status(llr_status)
    promotion_decision = (
        promotion.get("promotion_decision")
        if isinstance(promotion.get("promotion_decision"), dict)
        else {}
    )
    promotion_gate = promotion.get("gate_status") if isinstance(promotion.get("gate_status"), dict) else {}
    promotion_status = _normalize_status(
        promotion_gate.get("overall_status", promotion_decision.get("status"))
    )
    promoted = bool(promotion_decision.get("promoted")) if promotion_decision else False
    # 条件分岐: `promotion_path is not None` を満たす経路を評価する。
    if promotion_path is not None:
        llr_status = promotion_status
        llr_bias_status = promotion_status

    source_paths = [_safe_rel(llr_path, root)]
    # 条件分岐: `promotion_path is not None` を満たす経路を評価する。
    if promotion_path is not None:
        source_paths.append(_safe_rel(promotion_path, root))

    return (
        {
            "status": llr_status,
            "overall_status": overall,
            "selected_mode": selected_mode,
            "kappa_est": kappa_est,
            "kappa_sigma": kappa_sigma,
            "abs_z": abs_z,
            "gate_statuses": gate_statuses,
            "gate_status_map": gate_status_map,
            "bias_audit_status": llr_bias_status,
            "bias_audit_components": bias_components,
            "promotion_status": promotion_status,
            "promotion_promoted": promoted,
            "legacy_status": _combine_status([overall] + gate_statuses),
        },
        source_paths,
    )


# 関数: `_extract_messenger_channel` の入出力契約と処理意図を定義する。

def _extract_messenger_channel(root: Path) -> Tuple[Dict[str, Any], List[str]]:
    stage_j_path = _first_existing(
        [
            root / "output" / "public" / "mercury" / "messenger_beta_stage_j_final_gate_metrics.json",
            root / "output" / "private" / "mercury" / "messenger_beta_stage_j_final_gate_metrics.json",
        ]
    )
    # 条件分岐: `stage_j_path is None` を満たす経路を評価する。
    if stage_j_path is None:
        return {}, []

    stage_i_path = _first_existing(
        [
            root / "output" / "public" / "mercury" / "messenger_beta_stage_i_nuisance_sensitivity_metrics.json",
            root / "output" / "private" / "mercury" / "messenger_beta_stage_i_nuisance_sensitivity_metrics.json",
        ]
    )
    stage_h_path = _first_existing(
        [
            root / "output" / "public" / "mercury" / "messenger_beta_stage_h_segmentation_metrics.json",
            root / "output" / "private" / "mercury" / "messenger_beta_stage_h_segmentation_metrics.json",
        ]
    )
    stage_e_path = _first_existing(
        [
            root / "output" / "public" / "mercury" / "messenger_beta_stage_e_tnf_replay_metrics.json",
            root / "output" / "private" / "mercury" / "messenger_beta_stage_e_tnf_replay_metrics.json",
        ]
    )
    stage_e_sweep_path = _first_existing(
        [
            root / "output" / "public" / "mercury" / "messenger_beta_stage_e_replay_sweep_metrics.json",
            root / "output" / "private" / "mercury" / "messenger_beta_stage_e_replay_sweep_metrics.json",
        ]
    )
    stage_d_path = _first_existing(
        [
            root / "output" / "public" / "mercury" / "messenger_beta_stage_d_joint_metrics.json",
            root / "output" / "private" / "mercury" / "messenger_beta_stage_d_joint_metrics.json",
        ]
    )
    paths = [p for p in (stage_j_path, stage_i_path, stage_h_path, stage_e_path, stage_e_sweep_path, stage_d_path) if p is not None]

    stage_j = _read_json(stage_j_path)
    stage_i = _read_json(stage_i_path) if stage_i_path is not None else {}
    stage_h = _read_json(stage_h_path) if stage_h_path is not None else {}
    stage_e = _read_json(stage_e_path) if stage_e_path is not None else {}
    stage_e_sweep = _read_json(stage_e_sweep_path) if stage_e_sweep_path is not None else {}
    stage_d = _read_json(stage_d_path) if stage_d_path is not None else {}

    stage_j_status = _normalize_status(stage_j.get("overall_status"))
    stage_i_status = _normalize_status(stage_i.get("overall_status")) if stage_i_path is not None else "watch"
    stage_h_status = _normalize_status(stage_h.get("overall_status")) if stage_h_path is not None else "watch"
    stage_e_replay = stage_e.get("replay_vs_odf") if isinstance(stage_e.get("replay_vs_odf"), dict) else {}
    stage_e_replay_status = _normalize_status(stage_e_replay.get("status")) if stage_e_path is not None else "watch"
    stage_d_components = stage_d.get("status_components") if isinstance(stage_d.get("status_components"), dict) else {}
    stage_d_data_status = _normalize_status(stage_d_components.get("data")) if stage_d_path is not None else "watch"
    stage_d_sigma_status = _normalize_status(stage_d_components.get("sigma")) if stage_d_path is not None else "watch"
    stage_j_counts = stage_j.get("status_counts") if isinstance(stage_j.get("status_counts"), dict) else {}

    stage_i_policy = stage_i.get("scenario_policy") if isinstance(stage_i.get("scenario_policy"), dict) else {}
    stage_h_policy = stage_h.get("segment_policy") if isinstance(stage_h.get("segment_policy"), dict) else {}
    stage_i_decomp = _extract_stage_i_beta_decomposition(stage_i)
    stage_i_branch_summary = (
        stage_i_decomp.get("branch_summary_by_branch")
        if isinstance(stage_i_decomp.get("branch_summary_by_branch"), dict)
        else {}
    )
    stage_i_decomposition_by_branch = (
        stage_i_decomp.get("decomposition_by_branch")
        if isinstance(stage_i_decomp.get("decomposition_by_branch"), dict)
        else {}
    )
    stage_i_nuisance_priority = _build_stage_i_nuisance_priority(
        stage_i_policy=stage_i_policy,
        stage_i_decomposition_by_branch=stage_i_decomposition_by_branch,
    )
    odf_decomp = (
        stage_i_decomposition_by_branch.get("odf")
        if isinstance(stage_i_decomposition_by_branch.get("odf"), dict)
        else {}
    )
    tnf_decomp = (
        stage_i_decomposition_by_branch.get("tnf")
        if isinstance(stage_i_decomposition_by_branch.get("tnf"), dict)
        else {}
    )

    core_ids = stage_i_policy.get("core_scenario_ids") if isinstance(stage_i_policy.get("core_scenario_ids"), list) else []
    diag_ids = stage_i_policy.get("diagnostic_scenario_ids") if isinstance(stage_i_policy.get("diagnostic_scenario_ids"), list) else []
    required_by_branch = (
        stage_h_policy.get("required_segmentation_types_by_branch")
        if isinstance(stage_h_policy.get("required_segmentation_types_by_branch"), dict)
        else {}
    )
    diag_by_branch = (
        stage_h_policy.get("diagnostic_segmentation_types_by_branch")
        if isinstance(stage_h_policy.get("diagnostic_segmentation_types_by_branch"), dict)
        else {}
    )

    policy_note_parts: List[str] = []
    # 条件分岐: `core_ids` を満たす経路を評価する。
    if core_ids:
        policy_note_parts.append(f"stage_i_core={','.join(str(x) for x in core_ids)}")

    # 条件分岐: `diag_ids` を満たす経路を評価する。

    if diag_ids:
        policy_note_parts.append(f"stage_i_diag={','.join(str(x) for x in diag_ids)}")

    top_nuisance = str(stage_i_nuisance_priority.get("top_mitigation_scenario_id") or "")
    # 条件分岐: `top_nuisance` を満たす経路を評価する。
    if top_nuisance:
        policy_note_parts.append(f"stage_i_top_nuisance={top_nuisance}")

    # 条件分岐: `required_by_branch` を満たす経路を評価する。

    if required_by_branch:
        policy_note_parts.append(f"stage_h_required={required_by_branch}")

    # 条件分岐: `diag_by_branch` を満たす経路を評価する。

    if diag_by_branch:
        policy_note_parts.append(f"stage_h_diag={diag_by_branch}")

    policy_note = " | ".join(policy_note_parts)

    messenger_status = _combine_status(
        [
            stage_j_status,
            stage_i_status,
            stage_h_status,
            stage_e_replay_status,
            stage_d_data_status,
        ]
    )
    comparator_eligible = bool(
        messenger_status == "pass"
        and stage_j_status == "pass"
        and stage_e_replay_status == "pass"
        and stage_d_data_status == "pass"
    )

    return (
        {
            "status": messenger_status,
            "stage_j_status": stage_j_status,
            "stage_j_hard_fail": bool(stage_j.get("hard_fail", False)),
            "stage_j_status_counts": stage_j_counts,
            "stage_i_status": stage_i_status,
            "stage_h_status": stage_h_status,
            "stage_e_replay_status": stage_e_replay_status,
            "stage_e_replay_z_delta_beta": _to_float(stage_e_replay.get("z_delta_beta")),
            "stage_d_data_status": stage_d_data_status,
            "stage_d_sigma_status": stage_d_sigma_status,
            "beta_primary_definition": "beta_lt",
            "beta_primary_source": "stage_d.beta_lt",
            "beta_primary_est": _to_float(stage_d.get("beta_lt_estimate")),
            "beta_primary_sigma": _to_float(stage_d.get("beta_lt_sigma")),
            "beta_replay_est": _to_float(stage_e.get("beta_lt_estimate")),
            "beta_replay_sigma": _to_float(stage_e.get("beta_lt_sigma")),
            "beta_dyn_diagnostic_est": _to_float(stage_d.get("beta_dyn_estimate")),
            "beta_dyn_diagnostic_sigma": _to_float(stage_d.get("beta_sigma")),
            "beta_dyn_replay_diagnostic_est": _to_float(stage_e.get("beta_dyn_estimate")),
            "beta_dyn_replay_diagnostic_sigma": _to_float(stage_e.get("beta_sigma")),
            "bias_audit_status": _combine_status([stage_i_status, stage_h_status]),
            "comparator_eligible": comparator_eligible,
            "eligibility_reason": (
                "stage_j_pass+stage_e_replay_pass+stage_d_data_pass"
                if comparator_eligible
                else "requires_stage_j/stage_e_replay/stage_d_data_pass"
            ),
            "policy_note": policy_note,
            "stage_i_policy": stage_i_policy,
            "stage_h_policy": stage_h_policy,
            "stage_i_branch_summary": stage_i_branch_summary,
            "stage_i_beta_dyn_decomposition": stage_i_decomposition_by_branch,
            "stage_i_nuisance_priority": stage_i_nuisance_priority,
            "stage_i_odf_baseline_abs_z_beta_minus_1": _to_float(odf_decomp.get("baseline_abs_z_beta_minus_1")),
            "stage_i_odf_best_abs_z_beta_minus_1": _to_float(odf_decomp.get("best_abs_z_beta_minus_1")),
            "stage_i_odf_max_shift_delta_z": _to_float(odf_decomp.get("max_shift_delta_vs_baseline_z")),
            "stage_i_tnf_baseline_abs_z_beta_minus_1": _to_float(tnf_decomp.get("baseline_abs_z_beta_minus_1")),
            "stage_i_tnf_best_abs_z_beta_minus_1": _to_float(tnf_decomp.get("best_abs_z_beta_minus_1")),
            "stage_i_tnf_max_shift_delta_z": _to_float(tnf_decomp.get("max_shift_delta_vs_baseline_z")),
            "stage_e_replay_sweep_best": (
                stage_e_sweep.get("best_trial")
                if isinstance(stage_e_sweep.get("best_trial"), dict)
                else {}
            ),
        },
        [_safe_rel(p, root) for p in paths],
    )


# 関数: `_compute_beta_terminal` の入出力契約と処理意図を定義する。

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


# 関数: `_compute_beta_terminal` の入出力契約と処理意図を定義する。

def _compute_beta_terminal(vlbi: Dict[str, Any], llr: Dict[str, Any], messenger: Dict[str, Any], cross: Dict[str, Any]) -> Dict[str, Any]:
    beta_v = _to_float(vlbi.get("beta_weighted_mean"))
    sigma_v = _to_float(vlbi.get("beta_weighted_sigma"))
    beta_l = _to_float(llr.get("kappa_est"))
    sigma_l = _to_float(llr.get("kappa_sigma"))
    beta_m = _to_float(messenger.get("beta_primary_est"))
    sigma_m = _to_float(messenger.get("beta_primary_sigma"))

    cross_consistency_raw = _normalize_status(cross.get("beta_consistency_status"))
    subset_refit_eligible = bool(vlbi.get("subset_refit_eligible", True))
    messenger_eligible = bool(messenger.get("comparator_eligible", False))
    vlbi_bias_status = _normalize_status(vlbi.get("bias_audit_status", vlbi.get("status")))
    llr_bias_status = _normalize_status(llr.get("bias_audit_status", llr.get("status")))
    messenger_bias_status = _normalize_status(messenger.get("bias_audit_status", messenger.get("status")))
    messenger_stage_j_status = _normalize_status(messenger.get("stage_j_status"))
    messenger_replay_status = _normalize_status(messenger.get("stage_e_replay_status"))
    llr_promotion_status = _normalize_status(llr.get("promotion_status"))

    # 関数: `_build_policy` の入出力契約と処理意図を定義する。
    def _build_policy(
        policy_id: str,
        policy_label: str,
        channels: Sequence[tuple[str, Optional[float], Optional[float]]],
        bias_inputs: Sequence[str],
        cross_status: str,
        promotion_status: str,
        promotion_required: bool,
    ) -> Dict[str, Any]:
        beta_est, beta_sigma, used_channels = _fuse_beta(channels)
        availability_status = "pass" if (beta_est is not None and beta_sigma is not None) else "reject"
        beta_abs_z_minus_1 = None
        beta_minus_1_status = "reject"
        # 条件分岐: `beta_est is not None and beta_sigma is not None and beta_sigma > 0.0` を満たす経路を評価する。
        if beta_est is not None and beta_sigma is not None and beta_sigma > 0.0:
            beta_abs_z_minus_1 = abs((beta_est - 1.0) / beta_sigma)
            beta_minus_1_status = _status_from_abs_z(beta_abs_z_minus_1)

        bias_audit_status = _combine_status(list(bias_inputs))
        promotion_gate_status = _normalize_status(promotion_status) if promotion_required else "pass"
        terminal_status = _combine_status(
            [
                availability_status,
                _normalize_status(cross_status),
                beta_minus_1_status,
                bias_audit_status,
                promotion_gate_status,
            ]
        )
        return {
            "policy_id": policy_id,
            "policy_label": policy_label,
            "status": terminal_status,
            "availability_status": availability_status,
            "cross_consistency_status": _normalize_status(cross_status),
            "beta_minus_1_status": beta_minus_1_status,
            "bias_audit_status": bias_audit_status,
            "promotion_status": (_normalize_status(promotion_status) if promotion_required else "not_applicable"),
            "promotion_required": bool(promotion_required),
            "beta_est": beta_est,
            "beta_sigma": beta_sigma,
            "beta_abs_z_minus_1": beta_abs_z_minus_1,
            "channels_used": used_channels,
        }

    base_promotion = "not_applicable"
    policy_matrix: Dict[str, Dict[str, Any]] = {}

    policy_matrix["policy_A_hard_reject_keep"] = _build_policy(
        policy_id="policy_A_hard_reject_keep",
        policy_label="A: hard reject keep",
        channels=[("vlbi", beta_v, sigma_v), ("llr", beta_l, sigma_l)],
        bias_inputs=[vlbi_bias_status, llr_bias_status],
        cross_status=("reject" if subset_refit_eligible is False else cross_consistency_raw),
        promotion_status=base_promotion,
        promotion_required=False,
    )
    policy_matrix["policy_B_exclude_comparator_when_ineligible"] = _build_policy(
        policy_id="policy_B_exclude_comparator_when_ineligible",
        policy_label="B: exclude comparator",
        channels=[("llr", beta_l, sigma_l)],
        bias_inputs=[llr_bias_status],
        cross_status=("pass" if subset_refit_eligible is False else cross_consistency_raw),
        promotion_status=base_promotion,
        promotion_required=False,
    )
    policy_matrix["policy_C_exclude_plus_promotion_pass"] = _build_policy(
        policy_id="policy_C_exclude_plus_promotion_pass",
        policy_label="C: exclude + promotion-pass",
        channels=[("llr", beta_l, sigma_l)],
        bias_inputs=[llr_bias_status],
        cross_status=("pass" if subset_refit_eligible is False else cross_consistency_raw),
        promotion_status=llr_promotion_status,
        promotion_required=True,
    )

    messenger_pair_abs_z, messenger_pair_status = _pair_abs_z_status(
        beta_l,
        sigma_l,
        (beta_m if messenger_eligible else None),
        (sigma_m if messenger_eligible else None),
    )
    messenger_pair_decomposition = _build_pair_decomposition(
        beta_anchor=beta_l,
        sigma_anchor=sigma_l,
        beta_candidate=(beta_m if messenger_eligible else None),
        sigma_candidate=(sigma_m if messenger_eligible else None),
    )
    policy_d_stage_i_mapping = _build_policy_d_stage_i_mapping(
        messenger=messenger,
        pair_decomposition=messenger_pair_decomposition,
    )
    policy_d_cross = _combine_status(
        [
            ("pass" if subset_refit_eligible is False else cross_consistency_raw),
            messenger_pair_status,
        ]
    )
    policy_matrix["policy_D_exclude_plus_messenger_fusion"] = _build_policy(
        policy_id="policy_D_exclude_plus_messenger_fusion",
        policy_label="D: exclude + LLR+MESSENGER",
        channels=[("llr", beta_l, sigma_l), ("messenger", (beta_m if messenger_eligible else None), (sigma_m if messenger_eligible else None))],
        bias_inputs=[
            llr_bias_status,
            (messenger_bias_status if messenger_eligible else "reject"),
            (messenger_stage_j_status if messenger_eligible else "reject"),
            (messenger_replay_status if messenger_eligible else "reject"),
        ],
        cross_status=policy_d_cross,
        promotion_status=base_promotion,
        promotion_required=False,
    )
    policy_matrix["policy_D_exclude_plus_messenger_fusion"]["pair_abs_z_llr_messenger"] = messenger_pair_abs_z
    policy_matrix["policy_D_exclude_plus_messenger_fusion"]["pair_status_llr_messenger"] = messenger_pair_status
    policy_matrix["policy_D_exclude_plus_messenger_fusion"]["pair_decomposition"] = messenger_pair_decomposition
    policy_matrix["policy_D_exclude_plus_messenger_fusion"]["stage_i_mapping"] = policy_d_stage_i_mapping

    policy_matrix["policy_E_exclude_messenger_only"] = _build_policy(
        policy_id="policy_E_exclude_messenger_only",
        policy_label="E: exclude + MESSENGER-only",
        channels=[("messenger", (beta_m if messenger_eligible else None), (sigma_m if messenger_eligible else None))],
        bias_inputs=[
            (messenger_bias_status if messenger_eligible else "reject"),
            (messenger_stage_j_status if messenger_eligible else "reject"),
            (messenger_replay_status if messenger_eligible else "reject"),
        ],
        cross_status=("pass" if subset_refit_eligible is False else cross_consistency_raw),
        promotion_status=base_promotion,
        promotion_required=False,
    )

    active_policy_id = "policy_A_hard_reject_keep" if subset_refit_eligible else "policy_B_exclude_comparator_when_ineligible"
    active_policy = policy_matrix.get(active_policy_id, policy_matrix["policy_A_hard_reject_keep"])
    selection_reason = (
        "vlbi_subset_refit_eligible=true -> keep hard comparator gate"
        if subset_refit_eligible
        else "vlbi_subset_refit_eligible=false -> exclude comparator fallback"
    )
    policy_b = (
        policy_matrix.get("policy_B_exclude_comparator_when_ineligible")
        if isinstance(policy_matrix.get("policy_B_exclude_comparator_when_ineligible"), dict)
        else {}
    )
    policy_d = (
        policy_matrix.get("policy_D_exclude_plus_messenger_fusion")
        if isinstance(policy_matrix.get("policy_D_exclude_plus_messenger_fusion"), dict)
        else {}
    )
    policy_d_pair = (
        policy_d.get("pair_decomposition")
        if isinstance(policy_d, dict) and isinstance(policy_d.get("pair_decomposition"), dict)
        else {}
    )
    policy_d_promotion_conditions = {
        "messenger_comparator_eligible": ("pass" if messenger_eligible else "reject"),
        "policy_d_pair_gate_pass": _normalize_status(policy_d_pair.get("pass_gate_status")),
        "llr_bias_gate": llr_bias_status,
        "llr_promotion_gate": llr_promotion_status,
        "policy_d_bias_gate": _normalize_status(policy_d.get("bias_audit_status")),
    }
    policy_d_promotion_status = _combine_status(list(policy_d_promotion_conditions.values()))
    policy_d_promotion_ready = bool(policy_d_promotion_status == "pass")
    policy_d_promotion_blockers = [
        k for k, v in policy_d_promotion_conditions.items() if _normalize_status(v) != "pass"
    ]
    blocker_priority_seed: Dict[str, Dict[str, Any]] = {
        "llr_bias_gate": {
            "priority_rank": 1,
            "depends_on": [],
            "recommended_action": "close_llr_bias_audit_first",
            "resolution_condition": "llr.bias_audit_status=pass",
            "rationale": "upstream_root_gate",
        },
        "llr_promotion_gate": {
            "priority_rank": 2,
            "depends_on": ["llr_bias_gate"],
            "recommended_action": "recompute_llr_promotion_after_bias_pass",
            "resolution_condition": "llr.promotion_status=pass",
            "rationale": "promotion_gate_after_bias",
        },
        "policy_d_bias_gate": {
            "priority_rank": 3,
            "depends_on": ["llr_bias_gate"],
            "recommended_action": "recompute_policy_d_bias_after_llr_bias_pass",
            "resolution_condition": "policy_D.bias_audit_status=pass",
            "rationale": "downstream_fusion_bias_gate",
        },
    }
    policy_d_promotion_blocker_priority: List[Dict[str, Any]] = []
    for blocker_id in policy_d_promotion_blockers:
        meta = blocker_priority_seed.get(str(blocker_id), {})
        priority_rank = int(meta.get("priority_rank")) if str(meta.get("priority_rank", "")).strip() else 99
        blocker_status = _normalize_status(policy_d_promotion_conditions.get(str(blocker_id)))
        policy_d_promotion_blocker_priority.append(
            {
                "blocker_id": str(blocker_id),
                "status": blocker_status,
                "priority_rank": priority_rank,
                "depends_on": list(meta.get("depends_on") or []),
                "recommended_action": str(meta.get("recommended_action") or "review_blocker_resolution_path"),
                "resolution_condition": str(meta.get("resolution_condition") or ""),
                "rationale": str(meta.get("rationale") or "unspecified"),
            }
        )

    policy_d_promotion_blocker_priority = sorted(
        policy_d_promotion_blocker_priority,
        key=lambda x: (
            int(x.get("priority_rank") or 99),
            str(x.get("blocker_id") or ""),
        ),
    )
    policy_d_promotion_blocker_resolution_order = [
        str(x.get("blocker_id") or "") for x in policy_d_promotion_blocker_priority if str(x.get("blocker_id") or "").strip()
    ]
    policy_d_blocker_order_status = "pass" if len(policy_d_promotion_blocker_resolution_order) <= 0 else "watch"
    llr_gate_expected_order = ["llr_bias_gate", "llr_promotion_gate"]
    llr_gate_active_order = [str(x) for x in llr_gate_expected_order if str(x) in set(str(b) for b in policy_d_promotion_blockers)]
    llr_gate_observed_order = [str(x) for x in policy_d_promotion_blocker_resolution_order if str(x) in set(llr_gate_expected_order)]
    llr_gate_actions_ordered = [
        str((blocker_priority_seed.get(str(x)) or {}).get("recommended_action") or "")
        for x in llr_gate_active_order
        if str((blocker_priority_seed.get(str(x)) or {}).get("recommended_action") or "").strip()
    ]
    llr_gate_order_alignment = bool(
        (len(llr_gate_active_order) <= 1) or (llr_gate_observed_order == llr_gate_active_order)
    )
    llr_gate_order_status = "pass" if llr_gate_order_alignment else "watch"
    llr_gate_repro_commands_min = [
        "python -B scripts/llr/llr_kappa_llr_direct_fit.py",
        "python -B scripts/llr/llr_kappa_llr_beta_promotion_gate.py",
        "python -B scripts/summary/beta_cross_channel_registry.py",
        "python -B scripts/summary/beta_terminal_reject_checklist.py",
    ]
    expected_blocker_order = [
        str(blocker_id)
        for blocker_id, _rank in sorted(
            (
                (str(k), int(v.get("priority_rank") or 99))
                for k, v in blocker_priority_seed.items()
                if str(k) in set(str(x) for x in policy_d_promotion_blockers)
            ),
            key=lambda x: (int(x[1]), str(x[0])),
        )
    ]
    observed_blocker_order = [str(x) for x in policy_d_promotion_blocker_resolution_order]
    blocker_set_alignment = bool(set(str(x) for x in policy_d_promotion_blockers) == set(observed_blocker_order))
    blocker_order_alignment = bool(observed_blocker_order == expected_blocker_order)
    policy_d_matrix_status = _normalize_status(policy_d.get("status"))
    policy_d_status_alignment = bool(policy_d_matrix_status == policy_d_promotion_status)
    policy_d_promotion_reassessment = {
        "status": _combine_status(
            [
                ("pass" if blocker_set_alignment else "watch"),
                ("pass" if blocker_order_alignment else "watch"),
                ("pass" if policy_d_status_alignment else "watch"),
            ]
        ),
        "promotion_status_reassessed": policy_d_promotion_status,
        "promotion_ready_reassessed": policy_d_promotion_ready,
        "policy_d_matrix_status": policy_d_matrix_status,
        "policy_d_status_alignment": policy_d_status_alignment,
        "blocker_set_alignment": blocker_set_alignment,
        "blocker_order_alignment": blocker_order_alignment,
        "expected_blocker_order": expected_blocker_order,
        "observed_blocker_order": observed_blocker_order,
        "blocker_count_delta": int(len(observed_blocker_order) - len(policy_d_promotion_blockers)),
        "active_policy_update_required": False,
        "llr_gate_order_status": llr_gate_order_status,
        "llr_gate_execution_order_expected": llr_gate_active_order,
        "llr_gate_execution_order_observed": llr_gate_observed_order,
        "llr_gate_actions_ordered": llr_gate_actions_ordered,
    }
    policy_b_hold_conditions = {
        "vlbi_subset_refit_ineligible": ("pass" if subset_refit_eligible is False else "reject"),
        "policy_b_available": _normalize_status(policy_b.get("availability_status")),
        "active_policy_is_b": ("pass" if active_policy_id == "policy_B_exclude_comparator_when_ineligible" else "reject"),
        "policy_d_not_ready": ("pass" if not policy_d_promotion_ready else "watch"),
    }
    policy_b_hold_status = _combine_status(list(policy_b_hold_conditions.values()))
    recommended_active_policy_id = (
        "policy_A_hard_reject_keep"
        if subset_refit_eligible
        else ("policy_D_exclude_plus_messenger_fusion" if policy_d_promotion_ready else "policy_B_exclude_comparator_when_ineligible")
    )
    recommended_active_policy_reason = (
        "vlbi_subset_refit_eligible=true -> keep comparator hard gate"
        if subset_refit_eligible
        else (
            "policy_D_promotion_ready=true -> allow llr+messenger fusion promotion"
            if policy_d_promotion_ready
            else "policy_D_promotion_ready=false -> keep policy_B hold"
        )
    )
    active_policy_update_required = bool(
        str(recommended_active_policy_id).strip()
        and str(active_policy_id).strip()
        and (str(recommended_active_policy_id).strip() != str(active_policy_id).strip())
    )
    policy_switch_target_id = "policy_D_exclude_plus_messenger_fusion"
    active_policy_id_before_switch = str(active_policy_id)
    switch_required_now = bool(active_policy_update_required)
    switch_allowed_now = bool(
        str(recommended_active_policy_id).strip() == str(policy_switch_target_id)
        and bool(policy_d_promotion_ready)
        and len(policy_d_promotion_blockers) <= 0
    )
    switch_decision_id = "switch_to_policy_D_now" if switch_required_now and switch_allowed_now else "hold_policy_B_until_policy_D_ready"
    switch_applied_now = bool(switch_required_now and switch_allowed_now)
    if switch_applied_now:
        active_policy_id = str(recommended_active_policy_id)
        active_policy = policy_matrix.get(active_policy_id, active_policy)
        selection_reason = "policy_switch_decision=switch_to_policy_D_now -> auto-promoted to policy_D"
        active_policy_update_required = False

    switch_hold_reason = (
        ""
        if switch_decision_id == "switch_to_policy_D_now"
        else (
            "policy_D_promotion_ready=false"
            if not bool(policy_d_promotion_ready)
            else (
                f"policy_D_blockers={','.join(str(x) for x in policy_d_promotion_blockers)}"
                if len(policy_d_promotion_blockers) > 0
                else "active_policy_already_aligned"
            )
        )
    )
    policy_switch_decision = {
        "status": ("pass" if not switch_required_now or switch_allowed_now else "watch"),
        "redecision_step": "8.7.48.29",
        "from_policy_id": active_policy_id_before_switch,
        "to_policy_id": str(policy_switch_target_id),
        "recommended_policy_id": str(recommended_active_policy_id),
        "switch_required_now": switch_required_now,
        "switch_allowed_now": switch_allowed_now,
        "switch_applied_now": switch_applied_now,
        "decision_id": switch_decision_id,
        "hold_reason": switch_hold_reason,
        "unresolved_blockers": [str(x) for x in policy_d_promotion_blockers],
    }
    policy_aligned_now = bool(
        str(active_policy_id).strip() == str(recommended_active_policy_id).strip()
    )
    policy_terminal_watch_statement_id = (
        "watch_hold_policy_B_until_policy_D_ready"
        if not bool(policy_d_promotion_ready)
        else ("watch_resolved_policy_alignment_completed" if policy_aligned_now else "watch_pending_policy_selection_alignment")
    )
    policy_terminal_watch_statement_text = (
        "beta_terminal=watch; active policy is fixed to policy_B_exclude_comparator_when_ineligible "
        "until policy_D_promotion_ready=true and blockers "
        "(llr_bias_gate,llr_promotion_gate,policy_d_bias_gate) are resolved."
        if not bool(policy_d_promotion_ready)
        else (
            "beta_terminal=watch; policy alignment to governance recommendation is completed."
            if policy_aligned_now
            else "beta_terminal=watch; align active policy with governance recommendation after remaining checks."
        )
    )
    policy_terminal_watch_statement = {
        "status": (
            "pass"
            if (
                (str(active_policy_id) == "policy_B_exclude_comparator_when_ineligible" and not bool(policy_d_promotion_ready))
                or (bool(policy_d_promotion_ready) and policy_aligned_now)
            )
            else "watch"
        ),
        "statement_step": "8.7.48.30",
        "statement_id": policy_terminal_watch_statement_id,
        "statement_text": policy_terminal_watch_statement_text,
        "applies_when": "policy_D_promotion_ready=false",
        "release_condition": "policy_D_promotion_ready=true",
    }
    policy_d_promotion_reassessment["active_policy_update_required"] = active_policy_update_required
    policy_d_promotion_reassessment["status"] = _combine_status(
        [
            ("pass" if blocker_set_alignment else "watch"),
            ("pass" if blocker_order_alignment else "watch"),
            ("pass" if policy_d_status_alignment else "watch"),
            ("pass" if not active_policy_update_required else "watch"),
        ]
    )
    policy_governance = {
        "policy_b_hold_status": policy_b_hold_status,
        "policy_b_hold_conditions": policy_b_hold_conditions,
        "policy_d_promotion_status": policy_d_promotion_status,
        "policy_d_promotion_ready": policy_d_promotion_ready,
        "policy_d_promotion_conditions": policy_d_promotion_conditions,
        "policy_d_promotion_blockers": policy_d_promotion_blockers,
        "policy_d_promotion_blocker_priority": policy_d_promotion_blocker_priority,
        "policy_d_promotion_blocker_resolution_order": policy_d_promotion_blocker_resolution_order,
        "policy_d_promotion_blocker_order_status": policy_d_blocker_order_status,
        "llr_gate_execution_order": llr_gate_active_order,
        "llr_gate_actions_ordered": llr_gate_actions_ordered,
        "llr_gate_order_status": llr_gate_order_status,
        "llr_gate_repro_commands_min": llr_gate_repro_commands_min,
        "policy_switch_decision": policy_switch_decision,
        "policy_terminal_watch_statement": policy_terminal_watch_statement,
        "policy_d_promotion_reassessment": policy_d_promotion_reassessment,
        "recommended_active_policy_id": recommended_active_policy_id,
        "recommended_active_policy_reason": recommended_active_policy_reason,
    }

    bias_audit_components = {
        "vlbi_status": vlbi_bias_status,
        "vlbi_components": vlbi.get("bias_audit_components", {}),
        "llr_status": llr_bias_status,
        "llr_components": llr.get("bias_audit_components", {}),
        "messenger_status": messenger_bias_status,
        "messenger_components": {
            "stage_j_status": messenger_stage_j_status,
            "stage_e_replay_status": messenger_replay_status,
        },
    }
    return {
        "status": active_policy.get("status"),
        "availability_status": active_policy.get("availability_status"),
        "cross_consistency_status": active_policy.get("cross_consistency_status"),
        "beta_minus_1_status": active_policy.get("beta_minus_1_status"),
        "bias_audit_status": active_policy.get("bias_audit_status"),
        "beta_vlbi_est": beta_v,
        "beta_vlbi_sigma": sigma_v,
        "beta_llr_est": beta_l,
        "beta_llr_sigma": sigma_l,
        "beta_messenger_est": beta_m,
        "beta_messenger_sigma": sigma_m,
        "beta_combined_est": active_policy.get("beta_est"),
        "beta_combined_sigma": active_policy.get("beta_sigma"),
        "beta_combined_abs_z_minus_1": active_policy.get("beta_abs_z_minus_1"),
        "beta_consistency_abs_z": _to_float(cross.get("beta_consistency_abs_z")),
        "active_policy_id": active_policy_id,
        "active_policy_label": active_policy.get("policy_label"),
        "active_policy_selection_reason": selection_reason,
        "policy_d_pair_decomposition": (
            policy_matrix.get("policy_D_exclude_plus_messenger_fusion", {}).get("pair_decomposition")
            if isinstance(policy_matrix.get("policy_D_exclude_plus_messenger_fusion"), dict)
            else {}
        ),
        "policy_d_stage_i_mapping": (
            policy_matrix.get("policy_D_exclude_plus_messenger_fusion", {}).get("stage_i_mapping")
            if isinstance(policy_matrix.get("policy_D_exclude_plus_messenger_fusion"), dict)
            else {}
        ),
        "policy_governance": policy_governance,
        "policy_matrix": policy_matrix,
        "bias_audit_components": bias_audit_components,
    }


# 関数: `_compute_cross_channel` の入出力契約と処理意図を定義する。

def _compute_cross_channel(vlbi: Dict[str, Any], llr: Dict[str, Any]) -> Dict[str, Any]:
    vlbi_status = _normalize_status(vlbi.get("status"))
    llr_status = _normalize_status(llr.get("status"))
    beta_v = vlbi.get("beta_weighted_mean")
    sigma_v = vlbi.get("beta_weighted_sigma")
    beta_l = llr.get("kappa_est")
    sigma_l = llr.get("kappa_sigma")

    beta_consistency_abs_z = None
    beta_consistency_status = "reject"
    # 条件分岐: `all(v is not None for v in (beta_v, sigma_v, beta_l, sigma_l))` を満たす経路を評価する。
    if all(v is not None for v in (beta_v, sigma_v, beta_l, sigma_l)):
        try:
            b1 = float(beta_v)
            s1 = float(sigma_v)
            b2 = float(beta_l)
            s2 = float(sigma_l)
            denom = math.sqrt(max(s1 * s1 + s2 * s2, 1e-30))
            beta_consistency_abs_z = abs((b1 - b2) / denom) if denom > 0 else None
            beta_consistency_status = _status_from_abs_z(beta_consistency_abs_z)
        except Exception:
            beta_consistency_abs_z = None
            beta_consistency_status = "reject"

    channel_status = _combine_status([vlbi_status, llr_status])
    cross_status = _combine_status([channel_status, beta_consistency_status])
    return {
        "status": cross_status,
        "channel_status": channel_status,
        "beta_consistency_abs_z": beta_consistency_abs_z,
        "beta_consistency_status": beta_consistency_status,
    }


# 関数: `_write_csv_rows` の入出力契約と処理意図を定義する。

def _write_csv_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "channel",
        "status",
        "metric",
        "value",
        "note",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


# 関数: `_write_plot` の入出力契約と処理意図を定義する。

def _write_plot(payload: Dict[str, Any], out_pdf: Path, out_png: Path) -> None:
    vlbi = payload.get("vlbi") if isinstance(payload.get("vlbi"), dict) else {}
    llr = payload.get("llr") if isinstance(payload.get("llr"), dict) else {}
    messenger = payload.get("messenger") if isinstance(payload.get("messenger"), dict) else {}
    cross = payload.get("cross_channel") if isinstance(payload.get("cross_channel"), dict) else {}
    beta_terminal = payload.get("beta_terminal") if isinstance(payload.get("beta_terminal"), dict) else {}

    labels = ["VLBI channel", "LLR channel", "MESSENGER channel", "Cross channel", "Beta terminal gate"]
    statuses = [
        _normalize_status(vlbi.get("status")),
        _normalize_status(llr.get("status")),
        _normalize_status(messenger.get("status")),
        _normalize_status(cross.get("status")),
        _normalize_status(beta_terminal.get("status")),
    ]
    scores = [_status_to_score(s) for s in statuses]
    colors = [_status_color(s) for s in statuses]

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    y = np.arange(len(labels), dtype=float)
    ax.barh(y, scores, color=colors, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 3.0)
    ax.axvline(1.0, color="#999999", linestyle="--", linewidth=1.0)
    ax.axvline(2.0, color="#999999", linestyle="--", linewidth=1.0)
    ax.set_xlabel("status score (pass=0.5, watch=1.5, reject=2.8)")
    ax.set_title("Beta Cross-Channel Terminal Decision")
    ax.grid(axis="x", alpha=0.22)

    abs_z = _to_float(cross.get("beta_consistency_abs_z"))
    abs_z_1 = _to_float(beta_terminal.get("beta_combined_abs_z_minus_1"))
    beta_est = _to_float(beta_terminal.get("beta_combined_est"))
    beta_sig = _to_float(beta_terminal.get("beta_combined_sigma"))
    note_parts: List[str] = []
    # 条件分岐: `abs_z is not None` を満たす経路を評価する。
    if abs_z is not None:
        note_parts.append(f"|z(beta_vlbi-beta_llr)|={abs_z:.3f}")

    # 条件分岐: `beta_est is not None and beta_sig is not None` を満たす経路を評価する。

    if beta_est is not None and beta_sig is not None:
        note_parts.append(f"beta_comb={beta_est:.6f}±{beta_sig:.6f}")

    # 条件分岐: `abs_z_1 is not None` を満たす経路を評価する。

    if abs_z_1 is not None:
        note_parts.append(f"|z(beta_comb-1)|={abs_z_1:.3f}")

    active_policy_id = str(beta_terminal.get("active_policy_id") or "").strip()
    # 条件分岐: `active_policy_id` を満たす経路を評価する。
    if active_policy_id:
        note_parts.append(f"policy={active_policy_id}")

    messenger_replay_z = _to_float(messenger.get("stage_e_replay_z_delta_beta"))
    # 条件分岐: `messenger_replay_z is not None` を満たす経路を評価する。
    if messenger_replay_z is not None:
        note_parts.append(f"messenger_replay_z={messenger_replay_z:.3f}")

    policy_d_pair = (
        beta_terminal.get("policy_d_pair_decomposition")
        if isinstance(beta_terminal.get("policy_d_pair_decomposition"), dict)
        else {}
    )
    policy_d_pair_abs_z = _to_float(policy_d_pair.get("abs_z_pair"))
    policy_d_required_delta_watch = _to_float(policy_d_pair.get("required_abs_delta_watch"))
    policy_d_abs_delta = _to_float(policy_d_pair.get("abs_delta_beta"))
    # 条件分岐: `policy_d_pair_abs_z is not None` を満たす経路を評価する。
    if policy_d_pair_abs_z is not None:
        note_parts.append(f"policyD_pair_z={policy_d_pair_abs_z:.3f}")

    # 条件分岐: `policy_d_required_delta_watch is not None and policy_d_abs_delta is not None` を満たす経路を評価する。

    if policy_d_required_delta_watch is not None and policy_d_abs_delta is not None:
        note_parts.append(f"policyD_absDelta={policy_d_abs_delta:.3f}>watch<= {policy_d_required_delta_watch:.3f}")

    vlbi_bias = _normalize_status(vlbi.get("bias_audit_status"))
    llr_bias = _normalize_status(llr.get("bias_audit_status"))
    # 条件分岐: `vlbi_bias or llr_bias` を満たす経路を評価する。
    if vlbi_bias or llr_bias:
        note_parts.append(f"bias(vlbi/llr)={vlbi_bias}/{llr_bias}")

    note = " / ".join(note_parts) if note_parts else "|z|=NA"
    ax.text(0.02, 0.05, note, transform=ax.transAxes, ha="left", va="bottom", fontsize=10.0)

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
    ap = argparse.ArgumentParser(description="Build beta cross-channel (VLBI+LLR+MESSENGER) terminal registry.")
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

    out_dir.mkdir(parents=True, exist_ok=True)

    vlbi, vlbi_sources = _extract_vlbi_channel(_ROOT)
    llr, llr_sources = _extract_llr_channel(_ROOT)
    messenger, messenger_sources = _extract_messenger_channel(_ROOT)
    cross = _compute_cross_channel(vlbi, llr)
    beta_terminal = _compute_beta_terminal(vlbi, llr, messenger, cross)
    cross["status"] = _combine_status([_normalize_status(cross.get("status")), _normalize_status(beta_terminal.get("status"))])
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase": {"step": "8.7.47.20+8.7.48.registry_bridge+8.7.48.12+8.7.48.25"},
        "vlbi": vlbi,
        "llr": llr,
        "messenger": messenger,
        "cross_channel": cross,
        "beta_terminal": beta_terminal,
        "source_paths": {
            "vlbi": vlbi_sources,
            "llr": llr_sources,
            "messenger": messenger_sources,
        },
    }

    out_json = out_dir / "beta_cross_channel_registry.json"
    out_csv = out_dir / "beta_cross_channel_registry.csv"
    out_pdf = out_dir / "beta_cross_channel_registry.pdf"
    out_png = out_dir / "beta_cross_channel_registry.png"

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    policy_d = (
        (beta_terminal.get("policy_matrix") or {}).get("policy_D_exclude_plus_messenger_fusion")
        if isinstance(beta_terminal.get("policy_matrix"), dict)
        else {}
    )
    policy_d_pair = (
        policy_d.get("pair_decomposition") if isinstance(policy_d, dict) and isinstance(policy_d.get("pair_decomposition"), dict) else {}
    )
    policy_d_stage_i = (
        policy_d.get("stage_i_mapping") if isinstance(policy_d, dict) and isinstance(policy_d.get("stage_i_mapping"), dict) else {}
    )
    policy_governance = (
        beta_terminal.get("policy_governance")
        if isinstance(beta_terminal.get("policy_governance"), dict)
        else {}
    )
    stage_i_decomp = (
        messenger.get("stage_i_beta_dyn_decomposition")
        if isinstance(messenger.get("stage_i_beta_dyn_decomposition"), dict)
        else {}
    )
    stage_i_priority = (
        messenger.get("stage_i_nuisance_priority")
        if isinstance(messenger.get("stage_i_nuisance_priority"), dict)
        else {}
    )
    top_mitigation = (
        stage_i_priority.get("mitigation_priority")[0]
        if isinstance(stage_i_priority.get("mitigation_priority"), list) and stage_i_priority.get("mitigation_priority")
        else {}
    )
    top_stabilization = (
        stage_i_priority.get("stabilization_priority")[0]
        if isinstance(stage_i_priority.get("stabilization_priority"), list) and stage_i_priority.get("stabilization_priority")
        else {}
    )
    stage_i_odf = stage_i_decomp.get("odf") if isinstance(stage_i_decomp.get("odf"), dict) else {}
    stage_i_tnf = stage_i_decomp.get("tnf") if isinstance(stage_i_decomp.get("tnf"), dict) else {}
    csv_rows = [
        {"channel": "vlbi", "status": vlbi.get("status"), "metric": "threshold_chi2_dof", "value": vlbi.get("threshold_chi2_dof"), "note": ""},
        {"channel": "vlbi", "status": vlbi.get("status"), "metric": "stable_chi2_dof", "value": vlbi.get("stable_chi2_dof"), "note": ""},
        {"channel": "vlbi", "status": vlbi.get("status"), "metric": "subset_refit_status", "value": vlbi.get("subset_refit_status"), "note": vlbi.get("subset_refit_reason", "")},
        {"channel": "vlbi", "status": vlbi.get("status"), "metric": "subset_refit_baseline_abs_z_vs_llr", "value": vlbi.get("subset_refit_baseline_abs_z_vs_llr"), "note": ""},
        {"channel": "llr", "status": llr.get("status"), "metric": "kappa_est", "value": llr.get("kappa_est"), "note": ""},
        {"channel": "llr", "status": llr.get("status"), "metric": "abs_z", "value": llr.get("abs_z"), "note": ""},
        {"channel": "messenger", "status": messenger.get("status"), "metric": "stage_j_status", "value": messenger.get("stage_j_status"), "note": ""},
        {"channel": "messenger", "status": messenger.get("status"), "metric": "stage_e_replay_status", "value": messenger.get("stage_e_replay_status"), "note": ""},
        {"channel": "messenger", "status": messenger.get("status"), "metric": "stage_e_replay_z_delta_beta", "value": messenger.get("stage_e_replay_z_delta_beta"), "note": ""},
        {"channel": "messenger", "status": messenger.get("status"), "metric": "beta_primary_definition", "value": messenger.get("beta_primary_definition"), "note": messenger.get("beta_primary_source", "")},
        {"channel": "messenger", "status": messenger.get("status"), "metric": "beta_primary_est", "value": messenger.get("beta_primary_est"), "note": ""},
        {"channel": "messenger", "status": messenger.get("status"), "metric": "beta_primary_sigma", "value": messenger.get("beta_primary_sigma"), "note": ""},
        {"channel": "messenger", "status": messenger.get("status"), "metric": "beta_dyn_diagnostic_est", "value": messenger.get("beta_dyn_diagnostic_est"), "note": ""},
        {"channel": "messenger", "status": messenger.get("status"), "metric": "beta_dyn_diagnostic_sigma", "value": messenger.get("beta_dyn_diagnostic_sigma"), "note": ""},
        {"channel": "messenger", "status": messenger.get("status"), "metric": "stage_i_odf_baseline_abs_z_beta_minus_1", "value": stage_i_odf.get("baseline_abs_z_beta_minus_1"), "note": ""},
        {"channel": "messenger", "status": messenger.get("status"), "metric": "stage_i_odf_best_abs_z_beta_minus_1", "value": stage_i_odf.get("best_abs_z_beta_minus_1"), "note": str(stage_i_odf.get("best_abs_z_scenario_id") or "")},
        {"channel": "messenger", "status": messenger.get("status"), "metric": "stage_i_odf_max_shift_delta_vs_baseline_z", "value": stage_i_odf.get("max_shift_delta_vs_baseline_z"), "note": str(stage_i_odf.get("max_shift_scenario_id") or "")},
        {"channel": "messenger", "status": messenger.get("status"), "metric": "stage_i_tnf_baseline_abs_z_beta_minus_1", "value": stage_i_tnf.get("baseline_abs_z_beta_minus_1"), "note": ""},
        {"channel": "messenger", "status": messenger.get("status"), "metric": "stage_i_tnf_best_abs_z_beta_minus_1", "value": stage_i_tnf.get("best_abs_z_beta_minus_1"), "note": str(stage_i_tnf.get("best_abs_z_scenario_id") or "")},
        {"channel": "messenger", "status": messenger.get("status"), "metric": "stage_i_tnf_max_shift_delta_vs_baseline_z", "value": stage_i_tnf.get("max_shift_delta_vs_baseline_z"), "note": str(stage_i_tnf.get("max_shift_scenario_id") or "")},
        {"channel": "messenger", "status": messenger.get("status"), "metric": "stage_i_nuisance_priority_status", "value": stage_i_priority.get("priority_status"), "note": str(stage_i_priority.get("method") or "")},
        {"channel": "messenger", "status": messenger.get("status"), "metric": "stage_i_nuisance_top1_scenario", "value": top_mitigation.get("scenario_id"), "note": str(top_mitigation.get("recommended_action") or "")},
        {"channel": "messenger", "status": messenger.get("status"), "metric": "stage_i_nuisance_top1_max_abs_z_delta", "value": top_mitigation.get("max_abs_z_delta"), "note": str(top_mitigation.get("scenario_group") or "")},
        {"channel": "messenger", "status": messenger.get("status"), "metric": "stage_i_stabilization_top1_scenario", "value": top_stabilization.get("scenario_id"), "note": str(top_stabilization.get("recommended_action") or "")},
        {"channel": "messenger", "status": messenger.get("status"), "metric": "stage_i_srp_proxy_registered", "value": stage_i_priority.get("srp_proxy_registered"), "note": str(stage_i_priority.get("srp_proxy_note") or "")},
        {"channel": "messenger", "status": messenger.get("status"), "metric": "comparator_eligible", "value": messenger.get("comparator_eligible"), "note": messenger.get("eligibility_reason", "")},
        {"channel": "messenger", "status": messenger.get("status"), "metric": "policy_note", "value": "", "note": messenger.get("policy_note", "")},
        {
            "channel": "cross_channel",
            "status": cross.get("status"),
            "metric": "beta_consistency_abs_z",
            "value": cross.get("beta_consistency_abs_z"),
            "note": "",
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "beta_combined_est",
            "value": beta_terminal.get("beta_combined_est"),
            "note": "",
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "beta_combined_sigma",
            "value": beta_terminal.get("beta_combined_sigma"),
            "note": "",
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "beta_combined_abs_z_minus_1",
            "value": beta_terminal.get("beta_combined_abs_z_minus_1"),
            "note": "",
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "active_policy_id",
            "value": beta_terminal.get("active_policy_id"),
            "note": beta_terminal.get("active_policy_selection_reason", ""),
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "policy_A_status",
            "value": ((beta_terminal.get("policy_matrix") or {}).get("policy_A_hard_reject_keep") or {}).get("status"),
            "note": "",
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "policy_B_status",
            "value": ((beta_terminal.get("policy_matrix") or {}).get("policy_B_exclude_comparator_when_ineligible") or {}).get("status"),
            "note": "",
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "policy_B_hold_status",
            "value": policy_governance.get("policy_b_hold_status"),
            "note": "",
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "policy_D_status",
            "value": (policy_d or {}).get("status"),
            "note": "",
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "policy_D_promotion_status",
            "value": policy_governance.get("policy_d_promotion_status"),
            "note": "",
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "policy_D_promotion_ready",
            "value": policy_governance.get("policy_d_promotion_ready"),
            "note": "",
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "policy_D_llr_gate_order_status",
            "value": policy_governance.get("llr_gate_order_status"),
            "note": "",
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "policy_D_llr_gate_execution_order",
            "value": "->".join(str(x) for x in (policy_governance.get("llr_gate_execution_order") or [])),
            "note": "",
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "policy_D_llr_gate_actions_ordered",
            "value": "->".join(str(x) for x in (policy_governance.get("llr_gate_actions_ordered") or [])),
            "note": "",
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "policy_D_llr_gate_repro_commands_min",
            "value": " | ".join(str(x) for x in (policy_governance.get("llr_gate_repro_commands_min") or [])),
            "note": "minimal_repro_order",
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "policy_D_reassessment_status",
            "value": (
                (policy_governance.get("policy_d_promotion_reassessment") or {}).get("status")
                if isinstance(policy_governance.get("policy_d_promotion_reassessment"), dict)
                else ""
            ),
            "note": (
                f"order_match={str((policy_governance.get('policy_d_promotion_reassessment') or {}).get('blocker_order_alignment'))}, "
                f"count_delta={str((policy_governance.get('policy_d_promotion_reassessment') or {}).get('blocker_count_delta'))}"
                if isinstance(policy_governance.get("policy_d_promotion_reassessment"), dict)
                else ""
            ),
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "policy_recommended_active_policy_id",
            "value": policy_governance.get("recommended_active_policy_id"),
            "note": policy_governance.get("recommended_active_policy_reason", ""),
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "policy_switch_decision_id",
            "value": (
                (policy_governance.get("policy_switch_decision") or {}).get("decision_id")
                if isinstance(policy_governance.get("policy_switch_decision"), dict)
                else ""
            ),
            "note": (
                (policy_governance.get("policy_switch_decision") or {}).get("hold_reason")
                if isinstance(policy_governance.get("policy_switch_decision"), dict)
                else ""
            ),
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "policy_switch_required_now",
            "value": (
                (policy_governance.get("policy_switch_decision") or {}).get("switch_required_now")
                if isinstance(policy_governance.get("policy_switch_decision"), dict)
                else ""
            ),
            "note": "",
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "policy_switch_allowed_now",
            "value": (
                (policy_governance.get("policy_switch_decision") or {}).get("switch_allowed_now")
                if isinstance(policy_governance.get("policy_switch_decision"), dict)
                else ""
            ),
            "note": "",
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "policy_terminal_watch_statement_id",
            "value": (
                (policy_governance.get("policy_terminal_watch_statement") or {}).get("statement_id")
                if isinstance(policy_governance.get("policy_terminal_watch_statement"), dict)
                else ""
            ),
            "note": "",
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "policy_terminal_watch_statement_text",
            "value": (
                (policy_governance.get("policy_terminal_watch_statement") or {}).get("statement_text")
                if isinstance(policy_governance.get("policy_terminal_watch_statement"), dict)
                else ""
            ),
            "note": "8.7.48.30_canonical_watch_statement",
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "policy_D_promotion_blockers",
            "value": ",".join(str(x) for x in (policy_governance.get("policy_d_promotion_blockers") or [])),
            "note": "",
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "policy_D_blocker_order_status",
            "value": policy_governance.get("policy_d_promotion_blocker_order_status"),
            "note": "",
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "policy_D_blocker_resolution_order",
            "value": ",".join(
                str(x) for x in (policy_governance.get("policy_d_promotion_blocker_resolution_order") or [])
            ),
            "note": "",
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "policy_D_blocker_priority_top1",
            "value": (
                ((policy_governance.get("policy_d_promotion_blocker_priority") or [])[0] or {}).get("blocker_id")
                if isinstance(policy_governance.get("policy_d_promotion_blocker_priority"), list)
                and len(policy_governance.get("policy_d_promotion_blocker_priority") or []) > 0
                else ""
            ),
            "note": (
                ((policy_governance.get("policy_d_promotion_blocker_priority") or [])[0] or {}).get("recommended_action")
                if isinstance(policy_governance.get("policy_d_promotion_blocker_priority"), list)
                and len(policy_governance.get("policy_d_promotion_blocker_priority") or []) > 0
                else ""
            ),
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "policy_D_pair_abs_z_llr_messenger",
            "value": (policy_d or {}).get("pair_abs_z_llr_messenger"),
            "note": str((policy_d or {}).get("pair_status_llr_messenger") or ""),
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "policy_D_pair_abs_delta_beta",
            "value": policy_d_pair.get("abs_delta_beta"),
            "note": "",
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "policy_D_pair_required_abs_delta_watch",
            "value": policy_d_pair.get("required_abs_delta_watch"),
            "note": "",
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "policy_D_pair_watch_gate_status",
            "value": policy_d_pair.get("watch_gate_status"),
            "note": "pass if pair_abs_z<=3",
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "policy_D_pair_pass_gate_status",
            "value": policy_d_pair.get("pass_gate_status"),
            "note": "pass if pair_abs_z<=2",
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "policy_D_pair_gap_to_watch_z",
            "value": policy_d_pair.get("gap_to_watch_z"),
            "note": "",
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "policy_D_pair_gap_to_pass_z",
            "value": policy_d_pair.get("gap_to_pass_z"),
            "note": "",
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "policy_D_pair_gap_abs_delta_watch",
            "value": policy_d_pair.get("gap_abs_delta_watch"),
            "note": "",
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "policy_D_pair_gap_abs_delta_pass",
            "value": policy_d_pair.get("gap_abs_delta_pass"),
            "note": "",
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "policy_D_required_messenger_beta_min_watch",
            "value": policy_d_pair.get("required_candidate_beta_min_watch"),
            "note": "",
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "policy_D_required_messenger_beta_max_watch",
            "value": policy_d_pair.get("required_candidate_beta_max_watch"),
            "note": "",
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "policy_D_required_messenger_sigma_watch",
            "value": policy_d_pair.get("required_candidate_sigma_watch"),
            "note": "",
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "policy_D_stage_i_best_candidate_branch",
            "value": policy_d_stage_i.get("best_candidate_branch"),
            "note": "",
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "policy_D_stage_i_best_candidate_shift_needed_to_watch_range",
            "value": policy_d_stage_i.get("best_candidate_shift_needed_to_watch_range"),
            "note": "",
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "policy_D_stage_i_watch_gate_reachable_status",
            "value": policy_d_stage_i.get("watch_gate_reachable_status"),
            "note": "",
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "policy_E_status",
            "value": ((beta_terminal.get("policy_matrix") or {}).get("policy_E_exclude_messenger_only") or {}).get("status"),
            "note": "",
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "vlbi_bias_audit_status",
            "value": vlbi.get("bias_audit_status"),
            "note": "",
        },
        {
            "channel": "beta_terminal",
            "status": beta_terminal.get("status"),
            "metric": "llr_bias_audit_status",
            "value": llr.get("bias_audit_status"),
            "note": "",
        },
    ]
    _write_csv_rows(out_csv, csv_rows)
    _write_plot(payload, out_pdf=out_pdf, out_png=out_png)

    produced = [out_json, out_csv, out_pdf, out_png]
    synced = _sync_outputs_to_public(produced, private_root=out_dir, public_root=public_dir)
    print(f"[ok] wrote: {out_json}")
    print(f"[ok] wrote: {out_csv}")
    print(f"[ok] wrote: {out_pdf}")
    print(f"[ok] wrote: {out_png}")
    print(f"[ok] synced_to_public: {len(synced)} files")
    print(
        f"[summary] vlbi={_normalize_status(vlbi.get('status'))} "
        f"llr={_normalize_status(llr.get('status'))} "
        f"messenger={_normalize_status(messenger.get('status'))} "
        f"cross={_normalize_status(cross.get('status'))} "
        f"beta_terminal={_normalize_status(beta_terminal.get('status'))}"
    )
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
