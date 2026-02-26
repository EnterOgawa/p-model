#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
lagrangian_noether_rotational_closure_audit.py

Step 8.7.21.6 / 8.7.26.2:
回転結合の最小作用項 L_rot について、lambda_rot を独立事前拘束で固定し、
GP-B/LAGEOS の frame-dragging を「再構成一致」ではなく
「holdout 予言一致」で監査する。

出力:
  - output/public/quantum/lagrangian_noether_rotational_closure_audit.json
  - output/public/quantum/lagrangian_noether_rotational_closure_audit.csv
  - output/public/quantum/lagrangian_noether_rotational_closure_audit.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(ROOT) not in sys.path` を満たす経路を評価する。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summary import worklog  # noqa: E402


DEFAULT_PRIOR_SOURCE_IDS = ["lageos_frame_dragging"]


# 関数: `_iso_utc_now` の入出力契約と処理意図を定義する。
def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_rel` の入出力契約と処理意図を定義する。

def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        return {}

    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_to_float` の入出力契約と処理意図を定義する。

def _to_float(v: Any) -> Optional[float]:
    try:
        val = float(v)
    except Exception:
        return None

    # 条件分岐: `math.isnan(val) or math.isinf(val)` を満たす経路を評価する。

    if math.isnan(val) or math.isinf(val):
        return None

    return val


# 関数: `_status_from_pass` の入出力契約と処理意図を定義する。

def _status_from_pass(passed: Optional[bool], gate_level: str) -> str:
    # 条件分岐: `passed is True` を満たす経路を評価する。
    if passed is True:
        return "pass"

    # 条件分岐: `passed is None` を満たす経路を評価する。

    if passed is None:
        return "unknown"

    # 条件分岐: `gate_level == "hard"` を満たす経路を評価する。

    if gate_level == "hard":
        return "reject"

    return "watch"


# 関数: `_score_from_status` の入出力契約と処理意図を定義する。

def _score_from_status(status: str) -> float:
    # 条件分岐: `status == "pass"` を満たす経路を評価する。
    if status == "pass":
        return 1.0

    # 条件分岐: `status == "watch"` を満たす経路を評価する。

    if status == "watch":
        return 0.5

    return 0.0


# 関数: `_pick_frame_rows` の入出力契約と処理意図を定義する。

def _pick_frame_rows(rot_payload: Dict[str, Any], branch_name: str) -> List[Dict[str, Any]]:
    branches = rot_payload.get("branches") if isinstance(rot_payload.get("branches"), dict) else {}
    branch = branches.get(branch_name) if isinstance(branches, dict) else {}
    rows = branch.get("rows") if isinstance(branch, dict) else []
    # 条件分岐: `not isinstance(rows, list)` を満たす経路を評価する。
    if not isinstance(rows, list):
        return []

    return [r for r in rows if isinstance(r, dict) and str(r.get("kind") or "") == "frame_dragging"]


# 関数: `_derive_prior_from_channels` の入出力契約と処理意図を定義する。

def _derive_prior_from_channels(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    for preferred_id in DEFAULT_PRIOR_SOURCE_IDS:
        for row in rows:
            # 条件分岐: `str(row.get("id") or "") != preferred_id` を満たす経路を評価する。
            if str(row.get("id") or "") != preferred_id:
                continue

            observed = _to_float(row.get("observed"))
            sigma = _to_float(row.get("observed_sigma"))
            ref = _to_float(row.get("reference_prediction"))
            # 条件分岐: `observed is None or sigma is None or sigma <= 0.0 or ref is None or abs(ref)...` を満たす経路を評価する。
            if observed is None or sigma is None or sigma <= 0.0 or ref is None or abs(ref) <= 0.0:
                continue

            lambda_mean = observed / ref - 1.0
            lambda_sigma = abs(sigma / ref)
            return {
                "source": "fallback_from_rotating_audit_channel",
                "mode": "single_channel_external_proxy",
                "source_channel_ids": [preferred_id],
                "lambda_rot_mean": float(lambda_mean),
                "lambda_rot_sigma": float(lambda_sigma),
                "kappa_rot_mean": float(1.0 + lambda_mean),
                "kappa_rot_sigma": float(lambda_sigma),
                "note": "Derived from one weak-field spin channel only (used as independent prior for holdout prediction gate).",
            }

    for row in rows:
        observed = _to_float(row.get("observed"))
        sigma = _to_float(row.get("observed_sigma"))
        ref = _to_float(row.get("reference_prediction"))
        rid = str(row.get("id") or "")
        # 条件分岐: `observed is None or sigma is None or sigma <= 0.0 or ref is None or abs(ref)...` を満たす経路を評価する。
        if observed is None or sigma is None or sigma <= 0.0 or ref is None or abs(ref) <= 0.0 or not rid:
            continue

        lambda_mean = observed / ref - 1.0
        lambda_sigma = abs(sigma / ref)
        return {
            "source": "fallback_from_rotating_audit_channel",
            "mode": "single_channel_external_proxy",
            "source_channel_ids": [rid],
            "lambda_rot_mean": float(lambda_mean),
            "lambda_rot_sigma": float(lambda_sigma),
            "kappa_rot_mean": float(1.0 + lambda_mean),
            "kappa_rot_sigma": float(lambda_sigma),
            "note": "Derived from one available frame-dragging channel (fallback).",
        }

    return {
        "source": "fallback_failed",
        "mode": "none",
        "source_channel_ids": [],
        "lambda_rot_mean": None,
        "lambda_rot_sigma": None,
        "kappa_rot_mean": None,
        "kappa_rot_sigma": None,
        "note": "No finite channel was available to derive a fallback prior.",
    }


# 関数: `_load_prior` の入出力契約と処理意図を定義する。

def _load_prior(prior_json: Path, channel_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    # 条件分岐: `not prior_json.exists()` を満たす経路を評価する。
    if not prior_json.exists():
        fallback = _derive_prior_from_channels(channel_rows)
        fallback["loaded_from"] = None
        return fallback

    payload = _read_json(prior_json)
    prior_obj = payload.get("lambda_rot_prior") if isinstance(payload.get("lambda_rot_prior"), dict) else payload
    # 条件分岐: `not isinstance(prior_obj, dict)` を満たす経路を評価する。
    if not isinstance(prior_obj, dict):
        prior_obj = {}

    lambda_mean = _to_float(
        prior_obj.get("lambda_rot_mean")
        if prior_obj.get("lambda_rot_mean") is not None
        else prior_obj.get("lambda_rot")
    )
    lambda_sigma = _to_float(
        prior_obj.get("lambda_rot_sigma")
        if prior_obj.get("lambda_rot_sigma") is not None
        else prior_obj.get("lambda_sigma")
    )

    kappa_mean = _to_float(prior_obj.get("kappa_rot_mean"))
    # 条件分岐: `kappa_mean is None and lambda_mean is not None` を満たす経路を評価する。
    if kappa_mean is None and lambda_mean is not None:
        kappa_mean = 1.0 + lambda_mean

    kappa_sigma = _to_float(prior_obj.get("kappa_rot_sigma"))
    # 条件分岐: `kappa_sigma is None` を満たす経路を評価する。
    if kappa_sigma is None:
        kappa_sigma = lambda_sigma

    source_ids_raw = prior_obj.get("source_channel_ids")
    source_ids: List[str] = []
    # 条件分岐: `isinstance(source_ids_raw, list)` を満たす経路を評価する。
    if isinstance(source_ids_raw, list):
        for item in source_ids_raw:
            sid = str(item or "").strip()
            # 条件分岐: `sid` を満たす経路を評価する。
            if sid:
                source_ids.append(sid)

    # 条件分岐: `not source_ids` を満たす経路を評価する。

    if not source_ids:
        source_ids = list(DEFAULT_PRIOR_SOURCE_IDS)

    return {
        "source": "external_prior_json",
        "mode": str(prior_obj.get("mode") or payload.get("mode") or "external_prior"),
        "source_channel_ids": source_ids,
        "lambda_rot_mean": lambda_mean,
        "lambda_rot_sigma": lambda_sigma,
        "kappa_rot_mean": kappa_mean,
        "kappa_rot_sigma": kappa_sigma,
        "note": str(prior_obj.get("note") or payload.get("note") or ""),
        "loaded_from": _rel(prior_json),
    }


# 関数: `_build_channel_rows` の入出力契約と処理意図を定義する。

def _build_channel_rows(
    *,
    static_rows: List[Dict[str, Any]],
    vortex_rows: List[Dict[str, Any]],
    lambda_rot_prior: Optional[float],
    prior_source_ids: List[str],
) -> List[Dict[str, Any]]:
    by_id_static = {str(r.get("id") or ""): r for r in static_rows}
    by_id_vortex = {str(r.get("id") or ""): r for r in vortex_rows}
    ids = [cid for cid in by_id_vortex.keys() if cid]
    rows: List[Dict[str, Any]] = []
    for cid in ids:
        rv = by_id_vortex.get(cid, {})
        rs = by_id_static.get(cid, {})
        observed = _to_float(rv.get("observed"))
        sigma = _to_float(rv.get("observed_sigma"))
        ref = _to_float(rv.get("reference_prediction"))
        pred_vortex_fit = _to_float(rv.get("pmodel_prediction"))
        pred_lrot_prior = None
        # 条件分岐: `lambda_rot_prior is not None and ref is not None` を満たす経路を評価する。
        if lambda_rot_prior is not None and ref is not None:
            pred_lrot_prior = (1.0 + lambda_rot_prior) * ref

        residual_lrot_prior = None if observed is None or pred_lrot_prior is None else observed - pred_lrot_prior
        z_lrot_prior = None if residual_lrot_prior is None or sigma is None or sigma <= 0 else residual_lrot_prior / sigma
        is_prior_source = cid in prior_source_ids
        rows.append(
            {
                "id": cid,
                "label": str(rv.get("label") or cid),
                "experiment": str(rv.get("experiment") or ""),
                "unit": str(rv.get("unit") or ""),
                "observed": observed,
                "observed_sigma": sigma,
                "reference_prediction": ref,
                "pred_static": _to_float(rs.get("pmodel_prediction")),
                "pred_vortex_fit": pred_vortex_fit,
                "pred_lrot_prior": pred_lrot_prior,
                "pred_lrot": pred_lrot_prior,
                "residual_lrot_prior": residual_lrot_prior,
                "residual_lrot": residual_lrot_prior,
                "z_static": _to_float(rs.get("z_score")),
                "z_lrot_prior": z_lrot_prior,
                "z_lrot": z_lrot_prior,
                "is_prior_source": is_prior_source,
                "gate_role": "prior_source" if is_prior_source else "holdout_prediction",
            }
        )

    return rows


# 関数: `build_payload` の入出力契約と処理意図を定義する。

def build_payload(
    *,
    rotating_json: Path,
    closure_json: Path,
    drift_json: Path,
    prior_json: Path,
) -> Dict[str, Any]:
    rot = _read_json(rotating_json)
    closure = _read_json(closure_json)
    drift = _read_json(drift_json)

    calibration = rot.get("calibration") if isinstance(rot.get("calibration"), dict) else {}
    kappa_rot_fit_legacy = _to_float(calibration.get("kappa_rot"))
    kappa_sigma_fit_legacy = _to_float(calibration.get("kappa_sigma"))
    lambda_rot_fit_legacy = None if kappa_rot_fit_legacy is None else (kappa_rot_fit_legacy - 1.0)
    lambda_sigma_fit_legacy = kappa_sigma_fit_legacy

    static_summary = (
        (((rot.get("branches") or {}).get("static_iso") or {}).get("summary"))
        if isinstance(rot.get("branches"), dict)
        else {}
    )
    vortex_summary = (
        (((rot.get("branches") or {}).get("vortex_gradient") or {}).get("summary"))
        if isinstance(rot.get("branches"), dict)
        else {}
    )
    # 条件分岐: `not isinstance(static_summary, dict)` を満たす経路を評価する。
    if not isinstance(static_summary, dict):
        static_summary = {}

    # 条件分岐: `not isinstance(vortex_summary, dict)` を満たす経路を評価する。

    if not isinstance(vortex_summary, dict):
        vortex_summary = {}

    static_rows = _pick_frame_rows(rot, "static_iso")
    vortex_rows = _pick_frame_rows(rot, "vortex_gradient")
    prior = _load_prior(prior_json, vortex_rows)
    lambda_rot_prior = _to_float(prior.get("lambda_rot_mean"))
    lambda_sigma_prior = _to_float(prior.get("lambda_rot_sigma"))
    kappa_rot_prior = _to_float(prior.get("kappa_rot_mean"))
    kappa_sigma_prior = _to_float(prior.get("kappa_rot_sigma"))
    prior_source_ids = [str(x) for x in (prior.get("source_channel_ids") or []) if str(x or "")]
    channel_rows = _build_channel_rows(
        static_rows=static_rows,
        vortex_rows=vortex_rows,
        lambda_rot_prior=lambda_rot_prior,
        prior_source_ids=prior_source_ids,
    )

    closure_decision = closure.get("decision") if isinstance(closure.get("decision"), dict) else {}
    drift_decision = drift.get("decision") if isinstance(drift.get("decision"), dict) else {}

    fit_vs_prior_diffs: List[float] = []
    for row in channel_rows:
        pred_lrot = _to_float(row.get("pred_lrot_prior"))
        pred_vortex = _to_float(row.get("pred_vortex_fit"))
        # 条件分岐: `pred_lrot is None or pred_vortex is None` を満たす経路を評価する。
        if pred_lrot is None or pred_vortex is None:
            continue

        fit_vs_prior_diffs.append(abs(pred_lrot - pred_vortex))

    max_fit_vs_prior_delta = max(fit_vs_prior_diffs) if fit_vs_prior_diffs else None

    prior_source_rows = [r for r in channel_rows if bool(r.get("is_prior_source"))]
    holdout_rows = [r for r in channel_rows if not bool(r.get("is_prior_source"))]
    holdout_z_abs: List[float] = []
    holdout_reject_n = 0
    z_reject = 3.0
    for row in holdout_rows:
        z = _to_float(row.get("z_lrot_prior"))
        # 条件分岐: `z is None` を満たす経路を評価する。
        if z is None:
            continue

        abs_z = abs(z)
        holdout_z_abs.append(abs_z)
        # 条件分岐: `abs_z > z_reject` を満たす経路を評価する。
        if abs_z > z_reject:
            holdout_reject_n += 1

    holdout_max_abs_z = max(holdout_z_abs) if holdout_z_abs else None

    checks: List[Dict[str, Any]] = []

    # 関数: `add_check` の入出力契約と処理意図を定義する。
    def add_check(
        cid: str,
        metric: str,
        value: Any,
        expected: Any,
        passed: Optional[bool],
        gate_level: str,
        note: str,
    ) -> None:
        status = _status_from_pass(passed, gate_level)
        checks.append(
            {
                "id": cid,
                "metric": metric,
                "value": value,
                "expected": expected,
                "pass": passed,
                "gate_level": gate_level,
                "status": status,
                "score": _score_from_status(status),
                "note": note,
            }
        )

    add_check(
        "l_rot::input_channels",
        "frame_dragging_channels_n",
        len(channel_rows),
        ">=2",
        len(channel_rows) >= 2,
        "hard",
        "GP-B/LAGEOS の2チャネル以上が必要。",
    )
    add_check(
        "l_rot::static_reject",
        "static_iso_frame_reject_n/frame_channels_n",
        f"{int(static_summary.get('reject_n') or 0)}/{int(static_summary.get('frame_channels_n') or 0)}",
        "all reject",
        (
            int(static_summary.get("frame_channels_n") or 0) > 0
            and int(static_summary.get("reject_n") or 0) == int(static_summary.get("frame_channels_n") or 0)
        ),
        "hard",
        "δP_rot=0 分岐が frame-dragging を棄却していること。",
    )
    add_check(
        "l_rot::lambda_defined",
        "lambda_rot (prior mean)",
        lambda_rot_prior,
        "finite",
        lambda_rot_prior is not None,
        "hard",
        "独立事前拘束で λ_rot が定義されること。",
    )
    add_check(
        "l_rot::lambda_sigma",
        "sigma(lambda_rot prior)",
        lambda_sigma_prior,
        ">0",
        (lambda_sigma_prior is not None and lambda_sigma_prior > 0.0),
        "hard",
        "独立事前拘束の不確かさが正。",
    )
    add_check(
        "l_rot::prior_source_channels",
        "prior_source_channels_n",
        len(prior_source_rows),
        ">=1",
        len(prior_source_rows) >= 1,
        "hard",
        "弱場スピン指標を独立事前拘束の固定源として少なくとも1チャネル確保すること。",
    )
    add_check(
        "l_rot::prediction_holdout_channels",
        "holdout_prediction_channels_n",
        len(holdout_rows),
        ">=1",
        len(holdout_rows) >= 1,
        "hard",
        "事前拘束に使っていない holdout チャネルで予言一致を検証すること。",
    )
    add_check(
        "l_rot::prediction_holdout_reject",
        "holdout_prediction_reject_n",
        holdout_reject_n,
        "0",
        holdout_reject_n == 0,
        "hard",
        "holdout 予言チャネルで棄却がゼロであること。",
    )
    add_check(
        "l_rot::prediction_holdout_max_abs_z",
        "holdout_prediction_max_abs_z",
        holdout_max_abs_z,
        "<=3",
        (holdout_max_abs_z is not None and holdout_max_abs_z <= z_reject),
        "hard",
        "holdout 予言チャネルの |z| が 3 以下であること。",
    )
    add_check(
        "l_rot::closure_upstream",
        "lagrangian_noether_observable_closure.overall_status",
        str(closure_decision.get("overall_status") or ""),
        "pass",
        str(closure_decision.get("overall_status") or "") == "pass",
        "hard",
        "上流の L_total 閉包監査が pass。",
    )
    drift_overall = str(drift_decision.get("overall_status") or "")
    drift_recalc = bool(drift_decision.get("recalc_required"))
    add_check(
        "l_rot::drift_guard",
        "closure_drift(overall_status,recalc_required)",
        {"overall_status": drift_overall, "recalc_required": drift_recalc},
        "status in {pass,watch} and recalc_required=false",
        drift_overall in {"pass", "watch"} and (not drift_recalc),
        "hard",
        "drift監査で再計算トリガーが立っていないこと。",
    )
    add_check(
        "l_rot::small_coupling_watch",
        "abs(lambda_rot)",
        None if lambda_rot_prior is None else abs(lambda_rot_prior),
        "<=0.2",
        (lambda_rot_prior is not None and abs(lambda_rot_prior) <= 0.2),
        "watch",
        "最小拡張の範囲（過大結合回避）の運用監視。",
    )
    combined_sigma = None
    # 条件分岐: `lambda_sigma_prior is not None and lambda_sigma_fit_legacy is not None` を満たす経路を評価する。
    if lambda_sigma_prior is not None and lambda_sigma_fit_legacy is not None:
        combined_sigma = math.sqrt(lambda_sigma_prior * lambda_sigma_prior + lambda_sigma_fit_legacy * lambda_sigma_fit_legacy)

    prior_fit_delta = None
    # 条件分岐: `lambda_rot_prior is not None and lambda_rot_fit_legacy is not None` を満たす経路を評価する。
    if lambda_rot_prior is not None and lambda_rot_fit_legacy is not None:
        prior_fit_delta = lambda_rot_prior - lambda_rot_fit_legacy

    add_check(
        "l_rot::prior_vs_legacy_watch",
        "abs(lambda_prior - lambda_fit_legacy)",
        None if prior_fit_delta is None else abs(prior_fit_delta),
        "<=2*sqrt(sigma_prior^2 + sigma_fit_legacy^2)",
        (
            prior_fit_delta is not None
            and combined_sigma is not None
            and abs(prior_fit_delta) <= (2.0 * combined_sigma)
        ),
        "watch",
        "独立事前拘束と旧fitの乖離が過大でないこと（運用監視）。",
    )
    add_check(
        "l_rot::prior_source_role_watch",
        "prior_source_mode",
        str(prior.get("mode") or ""),
        "external_prior",
        str(prior.get("source") or "") == "external_prior_json",
        "watch",
        "独立事前拘束が外部JSONから固定されていること（fallback時はwatch）。",
    )

    hard_fail_ids = [str(c["id"]) for c in checks if c.get("gate_level") == "hard" and c.get("pass") is not True]
    watch_ids = [str(c["id"]) for c in checks if c.get("gate_level") == "watch" and c.get("pass") is not True]
    # 条件分岐: `hard_fail_ids` を満たす経路を評価する。
    if hard_fail_ids:
        overall_status = "reject"
        decision = "l_rot_closure_rejected"
    # 条件分岐: 前段条件が不成立で、`watch_ids` を追加評価する。
    elif watch_ids:
        overall_status = "watch"
        decision = "l_rot_closure_watch"
    else:
        overall_status = "pass"
        decision = "l_rot_closure_pass"

    payload: Dict[str, Any] = {
        "generated_utc": _iso_utc_now(),
        "phase": {"phase": 8, "step": "8.7.21.6+8.7.26.2", "name": "L_rot prior-fixed prediction closure audit"},
        "intent": (
            "Fix lambda_rot by an independent prior source and audit frame-dragging by "
            "holdout prediction consistency in the L_total route."
        ),
        "equations": {
            "l_total_with_rot": "L_total^rot = L_total + L_rot",
            "l_rot_minimal": (
                "L_rot = -(M_chi^2 c^2 / 2) * lambda_rot * (R^6/r^6) * sin(theta)^2 * (J_hat · grad(chi))^2"
            ),
            "mapping": "Omega_LT^(P) = (1 + lambda_rot) * Omega_LT^(ref)",
            "coefficient_relation": "lambda_rot = kappa_rot - 1",
        },
        "inputs": {
            "rotating_sphere_audit_json": _rel(rotating_json),
            "lagrangian_noether_observable_closure_audit_json": _rel(closure_json),
            "lagrangian_noether_observable_closure_drift_audit_json": _rel(drift_json),
            "rotational_coupling_prior_json": _rel(prior_json),
        },
        "calibration": {
            "kappa_rot": kappa_rot_prior,
            "kappa_sigma": kappa_sigma_prior,
            "lambda_rot": lambda_rot_prior,
            "lambda_sigma": lambda_sigma_prior,
            "prior_mode": str(prior.get("mode") or ""),
            "prior_source": str(prior.get("source") or ""),
            "prior_source_ids": prior_source_ids,
            "holdout_channel_ids": [str(r.get("id") or "") for r in holdout_rows if str(r.get("id") or "")],
            "fit_channels_n": len(prior_source_rows),
            "legacy_fit_channels_n": calibration.get("fit_channels_n"),
            "legacy_fit": {
                "kappa_rot_fit": kappa_rot_fit_legacy,
                "kappa_sigma_fit": kappa_sigma_fit_legacy,
                "lambda_rot_fit": lambda_rot_fit_legacy,
                "lambda_sigma_fit": lambda_sigma_fit_legacy,
            },
            "prior_vs_legacy": {
                "lambda_delta": prior_fit_delta,
                "combined_sigma": combined_sigma,
                "max_abs_pred_delta": max_fit_vs_prior_delta,
            },
            "note": str(prior.get("note") or ""),
        },
        "prediction_gate": {
            "z_reject": z_reject,
            "prior_source_channels_n": len(prior_source_rows),
            "holdout_channels_n": len(holdout_rows),
            "holdout_reject_n": holdout_reject_n,
            "holdout_max_abs_z": holdout_max_abs_z,
            "rule": "Use only holdout channels (not used in prior freeze) for the hard prediction gate.",
        },
        "channel_rows": channel_rows,
        "checks": checks,
        "decision": {
            "overall_status": overall_status,
            "decision": decision,
            "hard_fail_ids": hard_fail_ids,
            "watch_ids": watch_ids,
            "rule": "Reject if any hard check fails; watch if only watch-level checks fail; otherwise pass.",
        },
    }
    return payload


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "label",
                "experiment",
                "unit",
                "observed",
                "observed_sigma",
                "reference_prediction",
                "pred_static",
                "pred_vortex_fit",
                "pred_lrot_prior",
                "pred_lrot",
                "residual_lrot_prior",
                "residual_lrot",
                "z_static",
                "z_lrot_prior",
                "z_lrot",
                "is_prior_source",
                "gate_role",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# 関数: `_plot` の入出力契約と処理意図を定義する。

def _plot(path: Path, payload: Dict[str, Any]) -> None:
    channel_rows = payload.get("channel_rows") if isinstance(payload.get("channel_rows"), list) else []
    checks = payload.get("checks") if isinstance(payload.get("checks"), list) else []
    calib = payload.get("calibration") if isinstance(payload.get("calibration"), dict) else {}

    labels = [str(r.get("label") or r.get("id") or "") for r in channel_rows if isinstance(r, dict)]
    z_static = [float(r.get("z_static") or 0.0) for r in channel_rows if isinstance(r, dict)]
    z_lrot = [float(r.get("z_lrot_prior") or 0.0) for r in channel_rows if isinstance(r, dict)]
    kappa_rot = _to_float(calib.get("kappa_rot"))
    kappa_sigma = _to_float(calib.get("kappa_sigma"))
    lambda_rot = _to_float(calib.get("lambda_rot"))
    lambda_sigma = _to_float(calib.get("lambda_sigma"))
    legacy_fit = calib.get("legacy_fit") if isinstance(calib.get("legacy_fit"), dict) else {}
    lambda_rot_legacy = _to_float(legacy_fit.get("lambda_rot_fit"))
    lambda_sigma_legacy = _to_float(legacy_fit.get("lambda_sigma_fit"))

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15.0, 4.8), dpi=180)

    # 条件分岐: `lambda_rot is not None` を満たす経路を評価する。
    if lambda_rot is not None:
        err = 0.0 if lambda_sigma is None else float(lambda_sigma)
        ax0.errorbar([0.0], [lambda_rot], yerr=[err], fmt="o", color="#1f77b4", capsize=5, label="prior")

    # 条件分岐: `lambda_rot_legacy is not None` を満たす経路を評価する。

    if lambda_rot_legacy is not None:
        err_legacy = 0.0 if lambda_sigma_legacy is None else float(lambda_sigma_legacy)
        ax0.errorbar([0.25], [lambda_rot_legacy], yerr=[err_legacy], fmt="s", color="#ff7f0e", capsize=5, label="legacy fit")

    ax0.axhline(0.0, linestyle="--", color="#6b7280", linewidth=1.0)
    ax0.set_xlim(-0.4, 0.65)
    ax0.set_xticks([0.0, 0.25], ["prior", "legacy"])
    ax0.set_ylabel("coupling value")
    ax0.set_title("L_rot coupling freeze")
    # 条件分岐: `kappa_rot is not None` を満たす経路を評価する。
    if kappa_rot is not None:
        ax0.text(0.02, 0.92, f"kappa_rot={kappa_rot:.6f}", transform=ax0.transAxes, fontsize=9)

    # 条件分岐: `lambda_rot is not None` を満たす経路を評価する。

    if lambda_rot is not None:
        ax0.text(0.02, 0.84, f"lambda_rot={lambda_rot:.6f}", transform=ax0.transAxes, fontsize=9)

    ax0.grid(axis="y", alpha=0.25, linestyle=":")

    x = np.arange(len(labels), dtype=float)
    width = 0.36
    ax1.bar(x - width / 2.0, z_static, width=width, color="#d62728", alpha=0.88, label="static (deltaP_rot=0)")
    ax1.bar(x + width / 2.0, z_lrot, width=width, color="#2ca02c", alpha=0.88, label="L_rot prior-pred")
    ax1.axhline(3.0, linestyle="--", color="#6b7280", linewidth=1.0)
    ax1.axhline(-3.0, linestyle="--", color="#6b7280", linewidth=1.0)
    ax1.axhline(0.0, linestyle="-", color="#9ca3af", linewidth=0.9)
    ax1.set_xticks(x, labels, rotation=10, ha="right")
    ax1.set_ylabel("z = (obs - pred) / sigma")
    ax1.set_title("Frame-dragging gate: static vs holdout prediction")
    ax1.legend(loc="best", fontsize=8.8)
    ax1.grid(axis="y", alpha=0.25, linestyle=":")

    check_labels = [str(c.get("id") or "") for c in checks if isinstance(c, dict)]
    check_scores = [float(c.get("score") or 0.0) for c in checks if isinstance(c, dict)]
    check_colors: List[str] = []
    for c in checks:
        # 条件分岐: `not isinstance(c, dict)` を満たす経路を評価する。
        if not isinstance(c, dict):
            continue

        st = str(c.get("status") or "")
        # 条件分岐: `st == "pass"` を満たす経路を評価する。
        if st == "pass":
            check_colors.append("#2f9e44")
        # 条件分岐: 前段条件が不成立で、`st == "watch"` を追加評価する。
        elif st == "watch":
            check_colors.append("#eab308")
        # 条件分岐: 前段条件が不成立で、`st == "reject"` を追加評価する。
        elif st == "reject":
            check_colors.append("#dc2626")
        else:
            check_colors.append("#94a3b8")

    y = np.arange(len(check_labels), dtype=float)
    ax2.barh(y, check_scores, color=check_colors)
    ax2.axvline(1.0, linestyle="--", color="#6b7280", linewidth=1.0)
    ax2.set_yticks(y, check_labels)
    ax2.set_xlim(0.0, 1.05)
    ax2.set_xlabel("gate score")
    ax2.set_title("Operational checks")
    ax2.grid(axis="x", alpha=0.25, linestyle=":")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Step 8.7.21.6/8.7.26.2: prior-fixed predictive rotational closure audit (L_rot)."
    )
    parser.add_argument(
        "--rotating-json",
        type=str,
        default=str(ROOT / "output" / "public" / "theory" / "pmodel_rotating_sphere_p_distribution_audit.json"),
        help="Input rotating-sphere audit JSON (kappa_rot and branch rows).",
    )
    parser.add_argument(
        "--closure-json",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "lagrangian_noether_observable_closure_audit.json"),
        help="Input L_total closure audit JSON.",
    )
    parser.add_argument(
        "--drift-json",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "lagrangian_noether_observable_closure_drift_audit.json"),
        help="Input closure drift audit JSON.",
    )
    parser.add_argument(
        "--prior-json",
        type=str,
        default=str(ROOT / "data" / "theory" / "rotational_coupling_prior.json"),
        help="Input lambda_rot prior JSON (independent weak-field spin prior).",
    )
    parser.add_argument(
        "--out-json",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "lagrangian_noether_rotational_closure_audit.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "lagrangian_noether_rotational_closure_audit.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--out-png",
        type=str,
        default=str(ROOT / "output" / "public" / "quantum" / "lagrangian_noether_rotational_closure_audit.png"),
        help="Output PNG path.",
    )
    args = parser.parse_args(argv)

    rotating_json = Path(args.rotating_json).resolve()
    closure_json = Path(args.closure_json).resolve()
    drift_json = Path(args.drift_json).resolve()
    prior_json = Path(args.prior_json).resolve()
    out_json = Path(args.out_json).resolve()
    out_csv = Path(args.out_csv).resolve()
    out_png = Path(args.out_png).resolve()

    for name, path in [
        ("rotating-json", rotating_json),
        ("closure-json", closure_json),
        ("drift-json", drift_json),
    ]:
        # 条件分岐: `not path.exists()` を満たす経路を評価する。
        if not path.exists():
            print(f"[error] missing input ({name}): {_rel(path)}")
            return 2

    payload = build_payload(
        rotating_json=rotating_json,
        closure_json=closure_json,
        drift_json=drift_json,
        prior_json=prior_json,
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(out_csv, payload.get("channel_rows") if isinstance(payload.get("channel_rows"), list) else [])
    _plot(out_png, payload)

    decision = payload.get("decision") if isinstance(payload.get("decision"), dict) else {}
    print(f"[ok] wrote: {_rel(out_json)}")
    print(f"[ok] wrote: {_rel(out_csv)}")
    print(f"[ok] wrote: {_rel(out_png)}")
    print(
        "[summary] overall_status={0}, decision={1}".format(
            decision.get("overall_status"),
            decision.get("decision"),
        )
    )

    try:
        worklog.append_event(
            {
                "event_type": "quantum_lagrangian_noether_rotational_closure_audit",
                "phase": "8",
                "step": "8.7.21.6+8.7.26.2",
                "inputs": {
                    "rotating_json": _rel(rotating_json),
                    "closure_json": _rel(closure_json),
                    "drift_json": _rel(drift_json),
                    "prior_json": _rel(prior_json),
                },
                "outputs": {
                    "rotational_closure_audit_json": _rel(out_json),
                    "rotational_closure_audit_csv": _rel(out_csv),
                    "rotational_closure_audit_png": _rel(out_png),
                },
                "metrics": {
                    "overall_status": decision.get("overall_status"),
                    "decision": decision.get("decision"),
                    "hard_fail_ids_n": len(decision.get("hard_fail_ids") or []),
                    "watch_ids_n": len(decision.get("watch_ids") or []),
                    "lambda_rot": ((payload.get("calibration") or {}).get("lambda_rot")),
                    "holdout_reject_n": ((payload.get("prediction_gate") or {}).get("holdout_reject_n")),
                },
            }
        )
    except Exception as exc:
        print(f"[warn] worklog append skipped: {exc}")

    return 0 if str(decision.get("overall_status") or "") in {"pass", "watch"} else 1


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
