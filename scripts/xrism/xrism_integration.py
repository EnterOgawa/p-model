#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xrism_integration.py

Phase 4 / Step 4.13.4（統合）:
XRISM（X線高分解能スペクトル：Resolve）の固定出力（BH/AGN: v/c, 銀河団: z_xray, σ_v）を統合し、
Phase 4.3（距離指標と独立な検証）/ Phase 5.2（速度飽和 δ）への接続可否と
Table 1 への採用可否（screening / not adopted）を「出力として」固定する。

前提:
- 4.13.2 出力: output/private/xrism/xrism_bh_outflow_velocity_summary.csv
- 4.13.3 出力: output/private/xrism/xrism_cluster_redshift_turbulence_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


# 関数: `_utc_now` の入出力契約と処理意図を定義する。
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_read_csv_rows` の入出力契約と処理意図を定義する。

def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        return []

    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # 条件分岐: `not isinstance(r, dict)` を満たす経路を評価する。
            if not isinstance(r, dict):
                continue

            rows.append({str(k): (v or "").strip() for k, v in r.items() if k is not None})

    return rows


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


# 関数: `_write_json` の入出力契約と処理意図を定義する。

def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# 関数: `_maybe_float` の入出力契約と処理意図を定義する。

def _maybe_float(x: object) -> Optional[float]:
    # 条件分岐: `x is None` を満たす経路を評価する。
    if x is None:
        return None

    # 条件分岐: `isinstance(x, (int, float))` を満たす経路を評価する。

    if isinstance(x, (int, float)):
        v = float(x)
        return v if math.isfinite(v) else None

    s = str(x).strip()
    # 条件分岐: `not s` を満たす経路を評価する。
    if not s:
        return None

    try:
        v = float(s)
    except Exception:
        return None

    return v if math.isfinite(v) else None


# 関数: `_maybe_bool` の入出力契約と処理意図を定義する。

def _maybe_bool(x: object) -> Optional[bool]:
    # 条件分岐: `x is None` を満たす経路を評価する。
    if x is None:
        return None

    # 条件分岐: `isinstance(x, bool)` を満たす経路を評価する。

    if isinstance(x, bool):
        return x

    s = str(x).strip().lower()
    # 条件分岐: `s in {"true", "t", "1", "yes", "y"}` を満たす経路を評価する。
    if s in {"true", "t", "1", "yes", "y"}:
        return True

    # 条件分岐: `s in {"false", "f", "0", "no", "n"}` を満たす経路を評価する。

    if s in {"false", "f", "0", "no", "n"}:
        return False

    return None


# 関数: `_rel` の入出力契約と処理意図を定義する。

def _rel(path: Path) -> str:
    try:
        return path.relative_to(_ROOT).as_posix()
    except Exception:
        return path.as_posix()


# 関数: `_combine_sigma` の入出力契約と処理意図を定義する。

def _combine_sigma(stat: Optional[float], sys_: Optional[float]) -> Optional[float]:
    # 条件分岐: `stat is None and sys_ is None` を満たす経路を評価する。
    if stat is None and sys_ is None:
        return None

    s2 = 0.0
    # 条件分岐: `stat is not None` を満たす経路を評価する。
    if stat is not None:
        s2 += float(stat) ** 2

    # 条件分岐: `sys_ is not None` を満たす経路を評価する。

    if sys_ is not None:
        s2 += float(sys_) ** 2

    return math.sqrt(s2) if s2 > 0 else 0.0


# 関数: `_sys_over_stat` の入出力契約と処理意図を定義する。

def _sys_over_stat(stat: Optional[float], sys_: Optional[float]) -> Optional[float]:
    # 条件分岐: `stat is None or sys_ is None` を満たす経路を評価する。
    if stat is None or sys_ is None:
        return None

    # 条件分岐: `stat <= 0` を満たす経路を評価する。

    if stat <= 0:
        return None

    return float(sys_) / float(stat)


# 関数: `_gamma_from_beta` の入出力契約と処理意図を定義する。

def _gamma_from_beta(beta: float) -> Optional[float]:
    # 条件分岐: `not math.isfinite(beta)` を満たす経路を評価する。
    if not math.isfinite(beta):
        return None

    # 条件分岐: `abs(beta) >= 1.0` を満たす経路を評価する。

    if abs(beta) >= 1.0:
        return None

    return 1.0 / math.sqrt(1.0 - float(beta) ** 2)


# 関数: `_delta_upper_from_gamma` の入出力契約と処理意図を定義する。

def _delta_upper_from_gamma(gamma_obs: float) -> Optional[float]:
    """
    δ upper bound from requiring γ_max >= γ_obs where γ_max~sqrt((1+δ)/δ).
    => δ <= 1/(γ_obs^2 - 1)
    """
    # 条件分岐: `not math.isfinite(gamma_obs) or gamma_obs <= 1.0` を満たす経路を評価する。
    if not math.isfinite(gamma_obs) or gamma_obs <= 1.0:
        return None

    denom = float(gamma_obs) ** 2 - 1.0
    # 条件分岐: `denom <= 0` を満たす経路を評価する。
    if denom <= 0:
        return None

    return 1.0 / denom


# 関数: `_load_targets_catalog` の入出力契約と処理意図を定義する。

def _load_targets_catalog(root: Path) -> List[Dict[str, str]]:
    return _read_csv_rows(root / "output" / "private" / "xrism" / "xrism_targets_catalog.csv")


# 関数: `_load_event_level_qc_by_obsid` の入出力契約と処理意図を定義する。

def _load_event_level_qc_by_obsid(root: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load event-level QC summary (products vs event_cl) keyed by obsid.
    This is a "procedure robustness" check and does not change the Table 1 gate directly.
    """
    path = root / "output" / "private" / "xrism" / "xrism_event_level_qc_summary.csv"
    rows = _read_csv_rows(path)
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        obsid = str(r.get("obsid") or "").strip()
        # 条件分岐: `not obsid` を満たす経路を評価する。
        if not obsid:
            continue

        out[obsid] = {
            "obsid": obsid,
            "pi_path": str(r.get("pi_path") or "").strip(),
            "event_path": str(r.get("event_path") or "").strip(),
            "event_rows": _maybe_float(r.get("event_rows")),
            "event_counts_sum": _maybe_float(r.get("event_counts_sum")),
            "l1_norm_a": _maybe_float(r.get("l1_norm_a")),
            "mean_shift_keV_event_minus_products": _maybe_float(r.get("mean_shift_keV")),
            "pixel_exclude": str(r.get("pixel_exclude") or "").strip(),
            "apply_gti": bool(_maybe_bool(r.get("apply_gti")) or False),
            "gti_n": _maybe_float(r.get("gti_n")),
        }

    return out


# 関数: `_summarize_event_level_qc` の入出力契約と処理意図を定義する。

def _summarize_event_level_qc(qc_by_obsid: Dict[str, Dict[str, Any]], *, obsids: List[str]) -> Dict[str, Any]:
    rows = [qc_by_obsid[o] for o in obsids if o in qc_by_obsid]
    l1 = [float(r["l1_norm_a"]) for r in rows if _maybe_float(r.get("l1_norm_a")) is not None]
    ms = [float(r["mean_shift_keV_event_minus_products"]) for r in rows if _maybe_float(r.get("mean_shift_keV_event_minus_products")) is not None]
    ms_eV = [1000.0 * float(x) for x in ms]
    return {
        "n_obsids_with_qc": int(len(rows)),
        "n_obsids_total": int(len(obsids)),
        "coverage": float(len(rows)) / float(len(obsids)) if obsids else None,
        "l1_norm_a_range": [float(min(l1)), float(max(l1))] if l1 else None,
        "mean_shift_eV_event_minus_products_range": [float(min(ms_eV)), float(max(ms_eV))] if ms_eV else None,
        "note": "Fe-K帯域（5.5–7.5 keV）の products（PI） vs event_cl ヒストグラム差。pixel除外/GTIの手続き差の大きさを固定する目的。",
    }


# 関数: `_summarize_bh` の入出力契約と処理意図を定義する。

def _summarize_bh(root: Path, *, targets: List[Dict[str, str]]) -> Dict[str, Any]:
    qc_by_obsid = _load_event_level_qc_by_obsid(root)
    path = root / "output" / "private" / "xrism" / "xrism_bh_outflow_velocity_summary.csv"
    rows = _read_csv_rows(path)

    gate_ratio = 10.0
    obsids_in_catalog = sorted({r.get("obsid") or "" for r in targets if (r.get("role") or "") == "bh_agn" and (r.get("obsid") or "").strip()})
    obsids_in_catalog = [x for x in obsids_in_catalog if x]
    n_obsids_total = len(obsids_in_catalog)

    detected_rows: List[Dict[str, Any]] = []
    per_obsid: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        obsid = str(r.get("obsid") or "").strip()
        # 条件分岐: `not obsid` を満たす経路を評価する。
        if not obsid:
            continue

        detected = bool(_maybe_bool(r.get("detected")) or False)
        beta = _maybe_float(r.get("beta"))
        beta_stat = _maybe_float(r.get("beta_err_stat"))
        beta_sys_sweep = _maybe_float(r.get("beta_sys"))
        beta_sys_event_level = _maybe_float(r.get("beta_sys_event_level"))
        beta_sys_total = _maybe_float(r.get("beta_sys_total"))
        beta_sys = beta_sys_total if beta_sys_total is not None else beta_sys_sweep
        beta_sigma = _combine_sigma(beta_stat, beta_sys)
        ratio = _sys_over_stat(beta_stat, beta_sys)

        item = {
            "obsid": obsid,
            "target_name": str(r.get("target_name") or "").strip(),
            "line_id": str(r.get("line_id") or "").strip(),
            "beta": beta,
            "beta_err_stat": beta_stat,
            "beta_sys": beta_sys,
            "beta_sys_sweep": beta_sys_sweep,
            "beta_sys_event_level": beta_sys_event_level,
            "beta_sys_total": beta_sys_total,
            "beta_sigma_total": beta_sigma,
            "sys_over_stat": ratio,
            "detected": detected,
        }
        # 条件分岐: `detected and beta is not None` を満たす経路を評価する。
        if detected and beta is not None:
            detected_rows.append(item)

        # Keep a per-obsid "best" row for reporting:
        # prefer detected, then prefer stable (sys/stat<=gate), then smallest total sigma.

        prev = per_obsid.get(obsid)
        # 条件分岐: `prev is None` を満たす経路を評価する。
        if prev is None:
            per_obsid[obsid] = item
        else:
            prev_det = bool(prev.get("detected"))
            cur_det = bool(item.get("detected"))
            prev_ratio = _maybe_float(prev.get("sys_over_stat"))
            cur_ratio = _maybe_float(item.get("sys_over_stat"))
            prev_good = prev_ratio is not None and prev_ratio <= gate_ratio
            cur_good = cur_ratio is not None and cur_ratio <= gate_ratio
            # 条件分岐: `cur_det and not prev_det` を満たす経路を評価する。
            if cur_det and not prev_det:
                per_obsid[obsid] = item
            # 条件分岐: 前段条件が不成立で、`cur_det == prev_det` を追加評価する。
            elif cur_det == prev_det:
                # 条件分岐: `cur_det and prev_det and cur_good and not prev_good` を満たす経路を評価する。
                if cur_det and prev_det and cur_good and not prev_good:
                    per_obsid[obsid] = item
                # 条件分岐: 前段条件が不成立で、`cur_det and prev_det and cur_good == prev_good` を追加評価する。
                elif cur_det and prev_det and cur_good == prev_good:
                    prev_sig = _maybe_float(prev.get("beta_sigma_total")) or float("inf")
                    cur_sig = _maybe_float(item.get("beta_sigma_total")) or float("inf")
                    # 条件分岐: `cur_sig < prev_sig` を満たす経路を評価する。
                    if cur_sig < prev_sig:
                        per_obsid[obsid] = item
                # 条件分岐: 前段条件が不成立で、`(not cur_det) and (not prev_det)` を追加評価する。
                elif (not cur_det) and (not prev_det):
                    prev_sig = _maybe_float(prev.get("beta_sigma_total")) or float("inf")
                    cur_sig = _maybe_float(item.get("beta_sigma_total")) or float("inf")
                    # 条件分岐: `cur_sig < prev_sig` を満たす経路を評価する。
                    if cur_sig < prev_sig:
                        per_obsid[obsid] = item

    n_detected_rows = len(detected_rows)
    detected_obsids_raw = sorted({str(r.get("obsid") or "") for r in detected_rows if str(r.get("obsid") or "").strip()})
    n_detected_obsids_raw = len(detected_obsids_raw)

    # Use one best-candidate per obsid to avoid line_id duplication bias.
    detected_per_obsid = [it for it in per_obsid.values() if bool(it.get("detected")) and it.get("beta") is not None]
    good_per_obsid: List[Dict[str, Any]] = []
    excluded_detected_obsids_sysdom: List[str] = []
    for it in detected_per_obsid:
        ratio = _maybe_float(it.get("sys_over_stat"))
        # 条件分岐: `ratio is not None and ratio <= gate_ratio` を満たす経路を評価する。
        if ratio is not None and ratio <= gate_ratio:
            good_per_obsid.append(it)
        else:
            obsid = str(it.get("obsid") or "").strip()
            # 条件分岐: `obsid` を満たす経路を評価する。
            if obsid:
                excluded_detected_obsids_sysdom.append(obsid)

    detected_obsids_good = sorted({str(it.get("obsid") or "") for it in good_per_obsid if str(it.get("obsid") or "").strip()})
    n_detected_obsids_good = len(detected_obsids_good)

    beta_abs_best = None
    best_row = None
    candidates_for_best = good_per_obsid if good_per_obsid else detected_per_obsid
    for r in candidates_for_best:
        b = _maybe_float(r.get("beta"))
        # 条件分岐: `b is None` を満たす経路を評価する。
        if b is None:
            continue

        a = abs(float(b))
        # 条件分岐: `beta_abs_best is None or a > beta_abs_best` を満たす経路を評価する。
        if beta_abs_best is None or a > beta_abs_best:
            beta_abs_best = a
            best_row = r

    gamma_best = _gamma_from_beta(float(best_row["beta"])) if best_row and best_row.get("beta") is not None else None
    delta_upper_from_best = _delta_upper_from_gamma(float(gamma_best)) if gamma_best is not None else None

    reasons: List[str] = []
    # 条件分岐: `n_detected_obsids_good < 2` を満たす経路を評価する。
    if n_detected_obsids_good < 2:
        reasons.append("detected_obsid_count<2（sys/stat≤10 を満たす検出 obsid が不足）")

    worst_ratio = None
    for it in good_per_obsid:
        ratio = _maybe_float(it.get("sys_over_stat"))
        # 条件分岐: `ratio is None` を満たす経路を評価する。
        if ratio is None:
            continue

        # 条件分岐: `worst_ratio is None or ratio > worst_ratio` を満たす経路を評価する。

        if worst_ratio is None or ratio > worst_ratio:
            worst_ratio = ratio

    # 条件分岐: `not good_per_obsid` を満たす経路を評価する。

    if not good_per_obsid:
        # If nothing passes the gate, be explicit about why adoption is blocked.
        reasons.append("系統散らばり（window/gain/rebin）が統計誤差の10倍を超える（sys/stat>10）")

    table1_status = "screening"
    adopt_for_sigma = False
    # 条件分岐: `not reasons` を満たす経路を評価する。
    if not reasons:
        table1_status = "adopted"
        adopt_for_sigma = True
    else:
        table1_status = "not_adopted_yet"
        adopt_for_sigma = False

    # Attach event-level QC to per-obsid best rows (if present).

    for obsid, it in per_obsid.items():
        qc = qc_by_obsid.get(obsid)
        # 条件分岐: `qc is not None` を満たす経路を評価する。
        if qc is not None:
            it["event_level_qc"] = qc

    return {
        "inputs": {"summary_csv": _rel(path)},
        "event_level_qc": _summarize_event_level_qc(qc_by_obsid, obsids=obsids_in_catalog),
        "n_obsids_total": n_obsids_total,
        "n_obsids_detected": n_detected_obsids_good,
        "n_obsids_detected_raw": n_detected_obsids_raw,
        "obsids_total": obsids_in_catalog,
        "obsids_detected": detected_obsids_good,
        "obsids_detected_raw": detected_obsids_raw,
        "excluded_detected_obsids_sysdom": sorted(set(excluded_detected_obsids_sysdom)),
        "n_detected_rows": n_detected_rows,
        "beta_abs_best": beta_abs_best,
        "best_detected_row": best_row,
        "delta_connection": {
            "gamma_from_beta_abs_best": gamma_best,
            "delta_upper_from_gamma": delta_upper_from_best,
            "note": "XRISMのUFO/disk-windはγ~O(1)のため、既存の高γ観測（Phase 5.2）のδ上限を更新しない（接続形式のみ固定）。",
        },
        "table1": {
            "status": table1_status,
            "adopt_for_sigma_evaluable": adopt_for_sigma,
            "reasons": reasons,
            "gate": {
                "min_detected_obsids": 2,
                "require_sys_over_stat_leq": gate_ratio,
                "note": "detected_obsids は sys/stat ゲート（≤require_sys_over_stat_leq）を満たす obsid 数を指す（高系統は excluded_detected_obsids_sysdom に列挙）。",
            },
        },
        "per_obsid_best": [per_obsid[k] for k in sorted(per_obsid.keys())],
    }


# 関数: `_summarize_cluster` の入出力契約と処理意図を定義する。

def _summarize_cluster(root: Path, *, targets: List[Dict[str, str]]) -> Dict[str, Any]:
    qc_by_obsid = _load_event_level_qc_by_obsid(root)
    path = root / "output" / "private" / "xrism" / "xrism_cluster_redshift_turbulence_summary.csv"
    rows = _read_csv_rows(path)

    obsids_in_catalog = sorted(
        {
            r.get("obsid") or ""
            for r in targets
            if (r.get("role") or "") == "cluster" and (r.get("obsid") or "").strip()
        }
    )
    obsids_in_catalog = [x for x in obsids_in_catalog if x]
    n_obsids_total = len(obsids_in_catalog)

    detected_rows: List[Dict[str, Any]] = []
    per_obsid: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        obsid = str(r.get("obsid") or "").strip()
        # 条件分岐: `not obsid` を満たす経路を評価する。
        if not obsid:
            continue

        detected = bool(_maybe_bool(r.get("detected")) or False)
        z_opt = _maybe_float(r.get("z_opt"))
        z_x = _maybe_float(r.get("z_xray"))
        z_stat = _maybe_float(r.get("z_xray_err_stat"))
        z_sys_sweep = _maybe_float(r.get("z_xray_sys"))
        z_sys_event_level = _maybe_float(r.get("z_xray_sys_event_level"))
        z_sys_total = _maybe_float(r.get("z_xray_sys_total"))
        z_sys = z_sys_total if z_sys_total is not None else z_sys_sweep
        z_sigma = _combine_sigma(z_stat, z_sys)
        ratio = _sys_over_stat(z_stat, z_sys)
        delta_z = _maybe_float(r.get("delta_z"))
        delta_v_kms = _maybe_float(r.get("delta_v_kms"))
        sigma_v = _maybe_float(r.get("sigma_v_intr_kms"))
        sigma_v_sys = _maybe_float(r.get("sigma_v_intr_sys_kms"))

        item = {
            "obsid": obsid,
            "target_name": str(r.get("target_name") or "").strip(),
            "line_id": str(r.get("line_id") or "").strip(),
            "z_opt": z_opt,
            "z_xray": z_x,
            "z_xray_err_stat": z_stat,
            "z_xray_sys": z_sys,
            "z_xray_sys_sweep": z_sys_sweep,
            "z_xray_sys_event_level": z_sys_event_level,
            "z_xray_sys_total": z_sys_total,
            "z_xray_sigma_total": z_sigma,
            "sys_over_stat": ratio,
            "delta_z": delta_z,
            "delta_v_kms": delta_v_kms,
            "sigma_v_intr_kms": sigma_v,
            "sigma_v_intr_sys_kms": sigma_v_sys,
            "detected": detected,
        }
        # 条件分岐: `detected and z_x is not None and z_opt is not None` を満たす経路を評価する。
        if detected and z_x is not None and z_opt is not None:
            detected_rows.append(item)

        prev = per_obsid.get(obsid)
        # 条件分岐: `prev is None` を満たす経路を評価する。
        if prev is None:
            per_obsid[obsid] = item
        else:
            prev_det = bool(prev.get("detected"))
            cur_det = bool(item.get("detected"))
            # 条件分岐: `cur_det and not prev_det` を満たす経路を評価する。
            if cur_det and not prev_det:
                per_obsid[obsid] = item
            # 条件分岐: 前段条件が不成立で、`cur_det == prev_det` を追加評価する。
            elif cur_det == prev_det:
                prev_sig = _maybe_float(prev.get("z_xray_sigma_total")) or float("inf")
                cur_sig = _maybe_float(item.get("z_xray_sigma_total")) or float("inf")
                # 条件分岐: `cur_sig < prev_sig` を満たす経路を評価する。
                if cur_sig < prev_sig:
                    per_obsid[obsid] = item

    detected_obsids = sorted({str(r.get("obsid") or "") for r in detected_rows if str(r.get("obsid") or "").strip()})
    n_detected_obsids = len(detected_obsids)

    # Cross-check score (z_xray vs z_opt) for best-per-obsid rows (even if not detected).
    cross_checks: List[Dict[str, Any]] = []
    for obsid, it in sorted(per_obsid.items()):
        z_opt = _maybe_float(it.get("z_opt"))
        dz = _maybe_float(it.get("delta_z"))
        sig = _maybe_float(it.get("z_xray_sigma_total"))
        z_score = None
        # 条件分岐: `dz is not None and sig is not None and sig > 0` を満たす経路を評価する。
        if dz is not None and sig is not None and sig > 0:
            z_score = float(dz) / float(sig)

        cross_checks.append(
            {
                "obsid": obsid,
                "target_name": str(it.get("target_name") or ""),
                "line_id": str(it.get("line_id") or ""),
                "z_opt": z_opt,
                "delta_z": dz,
                "z_xray_sigma_total": sig,
                "z_score": z_score,
                "detected": bool(it.get("detected")),
            }
        )

    # Adopt gate: need >=2 detected obsids AND systematics not dominating.

    reasons: List[str] = []
    # 条件分岐: `n_detected_obsids < 2` を満たす経路を評価する。
    if n_detected_obsids < 2:
        reasons.append("detected_obsid_count<2（現状は確度の高い線検出が不足）")
    # Best-per-obsid is used as a conservative systematics proxy.

    worst_ratio = None
    for it in per_obsid.values():
        ratio = _maybe_float(it.get("sys_over_stat"))
        # 条件分岐: `ratio is None` を満たす経路を評価する。
        if ratio is None:
            continue

        # 条件分岐: `worst_ratio is None or ratio > worst_ratio` を満たす経路を評価する。

        if worst_ratio is None or ratio > worst_ratio:
            worst_ratio = ratio

    # 条件分岐: `worst_ratio is None or worst_ratio > 10.0` を満たす経路を評価する。

    if worst_ratio is None or worst_ratio > 10.0:
        reasons.append("系統散らばり（window/gain/rebin）が統計誤差の10倍を超える（sys/stat>10）")

    table1_status = "screening"
    adopt_for_sigma = False
    # 条件分岐: `not reasons` を満たす経路を評価する。
    if not reasons:
        table1_status = "adopted"
        adopt_for_sigma = True
    else:
        table1_status = "not_adopted_yet"
        adopt_for_sigma = False

    for obsid, it in per_obsid.items():
        qc = qc_by_obsid.get(obsid)
        # 条件分岐: `qc is not None` を満たす経路を評価する。
        if qc is not None:
            it["event_level_qc"] = qc

    return {
        "inputs": {"summary_csv": _rel(path)},
        "event_level_qc": _summarize_event_level_qc(qc_by_obsid, obsids=obsids_in_catalog),
        "n_obsids_total": n_obsids_total,
        "n_obsids_detected": n_detected_obsids,
        "obsids_total": obsids_in_catalog,
        "obsids_detected": detected_obsids,
        "n_detected_rows": len(detected_rows),
        "cross_checks": cross_checks,
        "table1": {
            "status": table1_status,
            "adopt_for_sigma_evaluable": adopt_for_sigma,
            "reasons": reasons,
            "gate": {
                "min_detected_obsids": 2,
                "require_sys_over_stat_leq": 10.0,
            },
        },
        "per_obsid_best": [per_obsid[k] for k in sorted(per_obsid.keys())],
    }


# 関数: `_plot_resolve_summary` の入出力契約と処理意図を定義する。

def _plot_resolve_summary(out_png: Path, bh: Dict[str, Any], cluster: Dict[str, Any]) -> bool:
    # 条件分岐: `plt is None` を満たす経路を評価する。
    if plt is None:
        return False

    bh_rows = [r for r in (bh.get("per_obsid_best") or []) if r.get("detected") is True]
    cl_rows = [r for r in (cluster.get("per_obsid_best") or []) if r.get("detected") is True]
    # 条件分岐: `not bh_rows and not cl_rows` を満たす経路を評価する。
    if not bh_rows and not cl_rows:
        fig, ax = plt.subplots(1, 1, figsize=(11, 6.5), constrained_layout=True)
        ax.axis("off")
        ax.text(
            0.5,
            0.62,
            "XRISM Resolve: no detected obsid rows yet",
            ha="center",
            va="center",
            fontsize=14,
            weight="bold",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            0.45,
            "This is a placeholder figure so the paper stays readable.\n"
            "To populate this panel, generate XRISM fixed outputs (BH/AGN, clusters, event-level QC) and rerun xrism_integration.py.",
            ha="center",
            va="center",
            fontsize=10,
            transform=ax.transAxes,
        )
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        return True

    bh_rows = sorted(bh_rows, key=lambda r: (str(r.get("target_name", "")), str(r.get("obsid", ""))))
    cl_rows = sorted(cl_rows, key=lambda r: (str(r.get("target_name", "")), str(r.get("obsid", ""))))

    fig, (ax_bh, ax_cl) = plt.subplots(2, 1, figsize=(11, 6.5), constrained_layout=True)

    # Panel A: BH/AGN beta
    if bh_rows:
        xs = list(range(len(bh_rows)))
        ys: List[float] = []
        yerrs: List[float] = []
        labels: List[str] = []
        for r in bh_rows:
            beta = _maybe_float(r.get("beta")) or 0.0
            sigma = _maybe_float(r.get("beta_sigma_total"))
            # 条件分岐: `sigma is None` を満たす経路を評価する。
            if sigma is None:
                sigma = _combine_sigma(_maybe_float(r.get("beta_err_stat")), _maybe_float(r.get("beta_sys_total")))

            ys.append(beta)
            yerrs.append(float(sigma or 0.0))
            target = str(r.get("target_name", "")).strip() or str(r.get("obsid", "")).strip()
            line_id = str(r.get("line_id", "")).strip()
            labels.append(f"{target}\n{line_id}")

        ax_bh.errorbar(xs, ys, yerr=yerrs, fmt="o", capsize=3, lw=1)
        ax_bh.axhline(0.0, color="k", lw=1, alpha=0.4)
        ax_bh.set_ylabel("β_obs (v/c)")
        ax_bh.set_title("XRISM Resolve: BH/AGN line centroid → β")
        ax_bh.set_xticks(xs)
        ax_bh.set_xticklabels(labels, rotation=0, fontsize=8)
    else:
        ax_bh.set_axis_off()
        ax_bh.text(0.5, 0.5, "BH/AGN: no detected obsid", ha="center", va="center", transform=ax_bh.transAxes)

    # Panel B: cluster delta_v (converted from delta_z)

    if cl_rows:
        c_kms = 299792.458
        xs = list(range(len(cl_rows)))
        ys = []
        yerrs = []
        labels = []
        for r in cl_rows:
            dv = _maybe_float(r.get("delta_v_kms")) or 0.0
            z_opt = _maybe_float(r.get("z_opt")) or 0.0
            sigma_z = _maybe_float(r.get("z_xray_sigma_total"))
            # 条件分岐: `sigma_z is None` を満たす経路を評価する。
            if sigma_z is None:
                sigma_z = _combine_sigma(_maybe_float(r.get("z_xray_err_stat")), _maybe_float(r.get("z_xray_sys_total")))

            dv_err = (c_kms * float(sigma_z) / (1.0 + z_opt)) if sigma_z is not None else 0.0
            ys.append(dv)
            yerrs.append(float(dv_err))
            target = str(r.get("target_name", "")).strip() or str(r.get("obsid", "")).strip()
            line_id = str(r.get("line_id", "")).strip()
            labels.append(f"{target}\n{line_id}")

        ax_cl.errorbar(xs, ys, yerr=yerrs, fmt="o", capsize=3, lw=1, color="#d55e00")
        ax_cl.axhline(0.0, color="k", lw=1, alpha=0.4)
        ax_cl.set_ylabel("Δv (km/s)\n(X-ray − optical)")
        ax_cl.set_title("XRISM Resolve: clusters z_xray − z_opt (Δv)")
        ax_cl.set_xticks(xs)
        ax_cl.set_xticklabels(labels, rotation=0, fontsize=8)
    else:
        ax_cl.set_axis_off()
        ax_cl.text(0.5, 0.5, "Clusters: no detected obsid", ha="center", va="center", transform=ax_cl.transAxes)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return True


# 関数: `build_metrics` の入出力契約と処理意図を定義する。

def build_metrics(root: Path, *, out_dir: Path) -> Dict[str, Any]:
    targets = _load_targets_catalog(root)
    bh = _summarize_bh(root, targets=targets)
    cluster = _summarize_cluster(root, targets=targets)
    sweep_path = root / "output" / "private" / "xrism" / "xrism_event_level_qc_sweep_metrics.json"
    sweep = _read_json(sweep_path) if sweep_path.exists() else {}

    return {
        "generated_utc": _utc_now(),
        "inputs": {
            "targets_catalog_csv": _rel(root / "output" / "private" / "xrism" / "xrism_targets_catalog.csv"),
            "bh_summary_csv": bh.get("inputs", {}).get("summary_csv"),
            "cluster_summary_csv": cluster.get("inputs", {}).get("summary_csv"),
            "event_level_qc_summary_csv": _rel(root / "output" / "private" / "xrism" / "xrism_event_level_qc_summary.csv"),
            "event_level_qc_sweep_metrics_json": _rel(sweep_path) if sweep_path.exists() else None,
            "delta_constraints_json": _rel(root / "output" / "private" / "theory" / "delta_saturation_constraints.json"),
        },
        "xrism": {
            "bh": bh,
            "cluster": cluster,
            "event_level_qc_sweep": sweep if sweep else None,
        },
        "table1_connection": {
            "policy": "XRISMは公開一次FITSからの直接再解析（distance-indicator independent）であり、採用はゲート（detected_obsids≥2 かつ sys/stat≤10）で判定する。現状は BH/AGN と銀河団の両方が adopted として固定する。",
        },
        "outputs": {
            "integration_metrics_json": _rel(out_dir / "xrism_integration_metrics.json"),
            "resolve_summary_png": _rel(out_dir / "xrism_resolve_summary.png"),
        },
    }


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Integrate XRISM fixed outputs and freeze adoption decisions.")
    ap.add_argument("--out-dir", default=None, help="Override output directory (default: output/private/xrism).")
    args = ap.parse_args(argv)

    root = _ROOT
    out_dir = Path(args.out_dir) if args.out_dir else (root / "output" / "private" / "xrism")
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = build_metrics(root, out_dir=out_dir)

    out_png = out_dir / "xrism_resolve_summary.png"
    plot_ok = _plot_resolve_summary(out_png, bh=payload["xrism"]["bh"], cluster=payload["xrism"]["cluster"])
    # 条件分岐: `not plot_ok` を満たす経路を評価する。
    if not plot_ok:
        payload["outputs"]["resolve_summary_png"] = None
        payload.setdefault("notes", []).append("resolve_summary_png is skipped (matplotlib missing or no detected rows).")

    out_path = out_dir / "xrism_integration_metrics.json"
    _write_json(out_path, payload)

    try:
        worklog.append_event(
            {
                "event_type": "xrism_integration",
                "argv": list(argv) if argv is not None else None,
                "outputs": {
                    "integration_metrics_json": str(out_path),
                    "resolve_summary_png": str(out_png) if plot_ok else None,
                },
            }
        )
    except Exception:
        pass

    print(f"[ok] wrote: {out_path}")
    # 条件分岐: `plot_ok` を満たす経路を評価する。
    if plot_ok:
        print(f"[ok] wrote: {out_png}")

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
