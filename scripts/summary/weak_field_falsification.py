#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weak_field_falsification.py

Phase 6 / Step 6.2.4:
弱場統合（Cassini/Viking/Mercury/LLR/GPS）の「棄却条件（反証条件）」を明文化し、
現在の固定出力（Step 6.2.3）に対して pass/fail/unknown を機械可読に評価する。

出力（固定）:
  - output/private/summary/weak_field_falsification.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


# 関数: `_iso_utc_now` の入出力契約と処理意図を定義する。
def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_relpath` の入出力契約と処理意図を定義する。

def _relpath(p: Path) -> str:
    try:
        return str(p.relative_to(_ROOT)).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_try_load_frozen_parameters` の入出力契約と処理意図を定義する。

def _try_load_frozen_parameters() -> Dict[str, Any]:
    p = _ROOT / "output" / "private" / "theory" / "frozen_parameters.json"
    # 条件分岐: `not p.exists()` を満たす経路を評価する。
    if not p.exists():
        return {"path": _relpath(p), "exists": False}

    try:
        data = _read_json(p)
    except Exception:
        return {"path": _relpath(p), "exists": True, "parse_error": True}

    out: Dict[str, Any] = {"path": _relpath(p), "exists": True}
    for k in ("beta", "beta_sigma", "gamma_pmodel", "gamma_pmodel_sigma", "delta"):
        # 条件分岐: `k in data` を満たす経路を評価する。
        if k in data:
            out[k] = data.get(k)

    policy = data.get("policy")
    # 条件分岐: `isinstance(policy, dict)` を満たす経路を評価する。
    if isinstance(policy, dict):
        out["policy"] = {kk: policy.get(kk) for kk in ("fit_predict_separation", "beta_source", "delta_source", "note")}

    return out


# 関数: `_get_nested` の入出力契約と処理意図を定義する。

def _get_nested(d: Dict[str, Any], path: List[str]) -> Any:
    cur: Any = d
    for k in path:
        # 条件分岐: `not isinstance(cur, dict)` を満たす経路を評価する。
        if not isinstance(cur, dict):
            return None

        cur = cur.get(k)

    return cur


# 関数: `_as_float` の入出力契約と処理意図を定義する。

def _as_float(v: Any) -> Optional[float]:
    # 条件分岐: `isinstance(v, (int, float)) and math.isfinite(float(v))` を満たす経路を評価する。
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v)

    return None


# 関数: `_criterion` の入出力契約と処理意図を定義する。

def _criterion(
    *,
    cid: str,
    test_id: str,
    title: str,
    value: Any,
    op: str,
    threshold: Any,
    passed: Optional[bool],
    gate: bool,
    unit: str = "",
    rationale: str = "",
    source: Optional[Dict[str, Any]] = None,
    notes: str = "",
) -> Dict[str, Any]:
    return {
        "id": str(cid),
        "test_id": str(test_id),
        "title": str(title),
        "value": value,
        "op": str(op),
        "threshold": threshold,
        "pass": passed,
        "gate": bool(gate),
        "unit": str(unit),
        "rationale": str(rationale),
        "source": source or None,
        "notes": str(notes),
    }


# 関数: `_eval_ge` の入出力契約と処理意図を定義する。

def _eval_ge(v: Optional[float], thr: float) -> Optional[bool]:
    # 条件分岐: `v is None` を満たす経路を評価する。
    if v is None:
        return None

    return bool(v >= thr)


# 関数: `_eval_le` の入出力契約と処理意図を定義する。

def _eval_le(v: Optional[float], thr: float) -> Optional[bool]:
    # 条件分岐: `v is None` を満たす経路を評価する。
    if v is None:
        return None

    return bool(v <= thr)


# 関数: `_eval_lt` の入出力契約と処理意図を定義する。

def _eval_lt(v: Optional[float], thr: float) -> Optional[bool]:
    # 条件分岐: `v is None` を満たす経路を評価する。
    if v is None:
        return None

    return bool(v < thr)


# 関数: `_eval_abs_le` の入出力契約と処理意図を定義する。

def _eval_abs_le(v: Optional[float], thr: float) -> Optional[bool]:
    # 条件分岐: `v is None` を満たす経路を評価する。
    if v is None:
        return None

    return bool(abs(v) <= thr)


# 関数: `_try_read_csv_head` の入出力契約と処理意図を定義する。

def _try_read_csv_head(path: Path, *, n: int = 2) -> List[Dict[str, str]]:
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        return []

    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            rows.append({k: str(v) for k, v in (r or {}).items() if k})
            # 条件分岐: `i + 1 >= n` を満たす経路を評価する。
            if i + 1 >= n:
                break

    return rows


# 関数: `build_falsification` の入出力契約と処理意図を定義する。

def build_falsification(consistency: Dict[str, Any], frozen: Dict[str, Any]) -> Dict[str, Any]:
    beta_frozen = _as_float(frozen.get("beta"))
    delta_frozen = _as_float(frozen.get("delta"))

    criteria: List[Dict[str, Any]] = []

    # ----------------
    # Cassini (Shapiro Doppler y(t))
    # ----------------
    cass_focus = _get_nested(consistency, ["results", "cassini_sce1_doppler", "focus"])
    cass_windows = _get_nested(consistency, ["results", "cassini_sce1_doppler", "windows"])
    cass_corr = _as_float((cass_focus or {}).get("corr") if isinstance(cass_focus, dict) else None)
    cass_rmse = _as_float((cass_focus or {}).get("rmse") if isinstance(cass_focus, dict) else None)

    cass_nrmse = None
    # 条件分岐: `isinstance(cass_windows, dict)` を満たす経路を評価する。
    if isinstance(cass_windows, dict):
        w = cass_windows.get("-10 to +10 days") or cass_windows.get("all (available points)") or {}
        # 条件分岐: `isinstance(w, dict)` を満たす経路を評価する。
        if isinstance(w, dict):
            rmse = _as_float(w.get("rmse"))
            max_obs = _as_float(w.get("max_obs"))
            min_obs = _as_float(w.get("min_obs"))
            # 条件分岐: `rmse is not None and max_obs is not None and min_obs is not None and (max_obs...` を満たす経路を評価する。
            if rmse is not None and max_obs is not None and min_obs is not None and (max_obs - min_obs) != 0.0:
                cass_nrmse = rmse / (max_obs - min_obs)

    # Parameter freeze check (metadata)

    cass_meta_path = _ROOT / "output" / "private" / "cassini" / "cassini_fig2_run_metadata.json"
    cass_beta_used = None
    # 条件分岐: `cass_meta_path.exists()` を満たす経路を評価する。
    if cass_meta_path.exists():
        try:
            m = _read_json(cass_meta_path)
            cass_beta_used = _as_float(_get_nested(m, ["params", "beta"]))
        except Exception:
            cass_beta_used = None

    cass_beta_match = None
    # 条件分岐: `beta_frozen is not None and cass_beta_used is not None` を満たす経路を評価する。
    if beta_frozen is not None and cass_beta_used is not None:
        cass_beta_match = abs(cass_beta_used - beta_frozen) <= 1e-12

    criteria.append(
        _criterion(
            cid="cassini_beta_frozen",
            test_id="cassini_sce1_doppler",
            title="β を frozen_parameters に固定して実行していること",
            value=cass_beta_used,
            op="==",
            threshold=beta_frozen,
            passed=cass_beta_match,
            gate=True,
            unit="",
            rationale="弱場統合では、データごとの β 付け替えを禁止する（Phase 6.2 の前提）。",
            source={"run_metadata_json": _relpath(cass_meta_path), "frozen_parameters_json": frozen.get("path")},
        )
    )
    criteria.append(
        _criterion(
            cid="cassini_corr_min",
            test_id="cassini_sce1_doppler",
            title="相関（±10日）が十分に高いこと",
            value=cass_corr,
            op=">=",
            threshold=0.95,
            passed=_eval_ge(cass_corr, 0.95),
            gate=True,
            unit="",
            rationale="形状一致（中心付近の対称性・尖り）を最低限担保する。",
            source={"weak_field_longterm_consistency_json": consistency.get("outputs", {}).get("weak_field_longterm_consistency_json")},
        )
    )
    criteria.append(
        _criterion(
            cid="cassini_nrmse_max",
            test_id="cassini_sce1_doppler",
            title="正規化RMSE（±10日）が過大でないこと",
            value=cass_nrmse,
            op="<=",
            threshold=0.10,
            passed=_eval_le(cass_nrmse, 0.10),
            gate=True,
            unit="(RMSE)/(max_obs-min_obs)",
            rationale="スケール依存を避けるため、観測振幅で正規化した誤差を用いる（10%は保守的な初版閾値）。",
            source={"cassini_fig2_metrics_csv": "output/private/cassini/cassini_fig2_metrics.csv"},
        )
    )

    # ----------------
    # GPS
    # ----------------
    gps_metrics = _get_nested(consistency, ["results", "gps_satellite_clock", "metrics"])
    gps_rel_corr = _as_float((gps_metrics or {}).get("rel_corr") if isinstance(gps_metrics, dict) else None)
    gps_brdc_med = _as_float((gps_metrics or {}).get("brdc_rms_ns_median") if isinstance(gps_metrics, dict) else None)
    gps_pmod_med = _as_float((gps_metrics or {}).get("pmodel_rms_ns_median") if isinstance(gps_metrics, dict) else None)

    gps_ratio = None
    # 条件分岐: `gps_brdc_med is not None and gps_pmod_med is not None and gps_brdc_med != 0.0` を満たす経路を評価する。
    if gps_brdc_med is not None and gps_pmod_med is not None and gps_brdc_med != 0.0:
        gps_ratio = gps_pmod_med / gps_brdc_med

    criteria.append(
        _criterion(
            cid="gps_rel_corr_min",
            test_id="gps_satellite_clock",
            title="dt_rel と P-model 周期成分の相関が高いこと",
            value=gps_rel_corr,
            op=">=",
            threshold=0.99,
            passed=_eval_ge(gps_rel_corr, 0.99),
            gate=True,
            unit="",
            rationale="少なくとも既知の近日点効果（周期成分）に対して整合することを要求する。",
            source={"gps_compare_metrics_json": "output/private/gps/gps_compare_metrics.json"},
        )
    )
    criteria.append(
        _criterion(
            cid="gps_rms_ratio_max",
            test_id="gps_satellite_clock",
            title="P-model の中央値RMSが BRDC に対して極端に悪化しないこと",
            value=gps_ratio,
            op="<=",
            threshold=2.0,
            passed=_eval_le(gps_ratio, 2.0),
            gate=False,
            unit="(pmodel_rms_median)/(brdc_rms_median)",
            rationale="現状は実装差/系統（衛星依存等）が残るため、初版では“破綻していない”条件として緩く監視する。",
            source={"gps_compare_metrics_json": "output/private/gps/gps_compare_metrics.json"},
        )
    )

    # ----------------
    # LLR
    # ----------------
    llr_summary_path = _ROOT / "output" / "private" / "llr" / "batch" / "llr_batch_summary.json"
    llr_beta_used = None
    llr_tide = None
    llr_nosh = None
    # 条件分岐: `llr_summary_path.exists()` を満たす経路を評価する。
    if llr_summary_path.exists():
        try:
            j = _read_json(llr_summary_path)
            llr_beta_used = _as_float(j.get("beta"))
            med = j.get("median_rms_ns") if isinstance(j.get("median_rms_ns"), dict) else {}
            llr_tide = _as_float(med.get("station_reflector_tropo_tide") if isinstance(med, dict) else None)
            llr_nosh = _as_float(med.get("station_reflector_tropo_no_shapiro") if isinstance(med, dict) else None)
        except Exception:
            pass

    llr_beta_match = None
    # 条件分岐: `beta_frozen is not None and llr_beta_used is not None` を満たす経路を評価する。
    if beta_frozen is not None and llr_beta_used is not None:
        llr_beta_match = abs(llr_beta_used - beta_frozen) <= 1e-12

    llr_ratio = None
    # 条件分岐: `llr_tide is not None and llr_nosh is not None and llr_nosh != 0.0` を満たす経路を評価する。
    if llr_tide is not None and llr_nosh is not None and llr_nosh != 0.0:
        llr_ratio = llr_tide / llr_nosh

    criteria.append(
        _criterion(
            cid="llr_beta_frozen",
            test_id="llr_batch",
            title="β を frozen_parameters に固定して実行していること",
            value=llr_beta_used,
            op="==",
            threshold=beta_frozen,
            passed=llr_beta_match,
            gate=True,
            rationale="弱場統合では β を固定し、LLR でも同じ β で矛盾しないことを確認する。",
            source={"llr_batch_summary_json": _relpath(llr_summary_path), "frozen_parameters_json": frozen.get("path")},
        )
    )
    criteria.append(
        _criterion(
            cid="llr_shapiro_improves",
            test_id="llr_batch",
            title="Shapiro項を除去するとRMSが悪化すること（中央値）",
            value=llr_ratio,
            op="<=",
            threshold=0.95,
            passed=_eval_le(llr_ratio, 0.95),
            gate=True,
            unit="(tropo+tide)/(tropo+no_shapiro)",
            rationale="Shapiro項の符号/係数が逆だと、RMS改善が消えるため、最小の反証条件として採用する。",
            source={"llr_batch_summary_json": _relpath(llr_summary_path)},
        )
    )

    # ----------------
    # Mercury (perihelion)
    # ----------------
    mer_path = _ROOT / "output" / "private" / "mercury" / "mercury_precession_metrics.json"
    mer_diff_pct = None
    # 条件分岐: `mer_path.exists()` を満たす経路を評価する。
    if mer_path.exists():
        try:
            m = _read_json(mer_path)
            ref = _as_float(m.get("reference_arcsec_century"))
            sim = _as_float(_get_nested(m, ["simulation_physical", "pmodel", "arcsec_per_century"]))
            # 条件分岐: `ref is not None and sim is not None and ref != 0.0` を満たす経路を評価する。
            if ref is not None and sim is not None and ref != 0.0:
                mer_diff_pct = 100.0 * (sim - ref) / ref
        except Exception:
            pass

    criteria.append(
        _criterion(
            cid="mercury_ref_agree",
            test_id="mercury_perihelion_precession",
            title="近日点移動（簡易）がおおむね一致すること（|Δ|≤1%）",
            value=mer_diff_pct,
            op="abs<=",
            threshold=1.0,
            passed=_eval_abs_le(mer_diff_pct, 1.0),
            gate=False,
            unit="percent",
            rationale="本スクリプトは精密暦ではないため、初版では“オーダー整合”の監視に留める。",
            source={"mercury_precession_metrics_json": _relpath(mer_path)},
        )
    )

    # ----------------
    # Viking (Shapiro peak)
    # ----------------
    vik_path = _ROOT / "output" / "private" / "viking" / "viking_shapiro_result.csv"
    vik_max = None
    # 条件分岐: `vik_path.exists()` を満たす経路を評価する。
    if vik_path.exists():
        try:
            with open(vik_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    v = r.get("shapiro_delay_us")
                    # 条件分岐: `v is None or str(v).strip() == ""` を満たす経路を評価する。
                    if v is None or str(v).strip() == "":
                        continue

                    try:
                        us = float(v)
                    except Exception:
                        continue

                    # 条件分岐: `vik_max is None or us > vik_max` を満たす経路を評価する。

                    if vik_max is None or us > vik_max:
                        vik_max = us
        except Exception:
            pass

    criteria.append(
        _criterion(
            cid="viking_peak_range",
            test_id="viking_shapiro_peak",
            title="Shapiro遅延のピークがオーダー整合すること（100–400 μs）",
            value=vik_max,
            op="range",
            threshold={"min": 100.0, "max": 400.0},
            passed=(None if vik_max is None else bool(100.0 <= float(vik_max) <= 400.0)),
            gate=False,
            unit="μs",
            rationale="現状は観測時系列未導入のため、オーダー確認としてのみ扱う。",
            source={"viking_shapiro_result_csv": _relpath(vik_path)},
        )
    )

    # Summary
    gate_fail = [c for c in criteria if c.get("gate") and c.get("pass") is False]
    gate_unknown = [c for c in criteria if c.get("gate") and c.get("pass") is None]
    overall_gate_pass = (len(gate_fail) == 0) and (len(gate_unknown) == 0)

    return {
        "generated_utc": _iso_utc_now(),
        "phase": {"phase": 6, "step": "6.2.4", "name": "弱場統合の反証条件（棄却基準）"},
        "inputs": {
            "weak_field_longterm_consistency_json": consistency.get("outputs", {}).get("weak_field_longterm_consistency_json")
            or "output/private/summary/weak_field_longterm_consistency.json",
            "frozen_parameters": frozen,
            "per_test_metadata": {
                "cassini_fig2_run_metadata_json": _relpath(cass_meta_path),
                "llr_batch_summary_json": _relpath(llr_summary_path),
            },
        },
        "policy": {
            "beta_frozen": beta_frozen,
            "delta_frozen": delta_frozen,
            "note": "初版の閾値は“逃げ道封鎖”の最低条件（破綻検出）として保守的に設定。必要に応じて一次ソースに基づき更新する。",
        },
        "criteria": criteria,
        "summary": {
            "gate_pass": bool(overall_gate_pass),
            "gate_fail_n": len(gate_fail),
            "gate_unknown_n": len(gate_unknown),
            "gate_fail_ids": [str(c.get("id")) for c in gate_fail],
            "gate_unknown_ids": [str(c.get("id")) for c in gate_unknown],
        },
    }


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate weak-field falsification criteria evaluation (Phase 6 / Step 6.2.4).")
    ap.add_argument(
        "--consistency",
        type=str,
        default=str(_ROOT / "output" / "private" / "summary" / "weak_field_longterm_consistency.json"),
        help="Input consistency JSON (default: output/private/summary/weak_field_longterm_consistency.json).",
    )
    ap.add_argument(
        "--out",
        type=str,
        default=str(_ROOT / "output" / "private" / "summary" / "weak_field_falsification.json"),
        help="Output JSON path (default: output/private/summary/weak_field_falsification.json).",
    )
    args = ap.parse_args(argv)

    in_path = Path(args.consistency)
    # 条件分岐: `not in_path.is_absolute()` を満たす経路を評価する。
    if not in_path.is_absolute():
        in_path = (_ROOT / in_path).resolve()

    # 条件分岐: `not in_path.exists()` を満たす経路を評価する。

    if not in_path.exists():
        print(f"[err] missing input: {in_path}")
        return 2

    out_path = Path(args.out)
    # 条件分岐: `not out_path.is_absolute()` を満たす経路を評価する。
    if not out_path.is_absolute():
        out_path = (_ROOT / out_path).resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    consistency = _read_json(in_path)
    frozen = _try_load_frozen_parameters()
    payload = build_falsification(consistency, frozen)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[ok] wrote: {_relpath(out_path)}")
    worklog.append_event(
        {
            "event_type": "summary_weak_field_falsification",
            "phase": "6.2.4",
            "inputs": {"weak_field_longterm_consistency_json": _relpath(in_path), "frozen_parameters_json": frozen.get("path")},
            "outputs": {"weak_field_falsification_json": _relpath(out_path)},
        }
    )
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
