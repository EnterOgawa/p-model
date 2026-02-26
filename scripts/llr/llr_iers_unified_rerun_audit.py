#!/usr/bin/env python3
"""
llr_iers_unified_rerun_audit.py

Step 6.2.5 用:
baseline（既定 batch）と IERS統一 rerun（station-coords=pos_eop）の差分を監査し、
「全局IERS統一の達成度」と主要運用指標の変化を固定出力する。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_ROOT = Path(__file__).resolve().parents[2]


# 関数: `_to_float` の入出力契約と処理意図を定義する。
def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


# 関数: `_safe_rel` の入出力契約と処理意図を定義する。

def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve()).replace("\\", "/")


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_op_metric` の入出力契約と処理意図を定義する。

def _op_metric(op_json: Dict[str, Any], subset: str, key: str) -> float:
    rows = op_json.get("summary")
    # 条件分岐: `not isinstance(rows, list)` を満たす経路を評価する。
    if not isinstance(rows, list):
        return float("nan")

    for row in rows:
        # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
        if not isinstance(row, dict):
            continue

        # 条件分岐: `str(row.get("subset", "")) == subset` を満たす経路を評価する。

        if str(row.get("subset", "")) == subset:
            return _to_float(row.get(key))

    return float("nan")


# 関数: `_precision_check_status` の入出力契約と処理意図を定義する。

def _precision_check_status(precision_json: Dict[str, Any], check_id: str) -> str:
    checks = precision_json.get("checks")
    # 条件分岐: `not isinstance(checks, list)` を満たす経路を評価する。
    if not isinstance(checks, list):
        return "unknown"

    for check in checks:
        # 条件分岐: `not isinstance(check, dict)` を満たす経路を評価する。
        if not isinstance(check, dict):
            continue

        # 条件分岐: `str(check.get("id", "")) == check_id` を満たす経路を評価する。

        if str(check.get("id", "")) == check_id:
            return str(check.get("status", "unknown"))

    return "unknown"


# 関数: `_station_meta_diag` の入出力契約と処理意図を定義する。

def _station_meta_diag(meta_json: Dict[str, Any]) -> Dict[str, Any]:
    stations = meta_json.get("stations")
    # 条件分岐: `not isinstance(stations, dict)` を満たす経路を評価する。
    if not isinstance(stations, dict):
        return {"n_stations": 0, "n_pos_eop": 0, "pos_eop_share": float("nan"), "missing_pos_eop": [], "by_station": []}

    rows: List[Dict[str, Any]] = []
    for station, rec in stations.items():
        src = ""
        # 条件分岐: `isinstance(rec, dict)` を満たす経路を評価する。
        if isinstance(rec, dict):
            src = str(rec.get("station_coord_source_used", "")).strip().lower()

        rows.append({"station": str(station), "source": src})

    rows.sort(key=lambda x: str(x["station"]))

    n_st = int(len(rows))
    n_pos = int(sum(1 for row in rows if row["source"] == "pos_eop"))
    share = float(n_pos / n_st) if n_st > 0 else float("nan")
    missing = [str(row["station"]) for row in rows if row["source"] != "pos_eop"]
    return {
        "n_stations": n_st,
        "n_pos_eop": n_pos,
        "pos_eop_share": share,
        "missing_pos_eop": missing,
        "by_station": rows,
    }


# 関数: `_plot` の入出力契約と処理意図を定義する。

def _plot(path: Path, metric_rows: List[Dict[str, Any]], station_rows: List[Dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))

    labels = [str(row["metric"]) for row in metric_rows]
    baseline = np.array([_to_float(row["baseline"]) for row in metric_rows], dtype=float)
    iers = np.array([_to_float(row["iers"]) for row in metric_rows], dtype=float)
    x = np.arange(len(labels), dtype=float)
    width = 0.38
    axes[0].bar(x - width / 2.0, baseline, width=width, label="baseline", color="#4e79a7")
    axes[0].bar(x + width / 2.0, iers, width=width, label="iers", color="#f28e2b")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha="right")
    axes[0].set_ylabel("ns")
    axes[0].set_title("Operational metrics: baseline vs IERS rerun")
    axes[0].legend(loc="best")

    st_labels = [str(row["station"]) for row in station_rows]
    st_vals = np.array([1.0 if str(row["source"]) == "pos_eop" else 0.0 for row in station_rows], dtype=float)
    sx = np.arange(len(st_labels), dtype=float)
    axes[1].bar(sx, st_vals, color="#59a14f")
    axes[1].set_xticks(sx)
    axes[1].set_xticklabels(st_labels, rotation=20, ha="right")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].set_ylabel("pos_eop used (0/1)")
    axes[1].set_title("IERS station-coordinate coverage")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    ap = argparse.ArgumentParser(description="Audit IERS-unified LLR rerun against baseline.")
    ap.add_argument("--baseline-summary", default=str(_ROOT / "output" / "private" / "llr" / "batch" / "llr_batch_summary.json"))
    ap.add_argument("--baseline-operational", default=str(_ROOT / "output" / "private" / "llr" / "llr_operational_metrics_audit.json"))
    ap.add_argument("--baseline-precision", default=str(_ROOT / "output" / "private" / "llr" / "llr_precision_reaudit.json"))
    ap.add_argument("--iers-summary", default=str(_ROOT / "output" / "private" / "llr" / "batch_iers_unified" / "llr_batch_summary.json"))
    ap.add_argument("--iers-operational", default=str(_ROOT / "output" / "private" / "llr" / "iers_unified" / "llr_operational_metrics_audit.json"))
    ap.add_argument("--iers-precision", default=str(_ROOT / "output" / "private" / "llr" / "iers_unified" / "llr_precision_reaudit.json"))
    ap.add_argument("--iers-station-meta", default=str(_ROOT / "output" / "private" / "llr" / "batch_iers_unified" / "llr_station_metadata_used.json"))
    ap.add_argument("--out-dir", default=str(_ROOT / "output" / "private" / "llr"))
    args = ap.parse_args()

    baseline_summary_path = Path(str(args.baseline_summary))
    baseline_operational_path = Path(str(args.baseline_operational))
    baseline_precision_path = Path(str(args.baseline_precision))
    iers_summary_path = Path(str(args.iers_summary))
    iers_operational_path = Path(str(args.iers_operational))
    iers_precision_path = Path(str(args.iers_precision))
    iers_station_meta_path = Path(str(args.iers_station_meta))
    out_dir = Path(str(args.out_dir))

    paths = [
        baseline_summary_path,
        baseline_operational_path,
        baseline_precision_path,
        iers_summary_path,
        iers_operational_path,
        iers_precision_path,
        iers_station_meta_path,
    ]
    resolved: List[Path] = []
    for path in paths:
        # 条件分岐: `not path.is_absolute()` を満たす経路を評価する。
        if not path.is_absolute():
            path = (_ROOT / path).resolve()

        resolved.append(path)

    (
        baseline_summary_path,
        baseline_operational_path,
        baseline_precision_path,
        iers_summary_path,
        iers_operational_path,
        iers_precision_path,
        iers_station_meta_path,
    ) = resolved

    # 条件分岐: `not out_dir.is_absolute()` を満たす経路を評価する。
    if not out_dir.is_absolute():
        out_dir = (_ROOT / out_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    for path in resolved:
        # 条件分岐: `not path.exists()` を満たす経路を評価する。
        if not path.exists():
            print(f"[err] missing required input: {path}")
            return 2

    baseline_summary = _read_json(baseline_summary_path)
    baseline_operational = _read_json(baseline_operational_path)
    baseline_precision = _read_json(baseline_precision_path)
    iers_summary = _read_json(iers_summary_path)
    iers_operational = _read_json(iers_operational_path)
    iers_precision = _read_json(iers_precision_path)
    iers_station_meta = _read_json(iers_station_meta_path)

    metric_rows: List[Dict[str, Any]] = []
    for metric, subset, key in [
        ("all_weighted_rms_ns", "all", "weighted_rms_ns_bin_rms"),
        ("modern_apol_ex_nglr1_weighted_rms_ns", "modern_2023_apol_exclude_nglr1", "weighted_rms_ns_bin_rms"),
        ("modern_apol_ex_nglr1_floor_ns", "modern_2023_apol_exclude_nglr1", "model_floor_ns_for_chi2eq1"),
    ]:
        base_v = _op_metric(baseline_operational, subset, key)
        iers_v = _op_metric(iers_operational, subset, key)
        metric_rows.append(
            {
                "metric": metric,
                "baseline": base_v,
                "iers": iers_v,
                "delta_iers_minus_baseline": iers_v - base_v if np.isfinite(base_v) and np.isfinite(iers_v) else float("nan"),
            }
        )

    station_diag = _station_meta_diag(iers_station_meta)
    iers_all_station_unified = bool(
        station_diag.get("n_stations", 0) > 0 and station_diag.get("n_pos_eop", 0) == station_diag.get("n_stations", 0)
    )
    iers_mode = str(iers_summary.get("station_coords_mode") or "unknown").strip().lower()

    precision_watch_checks: List[str] = []
    checks = iers_precision.get("checks")
    # 条件分岐: `isinstance(checks, list)` を満たす経路を評価する。
    if isinstance(checks, list):
        for check in checks:
            # 条件分岐: `not isinstance(check, dict)` を満たす経路を評価する。
            if not isinstance(check, dict):
                continue

            # 条件分岐: `str(check.get("status", "")).lower() == "watch"` を満たす経路を評価する。

            if str(check.get("status", "")).lower() == "watch":
                cid = str(check.get("id", "")).strip()
                # 条件分岐: `cid` を満たす経路を評価する。
                if cid:
                    precision_watch_checks.append(cid)

    modern_gate_status = _precision_check_status(iers_precision, "apol_modern_operational_gate")
    decision = "pass" if iers_all_station_unified and iers_mode == "pos_eop" and modern_gate_status == "pass" else "watch"
    reasons: List[str] = []
    # 条件分岐: `iers_mode != "pos_eop"` を満たす経路を評価する。
    if iers_mode != "pos_eop":
        reasons.append("IERS rerun が station-coords=pos_eop で実行されていない。")

    # 条件分岐: `not iers_all_station_unified` を満たす経路を評価する。

    if not iers_all_station_unified:
        missing = station_diag.get("missing_pos_eop") or []
        reasons.append(f"全局IERS統一が未達（pos+eop未対応局: {','.join(missing) if missing else 'N/A'}）。")

    # 条件分岐: `modern_gate_status != "pass"` を満たす経路を評価する。

    if modern_gate_status != "pass":
        reasons.append("modern APOL operational gate が pass ではない。")

    report: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "decision": decision,
        "inputs": {
            "baseline_summary": _safe_rel(baseline_summary_path, _ROOT),
            "baseline_operational": _safe_rel(baseline_operational_path, _ROOT),
            "baseline_precision": _safe_rel(baseline_precision_path, _ROOT),
            "iers_summary": _safe_rel(iers_summary_path, _ROOT),
            "iers_operational": _safe_rel(iers_operational_path, _ROOT),
            "iers_precision": _safe_rel(iers_precision_path, _ROOT),
            "iers_station_meta": _safe_rel(iers_station_meta_path, _ROOT),
        },
        "metrics_compare_ns": metric_rows,
        "iers_station_coord": {
            "station_coords_mode": iers_mode,
            **station_diag,
        },
        "iers_precision": {
            "overall_status": iers_precision.get("overall_status"),
            "decision": iers_precision.get("decision"),
            "watch_checks": precision_watch_checks,
        },
        "watch_reasons": reasons,
    }

    out_json = out_dir / "llr_iers_unified_rerun_audit.json"
    out_csv = out_dir / "llr_iers_unified_rerun_audit.csv"
    out_png = out_dir / "llr_iers_unified_rerun_audit.png"

    pd.DataFrame(metric_rows).to_csv(out_csv, index=False)
    station_rows = station_diag.get("by_station") if isinstance(station_diag.get("by_station"), list) else []
    _plot(out_png, metric_rows, station_rows)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[ok] {out_json}")
    print(f"[ok] {out_csv}")
    print(f"[ok] {out_png}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())

