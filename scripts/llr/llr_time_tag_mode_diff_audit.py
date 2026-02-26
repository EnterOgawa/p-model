#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
llr_time_tag_mode_diff_audit.py

Compare full-batch LLR outputs between auto time-tag mode and fixed tx mode.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(ROOT) not in sys.path` を満たす経路を評価する。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from scripts.summary import worklog  # type: ignore
except Exception:
    worklog = None


# 関数: `_utc_now` の入出力契約と処理意図を定義する。

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_rel` の入出力契約と処理意図を定義する。

def _rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except Exception:
        return path.as_posix()


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_load_metrics` の入出力契約と処理意図を定義する。

def _load_metrics(path: Path) -> Tuple[pd.DataFrame, List[str]]:
    df = pd.read_csv(path)
    for col in ("station", "target"):
        # 条件分岐: `col not in df.columns` を満たす経路を評価する。
        if col not in df.columns:
            raise ValueError(f"missing column in metrics: {col}")

    key_cols = ["station", "target"]
    exclude = set(key_cols + ["time_tag_mode", "beta", "has_station", "has_reflector"])
    metric_cols: List[str] = []
    for col in df.columns:
        # 条件分岐: `col in exclude` を満たす経路を評価する。
        if col in exclude:
            continue

        values = pd.to_numeric(df[col], errors="coerce")
        # 条件分岐: `values.notna().any()` を満たす経路を評価する。
        if values.notna().any():
            df[col] = values
            metric_cols.append(col)

    # 条件分岐: `not metric_cols` を満たす経路を評価する。

    if not metric_cols:
        raise ValueError("no comparable numeric metric columns found in metrics csv")

    keep = key_cols + metric_cols
    return df[keep].copy(), metric_cols


# 関数: `_mode_counts` の入出力契約と処理意図を定義する。

def _mode_counts(by_station: Any) -> Dict[str, int]:
    out: Dict[str, int] = {}
    # 条件分岐: `not isinstance(by_station, dict)` を満たす経路を評価する。
    if not isinstance(by_station, dict):
        return out

    for value in by_station.values():
        key = str(value)
        out[key] = int(out.get(key, 0) + 1)

    return out


# 関数: `_median_map` の入出力契約と処理意図を定義する。

def _median_map(summary: Dict[str, Any]) -> Dict[str, float]:
    src = summary.get("median_rms_ns")
    # 条件分岐: `not isinstance(src, dict)` を満たす経路を評価する。
    if not isinstance(src, dict):
        return {}

    out: Dict[str, float] = {}
    for key, value in src.items():
        try:
            out[str(key)] = float(value)
        except Exception:
            continue

    return out


# 関数: `_build_plot` の入出力契約と処理意図を定義する。

def _build_plot(df: pd.DataFrame, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    changed = df[df["abs_delta_rms_ns"] > 0].copy()
    # 条件分岐: `changed.empty` を満たす経路を評価する。
    if changed.empty:
        ax.axis("off")
        ax.text(
            0.5,
            0.56,
            "auto vs tx: no RMS differences\n(all station×target×model rows identical)",
            ha="center",
            va="center",
            fontsize=14,
        )
        ax.text(0.5, 0.40, "Δrms_ns = 0 for all matched rows", ha="center", va="center", fontsize=12)
    else:
        changed = changed.sort_values("abs_delta_rms_ns", ascending=False).head(25)
        labels = [
            f"{s}/{t}/{m}"
            for s, t, m in zip(
                changed["station"].astype(str),
                changed["target"].astype(str),
                changed["metric"].astype(str),
            )
        ]
        ax.barh(labels, changed["delta_rms_ns"].to_numpy(dtype=float), color="#2f6f95")
        ax.axvline(0.0, color="k", ls="--", lw=1.0)
        ax.set_xlabel("Δrms_ns (auto - tx)")
        ax.set_title("LLR full-batch time-tag mode difference audit")
        ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# 関数: `run` の入出力契約と処理意図を定義する。

def run(args: argparse.Namespace) -> int:
    auto_dir = (ROOT / args.auto_dir).resolve() if not Path(args.auto_dir).is_absolute() else Path(args.auto_dir)
    tx_dir = (ROOT / args.tx_dir).resolve() if not Path(args.tx_dir).is_absolute() else Path(args.tx_dir)
    out_json = (ROOT / args.out_json).resolve() if not Path(args.out_json).is_absolute() else Path(args.out_json)
    out_csv = (ROOT / args.out_csv).resolve() if not Path(args.out_csv).is_absolute() else Path(args.out_csv)
    out_png = (ROOT / args.out_png).resolve() if not Path(args.out_png).is_absolute() else Path(args.out_png)

    auto_summary = _read_json(auto_dir / "llr_batch_summary.json")
    tx_summary = _read_json(tx_dir / "llr_batch_summary.json")
    auto_metrics, auto_metric_cols = _load_metrics(auto_dir / "llr_batch_metrics.csv")
    tx_metrics, tx_metric_cols = _load_metrics(tx_dir / "llr_batch_metrics.csv")
    shared_metric_cols = [col for col in auto_metric_cols if col in set(tx_metric_cols)]
    # 条件分岐: `not shared_metric_cols` を満たす経路を評価する。
    if not shared_metric_cols:
        raise ValueError("no shared numeric metric columns between auto and tx metrics csv")

    merged = auto_metrics.merge(
        tx_metrics,
        on=["station", "target"],
        how="outer",
        suffixes=("_auto", "_tx"),
        indicator=True,
    )
    matched_rows = merged[merged["_merge"].eq("both")].copy()
    long_rows: List[Dict[str, Any]] = []
    for _, row in matched_rows.iterrows():
        station = str(row.get("station"))
        target = str(row.get("target"))
        for metric in shared_metric_cols:
            a_col = f"{metric}_auto"
            t_col = f"{metric}_tx"
            a_val = pd.to_numeric(pd.Series([row.get(a_col)]), errors="coerce").iloc[0]
            t_val = pd.to_numeric(pd.Series([row.get(t_col)]), errors="coerce").iloc[0]
            # 条件分岐: `pd.isna(a_val) and pd.isna(t_val)` を満たす経路を評価する。
            if pd.isna(a_val) and pd.isna(t_val):
                continue

            delta = float(a_val - t_val)
            long_rows.append(
                {
                    "station": station,
                    "target": target,
                    "metric": metric,
                    "value_auto": float(a_val),
                    "value_tx": float(t_val),
                    "delta_rms_ns": delta,
                    "abs_delta_rms_ns": abs(delta),
                }
            )

    diff_long = pd.DataFrame(long_rows)
    max_abs = float(diff_long["abs_delta_rms_ns"].max()) if not diff_long.empty else float("nan")
    mean_abs = float(diff_long["abs_delta_rms_ns"].mean()) if not diff_long.empty else float("nan")
    changed_n = int((diff_long["abs_delta_rms_ns"] > float(args.equivalent_tol_ns)).sum()) if not diff_long.empty else 0

    auto_med = _median_map(auto_summary)
    tx_med = _median_map(tx_summary)
    median_delta: Dict[str, float] = {}
    for key in sorted(set(auto_med) | set(tx_med)):
        # 条件分岐: `key in auto_med and key in tx_med` を満たす経路を評価する。
        if key in auto_med and key in tx_med:
            median_delta[key] = float(auto_med[key] - tx_med[key])

    decision = "equivalent" if changed_n == 0 else "different"
    status = "pass" if changed_n == 0 else "watch"

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    diff_long.sort_values(["station", "target", "metric"]).to_csv(out_csv, index=False)
    _build_plot(diff_long, out_png)

    result: Dict[str, Any] = {
        "generated_utc": _utc_now(),
        "decision": decision,
        "status": status,
        "equivalent_tol_ns": float(args.equivalent_tol_ns),
        "auto": {
            "dir": _rel(auto_dir),
            "time_tag_mode": auto_summary.get("time_tag_mode"),
            "mode_counts_by_station": _mode_counts(auto_summary.get("time_tag_mode_by_station")),
            "n_groups": auto_summary.get("n_groups"),
            "n_points_total": auto_summary.get("n_points_total"),
            "point_weighted_rms_ns": auto_summary.get("point_weighted_rms_ns"),
            "median_rms_ns": auto_summary.get("median_rms_ns"),
        },
        "tx": {
            "dir": _rel(tx_dir),
            "time_tag_mode": tx_summary.get("time_tag_mode"),
            "n_groups": tx_summary.get("n_groups"),
            "n_points_total": tx_summary.get("n_points_total"),
            "point_weighted_rms_ns": tx_summary.get("point_weighted_rms_ns"),
            "median_rms_ns": tx_summary.get("median_rms_ns"),
        },
        "metrics_diff": {
            "matched_rows": int(len(matched_rows)),
            "auto_only_rows": int((merged["_merge"] == "left_only").sum()),
            "tx_only_rows": int((merged["_merge"] == "right_only").sum()),
            "shared_metric_columns": shared_metric_cols,
            "compared_cells": int(len(diff_long)),
            "max_abs_delta_rms_ns": max_abs,
            "mean_abs_delta_rms_ns": mean_abs,
            "changed_rows_over_tol": changed_n,
        },
        "median_rms_delta_ns": median_delta,
        "artifacts": {
            "diff_csv": _rel(out_csv),
            "audit_png": _rel(out_png),
        },
    }
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] diff json: {out_json}")
    print(f"[ok] diff csv : {out_csv}")
    print(f"[ok] diff png : {out_png}")
    print(
        f"[ok] decision={decision} matched={result['metrics_diff']['matched_rows']} "
        f"max_abs_delta_rms_ns={result['metrics_diff']['max_abs_delta_rms_ns']:.6g}"
    )

    # 条件分岐: `worklog is not None` を満たす経路を評価する。
    if worklog is not None:
        try:
            worklog.append_event(
                "llr_time_tag_mode_diff_audit",
                {
                    "status": status,
                    "decision": decision,
                    "matched_rows": result["metrics_diff"]["matched_rows"],
                    "max_abs_delta_rms_ns": result["metrics_diff"]["max_abs_delta_rms_ns"],
                    "auto_dir": _rel(auto_dir),
                    "tx_dir": _rel(tx_dir),
                    "out_json": _rel(out_json),
                },
            )
        except Exception:
            pass

    return 0


# 関数: `build_parser` の入出力契約と処理意図を定義する。

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare LLR full-batch results: auto mode vs tx mode.")
    p.add_argument("--auto-dir", default="output/private/llr/batch", help="Auto-mode batch output directory.")
    p.add_argument("--tx-dir", default="output/private/llr/batch_tx_compare", help="Tx-mode batch output directory.")
    p.add_argument(
        "--out-json",
        default="output/private/llr/llr_time_tag_mode_auto_vs_tx_audit.json",
        help="Output JSON path.",
    )
    p.add_argument(
        "--out-csv",
        default="output/private/llr/llr_time_tag_mode_auto_vs_tx_diff.csv",
        help="Output CSV path.",
    )
    p.add_argument(
        "--out-png",
        default="output/private/llr/llr_time_tag_mode_auto_vs_tx_audit.png",
        help="Output PNG path.",
    )
    p.add_argument(
        "--equivalent-tol-ns",
        type=float,
        default=1e-9,
        help="Rows with abs(delta_rms_ns) <= tol are treated as equivalent.",
    )
    return p


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
