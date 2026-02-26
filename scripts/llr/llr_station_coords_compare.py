#!/usr/bin/env python3
"""
llr_station_coords_compare.py

LLR（P-model）バッチ結果について、観測局座標ソースの違い（pos+eop vs site log 等）が
残差RMSに与える影響を可視化する。

入力:
  - scripts/llr/llr_batch_eval.py の出力ディレクトリ（例: output/private/llr/batch）

出力:
  - output/private/llr/coord_compare/llr_station_coords_rms_scatter.png
  - output/private/llr/coord_compare/llr_station_coords_rms_by_station.png
  - output/private/llr/coord_compare/llr_grsm_monthly_rms_pos_eop_vs_slrlog.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


LLR_SHORT_NAME = "月レーザー測距（LLR: Lunar Laser Ranging）"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _set_japanese_font() -> None:
    try:
        import matplotlib as mpl
        import matplotlib.font_manager as fm

        preferred = [
            "Yu Gothic",
            "Meiryo",
            "BIZ UDGothic",
            "MS Gothic",
            "Yu Mincho",
            "MS Mincho",
        ]
        available = {f.name for f in fm.fontManager.ttflist}
        chosen = [name for name in preferred if name in available]
        # 条件分岐: `not chosen` を満たす経路を評価する。
        if not chosen:
            return

        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        return


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _require(path: Path) -> Path:
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        raise FileNotFoundError(str(path))

    return path


def _format_num(x: Any, *, digits: int = 3) -> str:
    try:
        # 条件分岐: `isinstance(x, int)` を満たす経路を評価する。
        if isinstance(x, int):
            return str(x)

        # 条件分岐: `isinstance(x, float)` を満たす経路を評価する。

        if isinstance(x, float):
            return f"{x:.{digits}g}"

        return str(x)
    except Exception:
        return str(x)


def main() -> int:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="Compare LLR batch results between station coordinate sources (pos+eop vs slrlog).")
    ap.add_argument(
        "--pos-eop-dir",
        type=str,
        default=str(Path("output") / "private" / "llr" / "batch"),
        help="LLR batch output dir for pos+eop/auto run (default: output/private/llr/batch)",
    )
    ap.add_argument(
        "--slrlog-dir",
        type=str,
        default=str(Path("output") / "private" / "llr" / "batch_slrlog"),
        help="LLR batch output dir for slrlog run (default: output/private/llr/batch_slrlog)",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(Path("output") / "private" / "llr" / "coord_compare"),
        help="Output directory for comparison figures (default: output/private/llr/coord_compare)",
    )
    ap.add_argument(
        "--metric-col",
        type=str,
        default="rms_sr_tropo_tide_ns",
        help="Metric column in llr_batch_metrics.csv to compare (default: rms_sr_tropo_tide_ns)",
    )
    ap.add_argument("--monthly-min-n", type=int, default=30, help="Monthly plot: minimum points within month (default: 30)")
    args = ap.parse_args()

    pos_dir = Path(str(args.pos_eop_dir))
    slr_dir = Path(str(args.slrlog_dir))
    out_dir = Path(str(args.out_dir))
    # 条件分岐: `not pos_dir.is_absolute()` を満たす経路を評価する。
    if not pos_dir.is_absolute():
        pos_dir = root / pos_dir

    # 条件分岐: `not slr_dir.is_absolute()` を満たす経路を評価する。

    if not slr_dir.is_absolute():
        slr_dir = root / slr_dir

    # 条件分岐: `not out_dir.is_absolute()` を満たす経路を評価する。

    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    metric_col = str(args.metric_col).strip()
    # 条件分岐: `not metric_col` を満たす経路を評価する。
    if not metric_col:
        print("[err] empty --metric-col")
        return 2

    _set_japanese_font()

    # ------------------------------------------------------------
    # 1) Scatter: group RMS (slrlog vs pos+eop)
    # ------------------------------------------------------------
    pos_metrics = _read_csv(_require(pos_dir / "llr_batch_metrics.csv"))
    slr_metrics = _read_csv(_require(slr_dir / "llr_batch_metrics.csv"))
    # 条件分岐: `metric_col not in pos_metrics.columns or metric_col not in slr_metrics.columns` を満たす経路を評価する。
    if metric_col not in pos_metrics.columns or metric_col not in slr_metrics.columns:
        print(f"[err] metric column not found in both CSVs: {metric_col}")
        print(f"  pos cols: {pos_metrics.columns.tolist()}")
        print(f"  slr cols: {slr_metrics.columns.tolist()}")
        return 2

    key_cols = ["station", "target"]
    pos = pos_metrics[key_cols + [metric_col]].rename(columns={metric_col: "rms_pos_eop_ns"})
    slr = slr_metrics[key_cols + [metric_col]].rename(columns={metric_col: "rms_slrlog_ns"})
    merged = pos.merge(slr, on=key_cols, how="inner")
    merged = merged.dropna(subset=["rms_pos_eop_ns", "rms_slrlog_ns"]).copy()
    merged["rms_pos_eop_ns"] = merged["rms_pos_eop_ns"].astype(float)
    merged["rms_slrlog_ns"] = merged["rms_slrlog_ns"].astype(float)
    merged = merged[np.isfinite(merged["rms_pos_eop_ns"]) & np.isfinite(merged["rms_slrlog_ns"])].copy()

    # 条件分岐: `merged.empty` を満たす経路を評価する。
    if merged.empty:
        print("[err] no common station×target rows to compare (merged empty)")
        return 2

    # Robust plot range

    p99 = float(np.nanpercentile(np.concatenate([merged["rms_pos_eop_ns"].to_numpy(), merged["rms_slrlog_ns"].to_numpy()]), 99))
    lim = max(5.0, p99 * 1.15)

    plt.figure(figsize=(7.2, 6.2))
    for st in sorted(merged["station"].astype(str).unique().tolist()):
        sub = merged[merged["station"].astype(str) == st]
        plt.scatter(sub["rms_slrlog_ns"], sub["rms_pos_eop_ns"], s=35, alpha=0.85, label=str(st))

    plt.plot([0, lim], [0, lim], color="#444444", lw=1.0, ls="--", alpha=0.8, label="y=x")
    plt.xlim(0, lim)
    plt.ylim(0, lim)
    plt.xlabel("RMS [ns]（slrlog）")
    plt.ylabel("RMS [ns]（pos+eop/auto）")
    plt.title(f"{LLR_SHORT_NAME}：局座標ソース比較（{metric_col}）")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    p_scatter = out_dir / "llr_station_coords_rms_scatter.png"
    plt.savefig(p_scatter, dpi=180)
    plt.close()

    # ------------------------------------------------------------
    # 2) Bar: median RMS per station
    # ------------------------------------------------------------
    by_st = (
        merged.groupby("station")[["rms_slrlog_ns", "rms_pos_eop_ns"]]
        .median(numeric_only=True)
        .reset_index()
        .sort_values("station")
    )
    xs = by_st["station"].astype(str).tolist()
    x = np.arange(len(xs), dtype=float)
    width = 0.38
    plt.figure(figsize=(10.5, 4.5))
    plt.bar(x - width / 2, by_st["rms_slrlog_ns"].to_numpy(dtype=float), width=width, label="slrlog")
    plt.bar(x + width / 2, by_st["rms_pos_eop_ns"].to_numpy(dtype=float), width=width, label="pos+eop/auto")
    plt.xticks(x, xs)
    plt.ylabel("残差RMS [ns]（station×target の中央値）")
    plt.title(f"{LLR_SHORT_NAME}：局座標ソース比較（局別）")
    plt.grid(True, axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    p_bar = out_dir / "llr_station_coords_rms_by_station.png"
    plt.savefig(p_bar, dpi=180)
    plt.close()

    # ------------------------------------------------------------
    # 2b) Combined figure (public-report friendly)
    # ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.1))
    ax0, ax1 = axes[0], axes[1]

    for st in sorted(merged["station"].astype(str).unique().tolist()):
        sub = merged[merged["station"].astype(str) == st]
        ax0.scatter(sub["rms_slrlog_ns"], sub["rms_pos_eop_ns"], s=30, alpha=0.85, label=str(st))

    ax0.plot([0, lim], [0, lim], color="#444444", lw=1.0, ls="--", alpha=0.8)
    ax0.set_xlim(0, lim)
    ax0.set_ylim(0, lim)
    ax0.set_xlabel("RMS [ns]（slrlog）")
    ax0.set_ylabel("RMS [ns]（pos+eop/auto）")
    ax0.set_title("RMS散布図（station×target）")
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="upper left", fontsize=9)

    ax1.bar(x - width / 2, by_st["rms_slrlog_ns"].to_numpy(dtype=float), width=width, label="slrlog")
    ax1.bar(x + width / 2, by_st["rms_pos_eop_ns"].to_numpy(dtype=float), width=width, label="pos+eop/auto")
    ax1.set_xticks(x)
    ax1.set_xticklabels(xs)
    ax1.set_ylabel("残差RMS [ns]（station×target の中央値）")
    ax1.set_title("局別の中央値RMS")
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend()

    fig.suptitle(f"{LLR_SHORT_NAME}：局座標ソース比較（{metric_col}）", y=0.995)
    fig.tight_layout()
    p_combo = out_dir / "llr_station_coords_rms_compare.png"
    fig.savefig(p_combo, dpi=180)
    plt.close(fig)

    # ------------------------------------------------------------
    # 3) GRSM: monthly RMS comparison (SR+Tropo+Tide)
    # ------------------------------------------------------------
    monthly_name = "llr_monthly_station_stats_models.csv"
    pos_monthly = _read_csv(_require(pos_dir / monthly_name))
    slr_monthly = _read_csv(_require(slr_dir / monthly_name))
    # Expected columns: model, station, year_month, n, rms_ns
    needed = {"model", "station", "year_month", "n", "rms_ns"}
    # 条件分岐: `not needed.issubset(set(pos_monthly.columns)) or not needed.issubset(set(slr_...` を満たす経路を評価する。
    if not needed.issubset(set(pos_monthly.columns)) or not needed.issubset(set(slr_monthly.columns)):
        # Optional: skip if unavailable
        print(f"[warn] monthly models CSV missing expected columns; skip monthly plot: {monthly_name}")
        return 0

    monthly_min_n = int(args.monthly_min_n)
    model_key = "SR+Tropo+Tide"

    def _prep_monthly(df: pd.DataFrame, *, label: str) -> pd.DataFrame:
        out = df.copy()
        out["station"] = out["station"].astype(str).str.upper()
        out["model"] = out["model"].astype(str)
        out = out[(out["station"] == "GRSM") & (out["model"] == model_key)].copy()
        out["n"] = pd.to_numeric(out["n"], errors="coerce")
        out["rms_ns"] = pd.to_numeric(out["rms_ns"], errors="coerce")
        out = out[(out["n"] >= monthly_min_n) & np.isfinite(out["rms_ns"])].copy()
        out["t"] = pd.to_datetime(out["year_month"].astype(str) + "-01", utc=True, errors="coerce")
        out = out.dropna(subset=["t"]).sort_values("t")
        out["label"] = label
        return out

    ppos = _prep_monthly(pos_monthly, label="pos+eop/auto")
    pslr = _prep_monthly(slr_monthly, label="slrlog")

    # 条件分岐: `not ppos.empty and not pslr.empty` を満たす経路を評価する。
    if not ppos.empty and not pslr.empty:
        plt.figure(figsize=(12, 4.8))
        plt.plot(np.array(pslr["t"].dt.to_pydatetime()), pslr["rms_ns"].to_numpy(dtype=float), marker="o", label="slrlog")
        plt.plot(np.array(ppos["t"].dt.to_pydatetime()), ppos["rms_ns"].to_numpy(dtype=float), marker="o", label="pos+eop/auto")
        plt.ylabel(f"残差RMS [ns]（月ごと, n>={monthly_min_n}, {model_key}）")
        plt.title(f"{LLR_SHORT_NAME}：GRSM 月別RMS（局座標ソース比較）")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        p_month = out_dir / "llr_grsm_monthly_rms_pos_eop_vs_slrlog.png"
        plt.savefig(p_month, dpi=180)
        plt.close()

    # Emit a tiny JSON for report wiring / provenance

    meta: Dict[str, Any] = {
        "pos_eop_dir": str(pos_dir.relative_to(root)).replace("\\", "/") if str(pos_dir).startswith(str(root)) else str(pos_dir),
        "slrlog_dir": str(slr_dir.relative_to(root)).replace("\\", "/") if str(slr_dir).startswith(str(root)) else str(slr_dir),
        "metric_col": metric_col,
        "monthly_model": model_key,
        "monthly_min_n": monthly_min_n,
        "n_groups_compared": int(len(merged)),
    }
    (out_dir / "llr_station_coords_compare_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"[ok] wrote: {p_scatter}")
    print(f"[ok] wrote: {p_bar}")
    print(f"[ok] wrote: {p_combo}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
