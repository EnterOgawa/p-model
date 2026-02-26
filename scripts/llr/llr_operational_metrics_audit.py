#!/usr/bin/env python3
"""
llr_operational_metrics_audit.py

LLR の実運用に近い指標（点重み付きRMS / 正規化残差 / χ²様）を再計算する監査。
`station×target の中央値RMS` を採否指標として誤用しないための補助パックを生成する。
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_ROOT = Path(__file__).resolve().parents[2]


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve()).replace("\\", "/")


def _read_record11_meta(path: Path, line_numbers: Sequence[int]) -> Dict[int, Dict[str, float]]:
    need = set(int(v) for v in line_numbers if v is not None and np.isfinite(float(v)))
    # 条件分岐: `not need` を満たす経路を評価する。
    if not need:
        return {}

    out: Dict[int, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for lineno, raw in enumerate(f, start=1):
            # 条件分岐: `lineno not in need` を満たす経路を評価する。
            if lineno not in need:
                continue

            toks = raw.strip().split()
            # 条件分岐: `not toks or toks[0] != "11"` を満たす経路を評価する。
            if not toks or toks[0] != "11":
                continue

            item: Dict[str, float] = {
                "np_window_s": _to_float(toks[5]) if len(toks) > 5 else float("nan"),
                "np_n_raw_ranges": _to_float(toks[6]) if len(toks) > 6 else float("nan"),
                "np_bin_rms_ps": _to_float(toks[7]) if len(toks) > 7 else float("nan"),
            }
            out[int(lineno)] = item

    return out


def _augment_with_np_meta(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    out = df.copy()
    out["source_file"] = out["source_file"].astype(str)
    out["lineno"] = pd.to_numeric(out["lineno"], errors="coerce")

    np_window = np.full((len(out),), np.nan, dtype=float)
    np_n_raw = np.full((len(out),), np.nan, dtype=float)
    np_bin_rms = np.full((len(out),), np.nan, dtype=float)

    grouped = out[["source_file", "lineno"]].dropna().groupby("source_file", dropna=False)
    cache: Dict[str, Dict[int, Dict[str, float]]] = {}
    for src, g in grouped:
        p = (root / Path(str(src))).resolve()
        # 条件分岐: `not p.exists()` を満たす経路を評価する。
        if not p.exists():
            continue

        ln = pd.to_numeric(g["lineno"], errors="coerce").dropna().astype(int).tolist()
        cache[str(src)] = _read_record11_meta(p, ln)

    for idx, row in out.iterrows():
        src = str(row.get("source_file", ""))
        ln = row.get("lineno")
        # 条件分岐: `not (src and np.isfinite(_to_float(ln)))` を満たす経路を評価する。
        if not (src and np.isfinite(_to_float(ln))):
            continue

        rec = cache.get(src, {}).get(int(float(ln)))
        # 条件分岐: `not rec` を満たす経路を評価する。
        if not rec:
            continue

        np_window[idx] = _to_float(rec.get("np_window_s"))
        np_n_raw[idx] = _to_float(rec.get("np_n_raw_ranges"))
        np_bin_rms[idx] = _to_float(rec.get("np_bin_rms_ps"))

    out["np_window_s"] = np_window
    out["np_n_raw_ranges"] = np_n_raw
    out["np_bin_rms_ps"] = np_bin_rms
    return out


def _solve_floor_ps(residual_ps: np.ndarray, sigma_ps: np.ndarray) -> float:
    # 条件分岐: `len(residual_ps) == 0` を満たす経路を評価する。
    if len(residual_ps) == 0:
        return float("nan")

    r2 = residual_ps * residual_ps
    s2 = sigma_ps * sigma_ps
    lo = 0.0
    hi = 1e8
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        val = float(np.mean(r2 / np.maximum(s2 + mid * mid, 1e-24)))
        # 条件分岐: `val > 1.0` を満たす経路を評価する。
        if val > 1.0:
            lo = mid
        else:
            hi = mid

    return float(hi)


def _subset_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    # 条件分岐: `df.empty` を満たす経路を評価する。
    if df.empty:
        return {"n_points": 0}

    res_ns = pd.to_numeric(df["residual_sr_tropo_tide_ns"], errors="coerce").to_numpy(dtype=float)
    bin_rms_ps = pd.to_numeric(df["np_bin_rms_ps"], errors="coerce").to_numpy(dtype=float)
    n_raw = pd.to_numeric(df.get("np_n_raw_ranges"), errors="coerce").to_numpy(dtype=float)

    ok = np.isfinite(res_ns) & np.isfinite(bin_rms_ps) & (bin_rms_ps > 0)
    res_ns = res_ns[ok]
    bin_rms_ps = bin_rms_ps[ok]
    n_raw = n_raw[ok]

    # 条件分岐: `len(res_ns) == 0` を満たす経路を評価する。
    if len(res_ns) == 0:
        return {"n_points": 0}

    res_ps = res_ns * 1000.0
    w = 1.0 / np.maximum(bin_rms_ps * bin_rms_ps, 1e-24)
    z_np = np.abs(res_ps) / bin_rms_ps
    chi2_like = float(np.mean((res_ps / bin_rms_ps) ** 2))
    wrms_ns = float(np.sqrt(np.sum(w * res_ps * res_ps) / np.sum(w)) / 1000.0)
    rms_ns = float(np.sqrt(np.mean(res_ns * res_ns)))

    ok_n = np.isfinite(n_raw) & (n_raw > 0)
    z_np_mean = np.array([], dtype=float)
    # 条件分岐: `np.any(ok_n)` を満たす経路を評価する。
    if np.any(ok_n):
        sigma_mean_ps = bin_rms_ps[ok_n] / np.sqrt(n_raw[ok_n])
        z_np_mean = np.abs(res_ps[ok_n]) / np.maximum(sigma_mean_ps, 1e-12)

    floor_ps = _solve_floor_ps(res_ps, bin_rms_ps)

    return {
        "n_points": int(len(res_ns)),
        "point_rms_ns": rms_ns,
        "weighted_rms_ns_bin_rms": wrms_ns,
        "median_abs_z_np": float(np.median(z_np)),
        "p90_abs_z_np": float(np.percentile(z_np, 90)),
        "p95_abs_z_np": float(np.percentile(z_np, 95)),
        "chi2_like_np": chi2_like,
        "median_abs_z_np_mean": float(np.median(z_np_mean)) if len(z_np_mean) else float("nan"),
        "p95_abs_z_np_mean": float(np.percentile(z_np_mean, 95)) if len(z_np_mean) else float("nan"),
        "model_floor_ns_for_chi2eq1": floor_ps / 1000.0 if np.isfinite(floor_ps) else float("nan"),
        "median_np_bin_rms_ns": float(np.median(bin_rms_ps) / 1000.0),
    }


def _write_plot(path: Path, summary_rows: List[Dict[str, Any]]) -> None:
    # 条件分岐: `not summary_rows` を満たす経路を評価する。
    if not summary_rows:
        return

    names = [str(r["subset"]) for r in summary_rows]
    wrms = [float(r.get("weighted_rms_ns_bin_rms", float("nan"))) for r in summary_rows]
    floor = [float(r.get("model_floor_ns_for_chi2eq1", float("nan"))) for r in summary_rows]
    p95z = [float(r.get("p95_abs_z_np", float("nan"))) for r in summary_rows]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    x = np.arange(len(names), dtype=float)

    axes[0].bar(x, wrms, color="#4e79a7")
    axes[0].set_title("Weighted RMS (bin-RMS weight)")
    axes[0].set_ylabel("ns")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=20, ha="right")

    axes[1].bar(x, floor, color="#f28e2b")
    axes[1].set_title("Model floor for χ²≈1")
    axes[1].set_ylabel("ns")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=20, ha="right")

    axes[2].bar(x, p95z, color="#e15759")
    axes[2].set_title("p95 |z| (residual / NP bin RMS)")
    axes[2].set_ylabel("dimensionless")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=20, ha="right")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description="LLR operational metrics audit (weighted RMS / normalized residual).")
    ap.add_argument(
        "--points-csv",
        type=str,
        default=str(_ROOT / "output" / "private" / "llr" / "batch" / "llr_batch_points.csv"),
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(_ROOT / "output" / "private" / "llr"),
    )
    ap.add_argument("--modern-start-year", type=int, default=2023)
    ap.add_argument("--exclude-target", type=str, default="nglr1")
    args = ap.parse_args()

    points_path = Path(str(args.points_csv))
    out_dir = Path(str(args.out_dir))
    # 条件分岐: `not points_path.is_absolute()` を満たす経路を評価する。
    if not points_path.is_absolute():
        points_path = (_ROOT / points_path).resolve()

    # 条件分岐: `not out_dir.is_absolute()` を満たす経路を評価する。

    if not out_dir.is_absolute():
        out_dir = (_ROOT / out_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    # 条件分岐: `not points_path.exists()` を満たす経路を評価する。
    if not points_path.exists():
        print(f"[err] missing points csv: {points_path}")
        return 2

    df = pd.read_csv(points_path)
    # 条件分岐: `df.empty` を満たす経路を評価する。
    if df.empty:
        print(f"[err] empty points csv: {points_path}")
        return 2

    df = _augment_with_np_meta(df, _ROOT)
    df["epoch_dt"] = pd.to_datetime(df["epoch_utc"], errors="coerce", utc=True)
    df["target_norm"] = df["target"].astype(str).str.strip().str.lower()
    exclude_target = str(args.exclude_target).strip().lower()

    subsets: List[Tuple[str, pd.DataFrame]] = [
        ("all", df),
        (f"exclude_{exclude_target}", df[df["target_norm"] != exclude_target]),
        (f"modern_{int(args.modern_start_year)}_all", df[df["epoch_dt"].dt.year >= int(args.modern_start_year)]),
        (
            f"modern_{int(args.modern_start_year)}_exclude_{exclude_target}",
            df[(df["epoch_dt"].dt.year >= int(args.modern_start_year)) & (df["target_norm"] != exclude_target)],
        ),
        (
            f"modern_{int(args.modern_start_year)}_apol_exclude_{exclude_target}",
            df[
                (df["epoch_dt"].dt.year >= int(args.modern_start_year))
                & (df["station"].astype(str).str.upper() == "APOL")
                & (df["target_norm"] != exclude_target)
            ],
        ),
    ]

    summary_rows: List[Dict[str, Any]] = []
    for name, sdf in subsets:
        row = {"subset": name}
        row.update(_subset_metrics(sdf))
        summary_rows.append(row)

    by_station: List[Dict[str, Any]] = []
    for st, sdf in df.groupby(df["station"].astype(str).str.upper(), dropna=False):
        row = {"station": str(st)}
        row.update(_subset_metrics(sdf))
        by_station.append(row)

    by_target: List[Dict[str, Any]] = []
    for tgt, sdf in df.groupby("target_norm", dropna=False):
        row = {"target": str(tgt)}
        row.update(_subset_metrics(sdf))
        by_target.append(row)

    out_json = out_dir / "llr_operational_metrics_audit.json"
    out_csv = out_dir / "llr_operational_metrics_audit.csv"
    out_png = out_dir / "llr_operational_metrics_audit.png"
    out_station_csv = out_dir / "llr_operational_metrics_by_station.csv"
    out_target_csv = out_dir / "llr_operational_metrics_by_target.csv"

    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    pd.DataFrame(by_station).to_csv(out_station_csv, index=False)
    pd.DataFrame(by_target).to_csv(out_target_csv, index=False)
    _write_plot(out_png, summary_rows)

    report: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "points_csv": _safe_rel(points_path, _ROOT),
            "exclude_target": exclude_target,
            "modern_start_year": int(args.modern_start_year),
        },
        "summary": summary_rows,
        "by_station": by_station,
        "by_target": by_target,
        "interpretation": {
            "operational_primary_metrics": [
                "point_weighted_rms (NP bin RMS 重み)",
                "normalized residual z = |residual| / NP_bin_rms",
                "chi2_like over z",
            ],
            "not_primary_metric": "station×target の中央値RMS（代表図用には有効だが採否の主判定には不向き）",
        },
        "outputs": {
            "summary_csv": _safe_rel(out_csv, _ROOT),
            "by_station_csv": _safe_rel(out_station_csv, _ROOT),
            "by_target_csv": _safe_rel(out_target_csv, _ROOT),
            "plot_png": _safe_rel(out_png, _ROOT),
        },
    }
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"[ok] {out_json}")
    print(f"[ok] {out_csv}")
    print(f"[ok] {out_station_csv}")
    print(f"[ok] {out_target_csv}")
    print(f"[ok] {out_png}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
