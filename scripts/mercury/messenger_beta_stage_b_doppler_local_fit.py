#!/usr/bin/env python3
"""
messenger_beta_stage_b_doppler_local_fit.py

Roadmap Step 8.7.48.2 (Stage B Doppler-only local orbit fit) の最小I/F実装。

目的:
- MESSENGER の Doppler 時系列を日次 arc に分割し、arc-local な線形 nuisance
  （intercept + drift）を推定して残差監査を固定する。
- ODF の完全 parser 導入前に、Stage B の入出力契約（arc summary / residuals / metrics）
  を先行固定する。

注意:
- 本スクリプトは Stage B の I/F 固定を目的とする最小実装であり、
  力学モデル（Sun-Mercury-spacecraft の数値積分）を代替するものではない。
- 完全前向きモデルは後段（8.7.48.3/.4）で統合する。
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from scripts.summary.worklog import append_event


# クラス: `ArcFitResult` の責務と境界条件を定義する。
@dataclass
class ArcFitResult:
    arc_id: str
    n_points: int
    intercept_hz: float
    drift_hz_per_s: float
    rms_hz: float
    status: str


# 関数: `_safe_rel` の入出力契約と処理意図を定義する。

def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve()).replace("\\", "/")


# 関数: `_resolve_path` の入出力契約と処理意図を定義する。

def _resolve_path(path_str: str, root: Path) -> Path:
    p = Path(str(path_str))
    if p.is_absolute():
        return p

    return (root / p).resolve()


# 関数: `_find_default_doppler_csv` の入出力契約と処理意図を定義する。

def _find_default_doppler_csv(data_root: Path) -> Optional[Path]:
    candidates = [
        data_root / "derived" / "odf_doppler_observations.csv",
        data_root / "derived" / "messenger_doppler_observations.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    if not data_root.exists():
        return None

    for p in data_root.rglob("*.csv"):
        name = p.name.lower()
        if "doppler" in name:
            return p

    return None


# 関数: `_detect_epoch_column` の入出力契約と処理意図を定義する。

def _detect_epoch_column(columns: Sequence[str]) -> Optional[str]:
    lowers = {c.lower(): c for c in columns}
    for key in ("epoch_utc", "time_utc", "utc", "epoch", "time"):
        if key in lowers:
            return lowers[key]

    return None


# 関数: `_detect_doppler_column` の入出力契約と処理意図を定義する。

def _detect_doppler_column(columns: Sequence[str]) -> Optional[str]:
    lowers = {c.lower(): c for c in columns}
    preferred = (
        "doppler_hz",
        "doppler",
        "doppler_residual_hz",
        "freq_hz",
        "frequency_hz",
        "frequency",
    )
    for key in preferred:
        if key in lowers:
            return lowers[key]

    return None


# 関数: `_parse_epoch_series` の入出力契約と処理意図を定義する。

def _parse_epoch_series(series: pd.Series) -> pd.Series:
    parsed_default = pd.to_datetime(series, utc=True, errors="coerce")
    nonnull_default = int(parsed_default.notna().sum())
    total = int(len(parsed_default))
    if total <= 0:
        return parsed_default

    keep_default = nonnull_default >= int(0.95 * total)
    if keep_default:
        return parsed_default

    parsed_best = parsed_default
    best_nonnull = nonnull_default
    for fmt in ("ISO8601", "mixed"):
        try:
            parsed_try = pd.to_datetime(series, utc=True, errors="coerce", format=fmt)
        except Exception:
            continue

        nonnull_try = int(parsed_try.notna().sum())
        if nonnull_try > best_nonnull:
            parsed_best = parsed_try
            best_nonnull = nonnull_try

    return parsed_best


# 関数: `_load_doppler_dataframe` の入出力契約と処理意図を定義する。

def _load_doppler_dataframe(path: Path) -> Tuple[pd.DataFrame, Dict[str, str]]:
    df = pd.read_csv(path)
    epoch_col = _detect_epoch_column(df.columns.tolist())
    doppler_col = _detect_doppler_column(df.columns.tolist())
    if epoch_col is None or doppler_col is None:
        raise ValueError("required columns not found (epoch and doppler).")

    work = df[[epoch_col, doppler_col]].copy()
    work.columns = ["epoch_utc", "doppler_hz"]
    work["epoch_utc"] = _parse_epoch_series(work["epoch_utc"])
    work["doppler_hz"] = pd.to_numeric(work["doppler_hz"], errors="coerce")
    work = work.dropna(subset=["epoch_utc", "doppler_hz"]).sort_values("epoch_utc").reset_index(drop=True)
    if len(work) <= 0:
        raise ValueError("no valid rows after epoch/doppler parse.")

    source_cols = {"epoch_col": epoch_col, "doppler_col": doppler_col}
    return work, source_cols


# 関数: `_fit_single_arc` の入出力契約と処理意図を定義する。

def _fit_single_arc(arc_id: str, arc_df: pd.DataFrame, min_points: int) -> ArcFitResult:
    n_points = int(len(arc_df))
    if n_points < min_points:
        return ArcFitResult(
            arc_id=arc_id,
            n_points=n_points,
            intercept_hz=float("nan"),
            drift_hz_per_s=float("nan"),
            rms_hz=float("nan"),
            status="reject",
        )

    t0 = arc_df["epoch_utc"].iloc[0]
    dt_s = (arc_df["epoch_utc"] - t0).dt.total_seconds().to_numpy(dtype=float)
    y = arc_df["doppler_hz"].to_numpy(dtype=float)
    design = np.column_stack([np.ones_like(dt_s), dt_s])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    fit = design @ coef
    resid = y - fit
    rms = float(np.sqrt(np.mean(resid**2)))
    return ArcFitResult(
        arc_id=arc_id,
        n_points=n_points,
        intercept_hz=float(coef[0]),
        drift_hz_per_s=float(coef[1]),
        rms_hz=rms,
        status="pass",
    )


# 関数: `_fit_arcs` の入出力契約と処理意図を定義する。

def _fit_arcs(df: pd.DataFrame, min_points_per_arc: int) -> Tuple[List[ArcFitResult], pd.DataFrame]:
    work = df.copy()
    work["arc_id"] = work["epoch_utc"].dt.strftime("%Y-%m-%d")
    results: List[ArcFitResult] = []
    residual_rows: List[Dict[str, object]] = []
    for arc_id, arc_df in work.groupby("arc_id", sort=True):
        result = _fit_single_arc(arc_id, arc_df, min_points=min_points_per_arc)
        results.append(result)
        if result.status != "pass":
            continue

        t0 = arc_df["epoch_utc"].iloc[0]
        dt_s = (arc_df["epoch_utc"] - t0).dt.total_seconds().to_numpy(dtype=float)
        fit = result.intercept_hz + result.drift_hz_per_s * dt_s
        resid = arc_df["doppler_hz"].to_numpy(dtype=float) - fit
        for idx in range(len(arc_df)):
            residual_rows.append(
                {
                    "arc_id": arc_id,
                    "epoch_utc": arc_df["epoch_utc"].iloc[idx].isoformat(),
                    "doppler_hz": float(arc_df["doppler_hz"].iloc[idx]),
                    "fit_hz": float(fit[idx]),
                    "residual_hz": float(resid[idx]),
                }
            )

    residual_df = pd.DataFrame(residual_rows)
    return results, residual_df


# 関数: `_write_arc_summary_csv` の入出力契約と処理意図を定義する。

def _write_arc_summary_csv(path: Path, rows: Sequence[ArcFitResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["arc_id", "n_points", "intercept_hz", "drift_hz_per_s", "rms_hz", "status"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "arc_id": r.arc_id,
                    "n_points": r.n_points,
                    "intercept_hz": r.intercept_hz,
                    "drift_hz_per_s": r.drift_hz_per_s,
                    "rms_hz": r.rms_hz,
                    "status": r.status,
                }
            )


# 関数: `_make_plot` の入出力契約と処理意図を定義する。

def _make_plot(rows: Sequence[ArcFitResult], out_pdf: Path, out_png: Path) -> Optional[str]:
    if plt is None:
        return "matplotlib_unavailable"

    valid = [r for r in rows if r.status == "pass" and np.isfinite(r.rms_hz)]
    if len(valid) <= 0:
        return "no_valid_arcs"

    labels = [r.arc_id for r in valid]
    values = [r.rms_hz for r in valid]
    fig, ax = plt.subplots(figsize=(11.8, 6.5))
    x = np.arange(len(labels))
    ax.bar(x, values, color="#1f77b4", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8.6)
    ax.set_ylabel("Doppler residual RMS [Hz]")
    ax.set_title("Roadmap 8.7.48.2: Stage B arc-local Doppler RMS")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return None


# 関数: `_sync_to_public` の入出力契約と処理意図を定義する。

def _sync_to_public(paths: Iterable[Path], private_root: Path, public_root: Path) -> List[Path]:
    public_root.mkdir(parents=True, exist_ok=True)
    synced: List[Path] = []
    for src in paths:
        try:
            rel = src.resolve().relative_to(private_root.resolve())
        except Exception:
            rel = Path(src.name)

        dst = public_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        synced.append(dst)

    return synced


# 関数: `_overall_status` の入出力契約と処理意図を定義する。

def _overall_status(valid_arcs: int, median_rms_hz: Optional[float], min_valid_arcs: int) -> str:
    if valid_arcs <= 0:
        return "reject"

    if valid_arcs < min_valid_arcs:
        return "watch"

    if median_rms_hz is None:
        return "watch"

    if median_rms_hz <= 1.0:
        return "pass"

    return "watch"


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    ap = argparse.ArgumentParser(description="Roadmap 8.7.48.2: Stage B Doppler-only local arc fit.")
    ap.add_argument(
        "--data-root",
        type=str,
        default=str(_ROOT / "data" / "mercury" / "messenger"),
        help="MESSENGER data root.",
    )
    ap.add_argument(
        "--doppler-csv",
        type=str,
        default="",
        help="Optional explicit Doppler CSV path; auto-detected when omitted.",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(_ROOT / "output" / "private" / "mercury"),
        help="Private output directory.",
    )
    ap.add_argument(
        "--public-dir",
        type=str,
        default=str(_ROOT / "output" / "public" / "mercury"),
        help="Public sync directory.",
    )
    ap.add_argument(
        "--min-points-per-arc",
        type=int,
        default=20,
        help="Minimum Doppler points per day-arc for a valid local fit.",
    )
    ap.add_argument(
        "--min-valid-arcs",
        type=int,
        default=3,
        help="Minimum number of valid arcs for Stage B pass candidate.",
    )
    args = ap.parse_args()

    data_root = _resolve_path(args.data_root, _ROOT)
    out_dir = _resolve_path(args.out_dir, _ROOT)
    public_dir = _resolve_path(args.public_dir, _ROOT)

    if str(args.doppler_csv).strip():
        doppler_csv = _resolve_path(str(args.doppler_csv), _ROOT)
    else:
        doppler_csv = _find_default_doppler_csv(data_root)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_arc_csv = out_dir / "messenger_beta_stage_b_arc_summary.csv"
    out_resid_csv = out_dir / "messenger_beta_stage_b_arc_residuals.csv"
    out_json = out_dir / "messenger_beta_stage_b_metrics.json"
    out_pdf = out_dir / "messenger_beta_stage_b_arc_rms.pdf"
    out_png = out_dir / "messenger_beta_stage_b_arc_rms.png"

    payload: Dict[str, object] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase_step": "8.7.48.2",
        "data_root": _safe_rel(data_root, _ROOT),
        "doppler_csv": _safe_rel(doppler_csv, _ROOT) if isinstance(doppler_csv, Path) else "",
    }

    if doppler_csv is None or (not doppler_csv.exists()):
        payload.update(
            {
                "overall_status": "reject",
                "reason": "doppler_csv_missing",
                "valid_arcs": 0,
                "median_rms_hz": None,
                "notes": "Provide derived Doppler CSV after ODF ingestion.",
            }
        )
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        produced = [out_json]
        synced = _sync_to_public(produced, private_root=out_dir, public_root=public_dir)
        append_event(
            {
                "event": "run_script",
                "script": "scripts/mercury/messenger_beta_stage_b_doppler_local_fit.py",
                "phase_step": "8.7.48.2",
                "status": "reject",
                "input": _safe_rel(data_root, _ROOT),
                "outputs": [_safe_rel(p, _ROOT) for p in produced],
                "metrics": {"reason": "doppler_csv_missing"},
            }
        )
        print("[warn] Stage B skipped: Doppler CSV is missing.")
        print(f"[ok] wrote: {out_json}")
        print(f"[ok] synced_to_public={len(synced)}")
        return 0

    try:
        doppler_df, source_cols = _load_doppler_dataframe(doppler_csv)
    except Exception as exc:
        payload.update(
            {
                "overall_status": "reject",
                "reason": f"doppler_csv_parse_error:{type(exc).__name__}",
                "error_message": str(exc),
            }
        )
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        produced = [out_json]
        synced = _sync_to_public(produced, private_root=out_dir, public_root=public_dir)
        append_event(
            {
                "event": "run_script",
                "script": "scripts/mercury/messenger_beta_stage_b_doppler_local_fit.py",
                "phase_step": "8.7.48.2",
                "status": "reject",
                "input": _safe_rel(doppler_csv, _ROOT),
                "outputs": [_safe_rel(p, _ROOT) for p in produced],
                "metrics": {"reason": "doppler_csv_parse_error"},
            }
        )
        print(f"[warn] Stage B parse error: {exc}")
        print(f"[ok] wrote: {out_json}")
        print(f"[ok] synced_to_public={len(synced)}")
        return 0

    arc_results, residual_df = _fit_arcs(doppler_df, min_points_per_arc=int(args.min_points_per_arc))
    _write_arc_summary_csv(out_arc_csv, arc_results)
    residual_df.to_csv(out_resid_csv, index=False)
    plot_note = _make_plot(arc_results, out_pdf=out_pdf, out_png=out_png)

    valid = [r for r in arc_results if r.status == "pass" and np.isfinite(r.rms_hz)]
    valid_arcs = int(len(valid))
    median_rms_hz = float(np.median([r.rms_hz for r in valid])) if valid_arcs > 0 else None
    status = _overall_status(valid_arcs, median_rms_hz, min_valid_arcs=int(args.min_valid_arcs))

    payload.update(
        {
            "overall_status": status,
            "source_columns": source_cols,
            "n_rows": int(len(doppler_df)),
            "n_arcs_total": int(len(arc_results)),
            "valid_arcs": valid_arcs,
            "median_rms_hz": median_rms_hz,
            "plot": "generated" if plot_note is None else plot_note,
            "notes": "Stage B MVP uses arc-local linear nuisance fit (intercept+drift).",
        }
    )
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    produced: List[Path] = [out_arc_csv, out_resid_csv, out_json]
    if plot_note is None:
        produced.extend([out_pdf, out_png])

    synced = _sync_to_public(produced, private_root=out_dir, public_root=public_dir)
    append_event(
        {
            "event": "run_script",
            "script": "scripts/mercury/messenger_beta_stage_b_doppler_local_fit.py",
            "phase_step": "8.7.48.2",
            "status": status,
            "input": _safe_rel(doppler_csv, _ROOT),
            "outputs": [_safe_rel(p, _ROOT) for p in produced],
            "metrics": {
                "n_rows": int(len(doppler_df)),
                "valid_arcs": valid_arcs,
                "median_rms_hz": median_rms_hz,
            },
        }
    )

    print(f"[ok] stage_b_overall={status}")
    print(f"[ok] wrote: {out_arc_csv}")
    print(f"[ok] wrote: {out_resid_csv}")
    print(f"[ok] wrote: {out_json}")
    if plot_note is None:
        print(f"[ok] wrote: {out_pdf}")
        print(f"[ok] wrote: {out_png}")
    else:
        print(f"[warn] plot skipped: {plot_note}")

    print(f"[ok] synced_to_public={len(synced)}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
