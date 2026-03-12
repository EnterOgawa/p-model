#!/usr/bin/env python3
"""
llr_kappa_llr_cluster_robust_audit.py

Step 8.7.47.13:
- station-target-night cluster で kappa の誤差を再評価する。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]


# 関数: `_safe_rel` の入出力契約と処理意図を定義する。
def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve()).replace("\\", "/")


# 関数: `_combine_status` の入出力契約と処理意図を定義する。

def _combine_status(values: Iterable[str]) -> str:
    norm = [str(v or "").strip().lower() for v in values if str(v or "").strip()]
    if not norm:
        return "reject"

    if any(v == "reject" for v in norm):
        return "reject"

    if all(v == "pass" for v in norm):
        return "pass"

    return "watch"


# 関数: `_parse_csv_list` の入出力契約と処理意図を定義する。

def _parse_csv_list(text: str) -> List[str]:
    out: List[str] = []
    for tok in str(text).split(","):
        t = tok.strip()
        if t:
            out.append(t)

    return out


# 関数: `_load_core_module` の入出力契約と処理意図を定義する。

def _load_core_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("llr_kappa_llr_direct_fit_core_cluster", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load core module spec: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# 関数: `_build_cluster_ids` の入出力契約と処理意図を定義する。

def _build_cluster_ids(df: pd.DataFrame) -> np.ndarray:
    if "epoch_utc" not in df.columns:
        raise ValueError("epoch_utc column is required for cluster IDs")

    if not pd.api.types.is_datetime64_any_dtype(df["epoch_utc"]):
        epoch = pd.to_datetime(df["epoch_utc"], utc=True, errors="coerce")
    else:
        epoch = df["epoch_utc"]

    night = epoch.dt.strftime("%Y-%m-%d").fillna("NA")
    station = df["station"].astype(str)
    target = df["target"].astype(str)
    cluster = station + "|" + target + "|" + night.astype(str)
    return cluster.to_numpy(dtype=object)


# 関数: `_fit_weighted_beta` の入出力契約と処理意図を定義する。

def _fit_weighted_beta(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = int(len(y))
    if sample_weight is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(sample_weight, dtype=float).reshape(-1)
        if len(w) != n:
            raise ValueError("sample_weight length mismatch")

    w = np.where(np.isfinite(w) & (w > 0.0), w, np.nan)
    ok = np.isfinite(w)
    if not np.any(ok):
        raise ValueError("all weights invalid")

    mean_w = float(np.nanmean(w[ok]))
    if not np.isfinite(mean_w) or mean_w <= 0.0:
        raise ValueError("invalid weight mean")

    w = np.where(ok, w / mean_w, 1.0)
    sw = np.sqrt(w)
    x_fit = x * sw[:, None]
    y_fit = y * sw
    beta_hat, _, _, _ = np.linalg.lstsq(x_fit, y_fit, rcond=None)
    resid_fit = y_fit - (x_fit @ beta_hat)
    return beta_hat, resid_fit, x_fit, w


# 関数: `_sandwich_slope_sigma` の入出力契約と処理意図を定義する。

def _sandwich_slope_sigma(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray],
    cluster_ids: np.ndarray,
) -> float:
    _, resid_fit, x_fit, _ = _fit_weighted_beta(x=x, y=y, sample_weight=sample_weight)
    n = int(x_fit.shape[0])
    k = int(x_fit.shape[1])
    if n <= k:
        return float("nan")

    cluster = np.asarray(cluster_ids, dtype=object).reshape(-1)
    if len(cluster) != n:
        return float("nan")

    xtx_inv = np.linalg.pinv(x_fit.T @ x_fit)
    unique_clusters = pd.Series(cluster).dropna().astype(str).unique().tolist()
    g = int(len(unique_clusters))
    if g <= 1:
        return float("nan")

    meat = np.zeros((k, k), dtype=float)
    for key in unique_clusters:
        mask = cluster.astype(str) == str(key)
        if not np.any(mask):
            continue

        xg = x_fit[mask, :]
        eg = resid_fit[mask]
        ug = xg.T @ eg
        meat += np.outer(ug, ug)

    cov = xtx_inv @ meat @ xtx_inv
    factor = (g / max(g - 1, 1)) * ((n - 1) / max(n - k, 1))
    cov = factor * cov
    var0 = float(cov[0, 0])
    if not np.isfinite(var0) or var0 < 0.0:
        return float("nan")

    return float(math.sqrt(var0))


# 関数: `_jackknife_kappa_sigma` の入出力契約と処理意図を定義する。

def _jackknife_kappa_sigma(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray],
    cluster_ids: np.ndarray,
) -> float:
    cluster = np.asarray(cluster_ids, dtype=object).reshape(-1)
    if len(cluster) != int(len(y)):
        return float("nan")

    unique_clusters = pd.Series(cluster).dropna().astype(str).unique().tolist()
    if len(unique_clusters) <= 1:
        return float("nan")

    theta: List[float] = []
    for key in unique_clusters:
        keep = cluster.astype(str) != str(key)
        if int(np.sum(keep)) <= int(x.shape[1]) + 1:
            continue

        x_sub = x[keep, :]
        y_sub = y[keep]
        if sample_weight is None:
            w_sub = None
        else:
            w_sub = np.asarray(sample_weight, dtype=float).reshape(-1)[keep]

        try:
            beta_hat, _, _, _ = _fit_weighted_beta(x=x_sub, y=y_sub, sample_weight=w_sub)
        except Exception:
            continue

        theta.append(float(1.0 + beta_hat[0]))

    m = int(len(theta))
    if m <= 1:
        return float("nan")

    arr = np.asarray(theta, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return float("nan")

    mu = float(np.mean(arr))
    var = float((len(arr) - 1) / len(arr) * np.sum(np.square(arr - mu)))
    if not np.isfinite(var) or var < 0.0:
        return float("nan")

    return float(math.sqrt(var))


# 関数: `_fit_with_cluster_sigma` の入出力契約と処理意図を定義する。

def _fit_with_cluster_sigma(
    core: Any,
    df_sub: pd.DataFrame,
    mode: str,
    sample_weight: Optional[np.ndarray],
    cluster_ids: np.ndarray,
) -> Optional[Dict[str, Any]]:
    if df_sub.empty:
        return None

    try:
        x, y, _ = core._build_design_matrix(df_sub, mode=mode)
        fr = core._fit_ols(mode=mode, x=x, y=y, sample_weight=sample_weight)
    except Exception:
        return None

    sigma_sand = _sandwich_slope_sigma(x=x, y=y, sample_weight=sample_weight, cluster_ids=cluster_ids)
    sigma_jack = _jackknife_kappa_sigma(x=x, y=y, sample_weight=sample_weight, cluster_ids=cluster_ids)
    sigma_candidates = [float(fr.kappa_sigma)]
    if np.isfinite(sigma_sand) and sigma_sand > 0.0:
        sigma_candidates.append(float(sigma_sand))

    if np.isfinite(sigma_jack) and sigma_jack > 0.0:
        sigma_candidates.append(float(sigma_jack))

    sigma_cluster = float(np.nanmax(np.asarray(sigma_candidates, dtype=float)))
    if not np.isfinite(sigma_cluster) or sigma_cluster <= 0.0:
        sigma_cluster = float("nan")

    delta = float(fr.kappa_est - 1.0)
    abs_z_cluster = float(abs(delta) / sigma_cluster) if np.isfinite(sigma_cluster) and sigma_cluster > 0.0 else float("nan")
    status_cluster = core._status_from_abs_z(abs_z_cluster)
    return {
        "kappa_est": float(fr.kappa_est),
        "kappa_sigma_indep": float(fr.kappa_sigma),
        "kappa_sigma_sandwich": float(sigma_sand),
        "kappa_sigma_jackknife": float(sigma_jack),
        "kappa_sigma_cluster": float(sigma_cluster),
        "abs_z_cluster": float(abs_z_cluster),
        "status_cluster": str(status_cluster),
        "rmse_ns": float(fr.rmse_ns),
        "aic_like": float(fr.aic_like),
        "n_points": int(fr.n_points),
    }


# 関数: `_build_group_table` の入出力契約と処理意図を定義する。

def _build_group_table(
    core: Any,
    df: pd.DataFrame,
    group_col: str,
    fit_mode: str,
    min_points: int,
    sample_weight: Optional[np.ndarray],
    cluster_ids: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    groups = sorted(set(df[group_col].astype(str).tolist()))
    for g in groups:
        mask = df[group_col].astype(str) == str(g)
        sub = df.loc[mask].copy()
        n = int(len(sub))
        if n < int(min_points):
            rows.append({group_col: str(g), "n_points": n, "fit_ok": False, "reason": f"n<{int(min_points)}"})
            continue

        w_sub: Optional[np.ndarray]
        if sample_weight is None:
            w_sub = None
        else:
            idx = np.flatnonzero(mask.to_numpy(dtype=bool))
            w_sub = np.asarray(sample_weight, dtype=float)[idx]

        cluster_sub = np.asarray(cluster_ids, dtype=object)[mask.to_numpy(dtype=bool)]
        fit = _fit_with_cluster_sigma(core=core, df_sub=sub, mode=fit_mode, sample_weight=w_sub, cluster_ids=cluster_sub)
        if fit is None:
            rows.append({group_col: str(g), "n_points": n, "fit_ok": False, "reason": "fit_failed"})
            continue

        rows.append(
            {
                group_col: str(g),
                "n_points": int(fit["n_points"]),
                "fit_ok": True,
                "reason": "",
                "kappa_est": float(fit["kappa_est"]),
                "kappa_sigma_indep": float(fit["kappa_sigma_indep"]),
                "kappa_sigma_sandwich": float(fit["kappa_sigma_sandwich"]),
                "kappa_sigma_jackknife": float(fit["kappa_sigma_jackknife"]),
                "kappa_sigma_cluster": float(fit["kappa_sigma_cluster"]),
                "abs_z_cluster": float(fit["abs_z_cluster"]),
                "status_cluster": str(fit["status_cluster"]),
            }
        )

    out_df = pd.DataFrame(rows).sort_values([group_col]).reset_index(drop=True)
    values = pd.to_numeric(out_df.get("kappa_est"), errors="coerce").to_numpy(dtype=float)
    sigma_cluster = pd.to_numeric(out_df.get("kappa_sigma_cluster"), errors="coerce").to_numpy(dtype=float)
    sigma_indep = pd.to_numeric(out_df.get("kappa_sigma_indep"), errors="coerce").to_numpy(dtype=float)
    stats_cluster = core._weighted_mean_and_chi2(values=values, sigma=sigma_cluster)
    stats_indep = core._weighted_mean_and_chi2(values=values, sigma=sigma_indep)
    chi2_cluster = float(stats_cluster.get("chi2_dof", float("nan")))
    chi2_indep = float(stats_indep.get("chi2_dof", float("nan")))
    summary = {
        "fit_mode": str(fit_mode),
        "min_points": int(min_points),
        "n_groups": int(len(out_df)),
        "indep_chi2_dof": chi2_indep,
        "cluster_chi2_dof": chi2_cluster,
        "indep_status": core._consistency_status_from_chi2_dof(chi2_indep),
        "cluster_status": core._consistency_status_from_chi2_dof(chi2_cluster),
    }
    return out_df, summary


# 関数: `_run_policy_cluster_audit` の入出力契約と処理意図を定義する。

def _run_policy_cluster_audit(
    core: Any,
    df: pd.DataFrame,
    cluster_ids: np.ndarray,
    fit_mode: str,
    schemes: Sequence[str],
    floor_station: int,
    floor_target: int,
    floor_station_target: int,
    max_weight_cap: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for scheme in schemes:
        w = core._build_imbalance_weight(
            df,
            scheme=str(scheme),
            floor_station=int(floor_station),
            floor_target=int(floor_target),
            floor_station_target=int(floor_station_target),
            max_weight_cap=float(max_weight_cap),
        )
        fit = _fit_with_cluster_sigma(core=core, df_sub=df, mode=fit_mode, sample_weight=w, cluster_ids=cluster_ids)
        row = {"weight_scheme": str(scheme), "fit_ok": fit is not None}
        if fit is not None:
            row.update(fit)

        rows.append(row)

    out_df = pd.DataFrame(rows).sort_values(["aic_like", "weight_scheme"], na_position="last").reset_index(drop=True)
    valid = out_df[
        np.isfinite(pd.to_numeric(out_df.get("kappa_est"), errors="coerce"))
        & np.isfinite(pd.to_numeric(out_df.get("kappa_sigma_cluster"), errors="coerce"))
        & (pd.to_numeric(out_df.get("kappa_sigma_cluster"), errors="coerce") > 0)
    ].copy()
    if valid.empty:
        return out_df, {"n_valid_policies": 0, "max_abs_z_vs_uniform_cluster": float("nan"), "z_status_cluster": "reject"}

    if "uniform" in set(valid["weight_scheme"].astype(str).tolist()):
        ref = valid[valid["weight_scheme"].astype(str) == "uniform"].iloc[0]
    else:
        ref = valid.iloc[0]

    ref_k = float(ref["kappa_est"])
    ref_s = float(ref["kappa_sigma_cluster"])
    z_vals: List[float] = []
    for row in valid.to_dict(orient="records"):
        kk = float(row.get("kappa_est", float("nan")))
        ss = float(row.get("kappa_sigma_cluster", float("nan")))
        denom = math.sqrt(max((ref_s * ref_s) + (ss * ss), 1e-30))
        if np.isfinite(kk) and np.isfinite(denom) and denom > 0.0:
            z_vals.append(abs((kk - ref_k) / denom))

    max_abs_z = float(np.nanmax(np.asarray(z_vals, dtype=float))) if z_vals else float("nan")
    values = pd.to_numeric(valid["kappa_est"], errors="coerce").to_numpy(dtype=float)
    sigma = pd.to_numeric(valid["kappa_sigma_cluster"], errors="coerce").to_numpy(dtype=float)
    stats = core._weighted_mean_and_chi2(values=values, sigma=sigma)
    chi2 = float(stats.get("chi2_dof", float("nan")))
    summary = {
        "n_valid_policies": int(len(valid)),
        "max_abs_z_vs_uniform_cluster": max_abs_z,
        "z_status_cluster": core._status_from_abs_z(max_abs_z),
        "cluster_chi2_dof": chi2,
        "cluster_status": core._consistency_status_from_chi2_dof(chi2),
    }
    return out_df, summary


# 関数: `_load_baseline_metrics` の入出力契約と処理意図を定義する。

def _load_baseline_metrics(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"available": False}

    try:
        j = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"available": False}

    imbalance = j.get("imbalance_audit") if isinstance(j.get("imbalance_audit"), dict) else {}
    station = imbalance.get("station_stratified") if isinstance(imbalance.get("station_stratified"), dict) else {}
    target = imbalance.get("target_stratified") if isinstance(imbalance.get("target_stratified"), dict) else {}
    robust = imbalance.get("robustness_envelope") if isinstance(imbalance.get("robustness_envelope"), dict) else {}
    return {
        "available": True,
        "path": str(path),
        "station_chi2_dof": float(station.get("chi2_dof", float("nan"))),
        "target_chi2_dof": float(target.get("chi2_dof", float("nan"))),
        "imbalance_max_abs_z_vs_uniform": float(robust.get("max_abs_z_vs_uniform", float("nan"))),
    }


# 関数: `_write_plot` の入出力契約と処理意図を定義する。

def _write_plot(
    baseline: Dict[str, Any],
    station_summary: Dict[str, Any],
    target_summary: Dict[str, Any],
    policy_summary: Dict[str, Any],
    out_pdf: Path,
    out_png: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.8))
    ax0, ax1, ax2 = axes

    labels = ["station", "target"]
    indep = [
        float(baseline.get("station_chi2_dof", float("nan"))),
        float(baseline.get("target_chi2_dof", float("nan"))),
    ]
    robust = [
        float(station_summary.get("cluster_chi2_dof", float("nan"))),
        float(target_summary.get("cluster_chi2_dof", float("nan"))),
    ]
    x = np.arange(2, dtype=float)
    ax0.bar(x - 0.16, indep, width=0.30, color="#999999", label="indep")
    ax0.bar(x + 0.16, robust, width=0.30, color="#1f77b4", label="cluster")
    ax0.axhline(2.0, color="#2ca02c", linestyle="--", linewidth=1.0)
    ax0.axhline(5.0, color="#ff7f0e", linestyle="--", linewidth=1.0)
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels)
    ax0.set_ylabel("chi2/dof")
    ax0.set_title("Stratified chi2/dof")
    ax0.grid(axis="y", alpha=0.25)
    ax0.legend(frameon=False, fontsize=9)

    ax1.bar(
        [0, 1],
        [
            float(baseline.get("imbalance_max_abs_z_vs_uniform", float("nan"))),
            float(policy_summary.get("max_abs_z_vs_uniform_cluster", float("nan"))),
        ],
        color=["#999999", "#d62728"],
        width=0.55,
    )
    ax1.axhline(2.0, color="#2ca02c", linestyle="--", linewidth=1.0)
    ax1.axhline(3.0, color="#ff7f0e", linestyle="--", linewidth=1.0)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["indep", "cluster"])
    ax1.set_ylabel("max |z(policy-uniform)|")
    ax1.set_title("Imbalance gate")
    ax1.grid(axis="y", alpha=0.25)

    txt = (
        f"station: {station_summary.get('cluster_status', 'reject')}\n"
        f"target: {target_summary.get('cluster_status', 'reject')}\n"
        f"imbalance: {policy_summary.get('z_status_cluster', 'reject')}\n"
        f"overall: {_combine_status([station_summary.get('cluster_status', 'reject'), target_summary.get('cluster_status', 'reject'), policy_summary.get('z_status_cluster', 'reject')])}"
    )
    ax2.axis("off")
    ax2.text(0.02, 0.95, txt, va="top", ha="left", fontsize=11.0)
    ax2.set_title("Cluster-Robust Gate Status")

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


# 関数: `_sync_outputs` の入出力契約と処理意図を定義する。

def _sync_outputs(paths: Iterable[Path], private_root: Path, public_root: Path) -> List[str]:
    synced: List[str] = []
    for src in paths:
        rel = src.resolve().relative_to(private_root.resolve())
        dst = public_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        synced.append(str(dst))

    return synced


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    ap = argparse.ArgumentParser(description="LLR cluster-robust error audit.")
    ap.add_argument(
        "--points-csv",
        type=str,
        default=str(ROOT / "output" / "private" / "llr" / "batch" / "llr_batch_points.csv"),
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "output" / "private" / "llr"),
    )
    ap.add_argument(
        "--public-dir",
        type=str,
        default=str(ROOT / "output" / "public" / "llr"),
    )
    ap.add_argument(
        "--baseline-metrics",
        type=str,
        default=str(ROOT / "output" / "public" / "llr" / "llr_kappa_llr_metrics.json"),
    )
    ap.add_argument(
        "--core-script",
        type=str,
        default=str(ROOT / "scripts" / "llr" / "llr_kappa_llr_direct_fit.py"),
    )
    ap.add_argument("--station-fit-mode", type=str, default="station_target_year")
    ap.add_argument("--target-fit-mode", type=str, default="station_target_year")
    ap.add_argument("--imbalance-fit-mode", type=str, default="station_target_year")
    ap.add_argument("--min-points-station", type=int, default=180)
    ap.add_argument("--min-points-target", type=int, default=180)
    ap.add_argument(
        "--imbalance-schemes",
        type=str,
        default="uniform,inv_station,inv_target,inv_station_target,station_cap_p95",
    )
    ap.add_argument("--stratified-weight-scheme", type=str, default="inv_station_target")
    ap.add_argument("--weight-floor-station", type=int, default=180)
    ap.add_argument("--weight-floor-target", type=int, default=180)
    ap.add_argument("--weight-floor-station-target", type=int, default=120)
    ap.add_argument("--max-weight-cap", type=float, default=8.0)
    args = ap.parse_args()

    points_csv = Path(str(args.points_csv))
    out_dir = Path(str(args.out_dir))
    public_dir = Path(str(args.public_dir))
    baseline_metrics = Path(str(args.baseline_metrics))
    core_script = Path(str(args.core_script))
    if not points_csv.is_absolute():
        points_csv = (ROOT / points_csv).resolve()

    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()

    if not public_dir.is_absolute():
        public_dir = (ROOT / public_dir).resolve()

    if not baseline_metrics.is_absolute():
        baseline_metrics = (ROOT / baseline_metrics).resolve()

    if not core_script.is_absolute():
        core_script = (ROOT / core_script).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)

    core = _load_core_module(core_script)
    df = core._read_points(points_csv)
    if df.empty:
        raise RuntimeError(f"no valid rows from {points_csv}")

    cluster_ids = _build_cluster_ids(df)
    n_clusters = int(len(pd.Series(cluster_ids).astype(str).unique()))
    schemes = _parse_csv_list(str(args.imbalance_schemes))
    w_strat = core._build_imbalance_weight(
        df,
        scheme=str(args.stratified_weight_scheme),
        floor_station=int(args.weight_floor_station),
        floor_target=int(args.weight_floor_target),
        floor_station_target=int(args.weight_floor_station_target),
        max_weight_cap=float(args.max_weight_cap),
    )
    station_df, station_summary = _build_group_table(
        core=core,
        df=df,
        group_col="station",
        fit_mode=str(args.station_fit_mode),
        min_points=int(args.min_points_station),
        sample_weight=w_strat,
        cluster_ids=cluster_ids,
    )
    target_df, target_summary = _build_group_table(
        core=core,
        df=df,
        group_col="target",
        fit_mode=str(args.target_fit_mode),
        min_points=int(args.min_points_target),
        sample_weight=w_strat,
        cluster_ids=cluster_ids,
    )
    policy_df, policy_summary = _run_policy_cluster_audit(
        core=core,
        df=df,
        cluster_ids=cluster_ids,
        fit_mode=str(args.imbalance_fit_mode),
        schemes=schemes,
        floor_station=int(args.weight_floor_station),
        floor_target=int(args.weight_floor_target),
        floor_station_target=int(args.weight_floor_station_target),
        max_weight_cap=float(args.max_weight_cap),
    )
    baseline = _load_baseline_metrics(baseline_metrics)
    station_status = str(station_summary.get("cluster_status", "reject"))
    target_status = str(target_summary.get("cluster_status", "reject"))
    imbalance_status = str(policy_summary.get("z_status_cluster", "reject"))
    overall = _combine_status([station_status, target_status, imbalance_status])

    policy_csv = out_dir / "llr_kappa_llr_cluster_robust_policy_summary.csv"
    station_csv = out_dir / "llr_kappa_llr_cluster_robust_station_summary.csv"
    target_csv = out_dir / "llr_kappa_llr_cluster_robust_target_summary.csv"
    metrics_json = out_dir / "llr_kappa_llr_cluster_robust_metrics.json"
    plot_pdf = out_dir / "llr_kappa_llr_cluster_robust_audit.pdf"
    plot_png = out_dir / "llr_kappa_llr_cluster_robust_audit.png"

    policy_df.to_csv(policy_csv, index=False)
    station_df.to_csv(station_csv, index=False)
    target_df.to_csv(target_csv, index=False)
    _write_plot(
        baseline=baseline,
        station_summary=station_summary,
        target_summary=target_summary,
        policy_summary=policy_summary,
        out_pdf=plot_pdf,
        out_png=plot_png,
    )

    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase": {"step": "8.7.47.13"},
        "input": {
            "points_csv": _safe_rel(points_csv, ROOT),
            "n_points": int(len(df)),
            "n_station": int(df["station"].nunique()),
            "n_target": int(df["target"].nunique()),
            "n_clusters_station_target_night": int(n_clusters),
            "station_fit_mode": str(args.station_fit_mode),
            "target_fit_mode": str(args.target_fit_mode),
            "imbalance_fit_mode": str(args.imbalance_fit_mode),
            "stratified_weight_scheme": str(args.stratified_weight_scheme),
            "imbalance_schemes": [str(s) for s in schemes],
        },
        "baseline_reference": baseline,
        "cluster_robust": {
            "station_stratified": station_summary,
            "target_stratified": target_summary,
            "imbalance_policy": policy_summary,
            "delta_vs_baseline": {
                "station_chi2_dof_delta_cluster_minus_indep": (
                    float(station_summary["cluster_chi2_dof"] - baseline["station_chi2_dof"])
                    if bool(baseline.get("available"))
                    else float("nan")
                ),
                "target_chi2_dof_delta_cluster_minus_indep": (
                    float(target_summary["cluster_chi2_dof"] - baseline["target_chi2_dof"])
                    if bool(baseline.get("available"))
                    else float("nan")
                ),
                "imbalance_max_abs_z_delta_cluster_minus_indep": (
                    float(policy_summary["max_abs_z_vs_uniform_cluster"] - baseline["imbalance_max_abs_z_vs_uniform"])
                    if bool(baseline.get("available"))
                    else float("nan")
                ),
            },
            "gate_status": {
                "station_stratified_gate": station_status,
                "target_stratified_gate": target_status,
                "imbalance_policy_gate": imbalance_status,
            },
            "overall_status": overall,
        },
        "outputs": {
            "policy_csv": _safe_rel(policy_csv, ROOT),
            "station_csv": _safe_rel(station_csv, ROOT),
            "target_csv": _safe_rel(target_csv, ROOT),
            "metrics_json": _safe_rel(metrics_json, ROOT),
            "plot_pdf": _safe_rel(plot_pdf, ROOT),
            "plot_png": _safe_rel(plot_png, ROOT),
        },
    }
    metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    synced = _sync_outputs(
        paths=[policy_csv, station_csv, target_csv, metrics_json, plot_pdf, plot_png],
        private_root=out_dir,
        public_root=public_dir,
    )

    print(f"Wrote: {policy_csv}")
    print(f"Wrote: {station_csv}")
    print(f"Wrote: {target_csv}")
    print(f"Wrote: {metrics_json}")
    print(f"Wrote: {plot_pdf}")
    print(f"Wrote: {plot_png}")
    print(f"Synced: {len(synced)} files")
    print(f"Status: {overall}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
