#!/usr/bin/env python3
"""
llr_kappa_llr_homogeneous_subset_audit.py

Step 8.7.47.15:
- LLR improvement roadmap Step 4 (homogeneous subset re-fit).
- Refit kappa on progressively homogeneous subsets:
  1) APOL post-ACS only
  2) Apollo reflector only
  3) full-Moon 30 deg exclusion
  4) multi-reflector night only
- Fix kappa and internal chi2/dof stability for subset progression.
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


# 関数: `_load_core_module` の入出力契約と処理意図を定義する。

def _load_core_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("llr_kappa_llr_core_homogeneous", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load core module spec: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# 関数: `_build_cluster_ids` の入出力契約と処理意図を定義する。

def _build_cluster_ids(df: pd.DataFrame) -> np.ndarray:
    epoch = (
        df["epoch_utc"]
        if pd.api.types.is_datetime64_any_dtype(df["epoch_utc"])
        else pd.to_datetime(df["epoch_utc"], utc=True, errors="coerce")
    )
    night = epoch.dt.strftime("%Y-%m-%d").fillna("NA")
    return (df["station"].astype(str) + "|" + df["target"].astype(str) + "|" + night.astype(str)).to_numpy(dtype=object)


# 関数: `_fit_weighted_beta` の入出力契約と処理意図を定義する。

def _fit_weighted_beta(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = np.ones(len(y), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float).reshape(-1)
    w = np.where(np.isfinite(w) & (w > 0.0), w, np.nan)
    ok = np.isfinite(w)
    if not np.any(ok):
        raise ValueError("all weights invalid")

    w = np.where(ok, w / float(np.nanmean(w[ok])), 1.0)
    sw = np.sqrt(w)
    x_fit = x * sw[:, None]
    y_fit = y * sw
    beta_hat, _, _, _ = np.linalg.lstsq(x_fit, y_fit, rcond=None)
    resid_fit = y_fit - (x_fit @ beta_hat)
    return beta_hat, resid_fit, x_fit


# 関数: `_sandwich_slope_sigma` の入出力契約と処理意図を定義する。

def _sandwich_slope_sigma(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray],
    cluster_ids: np.ndarray,
) -> float:
    _, resid_fit, x_fit = _fit_weighted_beta(x=x, y=y, sample_weight=sample_weight)
    n = int(x_fit.shape[0])
    k = int(x_fit.shape[1])
    if n <= k:
        return float("nan")

    cluster = np.asarray(cluster_ids, dtype=object).reshape(-1)
    if len(cluster) != n:
        return float("nan")

    keys = pd.Series(cluster).dropna().astype(str).unique().tolist()
    g = int(len(keys))
    if g <= 1:
        return float("nan")

    xtx_inv = np.linalg.pinv(x_fit.T @ x_fit)
    meat = np.zeros((k, k), dtype=float)
    for key in keys:
        mask = cluster.astype(str) == str(key)
        xg = x_fit[mask, :]
        eg = resid_fit[mask]
        ug = xg.T @ eg
        meat += np.outer(ug, ug)

    cov = xtx_inv @ meat @ xtx_inv
    cov *= (g / max(g - 1, 1)) * ((n - 1) / max(n - k, 1))
    var0 = float(cov[0, 0])
    return float(math.sqrt(var0)) if np.isfinite(var0) and var0 >= 0.0 else float("nan")


# 関数: `_jackknife_kappa_sigma` の入出力契約と処理意図を定義する。

def _jackknife_kappa_sigma(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray],
    cluster_ids: np.ndarray,
) -> float:
    cluster = np.asarray(cluster_ids, dtype=object).reshape(-1)
    keys = pd.Series(cluster).dropna().astype(str).unique().tolist()
    if len(keys) <= 1:
        return float("nan")

    vals: List[float] = []
    for key in keys:
        keep = cluster.astype(str) != str(key)
        if int(np.sum(keep)) <= int(x.shape[1]) + 1:
            continue

        x_sub = x[keep, :]
        y_sub = y[keep]
        w_sub = None if sample_weight is None else np.asarray(sample_weight, dtype=float).reshape(-1)[keep]
        try:
            beta_hat, _, _ = _fit_weighted_beta(x=x_sub, y=y_sub, sample_weight=w_sub)
        except Exception:
            continue

        vals.append(float(1.0 + beta_hat[0]))

    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return float("nan")

    mu = float(np.mean(arr))
    var = float((len(arr) - 1) / len(arr) * np.sum(np.square(arr - mu)))
    return float(math.sqrt(var)) if np.isfinite(var) and var >= 0.0 else float("nan")


# 関数: `_fit_with_cluster_sigma` の入出力契約と処理意図を定義する。

def _fit_with_cluster_sigma(
    core: Any,
    df_sub: pd.DataFrame,
    *,
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
    sigmas = [float(fr.kappa_sigma)]
    if np.isfinite(sigma_sand) and sigma_sand > 0.0:
        sigmas.append(float(sigma_sand))

    if np.isfinite(sigma_jack) and sigma_jack > 0.0:
        sigmas.append(float(sigma_jack))

    sigma_cluster = float(np.nanmax(np.asarray(sigmas, dtype=float)))
    abs_z = float(abs(fr.kappa_est - 1.0) / sigma_cluster) if np.isfinite(sigma_cluster) and sigma_cluster > 0.0 else float("nan")
    return {
        "kappa_est": float(fr.kappa_est),
        "kappa_sigma_cluster": float(sigma_cluster),
        "kappa_sigma_indep": float(fr.kappa_sigma),
        "kappa_sigma_sandwich": float(sigma_sand),
        "kappa_sigma_jackknife": float(sigma_jack),
        "abs_z_cluster": abs_z,
        "status_cluster": core._status_from_abs_z(abs_z),
        "n_points": int(fr.n_points),
    }


# 関数: `_moon_phase_elongation_deg` の入出力契約と処理意図を定義する。

def _moon_phase_elongation_deg(epoch_utc: pd.Series) -> np.ndarray:
    """Approximate lunar elongation from a fixed synodic phase model.

    This is used only for homogeneous subset gating (full-Moon ±30 deg exclusion),
    not for physical parameter estimation itself.
    """
    t = pd.to_datetime(epoch_utc, utc=True, errors="coerce")
    unix_s = t.astype("int64", copy=False).astype(float) / 1e9
    jd = unix_s / 86400.0 + 2440587.5
    synodic_month = 29.53058867
    ref_new_moon_jd = 2451550.1
    phase = np.mod((jd - ref_new_moon_jd) / synodic_month, 1.0)
    elong = 360.0 * phase
    return np.asarray(elong, dtype=float)


# 関数: `_phase_dist_from_full_deg` の入出力契約と処理意図を定義する。

def _phase_dist_from_full_deg(elong_deg: np.ndarray) -> np.ndarray:
    wrap = ((elong_deg - 180.0 + 180.0) % 360.0) - 180.0
    return np.abs(wrap)


# 関数: `_prepare_subset_columns` の入出力契約と処理意図を定義する。

def _prepare_subset_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    epoch = pd.to_datetime(out["epoch_utc"], utc=True, errors="coerce")
    out["epoch_utc"] = epoch
    out["night_utc"] = epoch.dt.strftime("%Y-%m-%d")
    out["station_target"] = out["station"].astype(str) + "|" + out["target"].astype(str)
    elong = _moon_phase_elongation_deg(epoch)
    out["moon_phase_elongation_deg"] = elong
    out["moon_phase_dist_from_full_deg"] = _phase_dist_from_full_deg(elong)
    return out


# 関数: `_subset_specs` の入出力契約と処理意図を定義する。

def _subset_specs() -> List[Dict[str, str]]:
    return [
        {
            "subset_id": "reference_all",
            "label": "Reference all",
            "description": "All inlier points after core quality gates.",
        },
        {
            "subset_id": "apol_post_acs",
            "label": "APOL post-ACS",
            "description": "Station APOL and year >= ACS cut.",
        },
        {
            "subset_id": "apol_post_acs_apollo_reflectors",
            "label": "APOL post+Apollo",
            "description": "APOL post-ACS and Apollo reflectors only.",
        },
        {
            "subset_id": "apol_post_acs_apollo_reflectors_no_fullmoon30",
            "label": "No full-Moon 30",
            "description": "Exclude full-Moon ±30 deg window.",
        },
        {
            "subset_id": "apol_post_acs_apollo_reflectors_no_fullmoon30_multiref_night",
            "label": "No fullM + multi-ref",
            "description": "Keep nights with >=2 Apollo reflectors.",
        },
    ]


# 関数: `_build_subset_mask` の入出力契約と処理意図を定義する。

def _build_subset_mask(
    df: pd.DataFrame,
    subset_id: str,
    *,
    apol_post_year: int,
    fullmoon_exclusion_deg: float,
) -> np.ndarray:
    station = df["station"].astype(str).to_numpy(dtype=object)
    year = pd.to_numeric(df["year"], errors="coerce").to_numpy(dtype=float)
    target = df["target"].astype(str).str.lower().to_numpy(dtype=object)
    phase_dist = pd.to_numeric(df["moon_phase_dist_from_full_deg"], errors="coerce").to_numpy(dtype=float)
    night = df["night_utc"].astype(str)
    apollo_targets = {"apollo11", "apollo14", "apollo15"}

    mask_apol_post = (station == "APOL") & np.isfinite(year) & (year >= float(apol_post_year))
    mask_apollo_ref = np.array([str(v) in apollo_targets for v in target], dtype=bool)
    mask_not_fullmoon = np.isfinite(phase_dist) & (phase_dist > float(fullmoon_exclusion_deg))

    if subset_id == "reference_all":
        return np.ones(len(df), dtype=bool)

    if subset_id == "apol_post_acs":
        return mask_apol_post

    if subset_id == "apol_post_acs_apollo_reflectors":
        return mask_apol_post & mask_apollo_ref

    if subset_id == "apol_post_acs_apollo_reflectors_no_fullmoon30":
        return mask_apol_post & mask_apollo_ref & mask_not_fullmoon

    if subset_id == "apol_post_acs_apollo_reflectors_no_fullmoon30_multiref_night":
        base = mask_apol_post & mask_apollo_ref & mask_not_fullmoon
        if not np.any(base):
            return base

        sdf = df.loc[base, ["night_utc", "target"]].copy()
        per_night = sdf.groupby("night_utc", observed=False)["target"].nunique()
        keep_nights = set(per_night[per_night >= 2].index.tolist())
        return base & night.isin(keep_nights).to_numpy(dtype=bool)

    raise ValueError(f"unknown subset_id: {subset_id}")


# 関数: `_fit_station_target_consistency` の入出力契約と処理意図を定義する。

def _fit_station_target_consistency(
    core: Any,
    df: pd.DataFrame,
    *,
    fit_mode: str,
    sample_weight: Optional[np.ndarray],
    cluster_ids: np.ndarray,
    min_points_group: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    groups = sorted(set(df["station_target"].astype(str).tolist()))
    for g in groups:
        mask = df["station_target"].astype(str) == g
        sub = df.loc[mask].copy()
        n = int(len(sub))
        if n < int(min_points_group):
            rows.append(
                {
                    "station_target": g,
                    "n_points": n,
                    "fit_ok": False,
                    "reason": f"n<{int(min_points_group)}",
                }
            )
            continue

        idx = np.flatnonzero(mask.to_numpy(dtype=bool))
        w_sub = None if sample_weight is None else np.asarray(sample_weight, dtype=float)[idx]
        c_sub = np.asarray(cluster_ids, dtype=object)[mask.to_numpy(dtype=bool)]
        fit = _fit_with_cluster_sigma(
            core=core,
            df_sub=sub,
            mode=fit_mode,
            sample_weight=w_sub,
            cluster_ids=c_sub,
        )
        if fit is None:
            rows.append(
                {
                    "station_target": g,
                    "n_points": n,
                    "fit_ok": False,
                    "reason": "fit_failed",
                }
            )
            continue

        rows.append(
            {
                "station_target": g,
                "n_points": n,
                "fit_ok": True,
                "reason": "",
                **fit,
            }
        )

    out_df = pd.DataFrame(rows).sort_values(["station_target"]).reset_index(drop=True)
    valid = out_df[
        out_df.get("fit_ok", pd.Series(dtype=bool)).astype(bool)
        & np.isfinite(pd.to_numeric(out_df.get("kappa_est"), errors="coerce"))
        & np.isfinite(pd.to_numeric(out_df.get("kappa_sigma_cluster"), errors="coerce"))
        & (pd.to_numeric(out_df.get("kappa_sigma_cluster"), errors="coerce") > 0.0)
    ].copy()

    if valid.empty:
        return out_df, {
            "n_groups_total": int(len(out_df)),
            "n_groups_valid": 0,
            "weighted_mean_kappa": float("nan"),
            "weighted_sigma_kappa": float("nan"),
            "chi2_dof_cluster": float("nan"),
            "status_cluster": "reject",
        }

    stats = core._weighted_mean_and_chi2(
        values=pd.to_numeric(valid["kappa_est"], errors="coerce").to_numpy(dtype=float),
        sigma=pd.to_numeric(valid["kappa_sigma_cluster"], errors="coerce").to_numpy(dtype=float),
    )
    chi2 = float(stats.get("chi2_dof", float("nan")))
    return out_df, {
        "n_groups_total": int(len(out_df)),
        "n_groups_valid": int(len(valid)),
        "weighted_mean_kappa": float(stats.get("weighted_mean", float("nan"))),
        "weighted_sigma_kappa": float(stats.get("weighted_sigma", float("nan"))),
        "chi2_dof_cluster": chi2,
        "status_cluster": core._consistency_status_from_chi2_dof(chi2),
    }


# 関数: `_write_plot` の入出力契約と処理意図を定義する。

def _write_plot(summary_df: pd.DataFrame, overall_status: str, out_pdf: Path, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.2))
    ax0, ax1, ax2 = axes

    s = summary_df.copy()
    s["idx"] = np.arange(len(s))
    labels = s["label"].astype(str).tolist()

    kappa = pd.to_numeric(s.get("kappa_est"), errors="coerce").to_numpy(dtype=float)
    sigma = pd.to_numeric(s.get("kappa_sigma_cluster"), errors="coerce").to_numpy(dtype=float)
    chi2 = pd.to_numeric(s.get("group_chi2_dof_cluster"), errors="coerce").to_numpy(dtype=float)
    z_ref = pd.to_numeric(s.get("abs_z_vs_reference"), errors="coerce").to_numpy(dtype=float)

    ax0.errorbar(
        s["idx"].to_numpy(dtype=float),
        kappa,
        yerr=sigma,
        fmt="o",
        color="#1f77b4",
        ecolor="#1f77b4",
        capsize=3,
    )
    ax0.axhline(1.0, color="#2ca02c", linestyle="--", linewidth=1.0)
    ax0.set_xticks(s["idx"].to_numpy(dtype=float))
    ax0.set_xticklabels(labels, rotation=28, ha="right")
    ax0.set_ylabel("kappa")
    ax0.set_title("Subset kappa (cluster-robust sigma)")
    ax0.grid(axis="y", alpha=0.25)

    ax1.bar(s["idx"].to_numpy(dtype=float), chi2, color="#d62728", width=0.6)
    ax1.axhline(2.0, color="#2ca02c", linestyle="--", linewidth=1.0)
    ax1.axhline(5.0, color="#ff7f0e", linestyle="--", linewidth=1.0)
    ax1.set_xticks(s["idx"].to_numpy(dtype=float))
    ax1.set_xticklabels(labels, rotation=28, ha="right")
    ax1.set_ylabel("chi2/dof")
    ax1.set_title("Internal consistency (station-target)")
    ax1.grid(axis="y", alpha=0.25)

    ax2.bar(s["idx"].to_numpy(dtype=float), z_ref, color="#9467bd", width=0.6)
    ax2.axhline(2.0, color="#2ca02c", linestyle="--", linewidth=1.0)
    ax2.axhline(3.0, color="#ff7f0e", linestyle="--", linewidth=1.0)
    ax2.set_xticks(s["idx"].to_numpy(dtype=float))
    ax2.set_xticklabels(labels, rotation=28, ha="right")
    ax2.set_ylabel("|z| vs reference_all")
    ax2.set_title("Subset stability")
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle(f"LLR homogeneous subset audit (8.7.47.15): overall={overall_status}", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


# 関数: `_sync_outputs` の入出力契約と処理意図を定義する。

def _sync_outputs(paths: Iterable[Path], *, private_root: Path, public_root: Path) -> List[str]:
    out: List[str] = []
    for src in paths:
        rel = src.resolve().relative_to(private_root.resolve())
        dst = public_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        out.append(str(dst))

    return out


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    ap = argparse.ArgumentParser(description="LLR homogeneous subset audit (Step 8.7.47.15).")
    ap.add_argument("--points-csv", type=str, default=str(ROOT / "output" / "private" / "llr" / "batch" / "llr_batch_points.csv"))
    ap.add_argument("--out-dir", type=str, default=str(ROOT / "output" / "private" / "llr"))
    ap.add_argument("--public-dir", type=str, default=str(ROOT / "output" / "public" / "llr"))
    ap.add_argument("--core-script", type=str, default=str(ROOT / "scripts" / "llr" / "llr_kappa_llr_direct_fit.py"))
    ap.add_argument("--fit-mode", type=str, default="station_target_year")
    ap.add_argument("--weight-scheme", type=str, default="inv_station_target")
    ap.add_argument("--weight-floor-station", type=int, default=180)
    ap.add_argument("--weight-floor-target", type=int, default=180)
    ap.add_argument("--weight-floor-station-target", type=int, default=120)
    ap.add_argument("--max-weight-cap", type=float, default=8.0)
    ap.add_argument("--min-points-subset", type=int, default=200)
    ap.add_argument("--min-points-group", type=int, default=80)
    ap.add_argument("--apol-post-year", type=int, default=2021)
    ap.add_argument("--fullmoon-exclusion-deg", type=float, default=30.0)
    args = ap.parse_args()

    points_csv = Path(str(args.points_csv))
    out_dir = Path(str(args.out_dir))
    public_dir = Path(str(args.public_dir))
    core_script = Path(str(args.core_script))
    if not points_csv.is_absolute():
        points_csv = (ROOT / points_csv).resolve()

    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()

    if not public_dir.is_absolute():
        public_dir = (ROOT / public_dir).resolve()

    if not core_script.is_absolute():
        core_script = (ROOT / core_script).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)
    core = _load_core_module(core_script)

    df = core._read_points(points_csv)
    if df.empty:
        raise RuntimeError(f"no valid rows from {points_csv}")

    df = _prepare_subset_columns(df)
    specs = _subset_specs()

    summary_rows: List[Dict[str, Any]] = []
    group_parts: List[pd.DataFrame] = []
    for spec in specs:
        subset_id = str(spec["subset_id"])
        mask = _build_subset_mask(
            df,
            subset_id=subset_id,
            apol_post_year=int(args.apol_post_year),
            fullmoon_exclusion_deg=float(args.fullmoon_exclusion_deg),
        )
        sub = df.loc[mask].copy().reset_index(drop=True)
        n_points = int(len(sub))
        n_station = int(sub["station"].nunique()) if n_points > 0 else 0
        n_target = int(sub["target"].nunique()) if n_points > 0 else 0
        n_night = int(sub["night_utc"].nunique()) if n_points > 0 else 0

        row: Dict[str, Any] = {
            "subset_id": subset_id,
            "label": str(spec["label"]),
            "description": str(spec["description"]),
            "n_points": n_points,
            "n_station": n_station,
            "n_target": n_target,
            "n_nights": n_night,
            "fit_ok": False,
            "reason": "",
            "kappa_est": float("nan"),
            "kappa_sigma_cluster": float("nan"),
            "kappa_sigma_indep": float("nan"),
            "kappa_sigma_sandwich": float("nan"),
            "kappa_sigma_jackknife": float("nan"),
            "abs_z_cluster": float("nan"),
            "status_cluster": "reject",
            "group_n_total": 0,
            "group_n_valid": 0,
            "group_chi2_dof_cluster": float("nan"),
            "group_status_cluster": "reject",
            "subset_status": "reject",
            "subset_internal_status": "reject",
        }

        if n_points < int(args.min_points_subset):
            row["reason"] = f"n<{int(args.min_points_subset)}"
            summary_rows.append(row)
            continue

        sample_weight = core._build_imbalance_weight(
            sub,
            scheme=str(args.weight_scheme),
            floor_station=int(args.weight_floor_station),
            floor_target=int(args.weight_floor_target),
            floor_station_target=int(args.weight_floor_station_target),
            max_weight_cap=float(args.max_weight_cap),
        )
        cluster_ids = _build_cluster_ids(sub)
        fit = _fit_with_cluster_sigma(
            core=core,
            df_sub=sub,
            mode=str(args.fit_mode),
            sample_weight=sample_weight,
            cluster_ids=cluster_ids,
        )
        if fit is None:
            row["reason"] = "fit_failed"
            summary_rows.append(row)
            continue

        group_df, group_summary = _fit_station_target_consistency(
            core=core,
            df=sub,
            fit_mode=str(args.fit_mode),
            sample_weight=sample_weight,
            cluster_ids=cluster_ids,
            min_points_group=int(args.min_points_group),
        )
        if not group_df.empty:
            g = group_df.copy()
            g.insert(0, "subset_id", subset_id)
            group_parts.append(g)

        row.update(
            {
                "fit_ok": True,
                "reason": "",
                "kappa_est": float(fit["kappa_est"]),
                "kappa_sigma_cluster": float(fit["kappa_sigma_cluster"]),
                "kappa_sigma_indep": float(fit["kappa_sigma_indep"]),
                "kappa_sigma_sandwich": float(fit["kappa_sigma_sandwich"]),
                "kappa_sigma_jackknife": float(fit["kappa_sigma_jackknife"]),
                "abs_z_cluster": float(fit["abs_z_cluster"]),
                "status_cluster": str(fit["status_cluster"]),
                "group_n_total": int(group_summary.get("n_groups_total", 0)),
                "group_n_valid": int(group_summary.get("n_groups_valid", 0)),
                "group_chi2_dof_cluster": float(group_summary.get("chi2_dof_cluster", float("nan"))),
                "group_status_cluster": str(group_summary.get("status_cluster", "reject")),
            }
        )
        row["subset_status"] = _combine_status([row["status_cluster"], row["group_status_cluster"]])
        row["subset_internal_status"] = str(row["group_status_cluster"])
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    if summary_df.empty:
        raise RuntimeError("subset summary is empty")

    ref_rows = summary_df[
        (summary_df["subset_id"].astype(str) == "reference_all")
        & summary_df["fit_ok"].astype(bool)
        & np.isfinite(pd.to_numeric(summary_df["kappa_est"], errors="coerce"))
        & np.isfinite(pd.to_numeric(summary_df["kappa_sigma_cluster"], errors="coerce"))
    ].copy()
    if ref_rows.empty:
        ref_kappa = float("nan")
        ref_sigma = float("nan")
    else:
        ref = ref_rows.iloc[0]
        ref_kappa = float(ref["kappa_est"])
        ref_sigma = float(ref["kappa_sigma_cluster"])

    z_ref_vals: List[float] = []
    z_ref_status: List[str] = []
    for rec in summary_df.to_dict(orient="records"):
        kk = float(rec.get("kappa_est", float("nan")))
        ss = float(rec.get("kappa_sigma_cluster", float("nan")))
        if np.isfinite(kk) and np.isfinite(ss) and ss > 0.0 and np.isfinite(ref_kappa) and np.isfinite(ref_sigma) and ref_sigma > 0.0:
            denom = math.sqrt(max(ss * ss + ref_sigma * ref_sigma, 1e-30))
            z = abs((kk - ref_kappa) / denom)
            z_ref_vals.append(float(z))
            z_ref_status.append(core._status_from_abs_z(float(z)))
        else:
            z_ref_vals.append(float("nan"))
            z_ref_status.append("reject")

    summary_df["abs_z_vs_reference"] = z_ref_vals
    summary_df["stability_status_vs_reference"] = z_ref_status
    summary_df = summary_df.sort_values(["subset_id"]).reset_index(drop=True)

    group_df = (
        pd.concat(group_parts, axis=0, ignore_index=True).sort_values(["subset_id", "station_target"]).reset_index(drop=True)
        if group_parts
        else pd.DataFrame()
    )

    max_abs_z_ref = float(np.nanmax(pd.to_numeric(summary_df["abs_z_vs_reference"], errors="coerce").to_numpy(dtype=float)))
    stability_status = core._status_from_abs_z(max_abs_z_ref)
    subset_status_all = _combine_status(summary_df["subset_status"].astype(str).tolist())
    subset_internal_status_all = _combine_status(summary_df["group_status_cluster"].astype(str).tolist())
    overall_status = _combine_status([subset_internal_status_all, stability_status])

    summary_csv = out_dir / "llr_kappa_llr_homogeneous_subset_summary.csv"
    group_csv = out_dir / "llr_kappa_llr_homogeneous_subset_group_consistency.csv"
    metrics_json = out_dir / "llr_kappa_llr_homogeneous_subset_metrics.json"
    plot_pdf = out_dir / "llr_kappa_llr_homogeneous_subset_audit.pdf"
    plot_png = out_dir / "llr_kappa_llr_homogeneous_subset_audit.png"

    summary_df.to_csv(summary_csv, index=False)
    group_df.to_csv(group_csv, index=False)
    _write_plot(summary_df=summary_df, overall_status=overall_status, out_pdf=plot_pdf, out_png=plot_png)

    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase": {"step": "8.7.47.15"},
        "input": {
            "points_csv": _safe_rel(points_csv, ROOT),
            "n_points_all": int(len(df)),
            "fit_mode": str(args.fit_mode),
            "weight_scheme": str(args.weight_scheme),
            "weight_floor_station": int(args.weight_floor_station),
            "weight_floor_target": int(args.weight_floor_target),
            "weight_floor_station_target": int(args.weight_floor_station_target),
            "max_weight_cap": float(args.max_weight_cap),
            "min_points_subset": int(args.min_points_subset),
            "min_points_group": int(args.min_points_group),
            "apol_post_year": int(args.apol_post_year),
            "fullmoon_exclusion_deg": float(args.fullmoon_exclusion_deg),
            "fullmoon_gate_note": "phase model is an approximate synodic mapping used for subset gating only",
        },
        "reference": {
            "subset_id": "reference_all",
            "kappa_est": float(ref_kappa),
            "kappa_sigma_cluster": float(ref_sigma),
        },
        "subset_summary": {
            "n_subsets": int(len(summary_df)),
            "subset_csv": _safe_rel(summary_csv, ROOT),
            "group_csv": _safe_rel(group_csv, ROOT),
            "max_abs_z_vs_reference": float(max_abs_z_ref),
            "subset_status_all": subset_status_all,
            "subset_internal_status_all": subset_internal_status_all,
            "stability_status_vs_reference": stability_status,
            "overall_status": overall_status,
        },
        "gate_status": {
            "subset_fit_and_internal_gate": subset_status_all,
            "subset_internal_gate": subset_internal_status_all,
            "subset_stability_gate": stability_status,
            "overall_status": overall_status,
        },
        "outputs": {
            "metrics_json": _safe_rel(metrics_json, ROOT),
            "plot_pdf": _safe_rel(plot_pdf, ROOT),
            "plot_png": _safe_rel(plot_png, ROOT),
        },
    }
    metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    produced = [summary_csv, group_csv, metrics_json, plot_pdf, plot_png]
    synced = _sync_outputs(paths=produced, private_root=out_dir, public_root=public_dir)
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {group_csv}")
    print(f"Wrote: {metrics_json}")
    print(f"Wrote: {plot_pdf}")
    print(f"Wrote: {plot_png}")
    print(f"Synced: {len(synced)} files")
    print(f"Status: {overall_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
