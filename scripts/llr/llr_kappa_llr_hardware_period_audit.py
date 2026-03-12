#!/usr/bin/env python3
"""
llr_kappa_llr_hardware_period_audit.py

Step 8.7.47.14:
- station_target_year fit を station_target_hardware_period fit へ拡張する。
- station log Date Installed と観測年ギャップから cut year を確定する。
- pre/post 境界不連続の z を監査する。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
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
    spec = importlib.util.spec_from_file_location("llr_kappa_llr_core_hw", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load core module spec: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# 関数: `_parse_station_cut_overrides` の入出力契約と処理意図を定義する。

def _parse_station_cut_overrides(text: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for tok in str(text).split(","):
        t = tok.strip()
        if not t or ":" not in t:
            continue

        st_raw, yy_raw = t.split(":", 1)
        st = st_raw.strip().upper()
        try:
            yy = int(yy_raw.strip())
        except Exception:
            continue

        if st:
            out[st] = yy

    return out


# 関数: `_parse_installed_years_from_log` の入出力契約と処理意図を定義する。

def _parse_installed_years_from_log(path: Path) -> List[int]:
    if not path.exists():
        return []

    txt = path.read_text(encoding="utf-8", errors="ignore")
    out: List[int] = []
    for m in re.finditer(r"Date Installed\s*:\s*([0-9]{4})-[0-9]{2}-[0-9]{2}", txt):
        try:
            out.append(int(m.group(1)))
        except Exception:
            continue

    return sorted(set(out))


# 関数: `_resolve_station_log_path` の入出力契約と処理意図を定義する。

def _resolve_station_log_path(meta_dir: Path, station: str) -> Optional[Path]:
    st = str(station).strip().upper()
    if not st:
        return None

    meta_json = meta_dir / f"{st.lower()}.json"
    if meta_json.exists():
        try:
            meta = json.loads(meta_json.read_text(encoding="utf-8"))
        except Exception:
            meta = {}

        log_filename = str(meta.get("log_filename", "")).strip()
        if log_filename:
            p = meta_dir / log_filename
            if p.exists():
                return p

    hits = sorted(meta_dir.glob(f"{st.lower()}_*.log"))
    if hits:
        return hits[-1]

    return None


# 関数: `_select_cut_year` の入出力契約と処理意図を定義する。

def _select_cut_year(
    years: np.ndarray,
    *,
    station: str,
    installed_years: Sequence[int],
    override_year: Optional[int],
    min_points_cut: int,
    min_gap_years: int,
) -> Tuple[Optional[int], str, pd.DataFrame]:
    yy = pd.Series(years).dropna().astype(int)
    if yy.empty:
        return None, "no_data", pd.DataFrame()

    obs_min = int(yy.min())
    obs_max = int(yy.max())
    rows: List[Dict[str, Any]] = []

    if override_year is not None:
        left = int((yy < int(override_year)).sum())
        right = int((yy >= int(override_year)).sum())
        viable = bool(
            int(override_year) > obs_min
            and int(override_year) <= obs_max
            and left >= int(min_points_cut)
            and right >= int(min_points_cut)
        )
        rows.append(
            {
                "station": station,
                "source": "override",
                "candidate_year": int(override_year),
                "n_left": left,
                "n_right": right,
                "score_balance": min(left, right),
                "gap_years": float("nan"),
                "viable": viable,
            }
        )
        if viable:
            return int(override_year), "override", pd.DataFrame(rows)

    for y in sorted(set(int(v) for v in installed_years)):
        if y <= obs_min or y > obs_max:
            continue

        left = int((yy < y).sum())
        right = int((yy >= y).sum())
        rows.append(
            {
                "station": station,
                "source": "station_log_date_installed",
                "candidate_year": int(y),
                "n_left": left,
                "n_right": right,
                "score_balance": min(left, right),
                "gap_years": float("nan"),
                "viable": bool(left >= int(min_points_cut) and right >= int(min_points_cut)),
            }
        )

    uniq = sorted(set(int(v) for v in yy.tolist()))
    for prev_y, next_y in zip(uniq, uniq[1:]):
        gap = int(next_y - prev_y)
        if gap < int(min_gap_years):
            continue

        cut = int(next_y)
        left = int((yy < cut).sum())
        right = int((yy >= cut).sum())
        rows.append(
            {
                "station": station,
                "source": "observation_year_gap",
                "candidate_year": cut,
                "n_left": left,
                "n_right": right,
                "score_balance": min(left, right),
                "gap_years": gap,
                "viable": bool(left >= int(min_points_cut) and right >= int(min_points_cut)),
            }
        )

    cand_df = pd.DataFrame(rows)
    if cand_df.empty:
        return None, "single_period", cand_df

    viable_df = cand_df[cand_df["viable"].astype(bool)].copy()
    if viable_df.empty:
        return None, "single_period", cand_df

    src_rank = {"station_log_date_installed": 2, "observation_year_gap": 1, "override": 3}
    viable_df["src_rank"] = viable_df["source"].map(src_rank).fillna(0).astype(int)
    viable_df = viable_df.sort_values(
        ["src_rank", "score_balance", "candidate_year"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    top = viable_df.iloc[0]
    return int(top["candidate_year"]), str(top["source"]), cand_df


# 関数: `_build_station_cuts` の入出力契約と処理意図を定義する。

def _build_station_cuts(
    df: pd.DataFrame,
    *,
    meta_dir: Path,
    overrides: Dict[str, int],
    min_points_cut: int,
    min_gap_years: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cut_rows: List[Dict[str, Any]] = []
    cand_parts: List[pd.DataFrame] = []
    stations = sorted(set(df["station"].astype(str).tolist()))
    for st in stations:
        sub = df[df["station"].astype(str) == st].copy()
        years = pd.to_numeric(sub["year"], errors="coerce").to_numpy(dtype=float)
        log_path = _resolve_station_log_path(meta_dir=meta_dir, station=st)
        installed = _parse_installed_years_from_log(log_path) if log_path is not None else []
        cut_year, source, cand_df = _select_cut_year(
            years=years,
            station=st,
            installed_years=installed,
            override_year=overrides.get(st),
            min_points_cut=int(min_points_cut),
            min_gap_years=int(min_gap_years),
        )
        if not cand_df.empty:
            cand_df = cand_df.copy()
            cand_df["selected"] = cand_df["candidate_year"].astype("Int64") == (
                int(cut_year) if cut_year is not None else pd.NA
            )
            cand_df["selected_source"] = source
            cand_df["log_path"] = str(log_path) if log_path is not None else ""
            cand_parts.append(cand_df)

        cut_rows.append(
            {
                "station": st,
                "n_points": int(len(sub)),
                "obs_min_year": int(pd.to_numeric(sub["year"], errors="coerce").dropna().min()),
                "obs_max_year": int(pd.to_numeric(sub["year"], errors="coerce").dropna().max()),
                "station_cut_year": int(cut_year) if cut_year is not None else pd.NA,
                "cut_source": source,
                "log_path": str(log_path) if log_path is not None else "",
                "installed_years": ",".join(str(v) for v in installed),
            }
        )

    cut_df = pd.DataFrame(cut_rows).sort_values(["station"]).reset_index(drop=True)
    if cand_parts:
        cand_df = pd.concat(cand_parts, axis=0, ignore_index=True)
        cand_df = cand_df.sort_values(["station", "source", "candidate_year"]).reset_index(drop=True)
    else:
        cand_df = pd.DataFrame()

    return cut_df, cand_df


# 関数: `_apply_hardware_period_labels` の入出力契約と処理意図を定義する。

def _apply_hardware_period_labels(df: pd.DataFrame, cut_df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cut_map: Dict[str, Optional[int]] = {}
    src_map: Dict[str, str] = {}
    for rec in cut_df.to_dict(orient="records"):
        st = str(rec.get("station", "")).strip()
        yy = rec.get("station_cut_year")
        if pd.notna(yy):
            try:
                cut_map[st] = int(yy)
            except Exception:
                cut_map[st] = None
        else:
            cut_map[st] = None

        src_map[st] = str(rec.get("cut_source", "single_period"))

    hw_phase: List[str] = []
    hw_period: List[str] = []
    cut_source: List[str] = []
    for rec in out.to_dict(orient="records"):
        st = str(rec.get("station", "")).strip()
        yy = int(rec.get("year"))
        cy = cut_map.get(st)
        if cy is None:
            phase = "single"
            period = "single"
        elif yy < int(cy):
            phase = "pre"
            period = f"pre_{int(cy)}"
        else:
            phase = "post"
            period = f"post_{int(cy)}"

        hw_phase.append(phase)
        hw_period.append(period)
        cut_source.append(src_map.get(st, "single_period"))

    out["hardware_phase"] = hw_phase
    out["hardware_period"] = hw_period
    out["cut_source"] = cut_source
    out["station_target_hardware_period"] = (
        out["station"].astype(str) + "|" + out["target"].astype(str) + "|" + out["hardware_period"].astype(str)
    )
    return out


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


# 関数: `_fit_station_target_hardware_period` の入出力契約と処理意図を定義する。

def _fit_station_target_hardware_period(
    core: Any,
    df: pd.DataFrame,
    *,
    fit_mode: str,
    sample_weight: Optional[np.ndarray],
    cluster_ids: np.ndarray,
    min_points_group: int,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    keys = (
        df[["station", "target", "hardware_period", "hardware_phase", "cut_source"]]
        .drop_duplicates()
        .sort_values(["station", "target", "hardware_period"])
    )
    for rec in keys.to_dict(orient="records"):
        st = str(rec["station"])
        tg = str(rec["target"])
        hp = str(rec["hardware_period"])
        phase = str(rec["hardware_phase"])
        source = str(rec["cut_source"])
        mask = (df["station"].astype(str) == st) & (df["target"].astype(str) == tg) & (df["hardware_period"].astype(str) == hp)
        sub = df.loc[mask].copy()
        n = int(len(sub))
        if n < int(min_points_group):
            rows.append({"station": st, "target": tg, "hardware_period": hp, "hardware_phase": phase, "cut_source": source, "n_points": n, "fit_ok": False, "reason": f"n<{int(min_points_group)}"})
            continue

        idx = np.flatnonzero(mask.to_numpy(dtype=bool))
        w_sub = None if sample_weight is None else np.asarray(sample_weight, dtype=float)[idx]
        c_sub = np.asarray(cluster_ids, dtype=object)[mask.to_numpy(dtype=bool)]
        fit = _fit_with_cluster_sigma(core=core, df_sub=sub, mode=fit_mode, sample_weight=w_sub, cluster_ids=c_sub)
        if fit is None:
            rows.append({"station": st, "target": tg, "hardware_period": hp, "hardware_phase": phase, "cut_source": source, "n_points": n, "fit_ok": False, "reason": "fit_failed"})
            continue

        rows.append({"station": st, "target": tg, "hardware_period": hp, "hardware_phase": phase, "cut_source": source, "fit_ok": True, "reason": "", **fit})

    out_df = pd.DataFrame(rows).sort_values(["station", "target", "hardware_period"]).reset_index(drop=True)
    valid = out_df[
        out_df.get("fit_ok", pd.Series(dtype=bool)).astype(bool)
        & np.isfinite(pd.to_numeric(out_df.get("kappa_est"), errors="coerce"))
        & np.isfinite(pd.to_numeric(out_df.get("kappa_sigma_cluster"), errors="coerce"))
        & (pd.to_numeric(out_df.get("kappa_sigma_cluster"), errors="coerce") > 0.0)
    ].copy()
    if valid.empty:
        return out_df, {"fit_mode": fit_mode, "n_groups_total": int(len(out_df)), "n_groups_valid": 0, "chi2_dof_cluster": float("nan"), "status_cluster": "reject"}

    stats = core._weighted_mean_and_chi2(
        values=pd.to_numeric(valid["kappa_est"], errors="coerce").to_numpy(dtype=float),
        sigma=pd.to_numeric(valid["kappa_sigma_cluster"], errors="coerce").to_numpy(dtype=float),
    )
    chi2 = float(stats.get("chi2_dof", float("nan")))
    return out_df, {
        "fit_mode": fit_mode,
        "n_groups_total": int(len(out_df)),
        "n_groups_valid": int(len(valid)),
        "weighted_mean_kappa": float(stats.get("weighted_mean", float("nan"))),
        "weighted_sigma_kappa": float(stats.get("weighted_sigma", float("nan"))),
        "chi2_dof_cluster": chi2,
        "status_cluster": core._consistency_status_from_chi2_dof(chi2),
    }


# 関数: `_compute_boundary_discontinuity` の入出力契約と処理意図を定義する。

def _compute_boundary_discontinuity(core: Any, group_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    valid = group_df[
        group_df.get("fit_ok", pd.Series(dtype=bool)).astype(bool)
        & np.isfinite(pd.to_numeric(group_df.get("kappa_est"), errors="coerce"))
        & np.isfinite(pd.to_numeric(group_df.get("kappa_sigma_cluster"), errors="coerce"))
        & (pd.to_numeric(group_df.get("kappa_sigma_cluster"), errors="coerce") > 0.0)
    ].copy()
    rows: List[Dict[str, Any]] = []
    keys = valid[["station", "target"]].drop_duplicates().sort_values(["station", "target"])
    for rec in keys.to_dict(orient="records"):
        st = str(rec["station"])
        tg = str(rec["target"])
        sub = valid[(valid["station"].astype(str) == st) & (valid["target"].astype(str) == tg)].copy()
        phases = set(sub["hardware_phase"].astype(str).tolist())
        if not ("pre" in phases and "post" in phases):
            continue

        pre = sub[sub["hardware_phase"].astype(str) == "pre"].iloc[0]
        post = sub[sub["hardware_phase"].astype(str) == "post"].iloc[0]
        k_pre = float(pre["kappa_est"])
        s_pre = float(pre["kappa_sigma_cluster"])
        k_post = float(post["kappa_est"])
        s_post = float(post["kappa_sigma_cluster"])
        delta = float(k_post - k_pre)
        s_delta = float(math.sqrt(max((s_pre * s_pre) + (s_post * s_post), 1e-30)))
        z = float(delta / s_delta) if np.isfinite(s_delta) and s_delta > 0.0 else float("nan")
        abs_z = float(abs(z)) if np.isfinite(z) else float("nan")
        rows.append({"station": st, "target": tg, "kappa_pre": k_pre, "sigma_pre": s_pre, "kappa_post": k_post, "sigma_post": s_post, "delta_post_minus_pre": delta, "sigma_delta": s_delta, "z_delta": z, "abs_z_delta": abs_z, "status": core._status_from_abs_z(abs_z)})

    out_df = pd.DataFrame(rows).sort_values(["station", "target"]).reset_index(drop=True)
    if out_df.empty:
        return out_df, {"n_pairs": 0, "max_abs_z_delta": float("nan"), "status": "reject", "note": "pre/post pair unavailable"}

    max_abs_z = float(np.nanmax(pd.to_numeric(out_df["abs_z_delta"], errors="coerce").to_numpy(dtype=float)))
    return out_df, {"n_pairs": int(len(out_df)), "max_abs_z_delta": max_abs_z, "status": core._status_from_abs_z(max_abs_z)}


# 関数: `_write_plot` の入出力契約と処理意図を定義する。

def _write_plot(station_cut_df: pd.DataFrame, group_summary: Dict[str, Any], boundary_summary: Dict[str, Any], overall_status: str, out_pdf: Path, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.8))
    ax0, ax1, ax2 = axes
    n_with_cut = int(station_cut_df["station_cut_year"].notna().sum()) if "station_cut_year" in station_cut_df.columns else 0
    n_single = int(len(station_cut_df) - n_with_cut)
    ax0.bar([0, 1], [n_with_cut, n_single], color=["#1f77b4", "#999999"], width=0.55)
    ax0.set_xticks([0, 1])
    ax0.set_xticklabels(["with_cut", "single"])
    ax0.set_ylabel("n station")
    ax0.set_title("Station Cut Availability")
    ax0.grid(axis="y", alpha=0.25)

    chi2 = float(group_summary.get("chi2_dof_cluster", float("nan")))
    ax1.bar([0], [chi2], color="#d62728", width=0.45)
    ax1.axhline(2.0, color="#2ca02c", linestyle="--", linewidth=1.0)
    ax1.axhline(5.0, color="#ff7f0e", linestyle="--", linewidth=1.0)
    ax1.set_xticks([0])
    ax1.set_xticklabels(["station-target-period"])
    ax1.set_ylabel("chi2/dof")
    ax1.set_title("Hardware-Period Consistency")
    ax1.grid(axis="y", alpha=0.25)

    max_abs_z = float(boundary_summary.get("max_abs_z_delta", float("nan")))
    ax2.bar([0], [max_abs_z], color="#9467bd", width=0.45)
    ax2.axhline(2.0, color="#2ca02c", linestyle="--", linewidth=1.0)
    ax2.axhline(3.0, color="#ff7f0e", linestyle="--", linewidth=1.0)
    ax2.set_xticks([0])
    ax2.set_xticklabels(["max |z(post-pre)|"])
    ax2.set_ylabel("|z|")
    ax2.set_title("Boundary Discontinuity")
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle(f"LLR hardware-period audit: group={group_summary.get('status_cluster', 'reject')}, boundary={boundary_summary.get('status', 'reject')}, overall={overall_status}", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
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
    ap = argparse.ArgumentParser(description="LLR station_target_hardware_period audit (Step 8.7.47.14).")
    ap.add_argument("--points-csv", type=str, default=str(ROOT / "output" / "private" / "llr" / "batch" / "llr_batch_points.csv"))
    ap.add_argument("--out-dir", type=str, default=str(ROOT / "output" / "private" / "llr"))
    ap.add_argument("--public-dir", type=str, default=str(ROOT / "output" / "public" / "llr"))
    ap.add_argument("--core-script", type=str, default=str(ROOT / "scripts" / "llr" / "llr_kappa_llr_direct_fit.py"))
    ap.add_argument("--station-meta-dir", type=str, default=str(ROOT / "data" / "llr" / "stations"))
    ap.add_argument("--fit-mode", type=str, default="station_target_year")
    ap.add_argument("--weight-scheme", type=str, default="inv_station_target")
    ap.add_argument("--weight-floor-station", type=int, default=180)
    ap.add_argument("--weight-floor-target", type=int, default=180)
    ap.add_argument("--weight-floor-station-target", type=int, default=120)
    ap.add_argument("--max-weight-cap", type=float, default=8.0)
    ap.add_argument("--min-points-cut", type=int, default=20)
    ap.add_argument("--min-gap-years", type=int, default=3)
    ap.add_argument("--min-points-group", type=int, default=50)
    ap.add_argument("--station-cut-years", type=str, default="", help="Optional overrides STATION:YYYY,...")
    args = ap.parse_args()

    points_csv = Path(str(args.points_csv))
    out_dir = Path(str(args.out_dir))
    public_dir = Path(str(args.public_dir))
    core_script = Path(str(args.core_script))
    station_meta_dir = Path(str(args.station_meta_dir))
    if not points_csv.is_absolute():
        points_csv = (ROOT / points_csv).resolve()

    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()

    if not public_dir.is_absolute():
        public_dir = (ROOT / public_dir).resolve()

    if not core_script.is_absolute():
        core_script = (ROOT / core_script).resolve()

    if not station_meta_dir.is_absolute():
        station_meta_dir = (ROOT / station_meta_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)
    core = _load_core_module(core_script)
    df = core._read_points(points_csv)
    if df.empty:
        raise RuntimeError(f"no valid rows from {points_csv}")

    cut_overrides = _parse_station_cut_overrides(str(args.station_cut_years))
    station_cut_df, candidate_cut_df = _build_station_cuts(df=df, meta_dir=station_meta_dir, overrides=cut_overrides, min_points_cut=int(args.min_points_cut), min_gap_years=int(args.min_gap_years))
    df_hw = _apply_hardware_period_labels(df=df, cut_df=station_cut_df)
    cluster_ids = _build_cluster_ids(df_hw)
    sample_weight = core._build_imbalance_weight(df_hw, scheme=str(args.weight_scheme), floor_station=int(args.weight_floor_station), floor_target=int(args.weight_floor_target), floor_station_target=int(args.weight_floor_station_target), max_weight_cap=float(args.max_weight_cap))
    group_df, group_summary = _fit_station_target_hardware_period(core=core, df=df_hw, fit_mode=str(args.fit_mode), sample_weight=sample_weight, cluster_ids=cluster_ids, min_points_group=int(args.min_points_group))
    boundary_df, boundary_summary = _compute_boundary_discontinuity(core=core, group_df=group_df)
    overall_status = _combine_status([str(group_summary.get("status_cluster", "reject")), str(boundary_summary.get("status", "reject"))])

    cut_csv = out_dir / "llr_kappa_llr_hardware_period_station_cut_summary.csv"
    cand_csv = out_dir / "llr_kappa_llr_hardware_period_candidate_cuts.csv"
    group_csv = out_dir / "llr_kappa_llr_station_target_hardware_period_summary.csv"
    boundary_csv = out_dir / "llr_kappa_llr_hardware_period_boundary_discontinuity.csv"
    metrics_json = out_dir / "llr_kappa_llr_hardware_period_metrics.json"
    plot_pdf = out_dir / "llr_kappa_llr_hardware_period_audit.pdf"
    plot_png = out_dir / "llr_kappa_llr_hardware_period_audit.png"

    station_cut_df.to_csv(cut_csv, index=False)
    candidate_cut_df.to_csv(cand_csv, index=False)
    group_df.to_csv(group_csv, index=False)
    boundary_df.to_csv(boundary_csv, index=False)
    _write_plot(station_cut_df=station_cut_df, group_summary=group_summary, boundary_summary=boundary_summary, overall_status=overall_status, out_pdf=plot_pdf, out_png=plot_png)

    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase": {"step": "8.7.47.14"},
        "input": {
            "points_csv": _safe_rel(points_csv, ROOT),
            "station_meta_dir": _safe_rel(station_meta_dir, ROOT),
            "n_points": int(len(df_hw)),
            "n_station": int(df_hw["station"].nunique()),
            "n_target": int(df_hw["target"].nunique()),
            "fit_mode": str(args.fit_mode),
            "weight_scheme": str(args.weight_scheme),
            "min_points_cut": int(args.min_points_cut),
            "min_gap_years": int(args.min_gap_years),
            "min_points_group": int(args.min_points_group),
            "station_cut_overrides": {k: int(v) for k, v in cut_overrides.items()},
        },
        "station_cut": {
            "n_station_total": int(len(station_cut_df)),
            "n_station_with_cut": int(station_cut_df["station_cut_year"].notna().sum()),
            "n_station_single_period": int(len(station_cut_df) - int(station_cut_df["station_cut_year"].notna().sum())),
            "station_cut_csv": _safe_rel(cut_csv, ROOT),
            "candidate_cut_csv": _safe_rel(cand_csv, ROOT),
        },
        "hardware_period_fit": {**group_summary, "summary_csv": _safe_rel(group_csv, ROOT)},
        "boundary_discontinuity": {**boundary_summary, "summary_csv": _safe_rel(boundary_csv, ROOT)},
        "gate_status": {
            "station_target_hardware_period_gate": str(group_summary.get("status_cluster", "reject")),
            "period_boundary_discontinuity_gate": str(boundary_summary.get("status", "reject")),
            "overall_status": overall_status,
        },
        "outputs": {"metrics_json": _safe_rel(metrics_json, ROOT), "plot_pdf": _safe_rel(plot_pdf, ROOT), "plot_png": _safe_rel(plot_png, ROOT)},
    }
    metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    produced = [cut_csv, cand_csv, group_csv, boundary_csv, metrics_json, plot_pdf, plot_png]
    synced = _sync_outputs(paths=produced, private_root=out_dir, public_root=public_dir)
    print(f"Wrote: {cut_csv}")
    print(f"Wrote: {cand_csv}")
    print(f"Wrote: {group_csv}")
    print(f"Wrote: {boundary_csv}")
    print(f"Wrote: {metrics_json}")
    print(f"Wrote: {plot_pdf}")
    print(f"Wrote: {plot_png}")
    print(f"Synced: {len(synced)} files")
    print(f"Status: {overall_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
