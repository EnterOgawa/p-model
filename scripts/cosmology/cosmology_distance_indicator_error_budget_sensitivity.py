#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_distance_indicator_error_budget_sensitivity.py

Step 14.2.18/14.2.19（距離指標の誤差予算：感度比較）:
距離指標（SNe Ia / BAO）の“誤差予算”が、SNeサンプル（Pantheon vs Pantheon+）や
共分散の扱い（diag-only vs full covariance）、
さらに Pantheon+ の前提（CID重複の扱い / SH0ES HF限定）でどれだけ動くかを比較する。

注意：
  - ここでの誤差予算は「距離指標の観測不確かさ（目安）」であり、距離指標の定義・校正の再導出とは別問題。
  - Pantheon+ の diag（*_ERR_DIAG）は一次ソース側でも “plotting/visual only” と明記されているため、
    本スクリプトでは full covariance を用いたSEM（bin平均誤差）も併記して感度を示す。

入力（固定）:
  - data/cosmology/pantheon_lcparam_full_long.txt（Pantheon; lcparam）
  - data/cosmology/pantheon_sys_full_long.txt（Pantheon; sys covariance, mag^2）
  - data/cosmology/pantheonplus_sh0es.dat（Pantheon+; distances）
  - data/cosmology/pantheonplus_sh0es_statonly.cov（Pantheon+; STATONLY covariance）
  - data/cosmology/pantheonplus_sh0es_stat_sys.cov（Pantheon+; STAT+SYS covariance）
  - data/cosmology/alcock_paczynski_constraints.json（BOSS DR12; BAO距離プロダクト）
  - data/cosmology/boss_dr12_baofs_consensus_reduced_covariance_cij.json（BOSS DR12; BAO+FS reduced covariance c_ij）
  - data/cosmology/bao_sound_horizon_constraints.json（Planck r_drag）
  - output/private/cosmology/cosmology_distance_indicator_reach_limit_metrics.json（DDR→必要補正 Δε）

出力（固定名）:
  - output/private/cosmology/cosmology_distance_indicator_error_budget_sensitivity.png
  - output/private/cosmology/cosmology_distance_indicator_error_budget_sensitivity_metrics.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.cosmology import cosmology_distance_indicator_error_budget as base  # noqa: E402
from scripts.summary import worklog  # noqa: E402


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
        if not chosen:
            return

        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_pantheonplus_dat(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", engine="python")
    # Normalize for convenience.
    if "zCMB" in df.columns and "zcmb" not in df.columns:
        df = df.rename(columns={"zCMB": "zcmb"})
    return df


def _as_int_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series([0] * int(len(df)))
    s = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    return s.astype(int, copy=False)


def _subset_df_and_cov(
    df: pd.DataFrame,
    *,
    cov_stat: Optional[np.ndarray],
    cov_tot: Optional[np.ndarray],
    idx: np.ndarray,
) -> Tuple[pd.DataFrame, Optional[np.ndarray], Optional[np.ndarray]]:
    idx = np.asarray(idx, dtype=int)
    idx = idx[(idx >= 0) & (idx < int(len(df)))]
    if idx.size == 0:
        return (df.iloc[0:0].copy(), None, None)
    df2 = df.iloc[idx].copy().reset_index(drop=True)
    cov_stat2 = cov_stat[np.ix_(idx, idx)].copy() if cov_stat is not None else None
    cov_tot2 = cov_tot[np.ix_(idx, idx)].copy() if cov_tot is not None else None
    return (df2, cov_stat2, cov_tot2)


def _pantheonplus_select_unique_cid_indices(df: pd.DataFrame, *, prefer_sigma: Optional[np.ndarray]) -> np.ndarray:
    """
    Select one row per CID (to remove duplicates).

    Rule:
      - pick the smallest diagonal sigma (prefer_sigma) within each CID.
      - tie-breaker: smaller IDSURVEY, then earlier row.
    """
    if "CID" not in df.columns:
        return np.arange(int(len(df)), dtype=int)
    cid = df["CID"].astype(str)
    ids = _as_int_series(df, "IDSURVEY").to_numpy(dtype=int, copy=False)
    orig = np.arange(int(len(df)), dtype=int)
    if prefer_sigma is None:
        # fallback: try MU_SH0ES_ERR_DIAG, else all NaN (then uses IDSURVEY/orig)
        sig = pd.to_numeric(df.get("MU_SH0ES_ERR_DIAG"), errors="coerce").to_numpy(dtype=float, copy=False)
    else:
        sig = np.asarray(prefer_sigma, dtype=float)
        if sig.shape != (int(len(df)),):
            sig = np.full(int(len(df)), np.nan, dtype=float)

    tmp = pd.DataFrame({"cid": cid, "sigma": sig, "idsurvey": ids, "orig": orig})
    tmp = tmp.sort_values(["cid", "sigma", "idsurvey", "orig"], kind="mergesort")
    picked = tmp.drop_duplicates("cid", keep="first")["orig"].to_numpy(dtype=int)
    picked.sort()
    return picked


def _pantheonplus_select_sh0es_hf_indices(df: pd.DataFrame) -> np.ndarray:
    """
    Select rows used in SH0ES 2021 Hubble Flow dataset (USED_IN_SH0ES_HF == 1).
    """
    used = _as_int_series(df, "USED_IN_SH0ES_HF").to_numpy(dtype=int, copy=False)
    return np.where(used == 1)[0].astype(int)


def _load_cov_txt(path: Path) -> np.ndarray:
    """
    Pantheon+ covariance format:
      - first line: N
      - then N*N floats (sequential)
    """
    if not path.exists():
        raise FileNotFoundError(str(path))
    first = path.read_text(encoding="utf-8", errors="replace").splitlines()[0].strip()
    try:
        n = int(first)
    except Exception as e:
        raise ValueError(f"Invalid first line for covariance (expected N int): {first!r}") from e

    data = np.loadtxt(path, dtype=float, skiprows=1)
    if data.size != int(n) * int(n):
        raise ValueError(f"covariance size mismatch: got {data.size}, expected {n*n}")
    return data.reshape((n, n))


def _compute_sn_binned_budget_pantheonplus(
    df: pd.DataFrame,
    *,
    z_max: float,
    bin_width: float,
    min_points: int,
    cov_statonly: Optional[np.ndarray],
    cov_total: Optional[np.ndarray],
) -> List[Dict[str, Any]]:
    if "zcmb" not in df.columns:
        raise ValueError("Pantheon+ .dat must contain column: zCMB (or normalized 'zcmb')")

    n_rows = int(len(df))
    if cov_statonly is not None and cov_statonly.shape != (n_rows, n_rows):
        raise ValueError(f"Pantheon+ STATONLY covariance shape mismatch: {cov_statonly.shape} vs n={n_rows}")
    if cov_total is not None and cov_total.shape != (n_rows, n_rows):
        raise ValueError(f"Pantheon+ STAT+SYS covariance shape mismatch: {cov_total.shape} vs n={n_rows}")

    diag_stat = np.sqrt(np.maximum(0.0, np.diag(cov_statonly))) if cov_statonly is not None else None
    diag_total = np.sqrt(np.maximum(0.0, np.diag(cov_total))) if cov_total is not None else None

    bins = np.arange(0.0, float(z_max) + float(bin_width) + 1e-9, float(bin_width))
    out: List[Dict[str, Any]] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sub0 = df[(df["zcmb"] >= lo) & (df["zcmb"] < hi)]
        n0 = int(len(sub0))
        z_center = 0.5 * (float(lo) + float(hi))

        idx0 = sub0.index.to_numpy(dtype=int)
        # Valid points: need diagonal sigma.
        sig0 = None
        if diag_stat is not None:
            sig0 = diag_stat[idx0]
            ok = np.isfinite(sig0) & (sig0 > 0.0)
        else:
            ok = np.zeros_like(idx0, dtype=bool)
        idx = idx0[ok]
        n = int(idx.size)

        if n < int(min_points):
            out.append(
                {
                    "z_center": z_center,
                    "z_lo": float(lo),
                    "z_hi": float(hi),
                    "n": n0,
                    "n_valid": n,
                    # stat-only (diag proxy)
                    "sigma_median_stat": None,
                    "sigma_sem_stat_diag": None,
                    "sigma_sem_stat_cov": None,
                    # total (stat+sys)
                    "sigma_median_total": None,
                    "sigma_sem_total_cov": None,
                }
            )
            continue

        sig = diag_stat[idx] if diag_stat is not None else np.full(n, np.nan, dtype=float)
        sigma_median_stat = float(np.nanmedian(sig))
        w = 1.0 / (sig * sig)
        sigma_sem_stat_diag = float(math.sqrt(1.0 / float(w.sum())))

        sigma_sem_stat_cov: Optional[float] = None
        if cov_statonly is not None:
            cov_bin = cov_statonly[np.ix_(idx, idx)].astype(float, copy=False)
            sigma_sem_stat_cov = base._cov_sem_weighted(cov_bin)

        sigma_median_total: Optional[float] = None
        sigma_sem_total_cov: Optional[float] = None
        if cov_total is not None:
            try:
                cov_bin_tot = cov_total[np.ix_(idx, idx)].astype(float, copy=False)
                sigma_sem_total_cov = base._cov_sem_weighted(cov_bin_tot)
                if diag_total is not None:
                    sigma_median_total = float(np.nanmedian(diag_total[idx]))
            except Exception:
                sigma_sem_total_cov = None
                sigma_median_total = None

        out.append(
            {
                "z_center": z_center,
                "z_lo": float(lo),
                "z_hi": float(hi),
                "n": n0,
                "n_valid": n,
                "sigma_median_stat": sigma_median_stat,
                "sigma_sem_stat_diag": sigma_sem_stat_diag,
                "sigma_sem_stat_cov": sigma_sem_stat_cov,
                "sigma_median_total": sigma_median_total,
                "sigma_sem_total_cov": sigma_sem_total_cov,
            }
        )
    return out


def _compute_reach_limits_for_sn_bins(
    *,
    sn_bins: List[Dict[str, Any]],
    sn_sem_key: str,
    sn_sem_total_key: str,
    sn_med_key: str,
    sn_med_total_key: str,
    bao_points: List[Dict[str, Any]],
    boss_baofs_cov: Optional[Dict[str, Any]],
    r_drag_budget: Optional[Dict[str, Any]],
    reach_reps: Dict[str, Any],
    z_max: float,
    sigma_multipliers: Sequence[float],
) -> Dict[str, Any]:
    # Prepare SN arrays.
    sn_z = np.array([float(r["z_center"]) for r in sn_bins], dtype=float)
    sn_sem = np.array(
        [float(r[sn_sem_key]) if r.get(sn_sem_key) is not None else float("nan") for r in sn_bins],
        dtype=float,
    )
    sn_sem_total = np.array(
        [float(r[sn_sem_total_key]) if r.get(sn_sem_total_key) is not None else float("nan") for r in sn_bins],
        dtype=float,
    )
    sn_med = np.array(
        [float(r[sn_med_key]) if r.get(sn_med_key) is not None else float("nan") for r in sn_bins],
        dtype=float,
    )
    sn_med_total = np.array(
        [float(r[sn_med_total_key]) if r.get(sn_med_total_key) is not None else float("nan") for r in sn_bins],
        dtype=float,
    )

    # BAO arrays (mu-equivalent).
    bao_z = np.array([float(r["z_eff"]) for r in bao_points], dtype=float)
    bao_mu = np.array([float(r["mu_sigma_mag_equiv"]) for r in bao_points], dtype=float)
    bao_dm = np.array([float(r["DM_scaled_mpc"]) for r in bao_points], dtype=float)

    z_grid = np.linspace(0.0, float(z_max), 1200)
    sn_sem_grid = base._interp_on_grid(sn_z, sn_sem, z_grid)
    sn_sem_total_grid = base._interp_on_grid(sn_z, sn_sem_total, z_grid)
    sn_med_grid = base._interp_on_grid(sn_z, sn_med, z_grid)
    sn_med_total_grid = base._interp_on_grid(sn_z, sn_med_total, z_grid)

    bao_mu_grid_naive = base._interp_on_grid(bao_z, bao_mu, z_grid) if bao_z.size else np.zeros_like(z_grid)
    bao_mu_grid_cov = np.array(bao_mu_grid_naive, copy=True)
    rdrag_mu = float(r_drag_budget["mu_sigma_mag_equiv"]) if isinstance(r_drag_budget, dict) else 0.0

    boss_dm_cov = None
    boss_dm_corr = None
    boss_dm_sigmas = None
    boss_dm_match = []
    if boss_baofs_cov is not None and bao_z.size:
        try:
            boss_params = list(boss_baofs_cov["parameters"])
            cij_1e4 = np.array(boss_baofs_cov["cij_1e4"], dtype=float)
            dm_idx, dm_params = base._match_boss_dm_indices(boss_params, bao_points)
            boss_dm_match = [
                {"z_eff": float(p.get("z_eff")), "id": str(p.get("id") or ""), "sigma": float(p.get("sigma"))}
                for p in dm_params
            ]
            dm_corr = (cij_1e4[np.ix_(dm_idx, dm_idx)] / 10000.0).astype(float)
            dm_sigma = np.array([float(p.get("sigma")) for p in dm_params], dtype=float)
            dm_cov = dm_corr * np.outer(dm_sigma, dm_sigma)
            dm_cov = 0.5 * (dm_cov + dm_cov.T)

            boss_dm_cov = dm_cov
            boss_dm_corr = dm_corr
            boss_dm_sigmas = dm_sigma
            bao_mu_grid_cov = base._bao_mu_sigma_mag_from_dm_cov_interpolated(
                z_grid=z_grid,
                z_points=bao_z,
                dm_points=bao_dm,
                dm_cov=dm_cov,
            )
        except Exception:
            boss_dm_cov = None
            boss_dm_corr = None
            boss_dm_sigmas = None
            bao_mu_grid_cov = np.array(bao_mu_grid_naive, copy=True)

    # Budgets
    budget_opt = np.sqrt(np.maximum(0.0, sn_sem_grid) ** 2 + np.maximum(0.0, bao_mu_grid_naive) ** 2 + rdrag_mu**2)
    budget_cons = np.sqrt(np.maximum(0.0, sn_med_grid) ** 2 + np.maximum(0.0, bao_mu_grid_naive) ** 2 + rdrag_mu**2)
    budget_opt_total = np.sqrt(
        np.maximum(0.0, sn_sem_total_grid) ** 2 + np.maximum(0.0, bao_mu_grid_cov) ** 2 + rdrag_mu**2
    )
    budget_cons_total = np.sqrt(
        np.maximum(0.0, sn_med_total_grid) ** 2 + np.maximum(0.0, bao_mu_grid_cov) ** 2 + rdrag_mu**2
    )

    reps = reach_reps.get("reach") or {}
    rep_bao = reps.get("bao") or {}
    rep_no = reps.get("no_bao") or {}

    dm_req_bao = (
        base._required_delta_mu_mag(float(rep_bao.get("delta_eps_needed", float("nan"))), z_grid) if rep_bao else None
    )
    dm_req_no = (
        base._required_delta_mu_mag(float(rep_no.get("delta_eps_needed", float("nan"))), z_grid) if rep_no else None
    )

    limits: Dict[str, Any] = {"opt": {}, "cons": {}}
    limits_total: Dict[str, Any] = {"opt": {}, "cons": {}}
    for label, dm_req in [("bao", dm_req_bao), ("no_bao", dm_req_no)]:
        if dm_req is None:
            continue
        limits["opt"][label] = {}
        limits["cons"][label] = {}
        limits_total["opt"][label] = {}
        limits_total["cons"][label] = {}
        for k, bud, bud_total in [
            ("budget_opt", budget_opt, budget_opt_total),
            ("budget_cons", budget_cons, budget_cons_total),
        ]:
            for m in sigma_multipliers:
                z_lim = base._z_limit(dm_req, bud, z_grid, sigma_multiplier=float(m))
                z_lim_total = base._z_limit(dm_req, bud_total, z_grid, sigma_multiplier=float(m))
                tgt = limits["opt"][label] if k == "budget_opt" else limits["cons"][label]
                tgt[f"{m}sigma"] = z_lim
                tgt2 = limits_total["opt"][label] if k == "budget_opt" else limits_total["cons"][label]
                tgt2[f"{m}sigma"] = z_lim_total

    return {
        "reach_limits": limits,
        "reach_limits_total": limits_total,
        "bao_covariance": {
            "used": boss_dm_cov is not None,
            "dm_cov": boss_dm_cov.tolist() if boss_dm_cov is not None else None,
            "dm_corr": boss_dm_corr.tolist() if boss_dm_corr is not None else None,
            "dm_sigmas": boss_dm_sigmas.tolist() if boss_dm_sigmas is not None else None,
            "dm_match": boss_dm_match,
        },
        "z_refs": [0.1, 0.5, 1.0, 2.0],
        "budgets_at_z": [
            {
                "z": float(z0),
                "sn_sem_mag": float(np.interp(z0, z_grid, sn_sem_grid)),
                "sn_sem_total_mag": float(np.interp(z0, z_grid, sn_sem_total_grid)),
                "sn_median_mag": float(np.interp(z0, z_grid, sn_med_grid)),
                "sn_median_total_mag": float(np.interp(z0, z_grid, sn_med_total_grid)),
                "bao_mu_mag_naive": float(np.interp(z0, z_grid, bao_mu_grid_naive)),
                "bao_mu_mag_cov": float(np.interp(z0, z_grid, bao_mu_grid_cov)),
                "r_drag_mu_mag": float(rdrag_mu),
                "budget_opt_mag": float(np.interp(z0, z_grid, budget_opt)),
                "budget_opt_total_mag": float(np.interp(z0, z_grid, budget_opt_total)),
                "budget_cons_mag": float(np.interp(z0, z_grid, budget_cons)),
                "budget_cons_total_mag": float(np.interp(z0, z_grid, budget_cons_total)),
            }
            for z0 in [0.1, 0.5, 1.0, 2.0]
        ],
    }


def _plot_sensitivity(
    *,
    out_png: Path,
    reach_reps: Dict[str, Any],
    scenarios: List[Dict[str, Any]],
    sigma_multipliers: Sequence[float],
) -> None:
    _set_japanese_font()
    import matplotlib.pyplot as plt

    reps = reach_reps.get("reach") or {}
    rep_bao = reps.get("bao") or {}
    rep_no = reps.get("no_bao") or {}

    # Prepare values to plot (opt budget).
    sigs = list(sigma_multipliers)
    labels = [str(s.get("label") or s.get("id") or "") for s in scenarios]
    x = np.arange(len(labels), dtype=float)

    def _pick_limits(s: Dict[str, Any], which: str, m: float) -> Tuple[Optional[float], Optional[float]]:
        lim = (((s.get("reach_limits") or {}).get("opt") or {}).get(which) or {}).get(f"{m}sigma")
        lim_tot = (((s.get("reach_limits_total") or {}).get("opt") or {}).get(which) or {}).get(f"{m}sigma")
        return (lim, lim_tot)

    colors = {sigs[0]: "#1f77b4", sigs[-1]: "#ff7f0e"}
    marker = {False: "o", True: "s"}  # False=stat, True=total

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.2))

    for ax, which, title, rep in [
        (ax1, "bao", "BAOを含む代表制約（tightest）", rep_bao),
        (ax2, "no_bao", "BAOなし代表制約（least rejecting）", rep_no),
    ]:
        for m in sigs:
            c = colors.get(m, "#333333")
            # stat-only
            y_stat = [(_pick_limits(s, which, m)[0]) for s in scenarios]
            # total (stat+sys)
            y_tot = [(_pick_limits(s, which, m)[1]) for s in scenarios]
            # Convert None to NaN for plotting.
            ys = np.array([float(v) if v is not None else float("nan") for v in y_stat], dtype=float)
            yt = np.array([float(v) if v is not None else float("nan") for v in y_tot], dtype=float)

            ax.plot(
                x,
                ys,
                color=c,
                marker=marker[False],
                linewidth=1.4,
                alpha=0.9,
                label=f"{m}σ（stat-only）" if ax is ax1 else None,
            )
            ax.plot(
                x,
                yt,
                color=c,
                marker=marker[True],
                linewidth=1.4,
                alpha=0.9,
                linestyle="--",
                label=f"{m}σ（stat+sys）" if ax is ax1 else None,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=10)
        ax.set_ylabel("z_limit（|Δμ_required| ≤ m×budget）", fontsize=11)
        ax.set_title(f"誤差予算の感度：{title}", fontsize=12)
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        # Keep a stable y-range for comparability; expand if needed.
        ymax = 0.35
        try:
            y_all = []
            for ss in scenarios:
                for mm in sigs:
                    a, b = _pick_limits(ss, which, mm)
                    if a is not None and math.isfinite(float(a)):
                        y_all.append(float(a))
                    if b is not None and math.isfinite(float(b)):
                        y_all.append(float(b))
            if y_all:
                ymax = max(ymax, float(np.nanmax(y_all)) * 1.25)
        except Exception:
            pass
        ax.set_ylim(0.0, ymax)

        if rep:
            eps = rep.get("epsilon0_obs")
            sig = rep.get("epsilon0_sigma")
            short = str(rep.get("short_label") or "")
            ax.text(
                0.02,
                0.98,
                f"ε0={eps}±{sig} / {short}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=10,
                color="#333333",
            )

    handles, labels_leg = ax1.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels_leg, loc="lower center", ncol=4, fontsize=10, frameon=False)
    fig.suptitle("距離指標の誤差予算：SNeサンプル/共分散の扱いによる z_limit の変動（SEMベース）", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.06, 1.0, 0.95))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def _plot_scan_heatmap(
    *,
    out_png: Path,
    bin_widths: Sequence[float],
    min_points_list: Sequence[int],
    values: np.ndarray,
    title: str,
) -> None:
    _set_japanese_font()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(10.5, 6.2))
    data = np.array(values, dtype=float)
    # Mask invalid.
    msk = ~np.isfinite(data)
    vmin = float(np.nanmin(data)) if np.isfinite(data).any() else 0.0
    vmax = float(np.nanmax(data)) if np.isfinite(data).any() else 1.0
    if vmax <= vmin:
        vmax = vmin + 1e-9

    cmap = plt.cm.viridis
    cmap = cmap.copy()
    cmap.set_bad(color="#dddddd")

    img = ax.imshow(np.ma.array(data, mask=msk), origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(bin_widths)))
    ax.set_xticklabels([str(bw) for bw in bin_widths], fontsize=10)
    ax.set_yticks(np.arange(len(min_points_list)))
    ax.set_yticklabels([str(mp) for mp in min_points_list], fontsize=10)
    ax.set_xlabel("bin_width（zビン幅）", fontsize=11)
    ax.set_ylabel("sn_min_points（bin最低点数）", fontsize=11)
    ax.set_title(title, fontsize=12)

    for iy in range(data.shape[0]):
        for ix in range(data.shape[1]):
            v = data[iy, ix]
            if not math.isfinite(float(v)):
                continue
            ax.text(ix, iy, f"{v:.3f}", ha="center", va="center", fontsize=9, color="white")

    cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("z_limit", fontsize=11)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=170)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--z-max", type=float, default=2.3)
    p.add_argument("--bin-width", type=float, default=0.1)
    p.add_argument("--sn-min-points", type=int, default=15)
    p.add_argument("--sigma-multipliers", type=str, default="1,3")
    args = p.parse_args(list(argv) if argv is not None else None)

    z_max = float(args.z_max)
    bin_width = float(args.bin_width)
    sn_min_points = int(args.sn_min_points)
    sigma_multipliers = [float(x) for x in str(args.sigma_multipliers).split(",") if str(x).strip()]
    if not sigma_multipliers:
        sigma_multipliers = [1.0, 3.0]

    out_dir = _ROOT / "output" / "private" / "cosmology"
    out_png = out_dir / "cosmology_distance_indicator_error_budget_sensitivity.png"
    out_json = out_dir / "cosmology_distance_indicator_error_budget_sensitivity_metrics.json"

    pantheon_path = _ROOT / "data" / "cosmology" / "pantheon_lcparam_full_long.txt"
    pantheon_sys_path = _ROOT / "data" / "cosmology" / "pantheon_sys_full_long.txt"
    pplus_dat_path = _ROOT / "data" / "cosmology" / "pantheonplus_sh0es.dat"
    pplus_cov_stat_path = _ROOT / "data" / "cosmology" / "pantheonplus_sh0es_statonly.cov"
    pplus_cov_tot_path = _ROOT / "data" / "cosmology" / "pantheonplus_sh0es_stat_sys.cov"
    boss_path = _ROOT / "data" / "cosmology" / "alcock_paczynski_constraints.json"
    boss_cov_path = _ROOT / "data" / "cosmology" / "boss_dr12_baofs_consensus_reduced_covariance_cij.json"
    rdrag_path = _ROOT / "data" / "cosmology" / "bao_sound_horizon_constraints.json"
    reach_path = _ROOT / "output" / "private" / "cosmology" / "cosmology_distance_indicator_reach_limit_metrics.json"

    reach_reps = _read_json(reach_path) if reach_path.exists() else {"reach": {}}

    # BAO / r_drag inputs.
    bao_points = base._bao_mu_sigma_mag_from_boss_ap(boss_path)
    r_drag_budget = base._r_drag_mu_sigma_mag(rdrag_path)
    boss_cov = base._load_boss_baofs_reduced_cov_cij(boss_cov_path) if boss_cov_path.exists() else None

    # Scenario 1: Pantheon (existing method).
    pantheon_df = base._load_pantheon_lcparam(pantheon_path)
    pantheon_sys = base._load_pantheon_sys_cov(pantheon_sys_path) if pantheon_sys_path.exists() else None
    pantheon_bins = base._compute_sn_binned_budget(
        pantheon_df,
        z_max=z_max,
        bin_width=bin_width,
        min_points=sn_min_points,
        sys_cov=pantheon_sys,
    )
    pantheon_metrics = _compute_reach_limits_for_sn_bins(
        sn_bins=pantheon_bins,
        sn_sem_key="dmb_sem_weighted",
        sn_sem_total_key="dmb_sem_total_cov",
        sn_med_key="dmb_median",
        sn_med_total_key="dmb_median_total",
        bao_points=bao_points,
        boss_baofs_cov=boss_cov,
        r_drag_budget=r_drag_budget,
        reach_reps=reach_reps,
        z_max=z_max,
        sigma_multipliers=sigma_multipliers,
    )

    # Pantheon+ base inputs (full sample).
    pplus_df = _load_pantheonplus_dat(pplus_dat_path)
    cov_stat = _load_cov_txt(pplus_cov_stat_path) if pplus_cov_stat_path.exists() else None
    cov_tot = _load_cov_txt(pplus_cov_tot_path) if pplus_cov_tot_path.exists() else None
    diag_stat = np.sqrt(np.maximum(0.0, np.diag(cov_stat))) if cov_stat is not None else None

    # Scenario group A: Pantheon+ full (diag-only vs cov-aware).
    pplus_bins_full = _compute_sn_binned_budget_pantheonplus(
        pplus_df,
        z_max=z_max,
        bin_width=bin_width,
        min_points=sn_min_points,
        cov_statonly=cov_stat,
        cov_total=cov_tot,
    )

    # diag-only: SEM uses diagonal proxy (like Pantheon stat-only). Total uses full covariance.
    pplus_full_diag_metrics = _compute_reach_limits_for_sn_bins(
        sn_bins=pplus_bins_full,
        sn_sem_key="sigma_sem_stat_diag",
        sn_sem_total_key="sigma_sem_total_cov",
        sn_med_key="sigma_median_stat",
        sn_med_total_key="sigma_median_total",
        bao_points=bao_points,
        boss_baofs_cov=boss_cov,
        r_drag_budget=r_drag_budget,
        reach_reps=reach_reps,
        z_max=z_max,
        sigma_multipliers=sigma_multipliers,
    )

    # cov-aware: SEM uses statonly covariance (intra-bin correlation) for the lower bound too.
    # If statonly covariance was not available, it falls back to diag-only (same as above).
    sem_stat_cov_key_full = (
        "sigma_sem_stat_cov" if any(r.get("sigma_sem_stat_cov") is not None for r in pplus_bins_full) else "sigma_sem_stat_diag"
    )
    pplus_full_cov_metrics = _compute_reach_limits_for_sn_bins(
        sn_bins=pplus_bins_full,
        sn_sem_key=sem_stat_cov_key_full,
        sn_sem_total_key="sigma_sem_total_cov",
        sn_med_key="sigma_median_stat",
        sn_med_total_key="sigma_median_total",
        bao_points=bao_points,
        boss_baofs_cov=boss_cov,
        r_drag_budget=r_drag_budget,
        reach_reps=reach_reps,
        z_max=z_max,
        sigma_multipliers=sigma_multipliers,
    )

    # Scenario group B: Pantheon+ subset (CID unique / SH0ES HF).
    idx_unique = _pantheonplus_select_unique_cid_indices(pplus_df, prefer_sigma=diag_stat)
    df_unique, cov_stat_unique, cov_tot_unique = _subset_df_and_cov(pplus_df, cov_stat=cov_stat, cov_tot=cov_tot, idx=idx_unique)
    bins_unique = _compute_sn_binned_budget_pantheonplus(
        df_unique,
        z_max=z_max,
        bin_width=bin_width,
        min_points=sn_min_points,
        cov_statonly=cov_stat_unique,
        cov_total=cov_tot_unique,
    )
    sem_stat_cov_key_unique = (
        "sigma_sem_stat_cov" if any(r.get("sigma_sem_stat_cov") is not None for r in bins_unique) else "sigma_sem_stat_diag"
    )
    metrics_unique_cov = _compute_reach_limits_for_sn_bins(
        sn_bins=bins_unique,
        sn_sem_key=sem_stat_cov_key_unique,
        sn_sem_total_key="sigma_sem_total_cov",
        sn_med_key="sigma_median_stat",
        sn_med_total_key="sigma_median_total",
        bao_points=bao_points,
        boss_baofs_cov=boss_cov,
        r_drag_budget=r_drag_budget,
        reach_reps=reach_reps,
        z_max=z_max,
        sigma_multipliers=sigma_multipliers,
    )

    idx_hf = _pantheonplus_select_sh0es_hf_indices(pplus_df)
    df_hf, cov_stat_hf, cov_tot_hf = _subset_df_and_cov(pplus_df, cov_stat=cov_stat, cov_tot=cov_tot, idx=idx_hf)
    bins_hf = _compute_sn_binned_budget_pantheonplus(
        df_hf,
        z_max=z_max,
        bin_width=bin_width,
        min_points=sn_min_points,
        cov_statonly=cov_stat_hf,
        cov_total=cov_tot_hf,
    )
    sem_stat_cov_key_hf = "sigma_sem_stat_cov" if any(r.get("sigma_sem_stat_cov") is not None for r in bins_hf) else "sigma_sem_stat_diag"
    metrics_hf_cov = _compute_reach_limits_for_sn_bins(
        sn_bins=bins_hf,
        sn_sem_key=sem_stat_cov_key_hf,
        sn_sem_total_key="sigma_sem_total_cov",
        sn_med_key="sigma_median_stat",
        sn_med_total_key="sigma_median_total",
        bao_points=bao_points,
        boss_baofs_cov=boss_cov,
        r_drag_budget=r_drag_budget,
        reach_reps=reach_reps,
        z_max=z_max,
        sigma_multipliers=sigma_multipliers,
    )

    # HF + unique CID
    diag_stat_hf = np.sqrt(np.maximum(0.0, np.diag(cov_stat_hf))) if cov_stat_hf is not None else None
    idx_hf_unique_local = _pantheonplus_select_unique_cid_indices(df_hf, prefer_sigma=diag_stat_hf)
    df_hf_u, cov_stat_hf_u, cov_tot_hf_u = _subset_df_and_cov(df_hf, cov_stat=cov_stat_hf, cov_tot=cov_tot_hf, idx=idx_hf_unique_local)
    bins_hf_u = _compute_sn_binned_budget_pantheonplus(
        df_hf_u,
        z_max=z_max,
        bin_width=bin_width,
        min_points=sn_min_points,
        cov_statonly=cov_stat_hf_u,
        cov_total=cov_tot_hf_u,
    )
    sem_stat_cov_key_hf_u = (
        "sigma_sem_stat_cov" if any(r.get("sigma_sem_stat_cov") is not None for r in bins_hf_u) else "sigma_sem_stat_diag"
    )
    metrics_hf_u_cov = _compute_reach_limits_for_sn_bins(
        sn_bins=bins_hf_u,
        sn_sem_key=sem_stat_cov_key_hf_u,
        sn_sem_total_key="sigma_sem_total_cov",
        sn_med_key="sigma_median_stat",
        sn_med_total_key="sigma_median_total",
        bao_points=bao_points,
        boss_baofs_cov=boss_cov,
        r_drag_budget=r_drag_budget,
        reach_reps=reach_reps,
        z_max=z_max,
        sigma_multipliers=sigma_multipliers,
    )

    scenarios_plot = [
        {
            "id": "pantheon",
            "label": "Pantheon",
            **pantheon_metrics,
        },
        {
            "id": "pantheonplus_diag",
            "label": "Pantheon+（full/diag）",
            **pplus_full_diag_metrics,
        },
        {
            "id": "pantheonplus_cov",
            "label": "Pantheon+（full/cov）",
            **pplus_full_cov_metrics,
        },
        {
            "id": "pantheonplus_unique_cid",
            "label": "Pantheon+（CID重複除去/cov）",
            **metrics_unique_cov,
        },
        {
            "id": "pantheonplus_sh0es_hf",
            "label": "Pantheon+（SH0ES HF/cov）",
            **metrics_hf_cov,
        },
        {
            "id": "pantheonplus_sh0es_hf_unique_cid",
            "label": "Pantheon+（HF+CID/cov）",
            **metrics_hf_u_cov,
        },
    ]
    _plot_sensitivity(out_png=out_png, reach_reps=reach_reps, scenarios=scenarios_plot, sigma_multipliers=sigma_multipliers)

    # Envelope (upper bound across scenarios) for quick reporting.
    envelope: Dict[str, Any] = {"opt": {}, "cons": {}, "opt_total": {}, "cons_total": {}}
    for which in ["bao", "no_bao"]:
        envelope["opt"][which] = {}
        envelope["cons"][which] = {}
        envelope["opt_total"][which] = {}
        envelope["cons_total"][which] = {}
        for m in sigma_multipliers:
            key = f"{m}sigma"
            vals_opt = [
                ((s.get("reach_limits") or {}).get("opt") or {}).get(which, {}).get(key)
                for s in scenarios_plot
            ]
            vals_cons = [
                ((s.get("reach_limits") or {}).get("cons") or {}).get(which, {}).get(key)
                for s in scenarios_plot
            ]
            vals_opt_t = [
                ((s.get("reach_limits_total") or {}).get("opt") or {}).get(which, {}).get(key)
                for s in scenarios_plot
            ]
            vals_cons_t = [
                ((s.get("reach_limits_total") or {}).get("cons") or {}).get(which, {}).get(key)
                for s in scenarios_plot
            ]

            def _max_or_none(vals: List[Optional[float]]) -> Optional[float]:
                xs = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
                return max(xs) if xs else None

            envelope["opt"][which][key] = _max_or_none(vals_opt)
            envelope["cons"][which][key] = _max_or_none(vals_cons)
            envelope["opt_total"][which][key] = _max_or_none(vals_opt_t)
            envelope["cons_total"][which][key] = _max_or_none(vals_cons_t)

    # Step 14.2.20: scan (HF subset) for plausible upper bounds from binning choices.
    scan_bin_widths = [0.05, 0.1, 0.2]
    scan_min_points = [8, 10, 15, 20]
    scan_values = np.full((len(scan_min_points), len(scan_bin_widths)), np.nan, dtype=float)
    scan_rows: List[Dict[str, Any]] = []
    for iy, mp in enumerate(scan_min_points):
        for ix, bw in enumerate(scan_bin_widths):
            try:
                bins_scan = _compute_sn_binned_budget_pantheonplus(
                    df_hf,
                    z_max=z_max,
                    bin_width=float(bw),
                    min_points=int(mp),
                    cov_statonly=cov_stat_hf,
                    cov_total=cov_tot_hf,
                )
                sem_key_scan = (
                    "sigma_sem_stat_cov"
                    if any(r.get("sigma_sem_stat_cov") is not None for r in bins_scan)
                    else "sigma_sem_stat_diag"
                )
                met_scan = _compute_reach_limits_for_sn_bins(
                    sn_bins=bins_scan,
                    sn_sem_key=sem_key_scan,
                    sn_sem_total_key="sigma_sem_total_cov",
                    sn_med_key="sigma_median_stat",
                    sn_med_total_key="sigma_median_total",
                    bao_points=bao_points,
                    boss_baofs_cov=boss_cov,
                    r_drag_budget=r_drag_budget,
                    reach_reps=reach_reps,
                    z_max=z_max,
                    sigma_multipliers=sigma_multipliers,
                )
                # focus metric: no-BAO representative, opt_total, 3σ (most relevant for “reach” upper bound).
                v = (
                    (((met_scan.get("reach_limits_total") or {}).get("opt") or {}).get("no_bao") or {}).get("3.0sigma")
                )
                scan_values[iy, ix] = float(v) if v is not None and math.isfinite(float(v)) else float("nan")
                scan_rows.append(
                    {
                        "bin_width": float(bw),
                        "sn_min_points": int(mp),
                        "sem_stat_key": sem_key_scan,
                        "no_bao_opt_total_3sigma": (float(v) if v is not None and math.isfinite(float(v)) else None),
                        "reach_limits_total_opt": ((met_scan.get("reach_limits_total") or {}).get("opt") or {}),
                    }
                )
            except Exception:
                scan_values[iy, ix] = float("nan")

    out_scan_png = out_dir / "cosmology_distance_indicator_error_budget_sensitivity_scan.png"
    _plot_scan_heatmap(
        out_png=out_scan_png,
        bin_widths=scan_bin_widths,
        min_points_list=scan_min_points,
        values=scan_values,
        title="誤差予算の上限スキャン（Pantheon+ SH0ES HF；no-BAO代表；opt_total 3σ）",
    )

    scan_best = None
    try:
        flat = scan_values[np.isfinite(scan_values)]
        if flat.size:
            best_val = float(np.nanmax(flat))
            best_idx = np.argwhere(np.isfinite(scan_values) & (scan_values == best_val))
            if best_idx.size:
                iy, ix = [int(x) for x in best_idx[0]]
                scan_best = {"no_bao_opt_total_3sigma": best_val, "bin_width": scan_bin_widths[ix], "sn_min_points": scan_min_points[iy]}
    except Exception:
        scan_best = None

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "definition": {
            "what": "距離指標の誤差予算（SNe Ia/BAO）の推定が、SNeサンプルと共分散の扱いでどれだけ動くかの感度比較。",
            "z_limit": "必要補正 |Δμ_required(z)| が、m×budget(z) を初めて超える直前の z（m=1,3...）。budget は SNe と BAO と r_drag の二乗和平方（目安）。",
            "sn_sem_stat": "bin平均の誤差（SEM）。Pantheonはdiagの逆分散SEM。Pantheon+は diag-only 版と full covariance 版を比較。",
            "bao": "BOSS DR12 の D_M 相対誤差を等価Δμへ変換。Table 8（BAO+FS）の reduced covariance を用いた補間（目安）も上限側に使用。",
        },
        "inputs": {
            "pantheon_lcparam": str(pantheon_path).replace("\\", "/"),
            "pantheon_lcparam_sha256": _sha256(pantheon_path) if pantheon_path.exists() else None,
            "pantheon_sys_cov": str(pantheon_sys_path).replace("\\", "/") if pantheon_sys_path.exists() else None,
            "pantheon_sys_cov_sha256": _sha256(pantheon_sys_path) if pantheon_sys_path.exists() else None,
            "pantheonplus_dat": str(pplus_dat_path).replace("\\", "/"),
            "pantheonplus_dat_sha256": _sha256(pplus_dat_path) if pplus_dat_path.exists() else None,
            "pantheonplus_cov_statonly": str(pplus_cov_stat_path).replace("\\", "/") if pplus_cov_stat_path.exists() else None,
            "pantheonplus_cov_statonly_sha256": _sha256(pplus_cov_stat_path) if pplus_cov_stat_path.exists() else None,
            "pantheonplus_cov_stat_sys": str(pplus_cov_tot_path).replace("\\", "/") if pplus_cov_tot_path.exists() else None,
            "pantheonplus_cov_stat_sys_sha256": _sha256(pplus_cov_tot_path) if pplus_cov_tot_path.exists() else None,
            "boss_dr12_consensus": str(boss_path).replace("\\", "/"),
            "boss_dr12_baofs_cov_cij": str(boss_cov_path).replace("\\", "/") if boss_cov_path.exists() else None,
            "boss_dr12_baofs_cov_cij_sha256": _sha256(boss_cov_path) if boss_cov_path.exists() else None,
            "bao_r_drag_constraints": str(rdrag_path).replace("\\", "/"),
            "reach_metrics": str(reach_path).replace("\\", "/") if reach_path.exists() else None,
        },
        "params": {
            "z_max": z_max,
            "bin_width": bin_width,
            "sn_min_points": sn_min_points,
            "sigma_multipliers": sigma_multipliers,
            "pantheonplus_sem_stat_key_for_cov_scenario": sem_stat_cov_key_full,
            "scan_bin_widths": scan_bin_widths,
            "scan_sn_min_points": scan_min_points,
        },
        "reach_representatives": (reach_reps.get("reach_representatives") or reach_reps.get("reach") or {}),
        "pantheon": {
            "sn_bins": pantheon_bins,
            "summary": pantheon_metrics,
        },
        "pantheonplus": {
            "sn_bins_full": pplus_bins_full,
            "sn_bins_unique_cid": bins_unique,
            "sn_bins_sh0es_hf": bins_hf,
            "sn_bins_sh0es_hf_unique_cid": bins_hf_u,
            "summary_full_diag": pplus_full_diag_metrics,
            "summary_full_cov": pplus_full_cov_metrics,
            "summary_unique_cid_cov": metrics_unique_cov,
            "summary_sh0es_hf_cov": metrics_hf_cov,
            "summary_sh0es_hf_unique_cid_cov": metrics_hf_u_cov,
            "counts": {
                "full_rows": int(len(pplus_df)),
                "full_unique_cid": int(pplus_df["CID"].nunique()) if "CID" in pplus_df.columns else None,
                "sh0es_hf_rows": int(len(df_hf)),
                "sh0es_hf_unique_cid": int(df_hf["CID"].nunique()) if "CID" in df_hf.columns else None,
            },
        },
        "scenarios": scenarios_plot,
        "envelope": envelope,
        "scan": {
            "focus": "Pantheon+ SH0ES HF / no-BAO representative / opt_total 3σ",
            "rows": scan_rows,
            "grid_no_bao_opt_total_3sigma": scan_values.tolist(),
            "best": scan_best,
        },
        "outputs": {
            "png": str(out_png).replace("\\", "/"),
            "scan_png": str(out_scan_png).replace("\\", "/"),
            "metrics_json": str(out_json).replace("\\", "/"),
        },
        "notes": [
            "Pantheon+ の diagエラーは一次ソース側で『可視化用途（plotting/visual only）』と明記されているため、cov版（SEM）も併記して感度を示す。",
            "Pantheon+ の CID重複は、同一SNの複数サーベイ測光が含まれるために生じる（STATONLY共分散にも off-diagonal が入る）。ここでは『1行=1SN』の近似として CID重複除去の感度を確認する。",
            "SH0ES HF は Pantheon+ の列 USED_IN_SH0ES_HF==1 による限定（低zのH0推定用サブサンプル）で、到達限界の低z領域に効きやすい。",
            "SNe の距離指標は絶対等級や校正と退化するため、ここでの z_limit は“観測不確かさのスケール感”の目安。",
        ],
    }
    _write_json(out_json, payload)

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_distance_indicator_error_budget_sensitivity",
                "argv": list(sys.argv),
                "inputs": {
                    "pantheon": pantheon_path,
                    "pantheon_sys_cov": (pantheon_sys_path if pantheon_sys_path.exists() else None),
                    "pantheonplus_dat": pplus_dat_path,
                    "pantheonplus_cov_statonly": (pplus_cov_stat_path if pplus_cov_stat_path.exists() else None),
                    "pantheonplus_cov_stat_sys": (pplus_cov_tot_path if pplus_cov_tot_path.exists() else None),
                    "boss_ap": boss_path,
                    "boss_baofs_cov_cij": (boss_cov_path if boss_cov_path.exists() else None),
                    "bao_rdrag": rdrag_path,
                    "reach_metrics": (reach_path if reach_path.exists() else None),
                },
                "outputs": {"png": out_png, "metrics_json": out_json},
                "metrics": {"z_max": z_max, "bin_width": bin_width, "sn_min_points": sn_min_points},
            }
        )
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
