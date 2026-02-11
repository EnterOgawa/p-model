#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_bao_distance_ratio_fit.py

Step 14.2.29（BAO(s_R) の一次データ/解析の多系統比較）:
BOSS DR12 だけに依存しない形で、異方的BAO（D_M/r_d, D_H/r_d）の複数系統（BOSS/eBOSS/DESI）の値から
静的背景Pモデル + 標準定規進化 r_d(z)∝(1+z)^{s_R} を fit し、s_R がどの程度安定かを固定出力する。

モデル（静的背景Pの距離スケール）:
  D_M(z) = (c/H0) ln(1+z)
  H(z)   = H0 (1+z)  →  D_H(z)=c/H(z)

標準定規（音響地平線; r_d=r_drag）の進化を
  r_d(z)=r_d0 (1+z)^{s_R}
とすると、観測量の比は（r_d0 で規格化した無次元形）
  D_M/r_d = Q ln(1+z) (1+z)^{-s_R}
  D_H/r_d = Q (1+z)^{-(1+s_R)}
ここで Q ≡ (c/H0)/r_d0 は未知の定数（スケール因子）で、s_R と同時に推定する。

入力（固定）:
  - data/cosmology/bao_anisotropic_distance_ratio_constraints.json

出力（固定名）:
  - output/cosmology/cosmology_bao_distance_ratio_fit.png
  - output/cosmology/cosmology_bao_distance_ratio_fit_residuals.png
  - output/cosmology/cosmology_bao_distance_ratio_fit_leave_one_out.png
  - output/cosmology/cosmology_bao_distance_ratio_fit_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402

_C_KM_S = 299_792.458


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


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _fmt_float(x: Optional[float], *, digits: int = 6) -> str:
    if x is None:
        return ""
    if not math.isfinite(float(x)):
        return ""
    x = float(x)
    if x == 0.0:
        return "0"
    ax = abs(x)
    if ax >= 1e4 or ax < 1e-3:
        return f"{x:.{digits}g}"
    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


@dataclass(frozen=True)
class BAORatioPoint:
    id: str
    short_label: str
    z_eff: float
    dm_over_rd: float
    dm_over_rd_sigma: float
    dh_over_rd: float
    dh_over_rd_sigma: float
    corr_dm_dh: float
    source: Dict[str, Any]

    @staticmethod
    def from_json(j: Dict[str, Any]) -> "BAORatioPoint":
        return BAORatioPoint(
            id=str(j.get("id") or ""),
            short_label=str(j.get("short_label") or j.get("id") or ""),
            z_eff=float(j["z_eff"]),
            dm_over_rd=float(j["DM_over_rd"]),
            dm_over_rd_sigma=float(j["DM_over_rd_sigma"]),
            dh_over_rd=float(j["DH_over_rd"]),
            dh_over_rd_sigma=float(j["DH_over_rd_sigma"]),
            corr_dm_dh=float(j.get("corr_DM_DH", 0.0)),
            source=dict(j.get("source") or {}),
        )


def _pred_dm_dh_over_rd(*, z: float, s_R: float, Q: float) -> Tuple[float, float]:
    op = 1.0 + float(z)
    if not (op > 0.0):
        raise ValueError("z must satisfy 1+z>0")
    s_R = float(s_R)
    Q = float(Q)
    dm = Q * math.log(op) * (op ** (-s_R))
    dh = Q * (op ** (-(1.0 + s_R)))
    return float(dm), float(dh)


def _inv_cov_2x2(*, sig1: float, sig2: float, corr: float) -> Tuple[float, float, float]:
    sig1 = float(sig1)
    sig2 = float(sig2)
    corr = float(corr)
    cov = corr * sig1 * sig2
    a = sig1 * sig1
    d = sig2 * sig2
    det = a * d - cov * cov
    if not (det > 0.0):
        return (float("nan"), float("nan"), float("nan"))
    inv11 = d / det
    inv22 = a / det
    inv12 = -cov / det
    return float(inv11), float(inv12), float(inv22)


def _profile_Q_for_sR(points: Sequence[BAORatioPoint], *, s_R: float) -> Dict[str, float]:
    """
    For fixed s_R, the model is linear in Q:
      y ≈ Q m(s_R)
    with y=[DM/rd, DH/rd]. We solve the scalar weighted least squares analytically.
    """
    num = 0.0
    den = 0.0
    for p in points:
        dm1, dh1 = _pred_dm_dh_over_rd(z=p.z_eff, s_R=float(s_R), Q=1.0)
        m = np.array([dm1, dh1], dtype=float)
        y = np.array([p.dm_over_rd, p.dh_over_rd], dtype=float)

        inv11, inv12, inv22 = _inv_cov_2x2(sig1=p.dm_over_rd_sigma, sig2=p.dh_over_rd_sigma, corr=p.corr_dm_dh)
        if not (math.isfinite(inv11) and math.isfinite(inv22) and math.isfinite(inv12)):
            return {"Q_best": float("nan"), "Q_sigma": float("nan"), "chi2": float("nan")}
        W = np.array([[inv11, inv12], [inv12, inv22]], dtype=float)

        num += float(m.T @ W @ y)
        den += float(m.T @ W @ m)

    if not (den > 0.0 and math.isfinite(den) and math.isfinite(num)):
        return {"Q_best": float("nan"), "Q_sigma": float("nan"), "chi2": float("nan")}
    Q = float(num / den)
    Q_sig = float(math.sqrt(1.0 / den))

    chi2 = 0.0
    for p in points:
        dm_pred, dh_pred = _pred_dm_dh_over_rd(z=p.z_eff, s_R=float(s_R), Q=Q)
        d_dm = float(p.dm_over_rd) - dm_pred
        d_dh = float(p.dh_over_rd) - dh_pred

        inv11, inv12, inv22 = _inv_cov_2x2(sig1=p.dm_over_rd_sigma, sig2=p.dh_over_rd_sigma, corr=p.corr_dm_dh)
        if not (math.isfinite(inv11) and math.isfinite(inv22) and math.isfinite(inv12)):
            return {"Q_best": float("nan"), "Q_sigma": float("nan"), "chi2": float("nan")}
        chi2 += inv11 * d_dm * d_dm + 2.0 * inv12 * d_dm * d_dh + inv22 * d_dh * d_dh

    return {"Q_best": float(Q), "Q_sigma": float(Q_sig), "chi2": float(chi2)}


def _fit_sR_Q(points: Sequence[BAORatioPoint], *, s_R_min: float = -1.0, s_R_max: float = 2.0) -> Dict[str, Any]:
    if not points:
        raise ValueError("no BAO ratio points")

    # 1) coarse grid
    s0 = np.linspace(float(s_R_min), float(s_R_max), 301, dtype=float)
    prof0 = []
    for s in s0:
        r = _profile_Q_for_sR(points, s_R=float(s))
        prof0.append({"s_R": float(s), **r})
    best0 = min(prof0, key=lambda x: (x["chi2"] if math.isfinite(x["chi2"]) else float("inf")))

    # 2) refine
    s_best0 = float(best0["s_R"])
    lo = max(float(s_R_min), s_best0 - 0.2)
    hi = min(float(s_R_max), s_best0 + 0.2)
    s1 = np.linspace(lo, hi, 801, dtype=float)
    prof = []
    for s in s1:
        r = _profile_Q_for_sR(points, s_R=float(s))
        prof.append({"s_R": float(s), **r})
    best = min(prof, key=lambda x: (x["chi2"] if math.isfinite(x["chi2"]) else float("inf")))

    chi2_min = float(best["chi2"])
    s_best = float(best["s_R"])
    Q_best = float(best["Q_best"])
    Q_sig = float(best["Q_sigma"])

    # 3) 1σ interval in s_R (profile likelihood; Δχ2=1)
    target = chi2_min + 1.0
    s_arr = np.array([float(r["s_R"]) for r in prof], dtype=float)
    c_arr = np.array([float(r["chi2"]) for r in prof], dtype=float)
    ok = np.isfinite(c_arr)
    if not np.any(ok):
        s_lo = float("nan")
        s_hi = float("nan")
    else:
        s_arr = s_arr[ok]
        c_arr = c_arr[ok]
        # left crossing
        left = (s_arr <= s_best) & (c_arr <= target)
        right = (s_arr >= s_best) & (c_arr <= target)
        s_lo = float(s_arr[left][0]) if np.any(left) else float("nan")
        s_hi = float(s_arr[right][-1]) if np.any(right) else float("nan")

        # Improve using linear interpolation to the nearest above-target point on each side.
        # left side
        i_best = int(np.argmin(np.abs(s_arr - s_best)))
        # walk left
        i = i_best
        while i > 0 and c_arr[i] <= target:
            i -= 1
        if i_best > 0 and c_arr[i] > target and c_arr[i + 1] <= target:
            # interpolate between i (above) and i+1 (below)
            s0i, s1i = float(s_arr[i]), float(s_arr[i + 1])
            c0i, c1i = float(c_arr[i]), float(c_arr[i + 1])
            denom = float(c1i - c0i)
            if abs(denom) < 1e-300:
                denom = math.copysign(1e-300, denom if denom != 0.0 else 1.0)
            t = (target - c0i) / denom
            s_lo = float(s0i + t * (s1i - s0i))
        # right side
        i = i_best
        while i < len(s_arr) - 1 and c_arr[i] <= target:
            i += 1
        if i_best < len(s_arr) - 1 and c_arr[i] > target and c_arr[i - 1] <= target:
            s0i, s1i = float(s_arr[i - 1]), float(s_arr[i])
            c0i, c1i = float(c_arr[i - 1]), float(c_arr[i])
            denom = float(c1i - c0i)
            if abs(denom) < 1e-300:
                denom = math.copysign(1e-300, denom if denom != 0.0 else 1.0)
            t = (target - c0i) / denom
            s_hi = float(s0i + t * (s1i - s0i))

    s_sig = float((s_hi - s_lo) / 2.0) if (math.isfinite(s_lo) and math.isfinite(s_hi)) else float("nan")

    dof = int(2 * len(points) - 2)
    chi2_dof = float(chi2_min / dof) if dof > 0 else None

    # Diagnostics per point (component-wise residuals and per-point chi2 contribution).
    rows = []
    for p in points:
        dm_pred, dh_pred = _pred_dm_dh_over_rd(z=p.z_eff, s_R=s_best, Q=Q_best)
        d_dm = float(p.dm_over_rd) - float(dm_pred)
        d_dh = float(p.dh_over_rd) - float(dh_pred)
        inv11, inv12, inv22 = _inv_cov_2x2(sig1=p.dm_over_rd_sigma, sig2=p.dh_over_rd_sigma, corr=p.corr_dm_dh)
        chi2_i = float("nan")
        if math.isfinite(inv11) and math.isfinite(inv12) and math.isfinite(inv22):
            chi2_i = float(inv11 * d_dm * d_dm + 2.0 * inv12 * d_dm * d_dh + inv22 * d_dh * d_dh)
        sqrt_chi2 = float(math.sqrt(chi2_i)) if (math.isfinite(chi2_i) and chi2_i >= 0.0) else float("nan")
        rows.append(
            {
                "id": p.id,
                "short_label": p.short_label,
                "z_eff": p.z_eff,
                "dm_over_rd_obs": p.dm_over_rd,
                "dm_over_rd_pred": dm_pred,
                "dm_residual_sigma": d_dm / max(1e-300, p.dm_over_rd_sigma),
                "dh_over_rd_obs": p.dh_over_rd,
                "dh_over_rd_pred": dh_pred,
                "dh_residual_sigma": d_dh / max(1e-300, p.dh_over_rd_sigma),
                "corr_dm_dh": p.corr_dm_dh,
                "chi2_contrib": chi2_i,
                "sqrt_chi2": sqrt_chi2,
            }
        )

    return {
        "best_fit": {
            "s_R": s_best,
            "s_R_sigma_1d": s_sig,
            "s_R_interval_1sigma": [s_lo, s_hi],
            "Q": Q_best,
            "Q_sigma_conditional": Q_sig,
        },
        "chi2": chi2_min,
        "dof": dof,
        "chi2_dof": chi2_dof,
        "profile_refined": [{"s_R": float(r["s_R"]), "Q_best": float(r["Q_best"]), "chi2": float(r["chi2"])} for r in prof],
        "per_point": rows,
    }


def _plot(
    *,
    out_png: Path,
    boss: Sequence[BAORatioPoint],
    eboss: Sequence[BAORatioPoint],
    desi: Sequence[BAORatioPoint],
    fit_boss: Dict[str, Any],
    fit_eboss: Dict[str, Any],
    fit_desi: Dict[str, Any],
    fit_all: Dict[str, Any],
) -> None:
    _set_japanese_font()
    import matplotlib.pyplot as plt

    def _curve(fit: Dict[str, Any], z_curve: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        s = float(((fit.get("best_fit") or {}).get("s_R")) or float("nan"))
        Q = float(((fit.get("best_fit") or {}).get("Q")) or float("nan"))
        dm = []
        dh = []
        for z in z_curve:
            d1, d2 = _pred_dm_dh_over_rd(z=float(z), s_R=float(s), Q=float(Q))
            dm.append(d1)
            dh.append(d2)
        return np.array(dm, dtype=float), np.array(dh, dtype=float)

    def _g(points: Sequence[BAORatioPoint]) -> Dict[str, np.ndarray]:
        z = np.array([p.z_eff for p in points], dtype=float)
        dm = np.array([p.dm_over_rd for p in points], dtype=float)
        dms = np.array([p.dm_over_rd_sigma for p in points], dtype=float)
        dh = np.array([p.dh_over_rd for p in points], dtype=float)
        dhs = np.array([p.dh_over_rd_sigma for p in points], dtype=float)
        return {"z": z, "dm": dm, "dms": dms, "dh": dh, "dhs": dhs}

    b = _g(boss)
    e = _g(eboss)
    d = _g(desi)

    z_all = [*b["z"].tolist(), *e["z"].tolist(), *d["z"].tolist()]
    if not z_all:
        return
    z_min = float(min(z_all))
    z_max = float(max(z_all))
    z_curve = np.linspace(max(0.0, z_min - 0.05), z_max + 0.1, 240, dtype=float)

    dm_all, dh_all = _curve(fit_all, z_curve)
    dm_b, dh_b = _curve(fit_boss, z_curve)
    dm_e, dh_e = _curve(fit_eboss, z_curve)
    dm_d, dh_d = _curve(fit_desi, z_curve)

    fig = plt.figure(figsize=(15.5, 6.8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    # Data points
    if boss:
        ax1.errorbar(b["z"], b["dm"], yerr=b["dms"], fmt="o", capsize=4, color="#111111", ecolor="#111111", label="BOSS DR12（比へ変換）")
        ax2.errorbar(b["z"], b["dh"], yerr=b["dhs"], fmt="o", capsize=4, color="#111111", ecolor="#111111", label="BOSS DR12（比へ変換）")
    if eboss:
        ax1.errorbar(e["z"], e["dm"], yerr=e["dms"], fmt="s", capsize=4, color="#1f77b4", ecolor="#1f77b4", label="eBOSS DR16（一次ソース比）")
        ax2.errorbar(e["z"], e["dh"], yerr=e["dhs"], fmt="s", capsize=4, color="#1f77b4", ecolor="#1f77b4", label="eBOSS DR16（一次ソース比）")
    if desi:
        ax1.errorbar(d["z"], d["dm"], yerr=d["dms"], fmt="^", capsize=4, color="#2ca02c", ecolor="#2ca02c", label="DESI DR1（bao_data mean/cov）")
        ax2.errorbar(d["z"], d["dh"], yerr=d["dhs"], fmt="^", capsize=4, color="#2ca02c", ecolor="#2ca02c", label="DESI DR1（bao_data mean/cov）")

    # Model curves
    ax1.plot(z_curve, dm_all, color="#d62728", linewidth=2.2, label="fit（BOSS+eBOSS+DESI）")
    ax2.plot(z_curve, dh_all, color="#d62728", linewidth=2.2, label="fit（BOSS+eBOSS+DESI）")
    ax1.plot(z_curve, dm_b, color="#555555", linewidth=1.6, linestyle="--", alpha=0.9, label="fit（BOSSのみ）")
    ax2.plot(z_curve, dh_b, color="#555555", linewidth=1.6, linestyle="--", alpha=0.9, label="fit（BOSSのみ）")
    ax1.plot(z_curve, dm_e, color="#1f77b4", linewidth=1.6, linestyle="--", alpha=0.9, label="fit（eBOSSのみ）")
    ax2.plot(z_curve, dh_e, color="#1f77b4", linewidth=1.6, linestyle="--", alpha=0.9, label="fit（eBOSSのみ）")
    ax1.plot(z_curve, dm_d, color="#2ca02c", linewidth=1.6, linestyle="--", alpha=0.9, label="fit（DESIのみ）")
    ax2.plot(z_curve, dh_d, color="#2ca02c", linewidth=1.6, linestyle="--", alpha=0.9, label="fit（DESIのみ）")

    ax1.set_xlabel("赤方偏移 z", fontsize=11)
    ax2.set_xlabel("赤方偏移 z", fontsize=11)
    ax1.set_ylabel("D_M(z)/r_d", fontsize=11)
    ax2.set_ylabel("D_H(z)/r_d", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="best", fontsize=9, frameon=True)
    ax2.legend(loc="best", fontsize=9, frameon=True)

    def _sline(fit: Dict[str, Any]) -> str:
        bf = dict(fit.get("best_fit") or {})
        s = bf.get("s_R")
        ss = bf.get("s_R_sigma_1d")
        if s is None or ss is None or not (math.isfinite(float(s)) and math.isfinite(float(ss))):
            return "s_R=na"
        return f"s_R={_fmt_float(float(s), digits=3)}±{_fmt_float(float(ss), digits=3)}"

    fig.suptitle("宇宙論（BAOの多系統比較）：D_M/r_d と D_H/r_d の fit から s_R を推定（静的背景P）", fontsize=14)
    fig.text(
        0.5,
        0.02,
        f"BOSSのみ: {_sline(fit_boss)} / eBOSSのみ: {_sline(fit_eboss)} / DESIのみ: {_sline(fit_desi)} / 併合: {_sline(fit_all)}",
        ha="center",
        fontsize=10,
    )
    plt.tight_layout(rect=(0.0, 0.05, 1.0, 0.93))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_residuals(
    *,
    out_png: Path,
    fit_all: Dict[str, Any],
) -> None:
    _set_japanese_font()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    per = fit_all.get("per_point") if isinstance(fit_all.get("per_point"), list) else []
    rows = [r for r in per if isinstance(r, dict)]
    if not rows:
        return

    # stable order: by z
    rows.sort(key=lambda r: float(r.get("z_eff") or 0.0))

    def _cat(pid: str) -> str:
        pid = str(pid or "")
        if pid.startswith("boss_dr12"):
            return "BOSS"
        if pid.startswith("eboss_dr16") or pid.startswith("eboss_"):
            return "eBOSS"
        if pid.startswith("desi_dr1") or pid.startswith("desi_"):
            return "DESI"
        return "other"

    colors = {"BOSS": "#111111", "eBOSS": "#1f77b4", "DESI": "#2ca02c", "other": "#777777"}

    labels = [f'{r.get("short_label")}\n(z={_fmt_float(float(r.get("z_eff") or 0.0), digits=3)})' for r in rows]
    vals = [float(r.get("sqrt_chi2") or float("nan")) for r in rows]
    cats = [_cat(str(r.get("id") or "")) for r in rows]
    bar_colors = [colors.get(c, "#777777") for c in cats]

    fig, ax = plt.subplots(figsize=(13.8, 6.8))
    y = np.arange(len(rows), dtype=float)

    # Replace non-finite with 0 (and annotate as NA)
    vals_plot = [v if math.isfinite(v) else 0.0 for v in vals]
    ax.barh(y, vals_plot, color=bar_colors, alpha=0.9)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("√(点ごとの χ²寄与)（目安）", fontsize=11)
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)

    ax.axvline(1.0, color="#2ca02c", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axvline(3.0, color="#ffbf00", linestyle="--", linewidth=1.5, alpha=0.6)

    for yi, v, r in zip(y, vals, rows):
        dmz = r.get("dm_residual_sigma")
        dhz = r.get("dh_residual_sigma")
        if not (isinstance(dmz, (int, float)) and isinstance(dhz, (int, float))):
            continue
        if not (math.isfinite(float(dmz)) and math.isfinite(float(dhz))):
            continue
        ax.text(
            max(0.02, vals_plot[int(yi)] + 0.05),
            float(yi),
            f"DM:{float(dmz):+.2f}σ / DH:{float(dhz):+.2f}σ",
            va="center",
            ha="left",
            fontsize=9,
            color="#333333",
        )

    bf = fit_all.get("best_fit") if isinstance(fit_all.get("best_fit"), dict) else {}
    sR = bf.get("s_R")
    sRs = bf.get("s_R_sigma_1d")
    chi2 = fit_all.get("chi2")
    dof = fit_all.get("dof")
    fig.suptitle("宇宙論（BAO多系統比較）：併合fit の点ごとの残差（√χ²寄与）", fontsize=13)
    fig.text(
        0.5,
        0.02,
        f"併合fit: s_R={_fmt_float(sR, digits=3)}±{_fmt_float(sRs, digits=3)} / χ²/dof={_fmt_float(chi2, digits=3)}/{dof}",
        ha="center",
        fontsize=10,
        color="#333333",
    )
    handles = [
        Patch(facecolor=colors["BOSS"], edgecolor="#333333", label="BOSS"),
        Patch(facecolor=colors["eBOSS"], edgecolor="#333333", label="eBOSS"),
        Patch(facecolor=colors["DESI"], edgecolor="#333333", label="DESI"),
        Patch(facecolor=colors["other"], edgecolor="#333333", label="other"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=10, frameon=True)

    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.93))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_leave_one_out(
    *,
    out_png: Path,
    fit_all: Dict[str, Any],
    leave_one_out: Sequence[Dict[str, Any]],
) -> None:
    """
    Leave-one-out sensitivity for the combined fit:
      for each omitted point, refit and show (s_R ± σ) and chi2/dof.
    """
    _set_japanese_font()
    import matplotlib.pyplot as plt

    rows = [r for r in leave_one_out if isinstance(r, dict)]
    if not rows:
        return

    def _cat(pid: str) -> str:
        pid = str(pid or "")
        if pid.startswith("boss_dr12"):
            return "BOSS"
        if pid.startswith("eboss_dr16") or pid.startswith("eboss_"):
            return "eBOSS"
        if pid.startswith("desi_dr1") or pid.startswith("desi_"):
            return "DESI"
        return "other"

    colors = {"BOSS": "#111111", "eBOSS": "#1f77b4", "DESI": "#2ca02c", "other": "#777777"}

    labels = []
    cats = []
    s_vals = []
    s_sig = []
    chi2_dof = []

    for r in rows:
        omit = r.get("omitted") if isinstance(r.get("omitted"), dict) else {}
        fit = r.get("fit") if isinstance(r.get("fit"), dict) else {}
        bf = fit.get("best_fit") if isinstance(fit.get("best_fit"), dict) else {}

        pid = str(omit.get("id") or "")
        cats.append(_cat(pid))

        z = omit.get("z_eff")
        z_txt = _fmt_float(float(z), digits=3) if isinstance(z, (int, float)) and math.isfinite(float(z)) else ""
        labels.append(f"drop: {omit.get('short_label')}\n(z={z_txt})")

        try:
            s = float(bf.get("s_R"))
            ss = float(bf.get("s_R_sigma_1d"))
        except Exception:
            s = float("nan")
            ss = float("nan")
        s_vals.append(s)
        s_sig.append(ss)

        try:
            cd = float(fit.get("chi2_dof"))
        except Exception:
            cd = float("nan")
        chi2_dof.append(cd)

    x = np.arange(len(rows), dtype=float)
    fig = plt.figure(figsize=(15.5, 7.8))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)

    # s_R with error bars
    for cat in ("BOSS", "eBOSS", "DESI", "other"):
        idx = [i for i, c in enumerate(cats) if c == cat]
        if not idx:
            continue
        ax1.errorbar(
            x[idx],
            [s_vals[i] for i in idx],
            yerr=[s_sig[i] for i in idx],
            fmt="o",
            capsize=4,
            color=colors[cat],
            ecolor=colors[cat],
            label=cat,
        )

    # baseline
    bf0 = fit_all.get("best_fit") if isinstance(fit_all.get("best_fit"), dict) else {}
    try:
        s0 = float(bf0.get("s_R"))
        s0s = float(bf0.get("s_R_sigma_1d"))
    except Exception:
        s0 = float("nan")
        s0s = float("nan")
    ax1.axhline(s0, color="#d62728", linestyle="--", linewidth=1.6, alpha=0.75, label="baseline（併合）")
    ax1.fill_between(
        [-0.5, float(len(rows)) - 0.5],
        [s0 - s0s, s0 - s0s],
        [s0 + s0s, s0 + s0s],
        color="#d62728",
        alpha=0.08,
        linewidth=0.0,
    )

    ax1.set_ylabel("s_R（併合fit; drop-1）", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(loc="best", fontsize=10, frameon=True)

    # chi2/dof
    bar_colors = [colors.get(c, "#777777") for c in cats]
    vals = [v if math.isfinite(float(v)) else 0.0 for v in chi2_dof]
    ax2.bar(x, vals, color=bar_colors, alpha=0.85)

    chi2_0 = fit_all.get("chi2")
    dof_0 = fit_all.get("dof")
    chi2d_0 = None
    try:
        chi2d_0 = float(chi2_0) / float(dof_0) if (float(dof_0) > 0.0) else None
    except Exception:
        chi2d_0 = None
    if chi2d_0 is not None and math.isfinite(float(chi2d_0)):
        ax2.axhline(float(chi2d_0), color="#d62728", linestyle="--", linewidth=1.6, alpha=0.75)
        ax2.text(
            0.01,
            float(chi2d_0) * 1.02,
            f"baseline χ²/dof≈{float(chi2d_0):.2f}",
            fontsize=9,
            color="#d62728",
            ha="left",
            va="bottom",
        )

    ax2.axhline(1.0, color="#2ca02c", linestyle="--", linewidth=1.2, alpha=0.7)
    ax2.set_ylabel("χ²/dof（併合fit; drop-1）", fontsize=11)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9, rotation=0)

    fig.suptitle("宇宙論（BAO多系統比較）：併合fit の leave-one-out 感度（どの点が結論/張力を支配するか）", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.93))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Cosmology: BAO distance-ratio fit for s_R (Step 14.2.29).")
    ap.add_argument(
        "--bao-ratio-data",
        type=str,
        default=str(_ROOT / "data" / "cosmology" / "bao_anisotropic_distance_ratio_constraints.json"),
        help="Input BAO distance-ratio constraints JSON.",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    in_path = Path(args.bao_ratio_data)
    raw = _read_json(in_path)
    pts = [BAORatioPoint.from_json(x) for x in (raw.get("constraints") or []) if isinstance(x, dict)]
    if not pts:
        raise ValueError("No constraints in input JSON")
    pts = sorted(pts, key=lambda p: p.z_eff)

    boss = [p for p in pts if p.id.startswith("boss_dr12")]
    eboss = [p for p in pts if p.id.startswith("eboss_dr16") or p.id.startswith("eboss_")]
    desi = [p for p in pts if p.id.startswith("desi_dr1") or p.id.startswith("desi_")]
    other = [p for p in pts if p not in boss and p not in eboss and p not in desi]
    if other:
        # Keep them in the combined fit, but label them as "other" in metrics.
        pass

    fit_boss = _fit_sR_Q(boss) if boss else {"best_fit": {}, "chi2": float("nan"), "dof": 0, "chi2_dof": None}
    fit_eboss = _fit_sR_Q(eboss) if eboss else {"best_fit": {}, "chi2": float("nan"), "dof": 0, "chi2_dof": None}
    fit_desi = _fit_sR_Q(desi) if desi else {"best_fit": {}, "chi2": float("nan"), "dof": 0, "chi2_dof": None}
    fit_all = _fit_sR_Q(pts)

    out_dir = _ROOT / "output" / "cosmology"
    out_png = out_dir / "cosmology_bao_distance_ratio_fit.png"
    out_png_resid = out_dir / "cosmology_bao_distance_ratio_fit_residuals.png"
    out_png_loo = out_dir / "cosmology_bao_distance_ratio_fit_leave_one_out.png"
    out_json = out_dir / "cosmology_bao_distance_ratio_fit_metrics.json"

    _plot(
        out_png=out_png,
        boss=boss,
        eboss=eboss,
        desi=desi,
        fit_boss=fit_boss,
        fit_eboss=fit_eboss,
        fit_desi=fit_desi,
        fit_all=fit_all,
    )
    _plot_residuals(out_png=out_png_resid, fit_all=fit_all)

    # Leave-one-out sensitivity for the combined fit (drop each point once).
    loo_rows: List[Dict[str, Any]] = []
    for p in pts:
        subset = [q for q in pts if q.id != p.id]
        if len(subset) < 2:
            continue
        fit = _fit_sR_Q(subset)
        fit_small = {
            "best_fit": dict(fit.get("best_fit") or {}),
            "chi2": fit.get("chi2"),
            "dof": fit.get("dof"),
            "chi2_dof": fit.get("chi2_dof"),
        }
        loo_rows.append(
            {
                "omitted": {"id": p.id, "short_label": p.short_label, "z_eff": p.z_eff},
                "fit": fit_small,
            }
        )
    _plot_leave_one_out(out_png=out_png_loo, fit_all=fit_all, leave_one_out=loo_rows)

    metrics: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "definition": {
            "model": {
                "DM_over_rd": "Q ln(1+z) (1+z)^(-s_R)",
                "DH_over_rd": "Q (1+z)^(-(1+s_R))",
                "Q": "Q ≡ (c/H0)/r_d0 (dimensionless scale factor)",
            },
            "notes": [
                "This is a consistency fit under static background-P geometry + effective ruler evolution.",
                "It does not imply that BAO physics or early-universe calibration is explained by this model.",
            ],
        },
        "inputs": {
            "bao_ratio": str(in_path.relative_to(_ROOT)).replace("\\", "/"),
        },
        "data_summary": {
            "n_total": len(pts),
            "n_boss": len(boss),
            "n_eboss": len(eboss),
            "n_desi": len(desi),
            "n_other": len(other),
            "ids_other": [p.id for p in other],
        },
        "results": {
            "boss_only": fit_boss,
            "eboss_only": fit_eboss,
            "desi_only": fit_desi,
            "combined": fit_all,
            "combined_leave_one_out": loo_rows,
        },
        "outputs": {
            "png": str(out_png.relative_to(_ROOT)).replace("\\", "/"),
            "residuals_png": str(out_png_resid.relative_to(_ROOT)).replace("\\", "/"),
            "leave_one_out_png": str(out_png_loo.relative_to(_ROOT)).replace("\\", "/"),
            "metrics_json": str(out_json.relative_to(_ROOT)).replace("\\", "/"),
        },
    }
    _write_json(out_json, metrics)

    worklog.append_event(
        {
            "kind": "cosmology",
            "step": "14.2.29",
            "task": "bao_distance_ratio_fit",
            "inputs": [str(in_path.relative_to(_ROOT)).replace("\\", "/")],
            "outputs": {"png": out_png, "residuals_png": out_png_resid, "leave_one_out_png": out_png_loo, "metrics_json": out_json},
            "metrics": {
                "boss_only_s_R": ((fit_boss.get("best_fit") or {}).get("s_R")),
                "eboss_only_s_R": ((fit_eboss.get("best_fit") or {}).get("s_R")),
                "combined_s_R": ((fit_all.get("best_fit") or {}).get("s_R")),
                "combined_chi2_dof": fit_all.get("chi2_dof"),
            },
        }
    )

    print(f"[ok] png : {out_png}")
    print(f"[ok] png : {out_png_resid}")
    print(f"[ok] png : {out_png_loo}")
    print(f"[ok] json: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
