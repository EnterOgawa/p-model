#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_bao_xi_multipole_peakfit.py

Phase 16（宇宙論）/ Step 16.4（BAO一次統計の再導出）:
圧縮出力（D_M/r_d, D_H/r_d）の前に、BOSS DR12 の post-reconstruction 相関関数 multipoles ξℓ（ℓ=0,2）
という“より一次に近い観測統計”から、P-model 側の距離変換（AP/異方歪み）を入れ替えて比較する入口を作る。

方針（合意）:
1) BOSS DR12 再構成後 ξℓ（ℓ=0,2）から開始
2) 距離変換（θ,z→s,μ）を P-model 側で定義し、AP/warping を入れ替える
3) フィットは「なめらか成分＋BAOピーク」だけ（最小モデル）
4) r_d はフリー（全体スケール α に吸収）。異方（ℓ=2）を重視して幾何整合（ε）を評価

入力（固定・キャッシュ）:
  - data/cosmology/ross_2016_combineddr12_corrfunc/
    - Ross_2016_COMBINEDDR12_zbin{1,2,3}_correlation_function_{monopole,quadrupole}_post_recon_bincent{0..4}.dat
    - Ross_2016_COMBINEDDR12_zbin{1,2,3}_covariance_monoquad_post_recon_bincent{0..4}.dat
  ※ 未取得の場合は、先に scripts/cosmology/fetch_boss_dr12_ross2016_corrfunc.py を実行する。

出力（固定名）:
  - output/private/cosmology/cosmology_bao_xi_multipole_peakfit.png
  - output/private/cosmology/cosmology_bao_xi_multipole_peakfit_metrics.json
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
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


@dataclass(frozen=True)
class ZBin:
    zbin: int
    z_eff: float
    label: str


_ZBINS: List[ZBin] = [
    ZBin(zbin=1, z_eff=0.38, label="z=0.38 (bin1)"),
    ZBin(zbin=2, z_eff=0.51, label="z=0.51 (bin2)"),
    ZBin(zbin=3, z_eff=0.61, label="z=0.61 (bin3)"),
]


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
        pass


def _read_table(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s: List[float] = []
    xi: List[float] = []
    sig: List[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        t = line.strip()
        # 条件分岐: `not t or t.startswith("#")` を満たす経路を評価する。
        if not t or t.startswith("#"):
            continue

        parts = t.split()
        # 条件分岐: `len(parts) < 2` を満たす経路を評価する。
        if len(parts) < 2:
            continue

        s.append(float(parts[0]))
        xi.append(float(parts[1]))
        sig.append(float(parts[2]) if len(parts) >= 3 else float("nan"))

    return np.asarray(s, dtype=float), np.asarray(xi, dtype=float), np.asarray(sig, dtype=float)


def _read_cov(path: Path) -> np.ndarray:
    rows: List[List[float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        t = line.strip()
        # 条件分岐: `not t or t.startswith("#")` を満たす経路を評価する。
        if not t or t.startswith("#"):
            continue

        rows.append([float(x) for x in t.split()])

    cov = np.asarray(rows, dtype=float)
    # 条件分岐: `cov.ndim != 2 or cov.shape[0] != cov.shape[1]` を満たす経路を評価する。
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"invalid covariance shape: {cov.shape} from {path}")

    cov = 0.5 * (cov + cov.T)
    return cov


def _read_table_two_cols(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    s: List[float] = []
    xi: List[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        t = line.strip()
        # 条件分岐: `not t or t.startswith("#")` を満たす経路を評価する。
        if not t or t.startswith("#"):
            continue

        parts = t.split()
        # 条件分岐: `len(parts) < 2` を満たす経路を評価する。
        if len(parts) < 2:
            continue

        s.append(float(parts[0]))
        xi.append(float(parts[1]))

    return np.asarray(s, dtype=float), np.asarray(xi, dtype=float)


def _read_satpathy_cov(path: Path) -> np.ndarray:
    # Same format as Ross (dense whitespace-separated matrix) but without comment header.
    rows: List[List[float]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        t = line.strip()
        # 条件分岐: `not t` を満たす経路を評価する。
        if not t:
            continue

        # 条件分岐: `t.startswith("#")` を満たす経路を評価する。

        if t.startswith("#"):
            continue

        rows.append([float(x) for x in t.split()])

    cov = np.asarray(rows, dtype=float)
    # 条件分岐: `cov.ndim != 2 or cov.shape[0] != cov.shape[1]` を満たす経路を評価する。
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"invalid covariance shape: {cov.shape} from {path}")

    cov = 0.5 * (cov + cov.T)
    return cov


def _satpathy_s_bins_from_cov(cov: np.ndarray) -> np.ndarray:
    # Pre-recon covariance is 48x48 (mono+quad). We assume 5 Mpc/h binning from 30.
    n2 = int(cov.shape[0])
    # 条件分岐: `n2 % 2 != 0` を満たす経路を評価する。
    if n2 % 2 != 0:
        raise ValueError(f"unexpected covariance size (not even): {cov.shape}")

    n = n2 // 2
    s0 = 30.0
    ds = 5.0
    return s0 + ds * np.arange(n, dtype=float)


def _select_by_exact_s(s_all: np.ndarray, y_all: np.ndarray, s_sel: np.ndarray) -> np.ndarray:
    lookup = {float(x): float(v) for x, v in zip(np.asarray(s_all, dtype=float), np.asarray(y_all, dtype=float))}
    out = []
    missing = []
    for s in np.asarray(s_sel, dtype=float):
        key = float(s)
        # 条件分岐: `key not in lookup` を満たす経路を評価する。
        if key not in lookup:
            missing.append(key)
            continue

        out.append(lookup[key])

    # 条件分岐: `missing` を満たす経路を評価する。

    if missing:
        raise ValueError(f"missing s bins in data: {missing[:10]}{'...' if len(missing)>10 else ''}")

    return np.asarray(out, dtype=float)


def _p2(mu: np.ndarray) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    return 0.5 * (3.0 * mu * mu - 1.0)


def _f_ap_pbg_exponential(z: float) -> float:
    op = 1.0 + float(z)
    # 条件分岐: `not (op > 0.0)` を満たす経路を評価する。
    if not (op > 0.0):
        return float("nan")

    return float(op * math.log(op))


def _f_ap_lcdm_flat(z: float, *, omega_m: float, n_grid: int = 4000) -> float:
    z = float(z)
    # 条件分岐: `z < 0` を満たす経路を評価する。
    if z < 0:
        return float("nan")

    # 条件分岐: `not (0.0 < float(omega_m) < 1.0)` を満たす経路を評価する。

    if not (0.0 < float(omega_m) < 1.0):
        return float("nan")

    z_grid = np.linspace(0.0, z, int(n_grid), dtype=float)
    one_p = 1.0 + z_grid
    ez_grid = np.sqrt(float(omega_m) * one_p**3 + (1.0 - float(omega_m)))
    integrand = 1.0 / ez_grid
    try:
        integral = float(np.trapezoid(integrand, z_grid))
    except AttributeError:
        integral = float(np.trapz(integrand, z_grid))

    ez = float(math.sqrt(float(omega_m) * (1.0 + z) ** 3 + (1.0 - float(omega_m))))
    return float(ez * integral)


def _eps_from_f_ap_ratio(*, f_ap_model: float, f_ap_fid: float) -> float:
    # 条件分岐: `not (math.isfinite(f_ap_model) and math.isfinite(f_ap_fid) and f_ap_model > 0...` を満たす経路を評価する。
    if not (math.isfinite(f_ap_model) and math.isfinite(f_ap_fid) and f_ap_model > 0.0 and f_ap_fid > 0.0):
        return float("nan")
    # alpha_parallel/alpha_perp = F_AP_fid / F_AP_model

    ratio = float(f_ap_fid / f_ap_model)
    return float(ratio ** (1.0 / 3.0) - 1.0)


def _subset_monoquad(
    *,
    s: np.ndarray,
    xi0: np.ndarray,
    xi2: np.ndarray,
    cov_monoquad: np.ndarray,
    r_min: float,
    r_max: float,
) -> Dict[str, Any]:
    s = np.asarray(s, dtype=float)
    xi0 = np.asarray(xi0, dtype=float)
    xi2 = np.asarray(xi2, dtype=float)

    # 条件分岐: `s.ndim != 1` を満たす経路を評価する。
    if s.ndim != 1:
        raise ValueError("s must be 1D")

    n = int(s.size)
    # 条件分岐: `cov_monoquad.shape != (2 * n, 2 * n)` を満たす経路を評価する。
    if cov_monoquad.shape != (2 * n, 2 * n):
        raise ValueError(f"covariance shape mismatch: expected {(2*n,2*n)}, got {cov_monoquad.shape}")

    mask = (s >= float(r_min)) & (s <= float(r_max))
    idx = np.where(mask)[0]
    # 条件分岐: `idx.size < 8` を満たす経路を評価する。
    if idx.size < 8:
        raise ValueError(f"too few bins after range cut: n={idx.size} (r_min={r_min}, r_max={r_max})")

    sel = np.concatenate([idx, idx + n])
    cov = cov_monoquad[np.ix_(sel, sel)]
    y = np.concatenate([xi0[idx], xi2[idx]])

    # For plotting: diag σ from the corresponding sub-blocks.
    cov00 = cov[: idx.size, : idx.size]
    cov22 = cov[idx.size :, idx.size :]
    sig0 = np.sqrt(np.maximum(0.0, np.diag(cov00)))
    sig2 = np.sqrt(np.maximum(0.0, np.diag(cov22)))

    return {
        "idx": idx,
        "s": s[idx],
        "y": y,
        "cov": cov,
        "sig0": sig0,
        "sig2": sig2,
        "cov00": cov00,
        "cov22": cov22,
    }


def _smooth_basis_labels(*, smooth_power_max: int) -> List[str]:
    pmax = int(smooth_power_max)
    # 条件分岐: `pmax < 0` を満たす経路を評価する。
    if pmax < 0:
        raise ValueError("smooth_power_max must be >= 0")

    labels = ["1"]
    for p in range(1, pmax + 1):
        # 条件分岐: `p == 1` を満たす経路を評価する。
        if p == 1:
            labels.append("1/r")
        else:
            labels.append(f"1/r^{p}")

    return labels


def _n_basis_terms(*, smooth_power_max: int) -> int:
    # constant + inv_s^1..inv_s^p + peak
    pmax = int(smooth_power_max)
    # 条件分岐: `pmax < 0` を満たす経路を評価する。
    if pmax < 0:
        raise ValueError("smooth_power_max must be >= 0")

    return 1 + pmax + 1


def _n_linear_params(*, smooth_power_max: int, n_components: int = 1) -> int:
    # xi0_true and xi2_true each use the same basis; repeated per component.
    n_components = int(n_components)
    # 条件分岐: `n_components <= 0` を満たす経路を評価する。
    if n_components <= 0:
        raise ValueError("n_components must be >= 1")

    return n_components * 2 * _n_basis_terms(smooth_power_max=smooth_power_max)


def _design_matrix(
    s_fid: np.ndarray,
    *,
    alpha: float,
    eps: float,
    r0_mpc_h: float,
    sigma_mpc_h: float,
    mu: np.ndarray,
    w: np.ndarray,
    p2_fid: np.ndarray,
    sqrt1mu2: np.ndarray,
    smooth_power_max: int = 2,
    n_components: int = 1,
) -> np.ndarray:
    """
    Build design matrix for y=[xi0(s), xi2(s)] (fid coordinate) given:
      - true multipoles xi0_true(s), xi2_true(s) parameterized as (smooth + peak)
      - AP warp (alpha, eps) mapping fid -> true
      - only ℓ=0,2 kept in the true ξ(s,μ) expansion

    Linear parameters:
      - xi0_true smooth: [c0, c1/s, ..., c_p/s^p]  (p = smooth_power_max)
      - xi0_true peak amplitude: A0
      - xi2_true smooth: [d0, d1/s, ..., d_p/s^p]  (p = smooth_power_max)
      - xi2_true peak amplitude: A2

    Nonlinear parameters:
      - alpha, eps (AP warp)
      - r0, sigma (peak location/width in template coordinates; r_d free is absorbed in alpha)
    """
    n_components = int(n_components)
    # 条件分岐: `n_components <= 0` を満たす経路を評価する。
    if n_components <= 0:
        raise ValueError("n_components must be >= 1")

    pmax = int(smooth_power_max)
    # 条件分岐: `pmax < 0` を満たす経路を評価する。
    if pmax < 0:
        raise ValueError("smooth_power_max must be >= 0")

    s_fid = np.asarray(s_fid, dtype=float)
    n = int(s_fid.size)
    # 条件分岐: `n == 0` を満たす経路を評価する。
    if n == 0:
        return np.zeros((0, 0), dtype=float)

    one_p_eps = 1.0 + float(eps)
    # 条件分岐: `not (one_p_eps > 0.0 and math.isfinite(one_p_eps))` を満たす経路を評価する。
    if not (one_p_eps > 0.0 and math.isfinite(one_p_eps)):
        raise ValueError("invalid eps (requires 1+eps>0)")

    alpha = float(alpha)
    # 条件分岐: `not (alpha > 0.0 and math.isfinite(alpha))` を満たす経路を評価する。
    if not (alpha > 0.0 and math.isfinite(alpha)):
        raise ValueError("invalid alpha")

    # Parameterization: alpha = (α⊥^2 α∥)^(1/3), 1+eps = (α∥/α⊥)^(1/3)

    alpha_perp = alpha / one_p_eps
    alpha_par = alpha * (one_p_eps**2)

    # mu_true is independent of s (depends only on warp and mu_fid).
    t = np.sqrt((alpha_par * mu) ** 2 + (alpha_perp * sqrt1mu2) ** 2)  # shape (m,)
    # 条件分岐: `np.any(t <= 0.0)` を満たす経路を評価する。
    if np.any(t <= 0.0):
        raise ValueError("invalid AP warp (t<=0)")

    mu_true = (alpha_par * mu) / t
    p2_true = _p2(mu_true)

    w0 = w
    w_p2 = w * p2_fid
    w_p2_true = w * p2_true
    w_p2_true_p2 = w * p2_true * p2_fid

    # s_true = s_fid * t(mu)
    s_true = s_fid[:, None] * t[None, :]

    inv_s = 1.0 / s_true
    inv_pows: List[np.ndarray] = []
    inv_cur = inv_s
    for p in range(1, pmax + 1):
        # 条件分岐: `p == 1` を満たす経路を評価する。
        if p == 1:
            inv_pows.append(inv_s)
        else:
            inv_cur = inv_cur * inv_s
            inv_pows.append(inv_cur)

    peak = np.exp(-0.5 * ((s_true - float(r0_mpc_h)) / float(sigma_mpc_h)) ** 2)

    # Basis list (smooth + peak) in true coordinates.
    # b0 is constant; avoid allocating a full ones matrix.
    sum_w0 = float(np.sum(w0))
    sum_wp2 = float(np.sum(w_p2))
    sum_wp2_true = float(np.sum(w_p2_true))
    sum_wp2_true_p2 = float(np.sum(w_p2_true_p2))

    i00_b0 = np.full(n, 0.5 * sum_w0, dtype=float)
    i20_b0 = np.full(n, 2.5 * sum_wp2, dtype=float)
    i02_b0 = np.full(n, 0.5 * sum_wp2_true, dtype=float)
    i22_b0 = np.full(n, 2.5 * sum_wp2_true_p2, dtype=float)

    def integrate(f: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        i00 = 0.5 * np.sum(f * w0[None, :], axis=1)
        i20 = 2.5 * np.sum(f * w_p2[None, :], axis=1)
        i02 = 0.5 * np.sum(f * w_p2_true[None, :], axis=1)
        i22 = 2.5 * np.sum(f * w_p2_true_p2[None, :], axis=1)
        return i00, i20, i02, i22

    i00_pk, i20_pk, i02_pk, i22_pk = integrate(peak)

    cols: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = [(i00_b0, i20_b0, i02_b0, i22_b0)]
    for term in inv_pows:
        cols.append(integrate(term))

    cols.append((i00_pk, i20_pk, i02_pk, i22_pk))

    n_basis = len(cols)
    n_lin = 2 * n_basis

    # Assemble M for y=[xi0, xi2] with 2*n_basis linear params (single component).
    mtx1 = np.zeros((2 * n, n_lin), dtype=float)
    # Column layout: [xi0:basis..., xi2:basis...]
    for k, (i00, i20, i02, i22) in enumerate(cols):
        # xi0_true basis -> contributes to xi0_fid and xi2_fid via P2(mu_fid)
        mtx1[:n, k] = i00
        mtx1[n:, k] = i20
        # xi2_true basis -> contributes via P2(mu_true) and mixing
        mtx1[:n, k + n_basis] = i02
        mtx1[n:, k + n_basis] = i22

    # 条件分岐: `n_components == 1` を満たす経路を評価する。

    if n_components == 1:
        return mtx1

    # Block-diagonal repetition for multi-component joint fit:
    # y ordering is assumed to be component-major blocks:
    #   [comp0: xi0(s0..), xi2(s0..), comp1: xi0(s0..), xi2(s0..), ...]

    mtx = np.zeros((2 * n * n_components, n_lin * n_components), dtype=float)
    for c in range(n_components):
        r0 = int(c * 2 * n)
        c0 = int(c * n_lin)
        mtx[r0 : r0 + 2 * n, c0 : c0 + n_lin] = mtx1

    return mtx


def _gls_fit(*, y: np.ndarray, mtx: np.ndarray, cov_inv: np.ndarray) -> Dict[str, Any]:
    a = mtx.T @ cov_inv @ mtx
    b = mtx.T @ cov_inv @ y
    try:
        x = np.linalg.solve(a, b)
    except np.linalg.LinAlgError:
        x = np.linalg.lstsq(a, b, rcond=None)[0]

    y_pred = mtx @ x
    r = y - y_pred
    chi2 = float(r.T @ cov_inv @ r)
    return {"x": x, "y_pred": y_pred, "residual": r, "chi2": chi2}


def _scan_grid(
    *,
    y: np.ndarray,
    cov_inv: np.ndarray,
    s_fid: np.ndarray,
    alpha_grid: np.ndarray,
    eps_grid: np.ndarray,
    r0_mpc_h: float,
    sigma_mpc_h: float,
    mu: np.ndarray,
    w: np.ndarray,
    p2_fid: np.ndarray,
    sqrt1mu2: np.ndarray,
    smooth_power_max: int = 2,
    n_components: int = 1,
    return_eps_profile: bool = False,
    return_alpha_profile: bool = False,
) -> Dict[str, Any]:
    best: Dict[str, Any] = {"chi2": float("inf")}
    profile_eps: Dict[float, Dict[str, float]] = {float(eps): {"chi2": float("inf"), "alpha": float("nan")} for eps in eps_grid}
    profile_alpha: Dict[float, Dict[str, float]] = {float(alpha): {"chi2": float("inf"), "eps": float("nan")} for alpha in alpha_grid}
    for alpha in alpha_grid:
        for eps in eps_grid:
            try:
                mtx = _design_matrix(
                    s_fid,
                    alpha=float(alpha),
                    eps=float(eps),
                    r0_mpc_h=r0_mpc_h,
                    sigma_mpc_h=sigma_mpc_h,
                    mu=mu,
                    w=w,
                    p2_fid=p2_fid,
                    sqrt1mu2=sqrt1mu2,
                    smooth_power_max=smooth_power_max,
                    n_components=int(n_components),
                )
            except Exception:
                continue

            fit = _gls_fit(y=y, mtx=mtx, cov_inv=cov_inv)
            # 条件分岐: `fit["chi2"] < best["chi2"]` を満たす経路を評価する。
            if fit["chi2"] < best["chi2"]:
                best = {"alpha": float(alpha), "eps": float(eps), **fit}

            # 条件分岐: `return_eps_profile` を満たす経路を評価する。

            if return_eps_profile:
                eps_key = float(eps)
                # 条件分岐: `float(fit["chi2"]) < float(profile_eps[eps_key]["chi2"])` を満たす経路を評価する。
                if float(fit["chi2"]) < float(profile_eps[eps_key]["chi2"]):
                    profile_eps[eps_key] = {"chi2": float(fit["chi2"]), "alpha": float(alpha)}

            # 条件分岐: `return_alpha_profile` を満たす経路を評価する。

            if return_alpha_profile:
                alpha_key = float(alpha)
                # 条件分岐: `float(fit["chi2"]) < float(profile_alpha[alpha_key]["chi2"])` を満たす経路を評価する。
                if float(fit["chi2"]) < float(profile_alpha[alpha_key]["chi2"]):
                    profile_alpha[alpha_key] = {"chi2": float(fit["chi2"]), "eps": float(eps)}

    # 条件分岐: `not math.isfinite(float(best.get("chi2", float("inf"))))` を満たす経路を評価する。

    if not math.isfinite(float(best.get("chi2", float("inf")))):
        raise RuntimeError("grid search failed (no finite chi2)")

    # 条件分岐: `return_eps_profile` を満たす経路を評価する。

    if return_eps_profile:
        eps_sorted = sorted(profile_eps.keys())
        best["eps_profile"] = [
            {"eps": float(eps), "chi2": float(profile_eps[eps]["chi2"]), "alpha": float(profile_eps[eps]["alpha"])}
            for eps in eps_sorted
        ]

    # 条件分岐: `return_alpha_profile` を満たす経路を評価する。

    if return_alpha_profile:
        alpha_sorted = sorted(profile_alpha.keys())
        best["alpha_profile"] = [
            {"alpha": float(alpha), "chi2": float(profile_alpha[alpha]["chi2"]), "eps": float(profile_alpha[alpha]["eps"])}
            for alpha in alpha_sorted
        ]

    return best


def _predict_curve(
    *,
    s_grid: np.ndarray,
    x: np.ndarray,
    alpha: float,
    eps: float,
    r0_mpc_h: float,
    sigma_mpc_h: float,
    mu: np.ndarray,
    w: np.ndarray,
    p2_fid: np.ndarray,
    sqrt1mu2: np.ndarray,
    smooth_power_max: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    mtx = _design_matrix(
        s_grid,
        alpha=alpha,
        eps=eps,
        r0_mpc_h=r0_mpc_h,
        sigma_mpc_h=sigma_mpc_h,
        mu=mu,
        w=w,
        p2_fid=p2_fid,
        sqrt1mu2=sqrt1mu2,
        smooth_power_max=smooth_power_max,
        n_components=max(1, int(x.size // max(1, _n_linear_params(smooth_power_max=smooth_power_max)))),
    )
    y = mtx @ x
    n = int(s_grid.size)
    n_components = int(y.size // max(1, 2 * n))
    # 条件分岐: `n_components <= 1` を満たす経路を評価する。
    if n_components <= 1:
        return y[:n], y[n:]

    yy = y.reshape(n_components, 2 * n)
    xi0 = yy[:, :n]
    xi2 = yy[:, n:]
    return xi0, xi2


def _chi2_block(*, residual: np.ndarray, cov_inv_block: np.ndarray) -> float:
    return float(residual.T @ cov_inv_block @ residual)


def _profile_ci(
    *,
    x: np.ndarray,
    chi2: np.ndarray,
    delta: float,
) -> Tuple[Optional[float], Optional[float]]:
    x = np.asarray(x, dtype=float)
    chi2 = np.asarray(chi2, dtype=float)
    # 条件分岐: `x.size == 0 or chi2.size != x.size` を満たす経路を評価する。
    if x.size == 0 or chi2.size != x.size:
        return None, None

    # 条件分岐: `not np.all(np.isfinite(chi2))` を満たす経路を評価する。

    if not np.all(np.isfinite(chi2)):
        return None, None

    i0 = int(np.nanargmin(chi2))
    chi2_min = float(chi2[i0])
    target = chi2_min + float(delta)
    d = chi2 - target
    inside = d <= 0.0
    # 条件分岐: `not bool(np.any(inside))` を満たす経路を評価する。
    if not bool(np.any(inside)):
        return None, None

    # Find contiguous inside segment that contains the minimum.

    left = i0
    while left - 1 >= 0 and bool(inside[left - 1]):
        left -= 1

    right = i0
    while right + 1 < int(x.size) and bool(inside[right + 1]):
        right += 1

    def interp_cross(i_out: int, i_in: int) -> Optional[float]:
        x0 = float(x[i_out])
        x1 = float(x[i_in])
        d0 = float(d[i_out])
        d1 = float(d[i_in])
        # 条件分岐: `not (math.isfinite(d0) and math.isfinite(d1))` を満たす経路を評価する。
        if not (math.isfinite(d0) and math.isfinite(d1)):
            return None

        # 条件分岐: `d0 == d1` を満たす経路を評価する。

        if d0 == d1:
            return None

        t = (0.0 - d0) / (d1 - d0)
        return x0 + t * (x1 - x0)

    lo: Optional[float]
    hi: Optional[float]

    # 条件分岐: `left == 0` を満たす経路を評価する。
    if left == 0:
        lo = float(x[0])
    else:
        lo = interp_cross(left - 1, left)
        # 条件分岐: `lo is None` を満たす経路を評価する。
        if lo is None:
            lo = float(x[left])

    # 条件分岐: `right == int(x.size) - 1` を満たす経路を評価する。

    if right == int(x.size) - 1:
        hi = float(x[-1])
    else:
        hi = interp_cross(right + 1, right)
        # 条件分岐: `hi is None` を満たす経路を評価する。
        if hi is None:
            hi = float(x[right])

    return lo, hi


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Cosmology: BAO peak fit from BOSS DR12 ξℓ (Ross 2016, post-recon).")
    ap.add_argument(
        "--dataset",
        choices=["ross_post", "satpathy_pre"],
        default="ross_post",
        help="Input dataset (default: ross_post)",
    )
    ap.add_argument(
        "--data-dir",
        type=str,
        default="",
        help="Dataset directory override (default depends on --dataset)",
    )
    ap.add_argument("--bincent", type=int, default=0, help="bin center shift index (0..4, default: 0)")
    ap.add_argument("--z-bins", type=str, default="1,2,3", help="z bins to use (default: 1,2,3)")
    ap.add_argument("--r-min", type=float, default=50.0, help="min separation r [Mpc/h] for fit (default: 50)")
    ap.add_argument("--r-max", type=float, default=150.0, help="max separation r [Mpc/h] for fit (default: 150)")
    ap.add_argument("--mu-n", type=int, default=80, help="Gauss-Legendre points for μ integral (default: 80)")
    ap.add_argument("--r0", type=float, default=105.0, help="BAO peak center in template coordinates [Mpc/h] (default: 105)")
    ap.add_argument("--sigma", type=float, default=10.0, help="BAO peak width [Mpc/h] (default: 10)")
    ap.add_argument(
        "--smooth-power-max",
        type=int,
        default=2,
        help="smooth basis max power p for 1/r^p (default: 2 => [1, 1/r, 1/r^2])",
    )
    ap.add_argument("--alpha-min", type=float, default=0.9, help="alpha grid min (default: 0.9)")
    ap.add_argument("--alpha-max", type=float, default=1.1, help="alpha grid max (default: 1.1)")
    ap.add_argument("--alpha-step", type=float, default=0.002, help="alpha grid step (default: 0.002)")
    ap.add_argument("--eps-min", type=float, default=-0.05, help="eps grid min (default: -0.05)")
    ap.add_argument("--eps-max", type=float, default=0.05, help="eps grid max (default: 0.05)")
    ap.add_argument("--eps-step", type=float, default=0.002, help="eps grid step (default: 0.002)")
    ap.add_argument("--lcdm-omega-m", type=float, default=0.315, help="Reference flat LCDM Ωm for eps_pred (default: 0.315)")
    ap.add_argument("--lcdm-n-grid", type=int, default=4000, help="z-integral grid points for LCDM F_AP (default: 4000)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    dataset = str(args.dataset)
    # 条件分岐: `str(args.data_dir).strip()` を満たす経路を評価する。
    if str(args.data_dir).strip():
        data_dir = Path(str(args.data_dir))
    else:
        # 条件分岐: `dataset == "ross_post"` を満たす経路を評価する。
        if dataset == "ross_post":
            data_dir = _ROOT / "data" / "cosmology" / "ross_2016_combineddr12_corrfunc"
        else:
            data_dir = _ROOT / "data" / "cosmology" / "satpathy_2016_combineddr12_fs_corrfunc_multipoles"

    # 条件分岐: `not data_dir.exists()` を満たす経路を評価する。

    if not data_dir.exists():
        raise SystemExit(f"data dir not found: {data_dir}")

    bincent = int(args.bincent)
    # 条件分岐: `bincent < 0 or bincent > 4` を満たす経路を評価する。
    if bincent < 0 or bincent > 4:
        raise SystemExit("--bincent must be in 0..4")

    z_bins = [int(x.strip()) for x in str(args.z_bins).split(",") if x.strip()]
    z_meta = [z for z in _ZBINS if z.zbin in z_bins]
    # 条件分岐: `not z_meta` を満たす経路を評価する。
    if not z_meta:
        raise SystemExit("no valid z bins selected")

    mu_n = int(args.mu_n)
    # 条件分岐: `mu_n < 20` を満たす経路を評価する。
    if mu_n < 20:
        raise SystemExit("--mu-n must be >= 20")

    mu, w = np.polynomial.legendre.leggauss(mu_n)
    sqrt1mu2 = np.sqrt(np.maximum(0.0, 1.0 - mu * mu))
    p2_fid = _p2(mu)

    alpha_grid = np.arange(float(args.alpha_min), float(args.alpha_max) + 0.5 * float(args.alpha_step), float(args.alpha_step))
    eps_grid = np.arange(float(args.eps_min), float(args.eps_max) + 0.5 * float(args.eps_step), float(args.eps_step))
    # 条件分岐: `alpha_grid.size < 5 or eps_grid.size < 5` を満たす経路を評価する。
    if alpha_grid.size < 5 or eps_grid.size < 5:
        raise SystemExit("grid too small; widen ranges or reduce steps")

    r_min = float(args.r_min)
    r_max = float(args.r_max)
    r0 = float(args.r0)
    sigma = float(args.sigma)
    smooth_power_max = int(args.smooth_power_max)
    # 条件分岐: `smooth_power_max < 0` を満たす経路を評価する。
    if smooth_power_max < 0:
        raise SystemExit("--smooth-power-max must be >= 0")

    omega_m = float(args.lcdm_omega_m)
    lcdm_n_grid = int(args.lcdm_n_grid)

    results: List[Dict[str, Any]] = []
    curves_for_plot: List[Dict[str, Any]] = []

    for zb in z_meta:
        # 条件分岐: `dataset == "ross_post"` を満たす経路を評価する。
        if dataset == "ross_post":
            mono_path = (
                data_dir
                / f"Ross_2016_COMBINEDDR12_zbin{zb.zbin}_correlation_function_monopole_post_recon_bincent{bincent}.dat"
            )
            quad_path = (
                data_dir
                / f"Ross_2016_COMBINEDDR12_zbin{zb.zbin}_correlation_function_quadrupole_post_recon_bincent{bincent}.dat"
            )
            cov_path = (
                data_dir / f"Ross_2016_COMBINEDDR12_zbin{zb.zbin}_covariance_monoquad_post_recon_bincent{bincent}.dat"
            )
            # 条件分岐: `not (mono_path.exists() and quad_path.exists() and cov_path.exists())` を満たす経路を評価する。
            if not (mono_path.exists() and quad_path.exists() and cov_path.exists()):
                raise SystemExit(f"missing Ross files for bin{zb.zbin} bincent{bincent}: {data_dir}")

            s0, xi0, _ = _read_table(mono_path)
            s2, xi2, _ = _read_table(quad_path)
            # 条件分岐: `s0.shape != s2.shape or np.max(np.abs(s0 - s2)) > 1e-9` を満たす経路を評価する。
            if s0.shape != s2.shape or np.max(np.abs(s0 - s2)) > 1e-9:
                raise SystemExit(f"s bins mismatch between mono/quad for bin{zb.zbin}")

            cov_full = _read_cov(cov_path)
            sub = _subset_monoquad(s=s0, xi0=xi0, xi2=xi2, cov_monoquad=cov_full, r_min=r_min, r_max=r_max)
        else:
            # 条件分岐: `bincent != 0` を満たす経路を評価する。
            if bincent != 0:
                raise SystemExit("satpathy_pre does not support --bincent (set to 0)")

            mono_path = data_dir / f"Satpathy_2016_COMBINEDDR12_Bin{zb.zbin}_Monopole_pre_recon.dat"
            quad_path = data_dir / f"Satpathy_2016_COMBINEDDR12_Bin{zb.zbin}_Quadrupole_pre_recon.dat"
            cov_name = (
                f"Satpathy_2016_COMBINEDDR12_Bin{zb.zbin}_Covariance_pre_recon.txt"
                if zb.zbin in (1, 2)
                else f"Satpathy_2016_COMBINEDDR12_Bin{zb.zbin}_CovarianceMatrix_pre_recon.txt"
            )
            cov_path = data_dir / cov_name
            # 条件分岐: `not (mono_path.exists() and quad_path.exists() and cov_path.exists())` を満たす経路を評価する。
            if not (mono_path.exists() and quad_path.exists() and cov_path.exists()):
                raise SystemExit(f"missing Satpathy files for bin{zb.zbin}: {data_dir}")

            cov_full = _read_satpathy_cov(cov_path)
            s_sel = _satpathy_s_bins_from_cov(cov_full)
            s0_all, xi0_all = _read_table_two_cols(mono_path)
            s2_all, xi2_all = _read_table_two_cols(quad_path)
            xi0 = _select_by_exact_s(s0_all, xi0_all, s_sel)
            xi2 = _select_by_exact_s(s2_all, xi2_all, s_sel)
            sub = _subset_monoquad(s=s_sel, xi0=xi0, xi2=xi2, cov_monoquad=cov_full, r_min=r_min, r_max=r_max)

        s = np.asarray(sub["s"], dtype=float)
        y = np.asarray(sub["y"], dtype=float)
        cov = np.asarray(sub["cov"], dtype=float)

        # Invert covariance once per z-bin.
        try:
            cov_inv = np.linalg.inv(cov)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov, rcond=1e-12)

        cov_inv = 0.5 * (cov_inv + cov_inv.T)

        # Fit 1) free (alpha, eps)
        best_free = _scan_grid(
            y=y,
            cov_inv=cov_inv,
            s_fid=s,
            alpha_grid=alpha_grid,
            eps_grid=eps_grid,
            r0_mpc_h=r0,
            sigma_mpc_h=sigma,
            mu=mu,
            w=w,
            p2_fid=p2_fid,
            sqrt1mu2=sqrt1mu2,
            smooth_power_max=smooth_power_max,
            return_eps_profile=True,
        )

        # Fit 2) eps fixed to 0 (fid-like geometry)
        best_eps0 = _scan_grid(
            y=y,
            cov_inv=cov_inv,
            s_fid=s,
            alpha_grid=alpha_grid,
            eps_grid=np.asarray([0.0], dtype=float),
            r0_mpc_h=r0,
            sigma_mpc_h=sigma,
            mu=mu,
            w=w,
            p2_fid=p2_fid,
            sqrt1mu2=sqrt1mu2,
            smooth_power_max=smooth_power_max,
            return_eps_profile=False,
        )

        # Fit 3) eps fixed to P_bg(static exponential) predicted warping vs LCDM fid (r_d cancels).
        f_ap_pbg = _f_ap_pbg_exponential(zb.z_eff)
        f_ap_fid = _f_ap_lcdm_flat(zb.z_eff, omega_m=omega_m, n_grid=lcdm_n_grid)
        eps_pbg = _eps_from_f_ap_ratio(f_ap_model=f_ap_pbg, f_ap_fid=f_ap_fid)
        best_eps_pbg = _scan_grid(
            y=y,
            cov_inv=cov_inv,
            s_fid=s,
            alpha_grid=alpha_grid,
            eps_grid=np.asarray([eps_pbg], dtype=float),
            r0_mpc_h=r0,
            sigma_mpc_h=sigma,
            mu=mu,
            w=w,
            p2_fid=p2_fid,
            sqrt1mu2=sqrt1mu2,
            smooth_power_max=smooth_power_max,
            return_eps_profile=False,
        )

        # Diagnostics: mono-only / quad-only chi2 (ignore cross-cov; used only as a rough indicator).
        n = int(s.size)
        cov00_inv = np.linalg.pinv(np.asarray(sub["cov00"]), rcond=1e-12)
        cov22_inv = np.linalg.pinv(np.asarray(sub["cov22"]), rcond=1e-12)
        res_free = np.asarray(best_free["residual"], dtype=float)
        chi2_0 = _chi2_block(residual=res_free[:n], cov_inv_block=cov00_inv)
        chi2_2 = _chi2_block(residual=res_free[n:], cov_inv_block=cov22_inv)

        n_linear = _n_linear_params(smooth_power_max=smooth_power_max)
        dof_free = int(y.size - (n_linear + 2))
        dof_eps_fixed = int(y.size - (n_linear + 1))
        eps_profile = list(best_free.get("eps_profile") or [])
        eps_grid_vals = np.array([float(r["eps"]) for r in eps_profile], dtype=float) if eps_profile else np.array([], dtype=float)
        chi2_prof = np.array([float(r["chi2"]) for r in eps_profile], dtype=float) if eps_profile else np.array([], dtype=float)
        eps_ci_1, eps_ci_2 = _profile_ci(x=eps_grid_vals, chi2=chi2_prof, delta=1.0)
        eps_ci_2s, eps_ci_2s_hi = _profile_ci(x=eps_grid_vals, chi2=chi2_prof, delta=4.0)
        rec = {
            "zbin": zb.zbin,
            "z_eff": zb.z_eff,
            "label": zb.label,
            "fit_range_mpc_h": [r_min, r_max],
            "template_peak": {"r0_mpc_h": r0, "sigma_mpc_h": sigma},
            "fid_reference": {"lcdm_flat": {"Omega_m": omega_m, "F_AP": f_ap_fid}},
            "model_reference": {"P_bg_exponential": {"F_AP": f_ap_pbg, "eps_pred": eps_pbg}},
            "fit": {
                "free": {
                    "alpha": float(best_free["alpha"]),
                    "eps": float(best_free["eps"]),
                    "chi2": float(best_free["chi2"]),
                    "dof": dof_free,
                    "chi2_dof": float(best_free["chi2"]) / float(dof_free) if dof_free > 0 else float("nan"),
                    "eps_ci_1sigma": [eps_ci_1, eps_ci_2],
                    "eps_ci_2sigma": [eps_ci_2s, eps_ci_2s_hi],
                },
                "eps_fixed_0": {
                    "alpha": float(best_eps0["alpha"]),
                    "eps": 0.0,
                    "chi2": float(best_eps0["chi2"]),
                    "dof": dof_eps_fixed,
                    "chi2_dof": float(best_eps0["chi2"]) / float(dof_eps_fixed) if dof_eps_fixed > 0 else float("nan"),
                },
                "eps_fixed_pbg": {
                    "alpha": float(best_eps_pbg["alpha"]),
                    "eps": float(eps_pbg),
                    "chi2": float(best_eps_pbg["chi2"]),
                    "dof": dof_eps_fixed,
                    "chi2_dof": float(best_eps_pbg["chi2"]) / float(dof_eps_fixed) if dof_eps_fixed > 0 else float("nan"),
                },
            },
            "diagnostics": {
                "chi2_mono_only_free": chi2_0,
                "chi2_quad_only_free": chi2_2,
            },
        }
        results.append(rec)

        # Curves for plotting (use each case's linear coeffs).
        s_grid = np.linspace(r_min, r_max, 400, dtype=float)
        xi0_free, xi2_free = _predict_curve(
            s_grid=s_grid,
            x=np.asarray(best_free["x"], dtype=float),
            alpha=float(best_free["alpha"]),
            eps=float(best_free["eps"]),
            r0_mpc_h=r0,
            sigma_mpc_h=sigma,
            mu=mu,
            w=w,
            p2_fid=p2_fid,
            sqrt1mu2=sqrt1mu2,
            smooth_power_max=smooth_power_max,
        )
        xi0_eps0, xi2_eps0 = _predict_curve(
            s_grid=s_grid,
            x=np.asarray(best_eps0["x"], dtype=float),
            alpha=float(best_eps0["alpha"]),
            eps=0.0,
            r0_mpc_h=r0,
            sigma_mpc_h=sigma,
            mu=mu,
            w=w,
            p2_fid=p2_fid,
            sqrt1mu2=sqrt1mu2,
            smooth_power_max=smooth_power_max,
        )
        xi0_pbg, xi2_pbg = _predict_curve(
            s_grid=s_grid,
            x=np.asarray(best_eps_pbg["x"], dtype=float),
            alpha=float(best_eps_pbg["alpha"]),
            eps=float(eps_pbg),
            r0_mpc_h=r0,
            sigma_mpc_h=sigma,
            mu=mu,
            w=w,
            p2_fid=p2_fid,
            sqrt1mu2=sqrt1mu2,
            smooth_power_max=smooth_power_max,
        )
        curves_for_plot.append(
            {
                "zbin": zb.zbin,
                "z_eff": zb.z_eff,
                "label": zb.label,
                "s_data": s,
                "xi0_data": y[:n],
                "xi2_data": y[n:],
                "sig0": np.asarray(sub["sig0"], dtype=float),
                "sig2": np.asarray(sub["sig2"], dtype=float),
                "s_grid": s_grid,
                "xi0_free": xi0_free,
                "xi2_free": xi2_free,
                "xi0_eps0": xi0_eps0,
                "xi2_eps0": xi2_eps0,
                "xi0_pbg": xi0_pbg,
                "xi2_pbg": xi2_pbg,
                "fit_annot": {
                    "free": {"alpha": float(best_free["alpha"]), "eps": float(best_free["eps"]), "chi2": float(best_free["chi2"])},
                    "eps0": {"alpha": float(best_eps0["alpha"]), "chi2": float(best_eps0["chi2"])},
                    "pbg": {"alpha": float(best_eps_pbg["alpha"]), "eps": float(eps_pbg), "chi2": float(best_eps_pbg["chi2"])},
                    "dof_free": dof_free,
                    "dof_eps_fixed": dof_eps_fixed,
                },
            }
        )

    out_dir = _ROOT / "output" / "private" / "cosmology"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if dataset == "ross_post" else "_pre_recon"
    out_png = out_dir / f"cosmology_bao_xi_multipole_peakfit{suffix}.png"
    out_json = out_dir / f"cosmology_bao_xi_multipole_peakfit{suffix}_metrics.json"

    # Plot
    _set_japanese_font()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(curves_for_plot), 2, figsize=(16, 4.8 * len(curves_for_plot)), sharex=True)
    # 条件分岐: `len(curves_for_plot) == 1` を満たす経路を評価する。
    if len(curves_for_plot) == 1:
        axes = np.array(axes).reshape(1, 2)

    for row, pack in enumerate(curves_for_plot):
        ax0 = axes[row, 0]
        ax2 = axes[row, 1]

        # Monopole
        ax0.errorbar(
            pack["s_data"],
            pack["xi0_data"],
            yerr=pack["sig0"],
            fmt="o",
            ms=4,
            capsize=2,
            color="#111111",
            ecolor="#111111",
            label="観測 ξ0（post-recon; Ross 2016）",
        )
        ax0.plot(pack["s_grid"], pack["xi0_free"], color="#1f77b4", linewidth=2.0, label="fit: α,ε free")
        ax0.plot(pack["s_grid"], pack["xi0_eps0"], color="#777777", linewidth=1.6, linestyle="--", label="fit: ε=0")
        ax0.plot(pack["s_grid"], pack["xi0_pbg"], color="#ff7f0e", linewidth=1.8, linestyle="--", label="fit: ε=ε(P_bg)")
        ax0.set_ylabel("ξ0", fontsize=11)
        ax0.grid(True, linestyle="--", alpha=0.5)
        ax0.set_title(f"ξ0（{pack['label']}）", fontsize=12)

        # Quadrupole
        ax2.errorbar(
            pack["s_data"],
            pack["xi2_data"],
            yerr=pack["sig2"],
            fmt="o",
            ms=4,
            capsize=2,
            color="#111111",
            ecolor="#111111",
            label="観測 ξ2（post-recon; Ross 2016）",
        )
        ax2.plot(pack["s_grid"], pack["xi2_free"], color="#1f77b4", linewidth=2.0, label="fit: α,ε free")
        ax2.plot(pack["s_grid"], pack["xi2_eps0"], color="#777777", linewidth=1.6, linestyle="--", label="fit: ε=0")
        ax2.plot(pack["s_grid"], pack["xi2_pbg"], color="#ff7f0e", linewidth=1.8, linestyle="--", label="fit: ε=ε(P_bg)")
        ax2.set_ylabel("ξ2", fontsize=11)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.set_title(f"ξ2（{pack['label']}）", fontsize=12)

        # Annotation box (use the quadrupole panel; focus on anisotropy)
        ann = pack["fit_annot"]
        dof_free = int(ann.get("dof_free", 0))
        dof_fixed = int(ann.get("dof_eps_fixed", 0))
        chi2d_free = ann["free"]["chi2"] / dof_free if dof_free > 0 else float("nan")
        chi2d_eps0 = ann["eps0"]["chi2"] / dof_fixed if dof_fixed > 0 else float("nan")
        chi2d_pbg = ann["pbg"]["chi2"] / dof_fixed if dof_fixed > 0 else float("nan")
        text = (
            f"range: r∈[{r_min:.0f},{r_max:.0f}] Mpc/h\n"
            f"free:   α={ann['free']['alpha']:.4f}, ε={ann['free']['eps']:+.4f}, χ²/dof={chi2d_free:.2f}\n"
            f"ε=0:    α={ann['eps0']['alpha']:.4f}, χ²/dof={chi2d_eps0:.2f}\n"
            f"P_bg:   ε={ann['pbg']['eps']:+.4f}, α={ann['pbg']['alpha']:.4f}, χ²/dof={chi2d_pbg:.2f}"
        )
        ax2.text(
            0.02,
            0.98,
            text,
            transform=ax2.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#999999", alpha=0.92),
        )

        # 条件分岐: `row == 0` を満たす経路を評価する。
        if row == 0:
            ax0.legend(fontsize=9, loc="upper right")
            ax2.legend(fontsize=9, loc="upper right")

    for ax in axes[-1, :]:
        ax.set_xlabel("分離 r [Mpc/h]（fid）", fontsize=11)

    fig.suptitle("宇宙論（BAO一次統計）：BOSS DR12 post-recon ξℓ（ℓ=0,2）のピークfit（smooth+peak）", fontsize=14)
    fig.text(
        0.5,
        0.01,
        "注：これは圧縮出力（D_M/r_d 等）ではなく ξℓ からの“入口”であり、再導出（ξ(s,μ)/P(k)）の本番は次工程。",
        ha="center",
        fontsize=10,
    )
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.94))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "domain": "cosmology",
        "step": "16.4 (BAO re-derivation entry: xi multipoles peak fit)",
        "dataset": dataset,
        "inputs": {
            "ross_corrfunc_dir": str(data_dir).replace("\\", "/"),
            "bincent": bincent,
            "z_bins": [z.zbin for z in z_meta],
        },
        "fit": {
            "range_mpc_h": [r_min, r_max],
            "mu_n": mu_n,
            "basis": {"smooth": _smooth_basis_labels(smooth_power_max=smooth_power_max), "peak": "Gaussian(r0,sigma)"},
            "smooth_power_max": int(smooth_power_max),
            "nonlinear_params": ["alpha", "eps"],
            "template_peak": {"r0_mpc_h": r0, "sigma_mpc_h": sigma},
        },
        "notes": [
            "P_bg の ε(pred) は、距離比（r_d を含む）ではなく F_AP=D_M H / c の比から算出（r_d が相殺）。",
            "mono-only / quad-only χ² は cross-cov を無視した粗い指標（ℓ=2強調の目安）。",
        ],
        "results": results,
        "outputs": {"png": str(out_png).replace("\\", "/"), "metrics_json": str(out_json).replace("\\", "/")},
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_bao_xi_multipole_peakfit" if dataset == "ross_post" else "cosmology_bao_xi_multipole_peakfit_pre_recon",
                "argv": list(sys.argv),
                "inputs": {"dataset": dataset, "data_dir": data_dir, "bincent": bincent, "z_bins": [z.zbin for z in z_meta]},
                "outputs": {"png": out_png, "metrics_json": out_json},
                "metrics": {"range_mpc_h": [r_min, r_max], "mu_n": mu_n},
            }
        )
    except Exception:
        pass

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
