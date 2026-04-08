#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_pantheonplus_cc_direct_fit.py

目的:
  Pantheon+（距離モジュラス μ(z)）と Cosmic Chronometers（H(z)）に対して、
  ΛCDM と P-model（H_eff^2 完全式）を同一I/Fで直接比較し、
  χ^2 と ΔAIC を固定出力する。

入力（既定）:
  - data/cosmology/pantheonplus_sh0es.dat
  - data/cosmology/sources/cosmic_chronometers_data_CC.dat

出力（固定名）:
  - output/private/cosmology/cosmology_pantheonplus_cc_direct_fit.{pdf,png}
  - output/private/cosmology/cosmology_pantheonplus_cc_direct_fit_metrics.json
  - output/private/cosmology/cosmology_pantheonplus_cc_direct_fit_summary.csv
  - output/public/cosmology/cosmology_pantheonplus_cc_direct_fit.{pdf,png}
  - output/public/cosmology/cosmology_pantheonplus_cc_direct_fit_metrics.json
  - output/public/cosmology/cosmology_pantheonplus_cc_direct_fit_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

# 条件分岐: `True` を満たす経路を評価する。
try:
    from scipy.optimize import minimize
except Exception:
    minimize = None

_ROOT = Path(__file__).resolve().parents[2]

# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


_C_KM_S = 299_792.458


# クラス: `SNData` の責務と境界条件を定義する。
@dataclass
class SNData:
    """Pantheon+ の直接比較に使う SN 入力（単位: z 無次元, μ mag, σ_μ mag）。"""

    z: np.ndarray
    mu: np.ndarray
    sigma_mu: np.ndarray


# クラス: `CCData` の責務と境界条件を定義する。

@dataclass
class CCData:
    """Cosmic Chronometers 入力（単位: z 無次元, H km/s/Mpc, σ_H km/s/Mpc）。"""

    z: np.ndarray
    hz: np.ndarray
    sigma_hz: np.ndarray
    references: list[str]


# クラス: `FitResult` の責務と境界条件を定義する。

@dataclass
class FitResult:
    """1モデル分の最小化結果（χ², AIC, best-fit パラメータ）を保持する。"""

    label: str
    shape_params: dict[str, float]
    h0_best: float
    delta_mu_best: float
    chi2_sn: float
    chi2_cc: float
    chi2_total: float
    aic: float
    n_params: int


# 関数: `_set_japanese_font` の入出力契約と処理意図を定義する。

def _set_japanese_font() -> None:
    """図内の日本語表示用に利用可能フォントを設定する。失敗時は無変更で継続する。"""

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


# 関数: `_cumtrapz` の入出力契約と処理意図を定義する。

def _cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """台形公式の累積積分を返す。x は単調増加を前提とする。"""

    dx = np.diff(x)
    ym = 0.5 * (y[1:] + y[:-1])
    out = np.zeros_like(x, dtype=float)
    out[1:] = np.cumsum(dx * ym)
    return out


# 関数: `_load_pantheon_plus` の入出力契約と処理意図を定義する。

def _load_pantheon_plus(path: Path, *, z_min: float) -> SNData:
    """
    Pantheon+ 公開テーブルを読み込み、直接比較に使う列を抽出する。

    失敗条件:
      - 必須列（zCMB, MU_SH0ES, MU_SH0ES_ERR_DIAG）が欠ける場合は ValueError。
    """

    df = pd.read_csv(path, sep=r"\s+")
    required = ["zCMB", "MU_SH0ES", "MU_SH0ES_ERR_DIAG"]
    missing = [c for c in required if c not in df.columns]

    # 条件分岐: `missing` を満たす経路を評価する。
    if missing:
        raise ValueError(f"Pantheon+ required columns are missing: {missing}")

    z = df["zCMB"].to_numpy(dtype=float)
    mu = df["MU_SH0ES"].to_numpy(dtype=float)
    sigma = df["MU_SH0ES_ERR_DIAG"].to_numpy(dtype=float)
    m = np.isfinite(z) & np.isfinite(mu) & np.isfinite(sigma) & (sigma > 0.0) & (z >= float(z_min))

    # 条件分岐: `not np.any(m)` を満たす経路を評価する。
    if not np.any(m):
        raise ValueError("No valid Pantheon+ rows after filtering.")

    return SNData(z=z[m], mu=mu[m], sigma_mu=sigma[m])


# 関数: `_load_cosmic_chronometers` の入出力契約と処理意図を定義する。

def _load_cosmic_chronometers(path: Path) -> CCData:
    """
    data_CC.dat（z, H, σ_H, reference）を読み込む。

    失敗条件:
      - 数値3列を解釈できない行が全件の場合は ValueError。
    """

    z_list: list[float] = []
    hz_list: list[float] = []
    sig_list: list[float] = []
    refs: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        s = raw.strip()

        # 条件分岐: `not s or s.startswith("#")` を満たす経路を評価する。
        if not s or s.startswith("#"):
            continue

        parts = [p.strip() for p in s.split(",")]

        # 条件分岐: `len(parts) < 4` を満たす経路を評価する。
        if len(parts) < 4:
            continue

        try:
            z_val = float(parts[0])
            h_val = float(parts[1])
            sig_val = float(parts[2])
        except Exception:
            continue

        # 条件分岐: `not (math.isfinite(...) and ... )` を満たす経路を評価する。

        if not (math.isfinite(z_val) and math.isfinite(h_val) and math.isfinite(sig_val) and sig_val > 0.0):
            continue

        z_list.append(z_val)
        hz_list.append(h_val)
        sig_list.append(sig_val)
        refs.append(parts[3])

    # 条件分岐: `not z_list` を満たす経路を評価する。

    if not z_list:
        raise ValueError("No valid cosmic chronometer rows were parsed.")

    return CCData(
        z=np.asarray(z_list, dtype=float),
        hz=np.asarray(hz_list, dtype=float),
        sigma_hz=np.asarray(sig_list, dtype=float),
        references=refs,
    )


# 関数: `_solve_linear_scale` の入出力契約と処理意図を定義する。

def _solve_linear_scale(obs: np.ndarray, basis: np.ndarray, sigma: np.ndarray) -> tuple[float, float]:
    """
    obs ≈ a * basis の重み付き最小二乗解 a と χ² を返す。

    単位:
      - obs, basis は同一単位（ここでは H の km/s/Mpc）。
    """

    w = 1.0 / np.maximum(sigma * sigma, 1e-30)
    denom = float(np.sum(w * basis * basis))

    # 条件分岐: `denom <= 0.0 or not math.isfinite(denom)` を満たす経路を評価する。
    if denom <= 0.0 or not math.isfinite(denom):
        return float("nan"), float("nan")

    scale = float(np.sum(w * obs * basis) / denom)
    res = (obs - scale * basis) / sigma
    chi2 = float(np.sum(res * res))
    return scale, chi2


# 関数: `_solve_mu_offset` の入出力契約と処理意図を定義する。

def _solve_mu_offset(mu_obs: np.ndarray, mu_model: np.ndarray, sigma_mu: np.ndarray) -> tuple[float, float]:
    """
    μ_obs ≈ μ_model + Δμ の重み付き最小二乗解 Δμ と χ² を返す。

    単位:
      - μ, Δμ は mag。
    """

    w = 1.0 / np.maximum(sigma_mu * sigma_mu, 1e-30)
    delta = float(np.sum(w * (mu_obs - mu_model)) / np.sum(w))
    res = (mu_obs - (mu_model + delta)) / sigma_mu
    chi2 = float(np.sum(res * res))
    return delta, chi2


# 関数: `_lcdm_ez` の入出力契約と処理意図を定義する。

def _lcdm_ez(z: np.ndarray, omega_m: float) -> np.ndarray:
    """flat ΛCDM の E(z)=H(z)/H0 を返す。"""

    return np.sqrt(float(omega_m) * np.power(1.0 + z, 3.0) + (1.0 - float(omega_m)))


# 関数: `_lcdm_dl_mpc` の入出力契約と処理意図を定義する。

def _lcdm_dl_mpc(z: np.ndarray, *, omega_m: float, h0: float) -> np.ndarray:
    """
    flat ΛCDM の光度距離 D_L(z) [Mpc] を返す。

    計算:
      D_L = (1+z) * (c/H0) * ∫ dz'/E(z')
    """

    z_max = float(np.max(z))
    z_grid = np.linspace(0.0, max(z_max, 1e-6), 4096)
    inv_e = 1.0 / _lcdm_ez(z_grid, omega_m)
    integ = _cumtrapz(inv_e, z_grid)
    i_of_z = np.interp(z, z_grid, integ)
    return (1.0 + z) * (_C_KM_S / float(h0)) * i_of_z


# 関数: `_pmodel_heff2_e2` の入出力契約と処理意図を定義する。

def _pmodel_heff2_e2(
    z: np.ndarray,
    *,
    alpha_r: float,
    alpha_m: float,
    alpha_lambda: float,
) -> np.ndarray:
    """
    完全式の無次元化 E^2(z)=H_eff^2(z)/H0^2 を返す。

    定義:
      E^2(z) = 1
             + αr * ((1+z)^4 - 1)
             + αm * ((1+z)^3 - 1)
             - αΛ * ln(1+z)
    """

    zp1 = 1.0 + z
    ln_zp1 = np.log(np.maximum(zp1, 1e-12))
    e2 = (
        1.0
        + float(alpha_r) * (np.power(zp1, 4.0) - 1.0)
        + float(alpha_m) * (np.power(zp1, 3.0) - 1.0)
        - float(alpha_lambda) * ln_zp1
    )
    return e2


# 関数: `_pmodel_heff_basis` の入出力契約と処理意図を定義する。

def _pmodel_heff_basis(
    z: np.ndarray,
    *,
    alpha_r: float,
    alpha_m: float,
    alpha_lambda: float,
) -> np.ndarray:
    """完全式 E^2(z) から H(z)=H0*E(z) の基底 E(z) を返す。"""

    e2 = _pmodel_heff2_e2(
        z,
        alpha_r=float(alpha_r),
        alpha_m=float(alpha_m),
        alpha_lambda=float(alpha_lambda),
    )
    safe = np.where(np.isfinite(e2), e2, np.nan)
    return np.sqrt(np.maximum(safe, 0.0))


# 関数: `_pmodel_dl_mpc_full` の入出力契約と処理意図を定義する。

def _pmodel_dl_mpc_full(
    z: np.ndarray,
    *,
    alpha_r: float,
    alpha_m: float,
    alpha_lambda: float,
    h0: float,
) -> np.ndarray:
    """
    完全式 H_eff^2(z) に基づく P-model の光度距離 D_L(z) [Mpc] を返す。

    計算:
      D_L = (1+z) * (c/H0) * ∫ dz'/E(z')
      E(z') = sqrt(E^2(z')).
    """

    z_max = float(np.max(z))
    z_grid = np.linspace(0.0, max(z_max, 1e-6), 4096)
    basis = _pmodel_heff_basis(
        z_grid,
        alpha_r=float(alpha_r),
        alpha_m=float(alpha_m),
        alpha_lambda=float(alpha_lambda),
    )
    inv_basis = 1.0 / np.maximum(basis, 1e-12)
    integ = _cumtrapz(inv_basis, z_grid)
    i_of_z = np.interp(z, z_grid, integ)
    return (1.0 + z) * (_C_KM_S / float(h0)) * i_of_z


# 関数: `_mu_from_dl_mpc` の入出力契約と処理意図を定義する。

def _mu_from_dl_mpc(dl_mpc: np.ndarray) -> np.ndarray:
    """光度距離 D_L [Mpc] から距離モジュラス μ [mag] を計算する。"""

    d = np.maximum(dl_mpc, 1e-12)
    return 5.0 * np.log10(d) + 25.0


# 関数: `_fit_lcdm` の入出力契約と処理意図を定義する。

def _fit_lcdm(omega_m: float, sn: SNData, cc: CCData) -> FitResult:
    """
    指定した Ωm で ΛCDM を評価し、H0・Δμ を解析解で最小化した結果を返す。

    n_params は Ωm, H0, Δμ の3自由度として固定する。
    """

    e_cc = _lcdm_ez(cc.z, omega_m)
    h0_best, chi2_cc = _solve_linear_scale(cc.hz, e_cc, cc.sigma_hz)
    dl = _lcdm_dl_mpc(sn.z, omega_m=omega_m, h0=h0_best)
    mu_th = _mu_from_dl_mpc(dl)
    delta_mu, chi2_sn = _solve_mu_offset(sn.mu, mu_th, sn.sigma_mu)
    chi2_total = float(chi2_sn + chi2_cc)
    n_params = 3
    return FitResult(
        label="flat_LCDM",
        shape_params={"Omega_m": float(omega_m)},
        h0_best=float(h0_best),
        delta_mu_best=float(delta_mu),
        chi2_sn=float(chi2_sn),
        chi2_cc=float(chi2_cc),
        chi2_total=chi2_total,
        aic=float(chi2_total + 2.0 * n_params),
        n_params=n_params,
    )


# 関数: `_fit_pmodel` の入出力契約と処理意図を定義する。

def _fit_pmodel_full(
    alpha_r: float,
    alpha_m: float,
    alpha_lambda: float,
    sn: SNData,
    cc: CCData,
) -> FitResult:
    """
    指定した完全式係数で P-model を評価し、H0・Δμ を解析解で最小化した結果を返す。

    n_params は (αr, αm, αΛ, H0, Δμ) の5自由度として固定する。
    """

    basis_cc = _pmodel_heff_basis(
        cc.z,
        alpha_r=float(alpha_r),
        alpha_m=float(alpha_m),
        alpha_lambda=float(alpha_lambda),
    )

    # 条件分岐: `not np.all(np.isfinite(basis_cc)) or np.any(basis_cc <= 0.0)` を満たす経路を評価する。
    if not np.all(np.isfinite(basis_cc)) or np.any(basis_cc <= 0.0):
        return FitResult(
            label="pmodel_heff2_full",
            shape_params={},
            h0_best=float("nan"),
            delta_mu_best=float("nan"),
            chi2_sn=float("nan"),
            chi2_cc=float("nan"),
            chi2_total=float("nan"),
            aic=float("nan"),
            n_params=5,
        )

    h0_best, chi2_cc = _solve_linear_scale(cc.hz, basis_cc, cc.sigma_hz)

    # 条件分岐: `not math.isfinite(h0_best) or h0_best <= 0.0` を満たす経路を評価する。
    if not math.isfinite(h0_best) or h0_best <= 0.0:
        return FitResult(
            label="pmodel_heff2_full",
            shape_params={},
            h0_best=float("nan"),
            delta_mu_best=float("nan"),
            chi2_sn=float("nan"),
            chi2_cc=float("nan"),
            chi2_total=float("nan"),
            aic=float("nan"),
            n_params=5,
        )

    dl = _pmodel_dl_mpc_full(
        sn.z,
        alpha_r=float(alpha_r),
        alpha_m=float(alpha_m),
        alpha_lambda=float(alpha_lambda),
        h0=h0_best,
    )
    mu_th = _mu_from_dl_mpc(dl)
    delta_mu, chi2_sn = _solve_mu_offset(sn.mu, mu_th, sn.sigma_mu)
    chi2_total = float(chi2_sn + chi2_cc)
    n_params = 5
    shape_params = {
        "alpha_r": float(alpha_r),
        "alpha_m": float(alpha_m),
        "alpha_lambda": float(alpha_lambda),
    }
    shape_params["omega_r_sq"] = float(2.0 * shape_params["alpha_r"] * (h0_best**2))
    shape_params["omega_m_sq"] = float(1.5 * shape_params["alpha_m"] * (h0_best**2))
    shape_params["omega_lambda_sq"] = float(0.5 * shape_params["alpha_lambda"] * (h0_best**2))
    return FitResult(
        label="pmodel_heff2_full",
        shape_params=shape_params,
        h0_best=float(h0_best),
        delta_mu_best=float(delta_mu),
        chi2_sn=float(chi2_sn),
        chi2_cc=float(chi2_cc),
        chi2_total=chi2_total,
        aic=float(chi2_total + 2.0 * n_params),
        n_params=n_params,
    )


# 関数: `_grid_minimize` の入出力契約と処理意図を定義する。

def _grid_minimize(
    *,
    fn: Callable[[float], FitResult],
    p_min: float,
    p_max: float,
    n_grid: int,
) -> FitResult:
    """
    1次元パラメータを等間隔グリッドで探索し、最小 χ² の結果を返す。

    失敗条件:
      - 全点で有限 χ² が得られない場合は ValueError。
    """

    grid = np.linspace(float(p_min), float(p_max), int(n_grid))
    best: FitResult | None = None
    for p in grid:
        r = fn(float(p))

        # 条件分岐: `not math.isfinite(r.chi2_total)` を満たす経路を評価する。
        if not math.isfinite(r.chi2_total):
            continue

        # 条件分岐: `best is None or r.chi2_total < best.chi2_total` を満たす経路を評価する。

        if best is None or r.chi2_total < best.chi2_total:
            best = r

    # 条件分岐: `best is None` を満たす経路を評価する。

    if best is None:
        raise ValueError("Grid minimization failed: no finite solution.")

    return best


# 関数: `_refine_minimum` の入出力契約と処理意図を定義する。

def _refine_minimum(
    *,
    fn: Callable[[float], FitResult],
    center: float,
    half_width: float,
    n_grid: int,
) -> FitResult:
    """中心値近傍を細かいグリッドで再探索する。"""

    return _grid_minimize(
        fn=fn,
        p_min=float(center) - float(half_width),
        p_max=float(center) + float(half_width),
        n_grid=int(n_grid),
    )


# 関数: `_fit_pmodel_full_minimize` の入出力契約と処理意図を定義する。

def _fit_pmodel_full_minimize(sn: SNData, cc: CCData) -> FitResult:
    """
    完全式係数 (αr, αm, αΛ) を bounded 最適化で推定し、最良 FitResult を返す。

    制約:
      - αr, αm, αΛ は非負（平方係数由来）として探索する。
      - 探索範囲はデータ帯域（z<=約2.5）で E^2>0 を確保できる保守範囲に固定する。
    """

    bounds = [(0.0, 0.12), (0.0, 0.85), (0.0, 1.6)]
    starts = [
        np.asarray([0.0, 0.24, 0.0], dtype=float),
        np.asarray([0.002, 0.30, 0.10], dtype=float),
        np.asarray([0.006, 0.18, 0.30], dtype=float),
        np.asarray([0.012, 0.12, 0.70], dtype=float),
    ]

    # 条件分岐: `len(starts) < 1` を満たす経路を評価する。
    if len(starts) < 1:
        raise ValueError("pmodel full minimization requires at least one initial point.")

    # 条件分岐: `minimize is None` を満たす経路を評価する。

    if minimize is None:
        grid_r = np.linspace(bounds[0][0], bounds[0][1], 7)
        grid_m = np.linspace(bounds[1][0], bounds[1][1], 45)
        grid_l = np.linspace(bounds[2][0], bounds[2][1], 9)
        best_fallback: FitResult | None = None
        for ar in grid_r:
            for am in grid_m:
                for al in grid_l:
                    r = _fit_pmodel_full(float(ar), float(am), float(al), sn, cc)

                    # 条件分岐: `not math.isfinite(r.chi2_total)` を満たす経路を評価する。
                    if not math.isfinite(r.chi2_total):
                        continue

                    # 条件分岐: `best_fallback is None or r.chi2_total < best_fallback.chi2_total` を満たす経路を評価する。

                    if best_fallback is None or r.chi2_total < best_fallback.chi2_total:
                        best_fallback = r

        # 条件分岐: `best_fallback is None` を満たす経路を評価する。

        if best_fallback is None:
            raise ValueError("P-model complete-form fallback grid failed: no finite solution.")

        return best_fallback

    # 関数: `_objective` の入出力契約と処理意図を定義する。

    def _objective(x: np.ndarray) -> float:
        r = _fit_pmodel_full(float(x[0]), float(x[1]), float(x[2]), sn, cc)

        # 条件分岐: `not math.isfinite(r.chi2_total)` を満たす経路を評価する。
        if not math.isfinite(r.chi2_total):
            return 1.0e30

        return float(r.chi2_total)

    best_result: FitResult | None = None
    for x0 in starts:
        opt = minimize(
            _objective,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 400},
        )
        r = _fit_pmodel_full(float(opt.x[0]), float(opt.x[1]), float(opt.x[2]), sn, cc)

        # 条件分岐: `not math.isfinite(r.chi2_total)` を満たす経路を評価する。
        if not math.isfinite(r.chi2_total):
            continue

        # 条件分岐: `best_result is None or r.chi2_total < best_result.chi2_total` を満たす経路を評価する。

        if best_result is None or r.chi2_total < best_result.chi2_total:
            best_result = r

    # 条件分岐: `best_result is None` を満たす経路を評価する。

    if best_result is None:
        raise ValueError("P-model complete-form minimization failed: no finite solution.")

    return best_result


# 関数: `_write_summary_csv` の入出力契約と処理意図を定義する。

def _write_summary_csv(path: Path, rows: list[FitResult], delta_aic: float) -> None:
    """比較結果の主要指標をCSVへ保存する。"""

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "shape_params_json",
                "H0_best_km_s_Mpc",
                "delta_mu_best_mag",
                "chi2_sn",
                "chi2_cc",
                "chi2_total",
                "AIC",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.label,
                    json.dumps(r.shape_params, ensure_ascii=False, sort_keys=True),
                    f"{r.h0_best:.12g}",
                    f"{r.delta_mu_best:.12g}",
                    f"{r.chi2_sn:.12g}",
                    f"{r.chi2_cc:.12g}",
                    f"{r.chi2_total:.12g}",
                    f"{r.aic:.12g}",
                ]
            )

        w.writerow([])
        w.writerow(["delta_aic_baseline_minus_pmodel", f"{delta_aic:.12g}"])


# 関数: `_copy_to_public` の入出力契約と処理意図を定義する。

def _copy_to_public(private_path: Path, public_path: Path) -> None:
    """private 出力を public へ同期する。親ディレクトリは自動作成する。"""

    public_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(private_path, public_path)


# 関数: `_build_figure` の入出力契約と処理意図を定義する。

def _build_figure(
    *,
    sn: SNData,
    cc: CCData,
    fit_lcdm: FitResult,
    fit_pmodel: FitResult,
    out_pdf: Path,
    out_png: Path,
    delta_aic: float,
) -> None:
    """μ(z) と H(z) の直接比較図を生成して PDF/PNG へ保存する。"""

    _set_japanese_font()
    import matplotlib.pyplot as plt

    z_mu = np.linspace(max(1e-3, float(np.min(sn.z))), float(np.max(sn.z)) * 1.03, 600)
    z_h = np.linspace(0.0, max(float(np.max(cc.z)) * 1.08, 2.2), 500)

    om = float(fit_lcdm.shape_params["Omega_m"])
    alpha_r = float(fit_pmodel.shape_params["alpha_r"])
    alpha_m = float(fit_pmodel.shape_params["alpha_m"])
    alpha_lambda = float(fit_pmodel.shape_params["alpha_lambda"])

    mu_lcdm = _mu_from_dl_mpc(_lcdm_dl_mpc(z_mu, omega_m=om, h0=fit_lcdm.h0_best))
    mu_lcdm += fit_lcdm.delta_mu_best
    mu_pmodel = _mu_from_dl_mpc(
        _pmodel_dl_mpc_full(
            z_mu,
            alpha_r=alpha_r,
            alpha_m=alpha_m,
            alpha_lambda=alpha_lambda,
            h0=fit_pmodel.h0_best,
        )
    )
    mu_pmodel += fit_pmodel.delta_mu_best

    hz_lcdm = fit_lcdm.h0_best * _lcdm_ez(z_h, om)
    hz_pmodel = fit_pmodel.h0_best * _pmodel_heff_basis(
        z_h,
        alpha_r=alpha_r,
        alpha_m=alpha_m,
        alpha_lambda=alpha_lambda,
    )

    fig, (ax_mu, ax_h) = plt.subplots(2, 1, figsize=(12.8, 9.8))

    ax_mu.scatter(
        sn.z,
        sn.mu,
        s=9.0,
        color="#4f6d7a",
        alpha=0.26,
        linewidths=0.0,
        label=f"Pantheon+（n={sn.z.size}）",
    )
    ax_mu.plot(
        z_mu,
        mu_lcdm,
        color="#d62828",
        linewidth=2.4,
        label=f"flat ΛCDM best（Ωm={om:.3f}）",
    )
    ax_mu.plot(
        z_mu,
        mu_pmodel,
        color="#1d3557",
        linewidth=2.4,
        linestyle="--",
        label=f"P-model complete best（αm={alpha_m:.3f}, αΛ={alpha_lambda:.3f}）",
    )
    ax_mu.set_xlabel("z", fontsize=15.8)
    ax_mu.set_ylabel("距離モジュラス μ(z) [mag]", fontsize=15.8)
    ax_mu.grid(True, linestyle="--", alpha=0.35)
    ax_mu.legend(fontsize=14.2, loc="lower right")
    ax_mu.tick_params(labelsize=13.8)

    ax_h.errorbar(
        cc.z,
        cc.hz,
        yerr=cc.sigma_hz,
        fmt="o",
        color="#2a9d8f",
        ecolor="#2a9d8f",
        markersize=5.0,
        elinewidth=1.0,
        capsize=2.5,
        alpha=0.9,
        label=f"Cosmic Chronometers（n={cc.z.size}）",
    )
    ax_h.plot(
        z_h,
        hz_lcdm,
        color="#d62828",
        linewidth=2.4,
        label=f"flat ΛCDM best（H0={fit_lcdm.h0_best:.2f}）",
    )
    ax_h.plot(
        z_h,
        hz_pmodel,
        color="#1d3557",
        linewidth=2.4,
        linestyle="--",
        label=f"P-model best（H0={fit_pmodel.h0_best:.2f}）",
    )
    ax_h.set_xlabel("z", fontsize=15.8)
    ax_h.set_ylabel("H(z) [km s$^{-1}$ Mpc$^{-1}$]", fontsize=15.8)
    ax_h.grid(True, linestyle="--", alpha=0.35)
    ax_h.legend(fontsize=14.2, loc="upper left")
    ax_h.tick_params(labelsize=13.8)

    fig.suptitle(
        "Pantheon+ μ(z) と Cosmic Chronometers H(z) の直接比較（同一I/F）",
        fontsize=17.4,
    )
    fig.text(
        0.5,
        0.014,
        (
            f"χ²_total: ΛCDM={fit_lcdm.chi2_total:.2f}, P-model={fit_pmodel.chi2_total:.2f}; "
            f"ΔAIC = AIC_baseline - AIC_P = {delta_aic:+.2f}（正値でP-model優位）"
        ),
        ha="center",
        fontsize=14.0,
    )
    plt.tight_layout(rect=(0.0, 0.085, 1.0, 0.95), h_pad=2.4)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: list[str] | None = None) -> int:
    """CLIエントリ。直接比較を実行し、図と指標を private/public へ保存する。"""

    parser = argparse.ArgumentParser(
        description="Direct fit: Pantheon+ mu(z) and Cosmic Chronometers H(z) for flat LCDM vs P-model.",
    )
    parser.add_argument(
        "--pantheon-file",
        type=Path,
        default=_ROOT / "data" / "cosmology" / "pantheonplus_sh0es.dat",
        help="Path to pantheonplus_sh0es.dat",
    )
    parser.add_argument(
        "--cc-file",
        type=Path,
        default=_ROOT / "data" / "cosmology" / "sources" / "cosmic_chronometers_data_CC.dat",
        help="Path to cosmic chronometer data file (data_CC.dat format).",
    )
    parser.add_argument(
        "--z-min",
        type=float,
        default=0.01,
        help="Minimum z cut for Pantheon+ rows (default: 0.01).",
    )
    args = parser.parse_args(argv)

    sn = _load_pantheon_plus(args.pantheon_file, z_min=float(args.z_min))
    cc = _load_cosmic_chronometers(args.cc_file)

    fit_lcdm_coarse = _grid_minimize(
        fn=lambda om: _fit_lcdm(om, sn, cc),
        p_min=0.05,
        p_max=0.60,
        n_grid=361,
    )
    fit_lcdm = _refine_minimum(
        fn=lambda om: _fit_lcdm(om, sn, cc),
        center=fit_lcdm_coarse.shape_params["Omega_m"],
        half_width=0.03,
        n_grid=241,
    )
    fit_pmodel = _fit_pmodel_full_minimize(sn, cc)

    delta_aic = float(fit_lcdm.aic - fit_pmodel.aic)
    winner = "pmodel" if delta_aic > 0.0 else "baseline_lcdm"

    out_private = _ROOT / "output" / "private" / "cosmology"
    out_public = _ROOT / "output" / "public" / "cosmology"
    out_private.mkdir(parents=True, exist_ok=True)
    out_public.mkdir(parents=True, exist_ok=True)

    base = "cosmology_pantheonplus_cc_direct_fit"
    fig_pdf_private = out_private / f"{base}.pdf"
    fig_png_private = out_private / f"{base}.png"
    metrics_private = out_private / f"{base}_metrics.json"
    summary_private = out_private / f"{base}_summary.csv"

    _build_figure(
        sn=sn,
        cc=cc,
        fit_lcdm=fit_lcdm,
        fit_pmodel=fit_pmodel,
        out_pdf=fig_pdf_private,
        out_png=fig_png_private,
        delta_aic=delta_aic,
    )
    _write_summary_csv(summary_private, [fit_lcdm, fit_pmodel], delta_aic)

    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "pantheon_file": str(args.pantheon_file),
            "cc_file": str(args.cc_file),
            "pantheon_z_min": float(args.z_min),
        },
        "dataset_sizes": {
            "pantheon_points_used": int(sn.z.size),
            "cc_points_used": int(cc.z.size),
        },
        "models": {
            "baseline": {
                "name": fit_lcdm.label,
                "shape_params": fit_lcdm.shape_params,
                "h0_best_km_s_Mpc": fit_lcdm.h0_best,
                "delta_mu_best_mag": fit_lcdm.delta_mu_best,
                "chi2_sn": fit_lcdm.chi2_sn,
                "chi2_cc": fit_lcdm.chi2_cc,
                "chi2_total": fit_lcdm.chi2_total,
                "AIC": fit_lcdm.aic,
            },
            "pmodel": {
                "name": fit_pmodel.label,
                "shape_params": fit_pmodel.shape_params,
                "h0_best_km_s_Mpc": fit_pmodel.h0_best,
                "delta_mu_best_mag": fit_pmodel.delta_mu_best,
                "chi2_sn": fit_pmodel.chi2_sn,
                "chi2_cc": fit_pmodel.chi2_cc,
                "chi2_total": fit_pmodel.chi2_total,
                "AIC": fit_pmodel.aic,
            },
        },
        "comparison": {
            "delta_aic_baseline_minus_pmodel": delta_aic,
            "delta_aic_sign_rule": "DeltaAIC = AIC_baseline - AIC_P_model (positive favors P-model)",
            "winner": winner,
        },
        "outputs_private": {
            "figure_pdf": str(fig_pdf_private),
            "figure_png": str(fig_png_private),
            "metrics_json": str(metrics_private),
            "summary_csv": str(summary_private),
        },
        "notes": [
            "Pantheon+ は diag誤差（MU_SH0ES_ERR_DIAG）を使用した直接比較（共分散完全版ではない）。",
            "P-model は H_eff^2(z)=H0^2+αr[(1+z)^4-1]+αm[(1+z)^3-1]-αΛ ln(1+z) の完全式を採用。",
        ],
    }
    metrics_private.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    fig_pdf_public = out_public / f"{base}.pdf"
    fig_png_public = out_public / f"{base}.png"
    metrics_public = out_public / f"{base}_metrics.json"
    summary_public = out_public / f"{base}_summary.csv"
    _copy_to_public(fig_pdf_private, fig_pdf_public)
    _copy_to_public(fig_png_private, fig_png_public)
    _copy_to_public(metrics_private, metrics_public)
    _copy_to_public(summary_private, summary_public)

    print(f"[ok] figure pdf (private): {fig_pdf_private}")
    print(f"[ok] metrics json (private): {metrics_private}")
    print(f"[ok] summary csv (private): {summary_private}")
    print(f"[ok] figure pdf (public) : {fig_pdf_public}")

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_pantheonplus_cc_direct_fit",
                "argv": sys.argv,
                "inputs": {
                    "pantheon_file": args.pantheon_file,
                    "cc_file": args.cc_file,
                    "z_min": float(args.z_min),
                },
                "metrics": {
                    "chi2_baseline": fit_lcdm.chi2_total,
                    "chi2_pmodel": fit_pmodel.chi2_total,
                    "delta_aic_baseline_minus_pmodel": delta_aic,
                    "winner": winner,
                },
                "outputs": {
                    "private_pdf": fig_pdf_private,
                    "private_json": metrics_private,
                    "private_csv": summary_private,
                    "public_pdf": fig_pdf_public,
                    "public_json": metrics_public,
                    "public_csv": summary_public,
                },
            }
        )
    except Exception:
        pass

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
