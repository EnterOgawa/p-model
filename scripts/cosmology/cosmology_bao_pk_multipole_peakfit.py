#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_bao_pk_multipole_peakfit.py

Phase 16（宇宙論）/ Step 16.4（BAO一次統計の再導出）:
BAO の圧縮出力（D_M/r_d, D_H/r_d）の前に、BOSS DR12 の power spectrum multipoles Pℓ(k)（ℓ=0,2）
という“より一次に近い観測統計”から、P-model 側の距離変換（AP/異方歪み）を入れ替えて比較するクロスチェックを行う。

方針（合意）:
1) BOSS DR12 再構成後 ξℓ（ℓ=0,2）から開始（別スクリプト）
2) 距離変換（θ,z→(s,μ)/(k,μ)）を P-model 側で定義し、AP/warping を入れ替える
3) フィットは「なめらか成分＋BAOピーク」だけ（最小モデル）
4) r_d はフリー（全体スケール α に吸収）。異方（ℓ=2）を重視して幾何整合（ε）を評価

入力（固定・キャッシュ）:
  - data/cosmology/beutler_2016_combineddr12_bao_powspec/
    - Beutleretal_pk_{monopole,quadrupole}_DR12_{NGC,SGC}_z{1..3}_{postrecon,prerecon}_120.dat
    - Beutleretal_cov_patchy_z{1..3}_{NGC,SGC}_{postrecon,prerecon}_*.dat
    - Beutleretal_window_z{1..3}_{NGC,SGC}.dat（窓関数: RR multipoles; --window 時に使用）
  ※ 未取得の場合は、先に scripts/cosmology/fetch_boss_dr12_beutler2016_bao_powspec.py を実行する。

出力（固定名）:
  - output/cosmology/cosmology_bao_pk_multipole_peakfit.png
  - output/cosmology/cosmology_bao_pk_multipole_peakfit_metrics.json
  - （--window の場合は末尾に _window を付与）
  - （pre-recon の場合は末尾に _pre_recon）
"""

from __future__ import annotations

import argparse
import json
import math
import re
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


@dataclass(frozen=True)
class ZBin:
    zbin: int
    z_eff: float
    label: str


@dataclass(frozen=True)
class PKDataset:
    region: str
    zbin: int
    z_eff: float
    label: str
    k: np.ndarray  # fiducial k (as tabulated)
    y: np.ndarray  # stacked [P0(k), P2(k)]
    cov: np.ndarray
    cov_inv: np.ndarray
    sig0: np.ndarray
    sig2: np.ndarray


@dataclass(frozen=True)
class WindowKernel:
    """
    Window convolution helper (RR multipoles) in configuration space.

    Pipeline (linear operator):
      Pℓ(k_in) -> ξℓ(s) (Hankel) -> window mixing in ξ -> Pℓ(k_out) (Hankel)

    Notes:
      - Implemented for ℓ=0,2 only.
      - Mixing uses window multipoles up to L=8 (as provided in Beutler window files).
    """

    k_in: np.ndarray  # (n_in,)
    s: np.ndarray  # (n_s,)
    w_s: np.ndarray  # (n_s,) trapezoid weights for s integral

    # Hankel matrices: ξℓ(s) = Aℓ @ Pℓ(k_in)
    a0: np.ndarray  # (n_s, n_in)
    a2: np.ndarray  # (n_s, n_in)

    # Discretization calibration matrices (right-multiply in k-space) so that
    # the identity-window operator is close to identity on the chosen (k_in, s) grids.
    c0: np.ndarray  # (n_in, n_in)
    c2: np.ndarray  # (n_in, n_in)

    # Window mixing (at each s): [ξ0c, ξ2c]^T = [[m00,m02],[m20,m22]] [ξ0,ξ2]^T
    m00: np.ndarray  # (n_s,)
    m02: np.ndarray  # (n_s,)
    m20: np.ndarray  # (n_s,)
    m22: np.ndarray  # (n_s,)

    def matrix_for_k_out(self, k_out: np.ndarray) -> np.ndarray:
        k_out = np.asarray(k_out, dtype=float)
        n_out = int(k_out.size)
        n_in = int(self.k_in.size)
        n_s = int(self.s.size)
        if n_out == 0 or n_in == 0 or n_s == 0:
            return np.zeros((0, 0), dtype=float)

        # Pℓ(k_out) = 4π (-i)^ℓ ∫ s^2 ξℓ(s) jℓ(k s) ds.
        x = k_out[:, None] * self.s[None, :]
        j0 = _sph_j0(x)
        j2 = _sph_j2(x)
        fac = self.w_s * (self.s * self.s)  # (n_s,)
        b0 = (4.0 * math.pi) * j0 * fac[None, :]  # (n_out, n_s)
        b2 = (-4.0 * math.pi) * j2 * fac[None, :]  # (n_out, n_s)

        k00 = (b0 @ (self.a0 * self.m00[:, None])) @ self.c0
        k02 = (b0 @ (self.a2 * self.m02[:, None])) @ self.c2
        k20 = (b2 @ (self.a0 * self.m20[:, None])) @ self.c0
        k22 = (b2 @ (self.a2 * self.m22[:, None])) @ self.c2

        k_mat = np.zeros((2 * n_out, 2 * n_in), dtype=float)
        k_mat[:n_out, :n_in] = k00
        k_mat[:n_out, n_in:] = k02
        k_mat[n_out:, :n_in] = k20
        k_mat[n_out:, n_in:] = k22
        return k_mat


@dataclass(frozen=True)
class WindowOp:
    kernel: WindowKernel
    k_data: np.ndarray  # (n_out,)
    k_mat: np.ndarray  # (2*n_out, 2*n_in)

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
        if not chosen:
            return

        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


_NUM_LINE = re.compile(r"^[-+]?\d+\.\d+")


def _trap_weights(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = int(x.size)
    if n == 0:
        return np.zeros((0,), dtype=float)
    if n == 1:
        return np.zeros((1,), dtype=float)
    w = np.zeros_like(x, dtype=float)
    w[1:-1] = 0.5 * (x[2:] - x[:-2])
    w[0] = 0.5 * (x[1] - x[0])
    w[-1] = 0.5 * (x[-1] - x[-2])
    return w


def _sph_j0(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    small = np.abs(x) < 1e-5
    if np.any(small):
        x2 = x[small] * x[small]
        out[small] = 1.0 - x2 / 6.0 + (x2 * x2) / 120.0
    if np.any(~small):
        xs = x[~small]
        out[~small] = np.sin(xs) / xs
    return out


def _sph_j2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    small = np.abs(x) < 1e-4
    if np.any(small):
        x2 = x[small] * x[small]
        out[small] = x2 / 15.0 - (x2 * x2) / 210.0 + (x2 * x2 * x2) / 7560.0
    if np.any(~small):
        xs = x[~small]
        xs2 = xs * xs
        sinx = np.sin(xs)
        cosx = np.cos(xs)
        out[~small] = (sinx * (3.0 - xs2) - 3.0 * xs * cosx) / (xs2 * xs)
    return out


def _legendre_p(l: int, x: np.ndarray) -> np.ndarray:
    l = int(l)
    if l < 0:
        raise ValueError("l must be >=0")
    coeff = np.zeros((l + 1,), dtype=float)
    coeff[l] = 1.0
    return np.polynomial.legendre.legval(np.asarray(x, dtype=float), coeff)


def _window_coupling_coeffs() -> Dict[Tuple[int, int, int], float]:
    """
    Return coupling coefficients for Legendre product expansion:
      P_ell * P_L = Σ_n c_{n,ell,L} P_n
    where c_{n,ell,L} = (2n+1)/2 ∫ P_n P_ell P_L dμ.

    We compute only what we need: n in {0,2}, ell in {0,2}, L in {0,2,4,6,8}.
    """
    mu_q, w_q = np.polynomial.legendre.leggauss(256)
    mu_q = np.asarray(mu_q, dtype=float)
    w_q = np.asarray(w_q, dtype=float)

    ls = [0, 2, 4, 6, 8]
    p = {l: _legendre_p(l, mu_q) for l in ls}

    out: Dict[Tuple[int, int, int], float] = {}
    for n in (0, 2):
        pn = p[n]
        for ell in (0, 2):
            pe = p[ell]
            for L in ls:
                pL = p[L]
                integral = float(np.sum(w_q * pn * pe * pL))
                out[(n, ell, L)] = 0.5 * float(2 * n + 1) * integral
    return out


_WINDOW_COEFFS = _window_coupling_coeffs()

def _read_pk_table(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    k: List[float] = []
    pk: List[float] = []
    sig: List[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        t = line.strip()
        if not t or t.startswith("#") or t.startswith("###"):
            continue
        if not _NUM_LINE.match(t):
            continue
        parts = t.split()
        if len(parts) < 4:
            continue
        k.append(float(parts[0]))
        pk.append(float(parts[2]))
        sig.append(float(parts[3]))
    return np.asarray(k, dtype=float), np.asarray(pk, dtype=float), np.asarray(sig, dtype=float)


def _read_window_rr(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    s: List[float] = []
    rr0: List[float] = []
    rr2: List[float] = []
    rr4: List[float] = []
    rr6: List[float] = []
    rr8: List[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        t = line.strip()
        if not t or t.startswith("#") or t.startswith("###"):
            continue
        parts = t.split()
        if not parts:
            continue
        if parts[0] == "s":
            continue
        if len(parts) < 7:
            continue
        try:
            ss = float(parts[0])
            r0 = float(parts[2])
            r2 = float(parts[3])
            r4 = float(parts[4])
            r6 = float(parts[5])
            r8 = float(parts[6])
        except Exception:
            continue
        s.append(ss)
        rr0.append(r0)
        rr2.append(r2)
        rr4.append(r4)
        rr6.append(r6)
        rr8.append(r8)
    return (
        np.asarray(s, dtype=float),
        np.asarray(rr0, dtype=float),
        np.asarray(rr2, dtype=float),
        np.asarray(rr4, dtype=float),
        np.asarray(rr6, dtype=float),
        np.asarray(rr8, dtype=float),
    )


def _build_window_kernel(
    *,
    data_dir: Path,
    zbin: int,
    region: str,
    k_in: np.ndarray,
    s_max: float,
) -> WindowKernel:
    path = data_dir / f"Beutleretal_window_z{int(zbin)}_{str(region).upper()}.dat"
    if not path.exists():
        raise FileNotFoundError(f"window file not found: {path}")

    s, rr0, rr2, rr4, rr6, rr8 = _read_window_rr(path)
    if s.size == 0:
        raise ValueError(f"empty window file: {path}")

    mask = (rr0 > 0.0) & np.isfinite(rr0) & np.isfinite(s) & (s <= float(s_max))
    if not np.any(mask):
        raise ValueError(f"no usable window rows after filtering (s_max={s_max}): {path}")
    s = np.asarray(s[mask], dtype=float)
    rr0 = np.asarray(rr0[mask], dtype=float)
    rr2 = np.asarray(rr2[mask], dtype=float)
    rr4 = np.asarray(rr4[mask], dtype=float)
    rr6 = np.asarray(rr6[mask], dtype=float)
    rr8 = np.asarray(rr8[mask], dtype=float)

    w2 = rr2 / rr0
    w4 = rr4 / rr0
    w6 = rr6 / rr0
    w8 = rr8 / rr0
    w0 = np.ones_like(w2, dtype=float)

    # Hankel: ξℓ(s) = i^ℓ/(2π^2) ∫ k^2 Pℓ(k) jℓ(k s) dk
    k_in = np.asarray(k_in, dtype=float)
    if k_in.ndim != 1 or k_in.size < 8:
        raise ValueError("k_in must be 1D with enough points")

    w_k = _trap_weights(k_in)
    fac_k = w_k * (k_in * k_in) / (2.0 * math.pi * math.pi)  # (n_in,)
    x = s[:, None] * k_in[None, :]
    j0 = _sph_j0(x)
    j2 = _sph_j2(x)
    a0 = j0 * fac_k[None, :]
    a2 = (-j2) * fac_k[None, :]  # i^2 = -1

    # Window mixing in ξ multipoles (per s).
    wL = {0: w0, 2: w2, 4: w4, 6: w6, 8: w8}

    def mix(n: int, ell: int) -> np.ndarray:
        out = np.zeros_like(s, dtype=float)
        for L in (0, 2, 4, 6, 8):
            c = float(_WINDOW_COEFFS[(int(n), int(ell), int(L))])
            if c == 0.0:
                continue
            out += c * wL[L]
        return out

    m00 = mix(0, 0)
    m02 = mix(0, 2)
    m20 = mix(2, 0)
    m22 = mix(2, 2)

    w_s = _trap_weights(s)

    # Discretization calibration: enforce that (identity window) maps k_in -> k_in approximately as I.
    # This greatly reduces sensitivity to finite (k_in, s) ranges in the numerical Hankel pair.
    fac_s = w_s * (s * s)  # (n_s,)
    x_in = k_in[:, None] * s[None, :]
    j0_in = _sph_j0(x_in)
    j2_in = _sph_j2(x_in)
    b0_in = (4.0 * math.pi) * j0_in * fac_s[None, :]
    b2_in = (-4.0 * math.pi) * j2_in * fac_s[None, :]
    d0 = b0_in @ a0
    d2 = b2_in @ a2
    try:
        c0 = np.linalg.pinv(d0, rcond=1e-10)
        c2 = np.linalg.pinv(d2, rcond=1e-10)
    except Exception:
        c0 = np.linalg.pinv(d0)
        c2 = np.linalg.pinv(d2)
    return WindowKernel(
        k_in=k_in,
        s=s,
        w_s=w_s,
        a0=a0,
        a2=a2,
        c0=c0,
        c2=c2,
        m00=m00,
        m02=m02,
        m20=m20,
        m22=m22,
    )

def _read_cov_index_to_k(path: Path) -> Dict[int, float]:
    idx_to_k: Dict[int, float] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        t = line.strip()
        if not t or t.startswith("#") or t.startswith("###"):
            continue
        parts = t.split()
        if len(parts) < 5:
            continue
        try:
            i = int(parts[0])
            j = int(parts[1])
            ki = float(parts[2])
            kj = float(parts[3])
            float(parts[4])
        except Exception:
            continue
        if i not in idx_to_k:
            idx_to_k[i] = ki
        if j not in idx_to_k:
            idx_to_k[j] = kj
    return idx_to_k


def _match_indices_for_k(*, k_data: np.ndarray, idx_to_k: Dict[int, float], tol: float = 5e-4) -> Tuple[List[int], List[int]]:
    # For each k, find the indices that correspond to that k (typically monopole and quadrupole blocks).
    k_data = np.asarray(k_data, dtype=float)
    idx0: List[int] = []
    idx2: List[int] = []
    for k in k_data:
        hits = [idx for idx, kk in idx_to_k.items() if abs(float(kk) - float(k)) <= float(tol)]
        hits = sorted(set(hits))
        if len(hits) < 2:
            raise ValueError(f"covariance does not contain two indices for k={k:.6g} (hits={hits})")
        # Heuristic: smaller index -> monopole, larger -> quadrupole.
        idx0.append(int(hits[0]))
        idx2.append(int(hits[-1]))
    return idx0, idx2


def _read_cov_submatrix(
    *, cov_path: Path, idx0: List[int], idx2: List[int], n: int
) -> np.ndarray:
    if len(idx0) != n or len(idx2) != n:
        raise ValueError("idx size mismatch")
    full_idx = [*idx0, *idx2]
    pos = {int(ix): i for i, ix in enumerate(full_idx)}
    cov = np.zeros((2 * n, 2 * n), dtype=float)
    for line in cov_path.read_text(encoding="utf-8", errors="replace").splitlines():
        t = line.strip()
        if not t or t.startswith("#") or t.startswith("###"):
            continue
        parts = t.split()
        if len(parts) < 5:
            continue
        try:
            i = int(parts[0])
            j = int(parts[1])
            cij = float(parts[4])
        except Exception:
            continue
        if i not in pos or j not in pos:
            continue
        ii = int(pos[i])
        jj = int(pos[j])
        cov[ii, jj] = cij
        cov[jj, ii] = cij
    cov = 0.5 * (cov + cov.T)
    return cov


def _p2(mu: np.ndarray) -> np.ndarray:
    mu = np.asarray(mu, dtype=float)
    return 0.5 * (3.0 * mu * mu - 1.0)


def _f_ap_pbg_exponential(z: float) -> float:
    op = 1.0 + float(z)
    if not (op > 0.0):
        return float("nan")
    return float(op * math.log(op))


def _f_ap_lcdm_flat(z: float, *, omega_m: float, n_grid: int = 4000) -> float:
    z = float(z)
    if z < 0:
        return float("nan")
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
    if not (math.isfinite(f_ap_model) and math.isfinite(f_ap_fid) and f_ap_model > 0.0 and f_ap_fid > 0.0):
        return float("nan")
    ratio = float(f_ap_fid / f_ap_model)  # alpha_parallel/alpha_perp
    return float(ratio ** (1.0 / 3.0) - 1.0)


def _design_matrix(
    k_fid: np.ndarray,
    *,
    alpha: float,
    eps: float,
    k0: float,
    sigma_k: float,
    mu: np.ndarray,
    w: np.ndarray,
    p2_fid: np.ndarray,
    sqrt1mu2: np.ndarray,
) -> np.ndarray:
    """
    Build design matrix for y=[P0(k), P2(k)] (fid coordinate) given:
      - true multipoles P0_true(k), P2_true(k) parameterized as (smooth + peak)
      - AP warp (alpha, eps) mapping fid -> true in k-space
      - only ℓ=0,2 kept in the true P(k,μ) expansion

    Linear parameters (8):
      - P0_true smooth: [c0, c1*k, c2*k^2]
      - P0_true peak amplitude: A0
      - P2_true smooth: [d0, d1*k, d2*k^2]
      - P2_true peak amplitude: A2

    Nonlinear parameters:
      - alpha, eps (AP warp)
      - k0, sigma_k (peak center/width in template coordinates; r_d free is absorbed in alpha)
    """
    k_fid = np.asarray(k_fid, dtype=float)
    n = int(k_fid.size)
    if n == 0:
        return np.zeros((0, 0), dtype=float)

    one_p_eps = 1.0 + float(eps)
    if not (one_p_eps > 0.0 and math.isfinite(one_p_eps)):
        raise ValueError("invalid eps (requires 1+eps>0)")

    alpha = float(alpha)
    if not (alpha > 0.0 and math.isfinite(alpha)):
        raise ValueError("invalid alpha")

    # Parameterization: alpha = (α⊥^2 α∥)^(1/3), 1+eps = (α∥/α⊥)^(1/3)
    alpha_perp = alpha / one_p_eps
    alpha_par = alpha * (one_p_eps**2)

    # k_true = k_fid * sqrt((mu/α∥)^2 + ((1-mu^2)^(1/2)/α⊥)^2)
    t = np.sqrt((mu / alpha_par) ** 2 + (sqrt1mu2 / alpha_perp) ** 2)
    if np.any(t <= 0.0):
        raise ValueError("invalid AP warp (t<=0)")
    mu_true = (mu / alpha_par) / t
    p2_true = _p2(mu_true)

    # Jacobian factor (can be absorbed in amplitudes but kept for consistency)
    jfac = 1.0 / (alpha_perp * alpha_perp * alpha_par)

    w0 = w
    w_p2 = w * p2_fid
    w_p2_true = w * p2_true
    w_p2_true_p2 = w * p2_true * p2_fid

    k_true = k_fid[:, None] * t[None, :]
    k1 = k_true
    k2 = k_true * k_true
    peak = np.exp(-0.5 * ((k_true - float(k0)) / float(sigma_k)) ** 2)

    sum_w0 = float(np.sum(w0))
    sum_wp2 = float(np.sum(w_p2))
    sum_wp2_true = float(np.sum(w_p2_true))
    sum_wp2_true_p2 = float(np.sum(w_p2_true_p2))

    i00_b0 = np.full(n, 0.5 * sum_w0 * jfac, dtype=float)
    i20_b0 = np.full(n, 2.5 * sum_wp2 * jfac, dtype=float)
    i02_b0 = np.full(n, 0.5 * sum_wp2_true * jfac, dtype=float)
    i22_b0 = np.full(n, 2.5 * sum_wp2_true_p2 * jfac, dtype=float)

    def integrate(f: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        i00 = 0.5 * np.sum(f * w0[None, :], axis=1) * jfac
        i20 = 2.5 * np.sum(f * w_p2[None, :], axis=1) * jfac
        i02 = 0.5 * np.sum(f * w_p2_true[None, :], axis=1) * jfac
        i22 = 2.5 * np.sum(f * w_p2_true_p2[None, :], axis=1) * jfac
        return i00, i20, i02, i22

    i00_b1, i20_b1, i02_b1, i22_b1 = integrate(k1)
    i00_b2, i20_b2, i02_b2, i22_b2 = integrate(k2)
    i00_pk, i20_pk, i02_pk, i22_pk = integrate(peak)

    mtx = np.zeros((2 * n, 8), dtype=float)
    # Column layout: [P0:b0,b1,b2,pk, P2:b0,b1,b2,pk]
    cols = [
        (i00_b0, i20_b0, i02_b0, i22_b0),
        (i00_b1, i20_b1, i02_b1, i22_b1),
        (i00_b2, i20_b2, i02_b2, i22_b2),
        (i00_pk, i20_pk, i02_pk, i22_pk),
    ]
    for kk, (i00, i20, i02, i22) in enumerate(cols):
        mtx[:n, kk] = i00
        mtx[n:, kk] = i20
        mtx[:n, kk + 4] = i02
        mtx[n:, kk + 4] = i22
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


def _scan_grid_joint(
    *,
    datasets: Sequence[PKDataset],
    window_ops: Optional[Dict[str, WindowOp]] = None,
    alpha_grid: np.ndarray,
    eps_grid: np.ndarray,
    k0: float,
    sigma_k: float,
    mu: np.ndarray,
    w: np.ndarray,
    p2_fid: np.ndarray,
    sqrt1mu2: np.ndarray,
    return_eps_profile: bool = False,
) -> Dict[str, Any]:
    best: Dict[str, Any] = {"chi2": float("inf")}
    profile_eps: Dict[float, Dict[str, float]] = {float(eps): {"chi2": float("inf"), "alpha": float("nan")} for eps in eps_grid}
    for alpha in alpha_grid:
        for eps in eps_grid:
            chi2_total = 0.0
            per_region: Dict[str, Any] = {}
            ok = True
            for ds in datasets:
                try:
                    wop = window_ops.get(str(ds.region)) if window_ops else None
                    if wop is None:
                        mtx = _design_matrix(
                            ds.k,
                            alpha=float(alpha),
                            eps=float(eps),
                            k0=k0,
                            sigma_k=sigma_k,
                            mu=mu,
                            w=w,
                            p2_fid=p2_fid,
                            sqrt1mu2=sqrt1mu2,
                        )
                    else:
                        mtx_u = _design_matrix(
                            wop.kernel.k_in,
                            alpha=float(alpha),
                            eps=float(eps),
                            k0=k0,
                            sigma_k=sigma_k,
                            mu=mu,
                            w=w,
                            p2_fid=p2_fid,
                            sqrt1mu2=sqrt1mu2,
                        )
                        mtx = wop.k_mat @ mtx_u
                except Exception:
                    ok = False
                    break
                fit = _gls_fit(y=ds.y, mtx=mtx, cov_inv=ds.cov_inv)
                chi2_total += float(fit["chi2"])
                per_region[ds.region] = {"x": fit["x"], "chi2": float(fit["chi2"])}
            if not ok:
                continue
            if chi2_total < float(best["chi2"]):
                best = {"alpha": float(alpha), "eps": float(eps), "chi2": float(chi2_total), "per_region": per_region}
            if return_eps_profile:
                eps_key = float(eps)
                if float(chi2_total) < float(profile_eps[eps_key]["chi2"]):
                    profile_eps[eps_key] = {"chi2": float(chi2_total), "alpha": float(alpha)}
    if not math.isfinite(float(best.get("chi2", float("inf")))):
        raise RuntimeError("grid search failed (no finite chi2)")
    if return_eps_profile:
        eps_sorted = sorted(profile_eps.keys())
        best["eps_profile"] = [
            {"eps": float(eps), "chi2": float(profile_eps[eps]["chi2"]), "alpha": float(profile_eps[eps]["alpha"])}
            for eps in eps_sorted
        ]
    return best


def _profile_ci(
    *,
    x: np.ndarray,
    chi2: np.ndarray,
    delta: float,
) -> Tuple[Optional[float], Optional[float]]:
    x = np.asarray(x, dtype=float)
    chi2 = np.asarray(chi2, dtype=float)
    if x.size == 0 or chi2.size != x.size:
        return None, None
    if not np.all(np.isfinite(chi2)):
        return None, None
    i0 = int(np.nanargmin(chi2))
    chi2_min = float(chi2[i0])
    target = chi2_min + float(delta)
    d = chi2 - target
    inside = d <= 0.0
    if not bool(np.any(inside)):
        return None, None

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
        if not (math.isfinite(d0) and math.isfinite(d1)):
            return None
        if d0 == d1:
            return None
        t = (0.0 - d0) / (d1 - d0)
        return x0 + t * (x1 - x0)

    lo: Optional[float]
    hi: Optional[float]

    if left == 0:
        lo = float(x[0])
    else:
        lo = interp_cross(left - 1, left)
        if lo is None:
            lo = float(x[left])

    if right == int(x.size) - 1:
        hi = float(x[-1])
    else:
        hi = interp_cross(right + 1, right)
        if hi is None:
            hi = float(x[right])

    return lo, hi


def _predict_curve(
    *,
    k_grid: np.ndarray,
    x: np.ndarray,
    alpha: float,
    eps: float,
    k0: float,
    sigma_k: float,
    mu: np.ndarray,
    w: np.ndarray,
    p2_fid: np.ndarray,
    sqrt1mu2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    mtx = _design_matrix(
        k_grid,
        alpha=alpha,
        eps=eps,
        k0=k0,
        sigma_k=sigma_k,
        mu=mu,
        w=w,
        p2_fid=p2_fid,
        sqrt1mu2=sqrt1mu2,
    )
    y = mtx @ x
    n = int(k_grid.size)
    return y[:n], y[n:]


def _load_dataset(
    *,
    data_dir: Path,
    recon: str,
    zbin: ZBin,
    region: str,
    k_min: float,
    k_max: float,
) -> PKDataset:
    base = f"Beutleretal_pk_{{ell}}_DR12_{region}_z{zbin.zbin}_{recon}_120.dat"
    p0_path = data_dir / base.format(ell="monopole")
    p2_path = data_dir / base.format(ell="quadrupole")
    cov_path = next(
        iter(
            sorted(
                data_dir.glob(
                    f"Beutleretal_cov_patchy_z{zbin.zbin}_{region}_{recon}_*.dat",
                )
            )
        ),
        None,
    )
    if cov_path is None:
        raise FileNotFoundError(f"cov file not found for z{zbin.zbin} {region} {recon} in {data_dir}")

    k0, p0, _sig0_diag = _read_pk_table(p0_path)
    k2, p2, _sig2_diag = _read_pk_table(p2_path)
    if k0.size != k2.size:
        raise ValueError(f"k bins mismatch between monopole/quadrupole: {p0_path} vs {p2_path}")
    if np.max(np.abs(k0 - k2)) > 5e-4:
        raise ValueError(f"k bins differ too much between monopole/quadrupole: {p0_path} vs {p2_path}")

    mask = (k0 >= float(k_min)) & (k0 <= float(k_max))
    if not np.any(mask):
        raise ValueError(f"no k bins in range [{k_min},{k_max}] for {p0_path}")
    k = np.asarray(k0[mask], dtype=float)
    p0 = np.asarray(p0[mask], dtype=float)
    p2 = np.asarray(p2[mask], dtype=float)

    idx_to_k = _read_cov_index_to_k(cov_path)
    idx0, idx2 = _match_indices_for_k(k_data=k, idx_to_k=idx_to_k)
    cov = _read_cov_submatrix(cov_path=cov_path, idx0=idx0, idx2=idx2, n=int(k.size))

    # Invert covariance.
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        cov_inv = np.linalg.pinv(cov, rcond=1e-12)
    cov_inv = 0.5 * (cov_inv + cov_inv.T)

    n = int(k.size)
    cov00 = cov[:n, :n]
    cov22 = cov[n:, n:]
    sig0 = np.sqrt(np.maximum(0.0, np.diag(cov00)))
    sig2 = np.sqrt(np.maximum(0.0, np.diag(cov22)))

    y = np.concatenate([p0, p2], axis=0)
    return PKDataset(
        region=region,
        zbin=zbin.zbin,
        z_eff=zbin.z_eff,
        label=zbin.label,
        k=k,
        y=y,
        cov=cov,
        cov_inv=cov_inv,
        sig0=sig0,
        sig2=sig2,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Cosmology: BAO peak fit from BOSS DR12 P(k) multipoles (Beutler data).")
    ap.add_argument("--recon", choices=["postrecon", "prerecon"], default="postrecon", help="reconstruction state")
    ap.add_argument("--data-dir", type=str, default="", help="Dataset directory override")
    ap.add_argument("--z-bins", type=str, default="1,2,3", help="z bins to use (default: 1,2,3)")
    ap.add_argument("--regions", type=str, default="NGC,SGC", help="regions to use (default: NGC,SGC)")
    ap.add_argument("--k-min", type=float, default=0.015, help="min k for fit [h/Mpc] (default: 0.015)")
    ap.add_argument("--k-max", type=float, default=0.15, help="max k for fit [h/Mpc] (default: 0.15)")
    ap.add_argument("--mu-n", type=int, default=80, help="Gauss-Legendre points for μ integral (default: 80)")
    ap.add_argument("--k0", type=float, default=0.08, help="BAO peak center in template coordinates [h/Mpc] (default: 0.08)")
    ap.add_argument("--sigma-k", type=float, default=0.015, help="BAO peak width [h/Mpc] (default: 0.015)")
    ap.add_argument("--alpha-min", type=float, default=0.8, help="alpha grid min (default: 0.8)")
    ap.add_argument("--alpha-max", type=float, default=1.2, help="alpha grid max (default: 1.2)")
    ap.add_argument("--alpha-step", type=float, default=0.004, help="alpha grid step (default: 0.004)")
    ap.add_argument("--eps-min", type=float, default=-0.2, help="eps grid min (default: -0.2)")
    ap.add_argument("--eps-max", type=float, default=0.2, help="eps grid max (default: 0.2)")
    ap.add_argument("--eps-step", type=float, default=0.004, help="eps grid step (default: 0.004)")
    ap.add_argument("--window", action="store_true", help="Apply survey window convolution using Beutler RR multipoles.")
    ap.add_argument("--window-s-max", type=float, default=2500.0, help="Max s for window Hankel integrals [h^-1 Mpc].")
    ap.add_argument("--window-k-in-max", type=float, default=0.4, help="Max k for window Hankel integrals [h/Mpc].")
    ap.add_argument("--window-k-in-step", type=float, default=0.002, help="Step for k_in grid in window Hankel integrals [h/Mpc].")
    ap.add_argument("--lcdm-omega-m", type=float, default=0.31, help="Reference flat LCDM Ωm for eps_pred (default: 0.31)")
    ap.add_argument("--lcdm-n-grid", type=int, default=4000, help="z-integral grid points for LCDM F_AP (default: 4000)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    recon = str(args.recon)
    if str(args.data_dir).strip():
        data_dir = Path(str(args.data_dir))
    else:
        data_dir = _ROOT / "data" / "cosmology" / "beutler_2016_combineddr12_bao_powspec"
    if not data_dir.exists():
        raise SystemExit(f"data dir not found: {data_dir}")

    z_bins = [int(x.strip()) for x in str(args.z_bins).split(",") if x.strip()]
    z_meta = [z for z in _ZBINS if z.zbin in z_bins]
    if not z_meta:
        raise SystemExit("no valid z bins selected")

    regions = [x.strip().upper() for x in str(args.regions).split(",") if x.strip()]
    if not regions:
        raise SystemExit("no regions selected")

    k_min = float(args.k_min)
    k_max = float(args.k_max)
    if not (k_max > k_min):
        raise SystemExit("--k-max must be > --k-min")

    mu_n = int(args.mu_n)
    if mu_n < 20:
        raise SystemExit("--mu-n must be >= 20")
    mu, w = np.polynomial.legendre.leggauss(mu_n)
    mu = np.asarray(mu, dtype=float)
    w = np.asarray(w, dtype=float)
    p2_fid = _p2(mu)
    sqrt1mu2 = np.sqrt(np.maximum(0.0, 1.0 - mu * mu))

    alpha_grid = np.arange(float(args.alpha_min), float(args.alpha_max) + 0.5 * float(args.alpha_step), float(args.alpha_step), dtype=float)
    eps_grid = np.arange(float(args.eps_min), float(args.eps_max) + 0.5 * float(args.eps_step), float(args.eps_step), dtype=float)
    if alpha_grid.size < 3 or eps_grid.size < 3:
        raise SystemExit("grid too small; adjust alpha/eps ranges or steps")

    k0_peak = float(args.k0)
    sigma_k = float(args.sigma_k)
    if not (sigma_k > 0.0):
        raise SystemExit("--sigma-k must be > 0")

    omega_m = float(args.lcdm_omega_m)
    lcdm_n_grid = int(args.lcdm_n_grid)

    use_window = bool(args.window)
    window_s_max = float(args.window_s_max)
    window_k_in_max = float(args.window_k_in_max)
    window_k_in_step = float(args.window_k_in_step)
    if use_window:
        if not (window_s_max > 0.0 and math.isfinite(window_s_max)):
            raise SystemExit("--window-s-max must be > 0")
        if not (window_k_in_max > 0.0 and math.isfinite(window_k_in_max)):
            raise SystemExit("--window-k-in-max must be > 0")
        if not (window_k_in_step > 0.0 and math.isfinite(window_k_in_step)):
            raise SystemExit("--window-k-in-step must be > 0")
        k_in = np.arange(
            0.0,
            float(window_k_in_max) + 0.5 * float(window_k_in_step),
            float(window_k_in_step),
            dtype=float,
        )
        if k_in.size < 32:
            raise SystemExit("window k_in grid too small; increase --window-k-in-max or decrease --window-k-in-step")
    else:
        k_in = np.array([], dtype=float)

    results: List[Dict[str, Any]] = []
    curves_for_plot: List[Dict[str, Any]] = []

    for zb in z_meta:
        datasets = [
            _load_dataset(
                data_dir=data_dir,
                recon=recon,
                zbin=zb,
                region=region,
                k_min=k_min,
                k_max=k_max,
            )
            for region in regions
        ]
        window_ops: Optional[Dict[str, WindowOp]] = None
        if use_window:
            window_ops = {}
            for ds in datasets:
                ker = _build_window_kernel(
                    data_dir=data_dir,
                    zbin=ds.zbin,
                    region=ds.region,
                    k_in=k_in,
                    s_max=window_s_max,
                )
                k_mat = ker.matrix_for_k_out(ds.k)
                window_ops[str(ds.region)] = WindowOp(kernel=ker, k_data=ds.k, k_mat=k_mat)
        y_size = int(sum(ds.y.size for ds in datasets))
        n_regions = int(len(datasets))
        dof_free = int(y_size - (8 * n_regions + 2))
        dof_eps_fixed = int(y_size - (8 * n_regions + 1))

        best_free = _scan_grid_joint(
            datasets=datasets,
            window_ops=window_ops,
            alpha_grid=alpha_grid,
            eps_grid=eps_grid,
            k0=k0_peak,
            sigma_k=sigma_k,
            mu=mu,
            w=w,
            p2_fid=p2_fid,
            sqrt1mu2=sqrt1mu2,
            return_eps_profile=True,
        )
        best_eps0 = _scan_grid_joint(
            datasets=datasets,
            window_ops=window_ops,
            alpha_grid=alpha_grid,
            eps_grid=np.asarray([0.0], dtype=float),
            k0=k0_peak,
            sigma_k=sigma_k,
            mu=mu,
            w=w,
            p2_fid=p2_fid,
            sqrt1mu2=sqrt1mu2,
            return_eps_profile=False,
        )

        f_ap_pbg = _f_ap_pbg_exponential(zb.z_eff)
        f_ap_fid = _f_ap_lcdm_flat(zb.z_eff, omega_m=omega_m, n_grid=lcdm_n_grid)
        eps_pbg = _eps_from_f_ap_ratio(f_ap_model=f_ap_pbg, f_ap_fid=f_ap_fid)
        best_eps_pbg = _scan_grid_joint(
            datasets=datasets,
            window_ops=window_ops,
            alpha_grid=alpha_grid,
            eps_grid=np.asarray([eps_pbg], dtype=float),
            k0=k0_peak,
            sigma_k=sigma_k,
            mu=mu,
            w=w,
            p2_fid=p2_fid,
            sqrt1mu2=sqrt1mu2,
            return_eps_profile=False,
        )

        eps_profile = list(best_free.get("eps_profile") or [])
        eps_grid_vals = np.array([float(r["eps"]) for r in eps_profile], dtype=float) if eps_profile else np.array([], dtype=float)
        chi2_prof = np.array([float(r["chi2"]) for r in eps_profile], dtype=float) if eps_profile else np.array([], dtype=float)
        eps_ci_1_lo, eps_ci_1_hi = _profile_ci(x=eps_grid_vals, chi2=chi2_prof, delta=1.0)
        eps_ci_2_lo, eps_ci_2_hi = _profile_ci(x=eps_grid_vals, chi2=chi2_prof, delta=4.0)

        rec = {
            "zbin": zb.zbin,
            "z_eff": zb.z_eff,
            "label": zb.label,
            "recon": recon,
            "regions": list(regions),
            "fit_range_k_h_mpc": [k_min, k_max],
            "template_peak": {"k0_h_mpc": k0_peak, "sigma_k_h_mpc": sigma_k},
            "window": {
                "enabled": use_window,
                "s_max_h_mpc": window_s_max if use_window else None,
                "k_in_max_h_mpc": window_k_in_max if use_window else None,
                "k_in_step_h_mpc": window_k_in_step if use_window else None,
            },
            "fid_reference": {"lcdm_flat": {"Omega_m": omega_m, "F_AP": f_ap_fid}},
            "model_reference": {"P_bg_exponential": {"F_AP": f_ap_pbg, "eps_pred": eps_pbg}},
            "fit": {
                "free": {
                    "alpha": float(best_free["alpha"]),
                    "eps": float(best_free["eps"]),
                    "chi2": float(best_free["chi2"]),
                    "dof": dof_free,
                    "chi2_dof": float(best_free["chi2"]) / float(dof_free) if dof_free > 0 else float("nan"),
                    "eps_ci_1sigma": [eps_ci_1_lo, eps_ci_1_hi],
                    "eps_ci_2sigma": [eps_ci_2_lo, eps_ci_2_hi],
                    "per_region_chi2": {k: float(v["chi2"]) for k, v in best_free["per_region"].items()},
                },
                "eps_fixed_0": {
                    "alpha": float(best_eps0["alpha"]),
                    "eps": 0.0,
                    "chi2": float(best_eps0["chi2"]),
                    "dof": dof_eps_fixed,
                    "chi2_dof": float(best_eps0["chi2"]) / float(dof_eps_fixed) if dof_eps_fixed > 0 else float("nan"),
                    "per_region_chi2": {k: float(v["chi2"]) for k, v in best_eps0["per_region"].items()},
                },
                "eps_fixed_pbg": {
                    "alpha": float(best_eps_pbg["alpha"]),
                    "eps": float(eps_pbg),
                    "chi2": float(best_eps_pbg["chi2"]),
                    "dof": dof_eps_fixed,
                    "chi2_dof": float(best_eps_pbg["chi2"]) / float(dof_eps_fixed) if dof_eps_fixed > 0 else float("nan"),
                    "per_region_chi2": {k: float(v["chi2"]) for k, v in best_eps_pbg["per_region"].items()},
                },
            },
        }
        results.append(rec)

        # Curves for plotting (free only; keep per-region linear params).
        per_region_pack: List[Dict[str, Any]] = []
        for ds in datasets:
            k_grid = np.linspace(float(np.min(ds.k)), float(np.max(ds.k)), 300, dtype=float)
            x_lin = np.asarray(best_free["per_region"][ds.region]["x"], dtype=float)
            if use_window:
                wop = window_ops.get(str(ds.region)) if window_ops else None
                if wop is None:
                    raise RuntimeError("window_ops missing for region in plot")
                mtx_u = _design_matrix(
                    wop.kernel.k_in,
                    alpha=float(best_free["alpha"]),
                    eps=float(best_free["eps"]),
                    k0=k0_peak,
                    sigma_k=sigma_k,
                    mu=mu,
                    w=w,
                    p2_fid=p2_fid,
                    sqrt1mu2=sqrt1mu2,
                )
                y_u = mtx_u @ x_lin
                k_mat_grid = wop.kernel.matrix_for_k_out(k_grid)
                y_c = k_mat_grid @ y_u
                n_grid = int(k_grid.size)
                p0_fit = y_c[:n_grid]
                p2_fit = y_c[n_grid:]
            else:
                p0_fit, p2_fit = _predict_curve(
                    k_grid=k_grid,
                    x=x_lin,
                    alpha=float(best_free["alpha"]),
                    eps=float(best_free["eps"]),
                    k0=k0_peak,
                    sigma_k=sigma_k,
                    mu=mu,
                    w=w,
                    p2_fid=p2_fid,
                    sqrt1mu2=sqrt1mu2,
                )
            n = int(ds.k.size)
            per_region_pack.append(
                {
                    "region": ds.region,
                    "k_data": ds.k,
                    "p0_data": ds.y[:n],
                    "p2_data": ds.y[n:],
                    "sig0": ds.sig0,
                    "sig2": ds.sig2,
                    "k_grid": k_grid,
                    "p0_fit": p0_fit,
                    "p2_fit": p2_fit,
                }
            )
        curves_for_plot.append(
            {
                "zbin": zb.zbin,
                "z_eff": zb.z_eff,
                "label": zb.label,
                "recon": recon,
                "regions": list(regions),
                "per_region": per_region_pack,
                "fit_annot": {
                    "free": {"alpha": float(best_free["alpha"]), "eps": float(best_free["eps"]), "chi2": float(best_free["chi2"])},
                    "eps0": {"alpha": float(best_eps0["alpha"]), "chi2": float(best_eps0["chi2"])},
                    "pbg": {"alpha": float(best_eps_pbg["alpha"]), "eps": float(eps_pbg), "chi2": float(best_eps_pbg["chi2"])},
                    "dof_free": dof_free,
                    "dof_eps_fixed": dof_eps_fixed,
                },
            }
        )

    out_dir = _ROOT / "output" / "cosmology"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if recon == "postrecon" else "_pre_recon"
    if use_window:
        suffix += "_window"
    out_png = out_dir / f"cosmology_bao_pk_multipole_peakfit{suffix}.png"
    out_json = out_dir / f"cosmology_bao_pk_multipole_peakfit{suffix}_metrics.json"

    # Plot
    _set_japanese_font()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(curves_for_plot), 2, figsize=(16, 4.8 * len(curves_for_plot)), sharex=True)
    if len(curves_for_plot) == 1:
        axes = np.array(axes).reshape(1, 2)

    color_map = {"NGC": "#1f77b4", "SGC": "#ff7f0e"}
    marker_map = {"NGC": "o", "SGC": "s"}

    for row, pack in enumerate(curves_for_plot):
        ax0 = axes[row, 0]
        ax2 = axes[row, 1]

        # Monopole / Quadrupole panels
        for pr in pack["per_region"]:
            region = str(pr["region"])
            col = color_map.get(region, "#333333")
            mk = marker_map.get(region, "o")
            ax0.errorbar(
                pr["k_data"],
                pr["p0_data"],
                yerr=pr["sig0"],
                fmt=mk,
                ms=4,
                capsize=2,
                color=col,
                ecolor=col,
                alpha=0.9,
                label=f"観測 P0（{region}）" if row == 0 else None,
            )
            ax0.plot(pr["k_grid"], pr["p0_fit"], color=col, linewidth=2.0, alpha=0.9, label=f"fit（{region}）" if row == 0 else None)

            ax2.errorbar(
                pr["k_data"],
                pr["p2_data"],
                yerr=pr["sig2"],
                fmt=mk,
                ms=4,
                capsize=2,
                color=col,
                ecolor=col,
                alpha=0.9,
                label=f"観測 P2（{region}）" if row == 0 else None,
            )
            ax2.plot(pr["k_grid"], pr["p2_fit"], color=col, linewidth=2.0, alpha=0.9, label=f"fit（{region}）" if row == 0 else None)

        ax0.set_ylabel("P0(k)", fontsize=11)
        ax2.set_ylabel("P2(k)", fontsize=11)
        ax0.grid(True, linestyle="--", alpha=0.5)
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax0.set_title(f"P0（{pack['label']} / {pack['recon']}）", fontsize=12)
        ax2.set_title(f"P2（{pack['label']} / {pack['recon']}）", fontsize=12)

        ann = pack["fit_annot"]
        dof_free = int(ann.get("dof_free", 0))
        dof_fixed = int(ann.get("dof_eps_fixed", 0))
        chi2d_free = ann["free"]["chi2"] / dof_free if dof_free > 0 else float("nan")
        chi2d_eps0 = ann["eps0"]["chi2"] / dof_fixed if dof_fixed > 0 else float("nan")
        chi2d_pbg = ann["pbg"]["chi2"] / dof_fixed if dof_fixed > 0 else float("nan")
        text = (
            f"k∈[{k_min:.3f},{k_max:.3f}] h/Mpc\n"
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

    for ax in axes[-1, :]:
        ax.set_xlabel("k [h/Mpc]（fid）", fontsize=11)
    if len(curves_for_plot) > 0:
        axes[0, 0].legend(fontsize=9, loc="upper right")
        axes[0, 1].legend(fontsize=9, loc="upper right")

    title = "宇宙論（BAO一次統計）：BOSS DR12 P(k) multipoles（ℓ=0,2）のピークfit（smooth+peak）"
    if use_window:
        title += "（窓関数込み）"
    fig.suptitle(title, fontsize=14)
    if use_window:
        note = "注：窓関数（RR multipoles; Beutler window）でモデルを畳み込み。テンプレート/RSD/full-shape 等の詳細は次工程で扱う。"
    else:
        note = "注：P(k)はBAO解析の一次統計の一部であり、窓関数/再構成/テンプレート等の詳細は次工程（P(k) full fit）で扱う。"
    fig.text(0.5, 0.01, note, ha="center", fontsize=10)
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.94))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "domain": "cosmology",
        "step": "16.4 (BAO re-derivation cross-check: P(k) multipoles peak fit)",
        "dataset": "beutler_boss_dr12_powspec",
        "inputs": {
            "data_dir": str(data_dir).replace("\\", "/"),
            "recon": recon,
            "z_bins": [z.zbin for z in z_meta],
            "regions": list(regions),
        },
        "fit": {
            "k_range_h_mpc": [k_min, k_max],
            "mu_n": mu_n,
            "basis": {"smooth": ["1", "k", "k^2"], "peak": "Gaussian(k0,sigma_k)"},
            "nonlinear_params": ["alpha", "eps"],
            "template_peak": {"k0_h_mpc": k0_peak, "sigma_k_h_mpc": sigma_k},
            "window": {
                "enabled": use_window,
                "s_max_h_mpc": window_s_max if use_window else None,
                "k_in_max_h_mpc": window_k_in_max if use_window else None,
                "k_in_step_h_mpc": window_k_in_step if use_window else None,
                "l_window_max": 8 if use_window else None,
            },
        },
        "notes": [
            "P_bg の ε(pred) は、距離比（r_d を含む）ではなく F_AP=D_M H / c の比から算出（r_d が相殺）。",
            "ここでは“入口の最小モデル”として smooth+peak を用いる（full-shape / RSD の詳細は次工程）。",
            "窓関数は --window 時に RR multipoles（Beutler window）で畳み込みを適用する（設定は metrics_json の fit.window を参照）。離散Hankelの有限範囲誤差を抑えるため、identity-window が概ね I になるよう擬似逆行列で補正（WindowKernel.c0/c2）を入れている。"
            if use_window
            else "窓関数は未適用（次工程で導入予定）。",
        ],
        "results": results,
        "outputs": {"png": str(out_png).replace("\\", "/"), "metrics_json": str(out_json).replace("\\", "/")},
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")

    try:
        event_type = "cosmology_bao_pk_multipole_peakfit"
        if recon != "postrecon":
            event_type += "_pre_recon"
        if use_window:
            event_type += "_window"
        worklog.append_event(
            {
                "event_type": event_type,
                "argv": list(sys.argv),
                "inputs": {"data_dir": data_dir, "recon": recon, "z_bins": [z.zbin for z in z_meta], "regions": list(regions)},
                "outputs": {"png": out_png, "metrics_json": out_json},
                "metrics": {"k_range_h_mpc": [k_min, k_max], "mu_n": mu_n},
            }
        )
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
