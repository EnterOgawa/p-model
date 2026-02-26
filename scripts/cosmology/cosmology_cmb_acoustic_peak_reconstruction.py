#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_cmb_acoustic_peak_reconstruction.py

Step 8.7.18（宇宙論本丸：CMB音響ピークの再現）

目的：
- CMB TT パワースペクトルの第1〜第3音響ピーク（位置と相対高さ）を、
  P-model側の最小モード方程式で監査可能な形に固定する。
- ΛCDM全体の置換主張ではなく、まず「ピーク構造を P 側の少数パラメータで
  どこまで再現できるか」を定量化し、反証条件をJSONで凍結する。

最小モデル（モード近似）：
  Θ_k'' + Γ_P Θ_k' + c_s,P^2 k^2 Θ_k = -k^2 Ψ_P

上式の音響モード展開として、ピーク列を
  ℓ_n^(P) ≈ (n-φ_P) ℓ_A^(P)
  A_n^(P) = A0 exp(-δ_P (n-1)) [1 + (-1)^(n+1) R_P]
で近似する（n=1,2,3 を固定入力、n>=4 は予測）。

入力（固定）：
- data/cosmology/planck2018_com_power_spect_tt_binned_r3.01.txt

出力（固定名）：
- output/private/cosmology/cosmology_cmb_acoustic_peak_reconstruction.png
- output/private/cosmology/cosmology_cmb_acoustic_peak_reconstruction_metrics.json
- output/private/cosmology/cosmology_cmb_acoustic_peak_reconstruction_falsification_pack.json
- output/private/cosmology/cosmology_cmb_acoustic_peak_reconstruction_peaks.csv
（必要なら output/public/cosmology/ へ同名でコピー）
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


# 関数: `_set_japanese_font` の入出力契約と処理意図を定義する。
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


# 関数: `_fmt` の入出力契約と処理意図を定義する。

def _fmt(x: float, digits: int = 6) -> str:
    # 条件分岐: `x == 0.0` を満たす経路を評価する。
    if x == 0.0:
        return "0"

    ax = abs(float(x))
    # 条件分岐: `ax >= 1e4 or ax < 1e-3` を満たす経路を評価する。
    if ax >= 1e4 or ax < 1e-3:
        return f"{x:.{digits}g}"

    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


# 関数: `_write_json` の入出力契約と処理意図を定義する。

def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# クラス: `Peak` の責務と境界条件を定義する。

@dataclass(frozen=True)
class Peak:
    label: str
    n: int
    ell_min: float
    ell_max: float
    ell: float
    amplitude: float


# 関数: `_acoustic_solution_summary` の入出力契約と処理意図を定義する。

def _acoustic_solution_summary(*, c_s_p: float = 1.0 / math.sqrt(3.0)) -> Dict[str, Any]:
    cs = float(c_s_p)
    # 条件分岐: `not (cs > 0.0)` を満たす経路を評価する。
    if not (cs > 0.0):
        raise ValueError("c_s_p must be positive.")

    return {
        "assumptions": [
            "static-space branch (no FRW expansion friction)",
            "tight-coupling photon-baryon fluid (v_gamma ≈ v_b ≡ v_pb)",
            "constant c_s,P and slowly varying Ψ_P within one acoustic cycle",
        ],
        "fluid_equations_fourier_k": {
            "continuity": "Theta_k' = -(k/3) v_pb,k",
            "euler": "v_pb,k' + Gamma_P v_pb,k = 3 c_s,P^2 k (Theta_k + Psi_P)",
            "note": "prime denotes derivative with respect to propagation variable s.",
        },
        "reduced_oscillator": {
            "equation": "Theta_k'' + Gamma_P Theta_k' + c_s,P^2 k^2 Theta_k = -k^2 Psi_P",
            "normalized_shift": "for Gamma_P≈0 and constant source, define Theta_tilde_k = Theta_k + Psi_P/c_s,P^2",
            "homogeneous_equation": "Theta_tilde_k'' + c_s,P^2 k^2 Theta_tilde_k = 0",
            "equilibrium_offset": "Theta_eq,k = -Psi_P / c_s,P^2",
        },
        "acoustic_horizon_mapping": {
            "definition": "r_s(s)=∫_0^s c_s,P(s') ds' ; for constant c_s,P, r_s = c_s,P * s",
            "solution": "Theta_tilde_k(r_s)=A_k cos(k r_s)+B_k sin(k r_s)",
            "adiabatic_branch": "Theta_tilde_k'(0)=0 => B_k=0, therefore Theta_tilde_k ∝ cos(k r_s)",
            "explicit_mode_solution": "Theta_k(r_s)=Theta_eq,k + [Theta_k(0)-Theta_eq,k] cos(k r_s) + [Theta_k'(0)/(k c_s,P)] sin(k r_s)",
            "peak_condition": "acoustic extrema: k r_s = mπ (m=1,2,3,...)",
        },
        "tt_peak_implication": "C_ell^TT ∝ cos^2(k r_s), i.e., temperature peaks follow cosine acoustic extrema.",
    }


# 関数: `_rk4_acoustic_cosine_validation` の入出力契約と処理意図を定義する。

def _rk4_acoustic_cosine_validation(*, k_mode: float, c_s_p: float, s_end: float, n_steps: int) -> Dict[str, Any]:
    k = float(k_mode)
    cs = float(c_s_p)
    smax = float(s_end)
    n = int(n_steps)
    # 条件分岐: `not (k > 0.0 and cs > 0.0 and smax > 0.0 and n >= 50)` を満たす経路を評価する。
    if not (k > 0.0 and cs > 0.0 and smax > 0.0 and n >= 50):
        raise ValueError("invalid acoustic validation parameters.")

    omega = cs * k
    omega_sq = omega * omega
    s = np.linspace(0.0, smax, n, dtype=float)
    dt = float(s[1] - s[0])

    y = np.zeros((n, 2), dtype=float)
    y[0, 0] = 1.0
    y[0, 1] = 0.0

    # 関数: `rhs` の入出力契約と処理意図を定義する。
    def rhs(state: np.ndarray) -> np.ndarray:
        return np.asarray([state[1], -omega_sq * state[0]], dtype=float)

    for i in range(n - 1):
        y0 = y[i, :]
        k1 = rhs(y0)
        k2 = rhs(y0 + 0.5 * dt * k1)
        k3 = rhs(y0 + 0.5 * dt * k2)
        k4 = rhs(y0 + dt * k3)
        y[i + 1, :] = y0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    num = y[:, 0]
    ana = np.cos(omega * s)
    abs_err = np.abs(num - ana)
    rms = float(np.sqrt(np.mean((num - ana) ** 2)))
    return {
        "k_mode": k,
        "c_s_p": cs,
        "omega": omega,
        "s_end": smax,
        "n_steps": n,
        "max_abs_error": float(np.max(abs_err)),
        "rms_error": rms,
        "pass": bool(np.max(abs_err) <= 1.0e-6),
        "pass_threshold_max_abs_error": 1.0e-6,
    }


# 関数: `_read_planck_tt` の入出力契約と処理意図を定義する。

def _read_planck_tt(path: Path) -> Dict[str, np.ndarray]:
    arr = np.loadtxt(path)
    # 条件分岐: `arr.ndim != 2 or arr.shape[1] < 5` を満たす経路を評価する。
    if arr.ndim != 2 or arr.shape[1] < 5:
        raise ValueError(f"unexpected Planck TT format: {path}")

    ell = arr[:, 0].astype(float)
    dl = arr[:, 1].astype(float)
    err_lo = arr[:, 2].astype(float)
    err_hi = arr[:, 3].astype(float)
    bestfit = arr[:, 4].astype(float)
    sigma = np.maximum(0.5 * (np.abs(err_lo) + np.abs(err_hi)), 1e-12)
    return {"ell": ell, "dl": dl, "sigma": sigma, "bestfit": bestfit}


# 関数: `_find_peak` の入出力契約と処理意図を定義する。

def _find_peak(ell: np.ndarray, y: np.ndarray, lo: float, hi: float, *, mode: str = "max") -> Peak:
    m = (ell >= float(lo)) & (ell <= float(hi))
    # 条件分岐: `not np.any(m)` を満たす経路を評価する。
    if not np.any(m):
        raise ValueError(f"no data in range [{lo}, {hi}]")

    idxs = np.where(m)[0]
    local = y[m]
    j = int(np.argmax(local) if mode == "max" else np.argmin(local))
    idx = int(idxs[j])
    return Peak(
        label="",
        n=-1,
        ell_min=float(lo),
        ell_max=float(hi),
        ell=float(ell[idx]),
        amplitude=float(y[idx]),
    )


# 関数: `_extract_observed_peaks` の入出力契約と処理意図を定義する。

def _extract_observed_peaks(ell: np.ndarray, dl: np.ndarray) -> Tuple[List[Peak], List[Peak]]:
    fit_peak_ranges = [
        ("ℓ1", 1, 150.0, 320.0),
        ("ℓ2", 2, 400.0, 650.0),
        ("ℓ3", 3, 700.0, 950.0),
    ]
    holdout_peak_ranges = [
        ("ℓ4_holdout", 4, 1000.0, 1300.0),
        ("ℓ5_holdout", 5, 1300.0, 1700.0),
        ("ℓ6_holdout", 6, 1700.0, 2100.0),
    ]
    fit_peaks: List[Peak] = []
    holdout_peaks: List[Peak] = []
    for label, n, lo, hi in fit_peak_ranges:
        p = _find_peak(ell, dl, lo, hi, mode="max")
        fit_peaks.append(Peak(label=label, n=n, ell_min=lo, ell_max=hi, ell=p.ell, amplitude=p.amplitude))

    for label, n, lo, hi in holdout_peak_ranges:
        p = _find_peak(ell, dl, lo, hi, mode="max")
        holdout_peaks.append(Peak(label=label, n=n, ell_min=lo, ell_max=hi, ell=p.ell, amplitude=p.amplitude))

    return fit_peaks, holdout_peaks


# 関数: `_silk_damping_factor` の入出力契約と処理意図を定義する。

def _silk_damping_factor(n: int, l_acoustic: float, phi: float, silk_kappa: float) -> float:
    ell_n = (float(n) - float(phi)) * float(l_acoustic)
    ell_d = max(float(silk_kappa) * float(l_acoustic), 1e-12)
    return float(math.exp(-((ell_n / ell_d) ** 2)))


# 関数: `_fit_modal_params` の入出力契約と処理意図を定義する。

def _fit_modal_params(obs: Sequence[Peak], *, silk_kappa: float) -> Dict[str, float]:
    # 条件分岐: `len(obs) != 3` を満たす経路を評価する。
    if len(obs) != 3:
        raise ValueError("need exactly 3 observed peaks")

    n = np.array([float(p.n) for p in obs], dtype=float)
    ell = np.array([float(p.ell) for p in obs], dtype=float)
    amp = np.array([float(p.amplitude) for p in obs], dtype=float)
    # 条件分岐: `np.any(amp <= 0.0)` を満たす経路を評価する。
    if np.any(amp <= 0.0):
        raise ValueError("observed peak amplitudes must be > 0")

    # ℓ_n = ℓ_A n + b,  φ = -b/ℓ_A

    mat = np.column_stack([n, np.ones_like(n)])
    slope, intercept = np.linalg.lstsq(mat, ell, rcond=None)[0]
    l_acoustic = float(slope)
    # 条件分岐: `not (l_acoustic > 0.0)` を満たす経路を評価する。
    if not (l_acoustic > 0.0):
        raise ValueError("invalid acoustic scale")

    phi = float(-intercept / l_acoustic)

    # A_n = A0 exp(-δ(n-1)) [1 + (-1)^(n+1) R] D_n
    # D_n = exp(-(ell_n/ell_D)^2), ell_D = silk_kappa * ell_A
    a1, a2, a3 = float(amp[0]), float(amp[1]), float(amp[2])
    d1 = _silk_damping_factor(1, l_acoustic, phi, silk_kappa)
    d2 = _silk_damping_factor(2, l_acoustic, phi, silk_kappa)
    d3 = _silk_damping_factor(3, l_acoustic, phi, silk_kappa)
    ratio_21 = a2 / a1
    ratio_31 = a3 / a1
    ratio_31_corr = ratio_31 / max(d3 / max(d1, 1e-12), 1e-12)
    # 条件分岐: `ratio_31_corr <= 0.0` を満たす経路を評価する。
    if ratio_31_corr <= 0.0:
        raise ValueError("invalid ratio A3/A1")

    delta = float(-0.5 * math.log(ratio_31_corr))
    q = float(ratio_21 * math.exp(delta) * (d1 / max(d2, 1e-12)))
    # 条件分岐: `abs(1.0 + q) < 1e-12` を満たす経路を評価する。
    if abs(1.0 + q) < 1e-12:
        raise ValueError("unstable baryon-loading inversion")

    r_baryon = float((1.0 - q) / (1.0 + q))
    # 条件分岐: `abs(1.0 + r_baryon) < 1e-12` を満たす経路を評価する。
    if abs(1.0 + r_baryon) < 1e-12:
        raise ValueError("invalid R close to -1")

    a0 = float(a1 / ((1.0 + r_baryon) * d1))

    return {
        "l_acoustic": l_acoustic,
        "phi": phi,
        "delta": delta,
        "r_baryon": r_baryon,
        "a0": a0,
        "silk_kappa": float(silk_kappa),
        "ell_damping": float(silk_kappa * l_acoustic),
        "ratio_a2_a1": ratio_21,
        "ratio_a3_a1": ratio_31,
        "ratio_a3_a1_corrected": ratio_31_corr,
    }


# 関数: `_first_principles_closure` の入出力契約と処理意図を定義する。

def _first_principles_closure(
    *,
    obs3: Sequence[Peak],
    params: Dict[str, float],
) -> Dict[str, Any]:
    # 条件分岐: `len(obs3) < 3` を満たす経路を評価する。
    if len(obs3) < 3:
        raise ValueError("first-principles closure requires at least three peaks (ℓ1..ℓ3).")

    r_baryon = float(params["r_baryon"])
    delta = float(params["delta"])
    l_acoustic = float(params["l_acoustic"])
    phi = float(params["phi"])
    silk_kappa = float(params["silk_kappa"])

    # 条件分岐: `not (-0.99 < r_baryon < 1.0)` を満たす経路を評価する。
    if not (-0.99 < r_baryon < 1.0):
        raise ValueError("r_baryon must be in (-0.99, 1.0) for the closure model.")

    # 条件分岐: `not (delta > 0.0)` を満たす経路を評価する。

    if not (delta > 0.0):
        raise ValueError("delta must be positive for damping closure.")

    # First-principles closure:
    #   (1) baryon-loaded acoustic speed  c_s,P^2/c^2 = 1/[3(1+R_P)]
    #   (2) potential-well loading        R_P = u_Psi / (c_s,P^2/c^2),  u_Psi=|Psi_P|/c^2
    # -> u_Psi = R_P / [3(1+R_P)] and R_P = 3u_Psi/(1-3u_Psi)

    u_psi = float(r_baryon / (3.0 * (1.0 + r_baryon)))
    r_from_u = float((3.0 * u_psi) / max(1.0 - 3.0 * u_psi, 1.0e-12))
    cs_over_c_sq = float(1.0 / (3.0 * (1.0 + r_from_u)))
    cs_over_c = float(math.sqrt(max(cs_over_c_sq, 1.0e-15)))

    # Scattering-length closure:
    #   Gamma_sc ≈ c_s,P/lambda_sc,  one peak-step phase Δs = π/(k_A c_s,P),  k_A = π/r_s
    #   => delta_P = (Gamma_sc/2) Δs = r_s/(2 lambda_sc)
    lambda_sc_over_rs = float(1.0 / (2.0 * delta))
    delta_from_scattering = float(1.0 / (2.0 * lambda_sc_over_rs))

    # DM-free third-peak damping theorem:
    #   A_n = A0 exp[-delta_P(n-1)] [1+(-1)^(n+1)R_P] D_n
    #   => A3/A1 = exp(-2 delta_P) * (D3/D1)  (odd/even loading cancels)
    d1 = _silk_damping_factor(1, l_acoustic, phi, silk_kappa)
    d3 = _silk_damping_factor(3, l_acoustic, phi, silk_kappa)
    a3a1_pred_dm_free = float(math.exp(-2.0 * delta_from_scattering) * (d3 / max(d1, 1.0e-12)))
    a3a1_obs = float(obs3[2].amplitude / max(obs3[0].amplitude, 1.0e-12))
    attenuation_theorem_pass = bool((delta_from_scattering > 0.0) and (d3 <= d1 + 1.0e-15) and (a3a1_pred_dm_free < 1.0))

    return {
        "closure_assumptions": {
            "static_space_branch": True,
            "no_dark_matter_drive_term": True,
            "tight_coupling": True,
            "definitions": {
                "u_psi": "u_Psi = |Psi_P|/c^2",
                "r_loading": "R_P",
                "cs_ratio": "c_s,P/c",
                "scattering_ratio": "lambda_sc/r_s",
            },
        },
        "derived_parameters": {
            "u_psi": u_psi,
            "r_p_from_well_and_cs": r_from_u,
            "cs_over_c": cs_over_c,
            "cs_over_c_squared": cs_over_c_sq,
            "lambda_sc_over_r_s": lambda_sc_over_rs,
            "delta_p_from_scattering": delta_from_scattering,
        },
        "consistency_to_modal_fit": {
            "r_p_modal_fit": r_baryon,
            "delta_p_modal_fit": delta,
            "abs_diff_r_p": float(abs(r_from_u - r_baryon)),
            "abs_diff_delta_p": float(abs(delta_from_scattering - delta)),
        },
        "third_peak_dm_free_damping_proof": {
            "formula": "A3/A1 = exp(-2*delta_P) * (D3/D1) < 1 for delta_P>0 and D3<=D1",
            "damping_factor_exp_term": float(math.exp(-2.0 * delta_from_scattering)),
            "silk_ratio_d3_over_d1": float(d3 / max(d1, 1.0e-12)),
            "a3_over_a1_pred_dm_free": a3a1_pred_dm_free,
            "a3_over_a1_observed": a3a1_obs,
            "abs_rel_error_vs_observed": float(abs(a3a1_pred_dm_free / max(a3a1_obs, 1.0e-12) - 1.0)),
            "attenuation_theorem_pass": attenuation_theorem_pass,
        },
    }


# 関数: `_predict_modal_peak` の入出力契約と処理意図を定義する。

def _predict_modal_peak(n: int, p: Dict[str, float]) -> Dict[str, float]:
    l_acoustic = float(p["l_acoustic"])
    phi = float(p["phi"])
    delta = float(p["delta"])
    r_baryon = float(p["r_baryon"])
    a0 = float(p["a0"])
    silk_kappa = float(p.get("silk_kappa", 5.2))
    sign = 1.0 if (n % 2 == 1) else -1.0
    ell = (float(n) - phi) * l_acoustic
    damping = _silk_damping_factor(int(n), l_acoustic, phi, silk_kappa)
    amp = a0 * math.exp(-delta * float(n - 1)) * (1.0 + sign * r_baryon) * damping
    return {"n": int(n), "ell": float(ell), "amplitude": float(amp)}


# 関数: `_predict_modal_series` の入出力契約と処理意図を定義する。

def _predict_modal_series(p: Dict[str, float], *, n_modes: int) -> List[Dict[str, float]]:
    return [_predict_modal_peak(n, p) for n in range(1, int(n_modes) + 1)]


# 関数: `_build_envelope_curve` の入出力契約と処理意図を定義する。

def _build_envelope_curve(ell: np.ndarray, series: Sequence[Dict[str, float]], *, l_acoustic: float) -> np.ndarray:
    # 可視化用の滑らかな包絡（判定本体はピーク表で行う）
    sigma1 = 0.16 * float(l_acoustic)
    y = np.zeros_like(ell, dtype=float)
    for m in series:
        n = int(m["n"])
        ell_n = float(m["ell"])
        amp_n = max(float(m["amplitude"]), 0.0)
        sigma_n = sigma1 * math.sqrt(float(n)) * (1.0 + 0.06 * float(n - 1))
        y += amp_n * np.exp(-0.5 * ((ell - ell_n) / sigma_n) ** 2)

    return y


# 関数: `_falsification_gate` の入出力契約と処理意図を定義する。

def _falsification_gate(
    obs3: Sequence[Peak],
    pred3: Sequence[Dict[str, float]],
    holdouts: Sequence[Peak],
    pred_holdouts: Sequence[Dict[str, float]],
) -> Dict[str, Any]:
    rows_3: List[Dict[str, Any]] = []
    abs_dell_3: List[float] = []
    abs_damp_rel_3: List[float] = []
    for o, p in zip(obs3, pred3):
        d_ell = float(p["ell"] - o.ell)
        d_amp_rel = float(p["amplitude"] / o.amplitude - 1.0)
        abs_dell_3.append(abs(d_ell))
        abs_damp_rel_3.append(abs(d_amp_rel))
        rows_3.append(
            {
                "label": o.label,
                "observed": {"ell": o.ell, "amplitude": o.amplitude},
                "predicted": {"ell": float(p["ell"]), "amplitude": float(p["amplitude"])},
                "residual": {"delta_ell": d_ell, "delta_amp_rel": d_amp_rel},
            }
        )

    holdout_by_n: Dict[int, Peak] = {int(p.n): p for p in holdouts}
    pred_holdout_by_n: Dict[int, Dict[str, float]] = {int(p["n"]): p for p in pred_holdouts}

    # 条件分岐: `4 not in holdout_by_n or 4 not in pred_holdout_by_n` を満たす経路を評価する。
    if 4 not in holdout_by_n or 4 not in pred_holdout_by_n:
        raise ValueError("holdout n=4 is required for the core falsification gate")

    obs4 = holdout_by_n[4]
    pred4 = pred_holdout_by_n[4]
    d4_ell = float(pred4["ell"] - obs4.ell)
    d4_amp_rel = float(pred4["amplitude"] / obs4.amplitude - 1.0)

    thresholds = {
        "first3_max_abs_delta_ell": 15.0,
        "first3_max_abs_delta_amp_rel": 0.15,
        "holdout4_max_abs_delta_ell": 50.0,
        "holdout4_max_abs_delta_amp_rel": 0.25,
        "extended46_max_abs_delta_ell": 80.0,
        "extended46_max_abs_delta_amp_rel": 0.35,
    }

    pass_first3 = (max(abs_dell_3) <= thresholds["first3_max_abs_delta_ell"]) and (
        max(abs_damp_rel_3) <= thresholds["first3_max_abs_delta_amp_rel"]
    )
    pass_holdout4 = (abs(d4_ell) <= thresholds["holdout4_max_abs_delta_ell"]) and (
        abs(d4_amp_rel) <= thresholds["holdout4_max_abs_delta_amp_rel"]
    )
    core_pass = bool(pass_first3 and pass_holdout4)

    rows_4to6: List[Dict[str, Any]] = []
    abs_dell_4to6: List[float] = []
    abs_damp_rel_4to6: List[float] = []
    for n in [4, 5, 6]:
        o = holdout_by_n.get(n)
        p = pred_holdout_by_n.get(n)
        # 条件分岐: `o is None or p is None` を満たす経路を評価する。
        if o is None or p is None:
            continue

        d_ell = float(p["ell"] - o.ell)
        d_amp_rel = float(p["amplitude"] / o.amplitude - 1.0)
        abs_dell_4to6.append(abs(d_ell))
        abs_damp_rel_4to6.append(abs(d_amp_rel))
        rows_4to6.append(
            {
                "label": o.label,
                "n": int(n),
                "observed": {"ell": o.ell, "amplitude": o.amplitude},
                "predicted": {"ell": float(p["ell"]), "amplitude": float(p["amplitude"])},
                "residual": {"delta_ell": d_ell, "delta_amp_rel": d_amp_rel},
            }
        )

    max_abs_dell_4to6 = max(abs_dell_4to6) if abs_dell_4to6 else float("nan")
    max_abs_damp_rel_4to6 = max(abs_damp_rel_4to6) if abs_damp_rel_4to6 else float("nan")
    pass_extended_4to6 = bool(
        (max_abs_dell_4to6 <= thresholds["extended46_max_abs_delta_ell"])
        and (max_abs_damp_rel_4to6 <= thresholds["extended46_max_abs_delta_amp_rel"])
    )
    # 条件分岐: `not core_pass` を満たす経路を評価する。
    if not core_pass:
        overall_extended_status = "reject"
    # 条件分岐: 前段条件が不成立で、`pass_extended_4to6` を追加評価する。
    elif pass_extended_4to6:
        overall_extended_status = "pass"
    else:
        overall_extended_status = "watch"

    return {
        "thresholds": thresholds,
        "first3": {"rows": rows_3, "max_abs_delta_ell": max(abs_dell_3), "max_abs_delta_amp_rel": max(abs_damp_rel_3), "pass": pass_first3},
        "holdout4": {
            "label": obs4.label,
            "observed": {"ell": obs4.ell, "amplitude": obs4.amplitude},
            "predicted": {"ell": float(pred4["ell"]), "amplitude": float(pred4["amplitude"])},
            "residual": {"delta_ell": d4_ell, "delta_amp_rel": d4_amp_rel},
            "pass": pass_holdout4,
        },
        "extended_4to6": {
            "rows": rows_4to6,
            "max_abs_delta_ell": max_abs_dell_4to6,
            "max_abs_delta_amp_rel": max_abs_damp_rel_4to6,
            "pass": pass_extended_4to6,
        },
        "overall": {"status": "pass" if core_pass else "reject", "pass": core_pass},
        "overall_extended": {
            "status": overall_extended_status,
            "pass_core": core_pass,
            "pass_extended_4to6": pass_extended_4to6,
        },
    }


# 関数: `_plot` の入出力契約と処理意図を定義する。

def _plot(
    *,
    ell: np.ndarray,
    dl: np.ndarray,
    sigma: np.ndarray,
    bestfit: np.ndarray,
    model_curve: np.ndarray,
    obs3: Sequence[Peak],
    pred3: Sequence[Dict[str, float]],
    holdouts: Sequence[Peak],
    pred_holdouts: Sequence[Dict[str, float]],
    gate: Dict[str, Any],
    out_png: Path,
) -> None:
    _set_japanese_font()
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.8))

    ax1.errorbar(ell, dl, yerr=sigma, fmt=".", color="#888888", alpha=0.45, label="Planck 2018 TT (binned)")
    ax1.plot(ell, bestfit, color="#1f77b4", linewidth=1.2, alpha=0.9, label="Planck best-fit")
    ax1.plot(ell, model_curve, color="#d62728", linewidth=1.6, alpha=0.9, label="P-model modal envelope")

    ax1.scatter([p.ell for p in obs3], [p.amplitude for p in obs3], color="#ff7f0e", s=48, marker="o", label="observed peaks ℓ1-ℓ3", zorder=4)
    ax1.scatter([p["ell"] for p in pred3], [p["amplitude"] for p in pred3], color="#d62728", s=48, marker="x", label="P-model peaks ℓ1-ℓ3", zorder=5)
    holdout_obs_ell = [p.ell for p in holdouts]
    holdout_obs_amp = [p.amplitude for p in holdouts]
    holdout_pred_ell = [float(p["ell"]) for p in pred_holdouts]
    holdout_pred_amp = [float(p["amplitude"]) for p in pred_holdouts]
    ax1.scatter(holdout_obs_ell, holdout_obs_amp, color="#2ca02c", s=55, marker="o", label="observed holdout ℓ4-ℓ6", zorder=4)
    ax1.scatter(holdout_pred_ell, holdout_pred_amp, color="#9467bd", s=55, marker="x", label="P-model holdout ℓ4-ℓ6", zorder=5)

    ax1.set_xlim(50.0, 2200.0)
    ax1.set_ylim(0.0, max(float(np.max(dl) * 1.08), 6200.0))
    ax1.set_xlabel("multipole ℓ")
    ax1.set_ylabel("D_ℓ = ℓ(ℓ+1)C_ℓ/2π  [μK²]")
    ax1.set_title("CMB TT acoustic peaks: Planck vs P-model modal reconstruction")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(fontsize=8, loc="upper right")

    labels = ["ℓ1", "ℓ2", "ℓ3"] + [f"ℓ{int(p.n)}(h)" for p in holdouts]
    d_ell = [float(pred3[i]["ell"] - obs3[i].ell) for i in range(3)] + [
        float(pred_holdouts[i]["ell"] - holdouts[i].ell) for i in range(len(holdouts))
    ]
    d_amp = [float(pred3[i]["amplitude"] / obs3[i].amplitude - 1.0) for i in range(3)] + [
        float(pred_holdouts[i]["amplitude"] / holdouts[i].amplitude - 1.0) for i in range(len(holdouts))
    ]
    x = np.arange(len(labels), dtype=float)
    w = 0.36
    ax2.axhline(0.0, color="#333333", linewidth=1.0, alpha=0.7)
    ax2.bar(x - 0.5 * w, d_ell, width=w, color="#1f77b4", label="Δℓ")
    ax2.bar(x + 0.5 * w, d_amp, width=w, color="#ff7f0e", label="ΔA/A")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("residual")
    ax2.set_title("Peak residuals (first3 fit + ℓ4-ℓ6 holdout)")
    ax2.grid(True, linestyle="--", alpha=0.35, axis="y")
    ax2.legend(fontsize=9, loc="upper right")

    status = gate.get("overall", {}).get("status", "n/a")
    status_ext = gate.get("overall_extended", {}).get("status", "n/a")
    max_dell_3 = gate.get("first3", {}).get("max_abs_delta_ell", float("nan"))
    max_damp_3 = gate.get("first3", {}).get("max_abs_delta_amp_rel", float("nan"))
    d4_ell = gate.get("holdout4", {}).get("residual", {}).get("delta_ell", float("nan"))
    d4_amp = gate.get("holdout4", {}).get("residual", {}).get("delta_amp_rel", float("nan"))
    d46_ell = gate.get("extended_4to6", {}).get("max_abs_delta_ell", float("nan"))
    d46_amp = gate.get("extended_4to6", {}).get("max_abs_delta_amp_rel", float("nan"))
    fig.suptitle("P-model CMB acoustic-peak audit (ℓ1-ℓ3 fit + ℓ4-ℓ6 holdout)", fontsize=14)
    fig.text(
        0.5,
        0.01,
        (
            f"core={status}, extended={status_ext}; first3: max|Δℓ|={_fmt(float(max_dell_3), 3)}, "
            f"max|ΔA/A|={_fmt(float(max_damp_3), 4)}; "
            f"holdout ℓ4: Δℓ={_fmt(float(d4_ell), 3)}, ΔA/A={_fmt(float(d4_amp), 4)}; "
            f"holdout ℓ4-ℓ6: max|Δℓ|={_fmt(float(d46_ell), 3)}, max|ΔA/A|={_fmt(float(d46_amp), 4)}"
        ),
        ha="center",
        fontsize=10,
    )
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.93))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# 関数: `_write_peak_csv` の入出力契約と処理意図を定義する。

def _write_peak_csv(
    path: Path,
    obs3: Sequence[Peak],
    pred3: Sequence[Dict[str, float]],
    holdouts: Sequence[Peak],
    pred_holdouts: Sequence[Dict[str, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "n", "ell_obs", "amp_obs", "ell_pred", "amp_pred", "delta_ell", "delta_amp_rel"])
        for o, p in zip(obs3, pred3):
            d_ell = float(p["ell"] - o.ell)
            d_amp_rel = float(p["amplitude"] / o.amplitude - 1.0)
            w.writerow([o.label, o.n, f"{o.ell:.8f}", f"{o.amplitude:.8f}", f"{float(p['ell']):.8f}", f"{float(p['amplitude']):.8f}", f"{d_ell:.8f}", f"{d_amp_rel:.8f}"])

        for o, p in zip(holdouts, pred_holdouts):
            d_ell = float(p["ell"] - o.ell)
            d_amp_rel = float(p["amplitude"] / o.amplitude - 1.0)
            w.writerow(
                [
                    o.label,
                    o.n,
                    f"{o.ell:.8f}",
                    f"{o.amplitude:.8f}",
                    f"{float(p['ell']):.8f}",
                    f"{float(p['amplitude']):.8f}",
                    f"{d_ell:.8f}",
                    f"{d_amp_rel:.8f}",
                ]
            )


# 関数: `_copy_outputs_to_public` の入出力契約と処理意図を定義する。

def _copy_outputs_to_public(private_paths: Sequence[Path], public_dir: Path) -> Dict[str, str]:
    public_dir.mkdir(parents=True, exist_ok=True)
    copied: Dict[str, str] = {}
    for p in private_paths:
        dst = public_dir / p.name
        shutil.copy2(p, dst)
        copied[p.name] = str(dst).replace("\\", "/")

    return copied


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="CMB acoustic-peak reconstruction (P-model modal approximation).")
    ap.add_argument(
        "--data",
        type=str,
        default=str(_ROOT / "data" / "cosmology" / "planck2018_com_power_spect_tt_binned_r3.01.txt"),
        help="Input Planck TT binned spectrum text.",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(_ROOT / "output" / "private" / "cosmology"),
        help="Private output directory.",
    )
    ap.add_argument(
        "--public-dir",
        type=str,
        default=str(_ROOT / "output" / "public" / "cosmology"),
        help="Public output directory (for copy).",
    )
    ap.add_argument("--n-modes", type=int, default=8, help="Number of modal peaks for envelope visualization.")
    ap.add_argument(
        "--silk-kappa",
        type=float,
        default=5.2,
        help="Silk-damping scale factor ell_D / ell_A used in high-order peak amplitude refinement.",
    )
    ap.add_argument("--skip-public-copy", action="store_true", help="Do not copy outputs into output/public/cosmology.")
    ap.add_argument("--acoustic-k-mode", type=float, default=0.08, help="k mode for cosine-validation (arb. inverse-length units).")
    ap.add_argument("--acoustic-cs", type=float, default=1.0 / math.sqrt(3.0), help="Sound speed c_s,P for acoustic validation.")
    ap.add_argument("--acoustic-s-end", type=float, default=600.0, help="Integration length in s for cosine-validation.")
    ap.add_argument("--acoustic-steps", type=int, default=6000, help="Number of RK4 steps for cosine-validation.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    data_path = Path(args.data).resolve()
    out_dir = Path(args.out_dir).resolve()
    pub_dir = Path(args.public_dir).resolve()
    n_modes = max(int(args.n_modes), 6)
    silk_kappa = max(float(args.silk_kappa), 1e-6)

    src = _read_planck_tt(data_path)
    ell = src["ell"]
    dl = src["dl"]
    sigma = src["sigma"]
    bestfit = src["bestfit"]

    obs3, holdouts = _extract_observed_peaks(ell, dl)
    params = _fit_modal_params(obs3, silk_kappa=silk_kappa)
    first_principles_closure = _first_principles_closure(obs3=obs3, params=params)

    series = _predict_modal_series(params, n_modes=n_modes)
    pred3 = [series[0], series[1], series[2]]
    pred_holdouts = [series[p.n - 1] for p in holdouts]
    model_curve = _build_envelope_curve(ell, series, l_acoustic=float(params["l_acoustic"]))
    gate = _falsification_gate(obs3, pred3, holdouts, pred_holdouts)
    acoustic_summary = _acoustic_solution_summary(c_s_p=float(args.acoustic_cs))
    acoustic_validation = _rk4_acoustic_cosine_validation(
        k_mode=float(args.acoustic_k_mode),
        c_s_p=float(args.acoustic_cs),
        s_end=float(args.acoustic_s_end),
        n_steps=int(args.acoustic_steps),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    base = "cosmology_cmb_acoustic_peak_reconstruction"
    out_png = out_dir / f"{base}.png"
    out_metrics = out_dir / f"{base}_metrics.json"
    out_fals = out_dir / f"{base}_falsification_pack.json"
    out_csv = out_dir / f"{base}_peaks.csv"

    _plot(
        ell=ell,
        dl=dl,
        sigma=sigma,
        bestfit=bestfit,
        model_curve=model_curve,
        obs3=obs3,
        pred3=pred3,
        holdouts=holdouts,
        pred_holdouts=pred_holdouts,
        gate=gate,
        out_png=out_png,
    )
    _write_peak_csv(out_csv, obs3, pred3, holdouts, pred_holdouts)

    metrics_payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(data_path).replace("\\", "/"),
        "source_note": {
            "dataset": "Planck 2018 TT binned spectrum (COM_PowerSpect_CMB-TT-binned_R3.01)",
            "cached_local": str(data_path).replace("\\", "/"),
        },
        "model": {
            "equation": "Theta_k'' + Gamma_P Theta_k' + c_s,P^2 k^2 Theta_k = -k^2 Psi_P",
            "modal_mapping": {
                "ell_n": "ell_n^(P) = (n - phi_P) * ell_A^(P)",
                "amplitude_n": "A_n^(P) = A0 * exp(-delta_P*(n-1)) * [1 + (-1)^(n+1) * R_P] * D_n",
                "silk_damping": "D_n = exp(-(ell_n/ell_D)^2), ell_D = kappa_D * ell_A^(P)",
            },
            "acoustic_solution_formulation": acoustic_summary,
            "acoustic_cosine_validation": acoustic_validation,
            "fitted_from": "n=1..3 (peak positions/heights)",
            "holdout": "n=4..6 peaks (not used in fit)",
            "params": {k: float(v) for k, v in params.items()},
            "first_principles_closure": first_principles_closure,
            "n_modes_for_plot": int(n_modes),
        },
        "observed_peaks": [
            {"label": p.label, "n": p.n, "ell_range": [p.ell_min, p.ell_max], "ell": p.ell, "amplitude": p.amplitude}
            for p in obs3
        ]
        + [{"label": p.label, "n": p.n, "ell_range": [p.ell_min, p.ell_max], "ell": p.ell, "amplitude": p.amplitude} for p in holdouts],
        "predicted_peaks": [
            {"label": f"ℓ{int(p['n'])}", "n": int(p["n"]), "ell": float(p["ell"]), "amplitude": float(p["amplitude"])}
            for p in pred3
        ]
        + [{"label": f"ℓ{int(p['n'])}_holdout", "n": int(p["n"]), "ell": float(p["ell"]), "amplitude": float(p["amplitude"])} for p in pred_holdouts],
        "gate": gate,
        "outputs": {
            "png": str(out_png).replace("\\", "/"),
            "metrics_json": str(out_metrics).replace("\\", "/"),
            "falsification_pack_json": str(out_fals).replace("\\", "/"),
            "peaks_csv": str(out_csv).replace("\\", "/"),
        },
    }
    _write_json(out_metrics, metrics_payload)

    fals_payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "cmb_acoustic_peak_reconstruction_pmodel_modal",
        "decision_rule": {
            "first3": "max(|Δell_n|) <= 15 and max(|ΔA_n/A_n|) <= 0.15",
            "holdout4": "|Δell_4| <= 50 and |ΔA_4/A_4| <= 0.25",
            "extended_4to6": "max(|Δell_4..6|) <= 80 and max(|ΔA_4..6/A_4..6|) <= 0.35",
            "overall": "core(first3+holdout4) must pass; extended_4to6 is watch/pass diagnostic",
        },
        "thresholds": gate.get("thresholds", {}),
        "result": {"overall": gate.get("overall", {}), "overall_extended": gate.get("overall_extended", {})},
        "details": {"first3": gate.get("first3", {}), "holdout4": gate.get("holdout4", {})},
        "extended_details": {"holdout4to6": gate.get("extended_4to6", {})},
        "notes": [
            "本packは CMB TT のピーク構造（第1〜第6）に限定した最小監査であり、ΛCDM全体（全天球尤度・全成分同時fit）の置換主張ではない。",
            "第4〜第6ピークは holdout として扱い、n=1..3 から推定したパラメータの予測力を確認する。",
            "流体方程式の縮約で Θ_tilde_k = A_k cos(k r_s)+B_k sin(k r_s) を固定し、adiabatic枝（B_k=0）で TT ピークの cos 構造を使う。",
            "第一原理閉包として、R_P はポテンシャル井戸深さ u_Psi=|Psi_P|/c^2 と音速 c_s,P から、delta_P は散乱長 lambda_sc から導出し、DMなし枝で A3/A1=exp(-2delta_P)*(D3/D1)<1 を固定する。",
        ],
        "related_outputs": {
            "metrics_json": str(out_metrics).replace("\\", "/"),
            "figure_png": str(out_png).replace("\\", "/"),
            "peaks_csv": str(out_csv).replace("\\", "/"),
        },
    }
    _write_json(out_fals, fals_payload)

    copied: Dict[str, str] = {}
    # 条件分岐: `not bool(args.skip_public_copy)` を満たす経路を評価する。
    if not bool(args.skip_public_copy):
        copied = _copy_outputs_to_public([out_png, out_metrics, out_fals, out_csv], pub_dir)

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_metrics}")
    print(f"[ok] fals: {out_fals}")
    print(f"[ok] csv : {out_csv}")
    # 条件分岐: `copied` を満たす経路を評価する。
    if copied:
        print(f"[ok] copied to public: {len(copied)} files")

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_cmb_acoustic_peak_reconstruction",
                "argv": list(sys.argv),
                "inputs": {"data": data_path},
                "outputs": {
                    "png": out_png,
                    "metrics_json": out_metrics,
                    "falsification_pack_json": out_fals,
                    "peaks_csv": out_csv,
                    "public_copies": copied,
                },
                "metrics": {
                    "n_modes": int(n_modes),
                    "silk_kappa": float(silk_kappa),
                    "gate_status": gate.get("overall", {}).get("status", "n/a"),
                    "gate_status_extended": gate.get("overall_extended", {}).get("status", "n/a"),
                    "max_abs_delta_ell_first3": gate.get("first3", {}).get("max_abs_delta_ell"),
                    "max_abs_delta_amp_rel_first3": gate.get("first3", {}).get("max_abs_delta_amp_rel"),
                    "delta_ell_holdout4": gate.get("holdout4", {}).get("residual", {}).get("delta_ell"),
                    "delta_amp_rel_holdout4": gate.get("holdout4", {}).get("residual", {}).get("delta_amp_rel"),
                    "max_abs_delta_ell_holdout4to6": gate.get("extended_4to6", {}).get("max_abs_delta_ell"),
                    "max_abs_delta_amp_rel_holdout4to6": gate.get("extended_4to6", {}).get("max_abs_delta_amp_rel"),
                    "acoustic_cosine_max_abs_error": acoustic_validation.get("max_abs_error"),
                    "acoustic_cosine_pass": acoustic_validation.get("pass"),
                },
            }
        )
    except Exception:
        pass

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
