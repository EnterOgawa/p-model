#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_bao_scaled_distance_fit_sensitivity.py

Step 14.2.27（BAO(s_R) の支配要因の整理）:
距離指標の再導出候補探索（best_independent）で limiting が BAO(s_R) に移ったため、
BOSS DR12（BAO+FS consensus）の (D_M,H) 出力から推定する標準定規進化 s_R が
「共分散の扱い（full/diag/ブロック）」「点の取り方（D_Mのみ/Hのみ）」でどれだけ動くかを定量化する。

目的：
  - BAO(s_R) の推定が、相関（特に cross-z covariance）を無視した近似で過小評価されていないかを確認する。
  - その上で、距離指標の再導出（DDR+独立一次ソース+独立プローブ+BAO）に対し、
    BAO 側の不確かさをどの程度「膨張」させれば 1σ に収まるかの目安を見積もる。

入力（固定）:
  - data/cosmology/alcock_paczynski_constraints.json（BOSS DR12 の D_M,H と ρ(D_M,H)）
  - data/cosmology/boss_dr12_baofs_consensus_reduced_covariance_cij.json（Table 8: reduced covariance c_ij）
  - data/cosmology/distance_duality_constraints.json（DDR ε0）
  - data/cosmology/cosmic_opacity_constraints.json（α）
  - data/cosmology/sn_standard_candle_evolution_constraints.json（s_L）
  - data/cosmology/sn_time_dilation_constraints.json（p_t）
  - data/cosmology/cmb_temperature_scaling_constraints.json（β_T → p_e）

出力（固定名）:
  - output/private/cosmology/cosmology_bao_scaled_distance_fit_sensitivity.png
  - output/private/cosmology/cosmology_bao_scaled_distance_fit_sensitivity_metrics.json
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

_C_KM_S = 299792.458


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


def _classify_sigma(abs_z: float) -> Tuple[str, str]:
    if not math.isfinite(abs_z):
        return ("na", "#999999")
    if abs_z < 3.0:
        return ("ok", "#2ca02c")
    if abs_z < 5.0:
        return ("mixed", "#ffbf00")
    return ("ng", "#d62728")


@dataclass(frozen=True)
class BAOPoint:
    id: str
    short_label: str
    z_eff: float
    dm: float
    dm_sigma: float
    h: float
    h_sigma: float
    corr_dm_h: float


@dataclass(frozen=True)
class DDRConstraint:
    id: str
    short_label: str
    epsilon0: float
    epsilon0_sigma: float
    uses_bao: bool


@dataclass(frozen=True)
class GaussianConstraint:
    id: str
    short_label: str
    mean: float
    sigma: float
    uses_bao: Optional[bool] = None
    uses_cmb: Optional[bool] = None
    assumes_cddr: Optional[bool] = None

    def is_independent(self) -> bool:
        if self.uses_bao is True:
            return False
        if self.uses_cmb is True:
            return False
        if self.assumes_cddr is True:
            return False
        return True


def _load_bao_points(path: Path) -> List[BAOPoint]:
    rows = _read_json(path).get("constraints") or []
    out: List[BAOPoint] = []
    for r in rows:
        out.append(
            BAOPoint(
                id=str(r.get("id") or ""),
                short_label=str(r.get("short_label") or r.get("id") or ""),
                z_eff=float(r["z_eff"]),
                dm=float(r["DM_scaled_mpc"]),
                dm_sigma=float(r["DM_scaled_sigma_mpc"]),
                h=float(r["H_scaled_km_s_mpc"]),
                h_sigma=float(r["H_scaled_sigma_km_s_mpc"]),
                corr_dm_h=float(r.get("corr_DM_H", 0.0)),
            )
        )
    out = sorted(out, key=lambda x: float(x.z_eff))
    if len(out) < 2:
        raise ValueError("need >=2 BAO points for s_R fit")
    return out


def _load_boss_corr_for_dm_h(
    boss_cov_path: Path,
    *,
    bao_points: Sequence[BAOPoint],
    z_tol: float = 5e-4,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Return correlation matrix (6x6) for [DM(z0), H(z0), DM(z1), H(z1), DM(z2), H(z2)].
    """
    src = _read_json(boss_cov_path)
    params = list(src.get("parameters") or [])
    cij_1e4 = np.array(src.get("cij_1e4"), dtype=float)
    if cij_1e4.ndim != 2 or cij_1e4.shape[0] != cij_1e4.shape[1]:
        raise ValueError(f"invalid cij_1e4 shape: {cij_1e4.shape}")
    if len(params) != int(cij_1e4.shape[0]):
        raise ValueError("cij_1e4 size mismatch with parameters")

    def match_index(kind: str, z: float) -> Tuple[int, Dict[str, Any]]:
        best_i = None
        best_d = float("inf")
        best_p = None
        for i, p in enumerate(params):
            if str(p.get("kind") or "") != kind:
                continue
            zz = p.get("z_eff")
            if zz is None:
                continue
            d = abs(float(zz) - float(z))
            if d < best_d:
                best_d = d
                best_i = int(i)
                best_p = dict(p)
        if best_i is None or best_p is None or best_d > float(z_tol):
            raise ValueError(f"cannot match kind={kind} z={z} (best_d={best_d})")
        return best_i, best_p

    indices: List[int] = []
    matched: List[Dict[str, Any]] = []
    for bp in bao_points:
        i_dm, p_dm = match_index("DM_scaled_mpc", float(bp.z_eff))
        i_h, p_h = match_index("H_scaled_km_s_mpc", float(bp.z_eff))
        indices.extend([i_dm, i_h])
        matched.extend([p_dm, p_h])

    corr = (cij_1e4[np.ix_(indices, indices)] / 10000.0).astype(float)
    corr = 0.5 * (corr + corr.T)
    return corr, matched


def _bao_pred_dm_h(*, z: float, s_R: float, B: float) -> Tuple[float, float]:
    op = 1.0 + float(z)
    if not (op > 0.0):
        raise ValueError("z must satisfy 1+z>0")
    B = float(B)
    if not (B > 0.0):
        raise ValueError("B must be > 0")

    dm_scaled = (_C_KM_S / B) * math.log(op) * (op ** (-float(s_R)))
    h_scaled = B * (op ** (1.0 + float(s_R)))
    return float(dm_scaled), float(h_scaled)


def _bao_pred_vec(*, z_points: Sequence[float], s_R: float, B: float, use_dm: bool, use_h: bool) -> np.ndarray:
    if not (use_dm or use_h):
        raise ValueError("use_dm/use_h must include at least one observable")
    vec: List[float] = []
    for z in z_points:
        dm_pred, h_pred = _bao_pred_dm_h(z=float(z), s_R=float(s_R), B=float(B))
        if use_dm:
            vec.append(float(dm_pred))
        if use_h:
            vec.append(float(h_pred))
    return np.array(vec, dtype=float)


def _chi2_from_cov_inv(residual: np.ndarray, cov_inv: np.ndarray) -> float:
    return float(residual.T @ cov_inv @ residual)


def _build_covariance(
    *,
    bao_points: Sequence[BAOPoint],
    mode: str,
    boss_corr_dm_h: Optional[np.ndarray],
    use_dm: bool,
    use_h: bool,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    z_points = [float(r.z_eff) for r in bao_points]
    labels: List[str] = []
    sigmas: List[float] = []
    for z, r in zip(z_points, bao_points, strict=True):
        if use_dm:
            labels.append(f"DM({z:.2f})")
            sigmas.append(float(r.dm_sigma))
        if use_h:
            labels.append(f"H({z:.2f})")
            sigmas.append(float(r.h_sigma))

    sig = np.array(sigmas, dtype=float)
    if mode == "diag":
        cov = np.diag(np.maximum(1e-300, sig) ** 2)
        return cov, np.linalg.inv(cov), labels

    if mode == "block":
        # per-z 2x2 (or 1x1) blocks
        n = int(len(sig))
        cov = np.zeros((n, n), dtype=float)
        cursor = 0
        for r in bao_points:
            if use_dm and use_h:
                a = float(r.dm_sigma) ** 2
                d = float(r.h_sigma) ** 2
                cov12 = float(r.corr_dm_h) * float(r.dm_sigma) * float(r.h_sigma)
                cov[cursor + 0, cursor + 0] = a
                cov[cursor + 1, cursor + 1] = d
                cov[cursor + 0, cursor + 1] = cov12
                cov[cursor + 1, cursor + 0] = cov12
                cursor += 2
            elif use_dm:
                cov[cursor, cursor] = float(r.dm_sigma) ** 2
                cursor += 1
            else:
                cov[cursor, cursor] = float(r.h_sigma) ** 2
                cursor += 1
        cov = 0.5 * (cov + cov.T)
        return cov, np.linalg.inv(cov), labels

    if mode == "full":
        if boss_corr_dm_h is None:
            raise ValueError("boss_corr_dm_h is required for full mode")
        # boss_corr_dm_h is for (DM,H) pairs; reduce if needed.
        corr = boss_corr_dm_h
        if use_dm and use_h:
            corr_use = corr
        else:
            # pick DM-only or H-only indices from the (DM,H) ordering.
            idx = list(range(0, corr.shape[0], 2)) if use_dm else list(range(1, corr.shape[0], 2))
            corr_use = corr[np.ix_(idx, idx)]
        cov = corr_use * np.outer(sig, sig)
        cov = 0.5 * (cov + cov.T)
        return cov, np.linalg.inv(cov), labels

    raise ValueError(f"unknown covariance mode: {mode}")


def _fit_bao_s_r(
    *,
    bao_points: Sequence[BAOPoint],
    cov_inv: np.ndarray,
    use_dm: bool,
    use_h: bool,
    s_R_min: float = -1.0,
    s_R_max: float = 2.0,
    B_min: float = 5.0,
    B_max: float = 200.0,
) -> Dict[str, Any]:
    z_points = [float(r.z_eff) for r in bao_points]

    y_obs_list: List[float] = []
    for r in bao_points:
        if use_dm:
            y_obs_list.append(float(r.dm))
        if use_h:
            y_obs_list.append(float(r.h))
    y_obs = np.array(y_obs_list, dtype=float)

    def chi2(s_R: float, B: float) -> float:
        y_pred = _bao_pred_vec(z_points=z_points, s_R=float(s_R), B=float(B), use_dm=use_dm, use_h=use_h)
        return _chi2_from_cov_inv(y_obs - y_pred, cov_inv)

    def profile_B(s_R: float, *, n_grid: int = 2501) -> Tuple[float, float]:
        Bs = np.linspace(float(B_min), float(B_max), int(n_grid), dtype=float)
        chi_vals = np.array([chi2(s_R=float(s_R), B=float(b)) for b in Bs], dtype=float)
        i = int(np.nanargmin(chi_vals))
        b0 = float(Bs[i])

        lo = max(float(B_min), b0 - 2.0)
        hi = min(float(B_max), b0 + 2.0)
        Bs2 = np.linspace(lo, hi, 2001, dtype=float)
        chi2_vals = np.array([chi2(s_R=float(s_R), B=float(b)) for b in Bs2], dtype=float)
        j = int(np.nanargmin(chi2_vals))
        return float(Bs2[j]), float(chi2_vals[j])

    # coarse scan
    s_grid0 = np.linspace(float(s_R_min), float(s_R_max), 301, dtype=float)
    prof0: List[Dict[str, float]] = []
    for s in s_grid0:
        b_best, chi_best = profile_B(float(s), n_grid=2001)
        prof0.append({"s_R": float(s), "B_best": float(b_best), "chi2": float(chi_best)})
    best0 = min(prof0, key=lambda x: (x["chi2"] if math.isfinite(x["chi2"]) else float("inf")))

    # refine scan
    s0 = float(best0["s_R"])
    lo = max(float(s_R_min), s0 - 0.2)
    hi = min(float(s_R_max), s0 + 0.2)
    s_grid = np.linspace(lo, hi, 801, dtype=float)
    prof: List[Dict[str, float]] = []
    for s in s_grid:
        b_best, chi_best = profile_B(float(s))
        prof.append({"s_R": float(s), "B_best": float(b_best), "chi2": float(chi_best)})

    best = min(prof, key=lambda x: (x["chi2"] if math.isfinite(x["chi2"]) else float("inf")))
    chi2_min = float(best["chi2"])
    s_best = float(best["s_R"])
    B_best = float(best["B_best"])

    # 1σ interval in s_R (profile Δχ2=1 for one parameter).
    target = chi2_min + 1.0
    s_prof = np.array([float(r["s_R"]) for r in prof], dtype=float)
    chi_prof = np.array([float(r["chi2"]) for r in prof], dtype=float)
    i_best = int(np.nanargmin(chi_prof))
    ok = np.isfinite(chi_prof) & (chi_prof <= target)

    i_left = i_best
    while i_left > 0 and bool(ok[i_left]):
        i_left -= 1
    left = float(s_prof[i_left + 1]) if not bool(ok[i_left]) else float(s_prof[0])

    i_right = i_best
    while i_right < len(s_prof) - 1 and bool(ok[i_right]):
        i_right += 1
    right = float(s_prof[i_right - 1]) if not bool(ok[i_right]) else float(s_prof[-1])

    if left == float(s_prof[0]) or right == float(s_prof[-1]):
        s_sig = float("nan")
        s_sig_asym = {"minus": float("nan"), "plus": float("nan")}
    else:
        s_sig_asym = {"minus": float(s_best - left), "plus": float(right - s_best)}
        s_sig = float(max(s_sig_asym["minus"], s_sig_asym["plus"]))

    n_obs = int(y_obs.size)
    dof = int(n_obs - 2)  # params: (s_R, B)
    return {
        "profile": prof,
        "best_fit": {
            "s_R": s_best,
            "s_R_sigma_1d": s_sig,
            "s_R_sigma_1d_asym": s_sig_asym,
            "B_best_km_s_mpc": B_best,
            "chi2": chi2_min,
            "dof": int(dof),
            "obs_dim": int(n_obs),
            "uses": {"DM": bool(use_dm), "H": bool(use_h)},
        },
    }


def _as_gaussian_list(
    rows: Sequence[Dict[str, Any]],
    *,
    mean_key: str,
    sigma_key: str,
    uses_bao_key: str = "uses_bao",
    uses_cmb_key: str = "uses_cmb",
    assumes_cddr_key: str = "assumes_cddr",
) -> List[GaussianConstraint]:
    out: List[GaussianConstraint] = []
    for r in rows:
        try:
            mean = float(r[mean_key])
            sig = float(r[sigma_key])
        except Exception:
            continue
        if not (sig > 0.0 and math.isfinite(sig)):
            continue
        out.append(
            GaussianConstraint(
                id=str(r.get("id") or ""),
                short_label=str(r.get("short_label") or r.get("id") or ""),
                mean=mean,
                sigma=sig,
                uses_bao=(bool(r.get(uses_bao_key)) if uses_bao_key in r and r[uses_bao_key] is not None else None),
                uses_cmb=(bool(r.get(uses_cmb_key)) if uses_cmb_key in r and r[uses_cmb_key] is not None else None),
                assumes_cddr=(
                    bool(r.get(assumes_cddr_key))
                    if assumes_cddr_key in r and r[assumes_cddr_key] is not None
                    else None
                ),
            )
        )
    return out


def _load_ddr_constraints(path: Path) -> List[DDRConstraint]:
    rows = _read_json(path).get("constraints") or []
    out: List[DDRConstraint] = []
    for r in rows:
        out.append(
            DDRConstraint(
                id=str(r.get("id") or ""),
                short_label=str(r.get("short_label") or r.get("id") or ""),
                epsilon0=float(r["epsilon0"]),
                epsilon0_sigma=float(r["epsilon0_sigma"]),
                uses_bao=bool(r.get("uses_bao", False)),
            )
        )
    return out


def _load_fixed_pt_pe(
    *,
    pt_path: Path,
    pe_path: Path,
) -> Tuple[GaussianConstraint, GaussianConstraint]:
    pt_rows = _read_json(pt_path).get("constraints") or []
    pt_all = _as_gaussian_list(pt_rows, mean_key="p_t", sigma_key="p_t_sigma")
    if not pt_all:
        raise ValueError("no p_t constraint found")
    p_t = pt_all[0]

    # Convert beta_T -> p_e=1-beta_T.
    pe_rows = _read_json(pe_path).get("constraints") or []
    pe = None
    for r in pe_rows:
        try:
            beta = float(r["beta_T"])
            sig = float(r["beta_T_sigma"])
        except Exception:
            continue
        if not (sig > 0.0 and math.isfinite(sig)):
            continue
        pe = GaussianConstraint(
            id=str(r.get("id") or ""),
            short_label=str(r.get("short_label") or r.get("id") or ""),
            mean=float(1.0 - beta),
            sigma=float(sig),
        )
        break
    if pe is None:
        raise ValueError("no CMB temperature scaling constraint found")
    return p_t, pe


def _wls_fit_candidate(
    *,
    ddr: DDRConstraint,
    sR_bao: float,
    sR_bao_sigma: float,
    opacity: GaussianConstraint,
    candle: GaussianConstraint,
    p_t: GaussianConstraint,
    p_e: GaussianConstraint,
) -> Dict[str, Any]:
    obs_names = [
        "DDR ε0",
        "BAO s_R",
        "Opacity α",
        "Candle s_L",
        "SN time dilation p_t",
        "CMB energy p_e",
    ]

    y = np.array(
        [
            float(ddr.epsilon0) + 2.0,
            float(sR_bao),
            float(opacity.mean),
            float(candle.mean),
            float(p_t.mean),
            float(p_e.mean),
        ],
        dtype=float,
    )
    sig = np.array(
        [
            float(ddr.epsilon0_sigma),
            float(sR_bao_sigma),
            float(opacity.sigma),
            float(candle.sigma),
            float(p_t.sigma),
            float(p_e.sigma),
        ],
        dtype=float,
    )
    # θ=[s_R, α, s_L, p_t, p_e]
    A = np.array(
        [
            [1.0, 1.0, -0.5, 0.5, 0.5],  # DDR (ε0+2)
            [1.0, 0.0, 0.0, 0.0, 0.0],  # BAO s_R
            [0.0, 1.0, 0.0, 0.0, 0.0],  # α
            [0.0, 0.0, 1.0, 0.0, 0.0],  # s_L
            [0.0, 0.0, 0.0, 1.0, 0.0],  # p_t
            [0.0, 0.0, 0.0, 0.0, 1.0],  # p_e
        ],
        dtype=float,
    )

    W = np.diag(1.0 / np.maximum(1e-300, sig) ** 2)
    theta = np.linalg.solve(A.T @ W @ A, A.T @ W @ y)
    pred = A @ theta
    z = (pred - y) / np.maximum(1e-300, sig)
    chi2 = float(np.sum(z**2))
    dof = int(len(y) - len(theta))
    max_abs_z = float(np.max(np.abs(z)))
    limiting_idx = int(np.argmax(np.abs(z)))
    z_d = {name: float(z[i]) for i, name in enumerate(obs_names)}
    return {
        "theta": {
            "s_R": float(theta[0]),
            "alpha_opacity": float(theta[1]),
            "s_L": float(theta[2]),
            "p_t": float(theta[3]),
            "p_e": float(theta[4]),
        },
        "z_scores": z_d,
        "chi2": chi2,
        "dof": dof,
        "chi2_dof": (chi2 / dof) if dof > 0 else None,
        "max_abs_z": max_abs_z,
        "limiting_observation": obs_names[limiting_idx],
    }


def _choose_best(candidates: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None

    def key(c: Dict[str, Any]) -> Tuple[float, float]:
        return (float(c["fit"]["max_abs_z"]), float(c["fit"]["chi2"]))

    return min(candidates, key=key)


def _evaluate_candidate_for_ddr(
    *,
    ddr: DDRConstraint,
    sR_bao: float,
    sR_bao_sigma: float,
    opacity_all: Sequence[GaussianConstraint],
    candle_all: Sequence[GaussianConstraint],
    p_t: GaussianConstraint,
    p_e: GaussianConstraint,
    independent_only: bool,
) -> Optional[Dict[str, Any]]:
    opacity_use = [c for c in opacity_all if (c.is_independent() if independent_only else True)]
    candle_use = [c for c in candle_all if (c.is_independent() if independent_only else True)]
    if not (opacity_use and candle_use):
        return None

    candidates: List[Dict[str, Any]] = []
    for op in opacity_use:
        for cd in candle_use:
            fit = _wls_fit_candidate(
                ddr=ddr,
                sR_bao=sR_bao,
                sR_bao_sigma=sR_bao_sigma,
                opacity=op,
                candle=cd,
                p_t=p_t,
                p_e=p_e,
            )
            candidates.append(
                {
                    "opacity": {"id": op.id, "short_label": op.short_label},
                    "candle": {"id": cd.id, "short_label": cd.short_label},
                    "fit": fit,
                }
            )
    return _choose_best(candidates)


def _plot(
    *,
    out_png: Path,
    bao_fits: Sequence[Dict[str, Any]],
    candidate_summary: Dict[str, Any],
) -> None:
    _set_japanese_font()
    import matplotlib.pyplot as plt

    mode_labels = [str(r.get("label") or r.get("mode") or "") for r in bao_fits]
    s_vals = np.array([float(r["best_fit"]["s_R"]) for r in bao_fits], dtype=float)
    s_sig = np.array([float(r["best_fit"]["s_R_sigma_1d"]) for r in bao_fits], dtype=float)

    # Candidate summary focuses on BAO-including DDR (Martinelli2021).
    cand = candidate_summary.get("best_independent") or {}
    cand_by_mode = cand.get("by_mode") or {}
    maxz = np.array([float((cand_by_mode.get(r["mode"]) or {}).get("max_abs_z", float("nan"))) for r in bao_fits])
    colors = [_classify_sigma(abs(v))[1] if math.isfinite(float(v)) else "#cccccc" for v in maxz]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2))

    # A) s_R fit under covariance modes
    ax = axes[0]
    y = np.arange(len(mode_labels), dtype=float)
    ax.errorbar(s_vals, y, xerr=np.where(np.isfinite(s_sig) & (s_sig > 0), s_sig, 0.0), fmt="o", capsize=5)
    ax.axvline(0.0, color="#999999", linewidth=1.0, linestyle="--", alpha=0.8, label="s_R=0（定規進化なし）")
    ax.set_yticks(y)
    ax.set_yticklabels(mode_labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("s_R（BOSS DR12 の (D_M,H) からfit）", fontsize=11)
    ax.set_title("BAO(s_R) の共分散感度", fontsize=13)
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="lower right")
    for i, (s, sig) in enumerate(zip(s_vals, s_sig, strict=True)):
        ax.text(float(s) + 0.02, float(i), f"{_fmt_float(s, digits=3)}±{_fmt_float(sig, digits=3)}", va="center")

    # B) Effect on candidate search (best_independent max|z|)
    ax = axes[1]
    ax.barh(y, maxz, color=colors, alpha=0.9)
    for xline, txt in [(1.0, "1σ"), (3.0, "3σ"), (5.0, "5σ")]:
        ax.axvline(xline, color="#333333", linewidth=1.0, alpha=0.2)
        ax.text(xline + 0.05, -0.7, txt, fontsize=9, color="#333333", alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(mode_labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("best_independent の max|z|（DDR+BAO+α+s_L+p_t+p_e）", fontsize=11)
    ax.set_title("再導出候補探索への影響（BAO含むDDR）", fontsize=13)
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)
    for i, v in enumerate(maxz):
        if math.isfinite(float(v)):
            ax.text(float(v) + 0.08, float(i), f"{_fmt_float(float(v), digits=3)}σ", va="center", fontsize=9)

    fig.suptitle("宇宙論：BAO(s_R) 推定の感度と、再導出候補探索（best_independent）への影響", fontsize=14)
    fig.text(
        0.5,
        0.01,
        "左：BOSS DR12 (D_M,H) からの s_R 推定（共分散の扱いでどれだけ動くか）。右：BAO含むDDR（SNIa+BAO）の best_independent の max|z|。",
        ha="center",
        fontsize=10,
    )
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.93))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ddr-id", default="martinelli2021_snIa_bao", help="Target DDR id for candidate impact panel.")
    args = ap.parse_args(argv)

    data_dir = _ROOT / "data" / "cosmology"
    out_dir = _ROOT / "output" / "private" / "cosmology"

    ap_path = data_dir / "alcock_paczynski_constraints.json"
    boss_cov_path = data_dir / "boss_dr12_baofs_consensus_reduced_covariance_cij.json"
    ddr_path = data_dir / "distance_duality_constraints.json"
    opacity_path = data_dir / "cosmic_opacity_constraints.json"
    candle_path = data_dir / "sn_standard_candle_evolution_constraints.json"
    pt_path = data_dir / "sn_time_dilation_constraints.json"
    pe_path = data_dir / "cmb_temperature_scaling_constraints.json"

    for p in (ap_path, boss_cov_path, ddr_path, opacity_path, candle_path, pt_path, pe_path):
        if not p.exists():
            raise FileNotFoundError(f"missing input: {p}")

    bao_points = _load_bao_points(ap_path)
    boss_corr_dm_h, boss_matched = _load_boss_corr_for_dm_h(boss_cov_path, bao_points=bao_points)

    # Fit modes
    fit_specs = [
        {"mode": "block", "label": "block（ρ(DM,H)のみ; cross-z無視）", "use_dm": True, "use_h": True},
        {"mode": "diag", "label": "diag（相関を全て無視）", "use_dm": True, "use_h": True},
        {"mode": "full", "label": "full（Table 8 c_ij: cross-z含む）", "use_dm": True, "use_h": True},
        {"mode": "full_dm_only", "label": "full（D_Mのみ）", "use_dm": True, "use_h": False},
        {"mode": "full_h_only", "label": "full（Hのみ）", "use_dm": False, "use_h": True},
    ]

    bao_fits: List[Dict[str, Any]] = []
    for spec in fit_specs:
        mode = str(spec["mode"])
        use_dm = bool(spec["use_dm"])
        use_h = bool(spec["use_h"])
        base_mode = mode.replace("_dm_only", "").replace("_h_only", "")
        cov, cov_inv, labels = _build_covariance(
            bao_points=bao_points,
            mode=("full" if base_mode == "full" else base_mode),
            boss_corr_dm_h=boss_corr_dm_h,
            use_dm=use_dm,
            use_h=use_h,
        )
        fit = _fit_bao_s_r(bao_points=bao_points, cov_inv=cov_inv, use_dm=use_dm, use_h=use_h)
        bao_fits.append(
            {
                "mode": mode,
                "label": str(spec.get("label") or mode),
                "covariance": {
                    "base_mode": base_mode,
                    "dim": int(cov.shape[0]),
                    "labels": labels,
                    "cond": float(np.linalg.cond(cov)),
                },
                "best_fit": fit.get("best_fit") or {},
            }
        )

    # Candidate impact: evaluate for a single DDR (default: Martinelli2021).
    ddr_all = _load_ddr_constraints(ddr_path)
    ddr = next((d for d in ddr_all if d.id == str(args.ddr_id)), None)
    if ddr is None:
        raise ValueError(f"ddr-id not found: {args.ddr_id}")

    opacity_all = _as_gaussian_list(_read_json(opacity_path).get("constraints") or [], mean_key="alpha_opacity", sigma_key="alpha_opacity_sigma")
    candle_all = _as_gaussian_list(_read_json(candle_path).get("constraints") or [], mean_key="s_L", sigma_key="s_L_sigma")
    p_t, p_e = _load_fixed_pt_pe(pt_path=pt_path, pe_path=pe_path)

    candidate_by_mode: Dict[str, Any] = {}
    for bf in bao_fits:
        best_fit = bf.get("best_fit") or {}
        sR_bao = float(best_fit.get("s_R", float("nan")))
        sR_sig = float(best_fit.get("s_R_sigma_1d", float("nan")))
        if not (math.isfinite(sR_bao) and math.isfinite(sR_sig) and sR_sig > 0):
            candidate_by_mode[str(bf["mode"])] = {"ok": False, "reason": "invalid_bao_fit"}
            continue

        best_ind = _evaluate_candidate_for_ddr(
            ddr=ddr,
            sR_bao=sR_bao,
            sR_bao_sigma=sR_sig,
            opacity_all=opacity_all,
            candle_all=candle_all,
            p_t=p_t,
            p_e=p_e,
            independent_only=True,
        )
        if best_ind is None:
            candidate_by_mode[str(bf["mode"])] = {"ok": False, "reason": "no_candidate"}
            continue

        z_bao = float((best_ind["fit"]["z_scores"] or {}).get("BAO s_R", float("nan")))
        sigma_scale_to_1sigma = abs(z_bao) if math.isfinite(z_bao) else float("nan")
        candidate_by_mode[str(bf["mode"])] = {
            "ok": True,
            "max_abs_z": float(best_ind["fit"]["max_abs_z"]),
            "limiting_observation": str(best_ind["fit"]["limiting_observation"]),
            "opacity": best_ind.get("opacity") or {},
            "candle": best_ind.get("candle") or {},
            "z_scores": best_ind["fit"]["z_scores"],
            "sigma_scale_to_make_bao_1sigma": sigma_scale_to_1sigma,
        }

    out_png = out_dir / "cosmology_bao_scaled_distance_fit_sensitivity.png"
    out_json = out_dir / "cosmology_bao_scaled_distance_fit_sensitivity_metrics.json"

    candidate_summary = {
        "ddr_id": ddr.id,
        "ddr_short_label": ddr.short_label,
        "best_independent": {"by_mode": candidate_by_mode},
        "fixed_constraints": {
            "p_t": {"id": p_t.id, "mean": p_t.mean, "sigma": p_t.sigma},
            "p_e": {"id": p_e.id, "mean": p_e.mean, "sigma": p_e.sigma},
        },
    }
    _plot(out_png=out_png, bao_fits=bao_fits, candidate_summary=candidate_summary)

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "alcock_paczynski_constraints": str(ap_path.relative_to(_ROOT)).replace("\\", "/"),
            "boss_reduced_cov_cij": str(boss_cov_path.relative_to(_ROOT)).replace("\\", "/"),
            "distance_duality_constraints": str(ddr_path.relative_to(_ROOT)).replace("\\", "/"),
            "cosmic_opacity_constraints": str(opacity_path.relative_to(_ROOT)).replace("\\", "/"),
            "sn_standard_candle_evolution_constraints": str(candle_path.relative_to(_ROOT)).replace("\\", "/"),
            "sn_time_dilation_constraints": str(pt_path.relative_to(_ROOT)).replace("\\", "/"),
            "cmb_temperature_scaling_constraints": str(pe_path.relative_to(_ROOT)).replace("\\", "/"),
        },
        "bao_fit_sensitivity": bao_fits,
        "candidate_impact": candidate_summary,
        "boss_matching_dm_h": boss_matched,
        "outputs": {
            "png": str(out_png.relative_to(_ROOT)).replace("\\", "/"),
            "metrics_json": str(out_json.relative_to(_ROOT)).replace("\\", "/"),
        },
    }
    _write_json(out_json, payload)

    try:
        worklog.append_event(
            event_type="cosmology_bao_scaled_distance_fit_sensitivity",
            argv=[str(a) for a in (argv or sys.argv)],
            inputs={
                "alcock_paczynski_constraints": str(ap_path.relative_to(_ROOT)).replace("\\", "/"),
                "boss_reduced_cov_cij": str(boss_cov_path.relative_to(_ROOT)).replace("\\", "/"),
            },
            outputs={
                "png": str(out_png.relative_to(_ROOT)).replace("\\", "/"),
                "metrics_json": str(out_json.relative_to(_ROOT)).replace("\\", "/"),
            },
        )
    except Exception:
        pass

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

