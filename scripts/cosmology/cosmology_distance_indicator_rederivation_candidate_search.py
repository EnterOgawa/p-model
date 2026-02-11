#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_distance_indicator_rederivation_candidate_search.py

Step 14.2.22（距離指標の再導出候補の探索）:
静的背景P（最小: ε0=-1）で「距離二重性（DDR）」を回復するには、
距離指標（標準光源/標準定規/不透明度）と独立プローブ（SN time dilation, CMB T(z)）の
どの組合せが同時に整合し得るかを、一次ソース拘束（Gaussian近似）で探索する。

Step 14.2.28（BAO(s_R) を soft constraint として扱う感度）:
BAO の `s_R` 拘束（BOSS DR12 fit）について、σ を f 倍して重みを緩めた場合に
best_any / best_independent の max|z| と limiting（支配する拘束）がどう動くかをスキャンし、
固定名で図と metrics を出力する（“BAOの支配”がどの程度強いかを可視化）。

方針:
  - モデル（再接続の最小パラメータ化）:
      ε0_model = (p_e + p_t - s_L)/2 - 2 + s_R + α
    p_e: 光子エネルギー赤方偏移の指数（CMB T(z) から p_e≈p_T）
    p_t: 時間伸長の指数（SN spectra aging rate から）
    s_L: 標準光源（SNe Ia）有効進化（L∝(1+z)^(s_L)）
    s_R: 標準定規（BAO）有効進化（r_d∝(1+z)^(s_R)）
    α  : 灰色不透明度（τ(z)=2α ln(1+z)）
  - 目的: DDR(ε0) と BAO(s_R; BOSS DR12 fit) と独立プローブ(p_t, p_e) を同時に満たす
    “再導出候補（必要な置換/補正の組合せ）”が存在するかを、z-score で評価する。
  - 探索: α と s_L の一次ソース拘束（複数候補）を入れ替え、WLSで最小の最大|z|を探す。
    併せて、BAO/CMB/CDDR 依存を避けた “independent-only” の候補でも同様に評価する。

入力（固定）:
  - data/cosmology/distance_duality_constraints.json
  - data/cosmology/cosmic_opacity_constraints.json
  - （任意）data/cosmology/gw_standard_siren_opacity_constraints.json（標準サイレン由来の不透明度拘束; 参考/将来予測など）
  - data/cosmology/sn_standard_candle_evolution_constraints.json
  - data/cosmology/sn_time_dilation_constraints.json
  - data/cosmology/cmb_temperature_scaling_constraints.json
  - output/cosmology/cosmology_bao_scaled_distance_fit_metrics.json（s_R の一次データfit）

出力（固定名）:
  - output/cosmology/cosmology_distance_indicator_rederivation_candidate_search.png
  - output/cosmology/cosmology_distance_indicator_rederivation_candidate_search_metrics.json
  - output/cosmology/cosmology_distance_indicator_rederivation_candidate_search_bao_sigma_scan.png
  - output/cosmology/cosmology_distance_indicator_rederivation_candidate_search_bao_sigma_scan_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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


def _maybe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def _classify_sigma(abs_z: float) -> Tuple[str, str]:
    if not math.isfinite(abs_z):
        return ("na", "#999999")
    if abs_z < 3.0:
        return ("ok", "#2ca02c")
    if abs_z < 5.0:
        return ("mixed", "#ffbf00")
    return ("ng", "#d62728")


@dataclass(frozen=True)
class DDRConstraint:
    id: str
    short_label: str
    title: str
    epsilon0: float
    epsilon0_sigma: float
    uses_bao: bool
    source: Dict[str, Any]
    epsilon0_sigma_raw: float = 0.0
    sigma_sys_category: Optional[float] = None
    sigma_policy: str = "raw"
    category: Optional[str] = None

    @staticmethod
    def from_json(j: Dict[str, Any]) -> "DDRConstraint":
        sigma = float(j["epsilon0_sigma"])
        return DDRConstraint(
            id=str(j["id"]),
            short_label=str(j.get("short_label") or j["id"]),
            title=str(j.get("title") or j.get("short_label") or j["id"]),
            epsilon0=float(j["epsilon0"]),
            epsilon0_sigma=sigma,
            uses_bao=bool(j.get("uses_bao", False)),
            source=dict(j.get("source") or {}),
            epsilon0_sigma_raw=sigma,
        )


@dataclass(frozen=True)
class GaussianConstraint:
    id: str
    short_label: str
    title: str
    mean: float
    sigma: float
    uses_bao: Optional[bool]
    uses_cmb: Optional[bool]
    assumes_cddr: Optional[bool]
    source: Dict[str, Any]

    def is_independent(self) -> bool:
        # Conservative filter: avoid BAO/CMB compression and avoid assuming CDDR.
        if self.uses_bao is True:
            return False
        if self.uses_cmb is True:
            return False
        if self.assumes_cddr is True:
            return False
        return True


def _pick_tightest(rows: Sequence[GaussianConstraint]) -> Optional[GaussianConstraint]:
    best: Optional[GaussianConstraint] = None
    best_sig = float("inf")
    for r in rows:
        sig = float(r.sigma)
        if not (sig > 0.0 and math.isfinite(sig)):
            continue
        if sig < best_sig:
            best_sig = sig
            best = r
    return best


def _load_ddr_systematics_envelope(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load id -> {sigma_total, sigma_sys_category, category} from
    `cosmology_distance_duality_systematics_envelope_metrics.json`.

    This file is generated from the fixed DDR metrics and captures category-level
    model spread as a systematic-width proxy (σ_cat).
    """
    if not path.exists():
        return {}
    try:
        j = _read_json(path)
    except Exception:
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    rows = j.get("rows") if isinstance(j.get("rows"), list) else []
    for r in rows:
        if not isinstance(r, dict):
            continue
        r_id = str(r.get("id") or "")
        if not r_id:
            continue
        sigma_total = _maybe_float(r.get("sigma_total"))
        if sigma_total is None or not (sigma_total > 0.0):
            continue
        out[r_id] = {
            "sigma_total": float(sigma_total),
            "sigma_sys_category": _maybe_float(r.get("sigma_sys_category")),
            "category": str(r.get("category") or "") or None,
        }
    return out


def _apply_ddr_sigma_policy(ddr: DDRConstraint, *, policy: str, envelope: Dict[str, Dict[str, Any]]) -> DDRConstraint:
    if policy != "category_sys":
        return replace(ddr, sigma_policy="raw")

    row = envelope.get(ddr.id)
    if not row:
        return replace(ddr, sigma_policy="raw")

    sigma_total = _maybe_float(row.get("sigma_total"))
    if sigma_total is None or not (sigma_total > 0.0):
        return replace(ddr, sigma_policy="raw")

    return replace(
        ddr,
        epsilon0_sigma=float(sigma_total),
        sigma_sys_category=_maybe_float(row.get("sigma_sys_category")),
        sigma_policy="category_sys",
        category=str(row.get("category") or "") or None,
    )


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
                title=str(r.get("title") or r.get("short_label") or r.get("id") or ""),
                mean=mean,
                sigma=sig,
                uses_bao=(bool(r.get(uses_bao_key)) if uses_bao_key in r and r[uses_bao_key] is not None else None),
                uses_cmb=(bool(r.get(uses_cmb_key)) if uses_cmb_key in r and r[uses_cmb_key] is not None else None),
                assumes_cddr=(
                    bool(r.get(assumes_cddr_key))
                    if assumes_cddr_key in r and r[assumes_cddr_key] is not None
                    else None
                ),
                source=dict(r.get("source") or {}),
            )
        )
    return out


def _as_pT_constraints(rows: Sequence[Dict[str, Any]]) -> List[GaussianConstraint]:
    """
    Convert CMB temperature scaling constraints (beta_T) into p_T=1-beta_T.
    We treat p_e ≈ p_T as the photon-energy redshift exponent.
    """
    out: List[GaussianConstraint] = []
    for r in rows:
        try:
            beta = float(r["beta_T"])
            sig = float(r["beta_T_sigma"])
        except Exception:
            continue
        if not (sig > 0.0 and math.isfinite(sig)):
            continue
        out.append(
            GaussianConstraint(
                id=str(r.get("id") or ""),
                short_label=str(r.get("short_label") or r.get("id") or ""),
                title=str(r.get("title") or r.get("short_label") or r.get("id") or ""),
                mean=float(1.0 - beta),
                sigma=float(sig),
                uses_bao=None,
                uses_cmb=None,
                assumes_cddr=None,
                source=dict(r.get("source") or {}),
            )
        )
    return out


def _wls_fit(
    *,
    ddr: DDRConstraint,
    sR_bao: float,
    sR_bao_sigma: float,
    opacity: GaussianConstraint,
    candle: GaussianConstraint,
    p_t: GaussianConstraint,
    p_e: GaussianConstraint,
) -> Dict[str, Any]:
    """
    Fit θ=[s_R, α, s_L, p_t, p_e] with observations:
      - DDR: ε0 = 0.5 p_e + 0.5 p_t -0.5 s_L + s_R + α - 2
      - BAO: s_R = sR_bao
      - α, s_L, p_t, p_e: direct constraints
    """

    obs_names = [
        "DDR ε0",
        "BAO s_R",
        "Opacity α",
        "Candle s_L",
        "SN time dilation p_t",
        "CMB energy p_e",
    ]

    # y is moved so that the model is linear without a constant term.
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

    # Design matrix A s.t. y ≈ A θ.
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
    ATA = A.T @ W @ A
    ATy = A.T @ W @ y
    theta = np.linalg.solve(ATA, ATy)

    pred = A @ theta
    z = (pred - y) / np.maximum(1e-300, sig)
    chi2 = float(np.sum(z**2))
    dof = int(len(y) - len(theta))
    max_abs_z = float(np.max(np.abs(z)))
    limiting_idx = int(np.argmax(np.abs(z)))

    theta_d = {
        "s_R": float(theta[0]),
        "alpha_opacity": float(theta[1]),
        "s_L": float(theta[2]),
        "p_t": float(theta[3]),
        "p_e": float(theta[4]),
    }
    z_d = {name: float(z[i]) for i, name in enumerate(obs_names)}

    return {
        "theta": theta_d,
        "z_scores": z_d,
        "chi2": chi2,
        "dof": dof,
        "chi2_dof": (chi2 / dof) if dof > 0 else None,
        "max_abs_z": max_abs_z,
        "limiting_observation": obs_names[limiting_idx],
    }


def _choose_best(
    candidates: Sequence[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None

    def key(c: Dict[str, Any]) -> Tuple[float, float]:
        return (float(c["fit"]["max_abs_z"]), float(c["fit"]["chi2"]))

    return min(candidates, key=key)


def _best_with_fixed_opacity(
    *,
    ddr: DDRConstraint,
    sR_bao: float,
    sR_bao_sigma: float,
    opacity_fixed: GaussianConstraint,
    candle_candidates: Sequence[GaussianConstraint],
    p_t: GaussianConstraint,
    p_e: GaussianConstraint,
) -> Optional[Dict[str, Any]]:
    if not candle_candidates:
        return None

    candidates: List[Dict[str, Any]] = []
    for cd in candle_candidates:
        fit = _wls_fit(
            ddr=ddr,
            sR_bao=sR_bao,
            sR_bao_sigma=sR_bao_sigma,
            opacity=opacity_fixed,
            candle=cd,
            p_t=p_t,
            p_e=p_e,
        )
        candidates.append(
            {
                "opacity": {"id": opacity_fixed.id, "short_label": opacity_fixed.short_label},
                "candle": {"id": cd.id, "short_label": cd.short_label},
                "fit": fit,
            }
        )
    return _choose_best(candidates)


def _compute_per_ddr(
    *,
    ddr: Sequence[DDRConstraint],
    sR_bao: float,
    sR_bao_sigma: float,
    opacity_all: Sequence[GaussianConstraint],
    candle_all: Sequence[GaussianConstraint],
    p_t: GaussianConstraint,
    p_e: GaussianConstraint,
    gw_siren_opacity_observed: Optional[GaussianConstraint],
    gw_siren_opacity_forecast: Optional[GaussianConstraint],
) -> List[Dict[str, Any]]:
    opacity_ind = [c for c in opacity_all if c.is_independent()]
    candle_ind = [c for c in candle_all if c.is_independent()]

    per_ddr: List[Dict[str, Any]] = []
    for d in ddr:
        candidates_any: List[Dict[str, Any]] = []
        for op in opacity_all:
            for cd in candle_all:
                fit = _wls_fit(
                    ddr=d,
                    sR_bao=sR_bao,
                    sR_bao_sigma=sR_bao_sigma,
                    opacity=op,
                    candle=cd,
                    p_t=p_t,
                    p_e=p_e,
                )
                candidates_any.append(
                    {
                        "opacity": {"id": op.id, "short_label": op.short_label},
                        "candle": {"id": cd.id, "short_label": cd.short_label},
                        "fit": fit,
                    }
                )

        candidates_ind: List[Dict[str, Any]] = []
        for op in opacity_ind:
            for cd in candle_ind:
                fit = _wls_fit(
                    ddr=d,
                    sR_bao=sR_bao,
                    sR_bao_sigma=sR_bao_sigma,
                    opacity=op,
                    candle=cd,
                    p_t=p_t,
                    p_e=p_e,
                )
                candidates_ind.append(
                    {
                        "opacity": {"id": op.id, "short_label": op.short_label},
                        "candle": {"id": cd.id, "short_label": cd.short_label},
                        "fit": fit,
                    }
                )

        best_any = _choose_best(candidates_any)
        best_ind = _choose_best(candidates_ind)
        best_ind_gw_obs = (
            None
            if gw_siren_opacity_observed is None
            else _best_with_fixed_opacity(
                ddr=d,
                sR_bao=sR_bao,
                sR_bao_sigma=sR_bao_sigma,
                opacity_fixed=gw_siren_opacity_observed,
                candle_candidates=candle_ind,
                p_t=p_t,
                p_e=p_e,
            )
        )
        best_ind_gw_forecast = (
            None
            if gw_siren_opacity_forecast is None
            else _best_with_fixed_opacity(
                ddr=d,
                sR_bao=sR_bao,
                sR_bao_sigma=sR_bao_sigma,
                opacity_fixed=gw_siren_opacity_forecast,
                candle_candidates=candle_ind,
                p_t=p_t,
                p_e=p_e,
            )
        )

        per_ddr.append(
            {
                "ddr": {
                    "id": d.id,
                    "short_label": d.short_label,
                    "uses_bao": d.uses_bao,
                    "epsilon0_obs": d.epsilon0,
                    "epsilon0_sigma": d.epsilon0_sigma,
                    "epsilon0_sigma_raw": d.epsilon0_sigma_raw,
                    "sigma_sys_category": d.sigma_sys_category,
                    "sigma_policy": d.sigma_policy,
                    "category": d.category,
                },
                "best_any": best_any,
                "best_independent": best_ind,
                "best_independent_gw_sirens": best_ind_gw_obs,
                "best_independent_gw_sirens_forecast": best_ind_gw_forecast,
            }
        )

    return per_ddr


def _pick_representative(
    rows: Sequence[Dict[str, Any]],
    uses_bao: bool,
    *,
    prefer_ind: bool,
) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_key = float("inf")
    for r in rows:
        if bool(r["ddr"]["uses_bao"]) != bool(uses_bao):
            continue
        cand = r.get("best_independent") if prefer_ind else r.get("best_any")
        if cand is None:
            continue
        v = float(cand["fit"]["max_abs_z"])
        if v < best_key:
            best_key = v
            best = r
    return best


def _parse_scales(s: str) -> List[float]:
    out: List[float] = []
    for tok in (s or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            v = float(tok)
        except Exception:
            continue
        if not (v > 0.0 and math.isfinite(v)):
            continue
        out.append(v)
    # Ensure unique & sorted (stable, deterministic).
    out = sorted(set(out))
    return out


def _plot_bao_sigma_scan(
    *,
    out_png: Path,
    scales: Sequence[float],
    rep_bao_series: Sequence[Dict[str, Any]],
    rep_no_bao_series: Sequence[Dict[str, Any]],
) -> None:
    _set_japanese_font()
    import matplotlib.pyplot as plt

    def _colors_for_limiting() -> Dict[str, str]:
        return {
            "DDR ε0": "#1f77b4",
            "BAO s_R": "#d62728",
            "Opacity α": "#2ca02c",
            "Candle s_L": "#9467bd",
            "SN time dilation p_t": "#8c564b",
            "CMB energy p_e": "#7f7f7f",
        }

    color_map = _colors_for_limiting()

    def _series_to_xy(series: Sequence[Dict[str, Any]]) -> Tuple[List[float], List[float], List[str]]:
        xs: List[float] = []
        ys: List[float] = []
        lims: List[str] = []
        for r in series:
            xs.append(float(r["bao_sigma_scale"]))
            ys.append(float(r["max_abs_z"]))
            lims.append(str(r.get("limiting_observation") or ""))
        return xs, ys, lims

    x_bao, y_bao, lim_bao = _series_to_xy(rep_bao_series)
    x_nb, y_nb, lim_nb = _series_to_xy(rep_no_bao_series)

    def _crossing_x_log(xs: Sequence[float], ys: Sequence[float], threshold: float) -> Optional[float]:
        if not xs or len(xs) != len(ys):
            return None
        if not math.isfinite(float(threshold)):
            return None
        # If already below threshold at the smallest f, return that value.
        if math.isfinite(float(ys[0])) and float(ys[0]) <= float(threshold):
            return float(xs[0])
        for i in range(1, len(xs)):
            x0 = float(xs[i - 1])
            x1 = float(xs[i])
            y0 = float(ys[i - 1])
            y1 = float(ys[i])
            if not (x0 > 0.0 and x1 > 0.0 and math.isfinite(x0) and math.isfinite(x1)):
                continue
            if not (math.isfinite(y0) and math.isfinite(y1)):
                continue
            # detect crossing from above to below (monotone not assumed)
            if (y0 - threshold) == 0.0:
                return x0
            if (y0 - threshold) * (y1 - threshold) > 0.0:
                continue
            if y1 == y0:
                return x1
            # interpolate in log-x space
            t = (threshold - y0) / (y1 - y0)
            lx0 = math.log10(x0)
            lx1 = math.log10(x1)
            lx = lx0 + float(t) * (lx1 - lx0)
            return float(10.0**lx)
        return None

    fig = plt.figure(figsize=(15.5, 6.5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, sharey=ax1)

    for ax, title, xs, ys, lims in [
        (ax1, "代表（BAO含むDDR）", x_bao, y_bao, lim_bao),
        (ax2, "代表（BAOなしDDR）", x_nb, y_nb, lim_nb),
    ]:
        ax.plot(xs, ys, color="#333333", linewidth=1.5, alpha=0.75, zorder=1)
        for x, y, lim in zip(xs, ys, lims):
            c = color_map.get(lim, "#999999")
            ax.scatter([x], [y], s=55, color=c, edgecolor="#111111", linewidth=0.4, zorder=3)
            ax.text(x, y + 0.08, f"{_fmt_float(y, digits=3)}σ", fontsize=9, ha="center", va="bottom", alpha=0.9)

        ax.set_xscale("log")
        ax.set_xlabel("BAO σスケール f（s_R の σ→fσ）", fontsize=10)
        ax.set_title(title, fontsize=12)
        ax.grid(True, axis="both", linestyle="--", alpha=0.30)
        for yline, txt in [(1.0, "1σ"), (3.0, "3σ"), (5.0, "5σ")]:
            ax.axhline(yline, color="#333333", linewidth=1.0, alpha=0.18)
            ax.text(xs[0], yline + 0.08, txt, fontsize=9, color="#333333", alpha=0.6)

    # Annotate where rep(BAO) reaches <=1σ (if ever).
    x_cross_1s = _crossing_x_log(x_bao, y_bao, 1.0)
    if x_cross_1s is not None and math.isfinite(float(x_cross_1s)) and float(x_cross_1s) > 0.0:
        ax1.axvline(float(x_cross_1s), color="#333333", linestyle=":", linewidth=1.2, alpha=0.35)
        ax1.text(
            float(x_cross_1s),
            0.18,
            f"max|z|≤1 → f≈{_fmt_float(float(x_cross_1s), digits=3)}",
            fontsize=9,
            rotation=90,
            ha="left",
            va="bottom",
            color="#333333",
            alpha=0.8,
        )

    ax1.set_ylabel("best_independent の max|z|（同時整合）", fontsize=10)

    # Legend (limiting observation color map)
    handles = []
    labels = []
    for k, c in color_map.items():
        h = ax2.scatter([], [], s=55, color=c, edgecolor="#111111", linewidth=0.4)
        handles.append(h)
        labels.append(k)
    ax2.legend(handles, labels, loc="upper left", fontsize=9, frameon=True, title="limiting（最大|z|の支配要因）")

    fig.suptitle("宇宙論（距離指標の再導出候補探索）：BAO(s_R) を soft constraint とした感度", fontsize=14)
    fig.text(
        0.5,
        0.02,
        "注：f>1 は BAO の重みを緩める（σを膨張）。点の色は best_independent の limiting（支配する拘束）を示す。",
        ha="center",
        fontsize=10,
    )
    plt.tight_layout(rect=(0.0, 0.05, 1.0, 0.92))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot(
    *,
    out_png: Path,
    per_ddr: Sequence[Dict[str, Any]],
    representatives: Optional[Dict[str, Any]] = None,
) -> None:
    _set_japanese_font()
    import matplotlib.pyplot as plt

    def _short_obs(s: Any) -> str:
        t = str(s or "").strip()
        if not t:
            return ""
        # Keep only the leading label (avoid extremely long strings in plots).
        return t.split(" / ")[0].strip()

    def _summarize_candidate(*, prefix: str, cand: Optional[Dict[str, Any]]) -> List[str]:
        if not isinstance(cand, dict):
            return [f"- {prefix}: なし"]
        fit = cand.get("fit") or {}
        theta = fit.get("theta") or {}
        z_scores = fit.get("z_scores") or {}
        max_abs_z = fit.get("max_abs_z")
        limiting = _short_obs(fit.get("limiting_observation"))
        opacity = cand.get("opacity") or {}
        candle = cand.get("candle") or {}
        lines = [
            f"- {prefix}: max|z|≈{_fmt_float(max_abs_z, digits=3)}σ（limiting={limiting}）",
            f"  α: {opacity.get('short_label','')} / s_L: {candle.get('short_label','')}",
            "  θ: "
            f"s_R={_fmt_float(theta.get('s_R'), digits=3)}, "
            f"α={_fmt_float(theta.get('alpha_opacity'), digits=3)}, "
            f"s_L={_fmt_float(theta.get('s_L'), digits=3)}, "
            f"p_t={_fmt_float(theta.get('p_t'), digits=3)}, "
            f"p_e={_fmt_float(theta.get('p_e'), digits=3)}",
        ]
        # Show top 4 |z| terms for readability.
        try:
            arr = [(k, float(v)) for k, v in z_scores.items() if isinstance(v, (int, float)) and math.isfinite(float(v))]
            arr.sort(key=lambda x: abs(x[1]), reverse=True)
            top = " / ".join([f"{k}={_fmt_float(v, digits=3)}" for k, v in arr[:4]])
            if top:
                lines.append(f"  z上位: {top}")
        except Exception:
            pass
        return lines

    labels = []
    any_vals = []
    ind_vals = []
    gw_obs_vals = []
    gw_fcst_vals = []
    any_colors = []
    ind_colors = []
    gw_obs_colors = []
    gw_fcst_colors = []

    for r in per_ddr:
        ddr = r["ddr"]
        label = str(ddr.get("short_label") or ddr.get("id") or "")
        if bool(ddr.get("uses_bao", False)):
            label = f"{label}（BAO含む）"
        labels.append(label)

        best_any = r.get("best_any")
        best_ind = r.get("best_independent")
        best_gw_obs = r.get("best_independent_gw_sirens")
        best_gw_fcst = r.get("best_independent_gw_sirens_forecast")

        v_any = float(best_any["fit"]["max_abs_z"]) if best_any else float("nan")
        v_ind = float(best_ind["fit"]["max_abs_z"]) if best_ind else float("nan")
        v_gw_obs = float(best_gw_obs["fit"]["max_abs_z"]) if best_gw_obs else float("nan")
        v_gw_fcst = float(best_gw_fcst["fit"]["max_abs_z"]) if best_gw_fcst else float("nan")
        any_vals.append(v_any)
        ind_vals.append(v_ind)
        gw_obs_vals.append(v_gw_obs)
        gw_fcst_vals.append(v_gw_fcst)

        any_colors.append(_classify_sigma(abs(v_any))[1] if math.isfinite(v_any) else "#cccccc")
        ind_colors.append(_classify_sigma(abs(v_ind))[1] if math.isfinite(v_ind) else "#dddddd")
        gw_obs_colors.append(_classify_sigma(abs(v_gw_obs))[1] if math.isfinite(v_gw_obs) else "#eeeeee")
        gw_fcst_colors.append(_classify_sigma(abs(v_gw_fcst))[1] if math.isfinite(v_gw_fcst) else "#f2f2f2")

    y = np.arange(len(labels))
    h = 0.18

    fig = plt.figure(figsize=(16, 8.8))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.18, 0.82))
    ax = fig.add_subplot(gs[0, 0])
    ax_text = fig.add_subplot(gs[0, 1])
    ax_text.axis("off")

    ax.barh(
        y - 1.5 * h,
        any_vals,
        height=h,
        color=any_colors,
        alpha=0.95,
        label="best_any（一次ソース候補から最小max|z|）",
    )
    ax.barh(
        y - 0.5 * h,
        ind_vals,
        height=h,
        color=ind_colors,
        alpha=0.65,
        label="best_independent（BAO/CMB/CDDR依存を避けた候補）",
    )
    ax.barh(
        y + 0.5 * h,
        gw_obs_vals,
        height=h,
        color=gw_obs_colors,
        alpha=0.45,
        label="GW（観測; αが未拘束に近い）を採用した best_independent（参考）",
    )
    ax.barh(
        y + 1.5 * h,
        gw_fcst_vals,
        height=h,
        color=gw_fcst_colors,
        alpha=0.35,
        label="GW（ET forecast; αを比較的強く拘束）を採用した best_independent（参考）",
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("最大|z|（DDR + BAO(s_R) + α + s_L + p_t + p_e の同時整合）", fontsize=11)
    ax.grid(True, axis="x", linestyle="--", alpha=0.35)
    for xline, txt in [(1.0, "1σ"), (3.0, "3σ"), (5.0, "5σ")]:
        ax.axvline(xline, color="#333333", linewidth=1.0, alpha=0.25)
        ax.text(xline + 0.08, -0.7, txt, fontsize=9, color="#333333", alpha=0.8)

    # Annotate values
    for i, v in enumerate(any_vals):
        if math.isfinite(v):
            ax.text(v + 0.08, float(i) - 1.5 * h, f"{_fmt_float(v, digits=3)}σ", va="center", fontsize=9)
    for i, v in enumerate(ind_vals):
        if math.isfinite(v):
            ax.text(v + 0.08, float(i) - 0.5 * h, f"{_fmt_float(v, digits=3)}σ", va="center", fontsize=9, alpha=0.9)
    for i, v in enumerate(gw_obs_vals):
        if math.isfinite(v):
            ax.text(v + 0.08, float(i) + 0.5 * h, f"{_fmt_float(v, digits=3)}σ", va="center", fontsize=9, alpha=0.8)
    for i, v in enumerate(gw_fcst_vals):
        if math.isfinite(v):
            ax.text(v + 0.08, float(i) + 1.5 * h, f"{_fmt_float(v, digits=3)}σ", va="center", fontsize=9, alpha=0.75)

    ax.legend(loc="lower right", fontsize=9, frameon=True)
    fig.suptitle("宇宙論（距離指標）：再導出候補の探索（同時整合できる組合せはあるか）", fontsize=14)
    fig.text(
        0.5,
        0.012,
        "max|z| は WLS で最小化（目的：どの拘束も極端に破らない組合せ）。色: <3σ（OK）/ 3〜5σ（要改善）/ >5σ（不一致）。"
        "GW（観測）は z≪1 のため αの拘束力が弱く、実質的に『αが未拘束ならどうなるか』の確認になる。"
        "GW（ET forecast）は将来の相対的に厳しい例。",
        ha="center",
        fontsize=10,
    )
    if isinstance(representatives, dict):
        rep_bao = representatives.get("bao") if isinstance(representatives.get("bao"), dict) else None
        rep_no = representatives.get("no_bao") if isinstance(representatives.get("no_bao"), dict) else None

        lines: List[str] = ["代表（詳細）", ""]
        if isinstance(rep_bao, dict):
            d = rep_bao.get("ddr") or {}
            lines.append(f"[BAO含むDDR] {d.get('short_label','')}")
            lines.extend(_summarize_candidate(prefix="best_any", cand=rep_bao.get("best_any")))
            lines.extend(_summarize_candidate(prefix="best_independent", cand=rep_bao.get("best_independent")))
            lines.append("")
        if isinstance(rep_no, dict):
            d = rep_no.get("ddr") or {}
            lines.append(f"[BAOなしDDR] {d.get('short_label','')}")
            lines.extend(_summarize_candidate(prefix="best_independent", cand=rep_no.get("best_independent")))
            lines.append("")

        lines.extend(
            [
                "注:",
                "- best_independent は BAO/CMB/CDDR依存を避けた一次ソース候補。",
                "- BAO(s_R) を緩める感度は別図（bao_sigma_scan）参照。",
            ]
        )
        ax_text.text(
            0.0,
            1.0,
            "\n".join(lines).strip(),
            ha="left",
            va="top",
            fontsize=10,
            color="#111111",
            linespacing=1.35,
        )

    plt.tight_layout(rect=(0.0, 0.04, 1.0, 0.93))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefer-independent", action="store_true", help="Prefer independent-only result in summary.")
    ap.add_argument(
        "--ddr-sigma-policy",
        type=str,
        default="category_sys",
        choices=["raw", "category_sys"],
        help=(
            "How to treat DDR ε0 uncertainty. "
            "'raw' uses epsilon0_sigma as-is from data. "
            "'category_sys' inflates it by category-level model spread (σ_cat) if the envelope metrics exists."
        ),
    )
    ap.add_argument(
        "--bao-sigma-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to BAO s_R sigma (soft constraint). default=1.0",
    )
    ap.add_argument(
        "--bao-sigma-scan-scales",
        type=str,
        default="1,1.5,2,2.5,3,4,6,10",
        help="Comma-separated positive floats for BAO sigma-scale scan (log-x).",
    )
    ap.add_argument("--skip-bao-sigma-scan", action="store_true", help="Skip BAO sigma-scale scan outputs.")
    args = ap.parse_args(argv)

    data_dir = _ROOT / "data" / "cosmology"
    out_dir = _ROOT / "output" / "cosmology"

    in_ddr = data_dir / "distance_duality_constraints.json"
    in_opacity = data_dir / "cosmic_opacity_constraints.json"
    in_gw_opacity = data_dir / "gw_standard_siren_opacity_constraints.json"
    in_candle = data_dir / "sn_standard_candle_evolution_constraints.json"
    in_pt = data_dir / "sn_time_dilation_constraints.json"
    in_pe = data_dir / "cmb_temperature_scaling_constraints.json"
    in_bao_fit = out_dir / "cosmology_bao_scaled_distance_fit_metrics.json"

    for p in (in_ddr, in_opacity, in_candle, in_pt, in_pe, in_bao_fit):
        if not p.exists():
            raise FileNotFoundError(f"missing input: {p}")

    ddr_rows = _read_json(in_ddr).get("constraints") or []
    opacity_rows = _read_json(in_opacity).get("constraints") or []
    gw_opacity_rows_all = (_read_json(in_gw_opacity).get("constraints") or []) if in_gw_opacity.exists() else []
    candle_rows = _read_json(in_candle).get("constraints") or []
    pt_rows = _read_json(in_pt).get("constraints") or []
    pe_rows = _read_json(in_pe).get("constraints") or []
    bao_fit = _read_json(in_bao_fit)

    ddr_sigma_policy = str(args.ddr_sigma_policy)
    ddr_env_path = out_dir / "cosmology_distance_duality_systematics_envelope_metrics.json"
    ddr_env = _load_ddr_systematics_envelope(ddr_env_path) if ddr_sigma_policy == "category_sys" else {}
    ddr = [
        _apply_ddr_sigma_policy(DDRConstraint.from_json(r), policy=ddr_sigma_policy, envelope=ddr_env) for r in ddr_rows
    ]
    opacity_all = _as_gaussian_list(opacity_rows, mean_key="alpha_opacity", sigma_key="alpha_opacity_sigma")
    gw_opacity_rows_observed = [r for r in gw_opacity_rows_all if not bool(r.get("is_forecast", False))]
    gw_opacity_rows_forecast = [r for r in gw_opacity_rows_all if bool(r.get("is_forecast", False))]
    gw_opacity_observed_all = _as_gaussian_list(
        gw_opacity_rows_observed, mean_key="alpha_opacity", sigma_key="alpha_opacity_sigma"
    )
    gw_opacity_forecast_all = _as_gaussian_list(
        gw_opacity_rows_forecast, mean_key="alpha_opacity", sigma_key="alpha_opacity_sigma"
    )
    gw_opacity_primary_observed = _pick_tightest(gw_opacity_observed_all)
    gw_opacity_primary_forecast = _pick_tightest(gw_opacity_forecast_all)
    candle_all = _as_gaussian_list(candle_rows, mean_key="s_L", sigma_key="s_L_sigma")
    pt_all = _as_gaussian_list(pt_rows, mean_key="p_t", sigma_key="p_t_sigma")
    pe_all = _as_gaussian_list(pe_rows, mean_key="p_T", sigma_key="p_T_sigma")
    pe_all_from_beta = _as_pT_constraints(pe_rows)
    if not pt_all:
        raise ValueError("no SN time dilation constraint found")
    if not pe_all_from_beta:
        raise ValueError("no CMB temperature scaling constraint found")

    # Use single constraints for p_t and p_e (the repo currently fixes one each).
    p_t = pt_all[0]
    p_e = pe_all_from_beta[0]

    try:
        sR_bao = float(bao_fit["fit"]["best_fit"]["s_R"])
        sR_bao_sigma = float(bao_fit["fit"]["best_fit"]["s_R_sigma_1d"])
    except Exception as e:
        raise ValueError("unexpected BAO fit metrics schema") from e

    if not (args.bao_sigma_scale > 0.0 and math.isfinite(float(args.bao_sigma_scale))):
        raise ValueError("--bao-sigma-scale must be positive and finite")
    sR_bao_sigma_used = float(sR_bao_sigma * float(args.bao_sigma_scale))

    per_ddr = _compute_per_ddr(
        ddr=ddr,
        sR_bao=sR_bao,
        sR_bao_sigma=sR_bao_sigma_used,
        opacity_all=opacity_all,
        candle_all=candle_all,
        p_t=p_t,
        p_e=p_e,
        gw_siren_opacity_observed=gw_opacity_primary_observed,
        gw_siren_opacity_forecast=gw_opacity_primary_forecast,
    )

    out_png = out_dir / "cosmology_distance_indicator_rederivation_candidate_search.png"
    out_metrics = out_dir / "cosmology_distance_indicator_rederivation_candidate_search_metrics.json"

    prefer_ind = bool(args.prefer_independent)
    rep_bao = _pick_representative(per_ddr, True, prefer_ind=prefer_ind)
    rep_no_bao = _pick_representative(per_ddr, False, prefer_ind=prefer_ind)

    _plot(out_png=out_png, per_ddr=per_ddr, representatives={"bao": rep_bao, "no_bao": rep_no_bao})

    applied_ddr_sigma_count = len([d for d in ddr if d.sigma_policy == "category_sys"])
    metrics: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "definition": {
            "epsilon0_model": "ε0 = (p_e + p_t - s_L)/2 - 2 + s_R + α",
            "objective": "minimize max|z| (then chi2) across DDR + BAO(s_R) + α + s_L + p_t + p_e",
                "notes": [
                    "This is a consistency scan, not a claim that any particular mechanism is correct.",
                    "If even best_any is large, the static infinite hypothesis would require redefining distance indicators beyond this parameterization.",
                    "If gw_standard_siren_opacity is provided, best_independent_gw_sirens uses the tightest observed standard-siren α and still scans s_L within independent candidates.",
                    "If forecast rows exist, best_independent_gw_sirens_forecast uses the tightest forecast α as a reference scenario.",
                    "DDR sigma can optionally include category-level systematics (σ_cat) via the systematics-envelope metrics.",
                ],
            },
        "ddr_sigma_policy": {
            "policy": ddr_sigma_policy,
            "envelope_metrics": (
                str(ddr_env_path.relative_to(_ROOT)).replace("\\", "/") if ddr_sigma_policy == "category_sys" else None
            ),
            "applied_count": applied_ddr_sigma_count,
            "note": "If envelope file is missing, σ_cat inflation is skipped for all rows (falls back to raw).",
        },
        "inputs": {
            "ddr": str(in_ddr.relative_to(_ROOT)).replace("\\", "/"),
            "opacity": str(in_opacity.relative_to(_ROOT)).replace("\\", "/"),
            "gw_standard_siren_opacity": (
                str(in_gw_opacity.relative_to(_ROOT)).replace("\\", "/") if in_gw_opacity.exists() else None
            ),
            "candle": str(in_candle.relative_to(_ROOT)).replace("\\", "/"),
            "sn_time_dilation": str(in_pt.relative_to(_ROOT)).replace("\\", "/"),
            "cmb_temperature_scaling": str(in_pe.relative_to(_ROOT)).replace("\\", "/"),
            "bao_fit": str(in_bao_fit.relative_to(_ROOT)).replace("\\", "/"),
        },
        "fixed_constraints": {
            "p_t": {"id": p_t.id, "mean": p_t.mean, "sigma": p_t.sigma, "short_label": p_t.short_label},
            "p_e": {"id": p_e.id, "mean": p_e.mean, "sigma": p_e.sigma, "short_label": p_e.short_label},
            "bao_s_R": {
                "mean": sR_bao,
                "sigma": sR_bao_sigma,
                "sigma_scale": float(args.bao_sigma_scale),
                "sigma_used": sR_bao_sigma_used,
            },
        },
        "independence_filter": {
            "rule": "uses_bao!=true AND uses_cmb!=true AND assumes_cddr!=true",
            "opacity_candidates": len([c for c in opacity_all if c.is_independent()]),
            "candle_candidates": len([c for c in candle_all if c.is_independent()]),
            "opacity_all": len(opacity_all),
            "candle_all": len(candle_all),
            "gw_siren_opacity_candidates_observed": len(gw_opacity_observed_all),
            "gw_siren_opacity_candidates_forecast": len(gw_opacity_forecast_all),
            "gw_siren_opacity_primary_observed": (
                None
                if gw_opacity_primary_observed is None
                else {
                    "id": gw_opacity_primary_observed.id,
                    "short_label": gw_opacity_primary_observed.short_label,
                    "mean": gw_opacity_primary_observed.mean,
                    "sigma": gw_opacity_primary_observed.sigma,
                }
            ),
            "gw_siren_opacity_primary_forecast": (
                None
                if gw_opacity_primary_forecast is None
                else {
                    "id": gw_opacity_primary_forecast.id,
                    "short_label": gw_opacity_primary_forecast.short_label,
                    "mean": gw_opacity_primary_forecast.mean,
                    "sigma": gw_opacity_primary_forecast.sigma,
                }
            ),
        },
        "results": {
            "per_ddr": per_ddr,
            "representatives": {
                "prefer_independent": prefer_ind,
                "bao": rep_bao,
                "no_bao": rep_no_bao,
            },
        },
        "outputs": {
            "png": str(out_png.relative_to(_ROOT)).replace("\\", "/"),
            "metrics_json": str(out_metrics.relative_to(_ROOT)).replace("\\", "/"),
        },
    }
    _write_json(out_metrics, metrics)

    worklog.append_event(
        {
            "kind": "cosmology",
            "step": "14.2.22",
            "task": "distance_indicator_rederivation_candidate_search",
            "inputs": [
                str(in_ddr.relative_to(_ROOT)).replace("\\", "/"),
                str(in_opacity.relative_to(_ROOT)).replace("\\", "/"),
                *([str(in_gw_opacity.relative_to(_ROOT)).replace("\\", "/")] if in_gw_opacity.exists() else []),
                str(in_candle.relative_to(_ROOT)).replace("\\", "/"),
                str(in_pt.relative_to(_ROOT)).replace("\\", "/"),
                str(in_pe.relative_to(_ROOT)).replace("\\", "/"),
                str(in_bao_fit.relative_to(_ROOT)).replace("\\", "/"),
            ],
            "outputs": {"png": out_png, "metrics_json": out_metrics},
            "metrics": {
                "bao_sigma_scale": float(args.bao_sigma_scale),
                "best_any_max_abs_z": {
                    r["ddr"]["id"]: (r.get("best_any") or {}).get("fit", {}).get("max_abs_z") for r in per_ddr
                },
                "best_independent_max_abs_z": {
                    r["ddr"]["id"]: (r.get("best_independent") or {}).get("fit", {}).get("max_abs_z")
                    for r in per_ddr
                },
                "best_independent_gw_sirens_max_abs_z": {
                    r["ddr"]["id"]: (r.get("best_independent_gw_sirens") or {}).get("fit", {}).get("max_abs_z")
                    for r in per_ddr
                },
                "best_independent_gw_sirens_forecast_max_abs_z": {
                    r["ddr"]["id"]: (r.get("best_independent_gw_sirens_forecast") or {}).get("fit", {}).get("max_abs_z")
                    for r in per_ddr
                },
            },
        }
    )

    if not bool(args.skip_bao_sigma_scan):
        scales = _parse_scales(str(args.bao_sigma_scan_scales))
        if not scales:
            scales = [1.0, 2.5, 10.0]
        # Ensure baseline f=1 is present for interpretability.
        if 1.0 not in scales:
            scales = sorted(set([1.0, *scales]))

        rep_bao_series: List[Dict[str, Any]] = []
        rep_no_bao_series: List[Dict[str, Any]] = []
        per_ddr_best_ind_max_abs_z_by_scale: Dict[str, Dict[str, float]] = {}
        for f in scales:
            per_ddr_f = _compute_per_ddr(
                ddr=ddr,
                sR_bao=sR_bao,
                sR_bao_sigma=float(sR_bao_sigma * float(f)),
                opacity_all=opacity_all,
                candle_all=candle_all,
                p_t=p_t,
                p_e=p_e,
                gw_siren_opacity_observed=gw_opacity_primary_observed,
                gw_siren_opacity_forecast=gw_opacity_primary_forecast,
            )

            rep_b = _pick_representative(per_ddr_f, True, prefer_ind=True)
            rep_n = _pick_representative(per_ddr_f, False, prefer_ind=True)

            def _extract(rep: Optional[Dict[str, Any]], *, f: float) -> Dict[str, Any]:
                if rep is None:
                    return {"bao_sigma_scale": float(f), "max_abs_z": float("nan"), "limiting_observation": "na"}
                cand = rep.get("best_independent")
                fit = (cand or {}).get("fit") or {}
                return {
                    "bao_sigma_scale": float(f),
                    "ddr_id": rep["ddr"]["id"],
                    "ddr_short_label": rep["ddr"]["short_label"],
                    "max_abs_z": float(fit.get("max_abs_z", float("nan"))),
                    "limiting_observation": str(fit.get("limiting_observation") or ""),
                    "z_scores": dict(fit.get("z_scores") or {}),
                    "theta": dict(fit.get("theta") or {}),
                    "selected_constraints": {
                        "opacity": (cand or {}).get("opacity"),
                        "candle": (cand or {}).get("candle"),
                    },
                }

            rep_bao_series.append(_extract(rep_b, f=f))
            rep_no_bao_series.append(_extract(rep_n, f=f))

            per_ddr_best_ind_max_abs_z_by_scale[str(f)] = {
                r["ddr"]["id"]: float(((r.get("best_independent") or {}).get("fit") or {}).get("max_abs_z", float("nan")))
                for r in per_ddr_f
            }

        out_scan_png = out_dir / "cosmology_distance_indicator_rederivation_candidate_search_bao_sigma_scan.png"
        out_scan_metrics = (
            out_dir / "cosmology_distance_indicator_rederivation_candidate_search_bao_sigma_scan_metrics.json"
        )

        def _crossing_x_log(xs: Sequence[float], ys: Sequence[float], threshold: float) -> Optional[float]:
            if not xs or len(xs) != len(ys):
                return None
            if not math.isfinite(float(threshold)):
                return None
            if math.isfinite(float(ys[0])) and float(ys[0]) <= float(threshold):
                return float(xs[0])
            for i in range(1, len(xs)):
                x0 = float(xs[i - 1])
                x1 = float(xs[i])
                y0 = float(ys[i - 1])
                y1 = float(ys[i])
                if not (x0 > 0.0 and x1 > 0.0 and math.isfinite(x0) and math.isfinite(x1)):
                    continue
                if not (math.isfinite(y0) and math.isfinite(y1)):
                    continue
                if (y0 - threshold) == 0.0:
                    return x0
                if (y0 - threshold) * (y1 - threshold) > 0.0:
                    continue
                if y1 == y0:
                    return x1
                t = (threshold - y0) / (y1 - y0)
                lx0 = math.log10(x0)
                lx1 = math.log10(x1)
                lx = lx0 + float(t) * (lx1 - lx0)
                return float(10.0**lx)
            return None

        rep_bao_x = [float(r.get("bao_sigma_scale", float("nan"))) for r in rep_bao_series]
        rep_bao_y = [float(r.get("max_abs_z", float("nan"))) for r in rep_bao_series]
        rep_no_x = [float(r.get("bao_sigma_scale", float("nan"))) for r in rep_no_bao_series]
        rep_no_y = [float(r.get("max_abs_z", float("nan"))) for r in rep_no_bao_series]

        rep_bao_f_1sigma = _crossing_x_log(rep_bao_x, rep_bao_y, 1.0)
        rep_no_f_1sigma = _crossing_x_log(rep_no_x, rep_no_y, 1.0)

        _plot_bao_sigma_scan(
            out_png=out_scan_png,
            scales=scales,
            rep_bao_series=rep_bao_series,
            rep_no_bao_series=rep_no_bao_series,
        )

        scan_metrics: Dict[str, Any] = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "definition": {
                "scan": "soften BAO(s_R) by scaling sigma: sigma_used = f * sigma_base",
                "objective": "for each f, compute best_independent and its max|z| and limiting observation",
            },
            "inputs": {
                "ddr": str(in_ddr.relative_to(_ROOT)).replace("\\", "/"),
                "opacity": str(in_opacity.relative_to(_ROOT)).replace("\\", "/"),
                "candle": str(in_candle.relative_to(_ROOT)).replace("\\", "/"),
                "sn_time_dilation": str(in_pt.relative_to(_ROOT)).replace("\\", "/"),
                "cmb_temperature_scaling": str(in_pe.relative_to(_ROOT)).replace("\\", "/"),
                "bao_fit": str(in_bao_fit.relative_to(_ROOT)).replace("\\", "/"),
            },
            "fixed_constraints": {
                "p_t": {"id": p_t.id, "mean": p_t.mean, "sigma": p_t.sigma, "short_label": p_t.short_label},
                "p_e": {"id": p_e.id, "mean": p_e.mean, "sigma": p_e.sigma, "short_label": p_e.short_label},
                "bao_s_R_base": {"mean": sR_bao, "sigma_base": sR_bao_sigma},
            },
            "results": {
                "scales": [float(f) for f in scales],
                "representatives_best_independent": {
                    "bao": rep_bao_series,
                    "no_bao": rep_no_bao_series,
                },
                "per_ddr_best_independent_max_abs_z_by_scale": per_ddr_best_ind_max_abs_z_by_scale,
                "thresholds": {
                    "rep_bao_f_for_max_abs_z_le_1sigma": rep_bao_f_1sigma,
                    "rep_no_bao_f_for_max_abs_z_le_1sigma": rep_no_f_1sigma,
                },
            },
            "outputs": {
                "png": str(out_scan_png.relative_to(_ROOT)).replace("\\", "/"),
                "metrics_json": str(out_scan_metrics.relative_to(_ROOT)).replace("\\", "/"),
            },
        }
        _write_json(out_scan_metrics, scan_metrics)

        worklog.append_event(
            {
                "kind": "cosmology",
                "step": "14.2.28",
                "task": "distance_indicator_rederivation_candidate_search_bao_sigma_scan",
                "inputs": [
                    str(in_ddr.relative_to(_ROOT)).replace("\\", "/"),
                    str(in_opacity.relative_to(_ROOT)).replace("\\", "/"),
                    str(in_candle.relative_to(_ROOT)).replace("\\", "/"),
                    str(in_pt.relative_to(_ROOT)).replace("\\", "/"),
                    str(in_pe.relative_to(_ROOT)).replace("\\", "/"),
                    str(in_bao_fit.relative_to(_ROOT)).replace("\\", "/"),
                ],
                "outputs": {"png": out_scan_png, "metrics_json": out_scan_metrics},
                "metrics": {
                    "scales": [float(f) for f in scales],
                    "rep_bao_max_abs_z_by_scale": {str(r["bao_sigma_scale"]): r["max_abs_z"] for r in rep_bao_series},
                    "rep_no_bao_max_abs_z_by_scale": {str(r["bao_sigma_scale"]): r["max_abs_z"] for r in rep_no_bao_series},
                    "rep_bao_f_for_max_abs_z_le_1sigma": rep_bao_f_1sigma,
                    "rep_no_bao_f_for_max_abs_z_le_1sigma": rep_no_f_1sigma,
                },
            }
        )

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_metrics}")
    if not bool(args.skip_bao_sigma_scan):
        print(f"[ok] png : {out_scan_png}")
        print(f"[ok] json: {out_scan_metrics}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
