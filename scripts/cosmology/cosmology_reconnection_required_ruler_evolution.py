#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_reconnection_required_ruler_evolution.py

Step 14.2.2（一次ソースで拘束を追加）:
距離二重性（DDR）の観測制約（ε0_obs）と、不透明度 α・標準光源進化 s_L の一次ソース拘束を同時に満たすために、
静的背景P（膨張なし）の“距離指標側の自由度”として必要となる標準定規進化 s_R を定量化する。

背景:
  DDR のパラメータ化:
    D_L(z) = (1+z)^(2+ε0) D_A(z)  →  η(z)=D_L/((1+z)^2 D_A)=(1+z)^(ε0)

  静的背景P（膨張なし）で、赤方偏移（光子エネルギー）と時間伸長（到達率）を一般化し、
  距離指標の前提（光度/定規/不透明度）を
    - 光子エネルギー: E ∝ (1+z)^(-p_e)
    - 時間伸長: Δt_obs = (1+z)^(p_t) Δt_em
    - 標準光源: L_em = L0 (1+z)^(s_L)
    - 標準定規: l_em = l0 (1+z)^(s_R)
    - 不透明度: exp(+τ/2)=(1+z)^α（τ(z)=2α ln(1+z)）
  と置くと、DDR破れ指数は（最小仮定の整理）

    ε0_model = (p_e + p_t - s_L)/2 - 2 + s_R + α

  となる（p_e=p_t=1, s_L=s_R=α=0 で静的最小 ε0=-1 を回収）。

  本スクリプトでは、独立プローブ（SN time dilation, CMB T(z)）の一次ソース拘束を取り込み（任意）、
  ε0_model=ε0_obs を満たすように s_R を解く（p_e/p_t は固定値でも可）。

  例えば p_e=p_t=1 の場合は

    s_R = ε0_obs - α + s_L/2 + 1

  となる（線形結合のため、平均とσは誤差伝播で解析的に求まる）。

入力（固定）:
  - data/cosmology/distance_duality_constraints.json
  - data/cosmology/cosmic_opacity_constraints.json
  - data/cosmology/sn_standard_candle_evolution_constraints.json
  - （任意）data/cosmology/sn_time_dilation_constraints.json
  - （任意）data/cosmology/cmb_temperature_scaling_constraints.json

出力（固定名）:
  - output/private/cosmology/cosmology_reconnection_required_ruler_evolution.png
  - output/private/cosmology/cosmology_reconnection_required_ruler_evolution_metrics.json
  - output/private/cosmology/cosmology_reconnection_required_ruler_evolution_independent.png
  - output/private/cosmology/cosmology_reconnection_required_ruler_evolution_independent_metrics.json
  - output/private/cosmology/cosmology_bao_scaled_distance_fit.png
  - output/private/cosmology/cosmology_bao_scaled_distance_fit_metrics.json
  - output/private/cosmology/cosmology_bao_scaled_distance_fit_independent.png
  - output/private/cosmology/cosmology_bao_scaled_distance_fit_independent_metrics.json
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
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


_C_KM_S = 299_792.458


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


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_write_json` の入出力契約と処理意図を定義する。

def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# 関数: `_safe_float` の入出力契約と処理意図を定義する。

def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None

    # 条件分岐: `math.isnan(v) or math.isinf(v)` を満たす経路を評価する。

    if math.isnan(v) or math.isinf(v):
        return None

    return v


# 関数: `_optional_bool` の入出力契約と処理意図を定義する。

def _optional_bool(j: Dict[str, Any], key: str) -> Optional[bool]:
    # 条件分岐: `key not in j` を満たす経路を評価する。
    if key not in j:
        return None

    v = j.get(key)
    # 条件分岐: `v is None` を満たす経路を評価する。
    if v is None:
        return None

    return bool(v)


# 関数: `_load_ddr_systematics_envelope` の入出力契約と処理意図を定義する。

def _load_ddr_systematics_envelope(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load id -> {sigma_total, sigma_sys_category, category} from
    `output/private/cosmology/cosmology_distance_duality_systematics_envelope_metrics.json`.

    This captures category-level model spread as a systematic-width proxy (σ_cat).
    """
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        return {}

    try:
        j = _read_json(path)
    except Exception:
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    rows = j.get("rows") if isinstance(j.get("rows"), list) else []
    for r in rows:
        # 条件分岐: `not isinstance(r, dict)` を満たす経路を評価する。
        if not isinstance(r, dict):
            continue

        r_id = str(r.get("id") or "")
        # 条件分岐: `not r_id` を満たす経路を評価する。
        if not r_id:
            continue

        sigma_total = _safe_float(r.get("sigma_total"))
        # 条件分岐: `sigma_total is None or not (sigma_total > 0.0)` を満たす経路を評価する。
        if sigma_total is None or not (sigma_total > 0.0):
            continue

        out[r_id] = {
            "sigma_total": float(sigma_total),
            "sigma_sys_category": _safe_float(r.get("sigma_sys_category")),
            "category": str(r.get("category") or "") or None,
        }

    return out


# 関数: `_apply_ddr_sigma_policy` の入出力契約と処理意図を定義する。

def _apply_ddr_sigma_policy(ddr: DDRConstraint, *, policy: str, envelope: Dict[str, Dict[str, Any]]) -> DDRConstraint:
    # 条件分岐: `policy != "category_sys"` を満たす経路を評価する。
    if policy != "category_sys":
        return replace(ddr, sigma_policy="raw")

    row = envelope.get(ddr.id)
    # 条件分岐: `not row` を満たす経路を評価する。
    if not row:
        return replace(ddr, sigma_policy="raw")

    sigma_total = _safe_float(row.get("sigma_total"))
    # 条件分岐: `sigma_total is None or not (sigma_total > 0.0)` を満たす経路を評価する。
    if sigma_total is None or not (sigma_total > 0.0):
        return replace(ddr, sigma_policy="raw")

    return replace(
        ddr,
        epsilon0_sigma=float(sigma_total),
        sigma_sys_category=_safe_float(row.get("sigma_sys_category")),
        sigma_policy="category_sys",
        category=str(row.get("category") or "") or None,
    )


# 関数: `_fmt_float` の入出力契約と処理意図を定義する。

def _fmt_float(x: Optional[float], *, digits: int = 6) -> str:
    # 条件分岐: `x is None` を満たす経路を評価する。
    if x is None:
        return ""

    # 条件分岐: `x == 0.0` を満たす経路を評価する。

    if x == 0.0:
        return "0"

    ax = abs(x)
    # 条件分岐: `ax >= 1e4 or ax < 1e-3` を満たす経路を評価する。
    if ax >= 1e4 or ax < 1e-3:
        return f"{x:.{digits}g}"

    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


# クラス: `DDRConstraint` の責務と境界条件を定義する。

@dataclass(frozen=True)
class DDRConstraint:
    id: str
    short_label: str
    title: str
    epsilon0: float
    epsilon0_sigma: float
    uses_bao: bool
    sigma_note: str
    source: Dict[str, Any]
    epsilon0_sigma_raw: float = 0.0
    sigma_sys_category: Optional[float] = None
    sigma_policy: str = "raw"
    category: Optional[str] = None

    # 関数: `from_json` の入出力契約と処理意図を定義する。
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
            sigma_note=str(j.get("sigma_note") or ""),
            source=dict(j.get("source") or {}),
            epsilon0_sigma_raw=sigma,
        )


# クラス: `OpacityConstraint` の責務と境界条件を定義する。

@dataclass(frozen=True)
class OpacityConstraint:
    id: str
    short_label: str
    title: str
    alpha_opacity: float
    alpha_opacity_sigma: float
    uses_bao: Optional[bool]
    uses_cmb: Optional[bool]
    sigma_note: str
    source: Dict[str, Any]

    # 関数: `from_json` の入出力契約と処理意図を定義する。
    @staticmethod
    def from_json(j: Dict[str, Any]) -> "OpacityConstraint":
        return OpacityConstraint(
            id=str(j["id"]),
            short_label=str(j.get("short_label") or j["id"]),
            title=str(j.get("title") or j.get("short_label") or j["id"]),
            alpha_opacity=float(j["alpha_opacity"]),
            alpha_opacity_sigma=float(j["alpha_opacity_sigma"]),
            uses_bao=_optional_bool(j, "uses_bao"),
            uses_cmb=_optional_bool(j, "uses_cmb"),
            sigma_note=str(j.get("sigma_note") or ""),
            source=dict(j.get("source") or {}),
        )


# クラス: `CandleEvoConstraint` の責務と境界条件を定義する。

@dataclass(frozen=True)
class CandleEvoConstraint:
    id: str
    short_label: str
    title: str
    s_L: float
    s_L_sigma: float
    uses_bao: Optional[bool]
    uses_cmb: Optional[bool]
    assumes_cddr: Optional[bool]
    sigma_note: str
    source: Dict[str, Any]

    # 関数: `from_json` の入出力契約と処理意図を定義する。
    @staticmethod
    def from_json(j: Dict[str, Any]) -> "CandleEvoConstraint":
        return CandleEvoConstraint(
            id=str(j["id"]),
            short_label=str(j.get("short_label") or j["id"]),
            title=str(j.get("title") or j.get("short_label") or j["id"]),
            s_L=float(j["s_L"]),
            s_L_sigma=float(j["s_L_sigma"]),
            uses_bao=_optional_bool(j, "uses_bao"),
            uses_cmb=_optional_bool(j, "uses_cmb"),
            assumes_cddr=_optional_bool(j, "assumes_cddr"),
            sigma_note=str(j.get("sigma_note") or ""),
            source=dict(j.get("source") or {}),
        )


# クラス: `TimeDilationConstraint` の責務と境界条件を定義する。

@dataclass(frozen=True)
class TimeDilationConstraint:
    id: str
    short_label: str
    title: str
    p_t: float
    p_t_sigma: float
    sigma_note: str
    source: Dict[str, Any]

    # 関数: `from_json` の入出力契約と処理意図を定義する。
    @staticmethod
    def from_json(j: Dict[str, Any]) -> "TimeDilationConstraint":
        return TimeDilationConstraint(
            id=str(j["id"]),
            short_label=str(j.get("short_label") or j["id"]),
            title=str(j.get("title") or j.get("short_label") or j["id"]),
            p_t=float(j["p_t"]),
            p_t_sigma=float(j["p_t_sigma"]),
            sigma_note=str(j.get("sigma_note") or ""),
            source=dict(j.get("source") or {}),
        )


# クラス: `CMBTemperatureConstraint` の責務と境界条件を定義する。

@dataclass(frozen=True)
class CMBTemperatureConstraint:
    id: str
    short_label: str
    title: str
    beta_T: float
    beta_T_sigma: float
    sigma_note: str
    source: Dict[str, Any]

    # 関数: `p_T` の入出力契約と処理意図を定義する。
    @property
    def p_T(self) -> float:
        return 1.0 - float(self.beta_T)

    # 関数: `p_T_sigma` の入出力契約と処理意図を定義する。

    @property
    def p_T_sigma(self) -> float:
        return float(self.beta_T_sigma)

    # 関数: `from_json` の入出力契約と処理意図を定義する。

    @staticmethod
    def from_json(j: Dict[str, Any]) -> "CMBTemperatureConstraint":
        return CMBTemperatureConstraint(
            id=str(j["id"]),
            short_label=str(j.get("short_label") or j["id"]),
            title=str(j.get("title") or j.get("short_label") or j["id"]),
            beta_T=float(j["beta_T"]),
            beta_T_sigma=float(j["beta_T_sigma"]),
            sigma_note=str(j.get("sigma_note") or ""),
            source=dict(j.get("source") or {}),
        )


# クラス: `BAOAnisotropyConstraint` の責務と境界条件を定義する。

@dataclass(frozen=True)
class BAOAnisotropyConstraint:
    id: str
    short_label: str
    z_eff: float
    DM_scaled_mpc: float
    DM_scaled_sigma_mpc: float
    H_scaled_km_s_mpc: float
    H_scaled_sigma_km_s_mpc: float
    corr_DM_H: float
    sigma_note: str
    source: Dict[str, Any]

    # 関数: `from_json` の入出力契約と処理意図を定義する。
    @staticmethod
    def from_json(j: Dict[str, Any]) -> "BAOAnisotropyConstraint":
        return BAOAnisotropyConstraint(
            id=str(j["id"]),
            short_label=str(j.get("short_label") or j["id"]),
            z_eff=float(j["z_eff"]),
            DM_scaled_mpc=float(j["DM_scaled_mpc"]),
            DM_scaled_sigma_mpc=float(j["DM_scaled_sigma_mpc"]),
            H_scaled_km_s_mpc=float(j["H_scaled_km_s_mpc"]),
            H_scaled_sigma_km_s_mpc=float(j["H_scaled_sigma_km_s_mpc"]),
            corr_DM_H=float(j.get("corr_DM_H") or 0.0),
            sigma_note=str(j.get("sigma_note") or ""),
            source=dict(j.get("source") or {}),
        )


# 関数: `_primary_by_sigma` の入出力契約と処理意図を定義する。

def _primary_by_sigma(rows: Sequence[Any], sigma_field: str) -> Any:
    # 条件分岐: `not rows` を満たす経路を評価する。
    if not rows:
        raise ValueError("empty constraint list")

    best = None
    best_sig = float("inf")
    for r in rows:
        sig = _safe_float(getattr(r, sigma_field, None))
        # 条件分岐: `sig is None or sig <= 0` を満たす経路を評価する。
        if sig is None or sig <= 0:
            continue

        # 条件分岐: `sig < best_sig` を満たす経路を評価する。

        if sig < best_sig:
            best_sig = sig
            best = r

    # 条件分岐: `best is None` を満たす経路を評価する。

    if best is None:
        # Fall back to first element (even if sigma is missing) to avoid hard failure.
        return rows[0]

    return best


# 関数: `_select_primary_opacity` の入出力契約と処理意図を定義する。

def _select_primary_opacity(
    rows: Sequence[OpacityConstraint], *, independent_only: bool
) -> Tuple[OpacityConstraint, Dict[str, Any]]:
    eligible = (
        [r for r in rows if (r.uses_bao is False) and (r.uses_cmb is False)] if independent_only else list(rows)
    )
    primary = _primary_by_sigma(eligible or rows, "alpha_opacity_sigma")
    return primary, {
        "independent_only": bool(independent_only),
        "eligible_count": int(len(eligible)),
        "eligible_ids": [r.id for r in eligible],
        "fallback_used": bool(independent_only and not eligible),
    }


# 関数: `_select_primary_candle` の入出力契約と処理意図を定義する。

def _select_primary_candle(
    rows: Sequence[CandleEvoConstraint], *, independent_only: bool
) -> Tuple[CandleEvoConstraint, Dict[str, Any]]:
    eligible = (
        [r for r in rows if (r.uses_bao is False) and (r.uses_cmb is False)] if independent_only else list(rows)
    )
    pool = eligible or list(rows)
    pool_no_cddr = [r for r in pool if r.assumes_cddr is not True]
    primary = _primary_by_sigma(pool_no_cddr or pool, "s_L_sigma")
    excluded = [r for r in pool if r.assumes_cddr is True]
    return primary, {
        "independent_only": bool(independent_only),
        "eligible_count": int(len(eligible)),
        "eligible_ids": [r.id for r in eligible],
        "fallback_used": bool(independent_only and not eligible),
        "cddr_excluded_count": int(len(excluded)),
        "cddr_excluded_ids": [r.id for r in excluded],
        "cddr_fallback_used": bool(not pool_no_cddr),
    }


# 関数: `_primary_flags` の入出力契約と処理意図を定義する。

def _primary_flags(payload: Any) -> Dict[str, Any]:
    uses_bao = getattr(payload, "uses_bao", None)
    uses_cmb = getattr(payload, "uses_cmb", None)
    assumes_cddr = getattr(payload, "assumes_cddr", None)
    return {
        "uses_bao": (None if uses_bao is None else bool(uses_bao)),
        "uses_cmb": (None if uses_cmb is None else bool(uses_cmb)),
        "assumes_cddr": (None if assumes_cddr is None else bool(assumes_cddr)),
    }


# 関数: `_required_s_r` の入出力契約と処理意図を定義する。

def _required_s_r(
    *,
    eps0: float,
    eps0_sigma: float,
    alpha: float,
    alpha_sigma: float,
    s_L: float,
    s_L_sigma: float,
    p_e: float,
    p_e_sigma: float,
    p_t: float,
    p_t_sigma: float,
) -> Tuple[float, float]:
    # Solve epsilon0_model = eps0 for s_R:
    #   eps0 = (p_e + p_t - s_L)/2 - 2 + s_R + alpha
    # -> s_R = eps0 - alpha - (p_e+p_t - s_L)/2 + 2
    # Linear in variables => gaussian propagation.
    s_r = float(eps0 - alpha - (p_e + p_t - s_L) / 2.0 + 2.0)
    var = 0.0
    # 条件分岐: `eps0_sigma > 0` を満たす経路を評価する。
    if eps0_sigma > 0:
        var += float(eps0_sigma) ** 2

    # 条件分岐: `alpha_sigma > 0` を満たす経路を評価する。

    if alpha_sigma > 0:
        var += float(alpha_sigma) ** 2

    # 条件分岐: `s_L_sigma > 0` を満たす経路を評価する。

    if s_L_sigma > 0:
        var += (0.5 * float(s_L_sigma)) ** 2

    # 条件分岐: `p_e_sigma > 0` を満たす経路を評価する。

    if p_e_sigma > 0:
        var += (0.5 * float(p_e_sigma)) ** 2

    # 条件分岐: `p_t_sigma > 0` を満たす経路を評価する。

    if p_t_sigma > 0:
        var += (0.5 * float(p_t_sigma)) ** 2

    sig = float(math.sqrt(var)) if var > 0 else float("nan")
    return s_r, sig


# 関数: `_lognormal_quantiles_from_normal` の入出力契約と処理意図を定義する。

def _lognormal_quantiles_from_normal(*, mu: float, sig: float, a: float) -> Dict[str, float]:
    """
    If X ~ Normal(mu, sig), return quantiles of Y = a^X at +-1σ:
      median = a^mu
      p16 = a^(mu - sig)
      p84 = a^(mu + sig)
    """
    ln_a = math.log(float(a))
    median = math.exp(mu * ln_a)
    p16 = math.exp((mu - sig) * ln_a)
    p84 = math.exp((mu + sig) * ln_a)
    return {"median": float(median), "p16": float(p16), "p84": float(p84)}


# 関数: `_bao_pred_dm_h` の入出力契約と処理意図を定義する。

def _bao_pred_dm_h(*, z: float, s_R: float, B: float) -> Tuple[float, float]:
    """
    Predict BOSS-style BAO anisotropy outputs under static background-P geometry + ruler evolution.

    Model:
      D_M(z) = (c/H0) ln(1+z)   (static geometry, exponential P_bg)
      H_eff(z) = H0 (1+z)

    BAO outputs are (D_M * r_d,fid/r_d,  H * r_d/r_d,fid).
    If the effective ruler evolves as r_d(z)=r_d0 (1+z)^{s_R}, then the outputs scale as:
      D_M_scaled(z) ∝ D_M(z) * (1+z)^{-s_R}
      H_scaled(z)   ∝ H(z)   * (1+z)^{+s_R}

    Here we absorb the unknown overall factor H0*(r_d0/r_d,fid) into a single parameter B [km/s/Mpc].
    """
    op = 1.0 + float(z)
    # 条件分岐: `not (op > 0.0)` を満たす経路を評価する。
    if not (op > 0.0):
        raise ValueError("z must satisfy 1+z>0")

    B = float(B)
    # 条件分岐: `not (B > 0.0)` を満たす経路を評価する。
    if not (B > 0.0):
        raise ValueError("B must be > 0")

    dm_scaled = (_C_KM_S / B) * math.log(op) * (op ** (-float(s_R)))
    h_scaled = B * (op ** (1.0 + float(s_R)))
    return float(dm_scaled), float(h_scaled)


# 関数: `_chi2_bao_dm_h` の入出力契約と処理意図を定義する。

def _chi2_bao_dm_h(rows: Sequence[BAOAnisotropyConstraint], *, s_R: float, B: float) -> float:
    chi2 = 0.0
    for r in rows:
        dm_pred, h_pred = _bao_pred_dm_h(z=r.z_eff, s_R=s_R, B=B)
        d_dm = float(r.DM_scaled_mpc) - dm_pred
        d_h = float(r.H_scaled_km_s_mpc) - h_pred

        sig_dm = float(r.DM_scaled_sigma_mpc)
        sig_h = float(r.H_scaled_sigma_km_s_mpc)
        cov = float(r.corr_DM_H) * sig_dm * sig_h
        a = sig_dm * sig_dm
        d = sig_h * sig_h
        det = a * d - cov * cov
        # 条件分岐: `not (det > 0.0)` を満たす経路を評価する。
        if not (det > 0.0):
            return float("nan")

        inv11 = d / det
        inv22 = a / det
        inv12 = -cov / det
        chi2 += inv11 * d_dm * d_dm + 2.0 * inv12 * d_dm * d_h + inv22 * d_h * d_h

    return float(chi2)


# 関数: `_fit_bao_B_for_sR` の入出力契約と処理意図を定義する。

def _fit_bao_B_for_sR(
    rows: Sequence[BAOAnisotropyConstraint],
    *,
    s_R: float,
    B_min: float = 5.0,
    B_max: float = 200.0,
    n_grid: int = 2501,
) -> Tuple[float, float]:
    Bs = np.linspace(float(B_min), float(B_max), int(n_grid), dtype=float)
    chi = np.array([_chi2_bao_dm_h(rows, s_R=float(s_R), B=float(b)) for b in Bs], dtype=float)
    # 条件分岐: `not np.any(np.isfinite(chi))` を満たす経路を評価する。
    if not np.any(np.isfinite(chi)):
        return float("nan"), float("nan")

    i = int(np.nanargmin(chi))
    b0 = float(Bs[i])

    # Refine locally.
    lo = max(float(B_min), b0 - 2.0)
    hi = min(float(B_max), b0 + 2.0)
    Bs2 = np.linspace(lo, hi, 2001, dtype=float)
    chi2 = np.array([_chi2_bao_dm_h(rows, s_R=float(s_R), B=float(b)) for b in Bs2], dtype=float)
    j = int(np.nanargmin(chi2))
    return float(Bs2[j]), float(chi2[j])


# 関数: `_fit_bao_ruler_evolution` の入出力契約と処理意図を定義する。

def _fit_bao_ruler_evolution(
    rows: Sequence[BAOAnisotropyConstraint],
    *,
    s_R_min: float = -1.0,
    s_R_max: float = 2.0,
) -> Dict[str, Any]:
    # 1) Coarse search over s_R, profiling out B.
    s_grid0 = np.linspace(float(s_R_min), float(s_R_max), 301, dtype=float)
    prof0: List[Dict[str, float]] = []
    for s in s_grid0:
        b, c2 = _fit_bao_B_for_sR(rows, s_R=float(s))
        prof0.append({"s_R": float(s), "B_best": float(b), "chi2": float(c2)})

    best0 = min(prof0, key=lambda x: (x["chi2"] if math.isfinite(x["chi2"]) else float("inf")))

    # 2) Refine around the best coarse point.
    s0 = float(best0["s_R"])
    lo = max(float(s_R_min), s0 - 0.2)
    hi = min(float(s_R_max), s0 + 0.2)
    s_grid = np.linspace(lo, hi, 801, dtype=float)
    prof: List[Dict[str, float]] = []
    for s in s_grid:
        b, c2 = _fit_bao_B_for_sR(rows, s_R=float(s))
        prof.append({"s_R": float(s), "B_best": float(b), "chi2": float(c2)})

    best = min(prof, key=lambda x: (x["chi2"] if math.isfinite(x["chi2"]) else float("inf")))
    chi2_min = float(best["chi2"])
    s_best = float(best["s_R"])

    # 3) 1σ interval in s_R (profile likelihood; Δχ2=1 for one parameter).
    target = chi2_min + 1.0
    s_prof = np.array([float(r["s_R"]) for r in prof], dtype=float)
    chi_prof = np.array([float(r["chi2"]) for r in prof], dtype=float)
    # 条件分岐: `not np.any(np.isfinite(chi_prof))` を満たす経路を評価する。
    if not np.any(np.isfinite(chi_prof)):
        s_sig = float("nan")
        s_sig_asym = {"minus": float("nan"), "plus": float("nan")}
    else:
        i_best = int(np.nanargmin(chi_prof))
        ok = np.isfinite(chi_prof) & (chi_prof <= target)

        # Walk left/right from the best-fit index until leaving the ok-region.
        i_left = i_best
        while i_left > 0 and bool(ok[i_left]):
            i_left -= 1

        left = float(s_prof[i_left + 1]) if not bool(ok[i_left]) else float(s_prof[0])

        i_right = i_best
        while i_right < len(s_prof) - 1 and bool(ok[i_right]):
            i_right += 1

        right = float(s_prof[i_right - 1]) if not bool(ok[i_right]) else float(s_prof[-1])

        # If the interval hits the scan boundary, treat as undetermined in this resolution.
        if left == float(s_prof[0]) or right == float(s_prof[-1]):
            s_sig = float("nan")
            s_sig_asym = {"minus": float("nan"), "plus": float("nan")}
        else:
            s_sig_asym = {"minus": float(s_best - left), "plus": float(right - s_best)}
            s_sig = float(max(s_sig_asym["minus"], s_sig_asym["plus"]))

    dof = 2 * len(rows) - 2  # (DM,H) per z minus (s_R,B)
    return {
        "profile": prof,
        "best_fit": {
            "s_R": s_best,
            "s_R_sigma_1d": s_sig,
            "s_R_sigma_1d_asym": s_sig_asym,
            "B_best_km_s_mpc": float(best["B_best"]),
            "chi2": chi2_min,
            "dof": int(dof),
        },
    }


# 関数: `_plot_bao_fit` の入出力契約と処理意図を定義する。

def _plot_bao_fit(
    bao_rows: Sequence[BAOAnisotropyConstraint],
    *,
    out_png: Path,
    fit: Dict[str, Any],
    s_R_required: float,
    s_R_required_sigma: float,
) -> None:
    _set_japanese_font()
    import matplotlib.pyplot as plt

    z = np.array([r.z_eff for r in bao_rows], dtype=float)
    dm = np.array([r.DM_scaled_mpc for r in bao_rows], dtype=float)
    dm_sig = np.array([r.DM_scaled_sigma_mpc for r in bao_rows], dtype=float)
    h = np.array([r.H_scaled_km_s_mpc for r in bao_rows], dtype=float)
    h_sig = np.array([r.H_scaled_sigma_km_s_mpc for r in bao_rows], dtype=float)

    best = dict(fit.get("best_fit") or {})
    s_best = float(best.get("s_R", float("nan")))
    B_best = float(best.get("B_best_km_s_mpc", float("nan")))

    # Fit B at s_R_required for a fair curve comparison.
    B_req, chi2_req = _fit_bao_B_for_sR(bao_rows, s_R=float(s_R_required))

    z_curve = np.linspace(0.0, float(np.max(z)) + 0.1, 500, dtype=float)
    dm_best = np.array([_bao_pred_dm_h(z=float(zz), s_R=s_best, B=B_best)[0] for zz in z_curve], dtype=float)
    h_best = np.array([_bao_pred_dm_h(z=float(zz), s_R=s_best, B=B_best)[1] for zz in z_curve], dtype=float)
    dm_req = np.array([_bao_pred_dm_h(z=float(zz), s_R=float(s_R_required), B=float(B_req))[0] for zz in z_curve], dtype=float)
    h_req = np.array([_bao_pred_dm_h(z=float(zz), s_R=float(s_R_required), B=float(B_req))[1] for zz in z_curve], dtype=float)

    prof = fit.get("profile") or []
    s_prof = np.array([float(r["s_R"]) for r in prof], dtype=float)
    chi_prof = np.array([float(r["chi2"]) for r in prof], dtype=float)
    chi_min = float(best.get("chi2", np.nan))
    dchi = chi_prof - chi_min

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(19.0, 6.6))

    ax1.errorbar(z, dm, yerr=dm_sig, fmt="o", capsize=4, color="#111111", ecolor="#111111", label="観測（BOSS DR12）")
    ax1.plot(z_curve, dm_best, color="#1f77b4", linewidth=2.0, label=f"best-fit: s_R={s_best:+.3f}")
    ax1.plot(z_curve, dm_req, color="#d62728", linewidth=1.8, linestyle="--", label=f"DDR必要: s_R={s_R_required:+.3f}")
    ax1.set_title("BAO: D_M×(r_d,fid/r_d)", fontsize=13)
    ax1.set_xlabel("赤方偏移 z", fontsize=11)
    ax1.set_ylabel("D_M scaled [Mpc]", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9)

    ax2.errorbar(z, h, yerr=h_sig, fmt="o", capsize=4, color="#111111", ecolor="#111111", label="観測（BOSS DR12）")
    ax2.plot(z_curve, h_best, color="#1f77b4", linewidth=2.0, label=f"best-fit: s_R={s_best:+.3f}")
    ax2.plot(z_curve, h_req, color="#d62728", linewidth=1.8, linestyle="--", label=f"DDR必要: s_R={s_R_required:+.3f}")
    ax2.set_title("BAO: H×(r_d/r_d,fid)", fontsize=13)
    ax2.set_xlabel("赤方偏移 z", fontsize=11)
    ax2.set_ylabel("H scaled [km/s/Mpc]", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9)

    ax3.plot(s_prof, dchi, color="#1f77b4", linewidth=2.0)
    ax3.axhline(1.0, color="#999999", linewidth=1.0, linestyle="--", alpha=0.9, label="Δχ2=1（1σ, 1 param）")
    ax3.axvline(s_best, color="#1f77b4", linewidth=1.2, alpha=0.9)
    ax3.axvline(float(s_R_required), color="#d62728", linewidth=1.2, alpha=0.9)
    ax3.set_title("BAO（D_M,H）からの s_R 制約", fontsize=13)
    ax3.set_xlabel("s_R", fontsize=11)
    ax3.set_ylabel("Δχ2（profile）", fontsize=11)
    ax3.set_ylim(bottom=-0.2, top=max(5.0, float(np.nanmax(dchi)) + 0.5))
    ax3.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("宇宙論（再接続の整合性チェック）：BAO（BOSS DR12）での標準定規進化 s_R", fontsize=14)
    fig.text(
        0.5,
        0.01,
        (
            f"best-fit: s_R={s_best:+.3f} (1σ≈{_fmt_float(best.get('s_R_sigma_1d'), digits=3)}), "
            f"B={_fmt_float(B_best, digits=3)} / DDR必要: s_R={s_R_required:+.3f}±{_fmt_float(s_R_required_sigma, digits=3)} "
            f"(B@req={_fmt_float(B_req, digits=3)}, χ2@req={_fmt_float(chi2_req, digits=3)})"
        ),
        ha="center",
        fontsize=10,
    )
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.93))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# 関数: `_compute_rows_out` の入出力契約と処理意図を定義する。

def _compute_rows_out(
    ddr_rows: Sequence[DDRConstraint],
    *,
    primary_op: OpacityConstraint,
    primary_candle: CandleEvoConstraint,
    p_e: float,
    p_e_sigma: float,
    p_t: float,
    p_t_sigma: float,
    td_primary: Optional[TimeDilationConstraint],
    tz_primary: Optional[CMBTemperatureConstraint],
) -> List[Dict[str, Any]]:
    rows_out: List[Dict[str, Any]] = []
    for ddr in ddr_rows:
        s_r, s_sig = _required_s_r(
            eps0=float(ddr.epsilon0),
            eps0_sigma=float(ddr.epsilon0_sigma),
            alpha=float(primary_op.alpha_opacity),
            alpha_sigma=float(primary_op.alpha_opacity_sigma),
            s_L=float(primary_candle.s_L),
            s_L_sigma=float(primary_candle.s_L_sigma),
            p_e=p_e,
            p_e_sigma=p_e_sigma,
            p_t=p_t,
            p_t_sigma=p_t_sigma,
        )

        z_no_evo = None
        # 条件分岐: `s_sig > 0 and math.isfinite(s_sig)` を満たす経路を評価する。
        if s_sig > 0 and math.isfinite(s_sig):
            z_no_evo = (s_r - 0.0) / s_sig

        ruler_factor_z1 = _lognormal_quantiles_from_normal(mu=s_r, sig=s_sig, a=2.0)
        rows_out.append(
            {
                "id": ddr.id,
                "short_label": ddr.short_label,
                "uses_bao": bool(ddr.uses_bao),
                "epsilon0_obs": float(ddr.epsilon0),
                "epsilon0_sigma": float(ddr.epsilon0_sigma),
                "epsilon0_sigma_raw": float(ddr.epsilon0_sigma_raw),
                "sigma_sys_category": ddr.sigma_sys_category,
                "sigma_policy": ddr.sigma_policy,
                "category": ddr.category,
                "alpha_used": float(primary_op.alpha_opacity),
                "alpha_sigma_used": float(primary_op.alpha_opacity_sigma),
                "s_L_used": float(primary_candle.s_L),
                "s_L_sigma_used": float(primary_candle.s_L_sigma),
                "assumptions": {
                    "p_e": float(p_e),
                    "p_e_sigma": float(p_e_sigma),
                    "p_t": float(p_t),
                    "p_t_sigma": float(p_t_sigma),
                },
                "s_R_required": float(s_r),
                "s_R_sigma": float(s_sig),
                "z_score_no_ruler_evolution": (None if z_no_evo is None else float(z_no_evo)),
                "ruler_factor_z1": ruler_factor_z1,
                "sources": {
                    "ddr": ddr.source,
                    "opacity": primary_op.source,
                    "candle_evolution": primary_candle.source,
                    **(
                        {}
                        if td_primary is None
                        else {
                            "sn_time_dilation": td_primary.source,
                            "cmb_temperature": tz_primary.source if tz_primary is not None else {},
                        }
                    ),
                },
                "sigma_note": {
                    "ddr": ddr.sigma_note,
                    "opacity": primary_op.sigma_note,
                    "candle": primary_candle.sigma_note,
                    **(
                        {}
                        if td_primary is None
                        else {
                            "sn_time_dilation": td_primary.sigma_note,
                            "cmb_temperature": tz_primary.sigma_note if tz_primary is not None else "",
                        }
                    ),
                },
            }
        )

    return rows_out


# 関数: `_write_variant_outputs` の入出力契約と処理意図を定義する。

def _write_variant_outputs(
    *,
    variant_suffix: str,
    rows_out: List[Dict[str, Any]],
    out_dir: Path,
    ddr_path: Path,
    opacity_path: Path,
    candle_path: Path,
    bao_path: Path,
    sn_td_path: Path,
    cmb_t_path: Path,
    ddr_sigma_policy_meta: Dict[str, Any],
    primary_op: OpacityConstraint,
    primary_candle: CandleEvoConstraint,
    primary_op_meta: Dict[str, Any],
    primary_candle_meta: Dict[str, Any],
    p_e: float,
    p_e_sigma: float,
    p_t: float,
    p_t_sigma: float,
    td_primary: Optional[TimeDilationConstraint],
    tz_primary: Optional[CMBTemperatureConstraint],
    bao_rows: Sequence[BAOAnisotropyConstraint],
    bao_fit: Dict[str, Any],
    s_R_bao: float,
    s_R_bao_sig: float,
) -> Dict[str, Any]:
    suffix = str(variant_suffix)
    out_png = out_dir / f"cosmology_reconnection_required_ruler_evolution{suffix}.png"
    ddr_sigma_note = ""
    # 条件分岐: `str(ddr_sigma_policy_meta.get("policy") or "") == "category_sys"` を満たす経路を評価する。
    if str(ddr_sigma_policy_meta.get("policy") or "") == "category_sys":
        ddr_sigma_note = "DDR σ: σ_total=√(σ_obs^2+σ_cat^2) を使用"
    else:
        ddr_sigma_note = "DDR σ: 観測σ（raw）を使用"

    _plot(
        rows_out,
        out_png=out_png,
        primary_alpha=primary_op,
        primary_candle=primary_candle,
        p_e=p_e,
        p_e_sigma=p_e_sigma,
        p_t=p_t,
        p_t_sigma=p_t_sigma,
        ddr_sigma_note=ddr_sigma_note,
    )

    ddr_primary_row_all = min(
        rows_out,
        key=lambda r: (float(r.get("epsilon0_sigma") or 0.0) if float(r.get("epsilon0_sigma") or 0.0) > 0 else float("inf")),
    )
    ddr_rows_no_bao = [r for r in rows_out if not bool(r.get("uses_bao", False))]
    ddr_primary_row_no_bao = (
        min(
            ddr_rows_no_bao,
            key=lambda r: (
                float(r.get("epsilon0_sigma") or 0.0) if float(r.get("epsilon0_sigma") or 0.0) > 0 else float("inf")
            ),
        )
        if ddr_rows_no_bao
        else ddr_primary_row_all
    )

    # For the BAO self-consistency check, prefer a DDR constraint that does NOT use BAO-derived distances
    # to reduce circularity.
    ddr_primary_row = ddr_primary_row_no_bao
    s_R_req = float(ddr_primary_row["s_R_required"])
    s_R_req_sig = float(ddr_primary_row["s_R_sigma"])

    # Compare DDR-implied s_R against BAO-preferred s_R for each DDR constraint (and track the best match).
    z_sR_req_vs_bao = None
    best_match_any: Optional[Dict[str, Any]] = None
    best_match_no_bao: Optional[Dict[str, Any]] = None
    # 条件分岐: `math.isfinite(s_R_bao) and math.isfinite(s_R_bao_sig) and s_R_bao_sig > 0` を満たす経路を評価する。
    if math.isfinite(s_R_bao) and math.isfinite(s_R_bao_sig) and s_R_bao_sig > 0:
        for r in rows_out:
            s_r = _safe_float(r.get("s_R_required"))
            s_sig = _safe_float(r.get("s_R_sigma"))
            # 条件分岐: `s_r is None or s_sig is None or not (math.isfinite(s_r) and math.isfinite(s_s...` を満たす経路を評価する。
            if s_r is None or s_sig is None or not (math.isfinite(s_r) and math.isfinite(s_sig)) or s_sig <= 0:
                continue

            denom = math.sqrt(s_sig**2 + s_R_bao_sig**2)
            # 条件分岐: `not (denom > 0)` を満たす経路を評価する。
            if not (denom > 0):
                continue

            z = (float(s_r) - s_R_bao) / denom
            r["z_score_sR_required_vs_bao"] = float(z)
            # 条件分岐: `best_match_any is None or abs(z) < abs(float(best_match_any.get("z_score") or...` を満たす経路を評価する。
            if best_match_any is None or abs(z) < abs(float(best_match_any.get("z_score") or float("inf"))):
                best_match_any = {
                    "id": str(r.get("id") or ""),
                    "short_label": str(r.get("short_label") or ""),
                    "uses_bao": bool(r.get("uses_bao", False)),
                    "s_R_required": float(s_r),
                    "s_R_sigma": float(s_sig),
                    "z_score": float(z),
                }

            # 条件分岐: `not bool(r.get("uses_bao", False))` を満たす経路を評価する。

            if not bool(r.get("uses_bao", False)):
                # 条件分岐: `best_match_no_bao is None or abs(z) < abs(float(best_match_no_bao.get("z_scor...` を満たす経路を評価する。
                if best_match_no_bao is None or abs(z) < abs(float(best_match_no_bao.get("z_score") or float("inf"))):
                    best_match_no_bao = {
                        "id": str(r.get("id") or ""),
                        "short_label": str(r.get("short_label") or ""),
                        "uses_bao": False,
                        "s_R_required": float(s_r),
                        "s_R_sigma": float(s_sig),
                        "z_score": float(z),
                    }

    # 条件分岐: `math.isfinite(s_R_req) and math.isfinite(s_R_req_sig) and math.isfinite(s_R_b...` を満たす経路を評価する。

    if math.isfinite(s_R_req) and math.isfinite(s_R_req_sig) and math.isfinite(s_R_bao) and math.isfinite(s_R_bao_sig):
        denom = math.sqrt(s_R_req_sig**2 + s_R_bao_sig**2)
        # 条件分岐: `denom > 0` を満たす経路を評価する。
        if denom > 0:
            z_sR_req_vs_bao = (s_R_req - s_R_bao) / denom

    out_bao_png = out_dir / f"cosmology_bao_scaled_distance_fit{suffix}.png"
    _plot_bao_fit(
        bao_rows,
        out_png=out_bao_png,
        fit=bao_fit,
        s_R_required=s_R_req,
        s_R_required_sigma=s_R_req_sig,
    )

    out_bao_json = out_dir / f"cosmology_bao_scaled_distance_fit{suffix}_metrics.json"
    bao_metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "variant_suffix": suffix,
        "ddr_sigma_policy": dict(ddr_sigma_policy_meta),
        "inputs": {"bao": str(bao_path)},
        "definition": {
            "data_products": "BOSS DR12 consensus: (D_M×(r_d,fid/r_d), H×(r_d/r_d,fid)) at z_eff",
            "model": {
                "geometry": "static background-P (exponential): D_M=(c/H0)ln(1+z), H_eff=H0(1+z)",
                "ruler_evolution": "r_d(z)=r_d0 (1+z)^{s_R} (effective; for DM/H scaling only)",
                "scale_param": "B ≡ H0 * (r_d0/r_d,fid) [km/s/Mpc] (absorbs unknown overall calibration)",
            },
        },
        "fit": bao_fit,
        "ddr_primary_used": {
            "id": str(ddr_primary_row.get("id") or ""),
            "short_label": str(ddr_primary_row.get("short_label") or ""),
            "uses_bao": bool(ddr_primary_row.get("uses_bao", False)),
            "epsilon0_obs": float(ddr_primary_row.get("epsilon0_obs") or float("nan")),
            "epsilon0_sigma": float(ddr_primary_row.get("epsilon0_sigma") or float("nan")),
            "epsilon0_sigma_raw": float(ddr_primary_row.get("epsilon0_sigma_raw") or float("nan")),
            "sigma_sys_category": ddr_primary_row.get("sigma_sys_category"),
            "sigma_policy": str(ddr_primary_row.get("sigma_policy") or ""),
            "category": ddr_primary_row.get("category"),
            "s_R_required": s_R_req,
            "s_R_sigma": s_R_req_sig,
        },
        "ddr_primary_used_all": {
            "id": str(ddr_primary_row_all.get("id") or ""),
            "short_label": str(ddr_primary_row_all.get("short_label") or ""),
            "uses_bao": bool(ddr_primary_row_all.get("uses_bao", False)),
            "epsilon0_obs": float(ddr_primary_row_all.get("epsilon0_obs") or float("nan")),
            "epsilon0_sigma": float(ddr_primary_row_all.get("epsilon0_sigma") or float("nan")),
            "epsilon0_sigma_raw": float(ddr_primary_row_all.get("epsilon0_sigma_raw") or float("nan")),
            "sigma_sys_category": ddr_primary_row_all.get("sigma_sys_category"),
            "sigma_policy": str(ddr_primary_row_all.get("sigma_policy") or ""),
            "category": ddr_primary_row_all.get("category"),
            "s_R_required": float(ddr_primary_row_all.get("s_R_required") or float("nan")),
            "s_R_sigma": float(ddr_primary_row_all.get("s_R_sigma") or float("nan")),
        },
        "comparison": {
            "z_score_sR_required_vs_bao": z_sR_req_vs_bao,
            "best_match_any_ddr": best_match_any,
            "best_match_no_bao_ddr": best_match_no_bao,
            "per_ddr": [
                {
                    "id": str(r.get("id") or ""),
                    "short_label": str(r.get("short_label") or ""),
                    "uses_bao": bool(r.get("uses_bao", False)),
                    "epsilon0_obs": float(r.get("epsilon0_obs") or float("nan")),
                    "epsilon0_sigma": float(r.get("epsilon0_sigma") or float("nan")),
                    "epsilon0_sigma_raw": float(r.get("epsilon0_sigma_raw") or float("nan")),
                    "sigma_sys_category": r.get("sigma_sys_category"),
                    "sigma_policy": str(r.get("sigma_policy") or ""),
                    "category": r.get("category"),
                    "s_R_required": float(r.get("s_R_required") or float("nan")),
                    "s_R_sigma": float(r.get("s_R_sigma") or float("nan")),
                    "z_score_sR_required_vs_bao": r.get("z_score_sR_required_vs_bao"),
                }
                for r in rows_out
            ],
            "note": "This compares the DDR-implied s_R (from distance-duality reconnection) against the BAO(DM,H)-preferred s_R under the static geometry model.",
        },
        "outputs": {"png": str(out_bao_png), "metrics_json": str(out_bao_json)},
        "notes": [
            "このチェックは『s_R を“BAOの標準定規進化”として解釈した場合に、BOSS DR12 の (D_M,H) 出力と整合するか』の目安。",
            "DDRの ε0 制約が BAO 距離に依存する場合、この比較は厳密な独立検定ではない（自己整合/張力の可視化）。",
        ],
    }
    _write_json(out_bao_json, bao_metrics)

    out_json = out_dir / f"cosmology_reconnection_required_ruler_evolution{suffix}_metrics.json"
    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "variant_suffix": suffix,
        "ddr_sigma_policy": dict(ddr_sigma_policy_meta),
        "inputs": {
            "ddr": str(ddr_path),
            "opacity": str(opacity_path),
            "candle_evolution": str(candle_path),
            **({} if td_primary is None else {"sn_time_dilation": str(sn_td_path), "cmb_temperature": str(cmb_t_path)}),
        },
        "definition": {
            "epsilon0_model": "ε0_model = (p_e + p_t - s_L)/2 - 2 + s_R + α",
            "solve_for_s_R": "s_R = ε0_obs - α - (p_e+p_t - s_L)/2 + 2",
            "ruler_factor_z1": "(1+z)^{s_R} at z=1 => 2^{s_R}（中央値と±1σ）",
        },
        "assumptions": {
            "p_e": p_e,
            "p_e_sigma": p_e_sigma,
            "p_t": p_t,
            "p_t_sigma": p_t_sigma,
            "independence": "ε0, α, s_L, p_e, p_t are treated as independent (Gaussian) for error propagation.",
            "mapping_note": (
                "When --use-independent-probes is enabled: "
                "p_t from SN time dilation (Blondin+2008), "
                "p_e from CMB temperature scaling p_T=1-β_T (Avgoustidis+2011)."
            ),
        },
        "primary_constraints_used": {
            "opacity": {"id": primary_op.id, "short_label": primary_op.short_label, **_primary_flags(primary_op)},
            "candle_evolution": {
                "id": primary_candle.id,
                "short_label": primary_candle.short_label,
                **_primary_flags(primary_candle),
            },
            "selection_policy": {"opacity": dict(primary_op_meta), "candle_evolution": dict(primary_candle_meta)},
            **(
                {}
                if td_primary is None
                else {
                    "sn_time_dilation": {"id": td_primary.id, "short_label": td_primary.short_label},
                    "cmb_temperature": {
                        "id": tz_primary.id if tz_primary is not None else "",
                        "short_label": tz_primary.short_label if tz_primary is not None else "",
                    },
                }
            ),
        },
        "rows": rows_out,
        "outputs": {
            "png": str(out_png),
            "metrics_json": str(out_json),
            "bao_fit_png": str(out_bao_png),
            "bao_fit_metrics_json": str(out_bao_json),
        },
        "notes": [
            "ここでの s_R は『静的背景P（膨張なし）』で現行の距離指標（SNIa/BAO）とDDRを両立させるために必要となる“有効な標準定規進化”の目安。",
            "BAOを“標準定規”として厳密に扱うには、定義・校正（r_d等）をP-model側で再導出し、一次ソース（解析パイプライン）と整合する形で評価し直す必要がある。",
        ],
    }
    _write_json(out_json, metrics)

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")
    print(f"[ok] bao : {out_bao_png}")
    print(f"[ok] bao : {out_bao_json}")

    return {
        "variant_suffix": suffix,
        "primary_opacity_id": primary_op.id,
        "primary_candle_id": primary_candle.id,
        "reconnection_png": out_png,
        "reconnection_metrics_json": out_json,
        "bao_png": out_bao_png,
        "bao_metrics_json": out_bao_json,
        "bao_fit": {
            "s_R_best": s_R_bao,
            "s_R_sigma_1d": s_R_bao_sig,
            "s_R_required": s_R_req,
            "s_R_required_sigma": s_R_req_sig,
            "z_score_required_vs_bao": z_sR_req_vs_bao,
        },
    }

# 関数: `_plot` の入出力契約と処理意図を定義する。

def _plot(
    rows: Sequence[Dict[str, Any]],
    *,
    out_png: Path,
    primary_alpha: OpacityConstraint,
    primary_candle: CandleEvoConstraint,
    p_e: float,
    p_e_sigma: float,
    p_t: float,
    p_t_sigma: float,
    ddr_sigma_note: str = "",
) -> None:
    _set_japanese_font()
    import matplotlib.pyplot as plt

    labels = [str(r.get("short_label") or r.get("id") or "") for r in rows]
    y = np.arange(len(rows), dtype=float)

    s_r = np.array([float(r["s_R_required"]) for r in rows], dtype=float)
    s_r_sig = np.array([float(r["s_R_sigma"]) for r in rows], dtype=float)

    fac_med = np.array([float(r["ruler_factor_z1"]["median"]) for r in rows], dtype=float)
    fac_p16 = np.array([float(r["ruler_factor_z1"]["p16"]) for r in rows], dtype=float)
    fac_p84 = np.array([float(r["ruler_factor_z1"]["p84"]) for r in rows], dtype=float)
    fac_xerr = np.vstack([fac_med - fac_p16, fac_p84 - fac_med])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18.0, 6.8))

    # Panel 1: required s_R
    ax1.axvline(0.0, color="#333333", linewidth=1.1, alpha=0.9, label="定規進化なし: s_R=0")
    ax1.axvline(1.0, color="#666666", linewidth=1.0, alpha=0.6, linestyle="--", label="z=1で“2倍”: s_R=1")
    ax1.errorbar(
        s_r,
        y,
        xerr=np.where(np.isfinite(s_r_sig) & (s_r_sig > 0), s_r_sig, 0.0),
        fmt="o",
        capsize=4,
        color="#1f77b4",
        ecolor="#1f77b4",
        label="必要な s_R（1σ, 誤差伝播）",
    )
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.invert_yaxis()
    ax1.set_xlabel("s_R（標準定規の有効進化: l_em = l0 (1+z)^(s_R)）", fontsize=11)
    ax1.set_title("DDRを満たすために必要な“標準定規進化”", fontsize=13)
    ax1.grid(True, linestyle="--", alpha=0.5, axis="x")
    ax1.legend(fontsize=9, loc="upper left")

    # Panel 2: implied factor at z=1
    ax2.axvline(1.0, color="#333333", linewidth=1.1, alpha=0.9, label="変化なし: 倍率=1")
    ax2.axvline(2.0, color="#666666", linewidth=1.0, alpha=0.6, linestyle="--", label="2倍（参考）")
    ax2.errorbar(
        fac_med,
        y,
        xerr=fac_xerr,
        fmt="o",
        capsize=4,
        color="#2ca02c",
        ecolor="#2ca02c",
        label="z=1での倍率（中央値と±1σ）",
    )
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels)
    ax2.invert_yaxis()
    ax2.set_xscale("log")
    ax2.set_xlabel("標準定規の倍率（z=1, (1+z)^{s_R} = 2^{s_R}）", fontsize=11)
    ax2.set_title("直感量（z=1で何倍の定規が必要か）", fontsize=13)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="x")
    ax2.legend(fontsize=9, loc="upper left")

    fig.suptitle("宇宙論（再接続）：一次ソース拘束（α, s_L）込みでDDRを満たす s_R の必要量", fontsize=14)
    fig.text(
        0.5,
        0.01,
        (
            f"前提: p_e={p_e:g}±{_fmt_float(p_e_sigma, digits=3)}, p_t={p_t:g}±{_fmt_float(p_t_sigma, digits=3)}"
            f" / α={primary_alpha.alpha_opacity:+.3f}±{primary_alpha.alpha_opacity_sigma:.3f}, s_L={primary_candle.s_L:+.3f}±{primary_candle.s_L_sigma:.3f}"
            f"{'（CDDR仮定）' if (primary_candle.assumes_cddr is True) else ''}"
            + (f" / {ddr_sigma_note}" if str(ddr_sigma_note or "").strip() else "")
        ),
        ha="center",
        fontsize=10,
    )
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.93))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Cosmology: required ruler evolution s_R to reconcile DDR with opacity + SN evolution constraints.",
    )
    ap.add_argument(
        "--ddr-data",
        type=str,
        default=str(_ROOT / "data" / "cosmology" / "distance_duality_constraints.json"),
        help="Input DDR constraints JSON (default: data/cosmology/distance_duality_constraints.json)",
    )
    ap.add_argument(
        "--opacity-data",
        type=str,
        default=str(_ROOT / "data" / "cosmology" / "cosmic_opacity_constraints.json"),
        help="Input opacity constraints JSON (default: data/cosmology/cosmic_opacity_constraints.json)",
    )
    ap.add_argument(
        "--candle-data",
        type=str,
        default=str(_ROOT / "data" / "cosmology" / "sn_standard_candle_evolution_constraints.json"),
        help="Input SN evolution constraints JSON (default: data/cosmology/sn_standard_candle_evolution_constraints.json)",
    )
    ap.add_argument(
        "--bao-data",
        type=str,
        default=str(_ROOT / "data" / "cosmology" / "alcock_paczynski_constraints.json"),
        help="Input BAO anisotropy (DM,H) constraints JSON (default: data/cosmology/alcock_paczynski_constraints.json)",
    )
    ap.add_argument(
        "--sn-time-dilation-data",
        type=str,
        default=str(_ROOT / "data" / "cosmology" / "sn_time_dilation_constraints.json"),
        help="Input SN time dilation constraints JSON (default: data/cosmology/sn_time_dilation_constraints.json)",
    )
    ap.add_argument(
        "--cmb-temperature-data",
        type=str,
        default=str(_ROOT / "data" / "cosmology" / "cmb_temperature_scaling_constraints.json"),
        help="Input CMB T(z) constraints JSON (default: data/cosmology/cmb_temperature_scaling_constraints.json)",
    )
    ap.add_argument(
        "--use-independent-probes",
        action="store_true",
        help="Use independent probes (SN time dilation, CMB T(z)) to set p_t and p_e with uncertainties.",
    )
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
    ap.add_argument("--p-e", type=float, default=1.0, help="Photon energy exponent p_e (default: 1.0)")
    ap.add_argument("--p-e-sigma", type=float, default=0.0, help="Uncertainty of p_e for propagation (default: 0)")
    ap.add_argument("--p-t", type=float, default=1.0, help="Time dilation exponent p_t (default: 1.0)")
    ap.add_argument("--p-t-sigma", type=float, default=0.0, help="Uncertainty of p_t for propagation (default: 0)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    p_e = float(args.p_e)
    p_t = float(args.p_t)
    p_e_sigma = float(args.p_e_sigma)
    p_t_sigma = float(args.p_t_sigma)

    ddr_path = Path(args.ddr_data)
    opacity_path = Path(args.opacity_data)
    candle_path = Path(args.candle_data)
    bao_path = Path(args.bao_data)
    sn_td_path = Path(args.sn_time_dilation_data)
    cmb_t_path = Path(args.cmb_temperature_data)

    ddr_raw = _read_json(ddr_path)
    opacity_raw = _read_json(opacity_path)
    candle_raw = _read_json(candle_path)
    bao_raw = _read_json(bao_path)

    td_primary: Optional[TimeDilationConstraint] = None
    tz_primary: Optional[CMBTemperatureConstraint] = None
    # 条件分岐: `bool(args.use_independent_probes)` を満たす経路を評価する。
    if bool(args.use_independent_probes):
        td_raw = _read_json(sn_td_path)
        tz_raw = _read_json(cmb_t_path)
        td_rows = [TimeDilationConstraint.from_json(x) for x in (td_raw.get("constraints") or []) if isinstance(x, dict)]
        tz_rows = [CMBTemperatureConstraint.from_json(x) for x in (tz_raw.get("constraints") or []) if isinstance(x, dict)]
        # 条件分岐: `not td_rows` を満たす経路を評価する。
        if not td_rows:
            raise ValueError("No SN time dilation constraints in input (use-independent-probes)")

        # 条件分岐: `not tz_rows` を満たす経路を評価する。

        if not tz_rows:
            raise ValueError("No CMB temperature constraints in input (use-independent-probes)")

        td_primary = _primary_by_sigma(td_rows, "p_t_sigma")
        tz_primary = _primary_by_sigma(tz_rows, "beta_T_sigma")

        # Assumption: photon energy exponent p_e follows CMB temperature exponent p_T=1-β_T.
        p_t = float(td_primary.p_t)
        p_t_sigma = float(td_primary.p_t_sigma)
        p_e = float(tz_primary.p_T)
        p_e_sigma = float(tz_primary.p_T_sigma)

    ddr_rows = [DDRConstraint.from_json(x) for x in (ddr_raw.get("constraints") or []) if isinstance(x, dict)]
    ddr_sigma_policy = str(args.ddr_sigma_policy)
    ddr_env_path = _ROOT / "output" / "private" / "cosmology" / "cosmology_distance_duality_systematics_envelope_metrics.json"
    ddr_env = _load_ddr_systematics_envelope(ddr_env_path) if ddr_sigma_policy == "category_sys" else {}
    ddr_rows = [_apply_ddr_sigma_policy(d, policy=ddr_sigma_policy, envelope=ddr_env) for d in ddr_rows]
    applied_ddr_sigma_count = len([d for d in ddr_rows if d.sigma_policy == "category_sys"])
    ddr_sigma_policy_meta = {
        "policy": ddr_sigma_policy,
        "envelope_metrics": (str(ddr_env_path).replace("\\", "/") if ddr_sigma_policy == "category_sys" else None),
        "applied_count": applied_ddr_sigma_count,
        "note": "If the envelope file is missing, σ_cat inflation is skipped for all rows (falls back to raw).",
    }
    op_rows = [OpacityConstraint.from_json(x) for x in (opacity_raw.get("constraints") or []) if isinstance(x, dict)]
    candle_rows = [CandleEvoConstraint.from_json(x) for x in (candle_raw.get("constraints") or []) if isinstance(x, dict)]
    bao_rows = [BAOAnisotropyConstraint.from_json(x) for x in (bao_raw.get("constraints") or []) if isinstance(x, dict)]
    # 条件分岐: `not ddr_rows` を満たす経路を評価する。
    if not ddr_rows:
        raise ValueError("No DDR constraints in input")

    # 条件分岐: `not op_rows` を満たす経路を評価する。

    if not op_rows:
        raise ValueError("No opacity constraints in input")

    # 条件分岐: `not candle_rows` を満たす経路を評価する。

    if not candle_rows:
        raise ValueError("No SN evolution constraints in input")

    # 条件分岐: `not bao_rows` を満たす経路を評価する。

    if not bao_rows:
        raise ValueError("No BAO constraints in input")

    bao_rows = sorted(bao_rows, key=lambda r: r.z_eff)

    out_dir = _ROOT / "output" / "private" / "cosmology"
    out_dir.mkdir(parents=True, exist_ok=True)

    bao_fit = _fit_bao_ruler_evolution(bao_rows)
    bao_best = dict(bao_fit.get("best_fit") or {})
    s_R_bao = float(bao_best.get("s_R", float("nan")))
    s_R_bao_sig = float(bao_best.get("s_R_sigma_1d", float("nan")))

    # Two variants:
    #   (A) full: choose the tightest α and s_L constraints by σ (may include BAO/CMB)
    #   (B) independent: choose α and s_L from sources that explicitly do NOT use BAO nor CMB
    primary_op_full, primary_op_full_meta = _select_primary_opacity(op_rows, independent_only=False)
    primary_candle_full, primary_candle_full_meta = _select_primary_candle(candle_rows, independent_only=False)
    rows_full = _compute_rows_out(
        ddr_rows,
        primary_op=primary_op_full,
        primary_candle=primary_candle_full,
        p_e=p_e,
        p_e_sigma=p_e_sigma,
        p_t=p_t,
        p_t_sigma=p_t_sigma,
        td_primary=td_primary,
        tz_primary=tz_primary,
    )
    full_result = _write_variant_outputs(
        variant_suffix="",
        rows_out=rows_full,
        out_dir=out_dir,
        ddr_path=ddr_path,
        opacity_path=opacity_path,
        candle_path=candle_path,
        bao_path=bao_path,
        sn_td_path=sn_td_path,
        cmb_t_path=cmb_t_path,
        ddr_sigma_policy_meta=dict(ddr_sigma_policy_meta),
        primary_op=primary_op_full,
        primary_candle=primary_candle_full,
        primary_op_meta=primary_op_full_meta,
        primary_candle_meta=primary_candle_full_meta,
        p_e=p_e,
        p_e_sigma=p_e_sigma,
        p_t=p_t,
        p_t_sigma=p_t_sigma,
        td_primary=td_primary,
        tz_primary=tz_primary,
        bao_rows=bao_rows,
        bao_fit=bao_fit,
        s_R_bao=s_R_bao,
        s_R_bao_sig=s_R_bao_sig,
    )

    primary_op_ind, primary_op_ind_meta = _select_primary_opacity(op_rows, independent_only=True)
    primary_candle_ind, primary_candle_ind_meta = _select_primary_candle(candle_rows, independent_only=True)
    rows_ind = _compute_rows_out(
        ddr_rows,
        primary_op=primary_op_ind,
        primary_candle=primary_candle_ind,
        p_e=p_e,
        p_e_sigma=p_e_sigma,
        p_t=p_t,
        p_t_sigma=p_t_sigma,
        td_primary=td_primary,
        tz_primary=tz_primary,
    )
    independent_result = _write_variant_outputs(
        variant_suffix="_independent",
        rows_out=rows_ind,
        out_dir=out_dir,
        ddr_path=ddr_path,
        opacity_path=opacity_path,
        candle_path=candle_path,
        bao_path=bao_path,
        sn_td_path=sn_td_path,
        cmb_t_path=cmb_t_path,
        ddr_sigma_policy_meta=dict(ddr_sigma_policy_meta),
        primary_op=primary_op_ind,
        primary_candle=primary_candle_ind,
        primary_op_meta=primary_op_ind_meta,
        primary_candle_meta=primary_candle_ind_meta,
        p_e=p_e,
        p_e_sigma=p_e_sigma,
        p_t=p_t,
        p_t_sigma=p_t_sigma,
        td_primary=td_primary,
        tz_primary=tz_primary,
        bao_rows=bao_rows,
        bao_fit=bao_fit,
        s_R_bao=s_R_bao,
        s_R_bao_sig=s_R_bao_sig,
    )

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_reconnection_required_ruler_evolution",
                "argv": sys.argv,
                "metrics": {
                    "p_e": p_e,
                    "p_t": p_t,
                    "p_e_sigma": p_e_sigma,
                    "p_t_sigma": p_t_sigma,
                    "primary_opacity_id": full_result.get("primary_opacity_id"),
                    "primary_candle_id": full_result.get("primary_candle_id"),
                    "primary_opacity_id_independent": independent_result.get("primary_opacity_id"),
                    "primary_candle_id_independent": independent_result.get("primary_candle_id"),
                    **(
                        {}
                        if td_primary is None
                        else {
                            "primary_sn_time_dilation_id": td_primary.id,
                            "primary_cmb_temperature_id": tz_primary.id if tz_primary is not None else "",
                        }
                    ),
                    "bao_fit": full_result.get("bao_fit") or {},
                    "bao_fit_independent": independent_result.get("bao_fit") or {},
                },
                "outputs": {
                    "png": full_result.get("reconnection_png"),
                    "metrics_json": full_result.get("reconnection_metrics_json"),
                    "bao_png": full_result.get("bao_png"),
                    "bao_metrics_json": full_result.get("bao_metrics_json"),
                    "png_independent": independent_result.get("reconnection_png"),
                    "metrics_json_independent": independent_result.get("reconnection_metrics_json"),
                    "bao_png_independent": independent_result.get("bao_png"),
                    "bao_metrics_json_independent": independent_result.get("bao_metrics_json"),
                },
            }
        )
    except Exception:
        pass

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
