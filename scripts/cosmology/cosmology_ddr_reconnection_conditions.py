#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_ddr_reconnection_conditions.py

Phase 4｜宇宙論 / Step 4.7（DDR再接続の条件定量化）:
距離二重性（DDR）の観測 ε0 と、静的背景P最小モデル（ε0=-1）の差を
「距離指標の前提（標準光源/標準定規/不透明度）」の必要補正として定量化する。

モデル（静的幾何＋背景Pで赤方偏移）:
  - 標準光源（SNe Ia）の有効光度進化：      L_em = L0 (1+z)^(s_L)
  - 標準定規（BAO）の有効スケール進化：    l_em = l0 (1+z)^(s_R)
  - 灰色不透明度（光子数非保存の有効項）：  exp(-τ) with τ(z)=2α ln(1+z)
  - 時間伸長：Δt_obs=(1+z)^(p_t) Δt_em
  - 光子エネルギー：E∝(1+z)^(-p_e)

このとき DDR の破れ指数（η=(1+z)^(ε0)）は

  ε0_model = (p_e + p_t - s_L)/2 - 2 + s_R + α

したがって、距離指標側の自由度の組合せは

  s_R + α - s_L/2 = ε0_obs + 2 - (p_e + p_t)/2

を満たす必要がある（p_e=p_t=1 なら右辺は ε0_obs+1）。

本スクリプトの役割：
  - 各DDR制約（一次ソース）に対して「必要な補正量（α / s_L / s_R の等価表現）」を一覧化し、
    既存の拘束（opacity / SNe Ia evolution / BAO ratio fit）とのスケール差を固定出力する。
  - どの機構が正しいかは決めない（Step 4.7.2/4.7.3 の“前提検証”へ接続するための定量表）。

入力（固定）:
  - output/cosmology/cosmology_distance_duality_constraints_metrics.json
  - output/cosmology/cosmology_sn_time_dilation_constraints_metrics.json
  - output/cosmology/cosmology_cmb_temperature_scaling_constraints_metrics.json
  - data/cosmology/cosmic_opacity_constraints.json
  - data/cosmology/sn_standard_candle_evolution_constraints.json
  - output/cosmology/cosmology_bao_distance_ratio_fit_metrics.json

出力（固定名）:
  - output/cosmology/cosmology_ddr_reconnection_conditions.png
  - output/cosmology/cosmology_ddr_reconnection_conditions_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def _load_required(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"missing required input: {path} (run scripts/summary/run_all.py --offline first)")
    return _read_json(path)


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def _fmt(x: Optional[float], *, digits: int = 3) -> str:
    if x is None:
        return ""
    x = float(x)
    if not math.isfinite(x):
        return ""
    if x == 0.0:
        return "0"
    ax = abs(x)
    if ax >= 1e4 or ax < 1e-3:
        return f"{x:.{digits}g}"
    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


@dataclass(frozen=True)
class DDRRow:
    id: str
    short_label: str
    uses_bao: bool
    epsilon0_obs: float
    epsilon0_sigma: float
    epsilon0_pred_pbg_static: float
    delta_mu_z1_mag: Optional[float]
    flux_dimming_z1: Optional[float]
    tau_equiv_z1: Optional[float]

    @staticmethod
    def from_metrics_row(r: Dict[str, Any]) -> "DDRRow":
        return DDRRow(
            id=str(r.get("id") or ""),
            short_label=str(r.get("short_label") or r.get("id") or ""),
            uses_bao=bool(r.get("uses_bao")),
            epsilon0_obs=float(r.get("epsilon0_obs")),
            epsilon0_sigma=float(r.get("epsilon0_sigma")),
            epsilon0_pred_pbg_static=float(r.get("epsilon0_pred_pbg_static", -1.0)),
            delta_mu_z1_mag=_safe_float(r.get("delta_distance_modulus_mag_z1")),
            flux_dimming_z1=_safe_float(r.get("flux_dimming_factor_needed_z1")),
            tau_equiv_z1=_safe_float(r.get("tau_equivalent_dimming_z1")),
        )


@dataclass(frozen=True)
class ScalarConstraint:
    id: str
    short_label: str
    mu: float
    sigma: float
    uses_bao: Optional[bool] = None
    uses_cmb: Optional[bool] = None
    assumes_cddr: Optional[bool] = None
    is_forecast: Optional[bool] = None

    @staticmethod
    def from_opacity_row(r: Dict[str, Any]) -> "ScalarConstraint":
        return ScalarConstraint(
            id=str(r.get("id") or ""),
            short_label=str(r.get("short_label") or r.get("id") or ""),
            mu=float(r.get("alpha_opacity")),
            sigma=float(r.get("alpha_opacity_sigma")),
            uses_bao=(None if r.get("uses_bao") is None else bool(r.get("uses_bao"))),
            uses_cmb=(None if r.get("uses_cmb") is None else bool(r.get("uses_cmb"))),
            assumes_cddr=None,
            is_forecast=(None if r.get("is_forecast") is None else bool(r.get("is_forecast"))),
        )

    @staticmethod
    def from_sn_evo_row(r: Dict[str, Any]) -> "ScalarConstraint":
        assumes_cddr = r.get("assumes_cddr")
        return ScalarConstraint(
            id=str(r.get("id") or ""),
            short_label=str(r.get("short_label") or r.get("id") or ""),
            mu=float(r.get("s_L")),
            sigma=float(r.get("s_L_sigma")),
            uses_bao=(None if r.get("uses_bao") is None else bool(r.get("uses_bao"))),
            uses_cmb=(None if r.get("uses_cmb") is None else bool(r.get("uses_cmb"))),
            assumes_cddr=(None if assumes_cddr is None else bool(assumes_cddr)),
        )


def _envelope(constraints: List[ScalarConstraint], *, nsigma: float) -> Tuple[Optional[float], Optional[float]]:
    lows: List[float] = []
    highs: List[float] = []
    for c in constraints:
        if not (math.isfinite(c.mu) and math.isfinite(c.sigma) and c.sigma > 0.0):
            continue
        lows.append(float(c.mu - float(nsigma) * c.sigma))
        highs.append(float(c.mu + float(nsigma) * c.sigma))
    if not lows:
        return (None, None)
    return (float(min(lows)), float(max(highs)))


def _extract_pt(pe_metrics: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    rows = pe_metrics.get("rows") or []
    if not isinstance(rows, list) or not rows:
        return (None, None)
    r0 = rows[0] if isinstance(rows[0], dict) else {}
    return (_safe_float(r0.get("p_t_obs")), _safe_float(r0.get("p_t_sigma")))


def _extract_pT(cmb_metrics: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    rows = cmb_metrics.get("rows") or []
    if not isinstance(rows, list) or not rows:
        return (None, None)
    r0 = rows[0] if isinstance(rows[0], dict) else {}
    return (_safe_float(r0.get("p_T_obs")), _safe_float(r0.get("p_T_sigma")))


def _extract_sR(bao_ratio_fit_metrics: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    try:
        comb = (bao_ratio_fit_metrics.get("results") or {}).get("combined") or {}
        bf = comb.get("best_fit") or {}
        return (_safe_float(bf.get("s_R")), _safe_float(bf.get("s_R_sigma_1d")))
    except Exception:
        return (None, None)


def _extract_sR_by_subset(bao_ratio_fit_metrics: Dict[str, Any]) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    out: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for key in ("boss_only", "eboss_only", "desi_only", "combined"):
        try:
            block = (bao_ratio_fit_metrics.get("results") or {}).get(key) or {}
            bf = block.get("best_fit") or {}
            out[key] = (_safe_float(bf.get("s_R")), _safe_float(bf.get("s_R_sigma_1d")))
        except Exception:
            out[key] = (None, None)
    return out


def _classify_sigma(abs_z: Optional[float]) -> Optional[str]:
    if abs_z is None:
        return None
    try:
        az = float(abs_z)
    except Exception:
        return None
    if not math.isfinite(az):
        return None
    if az < 3.0:
        return "ok"
    if az < 5.0:
        return "mixed"
    return "ng"


def _min_abs_z(rows: List[Dict[str, Any]], *, keep: Optional[callable] = None) -> Optional[float]:
    best: Optional[float] = None
    for r in rows:
        if not isinstance(r, dict):
            continue
        if keep is not None and not bool(keep(r)):
            continue
        az = r.get("abs_z")
        if az is None:
            continue
        try:
            azf = float(az)
        except Exception:
            continue
        if not math.isfinite(azf):
            continue
        if best is None or azf < best:
            best = azf
    return best


def _env_from_gaussian(mu: Optional[float], sigma: Optional[float], *, nsigma: float) -> Tuple[Optional[float], Optional[float]]:
    if mu is None or sigma is None:
        return (None, None)
    if not (math.isfinite(mu) and math.isfinite(sigma) and sigma > 0.0):
        return (None, None)
    return (float(mu - float(nsigma) * sigma), float(mu + float(nsigma) * sigma))


def _zscore_required(required: float, *, mu: float, sigma: float) -> Optional[float]:
    if not (math.isfinite(required) and math.isfinite(mu) and math.isfinite(sigma) and sigma > 0.0):
        return None
    return float((required - mu) / sigma)


def _factor_to_delta_mu_mag(factor: Optional[float]) -> Optional[float]:
    if factor is None:
        return None
    try:
        f = float(factor)
    except Exception:
        return None
    if not (math.isfinite(f) and f > 0.0):
        return None
    return float(5.0 * math.log10(f))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="DDR reconnection conditions (Step 4.7): quantify required α/s_L/s_R.")
    ap.add_argument("--out-dir", default=str(_ROOT / "output" / "cosmology"), help="Output dir (default: output/cosmology)")
    ap.add_argument("--nsigma", type=float, default=3.0, help="Envelope sigma multiplier (default: 3)")
    args = ap.parse_args(argv)

    out_dir = Path(str(args.out_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "cosmology_ddr_reconnection_conditions.png"
    out_json = out_dir / "cosmology_ddr_reconnection_conditions_metrics.json"

    nsigma = float(args.nsigma)
    if not (math.isfinite(nsigma) and nsigma > 0.0):
        raise SystemExit("--nsigma must be finite and > 0")

    _set_japanese_font()

    ddr_metrics = _load_required(_ROOT / "output" / "cosmology" / "cosmology_distance_duality_constraints_metrics.json")
    sn_td_metrics = _load_required(_ROOT / "output" / "cosmology" / "cosmology_sn_time_dilation_constraints_metrics.json")
    cmb_tz_metrics = _load_required(_ROOT / "output" / "cosmology" / "cosmology_cmb_temperature_scaling_constraints_metrics.json")
    bao_ratio_fit_metrics = _load_required(_ROOT / "output" / "cosmology" / "cosmology_bao_distance_ratio_fit_metrics.json")

    opacity_data = _load_required(_ROOT / "data" / "cosmology" / "cosmic_opacity_constraints.json")
    gw_opacity_path = _ROOT / "data" / "cosmology" / "gw_standard_siren_opacity_constraints.json"
    gw_opacity_data = _read_json(gw_opacity_path) if gw_opacity_path.exists() else None
    sn_evo_data = _load_required(_ROOT / "data" / "cosmology" / "sn_standard_candle_evolution_constraints.json")

    ddr_rows_raw = ddr_metrics.get("rows") or []
    if not isinstance(ddr_rows_raw, list) or not ddr_rows_raw:
        raise ValueError("invalid DDR metrics: missing rows")
    ddr_rows = [DDRRow.from_metrics_row(r) for r in ddr_rows_raw if isinstance(r, dict)]
    ddr_rows = [r for r in ddr_rows if r.id]
    if not ddr_rows:
        raise ValueError("no valid DDR rows")

    # Representative DDR row: smallest sigma.
    rep_ddr = min(ddr_rows, key=lambda r: r.epsilon0_sigma if r.epsilon0_sigma > 0.0 else float("inf"))

    # Independent probes: p_t from SN time dilation, and p_T from T(z) as a proxy for p_e.
    p_t_obs, p_t_sig = _extract_pt(sn_td_metrics)
    p_T_obs, p_T_sig = _extract_pT(cmb_tz_metrics)

    # BAO ratio fit: effective s_R (not fully independent; used as a scale reference).
    s_R_fit, s_R_fit_sig = _extract_sR(bao_ratio_fit_metrics)
    sR_by_subset = _extract_sR_by_subset(bao_ratio_fit_metrics)

    # Constraint envelopes (opacity, SN evolution).
    opacity_constraints_raw = opacity_data.get("constraints") or []
    opacity_constraints = [
        ScalarConstraint.from_opacity_row(r) for r in opacity_constraints_raw if isinstance(r, dict) and (r.get("id") is not None)
    ]
    opacity_constraints_nobao_nocmb = [c for c in opacity_constraints if (c.uses_bao is False and c.uses_cmb is False)]
    gw_opacity_constraints_raw = (gw_opacity_data.get("constraints") or []) if isinstance(gw_opacity_data, dict) else []
    gw_opacity_constraints = [
        ScalarConstraint.from_opacity_row(r)
        for r in gw_opacity_constraints_raw
        if isinstance(r, dict) and (r.get("id") is not None)
    ]
    gw_opacity_constraints_obs = [c for c in gw_opacity_constraints if c.is_forecast is False]
    gw_opacity_constraints_forecast = [c for c in gw_opacity_constraints if c.is_forecast is True]
    sn_evo_constraints_raw = sn_evo_data.get("constraints") or []
    sn_evo_constraints_all = [
        ScalarConstraint.from_sn_evo_row(r) for r in sn_evo_constraints_raw if isinstance(r, dict) and (r.get("id") is not None)
    ]
    sn_evo_constraints_nocddr = [c for c in sn_evo_constraints_all if c.assumes_cddr is not True]
    sn_evo_constraints_nobao_nocmb = [c for c in sn_evo_constraints_all if (c.uses_bao is False and c.uses_cmb is False)]
    sn_evo_constraints_nocddr_nobao_nocmb = [c for c in sn_evo_constraints_nocddr if (c.uses_bao is False and c.uses_cmb is False)]

    alpha_env_all = _envelope(opacity_constraints, nsigma=nsigma)
    alpha_env_nobao_nocmb = _envelope(opacity_constraints_nobao_nocmb, nsigma=nsigma)
    alpha_env_gw_forecast = _envelope(gw_opacity_constraints_forecast, nsigma=nsigma)
    sL_env_all = _envelope(sn_evo_constraints_all, nsigma=nsigma)
    sL_env_nocddr = _envelope(sn_evo_constraints_nocddr, nsigma=nsigma)
    sL_env_nobao_nocmb = _envelope(sn_evo_constraints_nobao_nocmb, nsigma=nsigma)
    sL_env_nocddr_nobao_nocmb = _envelope(sn_evo_constraints_nocddr_nobao_nocmb, nsigma=nsigma)
    sR_env = _env_from_gaussian(s_R_fit, s_R_fit_sig, nsigma=nsigma)
    sR_env_by_subset = {k: _env_from_gaussian(mu, sig, nsigma=nsigma) for k, (mu, sig) in sR_by_subset.items()}

    # Compute required correction for each DDR row, expressed as α-only / s_L-only / s_R-only.
    computed_rows: List[Dict[str, Any]] = []
    for r in ddr_rows:
        delta_eps = float(r.epsilon0_obs - r.epsilon0_pred_pbg_static)  # ~= ε0_obs + 1
        # Allow p_t/p_e to move (optional reference; does not change much with current constraints).
        corr_ptpe: Optional[float] = None
        if p_t_obs is not None and p_T_obs is not None:
            corr_ptpe = float(r.epsilon0_obs + 2.0 - (p_t_obs + p_T_obs) / 2.0)
        computed_rows.append(
            {
                "id": r.id,
                "short_label": r.short_label,
                "uses_bao": bool(r.uses_bao),
                "epsilon0_obs": float(r.epsilon0_obs),
                "epsilon0_sigma": float(r.epsilon0_sigma),
                "epsilon0_pred_pbg_static": float(r.epsilon0_pred_pbg_static),
                "delta_epsilon_needed": float(delta_eps),
                "combo_needed__sR_plus_alpha_minus_sL_over2": float(delta_eps),
                "combo_needed_with_pt_pe_obs": corr_ptpe,
                "alpha_needed_if_only_opacity": float(delta_eps),
                "s_L_needed_if_only_candle_evolution": float(-2.0 * delta_eps),
                "s_R_needed_if_only_ruler_evolution": float(delta_eps),
                "delta_distance_modulus_mag_z1": r.delta_mu_z1_mag,
                "flux_dimming_factor_needed_z1": r.flux_dimming_z1,
                "tau_equivalent_dimming_z1": r.tau_equiv_z1,
            }
        )

    # Sort for readability (BAO-based first, then by sigma).
    computed_rows.sort(key=lambda rr: (not bool(rr["uses_bao"]), float(rr["epsilon0_sigma"])))

    # Representative comparisons: how far required values sit from existing constraints.
    rep_required_alpha = float(rep_ddr.epsilon0_obs - rep_ddr.epsilon0_pred_pbg_static)
    rep_required_sL = float(-2.0 * rep_required_alpha)
    rep_required_sR = float(rep_required_alpha)

    rep_alpha_vs_constraints: List[Dict[str, Any]] = []
    for c in opacity_constraints:
        z = _zscore_required(rep_required_alpha, mu=c.mu, sigma=c.sigma)
        rep_alpha_vs_constraints.append(
            {
                "id": c.id,
                "short_label": c.short_label,
                "uses_bao": c.uses_bao,
                "uses_cmb": c.uses_cmb,
                "alpha_mu": float(c.mu),
                "alpha_sigma": float(c.sigma),
                "z_required_minus_mu_over_sigma": z,
                "abs_z": (None if z is None else abs(float(z))),
            }
        )
    rep_alpha_vs_constraints.sort(key=lambda r: (float("inf") if r["abs_z"] is None else float(r["abs_z"])), reverse=True)

    rep_alpha_vs_gw_constraints: List[Dict[str, Any]] = []
    for c in gw_opacity_constraints:
        z = _zscore_required(rep_required_alpha, mu=c.mu, sigma=c.sigma)
        rep_alpha_vs_gw_constraints.append(
            {
                "id": c.id,
                "short_label": c.short_label,
                "is_forecast": c.is_forecast,
                "alpha_mu": float(c.mu),
                "alpha_sigma": float(c.sigma),
                "z_required_minus_mu_over_sigma": z,
                "abs_z": (None if z is None else abs(float(z))),
            }
        )
    rep_alpha_vs_gw_constraints.sort(key=lambda r: (float("inf") if r["abs_z"] is None else float(r["abs_z"])), reverse=True)

    rep_sL_vs_constraints: List[Dict[str, Any]] = []
    for c in sn_evo_constraints_all:
        z = _zscore_required(rep_required_sL, mu=c.mu, sigma=c.sigma)
        rep_sL_vs_constraints.append(
            {
                "id": c.id,
                "short_label": c.short_label,
                "uses_bao": c.uses_bao,
                "uses_cmb": c.uses_cmb,
                "assumes_cddr": c.assumes_cddr,
                "s_L_mu": float(c.mu),
                "s_L_sigma": float(c.sigma),
                "z_required_minus_mu_over_sigma": z,
                "abs_z": (None if z is None else abs(float(z))),
            }
        )
    rep_sL_vs_constraints.sort(key=lambda r: (float("inf") if r["abs_z"] is None else float(r["abs_z"])), reverse=True)

    rep_sR_vs_bao_fit: List[Dict[str, Any]] = []
    for key, label in (
        ("combined", "combined (BOSS+eBOSS+DESI)"),
        ("boss_only", "BOSS-only"),
        ("eboss_only", "eBOSS-only"),
        ("desi_only", "DESI-only"),
    ):
        mu, sig = sR_by_subset.get(key, (None, None))
        z = None if (mu is None or sig is None) else _zscore_required(rep_required_sR, mu=float(mu), sigma=float(sig))
        rep_sR_vs_bao_fit.append(
            {
                "subset": key,
                "short_label": label,
                "s_R_mu": mu,
                "s_R_sigma_1d": sig,
                "z_required_minus_mu_over_sigma": z,
                "abs_z": (None if z is None else abs(float(z))),
            }
        )

    # Decision policy (explicit): classify by sigma distance to the closest compatible constraint.
    # NOTE: for opacity/s_L we report both "all" and a conservative subset that excludes BAO/CMB (and CDDR-assumed for s_L).
    rep_alpha_min_abs_z_all = _min_abs_z(rep_alpha_vs_constraints)
    rep_alpha_min_abs_z_nobao_nocmb = _min_abs_z(
        rep_alpha_vs_constraints, keep=lambda r: (r.get("uses_bao") is False and r.get("uses_cmb") is False)
    )
    rep_sL_min_abs_z_all = _min_abs_z(rep_sL_vs_constraints)
    rep_sL_min_abs_z_nocddr = _min_abs_z(rep_sL_vs_constraints, keep=lambda r: (r.get("assumes_cddr") is not True))
    rep_sL_min_abs_z_nocddr_nobao_nocmb = _min_abs_z(
        rep_sL_vs_constraints,
        keep=lambda r: (
            r.get("assumes_cddr") is not True and r.get("uses_bao") is False and r.get("uses_cmb") is False
        ),
    )
    rep_sR_abs_z_by_subset = {r["subset"]: r.get("abs_z") for r in rep_sR_vs_bao_fit if isinstance(r, dict)}

    # Martinelli2021-style d_L/d_A contribution bookkeeping (not unique; z=1 anchor).
    z_ref = 1.0
    eta_factor_z1 = (1.0 + z_ref) ** rep_required_alpha
    dl_factor_all_z1 = eta_factor_z1
    da_factor_all_z1 = (1.0 + z_ref) ** (-rep_required_alpha)
    dl_factor_half_z1 = (1.0 + z_ref) ** (0.5 * rep_required_alpha)
    da_factor_half_z1 = (1.0 + z_ref) ** (-0.5 * rep_required_alpha)

    # --- Figure
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 3, figsize=(18, 10), sharey=True)
    ax_a, ax_sl, ax_sr = axes

    labels = [str(r["short_label"]) for r in computed_rows]
    y = np.arange(len(computed_rows), dtype=float)

    def _plot_panel(
        ax: Any,
        *,
        x_name: str,
        xerr_name: str,
        title: str,
        envs: List[Tuple[Tuple[Optional[float], Optional[float]], str, str, float]],
        color_bao: str = "#1f77b4",
        color_nobao: str = "#2ca02c",
    ) -> None:
        for env, env_label, color, alpha in envs:
            lo, hi = env
            if lo is None or hi is None:
                continue
            ax.axvspan(lo, hi, color=color, alpha=float(alpha), lw=0.0, label=env_label)
        ax.axvline(0.0, color="#333333", lw=1.0, alpha=0.35)
        for i, row in enumerate(computed_rows):
            x = float(row[x_name])
            xerr = float(row[xerr_name])
            is_bao = bool(row["uses_bao"])
            c = color_bao if is_bao else color_nobao
            m = "o" if is_bao else "s"
            ax.errorbar(x, y[i], xerr=xerr, fmt=m, ms=5, color=c, ecolor=c, elinewidth=1.2, capsize=2, alpha=0.95)
        ax.set_title(title, fontsize=12)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.25)

    # Errorbars: use ε0_sigma (propagate for s_L-only case).
    for row in computed_rows:
        row["_sig_alpha"] = float(row["epsilon0_sigma"])
        row["_sig_sL"] = float(2.0 * row["epsilon0_sigma"])
        row["_sig_sR"] = float(row["epsilon0_sigma"])

    _plot_panel(
        ax_a,
        x_name="alpha_needed_if_only_opacity",
        xerr_name="_sig_alpha",
        title="必要 α（不透明度のみで吸収）",
        envs=[
            (alpha_env_all, f"opacity all ±{nsigma:g}σ", "#999999", 0.16),
            (alpha_env_nobao_nocmb, f"opacity no-BAO/CMB ±{nsigma:g}σ", "#666666", 0.22),
            (alpha_env_gw_forecast, f"GW sirens (forecast) ±{nsigma:g}σ", "#17becf", 0.16),
        ],
    )
    _plot_panel(
        ax_sl,
        x_name="s_L_needed_if_only_candle_evolution",
        xerr_name="_sig_sL",
        title="必要 s_L（SNe Ia 光度進化のみで吸収）",
        envs=[
            (sL_env_all, f"SNe evo all ±{nsigma:g}σ", "#999999", 0.12),
            (sL_env_nocddr, f"SNe evo assumes_cddr!=True ±{nsigma:g}σ", "#666666", 0.20),
            (sL_env_nocddr_nobao_nocmb, f"SNe evo no-BAO/CMB & assumes_cddr!=True ±{nsigma:g}σ", "#444444", 0.26),
        ],
    )
    _plot_panel(
        ax_sr,
        x_name="s_R_needed_if_only_ruler_evolution",
        xerr_name="_sig_sR",
        title="必要 s_R（BAO 定規進化のみで吸収）",
        envs=[
            (sR_env_by_subset.get("combined", (None, None)), f"BAO ratio combined (BOSS+eBOSS+DESI) ±{nsigma:g}σ", "#999999", 0.16),
            (sR_env_by_subset.get("boss_only", (None, None)), f"BAO ratio BOSS ±{nsigma:g}σ", "#1f77b4", 0.10),
            (sR_env_by_subset.get("eboss_only", (None, None)), f"BAO ratio eBOSS ±{nsigma:g}σ", "#ff7f0e", 0.10),
            (sR_env_by_subset.get("desi_only", (None, None)), f"BAO ratio DESI ±{nsigma:g}σ", "#9467bd", 0.10),
        ],
    )

    # Shared annotations
    fig.suptitle("DDR再接続の条件定量化（静的背景P最小: ε0=-1）", fontsize=14)
    subtitle = (
        f"rep DDR: {rep_ddr.short_label} (id={rep_ddr.id}, ε0={rep_ddr.epsilon0_obs:+.3f}±{rep_ddr.epsilon0_sigma:.3f})"
        + (
            f" / p_t={_fmt(p_t_obs)}±{_fmt(p_t_sig)} (SN time dilation), p_e≈p_T={_fmt(p_T_obs)}±{_fmt(p_T_sig)} (T(z))"
            if (p_t_obs is not None and p_T_obs is not None)
            else ""
        )
    )
    fig.text(0.5, 0.945, subtitle, ha="center", va="top", fontsize=10)

    # Envelope legend (deduplicate by label)
    handles_all: List[Any] = []
    labels_all: List[str] = []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles_all.extend(h)
        labels_all.extend(l)
    env_seen: Dict[str, Any] = {}
    for h, l in zip(handles_all, labels_all):
        if not l or l in env_seen:
            continue
        env_seen[l] = h

    legend_elems = [
        Line2D([0], [0], marker="o", color="w", label="DDR uses BAO", markerfacecolor="#1f77b4", markersize=7),
        Line2D([0], [0], marker="s", color="w", label="DDR without BAO", markerfacecolor="#2ca02c", markersize=7),
    ]
    ax_sr.legend(
        handles=legend_elems + list(env_seen.values()),
        labels=[h.get_label() for h in legend_elems] + list(env_seen.keys()),
        loc="lower right",
        fontsize=8,
        frameon=True,
    )

    plt.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "definition": {
            "epsilon0_model": "(p_e+p_t-s_L)/2 - 2 + s_R + alpha",
            "combo_needed": "s_R + alpha - s_L/2 = epsilon0_obs + 2 - (p_e+p_t)/2 (p_e=p_t=1 => epsilon0_obs+1)",
            "notes": [
                "This output summarizes required corrections as conditions; it does not select a mechanism.",
                "p_e is approximated by the CMB temperature-scaling exponent p_T as a proxy (assumption).",
            ],
        },
        "inputs": {
            "ddr_metrics": "output/cosmology/cosmology_distance_duality_constraints_metrics.json",
            "sn_time_dilation_metrics": "output/cosmology/cosmology_sn_time_dilation_constraints_metrics.json",
            "cmb_temperature_scaling_metrics": "output/cosmology/cosmology_cmb_temperature_scaling_constraints_metrics.json",
            "opacity_constraints": "data/cosmology/cosmic_opacity_constraints.json",
            "gw_opacity_constraints": (None if gw_opacity_data is None else "data/cosmology/gw_standard_siren_opacity_constraints.json"),
            "sn_evolution_constraints": "data/cosmology/sn_standard_candle_evolution_constraints.json",
            "bao_ratio_fit_metrics": "output/cosmology/cosmology_bao_distance_ratio_fit_metrics.json",
        },
        "independent_probes": {
            "p_t_from_sn_time_dilation": {"p_t_obs": p_t_obs, "p_t_sigma": p_t_sig},
            "p_e_proxy_from_Tz": {"p_T_obs": p_T_obs, "p_T_sigma": p_T_sig, "note": "proxy for p_e"},
        },
        "reference_constraints": {
            "opacity_alpha": {
                "n": int(len(opacity_constraints)),
                "envelope_nsigma": float(nsigma),
                "n_no_bao_no_cmb": int(len(opacity_constraints_nobao_nocmb)),
                "envelope_all": list(alpha_env_all),
                "envelope_no_bao_no_cmb": list(alpha_env_nobao_nocmb),
            },
            "opacity_alpha_gw": {
                "n_all": int(len(gw_opacity_constraints)),
                "n_observed": int(len(gw_opacity_constraints_obs)),
                "n_forecast": int(len(gw_opacity_constraints_forecast)),
                "envelope_nsigma": float(nsigma),
                "envelope_forecast": list(alpha_env_gw_forecast),
                "note": "GW standard sirens provide an opacity cross-check; observed GW170817 is local (z<<1) and effectively unconstraining for cosmic opacity.",
            },
            "sn_candle_evolution_s_L": {
                "n_all": int(len(sn_evo_constraints_all)),
                "n_no_cddr_assumed": int(len(sn_evo_constraints_nocddr)),
                "n_no_bao_no_cmb": int(len(sn_evo_constraints_nobao_nocmb)),
                "n_no_bao_no_cmb_no_cddr_assumed": int(len(sn_evo_constraints_nocddr_nobao_nocmb)),
                "envelope_nsigma": float(nsigma),
                "envelope_all": list(sL_env_all),
                "envelope_no_cddr_assumed": list(sL_env_nocddr),
                "envelope_no_bao_no_cmb": list(sL_env_nobao_nocmb),
                "envelope_no_bao_no_cmb_no_cddr_assumed": list(sL_env_nocddr_nobao_nocmb),
            },
            "bao_ratio_fit_s_R": {
                "s_R": s_R_fit,
                "s_R_sigma_1d": s_R_fit_sig,
                "envelope_nsigma": float(nsigma),
                "note": "combined includes BOSS+eBOSS+DESI in cosmology_bao_distance_ratio_fit.py",
                "envelope_combined": list(sR_env_by_subset.get("combined", (None, None))),
                "envelope_boss_only": list(sR_env_by_subset.get("boss_only", (None, None))),
                "envelope_eboss_only": list(sR_env_by_subset.get("eboss_only", (None, None))),
                "envelope_desi_only": list(sR_env_by_subset.get("desi_only", (None, None))),
            },
        },
        "decision_policy": {
            "sigma_classification": {"ok": "|z|<3", "mixed": "3≤|z|<5", "ng": "|z|≥5"},
            "note": "For each mechanism, we summarize the smallest |z| (best-case compatibility) against the listed reference constraints.",
        },
        "representative_ddr": {
            "id": rep_ddr.id,
            "short_label": rep_ddr.short_label,
            "epsilon0_obs": float(rep_ddr.epsilon0_obs),
            "epsilon0_sigma": float(rep_ddr.epsilon0_sigma),
            "epsilon0_pred_pbg_static": float(rep_ddr.epsilon0_pred_pbg_static),
            "delta_epsilon_needed": float(rep_required_alpha),
            "alpha_needed_if_only_opacity": float(rep_required_alpha),
            "s_L_needed_if_only_candle_evolution": float(rep_required_sL),
            "s_R_needed_if_only_ruler_evolution": float(rep_required_sR),
        },
        "representative_decision_summary": {
            "representative_ddr_id": rep_ddr.id,
            "opacity_only": {
                "alpha_required": float(rep_required_alpha),
                "min_abs_z_all": rep_alpha_min_abs_z_all,
                "class_all": _classify_sigma(rep_alpha_min_abs_z_all),
                "min_abs_z_no_bao_no_cmb": rep_alpha_min_abs_z_nobao_nocmb,
                "class_no_bao_no_cmb": _classify_sigma(rep_alpha_min_abs_z_nobao_nocmb),
            },
            "candle_only": {
                "s_L_required": float(rep_required_sL),
                "min_abs_z_all": rep_sL_min_abs_z_all,
                "class_all": _classify_sigma(rep_sL_min_abs_z_all),
                "min_abs_z_assumes_cddr!=True": rep_sL_min_abs_z_nocddr,
                "class_assumes_cddr!=True": _classify_sigma(rep_sL_min_abs_z_nocddr),
                "min_abs_z_no_bao_no_cmb_and_assumes_cddr!=True": rep_sL_min_abs_z_nocddr_nobao_nocmb,
                "class_no_bao_no_cmb_and_assumes_cddr!=True": _classify_sigma(rep_sL_min_abs_z_nocddr_nobao_nocmb),
            },
            "ruler_only": {
                "s_R_required": float(rep_required_sR),
                "abs_z_by_bao_ratio_subset": rep_sR_abs_z_by_subset,
                "class_by_bao_ratio_subset": {k: _classify_sigma(v) for k, v in rep_sR_abs_z_by_subset.items()},
            },
        },
        "representative_comparisons": {
            "alpha_required_vs_opacity_constraints": rep_alpha_vs_constraints,
            "alpha_required_vs_gw_opacity_constraints": rep_alpha_vs_gw_constraints,
            "s_L_required_vs_sn_evolution_constraints": rep_sL_vs_constraints,
            "s_R_required_vs_bao_ratio_fit": rep_sR_vs_bao_fit,
        },
        "representative_dL_dA_decomposition": {
            "z_ref": float(z_ref),
            "delta_epsilon_needed": float(rep_required_alpha),
            "eta_factor_needed_z1": float(eta_factor_z1),
            "note": "Not unique: any (dL_factor, dA_factor) with dL_factor/dA_factor = eta_factor works. Values below show two common extremes and a symmetric split.",
            "if_all_in_dL": {
                "dL_factor_z1": float(dl_factor_all_z1),
                "delta_mu_mag_z1": _factor_to_delta_mu_mag(dl_factor_all_z1),
                "flux_dimming_factor_z1": (None if dl_factor_all_z1 is None else float(dl_factor_all_z1**2)),
            },
            "if_all_in_dA": {"dA_factor_z1": float(da_factor_all_z1)},
            "if_split_symmetric": {"dL_factor_z1": float(dl_factor_half_z1), "dA_factor_z1": float(da_factor_half_z1)},
        },
        "ddr_rows": computed_rows,
        "outputs": {"png": str(out_png).replace("\\", "/"), "metrics_json": str(out_json).replace("\\", "/")},
    }
    _write_json(out_json, payload)

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_ddr_reconnection_conditions",
                "argv": list(sys.argv),
                "inputs": {
                    "ddr_metrics": _ROOT / "output" / "cosmology" / "cosmology_distance_duality_constraints_metrics.json",
                    "sn_time_dilation": _ROOT / "output" / "cosmology" / "cosmology_sn_time_dilation_constraints_metrics.json",
                    "cmb_temperature": _ROOT / "output" / "cosmology" / "cosmology_cmb_temperature_scaling_constraints_metrics.json",
                    "opacity": _ROOT / "data" / "cosmology" / "cosmic_opacity_constraints.json",
                    "gw_opacity": (gw_opacity_path if gw_opacity_data is not None else None),
                    "sn_evo": _ROOT / "data" / "cosmology" / "sn_standard_candle_evolution_constraints.json",
                    "bao_ratio_fit": _ROOT / "output" / "cosmology" / "cosmology_bao_distance_ratio_fit_metrics.json",
                },
                "outputs": {"png": out_png, "metrics_json": out_json},
                "params": {"nsigma": nsigma},
            }
        )
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
