#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_distance_duality_constraints.py

距離二重性（Etherington / DDR）を、一次ソース（公表されたε0制約）に結びつけて
「棄却条件（どの精度なら棄却できるか / 既に棄却されるか）」として固定出力する。

入力（固定）:
  - data/cosmology/distance_duality_constraints.json

定義（一次ソースのパラメータ化）:
  d_L(z) = (1+z)^(2+ε0) d_A(z)
  η(z) ≡ d_L/((1+z)^2 d_A) = (1+z)^(ε0)
  標準（FRW + 光子数保存）: ε0 = 0

本リポジトリの「背景P（膨張なし・静的幾何）」最小モデル（2.5節）:
  d_L = (1+z) d_A  →  ε0 = -1  →  η(z)=1/(1+z)
  併記：P-model固有指標 η^(P)≡d_L/((1+z)d_A) は最小で η^(P)=1

出力（固定名）:
  - output/private/cosmology/cosmology_distance_duality_constraints.png
  - output/private/cosmology/cosmology_distance_duality_constraints_metrics.json
  - output/private/cosmology/cosmology_ddr_pmodel_relation.json（P-model最小のDDR関係の固定）
  - output/private/cosmology/cosmology_ddr_epsilon_reinterpretation.json（代表ε0の再解釈：張力と必要補正）
  - output/private/cosmology/cosmology_ddr_pmodel_eta_constraints.json（η^(P)での再評価）
  - output/private/cosmology/cosmology_ddr_pmodel_eta_constraints.png（η^(P)での再評価）
  - output/private/cosmology/cosmology_ddr_pmodel_falsification_pack.json（η^(P)での反証条件パック）
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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


def _fmt_float(x: float, *, digits: int = 6) -> str:
    if x == 0.0:
        return "0"
    ax = abs(x)
    if ax >= 1e4 or ax < 1e-3:
        return f"{x:.{digits}g}"
    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


def _safe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


@dataclass(frozen=True)
class Constraint:
    id: str
    short_label: str
    title: str
    epsilon0: float
    epsilon0_sigma: float
    uses_bao: bool
    sigma_note: str
    source: Dict[str, Any]

    @staticmethod
    def from_json(j: Dict[str, Any]) -> "Constraint":
        return Constraint(
            id=str(j["id"]),
            short_label=str(j.get("short_label") or j["id"]),
            title=str(j.get("title") or j.get("short_label") or j["id"]),
            epsilon0=float(j["epsilon0"]),
            epsilon0_sigma=float(j["epsilon0_sigma"]),
            uses_bao=bool(j.get("uses_bao", False)),
            sigma_note=str(j.get("sigma_note") or ""),
            source=dict(j.get("source") or {}),
        )


def compute(rows: Sequence[Constraint]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    # Model predictions for epsilon0
    eps_frw = 0.0
    eps_pbg_static = -1.0
    z_ref = 1.0
    # NOTE: ε0 is constrained as a single exponent, but readers often want an intuition at a concrete z.
    # We therefore also report the "extra η-factor" at z=1, which corresponds to a simple multiplicative
    # correction to reconcile the static-min model with the observed ε0.

    for r in rows:
        sig = float(r.epsilon0_sigma)
        z_frw = None
        z_pbg = None
        if sig > 0:
            z_frw = (eps_frw - float(r.epsilon0)) / sig
            z_pbg = (eps_pbg_static - float(r.epsilon0)) / sig

        # "Non-rejection" threshold (3σ): require sig >= |eps_model - eps_obs| / 3.
        sig_need_nonreject_pbg_3sigma = None
        if sig > 0:
            sig_need_nonreject_pbg_3sigma = abs(eps_pbg_static - float(r.epsilon0)) / 3.0

        # Reconciliation view (important for "static infinite space" interpretations):
        # If some additional mechanism/systematic effectively shifts ε0 by Δε (e.g., opacity, evolution of
        # standard candles/rulers, or unit-scaling tied to P_bg), then:
        #   ε0_eff = ε0_model + Δε
        # To match an observed ε0_obs, we need:
        #   Δε_needed = ε0_obs - ε0_model
        delta_eps_needed = float(r.epsilon0) - eps_pbg_static
        # Equivalent multiplicative correction on η(z): η_eff = η_model * (1+z)^(Δε_needed)
        # Report at z=1 as an intuition anchor (factor ~2 when Δε≈1).
        extra_eta_factor_z1 = (1.0 + z_ref) ** delta_eps_needed
        # The same exponent implies an additional multiplicative correction on D_L relative to the static-min
        # model (D_L=(1+z)D_A): D_L_needed = D_L_static * (1+z)^(Δε).
        extra_dl_factor_z1 = extra_eta_factor_z1
        # Translate into "how big a systematic would be needed" in the language of distance indicators.
        # Distance modulus shift: Δμ = 5 log10(D_L_needed / D_L_static)
        delta_mu_mag_z1 = 5.0 * math.log10(extra_dl_factor_z1) if extra_dl_factor_z1 > 0 else None
        # Flux dimming factor (F ∝ 1/D_L^2): F_needed = F_static / dimming_factor
        flux_dimming_factor_z1 = extra_dl_factor_z1**2
        # Equivalent optical depth τ for a pure dimming interpretation: F -> F * exp(-τ) => τ = ln(dimming_factor)
        tau_equiv_z1 = math.log(flux_dimming_factor_z1) if flux_dimming_factor_z1 > 0 else None

        # Optional: interpret the needed correction as a power-law "opacity" term:
        #   η_eff = η_model * (1+z)^α   (equivalently exp(+τ/2) = (1+z)^α, so τ(z)=2α ln(1+z))
        # This is only one possible interpretation; here it is just a convenient re-parameterization.
        alpha_opacity_powerlaw_needed = delta_eps_needed

        sigma_multiplier_nonreject = None
        if sig > 0 and sig_need_nonreject_pbg_3sigma is not None:
            sigma_multiplier_nonreject = float(sig_need_nonreject_pbg_3sigma) / sig

        # Optional decomposition (not unique):
        # If one interprets Δε as being caused by evolution/systematics in distance indicators:
        #   ε0_eff = -1 + (s_ruler - s_luminosity/2)
        # where:
        #   ruler physical size:  l_em = l_0 * (1+z)^(s_ruler)
        #   candle luminosity:    L_em = L_0 * (1+z)^(s_luminosity)
        # Then the required relation is: s_ruler - s_luminosity/2 = Δε_needed.
        s_ruler_needed_if_no_lum = delta_eps_needed
        s_lum_needed_if_no_ruler = -2.0 * delta_eps_needed

        # P-model specific DDR indicator:
        #   η^(P)(z) ≡ D_L / [(1+z) D_A] = (1+z)^(1+ε0)
        # Under:
        #   FRW (ε0=0)    => η^(P)=(1+z)
        #   P-model min   => D_L=(1+z)D_A => η^(P)=1 (ε0=-1 in the common parameterization)
        eta_p_exponent_obs = 1.0 + float(r.epsilon0)
        eta_p_exponent_sigma = sig
        eta_p_obs_z1 = float(extra_eta_factor_z1)  # (1+z_ref)^(1+ε0_obs), since ε0_model=-1
        eta_p_sigma_approx_z1 = None
        z_eta_pbg_static_z1 = None
        z_eta_frw_z1 = None
        eta_p_pred_pbg_static_z1 = 1.0
        eta_p_pred_frw_z1 = 1.0 + z_ref
        if sig > 0:
            # Uncertainty propagation for a Gaussian ε0 (approximation):
            #   η^(P) = (1+z)^(1+ε0) => dη/dε = ln(1+z) * η
            eta_p_sigma_approx_z1 = abs(math.log(1.0 + z_ref)) * eta_p_obs_z1 * sig
            if eta_p_sigma_approx_z1 > 0:
                z_eta_pbg_static_z1 = (eta_p_pred_pbg_static_z1 - eta_p_obs_z1) / eta_p_sigma_approx_z1
                z_eta_frw_z1 = (eta_p_pred_frw_z1 - eta_p_obs_z1) / eta_p_sigma_approx_z1

        out.append(
            {
                "id": r.id,
                "short_label": r.short_label,
                "title": r.title,
                "uses_bao": bool(r.uses_bao),
                "epsilon0_obs": float(r.epsilon0),
                "epsilon0_sigma": sig,
                "epsilon0_pred_frw": eps_frw,
                "epsilon0_pred_pbg_static": eps_pbg_static,
                "z_frw": None if z_frw is None else float(z_frw),
                "z_pbg_static": None if z_pbg is None else float(z_pbg),
                "epsilon0_extra_needed_to_match_obs": float(delta_eps_needed),
                "extra_eta_factor_needed_z1": float(extra_eta_factor_z1),
                "extra_dl_factor_needed_z1": float(extra_dl_factor_z1),
                "eta_p_exponent_obs": float(eta_p_exponent_obs),
                "eta_p_exponent_sigma": float(eta_p_exponent_sigma),
                "eta_p_obs_z1": float(eta_p_obs_z1),
                "eta_p_sigma_approx_z1": (
                    None if eta_p_sigma_approx_z1 is None else float(eta_p_sigma_approx_z1)
                ),
                "eta_p_pred_frw_z1": float(eta_p_pred_frw_z1),
                "eta_p_pred_pbg_static_z1": float(eta_p_pred_pbg_static_z1),
                "z_eta_frw_z1": None if z_eta_frw_z1 is None else float(z_eta_frw_z1),
                "z_eta_pbg_static_z1": None if z_eta_pbg_static_z1 is None else float(z_eta_pbg_static_z1),
                "delta_distance_modulus_mag_z1": (None if delta_mu_mag_z1 is None else float(delta_mu_mag_z1)),
                "flux_dimming_factor_needed_z1": float(flux_dimming_factor_z1),
                "tau_equivalent_dimming_z1": (None if tau_equiv_z1 is None else float(tau_equiv_z1)),
                "reconciliation_notes": {
                    "z_ref": z_ref,
                    "interpretation": "静的背景P最小（ε0=-1）を観測ε0へ寄せるために必要な“有効Δε”。(1+z)^Δε は D_L の追加倍率であり、同時に η^(P)(z)=D_L/((1+z)D_A) の値（z_refでの目安）でもある。",
                    "opacity_powerlaw_interpretation": "補正を灰色不透明度（光子数非保存）として吸収するなら、exp(+τ/2)=(1+z)^α（τ(z)=2α ln(1+z)）と置け、必要条件は α=Δε_needed。",
                    "alpha_opacity_powerlaw_needed_if_no_evolution": float(alpha_opacity_powerlaw_needed),
                    "decomposition_example": "もし距離指標の系統を『標準定規のサイズ進化 s_ruler』『標準光源の光度進化 s_luminosity』で表すと、s_ruler - s_luminosity/2 = Δε_needed。",
                    "s_ruler_needed_if_no_luminosity_evolution": float(s_ruler_needed_if_no_lum),
                    "s_luminosity_needed_if_no_ruler_evolution": float(s_lum_needed_if_no_ruler),
                },
                "sigma_multiplier_to_not_reject_pbg_static_3sigma": (
                    None if sigma_multiplier_nonreject is None else float(sigma_multiplier_nonreject)
                ),
                "sigma_needed_to_not_reject_pbg_static_3sigma": (
                    None if sig_need_nonreject_pbg_3sigma is None else float(sig_need_nonreject_pbg_3sigma)
                ),
                "sigma_note": r.sigma_note,
                "source": r.source,
            }
        )
    return out


def _plot(rows: Sequence[Dict[str, Any]], *, out_png: Path, z_max: float) -> None:
    _set_japanese_font()
    import matplotlib.pyplot as plt

    z = np.linspace(0.0, float(z_max), 700)
    one_p_z = 1.0 + z

    # Model predictions for eta(z) = (1+z)^(epsilon0)
    eta_frw = np.ones_like(z)
    eta_pbg = 1.0 / one_p_z

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

    # Panel 1: eta(z) curves + observational band from epsilon0 constraint(s).
    ax1.plot(z, eta_frw, label="標準（FRW + 光子保存）: η=1", linewidth=2.0)
    ax1.plot(z, eta_pbg, label="背景P（静的）: η=1/(1+z)（ε0=-1）", linewidth=2.0)

    # Keep the left panel readable:
    # - BAO-based: show the tightest band (smallest σ)
    # - non-BAO: show the "least rejecting" band (min |z| against ε0=-1), to illustrate how strongly this depends
    #   on distance-indicator assumptions.
    best_bao = None
    best_no_bao = None
    best_bao_sig = float("inf")
    best_no_bao_abs_z = float("inf")
    for r in rows:
        eps = _safe_float(r.get("epsilon0_obs"))
        sig = _safe_float(r.get("epsilon0_sigma"))
        uses_bao = bool(r.get("uses_bao", False))
        if eps is None or sig is None or sig <= 0:
            continue
        if uses_bao and sig < best_bao_sig:
            best_bao_sig = sig
            best_bao = r
        if not uses_bao:
            z_pbg = _safe_float(r.get("z_pbg_static"))
            if z_pbg is None:
                continue
            az = abs(float(z_pbg))
            if az < best_no_bao_abs_z:
                best_no_bao_abs_z = az
                best_no_bao = r

    for r, color, label_prefix in [
        (best_bao, "#1f77b4", "観測制約（BAO含む, 1σ）"),
        (best_no_bao, "#ff7f0e", "観測制約（BAOなし, 1σ）"),
    ]:
        if not isinstance(r, dict):
            continue
        eps = _safe_float(r.get("epsilon0_obs"))
        sig = _safe_float(r.get("epsilon0_sigma"))
        if eps is None or sig is None or sig <= 0:
            continue
        eta_mid = one_p_z ** float(eps)
        eta_lo = one_p_z ** float(eps - sig)
        eta_hi = one_p_z ** float(eps + sig)
        short = str(r.get("short_label") or r.get("id") or "")
        ax1.fill_between(z, eta_lo, eta_hi, alpha=0.18, color=color, label=f"{label_prefix}: {short}")
        ax1.plot(z, eta_mid, color=color, linewidth=1.2, alpha=0.85)

    ax1.set_title("距離二重性 η(z) の観測制約（ε0パラメータ化）", fontsize=13)
    ax1.set_xlabel("赤方偏移 z", fontsize=11)
    ax1.set_ylabel("η(z)=d_L/((1+z)^2 d_A)", fontsize=11)
    ax1.set_ylim(0.0, 1.15)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9, loc="upper right")

    # Panel 2: epsilon0 summary
    y = np.arange(len(rows), dtype=float)
    eps_obs = np.array([float(r.get("epsilon0_obs", float("nan"))) for r in rows], dtype=float)
    eps_sig = np.array([float(r.get("epsilon0_sigma", float("nan"))) for r in rows], dtype=float)
    labels = [str(r.get("short_label") or r.get("id") or "") for r in rows]
    uses_bao = np.array([bool(r.get("uses_bao", False)) for r in rows], dtype=bool)

    ax2.axvline(0.0, color="#333333", linewidth=1.2, alpha=0.85, label="標準: ε0=0")
    ax2.axvline(-1.0, color="#d62728", linewidth=1.2, alpha=0.85, label="背景P（静的）: ε0=-1")

    # Distinguish whether the constraint uses BAO-derived distances.
    for mask, fmt, color, label in [
        (uses_bao, "o", "#1f77b4", "観測（BAO含む）"),
        (~uses_bao, "s", "#ff7f0e", "観測（BAOなし）"),
    ]:
        if not np.any(mask):
            continue
        ax2.errorbar(
            eps_obs[mask],
            y[mask],
            xerr=np.where(np.isfinite(eps_sig[mask]) & (eps_sig[mask] > 0), eps_sig[mask], 0.0),
            fmt=fmt,
            capsize=4,
            color=color,
            ecolor=color,
            label=label,
        )

    ax2.set_yticks(y)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("ε0", fontsize=11)
    ax2.set_title("ε0（DDRの破れ）の観測値とモデル予測", fontsize=13)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="x")
    ax2.legend(fontsize=9, loc="upper right")

    fig.suptitle("宇宙論（棄却条件）：距離二重性（DDR）で背景P（静的）を判別", fontsize=14)

    # Add a concise reconciliation hint (based on the first constraint, if present).
    if rows:
        r0 = rows[0] if isinstance(rows[0], dict) else {}
        delta_eps = _safe_float(r0.get("epsilon0_extra_needed_to_match_obs"))
        extra_eta_z1 = _safe_float(r0.get("extra_eta_factor_needed_z1"))
        delta_mu = _safe_float(r0.get("delta_distance_modulus_mag_z1"))
        flux_dim = _safe_float(r0.get("flux_dimming_factor_needed_z1"))
        if delta_eps is not None and extra_eta_z1 is not None:
            extra_txt = ""
            if delta_mu is not None and flux_dim is not None:
                extra_txt = (
                    f"（Δμ≈{_fmt_float(delta_mu, digits=2)} mag, "
                    f"追加減光≈1/{_fmt_float(flux_dim, digits=2)}）"
                )
            fig.text(
                0.5,
                0.015,
                (
                    "静的背景P最小（ε0=-1）→ 観測ε0へ寄せるには有効Δε≈"
                    f"{_fmt_float(delta_eps, digits=3)}（z=1でD_Lを約{_fmt_float(extra_eta_z1, digits=2)}倍補正）"
                    f"{extra_txt}"
                ),
                ha="center",
                fontsize=10,
            )
    fig.text(
        0.5,
        0.005,
        "棄却条件（例）：|ε_obs - ε_model| > 3σ ならそのモデルは棄却（統計的な目安）。",
        ha="center",
        fontsize=10,
    )
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.92))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_eta_pmodel(rows: Sequence[Dict[str, Any]], *, out_png: Path, z_max: float) -> None:
    """Plot η^(P)(z) ≡ D_L/((1+z)D_A) inferred from ε0 constraints (parameterization-dependent view)."""
    _set_japanese_font()
    import matplotlib.pyplot as plt

    z = np.linspace(0.0, float(z_max), 700)
    one_p_z = 1.0 + z

    # Model predictions for η^(P):
    # - FRW (ε0=0): η^(P)=(1+z)
    # - P-model minimal: η^(P)=1
    eta_frw = one_p_z
    eta_pbg = np.ones_like(z)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

    ax1.plot(z, eta_pbg, label="背景P（静的）: η^(P)=1", linewidth=2.0)
    ax1.plot(z, eta_frw, label="標準（FRW + 光子保存）: η^(P)=(1+z)", linewidth=2.0)

    reps = _select_representative(rows)
    for r, color, label_prefix in [
        (reps.get("bao"), "#1f77b4", "観測制約（BAO含む, 1σ）"),
        (reps.get("no_bao"), "#ff7f0e", "観測制約（BAOなし, 1σ）"),
    ]:
        if not isinstance(r, dict):
            continue
        eps = _safe_float(r.get("epsilon0_obs"))
        sig = _safe_float(r.get("epsilon0_sigma"))
        if eps is None or sig is None or sig <= 0:
            continue
        # η^(P)=(1+z)^(1+ε0)
        eta_mid = one_p_z ** float(1.0 + eps)
        eta_lo = one_p_z ** float(1.0 + eps - sig)
        eta_hi = one_p_z ** float(1.0 + eps + sig)
        short = str(r.get("short_label") or r.get("id") or "")
        ax1.fill_between(z, eta_lo, eta_hi, alpha=0.18, color=color, label=f"{label_prefix}: {short}")
        ax1.plot(z, eta_mid, color=color, linewidth=1.2, alpha=0.85)

    ax1.set_title("P-model固有DDR指標 η^(P)(z) の観測制約（ε0パラメータ化）", fontsize=13)
    ax1.set_xlabel("赤方偏移 z", fontsize=11)
    ax1.set_ylabel("η^(P)(z)=d_L/((1+z) d_A)", fontsize=11)
    ax1.set_ylim(0.0, max(3.0, float(1.0 + z_max) * 1.15))
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9, loc="upper left")

    # Right panel: exponent p = 1+ε0 summary.
    y = np.arange(len(rows), dtype=float)
    p_obs = np.array([float(r.get("eta_p_exponent_obs", float("nan"))) for r in rows], dtype=float)
    p_sig = np.array([float(r.get("eta_p_exponent_sigma", float("nan"))) for r in rows], dtype=float)
    labels = [str(r.get("short_label") or r.get("id") or "") for r in rows]
    uses_bao = np.array([bool(r.get("uses_bao", False)) for r in rows], dtype=bool)

    ax2.axvline(1.0, color="#333333", linewidth=1.2, alpha=0.85, label="標準（FRW）: p=1（ε0=0）")
    ax2.axvline(0.0, color="#d62728", linewidth=1.2, alpha=0.85, label="背景P（静的）: p=0（ε0=-1）")

    for mask, fmt, color, label in [
        (uses_bao, "o", "#1f77b4", "観測（BAO含む）"),
        (~uses_bao, "s", "#ff7f0e", "観測（BAOなし）"),
    ]:
        if not np.any(mask):
            continue
        ax2.errorbar(
            p_obs[mask],
            y[mask],
            xerr=np.where(np.isfinite(p_sig[mask]) & (p_sig[mask] > 0), p_sig[mask], 0.0),
            fmt=fmt,
            capsize=4,
            color=color,
            ecolor=color,
            label=label,
        )

    ax2.set_yticks(y)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("p = 1+ε0", fontsize=11)
    ax2.set_title("η^(P) の指数 p（=1+ε0）の観測値とモデル", fontsize=13)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="x")
    ax2.legend(fontsize=9, loc="upper right")

    fig.suptitle("DDR追補：η^(P)=d_L/((1+z)d_A) による再評価（前提依存）", fontsize=14)
    fig.text(
        0.5,
        0.005,
        "注意：ここでのη^(P)は、一次ソースが採用するε0パラメータ化（距離指標推定の前提を含む）をそのまま写像した“条件付き”の可視化である。",
        ha="center",
        fontsize=10,
    )
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.92))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _select_representative(rows: Sequence[Dict[str, Any]]) -> Dict[str, Optional[Dict[str, Any]]]:
    best_bao: Optional[Dict[str, Any]] = None
    best_bao_sig = float("inf")
    best_no_bao: Optional[Dict[str, Any]] = None
    best_no_bao_abs_z = float("inf")

    for r in rows:
        if not isinstance(r, dict):
            continue
        eps = _safe_float(r.get("epsilon0_obs"))
        sig = _safe_float(r.get("epsilon0_sigma"))
        if eps is None or sig is None or sig <= 0:
            continue
        uses_bao = bool(r.get("uses_bao", False))
        if uses_bao and sig < best_bao_sig:
            best_bao_sig = sig
            best_bao = r
        if not uses_bao:
            z_pbg = _safe_float(r.get("z_pbg_static"))
            if z_pbg is None:
                continue
            az = abs(float(z_pbg))
            if az < best_no_bao_abs_z:
                best_no_bao_abs_z = az
                best_no_bao = r

    return {"bao": best_bao, "no_bao": best_no_bao}


def _z_limit_for_budget_mag(*, delta_eps: float, budget_mag: float) -> Optional[float]:
    """Solve |Δμ(z)|=budget_mag for z, where Δμ(z)=5*Δε*log10(1+z)."""
    if not (budget_mag > 0):
        return None
    a = abs(float(delta_eps))
    if a == 0.0:
        return float("inf")
    exponent = float(budget_mag) / (5.0 * a)
    return float(10.0**exponent - 1.0)


def _reach_values_for_z(*, delta_eps: float, z: float) -> Dict[str, float]:
    op = 1.0 + float(z)
    if not (op > 0.0):
        raise ValueError("z must satisfy 1+z>0")
    extra_dl = op ** float(delta_eps)
    # Distance modulus shift: Δμ = 5 log10(D_L_needed / D_L_static) = 5 log10(extra_dl).
    delta_mu = 5.0 * math.log10(extra_dl) if extra_dl > 0 else float("nan")
    # Flux scales ~ 1/d_L^2 => dimming factor needed = extra_dl^2.
    flux_dimming = extra_dl * extra_dl
    tau_equiv = math.log(flux_dimming) if flux_dimming > 0 else float("nan")
    return {
        "z": float(z),
        "extra_dl_factor": float(extra_dl),
        "delta_mu_mag": float(delta_mu),
        "flux_dimming_factor": float(flux_dimming),
        "tau_equivalent_dimming": float(tau_equiv),
    }


def _build_reach_metrics(
    *,
    label: str,
    row: Dict[str, Any],
    budgets_mag: Sequence[float],
    z_refs: Sequence[float],
) -> Dict[str, Any]:
    delta_eps = float(row.get("epsilon0_extra_needed_to_match_obs", float("nan")))
    sig = float(row.get("epsilon0_sigma", float("nan")))
    out: Dict[str, Any] = {
        "label": str(label),
        "id": str(row.get("id") or ""),
        "short_label": str(row.get("short_label") or row.get("id") or ""),
        "uses_bao": bool(row.get("uses_bao", False)),
        "epsilon0_obs": float(row.get("epsilon0_obs", float("nan"))),
        "epsilon0_sigma": sig,
        "delta_eps_needed": delta_eps,
        "decomposition_example": {
            "alpha_opacity_needed": float(delta_eps),
            "s_R_needed_if_no_luminosity_evolution": float(delta_eps),
            "s_L_needed_if_no_ruler_evolution": float(-2.0 * delta_eps),
            "note": "分解は一意ではない（距離指標の前提/系統のどこに吸収するかの“解釈”）。",
        },
        "z_refs": [float(z) for z in z_refs],
        "values_at_z": [],
        "reach_z_limit_by_budget_mag": [],
        "notes": [
            "Δε_needed = ε0_obs - ε0_model（ここでは ε0_model=-1）",
            "extra_dl_factor=(1+z)^Δε_needed, Δμ=5 log10(extra_dl_factor), τ=ln(extra_dl_factor^2)=2Δε ln(1+z)",
        ],
    }

    for z in z_refs:
        out["values_at_z"].append(_reach_values_for_z(delta_eps=delta_eps, z=float(z)))

    for b in budgets_mag:
        z_lim = _z_limit_for_budget_mag(delta_eps=delta_eps, budget_mag=float(b))
        out["reach_z_limit_by_budget_mag"].append({"budget_mag": float(b), "z_limit": z_lim})

    # Also include a quick "±1σ" sanity range at z=1.
    z1 = 1.0
    out["z1_band_abs_delta_mu_mag"] = {
        "z": z1,
        "central": abs(_reach_values_for_z(delta_eps=delta_eps, z=z1)["delta_mu_mag"]),
        "minus_1sigma": abs(_reach_values_for_z(delta_eps=(delta_eps - sig), z=z1)["delta_mu_mag"]) if sig > 0 else None,
        "plus_1sigma": abs(_reach_values_for_z(delta_eps=(delta_eps + sig), z=z1)["delta_mu_mag"]) if sig > 0 else None,
    }

    return out


def _plot_reach_limit(
    *,
    reps: Dict[str, Optional[Dict[str, Any]]],
    out_png: Path,
    z_max: float,
    budgets_mag: Sequence[float],
) -> None:
    _set_japanese_font()
    import matplotlib.pyplot as plt

    z = np.linspace(0.0, float(z_max), 700)
    one_p_z = 1.0 + z

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))

    styles = [
        ("bao", "#1f77b4", "代表（BAO含む: 最小σ）"),
        ("no_bao", "#ff7f0e", "代表（BAOなし: |z|最小）"),
    ]
    for key, color, label_prefix in styles:
        r = reps.get(key)
        if not isinstance(r, dict):
            continue
        short = str(r.get("short_label") or r.get("id") or "")
        eps = float(r.get("epsilon0_obs", float("nan")))
        sig = float(r.get("epsilon0_sigma", float("nan")))
        delta_eps = float(r.get("epsilon0_extra_needed_to_match_obs", float("nan")))

        # |Δμ(z)| band from Δε ± 1σ(ε0)
        log10_op = np.log10(one_p_z)
        ln_op = np.log(one_p_z)
        dm0 = np.abs(5.0 * delta_eps * log10_op)
        dm1 = np.abs(5.0 * (delta_eps - sig) * log10_op) if sig > 0 else dm0
        dm2 = np.abs(5.0 * (delta_eps + sig) * log10_op) if sig > 0 else dm0
        dm_lo = np.minimum(dm1, dm2)
        dm_hi = np.maximum(dm1, dm2)

        tau0 = np.abs(2.0 * delta_eps * ln_op)
        tau1 = np.abs(2.0 * (delta_eps - sig) * ln_op) if sig > 0 else tau0
        tau2 = np.abs(2.0 * (delta_eps + sig) * ln_op) if sig > 0 else tau0
        tau_lo = np.minimum(tau1, tau2)
        tau_hi = np.maximum(tau1, tau2)

        label = f"{label_prefix}: {short}（ε0={_fmt_float(eps, digits=3)}±{_fmt_float(sig, digits=3)}）"
        ax1.fill_between(z, dm_lo, dm_hi, alpha=0.18, color=color)
        ax1.plot(z, dm0, color=color, linewidth=2.0, label=label)

        ax2.fill_between(z, tau_lo, tau_hi, alpha=0.18, color=color)
        ax2.plot(z, tau0, color=color, linewidth=2.0, label=label)

    # Reference budgets (purely illustrative; not a claim about actual systematics).
    for b in budgets_mag:
        ax1.axhline(float(b), color="#777777", linewidth=1.0, alpha=0.18)

    ax1.axvline(1.0, color="#333333", linewidth=1.2, alpha=0.25)
    ax2.axvline(1.0, color="#333333", linewidth=1.2, alpha=0.25)

    ax1.set_title("必要な距離モジュラス補正 |Δμ(z)|", fontsize=13)
    ax1.set_xlabel("赤方偏移 z", fontsize=11)
    ax1.set_ylabel("|Δμ(z)| [mag]", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9, loc="upper left")

    ax2.set_title("等価減光 τ（|τ(z)|）", fontsize=13)
    ax2.set_xlabel("赤方偏移 z", fontsize=11)
    ax2.set_ylabel("|τ(z)|（dimming; τ=ln(extra_dl^2)）", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=9, loc="upper left")

    fig.suptitle("宇宙論（到達限界の整理）：静的背景P最小を観測ε0へ寄せる“必要補正”のz依存", fontsize=14)
    fig.text(
        0.5,
        0.01,
        "Δε_needed=ε0_obs-(-1), extra_dl_factor=(1+z)^Δε_needed, Δμ=5 log10(extra_dl_factor), τ=ln(extra_dl_factor^2)=2Δε ln(1+z)",
        ha="center",
        fontsize=10,
    )
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.92))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Cosmology: distance duality (DDR) constraints and rejection condition.")
    ap.add_argument(
        "--data",
        type=str,
        default=str(_ROOT / "data" / "cosmology" / "distance_duality_constraints.json"),
        help="Input JSON (default: data/cosmology/distance_duality_constraints.json)",
    )
    ap.add_argument(
        "--z-max",
        type=float,
        default=1.6,
        help="Max redshift for plotting η(z) (default: 1.6).",
    )
    ap.add_argument(
        "--reach-z-max",
        type=float,
        default=2.3,
        help="Max redshift for plotting reach-limit curves (default: 2.3).",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    data_path = Path(args.data)
    z_max = float(args.z_max)
    if not (z_max > 0.0):
        raise ValueError("--z-max must be > 0")
    reach_z_max = float(args.reach_z_max)
    if not (reach_z_max > 0.0):
        raise ValueError("--reach-z-max must be > 0")

    src = _read_json(data_path)
    constraints = [Constraint.from_json(c) for c in (src.get("constraints") or [])]
    if not constraints:
        raise SystemExit(f"no constraints found in: {data_path}")

    rows = compute(constraints)

    out_dir = _ROOT / "output" / "private" / "cosmology"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 5.3.5: Fix the P-model minimal DDR relation as a lightweight, machine-readable artifact.
    # (The detailed derivation/assumption audit lives in doc/cosmology/*.md.)
    ddr_relation_json = out_dir / "cosmology_ddr_pmodel_relation.json"
    ddr_relation_payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Step 5.3.5: P-model最小（空間膨張なし＋背景P赤方偏移）でのDDR関係を固定する。",
        "definitions": {
            "z": "1+z ≡ ν_em/ν_obs",
            "D_L": "F_obs ≡ L_em/(4π D_L^2)（bolometric）",
            "D_A": "D_A^2 ≡ dA_em/dΩ_obs",
            "eta": "η(z) ≡ D_L / [(1+z)^2 D_A]",
            "eta_pmodel": "η^(P)(z) ≡ D_L / [(1+z) D_A]",
            "epsilon0_parameterization": "D_L=(1+z)^(2+ε0) D_A  ⇔  η=(1+z)^ε0",
        },
        "models": {
            "FRW_standard": {
                "assumptions": ["FRW幾何（空間膨張）", "光子数保存", "E∝ν（エネルギー比=周波数比）", "time dilation（到着率）"],
                "relation": "D_L = (1+z)^2 D_A",
                "eta": "1",
                "eta_pmodel": "(1+z)",
                "epsilon0": 0.0,
            },
            "P_bg_static_minimal": {
                "assumptions": [
                    "空間膨張なし（静的幾何）",
                    "背景Pで赤方偏移（1+z=P_em/P_obs）",
                    "光子数保存（opacityなし）",
                    "E_obs/E_em = 1/(1+z)",
                    "p_t=1（条件A採用）→ 到着率 1/(1+z)",
                ],
                "relation": "D_L^(P) = (1+z) D_A^(P)",
                "eta": "1/(1+z)",
                "eta_pmodel": "1",
                "epsilon0": -1.0,
            },
        },
        "difference": {
            "delta_epsilon0_pbg_minus_frw": -1.0,
            "delta_eta_factor": "(η_P / η_FRW) = 1/(1+z)",
            "delta_eta_pmodel_factor": "(η^(P)_P / η^(P)_FRW) = 1/(1+z)",
            "note": "差は主に『空間膨張に由来する D_A = D_M/(1+z)』の欠落（静的幾何）から生じる。",
        },
        "docs": {
            "frw_assumptions_ledger": "doc/cosmology/ddr_frw_assumptions_ledger.md",
            "frw_redshift_factor_decomposition": "doc/cosmology/ddr_redshift_factor_decomposition.md",
            "pmodel_assumption_audit": "doc/cosmology/ddr_pmodel_assumption_audit.md",
            "pmodel_axioms_mapping": "doc/cosmology/ddr_pmodel_axioms_mapping.md",
            "pmodel_distance_derivation": "doc/cosmology/ddr_pmodel_distance_derivation.md",
        },
    }
    _write_json(ddr_relation_json, ddr_relation_payload)
    png_path = out_dir / "cosmology_distance_duality_constraints.png"
    _plot(rows, out_png=png_path, z_max=z_max)

    out_json = out_dir / "cosmology_distance_duality_constraints_metrics.json"
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(data_path).replace("\\", "/"),
        "definition": src.get("definition") or {},
        "assumptions": {
            "parameterization": "d_L(z)=(1+z)^(2+ε0) d_A(z)  (=> η=(1+z)^(ε0))",
            "criterion_example": "|ε_obs - ε_model| > 3σ => reject (rule-of-thumb)",
        },
        "rows": rows,
        "params": {"z_max": z_max},
        "outputs": {"png": str(png_path).replace("\\", "/"), "metrics_json": str(out_json).replace("\\", "/")},
    }
    _write_json(out_json, payload)

    reps = _select_representative(rows)

    # Step 5.3.12: Re-evaluate DDR constraints using the P-model specific indicator
    #   η^(P)(z) ≡ D_L / [(1+z) D_A]
    # which is:
    #   FRW: η^(P)=(1+z)   (assuming photon conservation)
    #   P-model minimal: η^(P)=1
    z_ref_eta = 1.0
    eta_p_png = out_dir / "cosmology_ddr_pmodel_eta_constraints.png"
    _plot_eta_pmodel(rows, out_png=eta_p_png, z_max=z_max)

    eta_p_json = out_dir / "cosmology_ddr_pmodel_eta_constraints.json"
    eta_p_payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(data_path).replace("\\", "/"),
        "definition": {
            "eta_pmodel": "η^(P)(z) ≡ D_L / [(1+z) D_A]",
            "relation_to_common_parameterization": "D_L=(1+z)^(2+ε0)D_A ⇒ η^(P)=(1+z)^(1+ε0)",
            "models": {"FRW_photon_conservation": "η^(P)=(1+z)", "P_model_minimal": "η^(P)=1"},
            "z_ref": z_ref_eta,
            "eta_p_obs_z_ref": "(1+z_ref)^(1+ε0_obs)",
            "sigma_eta_p_approx_z_ref": "σ_eta≈ln(1+z_ref)*η*σ_ε0（Gaussian近似）",
        },
        "selection_policy": {
            "representative_bao": "uses_bao=true の中で最小σ（tightest）",
            "representative_no_bao": "uses_bao=false の中で |z_pbg_static| 最小（least rejecting）",
        },
        "representatives": {
            "bao": (None if reps.get("bao") is None else {"id": reps["bao"].get("id"), "short_label": reps["bao"].get("short_label")}),
            "no_bao": (
                None if reps.get("no_bao") is None else {"id": reps["no_bao"].get("id"), "short_label": reps["no_bao"].get("short_label")}
            ),
        },
        "rows": [
            {
                "id": r.get("id"),
                "short_label": r.get("short_label"),
                "uses_bao": r.get("uses_bao"),
                "epsilon0_obs": r.get("epsilon0_obs"),
                "epsilon0_sigma": r.get("epsilon0_sigma"),
                "eta_p_exponent_obs": r.get("eta_p_exponent_obs"),
                "eta_p_exponent_sigma": r.get("eta_p_exponent_sigma"),
                "eta_p_obs_z1": r.get("eta_p_obs_z1"),
                "eta_p_sigma_approx_z1": r.get("eta_p_sigma_approx_z1"),
                "eta_p_pred_frw_z1": r.get("eta_p_pred_frw_z1"),
                "eta_p_pred_pbg_static_z1": r.get("eta_p_pred_pbg_static_z1"),
                "z_eta_frw_z1": r.get("z_eta_frw_z1"),
                "z_eta_pbg_static_z1": r.get("z_eta_pbg_static_z1"),
                "sigma_note": r.get("sigma_note"),
                "source": r.get("source"),
            }
            for r in rows
            if isinstance(r, dict)
        ],
        "outputs": {"png": str(eta_p_png).replace("\\", "/"), "metrics_json": str(eta_p_json).replace("\\", "/")},
        "docs": {
            "frw_redshift_factor_decomposition": "doc/cosmology/ddr_redshift_factor_decomposition.md",
            "pmodel_axioms_mapping": "doc/cosmology/ddr_pmodel_axioms_mapping.md",
            "pmodel_distance_derivation": "doc/cosmology/ddr_pmodel_distance_derivation.md",
        },
        "notes": [
            "η^(P) は P-model側の検証枠組みとして便利だが、ここでの数値は一次ソースが採用する距離指標推定の前提（例：BAO圧縮出力、クラスター幾何モデル、SN標準化）に依存する。",
            "従って、ここでのzスコアは『その前提を固定した比較』であり、距離指標の再導出（前提監査）とセットで解釈する。",
        ],
    }

    # Step 5.3.16: Freeze a human-readable comparison table in the η^(P) space.
    eta_p_table_md = out_dir / "cosmology_ddr_pmodel_eta_constraints_table.md"

    def _fmt_pm(x: Any, s: Any, *, digits: int = 3) -> str:
        vx = _safe_float(x)
        vs = _safe_float(s)
        if vx is None or vs is None:
            return "n/a"
        return f"{vx:.{digits}f}±{vs:.{digits}f}"

    def _fmt_z(x: Any, *, digits: int = 2) -> str:
        vx = _safe_float(x)
        if vx is None:
            return "n/a"
        return f"{vx:+.{digits}f}"

    eta_rows = eta_p_payload.get("rows") if isinstance(eta_p_payload.get("rows"), list) else []
    table_lines = [
        "| 公表制約 | uses_BAO | ε0（1σ） | p=1+ε0（1σ） | η^(P)(z=1) | z(FRW) | z(P-model最小) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in eta_rows:
        if not isinstance(r, dict):
            continue
        short = str(r.get("short_label") or "").strip() or str(r.get("id") or "").strip()
        uses_bao = 1 if bool(r.get("uses_bao")) else 0
        eps = _fmt_pm(r.get("epsilon0_obs"), r.get("epsilon0_sigma"))
        p = _fmt_pm(r.get("eta_p_exponent_obs"), r.get("eta_p_exponent_sigma"))
        eta_p = _fmt_pm(r.get("eta_p_obs_z1"), r.get("eta_p_sigma_approx_z1"))
        z_frw = _fmt_z(r.get("z_eta_frw_z1"))
        z_p = _fmt_z(r.get("z_eta_pbg_static_z1"))
        table_lines.append(f"| {short} | {uses_bao} | {eps} | {p} | {eta_p} | {z_frw} | {z_p} |")

    eta_p_table_md.parent.mkdir(parents=True, exist_ok=True)
    eta_p_table_md.write_text("\n".join(table_lines) + "\n", encoding="utf-8")

    eta_p_payload["outputs"]["table_md"] = str(eta_p_table_md).replace("\\", "/")
    _write_json(eta_p_json, eta_p_payload)

    # Step 5.3.14: Falsification pack (operational thresholds) in the η^(P) framework.
    fals_json = out_dir / "cosmology_ddr_pmodel_falsification_pack.json"
    fals_payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "definition": {
            "eta_pmodel": "η^(P)(z) ≡ D_L / [(1+z) D_A]",
            "model_predictions": {"FRW_photon_conservation": "η^(P)=(1+z)", "P_model_minimal": "η^(P)=1"},
            "equivalent_exponent": "p ≡ 1+ε0  （η^(P)=(1+z)^p）",
        },
        "decision_rule_examples": [
            "距離指標推定が『FRWの膨張（D_A=D_M/(1+z)）を前提に埋め込んでいない』ことを監査したうえで、η^(P) を評価する。",
            "同一の推定I/F（入力・選別・校正）を固定した複数の独立データセットで、η^(P)=1（P-model最小）からの乖離が 3σ を超えて一貫するなら、P-model最小は棄却される。",
            "逆に、η^(P)=(1+z)（FRW+光子保存）からの乖離が 3σ を超えて一貫するなら、光子保存（または距離推定前提）が破れている可能性が高い（opacity/系統/定義差の切り分けへ進む）。",
        ],
        "precision_targets": {
            "exponent_space": {
                "delta_p_between_models": 1.0,
                "sigma_p_required_for_3sigma_separation": 1.0 / 3.0,
                "sigma_p_required_for_5sigma_separation": 1.0 / 5.0,
                "note": "p=1（FRW）と p=0（P-model最小）の識別に必要な精度（単純な目安）。",
            },
            "eta_space_examples": [
                {
                    "z_ref": z_ref_eta,
                    "delta_eta_between_models": float(z_ref_eta),
                    "sigma_eta_required_for_3sigma_separation": float(z_ref_eta) / 3.0,
                    "sigma_eta_required_for_5sigma_separation": float(z_ref_eta) / 5.0,
                    "note": "η^(P)_FRW-(η^(P)_P)= (1+z_ref)-1 = z_ref を用いた目安。",
                },
                {"z_ref": 0.5, "delta_eta_between_models": 0.5, "sigma_eta_required_for_3sigma_separation": 0.5 / 3.0},
                {"z_ref": 2.0, "delta_eta_between_models": 2.0, "sigma_eta_required_for_3sigma_separation": 2.0 / 3.0},
            ],
        },
        "representative_constraints": {
            k: (
                None
                if not isinstance(v, dict)
                else {
                    "id": v.get("id"),
                    "short_label": v.get("short_label"),
                    "epsilon0_obs": v.get("epsilon0_obs"),
                    "epsilon0_sigma": v.get("epsilon0_sigma"),
                    "p_obs": v.get("eta_p_exponent_obs"),
                    "p_sigma": v.get("eta_p_exponent_sigma"),
                    "eta_p_obs_z1": v.get("eta_p_obs_z1"),
                    "eta_p_sigma_approx_z1": v.get("eta_p_sigma_approx_z1"),
                    "z_vs_pmodel_in_epsilon_space": v.get("z_pbg_static"),
                    "z_vs_frw_in_epsilon_space": v.get("z_frw"),
                    "z_vs_pmodel_in_eta_space_z1": v.get("z_eta_pbg_static_z1"),
                    "z_vs_frw_in_eta_space_z1": v.get("z_eta_frw_z1"),
                    "source": v.get("source"),
                }
            )
            for k, v in reps.items()
        },
        "related_outputs": {
            "eta_p_constraints_png": str(eta_p_png).replace("\\", "/"),
            "eta_p_constraints_json": str(eta_p_json).replace("\\", "/"),
            "ddr_constraints_png": str(png_path).replace("\\", "/"),
            "ddr_constraints_metrics": str(out_json).replace("\\", "/"),
        },
        "notes": [
            "ここでの反証条件は『距離推定が前提を埋め込んでいない（循環でない）』ことを要請するため、最終的には距離指標の再導出（標準化/校正/系統の監査）と一体で固定する必要がある。",
            "一方で、p_t（時間伸長）や T(z) 等の独立プローブを固定したうえで、距離指標側の前提が許容できるかを詰める、という役割分担は維持する。",
        ],
    }
    _write_json(fals_json, fals_payload)

    # Step 5.3.6: reinterpret a representative ε0 constraint under the new DDR relation
    # (P-model minimal predicts η=1/(1+z) ⇔ ε0=-1 in the common parameterization).
    target_id = "martinelli2021_snIa_bao"
    target_row = next((r for r in rows if isinstance(r, dict) and str(r.get("id") or "") == target_id), None)
    if target_row is None:
        target_row = reps.get("bao") if isinstance(reps.get("bao"), dict) else None
    if target_row is None and rows:
        target_row = rows[0]

    eps_reinterp_json = out_dir / "cosmology_ddr_epsilon_reinterpretation.json"
    eps_reinterp_payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Step 5.3.6: 新しいDDR関係（P-model最小：ε0=-1）で、代表ε0制約を再解釈する。",
        "target_selection": {
            "preferred_id": target_id,
            "fallback_policy": "preferred_idが無い場合は uses_bao=true の最小σ（representative_bao）→先頭行。",
        },
        "target_constraint": (
            None
            if not isinstance(target_row, dict)
            else {
                "id": str(target_row.get("id") or ""),
                "short_label": str(target_row.get("short_label") or ""),
                "title": str(target_row.get("title") or ""),
                "uses_bao": bool(target_row.get("uses_bao", False)),
                "epsilon0_obs": target_row.get("epsilon0_obs"),
                "epsilon0_sigma": target_row.get("epsilon0_sigma"),
                "source": target_row.get("source") if isinstance(target_row.get("source"), dict) else {},
            }
        ),
        "models": {
            "FRW_standard": {"epsilon0": 0.0, "eta": "1"},
            "P_bg_static_minimal": {"epsilon0": -1.0, "eta": "1/(1+z)"},
        },
        "reinterpretation": (
            None
            if not isinstance(target_row, dict)
            else {
                "z_score_vs_pbg_static": target_row.get("z_pbg_static"),
                "delta_eps_needed_to_match_obs": target_row.get("epsilon0_extra_needed_to_match_obs"),
                "extra_eta_factor_needed_z1": target_row.get("extra_eta_factor_needed_z1"),
                "extra_dl_factor_needed_z1": target_row.get("extra_dl_factor_needed_z1"),
                "delta_distance_modulus_mag_z1": target_row.get("delta_distance_modulus_mag_z1"),
                "tau_equivalent_dimming_z1": target_row.get("tau_equivalent_dimming_z1"),
                "note": "Δε_needed=ε0_obs-(-1) を『有効不透明度/標準光源・標準定規の系統/単位スケール変化』などへ割り振る解釈は一意ではない（Step 4.7 の再接続条件へ接続）。",
            }
        ),
        "related_outputs": {
            "ddr_relation_json": str(ddr_relation_json).replace("\\", "/"),
            "ddr_constraints_metrics": str(out_json).replace("\\", "/"),
            "ddr_constraints_png": str(png_path).replace("\\", "/"),
            "reach_limit_metrics": str(
                (_ROOT / "output" / "private" / "cosmology" / "cosmology_distance_indicator_reach_limit_metrics.json")
            ).replace("\\", "/"),
            "reconnection_conditions_metrics": str(
                (_ROOT / "output" / "private" / "cosmology" / "cosmology_ddr_reconnection_conditions_metrics.json")
            ).replace("\\", "/"),
        },
        "docs": {
            "frw_assumptions_ledger": "doc/cosmology/ddr_frw_assumptions_ledger.md",
            "pmodel_assumption_audit": "doc/cosmology/ddr_pmodel_assumption_audit.md",
            "pmodel_distance_derivation": "doc/cosmology/ddr_pmodel_distance_derivation.md",
        },
        "notes": [
            "この再解釈は、ε0 が同一のパラメータ化（η=(1+z)^ε0）で比較できる、という条件付きの整理である。",
            "観測側の D_L/D_A 推定（SNe/BAO/クラスター等）は多数の前提を含むため、最終的な棄却/保留は Step 4.7 の再接続条件（α, s_L, s_R など）と整合して評価する。",
        ],
    }
    _write_json(eps_reinterp_json, eps_reinterp_payload)

    budgets_mag = [0.05, 0.1, 0.2, 0.5, 1.0]
    z_refs = [0.1, 0.5, 1.0, 2.0]
    z_refs = [z for z in z_refs if z <= reach_z_max]

    reach_png = out_dir / "cosmology_distance_indicator_reach_limit.png"
    _plot_reach_limit(reps=reps, out_png=reach_png, z_max=reach_z_max, budgets_mag=budgets_mag)

    reach_json = out_dir / "cosmology_distance_indicator_reach_limit_metrics.json"
    reach_payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(data_path).replace("\\", "/"),
        "definition": {
            "delta_eps_needed": "Δε_needed = ε0_obs - ε0_model（ここでは ε0_model=-1）",
            "extra_dl_factor": "extra_dl_factor=(1+z)^Δε_needed（静的背景P最小の d_L を観測ε0へ寄せる倍率）",
            "delta_mu": "Δμ(z)=5 log10(extra_dl_factor)",
            "tau_equivalent": "τ(z)=ln(extra_dl_factor^2)=2Δε ln(1+z)（不透明度解釈の等価減光）",
        },
        "selection_policy": {
            "representative_bao": "uses_bao=true の中で最小σ（tightest）",
            "representative_no_bao": "uses_bao=false の中で |z_pbg_static| 最小（least rejecting）",
        },
        "representatives": {
            "bao": (None if reps.get("bao") is None else {"id": reps["bao"].get("id"), "short_label": reps["bao"].get("short_label")}),
            "no_bao": (
                None if reps.get("no_bao") is None else {"id": reps["no_bao"].get("id"), "short_label": reps["no_bao"].get("short_label")}
            ),
        },
        "budgets_mag": budgets_mag,
        "z_refs": z_refs,
        "reach": {
            k: (_build_reach_metrics(label=k, row=v, budgets_mag=budgets_mag, z_refs=z_refs) if isinstance(v, dict) else None)
            for k, v in reps.items()
        },
        "params": {"z_max_eta_plot": z_max, "reach_z_max": reach_z_max},
        "outputs": {"png": str(reach_png).replace("\\", "/"), "metrics_json": str(reach_json).replace("\\", "/")},
        "notes": [
            "ここでの「到達限界」は、距離指標の前提（標準化/校正/系統）に吸収できる“許容補正”を仮定したとき、どのzまで静的背景P最小が隠れ得るかを定量化するための整理。",
            "Δμの予算（0.05〜1.0 mag）は“参考の目盛り”であり、実際の系統誤差を主張するものではない。",
        ],
    }
    _write_json(reach_json, reach_payload)

    print(f"[ok] png : {png_path}")
    print(f"[ok] json: {out_json}")
    print(f"[ok] ddr_relation_json: {ddr_relation_json}")
    print(f"[ok] eps_reinterp_json: {eps_reinterp_json}")
    print(f"[ok] reach_png : {reach_png}")
    print(f"[ok] reach_json: {reach_json}")
    print(f"[ok] eta_p_png : {eta_p_png}")
    print(f"[ok] eta_p_json: {eta_p_json}")
    print(f"[ok] eta_p_table_md: {eta_p_table_md}")
    print(f"[ok] fals_json: {fals_json}")

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_distance_duality_constraints",
                "argv": list(sys.argv),
                "inputs": {"data": data_path},
                "outputs": {
                    "png": png_path,
                    "metrics_json": out_json,
                    "ddr_relation_json": ddr_relation_json,
                    "eps_reinterp_json": eps_reinterp_json,
                    "reach_png": reach_png,
                    "reach_json": reach_json,
                    "eta_p_png": eta_p_png,
                    "eta_p_json": eta_p_json,
                    "eta_p_table_md": eta_p_table_md,
                    "eta_p_falsification_pack_json": fals_json,
                },
                "metrics": {"z_max": z_max, "reach_z_max": reach_z_max, "n_constraints": len(rows)},
            }
        )
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
