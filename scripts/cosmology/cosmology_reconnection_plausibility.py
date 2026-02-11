#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_reconnection_plausibility.py

Step 14.2.2（一次ソースで拘束を追加）:
静的背景P（最小: ε0=-1）と、距離二重性（DDR）の一次ソース制約（ε0_obs）との差を
「再接続に必要な補正量 Δε」として定量化し、
それを (i) 不透明度 α、(ii) 標準光源進化 s_L、(iii) 標準定規（BAO）スケール の
一次ソース拘束と比較する。

この図は「どれか1つの機構だけ」でΔεを埋める場合の必要量を示す（組合せは別図）。

入力（固定）:
  - data/cosmology/distance_duality_constraints.json
  - data/cosmology/cosmic_opacity_constraints.json
  - data/cosmology/sn_standard_candle_evolution_constraints.json
  - data/cosmology/bao_sound_horizon_constraints.json

出力（固定名）:
  - output/cosmology/cosmology_reconnection_plausibility.png
  - output/cosmology/cosmology_reconnection_plausibility_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, replace
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


def _maybe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return float(v)


def _load_ddr_systematics_envelope(path: Path) -> Dict[str, Dict[str, Any]]:
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


def _apply_ddr_sigma_policy(ddr: "DDRConstraint", *, policy: str, envelope: Dict[str, Dict[str, Any]]) -> "DDRConstraint":
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


def _fmt_float(x: Optional[float], *, digits: int = 6) -> str:
    if x is None:
        return ""
    if x == 0.0:
        return "0"
    ax = abs(x)
    if ax >= 1e4 or ax < 1e-3:
        return f"{x:.{digits}g}"
    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


@dataclass(frozen=True)
class DDRConstraint:
    id: str
    short_label: str
    title: str
    epsilon0: float
    epsilon0_sigma: float
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
            source=dict(j.get("source") or {}),
            epsilon0_sigma_raw=sigma,
        )


@dataclass(frozen=True)
class OpacityConstraint:
    id: str
    short_label: str
    title: str
    alpha_opacity: float
    alpha_opacity_sigma: float
    sigma_note: str
    source: Dict[str, Any]

    @staticmethod
    def from_json(j: Dict[str, Any]) -> "OpacityConstraint":
        return OpacityConstraint(
            id=str(j["id"]),
            short_label=str(j.get("short_label") or j["id"]),
            title=str(j.get("title") or j.get("short_label") or j["id"]),
            alpha_opacity=float(j["alpha_opacity"]),
            alpha_opacity_sigma=float(j["alpha_opacity_sigma"]),
            sigma_note=str(j.get("sigma_note") or ""),
            source=dict(j.get("source") or {}),
        )


@dataclass(frozen=True)
class CandleEvoConstraint:
    id: str
    short_label: str
    title: str
    s_L: float
    s_L_sigma: float
    assumes_cddr: Optional[bool]
    sigma_note: str
    source: Dict[str, Any]

    @staticmethod
    def from_json(j: Dict[str, Any]) -> "CandleEvoConstraint":
        assumes_cddr = j.get("assumes_cddr")
        return CandleEvoConstraint(
            id=str(j["id"]),
            short_label=str(j.get("short_label") or j["id"]),
            title=str(j.get("title") or j.get("short_label") or j["id"]),
            s_L=float(j["s_L"]),
            s_L_sigma=float(j["s_L_sigma"]),
            assumes_cddr=(None if assumes_cddr is None else bool(assumes_cddr)),
            sigma_note=str(j.get("sigma_note") or ""),
            source=dict(j.get("source") or {}),
        )


@dataclass(frozen=True)
class BAOConstraint:
    id: str
    short_label: str
    title: str
    r_drag_mpc: float
    r_drag_sigma_mpc: float
    sigma_note: str
    source: Dict[str, Any]

    @staticmethod
    def from_json(j: Dict[str, Any]) -> "BAOConstraint":
        return BAOConstraint(
            id=str(j["id"]),
            short_label=str(j.get("short_label") or j["id"]),
            title=str(j.get("title") or j.get("short_label") or j["id"]),
            r_drag_mpc=float(j["r_drag_mpc"]),
            r_drag_sigma_mpc=float(j["r_drag_sigma_mpc"]),
            sigma_note=str(j.get("sigma_note") or ""),
            source=dict(j.get("source") or {}),
        )


def _primary_ddr(ddr_rows: Sequence[DDRConstraint]) -> DDRConstraint:
    if not ddr_rows:
        raise ValueError("no DDR constraints")
    return min(ddr_rows, key=lambda r: (r.epsilon0_sigma if r.epsilon0_sigma > 0 else float("inf")))


def _primary_by_sigma(rows: Sequence[Any], sigma_field: str) -> Any:
    """Pick the most constraining row (smallest positive sigma)."""
    if not rows:
        raise ValueError("empty constraint list")
    best = None
    best_sig = float("inf")
    for r in rows:
        try:
            sig = float(getattr(r, sigma_field))
        except Exception:
            continue
        if not (sig > 0):
            continue
        if sig < best_sig:
            best_sig = sig
            best = r
    return best if best is not None else rows[0]


def _z_score(x: float, mu: float, sig: float) -> Optional[float]:
    if not (sig > 0):
        return None
    return (x - mu) / sig


def _interpret_z1(delta_eps: float) -> Dict[str, float]:
    one_p_z = 2.0
    eta_boost = one_p_z**delta_eps
    delta_mu = 5.0 * math.log10(eta_boost) if eta_boost > 0 else float("nan")
    flux_dimming_factor = eta_boost**2
    tau_equiv = math.log(flux_dimming_factor) if flux_dimming_factor > 0 else float("nan")
    return {
        "z_ref": 1.0,
        "eta_boost_factor_z1": float(eta_boost),
        "delta_distance_modulus_mag_z1": float(delta_mu),
        "flux_dimming_factor_needed_z1": float(flux_dimming_factor),
        "tau_equivalent_dimming_z1": float(tau_equiv),
    }


def _plot(
    *,
    out_png: Path,
    ddr: DDRConstraint,
    opacity_rows: Sequence[OpacityConstraint],
    candle_rows: Sequence[CandleEvoConstraint],
    bao_rows: Sequence[BAOConstraint],
    eps_min_model: float,
) -> Dict[str, Any]:
    eps_obs = float(ddr.epsilon0)
    eps_sig = float(ddr.epsilon0_sigma)
    ddr_sigma_suffix = ""
    if str(getattr(ddr, "sigma_policy", "raw")) == "category_sys":
        ddr_sigma_suffix = (
            f"（σ_cat≈{_fmt_float(ddr.sigma_sys_category, digits=3)}, rawσ≈{_fmt_float(ddr.epsilon0_sigma_raw, digits=3)}）"
        )

    delta_eps_needed = eps_obs - eps_min_model
    z1 = _interpret_z1(delta_eps_needed)
    eta_boost_z1 = float(z1["eta_boost_factor_z1"])

    # Required parameters if each mechanism alone explains Δε.
    alpha_req = delta_eps_needed
    s_L_req = -2.0 * delta_eps_needed
    s_R_req = delta_eps_needed

    opacity_sorted = sorted(
        list(opacity_rows),
        key=lambda r: (r.alpha_opacity_sigma if r.alpha_opacity_sigma > 0 else float("inf")),
    )
    candle_sorted = sorted(
        list(candle_rows),
        key=lambda r: (r.s_L_sigma if r.s_L_sigma > 0 else float("inf")),
    )
    bao_sorted = sorted(
        list(bao_rows),
        key=lambda r: (r.r_drag_sigma_mpc if r.r_drag_sigma_mpc > 0 else float("inf")),
    )

    primary_opacity = _primary_by_sigma(opacity_sorted, "alpha_opacity_sigma")
    candle_no_cddr = [r for r in candle_sorted if r.assumes_cddr is not True]
    primary_candle = _primary_by_sigma(candle_no_cddr or candle_sorted, "s_L_sigma")
    primary_bao = _primary_by_sigma(bao_sorted, "r_drag_sigma_mpc")

    z_alpha_primary = _z_score(alpha_req, float(primary_opacity.alpha_opacity), float(primary_opacity.alpha_opacity_sigma))
    z_sL_primary = _z_score(s_L_req, float(primary_candle.s_L), float(primary_candle.s_L_sigma))

    # BAO ruler: interpret as requiring r_drag to scale by 2^(s_R) at z=1.
    r_drag_obs = float(primary_bao.r_drag_mpc)
    r_drag_sig = float(primary_bao.r_drag_sigma_mpc)
    r_drag_req_z1 = r_drag_obs * (2.0**s_R_req)
    z_rdrag_primary = _z_score(r_drag_req_z1, r_drag_obs, r_drag_sig)

    _set_japanese_font()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.2))

    # 1) Opacity α
    ax = axes[0]
    y = np.arange(len(opacity_sorted), dtype=float)
    op_mu = np.array([float(r.alpha_opacity) for r in opacity_sorted], dtype=float)
    op_sig = np.array([float(r.alpha_opacity_sigma) for r in opacity_sorted], dtype=float)
    op_labels = [str(r.short_label or r.id) for r in opacity_sorted]
    ax.errorbar(
        op_mu,
        y,
        xerr=np.where(np.isfinite(op_sig) & (op_sig > 0), op_sig, 0.0),
        fmt="o",
        capsize=4,
        color="#1f77b4",
        ecolor="#1f77b4",
        label="一次ソース拘束（複数）",
    )
    ax.axvline(alpha_req, color="#d62728", linewidth=1.5, label="DDR再接続に必要（単独）")
    ax.set_title("不透明度 α（η=(1+z)^α）", fontsize=12)
    ax.set_xlabel("α", fontsize=11)
    ax.set_yticks(y)
    ax.set_yticklabels(op_labels, fontsize=9)
    ax.invert_yaxis()
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")
    ax.legend(fontsize=9, loc="upper left")
    if z_alpha_primary is not None:
        ax.text(
            0.02,
            0.08,
            (
                f"必要 α={_fmt_float(alpha_req, digits=3)}\n"
                f"代表拘束（最小σ）: {primary_opacity.short_label}\n"
                f"α={_fmt_float(primary_opacity.alpha_opacity, digits=3)}±{_fmt_float(primary_opacity.alpha_opacity_sigma, digits=3)}\n"
                f"z={_fmt_float(z_alpha_primary, digits=3)}"
            ),
            transform=ax.transAxes,
            fontsize=10,
            va="bottom",
        )

    # 2) Standard candle evolution s_L
    ax = axes[1]
    y = np.arange(len(candle_sorted), dtype=float)
    cd_mu = np.array([float(r.s_L) for r in candle_sorted], dtype=float)
    cd_sig = np.array([float(r.s_L_sigma) for r in candle_sorted], dtype=float)
    cd_labels = [str(r.short_label or r.id) for r in candle_sorted]
    ax.errorbar(
        cd_mu,
        y,
        xerr=np.where(np.isfinite(cd_sig) & (cd_sig > 0), cd_sig, 0.0),
        fmt="o",
        capsize=4,
        color="#1f77b4",
        ecolor="#1f77b4",
        label="一次ソース拘束（複数）",
    )
    ax.axvline(s_L_req, color="#d62728", linewidth=1.5, label="DDR再接続に必要（単独）")
    ax.set_title("標準光源進化 s_L（L∝(1+z)^{s_L}）", fontsize=12)
    ax.set_xlabel("s_L", fontsize=11)
    ax.set_yticks(y)
    ax.set_yticklabels(cd_labels, fontsize=9)
    ax.invert_yaxis()
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")
    ax.legend(fontsize=9, loc="upper left")
    if z_sL_primary is not None:
        ax.text(
            0.02,
            0.08,
            (
                f"必要 s_L={_fmt_float(s_L_req, digits=3)}\n"
                f"代表拘束（最小σ）: {primary_candle.short_label}\n"
                f"s_L={_fmt_float(primary_candle.s_L, digits=3)}±{_fmt_float(primary_candle.s_L_sigma, digits=3)}"
                f"{'（CDDR仮定）' if (primary_candle.assumes_cddr is True) else ''}\n"
                f"z={_fmt_float(z_sL_primary, digits=3)}"
            ),
            transform=ax.transAxes,
            fontsize=10,
            va="bottom",
        )

    # 3) BAO ruler r_drag
    ax = axes[2]
    y = np.arange(len(bao_sorted), dtype=float)
    bao_mu = np.array([float(r.r_drag_mpc) for r in bao_sorted], dtype=float)
    bao_sig = np.array([float(r.r_drag_sigma_mpc) for r in bao_sorted], dtype=float)
    bao_labels = [str(r.short_label or r.id) for r in bao_sorted]
    ax.errorbar(
        bao_mu,
        y,
        xerr=np.where(np.isfinite(bao_sig) & (bao_sig > 0), bao_sig, 0.0),
        fmt="o",
        capsize=4,
        color="#1f77b4",
        ecolor="#1f77b4",
        label="一次ソース拘束（複数）",
    )
    ax.axvline(r_drag_req_z1, color="#d62728", linewidth=1.5, label="DDR再接続に必要（z=1換算）")
    ax.set_title("標準定規 r_drag（Mpc）", fontsize=12)
    ax.set_xlabel("r_drag [Mpc]", fontsize=11)
    ax.set_yticks(y)
    ax.set_yticklabels(bao_labels, fontsize=9)
    ax.invert_yaxis()
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")
    ax.legend(fontsize=9, loc="upper left")
    if z_rdrag_primary is not None:
        ax.text(
            0.02,
            0.08,
            (
                f"必要 r_drag(z=1)={_fmt_float(r_drag_req_z1, digits=4)}\n"
                f"代表拘束（最小σ）: {primary_bao.short_label}\n"
                f"r_drag={_fmt_float(r_drag_obs, digits=4)}±{_fmt_float(r_drag_sig, digits=3)}\n"
                f"倍率 2^{_fmt_float(s_R_req, digits=3)}={_fmt_float(eta_boost_z1, digits=3)}\n"
                f"z={_fmt_float(z_rdrag_primary, digits=3)}"
            ),
            transform=ax.transAxes,
            fontsize=10,
            va="bottom",
        )

    fig.suptitle("宇宙論（Step 14.2.2）：DDR再接続に必要な補正量と一次ソース拘束の比較", fontsize=14)
    fig.text(
        0.5,
        0.01,
        (
            f"DDR一次ソース（{ddr.short_label}）: ε0={_fmt_float(eps_obs, digits=3)}±{_fmt_float(eps_sig, digits=3)}{ddr_sigma_suffix}. "
            f"静的背景P最小 ε0={_fmt_float(eps_min_model, digits=3)} → 必要 Δε={_fmt_float(delta_eps_needed, digits=3)} "
            f"（z=1で D_L×{_fmt_float(eta_boost_z1, digits=2)}, Δμ={_fmt_float(z1['delta_distance_modulus_mag_z1'], digits=2)} mag, τ={_fmt_float(z1['tau_equivalent_dimming_z1'], digits=2)}）"
        ),
        ha="center",
        fontsize=10,
    )
    plt.tight_layout(rect=(0.0, 0.04, 1.0, 0.92))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    def _z_scores_opacity() -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for r in opacity_sorted:
            z = _z_score(alpha_req, float(r.alpha_opacity), float(r.alpha_opacity_sigma))
            out[str(r.id)] = None if z is None else float(z)
        return out

    def _z_scores_candle() -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for r in candle_sorted:
            z = _z_score(s_L_req, float(r.s_L), float(r.s_L_sigma))
            out[str(r.id)] = None if z is None else float(z)
        return out

    def _z_scores_bao() -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for r in bao_sorted:
            z = _z_score(float(r.r_drag_mpc) * (2.0**s_R_req), float(r.r_drag_mpc), float(r.r_drag_sigma_mpc))
            out[str(r.id)] = None if z is None else float(z)
        return out

    return {
        "ddr_primary": {
            "id": ddr.id,
            "short_label": ddr.short_label,
            "epsilon0_obs": eps_obs,
            "epsilon0_sigma": eps_sig,
            "epsilon0_sigma_raw": float(ddr.epsilon0_sigma_raw),
            "sigma_sys_category": ddr.sigma_sys_category,
            "sigma_policy": str(ddr.sigma_policy),
            "category": ddr.category,
        },
        "eps_min_model": float(eps_min_model),
        "delta_eps_needed": float(delta_eps_needed),
        "z1_interpretation": z1,
        "required_single_mechanism": {
            "opacity_only": {"alpha_opacity": float(alpha_req)},
            "candle_only": {"s_L": float(s_L_req)},
            "ruler_only": {"s_R": float(s_R_req), "r_drag_required_z1_mpc": float(r_drag_req_z1)},
        },
        "constraints_used": {
            "opacity_primary": {
                "id": primary_opacity.id,
                "alpha_opacity": float(primary_opacity.alpha_opacity),
                "alpha_opacity_sigma": float(primary_opacity.alpha_opacity_sigma),
                "sigma_note": primary_opacity.sigma_note,
                "source": primary_opacity.source,
            },
            "opacity_all": [
                {
                    "id": r.id,
                    "short_label": r.short_label,
                    "alpha_opacity": float(r.alpha_opacity),
                    "alpha_opacity_sigma": float(r.alpha_opacity_sigma),
                    "sigma_note": r.sigma_note,
                    "source": r.source,
                }
                for r in opacity_sorted
            ],
            "candle_primary": {
                "id": primary_candle.id,
                "s_L": float(primary_candle.s_L),
                "s_L_sigma": float(primary_candle.s_L_sigma),
                "assumes_cddr": (None if primary_candle.assumes_cddr is None else bool(primary_candle.assumes_cddr)),
                "sigma_note": primary_candle.sigma_note,
                "source": primary_candle.source,
            },
            "candle_all": [
                {
                    "id": r.id,
                    "short_label": r.short_label,
                    "s_L": float(r.s_L),
                    "s_L_sigma": float(r.s_L_sigma),
                    "assumes_cddr": (None if r.assumes_cddr is None else bool(r.assumes_cddr)),
                    "sigma_note": r.sigma_note,
                    "source": r.source,
                }
                for r in candle_sorted
            ],
            "bao_primary": {
                "id": primary_bao.id,
                "r_drag_mpc": float(primary_bao.r_drag_mpc),
                "r_drag_sigma_mpc": float(primary_bao.r_drag_sigma_mpc),
                "sigma_note": primary_bao.sigma_note,
                "source": primary_bao.source,
            },
            "bao_all": [
                {
                    "id": r.id,
                    "short_label": r.short_label,
                    "r_drag_mpc": float(r.r_drag_mpc),
                    "r_drag_sigma_mpc": float(r.r_drag_sigma_mpc),
                    "sigma_note": r.sigma_note,
                    "source": r.source,
                }
                for r in bao_sorted
            ],
        },
        "z_scores": {
            "opacity_only_alpha_primary": (None if z_alpha_primary is None else float(z_alpha_primary)),
            "candle_only_s_L_primary": (None if z_sL_primary is None else float(z_sL_primary)),
            "ruler_only_r_drag_z1_primary": (None if z_rdrag_primary is None else float(z_rdrag_primary)),
            "opacity_only_alpha_all": _z_scores_opacity(),
            "candle_only_s_L_all": _z_scores_candle(),
            "ruler_only_r_drag_z1_all": _z_scores_bao(),
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Cosmology: reconnection plausibility vs external constraints (Step 14.2.2).")
    ap.add_argument(
        "--ddr",
        type=str,
        default=str(_ROOT / "data" / "cosmology" / "distance_duality_constraints.json"),
        help="DDR constraints JSON (default: data/cosmology/distance_duality_constraints.json)",
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
    ap.add_argument(
        "--opacity",
        type=str,
        default=str(_ROOT / "data" / "cosmology" / "cosmic_opacity_constraints.json"),
        help="Opacity constraints JSON (default: data/cosmology/cosmic_opacity_constraints.json)",
    )
    ap.add_argument(
        "--candle",
        type=str,
        default=str(_ROOT / "data" / "cosmology" / "sn_standard_candle_evolution_constraints.json"),
        help="Standard candle evolution constraints JSON (default: data/cosmology/sn_standard_candle_evolution_constraints.json)",
    )
    ap.add_argument(
        "--bao",
        type=str,
        default=str(_ROOT / "data" / "cosmology" / "bao_sound_horizon_constraints.json"),
        help="BAO sound horizon constraints JSON (default: data/cosmology/bao_sound_horizon_constraints.json)",
    )
    ap.add_argument(
        "--eps-min-model",
        type=float,
        default=-1.0,
        help="Minimal static background-P prediction for ε0 (default: -1.0).",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    ddr_path = Path(args.ddr)
    opacity_path = Path(args.opacity)
    candle_path = Path(args.candle)
    bao_path = Path(args.bao)
    eps_min_model = float(args.eps_min_model)

    ddr_src = _read_json(ddr_path)
    ddr_sigma_policy = str(args.ddr_sigma_policy)
    ddr_env_path = _ROOT / "output" / "cosmology" / "cosmology_distance_duality_systematics_envelope_metrics.json"
    ddr_env = _load_ddr_systematics_envelope(ddr_env_path) if ddr_sigma_policy == "category_sys" else {}
    ddr_rows = [
        _apply_ddr_sigma_policy(DDRConstraint.from_json(c), policy=ddr_sigma_policy, envelope=ddr_env)
        for c in (ddr_src.get("constraints") or [])
    ]
    applied_ddr_sigma_count = len([c for c in ddr_rows if c.sigma_policy == "category_sys"])
    ddr_sigma_policy_meta = {
        "policy": ddr_sigma_policy,
        "envelope_metrics": (str(ddr_env_path).replace("\\", "/") if ddr_sigma_policy == "category_sys" else None),
        "applied_count": applied_ddr_sigma_count,
        "note": "If the envelope file is missing, σ_cat inflation is skipped for all rows (falls back to raw).",
    }
    ddr_primary = _primary_ddr(ddr_rows)

    op_src = _read_json(opacity_path)
    op_rows = [OpacityConstraint.from_json(c) for c in (op_src.get("constraints") or [])]
    if not op_rows:
        raise SystemExit(f"no opacity constraints found in: {opacity_path}")

    cd_src = _read_json(candle_path)
    cd_rows = [CandleEvoConstraint.from_json(c) for c in (cd_src.get("constraints") or [])]
    if not cd_rows:
        raise SystemExit(f"no candle evolution constraints found in: {candle_path}")

    bao_src = _read_json(bao_path)
    bao_rows = [BAOConstraint.from_json(c) for c in (bao_src.get("constraints") or [])]
    if not bao_rows:
        raise SystemExit(f"no BAO constraints found in: {bao_path}")

    out_dir = _ROOT / "output" / "cosmology"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "cosmology_reconnection_plausibility.png"
    out_json = out_dir / "cosmology_reconnection_plausibility_metrics.json"

    metrics = _plot(
        out_png=out_png,
        ddr=ddr_primary,
        opacity_rows=op_rows,
        candle_rows=cd_rows,
        bao_rows=bao_rows,
        eps_min_model=eps_min_model,
    )

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "ddr": str(ddr_path).replace("\\", "/"),
            "ddr_sigma_policy": dict(ddr_sigma_policy_meta),
            "opacity": str(opacity_path).replace("\\", "/"),
            "candle": str(candle_path).replace("\\", "/"),
            "bao": str(bao_path).replace("\\", "/"),
        },
        "metrics": metrics,
        "outputs": {"png": str(out_png).replace("\\", "/"), "metrics_json": str(out_json).replace("\\", "/")},
    }
    _write_json(out_json, payload)

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")

    try:
        opacity_primary = _primary_by_sigma(op_rows, "alpha_opacity_sigma")
        candle_primary = _primary_by_sigma(cd_rows, "s_L_sigma")
        bao_primary = _primary_by_sigma(bao_rows, "r_drag_sigma_mpc")
        worklog.append_event(
            {
                "event_type": "cosmology_reconnection_plausibility",
                "argv": list(sys.argv),
                "inputs": {"ddr": ddr_path, "opacity": opacity_path, "candle": candle_path, "bao": bao_path},
                "outputs": {"png": out_png, "metrics_json": out_json},
                "metrics": {
                    "ddr_primary_id": ddr_primary.id,
                    "opacity_primary_id": opacity_primary.id,
                    "candle_primary_id": candle_primary.id,
                    "bao_primary_id": bao_primary.id,
                    "delta_eps_needed": metrics.get("delta_eps_needed"),
                },
            }
        )
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
