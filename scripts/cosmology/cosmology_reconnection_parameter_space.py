#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_reconnection_parameter_space.py

Phase 14.2（静的無限空間での再接続）向け：
距離二重性（DDR）の ε0 制約に対して、背景P（静的）最小モデル（ε0=-1）が外れる理由を
「距離指標の前提（標準光源/標準定規/不透明度）」の自由度として整理し、
どの組み合わせで ε0 を回復できるか（パラメータ空間）を可視化する。

このスクリプトは「何をどれだけ変えれば DDR を満たせるか」を示すだけで、どれが正しい機構かは決めない。
正しい機構を主張するには、独立プローブ（Tolman, SN time dilation, T(z), etc）での整合が必要。

モデル（静的幾何＋背景Pで赤方偏移）：
  観測で使う距離指標は標準化の前提を含むため、P-model側で以下を“可能性”として扱う：

  - 標準光源（例：SNIa）の有効光度進化：      L_em = L0 (1+z)^(s_L)
  - 標準定規（例：BAO）の有効スケール進化：    l_em = l0 (1+z)^(s_R)
  - 灰色不透明度（光子数非保存の有効項）：      exp(-τ) with τ(z)=2α ln(1+z)

  さらに、信号の時間伸長を Δt_obs=(1+z)^(p_t) Δt_em、光子エネルギーを E∝(1+z)^(-p_e) と一般化すると、
  DDR の破れ指数（η=(1+z)^(ε0)）は

    ε0_model = (p_e + p_t - s_L)/2 - 2 + s_R + α

  となる（p_e=p_t=1, s_L=s_R=α=0 で静的最小 ε0=-1 を回収）。

入力（固定）:
  - data/cosmology/distance_duality_constraints.json

出力（固定名）:
  - output/private/cosmology/cosmology_reconnection_parameter_space.png
  - output/private/cosmology/cosmology_reconnection_parameter_space_metrics.json
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

    # 条件分岐: `not math.isfinite(v)` を満たす経路を評価する。

    if not math.isfinite(v):
        return None

    return float(v)


def _load_ddr_systematics_envelope(path: Path) -> Dict[str, Dict[str, Any]]:
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

        sigma_total = _maybe_float(r.get("sigma_total"))
        # 条件分岐: `sigma_total is None or not (sigma_total > 0.0)` を満たす経路を評価する。
        if sigma_total is None or not (sigma_total > 0.0):
            continue

        out[r_id] = {
            "sigma_total": float(sigma_total),
            "sigma_sys_category": _maybe_float(r.get("sigma_sys_category")),
            "category": str(r.get("category") or "") or None,
        }

    return out


def _apply_ddr_sigma_policy(ddr: "DDRConstraint", *, policy: str, envelope: Dict[str, Dict[str, Any]]) -> "DDRConstraint":
    # 条件分岐: `policy != "category_sys"` を満たす経路を評価する。
    if policy != "category_sys":
        return replace(ddr, sigma_policy="raw")

    row = envelope.get(ddr.id)
    # 条件分岐: `not row` を満たす経路を評価する。
    if not row:
        return replace(ddr, sigma_policy="raw")

    sigma_total = _maybe_float(row.get("sigma_total"))
    # 条件分岐: `sigma_total is None or not (sigma_total > 0.0)` を満たす経路を評価する。
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


@dataclass(frozen=True)
class DDRConstraint:
    id: str
    short_label: str
    title: str
    epsilon0_obs: float
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
            epsilon0_obs=float(j["epsilon0"]),
            epsilon0_sigma=sigma,
            source=dict(j.get("source") or {}),
            epsilon0_sigma_raw=sigma,
        )


def epsilon0_model(
    *,
    p_e: float,
    p_t: float,
    s_L: float,
    s_R: float,
    alpha_opacity: float,
) -> float:
    return (p_e + p_t - s_L) / 2.0 - 2.0 + s_R + alpha_opacity


def _grid_zscore(
    *,
    eps_obs: float,
    sig: float,
    p_e: float,
    p_t: float,
    fixed: Dict[str, float],
    x_name: str,
    x_vals: np.ndarray,
    y_name: str,
    y_vals: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return z-score grid and epsilon0 grid."""
    # 条件分岐: `sig <= 0` を満たす経路を評価する。
    if sig <= 0:
        raise ValueError("sigma must be > 0")

    zz = np.zeros((len(y_vals), len(x_vals)), dtype=float)
    ee = np.zeros_like(zz)
    for iy, y in enumerate(y_vals):
        for ix, x in enumerate(x_vals):
            params = dict(fixed)
            params[x_name] = float(x)
            params[y_name] = float(y)
            eps = epsilon0_model(
                p_e=p_e,
                p_t=p_t,
                s_L=float(params.get("s_L", 0.0)),
                s_R=float(params.get("s_R", 0.0)),
                alpha_opacity=float(params.get("alpha_opacity", 0.0)),
            )
            ee[iy, ix] = eps
            zz[iy, ix] = (eps - eps_obs) / sig

    return zz, ee


def _solve_single_mechanism(
    *,
    eps_target: float,
    p_e: float,
    p_t: float,
    mode: str,
) -> Dict[str, float]:
    """Return a canonical parameter set that achieves eps_target with only one mechanism."""
    # Base (static-min) is p_e=p_t=1 and s_L=s_R=alpha=0 -> eps=-1.
    # Solve epsilon0_model = eps_target for one free variable.
    base_no_sr_no_alpha = (p_e + p_t - 0.0) / 2.0 - 2.0 + 0.0 + 0.0
    # 条件分岐: `mode == "opacity_only"` を満たす経路を評価する。
    if mode == "opacity_only":
        # Solve for alpha with s_L=s_R=0.
        alpha = eps_target - base_no_sr_no_alpha
        return {"alpha_opacity": float(alpha), "s_L": 0.0, "s_R": 0.0}

    # 条件分岐: `mode == "ruler_only"` を満たす経路を評価する。

    if mode == "ruler_only":
        s_R = eps_target - base_no_sr_no_alpha
        return {"alpha_opacity": 0.0, "s_L": 0.0, "s_R": float(s_R)}

    # 条件分岐: `mode == "candle_only"` を満たす経路を評価する。

    if mode == "candle_only":
        # Solve for s_L with s_R=alpha=0.
        # eps = (p_e+p_t - s_L)/2 -2 = base_no_sr_no_alpha - s_L/2
        s_L = -2.0 * (eps_target - base_no_sr_no_alpha)
        return {"alpha_opacity": 0.0, "s_L": float(s_L), "s_R": 0.0}

    raise ValueError(f"unknown mode: {mode}")


def _z1_interpretation(
    *,
    p_e: float,
    p_t: float,
    eps_target: float,
    params: Dict[str, float],
) -> Dict[str, float]:
    """Return human-interpretable factors at z=1 relative to the static-min baseline."""
    z = 1.0
    one_p_z = 1.0 + z

    # Express "how much η must be boosted" relative to the static-min model, at fixed p_e/p_t.
    # That is: Δη exponent = eps_target - eps_min (with s_L=s_R=alpha=0).
    eps_min = epsilon0_model(p_e=p_e, p_t=p_t, s_L=0.0, s_R=0.0, alpha_opacity=0.0)
    delta_eps_needed = eps_target - eps_min
    eta_boost = one_p_z**delta_eps_needed

    # Map to distance modulus shift (if attributed to D_L scaling).
    delta_mu = 5.0 * math.log10(eta_boost) if eta_boost > 0 else float("nan")
    flux_dimming_factor = eta_boost**2
    tau_equiv = math.log(flux_dimming_factor) if flux_dimming_factor > 0 else float("nan")

    out = {
        "z_ref": 1.0,
        "epsilon0_min_model": float(eps_min),
        "delta_epsilon0_needed": float(delta_eps_needed),
        "eta_boost_factor_z1": float(eta_boost),
        "delta_distance_modulus_mag_z1": float(delta_mu),
        "flux_dimming_factor_needed_z1": float(flux_dimming_factor),
        "tau_equivalent_dimming_z1": float(tau_equiv),
        "params": {k: float(v) for k, v in params.items()},
    }
    return out


def _plot(
    *,
    out_png: Path,
    constraint: DDRConstraint,
    p_e: float,
    p_t: float,
    ranges: Dict[str, Tuple[float, float]],
    n_grid: int,
) -> Dict[str, Any]:
    _set_japanese_font()
    import matplotlib.pyplot as plt

    eps_obs = float(constraint.epsilon0_obs)
    sig = float(constraint.epsilon0_sigma)

    # Build 3 slices to show the degeneracy.
    x1 = np.linspace(ranges["alpha_opacity"][0], ranges["alpha_opacity"][1], n_grid)
    y1 = np.linspace(ranges["s_R"][0], ranges["s_R"][1], n_grid)
    z1, e1 = _grid_zscore(
        eps_obs=eps_obs,
        sig=sig,
        p_e=p_e,
        p_t=p_t,
        fixed={"s_L": 0.0},
        x_name="alpha_opacity",
        x_vals=x1,
        y_name="s_R",
        y_vals=y1,
    )

    x2 = np.linspace(ranges["alpha_opacity"][0], ranges["alpha_opacity"][1], n_grid)
    y2 = np.linspace(ranges["s_L"][0], ranges["s_L"][1], n_grid)
    z2, e2 = _grid_zscore(
        eps_obs=eps_obs,
        sig=sig,
        p_e=p_e,
        p_t=p_t,
        fixed={"s_R": 0.0},
        x_name="alpha_opacity",
        x_vals=x2,
        y_name="s_L",
        y_vals=y2,
    )

    x3 = np.linspace(ranges["s_R"][0], ranges["s_R"][1], n_grid)
    y3 = np.linspace(ranges["s_L"][0], ranges["s_L"][1], n_grid)
    z3, e3 = _grid_zscore(
        eps_obs=eps_obs,
        sig=sig,
        p_e=p_e,
        p_t=p_t,
        fixed={"alpha_opacity": 0.0},
        x_name="s_R",
        x_vals=x3,
        y_name="s_L",
        y_vals=y3,
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8))

    def draw_panel(ax, xx, yy, zz, title: str, xlabel: str, ylabel: str) -> None:
        # Clip to keep colors readable.
        z_clip = np.clip(zz, -8.0, 8.0)
        im = ax.imshow(
            z_clip,
            origin="lower",
            aspect="auto",
            extent=(float(xx[0]), float(xx[-1]), float(yy[0]), float(yy[-1])),
            cmap="coolwarm",
            vmin=-8.0,
            vmax=8.0,
        )
        # Contours for |z| = 1 and 3, and z=0.
        cs0 = ax.contour(xx, yy, zz, levels=[0.0], colors=["#111111"], linewidths=1.2)
        cs1 = ax.contour(xx, yy, np.abs(zz), levels=[1.0, 3.0], colors=["#111111"], linestyles=["--", ":"], linewidths=1.0)
        ax.clabel(cs0, fmt={0.0: "z=0"}, inline=True, fontsize=8)
        ax.clabel(cs1, fmt={1.0: "|z|=1", 3.0: "|z|=3"}, inline=True, fontsize=8)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(False)
        return im

    im1 = draw_panel(
        axes[0],
        x1,
        y1,
        z1,
        "Slice A: 標準光源進化なし（s_L=0）\n不透明度 α vs 定規進化 s_R",
        "α（灰色不透明度のべき指数）",
        "s_R（標準定規のサイズ進化）",
    )
    draw_panel(
        axes[1],
        x2,
        y2,
        z2,
        "Slice B: 定規進化なし（s_R=0）\n不透明度 α vs 光源進化 s_L",
        "α（灰色不透明度のべき指数）",
        "s_L（標準光源の光度進化）",
    )
    draw_panel(
        axes[2],
        x3,
        y3,
        z3,
        "Slice C: 不透明度なし（α=0）\n定規進化 s_R vs 光源進化 s_L",
        "s_R（標準定規のサイズ進化）",
        "s_L（標準光源の光度進化）",
    )

    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.92, pad=0.02)
    cbar.set_label("zスコア = (ε0_model - ε0_obs)/σ（赤=大きい, 青=小さい）")

    fig.suptitle(
        "宇宙論（Phase 14.2）：静的背景PでDDR（ε0）を回復する“必要補正”のパラメータ空間",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.02,
        (
            f"入力: {constraint.short_label}（ε0_obs={_fmt_float(eps_obs, digits=3)}±{_fmt_float(sig, digits=3)}"
            + (
                f"; σ_cat≈{_fmt_float(constraint.sigma_sys_category, digits=3)}（rawσ={_fmt_float(constraint.epsilon0_sigma_raw, digits=3)}）"
                if str(getattr(constraint, "sigma_policy", "raw")) == "category_sys"
                else ""
            )
            + "）"
            f" / 仮定: p_e={_fmt_float(p_e, digits=2)}, p_t={_fmt_float(p_t, digits=2)}（p_t=1はSN時間伸長に整合）"
        ),
        ha="center",
        fontsize=10,
    )
    fig.subplots_adjust(left=0.05, right=0.985, bottom=0.12, top=0.86, wspace=0.28)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    # Canonical solutions (single mechanism).
    solutions: List[Dict[str, Any]] = []
    for mode in ("opacity_only", "ruler_only", "candle_only"):
        params = _solve_single_mechanism(eps_target=eps_obs, p_e=p_e, p_t=p_t, mode=mode)
        eps = epsilon0_model(p_e=p_e, p_t=p_t, s_L=params["s_L"], s_R=params["s_R"], alpha_opacity=params["alpha_opacity"])
        z = (eps - eps_obs) / sig if sig > 0 else None
        solutions.append(
            {
                "mode": mode,
                "params": params,
                "epsilon0_model": float(eps),
                "z_score": (None if z is None else float(z)),
                "z1": _z1_interpretation(p_e=p_e, p_t=p_t, eps_target=eps_obs, params=params),
            }
        )

    return {
        "constraint_id": constraint.id,
        "constraint_label": constraint.short_label,
        "epsilon0_obs": eps_obs,
        "epsilon0_sigma": sig,
        "model": {
            "definition": "ε0_model = (p_e+p_t - s_L)/2 - 2 + s_R + α",
            "p_e": p_e,
            "p_t": p_t,
            "notes": [
                "p_t=1 は時間伸長（SN light curve stretch）と整合する最小仮定。",
                "s_L/s_R/α は距離指標（標準光源/標準定規/不透明度）の“有効スケール”を表す。",
            ],
        },
        "grid": {
            "n_grid": int(n_grid),
            "ranges": {k: [float(v[0]), float(v[1])] for k, v in ranges.items()},
            "slices": [
                {"id": "A", "fixed": {"s_L": 0.0}, "axes": ["alpha_opacity", "s_R"]},
                {"id": "B", "fixed": {"s_R": 0.0}, "axes": ["alpha_opacity", "s_L"]},
                {"id": "C", "fixed": {"alpha_opacity": 0.0}, "axes": ["s_R", "s_L"]},
            ],
        },
        "canonical_solutions": solutions,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Cosmology: explore DDR reconnection parameter space (static background-P).")
    ap.add_argument(
        "--data",
        type=str,
        default=str(_ROOT / "data" / "cosmology" / "distance_duality_constraints.json"),
        help="Input JSON (default: data/cosmology/distance_duality_constraints.json)",
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
    ap.add_argument("--p-e", type=float, default=1.0, help="Photon energy redshift exponent (default: 1)")
    ap.add_argument(
        "--p-t",
        type=float,
        default=1.0,
        help="Signal time dilation exponent: Δt_obs=(1+z)^(p_t)Δt_em (default: 1)",
    )
    ap.add_argument("--grid", type=int, default=220, help="Grid resolution per axis (default: 220)")
    ap.add_argument("--alpha-min", type=float, default=-1.5, help="alpha opacity min (default: -1.5)")
    ap.add_argument("--alpha-max", type=float, default=2.5, help="alpha opacity max (default: 2.5)")
    ap.add_argument("--sl-min", type=float, default=-4.0, help="s_L min (default: -4.0)")
    ap.add_argument("--sl-max", type=float, default=4.0, help="s_L max (default: 4.0)")
    ap.add_argument("--sr-min", type=float, default=-1.0, help="s_R min (default: -1.0)")
    ap.add_argument("--sr-max", type=float, default=3.0, help="s_R max (default: 3.0)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    data_path = Path(args.data)
    src = _read_json(data_path)
    constraints_raw = src.get("constraints") if isinstance(src.get("constraints"), list) else []
    ddr_sigma_policy = str(args.ddr_sigma_policy)
    ddr_env_path = _ROOT / "output" / "private" / "cosmology" / "cosmology_distance_duality_systematics_envelope_metrics.json"
    ddr_env = _load_ddr_systematics_envelope(ddr_env_path) if ddr_sigma_policy == "category_sys" else {}
    constraints = [
        _apply_ddr_sigma_policy(DDRConstraint.from_json(c), policy=ddr_sigma_policy, envelope=ddr_env)
        for c in constraints_raw
        if isinstance(c, dict)
    ]
    applied_ddr_sigma_count = len([c for c in constraints if c.sigma_policy == "category_sys"])
    ddr_sigma_policy_meta = {
        "policy": ddr_sigma_policy,
        "envelope_metrics": (str(ddr_env_path).replace("\\", "/") if ddr_sigma_policy == "category_sys" else None),
        "applied_count": applied_ddr_sigma_count,
        "note": "If the envelope file is missing, σ_cat inflation is skipped for all rows (falls back to raw).",
    }
    # 条件分岐: `not constraints` を満たす経路を評価する。
    if not constraints:
        raise SystemExit(f"no constraints found in: {data_path}")

    out_dir = _ROOT / "output" / "private" / "cosmology"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "cosmology_reconnection_parameter_space.png"
    out_json = out_dir / "cosmology_reconnection_parameter_space_metrics.json"

    p_e = float(args.p_e)
    p_t = float(args.p_t)
    n_grid = int(args.grid)
    # 条件分岐: `n_grid < 40` を満たす経路を評価する。
    if n_grid < 40:
        raise ValueError("--grid must be >= 40")

    ranges = {
        "alpha_opacity": (float(args.alpha_min), float(args.alpha_max)),
        "s_L": (float(args.sl_min), float(args.sl_max)),
        "s_R": (float(args.sr_min), float(args.sr_max)),
    }

    # Plot the most constraining (smallest sigma) row to keep the figure compact.
    primary = min(
        constraints,
        key=lambda c: (c.epsilon0_sigma if c.epsilon0_sigma > 0 else float("inf")),
    )
    metrics = _plot(out_png=out_png, constraint=primary, p_e=p_e, p_t=p_t, ranges=ranges, n_grid=n_grid)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(data_path).replace("\\", "/"),
        "ddr_sigma_policy": dict(ddr_sigma_policy_meta),
        "constraints": [c.__dict__ for c in constraints],
        "primary": metrics,
        "outputs": {"png": str(out_png).replace("\\", "/"), "metrics_json": str(out_json).replace("\\", "/")},
    }
    _write_json(out_json, payload)

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_reconnection_parameter_space",
                "argv": list(sys.argv),
                "inputs": {"data": data_path},
                "outputs": {"png": out_png, "metrics_json": out_json},
                "params": {"p_e": p_e, "p_t": p_t, "grid": n_grid, "ranges": ranges},
            }
        )
    except Exception:
        pass

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
