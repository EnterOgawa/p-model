#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_fsigma8_growth_mapping.py

Step 8.7.18.2:
静的空間 + 動的P（遅延応答）から構造形成の最小写像を固定し、
RSD の fσ8 と同一I/Fで比較する。

理論骨子（本スクリプトで固定する式）:
  1) 連続の式（線形）:   dδ/dt + θ = 0,  θ = ∇·v
  2) オイラー（線形）:   dθ/dt = c^2 ∇²u - c_s² ∇²δ
  3) 遅延ポテンシャル:   τ_eff du/dt + u = u_N,
                         ∇²u_N = -(4πG/c²) ρ̄ δ
  4) 連立消去（k空間）:
       τ_eff δ''' + δ'' + τ_eff c_s² k² δ'
       + (c_s² k² - 4πGρ̄)δ = 0
  5) 低周波近似（slow manifold; ε_delay<<1）:
       δ'' + Γ_eff δ' + (c_s² k² - 4πGρ̄)δ = 0
       Γ_eff = 4πGρ̄ τ_eff
     -> 遅延由来の実効摩擦項 Γ_eff δ' が自然創発する。
  6) a_eff 変数（a_eff=1/(1+z)）:
       d²D/d(ln a_eff)² + B(a_eff) dD/d(ln a_eff) - C(a_eff)D = 0
       B = dlnH_eff/dln a_eff + Γ_eff/H_eff
       C = (4πGρ̄)/H_eff²
       f = dlnD/dln a_eff

入力（固定）:
  - data/cosmology/boss_dr12_baofs_consensus_reduced_covariance_cij.json
  - output/public/cosmology/cosmology_cluster_collision_pmu_jmu_separation_derivation.json

出力（固定名）:
  - output/private/cosmology/cosmology_fsigma8_growth_mapping.png
  - output/private/cosmology/cosmology_fsigma8_growth_mapping_metrics.json
  - output/private/cosmology/cosmology_fsigma8_growth_mapping_falsification_pack.json
  - output/private/cosmology/cosmology_fsigma8_growth_mapping_points.csv
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

ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(ROOT) not in sys.path` を満たす経路を評価する。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summary import worklog  # noqa: E402

MPC_M = 3.0856775814913673e22
KM_S_MPC_TO_SI = 1000.0 / MPC_M
G_NEWTON = 6.67430e-11
GIGAYEAR_S = 365.25 * 24.0 * 3600.0 * 1.0e9


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


# 関数: `_copy_to_public` の入出力契約と処理意図を定義する。

def _copy_to_public(private_paths: Sequence[Path], public_dir: Path) -> Dict[str, str]:
    public_dir.mkdir(parents=True, exist_ok=True)
    copied: Dict[str, str] = {}
    for src in private_paths:
        dst = public_dir / src.name
        shutil.copy2(src, dst)
        copied[src.name] = str(dst).replace("\\", "/")

    return copied


# クラス: `RSDPoint` の責務と境界条件を定義する。

@dataclass(frozen=True)
class RSDPoint:
    z: float
    a_eff: float
    h_km_s_mpc: float
    h_si: float
    fs8_obs: float
    fs8_sigma: float


# クラス: `BranchResult` の責務と境界条件を定義する。

@dataclass(frozen=True)
class BranchResult:
    branch: str
    tau_eff_s: float
    mu_ref: float
    sigma8_anchor: float
    chi2: float
    dof: int
    z_sorted_desc: np.ndarray
    a_sorted_desc: np.ndarray
    h_sorted_desc: np.ndarray
    f_sorted_desc: np.ndarray
    sigma8_sorted_desc: np.ndarray
    fs8_sorted_desc: np.ndarray
    gamma_over_h_sorted_desc: np.ndarray
    dlnh_dlnasorted_desc: np.ndarray
    c_sorted_desc: np.ndarray


# 関数: `_load_boss_rsd_points` の入出力契約と処理意図を定義する。

def _load_boss_rsd_points(path: Path) -> List[RSDPoint]:
    raw = _read_json(path)
    params = list(raw.get("parameters", []))
    h_map: Dict[float, float] = {}
    fs8_map: Dict[float, Tuple[float, float]] = {}
    for row in params:
        kind = str(row.get("kind", "")).strip()
        z = float(row.get("z_eff"))
        # 条件分岐: `kind == "H_scaled_km_s_mpc"` を満たす経路を評価する。
        if kind == "H_scaled_km_s_mpc":
            h_map[z] = float(row.get("mean"))
        # 条件分岐: 前段条件が不成立で、`kind == "f_sigma8"` を追加評価する。
        elif kind == "f_sigma8":
            fs8_map[z] = (float(row.get("mean")), float(row.get("sigma")))

    points: List[RSDPoint] = []
    for z in sorted(set(h_map.keys()) & set(fs8_map.keys())):
        fs8, sig = fs8_map[z]
        h = h_map[z]
        points.append(
            RSDPoint(
                z=float(z),
                a_eff=float(1.0 / (1.0 + z)),
                h_km_s_mpc=float(h),
                h_si=float(h * KM_S_MPC_TO_SI),
                fs8_obs=float(fs8),
                fs8_sigma=float(max(sig, 1.0e-9)),
            )
        )

    # 条件分岐: `len(points) < 3` を満たす経路を評価する。

    if len(points) < 3:
        raise RuntimeError(f"insufficient RSD points for fσ8 mapping: {len(points)}")

    return points


# 関数: `_load_tau_eff_seconds` の入出力契約と処理意図を定義する。

def _load_tau_eff_seconds(path: Path) -> float:
    raw = _read_json(path)
    tau_block = dict(raw.get("tau_origin_block", {}))
    components = dict(tau_block.get("derived_components_gyr", {}))
    tau_eff_gyr = float(components.get("tau_eff", float("nan")))
    # 条件分岐: `(not math.isfinite(tau_eff_gyr)) or tau_eff_gyr <= 0.0` を満たす経路を評価する。
    if (not math.isfinite(tau_eff_gyr)) or tau_eff_gyr <= 0.0:
        tau_eff_gyr = float(components.get("tau_eff_harmonic", float("nan")))

    # 条件分岐: `(not math.isfinite(tau_eff_gyr)) or tau_eff_gyr <= 0.0` を満たす経路を評価する。

    if (not math.isfinite(tau_eff_gyr)) or tau_eff_gyr <= 0.0:
        raise RuntimeError("tau_eff not found in tau derivation JSON")

    return float(tau_eff_gyr * GIGAYEAR_S)


# 関数: `_interp_log` の入出力契約と処理意図を定義する。

def _interp_log(a_nodes: np.ndarray, y_nodes: np.ndarray, a_eval: np.ndarray) -> np.ndarray:
    x_nodes = np.log(a_nodes)
    x_eval = np.log(a_eval)
    lny = np.log(y_nodes)
    lny_eval = np.interp(x_eval, x_nodes, lny)
    return np.exp(lny_eval)


# 関数: `_make_background_functions` の入出力契約と処理意図を定義する。

def _make_background_functions(a_nodes: np.ndarray, h_nodes: np.ndarray) -> Dict[str, Any]:
    x_nodes = np.log(a_nodes)
    ln_h_nodes = np.log(h_nodes)
    dlnh_dx_nodes = np.gradient(ln_h_nodes, x_nodes)

    # 関数: `h_of_x` の入出力契約と処理意図を定義する。
    def h_of_x(x: np.ndarray) -> np.ndarray:
        return np.exp(np.interp(x, x_nodes, ln_h_nodes))

    # 関数: `dlnh_dx_of_x` の入出力契約と処理意図を定義する。

    def dlnh_dx_of_x(x: np.ndarray) -> np.ndarray:
        return np.interp(x, x_nodes, dlnh_dx_nodes)

    return {
        "x_nodes": x_nodes,
        "ln_h_nodes": ln_h_nodes,
        "dlnh_dx_nodes": dlnh_dx_nodes,
        "h_of_x": h_of_x,
        "dlnh_dx_of_x": dlnh_dx_of_x,
    }


# 関数: `_integrate_growth` の入出力契約と処理意図を定義する。

def _integrate_growth(
    *,
    mu_ref: float,
    tau_eff_s: float,
    a_nodes: np.ndarray,
    h_nodes: np.ndarray,
    a_ref: float,
    n_steps: int = 800,
) -> Dict[str, np.ndarray]:
    background = _make_background_functions(a_nodes, h_nodes)
    h_of_x = background["h_of_x"]
    dlnh_dx_of_x = background["dlnh_dx_of_x"]

    x_min = float(np.min(np.log(a_nodes)))
    x_max = float(np.max(np.log(a_nodes)))
    x_grid = np.linspace(x_min, x_max, int(max(n_steps, 80)))
    a_grid = np.exp(x_grid)
    h_grid = h_of_x(x_grid)
    h_ref = float(np.interp(math.log(a_ref), x_grid, h_grid))

    # 関数: `rhs` の入出力契約と処理意図を定義する。
    def rhs(x: float, state: np.ndarray) -> np.ndarray:
        d = float(state[0])
        dp = float(state[1])
        h = float(h_of_x(np.array([x]))[0])
        dlnh = float(dlnh_dx_of_x(np.array([x]))[0])
        a = float(math.exp(x))
        c_val = float(mu_ref * (a / a_ref) ** (-3.0) * (h_ref * h_ref) / max(h * h, 1.0e-30))
        gamma_over_h = float(tau_eff_s * c_val * h)
        b_val = float(dlnh + gamma_over_h)
        ddp = float(-b_val * dp + c_val * d)
        return np.array([dp, ddp], dtype=float)

    h0 = float(h_of_x(np.array([x_grid[0]]))[0])
    dlnh0 = float(dlnh_dx_of_x(np.array([x_grid[0]]))[0])
    c0 = float(mu_ref * (math.exp(x_grid[0]) / a_ref) ** (-3.0) * (h_ref * h_ref) / max(h0 * h0, 1.0e-30))
    gamma_over_h0 = float(tau_eff_s * c0 * h0)
    b0 = float(dlnh0 + gamma_over_h0)
    discriminant = max(b0 * b0 + 4.0 * c0, 0.0)
    f0 = float(0.5 * (-b0 + math.sqrt(discriminant)))

    state = np.array([1.0, f0], dtype=float)
    d_grid = np.zeros_like(x_grid)
    dp_grid = np.zeros_like(x_grid)
    d_grid[0] = state[0]
    dp_grid[0] = state[1]

    for i in range(len(x_grid) - 1):
        x = float(x_grid[i])
        dx = float(x_grid[i + 1] - x_grid[i])
        k1 = rhs(x, state)
        k2 = rhs(x + 0.5 * dx, state + 0.5 * dx * k1)
        k3 = rhs(x + 0.5 * dx, state + 0.5 * dx * k2)
        k4 = rhs(x + dx, state + dx * k3)
        state = state + (dx / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        d_grid[i + 1] = state[0]
        dp_grid[i + 1] = state[1]

    f_grid = np.divide(dp_grid, np.maximum(np.abs(d_grid), 1.0e-20))
    c_grid = mu_ref * (a_grid / a_ref) ** (-3.0) * (h_ref * h_ref) / np.maximum(h_grid * h_grid, 1.0e-30)
    gamma_over_h_grid = tau_eff_s * c_grid * h_grid
    dlnh_dx_grid = dlnh_dx_of_x(x_grid)

    return {
        "x_grid": x_grid,
        "a_grid": a_grid,
        "h_grid": h_grid,
        "d_grid": d_grid,
        "dp_grid": dp_grid,
        "f_grid": f_grid,
        "c_grid": c_grid,
        "gamma_over_h_grid": gamma_over_h_grid,
        "dlnh_dx_grid": dlnh_dx_grid,
        "a_ref": np.array([a_ref], dtype=float),
    }


# 関数: `_evaluate_branch` の入出力契約と処理意図を定義する。

def _evaluate_branch(
    *,
    branch: str,
    tau_eff_s: float,
    mu_ref: float,
    points: List[RSDPoint],
    anchor_z: float,
) -> BranchResult:
    z_desc = np.array(sorted([p.z for p in points], reverse=True), dtype=float)
    a_desc = 1.0 / (1.0 + z_desc)
    h_desc = np.array([next(p.h_si for p in points if abs(p.z - z) < 1.0e-12) for z in z_desc], dtype=float)

    a_asc = a_desc[::-1]
    h_asc = h_desc[::-1]
    growth = _integrate_growth(
        mu_ref=mu_ref,
        tau_eff_s=tau_eff_s,
        a_nodes=a_asc,
        h_nodes=h_asc,
        a_ref=float(1.0 / (1.0 + anchor_z)),
        n_steps=1200,
    )

    x_grid = np.array(growth["x_grid"], dtype=float)
    d_grid = np.array(growth["d_grid"], dtype=float)
    f_grid = np.array(growth["f_grid"], dtype=float)
    c_grid = np.array(growth["c_grid"], dtype=float)
    gamma_over_h_grid = np.array(growth["gamma_over_h_grid"], dtype=float)
    dlnh_dx_grid = np.array(growth["dlnh_dx_grid"], dtype=float)

    x_desc = np.log(a_desc)
    d_desc = np.interp(x_desc, x_grid, d_grid)
    f_desc = np.interp(x_desc, x_grid, f_grid)
    c_desc = np.interp(x_desc, x_grid, c_grid)
    gamma_over_h_desc = np.interp(x_desc, x_grid, gamma_over_h_grid)
    dlnh_desc = np.interp(x_desc, x_grid, dlnh_dx_grid)

    anchor_idx = int(np.argmin(np.abs(z_desc - float(anchor_z))))
    f_anchor = float(f_desc[anchor_idx])
    # 条件分岐: `(not math.isfinite(f_anchor)) or abs(f_anchor) < 1.0e-9` を満たす経路を評価する。
    if (not math.isfinite(f_anchor)) or abs(f_anchor) < 1.0e-9:
        raise RuntimeError(f"invalid f(anchor) for branch={branch}, mu_ref={mu_ref}")

    fs8_anchor_obs = float(next(p.fs8_obs for p in points if abs(p.z - float(z_desc[anchor_idx])) < 1.0e-12))
    sigma8_anchor = float(fs8_anchor_obs / f_anchor)
    sigma8_desc = sigma8_anchor * (d_desc / max(float(d_desc[anchor_idx]), 1.0e-12))
    fs8_desc = f_desc * sigma8_desc

    chi2 = 0.0
    for i, z in enumerate(z_desc):
        p = next(pp for pp in points if abs(pp.z - float(z)) < 1.0e-12)
        dz = float((fs8_desc[i] - p.fs8_obs) / p.fs8_sigma)
        chi2 += dz * dz

    dof = max(len(points) - 1, 1)  # sigma8 anchor removes one dof

    return BranchResult(
        branch=branch,
        tau_eff_s=float(tau_eff_s),
        mu_ref=float(mu_ref),
        sigma8_anchor=float(sigma8_anchor),
        chi2=float(chi2),
        dof=int(dof),
        z_sorted_desc=z_desc,
        a_sorted_desc=a_desc,
        h_sorted_desc=h_desc,
        f_sorted_desc=f_desc,
        sigma8_sorted_desc=sigma8_desc,
        fs8_sorted_desc=fs8_desc,
        gamma_over_h_sorted_desc=gamma_over_h_desc,
        dlnh_dlnasorted_desc=dlnh_desc,
        c_sorted_desc=c_desc,
    )


# 関数: `_fit_mu_ref` の入出力契約と処理意図を定義する。

def _fit_mu_ref(
    *,
    branch: str,
    tau_eff_s: float,
    points: List[RSDPoint],
    anchor_z: float,
    mu_min: float,
    mu_max: float,
    n_grid: int,
) -> BranchResult:
    best: Optional[BranchResult] = None
    grid = np.linspace(float(mu_min), float(mu_max), int(max(n_grid, 30)))
    for mu in grid:
        try:
            result = _evaluate_branch(
                branch=branch,
                tau_eff_s=float(tau_eff_s),
                mu_ref=float(mu),
                points=points,
                anchor_z=float(anchor_z),
            )
        except Exception:
            continue

        # 条件分岐: `best is None or result.chi2 < best.chi2` を満たす経路を評価する。

        if best is None or result.chi2 < best.chi2:
            best = result

    # 条件分岐: `best is None` を満たす経路を評価する。

    if best is None:
        raise RuntimeError(f"unable to fit mu_ref for branch={branch}")

    return best


# 関数: `_write_points_csv` の入出力契約と処理意図を定義する。

def _write_points_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fieldnames = [
        "branch",
        "z",
        "a_eff",
        "H_eff_km_s_Mpc",
        "dlnH_dln_a_eff",
        "C_gravity",
        "Gamma_eff_over_H_eff",
        "f_pred",
        "sigma8_pred",
        "f_sigma8_pred",
        "f_sigma8_obs",
        "f_sigma8_sigma",
        "z_score",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


# 関数: `_plot` の入出力契約と処理意図を定義する。

def _plot(
    *,
    points: List[RSDPoint],
    delay: BranchResult,
    instant: BranchResult,
    tau_eff_gyr: float,
    out_png: Path,
) -> None:
    _set_japanese_font()
    import matplotlib.pyplot as plt

    z_obs = np.array([p.z for p in points], dtype=float)
    fs8_obs = np.array([p.fs8_obs for p in points], dtype=float)
    sig_obs = np.array([p.fs8_sigma for p in points], dtype=float)
    order = np.argsort(z_obs)
    z_obs = z_obs[order]
    fs8_obs = fs8_obs[order]
    sig_obs = sig_obs[order]

    z_desc = delay.z_sorted_desc
    z_asc = z_desc[::-1]
    delay_fs8 = delay.fs8_sorted_desc[::-1]
    instant_fs8 = instant.fs8_sorted_desc[::-1]
    delay_f = delay.f_sorted_desc[::-1]
    instant_f = instant.f_sorted_desc[::-1]
    delay_gamma = delay.gamma_over_h_sorted_desc[::-1]
    delay_dlnh = delay.dlnh_dlnasorted_desc[::-1]

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.4), dpi=160)

    axes[0].plot(z_asc, delay_dlnh, marker="o", color="#4c78a8", label="dlnH_eff/dln a_eff")
    axes[0].plot(z_asc, delay_gamma, marker="s", color="#f58518", label="Γ_eff/H_eff (delay)")
    axes[0].axhline(0.0, color="#888", linestyle="--", linewidth=1.0)
    axes[0].set_title("実効摩擦項の創発（無次元）")
    axes[0].set_xlabel("z")
    axes[0].set_ylabel("coefficient")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best", fontsize=9)

    axes[1].errorbar(z_obs, fs8_obs, yerr=sig_obs, fmt="o", color="#111111", capsize=3, label="BOSS DR12 fσ8")
    axes[1].plot(z_asc, delay_fs8, marker="o", color="#2ca02c", label="P-model delay branch")
    axes[1].plot(z_asc, instant_fs8, marker="^", color="#d62728", label="instant branch (τ_eff=0)")
    axes[1].set_title("fσ8(z): 観測 vs 写像")
    axes[1].set_xlabel("z")
    axes[1].set_ylabel("fσ8")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best", fontsize=9)

    axes[2].plot(z_asc, delay_f, marker="o", color="#1f77b4", label="f=dlnD/dln a_eff (delay)")
    axes[2].plot(z_asc, instant_f, marker="^", color="#ff7f0e", label="f (instant)")
    axes[2].set_title("成長率 f のスケーリング")
    axes[2].set_xlabel("z")
    axes[2].set_ylabel("f")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="best", fontsize=9)

    fig.suptitle("Step 8.7.18.2: fσ8 growth mapping with delayed P response")
    fig.text(
        0.01,
        -0.02,
        (
            f"tau_eff={tau_eff_gyr:.6f} Gyr, "
            f"chi2/dof(delay)={delay.chi2:.3f}/{delay.dof}, "
            f"chi2/dof(instant)={instant.chi2:.3f}/{instant.dof}"
        ),
        fontsize=9,
        va="top",
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Step 8.7.18.2: fσ8 growth mapping with delayed-P effective friction.")
    parser.add_argument(
        "--boss-input",
        type=str,
        default=str(ROOT / "data" / "cosmology" / "boss_dr12_baofs_consensus_reduced_covariance_cij.json"),
        help="BOSS DR12 consensus (includes fσ8 points).",
    )
    parser.add_argument(
        "--tau-input",
        type=str,
        default=str(ROOT / "output" / "public" / "cosmology" / "cosmology_cluster_collision_pmu_jmu_separation_derivation.json"),
        help="Derived tau_eff source JSON.",
    )
    parser.add_argument("--anchor-z", type=float, default=0.51, help="Anchor redshift for sigma8 normalization.")
    parser.add_argument("--mu-min", type=float, default=0.05, help="Minimum mu_ref grid value.")
    parser.add_argument("--mu-max", type=float, default=2.0, help="Maximum mu_ref grid value.")
    parser.add_argument("--mu-grid", type=int, default=400, help="Grid size for mu_ref scan.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "output" / "private" / "cosmology"),
    )
    parser.add_argument(
        "--public-dir",
        type=str,
        default=str(ROOT / "output" / "public" / "cosmology"),
    )
    parser.add_argument("--skip-public-copy", action="store_true")
    parser.add_argument("--step-tag", type=str, default="8.7.18.2")
    args = parser.parse_args(list(argv) if argv is not None else None)

    boss_path = Path(args.boss_input).resolve()
    tau_path = Path(args.tau_input).resolve()
    points = _load_boss_rsd_points(boss_path)
    tau_eff_s = _load_tau_eff_seconds(tau_path)
    tau_eff_gyr = tau_eff_s / GIGAYEAR_S

    delay = _fit_mu_ref(
        branch="delay",
        tau_eff_s=tau_eff_s,
        points=points,
        anchor_z=float(args.anchor_z),
        mu_min=float(args.mu_min),
        mu_max=float(args.mu_max),
        n_grid=int(args.mu_grid),
    )
    instant = _fit_mu_ref(
        branch="instant",
        tau_eff_s=0.0,
        points=points,
        anchor_z=float(args.anchor_z),
        mu_min=float(args.mu_min),
        mu_max=float(args.mu_max),
        n_grid=int(args.mu_grid),
    )

    by_z: Dict[float, Dict[str, Any]] = {}
    for p in points:
        by_z[p.z] = {
            "z": float(p.z),
            "a_eff": float(p.a_eff),
            "H_eff_km_s_Mpc": float(p.h_km_s_mpc),
            "f_sigma8_obs": float(p.fs8_obs),
            "f_sigma8_sigma": float(p.fs8_sigma),
        }

    rows: List[Dict[str, Any]] = []
    for branch_result in (delay, instant):
        for i, z in enumerate(branch_result.z_sorted_desc):
            row = dict(by_z[float(z)])
            row["branch"] = branch_result.branch
            row["dlnH_dln_a_eff"] = float(branch_result.dlnh_dlnasorted_desc[i])
            row["C_gravity"] = float(branch_result.c_sorted_desc[i])
            row["Gamma_eff_over_H_eff"] = float(branch_result.gamma_over_h_sorted_desc[i])
            row["f_pred"] = float(branch_result.f_sorted_desc[i])
            row["sigma8_pred"] = float(branch_result.sigma8_sorted_desc[i])
            row["f_sigma8_pred"] = float(branch_result.fs8_sorted_desc[i])
            row["z_score"] = float((row["f_sigma8_pred"] - row["f_sigma8_obs"]) / max(row["f_sigma8_sigma"], 1.0e-9))
            rows.append(row)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    base = "cosmology_fsigma8_growth_mapping"
    out_png = out_dir / f"{base}.png"
    out_json = out_dir / f"{base}_metrics.json"
    out_fals = out_dir / f"{base}_falsification_pack.json"
    out_csv = out_dir / f"{base}_points.csv"

    _plot(points=points, delay=delay, instant=instant, tau_eff_gyr=float(tau_eff_gyr), out_png=out_png)
    _write_points_csv(out_csv, rows)

    delay_rows = [r for r in rows if str(r.get("branch")) == "delay"]
    instant_rows = [r for r in rows if str(r.get("branch")) == "instant"]
    delay_max_abs_z = max(abs(float(r["z_score"])) for r in delay_rows) if delay_rows else float("nan")
    instant_max_abs_z = max(abs(float(r["z_score"])) for r in instant_rows) if instant_rows else float("nan")
    friction_present = all(float(r["Gamma_eff_over_H_eff"]) > 0.0 for r in delay_rows)

    # 条件分岐: `(not math.isfinite(delay_max_abs_z)) or (not friction_present)` を満たす経路を評価する。
    if (not math.isfinite(delay_max_abs_z)) or (not friction_present):
        overall_status = "reject"
    # 条件分岐: 前段条件が不成立で、`delay_max_abs_z <= 2.5 and delay.chi2 <= instant.chi2 + 0.25` を追加評価する。
    elif delay_max_abs_z <= 2.5 and delay.chi2 <= instant.chi2 + 0.25:
        overall_status = "pass"
    # 条件分岐: 前段条件が不成立で、`delay_max_abs_z <= 4.0` を追加評価する。
    elif delay_max_abs_z <= 4.0:
        overall_status = "watch"
    else:
        overall_status = "reject"

    metrics: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase": {"step": str(args.step_tag), "name": "fsigma8_growth_mapping"},
        "inputs": {
            "boss_consensus_json": str(boss_path).replace("\\", "/"),
            "tau_derivation_json": str(tau_path).replace("\\", "/"),
            "n_rsd_points": len(points),
            "anchor_z_for_sigma8": float(args.anchor_z),
        },
        "derivation": {
            "continuity_linear": "dδ/dt + θ = 0",
            "euler_linear": "dθ/dt = c^2 ∇²u - c_s^2 ∇²δ",
            "delayed_potential": "τ_eff du/dt + u = u_N, ∇²u_N = -(4πG/c²)ρ̄δ",
            "delta_equation_exact": "τ_eff δ''' + δ'' + τ_eff c_s²k² δ' + (c_s²k² - 4πGρ̄)δ = 0",
            "delta_equation_slow_manifold": "δ'' + Γ_eff δ' + (c_s²k² - 4πGρ̄)δ = 0, Γ_eff=4πGρ̄τ_eff",
            "a_eff_mapping": "a_eff = P_bg(t_obs)/P_bg(t) = 1/(1+z),  f = dlnD/dln a_eff",
        },
        "fit_policy": {
            "mu_ref_scan": {
                "mu_min": float(args.mu_min),
                "mu_max": float(args.mu_max),
                "n_grid": int(args.mu_grid),
                "note": "mu_ref is determined by fσ8 consistency under fixed τ_eff (no free per-z retuning).",
            },
            "sigma8_normalization": "sigma8(anchor) = fs8_obs(anchor) / f_pred(anchor)",
        },
        "branches": {
            "delay": {
                "tau_eff_gyr": float(tau_eff_gyr),
                "mu_ref_best": float(delay.mu_ref),
                "sigma8_anchor": float(delay.sigma8_anchor),
                "chi2": float(delay.chi2),
                "dof": int(delay.dof),
                "chi2_per_dof": float(delay.chi2 / max(delay.dof, 1)),
                "max_abs_z_score": float(delay_max_abs_z),
                "mean_gamma_over_h": float(np.mean([float(r["Gamma_eff_over_H_eff"]) for r in delay_rows])) if delay_rows else float("nan"),
            },
            "instant": {
                "tau_eff_gyr": 0.0,
                "mu_ref_best": float(instant.mu_ref),
                "sigma8_anchor": float(instant.sigma8_anchor),
                "chi2": float(instant.chi2),
                "dof": int(instant.dof),
                "chi2_per_dof": float(instant.chi2 / max(instant.dof, 1)),
                "max_abs_z_score": float(instant_max_abs_z),
            },
        },
        "gates": {
            "growth_mapping_defined": True,
            "fsigma8_gate_defined": True,
            "friction_term_positive": bool(friction_present),
            "delay_vs_instant_nonworse": bool(delay.chi2 <= instant.chi2 + 0.25),
            "overall_status": overall_status,
        },
        "outputs": {
            "png": str(out_png).replace("\\", "/"),
            "metrics_json": str(out_json).replace("\\", "/"),
            "falsification_pack_json": str(out_fals).replace("\\", "/"),
            "points_csv": str(out_csv).replace("\\", "/"),
        },
    }
    _write_json(out_json, metrics)

    fals_pack: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "fsigma8_growth_mapping",
        "decision_rule": {
            "mapping_defined": "growth_mapping_defined=true and fsigma8_gate_defined=true",
            "friction": "Gamma_eff/H_eff must be positive at all RSD points",
            "consistency": "max_abs_z_score(delay)<=2.5 => pass; <=4.0 => watch; else reject",
            "comparison": "delay chi2 should not be worse than instant branch by more than +0.25",
        },
        "result": metrics["gates"],
        "notes": [
            "fσ8 comparison uses BOSS DR12 consensus points (z=0.38, 0.51, 0.61).",
            "a_eff is defined from P_bg mapping (a_eff=1/(1+z)); no FRW scale factor is assumed as primary variable.",
        ],
        "related_outputs": {
            "metrics_json": str(out_json).replace("\\", "/"),
            "figure_png": str(out_png).replace("\\", "/"),
            "points_csv": str(out_csv).replace("\\", "/"),
        },
    }
    _write_json(out_fals, fals_pack)

    copied: Dict[str, str] = {}
    # 条件分岐: `not bool(args.skip_public_copy)` を満たす経路を評価する。
    if not bool(args.skip_public_copy):
        copied = _copy_to_public([out_png, out_json, out_fals, out_csv], Path(args.public_dir).resolve())

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")
    print(f"[ok] fals: {out_fals}")
    print(f"[ok] csv : {out_csv}")
    # 条件分岐: `copied` を満たす経路を評価する。
    if copied:
        print(f"[ok] copied to public: {len(copied)} files")

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_fsigma8_growth_mapping",
                "argv": list(sys.argv),
                "inputs": {
                    "boss_input": boss_path,
                    "tau_input": tau_path,
                },
                "outputs": {
                    "png": out_png,
                    "metrics_json": out_json,
                    "falsification_pack_json": out_fals,
                    "points_csv": out_csv,
                    "public_copies": copied,
                },
                "metrics": {
                    "tau_eff_gyr": float(tau_eff_gyr),
                    "mu_ref_delay": float(delay.mu_ref),
                    "chi2_delay": float(delay.chi2),
                    "chi2_instant": float(instant.chi2),
                    "max_abs_z_delay": float(delay_max_abs_z),
                    "overall_status": overall_status,
                },
            }
        )
    except Exception:
        pass

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())

