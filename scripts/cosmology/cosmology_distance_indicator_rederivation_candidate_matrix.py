#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_distance_indicator_rederivation_candidate_matrix.py

Step 14.2.23（再導出候補の感度分解：α×s_L マトリクス）:
Step 14.2.22 では DDRごとに best_any / best_independent の max|z| を要約したが、
「どの一次ソース拘束の組合せが整合/不一致を決めているか」を見える化するため、
代表DDR（BAO含む/なし）について (不透明度α候補 × 標準光源進化s_L候補) の全組合せで
max|z|（DDR + BAO(s_R) + α + s_L + p_t + p_e の同時整合）を計算し、ヒートマップで出力する。

モデル（同じ）:
  ε0_model = (p_e + p_t - s_L)/2 - 2 + s_R + α

入力（固定）:
  - data/cosmology/distance_duality_constraints.json
  - data/cosmology/cosmic_opacity_constraints.json
  - data/cosmology/sn_standard_candle_evolution_constraints.json
  - data/cosmology/sn_time_dilation_constraints.json
  - data/cosmology/cmb_temperature_scaling_constraints.json（beta_T→p_T）
  - output/private/cosmology/cosmology_bao_scaled_distance_fit_metrics.json（s_R の fit）

出力（固定名）:
  - output/private/cosmology/cosmology_distance_indicator_rederivation_candidate_matrix.png
  - output/private/cosmology/cosmology_distance_indicator_rederivation_candidate_matrix_metrics.json
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


def _fmt_float(x: Optional[float], *, digits: int = 3) -> str:
    # 条件分岐: `x is None` を満たす経路を評価する。
    if x is None:
        return ""

    # 条件分岐: `not math.isfinite(float(x))` を満たす経路を評価する。

    if not math.isfinite(float(x)):
        return ""

    x = float(x)
    # 条件分岐: `x == 0.0` を満たす経路を評価する。
    if x == 0.0:
        return "0"

    ax = abs(x)
    # 条件分岐: `ax >= 1e4 or ax < 1e-3` を満たす経路を評価する。
    if ax >= 1e4 or ax < 1e-3:
        return f"{x:.{digits}g}"

    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


def _maybe_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None

    # 条件分岐: `not math.isfinite(v)` を満たす経路を評価する。

    if not math.isfinite(v):
        return None

    return float(v)


@dataclass(frozen=True)
class DDRConstraint:
    id: str
    short_label: str
    epsilon0: float
    epsilon0_sigma: float
    uses_bao: bool
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
            epsilon0=float(j["epsilon0"]),
            epsilon0_sigma=sigma,
            uses_bao=bool(j.get("uses_bao", False)),
            epsilon0_sigma_raw=sigma,
        )


@dataclass(frozen=True)
class GaussianConstraint:
    id: str
    short_label: str
    mean: float
    sigma: float
    uses_bao: Optional[bool]
    uses_cmb: Optional[bool]
    assumes_cddr: Optional[bool]

    def is_independent(self) -> bool:
        # 条件分岐: `self.uses_bao is True` を満たす経路を評価する。
        if self.uses_bao is True:
            return False

        # 条件分岐: `self.uses_cmb is True` を満たす経路を評価する。

        if self.uses_cmb is True:
            return False

        # 条件分岐: `self.assumes_cddr is True` を満たす経路を評価する。

        if self.assumes_cddr is True:
            return False

        return True


def _load_ddr_systematics_envelope(path: Path) -> Dict[str, Dict[str, Any]]:
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        return {}

    try:
        j = _read_json(path)
    except Exception:
        return {}

    rows = j.get("rows") if isinstance(j.get("rows"), list) else []
    out: Dict[str, Dict[str, Any]] = {}
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


def _apply_ddr_sigma_policy(ddr: DDRConstraint, *, policy: str, envelope: Dict[str, Dict[str, Any]]) -> DDRConstraint:
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

        # 条件分岐: `not (sig > 0.0 and math.isfinite(sig))` を満たす経路を評価する。

        if not (sig > 0.0 and math.isfinite(sig)):
            continue

        out.append(
            GaussianConstraint(
                id=str(r.get("id") or ""),
                short_label=str(r.get("short_label") or r.get("id") or ""),
                mean=float(mean),
                sigma=float(sig),
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


def _as_pT_constraints(rows: Sequence[Dict[str, Any]]) -> List[GaussianConstraint]:
    out: List[GaussianConstraint] = []
    for r in rows:
        try:
            beta = float(r["beta_T"])
            sig = float(r["beta_T_sigma"])
        except Exception:
            continue

        # 条件分岐: `not (sig > 0.0 and math.isfinite(sig))` を満たす経路を評価する。

        if not (sig > 0.0 and math.isfinite(sig)):
            continue

        out.append(
            GaussianConstraint(
                id=str(r.get("id") or ""),
                short_label=str(r.get("short_label") or r.get("id") or ""),
                mean=float(1.0 - beta),
                sigma=float(sig),
                uses_bao=None,
                uses_cmb=None,
                assumes_cddr=None,
            )
        )

    return out


def _wls_max_abs_z(
    *,
    ddr: DDRConstraint,
    sR_bao: float,
    sR_bao_sigma: float,
    opacity: GaussianConstraint,
    candle: GaussianConstraint,
    p_t: GaussianConstraint,
    p_e: GaussianConstraint,
) -> Dict[str, Any]:
    # θ=[s_R, α, s_L, p_t, p_e]
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
    max_abs_z = float(np.max(np.abs(z)))
    limiting_idx = int(np.argmax(np.abs(z)))
    names = [
        "DDR ε0",
        "BAO s_R",
        "Opacity α",
        "Candle s_L",
        "SN time dilation p_t",
        "CMB energy p_e",
    ]
    return {
        "max_abs_z": max_abs_z,
        "limiting_observation": names[limiting_idx],
    }


def _plot_matrix(
    ax: Any,
    *,
    title: str,
    z_grid: np.ndarray,
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    independent_mask: np.ndarray,
    best_ij: Tuple[int, int],
    vmax: float,
) -> None:
    import matplotlib.colors as mcolors

    # Discrete bins: [0,3)=green, [3,5)=yellow, [5,inf)=red
    bounds = [0.0, 3.0, 5.0, vmax]
    cmap = mcolors.ListedColormap(["#2ca02c", "#ffbf00", "#d62728"])
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(np.clip(z_grid, 0.0, vmax), cmap=cmap, norm=norm, aspect="auto")
    ax.set_title(title, fontsize=12)
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(list(x_labels), rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(list(y_labels), fontsize=9)

    # Cell annotations and independent border
    for iy in range(z_grid.shape[0]):
        for ix in range(z_grid.shape[1]):
            v = float(z_grid[iy, ix])
            ax.text(ix, iy, _fmt_float(v, digits=2), ha="center", va="center", fontsize=8, color="#111111")
            # 条件分岐: `bool(independent_mask[iy, ix])` を満たす経路を評価する。
            if bool(independent_mask[iy, ix]):
                rect = __import__("matplotlib.patches").patches.Rectangle(
                    (ix - 0.5, iy - 0.5),
                    1.0,
                    1.0,
                    fill=False,
                    edgecolor="#000000",
                    linewidth=1.2,
                )
                ax.add_patch(rect)

    by, bx = best_ij
    ax.scatter([bx], [by], marker="*", s=120, c="#000000")

    # Grid lines
    ax.set_xticks(np.arange(-0.5, len(x_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(y_labels), 1), minor=True)
    ax.grid(which="minor", color="#ffffff", linestyle="-", linewidth=1.0, alpha=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
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
    args = ap.parse_args(argv)

    data_dir = _ROOT / "data" / "cosmology"
    out_dir = _ROOT / "output" / "private" / "cosmology"

    in_ddr = data_dir / "distance_duality_constraints.json"
    in_opacity = data_dir / "cosmic_opacity_constraints.json"
    in_candle = data_dir / "sn_standard_candle_evolution_constraints.json"
    in_pt = data_dir / "sn_time_dilation_constraints.json"
    in_pe = data_dir / "cmb_temperature_scaling_constraints.json"
    in_bao_fit = out_dir / "cosmology_bao_scaled_distance_fit_metrics.json"

    for p in (in_ddr, in_opacity, in_candle, in_pt, in_pe, in_bao_fit):
        # 条件分岐: `not p.exists()` を満たす経路を評価する。
        if not p.exists():
            raise FileNotFoundError(f"missing input: {p}")

    ddr_sigma_policy = str(args.ddr_sigma_policy)
    ddr_env_path = out_dir / "cosmology_distance_duality_systematics_envelope_metrics.json"
    ddr_env = _load_ddr_systematics_envelope(ddr_env_path) if ddr_sigma_policy == "category_sys" else {}

    ddr_all = [
        _apply_ddr_sigma_policy(DDRConstraint.from_json(r), policy=ddr_sigma_policy, envelope=ddr_env)
        for r in (_read_json(in_ddr).get("constraints") or [])
    ]
    # 条件分岐: `not ddr_all` を満たす経路を評価する。
    if not ddr_all:
        raise ValueError("no DDR constraints found")

    opacity_all = _as_gaussian_list(
        _read_json(in_opacity).get("constraints") or [],
        mean_key="alpha_opacity",
        sigma_key="alpha_opacity_sigma",
    )
    candle_all = _as_gaussian_list(_read_json(in_candle).get("constraints") or [], mean_key="s_L", sigma_key="s_L_sigma")
    pt_all = _as_gaussian_list(_read_json(in_pt).get("constraints") or [], mean_key="p_t", sigma_key="p_t_sigma")
    pe_all = _as_pT_constraints(_read_json(in_pe).get("constraints") or [])
    # 条件分岐: `not (opacity_all and candle_all and pt_all and pe_all)` を満たす経路を評価する。
    if not (opacity_all and candle_all and pt_all and pe_all):
        raise ValueError("missing constraints (opacity/candle/p_t/p_e)")

    p_t = pt_all[0]
    p_e = pe_all[0]

    bao_fit = _read_json(in_bao_fit)
    sR_bao = float(bao_fit["fit"]["best_fit"]["s_R"])
    sR_bao_sigma = float(bao_fit["fit"]["best_fit"]["s_R_sigma_1d"])

    # Representative DDR (BAO含む/なし): consistent with prior steps
    rep_bao_id = "martinelli2021_snIa_bao"
    rep_no_bao_id = "holanda2012_clusters_snIa_constitution_eta0_linear_noniso_spherical"
    ddr_bao = next((d for d in ddr_all if d.id == rep_bao_id), None)
    ddr_no = next((d for d in ddr_all if d.id == rep_no_bao_id), None)
    # 条件分岐: `ddr_bao is None or ddr_no is None` を満たす経路を評価する。
    if ddr_bao is None or ddr_no is None:
        raise ValueError("representative DDR ids not found (update constants if schema changed)")

    ddr_reps = [ddr_bao, ddr_no]
    applied_ddr_sigma_count = len([d for d in ddr_all if d.sigma_policy == "category_sys"])

    # Compute grids (y=candle, x=opacity)
    results: Dict[str, Any] = {"per_ddr": []}
    vmax = 12.0
    for ddr in ddr_reps:
        z_grid = np.zeros((len(candle_all), len(opacity_all)), dtype=float)
        ind_mask = np.zeros_like(z_grid, dtype=bool)
        limiting = [[None for _ in range(len(opacity_all))] for _ in range(len(candle_all))]
        best_val = float("inf")
        best_ij = (0, 0)
        for iy, cd in enumerate(candle_all):
            for ix, op in enumerate(opacity_all):
                fit = _wls_max_abs_z(
                    ddr=ddr,
                    sR_bao=sR_bao,
                    sR_bao_sigma=sR_bao_sigma,
                    opacity=op,
                    candle=cd,
                    p_t=p_t,
                    p_e=p_e,
                )
                v = float(fit["max_abs_z"])
                z_grid[iy, ix] = v
                limiting[iy][ix] = str(fit["limiting_observation"])
                ind_mask[iy, ix] = bool(op.is_independent() and cd.is_independent())
                # 条件分岐: `v < best_val` を満たす経路を評価する。
                if v < best_val:
                    best_val = v
                    best_ij = (iy, ix)

        results["per_ddr"].append(
            {
                "ddr": {
                    "id": ddr.id,
                    "short_label": ddr.short_label,
                    "uses_bao": ddr.uses_bao,
                    "epsilon0_obs": ddr.epsilon0,
                    "epsilon0_sigma": ddr.epsilon0_sigma,
                    "epsilon0_sigma_raw": ddr.epsilon0_sigma_raw,
                    "sigma_sys_category": ddr.sigma_sys_category,
                    "sigma_policy": ddr.sigma_policy,
                    "category": ddr.category,
                },
                "x_opacity": [{"id": o.id, "short_label": o.short_label, "independent": o.is_independent()} for o in opacity_all],
                "y_candle": [{"id": c.id, "short_label": c.short_label, "independent": c.is_independent()} for c in candle_all],
                "max_abs_z_grid": z_grid.tolist(),
                "independent_mask": ind_mask.tolist(),
                "limiting_observation_grid": limiting,
                "best_cell": {
                    "iy_candle": int(best_ij[0]),
                    "ix_opacity": int(best_ij[1]),
                    "max_abs_z": float(best_val),
                    "opacity": {"id": opacity_all[best_ij[1]].id, "short_label": opacity_all[best_ij[1]].short_label},
                    "candle": {"id": candle_all[best_ij[0]].id, "short_label": candle_all[best_ij[0]].short_label},
                },
            }
        )

    out_png = out_dir / "cosmology_distance_indicator_rederivation_candidate_matrix.png"
    out_metrics = out_dir / "cosmology_distance_indicator_rederivation_candidate_matrix_metrics.json"

    _set_japanese_font()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 9.2))
    gs = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.0))

    opacity_labels = [o.short_label for o in opacity_all]
    candle_labels = [c.short_label for c in candle_all]

    ims = []
    for k, (ddr, ax) in enumerate(zip(ddr_reps, [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])])):
        r = results["per_ddr"][k]
        z_grid = np.array(r["max_abs_z_grid"], dtype=float)
        ind_mask = np.array(r["independent_mask"], dtype=bool)
        best_ij = (int(r["best_cell"]["iy_candle"]), int(r["best_cell"]["ix_opacity"]))
        title = f"{ddr.short_label}（{'BAO含む' if ddr.uses_bao else 'BAOなし'}）: max|z|（★=最小）"
        im = _plot_matrix(
            ax,
            title=title,
            z_grid=z_grid,
            x_labels=opacity_labels,
            y_labels=candle_labels,
            independent_mask=ind_mask,
            best_ij=best_ij,
            vmax=vmax,
        )
        ims.append(im)

    fig.suptitle("宇宙論（距離指標）：再導出候補マトリクス（不透明度α × 標準光源進化s_L）", fontsize=14)
    fig.text(
        0.5,
        0.012,
        "各セルは WLS により θ=[s_R,α,s_L,p_t,p_e] を同時fitしたときの max|z|。枠線=independent（BAO/CMB/CDDR依存を避けた α と s_L）。",
        ha="center",
        fontsize=10,
    )
    plt.tight_layout(rect=(0.0, 0.04, 1.0, 0.93))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    metrics: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "definition": {
            "epsilon0_model": "ε0 = (p_e + p_t - s_L)/2 - 2 + s_R + α",
            "metric": "max|z| across DDR + BAO(s_R) + α + s_L + p_t + p_e",
            "color_bins": "<3 OK / 3-5 mixed / >5 NG",
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
            "candle": str(in_candle.relative_to(_ROOT)).replace("\\", "/"),
            "sn_time_dilation": str(in_pt.relative_to(_ROOT)).replace("\\", "/"),
            "cmb_temperature_scaling": str(in_pe.relative_to(_ROOT)).replace("\\", "/"),
            "bao_fit": str(in_bao_fit.relative_to(_ROOT)).replace("\\", "/"),
        },
        "fixed_constraints": {
            "p_t": {"id": p_t.id, "mean": p_t.mean, "sigma": p_t.sigma, "short_label": p_t.short_label},
            "p_e": {"id": p_e.id, "mean": p_e.mean, "sigma": p_e.sigma, "short_label": p_e.short_label},
            "bao_s_R": {"mean": sR_bao, "sigma": sR_bao_sigma},
        },
        "results": results,
        "outputs": {
            "png": str(out_png.relative_to(_ROOT)).replace("\\", "/"),
            "metrics_json": str(out_metrics.relative_to(_ROOT)).replace("\\", "/"),
        },
    }
    _write_json(out_metrics, metrics)

    worklog.append_event(
        {
            "kind": "cosmology",
            "step": "14.2.23",
            "task": "distance_indicator_rederivation_candidate_matrix",
            "inputs": [
                str(in_ddr.relative_to(_ROOT)).replace("\\", "/"),
                str(in_opacity.relative_to(_ROOT)).replace("\\", "/"),
                str(in_candle.relative_to(_ROOT)).replace("\\", "/"),
                str(in_pt.relative_to(_ROOT)).replace("\\", "/"),
                str(in_pe.relative_to(_ROOT)).replace("\\", "/"),
                str(in_bao_fit.relative_to(_ROOT)).replace("\\", "/"),
            ],
            "outputs": {"png": out_png, "metrics_json": out_metrics},
            "metrics": {
                "rep_bao_best": results["per_ddr"][0]["best_cell"],
                "rep_no_bao_best": results["per_ddr"][1]["best_cell"],
            },
        }
    )

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_metrics}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
