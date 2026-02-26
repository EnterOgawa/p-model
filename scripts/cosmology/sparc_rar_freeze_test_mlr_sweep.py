#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sparc_rar_freeze_test_mlr_sweep.py

Phase 6 / Step 6.5（SPARC：RAR/BTFR）:
freeze-test（fit→freeze→holdout; galaxy split）の安定性を、
恒星 M/L（Υ_disk, Υ_bulge）系統に対して評価して固定出力化する。

背景：
- `output/private/cosmology/sparc_rar_reconstruction.csv` の `vdisk_km_s` / `vbul_km_s` は Rotmod の成分速度（Υ=1）であり、
  g_bar は `vdisk*sqrt(Υ_disk)` / `vbul*sqrt(Υ_bulge)` で再計算できる（`sparc_rar_from_rotmod.py` と同一規約）。
- point-level は同一銀河内の多数点を独立扱いして独立度（SEM）を過大評価し得るため、
  採用判定の正は galaxy-level（銀河内平均を1サンプル）を用いる。

入力：
- output/private/cosmology/sparc_rar_reconstruction.csv
- output/private/cosmology/cosmology_redshift_pbg_metrics.json（H0^(P); candidate a0=κ c H0^(P) の計算に使用）

出力（固定）：
- output/private/cosmology/sparc_rar_freeze_test_mlr_sweep_metrics.json
- output/private/cosmology/sparc_rar_freeze_test_mlr_sweep.png（任意；matplotlib が無い場合はスキップ）
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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

from scripts.cosmology.sparc_falsification_pack import DEFAULT_PBG_KAPPA  # noqa: E402
from scripts.cosmology.sparc_rar_freeze_test import Point, _run_once, _summarize_sweep  # noqa: E402

try:
    from scripts.summary import worklog  # type: ignore # noqa: E402
except Exception:  # pragma: no cover
    worklog = None

KPC_TO_M = 3.0856775814913673e19
KM_TO_M = 1.0e3


# 関数: `_utc_now` の入出力契約と処理意図を定義する。
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_rel` の入出力契約と処理意図を定義する。

def _rel(path: Path) -> str:
    try:
        return path.relative_to(_ROOT).as_posix()
    except Exception:
        return path.as_posix()


# 関数: `_write_json` の入出力契約と処理意図を定義する。

def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# 関数: `_parse_grid` の入出力契約と処理意図を定義する。

def _parse_grid(start: float, stop: float, step: float) -> List[float]:
    # 条件分岐: `not (np.isfinite(start) and np.isfinite(stop) and np.isfinite(step) and step...` を満たす経路を評価する。
    if not (np.isfinite(start) and np.isfinite(stop) and np.isfinite(step) and step > 0):
        raise ValueError("invalid grid params")

    # 条件分岐: `stop < start` を満たす経路を評価する。

    if stop < start:
        raise ValueError("stop < start")

    n = int(math.floor((stop - start) / step + 0.5)) + 1
    vv = start + step * np.arange(n, dtype=float)
    vv = vv[(vv >= start - 1e-12) & (vv <= stop + 1e-12)]
    return [float(x) for x in vv.tolist()]


# 関数: `_unique_sorted` の入出力契約と処理意図を定義する。

def _unique_sorted(values: Sequence[float]) -> List[float]:
    return sorted({float(x) for x in values if np.isfinite(x)})


# 関数: `_splits` の入出力契約と処理意図を定義する。

def _splits(seeds: Sequence[int], train_fracs: Sequence[float]) -> List[Tuple[int, float]]:
    return [(int(s), float(f)) for s in seeds for f in train_fracs]


# クラス: `_RawRow` の責務と境界条件を定義する。

@dataclass(frozen=True)
class _RawRow:
    galaxy: str
    r_kpc: float
    vgas_km_s: float
    vdisk_km_s: float
    vbul_km_s: float
    g_obs_m_s2: float
    g_obs_sigma_m_s2: float


# 関数: `_read_raw_rows` の入出力契約と処理意図を定義する。

def _read_raw_rows(rar_csv: Path) -> List[_RawRow]:
    rows: List[_RawRow] = []
    with rar_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            gal = str(row.get("galaxy") or "").strip()
            # 条件分岐: `not gal` を満たす経路を評価する。
            if not gal:
                continue

            try:
                rr_kpc = float(row.get("r_kpc") or "nan")
                vgas = float(row.get("vgas_km_s") or "nan")
                vdisk = float(row.get("vdisk_km_s") or "nan")
                vbul = float(row.get("vbul_km_s") or "nan")
                g_obs = float(row.get("g_obs_m_s2") or "nan")
                sg_obs = float(row.get("g_obs_sigma_m_s2") or "nan")
            except Exception:
                continue

            # 条件分岐: `not (np.isfinite(rr_kpc) and rr_kpc > 0 and np.isfinite(vgas) and np.isfinite...` を満たす経路を評価する。

            if not (np.isfinite(rr_kpc) and rr_kpc > 0 and np.isfinite(vgas) and np.isfinite(vdisk) and np.isfinite(vbul) and np.isfinite(g_obs) and g_obs > 0):
                continue

            rows.append(
                _RawRow(
                    galaxy=gal,
                    r_kpc=float(rr_kpc),
                    vgas_km_s=float(vgas),
                    vdisk_km_s=float(vdisk),
                    vbul_km_s=float(vbul),
                    g_obs_m_s2=float(g_obs),
                    g_obs_sigma_m_s2=float(sg_obs),
                )
            )

    return rows


# 関数: `_points_for_upsilon` の入出力契約と処理意図を定義する。

def _points_for_upsilon(rows: Sequence[_RawRow], *, upsilon_disk: float, upsilon_bulge: float) -> List[Point]:
    sd = float(max(float(upsilon_disk), 0.0))
    sb = float(max(float(upsilon_bulge), 0.0))
    f_disk = math.sqrt(sd)
    f_bul = math.sqrt(sb)
    pts: List[Point] = []
    for r in rows:
        rr_m = float(r.r_kpc) * KPC_TO_M
        # 条件分岐: `not np.isfinite(rr_m) or rr_m <= 0` を満たす経路を評価する。
        if not np.isfinite(rr_m) or rr_m <= 0:
            continue

        vgas = float(r.vgas_km_s) * KM_TO_M
        vdisk = float(r.vdisk_km_s) * KM_TO_M * f_disk
        vbul = float(r.vbul_km_s) * KM_TO_M * f_bul
        vbar2 = vgas * vgas + vdisk * vdisk + vbul * vbul
        g_bar = vbar2 / rr_m
        # 条件分岐: `not (np.isfinite(g_bar) and g_bar > 0)` を満たす経路を評価する。
        if not (np.isfinite(g_bar) and g_bar > 0):
            continue

        pts.append(
            Point(
                galaxy=str(r.galaxy),
                g_bar=float(g_bar),
                g_obs=float(r.g_obs_m_s2),
                sg_obs=float(r.g_obs_sigma_m_s2),
            )
        )

    return pts


# 関数: `_plot_heatmap` の入出力契約と処理意図を定義する。

def _plot_heatmap(
    *,
    out_png: Path,
    ud_list: Sequence[float],
    ub_list: Sequence[float],
    grid: Dict[Tuple[float, float], float],
    title: str,
    ref_ud: float,
    ref_ub: float,
) -> None:
    try:
        import matplotlib.pyplot as plt  # noqa: WPS433
    except Exception:
        return

    ud = np.asarray(list(ud_list), dtype=float)
    ub = np.asarray(list(ub_list), dtype=float)
    ud = ud[np.isfinite(ud)]
    ub = ub[np.isfinite(ub)]
    ud = np.unique(ud)
    ub = np.unique(ub)
    # 条件分岐: `ud.size == 0 or ub.size == 0` を満たす経路を評価する。
    if ud.size == 0 or ub.size == 0:
        return

    m = np.full((ud.size, ub.size), np.nan, dtype=float)
    for i, udi in enumerate(ud.tolist()):
        for j, ubj in enumerate(ub.tolist()):
            m[i, j] = float(grid.get((float(udi), float(ubj)), float("nan")))

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    im = ax.imshow(m, origin="lower", aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(np.arange(ub.size))
    ax.set_yticks(np.arange(ud.size))
    ax.set_xticklabels([f"{x:g}" for x in ub.tolist()])
    ax.set_yticklabels([f"{x:g}" for x in ud.tolist()])
    ax.set_xlabel("Υ_bulge")
    ax.set_ylabel("Υ_disk")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("pass_rate(|z|<3) [galaxy-level]")

    # annotate values
    for i in range(ud.size):
        for j in range(ub.size):
            v = m[i, j]
            # 条件分岐: `np.isfinite(v)` を満たす経路を評価する。
            if np.isfinite(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8, color="w" if v < 0.55 else "k")

    # mark reference (default) cell if present

    try:
        i_ref = int(np.where(np.isclose(ud, float(ref_ud)))[0][0])
        j_ref = int(np.where(np.isclose(ub, float(ref_ub)))[0][0])
        ax.plot([j_ref], [i_ref], marker="s", markersize=16, markerfacecolor="none", markeredgecolor="tab:orange", markeredgewidth=2)
    except Exception:
        pass

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--rar-csv",
        default=str(_ROOT / "output" / "private" / "cosmology" / "sparc_rar_reconstruction.csv"),
        help="RAR reconstruction CSV (default: output/private/cosmology/sparc_rar_reconstruction.csv)",
    )
    p.add_argument(
        "--h0p-metrics",
        default=str(_ROOT / "output" / "private" / "cosmology" / "cosmology_redshift_pbg_metrics.json"),
        help="Path to cosmology_redshift_pbg_metrics.json (default: output/private/cosmology/cosmology_redshift_pbg_metrics.json)",
    )
    p.add_argument("--h0p-km-s-mpc", type=float, default=None, help="Override H0^(P) in km/s/Mpc (optional)")
    p.add_argument("--pbg-kappa", type=float, default=DEFAULT_PBG_KAPPA, help="a0 = kappa * c * H0^(P) (default: 1/(2π))")

    p.add_argument("--sigma-floor-dex", type=float, default=0.01, help="Floor for sigma(log10 g_obs) in dex (default: 0.01)")
    p.add_argument("--low-accel-cut", type=float, default=-10.5, help="Low-acceleration cut on log10(g_bar) (default: -10.5)")

    p.add_argument("--seed-start", type=int, default=20260129, help="Seed sweep start (default: 20260129)")
    p.add_argument("--seed-count", type=int, default=50, help="Seed sweep count (default: 50)")
    p.add_argument("--train-frac-start", type=float, default=0.5, help="Train fraction sweep start (default: 0.5)")
    p.add_argument("--train-frac-stop", type=float, default=0.9, help="Train fraction sweep stop (default: 0.9)")
    p.add_argument("--train-frac-step", type=float, default=0.1, help="Train fraction sweep step (default: 0.1)")

    p.add_argument("--upsilon-disk", action="append", type=float, default=[], help="Υ_disk value to include (repeatable)")
    p.add_argument("--upsilon-bulge", action="append", type=float, default=[], help="Υ_bulge value to include (repeatable)")
    p.add_argument("--out", default=str(_ROOT / "output" / "private" / "cosmology" / "sparc_rar_freeze_test_mlr_sweep_metrics.json"), help="Output JSON path")
    p.add_argument("--out-png", default=str(_ROOT / "output" / "private" / "cosmology" / "sparc_rar_freeze_test_mlr_sweep.png"), help="Output plot PNG path")
    args = p.parse_args(list(argv) if argv is not None else None)

    rar_csv = Path(args.rar_csv)
    # 条件分岐: `not rar_csv.exists()` を満たす経路を評価する。
    if not rar_csv.exists():
        raise SystemExit(f"missing rar csv: {rar_csv}")

    h0p_metrics = Path(args.h0p_metrics)
    # 条件分岐: `not h0p_metrics.exists()` を満たす経路を評価する。
    if not h0p_metrics.exists():
        raise SystemExit(f"missing h0p metrics: {h0p_metrics}")

    # Prior (canonical): SPARC/RAR analyses commonly adopt Υ≈0.5 (disk) and Υ≈0.7 (bulge) at 3.6μm.
    # Default is a tight grid around these values to match the fixed-output robustness gate.

    ud_list = _unique_sorted([float(x) for x in (args.upsilon_disk or [])] or [0.45, 0.5, 0.55])
    ub_list = _unique_sorted([float(x) for x in (args.upsilon_bulge or [])] or [0.65, 0.7, 0.75])
    # 条件分岐: `not ud_list or not ub_list` を満たす経路を評価する。
    if not ud_list or not ub_list:
        raise SystemExit("empty upsilon grid")

    seeds = list(range(int(args.seed_start), int(args.seed_start) + int(max(args.seed_count, 1))))
    train_fracs = _parse_grid(float(args.train_frac_start), float(args.train_frac_stop), float(args.train_frac_step))
    split_list = _splits(seeds, train_fracs)
    # 条件分岐: `not split_list` を満たす経路を評価する。
    if not split_list:
        raise SystemExit("no splits")

    rows = _read_raw_rows(rar_csv)
    # 条件分岐: `len(rows) < 100` を満たす経路を評価する。
    if len(rows) < 100:
        raise SystemExit(f"not enough rows: {len(rows)}")

    variants: List[Dict[str, Any]] = []
    # grid for plotting candidate pass_rate (galaxy-level)
    cand_grid: Dict[Tuple[float, float], float] = {}

    for ud in ud_list:
        for ub in ub_list:
            pts = _points_for_upsilon(rows, upsilon_disk=float(ud), upsilon_bulge=float(ub))
            galaxies = sorted({p.galaxy for p in pts})
            # 条件分岐: `len(pts) < 100 or len(galaxies) < 50` を満たす経路を評価する。
            if len(pts) < 100 or len(galaxies) < 50:
                variants.append(
                    {
                        "upsilon_disk": float(ud),
                        "upsilon_bulge": float(ub),
                        "status": "not_enough_points",
                        "counts": {"n_points": int(len(pts)), "n_galaxies": int(len(galaxies))},
                    }
                )
                continue

            by_model_point: Dict[str, List[float]] = {}
            by_model_galaxy: Dict[str, List[float]] = {}
            for seed, train_frac in split_list:
                run = _run_once(
                    pts,
                    seed=int(seed),
                    train_frac=float(train_frac),
                    h0p_metrics=h0p_metrics,
                    h0p_km_s_mpc_override=args.h0p_km_s_mpc,
                    pbg_kappa=float(args.pbg_kappa),
                    sigma_floor_dex=float(args.sigma_floor_dex),
                    low_accel_cut_log10_gbar=float(args.low_accel_cut),
                )
                for m in run.get("models", []) if isinstance(run.get("models"), list) else []:
                    # 条件分岐: `not isinstance(m, dict)` を満たす経路を評価する。
                    if not isinstance(m, dict):
                        continue

                    name = str(m.get("name") or "")
                    te = (m.get("test") or {}).get("with_sigma_int") or {}
                    z = ((te.get("low_accel") or {}).get("z"))
                    # 条件分岐: `isinstance(z, (int, float)) and np.isfinite(z)` を満たす経路を評価する。
                    if isinstance(z, (int, float)) and np.isfinite(z):
                        by_model_point.setdefault(name, []).append(float(z))

                    z_gal = ((te.get("low_accel_galaxy") or {}).get("z"))
                    # 条件分岐: `isinstance(z_gal, (int, float)) and np.isfinite(z_gal)` を満たす経路を評価する。
                    if isinstance(z_gal, (int, float)) and np.isfinite(z_gal):
                        by_model_galaxy.setdefault(name, []).append(float(z_gal))

            sweep_summary = {k: _summarize_sweep(v, threshold=3.0) for k, v in sorted(by_model_point.items())}
            sweep_summary_galaxy = {k: _summarize_sweep(v, threshold=3.0) for k, v in sorted(by_model_galaxy.items())}

            # Candidate pass_rate for plotting/robustness
            cand = sweep_summary_galaxy.get("candidate_rar_pbg_a0_fixed_kappa") or {}
            pr = cand.get("pass_rate_abs_lt_threshold")
            # 条件分岐: `isinstance(pr, (int, float)) and np.isfinite(pr)` を満たす経路を評価する。
            if isinstance(pr, (int, float)) and np.isfinite(pr):
                cand_grid[(float(ud), float(ub))] = float(pr)

            variants.append(
                {
                    "upsilon_disk": float(ud),
                    "upsilon_bulge": float(ub),
                    "status": "ok",
                    "counts": {"n_points": int(len(pts)), "n_galaxies": int(len(galaxies)), "n_splits": int(len(split_list))},
                    "sweep_summary": sweep_summary,
                    "sweep_summary_galaxy": sweep_summary_galaxy,
                }
            )

    # envelope across variants (galaxy-level pass_rate)

    def _envelope_pass_rate(model: str) -> Dict[str, Any]:
        vv: List[float] = []
        for v in variants:
            # 条件分岐: `v.get("status") != "ok"` を満たす経路を評価する。
            if v.get("status") != "ok":
                continue

            ss_g = v.get("sweep_summary_galaxy") if isinstance(v.get("sweep_summary_galaxy"), dict) else {}
            m = ss_g.get(model) if isinstance(ss_g.get(model), dict) else {}
            pr = m.get("pass_rate_abs_lt_threshold")
            # 条件分岐: `isinstance(pr, (int, float)) and np.isfinite(pr)` を満たす経路を評価する。
            if isinstance(pr, (int, float)) and np.isfinite(pr):
                vv.append(float(pr))

        # 条件分岐: `not vv` を満たす経路を評価する。

        if not vv:
            return {"status": "missing"}

        return {"status": "ok", "min": float(min(vv)), "max": float(max(vv)), "median": float(np.median(np.asarray(vv, dtype=float))), "n": int(len(vv))}

    env = {
        "baryons_only": _envelope_pass_rate("baryons_only"),
        "baseline_rar_mcgaugh2016_fit_a0": _envelope_pass_rate("baseline_rar_mcgaugh2016_fit_a0"),
        "candidate_rar_pbg_a0_fixed_kappa": _envelope_pass_rate("candidate_rar_pbg_a0_fixed_kappa"),
        "candidate_rar_pbg_fit_kappa": _envelope_pass_rate("candidate_rar_pbg_fit_kappa"),
    }

    # Robustness rule (explicit): require min pass_rate >= 0.95 across the M/L grid.
    cand_env = env.get("candidate_rar_pbg_a0_fixed_kappa") if isinstance(env.get("candidate_rar_pbg_a0_fixed_kappa"), dict) else {}
    robust_adopted = None
    # 条件分岐: `cand_env.get("status") == "ok"` を満たす経路を評価する。
    if cand_env.get("status") == "ok":
        robust_adopted = bool(float(cand_env.get("min")) >= 0.95)

    # 関数: `_grid_pass_rate` の入出力契約と処理意図を定義する。

    def _grid_pass_rate(model: str) -> Dict[Tuple[float, float], float]:
        grid: Dict[Tuple[float, float], float] = {}
        for v in variants:
            # 条件分岐: `not isinstance(v, dict) or v.get("status") != "ok"` を満たす経路を評価する。
            if not isinstance(v, dict) or v.get("status") != "ok":
                continue

            ud = v.get("upsilon_disk")
            ub = v.get("upsilon_bulge")
            # 条件分岐: `not (isinstance(ud, (int, float)) and isinstance(ub, (int, float)) and np.isf...` を満たす経路を評価する。
            if not (isinstance(ud, (int, float)) and isinstance(ub, (int, float)) and np.isfinite(ud) and np.isfinite(ub)):
                continue

            ss_g = v.get("sweep_summary_galaxy") if isinstance(v.get("sweep_summary_galaxy"), dict) else {}
            m = ss_g.get(model) if isinstance(ss_g.get(model), dict) else {}
            pr = m.get("pass_rate_abs_lt_threshold")
            # 条件分岐: `isinstance(pr, (int, float)) and np.isfinite(pr)` を満たす経路を評価する。
            if isinstance(pr, (int, float)) and np.isfinite(pr):
                grid[(float(ud), float(ub))] = float(pr)

        return grid

    # 関数: `_marginal_by_disk` の入出力契約と処理意図を定義する。

    def _marginal_by_disk(*, grid: Dict[Tuple[float, float], float]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for ud in ud_list:
            vv: List[float] = []
            for ub in ub_list:
                pr = grid.get((float(ud), float(ub)))
                # 条件分岐: `pr is not None and np.isfinite(pr)` を満たす経路を評価する。
                if pr is not None and np.isfinite(pr):
                    vv.append(float(pr))

            # 条件分岐: `not vv` を満たす経路を評価する。

            if not vv:
                rows.append({"upsilon_disk": float(ud), "status": "missing"})
                continue

            rows.append(
                {
                    "upsilon_disk": float(ud),
                    "status": "ok",
                    "n": int(len(vv)),
                    "min": float(min(vv)),
                    "max": float(max(vv)),
                    "median": float(np.median(np.asarray(vv, dtype=float))),
                }
            )

        return rows

    # 関数: `_marginal_by_bulge` の入出力契約と処理意図を定義する。

    def _marginal_by_bulge(*, grid: Dict[Tuple[float, float], float]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for ub in ub_list:
            vv: List[float] = []
            for ud in ud_list:
                pr = grid.get((float(ud), float(ub)))
                # 条件分岐: `pr is not None and np.isfinite(pr)` を満たす経路を評価する。
                if pr is not None and np.isfinite(pr):
                    vv.append(float(pr))

            # 条件分岐: `not vv` を満たす経路を評価する。

            if not vv:
                rows.append({"upsilon_bulge": float(ub), "status": "missing"})
                continue

            rows.append(
                {
                    "upsilon_bulge": float(ub),
                    "status": "ok",
                    "n": int(len(vv)),
                    "min": float(min(vv)),
                    "max": float(max(vv)),
                    "median": float(np.median(np.asarray(vv, dtype=float))),
                }
            )

        return rows

    marginal: Dict[str, Any] = {"pass_rate_abs_lt_threshold_galaxy": {}}
    for model in ["baseline_rar_mcgaugh2016_fit_a0", "candidate_rar_pbg_a0_fixed_kappa"]:
        g = _grid_pass_rate(model)
        marginal["pass_rate_abs_lt_threshold_galaxy"][model] = {"by_upsilon_disk": _marginal_by_disk(grid=g), "by_upsilon_bulge": _marginal_by_bulge(grid=g)}

    out_path = Path(args.out)
    payload: Dict[str, Any] = {
        "generated_utc": _utc_now(),
        "inputs": {
            "rar_csv": _rel(rar_csv),
            "h0p_metrics": _rel(h0p_metrics),
            "h0p_km_s_mpc": float(args.h0p_km_s_mpc) if args.h0p_km_s_mpc is not None else None,
            "pbg_kappa": float(args.pbg_kappa),
            "seeds": {"start": int(args.seed_start), "count": int(args.seed_count)},
            "train_fracs": {"start": float(args.train_frac_start), "stop": float(args.train_frac_stop), "step": float(args.train_frac_step)},
            "sigma_floor_dex": float(args.sigma_floor_dex),
            "low_accel_cut_log10_gbar": float(args.low_accel_cut),
            "upsilon_disk": [float(x) for x in ud_list],
            "upsilon_bulge": [float(x) for x in ub_list],
            "preferred_metric": "sweep_summary_galaxy",
            "threshold_abs_z": 3.0,
            "ref_upsilon": {"disk": 0.5, "bulge": 0.7},
            "note": "g_bar is recomputed from vgas/vdisk/vbul with vdisk→vdisk*sqrt(Υ_disk), vbul→vbul*sqrt(Υ_bulge).",
        },
        "counts": {"n_rows": int(len(rows)), "n_variants": int(len(ud_list) * len(ub_list)), "n_splits": int(len(split_list))},
        "variants": variants,
        "envelope": {"pass_rate_abs_lt_threshold_galaxy": env},
        "marginal": marginal,
        "robustness": {
            "decision_rule": "robust_adopted := (min pass_rate(|z|<3) across M/L variants) >= 0.95 (galaxy-level).",
            "pass_rate_required": 0.95,
            "candidate": cand_env,
            "robust_adopted": robust_adopted,
        },
        "outputs": {"metrics_json": _rel(out_path), "plot_png": _rel(Path(args.out_png))},
    }
    _write_json(out_path, payload)

    out_png = Path(args.out_png)
    # 条件分岐: `cand_grid` を満たす経路を評価する。
    if cand_grid:
        _plot_heatmap(
            out_png=out_png,
            ud_list=ud_list,
            ub_list=ub_list,
            grid=cand_grid,
            title="SPARC freeze-test robustness vs M/L (candidate; galaxy-level pass_rate)",
            ref_ud=0.5,
            ref_ub=0.7,
        )

    # 条件分岐: `worklog is not None` を満たす経路を評価する。

    if worklog is not None:
        try:
            worklog.append_event(
                {
                    "ts_utc": payload["generated_utc"],
                    "topic": "cosmology_sparc",
                    "action": "sparc_rar_freeze_test_mlr_sweep",
                    "outputs": payload["outputs"],
                }
            )
        except Exception:
            pass

    print(json.dumps({"metrics": _rel(out_path), "plot": _rel(out_png)}, ensure_ascii=False))
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
