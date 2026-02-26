#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sparc_rar_pbg_kappa_sweep.py

Phase 6 / Step 6.5（SPARC：RAR）:
候補 a0 = κ c H0^(P)（外部固定）について、κ を走査し、
freeze-test（fit→freeze→holdout; galaxy split）の holdout 指標（low-accel z）の
安定性（pass_rate(|z|<3)）がどの程度改善し得るかを fixed output として保存する。

注意：
- κ を SPARC から fit するのではなく、候補の感度（kappa prior の許容域）を把握する目的。
- baseline（McGaugh+2016; a0 fit）も同一 split 群で併記し、等価性/比較を崩さない。

入力：
- output/private/cosmology/sparc_rar_reconstruction.csv
- output/private/cosmology/cosmology_redshift_pbg_metrics.json（H0^(P)）

出力（固定）：
- output/private/cosmology/sparc_rar_pbg_kappa_sweep_metrics.json
- output/private/cosmology/sparc_rar_pbg_kappa_sweep.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.cosmology.sparc_falsification_pack import (  # noqa: E402
    C_LIGHT_M_S,
    DEFAULT_PBG_KAPPA,
    _fit_log10_a0_grid,
    _get_h0p_si,
    _rar_mcgaugh2016_log10_pred,
)
from scripts.cosmology.sparc_rar_freeze_test import (  # noqa: E402
    Point,
    _eval_model,
    _read_points,
    _solve_sigma_int,
    _split_by_galaxy,
    _sigma_log10_gobs,
    _summarize_sweep,
)

try:
    from scripts.summary import worklog  # type: ignore # noqa: E402
except Exception:  # pragma: no cover
    worklog = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel(path: Path) -> str:
    try:
        return path.relative_to(_ROOT).as_posix()
    except Exception:
        return path.as_posix()


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _unique_sorted(values: Sequence[float]) -> List[float]:
    out = sorted({float(x) for x in values if np.isfinite(x)})
    return out


def _splits(seeds: Sequence[int], train_fracs: Sequence[float]) -> List[Tuple[int, float]]:
    return [(int(s), float(f)) for s in seeds for f in train_fracs]


def _extract_low_accel_z(metrics: Dict[str, Any], model_name: str) -> float:
    for m in metrics.get("models", []):
        # 条件分岐: `m.get("name") != model_name` を満たす経路を評価する。
        if m.get("name") != model_name:
            continue

        te = m.get("test", {}).get("with_sigma_int", {})
        return float(te.get("low_accel", {}).get("z", float("nan")))

    return float("nan")


def _extract_sigma_int(metrics: Dict[str, Any], model_name: str) -> float:
    for m in metrics.get("models", []):
        # 条件分岐: `m.get("name") != model_name` を満たす経路を評価する。
        if m.get("name") != model_name:
            continue

        fit = m.get("fit", {})
        return float(fit.get("sigma_int_dex", float("nan")))

    return float("nan")


def _a0_from_kappa(kappa: float, h0_si: float) -> float:
    return float(kappa) * float(C_LIGHT_M_S) * float(h0_si)


def _run_split(
    pts: Sequence[Point],
    *,
    seed: int,
    train_frac: float,
    h0_si: float,
    sigma_floor_dex: float,
    low_accel_cut_log10_gbar: float,
    kappas: Sequence[float],
) -> Dict[str, Any]:
    train, test = _split_by_galaxy(pts, seed=int(seed), train_frac=float(train_frac))

    gb_tr = np.asarray([p.g_bar for p in train], dtype=float)
    go_tr = np.asarray([p.g_obs for p in train], dtype=float)
    sgo_tr = np.asarray([p.sg_obs for p in train], dtype=float)
    y_tr = np.log10(go_tr)
    sy_tr = _sigma_log10_gobs(go_tr, sgo_tr, floor_dex=float(sigma_floor_dex))

    # baseline (McGaugh+2016; fit a0 on train)
    rar_fit = _fit_log10_a0_grid(gb_tr, y_tr, sy_tr)
    la0_best = float(rar_fit.get("log10_a0_best_m_s2") or float("nan"))
    ypred_rar_tr = _rar_mcgaugh2016_log10_pred(gb_tr, log10_a0=la0_best) if np.isfinite(la0_best) else np.full_like(y_tr, np.nan)
    r_rar_tr = y_tr - ypred_rar_tr
    sigma_int_rar = _solve_sigma_int(r_rar_tr, sy_tr, dof=int(np.isfinite(r_rar_tr).sum()) - 1)

    gb_te = np.asarray([p.g_bar for p in test], dtype=float)
    ypred_rar_te = _rar_mcgaugh2016_log10_pred(gb_te, log10_a0=la0_best) if np.isfinite(la0_best) else np.full(gb_te.shape, np.nan, dtype=float)
    te_rar = _eval_model(
        "baseline_rar_mcgaugh2016_fit_a0",
        test,
        y_pred=ypred_rar_te,
        sigma_floor_dex=float(sigma_floor_dex),
        low_accel_cut_log10_gbar=float(low_accel_cut_log10_gbar),
        sigma_int_dex=float(sigma_int_rar),
    )

    out: Dict[str, Any] = {
        "seed": int(seed),
        "train_frac": float(train_frac),
        "baseline": {
            "log10_a0_best_m_s2": float(la0_best),
            "sigma_int_dex": float(sigma_int_rar),
            "low_accel_z": float(te_rar["low_accel"]["z"]),
            "low_accel_galaxy_z": float(te_rar["low_accel_galaxy"]["z"]),
        },
        "candidate_pbg": {"kappas": []},
    }

    # candidate: a0 = kappa * c * H0^(P) (no fit on SPARC)
    for kappa in kappas:
        a0 = _a0_from_kappa(float(kappa), float(h0_si))
        la0 = float(math.log10(a0)) if np.isfinite(a0) and a0 > 0 else float("nan")
        ypred_tr = _rar_mcgaugh2016_log10_pred(gb_tr, log10_a0=la0) if np.isfinite(la0) else np.full_like(y_tr, np.nan)
        r_tr = y_tr - ypred_tr
        sigma_int = _solve_sigma_int(r_tr, sy_tr, dof=int(np.isfinite(r_tr).sum()) - 1)

        ypred_te = _rar_mcgaugh2016_log10_pred(gb_te, log10_a0=la0) if np.isfinite(la0) else np.full(gb_te.shape, np.nan, dtype=float)
        te = _eval_model(
            "candidate_rar_pbg_a0_fixed_kappa",
            test,
            y_pred=ypred_te,
            sigma_floor_dex=float(sigma_floor_dex),
            low_accel_cut_log10_gbar=float(low_accel_cut_log10_gbar),
            sigma_int_dex=float(sigma_int),
        )
        out["candidate_pbg"]["kappas"].append(
            {
                "kappa": float(kappa),
                "a0_m_s2": float(a0),
                "log10_a0_m_s2": float(la0),
                "sigma_int_dex": float(sigma_int),
                "low_accel_z": float(te["low_accel"]["z"]),
                "low_accel_galaxy_z": float(te["low_accel_galaxy"]["z"]),
            }
        )

    return out


def _plot_pass_rate(
    rows: List[Dict[str, Any]],
    *,
    out_png: Path,
    kappa_ref: float,
) -> None:
    import matplotlib.pyplot as plt  # noqa: WPS433

    kappas = np.asarray([r["kappa"] for r in rows], dtype=float)
    pass_rate_point = np.asarray([r["pass_rate_abs_lt_threshold"] for r in rows], dtype=float)
    pass_rate_gal = np.asarray([r["pass_rate_abs_lt_threshold_galaxy"] for r in rows], dtype=float)
    z_med_point = np.asarray([r["median"] for r in rows], dtype=float)
    z_med_gal = np.asarray([r["median_galaxy"] for r in rows], dtype=float)

    fig, ax = plt.subplots(2, 1, figsize=(8.4, 7.2), sharex=True)
    ax[0].plot(kappas, pass_rate_point, "-o", ms=3, lw=1.2, label="point-level z")
    ax[0].plot(kappas, pass_rate_gal, "-o", ms=3, lw=1.2, label="galaxy-level z")
    ax[0].axhline(0.95, color="k", lw=1, ls="--", alpha=0.6)
    ax[0].axvline(kappa_ref, color="tab:orange", lw=1, ls="--", alpha=0.8)
    ax[0].set_ylabel("pass_rate(|z|<3) on holdout")
    ax[0].set_ylim(-0.02, 1.02)
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(loc="lower right", frameon=False)

    ax[1].plot(kappas, z_med_point, "-o", ms=3, lw=1.2, label="point-level z")
    ax[1].plot(kappas, z_med_gal, "-o", ms=3, lw=1.2, label="galaxy-level z")
    ax[1].axhline(0.0, color="k", lw=1, ls="--", alpha=0.6)
    ax[1].axvline(kappa_ref, color="tab:orange", lw=1, ls="--", alpha=0.8)
    ax[1].set_ylabel("median z (low-accel holdout)")
    ax[1].set_xlabel("kappa (a0 = kappa * c * H0^(P))")
    ax[1].grid(True, alpha=0.3)

    fig.suptitle("SPARC RAR freeze-test: pbg kappa sweep (candidate a0 fixed)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


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
    p.add_argument("--out", default=str(_ROOT / "output" / "private" / "cosmology" / "sparc_rar_pbg_kappa_sweep_metrics.json"), help="Output JSON path")
    p.add_argument("--out-png", default=str(_ROOT / "output" / "private" / "cosmology" / "sparc_rar_pbg_kappa_sweep.png"), help="Output plot PNG path")
    p.add_argument("--sigma-floor-dex", type=float, default=0.01, help="Floor for sigma(log10 g_obs) in dex")
    p.add_argument("--low-accel-cut", type=float, default=-10.5, help="Low-acceleration cut on log10(g_bar)")

    p.add_argument("--seed-start", type=int, default=20260129, help="Seed sweep start (default: 20260129)")
    p.add_argument("--seed-count", type=int, default=50, help="Seed sweep count (default: 50)")
    p.add_argument("--train-frac-start", type=float, default=0.5, help="Train fraction sweep start (default: 0.5)")
    p.add_argument("--train-frac-stop", type=float, default=0.9, help="Train fraction sweep stop (default: 0.9)")
    p.add_argument("--train-frac-step", type=float, default=0.1, help="Train fraction sweep step (default: 0.1)")

    p.add_argument("--kappa", action="append", type=float, default=[], help="kappa value to include (repeatable)")
    p.add_argument("--kappa-start", type=float, default=0.08, help="kappa grid start (default: 0.08)")
    p.add_argument("--kappa-stop", type=float, default=0.30, help="kappa grid stop (default: 0.30)")
    p.add_argument("--kappa-step", type=float, default=0.01, help="kappa grid step (default: 0.01)")
    args = p.parse_args(list(argv) if argv is not None else None)

    rar_csv = Path(args.rar_csv)
    # 条件分岐: `not rar_csv.exists()` を満たす経路を評価する。
    if not rar_csv.exists():
        raise SystemExit(f"missing rar csv: {rar_csv}")

    h0p_metrics = Path(args.h0p_metrics)
    # 条件分岐: `not h0p_metrics.exists()` を満たす経路を評価する。
    if not h0p_metrics.exists():
        raise SystemExit(f"missing h0p metrics: {h0p_metrics}")

    seeds = list(range(int(args.seed_start), int(args.seed_start) + int(max(args.seed_count, 1))))
    train_fracs = _parse_grid(float(args.train_frac_start), float(args.train_frac_stop), float(args.train_frac_step))
    kappas = _unique_sorted([float(DEFAULT_PBG_KAPPA)] + list(args.kappa) + _parse_grid(float(args.kappa_start), float(args.kappa_stop), float(args.kappa_step)))
    # 条件分岐: `not kappas` を満たす経路を評価する。
    if not kappas:
        kappas = [float(DEFAULT_PBG_KAPPA)]

    pts = _read_points(rar_csv)
    # 条件分岐: `len(pts) < 100` を満たす経路を評価する。
    if len(pts) < 100:
        raise SystemExit(f"not enough points: {len(pts)}")

    h0_si, h0_src = _get_h0p_si(h0p_metrics=h0p_metrics, h0p_km_s_mpc_override=args.h0p_km_s_mpc)

    split_list = _splits(seeds, train_fracs)
    candidate_z_by_kappa: Dict[float, List[float]] = {float(k): [] for k in kappas}
    candidate_z_galaxy_by_kappa: Dict[float, List[float]] = {float(k): [] for k in kappas}
    candidate_sigma_int_by_kappa: Dict[float, List[float]] = {float(k): [] for k in kappas}
    baseline_z: List[float] = []
    baseline_z_galaxy: List[float] = []

    for seed, train_frac in split_list:
        run = _run_split(
            pts,
            seed=int(seed),
            train_frac=float(train_frac),
            h0_si=float(h0_si),
            sigma_floor_dex=float(args.sigma_floor_dex),
            low_accel_cut_log10_gbar=float(args.low_accel_cut),
            kappas=kappas,
        )
        baseline_z.append(float(run["baseline"]["low_accel_z"]))
        baseline_z_galaxy.append(float(run["baseline"]["low_accel_galaxy_z"]))
        for row in run["candidate_pbg"]["kappas"]:
            k = float(row["kappa"])
            candidate_z_by_kappa[k].append(float(row["low_accel_z"]))
            candidate_z_galaxy_by_kappa[k].append(float(row["low_accel_galaxy_z"]))
            candidate_sigma_int_by_kappa[k].append(float(row["sigma_int_dex"]))

    sweep_rows: List[Dict[str, Any]] = []
    for kappa in kappas:
        zz = candidate_z_by_kappa[float(kappa)]
        zz_gal = candidate_z_galaxy_by_kappa[float(kappa)]
        sig = candidate_sigma_int_by_kappa[float(kappa)]
        s = _summarize_sweep(zz, threshold=3.0)
        s_gal = _summarize_sweep(zz_gal, threshold=3.0)
        s_sig = _summarize_sweep(sig, threshold=float("inf"))
        sweep_rows.append(
            {
                "kappa": float(kappa),
                "a0_m_s2": float(_a0_from_kappa(float(kappa), float(h0_si))),
                "log10_a0_m_s2": float(math.log10(_a0_from_kappa(float(kappa), float(h0_si)))),
                "pass_rate_abs_lt_threshold": float(s["pass_rate_abs_lt_threshold"]),
                "threshold_abs_z": float(s["threshold_abs_z"]),
                "n": int(s["n"]),
                "min": float(s["min"]),
                "p16": float(s["p16"]),
                "median": float(s["median"]),
                "p84": float(s["p84"]),
                "max": float(s["max"]),
                "pass_rate_abs_lt_threshold_galaxy": float(s_gal["pass_rate_abs_lt_threshold"]),
                "threshold_abs_z_galaxy": float(s_gal["threshold_abs_z"]),
                "n_galaxy": int(s_gal["n"]),
                "min_galaxy": float(s_gal["min"]),
                "p16_galaxy": float(s_gal["p16"]),
                "median_galaxy": float(s_gal["median"]),
                "p84_galaxy": float(s_gal["p84"]),
                "max_galaxy": float(s_gal["max"]),
                "sigma_int_summary_dex": {
                    "n": int(s_sig["n"]),
                    "min": float(s_sig["min"]),
                    "p16": float(s_sig["p16"]),
                    "median": float(s_sig["median"]),
                    "p84": float(s_sig["p84"]),
                    "max": float(s_sig["max"]),
                },
            }
        )

    baseline_summary = _summarize_sweep(baseline_z, threshold=3.0)
    baseline_summary_galaxy = _summarize_sweep(baseline_z_galaxy, threshold=3.0)

    best = max(sweep_rows, key=lambda r: float(r["pass_rate_abs_lt_threshold"])) if sweep_rows else None
    out = {
        "generated_utc": _utc_now(),
        "inputs": {
            "rar_csv": _rel(rar_csv),
            "h0p_metrics": _rel(h0p_metrics),
            "h0p_source": h0_src,
            "seeds": {"start": int(args.seed_start), "count": int(args.seed_count)},
            "train_fracs": {"start": float(args.train_frac_start), "stop": float(args.train_frac_stop), "step": float(args.train_frac_step)},
            "kappa_grid": {"start": float(args.kappa_start), "stop": float(args.kappa_stop), "step": float(args.kappa_step), "extra": [float(x) for x in args.kappa]},
            "sigma_floor_dex": float(args.sigma_floor_dex),
            "low_accel_cut_log10_gbar": float(args.low_accel_cut),
            "threshold_abs_z": 3.0,
            "kappa_ref": float(DEFAULT_PBG_KAPPA),
        },
        "counts": {"n_points": int(len(pts)), "n_splits": int(len(split_list)), "n_kappas": int(len(kappas))},
        "derived": {
            "H0P_SI_s^-1": float(h0_si),
            "kappa_ref": float(DEFAULT_PBG_KAPPA),
            "a0_ref_m_s2": float(_a0_from_kappa(float(DEFAULT_PBG_KAPPA), float(h0_si))),
        },
        "baseline_rar_mcgaugh2016_fit_a0": {"point": baseline_summary, "galaxy": baseline_summary_galaxy},
        "candidate_rar_pbg_a0_fixed_kappa_sweep": {"rows": sweep_rows, "best_by_pass_rate": best},
        "outputs": {"metrics_json": _rel(Path(args.out)), "plot_png": _rel(Path(args.out_png))},
    }

    out_path = Path(args.out)
    _write_json(out_path, out)

    out_png = Path(args.out_png)
    _plot_pass_rate(sweep_rows, out_png=out_png, kappa_ref=float(DEFAULT_PBG_KAPPA))

    # 条件分岐: `worklog is not None` を満たす経路を評価する。
    if worklog is not None:
        try:
            worklog.append_event(
                {
                    "ts_utc": out["generated_utc"],
                    "topic": "cosmology_sparc",
                    "action": "sparc_rar_pbg_kappa_sweep",
                    "outputs": out["outputs"],
                }
            )
        except Exception:
            pass

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
