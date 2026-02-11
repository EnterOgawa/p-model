#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sparc_rar_freeze_test_procedure_sweep.py

Phase 6 / Step 6.5（SPARC：RAR/BTFR）:
freeze-test（fit→freeze→holdout; galaxy split）の「手続き系統」を sweep して固定出力化する。

目的：
- 同一データ/同一モデルでも、評価手続き（low-accel domain の定義、σの床、galaxy-level集約条件）で
  統計量（holdout |z|）が動く可能性があるため、系統として台帳化する。

入力：
- output/private/cosmology/sparc_rar_reconstruction.csv（g_bar/g_obs の再構成；既定Υで固定）
- output/private/cosmology/cosmology_redshift_pbg_metrics.json（H0^(P)；a0=κ c H0^(P)）

出力（固定）：
- output/private/cosmology/sparc_rar_freeze_test_procedure_sweep_metrics.json
- output/private/cosmology/sparc_rar_freeze_test_procedure_sweep.png（任意；matplotlib が無い場合はスキップ）
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.cosmology.sparc_falsification_pack import DEFAULT_PBG_KAPPA  # noqa: E402
from scripts.cosmology.sparc_rar_freeze_test import _run_once, _summarize_sweep  # noqa: E402
from scripts.cosmology.sparc_rar_freeze_test import _read_points as _read_points_from_csv  # noqa: E402

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
    if not (np.isfinite(start) and np.isfinite(stop) and np.isfinite(step) and step > 0):
        raise ValueError("invalid grid params")
    if stop < start:
        raise ValueError("stop < start")
    n = int(math.floor((stop - start) / step + 0.5)) + 1
    vv = start + step * np.arange(n, dtype=float)
    vv = vv[(vv >= start - 1e-12) & (vv <= stop + 1e-12)]
    return [float(x) for x in vv.tolist()]


def _unique_sorted(values: Sequence[float]) -> List[float]:
    return sorted({float(x) for x in values if np.isfinite(x)})


def _splits(seeds: Sequence[int], train_fracs: Sequence[float]) -> List[Tuple[int, float]]:
    return [(int(s), float(f)) for s in seeds for f in train_fracs]


def _parse_clipping_specs(specs: Sequence[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for raw in specs:
        s = str(raw or "").strip().lower()
        if not s:
            continue
        if s in {"none", "off", "no"}:
            out.append({"method": "none", "k": None})
            continue
        if s.startswith("mad"):
            k = 3.5
            if ":" in s:
                _, tail = s.split(":", 1)
                k = float(tail.strip())
            if not (np.isfinite(k) and float(k) > 0):
                raise ValueError(f"invalid clipping spec: {raw}")
            out.append({"method": "mad", "k": float(k)})
            continue
        raise ValueError(f"unknown clipping spec: {raw}")

    # unique + stable order
    seen: set[Tuple[str, Optional[float]]] = set()
    uniq: List[Dict[str, Any]] = []
    for v in out:
        key = (str(v.get("method")), v.get("k") if v.get("k") is not None else None)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(v)
    return uniq


@dataclass(frozen=True)
class _Variant:
    sigma_floor_dex: float
    low_accel_cut_log10_gbar: float
    min_points_per_galaxy: int


def _plot_heatmaps(
    *,
    out_png: Path,
    variants: Sequence[Dict[str, Any]],
    sigma_floor_list: Sequence[float],
    low_accel_cut_list: Sequence[float],
    mpg_list: Sequence[int],
    plot_clipping: Dict[str, Any],
    model_name: str,
) -> None:
    try:
        import matplotlib.pyplot as plt  # noqa: WPS433
    except Exception:
        return

    sf = np.asarray(list(sigma_floor_list), dtype=float)
    lc = np.asarray(list(low_accel_cut_list), dtype=float)
    if sf.size == 0 or lc.size == 0:
        return
    sf = np.unique(sf[np.isfinite(sf)])
    lc = np.unique(lc[np.isfinite(lc)])
    mpg = [int(x) for x in mpg_list if int(x) > 0]
    if sf.size == 0 or lc.size == 0 or not mpg:
        return

    # index variants by params
    idx: Dict[Tuple[float, float, int], float] = {}
    for v in variants:
        if not isinstance(v, dict) or v.get("status") != "ok":
            continue
        par = v.get("params", {}) if isinstance(v.get("params"), dict) else {}
        sfd = par.get("sigma_floor_dex")
        lac = par.get("low_accel_cut_log10_gbar")
        mpp = par.get("min_points_per_galaxy")
        clip = par.get("galaxy_clipping", {}) if isinstance(par.get("galaxy_clipping"), dict) else {}
        if str(clip.get("method") or "") != str(plot_clipping.get("method") or ""):
            continue
        if (clip.get("k") is None) != (plot_clipping.get("k") is None):
            continue
        if clip.get("k") is not None and not np.isclose(float(clip.get("k")), float(plot_clipping.get("k"))):
            continue
        if not (isinstance(sfd, (int, float)) and isinstance(lac, (int, float)) and isinstance(mpp, int)):
            continue
        ss_g = v.get("sweep_summary_galaxy", {}) if isinstance(v.get("sweep_summary_galaxy"), dict) else {}
        m = ss_g.get(model_name, {}) if isinstance(ss_g.get(model_name), dict) else {}
        pr = m.get("pass_rate_abs_lt_threshold")
        if isinstance(pr, (int, float)) and np.isfinite(pr):
            idx[(float(sfd), float(lac), int(mpp))] = float(pr)

    fig, axes = plt.subplots(1, len(mpg), figsize=(4.4 * len(mpg), 4.0), squeeze=False)
    for k, mpp in enumerate(mpg):
        ax = axes[0, k]
        m = np.full((lc.size, sf.size), np.nan, dtype=float)
        for i, lcv in enumerate(lc.tolist()):
            for j, sfv in enumerate(sf.tolist()):
                m[i, j] = float(idx.get((float(sfv), float(lcv), int(mpp)), float("nan")))
        im = ax.imshow(m, origin="lower", aspect="auto", vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_xticks(np.arange(sf.size))
        ax.set_yticks(np.arange(lc.size))
        ax.set_xticklabels([f"{x:g}" for x in sf.tolist()])
        ax.set_yticklabels([f"{x:g}" for x in lc.tolist()])
        ax.set_xlabel("sigma_floor_dex")
        ax.set_ylabel("low_accel_cut_log10_gbar")
        ax.set_title(f"min_points_per_galaxy={int(mpp)}")
        for i in range(lc.size):
            for j in range(sf.size):
                v = m[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8, color="w" if v < 0.55 else "k")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("pass_rate(|z|<3) [galaxy-level]")

    spec = str(plot_clipping.get("method") or "none")
    if plot_clipping.get("k") is not None:
        spec = f"{spec}:{float(plot_clipping.get('k')):g}"
    fig.suptitle(f"SPARC freeze-test procedure sweep (candidate; galaxy-level pass_rate) [{spec}]")
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
    p.add_argument("--pbg-kappa", type=float, default=DEFAULT_PBG_KAPPA, help="a0 = kappa * c * H0^(P) (default: 1/(2π))")

    p.add_argument("--seed-start", type=int, default=20260129, help="Seed sweep start (default: 20260129)")
    p.add_argument("--seed-count", type=int, default=20, help="Seed sweep count (default: 20)")
    p.add_argument("--train-frac-start", type=float, default=0.6, help="Train fraction sweep start (default: 0.6)")
    p.add_argument("--train-frac-stop", type=float, default=0.8, help="Train fraction sweep stop (default: 0.8)")
    p.add_argument("--train-frac-step", type=float, default=0.1, help="Train fraction sweep step (default: 0.1)")

    p.add_argument("--sigma-floor", action="append", type=float, default=[], help="sigma_floor_dex value to include (repeatable)")
    p.add_argument("--low-accel-cut", action="append", type=float, default=[], help="low_accel_cut_log10_gbar value to include (repeatable)")
    p.add_argument("--min-points-per-galaxy", action="append", type=int, default=[], help="min points per galaxy in galaxy-level aggregation (repeatable)")
    p.add_argument(
        "--galaxy-clipping",
        action="append",
        default=[],
        help="Optional clipping spec(s) for galaxy means. Examples: none, mad:3.5, mad:5 (repeatable)",
    )

    p.add_argument(
        "--out",
        default=str(_ROOT / "output" / "private" / "cosmology" / "sparc_rar_freeze_test_procedure_sweep_metrics.json"),
        help="Output JSON path",
    )
    p.add_argument(
        "--out-png",
        default=str(_ROOT / "output" / "private" / "cosmology" / "sparc_rar_freeze_test_procedure_sweep.png"),
        help="Output plot PNG path",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    rar_csv = Path(args.rar_csv)
    if not rar_csv.exists():
        raise SystemExit(f"missing rar csv: {rar_csv}")
    h0p_metrics = Path(args.h0p_metrics)
    if not h0p_metrics.exists():
        raise SystemExit(f"missing h0p metrics: {h0p_metrics}")

    sigma_floor_list = _unique_sorted([float(x) for x in (args.sigma_floor or [])] or [0.005, 0.01, 0.02])
    low_accel_cut_list = _unique_sorted([float(x) for x in (args.low_accel_cut or [])] or [-10.7, -10.5, -10.3])
    mpg_list = sorted({int(x) for x in (args.min_points_per_galaxy or [])} or {2, 3, 5})
    mpg_list = [int(x) for x in mpg_list if int(x) > 0]
    clipping_list = _parse_clipping_specs(list(args.galaxy_clipping or []) or ["none", "mad:3.5"])
    if not sigma_floor_list or not low_accel_cut_list or not mpg_list or not clipping_list:
        raise SystemExit("empty procedure grid")

    seeds = list(range(int(args.seed_start), int(args.seed_start) + int(max(args.seed_count, 1))))
    train_fracs = _parse_grid(float(args.train_frac_start), float(args.train_frac_stop), float(args.train_frac_step))
    split_list = _splits(seeds, train_fracs)
    if not split_list:
        raise SystemExit("no splits")

    pts = _read_points_from_csv(rar_csv)
    if len(pts) < 100:
        raise SystemExit(f"not enough points: {len(pts)}")

    # Speed: for the sweep we only need the fixed-kappa candidate (and baryons-only as a null).
    model_subset = ["baryons_only", "candidate_rar_pbg_a0_fixed_kappa"]

    variants: List[Dict[str, Any]] = []
    for sfd in sigma_floor_list:
        for lac in low_accel_cut_list:
            for mpg in mpg_list:
                for clip in clipping_list:
                    by_model_galaxy: Dict[str, List[float]] = {}
                    for seed, train_frac in split_list:
                        run = _run_once(
                            pts,
                            seed=int(seed),
                            train_frac=float(train_frac),
                            h0p_metrics=h0p_metrics,
                            h0p_km_s_mpc_override=args.h0p_km_s_mpc,
                            pbg_kappa=float(args.pbg_kappa),
                            sigma_floor_dex=float(sfd),
                            low_accel_cut_log10_gbar=float(lac),
                            min_points_per_galaxy=int(mpg),
                            galaxy_clipping_method=str(clip.get("method") or "none"),
                            galaxy_clipping_k=float(clip.get("k") or 3.5),
                            models=model_subset,
                        )
                        for m in run.get("models", []) if isinstance(run.get("models"), list) else []:
                            if not isinstance(m, dict):
                                continue
                            name = str(m.get("name") or "")
                            te = (m.get("test") or {}).get("with_sigma_int") or {}
                            z_gal = ((te.get("low_accel_galaxy") or {}).get("z"))
                            if isinstance(z_gal, (int, float)) and np.isfinite(z_gal):
                                by_model_galaxy.setdefault(name, []).append(float(z_gal))

                    sweep_summary_galaxy = {k: _summarize_sweep(v, threshold=3.0) for k, v in sorted(by_model_galaxy.items())}
                    variants.append(
                        {
                            "status": "ok",
                            "params": {
                                "sigma_floor_dex": float(sfd),
                                "low_accel_cut_log10_gbar": float(lac),
                                "min_points_per_galaxy": int(mpg),
                                "galaxy_clipping": {"method": str(clip.get("method") or "none"), "k": clip.get("k")},
                            },
                            "counts": {"n_splits": int(len(split_list))},
                            "sweep_summary_galaxy": sweep_summary_galaxy,
                        }
                    )

    # Envelope across variants: candidate pass_rate
    def _env_pass_rate(model: str) -> Dict[str, Any]:
        vv: List[float] = []
        for v in variants:
            ss = v.get("sweep_summary_galaxy", {}) if isinstance(v.get("sweep_summary_galaxy"), dict) else {}
            m = ss.get(model, {}) if isinstance(ss.get(model), dict) else {}
            pr = m.get("pass_rate_abs_lt_threshold")
            if isinstance(pr, (int, float)) and np.isfinite(pr):
                vv.append(float(pr))
        if not vv:
            return {"status": "missing"}
        return {"status": "ok", "min": float(min(vv)), "max": float(max(vv)), "median": float(np.median(np.asarray(vv, dtype=float))), "n": int(len(vv))}

    env = {
        "baryons_only": _env_pass_rate("baryons_only"),
        "candidate_rar_pbg_a0_fixed_kappa": _env_pass_rate("candidate_rar_pbg_a0_fixed_kappa"),
    }

    # Identify worst/best variant for candidate by pass_rate
    worst = None
    best = None
    for v in variants:
        ss = v.get("sweep_summary_galaxy", {}) if isinstance(v.get("sweep_summary_galaxy"), dict) else {}
        cand = ss.get("candidate_rar_pbg_a0_fixed_kappa", {}) if isinstance(ss.get("candidate_rar_pbg_a0_fixed_kappa"), dict) else {}
        pr = cand.get("pass_rate_abs_lt_threshold")
        if not (isinstance(pr, (int, float)) and np.isfinite(pr)):
            continue
        row = {"params": v.get("params"), "pass_rate_abs_lt_threshold": float(pr), "median_z": cand.get("median")}
        if worst is None or float(row["pass_rate_abs_lt_threshold"]) < float(worst["pass_rate_abs_lt_threshold"]):
            worst = row
        if best is None or float(row["pass_rate_abs_lt_threshold"]) > float(best["pass_rate_abs_lt_threshold"]):
            best = row

    out_path = Path(args.out)
    out_png = Path(args.out_png)
    payload: Dict[str, Any] = {
        "generated_utc": _utc_now(),
        "inputs": {
            "rar_csv": _rel(rar_csv),
            "h0p_metrics": _rel(h0p_metrics),
            "h0p_km_s_mpc": float(args.h0p_km_s_mpc) if args.h0p_km_s_mpc is not None else None,
            "pbg_kappa": float(args.pbg_kappa),
            "seeds": {"start": int(args.seed_start), "count": int(args.seed_count)},
            "train_fracs": {"start": float(args.train_frac_start), "stop": float(args.train_frac_stop), "step": float(args.train_frac_step)},
            "procedure_grid": {
                "sigma_floor_dex": [float(x) for x in sigma_floor_list],
                "low_accel_cut_log10_gbar": [float(x) for x in low_accel_cut_list],
                "min_points_per_galaxy": [int(x) for x in mpg_list],
                "galaxy_clipping": clipping_list,
            },
            "preferred_metric": "sweep_summary_galaxy",
            "threshold_abs_z": 3.0,
            "models": model_subset,
            "note": "This sweep keeps g_bar fixed (from sparc_rar_reconstruction.csv) and varies only evaluation procedure knobs.",
        },
        "counts": {"n_points": int(len(pts)), "n_splits": int(len(split_list)), "n_variants": int(len(variants))},
        "variants": variants,
        "envelope_pass_rate_abs_lt_threshold_galaxy": env,
        "candidate": {"worst": worst, "best": best},
        "outputs": {"metrics_json": _rel(out_path), "plot_png": _rel(out_png)},
    }
    _write_json(out_path, payload)

    _plot_heatmaps(
        out_png=out_png,
        variants=variants,
        sigma_floor_list=sigma_floor_list,
        low_accel_cut_list=low_accel_cut_list,
        mpg_list=mpg_list,
        plot_clipping=next((c for c in clipping_list if c.get("method") == "none"), clipping_list[0]),
        model_name="candidate_rar_pbg_a0_fixed_kappa",
    )

    if worklog is not None:
        try:
            worklog.append_event(
                {
                    "ts_utc": payload["generated_utc"],
                    "topic": "cosmology_sparc",
                    "action": "sparc_rar_freeze_test_procedure_sweep",
                    "outputs": payload["outputs"],
                }
            )
        except Exception:
            pass

    print(json.dumps({"metrics": _rel(out_path), "plot": _rel(out_png)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
