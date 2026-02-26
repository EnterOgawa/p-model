#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sparc_rar_from_rotmod.py

Phase 6 / Step 6.5（SPARC：RAR）:
SPARC の Rotmod_LTG.zip（各銀河の rotmod.dat）から g_obs / g_bar を再構築し、
固定出力（CSV/PNG/metrics）として保存する。

入力（一次）:
- data/cosmology/sparc/raw/Rotmod_LTG.zip

出力（固定）:
- output/private/cosmology/sparc_rar_reconstruction.csv
- output/private/cosmology/sparc_rar_metrics.json
- output/private/cosmology/sparc_rar_scatter.png

注意:
- これは「取得済みの rotmod.dat から RAR を再構築する最小版」。
- P-model の銀河スケール予測（Step 6.5.1）を導入する前に、まず観測側の再現I/Fを固定する。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None


_DIST_RE = re.compile(r"^#\s*Distance\s*=\s*(?P<d>[0-9.+-Ee]+)\s*Mpc\s*$")

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


# クラス: `RotmodRow` の責務と境界条件を定義する。

@dataclass(frozen=True)
class RotmodRow:
    r_kpc: float
    vobs_km_s: float
    evobs_km_s: float
    vgas_km_s: float
    vdisk_km_s: float
    vbul_km_s: float
    sbdisk_l_pc2: float
    sbbul_l_pc2: float


# 関数: `_parse_rotmod_text` の入出力契約と処理意図を定義する。

def _parse_rotmod_text(lines: Iterable[str]) -> Tuple[Optional[float], List[RotmodRow]]:
    dist_mpc: Optional[float] = None
    rows: List[RotmodRow] = []
    for raw in lines:
        s = raw.strip()
        # 条件分岐: `not s` を満たす経路を評価する。
        if not s:
            continue

        m = _DIST_RE.match(s)
        # 条件分岐: `m` を満たす経路を評価する。
        if m:
            try:
                dist_mpc = float(m.group("d"))
            except Exception:
                dist_mpc = None

            continue

        # 条件分岐: `s.startswith("#")` を満たす経路を評価する。

        if s.startswith("#"):
            continue

        parts = s.split()
        # 条件分岐: `len(parts) < 6` を満たす経路を評価する。
        if len(parts) < 6:
            continue

        try:
            # Rad Vobs errV Vgas Vdisk Vbul [SBdisk SBbul]
            r = float(parts[0])
            vobs = float(parts[1])
            ev = float(parts[2])
            vgas = float(parts[3])
            vdisk = float(parts[4])
            vbul = float(parts[5])
            sbd = float(parts[6]) if len(parts) > 6 else float("nan")
            sbb = float(parts[7]) if len(parts) > 7 else float("nan")
        except Exception:
            continue

        rows.append(
            RotmodRow(
                r_kpc=r,
                vobs_km_s=vobs,
                evobs_km_s=ev,
                vgas_km_s=vgas,
                vdisk_km_s=vdisk,
                vbul_km_s=vbul,
                sbdisk_l_pc2=sbd,
                sbbul_l_pc2=sbb,
            )
        )

    return dist_mpc, rows


# 関数: `_compute_accel_points` の入出力契約と処理意図を定義する。

def _compute_accel_points(
    galaxy: str,
    dist_mpc: Optional[float],
    rows: Sequence[RotmodRow],
    *,
    upsilon_disk: float,
    upsilon_bulge: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    sd = float(max(upsilon_disk, 0.0))
    sb = float(max(upsilon_bulge, 0.0))
    f_disk = math.sqrt(sd)
    f_bul = math.sqrt(sb)
    for r in rows:
        rr_m = float(r.r_kpc) * KPC_TO_M
        # 条件分岐: `not np.isfinite(rr_m) or rr_m <= 0` を満たす経路を評価する。
        if not np.isfinite(rr_m) or rr_m <= 0:
            continue

        vobs = float(r.vobs_km_s) * KM_TO_M
        evobs = float(r.evobs_km_s) * KM_TO_M
        vgas = float(r.vgas_km_s) * KM_TO_M
        vdisk = float(r.vdisk_km_s) * KM_TO_M * f_disk
        vbul = float(r.vbul_km_s) * KM_TO_M * f_bul

        g_obs = (vobs * vobs) / rr_m
        # propagate σ(g_obs) ≈ 2 V σV / r
        sg_obs = (2.0 * abs(vobs) * abs(evobs) / rr_m) if np.isfinite(evobs) else float("nan")

        vbar2 = vgas * vgas + vdisk * vdisk + vbul * vbul
        g_bar = vbar2 / rr_m

        out.append(
            {
                "galaxy": galaxy,
                "distance_mpc": float(dist_mpc) if dist_mpc is not None and np.isfinite(dist_mpc) else float("nan"),
                "r_kpc": float(r.r_kpc),
                "vobs_km_s": float(r.vobs_km_s),
                "evobs_km_s": float(r.evobs_km_s),
                "vgas_km_s": float(r.vgas_km_s),
                "vdisk_km_s": float(r.vdisk_km_s),
                "vbul_km_s": float(r.vbul_km_s),
                "g_obs_m_s2": float(g_obs),
                "g_obs_sigma_m_s2": float(sg_obs),
                "g_bar_m_s2": float(g_bar),
            }
        )

    return out


# 関数: `_plot_scatter` の入出力契約と処理意図を定義する。

def _plot_scatter(out_png: Path, points: Sequence[Dict[str, Any]]) -> None:
    # 条件分岐: `plt is None` を満たす経路を評価する。
    if plt is None:
        return

    g_bar = np.asarray([float(p["g_bar_m_s2"]) for p in points], dtype=float)
    g_obs = np.asarray([float(p["g_obs_m_s2"]) for p in points], dtype=float)
    m = np.isfinite(g_bar) & np.isfinite(g_obs) & (g_bar > 0) & (g_obs > 0)
    # 条件分岐: `np.count_nonzero(m) < 10` を満たす経路を評価する。
    if np.count_nonzero(m) < 10:
        return

    x = np.log10(g_bar[m])
    y = np.log10(g_obs[m])
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, s=6, alpha=0.35, edgecolors="none")
    plt.xlabel("log10 g_bar [m/s^2]")
    plt.ylabel("log10 g_obs [m/s^2]")
    plt.title("SPARC RAR reconstructed (Rotmod_LTG)")
    plt.grid(True, alpha=0.3)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--rotmod-zip",
        default=str(_ROOT / "data" / "cosmology" / "sparc" / "raw" / "Rotmod_LTG.zip"),
        help="Path to Rotmod_LTG.zip",
    )
    p.add_argument("--upsilon-disk", type=float, default=0.5, help="Stellar M/L for disk (default: 0.5)")
    p.add_argument("--upsilon-bulge", type=float, default=0.7, help="Stellar M/L for bulge (default: 0.7)")
    p.add_argument(
        "--out-dir",
        default=str(_ROOT / "output" / "private" / "cosmology"),
        help="Output directory (default: output/private/cosmology)",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    rotmod_zip = Path(args.rotmod_zip)
    # 条件分岐: `not rotmod_zip.exists()` を満たす経路を評価する。
    if not rotmod_zip.exists():
        raise FileNotFoundError(f"missing Rotmod zip: {rotmod_zip}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    points: List[Dict[str, Any]] = []
    per_galaxy: Dict[str, Any] = {}

    with zipfile.ZipFile(rotmod_zip, "r") as zf:
        names = [n for n in zf.namelist() if n.lower().endswith("_rotmod.dat")]
        names.sort()
        for n in names:
            galaxy = Path(n).name.replace("_rotmod.dat", "")
            text = zf.read(n).decode("utf-8", errors="replace").splitlines()
            dist_mpc, rows = _parse_rotmod_text(text)
            pts = _compute_accel_points(
                galaxy,
                dist_mpc,
                rows,
                upsilon_disk=float(args.upsilon_disk),
                upsilon_bulge=float(args.upsilon_bulge),
            )
            per_galaxy[galaxy] = {"n_points": int(len(pts)), "distance_mpc": float(dist_mpc) if dist_mpc else float("nan")}
            points.extend(pts)

    out_csv = out_dir / "sparc_rar_reconstruction.csv"
    cols = [
        "galaxy",
        "distance_mpc",
        "r_kpc",
        "vobs_km_s",
        "evobs_km_s",
        "vgas_km_s",
        "vdisk_km_s",
        "vbul_km_s",
        "g_obs_m_s2",
        "g_obs_sigma_m_s2",
        "g_bar_m_s2",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for pnt in points:
            w.writerow({k: pnt.get(k, "") for k in cols})

    out_png = out_dir / "sparc_rar_scatter.png"
    _plot_scatter(out_png, points)

    m = np.asarray(
        [
            np.isfinite(float(p["g_bar_m_s2"]))
            and np.isfinite(float(p["g_obs_m_s2"]))
            and float(p["g_bar_m_s2"]) > 0.0
            and float(p["g_obs_m_s2"]) > 0.0
            for p in points
        ],
        dtype=bool,
    )
    x = np.log10(np.asarray([float(p["g_bar_m_s2"]) for p in points], dtype=float)[m]) if np.any(m) else np.asarray([], dtype=float)
    y = np.log10(np.asarray([float(p["g_obs_m_s2"]) for p in points], dtype=float)[m]) if np.any(m) else np.asarray([], dtype=float)
    fit: Dict[str, Any] = {"status": "not_enough_points"}
    # 条件分岐: `x.size >= 10` を満たす経路を評価する。
    if x.size >= 10:
        slope, intercept = np.polyfit(x, y, deg=1)
        yhat = intercept + slope * x
        resid = y - yhat
        fit = {
            "status": "ok",
            "model": "log10(g_obs)=intercept+slope*log10(g_bar)",
            "n_used": int(x.size),
            "slope": float(slope),
            "intercept": float(intercept),
            "rms_residual_dex": float(np.sqrt(np.mean(resid * resid))),
            "std_residual_dex": float(np.std(resid)),
            "residual_quantiles_dex": {
                "q05": float(np.quantile(resid, 0.05)),
                "q50": float(np.quantile(resid, 0.50)),
                "q95": float(np.quantile(resid, 0.95)),
            },
        }

    out_metrics = out_dir / "sparc_rar_metrics.json"
    metrics = {
        "generated_utc": _utc_now(),
        "inputs": {
            "rotmod_zip": _rel(rotmod_zip),
            "upsilon_disk": float(args.upsilon_disk),
            "upsilon_bulge": float(args.upsilon_bulge),
            "constants": {"KPC_TO_M": KPC_TO_M, "KM_TO_M": KM_TO_M},
        },
        "counts": {
            "n_galaxies": int(len(per_galaxy)),
            "n_points_total": int(len(points)),
            "n_points_positive": int(np.count_nonzero(m)),
        },
        "fit_loglog": fit,
        "outputs": {"csv": _rel(out_csv), "png": _rel(out_png) if out_png.exists() else ""},
    }
    _write_json(out_metrics, metrics)

    try:
        worklog.append_event(
            "cosmology.sparc_rar_from_rotmod",
            {
                "csv": _rel(out_csv),
                "metrics": _rel(out_metrics),
                "png": _rel(out_png) if out_png.exists() else "",
                "n_galaxies": int(len(per_galaxy)),
                "n_points": int(len(points)),
            },
        )
    except Exception:
        pass

    print(json.dumps({"csv": _rel(out_csv), "n_points": len(points)}, ensure_ascii=False))
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
