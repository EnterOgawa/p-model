#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sparc_btfr_metrics.py

Phase 6 / Step 6.5（SPARC：BTFR）:
SPARC の BTFR_Lelli2019.mrt から baryonic Tully–Fisher relation（BTFR）を最小再構築し、
固定出力（metrics/図）として保存する。

入力（一次）:
- data/cosmology/sparc/raw/BTFR_Lelli2019.mrt

出力（固定）:
- output/cosmology/sparc_btfr_metrics.json
- output/cosmology/sparc_btfr_scatter.png

注意:
- ここでは BTFR の「観測側 cross-check」を目的とし、P-model の予測は導入しない。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None


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


@dataclass(frozen=True)
class BtfrRow:
    name: str
    log_mb: float
    e_log_mb: float
    inc_deg: float
    e_inc_deg: float
    vf_km_s: float
    e_vf_km_s: float
    v2exp_km_s: float
    e_v2exp_km_s: float
    v2eff_km_s: float
    e_v2eff_km_s: float
    vmax_km_s: float
    e_vmax_km_s: float
    wp20_km_s: float
    e_wp20_km_s: float
    wm50_km_s: float
    e_wm50_km_s: float
    wm50c_km_s: float
    e_wm50c_km_s: float


def _iter_rows(path: Path) -> Iterable[BtfrRow]:
    # BTFR_Lelli2019.mrt is a CDS-style table with an ASCII header and whitespace-separated rows.
    # Data lines have 19 tokens: Name + 18 numeric fields.
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith(("Title:", "Authors:", "Table:", "Byte", "Note", "====", "---")):
            continue
        if not (s[0].isalnum() or s[0] in ("_", "+", "-")):
            continue
        parts = s.split()
        if len(parts) != 19:
            continue
        try:
            name = parts[0]
            nums = [float(x) for x in parts[1:]]
        except Exception:
            continue
        yield BtfrRow(
            name=name,
            log_mb=nums[0],
            e_log_mb=nums[1],
            inc_deg=nums[2],
            e_inc_deg=nums[3],
            vf_km_s=nums[4],
            e_vf_km_s=nums[5],
            v2exp_km_s=nums[6],
            e_v2exp_km_s=nums[7],
            v2eff_km_s=nums[8],
            e_v2eff_km_s=nums[9],
            vmax_km_s=nums[10],
            e_vmax_km_s=nums[11],
            wp20_km_s=nums[12],
            e_wp20_km_s=nums[13],
            wm50_km_s=nums[14],
            e_wm50_km_s=nums[15],
            wm50c_km_s=nums[16],
            e_wm50c_km_s=nums[17],
        )


def _fit_logmb_vs_logv(x: np.ndarray, y: np.ndarray, *, w: Optional[np.ndarray]) -> Dict[str, Any]:
    if x.size < 10:
        return {"status": "not_enough_points", "n_used": int(x.size)}
    if w is None:
        slope, intercept = np.polyfit(x, y, deg=1)
    else:
        slope, intercept = np.polyfit(x, y, deg=1, w=w)
    yhat = intercept + slope * x
    resid = y - yhat
    return {
        "status": "ok",
        "model": "log10(M_b/Msun)=intercept+slope*log10(V/km_s)",
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


def _plot(out_png: Path, *, x: np.ndarray, y: np.ndarray, fit: Dict[str, Any], xlabel: str) -> None:
    if plt is None:
        return
    if x.size < 10:
        return
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, s=10, alpha=0.55, edgecolors="none")
    plt.xlabel(xlabel)
    plt.ylabel("log10 M_b [Msun]")
    plt.title("SPARC BTFR (Lelli+2019)")
    plt.grid(True, alpha=0.3)
    if fit.get("status") == "ok":
        slope = float(fit["slope"])
        intercept = float(fit["intercept"])
        xx = np.linspace(float(np.min(x)), float(np.max(x)), 200)
        yy = intercept + slope * xx
        plt.plot(xx, yy, color="black", lw=1.5, label=f"slope={slope:.3g}, rms={float(fit['rms_residual_dex']):.3g} dex")
        plt.legend(loc="best", frameon=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close()


def _extract_xy(rows: Sequence[BtfrRow], *, velocity_key: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # returns x=log10(V), y=logMb, w=1/sigma_y
    vx: List[float] = []
    vy: List[float] = []
    vw: List[float] = []
    for r in rows:
        v = getattr(r, velocity_key)
        if not np.isfinite(v) or v <= 0.0:
            continue
        if not np.isfinite(r.log_mb):
            continue
        x = math.log10(float(v))
        y = float(r.log_mb)
        vx.append(x)
        vy.append(y)
        if np.isfinite(r.e_log_mb) and r.e_log_mb > 0:
            vw.append(1.0 / float(r.e_log_mb))
        else:
            vw.append(1.0)
    return np.asarray(vx, dtype=float), np.asarray(vy, dtype=float), np.asarray(vw, dtype=float)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--btfr-mrt",
        default=str(_ROOT / "data" / "cosmology" / "sparc" / "raw" / "BTFR_Lelli2019.mrt"),
        help="Path to BTFR_Lelli2019.mrt",
    )
    p.add_argument(
        "--out-dir",
        default=str(_ROOT / "output" / "cosmology"),
        help="Output directory (default: output/cosmology)",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    btfr_path = Path(args.btfr_mrt)
    if not btfr_path.exists():
        raise FileNotFoundError(f"missing BTFR mrt: {btfr_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(_iter_rows(btfr_path))

    # Fit for multiple velocity definitions for cross-check (primary is Vf)
    velocity_defs = [
        ("vf_km_s", "Vf (flat)"),
        ("vmax_km_s", "Vmax"),
        ("v2exp_km_s", "V2exp"),
        ("v2eff_km_s", "V2eff"),
    ]

    fits: Dict[str, Any] = {}
    primary_key = "vf_km_s"
    primary_scatter = float("inf")
    primary_xy: Tuple[np.ndarray, np.ndarray, np.ndarray] = (np.asarray([]), np.asarray([]), np.asarray([]))

    for key, label in velocity_defs:
        x, y, w = _extract_xy(rows, velocity_key=key)
        fit_unweighted = _fit_logmb_vs_logv(x, y, w=None)
        fit_weighted = _fit_logmb_vs_logv(x, y, w=w)
        fits[key] = {"label": label, "unweighted": fit_unweighted, "weighted": fit_weighted, "n_used": int(x.size)}
        if key == primary_key:
            primary_xy = (x, y, w)
            if fit_weighted.get("status") == "ok":
                primary_scatter = float(fit_weighted["rms_residual_dex"])

    out_png = out_dir / "sparc_btfr_scatter.png"
    x, y, _w = primary_xy
    primary_fit = fits.get(primary_key, {}).get("weighted", {})
    _plot(out_png, x=x, y=y, fit=primary_fit, xlabel="log10 Vf [km/s]")

    out_metrics = out_dir / "sparc_btfr_metrics.json"
    metrics = {
        "generated_utc": _utc_now(),
        "inputs": {"btfr_mrt": _rel(btfr_path)},
        "counts": {"n_rows_total": int(len(rows)), "n_rows_with_vf": int(fits.get(primary_key, {}).get("n_used", 0))},
        "fits": fits,
        "primary": {"velocity_key": primary_key, "velocity_label": "Vf (flat)", "weighted_rms_residual_dex": primary_scatter},
        "outputs": {"metrics": _rel(out_metrics), "png": _rel(out_png) if out_png.exists() else ""},
    }
    _write_json(out_metrics, metrics)

    try:
        worklog.append_event(
            "cosmology.sparc_btfr_metrics",
            {"metrics": _rel(out_metrics), "png": _rel(out_png) if out_png.exists() else "", "n_rows": int(len(rows))},
        )
    except Exception:
        pass

    print(json.dumps({"metrics": _rel(out_metrics), "n_rows": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

