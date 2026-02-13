#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.eht.eht_shadow_compare import _kerr_shadow_diameter_coeff_avg_width_height  # noqa: E402
from scripts.summary import worklog  # noqa: E402


def _repo_root() -> Path:
    return _ROOT


def _set_japanese_font() -> None:
    try:
        import matplotlib as mpl
        import matplotlib.font_manager as fm

        preferred = ["Yu Gothic", "Meiryo", "BIZ UDGothic", "MS Gothic", "Yu Mincho", "MS Mincho"]
        available = {f.name for f in fm.fontManager.ttflist}
        chosen = [name for name in preferred if name in available]
        if chosen:
            mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
            mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_beta(root: Path) -> Tuple[float, str]:
    eht_json = root / "data" / "eht" / "eht_black_holes.json"
    beta = float("nan")
    beta_source = "none"
    try:
        ref = _read_json(eht_json) if eht_json.exists() else {}
        beta = float(((ref.get("pmodel") or {}).get("beta")) or float("nan"))
        beta_source = "data/eht/eht_black_holes.json:pmodel.beta"
    except Exception:
        beta = float("nan")
        beta_source = "data/eht/eht_black_holes.json:pmodel.beta (read failed)"

    if not (math.isfinite(beta) and beta > 0):
        frozen_path = root / "output" / "private" / "theory" / "frozen_parameters.json"
        try:
            frozen = _read_json(frozen_path) if frozen_path.exists() else {}
            beta = float(frozen.get("beta"))
            beta_source = "output/private/theory/frozen_parameters.json:beta"
        except Exception:
            beta = float("nan")
            beta_source = "output/private/theory/frozen_parameters.json:beta (read failed)"

    if not (math.isfinite(beta) and beta > 0):
        beta = 1.0
        beta_source = "default(beta=1.0)"
    return float(beta), beta_source


def _parse_object_constraints(o: Dict[str, Any]) -> Dict[str, Any]:
    key = str(o.get("key") or "")
    name = str(o.get("display_name") or key or "unknown")

    a0 = o.get("kerr_a_star_min")
    a1 = o.get("kerr_a_star_max")
    i0 = o.get("kerr_inc_deg_min")
    i1 = o.get("kerr_inc_deg_max")

    out: Dict[str, Any] = {"key": key, "name": name}
    if a0 is not None:
        out["a_star_min"] = float(a0)
    if a1 is not None:
        out["a_star_max"] = float(a1)
    if i0 is not None:
        out["inc_deg_min"] = float(i0)
    if i1 is not None:
        out["inc_deg_max"] = float(i1)
    return out


def _finite_minmax_with_location(
    coeff_grid: List[List[float]],
    a_values: List[float],
    inc_values: List[float],
    *,
    a_min: float,
    a_max: float,
    inc_min: float,
    inc_max: float,
) -> Dict[str, Any]:
    best_min: Optional[Tuple[float, float, float]] = None  # (coeff, a, inc)
    best_max: Optional[Tuple[float, float, float]] = None

    for i, inc in enumerate(inc_values):
        if not (float(inc_min) <= float(inc) <= float(inc_max)):
            continue
        row = coeff_grid[i]
        for j, a in enumerate(a_values):
            if not (float(a_min) <= float(a) <= float(a_max)):
                continue
            coeff = float(row[j])
            if not (math.isfinite(coeff) and coeff > 0):
                continue
            if best_min is None or coeff < best_min[0]:
                best_min = (coeff, float(a), float(inc))
            if best_max is None or coeff > best_max[0]:
                best_max = (coeff, float(a), float(inc))

    if best_min is None or best_max is None:
        return {"coeff_min": None, "coeff_max": None}
    coeff_min, a_at_min, inc_at_min = best_min
    coeff_max, a_at_max, inc_at_max = best_max
    return {
        "coeff_min": float(coeff_min),
        "coeff_min_at": {"a_star": float(a_at_min), "inc_deg": float(inc_at_min)},
        "coeff_max": float(coeff_max),
        "coeff_max_at": {"a_star": float(a_at_max), "inc_deg": float(inc_at_max)},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Kerr shadow coefficient grid (reference systematic; not an EHT emission fit).")
    ap.add_argument("--a-max", type=float, default=0.999)
    ap.add_argument("--a-samples", type=int, default=60)
    ap.add_argument("--inc-min", type=float, default=5.0)
    ap.add_argument("--inc-max", type=float, default=90.0)
    ap.add_argument("--inc-step", type=float, default=5.0)
    ap.add_argument("--n-r", type=int, default=2500, help="Radial grid points per (a*,inc) evaluation.")
    ap.add_argument("--outdir", type=str, default="", help="Override output dir (default: output/private/eht)")
    args = ap.parse_args()

    root = _repo_root()
    outdir = Path(args.outdir) if str(args.outdir).strip() else (root / "output" / "private" / "eht")
    outdir.mkdir(parents=True, exist_ok=True)
    out_png = outdir / "eht_kerr_shadow_coeff_grid.png"
    out_public_png = outdir / "eht_kerr_shadow_coeff_grid_public.png"
    out_json = outdir / "eht_kerr_shadow_coeff_grid_metrics.json"

    a_max = float(args.a_max)
    a_max = max(0.0, min(0.999, a_max))
    a_samples = max(2, int(args.a_samples))
    inc_min = float(args.inc_min)
    inc_max = float(args.inc_max)
    if inc_max < inc_min:
        inc_min, inc_max = inc_max, inc_min
    inc_min = max(0.0, min(90.0, inc_min))
    inc_max = max(0.0, min(90.0, inc_max))
    # avoid inc=0° where the standard (alpha,beta) parameterization becomes singular.
    inc_min_eff = max(5.0, inc_min)
    inc_max_eff = max(inc_min_eff, inc_max)
    inc_step = float(args.inc_step)
    if inc_step <= 0:
        inc_step = 5.0
    n_r = max(800, int(args.n_r))

    try:
        import numpy as np
    except Exception as e:
        print(f"[err] numpy is required: {e}")
        return 2

    a_values = np.linspace(0.0, float(a_max), int(a_samples), dtype=float).tolist()
    inc_values = np.arange(float(inc_min_eff), float(inc_max_eff) + 1e-9, float(inc_step), dtype=float).tolist()
    if not inc_values:
        inc_values = [float(inc_min_eff)]

    coeff_schw = 2.0 * math.sqrt(27.0)
    beta, beta_source = _load_beta(root)
    coeff_pmodel = 4.0 * math.e * float(beta)

    # Compute coefficient grid.
    coeff_grid: List[List[float]] = []
    for inc in inc_values:
        row: List[float] = []
        for a in a_values:
            if abs(float(a)) < 1e-12:
                row.append(float(coeff_schw))
                continue
            coeff = _kerr_shadow_diameter_coeff_avg_width_height(float(a), float(inc), n_r=int(n_r))
            row.append(float(coeff))
        coeff_grid.append(row)

    # Global min/max (within the computed grid).
    full_minmax = _finite_minmax_with_location(
        coeff_grid,
        a_values,
        inc_values,
        a_min=0.0,
        a_max=float(a_max),
        inc_min=float(inc_min_eff),
        inc_max=float(inc_max_eff),
    )

    # Object overlays (using eht_black_holes.json optional constraints).
    obj_in = root / "data" / "eht" / "eht_black_holes.json"
    objects: List[Dict[str, Any]] = []
    try:
        eht = _read_json(obj_in) if obj_in.exists() else {}
        objects = [o for o in (eht.get("objects") or []) if isinstance(o, dict)]
    except Exception:
        objects = []

    overlays: List[Dict[str, Any]] = []
    for o in objects:
        c = _parse_object_constraints(o)
        if not c.get("key"):
            continue
        a0 = float(c.get("a_star_min", 0.0))
        a1 = float(c.get("a_star_max", float(a_max)))
        i0 = float(c.get("inc_deg_min", float(inc_min_eff)))
        i1 = float(c.get("inc_deg_max", float(inc_max_eff)))
        # clip
        a0 = max(0.0, min(float(a_max), a0))
        a1 = max(0.0, min(float(a_max), a1))
        if a1 < a0:
            a0, a1 = a1, a0
        i0 = max(float(inc_min_eff), min(float(inc_max_eff), i0))
        i1 = max(float(inc_min_eff), min(float(inc_max_eff), i1))
        if i1 < i0:
            i0, i1 = i1, i0

        mm = _finite_minmax_with_location(
            coeff_grid,
            a_values,
            inc_values,
            a_min=float(a0),
            a_max=float(a1),
            inc_min=float(i0),
            inc_max=float(i1),
        )
        coeff_min = mm.get("coeff_min")
        coeff_max = mm.get("coeff_max")
        delta_p_over_kerr_min_pct = (
            100.0 * (float(coeff_pmodel) / float(coeff_max) - 1.0) if coeff_max and float(coeff_max) > 0 else None
        )
        delta_p_over_kerr_max_pct = (
            100.0 * (float(coeff_pmodel) / float(coeff_min) - 1.0) if coeff_min and float(coeff_min) > 0 else None
        )
        overlays.append(
            {
                **c,
                "a_star_min_used": float(a0),
                "a_star_max_used": float(a1),
                "inc_deg_min_used": float(i0),
                "inc_deg_max_used": float(i1),
                **mm,
                "delta_p_over_kerr_min_percent": float(delta_p_over_kerr_min_pct) if delta_p_over_kerr_min_pct is not None else None,
                "delta_p_over_kerr_max_percent": float(delta_p_over_kerr_max_pct) if delta_p_over_kerr_max_pct is not None else None,
            }
        )

    # Plot
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception as e:
        print(f"[err] matplotlib is required: {e}")
        return 2

    _set_japanese_font()

    coeff = np.array(coeff_grid, dtype=float)
    ratio_schw_pct = 100.0 * (coeff / float(coeff_schw) - 1.0)
    ratio_p_over_kerr_pct = 100.0 * (float(coeff_pmodel) / coeff - 1.0)
    ratio_p_over_kerr_pct[~np.isfinite(ratio_p_over_kerr_pct)] = np.nan

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10.5, 8.0), constrained_layout=True)

    extent = [min(a_values), max(a_values), min(inc_values), max(inc_values)]
    # Panel 1: Kerr deviation from Schwarzschild
    ax0 = axes[0]
    vmin0 = float(np.nanmin(ratio_schw_pct)) if np.isfinite(np.nanmin(ratio_schw_pct)) else -10.0
    vmax0 = float(np.nanmax(ratio_schw_pct)) if np.isfinite(np.nanmax(ratio_schw_pct)) else 0.0
    vmax0 = max(0.0, vmax0)
    vmin0 = min(vmin0, -1e-6)
    im0 = ax0.imshow(
        ratio_schw_pct,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="RdBu_r",
        vmin=vmin0,
        vmax=vmax0,
    )
    ax0.set_title("Kerr shadow 直径係数（avg(width,height)）の Schwarzschild からの変化率（%）")
    ax0.set_xlabel("a* (dimensionless spin)")
    ax0.set_ylabel("inc (deg)")
    cb0 = fig.colorbar(im0, ax=ax0)
    cb0.set_label("Δ% vs Schwarzschild")

    # Panel 2: P-model vs Kerr difference
    ax1 = axes[1]
    vmin1 = float(np.nanmin(ratio_p_over_kerr_pct)) if np.isfinite(np.nanmin(ratio_p_over_kerr_pct)) else 0.0
    vmax1 = float(np.nanmax(ratio_p_over_kerr_pct)) if np.isfinite(np.nanmax(ratio_p_over_kerr_pct)) else 15.0
    im1 = ax1.imshow(
        ratio_p_over_kerr_pct,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="viridis",
        vmin=vmin1,
        vmax=vmax1,
    )
    ax1.set_title(f"P-model 係数 (4eβ) と Kerr 係数の差（%）：β={beta:g}（{beta_source}）")
    ax1.set_xlabel("a* (dimensionless spin)")
    ax1.set_ylabel("inc (deg)")
    cb1 = fig.colorbar(im1, ax=ax1)
    cb1.set_label("100*(coeff_P/coeff_Kerr - 1) [%]")

    # Overlays (rectangles)
    colors = ["white", "black", "magenta", "cyan", "yellow"]
    legend_handles = []
    legend_labels = []
    for idx, ov in enumerate(overlays):
        try:
            a0 = float(ov["a_star_min_used"])
            a1 = float(ov["a_star_max_used"])
            i0 = float(ov["inc_deg_min_used"])
            i1 = float(ov["inc_deg_max_used"])
        except Exception:
            continue
        col = colors[idx % len(colors)]
        r0 = Rectangle((a0, i0), (a1 - a0), (i1 - i0), fill=False, linewidth=2.0, edgecolor=col)
        r1 = Rectangle((a0, i0), (a1 - a0), (i1 - i0), fill=False, linewidth=2.0, edgecolor=col)
        ax0.add_patch(r0)
        ax1.add_patch(r1)
        legend_handles.append(r0)
        legend_labels.append(str(ov.get("name") or ov.get("key") or f"obj{idx}"))

    if legend_handles:
        ax0.legend(legend_handles, legend_labels, loc="lower right", framealpha=0.85, fontsize=9)

    note = (
        "注: これは Kerr ジオデシックからの reference systematic（放射モデルfitではない）。\n"
        f"Schwarzschild 係数=2√27={coeff_schw:.6g}, P-model 係数=4eβ={coeff_pmodel:.6g}"
    )
    fig.suptitle(note, fontsize=10)

    fig.savefig(out_png, dpi=160)
    fig.savefig(out_public_png, dpi=160)
    plt.close(fig)

    # Metrics payload
    generated_utc = datetime.now(timezone.utc).isoformat()
    payload: Dict[str, Any] = {
        "generated_utc": generated_utc,
        "inputs": {
            "eht_black_holes_json": str(obj_in.relative_to(root)).replace("\\", "/"),
            "beta_source": beta_source,
        },
        "outputs": {
            "png": str(out_png.relative_to(root)).replace("\\", "/"),
            "public_png": str(out_public_png.relative_to(root)).replace("\\", "/"),
            "metrics_json": str(out_json.relative_to(root)).replace("\\", "/"),
        },
        "definition": {
            "kerr_effective_diameter": "avg(width,height) in Bardeen(1973) alpha/beta plane (reference systematic)",
            "notes": [
                "Kerr shadow は非円形であり、単一の直径への写像は定義依存。",
                "ここでは (width+height)/2 を採用し、spin/inclination 依存の上限・下限を示す。",
                "inc=0° は標準表現が特異なので 5° 以上を使用。",
            ],
        },
        "grid": {
            "a_star_min": 0.0,
            "a_star_max": float(a_max),
            "a_samples": int(a_samples),
            "inc_deg_min": float(inc_min_eff),
            "inc_deg_max": float(inc_max_eff),
            "inc_step_deg": float(inc_step),
            "n_r": int(n_r),
        },
        "coefficients": {
            "coeff_schwarzschild": float(coeff_schw),
            "beta": float(beta),
            "coeff_pmodel": float(coeff_pmodel),
            "delta_p_over_schwarzschild_percent": 100.0 * (float(coeff_pmodel) / float(coeff_schw) - 1.0),
        },
        "full_range_on_grid": full_minmax,
        "object_overlays": overlays,
    }

    # add p-vs-kerr global min/max on grid (using coeff_min/max)
    try:
        cmin = full_minmax.get("coeff_min")
        cmax = full_minmax.get("coeff_max")
        if cmin and cmax and float(cmin) > 0 and float(cmax) > 0:
            payload["coefficients"]["delta_p_over_kerr_min_percent"] = 100.0 * (float(coeff_pmodel) / float(cmax) - 1.0)
            payload["coefficients"]["delta_p_over_kerr_max_percent"] = 100.0 * (float(coeff_pmodel) / float(cmin) - 1.0)
    except Exception:
        pass

    _write_json(out_json, payload)

    try:
        worklog.append_event(
            {
                "ts_utc": generated_utc,
                "topic": "eht",
                "action": "eht_kerr_shadow_coeff_grid",
                "outputs": [
                    str(out_png.relative_to(root)).replace("\\", "/"),
                    str(out_public_png.relative_to(root)).replace("\\", "/"),
                    str(out_json.relative_to(root)).replace("\\", "/"),
                ],
                "metrics": {
                    "a_samples": int(a_samples),
                    "inc_bins": int(len(inc_values)),
                },
            }
        )
    except Exception:
        pass

    print(f"[ok] png: {out_png}")
    print(f"[ok] json: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
