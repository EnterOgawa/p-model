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
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.eht.eht_shadow_compare import (  # noqa: E402
    _kerr_shadow_boundary_stats,
    _kerr_shadow_diameter_coeff_from_stats,
)
from scripts.summary import worklog  # noqa: E402


# 関数: `_repo_root` の入出力契約と処理意図を定義する。
def _repo_root() -> Path:
    return _ROOT


# 関数: `_set_japanese_font` の入出力契約と処理意図を定義する。

def _set_japanese_font() -> None:
    try:
        import matplotlib as mpl
        import matplotlib.font_manager as fm

        preferred = ["Yu Gothic", "Meiryo", "BIZ UDGothic", "MS Gothic", "Yu Mincho", "MS Mincho"]
        available = {f.name for f in fm.fontManager.ttflist}
        chosen = [name for name in preferred if name in available]
        # 条件分岐: `chosen` を満たす経路を評価する。
        if chosen:
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
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# 関数: `_load_beta` の入出力契約と処理意図を定義する。

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

    # 条件分岐: `not (math.isfinite(beta) and beta > 0)` を満たす経路を評価する。

    if not (math.isfinite(beta) and beta > 0):
        frozen_path = root / "output" / "private" / "theory" / "frozen_parameters.json"
        try:
            frozen = _read_json(frozen_path) if frozen_path.exists() else {}
            beta = float(frozen.get("beta"))
            beta_source = "output/private/theory/frozen_parameters.json:beta"
        except Exception:
            beta = float("nan")
            beta_source = "output/private/theory/frozen_parameters.json:beta (read failed)"

    # 条件分岐: `not (math.isfinite(beta) and beta > 0)` を満たす経路を評価する。

    if not (math.isfinite(beta) and beta > 0):
        beta = 1.0
        beta_source = "default(beta=1.0)"

    return float(beta), beta_source


# 関数: `_parse_object_constraints` の入出力契約と処理意図を定義する。

def _parse_object_constraints(o: Dict[str, Any]) -> Dict[str, Any]:
    key = str(o.get("key") or "")
    name = str(o.get("display_name") or key or "unknown")

    a0 = o.get("kerr_a_star_min")
    a1 = o.get("kerr_a_star_max")
    i0 = o.get("kerr_inc_deg_min")
    i1 = o.get("kerr_inc_deg_max")

    out: Dict[str, Any] = {"key": key, "name": name}
    # 条件分岐: `a0 is not None` を満たす経路を評価する。
    if a0 is not None:
        out["a_star_min"] = float(a0)

    # 条件分岐: `a1 is not None` を満たす経路を評価する。

    if a1 is not None:
        out["a_star_max"] = float(a1)

    # 条件分岐: `i0 is not None` を満たす経路を評価する。

    if i0 is not None:
        out["inc_deg_min"] = float(i0)

    # 条件分岐: `i1 is not None` を満たす経路を評価する。

    if i1 is not None:
        out["inc_deg_max"] = float(i1)

    return out


# 関数: `_finite_minmax_with_location` の入出力契約と処理意図を定義する。

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
        # 条件分岐: `not (float(inc_min) <= float(inc) <= float(inc_max))` を満たす経路を評価する。
        if not (float(inc_min) <= float(inc) <= float(inc_max)):
            continue

        row = coeff_grid[i]
        for j, a in enumerate(a_values):
            # 条件分岐: `not (float(a_min) <= float(a) <= float(a_max))` を満たす経路を評価する。
            if not (float(a_min) <= float(a) <= float(a_max)):
                continue

            coeff = float(row[j])
            # 条件分岐: `not (math.isfinite(coeff) and coeff > 0)` を満たす経路を評価する。
            if not (math.isfinite(coeff) and coeff > 0):
                continue

            # 条件分岐: `best_min is None or coeff < best_min[0]` を満たす経路を評価する。

            if best_min is None or coeff < best_min[0]:
                best_min = (coeff, float(a), float(inc))

            # 条件分岐: `best_max is None or coeff > best_max[0]` を満たす経路を評価する。

            if best_max is None or coeff > best_max[0]:
                best_max = (coeff, float(a), float(inc))

    # 条件分岐: `best_min is None or best_max is None` を満たす経路を評価する。

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


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    ap = argparse.ArgumentParser(description="Quantify Kerr shadow effective-diameter definition sensitivity.")
    ap.add_argument("--a-max", type=float, default=0.999)
    ap.add_argument("--a-samples", type=int, default=60)
    ap.add_argument("--inc-min", type=float, default=5.0)
    ap.add_argument("--inc-max", type=float, default=90.0)
    ap.add_argument("--inc-step", type=float, default=5.0)
    ap.add_argument("--n-r", type=int, default=2500)
    ap.add_argument("--outdir", type=str, default="", help="Override output dir (default: output/private/eht)")
    args = ap.parse_args()

    root = _repo_root()
    outdir = Path(args.outdir) if str(args.outdir).strip() else (root / "output" / "private" / "eht")
    outdir.mkdir(parents=True, exist_ok=True)

    out_png = outdir / "eht_kerr_shadow_coeff_definition_sensitivity.png"
    out_public_png = outdir / "eht_kerr_shadow_coeff_definition_sensitivity_public.png"
    out_json = outdir / "eht_kerr_shadow_coeff_definition_sensitivity_metrics.json"

    a_max = max(0.0, min(0.999, float(args.a_max)))
    a_samples = max(2, int(args.a_samples))
    inc_min = float(args.inc_min)
    inc_max = float(args.inc_max)
    # 条件分岐: `inc_max < inc_min` を満たす経路を評価する。
    if inc_max < inc_min:
        inc_min, inc_max = inc_max, inc_min

    inc_min = max(0.0, min(90.0, inc_min))
    inc_max = max(0.0, min(90.0, inc_max))
    inc_min_eff = max(5.0, inc_min)  # avoid inc=0 singularity in standard parameterization
    inc_max_eff = max(inc_min_eff, inc_max)
    inc_step = float(args.inc_step)
    # 条件分岐: `inc_step <= 0` を満たす経路を評価する。
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
    # 条件分岐: `not inc_values` を満たす経路を評価する。
    if not inc_values:
        inc_values = [float(inc_min_eff)]

    coeff_schw = 2.0 * math.sqrt(27.0)
    beta, beta_source = _load_beta(root)
    coeff_pmodel = 4.0 * math.e * float(beta)

    methods: List[Tuple[str, str]] = [
        ("avg_wh", "avg(width,height)"),
        ("geom_mean_wh", "sqrt(width*height)"),
        ("area_eq", "area-equivalent circle"),
        ("perimeter_eq", "perimeter-equivalent circle"),
    ]

    # Build coefficient grids for each method (inc x a).
    coeff_grids: Dict[str, List[List[float]]] = {m: [] for m, _ in methods}
    spread_rel_max = float("nan")
    spread_rel_median = float("nan")
    spread_rel_samples: List[float] = []

    for inc in inc_values:
        # Pre-compute stats per (a,inc) and derive all coefficients from it.
        row_stats: List[Dict[str, float]] = []
        for a in a_values:
            row_stats.append(_kerr_shadow_boundary_stats(float(a), float(inc), n_r=int(n_r)))

        for m, _label in methods:
            row: List[float] = []
            for st in row_stats:
                row.append(float(_kerr_shadow_diameter_coeff_from_stats(st, method=m)))

            coeff_grids[m].append(row)

        # track definition spread per point

        for j, a in enumerate(a_values):
            vals: List[float] = []
            for m, _ in methods:
                v = float(coeff_grids[m][-1][j])
                # 条件分岐: `math.isfinite(v) and v > 0` を満たす経路を評価する。
                if math.isfinite(v) and v > 0:
                    vals.append(v)

            # 条件分岐: `len(vals) >= 2` を満たす経路を評価する。

            if len(vals) >= 2:
                lo = min(vals)
                hi = max(vals)
                mid = 0.5 * (lo + hi)
                # 条件分岐: `mid > 0` を満たす経路を評価する。
                if mid > 0:
                    spread_rel = (hi - lo) / mid
                    spread_rel_samples.append(float(spread_rel))

    # 条件分岐: `spread_rel_samples` を満たす経路を評価する。

    if spread_rel_samples:
        spread_rel_max = float(max(spread_rel_samples))
        spread_rel_median = float(np.median(np.array(spread_rel_samples, dtype=float)))

    # Global ranges

    ranges: Dict[str, Any] = {}
    env_min: Optional[Tuple[float, float, float, str]] = None
    env_max: Optional[Tuple[float, float, float, str]] = None
    for m, label in methods:
        mm = _finite_minmax_with_location(
            coeff_grids[m],
            a_values,
            inc_values,
            a_min=0.0,
            a_max=float(a_max),
            inc_min=float(inc_min_eff),
            inc_max=float(inc_max_eff),
        )
        ranges[m] = {"label": label, **mm}

        cmin = mm.get("coeff_min")
        cmax = mm.get("coeff_max")
        # 条件分岐: `cmin is not None` を満たす経路を評価する。
        if cmin is not None:
            a_at = float(mm.get("coeff_min_at", {}).get("a_star"))
            i_at = float(mm.get("coeff_min_at", {}).get("inc_deg"))
            # 条件分岐: `env_min is None or float(cmin) < env_min[0]` を満たす経路を評価する。
            if env_min is None or float(cmin) < env_min[0]:
                env_min = (float(cmin), a_at, i_at, m)

        # 条件分岐: `cmax is not None` を満たす経路を評価する。

        if cmax is not None:
            a_at = float(mm.get("coeff_max_at", {}).get("a_star"))
            i_at = float(mm.get("coeff_max_at", {}).get("inc_deg"))
            # 条件分岐: `env_max is None or float(cmax) > env_max[0]` を満たす経路を評価する。
            if env_max is None or float(cmax) > env_max[0]:
                env_max = (float(cmax), a_at, i_at, m)

    envelope = {
        "coeff_min": (env_min[0] if env_min is not None else None),
        "coeff_min_at": (
            {
                "a_star": env_min[1],
                "inc_deg": env_min[2],
                "method": ranges.get(env_min[3], {}).get("label", env_min[3]),
            }
            if env_min is not None
            else None
        ),
        "coeff_max": (env_max[0] if env_max is not None else None),
        "coeff_max_at": (
            {
                "a_star": env_max[1],
                "inc_deg": env_max[2],
                "method": ranges.get(env_max[3], {}).get("label", env_max[3]),
            }
            if env_max is not None
            else None
        ),
    }

    # Object overlays (optional constraints in eht_black_holes.json)
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
        # 条件分岐: `not c.get("key")` を満たす経路を評価する。
        if not c.get("key"):
            continue

        a0 = float(c.get("a_star_min", 0.0))
        a1 = float(c.get("a_star_max", float(a_max)))
        i0 = float(c.get("inc_deg_min", float(inc_min_eff)))
        i1 = float(c.get("inc_deg_max", float(inc_max_eff)))
        a0 = max(0.0, min(float(a_max), a0))
        a1 = max(0.0, min(float(a_max), a1))
        # 条件分岐: `a1 < a0` を満たす経路を評価する。
        if a1 < a0:
            a0, a1 = a1, a0

        i0 = max(float(inc_min_eff), min(float(inc_max_eff), i0))
        i1 = max(float(inc_min_eff), min(float(inc_max_eff), i1))
        # 条件分岐: `i1 < i0` を満たす経路を評価する。
        if i1 < i0:
            i0, i1 = i1, i0

        per_method: Dict[str, Any] = {}
        env_min_obj: Optional[Tuple[float, float, float, str]] = None
        env_max_obj: Optional[Tuple[float, float, float, str]] = None
        for m, label in methods:
            mm = _finite_minmax_with_location(
                coeff_grids[m],
                a_values,
                inc_values,
                a_min=float(a0),
                a_max=float(a1),
                inc_min=float(i0),
                inc_max=float(i1),
            )
            per_method[m] = {"label": label, **mm}

            cmin = mm.get("coeff_min")
            cmax = mm.get("coeff_max")
            # 条件分岐: `cmin is not None` を満たす経路を評価する。
            if cmin is not None:
                a_at = float(mm.get("coeff_min_at", {}).get("a_star"))
                i_at = float(mm.get("coeff_min_at", {}).get("inc_deg"))
                # 条件分岐: `env_min_obj is None or float(cmin) < env_min_obj[0]` を満たす経路を評価する。
                if env_min_obj is None or float(cmin) < env_min_obj[0]:
                    env_min_obj = (float(cmin), a_at, i_at, m)

            # 条件分岐: `cmax is not None` を満たす経路を評価する。

            if cmax is not None:
                a_at = float(mm.get("coeff_max_at", {}).get("a_star"))
                i_at = float(mm.get("coeff_max_at", {}).get("inc_deg"))
                # 条件分岐: `env_max_obj is None or float(cmax) > env_max_obj[0]` を満たす経路を評価する。
                if env_max_obj is None or float(cmax) > env_max_obj[0]:
                    env_max_obj = (float(cmax), a_at, i_at, m)

        env_obj = {
            "coeff_min": (env_min_obj[0] if env_min_obj is not None else None),
            "coeff_min_at": (
                {
                    "a_star": env_min_obj[1],
                    "inc_deg": env_min_obj[2],
                    "method": per_method.get(env_min_obj[3], {}).get("label", env_min_obj[3]),
                }
                if env_min_obj is not None
                else None
            ),
            "coeff_max": (env_max_obj[0] if env_max_obj is not None else None),
            "coeff_max_at": (
                {
                    "a_star": env_max_obj[1],
                    "inc_deg": env_max_obj[2],
                    "method": per_method.get(env_max_obj[3], {}).get("label", env_max_obj[3]),
                }
                if env_max_obj is not None
                else None
            ),
        }

        delta_p_over_kerr_min_pct = (
            100.0 * (float(coeff_pmodel) / float(env_obj["coeff_max"]) - 1.0)
            if env_obj.get("coeff_max") is not None and float(env_obj["coeff_max"]) > 0
            else None
        )
        delta_p_over_kerr_max_pct = (
            100.0 * (float(coeff_pmodel) / float(env_obj["coeff_min"]) - 1.0)
            if env_obj.get("coeff_min") is not None and float(env_obj["coeff_min"]) > 0
            else None
        )

        overlays.append(
            {
                **c,
                "a_star_min_used": float(a0),
                "a_star_max_used": float(a1),
                "inc_deg_min_used": float(i0),
                "inc_deg_max_used": float(i1),
                "ranges_per_method": per_method,
                "envelope": env_obj,
                "delta_p_over_kerr_min_percent": float(delta_p_over_kerr_min_pct) if delta_p_over_kerr_min_pct is not None else None,
                "delta_p_over_kerr_max_percent": float(delta_p_over_kerr_max_pct) if delta_p_over_kerr_max_pct is not None else None,
            }
        )

    # Plot: global per-method coefficient ranges + envelope

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[err] matplotlib is required: {e}")
        return 2

    _set_japanese_font()

    fig, ax = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)

    rows_plot: List[Tuple[str, str, Optional[float], Optional[float]]] = []
    for m, label in methods:
        mm = ranges.get(m, {})
        rows_plot.append((m, label, mm.get("coeff_min"), mm.get("coeff_max")))

    rows_plot.append(("envelope", "envelope (all definitions)", envelope.get("coeff_min"), envelope.get("coeff_max")))

    y = list(range(len(rows_plot)))
    for yi, (_m, label, cmin, cmax) in enumerate(rows_plot):
        # 条件分岐: `cmin is None or cmax is None` を満たす経路を評価する。
        if cmin is None or cmax is None:
            continue

        ax.hlines(yi, float(cmin), float(cmax), color="#1f77b4" if _m != "envelope" else "#d62728", linewidth=5.0)
        ax.plot([float(cmin), float(cmax)], [yi, yi], "o", color="#1f77b4" if _m != "envelope" else "#d62728", alpha=0.9)
        ax.text(float(cmax) + 0.02, yi, label, va="center", ha="left", fontsize=10)

    ax.axvline(coeff_schw, color="#333333", linestyle="--", linewidth=1.4, label=f"Schwarzschild 2√27={coeff_schw:.3f}")
    ax.axvline(coeff_pmodel, color="#2ca02c", linestyle="-", linewidth=1.4, label=f"P-model 4eβ={coeff_pmodel:.3f}")

    ax.set_yticks([])
    ax.set_xlabel("shadow diameter coefficient  (× GM/(c²D))")
    ax.set_title("Kerr shadow 直径係数：定義依存（effective diameter）の感度（reference systematic）")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(loc="lower left")

    note = (
        f"β={beta:g}（{beta_source}）\n"
        f"definition spread max ≈ {100.0*spread_rel_max:.2f}% / median ≈ {100.0*spread_rel_median:.2f}%"
        if math.isfinite(spread_rel_max) and math.isfinite(spread_rel_median)
        else f"β={beta:g}（{beta_source}）"
    )
    ax.text(0.02, 0.98, note, transform=ax.transAxes, va="top", ha="left", fontsize=10, color="#444")

    fig.savefig(out_png, dpi=180)
    fig.savefig(out_public_png, dpi=180)
    plt.close(fig)

    generated_utc = datetime.now(timezone.utc).isoformat()
    payload: Dict[str, Any] = {
        "generated_utc": generated_utc,
        "inputs": {"eht_black_holes_json": str(obj_in.relative_to(root)).replace("\\", "/"), "beta_source": beta_source},
        "outputs": {
            "png": str(out_png.relative_to(root)).replace("\\", "/"),
            "public_png": str(out_public_png.relative_to(root)).replace("\\", "/"),
            "metrics_json": str(out_json.relative_to(root)).replace("\\", "/"),
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
        "methods": [{"id": m, "label": label} for m, label in methods],
        "ranges_global": {"per_method": ranges, "envelope": envelope},
        "definition_spread_rel_max": float(spread_rel_max) if math.isfinite(spread_rel_max) else None,
        "definition_spread_rel_median": float(spread_rel_median) if math.isfinite(spread_rel_median) else None,
        "object_overlays": overlays,
        "policy": {
            "kerr_effective_diameter_definition_policy": "use envelope(across methods) as GR systematic in κ budget",
            "notes": [
                "定義依存は spin/inc と別の系統として扱い得るが、ここでは保守的に envelope として吸収する。",
                "この envelope は EHT の放射モデルfitではなく、GRジオデシック由来の reference systematic である。",
            ],
        },
    }

    _write_json(out_json, payload)

    try:
        worklog.append_event(
            {
                "ts_utc": generated_utc,
                "topic": "eht",
                "action": "eht_kerr_shadow_coeff_definition_sensitivity",
                "outputs": [
                    str(out_png.relative_to(root)).replace("\\", "/"),
                    str(out_public_png.relative_to(root)).replace("\\", "/"),
                    str(out_json.relative_to(root)).replace("\\", "/"),
                ],
                "metrics": {"methods": int(len(methods)), "a_samples": int(a_samples), "inc_bins": int(len(inc_values))},
            }
        )
    except Exception:
        pass

    print(f"[ok] png: {out_png}")
    print(f"[ok] json: {out_json}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
