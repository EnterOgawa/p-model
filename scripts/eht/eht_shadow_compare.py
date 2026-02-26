#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

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
        # 条件分岐: `chosen` を満たす経路を評価する。
        if chosen:
            mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
            mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_phase4_coeff_ratio(
    root: Path, coeff_ratio_baseline: float
) -> Tuple[float, float, str, Optional[str]]:
    """Resolve phase-4 (strong-field corrected) coefficient ratio.

    Returns:
      (coeff_ratio, coeff_diff_percent, source, source_relpath_or_none)
    """
    diff_baseline = (
        (coeff_ratio_baseline - 1.0) * 100.0 if math.isfinite(coeff_ratio_baseline) else float("nan")
    )
    candidates = [
        root / "output" / "public" / "theory" / "pmodel_effective_metric_n0_source_solution_audit.json",
        root / "output" / "private" / "theory" / "pmodel_effective_metric_n0_source_solution_audit.json",
    ]
    for path in candidates:
        # 条件分岐: `not path.exists()` を満たす経路を評価する。
        if not path.exists():
            continue

        try:
            payload = _read_json(path)
            budget = payload.get("ring_coefficient_budget")
            # 条件分岐: `not isinstance(budget, dict)` を満たす経路を評価する。
            if not isinstance(budget, dict):
                continue

            for key in ("core_plus_n0_gap_pct", "core_plus_n0_closed_gap_pct"):
                pct = budget.get(key)
                pct_v = float(pct)
                # 条件分岐: `math.isfinite(pct_v)` を満たす経路を評価する。
                if math.isfinite(pct_v):
                    return (
                        1.0 + (pct_v / 100.0),
                        pct_v,
                        f"{path.as_posix()}:ring_coefficient_budget.{key}",
                        path.relative_to(root).as_posix(),
                    )
        except Exception:
            continue

    return (
        coeff_ratio_baseline,
        diff_baseline,
        "baseline(4eβ / 2√27)",
        None,
    )


def _kerr_shadow_boundary_stats(a_star: float, inc_deg: float, *, n_r: int = 2500) -> Dict[str, float]:
    """Compute basic geometric stats of a Kerr shadow boundary in units of M=GM/c^2.

    Notes:
    - This is a *reference* GR systematic computation (spin/inclination + definition choices), not an
      EHT emission model fit.
    - For a=0, returns the exact Schwarzschild circle stats.
    - For Kerr, we parameterize the shadow boundary using the standard (α,β) plane (Bardeen 1973 form)
      derived from spherical photon orbits, then mirror the upper half to close the curve.
    """

    # 条件分岐: `not (math.isfinite(a_star) and math.isfinite(inc_deg))` を満たす経路を評価する。
    if not (math.isfinite(a_star) and math.isfinite(inc_deg)):
        return {"width": float("nan"), "height": float("nan"), "area": float("nan"), "perimeter": float("nan")}

    coeff_schw = 2.0 * math.sqrt(27.0)
    a = float(a_star)
    # 条件分岐: `abs(a) < 1e-12` を満たす経路を評価する。
    if abs(a) < 1e-12:
        r0 = 0.5 * coeff_schw
        return {
            "width": float(coeff_schw),
            "height": float(coeff_schw),
            "area": float(math.pi * (r0**2)),
            "perimeter": float(math.pi * coeff_schw),
        }

    import numpy as np

    inc = math.radians(float(inc_deg))
    sin_inc = math.sin(inc)
    cos_inc = math.cos(inc)
    # 条件分岐: `abs(sin_inc) < 1e-9` を満たす経路を評価する。
    if abs(sin_inc) < 1e-9:
        sin_inc = 1e-9

    cot_inc = cos_inc / sin_inc

    M = 1.0
    r_pro = 2.0 * (1.0 + math.cos((2.0 / 3.0) * math.acos(-a)))  # prograde equatorial photon orbit radius
    r_ret = 2.0 * (1.0 + math.cos((2.0 / 3.0) * math.acos(a)))  # retrograde equatorial photon orbit radius
    r_min = min(r_pro, r_ret) + 1e-6
    r_max = max(r_pro, r_ret) - 1e-6
    # 条件分岐: `not (math.isfinite(r_min) and math.isfinite(r_max)) or r_max <= r_min` を満たす経路を評価する。
    if not (math.isfinite(r_min) and math.isfinite(r_max)) or r_max <= r_min:
        r_min = 1.0 + 1e-3
        r_max = 12.0

    r = np.linspace(r_min, r_max, int(n_r))

    xi = (r**2 * (r - 3.0 * M) + (a**2) * (r + M)) / (a * (M - r))
    eta = (r**3 * (4.0 * (a**2) * M - r * (r - 3.0 * M) ** 2)) / ((a**2) * (M - r) ** 2)

    mask = eta >= 0
    # 条件分岐: `not np.any(mask)` を満たす経路を評価する。
    if not np.any(mask):
        return {"width": float("nan"), "height": float("nan"), "area": float("nan"), "perimeter": float("nan")}

    xi = xi[mask]
    eta = eta[mask]

    alpha = -xi / sin_inc
    beta2 = eta + (a**2) * (cos_inc**2) - (xi**2) * (cot_inc**2)
    mask2 = beta2 >= 0
    # 条件分岐: `not np.any(mask2)` を満たす経路を評価する。
    if not np.any(mask2):
        return {"width": float("nan"), "height": float("nan"), "area": float("nan"), "perimeter": float("nan")}

    alpha = alpha[mask2]
    beta = np.sqrt(beta2[mask2])

    # 条件分岐: `alpha.size < 3` を満たす経路を評価する。
    if alpha.size < 3:
        return {"width": float("nan"), "height": float("nan"), "area": float("nan"), "perimeter": float("nan")}

    width = float(alpha.max() - alpha.min())
    height = float(2.0 * beta.max())

    # Build a closed boundary polygon by mirroring the upper curve.
    x = np.concatenate([alpha, alpha[::-1]])
    y = np.concatenate([beta, -beta[::-1]])
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    # 条件分岐: `x.size < 3` を満たす経路を評価する。
    if x.size < 3:
        return {"width": float(width), "height": float(height), "area": float("nan"), "perimeter": float("nan")}

    # Shoelace area and polygon perimeter.

    x2 = np.concatenate([x, x[:1]])
    y2 = np.concatenate([y, y[:1]])
    area = 0.5 * abs(float(np.sum(x2[:-1] * y2[1:] - x2[1:] * y2[:-1])))

    dx = np.diff(x2)
    dy = np.diff(y2)
    perimeter = float(np.sum(np.sqrt(dx * dx + dy * dy)))
    return {"width": float(width), "height": float(height), "area": float(area), "perimeter": float(perimeter)}


def _kerr_shadow_diameter_coeff_from_stats(stats: Dict[str, float], *, method: str) -> float:
    width = float(stats.get("width", float("nan")))
    height = float(stats.get("height", float("nan")))
    area = float(stats.get("area", float("nan")))
    perimeter = float(stats.get("perimeter", float("nan")))

    # 条件分岐: `str(method) == "avg_wh"` を満たす経路を評価する。
    if str(method) == "avg_wh":
        return 0.5 * (width + height) if (math.isfinite(width) and math.isfinite(height)) else float("nan")

    # 条件分岐: `str(method) == "geom_mean_wh"` を満たす経路を評価する。

    if str(method) == "geom_mean_wh":
        return math.sqrt(width * height) if (math.isfinite(width) and math.isfinite(height) and width > 0 and height > 0) else float("nan")

    # 条件分岐: `str(method) == "area_eq"` を満たす経路を評価する。

    if str(method) == "area_eq":
        return (2.0 * math.sqrt(area / math.pi)) if (math.isfinite(area) and area > 0) else float("nan")

    # 条件分岐: `str(method) == "perimeter_eq"` を満たす経路を評価する。

    if str(method) == "perimeter_eq":
        return (perimeter / math.pi) if (math.isfinite(perimeter) and perimeter > 0) else float("nan")

    return float("nan")


def _kerr_shadow_diameter_coeff_effective(
    a_star: float,
    inc_deg: float,
    *,
    n_r: int = 2500,
    method: str = "avg_wh",
) -> float:
    """Kerr shadow effective diameter coefficient in units of GM/(c^2 D).

    Supported methods:
    - avg_wh: (width+height)/2
    - geom_mean_wh: sqrt(width*height)
    - area_eq: area-equivalent circle diameter
    - perimeter_eq: perimeter-equivalent circle diameter

    For a=0, returns the exact Schwarzschild value 2√27 for all methods.
    """

    coeff_schw = 2.0 * math.sqrt(27.0)
    a = float(a_star)
    # 条件分岐: `math.isfinite(a) and abs(a) < 1e-12` を満たす経路を評価する。
    if math.isfinite(a) and abs(a) < 1e-12:
        return float(coeff_schw)

    stats = _kerr_shadow_boundary_stats(float(a_star), float(inc_deg), n_r=int(n_r))
    return _kerr_shadow_diameter_coeff_from_stats(stats, method=str(method))


def _kerr_shadow_diameter_coeff_avg_width_height(a_star: float, inc_deg: float, *, n_r: int = 2500) -> float:
    """Kerr shadow "effective diameter coefficient" in units of GM/(c^2 D).

    Returns (width+height)/2 in units of M (where M=GM/c^2), i.e.:
      coeff = θ_shadow / (GM/(c^2 D))

    Notes:
    - This is a *reference* GR systematic range (spin/inclination), not an EHT measurement model.
    - For a=0, the exact Schwarzschild value is 2√27.
    - Definition depends on how one maps a non-circular Kerr shadow to a single "diameter".
    """

    return _kerr_shadow_diameter_coeff_effective(float(a_star), float(inc_deg), n_r=int(n_r), method="avg_wh")


def _kerr_shadow_coeff_range(
    *,
    a_star_min: float = 0.0,
    a_star_max: float = 0.999,
    inc_deg_min: float = 5.0,
    inc_deg_max: float = 90.0,
    a_samples: int = 50,
    inc_step_deg: float = 5.0,
    method: str = "avg_wh",
) -> Dict[str, Any]:
    """Reference range of GR (Kerr) shadow diameter coefficient (effective).

    This is intentionally a *reference systematic* for GR (Kerr), not an EHT emission model fit.

    Notes:
    - The range can be optionally constrained by (a*, inc) bounds (e.g., when a primary-source
      constraint exists for a given object). When not provided, we use a broad default range.
    - We avoid inc=0° where α=-ξ/sin(inc) becomes singular in the standard parameterization.
    """

    import numpy as np

    a0 = float(a_star_min)
    a1 = float(a_star_max)
    # 条件分岐: `not (math.isfinite(a0) and math.isfinite(a1))` を満たす経路を評価する。
    if not (math.isfinite(a0) and math.isfinite(a1)):
        a0, a1 = 0.0, 0.999

    # 条件分岐: `a0 > a1` を満たす経路を評価する。

    if a0 > a1:
        a0, a1 = a1, a0

    a0 = max(0.0, min(0.999, a0))
    a1 = max(0.0, min(0.999, a1))
    # 条件分岐: `a1 < a0` を満たす経路を評価する。
    if a1 < a0:
        a0, a1 = a1, a0

    i0 = float(inc_deg_min)
    i1 = float(inc_deg_max)
    # 条件分岐: `not (math.isfinite(i0) and math.isfinite(i1))` を満たす経路を評価する。
    if not (math.isfinite(i0) and math.isfinite(i1)):
        i0, i1 = 5.0, 90.0

    # 条件分岐: `i0 > i1` を満たす経路を評価する。

    if i0 > i1:
        i0, i1 = i1, i0

    i0 = max(0.0, min(90.0, i0))
    i1 = max(0.0, min(90.0, i1))
    # avoid exactly 0° where α=-ξ/sin(inc) is singular; include near-pole via 5°.
    i0_eff = max(5.0, i0)
    i1_eff = max(i0_eff, i1)

    coeff_schw = 2.0 * math.sqrt(27.0)
    method_label = {
        "avg_wh": "avg(width,height)",
        "geom_mean_wh": "sqrt(width*height)",
        "area_eq": "area-equivalent circle",
        "perimeter_eq": "perimeter-equivalent circle",
    }.get(str(method), str(method))
    coeffs: List[Tuple[float, float, float]] = []  # (coeff, a, inc)

    inc_list = np.arange(float(i0_eff), float(i1_eff) + 1e-9, float(inc_step_deg))
    # 条件分岐: `inc_list.size <= 0` を満たす経路を評価する。
    if inc_list.size <= 0:
        inc_list = np.array([float(i0_eff)], dtype=float)

    a_n = max(2, int(a_samples))
    a_list = np.linspace(float(a0), float(a1), a_n)
    for a in a_list:
        # 条件分岐: `abs(float(a)) < 1e-12` を満たす経路を評価する。
        if abs(float(a)) < 1e-12:
            # Schwarzschild: independent of inclination
            coeffs.append((coeff_schw, 0.0, float(inc_list[0])))
            continue

        for inc in inc_list:
            coeff = _kerr_shadow_diameter_coeff_effective(float(a), float(inc), method=str(method))
            # 条件分岐: `math.isfinite(coeff) and coeff > 0` を満たす経路を評価する。
            if math.isfinite(coeff) and coeff > 0:
                coeffs.append((float(coeff), float(a), float(inc)))

    # 条件分岐: `not coeffs` を満たす経路を評価する。

    if not coeffs:
        return {
            "method": "avg(width,height)",
            "coeff_schwarzschild": coeff_schw,
            "coeff_min": None,
            "coeff_max": None,
        }

    coeff_min, a_min, inc_min = min(coeffs, key=lambda x: x[0])
    coeff_max, a_max, inc_max = max(coeffs, key=lambda x: x[0])
    return {
        "method": method_label,
        "range": {"a_star_min": a0, "a_star_max": a1, "inc_deg_min": i0_eff, "inc_deg_max": i1_eff},
        "coeff_schwarzschild": coeff_schw,
        "coeff_min": coeff_min,
        "coeff_min_at": {"a_star": a_min, "inc_deg": inc_min},
        "coeff_max": coeff_max,
        "coeff_max_at": {"a_star": a_max, "inc_deg": inc_max},
        "grid": {"a_samples": int(a_samples), "inc_step_deg": float(inc_step_deg)},
    }


def _kerr_shadow_coeff_range_definition_envelope(
    *,
    methods: List[str],
    a_star_min: float = 0.0,
    a_star_max: float = 0.999,
    inc_deg_min: float = 5.0,
    inc_deg_max: float = 90.0,
    a_samples: int = 50,
    inc_step_deg: float = 5.0,
    n_r: int = 2500,
) -> Dict[str, Any]:
    """Envelope of Kerr shadow coefficient range across multiple effective-diameter definitions."""

    import numpy as np

    a0 = float(a_star_min)
    a1 = float(a_star_max)
    # 条件分岐: `not (math.isfinite(a0) and math.isfinite(a1))` を満たす経路を評価する。
    if not (math.isfinite(a0) and math.isfinite(a1)):
        a0, a1 = 0.0, 0.999

    # 条件分岐: `a0 > a1` を満たす経路を評価する。

    if a0 > a1:
        a0, a1 = a1, a0

    a0 = max(0.0, min(0.999, a0))
    a1 = max(0.0, min(0.999, a1))

    i0 = float(inc_deg_min)
    i1 = float(inc_deg_max)
    # 条件分岐: `not (math.isfinite(i0) and math.isfinite(i1))` を満たす経路を評価する。
    if not (math.isfinite(i0) and math.isfinite(i1)):
        i0, i1 = 5.0, 90.0

    # 条件分岐: `i0 > i1` を満たす経路を評価する。

    if i0 > i1:
        i0, i1 = i1, i0

    i0 = max(0.0, min(90.0, i0))
    i1 = max(0.0, min(90.0, i1))
    i0_eff = max(5.0, i0)
    i1_eff = max(i0_eff, i1)

    coeff_schw = 2.0 * math.sqrt(27.0)

    inc_list = np.arange(float(i0_eff), float(i1_eff) + 1e-9, float(inc_step_deg))
    # 条件分岐: `inc_list.size <= 0` を満たす経路を評価する。
    if inc_list.size <= 0:
        inc_list = np.array([float(i0_eff)], dtype=float)

    a_n = max(2, int(a_samples))
    a_list = np.linspace(float(a0), float(a1), a_n)

    labels = {
        "avg_wh": "avg(width,height)",
        "geom_mean_wh": "sqrt(width*height)",
        "area_eq": "area-equivalent circle",
        "perimeter_eq": "perimeter-equivalent circle",
    }
    methods_used = [str(m) for m in methods if str(m).strip()]
    # 条件分岐: `not methods_used` を満たす経路を評価する。
    if not methods_used:
        methods_used = ["avg_wh"]

    env_min: Optional[Tuple[float, float, float, str]] = None  # (coeff, a, inc, method)
    env_max: Optional[Tuple[float, float, float, str]] = None
    spread_rel_max = float("nan")

    for a in a_list:
        for inc in inc_list:
            stats = _kerr_shadow_boundary_stats(float(a), float(inc), n_r=int(n_r))
            coeffs_here: List[float] = []
            for m in methods_used:
                coeff = _kerr_shadow_diameter_coeff_from_stats(stats, method=str(m))
                # 条件分岐: `math.isfinite(coeff) and coeff > 0` を満たす経路を評価する。
                if math.isfinite(coeff) and coeff > 0:
                    coeffs_here.append(float(coeff))
                    # 条件分岐: `env_min is None or coeff < env_min[0]` を満たす経路を評価する。
                    if env_min is None or coeff < env_min[0]:
                        env_min = (float(coeff), float(a), float(inc), str(m))

                    # 条件分岐: `env_max is None or coeff > env_max[0]` を満たす経路を評価する。

                    if env_max is None or coeff > env_max[0]:
                        env_max = (float(coeff), float(a), float(inc), str(m))

            # 条件分岐: `len(coeffs_here) >= 2` を満たす経路を評価する。

            if len(coeffs_here) >= 2:
                lo = min(coeffs_here)
                hi = max(coeffs_here)
                mid = 0.5 * (lo + hi)
                # 条件分岐: `math.isfinite(mid) and mid > 0` を満たす経路を評価する。
                if math.isfinite(mid) and mid > 0:
                    spread_rel = (hi - lo) / mid
                    # 条件分岐: `(not math.isfinite(spread_rel_max)) or spread_rel > spread_rel_max` を満たす経路を評価する。
                    if (not math.isfinite(spread_rel_max)) or spread_rel > spread_rel_max:
                        spread_rel_max = float(spread_rel)

    return {
        "methods": [{"id": m, "label": labels.get(m, m)} for m in methods_used],
        "range": {"a_star_min": a0, "a_star_max": a1, "inc_deg_min": i0_eff, "inc_deg_max": i1_eff},
        "coeff_schwarzschild": coeff_schw,
        "coeff_min": (env_min[0] if env_min is not None else None),
        "coeff_min_at": (
            {"a_star": env_min[1], "inc_deg": env_min[2], "method": labels.get(env_min[3], env_min[3])}
            if env_min is not None
            else None
        ),
        "coeff_max": (env_max[0] if env_max is not None else None),
        "coeff_max_at": (
            {"a_star": env_max[1], "inc_deg": env_max[2], "method": labels.get(env_max[3], env_max[3])}
            if env_max is not None
            else None
        ),
        "definition_spread_rel_max": spread_rel_max if math.isfinite(spread_rel_max) else None,
        "grid": {"a_samples": int(a_samples), "inc_step_deg": float(inc_step_deg), "n_r": int(n_r)},
    }


@dataclass(frozen=True)
class BH:
    key: str
    display_name: str
    mass_msun: float
    mass_msun_sigma: float
    distance_m: float
    distance_m_sigma: float
    ring_diameter_uas: float
    ring_diameter_uas_sigma: float
    scattering_kernel_fwhm_uas: Optional[float]
    scattering_kernel_fwhm_major_uas: Optional[float]
    scattering_kernel_fwhm_major_uas_sigma: Optional[float]
    scattering_kernel_fwhm_minor_uas: Optional[float]
    scattering_kernel_fwhm_minor_uas_sigma: Optional[float]
    scattering_kernel_pa_deg: Optional[float]
    scattering_kernel_pa_deg_sigma: Optional[float]
    refractive_wander_uas_min: Optional[float]
    refractive_wander_uas_max: Optional[float]
    refractive_distortion_uas_min: Optional[float]
    refractive_distortion_uas_max: Optional[float]
    refractive_asymmetry_uas_min: Optional[float]
    refractive_asymmetry_uas_max: Optional[float]
    ring_fractional_width_min: Optional[float]
    ring_fractional_width_max: Optional[float]
    ring_brightness_asymmetry_min: Optional[float]
    ring_brightness_asymmetry_max: Optional[float]
    shadow_diameter_uas: Optional[float]
    shadow_diameter_uas_sigma: Optional[float]
    delta_schwarzschild_vlti: Optional[float]
    delta_schwarzschild_vlti_sigma_plus: Optional[float]
    delta_schwarzschild_vlti_sigma_minus: Optional[float]
    delta_schwarzschild_keck: Optional[float]
    delta_schwarzschild_keck_sigma_plus: Optional[float]
    delta_schwarzschild_keck_sigma_minus: Optional[float]
    delta_kerr_min: Optional[float]
    delta_kerr_max: Optional[float]
    kerr_a_star_min: Optional[float]
    kerr_a_star_max: Optional[float]
    kerr_inc_deg_min: Optional[float]
    kerr_inc_deg_max: Optional[float]
    source_keys: Tuple[str, ...]


def _parse_bh(o: Dict[str, Any]) -> BH:
    key = str(o.get("key") or "")
    display_name = str(o.get("display_name") or key or "unknown")
    mass_msun = float(o["mass_msun"])
    mass_msun_sigma = float(o.get("mass_msun_sigma") or 0.0)

    # distance: either kpc or Mpc
    if "distance_kpc" in o:
        distance_m = float(o["distance_kpc"]) * 1e3 * 3.085677581491367e16  # kpc->m
        distance_m_sigma = float(o.get("distance_kpc_sigma") or 0.0) * 1e3 * 3.085677581491367e16
    # 条件分岐: 前段条件が不成立で、`"distance_mpc" in o` を追加評価する。
    elif "distance_mpc" in o:
        distance_m = float(o["distance_mpc"]) * 1e6 * 3.085677581491367e16  # Mpc->m
        distance_m_sigma = float(o.get("distance_mpc_sigma") or 0.0) * 1e6 * 3.085677581491367e16
    else:
        raise ValueError(f"missing distance for {key}")

    ring_diameter_uas = float(o["ring_diameter_uas"])
    ring_diameter_uas_sigma = float(o.get("ring_diameter_uas_sigma") or 0.0)
    scattering_kernel_fwhm_uas = float(o["scattering_kernel_fwhm_uas"]) if "scattering_kernel_fwhm_uas" in o else None
    scattering_kernel_fwhm_major_uas = (
        float(o["scattering_kernel_fwhm_major_uas"]) if "scattering_kernel_fwhm_major_uas" in o else None
    )
    scattering_kernel_fwhm_major_uas_sigma = (
        float(o["scattering_kernel_fwhm_major_uas_sigma"]) if "scattering_kernel_fwhm_major_uas_sigma" in o else None
    )
    scattering_kernel_fwhm_minor_uas = (
        float(o["scattering_kernel_fwhm_minor_uas"]) if "scattering_kernel_fwhm_minor_uas" in o else None
    )
    scattering_kernel_fwhm_minor_uas_sigma = (
        float(o["scattering_kernel_fwhm_minor_uas_sigma"]) if "scattering_kernel_fwhm_minor_uas_sigma" in o else None
    )
    scattering_kernel_pa_deg = float(o["scattering_kernel_pa_deg"]) if "scattering_kernel_pa_deg" in o else None
    scattering_kernel_pa_deg_sigma = (
        float(o["scattering_kernel_pa_deg_sigma"]) if "scattering_kernel_pa_deg_sigma" in o else None
    )
    refractive_wander_uas_min = float(o["refractive_wander_uas_min"]) if "refractive_wander_uas_min" in o else None
    refractive_wander_uas_max = float(o["refractive_wander_uas_max"]) if "refractive_wander_uas_max" in o else None
    refractive_distortion_uas_min = (
        float(o["refractive_distortion_uas_min"]) if "refractive_distortion_uas_min" in o else None
    )
    refractive_distortion_uas_max = (
        float(o["refractive_distortion_uas_max"]) if "refractive_distortion_uas_max" in o else None
    )
    refractive_asymmetry_uas_min = (
        float(o["refractive_asymmetry_uas_min"]) if "refractive_asymmetry_uas_min" in o else None
    )
    refractive_asymmetry_uas_max = (
        float(o["refractive_asymmetry_uas_max"]) if "refractive_asymmetry_uas_max" in o else None
    )
    ring_fractional_width_min = (
        float(o["ring_fractional_width_min"]) if "ring_fractional_width_min" in o else None
    )
    ring_fractional_width_max = (
        float(o["ring_fractional_width_max"]) if "ring_fractional_width_max" in o else None
    )
    ring_brightness_asymmetry_min = (
        float(o["ring_brightness_asymmetry_min"]) if "ring_brightness_asymmetry_min" in o else None
    )
    ring_brightness_asymmetry_max = (
        float(o["ring_brightness_asymmetry_max"]) if "ring_brightness_asymmetry_max" in o else None
    )
    shadow_diameter_uas = float(o["shadow_diameter_uas"]) if "shadow_diameter_uas" in o else None
    shadow_diameter_uas_sigma = (
        float(o.get("shadow_diameter_uas_sigma") or 0.0) if shadow_diameter_uas is not None else None
    )
    delta_schwarzschild_vlti = (
        float(o["delta_schwarzschild_vlti"]) if "delta_schwarzschild_vlti" in o else None
    )
    delta_schwarzschild_vlti_sigma_plus = (
        float(o["delta_schwarzschild_vlti_sigma_plus"]) if "delta_schwarzschild_vlti_sigma_plus" in o else None
    )
    delta_schwarzschild_vlti_sigma_minus = (
        float(o["delta_schwarzschild_vlti_sigma_minus"]) if "delta_schwarzschild_vlti_sigma_minus" in o else None
    )
    delta_schwarzschild_keck = float(o["delta_schwarzschild_keck"]) if "delta_schwarzschild_keck" in o else None
    delta_schwarzschild_keck_sigma_plus = (
        float(o["delta_schwarzschild_keck_sigma_plus"]) if "delta_schwarzschild_keck_sigma_plus" in o else None
    )
    delta_schwarzschild_keck_sigma_minus = (
        float(o["delta_schwarzschild_keck_sigma_minus"]) if "delta_schwarzschild_keck_sigma_minus" in o else None
    )
    delta_kerr_min = float(o["delta_kerr_min"]) if "delta_kerr_min" in o else None
    delta_kerr_max = float(o["delta_kerr_max"]) if "delta_kerr_max" in o else None
    kerr_a_star_min = float(o["kerr_a_star_min"]) if "kerr_a_star_min" in o else None
    kerr_a_star_max = float(o["kerr_a_star_max"]) if "kerr_a_star_max" in o else None
    kerr_inc_deg_min = float(o["kerr_inc_deg_min"]) if "kerr_inc_deg_min" in o else None
    kerr_inc_deg_max = float(o["kerr_inc_deg_max"]) if "kerr_inc_deg_max" in o else None
    source_keys = tuple(str(x) for x in (o.get("source_keys") or []) if str(x))
    return BH(
        key=key,
        display_name=display_name,
        mass_msun=mass_msun,
        mass_msun_sigma=mass_msun_sigma,
        distance_m=distance_m,
        distance_m_sigma=distance_m_sigma,
        ring_diameter_uas=ring_diameter_uas,
        ring_diameter_uas_sigma=ring_diameter_uas_sigma,
        scattering_kernel_fwhm_uas=scattering_kernel_fwhm_uas,
        scattering_kernel_fwhm_major_uas=scattering_kernel_fwhm_major_uas,
        scattering_kernel_fwhm_major_uas_sigma=scattering_kernel_fwhm_major_uas_sigma,
        scattering_kernel_fwhm_minor_uas=scattering_kernel_fwhm_minor_uas,
        scattering_kernel_fwhm_minor_uas_sigma=scattering_kernel_fwhm_minor_uas_sigma,
        scattering_kernel_pa_deg=scattering_kernel_pa_deg,
        scattering_kernel_pa_deg_sigma=scattering_kernel_pa_deg_sigma,
        refractive_wander_uas_min=refractive_wander_uas_min,
        refractive_wander_uas_max=refractive_wander_uas_max,
        refractive_distortion_uas_min=refractive_distortion_uas_min,
        refractive_distortion_uas_max=refractive_distortion_uas_max,
        refractive_asymmetry_uas_min=refractive_asymmetry_uas_min,
        refractive_asymmetry_uas_max=refractive_asymmetry_uas_max,
        ring_fractional_width_min=ring_fractional_width_min,
        ring_fractional_width_max=ring_fractional_width_max,
        ring_brightness_asymmetry_min=ring_brightness_asymmetry_min,
        ring_brightness_asymmetry_max=ring_brightness_asymmetry_max,
        shadow_diameter_uas=shadow_diameter_uas,
        shadow_diameter_uas_sigma=shadow_diameter_uas_sigma,
        delta_schwarzschild_vlti=delta_schwarzschild_vlti,
        delta_schwarzschild_vlti_sigma_plus=delta_schwarzschild_vlti_sigma_plus,
        delta_schwarzschild_vlti_sigma_minus=delta_schwarzschild_vlti_sigma_minus,
        delta_schwarzschild_keck=delta_schwarzschild_keck,
        delta_schwarzschild_keck_sigma_plus=delta_schwarzschild_keck_sigma_plus,
        delta_schwarzschild_keck_sigma_minus=delta_schwarzschild_keck_sigma_minus,
        delta_kerr_min=delta_kerr_min,
        delta_kerr_max=delta_kerr_max,
        kerr_a_star_min=kerr_a_star_min,
        kerr_a_star_max=kerr_a_star_max,
        kerr_inc_deg_min=kerr_inc_deg_min,
        kerr_inc_deg_max=kerr_inc_deg_max,
        source_keys=source_keys,
    )


def _prop_sigma_ratio(m: float, dm: float, d: float, dd: float) -> float:
    eps = 0.0
    # 条件分岐: `m > 0 and dm > 0` を満たす経路を評価する。
    if m > 0 and dm > 0:
        eps += (dm / m) ** 2

    # 条件分岐: `d > 0 and dd > 0` を満たす経路を評価する。

    if d > 0 and dd > 0:
        eps += (dd / d) ** 2

    return math.sqrt(eps)


def _sigma_needed_for_discrimination(
    diff: float,
    *,
    sigma_pred_a: float,
    sigma_pred_b: float,
    n_sigma: float = 3.0,
) -> float:
    """Required 1σ observational uncertainty to separate two model predictions by n_sigma.

    Model A: X_a ~ N(mu_a, sigma_pred_a^2 + sigma_obs^2)
    Model B: X_b ~ N(mu_b, sigma_pred_b^2 + sigma_obs^2)

    A simple overlap criterion is:
      |mu_a - mu_b| > n_sigma * sqrt(sigma_pred_a^2 + sigma_pred_b^2 + 2*sigma_obs^2)

    Solve for sigma_obs (>=0). Returns NaN if impossible (e.g., the current parameter
    uncertainties already make the predictions overlap too much).
    """

    if not (
        math.isfinite(diff)
        and math.isfinite(sigma_pred_a)
        and math.isfinite(sigma_pred_b)
        and float(n_sigma) > 0
    ):
        return float("nan")

    a2 = (abs(float(diff)) / float(n_sigma)) ** 2 - float(sigma_pred_a) ** 2 - float(sigma_pred_b) ** 2
    # 条件分岐: `a2 <= 0` を満たす経路を評価する。
    if a2 <= 0:
        return float("nan")

    return math.sqrt(a2 / 2.0)


def main() -> int:
    root = _repo_root()
    in_path = root / "data" / "eht" / "eht_black_holes.json"
    out_dir = root / "output" / "private" / "eht"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 条件分岐: `not in_path.exists()` を満たす経路を評価する。
    if not in_path.exists():
        print(f"[err] missing input: {in_path}")
        return 2

    ref = _read_json(in_path)
    beta = float(((ref.get("pmodel") or {}).get("beta")) or float("nan"))
    beta_source = "data/eht/eht_black_holes.json:pmodel.beta"
    # 条件分岐: `not (math.isfinite(beta) and beta > 0)` を満たす経路を評価する。
    if not (math.isfinite(beta) and beta > 0):
        frozen_path = root / "output" / "private" / "theory" / "frozen_parameters.json"
        try:
            frozen = _read_json(frozen_path) if frozen_path.exists() else {}
            beta = float(frozen.get("beta"))
            beta_source = "output/private/theory/frozen_parameters.json:beta"
        except Exception:
            beta = float("nan")

    # 条件分岐: `not (math.isfinite(beta) and beta > 0)` を満たす経路を評価する。

    if not (math.isfinite(beta) and beta > 0):
        beta = 1.0
        beta_source = "default(beta=1.0)"

    objects = ref.get("objects") or []
    bhs = [_parse_bh(o) for o in objects if isinstance(o, dict)]
    # 条件分岐: `not bhs` を満たす経路を評価する。
    if not bhs:
        print("[err] no objects in eht_black_holes.json")
        return 2

    # constants

    G = 6.67430e-11
    c = 299792458.0
    M_sun = 1.98847e30
    rad_to_uas = (180.0 / math.pi) * 3600.0 * 1e6

    coeff_pmodel = 4.0 * math.e * beta  # θ_shadow = coeff * GM/(c^2 D)
    coeff_gr = 2.0 * math.sqrt(27.0)  # Schwarzschild shadow diameter
    coeff_ratio_p_over_gr_baseline = (coeff_pmodel / coeff_gr) if coeff_gr != 0 else float("nan")
    (
        coeff_ratio_p_over_gr,
        coeff_diff_pct_phase4,
        coeff_ratio_source,
        coeff_ratio_source_relpath,
    ) = _resolve_phase4_coeff_ratio(root, coeff_ratio_p_over_gr_baseline)
    coeff_pmodel_phase4 = (coeff_gr * coeff_ratio_p_over_gr) if math.isfinite(coeff_ratio_p_over_gr) else coeff_pmodel
    delta_coeff_p_minus_gr = (coeff_ratio_p_over_gr - 1.0) if math.isfinite(coeff_ratio_p_over_gr) else float("nan")
    delta_sigma_required_3sigma = (
        (abs(float(delta_coeff_p_minus_gr)) / 3.0) if math.isfinite(delta_coeff_p_minus_gr) else float("nan")
    )
    kerr_range_full = _kerr_shadow_coeff_range(method="avg_wh")
    kerr_definition_methods = ["avg_wh", "geom_mean_wh", "area_eq", "perimeter_eq"]
    kerr_range_full_def_env = _kerr_shadow_coeff_range_definition_envelope(methods=kerr_definition_methods)

    rows: List[Dict[str, Any]] = []
    for bh in bhs:
        M = bh.mass_msun * M_sun
        dM = bh.mass_msun_sigma * M_sun
        D = bh.distance_m
        dD = bh.distance_m_sigma

        theta_unit_uas = (G * M / (c**2 * D)) * rad_to_uas
        ratio_sigma = _prop_sigma_ratio(M, dM, D, dD)
        theta_unit_uas_sigma = abs(theta_unit_uas) * ratio_sigma

        theta_p_uas = coeff_pmodel * theta_unit_uas
        theta_p_uas_sigma = abs(theta_p_uas) * ratio_sigma

        theta_gr_uas = coeff_gr * theta_unit_uas
        theta_gr_uas_sigma = abs(theta_gr_uas) * ratio_sigma

        # Reference GR systematic range (Kerr): coefficient depends on spin/inclination and on how we define an "effective diameter".
        # This is a reference range (not a fitted EHT emission model). When object-specific bounds exist, we optionally compute a
        # "constrained" Kerr range for that object and use it for the κ budget.

        # Full-range GR Kerr systematic: (spin/inc) + effective-diameter definition envelope.
        kerr_coeff_min_full = (
            float(kerr_range_full_def_env.get("coeff_min") or float("nan"))
            if isinstance(kerr_range_full_def_env, dict)
            else float("nan")
        )
        kerr_coeff_max_full = (
            float(kerr_range_full_def_env.get("coeff_max") or float("nan"))
            if isinstance(kerr_range_full_def_env, dict)
            else float("nan")
        )
        theta_gr_kerr_min_full_uas = (kerr_coeff_min_full * theta_unit_uas) if math.isfinite(kerr_coeff_min_full) else float("nan")
        theta_gr_kerr_max_full_uas = (kerr_coeff_max_full * theta_unit_uas) if math.isfinite(kerr_coeff_max_full) else float("nan")

        # Spin/inc only (legacy reference): avg(width,height) definition.
        kerr_coeff_min_full_spin_only = (
            float(kerr_range_full.get("coeff_min") or float("nan")) if isinstance(kerr_range_full, dict) else float("nan")
        )
        kerr_coeff_max_full_spin_only = (
            float(kerr_range_full.get("coeff_max") or float("nan")) if isinstance(kerr_range_full, dict) else float("nan")
        )
        theta_gr_kerr_min_full_spin_only_uas = (
            (kerr_coeff_min_full_spin_only * theta_unit_uas) if math.isfinite(kerr_coeff_min_full_spin_only) else float("nan")
        )
        theta_gr_kerr_max_full_spin_only_uas = (
            (kerr_coeff_max_full_spin_only * theta_unit_uas) if math.isfinite(kerr_coeff_max_full_spin_only) else float("nan")
        )

        has_kerr_constraints = any(
            x is not None for x in (bh.kerr_a_star_min, bh.kerr_a_star_max, bh.kerr_inc_deg_min, bh.kerr_inc_deg_max)
        )
        kerr_a_star_min_used = float(bh.kerr_a_star_min) if bh.kerr_a_star_min is not None else 0.0
        kerr_a_star_max_used = float(bh.kerr_a_star_max) if bh.kerr_a_star_max is not None else 0.999
        kerr_inc_deg_min_used = float(bh.kerr_inc_deg_min) if bh.kerr_inc_deg_min is not None else 5.0
        kerr_inc_deg_max_used = float(bh.kerr_inc_deg_max) if bh.kerr_inc_deg_max is not None else 90.0

        kerr_range_used_spin_only = (
            _kerr_shadow_coeff_range(
                a_star_min=kerr_a_star_min_used,
                a_star_max=kerr_a_star_max_used,
                inc_deg_min=kerr_inc_deg_min_used,
                inc_deg_max=kerr_inc_deg_max_used,
                method="avg_wh",
            )
            if has_kerr_constraints
            else kerr_range_full
        )
        kerr_range_used = (
            _kerr_shadow_coeff_range_definition_envelope(
                methods=kerr_definition_methods,
                a_star_min=kerr_a_star_min_used,
                a_star_max=kerr_a_star_max_used,
                inc_deg_min=kerr_inc_deg_min_used,
                inc_deg_max=kerr_inc_deg_max_used,
            )
            if has_kerr_constraints
            else kerr_range_full_def_env
        )
        kerr_range_mode = "constrained" if has_kerr_constraints else "full"

        kerr_coeff_min = float(kerr_range_used.get("coeff_min") or float("nan")) if isinstance(kerr_range_used, dict) else float("nan")
        kerr_coeff_max = float(kerr_range_used.get("coeff_max") or float("nan")) if isinstance(kerr_range_used, dict) else float("nan")
        theta_gr_kerr_min_uas = (kerr_coeff_min * theta_unit_uas) if math.isfinite(kerr_coeff_min) else float("nan")
        theta_gr_kerr_max_uas = (kerr_coeff_max * theta_unit_uas) if math.isfinite(kerr_coeff_max) else float("nan")

        kerr_coeff_min_spin_only = (
            float(kerr_range_used_spin_only.get("coeff_min") or float("nan"))
            if isinstance(kerr_range_used_spin_only, dict)
            else float("nan")
        )
        kerr_coeff_max_spin_only = (
            float(kerr_range_used_spin_only.get("coeff_max") or float("nan"))
            if isinstance(kerr_range_used_spin_only, dict)
            else float("nan")
        )
        theta_gr_kerr_min_spin_only_uas = (
            (kerr_coeff_min_spin_only * theta_unit_uas) if math.isfinite(kerr_coeff_min_spin_only) else float("nan")
        )
        theta_gr_kerr_max_spin_only_uas = (
            (kerr_coeff_max_spin_only * theta_unit_uas) if math.isfinite(kerr_coeff_max_spin_only) else float("nan")
        )

        obs = bh.ring_diameter_uas
        obs_sigma = bh.ring_diameter_uas_sigma
        shadow_obs = bh.shadow_diameter_uas
        shadow_obs_sigma = bh.shadow_diameter_uas_sigma

        # beta that matches observation under the P-model shadow formula (linear in beta)
        theta_p_beta1_uas = (4.0 * math.e * 1.0) * theta_unit_uas
        theta_p_beta1_uas_sigma = abs(theta_p_beta1_uas) * ratio_sigma
        beta_fit = obs / theta_p_beta1_uas if theta_p_beta1_uas != 0 else float("nan")
        beta_fit_sigma = abs(beta_fit) * math.sqrt(
            ((obs_sigma / obs) ** 2 if obs > 0 and obs_sigma > 0 else 0.0)
            + ((theta_p_beta1_uas_sigma / theta_p_beta1_uas) ** 2 if theta_p_beta1_uas != 0 else 0.0)
        )

        def _z(res: float, s1: float, s2: float) -> float:
            denom = math.sqrt((s1 or 0.0) ** 2 + (s2 or 0.0) ** 2)
            return (res / denom) if denom > 0 else float("nan")

        z_p = _z(obs - theta_p_uas, obs_sigma, theta_p_uas_sigma)
        z_gr = _z(obs - theta_gr_uas, obs_sigma, theta_gr_uas_sigma)

        z_p_shadowobs = (
            _z(float(shadow_obs) - theta_p_uas, float(shadow_obs_sigma or 0.0), theta_p_uas_sigma)
            if shadow_obs is not None and shadow_obs_sigma is not None
            else float("nan")
        )
        z_gr_shadowobs = (
            _z(float(shadow_obs) - theta_gr_uas, float(shadow_obs_sigma or 0.0), theta_gr_uas_sigma)
            if shadow_obs is not None and shadow_obs_sigma is not None
            else float("nan")
        )

        theta_p_phase4_uas = coeff_pmodel_phase4 * theta_unit_uas
        theta_p_phase4_uas_sigma = abs(theta_p_phase4_uas) * ratio_sigma

        diff_uas = theta_p_phase4_uas - theta_gr_uas
        diff_uas_sigma = abs(diff_uas) * ratio_sigma
        diff_pct = (100.0 * diff_uas / theta_gr_uas) if theta_gr_uas != 0 else float("nan")
        sigma_obs_needed_3sigma_uas = _sigma_needed_for_discrimination(
            diff_uas,
            sigma_pred_a=float(theta_p_phase4_uas_sigma),
            sigma_pred_b=float(theta_gr_uas_sigma),
            n_sigma=3.0,
        )
        # If mass/distance uncertainty dominates, even sigma_obs→0 cannot separate the two models.
        theta_unit_rel_sigma_required_3sigma = (
            abs(coeff_pmodel_phase4 - coeff_gr) / (3.0 * math.sqrt(coeff_pmodel_phase4**2 + coeff_gr**2))
            if (
                math.isfinite(coeff_pmodel_phase4)
                and math.isfinite(coeff_gr)
                and (coeff_pmodel_phase4 > 0)
                and (coeff_gr > 0)
            )
            else float("nan")
        )
        ring_sigma_required_3sigma_uas_if_kappa1 = (
            float(sigma_obs_needed_3sigma_uas) if math.isfinite(sigma_obs_needed_3sigma_uas) else float("nan")
        )
        ring_sigma_improvement_factor_to_3sigma_if_kappa1 = (
            (obs_sigma / ring_sigma_required_3sigma_uas_if_kappa1)
            if (math.isfinite(ring_sigma_required_3sigma_uas_if_kappa1) and ring_sigma_required_3sigma_uas_if_kappa1 > 0)
            else float("nan")
        )
        theta_unit_rel_sigma_improvement_factor_to_3sigma = (
            (ratio_sigma / theta_unit_rel_sigma_required_3sigma)
            if (math.isfinite(theta_unit_rel_sigma_required_3sigma) and theta_unit_rel_sigma_required_3sigma > 0)
            else float("nan")
        )

        # Systematic: ring diameter is not strictly identical to shadow diameter (emission/scattering/spin).
        # Introduce κ such that: ring_diameter ≈ κ * shadow_diameter. Here we *infer* κ from public values.
        def _sigma_ratio(a0: float, da0: float, b0: float, db0: float) -> float:
            eps2 = 0.0
            # 条件分岐: `a0 > 0 and da0 > 0` を満たす経路を評価する。
            if a0 > 0 and da0 > 0:
                eps2 += (da0 / a0) ** 2

            # 条件分岐: `b0 > 0 and db0 > 0` を満たす経路を評価する。

            if b0 > 0 and db0 > 0:
                eps2 += (db0 / b0) ** 2

            return math.sqrt(eps2)

        kappa_p = (obs / theta_p_uas) if theta_p_uas != 0 else float("nan")
        kappa_p_sigma = abs(kappa_p) * _sigma_ratio(obs, obs_sigma, theta_p_uas, theta_p_uas_sigma)
        kappa_gr = (obs / theta_gr_uas) if theta_gr_uas != 0 else float("nan")
        kappa_gr_sigma = abs(kappa_gr) * _sigma_ratio(obs, obs_sigma, theta_gr_uas, theta_gr_uas_sigma)

        kappa_obs = (obs / float(shadow_obs)) if shadow_obs is not None and float(shadow_obs) != 0 else float("nan")
        kappa_obs_sigma = (
            abs(kappa_obs) * _sigma_ratio(obs, obs_sigma, float(shadow_obs), float(shadow_obs_sigma))
            if shadow_obs is not None and shadow_obs_sigma is not None
            else float("nan")
        )

        # κ range induced by Kerr coefficient range only (mass/distance central values), as a quick systematic scale indicator.
        # κ = ring / shadow, and shadow ∝ coeff, so κ ∝ 1/coeff.
        kappa_gr_kerr_low = (
            (obs / theta_gr_kerr_max_uas) if math.isfinite(theta_gr_kerr_max_uas) and theta_gr_kerr_max_uas != 0 else float("nan")
        )
        kappa_gr_kerr_high = (
            (obs / theta_gr_kerr_min_uas) if math.isfinite(theta_gr_kerr_min_uas) and theta_gr_kerr_min_uas != 0 else float("nan")
        )
        kappa_gr_kerr_low_full = (
            (obs / theta_gr_kerr_max_full_uas)
            if math.isfinite(theta_gr_kerr_max_full_uas) and theta_gr_kerr_max_full_uas != 0
            else float("nan")
        )
        kappa_gr_kerr_high_full = (
            (obs / theta_gr_kerr_min_full_uas)
            if math.isfinite(theta_gr_kerr_min_full_uas) and theta_gr_kerr_min_full_uas != 0
            else float("nan")
        )

        kappa_gr_kerr_low_spin_only = (
            (obs / theta_gr_kerr_max_spin_only_uas)
            if math.isfinite(theta_gr_kerr_max_spin_only_uas) and theta_gr_kerr_max_spin_only_uas != 0
            else float("nan")
        )
        kappa_gr_kerr_high_spin_only = (
            (obs / theta_gr_kerr_min_spin_only_uas)
            if math.isfinite(theta_gr_kerr_min_spin_only_uas) and theta_gr_kerr_min_spin_only_uas != 0
            else float("nan")
        )
        kappa_gr_kerr_low_full_spin_only = (
            (obs / theta_gr_kerr_max_full_spin_only_uas)
            if math.isfinite(theta_gr_kerr_max_full_spin_only_uas) and theta_gr_kerr_max_full_spin_only_uas != 0
            else float("nan")
        )
        kappa_gr_kerr_high_full_spin_only = (
            (obs / theta_gr_kerr_min_full_spin_only_uas)
            if math.isfinite(theta_gr_kerr_min_full_spin_only_uas) and theta_gr_kerr_min_full_spin_only_uas != 0
            else float("nan")
        )

        # Requirement: κ precision needed to reach 3σ separation (best-case: ring σ→0; current-case: ring σ fixed).
        kappa_sigma_required_3sigma_if_ring_sigma_zero = (
            (float(sigma_obs_needed_3sigma_uas) / obs)
            if (math.isfinite(float(sigma_obs_needed_3sigma_uas)) and float(sigma_obs_needed_3sigma_uas) > 0 and obs > 0)
            else float("nan")
        )
        kappa_sigma_required_3sigma_if_ring_sigma_current = float("nan")
        if (
            math.isfinite(float(sigma_obs_needed_3sigma_uas))
            and float(sigma_obs_needed_3sigma_uas) > 0
            and obs > 0
            and math.isfinite(obs_sigma)
            and obs_sigma >= 0
        ):
            rem2 = float(sigma_obs_needed_3sigma_uas) ** 2 - float(obs_sigma) ** 2
            # 条件分岐: `rem2 > 0` を満たす経路を評価する。
            if rem2 > 0:
                kappa_sigma_required_3sigma_if_ring_sigma_current = math.sqrt(rem2) / obs

        # Reference κ systematic scale from Kerr coefficient range (uniform-in-range approximation).

        kappa_sigma_assumed_kerr = (
            (kappa_gr_kerr_high - kappa_gr_kerr_low) / math.sqrt(12.0)
            if (math.isfinite(kappa_gr_kerr_low) and math.isfinite(kappa_gr_kerr_high) and kappa_gr_kerr_high > kappa_gr_kerr_low)
            else float("nan")
        )
        kappa_sigma_assumed_kerr_full = (
            (kappa_gr_kerr_high_full - kappa_gr_kerr_low_full) / math.sqrt(12.0)
            if (
                math.isfinite(kappa_gr_kerr_low_full)
                and math.isfinite(kappa_gr_kerr_high_full)
                and kappa_gr_kerr_high_full > kappa_gr_kerr_low_full
            )
            else float("nan")
        )
        kappa_sigma_assumed_kerr_spin_only = (
            (kappa_gr_kerr_high_spin_only - kappa_gr_kerr_low_spin_only) / math.sqrt(12.0)
            if (
                math.isfinite(kappa_gr_kerr_low_spin_only)
                and math.isfinite(kappa_gr_kerr_high_spin_only)
                and kappa_gr_kerr_high_spin_only > kappa_gr_kerr_low_spin_only
            )
            else float("nan")
        )
        kappa_sigma_assumed_kerr_full_spin_only = (
            (kappa_gr_kerr_high_full_spin_only - kappa_gr_kerr_low_full_spin_only) / math.sqrt(12.0)
            if (
                math.isfinite(kappa_gr_kerr_low_full_spin_only)
                and math.isfinite(kappa_gr_kerr_high_full_spin_only)
                and kappa_gr_kerr_high_full_spin_only > kappa_gr_kerr_low_full_spin_only
            )
            else float("nan")
        )

        # Reference: δ (Schwarzschild shadow deviation) constraints are GR-derived and model-dependent.
        delta_vlti = float(bh.delta_schwarzschild_vlti) if bh.delta_schwarzschild_vlti is not None else float("nan")
        delta_vlti_p = (
            float(bh.delta_schwarzschild_vlti_sigma_plus)
            if bh.delta_schwarzschild_vlti_sigma_plus is not None
            else float("nan")
        )
        delta_vlti_m = (
            float(bh.delta_schwarzschild_vlti_sigma_minus)
            if bh.delta_schwarzschild_vlti_sigma_minus is not None
            else float("nan")
        )
        delta_vlti_sigma_sym = (
            max(abs(delta_vlti_p), abs(delta_vlti_m))
            if (math.isfinite(delta_vlti_p) and math.isfinite(delta_vlti_m))
            else float("nan")
        )
        delta_vlti_improvement_factor_to_3sigma = (
            (delta_vlti_sigma_sym / delta_sigma_required_3sigma)
            if (math.isfinite(delta_vlti_sigma_sym) and math.isfinite(delta_sigma_required_3sigma) and delta_sigma_required_3sigma > 0)
            else float("nan")
        )

        delta_keck = float(bh.delta_schwarzschild_keck) if bh.delta_schwarzschild_keck is not None else float("nan")
        delta_keck_p = (
            float(bh.delta_schwarzschild_keck_sigma_plus)
            if bh.delta_schwarzschild_keck_sigma_plus is not None
            else float("nan")
        )
        delta_keck_m = (
            float(bh.delta_schwarzschild_keck_sigma_minus)
            if bh.delta_schwarzschild_keck_sigma_minus is not None
            else float("nan")
        )
        delta_keck_sigma_sym = (
            max(abs(delta_keck_p), abs(delta_keck_m))
            if (math.isfinite(delta_keck_p) and math.isfinite(delta_keck_m))
            else float("nan")
        )
        delta_keck_improvement_factor_to_3sigma = (
            (delta_keck_sigma_sym / delta_sigma_required_3sigma)
            if (math.isfinite(delta_keck_sigma_sym) and math.isfinite(delta_sigma_required_3sigma) and delta_sigma_required_3sigma > 0)
            else float("nan")
        )

        delta_kerr_min = float(bh.delta_kerr_min) if bh.delta_kerr_min is not None else float("nan")
        delta_kerr_max = float(bh.delta_kerr_max) if bh.delta_kerr_max is not None else float("nan")
        delta_kerr_sigma_uniform = (
            (delta_kerr_max - delta_kerr_min) / math.sqrt(12.0)
            if (math.isfinite(delta_kerr_min) and math.isfinite(delta_kerr_max) and delta_kerr_max > delta_kerr_min)
            else float("nan")
        )
        delta_kerr_improvement_factor_to_3sigma = (
            (delta_kerr_sigma_uniform / delta_sigma_required_3sigma)
            if (math.isfinite(delta_kerr_sigma_uniform) and math.isfinite(delta_sigma_required_3sigma) and delta_sigma_required_3sigma > 0)
            else float("nan")
        )

        rows.append(
            {
                "key": bh.key,
                "name": bh.display_name,
                "ring_diameter_obs_uas": obs,
                "ring_diameter_obs_uas_sigma": obs_sigma,
                "scattering_kernel_fwhm_uas": (
                    float(bh.scattering_kernel_fwhm_uas) if bh.scattering_kernel_fwhm_uas is not None else float("nan")
                ),
                "scattering_kernel_fwhm_major_uas": (
                    float(bh.scattering_kernel_fwhm_major_uas)
                    if bh.scattering_kernel_fwhm_major_uas is not None
                    else float("nan")
                ),
                "scattering_kernel_fwhm_major_uas_sigma": (
                    float(bh.scattering_kernel_fwhm_major_uas_sigma)
                    if bh.scattering_kernel_fwhm_major_uas_sigma is not None
                    else float("nan")
                ),
                "scattering_kernel_fwhm_minor_uas": (
                    float(bh.scattering_kernel_fwhm_minor_uas)
                    if bh.scattering_kernel_fwhm_minor_uas is not None
                    else float("nan")
                ),
                "scattering_kernel_fwhm_minor_uas_sigma": (
                    float(bh.scattering_kernel_fwhm_minor_uas_sigma)
                    if bh.scattering_kernel_fwhm_minor_uas_sigma is not None
                    else float("nan")
                ),
                "scattering_kernel_pa_deg": (
                    float(bh.scattering_kernel_pa_deg) if bh.scattering_kernel_pa_deg is not None else float("nan")
                ),
                "scattering_kernel_pa_deg_sigma": (
                    float(bh.scattering_kernel_pa_deg_sigma)
                    if bh.scattering_kernel_pa_deg_sigma is not None
                    else float("nan")
                ),
                "refractive_wander_uas_min": (
                    float(bh.refractive_wander_uas_min) if bh.refractive_wander_uas_min is not None else float("nan")
                ),
                "refractive_wander_uas_max": (
                    float(bh.refractive_wander_uas_max) if bh.refractive_wander_uas_max is not None else float("nan")
                ),
                "refractive_distortion_uas_min": (
                    float(bh.refractive_distortion_uas_min)
                    if bh.refractive_distortion_uas_min is not None
                    else float("nan")
                ),
                "refractive_distortion_uas_max": (
                    float(bh.refractive_distortion_uas_max)
                    if bh.refractive_distortion_uas_max is not None
                    else float("nan")
                ),
                "refractive_asymmetry_uas_min": (
                    float(bh.refractive_asymmetry_uas_min)
                    if bh.refractive_asymmetry_uas_min is not None
                    else float("nan")
                ),
                "refractive_asymmetry_uas_max": (
                    float(bh.refractive_asymmetry_uas_max)
                    if bh.refractive_asymmetry_uas_max is not None
                    else float("nan")
                ),
                "ring_fractional_width_min": (
                    float(bh.ring_fractional_width_min) if bh.ring_fractional_width_min is not None else float("nan")
                ),
                "ring_fractional_width_max": (
                    float(bh.ring_fractional_width_max) if bh.ring_fractional_width_max is not None else float("nan")
                ),
                "ring_brightness_asymmetry_min": (
                    float(bh.ring_brightness_asymmetry_min)
                    if bh.ring_brightness_asymmetry_min is not None
                    else float("nan")
                ),
                "ring_brightness_asymmetry_max": (
                    float(bh.ring_brightness_asymmetry_max)
                    if bh.ring_brightness_asymmetry_max is not None
                    else float("nan")
                ),
                "shadow_diameter_obs_uas": (float(shadow_obs) if shadow_obs is not None else float("nan")),
                "shadow_diameter_obs_uas_sigma": (float(shadow_obs_sigma) if shadow_obs_sigma is not None else float("nan")),
                "delta_sigma_required_3sigma": delta_sigma_required_3sigma,
                "delta_schwarzschild_vlti": delta_vlti,
                "delta_schwarzschild_vlti_sigma_plus": delta_vlti_p,
                "delta_schwarzschild_vlti_sigma_minus": delta_vlti_m,
                "delta_schwarzschild_vlti_sigma_sym": delta_vlti_sigma_sym,
                "delta_schwarzschild_vlti_improvement_factor_to_3sigma": delta_vlti_improvement_factor_to_3sigma,
                "delta_schwarzschild_keck": delta_keck,
                "delta_schwarzschild_keck_sigma_plus": delta_keck_p,
                "delta_schwarzschild_keck_sigma_minus": delta_keck_m,
                "delta_schwarzschild_keck_sigma_sym": delta_keck_sigma_sym,
                "delta_schwarzschild_keck_improvement_factor_to_3sigma": delta_keck_improvement_factor_to_3sigma,
                "delta_kerr_min": delta_kerr_min,
                "delta_kerr_max": delta_kerr_max,
                "delta_kerr_sigma_uniform": delta_kerr_sigma_uniform,
                "delta_kerr_sigma_uniform_improvement_factor_to_3sigma": delta_kerr_improvement_factor_to_3sigma,
                "theta_unit_uas": theta_unit_uas,
                "theta_unit_uas_sigma": theta_unit_uas_sigma,
                "theta_unit_rel_sigma": ratio_sigma,
                "shadow_diameter_pmodel_uas": theta_p_uas,
                "shadow_diameter_pmodel_uas_sigma": theta_p_uas_sigma,
                "shadow_diameter_pmodel_phase4_uas": theta_p_phase4_uas,
                "shadow_diameter_pmodel_phase4_uas_sigma": theta_p_phase4_uas_sigma,
                "shadow_diameter_gr_uas": theta_gr_uas,
                "shadow_diameter_gr_uas_sigma": theta_gr_uas_sigma,
                "shadow_diameter_gr_kerr_min_uas": theta_gr_kerr_min_uas,
                "shadow_diameter_gr_kerr_max_uas": theta_gr_kerr_max_uas,
                "shadow_diameter_gr_kerr_min_uas_spin_only": theta_gr_kerr_min_spin_only_uas,
                "shadow_diameter_gr_kerr_max_uas_spin_only": theta_gr_kerr_max_spin_only_uas,
                "shadow_diameter_gr_kerr_min_full_uas": theta_gr_kerr_min_full_uas,
                "shadow_diameter_gr_kerr_max_full_uas": theta_gr_kerr_max_full_uas,
                "shadow_diameter_gr_kerr_min_full_uas_spin_only": theta_gr_kerr_min_full_spin_only_uas,
                "shadow_diameter_gr_kerr_max_full_uas_spin_only": theta_gr_kerr_max_full_spin_only_uas,
                "shadow_diameter_diff_p_minus_gr_uas": diff_uas,
                "shadow_diameter_diff_p_minus_gr_uas_sigma": diff_uas_sigma,
                "shadow_diameter_diff_percent": diff_pct,
                "shadow_diameter_coeff_ratio_p_over_gr": coeff_ratio_p_over_gr,
                "shadow_diameter_coeff_ratio_p_over_gr_baseline": coeff_ratio_p_over_gr_baseline,
                "shadow_diameter_coeff_ratio_source": coeff_ratio_source,
                "shadow_diameter_sigma_obs_needed_3sigma_uas": sigma_obs_needed_3sigma_uas,
                "theta_unit_rel_sigma_required_3sigma": theta_unit_rel_sigma_required_3sigma,
                "theta_unit_rel_sigma_improvement_factor_to_3sigma": theta_unit_rel_sigma_improvement_factor_to_3sigma,
                "ring_diameter_sigma_required_3sigma_uas_if_kappa1": ring_sigma_required_3sigma_uas_if_kappa1,
                "ring_diameter_sigma_improvement_factor_to_3sigma_if_kappa1": ring_sigma_improvement_factor_to_3sigma_if_kappa1,
                "kerr_range_mode": kerr_range_mode,
                "kerr_effective_diameter_definition_policy": "envelope(across methods)",
                "kerr_a_star_min_used": kerr_a_star_min_used,
                "kerr_a_star_max_used": kerr_a_star_max_used,
                "kerr_inc_deg_min_used": kerr_inc_deg_min_used,
                "kerr_inc_deg_max_used": kerr_inc_deg_max_used,
                "beta_used": beta,
                "beta_fit_from_obs": beta_fit,
                "beta_fit_from_obs_sigma": beta_fit_sigma,
                "kappa_ring_over_shadow_obs": kappa_obs,
                "kappa_ring_over_shadow_obs_sigma": kappa_obs_sigma,
                "kappa_ring_over_shadow_fit_pmodel": kappa_p,
                "kappa_ring_over_shadow_fit_pmodel_sigma": kappa_p_sigma,
                "kappa_ring_over_shadow_fit_gr": kappa_gr,
                "kappa_ring_over_shadow_fit_gr_sigma": kappa_gr_sigma,
                "kappa_gr_kerr_coeff_range_low": kappa_gr_kerr_low,
                "kappa_gr_kerr_coeff_range_high": kappa_gr_kerr_high,
                "kappa_gr_kerr_coeff_range_low_full": kappa_gr_kerr_low_full,
                "kappa_gr_kerr_coeff_range_high_full": kappa_gr_kerr_high_full,
                "kappa_sigma_assumed_kerr": kappa_sigma_assumed_kerr,
                "kappa_sigma_assumed_kerr_full": kappa_sigma_assumed_kerr_full,
                "kappa_gr_kerr_coeff_range_low_spin_only": kappa_gr_kerr_low_spin_only,
                "kappa_gr_kerr_coeff_range_high_spin_only": kappa_gr_kerr_high_spin_only,
                "kappa_gr_kerr_coeff_range_low_full_spin_only": kappa_gr_kerr_low_full_spin_only,
                "kappa_gr_kerr_coeff_range_high_full_spin_only": kappa_gr_kerr_high_full_spin_only,
                "kappa_sigma_assumed_kerr_spin_only": kappa_sigma_assumed_kerr_spin_only,
                "kappa_sigma_assumed_kerr_full_spin_only": kappa_sigma_assumed_kerr_full_spin_only,
                "kappa_sigma_required_3sigma_if_ring_sigma_zero": kappa_sigma_required_3sigma_if_ring_sigma_zero,
                "kappa_sigma_required_3sigma_if_ring_sigma_current": kappa_sigma_required_3sigma_if_ring_sigma_current,
                "zscore_pmodel": z_p,
                "zscore_gr": z_gr,
                "zscore_pmodel_shadowobs": z_p_shadowobs,
                "zscore_gr_shadowobs": z_gr_shadowobs,
                "source_keys": ";".join(bh.source_keys),
            }
        )

    # Save CSV

    csv_path = out_dir / "eht_shadow_compare.csv"
    header = list(rows[0].keys())
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(str(r.get(k, "")) for k in header))

    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Save JSON
    json_path = out_dir / "eht_shadow_compare.json"
    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(in_path).replace("\\", "/"),
        "pmodel": {
            "beta": beta,
            "beta_source": beta_source,
            "shadow_diameter_coeff_rg": coeff_pmodel,  # coefficient multiplying GM/(c^2 D)
            "shadow_diameter_coeff_rg_phase4": coeff_pmodel_phase4,
            "notes": [
                "最小モデル: P/P0=exp(GM/(c^2 r)), n=(P/P0)^(2β).",
                "球対称屈折率で b=n(r)r の最小が捕獲境界となり、θ_shadow=4eβ(GM/(c^2D)).",
            ],
        },
        "reference_gr": {
            "shadow_diameter_coeff_rg": coeff_gr,
            "notes": ["Schwarzschild近似: θ_shadow = 2√27 (GM/(c^2 D))."],
        },
        "reference_gr_kerr": kerr_range_full,
        "reference_gr_kerr_definition_sensitivity": kerr_range_full_def_env,
        "phase4": {
            "shadow_diameter_coeff_ratio_p_over_gr": coeff_ratio_p_over_gr,
            "shadow_diameter_coeff_ratio_p_over_gr_baseline": coeff_ratio_p_over_gr_baseline,
            "shadow_diameter_coeff_source": coeff_ratio_source,
            "shadow_diameter_coeff_source_relpath": coeff_ratio_source_relpath,
            "shadow_diameter_coeff_phase4_diff_percent": coeff_diff_pct_phase4,
            "shadow_diameter_coeff_diff_percent": (coeff_ratio_p_over_gr - 1.0) * 100.0
            if math.isfinite(coeff_ratio_p_over_gr)
            else None,
            "notes": [
                "差分予測: βをCassini等で固定した上で、ブラックホールのシャドウ直径係数がGRと数%ずれる。",
                "この差を検出するには、リング≒シャドウ近似の系統誤差・質量/距離の不確かさ・観測誤差を同時に詰める必要がある。",
            ],
        },
        "delta_reference": {
            "delta_coeff_p_minus_gr_schwarzschild": delta_coeff_p_minus_gr,
            "delta_sigma_required_3sigma": delta_sigma_required_3sigma,
            "notes": [
                "δ（Schwarzschild shadow deviation）は EHT 論文側の定義に基づく派生量で、モデル依存の系統（κ/放射/散乱/再構成）を含む可能性がある。",
                "ここでは「現状のGR解析の精度スケール」を示す参考指標として扱う（P-modelの検証は κ を明示して進める）。",
            ],
        },
        "rows": rows,
    }
    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # Plot (public-friendly)
    diff_png_path: Optional[Path] = None
    try:
        import matplotlib.pyplot as plt

        _set_japanese_font()

        names = [r["name"] for r in rows]
        x = list(range(len(names)))
        width = 0.26

        obs = [float(r["ring_diameter_obs_uas"]) for r in rows]
        obs_sig = [float(r["ring_diameter_obs_uas_sigma"]) for r in rows]
        pval = [float(r["shadow_diameter_pmodel_uas"]) for r in rows]
        psig = [float(r["shadow_diameter_pmodel_uas_sigma"]) for r in rows]
        gval = [float(r["shadow_diameter_gr_uas"]) for r in rows]
        gsig = [float(r["shadow_diameter_gr_uas_sigma"]) for r in rows]

        fig, ax = plt.subplots(figsize=(11.5, 6.0))

        ax.bar(
            [i - width for i in x],
            pval,
            width=width,
            yerr=psig,
            error_kw={"ecolor": "#1f77b4", "capsize": 3, "alpha": 0.75},
            label=f"P-model（影直径; β={beta:g}）",
            color="#1f77b4",
            alpha=0.9,
        )
        ax.bar(
            [i for i in x],
            gval,
            width=width,
            yerr=gsig,
            error_kw={"ecolor": "#666666", "capsize": 3, "alpha": 0.75},
            label="標準理論（影直径; GR, Schwarzschild）",
            color="#9aa0a6",
            alpha=0.9,
        )
        ax.errorbar(
            [i + width for i in x],
            obs,
            yerr=obs_sig,
            fmt="o",
            color="#d62728",
            label="観測（リング直径 θ_ring）",
            zorder=4,
        )

        shadow_obs = [float(r.get("shadow_diameter_obs_uas", float("nan"))) for r in rows]
        shadow_obs_sig = [float(r.get("shadow_diameter_obs_uas_sigma", float("nan"))) for r in rows]
        # 条件分岐: `any(math.isfinite(v) for v in shadow_obs)` を満たす経路を評価する。
        if any(math.isfinite(v) for v in shadow_obs):
            ax.errorbar(
                [i + width * 1.55 for i in x],
                shadow_obs,
                yerr=shadow_obs_sig,
                fmt="D",
                color="#111111",
                label="参考（推定影直径 d_sh；リングからの推定）",
                zorder=5,
            )

        # Clarify: ring vs inferred d_sh (central value); error bars can overlap (esp. Sgr A*).

        for i in range(len(rows)):
            so = float(shadow_obs[i]) if i < len(shadow_obs) else float("nan")
            # 条件分岐: `not (math.isfinite(so) and math.isfinite(obs[i]))` を満たす経路を評価する。
            if not (math.isfinite(so) and math.isfinite(obs[i])):
                continue

            dr = float(obs[i]) - float(so)
            ax.text(
                i + width * 1.25,
                max(float(obs[i]) + float(obs_sig[i]), float(so) + float(shadow_obs_sig[i])) + 1.0,
                f"θ_ring−d_sh={dr:+.1f} µas",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#111111",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel("角直径 [µas]")
        ax.set_title("EHT：観測リング θ_ring と、影直径 θ_sh（モデル；κ=1）の比較")
        ax.grid(True, alpha=0.25, axis="y")

        # Legend order: observables first, then model predictions.
        handles, labels = ax.get_legend_handles_labels()
        want = [
            "観測（リング直径 θ_ring）",
            "参考（推定影直径 d_sh；リングからの推定）",
        ]
        # Keep the remaining labels (model predictions) in their original order.
        order: List[int] = []
        for w in want:
            # 条件分岐: `w in labels` を満たす経路を評価する。
            if w in labels:
                order.append(labels.index(w))

        order += [i for i in range(len(labels)) if i not in order]
        ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower right")

        # annotate κ_fit (ring / shadow). β はCassini等で固定する前提のため、EHTでは κ を主に見せる。
        for i, row in enumerate(rows):
            kfit = float(row.get("kappa_ring_over_shadow_fit_pmodel", float("nan")))
            ksig = float(row.get("kappa_ring_over_shadow_fit_pmodel_sigma", float("nan")))
            kobs = float(row.get("kappa_ring_over_shadow_obs", float("nan")))
            kobs_sig = float(row.get("kappa_ring_over_shadow_obs_sigma", float("nan")))
            # 条件分岐: `math.isfinite(kfit) and math.isfinite(ksig)` を満たす経路を評価する。
            if math.isfinite(kfit) and math.isfinite(ksig):
                txt = f"κ_fit={kfit:.3f}±{ksig:.3f}"
            else:
                txt = "κ_fit=n/a"

            # 条件分岐: `math.isfinite(kobs) and math.isfinite(kobs_sig)` を満たす経路を評価する。

            if math.isfinite(kobs) and math.isfinite(kobs_sig):
                txt = f"{txt}\nκ_ref(d_sh)={kobs:.3f}±{kobs_sig:.3f}"

            ax.text(i, max(pval[i], gval[i], obs[i]) + 1.0, txt, ha="center", va="bottom", fontsize=9)

        fig.tight_layout()
        png_path = out_dir / "eht_shadow_compare.png"
        fig.savefig(png_path, dpi=220)
        plt.close(fig)
    except Exception as e:
        print(f"[warn] plot skipped: {e}")
        png_path = None

    # Differential prediction (Phase 4)

    diff_public_png_path: Optional[Path] = None
    try:
        import matplotlib.pyplot as plt

        _set_japanese_font()

        names = [r["name"] for r in rows]
        x = list(range(len(names)))
        diff = [float(r["shadow_diameter_diff_p_minus_gr_uas"]) for r in rows]
        diff_sig = [float(r["shadow_diameter_diff_p_minus_gr_uas_sigma"]) for r in rows]
        sigma_need = [float(r["shadow_diameter_sigma_obs_needed_3sigma_uas"]) for r in rows]

        fig2, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.5, 5.2))

        ax0.bar(x, diff, color="#9467bd", alpha=0.9, yerr=diff_sig)
        ax0.axhline(0.0, color="#333333", linewidth=1.0)
        ax0.set_xticks(x)
        ax0.set_xticklabels(names)
        ax0.set_ylabel("差（P-model − GR）[μas]")
        ax0.set_title("差分予測：シャドウ直径の差")
        ax0.grid(True, alpha=0.25, axis="y")

        sigma_need_plot: List[float] = []
        nan_idx: List[int] = []
        for i, v in enumerate(sigma_need):
            # 条件分岐: `math.isfinite(v) and v > 0` を満たす経路を評価する。
            if math.isfinite(v) and v > 0:
                sigma_need_plot.append(v)
            else:
                sigma_need_plot.append(0.0)
                nan_idx.append(i)

        ax1.bar(x, sigma_need_plot, color="#2ca02c", alpha=0.9)
        for i in nan_idx:
            ax1.bar(i, 0.0, color="#2ca02c", alpha=0.25, hatch="///", edgecolor="#2ca02c")
            ax1.text(i, max(sigma_need_plot) * 0.05 if sigma_need_plot else 0.1, "n/a", ha="center", va="bottom", fontsize=9)

        ax1.set_xticks(x)
        ax1.set_xticklabels(names)
        ax1.set_ylabel("必要な観測誤差（1σ）[μas]")
        ax1.set_title("3σで判別するための必要精度（目安）")
        ax1.grid(True, alpha=0.25, axis="y")

        # 条件分岐: `math.isfinite(coeff_ratio_p_over_gr)` を満たす経路を評価する。
        if math.isfinite(coeff_ratio_p_over_gr):
            fig2.suptitle(
                f"EHT：P-model と GR の差分予測（係数比 {coeff_ratio_p_over_gr:.4f}、差 {((coeff_ratio_p_over_gr-1)*100):.2f}%）"
            )
        else:
            fig2.suptitle("EHT：P-model と GR の差分予測（シャドウ直径）")

        fig2.tight_layout()
        diff_png_path = out_dir / "eht_shadow_differential.png"
        fig2.savefig(diff_png_path, dpi=220)
        diff_public_png_path = out_dir / "eht_shadow_differential_public.png"
        fig2.savefig(diff_public_png_path, dpi=220)
        plt.close(fig2)
    except Exception as e:
        print(f"[warn] differential plot skipped: {e}")
        diff_png_path = None
        diff_public_png_path = None

    # Systematics / sensitivity (public-friendly)

    sys_png_path: Optional[Path] = None
    sys_public_png_path: Optional[Path] = None
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        _set_japanese_font()

        fig3, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.8, 5.2))

        # Left: coefficient comparison (P-model vs GR, with Kerr range as a reference)
        beta_grid = np.linspace(0.8, 1.2, 200)
        ax0.plot(beta_grid, 4.0 * math.e * beta_grid, color="#1f77b4", label="P-model：4eβ")
        ax0.axhline(coeff_gr, color="#666666", linestyle="--", label="GR（Schwarzschild）：2√27")

        kmin = kerr_range_full.get("coeff_min") if isinstance(kerr_range_full, dict) else None
        kmax = kerr_range_full.get("coeff_max") if isinstance(kerr_range_full, dict) else None
        # 条件分岐: `isinstance(kmin, (int, float)) and isinstance(kmax, (int, float)) and math.is...` を満たす経路を評価する。
        if isinstance(kmin, (int, float)) and isinstance(kmax, (int, float)) and math.isfinite(kmin) and math.isfinite(kmax):
            ax0.fill_between(beta_grid, float(kmin), float(kmax), color="#9aa0a6", alpha=0.18, label="GR（Kerr 参考レンジ）")

        ax0.scatter([beta], [coeff_pmodel], color="#1f77b4", zorder=3)
        ax0.set_xlabel("β")
        ax0.set_ylabel("シャドウ直径係数（θ / (GM/(c^2 D)))")
        ax0.set_title("係数の比較（βとスピン依存）")
        ax0.grid(True, alpha=0.25)
        ax0.legend(loc="upper left")

        # 条件分岐: `math.isfinite(coeff_ratio_p_over_gr)` を満たす経路を評価する。
        if math.isfinite(coeff_ratio_p_over_gr):
            ax0.text(
                0.805,
                max(coeff_pmodel, coeff_gr) * 0.995,
                f"β={beta:g} での差: {(coeff_ratio_p_over_gr-1)*100:+.2f}%",
                fontsize=10,
                ha="left",
                va="top",
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "#dddddd"},
            )

        # Right: κ inference (ring ≈ κ * shadow)

        names = [r["name"] for r in rows]
        x = np.arange(len(names), dtype=float)
        width = 0.34

        kp = np.array([float(r.get("kappa_ring_over_shadow_fit_pmodel", float("nan"))) for r in rows], dtype=float)
        kps = np.array([float(r.get("kappa_ring_over_shadow_fit_pmodel_sigma", float("nan"))) for r in rows], dtype=float)
        kg = np.array([float(r.get("kappa_ring_over_shadow_fit_gr", float("nan"))) for r in rows], dtype=float)
        kgs = np.array([float(r.get("kappa_ring_over_shadow_fit_gr_sigma", float("nan"))) for r in rows], dtype=float)
        ko = np.array([float(r.get("kappa_ring_over_shadow_obs", float("nan"))) for r in rows], dtype=float)
        kos = np.array([float(r.get("kappa_ring_over_shadow_obs_sigma", float("nan"))) for r in rows], dtype=float)
        kg_lo = np.array([float(r.get("kappa_gr_kerr_coeff_range_low", float("nan"))) for r in rows], dtype=float)
        kg_hi = np.array([float(r.get("kappa_gr_kerr_coeff_range_high", float("nan"))) for r in rows], dtype=float)

        ax1.bar(x - width / 2, kp, width=width, color="#1f77b4", alpha=0.9, label="κ（P-model, β固定）")
        ax1.bar(x + width / 2, kg, width=width, color="#9aa0a6", alpha=0.9, label="κ（GR, Schwarzschild）")
        ax1.errorbar(x - width / 2, kp, yerr=kps, fmt="none", ecolor="#1f77b4", capsize=3)
        ax1.errorbar(x + width / 2, kg, yerr=kgs, fmt="none", ecolor="#666666", capsize=3)

        # Reference κ derived from inferred shadow diameter d_sh, if available.
        ko_mask = np.isfinite(ko) & np.isfinite(kos) & (kos > 0)
        # 条件分岐: `bool(np.any(ko_mask))` を満たす経路を評価する。
        if bool(np.any(ko_mask)):
            ax1.errorbar(
                x[ko_mask],
                ko[ko_mask],
                yerr=kos[ko_mask],
                fmt="D",
                color="#111111",
                ecolor="#111111",
                capsize=3,
                label="κ_ref(d_sh)（一次ソース）",
                zorder=4,
            )

        # Kerr coefficient range → κ range indicator (coefficient systematic only; does not include obs/mass-distance σ).

        for i in range(len(names)):
            # 条件分岐: `math.isfinite(float(kg_lo[i])) and math.isfinite(float(kg_hi[i]))` を満たす経路を評価する。
            if math.isfinite(float(kg_lo[i])) and math.isfinite(float(kg_hi[i])):
                ax1.vlines(
                    float(x[i] + width / 2),
                    float(kg_lo[i]),
                    float(kg_hi[i]),
                    color="#666666",
                    alpha=0.45,
                    linewidth=3.0,
                )

        ax1.axhline(1.0, color="#333333", linewidth=1.0)

        ax1.set_xticks(x)
        ax1.set_xticklabels(names)
        ax1.set_ylabel("κ = リング直径 / シャドウ直径")
        ax1.set_title("リング≒シャドウ近似の系統誤差（κ）\n（縦線=Kerr係数レンジによるκの幅）")
        ax1.grid(True, alpha=0.25, axis="y")
        ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

        fig3.suptitle("EHT：系統誤差（κ）と GR スピン依存（参考）")
        fig3.tight_layout()
        sys_png_path = out_dir / "eht_shadow_systematics.png"
        fig3.savefig(sys_png_path, dpi=220, bbox_inches="tight")
        sys_public_png_path = out_dir / "eht_shadow_systematics_public.png"
        fig3.savefig(sys_public_png_path, dpi=220, bbox_inches="tight")
        plt.close(fig3)
    except Exception as e:
        print(f"[warn] systematics plot skipped: {e}")
        sys_png_path = None
        sys_public_png_path = None

    # Ring morphology (primary-source ranges, if available)

    ring_png_path: Optional[Path] = None
    ring_public_png_path: Optional[Path] = None
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        _set_japanese_font()

        names = [str(r.get("name") or r.get("key") or "") for r in rows]
        x = np.arange(len(names), dtype=float)

        w_min = np.array([float(r.get("ring_fractional_width_min", float("nan"))) for r in rows], dtype=float)
        w_max = np.array([float(r.get("ring_fractional_width_max", float("nan"))) for r in rows], dtype=float)
        a_min = np.array([float(r.get("ring_brightness_asymmetry_min", float("nan"))) for r in rows], dtype=float)
        a_max = np.array([float(r.get("ring_brightness_asymmetry_max", float("nan"))) for r in rows], dtype=float)

        fig5, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.8, 4.8))

        # Width: range or one-sided limit (if only an upper/lower bound is available).
        any_w_lim_hi = False
        any_w_lim_lo = False
        for i in range(len(names)):
            v0 = float(w_min[i])
            v1 = float(w_max[i])
            # 条件分岐: `math.isfinite(v0) and math.isfinite(v1) and v1 >= v0` を満たす経路を評価する。
            if math.isfinite(v0) and math.isfinite(v1) and v1 >= v0:
                mid = 0.5 * (v0 + v1)
                err = 0.5 * (v1 - v0)
                ax0.errorbar(float(x[i]), mid, yerr=err, fmt="o", capsize=4, color="#1f77b4")
                continue

            # 条件分岐: `math.isfinite(v1) and not math.isfinite(v0)` を満たす経路を評価する。

            if math.isfinite(v1) and not math.isfinite(v0):
                any_w_lim_hi = True
                ax0.scatter(float(x[i]), v1, marker="v", color="#1f77b4", zorder=3)
                ax0.text(float(x[i]), v1 + 0.02, f"≤{v1:g}", ha="center", va="bottom", fontsize=9, color="#1f77b4")
                continue

            # 条件分岐: `math.isfinite(v0) and not math.isfinite(v1)` を満たす経路を評価する。

            if math.isfinite(v0) and not math.isfinite(v1):
                any_w_lim_lo = True
                ax0.scatter(float(x[i]), v0, marker="^", color="#1f77b4", zorder=3)
                ax0.text(float(x[i]), v0 + 0.02, f"≥{v0:g}", ha="center", va="bottom", fontsize=9, color="#1f77b4")
                continue

            ax0.text(float(x[i]), 0.02, "n/a", ha="center", va="bottom", fontsize=9, color="#666666")

        ax0.set_xticks(x)
        ax0.set_xticklabels(names)
        ax0.set_xlabel("ターゲット")
        ax0.set_ylabel("W/d（リング幅 / 直径）")
        ax0.set_title("リングの幅（fractional width）")
        ax0.set_ylim(0.0, 0.65)
        ax0.grid(True, alpha=0.25, axis="y")
        # 条件分岐: `any_w_lim_hi or any_w_lim_lo` を満たす経路を評価する。
        if any_w_lim_hi or any_w_lim_lo:
            handles = []
            labels = []
            # 条件分岐: `any_w_lim_hi` を満たす経路を評価する。
            if any_w_lim_hi:
                handles.append(ax0.scatter([], [], marker="v", color="#1f77b4"))
                labels.append("上限（≤）")

            # 条件分岐: `any_w_lim_lo` を満たす経路を評価する。

            if any_w_lim_lo:
                handles.append(ax0.scatter([], [], marker="^", color="#1f77b4"))
                labels.append("下限（≥）")

            # 条件分岐: `handles` を満たす経路を評価する。

            if handles:
                ax0.legend(handles, labels, loc="upper right")

        # Asymmetry: range or one-sided limit.

        any_a_lim_hi = False
        any_a_lim_lo = False
        for i in range(len(names)):
            v0 = float(a_min[i])
            v1 = float(a_max[i])
            # 条件分岐: `math.isfinite(v0) and math.isfinite(v1) and v1 >= v0` を満たす経路を評価する。
            if math.isfinite(v0) and math.isfinite(v1) and v1 >= v0:
                mid = 0.5 * (v0 + v1)
                err = 0.5 * (v1 - v0)
                ax1.errorbar(float(x[i]), mid, yerr=err, fmt="o", capsize=4, color="#ff7f0e")
                continue

            # 条件分岐: `math.isfinite(v1) and not math.isfinite(v0)` を満たす経路を評価する。

            if math.isfinite(v1) and not math.isfinite(v0):
                any_a_lim_hi = True
                ax1.scatter(float(x[i]), v1, marker="v", color="#ff7f0e", zorder=3)
                ax1.text(float(x[i]), v1 + 0.01, f"≤{v1:g}", ha="center", va="bottom", fontsize=9, color="#ff7f0e")
                continue

            # 条件分岐: `math.isfinite(v0) and not math.isfinite(v1)` を満たす経路を評価する。

            if math.isfinite(v0) and not math.isfinite(v1):
                any_a_lim_lo = True
                ax1.scatter(float(x[i]), v0, marker="^", color="#ff7f0e", zorder=3)
                ax1.text(float(x[i]), v0 + 0.01, f"≥{v0:g}", ha="center", va="bottom", fontsize=9, color="#ff7f0e")
                continue

            ax1.text(float(x[i]), 0.01, "n/a", ha="center", va="bottom", fontsize=9, color="#666666")

        ax1.set_xticks(x)
        ax1.set_xticklabels(names)
        ax1.set_xlabel("ターゲット")
        ax1.set_ylabel("A（brightness asymmetry）")
        ax1.set_title("リングの非対称性（brightness asymmetry）")
        ax1.set_ylim(0.0, 0.35)
        ax1.grid(True, alpha=0.25, axis="y")
        # 条件分岐: `any_a_lim_hi or any_a_lim_lo` を満たす経路を評価する。
        if any_a_lim_hi or any_a_lim_lo:
            handles = []
            labels = []
            # 条件分岐: `any_a_lim_hi` を満たす経路を評価する。
            if any_a_lim_hi:
                handles.append(ax1.scatter([], [], marker="v", color="#ff7f0e"))
                labels.append("上限（≤）")

            # 条件分岐: `any_a_lim_lo` を満たす経路を評価する。

            if any_a_lim_lo:
                handles.append(ax1.scatter([], [], marker="^", color="#ff7f0e"))
                labels.append("下限（≥）")

            # 条件分岐: `handles` を満たす経路を評価する。

            if handles:
                ax1.legend(handles, labels, loc="upper right")

        fig5.suptitle("EHT：リング形状指標（一次ソースの範囲）")
        fig5.tight_layout()
        ring_png_path = out_dir / "eht_ring_morphology.png"
        fig5.savefig(ring_png_path, dpi=220)
        ring_public_png_path = out_dir / "eht_ring_morphology_public.png"
        fig5.savefig(ring_public_png_path, dpi=220)
        plt.close(fig5)
    except Exception as e:
        print(f"[warn] ring morphology plot skipped: {e}")
        ring_png_path = None
        ring_public_png_path = None

    # Scattering budget (scale comparison): ring diameter / width vs scattering kernel.

    scat_png_path: Optional[Path] = None
    scat_public_png_path: Optional[Path] = None
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        _set_japanese_font()

        names = [str(r.get("name") or r.get("key") or "") for r in rows]
        x = np.arange(len(names), dtype=float)

        ring = np.array([float(r.get("ring_diameter_obs_uas", float("nan"))) for r in rows], dtype=float)
        ring_s = np.array([float(r.get("ring_diameter_obs_uas_sigma", float("nan"))) for r in rows], dtype=float)

        sca = np.array([float(r.get("scattering_kernel_fwhm_major_uas", float("nan"))) for r in rows], dtype=float)
        sca_s = np.array(
            [float(r.get("scattering_kernel_fwhm_major_uas_sigma", float("nan"))) for r in rows], dtype=float
        )
        scb = np.array([float(r.get("scattering_kernel_fwhm_minor_uas", float("nan"))) for r in rows], dtype=float)
        scb_s = np.array(
            [float(r.get("scattering_kernel_fwhm_minor_uas_sigma", float("nan"))) for r in rows], dtype=float
        )

        w_min = np.array([float(r.get("ring_fractional_width_min", float("nan"))) for r in rows], dtype=float)
        w_max = np.array([float(r.get("ring_fractional_width_max", float("nan"))) for r in rows], dtype=float)

        # Convert fractional width W/d to absolute ring width W (µas) using observed ring diameter.
        w_abs_min = w_min * ring
        w_abs_max = w_max * ring

        fig7, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.0, 4.9))

        # Panel A: ring diameter vs scattering kernel axes.
        ax0.errorbar(x, ring, yerr=np.where(np.isfinite(ring_s) & (ring_s > 0), ring_s, 0.0), fmt="o", capsize=4, color="#1f77b4", label="リング直径（観測）")

        a_ok = np.isfinite(sca)
        # 条件分岐: `bool(np.any(a_ok))` を満たす経路を評価する。
        if bool(np.any(a_ok)):
            ax0.errorbar(
                x[a_ok] + 0.06,
                sca[a_ok],
                yerr=np.where(np.isfinite(sca_s[a_ok]) & (sca_s[a_ok] > 0), sca_s[a_ok], 0.0),
                fmt="x",
                capsize=3,
                color="#d62728",
                label="散乱カーネルFWHM（長軸, 230 GHz）",
            )

        b_ok = np.isfinite(scb)
        # 条件分岐: `bool(np.any(b_ok))` を満たす経路を評価する。
        if bool(np.any(b_ok)):
            ax0.errorbar(
                x[b_ok] + 0.06,
                scb[b_ok],
                yerr=np.where(np.isfinite(scb_s[b_ok]) & (scb_s[b_ok] > 0), scb_s[b_ok], 0.0),
                fmt="+",
                capsize=3,
                color="#ff7f0e",
                label="散乱カーネルFWHM（短軸, 230 GHz）",
            )

        for i, name in enumerate(names):
            # 条件分岐: `not name` を満たす経路を評価する。
            if not name:
                continue

            # 条件分岐: `not (math.isfinite(ring[i]) and ring[i] > 0)` を満たす経路を評価する。

            if not (math.isfinite(ring[i]) and ring[i] > 0):
                ax0.text(float(x[i]), 0.5, "n/a", ha="center", va="bottom", fontsize=9, color="#666666")

        ax0.set_xticks(x)
        ax0.set_xticklabels(names)
        ax0.set_xlabel("ターゲット")
        ax0.set_ylabel("角サイズ（µas）")
        ax0.set_title("直径スケール：観測リングと散乱blur（参考）")
        ax0.grid(True, alpha=0.25, axis="y")
        ax0.legend(loc="upper right")

        # Panel B: ring width vs scattering kernel axes.
        any_w_lim_hi = False
        any_w_lim_lo = False
        for i in range(len(names)):
            v0 = float(w_abs_min[i])
            v1 = float(w_abs_max[i])
            # 条件分岐: `math.isfinite(v0) and math.isfinite(v1) and v1 >= v0` を満たす経路を評価する。
            if math.isfinite(v0) and math.isfinite(v1) and v1 >= v0:
                mid = 0.5 * (v0 + v1)
                err = 0.5 * (v1 - v0)
                ax1.errorbar(float(x[i]), mid, yerr=err, fmt="o", capsize=4, color="#1f77b4", label=None)
                continue

            # 条件分岐: `math.isfinite(v1) and not math.isfinite(v0)` を満たす経路を評価する。

            if math.isfinite(v1) and not math.isfinite(v0):
                any_w_lim_hi = True
                ax1.scatter(float(x[i]), v1, marker="v", color="#1f77b4", zorder=3)
                ax1.text(float(x[i]), v1 + 0.8, f"≤{v1:.1f}", ha="center", va="bottom", fontsize=9, color="#1f77b4")
                continue

            # 条件分岐: `math.isfinite(v0) and not math.isfinite(v1)` を満たす経路を評価する。

            if math.isfinite(v0) and not math.isfinite(v1):
                any_w_lim_lo = True
                ax1.scatter(float(x[i]), v0, marker="^", color="#1f77b4", zorder=3)
                ax1.text(float(x[i]), v0 + 0.8, f"≥{v0:.1f}", ha="center", va="bottom", fontsize=9, color="#1f77b4")
                continue

            ax1.text(float(x[i]), 0.6, "n/a", ha="center", va="bottom", fontsize=9, color="#666666")

        # Overlay scattering kernel axes for scale comparison.

        if bool(np.any(a_ok)):
            ax1.errorbar(
                x[a_ok] + 0.06,
                sca[a_ok],
                yerr=np.where(np.isfinite(sca_s[a_ok]) & (sca_s[a_ok] > 0), sca_s[a_ok], 0.0),
                fmt="x",
                capsize=3,
                color="#d62728",
                label="散乱（長軸）",
            )

        # 条件分岐: `bool(np.any(b_ok))` を満たす経路を評価する。

        if bool(np.any(b_ok)):
            ax1.errorbar(
                x[b_ok] + 0.06,
                scb[b_ok],
                yerr=np.where(np.isfinite(scb_s[b_ok]) & (scb_s[b_ok] > 0), scb_s[b_ok], 0.0),
                fmt="+",
                capsize=3,
                color="#ff7f0e",
                label="散乱（短軸）",
            )

        ax1.set_xticks(x)
        ax1.set_xticklabels(names)
        ax1.set_xlabel("ターゲット")
        ax1.set_ylabel("角サイズ（µas）")
        ax1.set_title("形状スケール：リング幅（W/d→µas）と散乱blur（参考）")
        ax1.grid(True, alpha=0.25, axis="y")

        # 条件分岐: `any_w_lim_hi or any_w_lim_lo` を満たす経路を評価する。
        if any_w_lim_hi or any_w_lim_lo:
            handles = []
            labels = []
            # 条件分岐: `any_w_lim_hi` を満たす経路を評価する。
            if any_w_lim_hi:
                handles.append(ax1.scatter([], [], marker="v", color="#1f77b4"))
                labels.append("幅の上限（≤）")

            # 条件分岐: `any_w_lim_lo` を満たす経路を評価する。

            if any_w_lim_lo:
                handles.append(ax1.scatter([], [], marker="^", color="#1f77b4"))
                labels.append("幅の下限（≥）")

            # 条件分岐: `handles` を満たす経路を評価する。

            if handles:
                ax1.legend(handles, labels, loc="upper left")

        fig7.suptitle("EHT：散乱（Sgr A*）のスケール感（κ系統の背景）")
        fig7.tight_layout()
        scat_png_path = out_dir / "eht_scattering_budget.png"
        fig7.savefig(scat_png_path, dpi=220)
        scat_public_png_path = out_dir / "eht_scattering_budget_public.png"
        fig7.savefig(scat_public_png_path, dpi=220)
        plt.close(fig7)
    except Exception as e:
        print(f"[warn] scattering budget plot skipped: {e}")
        scat_png_path = None
        scat_public_png_path = None

    # Refractive scattering (Zhu 2018): compare distortion scale vs required precision.

    refr_png_path: Optional[Path] = None
    refr_public_png_path: Optional[Path] = None
    refr_metrics_json_path: Optional[Path] = None
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        _set_japanese_font()

        names = [str(r.get("name") or r.get("key") or "") for r in rows]
        x = np.arange(len(names), dtype=float)

        ring_s = np.array([float(r.get("ring_diameter_obs_uas_sigma", float("nan"))) for r in rows], dtype=float)
        need_s = np.array(
            [float(r.get("shadow_diameter_sigma_obs_needed_3sigma_uas", float("nan"))) for r in rows], dtype=float
        )
        sca_s = np.array(
            [float(r.get("scattering_kernel_fwhm_major_uas_sigma", float("nan"))) for r in rows], dtype=float
        )
        scb_s = np.array(
            [float(r.get("scattering_kernel_fwhm_minor_uas_sigma", float("nan"))) for r in rows], dtype=float
        )
        # Elementwise max ignoring NaN (avoids "All-NaN slice" warnings).
        kernel_s = np.fmax(sca_s, scb_s)

        wmin = np.array([float(r.get("refractive_wander_uas_min", float("nan"))) for r in rows], dtype=float)
        wmax = np.array([float(r.get("refractive_wander_uas_max", float("nan"))) for r in rows], dtype=float)
        w_ok = np.isfinite(wmin) & np.isfinite(wmax) & (wmax >= wmin)
        w_mid = np.where(w_ok, 0.5 * (wmin + wmax), np.nan)
        w_err = np.where(w_ok, 0.5 * (wmax - wmin), 0.0)

        dmin = np.array([float(r.get("refractive_distortion_uas_min", float("nan"))) for r in rows], dtype=float)
        dmax = np.array([float(r.get("refractive_distortion_uas_max", float("nan"))) for r in rows], dtype=float)
        d_ok = np.isfinite(dmin) & np.isfinite(dmax) & (dmax >= dmin)
        d_mid = np.where(d_ok, 0.5 * (dmin + dmax), np.nan)
        d_err = np.where(d_ok, 0.5 * (dmax - dmin), 0.0)

        amin = np.array([float(r.get("refractive_asymmetry_uas_min", float("nan"))) for r in rows], dtype=float)
        amax = np.array([float(r.get("refractive_asymmetry_uas_max", float("nan"))) for r in rows], dtype=float)
        a_ok = np.isfinite(amin) & np.isfinite(amax) & (amax >= amin)
        a_mid = np.where(a_ok, 0.5 * (amin + amax), np.nan)
        a_err = np.where(a_ok, 0.5 * (amax - amin), 0.0)

        fig8, ax = plt.subplots(1, 1, figsize=(11.8, 4.9))

        ax.scatter(x - 0.06, ring_s, marker="o", color="#1f77b4", label="観測リング直径の統計誤差 σ_obs")
        ax.scatter(x + 0.06, need_s, marker="D", color="#2ca02c", label="3σ判別に必要なσ_obs（理想）")

        k_ok = np.isfinite(kernel_s)
        # 条件分岐: `bool(np.any(k_ok))` を満たす経路を評価する。
        if bool(np.any(k_ok)):
            ax.scatter(
                x[k_ok],
                kernel_s[k_ok],
                marker="x",
                color="#d62728",
                label="散乱カーネル係数の不確かさ（1σ, 長軸/短軸）",
                zorder=3,
            )

        # 条件分岐: `bool(np.any(w_ok))` を満たす経路を評価する。

        if bool(np.any(w_ok)):
            ax.errorbar(
                x[w_ok] - 0.06,
                w_mid[w_ok],
                yerr=w_err[w_ok],
                fmt="^",
                capsize=5,
                color="#9467bd",
                alpha=0.95,
                label="屈折散乱 wander（min–max, Zhu 2018; 230 GHz）",
            )

        # 条件分岐: `bool(np.any(d_ok))` を満たす経路を評価する。

        if bool(np.any(d_ok)):
            ax.errorbar(
                x[d_ok],
                d_mid[d_ok],
                yerr=d_err[d_ok],
                fmt="s",
                capsize=5,
                color="#ff7f0e",
                alpha=0.95,
                label="屈折散乱 distortion（min–max, Zhu 2018; 230 GHz）",
            )

        # 条件分岐: `bool(np.any(a_ok))` を満たす経路を評価する。

        if bool(np.any(a_ok)):
            ax.errorbar(
                x[a_ok] + 0.06,
                a_mid[a_ok],
                yerr=a_err[a_ok],
                fmt="v",
                capsize=5,
                color="#8c564b",
                alpha=0.95,
                label="屈折散乱 asymmetry（min–max, Zhu 2018; 230 GHz）",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_xlabel("ターゲット")
        ax.set_ylabel("角スケール（μas）")
        ax.set_title("EHT：屈折散乱のゆらぎスケール（κ系統の一要因）と必要精度（参考）")
        ax.grid(True, alpha=0.25, axis="y")
        ax.legend(loc="upper left")

        ymax = float(
            np.nanmax(
                np.concatenate(
                    [
                        ring_s[np.isfinite(ring_s)],
                        need_s[np.isfinite(need_s)],
                        kernel_s[np.isfinite(kernel_s)],
                        wmax[np.isfinite(wmax)],
                        dmax[np.isfinite(dmax)],
                        amax[np.isfinite(amax)],
                    ]
                )
            )
        )
        # 条件分岐: `math.isfinite(ymax) and ymax > 0` を満たす経路を評価する。
        if math.isfinite(ymax) and ymax > 0:
            ax.set_ylim(0.0, max(1.2 * ymax, 1.0))

        fig8.tight_layout()
        refr_png_path = out_dir / "eht_refractive_scattering_limits.png"
        fig8.savefig(refr_png_path, dpi=220)
        refr_public_png_path = out_dir / "eht_refractive_scattering_limits_public.png"
        fig8.savefig(refr_public_png_path, dpi=220)
        plt.close(fig8)

        # Save metrics (for reproducible decision-making / gating).
        def _ratio(a: float, b: float) -> Optional[float]:
            # 条件分岐: `not (math.isfinite(a) and math.isfinite(b)) or b == 0` を満たす経路を評価する。
            if not (math.isfinite(a) and math.isfinite(b)) or b == 0:
                return None

            return a / b

        out_rows: List[Dict[str, Any]] = []
        for i, r in enumerate(rows):
            name = str(r.get("name") or r.get("key") or "")
            out_rows.append(
                {
                    "key": str(r.get("key") or ""),
                    "name": name,
                    "ring_sigma_uas": float(ring_s[i]) if math.isfinite(float(ring_s[i])) else None,
                    "sigma_needed_3sigma_uas": float(need_s[i]) if math.isfinite(float(need_s[i])) else None,
                    "scatter_kernel_coeff_sigma_uas": float(kernel_s[i]) if math.isfinite(float(kernel_s[i])) else None,
                    "refractive_wander_uas_min": float(wmin[i]) if math.isfinite(float(wmin[i])) else None,
                    "refractive_wander_uas_max": float(wmax[i]) if math.isfinite(float(wmax[i])) else None,
                    "refractive_distortion_uas_min": float(dmin[i]) if math.isfinite(float(dmin[i])) else None,
                    "refractive_distortion_uas_max": float(dmax[i]) if math.isfinite(float(dmax[i])) else None,
                    "refractive_asymmetry_uas_min": float(amin[i]) if math.isfinite(float(amin[i])) else None,
                    "refractive_asymmetry_uas_max": float(amax[i]) if math.isfinite(float(amax[i])) else None,
                    "ratios_to_sigma_needed_3sigma": {
                        "wander_min_over_need": _ratio(float(wmin[i]), float(need_s[i])),
                        "wander_max_over_need": _ratio(float(wmax[i]), float(need_s[i])),
                        "distortion_min_over_need": _ratio(float(dmin[i]), float(need_s[i])),
                        "distortion_max_over_need": _ratio(float(dmax[i]), float(need_s[i])),
                        "asymmetry_min_over_need": _ratio(float(amin[i]), float(need_s[i])),
                        "asymmetry_max_over_need": _ratio(float(amax[i]), float(need_s[i])),
                    },
                }
            )

        refr_metrics_json_path = out_dir / "eht_refractive_scattering_limits_metrics.json"
        refr_metrics_json_path.write_text(
            json.dumps(
                {
                    "generated_utc": datetime.now(timezone.utc).isoformat(),
                    "input": str(in_path).replace("\\", "/"),
                    "figure_png": str(refr_png_path).replace("\\", "/") if isinstance(refr_png_path, Path) else None,
                    "figure_png_public": (
                        str(refr_public_png_path).replace("\\", "/") if isinstance(refr_public_png_path, Path) else None
                    ),
                    "rows": out_rows,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    except Exception as e:
        print(f"[warn] refractive scattering plot skipped: {e}")
        refr_png_path = None
        refr_public_png_path = None
        refr_metrics_json_path = None

    # Z-score summary (how well each model matches the observed ring diameter under κ=1).

    z_png_path: Optional[Path] = None
    z_public_png_path: Optional[Path] = None
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        _set_japanese_font()

        names = [str(r.get("name") or r.get("key") or "") for r in rows]
        y = np.arange(len(names), dtype=float)
        z_p = np.array([float(r.get("zscore_pmodel", float("nan"))) for r in rows], dtype=float)
        z_gr = np.array([float(r.get("zscore_gr", float("nan"))) for r in rows], dtype=float)

        fig4, ax = plt.subplots(1, 1, figsize=(11.8, 4.9))
        h = 0.36
        ax.barh(y - h / 2, z_p, height=h, color="#1f77b4", alpha=0.9, label="P-model（β固定, κ=1）")
        ax.barh(y + h / 2, z_gr, height=h, color="#9aa0a6", alpha=0.9, label="GR（Schwarzschild, κ=1）")

        ax.axvline(0.0, color="#333333", linewidth=1.0)
        for v, c0 in [(1.0, "#2ca02c"), (2.0, "#ff7f0e"), (3.0, "#d62728")]:
            ax.axvline(+v, color=c0, linestyle="--", linewidth=1.0, alpha=0.55)
            ax.axvline(-v, color=c0, linestyle="--", linewidth=1.0, alpha=0.55)

        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.set_xlabel("zスコア = (観測 - 予測) / σ_total")
        ax.set_title("EHT：観測とモデルのずれ（zスコア, κ=1 の仮定）")
        ax.grid(True, alpha=0.25, axis="x")
        ax.legend(loc="lower right")

        fig4.tight_layout()
        z_png_path = out_dir / "eht_shadow_zscores.png"
        fig4.savefig(z_png_path, dpi=220)
        z_public_png_path = out_dir / "eht_shadow_zscores_public.png"
        fig4.savefig(z_public_png_path, dpi=220)
        plt.close(fig4)
    except Exception as e:
        print(f"[warn] zscore plot skipped: {e}")
        z_png_path = None
        z_public_png_path = None

    # Public: κ (ring/shadow) inferred from obs vs model shadow.

    kappa_png_path: Optional[Path] = None
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        _set_japanese_font()

        names = [str(r.get("name") or r.get("key") or "") for r in rows]
        x = np.arange(len(names), dtype=float)

        kfit = np.array([float(r.get("kappa_ring_over_shadow_fit_pmodel", float("nan"))) for r in rows], dtype=float)
        kfit_sig = np.array(
            [float(r.get("kappa_ring_over_shadow_fit_pmodel_sigma", float("nan"))) for r in rows], dtype=float
        )
        kobs = np.array([float(r.get("kappa_ring_over_shadow_obs", float("nan"))) for r in rows], dtype=float)
        kobs_sig = np.array([float(r.get("kappa_ring_over_shadow_obs_sigma", float("nan"))) for r in rows], dtype=float)

        fig6, ax = plt.subplots(1, 1, figsize=(11.2, 4.9))
        ax.axhline(1.0, color="#666666", lw=1.2, ls="--", alpha=0.85, label="κ=1（リング≒シャドウの目安）")

        ax.errorbar(
            x,
            kfit,
            yerr=np.where(np.isfinite(kfit_sig) & (kfit_sig > 0), kfit_sig, 0.0),
            fmt="o",
            capsize=5,
            color="#1f77b4",
            alpha=0.9,
            label="κ_fit（観測リング ÷ P-modelシャドウ）",
        )

        # 条件分岐: `np.any(np.isfinite(kobs))` を満たす経路を評価する。
        if np.any(np.isfinite(kobs)):
            ax.errorbar(
                x + 0.06,
                kobs,
                yerr=np.where(np.isfinite(kobs_sig) & (kobs_sig > 0), kobs_sig, 0.0),
                fmt="D",
                capsize=5,
                color="#111111",
                alpha=0.9,
                label="κ_ref(d_sh)（一次ソース）",
            )

        for xi, v, sig in zip(x, kfit, kfit_sig):
            # 条件分岐: `not (math.isfinite(v) and v > 0)` を満たす経路を評価する。
            if not (math.isfinite(v) and v > 0):
                continue

            txt = f"{v:.3f}"
            # 条件分岐: `math.isfinite(sig) and sig > 0` を満たす経路を評価する。
            if math.isfinite(sig) and sig > 0:
                txt += f"±{sig:.3f}"

            ax.text(float(xi), float(v) + 0.02, txt, ha="center", va="bottom", fontsize=9.5, color="#111")

        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_xlabel("ターゲット")
        ax.set_ylabel("κ（リング直径 / シャドウ直径）")
        ax.set_title("EHT：κ（リング/シャドウ）— κ_fit（モデル）と κ_ref(d_sh)")
        ax.grid(True, alpha=0.25, axis="y")
        ax.legend(loc="upper right")

        finite = [float(v) for v in kfit.tolist() if math.isfinite(float(v))]
        # 条件分岐: `finite` を満たす経路を評価する。
        if finite:
            lo = min(finite)
            hi = max(finite)
            pad = max(0.06, 0.35 * (hi - lo))
            ax.set_ylim(lo - pad, hi + pad)

        fig6.tight_layout()
        kappa_png_path = out_dir / "eht_kappa_fit.png"
        fig6.savefig(kappa_png_path, dpi=220)
        kappa_public_png_path = out_dir / "eht_kappa_fit_public.png"
        fig6.savefig(kappa_public_png_path, dpi=220)
        plt.close(fig6)
    except Exception as e:
        print(f"[warn] kappa plot skipped: {e}")
        kappa_png_path = None
        kappa_public_png_path = None

    # κ precision requirements (3σ separation) vs current κ/ring uncertainties.

    kappa_req_png_path: Optional[Path] = None
    kappa_req_public_png_path: Optional[Path] = None
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        _set_japanese_font()

        names = [str(r.get("name") or r.get("key") or "") for r in rows]
        x = np.arange(len(names), dtype=float)

        ring = np.array([float(r.get("ring_diameter_obs_uas", float("nan"))) for r in rows], dtype=float)
        ring_sig = np.array([float(r.get("ring_diameter_obs_uas_sigma", float("nan"))) for r in rows], dtype=float)
        ring_sig_pct = 100.0 * (ring_sig / ring)

        kreq = np.array(
            [float(r.get("kappa_sigma_required_3sigma_if_ring_sigma_zero", float("nan"))) for r in rows], dtype=float
        )
        kreq_pct = 100.0 * kreq

        # Current κ uncertainty estimate: prefer κ_ref(d_sh) (if available), else Kerr-range proxy.
        ksig_pct = np.full(len(rows), float("nan"), dtype=float)
        for i, r in enumerate(rows):
            kobs = float(r.get("kappa_ring_over_shadow_obs", float("nan")))
            kobs_sig = float(r.get("kappa_ring_over_shadow_obs_sigma", float("nan")))
            # 条件分岐: `math.isfinite(kobs) and kobs > 0 and math.isfinite(kobs_sig) and kobs_sig >= 0` を満たす経路を評価する。
            if math.isfinite(kobs) and kobs > 0 and math.isfinite(kobs_sig) and kobs_sig >= 0:
                ksig_pct[i] = 100.0 * (kobs_sig / kobs)
                continue

            klo = float(r.get("kappa_gr_kerr_coeff_range_low", float("nan")))
            khi = float(r.get("kappa_gr_kerr_coeff_range_high", float("nan")))
            # 条件分岐: `math.isfinite(klo) and math.isfinite(khi) and khi > klo` を満たす経路を評価する。
            if math.isfinite(klo) and math.isfinite(khi) and khi > klo:
                kmean = 0.5 * (klo + khi)
                ksig = (khi - klo) / math.sqrt(12.0)
                # 条件分岐: `math.isfinite(kmean) and kmean > 0` を満たす経路を評価する。
                if math.isfinite(kmean) and kmean > 0:
                    ksig_pct[i] = 100.0 * (ksig / kmean)

        fig7, ax = plt.subplots(1, 1, figsize=(12.8, 4.9))

        ax.bar(x, kreq_pct, color="#1f77b4", alpha=0.25, label="必要 κ精度（1σ, ringσ→0 の最良ケース）")
        ax.plot(x, ring_sig_pct, "o", color="#ff7f0e", alpha=0.95, label="現状 ring σ/diameter（%）")
        # 条件分岐: `np.any(np.isfinite(ksig_pct))` を満たす経路を評価する。
        if np.any(np.isfinite(ksig_pct)):
            ax.plot(x + 0.06, ksig_pct, "D", color="#111111", alpha=0.9, label="現状 κσ/κ（一次ソース/参考）")

        for v, c0, lab in [(1.0, "#2ca02c", "1%"), (2.0, "#2ca02c", "2%")]:
            ax.axhline(v, color=c0, linestyle="--", linewidth=1.0, alpha=0.55)

        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_xlabel("ターゲット")
        ax.set_ylabel("相対不確かさ（%）")
        ax.set_title("EHT：κ（リング/シャドウ変換）を何%まで詰める必要があるか（3σ判別の入口）")
        ax.grid(True, alpha=0.25, axis="y")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

        finite = [float(v) for v in np.concatenate([kreq_pct, ring_sig_pct, ksig_pct]) if math.isfinite(float(v))]
        # 条件分岐: `finite` を満たす経路を評価する。
        if finite:
            hi = max(finite)
            ax.set_ylim(0.0, max(5.0, 1.15 * hi))

        fig7.tight_layout()
        kappa_req_png_path = out_dir / "eht_kappa_precision_required.png"
        fig7.savefig(kappa_req_png_path, dpi=220, bbox_inches="tight")
        kappa_req_public_png_path = out_dir / "eht_kappa_precision_required_public.png"
        fig7.savefig(kappa_req_public_png_path, dpi=220, bbox_inches="tight")
        plt.close(fig7)
    except Exception as e:
        print(f"[warn] kappa precision plot skipped: {e}")
        kappa_req_png_path = None
        kappa_req_public_png_path = None

    # Reference: δ (Schwarzschild shadow deviation) precision required to see the coeff-level difference (model-dependent quantity).

    delta_req_png_path: Optional[Path] = None
    delta_req_public_png_path: Optional[Path] = None
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        _set_japanese_font()

        names = [str(r.get("name") or r.get("key") or "") for r in rows]
        x = np.arange(len(names), dtype=float)

        req_pct = np.array(
            [100.0 * float(r.get("delta_sigma_required_3sigma", float("nan"))) for r in rows], dtype=float
        )
        vlti_pct = np.array(
            [100.0 * float(r.get("delta_schwarzschild_vlti_sigma_sym", float("nan"))) for r in rows], dtype=float
        )
        keck_pct = np.array(
            [100.0 * float(r.get("delta_schwarzschild_keck_sigma_sym", float("nan"))) for r in rows], dtype=float
        )
        kerr_pct = np.array(
            [100.0 * float(r.get("delta_kerr_sigma_uniform", float("nan"))) for r in rows], dtype=float
        )

        fig9, ax = plt.subplots(1, 1, figsize=(12.8, 4.9))
        req_label = "必要 δ精度（1σ, 係数差の3σ判別; 参考）"
        # 条件分岐: `math.isfinite(coeff_diff_pct_phase4)` を満たす経路を評価する。
        if math.isfinite(coeff_diff_pct_phase4):
            req_label = f"必要 δ精度（1σ, 係数差{coeff_diff_pct_phase4:.4f}%の3σ判別; 参考）"

        ax.bar(x, req_pct, color="#1f77b4", alpha=0.25, label=req_label)

        # 条件分岐: `np.any(np.isfinite(vlti_pct))` を満たす経路を評価する。
        if np.any(np.isfinite(vlti_pct)):
            ax.plot(x, vlti_pct, "o", color="#ff7f0e", alpha=0.95, label="現状 δσ（VLTI; 一次ソース）")

        # 条件分岐: `np.any(np.isfinite(keck_pct))` を満たす経路を評価する。

        if np.any(np.isfinite(keck_pct)):
            ax.plot(x + 0.06, keck_pct, "D", color="#111111", alpha=0.9, label="現状 δσ（Keck; 一次ソース）")

        # 条件分岐: `np.any(np.isfinite(kerr_pct))` を満たす経路を評価する。

        if np.any(np.isfinite(kerr_pct)):
            ax.plot(x - 0.06, kerr_pct, "s", color="#9aa0a6", alpha=0.9, label="Kerrレンジ由来の δ系統（uniform仮定; 参考）")

        for v, c0 in [(1.0, "#2ca02c"), (2.0, "#2ca02c")]:
            ax.axhline(v, color=c0, linestyle="--", linewidth=1.0, alpha=0.55)

        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_xlabel("ターゲット")
        ax.set_ylabel("相対不確かさ（%）")
        ax.set_title("EHT：δ（Schwarzschild shadow deviation）の必要精度（参考; δはモデル依存）")
        ax.grid(True, alpha=0.25, axis="y")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)

        finite = [float(v) for v in np.concatenate([req_pct, vlti_pct, keck_pct, kerr_pct]) if math.isfinite(float(v))]
        # 条件分岐: `finite` を満たす経路を評価する。
        if finite:
            ax.set_ylim(0.0, max(5.0, 1.15 * max(finite)))

        fig9.tight_layout()
        delta_req_png_path = out_dir / "eht_delta_precision_required.png"
        fig9.savefig(delta_req_png_path, dpi=220, bbox_inches="tight")
        delta_req_public_png_path = out_dir / "eht_delta_precision_required_public.png"
        fig9.savefig(delta_req_public_png_path, dpi=220, bbox_inches="tight")
        plt.close(fig9)
    except Exception as e:
        print(f"[warn] delta precision plot skipped: {e}")
        delta_req_png_path = None
        delta_req_public_png_path = None

    # κ tradeoff: allowed κσ vs ringσ for 3σ discrimination (under κ≈1 error propagation).

    kappa_trade_png_path: Optional[Path] = None
    kappa_trade_public_png_path: Optional[Path] = None
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        _set_japanese_font()

        n = len(rows)
        # 条件分岐: `n <= 0` を満たす経路を評価する。
        if n <= 0:
            raise RuntimeError("no rows")

        fig8, axes = plt.subplots(1, n, figsize=(5.6 * n, 4.9), sharey=True)
        # 条件分岐: `n == 1` を満たす経路を評価する。
        if n == 1:
            axes = [axes]

        for ax, r in zip(axes, rows, strict=False):
            name = str(r.get("name") or r.get("key") or "")
            ring = float(r.get("ring_diameter_obs_uas", float("nan")))
            ring_sig = float(r.get("ring_diameter_obs_uas_sigma", float("nan")))
            sigma_need = float(r.get("shadow_diameter_sigma_obs_needed_3sigma_uas", float("nan")))
            theta_rel = float(r.get("theta_unit_rel_sigma", float("nan")))
            theta_rel_req = float(r.get("theta_unit_rel_sigma_required_3sigma", float("nan")))

            # Current ring relative uncertainty (%)
            ring_rel_now_pct = (
                (100.0 * ring_sig / ring) if (math.isfinite(ring) and ring > 0 and math.isfinite(ring_sig)) else float("nan")
            )

            # Kerr κ systematic (relative, %) from constrained/full ranges.
            klo = float(r.get("kappa_gr_kerr_coeff_range_low", float("nan")))
            khi = float(r.get("kappa_gr_kerr_coeff_range_high", float("nan")))
            klo_f = float(r.get("kappa_gr_kerr_coeff_range_low_full", float("nan")))
            khi_f = float(r.get("kappa_gr_kerr_coeff_range_high_full", float("nan")))

            kerr_rel_pct = float("nan")
            # 条件分岐: `math.isfinite(klo) and math.isfinite(khi) and khi > klo` を満たす経路を評価する。
            if math.isfinite(klo) and math.isfinite(khi) and khi > klo:
                kmean = 0.5 * (klo + khi)
                ksig = (khi - klo) / math.sqrt(12.0)
                # 条件分岐: `math.isfinite(kmean) and kmean > 0` を満たす経路を評価する。
                if math.isfinite(kmean) and kmean > 0:
                    kerr_rel_pct = 100.0 * (ksig / kmean)

            kerr_rel_pct_full = float("nan")
            # 条件分岐: `math.isfinite(klo_f) and math.isfinite(khi_f) and khi_f > klo_f` を満たす経路を評価する。
            if math.isfinite(klo_f) and math.isfinite(khi_f) and khi_f > klo_f:
                kmean_f = 0.5 * (klo_f + khi_f)
                ksig_f = (khi_f - klo_f) / math.sqrt(12.0)
                # 条件分岐: `math.isfinite(kmean_f) and kmean_f > 0` を満たす経路を評価する。
                if math.isfinite(kmean_f) and kmean_f > 0:
                    kerr_rel_pct_full = 100.0 * (ksig_f / kmean_f)

            # 条件分岐: `math.isfinite(sigma_need) and sigma_need > 0 and math.isfinite(ring) and ring...` を満たす経路を評価する。

            if math.isfinite(sigma_need) and sigma_need > 0 and math.isfinite(ring) and ring > 0:
                ring_sigma_grid = np.linspace(0.0, float(sigma_need), 200)
                ring_rel_grid = 100.0 * (ring_sigma_grid / ring)
                kappa_sig_grid = np.sqrt(np.maximum(float(sigma_need) ** 2 - ring_sigma_grid**2, 0.0)) / ring
                kappa_rel_grid = 100.0 * kappa_sig_grid

                ax.fill_between(ring_rel_grid, 0.0, kappa_rel_grid, color="#1f77b4", alpha=0.12)
                ax.plot(ring_rel_grid, kappa_rel_grid, color="#1f77b4", lw=2.0, label="3σ判別の許容域（下側）")

                for v, c0 in [(1.0, "#2ca02c"), (2.0, "#2ca02c")]:
                    ax.axhline(v, color=c0, linestyle="--", linewidth=1.0, alpha=0.55)
            else:
                msg = "3σ判別: n/a"
                # 条件分岐: `math.isfinite(theta_rel) and math.isfinite(theta_rel_req) and theta_rel_req > 0` を満たす経路を評価する。
                if math.isfinite(theta_rel) and math.isfinite(theta_rel_req) and theta_rel_req > 0:
                    msg += f"（θ_unit相対誤差={theta_rel*100:.1f}% > 要求={theta_rel_req*100:.1f}%）"

                ax.text(0.02, 0.95, msg, transform=ax.transAxes, ha="left", va="top", fontsize=10, color="#444")

            # 条件分岐: `math.isfinite(ring_rel_now_pct)` を満たす経路を評価する。

            if math.isfinite(ring_rel_now_pct):
                ax.axvline(ring_rel_now_pct, color="#ff7f0e", linestyle="--", linewidth=1.0, alpha=0.55, label="現状 ring σ/diameter")
                # 条件分岐: `math.isfinite(kerr_rel_pct)` を満たす経路を評価する。
                if math.isfinite(kerr_rel_pct):
                    ax.plot(ring_rel_now_pct, kerr_rel_pct, "D", color="#111111", alpha=0.9, label="参考: Kerr κ系統（constrained）")

                # 条件分岐: `math.isfinite(kerr_rel_pct_full)` を満たす経路を評価する。

                if math.isfinite(kerr_rel_pct_full):
                    ax.plot(ring_rel_now_pct, kerr_rel_pct_full, "s", color="#7f7f7f", alpha=0.9, label="参考: Kerr κ系統（full）")

            ax.set_title(name)
            ax.set_xlabel("リング直径の相対誤差（1σ, %）")
            ax.grid(True, alpha=0.25)

            x_max = 0.0
            # 条件分岐: `math.isfinite(ring_rel_now_pct)` を満たす経路を評価する。
            if math.isfinite(ring_rel_now_pct):
                x_max = max(x_max, float(ring_rel_now_pct))

            # 条件分岐: `math.isfinite(sigma_need) and sigma_need > 0 and math.isfinite(ring) and ring...` を満たす経路を評価する。

            if math.isfinite(sigma_need) and sigma_need > 0 and math.isfinite(ring) and ring > 0:
                x_max = max(x_max, float(100.0 * sigma_need / ring))

            ax.set_xlim(0.0, max(5.0, 1.15 * x_max))

        axes[0].set_ylabel("許容 κ の相対誤差（1σ, %）")
        fig8.suptitle("EHT：3σ判別のための誤差予算（ringσ と κσ のトレードオフ；κ≈1）", fontsize=13, y=1.03)

        # De-duplicate legend entries across subplots.
        handles, labels = [], []
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            for hh, ll in zip(h, l, strict=False):
                # 条件分岐: `ll not in labels` を満たす経路を評価する。
                if ll not in labels:
                    handles.append(hh)
                    labels.append(ll)

        # 条件分岐: `handles` を満たす経路を評価する。

        if handles:
            fig8.legend(handles, labels, loc="lower center", ncol=min(3, len(labels)), bbox_to_anchor=(0.5, 0.02))

        fig8.tight_layout(rect=[0.0, 0.08, 1.0, 0.95])
        kappa_trade_png_path = out_dir / "eht_kappa_tradeoff.png"
        fig8.savefig(kappa_trade_png_path, dpi=220, bbox_inches="tight")
        kappa_trade_public_png_path = out_dir / "eht_kappa_tradeoff_public.png"
        fig8.savefig(kappa_trade_public_png_path, dpi=220, bbox_inches="tight")
        plt.close(fig8)
    except Exception as e:
        print(f"[warn] kappa tradeoff plot skipped: {e}")
        kappa_trade_png_path = None
        kappa_trade_public_png_path = None

    try:
        worklog.append_event(
            {
                "event_type": "eht_shadow_compare",
                "argv": sys.argv,
                "inputs": {"eht_black_holes": in_path},
                "outputs": {
                    "csv": csv_path,
                    "json": json_path,
                    "png": (png_path if isinstance(png_path, Path) else None),
                    "png_differential": (diff_png_path if isinstance(diff_png_path, Path) else None),
                    "png_differential_public": (
                        diff_public_png_path if isinstance(diff_public_png_path, Path) else None
                    ),
                    "png_systematics": (sys_png_path if isinstance(sys_png_path, Path) else None),
                    "png_systematics_public": (
                        sys_public_png_path if isinstance(sys_public_png_path, Path) else None
                    ),
                    "png_ring_morphology": (ring_png_path if isinstance(ring_png_path, Path) else None),
                    "png_ring_morphology_public": (
                        ring_public_png_path if isinstance(ring_public_png_path, Path) else None
                    ),
                    "png_scattering_budget": (scat_png_path if isinstance(scat_png_path, Path) else None),
                    "png_scattering_budget_public": (
                        scat_public_png_path if isinstance(scat_public_png_path, Path) else None
                    ),
                    "png_refractive_scattering_limits": (
                        refr_png_path if isinstance(refr_png_path, Path) else None
                    ),
                    "png_refractive_scattering_limits_public": (
                        refr_public_png_path if isinstance(refr_public_png_path, Path) else None
                    ),
                    "json_refractive_scattering_limits_metrics": (
                        refr_metrics_json_path if isinstance(refr_metrics_json_path, Path) else None
                    ),
                    "png_zscores": (z_png_path if isinstance(z_png_path, Path) else None),
                    "png_zscores_public": (z_public_png_path if isinstance(z_public_png_path, Path) else None),
                    "png_kappa": (kappa_png_path if isinstance(kappa_png_path, Path) else None),
                    "png_kappa_public": (
                        kappa_public_png_path if isinstance(kappa_public_png_path, Path) else None
                    ),
                    "png_kappa_precision_required": (
                        kappa_req_png_path if isinstance(kappa_req_png_path, Path) else None
                    ),
                    "png_kappa_precision_required_public": (
                        kappa_req_public_png_path if isinstance(kappa_req_public_png_path, Path) else None
                    ),
                    "png_delta_precision_required": (
                        delta_req_png_path if isinstance(delta_req_png_path, Path) else None
                    ),
                    "png_delta_precision_required_public": (
                        delta_req_public_png_path if isinstance(delta_req_public_png_path, Path) else None
                    ),
                    "png_kappa_tradeoff": (
                        kappa_trade_png_path if isinstance(kappa_trade_png_path, Path) else None
                    ),
                    "png_kappa_tradeoff_public": (
                        kappa_trade_public_png_path if isinstance(kappa_trade_public_png_path, Path) else None
                    ),
                },
                "metrics": {
                    "n_objects": len(rows),
                    "beta": beta,
                    "coeff_pmodel": coeff_pmodel,
                    "coeff_pmodel_phase4": coeff_pmodel_phase4,
                    "coeff_gr": coeff_gr,
                    "coeff_ratio_p_over_gr": coeff_ratio_p_over_gr,
                    "coeff_ratio_p_over_gr_baseline": coeff_ratio_p_over_gr_baseline,
                    "coeff_ratio_source": coeff_ratio_source,
                    "coeff_diff_pct_phase4": coeff_diff_pct_phase4,
                    "kerr_coeff_min": (kerr_range_full.get("coeff_min") if isinstance(kerr_range_full, dict) else None),
                    "kerr_coeff_max": (kerr_range_full.get("coeff_max") if isinstance(kerr_range_full, dict) else None),
                },
            }
        )
    except Exception:
        pass

    print(f"[ok] csv : {csv_path}")
    print(f"[ok] json: {json_path}")
    # 条件分岐: `isinstance(png_path, Path)` を満たす経路を評価する。
    if isinstance(png_path, Path):
        print(f"[ok] png : {png_path}")

    # 条件分岐: `isinstance(diff_png_path, Path)` を満たす経路を評価する。

    if isinstance(diff_png_path, Path):
        print(f"[ok] png : {diff_png_path}")

    # 条件分岐: `isinstance(diff_public_png_path, Path)` を満たす経路を評価する。

    if isinstance(diff_public_png_path, Path):
        print(f"[ok] png : {diff_public_png_path}")

    # 条件分岐: `isinstance(sys_png_path, Path)` を満たす経路を評価する。

    if isinstance(sys_png_path, Path):
        print(f"[ok] png : {sys_png_path}")

    # 条件分岐: `isinstance(sys_public_png_path, Path)` を満たす経路を評価する。

    if isinstance(sys_public_png_path, Path):
        print(f"[ok] png : {sys_public_png_path}")

    # 条件分岐: `isinstance(ring_png_path, Path)` を満たす経路を評価する。

    if isinstance(ring_png_path, Path):
        print(f"[ok] png : {ring_png_path}")

    # 条件分岐: `isinstance(ring_public_png_path, Path)` を満たす経路を評価する。

    if isinstance(ring_public_png_path, Path):
        print(f"[ok] png : {ring_public_png_path}")

    # 条件分岐: `isinstance(scat_png_path, Path)` を満たす経路を評価する。

    if isinstance(scat_png_path, Path):
        print(f"[ok] png : {scat_png_path}")

    # 条件分岐: `isinstance(scat_public_png_path, Path)` を満たす経路を評価する。

    if isinstance(scat_public_png_path, Path):
        print(f"[ok] png : {scat_public_png_path}")

    # 条件分岐: `isinstance(refr_png_path, Path)` を満たす経路を評価する。

    if isinstance(refr_png_path, Path):
        print(f"[ok] png : {refr_png_path}")

    # 条件分岐: `isinstance(refr_public_png_path, Path)` を満たす経路を評価する。

    if isinstance(refr_public_png_path, Path):
        print(f"[ok] png : {refr_public_png_path}")

    # 条件分岐: `isinstance(refr_metrics_json_path, Path)` を満たす経路を評価する。

    if isinstance(refr_metrics_json_path, Path):
        print(f"[ok] json: {refr_metrics_json_path}")

    # 条件分岐: `isinstance(z_png_path, Path)` を満たす経路を評価する。

    if isinstance(z_png_path, Path):
        print(f"[ok] png : {z_png_path}")

    # 条件分岐: `isinstance(z_public_png_path, Path)` を満たす経路を評価する。

    if isinstance(z_public_png_path, Path):
        print(f"[ok] png : {z_public_png_path}")

    # 条件分岐: `isinstance(kappa_png_path, Path)` を満たす経路を評価する。

    if isinstance(kappa_png_path, Path):
        print(f"[ok] png : {kappa_png_path}")

    # 条件分岐: `isinstance(kappa_public_png_path, Path)` を満たす経路を評価する。

    if isinstance(kappa_public_png_path, Path):
        print(f"[ok] png : {kappa_public_png_path}")

    # 条件分岐: `isinstance(kappa_req_png_path, Path)` を満たす経路を評価する。

    if isinstance(kappa_req_png_path, Path):
        print(f"[ok] png : {kappa_req_png_path}")

    # 条件分岐: `isinstance(kappa_req_public_png_path, Path)` を満たす経路を評価する。

    if isinstance(kappa_req_public_png_path, Path):
        print(f"[ok] png : {kappa_req_public_png_path}")

    # 条件分岐: `isinstance(delta_req_png_path, Path)` を満たす経路を評価する。

    if isinstance(delta_req_png_path, Path):
        print(f"[ok] png : {delta_req_png_path}")

    # 条件分岐: `isinstance(delta_req_public_png_path, Path)` を満たす経路を評価する。

    if isinstance(delta_req_public_png_path, Path):
        print(f"[ok] png : {delta_req_public_png_path}")

    # 条件分岐: `isinstance(kappa_trade_png_path, Path)` を満たす経路を評価する。

    if isinstance(kappa_trade_png_path, Path):
        print(f"[ok] png : {kappa_trade_png_path}")

    # 条件分岐: `isinstance(kappa_trade_public_png_path, Path)` を満たす経路を評価する。

    if isinstance(kappa_trade_public_png_path, Path):
        print(f"[ok] png : {kappa_trade_public_png_path}")

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
