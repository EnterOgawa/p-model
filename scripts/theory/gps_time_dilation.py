from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt


# Physical constants
C = 299_792_458.0  # m/s
MU_E = 3.986004418e14  # m^3/s^2


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
        if not chosen:
            return

        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def grav_rate(mu: float, r_m: float) -> float:
    # P-model gravitational factor used in docs: exp(-mu/(c^2 r))
    return math.exp(-mu / (C * C * r_m))


def vel_rate(v_m_s: float, delta: float) -> float:
    # Velocity factor:
    #   core: (dτ/dt)_v = sqrt(1 - v^2/c^2)
    #   optional saturation (extension): sqrt((1 - v^2/c^2 + δ0)/(1+δ0))
    return math.sqrt((1.0 - (v_m_s * v_m_s) / (C * C) + delta) / (1.0 + delta))


def circular_orbit_speed(mu: float, r_m: float) -> float:
    return math.sqrt(mu / r_m)


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    default_outdir = root / "output" / "private" / "theory"

    ap = argparse.ArgumentParser(description="GPS time dilation breakdown (P-model, weak field).")
    ap.add_argument("--delta", type=float, default=0.0, help="Clock saturation δ0 (extension; default: 0.0=disabled)")
    ap.add_argument("--r-ground-m", type=float, default=6_378_137.0, help="Ground radius from Earth's center [m]")
    ap.add_argument("--r-sat-m", type=float, default=26_560_000.0, help="Satellite orbit radius [m] (GPS ~ 26,560 km)")
    ap.add_argument("--seconds", type=float, default=86_400.0, help="Integration interval [s] (default: 1 day)")
    ap.add_argument(
        "--outdir",
        type=str,
        default=str(default_outdir),
        help="Output directory (default: output/private/theory)",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    delta = float(args.delta)
    r_ground = float(args.r_ground_m)
    r_sat = float(args.r_sat_m)
    seconds = float(args.seconds)

    # Simple circular orbit speed; ground is treated as v=0 for the canonical textbook breakdown.
    v_sat = circular_orbit_speed(MU_E, r_sat)
    v_ground = 0.0

    g_ground = grav_rate(MU_E, r_ground)
    g_sat = grav_rate(MU_E, r_sat)
    vfac_ground = vel_rate(v_ground, delta)
    vfac_sat = vel_rate(v_sat, delta)

    rate_ground = g_ground * vfac_ground
    rate_sat = g_sat * vfac_sat

    # Exact (within this model) time offsets over the interval.
    dt_grav_s = (g_sat - g_ground) * seconds
    dt_vel_s = (vfac_sat - vfac_ground) * seconds
    dt_total_s = (rate_sat - rate_ground) * seconds

    # Weak-field approximations (useful for cross-checking / known GPS numbers).
    dt_grav_approx_s = (MU_E / (C * C)) * (1.0 / r_ground - 1.0 / r_sat) * seconds
    dt_sr_approx_s = -0.5 * (v_sat * v_sat) / (C * C) * seconds
    dt_net_approx_s = dt_grav_approx_s + dt_sr_approx_s

    # Canonical textbook values often cited for GPS (per day).
    ref_grav_us_day = 45.7
    ref_sr_us_day = -7.2
    ref_net_us_day = 38.5

    metrics: Dict[str, float] = {
        "delta": delta,
        "r_ground_m": r_ground,
        "r_sat_m": r_sat,
        "v_sat_m_s": v_sat,
        "interval_s": seconds,
        "grav_us": dt_grav_s * 1e6,
        "vel_us": dt_vel_s * 1e6,
        "net_us": dt_total_s * 1e6,
        "grav_approx_us": dt_grav_approx_s * 1e6,
        "sr_approx_us": dt_sr_approx_s * 1e6,
        "net_approx_us": dt_net_approx_s * 1e6,
        "ref_grav_us_day": ref_grav_us_day * (seconds / 86_400.0),
        "ref_sr_us_day": ref_sr_us_day * (seconds / 86_400.0),
        "ref_net_us_day": ref_net_us_day * (seconds / 86_400.0),
    }
    metrics["abs_error_net_us_vs_ref"] = abs(metrics["net_approx_us"] - metrics["ref_net_us_day"])

    # Plot (bar chart)
    _set_japanese_font()

    labels = ["重力", "速度", "合計"]
    values = [metrics["grav_approx_us"], metrics["sr_approx_us"], metrics["net_approx_us"]]
    refs = [metrics["ref_grav_us_day"], metrics["ref_sr_us_day"], metrics["ref_net_us_day"]]

    x = range(len(labels))
    plt.figure(figsize=(8.5, 5.0))
    plt.bar([i - 0.18 for i in x], values, width=0.36, label="P-model（近似）")
    plt.bar([i + 0.18 for i in x], refs, width=0.36, label="教科書値（代表）")
    plt.axhline(0.0, linewidth=1)
    plt.xticks(list(x), labels)
    plt.ylabel("時間差 [マイクロ秒]")
    plt.title("GPSの時間補正（区間あたり）")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    png_path = outdir / "gps_time_dilation.png"
    plt.savefig(png_path, dpi=200)
    plt.close()

    out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model": "P-model",
        "notes": [
            "教科書的な内訳（重力 +45.7us/日、速度 -7.2us/日）に合わせ、地上の速度は0として計算しています。",
            "本スクリプトの“exact”は exp(-mu/(c^2 r)) と、delta付きの速度因子（飽和）を用います。",
        ],
        "constants": {"C_m_s": C, "MU_E_m3_s2": MU_E},
        "metrics": metrics,
        "outputs": {"plot_png": str(png_path)},
    }
    json_path = outdir / "gps_time_dilation_metrics.json"
    json_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] plot : {png_path}")
    print(f"[ok] metrics: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
