from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402

_C = 299_792_458.0  # m/s (exact)
_G = 6.67430e-11  # m^3 kg^-1 s^-2 (CODATA 2018)
_M_SUN = 1.98847e30  # kg
_SEC_PER_YEAR = 365.25 * 24.0 * 3600.0


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _chirp_mass_solar(m1_solar: float, m2_solar: float) -> float:
    m1 = float(m1_solar)
    m2 = float(m2_solar)
    if m1 <= 0 or m2 <= 0:
        raise ValueError("m1,m2 must be positive")
    return (m1 * m2) ** (3.0 / 5.0) / (m1 + m2) ** (1.0 / 5.0)


def _pbdot_quadrupole_s_per_s(mc_solar: float, pb_s: float, e: float) -> float:
    if pb_s <= 0:
        raise ValueError("pb_s must be positive")
    if not (0.0 <= e < 1.0):
        raise ValueError("eccentricity e must satisfy 0<=e<1")

    mc_kg = float(mc_solar) * _M_SUN
    x = (2.0 * math.pi * _G * mc_kg) / (_C**3 * float(pb_s))

    # Peters–Mathews (quadrupole) eccentricity factor
    e2 = float(e) ** 2
    fe = (1.0 + (73.0 / 24.0) * e2 + (37.0 / 96.0) * (e2**2)) / (1.0 - e2) ** (7.0 / 2.0)

    return -(192.0 * math.pi / 5.0) * (x ** (5.0 / 3.0)) * fe


def _ttc_quadrupole_s(mc_solar: float, f_hz: float) -> float:
    if f_hz <= 0:
        raise ValueError("f_hz must be positive")
    mc_kg = float(mc_solar) * _M_SUN
    tau = (_G * mc_kg) / (_C**3)
    return (5.0 / 256.0) * (tau ** (-5.0 / 3.0)) * ((math.pi * float(f_hz)) ** (-8.0 / 3.0))


def _render_plot(out_png: Path, *, zmax_pb_hours: float, zmax_f_hz: float) -> Dict[str, Any]:
    _set_japanese_font()

    # Panel A: orbital decay scaling vs orbital period
    mc_ns = _chirp_mass_solar(1.35, 1.35)  # typical NS-NS
    pb_hours = np.logspace(math.log10(0.3), math.log10(zmax_pb_hours), 240)
    pb_s = pb_hours * 3600.0
    pbdot_circ = np.array([_pbdot_quadrupole_s_per_s(mc_ns, float(p), 0.0) for p in pb_s])
    pbdot_ecc = np.array([_pbdot_quadrupole_s_per_s(mc_ns, float(p), 0.6) for p in pb_s])

    y_circ_usyr = np.abs(pbdot_circ) * _SEC_PER_YEAR * 1e6
    y_ecc_usyr = np.abs(pbdot_ecc) * _SEC_PER_YEAR * 1e6

    # Panel B: chirp time-to-coalescence scaling vs frequency
    mc_bh = _chirp_mass_solar(36.0, 29.0)  # GW150914-like
    f = np.logspace(math.log10(10.0), math.log10(zmax_f_hz), 260)
    t_ns = np.array([_ttc_quadrupole_s(mc_ns, float(ff)) for ff in f])
    t_bh = np.array([_ttc_quadrupole_s(mc_bh, float(ff)) for ff in f])

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.6, 4.8), dpi=160)

    ax0.plot(pb_hours, y_circ_usyr, lw=2.0, label=f"円軌道（e=0）, M_c≈{mc_ns:.3f} M☉")
    ax0.plot(pb_hours, y_ecc_usyr, lw=2.0, label="離心（e=0.6）")
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlabel("軌道周期 P_b [hours]")
    ax0.set_ylabel("|dP_b/dt| [μs/year]")
    ax0.set_title("二重パルサー：軌道減衰のスケーリング（四重極）")
    ax0.grid(True, which="both", alpha=0.22)
    ax0.legend(loc="lower left", fontsize=9)

    ax1.plot(f, t_ns / 60.0, lw=2.0, label=f"NS-NS（M_c≈{mc_ns:.3f} M☉）")
    ax1.plot(f, t_bh / 60.0, lw=2.0, label=f"BH-BH（M_c≈{mc_bh:.2f} M☉）")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("周波数 f [Hz]")
    ax1.set_ylabel("合体までの残り時間 t_c - t [minutes]")
    ax1.set_title("重力波：chirp のスケーリング（四重極）")
    ax1.grid(True, which="both", alpha=0.22)
    ax1.legend(loc="upper right", fontsize=9)

    fig.suptitle("動的P（放射・波動）：弱場遠方で採用する四重極則（概念スケール）", y=1.02)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    example_pb_s = 8.0 * 3600.0
    example = {
        "chirp_mass_nsns_solar": mc_ns,
        "chirp_mass_bhbh_solar": mc_bh,
        "orbital_decay": {
            "pb_hours": 8.0,
            "mc_solar": mc_ns,
            "pbdot_circular_s_per_s": _pbdot_quadrupole_s_per_s(mc_ns, example_pb_s, 0.0),
            "pbdot_e0p6_s_per_s": _pbdot_quadrupole_s_per_s(mc_ns, example_pb_s, 0.6),
            "pbdot_circular_us_per_year": _pbdot_quadrupole_s_per_s(mc_ns, example_pb_s, 0.0) * _SEC_PER_YEAR * 1e6,
            "pbdot_e0p6_us_per_year": _pbdot_quadrupole_s_per_s(mc_ns, example_pb_s, 0.6) * _SEC_PER_YEAR * 1e6,
        },
        "chirp": {
            "f_hz": 30.0,
            "mc_ns_solar": mc_ns,
            "mc_bh_solar": mc_bh,
            "ttc_ns_s": _ttc_quadrupole_s(mc_ns, 30.0),
            "ttc_bh_s": _ttc_quadrupole_s(mc_bh, 30.0),
        },
    }

    return example


def main(argv: list[str] | None = None) -> int:
    root = _ROOT
    default_png = root / "output" / "private" / "theory" / "dynamic_p_quadrupole_scalings.png"
    default_json = root / "output" / "private" / "theory" / "dynamic_p_quadrupole_scalings_metrics.json"

    ap = argparse.ArgumentParser(description="Dynamic P: quadrupole scaling plots (theory-only, no fit).")
    ap.add_argument("--out-png", type=str, default=str(default_png))
    ap.add_argument("--out-json", type=str, default=str(default_json))
    ap.add_argument("--pb-hours-max", type=float, default=120.0)
    ap.add_argument("--f-hz-max", type=float, default=1000.0)
    args = ap.parse_args(argv)

    out_png = Path(args.out_png)
    out_json = Path(args.out_json)

    example = _render_plot(out_png, zmax_pb_hours=float(args.pb_hours_max), zmax_f_hz=float(args.f_hz_max))

    payload: Dict[str, Any] = {
        "generated_utc": _iso_utc_now(),
        "purpose": "動的P（放射・波動）の最小仮定として採用する四重極則のスケーリングを可視化（データフィットはしない）。",
        "constants": {"c_m_per_s": _C, "G_m3_kg_s2": _G, "M_sun_kg": _M_SUN, "sec_per_year": _SEC_PER_YEAR},
        "formulas": {
            "pbdot_quadrupole": "Pdot_b = -(192π/5) * (2π G M_c /(c^3 P_b))^(5/3) * F(e),  F(e)=(1+73/24 e^2+37/96 e^4)/(1-e^2)^(7/2)",
            "chirp_time": "t_c - t = (5/256) * (G M_c / c^3)^(-5/3) * (π f)^(-8/3)",
        },
        "example_values": example,
        "outputs": {"png": str(out_png), "metrics_json": str(out_json)},
    }
    _write_json(out_json, payload)

    try:
        worklog.append_event(
            {
                "event_type": "theory_dynamic_p_quadrupole_scalings",
                "argv": list(sys.argv),
                "outputs": {"png": out_png, "metrics_json": out_json},
                "metrics": {
                    "mc_ns_solar": example["chirp_mass_nsns_solar"],
                    "mc_bh_solar": example["chirp_mass_bhbh_solar"],
                },
            }
        )
    except Exception:
        pass

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
