from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Config:
    # --- Reference Earth parameters (SI) ---
    c_m_per_s: float = 299_792_458.0
    gm_earth_m3_s2: float = 3.986_004_418e14  # WGS84
    r_earth_m: float = 6_378_137.0  # WGS84 equatorial radius
    g0_m_per_s2: float = 9.80665

    gm_sun_m3_s2: float = 1.327_124_400_18e20  # IAU
    au_m: float = 149_597_870_700.0  # exact

    # --- Atomic interferometer (Mueller 2007; Cs Raman gravimeter) ---
    m_u_kg: float = 1.660_539_066_60e-27  # atomic mass constant
    cs133_mass_u: float = 132.905_451_96  # standard atomic weight (sufficient here)
    omega_earth_rad_per_s: float = 7.292_115_9e-5
    atom_gyro_transverse_velocity_m_per_s: float = 5.0
    atom_gyro_phase_fractional_precision: float = 1e-6

    # --- Current precision assumptions (order; to avoid claiming dataset-specific numbers) ---
    cow_phase_fractional_precision: float = 1e-2
    atom_interferometer_phase_fractional_precision: float = 1e-9
    aces_fractional_precision_goal: float = 1e-16  # mission-scale order; not a fixed spec here

    # Strong-field illustrative case (not a lab experiment):
    neutron_star_mass_msun: float = 1.4
    neutron_star_radius_m: float = 12_000.0

    # Plot settings
    fig_w_in: float = 12.5
    fig_h_in: float = 6.2
    dpi: int = 175


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _relpath(root: Path, p: Path) -> str:
    try:
        return str(p.relative_to(root)).replace("\\", "/")
    except Exception:
        return str(p).replace("\\", "/")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _x(*, gm_m3_s2: float, r_m: float, c_m_per_s: float) -> float:
    # x ≡ GM/(c^2 r)
    return float(gm_m3_s2 / ((c_m_per_s**2) * r_m))


def _clock_rate_pmodel(*, x_dimless: float) -> float:
    # P-model stationary clock map for point mass:
    #   ln(P/P0) = x  => P0/P = exp(-x)
    return float(math.exp(-x_dimless))


def _clock_rate_gr_schwarzschild(*, x_dimless: float) -> float:
    # GR stationary clock map (Schwarzschild):
    #   dτ/dt = sqrt(1 - 2x)
    if x_dimless >= 0.5:
        return float("nan")

    return float(math.sqrt(1.0 - 2.0 * x_dimless))


def _redshift_between_radii(
    *, gm_m3_s2: float, r_low_m: float, r_high_m: float, cfg: Config, body_label: str = ""
) -> Dict[str, float]:
    x_low = _x(gm_m3_s2=gm_m3_s2, r_m=r_low_m, c_m_per_s=cfg.c_m_per_s)
    x_high = _x(gm_m3_s2=gm_m3_s2, r_m=r_high_m, c_m_per_s=cfg.c_m_per_s)

    # Use numerically stable forms (expm1/log1p) because many of these z are << machine epsilon.
    # P-model: ratio = exp(-(x_high-x_low)) = exp(delta_x), z = expm1(delta_x).
    delta_x = float(x_low - x_high)
    z_p = float(math.expm1(delta_x))
    ratio_p = float(1.0 + z_p)

    # GR: ratio = sqrt((1-2x_high)/(1-2x_low)).
    # Use log ratio to preserve small differences:
    #   ln ratio = 0.5 [ ln(1-2x_high) - ln(1-2x_low) ].
    a_high = float(-2.0 * x_high)
    a_low = float(-2.0 * x_low)
    ln_ratio_gr = 0.5 * (math.log1p(a_high) - math.log1p(a_low))
    z_gr = float(math.expm1(ln_ratio_gr))
    ratio_gr = float(1.0 + z_gr)

    dz_raw = float(z_p - z_gr)
    # Difference scale (P-model - GR):
    # - In weak field (x≪1), exp vs sqrt differs at O(x^2); for small redshift z:
    #     δz/z ≈ -(x_low + x_high)  (sign: P-model smaller than GR for the same Δr).
    #   This avoids precision loss when δz is ~1e-27 (COW-scale height differences).
    # - In stronger fields, fall back to the exact (raw) difference.
    use_weak_series = (max(abs(x_low), abs(x_high)) < 1e-4) and (abs(z_gr) < 1e-6)
    # 条件分岐: `use_weak_series and z_gr != 0.0` を満たす経路を評価する。
    if use_weak_series and z_gr != 0.0:
        rel = float(-(x_low + x_high))
        dz = float(rel * z_gr)
    else:
        dz = dz_raw
        rel = float(dz / z_gr) if z_gr != 0.0 else float("nan")

    return {
        "body": str(body_label),
        "x_low": x_low,
        "x_high": x_high,
        "ratio_p": float(ratio_p),
        "ratio_gr": float(ratio_gr),
        "z_p": float(z_p),
        "z_gr": float(z_gr),
        "delta_z": float(dz),
        "delta_z_raw": float(dz_raw),
        "delta_z_over_z_gr": float(rel),
    }


def _required_precision_for_3sigma(*, delta_abs: float) -> Optional[float]:
    # 条件分岐: `not math.isfinite(delta_abs) or delta_abs <= 0` を満たす経路を評価する。
    if not math.isfinite(delta_abs) or delta_abs <= 0:
        return None

    return float(delta_abs / 3.0)


def _atom_gyro_phase_rad(*, keff_1_per_m: float, v_transverse_m_per_s: float, omega_rad_per_s: float, T_s: float) -> float:
    # Light-pulse atom gyroscope proxy (Sagnac-like term):
    #   φ_Ω ≈ 2 k_eff v_trans Ω T²
    return float(2.0 * keff_1_per_m * v_transverse_m_per_s * omega_rad_per_s * (T_s**2))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    # 条件分岐: `not rows` を満たす経路を評価する。
    if not rows:
        return

    fields = [
        "channel",
        "observable",
        "phase_ref_rad",
        "characteristic_scale_m",
        "delta_z",
        "delta_z_over_z_gr",
        "delta_observable",
        "required_precision_3sigma",
        "current_precision",
        "detectable_under_current",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config()

    # Load baseline observables (fixed elsewhere in the repo).
    cow_m = _read_json(root / "output" / "public" / "quantum" / "cow_phase_shift_metrics.json")
    atom_m = _read_json(root / "output" / "public" / "quantum" / "atom_interferometer_gravimeter_phase_metrics.json")
    clock_m = _read_json(root / "output" / "public" / "quantum" / "optical_clock_chronometric_leveling_metrics.json")

    cow_phase_rad = abs(float(((cow_m.get("results") or {}) if isinstance(cow_m.get("results"), dict) else {}).get("phi0_rad")))
    cow_h_m = float(((cow_m.get("config") or {}) if isinstance(cow_m.get("config"), dict) else {}).get("H_m"))

    atom_phi_rad = abs(float(((atom_m.get("results") or {}) if isinstance(atom_m.get("results"), dict) else {}).get("phi_ref_rad")))
    atom_T_s = float(((atom_m.get("config") or {}) if isinstance(atom_m.get("config"), dict) else {}).get("T_s"))
    atom_keff = float(((atom_m.get("derived") or {}) if isinstance(atom_m.get("derived"), dict) else {}).get("k_eff_1_per_m"))
    gyro_phi_rad = abs(
        _atom_gyro_phase_rad(
            keff_1_per_m=atom_keff,
            v_transverse_m_per_s=cfg.atom_gyro_transverse_velocity_m_per_s,
            omega_rad_per_s=cfg.omega_earth_rad_per_s,
            T_s=atom_T_s,
        )
    )

    # Effective arm separation scale for a Raman AI (order): Δz ~ v_rec T, with v_rec = ħ k_eff / m.
    hbar = 1.054_571_817e-34
    m_cs_kg = float(cfg.cs133_mass_u * cfg.m_u_kg)
    v_rec_m_per_s = float(hbar * atom_keff / m_cs_kg)
    atom_dz_m = float(v_rec_m_per_s * atom_T_s)
    gyro_dz_m = atom_dz_m

    # Optical clock: use the ΔU values already fixed in the repo.
    src_abs = (clock_m.get("source") or {}) if isinstance(clock_m.get("source"), dict) else {}
    abs_vals = (src_abs.get("abstract_values") or {}) if isinstance(src_abs.get("abstract_values"), dict) else {}
    du_clock = float(abs_vals.get("delta_u_clock_m2_s2"))
    sigma_clock = float(abs_vals.get("sigma_clock_m2_s2"))
    du_geo = float(abs_vals.get("delta_u_geodetic_m2_s2"))
    sigma_geo = float(abs_vals.get("sigma_geodetic_m2_s2"))
    z_clock_sigma = float(sigma_clock / (cfg.c_m_per_s**2))
    z_geo = float(du_geo / (cfg.c_m_per_s**2))
    z_geo_sigma = float(sigma_geo / (cfg.c_m_per_s**2))
    clock_h_m = float(du_geo / cfg.g0_m_per_s2)

    # Earth near-surface reference: r_low = R_earth.
    r0 = cfg.r_earth_m

    # Compute redshift deltas (Earth field only; stationary).
    cow_redshift = _redshift_between_radii(
        gm_m3_s2=cfg.gm_earth_m3_s2, r_low_m=r0, r_high_m=r0 + cow_h_m, cfg=cfg, body_label="Earth"
    )
    atom_redshift = _redshift_between_radii(
        gm_m3_s2=cfg.gm_earth_m3_s2, r_low_m=r0, r_high_m=r0 + atom_dz_m, cfg=cfg, body_label="Earth"
    )
    gyro_redshift = _redshift_between_radii(
        gm_m3_s2=cfg.gm_earth_m3_s2, r_low_m=r0, r_high_m=r0 + gyro_dz_m, cfg=cfg, body_label="Earth"
    )
    clock_redshift = _redshift_between_radii(
        gm_m3_s2=cfg.gm_earth_m3_s2, r_low_m=r0, r_high_m=r0 + clock_h_m, cfg=cfg, body_label="Earth"
    )

    def _phase_delta_from_rel(phi_rad: float, rel: float) -> float:
        return float(phi_rad * rel)

    cow_rel = float(cow_redshift["delta_z_over_z_gr"])
    atom_rel = float(atom_redshift["delta_z_over_z_gr"])
    gyro_rel = float(gyro_redshift["delta_z_over_z_gr"])
    cow_delta_phi = _phase_delta_from_rel(cow_phase_rad, cow_rel)
    atom_delta_phi = _phase_delta_from_rel(atom_phi_rad, atom_rel)
    gyro_delta_phi = _phase_delta_from_rel(gyro_phi_rad, gyro_rel)

    # Required (3σ) fractional precision for detecting the model difference in a z-like observable.
    cow_req_frac = _required_precision_for_3sigma(delta_abs=abs(cow_rel))
    atom_req_frac = _required_precision_for_3sigma(delta_abs=abs(atom_rel))
    gyro_req_frac = _required_precision_for_3sigma(delta_abs=abs(gyro_rel))
    clock_req_abs = _required_precision_for_3sigma(delta_abs=abs(float(clock_redshift["delta_z"])))

    # Simple mission-scale comparators (not claimed as fixed specs).
    # ACES-like: ground ↔ ISS (~400 km).
    iss_h_m = 400_000.0
    aces_redshift = _redshift_between_radii(
        gm_m3_s2=cfg.gm_earth_m3_s2, r_low_m=r0, r_high_m=r0 + iss_h_m, cfg=cfg, body_label="Earth"
    )
    aces_req_abs = _required_precision_for_3sigma(delta_abs=abs(float(aces_redshift["delta_z"])))

    # "Where it could become observable": Earth ↔ high orbit.
    # GPS altitude ~20,200 km above surface; GEO altitude ~35,786 km.
    gps_h_m = 20_200_000.0
    geo_h_m = 35_786_000.0
    gps_redshift = _redshift_between_radii(
        gm_m3_s2=cfg.gm_earth_m3_s2, r_low_m=r0, r_high_m=r0 + gps_h_m, cfg=cfg, body_label="Earth"
    )
    geo_redshift = _redshift_between_radii(
        gm_m3_s2=cfg.gm_earth_m3_s2, r_low_m=r0, r_high_m=r0 + geo_h_m, cfg=cfg, body_label="Earth"
    )

    # Solar-potential lever arm: 1 AU ↔ 0.3 AU / 0.05 AU (illustrative deep-space clock baseline).
    sun_r_1au = cfg.au_m
    sun_r_0p3au = 0.3 * cfg.au_m
    sun_r_0p05au = 0.05 * cfg.au_m
    sun_0p3_to_1au = _redshift_between_radii(
        gm_m3_s2=cfg.gm_sun_m3_s2,
        r_low_m=sun_r_0p3au,
        r_high_m=sun_r_1au,
        cfg=cfg,
        body_label="Sun",
    )
    sun_0p05_to_1au = _redshift_between_radii(
        gm_m3_s2=cfg.gm_sun_m3_s2,
        r_low_m=sun_r_0p05au,
        r_high_m=sun_r_1au,
        cfg=cfg,
        body_label="Sun",
    )
    sun_0p3_req_abs = _required_precision_for_3sigma(delta_abs=abs(float(sun_0p3_to_1au["delta_z"])))
    sun_0p05_req_abs = _required_precision_for_3sigma(delta_abs=abs(float(sun_0p05_to_1au["delta_z"])))

    # Neutron star (surface → infinity): strong-field illustration.
    gm_ns = float(cfg.neutron_star_mass_msun * cfg.gm_sun_m3_s2)
    r_ns = float(cfg.neutron_star_radius_m)
    ns_surface_to_infty = _redshift_between_radii(
        gm_m3_s2=gm_ns,
        r_low_m=r_ns,
        r_high_m=1e20,  # effectively infinity
        cfg=cfg,
        body_label="NeutronStar",
    )
    ns_req_abs = _required_precision_for_3sigma(delta_abs=abs(float(ns_surface_to_infty["delta_z"])))

    # Plot: required precision (3σ) vs assumed current precision for each lab-scale case.
    labels = [
        "COW (phase frac)",
        "Atom gravimeter (phase frac)",
        "Atom gyroscope (phase frac)",
        "Clock (Δf/f abs)",
        "ACES-like (Δf/f abs)",
    ]
    req = [
        float(cow_req_frac or float("nan")),
        float(atom_req_frac or float("nan")),
        float(gyro_req_frac or float("nan")),
        float(clock_req_abs or float("nan")),
        float(aces_req_abs or float("nan")),
    ]
    cur = [
        cfg.cow_phase_fractional_precision,
        cfg.atom_interferometer_phase_fractional_precision,
        cfg.atom_gyro_phase_fractional_precision,
        z_clock_sigma,  # treat as achieved absolute σ(Δf/f) for the leveling result
        cfg.aces_fractional_precision_goal,
    ]

    x = np.arange(len(labels), dtype=float)
    w = 0.38
    fig, ax = plt.subplots(figsize=(cfg.fig_w_in, cfg.fig_h_in), dpi=cfg.dpi)
    ax.bar(x - w / 2, cur, width=w, label="current/assumed precision", color="#2b6cb0", alpha=0.85)
    ax.bar(x + w / 2, req, width=w, label="required (3σ) to see Δ", color="#c53030", alpha=0.75)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12, ha="right")
    ax.set_ylabel("precision (fractional or absolute Δf/f)")
    ax.set_title("Phase 7 / Step 7.16.12: atom-interferometer unified audit (P-model vs GR; Earth field)")
    ax.grid(True, axis="y", ls=":", lw=0.7, alpha=0.6)
    ax.legend(loc="upper left", fontsize=9, frameon=True)

    note = (
        "P-model: dτ/dt = exp(-x), GR: dτ/dt = sqrt(1-2x), x=GM/(c^2 r).  "
        "We compare Δf/f between r and r+Δr in the Earth field (stationary)."
    )
    fig.text(0.01, -0.02, note, fontsize=9)
    fig.tight_layout()
    out_png = out_dir / "gravity_quantum_interference_delta_predictions.png"
    out_png_unified = out_dir / "atom_interferometer_unified_audit.png"
    fig.savefig(out_png, bbox_inches="tight")
    fig.savefig(out_png_unified, bbox_inches="tight")
    plt.close(fig)

    summary_rows: list[dict[str, Any]] = [
        {
            "channel": "cow_neutron",
            "observable": "phase_fractional",
            "phase_ref_rad": cow_phase_rad,
            "characteristic_scale_m": cow_h_m,
            "delta_z": float(cow_redshift["delta_z"]),
            "delta_z_over_z_gr": cow_rel,
            "delta_observable": cow_delta_phi,
            "required_precision_3sigma": cow_req_frac,
            "current_precision": cfg.cow_phase_fractional_precision,
            "detectable_under_current": bool(
                cow_req_frac is not None and cfg.cow_phase_fractional_precision <= cow_req_frac
            ),
        },
        {
            "channel": "atom_gravimeter",
            "observable": "phase_fractional",
            "phase_ref_rad": atom_phi_rad,
            "characteristic_scale_m": atom_dz_m,
            "delta_z": float(atom_redshift["delta_z"]),
            "delta_z_over_z_gr": atom_rel,
            "delta_observable": atom_delta_phi,
            "required_precision_3sigma": atom_req_frac,
            "current_precision": cfg.atom_interferometer_phase_fractional_precision,
            "detectable_under_current": bool(
                atom_req_frac is not None and cfg.atom_interferometer_phase_fractional_precision <= atom_req_frac
            ),
        },
        {
            "channel": "atom_gyroscope_proxy",
            "observable": "phase_fractional",
            "phase_ref_rad": gyro_phi_rad,
            "characteristic_scale_m": gyro_dz_m,
            "delta_z": float(gyro_redshift["delta_z"]),
            "delta_z_over_z_gr": gyro_rel,
            "delta_observable": gyro_delta_phi,
            "required_precision_3sigma": gyro_req_frac,
            "current_precision": cfg.atom_gyro_phase_fractional_precision,
            "detectable_under_current": bool(
                gyro_req_frac is not None and cfg.atom_gyro_phase_fractional_precision <= gyro_req_frac
            ),
        },
        {
            "channel": "optical_clock_leveling",
            "observable": "delta_f_over_f_abs",
            "phase_ref_rad": None,
            "characteristic_scale_m": clock_h_m,
            "delta_z": float(clock_redshift["delta_z"]),
            "delta_z_over_z_gr": float(clock_redshift["delta_z_over_z_gr"]),
            "delta_observable": float(clock_redshift["delta_z"]),
            "required_precision_3sigma": clock_req_abs,
            "current_precision": z_clock_sigma,
            "detectable_under_current": bool(clock_req_abs is not None and z_clock_sigma <= clock_req_abs),
        },
    ]

    summary_csv = out_dir / "atom_interferometer_unified_audit_summary.csv"
    _write_csv(summary_csv, summary_rows)

    detectable_n = sum(1 for row in summary_rows if bool(row.get("detectable_under_current")))
    summary_metrics = {
        "generated_utc": _iso_utc_now(),
        "phase": 7,
        "step": "7.16.12",
        "title": "Atom-interferometer unified audit (gravimeter/gyroscope/clock)",
        "channels_n": len(summary_rows),
        "detectable_under_current_n": int(detectable_n),
        "rows": summary_rows,
        "outputs": {
            "summary_csv": str(summary_csv),
            "summary_png": str(out_png_unified),
            "legacy_png": str(out_png),
        },
    }
    summary_json = out_dir / "atom_interferometer_unified_audit_metrics.json"
    summary_json.write_text(json.dumps(summary_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    out = {
        "generated_utc": _iso_utc_now(),
        "phase": 7,
        "step": "7.16.12",
        "title": "P-model vs GR post-Newtonian difference scale (gravity × quantum interference; unified atom audit)",
        "definitions": {
            "x": "x ≡ GM/(c^2 r)",
            "pmodel_stationary_clock": "dτ/dt = P0/P = exp(-x) (for point mass; ln(P/P0)=x)",
            "gr_stationary_clock": "dτ/dt = sqrt(1-2x) (Schwarzschild)",
            "note": "This file quantifies the *difference* between the two mappings at weak field and relates it to phase/clock observables as a detectability estimate.",
        },
        "constants": {
            "c_m_per_s": cfg.c_m_per_s,
            "gm_earth_m3_s2": cfg.gm_earth_m3_s2,
            "r_earth_m": cfg.r_earth_m,
            "g0_m_per_s2": cfg.g0_m_per_s2,
            "gm_sun_m3_s2": cfg.gm_sun_m3_s2,
            "au_m": cfg.au_m,
            "hbar_J_s": hbar,
            "m_cs133_kg": m_cs_kg,
            "neutron_star_mass_msun": cfg.neutron_star_mass_msun,
            "neutron_star_radius_m": cfg.neutron_star_radius_m,
        },
        "baselines": {
            "cow": {
                "phase_ref_rad": cow_phase_rad,
                "height_scale_m": cow_h_m,
                "source_metrics": "output/public/quantum/cow_phase_shift_metrics.json",
                "current_phase_fractional_precision_assumed": cfg.cow_phase_fractional_precision,
            },
            "atom_interferometer": {
                "phi_ref_rad": atom_phi_rad,
                "T_s": atom_T_s,
                "k_eff_1_per_m": atom_keff,
                "v_rec_m_per_s": v_rec_m_per_s,
                "arm_separation_scale_m": atom_dz_m,
                "source_metrics": "output/public/quantum/atom_interferometer_gravimeter_phase_metrics.json",
                "current_phase_fractional_precision_assumed": cfg.atom_interferometer_phase_fractional_precision,
                "note": "arm separation is estimated as Δz~v_rec T (order).",
            },
            "atom_gyroscope_proxy": {
                "phi_ref_rad": gyro_phi_rad,
                "T_s": atom_T_s,
                "k_eff_1_per_m": atom_keff,
                "omega_earth_rad_per_s": cfg.omega_earth_rad_per_s,
                "v_transverse_m_per_s": cfg.atom_gyro_transverse_velocity_m_per_s,
                "arm_separation_scale_m": gyro_dz_m,
                "formula": "φ_Ω ≈ 2 k_eff v_trans Ω T²",
                "current_phase_fractional_precision_assumed": cfg.atom_gyro_phase_fractional_precision,
                "note": "Proxy channel for unified systematics audit (no single-dataset fit in this step).",
            },
            "optical_clock_leveling": {
                "delta_u_geodetic_m2_s2": du_geo,
                "sigma_geodetic_m2_s2": sigma_geo,
                "delta_u_clock_m2_s2": du_clock,
                "sigma_clock_m2_s2": sigma_clock,
                "height_scale_m": clock_h_m,
                "z_geodetic_delta_f_over_f": z_geo,
                "sigma_z_clock_abs": z_clock_sigma,
                "source_metrics": "output/public/quantum/optical_clock_chronometric_leveling_metrics.json",
            },
        },
        "comparisons_earth_field": {
            "cow": {
                "redshift": cow_redshift,
                "delta_phase_rad_est": cow_delta_phi,
                "required_fractional_precision_for_3sigma": cow_req_frac,
                "detectable_under_assumed_precision": bool(
                    cow_req_frac is not None and cfg.cow_phase_fractional_precision <= cow_req_frac
                ),
            },
            "atom_interferometer": {
                "redshift": atom_redshift,
                "delta_phase_rad_est": atom_delta_phi,
                "required_fractional_precision_for_3sigma": atom_req_frac,
                "detectable_under_assumed_precision": bool(
                    atom_req_frac is not None and cfg.atom_interferometer_phase_fractional_precision <= atom_req_frac
                ),
            },
            "atom_gyroscope_proxy": {
                "redshift": gyro_redshift,
                "delta_phase_rad_est": gyro_delta_phi,
                "required_fractional_precision_for_3sigma": gyro_req_frac,
                "detectable_under_assumed_precision": bool(
                    gyro_req_frac is not None and cfg.atom_gyro_phase_fractional_precision <= gyro_req_frac
                ),
            },
            "optical_clock_leveling": {
                "redshift": clock_redshift,
                "required_abs_precision_for_3sigma_delta_f_over_f": clock_req_abs,
                "detectable_under_clock_sigma": bool(clock_req_abs is not None and z_clock_sigma <= clock_req_abs),
            },
        },
        "future_regimes_examples": {
            "iss_400km": {
                "r_high_m": r0 + iss_h_m,
                "redshift": aces_redshift,
                "required_abs_precision_for_3sigma_delta_f_over_f": aces_req_abs,
                "note": "ACES-like comparator. The 'current precision' here is a rough mission-scale order (not a fixed spec).",
            },
            "gps_20200km": {"r_high_m": r0 + gps_h_m, "redshift": gps_redshift},
            "geo_35786km": {"r_high_m": r0 + geo_h_m, "redshift": geo_redshift},
            "sun_0p3au_to_1au": {
                "r_low_m": sun_r_0p3au,
                "r_high_m": sun_r_1au,
                "redshift": sun_0p3_to_1au,
                "required_abs_precision_for_3sigma_delta_f_over_f": sun_0p3_req_abs,
                "note": "Solar-potential lever arm (illustrative deep-space clock baseline; stationary).",
            },
            "sun_0p05au_to_1au": {
                "r_low_m": sun_r_0p05au,
                "r_high_m": sun_r_1au,
                "redshift": sun_0p05_to_1au,
                "required_abs_precision_for_3sigma_delta_f_over_f": sun_0p05_req_abs,
                "note": "Solar-potential lever arm (illustrative deep-space clock baseline; stationary).",
            },
            "neutron_star_surface_to_infty": {
                "r_low_m": r_ns,
                "r_high_m": 1e20,
                "redshift": ns_surface_to_infty,
                "required_abs_precision_for_3sigma_delta_f_over_f": ns_req_abs,
                "note": "Strong-field illustration (not a lab experiment; astrophysical systematics dominate).",
            },
        },
        "outputs": {
            "png": str(out_png),
            "atom_interferometer_unified_audit_png": str(out_png_unified),
            "atom_interferometer_unified_audit_csv": str(summary_csv),
            "atom_interferometer_unified_audit_metrics": str(summary_json),
        },
        "notes": [
            "For interferometers we estimate Δphase by scaling the baseline phase with the fractional z-difference between P-model and GR at the relevant Δr.",
            "This is a detectability-oriented estimate; detailed AI phase budgets include additional cancellations (laser phase, path geometry, gradients).",
        ],
    }

    out_json = out_dir / "gravity_quantum_interference_delta_predictions.json"
    out_json.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
