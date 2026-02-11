#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_alcock_paczynski_constraints.py

Step 14.2.3（独立プローブで相互検証）:
Alcock–Paczynski（AP）パラメータ F_AP(z)=D_M(z)H(z)/c を一次ソース値から固定出力し、
静的背景P（宇宙膨張なし）モデルの最小形（指数P_bg）と比較する。

入力（固定）:
  - data/cosmology/alcock_paczynski_constraints.json

出力（固定名）:
  - output/cosmology/cosmology_alcock_paczynski_constraints.png
  - output/cosmology/cosmology_alcock_paczynski_constraints_metrics.json

モデル（比較用）:
1) 静的背景P（指数P_bg, H0^(P)=const）:
   - D(z) = (c/H0) ln(1+z)（静的近似：D≈cΔt）
   - H_eff(z) = H0 (1+z)（dD/dz=c/(H0(1+z)) から定義）
   - F_AP = D(z) H_eff(z)/c = (1+z) ln(1+z)
     ※H0は相殺される（F_APは無次元）。

2) 参考：flat ΛCDM（Planck 2018代表値）:
   - H(z)=H0 sqrt(Ωm(1+z)^3 + (1-Ωm))
   - D_M(z)=c/H0 ∫_0^z dz'/E(z')（flat）
   - F_AP=D_M H/c
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

from scripts.summary import worklog  # noqa: E402


_C_KM_S = 299_792.458


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


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _fmt_float(x: Optional[float], *, digits: int = 6) -> str:
    if x is None:
        return ""
    if x == 0.0:
        return "0"
    ax = abs(x)
    if ax >= 1e4 or ax < 1e-3:
        return f"{x:.{digits}g}"
    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


@dataclass(frozen=True)
class Constraint:
    id: str
    short_label: str
    z_eff: float
    DM_scaled_mpc: float
    DM_scaled_sigma_mpc: float
    H_scaled_km_s_mpc: float
    H_scaled_sigma_km_s_mpc: float
    corr_DM_H: float
    sigma_note: str
    source: Dict[str, Any]

    @staticmethod
    def from_json(j: Dict[str, Any]) -> "Constraint":
        return Constraint(
            id=str(j["id"]),
            short_label=str(j.get("short_label") or j["id"]),
            z_eff=float(j["z_eff"]),
            DM_scaled_mpc=float(j["DM_scaled_mpc"]),
            DM_scaled_sigma_mpc=float(j["DM_scaled_sigma_mpc"]),
            H_scaled_km_s_mpc=float(j["H_scaled_km_s_mpc"]),
            H_scaled_sigma_km_s_mpc=float(j["H_scaled_sigma_km_s_mpc"]),
            corr_DM_H=float(j.get("corr_DM_H") or 0.0),
            sigma_note=str(j.get("sigma_note") or ""),
            source=dict(j.get("source") or {}),
        )


def _f_ap_obs(
    *,
    DM_mpc: float,
    DM_sigma_mpc: float,
    H_km_s_mpc: float,
    H_sigma_km_s_mpc: float,
    corr: float,
) -> Tuple[float, float]:
    f = (DM_mpc * H_km_s_mpc) / _C_KM_S
    cov = float(corr) * float(DM_sigma_mpc) * float(H_sigma_km_s_mpc)
    var = (H_km_s_mpc / _C_KM_S) ** 2 * (DM_sigma_mpc**2) + (DM_mpc / _C_KM_S) ** 2 * (H_sigma_km_s_mpc**2)
    var += 2.0 * (DM_mpc * H_km_s_mpc / (_C_KM_S**2)) * cov
    return float(f), float(math.sqrt(var)) if var > 0 else float("nan")


def _f_ap_pbg_exponential(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    one_p = 1.0 + z
    # F_AP = (1+z) ln(1+z)
    return one_p * np.log(one_p)


def _f_ap_lcdm_flat(
    z: np.ndarray,
    *,
    h0_km_s_mpc: float,
    omega_m: float,
) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    if np.any(z < 0):
        raise ValueError("z must be >=0")
    if not (0.0 < omega_m < 1.0):
        raise ValueError("omega_m must be in (0,1)")

    # Compute D_M(z) for a grid by integrating 1/E(z).
    z_grid = np.linspace(0.0, float(np.max(z)), 2000)
    one_p = 1.0 + z_grid
    E = np.sqrt(omega_m * one_p**3 + (1.0 - omega_m))
    integrand = 1.0 / E
    # cumulative trapezoid integral
    dz = np.diff(z_grid)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * dz)])
    # D_M = c/H0 * integral
    Dm = (_C_KM_S / h0_km_s_mpc) * cum

    # Interpolate D_M at requested z
    Dm_z = np.interp(z, z_grid, Dm)
    Ez = np.sqrt(omega_m * (1.0 + z) ** 3 + (1.0 - omega_m))
    Hz = h0_km_s_mpc * Ez
    return (Dm_z * Hz) / _C_KM_S


def compute(
    rows: Sequence[Constraint],
    *,
    h0_lcdm: float,
    omega_m_lcdm: float,
) -> List[Dict[str, Any]]:
    z_arr = np.array([r.z_eff for r in rows], dtype=float)
    f_lcdm = _f_ap_lcdm_flat(z_arr, h0_km_s_mpc=h0_lcdm, omega_m=omega_m_lcdm)
    f_pbg = _f_ap_pbg_exponential(z_arr)

    out: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        f_obs, f_sig = _f_ap_obs(
            DM_mpc=r.DM_scaled_mpc,
            DM_sigma_mpc=r.DM_scaled_sigma_mpc,
            H_km_s_mpc=r.H_scaled_km_s_mpc,
            H_sigma_km_s_mpc=r.H_scaled_sigma_km_s_mpc,
            corr=r.corr_DM_H,
        )
        z_pbg = (float(f_pbg[i]) - f_obs) / f_sig if f_sig > 0 else None
        z_lcdm = (float(f_lcdm[i]) - f_obs) / f_sig if f_sig > 0 else None

        out.append(
            {
                "id": r.id,
                "short_label": r.short_label,
                "z_eff": float(r.z_eff),
                "DM_scaled_mpc": float(r.DM_scaled_mpc),
                "DM_scaled_sigma_mpc": float(r.DM_scaled_sigma_mpc),
                "H_scaled_km_s_mpc": float(r.H_scaled_km_s_mpc),
                "H_scaled_sigma_km_s_mpc": float(r.H_scaled_sigma_km_s_mpc),
                "corr_DM_H": float(r.corr_DM_H),
                "F_AP_obs": float(f_obs),
                "F_AP_sigma": float(f_sig),
                "models": {
                    "P_bg_exponential": {"F_AP": float(f_pbg[i]), "z_score": (None if z_pbg is None else float(z_pbg))},
                    "LCDM_flat": {"F_AP": float(f_lcdm[i]), "z_score": (None if z_lcdm is None else float(z_lcdm))},
                },
                "sigma_note": r.sigma_note,
                "source": r.source,
            }
        )
    return out


def _plot(
    rows: Sequence[Dict[str, Any]],
    *,
    out_png: Path,
    h0_lcdm: float,
    omega_m_lcdm: float,
) -> None:
    z_pts = np.array([float(r["z_eff"]) for r in rows], dtype=float)
    f_obs = np.array([float(r["F_AP_obs"]) for r in rows], dtype=float)
    f_sig = np.array([float(r["F_AP_sigma"]) for r in rows], dtype=float)
    f_pbg = np.array([float(r["models"]["P_bg_exponential"]["F_AP"]) for r in rows], dtype=float)
    f_lcdm = np.array([float(r["models"]["LCDM_flat"]["F_AP"]) for r in rows], dtype=float)

    labels = [str(r.get("short_label") or r.get("id") or "") for r in rows]
    y = np.arange(len(rows))[::-1]
    z_pbg = np.array([float(r["models"]["P_bg_exponential"]["z_score"]) for r in rows], dtype=float)
    z_lcdm = np.array([float(r["models"]["LCDM_flat"]["z_score"]) for r in rows], dtype=float)

    z_max = max(1.2, float(np.max(z_pts)) + 0.1)
    z_curve = np.linspace(0.0, z_max, 500)
    f_curve_pbg = _f_ap_pbg_exponential(z_curve)
    f_curve_lcdm = _f_ap_lcdm_flat(z_curve, h0_km_s_mpc=h0_lcdm, omega_m=omega_m_lcdm)

    _set_japanese_font()
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.8, 6.8))

    # F_AP curves + points
    ax1.plot(z_curve, f_curve_pbg, linewidth=2.2, label="静的背景P（指数）: F=(1+z)ln(1+z)")
    ax1.plot(
        z_curve,
        f_curve_lcdm,
        linewidth=2.0,
        alpha=0.85,
        label=f"参考: flat ΛCDM（H0={h0_lcdm:.1f}, Ωm={omega_m_lcdm:.3f}）",
    )
    ax1.errorbar(
        z_pts,
        f_obs,
        yerr=f_sig,
        fmt="o",
        capsize=4,
        color="#111111",
        ecolor="#111111",
        label="観測（一次ソースから計算）",
    )
    ax1.set_title("Alcock–Paczynski：F_AP(z)=D_M H / c", fontsize=13)
    ax1.set_xlabel("赤方偏移 z", fontsize=11)
    ax1.set_ylabel("F_AP", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=9, loc="upper left")

    # z-scores per model
    ax2.axvline(0.0, color="k", linewidth=1.0, alpha=0.6)
    ax2.axvline(-3.0, color="#999999", linewidth=1.0, alpha=0.7, linestyle="--")
    ax2.axvline(3.0, color="#999999", linewidth=1.0, alpha=0.7, linestyle="--")

    dy = 0.12
    ax2.scatter(z_pbg, y + dy, s=45, label="静的背景P（指数）", color="#1f77b4")
    ax2.scatter(z_lcdm, y - dy, s=45, label="flat ΛCDM（参考）", color="#2ca02c")
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels)
    ax2.set_xlabel("z-score（(F_model - F_obs)/σ）", fontsize=11)
    ax2.set_title("モデル比較（z-score）", fontsize=13)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="x")
    ax2.legend(fontsize=9, loc="lower right")

    fig.suptitle("宇宙論（独立プローブ）：Alcock–Paczynski（AP）制約", fontsize=14)
    fig.text(
        0.5,
        0.01,
        "注：一次ソースは D_M×(r_d,fid/r_d) と H×(r_d/r_d,fid) を与えるため、F_AP は r_d に依存せず計算できる。",
        ha="center",
        fontsize=10,
    )
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.93))

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Cosmology: Alcock–Paczynski (F_AP) constraints vs static background-P.")
    ap.add_argument(
        "--data",
        type=str,
        default=str(_ROOT / "data" / "cosmology" / "alcock_paczynski_constraints.json"),
        help="Input JSON (default: data/cosmology/alcock_paczynski_constraints.json)",
    )
    ap.add_argument("--lcdm-h0", type=float, default=67.4, help="Reference LCDM H0 in km/s/Mpc (default: 67.4)")
    ap.add_argument("--lcdm-omega-m", type=float, default=0.315, help="Reference LCDM Ωm (default: 0.315)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    data_path = Path(args.data)
    src = _read_json(data_path)
    rows = [Constraint.from_json(c) for c in (src.get("constraints") or [])]
    if not rows:
        raise SystemExit(f"no constraints found in: {data_path}")

    # Sort by z for plot aesthetics
    rows = sorted(rows, key=lambda r: r.z_eff)

    h0_lcdm = float(args.lcdm_h0)
    omega_m_lcdm = float(args.lcdm_omega_m)

    out_dir = _ROOT / "output" / "cosmology"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "cosmology_alcock_paczynski_constraints.png"
    out_json = out_dir / "cosmology_alcock_paczynski_constraints_metrics.json"

    computed = compute(rows, h0_lcdm=h0_lcdm, omega_m_lcdm=omega_m_lcdm)
    _plot(computed, out_png=out_png, h0_lcdm=h0_lcdm, omega_m_lcdm=omega_m_lcdm)

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(data_path).replace("\\", "/"),
        "definition": src.get("definition") or {},
        "assumptions": {
            "F_AP": "F_AP(z)=D_M(z)H(z)/c",
            "obs_computation": "F_obs = (DM_scaled_mpc * H_scaled_km_s_Mpc) / c_km_s",
            "error_propagation": "σ^2 = (H/c)^2 σ_DM^2 + (DM/c)^2 σ_H^2 + 2 (DM*H/c^2) cov(DM,H), cov=ρ σ_DM σ_H",
            "P_bg_exponential": "F=(1+z)ln(1+z)",
            "LCDM_flat_reference": {"H0_km_s_Mpc": h0_lcdm, "Omega_m": omega_m_lcdm},
        },
        "rows": computed,
        "outputs": {"png": str(out_png).replace("\\", "/"), "metrics_json": str(out_json).replace("\\", "/")},
    }
    _write_json(out_json, payload)

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_alcock_paczynski_constraints",
                "argv": list(sys.argv),
                "inputs": {"data": data_path},
                "outputs": {"png": out_png, "metrics_json": out_json},
                "metrics": {"n_constraints": len(computed), "lcdm_ref": {"H0": h0_lcdm, "Omega_m": omega_m_lcdm}},
            }
        )
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

