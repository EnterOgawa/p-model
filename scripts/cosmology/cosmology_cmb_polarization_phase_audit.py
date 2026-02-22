#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_cmb_polarization_phase_audit.py

Step 8.7.18.3（CMB polarization: TT/EE/TE phase audit）

目的：
- Part I 2.7.5 の輻射輸送式を Stokes ベクトル（I,Q,U）へ拡張し、
  Thomson 散乱の位相行列を使って EE / TE の位相予言を固定する。
- Planck 2018 binned spectra（TT/TE/EE）に対して、
  TT に対する EE / TE の位相ずれを同一I/Fで監査し、反証条件を JSON 化する。

入力（固定）：
- data/cosmology/planck2018_com_power_spect_tt_binned_r3.01.txt
- data/cosmology/planck2018_com_power_spect_te_binned_r3.02.txt
- data/cosmology/planck2018_com_power_spect_ee_binned_r3.02.txt

出力（固定名）：
- output/private/cosmology/cosmology_cmb_polarization_phase_audit.png
- output/private/cosmology/cosmology_cmb_polarization_phase_audit_metrics.json
- output/private/cosmology/cosmology_cmb_polarization_phase_audit_falsification_pack.json
- output/private/cosmology/cosmology_cmb_polarization_phase_audit_peaks.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summary import worklog  # noqa: E402


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


def _fmt(value: float, digits: int = 6) -> str:
    if value == 0.0:
        return "0"
    abs_value = abs(float(value))
    if abs_value >= 1e4 or abs_value < 1e-3:
        return f"{value:.{digits}g}"
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _quadrupole_velocity_gradient_summary(*, c_s_p: float = 1.0 / math.sqrt(3.0)) -> Dict[str, Any]:
    cs = float(c_s_p)
    if not (cs > 0.0):
        raise ValueError("c_s_p must be positive.")
    return {
        "continuity_real_space": "Theta' + (1/3) nabla·v_pb = 0",
        "continuity_fourier": "Theta_k' = -(k/3) v_pb,k",
        "temperature_mode": "Theta_k(r_s)=Theta_eq,k + DeltaTheta_k cos(k r_s)",
        "velocity_mode_from_continuity": "v_pb,k(r_s)=3 c_s,P DeltaTheta_k sin(k r_s)",
        "velocity_gradient_relation": "nabla·v_pb,k = i k v_pb,k ∝ sin(k r_s)",
        "quadrupole_hierarchy": "Pi_k' + dot_tau_T Pi_k = (8/15) k v_pb,k",
        "tight_coupling_limit": "dot_tau_T >> partial_s => Pi_k ≈ (8/15)(k/dot_tau_T) v_pb,k ≈ (8/15)(1/dot_tau_T) nabla·v_pb,k",
        "polarization_source_mapping": "Q/U source in Thomson term is proportional to (1-mu^2) Pi_k; therefore E-mode is sourced by velocity-gradient phase (sin branch).",
    }


def _phase_shift_complete_proof(
    *,
    delta_ee: float,
    delta_te: float,
    gate: Dict[str, Any],
) -> Dict[str, Any]:
    checks = gate.get("checks", {})
    summary = gate.get("summary", {})
    return {
        "symbolic_chain": {
            "temperature_mode": "Theta_k = Theta_eq,k + DeltaTheta_k cos(x), x = k r_s",
            "quadrupole_mode": "Pi_k ∝ sin(x) (from continuity + velocity-gradient + Thomson hierarchy)",
            "polarization_mode": "E_k ∝ Pi_k ∝ sin(x)",
            "tt_power": "C_ell^TT ∝ |Theta_k|^2 ∝ cos^2(x)",
            "ee_power": "C_ell^EE ∝ |E_k|^2 ∝ sin^2(x)",
            "te_cross": "C_ell^TE ∝ Re[Theta_k E_k*] ∝ sin(x)cos(x) = 0.5 sin(2x)",
        },
        "peak_positions_x": {
            "tt_maxima": "x_n^TT = nπ",
            "ee_maxima": "x_n^EE = (n+1/2)π",
            "te_extrema": "x_m^TE = (m+1/4)π",
            "phase_offsets": {
                "EE_minus_TT": "Δx = π/2 (half-wavelength)",
                "TE_minus_TT": "Δx = π/4 (intermediate)",
            },
        },
        "multipole_mapping": {
            "tt": "ell_n^TT ≈ (n-phi) ell_A",
            "ee": "ell_n^EE ≈ (n-phi+1/2) ell_A",
            "te": "ell_m^TE,ext ≈ (m-phi+1/4) ell_A",
        },
        "planck_consistency": {
            "delta_ee_expected": 0.5,
            "delta_te_expected": 0.25,
            "delta_ee_fit": float(delta_ee),
            "delta_te_fit": float(delta_te),
            "abs_delta_ee_from_expected": float(summary.get("abs_delta_ee_from_half", float("nan"))),
            "abs_delta_te_from_expected": float(summary.get("abs_delta_te_from_quarter", float("nan"))),
            "max_abs_delta_ell_ee": float(summary.get("ee_max_abs_delta_ell", float("nan"))),
            "max_abs_delta_ell_te": float(summary.get("te_max_abs_delta_ell", float("nan"))),
            "gate_checks": {
                "phase_shift_hard_gate": bool(checks.get("phase_shift_hard_gate", False)),
                "ee_phase": bool(checks.get("ee_phase", False)),
                "te_phase": bool(checks.get("te_phase", False)),
                "ee_peak_positions": bool(checks.get("ee_peak_positions", False)),
                "te_peak_positions": bool(checks.get("te_peak_positions", False)),
                "te_sign_alternation": bool(checks.get("te_sign_alternation", False)),
            },
            "overall_status": str(gate.get("overall_status", "n/a")),
        },
    }


@dataclass(frozen=True)
class Spectrum:
    ell: np.ndarray
    dl: np.ndarray
    sigma: np.ndarray
    bestfit: np.ndarray


@dataclass(frozen=True)
class Feature:
    channel: str
    mode: int
    kind: str
    ell_pred: float
    ell_obs: float
    dl_obs: float
    dl_bestfit: float
    sigma_dl: float
    ell_half_width: float

    @property
    def delta_ell(self) -> float:
        return float(self.ell_obs - self.ell_pred)

    @property
    def z_ell(self) -> float:
        denom = max(float(self.ell_half_width), 1e-9)
        return float(self.delta_ell / denom)


def _read_binned_spectrum(path: Path) -> Spectrum:
    arr = np.loadtxt(path)
    if arr.ndim != 2 or arr.shape[1] < 5:
        raise ValueError(f"unexpected spectrum format: {path}")
    ell = arr[:, 0].astype(float)
    dl = arr[:, 1].astype(float)
    err_lo = np.abs(arr[:, 2].astype(float))
    err_hi = np.abs(arr[:, 3].astype(float))
    sigma = np.maximum(0.5 * (err_lo + err_hi), 1e-12)
    bestfit = arr[:, 4].astype(float)
    return Spectrum(ell=ell, dl=dl, sigma=sigma, bestfit=bestfit)


def _find_extremum(ell: np.ndarray, y: np.ndarray, lo: float, hi: float, *, kind: str) -> int:
    mask = (ell >= float(lo)) & (ell <= float(hi))
    if not np.any(mask):
        raise ValueError(f"no bins in [{lo}, {hi}]")
    idxs = np.where(mask)[0]
    local = y[mask]
    if kind == "max":
        local_idx = int(np.argmax(local))
    elif kind == "min":
        local_idx = int(np.argmin(local))
    else:
        raise ValueError(f"unknown extremum kind: {kind}")
    return int(idxs[local_idx])


def _ell_half_width(ell: np.ndarray, index: int) -> float:
    if index <= 0:
        return float(0.5 * abs(ell[1] - ell[0])) if ell.size >= 2 else 1.0
    if index >= ell.size - 1:
        return float(0.5 * abs(ell[-1] - ell[-2])) if ell.size >= 2 else 1.0
    return float(0.5 * abs(ell[index + 1] - ell[index - 1]))


def _fit_tt_acoustic(tt: Spectrum) -> Dict[str, Any]:
    peak_ranges = [(150.0, 320.0), (400.0, 650.0), (700.0, 950.0)]
    tt_peaks: List[Feature] = []
    observed_ell: List[float] = []
    for mode, (lo, hi) in enumerate(peak_ranges, start=1):
        idx = _find_extremum(tt.ell, tt.bestfit, lo, hi, kind="max")
        ell_obs = float(tt.ell[idx])
        observed_ell.append(ell_obs)
        tt_peaks.append(
            Feature(
                channel="TT",
                mode=int(mode),
                kind="max",
                ell_pred=float("nan"),
                ell_obs=ell_obs,
                dl_obs=float(tt.dl[idx]),
                dl_bestfit=float(tt.bestfit[idx]),
                sigma_dl=float(tt.sigma[idx]),
                ell_half_width=_ell_half_width(tt.ell, idx),
            )
        )

    mode_axis = np.array([1.0, 2.0, 3.0], dtype=float)
    mat = np.column_stack([mode_axis, np.ones_like(mode_axis)])
    ell_a, intercept = np.linalg.lstsq(mat, np.array(observed_ell, dtype=float), rcond=None)[0]
    ell_a = float(ell_a)
    phi = float(-intercept / max(ell_a, 1e-12))

    tt_fixed: List[Feature] = []
    for feature in tt_peaks:
        ell_pred = float((float(feature.mode) - phi) * ell_a)
        tt_fixed.append(
            Feature(
                channel=feature.channel,
                mode=feature.mode,
                kind=feature.kind,
                ell_pred=ell_pred,
                ell_obs=feature.ell_obs,
                dl_obs=feature.dl_obs,
                dl_bestfit=feature.dl_bestfit,
                sigma_dl=feature.sigma_dl,
                ell_half_width=feature.ell_half_width,
            )
        )

    return {"ell_a": ell_a, "phi": phi, "features": tt_fixed}


def _detect_ee_peaks(ee: Spectrum, *, ell_a: float, phi: float, count: int, window: float) -> List[Feature]:
    rows: List[Feature] = []
    for mode in range(1, int(count) + 1):
        ell_pred = float((float(mode) - phi + 0.5) * ell_a)
        idx = _find_extremum(ee.ell, ee.bestfit, ell_pred - window, ell_pred + window, kind="max")
        rows.append(
            Feature(
                channel="EE",
                mode=int(mode),
                kind="max",
                ell_pred=ell_pred,
                ell_obs=float(ee.ell[idx]),
                dl_obs=float(ee.dl[idx]),
                dl_bestfit=float(ee.bestfit[idx]),
                sigma_dl=float(ee.sigma[idx]),
                ell_half_width=_ell_half_width(ee.ell, idx),
            )
        )
    return rows


def _detect_te_extrema(te: Spectrum, *, ell_a: float, phi: float, count: int, window: float) -> List[Feature]:
    rows: List[Feature] = []
    for mode in range(1, int(count) + 1):
        ell_pred = float((float(mode) - phi + 0.25) * ell_a)
        kind = "max" if (mode % 2 == 1) else "min"
        idx = _find_extremum(te.ell, te.bestfit, ell_pred - window, ell_pred + window, kind=kind)
        rows.append(
            Feature(
                channel="TE",
                mode=int(mode),
                kind=kind,
                ell_pred=ell_pred,
                ell_obs=float(te.ell[idx]),
                dl_obs=float(te.dl[idx]),
                dl_bestfit=float(te.bestfit[idx]),
                sigma_dl=float(te.sigma[idx]),
                ell_half_width=_ell_half_width(te.ell, idx),
            )
        )
    return rows


def _fit_phase_offset(features: Sequence[Feature], *, ell_a: float, phi: float) -> Dict[str, float]:
    if not features:
        return {"delta_fit": float("nan"), "delta_sigma": float("nan")}
    implied = np.array([f.ell_obs / ell_a - float(f.mode) + phi for f in features], dtype=float)
    delta_fit = float(np.mean(implied))
    if implied.size <= 1:
        delta_sigma = float("nan")
    else:
        delta_sigma = float(np.std(implied, ddof=1) / math.sqrt(float(implied.size)))
    return {"delta_fit": delta_fit, "delta_sigma": delta_sigma}


def _phase_gate(
    *,
    ee_features: Sequence[Feature],
    te_features: Sequence[Feature],
    delta_ee: float,
    delta_te: float,
) -> Dict[str, Any]:
    ee_max_abs_dell = float(max(abs(f.delta_ell) for f in ee_features)) if ee_features else float("nan")
    te_max_abs_dell = float(max(abs(f.delta_ell) for f in te_features)) if te_features else float("nan")
    te_sign_alternation = True
    for row in te_features:
        if row.kind == "max" and row.dl_bestfit <= 0.0:
            te_sign_alternation = False
        if row.kind == "min" and row.dl_bestfit >= 0.0:
            te_sign_alternation = False

    thresholds = {
        "phase_abs_delta_ee_from_half": 0.12,
        "phase_abs_delta_te_from_quarter": 0.12,
        "max_abs_delta_ell_ee": 50.0,
        "max_abs_delta_ell_te": 55.0,
        "te_sign_alternation_required": True,
        "phase_shift_hard_gate_required": True,
    }

    cond_ee_phase = bool(abs(delta_ee - 0.5) <= thresholds["phase_abs_delta_ee_from_half"])
    cond_te_phase = bool(abs(delta_te - 0.25) <= thresholds["phase_abs_delta_te_from_quarter"])
    cond_ee_ell = bool(ee_max_abs_dell <= thresholds["max_abs_delta_ell_ee"])
    cond_te_ell = bool(te_max_abs_dell <= thresholds["max_abs_delta_ell_te"])
    cond_te_sign = bool(te_sign_alternation)

    hard_gate_phase_shift = bool(cond_ee_phase and cond_te_phase and cond_te_sign)
    hard_fail = not hard_gate_phase_shift
    if hard_fail:
        status = "reject"
    elif cond_ee_ell and cond_te_ell:
        status = "pass"
    else:
        status = "watch"

    return {
        "thresholds": thresholds,
        "checks": {
            "phase_shift_hard_gate": hard_gate_phase_shift,
            "ee_phase": cond_ee_phase,
            "te_phase": cond_te_phase,
            "ee_peak_positions": cond_ee_ell,
            "te_peak_positions": cond_te_ell,
            "te_sign_alternation": cond_te_sign,
        },
        "hard_gate": {
            "name": "cmb_phase_shift_lock",
            "rule": "abs(Δφ_EE-0.5)<=0.12 and abs(Δφ_TE-0.25)<=0.12 and TE sign alternation",
            "pass": hard_gate_phase_shift,
            "components": {
                "ee_phase": cond_ee_phase,
                "te_phase": cond_te_phase,
                "te_sign_alternation": cond_te_sign,
            },
        },
        "summary": {
            "ee_max_abs_delta_ell": ee_max_abs_dell,
            "te_max_abs_delta_ell": te_max_abs_dell,
            "abs_delta_ee_from_half": float(abs(delta_ee - 0.5)),
            "abs_delta_te_from_quarter": float(abs(delta_te - 0.25)),
        },
        "overall_status": status,
        "hard_fail": hard_fail,
    }


def _plot(
    *,
    tt: Spectrum,
    te: Spectrum,
    ee: Spectrum,
    tt_features: Sequence[Feature],
    ee_features: Sequence[Feature],
    te_features: Sequence[Feature],
    phase_fit_ee: Dict[str, float],
    phase_fit_te: Dict[str, float],
    gate: Dict[str, Any],
    out_png: Path,
) -> None:
    _set_japanese_font()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    ax_tt, ax_ee, ax_te, ax_phase = axes.ravel()

    ax_tt.plot(tt.ell, tt.bestfit, color="#1f77b4", linewidth=1.4, label="TT best-fit")
    ax_tt.scatter([f.ell_obs for f in tt_features], [f.dl_bestfit for f in tt_features], color="#d62728", s=55, zorder=4, label="TT peaks (fit)")
    for f in tt_features:
        ax_tt.axvline(f.ell_pred, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.5)
    ax_tt.set_xlim(120, 1100)
    ax_tt.set_xlabel("multipole ℓ")
    ax_tt.set_ylabel("D_ℓ [μK²]")
    ax_tt.set_title("TT peaks used to fix ℓ_A and φ")
    ax_tt.grid(True, linestyle="--", alpha=0.35)
    ax_tt.legend(fontsize=8, loc="upper right")

    ax_ee.errorbar(ee.ell, ee.dl, yerr=ee.sigma, fmt=".", color="#bbbbbb", alpha=0.35, label="EE observed")
    ax_ee.plot(ee.ell, ee.bestfit, color="#2ca02c", linewidth=1.3, label="EE best-fit")
    ax_ee.scatter([f.ell_obs for f in ee_features], [f.dl_bestfit for f in ee_features], color="#2ca02c", s=55, zorder=4, label="EE peaks")
    for f in ee_features:
        ax_ee.axvline(f.ell_pred, color="#9467bd", linestyle="--", linewidth=1.0, alpha=0.65)
    ax_ee.set_xlim(220, 1250)
    ax_ee.set_xlabel("multipole ℓ")
    ax_ee.set_ylabel("D_ℓ [μK²]")
    ax_ee.set_title("EE: predicted half-phase shift from TT")
    ax_ee.grid(True, linestyle="--", alpha=0.35)
    ax_ee.legend(fontsize=8, loc="upper right")

    ax_te.errorbar(te.ell, te.dl, yerr=te.sigma, fmt=".", color="#bbbbbb", alpha=0.35, label="TE observed")
    ax_te.plot(te.ell, te.bestfit, color="#ff7f0e", linewidth=1.3, label="TE best-fit")
    ax_te.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0, alpha=0.6)
    max_rows = [f for f in te_features if f.kind == "max"]
    min_rows = [f for f in te_features if f.kind == "min"]
    ax_te.scatter([f.ell_obs for f in max_rows], [f.dl_bestfit for f in max_rows], color="#ff7f0e", s=52, marker="o", zorder=4, label="TE max")
    ax_te.scatter([f.ell_obs for f in min_rows], [f.dl_bestfit for f in min_rows], color="#8c564b", s=52, marker="v", zorder=4, label="TE min")
    for f in te_features:
        ax_te.axvline(f.ell_pred, color="#17becf", linestyle="--", linewidth=1.0, alpha=0.65)
    ax_te.set_xlim(180, 1900)
    ax_te.set_xlabel("multipole ℓ")
    ax_te.set_ylabel("D_ℓ [μK²]")
    ax_te.set_title("TE: predicted quarter-phase shift from TT")
    ax_te.grid(True, linestyle="--", alpha=0.35)
    ax_te.legend(fontsize=8, loc="upper right")

    expected = [0.5, 0.25]
    fitted = [float(phase_fit_ee["delta_fit"]), float(phase_fit_te["delta_fit"])]
    labels = ["EE phase shift", "TE phase shift"]
    x = np.arange(len(labels), dtype=float)
    w = 0.36
    ax_phase.bar(x - 0.5 * w, expected, width=w, color="#7f7f7f", label="expected")
    ax_phase.bar(x + 0.5 * w, fitted, width=w, color="#1f77b4", label="fitted")
    ax_phase.set_xticks(x)
    ax_phase.set_xticklabels(labels)
    ax_phase.set_ylabel("phase offset Δφ")
    ax_phase.set_ylim(0.0, 0.9)
    ax_phase.grid(True, linestyle="--", alpha=0.35, axis="y")
    ax_phase.set_title("Phase-offset audit (Planck TT/TE/EE)")
    ax_phase.legend(fontsize=9, loc="upper right")

    status = gate.get("overall_status", "n/a")
    ee_shift = fitted[0] - expected[0]
    te_shift = fitted[1] - expected[1]
    fig.suptitle("CMB polarization transfer audit (Stokes extension + Thomson source)", fontsize=14)
    fig.text(
        0.5,
        0.01,
        (
            f"overall={status}; Δφ_EE(obs-exp)={_fmt(float(ee_shift), 4)}, "
            f"Δφ_TE(obs-exp)={_fmt(float(te_shift), 4)}; "
            f"max|Δℓ|_EE={_fmt(float(gate['summary']['ee_max_abs_delta_ell']), 4)}, "
            f"max|Δℓ|_TE={_fmt(float(gate['summary']['te_max_abs_delta_ell']), 4)}"
        ),
        ha="center",
        fontsize=10,
    )
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.94))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _write_peaks_csv(path: Path, rows: Sequence[Feature]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "channel",
                "mode",
                "kind",
                "ell_pred",
                "ell_obs",
                "delta_ell",
                "ell_half_width",
                "z_ell",
                "dl_obs",
                "dl_bestfit",
                "sigma_dl",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.channel,
                    row.mode,
                    row.kind,
                    f"{row.ell_pred:.8f}",
                    f"{row.ell_obs:.8f}",
                    f"{row.delta_ell:.8f}",
                    f"{row.ell_half_width:.8f}",
                    f"{row.z_ell:.8f}",
                    f"{row.dl_obs:.8f}",
                    f"{row.dl_bestfit:.8f}",
                    f"{row.sigma_dl:.8f}",
                ]
            )


def _copy_to_public(private_paths: Sequence[Path], public_dir: Path) -> Dict[str, str]:
    public_dir.mkdir(parents=True, exist_ok=True)
    copied: Dict[str, str] = {}
    for src in private_paths:
        dst = public_dir / src.name
        shutil.copy2(src, dst)
        copied[src.name] = str(dst).replace("\\", "/")
    return copied


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="CMB polarization phase audit (TT/TE/EE with Stokes transport extension).")
    parser.add_argument(
        "--tt-data",
        type=str,
        default=str(ROOT / "data" / "cosmology" / "planck2018_com_power_spect_tt_binned_r3.01.txt"),
    )
    parser.add_argument(
        "--te-data",
        type=str,
        default=str(ROOT / "data" / "cosmology" / "planck2018_com_power_spect_te_binned_r3.02.txt"),
    )
    parser.add_argument(
        "--ee-data",
        type=str,
        default=str(ROOT / "data" / "cosmology" / "planck2018_com_power_spect_ee_binned_r3.02.txt"),
    )
    parser.add_argument("--ee-modes", type=int, default=3, help="Number of EE peaks for phase-fit.")
    parser.add_argument("--te-modes", type=int, default=6, help="Number of TE extrema for phase-fit.")
    parser.add_argument("--ee-window", type=float, default=120.0, help="Half-width [ell] for EE peak search.")
    parser.add_argument("--te-window", type=float, default=70.0, help="Half-width [ell] for TE extrema search.")
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "output" / "private" / "cosmology"),
    )
    parser.add_argument(
        "--public-dir",
        type=str,
        default=str(ROOT / "output" / "public" / "cosmology"),
    )
    parser.add_argument("--skip-public-copy", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    tt = _read_binned_spectrum(Path(args.tt_data).resolve())
    te = _read_binned_spectrum(Path(args.te_data).resolve())
    ee = _read_binned_spectrum(Path(args.ee_data).resolve())

    tt_fit = _fit_tt_acoustic(tt)
    ell_a = float(tt_fit["ell_a"])
    phi = float(tt_fit["phi"])
    tt_features: List[Feature] = list(tt_fit["features"])

    ee_features = _detect_ee_peaks(ee, ell_a=ell_a, phi=phi, count=max(int(args.ee_modes), 1), window=max(float(args.ee_window), 1.0))
    te_features = _detect_te_extrema(te, ell_a=ell_a, phi=phi, count=max(int(args.te_modes), 2), window=max(float(args.te_window), 1.0))

    phase_fit_ee = _fit_phase_offset(ee_features, ell_a=ell_a, phi=phi)
    phase_fit_te = _fit_phase_offset(te_features, ell_a=ell_a, phi=phi)
    delta_ee = float(phase_fit_ee["delta_fit"])
    delta_te = float(phase_fit_te["delta_fit"])

    gate = _phase_gate(ee_features=ee_features, te_features=te_features, delta_ee=delta_ee, delta_te=delta_te)
    quadrupole_derivation = _quadrupole_velocity_gradient_summary()
    phase_shift_proof = _phase_shift_complete_proof(delta_ee=delta_ee, delta_te=delta_te, gate=gate)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    base = "cosmology_cmb_polarization_phase_audit"
    out_png = out_dir / f"{base}.png"
    out_json = out_dir / f"{base}_metrics.json"
    out_fals = out_dir / f"{base}_falsification_pack.json"
    out_csv = out_dir / f"{base}_peaks.csv"

    _plot(
        tt=tt,
        te=te,
        ee=ee,
        tt_features=tt_features,
        ee_features=ee_features,
        te_features=te_features,
        phase_fit_ee=phase_fit_ee,
        phase_fit_te=phase_fit_te,
        gate=gate,
        out_png=out_png,
    )
    all_rows = list(tt_features) + list(ee_features) + list(te_features)
    _write_peaks_csv(out_csv, all_rows)

    metrics_payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "tt": str(Path(args.tt_data).resolve()).replace("\\", "/"),
            "te": str(Path(args.te_data).resolve()).replace("\\", "/"),
            "ee": str(Path(args.ee_data).resolve()).replace("\\", "/"),
        },
        "model": {
            "stokes_transport_equation": "k^μ ∇_μ S̃ = J̃ - Ã S̃ + n_e σ_T M(μ,φ) S̃, S̃=(Ĩ,Q̃,Ũ)^T",
            "thomson_phase_matrix_axisymmetric": "(3/4)*[[1+μ^2,1-μ^2,0],[1-μ^2,1+μ^2,0],[0,0,2μ]]",
            "quadrupole_source_note": "Q/U source term is proportional to local quadrupole Π and (1-μ^2).",
            "quadrupole_generation_derivation": quadrupole_derivation,
            "phase_shift_complete_proof": phase_shift_proof,
            "acoustic_phase_relations": {
                "TT": "Θ ∝ cos(x), x=k r_s  => ℓ_n^TT ≈ (n-φ)ℓ_A",
                "EE": "E ∝ sin(x) => C_ℓ^EE ∝ sin^2(x) => ℓ_n^EE ≈ (n-φ+1/2)ℓ_A",
                "TE": "C_ℓ^TE ∝ sin(x)cos(x)=0.5 sin(2x) => ℓ_m^TE,ext ≈ (m-φ+1/4)ℓ_A",
            },
            "tt_fit": {"ell_a": ell_a, "phi": phi},
            "search_windows": {"ee_half_width_ell": float(args.ee_window), "te_half_width_ell": float(args.te_window)},
        },
        "phase_fit": {
            "ee": {
                "expected_delta": 0.5,
                **phase_fit_ee,
                "abs_shift_from_expected": float(abs(delta_ee - 0.5)),
            },
            "te": {
                "expected_delta": 0.25,
                **phase_fit_te,
                "abs_shift_from_expected": float(abs(delta_te - 0.25)),
            },
        },
        "features": [
            {
                "channel": row.channel,
                "mode": int(row.mode),
                "kind": row.kind,
                "ell_pred": float(row.ell_pred),
                "ell_obs": float(row.ell_obs),
                "delta_ell": float(row.delta_ell),
                "ell_half_width": float(row.ell_half_width),
                "z_ell": float(row.z_ell),
                "dl_obs": float(row.dl_obs),
                "dl_bestfit": float(row.dl_bestfit),
                "sigma_dl": float(row.sigma_dl),
            }
            for row in all_rows
        ],
        "gate": gate,
        "outputs": {
            "png": str(out_png).replace("\\", "/"),
            "metrics_json": str(out_json).replace("\\", "/"),
            "falsification_pack_json": str(out_fals).replace("\\", "/"),
            "peaks_csv": str(out_csv).replace("\\", "/"),
        },
    }
    _write_json(out_json, metrics_payload)

    fals_payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "kind": "cmb_polarization_phase_audit",
        "decision_rule": {
            "phase_hard_gate": "|Δφ_EE-0.5|<=0.12 and |Δφ_TE-0.25|<=0.12 and TE signs alternate",
            "position": "max|Δℓ_EE|<=50 and max|Δℓ_TE|<=55",
            "overall": "phase_hard_gate fail => reject; phase_hard_gate pass + position pass => pass; otherwise watch",
        },
        "result": gate,
        "notes": [
            "本監査は Planck binned TT/TE/EE の位相関係に限定した最小監査であり、全天球尤度の全パラメータ同時推定ではない。",
            "TE/EE の位相ずれは Part I 2.7.5 の Stokes 輻射輸送拡張と Thomson 散乱源（四重極）から導く。 ",
        ],
        "related_outputs": {
            "metrics_json": str(out_json).replace("\\", "/"),
            "figure_png": str(out_png).replace("\\", "/"),
            "peaks_csv": str(out_csv).replace("\\", "/"),
        },
    }
    _write_json(out_fals, fals_payload)

    copied: Dict[str, str] = {}
    if not bool(args.skip_public_copy):
        copied = _copy_to_public([out_png, out_json, out_fals, out_csv], Path(args.public_dir).resolve())

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")
    print(f"[ok] fals: {out_fals}")
    print(f"[ok] csv : {out_csv}")
    if copied:
        print(f"[ok] copied to public: {len(copied)} files")

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_cmb_polarization_phase_audit",
                "argv": list(sys.argv),
                "inputs": {"tt": Path(args.tt_data).resolve(), "te": Path(args.te_data).resolve(), "ee": Path(args.ee_data).resolve()},
                "outputs": {
                    "png": out_png,
                    "metrics_json": out_json,
                    "falsification_pack_json": out_fals,
                    "peaks_csv": out_csv,
                    "public_copies": copied,
                },
                "metrics": {
                    "ell_a": ell_a,
                    "phi": phi,
                    "delta_ee_fit": delta_ee,
                    "delta_te_fit": delta_te,
                    "overall_status": gate.get("overall_status", "n/a"),
                    "ee_max_abs_delta_ell": gate.get("summary", {}).get("ee_max_abs_delta_ell"),
                    "te_max_abs_delta_ell": gate.get("summary", {}).get("te_max_abs_delta_ell"),
                },
            }
        )
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
