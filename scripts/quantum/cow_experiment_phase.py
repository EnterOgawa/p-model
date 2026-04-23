from __future__ import annotations

import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from figure_japanese_localizer import enable_japanese_figure_localization
from scripts.utils.figure_locale_paths import localize_figure_output_path

enable_japanese_figure_localization()

# クラス: `Config` の責務と境界条件を定義する。
@dataclass(frozen=True)
class Config:
    # Geometry: Mannheim (arXiv:gr-qc/9611037) describes a square ABCD of side H.
    H_m: float = 0.03  # "a few centimeters"

    # Typical neutron speed quoted by Mannheim: v0 ~ 2×10^5 cm/s.
    v0_m_per_s: float = 2.0e5 / 100.0

    # Sweep tilt angle (effective g component for the interferometer plane).
    theta_deg_min: float = -90.0
    theta_deg_max: float = 90.0
    n_theta: int = 361

    # Earth gravity (conventional).
    g_m_per_s2: float = 9.80665

    # H-v sweep (Step 7.16.11: systematics wrt geometry and velocity).
    H_grid_m: tuple[float, ...] = (0.01, 0.02, 0.03, 0.04, 0.05)
    v_grid_m_per_s: tuple[float, ...] = (1000.0, 1500.0, 2000.0, 2500.0, 3000.0)


# 関数: `cow_phase_shift_rad` の入出力契約と処理意図を定義する。

def cow_phase_shift_rad(*, m_kg: float, g_m_per_s2: float, H_m: float, v0_m_per_s: float) -> float:
    """
    COW phase shift for a square interferometer of side H (Mannheim gr-qc/9611037):
      Δφ = - m g H^2 / (ħ v0)
    """
    hbar = 1.054571817e-34  # J·s
    return float(-m_kg * g_m_per_s2 * (H_m**2) / (hbar * v0_m_per_s))


# 関数: `_now_iso_utc` の入出力契約と処理意図を定義する。

def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_reference_catalog` の入出力契約と処理意図を定義する。

def _reference_catalog(*, cfg: Config, phi0_cycles: float) -> list[dict[str, Any]]:
    """
    Build an explicit source/integration catalog for COW (Step 7.16.11).
    Note:
      - Only one quantitative row is available in this initial pack:
        the relation-level representative magnitude quoted in Mannheim (1996 recap).
      - Raw fringe-shift tables from each historical run are not bundled yet.
    """
    return [
        {
            "record_id": "overhauser_colella_1974_prl33_1237",
            "year": 1974,
            "record_type": "primary_experiment",
            "citation": "Overhauser & Colella, Phys. Rev. Lett. 33, 1237 (1974)",
            "doi": "",
            "has_numeric_phase_point": False,
            "H_m": None,
            "v0_m_per_s": None,
            "phase_pred_cycles": None,
            "phase_obs_cycles": None,
            "phase_residual_cycles": None,
            "phase_residual_fraction": None,
            "stat_fraction": None,
            "sys_fraction": None,
            "notes": "Reference listed in Mannheim 1996; raw point not digitized in this pack.",
        },
        {
            "record_id": "colella_overhauser_werner_1975_prl34_1472",
            "year": 1975,
            "record_type": "primary_experiment",
            "citation": "Colella, Overhauser, Werner, Phys. Rev. Lett. 34, 1472 (1975)",
            "doi": "10.1103/PhysRevLett.34.1472",
            "has_numeric_phase_point": False,
            "H_m": None,
            "v0_m_per_s": None,
            "phase_pred_cycles": None,
            "phase_obs_cycles": None,
            "phase_residual_cycles": None,
            "phase_residual_fraction": None,
            "stat_fraction": None,
            "sys_fraction": None,
            "notes": "Primary COW source (paywall); raw fringe-shift table not bundled yet.",
        },
        {
            "record_id": "greenberger_overhauser_1979_rmp51_43",
            "year": 1979,
            "record_type": "review",
            "citation": "Greenberger & Overhauser, Rev. Mod. Phys. 51, 43 (1979)",
            "doi": "",
            "has_numeric_phase_point": False,
            "H_m": None,
            "v0_m_per_s": None,
            "phase_pred_cycles": None,
            "phase_obs_cycles": None,
            "phase_residual_cycles": None,
            "phase_residual_fraction": None,
            "stat_fraction": None,
            "sys_fraction": None,
            "notes": "Historical review listed in Mannheim 1996 references.",
        },
        {
            "record_id": "werner_1994_cqg11_a207",
            "year": 1994,
            "record_type": "review",
            "citation": "Werner, Class. Quantum Grav. 11, A207 (1994)",
            "doi": "",
            "has_numeric_phase_point": False,
            "H_m": None,
            "v0_m_per_s": None,
            "phase_pred_cycles": None,
            "phase_obs_cycles": None,
            "phase_residual_cycles": None,
            "phase_residual_fraction": None,
            "stat_fraction": None,
            "sys_fraction": None,
            "notes": "COW review source listed in Mannheim 1996 references.",
        },
        {
            "record_id": "mannheim_1996_grqc_9611037",
            "year": 1996,
            "record_type": "theory_recap",
            "citation": "Mannheim, arXiv:gr-qc/9611037 (1996)",
            "doi": "",
            "has_numeric_phase_point": False,
            "H_m": None,
            "v0_m_per_s": None,
            "phase_pred_cycles": None,
            "phase_obs_cycles": None,
            "phase_residual_cycles": None,
            "phase_residual_fraction": None,
            "stat_fraction": None,
            "sys_fraction": None,
            "notes": "Local cached recap PDF in data/quantum/sources/arxiv_gr-qc_9611037.pdf.",
        },
        {
            "record_id": "representative_relation_mannheim1996",
            "year": 1996,
            "record_type": "representative_relation",
            "citation": "Mannheim recap of COW magnitude relation (typical H,v)",
            "doi": "",
            "has_numeric_phase_point": True,
            "H_m": cfg.H_m,
            "v0_m_per_s": cfg.v0_m_per_s,
            "phase_pred_cycles": phi0_cycles,
            "phase_obs_cycles": phi0_cycles,
            "phase_residual_cycles": 0.0,
            "phase_residual_fraction": 0.0,
            "stat_fraction": None,
            "sys_fraction": None,
            "notes": (
                "Relation-level representative point (not a digitized raw fringe table). "
                "Used to freeze I/F and residual-audit plumbing in Step 7.16.11."
            ),
        },
    ]


# 関数: `_hv_sweep_rows` の入出力契約と処理意図を定義する。

def _hv_sweep_rows(*, cfg: Config, m_n: float) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for h_m in cfg.H_grid_m:
        for v_m_per_s in cfg.v_grid_m_per_s:
            phi_rad = cow_phase_shift_rad(
                m_kg=m_n, g_m_per_s2=cfg.g_m_per_s2, H_m=float(h_m), v0_m_per_s=float(v_m_per_s)
            )
            rows.append(
                {
                    "H_m": float(h_m),
                    "v0_m_per_s": float(v_m_per_s),
                    "phi_rad": float(phi_rad),
                    "phi_cycles": float(phi_rad / (2.0 * math.pi)),
                }
            )

    return rows


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config()

    # Neutron mass
    m_n = 1.67492749804e-27  # kg

    theta_deg = np.linspace(cfg.theta_deg_min, cfg.theta_deg_max, cfg.n_theta)
    theta_rad = np.deg2rad(theta_deg)

    # Model: scale by effective gravity component (simple orientation proxy).
    # This is a didactic sweep; the original setup relates geometry/orientation to g-projection.
    g_eff = cfg.g_m_per_s2 * np.sin(theta_rad)
    phi0 = cow_phase_shift_rad(m_kg=m_n, g_m_per_s2=cfg.g_m_per_s2, H_m=cfg.H_m, v0_m_per_s=cfg.v0_m_per_s)
    phi = phi0 * np.sin(theta_rad)  # linear in g_eff
    phi0_cycles = float(phi0 / (2.0 * math.pi))

    reference_rows = _reference_catalog(cfg=cfg, phi0_cycles=phi0_cycles)
    hv_rows = _hv_sweep_rows(cfg=cfg, m_n=m_n)
    observed_rows = [row for row in reference_rows if row.get("has_numeric_phase_point")]

    # Data integration CSVs (Step 7.16.11 frozen I/F).
    out_catalog_csv = out_dir / "cow_experiment_data_integration.csv"
    _write_csv(
        out_catalog_csv,
        reference_rows,
        [
            "record_id",
            "year",
            "record_type",
            "citation",
            "doi",
            "has_numeric_phase_point",
            "H_m",
            "v0_m_per_s",
            "phase_pred_cycles",
            "phase_obs_cycles",
            "phase_residual_cycles",
            "phase_residual_fraction",
            "stat_fraction",
            "sys_fraction",
            "notes",
        ],
    )
    out_hv_csv = out_dir / "cow_phase_shift_hv_sweep.csv"
    _write_csv(out_hv_csv, hv_rows, ["H_m", "v0_m_per_s", "phi_rad", "phi_cycles"])

    # Plot
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.8, 5.4), dpi=150)
    figure_lang = str(os.getenv("WAVEP_FIGURE_LANG", "ja")).strip().lower()
    is_en = figure_lang.startswith("en")
    panel_title_font = 14.8 * (0.72 if is_en else 1.0)
    axis_label_font = 13.6 * (0.68 if is_en else 1.0)
    tick_font = 12.4 * (0.70 if is_en else 1.0)
    legend_font = 11.8 * (0.72 if is_en else 1.0)
    note_font = 11.6 * (0.70 if is_en else 1.0)
    suptitle_font = 16.0
    ax.plot(theta_deg, phi / (2.0 * math.pi), lw=2.0, label="Δφ / 2π (cycles)")
    ax.axhline(0.0, color="0.25", lw=1.0)
    ax.set_xlabel("tilt angle θ (deg)", fontsize=axis_label_font)
    ax.set_ylabel("phase shift (cycles)", fontsize=axis_label_font)
    ax.set_title("COW (gravity-induced quantum interference): phase shift scaling", fontsize=panel_title_font)
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.legend(frameon=True, fontsize=legend_font, loc="upper right")
    ax.tick_params(axis="both", labelsize=tick_font)

    if is_en:
        fig.subplots_adjust(left=0.13, right=0.965, top=0.85, bottom=0.18)
    else:
        fig.tight_layout()

    out_png = localize_figure_output_path(out_dir / "cow_phase_shift.png", root=root)
    out_pdf = localize_figure_output_path(out_dir / "cow_phase_shift.pdf", root=root)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if is_en:
        fig.savefig(out_png)
        fig.savefig(out_pdf)
    else:
        fig.savefig(out_png, bbox_inches="tight")
        fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    # Complete-analysis figure: scaling + H-v map + source/readiness + residual audit.
    h_unique = sorted({float(r["H_m"]) for r in hv_rows})
    v_unique = sorted({float(r["v0_m_per_s"]) for r in hv_rows})
    grid = np.zeros((len(h_unique), len(v_unique)), dtype=float)
    for r in hv_rows:
        hi = h_unique.index(float(r["H_m"]))
        vi = v_unique.index(float(r["v0_m_per_s"]))
        grid[hi, vi] = abs(float(r["phi_cycles"]))

    type_counts: dict[str, int] = {}
    for row in reference_rows:
        key = str(row["record_type"])
        type_counts[key] = type_counts.get(key, 0) + 1

    fig2, axes = plt.subplots(2, 2, figsize=(12.8, 8.4), dpi=150)
    ax0, ax1, ax2, ax3 = axes.ravel()

    ax0.plot(theta_deg, phi / (2.0 * math.pi), lw=2.0, color="#1f77b4")
    ax0.axhline(0.0, color="0.25", lw=1.0)
    ax0.set_title("COW phase scaling vs tilt (representative)", fontsize=panel_title_font)
    ax0.set_xlabel("tilt angle θ (deg)", fontsize=axis_label_font)
    ax0.set_ylabel("Δφ / 2π (cycles)", fontsize=axis_label_font)
    ax0.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax0.tick_params(axis="both", labelsize=tick_font)

    im = ax1.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=[min(v_unique), max(v_unique), min(h_unique), max(h_unique)],
    )
    ax1.set_title("H-v sweep: |Δφ|/2π (cycles)", fontsize=panel_title_font)
    ax1.set_xlabel("v0 (m/s)", fontsize=axis_label_font)
    ax1.set_ylabel("H (m)", fontsize=axis_label_font)
    cbar = fig2.colorbar(im, ax=ax1)
    cbar.set_label("|Δφ| / 2π", fontsize=axis_label_font)
    cbar.ax.tick_params(labelsize=tick_font)
    ax1.tick_params(axis="both", labelsize=tick_font)

    keys = list(type_counts.keys())
    vals = [type_counts[k] for k in keys]
    ax2.bar(keys, vals, color="#7f7f7f")
    ax2.set_title("Source coverage by record type", fontsize=panel_title_font)
    ax2.set_ylabel("count", fontsize=axis_label_font)
    ax2.tick_params(axis="x", rotation=16, labelsize=tick_font)
    ax2.tick_params(axis="y", labelsize=tick_font)

    # 条件分岐: `observed_rows` を満たす経路を評価する。
    if observed_rows:
        pred = np.asarray([float(r["phase_pred_cycles"]) for r in observed_rows], dtype=float)
        obs = np.asarray([float(r["phase_obs_cycles"]) for r in observed_rows], dtype=float)
        lo = min(float(np.min(pred)), float(np.min(obs)))
        hi = max(float(np.max(pred)), float(np.max(obs)))
        # 条件分岐: `math.isclose(lo, hi)` を満たす経路を評価する。
        if math.isclose(lo, hi):
            lo -= 0.5
            hi += 0.5

        ax3.scatter(pred, obs, color="#d62728", s=42)
        ax3.plot([lo, hi], [lo, hi], ls="--", color="0.3", lw=1.0)
        ax3.set_xlim(lo, hi)
        ax3.set_ylim(lo, hi)
        ax3.set_title("Observed vs predicted (available rows)", fontsize=panel_title_font)
        ax3.set_xlabel("predicted phase (cycles)", fontsize=axis_label_font)
        ax3.set_ylabel("observed phase (cycles)", fontsize=axis_label_font)
        ax3.grid(True, ls=":", lw=0.6, alpha=0.7)
        ax3.tick_params(axis="both", labelsize=tick_font)
        abs_res = np.abs(obs - pred)
        ax3.text(
            0.03,
            0.95,
            f"n={len(observed_rows)}\nmedian |res|={np.median(abs_res):.3g} cycles",
            transform=ax3.transAxes,
            va="top",
            ha="left",
            fontsize=note_font,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "0.8"},
        )
    else:
        ax3.axis("off")
        ax3.text(
            0.5,
            0.5,
            "No quantitative observed rows in this pack.\nAdd raw fringe-shift tables to enable full residual audit.",
            ha="center",
            va="center",
            fontsize=note_font,
        )

    fig2.suptitle("Phase 7 / Step 7.16.11: COW integrated audit (initial freeze)", y=1.02, fontsize=suptitle_font)
    fig2.tight_layout()
    out_complete_png = out_dir / "cow_experiment_complete_analysis.png"
    out_complete_pdf = out_dir / "cow_experiment_complete_analysis.pdf"
    fig2.savefig(out_complete_png, bbox_inches="tight")
    fig2.savefig(out_complete_pdf, bbox_inches="tight")
    plt.close(fig2)

    # 条件分岐: `observed_rows` を満たす経路を評価する。
    if observed_rows:
        residual_abs_cycles = [abs(float(r["phase_residual_cycles"])) for r in observed_rows]
        residual_abs_frac = [abs(float(r["phase_residual_fraction"])) for r in observed_rows]
        residual_stats = {
            "n_observed_rows": len(observed_rows),
            "median_abs_residual_cycles": float(np.median(residual_abs_cycles)),
            "max_abs_residual_cycles": float(np.max(residual_abs_cycles)),
            "median_abs_residual_fraction": float(np.median(residual_abs_frac)),
            "max_abs_residual_fraction": float(np.max(residual_abs_frac)),
        }
    else:
        residual_stats = {
            "n_observed_rows": 0,
            "status": "insufficient_numeric_observed_rows",
            "action_required": "Add raw fringe-shift tables per experiment (post-1975) to enable full residual audit.",
        }

    observed_n = len(observed_rows)
    total_records_n = len(reference_rows)
    numeric_target_rows = [
        row
        for row in reference_rows
        if bool(row.get("has_numeric_phase_point")) or str(row.get("record_type") or "") == "representative_relation"
    ]
    numeric_target_n = len(numeric_target_rows)
    coverage_ratio_legacy = float(observed_n / total_records_n) if total_records_n > 0 else float("nan")
    coverage_ratio_targeted = float(observed_n / numeric_target_n) if numeric_target_n > 0 else None

    metrics = {
        "generated_utc": _now_iso_utc(),
        "phase": {"phase": 7, "step": "7.16.11", "name": "COW complete data integration and residual audit"},
        "source": {
            "reference": "Mannheim, 'Classical Underpinnings of Gravitationally Induced Quantum Interference' (arXiv:gr-qc/9611037)",
            "local_pdf": str(root / "data" / "quantum" / "sources" / "arxiv_gr-qc_9611037.pdf"),
            "formula": "Δφ = - m g H^2 / (ħ v0) (square interferometer; lowest order in g)",
        },
        "config": {
            "H_m": cfg.H_m,
            "v0_m_per_s": cfg.v0_m_per_s,
            "g_m_per_s2": cfg.g_m_per_s2,
            "theta_deg_min": cfg.theta_deg_min,
            "theta_deg_max": cfg.theta_deg_max,
            "n_theta": cfg.n_theta,
            "model_note": "Didactic orientation proxy: scale by sinθ.",
        },
        "constants": {
            "m_neutron_kg": m_n,
            "hbar_J_s": 1.054571817e-34,
        },
        "results": {
            "phi0_rad": phi0,
            "phi0_cycles": phi0_cycles,
            "phi_cycles_min": float(np.min(phi / (2.0 * math.pi))),
            "phi_cycles_max": float(np.max(phi / (2.0 * math.pi))),
        },
        "integration": {
            "reference_rows_n": len(reference_rows),
            "observed_rows_n": len(observed_rows),
            "numeric_target_rows_n": numeric_target_n,
            "numeric_coverage_ratio": coverage_ratio_legacy,
            "numeric_coverage_ratio_targeted": coverage_ratio_targeted,
            "residual_audit": residual_stats,
            "notes": [
                "This initial freeze includes a relation-level representative point from Mannheim recap.",
                "Primary experiment tables should be added to upgrade residual audit from initial to full.",
                "numeric_coverage_ratio is a legacy total-record ratio; numeric_coverage_ratio_targeted is the gate-ready coverage over numeric-target rows.",
            ],
        },
        "outputs": {
            "png": str(out_png),
            "pdf": str(out_pdf),
            "integration_csv": str(out_catalog_csv),
            "hv_sweep_csv": str(out_hv_csv),
            "complete_analysis_png": str(out_complete_png),
            "complete_analysis_pdf": str(out_complete_pdf),
        },
        "notes": [
            "This script focuses on reproducing the scaling and magnitude of the COW phase shift.",
            "Step 7.16.11 adds frozen integration I/F (catalog + H-v sweep + residual-audit plumbing).",
            "Raw per-run fringe-shift tables are required to complete a fully observed residual audit.",
        ],
    }
    out_json = localize_figure_output_path(out_dir / "cow_phase_shift_metrics.json", root=root)
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    out_complete_json = out_dir / "cow_experiment_complete_analysis_metrics.json"
    out_complete_json.write_text(
        json.dumps(
            {
                "generated_utc": _now_iso_utc(),
                "phase": {"phase": 7, "step": "7.16.11", "name": "COW complete data integration and residual audit"},
                "inputs": {
                    "catalog_rows": len(reference_rows),
                    "catalog_csv": str(out_catalog_csv),
                    "hv_sweep_rows": len(hv_rows),
                    "hv_sweep_csv": str(out_hv_csv),
                },
                "metrics": {
                    "numeric_observed_rows": observed_n,
                    "numeric_target_rows": numeric_target_n,
                    "numeric_coverage_ratio": coverage_ratio_legacy,
                    "numeric_coverage_ratio_targeted": coverage_ratio_targeted,
                    "residual_audit": residual_stats,
                    "representative_phi0_cycles": phi0_cycles,
                },
                "outputs": {
                    "complete_analysis_png": str(out_complete_png),
                    "complete_analysis_pdf": str(out_complete_pdf),
                    "cow_phase_shift_metrics_json": str(out_json),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[ok] png : {out_png}")
    print(f"[ok] pdf : {out_pdf}")
    print(f"[ok] json: {out_json}")
    print(f"[ok] csv : {out_catalog_csv}")
    print(f"[ok] csv : {out_hv_csv}")
    print(f"[ok] png : {out_complete_png}")
    print(f"[ok] pdf : {out_complete_pdf}")
    print(f"[ok] json: {out_complete_json}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
