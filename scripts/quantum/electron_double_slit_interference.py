from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Config:
    # Bach et al. 2012 (arXiv:1210.6243v1): electron beam energy.
    energy_eV: float = 600.0

    # Double-slit geometry (two equal slits):
    slit_width_m: float = 50e-9  # 50 nm
    slit_separation_m: float = 280e-9  # center-to-center 280 nm

    # Angle grid for far-field pattern (Fraunhofer), in milliradians.
    theta_mrad_min: float = -2.0
    theta_mrad_max: float = 2.0
    n_theta: int = 4001


def electron_de_broglie_wavelength_m(*, energy_eV: float) -> float:
    """
    Relativistic de Broglie wavelength for electrons accelerated to kinetic energy K = eV:

      p c = sqrt(K^2 + 2 K m c^2)
      λ = h / p
    """
    e_J_per_eV = 1.602176634e-19
    c = 299_792_458.0
    m_e = 9.1093837015e-31
    h = 6.62607015e-34

    K = float(energy_eV) * e_J_per_eV
    p = math.sqrt(K * K + 2.0 * K * m_e * c * c) / c
    return float(h / p)


def sinc(x: np.ndarray) -> np.ndarray:
    # numpy.sinc is sin(pi x)/(pi x); we want sin(x)/x.
    y = np.ones_like(x, dtype=float)
    nz = x != 0.0
    y[nz] = np.sin(x[nz]) / x[nz]
    return y


def _load_json(path: Path) -> dict | None:
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        return None

    return json.loads(path.read_text(encoding="utf-8"))


def _channel_row(
    *,
    channel: str,
    observable: str,
    metric_name: str,
    metric_value: float | None,
    threshold_3sigma: float | None,
    pass_3sigma: bool | None,
    current_precision: float | None,
    required_precision_3sigma: float | None,
    source: str,
    data_status: str,
) -> dict:
    return {
        "channel": channel,
        "observable": observable,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "threshold_3sigma": threshold_3sigma,
        "pass_3sigma": pass_3sigma,
        "current_precision": current_precision,
        "required_precision_3sigma": required_precision_3sigma,
        "source": source,
        "data_status": data_status,
    }


def build_matter_wave_interference_precision_audit(
    *,
    root: Path,
    out_dir: Path,
    electron_metrics: dict,
) -> tuple[Path, Path]:
    alpha_metrics_path = out_dir / "de_broglie_precision_alpha_consistency_metrics.json"
    atom_audit_path = out_dir / "atom_interferometer_unified_audit_metrics.json"
    molecular_metrics_path = out_dir / "molecular_isotopic_scaling_metrics.json"

    alpha_metrics = _load_json(alpha_metrics_path)
    atom_audit_metrics = _load_json(atom_audit_path)
    molecular_metrics = _load_json(molecular_metrics_path)

    rows: list[dict] = []
    notes: list[str] = []
    precision_gap_watch: dict | None = None

    electron_derived = electron_metrics["derived"]
    fringe_mrad = float(electron_derived["fringe_spacing_theta_mrad"])
    envelope_zero_mrad = float(electron_derived["envelope_first_zero_theta_mrad"])
    envelope_to_fringe = envelope_zero_mrad / fringe_mrad if fringe_mrad > 0 else float("nan")
    rows.append(
        _channel_row(
            channel="electron_double_slit",
            observable="fringe_spacing_theta_mrad",
            metric_name="envelope_to_fringe_ratio",
            metric_value=envelope_to_fringe,
            threshold_3sigma=1.0,
            pass_3sigma=bool(envelope_to_fringe >= 1.0),
            current_precision=None,
            required_precision_3sigma=None,
            source="Bach et al. 2012 (arXiv:1210.6243v1)",
            data_status="direct",
        )
    )

    alpha_z_abs = None
    epsilon_sigma = None
    # 条件分岐: `alpha_metrics is not None` を満たす経路を評価する。
    if alpha_metrics is not None:
        alpha_derived = alpha_metrics.get("derived", {})
        alpha_z_abs = abs(float(alpha_derived.get("z_score", float("nan"))))
        epsilon_sigma = float(alpha_derived.get("epsilon_mc_sigma", float("nan")))
        rows.append(
            _channel_row(
                channel="atom_recoil_alpha",
                observable="alpha_consistency_z",
                metric_name="abs_z",
                metric_value=alpha_z_abs,
                threshold_3sigma=3.0,
                pass_3sigma=bool(np.isfinite(alpha_z_abs) and alpha_z_abs <= 3.0),
                current_precision=epsilon_sigma,
                required_precision_3sigma=None,
                source="Bouchendira 2008 + Gabrielse 2008",
                data_status="direct",
            )
        )
    else:
        notes.append("Missing de_broglie_precision_alpha_consistency_metrics.json; atom recoil alpha row skipped.")

    atom_precision_ratios: list[float] = []
    atom_precision_labels: list[str] = []
    atom_precision_by_channel: dict[str, float] = {}
    # 条件分岐: `atom_audit_metrics is not None` を満たす経路を評価する。
    if atom_audit_metrics is not None:
        for entry in atom_audit_metrics.get("rows", []):
            req = entry.get("required_precision_3sigma")
            cur = entry.get("current_precision")
            # 条件分岐: `isinstance(req, (int, float)) and isinstance(cur, (int, float)) and req > 0 a...` を満たす経路を評価する。
            if isinstance(req, (int, float)) and isinstance(cur, (int, float)) and req > 0 and cur > 0:
                ratio = float(cur / req)
                channel_name = str(entry.get("channel", "unknown"))
                atom_precision_ratios.append(ratio)
                atom_precision_labels.append(channel_name)
                atom_precision_by_channel[channel_name] = ratio

        # 条件分岐: `atom_precision_ratios` を満たす経路を評価する。

        if atom_precision_ratios:
            ratio_array = np.asarray(atom_precision_ratios, dtype=float)
            ratio_min = float(np.min(ratio_array))
            ratio_median = float(np.median(ratio_array))
            ratio_max = float(np.max(ratio_array))
            preferred_channel = "atom_gravimeter" if "atom_gravimeter" in atom_precision_by_channel else atom_precision_labels[0]
            preferred_ratio = float(atom_precision_by_channel.get(preferred_channel, ratio_min))
            preferred_log10_ratio = float(np.log10(preferred_ratio)) if preferred_ratio > 0 else None
            precision_gap_watch = {
                "observable": "current_over_required_precision_ratio",
                "threshold_3sigma": 1.0,
                "min_ratio": ratio_min,
                "median_ratio": ratio_median,
                "max_ratio": ratio_max,
                "visibility_reference_channel": preferred_channel,
                "visibility_reference_ratio": preferred_ratio,
                "visibility_reference_log10_ratio": preferred_log10_ratio,
                "visibility_reference_log10_threshold": 1.0,
                "pass_if_min_le_threshold": bool(ratio_min <= 1.0),
                "pass_if_median_le_threshold": bool(ratio_median <= 1.0),
                "pass_if_visibility_reference_log10_le_threshold": bool(
                    preferred_log10_ratio is not None and preferred_log10_ratio <= 1.0
                ),
                "rows": [
                    {"channel": atom_precision_labels[idx], "ratio": float(atom_precision_ratios[idx])}
                    for idx in range(len(atom_precision_ratios))
                ],
                "source": "atom_interferometer_unified_audit (Step 7.16.12)",
            }
            notes.append(
                "Atom-interferometer precision gap is tracked as diagnostics; "
                "visibility reference uses atom_gravimeter log10(current/required) with a one-decade watch threshold."
            )
    else:
        notes.append("Missing atom_interferometer_unified_audit_metrics.json; atom precision-gap row skipped.")

    molecular_dev_abs: list[float] = []
    # 条件分岐: `molecular_metrics is not None` を満たす経路を評価する。
    if molecular_metrics is not None:
        for entry in molecular_metrics.get("rows", []):
            for key in ("omega_e_ratio_meas_over_pred", "B_e_ratio_meas_over_pred"):
                ratio = entry.get(key)
                # 条件分岐: `isinstance(ratio, (int, float)) and ratio > 0` を満たす経路を評価する。
                if isinstance(ratio, (int, float)) and ratio > 0:
                    molecular_dev_abs.append(abs(float(ratio) - 1.0))

        # 条件分岐: `molecular_dev_abs` を満たす経路を評価する。

        if molecular_dev_abs:
            dev_arr = np.asarray(molecular_dev_abs, dtype=float)
            dev_sigma = float(np.std(dev_arr, ddof=1)) if dev_arr.size > 1 else 0.0
            dev_max = float(np.max(dev_arr))
            z_max = float(dev_max / dev_sigma) if dev_sigma > 0 else float("nan")
            rows.append(
                _channel_row(
                    channel="molecular_isotopic_scaling",
                    observable="abs(meas/pred-1)",
                    metric_name="z_max",
                    metric_value=z_max,
                    threshold_3sigma=3.0,
                    pass_3sigma=bool(np.isfinite(z_max) and z_max <= 3.0),
                    current_precision=dev_max,
                    required_precision_3sigma=(3.0 * dev_sigma if dev_sigma > 0 else None),
                    source="NIST/WebBook isotopic reduced-mass scaling (Step 7.12)",
                    data_status="proxy",
                )
            )
    else:
        notes.append("Missing molecular_isotopic_scaling_metrics.json; molecular row skipped.")

    summary_csv = out_dir / "matter_wave_interference_precision_audit_summary.csv"
    csv_fields = [
        "channel",
        "observable",
        "metric_name",
        "metric_value",
        "threshold_3sigma",
        "pass_3sigma",
        "current_precision",
        "required_precision_3sigma",
        "source",
        "data_status",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), dpi=140)

    ax0 = axes[0, 0]
    ax0.bar(
        ["fringe Δθ", "envelope θ0"],
        [fringe_mrad, envelope_zero_mrad],
        color=["#1f77b4", "#7f7f7f"],
    )
    ax0.set_ylabel("mrad")
    ax0.set_title("Electron double-slit angular scales")
    ax0.grid(True, axis="y", ls=":", lw=0.6, alpha=0.7)

    ax1 = axes[0, 1]
    z_labels: list[str] = []
    z_values: list[float] = []
    # 条件分岐: `alpha_z_abs is not None and np.isfinite(alpha_z_abs)` を満たす経路を評価する。
    if alpha_z_abs is not None and np.isfinite(alpha_z_abs):
        z_labels.append("atom α-consistency")
        z_values.append(float(alpha_z_abs))

    for row in rows:
        # 条件分岐: `row["channel"] == "molecular_isotopic_scaling" and isinstance(row["metric_val...` を満たす経路を評価する。
        if row["channel"] == "molecular_isotopic_scaling" and isinstance(row["metric_value"], (int, float)):
            # 条件分岐: `np.isfinite(float(row["metric_value"]))` を満たす経路を評価する。
            if np.isfinite(float(row["metric_value"])):
                z_labels.append("molecular scaling")
                z_values.append(float(row["metric_value"]))

    # 条件分岐: `z_values` を満たす経路を評価する。

    if z_values:
        ax1.bar(z_labels, z_values, color=["#ff7f0e", "#2ca02c"][: len(z_values)])

    ax1.axhline(3.0, color="0.25", ls="--", lw=1.2)
    ax1.set_ylabel("z")
    ax1.set_title("Cross-channel consistency (|z|)")
    ax1.grid(True, axis="y", ls=":", lw=0.6, alpha=0.7)

    ax2 = axes[1, 0]
    # 条件分岐: `atom_precision_ratios` を満たす経路を評価する。
    if atom_precision_ratios:
        x = np.arange(len(atom_precision_ratios))
        ax2.bar(x, atom_precision_ratios, color="#9467bd")
        ax2.set_xticks(x)
        ax2.set_xticklabels(atom_precision_labels, rotation=20, ha="right")

    ax2.axhline(1.0, color="0.25", ls="--", lw=1.2)
    ax2.set_yscale("log")
    ax2.set_ylabel("current / required(3σ)")
    ax2.set_title("Atom-interferometer precision gap")
    ax2.grid(True, axis="y", ls=":", lw=0.6, alpha=0.7)

    ax3 = axes[1, 1]
    # 条件分岐: `molecular_dev_abs` を満たす経路を評価する。
    if molecular_dev_abs:
        ax3.hist(molecular_dev_abs, bins=min(8, max(3, len(molecular_dev_abs))), color="#17becf", alpha=0.85)

    ax3.set_xlabel("abs(meas/pred - 1)")
    ax3.set_ylabel("count")
    ax3.set_title("Molecular isotopic scaling residuals")
    ax3.grid(True, axis="y", ls=":", lw=0.6, alpha=0.7)

    fig.suptitle("Matter-wave interference precision audit", y=0.98)
    fig.tight_layout()

    out_png = out_dir / "matter_wave_interference_precision_audit.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase": 7,
        "step": "7.16.13",
        "title": "Matter-wave interference precision audit (electron/atom/molecule)",
        "channels_n": len(rows),
        "rows": rows,
        "inputs": {
            "electron_metrics": str(out_dir / "electron_double_slit_interference_metrics.json"),
            "alpha_metrics": str(alpha_metrics_path),
            "atom_audit_metrics": str(atom_audit_path),
            "molecular_metrics": str(molecular_metrics_path),
        },
        "outputs": {
            "summary_csv": str(summary_csv),
            "summary_png": str(out_png),
        },
        "precision_gap_watch": precision_gap_watch,
        "notes": notes
        + [
            "Molecular channel is a proxy audit based on isotopic reduced-mass scaling because a unified public C60/C70 raw interference cache is not frozen yet.",
            "Step 7.16.13 freezes a cross-particle I/F with available primary-backed outputs; fullerene raw-integration can be appended without changing this schema.",
        ],
    }
    out_json = out_dir / "matter_wave_interference_precision_audit_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    return summary_csv, out_json


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    lam = electron_de_broglie_wavelength_m(energy_eV=cfg.energy_eV)

    theta_mrad = np.linspace(cfg.theta_mrad_min, cfg.theta_mrad_max, cfg.n_theta)
    theta_rad = theta_mrad * 1e-3
    sin_theta = np.sin(theta_rad)

    a = float(cfg.slit_width_m)
    d = float(cfg.slit_separation_m)

    # Single-slit envelope: sinc^2(π a sinθ / λ)
    beta = math.pi * a * sin_theta / lam
    envelope = sinc(beta) ** 2

    # Double-slit interference term: cos^2(π d sinθ / λ)
    delta = math.pi * d * sin_theta / lam
    interference = (np.cos(delta) ** 2).astype(float)

    p1 = envelope.copy()
    p2 = envelope.copy()
    p12 = 4.0 * envelope * interference  # coherent sum of equal slits (intensity)
    p1p2 = p1 + p2

    # Normalize to peak=1 for display.
    def _norm(y: np.ndarray) -> np.ndarray:
        m = float(np.max(y)) if y.size else 1.0
        return y / m if m > 0 else y

    p1_n = _norm(p1)
    p1p2_n = _norm(p1p2)
    p12_n = _norm(p12)

    # Derived scales (small-angle approximation).
    theta_fringe_rad = lam / d  # Δθ ≈ λ/d
    theta_envelope_zero_rad = lam / a  # first zero of sinc: θ ≈ λ/a

    # Plot
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10.8, 5.4), dpi=150)
    ax.plot(theta_mrad, p12_n, lw=2.0, label="P12 (double-slit; coherent)")
    ax.plot(theta_mrad, p1p2_n, lw=1.6, ls="--", label="P1+P2 (sum of single-slit)")
    ax.plot(theta_mrad, p1_n, lw=1.2, ls=":", label="P1 (single-slit envelope)")

    ax.axhline(0.0, color="0.25", lw=1.0)
    ax.set_xlabel("diffraction angle θ (mrad)")
    ax.set_ylabel("normalized intensity (arb.)")
    ax.set_title("Electron double-slit diffraction (arXiv:1210.6243v1; 600 eV; 50 nm slits, 280 nm sep.)")
    ax.grid(True, ls=":", lw=0.6, alpha=0.7)
    ax.legend(frameon=True, fontsize=9, loc="upper right")

    note = (
        f"λ_e(600 eV; rel)≈{lam*1e12:.2f} pm.  "
        f"fringe spacing Δθ≈λ/d≈{theta_fringe_rad*1e3:.3f} mrad.  "
        f"envelope first zero θ≈λ/a≈{theta_envelope_zero_rad*1e3:.3f} mrad."
    )
    fig.text(0.01, -0.02, note, fontsize=9)
    fig.tight_layout()

    out_png = out_dir / "electron_double_slit_interference.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    src_pdf = root / "data" / "quantum" / "sources" / "arxiv_1210.6243v1.pdf"
    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source": {
            "reference": "Bach et al., 'Controlled double-slit electron diffraction' (arXiv:1210.6243v1)",
            "url": "https://arxiv.org/abs/1210.6243",
            "local_pdf": str(src_pdf),
            "local_pdf_sha256": "3FF2F5358B25586E10469081D9B0A6E07671CCED9D3F3A5074E55F5F6C749C74",
            "paper_parameters": {
                "electron_energy_eV": 600.0,
                "slit_width_nm": 50.0,
                "slit_separation_nm": 280.0,
            },
        },
        "config": {
            "energy_eV": cfg.energy_eV,
            "slit_width_m": cfg.slit_width_m,
            "slit_separation_m": cfg.slit_separation_m,
            "theta_mrad_min": cfg.theta_mrad_min,
            "theta_mrad_max": cfg.theta_mrad_max,
            "n_theta": cfg.n_theta,
        },
        "derived": {
            "electron_wavelength_m": lam,
            "electron_wavelength_pm": lam * 1e12,
            "fringe_spacing_theta_rad": theta_fringe_rad,
            "fringe_spacing_theta_mrad": theta_fringe_rad * 1e3,
            "envelope_first_zero_theta_rad": theta_envelope_zero_rad,
            "envelope_first_zero_theta_mrad": theta_envelope_zero_rad * 1e3,
        },
        "outputs": {"png": str(out_png)},
        "notes": [
            "This is a Fraunhofer far-field model (intensity vs diffraction angle).",
            "The experiment uses electron optics (magnification) for imaging; this script focuses on the slit-defined angular pattern.",
        ],
    }
    out_json = out_dir / "electron_double_slit_interference_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    audit_csv, audit_json = build_matter_wave_interference_precision_audit(
        root=root,
        out_dir=out_dir,
        electron_metrics=metrics,
    )

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")
    print(f"[ok] csv : {audit_csv}")
    print(f"[ok] json: {audit_json}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
