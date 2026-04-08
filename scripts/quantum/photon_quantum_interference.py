"""
目的: 量子 topic の photon quantum interference に対応する公開図・表・監査指標を再生成する。
入力: script 内の既定パラメータと必要な公開データまたは基準値を用いる。
出力: output/public と output/private の canonical artifact を更新する。
前提: 論文本文と README はこの script が出力する公開成果物を正として参照する。
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


from figure_japanese_localizer import enable_japanese_figure_localization
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.utils.plot_style import apply_wavep_figure_layout, get_wavep_font_size

enable_japanese_figure_localization()

# クラス: `Config` の責務と境界条件を定義する。
@dataclass(frozen=True)
class Config:
    # Single-photon interference (Mach–Zehnder): representative telecom wavelength.
    wavelength_nm: float = 1550.0
    # From Kimura et al. (quant-ph/0403104v2): "observed fringe visibility was more than 80% after 150 km transmission".
    single_photon_visibility_lower: float = 0.80

    # HOM (quantum dots; arXiv:2106.03871v2): reported visibilities at two temporal separations.
    hom_d_ns: tuple[float, ...] = (13.0, 1_000.0)  # 13 ns and 1 µs
    hom_visibility: tuple[float, ...] = (0.982, 0.987)
    hom_visibility_err_plus: tuple[float, ...] = (0.013, 0.013)
    hom_visibility_err_minus: tuple[float, ...] = (0.013, 0.020)

    # Zenodo (6371310): Extended Data Fig.3 frequency noise PSD (low-frequency dominated).
    zenodo_record: str = "6371310"
    psd_zip_rel: str = "data/quantum/sources/zenodo_6371310/DataExfig3.zip"
    psd_csv_name: str = "DataExfig3b.csv"

    # Squeezed light benchmark (Vahlbruch et al. arXiv:0706.1431v1): 10 dB.
    observed_squeezing_db: float = 10.0


# 関数: `_sigma_path_nm_from_visibility` の入出力契約と処理意図を定義する。

def _sigma_path_nm_from_visibility(v: float, *, wavelength_nm: float) -> float:
    # For Gaussian path-length noise σL:
    # V = exp(-σφ^2/2), σφ = 2π σL / λ.
    if v <= 0.0 or v >= 1.0:
        return float("nan")

    return float(wavelength_nm / (2.0 * math.pi) * math.sqrt(-2.0 * math.log(v)))


# 関数: `_variance_ratio_from_db` の入出力契約と処理意図を定義する。

def _variance_ratio_from_db(db: float) -> float:
    return float(10.0 ** (-db / 10.0))


# 関数: `_read_psd_from_zip` の入出力契約と処理意図を定義する。

def _read_psd_from_zip(zip_path: Path, *, csv_name: str) -> tuple[np.ndarray, np.ndarray]:
    # 条件分岐: `not zip_path.exists()` を満たす経路を評価する。
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        with z.open(csv_name, "r") as f:
            header = f.readline()
            _ = header
            freq: list[float] = []
            psd: list[float] = []
            for line in f:
                s = line.decode("utf-8", "replace").strip()
                # 条件分岐: `not s` を満たす経路を評価する。
                if not s:
                    continue

                parts = s.split(",")
                # 条件分岐: `len(parts) != 2` を満たす経路を評価する。
                if len(parts) != 2:
                    continue

                try:
                    freq.append(float(parts[0]))
                    psd.append(float(parts[1]))
                except ValueError:
                    continue

    return np.asarray(freq, dtype=float), np.asarray(psd, dtype=float)


# 関数: `_conservative_sigma` の入出力契約と処理意図を定義する。

def _conservative_sigma(plus: float, minus: float) -> float:
    return float(max(abs(plus), abs(minus)))


# 関数: `_interp_log_psd` の入出力契約と処理意図を定義する。

def _interp_log_psd(freq_hz: np.ndarray, psd: np.ndarray, target_hz: float) -> float:
    # 条件分岐: `target_hz <= 0` を満たす経路を評価する。
    if target_hz <= 0:
        return float("nan")

    mask = (freq_hz > 0) & (psd > 0)
    # 条件分岐: `np.count_nonzero(mask) < 2` を満たす経路を評価する。
    if np.count_nonzero(mask) < 2:
        return float("nan")

    xf = np.log10(freq_hz[mask])
    yf = np.log10(psd[mask])
    xt = math.log10(target_hz)
    yt = np.interp(xt, xf, yf)
    return float(10.0 ** yt)


# 関数: `_build_hom_squeezed_light_audit` の入出力契約と処理意図を定義する。

def _build_hom_squeezed_light_audit(
    *,
    out_dir: Path,
    d_ns: np.ndarray,
    v_hom: np.ndarray,
    v_hom_plus: np.ndarray,
    v_hom_minus: np.ndarray,
    var_ratio: float,
    eta_lower_if_perfect_intrinsic: float,
    f_hz: np.ndarray,
    psd: np.ndarray,
) -> tuple[Path, Path, Path]:
    rows: list[dict] = []

    classical_limit = 0.5
    for i in range(d_ns.size):
        sigma = _conservative_sigma(float(v_hom_plus[i]), float(v_hom_minus[i]))
        z = (float(v_hom[i]) - classical_limit) / sigma if sigma > 0 else float("nan")
        rows.append(
            {
                "channel": f"hom_visibility_d{int(round(float(d_ns[i])))}ns",
                "observable": "visibility",
                "metric_name": "z_vs_classical_0p5",
                "metric_value": z,
                "threshold_3sigma": 3.0,
                "pass_3sigma": bool(np.isfinite(z) and z >= 3.0),
                "current_precision": sigma,
                "required_precision_3sigma": max((float(v_hom[i]) - classical_limit) / 3.0, 0.0),
                "source": "arXiv:2106.03871v2 (reported HOM visibilities)",
                "data_status": "direct",
            }
        )

    # 条件分岐: `d_ns.size >= 2` を満たす経路を評価する。

    if d_ns.size >= 2:
        sigma_delta = math.sqrt(
            _conservative_sigma(float(v_hom_plus[0]), float(v_hom_minus[0])) ** 2
            + _conservative_sigma(float(v_hom_plus[-1]), float(v_hom_minus[-1])) ** 2
        )
        delta_vis = abs(float(v_hom[-1]) - float(v_hom[0]))
        z_delta = delta_vis / sigma_delta if sigma_delta > 0 else float("nan")
        rows.append(
            {
                "channel": "hom_delay_dependence",
                "observable": "abs(Delta_visibility)",
                "metric_name": "z_delta_between_13ns_1us",
                "metric_value": z_delta,
                "threshold_3sigma": 3.0,
                "pass_3sigma": bool(np.isfinite(z_delta) and z_delta <= 3.0),
                "current_precision": sigma_delta,
                "required_precision_3sigma": 3.0 * sigma_delta,
                "source": "arXiv:2106.03871v2 (13 ns vs 1 us)",
                "data_status": "direct",
            }
        )

    rows.append(
        {
            "channel": "squeezed_light_10db",
            "observable": "variance_ratio",
            "metric_name": "ratio_vs_3dB_threshold",
            "metric_value": float(var_ratio),
            "threshold_3sigma": 0.5,
            "pass_3sigma": bool(var_ratio <= 0.5),
            "current_precision": float(var_ratio),
            "required_precision_3sigma": 0.5,
            "source": "arXiv:0706.1431v1",
            "data_status": "direct",
        }
    )

    psd_10k = _interp_log_psd(f_hz, psd, 1.0e4)
    psd_100k = _interp_log_psd(f_hz, psd, 1.0e5)
    lf_hf_ratio = psd_10k / psd_100k if (psd_10k > 0 and psd_100k > 0) else float("nan")
    rows.append(
        {
            "channel": "noise_psd_shape",
            "observable": "PSD(10kHz)/PSD(100kHz)",
            "metric_name": "lf_to_hf_ratio",
            "metric_value": lf_hf_ratio,
            "threshold_3sigma": 1.0,
            "pass_3sigma": bool(np.isfinite(lf_hf_ratio) and lf_hf_ratio >= 1.0),
            "current_precision": None,
            "required_precision_3sigma": None,
            "source": "Zenodo 6371310 / DataExfig3b.csv",
            "data_status": "direct",
        }
    )

    out_csv = out_dir / "hom_squeezed_light_unified_audit_summary.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, dpi=140)
    apply_wavep_figure_layout(fig, template="part2_four_panel_spacious")
    fig.set_size_inches(fig.get_figwidth(), 9.30, forward=True)
    fig.subplots_adjust(left=0.120, right=0.985, top=0.905, bottom=0.085, hspace=0.80)

    ax0 = axes[0]
    ax0.errorbar(
        d_ns,
        v_hom * 100.0,
        yerr=np.vstack([v_hom_minus, v_hom_plus]) * 100.0,
        fmt="o",
        ms=6,
        lw=1.4,
        capsize=4,
    )
    panel_title_font = get_wavep_font_size("title") * 0.82
    axis_label_font = get_wavep_font_size("axis")
    tick_font = get_wavep_font_size("tick")
    legend_font = get_wavep_font_size("legend")
    note_font = get_wavep_font_size("note")
    suptitle_font = get_wavep_font_size("suptitle") + 2.0

    ax0.axhline(50.0, color="0.25", ls="--", lw=1.2)
    ax0.set_xscale("log")
    ax0.set_xlabel("D (ns)", fontsize=axis_label_font)
    ax0.set_ylabel("可視度 (%)", fontsize=axis_label_font)
    ax0.set_title("HOM 可視度（50% 基準付き）", fontsize=panel_title_font, pad=5.0)
    ax0.grid(True, which="both", ls=":", lw=0.6, alpha=0.7)
    ax0.tick_params(axis="both", labelsize=tick_font)

    ax1 = axes[1]
    hom_rows = [r for r in rows if str(r["channel"]).startswith("hom_visibility_d")]
    labels = [str(r["channel"]).replace("hom_visibility_", "") for r in hom_rows]
    z_vals = [float(r["metric_value"]) for r in hom_rows]
    ax1.bar(labels, z_vals, color="#ff7f0e")
    ax1.axhline(3.0, color="0.25", ls="--", lw=1.2)
    ax1.set_ylabel("z（0.5 基準）", fontsize=axis_label_font)
    ax1.set_title("HOM の有意性", fontsize=panel_title_font, pad=5.0)
    ax1.grid(True, axis="y", ls=":", lw=0.6, alpha=0.7)
    ax1.tick_params(axis="both", labelsize=tick_font)

    ax2 = axes[2]
    ax2.bar(["variance ratio"], [var_ratio], color="#2ca02c")
    ax2.axhline(0.5, color="0.25", ls="--", lw=1.2, label="3 dB 基準")
    ax2.set_ylabel("分散比", fontsize=axis_label_font)
    ax2.set_title("スクイージング規模", fontsize=panel_title_font, pad=5.0)
    ax2.grid(True, axis="y", ls=":", lw=0.6, alpha=0.7)
    ax2.legend(frameon=True, fontsize=legend_font, loc="upper right")
    ax2.text(
        0.02,
        0.05,
        f"loss-only: η >= {eta_lower_if_perfect_intrinsic:.3f}",
        transform=ax2.transAxes,
        fontsize=note_font,
    )
    ax2.tick_params(axis="both", labelsize=tick_font)

    ax3 = axes[3]
    # 条件分岐: `np.isfinite(psd_10k) and np.isfinite(psd_100k)` を満たす経路を評価する。
    if np.isfinite(psd_10k) and np.isfinite(psd_100k):
        ax3.bar(["PSD@10kHz", "PSD@100kHz"], [psd_10k, psd_100k], color=["#1f77b4", "#7f7f7f"])

    ax3.set_yscale("log")
    ax3.set_ylabel("PSD (arb.)", fontsize=axis_label_font)
    ax3.set_title("ノイズ PSD の規模指標", fontsize=panel_title_font, pad=5.0)
    ax3.grid(True, axis="y", ls=":", lw=0.6, alpha=0.7)
    ax3.tick_params(axis="both", labelsize=tick_font)

    fig.suptitle("HOM + スクイーズド光の統合監査", y=0.994, fontsize=suptitle_font)

    out_png = out_dir / "hom_squeezed_light_unified_audit.png"
    out_pdf = out_dir / "hom_squeezed_light_unified_audit.pdf"
    prev_disable_normalize = os.environ.get("WAVEP_MPL_DISABLE_CANVAS_NORMALIZE")
    os.environ["WAVEP_MPL_DISABLE_CANVAS_NORMALIZE"] = "1"
    try:
        with plt.rc_context({"savefig.bbox": "standard", "savefig.pad_inches": 0.0}):
            fig.savefig(out_png)
            fig.savefig(out_pdf)
    finally:
        if prev_disable_normalize is None:
            os.environ.pop("WAVEP_MPL_DISABLE_CANVAS_NORMALIZE", None)
        else:
            os.environ["WAVEP_MPL_DISABLE_CANVAS_NORMALIZE"] = prev_disable_normalize

    plt.close(fig)

    out_json = out_dir / "hom_squeezed_light_unified_audit_metrics.json"
    out_json.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "phase": 7,
                "step": "7.16.14",
                "title": "HOM and squeezed-light unified audit",
                "channels_n": len(rows),
                "rows": rows,
                "derived": {
                    "hom_classical_limit": classical_limit,
                    "hom_delta_visibility_13ns_to_1us": (
                        abs(float(v_hom[-1]) - float(v_hom[0])) if d_ns.size >= 2 else None
                    ),
                    "squeezing_variance_ratio": float(var_ratio),
                    "squeezing_eta_lower_if_perfect_intrinsic": float(eta_lower_if_perfect_intrinsic),
                    "psd_at_10khz": psd_10k,
                    "psd_at_100khz": psd_100k,
                    "psd_lf_to_hf_ratio": lf_hf_ratio,
                },
                "outputs": {"summary_csv": str(out_csv), "summary_png": str(out_png), "summary_pdf": str(out_pdf)},
                "notes": [
                    "HOM significance is measured against the classical 50% reference and delay-stability between 13 ns and 1 us.",
                    "Squeezing row uses a conservative operational threshold at 3 dB (variance ratio <= 0.5).",
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return out_csv, out_json, out_png


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    cfg = Config()
    root = ROOT
    out_dir = root / "output" / "public" / "quantum"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Panel A: single-photon interference (visibility → path-length noise upper bound) ---
    lam_nm = cfg.wavelength_nm
    v0 = cfg.single_photon_visibility_lower
    sigma_nm = _sigma_path_nm_from_visibility(v0, wavelength_nm=lam_nm)

    # --- Panel B: HOM visibility vs temporal separation (reported) ---
    d_ns = np.asarray(cfg.hom_d_ns, dtype=float)
    v_hom = np.asarray(cfg.hom_visibility, dtype=float)
    v_hom_plus = np.asarray(cfg.hom_visibility_err_plus, dtype=float)
    v_hom_minus = np.asarray(cfg.hom_visibility_err_minus, dtype=float)

    # --- Panel C: low-frequency noise PSD (Zenodo raw data) ---
    psd_zip = root / Path(cfg.psd_zip_rel)
    f_hz, psd = _read_psd_from_zip(psd_zip, csv_name=cfg.psd_csv_name)

    # --- Squeezing mapping ---
    sq_db = cfg.observed_squeezing_db
    var_ratio = _variance_ratio_from_db(sq_db)
    eta_lower_if_perfect_intrinsic = 1.0 - var_ratio  # since V_obs >= 1-η if V_intrinsic→0

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, dpi=170)
    apply_wavep_figure_layout(fig, template="part2_three_panel_tall")
    fig.set_size_inches(fig.get_figwidth(), 7.20, forward=True)
    fig.subplots_adjust(left=0.155, right=0.985, top=0.920, bottom=0.095, hspace=0.66)
    panel_title_font = get_wavep_font_size("title") * 0.82
    axis_label_font = get_wavep_font_size("axis")
    tick_font = get_wavep_font_size("tick")
    legend_font = get_wavep_font_size("legend")
    note_font = get_wavep_font_size("note")
    suptitle_font = get_wavep_font_size("suptitle") + 2.0

    # A
    ax0 = axes[0]
    v_grid = np.linspace(0.55, 0.999, 240)
    sig_grid = np.asarray([_sigma_path_nm_from_visibility(v, wavelength_nm=lam_nm) for v in v_grid], dtype=float)
    ax0.plot(v_grid, sig_grid, lw=2.0, label=f"λ={lam_nm:.0f} nm")
    ax0.scatter([v0], [sigma_nm], s=40, zorder=5, label=f"Kimura+2004: V≥{v0:.2f}")
    ax0.set_xlabel("可視度 V", fontsize=axis_label_font)
    ax0.set_ylabel("等価な経路長ノイズ σL (nm)", fontsize=axis_label_font)
    ax0.set_title("単一光子干渉：V → σL", fontsize=panel_title_font, pad=5.0)
    ax0.grid(True, ls=":", lw=0.6, alpha=0.6)
    ax0.legend(frameon=True, fontsize=legend_font, loc="upper right")
    ax0.tick_params(axis="both", labelsize=tick_font)

    # B
    ax1 = axes[1]
    ax1.errorbar(
        d_ns,
        v_hom * 100.0,
        yerr=np.vstack([v_hom_minus, v_hom_plus]) * 100.0,
        fmt="o",
        ms=6,
        lw=1.5,
        capsize=4,
        label="reported（QD2; corrected）",
    )
    ax1.set_xscale("log")
    ax1.set_xlabel("分離時間 D (ns)", fontsize=axis_label_font)
    ax1.set_ylabel("HOM 可視度 (%)", fontsize=axis_label_font)
    ax1.set_title("HOM：分離時間に対する可視度", fontsize=panel_title_font, pad=5.0)
    ax1.grid(True, which="both", ls=":", lw=0.6, alpha=0.6)
    ax1.set_ylim(90.0, 100.0)
    ax1.legend(frameon=True, fontsize=legend_font, loc="lower left")
    ax1.text(
        0.02,
        0.98,
        "定義: V=1−(C∥/C⊥) at zero delay\n（Methods, arXiv:2106.03871v2）",
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=note_font,
    )
    ax1.tick_params(axis="both", labelsize=tick_font)

    # C
    ax2 = axes[2]
    ax2.plot(f_hz, psd, lw=1.2, label="Zenodo 6371310（ExData Fig.3b）")
    ax2.axvline(1e4, color="k", lw=1.0, ls="--", alpha=0.7, label="10^4 Hz")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("周波数 (Hz)", fontsize=axis_label_font)
    ax2.set_ylabel("PSD (arb. units)", fontsize=axis_label_font)
    ax2.set_title("低周波ノイズ PSD（識別不能性の proxy）", fontsize=panel_title_font, pad=5.0)
    ax2.grid(True, which="both", ls=":", lw=0.6, alpha=0.6)
    ax2.legend(frameon=True, fontsize=legend_font, loc="lower left")
    ax2.text(
        0.02,
        0.98,
        f"スクイージング: {sq_db:.1f} dB → 分散比={var_ratio:.3f}\n"
        f"loss-only bound: η ≥ {eta_lower_if_perfect_intrinsic:.3f}",
        transform=ax2.transAxes,
        va="top",
        ha="left",
        fontsize=note_font,
    )
    ax2.tick_params(axis="both", labelsize=tick_font)

    fig.suptitle("光子干渉の観測量（可視度・HOM・スクイージング/ノイズ）", y=0.994, fontsize=suptitle_font)

    out_png = out_dir / "photon_quantum_interference.png"
    out_pdf = out_dir / "photon_quantum_interference.pdf"
    prev_disable_normalize = os.environ.get("WAVEP_MPL_DISABLE_CANVAS_NORMALIZE")
    os.environ["WAVEP_MPL_DISABLE_CANVAS_NORMALIZE"] = "1"
    try:
        with plt.rc_context({"savefig.bbox": "standard", "savefig.pad_inches": 0.0}):
            fig.savefig(out_png)
            fig.savefig(out_pdf)
    finally:
        if prev_disable_normalize is None:
            os.environ.pop("WAVEP_MPL_DISABLE_CANVAS_NORMALIZE", None)
        else:
            os.environ["WAVEP_MPL_DISABLE_CANVAS_NORMALIZE"] = prev_disable_normalize

    plt.close(fig)

    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase": 7,
        "step": "7.7",
        "sources": [
            {
                "topic": "single_photon_interference_150km",
                "url": "https://arxiv.org/abs/quant-ph/0403104",
                "local_pdf": str(root / "data" / "quantum" / "sources" / "arxiv_quant-ph_0403104v2.pdf"),
                "key_value": {"visibility": "more than 80% after 150 km"},
            },
            {
                "topic": "hom_quantum_dots_reported",
                "url": "https://arxiv.org/abs/2106.03871",
                "local_pdf": str(root / "data" / "quantum" / "sources" / "arxiv_2106.03871v2.pdf"),
                "key_values": {
                    "V_13ns": 0.982,
                    "V_13ns_err": 0.013,
                    "V_1us": 0.987,
                    "V_1us_err_plus": 0.013,
                    "V_1us_err_minus": 0.020,
                },
                "note": "Values taken from the paper text (QD2; corrected visibility).",
            },
            {
                "topic": "indistinguishability_noise_psd",
                "url": "https://arxiv.org/abs/2106.03871",
                "zenodo": {"doi": "10.5281/zenodo.6371310", "record": cfg.zenodo_record},
                "psd": {"zip": str(psd_zip), "csv": cfg.psd_csv_name},
            },
            {
                "topic": "squeezed_light_10db",
                "url": "https://arxiv.org/abs/0706.1431",
                "local_pdf": str(root / "data" / "quantum" / "sources" / "arxiv_0706.1431v1.pdf"),
                "key_value": {"squeezing_db": float(sq_db)},
            },
        ],
        "config": {
            "wavelength_nm": cfg.wavelength_nm,
            "single_photon_visibility_lower": cfg.single_photon_visibility_lower,
            "hom_d_ns": list(map(float, cfg.hom_d_ns)),
            "hom_visibility": list(map(float, cfg.hom_visibility)),
            "hom_visibility_err_plus": list(map(float, cfg.hom_visibility_err_plus)),
            "hom_visibility_err_minus": list(map(float, cfg.hom_visibility_err_minus)),
            "psd_zip_rel": cfg.psd_zip_rel,
            "psd_csv_name": cfg.psd_csv_name,
            "observed_squeezing_db": float(sq_db),
        },
        "single_photon_interference": {
            "visibility_lower": float(v0),
            "sigma_path_nm_from_visibility": sigma_nm,
            "model": "V=exp(-σφ^2/2), σφ=2π σL/λ (Gaussian path noise)",
        },
        "hom": {
            "d_ns": list(map(float, d_ns.tolist())),
            "visibility": list(map(float, v_hom.tolist())),
            "visibility_err_plus": list(map(float, v_hom_plus.tolist())),
            "visibility_err_minus": list(map(float, v_hom_minus.tolist())),
        },
        "psd": {
            "f_hz_min": float(np.min(f_hz)) if f_hz.size else float("nan"),
            "f_hz_max": float(np.max(f_hz)) if f_hz.size else float("nan"),
            "n_points": int(f_hz.size),
            "zip": str(psd_zip),
            "csv": cfg.psd_csv_name,
        },
        "squeezing": {
            "observed_squeezing_db": float(sq_db),
            "variance_ratio": var_ratio,
            "eta_lower_if_perfect_intrinsic": eta_lower_if_perfect_intrinsic,
            "model_note": "Loss-only inequality: V_obs >= 1-η (since V_intrinsic>=0).",
        },
        "outputs": {"png": str(out_png), "pdf": str(out_pdf)},
        "notes": [
            "This script fixes key observables and provides simple mappings; it is not a full quantum-optics derivation.",
            "HOM values are taken from the paper text; Zenodo is used for the noise PSD shown as Extended Data Fig.3b.",
        ],
    }
    out_json = out_dir / "photon_quantum_interference_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    unified_csv, unified_json, unified_png = _build_hom_squeezed_light_audit(
        out_dir=out_dir,
        d_ns=d_ns,
        v_hom=v_hom,
        v_hom_plus=v_hom_plus,
        v_hom_minus=v_hom_minus,
        var_ratio=var_ratio,
        eta_lower_if_perfect_intrinsic=eta_lower_if_perfect_intrinsic,
        f_hz=f_hz,
        psd=psd,
    )

    print(f"[ok] png : {out_png}")
    print(f"[ok] pdf : {out_pdf}")
    print(f"[ok] json: {out_json}")
    print(f"[ok] csv : {unified_csv}")
    print(f"[ok] json: {unified_json}")
    print(f"[ok] png : {unified_png}")


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
