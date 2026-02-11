#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


def _repo_root() -> Path:
    return _ROOT


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _set_japanese_font() -> None:
    try:
        import matplotlib as mpl
        import matplotlib.font_manager as fm

        preferred = ["Yu Gothic", "Meiryo", "BIZ UDGothic", "MS Gothic", "Yu Mincho", "MS Mincho"]
        available = {f.name for f in fm.fontManager.ttflist}
        chosen = [name for name in preferred if name in available]
        if not chosen:
            return

        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def _maybe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _std(values: Sequence[float]) -> Optional[float]:
    x = np.array(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return None
    return float(np.std(x, ddof=1))


def _plot_budget(*, title: str, items: Dict[str, float], required: Optional[float], out_png: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(f"matplotlib not available: {e}") from e

    _set_japanese_font()

    def _short_label(s: str) -> str:
        s = str(s)
        s = s.replace("σ(κ) proxy: ", "proxy: ")
        s = s.replace("σ(κ) from ", "from ")
        s = s.replace("tab:", "")
        return s

    pairs = sorted(((str(k), float(v)) for k, v in items.items()), key=lambda kv: kv[1], reverse=True)
    labels = [_short_label(k) for k, _ in pairs]
    vals = [v for _, v in pairs]
    y = np.arange(len(labels), dtype=float)

    # Dynamic height: make long label lists readable in paper figures.
    fig_h = max(5.0, 0.34 * len(labels) + 1.4)
    fig = plt.figure(figsize=(11.8, fig_h))
    ax = fig.add_subplot(1, 1, 1)
    ax.barh(y, vals, color="#1f77b4", alpha=0.82)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("σ(κ) [1σ]")
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    if required is not None and math.isfinite(required) and required > 0:
        ax.axvline(required, color="#d62728", linewidth=2.0, label=f"target σ(κ) ≈ {required:.4f}")
        ax.legend(loc="lower right")
    fig.tight_layout(pad=0.6)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _finite_pos(values: Sequence[Optional[float]]) -> List[float]:
    out: List[float] = []
    for v in values:
        if v is None:
            continue
        try:
            x = float(v)
        except Exception:
            continue
        if math.isfinite(x) and x > 0:
            out.append(x)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    root = _repo_root()
    default_shadow = root / "output" / "eht" / "eht_shadow_compare.json"
    default_ringfit = root / "output" / "eht" / "eht_sgra_ringfit_table_metrics.json"
    default_calib = root / "output" / "eht" / "eht_sgra_calibration_systematics_metrics.json"
    default_var = root / "output" / "eht" / "eht_sgra_variability_noise_model_metrics.json"
    default_mring = root / "output" / "eht" / "eht_sgra_mringfits_table_metrics.json"
    default_paper4 = root / "output" / "eht" / "eht_sgra_paper4_alpha_calibration_metrics.json"
    default_paper4_morph = root / "output" / "eht" / "eht_sgra_paper4_morphology_table_metrics.json"
    default_paper4_thetag = root / "output" / "eht" / "eht_sgra_paper4_thetag_table_metrics.json"
    default_paper4_debiased_noise = root / "output" / "eht" / "eht_sgra_paper4_debiased_noise_table_metrics.json"
    default_paper6 = root / "output" / "eht" / "eht_sgra_paper6_metric_constraints.json"
    default_paper2_gains = root / "output" / "eht" / "eht_sgra_paper2_gain_uncertainties_metrics.json"
    default_paper2_syserr = root / "output" / "eht" / "eht_sgra_paper2_syserr_table_metrics.json"
    default_outdir = root / "output" / "eht"

    ap = argparse.ArgumentParser(description="κ error budget helper (uses Paper III ringfit scatter as a systematic scale).")
    ap.add_argument("--shadow-compare-json", type=str, default=str(default_shadow))
    ap.add_argument("--ringfit-metrics-json", type=str, default=str(default_ringfit))
    ap.add_argument("--calibration-metrics-json", type=str, default=str(default_calib))
    ap.add_argument("--variability-metrics-json", type=str, default=str(default_var))
    ap.add_argument("--mringfits-metrics-json", type=str, default=str(default_mring))
    ap.add_argument("--paper4-alpha-calibration-json", type=str, default=str(default_paper4))
    ap.add_argument("--paper4-morphology-metrics-json", type=str, default=str(default_paper4_morph))
    ap.add_argument("--paper4-thetag-metrics-json", type=str, default=str(default_paper4_thetag))
    ap.add_argument("--paper4-debiased-noise-table-json", type=str, default=str(default_paper4_debiased_noise))
    ap.add_argument("--paper6-metric-constraints-json", type=str, default=str(default_paper6))
    ap.add_argument("--paper2-gain-uncertainties-json", type=str, default=str(default_paper2_gains))
    ap.add_argument("--paper2-syserr-table-json", type=str, default=str(default_paper2_syserr))
    ap.add_argument("--outdir", type=str, default=str(default_outdir))
    args = ap.parse_args(list(argv) if argv is not None else None)

    shadow_path = Path(args.shadow_compare_json)
    ringfit_path = Path(args.ringfit_metrics_json)
    calib_path = Path(args.calibration_metrics_json)
    var_path = Path(args.variability_metrics_json)
    mring_path = Path(args.mringfits_metrics_json)
    paper4_path = Path(args.paper4_alpha_calibration_json)
    paper4_morph_path = Path(args.paper4_morphology_metrics_json)
    paper4_thetag_path = Path(args.paper4_thetag_metrics_json)
    paper4_debiased_noise_path = Path(args.paper4_debiased_noise_table_json)
    paper6_path = Path(args.paper6_metric_constraints_json)
    paper2_gains_path = Path(args.paper2_gain_uncertainties_json)
    paper2_syserr_path = Path(args.paper2_syserr_table_json)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    out_json = outdir / "eht_kappa_error_budget.json"
    out_png = outdir / "eht_kappa_error_budget.png"

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "shadow_compare_json": str(shadow_path),
            "ringfit_metrics_json": str(ringfit_path),
            "calibration_metrics_json": str(calib_path),
            "variability_metrics_json": str(var_path),
            "mringfits_metrics_json": str(mring_path),
            "paper4_alpha_calibration_json": str(paper4_path),
            "paper4_morphology_metrics_json": str(paper4_morph_path),
            "paper4_thetag_metrics_json": str(paper4_thetag_path),
            "paper4_debiased_noise_table_json": str(paper4_debiased_noise_path),
            "paper6_metric_constraints_json": str(paper6_path),
            "paper2_gain_uncertainties_json": str(paper2_gains_path),
            "paper2_syserr_table_json": str(paper2_syserr_path),
        },
        "ok": True,
        "rows": {},
        "outputs": {"json": str(out_json), "plot_png": str(out_png)},
    }

    if not shadow_path.exists():
        payload["ok"] = False
        payload["reason"] = "missing_shadow_compare_json"
        _write_json(out_json, payload)
        print(f"[warn] missing input; wrote: {out_json}")
        return 0

    shadow = _read_json(shadow_path)
    rows = shadow.get("rows") or []
    row_sgra = None
    for r in rows:
        if isinstance(r, dict) and r.get("key") == "sgra":
            row_sgra = r
            break

    if row_sgra is None:
        payload["ok"] = False
        payload["reason"] = "sgra_row_not_found"
        _write_json(out_json, payload)
        print(f"[warn] missing sgra row; wrote: {out_json}")
        return 0

    ring = _maybe_float(row_sgra.get("ring_diameter_obs_uas"))
    ring_sigma = _maybe_float(row_sgra.get("ring_diameter_obs_uas_sigma"))
    kappa_sigma_req = _maybe_float(row_sgra.get("kappa_sigma_required_3sigma_if_ring_sigma_zero"))
    kappa_sigma_kerr = _maybe_float(row_sgra.get("kappa_sigma_assumed_kerr"))
    scattering_kernel_major_sigma = _maybe_float(row_sgra.get("scattering_kernel_fwhm_major_uas_sigma"))
    refr_wander_min = _maybe_float(row_sgra.get("refractive_wander_uas_min"))
    refr_wander_max = _maybe_float(row_sgra.get("refractive_wander_uas_max"))
    refr_distortion_min = _maybe_float(row_sgra.get("refractive_distortion_uas_min"))
    refr_distortion_max = _maybe_float(row_sgra.get("refractive_distortion_uas_max"))
    refr_asymmetry_min = _maybe_float(row_sgra.get("refractive_asymmetry_uas_min"))
    refr_asymmetry_max = _maybe_float(row_sgra.get("refractive_asymmetry_uas_max"))

    if ring is None or ring <= 0:
        payload["ok"] = False
        payload["reason"] = "invalid_ring_diameter"
        _write_json(out_json, payload)
        print(f"[warn] invalid ring diameter; wrote: {out_json}")
        return 0

    ring_rel_sigma_published = (ring_sigma / ring) if (ring_sigma is not None and ring_sigma >= 0) else None

    ringfit = None
    if ringfit_path.exists():
        ringfit = _read_json(ringfit_path)

    calib = None
    if calib_path.exists():
        calib = _read_json(calib_path)

    var = None
    if var_path.exists():
        var = _read_json(var_path)

    mring = None
    if mring_path.exists():
        mring = _read_json(mring_path)

    paper4 = None
    if paper4_path.exists():
        paper4 = _read_json(paper4_path)

    paper4_morph = None
    if paper4_morph_path.exists():
        paper4_morph = _read_json(paper4_morph_path)

    paper4_thetag = None
    if paper4_thetag_path.exists():
        paper4_thetag = _read_json(paper4_thetag_path)

    paper4_debiased_noise = None
    if paper4_debiased_noise_path.exists():
        paper4_debiased_noise = _read_json(paper4_debiased_noise_path)

    paper6 = None
    if paper6_path.exists():
        paper6 = _read_json(paper6_path)

    paper2_gains = None
    if paper2_gains_path.exists():
        paper2_gains = _read_json(paper2_gains_path)

    paper2_syserr = None
    if paper2_syserr_path.exists():
        paper2_syserr = _read_json(paper2_syserr_path)

    def _calib_fraction(key: str) -> Optional[float]:
        if not isinstance(calib, dict):
            return None
        if not bool(calib.get("ok")):
            return None
        derived = calib.get("derived") or {}
        return _maybe_float(derived.get(key))

    def _var_fraction(*keys: str) -> Optional[float]:
        if not isinstance(var, dict):
            return None
        if not bool(var.get("ok")):
            return None
        cur: Any = var
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return _maybe_float(cur)

    def _mring_d_std_uas() -> Optional[float]:
        if not isinstance(mring, dict):
            return None
        if not bool(mring.get("ok")):
            return None
        dsum = ((mring.get("derived") or {}).get("d_uas_summary") or {})
        return _maybe_float(dsum.get("std"))

    def _paper6_kappa_sigma_proxy_median() -> Optional[float]:
        if not isinstance(paper6, dict):
            return None
        if not bool(paper6.get("ok")):
            return None
        d = (paper6.get("derived") or {}).get("kappa_from_paper6_shadow_diameter_table") or {}
        return _maybe_float(d.get("sigma_avg_median"))

    def _paper6_kappa_sigma_proxy_kappa_mid_std() -> Optional[float]:
        if not isinstance(paper6, dict):
            return None
        if not bool(paper6.get("ok")):
            return None
        d = (paper6.get("derived") or {}).get("kappa_from_paper6_shadow_diameter_table") or {}
        mid_sum = d.get("kappa_mid_summary") if isinstance(d.get("kappa_mid_summary"), dict) else {}
        return _maybe_float(mid_sum.get("std"))

    def _paper2_gain_val(*keys: str) -> Optional[float]:
        if not isinstance(paper2_gains, dict):
            return None
        if not bool(paper2_gains.get("ok")):
            return None
        cur: Any = paper2_gains
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return _maybe_float(cur)

    def _paper2_syserr_val(*keys: str) -> Optional[float]:
        if not isinstance(paper2_syserr, dict):
            return None
        if not bool(paper2_syserr.get("ok")):
            return None
        cur: Any = paper2_syserr
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return _maybe_float(cur)

    def _paper4_debiased_noise_val(*keys: str) -> Optional[float]:
        if not isinstance(paper4_debiased_noise, dict):
            return None
        if not bool(paper4_debiased_noise.get("ok")):
            return None
        cur: Any = paper4_debiased_noise
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return _maybe_float(cur)

    def _paper4_kappa_sigma_proxy_gr_min() -> Optional[float]:
        if not isinstance(paper4, dict):
            return None
        if not bool(paper4.get("ok")):
            return None
        d = paper4.get("derived") if isinstance(paper4.get("derived"), dict) else {}
        return _maybe_float(d.get("kappa_sigma_proxy_gr_from_alpha_tot_sym_min"))

    def _paper4_morphology_kappa_sigma_proxy_hops_imaging_dhat_std() -> Optional[float]:
        if not isinstance(paper4_morph, dict):
            return None
        if not bool(paper4_morph.get("ok")):
            return None
        d = paper4_morph.get("derived") if isinstance(paper4_morph.get("derived"), dict) else {}
        return _maybe_float(d.get("kappa_sigma_proxy_paper4_morphology_hops_imaging_dhat_std"))

    def _paper4_thetag_kappa_sigma_proxy_hops_method_std_over_theta_unit() -> Optional[float]:
        if not isinstance(paper4_thetag, dict):
            return None
        if not bool(paper4_thetag.get("ok")):
            return None
        d = paper4_thetag.get("derived") if isinstance(paper4_thetag.get("derived"), dict) else {}
        return _maybe_float(d.get("kappa_sigma_proxy_paper4_thetag_hops_method_std_over_theta_unit"))

    def _ringfit_rel_std(table_key: str) -> Optional[float]:
        if not isinstance(ringfit, dict):
            return None
        metrics = (ringfit.get("metrics") or {}).get(table_key) or {}
        all_sum = metrics.get("d_mean_uas_summary_all") or {}
        std = _maybe_float(all_sum.get("std"))
        if std is None:
            return None
        return float(std) / float(ring)

    def _image_analysis_rel_std(table_key: str) -> Optional[float]:
        if not isinstance(ringfit, dict):
            return None
        metrics = (ringfit.get("metrics_image_analysis_ringfit") or {}).get(table_key) or {}
        all_sum = metrics.get("d_mean_uas_summary_all") or {}
        std = _maybe_float(all_sum.get("std"))
        if std is None:
            return None
        return float(std) / float(ring)

    def _ringfit_pipeline_rel_std(table_key: str) -> Optional[float]:
        if not isinstance(ringfit, dict):
            return None
        by_pipeline = ((ringfit.get("metrics") or {}).get(table_key) or {}).get("by_pipeline") or {}
        means = []
        for p in ("difmap", "ehtim", "smili", "themis"):
            dsum = ((by_pipeline.get(p) or {}).get("d_mean_uas_summary") or {})
            m = _maybe_float(dsum.get("mean"))
            if m is not None:
                means.append(float(m))
        if len(means) < 2:
            return None
        s = _std(means)
        if s is None:
            return None
        return float(s) / float(ring)

    kappa_sigma_ringfit_desc_all = _ringfit_rel_std("descattered")
    kappa_sigma_ringfit_on_all = _ringfit_rel_std("on_sky")
    kappa_sigma_ringfit_desc_pipe = _ringfit_pipeline_rel_std("descattered")
    kappa_sigma_ringfit_on_pipe = _ringfit_pipeline_rel_std("on_sky")

    kappa_sigma_image_analysis_desc_all = _image_analysis_rel_std("descattered")
    kappa_sigma_proxy_gain_uniform_typical = _calib_fraction("gain_uncertainty_fraction_sigma_uniform_typical")
    kappa_sigma_proxy_non_closing_vis_amp = _calib_fraction("non_closing_vis_amp_fraction")
    kappa_sigma_proxy_gain_synthetic_combined_max = _calib_fraction("gain_synthetic_combined_mean_max")
    kappa_sigma_proxy_variability_max_2to6 = _var_fraction("derived", "representative", "sigma_var_max_2to6_midpoint", "max")
    mring_d_std_uas = _mring_d_std_uas()
    kappa_sigma_proxy_mringfits_scan_std = (float(mring_d_std_uas) / float(ring)) if (mring_d_std_uas is not None) else None
    kappa_sigma_proxy_paper4_alpha_tot_sym_min_gr = _paper4_kappa_sigma_proxy_gr_min()
    kappa_sigma_proxy_paper4_morphology_hops_imaging_dhat_std = _paper4_morphology_kappa_sigma_proxy_hops_imaging_dhat_std()
    kappa_sigma_proxy_paper4_thetag_hops_method_std_over_theta_unit = _paper4_thetag_kappa_sigma_proxy_hops_method_std_over_theta_unit()
    kappa_sigma_proxy_paper4_debiased_noise_a4_fraction = _paper4_debiased_noise_val(
        "derived", "a4_fraction_at_4gly", "mid"
    )
    kappa_sigma_proxy_paper6_dsh_table_sigma_avg_median = _paper6_kappa_sigma_proxy_median()
    kappa_sigma_proxy_paper6_dsh_table_kappa_mid_std = _paper6_kappa_sigma_proxy_kappa_mid_std()
    kappa_sigma_proxy_paper2_delta_g_alma = _paper2_gain_val("extracted", "delta_g_alma_smt", "delta_g_alma")
    kappa_sigma_proxy_paper2_delta_g_smt = _paper2_gain_val("extracted", "delta_g_alma_smt", "delta_g_smt")
    kappa_sigma_proxy_paper2_delta_g_lmt = _paper2_gain_val("extracted", "delta_g_lmt_tot", "delta_g_lmt")
    kappa_sigma_proxy_paper2_delta_g_tot = _paper2_gain_val("extracted", "delta_g_lmt_tot", "delta_g_tot")
    kappa_sigma_proxy_paper2_delta_g_alma_smt_quadrature = _paper2_gain_val("derived", "delta_g_alma_smt_quadrature")
    kappa_sigma_proxy_paper2_delta_g_lmt_smt_tot_quadrature = _paper2_gain_val(
        "derived", "delta_g_lmt_smt_tot_quadrature"
    )
    kappa_sigma_proxy_paper2_syserr_amp_fraction_max_over_pipelines = _paper2_syserr_val(
        "derived", "sgra_amp_fraction_max_over_pipelines"
    )
    kappa_sigma_proxy_scattering_kernel_major_over_ring = (
        (float(scattering_kernel_major_sigma) / float(ring)) if (scattering_kernel_major_sigma is not None) else None
    )
    kappa_sigma_proxy_refractive_wander_mid_over_ring = (
        (0.5 * (float(refr_wander_min) + float(refr_wander_max)) / float(ring))
        if (refr_wander_min is not None and refr_wander_max is not None and refr_wander_max >= refr_wander_min)
        else None
    )
    kappa_sigma_proxy_refractive_distortion_mid_over_ring = (
        (0.5 * (float(refr_distortion_min) + float(refr_distortion_max)) / float(ring))
        if (
            refr_distortion_min is not None
            and refr_distortion_max is not None
            and refr_distortion_max >= refr_distortion_min
        )
        else None
    )
    kappa_sigma_proxy_refractive_distortion_max_over_ring = (
        (float(refr_distortion_max) / float(ring)) if (refr_distortion_max is not None) else None
    )
    kappa_sigma_proxy_refractive_asymmetry_mid_over_ring = (
        (0.5 * (float(refr_asymmetry_min) + float(refr_asymmetry_max)) / float(ring))
        if (
            refr_asymmetry_min is not None
            and refr_asymmetry_max is not None
            and refr_asymmetry_max >= refr_asymmetry_min
        )
        else None
    )

    # Adopted κσ (for the falsification pack): use method-scatter median as a robust "current-scale" indicator.
    # NOTE: these are proxies, not statistically independent error bars.
    kappa_sigma_method_scatter_values = _finite_pos(
        [
            kappa_sigma_proxy_paper4_morphology_hops_imaging_dhat_std,
            kappa_sigma_proxy_paper4_thetag_hops_method_std_over_theta_unit,
            kappa_sigma_proxy_paper6_dsh_table_kappa_mid_std,
            kappa_sigma_image_analysis_desc_all,
            kappa_sigma_ringfit_desc_all,
            kappa_sigma_ringfit_on_all,
            kappa_sigma_ringfit_desc_pipe,
            kappa_sigma_ringfit_on_pipe,
        ]
    )
    kappa_sigma_method_scatter_median = (
        float(np.median(np.array(kappa_sigma_method_scatter_values, dtype=float))) if kappa_sigma_method_scatter_values else None
    )

    # Additional floor motivated by emission-model dependence (e.g., thick disk; Vincent 2022):
    # use the Paper VI inferred-shadow table scatter as an operational κ-scale proxy.
    kappa_sigma_model_floor = kappa_sigma_proxy_paper6_dsh_table_sigma_avg_median
    kappa_sigma_adopted_for_falsification = _maybe_float(
        max(
            _finite_pos(
                [
                    kappa_sigma_method_scatter_median,
                    ring_rel_sigma_published,
                    kappa_sigma_kerr,
                    kappa_sigma_model_floor,
                ]
            ),
            default=float("nan"),
        )
    )

    items = {}
    if ring_rel_sigma_published is not None:
        items["σ(κ) from ring σ (published)"] = float(ring_rel_sigma_published)
    if kappa_sigma_kerr is not None:
        items["σ(κ) from Kerr range (ref)"] = float(kappa_sigma_kerr)
    if kappa_sigma_proxy_gain_uniform_typical is not None:
        items["σ(κ) proxy: gain (5–15% uniform σ)"] = float(kappa_sigma_proxy_gain_uniform_typical)
    if kappa_sigma_proxy_non_closing_vis_amp is not None:
        items["σ(κ) proxy: non-closing vis amp (4%)"] = float(kappa_sigma_proxy_non_closing_vis_amp)
    if kappa_sigma_proxy_gain_synthetic_combined_max is not None:
        items["σ(κ) proxy: synthetic gains max √(offset²+p²)"] = float(kappa_sigma_proxy_gain_synthetic_combined_max)
    if kappa_sigma_proxy_variability_max_2to6 is not None:
        items["σ(κ) proxy: variability noise (max 2–6 Gλ)"] = float(kappa_sigma_proxy_variability_max_2to6)
    if kappa_sigma_proxy_scattering_kernel_major_over_ring is not None:
        items["σ(κ) proxy: scattering kernel σ (major)"] = float(kappa_sigma_proxy_scattering_kernel_major_over_ring)
    if kappa_sigma_proxy_refractive_wander_mid_over_ring is not None:
        items["σ(κ) proxy: refractive wander (mid)"] = float(kappa_sigma_proxy_refractive_wander_mid_over_ring)
    if kappa_sigma_proxy_refractive_distortion_mid_over_ring is not None:
        items["σ(κ) proxy: refractive distortion (mid)"] = float(kappa_sigma_proxy_refractive_distortion_mid_over_ring)
    if kappa_sigma_proxy_refractive_distortion_max_over_ring is not None:
        items["σ(κ) proxy: refractive distortion (max)"] = float(kappa_sigma_proxy_refractive_distortion_max_over_ring)
    if kappa_sigma_proxy_refractive_asymmetry_mid_over_ring is not None:
        items["σ(κ) proxy: refractive asymmetry (mid)"] = float(kappa_sigma_proxy_refractive_asymmetry_mid_over_ring)
    if kappa_sigma_proxy_mringfits_scan_std is not None:
        items["σ(κ) proxy: m-ring fits (scan-to-scan std)"] = float(kappa_sigma_proxy_mringfits_scan_std)
    if kappa_sigma_proxy_paper4_alpha_tot_sym_min_gr is not None:
        items["σ(κ) proxy: Paper IV α calibration (best σ_tot)"] = float(kappa_sigma_proxy_paper4_alpha_tot_sym_min_gr)
    if kappa_sigma_proxy_paper4_morphology_hops_imaging_dhat_std is not None:
        items["σ(κ) proxy: Paper IV morphology (HOPS imaging d_hat scatter)"] = float(
            kappa_sigma_proxy_paper4_morphology_hops_imaging_dhat_std
        )
    if kappa_sigma_proxy_paper4_thetag_hops_method_std_over_theta_unit is not None:
        items["σ(κ) proxy: Paper IV θ_g method scatter (HOPS)"] = float(
            kappa_sigma_proxy_paper4_thetag_hops_method_std_over_theta_unit
        )
    if kappa_sigma_proxy_paper4_debiased_noise_a4_fraction is not None:
        items["σ(κ) proxy: Paper IV debiased a₄ (|u|=4 Gλ)"] = float(kappa_sigma_proxy_paper4_debiased_noise_a4_fraction)
    if kappa_sigma_proxy_paper6_dsh_table_sigma_avg_median is not None:
        items["σ(κ) proxy: emission model (thick disk; Paper VI d_sh table median)"] = float(
            kappa_sigma_proxy_paper6_dsh_table_sigma_avg_median
        )
    if kappa_sigma_proxy_paper6_dsh_table_kappa_mid_std is not None:
        items["σ(κ) proxy: Paper VI κ mid scatter (std)"] = float(kappa_sigma_proxy_paper6_dsh_table_kappa_mid_std)
    if kappa_sigma_proxy_paper2_delta_g_alma_smt_quadrature is not None:
        items["σ(κ) proxy: Paper II Δg(ALMA,SMT) quadrature"] = float(kappa_sigma_proxy_paper2_delta_g_alma_smt_quadrature)
    if kappa_sigma_proxy_paper2_delta_g_lmt_smt_tot_quadrature is not None:
        items["σ(κ) proxy: Paper II Δg(LMT,SMT,tot) quadrature"] = float(
            kappa_sigma_proxy_paper2_delta_g_lmt_smt_tot_quadrature
        )
    if kappa_sigma_proxy_paper2_syserr_amp_fraction_max_over_pipelines is not None:
        items["σ(κ) proxy: Paper II tab:syserr amp s (max)"] = float(kappa_sigma_proxy_paper2_syserr_amp_fraction_max_over_pipelines)
    if kappa_sigma_image_analysis_desc_all is not None:
        items["σ(κ) proxy: image analysis table (descattered std)"] = float(kappa_sigma_image_analysis_desc_all)
    if kappa_sigma_ringfit_desc_all is not None:
        items["σ(κ) from ringfits (descattered std)"] = float(kappa_sigma_ringfit_desc_all)
    if kappa_sigma_ringfit_on_all is not None:
        items["σ(κ) from ringfits (on-sky std)"] = float(kappa_sigma_ringfit_on_all)
    if kappa_sigma_ringfit_desc_pipe is not None:
        items["σ(κ) from ringfits (descattered pipeline scatter)"] = float(kappa_sigma_ringfit_desc_pipe)
    if kappa_sigma_ringfit_on_pipe is not None:
        items["σ(κ) from ringfits (on-sky pipeline scatter)"] = float(kappa_sigma_ringfit_on_pipe)

    payload["rows"]["sgra"] = {
        "ring_diameter_uas": float(ring),
        "ring_diameter_sigma_uas_published": ring_sigma,
        "kappa_sigma_required_3sigma_if_ring_sigma_zero": kappa_sigma_req,
        "kappa_sigma_assumed_kerr": kappa_sigma_kerr,
        "kappa_sigma_from_ring_published": ring_rel_sigma_published,
        "kappa_sigma_from_ringfits_descattered_all_std": kappa_sigma_ringfit_desc_all,
        "kappa_sigma_from_ringfits_on_sky_all_std": kappa_sigma_ringfit_on_all,
        "kappa_sigma_from_ringfits_descattered_pipeline_std": kappa_sigma_ringfit_desc_pipe,
        "kappa_sigma_from_ringfits_on_sky_pipeline_std": kappa_sigma_ringfit_on_pipe,
        "kappa_sigma_proxy_image_analysis_descattered_all_std": kappa_sigma_image_analysis_desc_all,
        "kappa_sigma_proxy_gain_uniform_typical": kappa_sigma_proxy_gain_uniform_typical,
        "kappa_sigma_proxy_non_closing_vis_amp": kappa_sigma_proxy_non_closing_vis_amp,
        "kappa_sigma_proxy_gain_synthetic_combined_mean_max": kappa_sigma_proxy_gain_synthetic_combined_max,
        "kappa_sigma_proxy_variability_noise_max_2to6": kappa_sigma_proxy_variability_max_2to6,
        "kappa_sigma_proxy_scattering_kernel_major_over_ring": kappa_sigma_proxy_scattering_kernel_major_over_ring,
        "kappa_sigma_proxy_refractive_wander_mid_over_ring": kappa_sigma_proxy_refractive_wander_mid_over_ring,
        "kappa_sigma_proxy_refractive_distortion_mid_over_ring": kappa_sigma_proxy_refractive_distortion_mid_over_ring,
        "kappa_sigma_proxy_refractive_distortion_max_over_ring": kappa_sigma_proxy_refractive_distortion_max_over_ring,
        "kappa_sigma_proxy_refractive_asymmetry_mid_over_ring": kappa_sigma_proxy_refractive_asymmetry_mid_over_ring,
        "kappa_sigma_proxy_mringfits_scan_std": kappa_sigma_proxy_mringfits_scan_std,
        "kappa_sigma_proxy_paper4_alpha_tot_sym_min_gr": kappa_sigma_proxy_paper4_alpha_tot_sym_min_gr,
        "kappa_sigma_proxy_paper4_morphology_hops_imaging_dhat_std": kappa_sigma_proxy_paper4_morphology_hops_imaging_dhat_std,
        "kappa_sigma_proxy_paper4_thetag_hops_method_std_over_theta_unit": kappa_sigma_proxy_paper4_thetag_hops_method_std_over_theta_unit,
        "kappa_sigma_proxy_paper4_debiased_noise_a4_fraction": kappa_sigma_proxy_paper4_debiased_noise_a4_fraction,
        "kappa_sigma_proxy_paper6_dsh_table_sigma_avg_median": kappa_sigma_proxy_paper6_dsh_table_sigma_avg_median,
        "kappa_sigma_proxy_paper6_dsh_table_kappa_mid_std": kappa_sigma_proxy_paper6_dsh_table_kappa_mid_std,
        "kappa_sigma_proxy_paper2_delta_g_alma": kappa_sigma_proxy_paper2_delta_g_alma,
        "kappa_sigma_proxy_paper2_delta_g_smt": kappa_sigma_proxy_paper2_delta_g_smt,
        "kappa_sigma_proxy_paper2_delta_g_lmt": kappa_sigma_proxy_paper2_delta_g_lmt,
        "kappa_sigma_proxy_paper2_delta_g_tot": kappa_sigma_proxy_paper2_delta_g_tot,
        "kappa_sigma_proxy_paper2_delta_g_alma_smt_quadrature": kappa_sigma_proxy_paper2_delta_g_alma_smt_quadrature,
        "kappa_sigma_proxy_paper2_delta_g_lmt_smt_tot_quadrature": kappa_sigma_proxy_paper2_delta_g_lmt_smt_tot_quadrature,
        "kappa_sigma_proxy_paper2_syserr_amp_fraction_max_over_pipelines": kappa_sigma_proxy_paper2_syserr_amp_fraction_max_over_pipelines,
        "kappa_sigma_method_scatter_values": kappa_sigma_method_scatter_values,
        "kappa_sigma_method_scatter_median": kappa_sigma_method_scatter_median,
        "kappa_sigma_adopted_for_falsification": kappa_sigma_adopted_for_falsification,
        "notes": [
            "ringfits-derived scatter is treated here as a systematic-scale indicator, not a direct measurement uncertainty of the published ring diameter.",
            "Kerr-range term is a reference systematic scale from the GR shadow coefficient range used in eht_shadow_compare.",
            "gain/non-closing terms are heuristic scale indicators from Paper III (observations) calibration/systematics discussion, not a direct mapping to ring diameter uncertainty.",
            "image-analysis table term is a heuristic scale indicator from Paper III image_analysis Table tab:SgrA_ringfit (REx/VIDA; pipeline/day means), not a direct mapping to ring diameter uncertainty.",
            "synthetic gain term is a heuristic scale indicator from Paper III appendix_synthetic (gain parameters used for synthetic data generation), not a direct mapping to ring diameter uncertainty.",
            "variability noise term is a heuristic scale indicator from Paper III pre-imaging eq:PSD_noise + tab:premodeling, not a direct mapping to ring diameter uncertainty.",
            "scattering kernel term is a proxy based on the major-axis FWHM uncertainty of the scattering kernel; treated as an order-of-magnitude κ-scale indicator via σ(kernel)/ring_diameter.",
            "refractive wander/distortion/asymmetry terms are proxies based on the quoted range; treated as order-of-magnitude κ-scale indicators via (range midpoint or max)/ring_diameter.",
            "m-ring fits term is a heuristic scale indicator from Paper V tab:mringfits (ML fits on selected 120s scans), not a direct mapping to the published ring diameter uncertainty.",
            "Paper IV alpha term is a proxy derived from Paper IV tab:alphacal (GRMHD-calibrated alpha=d/theta_g) total uncertainty; treated as a kappa-scale indicator via sigma(alpha)/shadow_coeff_gr.",
            "Paper IV morphology term is a proxy derived from Paper IV tab:SgrAMorphology debiased diameter d_hat scatter across imaging methods (HOPS pipeline); treated as a kappa-scale indicator via sigma(d_hat)/ring_diameter_used_in_shadow_compare.",
            "Paper IV theta_g term is a proxy derived from Paper IV tab:thetag method scatter in theta_g (HOPS pipeline); treated as a kappa-scale indicator via std(theta_g)/theta_unit_uas.",
            "Paper IV debiased noise term is a proxy derived from Paper IV tab:debiased_noise a4 (excess noise at |u|=4 Gλ); treated as an amplitude-variability systematic scale indicator (not a direct uncertainty of the published ring diameter).",
            "Paper VI d_sh table term is a proxy derived from Paper VI inferred shadow diameter (Table: The Inferred Shadow Diameter of Sgr A*) and the ring diameter used in eht_shadow_compare.",
            "Paper VI κ mid scatter term is a proxy based on the standard deviation of κ midpoints across methods/models in the Paper VI d_sh table; treated as a kappa-scale method-scatter indicator (not a direct uncertainty of the published ring diameter).",
            "Paper II Δg terms are proxies based on quoted residual gain uncertainties after calibrator-based calibration; treated as amplitude-calibration systematic scales (not a direct uncertainty of the published ring diameter).",
            "Paper II tab:syserr term is a proxy based on the maximum systematic error budget s for percent-valued closure quantities; treated as a calibration/systematics scale indicator (not a direct uncertainty of the published ring diameter).",
            "kappa_sigma_adopted_for_falsification is a conservative 'current-scale' κσ adopted for the falsification pack; it is based on the method-scatter median with a floor at max(published ring rel σ, Kerr-range κσ, emission-model proxy).",
            "The emission-model floor is motivated by the fact that ring↔shadow mapping is model-dependent (e.g., thick disk effects; Vincent 2022); here we use the Paper VI inferred-shadow table scatter as an operational proxy scale.",
        ],
        "source_paths": {
            "shadow_compare": str(shadow_path),
            "ringfits_metrics": str(ringfit_path) if ringfit_path.exists() else None,
            "calibration_metrics": str(calib_path) if calib_path.exists() else None,
            "variability_metrics": str(var_path) if var_path.exists() else None,
            "mringfits_metrics": str(mring_path) if mring_path.exists() else None,
            "paper4_alpha_calibration": str(paper4_path) if paper4_path.exists() else None,
            "paper4_morphology_metrics": str(paper4_morph_path) if paper4_morph_path.exists() else None,
            "paper4_thetag_metrics": str(paper4_thetag_path) if paper4_thetag_path.exists() else None,
            "paper4_debiased_noise_table": str(paper4_debiased_noise_path) if paper4_debiased_noise_path.exists() else None,
            "paper6_metric_constraints": str(paper6_path) if paper6_path.exists() else None,
            "paper2_gain_uncertainties": str(paper2_gains_path) if paper2_gains_path.exists() else None,
            "paper2_syserr_table": str(paper2_syserr_path) if paper2_syserr_path.exists() else None,
        },
    }

    try:
        _plot_budget(
            title="Sgr A* κ error budget (scale indicators)",
            items=items,
            required=kappa_sigma_req,
            out_png=out_png,
        )
    except Exception as e:
        payload["rows"]["sgra"]["plot_error"] = str(e)

    _write_json(out_json, payload)

    try:
        worklog.append_event(
            {
                "ts_utc": payload["generated_utc"],
                "topic": "eht",
                "action": "eht_kappa_error_budget",
                "outputs": [str(out_json.relative_to(root)).replace("\\", "/"), str(out_png.relative_to(root)).replace("\\", "/")],
                "metrics": {"ok": bool(payload.get("ok")), "items": int(len(items))},
            }
        )
    except Exception:
        pass

    print(f"[ok] json: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
