#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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


def _ks_c_alpha(alpha: float) -> Optional[float]:
    # Standard 2-sample KS asymptotic constants.
    if abs(alpha - 0.10) < 1e-12:
        return 1.22
    if abs(alpha - 0.05) < 1e-12:
        return 1.36
    if abs(alpha - 0.01) < 1e-12:
        return 1.63
    if abs(alpha - 0.001) < 1e-12:
        return 1.95
    return None


def _ks_two_sample_dcrit(alpha: float, n: int, m: int) -> Optional[float]:
    if n <= 0 or m <= 0:
        return None
    c = _ks_c_alpha(alpha)
    if c is None:
        return None
    return float(c * math.sqrt((n + m) / (n * m)))


def _ks_two_sample_d(sample_a: Sequence[float], sample_b: Sequence[float]) -> Optional[float]:
    a = [float(x) for x in sample_a if isinstance(x, (int, float))]
    b = [float(x) for x in sample_b if isinstance(x, (int, float))]
    if not a or not b:
        return None
    a.sort()
    b.sort()
    n = len(a)
    m = len(b)

    i = 0
    j = 0
    cdf_a = 0.0
    cdf_b = 0.0
    d = 0.0

    while i < n and j < m:
        va = a[i]
        vb = b[j]
        if va < vb:
            v = va
            while i < n and a[i] == v:
                i += 1
            cdf_a = i / n
        elif vb < va:
            v = vb
            while j < m and b[j] == v:
                j += 1
            cdf_b = j / m
        else:
            v = va
            while i < n and a[i] == v:
                i += 1
            while j < m and b[j] == v:
                j += 1
            cdf_a = i / n
            cdf_b = j / m
        d = max(d, abs(cdf_a - cdf_b))

    while i < n:
        v = a[i]
        while i < n and a[i] == v:
            i += 1
        cdf_a = i / n
        d = max(d, abs(cdf_a - cdf_b))

    while j < m:
        v = b[j]
        while j < m and b[j] == v:
            j += 1
        cdf_b = j / m
        d = max(d, abs(cdf_a - cdf_b))

    return float(d)


def _ks_two_sample_p_asymptotic(d: float, n: int, m: int) -> Optional[float]:
    # Reference: standard asymptotic KS distribution (same functional form as scipy.stats.ks_2samp(mode="asymp")).
    if not isinstance(d, (int, float)) or d < 0:
        return None
    if float(d) <= 0.0:
        return 1.0
    if n <= 0 or m <= 0:
        return None
    ne = (n * m) / (n + m)
    if ne <= 0:
        return None
    en = math.sqrt(ne)
    lam = (en + 0.12 + 0.11 / en) * float(d)

    s = 0.0
    for j in range(1, 101):
        term = ((-1.0) ** (j - 1)) * math.exp(-2.0 * (lam**2) * (j**2))
        s += term
        if abs(term) < 1e-12:
            break
    p = 2.0 * s
    if p < 0.0:
        p = 0.0
    if p > 1.0:
        p = 1.0
    return float(p)


def _ks_solve_d_for_p_asymptotic(alpha: float, n: int, m: int) -> Optional[float]:
    # Solve p_asymptotic(d) = alpha for d in [0, 1] via bisection.
    if not isinstance(alpha, (int, float)) or alpha <= 0.0 or alpha >= 1.0:
        return None
    if n <= 0 or m <= 0:
        return None

    lo = 0.0
    hi = 1.0
    plo = _ks_two_sample_p_asymptotic(lo, n, m)
    phi = _ks_two_sample_p_asymptotic(hi, n, m)
    if plo is None or phi is None:
        return None
    if plo < alpha:
        return lo
    if phi > alpha:
        return hi

    for _ in range(64):
        mid = 0.5 * (lo + hi)
        pmid = _ks_two_sample_p_asymptotic(mid, n, m)
        if pmid is None:
            return None
        if pmid > alpha:
            lo = mid
        else:
            hi = mid
    return float(hi)


def _near_passing_rows(pass_fraction_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    near = (pass_fraction_metrics.get("extracted") or {}).get("near_passing") or {}
    for k in ("fail_one_thermal", "fail_one_nonthermal", "fail_none"):
        block = near.get(k) or {}
        out.extend(block.get("rows") or [])
    return out


def _count_relax_pass(rows: List[Dict[str, Any]], relax: Sequence[str]) -> int:
    relax_set = set(relax)
    n = 0
    for r in rows:
        failed = set(r.get("failed_constraints_norm") or [])
        if failed and failed.issubset(relax_set):
            n += 1
    return n


def _extract_near_combo_counts(pass_fraction_metrics: Dict[str, Any]) -> Dict[str, Any]:
    combined = (
        (pass_fraction_metrics.get("derived") or {})
        .get("near_passing", {})
        .get("combined_summary", {})
    )
    if not combined:
        return {}
    return combined


def main(argv: Optional[Sequence[str]] = None) -> int:
    root = _repo_root()

    default_key = root / "output" / "private" / "eht" / "eht_sgra_paper5_key_constraints_metrics.json"
    default_pf = root / "output" / "private" / "eht" / "eht_sgra_paper5_pass_fraction_tables_metrics.json"
    default_sweep = root / "output" / "private" / "eht" / "eht_sgra_paper5_constraint_relaxation_sweep_metrics.json"
    default_gravity = root / "output" / "private" / "eht" / "gravity_sgra_flux_distribution_metrics.json"
    default_wielgus = root / "output" / "private" / "eht" / "wielgus2022_m3_observed_metrics.json"
    default_m3_hist_vals = root / "output" / "private" / "eht" / "eht_sgra_paper5_m3_historical_distribution_values.json"
    default_out = root / "output" / "private" / "eht" / "eht_sgra_paper5_m3_nir_reconnection_conditions_metrics.json"
    default_png = root / "output" / "private" / "eht" / "eht_sgra_paper5_m3_nir_reconnection_conditions.png"

    ap = argparse.ArgumentParser(description="Quantify reconnection conditions for Paper V M3 / 2.2μm constraints (assumption sensitivity).")
    ap.add_argument("--key-metrics", type=str, default=str(default_key))
    ap.add_argument("--pass-fraction-metrics", type=str, default=str(default_pf))
    ap.add_argument("--relax-sweep-metrics", type=str, default=str(default_sweep))
    ap.add_argument(
        "--gravity-flux-metrics",
        type=str,
        default=str(default_gravity),
        help="Optional: output/private/eht/gravity_sgra_flux_distribution_metrics.json (GRAVITY 2020 flux distribution percentiles).",
    )
    ap.add_argument(
        "--wielgus2022-m3-metrics",
        type=str,
        default=str(default_wielgus),
        help="Optional: output/private/eht/wielgus2022_m3_observed_metrics.json (Wielgus+2022; (sigma/mu)_3h table and DRW tau).",
    )
    ap.add_argument(
        "--m3-historical-values-json",
        type=str,
        default=str(default_m3_hist_vals),
        help="Optional: output/private/eht/eht_sgra_paper5_m3_historical_distribution_values.json (reconstructed 3h mi3 values; KS vs 2017 sample).",
    )
    ap.add_argument("--out", type=str, default=str(default_out))
    ap.add_argument("--out-png", type=str, default=str(default_png))
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    key_path = Path(args.key_metrics)
    pf_path = Path(args.pass_fraction_metrics)
    sweep_path = Path(args.relax_sweep_metrics)
    gravity_path = Path(args.gravity_flux_metrics) if getattr(args, "gravity_flux_metrics", None) else None
    wielgus_path = Path(args.wielgus2022_m3_metrics) if getattr(args, "wielgus2022_m3_metrics", None) else None
    m3_hist_vals_path = (
        Path(args.m3_historical_values_json) if getattr(args, "m3_historical_values_json", None) else None
    )
    out_json = Path(args.out)
    out_png = Path(args.out_png)

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "ok": True,
        "inputs": {
            "key_constraints_metrics_json": str(key_path),
            "pass_fraction_tables_metrics_json": str(pf_path),
            "constraint_relaxation_sweep_metrics_json": str(sweep_path),
            "gravity_flux_distribution_metrics_json": str(gravity_path) if gravity_path is not None else None,
            "wielgus2022_m3_observed_metrics_json": str(wielgus_path) if wielgus_path is not None else None,
            "m3_historical_distribution_values_json": str(m3_hist_vals_path) if m3_hist_vals_path is not None else None,
        },
        "derived": {},
        "outputs": {"json": str(out_json), "png": str(out_png)},
    }

    missing = [str(p) for p in (key_path, pf_path, sweep_path) if not p.exists()]
    if missing:
        payload["ok"] = False
        payload["reason"] = "missing_inputs"
        payload["missing_files"] = missing
        _write_json(out_json, payload)
        print(f"[warn] missing inputs; wrote: {out_json}")
        return 0

    key = _read_json(key_path)
    pf = _read_json(pf_path)
    sweep = _read_json(sweep_path)
    gravity: Optional[Dict[str, Any]] = None
    if gravity_path is not None and gravity_path.exists():
        try:
            gravity = _read_json(gravity_path)
        except Exception:
            gravity = None

    wielgus: Optional[Dict[str, Any]] = None
    if wielgus_path is not None and wielgus_path.exists():
        try:
            wielgus = _read_json(wielgus_path)
        except Exception:
            wielgus = None

    m3_hist_vals: Optional[Dict[str, Any]] = None
    if m3_hist_vals_path is not None and m3_hist_vals_path.exists():
        try:
            m3_hist_vals = _read_json(m3_hist_vals_path)
        except Exception:
            m3_hist_vals = None

    nir = ((key.get("extracted") or {}).get("nir_2p2um") or {})
    m3 = ((key.get("extracted") or {}).get("m3") or {})
    sens = ((key.get("extracted") or {}).get("m3_sensitivity") or {})

    # --- NIR (2.2 μm) threshold reinterpretation knobs ---
    med = nir.get("observed_median_mJy")
    sig = nir.get("observed_sigma_mJy")
    thr = nir.get("threshold_mJy")
    nir_out: Dict[str, Any] = {"ok": True}
    if isinstance(med, (int, float)) and isinstance(sig, (int, float)) and sig > 0 and isinstance(thr, (int, float)):
        med = float(med)
        sig = float(sig)
        thr = float(thr)
        nir_out.update(
            {
                "observed_median_mJy": med,
                "observed_sigma_mJy": sig,
                "threshold_mJy": thr,
                "threshold_minus_median_sigma": float((thr - med) / sig),
                "threshold_over_median": float(thr / med) if med != 0.0 else None,
                "threshold_for_k_sigma_above_median": {f"k={k}": float(med + k * sig) for k in (1, 2, 3)},
                # If observed median is biased upward by flares by b_mJy (so quiescent median = med - b),
                # then current threshold corresponds to (thr - (med - b)) / sig sigmas above quiescent median.
                "threshold_sigma_above_quiescent_if_flare_bias_b_mJy": [
                    {"b_mJy": float(b), "k_sigma": float((thr - (med - b)) / sig)} for b in (0.0, 0.1, 0.2, 0.3, 0.4)
                ],
                # Solve b such that thr = (med - b) + k*sig  => b = med + k*sig - thr
                "required_flare_bias_b_mJy_for_threshold_equals_quiescent_plus_k_sigma": {
                    f"k={k}": float(med + k * sig - thr) for k in (1, 2, 3)
                },
            }
        )
        if isinstance(gravity, dict):
            by_year = (gravity.get("derived") or {}).get("percentiles_avg_by_year") or {}
            if isinstance(by_year, dict):
                nir_out["gravity2020_percentiles_avg_by_year"] = {
                    "2017": by_year.get("2017"),
                    "2018": by_year.get("2018"),
                    "2019": by_year.get("2019"),
                    "2017-2019": by_year.get("2017-2019"),
                }
                # Derived convenience ratios (ignore struck-out cells).
                try:
                    p5_all = ((by_year.get("2017-2019") or {}).get("p5") or {})
                    if isinstance(p5_all, dict) and not bool(p5_all.get("struck_out")):
                        p5_val = p5_all.get("value_mJy")
                        if isinstance(p5_val, (int, float)) and p5_val > 0:
                            nir_out["threshold_over_gravity2020_p5_all"] = float(thr / float(p5_val))
                except Exception:
                    pass
    else:
        nir_out["ok"] = False
        nir_out["reason"] = "missing_or_invalid_nir_numbers"

    # --- M3 (KS-test) sample-size dependence proxy ---
    alpha = float(m3.get("ks_reject_if_p_lt") or 0.01)
    n_hist = m3.get("observed_samples_historical_n")
    n_2017 = m3.get("observed_samples_2017_n")
    n_model_exp = m3.get("model_samples_exploratory_n")
    n_model_fid = m3.get("model_samples_fiducial_n") or []
    n_model_list = []
    if isinstance(n_model_exp, int):
        n_model_list.append(int(n_model_exp))
    if isinstance(n_model_fid, list):
        for x in n_model_fid:
            if isinstance(x, int):
                n_model_list.append(int(x))
    n_model_list = sorted({n for n in n_model_list if n > 0})

    m3_out: Dict[str, Any] = {"ok": True, "alpha": alpha, "c_alpha": _ks_c_alpha(alpha)}
    if isinstance(n_hist, int) and isinstance(n_2017, int) and n_hist > 0 and n_2017 > 0 and n_model_list:
        d_hist = {f"n_model={m}": _ks_two_sample_dcrit(alpha, int(n_hist), int(m)) for m in n_model_list}
        d_2017 = {f"n_model={m}": _ks_two_sample_dcrit(alpha, int(n_2017), int(m)) for m in n_model_list}
        ratio = {}
        for m in n_model_list:
            a = d_2017.get(f"n_model={m}")
            b = d_hist.get(f"n_model={m}")
            ratio[f"n_model={m}"] = (float(a) / float(b)) if (a is not None and b not in (None, 0.0)) else None
        m3_out.update(
            {
                "observed_samples_historical_n": int(n_hist),
                "observed_samples_2017_n": int(n_2017),
                "model_samples_candidates": n_model_list,
                "ks_dcrit_hist_vs_model_n": d_hist,
                "ks_dcrit_2017_vs_model_n": d_2017,
                "ks_dcrit_ratio_2017_over_hist": ratio,
            }
        )
    else:
        m3_out["ok"] = False
        m3_out["reason"] = "missing_or_invalid_m3_sample_sizes"

    # Optional: attach observed (sigma/mu)_3h table values and DRW tau context (Wielgus+2022).
    if isinstance(wielgus, dict):
        w_der = wielgus.get("derived") or {}
        w_ex = wielgus.get("extracted") or {}
        w_sel = w_der.get("paper5_m3_2017_7sample_candidate") if isinstance(w_der, dict) else None
        if isinstance(w_sel, dict) and bool(w_sel.get("ok")):
            m3_out["wielgus2022_paper5_2017_7sample_candidate"] = {
                "samples_n": w_sel.get("samples_n"),
                "summary_all": w_sel.get("summary_all"),
                "summary_nonflare_days": w_sel.get("summary_nonflare_days"),
                "summary_flare_day_2017_04_11": w_sel.get("summary_flare_day_2017_04_11"),
                "samples": w_sel.get("samples"),
                "note": w_sel.get("note"),
            }
        w_ks = w_der.get("paper5_m3_ks_sanity_2017_vs_historical_proxy") if isinstance(w_der, dict) else None
        if isinstance(w_ks, dict) and bool(w_ks.get("ok")):
            m3_out["wielgus2022_paper5_m3_ks_sanity_2017_vs_historical_proxy"] = w_ks
        w_hist = w_der.get("paper5_m3_historical_distribution_candidate_pre_eht_2017_apr11") if isinstance(w_der, dict) else None
        if isinstance(w_hist, dict) and bool(w_hist.get("ok")):
            m3_out["wielgus2022_paper5_historical_distribution_candidate_pre_eht_2017_apr11"] = {
                "deltaT_hours": w_hist.get("deltaT_hours"),
                "date_cutoff_ymd": w_hist.get("date_cutoff_ymd"),
                "duration_h_min": w_hist.get("duration_h_min"),
                "curves_n": w_hist.get("curves_n"),
                "segments_n": w_hist.get("segments_n"),
                "segments_by_array": w_hist.get("segments_by_array"),
                "curves_by_array": w_hist.get("curves_by_array"),
                "curves_with_multiple_segments_n": w_hist.get("curves_with_multiple_segments_n"),
                "curves_with_multiple_segments_top": w_hist.get("curves_with_multiple_segments_top"),
                "proxy_sigma_over_mu_summary": w_hist.get("proxy_sigma_over_mu_summary"),
                "note": w_hist.get("note"),
            }
        w_hist_all = w_der.get("paper5_m3_historical_distribution_candidate_2017_inclusive") if isinstance(w_der, dict) else None
        if isinstance(w_hist_all, dict) and bool(w_hist_all.get("ok")):
            m3_out["wielgus2022_paper5_historical_distribution_candidate_2017_inclusive"] = {
                "date_cutoff_ymd": w_hist_all.get("date_cutoff_ymd"),
                "curves_n": w_hist_all.get("curves_n"),
                "segments_n": w_hist_all.get("segments_n"),
                "segments_by_array": w_hist_all.get("segments_by_array"),
            }

        # Sensitivity of KS strength to "effective n" assumptions for the historical distribution.
        if m3_out.get("ok") and isinstance(n_hist, int) and n_hist > 0 and n_model_list:
            scenarios: Dict[str, Any] = {}
            scenarios["paper5_stated_historical_n"] = {
                "n_obs": int(n_hist),
                "ks_dcrit_vs_model_n": {f"n_model={m}": _ks_two_sample_dcrit(alpha, int(n_hist), int(m)) for m in n_model_list},
            }

            if isinstance(w_hist, dict) and bool(w_hist.get("ok")):
                n_seg = w_hist.get("segments_n")
                n_curves = w_hist.get("curves_n")
                seg_by = w_hist.get("segments_by_array") if isinstance(w_hist.get("segments_by_array"), dict) else {}
                n_sma_carma = None
                if isinstance(seg_by, dict):
                    try:
                        n_sma_carma = int((seg_by.get("SMA") or 0) + (seg_by.get("CARMA") or 0))
                    except Exception:
                        n_sma_carma = None

                def _add(name: str, n_obs: Any) -> None:
                    if not isinstance(n_obs, int) or n_obs <= 0:
                        return
                    scenarios[name] = {
                        "n_obs": int(n_obs),
                        "ks_dcrit_vs_model_n": {f"n_model={m}": _ks_two_sample_dcrit(alpha, int(n_obs), int(m)) for m in n_model_list},
                    }

                _add("wielgus2022_pre_eht_segments_n", int(n_seg) if isinstance(n_seg, int) else None)
                _add("wielgus2022_pre_eht_curves_n", int(n_curves) if isinstance(n_curves, int) else None)
                _add("wielgus2022_pre_eht_sma_carma_segments_n", n_sma_carma)

            if isinstance(w_hist_all, dict) and bool(w_hist_all.get("ok")):
                n_all = w_hist_all.get("segments_n")
                if isinstance(n_all, int) and n_all > 0:
                    scenarios["wielgus2022_2017_inclusive_segments_n"] = {
                        "n_obs": int(n_all),
                        "ks_dcrit_vs_model_n": {f"n_model={m}": _ks_two_sample_dcrit(alpha, int(n_all), int(m)) for m in n_model_list},
                    }

            m3_out["wielgus2022_historical_distribution_effective_n_sensitivity"] = {
                "ok": True,
                "note": "KS D_crit values use the standard asymptotic two-sample KS approximation; larger D_crit => weaker rejection power for fixed model distribution differences.",
                "alpha": float(alpha),
                "model_samples_candidates": n_model_list,
                "scenarios": scenarios,
            }
        if isinstance(w_der, dict) and isinstance(w_der.get("deltaT_over_tau_by_row"), dict):
            m3_out["wielgus2022_deltaT_over_tau_by_row"] = w_der.get("deltaT_over_tau_by_row")
        if isinstance(w_ex, dict):
            drw = w_ex.get("drw_predicted_sigma_over_mu_3h")
            if isinstance(drw, dict) and bool(drw.get("ok")):
                m3_out["wielgus2022_drw_predicted_sigma_over_mu_3h"] = {
                    "value": drw.get("value"),
                    "plus": drw.get("plus"),
                    "minus": drw.get("minus"),
                    "source_anchor": drw.get("source_anchor"),
                }

    # Optional: integrate reconstructed historical mi3 values (full reconstruction) to quantify "margin" vs p=alpha.
    if isinstance(m3_hist_vals, dict) and bool(m3_hist_vals.get("ok")):
        data_block = m3_hist_vals.get("data") or {}
        sample_2017 = data_block.get("sample_2017_7")
        hist_values = data_block.get("historical_mi3_values")
        if isinstance(sample_2017, list) and isinstance(hist_values, list):
            d_obs = _ks_two_sample_d(sample_2017, hist_values)
            if d_obs is not None:
                n7 = len([x for x in sample_2017 if isinstance(x, (int, float))])
                nh = len([x for x in hist_values if isinstance(x, (int, float))])
                p_obs = _ks_two_sample_p_asymptotic(d_obs, n7, nh)
                dcrit = _ks_two_sample_dcrit(alpha, n7, nh) if isinstance(alpha, (int, float)) else None
                d_for_p_alpha = _ks_solve_d_for_p_asymptotic(alpha, n7, nh) if isinstance(alpha, (int, float)) else None

                reported = ((m3_hist_vals.get("derived") or {}).get("ks_2017_7sample_vs_historical") or {})
                rep_d = reported.get("d") if isinstance(reported, dict) else None
                rep_p = reported.get("p_asymptotic") if isinstance(reported, dict) else None

                hist_ks: Dict[str, Any] = {
                    "ok": True,
                    "alpha": float(alpha),
                    "n_2017": int(n7),
                    "n_historical": int(nh),
                    "d": float(d_obs),
                    "p_asymptotic": float(p_obs) if p_obs is not None else None,
                    "dcrit_asymptotic": float(dcrit) if dcrit is not None else None,
                    "margin": {
                        "p_minus_alpha": (float(p_obs) - float(alpha)) if p_obs is not None else None,
                        "dcrit_minus_d": (float(dcrit) - float(d_obs)) if dcrit is not None else None,
                        "d_for_p_eq_alpha": float(d_for_p_alpha) if d_for_p_alpha is not None else None,
                        "d_delta_to_p_eq_alpha": (float(d_for_p_alpha) - float(d_obs)) if d_for_p_alpha is not None else None,
                    },
                    "source_json_reported": {
                        "d": float(rep_d) if isinstance(rep_d, (int, float)) else None,
                        "p_asymptotic": float(rep_p) if isinstance(rep_p, (int, float)) else None,
                        "note": reported.get("note") if isinstance(reported, dict) else None,
                    },
                }

                # Discreteness/fragility: D can only change in multiples of 1/lcm(n,m).
                try:
                    import math as _math

                    g = int(_math.gcd(int(n7), int(nh)))
                    lcm = int(int(n7) * int(nh) / g) if g > 0 else None
                    if isinstance(lcm, int) and lcm > 0:
                        d_step = 1.0 / float(lcm)
                        hist_ks["d_discreteness"] = {
                            "lcm_n_2017_n_historical": int(lcm),
                            "d_step": float(d_step),
                            "d_next_up": float(d_obs + d_step),
                            "p_next_up_asymptotic": _ks_two_sample_p_asymptotic(float(d_obs + d_step), int(n7), int(nh)),
                            "d_next_down": float(max(0.0, d_obs - d_step)),
                            "p_next_down_asymptotic": _ks_two_sample_p_asymptotic(float(max(0.0, d_obs - d_step)), int(n7), int(nh)),
                        }
                except Exception:
                    pass

                # Identify where (t,i,j) the max D is achieved to derive a concrete "one-step flip" condition.
                try:
                    s = sorted(float(x) for x in sample_2017 if isinstance(x, (int, float)))
                    h = sorted(float(x) for x in hist_values if isinstance(x, (int, float)))
                    combined = sorted(set(s + h))
                    i = 0
                    j = 0
                    best_t = None
                    best_i = None
                    best_j = None
                    best_d = -1.0
                    for t in combined:
                        while i < len(s) and s[i] <= t:
                            i += 1
                        while j < len(h) and h[j] <= t:
                            j += 1
                        d_here = abs((i / len(s)) - (j / len(h))) if (len(s) > 0 and len(h) > 0) else None
                        if d_here is None:
                            continue
                        if d_here > best_d + 1e-15:
                            best_d = float(d_here)
                            best_t = float(t)
                            best_i = int(i)
                            best_j = int(j)
                    if best_t is not None and best_i is not None and best_j is not None and best_d >= 0.0:
                        hist_ks["d_argmax"] = {
                            "t": float(best_t),
                            "i_2017_le_t": int(best_i),
                            "j_hist_le_t": int(best_j),
                            "i_over_n": float(best_i / len(s)) if len(s) else None,
                            "j_over_m": float(best_j / len(h)) if len(h) else None,
                            "d_at_t": float(best_d),
                            "signed_diff_i_over_n_minus_j_over_m": float((best_i / len(s)) - (best_j / len(h))) if (len(s) and len(h)) else None,
                        }

                        # If the maximal D is attained at a 2017 sample point t, then moving one historical point
                        # across t flips j by ±1 and changes D by 1/lcm(n,m). This quantifies the "digitize" fragility.
                        if 0 < best_j < len(h):
                            v_le = float(h[best_j - 1])
                            v_gt = float(h[best_j])
                            if v_le > 0 and v_gt > 0:
                                scale_up = float(best_t / v_le) if v_le != 0.0 else None
                                scale_down = float(best_t / v_gt) if v_gt != 0.0 else None
                                if isinstance(scale_up, float) and isinstance(scale_down, float):
                                    eps = 1e-9
                                    scaled_up = [float(x) * float(scale_up * (1.0 + eps)) for x in h]
                                    scaled_dn = [float(x) * float(scale_down * (1.0 - eps)) for x in h]
                                    d_up = _ks_two_sample_d(s, scaled_up)
                                    d_dn = _ks_two_sample_d(s, scaled_dn)
                                    hist_ks["digitize_scale_thresholds"] = {
                                        "t": float(best_t),
                                        "hist_le_t_count": int(best_j),
                                        "hist_le_t_max_value": float(v_le),
                                        "hist_gt_t_min_value": float(v_gt),
                                        "delta_abs_to_move_hist_le_max_to_t": float(best_t - v_le),
                                        "delta_rel_to_move_hist_le_max_to_t": float((best_t / v_le) - 1.0) if v_le != 0.0 else None,
                                        "scale_up_to_decrease_hist_le_count_by_1": float(scale_up),
                                        "scale_down_to_increase_hist_le_count_by_1": float(scale_down),
                                        "scenario_scale_up_eps": {
                                            "scale": float(scale_up * (1.0 + eps)),
                                            "d": float(d_up) if d_up is not None else None,
                                            "p_asymptotic": _ks_two_sample_p_asymptotic(float(d_up), len(s), len(scaled_up)) if d_up is not None else None,
                                        },
                                        "scenario_scale_down_eps": {
                                            "scale": float(scale_down * (1.0 - eps)),
                                            "d": float(d_dn) if d_dn is not None else None,
                                            "p_asymptotic": _ks_two_sample_p_asymptotic(float(d_dn), len(s), len(scaled_dn)) if d_dn is not None else None,
                                        },
                                        "note": "Uniform scaling of mi3 values is a deterministic proxy for digitization/systematic bias (mi3 is dimensionless).",
                                    }
                except Exception:
                    pass

                # Sensitivity: interpret correlations as reduced effective n_hist (keep D fixed; recompute p/dcrit).
                scenarios_eff_n = {}
                eff_scen = (m3_out.get("wielgus2022_historical_distribution_effective_n_sensitivity") or {}).get(
                    "scenarios"
                )
                if isinstance(eff_scen, dict):
                    for name, block in eff_scen.items():
                        n_eff = (block or {}).get("n_obs") if isinstance(block, dict) else None
                        if not isinstance(n_eff, int) or n_eff <= 0:
                            continue
                        scenarios_eff_n[name] = {
                            "n_historical_effective": int(n_eff),
                            "p_asymptotic": _ks_two_sample_p_asymptotic(d_obs, n7, int(n_eff)),
                            "dcrit_asymptotic": _ks_two_sample_dcrit(alpha, n7, int(n_eff)),
                        }
                if scenarios_eff_n:
                    hist_ks["effective_n_scenarios"] = scenarios_eff_n
                    hist_ks["effective_n_note"] = "Keep D fixed at the reconstructed value; recompute p/dcrit as if historical sample size were n_eff."

                # Digitization/scale sensitivity checks (deterministic): rounding and global scaling of historical mi3.
                digitize_checks: Dict[str, Any] = {}

                round_rows = []
                for dec in (3, 2, 1):
                    hist_round = [round(float(x), int(dec)) for x in hist_values if isinstance(x, (int, float))]
                    d_r = _ks_two_sample_d(sample_2017, hist_round)
                    if d_r is None:
                        continue
                    p_r = _ks_two_sample_p_asymptotic(d_r, n7, len(hist_round))
                    round_rows.append(
                        {
                            "round_decimals": int(dec),
                            "n_historical": int(len(hist_round)),
                            "d": float(d_r),
                            "p_asymptotic": float(p_r) if p_r is not None else None,
                            "p_minus_alpha": (float(p_r) - float(alpha)) if p_r is not None else None,
                        }
                    )
                if round_rows:
                    digitize_checks["round_historical"] = round_rows

                scale_rows = []
                for f in (0.9, 0.95, 1.0, 1.05, 1.1):
                    hist_scaled = [float(x) * float(f) for x in hist_values if isinstance(x, (int, float))]
                    d_s = _ks_two_sample_d(sample_2017, hist_scaled)
                    if d_s is None:
                        continue
                    p_s = _ks_two_sample_p_asymptotic(d_s, n7, len(hist_scaled))
                    scale_rows.append(
                        {
                            "scale_factor_historical": float(f),
                            "n_historical": int(len(hist_scaled)),
                            "d": float(d_s),
                            "p_asymptotic": float(p_s) if p_s is not None else None,
                            "p_minus_alpha": (float(p_s) - float(alpha)) if p_s is not None else None,
                        }
                    )
                if scale_rows:
                    digitize_checks["scale_historical"] = scale_rows

                if digitize_checks:
                    hist_ks["digitize_sensitivity_checks"] = digitize_checks
                    hist_ks["digitize_note"] = "These are deterministic robustness checks; they do not represent a calibrated uncertainty model."

                m3_out["historical_distribution_values_ks"] = hist_ks

    # Extended flux mapping (from Paper V text): mi_obs = mi_true / (1+f_ext)
    ext_out: Dict[str, Any] = {"ok": True}
    if sens.get("extended_flux_suppression_factor") == "1/(1+f_ext)":
        # Provide a compact set of mappings; if key-metrics already has a list, reuse that.
        derived_list = (key.get("derived") or {}).get("m3_suppression_factor_s_to_f_ext")
        if isinstance(derived_list, list) and derived_list:
            ext_out["m3_suppression_factor_s_to_f_ext"] = derived_list
        else:
            ss = [0.9, 0.8, 0.7, 0.6]
            ext_out["m3_suppression_factor_s_to_f_ext"] = [
                {
                    "suppression_factor_s": float(s),
                    "f_ext_over_compact": float((1.0 / float(s)) - 1.0),
                    "extended_flux_fraction_of_total": float(1.0 - float(s)),
                }
                for s in ss
            ]
        # Highlight the frequently mentioned ~30% scale (s=0.7).
        s = 0.7
        ext_out["for_s=0.70"] = {
            "suppression_factor_s": s,
            "f_ext_over_compact": float((1.0 / s) - 1.0),
            "extended_flux_fraction_of_total": float(1.0 - s),
        }
    else:
        ext_out["ok"] = False
        ext_out["reason"] = "extended_flux_factor_not_found_in_inputs"

    # --- Near-passing salvage counts (directly from near_passing tables) ---
    near_rows = _near_passing_rows(pf)
    near_combined = _extract_near_combo_counts(pf)
    near_out: Dict[str, Any] = {
        "rows_total_n": len(near_rows),
        "constraints_ranked_by_count": near_combined.get("constraints_ranked_by_count"),
        "salvage_if_relax": {
            "relax_M3": _count_relax_pass(near_rows, ["M3"]),
            "relax_F_2um": _count_relax_pass(near_rows, ["F_2um"]),
            "relax_M3_plus_F_2um": _count_relax_pass(near_rows, ["M3", "F_2um"]),
        },
    }

    # --- Global sweep (upper bound on effect; based on Pass/Fail tables only) ---
    glob = (sweep.get("derived") or {}).get("global") or {}
    global_out = {
        "rows_total_n": (sweep.get("extracted") or {}).get("rows_total_n"),
        "baseline": glob.get("baseline"),
        "relax_M3": glob.get("relax_M3"),
        "relax_F_2um": glob.get("relax_F_2um"),
        "relax_M3_plus_F_2um": glob.get("relax_M3_plus_F_2um"),
    }

    payload["derived"] = {
        "nir_2p2um": nir_out,
        "m3": m3_out,
        "m3_extended_flux_mapping": ext_out,
        "near_passing_salvage": near_out,
        "global_relaxation_sweep": global_out,
        "notes": {
            "scope": "This output quantifies sensitivity knobs; it does NOT compute model-by-model required threshold shifts because numeric model F_2um and mi3 distributions are not in the Pass/Fail tables.",
            "ks_test": "KS D_crit and p_asymptotic use the standard asymptotic two-sample KS approximation; Paper V implementation details may differ.",
        },
    }

    if not bool(args.no_plot):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            out_png.parent.mkdir(parents=True, exist_ok=True)

            fig, axes = plt.subplots(2, 2, figsize=(12, 7))

            # (A) NIR
            ax = axes[0, 0]
            ax.set_title("2.2 μm median flux threshold (obs vs threshold)")
            ax.set_xlabel("mJy")
            ax.set_yticks([])
            if nir_out.get("ok"):
                med = float(nir_out["observed_median_mJy"])
                sig = float(nir_out["observed_sigma_mJy"])
                thr = float(nir_out["threshold_mJy"])
                ax.errorbar([med], [0], xerr=[sig], fmt="o", color="black", capsize=4, label="median ±1σ")
                ax.axvline(thr, color="tab:red", linestyle="--", label="threshold (1.0 mJy)")
                # Context from GRAVITY 2020 percentiles (Table 1): p86/p95 (2017) and p5 (2017-2019 avg).
                gby = nir_out.get("gravity2020_percentiles_avg_by_year") if isinstance(nir_out.get("gravity2020_percentiles_avg_by_year"), dict) else None
                if isinstance(gby, dict):
                    p86 = (((gby.get("2017") or {}).get("p86")) or {})
                    p95 = (((gby.get("2017") or {}).get("p95")) or {})
                    p5_all = (((gby.get("2017-2019") or {}).get("p5")) or {})
                    if isinstance(p86, dict) and isinstance(p86.get("value_mJy"), (int, float)):
                        ax.axvline(float(p86["value_mJy"]), color="#666666", linestyle="-.", linewidth=1, label="GRAVITY20 p86 (2017)")
                    if isinstance(p95, dict) and isinstance(p95.get("value_mJy"), (int, float)):
                        ax.axvline(float(p95["value_mJy"]), color="#999999", linestyle="-.", linewidth=1, label="GRAVITY20 p95 (2017)")
                    if isinstance(p5_all, dict) and (not bool(p5_all.get("struck_out"))) and isinstance(p5_all.get("value_mJy"), (int, float)):
                        ax.axvline(float(p5_all["value_mJy"]), color="#bbbbbb", linestyle=":", linewidth=1, label="GRAVITY20 p5 (2017-2019 avg)")
                for k, col in [(1, "#777777"), (2, "#999999"), (3, "#bbbbbb")]:
                    ax.axvline(med + k * sig, color=col, linestyle=":", linewidth=1)
                ax.legend(loc="lower right", fontsize=8)

            # (B) M3 KS Dcrit sensitivity to sample size
            ax = axes[0, 1]
            ax.set_title("M3 KS-test tolerance (proxy; D_crit)")
            ax.set_xlabel("model samples (n_model)")
            ax.set_ylabel("D_crit (approx)")
            if m3_out.get("ok"):
                ms = [int(x) for x in m3_out.get("model_samples_candidates") or []]
                dh = [m3_out["ks_dcrit_hist_vs_model_n"].get(f"n_model={m}") for m in ms]
                d7 = [m3_out["ks_dcrit_2017_vs_model_n"].get(f"n_model={m}") for m in ms]
                ax.plot(ms, dh, marker="o", label=f"obs n={m3_out['observed_samples_historical_n']}")
                ax.plot(ms, d7, marker="o", label=f"obs n={m3_out['observed_samples_2017_n']}")
                ks_vals = m3_out.get("historical_distribution_values_ks")
                if isinstance(ks_vals, dict) and bool(ks_vals.get("ok")):
                    d = ks_vals.get("d")
                    p = ks_vals.get("p_asymptotic")
                    a = ks_vals.get("alpha")
                    if isinstance(d, (int, float)) and isinstance(p, (int, float)) and isinstance(a, (int, float)):
                        ax.text(
                            0.02,
                            0.98,
                            f"2017 vs hist (values):\\nD={d:.3f}, p={p:.4f} (α={a:.2g})",
                            transform=ax.transAxes,
                            ha="left",
                            va="top",
                            fontsize=8,
                            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
                        )
                ax.legend(fontsize=8)

            # (C) near-passing salvage counts
            ax = axes[1, 0]
            ax.set_title("Near-passing (45 rows): salvage counts")
            labels = ["relax M3", "relax F_2um", "relax both"]
            vals = [
                near_out["salvage_if_relax"]["relax_M3"],
                near_out["salvage_if_relax"]["relax_F_2um"],
                near_out["salvage_if_relax"]["relax_M3_plus_F_2um"],
            ]
            ax.bar(labels, vals, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
            ax.set_ylabel("models that would pass (count)")
            ax.set_ylim(0, max(vals) * 1.25 + 1)
            for i, v in enumerate(vals):
                ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
            ax.tick_params(axis="x", labelrotation=15)

            # (D) global sweep
            ax = axes[1, 1]
            ax.set_title("All Pass/Fail tables (1924 rows): pass_n under relax")
            labels = ["baseline", "relax M3", "relax F_2um", "relax both"]
            v0 = int((global_out.get("baseline") or {}).get("pass_n") or 0)
            v1 = int((global_out.get("relax_M3") or {}).get("pass_n") or 0)
            v2 = int((global_out.get("relax_F_2um") or {}).get("pass_n") or 0)
            v3 = int((global_out.get("relax_M3_plus_F_2um") or {}).get("pass_n") or 0)
            vals = [v0, v1, v2, v3]
            ax.bar(labels, vals, color=["#777777", "#1f77b4", "#ff7f0e", "#2ca02c"])
            ax.set_ylabel("pass_n (count)")
            ax.set_ylim(0, max(vals) * 1.25 + 1)
            for i, v in enumerate(vals):
                ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
            ax.tick_params(axis="x", labelrotation=15)

            fig.tight_layout()
            fig.savefig(out_png, dpi=160)
            plt.close(fig)
        except Exception:
            pass

    _write_json(out_json, payload)

    try:
        worklog.append_event(
            {
                "ts_utc": payload["generated_utc"],
                "topic": "eht",
                "action": "eht_sgra_paper5_m3_nir_reconnection_conditions",
                "outputs": [
                    str(out_json.relative_to(root)).replace("\\", "/"),
                    str(out_png.relative_to(root)).replace("\\", "/"),
                ],
                "metrics": {
                    "ok": bool(payload.get("ok")),
                    "near_rows_n": int(payload.get("derived", {}).get("near_passing_salvage", {}).get("rows_total_n") or 0),
                },
            }
        )
    except Exception:
        pass

    print(f"[ok] json: {out_json}")
    if out_png.exists():
        print(f"[ok] png : {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
