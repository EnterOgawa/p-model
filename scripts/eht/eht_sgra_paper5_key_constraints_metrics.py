#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


# 関数: `_repo_root` の入出力契約と処理意図を定義する。
def _repo_root() -> Path:
    return _ROOT


# 関数: `_read_lines` の入出力契約と処理意図を定義する。

def _read_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


# 関数: `_write_json` の入出力契約と処理意図を定義する。

def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# 関数: `_find_first` の入出力契約と処理意図を定義する。

def _find_first(lines: Sequence[str], pattern: re.Pattern[str]) -> Optional[Tuple[int, re.Match[str]]]:
    for i, line in enumerate(lines, start=1):
        m = pattern.search(line)
        # 条件分岐: `m` を満たす経路を評価する。
        if m:
            return (i, m)

    return None


# 関数: `_maybe_float` の入出力契約と処理意図を定義する。

def _maybe_float(x: str) -> Optional[float]:
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


# 関数: `_anchor_snippet` の入出力契約と処理意図を定義する。

def _anchor_snippet(line: str, match: Optional[re.Match[str]] = None, *, max_len: int = 240) -> str:
    s = line.rstrip("\n")
    # 条件分岐: `match is None` を満たす経路を評価する。
    if match is None:
        return s.strip()[:max_len]

    center = (match.start() + match.end()) // 2
    half = max_len // 2
    start = max(0, center - half)
    end = min(len(s), start + max_len)
    start = max(0, end - max_len)
    return s[start:end].strip()


# 関数: `_get_anchor` の入出力契約と処理意図を定義する。

def _get_anchor(
    path: Path, line: int, *, label: str, snippet: str, match: Optional[re.Match[str]] = None
) -> Dict[str, Any]:
    return {"path": str(path), "line": int(line), "label": label, "snippet": _anchor_snippet(snippet, match)}


# 関数: `_ks_c_alpha` の入出力契約と処理意図を定義する。

def _ks_c_alpha(alpha: float) -> Optional[float]:
    # Standard 2-sample KS asymptotic constants (commonly tabulated).
    if abs(alpha - 0.10) < 1e-12:
        return 1.22

    # 条件分岐: `abs(alpha - 0.05) < 1e-12` を満たす経路を評価する。

    if abs(alpha - 0.05) < 1e-12:
        return 1.36

    # 条件分岐: `abs(alpha - 0.01) < 1e-12` を満たす経路を評価する。

    if abs(alpha - 0.01) < 1e-12:
        return 1.63

    # 条件分岐: `abs(alpha - 0.001) < 1e-12` を満たす経路を評価する。

    if abs(alpha - 0.001) < 1e-12:
        return 1.95

    return None


# 関数: `_ks_two_sample_dcrit` の入出力契約と処理意図を定義する。

def _ks_two_sample_dcrit(alpha: float, n: int, m: int) -> Optional[float]:
    # 条件分岐: `n <= 0 or m <= 0` を満たす経路を評価する。
    if n <= 0 or m <= 0:
        return None

    c = _ks_c_alpha(alpha)
    # 条件分岐: `c is None` を満たす経路を評価する。
    if c is None:
        return None

    return float(c * math.sqrt((n + m) / (n * m)))


# 関数: `_parse_nir_constraint` の入出力契約と処理意図を定義する。

def _parse_nir_constraint(observations_tex: Path) -> Dict[str, Any]:
    lines = _read_lines(observations_tex)

    obs_pat = re.compile(r"median\s+2\.2\\um\s+flux\s+\$=\s*([0-9.]+)\s*\\pm\s*([0-9.]+)\\,\\mathrm\{mJy\}")
    thr_pat = re.compile(r"threshold of\s+\$([0-9.]+)\$\\,mJy")

    obs_m = _find_first(lines, obs_pat)
    thr_m = _find_first(lines, thr_pat)

    out: Dict[str, Any] = {"ok": True, "observed_median_mJy": None, "observed_sigma_mJy": None, "threshold_mJy": None}

    # 条件分岐: `obs_m is None` を満たす経路を評価する。
    if obs_m is None:
        out["ok"] = False
        out["missing"] = (out.get("missing") or []) + ["observed_median_line_not_found"]
    else:
        lineno, m = obs_m
        median = _maybe_float(m.group(1))
        sigma = _maybe_float(m.group(2))
        out["observed_median_mJy"] = median
        out["observed_sigma_mJy"] = sigma
        out["observed_anchor"] = _get_anchor(
            observations_tex,
            lineno,
            label="nir_observed_median_2017",
            snippet=lines[lineno - 1],
            match=m,
        )

    # 条件分岐: `thr_m is None` を満たす経路を評価する。

    if thr_m is None:
        out["ok"] = False
        out["missing"] = (out.get("missing") or []) + ["threshold_line_not_found"]
    else:
        lineno, m = thr_m
        thr = _maybe_float(m.group(1))
        out["threshold_mJy"] = thr
        out["threshold_anchor"] = _get_anchor(
            observations_tex,
            lineno,
            label="nir_threshold_conservative",
            snippet=lines[lineno - 1],
            match=m,
        )

    # Policy statement (reject if median exceeds threshold) exists nearby; keep a short anchor.

    policy_pat = re.compile(r"reject the model if its\s*median\s*2\.2\\um\s*flux density exceeds threshold", re.IGNORECASE)
    pol_m = _find_first(lines, policy_pat)
    # 条件分岐: `pol_m is not None` を満たす経路を評価する。
    if pol_m is not None:
        lineno, _ = pol_m
        out["reject_policy_anchor"] = _get_anchor(
            observations_tex,
            lineno,
            label="nir_reject_if_above_threshold",
            snippet=lines[lineno - 1],
        )

    return out


# 関数: `_parse_m3_constraint` の入出力契約と処理意図を定義する。

def _parse_m3_constraint(observations_tex: Path) -> Dict[str, Any]:
    lines = _read_lines(observations_tex)

    dt_pat = re.compile(r"\\Delta\s*T\s*=\s*([0-9.]+)\s*\$?\\,hours")
    tg_pat = re.compile(r"\(\s*\$?\\sim\s*([0-9.]+)\\,\\tg\s*\$?\)")
    ks_pat = re.compile(r"reject the model if\s*\$p\s*<\s*([0-9.]+)\$")
    samp_pat = re.compile(r"provide\s+(\d+)\s+samples.*?yields\s+(\d+)\s+samples")
    model_samp_pat = re.compile(r"\((\d+)\s+or\s+(\d+)\s+samples\).*?\((\d+)\s+samples\)")

    out: Dict[str, Any] = {"ok": True}

    dt_m = _find_first(lines, dt_pat)
    # 条件分岐: `dt_m is None` を満たす経路を評価する。
    if dt_m is None:
        # Fallback: infer from the "3-hour modulation index" phrase if present.
        dt_fallback_pat = re.compile(r"using the\s+([0-9.]+)-hour\s+\{\\em\s+modulation index\}", re.IGNORECASE)
        dt_fb_m = _find_first(lines, dt_fallback_pat)
        # 条件分岐: `dt_fb_m is None` を満たす経路を評価する。
        if dt_fb_m is None:
            out["ok"] = False
            out["missing"] = (out.get("missing") or []) + ["deltaT_hours_not_found"]
        else:
            lineno, m = dt_fb_m
            out["deltaT_hours"] = _maybe_float(m.group(1))
            out["deltaT_hours_anchor"] = _get_anchor(
                observations_tex, lineno, label="m3_deltaT_hours_fallback", snippet=lines[lineno - 1], match=m
            )
    else:
        lineno, m = dt_m
        out["deltaT_hours"] = _maybe_float(m.group(1))
        out["deltaT_hours_anchor"] = _get_anchor(
            observations_tex, lineno, label="m3_deltaT_hours", snippet=lines[lineno - 1], match=m
        )

    tg_m = _find_first(lines, tg_pat)
    # 条件分岐: `tg_m is not None` を満たす経路を評価する。
    if tg_m is not None:
        lineno, m = tg_m
        out["deltaT_tg_approx"] = _maybe_float(m.group(1))
        out["deltaT_tg_anchor"] = _get_anchor(observations_tex, lineno, label="m3_deltaT_tg", snippet=lines[lineno - 1], match=m)

    ks_m = _find_first(lines, ks_pat)
    # 条件分岐: `ks_m is None` を満たす経路を評価する。
    if ks_m is None:
        out["ok"] = False
        out["missing"] = (out.get("missing") or []) + ["ks_reject_threshold_not_found"]
    else:
        lineno, m = ks_m
        out["ks_reject_if_p_lt"] = _maybe_float(m.group(1))
        out["ks_anchor"] = _get_anchor(observations_tex, lineno, label="m3_ks_p_threshold", snippet=lines[lineno - 1], match=m)

    samp_m = _find_first(lines, samp_pat)
    # 条件分岐: `samp_m is not None` を満たす経路を評価する。
    if samp_m is not None:
        lineno, m = samp_m
        out["observed_samples_2017_n"] = int(m.group(1))
        out["observed_samples_historical_n"] = int(m.group(2))
        out["observed_samples_anchor"] = _get_anchor(
            observations_tex, lineno, label="m3_observed_samples", snippet=lines[lineno - 1]
        )

    model_samp_m = _find_first(lines, model_samp_pat)
    # 条件分岐: `model_samp_m is not None` を満たす経路を評価する。
    if model_samp_m is not None:
        lineno, m = model_samp_m
        out["model_samples_fiducial_n"] = [int(m.group(1)), int(m.group(2))]
        out["model_samples_exploratory_n"] = int(m.group(3))
        out["model_samples_anchor"] = _get_anchor(
            observations_tex, lineno, label="m3_model_samples_by_duration", snippet=lines[lineno - 1]
        )

    # Definition line (mi = sigma/mu)

    def_pat = re.compile(r"\\mi\{3\}.*?\\mi\{\\Delta T\}\s*\\equiv\s*\\sigma_", re.IGNORECASE)
    def_m = _find_first(lines, def_pat)
    # 条件分岐: `def_m is not None` を満たす経路を評価する。
    if def_m is not None:
        lineno, _ = def_m
        out["definition_anchor"] = _get_anchor(observations_tex, lineno, label="m3_definition", snippet=lines[lineno - 1])

    return out


# 関数: `_parse_m3_sensitivity` の入出力契約と処理意図を定義する。

def _parse_m3_sensitivity(*, discussion_tex: Path, conclusions_tex: Path) -> Dict[str, Any]:
    disc_lines = _read_lines(discussion_tex)
    conc_lines = _read_lines(conclusions_tex)

    out: Dict[str, Any] = {"ok": True, "notes": []}

    # 1/(1+f_ext) factor
    fac_pat = re.compile(r"factor of\s*\$1/\(1\s*\+\s*f_\\mathrm\{ext\}\)\$")
    fac_m = _find_first(disc_lines, fac_pat)
    # 条件分岐: `fac_m is not None` を満たす経路を評価する。
    if fac_m is not None:
        lineno, _ = fac_m
        out["extended_flux_suppression_factor"] = "1/(1+f_ext)"
        out["extended_flux_anchor"] = _get_anchor(
            discussion_tex, lineno, label="m3_extended_flux_suppression", snippet=disc_lines[lineno - 1]
        )

    # "A reduction of ~30% in the compact flux"

    comp_pat = re.compile(r"reduction of\s*\$?\\sim\s*30\\%\$?\s*in the compact flux", re.IGNORECASE)
    comp_m = _find_first(disc_lines, comp_pat)
    # 条件分岐: `comp_m is not None` を満たす経路を評価する。
    if comp_m is not None:
        lineno, m = comp_m
        out["compact_flux_change_fraction"] = 0.30
        out["compact_flux_change_anchor"] = _get_anchor(
            discussion_tex, lineno, label="compact_flux_change_30pct", snippet=disc_lines[lineno - 1], match=m
        )

    # "a 15% change in model density normalization"

    dens_pat = re.compile(r"a\s*\$?([0-9.]+)\\%\$?\s*change in model density normalization", re.IGNORECASE)
    dens_m = _find_first(disc_lines, dens_pat)
    # 条件分岐: `dens_m is not None` を満たす経路を評価する。
    if dens_m is not None:
        lineno, m = dens_m
        out["density_normalization_change_percent"] = _maybe_float(m.group(1))
        out["density_normalization_change_anchor"] = _get_anchor(
            discussion_tex, lineno, label="density_normalization_change_percent", snippet=disc_lines[lineno - 1], match=m
        )

    # "~ 30% in mi3 is sufficient ..."

    m3_pat = re.compile(r"30\\%\s+in\s+\\mi\{3\}", re.IGNORECASE)
    m3_m = _find_first(disc_lines, m3_pat)
    # 条件分岐: `m3_m is not None` を満たす経路を評価する。
    if m3_m is not None:
        lineno, m = m3_m
        out["m3_change_fraction_sufficient_to_flip_models"] = 0.30
        out["m3_change_anchor"] = _get_anchor(
            discussion_tex, lineno, label="m3_change_30pct_sufficient", snippet=disc_lines[lineno - 1], match=m
        )

    # "reduce mi3 by 30%" (conclusions)

    conc_pat = re.compile(r"reduce\s+\\mi\{3\}\s+by\s+30\\%", re.IGNORECASE)
    conc_m = _find_first(conc_lines, conc_pat)
    # 条件分岐: `conc_m is not None` を満たす経路を評価する。
    if conc_m is not None:
        lineno, m = conc_m
        out["m3_reduction_fraction_stated_in_conclusions"] = 0.30
        out["conclusions_anchor"] = _get_anchor(
            conclusions_tex, lineno, label="m3_reduce_30pct_conclusions", snippet=conc_lines[lineno - 1], match=m
        )

    return out


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[Sequence[str]] = None) -> int:
    root = _repo_root()
    default_obs = root / "data" / "eht" / "sources" / "arxiv_2311.09478" / "observations.tex"
    default_disc = root / "data" / "eht" / "sources" / "arxiv_2311.09478" / "discussion.tex"
    default_conc = root / "data" / "eht" / "sources" / "arxiv_2311.09478" / "conclusions.tex"
    default_out = root / "output" / "private" / "eht" / "eht_sgra_paper5_key_constraints_metrics.json"
    default_png = root / "output" / "private" / "eht" / "eht_sgra_paper5_key_constraints_sensitivity.png"

    ap = argparse.ArgumentParser(description="Extract Paper V key constraint definitions (M3, 2.2um) from TeX sources.")
    ap.add_argument("--observations-tex", type=str, default=str(default_obs))
    ap.add_argument("--discussion-tex", type=str, default=str(default_disc))
    ap.add_argument("--conclusions-tex", type=str, default=str(default_conc))
    ap.add_argument("--out", type=str, default=str(default_out))
    ap.add_argument("--out-png", type=str, default=str(default_png))
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    observations_tex = Path(args.observations_tex)
    discussion_tex = Path(args.discussion_tex)
    conclusions_tex = Path(args.conclusions_tex)
    out_json = Path(args.out)
    out_png = Path(args.out_png)

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "ok": True,
        "inputs": {
            "observations_tex": str(observations_tex),
            "discussion_tex": str(discussion_tex),
            "conclusions_tex": str(conclusions_tex),
        },
        "extracted": {},
        "derived": {},
        "outputs": {"json": str(out_json), "png": str(out_png)},
    }

    missing_files = [str(p) for p in (observations_tex, discussion_tex, conclusions_tex) if not p.exists()]
    # 条件分岐: `missing_files` を満たす経路を評価する。
    if missing_files:
        payload["ok"] = False
        payload["reason"] = "missing_input_tex"
        payload["missing_files"] = missing_files
        _write_json(out_json, payload)
        print(f"[warn] missing inputs; wrote: {out_json}")
        return 0

    nir = _parse_nir_constraint(observations_tex)
    m3 = _parse_m3_constraint(observations_tex)
    sens = _parse_m3_sensitivity(discussion_tex=discussion_tex, conclusions_tex=conclusions_tex)
    payload["extracted"] = {"nir_2p2um": nir, "m3": m3, "m3_sensitivity": sens}

    # Derived quick summaries (do not assume these are the only interpretations).
    derived: Dict[str, Any] = {}
    if (
        nir.get("observed_median_mJy") is not None
        and nir.get("observed_sigma_mJy") not in (None, 0.0)
        and nir.get("threshold_mJy") is not None
    ):
        med = float(nir["observed_median_mJy"])
        sig = float(nir["observed_sigma_mJy"])
        thr = float(nir["threshold_mJy"])
        derived["nir_threshold_minus_median_sigma"] = float((thr - med) / sig)
        derived["nir_threshold_over_median"] = float(thr / med) if med != 0.0 else None
        derived["nir_threshold_as_median_plus_k_sigma"] = float((thr - med) / sig)
        derived["nir_threshold_for_k_sigma"] = {
            "k=1": float(med + 1.0 * sig),
            "k=2": float(med + 2.0 * sig),
            "k=3": float(med + 3.0 * sig),
        }

    # If suppression factor is 1/(1+f_ext), compute f_ext for a 30% suppression (factor 0.7).

    if sens.get("extended_flux_suppression_factor") and sens.get("m3_change_fraction_sufficient_to_flip_models") == 0.30:
        factor = 0.70
        derived["f_ext_for_m3_factor_0p70"] = float((1.0 / factor) - 1.0)
        derived["extended_flux_fraction_of_total_for_m3_factor_0p70"] = float(1.0 - factor)
        derived["m3_suppression_factor_s_to_f_ext"] = [
            {
                "suppression_factor_s": float(s),
                "f_ext_over_compact": float((1.0 / float(s)) - 1.0),
                "extended_flux_fraction_of_total": float(1.0 - float(s)),
            }
            for s in (0.9, 0.8, 0.7, 0.6)
        ]

    # If "compact flux change fraction" is present, compute density normalization factor assuming F ~ rho^2.

    cf = sens.get("compact_flux_change_fraction")
    # 条件分岐: `isinstance(cf, (int, float)) and 0.0 < float(cf) < 1.0` を満たす経路を評価する。
    if isinstance(cf, (int, float)) and 0.0 < float(cf) < 1.0:
        flux_factor = float(1.0 - float(cf))
        derived["density_normalization_factor_if_F230_propto_rho2"] = float(math.sqrt(flux_factor))
        derived["density_normalization_change_percent_if_F230_propto_rho2"] = float(100.0 * (math.sqrt(flux_factor) - 1.0))

    # KS-test "power" proxy: D_crit depends on sample sizes (n_obs, n_model).

    obs_n = m3.get("observed_samples_historical_n") or m3.get("observed_samples_2017_n")
    fid = m3.get("model_samples_fiducial_n")
    exp = m3.get("model_samples_exploratory_n")
    # 条件分岐: `isinstance(obs_n, int) and obs_n > 0 and (isinstance(fid, list) or isinstance...` を満たす経路を評価する。
    if isinstance(obs_n, int) and obs_n > 0 and (isinstance(fid, list) or isinstance(exp, int)):
        alpha = float(m3.get("ks_reject_if_p_lt") or 0.01)
        derived["m3_ks_two_sample_dcrit_approx"] = {
            "alpha": alpha,
            "c_alpha": _ks_c_alpha(alpha),
            "obs_n": int(obs_n),
            "dcrit_by_model_n": {
                "exploratory_n": _ks_two_sample_dcrit(alpha, int(obs_n), int(exp)) if isinstance(exp, int) else None,
                "fiducial_n_min": _ks_two_sample_dcrit(alpha, int(obs_n), int(min(fid))) if isinstance(fid, list) and fid else None,
                "fiducial_n_max": _ks_two_sample_dcrit(alpha, int(obs_n), int(max(fid))) if isinstance(fid, list) and fid else None,
            },
        }

    payload["derived"] = derived

    # 条件分岐: `not bool(args.no_plot)` を満たす経路を評価する。
    if not bool(args.no_plot):
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            out_png.parent.mkdir(parents=True, exist_ok=True)

            fig, axes = plt.subplots(1, 3, figsize=(14, 4))

            # (1) NIR threshold vs observed median
            ax = axes[0]
            med = nir.get("observed_median_mJy")
            sig = nir.get("observed_sigma_mJy")
            thr = nir.get("threshold_mJy")
            ax.set_title("2.2 μm median flux (GRAVITY)")
            ax.set_xlabel("mJy")
            ax.set_yticks([])
            # 条件分岐: `isinstance(med, (int, float)) and isinstance(sig, (int, float))` を満たす経路を評価する。
            if isinstance(med, (int, float)) and isinstance(sig, (int, float)):
                ax.errorbar([float(med)], [0], xerr=[float(sig)], fmt="o", color="black", capsize=4, label="observed median ±1σ")

            # 条件分岐: `isinstance(thr, (int, float))` を満たす経路を評価する。

            if isinstance(thr, (int, float)):
                ax.axvline(float(thr), color="tab:red", linestyle="--", label="threshold")

            ax.legend(loc="lower right", fontsize=8)

            # (2) KS D_crit vs model sample size
            ax = axes[1]
            ax.set_title("M3 KS-test tolerance (proxy)")
            ax.set_xlabel("model samples (n_model)")
            ax.set_ylabel("D_crit (approx)")
            n_obs = m3.get("observed_samples_historical_n")
            alpha = float(m3.get("ks_reject_if_p_lt") or 0.01)
            # 条件分岐: `isinstance(n_obs, int) and n_obs > 0` を満たす経路を評価する。
            if isinstance(n_obs, int) and n_obs > 0:
                ms = [9, 18, 28]
                ds = [_ks_two_sample_dcrit(alpha, n_obs, m) for m in ms]
                ax.plot(ms, ds, marker="o")
                for m, d in zip(ms, ds):
                    # 条件分岐: `d is not None` を満たす経路を評価する。
                    if d is not None:
                        ax.text(m, d, f"{d:.3f}", ha="center", va="bottom", fontsize=8)

                ax.set_xticks(ms)

            # (3) Extended flux fraction mapping

            ax = axes[2]
            ax.set_title("Extended flux suppression mapping")
            ax.set_xlabel("suppression factor s = mi_obs/mi_true")
            ax.set_ylabel("f_ext / f_compact")
            ss = [0.95, 0.9, 0.8, 0.7, 0.6]
            fs = [(1.0 / s) - 1.0 for s in ss]
            ax.plot(ss, fs, marker="o")
            ax.invert_xaxis()

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
                "action": "eht_sgra_paper5_key_constraints_metrics",
                "outputs": [
                    str(out_json.relative_to(root)).replace("\\", "/"),
                    str(out_png.relative_to(root)).replace("\\", "/"),
                ],
                "metrics": {
                    "ok": bool(payload.get("ok")),
                    "nir_threshold_mJy": payload.get("extracted", {}).get("nir_2p2um", {}).get("threshold_mJy"),
                    "m3_ks_p": payload.get("extracted", {}).get("m3", {}).get("ks_reject_if_p_lt"),
                },
            }
        )
    except Exception:
        pass

    print(f"[ok] json: {out_json}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
