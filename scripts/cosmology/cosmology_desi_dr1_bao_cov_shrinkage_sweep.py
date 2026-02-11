#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_desi_dr1_bao_cov_shrinkage_sweep.py

Phase 4 / Step 4.5（DESI DR1）:
sky-jackknife covariance の off-diagonal の扱い（diag shrinkage λ）に対して、
DESI DR1（任意tracer）の ε_fit−ε_expected（Y1data）z_score がどれだけ敏感かを可視化する。

背景：
- jackknife cov をそのまま pinv すると、ε が不安定化して「差」が作れてしまう可能性がある。
- そこで `cosmology_bao_catalog_peakfit.py --cov-shrinkage λ` を sweep し、結果の頑健性を確認する。

入出力：
- 入力：
  - output/cosmology/cosmology_bao_xi_from_catalogs_*__<out_tag>*.npz
  - 対応する __jk_cov.npz
- 出力：
  - output/cosmology/cosmology_desi_dr1_bao_cov_shrinkage_sweep__<out_tag>.png
  - output/cosmology/cosmology_desi_dr1_bao_cov_shrinkage_sweep__<out_tag>.json
  - output/cosmology/cosmology_desi_dr1_bao_cov_shrinkage_sweep__<out_tag>.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.cosmology import cosmology_bao_catalog_peakfit as _peakfit  # noqa: E402
from scripts.summary import worklog  # noqa: E402


@dataclass(frozen=True)
class _Row:
    out_tag: str
    lam: float
    tracer: str
    dist: str
    eps_fit: float
    eps_fit_sigma: float
    eps_expected: float
    eps_expected_sigma: float
    z_score_combined: float
    z_score_vs_y1data: float


def _parse_lams(spec: str) -> List[float]:
    s = str(spec).strip()
    if not s:
        raise ValueError("empty --lams")
    out: List[float] = []
    for part in s.split(","):
        p = part.strip()
        if not p:
            continue
        lam = float(p)
        if not (math.isfinite(lam) and 0.0 <= lam <= 1.0):
            raise ValueError(f"invalid lambda: {p} (must be within [0,1])")
        out.append(lam)
    if not out:
        raise ValueError("no lambdas parsed")
    # unique, keep order
    seen: set[float] = set()
    uniq: List[float] = []
    for x in out:
        if x in seen:
            continue
        seen.add(x)
        uniq.append(x)
    return uniq


def _lam_tag(lam: float) -> str:
    # 0.2 -> 0p2, 1.0 -> 1p0
    s = f"{float(lam):.4f}".rstrip("0").rstrip(".")
    if s == "":
        s = "0"
    return s.replace("-", "m").replace(".", "p")


def _peakfit_metrics_path(*, sample: str, caps: str, out_tag: str, output_suffix: str) -> Path:
    tag = f"{sample}_{caps}__{out_tag}__{output_suffix}"
    return _ROOT / "output" / "cosmology" / f"cosmology_bao_catalog_peakfit_{tag}_metrics.json"


def _cross_metrics_path(*, out_tag: str) -> Path:
    return _ROOT / "output" / "cosmology" / f"cosmology_desi_dr1_bao_y1data_eps_crosscheck__{out_tag}_metrics.json"


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _collect_rows(cross_metrics: Dict[str, Any], *, out_tag: str, lam: float) -> List[_Row]:
    res = cross_metrics.get("results") if isinstance(cross_metrics.get("results"), dict) else {}
    tracers = res.get("tracers") if isinstance(res.get("tracers"), list) else []
    expected = res.get("expected_from_y1data") if isinstance(res.get("expected_from_y1data"), dict) else {}
    fit = res.get("fit_from_catalog_peakfit") if isinstance(res.get("fit_from_catalog_peakfit"), dict) else {}
    deltas = res.get("delta_fit_minus_expected") if isinstance(res.get("delta_fit_minus_expected"), dict) else {}
    out: List[_Row] = []
    for t in tracers:
        if t not in expected or t not in fit or t not in deltas:
            continue
        for dist in ("lcdm", "pbg"):
            e = expected.get(t, {}).get(dist, {})
            f = fit.get(t, {}).get(dist, {})
            d = deltas.get(t, {}).get(dist, {})
            out.append(
                _Row(
                    out_tag=str(out_tag),
                    lam=float(lam),
                    tracer=str(t),
                    dist=str(dist),
                    eps_fit=_safe_float(f.get("eps")),
                    eps_fit_sigma=_safe_float(f.get("eps_sigma")),
                    eps_expected=_safe_float(e.get("eps")),
                    eps_expected_sigma=_safe_float(e.get("eps_sigma")),
                    z_score_combined=_safe_float(d.get("z_score_combined")),
                    z_score_vs_y1data=_safe_float(d.get("z_score_vs_y1data")),
                )
            )
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="DESI DR1: covariance shrinkage sweep (epsilon cross-check).")
    ap.add_argument(
        "--out-tag",
        type=str,
        required=True,
        help="xi_from_catalogs out_tag to filter (e.g., w_desi_default_ms_off_y1bins_reservoir_r0to17_mix)",
    )
    ap.add_argument("--sample", type=str, default="lrg", help="catalog sample id for peakfit (default: lrg)")
    ap.add_argument("--caps", type=str, default="combined", help="caps (default: combined)")
    ap.add_argument(
        "--tracers",
        type=str,
        default="LRG1,LRG2",
        help=(
            "Comma-separated tracer list for the Y1data epsilon cross-check (default: LRG1,LRG2). "
            "Example: --tracers ELG2"
        ),
    )
    ap.add_argument("--smooth-power-max", type=int, default=2, help="peakfit smooth_power_max (default: 2)")
    ap.add_argument(
        "--cov-source",
        choices=["jackknife", "rascalc", "vac"],
        default="jackknife",
        help="Covariance source for peakfit (default: jackknife).",
    )
    ap.add_argument(
        "--cov-suffix",
        type=str,
        default="",
        help=(
            "Optional per-case jackknife cov suffix for peakfit. "
            "Example: --cov-suffix jk_cov_ra_dec_per_cap_unwrapped_n48_dec4_ra6"
        ),
    )
    ap.add_argument(
        "--cov-zero-xi02",
        action="store_true",
        help="Pass through to peakfit: zero xi0-xi2 cross-cov blocks before inversion (default: off).",
    )
    ap.add_argument(
        "--cov-bandwidth-bins",
        type=int,
        default=-1,
        help="Pass through to peakfit: band dv covariance by |Δs_bin| in fit space (-1 disables; default: -1).",
    )
    ap.add_argument(
        "--cov-bandwidth-xi02-bins",
        type=int,
        default=-1,
        help=(
            "Pass through to peakfit: optional override bandwidth for xi0-xi2 cross blocks (-1 => same as --cov-bandwidth-bins)."
        ),
    )
    ap.add_argument("--lams", type=str, default="0,0.05,0.1,0.2,0.4,0.6,0.8,1.0", help="comma-separated λ within [0,1]")
    ap.add_argument("--skip-existing", action="store_true", help="skip if per-lambda outputs already exist")
    args = ap.parse_args(list(argv) if argv is not None else None)

    out_tag = str(args.out_tag).strip()
    if not out_tag:
        raise SystemExit("--out-tag must be non-empty")
    sample = str(args.sample).strip()
    caps = str(args.caps).strip()
    tracers = str(args.tracers).strip()
    if not sample:
        raise SystemExit("--sample must be non-empty")
    if not caps:
        raise SystemExit("--caps must be non-empty")
    if not tracers:
        raise SystemExit("--tracers must be non-empty")
    smooth_power_max = int(args.smooth_power_max)
    if smooth_power_max < 0:
        raise SystemExit("--smooth-power-max must be >=0")
    lams = _parse_lams(str(args.lams))

    cov_source = str(args.cov_source).strip().lower()
    if cov_source not in ("jackknife", "rascalc", "vac"):
        raise SystemExit("--cov-source must be jackknife, rascalc, or vac")
    cov_suffix = str(args.cov_suffix).strip()
    cov_bandwidth_bins = int(args.cov_bandwidth_bins)
    cov_bandwidth_xi02_bins = int(args.cov_bandwidth_xi02_bins)
    if cov_bandwidth_bins < -1:
        raise SystemExit("--cov-bandwidth-bins must be >= -1")
    if cov_bandwidth_xi02_bins < -1:
        raise SystemExit("--cov-bandwidth-xi02-bins must be >= -1")

    default_cov_tag = "jk_cov" if cov_source == "jackknife" else ("rascalc_cov" if cov_source == "rascalc" else "vac_cov")
    reg_tag = cov_suffix if cov_suffix else default_cov_tag
    if bool(args.cov_zero_xi02):
        reg_tag = f"{reg_tag}_xi02zero"
    if cov_bandwidth_bins >= 0:
        reg_tag = f"{reg_tag}_band{int(cov_bandwidth_bins)}"
    if cov_bandwidth_xi02_bins >= 0:
        reg_tag = f"{reg_tag}_x{int(cov_bandwidth_xi02_bins)}"
    if int(smooth_power_max) != 2:
        reg_tag = f"{reg_tag}_smooth{int(smooth_power_max)}"
    is_default_reg = (
        (cov_source == "jackknife")
        and (not cov_suffix)
        and (not bool(args.cov_zero_xi02))
        and (cov_bandwidth_bins < 0)
        and (cov_bandwidth_xi02_bins < 0)
        and (int(smooth_power_max) == 2)
    )
    out_tag_for_outputs = out_tag if is_default_reg else f"{out_tag}__{reg_tag}"

    out_dir = _ROOT / "output" / "cosmology"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"cosmology_desi_dr1_bao_cov_shrinkage_sweep__{out_tag_for_outputs}.png"
    out_json = out_dir / f"cosmology_desi_dr1_bao_cov_shrinkage_sweep__{out_tag_for_outputs}.json"
    out_csv = out_dir / f"cosmology_desi_dr1_bao_cov_shrinkage_sweep__{out_tag_for_outputs}.csv"

    rows: List[_Row] = []
    per_lambda: List[Dict[str, Any]] = []
    cross_script = _ROOT / "scripts" / "cosmology" / "cosmology_desi_dr1_bao_y1data_eps_crosscheck.py"
    for lam in lams:
        lam_tag = _lam_tag(lam)
        suffix = f"{reg_tag}_shrink{lam_tag}_sweep"
        peakfit_metrics = _peakfit_metrics_path(sample=sample, caps=caps, out_tag=out_tag, output_suffix=suffix)
        cross_out_tag = f"{out_tag}__{suffix}"
        cross_metrics_path = _cross_metrics_path(out_tag=cross_out_tag)

        if bool(args.skip_existing) and peakfit_metrics.exists() and cross_metrics_path.exists():
            pass
        else:
            peakfit_argv = [
                    "--sample",
                    sample,
                    "--caps",
                    caps,
                    "--out-tag",
                    out_tag,
                    "--cov-source",
                    str(cov_source),
                    "--cov-shrinkage",
                    f"{float(lam):.6g}",
                    "--cov-bandwidth-bins",
                    str(int(cov_bandwidth_bins)),
                    "--cov-bandwidth-xi02-bins",
                    str(int(cov_bandwidth_xi02_bins)),
                    "--smooth-power-max",
                    str(int(smooth_power_max)),
                    "--output-suffix",
                    suffix,
                ]
            if cov_suffix:
                peakfit_argv.extend(["--cov-suffix", cov_suffix])
            if bool(args.cov_zero_xi02):
                peakfit_argv.append("--cov-zero-xi02")

            ret = _peakfit.main(peakfit_argv)
            if int(ret) != 0:
                raise SystemExit(f"peakfit failed (lambda={lam})")

            cmd = [
                sys.executable,
                "-B",
                str(cross_script),
                "--peakfit-metrics-json",
                str(peakfit_metrics),
                "--out-tag",
                cross_out_tag,
                "--tracers",
                tracers,
            ]
            proc = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True)
            if proc.returncode != 0:
                sys.stderr.write(proc.stdout)
                sys.stderr.write(proc.stderr)
                raise SystemExit(f"crosscheck failed (lambda={lam})")

        cm = json.loads(cross_metrics_path.read_text(encoding="utf-8"))
        rows_l = _collect_rows(cm, out_tag=out_tag, lam=lam)
        rows.extend(rows_l)
        per_lambda.append(
            {
                "lambda": float(lam),
                "reg_tag": str(reg_tag),
                "peakfit_metrics_json": str(peakfit_metrics),
                "crosscheck_metrics_json": str(cross_metrics_path),
            }
        )

    # Write CSV.
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "out_tag",
                "lambda",
                "tracer",
                "dist",
                "eps_fit",
                "eps_fit_sigma",
                "eps_expected",
                "eps_expected_sigma",
                "z_score_combined",
                "z_score_vs_y1data",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.out_tag,
                    f"{r.lam:.6g}",
                    r.tracer,
                    r.dist,
                    f"{r.eps_fit:.8g}",
                    f"{r.eps_fit_sigma:.8g}",
                    f"{r.eps_expected:.8g}",
                    f"{r.eps_expected_sigma:.8g}",
                    f"{r.z_score_combined:.8g}",
                    f"{r.z_score_vs_y1data:.8g}",
                ]
            )

    # Plot.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Organize data: tracer -> dist -> (x,y)
        by_t: Dict[str, Dict[str, Dict[float, float]]] = {}
        for r in rows:
            by_t.setdefault(r.tracer, {}).setdefault(r.dist, {})[float(r.lam)] = float(r.z_score_combined)

        tracers = sorted(by_t.keys())
        fig, axes = plt.subplots(
            nrows=1,
            ncols=max(1, len(tracers)),
            figsize=(7.4 * max(1, len(tracers)), 4.8),
            sharey=True,
            dpi=180,
        )
        if not isinstance(axes, (list, tuple)):
            axes = [axes]

        colors = {"lcdm": "#1f77b4", "pbg": "#ff7f0e"}
        for ax, t in zip(axes, tracers, strict=True):
            ax.axhline(0.0, color="0.6", lw=1.0)
            ax.axhline(+2.0, color="0.8", lw=1.0, ls="--")
            ax.axhline(-2.0, color="0.8", lw=1.0, ls="--")
            for dist in ("lcdm", "pbg"):
                m = by_t.get(t, {}).get(dist, {})
                xs = sorted(m.keys())
                ys = [m[x] for x in xs]
                ax.plot(xs, ys, marker="o", ms=5, lw=2.0, label=dist, color=colors.get(dist, None))
            ax.set_title(f"{t}: z_score_combined vs shrinkage λ", fontsize=12)
            ax.set_xlabel("λ (shrink -> diag)")
            ax.grid(True, alpha=0.25)

        # Common legend outside (avoid covering data).
        try:
            handles, labels = axes[0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0, fontsize=10)
        except Exception:
            pass

        axes[0].set_ylabel("z_score_combined (ε_fit−ε_expected)", fontsize=12)
        fig.suptitle(f"DESI DR1 BAO: jackknife cov shrinkage sensitivity ({out_tag})", fontsize=13, y=1.03)
        fig.tight_layout()
        fig.savefig(out_png, dpi=220, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        # Keep CSV/JSON even if plotting fails (headless env etc.)
        pass

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "domain": "cosmology",
        "step": "4.5 (DESI DR1 covariance sensitivity: shrinkage sweep)",
        "inputs": {
            "out_tag": out_tag,
            "lams": [float(x) for x in lams],
            "cov_source": str(cov_source),
            "cov_suffix": str(cov_suffix),
            "cov_zero_xi02": bool(args.cov_zero_xi02),
            "cov_bandwidth_bins": int(cov_bandwidth_bins),
            "cov_bandwidth_xi02_bins": int(cov_bandwidth_xi02_bins),
            "reg_tag": str(reg_tag),
            "out_tag_for_outputs": str(out_tag_for_outputs),
        },
        "outputs": {"png": str(out_png), "json": str(out_json), "csv": str(out_csv)},
        "per_lambda": per_lambda,
        "rows": [r.__dict__ for r in rows],
        "notes": [
            "z_score_combined は Y1data ε_expected の誤差と peakfit の ε_sigma（CI由来）を二乗和で合成したもの。",
            "λ=0 は jackknife cov の full off-diagonal をそのまま使用、λ=1 は diag-only。",
            "cov_bandwidth_bins>=0 は dv=[xi0,xi2] covariance を |Δs_bin| で banding してから inversion する（peakfit側実装）。",
        ],
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    worklog.append_event(
        {
            "domain": "cosmology",
            "action": "desi_dr1_bao_cov_shrinkage_sweep",
            "inputs": [out_tag],
            "outputs": [out_png, out_json, out_csv],
            "params": payload["inputs"],
        }
    )

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")
    print(f"[ok] csv : {out_csv}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
