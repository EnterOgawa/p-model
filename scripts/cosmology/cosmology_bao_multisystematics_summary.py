#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_bao_multisystematics_summary.py

Phase 6 / Step 6.3（BAO多系統）:
BOSS / eBOSS など、複数サーベイ・複数tracerの BAO peakfit（ε）結果を横並びにし、
特に cap 依存（NGC/SGC）の張力を定量化して固定出力する。

入力（例）:
- output/cosmology/cosmology_bao_catalog_peakfit_<sample>_<caps>_metrics.json

出力（固定）:
- output/cosmology/cosmology_bao_multisystematics_summary.png
- output/cosmology/cosmology_bao_multisystematics_summary_metrics.json
- output/cosmology/cosmology_bao_multisystematics_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


def _set_japanese_font() -> None:
    try:
        import japanize_matplotlib  # type: ignore  # noqa: F401

        return
    except Exception:
        pass

    candidates = [
        "IPAexGothic",
        "IPAGothic",
        "Yu Gothic",
        "Meiryo",
        "Noto Sans CJK JP",
    ]
    installed = {f.name for f in mpl.font_manager.fontManager.ttflist}
    chosen = [name for name in candidates if name in installed]
    if chosen:
        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False


def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


@dataclass(frozen=True)
class PeakfitPoint:
    sample: str
    caps: str
    dist: str
    z_eff: float
    eps: float
    sigma_eps_1sigma: Optional[float]
    sigma_is_lower_bound: bool
    status: str
    source_metrics: str


def _load_peakfit_points(metrics_path: Path) -> List[PeakfitPoint]:
    d = json.loads(metrics_path.read_text(encoding="utf-8"))
    out: List[PeakfitPoint] = []
    for r in d.get("results", []) if isinstance(d, dict) else []:
        if not isinstance(r, dict):
            continue
        sample = str(r.get("sample") or "")
        caps = str(r.get("caps") or "")
        dist = str(r.get("dist") or "")
        z_eff = _safe_float(r.get("z_eff"))
        fit_free = (r.get("fit") or {}).get("free") if isinstance(r.get("fit"), dict) else None
        eps = _safe_float((fit_free or {}).get("eps") if isinstance(fit_free, dict) else None)
        screening = r.get("screening") if isinstance(r.get("screening"), dict) else {}
        sigma = _safe_float((screening or {}).get("sigma_eps_1sigma"))
        sigma_is_lower_bound = bool((screening or {}).get("abs_sigma_is_lower_bound", False))
        status = str((screening or {}).get("status") or "")

        if sigma is None:
            # Fallback: infer from CI width when available.
            ci = (fit_free or {}).get("eps_ci_1sigma") if isinstance(fit_free, dict) else None
            if isinstance(ci, list) and len(ci) == 2:
                lo = _safe_float(ci[0])
                hi = _safe_float(ci[1])
                if lo is not None and hi is not None:
                    sigma = 0.5 * abs(hi - lo)
                    scan = (screening or {}).get("scan") if isinstance((screening or {}).get("scan"), dict) else {}
                    if bool((scan or {}).get("ci_clipped", False)):
                        sigma_is_lower_bound = True

        if not sample or not caps or not dist or z_eff is None or eps is None:
            continue

        out.append(
            PeakfitPoint(
                sample=sample,
                caps=caps,
                dist=dist,
                z_eff=float(z_eff),
                eps=float(eps),
                sigma_eps_1sigma=sigma,
                sigma_is_lower_bound=bool(sigma_is_lower_bound),
                status=status,
                source_metrics=str(metrics_path.as_posix()),
            )
        )
    return out


def _dist_style(dist: str) -> Tuple[str, str]:
    dist = str(dist)
    if dist == "lcdm":
        return "#1f77b4", "o"
    if dist == "pbg":
        return "#ff7f0e", "s"
    return "#7f7f7f", "D"


def _sample_label(sample: str) -> str:
    # Keep names compact and stable.
    if sample == "lowz":
        return "BOSS LOWZ"
    if sample == "cmass":
        return "BOSS CMASS"
    if sample == "lrgpcmass_rec":
        return "eBOSS LRG"
    if sample == "qso":
        return "eBOSS QSO"
    return sample


def _caps_label(caps: str) -> str:
    if caps == "combined":
        return "combined"
    if caps == "north":
        return "NGC"
    if caps == "south":
        return "SGC"
    return caps


def _calc_delta_z(
    a: PeakfitPoint | None, b: PeakfitPoint | None
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (delta=a-b, sigma_delta, z=delta/sigma_delta)."""
    if a is None or b is None:
        return None, None, None
    if a.sigma_eps_1sigma is None or b.sigma_eps_1sigma is None:
        return None, None, None
    sigma_delta = math.sqrt(float(a.sigma_eps_1sigma) ** 2 + float(b.sigma_eps_1sigma) ** 2)
    if not (sigma_delta > 0):
        return None, None, None
    delta = float(a.eps) - float(b.eps)
    return delta, sigma_delta, float(delta / sigma_delta)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Cosmology: BAO multi-systematics summary (BOSS/eBOSS peakfit eps + cap tension).")
    ap.add_argument(
        "--samples",
        type=str,
        default="lowz,cmass,lrgpcmass_rec,qso",
        help="comma-separated sample ids (default: lowz,cmass,lrgpcmass_rec,qso)",
    )
    ap.add_argument(
        "--caps",
        type=str,
        default="combined,north,south",
        help="comma-separated caps (default: combined,north,south)",
    )
    ap.add_argument("--dists", type=str, default="lcdm,pbg", help="comma-separated dist models (default: lcdm,pbg)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    samples = [s.strip() for s in str(args.samples).split(",") if s.strip()]
    caps_list = [c.strip() for c in str(args.caps).split(",") if c.strip()]
    dists = [d.strip() for d in str(args.dists).split(",") if d.strip()]

    out_dir = _ROOT / "output" / "cosmology"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "cosmology_bao_multisystematics_summary.png"
    out_json = out_dir / "cosmology_bao_multisystematics_summary_metrics.json"
    out_csv = out_dir / "cosmology_bao_multisystematics_summary.csv"

    # Load points
    by_key: Dict[Tuple[str, str, str], PeakfitPoint] = {}
    missing: List[str] = []
    inputs: List[str] = []
    for sample in samples:
        for caps in caps_list:
            metrics_path = out_dir / f"cosmology_bao_catalog_peakfit_{sample}_{caps}_metrics.json"
            if not metrics_path.exists():
                missing.append(f"{sample}/{caps}")
                continue
            inputs.append(metrics_path.as_posix())
            for pt in _load_peakfit_points(metrics_path):
                if pt.dist not in dists:
                    continue
                by_key[(pt.sample, pt.caps, pt.dist)] = pt

    if not by_key:
        raise SystemExit(f"no inputs found (missing={missing})")

    def get(sample: str, caps: str, dist: str) -> PeakfitPoint | None:
        return by_key.get((sample, caps, dist))

    # Derived stats: cap tension (north - south)
    cap_tension: Dict[str, Dict[str, Any]] = {}
    for sample in samples:
        cap_tension[sample] = {}
        for dist in dists:
            pn = get(sample, "north", dist)
            ps = get(sample, "south", dist)
            delta, sigma_delta, z = _calc_delta_z(pn, ps)
            cap_tension[sample][dist] = {
                "delta_eps_north_minus_south": delta,
                "sigma_delta": sigma_delta,
                "z_delta": z,
                "north": None if pn is None else {"eps": pn.eps, "sigma": pn.sigma_eps_1sigma, "status": pn.status, "z_eff": pn.z_eff},
                "south": None if ps is None else {"eps": ps.eps, "sigma": ps.sigma_eps_1sigma, "status": ps.status, "z_eff": ps.z_eff},
            }

    # CSV rows
    csv_rows: List[Dict[str, Any]] = []
    for sample in samples:
        for dist in dists:
            pc = get(sample, "combined", dist)
            pn = get(sample, "north", dist)
            ps = get(sample, "south", dist)
            delta, sigma_delta, z = _calc_delta_z(pn, ps)
            csv_rows.append(
                {
                    "sample": sample,
                    "sample_label": _sample_label(sample),
                    "dist": dist,
                    "z_eff_combined": None if pc is None else pc.z_eff,
                    "eps_combined": None if pc is None else pc.eps,
                    "sigma_eps_1sigma_combined": None if pc is None else pc.sigma_eps_1sigma,
                    "status_combined": None if pc is None else pc.status,
                    "z_eff_north": None if pn is None else pn.z_eff,
                    "eps_north": None if pn is None else pn.eps,
                    "sigma_eps_1sigma_north": None if pn is None else pn.sigma_eps_1sigma,
                    "status_north": None if pn is None else pn.status,
                    "z_eff_south": None if ps is None else ps.z_eff,
                    "eps_south": None if ps is None else ps.eps,
                    "sigma_eps_1sigma_south": None if ps is None else ps.sigma_eps_1sigma,
                    "status_south": None if ps is None else ps.status,
                    "delta_eps_north_minus_south": delta,
                    "sigma_delta": sigma_delta,
                    "z_delta": z,
                }
            )

    # Plot
    _set_japanese_font()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.5, 5.6), dpi=140)

    # Panel A: eps(z) for combined points (BOSS/eBOSS overview)
    offsets = {"lcdm": -0.01, "pbg": +0.01}
    for sample in samples:
        for dist in dists:
            pt = get(sample, "combined", dist)
            if pt is None:
                continue
            color, marker = _dist_style(dist)
            x = float(pt.z_eff) + float(offsets.get(dist, 0.0))
            y = float(pt.eps)
            sig = pt.sigma_eps_1sigma
            if sig is not None and sig > 0:
                ax1.errorbar([x], [y], yerr=[sig], fmt=marker, color=color, ecolor=color, capsize=3, markersize=7)
            else:
                ax1.plot([x], [y], marker=marker, color=color, markersize=7, linestyle="none")
            ax1.annotate(
                _sample_label(sample),
                (x, y),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=9,
                color="#222222",
            )

    ax1.axhline(0.0, color="#777777", linewidth=1, alpha=0.6)
    ax1.set_xlabel("z_eff（combined）")
    ax1.set_ylabel("ε（peakfit; screening）")
    ax1.set_title("ε vs z_eff（BOSS/eBOSS; combined）")
    ax1.grid(True, alpha=0.25)

    # Legend: dist only
    handles = []
    labels = []
    for dist in dists:
        color, marker = _dist_style(dist)
        handles.append(
            mpl.lines.Line2D([0], [0], marker=marker, color=color, linestyle="none", markersize=7)
        )
        labels.append(dist)
    ax1.legend(handles, labels, loc="upper right", frameon=True, title="dist")

    # Panel B: cap tension z-score (north - south)
    x = np.arange(len(samples), dtype=float)
    width = 0.35 if len(dists) <= 2 else 0.8 / max(1, len(dists))
    for j, dist in enumerate(dists):
        zvals: List[float] = []
        for sample in samples:
            zd = cap_tension.get(sample, {}).get(dist, {}).get("z_delta")
            zvals.append(float(zd) if isinstance(zd, (int, float)) and math.isfinite(float(zd)) else float("nan"))
        color, _marker = _dist_style(dist)
        ax2.bar(x + (j - (len(dists) - 1) / 2.0) * width, zvals, width=width, color=color, alpha=0.85, label=dist)

    ax2.axhline(0.0, color="#777777", linewidth=1, alpha=0.6)
    for thr, ls, a in [(3.0, "--", 0.6), (5.0, ":", 0.5)]:
        ax2.axhline(+thr, color="#444444", linestyle=ls, linewidth=1, alpha=a)
        ax2.axhline(-thr, color="#444444", linestyle=ls, linewidth=1, alpha=a)
    ax2.set_xticks(x, [_sample_label(s) for s in samples], rotation=20, ha="right")
    ax2.set_ylabel("z = (ε_NGC − ε_SGC) / √(σ_N^2+σ_S^2)")
    ax2.set_title("cap依存（NGC/SGC）: 張力の定量化")
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.legend(loc="upper right", frameon=True, title="dist")

    fig.suptitle("BAO multi-systematics summary (BOSS/eBOSS)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    # Write CSV
    csv_fields = list(csv_rows[0].keys())
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields)
        w.writeheader()
        for row in csv_rows:
            w.writerow(row)

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "domain": "cosmology",
        "step": "6.3 (BAO multisystematics summary)",
        "inputs": {"metrics_json": inputs, "missing": missing},
        "filters": {"samples": samples, "caps": caps_list, "dists": dists},
        "derived": {"cap_tension": cap_tension},
        "outputs": {
            "png": str(out_png),
            "metrics_json": str(out_json),
            "csv": str(out_csv),
        },
        "notes": [
            "cap張力は north/south が両方存在し、σが有限な場合のみ計算。",
            "σが無い場合は eps_ci_1sigma から近似（ただし ci_clipped のとき lower-bound 扱い）。",
        ],
        "repro": f"python -B {Path(__file__).as_posix()}",
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] png : {out_png}")
    print(f"[ok] csv : {out_csv}")
    print(f"[ok] json: {out_json}")

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_bao_multisystematics_summary",
                "argv": sys.argv,
                "outputs": {"png": out_png, "csv": out_csv, "metrics_json": out_json},
                "metrics": payload.get("filters", {}),
            }
        )
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
