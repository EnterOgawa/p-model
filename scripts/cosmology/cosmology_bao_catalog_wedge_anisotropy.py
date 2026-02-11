#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_bao_catalog_wedge_anisotropy.py

Step 16.4（BAO一次情報：銀河+random） / Phase A（スクリーニング）:
`cosmology_bao_xi_from_catalogs.py` の出力（*_metrics.json / *.npz）から、
μ-wedge（横方向/縦方向）の BAOピーク位置を見積もり、異方（幾何）の差を
定量化できる形にまとめる。

狙い：
- r_d をフリーにしても「横/縦でのピーク差（AP/warping）」は残るため、
  ξ2を重視した幾何整合チェックの入口として使う。

出力（固定）:
- output/cosmology/cosmology_bao_catalog_wedge_anisotropy.png
- output/cosmology/cosmology_bao_catalog_wedge_anisotropy_metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402
from scripts.cosmology.cosmology_bao_xi_from_catalogs import (  # noqa: E402
    _estimate_bao_peak_s2_xi,
    _xi_wedge_from_multipoles,
)

_WIN_ABS_RE = re.compile(r"^[a-zA-Z]:[\\/]")
_WSL_ABS_RE = re.compile(r"^/mnt/([a-zA-Z])/(.+)$")


def _resolve_path_like(p: Any) -> Optional[Path]:
    if p is None:
        return None
    s = str(p).strip()
    if not s:
        return None
    if os.name == "nt":
        m = _WSL_ABS_RE.match(s)
        if m:
            drive = m.group(1).upper()
            rest = m.group(2).replace("/", "\\")
            return Path(f"{drive}:\\{rest}")
    else:
        if _WIN_ABS_RE.match(s):
            drive = s[0].lower()
            rest = s[2:].lstrip("\\/").replace("\\", "/")
            return Path(f"/mnt/{drive}/{rest}")
    path = Path(s)
    if path.is_absolute():
        return path
    return _ROOT / path


@dataclass(frozen=True)
class WedgePoint:
    sample: str
    caps: str
    dist: str
    out_tag: Optional[str]
    z_eff: float
    z_bin: str
    mu_split: float
    s_perp: float
    s_par: float
    delta_s: float
    ratio: float
    eps_proxy: float
    abs_eps_proxy: float
    reliable: bool
    flags: List[str]
    source_json: str


def _eps_proxy_from_ratio(ratio: float) -> float:
    r = float(ratio)
    if not (math.isfinite(r) and r > 0.0):
        return float("nan")
    # Parameterization used elsewhere in this repo:
    #   1+ε = (α∥/α⊥)^(1/3)
    # Here ratio is stored as s∥/s⊥ (fid-coordinate wedge peak positions). Under AP,
    #   s∥/s⊥ ≈ (1/α∥)/(1/α⊥) = α⊥/α∥ = 1/(α∥/α⊥)
    # so we invert it to match the ε sign convention used in peakfit.
    return float(r ** (-1.0 / 3.0) - 1.0)


def _status_from_abs_eps(abs_eps: float, *, ok_max: float = 0.015, mixed_max: float = 0.03) -> str:
    if not math.isfinite(float(abs_eps)):
        return "info"
    x = float(abs_eps)
    if x <= float(ok_max):
        return "ok"
    if x <= float(mixed_max):
        return "mixed"
    return "ng"


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _load_from_metrics(d: Dict[str, Any], *, source_json: str) -> Optional[WedgePoint]:
    params = d.get("params", {}) if isinstance(d.get("params", {}), dict) else {}
    derived = d.get("derived", {}) if isinstance(d.get("derived", {}), dict) else {}
    wedges = derived.get("bao_wedges", {}) if isinstance(derived.get("bao_wedges", {}), dict) else {}
    if not wedges.get("enabled", False):
        return None

    sample = str(params.get("sample"))
    caps = str(params.get("caps"))
    dist = str(params.get("distance_model"))
    out_tag_raw = params.get("out_tag")
    out_tag = None if (out_tag_raw is None) else str(out_tag_raw)
    z_cut = params.get("z_cut", {}) if isinstance(params.get("z_cut", {}), dict) else {}
    z_bin = str(z_cut.get("bin", "none"))
    z_eff = float(derived.get("z_eff_gal_weighted"))
    mu_split = float(wedges.get("mu_split"))

    s_perp = float(wedges.get("transverse", {}).get("peak", {}).get("s_peak"))
    s_par = float(wedges.get("radial", {}).get("peak", {}).get("s_peak"))
    delta_s = float(wedges.get("delta_s_peak"))
    ratio = float(wedges.get("ratio_s_peak"))
    eps_proxy = _eps_proxy_from_ratio(ratio)
    abs_eps_proxy = float(abs(eps_proxy)) if math.isfinite(eps_proxy) else float("nan")
    reliable = bool(wedges.get("reliable", True))
    flags_raw = wedges.get("flags", [])
    flags = [str(x) for x in flags_raw] if isinstance(flags_raw, list) else []

    return WedgePoint(
        sample=sample,
        caps=caps,
        dist=dist,
        out_tag=out_tag,
        z_eff=z_eff,
        z_bin=z_bin,
        mu_split=mu_split,
        s_perp=s_perp,
        s_par=s_par,
        delta_s=delta_s,
        ratio=ratio,
        eps_proxy=float(eps_proxy),
        abs_eps_proxy=float(abs_eps_proxy),
        reliable=reliable,
        flags=flags,
        source_json=source_json,
    )


def _infer_npz_path(metrics_path: Path, d: Dict[str, Any]) -> Optional[Path]:
    outputs = d.get("outputs", {}) if isinstance(d.get("outputs", {}), dict) else {}
    if isinstance(outputs, dict) and outputs.get("npz"):
        return _resolve_path_like(outputs.get("npz"))
    if str(metrics_path).endswith("_metrics.json"):
        return Path(str(metrics_path)[: -len("_metrics.json")] + ".npz")
    return None


def _npz_scalar(arr: Any, key: str) -> Optional[float]:
    try:
        if key not in arr:
            return None
        v = arr[key]
        if isinstance(v, np.ndarray):
            v = v.reshape(-1)[0]
        return float(v)
    except Exception:
        return None


def _load_from_npz(metrics_path: Path, d: Dict[str, Any], *, mu_split_default: float) -> Optional[WedgePoint]:
    params = d.get("params", {}) if isinstance(d.get("params", {}), dict) else {}
    derived = d.get("derived", {}) if isinstance(d.get("derived", {}), dict) else {}

    sample = str(params.get("sample"))
    caps = str(params.get("caps"))
    dist = str(params.get("distance_model"))
    out_tag_raw = params.get("out_tag")
    out_tag = None if (out_tag_raw is None) else str(out_tag_raw)
    z_cut = params.get("z_cut", {}) if isinstance(params.get("z_cut", {}), dict) else {}
    z_bin = str(z_cut.get("bin", "none"))
    z_eff = _safe_float(derived.get("z_eff_gal_weighted"))
    if z_eff is None:
        return None

    mu_bins = params.get("mu_bins", {}) if isinstance(params.get("mu_bins", {}), dict) else {}
    mu_max = _safe_float(mu_bins.get("mu_max", 1.0))
    if mu_max is None:
        mu_max = 1.0
    if abs(float(mu_max) - 1.0) > 1e-6:
        return None

    npz_path = _infer_npz_path(metrics_path, d)
    if not npz_path or not npz_path.exists():
        return None

    try:
        with np.load(str(npz_path)) as arr:
            s = np.asarray(arr["s"], dtype=float).reshape(-1)
            xi0 = np.asarray(arr["xi0"], dtype=float)
            xi2 = np.asarray(arr["xi2"], dtype=float)

            if xi0.ndim == 2 and xi0.shape[0] == 1:
                xi0 = xi0.reshape(-1)
            if xi2.ndim == 2 and xi2.shape[0] == 1:
                xi2 = xi2.reshape(-1)
            if xi0.ndim != 1 or xi2.ndim != 1:
                return None
            if (xi0.size != s.size) or (xi2.size != s.size):
                return None

            mu_split = _safe_float(mu_bins.get("mu_split"))
            if mu_split is None:
                mu_split = _npz_scalar(arr, "mu_split")
            if mu_split is None:
                mu_split = float(mu_split_default)
            if not (0.0 < float(mu_split) < 1.0):
                return None

            # Prefer stored wedges if present.
            if ("xi_wedge_transverse" in arr) and ("xi_wedge_radial" in arr):
                xi_perp = np.asarray(arr["xi_wedge_transverse"], dtype=float)
                xi_par = np.asarray(arr["xi_wedge_radial"], dtype=float)
                if xi_perp.ndim == 2 and xi_perp.shape[0] == 1:
                    xi_perp = xi_perp.reshape(-1)
                if xi_par.ndim == 2 and xi_par.shape[0] == 1:
                    xi_par = xi_par.reshape(-1)
                if xi_perp.ndim != 1 or xi_par.ndim != 1:
                    return None
                if (xi_perp.size != s.size) or (xi_par.size != s.size):
                    return None
            else:
                xi_perp, _ = _xi_wedge_from_multipoles(xi0=xi0, xi2=xi2, mu0=0.0, mu1=float(mu_split))
                xi_par, _ = _xi_wedge_from_multipoles(xi0=xi0, xi2=xi2, mu0=float(mu_split), mu1=1.0)

            try:
                peak_perp = _estimate_bao_peak_s2_xi(s=s, xi=xi_perp)
                peak_par = _estimate_bao_peak_s2_xi(s=s, xi=xi_par)
            except Exception:
                return None
            s_perp = float(peak_perp.get("s_peak"))
            s_par = float(peak_par.get("s_peak"))
    except Exception:
        return None

    delta_s = s_par - s_perp
    ratio = (s_par / s_perp) if (s_perp != 0.0) else float("nan")
    eps_proxy = _eps_proxy_from_ratio(float(ratio))
    abs_eps_proxy = float(abs(eps_proxy)) if math.isfinite(eps_proxy) else float("nan")

    return WedgePoint(
        sample=sample,
        caps=caps,
        dist=dist,
        out_tag=out_tag,
        z_eff=float(z_eff),
        z_bin=z_bin,
        mu_split=float(mu_split),
        s_perp=s_perp,
        s_par=s_par,
        delta_s=float(delta_s),
        ratio=float(ratio),
        eps_proxy=float(eps_proxy),
        abs_eps_proxy=float(abs_eps_proxy),
        reliable=True,
        flags=[],
        source_json=str(metrics_path),
    )


def _load_points(paths: Iterable[Path], *, mu_split_default: float) -> List[WedgePoint]:
    out: List[WedgePoint] = []
    for p in paths:
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        pt = _load_from_metrics(d, source_json=str(p))
        if pt is None:
            pt = _load_from_npz(p, d, mu_split_default=mu_split_default)
        if pt is not None:
            out.append(pt)
    return out


def _group(points: List[WedgePoint]) -> Dict[Tuple[str, str, str], List[WedgePoint]]:
    g: Dict[Tuple[str, str, str], List[WedgePoint]] = {}
    for p in points:
        g.setdefault((p.sample, p.caps, p.dist), []).append(p)
    for k in list(g):
        g[k] = sorted(g[k], key=lambda x: x.z_eff)
    return g


def _drift(values: List[float]) -> Dict[str, float]:
    a = np.asarray(list(values), dtype=np.float64)
    if a.size == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"), "span": float("nan")}
    return {
        "n": int(a.size),
        "mean": float(np.mean(a)),
        "std": float(np.std(a, ddof=0)),
        "span": float(np.max(a) - np.min(a)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Summarize BAO wedge anisotropy from catalog-based xi outputs.")
    ap.add_argument("--glob", default="output/cosmology/cosmology_bao_xi_from_catalogs_*_metrics.json", help="metrics glob")
    ap.add_argument("--sample", default="", help="filter by sample (e.g., cmasslowztot); empty => all")
    ap.add_argument("--caps", default="", help="filter by caps (north/south/combined); empty => all")
    ap.add_argument(
        "--out-tag",
        default="none",
        help="Filter out_tag in inputs: none (default; only out_tag=null), any, or exact string",
    )
    ap.add_argument("--mu-split", type=float, default=0.5, help="fallback mu_split when not present in metrics/npz")
    ap.add_argument(
        "--require-zbin",
        action="store_true",
        help="only include metrics with z_bin != none (default: include all)",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    paths = sorted(_ROOT.glob(str(args.glob)))
    pts = _load_points(paths, mu_split_default=float(args.mu_split))

    sample_f = str(args.sample).strip().lower()
    if sample_f:
        pts = [p for p in pts if p.sample.lower() == sample_f]
    caps_f = str(args.caps).strip().lower()
    if caps_f:
        pts = [p for p in pts if p.caps.lower() == caps_f]
    out_tag_f = str(args.out_tag).strip()
    if out_tag_f == "none":
        pts = [p for p in pts if p.out_tag is None]
    elif out_tag_f == "any":
        pts = [p for p in pts if p.out_tag is not None]
    elif out_tag_f:
        pts = [p for p in pts if (p.out_tag == out_tag_f)]
    if bool(args.require_zbin):
        pts = [p for p in pts if p.z_bin != "none"]

    groups = _group(pts)

    out_dir = _ROOT / "output" / "cosmology"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Keep the default output names stable for run_all (which may pass sample/caps filters).
    # Only add a suffix when out_tag is explicitly non-default to avoid clobbering baseline outputs.
    suffix = ""
    if out_tag_f == "any":
        suffix = "__outtag_any"
    elif out_tag_f not in ("none", "any") and out_tag_f:
        suffix = f"__{out_tag_f}"

    out_png = out_dir / f"cosmology_bao_catalog_wedge_anisotropy{suffix}.png"
    out_json = out_dir / f"cosmology_bao_catalog_wedge_anisotropy{suffix}_metrics.json"

    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.0, 6.8))
    colors = {"lcdm": "#1f77b4", "pbg": "#ff7f0e"}
    markers = {"lcdm": "o", "pbg": "s"}

    for (sample, caps, dist), xs in sorted(groups.items()):
        label = f"{sample}/{caps}/{dist}"

        color = colors.get(dist, "#333333")
        base_marker = markers.get(dist, "o")

        rel = [p for p in xs if bool(getattr(p, "reliable", True))]
        unrel = [p for p in xs if not bool(getattr(p, "reliable", True))]

        if rel:
            z_rel = [p.z_eff for p in rel]
            delta_rel = [p.delta_s for p in rel]
            ratio_rel = [p.ratio for p in rel]
            ax1.plot(z_rel, delta_rel, marker=base_marker, color=color, linewidth=1.8, markersize=6, label=label)
            ax2.plot(z_rel, ratio_rel, marker=base_marker, color=color, linewidth=1.8, markersize=6, label=label)

        if unrel:
            z_u = [p.z_eff for p in unrel]
            delta_u = [p.delta_s for p in unrel]
            ratio_u = [p.ratio for p in unrel]
            if rel:
                ax1.scatter(z_u, delta_u, marker="x", color=color, s=45, alpha=0.6)
                ax2.scatter(z_u, ratio_u, marker="x", color=color, s=45, alpha=0.6)
            else:
                ax1.plot(z_u, delta_u, linestyle="None", marker="x", color=color, markersize=7, label=f"{label} (unreliable)")
                ax2.plot(z_u, ratio_u, linestyle="None", marker="x", color=color, markersize=7, label=f"{label} (unreliable)")

    ax1.axhline(0.0, color="#888888", linewidth=1.0, linestyle="--", alpha=0.6)
    ax1.set_xlabel("z_eff (galaxy-weighted)")
    ax1.set_ylabel("Δs_peak = s∥ - s⊥  [Mpc/h]")
    ax1.grid(True, linestyle="--", alpha=0.35)
    ax1.legend(fontsize=9, loc="best")

    ax2.axhline(1.0, color="#888888", linewidth=1.0, linestyle="--", alpha=0.6)
    ax2.set_xlabel("z_eff (galaxy-weighted)")
    ax2.set_ylabel("ratio = s∥ / s⊥")
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(fontsize=9, loc="best")

    title_suffix = ""
    if sample_f:
        title_suffix += f" sample={sample_f}"
    if caps_f:
        title_suffix += f" caps={caps_f}"
    if out_tag_f != "none":
        title_suffix += f" out_tag={out_tag_f or 'none'}"
    if args.require_zbin:
        title_suffix += " zbin_only"
    fig.suptitle(f"BAO wedge anisotropy (catalog-based ξℓ){title_suffix}", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.02, 1.0, 0.94))
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    payload: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "domain": "cosmology",
        "step": "16.4 (BAO catalog-based wedge anisotropy summary)",
        "inputs": {"glob": str(args.glob), "n_files_scanned": int(len(paths))},
        "filters": {
            "sample": sample_f or None,
            "caps": caps_f or None,
            "out_tag": out_tag_f,
            "require_zbin": bool(args.require_zbin),
        },
        "outputs": {"png": str(out_png)},
        "policy": {
            "eps_proxy_from_ratio": {
                "definition": "ε_proxy = (1/ratio)^(1/3) - 1, ratio=s∥/s⊥ (wedge peak; fid coordinate)",
                "status_thresholds": {"ok_max": 0.015, "mixed_max": 0.03},
                "note": "wedge-based proxy (screening). Phase A only; will be replaced by full peakfit with covariance.",
            }
        },
        "groups": {},
    }

    for (sample, caps, dist), xs in sorted(groups.items()):
        abs_eps_all = [float(p.abs_eps_proxy) for p in xs if bool(getattr(p, "reliable", True)) and math.isfinite(float(p.abs_eps_proxy))]
        worst_abs_eps = max(abs_eps_all) if abs_eps_all else float("nan")
        payload["groups"][f"{sample}/{caps}/{dist}"] = {
            "points": [
                {
                    "z_eff": p.z_eff,
                    "z_bin": p.z_bin,
                    "mu_split": p.mu_split,
                    "s_perp": p.s_perp,
                    "s_par": p.s_par,
                    "delta_s": p.delta_s,
                    "ratio": p.ratio,
                    "eps_proxy": p.eps_proxy,
                    "abs_eps_proxy": p.abs_eps_proxy,
                    "status": _status_from_abs_eps(float(p.abs_eps_proxy))
                    if bool(getattr(p, "reliable", True))
                    else "info",
                    "reliable": bool(getattr(p, "reliable", True)),
                    "flags": list(getattr(p, "flags", [])),
                    "source_json": p.source_json,
                }
                for p in xs
            ],
            "summary": {
                "worst_abs_eps_proxy": float(worst_abs_eps) if math.isfinite(float(worst_abs_eps)) else None,
                "status": _status_from_abs_eps(float(worst_abs_eps)) if math.isfinite(float(worst_abs_eps)) else "info",
            },
            "drift": {
                "delta_s": _drift([p.delta_s for p in xs]),
                "ratio": _drift([p.ratio for p in xs]),
                "abs_eps_proxy": _drift([p.abs_eps_proxy for p in xs if math.isfinite(float(p.abs_eps_proxy))]),
            },
        }

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_bao_catalog_wedge_anisotropy",
                "argv": sys.argv,
                "outputs": {"png": out_png, "metrics_json": out_json},
                "metrics": payload.get("filters", {}),
            }
        )
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
