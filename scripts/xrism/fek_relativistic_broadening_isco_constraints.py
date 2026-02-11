#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fek_relativistic_broadening_isco_constraints.py

Phase 4 / Step 4.8.7（Fe-Kα relativistic broad line / ISCO）:
ISCO（内縁半径 r_in）制約の I/F を固定するための出力スケルトン。

現段階では「一次データの取得・キャッシュ状況」と「系統ノブ台帳」を固定し、
反射モデル fit による r_in 推定は次段で実装する（XSPEC / heasoft / SAS 等の外部依存が絡むため）。
その代替として、XMM（PPS）と NuSTAR（event_cl）の「応答折り込み無し proxy」fit を固定出力化し、
ノブ（band/gain/rebin 等）に対する感度を先に台帳化する。

出力（固定）:
- output/private/xrism/fek_relativistic_broadening_isco_constraints.csv
- output/private/xrism/fek_relativistic_broadening_isco_constraints_metrics.json
- output/private/xrism/fek_relativistic_broadening_model_systematics.json
"""

from __future__ import annotations

import argparse
import csv
import io
import gzip
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    from scipy.optimize import least_squares  # type: ignore
    import scipy.sparse as sp  # type: ignore
except Exception:  # pragma: no cover
    least_squares = None
    sp = None

from scripts.cosmology.boss_dr12v5_fits import (  # noqa: E402
    _iter_cards_from_header_bytes,
    _read_header_blocks,
    read_bintable_columns,
    read_first_bintable_layout,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel(path: Path) -> str:
    try:
        return path.relative_to(_ROOT).as_posix()
    except Exception:
        return path.as_posix()


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _as_bool01(x: bool) -> str:
    return "1" if x else "0"


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not isinstance(r, dict):
                continue
            rows.append({str(k): (v or "").strip() for k, v in r.items() if k is not None})
    return rows


def _maybe_float(x: object) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        v = float(x)
        return v if math.isfinite(v) else None
    s = str(x).strip()
    if not s:
        return None
    try:
        v = float(s)
    except Exception:
        return None
    return v if math.isfinite(v) else None


_RIN_BOUND_RE = re.compile(r"r_in_bound[:=](upper|lower)", flags=re.IGNORECASE)


def _isco_gr_schwarzschild_rg() -> float:
    # diskline の r_in は通常 r_g=GM/c^2 単位（Schwarzschild）で与えられるため、
    # 参照枠として a*=0 の ISCO=6 r_g を基準線にする。
    return 6.0


def _isco_pmodel_exponential_metric_rg() -> float:
    # P-model の強場側の最小候補として「exponential metric」（Yilmaz型）
    #   g_tt = exp(-2m/r), g_space = exp(+2m/r)  （m≡GM/c^2）
    # を用いたときの test-particle（timelike）ISCO を返す。
    #
    # isotropic 半径 x=r/m の ISCO は解析的に x = 3 + sqrt(5)。
    # 物理半径としての比較は、周長半径 R = sqrt(g_θθ) = exp(1/x) * r を用い、
    # 出力は R/m（= r_g 単位）へ正規化する。
    x_isco = 3.0 + math.sqrt(5.0)
    return x_isco * math.exp(1.0 / x_isco)


def _compute_z_fields(
    *,
    r_in_rg: str,
    r_in_rg_stat: str,
    r_in_rg_sys: str,
    r_isco_gr_rg: float,
    r_isco_pmodel_rg: float,
) -> Dict[str, str]:
    rin = _maybe_float(r_in_rg)
    s = _maybe_float(r_in_rg_stat)
    u = _maybe_float(r_in_rg_sys)
    if rin is None or s is None or u is None:
        return {"delta_gr_rg": "", "z_gr": "", "delta_pmodel_rg": "", "z_pmodel": ""}
    sigma = math.sqrt(float(s) * float(s) + float(u) * float(u))
    if not (sigma > 0.0 and math.isfinite(sigma)):
        return {"delta_gr_rg": "", "z_gr": "", "delta_pmodel_rg": "", "z_pmodel": ""}

    d_gr = float(rin) - float(r_isco_gr_rg)
    d_p = float(rin) - float(r_isco_pmodel_rg)
    return {
        "delta_gr_rg": f"{d_gr:.6g}",
        "z_gr": f"{(d_gr / sigma):.6g}",
        "delta_pmodel_rg": f"{d_p:.6g}",
        "z_pmodel": f"{(d_p / sigma):.6g}",
    }


def _extract_sys_components_from_detail_json(detail_json: str) -> Dict[str, str]:
    if not detail_json:
        return {
            "sys_band_rg": "",
            "sys_gain_rg": "",
            "sys_rebin_rg": "",
            "sys_region_rg": "",
        }
    try:
        p = Path(detail_json)
        if not p.is_absolute():
            p = _ROOT / p
        obj = _read_json(p)
    except Exception:
        return {
            "sys_band_rg": "",
            "sys_gain_rg": "",
            "sys_rebin_rg": "",
            "sys_region_rg": "",
        }

    sys_obj = obj.get("systematics", {}) if isinstance(obj, dict) else {}
    comps = sys_obj.get("components", {}) if isinstance(sys_obj, dict) else {}
    if not isinstance(comps, dict):
        return {
            "sys_band_rg": "",
            "sys_gain_rg": "",
            "sys_rebin_rg": "",
            "sys_region_rg": "",
        }

    out: Dict[str, str] = {}
    for k in ("band", "gain", "rebin", "region"):
        v = comps.get(k)
        if isinstance(v, (int, float)) and math.isfinite(float(v)):
            out[f"sys_{k}_rg"] = f"{float(v):.6g}"
        else:
            out[f"sys_{k}_rg"] = ""
    for k in ("sys_band_rg", "sys_gain_rg", "sys_rebin_rg", "sys_region_rg"):
        out.setdefault(k, "")
    return out


def _load_cached_nustar_proxy_detail_json(detail_json: Path) -> Optional[Dict[str, Any]]:
    try:
        obj = _read_json(detail_json)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None

    fit_default = obj.get("fit_default")
    if not isinstance(fit_default, dict):
        return None
    if str(fit_default.get("status") or "").strip().lower() != "ok":
        return None

    sys_obj = obj.get("systematics")
    if not isinstance(sys_obj, dict):
        return None

    r_in_rg = _maybe_float(fit_default.get("r_in_rg"))
    r_in_rg_stat = _maybe_float(fit_default.get("r_in_rg_stat"))
    r_in_rg_sys = _maybe_float(sys_obj.get("sys_total"))
    if r_in_rg is None or r_in_rg_stat is None or r_in_rg_sys is None:
        return None

    line_det = obj.get("line_detection")
    line_det = line_det if isinstance(line_det, dict) else {}
    region = obj.get("region")
    region = region if isinstance(region, dict) else {}

    return {
        "status": "ok",
        "method_tag": "proxy_diskline_v1",
        "detail_json": _rel(detail_json),
        "r_in_rg": float(r_in_rg),
        "r_in_rg_stat": float(r_in_rg_stat),
        "r_in_rg_sys": float(r_in_rg_sys),
        "r_in_bound": str(fit_default.get("r_in_bound") or ""),
        "proxy_quality": str(line_det.get("proxy_quality") or ""),
        "proxy_line_detected": bool(line_det.get("detected")),
        "proxy_delta_chi2": float(line_det.get("delta_chi2", float("nan"))),
        "proxy_net_counts_band_pos": float(line_det.get("net_counts_band_pos", float("nan"))),
        "proxy_snr_band": float(line_det.get("snr_band_proxy", float("nan"))),
        "proxy_isco_constrained": bool(line_det.get("isco_constrained_proxy")),
        "proxy_region_mode": str(region.get("mode") or ""),
    }


def _compute_error_fields(*, r_in_bound: str, r_in_rg_stat: str, r_in_rg_sys: str) -> Dict[str, str]:
    if r_in_bound:
        return {"sigma_total_rg": "", "sys_stat_ratio": ""}
    s = _maybe_float(r_in_rg_stat)
    u = _maybe_float(r_in_rg_sys)
    if s is None or u is None:
        return {"sigma_total_rg": "", "sys_stat_ratio": ""}
    sigma_total = math.sqrt(float(s) * float(s) + float(u) * float(u))
    if not (sigma_total > 0.0 and math.isfinite(sigma_total)):
        return {"sigma_total_rg": "", "sys_stat_ratio": ""}
    ratio = float(u) / float(s) if float(s) != 0.0 else float("nan")
    return {
        "sigma_total_rg": f"{sigma_total:.6g}",
        "sys_stat_ratio": f"{ratio:.6g}" if math.isfinite(ratio) else "",
    }


def _plot_isco_constraints(rows: List[Dict[str, str]], *, out_png: Path) -> None:
    if plt is None:
        return

    out_png.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        fig, ax = plt.subplots(1, 1, figsize=(10.0, 2.6))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "Fe-Kα broad line / ISCO: no data\n(run the proxy pipeline to generate the CSV)",
            ha="center",
            va="center",
            fontsize=12,
            color="#333333",
        )
        fig.savefig(out_png, dpi=160)
        plt.close(fig)
        return

    ordered = sorted(rows, key=lambda r: (str(r.get("mission") or ""), str(r.get("obsid") or "")))
    labels: List[str] = []
    xs: List[float] = []
    stat_err: List[float] = []
    sys_err: List[float] = []
    colors: List[str] = []
    markers: List[str] = []

    col_map = {"xmm": "#1f77b4", "nustar": "#ff7f0e", "xrism": "#2ca02c"}

    for r in ordered:
        mission = str(r.get("mission") or "").strip().lower() or "?"
        obsid = str(r.get("obsid") or "").strip() or "?"
        instr = str(r.get("instrument_hint") or "").strip()
        label = f"{mission.upper()} {obsid}"
        if instr:
            label += f" ({instr})"

        rin = _maybe_float(r.get("r_in_rg"))
        if rin is None:
            continue
        s = _maybe_float(r.get("r_in_rg_stat")) or 0.0
        u = _maybe_float(r.get("r_in_rg_sys")) or 0.0

        note = str(r.get("note") or "")
        quality = str(r.get("proxy_quality") or "")
        bound = str(r.get("r_in_bound") or "").strip().lower()
        if not bound:
            m = _RIN_BOUND_RE.search(note)
            if m:
                bound = str(m.group(1)).lower()

        if quality == "no_broad_line":
            marker = "x"
        elif bound == "upper":
            marker = ">"
        elif bound == "lower":
            marker = "<"
        else:
            marker = "o"

        labels.append(label)
        xs.append(float(rin))
        stat_err.append(float(abs(s)))
        sys_err.append(float(abs(u)))
        colors.append(col_map.get(mission, "#444444"))
        markers.append(marker)

    if not xs:
        fig, ax = plt.subplots(1, 1, figsize=(10.0, 2.6))
        ax.axis("off")
        ax.text(0.5, 0.5, "Fe-Kα broad line / ISCO: no valid rows", ha="center", va="center", fontsize=12)
        fig.savefig(out_png, dpi=160)
        plt.close(fig)
        return

    n = len(xs)
    fig_h = max(3.8, 0.55 * n + 1.6)
    fig, ax = plt.subplots(1, 1, figsize=(12.0, fig_h))
    y = np.arange(n)

    for i in range(n):
        rin = float(xs[i])
        s = float(stat_err[i])
        u = float(sys_err[i])
        total = math.sqrt(s * s + u * u)
        c = colors[i]

        if total > 0:
            ax.hlines(float(y[i]), rin - total, rin + total, color=c, alpha=0.22, linewidth=6.0, zorder=1)
        if s > 0:
            ax.hlines(float(y[i]), rin - s, rin + s, color=c, alpha=0.9, linewidth=1.8, zorder=2)
        ax.plot(rin, float(y[i]), marker=markers[i], color=c, markersize=8, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("inner radius r_in (r_g)")
    ax.set_title("Fe-Kα broad line (GX 339-4): inner radius proxy constraints (method-dependent)")

    # Reference ISCO lines (to connect with Step 4.13.7 falsifiability).
    try:
        from matplotlib.lines import Line2D  # type: ignore

        r_isco_gr = _isco_gr_schwarzschild_rg()
        r_isco_p = _isco_pmodel_exponential_metric_rg()
        ax.axvline(r_isco_gr, color="#444444", linestyle="--", linewidth=1.2, alpha=0.9, zorder=0)
        ax.axvline(r_isco_p, color="#444444", linestyle=":", linewidth=1.2, alpha=0.9, zorder=0)
        ax.legend(
            handles=[
                Line2D([], [], color="#444444", linestyle="--", linewidth=1.2, label=f"GR ISCO (a*=0): {r_isco_gr:.2f} r_g"),
                Line2D(
                    [],
                    [],
                    color="#444444",
                    linestyle=":",
                    linewidth=1.2,
                    label=f"P-model ISCO (exp metric): {r_isco_p:.2f} r_g",
                ),
            ],
            loc="lower right",
            fontsize=8,
            frameon=False,
        )
    except Exception:
        pass

    x_max = max(xs) + max(1.0, 0.15 * max(xs))
    ax.set_xlim(left=0.0, right=float(min(60.0, max(10.0, x_max))))
    ax.grid(axis="x", color="#dddddd", linewidth=0.8, linestyle="--", alpha=0.8)

    fig.subplots_adjust(left=0.38, right=0.98, top=0.92, bottom=0.10)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _detect_cache_xmm(cache_root: Path, *, obsid: str, rev: str) -> bool:
    p = cache_root / "xmm" / rev / obsid / "PPS"
    if not p.exists():
        return False
    return any(p.glob("*.FTZ")) or any(p.glob("*.HTM"))


def _detect_cache_nustar(cache_root: Path, *, obsid: str) -> bool:
    p = cache_root / "nustar" / obsid / "event_cl"
    if not p.exists():
        return False
    return any(p.glob("*.evt.gz")) or any(p.glob("*.fits.gz")) or any(p.glob("*.gz"))


def _parse_header_kv(header_bytes: bytes) -> Dict[str, str]:
    kv: Dict[str, str] = {}
    for card in _iter_cards_from_header_bytes(header_bytes):
        key = card[:8].strip()
        if not key or "=" not in card:
            continue
        rhs = card.split("=", 1)[1]
        rhs = rhs.split("/", 1)[0].strip()
        kv[key] = rhs
    return kv


def _fits_read_spectrum_header(path: Path) -> Dict[str, str]:
    opener = gzip.open if path.name.lower().endswith(".ftz") or path.name.lower().endswith(".gz") else Path.open
    with opener(path, "rb") as f:  # type: ignore[arg-type]
        _ = _read_header_blocks(f)  # primary
        hdr = _read_header_blocks(f)  # first ext (SPECTRUM for PPS)
    return _parse_header_kv(hdr)


def _fits_read_spectrum_counts(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    opener = gzip.open if path.name.lower().endswith(".ftz") or path.name.lower().endswith(".gz") else Path.open
    with opener(path, "rb") as f:  # type: ignore[arg-type]
        layout = read_first_bintable_layout(f)
        cols = read_bintable_columns(f, layout=layout, columns=["CHANNEL", "COUNTS"])
    ch = np.asarray(cols["CHANNEL"], dtype=float)
    counts = np.asarray(cols["COUNTS"], dtype=float)
    return ch, counts


def _parse_float(kv: Dict[str, str], key: str, default: float = float("nan")) -> float:
    v = kv.get(key)
    if v is None:
        return float(default)
    s = str(v).strip().strip("'")
    try:
        return float(s)
    except Exception:
        return float(default)


def _parse_str(kv: Dict[str, str], key: str, default: str = "") -> str:
    v = kv.get(key)
    if v is None:
        return default
    return str(v).strip().strip("'")


def _energy_from_pi_channel(channel: np.ndarray, *, detchans: int) -> np.ndarray:
    # XMM EPIC PI bins are typically 5 eV; for DETCHANS=2400 this maps to 0–12 keV.
    # We use this as a proxy mapping to avoid requiring external RMF files.
    ch = np.asarray(channel, dtype=float)
    step_keV = 0.005
    # center of bin
    return (ch + 0.5) * step_keV


def _energy_from_nustar_pi(pi: np.ndarray) -> np.ndarray:
    # NuSTAR PI channels are typically 40 eV (0.04 keV) bins.
    # We use this as a proxy mapping (no RMF folding) to allow offline reproducibility.
    ch = np.asarray(pi, dtype=float)
    step_keV = 0.04
    return (ch + 0.5) * step_keV


def _fits_read_event_pi(path: Path) -> np.ndarray:
    opener = gzip.open if path.name.lower().endswith(".gz") else Path.open
    with opener(path, "rb") as f:  # type: ignore[arg-type]
        layout = read_first_bintable_layout(f)
        col_map = {str(c).upper(): str(c) for c in layout.columns}
        pi_key = col_map.get("PI")
        if pi_key is None:
            raise ValueError("PI column not found in event file")
        cols = read_bintable_columns(f, layout=layout, columns=[pi_key])
    pi = np.asarray(cols[pi_key], dtype=float)
    pi = pi[np.isfinite(pi)]
    pi = np.asarray(np.round(pi), dtype=int)
    pi = pi[pi >= 0]
    return pi


def _fits_read_event_columns(path: Path, columns: Sequence[str]) -> Dict[str, np.ndarray]:
    """
    Read selected scalar columns from an event file (first BINTABLE extension).
    Column names are matched case-insensitively.
    """
    opener = gzip.open if path.name.lower().endswith(".gz") else Path.open
    with opener(path, "rb") as f:  # type: ignore[arg-type]
        layout = read_first_bintable_layout(f)
        col_map = {str(c).upper(): str(c) for c in layout.columns}
        resolved: List[str] = []
        for c in columns:
            key = col_map.get(str(c).upper())
            if key is None:
                raise ValueError(f"column not found in event file: {c!r}")
            resolved.append(key)
        cols = read_bintable_columns(f, layout=layout, columns=resolved)
    # return with requested names (upper-case keys)
    out: Dict[str, np.ndarray] = {}
    for c_req, c_real in zip(columns, resolved):
        out[str(c_req).upper()] = np.asarray(cols[c_real], dtype=float)
    return out


def _accumulate_bincount(acc: np.ndarray, bc: np.ndarray) -> np.ndarray:
    if bc.size > acc.size:
        acc = np.pad(acc, (0, int(bc.size - acc.size)), mode="constant")
    acc[: int(bc.size)] += bc.astype(float)
    return acc


def _nustar_det1_peak_center(
    event_files: Sequence[Path],
    *,
    bin_size: int,
    det1_max: int,
) -> Tuple[Optional[Tuple[float, float]], Dict[str, Any]]:
    """
    Estimate a source center in DET1 coordinates by finding the peak in a coarse 2D histogram.
    This is an approximation to avoid external region/extraction tools (HEASoft).
    """
    dbg: Dict[str, Any] = {"method": "det1_hist_peak", "bin_size": int(bin_size), "det1_max": int(det1_max), "files": []}
    if int(bin_size) <= 0:
        raise ValueError("bin_size must be > 0")
    if int(det1_max) <= 0:
        raise ValueError("det1_max must be > 0")
    n_bins = int(det1_max) // int(bin_size) + 1
    if n_bins < 4:
        n_bins = 4

    H = np.zeros((n_bins, n_bins), dtype=np.int64)
    n_used = 0
    for p in event_files:
        try:
            cols = _fits_read_event_columns(p, ["DET1X", "DET1Y"])
            x = cols["DET1X"]
            y = cols["DET1Y"]
            m = (
                np.isfinite(x)
                & np.isfinite(y)
                & (x >= 0.0)
                & (y >= 0.0)
                & (x < float(det1_max))
                & (y < float(det1_max))
            )
            if not np.any(m):
                dbg["files"].append({"path": _rel(p), "n_rows": int(x.size), "n_used": 0})
                continue
            xi = np.asarray(np.floor(x[m] / float(bin_size)), dtype=np.int64)
            yi = np.asarray(np.floor(y[m] / float(bin_size)), dtype=np.int64)
            xi = np.clip(xi, 0, n_bins - 1)
            yi = np.clip(yi, 0, n_bins - 1)
            idx = xi * int(n_bins) + yi
            bc = np.bincount(idx, minlength=int(n_bins * n_bins)).astype(np.int64)
            H += bc.reshape(n_bins, n_bins)
            n_used += int(np.count_nonzero(m))
            dbg["files"].append({"path": _rel(p), "n_rows": int(x.size), "n_used": int(np.count_nonzero(m))})
        except Exception as e:
            dbg["files"].append({"path": _rel(p), "error": str(e)})
            continue

    if int(np.sum(H)) <= 0:
        dbg["status"] = "no_valid_events"
        return None, dbg

    ij = np.unravel_index(int(np.argmax(H)), H.shape)
    i = int(ij[0])
    j = int(ij[1])
    cx = (float(i) + 0.5) * float(bin_size)
    cy = (float(j) + 0.5) * float(bin_size)
    dbg["status"] = "ok"
    dbg["peak_bin"] = [int(i), int(j)]
    dbg["peak_count"] = int(H[i, j])
    dbg["n_used_total"] = int(n_used)
    dbg["center_det1"] = [float(cx), float(cy)]
    return (float(cx), float(cy)), dbg


def _nustar_extract_net_spectrum_det1(
    event_files: Sequence[Path],
    *,
    center_det1: Tuple[float, float],
    src_radius: float,
    bkg_inner: float,
    bkg_outer: float,
    det1_max: int,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Build a crude net spectrum from NuSTAR event files using DET1 circular source region
    and annular background region. Background is scaled by geometric area ratio.
    """
    cx, cy = float(center_det1[0]), float(center_det1[1])
    r_src = float(src_radius)
    r_bi = float(bkg_inner)
    r_bo = float(bkg_outer)
    if not (r_src > 0 and r_bi >= 0 and r_bo > r_bi):
        raise ValueError("invalid radii for NuSTAR region extraction")

    dbg: Dict[str, Any] = {
        "method": "det1_circ_minus_annulus",
        "center_det1": [cx, cy],
        "src_radius": r_src,
        "bkg_inner": r_bi,
        "bkg_outer": r_bo,
        "det1_max": int(det1_max),
        "files": [],
    }

    src_counts = np.zeros(0, dtype=float)
    bkg_counts = np.zeros(0, dtype=float)
    n_src = 0
    n_bkg = 0
    n_total_valid = 0

    r2_src = r_src * r_src
    r2_bi = r_bi * r_bi
    r2_bo = r_bo * r_bo

    for p in event_files:
        try:
            cols = _fits_read_event_columns(p, ["PI", "DET1X", "DET1Y"])
            pi0 = cols["PI"]
            x0 = cols["DET1X"]
            y0 = cols["DET1Y"]
            m = (
                np.isfinite(pi0)
                & np.isfinite(x0)
                & np.isfinite(y0)
                & (pi0 >= 0.0)
                & (x0 >= 0.0)
                & (y0 >= 0.0)
                & (x0 < float(det1_max))
                & (y0 < float(det1_max))
            )
            if not np.any(m):
                dbg["files"].append({"path": _rel(p), "n_rows": int(pi0.size), "n_valid": 0})
                continue
            pi = np.asarray(np.round(pi0[m]), dtype=np.int64)
            x = np.asarray(x0[m], dtype=float)
            y = np.asarray(y0[m], dtype=float)
            n_total_valid += int(pi.size)
            dx = x - cx
            dy = y - cy
            d2 = dx * dx + dy * dy
            m_src = d2 <= r2_src
            m_bkg = (d2 >= r2_bi) & (d2 <= r2_bo)

            if np.any(m_src):
                bc = np.bincount(pi[m_src], minlength=int(np.max(pi[m_src])) + 1).astype(float)
                src_counts = _accumulate_bincount(src_counts, bc)
                n_src += int(np.count_nonzero(m_src))
            if np.any(m_bkg):
                bc = np.bincount(pi[m_bkg], minlength=int(np.max(pi[m_bkg])) + 1).astype(float)
                bkg_counts = _accumulate_bincount(bkg_counts, bc)
                n_bkg += int(np.count_nonzero(m_bkg))

            dbg["files"].append(
                {
                    "path": _rel(p),
                    "n_rows": int(pi0.size),
                    "n_valid": int(pi.size),
                    "n_src": int(np.count_nonzero(m_src)),
                    "n_bkg": int(np.count_nonzero(m_bkg)),
                }
            )
        except Exception as e:
            dbg["files"].append({"path": _rel(p), "error": str(e)})
            continue

    if src_counts.size < 10 or float(np.sum(src_counts)) <= 0:
        dbg["status"] = "empty_source_counts"
        dbg["n_src_events"] = int(n_src)
        dbg["n_bkg_events"] = int(n_bkg)
        dbg["n_valid_events_total"] = int(n_total_valid)
        return None, dbg

    # Scale background by geometric area ratio (proxy).
    area_src = math.pi * r2_src
    area_bkg = math.pi * max((r2_bo - r2_bi), 1e-12)
    scale = float(area_src / area_bkg) if float(np.sum(bkg_counts)) > 0 else 0.0

    # align arrays
    if bkg_counts.size > src_counts.size:
        src_counts = np.pad(src_counts, (0, int(bkg_counts.size - src_counts.size)), mode="constant")
    elif src_counts.size > bkg_counts.size:
        bkg_counts = np.pad(bkg_counts, (0, int(src_counts.size - bkg_counts.size)), mode="constant")

    net = src_counts - float(scale) * bkg_counts
    var = np.clip(src_counts, 0.0, None) + (float(scale) ** 2) * np.clip(bkg_counts, 0.0, None)

    dbg["status"] = "ok"
    dbg["n_src_events"] = int(n_src)
    dbg["n_bkg_events"] = int(n_bkg)
    dbg["n_valid_events_total"] = int(n_total_valid)
    dbg["background_scale_area_ratio"] = float(scale)
    dbg["src_counts_sum"] = float(np.sum(src_counts))
    dbg["bkg_counts_sum"] = float(np.sum(bkg_counts))
    dbg["net_counts_sum"] = float(np.sum(net))

    return {"src_counts": src_counts, "bkg_counts": bkg_counts, "net_counts": net, "var_counts": var}, dbg


_DS9_CIRCLE_RE = re.compile(
    r"CIRCLE\s*\(\s*(?P<x>[-+0-9.eE]+)\s*,\s*(?P<y>[-+0-9.eE]+)\s*,\s*(?P<r>[-+0-9.eE]+)\s*\)",
    flags=re.IGNORECASE,
)


def _parse_ds9_circle_region(path: Path) -> Optional[Tuple[float, float, float]]:
    """
    Parse a minimal DS9 region file containing a single circle, e.g.:
      CIRCLE (500.5, 500.5, 20)

    NuSTAR event_cl provides *_src.reg in (X,Y) image coordinates (not DET1).
    """
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    m = _DS9_CIRCLE_RE.search(text.replace("\n", " ").strip())
    if not m:
        return None
    try:
        x = float(m.group("x"))
        y = float(m.group("y"))
        r = float(m.group("r"))
    except Exception:
        return None
    if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(r) and r > 0):
        return None
    return float(x), float(y), float(r)


def _nustar_extract_net_spectrum_xy_circle(
    event_files: Sequence[Path],
    *,
    center_xy: Tuple[float, float],
    src_radius: float,
    bkg_inner: float,
    bkg_outer: float,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Build a crude net spectrum from NuSTAR event files using (X,Y) circular source region
    and annular background region. Background is scaled by geometric area ratio.

    This mode is intended to consume NuSTAR pipeline region files (event_cl/*_src.reg),
    which are defined in (X,Y) image coordinates rather than DET1.
    """
    cx, cy = float(center_xy[0]), float(center_xy[1])
    r_src = float(src_radius)
    r_bi = float(bkg_inner)
    r_bo = float(bkg_outer)
    if not (r_src > 0 and r_bi >= 0 and r_bo > r_bi):
        raise ValueError("invalid radii for NuSTAR XY region extraction")

    dbg: Dict[str, Any] = {
        "method": "xy_circ_minus_annulus",
        "center_xy": [cx, cy],
        "src_radius": r_src,
        "bkg_inner": r_bi,
        "bkg_outer": r_bo,
        "files": [],
    }

    src_counts = np.zeros(0, dtype=float)
    bkg_counts = np.zeros(0, dtype=float)
    n_src = 0
    n_bkg = 0
    n_total_valid = 0

    r2_src = r_src * r_src
    r2_bi = r_bi * r_bi
    r2_bo = r_bo * r_bo

    for p in event_files:
        try:
            cols = _fits_read_event_columns(p, ["PI", "X", "Y"])
            pi0 = cols["PI"]
            x0 = cols["X"]
            y0 = cols["Y"]
            m = np.isfinite(pi0) & np.isfinite(x0) & np.isfinite(y0) & (pi0 >= 0.0) & (x0 >= 0.0) & (y0 >= 0.0)
            if not np.any(m):
                dbg["files"].append({"path": _rel(p), "n_rows": int(pi0.size), "n_valid": 0})
                continue
            pi = np.asarray(np.round(pi0[m]), dtype=np.int64)
            x = np.asarray(x0[m], dtype=float)
            y = np.asarray(y0[m], dtype=float)
            n_total_valid += int(pi.size)
            dx = x - cx
            dy = y - cy
            d2 = dx * dx + dy * dy
            m_src = d2 <= r2_src
            m_bkg = (d2 >= r2_bi) & (d2 <= r2_bo)

            if np.any(m_src):
                bc = np.bincount(pi[m_src], minlength=int(np.max(pi[m_src])) + 1).astype(float)
                src_counts = _accumulate_bincount(src_counts, bc)
                n_src += int(np.count_nonzero(m_src))
            if np.any(m_bkg):
                bc = np.bincount(pi[m_bkg], minlength=int(np.max(pi[m_bkg])) + 1).astype(float)
                bkg_counts = _accumulate_bincount(bkg_counts, bc)
                n_bkg += int(np.count_nonzero(m_bkg))

            dbg["files"].append(
                {
                    "path": _rel(p),
                    "n_rows": int(pi0.size),
                    "n_valid": int(pi.size),
                    "n_src": int(np.count_nonzero(m_src)),
                    "n_bkg": int(np.count_nonzero(m_bkg)),
                }
            )
        except Exception as e:
            dbg["files"].append({"path": _rel(p), "error": str(e)})
            continue

    if src_counts.size < 10 or float(np.sum(src_counts)) <= 0:
        dbg["status"] = "empty_source_counts"
        dbg["n_src_events"] = int(n_src)
        dbg["n_bkg_events"] = int(n_bkg)
        dbg["n_valid_events_total"] = int(n_total_valid)
        return None, dbg

    area_src = math.pi * r2_src
    area_bkg = math.pi * max((r2_bo - r2_bi), 1e-12)
    scale = float(area_src / area_bkg) if float(np.sum(bkg_counts)) > 0 else 0.0

    if bkg_counts.size > src_counts.size:
        src_counts = np.pad(src_counts, (0, int(bkg_counts.size - src_counts.size)), mode="constant")
    elif src_counts.size > bkg_counts.size:
        bkg_counts = np.pad(bkg_counts, (0, int(src_counts.size - bkg_counts.size)), mode="constant")

    net = src_counts - float(scale) * bkg_counts
    var = np.clip(src_counts, 0.0, None) + (float(scale) ** 2) * np.clip(bkg_counts, 0.0, None)

    dbg["status"] = "ok"
    dbg["n_src_events"] = int(n_src)
    dbg["n_bkg_events"] = int(n_bkg)
    dbg["n_valid_events_total"] = int(n_total_valid)
    dbg["background_scale_area_ratio"] = float(scale)
    dbg["src_counts_sum"] = float(np.sum(src_counts))
    dbg["bkg_counts_sum"] = float(np.sum(bkg_counts))
    dbg["net_counts_sum"] = float(np.sum(net))

    return {"src_counts": src_counts, "bkg_counts": bkg_counts, "net_counts": net, "var_counts": var}, dbg


def _rebin_min_counts(
    energy: np.ndarray,
    net_counts: np.ndarray,
    var_counts: np.ndarray,
    src_counts: np.ndarray,
    *,
    min_counts: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    e = np.asarray(energy, dtype=float)
    y = np.asarray(net_counts, dtype=float)
    v = np.asarray(var_counts, dtype=float)
    s = np.asarray(src_counts, dtype=float)
    if e.size != y.size or e.size != v.size or e.size != s.size:
        raise ValueError("shape mismatch in rebin")
    if min_counts <= 1:
        return e, y, v
    out_e: List[float] = []
    out_y: List[float] = []
    out_v: List[float] = []
    acc_e = 0.0
    acc_y = 0.0
    acc_v = 0.0
    acc_s = 0.0
    acc_n = 0
    for i in range(int(e.size)):
        acc_e += float(e[i])
        acc_y += float(y[i])
        acc_v += float(v[i])
        acc_s += float(max(s[i], 0.0))
        acc_n += 1
        if acc_s >= float(min_counts):
            out_e.append(acc_e / float(acc_n))
            out_y.append(acc_y)
            out_v.append(max(acc_v, 1.0))
            acc_e = acc_y = acc_v = acc_s = 0.0
            acc_n = 0
    if acc_n > 0:
        out_e.append(acc_e / float(acc_n))
        out_y.append(acc_y)
        out_v.append(max(acc_v, 1.0))
    return np.asarray(out_e, dtype=float), np.asarray(out_y, dtype=float), np.asarray(out_v, dtype=float)


def _rebin_min_counts_groups(
    energy: np.ndarray,
    net_counts: np.ndarray,
    var_counts: np.ndarray,
    src_counts: np.ndarray,
    *,
    min_counts: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Same as _rebin_min_counts, but also return group start indices (0-based)
    suitable for np.add.reduceat on the original arrays.
    """
    e = np.asarray(energy, dtype=float)
    y = np.asarray(net_counts, dtype=float)
    v = np.asarray(var_counts, dtype=float)
    s = np.asarray(src_counts, dtype=float)
    if e.size != y.size or e.size != v.size or e.size != s.size:
        raise ValueError("shape mismatch in rebin")
    if min_counts <= 1:
        starts = np.arange(int(e.size), dtype=np.int64)
        return e, y, v, starts

    out_e: List[float] = []
    out_y: List[float] = []
    out_v: List[float] = []
    out_starts: List[int] = []

    acc_e = 0.0
    acc_y = 0.0
    acc_v = 0.0
    acc_s = 0.0
    acc_n = 0
    start = 0

    for i in range(int(e.size)):
        acc_e += float(e[i])
        acc_y += float(y[i])
        acc_v += float(v[i])
        acc_s += float(max(s[i], 0.0))
        acc_n += 1
        if acc_s >= float(min_counts):
            out_starts.append(int(start))
            out_e.append(acc_e / float(acc_n))
            out_y.append(acc_y)
            out_v.append(max(acc_v, 1.0))
            start = int(i) + 1
            acc_e = acc_y = acc_v = acc_s = 0.0
            acc_n = 0

    if acc_n > 0:
        out_starts.append(int(start))
        out_e.append(acc_e / float(acc_n))
        out_y.append(acc_y)
        out_v.append(max(acc_v, 1.0))

    starts = np.asarray(out_starts, dtype=np.int64)
    return np.asarray(out_e, dtype=float), np.asarray(out_y, dtype=float), np.asarray(out_v, dtype=float), starts


def _rebin_min_counts_weighted(
    energy: np.ndarray,
    net_counts: np.ndarray,
    var_counts: np.ndarray,
    weight: np.ndarray,
    src_counts: np.ndarray,
    *,
    min_counts: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rebin bins until src_counts sum reaches min_counts, while producing a
    weight-normalized y and variance:
      y = sum(net)/sum(w)
      var = sum(var)/sum(w)^2

    This is used when we treat throughput as a known multiplicative weight w(E)
    and fit in "flux proxy" space.
    """
    e = np.asarray(energy, dtype=float)
    y = np.asarray(net_counts, dtype=float)
    v = np.asarray(var_counts, dtype=float)
    w = np.asarray(weight, dtype=float)
    s = np.asarray(src_counts, dtype=float)
    if e.size != y.size or e.size != v.size or e.size != w.size or e.size != s.size:
        raise ValueError("shape mismatch in weighted rebin")
    if min_counts <= 1:
        ww = np.clip(w, 1e-30, None)
        return e, (y / ww), (v / (ww**2))

    out_e: List[float] = []
    out_y: List[float] = []
    out_v: List[float] = []

    acc_w = 0.0
    acc_e_w = 0.0
    acc_y = 0.0
    acc_v = 0.0
    acc_s = 0.0

    for i in range(int(e.size)):
        wi = float(max(float(w[i]), 0.0))
        acc_w += wi
        acc_e_w += float(e[i]) * wi
        acc_y += float(y[i])
        acc_v += float(v[i])
        acc_s += float(max(s[i], 0.0))
        if acc_s >= float(min_counts):
            ww = float(max(acc_w, 1e-30))
            out_e.append(acc_e_w / ww)
            out_y.append(acc_y / ww)
            out_v.append(max(acc_v, 1.0) / (ww**2))
            acc_w = acc_e_w = acc_y = acc_v = acc_s = 0.0

    if acc_w > 0.0:
        ww = float(max(acc_w, 1e-30))
        out_e.append(acc_e_w / ww)
        out_y.append(acc_y / ww)
        out_v.append(max(acc_v, 1.0) / (ww**2))

    return np.asarray(out_e, dtype=float), np.asarray(out_y, dtype=float), np.asarray(out_v, dtype=float)


def _diskline_profile_grid(
    *,
    energy: np.ndarray,
    e0_keV: float,
    incl_deg: float,
    emissivity_q: float,
    r_out_rg: float,
    r_in_grid: np.ndarray,
    sigma_instr_keV: float,
    n_r: int = 200,
    n_phi: int = 200,
) -> np.ndarray:
    e = np.asarray(energy, dtype=float)
    if e.size < 10:
        raise ValueError("energy grid too small")
    # build bin edges from midpoints (assume roughly uniform; fallback to nearest-neighbor edges)
    edges = np.empty(e.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (e[:-1] + e[1:])
    edges[0] = e[0] - (edges[1] - e[0])
    edges[-1] = e[-1] + (e[-1] - edges[-2])

    incl = math.radians(float(incl_deg))
    sin_i = math.sin(incl)
    phi = np.linspace(0.0, 2.0 * math.pi, int(n_phi), endpoint=False, dtype=float)
    cosphi = np.cos(phi, dtype=float)

    prof = np.zeros((int(r_in_grid.size), int(e.size)), dtype=float)

    # Precompute radial nodes in a normalized coordinate u in [0,1], then map to [r_in,r_out]
    u = np.linspace(0.0, 1.0, int(n_r), dtype=float)
    du = float(u[1] - u[0]) if u.size > 1 else 1.0

    for i, r_in in enumerate(np.asarray(r_in_grid, dtype=float)):
        rin = float(max(r_in, 2.05))  # avoid inside r=2M for Schwarzschild proxy
        rout = float(max(r_out_rg, rin + 1e-6))
        # log spacing captures inner region better
        r = rin * (rout / rin) ** u
        dr = np.gradient(r)
        rr = r.reshape(-1, 1)

        v = np.sqrt(1.0 / rr)  # Newtonian proxy
        v = np.clip(v, 0.0, 0.95)
        gamma = 1.0 / np.sqrt(1.0 - v**2)
        g_gr = np.sqrt(np.clip(1.0 - 2.0 / rr, 1e-6, None))
        denom = gamma * (1.0 - v * sin_i * cosphi.reshape(1, -1))
        g = g_gr / denom
        ee = float(e0_keV) * g

        # emissivity weight ~ r^{-q} * area element
        w_r = (rr[:, 0] ** (-float(emissivity_q))) * rr[:, 0] * dr
        w = (w_r.reshape(-1, 1) * np.ones((1, int(n_phi)), dtype=float)).reshape(-1)
        ee_flat = ee.reshape(-1)

        hist, _ = np.histogram(ee_flat, bins=edges, weights=w)
        h = np.asarray(hist, dtype=float)
        if sigma_instr_keV > 0:
            # Gaussian smoothing in energy bins
            # approximate sigma in bins using local step
            step = float(np.nanmedian(np.diff(e))) if e.size > 1 else 1.0
            sig_bins = max(float(sigma_instr_keV) / step, 0.5)
            half = int(max(3, math.ceil(4.0 * sig_bins)))
            x = np.arange(-half, half + 1, dtype=float)
            ker = np.exp(-0.5 * (x / sig_bins) ** 2)
            ker /= float(np.sum(ker)) if float(np.sum(ker)) > 0 else 1.0
            h = np.convolve(h, ker, mode="same")

        s = float(np.sum(h))
        if s > 0:
            h = h / s
        prof[i, :] = h

    return prof


def _interp_profile(r_in: float, r_grid: np.ndarray, prof_grid: np.ndarray) -> np.ndarray:
    rg = np.asarray(r_grid, dtype=float)
    if r_in <= float(rg[0]):
        return np.asarray(prof_grid[0], dtype=float)
    if r_in >= float(rg[-1]):
        return np.asarray(prof_grid[-1], dtype=float)
    j = int(np.searchsorted(rg, float(r_in), side="right")) - 1
    j = max(0, min(j, int(rg.size) - 2))
    r0 = float(rg[j])
    r1 = float(rg[j + 1])
    t = (float(r_in) - r0) / (r1 - r0) if r1 != r0 else 0.0
    return (1.0 - t) * np.asarray(prof_grid[j], dtype=float) + t * np.asarray(prof_grid[j + 1], dtype=float)


def _fit_powerlaw_only_proxy(
    *,
    energy_keV: np.ndarray,
    net_counts: np.ndarray,
    var_counts: np.ndarray,
    weight: Optional[np.ndarray] = None,
    band_keV: Tuple[float, float],
    min_counts: int,
    gain_shift: float,
) -> Dict[str, Any]:
    if least_squares is None:
        raise RuntimeError("scipy is required for fitting (scipy.optimize.least_squares)")
    e = np.asarray(energy_keV, dtype=float) * (1.0 + float(gain_shift))
    y = np.asarray(net_counts, dtype=float)
    v = np.asarray(var_counts, dtype=float)
    w = np.asarray(weight, dtype=float) if weight is not None else None
    m_band = (e >= float(band_keV[0])) & (e <= float(band_keV[1])) & np.isfinite(e) & np.isfinite(y) & np.isfinite(v)
    if not np.any(m_band):
        raise RuntimeError("empty band after filtering")
    e = e[m_band]
    y = y[m_band]
    v = np.clip(v[m_band], 1.0, None)
    if w is not None:
        w = np.clip(w[m_band], 0.0, None)

    # Rebin for stability (match diskline proxy).
    src_proxy = np.clip(y + np.sqrt(v), 0.0, None)  # not exact, but monotone proxy
    if w is None:
        e, y, v = _rebin_min_counts(e, y, v, src_proxy, min_counts=int(min_counts))
        sigma = np.sqrt(np.clip(v, 1.0, None))
    else:
        e, y, v = _rebin_min_counts_weighted(e, y, v, w, src_proxy, min_counts=int(min_counts))
        sigma = np.sqrt(np.clip(v, 1e-30, None))

    # Initial guess from log-linear regression.
    ee = e
    y_floor = 1.0 if w is None else 1e-30
    yy = np.clip(y, float(y_floor), None)
    A0 = float(np.median(yy))
    g0 = 2.0
    try:
        X = np.vstack([np.ones_like(ee), -np.log(np.clip(ee, 1e-3, None))]).T
        beta, *_ = np.linalg.lstsq(X, np.log(yy), rcond=None)
        A0 = float(np.exp(beta[0]))
        g0 = float(np.clip(beta[1], 0.0, 5.0))
    except Exception:
        pass

    lo = np.asarray([0.0, 0.0], dtype=float)
    hi = np.asarray([np.inf, 5.0], dtype=float)

    def _model(params: np.ndarray) -> np.ndarray:
        A, gamma = [float(x) for x in params]
        return A * (np.clip(e, 1e-3, None) ** (-gamma))

    def _resid(params: np.ndarray) -> np.ndarray:
        return (y - _model(params)) / sigma

    x0 = np.asarray([A0, g0], dtype=float)
    res = least_squares(_resid, x0=x0, bounds=(lo, hi), max_nfev=300)
    p = np.asarray(res.x, dtype=float)
    yhat = _model(p)
    chi2 = float(np.sum(((y - yhat) / sigma) ** 2))
    dof = int(max(int(y.size) - int(p.size), 1))
    redchi2 = chi2 / float(dof)

    return {
        "status": "ok",
        "band_keV": [float(band_keV[0]), float(band_keV[1])],
        "min_counts": int(min_counts),
        "gain_shift": float(gain_shift),
        "params": {"A": float(p[0]), "gamma": float(p[1])},
        "fit": {"chi2": float(chi2), "dof": int(dof), "redchi2": float(redchi2), "nfev": int(res.nfev)},
    }


def _fit_powerlaw_plus_diskline_proxy(
    *,
    energy_keV: np.ndarray,
    net_counts: np.ndarray,
    var_counts: np.ndarray,
    weight: Optional[np.ndarray] = None,
    band_keV: Tuple[float, float],
    min_counts: int,
    gain_shift: float,
    e0_keV: float = 6.4,
    incl_deg: float = 45.0,
    emissivity_q: float = 3.0,
    r_out_rg: float = 400.0,
    sigma_instr_keV: float = 0.07,
) -> Dict[str, Any]:
    if least_squares is None:
        raise RuntimeError("scipy is required for fitting (scipy.optimize.least_squares)")
    e = np.asarray(energy_keV, dtype=float) * (1.0 + float(gain_shift))
    y = np.asarray(net_counts, dtype=float)
    v = np.asarray(var_counts, dtype=float)
    w = np.asarray(weight, dtype=float) if weight is not None else None
    m_band = (e >= float(band_keV[0])) & (e <= float(band_keV[1])) & np.isfinite(e) & np.isfinite(y) & np.isfinite(v)
    if not np.any(m_band):
        raise RuntimeError("empty band after filtering")
    e = e[m_band]
    y = y[m_band]
    v = np.clip(v[m_band], 1.0, None)
    if w is not None:
        w = np.clip(w[m_band], 0.0, None)

    # Rebin for stability
    src_proxy = np.clip(y + np.sqrt(v), 0.0, None)  # not exact, but monotone proxy
    if w is None:
        e, y, v = _rebin_min_counts(e, y, v, src_proxy, min_counts=min_counts)
    else:
        e, y, v = _rebin_min_counts_weighted(e, y, v, w, src_proxy, min_counts=min_counts)
    if w is None:
        sigma = np.sqrt(np.clip(v, 1.0, None))
    else:
        sigma = np.sqrt(np.clip(v, 1e-30, None))

    # Diskline profile precompute (r_in grid in rg)
    r_grid = np.unique(np.concatenate([np.linspace(2.5, 10.0, 16), np.linspace(10.0, 50.0, 21)]))
    prof_grid = _diskline_profile_grid(
        energy=e,
        e0_keV=float(e0_keV),
        incl_deg=float(incl_deg),
        emissivity_q=float(emissivity_q),
        r_out_rg=float(r_out_rg),
        r_in_grid=r_grid,
        sigma_instr_keV=float(sigma_instr_keV),
    )

    # Initial continuum guess (fit powerlaw to two side bands).
    # If weight is provided, y is already normalized by weight (flux-proxy space).
    cont_mask = ((e >= float(band_keV[0])) & (e <= 5.0)) | ((e >= 7.0) & (e <= float(band_keV[1])))
    if np.count_nonzero(cont_mask) < 10:
        cont_mask = np.ones_like(e, dtype=bool)
    ee = e[cont_mask]
    y_floor = 1.0 if w is None else 1e-30
    yy = np.clip(y[cont_mask], float(y_floor), None)
    # log-linear fit: log y = log A - gamma log E
    A0 = float(np.median(yy))
    g0 = 2.0
    try:
        X = np.vstack([np.ones_like(ee), -np.log(np.clip(ee, 1e-3, None))]).T
        beta, *_ = np.linalg.lstsq(X, np.log(yy), rcond=None)
        A0 = float(np.exp(beta[0]))
        g0 = float(np.clip(beta[1], 0.0, 5.0))
    except Exception:
        pass
    # Line amplitude rough guess from residual around 6–7 keV
    m_line = (e >= 5.5) & (e <= 7.5)
    cont0 = A0 * (np.clip(e, 1e-3, None) ** (-g0))
    line0 = float(np.sum(np.clip(y[m_line] - cont0[m_line], 0.0, None)))
    r0 = 6.0

    lo = np.asarray([0.0, 0.0, 0.0, 2.5], dtype=float)
    hi = np.asarray([np.inf, 5.0, np.inf, 50.0], dtype=float)

    def _model(params: np.ndarray) -> np.ndarray:
        A, gamma, n_line, r_in = [float(x) for x in params]
        cont = A * (np.clip(e, 1e-3, None) ** (-gamma))
        prof = _interp_profile(r_in, r_grid, prof_grid)
        line = n_line * prof
        return cont + line

    def _resid(params: np.ndarray) -> np.ndarray:
        return (y - _model(params)) / sigma

    # Multi-start on r_in to reduce local-minimum/boundary sticking in proxy fits.
    r_in_inits = [3.0, 4.5, 6.0, 10.0, 20.0, 40.0]
    best_res = None
    best_chi2 = float("inf")
    best_dof = 1
    for r_init in r_in_inits:
        x0 = np.asarray([A0, g0, max(line0, float(y_floor)), float(r_init)], dtype=float)
        try:
            res0 = least_squares(_resid, x0=x0, bounds=(lo, hi), max_nfev=500)
        except Exception:
            continue
        p0 = np.asarray(res0.x, dtype=float)
        yhat0 = _model(p0)
        chi20 = float(np.sum(((y - yhat0) / sigma) ** 2))
        dof0 = int(max(int(y.size) - int(p0.size), 1))
        if chi20 < best_chi2:
            best_chi2 = chi20
            best_res = res0
            best_dof = dof0

    if best_res is None:
        raise RuntimeError("least_squares failed for all initializations")

    res = best_res
    p = np.asarray(res.x, dtype=float)
    yhat = _model(p)
    chi2 = float(np.sum(((y - yhat) / sigma) ** 2))
    dof = int(best_dof)
    redchi2 = chi2 / float(dof)

    # Approximate covariance
    r_in_stat = float("nan")
    try:
        J = np.asarray(res.jac, dtype=float)
        JTJ = J.T @ J
        cov = np.linalg.inv(JTJ) * float(redchi2)
        r_in_stat = float(np.sqrt(max(float(cov[3, 3]), 0.0)))
    except Exception:
        pass

    r_in = float(p[3])
    r_in_bound = ""
    if abs(r_in - float(lo[3])) < 1e-6:
        r_in_bound = "lower"
    elif abs(r_in - float(hi[3])) < 1e-6:
        r_in_bound = "upper"

    return {
        "status": "ok",
        "band_keV": [float(band_keV[0]), float(band_keV[1])],
        "min_counts": int(min_counts),
        "gain_shift": float(gain_shift),
        "params": {"A": float(p[0]), "gamma": float(p[1]), "n_line": float(p[2]), "r_in_rg": r_in},
        "r_in_rg": r_in,
        "r_in_rg_stat": float(r_in_stat),
        "r_in_bound": r_in_bound,
        "fit": {"chi2": float(chi2), "dof": int(dof), "redchi2": float(redchi2), "nfev": int(res.nfev)},
    }


def _fit_powerlaw_only_rmf(
    *,
    instruments: Sequence[Dict[str, Any]],
    band_keV: Tuple[float, float],
    min_counts: int,
    gain_shift: float,
) -> Dict[str, Any]:
    """
    Forward-fold fit in channel space (continuum only):
      model(E) --(ARF×exposure)--> photons/bin --(RMF)--> counts/channel.
    """
    if least_squares is None or sp is None:
        raise RuntimeError("scipy is required for RMF folding fit")
    if not instruments:
        raise ValueError("no instruments")

    rmf0 = instruments[0].get("rmf")
    if not isinstance(rmf0, _OgipRmf):
        raise ValueError("rmf missing for instrument 0")
    detchans = int(rmf0.detchans)

    src_sum = np.zeros(detchans, dtype=float)
    net_sum = np.zeros(detchans, dtype=float)
    var_sum = np.zeros(detchans, dtype=float)
    for inst in instruments:
        rmf = inst.get("rmf")
        if not isinstance(rmf, _OgipRmf) or int(rmf.detchans) != detchans:
            raise ValueError("inconsistent RMF/detchans across instruments")
        src = np.asarray(inst.get("src_counts"), dtype=float)
        net = np.asarray(inst.get("net_counts"), dtype=float)
        var = np.asarray(inst.get("var_counts"), dtype=float)
        if src.size != detchans or net.size != detchans or var.size != detchans:
            raise ValueError("spectrum arrays must have length detchans")
        src_sum += np.clip(src, 0.0, None)
        net_sum += net
        var_sum += np.clip(var, 0.0, None)

    e_chan = np.asarray(rmf0.chan_e_mid, dtype=float) * (1.0 + float(gain_shift))
    m_band = (
        (e_chan >= float(band_keV[0]))
        & (e_chan <= float(band_keV[1]))
        & np.isfinite(e_chan)
        & np.isfinite(net_sum)
        & np.isfinite(var_sum)
    )
    if not np.any(m_band):
        raise RuntimeError("empty band after filtering")

    e_sel = e_chan[m_band]
    net_sel = net_sum[m_band]
    var_sel = np.clip(var_sum[m_band], 1.0, None)
    src_sel = src_sum[m_band]

    e_reb, y_reb, v_reb, starts = _rebin_min_counts_groups(e_sel, net_sel, var_sel, src_sel, min_counts=int(min_counts))
    sigma = np.sqrt(np.clip(v_reb, 1.0, None))

    e_model = np.asarray(rmf0.e_mid, dtype=float) * (1.0 + float(gain_shift))
    dE = np.asarray(rmf0.dE, dtype=float) * (1.0 + float(gain_shift))
    dE = np.clip(dE, 0.0, None)

    band_w = float(max(float(band_keV[1]) - float(band_keV[0]), 1e-6))
    counts_pos = float(np.sum(np.clip(y_reb, 0.0, None)))
    thr = 0.0
    m_thr = (e_model >= float(band_keV[0])) & (e_model <= float(band_keV[1])) & np.isfinite(e_model)
    for inst in instruments:
        exp = float(inst.get("exposure") or 0.0)
        arf_e = np.asarray(inst.get("arf_e_mid"), dtype=float)
        arf_a = np.asarray(inst.get("arf_area"), dtype=float)
        area = _interp_arf_area(e_model, arf_e_mid=arf_e, arf_area=arf_a)
        area_band = float(np.nanmedian(area[m_thr])) if np.any(m_thr) else float(np.nanmedian(area))
        if not math.isfinite(area_band) or area_band <= 0:
            area_band = 1.0
        thr += max(exp, 0.0) * area_band
    if not (thr > 0.0) or not math.isfinite(thr):
        thr = 1.0

    A0 = counts_pos / (thr * band_w) if counts_pos > 0 else 1e-6
    g0 = 2.0

    lo = np.asarray([0.0, 0.0], dtype=float)
    hi = np.asarray([np.inf, 5.0], dtype=float)

    def _predict(params: np.ndarray) -> np.ndarray:
        A, gamma = [float(x) for x in params]
        photons = A * (np.clip(e_model, 1e-6, None) ** (-gamma)) * dE

        pred_ch = np.zeros(detchans, dtype=float)
        for inst in instruments:
            rmf = inst["rmf"]
            exp = float(inst.get("exposure") or 0.0)
            arf_e = np.asarray(inst.get("arf_e_mid"), dtype=float)
            arf_a = np.asarray(inst.get("arf_area"), dtype=float)
            area = _interp_arf_area(e_model, arf_e_mid=arf_e, arf_area=arf_a)
            counts_bin = photons * area * max(exp, 0.0)
            pred_ch += np.asarray(rmf.rsp.dot(counts_bin)).reshape(-1)

        pred_sel = pred_ch[m_band]
        if starts.size <= 0:
            return pred_sel
        out = np.add.reduceat(pred_sel, starts)
        return np.asarray(out[: int(y_reb.size)], dtype=float)

    def _resid(params: np.ndarray) -> np.ndarray:
        return (y_reb - _predict(params)) / sigma

    x0 = np.asarray([max(A0, 0.0), g0], dtype=float)
    res = least_squares(_resid, x0=x0, bounds=(lo, hi), max_nfev=220)

    p = np.asarray(res.x, dtype=float)
    yhat = _predict(p)
    chi2 = float(np.sum(((y_reb - yhat) / sigma) ** 2))
    dof = int(max(int(y_reb.size) - int(p.size), 1))
    redchi2 = chi2 / float(dof)

    return {
        "status": "ok",
        "band_keV": [float(band_keV[0]), float(band_keV[1])],
        "min_counts": int(min_counts),
        "gain_shift": float(gain_shift),
        "params": {"A": float(p[0]), "gamma": float(p[1])},
        "fit": {"chi2": float(chi2), "dof": int(dof), "redchi2": float(redchi2), "nfev": int(res.nfev)},
        "debug": {"n_data_rebinned": int(y_reb.size), "thr_proxy": float(thr), "counts_pos": float(counts_pos)},
    }


def _fit_powerlaw_plus_diskline_rmf(
    *,
    instruments: Sequence[Dict[str, Any]],
    band_keV: Tuple[float, float],
    min_counts: int,
    gain_shift: float,
    e0_keV: float = 6.4,
    incl_deg: float = 45.0,
    emissivity_q: float = 3.0,
    r_out_rg: float = 400.0,
) -> Dict[str, Any]:
    """
    Forward-fold fit in channel space:
      model(E) --(ARF×exposure)--> photons/bin --(RMF)--> counts/channel.
    """
    if least_squares is None or sp is None:
        raise RuntimeError("scipy is required for RMF folding fit")
    if not instruments:
        raise ValueError("no instruments")

    rmf0 = instruments[0].get("rmf")
    if not isinstance(rmf0, _OgipRmf):
        raise ValueError("rmf missing for instrument 0")
    detchans = int(rmf0.detchans)

    # Combine observed spectra (MOS1+MOS2) in channel space.
    src_sum = np.zeros(detchans, dtype=float)
    net_sum = np.zeros(detchans, dtype=float)
    var_sum = np.zeros(detchans, dtype=float)
    for inst in instruments:
        rmf = inst.get("rmf")
        if not isinstance(rmf, _OgipRmf) or int(rmf.detchans) != detchans:
            raise ValueError("inconsistent RMF/detchans across instruments")
        src = np.asarray(inst.get("src_counts"), dtype=float)
        net = np.asarray(inst.get("net_counts"), dtype=float)
        var = np.asarray(inst.get("var_counts"), dtype=float)
        if src.size != detchans or net.size != detchans or var.size != detchans:
            raise ValueError("spectrum arrays must have length detchans")
        src_sum += np.clip(src, 0.0, None)
        net_sum += net
        var_sum += np.clip(var, 0.0, None)

    # Channel energies for band selection / rebin.
    e_chan = np.asarray(rmf0.chan_e_mid, dtype=float) * (1.0 + float(gain_shift))
    m_band = (
        (e_chan >= float(band_keV[0]))
        & (e_chan <= float(band_keV[1]))
        & np.isfinite(e_chan)
        & np.isfinite(net_sum)
        & np.isfinite(var_sum)
    )
    if not np.any(m_band):
        raise RuntimeError("empty band after filtering")

    e_sel = e_chan[m_band]
    net_sel = net_sum[m_band]
    var_sel = np.clip(var_sum[m_band], 1.0, None)
    src_sel = src_sum[m_band]

    # Rebin for stability and cache the group starts (reduceat).
    e_reb, y_reb, v_reb, starts = _rebin_min_counts_groups(e_sel, net_sel, var_sel, src_sel, min_counts=int(min_counts))
    sigma = np.sqrt(np.clip(v_reb, 1.0, None))

    # Model energy grid: RMF MATRIX bins (photon energy bins).
    e_model = np.asarray(rmf0.e_mid, dtype=float) * (1.0 + float(gain_shift))
    dE = np.asarray(rmf0.dE, dtype=float) * (1.0 + float(gain_shift))
    dE = np.clip(dE, 0.0, None)

    r_grid = np.unique(np.concatenate([np.linspace(2.5, 10.0, 16), np.linspace(10.0, 50.0, 21)]))
    prof_grid = _diskline_profile_grid(
        energy=e_model,
        e0_keV=float(e0_keV) * (1.0 + float(gain_shift)),
        incl_deg=float(incl_deg),
        emissivity_q=float(emissivity_q),
        r_out_rg=float(r_out_rg),
        r_in_grid=r_grid,
        sigma_instr_keV=0.0,
        n_r=220,
        n_phi=220,
    )

    # Rough throughput to initialize amplitudes.
    band_w = float(max(float(band_keV[1]) - float(band_keV[0]), 1e-6))
    counts_pos = float(np.sum(np.clip(y_reb, 0.0, None)))
    thr = 0.0
    m_thr = (e_model >= float(band_keV[0])) & (e_model <= float(band_keV[1])) & np.isfinite(e_model)
    for inst in instruments:
        exp = float(inst.get("exposure") or 0.0)
        arf_e = np.asarray(inst.get("arf_e_mid"), dtype=float)
        arf_a = np.asarray(inst.get("arf_area"), dtype=float)
        area = _interp_arf_area(e_model, arf_e_mid=arf_e, arf_area=arf_a)
        area_band = float(np.nanmedian(area[m_thr])) if np.any(m_thr) else float(np.nanmedian(area))
        if not math.isfinite(area_band) or area_band <= 0:
            area_band = 1.0
        thr += max(exp, 0.0) * area_band
    if not (thr > 0.0) or not math.isfinite(thr):
        thr = 1.0

    A0 = counts_pos / (thr * band_w) if counts_pos > 0 else 1e-6
    g0 = 2.0
    n_line0 = max(0.0, 0.1 * A0 * band_w)

    lo = np.asarray([0.0, 0.0, 0.0, 2.5], dtype=float)
    hi = np.asarray([np.inf, 5.0, np.inf, 50.0], dtype=float)

    def _predict(params: np.ndarray) -> np.ndarray:
        A, gamma, n_line, r_in = [float(x) for x in params]
        cont = A * (np.clip(e_model, 1e-6, None) ** (-gamma)) * dE
        prof = _interp_profile(float(r_in), r_grid, prof_grid)
        line = float(n_line) * prof
        photons = cont + line

        pred_ch = np.zeros(detchans, dtype=float)
        for inst in instruments:
            rmf = inst["rmf"]
            exp = float(inst.get("exposure") or 0.0)
            arf_e = np.asarray(inst.get("arf_e_mid"), dtype=float)
            arf_a = np.asarray(inst.get("arf_area"), dtype=float)
            area = _interp_arf_area(e_model, arf_e_mid=arf_e, arf_area=arf_a)
            counts_bin = photons * area * max(exp, 0.0)
            pred_ch += np.asarray(rmf.rsp.dot(counts_bin)).reshape(-1)

        pred_sel = pred_ch[m_band]
        if starts.size <= 0:
            return pred_sel
        out = np.add.reduceat(pred_sel, starts)
        return np.asarray(out[: int(y_reb.size)], dtype=float)

    def _resid(params: np.ndarray) -> np.ndarray:
        return (y_reb - _predict(params)) / sigma

    r_in_inits = [3.0, 4.5, 6.0, 10.0, 20.0, 40.0]
    best_res = None
    best_chi2 = float("inf")
    best_dof = 1
    for r_init in r_in_inits:
        x0 = np.asarray([max(A0, 0.0), g0, max(n_line0, 0.0), float(r_init)], dtype=float)
        try:
            res0 = least_squares(_resid, x0=x0, bounds=(lo, hi), max_nfev=220)
        except Exception:
            continue
        p0 = np.asarray(res0.x, dtype=float)
        yhat0 = _predict(p0)
        chi20 = float(np.sum(((y_reb - yhat0) / sigma) ** 2))
        dof0 = int(max(int(y_reb.size) - int(p0.size), 1))
        if chi20 < best_chi2:
            best_chi2 = chi20
            best_res = res0
            best_dof = dof0

    if best_res is None:
        raise RuntimeError("least_squares failed for all initializations")

    res = best_res
    p = np.asarray(res.x, dtype=float)
    yhat = _predict(p)
    chi2 = float(np.sum(((y_reb - yhat) / sigma) ** 2))
    dof = int(best_dof)
    redchi2 = chi2 / float(dof)

    r_in_stat = float("nan")
    try:
        J = np.asarray(res.jac, dtype=float)
        JTJ = J.T @ J
        cov = np.linalg.inv(JTJ) * float(redchi2)
        r_in_stat = float(np.sqrt(max(float(cov[3, 3]), 0.0)))
    except Exception:
        pass

    r_in = float(p[3])
    r_in_bound = ""
    if abs(r_in - float(lo[3])) < 1e-6:
        r_in_bound = "lower"
    elif abs(r_in - float(hi[3])) < 1e-6:
        r_in_bound = "upper"

    return {
        "status": "ok",
        "band_keV": [float(band_keV[0]), float(band_keV[1])],
        "min_counts": int(min_counts),
        "gain_shift": float(gain_shift),
        "params": {"A": float(p[0]), "gamma": float(p[1]), "n_line": float(p[2]), "r_in_rg": r_in},
        "r_in_rg": r_in,
        "r_in_rg_stat": float(r_in_stat),
        "r_in_bound": r_in_bound,
        "fit": {"chi2": float(chi2), "dof": int(dof), "redchi2": float(redchi2), "nfev": int(res.nfev)},
        "debug": {"n_data_rebinned": int(y_reb.size), "thr_proxy": float(thr), "counts_pos": float(counts_pos)},
    }


def _choose_xmm_target_spectra(
    pps_dir: Path, *, obsid: str, inst_tag: str, band_keV: Tuple[float, float]
) -> Tuple[Optional[Path], Dict[str, Any]]:
    """
    Choose the brightest PPS SRSPEC within a band by net counts (background optional).
    Returns (spectrum_path, debug).
    """
    pattern = f"P{obsid}{inst_tag}U002SRSPEC*.FTZ"
    cands = sorted(pps_dir.glob(pattern))
    debug: Dict[str, Any] = {"pattern": pattern, "n_candidates": int(len(cands)), "candidates": []}
    if not cands:
        return None, debug

    best = None
    best_score = -float("inf")
    for p in cands:
        try:
            hdr = _fits_read_spectrum_header(p)
            detchans = int(_parse_float(hdr, "DETCHANS", default=0.0) or 0)
            ch, src_counts = _fits_read_spectrum_counts(p)
            e = _energy_from_pi_channel(ch, detchans=detchans)
            m = (e >= float(band_keV[0])) & (e <= float(band_keV[1]))
            score = float(np.sum(np.clip(src_counts[m], 0.0, None)))
            debug["candidates"].append({"path": _rel(p), "score_counts": score})
            if score > best_score:
                best_score = score
                best = p
        except Exception as e:
            debug["candidates"].append({"path": _rel(p), "error": str(e)})
            continue
    debug["best"] = {"path": _rel(best) if best else None, "score_counts": best_score}
    return best, debug


def _load_xmm_net_spectrum(spec_path: Path) -> Dict[str, Any]:
    hdr = _fits_read_spectrum_header(spec_path)
    detchans = int(_parse_float(hdr, "DETCHANS", default=0.0) or 0)
    ch, src_counts = _fits_read_spectrum_counts(spec_path)
    e = _energy_from_pi_channel(ch, detchans=detchans)

    bkg_name = _parse_str(hdr, "BACKFILE", default="")
    bkg_path = spec_path.parent / bkg_name if bkg_name else None
    has_bkg = bool(bkg_path is not None and bkg_path.exists())
    bkg_counts = np.zeros_like(src_counts, dtype=float)
    scale = 0.0
    if has_bkg and bkg_path is not None:
        hdr_b = _fits_read_spectrum_header(bkg_path)
        ch_b, bkg_counts = _fits_read_spectrum_counts(bkg_path)
        if ch_b.size == ch.size and float(np.max(np.abs(ch_b - ch))) < 1e-6:
            exp_s = float(_parse_float(hdr, "EXPOSURE", default=1.0) or 1.0)
            exp_b = float(_parse_float(hdr_b, "EXPOSURE", default=1.0) or 1.0)
            bsc_s = float(_parse_float(hdr, "BACKSCAL", default=1.0) or 1.0)
            bsc_b = float(_parse_float(hdr_b, "BACKSCAL", default=1.0) or 1.0)
            asc_s = float(_parse_float(hdr, "AREASCAL", default=1.0) or 1.0)
            asc_b = float(_parse_float(hdr_b, "AREASCAL", default=1.0) or 1.0)
            scale = (exp_s / exp_b) * (bsc_s / bsc_b) * (asc_s / asc_b)
        else:
            has_bkg = False
            bkg_counts = np.zeros_like(src_counts, dtype=float)
            scale = 0.0

    net = src_counts - float(scale) * bkg_counts
    var = np.clip(src_counts, 0.0, None) + (float(scale) ** 2) * np.clip(bkg_counts, 0.0, None)
    return {
        "spec_path": _rel(spec_path),
        "back_path": _rel(bkg_path) if has_bkg and bkg_path is not None else "",
        "has_background": bool(has_bkg),
        "background_scale": float(scale),
        "detchans": int(detchans),
        "channel": np.asarray(ch, dtype=float),
        "energy_keV": e,
        "src_counts": src_counts,
        "bkg_counts": bkg_counts,
        "net_counts": net,
        "var_counts": var,
        "header": {
            "TELESCOP": _parse_str(hdr, "TELESCOP"),
            "INSTRUME": _parse_str(hdr, "INSTRUME"),
            "FILTER": _parse_str(hdr, "FILTER"),
            "RESPFILE": _parse_str(hdr, "RESPFILE"),
            "ANCRFILE": _parse_str(hdr, "ANCRFILE"),
            "BACKFILE": _parse_str(hdr, "BACKFILE"),
            "EXPOSURE": float(_parse_float(hdr, "EXPOSURE")),
            "BACKSCAL": float(_parse_float(hdr, "BACKSCAL")),
        },
    }


def _fits_read_arf(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    opener = gzip.open if path.name.lower().endswith(".ftz") or path.name.lower().endswith(".gz") else Path.open
    with opener(path, "rb") as f:  # type: ignore[arg-type]
        layout = read_first_bintable_layout(f)
        cols = read_bintable_columns(f, layout=layout, columns=["ENERG_LO", "ENERG_HI", "SPECRESP"])
    lo = np.asarray(cols["ENERG_LO"], dtype=float)
    hi = np.asarray(cols["ENERG_HI"], dtype=float)
    a = np.asarray(cols["SPECRESP"], dtype=float)
    m = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(a) & (hi >= lo)
    lo = lo[m]
    hi = hi[m]
    a = a[m]
    e_mid = 0.5 * (lo + hi)
    return np.asarray(e_mid, dtype=float), np.asarray(a, dtype=float)


@dataclass(frozen=True)
class _OgipRmf:
    rmf_path: str
    detchans: int
    chan: np.ndarray  # (detchans,)
    chan_e_min: np.ndarray  # (detchans,)
    chan_e_max: np.ndarray  # (detchans,)
    chan_e_mid: np.ndarray  # (detchans,)
    e_lo: np.ndarray  # (n_energy_bins,)
    e_hi: np.ndarray  # (n_energy_bins,)
    e_mid: np.ndarray  # (n_energy_bins,)
    dE: np.ndarray  # (n_energy_bins,)
    rsp: object  # scipy.sparse.csr_matrix with shape=(detchans, n_energy_bins)
    nnz: int
    header: Dict[str, str]


_TFORM_CODE_RE = re.compile(r"^\s*(?P<rep>\d*)(?P<code>[A-Z]|[PQ][A-Z])\s*$")


def _parse_int(kv: Dict[str, str], key: str, default: int = 0) -> int:
    v = _parse_float(kv, key, default=float(default))
    if not math.isfinite(float(v)):
        return int(default)
    return int(float(v))


def _fits_hdu_data_size_bytes(kv: Dict[str, str]) -> int:
    xt = _parse_str(kv, "XTENSION", default="").upper()
    if "BINTABLE" in xt:
        naxis1 = _parse_int(kv, "NAXIS1", default=0)
        naxis2 = _parse_int(kv, "NAXIS2", default=0)
        pcount = _parse_int(kv, "PCOUNT", default=0)
        return int(naxis1) * int(naxis2) + int(pcount)

    # Primary HDU or IMAGE extension (not expected for RMF, but keep minimal support).
    naxis = _parse_int(kv, "NAXIS", default=0)
    if naxis <= 0:
        return 0
    bitpix = abs(_parse_int(kv, "BITPIX", default=0))
    if bitpix <= 0:
        return 0
    bytes_per = int(bitpix) // 8
    if bytes_per <= 0:
        return 0
    n = 1
    for i in range(1, int(naxis) + 1):
        n *= max(_parse_int(kv, f"NAXIS{i}", default=0), 0)
    return int(n) * int(bytes_per)


def _fits_skip_padded(stream: io.BytesIO, n_bytes: int) -> None:
    n = int(max(n_bytes, 0))
    pad = (-n) % 2880
    stream.seek(stream.tell() + n + pad)


def _tform_spec(tform: str) -> Dict[str, Any]:
    s = str(tform or "").strip()
    if not s:
        raise ValueError("empty TFORM")
    # Strip maxlen, e.g. "1PI(43)" -> "1PI"
    s = s.split("(", 1)[0].strip()
    m = _TFORM_CODE_RE.match(s)
    if not m:
        raise ValueError(f"unsupported TFORM: {tform!r}")
    rep = int(m.group("rep") or "1")
    code = str(m.group("code") or "").strip().upper()
    if rep < 1:
        raise ValueError(f"invalid TFORM repeat: {tform!r}")
    if len(code) == 2 and code[0] in {"P", "Q"}:
        kind = code[0]
        base = code[1]
        return {
            "repeat": rep,
            "is_var": True,
            "var_kind": kind,
            "base": base,
            "row_width": rep * (8 if kind == "P" else 16),
        }
    base = code
    if base == "A":
        width = rep
    elif base == "I":
        width = 2 * rep
    elif base == "J":
        width = 4 * rep
    elif base == "K":
        width = 8 * rep
    elif base == "E":
        width = 4 * rep
    elif base == "D":
        width = 8 * rep
    elif base == "B":
        width = 1 * rep
    elif base == "L":
        width = 1 * rep
    else:
        raise ValueError(f"unsupported TFORM base code: {base!r} (tform={tform!r})")
    return {"repeat": rep, "is_var": False, "base": base, "row_width": int(width)}


def _tform_numpy_dtype(base: str, *, repeat: int) -> np.dtype:
    b = str(base or "").strip().upper()
    rep = int(repeat)
    if b == "A":
        return np.dtype(f"S{rep}")
    if b == "I":
        return np.dtype(">i2")
    if b == "J":
        return np.dtype(">i4")
    if b == "K":
        return np.dtype(">i8")
    if b == "E":
        return np.dtype(">f4")
    if b == "D":
        return np.dtype(">f8")
    if b == "B":
        return np.dtype("u1")
    if b == "L":
        return np.dtype("S1")
    raise ValueError(f"unsupported base dtype: {base!r}")


def _fits_parse_bintable_layout(kv: Dict[str, str], *, row_bytes: int) -> Tuple[Dict[str, int], Dict[str, str], Dict[str, Dict[str, Any]]]:
    tfields = _parse_int(kv, "TFIELDS", default=0)
    if tfields <= 0:
        raise ValueError("TFIELDS missing/invalid")
    offsets: Dict[str, int] = {}
    tforms: Dict[str, str] = {}
    specs: Dict[str, Dict[str, Any]] = {}
    off = 0
    for i in range(1, int(tfields) + 1):
        name = _parse_str(kv, f"TTYPE{i}", default="").strip()
        tform = _parse_str(kv, f"TFORM{i}", default="")
        if not name or not tform:
            raise ValueError(f"missing TTYPE/TFORM for field {i}")
        spec = _tform_spec(tform)
        offsets[name] = int(off)
        tforms[name] = str(tform)
        specs[name] = spec
        off += int(spec["row_width"])
    if int(off) != int(row_bytes):
        raise ValueError(f"row size mismatch: sum(TFORM widths)={off} != NAXIS1={row_bytes}")
    return offsets, tforms, specs


def _fits_find_bintable(buf: bytes, *, extname: str) -> Tuple[Dict[str, str], int, int, int, int, int, Dict[str, int], Dict[str, str], Dict[str, Dict[str, Any]]]:
    s = io.BytesIO(buf)

    # primary HDU
    hdr0 = _read_header_blocks(s)
    kv0 = _parse_header_kv(hdr0)
    _fits_skip_padded(s, _fits_hdu_data_size_bytes(kv0))

    want = str(extname or "").strip().upper()
    while True:
        try:
            hdr = _read_header_blocks(s)
        except EOFError as e:
            raise KeyError(f"FITS extension not found: {want}") from e
        kv = _parse_header_kv(hdr)
        data_offset = int(s.tell())
        xt = _parse_str(kv, "XTENSION", default="").upper()
        if "BINTABLE" in xt:
            name = _parse_str(kv, "EXTNAME", default="").strip().upper()
            row_bytes = _parse_int(kv, "NAXIS1", default=0)
            n_rows = _parse_int(kv, "NAXIS2", default=0)
            pcount = _parse_int(kv, "PCOUNT", default=0)
            theap_default = int(row_bytes) * int(n_rows)
            theap = _parse_int(kv, "THEAP", default=theap_default)
            offsets, tforms, specs = _fits_parse_bintable_layout(kv, row_bytes=row_bytes)
            if name == want:
                return kv, data_offset, row_bytes, n_rows, pcount, theap, offsets, tforms, specs

        _fits_skip_padded(s, _fits_hdu_data_size_bytes(kv))


def _rmf_load_ogip(path: Path) -> _OgipRmf:
    if sp is None:
        raise RuntimeError("scipy.sparse is required for RMF folding")
    buf = path.read_bytes()

    kv_e, off_e, row_e, n_e, _p_e, _t_e, offsets_e, tforms_e, specs_e = _fits_find_bintable(buf, extname="EBOUNDS")
    kv_m, off_m, row_m, n_m, p_m, theap_m, offsets_m, tforms_m, specs_m = _fits_find_bintable(buf, extname="MATRIX")

    # EBOUNDS: CHANNEL, E_MIN, E_MAX
    need_e = ["CHANNEL", "E_MIN", "E_MAX"]
    for c in need_e:
        if c not in offsets_e:
            raise KeyError(f"EBOUNDS column not found: {c}")
    main_e = memoryview(buf)[int(off_e) : int(off_e) + int(row_e) * int(n_e)]
    dt_e = np.dtype(
        {
            "names": ["CHANNEL", "E_MIN", "E_MAX"],
            "formats": [
                _tform_numpy_dtype(specs_e["CHANNEL"]["base"], repeat=int(specs_e["CHANNEL"]["repeat"])),
                _tform_numpy_dtype(specs_e["E_MIN"]["base"], repeat=int(specs_e["E_MIN"]["repeat"])),
                _tform_numpy_dtype(specs_e["E_MAX"]["base"], repeat=int(specs_e["E_MAX"]["repeat"])),
            ],
            "offsets": [int(offsets_e["CHANNEL"]), int(offsets_e["E_MIN"]), int(offsets_e["E_MAX"])],
            "itemsize": int(row_e),
        }
    )
    arr_e = np.frombuffer(main_e, dtype=dt_e, count=int(n_e))
    chan = np.asarray(arr_e["CHANNEL"], dtype=np.int64)
    e_min = np.asarray(arr_e["E_MIN"], dtype=np.float64)
    e_max = np.asarray(arr_e["E_MAX"], dtype=np.float64)
    e_mid_chan = 0.5 * (e_min + e_max)

    detchans = _parse_int(kv_e, "DETCHANS", default=int(chan.max() + 1 if chan.size else 0))
    if detchans <= 0:
        detchans = int(chan.size)

    # MATRIX: ENERG_LO/HI, N_GRP, F_CHAN, N_CHAN, MATRIX
    need_m = ["ENERG_LO", "ENERG_HI", "N_GRP", "F_CHAN", "N_CHAN", "MATRIX"]
    for c in need_m:
        if c not in offsets_m:
            raise KeyError(f"MATRIX column not found: {c}")
    if any(bool(specs_m[c].get("is_var")) for c in ["ENERG_LO", "ENERG_HI", "N_GRP"]):
        raise ValueError("unexpected varlen scalar in MATRIX")
    f_is_var = bool(specs_m["F_CHAN"].get("is_var"))
    n_is_var = bool(specs_m["N_CHAN"].get("is_var"))
    m_is_var = bool(specs_m["MATRIX"].get("is_var"))
    if not m_is_var or str(specs_m["MATRIX"].get("var_kind")) != "P":
        raise ValueError("MATRIX must be a 'P' varlen array for RMF")
    if f_is_var != n_is_var:
        raise ValueError("mixed scalar/varlen layout for F_CHAN vs N_CHAN")
    if f_is_var and any(str(specs_m[c].get("var_kind")) != "P" for c in ["F_CHAN", "N_CHAN"]):
        raise ValueError("only 'P' varlen arrays are supported for F_CHAN/N_CHAN")
    if (not f_is_var) and (
        int(specs_m["F_CHAN"].get("repeat") or 1) != 1 or int(specs_m["N_CHAN"].get("repeat") or 1) != 1
    ):
        raise ValueError("fixed-length array F_CHAN/N_CHAN (repeat>1) is not supported")

    main_m = memoryview(buf)[int(off_m) : int(off_m) + int(row_m) * int(n_m)]
    heap_start = int(off_m) + int(theap_m)
    heap_end = int(off_m) + int(row_m) * int(n_m) + int(p_m)
    heap = memoryview(buf)[heap_start:heap_end]

    names_m: List[str] = ["ENERG_LO", "ENERG_HI", "N_GRP"]
    fmts_m: List[object] = [
        _tform_numpy_dtype(specs_m["ENERG_LO"]["base"], repeat=int(specs_m["ENERG_LO"]["repeat"])),
        _tform_numpy_dtype(specs_m["ENERG_HI"]["base"], repeat=int(specs_m["ENERG_HI"]["repeat"])),
        _tform_numpy_dtype(specs_m["N_GRP"]["base"], repeat=int(specs_m["N_GRP"]["repeat"])),
    ]
    offs_m: List[int] = [int(offsets_m["ENERG_LO"]), int(offsets_m["ENERG_HI"]), int(offsets_m["N_GRP"])]
    if f_is_var:
        names_m.extend(["F_DESC", "N_DESC"])
        fmts_m.extend([(np.dtype(">u4"), (2,)), (np.dtype(">u4"), (2,))])
        offs_m.extend([int(offsets_m["F_CHAN"]), int(offsets_m["N_CHAN"])])
    else:
        names_m.extend(["F_SCALAR", "N_SCALAR"])
        fmts_m.extend(
            [
                _tform_numpy_dtype(specs_m["F_CHAN"]["base"], repeat=1),
                _tform_numpy_dtype(specs_m["N_CHAN"]["base"], repeat=1),
            ]
        )
        offs_m.extend([int(offsets_m["F_CHAN"]), int(offsets_m["N_CHAN"])])
    names_m.append("M_DESC")
    fmts_m.append((np.dtype(">u4"), (2,)))
    offs_m.append(int(offsets_m["MATRIX"]))

    dt_m = np.dtype({"names": names_m, "formats": fmts_m, "offsets": offs_m, "itemsize": int(row_m)})
    arr_m = np.frombuffer(main_m, dtype=dt_m, count=int(n_m))
    e_lo = np.asarray(arr_m["ENERG_LO"], dtype=np.float64)
    e_hi = np.asarray(arr_m["ENERG_HI"], dtype=np.float64)
    n_grp = np.asarray(arr_m["N_GRP"], dtype=np.int64)
    m_desc = np.asarray(arr_m["M_DESC"], dtype=np.int64)
    f_desc = np.asarray(arr_m["F_DESC"], dtype=np.int64) if f_is_var else None
    n_desc = np.asarray(arr_m["N_DESC"], dtype=np.int64) if f_is_var else None
    f_scalar = np.asarray(arr_m["F_SCALAR"], dtype=np.int64) if not f_is_var else None
    n_scalar = np.asarray(arr_m["N_SCALAR"], dtype=np.int64) if not f_is_var else None

    e_mid = 0.5 * (e_lo + e_hi)
    dE = np.clip(e_hi - e_lo, 0.0, None)

    dt_r = _tform_numpy_dtype(str(specs_m["MATRIX"]["base"]), repeat=1)
    r_size = int(dt_r.itemsize)
    dt_f = _tform_numpy_dtype(str(specs_m["F_CHAN"]["base"]), repeat=1)
    dt_n = _tform_numpy_dtype(str(specs_m["N_CHAN"]["base"]), repeat=1)
    f_size = int(dt_f.itemsize)
    n_size = int(dt_n.itemsize)

    rows_list: List[np.ndarray] = []
    cols_list: List[np.ndarray] = []
    data_list: List[np.ndarray] = []

    for i in range(int(n_m)):
        ng = int(n_grp[i]) if i < n_grp.size else 0
        if ng <= 0:
            continue

        nr, off_r = int(m_desc[i, 0]), int(m_desc[i, 1])
        if nr <= 0:
            continue

        r0 = int(off_r)
        if r0 < 0:
            continue
        if r0 + nr * r_size > heap.nbytes:
            continue

        resp = np.frombuffer(heap[r0 : r0 + nr * r_size], dtype=dt_r, count=nr).astype(np.float64, copy=False)

        if f_is_var:
            assert f_desc is not None and n_desc is not None
            nf, off_f = int(f_desc[i, 0]), int(f_desc[i, 1])
            nn, off_n = int(n_desc[i, 0]), int(n_desc[i, 1])
            if nf <= 0 or nn <= 0:
                continue
            f0 = int(off_f)
            n0 = int(off_n)
            if f0 < 0 or n0 < 0:
                continue
            if f0 + nf * f_size > heap.nbytes or n0 + nn * n_size > heap.nbytes:
                continue
            f_chan = np.frombuffer(heap[f0 : f0 + nf * f_size], dtype=dt_f, count=nf).astype(np.int64, copy=False)
            n_chan = np.frombuffer(heap[n0 : n0 + nn * n_size], dtype=dt_n, count=nn).astype(np.int64, copy=False)
            if f_chan.size < ng or n_chan.size < ng:
                continue

            pos = 0
            for g in range(int(ng)):
                start = int(f_chan[g])
                length = int(n_chan[g])
                if length <= 0:
                    continue
                if pos + length > resp.size:
                    break
                if start < 0:
                    pos += length
                    continue
                end = start + length
                if start >= detchans:
                    pos += length
                    continue
                if end > detchans:
                    length = int(max(detchans - start, 0))
                    end = start + length
                if length <= 0:
                    pos += int(n_chan[g])
                    continue
                vals = resp[pos : pos + length]
                idx = np.arange(int(start), int(end), dtype=np.int64)
                rows_list.append(idx)
                cols_list.append(np.full(int(idx.size), int(i), dtype=np.int64))
                data_list.append(np.asarray(vals, dtype=np.float64))
                pos += int(n_chan[g])
        else:
            assert f_scalar is not None and n_scalar is not None
            # Some canned RMFs use N_GRP=1 with scalar F_CHAN/N_CHAN.
            if int(ng) != 1:
                continue
            start = int(f_scalar[i])
            length = int(n_scalar[i])
            if length <= 0 or start < 0:
                continue
            length = int(min(length, resp.size))
            end = start + length
            if start >= detchans:
                continue
            if end > detchans:
                length = int(max(detchans - start, 0))
                end = start + length
            if length <= 0:
                continue
            vals = resp[:length]
            idx = np.arange(int(start), int(end), dtype=np.int64)
            rows_list.append(idx)
            cols_list.append(np.full(int(idx.size), int(i), dtype=np.int64))
            data_list.append(np.asarray(vals, dtype=np.float64))

    if rows_list:
        rows = np.concatenate(rows_list).astype(np.int64, copy=False)
        cols = np.concatenate(cols_list).astype(np.int64, copy=False)
        data = np.concatenate(data_list).astype(np.float64, copy=False)
    else:
        rows = np.zeros(0, dtype=np.int64)
        cols = np.zeros(0, dtype=np.int64)
        data = np.zeros(0, dtype=np.float64)

    mat = sp.csr_matrix((data, (rows, cols)), shape=(int(detchans), int(n_m)))

    return _OgipRmf(
        rmf_path=_rel(path),
        detchans=int(detchans),
        chan=np.asarray(chan, dtype=np.int64),
        chan_e_min=np.asarray(e_min, dtype=np.float64),
        chan_e_max=np.asarray(e_max, dtype=np.float64),
        chan_e_mid=np.asarray(e_mid_chan, dtype=np.float64),
        e_lo=np.asarray(e_lo, dtype=np.float64),
        e_hi=np.asarray(e_hi, dtype=np.float64),
        e_mid=np.asarray(e_mid, dtype=np.float64),
        dE=np.asarray(dE, dtype=np.float64),
        rsp=mat,
        nnz=int(mat.nnz),
        header={"EBOUNDS_DETCHANS": str(detchans), "MATRIX_NROWS": str(n_m)},
    )


_RMF_CACHE: Dict[str, _OgipRmf] = {}


def _rmf_cached(path: Path) -> _OgipRmf:
    key = str(path.resolve())
    if key in _RMF_CACHE:
        return _RMF_CACHE[key]
    rmf = _rmf_load_ogip(path)
    _RMF_CACHE[key] = rmf
    return rmf


def _xmm_local_rmf_path(respfile: str) -> Optional[Path]:
    name = str(respfile or "").strip().strip("'")
    if not name:
        return None
    low = name.lower()
    base = _ROOT / "data" / "xrism" / "xmm_epic_responses"
    if low.startswith(("m1_", "m2_", "m11_", "m21_")):
        return base / "MOS" / name
    if low.startswith(("epn_", "pn_", "pnu_", "p11_", "p21_")):
        return base / "PN" / name
    return None


def _rebin_spectrum_to_detchans(
    channel: np.ndarray,
    src_counts: np.ndarray,
    net_counts: np.ndarray,
    var_counts: np.ndarray,
    *,
    detchans: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Some XMM products use finer PI binning than the canned RMF's EBOUNDS.
    Example: spectrum has 2400 PI bins (5 eV), RMF has 800 PI bins (15 eV).

    This helper rebins counts by an integer factor to match the RMF detchans.
    """
    ch = np.asarray(channel, dtype=float)
    src = np.asarray(src_counts, dtype=float)
    net = np.asarray(net_counts, dtype=float)
    var = np.asarray(var_counts, dtype=float)
    n = int(src.size)
    d = int(detchans)
    if n != int(net.size) or n != int(var.size) or n != int(ch.size):
        raise ValueError("shape mismatch in spectrum arrays")
    if d <= 0:
        raise ValueError("invalid detchans")
    if n == d:
        return src, net, var, {"status": "no_rebin", "factor": 1}
    if n % d != 0:
        raise ValueError(f"cannot rebin spectrum: n={n} not divisible by detchans={d}")
    factor = int(n // d)

    order = np.argsort(ch)
    ch2 = ch[order]
    if not np.all(np.isfinite(ch2)):
        raise ValueError("non-finite channel values")
    if float(np.max(np.abs(ch2 - np.arange(n, dtype=float)))) > 1e-6:
        raise ValueError("unexpected CHANNEL numbering (expected 0..N-1)")

    src2 = src[order].reshape(d, factor).sum(axis=1)
    net2 = net[order].reshape(d, factor).sum(axis=1)
    var2 = var[order].reshape(d, factor).sum(axis=1)
    return (
        np.asarray(src2, dtype=float),
        np.asarray(net2, dtype=float),
        np.asarray(var2, dtype=float),
        {"status": "rebinned", "factor": int(factor), "n_in": int(n), "n_out": int(d)},
    )


def _interp_arf_area(energy_keV: np.ndarray, *, arf_e_mid: np.ndarray, arf_area: np.ndarray) -> np.ndarray:
    e = np.asarray(energy_keV, dtype=float)
    x = np.asarray(arf_e_mid, dtype=float)
    y = np.asarray(arf_area, dtype=float)
    if x.size < 2 or y.size != x.size:
        return np.ones_like(e, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    return np.asarray(np.interp(e, x, y, left=0.0, right=0.0), dtype=float)


def _fit_xmm_obsid_diskline(
    cache_root: Path, *, obsid: str, rev: str, out_dir: Path
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    pps_dir = cache_root / "xmm" / rev / obsid / "PPS"
    debug: Dict[str, Any] = {"obsid": obsid, "rev": rev, "pps_dir": _rel(pps_dir)}
    if not pps_dir.exists():
        return None, {"status": "missing_pps", **debug}

    band_default = (3.0, 10.0)
    m1_path, dbg_m1 = _choose_xmm_target_spectra(pps_dir, obsid=obsid, inst_tag="M1", band_keV=band_default)
    m2_path, dbg_m2 = _choose_xmm_target_spectra(pps_dir, obsid=obsid, inst_tag="M2", band_keV=band_default)
    debug["select_m1"] = dbg_m1
    debug["select_m2"] = dbg_m2
    if m1_path is None and m2_path is None:
        return None, {"status": "no_spectra", **debug}

    specs: List[Dict[str, Any]] = []
    for p in [m1_path, m2_path]:
        if p is None:
            continue
        specs.append(_load_xmm_net_spectrum(p))

    # Prefer RMF/ARF forward-folding if canned RMF is available locally.
    rmf_instruments: List[Dict[str, Any]] = []
    rmf_sources: List[Dict[str, Any]] = []
    rmf_errors: List[Dict[str, Any]] = []
    for sp in specs:
        try:
            hdr = dict(sp.get("header", {}) or {})
            spec_rel = str(sp.get("spec_path") or "")
            spec_path = (_ROOT / Path(spec_rel)).resolve() if spec_rel else None
            respfile = str(hdr.get("RESPFILE") or "").strip()
            rmf_path = _xmm_local_rmf_path(respfile)
            if rmf_path is None or not rmf_path.exists():
                rmf_errors.append(
                    {
                        "spec": spec_rel,
                        "respfile": respfile,
                        "rmf_local": _rel(rmf_path) if rmf_path else "",
                        "status": "missing_rmf",
                    }
                )
                continue
            rmf = _rmf_cached(rmf_path)

            ancr = str(hdr.get("ANCRFILE") or "").strip()
            arf_path = (spec_path.parent / ancr) if spec_path is not None and ancr else None
            if arf_path is None or not arf_path.exists():
                rmf_errors.append(
                    {
                        "spec": spec_rel,
                        "respfile": respfile,
                        "rmf_local": _rel(rmf_path),
                        "arf": _rel(arf_path) if arf_path else "",
                        "status": "missing_arf",
                    }
                )
                continue
            arf_e, arf_a = _fits_read_arf(arf_path)

            exp = float(hdr.get("EXPOSURE", 0.0) or 0.0)
            src0 = np.asarray(sp.get("src_counts"), dtype=float)
            net0 = np.asarray(sp.get("net_counts"), dtype=float)
            var0 = np.asarray(sp.get("var_counts"), dtype=float)
            ch0 = np.asarray(sp.get("channel"), dtype=float)
            try:
                src_r, net_r, var_r, rebin_dbg = _rebin_spectrum_to_detchans(
                    ch0,
                    src0,
                    net0,
                    var0,
                    detchans=int(rmf.detchans),
                )
            except Exception as ex_rebin:
                rmf_errors.append(
                    {
                        "spec": spec_rel,
                        "respfile": respfile,
                        "rmf_local": _rel(rmf_path),
                        "rmf_detchans": int(rmf.detchans),
                        "spec_n_channels": int(src0.size),
                        "status": "spectrum_rebin_error",
                        "error": str(ex_rebin),
                    }
                )
                continue
            rmf_instruments.append(
                {
                    "rmf": rmf,
                    "exposure": float(exp),
                    "arf_e_mid": np.asarray(arf_e, dtype=float),
                    "arf_area": np.asarray(arf_a, dtype=float),
                    "src_counts": np.asarray(src_r, dtype=float),
                    "net_counts": np.asarray(net_r, dtype=float),
                    "var_counts": np.asarray(var_r, dtype=float),
                }
            )
            rmf_sources.append(
                {
                    "spec": spec_rel,
                    "bkg": str(sp.get("back_path") or ""),
                    "has_bkg": bool(sp.get("has_background")),
                    "bkg_scale": float(sp.get("background_scale") or 0.0),
                    "respfile": respfile,
                    "rmf_local": rmf.rmf_path,
                    "rmf_detchans": int(rmf.detchans),
                    "spectrum_rebin": rebin_dbg,
                    "rmf_matrix_nnz": int(rmf.nnz),
                    "arf": _rel(arf_path),
                    "exposure": float(exp),
                }
            )
        except Exception as ex:
            rmf_errors.append({"spec": str(sp.get("spec_path") or ""), "status": "rmf_setup_error", "error": str(ex)})

    debug["rmf_sources"] = rmf_sources
    debug["rmf_errors"] = rmf_errors

    if rmf_instruments:
        try:
            fit0_rmf = _fit_powerlaw_plus_diskline_rmf(
                instruments=rmf_instruments,
                band_keV=band_default,
                min_counts=50,
                gain_shift=0.0,
            )

            # Broad line detection check (RMF-folded): continuum-only vs continuum+diskline.
            delta_chi2_threshold = 9.0
            delta_chi2 = float("nan")
            line_detected = False
            cont_fit_rmf: Dict[str, Any] = {}
            try:
                cont_fit_rmf = _fit_powerlaw_only_rmf(
                    instruments=rmf_instruments,
                    band_keV=band_default,
                    min_counts=50,
                    gain_shift=0.0,
                )
                chi2_cont = float(cont_fit_rmf.get("fit", {}).get("chi2", float("nan")))
                chi2_full = float(fit0_rmf.get("fit", {}).get("chi2", float("nan")))
                if np.isfinite(chi2_cont) and np.isfinite(chi2_full):
                    delta_chi2 = float(chi2_cont - chi2_full)
                    line_detected = bool(delta_chi2 >= float(delta_chi2_threshold))
            except Exception as ex:
                debug["line_detection_error"] = str(ex)

            # Simple S/N proxy in the fitted band (channel space).
            net_counts_band_pos = float("nan")
            var_counts_band = float("nan")
            snr_band_proxy = float("nan")
            try:
                rmf0 = rmf_instruments[0]["rmf"]
                e_chan = np.asarray(rmf0.chan_e_mid, dtype=float)
                m_band = (e_chan >= float(band_default[0])) & (e_chan <= float(band_default[1])) & np.isfinite(e_chan)
                net_sum = np.zeros(int(rmf0.detchans), dtype=float)
                var_sum = np.zeros(int(rmf0.detchans), dtype=float)
                for inst in rmf_instruments:
                    net_sum += np.asarray(inst.get("net_counts"), dtype=float)
                    var_sum += np.asarray(inst.get("var_counts"), dtype=float)
                if np.any(m_band):
                    net_counts_band_pos = float(np.sum(np.clip(net_sum[m_band], 0.0, None)))
                    var_counts_band = float(np.sum(np.clip(var_sum[m_band], 0.0, None)))
                    snr_band_proxy = float(net_counts_band_pos / math.sqrt(var_counts_band)) if var_counts_band > 0 else float("nan")
            except Exception:
                pass

            r_in_bound0 = str(fit0_rmf.get("r_in_bound") or "")
            r_in_stat0 = float(fit0_rmf.get("r_in_rg_stat", float("nan")))
            isco_constrained_proxy = bool(line_detected and (not r_in_bound0) and np.isfinite(r_in_stat0))
            proxy_quality = "ok"
            if not line_detected:
                proxy_quality = "no_broad_line"
            elif r_in_bound0:
                proxy_quality = f"r_in_bound:{r_in_bound0}"
            elif not np.isfinite(r_in_stat0):
                proxy_quality = "no_cov"

            # Systematics sweep (band / gain / rebin) in RMF-folded space.
            sys_band = [(2.5, 10.0), (3.0, 10.0), (4.0, 10.0)]
            sys_gain = [-1e-3, 0.0, 1e-3]
            sys_rebin = [25, 50, 100]

            variants_rmf: List[Dict[str, Any]] = []
            for b in sys_band:
                for g in sys_gain:
                    for mc in sys_rebin:
                        try:
                            r = _fit_powerlaw_plus_diskline_rmf(
                                instruments=rmf_instruments,
                                band_keV=b,
                                min_counts=int(mc),
                                gain_shift=float(g),
                            )
                        except Exception as ex:
                            r = {
                                "status": "fail",
                                "band_keV": [float(b[0]), float(b[1])],
                                "min_counts": int(mc),
                                "gain_shift": float(g),
                                "error": str(ex),
                            }
                        variants_rmf.append(r)

            r0 = float(fit0_rmf.get("r_in_rg", float("nan")))

            def _sys_component(selector: Iterable[Dict[str, Any]]) -> float:
                vals = []
                for it in selector:
                    if it.get("status") != "ok":
                        continue
                    vals.append(float(it.get("r_in_rg", float("nan"))))
                vals = [v for v in vals if np.isfinite(v)]
                if not vals or not np.isfinite(r0):
                    return float("nan")
                return float(max(abs(v - r0) for v in vals))

            sys_band_val = _sys_component([v for v in variants_rmf if v.get("min_counts") == 50 and v.get("gain_shift") == 0.0])
            sys_gain_val = _sys_component([v for v in variants_rmf if v.get("min_counts") == 50 and v.get("band_keV") == [3.0, 10.0]])
            sys_rebin_val = _sys_component([v for v in variants_rmf if v.get("band_keV") == [3.0, 10.0] and v.get("gain_shift") == 0.0])
            sys_terms = [x for x in [sys_band_val, sys_gain_val, sys_rebin_val] if np.isfinite(x)]
            sys_total = float(math.sqrt(float(np.sum(np.square(sys_terms))))) if sys_terms else float("nan")

            out_detail = out_dir / f"xmm_{obsid}__fek_broad_line_rmf_diskline.json"
            detail = {
                "generated_utc": _utc_now(),
                "obsid": obsid,
                "rev": rev,
                "method": "powerlaw + diskline_proxy (RMF/ARF folded; canned RMF from HEASARC CALDB + PPS ARF)",
                "inputs": {"band_default_keV": list(band_default), "min_counts_default": 50},
                "selected": {"m1": specs[0]["spec_path"] if specs else "", "m2": specs[1]["spec_path"] if len(specs) > 1 else ""},
                "rmf_sources": rmf_sources,
                "line_detection": {
                    "delta_chi2": float(delta_chi2),
                    "threshold": float(delta_chi2_threshold),
                    "detected": bool(line_detected),
                    "isco_constrained_proxy": bool(isco_constrained_proxy),
                    "proxy_quality": str(proxy_quality),
                    "net_counts_band_pos": float(net_counts_band_pos),
                    "var_counts_band": float(var_counts_band),
                    "snr_band_proxy": float(snr_band_proxy),
                    "continuum_fit": cont_fit_rmf or {},
                },
                "fit_default": fit0_rmf,
                "systematics": {
                    "components": {"band": sys_band_val, "gain": sys_gain_val, "rebin": sys_rebin_val},
                    "sys_total": sys_total,
                    "variants": variants_rmf,
                },
                "debug": debug,
            }
            _write_json(out_detail, detail)

            out = {
                "status": "ok",
                "method_tag": "rmf_diskline_v1",
                "r_in_rg": r0,
                "r_in_rg_stat": float(fit0_rmf.get("r_in_rg_stat", float("nan"))),
                "r_in_rg_sys": sys_total,
                "r_in_bound": str(fit0_rmf.get("r_in_bound") or ""),
                "proxy_quality": str(proxy_quality),
                "proxy_line_detected": bool(line_detected),
                "proxy_delta_chi2": float(delta_chi2),
                "proxy_net_counts_band_pos": float(net_counts_band_pos),
                "proxy_snr_band": float(snr_band_proxy),
                "proxy_isco_constrained": bool(isco_constrained_proxy),
                "proxy_region_mode": "",
                "detail_json": _rel(out_detail),
            }
            return out, debug
        except Exception as ex:
            debug["rmf_fit_error"] = str(ex)

    # Combine by summing counts/vars (energy grids should match)
    e = np.asarray(specs[0]["energy_keV"], dtype=float)
    net = np.zeros_like(e, dtype=float)
    var = np.zeros_like(e, dtype=float)
    src = np.zeros_like(e, dtype=float)
    for s in specs:
        if np.asarray(s["energy_keV"]).shape != e.shape or float(np.max(np.abs(np.asarray(s["energy_keV"]) - e))) > 1e-9:
            return None, {"status": "energy_grid_mismatch", **debug}
        net += np.asarray(s["net_counts"], dtype=float)
        var += np.asarray(s["var_counts"], dtype=float)
        src += np.asarray(s["src_counts"], dtype=float)

    # Default fit
    w_sum = np.zeros_like(e, dtype=float)
    weight_sources: List[Dict[str, Any]] = []
    for sp in specs:
        try:
            exp = float(sp.get("header", {}).get("EXPOSURE", 0.0) or 0.0)
            ancr = str(sp.get("header", {}).get("ANCRFILE") or "").strip()
            respfile = str(sp.get("header", {}).get("RESPFILE") or "").strip()
            arf_path = (pps_dir / ancr) if ancr else None
            if arf_path is not None and arf_path.exists():
                arf_e, arf_a = _fits_read_arf(arf_path)
                area = _interp_arf_area(e, arf_e_mid=arf_e, arf_area=arf_a)
                w = np.clip(area, 0.0, None) * max(exp, 0.0)
                w_sum += w
                weight_sources.append(
                    {
                        "spec": sp.get("spec_path", ""),
                        "arf": _rel(arf_path),
                        "respfile": respfile,
                        "exposure": exp,
                        "area_min": float(np.nanmin(area)) if np.any(np.isfinite(area)) else float("nan"),
                        "area_max": float(np.nanmax(area)) if np.any(np.isfinite(area)) else float("nan"),
                    }
                )
            else:
                w_sum += max(exp, 0.0)
                weight_sources.append({"spec": sp.get("spec_path", ""), "arf": "", "respfile": respfile, "exposure": exp})
        except Exception as ex:
            weight_sources.append({"spec": sp.get("spec_path", ""), "error": str(ex)})

    weight = w_sum if bool(np.any(w_sum > 0)) else np.ones_like(e, dtype=float)
    fit0 = _fit_powerlaw_plus_diskline_proxy(
        energy_keV=e,
        net_counts=net,
        var_counts=var,
        weight=weight,
        band_keV=band_default,
        min_counts=50,
        gain_shift=0.0,
    )

    # Broad line detection check (proxy): continuum-only vs continuum+diskline.
    delta_chi2_threshold = 9.0
    delta_chi2 = float("nan")
    line_detected = False
    cont_fit_proxy: Dict[str, Any] = {}
    try:
        cont_fit_proxy = _fit_powerlaw_only_proxy(
            energy_keV=e,
            net_counts=net,
            var_counts=var,
            weight=weight,
            band_keV=band_default,
            min_counts=50,
            gain_shift=0.0,
        )
        chi2_cont = float(cont_fit_proxy.get("fit", {}).get("chi2", float("nan")))
        chi2_full = float(fit0.get("fit", {}).get("chi2", float("nan")))
        if np.isfinite(chi2_cont) and np.isfinite(chi2_full):
            delta_chi2 = float(chi2_cont - chi2_full)
            line_detected = bool(delta_chi2 >= float(delta_chi2_threshold))
    except Exception as ex:
        debug["line_detection_error_proxy"] = str(ex)

    m_band0 = (e >= float(band_default[0])) & (e <= float(band_default[1])) & np.isfinite(e) & np.isfinite(net) & np.isfinite(var)
    net_counts_band_pos = float(np.sum(np.clip(net[m_band0], 0.0, None))) if np.any(m_band0) else float("nan")
    var_counts_band = float(np.sum(np.clip(var[m_band0], 0.0, None))) if np.any(m_band0) else float("nan")
    snr_band_proxy = float(net_counts_band_pos / math.sqrt(var_counts_band)) if var_counts_band > 0 else float("nan")

    r_in_bound0 = str(fit0.get("r_in_bound") or "")
    r_in_stat0 = float(fit0.get("r_in_rg_stat", float("nan")))
    isco_constrained_proxy = bool(line_detected and (not r_in_bound0) and np.isfinite(r_in_stat0))
    proxy_quality = "ok"
    if not line_detected:
        proxy_quality = "no_broad_line"
    elif r_in_bound0:
        proxy_quality = f"r_in_bound:{r_in_bound0}"
    elif not np.isfinite(r_in_stat0):
        proxy_quality = "no_cov"

    # Systematics sweep (band / gain / rebin)
    sys_band = [(2.5, 10.0), (3.0, 10.0), (4.0, 10.0)]
    sys_gain = [-1e-3, 0.0, 1e-3]
    sys_rebin = [25, 50, 100]

    variants: List[Dict[str, Any]] = []
    for b in sys_band:
        for g in sys_gain:
            for mc in sys_rebin:
                try:
                    r = _fit_powerlaw_plus_diskline_proxy(
                        energy_keV=e,
                        net_counts=net,
                        var_counts=var,
                        weight=weight,
                        band_keV=b,
                        min_counts=int(mc),
                        gain_shift=float(g),
                    )
                except Exception as ex:
                    r = {"status": "fail", "band_keV": [float(b[0]), float(b[1])], "min_counts": int(mc), "gain_shift": float(g), "error": str(ex)}
                variants.append(r)

    r0 = float(fit0.get("r_in_rg", float("nan")))

    def _sys_component(selector: Iterable[Dict[str, Any]]) -> float:
        vals = []
        for it in selector:
            if it.get("status") != "ok":
                continue
            vals.append(float(it.get("r_in_rg", float("nan"))))
        vals = [v for v in vals if np.isfinite(v)]
        if not vals or not np.isfinite(r0):
            return float("nan")
        return float(max(abs(v - r0) for v in vals))

    sys_band_val = _sys_component([v for v in variants if v.get("min_counts") == 50 and v.get("gain_shift") == 0.0])
    sys_gain_val = _sys_component([v for v in variants if v.get("min_counts") == 50 and v.get("band_keV") == [3.0, 10.0]])
    sys_rebin_val = _sys_component([v for v in variants if v.get("band_keV") == [3.0, 10.0] and v.get("gain_shift") == 0.0])
    sys_terms = [x for x in [sys_band_val, sys_gain_val, sys_rebin_val] if np.isfinite(x)]
    sys_total = float(math.sqrt(float(np.sum(np.square(sys_terms))))) if sys_terms else float("nan")

    out_detail = out_dir / f"xmm_{obsid}__fek_broad_line_proxy_diskline.json"
    detail = {
        "generated_utc": _utc_now(),
        "obsid": obsid,
        "rev": rev,
        "method": "powerlaw + diskline_proxy (no RMF folding; PI->E uses 5 eV/bin assumption; ARF×exposure weighting enabled)",
        "inputs": {"band_default_keV": list(band_default), "min_counts_default": 50},
        "selected": {"m1": specs[0]["spec_path"] if specs else "", "m2": specs[1]["spec_path"] if len(specs) > 1 else ""},
        "spectra": [{"spec": s["spec_path"], "bkg": s["back_path"], "has_bkg": s["has_background"], "scale": s["background_scale"]} for s in specs],
        "weights": {"type": "arf_specread × exposure", "sources": weight_sources},
        "line_detection": {
            "delta_chi2": float(delta_chi2),
            "threshold": float(delta_chi2_threshold),
            "detected": bool(line_detected),
            "isco_constrained_proxy": bool(isco_constrained_proxy),
            "proxy_quality": str(proxy_quality),
            "net_counts_band_pos": float(net_counts_band_pos),
            "var_counts_band": float(var_counts_band),
            "snr_band_proxy": float(snr_band_proxy),
            "continuum_fit": cont_fit_proxy or {},
        },
        "fit_default": fit0,
        "systematics": {
            "components": {"band": sys_band_val, "gain": sys_gain_val, "rebin": sys_rebin_val},
            "sys_total": sys_total,
            "variants": variants,
        },
        "debug": debug,
    }
    _write_json(out_detail, detail)

    out = {
        "status": "ok",
        "method_tag": "proxy_diskline_v1",
        "r_in_rg": r0,
        "r_in_rg_stat": float(fit0.get("r_in_rg_stat", float("nan"))),
        "r_in_rg_sys": sys_total,
        "r_in_bound": str(fit0.get("r_in_bound") or ""),
        "proxy_quality": str(proxy_quality),
        "proxy_line_detected": bool(line_detected),
        "proxy_delta_chi2": float(delta_chi2),
        "proxy_net_counts_band_pos": float(net_counts_band_pos),
        "proxy_snr_band": float(snr_band_proxy),
        "proxy_isco_constrained": bool(isco_constrained_proxy),
        "proxy_region_mode": "",
        "detail_json": _rel(out_detail),
    }
    return out, debug


def _fit_nustar_obsid_proxy(
    cache_root: Path,
    *,
    obsid: str,
    out_dir: Path,
    region_mode: str,
    det1_bin_size: int,
    det1_max: int,
    src_radius: float,
    bkg_inner: float,
    bkg_outer: float,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    event_dir = cache_root / "nustar" / obsid / "event_cl"
    debug: Dict[str, Any] = {"event_dir": _rel(event_dir), "files": [], "region_mode": str(region_mode)}
    if not event_dir.exists():
        return None, {"status": "missing_event_dir", **debug}

    # Use all cleaned event files for A/B modules under event_cl.
    cands = sorted(event_dir.glob(f"nu{obsid}[AB]*_cl.evt*"))
    if not cands:
        # Some archives may not use ".evt" suffix; allow ".fits" too as a fallback.
        cands = sorted(event_dir.glob(f"nu{obsid}[AB]*_cl*.fits*"))
    if not cands:
        return None, {"status": "no_event_files", **debug}

    requested_mode = str(region_mode).strip().lower()
    if requested_mode not in {"none", "auto_det1", "src_reg"}:
        raise ValueError("invalid region_mode for NuSTAR (expected: none|auto_det1|src_reg)")

    band_default = (3.0, 10.0)

    def _build_spectrum_none() -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        counts0 = np.zeros(0, dtype=float)
        n_events0 = 0
        files0: List[Dict[str, Any]] = []
        for p in cands:
            try:
                pi = _fits_read_event_pi(p)
                n_events0 += int(pi.size)
                if pi.size == 0:
                    files0.append({"path": _rel(p), "n_events": 0})
                    continue
                c = np.bincount(pi, minlength=int(np.max(pi)) + 1).astype(float)
                counts0 = _accumulate_bincount(counts0, c)
                files0.append({"path": _rel(p), "n_events": int(pi.size), "max_pi": int(np.max(pi))})
            except Exception as e:
                files0.append({"path": _rel(p), "error": str(e)})
                continue
        if counts0.size < 10 or float(np.sum(counts0)) <= 0:
            raise RuntimeError(f"empty_counts (n_events={n_events0})")
        ch0 = np.arange(int(counts0.size), dtype=float)
        e0 = _energy_from_nustar_pi(ch0)
        y0 = np.asarray(counts0, dtype=float)
        v0 = np.asarray(counts0, dtype=float)
        # Persist per-file stats only for this mode.
        debug["files"] = files0
        return e0, y0, v0, {"mode": "none", "note": "no region selection; full-FOV PI histogram"}

    def _build_spectrum_auto_det1() -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        spec_dbg0: Dict[str, Any] = {}
        center, cdbg = _nustar_det1_peak_center(cands, bin_size=int(det1_bin_size), det1_max=int(det1_max))
        spec_dbg0["center_det1"] = cdbg
        if center is None:
            raise RuntimeError("center_not_found")
        spec, sdbg = _nustar_extract_net_spectrum_det1(
            cands,
            center_det1=(float(center[0]), float(center[1])),
            src_radius=float(src_radius),
            bkg_inner=float(bkg_inner),
            bkg_outer=float(bkg_outer),
            det1_max=int(det1_max),
        )
        spec_dbg0["spectrum"] = sdbg
        if spec is None:
            raise RuntimeError("spectrum_extract_failed")
        counts0 = np.asarray(spec["net_counts"], dtype=float)
        var0 = np.asarray(spec["var_counts"], dtype=float)
        ch0 = np.arange(int(counts0.size), dtype=float)
        e0 = _energy_from_nustar_pi(ch0)
        return e0, counts0, var0, spec_dbg0

    def _build_spectrum_src_reg() -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Use NuSTAR pipeline source region files (*_src.reg) in (X,Y) to build a net spectrum.
        Background is an annulus centered at the same (X,Y) with radii scaled from the src radius.
        """
        reg_files = sorted(event_dir.glob(f"nu{obsid}[AB]*_src.reg"))
        reg_dbg: Dict[str, Any] = {"mode": "src_reg", "region_files": [_rel(p) for p in reg_files], "regions": {}, "modules": []}
        if not reg_files:
            raise RuntimeError("no_src_reg_files")

        regions: Dict[str, Tuple[float, float, float]] = {}
        for p in reg_files:
            m = re.search(rf"nu{re.escape(obsid)}(?P<mod>[AB])", p.name)
            mod = str(m.group("mod")) if m else ""
            circ = _parse_ds9_circle_region(p)
            if circ is None:
                reg_dbg["regions"][mod or p.name] = {"path": _rel(p), "status": "parse_failed"}
                continue
            regions[mod] = circ
            reg_dbg["regions"][mod] = {"path": _rel(p), "center_xy": [float(circ[0]), float(circ[1])], "src_radius": float(circ[2])}

        if not regions:
            raise RuntimeError("no_parseable_src_reg")

        net_sum = np.zeros(0, dtype=float)
        var_sum = np.zeros(0, dtype=float)
        used_any = False
        for mod in ["A", "B"]:
            files_mod = [p for p in cands if f"nu{obsid}{mod}" in p.name]
            if not files_mod:
                continue
            cx, cy, r_src = regions.get(mod) or next(iter(regions.values()))
            bkg_inner0 = 3.0 * float(r_src)
            bkg_outer0 = 6.0 * float(r_src)
            spec, sdbg = _nustar_extract_net_spectrum_xy_circle(
                files_mod,
                center_xy=(float(cx), float(cy)),
                src_radius=float(r_src),
                bkg_inner=float(bkg_inner0),
                bkg_outer=float(bkg_outer0),
            )
            reg_dbg["modules"].append({"module": mod, "n_event_files": int(len(files_mod)), "region_xy": [float(cx), float(cy)], "src_radius": float(r_src), "bkg_inner": float(bkg_inner0), "bkg_outer": float(bkg_outer0), "spectrum": sdbg})
            if spec is None:
                continue
            used_any = True
            net_sum = _accumulate_bincount(net_sum, np.asarray(spec["net_counts"], dtype=float))
            var_sum = _accumulate_bincount(var_sum, np.asarray(spec["var_counts"], dtype=float))

        if not used_any or net_sum.size < 10 or float(np.sum(np.clip(net_sum, 0.0, None))) <= 0:
            raise RuntimeError("empty_net_counts_src_reg")

        ch0 = np.arange(int(net_sum.size), dtype=float)
        e0 = _energy_from_nustar_pi(ch0)
        return e0, net_sum, var_sum, reg_dbg

    fit_attempts: List[Dict[str, Any]] = []
    fit0: Optional[Dict[str, Any]] = None
    e = y = v = None  # type: ignore[assignment]
    region_info: Dict[str, Any] = {}
    min_counts0 = 25

    if requested_mode == "src_reg":
        attempt_modes = ["src_reg", "auto_det1", "none"]
    elif requested_mode == "auto_det1":
        attempt_modes = ["auto_det1", "src_reg", "none"]
    else:
        attempt_modes = ["none"]
    for mode in attempt_modes:
        try:
            if mode == "src_reg":
                e, y, v, region_info = _build_spectrum_src_reg()
            elif mode == "auto_det1":
                e, y, v, region_info = _build_spectrum_auto_det1()
            else:
                e, y, v, region_info = _build_spectrum_none()
        except Exception as ex:
            fit_attempts.append({"mode": mode, "stage": "spectrum", "error": str(ex)})
            continue

        m_band = (e >= float(band_default[0])) & (e <= float(band_default[1])) & np.isfinite(e) & np.isfinite(v)
        band_proxy = float(np.sum(np.clip(v[m_band], 0.0, None))) if np.any(m_band) else float(np.sum(np.clip(v, 0.0, None)))
        min_counts0 = 25
        if band_proxy < 5000.0:
            min_counts0 = 10
        if band_proxy < 1000.0:
            min_counts0 = 5

        try:
            fit0 = _fit_powerlaw_plus_diskline_proxy(
                energy_keV=e,
                net_counts=y,
                var_counts=v,
                band_keV=band_default,
                min_counts=int(min_counts0),
                gain_shift=0.0,
                sigma_instr_keV=0.18,
            )
            debug["region"] = region_info
            debug["region_mode_effective"] = str(mode)
            break
        except Exception as ex:
            fit_attempts.append({"mode": mode, "stage": "fit", "min_counts": int(min_counts0), "error": str(ex)})
            fit0 = None
            continue

    debug["fit_attempts"] = fit_attempts
    if fit0 is None or e is None or y is None or v is None:
        return None, {"status": "fit_failed", **debug}

    # Proxy "broad line detected?" check: compare continuum-only vs continuum+diskline.
    # This is used to freeze the handling of low-S/N obsids (no broad line -> no ISCO constraint).
    delta_chi2 = float("nan")
    line_detected = False
    delta_chi2_threshold = 9.0
    cont_fit: Optional[Dict[str, Any]] = None
    try:
        cont_fit = _fit_powerlaw_only_proxy(
            energy_keV=e,
            net_counts=y,
            var_counts=v,
            band_keV=band_default,
            min_counts=int(min_counts0),
            gain_shift=0.0,
        )
        chi2_cont = float(cont_fit.get("fit", {}).get("chi2", float("nan")))
        chi2_full = float(fit0.get("fit", {}).get("chi2", float("nan")))
        if np.isfinite(chi2_cont) and np.isfinite(chi2_full):
            delta_chi2 = float(chi2_cont - chi2_full)
            line_detected = bool(delta_chi2 >= float(delta_chi2_threshold))
    except Exception as ex:
        debug["line_detection_error"] = str(ex)

    m_band0 = (e >= float(band_default[0])) & (e <= float(band_default[1])) & np.isfinite(e) & np.isfinite(y) & np.isfinite(v)
    net_counts_band_pos = float(np.sum(np.clip(y[m_band0], 0.0, None))) if np.any(m_band0) else float("nan")
    var_counts_band = float(np.sum(np.clip(v[m_band0], 0.0, None))) if np.any(m_band0) else float("nan")
    snr_band_proxy = float(net_counts_band_pos / math.sqrt(var_counts_band)) if var_counts_band > 0 else float("nan")

    r_in_bound0 = str(fit0.get("r_in_bound") or "")
    r_in_stat0 = float(fit0.get("r_in_rg_stat", float("nan")))
    isco_constrained_proxy = bool(line_detected and (not r_in_bound0) and np.isfinite(r_in_stat0))
    proxy_quality = "ok"
    if not line_detected:
        proxy_quality = "no_broad_line"
    elif r_in_bound0:
        proxy_quality = f"r_in_bound:{r_in_bound0}"
    elif not np.isfinite(r_in_stat0):
        proxy_quality = "no_cov"

    sys_band = [(3.0, 10.0), (3.0, 12.0), (4.0, 10.0)]
    sys_gain = [-1e-3, 0.0, 1e-3]
    sys_rebin = sorted({max(2, int(min_counts0 // 2)), int(min_counts0), int(min_counts0 * 2), int(min_counts0 * 4)})
    sys_rebin = [int(x) for x in sys_rebin if int(x) <= 200]

    variants: List[Dict[str, Any]] = []
    for b in sys_band:
        for g in sys_gain:
            for mc in sys_rebin:
                try:
                    r = _fit_powerlaw_plus_diskline_proxy(
                        energy_keV=e,
                        net_counts=y,
                        var_counts=v,
                        band_keV=b,
                        min_counts=int(mc),
                        gain_shift=float(g),
                        sigma_instr_keV=0.18,
                    )
                except Exception as ex:
                    r = {
                        "status": "fail",
                        "band_keV": [float(b[0]), float(b[1])],
                        "min_counts": int(mc),
                        "gain_shift": float(g),
                        "error": str(ex),
                    }
                variants.append(r)

    r0 = float(fit0.get("r_in_rg", float("nan")))

    def _sys_component(selector: Iterable[Dict[str, Any]]) -> float:
        vals = []
        for it in selector:
            if it.get("status") != "ok":
                continue
            vals.append(float(it.get("r_in_rg", float("nan"))))
        vals = [v for v in vals if np.isfinite(v)]
        if not vals or not np.isfinite(r0):
            return float("nan")
        return float(max(abs(v - r0) for v in vals))

    sys_band_val = _sys_component([v for v in variants if v.get("min_counts") == int(min_counts0) and v.get("gain_shift") == 0.0])
    sys_gain_val = _sys_component(
        [v for v in variants if v.get("min_counts") == int(min_counts0) and v.get("band_keV") == [3.0, 10.0]]
    )
    sys_rebin_val = _sys_component([v for v in variants if v.get("band_keV") == [3.0, 10.0] and v.get("gain_shift") == 0.0])

    # Region-extraction systematic: compare src_reg vs auto_det1 when possible.
    sys_region_val = float("nan")
    region_variants: List[Dict[str, Any]] = []
    try:
        effective_mode = str(debug.get("region_mode_effective") or requested_mode or "").strip().lower()
        alt_modes: List[str] = []
        if effective_mode == "src_reg":
            alt_modes = ["auto_det1"]
        elif effective_mode == "auto_det1":
            alt_modes = ["src_reg"]
        for alt in alt_modes:
            try:
                if alt == "src_reg":
                    e_alt, y_alt, v_alt, info_alt = _build_spectrum_src_reg()
                else:
                    e_alt, y_alt, v_alt, info_alt = _build_spectrum_auto_det1()
                fit_alt = _fit_powerlaw_plus_diskline_proxy(
                    energy_keV=e_alt,
                    net_counts=y_alt,
                    var_counts=v_alt,
                    band_keV=band_default,
                    min_counts=int(min_counts0),
                    gain_shift=0.0,
                    sigma_instr_keV=0.18,
                )
                r_alt = float(fit_alt.get("r_in_rg", float("nan")))
                region_variants.append({"mode": alt, "r_in_rg": r_alt, "fit": fit_alt, "region": info_alt})
            except Exception as ex:
                region_variants.append({"mode": alt, "status": "fail", "error": str(ex)})
        if np.isfinite(r0):
            vals = [float(v.get("r_in_rg", float("nan"))) for v in region_variants if v.get("status") != "fail"]
            vals = [v for v in vals if np.isfinite(v)]
            if vals:
                sys_region_val = float(max(abs(v - r0) for v in vals))
    except Exception:
        pass

    sys_terms = [x for x in [sys_band_val, sys_gain_val, sys_rebin_val, sys_region_val] if np.isfinite(x)]
    sys_total = float(math.sqrt(float(np.sum(np.square(sys_terms))))) if sys_terms else float("nan")

    out_detail = out_dir / f"nustar_{obsid}__fek_broad_line_proxy_diskline.json"
    detail = {
        "generated_utc": _utc_now(),
        "obsid": obsid,
        "method": "powerlaw + diskline_proxy (no RMF/ARF folding; PI->E uses 0.04 keV/bin assumption; region via src_reg (X,Y) or auto_det1 (DET1))",
        "inputs": {"band_default_keV": list(band_default), "min_counts_default": int(min_counts0), "sigma_instr_keV": 0.18},
        "event_files": [it for it in debug.get("files", [])],
        "region": debug.get("region", {}),
        "line_detection": {
            "delta_chi2": float(delta_chi2),
            "threshold": float(delta_chi2_threshold),
            "detected": bool(line_detected),
            "isco_constrained_proxy": bool(isco_constrained_proxy),
            "proxy_quality": str(proxy_quality),
            "net_counts_band_pos": float(net_counts_band_pos),
            "var_counts_band": float(var_counts_band),
            "snr_band_proxy": float(snr_band_proxy),
            "continuum_fit": cont_fit or {},
        },
        "fit_default": fit0,
        "systematics": {
            "components": {"band": sys_band_val, "gain": sys_gain_val, "rebin": sys_rebin_val, "region": sys_region_val},
            "sys_total": sys_total,
            "variants": variants,
            "region_variants": region_variants,
        },
    }
    _write_json(out_detail, detail)

    out = {
        "status": "ok",
        "r_in_rg": r0,
        "r_in_rg_stat": float(fit0.get("r_in_rg_stat", float("nan"))),
        "r_in_rg_sys": sys_total,
        "r_in_bound": str(fit0.get("r_in_bound") or ""),
        "proxy_quality": str(proxy_quality),
        "proxy_line_detected": bool(line_detected),
        "proxy_delta_chi2": float(delta_chi2),
        "proxy_net_counts_band_pos": float(net_counts_band_pos),
        "proxy_snr_band": float(snr_band_proxy),
        "proxy_isco_constrained": bool(isco_constrained_proxy),
        "proxy_region_mode": str(debug.get("region_mode_effective") or debug.get("region_mode") or ""),
        "detail_json": _rel(out_detail),
    }
    return out, debug


def _systematics_template() -> Dict[str, Any]:
    return {
        "version": "isco_constraints_v7",
        "generated_utc": _utc_now(),
        "knobs": {
            "continuum_model": {
                "default": "powerlaw",
                "candidates": ["powerlaw", "cutoffpl", "nthcomp"],
                "note": "強場の反射fitで continuum 依存の系統を台帳化する。",
            },
            "reflection_model": {
                "default": "relxill",
                "candidates": ["relxill", "relxillCp", "reflionx+relconv"],
                "note": "relativistic blurring を含む反射モデルの依存性を系統として扱う。",
            },
            "energy_band_keV": {
                "default": [3.0, 10.0],
                "candidates": [[2.5, 10.0], [3.0, 12.0], [4.0, 10.0]],
                "note": "Fe-K 周辺の帯域選択が r_in に与える影響を sys として分解する。",
            },
            "absorption_model": {
                "default": "tbabs",
                "candidates": ["tbabs", "tbnew", "tbabs+warmabs"],
                "note": "吸収モデル選択（低エネルギー側）を sys として sweep する。",
            },
            "cross_calibration": {
                "default": "free_constant_per_instrument",
                "candidates": ["free_constant_per_instrument"],
                "note": "XMM/NuSTAR/XRISM の cross-cal は定数項で吸収し、残差を sys に計上する。",
            },
            "rebinning": {
                "default": {"min_counts": 50},
                "candidates": [{"min_counts": 25}, {"min_counts": 50}, {"min_counts": 100}],
                "note": "ビニング依存を sys_rebin として評価する。",
            },
            "gain_shift": {
                "default": 0.0,
                "candidates": [-1e-3, 0.0, 1e-3],
                "note": "エネルギースケール（gain）不確かさを E->E(1+g) の形で proxy 評価する。",
            },
            "line_rest_energy_keV": {
                "default": 6.4,
                "candidates": [6.4],
                "note": "初版は Fe Kα を 6.4 keV 固定で proxy fit（ionized line は次段）。",
            },
            "arf_weighting": {
                "default": "enabled",
                "candidates": ["enabled"],
                "note": "XMM proxy では ARF×exposure の重み（throughput）を組み込み、帯域内の形状歪みを最低限反映する。",
            },
            "nustar_region": {
                "default": {"mode": "src_reg"},
                "candidates": [
                    {"mode": "none"},
                    {"mode": "auto_det1", "src_radius": 60.0, "bkg_inner": 120.0, "bkg_outer": 180.0},
                    {"mode": "src_reg"},
                ],
                "note": "NuSTAR proxy は優先的に event_cl の *_src.reg（X,Y）を用いて source region を固定し、background は src 半径に比例した annulus を用いる。fallback として DET1-based（auto_det1）も残す（HEASoft 非依存）。",
            },
        },
        "outputs": {
            "isco_constraints_csv": "output/private/xrism/fek_relativistic_broadening_isco_constraints.csv",
            "isco_constraints_metrics_json": "output/private/xrism/fek_relativistic_broadening_isco_constraints_metrics.json",
            "model_systematics_json": "output/private/xrism/fek_relativistic_broadening_model_systematics.json",
            "xmm_proxy_detail_json": "output/private/xrism/xmm_<obsid>__fek_broad_line_proxy_diskline.json",
            "nustar_proxy_detail_json": "output/private/xrism/nustar_<obsid>__fek_broad_line_proxy_diskline.json",
            "xmm_reflection_xspec_json": "output/private/xrism/xmm_<obsid>__fek_broad_line_reflection_xspec.json",
        },
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--targets",
        default=str(_ROOT / "data/xrism/sources/xmm_nustar_targets.json"),
        help="Target seed JSON (default: data/xrism/sources/xmm_nustar_targets.json)",
    )
    p.add_argument(
        "--cache-root",
        default=str(_ROOT / "data/xrism/xmm_nustar"),
        help="Cache root produced by fetch_xmm_nustar_heasarc.py",
    )
    p.add_argument(
        "--out-dir",
        default=str(_ROOT / "output/private/xrism"),
        help="Output directory",
    )
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="Only (re)generate the summary PNG from the existing CSV (no proxy recomputation).",
    )
    p.add_argument(
        "--force-recompute-proxy",
        action="store_true",
        help="Ignore existing per-obsid proxy detail JSON and recompute proxy fits from cached files.",
    )

    p.add_argument(
        "--nustar-region",
        default="src_reg",
        choices=["src_reg", "auto_det1", "none"],
        help="NuSTAR proxy region extraction mode (default: src_reg).",
    )
    p.add_argument("--nustar-det1-bin-size", type=int, default=10, help="DET1 histogram bin size (pixels) for auto_det1.")
    p.add_argument("--nustar-det1-max", type=int, default=400, help="Valid DET1 coordinate range: [0, det1_max).")
    p.add_argument("--nustar-src-radius", type=float, default=60.0, help="Source extraction radius in DET1 pixels (auto_det1).")
    p.add_argument("--nustar-bkg-inner", type=float, default=120.0, help="Background annulus inner radius in DET1 pixels.")
    p.add_argument("--nustar-bkg-outer", type=float, default=180.0, help="Background annulus outer radius in DET1 pixels.")
    args = p.parse_args(list(argv) if argv is not None else None)

    targets_path = Path(args.targets)
    cache_root = Path(args.cache_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "fek_relativistic_broadening_isco_constraints.csv"
    out_png = out_dir / "fek_relativistic_broadening_isco_constraints.png"

    if bool(args.plot_only):
        rows = _read_csv_rows(out_csv)
        _plot_isco_constraints(rows, out_png=out_png)
        print(json.dumps({"png": _rel(out_png), "n_rows": len(rows)}, ensure_ascii=False))
        return 0

    targets = _read_json(targets_path)
    target_items = targets.get("targets", [])
    if not isinstance(target_items, list) or not target_items:
        raise RuntimeError(f"no targets in {targets_path}")

    rows: List[Dict[str, str]] = []
    n_total = 0
    n_cached = 0
    computed = 0
    blocked = 0
    xspec_seen = 0
    xspec_ok = 0
    xspec_blocked = 0
    xspec_fail = 0
    xspec_dry_run = 0

    per_obsid_detail: List[Dict[str, Any]] = []

    r_isco_gr_rg = _isco_gr_schwarzschild_rg()
    r_isco_pmodel_rg = _isco_pmodel_exponential_metric_rg()

    for t in target_items:
        if not isinstance(t, dict):
            continue
        target_name = str(t.get("target_name") or "")
        target_type = str(t.get("target_type") or "")

        xmm_list = t.get("xmm", [])
        if isinstance(xmm_list, list):
            for d in xmm_list:
                if not isinstance(d, dict):
                    continue
                obsid = str(d.get("obsid") or "")
                rev = str(d.get("rev") or "rev0")
                cached = _detect_cache_xmm(cache_root, obsid=obsid, rev=rev)
                fit_res: Optional[Dict[str, Any]] = None
                xspec_meta: Dict[str, Any] = {}

                # Prefer XSPEC/pyXspec forward-fold fit results if they exist.
                # (They are produced by scripts/xrism/fek_relativistic_broadening_reflection_fit.py)
                xspec_result_path = out_dir / f"xmm_{obsid}__fek_broad_line_reflection_xspec.json"
                if xspec_result_path.exists():
                    xspec_obj = _read_json(xspec_result_path)
                    xspec_status = str(xspec_obj.get("status") or "").strip().lower()
                    xspec_seen += 1
                    xspec_meta = {
                        "status": xspec_status,
                        "method_tag": str(xspec_obj.get("method_tag") or ""),
                        "json": _rel(xspec_result_path),
                        "blocked_reason": str(xspec_obj.get("blocked_reason") or xspec_obj.get("xspec_import_error") or ""),
                        "error": str(xspec_obj.get("error") or ""),
                        "xspec_available": bool(xspec_obj.get("xspec_available")) if "xspec_available" in xspec_obj else None,
                    }
                    if xspec_status == "ok":
                        xspec_ok += 1
                        fit_res = {
                            "status": "ok",
                            "method_tag": str(xspec_obj.get("method_tag") or "xspec_diskline_v2"),
                            "r_in_rg": float(xspec_obj.get("r_in_rg", float("nan"))),
                            "r_in_rg_stat": float(xspec_obj.get("r_in_rg_stat", float("nan"))),
                            "r_in_rg_sys": float(xspec_obj.get("r_in_rg_sys", float("nan"))),
                            "r_in_bound": str(xspec_obj.get("r_in_bound") or ""),
                            # proxy fields are not defined for XSPEC at this stage; keep minimal placeholders.
                            "proxy_quality": "ok",
                            "proxy_line_detected": True,
                            "proxy_delta_chi2": float("nan"),
                            "proxy_net_counts_band_pos": float("nan"),
                            "proxy_snr_band": float("nan"),
                            "proxy_isco_constrained": True,
                            "proxy_region_mode": "",
                            "detail_json": _rel(xspec_result_path),
                        }
                    elif xspec_status.startswith("blocked"):
                        xspec_blocked += 1
                    elif xspec_status == "dry_run":
                        xspec_dry_run += 1
                    elif xspec_status == "fail":
                        xspec_fail += 1

                if cached:
                    try:
                        if fit_res is None:
                            fit_res, _ = _fit_xmm_obsid_diskline(cache_root, obsid=obsid, rev=rev, out_dir=out_dir)
                    except Exception as e:
                        fit_res = {"status": "fail", "error": str(e)}
                n_total += 1
                n_cached += int(cached)
                status = "not_computed"
                note = "PPS cache + reflection fit pending"
                r_in = ""
                r_stat = ""
                r_sys = ""
                r_in_bound = ""
                proxy_quality = ""
                proxy_line_detected = ""
                proxy_delta_chi2 = ""
                proxy_net_counts_band_pos = ""
                proxy_snr_band = ""
                proxy_isco_constrained = ""
                proxy_region_mode = ""
                if fit_res is not None:
                    if fit_res.get("status") == "ok":
                        method_tag = str(fit_res.get("method_tag") or "").strip() or "proxy_diskline_v1"
                        status = method_tag
                        bound = str(fit_res.get("r_in_bound") or "")
                        r_in_bound = bound
                        if method_tag.startswith("xspec_"):
                            method_label = "XSPEC forward-fold fit"
                        elif method_tag.startswith("rmf_"):
                            method_label = "RMF-folded fit"
                        else:
                            method_label = "proxy fit (no RMF)"
                        note = f"XMM PPS {method_label}: detail={fit_res.get('detail_json','')}"
                        if bound:
                            note += f"; r_in_bound={bound}"
                        r_in = f"{float(fit_res.get('r_in_rg', float('nan'))):.6g}"
                        r_stat = f"{float(fit_res.get('r_in_rg_stat', float('nan'))):.6g}"
                        r_sys = f"{float(fit_res.get('r_in_rg_sys', float('nan'))):.6g}"
                        proxy_quality = str(fit_res.get("proxy_quality") or "")
                        proxy_line_detected = _as_bool01(bool(fit_res.get("proxy_line_detected")))
                        proxy_delta_chi2 = f"{float(fit_res.get('proxy_delta_chi2', float('nan'))):.6g}"
                        proxy_net_counts_band_pos = f"{float(fit_res.get('proxy_net_counts_band_pos', float('nan'))):.6g}"
                        proxy_snr_band = f"{float(fit_res.get('proxy_snr_band', float('nan'))):.6g}"
                        proxy_isco_constrained = _as_bool01(bool(fit_res.get("proxy_isco_constrained")))
                        proxy_region_mode = str(fit_res.get("proxy_region_mode") or "")
                        computed += 1
                    else:
                        status = "failed"
                        note = f"XMM fit failed: {fit_res.get('error','')}"
                z_fields = (
                    _compute_z_fields(
                        r_in_rg=r_in,
                        r_in_rg_stat=r_stat,
                        r_in_rg_sys=r_sys,
                        r_isco_gr_rg=float(r_isco_gr_rg),
                        r_isco_pmodel_rg=float(r_isco_pmodel_rg),
                    )
                    if not r_in_bound
                    else {"delta_gr_rg": "", "z_gr": "", "delta_pmodel_rg": "", "z_pmodel": ""}
                )
                err_fields = _compute_error_fields(r_in_bound=r_in_bound, r_in_rg_stat=r_stat, r_in_rg_sys=r_sys)
                detail_json = str(fit_res.get("detail_json") or "") if isinstance(fit_res, dict) else ""
                sys_comps = _extract_sys_components_from_detail_json(detail_json)
                row = {
                    "target_name": target_name,
                    "target_type": target_type,
                    "mission": "xmm",
                    "obsid": obsid,
                    "instrument_hint": "EPIC",
                    "data_cached": _as_bool01(cached),
                    "r_in_rg": r_in,
                    "r_in_rg_stat": r_stat,
                    "r_in_rg_sys": r_sys,
                    "r_in_bound": r_in_bound,
                    "status": status,
                    "note": note,
                    "proxy_quality": proxy_quality,
                    "proxy_line_detected": proxy_line_detected,
                    "proxy_delta_chi2": proxy_delta_chi2,
                    "proxy_net_counts_band_pos": proxy_net_counts_band_pos,
                    "proxy_snr_band": proxy_snr_band,
                    "proxy_isco_constrained": proxy_isco_constrained,
                    "proxy_region_mode": proxy_region_mode,
                }
                row.update(z_fields)
                row.update(err_fields)
                row.update(sys_comps)
                rows.append(row)
                if fit_res is not None and isinstance(fit_res, dict):
                    rec: Dict[str, Any] = {"mission": "xmm", "obsid": obsid, "rev": rev, "fit": fit_res}
                    if xspec_meta:
                        rec["xspec"] = xspec_meta
                    per_obsid_detail.append(rec)
                elif xspec_meta:
                    per_obsid_detail.append({"mission": "xmm", "obsid": obsid, "rev": rev, "xspec": xspec_meta})

        nustar_list = t.get("nustar", [])
        if isinstance(nustar_list, list):
            for d in nustar_list:
                if not isinstance(d, dict):
                    continue
                obsid = str(d.get("obsid") or "")
                cached = _detect_cache_nustar(cache_root, obsid=obsid)
                fit_res: Optional[Dict[str, Any]] = None
                debug: Dict[str, Any] = {}
                n_total += 1
                n_cached += int(cached)
                status = "not_computed"
                note = "event_cl cache + reflection fit pending"
                if cached:
                    try:
                        cached_detail_json = out_dir / f"nustar_{obsid}__fek_broad_line_proxy_diskline.json"
                        if not bool(args.force_recompute_proxy) and cached_detail_json.exists():
                            cached_fit = _load_cached_nustar_proxy_detail_json(cached_detail_json)
                            if isinstance(cached_fit, dict):
                                fit_res = cached_fit
                                debug = {"reused_detail_json": _rel(cached_detail_json)}
                            else:
                                fit_res, debug = _fit_nustar_obsid_proxy(
                                    cache_root,
                                    obsid=obsid,
                                    out_dir=out_dir,
                                    region_mode=str(args.nustar_region),
                                    det1_bin_size=int(args.nustar_det1_bin_size),
                                    det1_max=int(args.nustar_det1_max),
                                    src_radius=float(args.nustar_src_radius),
                                    bkg_inner=float(args.nustar_bkg_inner),
                                    bkg_outer=float(args.nustar_bkg_outer),
                                )
                        else:
                            fit_res, debug = _fit_nustar_obsid_proxy(
                                cache_root,
                                obsid=obsid,
                                out_dir=out_dir,
                                region_mode=str(args.nustar_region),
                                det1_bin_size=int(args.nustar_det1_bin_size),
                                det1_max=int(args.nustar_det1_max),
                                src_radius=float(args.nustar_src_radius),
                                bkg_inner=float(args.nustar_bkg_inner),
                                bkg_outer=float(args.nustar_bkg_outer),
                            )
                    except Exception as e:
                        fit_res = {"status": "fail", "error": str(e)}
                r_in = ""
                r_stat = ""
                r_sys = ""
                r_in_bound = ""
                proxy_quality = ""
                proxy_line_detected = ""
                proxy_delta_chi2 = ""
                proxy_net_counts_band_pos = ""
                proxy_snr_band = ""
                proxy_isco_constrained = ""
                proxy_region_mode = ""
                if fit_res is not None:
                    if fit_res.get("status") == "ok":
                        status = "proxy_diskline_v1"
                        bound = str(fit_res.get("r_in_bound") or "")
                        r_in_bound = bound
                        note = f"NuSTAR event_cl proxy fit (no RMF): detail={fit_res.get('detail_json','')}"
                        if bound:
                            note += f"; r_in_bound={bound}"
                        r_in = f"{float(fit_res.get('r_in_rg', float('nan'))):.6g}"
                        r_stat = f"{float(fit_res.get('r_in_rg_stat', float('nan'))):.6g}"
                        r_sys = f"{float(fit_res.get('r_in_rg_sys', float('nan'))):.6g}"
                        proxy_quality = str(fit_res.get("proxy_quality") or "")
                        proxy_line_detected = _as_bool01(bool(fit_res.get("proxy_line_detected")))
                        proxy_delta_chi2 = f"{float(fit_res.get('proxy_delta_chi2', float('nan'))):.6g}"
                        proxy_net_counts_band_pos = f"{float(fit_res.get('proxy_net_counts_band_pos', float('nan'))):.6g}"
                        proxy_snr_band = f"{float(fit_res.get('proxy_snr_band', float('nan'))):.6g}"
                        proxy_isco_constrained = _as_bool01(bool(fit_res.get("proxy_isco_constrained")))
                        proxy_region_mode = str(fit_res.get("proxy_region_mode") or "")
                        computed += 1
                    else:
                        status = "failed"
                        note = f"proxy fit failed: {fit_res.get('error','')}"
                z_fields = (
                    _compute_z_fields(
                        r_in_rg=r_in,
                        r_in_rg_stat=r_stat,
                        r_in_rg_sys=r_sys,
                        r_isco_gr_rg=float(r_isco_gr_rg),
                        r_isco_pmodel_rg=float(r_isco_pmodel_rg),
                    )
                    if not r_in_bound
                    else {"delta_gr_rg": "", "z_gr": "", "delta_pmodel_rg": "", "z_pmodel": ""}
                )
                err_fields = _compute_error_fields(r_in_bound=r_in_bound, r_in_rg_stat=r_stat, r_in_rg_sys=r_sys)
                detail_json = str(fit_res.get("detail_json") or "") if isinstance(fit_res, dict) else ""
                sys_comps = _extract_sys_components_from_detail_json(detail_json)
                row = {
                    "target_name": target_name,
                    "target_type": target_type,
                    "mission": "nustar",
                    "obsid": obsid,
                    "instrument_hint": "FPMA/FPMB",
                    "data_cached": _as_bool01(cached),
                    "r_in_rg": r_in,
                    "r_in_rg_stat": r_stat,
                    "r_in_rg_sys": r_sys,
                    "r_in_bound": r_in_bound,
                    "status": status,
                    "note": note,
                    "proxy_quality": proxy_quality,
                    "proxy_line_detected": proxy_line_detected,
                    "proxy_delta_chi2": proxy_delta_chi2,
                    "proxy_net_counts_band_pos": proxy_net_counts_band_pos,
                    "proxy_snr_band": proxy_snr_band,
                    "proxy_isco_constrained": proxy_isco_constrained,
                    "proxy_region_mode": proxy_region_mode,
                }
                row.update(z_fields)
                row.update(err_fields)
                row.update(sys_comps)
                rows.append(row)
                if fit_res is not None and isinstance(fit_res, dict):
                    per_obsid_detail.append({"mission": "nustar", "obsid": obsid, "fit": fit_res, "debug": debug})

    header = [
        "target_name",
        "target_type",
        "mission",
        "obsid",
        "instrument_hint",
        "data_cached",
        "r_in_rg",
        "r_in_rg_stat",
        "r_in_rg_sys",
        "r_in_bound",
        "delta_gr_rg",
        "z_gr",
        "delta_pmodel_rg",
        "z_pmodel",
        "sigma_total_rg",
        "sys_stat_ratio",
        "sys_band_rg",
        "sys_gain_rg",
        "sys_rebin_rg",
        "sys_region_rg",
        "status",
        "note",
        "proxy_quality",
        "proxy_line_detected",
        "proxy_delta_chi2",
        "proxy_net_counts_band_pos",
        "proxy_snr_band",
        "proxy_isco_constrained",
        "proxy_region_mode",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})

    out_metrics = out_dir / "fek_relativistic_broadening_isco_constraints_metrics.json"
    n_nustar_line_detected = sum(1 for r in rows if r.get("mission") == "nustar" and r.get("proxy_line_detected") == "1")
    n_nustar_isco_constrained = sum(1 for r in rows if r.get("mission") == "nustar" and r.get("proxy_isco_constrained") == "1")
    n_isco_constrained = sum(1 for r in rows if r.get("proxy_isco_constrained") == "1")
    n_isco_constrained_sys_stat_le10 = sum(
        1
        for r in rows
        if r.get("proxy_isco_constrained") == "1"
        and (lambda x: (x is not None and math.isfinite(x) and x <= 10.0))(_maybe_float(r.get("sys_stat_ratio")))
    )
    metrics = {
        "generated_utc": _utc_now(),
        "targets": _rel(targets_path),
        "cache_root": _rel(cache_root),
        "n_rows": len(rows),
        "n_obsids_total": n_total,
        "n_obsids_cached": n_cached,
        "cache_coverage": (float(n_cached) / float(n_total)) if n_total else 0.0,
        "n_proxy_computed": int(computed),
        "n_nustar_line_detected": int(n_nustar_line_detected),
        "n_nustar_isco_constrained_proxy": int(n_nustar_isco_constrained),
        "n_isco_constrained": int(n_isco_constrained),
        "n_isco_constrained_sys_stat_le10": int(n_isco_constrained_sys_stat_le10),
        "n_xmm_xspec_seen": int(xspec_seen),
        "n_xmm_xspec_ok": int(xspec_ok),
        "n_xmm_xspec_blocked": int(xspec_blocked),
        "n_xmm_xspec_dry_run": int(xspec_dry_run),
        "n_xmm_xspec_fail": int(xspec_fail),
        "n_blocked_missing_dependency": int(blocked),
        "isco_reference": {
            "units": "r_g=GM/c^2",
            "gr": {"metric": "Schwarzschild", "spin_a_star": 0.0, "r_isco_rg": float(r_isco_gr_rg)},
            "pmodel": {
                "metric": "exponential_metric_yilmaz",
                "definition": "timelike ISCO; circumference radius",
                "r_isco_rg": float(r_isco_pmodel_rg),
            },
        },
        "outputs": {"csv": _rel(out_csv), "png": _rel(out_png)},
        "status": "isco_constraints_v7",
    }
    _write_json(out_metrics, metrics)

    out_sys = out_dir / "fek_relativistic_broadening_model_systematics.json"
    _write_json(out_sys, _systematics_template())

    out_detail = out_dir / "fek_relativistic_broadening_isco_constraints_detail.json"
    _write_json(
        out_detail,
        {
            "generated_utc": _utc_now(),
            "status": "isco_constraints_v7",
            "note": "XMM uses PPS spectra + ARF and a locally cached canned RMF (HEASARC CALDB) to forward-fold a diskline proxy in channel space (still not a full reflection model). NuSTAR uses event_cl and prefers pipeline *_src.reg (X,Y) when available, falling back to a DET1-based source region + background annulus (still no RMF/ARF folding). A continuum-vs-diskline delta-chi2 is recorded to freeze the handling of low-S/N obsids (no broad line -> no ISCO constraint). Full reflection fit (e.g., relxill+blurring) remains pending.",
            "per_obsid": per_obsid_detail,
        },
    )

    try:
        worklog.append_event(
            "xrism.fek_relativistic_broadening_isco_constraints.stub",
            {
                "csv": _rel(out_csv),
                "metrics": _rel(out_metrics),
                "systematics": _rel(out_sys),
                "detail": _rel(out_detail),
                "n_rows": len(rows),
                "n_proxy_computed": int(computed),
                "n_blocked_missing_dependency": int(blocked),
            },
        )
    except Exception:
        pass

    _plot_isco_constraints(rows, out_png=out_png)

    print(json.dumps({"csv": _rel(out_csv), "n_rows": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
