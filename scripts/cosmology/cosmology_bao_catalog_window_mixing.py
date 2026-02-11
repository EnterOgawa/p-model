#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_bao_catalog_window_mixing.py

Phase 4 / Step 4.5（BAOを一次統計から再構築）Stage B（確証決定打）:
catalog-based（Corrfunc; galaxy+random / recon）で得た pair-count grid（RR0/SS）から、
窓関数（survey selection/geometry）の multipoles を推定し、ξ0↔ξ2 の window mixing が
どの程度の大きさになり得るかを定量化する。

狙い：
- Ross post-recon の ξ2 ギャップが「窓関数の異方性による ξ0↔ξ2 混合」で説明できるかを、
  order-of-magnitude で切り分ける（mixing が小さければ、主因は recon 仕様差側へ寄せる）。

出力（固定）:
- output/cosmology/cosmology_bao_catalog_window_mixing.png
- output/cosmology/cosmology_bao_catalog_window_mixing_metrics.json

注意：
- 本スクリプト自体は Corrfunc を使わず、既に生成済みの npz を読むだけ（Windowsでも実行可）。
- Corrfunc 計算（npz 生成）は AGENTS.md の通り WSL（threads=24）で行う。
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


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


def _legendre_p(l: int, x: np.ndarray) -> np.ndarray:
    l = int(l)
    if l < 0:
        raise ValueError("l must be >=0")
    coeff = np.zeros((l + 1,), dtype=float)
    coeff[l] = 1.0
    return np.polynomial.legendre.legval(np.asarray(x, dtype=float), coeff)


def _window_coupling_coeffs() -> Dict[Tuple[int, int, int], float]:
    """
    Return coupling coefficients for Legendre product expansion:
      P_ell * P_L = Σ_n c_{n,ell,L} P_n
    where c_{n,ell,L} = (2n+1)/2 ∫_{-1..1} P_n P_ell P_L dμ.

    We compute only what we need: n in {0,2}, ell in {0,2,4}, L in {0,2,4,6,8}.
    """
    mu_q, w_q = np.polynomial.legendre.leggauss(256)
    mu_q = np.asarray(mu_q, dtype=float)
    w_q = np.asarray(w_q, dtype=float)

    ls = [0, 2, 4, 6, 8]
    p = {l: _legendre_p(l, mu_q) for l in ls}

    out: Dict[Tuple[int, int, int], float] = {}
    for n in (0, 2):
        pn = p[n]
        for ell in (0, 2, 4):
            pe = p[ell]
            for L in ls:
                pL = p[L]
                integral = float(np.sum(w_q * pn * pe * pL))
                out[(n, ell, L)] = 0.5 * float(2 * n + 1) * integral
    return out


_WINDOW_COEFFS = _window_coupling_coeffs()


def _window_wL(
    *,
    counts: np.ndarray,
    mu_edges: np.ndarray,
    ls: Tuple[int, ...] = (0, 2, 4, 6, 8),
) -> Dict[int, np.ndarray]:
    """
    Estimate window multipole ratios w_L(s)=R_L(s)/R_0(s) from μ-binned pair counts.

    Corrfunc convention uses μ∈[0,1] bins (evenness assumed), so:
      R_L(s) ∝ (2L+1) ∫_0^1 R(s,μ) P_L(μ) dμ
    We omit constant dμ factors (they cancel in ratios).
    """
    counts = np.asarray(counts, dtype=np.float64)
    mu_edges = np.asarray(mu_edges, dtype=np.float64)
    if counts.ndim != 2:
        raise ValueError("counts must be 2D (nbins,nmu)")
    if mu_edges.ndim != 1 or mu_edges.size < 2:
        raise ValueError("mu_edges must be 1D (nmu+1)")
    nmu = int(mu_edges.size - 1)
    if counts.shape[1] != nmu:
        raise ValueError(f"nmu mismatch: counts={counts.shape[1]} mu_edges={nmu}")

    mu_mid = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    r0 = np.sum(counts, axis=1)
    out: Dict[int, np.ndarray] = {0: np.ones_like(r0, dtype=np.float64)}
    for L in ls:
        if int(L) == 0:
            continue
        pL = _legendre_p(int(L), mu_mid)
        rL = float(2 * int(L) + 1) * np.sum(counts * pL[None, :], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            w = rL / r0
        out[int(L)] = np.where(np.isfinite(w), w, np.nan).astype(np.float64)
    return out


@dataclass(frozen=True)
class MixingSeries:
    s: np.ndarray
    m00: np.ndarray
    m02: np.ndarray
    m20: np.ndarray
    m22: np.ndarray
    m24: np.ndarray


def _mixing_from_w(wL: Dict[int, np.ndarray]) -> MixingSeries:
    s = None
    for v in wL.values():
        if s is None:
            s = np.zeros_like(v, dtype=np.float64)
            break
    if s is None:
        raise ValueError("empty wL")

    ls = (0, 2, 4, 6, 8)

    def mix(n: int, ell: int) -> np.ndarray:
        out = np.zeros_like(s, dtype=np.float64)
        for L in ls:
            c = float(_WINDOW_COEFFS[(int(n), int(ell), int(L))])
            if c == 0.0:
                continue
            w = wL.get(int(L), None)
            if w is None:
                continue
            out += c * np.asarray(w, dtype=np.float64)
        return out

    return MixingSeries(
        s=np.asarray(s, dtype=np.float64),
        m00=mix(0, 0),
        m02=mix(0, 2),
        m20=mix(2, 0),
        m22=mix(2, 2),
        m24=mix(2, 4),
    )


def _safe_max_abs_in_range(s: np.ndarray, y: np.ndarray, *, s_min: float, s_max: float) -> float:
    s = np.asarray(s, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = (s >= float(s_min)) & (s <= float(s_max)) & np.isfinite(y)
    if not np.any(m):
        return float("nan")
    return float(np.nanmax(np.abs(y[m])))


def _path_for(*, sample: str, caps: str, dist: str, zbin: str, suffix: str) -> Path:
    base = f"cosmology_bao_xi_from_catalogs_{sample}_{caps}_{dist}_{zbin}"
    if suffix:
        base = f"{base}{suffix}"
    return _ROOT / "output" / "cosmology" / f"{base}.npz"


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def _xi_l_from_xi_mu(*, xi_mu: np.ndarray, mu_edges: np.ndarray, ell: int) -> np.ndarray:
    xi_mu = np.asarray(xi_mu, dtype=np.float64)
    mu_edges = np.asarray(mu_edges, dtype=np.float64).reshape(-1)
    if xi_mu.ndim != 2:
        raise ValueError("xi_mu must be 2D (nbins,nmu)")
    if mu_edges.ndim != 1 or mu_edges.size < 2:
        raise ValueError("mu_edges must be 1D (nmu+1)")
    nmu = int(mu_edges.size - 1)
    if xi_mu.shape[1] != nmu:
        raise ValueError(f"xi_mu nmu mismatch: xi_mu={xi_mu.shape[1]} mu_edges={nmu}")

    mu_mid = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    dmu = np.diff(mu_edges)
    if not np.allclose(dmu, float(dmu[0]), rtol=0.0, atol=1e-12):
        # Should not happen for our Corrfunc configs, but support just in case.
        w = dmu[None, :]
    else:
        w = float(dmu[0])
    pL = _legendre_p(int(ell), mu_mid)
    if isinstance(w, float):
        integ = np.sum(xi_mu * pL[None, :], axis=1) * w
    else:
        integ = np.sum(xi_mu * pL[None, :] * w, axis=1)
    return (float(2 * int(ell) + 1) * integ).astype(np.float64)


def _try_load_recon_gap_rmse() -> Dict[str, Any]:
    path = _ROOT / "output" / "cosmology" / "cosmology_bao_recon_gap_summary_metrics.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("results", {}) if isinstance(data, dict) else {}
    except Exception:
        return {}


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Quantify window mixing impact (xi0<->xi2) from catalog-based RR/SS grids.")
    ap.add_argument("--sample", default="cmasslowztot", help="sample (default: cmasslowztot)")
    ap.add_argument("--caps", default="combined", help="caps (default: combined)")
    ap.add_argument("--dist", default="lcdm", help="distance model (default: lcdm)")
    ap.add_argument("--z-bins", default="b1,b2,b3", help="comma-separated z-bins (default: b1,b2,b3)")
    ap.add_argument(
        "--suffixes",
        default=";__recon_grid_iso;__recon_grid_ani_rsdshift0",
        help="semicolon-separated NPZ suffixes to include (default: ';__recon_grid_iso;__recon_grid_ani_rsdshift0')",
    )
    ap.add_argument("--s-min", type=float, default=30.0, help="eval s_min (default: 30)")
    ap.add_argument("--s-max", type=float, default=150.0, help="eval s_max (default: 150)")
    ap.add_argument(
        "--out-png",
        default=str(_ROOT / "output" / "cosmology" / "cosmology_bao_catalog_window_mixing.png"),
    )
    ap.add_argument(
        "--out-json",
        default=str(_ROOT / "output" / "cosmology" / "cosmology_bao_catalog_window_mixing_metrics.json"),
    )
    args = ap.parse_args(argv)

    sample = str(args.sample)
    caps = str(args.caps)
    dist = str(args.dist)
    zbins = [z.strip() for z in str(args.z_bins).split(",") if z.strip()]
    suffixes_raw = [s for s in str(args.suffixes).split(";")]
    suffixes: list[str] = []
    for s in suffixes_raw:
        s2 = s.strip()
        if not s2:
            suffixes.append("")
            continue
        if not s2.startswith("__"):
            s2 = "__" + s2
        suffixes.append(s2)

    out_png = Path(str(args.out_png))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_json = Path(str(args.out_json))
    out_json.parent.mkdir(parents=True, exist_ok=True)

    rmse_lookup = _try_load_recon_gap_rmse()

    loaded: Dict[str, Dict[str, Dict[str, Any]]] = {z: {} for z in zbins}
    missing: list[str] = []
    for zbin in zbins:
        for suf in suffixes:
            p = _path_for(sample=sample, caps=caps, dist=dist, zbin=zbin, suffix=suf)
            if not p.exists():
                missing.append(str(p))
                continue
            try:
                z = _load_npz(p)
                if "s" not in z or "mu_edges" not in z or "rr_w" not in z or "xi0" not in z or "xi2" not in z:
                    raise ValueError("missing required keys")
                loaded[zbin][suf or ""] = {"path": str(p), "npz": z}
            except Exception:
                missing.append(str(p))

    any_loaded = any(bool(loaded[z]) for z in zbins)
    if not any_loaded:
        print("[skip] cosmology_bao_catalog_window_mixing: no input npz found")
        return 0

    # Compute mixing + derived diagnostics
    results: Dict[str, Any] = {}
    for zbin in zbins:
        out_z: Dict[str, Any] = {}
        for suf_key, payload in loaded[zbin].items():
            z = payload["npz"]
            s = np.asarray(z["s"], dtype=np.float64)
            mu_edges = np.asarray(z["mu_edges"], dtype=np.float64)
            rr = np.asarray(z["rr_w"], dtype=np.float64)
            xi0 = np.asarray(z["xi0"], dtype=np.float64)
            xi2 = np.asarray(z["xi2"], dtype=np.float64)
            xi4: np.ndarray | None = None
            try:
                if "xi_mu" in z:
                    xi4 = _xi_l_from_xi_mu(xi_mu=z["xi_mu"], mu_edges=mu_edges, ell=4)
            except Exception:
                xi4 = None

            w_rr = _window_wL(counts=rr, mu_edges=mu_edges)
            mix_rr = _mixing_from_w(w_rr)

            # Leakage-only estimate: assume ξ2_true≈0, estimate ξ2 produced by window mixing of ξ0.
            xi2_leak = mix_rr.m20 * xi0
            xi2_leak_4 = (mix_rr.m24 * xi4) if (xi4 is not None) else None

            # Distortion estimate: apply (M-I) to current (ξ0, ξ2) as a small-mixing proxy.
            xi2_mixed = mix_rr.m20 * xi0 + mix_rr.m22 * xi2
            delta_xi2 = xi2_mixed - xi2
            xi2_mixed_024 = (mix_rr.m20 * xi0 + mix_rr.m22 * xi2 + mix_rr.m24 * xi4) if (xi4 is not None) else None

            # Recon has SS grid (shifted randoms); compute the same proxies for reference.
            ss_entry: Dict[str, Any] | None = None
            if "ss_w" in z and np.asarray(z["ss_w"]).size > 0:
                ss = np.asarray(z["ss_w"], dtype=np.float64)
                w_ss = _window_wL(counts=ss, mu_edges=mu_edges)
                mix_ss = _mixing_from_w(w_ss)
                xi2_leak_ss = mix_ss.m20 * xi0
                delta_xi2_ss = (mix_ss.m20 * xi0 + mix_ss.m22 * xi2) - xi2
                xi2_leak4_ss = (mix_ss.m24 * xi4) if (xi4 is not None) else None
                delta_xi2_024_ss = (
                    (mix_ss.m20 * xi0 + mix_ss.m22 * xi2 + mix_ss.m24 * xi4) - xi2 if (xi4 is not None) else None
                )
                ss_entry = {
                    "max_abs_m20_in_eval_range": _safe_max_abs_in_range(
                        s, mix_ss.m20, s_min=float(args.s_min), s_max=float(args.s_max)
                    ),
                    "max_abs_m22_minus1_in_eval_range": _safe_max_abs_in_range(
                        s, mix_ss.m22 - 1.0, s_min=float(args.s_min), s_max=float(args.s_max)
                    ),
                    "max_abs_m24_in_eval_range": _safe_max_abs_in_range(
                        s, mix_ss.m24, s_min=float(args.s_min), s_max=float(args.s_max)
                    ),
                    "max_abs_s2_xi2_leak_in_eval_range": _safe_max_abs_in_range(
                        s, (s * s) * xi2_leak_ss, s_min=float(args.s_min), s_max=float(args.s_max)
                    ),
                    **(
                        {
                            "max_abs_s2_xi2_leak4_in_eval_range": _safe_max_abs_in_range(
                                s,
                                (s * s) * np.asarray(xi2_leak4_ss, dtype=np.float64),
                                s_min=float(args.s_min),
                                s_max=float(args.s_max),
                            ),
                            "max_abs_s2_delta_xi2_024_in_eval_range": _safe_max_abs_in_range(
                                s,
                                (s * s) * np.asarray(delta_xi2_024_ss, dtype=np.float64),
                                s_min=float(args.s_min),
                                s_max=float(args.s_max),
                            ),
                        }
                        if (xi2_leak4_ss is not None and delta_xi2_024_ss is not None)
                        else {}
                    ),
                    "max_abs_s2_delta_xi2_in_eval_range": _safe_max_abs_in_range(
                        s, (s * s) * delta_xi2_ss, s_min=float(args.s_min), s_max=float(args.s_max)
                    ),
                }

            # Optional: compare with recon gap RMSE if available.
            rmse_key = suf_key.lstrip("_") if suf_key else None
            rmse_s2_xi2: float | None = None
            if rmse_key and rmse_key in rmse_lookup:
                try:
                    rmse_s2_xi2 = float(rmse_lookup[rmse_key]["rmse_s2_xi2_ross_post_recon__catalog_recon"][str(int(zbin[-1]))])
                except Exception:
                    rmse_s2_xi2 = None

            out_z[suf_key or "pre"] = {
                "npz": payload.get("path"),
                "xi4_available": bool(xi4 is not None),
                "rr0": {
                    "max_abs_w2_in_eval_range": _safe_max_abs_in_range(
                        s, w_rr.get(2, np.nan), s_min=float(args.s_min), s_max=float(args.s_max)
                    ),
                    "max_abs_w4_in_eval_range": _safe_max_abs_in_range(
                        s, w_rr.get(4, np.nan), s_min=float(args.s_min), s_max=float(args.s_max)
                    ),
                    "max_abs_w6_in_eval_range": _safe_max_abs_in_range(
                        s, w_rr.get(6, np.nan), s_min=float(args.s_min), s_max=float(args.s_max)
                    ),
                    "max_abs_w8_in_eval_range": _safe_max_abs_in_range(
                        s, w_rr.get(8, np.nan), s_min=float(args.s_min), s_max=float(args.s_max)
                    ),
                    "max_abs_m20_in_eval_range": _safe_max_abs_in_range(
                        s, mix_rr.m20, s_min=float(args.s_min), s_max=float(args.s_max)
                    ),
                    "max_abs_m22_minus1_in_eval_range": _safe_max_abs_in_range(
                        s, mix_rr.m22 - 1.0, s_min=float(args.s_min), s_max=float(args.s_max)
                    ),
                    "max_abs_m24_in_eval_range": _safe_max_abs_in_range(
                        s, mix_rr.m24, s_min=float(args.s_min), s_max=float(args.s_max)
                    ),
                    "max_abs_s2_xi2_leak_in_eval_range": _safe_max_abs_in_range(
                        s, (s * s) * xi2_leak, s_min=float(args.s_min), s_max=float(args.s_max)
                    ),
                    **(
                        {
                            "max_abs_s2_xi2_leak4_in_eval_range": _safe_max_abs_in_range(
                                s,
                                (s * s) * np.asarray(xi2_leak_4, dtype=np.float64),
                                s_min=float(args.s_min),
                                s_max=float(args.s_max),
                            ),
                            "max_abs_s2_delta_xi2_024_in_eval_range": _safe_max_abs_in_range(
                                s,
                                (s * s) * np.asarray((xi2_mixed_024 - xi2), dtype=np.float64),
                                s_min=float(args.s_min),
                                s_max=float(args.s_max),
                            ),
                        }
                        if (xi2_leak_4 is not None and xi2_mixed_024 is not None)
                        else {}
                    ),
                    "max_abs_s2_delta_xi2_in_eval_range": _safe_max_abs_in_range(
                        s, (s * s) * delta_xi2, s_min=float(args.s_min), s_max=float(args.s_max)
                    ),
                    "recon_gap_rmse_s2_xi2_ross_post_recon": rmse_s2_xi2,
                },
                **({"ss": ss_entry} if ss_entry is not None else {}),
            }
        results[zbin] = out_z

    # Plot: (left) m20(s) proxy, (right) s^2 * xi2_leak(s) from xi0 (and optionally xi4)
    _set_japanese_font()
    import matplotlib.pyplot as plt  # noqa: E402

    fig, axes = plt.subplots(len(zbins), 2, figsize=(12, 8), sharex=True)
    if len(zbins) == 1:
        axes = np.asarray([axes])

    colors = {
        "": "#7f7f7f",  # pre
        "__recon_grid_iso": "#1f77b4",
        "__recon_grid_ani_rsdshift0": "#ff7f0e",
    }

    for i, zbin in enumerate(zbins):
        ax_m = axes[i, 0]
        ax_y = axes[i, 1]
        ax_m.axhline(0.0, color="#999999", linewidth=1.0, linestyle="--")
        ax_y.axhline(0.0, color="#999999", linewidth=1.0, linestyle="--")

        variants = loaded[zbin]
        for suf_key, payload in variants.items():
            z = payload["npz"]
            s = np.asarray(z["s"], dtype=np.float64)
            mu_edges = np.asarray(z["mu_edges"], dtype=np.float64)
            rr = np.asarray(z["rr_w"], dtype=np.float64)
            xi0 = np.asarray(z["xi0"], dtype=np.float64)
            xi4 = None
            try:
                if "xi_mu" in z:
                    xi4 = _xi_l_from_xi_mu(xi_mu=z["xi_mu"], mu_edges=mu_edges, ell=4)
            except Exception:
                xi4 = None
            w_rr = _window_wL(counts=rr, mu_edges=mu_edges)
            mix_rr = _mixing_from_w(w_rr)

            c = colors.get(suf_key, None) or "#2ca02c"
            label = suf_key.lstrip("_") if suf_key else "pre"
            ax_m.plot(s, mix_rr.m20, color=c, linewidth=1.5, alpha=0.9, label=label)

            xi2_leak = mix_rr.m20 * xi0
            ax_y.plot(s, (s * s) * xi2_leak, color=c, linewidth=1.5, alpha=0.9, label=label)
            if xi4 is not None:
                ax_y.plot(
                    s,
                    (s * s) * (mix_rr.m24 * xi4),
                    color=c,
                    linewidth=1.2,
                    alpha=0.6,
                    linestyle=":",
                    label=(label + "_xi4") if i == 0 else None,
                )

        ax_m.set_title(f"{zbin}: window mixing 係数 m20（ξ0→ξ2 への漏れ）")
        ax_y.set_title(f"{zbin}: 推定漏れ s²(m20·ξ0)")
        ax_m.grid(True, alpha=0.3)
        ax_y.grid(True, alpha=0.3)
        if i == len(zbins) - 1:
            ax_m.set_xlabel("s [Mpc/h]")
            ax_y.set_xlabel("s [Mpc/h]")
        ax_m.set_ylabel("m20(s)")
        ax_y.set_ylabel("s² ξ2_leak")

    axes[0, 0].legend(fontsize=8, loc="upper right", framealpha=0.9)
    fig.suptitle(f"BAO 窓関数：ξ0↔ξ2 の window mixing 規模（{sample}/{caps}/{dist}）", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=160)
    plt.close(fig)

    metrics: Dict[str, Any] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "domain": "cosmology",
        "step": "4.5B (window mixing diagnostic)",
        "inputs": {
            "sample": sample,
            "caps": caps,
            "dist": dist,
            "z_bins": zbins,
            "suffixes": suffixes,
            "s_eval_range_mpc_over_h": [float(args.s_min), float(args.s_max)],
            "npz_missing": missing[:50],  # cap
            "recon_gap_summary_metrics_json": str(
                _ROOT / "output" / "cosmology" / "cosmology_bao_recon_gap_summary_metrics.json"
            ),
        },
        "results": results,
        "notes": [
            "wL は RR(s,μ) の multipoles 比（R_L/R_0）から推定。",
            "m20 は window による ξ0→ξ2 の混合（漏れ）の係数。",
            "m24 は window による ξ4→ξ2 の混合（漏れ）の係数（xi_mu がある場合のみ評価）。",
            "xi2_leak は ξ2_true≈0 と仮定したときの漏れ（order-of-magnitude）。",
            "xi2_leak4 は xi4 を同様に用いた漏れ（order-of-magnitude）。",
            "delta_xi2 は (M-I)·(ξ0,ξ2) を用いた小混合近似（true 不明のため参考）。",
        ],
        "outputs": {"png": str(out_png).replace("\\", "/"), "metrics_json": str(out_json).replace("\\", "/")},
    }
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    worklog.append_event(
        {
            "event": "cosmology_bao_catalog_window_mixing",
            "generated_utc": metrics["generated_utc"],
            "inputs": metrics["inputs"],
            "outputs": {"png": str(out_png), "json": str(out_json)},
        }
    )

    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
