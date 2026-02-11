#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xrism_cluster_redshift_turbulence.py

Phase 4 / Step 4.8.3（銀河団）:
XRISM（Resolve）の公開一次データ（PI + RMF）を直接再解析し、輝線の centroid と線幅から
z（距離指標非依存）と速度分散 σ_v（turbulence proxy）を固定出力化する。

前提:
- 解析前に `scripts/xrism/fetch_xrism_heasarc.py` で products をキャッシュしておく。
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None

try:
    from scipy.optimize import curve_fit  # type: ignore
except Exception:  # pragma: no cover
    curve_fit = None

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.cosmology.boss_dr12v5_fits import read_bintable_columns, read_first_bintable_layout  # noqa: E402
from scripts.summary import worklog  # noqa: E402

_CARD = 80
_BLOCK = 2880
_C_KMS = 299_792.458


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _relpath(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    try:
        return path.relative_to(_ROOT).as_posix()
    except Exception:
        return path.as_posix()


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


def _maybe_bool(x: object) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if not s:
        return None
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    return None


def _combine_in_quadrature(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return math.sqrt(float(a) ** 2 + float(b) ** 2)


def _load_event_level_qc_summary_by_obsid(out_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load event-level QC summary (products vs event_cl) keyed by obsid.
    Used as an additional "procedure-difference" systematic term for centroid/z.
    """
    path = out_dir / "xrism_event_level_qc_summary.csv"
    rows = _read_csv_rows(path)
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        obsid = str(r.get("obsid") or "").strip()
        if not obsid:
            continue
        out[obsid] = {
            "obsid": obsid,
            "l1_norm_a": _maybe_float(r.get("l1_norm_a")),
            "mean_shift_keV_event_minus_products": _maybe_float(r.get("mean_shift_keV")),
            "pixel_exclude": str(r.get("pixel_exclude") or "").strip(),
            "apply_gti": bool(_maybe_bool(r.get("apply_gti")) or False),
            "gti_n": _maybe_float(r.get("gti_n")),
            "note": "Fe-K帯域（5.5–7.5 keV）での products（PI） vs event_cl ヒストグラム差（平均エネルギー差）。line fit の追加系統（手続き差）として扱う。",
        }
    return out


def _iter_cards_from_header_bytes(header_bytes: bytes) -> Iterable[str]:
    for i in range(0, len(header_bytes), _CARD):
        yield header_bytes[i : i + _CARD].decode("ascii", errors="ignore")


def _read_exact(f, n: int) -> bytes:
    b = f.read(n)
    if b is None:
        return b""
    return b


def _read_header_blocks(f) -> bytes:
    chunks: List[bytes] = []
    while True:
        block = _read_exact(f, _BLOCK)
        if len(block) != _BLOCK:
            raise EOFError("unexpected EOF while reading FITS header")
        chunks.append(block)
        for card in _iter_cards_from_header_bytes(block):
            if card.startswith("END"):
                return b"".join(chunks)


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


_TFORM_RE = re.compile(r"^\s*(?P<rep>\d*)(?P<code>[A-Z])\s*$")


def _tform_to_numpy_dtype(tform: str) -> Tuple[np.dtype, int, int]:
    m = _TFORM_RE.match(tform)
    if not m:
        raise ValueError(f"unsupported TFORM: {tform!r}")
    rep = int(m.group("rep") or "1")
    code = m.group("code")
    if rep < 1:
        raise ValueError(f"invalid repeat in TFORM: {tform!r}")
    if code == "I":
        return np.dtype(">i2"), rep, 2 * rep
    if code == "J":
        return np.dtype(">i4"), rep, 4 * rep
    if code == "K":
        return np.dtype(">i8"), rep, 8 * rep
    if code == "E":
        return np.dtype(">f4"), rep, 4 * rep
    if code == "D":
        return np.dtype(">f8"), rep, 8 * rep
    if code == "A":
        return np.dtype(f"S{rep}"), rep, rep
    if code == "B":
        return np.dtype("u1"), rep, 1 * rep
    if code == "L":
        return np.dtype("S1"), rep, 1 * rep
    raise ValueError(f"unsupported TFORM code: {code!r} (tform={tform!r})")


def _skip_hdu_data(f, header_kv: Dict[str, str]) -> None:
    naxis = int(header_kv.get("NAXIS", "0") or "0")
    if naxis <= 0:
        return
    naxis1 = int(header_kv.get("NAXIS1", "0") or "0")
    naxis2 = int(header_kv.get("NAXIS2", "0") or "0")
    pcount = int(header_kv.get("PCOUNT", "0") or "0")
    gcount = int(header_kv.get("GCOUNT", "1") or "1")
    data_bytes = naxis1 * naxis2 * max(gcount, 1) + max(pcount, 0)
    pad = ((int(data_bytes) + _BLOCK - 1) // _BLOCK) * _BLOCK
    if pad > 0:
        f.seek(pad, 1)


def _read_ebounds_table(rmf_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (channel[int], e_mid_keV[float]) from RMF's EBOUNDS extension.
    """
    opener = gzip.open if rmf_path.name.endswith(".gz") else Path.open
    with opener(rmf_path, "rb") as f:  # type: ignore[arg-type]
        _ = _read_header_blocks(f)  # primary
        while True:
            hdr = _read_header_blocks(f)
            kv = _parse_header_kv(hdr)
            extname = kv.get("EXTNAME", "").strip().strip("'").strip()
            if extname.upper() == "EBOUNDS":
                row_bytes = int(kv.get("NAXIS1", "0") or "0")
                n_rows = int(kv.get("NAXIS2", "0") or "0")
                tfields = int(kv.get("TFIELDS", "0") or "0")
                if row_bytes <= 0 or n_rows <= 0 or tfields <= 0:
                    raise ValueError("invalid EBOUNDS header (missing NAXIS1/NAXIS2/TFIELDS)")

                ttype: Dict[int, str] = {}
                tform: Dict[int, str] = {}
                for card in _iter_cards_from_header_bytes(hdr):
                    key = card[:8].strip()
                    if key.startswith("TTYPE"):
                        try:
                            i = int(key[5:])
                        except Exception:
                            continue
                        v = card.split("=", 1)[1].split("/", 1)[0].strip()
                        if len(v) >= 2 and v[0] == "'" and v[-1] == "'":
                            v = v[1:-1]
                        ttype[i] = v.strip()
                    elif key.startswith("TFORM"):
                        try:
                            i = int(key[5:])
                        except Exception:
                            continue
                        v = card.split("=", 1)[1].split("/", 1)[0].strip()
                        if len(v) >= 2 and v[0] == "'" and v[-1] == "'":
                            v = v[1:-1]
                        tform[i] = v.strip()

                columns: List[str] = []
                offsets: Dict[str, int] = {}
                formats: Dict[str, str] = {}
                off = 0
                for i in range(1, tfields + 1):
                    name = ttype.get(i)
                    fmt = tform.get(i)
                    if name is None or fmt is None:
                        raise ValueError(f"missing TTYPE/TFORM for field {i}")
                    _, rep, width = _tform_to_numpy_dtype(fmt)
                    if rep != 1:
                        raise ValueError("EBOUNDS repeat!=1 is not supported")
                    columns.append(name)
                    offsets[name] = off
                    formats[name] = fmt
                    off += width
                if off != row_bytes:
                    raise ValueError(f"row size mismatch in EBOUNDS: {off} != {row_bytes}")

                names: List[str] = []
                fmts: List[np.dtype] = []
                offs: List[int] = []
                for name in columns:
                    dt, rep, _ = _tform_to_numpy_dtype(formats[name])
                    if rep != 1:
                        raise ValueError("EBOUNDS repeat!=1 is not supported")
                    names.append(name)
                    fmts.append(dt)
                    offs.append(int(offsets[name]))
                dt_struct = np.dtype({"names": names, "formats": fmts, "offsets": offs, "itemsize": row_bytes})

                b = _read_exact(f, row_bytes * n_rows)
                if len(b) != row_bytes * n_rows:
                    raise EOFError("unexpected EOF while reading EBOUNDS data")
                arr = np.frombuffer(b, dtype=dt_struct, count=n_rows)
                col_map = {str(c).upper(): str(c) for c in columns}
                ch_key = col_map.get("CHANNEL")
                e_min_key = col_map.get("E_MIN")
                e_max_key = col_map.get("E_MAX")
                if ch_key is None or e_min_key is None or e_max_key is None:
                    raise ValueError("EBOUNDS missing CHANNEL/E_MIN/E_MAX")
                ch = np.asarray(arr[ch_key], dtype=int)
                e_min = np.asarray(arr[e_min_key], dtype=float)
                e_max = np.asarray(arr[e_max_key], dtype=float)
                return ch, 0.5 * (e_min + e_max)

            _skip_hdu_data(f, kv)


def _load_pi_spectrum(pi_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    opener = gzip.open if pi_path.name.endswith(".gz") else Path.open
    with opener(pi_path, "rb") as f:  # type: ignore[arg-type]
        layout = read_first_bintable_layout(f)
        col_map = {str(c).upper(): str(c) for c in layout.columns}
        ch_key = col_map.get("CHANNEL")
        cnt_key = col_map.get("COUNTS")
        if ch_key is None or cnt_key is None:
            raise ValueError("PI missing CHANNEL/COUNTS")
        cols = read_bintable_columns(f, layout=layout, columns=[ch_key, cnt_key])
        return np.asarray(cols[ch_key], dtype=int), np.asarray(cols[cnt_key], dtype=float)


def _find_local_obs_root(data_root: Path, obsid: str) -> Tuple[Optional[str], Optional[Path]]:
    direct = data_root / obsid
    if direct.is_dir():
        return "direct", direct
    for cand in sorted(data_root.glob(f"*/{obsid}")):
        if cand.is_dir():
            return cand.parent.name, cand
    return None, None


_PX_RE = re.compile(r"px(?P<px>\d+)", flags=re.IGNORECASE)


def _px_score(name: str) -> Tuple[int, int, str]:
    m = _PX_RE.search(name)
    px = int(m.group("px")) if m else 10**9
    pref = {1000: 0, 0: 1, 5000: 2}.get(px, 9)
    return pref, px, name


def _choose_pi_rmf_pair(products_dir: Path) -> Tuple[Path, Path]:
    pis = sorted(products_dir.glob("*_src.pi*"))
    if not pis:
        raise FileNotFoundError(f"no *_src.pi* under {products_dir}")
    pairs: List[Tuple[Tuple[int, int, str], Path, Path]] = []
    for pi in pis:
        rmf_name = re.sub(r"_src\.pi(\.gz)?$", r".rmf\1", pi.name)
        rmf = products_dir / rmf_name
        if rmf.exists():
            pairs.append((_px_score(pi.name), pi, rmf))
    if not pairs:
        rmfs = sorted(products_dir.glob("*.rmf*"))
        if not rmfs:
            raise FileNotFoundError(f"no rmf found under {products_dir}")
        return pis[0], rmfs[0]
    pairs.sort(key=lambda x: x[0])
    return pairs[0][1], pairs[0][2]


def _rebin_min_counts(energy: np.ndarray, counts: np.ndarray, *, min_counts: float) -> Tuple[np.ndarray, np.ndarray]:
    if min_counts <= 0:
        return energy, counts
    out_e: List[float] = []
    out_c: List[float] = []
    acc_c = 0.0
    acc_ec = 0.0
    for e, c in zip(energy, counts):
        if not np.isfinite(e) or not np.isfinite(c):
            continue
        acc_c += float(c)
        acc_ec += float(e) * float(c)
        if acc_c >= float(min_counts):
            out_c.append(acc_c)
            out_e.append(acc_ec / acc_c if acc_c > 0 else float(e))
            acc_c = 0.0
            acc_ec = 0.0
    if acc_c > 0:
        out_c.append(acc_c)
        out_e.append(acc_ec / acc_c if acc_c > 0 else float(energy[-1]))
    return np.asarray(out_e, dtype=float), np.asarray(out_c, dtype=float)


def _model_counts_const_em_gauss(E: np.ndarray, c0: float, amp: float, centroid: float, sigma: float) -> np.ndarray:
    """
    counts(E) = c0 + amp * exp(-(E-centroid)^2/(2*sigma^2)).
    Local (narrow-window) baseline model for cluster emission lines.
    """
    E = np.asarray(E, dtype=float)
    prof = np.exp(-0.5 * np.square((E - float(centroid)) / max(float(sigma), 1e-6)))
    return float(c0) + float(amp) * prof


def _fit_emission_line(
    energy_keV: np.ndarray,
    counts: np.ndarray,
    *,
    window_keV: Tuple[float, float],
    centroid_bounds_keV: Tuple[float, float],
    min_counts: float,
) -> Dict[str, Any]:
    if curve_fit is None:
        raise RuntimeError("scipy is required for fitting")

    e0, e1 = float(window_keV[0]), float(window_keV[1])
    m = (energy_keV >= min(e0, e1)) & (energy_keV <= max(e0, e1)) & np.isfinite(energy_keV) & np.isfinite(counts)
    e = np.asarray(energy_keV[m], dtype=float)
    y = np.asarray(counts[m], dtype=float)
    if e.size < 10 or float(np.nansum(y)) <= 0.0:
        return {"ok": False, "reason": "insufficient data in window"}

    e, y = _rebin_min_counts(e, y, min_counts=min_counts)
    if e.size < 8:
        return {"ok": False, "reason": "insufficient bins after rebin"}

    # initial guesses
    base0 = float(np.nanmedian(y)) if np.isfinite(np.nanmedian(y)) else 0.0
    idx = int(np.nanargmax(y))
    c0 = float(np.clip(float(e[idx]), centroid_bounds_keV[0], centroid_bounds_keV[1]))
    amp0 = float(max(0.0, float(np.nanmax(y) - base0)))
    sigma0 = 0.01
    p0 = [base0, amp0, c0, sigma0]

    lower = [0.0, 0.0, centroid_bounds_keV[0], 0.001]
    upper = [float("inf"), float("inf"), centroid_bounds_keV[1], 0.2]
    err = np.sqrt(np.clip(y, 0.0, None) + 1.0)

    try:
        popt, pcov = curve_fit(
            _model_counts_const_em_gauss,
            e,
            y,
            p0=p0,
            bounds=(lower, upper),
            sigma=err,
            absolute_sigma=True,
            maxfev=20000,
        )
    except Exception as ex:
        return {"ok": False, "reason": f"fit failed: {ex}"}

    model = _model_counts_const_em_gauss(e, *popt)
    resid = (y - model) / err
    rss = float(np.nansum(resid**2))
    dof = int(max(0, e.size - len(popt)))
    chi2_red = float(rss / dof) if dof > 0 else float("nan")

    perr = np.full(len(popt), float("nan"))
    if pcov is not None and np.all(np.isfinite(pcov)):
        perr = np.sqrt(np.clip(np.diag(pcov), 0.0, None))

    c_base, amp, centroid, sigma = [float(x) for x in popt]
    amp_err = float(perr[1]) if np.isfinite(perr[1]) else float("nan")
    centroid_err = float(perr[2]) if np.isfinite(perr[2]) else float("nan")
    sigma_err = float(perr[3]) if np.isfinite(perr[3]) else float("nan")
    detected = bool(np.isfinite(amp_err) and amp_err > 0 and (amp / amp_err) >= 3.0)

    return {
        "ok": True,
        "n_bins": int(e.size),
        "window_keV": [e0, e1],
        "min_counts": float(min_counts),
        "params": {
            "c0": c_base,
            "amp": amp,
            "centroid_keV": centroid,
            "sigma_keV": sigma,
        },
        "errors_1sigma": {
            "amp": float(perr[1]) if np.isfinite(perr[1]) else None,
            "centroid_keV": float(perr[2]) if np.isfinite(perr[2]) else None,
            "sigma_keV": float(perr[3]) if np.isfinite(perr[3]) else None,
        },
        "fit_quality": {"chi2_red": chi2_red, "dof": dof},
        "detected": detected,
        "plot": {"energy_keV": e, "counts": y, "model": model},
    }


def _z_from_centroid(*, E_rest_keV: float, centroid_keV: float) -> Optional[float]:
    if not np.isfinite(E_rest_keV) or E_rest_keV <= 0:
        return None
    if not np.isfinite(centroid_keV) or centroid_keV <= 0:
        return None
    return float(E_rest_keV / centroid_keV - 1.0)


def _z_err_from_centroid_err(*, E_rest_keV: float, centroid_keV: float, centroid_err_keV: float) -> Optional[float]:
    if not np.isfinite(centroid_err_keV) or centroid_err_keV <= 0:
        return None
    if not np.isfinite(E_rest_keV) or E_rest_keV <= 0:
        return None
    if not np.isfinite(centroid_keV) or centroid_keV <= 0:
        return None
    dz_dc = -float(E_rest_keV) / float(centroid_keV) ** 2
    return abs(dz_dc) * float(centroid_err_keV)


def _sigma_instr_keV_from_fwhm_ev(fwhm_ev: float) -> float:
    fwhm_ev = float(fwhm_ev)
    if not np.isfinite(fwhm_ev) or fwhm_ev <= 0:
        return float("nan")
    return (fwhm_ev / 2.355) / 1000.0


def _sigma_v_kms_from_sigma_E(*, centroid_keV: float, sigma_E_keV: float) -> Optional[float]:
    if not np.isfinite(centroid_keV) or centroid_keV <= 0:
        return None
    if not np.isfinite(sigma_E_keV) or sigma_E_keV < 0:
        return None
    return _C_KMS * float(sigma_E_keV) / float(centroid_keV)


def _window_sweep(base: Tuple[float, float]) -> List[Tuple[float, float]]:
    lo, hi = float(base[0]), float(base[1])
    return [(max(0.1, lo - 0.2), hi + 0.2), (lo, hi), (lo + 0.2, hi - 0.2)]


def _gain_sweep() -> List[float]:
    return [-1e-3, 0.0, +1e-3]


@dataclass(frozen=True)
class LineSpec:
    line_id: str
    E_rest_keV: float
    window_halfwidth_keV: float
    centroid_halfwidth_keV: float


_LINES: List[LineSpec] = [
    LineSpec("FeXXV_HeA", 6.700, 0.35, 0.20),
    LineSpec("FeXXVI_LyA", 6.966, 0.35, 0.20),
]


def _plot_fit(out_png: Path, *, energy: np.ndarray, counts: np.ndarray, model: np.ndarray, title: str) -> Optional[str]:
    if plt is None:
        return "matplotlib is not available"
    try:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig = plt.figure(figsize=(10, 4.2), dpi=140)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(energy, counts, lw=0.8, label="data")
        ax.plot(energy, model, lw=1.0, label="model")
        ax.set_xlabel("energy (keV)")
        ax.set_ylabel("counts (rebinned)")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(out_png)
        plt.close(fig)
        return None
    except Exception as e:
        return str(e)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--targets", default=str(_ROOT / "output" / "xrism" / "xrism_targets_catalog.csv"))
    p.add_argument("--obsid", action="append", default=[], help="override: obsid(s) to analyze (repeatable)")
    p.add_argument("--role", default="cluster", help="targets role to include when --obsid is not provided")
    p.add_argument("--min-counts", type=float, default=30.0, help="min counts per rebinned bin (base)")
    p.add_argument("--instr-fwhm-ev", type=float, default=5.0, help="Resolve energy resolution (FWHM; eV) for σ_v deconvolution")
    p.add_argument("--out-dir", default=str(_ROOT / "output" / "xrism"))
    p.add_argument("--data-root", default=str(_ROOT / "data" / "xrism" / "heasarc" / "obs"))
    args = p.parse_args(list(argv) if argv is not None else None)

    targets_csv = Path(args.targets)
    rows = _read_csv_rows(targets_csv)

    if args.obsid:
        obsids = [str(x).strip() for x in args.obsid if str(x).strip()]
    else:
        obsids = []
        for r in rows:
            if (r.get("role") or "").strip() != str(args.role):
                continue
            o = (r.get("obsid") or "").strip()
            if o:
                obsids.append(o)
    obsids = sorted(set(obsids))
    if not obsids:
        print(f"[warn] no obsids found (targets={targets_csv})")
        return 0

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    qc_by_obsid = _load_event_level_qc_summary_by_obsid(out_dir)

    sigma_instr_keV = _sigma_instr_keV_from_fwhm_ev(float(args.instr_fwhm_ev))
    if not np.isfinite(sigma_instr_keV):
        sigma_instr_keV = float("nan")

    summary_rows: List[Dict[str, Any]] = []
    per_obs: Dict[str, Any] = {}

    for obsid in obsids:
        row = next((r for r in rows if (r.get("obsid") or "").strip() == obsid), {})
        target_name = (row.get("target_name") or "").strip()
        z_opt_raw = (row.get("z_sys") or "").strip()
        try:
            z_opt = float(z_opt_raw) if z_opt_raw else float("nan")
        except Exception:
            z_opt = float("nan")

        cat, local_obs_root = _find_local_obs_root(data_root, obsid)
        if local_obs_root is None:
            per_obs[obsid] = {"status": "missing_cache"}
            continue

        products_dir = local_obs_root / "resolve" / "products"
        if not products_dir.is_dir():
            per_obs[obsid] = {"status": "missing_products", "products_dir": _relpath(products_dir)}
            continue

        try:
            pi_path, rmf_path = _choose_pi_rmf_pair(products_dir)
        except Exception as e:
            per_obs[obsid] = {"status": "missing_files", "error": str(e)}
            continue

        ch, counts = _load_pi_spectrum(pi_path)
        rmf_ch, e_mid = _read_ebounds_table(rmf_path)
        max_ch = int(max(int(np.max(rmf_ch)), int(np.max(ch))))
        e_map = np.full(max_ch + 1, np.nan, dtype=float)
        e_map[np.asarray(rmf_ch, dtype=int)] = np.asarray(e_mid, dtype=float)
        energy = e_map[np.asarray(ch, dtype=int)]

        base_min_counts = float(args.min_counts)
        gain_list = _gain_sweep()
        min_counts_list = [base_min_counts, 2.0 * base_min_counts]

        obs_metrics: Dict[str, Any] = {
            "generated_utc": _utc_now(),
            "status": "ok",
            "obsid": obsid,
            "target_name": target_name,
            "z_opt": float(z_opt) if np.isfinite(z_opt) else None,
            "cache": {
                "cat": cat,
                "obs_root": _relpath(local_obs_root),
                "products_dir": _relpath(products_dir),
                "pi": _relpath(pi_path),
                "rmf": _relpath(rmf_path),
            },
            "analysis": {
                "lines": [ls.__dict__ for ls in _LINES],
                "gain_frac_sweep": gain_list,
                "min_counts_sweep": min_counts_list,
                "instr_fwhm_ev": float(args.instr_fwhm_ev),
                "instr_sigma_keV": float(sigma_instr_keV) if np.isfinite(sigma_instr_keV) else None,
                "detection_rule": "amp/σ_amp >= 3",
            },
            "results": {},
        }

        for ls in _LINES:
            if np.isfinite(z_opt):
                E_exp = float(ls.E_rest_keV / (1.0 + float(z_opt)))
            else:
                E_exp = float(ls.E_rest_keV)
            base_window = (max(0.1, E_exp - float(ls.window_halfwidth_keV)), E_exp + float(ls.window_halfwidth_keV))
            centroid_bounds = (
                max(0.1, E_exp - float(ls.centroid_halfwidth_keV)),
                E_exp + float(ls.centroid_halfwidth_keV),
            )
            win_list = _window_sweep(base_window)
            variations: List[Dict[str, Any]] = []
            best: Optional[Dict[str, Any]] = None
            best_chi = float("inf")

            for w in win_list:
                for gain in gain_list:
                    for mc in min_counts_list:
                        e_adj = energy * (1.0 + float(gain))
                        fit = _fit_emission_line(
                            e_adj,
                            counts,
                            window_keV=w,
                            centroid_bounds_keV=centroid_bounds,
                            min_counts=mc,
                        )
                        rec = {
                            "window_keV": [float(w[0]), float(w[1])],
                            "gain_frac": float(gain),
                            "min_counts": float(mc),
                            "fit": {k: v for k, v in fit.items() if k != "plot"},
                        }
                        variations.append(rec)
                        if not fit.get("ok"):
                            continue
                        chi = float(fit.get("fit_quality", {}).get("chi2_red", float("inf")))
                        if np.isfinite(chi) and chi < best_chi:
                            best_chi = chi
                            best = fit
                            best["_best_window_keV"] = [float(w[0]), float(w[1])]
                            best["_best_gain_frac"] = float(gain)
                            best["_best_min_counts"] = float(mc)

            if best is None or not bool(best.get("ok")):
                obs_metrics["results"][ls.line_id] = {"ok": False, "reason": "no successful fit", "variations": variations}
                continue

            centroid = float(best["params"]["centroid_keV"])
            sigma_E = float(best["params"]["sigma_keV"])
            centroid_err = best.get("errors_1sigma", {}).get("centroid_keV")
            sigma_err = best.get("errors_1sigma", {}).get("sigma_keV")
            centroid_err_f = float(centroid_err) if centroid_err is not None else float("nan")
            sigma_err_f = float(sigma_err) if sigma_err is not None else float("nan")

            z_x = _z_from_centroid(E_rest_keV=ls.E_rest_keV, centroid_keV=centroid)
            z_x_err = _z_err_from_centroid_err(
                E_rest_keV=ls.E_rest_keV,
                centroid_keV=centroid,
                centroid_err_keV=centroid_err_f,
            )

            sigma_v_tot = _sigma_v_kms_from_sigma_E(centroid_keV=centroid, sigma_E_keV=sigma_E)
            sigma_v_tot_err = (
                _sigma_v_kms_from_sigma_E(centroid_keV=centroid, sigma_E_keV=sigma_err_f)
                if np.isfinite(sigma_err_f)
                else None
            )

            sigma_intr = float("nan")
            if np.isfinite(sigma_instr_keV):
                sigma_intr = float(max(0.0, sigma_E * sigma_E - sigma_instr_keV * sigma_instr_keV) ** 0.5)
            sigma_v_intr = _sigma_v_kms_from_sigma_E(centroid_keV=centroid, sigma_E_keV=sigma_intr) if np.isfinite(sigma_intr) else None

            delta_z = float(z_x - z_opt) if (z_x is not None and np.isfinite(z_opt)) else None
            delta_v_kms = (_C_KMS * float(delta_z) / (1.0 + float(z_opt))) if (delta_z is not None and np.isfinite(z_opt)) else None

            # systematic scatter over successful fits (window/gain/rebin)
            z_vars: List[float] = []
            centroid_vars: List[float] = []
            sigmaE_vars: List[float] = []
            sigv_intr_vars: List[float] = []
            for rec in variations:
                fit0 = rec.get("fit") or {}
                if not fit0.get("ok"):
                    continue
                c0 = float((fit0.get("params") or {}).get("centroid_keV", float("nan")))
                s0 = float((fit0.get("params") or {}).get("sigma_keV", float("nan")))
                if not np.isfinite(c0) or not np.isfinite(s0):
                    continue
                z0 = _z_from_centroid(E_rest_keV=ls.E_rest_keV, centroid_keV=c0)
                if z0 is not None and np.isfinite(z0):
                    z_vars.append(float(z0))
                centroid_vars.append(c0)
                sigmaE_vars.append(s0)
                if np.isfinite(sigma_instr_keV):
                    s_intr0 = float(max(0.0, s0 * s0 - sigma_instr_keV * sigma_instr_keV) ** 0.5)
                    v_intr0 = _sigma_v_kms_from_sigma_E(centroid_keV=c0, sigma_E_keV=s_intr0)
                    if v_intr0 is not None and np.isfinite(v_intr0):
                        sigv_intr_vars.append(float(v_intr0))

            centroid_sys = float(np.nanstd(centroid_vars, ddof=1)) if len(centroid_vars) >= 3 else float("nan")
            sigmaE_sys = float(np.nanstd(sigmaE_vars, ddof=1)) if len(sigmaE_vars) >= 3 else float("nan")
            z_sys_scatter = float(np.nanstd(z_vars, ddof=1)) if len(z_vars) >= 3 else float("nan")
            sigv_intr_sys = float(np.nanstd(sigv_intr_vars, ddof=1)) if len(sigv_intr_vars) >= 3 else float("nan")

            # Additional systematic from event-level QC (products vs event_cl).
            qc = qc_by_obsid.get(obsid) or {}
            mean_shift_keV = qc.get("mean_shift_keV_event_minus_products")
            mean_shift_keV_f = float(mean_shift_keV) if mean_shift_keV is not None and math.isfinite(float(mean_shift_keV)) else None
            centroid_sys_event = abs(mean_shift_keV_f) if mean_shift_keV_f is not None else None
            z_sys_event = (
                _z_err_from_centroid_err(
                    E_rest_keV=ls.E_rest_keV,
                    centroid_keV=centroid,
                    centroid_err_keV=float(centroid_sys_event),
                )
                if centroid_sys_event is not None
                else None
            )
            centroid_sys_base = centroid_sys if np.isfinite(centroid_sys) else None
            z_sys_base = z_sys_scatter if np.isfinite(z_sys_scatter) else None
            centroid_sys_total = _combine_in_quadrature(centroid_sys_base, centroid_sys_event)
            z_sys_total = _combine_in_quadrature(z_sys_base, z_sys_event)

            obs_metrics["results"][ls.line_id] = {
                "ok": True,
                "E_rest_keV": ls.E_rest_keV,
                "best": {k: v for k, v in best.items() if k != "plot"},
                "derived": {
                    "E_exp_keV": E_exp,
                    "centroid_keV": centroid,
                    "centroid_err_stat_keV": float(centroid_err_f) if np.isfinite(centroid_err_f) else None,
                    "centroid_sys_keV": centroid_sys if np.isfinite(centroid_sys) else None,
                    "centroid_sys_event_level_keV": float(centroid_sys_event) if centroid_sys_event is not None else None,
                    "centroid_sys_total_keV": float(centroid_sys_total) if centroid_sys_total is not None else None,
                    "z_xray": float(z_x) if z_x is not None and np.isfinite(z_x) else None,
                    "z_xray_err_stat": float(z_x_err) if z_x_err is not None and np.isfinite(z_x_err) else None,
                    "z_xray_sys": z_sys_scatter if np.isfinite(z_sys_scatter) else None,
                    "z_xray_sys_event_level": float(z_sys_event) if z_sys_event is not None else None,
                    "z_xray_sys_total": float(z_sys_total) if z_sys_total is not None else None,
                    "sigma_E_keV": sigma_E,
                    "sigma_E_err_stat_keV": float(sigma_err_f) if np.isfinite(sigma_err_f) else None,
                    "sigma_E_sys_keV": sigmaE_sys if np.isfinite(sigmaE_sys) else None,
                    "sigma_v_total_kms": float(sigma_v_tot) if sigma_v_tot is not None and np.isfinite(sigma_v_tot) else None,
                    "sigma_v_total_err_stat_kms": float(sigma_v_tot_err)
                    if sigma_v_tot_err is not None and np.isfinite(sigma_v_tot_err)
                    else None,
                    "sigma_v_intr_kms": float(sigma_v_intr) if sigma_v_intr is not None and np.isfinite(sigma_v_intr) else None,
                    "sigma_v_intr_sys_kms": sigv_intr_sys if np.isfinite(sigv_intr_sys) else None,
                    "delta_z": float(delta_z) if delta_z is not None and np.isfinite(delta_z) else None,
                    "delta_v_kms": float(delta_v_kms) if delta_v_kms is not None and np.isfinite(delta_v_kms) else None,
                },
                "event_level_qc": qc or None,
                "variations": variations,
            }

            summary_rows.append(
                {
                    "obsid": obsid,
                    "target_name": target_name,
                    "z_opt": float(z_opt) if np.isfinite(z_opt) else "",
                    "line_id": ls.line_id,
                    "E_rest_keV": ls.E_rest_keV,
                    "centroid_keV": centroid,
                    "centroid_err_stat_keV": float(centroid_err_f) if np.isfinite(centroid_err_f) else "",
                    "centroid_sys_keV": centroid_sys if np.isfinite(centroid_sys) else "",
                    "centroid_sys_event_level_keV": float(centroid_sys_event) if centroid_sys_event is not None else "",
                    "centroid_sys_total_keV": float(centroid_sys_total) if centroid_sys_total is not None else "",
                    "z_xray": float(z_x) if z_x is not None and np.isfinite(z_x) else "",
                    "z_xray_err_stat": float(z_x_err) if z_x_err is not None and np.isfinite(z_x_err) else "",
                    "z_xray_sys": z_sys_scatter if np.isfinite(z_sys_scatter) else "",
                    "z_xray_sys_event_level": float(z_sys_event) if z_sys_event is not None else "",
                    "z_xray_sys_total": float(z_sys_total) if z_sys_total is not None else "",
                    "delta_z": float(delta_z) if delta_z is not None and np.isfinite(delta_z) else "",
                    "delta_v_kms": float(delta_v_kms) if delta_v_kms is not None and np.isfinite(delta_v_kms) else "",
                    "sigma_E_keV": sigma_E,
                    "sigma_E_err_stat_keV": float(sigma_err_f) if np.isfinite(sigma_err_f) else "",
                    "sigma_E_sys_keV": sigmaE_sys if np.isfinite(sigmaE_sys) else "",
                    "sigma_v_intr_kms": float(sigma_v_intr) if sigma_v_intr is not None and np.isfinite(sigma_v_intr) else "",
                    "sigma_v_intr_sys_kms": sigv_intr_sys if np.isfinite(sigv_intr_sys) else "",
                    "detected": bool(best.get("detected")),
                    "best_window_keV": ",".join(map(str, best.get("_best_window_keV", []))),
                    "best_gain_frac": best.get("_best_gain_frac", ""),
                    "best_min_counts": best.get("_best_min_counts", ""),
                    "pi": _relpath(pi_path),
                    "rmf": _relpath(rmf_path),
                }
            )

            # best-fit plot (optional)
            plot = best.get("plot") or {}
            eplt = plot.get("energy_keV")
            yplt = plot.get("counts")
            mplt = plot.get("model")
            if plt is not None and isinstance(eplt, np.ndarray) and isinstance(yplt, np.ndarray) and isinstance(mplt, np.ndarray):
                out_png = out_dir / f"{obsid}__{ls.line_id}__fit.png"
                _ = _plot_fit(out_png, energy=eplt, counts=yplt, model=mplt, title=f"XRISM {obsid} {ls.line_id} fit")

        # per-obs outputs
        out_csv = out_dir / f"{obsid}__line_fit.csv"
        out_json = out_dir / f"{obsid}__line_fit_metrics.json"
        fieldnames = [
            "obsid",
            "target_name",
            "z_opt",
            "line_id",
            "E_rest_keV",
            "centroid_keV",
            "centroid_err_stat_keV",
            "centroid_sys_keV",
            "centroid_sys_event_level_keV",
            "centroid_sys_total_keV",
            "z_xray",
            "z_xray_err_stat",
            "z_xray_sys",
            "z_xray_sys_event_level",
            "z_xray_sys_total",
            "delta_z",
            "delta_v_kms",
            "sigma_E_keV",
            "sigma_E_err_stat_keV",
            "sigma_E_sys_keV",
            "sigma_v_intr_kms",
            "sigma_v_intr_sys_kms",
            "detected",
            "best_window_keV",
            "best_gain_frac",
            "best_min_counts",
            "pi",
            "rmf",
        ]
        per_rows = [r for r in summary_rows if r.get("obsid") == obsid]
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in per_rows:
                w.writerow({k: r.get(k, "") for k in fieldnames})
        _write_json(out_json, obs_metrics)
        per_obs[obsid] = {"status": "ok", "line_fit_csv": _relpath(out_csv), "metrics_json": _relpath(out_json)}

    # summary outputs
    out_sum_csv = out_dir / "xrism_cluster_redshift_turbulence_summary.csv"
    out_sum_json = out_dir / "xrism_cluster_redshift_turbulence_summary_metrics.json"
    fieldnames_sum = [
        "obsid",
        "target_name",
        "z_opt",
        "line_id",
        "E_rest_keV",
        "centroid_keV",
        "centroid_err_stat_keV",
        "centroid_sys_keV",
        "centroid_sys_event_level_keV",
        "centroid_sys_total_keV",
        "z_xray",
        "z_xray_err_stat",
        "z_xray_sys",
        "z_xray_sys_event_level",
        "z_xray_sys_total",
        "delta_z",
        "delta_v_kms",
        "sigma_E_keV",
        "sigma_E_err_stat_keV",
        "sigma_E_sys_keV",
        "sigma_v_intr_kms",
        "sigma_v_intr_sys_kms",
        "detected",
        "best_window_keV",
        "best_gain_frac",
        "best_min_counts",
        "pi",
        "rmf",
    ]
    with out_sum_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_sum)
        w.writeheader()
        for r in summary_rows:
            w.writerow({k: r.get(k, "") for k in fieldnames_sum})

    summary_metrics: Dict[str, Any] = {
        "generated_utc": _utc_now(),
        "argv": list(sys.argv),
        "targets_csv": _relpath(targets_csv),
        "event_level_qc_summary_csv": _relpath(out_dir / "xrism_event_level_qc_summary.csv")
        if (out_dir / "xrism_event_level_qc_summary.csv").exists()
        else None,
        "obsids": obsids,
        "outputs": {"summary_csv": _relpath(out_sum_csv), "summary_metrics_json": _relpath(out_sum_json)},
        "per_obs": per_obs,
        "notes": {
            "systematics": "centroid_sys_keV/z_xray_sys は (window/gain/rebin) sweep の散らばり。centroid_sys_event_level_keV/z_xray_sys_event_level は event_cl と products の手続き差（平均エネルギー差）を追加系統として入れたもの。total は両者を二乗和で合成。",
        },
    }
    _write_json(out_sum_json, summary_metrics)

    try:
        worklog.append_event(
            {
                "event_type": "xrism_cluster_redshift_turbulence",
                "argv": list(sys.argv),
                "inputs": {"targets_csv": targets_csv},
                "outputs": {"summary_csv": out_sum_csv, "summary_metrics_json": out_sum_json},
                "summary": {"obsids": obsids, "n_rows": len(summary_rows)},
            }
        )
    except Exception:
        pass

    print(f"[ok] targets : {targets_csv}")
    print(f"[ok] out dir : {out_dir}")
    print(f"[ok] summary : {out_sum_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
