#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xrism_bh_outflow_velocity.py

Phase 4 / Step 4.8.2（BH/AGN）:
XRISM（Resolve）の公開一次データ（PI + RMF）を直接再解析し、Fe-K 吸収線の centroid ずれから
アウトフロー速度 v/c を固定出力化する（統計と系統の分離を出力として残す）。

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
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.cosmology.boss_dr12v5_fits import read_bintable_columns, read_first_bintable_layout  # noqa: E402
from scripts.summary import worklog  # noqa: E402

_CARD = 80
_BLOCK = 2880


# 関数: `_utc_now` の入出力契約と処理意図を定義する。
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_write_json` の入出力契約と処理意図を定義する。

def _write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


# 関数: `_relpath` の入出力契約と処理意図を定義する。

def _relpath(path: Optional[Path]) -> Optional[str]:
    # 条件分岐: `path is None` を満たす経路を評価する。
    if path is None:
        return None

    try:
        return path.relative_to(_ROOT).as_posix()
    except Exception:
        return path.as_posix()


# 関数: `_read_csv_rows` の入出力契約と処理意図を定義する。

def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        return []

    rows: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # 条件分岐: `not isinstance(r, dict)` を満たす経路を評価する。
            if not isinstance(r, dict):
                continue

            rows.append({str(k): (v or "").strip() for k, v in r.items() if k is not None})

    return rows


# 関数: `_maybe_float` の入出力契約と処理意図を定義する。

def _maybe_float(x: object) -> Optional[float]:
    # 条件分岐: `x is None` を満たす経路を評価する。
    if x is None:
        return None

    # 条件分岐: `isinstance(x, (int, float))` を満たす経路を評価する。

    if isinstance(x, (int, float)):
        v = float(x)
        return v if math.isfinite(v) else None

    s = str(x).strip()
    # 条件分岐: `not s` を満たす経路を評価する。
    if not s:
        return None

    try:
        v = float(s)
    except Exception:
        return None

    return v if math.isfinite(v) else None


# 関数: `_maybe_bool` の入出力契約と処理意図を定義する。

def _maybe_bool(x: object) -> Optional[bool]:
    # 条件分岐: `x is None` を満たす経路を評価する。
    if x is None:
        return None

    # 条件分岐: `isinstance(x, bool)` を満たす経路を評価する。

    if isinstance(x, bool):
        return x

    s = str(x).strip().lower()
    # 条件分岐: `not s` を満たす経路を評価する。
    if not s:
        return None

    # 条件分岐: `s in {"1", "true", "t", "yes", "y"}` を満たす経路を評価する。

    if s in {"1", "true", "t", "yes", "y"}:
        return True

    # 条件分岐: `s in {"0", "false", "f", "no", "n"}` を満たす経路を評価する。

    if s in {"0", "false", "f", "no", "n"}:
        return False

    return None


# 関数: `_combine_in_quadrature` の入出力契約と処理意図を定義する。

def _combine_in_quadrature(a: Optional[float], b: Optional[float]) -> Optional[float]:
    # 条件分岐: `a is None and b is None` を満たす経路を評価する。
    if a is None and b is None:
        return None

    # 条件分岐: `a is None` を満たす経路を評価する。

    if a is None:
        return b

    # 条件分岐: `b is None` を満たす経路を評価する。

    if b is None:
        return a

    return math.sqrt(float(a) ** 2 + float(b) ** 2)


# 関数: `_load_event_level_qc_summary_by_obsid` の入出力契約と処理意図を定義する。

def _load_event_level_qc_summary_by_obsid(out_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load event-level QC summary (products vs event_cl) keyed by obsid.
    Used as an additional "procedure-difference" systematic term for centroid/beta.
    """
    path = out_dir / "xrism_event_level_qc_summary.csv"
    rows = _read_csv_rows(path)
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        obsid = str(r.get("obsid") or "").strip()
        # 条件分岐: `not obsid` を満たす経路を評価する。
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


# 関数: `_iter_cards_from_header_bytes` の入出力契約と処理意図を定義する。

def _iter_cards_from_header_bytes(header_bytes: bytes) -> Iterable[str]:
    for i in range(0, len(header_bytes), _CARD):
        yield header_bytes[i : i + _CARD].decode("ascii", errors="ignore")


# 関数: `_read_exact` の入出力契約と処理意図を定義する。

def _read_exact(f, n: int) -> bytes:
    b = f.read(n)
    # 条件分岐: `b is None` を満たす経路を評価する。
    if b is None:
        return b""

    return b


# 関数: `_read_header_blocks` の入出力契約と処理意図を定義する。

def _read_header_blocks(f) -> bytes:
    chunks: List[bytes] = []
    while True:
        block = _read_exact(f, _BLOCK)
        # 条件分岐: `len(block) != _BLOCK` を満たす経路を評価する。
        if len(block) != _BLOCK:
            raise EOFError("unexpected EOF while reading FITS header")

        chunks.append(block)
        for card in _iter_cards_from_header_bytes(block):
            # 条件分岐: `card.startswith("END")` を満たす経路を評価する。
            if card.startswith("END"):
                return b"".join(chunks)


# 関数: `_parse_header_kv` の入出力契約と処理意図を定義する。

def _parse_header_kv(header_bytes: bytes) -> Dict[str, str]:
    kv: Dict[str, str] = {}
    for card in _iter_cards_from_header_bytes(header_bytes):
        key = card[:8].strip()
        # 条件分岐: `not key or "=" not in card` を満たす経路を評価する。
        if not key or "=" not in card:
            continue

        rhs = card.split("=", 1)[1]
        rhs = rhs.split("/", 1)[0].strip()
        kv[key] = rhs

    return kv


_TFORM_RE = re.compile(r"^\s*(?P<rep>\d*)(?P<code>[A-Z])\s*$")


# 関数: `_tform_to_numpy_dtype` の入出力契約と処理意図を定義する。
def _tform_to_numpy_dtype(tform: str) -> Tuple[np.dtype, int, int]:
    """
    Return (dtype, repeat, nbytes). FITS binary tables are big-endian.
    Minimal support for scalar fields (rep==1) used by XRISM PI/EBOUNDS.
    """
    m = _TFORM_RE.match(tform)
    # 条件分岐: `not m` を満たす経路を評価する。
    if not m:
        raise ValueError(f"unsupported TFORM: {tform!r}")

    rep = int(m.group("rep") or "1")
    code = m.group("code")
    # 条件分岐: `rep < 1` を満たす経路を評価する。
    if rep < 1:
        raise ValueError(f"invalid repeat in TFORM: {tform!r}")

    # 条件分岐: `code == "I"` を満たす経路を評価する。

    if code == "I":
        return np.dtype(">i2"), rep, 2 * rep

    # 条件分岐: `code == "J"` を満たす経路を評価する。

    if code == "J":
        return np.dtype(">i4"), rep, 4 * rep

    # 条件分岐: `code == "K"` を満たす経路を評価する。

    if code == "K":
        return np.dtype(">i8"), rep, 8 * rep

    # 条件分岐: `code == "E"` を満たす経路を評価する。

    if code == "E":
        return np.dtype(">f4"), rep, 4 * rep

    # 条件分岐: `code == "D"` を満たす経路を評価する。

    if code == "D":
        return np.dtype(">f8"), rep, 8 * rep

    # 条件分岐: `code == "A"` を満たす経路を評価する。

    if code == "A":
        return np.dtype(f"S{rep}"), rep, rep

    # 条件分岐: `code == "B"` を満たす経路を評価する。

    if code == "B":
        return np.dtype("u1"), rep, 1 * rep

    # 条件分岐: `code == "L"` を満たす経路を評価する。

    if code == "L":
        return np.dtype("S1"), rep, 1 * rep

    raise ValueError(f"unsupported TFORM code: {code!r} (tform={tform!r})")


# 関数: `_skip_hdu_data` の入出力契約と処理意図を定義する。

def _skip_hdu_data(f, header_kv: Dict[str, str]) -> None:
    """
    Skip data payload for the current HDU (best-effort).
    This is enough to skip OGIP RMF MATRIX (with heap) and reach EBOUNDS.
    """
    naxis = int(header_kv.get("NAXIS", "0") or "0")
    # 条件分岐: `naxis <= 0` を満たす経路を評価する。
    if naxis <= 0:
        return

    naxis1 = int(header_kv.get("NAXIS1", "0") or "0")
    naxis2 = int(header_kv.get("NAXIS2", "0") or "0")
    pcount = int(header_kv.get("PCOUNT", "0") or "0")
    gcount = int(header_kv.get("GCOUNT", "1") or "1")
    data_bytes = naxis1 * naxis2 * max(gcount, 1) + max(pcount, 0)
    pad = ((int(data_bytes) + _BLOCK - 1) // _BLOCK) * _BLOCK
    # 条件分岐: `pad > 0` を満たす経路を評価する。
    if pad > 0:
        f.seek(pad, 1)


# 関数: `_read_ebounds_table` の入出力契約と処理意図を定義する。

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
            # 条件分岐: `extname.upper() == "EBOUNDS"` を満たす経路を評価する。
            if extname.upper() == "EBOUNDS":
                row_bytes = int(kv.get("NAXIS1", "0") or "0")
                n_rows = int(kv.get("NAXIS2", "0") or "0")
                tfields = int(kv.get("TFIELDS", "0") or "0")
                # 条件分岐: `row_bytes <= 0 or n_rows <= 0 or tfields <= 0` を満たす経路を評価する。
                if row_bytes <= 0 or n_rows <= 0 or tfields <= 0:
                    raise ValueError("invalid EBOUNDS header (missing NAXIS1/NAXIS2/TFIELDS)")

                ttype: Dict[int, str] = {}
                tform: Dict[int, str] = {}
                for card in _iter_cards_from_header_bytes(hdr):
                    key = card[:8].strip()
                    # 条件分岐: `key.startswith("TTYPE")` を満たす経路を評価する。
                    if key.startswith("TTYPE"):
                        try:
                            i = int(key[5:])
                        except Exception:
                            continue

                        v = card.split("=", 1)[1].split("/", 1)[0].strip()
                        # 条件分岐: `len(v) >= 2 and v[0] == "'" and v[-1] == "'"` を満たす経路を評価する。
                        if len(v) >= 2 and v[0] == "'" and v[-1] == "'":
                            v = v[1:-1]

                        ttype[i] = v.strip()
                    # 条件分岐: 前段条件が不成立で、`key.startswith("TFORM")` を追加評価する。
                    elif key.startswith("TFORM"):
                        try:
                            i = int(key[5:])
                        except Exception:
                            continue

                        v = card.split("=", 1)[1].split("/", 1)[0].strip()
                        # 条件分岐: `len(v) >= 2 and v[0] == "'" and v[-1] == "'"` を満たす経路を評価する。
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
                    # 条件分岐: `name is None or fmt is None` を満たす経路を評価する。
                    if name is None or fmt is None:
                        raise ValueError(f"missing TTYPE/TFORM for field {i}")

                    _, rep, width = _tform_to_numpy_dtype(fmt)
                    # 条件分岐: `rep != 1` を満たす経路を評価する。
                    if rep != 1:
                        raise ValueError("EBOUNDS repeat!=1 is not supported")

                    columns.append(name)
                    offsets[name] = off
                    formats[name] = fmt
                    off += width

                # 条件分岐: `off != row_bytes` を満たす経路を評価する。

                if off != row_bytes:
                    raise ValueError(f"row size mismatch in EBOUNDS: {off} != {row_bytes}")

                names: List[str] = []
                fmts: List[np.dtype] = []
                offs: List[int] = []
                for name in columns:
                    dt, rep, _ = _tform_to_numpy_dtype(formats[name])
                    # 条件分岐: `rep != 1` を満たす経路を評価する。
                    if rep != 1:
                        raise ValueError("EBOUNDS repeat!=1 is not supported")

                    names.append(name)
                    fmts.append(dt)
                    offs.append(int(offsets[name]))

                dt_struct = np.dtype({"names": names, "formats": fmts, "offsets": offs, "itemsize": row_bytes})

                b = _read_exact(f, row_bytes * n_rows)
                # 条件分岐: `len(b) != row_bytes * n_rows` を満たす経路を評価する。
                if len(b) != row_bytes * n_rows:
                    raise EOFError("unexpected EOF while reading EBOUNDS data")

                arr = np.frombuffer(b, dtype=dt_struct, count=n_rows)
                col_map = {str(c).upper(): str(c) for c in columns}
                ch_key = col_map.get("CHANNEL")
                e_min_key = col_map.get("E_MIN")
                e_max_key = col_map.get("E_MAX")
                # 条件分岐: `ch_key is None or e_min_key is None or e_max_key is None` を満たす経路を評価する。
                if ch_key is None or e_min_key is None or e_max_key is None:
                    raise ValueError("EBOUNDS missing CHANNEL/E_MIN/E_MAX")

                ch = np.asarray(arr[ch_key], dtype=int)
                e_min = np.asarray(arr[e_min_key], dtype=float)
                e_max = np.asarray(arr[e_max_key], dtype=float)
                return ch, 0.5 * (e_min + e_max)

            _skip_hdu_data(f, kv)


# 関数: `_load_pi_spectrum` の入出力契約と処理意図を定義する。

def _load_pi_spectrum(pi_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    opener = gzip.open if pi_path.name.endswith(".gz") else Path.open
    with opener(pi_path, "rb") as f:  # type: ignore[arg-type]
        layout = read_first_bintable_layout(f)
        col_map = {str(c).upper(): str(c) for c in layout.columns}
        ch_key = col_map.get("CHANNEL")
        cnt_key = col_map.get("COUNTS")
        # 条件分岐: `ch_key is None or cnt_key is None` を満たす経路を評価する。
        if ch_key is None or cnt_key is None:
            raise ValueError("PI missing CHANNEL/COUNTS")

        cols = read_bintable_columns(f, layout=layout, columns=[ch_key, cnt_key])
        return np.asarray(cols[ch_key], dtype=int), np.asarray(cols[cnt_key], dtype=float)


# 関数: `_find_local_obs_root` の入出力契約と処理意図を定義する。

def _find_local_obs_root(data_root: Path, obsid: str) -> Tuple[Optional[str], Optional[Path]]:
    direct = data_root / obsid
    # 条件分岐: `direct.is_dir()` を満たす経路を評価する。
    if direct.is_dir():
        return "direct", direct

    for cand in sorted(data_root.glob(f"*/{obsid}")):
        # 条件分岐: `cand.is_dir()` を満たす経路を評価する。
        if cand.is_dir():
            return cand.parent.name, cand

    return None, None


_PX_RE = re.compile(r"px(?P<px>\d+)", flags=re.IGNORECASE)


# 関数: `_px_score` の入出力契約と処理意図を定義する。
def _px_score(name: str) -> Tuple[int, int, str]:
    m = _PX_RE.search(name)
    px = int(m.group("px")) if m else 10**9
    # prefer px=1000, then 0000, then 5000
    pref = {1000: 0, 0: 1, 5000: 2}.get(px, 9)
    return pref, px, name


# 関数: `_choose_pi_rmf_pair` の入出力契約と処理意図を定義する。

def _choose_pi_rmf_pair(products_dir: Path) -> Tuple[Path, Path]:
    pis = sorted(products_dir.glob("*_src.pi*"))
    # 条件分岐: `not pis` を満たす経路を評価する。
    if not pis:
        raise FileNotFoundError(f"no *_src.pi* under {products_dir}")

    pairs: List[Tuple[Tuple[int, int, str], Path, Path]] = []
    for pi in pis:
        rmf_name = re.sub(r"_src\.pi(\.gz)?$", r".rmf\1", pi.name)
        rmf = products_dir / rmf_name
        # 条件分岐: `rmf.exists()` を満たす経路を評価する。
        if rmf.exists():
            pairs.append((_px_score(pi.name), pi, rmf))

    # 条件分岐: `not pairs` を満たす経路を評価する。

    if not pairs:
        rmfs = sorted(products_dir.glob("*.rmf*"))
        # 条件分岐: `not rmfs` を満たす経路を評価する。
        if not rmfs:
            raise FileNotFoundError(f"no rmf found under {products_dir}")

        return pis[0], rmfs[0]

    pairs.sort(key=lambda x: x[0])
    return pairs[0][1], pairs[0][2]


# 関数: `_rebin_min_counts` の入出力契約と処理意図を定義する。

def _rebin_min_counts(energy: np.ndarray, counts: np.ndarray, *, min_counts: float) -> Tuple[np.ndarray, np.ndarray]:
    # 条件分岐: `min_counts <= 0` を満たす経路を評価する。
    if min_counts <= 0:
        return energy, counts

    out_e: List[float] = []
    out_c: List[float] = []
    acc_c = 0.0
    acc_ec = 0.0
    for e, c in zip(energy, counts):
        # 条件分岐: `not np.isfinite(e) or not np.isfinite(c)` を満たす経路を評価する。
        if not np.isfinite(e) or not np.isfinite(c):
            continue

        acc_c += float(c)
        acc_ec += float(e) * float(c)
        # 条件分岐: `acc_c >= float(min_counts)` を満たす経路を評価する。
        if acc_c >= float(min_counts):
            out_c.append(acc_c)
            out_e.append(acc_ec / acc_c if acc_c > 0 else float(e))
            acc_c = 0.0
            acc_ec = 0.0

    # 条件分岐: `acc_c > 0` を満たす経路を評価する。

    if acc_c > 0:
        out_c.append(acc_c)
        out_e.append(acc_ec / acc_c if acc_c > 0 else float(energy[-1]))

    return np.asarray(out_e, dtype=float), np.asarray(out_c, dtype=float)


# 関数: `_model_counts_abs_gauss` の入出力契約と処理意図を定義する。

def _model_counts_abs_gauss(E: np.ndarray, norm: float, gamma: float, depth: float, centroid: float, sigma: float) -> np.ndarray:
    """
    counts(E) = norm * E^{-gamma} * (1 - depth * exp(-(E-centroid)^2/(2*sigma^2))).
    """
    E = np.asarray(E, dtype=float)
    cont = float(norm) * np.power(np.clip(E, 1e-6, None), -float(gamma))
    prof = np.exp(-0.5 * np.square((E - float(centroid)) / max(float(sigma), 1e-6)))
    return cont * (1.0 - float(depth) * prof)


# 関数: `_fit_absorption_line` の入出力契約と処理意図を定義する。

def _fit_absorption_line(
    energy_keV: np.ndarray,
    counts: np.ndarray,
    *,
    window_keV: Tuple[float, float],
    centroid_bounds_keV: Tuple[float, float],
    min_counts: float,
    sigma_min_keV: float,
) -> Dict[str, Any]:
    # 条件分岐: `curve_fit is None` を満たす経路を評価する。
    if curve_fit is None:
        raise RuntimeError("scipy is required for fitting")

    e0, e1 = float(window_keV[0]), float(window_keV[1])
    m = (energy_keV >= min(e0, e1)) & (energy_keV <= max(e0, e1)) & np.isfinite(energy_keV) & np.isfinite(counts)
    e = np.asarray(energy_keV[m], dtype=float)
    y = np.asarray(counts[m], dtype=float)
    # 条件分岐: `e.size < 10 or float(np.nansum(y)) <= 0.0` を満たす経路を評価する。
    if e.size < 10 or float(np.nansum(y)) <= 0.0:
        return {"ok": False, "reason": "insufficient data in window"}

    e, y = _rebin_min_counts(e, y, min_counts=min_counts)
    # 条件分岐: `e.size < 8` を満たす経路を評価する。
    if e.size < 8:
        return {"ok": False, "reason": "insufficient bins after rebin"}

    # initial guesses

    norm0 = float(np.nanpercentile(y, 90) * np.nanpercentile(e, 90)) if np.isfinite(np.nanpercentile(y, 90)) else 1.0
    gamma0 = 1.0
    depth0 = 0.1
    c0 = float(np.clip(float(np.nanmedian(e)), centroid_bounds_keV[0], centroid_bounds_keV[1]))
    sigma_min = float(sigma_min_keV) if np.isfinite(sigma_min_keV) else 0.001
    sigma_min = max(1e-6, sigma_min)
    sigma0 = max(0.01, sigma_min)
    p0 = [norm0, gamma0, depth0, c0, sigma0]

    lower = [0.0, -5.0, 0.0, centroid_bounds_keV[0], sigma_min]
    upper = [float("inf"), 8.0, 0.95, centroid_bounds_keV[1], 0.2]
    err = np.sqrt(np.clip(y, 0.0, None) + 1.0)

    try:
        popt, pcov = curve_fit(
            _model_counts_abs_gauss,
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

    model = _model_counts_abs_gauss(e, *popt)
    resid = (y - model) / err
    rss = float(np.nansum(resid**2))
    dof = int(max(0, e.size - len(popt)))
    chi2_red = float(rss / dof) if dof > 0 else float("nan")

    perr = np.full(len(popt), float("nan"))
    # 条件分岐: `pcov is not None and np.all(np.isfinite(pcov))` を満たす経路を評価する。
    if pcov is not None and np.all(np.isfinite(pcov)):
        perr = np.sqrt(np.clip(np.diag(pcov), 0.0, None))

    norm, gamma, depth, centroid, sigma = [float(x) for x in popt]
    depth_err = float(perr[2]) if np.isfinite(perr[2]) else float("nan")
    centroid_err = float(perr[3]) if np.isfinite(perr[3]) else float("nan")
    detected = bool(np.isfinite(depth_err) and depth_err > 0 and (depth / depth_err) >= 3.0)

    return {
        "ok": True,
        "n_bins": int(e.size),
        "window_keV": [e0, e1],
        "min_counts": float(min_counts),
        "params": {
            "norm": norm,
            "gamma": gamma,
            "depth": depth,
            "centroid_keV": centroid,
            "sigma_keV": sigma,
        },
        "errors_1sigma": {
            "depth": float(perr[2]) if np.isfinite(perr[2]) else None,
            "centroid_keV": float(perr[3]) if np.isfinite(perr[3]) else None,
        },
        "fit_quality": {"chi2_red": chi2_red, "dof": dof},
        "detected": detected,
        "plot": {"energy_keV": e, "counts": y, "model": model},
    }


# 関数: `_beta_from_energy` の入出力契約と処理意図を定義する。

def _beta_from_energy(E_obs_keV: float, *, E_rest_keV: float, z_sys: float) -> Optional[float]:
    """
    β = (D^2 - 1)/(D^2 + 1), D = E_obs*(1+z_sys)/E_rest
    """
    # 条件分岐: `not np.isfinite(E_obs_keV) or E_obs_keV <= 0` を満たす経路を評価する。
    if not np.isfinite(E_obs_keV) or E_obs_keV <= 0:
        return None

    # 条件分岐: `not np.isfinite(E_rest_keV) or E_rest_keV <= 0` を満たす経路を評価する。

    if not np.isfinite(E_rest_keV) or E_rest_keV <= 0:
        return None

    # 条件分岐: `not np.isfinite(z_sys)` を満たす経路を評価する。

    if not np.isfinite(z_sys):
        z_sys = 0.0

    D = (float(E_obs_keV) * (1.0 + float(z_sys))) / float(E_rest_keV)
    # 条件分岐: `D <= 0` を満たす経路を評価する。
    if D <= 0:
        return None

    D2 = D * D
    return (D2 - 1.0) / (D2 + 1.0)


# 関数: `_beta_err_from_energy_err` の入出力契約と処理意図を定義する。

def _beta_err_from_energy_err(
    E_obs_keV: float,
    E_obs_err_keV: float,
    *,
    E_rest_keV: float,
    z_sys: float,
) -> Optional[float]:
    # 条件分岐: `not np.isfinite(E_obs_err_keV) or E_obs_err_keV <= 0` を満たす経路を評価する。
    if not np.isfinite(E_obs_err_keV) or E_obs_err_keV <= 0:
        return None

    b0 = _beta_from_energy(E_obs_keV, E_rest_keV=E_rest_keV, z_sys=z_sys)
    # 条件分岐: `b0 is None` を満たす経路を評価する。
    if b0 is None:
        return None

    eps = float(E_obs_err_keV)
    b1 = _beta_from_energy(E_obs_keV + eps, E_rest_keV=E_rest_keV, z_sys=z_sys)
    b2 = _beta_from_energy(max(1e-9, E_obs_keV - eps), E_rest_keV=E_rest_keV, z_sys=z_sys)
    # 条件分岐: `b1 is None or b2 is None` を満たす経路を評価する。
    if b1 is None or b2 is None:
        return None

    return 0.5 * abs(float(b1) - float(b2))


# クラス: `LineSpec` の責務と境界条件を定義する。

@dataclass(frozen=True)
class LineSpec:
    line_id: str
    E_rest_keV: float
    centroid_bounds_keV: Tuple[float, float]
    base_window_keV: Tuple[float, float]


# NOTE: rest energy values are used as fixed references for line_id.

_LINES: List[LineSpec] = [
    LineSpec("FeXXV_HeA", 6.700, (6.0, 12.0), (5.0, 12.0)),
    LineSpec("FeXXVI_LyA", 6.966, (6.0, 12.0), (5.0, 12.0)),
]

# 関数: `_doppler_D` の入出力契約と処理意図を定義する。
def _doppler_D(beta: float) -> Optional[float]:
    # 条件分岐: `not math.isfinite(beta)` を満たす経路を評価する。
    if not math.isfinite(beta):
        return None

    # 条件分岐: `abs(beta) >= 1.0` を満たす経路を評価する。

    if abs(beta) >= 1.0:
        return None

    return math.sqrt((1.0 + float(beta)) / (1.0 - float(beta)))


# 関数: `_line_energy_bounds` の入出力契約と処理意図を定義する。

def _line_energy_bounds(
    *,
    E_rest_keV: float,
    z_sys: float,
    beta_min: float,
    beta_max: float,
) -> Tuple[float, float, float]:
    """
    Return (E_obs_min_keV, E_obs_max_keV, E_obs_beta0_keV).
    E_obs = D(beta) * E_rest/(1+z_sys)
    """
    denom = 1.0 + float(z_sys)
    denom = denom if denom > 0 else 1.0
    E0 = float(E_rest_keV) / denom
    Dmin = _doppler_D(float(beta_min))
    Dmax = _doppler_D(float(beta_max))
    # 条件分岐: `Dmin is None or Dmax is None` を満たす経路を評価する。
    if Dmin is None or Dmax is None:
        raise ValueError("invalid beta range")

    e_min = float(min(Dmin, Dmax)) * E0
    e_max = float(max(Dmin, Dmax)) * E0
    return e_min, e_max, E0


# 関数: `_window_sweep` の入出力契約と処理意図を定義する。

def _window_sweep(base: Tuple[float, float], *, delta_keV: float) -> List[Tuple[float, float]]:
    lo, hi = float(base[0]), float(base[1])
    # 条件分岐: `not math.isfinite(lo) or not math.isfinite(hi)` を満たす経路を評価する。
    if not math.isfinite(lo) or not math.isfinite(hi):
        return [(lo, hi)]

    # 条件分岐: `hi <= lo` を満たす経路を評価する。

    if hi <= lo:
        return [(lo, hi)]

    width = hi - lo
    d = float(delta_keV)
    # 条件分岐: `not math.isfinite(d) or d <= 0` を満たす経路を評価する。
    if not math.isfinite(d) or d <= 0:
        return [(lo, hi)]

    d = min(d, 0.25 * width)
    # 条件分岐: `d <= 0` を満たす経路を評価する。
    if d <= 0:
        return [(lo, hi)]

    return [(lo, hi), (lo, hi - d), (lo + d, hi)]


# 関数: `_gain_sweep` の入出力契約と処理意図を定義する。

def _gain_sweep() -> List[float]:
    return [-1e-3, 0.0, +1e-3]


# 関数: `_plot_fit` の入出力契約と処理意図を定義する。

def _plot_fit(out_png: Path, *, energy: np.ndarray, counts: np.ndarray, model: np.ndarray, title: str) -> Optional[str]:
    # 条件分岐: `plt is None` を満たす経路を評価する。
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


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--targets", default=str(_ROOT / "output" / "private" / "xrism" / "xrism_targets_catalog.csv"))
    p.add_argument("--obsid", action="append", default=[], help="override: obsid(s) to analyze (repeatable)")
    p.add_argument("--role", default="bh_agn", help="targets role to include when --obsid is not provided")
    p.add_argument("--min-counts", type=float, default=30.0, help="min counts per rebinned bin (base)")
    p.add_argument("--beta-min", type=float, default=-0.05, help="beta search min (defines centroid bounds/window)")
    p.add_argument("--beta-max", type=float, default=0.35, help="beta search max (defines centroid bounds/window)")
    p.add_argument("--window-pad-keV", type=float, default=0.30, help="pad added to derived [Emin,Emax] window")
    p.add_argument("--window-sweep-delta-keV", type=float, default=0.30, help="delta for (lo,hi-d)/(lo+d,hi) sweep")
    p.add_argument(
        "--sigma-min-keV",
        type=float,
        default=0.001,
        help="lower bound for Gaussian sigma in keV (default: 0.001; note: Resolve 5 eV FWHM ~= sigma=0.00212)",
    )
    p.add_argument(
        "--sys-chi-tau",
        type=float,
        default=1.0,
        help="tau for sys weighting by fit quality: w=exp(-(chi2_red-min_chi2_red)/tau) over detected fits (default: 1.0)",
    )
    p.add_argument("--out-dir", default=str(_ROOT / "output" / "private" / "xrism"))
    p.add_argument("--data-root", default=str(_ROOT / "data" / "xrism" / "heasarc" / "obs"))
    args = p.parse_args(list(argv) if argv is not None else None)

    targets_csv = Path(args.targets)
    rows = _read_csv_rows(targets_csv)

    # 条件分岐: `args.obsid` を満たす経路を評価する。
    if args.obsid:
        obsids = [str(x).strip() for x in args.obsid if str(x).strip()]
    else:
        obsids = []
        for r in rows:
            # 条件分岐: `(r.get("role") or "").strip() != str(args.role)` を満たす経路を評価する。
            if (r.get("role") or "").strip() != str(args.role):
                continue

            o = (r.get("obsid") or "").strip()
            # 条件分岐: `o` を満たす経路を評価する。
            if o:
                obsids.append(o)

    obsids = sorted(set(obsids))
    # 条件分岐: `not obsids` を満たす経路を評価する。
    if not obsids:
        print(f"[warn] no obsids found (targets={targets_csv})")
        return 0

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    qc_by_obsid = _load_event_level_qc_summary_by_obsid(out_dir)

    summary_rows: List[Dict[str, Any]] = []
    per_obs: Dict[str, Any] = {}

    for obsid in obsids:
        row = next((r for r in rows if (r.get("obsid") or "").strip() == obsid), {})
        target_name = (row.get("target_name") or "").strip()
        z_sys_raw = (row.get("z_sys") or "").strip()
        try:
            z_sys = float(z_sys_raw) if z_sys_raw else 0.0
        except Exception:
            z_sys = 0.0

        cat, local_obs_root = _find_local_obs_root(data_root, obsid)
        # 条件分岐: `local_obs_root is None` を満たす経路を評価する。
        if local_obs_root is None:
            per_obs[obsid] = {"status": "missing_cache"}
            continue

        products_dir = local_obs_root / "resolve" / "products"
        # 条件分岐: `not products_dir.is_dir()` を満たす経路を評価する。
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
        finite_energy = energy[np.isfinite(energy)]
        # 条件分岐: `finite_energy.size == 0` を満たす経路を評価する。
        if finite_energy.size == 0:
            per_obs[obsid] = {"status": "invalid_energy_map"}
            continue

        e_data_min = float(np.nanmin(finite_energy))
        e_data_max = float(np.nanmax(finite_energy))

        base_min_counts = float(args.min_counts)
        gain_list = _gain_sweep()
        min_counts_list = [base_min_counts, 2.0 * base_min_counts]

        obs_metrics: Dict[str, Any] = {
            "generated_utc": _utc_now(),
            "status": "ok",
            "obsid": obsid,
            "target_name": target_name,
            "z_sys": z_sys,
            "cache": {
                "cat": cat,
                "obs_root": _relpath(local_obs_root),
                "products_dir": _relpath(products_dir),
                "pi": _relpath(pi_path),
                "rmf": _relpath(rmf_path),
            },
            "analysis": {
                "lines": [ls.__dict__ for ls in _LINES],
                "best_selection_policy": "min(chi2_red) among detected fits; fallback to min(chi2_red) among all fits",
                "systematics_policy": "sys is estimated as weighted std(beta) over detected+ok fit variations with w=exp(-(chi2_red-min_chi2_red)/tau); fallback: unweighted std(beta) over detected+ok fits.",
                "window_policy": {
                    "beta_min": float(args.beta_min),
                    "beta_max": float(args.beta_max),
                    "window_pad_keV": float(args.window_pad_keV),
                    "window_sweep_delta_keV": float(args.window_sweep_delta_keV),
                    "note": "各line_idについて z_sys と (beta_min,beta_max) から [Emin,Emax] を導出し、data rangeへclipして window/bounds を定義する。",
                },
                "systematics_fit_quality_weighting": {"tau": float(args.sys_chi_tau)},
                "gain_frac_sweep": gain_list,
                "min_counts_sweep": min_counts_list,
                "detection_rule": "depth/σ_depth >= 3",
            },
            "results": {},
        }

        for ls in _LINES:
            e_min, e_max, e0 = _line_energy_bounds(
                E_rest_keV=float(ls.E_rest_keV),
                z_sys=float(z_sys),
                beta_min=float(args.beta_min),
                beta_max=float(args.beta_max),
            )
            pad = float(args.window_pad_keV)
            win_base = (max(e_data_min, e_min - pad), min(e_data_max, e_max + pad))
            win_list = _window_sweep(win_base, delta_keV=float(args.window_sweep_delta_keV))
            centroid_bounds = (max(e_data_min, e_min), min(e_data_max, e_max))

            variations: List[Dict[str, Any]] = []
            best_any: Optional[Dict[str, Any]] = None
            best_any_chi = float("inf")
            best_det: Optional[Dict[str, Any]] = None
            best_det_chi = float("inf")

            for w in win_list:
                for gain in gain_list:
                    for mc in min_counts_list:
                        e_adj = energy * (1.0 + float(gain))
                        fit = _fit_absorption_line(
                            e_adj,
                            counts,
                            window_keV=w,
                            centroid_bounds_keV=centroid_bounds,
                            min_counts=mc,
                            sigma_min_keV=float(args.sigma_min_keV),
                        )
                        variations.append(
                            {
                                "window_keV": [float(w[0]), float(w[1])],
                                "gain_frac": float(gain),
                                "min_counts": float(mc),
                                "fit": {k: v for k, v in fit.items() if k != "plot"},
                            }
                        )
                        # 条件分岐: `not fit.get("ok")` を満たす経路を評価する。
                        if not fit.get("ok"):
                            continue

                        chi = float(fit.get("fit_quality", {}).get("chi2_red", float("inf")))
                        # 条件分岐: `not np.isfinite(chi)` を満たす経路を評価する。
                        if not np.isfinite(chi):
                            continue

                        # 条件分岐: `chi < best_any_chi` を満たす経路を評価する。

                        if chi < best_any_chi:
                            best_any_chi = chi
                            best_any = fit
                            best_any["_best_window_keV"] = [float(w[0]), float(w[1])]
                            best_any["_best_gain_frac"] = float(gain)
                            best_any["_best_min_counts"] = float(mc)

                        # 条件分岐: `bool(fit.get("detected")) and chi < best_det_chi` を満たす経路を評価する。

                        if bool(fit.get("detected")) and chi < best_det_chi:
                            best_det_chi = chi
                            best_det = fit
                            best_det["_best_window_keV"] = [float(w[0]), float(w[1])]
                            best_det["_best_gain_frac"] = float(gain)
                            best_det["_best_min_counts"] = float(mc)

            best = best_det or best_any
            # 条件分岐: `best is None or not bool(best.get("ok"))` を満たす経路を評価する。
            if best is None or not bool(best.get("ok")):
                obs_metrics["results"][ls.line_id] = {"ok": False, "reason": "no successful fit", "variations": variations}
                continue

            centroid = float(best["params"]["centroid_keV"])
            centroid_err = best.get("errors_1sigma", {}).get("centroid_keV")
            centroid_err_f = float(centroid_err) if centroid_err is not None else float("nan")
            beta = _beta_from_energy(centroid, E_rest_keV=ls.E_rest_keV, z_sys=z_sys)
            beta_err = _beta_err_from_energy_err(
                centroid,
                centroid_err_f,
                E_rest_keV=ls.E_rest_keV,
                z_sys=z_sys,
            )

            beta_vars: List[float] = []
            centroid_vars: List[float] = []
            beta_vars_det: List[float] = []
            centroid_vars_det: List[float] = []
            chi_det: List[float] = []
            for rec in variations:
                fit0 = rec.get("fit") or {}
                # 条件分岐: `not fit0.get("ok")` を満たす経路を評価する。
                if not fit0.get("ok"):
                    continue

                c0 = float((fit0.get("params") or {}).get("centroid_keV", float("nan")))
                # 条件分岐: `not np.isfinite(c0)` を満たす経路を評価する。
                if not np.isfinite(c0):
                    continue

                b0 = _beta_from_energy(c0, E_rest_keV=ls.E_rest_keV, z_sys=z_sys)
                # 条件分岐: `b0 is None or not np.isfinite(b0)` を満たす経路を評価する。
                if b0 is None or not np.isfinite(b0):
                    continue

                centroid_vars.append(c0)
                beta_vars.append(float(b0))
                # 条件分岐: `bool(fit0.get("detected"))` を満たす経路を評価する。
                if bool(fit0.get("detected")):
                    chi0 = float((fit0.get("fit_quality") or {}).get("chi2_red", float("nan")))
                    # 条件分岐: `np.isfinite(chi0)` を満たす経路を評価する。
                    if np.isfinite(chi0):
                        chi_det.append(chi0)

                    centroid_vars_det.append(c0)
                    beta_vars_det.append(float(b0))
            # Prefer detected variations for sys (avoid non-detection local minima inflating sys).

            beta_sys_src = beta_vars_det if len(beta_vars_det) >= 2 else beta_vars
            centroid_sys_src = centroid_vars_det if len(centroid_vars_det) >= 2 else centroid_vars

            # 関数: `_weighted_std` の入出力契約と処理意図を定義する。
            def _weighted_std(vals: List[float], chis: List[float], *, tau: float) -> Optional[float]:
                # 条件分岐: `len(vals) < 2 or len(vals) != len(chis)` を満たす経路を評価する。
                if len(vals) < 2 or len(vals) != len(chis):
                    return None

                # 条件分岐: `not np.isfinite(tau) or tau <= 0` を満たす経路を評価する。

                if not np.isfinite(tau) or tau <= 0:
                    return None

                cmin = float(np.nanmin(np.asarray(chis, dtype=float)))
                # 条件分岐: `not np.isfinite(cmin)` を満たす経路を評価する。
                if not np.isfinite(cmin):
                    return None

                w = np.exp(-(np.asarray(chis, dtype=float) - cmin) / float(tau))
                w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
                ws = float(np.sum(w))
                # 条件分岐: `ws <= 0` を満たす経路を評価する。
                if ws <= 0:
                    return None

                x = np.asarray(vals, dtype=float)
                m = float(np.sum(w * x) / ws)
                v = float(np.sum(w * np.square(x - m)) / ws)
                return math.sqrt(v) if v > 0 else 0.0

            beta_sys_w = _weighted_std(beta_vars_det, chi_det, tau=float(args.sys_chi_tau)) if len(beta_vars_det) == len(chi_det) else None
            centroid_sys_w = _weighted_std(centroid_vars_det, chi_det, tau=float(args.sys_chi_tau)) if len(centroid_vars_det) == len(chi_det) else None

            beta_sys = float(beta_sys_w) if beta_sys_w is not None else (float(np.nanstd(beta_sys_src, ddof=1)) if len(beta_sys_src) >= 2 else float("nan"))
            centroid_sys = float(centroid_sys_w) if centroid_sys_w is not None else (float(np.nanstd(centroid_sys_src, ddof=1)) if len(centroid_sys_src) >= 2 else float("nan"))

            # Additional systematic from event-level QC (products vs event_cl).
            qc = qc_by_obsid.get(obsid) or {}
            mean_shift_keV = qc.get("mean_shift_keV_event_minus_products")
            mean_shift_keV_f = float(mean_shift_keV) if mean_shift_keV is not None and math.isfinite(float(mean_shift_keV)) else None
            centroid_sys_event = abs(mean_shift_keV_f) if mean_shift_keV_f is not None else None
            beta_sys_event = (
                _beta_err_from_energy_err(
                    centroid,
                    float(centroid_sys_event),
                    E_rest_keV=ls.E_rest_keV,
                    z_sys=z_sys,
                )
                if centroid_sys_event is not None
                else None
            )
            centroid_sys_base = centroid_sys if np.isfinite(centroid_sys) else None
            beta_sys_base = beta_sys if np.isfinite(beta_sys) else None
            centroid_sys_total = _combine_in_quadrature(centroid_sys_base, centroid_sys_event)
            beta_sys_total = _combine_in_quadrature(beta_sys_base, beta_sys_event)

            obs_metrics["results"][ls.line_id] = {
                "ok": True,
                "E_rest_keV": ls.E_rest_keV,
                "derived_window": {
                    "E_beta0_keV": float(e0),
                    "E_beta_min_keV": float(e_min),
                    "E_beta_max_keV": float(e_max),
                    "centroid_bounds_keV": [float(centroid_bounds[0]), float(centroid_bounds[1])],
                    "base_window_keV": [float(win_base[0]), float(win_base[1])],
                    "window_sweep": [[float(w[0]), float(w[1])] for w in win_list],
                },
                "best": {k: v for k, v in best.items() if k != "plot"},
                "derived": {
                    "centroid_keV": centroid,
                    "centroid_err_stat_keV": float(centroid_err_f) if np.isfinite(centroid_err_f) else None,
                    "centroid_sys_keV": centroid_sys if np.isfinite(centroid_sys) else None,
                    "centroid_sys_event_level_keV": float(centroid_sys_event) if centroid_sys_event is not None else None,
                    "centroid_sys_total_keV": float(centroid_sys_total) if centroid_sys_total is not None else None,
                    "beta": float(beta) if beta is not None and np.isfinite(beta) else None,
                    "beta_err_stat": float(beta_err) if beta_err is not None and np.isfinite(beta_err) else None,
                    "beta_sys": beta_sys if np.isfinite(beta_sys) else None,
                    "beta_sys_event_level": float(beta_sys_event) if beta_sys_event is not None else None,
                    "beta_sys_total": float(beta_sys_total) if beta_sys_total is not None else None,
                },
                "event_level_qc": qc or None,
                "variations": variations,
            }

            summary_rows.append(
                {
                    "obsid": obsid,
                    "target_name": target_name,
                    "z_sys": z_sys,
                    "line_id": ls.line_id,
                    "E_rest_keV": ls.E_rest_keV,
                    "centroid_keV": centroid,
                    "centroid_err_stat_keV": float(centroid_err_f) if np.isfinite(centroid_err_f) else "",
                    "centroid_sys_keV": centroid_sys if np.isfinite(centroid_sys) else "",
                    "centroid_sys_event_level_keV": float(centroid_sys_event) if centroid_sys_event is not None else "",
                    "centroid_sys_total_keV": float(centroid_sys_total) if centroid_sys_total is not None else "",
                    "beta": float(beta) if beta is not None and np.isfinite(beta) else "",
                    "beta_err_stat": float(beta_err) if beta_err is not None and np.isfinite(beta_err) else "",
                    "beta_sys": beta_sys if np.isfinite(beta_sys) else "",
                    "beta_sys_event_level": float(beta_sys_event) if beta_sys_event is not None else "",
                    "beta_sys_total": float(beta_sys_total) if beta_sys_total is not None else "",
                    "detected": bool(best.get("detected")),
                    "best_window_keV": ",".join(map(str, best.get("_best_window_keV", []))),
                    "best_gain_frac": best.get("_best_gain_frac", ""),
                    "best_min_counts": best.get("_best_min_counts", ""),
                    "pi": _relpath(pi_path),
                    "rmf": _relpath(rmf_path),
                }
            )

            plot = best.get("plot") or {}
            eplt = plot.get("energy_keV")
            yplt = plot.get("counts")
            mplt = plot.get("model")
            # 条件分岐: `plt is not None and isinstance(eplt, np.ndarray) and isinstance(yplt, np.ndar...` を満たす経路を評価する。
            if plt is not None and isinstance(eplt, np.ndarray) and isinstance(yplt, np.ndarray) and isinstance(mplt, np.ndarray):
                out_png = out_dir / f"{obsid}__{ls.line_id}__fit.png"
                _ = _plot_fit(out_png, energy=eplt, counts=yplt, model=mplt, title=f"XRISM {obsid} {ls.line_id} fit")

        out_csv = out_dir / f"{obsid}__line_fit.csv"
        out_json = out_dir / f"{obsid}__line_fit_metrics.json"
        fieldnames = [
            "obsid",
            "target_name",
            "z_sys",
            "line_id",
            "E_rest_keV",
            "centroid_keV",
            "centroid_err_stat_keV",
            "centroid_sys_keV",
            "centroid_sys_event_level_keV",
            "centroid_sys_total_keV",
            "beta",
            "beta_err_stat",
            "beta_sys",
            "beta_sys_event_level",
            "beta_sys_total",
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

    out_sum_csv = out_dir / "xrism_bh_outflow_velocity_summary.csv"
    out_sum_json = out_dir / "xrism_bh_outflow_velocity_summary_metrics.json"
    fieldnames_sum = [
        "obsid",
        "target_name",
        "z_sys",
        "line_id",
        "E_rest_keV",
        "centroid_keV",
        "centroid_err_stat_keV",
        "centroid_sys_keV",
        "centroid_sys_event_level_keV",
        "centroid_sys_total_keV",
        "beta",
        "beta_err_stat",
        "beta_sys",
        "beta_sys_event_level",
        "beta_sys_total",
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
            "systematics": "centroid_sys_keV/beta_sys は (window/gain/rebin) sweep の散らばり。centroid_sys_event_level_keV/beta_sys_event_level は event_cl と products の手続き差（平均エネルギー差）を追加系統として入れたもの。total は両者を二乗和で合成。",
        },
    }
    _write_json(out_sum_json, summary_metrics)

    try:
        worklog.append_event(
            {
                "event_type": "xrism_bh_outflow_velocity",
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


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
