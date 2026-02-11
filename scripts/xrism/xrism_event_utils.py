#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
xrism_event_utils.py

XRISM/Resolve の event_cl（FITS BINTABLE）を、astropy無しで最小限読み出すユーティリティ。

目的:
- Pixel除外（例：Pixel 27）や GTI 適用を行い、event-level から PI histogram を再構成する。
- products（*_src.pi.gz）由来のスペクトルと比較し、解析I/Fの頑健性確認へ接続する。

注意:
- FITS の完全実装ではない。必要最小限（TIME/PI/PIXEL, GTI START/STOP）に限定する。
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_CARD = 80
_BLOCK = 2880

from scripts.cosmology.boss_dr12v5_fits import (
    FitsBintableLayout,
    iter_bintable_column_chunks,
    read_bintable_columns,
    read_first_bintable_layout,
    _tform_to_numpy_dtype,
)

_PX_RE = re.compile(r"px(?P<px>\d+)", flags=re.IGNORECASE)


def _px_score(name: str) -> Tuple[int, int, str]:
    m = _PX_RE.search(name)
    px = int(m.group("px")) if m else 10**9
    # prefer px=1000, then 0000, then 5000
    pref = {1000: 0, 0: 1, 5000: 2}.get(px, 9)
    return pref, px, name


def choose_event_file(event_cl_dir: Path) -> Path:
    """
    Choose one event file under resolve/event_cl.
    Prefer px=1000, then 0000, then 5000.
    """
    cands = sorted(event_cl_dir.glob("*.evt*"))
    if not cands:
        raise FileNotFoundError(f"no event file found under {event_cl_dir}")
    cands.sort(key=lambda p: _px_score(p.name))
    return cands[0]


def _build_channel_index_map(channels: np.ndarray) -> np.ndarray:
    ch = np.asarray(channels, dtype=int)
    if ch.size < 1:
        raise ValueError("channels is empty")
    max_ch = int(np.max(ch))
    if max_ch < 0:
        raise ValueError("invalid channels (max<0)")
    idx = -np.ones(max_ch + 1, dtype=int)
    ok = (ch >= 0) & (ch <= max_ch)
    idx[ch[ok]] = np.arange(int(ch.size), dtype=int)[ok]
    return idx


def load_gti_intervals(gti_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read GTI file (auxil/*_gen.gti.gz) and return (start, stop) arrays.
    """
    opener = gzip.open if gti_path.name.endswith(".gz") else Path.open
    with opener(gti_path, "rb") as f:  # type: ignore[arg-type]
        layout = read_first_bintable_layout(f)
        col_map = {str(c).upper(): str(c) for c in layout.columns}
        s_key = col_map.get("START")
        t_key = col_map.get("STOP")
        if s_key is None or t_key is None:
            raise ValueError("GTI missing START/STOP")
        cols = read_bintable_columns(f, layout=layout, columns=[s_key, t_key])
    start = np.asarray(cols[s_key], dtype=float)
    stop = np.asarray(cols[t_key], dtype=float)
    if start.size != stop.size:
        raise ValueError("GTI START/STOP size mismatch")
    m = np.isfinite(start) & np.isfinite(stop) & (stop >= start)
    start = np.asarray(start[m], dtype=float)
    stop = np.asarray(stop[m], dtype=float)
    if start.size == 0:
        return start, stop
    order = np.argsort(start)
    return np.asarray(start[order], dtype=float), np.asarray(stop[order], dtype=float)


def _read_exact(f, n: int) -> bytes:  # type: ignore[no-untyped-def]
    b = f.read(n)
    if b is None:
        return b""
    return b


def _iter_cards_from_header_bytes(header_bytes: bytes) -> Iterable[str]:
    for i in range(0, len(header_bytes), _CARD):
        yield header_bytes[i : i + _CARD].decode("ascii", errors="ignore")


def _read_header_blocks(f) -> bytes:  # type: ignore[no-untyped-def]
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


def _skip_hdu_data(f, header_kv: Dict[str, str]) -> None:  # type: ignore[no-untyped-def]
    naxis = int(float(header_kv.get("NAXIS", "0") or "0"))
    if naxis <= 0:
        return
    naxis1 = int(float(header_kv.get("NAXIS1", "0") or "0"))
    naxis2 = int(float(header_kv.get("NAXIS2", "0") or "0"))
    pcount = int(float(header_kv.get("PCOUNT", "0") or "0"))
    gcount = int(float(header_kv.get("GCOUNT", "1") or "1"))
    data_bytes = naxis1 * naxis2 * max(gcount, 1) + max(pcount, 0)
    pad = ((int(data_bytes) + _BLOCK - 1) // _BLOCK) * _BLOCK
    if pad > 0:
        f.seek(pad, 1)


def load_gti_intervals_from_event(event_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read GTI extension from an event_cl file (preferable to auxil/*_gen.gti.gz in many cases).
    """
    opener = gzip.open if event_path.name.endswith(".gz") else Path.open
    with opener(event_path, "rb") as f:  # type: ignore[arg-type]
        _ = _read_header_blocks(f)  # primary
        hdr_evt = _read_header_blocks(f)
        kv_evt = _parse_header_kv(hdr_evt)
        _skip_hdu_data(f, kv_evt)

        # Next extension should be GTI for standard event_cl.
        hdr_gti = _read_header_blocks(f)
        kv_gti = _parse_header_kv(hdr_gti)
        extname = kv_gti.get("EXTNAME", "").strip().strip("'").strip()
        if "GTI" not in extname.upper():
            raise ValueError(f"unexpected EXTNAME for GTI extension: {extname!r}")

        row_bytes = int(float(kv_gti.get("NAXIS1", "0") or "0"))
        n_rows = int(float(kv_gti.get("NAXIS2", "0") or "0"))
        tfields = int(float(kv_gti.get("TFIELDS", "0") or "0"))
        if row_bytes <= 0 or n_rows < 0 or tfields <= 0:
            raise ValueError("invalid GTI header (missing NAXIS1/NAXIS2/TFIELDS)")

        ttype: Dict[int, str] = {}
        tform: Dict[int, str] = {}
        for card in _iter_cards_from_header_bytes(hdr_gti):
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
            _, _, width = _tform_to_numpy_dtype(fmt)
            columns.append(name)
            offsets[name] = int(off)
            formats[name] = fmt
            off += int(width)
        if off != row_bytes:
            raise ValueError(f"row size mismatch in GTI: {off} != {row_bytes}")

        layout = FitsBintableLayout(row_bytes=int(row_bytes), n_rows=int(n_rows), columns=columns, offsets=offsets, formats=formats)
        col_map = {str(c).upper(): str(c) for c in layout.columns}
        s_key = col_map.get("START")
        t_key = col_map.get("STOP")
        if s_key is None or t_key is None:
            raise ValueError("event GTI missing START/STOP")
        cols = read_bintable_columns(f, layout=layout, columns=[s_key, t_key])
        start = np.asarray(cols[s_key], dtype=float)
        stop = np.asarray(cols[t_key], dtype=float)

    if start.size != stop.size:
        raise ValueError("event GTI START/STOP size mismatch")
    m = np.isfinite(start) & np.isfinite(stop) & (stop >= start)
    start = np.asarray(start[m], dtype=float)
    stop = np.asarray(stop[m], dtype=float)
    if start.size == 0:
        return start, stop
    order = np.argsort(start)
    return np.asarray(start[order], dtype=float), np.asarray(stop[order], dtype=float)


def _mask_in_gti(times: np.ndarray, start: np.ndarray, stop: np.ndarray) -> np.ndarray:
    if start.size == 0 or stop.size == 0:
        return np.ones_like(times, dtype=bool)
    t = np.asarray(times, dtype=float)
    idx = np.searchsorted(start, t, side="right") - 1
    ok = (idx >= 0) & (idx < int(stop.size))
    out = np.zeros_like(t, dtype=bool)
    out[ok] = t[ok] <= stop[idx[ok]]
    return out


@dataclass(frozen=True)
class EventSpectrumResult:
    channels: np.ndarray
    counts: np.ndarray
    qc: Dict[str, Any]


def extract_pi_spectrum_from_event(
    event_path: Path,
    *,
    channels: np.ndarray,
    gti_start: Optional[np.ndarray] = None,
    gti_stop: Optional[np.ndarray] = None,
    pixel_exclude: Sequence[int] = (),
    chunk_rows: int = 200_000,
) -> EventSpectrumResult:
    """
    Build PI histogram aligned to `channels` from an event_cl file.

    - channels: RMF (EBOUNDS) channel values.
    - gti_start/stop: optional GTI intervals (same time unit as event TIME).
    - pixel_exclude: pixel IDs to exclude (e.g., [27]).
    """
    ch = np.asarray(channels, dtype=int)
    idx_map = _build_channel_index_map(ch)

    opener = gzip.open if event_path.name.endswith(".gz") else Path.open
    with opener(event_path, "rb") as f:  # type: ignore[arg-type]
        layout = read_first_bintable_layout(f)
        col_map = {str(c).upper(): str(c) for c in layout.columns}

        time_key = col_map.get("TIME")
        pi_key = col_map.get("PI")
        pixel_key = col_map.get("PIXEL")
        if pi_key is None:
            raise ValueError("event file missing PI column")

        want: List[str] = [pi_key]
        if time_key is not None and gti_start is not None and gti_stop is not None:
            want.append(time_key)
        if pixel_key is not None and pixel_exclude:
            want.append(pixel_key)

        counts = np.zeros(int(ch.size), dtype=np.float64)
        n_total = 0
        n_in_gti = 0
        n_after_pixel = 0
        n_in_range = 0
        pixel_counts: Dict[int, int] = {}

        for chunk in iter_bintable_column_chunks(f, layout=layout, columns=want, chunk_rows=int(chunk_rows)):
            n = int(next(iter(chunk.values())).size) if chunk else 0
            if n <= 0:
                continue
            n_total += n

            pi = np.asarray(chunk[pi_key], dtype=np.int64)
            m = np.isfinite(pi)
            if time_key is not None and time_key in chunk and gti_start is not None and gti_stop is not None:
                tm = np.asarray(chunk[time_key], dtype=float)
                m = m & _mask_in_gti(tm, np.asarray(gti_start, dtype=float), np.asarray(gti_stop, dtype=float))
            n_in_gti += int(np.count_nonzero(m))

            if pixel_key is not None and pixel_key in chunk and pixel_exclude:
                px = np.asarray(chunk[pixel_key], dtype=np.int64)
                # Update pixel counts before exclusion (for QC).
                for v in px[m]:
                    pv = int(v)
                    pixel_counts[pv] = int(pixel_counts.get(pv, 0)) + 1
                m = m & (~np.isin(px, np.asarray(list(pixel_exclude), dtype=np.int64)))
            n_after_pixel += int(np.count_nonzero(m))

            if not np.any(m):
                continue
            pi_sel = pi[m]
            ok = (pi_sel >= 0) & (pi_sel < int(idx_map.size))
            if not np.any(ok):
                continue
            idx = idx_map[pi_sel[ok]]
            ok2 = idx >= 0
            if not np.any(ok2):
                continue
            idx2 = idx[ok2].astype(int)
            n_in_range += int(idx2.size)
            counts += np.bincount(idx2, minlength=int(ch.size)).astype(np.float64)

    # Pixel QC summary
    top_pixels = sorted(pixel_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    exclude_counts = {int(p): int(pixel_counts.get(int(p), 0)) for p in pixel_exclude}
    denom = float(n_in_gti) if n_in_gti > 0 else float(n_total if n_total > 0 else 1.0)
    qc = {
        "event_path": str(event_path),
        "event_n_rows": int(n_total),
        "event_counts_sum": float(np.nansum(counts)),
        "columns_used": {"pi": pi_key, "time": time_key, "pixel": pixel_key},
        "filters": {
            "has_gti": bool(gti_start is not None and gti_stop is not None and np.asarray(gti_start).size > 0),
            "pixel_exclude": [int(x) for x in pixel_exclude],
            "chunk_rows": int(chunk_rows),
        },
        "flow": {
            "n_total": int(n_total),
            "n_after_gti": int(n_in_gti),
            "n_after_pixel": int(n_after_pixel),
            "n_in_channel_range": int(n_in_range),
        },
        "pixel_counts_top": [{"pixel": int(p), "n": int(c)} for p, c in top_pixels],
        "pixel_exclude_counts": [{"pixel": int(p), "n": int(exclude_counts.get(int(p), 0))} for p in pixel_exclude],
        "pixel_exclude_fraction_of_after_gti": [
            {"pixel": int(p), "frac": (float(exclude_counts.get(int(p), 0)) / denom if denom > 0 else float("nan"))}
            for p in pixel_exclude
        ],
    }
    return EventSpectrumResult(channels=ch, counts=counts, qc=qc)


def compute_spectrum_diff_metrics(
    *,
    channels: np.ndarray,
    energy_keV: np.ndarray,
    counts_a: np.ndarray,
    counts_b: np.ndarray,
    fek_band: Tuple[float, float] = (5.5, 7.5),
) -> Dict[str, Any]:
    """
    Compare two channel-aligned spectra (A vs B).
    Returns normalized L1 and energy-weighted mean shift in Fe-K band.
    """
    ch = np.asarray(channels, dtype=int)
    e = np.asarray(energy_keV, dtype=float)
    a = np.asarray(counts_a, dtype=float)
    b = np.asarray(counts_b, dtype=float)
    if ch.size != a.size or ch.size != b.size or e.size != ch.size:
        raise ValueError("shape mismatch in compute_spectrum_diff_metrics")

    def _l1(x: np.ndarray, y: np.ndarray) -> float:
        denom = float(np.nansum(np.abs(x))) if float(np.nansum(np.abs(x))) > 0 else 1.0
        return float(np.nansum(np.abs(x - y)) / denom)

    lo, hi = float(fek_band[0]), float(fek_band[1])
    m = (e >= lo) & (e <= hi) & np.isfinite(e) & np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return {"ok": False, "reason": "empty band"}

    def _mean_E(x: np.ndarray) -> float:
        w = np.clip(x, 0.0, None)
        s = float(np.nansum(w[m]))
        if s <= 0:
            return float("nan")
        return float(np.nansum(e[m] * w[m]) / s)

    mean_a = _mean_E(a)
    mean_b = _mean_E(b)
    return {
        "ok": True,
        "band_keV": [lo, hi],
        "counts_sum_a": float(np.nansum(a[m])),
        "counts_sum_b": float(np.nansum(b[m])),
        "l1_norm_a": _l1(a[m], b[m]),
        "mean_energy_keV_a": mean_a,
        "mean_energy_keV_b": mean_b,
        "mean_energy_shift_keV_b_minus_a": (float(mean_b) - float(mean_a)) if np.isfinite(mean_a) and np.isfinite(mean_b) else float("nan"),
    }
