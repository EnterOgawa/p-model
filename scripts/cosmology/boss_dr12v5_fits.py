#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
boss_dr12v5_fits.py

BOSS DR12v5 LSS の FITS (BINTABLE) を、astropy無しで最小限読み出すためのヘルパ。

目的：
- Windows/Python 3.13 環境で astropy が使えないケースでも、DR12v5 の銀河/ランダム catalog から
  RA/DEC/z/weight を抽出できるようにする。
- ランダム catalog は巨大なため「先頭N行だけ読む」等の部分抽出をサポートする。

想定入力：
- BOSS DR12v5 LSS の `*.fits.gz`（primary HDU + 1つ目の BINTABLE）
  - 推奨（高速ミラー）: https://dr12.sdss3.org/sas/dr12/boss/lss/
  - 旧（遅い場合あり）: https://data.sdss.org/sas/dr12/boss/lss/

注意：
- FITS の完全実装ではない（BINTABLEの一般仕様の一部のみ）。
- BOSS DR12v5 の該当ファイルで必要な範囲（A/I/J/E/D など）を優先。
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import BinaryIO, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np

_CARD = 80
_BLOCK = 2880

_TFORM_RE = re.compile(r"^\s*(?P<rep>\d*)(?P<code>[A-Z])\s*$")


@dataclass(frozen=True)
class FitsBintableLayout:
    row_bytes: int
    n_rows: int
    columns: List[str]  # in order
    offsets: Dict[str, int]
    formats: Dict[str, str]  # FITS TFORM (raw)


def _read_exact(f: BinaryIO, n: int) -> bytes:
    b = f.read(n)
    if b is None:
        return b""
    return b


def _iter_cards_from_header_bytes(header_bytes: bytes) -> Iterable[str]:
    for i in range(0, len(header_bytes), _CARD):
        yield header_bytes[i : i + _CARD].decode("ascii", errors="ignore")


def _read_header_blocks(f: BinaryIO) -> bytes:
    """
    Read a FITS header (multiple of 2880 bytes) and return header bytes.
    Leaves file pointer at the start of the next HDU payload.
    """
    chunks: List[bytes] = []
    while True:
        block = _read_exact(f, _BLOCK)
        if len(block) != _BLOCK:
            raise EOFError("unexpected EOF while reading FITS header")
        chunks.append(block)
        # END card may be in this block
        for card in _iter_cards_from_header_bytes(block):
            if card.startswith("END"):
                return b"".join(chunks)


def _parse_int_card(card: str) -> Optional[int]:
    if "=" not in card:
        return None
    # KEYWORD = value / comment
    rhs = card.split("=", 1)[1]
    rhs = rhs.split("/", 1)[0].strip()
    if not rhs:
        return None
    try:
        return int(rhs)
    except Exception:
        return None


def _parse_str_card(card: str) -> Optional[str]:
    if "=" not in card:
        return None
    rhs = card.split("=", 1)[1]
    rhs = rhs.split("/", 1)[0].strip()
    if len(rhs) >= 2 and rhs[0] == "'" and rhs[-1] == "'":
        return rhs[1:-1]
    return None


def _tform_to_numpy_dtype(tform: str) -> Tuple[np.dtype, int, int]:
    """
    Return (dtype, repeat, nbytes) for a scalar/array field.
    FITS binary tables are big-endian.
    """
    m = _TFORM_RE.match(tform)
    if not m:
        raise ValueError(f"unsupported TFORM: {tform!r}")
    rep = int(m.group("rep") or "1")
    code = m.group("code")
    if rep < 1:
        raise ValueError(f"invalid repeat in TFORM: {tform!r}")

    if code == "A":
        dt = np.dtype(f"S{rep}")
        return dt, rep, rep
    if code == "I":
        dt = np.dtype(">i2")
        return dt, rep, 2 * rep
    if code == "J":
        dt = np.dtype(">i4")
        return dt, rep, 4 * rep
    if code == "K":
        dt = np.dtype(">i8")
        return dt, rep, 8 * rep
    if code == "E":
        dt = np.dtype(">f4")
        return dt, rep, 4 * rep
    if code == "D":
        dt = np.dtype(">f8")
        return dt, rep, 8 * rep
    if code == "B":
        dt = np.dtype("u1")
        return dt, rep, 1 * rep
    if code == "L":
        # Logical (boolean): stored as 1 byte per element (typically 'T'/'F').
        # We keep the raw byte here; higher-level code may map 'T'->True.
        dt = np.dtype("S1")
        return dt, rep, 1 * rep
    if code == "X":
        # Bit array: rep is in bits, storage is ceil(rep/8) bytes.
        # We expose raw bytes here. Most callers won't request bit-array columns.
        dt = np.dtype("u1")
        nbytes = (rep + 7) // 8
        return dt, rep, int(nbytes)
    raise ValueError(f"unsupported TFORM code: {code!r} (tform={tform!r})")


def read_first_bintable_layout(f: BinaryIO) -> FitsBintableLayout:
    """
    Parse primary header and the first BINTABLE extension header, and return layout.
    The stream position will be at the start of the BINTABLE data section.
    """
    # primary header
    _ = _read_header_blocks(f)
    # extension header
    hdr = _read_header_blocks(f)

    row_bytes: Optional[int] = None
    n_rows: Optional[int] = None
    tfields: Optional[int] = None
    ttype: Dict[int, str] = {}
    tform: Dict[int, str] = {}

    for card in _iter_cards_from_header_bytes(hdr):
        key = card[:8].strip()
        if key == "NAXIS1":
            row_bytes = _parse_int_card(card)
        elif key == "NAXIS2":
            n_rows = _parse_int_card(card)
        elif key == "TFIELDS":
            tfields = _parse_int_card(card)
        elif key.startswith("TTYPE"):
            try:
                i = int(key[5:])
            except Exception:
                continue
            v = _parse_str_card(card)
            if v is not None:
                ttype[i] = v.strip()
        elif key.startswith("TFORM"):
            try:
                i = int(key[5:])
            except Exception:
                continue
            v = _parse_str_card(card)
            if v is not None:
                tform[i] = v.strip()

    if row_bytes is None or n_rows is None or tfields is None:
        raise ValueError("failed to parse BINTABLE header (missing NAXIS1/NAXIS2/TFIELDS)")
    if tfields < 1:
        raise ValueError(f"invalid TFIELDS: {tfields}")

    # compute offsets by walking all fields in order
    columns: List[str] = []
    offsets: Dict[str, int] = {}
    formats: Dict[str, str] = {}
    off = 0
    for i in range(1, tfields + 1):
        name = ttype.get(i)
        fmt = tform.get(i)
        if name is None or fmt is None:
            raise ValueError(f"missing TTYPE/TFORM for field {i} (got TTYPE={name}, TFORM={fmt})")
        _, _, width = _tform_to_numpy_dtype(fmt)
        columns.append(name)
        offsets[name] = off
        formats[name] = fmt
        off += width

    if off != row_bytes:
        # Accept but warn via exception? Here we keep strict to avoid mis-parse.
        raise ValueError(f"row size mismatch: sum(TFORM widths)={off} != NAXIS1={row_bytes}")

    return FitsBintableLayout(
        row_bytes=int(row_bytes),
        n_rows=int(n_rows),
        columns=columns,
        offsets=offsets,
        formats=formats,
    )


def read_bintable_columns(
    f: BinaryIO,
    *,
    layout: FitsBintableLayout,
    columns: List[str],
    max_rows: Optional[int] = None,
    chunk_rows: int = 200_000,
) -> Dict[str, np.ndarray]:
    """
    Read selected scalar columns from a BINTABLE data section.
    The stream position must be at the start of the data section.
    """
    want = [c.strip() for c in columns]
    for c in want:
        if c not in layout.offsets:
            raise KeyError(f"column not found: {c!r}")

    n_total = int(layout.n_rows)
    n_read = n_total if max_rows is None else min(n_total, int(max_rows))
    if n_read < 0:
        raise ValueError("max_rows must be >= 0")

    # Build numpy structured dtype with offsets and itemsize = row_bytes.
    names: List[str] = []
    fmts: List[np.dtype] = []
    offs: List[int] = []
    for c in want:
        tform = layout.formats[c]
        dt, rep, _ = _tform_to_numpy_dtype(tform)
        if rep != 1:
            raise ValueError(f"only scalar fields supported in read_bintable_columns (col={c!r}, TFORM={tform!r})")
        names.append(c)
        fmts.append(dt)
        offs.append(int(layout.offsets[c]))
    dt_struct = np.dtype({"names": names, "formats": fmts, "offsets": offs, "itemsize": int(layout.row_bytes)})

    out: Dict[str, np.ndarray] = {c: np.empty(n_read, dtype=np.float64) for c in want}

    row_bytes = int(layout.row_bytes)
    i0 = 0
    while i0 < n_read:
        n_chunk = min(int(chunk_rows), n_read - i0)
        b = _read_exact(f, n_chunk * row_bytes)
        if len(b) != n_chunk * row_bytes:
            raise EOFError("unexpected EOF while reading BINTABLE data")
        arr = np.frombuffer(b, dtype=dt_struct, count=n_chunk)
        for c in want:
            # Convert big-endian scalar to native float64.
            out[c][i0 : i0 + n_chunk] = np.asarray(arr[c], dtype=np.float64)
        i0 += n_chunk

    # If max_rows < total rows, we stop here (caller may close stream early).
    return out


def iter_bintable_column_chunks(
    f: BinaryIO,
    *,
    layout: FitsBintableLayout,
    columns: List[str],
    max_rows: Optional[int] = None,
    chunk_rows: int = 200_000,
) -> Iterator[Dict[str, np.ndarray]]:
    """
    Yield selected scalar columns from a BINTABLE data section in chunks.
    The stream position must be at the start of the data section.

    - Values are converted to float64 (matching `read_bintable_columns`).
    - This is useful for reservoir sampling without loading the full table.
    """
    want = [c.strip() for c in columns]
    for c in want:
        if c not in layout.offsets:
            raise KeyError(f"column not found: {c!r}")

    n_total = int(layout.n_rows)
    n_read = n_total if max_rows is None else min(n_total, int(max_rows))
    if n_read < 0:
        raise ValueError("max_rows must be >= 0")
    if int(chunk_rows) <= 0:
        raise ValueError("chunk_rows must be > 0")

    names: List[str] = []
    fmts: List[np.dtype] = []
    offs: List[int] = []
    for c in want:
        tform = layout.formats[c]
        dt, rep, _ = _tform_to_numpy_dtype(tform)
        if rep != 1:
            raise ValueError(f"only scalar fields supported in iter_bintable_column_chunks (col={c!r}, TFORM={tform!r})")
        names.append(c)
        fmts.append(dt)
        offs.append(int(layout.offsets[c]))
    dt_struct = np.dtype({"names": names, "formats": fmts, "offsets": offs, "itemsize": int(layout.row_bytes)})

    row_bytes = int(layout.row_bytes)
    i0 = 0
    while i0 < n_read:
        n_chunk = min(int(chunk_rows), n_read - i0)
        b = _read_exact(f, n_chunk * row_bytes)
        if len(b) != n_chunk * row_bytes:
            raise EOFError("unexpected EOF while reading BINTABLE data")
        arr = np.frombuffer(b, dtype=dt_struct, count=n_chunk)
        out: Dict[str, np.ndarray] = {}
        for c in want:
            out[c] = np.asarray(arr[c], dtype=np.float64)
        i0 += n_chunk
        yield out


def open_gz_stream_from_bytes(prefix: bytes) -> BinaryIO:
    """
    Helper for tests: wrap bytes as a file-like object.
    """
    return io.BytesIO(prefix)
