#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fek_relativistic_broadening_reflection_fit.py

Phase 4 / Step 4.13.7（Fe-Kα relativistic broad line / ISCO）:
XSPEC（pyXspec）を用いた forward-fold（RMF/ARF）fit の入口。

本リポジトリの既定環境では XSPEC/HEASoft が無い場合があるため、
その場合は「ブロック状態（missing_xspec）」として plan 付き JSON を出力して終了する。

入力（推奨）:
- 事前に `scripts/xrism/fek_relativistic_broadening_isco_constraints.py` を実行して、
  XMM の PPS スペクトル選定（rmf_sources）を凍結した detail JSON を生成する。
  - `output/private/xrism/xmm_<obsid>__fek_broad_line_rmf_diskline.json`

出力（固定名）:
- `output/private/xrism/xmm_<obsid>__fek_broad_line_reflection_xspec.json`
  - status=ok の場合：best-fit の r_in（proxy）等を含む
  - status=blocked_missing_xspec の場合：入力 plan のみを含む（後段で再実行可能）

注意:
- ここでは「ISCO制約のI/F（ファイル帰属・実行入口）」を閉じることが目的。
  物理モデル（relxill等）や系統分解（cross-cal/abs/continuum）は次段で拡張する。
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


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


def _load_rmf_sources_from_detail(detail_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    debug = detail_json.get("debug", {})
    rmf_sources = debug.get("rmf_sources", [])
    out: List[Dict[str, Any]] = []
    if not isinstance(rmf_sources, list):
        return out
    for s in rmf_sources:
        if not isinstance(s, dict):
            continue
        spec = str(s.get("spec") or "").strip()
        bkg = str(s.get("bkg") or "").strip()
        arf = str(s.get("arf") or "").strip()
        rmf = str(s.get("rmf_local") or "").strip()
        if not spec:
            continue
        out.append(
            {
                "spec": spec,
                "bkg": bkg,
                "arf": arf,
                "rmf": rmf,
                "rmf_detchans": int(s.get("rmf_detchans") or 0),
                "spectrum_rebin": s.get("spectrum_rebin") if isinstance(s.get("spectrum_rebin"), dict) else {},
            }
        )
    return out


def _try_import_xspec() -> Tuple[Optional[object], str]:
    try:
        import xspec  # type: ignore

        return xspec, ""
    except Exception as e:
        return None, str(e)


def _find_xspec_cli(path_hint: str) -> Optional[Path]:
    hint = str(path_hint or "").strip()
    if hint:
        p = Path(hint)
        if p.exists():
            return p
        # Allow passing a bare command name.
        q = shutil.which(hint)
        if q:
            return Path(q)
        return None
    q = shutil.which("xspec")
    return Path(q) if q else None


_FLOAT_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_ERR_FLAGS_RE = re.compile(r"\b[TF]{9}\b")


def _parse_xspec_tclout_line(stdout: str, key: str) -> str:
    # Use the last occurrence to tolerate repeated commands.
    m = None
    for mm in re.finditer(rf"^{re.escape(key)}=(.*)$", stdout, flags=re.MULTILINE):
        m = mm
    return (m.group(1).strip() if m else "").strip()


def _parse_float_list(text: str) -> List[float]:
    out: List[float] = []
    for s in _FLOAT_RE.findall(text):
        try:
            out.append(float(s))
        except Exception:
            continue
    return out


_CARD = 80
_BLOCK = 2880


def _read_exact(f, n: int) -> bytes:  # type: ignore[no-untyped-def]
    b = f.read(int(n))
    if b is None:
        return b""
    return b


def _read_header_blocks(f) -> bytes:  # type: ignore[no-untyped-def]
    chunks: List[bytes] = []
    while True:
        block = _read_exact(f, _BLOCK)
        if len(block) != _BLOCK:
            raise EOFError("unexpected EOF while reading FITS header")
        chunks.append(block)
        for i in range(0, _BLOCK, _CARD):
            card = block[i : i + _CARD].decode("ascii", errors="ignore")
            if card.startswith("END"):
                return b"".join(chunks)


def _parse_header_kv(header_bytes: bytes) -> Dict[str, str]:
    kv: Dict[str, str] = {}
    for i in range(0, len(header_bytes), _CARD):
        card = header_bytes[i : i + _CARD].decode("ascii", errors="ignore")
        key = card[:8].strip()
        if not key or "=" not in card:
            continue
        rhs = card.split("=", 1)[1]
        rhs = rhs.split("/", 1)[0].strip()
        kv[key] = rhs
    return kv


def _read_ogip_spectrum_columns(path: Path) -> Dict[str, List[int]]:
    opener = gzip.open if path.name.lower().endswith(".ftz") or path.name.lower().endswith(".gz") else Path.open
    with opener(path, "rb") as f:  # type: ignore[arg-type]
        _ = _read_header_blocks(f)  # primary
        hdr_ext = _read_header_blocks(f)  # first ext (SPECTRUM)
        kv = _parse_header_kv(hdr_ext)
        row_bytes = int(float(kv.get("NAXIS1", "0") or "0"))
        n_rows = int(float(kv.get("NAXIS2", "0") or "0"))
        tfields = int(float(kv.get("TFIELDS", "0") or "0"))
        if row_bytes <= 0 or n_rows <= 0 or tfields <= 0:
            raise ValueError("invalid SPECTRUM header (missing NAXIS1/NAXIS2/TFIELDS)")

        ttype: Dict[int, str] = {}
        tform: Dict[int, str] = {}
        for i in range(1, tfields + 1):
            t = kv.get(f"TTYPE{i}")
            if t is not None:
                ttype[i] = str(t).strip().strip("'").strip()
            fm = kv.get(f"TFORM{i}")
            if fm is not None:
                tform[i] = str(fm).strip().strip("'").strip()

        offsets: Dict[str, int] = {}
        widths: Dict[str, int] = {}
        off = 0
        for i in range(1, tfields + 1):
            name = ttype.get(i, "")
            fmt = tform.get(i, "")
            if not name or not fmt:
                raise ValueError(f"missing TTYPE/TFORM for field {i}")
            code = fmt.split("(", 1)[0].strip().upper()
            if code.startswith("1"):
                code = code[1:]
            if code == "I":
                width = 2
            elif code == "J":
                width = 4
            else:
                raise ValueError(f"unsupported TFORM for OGIP spectrum: {fmt!r}")
            offsets[name] = int(off)
            widths[name] = int(width)
            off += int(width)
        if off != row_bytes:
            raise ValueError(f"row size mismatch: sum(TFORM widths)={off} != NAXIS1={row_bytes}")

        raw = _read_exact(f, row_bytes * n_rows)
        if len(raw) != row_bytes * n_rows:
            raise EOFError("unexpected EOF while reading SPECTRUM table")

    want = ["CHANNEL", "COUNTS", "GROUPING", "QUALITY"]
    out: Dict[str, List[int]] = {}
    for k in want:
        if k in offsets:
            out[k] = [0] * int(n_rows)

    mv = memoryview(raw)
    for i in range(int(n_rows)):
        base = i * row_bytes
        for k, n_b in widths.items():
            if k not in out:
                continue
            o = offsets[k]
            out[k][i] = int.from_bytes(mv[base + o : base + o + n_b], byteorder="big", signed=True)
    return out


def _pad_spaces_to_block(f, n_written: int) -> None:  # type: ignore[no-untyped-def]
    pad = (-int(n_written)) % _BLOCK
    if pad:
        f.write(b" " * pad)


def _fits_card(key: str, value: str, comment: str = "") -> str:
    k = str(key)[:8].ljust(8)
    v = str(value).strip()
    s = f"{k}= {v}"
    if comment:
        s += f" / {comment}"
    if len(s) > _CARD:
        s = s[:_CARD]
    return s.ljust(_CARD)


def _fits_end_card() -> str:
    return "END".ljust(_CARD)


@dataclass(frozen=True)
class _PhaHeaderSeed:
    telescop: str
    instrume: str
    filter: str
    chantype: str
    exposure: str
    backscal: str
    areascal: str
    corrscal: str
    poisserr: str
    sys_err: str
    hduclass: str
    hduclas1: str
    hduclas2: str
    hduclas3: str
    hduvers1: str


def _read_pha_header_seed(path: Path) -> _PhaHeaderSeed:
    opener = gzip.open if path.name.lower().endswith(".ftz") or path.name.lower().endswith(".gz") else Path.open
    with opener(path, "rb") as f:  # type: ignore[arg-type]
        _ = _read_header_blocks(f)  # primary
        hdr_ext = _read_header_blocks(f)  # SPECTRUM
    kv = _parse_header_kv(hdr_ext)

    def g(key: str, default: str) -> str:
        return str(kv.get(key, default)).strip()

    return _PhaHeaderSeed(
        telescop=g("TELESCOP", "''"),
        instrume=g("INSTRUME", "''"),
        filter=g("FILTER", "''"),
        chantype=g("CHANTYPE", "'PI'"),
        exposure=g("EXPOSURE", "0.0"),
        backscal=g("BACKSCAL", "1.0"),
        areascal=g("AREASCAL", "1.0"),
        corrscal=g("CORRSCAL", ""),
        poisserr=g("POISSERR", "T"),
        sys_err=g("SYS_ERR", "0"),
        hduclass=g("HDUCLASS", "'OGIP'"),
        hduclas1=g("HDUCLAS1", "'SPECTRUM'"),
        hduclas2=g("HDUCLAS2", "'TOTAL'"),
        hduclas3=g("HDUCLAS3", "'COUNT'"),
        hduvers1=g("HDUVERS1", "'1.1.0'"),
    )


def _write_ogip_pha(
    out_path: Path,
    *,
    channels: List[int],
    counts: List[int],
    grouping: Optional[List[int]],
    quality: Optional[List[int]],
    detchans: int,
    seed: _PhaHeaderSeed,
    include_links: bool = False,
) -> None:
    n_rows = int(len(channels))
    if n_rows <= 0 or n_rows != int(len(counts)):
        raise ValueError("invalid channels/counts length")
    has_gq = grouping is not None and quality is not None
    if has_gq and (len(grouping or []) != n_rows or len(quality or []) != n_rows):
        raise ValueError("invalid grouping/quality length")

    if has_gq:
        tfields = 4
        row_bytes = 10
        ttypes = ["CHANNEL", "COUNTS", "GROUPING", "QUALITY"]
        tforms = ["I", "J", "I", "I"]
    else:
        tfields = 2
        row_bytes = 6
        ttypes = ["CHANNEL", "COUNTS"]
        tforms = ["I", "J"]

    cards_primary = [
        _fits_card("SIMPLE", "T"),
        _fits_card("BITPIX", "8"),
        _fits_card("NAXIS", "0"),
        _fits_card("EXTEND", "T"),
        _fits_end_card(),
    ]

    cards_ext: List[str] = [
        _fits_card("XTENSION", "'BINTABLE'"),
        _fits_card("BITPIX", "8"),
        _fits_card("NAXIS", "2"),
        _fits_card("NAXIS1", str(int(row_bytes))),
        _fits_card("NAXIS2", str(int(n_rows))),
        _fits_card("PCOUNT", "0"),
        _fits_card("GCOUNT", "1"),
        _fits_card("TFIELDS", str(int(tfields))),
    ]
    for i, (tt, tf) in enumerate(zip(ttypes, tforms), start=1):
        cards_ext.append(_fits_card(f"TTYPE{i}", f"'{tt}'"))
        cards_ext.append(_fits_card(f"TFORM{i}", f"'{tf}'"))
    cards_ext.extend(
        [
            _fits_card("EXTNAME", "'SPECTRUM'"),
            _fits_card("HDUCLASS", seed.hduclass),
            _fits_card("HDUCLAS1", seed.hduclas1),
            _fits_card("HDUCLAS2", seed.hduclas2),
            _fits_card("HDUCLAS3", seed.hduclas3),
            _fits_card("HDUVERS1", seed.hduvers1),
            _fits_card("TELESCOP", seed.telescop),
            _fits_card("INSTRUME", seed.instrume),
            _fits_card("FILTER", seed.filter),
            _fits_card("CHANTYPE", seed.chantype),
            _fits_card("DETCHANS", str(int(detchans))),
            _fits_card("EXPOSURE", seed.exposure),
            _fits_card("BACKSCAL", seed.backscal),
            _fits_card("AREASCAL", seed.areascal),
            _fits_card("POISSERR", seed.poisserr),
            _fits_card("SYS_ERR", seed.sys_err),
        ]
    )
    if bool(include_links):
        # XSPEC expects these keywords to exist for OGIP spectra. We keep them as 'none'
        # because the analysis fixes response/arf/background explicitly in the .xcm.
        cards_ext.append(_fits_card("RESPFILE", "'none'"))
        cards_ext.append(_fits_card("ANCRFILE", "'none'"))
        cards_ext.append(_fits_card("BACKFILE", "'none'"))
    if str(seed.corrscal).strip():
        cards_ext.append(_fits_card("CORRSCAL", seed.corrscal))
    cards_ext.append(_fits_end_card())

    data = bytearray(int(row_bytes) * int(n_rows))
    for i in range(n_rows):
        base = i * row_bytes
        data[base : base + 2] = int(channels[i]).to_bytes(2, byteorder="big", signed=True)
        data[base + 2 : base + 6] = int(counts[i]).to_bytes(4, byteorder="big", signed=True)
        if has_gq and grouping is not None and quality is not None:
            data[base + 6 : base + 8] = int(grouping[i]).to_bytes(2, byteorder="big", signed=True)
            data[base + 8 : base + 10] = int(quality[i]).to_bytes(2, byteorder="big", signed=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        hdr_primary = "".join(cards_primary).encode("ascii")
        f.write(hdr_primary)
        _pad_spaces_to_block(f, len(hdr_primary))

        hdr_ext = "".join(cards_ext).encode("ascii")
        f.write(hdr_ext)
        _pad_spaces_to_block(f, len(hdr_ext))

        f.write(bytes(data))
        pad = (-len(data)) % _BLOCK
        if pad:
            f.write(b"\0" * pad)


def _rebin_sum(values: List[int], factor: int) -> List[int]:
    if factor <= 1:
        return list(values)
    n = int(len(values))
    if n % factor != 0:
        raise ValueError(f"length not divisible by factor: {n} % {factor} != 0")
    return [int(sum(values[i : i + factor])) for i in range(0, n, factor)]


def _rebin_max(values: List[int], factor: int) -> List[int]:
    if factor <= 1:
        return list(values)
    n = int(len(values))
    if n % factor != 0:
        raise ValueError(f"length not divisible by factor: {n} % {factor} != 0")
    return [int(max(values[i : i + factor])) for i in range(0, n, factor)]


def _maybe_rebin_pha_for_xspec(*, obsid: str, inst_tag: str, ds: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    rebin = ds.get("spectrum_rebin") if isinstance(ds.get("spectrum_rebin"), dict) else {}
    if str(rebin.get("status") or "") != "rebinned":
        return {"spec": str(ds.get("spec") or ""), "bkg": str(ds.get("bkg") or ""), "rebinned": False}

    factor = int(rebin.get("factor") or 0)
    n_out = int(rebin.get("n_out") or 0)
    if factor <= 1 or n_out <= 0:
        return {"spec": str(ds.get("spec") or ""), "bkg": str(ds.get("bkg") or ""), "rebinned": False}

    spec_in = (_ROOT / str(ds.get("spec") or "")).resolve()
    bkg_in = (_ROOT / str(ds.get("bkg") or "")).resolve() if str(ds.get("bkg") or "") else None
    if not spec_in.exists():
        raise FileNotFoundError(f"missing spectrum: {spec_in}")

    spec_out = out_dir / f"xmm_{obsid}__{inst_tag}__spec_rebin{n_out}.pha"
    bkg_out = out_dir / f"xmm_{obsid}__{inst_tag}__bkg_rebin{n_out}.pha" if bkg_in and bkg_in.exists() else None

    seed = _read_pha_header_seed(spec_in)
    cols = _read_ogip_spectrum_columns(spec_in)
    ch_in = cols.get("CHANNEL", [])
    counts_in = cols.get("COUNTS", [])
    qual_in = cols.get("QUALITY")
    if len(ch_in) != len(counts_in) or len(ch_in) <= 0:
        raise ValueError("invalid spectrum columns")
    if len(ch_in) != int(n_out) * int(factor):
        raise ValueError(f"unexpected spectrum length: {len(ch_in)} (expected {n_out}*{factor})")

    # Ensure standard 0..N-1 channel numbering.
    order = sorted(range(len(ch_in)), key=lambda i: ch_in[i])
    ch_sorted = [int(ch_in[i]) for i in order]
    if ch_sorted != list(range(len(ch_sorted))):
        raise ValueError("unexpected CHANNEL numbering (expected 0..N-1)")
    counts_sorted = [int(counts_in[i]) for i in order]
    counts_out = _rebin_sum(counts_sorted, factor)

    quality_out: Optional[List[int]] = None
    grouping_out: Optional[List[int]] = None
    if isinstance(qual_in, list) and len(qual_in) == len(ch_in):
        qual_sorted = [int(qual_in[i]) for i in order]
        quality_out = _rebin_max(qual_sorted, factor)
        grouping_out = [1] * int(n_out)

    _write_ogip_pha(
        spec_out,
        channels=list(range(int(n_out))),
        counts=counts_out,
        grouping=grouping_out,
        quality=quality_out,
        detchans=int(n_out),
        seed=seed,
        include_links=True,
    )

    if bkg_out is not None and bkg_in is not None and bkg_in.exists():
        seed_b = _read_pha_header_seed(bkg_in)
        cols_b = _read_ogip_spectrum_columns(bkg_in)
        ch_b = cols_b.get("CHANNEL", [])
        counts_b = cols_b.get("COUNTS", [])
        if len(ch_b) != len(counts_b) or len(ch_b) <= 0:
            raise ValueError("invalid background columns")
        if len(ch_b) != int(n_out) * int(factor):
            raise ValueError(f"unexpected background length: {len(ch_b)} (expected {n_out}*{factor})")
        order_b = sorted(range(len(ch_b)), key=lambda i: ch_b[i])
        ch_sorted_b = [int(ch_b[i]) for i in order_b]
        if ch_sorted_b != list(range(len(ch_sorted_b))):
            raise ValueError("unexpected background CHANNEL numbering (expected 0..N-1)")
        counts_sorted_b = [int(counts_b[i]) for i in order_b]
        counts_out_b = _rebin_sum(counts_sorted_b, factor)
        _write_ogip_pha(
            bkg_out,
            channels=list(range(int(n_out))),
            counts=counts_out_b,
            grouping=None,
            quality=None,
            detchans=int(n_out),
            seed=seed_b,
            include_links=False,
        )

    return {
        "spec": _rel(spec_out),
        "bkg": _rel(bkg_out) if bkg_out is not None else "",
        "rebinned": True,
        "factor": int(factor),
        "n_out": int(n_out),
    }


def _build_xspec_xcm_diskline(
    *,
    spec_path: Path,
    bkg_path: Optional[Path],
    rmf_path: Optional[Path],
    arf_path: Optional[Path],
    band_keV: Tuple[float, float],
    par_rin: int,
) -> str:
    lo, hi = float(band_keV[0]), float(band_keV[1])

    def q(p: Optional[Path]) -> str:
        if p is None:
            return ""
        return f"\"{p.resolve().as_posix()}\""

    lines: List[str] = [
        "query yes",
        "chatter 5",
        "log chatter 5",
        f"data {q(spec_path)}",
    ]
    if bkg_path is not None and bkg_path.exists():
        lines.append(f"backgrnd {q(bkg_path)}")
    if rmf_path is not None and rmf_path.exists():
        lines.append(f"response {q(rmf_path)}")
    if arf_path is not None and arf_path.exists():
        lines.append(f"arf {q(arf_path)}")
    lines.extend(
        [
            f"ignore **-{lo} {hi}-**",
            "statistic chi",
            "model tbabs*(powerlaw+diskline)",
            # Conservative initial values (placeholders; refined in later steps).
            "0.5",  # tbabs nH
            "1.7",  # powerlaw PhoIndex
            "1e-2",  # powerlaw norm
            "6.4",  # diskline LineE
            "-2.0",  # diskline Beta
            "6.0",  # diskline Rin
            "400.0",  # diskline Rout
            "30.0",  # diskline Incl
            "1e-4",  # diskline norm
            "fit",
            "tclout stat",
            'puts "FIT_STAT=$xspec_tclout"',
            "tclout dof",
            'puts "FIT_DOF=$xspec_tclout"',
            f"tclout param {par_rin}",
            'puts "RIN_PARAM=$xspec_tclout"',
            # 1σ interval (Δχ²=1.0 for 1 parameter).
            f"error 1.0 {par_rin}",
            f"tclout error {par_rin}",
            'puts "RIN_ERR=$xspec_tclout"',
            "exit",
        ]
    )
    return "\n".join(lines) + "\n"


def _run_xspec_fit_diskline_cli(
    *,
    xspec_bin: Path,
    xcm_path: Path,
    spec_path: Path,
    bkg_path: Optional[Path],
    rmf_path: Optional[Path],
    arf_path: Optional[Path],
    band_keV: Tuple[float, float],
    par_rin: int,
    timeout_s: int,
) -> Dict[str, Any]:
    xcm_text = _build_xspec_xcm_diskline(
        spec_path=spec_path,
        bkg_path=bkg_path,
        rmf_path=rmf_path,
        arf_path=arf_path,
        band_keV=band_keV,
        par_rin=par_rin,
    )
    xcm_path.parent.mkdir(parents=True, exist_ok=True)
    xcm_path.write_text(xcm_text, encoding="utf-8")

    # Prefer running xspec in stdin mode to avoid interactive prompts.
    # The xcm ends with "exit", so xspec should terminate without extra input.
    try:
        res = subprocess.run(
            [str(xspec_bin), "-"],
            input=f"@{xcm_path.resolve().as_posix()}\n",
            text=True,
            capture_output=True,
            cwd=str(_ROOT),
            timeout=max(1, int(timeout_s)),
            check=False,
        )
    except Exception as e:
        return {"status": "fail", "error": f"xspec_cli_run_failed: {e}", "method_tag": "xspec_diskline_v2"}

    stdout = (res.stdout or "").strip()
    stderr = (res.stderr or "").strip()
    if res.returncode != 0:
        return {
            "status": "fail",
            "error": f"xspec_cli_returncode={res.returncode}",
            "stdout_tail": stdout[-4000:],
            "stderr_tail": stderr[-4000:],
            "method_tag": "xspec_diskline_v2",
        }

    stat_s = _parse_xspec_tclout_line(stdout, "FIT_STAT")
    dof_s = _parse_xspec_tclout_line(stdout, "FIT_DOF")
    rin_param_s = _parse_xspec_tclout_line(stdout, "RIN_PARAM")
    rin_err_s = _parse_xspec_tclout_line(stdout, "RIN_ERR")

    if not stat_s or not dof_s or not rin_param_s:
        return {
            "status": "fail",
            "error": "xspec_missing_tclout",
            "stdout_tail": stdout[-4000:],
            "stderr_tail": stderr[-4000:],
            "method_tag": "xspec_diskline_v2",
        }

    stat = float("nan")
    dof = 0
    try:
        stat = float(_parse_float_list(stat_s)[0])
    except Exception:
        pass
    try:
        dof = int(float(_parse_float_list(dof_s)[0]))
    except Exception:
        dof = 0

    param_vals = _parse_float_list(rin_param_s)
    # tclout param returns: (value, delta, min, low, high, max)
    r_in = float(param_vals[0]) if len(param_vals) >= 1 else float("nan")
    p_min = float(param_vals[2]) if len(param_vals) >= 3 else float("nan")
    p_max = float(param_vals[5]) if len(param_vals) >= 6 else float("nan")

    err_vals = _parse_float_list(rin_err_s)
    err_flags_m = _ERR_FLAGS_RE.search(rin_err_s)
    err_flags = err_flags_m.group(0) if err_flags_m else ""

    # tclout error returns: low high + 9-letter flags string (see XSPEC manual).
    err_low = float(err_vals[0]) if len(err_vals) >= 1 else float("nan")
    err_high = float(err_vals[1]) if len(err_vals) >= 2 else float("nan")
    r_in_stat = float("nan")
    if math.isfinite(r_in) and math.isfinite(err_low) and math.isfinite(err_high):
        # XSPEC may report either absolute bounds (low < value < high) or deltas (low < 0 < high).
        # When pegged at a hard limit, it may return 0 0 with flags; treat as "no estimate".
        if err_low == 0.0 and err_high == 0.0:
            r_in_stat = float("nan")
        elif err_low < 0.0 and err_high > 0.0:
            r_in_stat = max(abs(err_low), abs(err_high))
        elif err_low <= r_in <= err_high:
            r_in_stat = max(abs(r_in - err_low), abs(err_high - r_in))
        else:
            r_in_stat = max(abs(err_low), abs(err_high))

    r_in_bound = ""
    if err_flags and len(err_flags) == 9:
        # 4: hit hard lower limit, 5: hit hard upper limit (1-indexed in manual).
        if err_flags[3] == "T":
            r_in_bound = "lower"
        elif err_flags[4] == "T":
            r_in_bound = "upper"
    # Fallback bound detection from hard limits.
    if not r_in_bound and math.isfinite(r_in):
        if math.isfinite(p_min) and abs(r_in - p_min) <= 1e-6:
            r_in_bound = "lower"
        elif math.isfinite(p_max) and abs(r_in - p_max) <= 1e-6:
            r_in_bound = "upper"
    if r_in_bound:
        r_in_stat = float("nan")

    redchi2 = float(stat / dof) if dof > 0 and math.isfinite(stat) else float("nan")
    if not (math.isfinite(stat) and dof > 0 and math.isfinite(r_in)):
        return {
            "status": "fail",
            "error": "xspec_non_finite_fit",
            "stdout_tail": stdout[-4000:],
            "stderr_tail": stderr[-4000:],
            "method_tag": "xspec_diskline_v2",
        }
    return {
        "status": "ok",
        "band_keV": [float(band_keV[0]), float(band_keV[1])],
        "fit": {"chi2": stat, "dof": int(dof), "redchi2": redchi2},
        "r_in_rg": r_in,
        "r_in_rg_stat": r_in_stat,
        "r_in_rg_sys": float("nan"),
        "r_in_bound": r_in_bound,
        "xspec_cli": {
            "xspec_bin": xspec_bin.resolve().as_posix(),
            "xcm": _rel(xcm_path),
            "par_rin": int(par_rin),
            "rin_param_raw": rin_param_s,
            "rin_err_raw": rin_err_s,
            "rin_err_flags": err_flags,
        },
        "stdout_tail": stdout[-4000:],
        "stderr_tail": stderr[-4000:],
        "method_tag": "xspec_diskline_v2",
    }


def _run_xspec_fit_diskline(
    *,
    spec_path: Path,
    bkg_path: Optional[Path],
    rmf_path: Optional[Path],
    arf_path: Optional[Path],
    band_keV: Tuple[float, float],
) -> Dict[str, Any]:
    # Import here to avoid hard dependency.
    from xspec import AllData, AllModels, Fit, Model, Spectrum, Xset  # type: ignore

    AllData.clear()
    AllModels.clear()

    Xset.chatter = 5
    Xset.logChatter = 5

    s = Spectrum(str(spec_path))
    if bkg_path is not None and bkg_path.exists():
        s.background = str(bkg_path)
    if rmf_path is not None and rmf_path.exists():
        s.response = str(rmf_path)
    if arf_path is not None and arf_path.exists():
        try:
            s.response.arf = str(arf_path)
        except Exception:
            # Some XSPEC setups may not allow setting ARF via response. Keep as-is.
            pass

    lo, hi = float(band_keV[0]), float(band_keV[1])
    AllData.ignore(f"**-{lo} {hi}-**")

    # Minimal starter model. (Full reflection models are handled in later steps.)
    m = Model("tbabs*(powerlaw+diskline)")

    # Conservative initial values (GX 339-4 like; placeholders).
    try:
        m.tbabs.nH.values = [0.5]
    except Exception:
        pass
    try:
        m.powerlaw.PhoIndex.values = [1.7]
    except Exception:
        pass
    try:
        m.powerlaw.norm.values = [1e-2]
    except Exception:
        pass
    try:
        m.diskline.LineE.values = [6.4]
        m.diskline.Beta.values = [-2.0]
        m.diskline.Rin.values = [6.0]
        m.diskline.Rout.values = [400.0]
        m.diskline.Incl.values = [30.0]
        m.diskline.norm.values = [1e-4]
    except Exception:
        pass

    Fit.statMethod = "chi"
    Fit.query = "yes"
    Fit.nIterations = 200
    Fit.perform()

    chi2 = float(getattr(Fit, "statistic", float("nan")))
    dof = int(getattr(Fit, "dof", 0) or 0)
    redchi2 = float(chi2 / dof) if dof > 0 and chi2 == chi2 else float("nan")

    r_in = float("nan")
    r_in_stat = float("nan")
    r_in_bound = ""

    try:
        r_in = float(m.diskline.Rin.values[0])
    except Exception:
        pass

    # Try XSPEC error command (may fail depending on environment).
    try:
        # diskline Rin is typically the 7th parameter for tbabs*(powerlaw+diskline),
        # but we avoid relying on global indices; use a best-effort error call.
        Fit.error("1.0 7")
        # After Fit.error, the parameter stores (low, high) bounds in values[2:4] in some setups.
        # Keep stat as NaN if we can't extract it robustly.
    except Exception:
        pass

    return {
        "status": "ok",
        "band_keV": [lo, hi],
        "fit": {"chi2": chi2, "dof": dof, "redchi2": redchi2},
        "r_in_rg": r_in,
        "r_in_rg_stat": r_in_stat,
        "r_in_rg_sys": float("nan"),
        "r_in_bound": r_in_bound,
        "method_tag": "xspec_diskline_v2",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--targets",
        default=str(_ROOT / "data/xrism/sources/xmm_nustar_targets.json"),
        help="Target seed JSON (default: data/xrism/sources/xmm_nustar_targets.json)",
    )
    p.add_argument(
        "--detail-dir",
        default=str(_ROOT / "output/private/xrism"),
        help="Directory containing per-obsid detail JSONs (default: output/private/xrism)",
    )
    p.add_argument(
        "--out-dir",
        default=str(_ROOT / "output/private/xrism"),
        help="Output directory (default: output/private/xrism)",
    )
    p.add_argument("--obsid", default="", help="Optional: run only for this obsid (XMM).")
    p.add_argument("--band-lo", type=float, default=3.0, help="Fit band lower edge (keV).")
    p.add_argument("--band-hi", type=float, default=10.0, help="Fit band upper edge (keV).")
    p.add_argument("--dry-run", action="store_true", help="Only write plan JSON (do not run XSPEC).")
    p.add_argument(
        "--xspec-bin",
        default="",
        help="Optional: XSPEC executable (for CLI fallback). If omitted, tries to find `xspec` in PATH.",
    )
    p.add_argument("--prefer-cli", action="store_true", help="Prefer XSPEC CLI even if pyXspec is available.")
    p.add_argument("--timeout-s", type=int, default=900, help="Timeout for XSPEC CLI run (seconds).")
    args = p.parse_args(list(argv) if argv is not None else None)

    targets_path = Path(args.targets)
    detail_dir = Path(args.detail_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = _read_json(targets_path)
    items = targets.get("targets", [])
    if not isinstance(items, list) or not items:
        raise RuntimeError(f"no targets in {targets_path}")

    pyxspec_mod, pyxspec_err = _try_import_xspec()
    pyxspec_available = pyxspec_mod is not None
    xspec_bin = _find_xspec_cli(str(args.xspec_bin))
    xspec_cli_available = xspec_bin is not None
    xspec_available = pyxspec_available or xspec_cli_available

    band = (float(args.band_lo), float(args.band_hi))
    only_obsid = str(args.obsid or "").strip()
    timeout_s = int(args.timeout_s)

    # For tbabs*(powerlaw+diskline), Rin is expected at parameter 6:
    # 1:tbabs.nH, 2:powerlaw.PhoIndex, 3:powerlaw.norm,
    # 4:diskline.LineE, 5:diskline.Beta, 6:diskline.Rin, ...
    par_rin = 6

    blocked_any = False
    fail_any = False
    n_written = 0

    for t in items:
        if not isinstance(t, dict):
            continue
        xmm_list = t.get("xmm", [])
        if not isinstance(xmm_list, list):
            continue

        for d in xmm_list:
            if not isinstance(d, dict):
                continue
            obsid = str(d.get("obsid") or "").strip()
            if not obsid:
                continue
            if only_obsid and obsid != only_obsid:
                continue

            detail_path = detail_dir / f"xmm_{obsid}__fek_broad_line_rmf_diskline.json"
            out_path = out_dir / f"xmm_{obsid}__fek_broad_line_reflection_xspec.json"
            xcm_path = out_dir / f"xmm_{obsid}__fek_broad_line_reflection_xspec.xcm"

            detail = _read_json(detail_path) if detail_path.exists() else {}
            rmf_sources = _load_rmf_sources_from_detail(detail)

            plan = {
                "generated_utc": _utc_now(),
                "targets": _rel(targets_path),
                "obsid": obsid,
                "band_keV": [float(band[0]), float(band[1])],
                "detail_json": _rel(detail_path) if detail_path.exists() else "",
                "datasets": rmf_sources,
                "xspec": {
                    "pyxspec_available": bool(pyxspec_available),
                    "pyxspec_import_error": pyxspec_err if not pyxspec_available else "",
                    "cli_available": bool(xspec_cli_available),
                    "cli_xspec_bin": xspec_bin.resolve().as_posix() if xspec_bin else "",
                    "cli_timeout_s": int(timeout_s),
                    "cli_xcm": _rel(xcm_path),
                    "par_rin": int(par_rin),
                },
            }

            if bool(args.dry_run):
                # Do not overwrite an already-fixed result with a dry-run placeholder.
                prev_ok = False
                if out_path.exists():
                    prev = _read_json(out_path)
                    prev_status = str(prev.get("status") or "").strip().lower()
                    prev_ok = prev_status == "ok"

                # Always freeze the XSPEC command plan as an .xcm file.
                if rmf_sources:
                    ds = rmf_sources[0]
                    spec_path = _ROOT / str(ds["spec"])
                    bkg_path = (_ROOT / str(ds["bkg"])) if ds.get("bkg") else None
                    rmf_path = (_ROOT / str(ds["rmf"])) if ds.get("rmf") else None
                    arf_path = (_ROOT / str(ds["arf"])) if ds.get("arf") else None

                    inst_tag = "m1" if "M1" in str(spec_path.name).upper() else "m2"
                    rebin_artifacts = _maybe_rebin_pha_for_xspec(obsid=obsid, inst_tag=inst_tag, ds=ds, out_dir=out_dir)
                    plan["xspec"]["pha_rebin"] = rebin_artifacts
                    if rebin_artifacts.get("rebinned") and str(rebin_artifacts.get("spec") or ""):
                        spec_path = _ROOT / str(rebin_artifacts["spec"])
                    if rebin_artifacts.get("rebinned") and str(rebin_artifacts.get("bkg") or ""):
                        bkg_path = _ROOT / str(rebin_artifacts["bkg"])

                    xcm_text = _build_xspec_xcm_diskline(
                        spec_path=spec_path,
                        bkg_path=bkg_path,
                        rmf_path=rmf_path,
                        arf_path=arf_path,
                        band_keV=band,
                        par_rin=par_rin,
                    )
                    xcm_path.parent.mkdir(parents=True, exist_ok=True)
                    xcm_path.write_text(xcm_text, encoding="utf-8")

                obj = {
                    "status": "dry_run",
                    "generated_utc": _utc_now(),
                    "obsid": obsid,
                    "method_tag": "xspec_diskline_v2",
                    "xspec_available": bool(xspec_available),
                    "xspec_import_error": pyxspec_err if not pyxspec_available else "",
                    "plan": plan,
                }
                if not prev_ok:
                    _write_json(out_path, obj)
                    n_written += 1
                continue

            if not xspec_available:
                obj = {
                    "status": "blocked_missing_xspec",
                    "generated_utc": _utc_now(),
                    "obsid": obsid,
                    "method_tag": "xspec_diskline_v2",
                    "blocked_reason": f"missing_pyxspec: {pyxspec_err}; missing_xspec_cli: xspec_not_found_in_path",
                    "plan": plan,
                }
                _write_json(out_path, obj)
                n_written += 1
                blocked_any = True
                continue

            if not rmf_sources:
                obj = {
                    "status": "fail",
                    "generated_utc": _utc_now(),
                    "obsid": obsid,
                    "method_tag": "xspec_diskline_v2",
                    "error": f"missing_or_invalid_detail_json: {detail_path}",
                    "plan": plan,
                }
                _write_json(out_path, obj)
                n_written += 1
                fail_any = True
                continue

            # Fit MOS1 first when available (keep the minimal workflow stable).
            ds = rmf_sources[0]
            spec_path = _ROOT / str(ds["spec"])
            bkg_path = (_ROOT / str(ds["bkg"])) if ds.get("bkg") else None
            rmf_path = (_ROOT / str(ds["rmf"])) if ds.get("rmf") else None
            arf_path = (_ROOT / str(ds["arf"])) if ds.get("arf") else None

            try:
                inst_tag = "m1" if "M1" in str(spec_path.name).upper() else "m2"
                rebin_artifacts = _maybe_rebin_pha_for_xspec(obsid=obsid, inst_tag=inst_tag, ds=ds, out_dir=out_dir)
                plan["xspec"]["pha_rebin"] = rebin_artifacts
                if rebin_artifacts.get("rebinned") and str(rebin_artifacts.get("spec") or ""):
                    spec_path = _ROOT / str(rebin_artifacts["spec"])
                if rebin_artifacts.get("rebinned") and str(rebin_artifacts.get("bkg") or ""):
                    bkg_path = _ROOT / str(rebin_artifacts["bkg"])

                if bool(args.prefer_cli) or not pyxspec_available:
                    if xspec_bin is None:
                        raise RuntimeError("xspec_cli_not_found")
                    fit = _run_xspec_fit_diskline_cli(
                        xspec_bin=xspec_bin,
                        xcm_path=xcm_path,
                        spec_path=spec_path,
                        bkg_path=bkg_path,
                        rmf_path=rmf_path,
                        arf_path=arf_path,
                        band_keV=band,
                        par_rin=par_rin,
                        timeout_s=timeout_s,
                    )
                else:
                    fit = _run_xspec_fit_diskline(
                        spec_path=spec_path,
                        bkg_path=bkg_path,
                        rmf_path=rmf_path,
                        arf_path=arf_path,
                        band_keV=band,
                    )
                obj = {
                    "generated_utc": _utc_now(),
                    "obsid": obsid,
                    "status": str(fit.get("status") or "ok"),
                    "method_tag": str(fit.get("method_tag") or "xspec_diskline_v2"),
                    "xspec_mode": "cli" if "xspec_cli" in fit else "pyxspec",
                    "inputs": {
                        "spec": _rel(spec_path),
                        "bkg": _rel(bkg_path) if bkg_path else "",
                        "rmf": _rel(rmf_path) if rmf_path else "",
                        "arf": _rel(arf_path) if arf_path else "",
                    },
                    "fit": fit.get("fit", {}),
                    "band_keV": fit.get("band_keV", list(plan["band_keV"])),
                    "r_in_rg": fit.get("r_in_rg"),
                    "r_in_rg_stat": fit.get("r_in_rg_stat"),
                    "r_in_rg_sys": fit.get("r_in_rg_sys"),
                    "r_in_bound": fit.get("r_in_bound", ""),
                    "xspec_cli": fit.get("xspec_cli", {}),
                    "stdout_tail": fit.get("stdout_tail", ""),
                    "stderr_tail": fit.get("stderr_tail", ""),
                    "plan": plan,
                }
                _write_json(out_path, obj)
                n_written += 1
            except Exception as e:
                obj = {
                    "status": "fail",
                    "generated_utc": _utc_now(),
                    "obsid": obsid,
                    "method_tag": "xspec_diskline_v2",
                    "error": str(e),
                    "plan": plan,
                }
                _write_json(out_path, obj)
                n_written += 1
                fail_any = True

    print(json.dumps({"written": n_written, "out_dir": _rel(out_dir)}, ensure_ascii=False))
    if fail_any:
        return 1
    if blocked_any:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
