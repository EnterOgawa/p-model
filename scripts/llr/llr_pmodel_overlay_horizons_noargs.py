#!/usr/bin/env python3
"""
llr_pmodel_overlay_horizons_noargs.py

【引数なしで動く】LLR (CRD v2.*) の Normal Point (record "11") を読み込み、
JPL Horizons API から Earth-centered の Moon/Sun ベクトルを取得して
「観測TOF」と「P-model（太陽Shapiro遅延を含むTOFモデル）」を重ね描画します。

■ 引数なし実行:
  - 実行ディレクトリ配下（このスクリプトと同じフォルダ）から
    *.crd / *.npt / *.crd.gz / *.npt.gz を再帰検索し、先頭1件を処理
  - 見つからなければ demo_llr_like.crd があればそれを使う

■ 重要（検証用の割り切り）
  - 地上局/反射器の厳密座標や潮汐/秤動等は入れず、Earth center ↔ Moon center の距離を使用。
  - 絶対値はズレるので、観測とモデルの間に「定数オフセット（平均差）」を1つ入れて整列。
  - まずは “太陽配置依存の変動成分の整合” を見る用途。

■ 出力（固定: output/private/llr/out_llr/）
  output/private/llr/out_llr/<stem>_overlay_tof.png
  output/private/llr/out_llr/<stem>_residual.png
  output/private/llr/out_llr/<stem>_table.csv

■ SSL/Proxy 対応
  会社VPN/ProxyでSSL中継されている場合、Pythonが社内CAを信頼せずに失敗することがあります。
  その場合は以下のいずれかを設定してください。

  推奨: 社内CA(Proxy CA) PEMを指定
    set HORIZONS_CA_BUNDLE=C:\\certs\\corp_root.pem
    py llr_pmodel_overlay_horizons_noargs.py

  最終手段（危険）: 検証を無効化
    set HORIZONS_INSECURE=1
    py llr_pmodel_overlay_horizons_noargs.py
"""

from __future__ import annotations

import os
import math
import ssl
import gzip
import io
import re
import hashlib
import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Defaults (no-args mode)
# -------------------------
DEFAULT_BETA = 1.0
DEFAULT_OUTDIR = "out_llr"
DEFAULT_CHUNK = 50                 # Horizons TLIST chunk size (smaller avoids proxy/URL limits)
ASSUME_TWO_WAY_IF_MISSING = True   # LLRは通常 two-way
SEARCH_RECURSIVE = True

LLR_SHORT_NAME = "月レーザー測距（LLR: Lunar Laser Ranging）"

# -------------------------
# Physical constants
# -------------------------
C = 299_792_458.0  # m/s
GM_SUN = 1.32712440018e20  # m^3/s^2 (commonly used)
HORIZONS_API = "https://ssd.jpl.nasa.gov/api/horizons.api"
WGS84_A = 6_378_137.0
WGS84_F = 1.0 / 298.257223563
REFLECTOR_CATALOG_REL = Path("data") / "llr" / "reflectors_de421_pa.json"
SPICE_KERNEL_DIR_REL = Path("data") / "llr" / "kernels" / "naif"
POS_EOP_SNX_DIR_REL = Path("data") / "llr" / "pos_eop" / "snx"

# Horizons prints epochs with 4 decimal digits in seconds (0.0001s).
# For stable merging/caching, we round inputs to this resolution.
HORIZONS_TIME_QUANTUM_US = 100


# ============================================================
# EDC pos+eop (SINEX) helper: station coords / EOP primary source
# ============================================================
_POS_EOP_PARSED_CACHE: Dict[Path, Dict[str, Any]] = {}


# 関数: `_parse_yymmdd_to_date` の入出力契約と処理意図を定義する。
def _parse_yymmdd_to_date(yymmdd: str) -> Optional[datetime]:
    """
    Parse YYMMDD into UTC datetime at 00:00.
    Returns None if invalid.
    """
    s = str(yymmdd).strip()
    # 条件分岐: `not re.fullmatch(r"\d{6}", s)` を満たす経路を評価する。
    if not re.fullmatch(r"\d{6}", s):
        return None

    yy = int(s[0:2])
    yyyy = 2000 + yy if yy < 80 else 1900 + yy
    mm = int(s[2:4])
    dd = int(s[4:6])
    try:
        return datetime(yyyy, mm, dd, tzinfo=timezone.utc)
    except Exception:
        return None


# 関数: `_pos_eop_snx_path` の入出力契約と処理意図を定義する。

def _pos_eop_snx_path(repo_root: Path, yymmdd: str) -> Path:
    dt = _parse_yymmdd_to_date(yymmdd)
    # 条件分岐: `dt is None` を満たす経路を評価する。
    if dt is None:
        raise ValueError(f"invalid yymmdd: {yymmdd!r}")

    year = dt.year
    return repo_root / POS_EOP_SNX_DIR_REL / str(year) / str(yymmdd) / f"pos_eop_{yymmdd}.snx.gz"


# 関数: `_iter_cached_pos_eop_yymmdd` の入出力契約と処理意図を定義する。

def _iter_cached_pos_eop_yymmdd(repo_root: Path) -> List[str]:
    root = repo_root / POS_EOP_SNX_DIR_REL
    # 条件分岐: `not root.exists()` を満たす経路を評価する。
    if not root.exists():
        return []

    yymmdd: set[str] = set()
    for p in root.glob("*/*/pos_eop_*.snx.gz"):
        m = re.search(r"pos_eop_(\d{6})\.snx\.gz$", p.name, re.IGNORECASE)
        # 条件分岐: `m` を満たす経路を評価する。
        if m:
            yymmdd.add(m.group(1))

    return sorted(yymmdd)


# 関数: `_nearest_pos_eop_yymmdd` の入出力契約と処理意図を定義する。

def _nearest_pos_eop_yymmdd(repo_root: Path, target_dt: datetime, *, max_days: int = 45) -> Optional[str]:
    # 条件分岐: `target_dt.tzinfo is None` を満たす経路を評価する。
    if target_dt.tzinfo is None:
        target_dt = target_dt.replace(tzinfo=timezone.utc)

    target_dt = target_dt.astimezone(timezone.utc)
    yymmdd_all = _iter_cached_pos_eop_yymmdd(repo_root)
    # 条件分岐: `not yymmdd_all` を満たす経路を評価する。
    if not yymmdd_all:
        return None

    best: Optional[tuple[int, str]] = None
    for y in yymmdd_all:
        d0 = _parse_yymmdd_to_date(y)
        # 条件分岐: `d0 is None` を満たす経路を評価する。
        if d0 is None:
            continue

        days = abs((d0.date() - target_dt.date()).days)
        # 条件分岐: `days > int(max_days)` を満たす経路を評価する。
        if days > int(max_days):
            continue

        # 条件分岐: `best is None or days < best[0]` を満たす経路を評価する。

        if best is None or days < best[0]:
            best = (days, y)

    return best[1] if best else None


# 関数: `_nearest_pos_eop_yymmdd_for_code` の入出力契約と処理意図を定義する。

def _nearest_pos_eop_yymmdd_for_code(
    repo_root: Path,
    target_dt: datetime,
    *,
    code_key: str,
    max_days: int = 45,
) -> Optional[str]:
    """
    Return nearest cached YYMMDD (within max_days) that actually contains station code_key.

    This is a fallback for cases where the globally-nearest pos+eop day exists but does not
    include the requested station in SOLUTION/ESTIMATE.
    """
    # 条件分岐: `target_dt.tzinfo is None` を満たす経路を評価する。
    if target_dt.tzinfo is None:
        target_dt = target_dt.replace(tzinfo=timezone.utc)

    target_dt = target_dt.astimezone(timezone.utc)

    yymmdd_all = _iter_cached_pos_eop_yymmdd(repo_root)
    # 条件分岐: `not yymmdd_all` を満たす経路を評価する。
    if not yymmdd_all:
        return None

    candidates: List[tuple[int, str]] = []
    for y in yymmdd_all:
        d0 = _parse_yymmdd_to_date(y)
        # 条件分岐: `d0 is None` を満たす経路を評価する。
        if d0 is None:
            continue

        days = abs((d0.date() - target_dt.date()).days)
        # 条件分岐: `days > int(max_days)` を満たす経路を評価する。
        if days > int(max_days):
            continue

        candidates.append((days, y))

    # Sort by nearest first, and for ties prefer newer dates.

    candidates.sort(key=lambda x: (x[0], -int(x[1])))

    for _, y in candidates:
        snx_path = _pos_eop_snx_path(repo_root, y)
        # 条件分岐: `not snx_path.exists()` を満たす経路を評価する。
        if not snx_path.exists():
            continue

        parsed = parse_pos_eop_snx_gz(snx_path)
        stations = parsed.get("stations") if isinstance(parsed, dict) else None
        # 条件分岐: `not isinstance(stations, dict)` を満たす経路を評価する。
        if not isinstance(stations, dict):
            continue

        rec = stations.get(code_key)
        # 条件分岐: `isinstance(rec, dict) and all(k in rec for k in ("x_m", "y_m", "z_m"))` を満たす経路を評価する。
        if isinstance(rec, dict) and all(k in rec for k in ("x_m", "y_m", "z_m")):
            return y

    return None


# 関数: `_sinex_epoch_to_utc` の入出力契約と処理意図を定義する。

def _sinex_epoch_to_utc(epoch: str) -> Optional[datetime]:
    # YY:DOY:SSSSS (seconds of day) -> UTC
    s = str(epoch).strip()
    m = re.fullmatch(r"(\d{2}):(\d{3}):([0-9.]+)", s)
    # 条件分岐: `not m` を満たす経路を評価する。
    if not m:
        return None

    yy = int(m.group(1))
    yyyy = 2000 + yy if yy < 80 else 1900 + yy
    doy = int(m.group(2))
    sec = float(m.group(3))
    try:
        return datetime(yyyy, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1, seconds=sec)
    except Exception:
        return None


# 関数: `parse_pos_eop_snx_gz` の入出力契約と処理意図を定義する。

def parse_pos_eop_snx_gz(path: Path) -> Dict[str, Any]:
    """
    Parse a pos+eop SINEX (.snx.gz) and return a compact dict:
      - stations: { "<CODE>": {"x_m","y_m","z_m","std_x_m","std_y_m","std_z_m","ref_epoch_utc"} }
      - eop: { "XPO":[...], "YPO":[...], "LOD":[...] } (best-effort; not required by current LLR model)
    """
    # 条件分岐: `path in _POS_EOP_PARSED_CACHE` を満たす経路を評価する。
    if path in _POS_EOP_PARSED_CACHE:
        return _POS_EOP_PARSED_CACHE[path]

    stations: Dict[str, Dict[str, Any]] = {}
    eop: Dict[str, List[Dict[str, Any]]] = {"XPO": [], "YPO": [], "LOD": []}

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        in_est = False
        for raw in f:
            line = raw.rstrip("\n")
            # 条件分岐: `line.startswith("+SOLUTION/ESTIMATE")` を満たす経路を評価する。
            if line.startswith("+SOLUTION/ESTIMATE"):
                in_est = True
                continue

            # 条件分岐: `in_est and line.startswith("-SOLUTION/ESTIMATE")` を満たす経路を評価する。

            if in_est and line.startswith("-SOLUTION/ESTIMATE"):
                break

            # 条件分岐: `not in_est` を満たす経路を評価する。

            if not in_est:
                continue

            # 条件分岐: `not line.strip() or line.startswith("*")` を満たす経路を評価する。

            if not line.strip() or line.startswith("*"):
                continue

            parts = line.split()
            # Expected: index, type, code, pt, soln, ref_epoch, unit, S, value, std
            if len(parts) < 10:
                continue

            typ = parts[1].strip().upper()
            code = parts[2].strip()
            ref_epoch = parts[5].strip()
            try:
                value = float(parts[-2])
                std = float(parts[-1])
            except Exception:
                continue

            ref_dt = _sinex_epoch_to_utc(ref_epoch)

            # 条件分岐: `typ in ("STAX", "STAY", "STAZ")` を満たす経路を評価する。
            if typ in ("STAX", "STAY", "STAZ"):
                rec = stations.setdefault(code, {})
                # 条件分岐: `typ == "STAX"` を満たす経路を評価する。
                if typ == "STAX":
                    rec["x_m"] = float(value)
                    rec["std_x_m"] = float(std)
                # 条件分岐: 前段条件が不成立で、`typ == "STAY"` を追加評価する。
                elif typ == "STAY":
                    rec["y_m"] = float(value)
                    rec["std_y_m"] = float(std)
                # 条件分岐: 前段条件が不成立で、`typ == "STAZ"` を追加評価する。
                elif typ == "STAZ":
                    rec["z_m"] = float(value)
                    rec["std_z_m"] = float(std)

                # 条件分岐: `ref_dt is not None` を満たす経路を評価する。

                if ref_dt is not None:
                    rec["ref_epoch_utc"] = ref_dt.isoformat()

                continue

            # 条件分岐: `typ in eop` を満たす経路を評価する。

            if typ in eop:
                item = {
                    "type": typ,
                    "code": code,
                    "ref_epoch_utc": ref_dt.isoformat() if ref_dt else None,
                    "value": float(value),
                    "std": float(std),
                }
                eop[typ].append(item)

    out = {"stations": stations, "eop": eop, "snx_path": str(path)}
    _POS_EOP_PARSED_CACHE[path] = out
    return out


# 関数: `load_station_xyz_from_pos_eop` の入出力契約と処理意図を定義する。

def load_station_xyz_from_pos_eop(
    repo_root: Path,
    *,
    station_code: str,
    pad_id: Optional[int],
    target_dt: datetime,
    max_days: int = 45,
    preferred_yymmdd: str = "",
) -> Optional[Dict[str, Any]]:
    """
    Return station XYZ (ECEF, meters) from cached pos+eop SINEX.

    - If preferred_yymmdd is provided, it is used directly (must be cached).
    - Else, pick the nearest cached yymmdd within max_days of target_dt.

    Returns:
      {"x_m","y_m","z_m","std_x_m","std_y_m","std_z_m","pos_eop_yymmdd","pos_eop_ref_epoch_utc"}
    or None if unavailable.
    """
    # 条件分岐: `pad_id is None` を満たす経路を評価する。
    if pad_id is None:
        return None

    code_key = str(int(pad_id))
    yymmdd = str(preferred_yymmdd).strip()
    # 条件分岐: `yymmdd` を満たす経路を評価する。
    if yymmdd:
        # 条件分岐: `not re.fullmatch(r"\d{6}", yymmdd)` を満たす経路を評価する。
        if not re.fullmatch(r"\d{6}", yymmdd):
            # Accept YYYYMMDD too
            if re.fullmatch(r"\d{8}", yymmdd):
                yymmdd = yymmdd[2:]
            else:
                return None

        snx_path = _pos_eop_snx_path(repo_root, yymmdd)
        # 条件分岐: `not snx_path.exists()` を満たす経路を評価する。
        if not snx_path.exists():
            return None
    else:
        nearest = _nearest_pos_eop_yymmdd(repo_root, target_dt, max_days=int(max_days))
        # 条件分岐: `not nearest` を満たす経路を評価する。
        if not nearest:
            return None

        yymmdd = nearest
        snx_path = _pos_eop_snx_path(repo_root, yymmdd)
        # 条件分岐: `not snx_path.exists()` を満たす経路を評価する。
        if not snx_path.exists():
            return None

    parsed = parse_pos_eop_snx_gz(snx_path)
    stations = parsed.get("stations") if isinstance(parsed, dict) else None
    # 条件分岐: `not isinstance(stations, dict)` を満たす経路を評価する。
    if not isinstance(stations, dict):
        return None

    rec = stations.get(code_key)
    # 条件分岐: `not isinstance(rec, dict)` を満たす経路を評価する。
    if not isinstance(rec, dict):
        # Fallback: nearest cached day that actually includes this station code.
        y2 = _nearest_pos_eop_yymmdd_for_code(
            repo_root,
            target_dt,
            code_key=code_key,
            max_days=int(max_days),
        )
        # 条件分岐: `not y2` を満たす経路を評価する。
        if not y2:
            # Some stations are absent in pos+eop products.
            return None

        yymmdd = y2
        snx_path = _pos_eop_snx_path(repo_root, yymmdd)
        # 条件分岐: `not snx_path.exists()` を満たす経路を評価する。
        if not snx_path.exists():
            return None

        parsed = parse_pos_eop_snx_gz(snx_path)
        stations = parsed.get("stations") if isinstance(parsed, dict) else None
        # 条件分岐: `not isinstance(stations, dict)` を満たす経路を評価する。
        if not isinstance(stations, dict):
            return None

        rec = stations.get(code_key)
        # 条件分岐: `not isinstance(rec, dict)` を満たす経路を評価する。
        if not isinstance(rec, dict):
            return None

    # 条件分岐: `not all(k in rec for k in ("x_m", "y_m", "z_m"))` を満たす経路を評価する。

    if not all(k in rec for k in ("x_m", "y_m", "z_m")):
        return None

    return {
        "station": str(station_code).upper(),
        "pad_id": int(pad_id),
        "pos_eop_yymmdd": str(yymmdd),
        "pos_eop_ref_epoch_utc": rec.get("ref_epoch_utc"),
        "x_m": float(rec["x_m"]),
        "y_m": float(rec["y_m"]),
        "z_m": float(rec["z_m"]),
        "std_x_m": float(rec.get("std_x_m")) if rec.get("std_x_m") is not None else None,
        "std_y_m": float(rec.get("std_y_m")) if rec.get("std_y_m") is not None else None,
        "std_z_m": float(rec.get("std_z_m")) if rec.get("std_z_m") is not None else None,
    }


# 関数: `_set_japanese_font` の入出力契約と処理意図を定義する。

def _set_japanese_font() -> None:
    try:
        import matplotlib as mpl
        import matplotlib.font_manager as fm

        preferred = [
            "Yu Gothic",
            "Meiryo",
            "BIZ UDGothic",
            "MS Gothic",
            "Yu Mincho",
            "MS Mincho",
        ]
        available = {f.name for f in fm.fontManager.ttflist}
        chosen = [name for name in preferred if name in available]
        # 条件分岐: `not chosen` を満たす経路を評価する。
        if not chosen:
            return

        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


# ============================================================
# SPICE (NAIF) helper for high-accuracy lunar orientation (DE421)
# ============================================================

_SPICE_CTX: Optional[Dict[str, Any]] = None


# 関数: `_try_load_spice` の入出力契約と処理意図を定義する。
def _try_load_spice(repo_root: Path) -> Optional[Any]:
    """
    Return spiceypy module after loading required kernels, or None.

    Required (downloaded via scripts/llr/fetch_moon_kernels_naif.py):
      - naif0012.tls
      - pck00010.tpc
      - moon_080317.tf
      - moon_pa_de421_1900-2050.bpc
    """
    global _SPICE_CTX
    # 条件分岐: `_SPICE_CTX is not None` を満たす経路を評価する。
    if _SPICE_CTX is not None:
        return _SPICE_CTX.get("spice")

    try:
        import spiceypy as sp  # type: ignore
    except Exception:
        _SPICE_CTX = {"spice": None, "status": "missing_spiceypy"}
        return None

    kdir = repo_root / SPICE_KERNEL_DIR_REL
    required = {
        "lsk": kdir / "naif0012.tls",
        "pck": kdir / "pck00010.tpc",
        "fk": kdir / "moon_080317.tf",
        "moon_pck": kdir / "moon_pa_de421_1900-2050.bpc",
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    # 条件分岐: `missing` を満たす経路を評価する。
    if missing:
        _SPICE_CTX = {"spice": None, "status": "missing_kernels", "missing": missing}
        return None

    try:
        sp.kclear()
        for p in required.values():
            sp.furnsh(str(p))
    except Exception as e:
        _SPICE_CTX = {"spice": None, "status": "kernel_load_failed", "error": str(e)}
        return None

    _SPICE_CTX = {"spice": sp, "status": "ok", "kernel_dir": str(kdir)}
    return sp


# 関数: `_moon_pa_de421_to_j2000_matrix` の入出力契約と処理意図を定義する。

def _moon_pa_de421_to_j2000_matrix(repo_root: Path, dt_utc: datetime) -> Optional[np.ndarray]:
    """
    High-accuracy Moon Principal Axes (DE421) to J2000 rotation matrix via SPICE.
    Returns None if SPICE kernels are unavailable.
    """
    sp = _try_load_spice(repo_root)
    # 条件分岐: `sp is None` を満たす経路を評価する。
    if sp is None:
        return None

    # SPICE expects UTC string; leap seconds kernel handles UTC->ET.

    t = dt_utc.astimezone(timezone.utc)
    et = sp.str2et(t.strftime("%Y-%m-%dT%H:%M:%S.%f"))
    mat = sp.pxform("MOON_PA_DE421", "J2000", et)
    return np.array(mat, dtype=float)


# ============================================================
# SSL context helper (proxy/self-signed CA handling)
# ============================================================

def _make_ssl_context() -> ssl.SSLContext:
    """
    - HORIZONS_INSECURE=1 なら検証無効（最終手段）
    - HORIZONS_CA_BUNDLE / SSL_CERT_FILE / REQUESTS_CA_BUNDLE があればそれをcafileとして利用
    - それ以外はデフォルトのCAを利用
    """
    # 条件分岐: `os.environ.get("HORIZONS_INSECURE", "").strip() == "1"` を満たす経路を評価する。
    if os.environ.get("HORIZONS_INSECURE", "").strip() == "1":
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx

    cafile = (
        os.environ.get("HORIZONS_CA_BUNDLE")
        or os.environ.get("SSL_CERT_FILE")
        or os.environ.get("REQUESTS_CA_BUNDLE")
    )

    # 条件分岐: `cafile` を満たす経路を評価する。
    if cafile:
        return ssl.create_default_context(cafile=cafile)

    return ssl.create_default_context()


# ============================================================
# Station geometry (WGS84 + GMST; simplified ECEF->ECI)
# ============================================================

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_load_station_geodetic` の入出力契約と処理意図を定義する。

def _load_station_geodetic(repo_root: Path, station_code: Optional[str]) -> Optional[Dict[str, Any]]:
    # 条件分岐: `not station_code` を満たす経路を評価する。
    if not station_code:
        return None

    p = repo_root / "data" / "llr" / "stations" / f"{station_code.lower()}.json"
    # 条件分岐: `not p.exists()` を満たす経路を評価する。
    if not p.exists():
        return None

    try:
        meta = _read_json(p)
    except Exception:
        return None

    # Best-effort: enrich with geocentric XYZ from the cached site log if missing in JSON.

    try:
        # 条件分岐: `isinstance(meta, dict) and not all(meta.get(k) is not None for k in ("x_m", "...` を満たす経路を評価する。
        if isinstance(meta, dict) and not all(meta.get(k) is not None for k in ("x_m", "y_m", "z_m")):
            log_name = str(meta.get("log_filename") or "").strip()
            # 条件分岐: `log_name` を満たす経路を評価する。
            if log_name:
                log_path = repo_root / "data" / "llr" / "stations" / log_name
                # 条件分岐: `log_path.exists()` を満たす経路を評価する。
                if log_path.exists():
                    txt = log_path.read_text(encoding="utf-8", errors="replace")
                    import re

                    mx = re.search(r"X coordinate\s*\[m\]\s*:\s*([0-9.+-]+)", txt, re.IGNORECASE)
                    my = re.search(r"Y coordinate\s*\[m\]\s*:\s*([0-9.+-]+)", txt, re.IGNORECASE)
                    mz = re.search(r"Z coordinate\s*\[m\]\s*:\s*([0-9.+-]+)", txt, re.IGNORECASE)
                    # 条件分岐: `mx and my and mz` を満たす経路を評価する。
                    if mx and my and mz:
                        meta = dict(meta)
                        meta["x_m"] = float(mx.group(1))
                        meta["y_m"] = float(my.group(1))
                        meta["z_m"] = float(mz.group(1))
    except Exception:
        pass

    return meta


# 関数: `_norm_key` の入出力契約と処理意図を定義する。

def _norm_key(s: Optional[str]) -> str:
    # 条件分岐: `not s` を満たす経路を評価する。
    if not s:
        return ""

    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


# 関数: `_load_reflector_pa` の入出力契約と処理意図を定義する。

def _load_reflector_pa(repo_root: Path, target_name: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Load Moon-centered reflector coordinates in the Moon DE421 principal-axis (body-fixed) frame.
    Returns dict with x_m/y_m/z_m, or None if not found.
    """
    p = repo_root / REFLECTOR_CATALOG_REL
    # 条件分岐: `not p.exists()` を満たす経路を評価する。
    if not p.exists():
        return None

    try:
        cat = _read_json(p)
    except Exception:
        return None

    refs = (cat.get("reflectors") or {}) if isinstance(cat, dict) else {}
    # 条件分岐: `not isinstance(refs, dict)` を満たす経路を評価する。
    if not isinstance(refs, dict):
        return None

    key = _norm_key(target_name)
    # 条件分岐: `key in refs and isinstance(refs[key], dict)` を満たす経路を評価する。
    if key in refs and isinstance(refs[key], dict):
        return refs[key]

    # common aliases

    aliases = {
        "apollo11": ["ap11", "a11"],
        "apollo14": ["ap14", "a14"],
        "apollo15": ["ap15", "a15"],
        "luna17": ["lunokhod1", "lk1", "luno17"],
        "luna21": ["lunokhod2", "lk2", "luno21"],
    }
    for canon, al in aliases.items():
        # 条件分岐: `key == canon or key in al` を満たす経路を評価する。
        if key == canon or key in al:
            v = refs.get(canon)
            return v if isinstance(v, dict) else None

    return None


# 関数: `ecef_from_geodetic` の入出力契約と処理意図を定義する。

def ecef_from_geodetic(lat_deg: float, lon_deg: float, h_m: float) -> np.ndarray:
    # WGS84 geodetic -> ECEF (meters)
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    a = WGS84_A
    f = WGS84_F
    e2 = f * (2.0 - f)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)
    N = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    x = (N + h_m) * cos_lat * cos_lon
    y = (N + h_m) * cos_lat * sin_lon
    z = (N * (1.0 - e2) + h_m) * sin_lat
    return np.array([x, y, z], dtype=float)


# 関数: `geodetic_from_ecef` の入出力契約と処理意図を定義する。

def geodetic_from_ecef(x_m: float, y_m: float, z_m: float) -> tuple[float, float, float]:
    """
    ECEF (meters) -> WGS84 geodetic (lon_deg, lat_deg, h_m).

    Note:
      - This is a lightweight iterative inversion (sufficient for converting site-log XYZ to lon/lat/h
        for Horizons' GEODETIC mode).
    """
    a = float(WGS84_A)
    f = float(WGS84_F)
    e2 = f * (2.0 - f)

    x = float(x_m)
    y = float(y_m)
    z = float(z_m)

    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    # 条件分岐: `p < 1e-9` を満たす経路を評価する。
    if p < 1e-9:
        lat = math.copysign(math.pi / 2.0, z)
        h = abs(z) - a * math.sqrt(1.0 - e2)
        return (math.degrees(lon), math.degrees(lat), float(h))

    # Initial guess

    lat = math.atan2(z, p * (1.0 - e2))
    h = 0.0
    for _ in range(8):
        sin_lat = math.sin(lat)
        N = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
        h = p / max(math.cos(lat), 1e-12) - N
        lat = math.atan2(z, p * (1.0 - e2 * (N / (N + h))))

    return (math.degrees(lon), math.degrees(lat), float(h))


# 関数: `_julian_date_utc` の入出力契約と処理意図を定義する。

def _julian_date_utc(dt: datetime) -> float:
    # UTC as UT1 approximation (sufficient for this stage).
    dt = dt.astimezone(timezone.utc)
    y = dt.year
    m = dt.month
    d = dt.day
    hh = dt.hour
    mm = dt.minute
    ss = dt.second + dt.microsecond / 1e6

    # 条件分岐: `m <= 2` を満たす経路を評価する。
    if m <= 2:
        y -= 1
        m += 12

    A = y // 100
    B = 2 - A + (A // 4)
    day = d + (hh + (mm + ss / 60.0) / 60.0) / 24.0
    JD = int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + day + B - 1524.5
    return float(JD)


# 関数: `gmst_rad` の入出力契約と処理意図を定義する。

def gmst_rad(dt_utc: datetime) -> float:
    # Vallado-style GMST (IAU 1982), good enough for ECEF->ECI z-rotation in this project phase.
    jd = _julian_date_utc(dt_utc)
    T = (jd - 2451545.0) / 36525.0
    gmst_sec = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * T
        + 0.093104 * T * T
        - 6.2e-6 * T * T * T
    )
    gmst_sec = gmst_sec % 86400.0
    return (gmst_sec / 86400.0) * 2.0 * math.pi


# 関数: `ecef_to_eci_zrot` の入出力契約と処理意図を定義する。

def ecef_to_eci_zrot(ecef_m: np.ndarray, times_utc: List[datetime]) -> np.ndarray:
    # ECEF -> ECI using only Earth rotation (GMST). Shape: (n,3)
    x, y, z = float(ecef_m[0]), float(ecef_m[1]), float(ecef_m[2])
    thetas = np.array([gmst_rad(t) for t in times_utc], dtype=float)
    c = np.cos(thetas)
    s = np.sin(thetas)
    x_eci = c * x - s * y
    y_eci = s * x + c * y
    z_eci = np.full_like(x_eci, z)
    return np.stack([x_eci, y_eci, z_eci], axis=1)


# ============================================================
# Moon orientation (IAU 2000-style, simplified) for reflector transform
# ============================================================

def _moon_iau_alpha_delta_w_rad(dt_utc: datetime) -> tuple[float, float, float]:
    """
    IAU 2000-style Moon orientation angles (approx):
      - alpha: right ascension of north pole [rad]
      - delta: declination of north pole [rad]
      - W: prime meridian angle [rad]

    Uses UTC as TT/TDB approximation (adequate for this project phase).
    """
    jd = _julian_date_utc(dt_utc)
    d = jd - 2451545.0
    T = d / 36525.0

    # 関数: `_deg` の入出力契約と処理意図を定義する。
    def _deg(x: float) -> float:
        return math.radians(x)

    # Fundamental arguments (deg), IAU 2000 report style

    E1 = _deg(125.045 - 0.0529921 * d)
    E2 = _deg(250.089 - 0.1059842 * d)
    E3 = _deg(260.008 + 13.0120009 * d)
    E4 = _deg(176.625 + 13.3407154 * d)
    E5 = _deg(357.529 + 0.9856003 * d)
    E6 = _deg(311.589 + 26.4057084 * d)
    E7 = _deg(134.963 + 13.0649930 * d)
    E8 = _deg(276.617 + 0.3287146 * d)
    E9 = _deg(34.226 + 1.7484877 * d)
    E10 = _deg(15.134 - 0.1589763 * d)
    E11 = _deg(119.743 + 0.0036096 * d)
    E12 = _deg(239.961 + 0.1643573 * d)
    E13 = _deg(25.053 + 12.9590088 * d)

    alpha_deg = (
        269.9949
        + 0.0031 * T
        - 3.8787 * math.sin(E1)
        - 0.1204 * math.sin(E2)
        + 0.0700 * math.sin(E3)
        - 0.0172 * math.sin(E4)
        + 0.0072 * math.sin(E6)
        - 0.0052 * math.sin(E10)
        + 0.0043 * math.sin(E13)
    )

    delta_deg = (
        66.5392
        + 0.0130 * T
        + 1.5419 * math.cos(E1)
        + 0.0239 * math.cos(E2)
        - 0.0278 * math.cos(E3)
        + 0.0068 * math.cos(E4)
        - 0.0029 * math.cos(E6)
        + 0.0009 * math.cos(E7)
        + 0.0008 * math.cos(E10)
        - 0.0009 * math.cos(E13)
    )

    W_deg = (
        38.3213
        + 13.17635815 * d
        + 3.5610 * math.sin(E1)
        + 0.1208 * math.sin(E2)
        - 0.0642 * math.sin(E3)
        + 0.0158 * math.sin(E4)
        + 0.0252 * math.sin(E5)
        - 0.0066 * math.sin(E6)
        - 0.0047 * math.sin(E7)
        - 0.0046 * math.sin(E8)
        + 0.0028 * math.sin(E9)
        + 0.0052 * math.sin(E10)
        + 0.0040 * math.sin(E11)
        + 0.0019 * math.sin(E12)
        - 0.0044 * math.sin(E13)
    )

    alpha = math.radians(alpha_deg)
    delta = math.radians(delta_deg)
    W = math.radians(W_deg % 360.0)
    return alpha, delta, W


# 関数: `moon_pa_to_icrf_matrix` の入出力契約と処理意図を定義する。

def moon_pa_to_icrf_matrix(dt_utc: datetime) -> np.ndarray:
    """
    Rotation matrix R (3x3) that maps a vector in Moon body-fixed principal-axis frame
    to ICRF/J2000: v_icrf = R @ v_moon_pa

    This is a simplified IAU model for this repository's "model first" phase.
    """
    alpha, delta, W = _moon_iau_alpha_delta_w_rad(dt_utc)

    ca = math.cos(alpha); sa = math.sin(alpha)
    cd = math.cos(delta); sd = math.sin(delta)

    # Moon north pole unit vector (ICRF)
    z = np.array([cd * ca, cd * sa, sd], dtype=float)

    k = np.array([0.0, 0.0, 1.0], dtype=float)
    n = np.cross(k, z)
    n_norm = float(np.linalg.norm(n))
    # 条件分岐: `n_norm < 1e-12` を満たす経路を評価する。
    if n_norm < 1e-12:
        n = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        n = n / n_norm

    q = np.cross(z, n)  # completes RH basis in Moon equator plane

    # NOTE: W is the prime meridian angle (east-positive). The sign here matters.
    # In this repository, +sin(W) is used so that increasing W rotates the x-axis eastward in the body equator plane.
    cW = math.cos(W); sW = math.sin(W)
    x = cW * n + sW * q
    y = np.cross(z, x)

    R = np.stack([x, y, z], axis=1)  # columns are body axes in ICRF
    return R


# ============================================================
# CRD minimal reader (H2/H3/H4 + 11)
# ============================================================

def _open_text(path: Path) -> io.TextIOBase:
    # 条件分岐: `path.suffix.lower() == ".gz"` を満たす経路を評価する。
    if path.suffix.lower() == ".gz":
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")

    return path.open("r", encoding="utf-8", errors="replace")


# 関数: `_rtype` の入出力契約と処理意図を定義する。

def _rtype(line: str) -> str:
    return line[:2].strip().upper() if line else ""


# 関数: `_is_na` の入出力契約と処理意図を定義する。

def _is_na(tok: str) -> bool:
    return tok.lower() in ("na", "nan")


# 関数: `_to_int` の入出力契約と処理意図を定義する。

def _to_int(tok: str) -> Optional[int]:
    # 条件分岐: `tok is None or _is_na(tok)` を満たす経路を評価する。
    if tok is None or _is_na(tok):
        return None

    try:
        return int(tok)
    except ValueError:
        return None


# 関数: `_to_float` の入出力契約と処理意図を定義する。

def _to_float(tok: str) -> Optional[float]:
    # 条件分岐: `tok is None or _is_na(tok)` を満たす経路を評価する。
    if tok is None or _is_na(tok):
        return None

    try:
        return float(tok)
    except ValueError:
        return None


# クラス: `Context` の責務と境界条件を定義する。

@dataclass
class Context:
    station: Optional[str] = None
    target: Optional[str] = None
    session_day_utc: Optional[datetime] = None  # midnight UTC of session start day
    range_type: Optional[int] = None            # 1=one-way, 2=two-way


# 関数: `parse_crd_npt11` の入出力契約と処理意図を定義する。

def parse_crd_npt11(path: Path, assume_two_way_if_missing: bool = True) -> pd.DataFrame:
    """
    CRDファイルから record "11" (Normal Point) を抽出して DataFrame にする。
    必要最小限の H2/H3/H4 コンテキスト（station/target/session day/range_type）を追跡する。
    """
    ctx = Context()
    rows: List[Dict[str, Any]] = []
    met_rows: List[Dict[str, Any]] = []

    with _open_text(path) as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            # 条件分岐: `not line` を満たす経路を評価する。
            if not line:
                continue

            rt = _rtype(line)
            toks = line.split()

            # 条件分岐: `rt == "H2" and len(toks) >= 2 and not _is_na(toks[1])` を満たす経路を評価する。
            if rt == "H2" and len(toks) >= 2 and not _is_na(toks[1]):
                ctx.station = toks[1]

            # 条件分岐: 前段条件が不成立で、`rt == "H3" and len(toks) >= 2 and not _is_na(toks[1])` を追加評価する。
            elif rt == "H3" and len(toks) >= 2 and not _is_na(toks[1]):
                ctx.target = toks[1]

            # 条件分岐: 前段条件が不成立で、`rt == "H4"` を追加評価する。
            elif rt == "H4":
                # H4 ... start Y M D h m s ... end Y M D h m s ... range_type ...
                def _dt_at(i: int) -> Optional[datetime]:
                    try:
                        y = _to_int(toks[i]); mo = _to_int(toks[i+1]); d = _to_int(toks[i+2])
                        hh = _to_int(toks[i+3]); mm = _to_int(toks[i+4]); ss = _to_int(toks[i+5])
                    except Exception:
                        return None

                    # 条件分岐: `None in (y, mo, d, hh, mm, ss)` を満たす経路を評価する。

                    if None in (y, mo, d, hh, mm, ss):
                        return None

                    return datetime(y, mo, d, hh, mm, ss, tzinfo=timezone.utc)

                start = _dt_at(2)
                # 条件分岐: `start` を満たす経路を評価する。
                if start:
                    ctx.session_day_utc = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)

                rt_val = None
                # v2.01ではrange_typeが token[20] のことが多い
                if len(toks) >= 21:
                    rt_val = _to_int(toks[20])
                # fallback: 末尾から2つ目

                if rt_val is None and len(toks) >= 2:
                    rt_val = _to_int(toks[-2])

                # 条件分岐: `rt_val is None and assume_two_way_if_missing` を満たす経路を評価する。

                if rt_val is None and assume_two_way_if_missing:
                    rt_val = 2

                ctx.range_type = rt_val

            # 条件分岐: 前段条件が不成立で、`rt == "11"` を追加評価する。
            elif rt == "11":
                sod = _to_float(toks[1]) if len(toks) > 1 else None
                tof = _to_float(toks[2]) if len(toks) > 2 else None
                # 条件分岐: `ctx.session_day_utc is not None and sod is not None` を満たす経路を評価する。
                if ctx.session_day_utc is not None and sod is not None:
                    epoch = ctx.session_day_utc + timedelta(seconds=float(sod))
                else:
                    epoch = None

                row = {
                    "file": path.name,
                    "lineno": lineno,
                    "station": ctx.station,
                    "target": ctx.target,
                    "range_type": ctx.range_type,
                    "seconds_of_day": sod,
                    "tof_obs_s": tof,
                    "epoch_utc": epoch,
                    # Meteorological (record 20) if available
                    "pressure_hpa": None,
                    "temp_k": None,
                    "rh_percent": None,
                    "met_source": None,
                }
                rows.append(row)

            # 条件分岐: 前段条件が不成立で、`rt == "20"` を追加評価する。
            elif rt == "20":
                # Meteorological record: "20  SOD  pressure[hPa]  temperature[K]  humidity[%]  source"
                sod20 = _to_float(toks[1]) if len(toks) > 1 else None
                # 条件分岐: `ctx.session_day_utc is None or sod20 is None` を満たす経路を評価する。
                if ctx.session_day_utc is None or sod20 is None:
                    continue

                epoch20 = ctx.session_day_utc + timedelta(seconds=float(sod20))
                met_rows.append(
                    {
                        "file": path.name,
                        "station": ctx.station,
                        "target": ctx.target,
                        "seconds_of_day": sod20,
                        "epoch_utc": epoch20,
                        "pressure_hpa": _to_float(toks[2]) if len(toks) > 2 else None,
                        "temp_k": _to_float(toks[3]) if len(toks) > 3 else None,
                        "rh_percent": _to_float(toks[4]) if len(toks) > 4 else None,
                        "met_source": _to_int(toks[5]) if len(toks) > 5 else None,
                    }
                )

    df = pd.DataFrame(rows)
    # 条件分岐: `df.empty` を満たす経路を評価する。
    if df.empty:
        return df

    df["epoch_utc"] = pd.to_datetime(df["epoch_utc"], utc=True, errors="coerce")

    # Match meteorological record 20 to the nearest NP epoch (within tolerance) per station×target.
    # Many EDC files place record 20 BEFORE record 11, so "11->20 immediate" is not reliable.
    if met_rows:
        try:
            df_met = pd.DataFrame(met_rows)
            df_met["epoch_utc"] = pd.to_datetime(df_met["epoch_utc"], utc=True, errors="coerce")

            df["_st_key"] = df["station"].fillna("").astype(str)
            df["_tgt_key"] = df["target"].fillna("").astype(str)
            df_met["_st_key"] = df_met["station"].fillna("").astype(str)
            df_met["_tgt_key"] = df_met["target"].fillna("").astype(str)

            # NOTE:
            #   pandas.merge_asof requires the *on* key to be globally sorted (not just within each group).
            #   When a file contains multiple station/target sessions, sorting by (_st,_tgt,epoch) breaks
            #   monotonicity of epoch_utc and merge_asof fails with "keys must be sorted".
            #   Keep epoch_utc as the primary sort key (and use station/target only as tie-breakers).
            df = df.dropna(subset=["epoch_utc"]).sort_values(["epoch_utc", "_st_key", "_tgt_key"]).reset_index(drop=True)
            df_met = (
                df_met.dropna(subset=["epoch_utc"])
                .sort_values(["epoch_utc", "_st_key", "_tgt_key"])
                .reset_index(drop=True)
            )

            merged = pd.merge_asof(
                df,
                df_met[
                    [
                        "_st_key",
                        "_tgt_key",
                        "epoch_utc",
                        "pressure_hpa",
                        "temp_k",
                        "rh_percent",
                        "met_source",
                    ]
                ].rename(
                    columns={
                        "pressure_hpa": "pressure_hpa_met",
                        "temp_k": "temp_k_met",
                        "rh_percent": "rh_percent_met",
                        "met_source": "met_source_met",
                    }
                ),
                on="epoch_utc",
                by=["_st_key", "_tgt_key"],
                direction="nearest",
                tolerance=pd.Timedelta(hours=6),
            )
            for a, b in [
                ("pressure_hpa", "pressure_hpa_met"),
                ("temp_k", "temp_k_met"),
                ("rh_percent", "rh_percent_met"),
                ("met_source", "met_source_met"),
            ]:
                # 条件分岐: `b in merged.columns` を満たす経路を評価する。
                if b in merged.columns:
                    # Fill only where missing on the NP side.
                    merged[a] = merged[a].where(merged[a].notna(), merged[b])
                    merged = merged.drop(columns=[b])

            df = merged.drop(columns=[c for c in ("_st_key", "_tgt_key") if c in merged.columns])
        except Exception:
            # Keep parser robust even if meteo merge fails.
            pass

    df = df.dropna(subset=["epoch_utc", "tof_obs_s"]).sort_values("epoch_utc").reset_index(drop=True)
    return df


# ============================================================
# Horizons API helper
# ============================================================

_VEC_EPOCH_LINE_RE = re.compile(
    r"^\s*\d+\.\d+\s*=\s*A\.D\.\s*(?P<cal>\d{4}-[A-Za-z]{3}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)\s+(?P<tz>[A-Za-z~]+)\s*$"
)
_VEC_XYZ_LINE_RE = re.compile(
    r"^\s*X\s*=\s*(?P<x>[-+\d\.E]+)\s+Y\s*=\s*(?P<y>[-+\d\.E]+)\s+Z\s*=\s*(?P<z>[-+\d\.E]+)\s*$"
)


# 関数: `_horizons_get_text` の入出力契約と処理意図を定義する。
def _horizons_get_text(params: Dict[str, str]) -> str:
    q = urllib.parse.urlencode(params, safe="'@,;:+ ")
    url = HORIZONS_API + "?" + q
    ctx = _make_ssl_context()
    # Proxyが必要な場合は OS環境変数 HTTPS_PROXY/HTTP_PROXY を利用（urllibが見に行く）
    max_tries = int(os.environ.get("HORIZONS_RETRY", "").strip() or "5")
    backoff_s = float(os.environ.get("HORIZONS_RETRY_BACKOFF_S", "").strip() or "1.0")
    for attempt in range(1, max_tries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120, context=ctx) as r:
                return r.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:  # type: ignore[attr-defined]
            code = getattr(e, "code", None)
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""

            msg = f"Horizons HTTP {code if code is not None else '??'}: {getattr(e, 'reason', '')}"
            # 条件分岐: `body` を満たす経路を評価する。
            if body:
                msg += "\n" + body[:2000]

            # Transient errors: retry with exponential backoff.

            if code in (429, 502, 503, 504) and attempt < max_tries:
                wait_s = backoff_s * (2 ** (attempt - 1))
                print(f"[warn] Horizons HTTP {code}; retrying in {wait_s:.1f}s ({attempt}/{max_tries})")
                time.sleep(wait_s)
                continue

            raise RuntimeError(msg) from e
        except Exception as e:
            # 条件分岐: `attempt < max_tries` を満たす経路を評価する。
            if attempt < max_tries:
                wait_s = backoff_s * (2 ** (attempt - 1))
                print(f"[warn] Horizons request failed ({type(e).__name__}); retrying in {wait_s:.1f}s ({attempt}/{max_tries})")
                time.sleep(wait_s)
                continue

            raise


# 関数: `_quantize_utc_for_horizons` の入出力契約と処理意図を定義する。

def _quantize_utc_for_horizons(dt: datetime) -> datetime:
    """
    Horizonsのepoch表示は秒の小数4桁（0.0001s=100µs）。それに合わせて入力時刻を丸める。
    """
    dt = dt.astimezone(timezone.utc)
    q = int(HORIZONS_TIME_QUANTUM_US)
    us = dt.microsecond
    us_round = int(round(us / q)) * q
    # 条件分岐: `us_round >= 1_000_000` を満たす経路を評価する。
    if us_round >= 1_000_000:
        dt = dt + timedelta(seconds=1)
        us_round = 0

    return dt.replace(microsecond=us_round)


# 関数: `_format_calendar_utc` の入出力契約と処理意図を定義する。

def _format_calendar_utc(dt: datetime) -> str:
    dt = _quantize_utc_for_horizons(dt)
    # 6桁microsecondを4桁にして渡す（Horizons側の表記に合わせる）
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")[:-2]


# 関数: `_format_tlist` の入出力契約と処理意図を定義する。

def _format_tlist(times_utc: List[datetime]) -> str:
    # Horizons accepts TLIST with quoted calendar strings.
    items = []
    for t in times_utc:
        items.append("'" + _format_calendar_utc(t) + "'")

    return ",".join(items)


# 関数: `fetch_vectors` の入出力契約と処理意図を定義する。

def fetch_vectors(
    command: str,
    center: str,
    times_utc: List[datetime],
    *,
    ref_system: str = "ICRF",
    ref_plane: str = "FRAME",
    coord_type: Optional[str] = None,
    site_coord: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch geometric vectors (X,Y,Z in km) for given target COMMAND and CENTER at discrete times (TLIST).
    Returns DataFrame with columns: epoch_utc, x_km, y_km, z_km
    """
    params = {
        "format": "text",
        "MAKE_EPHEM": "YES",
        "EPHEM_TYPE": "V",
        "COMMAND": f"'{command}'",
        "CENTER": f"'{center}'",
        "VEC_TABLE": "3",
        "REF_SYSTEM": f"'{ref_system}'",
        # IMPORTANT: The default plane is Ecliptic of J2000, which breaks station-vector subtraction.
        # Use 'FRAME' so vectors are in the requested reference frame (ICRF/J2000 equator).
        "REF_PLANE": ref_plane,
        "OUT_UNITS": "'KM-S'",
        "TIME_TYPE": "'UT'",
        "TLIST": _format_tlist(times_utc),
        "OBJ_DATA": "'NO'",
    }
    # 条件分岐: `coord_type` を満たす経路を評価する。
    if coord_type:
        params["COORD_TYPE"] = coord_type if coord_type.startswith("'") else f"'{coord_type}'"

    # 条件分岐: `site_coord` を満たす経路を評価する。

    if site_coord:
        params["SITE_COORD"] = site_coord if site_coord.startswith("'") else f"'{site_coord}'"

    txt = _horizons_get_text(params)

    in_block = False
    cur_epoch: Optional[datetime] = None
    rows = []
    for line in txt.splitlines():
        # 条件分岐: `"$$SOE" in line` を満たす経路を評価する。
        if "$$SOE" in line:
            in_block = True
            cur_epoch = None
            continue

        # 条件分岐: `"$$EOE" in line` を満たす経路を評価する。

        if "$$EOE" in line:
            break

        # 条件分岐: `not in_block` を満たす経路を評価する。

        if not in_block:
            continue

        m_epoch = _VEC_EPOCH_LINE_RE.match(line)
        # 条件分岐: `m_epoch` を満たす経路を評価する。
        if m_epoch:
            cal = m_epoch.group("cal")
            cur_epoch = datetime.strptime(cal, "%Y-%b-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
            continue

        m_xyz = _VEC_XYZ_LINE_RE.match(line)
        # 条件分岐: `m_xyz and cur_epoch is not None` を満たす経路を評価する。
        if m_xyz and cur_epoch is not None:
            rows.append((cur_epoch, float(m_xyz.group("x")), float(m_xyz.group("y")), float(m_xyz.group("z"))))
            cur_epoch = None
            continue

    return pd.DataFrame(rows, columns=["epoch_utc", "x_km", "y_km", "z_km"]).sort_values("epoch_utc")


# 関数: `fetch_vectors_chunked` の入出力契約と処理意図を定義する。

def fetch_vectors_chunked(
    command: str,
    center: str,
    times_utc: List[datetime],
    chunk: int,
    *,
    ref_system: str = "ICRF",
    ref_plane: str = "FRAME",
    coord_type: Optional[str] = None,
    site_coord: Optional[str] = None,
) -> pd.DataFrame:
    dfs = []
    i = 0
    while i < len(times_utc):
        sub = times_utc[i : i + chunk]
        n_sub = int(len(sub))
        # 条件分岐: `n_sub <= 0` を満たす経路を評価する。
        if n_sub <= 0:
            break

        try:
            dfs.append(
                fetch_vectors(
                    command,
                    center,
                    sub,
                    ref_system=ref_system,
                    ref_plane=ref_plane,
                    coord_type=coord_type,
                    site_coord=site_coord,
                )
            )
        except Exception as e:
            # Retry with smaller chunks (helps with transient 5xx or long-query issues).
            if n_sub <= 1:
                raise

            new_chunk = max(1, n_sub // 2)
            print(f"[warn] Horizons chunk failed (n={n_sub}); splitting to chunk={new_chunk}: {e}")
            dfs.append(
                fetch_vectors_chunked(
                    command,
                    center,
                    sub,
                    chunk=new_chunk,
                    ref_system=ref_system,
                    ref_plane=ref_plane,
                    coord_type=coord_type,
                    site_coord=site_coord,
                )
            )
        finally:
            i += n_sub

    return pd.concat(dfs, ignore_index=True).sort_values("epoch_utc").reset_index(drop=True)


# ============================================================
# Horizons cache (offline replay)
# ============================================================

def _times_fingerprint(times_utc: List[datetime]) -> str:
    h = hashlib.sha256()
    for t in times_utc:
        t = _quantize_utc_for_horizons(t)
        h.update(t.isoformat().encode("utf-8"))
        h.update(b"\n")

    return h.hexdigest()[:16]


# 関数: `fetch_vectors_chunked_cached` の入出力契約と処理意図を定義する。

def fetch_vectors_chunked_cached(
    command: str,
    center: str,
    times_utc: List[datetime],
    *,
    chunk: int,
    cache_dir: Path,
    offline: bool,
    ref_system: str = "ICRF",
    ref_plane: str = "FRAME",
    coord_type: Optional[str] = None,
    site_coord: Optional[str] = None,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 関数: `_read_cache_csv` の入出力契約と処理意図を定義する。
    def _read_cache_csv(p: Path) -> pd.DataFrame:
        df = pd.read_csv(p)
        # NOTE: Pandas may fail to parse mixed precision timestamps unless format is specified.
        df["epoch_utc"] = pd.to_datetime(df["epoch_utc"], utc=True, errors="coerce", format="mixed")
        df = df.dropna(subset=["epoch_utc"]).sort_values("epoch_utc").reset_index(drop=True)
        return df

    times_key = _times_fingerprint(times_utc)
    req_key = hashlib.sha256(
        f"cmd={command}|center={center}|ref_system={ref_system}|ref_plane={ref_plane}"
        f"|coord_type={coord_type or ''}|site_coord={site_coord or ''}|times={times_key}".encode("utf-8")
    ).hexdigest()[:16]
    csv_path = cache_dir / f"horizons_vectors_{command}_{req_key}.csv"
    meta_path = cache_dir / f"horizons_vectors_{command}_{req_key}.json"

    # 条件分岐: `csv_path.exists() and meta_path.exists()` を満たす経路を評価する。
    if csv_path.exists() and meta_path.exists():
        print(f"[cache] Using {csv_path}")
        df = _read_cache_csv(csv_path)
        # 条件分岐: `len(df) >= len(times_utc)` を満たす経路を評価する。
        if len(df) >= len(times_utc):
            return df

        msg = f"[cache] Cache {csv_path} has {len(df)} rows, expected {len(times_utc)}; treating as invalid."
        # 条件分岐: `offline` を満たす経路を評価する。
        if offline:
            raise RuntimeError(msg)

        print(msg)

    # 条件分岐: `csv_path.exists() and not meta_path.exists()` を満たす経路を評価する。

    if csv_path.exists() and not meta_path.exists():
        msg = f"[cache] Found {csv_path} but missing {meta_path}; treating as invalid."
        # 条件分岐: `offline` を満たす経路を評価する。
        if offline:
            raise RuntimeError(msg)

        print(msg)

    # 条件分岐: `offline` を満たす経路を評価する。

    if offline:
        # Try to reuse an existing cache that contains a superset of requested times.
        times_req = sorted({_quantize_utc_for_horizons(t) for t in times_utc})

        candidates: List[Tuple[Optional[int], Path]] = []
        for meta in cache_dir.glob(f"horizons_vectors_{command}_*.json"):
            try:
                m = json.loads(meta.read_text(encoding="utf-8"))
            except Exception:
                continue

            # 条件分岐: `str(m.get("command")) != str(command)` を満たす経路を評価する。

            if str(m.get("command")) != str(command):
                continue

            # 条件分岐: `str(m.get("center")) != str(center)` を満たす経路を評価する。

            if str(m.get("center")) != str(center):
                continue

            # 条件分岐: `str(m.get("ref_system", "ICRF")) != str(ref_system)` を満たす経路を評価する。

            if str(m.get("ref_system", "ICRF")) != str(ref_system):
                continue

            # 条件分岐: `str(m.get("ref_plane", "FRAME")) != str(ref_plane)` を満たす経路を評価する。

            if str(m.get("ref_plane", "FRAME")) != str(ref_plane):
                continue

            # 条件分岐: `(m.get("coord_type") or None) != (coord_type or None)` を満たす経路を評価する。

            if (m.get("coord_type") or None) != (coord_type or None):
                continue

            # 条件分岐: `(m.get("site_coord") or None) != (site_coord or None)` を満たす経路を評価する。

            if (m.get("site_coord") or None) != (site_coord or None):
                continue

            try:
                t0 = datetime.fromisoformat(str(m.get("times_min_utc"))).astimezone(timezone.utc)
                t1 = datetime.fromisoformat(str(m.get("times_max_utc"))).astimezone(timezone.utc)
            except Exception:
                continue

            # 条件分岐: `not times_req` を満たす経路を評価する。

            if not times_req:
                continue

            # 条件分岐: `min(times_req) < t0 or max(times_req) > t1` を満たす経路を評価する。

            if min(times_req) < t0 or max(times_req) > t1:
                continue

            n_times = m.get("n_times")
            csv = meta.with_suffix(".csv")
            # 条件分岐: `not csv.exists()` を満たす経路を評価する。
            if not csv.exists():
                continue

            candidates.append((int(n_times) if isinstance(n_times, int) else None, csv))

        # Prefer smaller caches first (faster load), but require full coverage of requested times.

        candidates.sort(key=lambda t: (t[0] is None, t[0] or 0, str(t[1])))
        for n_times, csv in candidates:
            df_all = _read_cache_csv(csv)
            # Quantize to Horizons epoch resolution for stable matching.
            epoch_q = df_all["epoch_utc"].map(lambda x: _quantize_utc_for_horizons(x.to_pydatetime()))
            df_all = df_all.assign(epoch_q=epoch_q)
            lut = {t: i for i, t in enumerate(df_all["epoch_q"].tolist())}
            idx = [lut.get(t) for t in times_req]
            # 条件分岐: `all(i is not None for i in idx)` を満たす経路を評価する。
            if all(i is not None for i in idx):
                df_out = df_all.iloc[idx].drop(columns=["epoch_q"]).reset_index(drop=True)
                extra = f", n_times={n_times}" if n_times is not None else ""
                print(f"[cache] Reusing superset cache {csv} ({len(df_all)} rows{extra}) for {len(times_req)} times")
                return df_out

        raise RuntimeError(f"Offline mode: cache not found: {csv_path}")

    df = fetch_vectors_chunked(
        command,
        center,
        times_utc,
        chunk=chunk,
        ref_system=ref_system,
        ref_plane=ref_plane,
        coord_type=coord_type,
        site_coord=site_coord,
    )
    df2 = df.copy()
    df2["epoch_utc"] = pd.to_datetime(df2["epoch_utc"], utc=True, errors="coerce", format="mixed")
    df2.to_csv(csv_path, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "cache_format": "waveP.horizons_vectors.v1",
                "command": command,
                "center": center,
                "ref_system": ref_system,
                "ref_plane": ref_plane,
                "coord_type": coord_type,
                "site_coord": site_coord,
                "n_times": int(len(times_utc)),
                "times_min_utc": min(times_utc).astimezone(timezone.utc).isoformat() if times_utc else None,
                "times_max_utc": max(times_utc).astimezone(timezone.utc).isoformat() if times_utc else None,
                "times_fingerprint": times_key,
                "chunk": int(chunk),
                "saved_utc": datetime.now(timezone.utc).isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[cache] Saved {csv_path}")
    return df


# ============================================================
# Physics: Shapiro delay (Sun) for endpoints Earth<->Moon
# ============================================================

def shapiro_oneway_sun(mu: float, r1_m: np.ndarray, r2_m: np.ndarray, R12_m: np.ndarray, coeff: float) -> np.ndarray:
    """
    One-way Shapiro delay due to Sun.

    Δt = coeff * mu/c^3 * ln((r1+r2+R12)/(r1+r2-R12))

    - GR: coeff = (1+gamma) = 2
    - P-model mapping used here: coeff = 2*beta  (beta=1 -> 2)
    """
    num = r1_m + r2_m + R12_m
    den = np.maximum(r1_m + r2_m - R12_m, 1e-6)
    return (coeff * mu / (C ** 3)) * np.log(num / den)


# ============================================================
# Auto-pick input file
# ============================================================

def pick_input_file(search_root: Path) -> Path:
    # Prefer a deterministic "primary" file if present (used by run_all/public dashboard).
    for cand in [
        "llr_primary.np2",
        "llr_primary.npt",
        "llr_primary.crd",
        "llr_primary.np2.gz",
        "llr_primary.npt.gz",
        "llr_primary.crd.gz",
    ]:
        p = search_root / cand
        # 条件分岐: `p.exists()` を満たす経路を評価する。
        if p.exists():
            return p

    patterns = [
        "*.crd", "*.npt", "*.np2",
        "*.CRD", "*.NPT", "*.NP2",
        "*.crd.gz", "*.npt.gz", "*.np2.gz",
        "*.CRD.gz", "*.NPT.gz", "*.NP2.gz",
    ]
    found: List[Path] = []
    for pat in patterns:
        found.extend(search_root.rglob(pat) if SEARCH_RECURSIVE else search_root.glob(pat))

    found = sorted(set(found))

    # 条件分岐: `found` を満たす経路を評価する。
    if found:
        return found[0]

    demo = search_root / "demo_llr_like.crd"
    # 条件分岐: `demo.exists()` を満たす経路を評価する。
    if demo.exists():
        return demo

    raise FileNotFoundError(f"No CRD/NPT file found under: {search_root}")


# ============================================================
# Main pipeline
# ============================================================

def run(crd_path: Path, beta: float, outdir: Path, chunk: int) -> Dict[str, Path]:
    outdir.mkdir(parents=True, exist_ok=True)

    df = parse_crd_npt11(crd_path, assume_two_way_if_missing=ASSUME_TWO_WAY_IF_MISSING)
    # 条件分岐: `df.empty` を満たす経路を評価する。
    if df.empty:
        raise RuntimeError(f"No record 11 found in CRD file: {crd_path}")

    repo = Path(__file__).resolve().parents[2]

    # If the file contains multiple stations/targets, pick the most common pair for a clean overlay plot.
    df = df.dropna(subset=["epoch_utc", "tof_obs_s"]).reset_index(drop=True)
    # 条件分岐: `df.empty` を満たす経路を評価する。
    if df.empty:
        raise RuntimeError(f"No usable record 11 rows (missing epoch/tof): {crd_path}")

    n_total = int(len(df))
    # 条件分岐: `"station" in df.columns` を満たす経路を評価する。
    if "station" in df.columns:
        df["station"] = df["station"].astype(str).str.strip().str.upper()

    # 条件分岐: `"target" in df.columns` を満たす経路を評価する。

    if "target" in df.columns:
        df["target"] = df["target"].astype(str).str.strip().str.lower()

    station_code = None
    # 条件分岐: `"station" in df.columns and df["station"].notna().any()` を満たす経路を評価する。
    if "station" in df.columns and df["station"].notna().any():
        try:
            station_code = str(df["station"].value_counts().index[0])
        except Exception:
            station_code = None

    target_name = None
    # 条件分岐: `"target" in df.columns and df["target"].notna().any()` を満たす経路を評価する。
    if "target" in df.columns and df["target"].notna().any():
        try:
            target_name = str(df["target"].value_counts().index[0])
        except Exception:
            target_name = None

    # 条件分岐: `station_code and target_name and ("station" in df.columns) and ("target" in d...` を満たす経路を評価する。

    if station_code and target_name and ("station" in df.columns) and ("target" in df.columns):
        df = df[(df["station"] == station_code) & (df["target"] == target_name)].copy().reset_index(drop=True)
    # 条件分岐: 前段条件が不成立で、`station_code and ("station" in df.columns)` を追加評価する。
    elif station_code and ("station" in df.columns):
        df = df[df["station"] == station_code].copy().reset_index(drop=True)
    # 条件分岐: 前段条件が不成立で、`target_name and ("target" in df.columns)` を追加評価する。
    elif target_name and ("target" in df.columns):
        df = df[df["target"] == target_name].copy().reset_index(drop=True)

    n_used = int(len(df))
    # 条件分岐: `df.empty` を満たす経路を評価する。
    if df.empty:
        raise RuntimeError(f"Selection produced empty dataset (station={station_code}, target={target_name}): {crd_path}")

    # Observation time-tag interpretation (UTC)
    #
    # LLRでは「epoch_utc」が tx/rx/mid のどれを指すかで幾何が変わり、ns級の残差に効く。
    # 既定は "auto" とし、事前に scripts/llr/llr_batch_eval.py で推定した
    # output/private/llr/batch/llr_time_tag_best_by_station.json を参照して局別に決める。
    #
    # 強制する場合は環境変数で指定:
    #   LLR_TIME_TAG=tx|rx|mid|auto

    requested = os.environ.get("LLR_TIME_TAG", "").strip().lower() or "auto"

    # 関数: `_load_time_tag_best_by_station` の入出力契約と処理意図を定義する。
    def _load_time_tag_best_by_station(repo_root: Path) -> Optional[Dict[str, str]]:
        p = repo_root / "output" / "private" / "llr" / "batch" / "llr_time_tag_best_by_station.json"
        # 条件分岐: `not p.exists()` を満たす経路を評価する。
        if not p.exists():
            return None

        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None

        # 条件分岐: `not isinstance(d, dict)` を満たす経路を評価する。

        if not isinstance(d, dict):
            return None
        # Current format (preferred):
        #   {"best_mode_by_station": {"APOL":"tx", ...}, ...}
        # Legacy/compat:
        #   {"APOL":"tx", ...}

        mapping = d.get("best_mode_by_station") if isinstance(d.get("best_mode_by_station"), dict) else d
        # 条件分岐: `not isinstance(mapping, dict)` を満たす経路を評価する。
        if not isinstance(mapping, dict):
            return None

        out: Dict[str, str] = {}
        for k, v in mapping.items():
            ks = str(k).strip().upper()
            vs = str(v).strip().lower()
            # 条件分岐: `ks and vs in ("tx", "rx", "mid")` を満たす経路を評価する。
            if ks and vs in ("tx", "rx", "mid"):
                out[ks] = vs

        return out or None

    # 条件分岐: `requested == "auto"` を満たす経路を評価する。

    if requested == "auto":
        best = _load_time_tag_best_by_station(repo)
        # 条件分岐: `best and station_code and str(station_code).strip().upper() in best` を満たす経路を評価する。
        if best and station_code and str(station_code).strip().upper() in best:
            time_tag_mode = best[str(station_code).strip().upper()]
            print(f"[info] LLR_TIME_TAG=auto → {station_code} は {time_tag_mode} を使用")
        else:
            time_tag_mode = "tx"
            # 条件分岐: `not best` を満たす経路を評価する。
            if not best:
                print("[info] LLR_TIME_TAG=auto だが best_by_station が無いので tx を使用（先に llr_batch_eval.py を実行してください）")
            else:
                print("[info] LLR_TIME_TAG=auto だが station が未解決なので tx を使用")
    # 条件分岐: 前段条件が不成立で、`requested in ("tx", "rx", "mid")` を追加評価する。
    elif requested in ("tx", "rx", "mid"):
        time_tag_mode = requested
    else:
        raise ValueError(f"Invalid LLR_TIME_TAG={requested!r} (expected tx/rx/mid/auto)")

    # Convert to python datetimes (avoid pandas FutureWarning drift)

    tag_times = [t.to_pydatetime() for t in df["epoch_utc"].tolist()]
    obs = df["tof_obs_s"].to_numpy(dtype=float)

    # 関数: `_sec` の入出力契約と処理意図を定義する。
    def _sec(x: float) -> timedelta:
        return timedelta(seconds=float(x))

    tx_times: List[datetime] = []
    bounce_times: List[datetime] = []
    rx_times: List[datetime] = []
    for t_tag, tof_s in zip(tag_times, obs):
        # 条件分岐: `time_tag_mode == "tx"` を満たす経路を評価する。
        if time_tag_mode == "tx":
            t_tx = t_tag
            t_b = t_tag + _sec(tof_s / 2.0)
            t_rx = t_tag + _sec(tof_s)
        # 条件分岐: 前段条件が不成立で、`time_tag_mode == "rx"` を追加評価する。
        elif time_tag_mode == "rx":
            t_rx = t_tag
            t_b = t_tag - _sec(tof_s / 2.0)
            t_tx = t_tag - _sec(tof_s)
        # 条件分岐: 前段条件が不成立で、`time_tag_mode == "mid"` を追加評価する。
        elif time_tag_mode == "mid":
            t_b = t_tag
            t_tx = t_tag - _sec(tof_s / 2.0)
            t_rx = t_tag + _sec(tof_s / 2.0)
        else:
            raise ValueError(f"Invalid LLR_TIME_TAG={time_tag_mode!r} (expected tx/rx/mid)")

        tx_times.append(_quantize_utc_for_horizons(t_tx))
        bounce_times.append(_quantize_utc_for_horizons(t_b))
        rx_times.append(_quantize_utc_for_horizons(t_rx))

    st_meta = _load_station_geodetic(repo, station_code)
    station_available = bool(st_meta and all(k in st_meta for k in ("lat_deg", "lon_deg", "height_m")))
    # 条件分岐: `not station_available` を満たす経路を評価する。
    if not station_available:
        print("[warn] station geodetic not found; using Earth center (run scripts/llr/fetch_station_edc.py).")

    refl_meta = _load_reflector_pa(repo, target_name)
    refl_pa_m = None
    # 条件分岐: `refl_meta and all(k in refl_meta for k in ("x_m", "y_m", "z_m"))` を満たす経路を評価する。
    if refl_meta and all(k in refl_meta for k in ("x_m", "y_m", "z_m")):
        try:
            refl_pa_m = np.array([float(refl_meta["x_m"]), float(refl_meta["y_m"]), float(refl_meta["z_m"])], dtype=float)
        except Exception:
            refl_pa_m = None

    # 条件分岐: `refl_pa_m is None and target_name` を満たす経路を評価する。

    if refl_pa_m is None and target_name:
        print(f"[warn] reflector coords not found for target={target_name} (expected {REFLECTOR_CATALOG_REL}). Using Moon center.")

    offline = os.environ.get("HORIZONS_OFFLINE", "").strip() == "1"
    cache_dir = outdir.parent / "horizons_cache"

    # 関数: `_unique_sorted` の入出力契約と処理意図を定義する。
    def _unique_sorted(times: List[datetime]) -> List[datetime]:
        return sorted({t.astimezone(timezone.utc) for t in times})

    # Ephemerides needed: Earth->Moon(301) and Earth->Sun(10) at tx/bounce/rx

    times_all = _unique_sorted(tx_times + bounce_times + rx_times)
    moon_all = fetch_vectors_chunked_cached("301", "500@399", times_all, chunk=chunk, cache_dir=cache_dir, offline=offline, ref_plane="FRAME")
    sun_all = fetch_vectors_chunked_cached("10", "500@399", times_all, chunk=chunk, cache_dir=cache_dir, offline=offline, ref_plane="FRAME")

    # 関数: `_to_map` の入出力契約と処理意図を定義する。
    def _to_map(vdf: pd.DataFrame) -> Dict[datetime, np.ndarray]:
        out_map: Dict[datetime, np.ndarray] = {}
        for r in vdf.itertuples(index=False):
            t = getattr(r, "epoch_utc")
            # 条件分岐: `isinstance(t, pd.Timestamp)` を満たす経路を評価する。
            if isinstance(t, pd.Timestamp):
                t = t.to_pydatetime()

            out_map[_quantize_utc_for_horizons(t)] = np.array([getattr(r, "x_km"), getattr(r, "y_km"), getattr(r, "z_km")], dtype=float) * 1000.0

        return out_map

    moon_map = _to_map(moon_all)
    sun_map = _to_map(sun_all)

    # Station topocentric (station->moon) at tx/rx
    station_site_coord = None
    station_coord_type = None
    moon_site_map: Dict[datetime, np.ndarray] = {}
    # 条件分岐: `station_available and st_meta is not None` を満たす経路を評価する。
    if station_available and st_meta is not None:
        station_coord_type = "GEODETIC"
        # 条件分岐: `all(st_meta.get(k) is not None for k in ("x_m", "y_m", "z_m"))` を満たす経路を評価する。
        if all(st_meta.get(k) is not None for k in ("x_m", "y_m", "z_m")):
            lon_deg, lat_deg, h_m = geodetic_from_ecef(float(st_meta["x_m"]), float(st_meta["y_m"]), float(st_meta["z_m"]))
            station_site_coord = f"{lon_deg:.10f},{lat_deg:.10f},{h_m/1000.0:.6f}"
        else:
            station_site_coord = f"{float(st_meta['lon_deg']):.10f},{float(st_meta['lat_deg']):.10f},{float(st_meta['height_m'])/1000.0:.6f}"

        times_sm = _unique_sorted(tx_times + rx_times)
        moon_site = fetch_vectors_chunked_cached(
            "301",
            "coord@399",
            times_sm,
            chunk=chunk,
            cache_dir=cache_dir,
            offline=offline,
            ref_plane="FRAME",
            coord_type="GEODETIC",
            site_coord=station_site_coord,
        )
        moon_site_map = _to_map(moon_site)

    # Build per-observation vectors (meters)

    r_em_tx = np.stack([moon_map[t] for t in tx_times], axis=0)
    r_em_rx = np.stack([moon_map[t] for t in rx_times], axis=0)
    r_em_b = np.stack([moon_map[t] for t in bounce_times], axis=0)

    r_es_tx = np.stack([sun_map[t] for t in tx_times], axis=0)
    r_es_rx = np.stack([sun_map[t] for t in rx_times], axis=0)
    r_es_b = np.stack([sun_map[t] for t in bounce_times], axis=0)

    # 条件分岐: `station_available` を満たす経路を評価する。
    if station_available:
        r_sm_tx = np.stack([moon_site_map[t] for t in tx_times], axis=0)
        r_sm_rx = np.stack([moon_site_map[t] for t in rx_times], axis=0)
        r_st_tx = r_em_tx - r_sm_tx  # station position (ICRF/J2000 approx)
        r_st_rx = r_em_rx - r_sm_rx
    else:
        r_sm_tx = np.zeros_like(r_em_tx)
        r_sm_rx = np.zeros_like(r_em_rx)
        r_st_tx = np.zeros_like(r_em_tx)
        r_st_rx = np.zeros_like(r_em_rx)

    # Target at bounce time (Moon center)

    r_moon_b = r_em_b

    # Geometric distances (one-way) for each model
    up_gc = np.linalg.norm(r_moon_b, axis=1)
    down_gc = up_gc.copy()

    up_sm = np.linalg.norm(r_moon_b - r_st_tx, axis=1)
    down_sm = np.linalg.norm(r_moon_b - r_st_rx, axis=1)

    # Reflector at bounce time (Moon PA DE421)
    use_reflector = refl_pa_m is not None
    r_refl_b = None
    up_sr = None
    down_sr = None
    # 条件分岐: `use_reflector and refl_pa_m is not None` を満たす経路を評価する。
    if use_reflector and refl_pa_m is not None:
        mats: List[np.ndarray] = []
        moon_rot_model = None
        for t in bounce_times:
            mat = _moon_pa_de421_to_j2000_matrix(repo, t)
            # 条件分岐: `mat is None` を満たす経路を評価する。
            if mat is None:
                mat = moon_pa_to_icrf_matrix(t)
                moon_rot_model = moon_rot_model or "iau_approx"
            else:
                moon_rot_model = moon_rot_model or "spice:MOON_PA_DE421"

            mats.append(mat)

        rot = np.stack(mats, axis=0)  # (n,3,3)
        refl_icrf = rot @ refl_pa_m  # (n,3)
        r_refl_b = r_em_b + refl_icrf
        up_sr = np.linalg.norm(r_refl_b - r_st_tx, axis=1)
        down_sr = np.linalg.norm(r_refl_b - r_st_rx, axis=1)
    else:
        moon_rot_model = "moon_center_only"

    # Two-way geometric TOF

    tof_geo_gc_s = (up_gc + down_gc) / C
    tof_geo_sm_s = (up_sm + down_sm) / C
    tof_geo_sr_s = (up_sr + down_sr) / C if (up_sr is not None and down_sr is not None) else None

    # Sun Shapiro (two-way) as sum of two one-way legs (P-model coefficient = 2*beta)
    coeff = 2.0 * float(beta)

    r2_gc_m = np.linalg.norm(r_es_b - r_moon_b, axis=1)  # Moon->Sun at bounce
    dt_up_gc = shapiro_oneway_sun(GM_SUN, np.linalg.norm(r_es_tx, axis=1), r2_gc_m, up_gc, coeff=coeff)
    dt_dn_gc = shapiro_oneway_sun(GM_SUN, np.linalg.norm(r_es_rx, axis=1), r2_gc_m, down_gc, coeff=coeff)
    dt_two_gc = dt_up_gc + dt_dn_gc
    tof_gc_raw_s = tof_geo_gc_s + dt_two_gc

    r1_sm_tx = np.linalg.norm(r_es_tx - r_st_tx, axis=1)
    r1_sm_rx = np.linalg.norm(r_es_rx - r_st_rx, axis=1)
    r2_sm_m = np.linalg.norm(r_es_b - r_moon_b, axis=1)
    dt_up_sm = shapiro_oneway_sun(GM_SUN, r1_sm_tx, r2_sm_m, up_sm, coeff=coeff)
    dt_dn_sm = shapiro_oneway_sun(GM_SUN, r1_sm_rx, r2_sm_m, down_sm, coeff=coeff)
    dt_two_sm = dt_up_sm + dt_dn_sm
    tof_sm_raw_s = tof_geo_sm_s + dt_two_sm

    dt_two_sr = None
    tof_sr_raw_s = None
    # 条件分岐: `use_reflector and (r_refl_b is not None) and (tof_geo_sr_s is not None) and (...` を満たす経路を評価する。
    if use_reflector and (r_refl_b is not None) and (tof_geo_sr_s is not None) and (up_sr is not None) and (down_sr is not None):
        r2_sr_m = np.linalg.norm(r_es_b - r_refl_b, axis=1)
        dt_up_sr = shapiro_oneway_sun(GM_SUN, r1_sm_tx, r2_sr_m, up_sr, coeff=coeff)
        dt_dn_sr = shapiro_oneway_sun(GM_SUN, r1_sm_rx, r2_sr_m, down_sr, coeff=coeff)
        dt_two_sr = dt_up_sr + dt_dn_sr
        tof_sr_raw_s = tof_geo_sr_s + dt_two_sr

    # Fit constant offset to align model to observation

    k_gc = float(np.mean(obs - tof_gc_raw_s))
    k_sm = float(np.mean(obs - tof_sm_raw_s))
    tof_gc_s = tof_gc_raw_s + k_gc
    tof_sm_s = tof_sm_raw_s + k_sm

    k_sr = None
    tof_sr_s = None
    # 条件分岐: `tof_sr_raw_s is not None` を満たす経路を評価する。
    if tof_sr_raw_s is not None:
        k_sr = float(np.mean(obs - tof_sr_raw_s))
        tof_sr_s = tof_sr_raw_s + k_sr

    use_reflector = tof_sr_s is not None
    tof_model_s = tof_sr_s if use_reflector else tof_sm_s
    tof_model_raw_s = tof_sr_raw_s if (use_reflector and tof_sr_raw_s is not None) else tof_sm_raw_s
    k_used = k_sr if use_reflector else k_sm
    dt_two_used = dt_two_sr if (use_reflector and dt_two_sr is not None) else dt_two_sm
    R_used_m = ((up_sr + down_sr) / 2.0) if (use_reflector and up_sr is not None and down_sr is not None) else ((up_sm + down_sm) / 2.0)

    nan_vec = np.full(len(obs), np.nan, dtype=float)
    tof_sr_aligned_col = tof_sr_s if tof_sr_s is not None else nan_vec
    tof_sr_raw_col = tof_sr_raw_s if tof_sr_raw_s is not None else nan_vec
    dt_two_sr_col = dt_two_sr if dt_two_sr is not None else nan_vec
    R_sr_col = ((up_sr + down_sr) / 2.0) if (up_sr is not None and down_sr is not None) else nan_vec

    out = pd.DataFrame({
        "epoch_utc": df["epoch_utc"],
        "epoch_tx_utc": tx_times,
        "epoch_bounce_utc": bounce_times,
        "epoch_rx_utc": rx_times,
        "time_tag_mode": time_tag_mode,
        "moon_rotation_model": moon_rot_model,
        "tof_obs_s": obs,
        "tof_model_s": tof_model_s,
        "tof_model_raw_s": tof_model_raw_s,
        "offset_s": k_used,
        "dt_shapiro_two_way_s": dt_two_used,
        "R_station_target_m": R_used_m,
        "target_mode": "reflector" if use_reflector else "moon_center",
        "tof_station_moon_aligned_s": tof_sm_s,
        "tof_station_moon_raw_s": tof_sm_raw_s,
        "offset_station_moon_s": k_sm,
        "dt_shapiro_two_way_station_moon_s": dt_two_sm,
        "R_station_moon_m": (up_sm + down_sm) / 2.0,
        "tof_station_reflector_aligned_s": tof_sr_aligned_col,
        "tof_station_reflector_raw_s": tof_sr_raw_col,
        "offset_station_reflector_s": k_sr,
        "dt_shapiro_two_way_station_reflector_s": dt_two_sr_col,
        "R_station_reflector_m": R_sr_col,
        "R_earth_moon_center_m": up_gc,
        "tof_geocenter_aligned_s": tof_gc_s,
        "tof_geocenter_raw_s": tof_gc_raw_s,
        "offset_geocenter_s": k_gc,
        "dt_shapiro_two_way_geocenter_s": dt_two_gc,
        "station_code": station_code,
        "station_lat_deg": float(st_meta["lat_deg"]) if st_meta else None,
        "station_lon_deg": float(st_meta["lon_deg"]) if st_meta else None,
        "station_height_m": float(st_meta["height_m"]) if st_meta else None,
        "station": df.get("station"),
        "target": df.get("target"),
        "reflector_name": refl_meta.get("name") if isinstance(refl_meta, dict) else None,
        "station_coord_type": station_coord_type,
        "station_site_coord": station_site_coord,
    })

    stem = crd_path.stem.replace(".crd", "").replace(".npt", "")
    table_path = outdir / f"{stem}_table.csv"
    out.to_csv(table_path, index=False)

    # Overlay plot (mean removed, ns)
    t = out["epoch_utc"]
    y_obs = (out["tof_obs_s"] - out["tof_obs_s"].mean()) * 1e9
    y_moon = (out["tof_station_moon_aligned_s"] - out["tof_station_moon_aligned_s"].mean()) * 1e9
    y_refl = (
        (out["tof_station_reflector_aligned_s"] - out["tof_station_reflector_aligned_s"].mean()) * 1e9
        if use_reflector
        else None
    )

    _set_japanese_font()
    plt.figure(figsize=(10, 4.8))
    plt.plot(t, y_obs, marker="o", label="観測 TOF（平均除去）")
    plt.plot(t, y_moon, marker="o", label=f"月中心モデル（局あり, β={beta:g}, 定数オフセット整列, 平均除去）")
    # 条件分岐: `use_reflector and y_refl is not None` を満たす経路を評価する。
    if use_reflector and y_refl is not None:
        plt.plot(t, y_refl, marker="o", label=f"反射器モデル（局+月回転, β={beta:g}, 定数オフセット整列, 平均除去）")

    plt.xlabel("UTC時刻")
    plt.ylabel("往復TOF偏差 [ns]")
    plt.title(f"{LLR_SHORT_NAME}（CRD Normal Point）：観測 vs モデル（太陽Shapiro含む）")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    overlay_path = outdir / f"{stem}_overlay_tof.png"
    plt.savefig(overlay_path, dpi=200)
    plt.close()

    # Residual plot (ns) for the currently selected model (moon-center or reflector)
    res_ns = (out["tof_obs_s"] - out["tof_model_s"]) * 1e9
    model_label = "反射器" if use_reflector else "月中心"
    plt.figure(figsize=(10, 4.5))
    plt.plot(t, res_ns, marker="o")
    plt.axhline(0, linewidth=1)
    plt.xlabel("UTC時刻")
    plt.ylabel("残差（観測 - モデル）[ns]")
    plt.title(f"定数オフセット整列後の残差（{model_label}モデル, 太陽Shapiro含む）")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    res_path = outdir / f"{stem}_residual.png"
    plt.savefig(res_path, dpi=200)
    plt.close()

    # 関数: `_rms_ns` の入出力契約と処理意図を定義する。
    def _rms_ns(series: Any) -> float:
        try:
            arr = np.asarray(series, dtype=float)
        except Exception:
            return float("nan")

        arr = arr[np.isfinite(arr)]
        return float(np.sqrt(np.mean(arr ** 2))) if len(arr) else float("nan")

    # Compare residuals: geocenter vs station->moon vs station->reflector (if available)

    res_gc_ns = (out["tof_obs_s"] - out["tof_geocenter_aligned_s"]) * 1e9
    res_sm_ns = (out["tof_obs_s"] - out["tof_station_moon_aligned_s"]) * 1e9
    rms_gc = _rms_ns(res_gc_ns)
    rms_sm = _rms_ns(res_sm_ns)

    rms_sr = float("nan")
    res_sr_ns = None
    # 条件分岐: `use_reflector` を満たす経路を評価する。
    if use_reflector:
        res_sr_ns = (out["tof_obs_s"] - out["tof_station_reflector_aligned_s"]) * 1e9
        rms_sr = _rms_ns(res_sr_ns)

    plt.figure(figsize=(10, 4.8))
    plt.plot(t, res_gc_ns, marker="o", label=f"地球中心→月中心（RMS={rms_gc:.3g} ns）")
    plt.plot(t, res_sm_ns, marker="o", label=f"観測局→月中心（RMS={rms_sm:.3g} ns）")
    # 条件分岐: `use_reflector and res_sr_ns is not None` を満たす経路を評価する。
    if use_reflector and res_sr_ns is not None:
        plt.plot(t, res_sr_ns, marker="o", label=f"観測局→反射器（RMS={rms_sr:.3g} ns）")

    plt.axhline(0, linewidth=1)
    plt.xlabel("UTC時刻")
    plt.ylabel("残差 [ns]（定数オフセット整列後）")
    plt.title(f"{LLR_SHORT_NAME}：モデル改善（地球中心 → 観測局 → 反射器）")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    cmp_path = outdir / f"{stem}_residual_compare.png"
    plt.savefig(cmp_path, dpi=200)
    plt.close()

    metrics = {
        "input": str(crd_path),
        "n_points_total": n_total,
        "n_points_used": n_used,
        "n_points": n_used,
        "beta": float(beta),
        "time_tag_mode": time_tag_mode,
        "moon_rotation_model": moon_rot_model,
        "station_code": station_code,
        "station_meta": st_meta or None,
        "station_site_coord": station_site_coord,
        "station_coord_type": station_coord_type,
        "target": target_name,
        "reflector_used": bool(use_reflector),
        "reflector_name": refl_meta.get("name") if isinstance(refl_meta, dict) else None,
        "reflector_pa_xyz_m": (
            {"x_m": float(refl_pa_m[0]), "y_m": float(refl_pa_m[1]), "z_m": float(refl_pa_m[2])}
            if refl_pa_m is not None
            else None
        ),
        "rms_residual_geocenter_ns": rms_gc,
        "rms_residual_station_moon_ns": rms_sm,
        "rms_residual_station_reflector_ns": rms_sr,
        # Backward-compatible key (旧: 観測局モデル=月中心)
        "rms_residual_station_ns": rms_sm,
    }
    metrics_path = outdir / f"{stem}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "input": crd_path,
        "table": table_path,
        "overlay": overlay_path,
        "residual": res_path,
        "residual_compare": cmp_path,
        "metrics": metrics_path,
    }


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    data_root = repo / "data" / "llr"
    crd = pick_input_file(data_root)
    outdir = repo / "output" / "private" / "llr" / DEFAULT_OUTDIR

    paths = run(crd, beta=DEFAULT_BETA, outdir=outdir, chunk=DEFAULT_CHUNK)

    print("[ok] input :", paths["input"])
    print("[ok] table :", paths["table"])
    print("[ok] overlay:", paths["overlay"])
    print("[ok] resid :", paths["residual"])


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    main()
