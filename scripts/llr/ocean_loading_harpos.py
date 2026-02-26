#!/usr/bin/env python3
"""
ocean_loading_harpos.py

HARPOS 形式（IMLS: International Mass Loading Service）の
海洋潮汐荷重（TOC: tidal ocean loading）を読み込み、指定時刻(UTC)での
局変位（Up/East/North; meters）を計算するための最小ユーティリティ。

前提:
  - HARPOS の時刻 t は TDT/TT で、基準 epoch は J2000.0（2000-01-01 12:00 TT）
  - 観測の時刻は UTC なので、UTC -> TT 変換が必要（leap seconds + 32.184s）

注意:
  - ここでは軽量な leap seconds テーブルを内蔵する（2017-01-01 まで）。
    LLR解析対象（2012-2025）では十分。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple


J2000_TT = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


# Leap seconds table: (effective UTC, TAI-UTC seconds after that instant)
# Source: IERS leap second history (stable; no changes since 2017-01-01).
_LEAP_TABLE: list[tuple[datetime, int]] = [
    (datetime(1972, 1, 1, 0, 0, 0, tzinfo=timezone.utc), 10),
    (datetime(1972, 7, 1, 0, 0, 0, tzinfo=timezone.utc), 11),
    (datetime(1973, 1, 1, 0, 0, 0, tzinfo=timezone.utc), 12),
    (datetime(1974, 1, 1, 0, 0, 0, tzinfo=timezone.utc), 13),
    (datetime(1975, 1, 1, 0, 0, 0, tzinfo=timezone.utc), 14),
    (datetime(1976, 1, 1, 0, 0, 0, tzinfo=timezone.utc), 15),
    (datetime(1977, 1, 1, 0, 0, 0, tzinfo=timezone.utc), 16),
    (datetime(1978, 1, 1, 0, 0, 0, tzinfo=timezone.utc), 17),
    (datetime(1979, 1, 1, 0, 0, 0, tzinfo=timezone.utc), 18),
    (datetime(1980, 1, 1, 0, 0, 0, tzinfo=timezone.utc), 19),
    (datetime(1981, 7, 1, 0, 0, 0, tzinfo=timezone.utc), 20),
    (datetime(1982, 7, 1, 0, 0, 0, tzinfo=timezone.utc), 21),
    (datetime(1983, 7, 1, 0, 0, 0, tzinfo=timezone.utc), 22),
    (datetime(1985, 7, 1, 0, 0, 0, tzinfo=timezone.utc), 23),
    (datetime(1988, 1, 1, 0, 0, 0, tzinfo=timezone.utc), 24),
    (datetime(1990, 1, 1, 0, 0, 0, tzinfo=timezone.utc), 25),
    (datetime(1991, 1, 1, 0, 0, 0, tzinfo=timezone.utc), 26),
    (datetime(1992, 7, 1, 0, 0, 0, tzinfo=timezone.utc), 27),
    (datetime(1993, 7, 1, 0, 0, 0, tzinfo=timezone.utc), 28),
    (datetime(1994, 7, 1, 0, 0, 0, tzinfo=timezone.utc), 29),
    (datetime(1996, 1, 1, 0, 0, 0, tzinfo=timezone.utc), 30),
    (datetime(1997, 7, 1, 0, 0, 0, tzinfo=timezone.utc), 31),
    (datetime(1999, 1, 1, 0, 0, 0, tzinfo=timezone.utc), 32),
    (datetime(2006, 1, 1, 0, 0, 0, tzinfo=timezone.utc), 33),
    (datetime(2009, 1, 1, 0, 0, 0, tzinfo=timezone.utc), 34),
    (datetime(2012, 7, 1, 0, 0, 0, tzinfo=timezone.utc), 35),
    (datetime(2015, 7, 1, 0, 0, 0, tzinfo=timezone.utc), 36),
    (datetime(2017, 1, 1, 0, 0, 0, tzinfo=timezone.utc), 37),
]


# 関数: `tai_minus_utc_seconds` の入出力契約と処理意図を定義する。
def tai_minus_utc_seconds(dt_utc: datetime) -> int:
    # 条件分岐: `dt_utc.tzinfo is None` を満たす経路を評価する。
    if dt_utc.tzinfo is None:
        raise ValueError("dt_utc must be timezone-aware (UTC)")

    dt = dt_utc.astimezone(timezone.utc)
    # Find the last entry with effective <= dt
    out = _LEAP_TABLE[0][1]
    for eff, val in _LEAP_TABLE:
        # 条件分岐: `dt >= eff` を満たす経路を評価する。
        if dt >= eff:
            out = val
        else:
            break

    return int(out)


# 関数: `utc_to_tt` の入出力契約と処理意図を定義する。

def utc_to_tt(dt_utc: datetime) -> datetime:
    # TT = UTC + (TAI-UTC) + 32.184s
    off = float(tai_minus_utc_seconds(dt_utc)) + 32.184
    return dt_utc.astimezone(timezone.utc) + timedelta(seconds=off)


# 関数: `seconds_since_j2000_tt` の入出力契約と処理意図を定義する。

def seconds_since_j2000_tt(dt_utc: datetime) -> float:
    dt_tt = utc_to_tt(dt_utc)
    return (dt_tt - J2000_TT).total_seconds()


# クラス: `HarmonicDef` の責務と境界条件を定義する。

@dataclass(frozen=True)
class HarmonicDef:
    phase_rad: float
    freq_rad_s: float
    accel_rad_s2: float


# クラス: `SiteDef` の責務と境界条件を定義する。

@dataclass(frozen=True)
class SiteDef:
    x_m: float
    y_m: float
    z_m: float


# クラス: `UENCoef` の責務と境界条件を定義する。

@dataclass(frozen=True)
class UENCoef:
    up_cos_m: float
    east_cos_m: float
    north_cos_m: float
    up_sin_m: float
    east_sin_m: float
    north_sin_m: float


# クラス: `HarposModel` の責務と境界条件を定義する。

@dataclass
class HarposModel:
    path: Path
    validity_radius_m: float
    harmonics: Dict[str, HarmonicDef]
    sites: Dict[str, SiteDef]
    # coeffs[site_id][harmonic] -> UENCoef
    coeffs: Dict[str, Dict[str, UENCoef]]


# 関数: `_f_d` の入出力契約と処理意図を定義する。

def _f_d(s: str) -> float:
    # HARPOS uses Fortran 'D' exponent in some places.
    return float(s.replace("D", "E").replace("d", "E"))


# 関数: `parse_harpos` の入出力契約と処理意図を定義する。

def parse_harpos(path: Path, *, keep_sites: Optional[Iterable[str]] = None) -> HarposModel:
    keep: Optional[set[str]] = None
    # 条件分岐: `keep_sites is not None` を満たす経路を評価する。
    if keep_sites is not None:
        keep = {str(k).strip() for k in keep_sites if str(k).strip()}

    harmonics: Dict[str, HarmonicDef] = {}
    sites: Dict[str, SiteDef] = {}
    coeffs: Dict[str, Dict[str, UENCoef]] = {}
    validity_radius_m = float("nan")

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        # 条件分岐: `not raw` を満たす経路を評価する。
        if not raw:
            continue

        # 条件分岐: `raw.startswith("#")` を満たす経路を評価する。

        if raw.startswith("#"):
            continue

        line = raw.rstrip("\n")
        # 条件分岐: `not line.strip()` を満たす経路を評価する。
        if not line.strip():
            continue

        # Header/trailer (HARPOS ...) or other descriptive lines

        if line.startswith("HARPOS"):
            continue

        rec = line[:1]
        # 条件分岐: `rec == "A"` を満たす経路を評価する。
        if rec == "A":
            # Validity radius line: "A  3000.000000"
            toks = line.split()
            # 条件分岐: `len(toks) >= 2` を満たす経路を評価する。
            if len(toks) >= 2:
                try:
                    validity_radius_m = float(toks[1])
                except Exception:
                    pass

            continue

        # 条件分岐: `rec == "H"` を満たす経路を評価する。

        if rec == "H":
            toks = line.split()
            # 条件分岐: `len(toks) < 5` を満たす経路を評価する。
            if len(toks) < 5:
                continue

            name = toks[1].strip()
            harmonics[name] = HarmonicDef(
                phase_rad=_f_d(toks[2]),
                freq_rad_s=_f_d(toks[3]),
                accel_rad_s2=_f_d(toks[4]),
            )
            continue

        # 条件分岐: `rec == "S"` を満たす経路を評価する。

        if rec == "S":
            toks = line.split()
            # 条件分岐: `len(toks) < 5` を満たす経路を評価する。
            if len(toks) < 5:
                continue

            site = toks[1].strip()
            try:
                sites[site] = SiteDef(x_m=float(toks[2]), y_m=float(toks[3]), z_m=float(toks[4]))
            except Exception:
                continue

            continue

        # 条件分岐: `rec == "D"` を満たす経路を評価する。

        if rec == "D":
            toks = line.split()
            # 条件分岐: `len(toks) < 9` を満たす経路を評価する。
            if len(toks) < 9:
                continue

            harm = toks[1].strip()
            site = toks[2].strip()
            # 条件分岐: `keep is not None and site not in keep` を満たす経路を評価する。
            if keep is not None and site not in keep:
                continue

            # 条件分岐: `site not in coeffs` を満たす経路を評価する。

            if site not in coeffs:
                coeffs[site] = {}

            try:
                coeffs[site][harm] = UENCoef(
                    up_cos_m=float(toks[3]),
                    east_cos_m=float(toks[4]),
                    north_cos_m=float(toks[5]),
                    up_sin_m=float(toks[6]),
                    east_sin_m=float(toks[7]),
                    north_sin_m=float(toks[8]),
                )
            except Exception:
                continue

            continue

        # Ignore unknown record types

    if not math.isfinite(validity_radius_m):
        validity_radius_m = 0.0

    # Ensure sites exist for all kept coeff sites (best-effort).

    for site in list(coeffs.keys()):
        # 条件分岐: `site in sites` を満たす経路を評価する。
        if site in sites:
            continue
        # Not fatal: the caller may still use displacement (UEN) without coordinates.

        sites[site] = SiteDef(x_m=float("nan"), y_m=float("nan"), z_m=float("nan"))

    return HarposModel(path=path, validity_radius_m=float(validity_radius_m), harmonics=harmonics, sites=sites, coeffs=coeffs)


# 関数: `best_site_by_ecef` の入出力契約と処理意図を定義する。

def best_site_by_ecef(model: HarposModel, *, x_m: float, y_m: float, z_m: float) -> Tuple[Optional[str], float]:
    best_site: Optional[str] = None
    best_dist = float("inf")
    for sid, s in model.sites.items():
        # 条件分岐: `not (math.isfinite(s.x_m) and math.isfinite(s.y_m) and math.isfinite(s.z_m))` を満たす経路を評価する。
        if not (math.isfinite(s.x_m) and math.isfinite(s.y_m) and math.isfinite(s.z_m)):
            continue

        dx = float(s.x_m - x_m)
        dy = float(s.y_m - y_m)
        dz = float(s.z_m - z_m)
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        # 条件分岐: `d < best_dist` を満たす経路を評価する。
        if d < best_dist:
            best_dist = d
            best_site = sid

    return best_site, float(best_dist)


# 関数: `displacement_uen_m` の入出力契約と処理意図を定義する。

def displacement_uen_m(model: HarposModel, *, site_id: str, dt_utc: datetime) -> Tuple[float, float, float]:
    """
    Return (Up, East, North) displacement [m] for a given site at dt_utc (UTC).
    """
    sid = str(site_id).strip()
    site_coeffs = model.coeffs.get(sid)
    # 条件分岐: `not site_coeffs` を満たす経路を評価する。
    if not site_coeffs:
        return 0.0, 0.0, 0.0

    t = seconds_since_j2000_tt(dt_utc)
    up = 0.0
    east = 0.0
    north = 0.0

    for harm, c in site_coeffs.items():
        hdef = model.harmonics.get(harm)
        # 条件分岐: `hdef is None` を満たす経路を評価する。
        if hdef is None:
            continue

        theta = hdef.phase_rad + hdef.freq_rad_s * t + 0.5 * hdef.accel_rad_s2 * t * t
        ct = math.cos(theta)
        st = math.sin(theta)
        up += c.up_cos_m * ct + c.up_sin_m * st
        east += c.east_cos_m * ct + c.east_sin_m * st
        north += c.north_cos_m * ct + c.north_sin_m * st

    return float(up), float(east), float(north)
