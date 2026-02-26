#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cosmology_bao_xi_from_catalogs.py

Phase 16（宇宙論）/ Step 16.4（BAO一次統計の再導出）:
実際の銀河 catalog + ランダム catalog から、距離写像（z→D_M）を入れ替えて
ξ(s,μ) → ξℓ（ℓ=0,2）を再計算する。

目的：
- BAO圧縮出力ではなく、観測の一次入力（RA/DEC/z）から距離変換を定義し直して比較する。
- まずは「幾何が合う/合わない」のスクリーニング（Phase A）として pre-recon を確認し、
  次に --recon で reconstruction（Phase B: 簡易Zel'dovich）を入れて post-recon 方向へ接続する。

入力：
- fetch_* スクリプト（例：`fetch_boss_dr12v5_lss.py` / `fetch_eboss_dr16_lss.py`）が作る抽出NPZと `manifest.json`
  （既定：`data/cosmology/boss_dr12v5_lss/manifest.json`、切替：`--data-dir`）

出力（固定名）:
  - output/private/cosmology/cosmology_bao_xi_from_catalogs_{sample}_{caps}_{dist}.png
  - output/private/cosmology/cosmology_bao_xi_from_catalogs_{sample}_{caps}_{dist}_metrics.json
  - output/private/cosmology/cosmology_bao_xi_from_catalogs_{sample}_{caps}_{dist}.npz

依存：
- Corrfunc（Linux/WSL推奨）。Windows単体では Corrfunc が動かないため、
  本スクリプトは Corrfunc が import できない場合は停止する。

座標化仕様（A0で固定する前提：議論のブレ防止）:
- redshift（z）の扱い：catalog列をそのまま使用（既定：Z）。z_CMB 等の別定義は、列が存在する場合のみ選択可能。
- RSD/LOS（line-of-sight）の定義：Corrfunc の定義に従う（pairwise midpoint; l=(v1+v2)/2, mu=cos(s,l)）。
- comoving distance D_M(z) の算出：
  - lcdm: 台形則の累積積分（一様zグリッド）+ 線形補間（np.interp）
  - pbg: 解析式 D_M=(c/100) ln(1+z)
- 距離写像の差し替え点：同一インターフェース（z→D_M）で model を切替える。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402
from scripts.cosmology.mw_recon_code import run_mw_recon  # noqa: E402

_C_KM_S = 299_792.458
_C_OVER_100 = _C_KM_S / 100.0  # (Mpc/h) because H0=100 h km/s/Mpc

# CMB dipole (Solar System barycenter velocity relative to the CMB rest frame).
# We use J2000 equatorial apex (RA/Dec), derived from the commonly cited galactic
# direction (l,b)=(264.021°, 48.253°) via the standard IAU J2000 rotation matrix.
# This is used only when `--z-source cmb` is requested and the catalog does not
# provide an explicit Z_CMB column.
_CMB_DIPOLE_V_KM_S = 369.82
_CMB_DIPOLE_APEX_RA_DEG = 167.9419082184
_CMB_DIPOLE_APEX_DEC_DEG = -6.94426413746

_WIN_ABS_RE = re.compile(r"^[A-Za-z]:[\\/]")

# Supported redshift definitions (column candidates; first match wins).
# Note: current BOSS extracted NPZ contains only "Z". Others are future-proofing.
_Z_SOURCE_TO_COLUMN_CANDIDATES: dict[str, list[str]] = {
    "obs": ["Z", "ZOBS", "Z_OBS"],
    "cosmic": ["ZCOSMO", "Z_COSMO", "Z_COSMIC"],
    "cmb": ["Z_CMB", "ZCMB"],
}


def _set_japanese_font() -> None:
    try:
        # NOTE: This script is often executed under WSL because Corrfunc is not reliably available on Windows.
        # WSL environments typically lack Japanese fonts by default, so we try to register Windows fonts
        # (mounted under /mnt/c) when available.
        import matplotlib as mpl
        import matplotlib.font_manager as fm

        # 条件分岐: `os.name != "nt"` を満たす経路を評価する。
        if os.name != "nt":
            win_font_dir = Path("/mnt/c/Windows/Fonts")
            # 条件分岐: `win_font_dir.exists()` を満たす経路を評価する。
            if win_font_dir.exists():
                # Prefer fonts that are commonly present on Windows installations.
                # (Use addfont so the family name becomes selectable by rcParams.)
                for fname in [
                    "YuGothR.ttc",
                    "YuGothM.ttc",
                    "YuGothB.ttc",
                    "YuGothL.ttc",
                    "meiryo.ttc",
                    "meiryob.ttc",
                    "BIZ-UDGothicR.ttc",
                    "BIZ-UDGothicB.ttc",
                    "msgothic.ttc",
                    "msmincho.ttc",
                ]:
                    fp = win_font_dir / fname
                    # 条件分岐: `fp.exists()` を満たす経路を評価する。
                    if fp.exists():
                        try:
                            fm.fontManager.addfont(str(fp))
                        except Exception:
                            pass

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


def _comoving_distance_lcdm_mpc_over_h(
    z: np.ndarray, *, omega_m: float, n_grid: int = 6000, z_grid_max: float | None = None
) -> np.ndarray:
    """
    Flat LCDM comoving distance D_M(z) in [Mpc/h]:
      D_M = (c/100) * ∫_0^z dz'/E(z'),  E(z)=sqrt(Ωm(1+z)^3 + 1-Ωm)
    """
    z = np.asarray(z, dtype=float)
    # 条件分岐: `np.any(z < 0)` を満たす経路を評価する。
    if np.any(z < 0):
        raise ValueError("z must be >= 0")

    # 条件分岐: `not (0.0 < float(omega_m) < 1.0)` を満たす経路を評価する。

    if not (0.0 < float(omega_m) < 1.0):
        raise ValueError("omega_m must be in (0,1)")

    z_max_in = float(np.max(z)) if z.size else 0.0
    z_max = float(z_grid_max) if (z_grid_max is not None) else z_max_in
    # 条件分岐: `z_max < z_max_in` を満たす経路を評価する。
    if z_max < z_max_in:
        raise ValueError(f"z_grid_max ({z_max:g}) must be >= max(z) ({z_max_in:g})")

    z_grid = np.linspace(0.0, z_max, int(n_grid), dtype=float)
    one_p = 1.0 + z_grid
    e = np.sqrt(float(omega_m) * one_p**3 + (1.0 - float(omega_m)))
    inv_e = 1.0 / e
    dz = np.diff(z_grid)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (inv_e[1:] + inv_e[:-1]) * dz)])
    dm_grid = _C_OVER_100 * cum
    return np.interp(z, z_grid, dm_grid).astype(np.float64, copy=False)


def _comoving_distance_pbg_static_mpc_over_h(z: np.ndarray) -> np.ndarray:
    """
    Static background-P (exponential) distance mapping in [Mpc/h]:
      D_M = (c/100) ln(1+z)
    """
    z = np.asarray(z, dtype=float)
    op = 1.0 + z
    # 条件分岐: `np.any(op <= 0.0)` を満たす経路を評価する。
    if np.any(op <= 0.0):
        raise ValueError("requires 1+z > 0")

    return (_C_OVER_100 * np.log(op)).astype(np.float64, copy=False)


def _z_cmb_dipole_from_obs_z(
    z_obs: np.ndarray, *, ra_deg: np.ndarray, dec_deg: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Approximate heliocentric/barycentric -> CMB rest-frame redshift correction
    using the Solar System barycenter CMB dipole.

    Model:
      (1+z_cmb) = (1+z_obs) * γ * (1 + β cosθ)
    where θ is the angle between the object's direction and the CMB dipole apex.

    Notes:
    - This is intended for *sensitivity checks* of the coordinateization spec.
    - The catalog-provided Z may already include some corrections depending on
      survey pipeline; therefore the absolute meaning should be treated with care.
    """
    z_obs = np.asarray(z_obs, dtype=np.float64)
    ra = np.deg2rad(np.asarray(ra_deg, dtype=np.float64))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))
    # 条件分岐: `ra.shape != z_obs.shape or dec.shape != z_obs.shape` を満たす経路を評価する。
    if ra.shape != z_obs.shape or dec.shape != z_obs.shape:
        raise ValueError("shape mismatch: z_obs, RA, DEC must have same shape")

    beta = float(_CMB_DIPOLE_V_KM_S) / float(_C_KM_S)
    gamma = 1.0 / math.sqrt(max(1e-18, 1.0 - beta * beta))

    ra0 = math.radians(float(_CMB_DIPOLE_APEX_RA_DEG))
    dec0 = math.radians(float(_CMB_DIPOLE_APEX_DEC_DEG))
    dip = np.array([math.cos(dec0) * math.cos(ra0), math.cos(dec0) * math.sin(ra0), math.sin(dec0)], dtype=np.float64)

    cdec = np.cos(dec)
    x = cdec * np.cos(ra)
    y = cdec * np.sin(ra)
    z = np.sin(dec)
    cos_theta = (x * dip[0] + y * dip[1] + z * dip[2]).astype(np.float64, copy=False)

    z_cmb = ((1.0 + z_obs) * float(gamma) * (1.0 + float(beta) * cos_theta) - 1.0).astype(np.float64, copy=False)
    meta = {
        "method": "cmb_dipole",
        "formula": "(1+z_cmb)=(1+z_obs)*gamma*(1+beta*cos(theta))",
        "beta": float(beta),
        "gamma": float(gamma),
        "v_km_s": float(_CMB_DIPOLE_V_KM_S),
        "apex_equatorial_j2000": {"ra_deg": float(_CMB_DIPOLE_APEX_RA_DEG), "dec_deg": float(_CMB_DIPOLE_APEX_DEC_DEG)},
    }
    return z_cmb, meta


def _select_redshift(cols: Dict[str, np.ndarray], *, z_source: str) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Select redshift array from catalog columns based on a fixed "z definition".

    Notes:
    - For BOSS DR12v5 LSS extracted NPZ, only "Z" is currently stored.
    - Other modes (cmb/cosmic) are supported only if the column exists.
    """
    z_source = str(z_source)
    # 条件分岐: `z_source not in _Z_SOURCE_TO_COLUMN_CANDIDATES` を満たす経路を評価する。
    if z_source not in _Z_SOURCE_TO_COLUMN_CANDIDATES:
        raise ValueError(f"invalid z_source: {z_source} (expected one of {sorted(_Z_SOURCE_TO_COLUMN_CANDIDATES)})")

    tried = list(_Z_SOURCE_TO_COLUMN_CANDIDATES[z_source])
    for col in tried:
        # 条件分岐: `col in cols` を満たす経路を評価する。
        if col in cols:
            z = np.asarray(cols[col], dtype=np.float64)
            return z, {"source": z_source, "column": col, "candidates": tried}

    # Fallback: if the catalog does not provide an explicit Z_CMB column,
    # approximate it from observed Z and sky position (RA/DEC).

    if z_source == "cmb":
        # Base (observed) redshift column.
        base_tried = list(_Z_SOURCE_TO_COLUMN_CANDIDATES["obs"])
        base_col = next((c for c in base_tried if c in cols), None)
        # 条件分岐: `base_col is None` を満たす経路を評価する。
        if base_col is None:
            raise KeyError(
                f"z_source=cmb requires one of columns {tried} (preferred) or {base_tried} (fallback), "
                f"but available keys are {sorted(cols.keys())[:20]}..."
            )

        # 条件分岐: `"RA" not in cols or "DEC" not in cols` を満たす経路を評価する。

        if "RA" not in cols or "DEC" not in cols:
            raise KeyError("z_source=cmb fallback requires RA/DEC columns")

        z_obs = np.asarray(cols[base_col], dtype=np.float64)
        z_cmb, meta = _z_cmb_dipole_from_obs_z(z_obs, ra_deg=np.asarray(cols["RA"]), dec_deg=np.asarray(cols["DEC"]))
        return z_cmb, {
            "source": z_source,
            "column": None,
            "candidates": tried,
            "computed": {"from": {"source": "obs", "column": base_col, "candidates": base_tried}, **meta},
        }

    raise KeyError(
        f"z_source={z_source} requires one of columns {tried}, but available keys are {sorted(cols.keys())[:20]}..."
    )


def _comoving_distance_mpc_over_h(
    z: np.ndarray,
    *,
    model: str,
    lcdm_omega_m: float,
    lcdm_n_grid: int,
    lcdm_z_grid_max: float | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Unified distance-mapping interface (z -> D_M) for swapping models.
    """
    model = str(model)
    # 条件分岐: `model == "lcdm"` を満たす経路を評価する。
    if model == "lcdm":
        z_grid_max = float(lcdm_z_grid_max) if (lcdm_z_grid_max is not None) else (float(np.max(z)) if z.size else 0.0)
        dm = _comoving_distance_lcdm_mpc_over_h(
            z,
            omega_m=float(lcdm_omega_m),
            n_grid=int(lcdm_n_grid),
            z_grid_max=z_grid_max,
        )
        meta = {
            "model": "lcdm",
            "dm_definition": "D_M=(c/100)∫0^z dz'/E(z'), E(z)=sqrt(Ωm(1+z)^3+1-Ωm)",
            "integrator": {
                "method": "trapz_cum",
                "z_grid": {"type": "linspace", "n_grid": int(lcdm_n_grid), "z_max": float(z_grid_max)},
                "interp": "linear",
            },
            "params": {"Omega_m": float(lcdm_omega_m)},
        }
        return dm, meta

    # 条件分岐: `model == "pbg"` を満たす経路を評価する。

    if model == "pbg":
        dm = _comoving_distance_pbg_static_mpc_over_h(z)
        meta = {"model": "pbg", "dm_definition": "D_M=(c/100) ln(1+z)", "integrator": {"method": "analytic"}, "params": {}}
        return dm, meta

    raise ValueError(f"invalid distance model: {model}")


def _ones_like_any(cols: Dict[str, np.ndarray]) -> np.ndarray:
    for v in cols.values():
        return np.ones(int(np.asarray(v).shape[0]), dtype=np.float64)

    return np.ones(0, dtype=np.float64)


def _weights_galaxy(cols: Dict[str, np.ndarray], *, scheme: str) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Return per-object weights for the galaxy catalog.

    schemes:
      - boss_default: w = WEIGHT_FKP * WEIGHT_SYSTOT * (WEIGHT_CP + WEIGHT_NOZ - 1)
      - desi_default: w = WEIGHT_FKP * WEIGHT
      - fkp_only:     w = WEIGHT_FKP
      - none:         w = 1
    """
    scheme = str(scheme)
    # 条件分岐: `scheme == "boss_default"` を満たす経路を評価する。
    if scheme == "boss_default":
        w_fkp = np.asarray(cols["WEIGHT_FKP"], dtype=np.float64)
        w_cp = np.asarray(cols["WEIGHT_CP"], dtype=np.float64)
        w_noz = np.asarray(cols["WEIGHT_NOZ"], dtype=np.float64)
        w_sys = np.asarray(cols["WEIGHT_SYSTOT"], dtype=np.float64)
        w = w_fkp * w_sys * (w_cp + w_noz - 1.0)
        meta = {
            "scheme": scheme,
            "definition": "w=WEIGHT_FKP*WEIGHT_SYSTOT*(WEIGHT_CP+WEIGHT_NOZ-1)",
            "columns": ["WEIGHT_FKP", "WEIGHT_SYSTOT", "WEIGHT_CP", "WEIGHT_NOZ"],
        }
        return w, meta

    # 条件分岐: `scheme == "desi_default"` を満たす経路を評価する。

    if scheme == "desi_default":
        w_fkp = np.asarray(cols["WEIGHT_FKP"], dtype=np.float64)
        w_base = np.asarray(cols["WEIGHT"], dtype=np.float64)
        w = w_fkp * w_base
        meta = {
            "scheme": scheme,
            "definition": "w=WEIGHT_FKP*WEIGHT",
            "columns": ["WEIGHT_FKP", "WEIGHT"],
        }
        return w, meta

    # 条件分岐: `scheme == "fkp_only"` を満たす経路を評価する。

    if scheme == "fkp_only":
        w = np.asarray(cols["WEIGHT_FKP"], dtype=np.float64)
        meta = {"scheme": scheme, "definition": "w=WEIGHT_FKP", "columns": ["WEIGHT_FKP"]}
        return w, meta

    # 条件分岐: `scheme == "none"` を満たす経路を評価する。

    if scheme == "none":
        w = _ones_like_any(cols)
        meta = {"scheme": scheme, "definition": "w=1", "columns": []}
        return w, meta

    raise ValueError(f"invalid weight scheme: {scheme} (expected boss_default/desi_default/fkp_only/none)")


def _weights_random(cols: Dict[str, np.ndarray], *, scheme: str) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Return per-object weights for the random catalog.

    schemes:
      - boss_default: w = WEIGHT_FKP
      - desi_default: w = WEIGHT_FKP * WEIGHT
      - fkp_only:     w = WEIGHT_FKP
      - none:         w = 1
    """
    scheme = str(scheme)
    # 条件分岐: `scheme == "desi_default"` を満たす経路を評価する。
    if scheme == "desi_default":
        w_fkp = np.asarray(cols["WEIGHT_FKP"], dtype=np.float64)
        w_base = np.asarray(cols["WEIGHT"], dtype=np.float64)
        w = w_fkp * w_base
        meta = {
            "scheme": scheme,
            "definition": "w=WEIGHT_FKP*WEIGHT",
            "columns": ["WEIGHT_FKP", "WEIGHT"],
        }
        return w, meta

    # 条件分岐: `scheme in ("boss_default", "fkp_only")` を満たす経路を評価する。

    if scheme in ("boss_default", "fkp_only"):
        w = np.asarray(cols["WEIGHT_FKP"], dtype=np.float64)
        meta = {"scheme": scheme, "definition": "w=WEIGHT_FKP", "columns": ["WEIGHT_FKP"]}
        return w, meta

    # 条件分岐: `scheme == "none"` を満たす経路を評価する。

    if scheme == "none":
        w = _ones_like_any(cols)
        meta = {"scheme": scheme, "definition": "w=1", "columns": []}
        return w, meta

    raise ValueError(f"invalid weight scheme: {scheme} (expected boss_default/desi_default/fkp_only/none)")


def _weights_galaxy_recon(
    cols: Dict[str, np.ndarray], *, scheme: str, pair_weight_scheme: str
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Return per-object weights for the *reconstruction density field*.

    Why separate from `--weight-scheme`?
      - `--weight-scheme` is used for the correlation estimator pair counts (DD/DR/RR),
        matching the usual BOSS catalog weighting.
      - Reconstruction can use a different weighting when building the density field
        that sources the displacement.

    schemes:
      - same:      use the same scheme as the pair-counts weight (`pair_weight_scheme`)
      - boss_recon: w = WEIGHT_SYSTOT * (WEIGHT_CP + WEIGHT_NOZ - 1)   (no FKP)
    """
    scheme = str(scheme)
    pair_weight_scheme = str(pair_weight_scheme)
    # 条件分岐: `scheme == "same"` を満たす経路を評価する。
    if scheme == "same":
        w, meta = _weights_galaxy(cols, scheme=pair_weight_scheme)
        return w, {"scheme": scheme, "resolved_scheme": pair_weight_scheme, "resolved": meta}

    # 条件分岐: `scheme == "boss_recon"` を満たす経路を評価する。

    if scheme == "boss_recon":
        w_cp = np.asarray(cols["WEIGHT_CP"], dtype=np.float64)
        w_noz = np.asarray(cols["WEIGHT_NOZ"], dtype=np.float64)
        w_sys = np.asarray(cols["WEIGHT_SYSTOT"], dtype=np.float64)
        w = w_sys * (w_cp + w_noz - 1.0)
        meta = {
            "scheme": scheme,
            "definition": "w=WEIGHT_SYSTOT*(WEIGHT_CP+WEIGHT_NOZ-1)",
            "columns": ["WEIGHT_SYSTOT", "WEIGHT_CP", "WEIGHT_NOZ"],
        }
        return w, meta

    raise ValueError(f"invalid recon weight scheme: {scheme} (expected same/boss_recon)")


def _weights_random_recon(
    cols: Dict[str, np.ndarray], *, scheme: str, pair_weight_scheme: str
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Return per-object weights for the *reconstruction density field* (random catalog).

    schemes:
      - same:      use the same scheme as the pair-counts weight (`pair_weight_scheme`)
      - boss_recon: w = 1
    """
    scheme = str(scheme)
    pair_weight_scheme = str(pair_weight_scheme)
    # 条件分岐: `scheme == "same"` を満たす経路を評価する。
    if scheme == "same":
        w, meta = _weights_random(cols, scheme=pair_weight_scheme)
        return w, {"scheme": scheme, "resolved_scheme": pair_weight_scheme, "resolved": meta}

    # 条件分岐: `scheme == "boss_recon"` を満たす経路を評価する。

    if scheme == "boss_recon":
        w = _ones_like_any(cols)
        meta = {"scheme": scheme, "definition": "w=1", "columns": []}
        return w, meta

    raise ValueError(f"invalid recon weight scheme: {scheme} (expected same/boss_recon)")


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def _sector_keys(cols: Dict[str, np.ndarray], *, sector_key: str) -> np.ndarray | None:
    sector_key = str(sector_key).strip().lower()
    # 条件分岐: `sector_key == "isect"` を満たす経路を評価する。
    if sector_key == "isect":
        # 条件分岐: `"ISECT" not in cols` を満たす経路を評価する。
        if "ISECT" not in cols:
            return None

        isect = np.asarray(cols["ISECT"])
        m = np.isfinite(isect)
        is_i = isect.astype(np.int64, copy=False)
        out = np.full(int(isect.shape[0]), -1, dtype=np.int64)
        out[m] = is_i[m]
        return out

    # 条件分岐: `sector_key == "ipoly_isect"` を満たす経路を評価する。

    if sector_key == "ipoly_isect":
        # 条件分岐: `"IPOLY" not in cols or "ISECT" not in cols` を満たす経路を評価する。
        if "IPOLY" not in cols or "ISECT" not in cols:
            return None

        ip = np.asarray(cols["IPOLY"])
        isect = np.asarray(cols["ISECT"])
        # 条件分岐: `ip.shape != isect.shape` を満たす経路を評価する。
        if ip.shape != isect.shape:
            raise ValueError("IPOLY/ISECT shape mismatch")

        m = np.isfinite(ip) & np.isfinite(isect)
        ip_i = ip.astype(np.int64, copy=False)
        is_i = isect.astype(np.int64, copy=False)
        out = np.full(int(ip.shape[0]), -1, dtype=np.int64)
        # Compose a stable key (assume non-negative IDs; keep low 32-bit for ISECT).
        out[m] = (ip_i[m] << 32) + (is_i[m] & np.int64(0xFFFFFFFF))
        return out

    raise ValueError(f"invalid sector_key: {sector_key} (expected isect/ipoly_isect)")


def _resolve_manifest_path(p: str) -> Path:
    """
    Resolve manifest paths across Windows/WSL.
    Manifest is expected to store repo-relative paths, but legacy absolute sees:
      C:\\...  -> /mnt/c/...
    """
    p = str(p).strip()
    # 条件分岐: `not p` を満たす経路を評価する。
    if not p:
        raise ValueError("empty path in manifest")

    # Legacy: Windows absolute path in a WSL run.

    if os.name != "nt" and _WIN_ABS_RE.match(p):
        drive = p[0].lower()
        rest = p[2:].lstrip("\\/").replace("\\", "/")
        return Path(f"/mnt/{drive}/{rest}")

    path = Path(p)
    # 条件分岐: `path.is_absolute()` を満たす経路を評価する。
    if path.is_absolute():
        return path

    return _ROOT / path


def _growth_rate_lcdm(omega_m0: float, z: float) -> float:
    """
    Approximate linear growth rate f(z) ≈ Ωm(z)^0.55 for flat LCDM.
    Used only as a practical RSD-removal parameter in reconstruction.
    """
    z = float(z)
    op = 1.0 + z
    ez2 = float(omega_m0) * op**3 + (1.0 - float(omega_m0))
    # 条件分岐: `ez2 <= 0` を満たす経路を評価する。
    if ez2 <= 0:
        return float("nan")

    omz = float(omega_m0) * op**3 / ez2
    return float(omz**0.55)


def _radec_to_xyz_mpc_over_h(ra_deg: np.ndarray, dec_deg: np.ndarray, dist_mpc_over_h: np.ndarray) -> np.ndarray:
    ra = np.deg2rad(np.asarray(ra_deg, dtype=np.float64))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=np.float64))
    r = np.asarray(dist_mpc_over_h, dtype=np.float64)
    cosd = np.cos(dec)
    x = r * cosd * np.cos(ra)
    y = r * cosd * np.sin(ra)
    z = r * np.sin(dec)
    return np.stack([x, y, z], axis=1).astype(np.float64, copy=False)


def _xyz_to_radec_mpc_over_h(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xyz = np.asarray(xyz, dtype=np.float64)
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    r = np.sqrt(x * x + y * y + z * z)
    r_safe = np.where(r > 0.0, r, 1.0)
    dec = np.arcsin(np.clip(z / r_safe, -1.0, 1.0))
    ra = np.arctan2(y, x)
    ra_deg = (np.rad2deg(ra) % 360.0).astype(np.float64, copy=False)
    dec_deg = np.rad2deg(dec).astype(np.float64, copy=False)
    return ra_deg, dec_deg, r.astype(np.float64, copy=False)


def _build_recon_box(
    xyz_all: np.ndarray, *, ngrid: int, pad_fraction: float, box_shape: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    xyz_all = np.asarray(xyz_all, dtype=np.float64)
    # 条件分岐: `xyz_all.ndim != 2 or xyz_all.shape[1] != 3` を満たす経路を評価する。
    if xyz_all.ndim != 2 or xyz_all.shape[1] != 3:
        raise ValueError("xyz_all must be (N,3)")

    # 条件分岐: `not (int(ngrid) >= 16)` を満たす経路を評価する。

    if not (int(ngrid) >= 16):
        raise ValueError("--recon-grid must be >= 16")

    # 条件分岐: `not (float(pad_fraction) >= 0.0)` を満たす経路を評価する。

    if not (float(pad_fraction) >= 0.0):
        raise ValueError("--recon-pad-fraction must be >= 0")

    box_shape = str(box_shape)
    # 条件分岐: `box_shape not in ("rect", "cube")` を満たす経路を評価する。
    if box_shape not in ("rect", "cube"):
        raise ValueError("--recon-box-shape must be rect/cube")

    mn = np.min(xyz_all, axis=0)
    mx = np.max(xyz_all, axis=0)
    span = mx - mn
    # Guard against degenerate axes.
    span = np.where(span > 0.0, span, 1.0)
    # 条件分岐: `box_shape == "cube"` を満たす経路を評価する。
    if box_shape == "cube":
        side = float(np.max(span))
        side = side if side > 0.0 else 1.0
        center = 0.5 * (mn + mx)
        pad_s = float(pad_fraction) * side
        origin = (center - 0.5 * side - pad_s).astype(np.float64, copy=False)
        box = np.array([side + 2.0 * pad_s, side + 2.0 * pad_s, side + 2.0 * pad_s], dtype=np.float64)
    else:
        pad = float(pad_fraction) * span
        origin = mn - pad
        box = span + 2.0 * pad

    cell = box / float(int(ngrid))
    meta = {
        "origin_mpc_over_h": origin.tolist(),
        "box_mpc_over_h": box.tolist(),
        "cell_mpc_over_h": cell.tolist(),
        "pad_fraction": float(pad_fraction),
        "ngrid": int(ngrid),
        "box_shape": box_shape,
    }
    return origin, box, cell, meta


def _pca_rotation_matrix(xyz_all: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Principal-axis rotation matrix for the reconstruction box frame.

    Returns a 3x3 orthonormal matrix R such that:
      xyz_rot = xyz @ R

    Notes:
    - The PCA axes are computed from the covariance of centered coordinates, but the
      transform is applied as a pure rotation about the origin (no translation), so
      radial directions and norms are preserved.
    - This is used to reduce AABB elongation for wide sky footprints, improving the
      effective grid resolution along the longest axis without changing physics.
    """
    xyz_all = np.asarray(xyz_all, dtype=np.float64)
    # 条件分岐: `xyz_all.ndim != 2 or xyz_all.shape[1] != 3` を満たす経路を評価する。
    if xyz_all.ndim != 2 or xyz_all.shape[1] != 3:
        raise ValueError("xyz_all must be (N,3)")

    n = int(xyz_all.shape[0])
    # 条件分岐: `n < 3` を満たす経路を評価する。
    if n < 3:
        r = np.eye(3, dtype=np.float64)
        return r, {"name": "pca", "method": "degenerate_n<3", "n": n, "matrix": r.tolist(), "det": 1.0}

    mean = np.mean(xyz_all, axis=0)
    x0 = xyz_all - mean[None, :]
    cov = (x0.T @ x0) / float(n)
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    det = float(np.linalg.det(eigvecs))
    # 条件分岐: `not np.isfinite(det)` を満たす経路を評価する。
    if not np.isfinite(det):
        det = 0.0

    # 条件分岐: `det < 0.0` を満たす経路を評価する。

    if det < 0.0:
        # Enforce right-handed rotation.
        eigvecs[:, 2] *= -1.0
        det = float(np.linalg.det(eigvecs))

    r = np.asarray(eigvecs, dtype=np.float64)
    meta = {
        "name": "pca",
        "method": "cov_eigh",
        "n": n,
        "mean_mpc_over_h": [float(x) for x in mean.tolist()],
        "eigvals": [float(x) for x in eigvals.tolist()],
        "det": float(det),
        "matrix": [[float(x) for x in row] for row in r.tolist()],
    }
    return r, meta


def _histogram3d_weighted(
    xyz: np.ndarray, w: np.ndarray, *, origin: np.ndarray, box: np.ndarray, ngrid: int
) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    # 条件分岐: `xyz.ndim != 2 or xyz.shape[1] != 3` を満たす経路を評価する。
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must be (N,3)")

    # 条件分岐: `w.shape[0] != xyz.shape[0]` を満たす経路を評価する。

    if w.shape[0] != xyz.shape[0]:
        raise ValueError("weights shape mismatch")

    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    box = np.asarray(box, dtype=np.float64).reshape(3)

    # Bin edges per axis.
    edges = [np.linspace(origin[i], origin[i] + box[i], int(ngrid) + 1, dtype=np.float64) for i in range(3)]
    h, _ = np.histogramdd(xyz, bins=edges, weights=w)
    return np.asarray(h, dtype=np.float32)


def _histogram3d_weighted_cic(
    xyz: np.ndarray, w: np.ndarray, *, origin: np.ndarray, cell: np.ndarray, ngrid: int
) -> np.ndarray:
    """
    Cloud-in-Cell (CIC) mass assignment for a regular grid.

    Notes:
    - This is slower than np.histogramdd but reduces shot-noise/aliasing relative to NGP.
    - Boundaries: contributions falling outside [0, ngrid) are dropped (the recon box is padded to include all points).
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    # 条件分岐: `xyz.ndim != 2 or xyz.shape[1] != 3` を満たす経路を評価する。
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must be (N,3)")

    # 条件分岐: `w.shape[0] != xyz.shape[0]` を満たす経路を評価する。

    if w.shape[0] != xyz.shape[0]:
        raise ValueError("weights shape mismatch")

    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    cell = np.asarray(cell, dtype=np.float64).reshape(3)
    # 条件分岐: `np.any(cell <= 0.0)` を満たす経路を評価する。
    if np.any(cell <= 0.0):
        raise ValueError("cell must be > 0")

    n = int(ngrid)
    # 条件分岐: `not (n >= 16)` を満たす経路を評価する。
    if not (n >= 16):
        raise ValueError("ngrid must be >= 16")

    # Treat grid values as cell-centered (i+0.5). Convert to that coordinate system.

    g = (xyz - origin[None, :]) / cell[None, :] - 0.5
    i0 = np.floor(g).astype(np.int64)
    frac = (g - i0).astype(np.float64, copy=False)
    ix0, iy0, iz0 = i0[:, 0], i0[:, 1], i0[:, 2]
    ix1, iy1, iz1 = ix0 + 1, iy0 + 1, iz0 + 1

    wx1 = frac[:, 0]
    wy1 = frac[:, 1]
    wz1 = frac[:, 2]
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1
    wz0 = 1.0 - wz1

    out = np.zeros((n, n, n), dtype=np.float32)

    def _add(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray, ww: np.ndarray) -> None:
        m = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n) & (iz >= 0) & (iz < n) & np.isfinite(ww)
        # 条件分岐: `not np.any(m)` を満たす経路を評価する。
        if not np.any(m):
            return

        np.add.at(out, (ix[m], iy[m], iz[m]), ww[m].astype(np.float32, copy=False))

    base = w.astype(np.float64, copy=False)
    _add(ix0, iy0, iz0, base * wx0 * wy0 * wz0)
    _add(ix0, iy0, iz1, base * wx0 * wy0 * wz1)
    _add(ix0, iy1, iz0, base * wx0 * wy1 * wz0)
    _add(ix0, iy1, iz1, base * wx0 * wy1 * wz1)
    _add(ix1, iy0, iz0, base * wx1 * wy0 * wz0)
    _add(ix1, iy0, iz1, base * wx1 * wy0 * wz1)
    _add(ix1, iy1, iz0, base * wx1 * wy1 * wz0)
    _add(ix1, iy1, iz1, base * wx1 * wy1 * wz1)

    return out


def _recon_displacement_grid_ngp(
    delta: np.ndarray,
    *,
    cell: np.ndarray,
    smoothing_mpc_over_h: float,
    bias: float,
    psi_k_sign: int,
    f_rsd: float | None = None,
    los_unit: np.ndarray | None = None,
    rsd_denom_model: str = "kaiser_beta",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Zel'dovich-like displacement/shift field on a regular grid:
      Ψ(k) = (sign) i k / k^2 * δ_s(k) / b
    where δ_s is Gaussian-smoothed overdensity, b is linear bias, and sign ∈ {+1,-1}.

    Notes on sign:
    - Different BAO reconstruction conventions exist (shift field vs displacement field).
    - Expose the Fourier-sign as a parameter for controlled comparisons against published pipelines.

    Notes:
    - This uses an FFT assuming periodic boundary conditions on the box; for a survey volume this is an approximation.
    - NGP assignment/interpolation is used for simplicity (Phase B baseline).
    """
    delta = np.asarray(delta, dtype=np.float32)
    # 条件分岐: `delta.ndim != 3 or delta.shape[0] != delta.shape[1] or delta.shape[1] != delt...` を満たす経路を評価する。
    if delta.ndim != 3 or delta.shape[0] != delta.shape[1] or delta.shape[1] != delta.shape[2]:
        raise ValueError("delta must be cubic (ngrid,ngrid,ngrid)")

    n = int(delta.shape[0])
    # 条件分岐: `not (n >= 16)` を満たす経路を評価する。
    if not (n >= 16):
        raise ValueError("ngrid must be >= 16")

    cell = np.asarray(cell, dtype=np.float64).reshape(3)
    # 条件分岐: `np.any(cell <= 0.0)` を満たす経路を評価する。
    if np.any(cell <= 0.0):
        raise ValueError("cell must be > 0")

    # 条件分岐: `not (float(smoothing_mpc_over_h) > 0.0)` を満たす経路を評価する。

    if not (float(smoothing_mpc_over_h) > 0.0):
        raise ValueError("--recon-smoothing must be > 0")

    # 条件分岐: `not (float(bias) > 0.0)` を満たす経路を評価する。

    if not (float(bias) > 0.0):
        raise ValueError("--recon-bias must be > 0")

    psi_k_sign = int(psi_k_sign)
    # 条件分岐: `psi_k_sign not in (-1, 1)` を満たす経路を評価する。
    if psi_k_sign not in (-1, 1):
        raise ValueError("psi_k_sign must be -1 or +1")

    # FFT of overdensity field.

    delta_k = np.fft.rfftn(delta)

    twopi = 2.0 * math.pi
    kx = (twopi * np.fft.fftfreq(n, d=float(cell[0]))).astype(np.float64)
    ky = (twopi * np.fft.fftfreq(n, d=float(cell[1]))).astype(np.float64)
    kz = (twopi * np.fft.rfftfreq(n, d=float(cell[2]))).astype(np.float64)

    k2 = (kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2).astype(np.float64, copy=False)
    k2[0, 0, 0] = np.inf
    smooth = np.exp(-0.5 * k2 * float(smoothing_mpc_over_h) ** 2).astype(np.float64, copy=False)

    f_eff = None if (f_rsd is None) else float(f_rsd)
    do_rsd = (
        (f_eff is not None)
        and np.isfinite(float(f_eff))
        and (abs(float(f_eff)) > 0.0)
        and (los_unit is not None)
    )

    # 条件分岐: `do_rsd` を満たす経路を評価する。
    if do_rsd:
        los = np.asarray(los_unit, dtype=np.float64).reshape(3)
        los_norm = float(np.linalg.norm(los))
        # 条件分岐: `not (np.isfinite(los_norm) and (los_norm > 0.0))` を満たす経路を評価する。
        if not (np.isfinite(los_norm) and (los_norm > 0.0)):
            raise ValueError("invalid los_unit for recon")

        los = los / los_norm
        # Global-LOS (plane-parallel) approximation:
        # Two common plane-parallel (global LOS) approximations are supported:
        #
        # 1) "kaiser_beta" (legacy default; Kaiser inversion):
        #    Assume δ_s = (b + f μ^2) δ_m = b(1 + β μ^2) δ_m, β=f/b, then
        #      Ψ(k) = (sign) i k δ_s(k) / (b k^2 (1 + β μ^2))
        #
        # 2) "padmanabhan_pp" (Padmanabhan et al. 2012, plane-parallel limit):
        #    Start from ∇·Ψ + f ∂Ψ_s/∂s = -δ_gal/b, giving
        #      Ψ(k) = (sign) i k δ_gal(k) / (b k^2 (1 + f μ^2))
        #
        # where μ = (k·n)/|k| for a constant LOS unit vector n (plane-parallel).
        kn = (kx[:, None, None] * los[0] + ky[None, :, None] * los[1] + kz[None, None, :] * los[2]).astype(
            np.float64, copy=False
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            mu2 = (kn * kn) / k2

        mu2 = np.where(np.isfinite(mu2), mu2, 0.0)
        rsd_denom_model = str(rsd_denom_model)
        beta: float | None = None
        # 条件分岐: `rsd_denom_model == "kaiser_beta"` を満たす経路を評価する。
        if rsd_denom_model == "kaiser_beta":
            beta = float(f_eff) / float(bias)
            denom_rsd = (1.0 + beta * mu2).astype(np.float64, copy=False)
        # 条件分岐: 前段条件が不成立で、`rsd_denom_model == "padmanabhan_pp"` を追加評価する。
        elif rsd_denom_model == "padmanabhan_pp":
            denom_rsd = (1.0 + float(f_eff) * mu2).astype(np.float64, copy=False)
        else:
            raise ValueError(f"invalid rsd_denom_model: {rsd_denom_model}")
        # (k=0) mode is already nulled by k2=inf; keep denom finite to avoid 0/0.

        denom_rsd[0, 0, 0] = 1.0
        delta_k_sm = (delta_k * smooth) / (float(bias) * k2 * denom_rsd)
    else:
        delta_k_sm = (delta_k * smooth) / (float(bias) * k2)
    # Ψ_i(k) = (sign) i k_i δ(k) / k^2

    axes = (0, 1, 2)
    sign = float(psi_k_sign)
    psi_x = np.fft.irfftn((sign * 1j) * kx[:, None, None] * delta_k_sm, s=delta.shape, axes=axes).astype(
        np.float32, copy=False
    )
    psi_y = np.fft.irfftn((sign * 1j) * ky[None, :, None] * delta_k_sm, s=delta.shape, axes=axes).astype(
        np.float32, copy=False
    )
    psi_z = np.fft.irfftn((sign * 1j) * kz[None, None, :] * delta_k_sm, s=delta.shape, axes=axes).astype(
        np.float32, copy=False
    )

    meta = {
        "method": "zeldovich_fft_ngp",
        "ngrid": int(n),
        "smoothing_mpc_over_h": float(smoothing_mpc_over_h),
        "bias": float(bias),
        "psi_k_sign": int(psi_k_sign),
        "rsd_solver": {
            "enabled": bool(do_rsd),
            "f": float(f_eff) if (f_eff is not None) else None,
            "los_model": "global_constant",
            "denom_model": None,
        },
        "periodic_assumption": True,
        "assignment": "ngp",
        "dtype": "float32",
    }
    # 条件分岐: `do_rsd` を満たす経路を評価する。
    if do_rsd:
        meta["rsd_solver"]["los_unit"] = [float(x) for x in los.tolist()]
        meta["rsd_solver"]["denom_model"] = str(rsd_denom_model)
        # 条件分岐: `rsd_denom_model == "kaiser_beta"` を満たす経路を評価する。
        if rsd_denom_model == "kaiser_beta":
            # Only defined for the Kaiser-inversion branch.
            meta["rsd_solver"]["beta"] = float(beta) if (beta is not None) else None

    return psi_x, psi_y, psi_z, meta


def _sample_grid_ngp(
    psi_x: np.ndarray, psi_y: np.ndarray, psi_z: np.ndarray, *, xyz: np.ndarray, origin: np.ndarray, cell: np.ndarray
) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    cell = np.asarray(cell, dtype=np.float64).reshape(3)
    n = int(psi_x.shape[0])
    # 条件分岐: `xyz.ndim != 2 or xyz.shape[1] != 3` を満たす経路を評価する。
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must be (N,3)")

    idx = np.floor((xyz - origin[None, :]) / cell[None, :]).astype(np.int64)
    idx = np.clip(idx, 0, n - 1)
    ix = idx[:, 0]
    iy = idx[:, 1]
    iz = idx[:, 2]
    dx = np.asarray(psi_x[ix, iy, iz], dtype=np.float64)
    dy = np.asarray(psi_y[ix, iy, iz], dtype=np.float64)
    dz = np.asarray(psi_z[ix, iy, iz], dtype=np.float64)
    return np.stack([dx, dy, dz], axis=1).astype(np.float64, copy=False)


def _sample_grid_cic(
    psi_x: np.ndarray, psi_y: np.ndarray, psi_z: np.ndarray, *, xyz: np.ndarray, origin: np.ndarray, cell: np.ndarray
) -> np.ndarray:
    """
    Trilinear (CIC) interpolation of a vector field defined on a regular grid.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    cell = np.asarray(cell, dtype=np.float64).reshape(3)
    n = int(psi_x.shape[0])
    # 条件分岐: `xyz.ndim != 2 or xyz.shape[1] != 3` を満たす経路を評価する。
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must be (N,3)")

    # 条件分岐: `np.any(cell <= 0.0)` を満たす経路を評価する。

    if np.any(cell <= 0.0):
        raise ValueError("cell must be > 0")

    # Treat grid values as cell-centered (i+0.5). Convert to that coordinate system.

    g = (xyz - origin[None, :]) / cell[None, :] - 0.5
    i0 = np.floor(g).astype(np.int64)
    frac = (g - i0).astype(np.float64, copy=False)
    ix0, iy0, iz0 = i0[:, 0], i0[:, 1], i0[:, 2]
    ix1, iy1, iz1 = ix0 + 1, iy0 + 1, iz0 + 1

    wx1 = frac[:, 0]
    wy1 = frac[:, 1]
    wz1 = frac[:, 2]
    wx0 = 1.0 - wx1
    wy0 = 1.0 - wy1
    wz0 = 1.0 - wz1

    dx = np.zeros(int(xyz.shape[0]), dtype=np.float64)
    dy = np.zeros(int(xyz.shape[0]), dtype=np.float64)
    dz = np.zeros(int(xyz.shape[0]), dtype=np.float64)

    def _acc(ix: np.ndarray, iy: np.ndarray, iz: np.ndarray, ww: np.ndarray) -> None:
        m = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n) & (iz >= 0) & (iz < n) & np.isfinite(ww)
        # 条件分岐: `not np.any(m)` を満たす経路を評価する。
        if not np.any(m):
            return

        wv = ww[m].astype(np.float64, copy=False)
        dx[m] += wv * np.asarray(psi_x[ix[m], iy[m], iz[m]], dtype=np.float64)
        dy[m] += wv * np.asarray(psi_y[ix[m], iy[m], iz[m]], dtype=np.float64)
        dz[m] += wv * np.asarray(psi_z[ix[m], iy[m], iz[m]], dtype=np.float64)

    _acc(ix0, iy0, iz0, wx0 * wy0 * wz0)
    _acc(ix0, iy0, iz1, wx0 * wy0 * wz1)
    _acc(ix0, iy1, iz0, wx0 * wy1 * wz0)
    _acc(ix0, iy1, iz1, wx0 * wy1 * wz1)
    _acc(ix1, iy0, iz0, wx1 * wy0 * wz0)
    _acc(ix1, iy0, iz1, wx1 * wy0 * wz1)
    _acc(ix1, iy1, iz0, wx1 * wy1 * wz0)
    _acc(ix1, iy1, iz1, wx1 * wy1 * wz1)

    return np.stack([dx, dy, dz], axis=1).astype(np.float64, copy=False)


def _apply_reconstruction_grid(
    *,
    ra_g: np.ndarray,
    dec_g: np.ndarray,
    d_g: np.ndarray,
    w_g: np.ndarray,
    z_g: np.ndarray,
    ra_r: np.ndarray,
    dec_r: np.ndarray,
    d_r: np.ndarray,
    w_r: np.ndarray,
    omega_m: float,
    recon_grid: int,
    recon_smoothing: float,
    recon_assignment: str,
    recon_bias: float,
    recon_psi_k_sign: int,
    recon_f: float | None,
    recon_f_source: str,
    recon_mode: str,
    recon_rsd_solver: str,
    recon_los_shift: str,
    recon_rsd_shift: float,
    recon_pad_fraction: float,
    recon_mask_expected_frac: float | None,
    recon_box_shape: str,
    recon_box_frame: str,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, Any]]:
    """
    Apply a simple BAO reconstruction (Zel'dovich on a padded box grid).
    Returns updated (ra,dec,dist) for galaxies and randoms, and reconstruction metadata.
    """
    recon_mode = str(recon_mode)
    # 条件分岐: `recon_mode not in ("ani", "iso")` を満たす経路を評価する。
    if recon_mode not in ("ani", "iso"):
        raise ValueError("--recon-mode must be ani/iso")

    recon_rsd_solver = str(recon_rsd_solver)
    # 条件分岐: `recon_rsd_solver not in ("global_constant", "padmanabhan_pp", "none")` を満たす経路を評価する。
    if recon_rsd_solver not in ("global_constant", "padmanabhan_pp", "none"):
        raise ValueError("--recon-rsd-solver must be global_constant/padmanabhan_pp/none")

    recon_los_shift = str(recon_los_shift)
    # 条件分岐: `recon_los_shift not in ("global_mean_direction", "radial")` を満たす経路を評価する。
    if recon_los_shift not in ("global_mean_direction", "radial"):
        raise ValueError("--recon-los-shift must be global_mean_direction/radial")

    recon_rsd_shift = float(recon_rsd_shift)
    # 条件分岐: `not np.isfinite(recon_rsd_shift)` を満たす経路を評価する。
    if not np.isfinite(recon_rsd_shift):
        raise ValueError("--recon-rsd-shift must be finite")

    recon_assignment = str(recon_assignment)
    # 条件分岐: `recon_assignment not in ("ngp", "cic")` を満たす経路を評価する。
    if recon_assignment not in ("ngp", "cic"):
        raise ValueError("--recon-assignment must be ngp/cic")

    recon_psi_k_sign = int(recon_psi_k_sign)
    # 条件分岐: `recon_psi_k_sign not in (-1, 1)` を満たす経路を評価する。
    if recon_psi_k_sign not in (-1, 1):
        raise ValueError("--recon-psi-k-sign must be -1 or +1")

    recon_box_frame = str(recon_box_frame)
    # 条件分岐: `recon_box_frame not in ("raw", "pca")` を満たす経路を評価する。
    if recon_box_frame not in ("raw", "pca"):
        raise ValueError("--recon-box-frame must be raw/pca")

    # Convert to Cartesian.

    xyz_g0 = _radec_to_xyz_mpc_over_h(ra_g, dec_g, d_g)
    xyz_r0 = _radec_to_xyz_mpc_over_h(ra_r, dec_r, d_r)
    xyz_all0 = np.concatenate([xyz_g0, xyz_r0], axis=0)

    rot: np.ndarray | None = None
    frame_meta: dict[str, Any] = {"name": "raw"}
    # 条件分岐: `recon_box_frame == "pca"` を満たす経路を評価する。
    if recon_box_frame == "pca":
        rot, frame_meta = _pca_rotation_matrix(xyz_all0)
        xyz_g = xyz_g0 @ rot
        xyz_r = xyz_r0 @ rot
        xyz_all = xyz_all0 @ rot
    else:
        xyz_g = xyz_g0
        xyz_r = xyz_r0
        xyz_all = xyz_all0

    origin, box, cell, box_meta = _build_recon_box(
        xyz_all,
        ngrid=int(recon_grid),
        pad_fraction=float(recon_pad_fraction),
        box_shape=str(recon_box_shape),
    )

    # 条件分岐: `recon_assignment == "cic"` を満たす経路を評価する。
    if recon_assignment == "cic":
        gal_grid = _histogram3d_weighted_cic(xyz_g, w_g, origin=origin, cell=cell, ngrid=int(recon_grid))
        rnd_grid = _histogram3d_weighted_cic(xyz_r, w_r, origin=origin, cell=cell, ngrid=int(recon_grid))
    else:
        gal_grid = _histogram3d_weighted(xyz_g, w_g, origin=origin, box=box, ngrid=int(recon_grid))
        rnd_grid = _histogram3d_weighted(xyz_r, w_r, origin=origin, box=box, ngrid=int(recon_grid))

    sum_wg = float(np.sum(w_g))
    sum_wr = float(np.sum(w_r))
    alpha = sum_wg / max(1e-30, sum_wr)
    expected = (alpha * rnd_grid).astype(np.float32, copy=False)

    # Survey mask via the expected (random) density.
    #
    # For CIC assignment, the mask gets "smeared" and many cells become nonzero with tiny expected values,
    # which can explode delta=(n-expected)/expected and yield unphysical displacements.
    # We therefore define an expected floor based on a fraction of the median expected in nonzero cells.
    expected_nonzero = expected[expected > 0.0]
    # 条件分岐: `expected_nonzero.size == 0` を満たす経路を評価する。
    if expected_nonzero.size == 0:
        raise ValueError("recon expected grid is entirely zero (check random catalog / footprint)")

    # 条件分岐: `recon_mask_expected_frac is None` を満たす経路を評価する。

    if recon_mask_expected_frac is None:
        # Empirically, CIC tends to create many "tiny expected" cells. Using a stronger floor (>= median)
        # keeps the effective mask closer to the NGP case (~few % of cells) and avoids delta blow-up.
        recon_mask_expected_frac_eff = 1.5 if (recon_assignment == "cic") else 0.0
    else:
        recon_mask_expected_frac_eff = float(recon_mask_expected_frac)

    # 条件分岐: `not (recon_mask_expected_frac_eff >= 0.0)` を満たす経路を評価する。

    if not (recon_mask_expected_frac_eff >= 0.0):
        raise ValueError("--recon-mask-expected-frac must be >= 0")

    expected_median_nonzero = float(np.median(expected_nonzero))
    expected_floor = float(recon_mask_expected_frac_eff) * expected_median_nonzero

    delta = np.zeros_like(expected, dtype=np.float32)
    m = expected > expected_floor
    delta[m] = (gal_grid[m] - expected[m]) / expected[m]

    # Growth-rate for RSD terms (if enabled).
    z_eff = float(np.sum(np.asarray(z_g, dtype=np.float64) * np.asarray(w_g, dtype=np.float64)) / max(1e-30, sum_wg))
    f_eff = float(recon_f) if (recon_f is not None) else _growth_rate_lcdm(float(omega_m), z_eff)

    los_unit = None
    need_global_los = (recon_mode == "ani") and (
        (recon_rsd_solver in ("global_constant", "padmanabhan_pp")) or (recon_los_shift == "global_mean_direction")
    )
    # 条件分岐: `need_global_los` を満たす経路を評価する。
    if need_global_los:
        # Use a global LOS direction (plane-parallel) as an approximation for wide-angle surveys.
        # This is used either by the optional RSD-aware solver, or by the global-LOS shift term.
        v = np.sum(np.asarray(xyz_g, dtype=np.float64) * np.asarray(w_g, dtype=np.float64)[:, None], axis=0)
        v_norm = float(np.linalg.norm(v))
        # 条件分岐: `not (np.isfinite(v_norm) and (v_norm > 0.0))` を満たす経路を評価する。
        if not (np.isfinite(v_norm) and (v_norm > 0.0)):
            v = np.mean(np.asarray(xyz_g, dtype=np.float64), axis=0)
            v_norm = float(np.linalg.norm(v))

        # 条件分岐: `not (np.isfinite(v_norm) and (v_norm > 0.0))` を満たす経路を評価する。

        if not (np.isfinite(v_norm) and (v_norm > 0.0)):
            v = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            v_norm = 1.0

        los_unit = (v / v_norm).astype(np.float64, copy=False)

    psi_x, psi_y, psi_z, psi_meta = _recon_displacement_grid_ngp(
        delta,
        cell=cell,
        smoothing_mpc_over_h=float(recon_smoothing),
        bias=float(recon_bias),
        psi_k_sign=int(recon_psi_k_sign),
        f_rsd=float(f_eff) if ((recon_mode == "ani") and (recon_rsd_solver in ("global_constant", "padmanabhan_pp"))) else None,
        los_unit=los_unit if ((recon_mode == "ani") and (recon_rsd_solver in ("global_constant", "padmanabhan_pp"))) else None,
        rsd_denom_model=("kaiser_beta" if (recon_rsd_solver == "global_constant") else "padmanabhan_pp"),
    )
    # Align metadata with the actual assignment used for sampling.
    try:
        psi_meta = dict(psi_meta)
        psi_meta["assignment"] = recon_assignment
    except Exception:
        pass

    # 条件分岐: `recon_assignment == "cic"` を満たす経路を評価する。

    if recon_assignment == "cic":
        psi_g = _sample_grid_cic(psi_x, psi_y, psi_z, xyz=xyz_g, origin=origin, cell=cell)
        psi_r = _sample_grid_cic(psi_x, psi_y, psi_z, xyz=xyz_r, origin=origin, cell=cell)
    else:
        psi_g = _sample_grid_ngp(psi_x, psi_y, psi_z, xyz=xyz_g, origin=origin, cell=cell)
        psi_r = _sample_grid_ngp(psi_x, psi_y, psi_z, xyz=xyz_r, origin=origin, cell=cell)

    def _psi_stats(psi: np.ndarray) -> dict[str, Any]:
        mag = np.linalg.norm(np.asarray(psi, dtype=np.float64), axis=1)
        # 条件分岐: `mag.size == 0` を満たす経路を評価する。
        if mag.size == 0:
            return {"n": 0}

        return {
            "n": int(mag.size),
            "mean_mpc_over_h": float(np.mean(mag)),
            "std_mpc_over_h": float(np.std(mag)),
            "p95_mpc_over_h": float(np.percentile(mag, 95.0)),
            "max_mpc_over_h": float(np.max(mag)),
        }

    # RSD correction term (optional): Ψ_s = Ψ + f (Ψ·rhat) rhat.

    if recon_mode == "ani":
        # 条件分岐: `recon_los_shift == "global_mean_direction"` を満たす経路を評価する。
        if recon_los_shift == "global_mean_direction":
            assert los_unit is not None
            proj = np.sum(psi_g * los_unit[None, :], axis=1)
            psi_g_eff = psi_g + float(recon_rsd_shift) * float(f_eff) * proj[:, None] * los_unit[None, :]
        else:
            # Radial LOS per object (more faithful than a single global LOS).
            r = np.sqrt(np.sum(xyz_g * xyz_g, axis=1)).astype(np.float64, copy=False)
            r_safe = np.where(r > 0.0, r, 1.0)
            rhat = (xyz_g / r_safe[:, None]).astype(np.float64, copy=False)
            proj = np.sum(psi_g * rhat, axis=1)
            psi_g_eff = psi_g + float(recon_rsd_shift) * float(f_eff) * proj[:, None] * rhat
    else:
        psi_g_eff = psi_g

    xyz_g_rec_rot = xyz_g - psi_g_eff
    xyz_r_rec_rot = xyz_r - psi_r
    # 条件分岐: `rot is not None` を満たす経路を評価する。
    if rot is not None:
        xyz_g_rec = xyz_g_rec_rot @ rot.T
        xyz_r_rec = xyz_r_rec_rot @ rot.T
    else:
        xyz_g_rec = xyz_g_rec_rot
        xyz_r_rec = xyz_r_rec_rot

    ra_g_rec, dec_g_rec, d_g_rec = _xyz_to_radec_mpc_over_h(xyz_g_rec)
    ra_r_rec, dec_r_rec, d_r_rec = _xyz_to_radec_mpc_over_h(xyz_r_rec)

    meta = {
        "spec": {
            "recon": "grid",
            "mode": recon_mode,
            "rsd_solver": recon_rsd_solver if (recon_mode == "ani") else "none",
            "los_solver_model": ("global_mean_direction" if (need_global_los and recon_rsd_solver == "global_constant") else "none"),
            "los_shift_model": (recon_los_shift if (recon_mode == "ani") else "none"),
            "box_shape": str(recon_box_shape),
            "box_frame": str(recon_box_frame),
            "grid": int(recon_grid),
            "smoothing_mpc_over_h": float(recon_smoothing),
            "assignment": recon_assignment,
            "mask_expected_frac": float(recon_mask_expected_frac_eff),
            "bias": float(recon_bias),
            "psi_k_sign": int(recon_psi_k_sign),
            "f": float(f_eff),
            "rsd_shift": float(recon_rsd_shift) if (recon_mode == "ani") else 0.0,
            "f_source": str(recon_f_source) if (recon_f is not None) else "lcdm_omega_m(z_eff)^0.55",
        },
        "runtime": {
            "box": box_meta,
            "frame": frame_meta,
            "psi_grid": psi_meta,
            "alpha": float(alpha),
            "los_unit": None if (los_unit is None) else [float(x) for x in np.asarray(los_unit, dtype=float).reshape(3).tolist()],
            "z_eff_gal_weighted": float(z_eff),
            "sum_w_gal": float(sum_wg),
            "sum_w_rnd": float(sum_wr),
            "mask": {
                "expected_median_nonzero": expected_median_nonzero,
                "expected_floor": expected_floor,
                "kept_cell_fraction": float(np.mean(m)),
            },
            "psi_sample": {"gal": _psi_stats(psi_g), "rnd": _psi_stats(psi_r)},
        },
        "notes": [
            "簡易reconstruction（Zel'dovich, FFT, periodic box）。survey境界・選択関数は近似。",
            "mask: expected(random) の極小セルを除外（CICでの delta 爆発を抑制）。",
            "randomのprefix_rows抽出を含む場合、reconの系統はPhase Bで全量/均一サンプルに更新。",
        ],
    }
    return (ra_g_rec, dec_g_rec, d_g_rec), (ra_r_rec, dec_r_rec, d_r_rec), meta


def _apply_reconstruction_mw_multigrid(
    *,
    ra_g: np.ndarray,
    dec_g: np.ndarray,
    z_g: np.ndarray,
    d_g: np.ndarray,
    w_g: np.ndarray,
    ra_r: np.ndarray,
    dec_r: np.ndarray,
    z_r: np.ndarray,
    d_r: np.ndarray,
    w_r: np.ndarray,
    dist_model: str,
    omega_m: float,
    recon_smoothing: float,
    recon_bias: float,
    recon_f: float | None,
    recon_f_source: str,
    recon_mode: str,
    mw_random_rsd: bool,
    mw_force_rebuild: bool,
    nthreads: int,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray], dict[str, Any]]:
    """
    External BAO reconstruction backend (Martin White recon_code):
    - radial LOS (g=beta/r^2) + finite-difference solver (multigrid)
    - requires FFTW (auto-built locally under `.tmp_recon_code/_deps`)

    Notes:
    - LCDM: upstream reads (ra,dec,z) and computes distances internally.
    - non-LCDM: we precompute comoving distances D_M(z) and pass chi directly.
    - By default, shifted randoms do NOT include the RSD-enhanced LOS term (Padmanabhan 2012 convention).
      Use `--mw-random-rsd` to enable the alternative behavior.
    """
    recon_mode = str(recon_mode)
    # 条件分岐: `recon_mode not in ("ani", "iso")` を満たす経路を評価する。
    if recon_mode not in ("ani", "iso"):
        raise ValueError("--recon-mode must be ani/iso")

    sum_wg = float(np.sum(np.asarray(w_g, dtype=np.float64)))
    z_eff = float(
        np.sum(np.asarray(z_g, dtype=np.float64) * np.asarray(w_g, dtype=np.float64)) / max(1e-30, sum_wg)
    )

    # 条件分岐: `recon_mode == "iso"` を満たす経路を評価する。
    if recon_mode == "iso":
        f_eff = 0.0
        f_source_eff = "forced_zero_iso"
    else:
        # 条件分岐: `recon_f is None` を満たす経路を評価する。
        if recon_f is None:
            f_eff = float(_growth_rate_lcdm(float(omega_m), float(z_eff)))
            f_source_eff = "lcdm_omega_m(z_eff)^0.55"
        else:
            f_eff = float(recon_f)
            f_source_eff = str(recon_f_source)

    dist_model = str(dist_model)
    input_mode = "z" if (dist_model == "lcdm") else "chi"
    (ra_g2, dec_g2, d_g2), (ra_r2, dec_r2, d_r2), mw_meta = run_mw_recon(
        root=_ROOT,
        ra_g=np.asarray(ra_g, dtype=np.float64),
        dec_g=np.asarray(dec_g, dtype=np.float64),
        z_g=np.asarray(z_g, dtype=np.float64),
        dist_g=None if (input_mode == "z") else np.asarray(d_g, dtype=np.float64),
        w_g_recon=np.asarray(w_g, dtype=np.float64),
        ra_r=np.asarray(ra_r, dtype=np.float64),
        dec_r=np.asarray(dec_r, dtype=np.float64),
        z_r=np.asarray(z_r, dtype=np.float64),
        dist_r=None if (input_mode == "z") else np.asarray(d_r, dtype=np.float64),
        w_r_recon=np.asarray(w_r, dtype=np.float64),
        bias=float(recon_bias),
        f_growth=float(f_eff),
        smoothing_mpc_over_h=float(recon_smoothing),
        omega_m=float(omega_m),
        nthreads=int(nthreads),
        random_rsd=bool(mw_random_rsd),
        force_rebuild=bool(mw_force_rebuild),
        input_mode=str(input_mode),
    )

    # Normalize meta schema to match the in-repo recon metadata conventions.
    meta = {
        "backend": "mw_multigrid",
        "spec": {
            "recon": "mw_multigrid",
            "mode": recon_mode,
            "grid": 512,
            "smoothing_mpc_over_h": float(recon_smoothing),
            "bias": float(recon_bias),
            "f": float(f_eff),
            "f_source": str(f_source_eff),
            "omega_m": float(omega_m),
            "random_rsd": bool(mw_random_rsd),
            "input_mode": str(input_mode),
        },
        "runtime": {
            "z_eff_gal_weighted": float(z_eff),
            "mw": {k: mw_meta.get(k) for k in ("recon_code", "runtime") if k in mw_meta},
        },
        "notes": [
            "外部recon（Martin White recon_code; multigrid + FFT smoothing）。",
            "radial LOS の有限差分ソルバーで Ψ=∇φ を推定し、D/S を生成。",
        ],
    }
    return (ra_g2, dec_g2, d_g2), (ra_r2, dec_r2, d_r2), meta


def _coordinate_spec_signature(coord: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a comparison-friendly coordinate_spec signature.

    When combining NGC/SGC, some runtime details can legitimately differ across caps
    (e.g., reconstruction grid box extents). Those should not cause a hard failure
    as long as the *spec* (z definition / LOS / distance mapping / recon params) matches.
    """
    try:
        out: Dict[str, Any] = json.loads(json.dumps(coord))
    except Exception:
        # Fallback: shallow copy (best effort).
        out = dict(coord)

    rec = out.get("reconstruction", None)
    # 条件分岐: `isinstance(rec, dict)` を満たす経路を評価する。
    if isinstance(rec, dict):
        meta = rec.get("meta", None)
        # 条件分岐: `isinstance(meta, dict)` を満たす経路を評価する。
        if isinstance(meta, dict):
            # Keep only the stable spec (and notes); drop runtime-derived fields (box, alpha, etc).
            meta2: Dict[str, Any] = {}
            # 条件分岐: `"spec" in meta` を満たす経路を評価する。
            if "spec" in meta:
                meta2["spec"] = meta.get("spec")

            # 条件分岐: `"notes" in meta` を満たす経路を評価する。

            if "notes" in meta:
                meta2["notes"] = meta.get("notes")

            rec["meta"] = meta2
            out["reconstruction"] = rec

    return out


def _estimator_spec_signature(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a comparison-friendly estimator_spec signature.

    Estimator specs may include per-cap runtime diagnostics (e.g. random weight
    scaling factors). Those should not affect equality as long as the *policy*
    and binning definitions match.
    """
    try:
        out: Dict[str, Any] = json.loads(json.dumps(spec))
    except Exception:
        out = dict(spec)

    comb = out.get("combine_caps", None)
    # 条件分岐: `isinstance(comb, dict)` を満たす経路を評価する。
    if isinstance(comb, dict):
        # Drop runtime-derived per-cap scaling values; keep only the policy.
        if "random_weight_rescale" in comb and isinstance(comb.get("random_weight_rescale"), dict):
            rwr = dict(comb.get("random_weight_rescale") or {})
            rwr.pop("runtime_by_cap", None)
            comb["random_weight_rescale"] = rwr

        out["combine_caps"] = comb

    return out


def _combine_coordinate_spec_by_cap(cap_packs: list[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Combine per-cap coordinate_spec into a single dict for the combined output.
    Keeps a stable signature and attaches runtime reconstruction metadata per cap (if available).
    """
    # 条件分岐: `not cap_packs` を満たす経路を評価する。
    if not cap_packs:
        return {}

    base_raw = cap_packs[0].get("coordinate_spec", {}) or {}
    base = _coordinate_spec_signature(base_raw)
    rec = base.get("reconstruction", None)
    # 条件分岐: `isinstance(rec, dict) and rec.get("enabled", False)` を満たす経路を評価する。
    if isinstance(rec, dict) and rec.get("enabled", False):
        by_cap: list[dict[str, Any]] = []
        for p in cap_packs:
            cap = str(p.get("cap", ""))
            meta = (p.get("coordinate_spec", {}) or {}).get("reconstruction", {}).get("meta", None)
            runtime = meta.get("runtime") if isinstance(meta, dict) else None
            # 条件分岐: `runtime is not None` を満たす経路を評価する。
            if runtime is not None:
                by_cap.append({"cap": cap, "runtime": runtime})

        meta2 = rec.get("meta", None)
        # 条件分岐: `not isinstance(meta2, dict)` を満たす経路を評価する。
        if not isinstance(meta2, dict):
            meta2 = {}

        meta2["runtime_by_cap"] = by_cap
        rec["meta"] = meta2
        base["reconstruction"] = rec

    return base


def _make_s_bins_file(out_dir: Path, *, s_min: float, s_max: float, s_step: float) -> Tuple[Path, np.ndarray]:
    edges = np.arange(float(s_min), float(s_max) + 0.5 * float(s_step), float(s_step), dtype=float)
    # 条件分岐: `edges.size < 2` を満たす経路を評価する。
    if edges.size < 2:
        raise ValueError("invalid s bins")

    # 条件分岐: `edges[0] <= 0` を満たす経路を評価する。

    if edges[0] <= 0:
        raise ValueError("s_min must be > 0 (Corrfunc requirement)")

    path = out_dir / f"bao_s_bins_{s_min:g}_{s_max:g}_step{s_step:g}.txt"
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        lines = [f"{edges[i]:.6g} {edges[i+1]:.6g}\n" for i in range(edges.size - 1)]
        path.write_text("".join(lines), encoding="utf-8")

    return path, edges


def _corrfunc_paircounts_smu(
    *,
    ra1: np.ndarray,
    dec1: np.ndarray,
    dist1: np.ndarray,
    w1: np.ndarray,
    ra2: np.ndarray | None,
    dec2: np.ndarray | None,
    dist2: np.ndarray | None,
    w2: np.ndarray | None,
    s_bins_file: Path,
    mu_max: float,
    nmu: int,
    nthreads: int,
    autocorr: int,
) -> np.ndarray:
    try:
        from Corrfunc.mocks import DDsmu_mocks
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Corrfunc is required for catalog-based xi computation. "
            "On Windows, run under WSL/Linux with Corrfunc installed."
        ) from e

    # Corrfunc has two relevant APIs in the wild:
    # - pip release: DDsmu_mocks(autocorr, cosmology, nthreads, mu_max, nmu_bins, binfile, RA1, DEC1, CZ1, ...)
    # - DESI branch (cosmodesi/Corrfunc@desi): DDsmu_mocks(autocorr, nthreads, binfile, mumax, nmubins, X1, Y1, Z1, ..., los_type='midpoint')
    #
    # We support both here so the repo remains reproducible across environments.

    import inspect

    params = inspect.signature(DDsmu_mocks).parameters
    uses_sky_api = ("RA1" in params) or ("DEC1" in params) or ("CZ1" in params) or ("cosmology" in params)
    # 条件分岐: `uses_sky_api` を満たす経路を評価する。
    if uses_sky_api:
        res = DDsmu_mocks(
            autocorr,
            1,  # cosmology id (ignored when is_comoving_dist=True)
            nthreads,
            float(mu_max),
            int(nmu),
            str(s_bins_file),
            np.asarray(ra1, dtype=np.float64),
            np.asarray(dec1, dtype=np.float64),
            np.asarray(dist1, dtype=np.float64),
            weights1=np.asarray(w1, dtype=np.float64),
            RA2=None if ra2 is None else np.asarray(ra2, dtype=np.float64),
            DEC2=None if dec2 is None else np.asarray(dec2, dtype=np.float64),
            CZ2=None if dist2 is None else np.asarray(dist2, dtype=np.float64),
            weights2=None if w2 is None else np.asarray(w2, dtype=np.float64),
            is_comoving_dist=True,
            weight_type="pair_product",
            output_savg=False,
        )
        # weighted pair sum per bin
        return (np.asarray(res["npairs"], dtype=np.float64) * np.asarray(res["weightavg"], dtype=np.float64)).astype(np.float64)

    # DESI-branch (cartesian positions; symmetric mu binning). We fold -mu/+mu into |mu| bins
    # so downstream computations keep the historical (0<=mu<=mu_max) interface.

    xyz1 = _radec_to_xyz_mpc_over_h(ra1, dec1, dist1)
    xyz2 = None
    # 条件分岐: `(ra2 is not None) and (dec2 is not None) and (dist2 is not None)` を満たす経路を評価する。
    if (ra2 is not None) and (dec2 is not None) and (dist2 is not None):
        xyz2 = _radec_to_xyz_mpc_over_h(ra2, dec2, dist2)

    nmubins_full = int(2 * int(nmu))
    # 条件分岐: `nmubins_full <= 0` を満たす経路を評価する。
    if nmubins_full <= 0:
        raise ValueError("nmu must be >= 1")

    res = DDsmu_mocks(
        int(autocorr),
        int(nthreads),
        str(s_bins_file),
        float(mu_max),
        int(nmubins_full),
        np.asarray(xyz1[:, 0], dtype=np.float64),
        np.asarray(xyz1[:, 1], dtype=np.float64),
        np.asarray(xyz1[:, 2], dtype=np.float64),
        weights1=np.asarray(w1, dtype=np.float64),
        X2=None if xyz2 is None else np.asarray(xyz2[:, 0], dtype=np.float64),
        Y2=None if xyz2 is None else np.asarray(xyz2[:, 1], dtype=np.float64),
        Z2=None if xyz2 is None else np.asarray(xyz2[:, 2], dtype=np.float64),
        weights2=None if w2 is None else np.asarray(w2, dtype=np.float64),
        weight_type="pair_product",
        bin_type="custom",
        los_type="midpoint",
        output_savg=False,
    )

    wcounts_full = (np.asarray(res["npairs"], dtype=np.float64) * np.asarray(res["weightavg"], dtype=np.float64)).astype(np.float64)
    try:
        wcounts_full = wcounts_full.reshape(-1, nmubins_full)
    except Exception as e:
        raise ValueError(f"unexpected Corrfunc output shape: size={wcounts_full.size}, nmubins_full={nmubins_full}") from e

    half = int(nmubins_full // 2)
    wcounts_pos = wcounts_full[:, half:] + wcounts_full[:, :half][:, ::-1]
    return wcounts_pos.reshape(-1)


def _pycorr_paircounts_smu_ls(
    *,
    ra_g: np.ndarray,
    dec_g: np.ndarray,
    d_g: np.ndarray,
    w_g: np.ndarray,
    ra_r: np.ndarray,
    dec_r: np.ndarray,
    d_r: np.ndarray,
    w_r: np.ndarray,
    edges: np.ndarray,
    mu_max: float,
    nmu: int,
    nthreads: int,
) -> Dict[str, np.ndarray]:
    """
    Paircounts for LS estimator via pycorr (Corrfunc engine).

    pycorr's Corrfunc backend requires symmetric, linearly-spaced μ edges.
    We fold μ∈[-mu_max,mu_max] into |μ|∈[0,mu_max] bins so downstream code can keep
    the historical interface consistent with our Corrfunc direct path.
    """
    try:
        from pycorr import TwoPointCorrelationFunction
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "pycorr is required for --paircounts-backend pycorr. "
            "Install it in the WSL venv (.venv_wsl)."
        ) from e

    # 条件分岐: `int(nmu) <= 0` を満たす経路を評価する。

    if int(nmu) <= 0:
        raise ValueError("nmu must be >= 1")

    mu_edges = np.linspace(-float(mu_max), float(mu_max), int(2 * int(nmu)) + 1, dtype=float)

    pos_g = _radec_to_xyz_mpc_over_h(ra_g, dec_g, d_g)
    pos_r = _radec_to_xyz_mpc_over_h(ra_r, dec_r, d_r)

    cf = TwoPointCorrelationFunction(
        "smu",
        (np.asarray(edges, dtype=float), mu_edges),
        data_positions1=np.asarray(pos_g, dtype=np.float64),
        randoms_positions1=np.asarray(pos_r, dtype=np.float64),
        data_weights1=np.asarray(w_g, dtype=np.float64),
        randoms_weights1=np.asarray(w_r, dtype=np.float64),
        position_type="pos",
        engine="corrfunc",
        nthreads=int(nthreads),
        compute_sepsavg=False,
        los="midpoint",
    )

    dd_full = np.asarray(cf.D1D2.wcounts, dtype=np.float64)
    dr_full = np.asarray(cf.D1R2.wcounts, dtype=np.float64)
    rr_full = np.asarray(cf.R1R2.wcounts, dtype=np.float64)

    # 条件分岐: `dd_full.shape != dr_full.shape or dd_full.shape != rr_full.shape` を満たす経路を評価する。
    if dd_full.shape != dr_full.shape or dd_full.shape != rr_full.shape:
        raise ValueError(f"pycorr unexpected count shapes: DD={dd_full.shape}, DR={dr_full.shape}, RR={rr_full.shape}")

    # 条件分岐: `dd_full.ndim != 2` を満たす経路を評価する。

    if dd_full.ndim != 2:
        raise ValueError(f"pycorr expected 2D counts (s,mu), got shape={dd_full.shape}")

    # 条件分岐: `dd_full.shape[1] != int(2 * int(nmu))` を満たす経路を評価する。

    if dd_full.shape[1] != int(2 * int(nmu)):
        raise ValueError(f"pycorr mu-bin mismatch: got={dd_full.shape[1]}, expected={2*int(nmu)}")

    half = int(dd_full.shape[1] // 2)
    dd_pos = dd_full[:, half:] + dd_full[:, :half][:, ::-1]
    dr_pos = dr_full[:, half:] + dr_full[:, :half][:, ::-1]
    rr_pos = rr_full[:, half:] + rr_full[:, :half][:, ::-1]

    return {"DD_w": dd_pos.reshape(-1), "DR_w": dr_pos.reshape(-1), "RR_w": rr_pos.reshape(-1)}


def _xi_multipoles_from_catalogs(
    *,
    gal: Dict[str, np.ndarray],
    rnd: Dict[str, np.ndarray],
    weight_scheme: str,
    recon_weight_scheme: str,
    dist_model: str,
    omega_m: float,
    z_source: str,
    los: str,
    lcdm_n_grid: int,
    lcdm_z_grid_max: float | None,
    z_min: float | None,
    z_max: float | None,
    recon_z_min: float | None,
    recon_z_max: float | None,
    s_bins_file: Path,
    edges: np.ndarray,
    mu_max: float,
    nmu: int,
    nthreads: int,
    paircounts_backend: str,
    match_sectors: bool,
    sector_key: str,
    recon: str,
    recon_grid: int,
    recon_smoothing: float,
    recon_assignment: str,
    recon_bias: float,
    recon_psi_k_sign: int,
    recon_f: float | None,
    recon_f_source: str,
    recon_mode: str,
    recon_rsd_solver: str,
    recon_los_shift: str,
    recon_rsd_shift: float,
    recon_pad_fraction: float,
    recon_mask_expected_frac: float | None,
    recon_box_shape: str,
    recon_box_frame: str,
    mw_random_rsd: bool,
    mw_force_rebuild: bool,
) -> Dict[str, Any]:
    los = str(los)
    # 条件分岐: `los != "midpoint"` を満たす経路を評価する。
    if los != "midpoint":
        raise ValueError("only --los midpoint is supported (Corrfunc uses pairwise-midpoint LOS definition)")

    recon = str(recon)
    paircounts_backend = str(paircounts_backend)
    # 条件分岐: `paircounts_backend not in ("corrfunc", "pycorr")` を満たす経路を評価する。
    if paircounts_backend not in ("corrfunc", "pycorr"):
        raise ValueError("--paircounts-backend must be corrfunc/pycorr")

    # 条件分岐: `paircounts_backend == "pycorr" and recon != "none"` を満たす経路を評価する。

    if paircounts_backend == "pycorr" and recon != "none":
        raise ValueError("--paircounts-backend pycorr currently supports --recon none only")

    ra_g = np.asarray(gal["RA"], dtype=np.float64)
    dec_g = np.asarray(gal["DEC"], dtype=np.float64)
    z_g, z_meta_g = _select_redshift(gal, z_source=str(z_source))
    w_g, w_meta_g = _weights_galaxy(gal, scheme=str(weight_scheme))
    w_g_recon_full: np.ndarray | None = None
    w_meta_g_recon: dict[str, Any] | None = None
    # 条件分岐: `recon != "none"` を満たす経路を評価する。
    if recon != "none":
        w_g_recon_full, w_meta_g_recon = _weights_galaxy_recon(
            gal, scheme=str(recon_weight_scheme), pair_weight_scheme=str(weight_scheme)
        )

    ra_r = np.asarray(rnd["RA"], dtype=np.float64)
    dec_r = np.asarray(rnd["DEC"], dtype=np.float64)
    z_r, z_meta_r = _select_redshift(rnd, z_source=str(z_source))
    w_r, w_meta_r = _weights_random(rnd, scheme=str(weight_scheme))
    w_r_recon_full: np.ndarray | None = None
    w_meta_r_recon: dict[str, Any] | None = None
    # 条件分岐: `recon != "none"` を満たす経路を評価する。
    if recon != "none":
        w_r_recon_full, w_meta_r_recon = _weights_random_recon(
            rnd, scheme=str(recon_weight_scheme), pair_weight_scheme=str(weight_scheme)
        )

    # Guard: some catalogs can contain tiny negative z (e.g. placeholder/rounding).

    m_g0 = np.isfinite(z_g) & (z_g > 0.0)
    m_r0 = np.isfinite(z_r) & (z_r > 0.0)

    # Reconstruction can use a wider z-range than the measurement bin.
    # This avoids mixing "recon input changes" with "physics/model differences" in comparisons.
    if recon == "none":
        z_min_recon_eff = z_min
        z_max_recon_eff = z_max
    else:
        # 条件分岐: `(recon_z_min is not None) and (z_min is not None) and (float(recon_z_min) > f...` を満たす経路を評価する。
        if (recon_z_min is not None) and (z_min is not None) and (float(recon_z_min) > float(z_min)):
            raise ValueError("--recon-z-min must be <= measurement z_min (or omit --recon-z-min)")

        # 条件分岐: `(recon_z_max is not None) and (z_max is not None) and (float(recon_z_max) < f...` を満たす経路を評価する。

        if (recon_z_max is not None) and (z_max is not None) and (float(recon_z_max) < float(z_max)):
            raise ValueError("--recon-z-max must be >= measurement z_max (or omit --recon-z-max)")

        z_min_recon_eff = z_min if (recon_z_min is None) else float(recon_z_min)
        z_max_recon_eff = z_max if (recon_z_max is None) else float(recon_z_max)

    # Recon selection (possibly wider). We keep it half-open [z_min, z_max) to match the bin-tiling convention.

    m_g_recon = m_g0
    m_r_recon = m_r0
    # 条件分岐: `z_min_recon_eff is not None` を満たす経路を評価する。
    if z_min_recon_eff is not None:
        m_g_recon = m_g_recon & (z_g >= float(z_min_recon_eff))
        m_r_recon = m_r_recon & (z_r >= float(z_min_recon_eff))

    # 条件分岐: `z_max_recon_eff is not None` を満たす経路を評価する。

    if z_max_recon_eff is not None:
        m_g_recon = m_g_recon & (z_g < float(z_max_recon_eff))
        m_r_recon = m_r_recon & (z_r < float(z_max_recon_eff))

    # Recon inputs (possibly wider z-range)

    ra_g_rec, dec_g_rec, z_g_rec, w_g_rec = ra_g[m_g_recon], dec_g[m_g_recon], z_g[m_g_recon], w_g[m_g_recon]
    ra_r_rec, dec_r_rec, z_r_rec, w_r_rec = ra_r[m_r_recon], dec_r[m_r_recon], z_r[m_r_recon], w_r[m_r_recon]
    w_g_recon: np.ndarray | None = None
    w_r_recon: np.ndarray | None = None
    # 条件分岐: `recon != "none"` を満たす経路を評価する。
    if recon != "none":
        assert w_g_recon_full is not None and w_r_recon_full is not None
        w_g_recon = w_g_recon_full[m_g_recon]
        w_r_recon = w_r_recon_full[m_r_recon]

    # Match galaxy/random footprint for the recon inputs (to stabilize recon on subsampled catalogs).

    sector_meta: Dict[str, Any] = {"enabled": False}
    # 条件分岐: `match_sectors` を満たす経路を評価する。
    if match_sectors:
        kg = _sector_keys(gal, sector_key=sector_key)
        kr = _sector_keys(rnd, sector_key=sector_key)
        # 条件分岐: `kg is not None and kr is not None` を満たす経路を評価する。
        if kg is not None and kr is not None:
            kg = kg[m_g_recon]
            kr = kr[m_r_recon]
            valid_g = kg >= 0
            valid_r = kr >= 0
            ug = np.unique(kg[valid_g])
            ur = np.unique(kr[valid_r])
            common = np.intersect1d(ug, ur, assume_unique=True)
            # 条件分岐: `common.size > 0` を満たす経路を評価する。
            if common.size > 0:
                keep_g = valid_g & np.isin(kg, common)
                keep_r = valid_r & np.isin(kr, common)
                ra_g_rec, dec_g_rec, z_g_rec, w_g_rec = (
                    ra_g_rec[keep_g],
                    dec_g_rec[keep_g],
                    z_g_rec[keep_g],
                    w_g_rec[keep_g],
                )
                ra_r_rec, dec_r_rec, z_r_rec, w_r_rec = (
                    ra_r_rec[keep_r],
                    dec_r_rec[keep_r],
                    z_r_rec[keep_r],
                    w_r_rec[keep_r],
                )
                # 条件分岐: `recon != "none"` を満たす経路を評価する。
                if recon != "none":
                    assert w_g_recon is not None and w_r_recon is not None
                    w_g_recon = w_g_recon[keep_g]
                    w_r_recon = w_r_recon[keep_r]

                sector_meta = {
                    "enabled": True,
                    "sector_key": str(sector_key),
                    "n_sectors_gal": int(ug.size),
                    "n_sectors_rnd": int(ur.size),
                    "n_sectors_common": int(common.size),
                    "kept_frac_gal": float(np.mean(keep_g)),
                    "kept_frac_rnd": float(np.mean(keep_r)),
                }

    # Measurement selection is applied after the recon selection (measurement ⊂ recon by construction).

    m_g = np.isfinite(z_g_rec)
    m_r = np.isfinite(z_r_rec)
    # 条件分岐: `z_min is not None` を満たす経路を評価する。
    if z_min is not None:
        m_g = m_g & (z_g_rec >= float(z_min))
        m_r = m_r & (z_r_rec >= float(z_min))

    # 条件分岐: `z_max is not None` を満たす経路を評価する。

    if z_max is not None:
        m_g = m_g & (z_g_rec < float(z_max))
        m_r = m_r & (z_r_rec < float(z_max))

    ra_g, dec_g, z_g, w_g = ra_g_rec[m_g], dec_g_rec[m_g], z_g_rec[m_g], w_g_rec[m_g]
    ra_r, dec_r, z_r, w_r = ra_r_rec[m_r], dec_r_rec[m_r], z_r_rec[m_r], w_r_rec[m_r]

    # Distances are computed once on the recon inputs; measurement arrays are a subset.
    z_all = np.concatenate([z_g_rec, z_r_rec], axis=0).astype(np.float64, copy=False)
    d_all, dist_meta = _comoving_distance_mpc_over_h(
        z_all,
        model=str(dist_model),
        lcdm_omega_m=float(omega_m),
        lcdm_n_grid=int(lcdm_n_grid),
        lcdm_z_grid_max=lcdm_z_grid_max,
    )
    d_all = np.asarray(d_all, dtype=np.float64)
    d_g_rec = d_all[: int(z_g_rec.size)]
    d_r_rec = d_all[int(z_g_rec.size) :]

    d_g = d_g_rec[m_g]
    d_r = d_r_rec[m_r]

    recon_meta: dict[str, Any] | None = None
    # Keep an unshifted copy of randoms for the reconstruction estimator denominator (RR0).
    # NOTE: We intentionally keep views (not deep copies) to avoid memory spikes for large random catalogs.
    ra_r0, dec_r0, d_r0 = ra_r, dec_r, d_r
    # 条件分岐: `recon != "none"` を満たす経路を評価する。
    if recon != "none":
        # 条件分岐: `recon not in ("grid", "mw_multigrid")` を満たす経路を評価する。
        if recon not in ("grid", "mw_multigrid"):
            raise ValueError("--recon must be none/grid/mw_multigrid")

        # 条件分岐: `w_g_recon is None or w_r_recon is None` を満たす経路を評価する。

        if w_g_recon is None or w_r_recon is None:
            raise ValueError("recon enabled but recon weights are missing (internal error)")

        # 条件分岐: `recon == "grid"` を満たす経路を評価する。

        if recon == "grid":
            (ra_g2, dec_g2, d_g2), (ra_r2, dec_r2, d_r2), recon_meta = _apply_reconstruction_grid(
                ra_g=ra_g_rec,
                dec_g=dec_g_rec,
                d_g=d_g_rec,
                w_g=w_g_recon,
                z_g=z_g_rec,
                ra_r=ra_r_rec,
                dec_r=dec_r_rec,
                d_r=d_r_rec,
                w_r=w_r_recon,
                omega_m=float(omega_m),
                recon_grid=int(recon_grid),
                recon_smoothing=float(recon_smoothing),
                recon_assignment=str(recon_assignment),
                recon_bias=float(recon_bias),
                recon_psi_k_sign=int(recon_psi_k_sign),
                recon_f=recon_f,
                recon_f_source=str(recon_f_source),
                recon_mode=str(recon_mode),
                recon_rsd_solver=str(recon_rsd_solver),
                recon_los_shift=str(recon_los_shift),
                recon_rsd_shift=float(recon_rsd_shift),
                recon_pad_fraction=float(recon_pad_fraction),
                recon_mask_expected_frac=recon_mask_expected_frac,
                recon_box_shape=str(recon_box_shape),
                recon_box_frame=str(recon_box_frame),
            )
        else:
            # 条件分岐: `int(recon_grid) != 512` を満たす経路を評価する。
            if int(recon_grid) != 512:
                raise ValueError("mw_multigrid recon requires --recon-grid 512 (upstream grid is fixed)")

            (ra_g2, dec_g2, d_g2), (ra_r2, dec_r2, d_r2), recon_meta = _apply_reconstruction_mw_multigrid(
                ra_g=ra_g_rec,
                dec_g=dec_g_rec,
                z_g=z_g_rec,
                d_g=d_g_rec,
                w_g=w_g_recon,
                ra_r=ra_r_rec,
                dec_r=dec_r_rec,
                z_r=z_r_rec,
                d_r=d_r_rec,
                w_r=w_r_recon,
                dist_model=str(dist_model),
                omega_m=float(omega_m),
                recon_smoothing=float(recon_smoothing),
                recon_bias=float(recon_bias),
                recon_f=recon_f,
                recon_f_source=str(recon_f_source),
                recon_mode=str(recon_mode),
                mw_random_rsd=bool(mw_random_rsd),
                mw_force_rebuild=bool(mw_force_rebuild),
                nthreads=int(nthreads),
            )

        # Slice reconstructed positions to the measurement bin.

        ra_g, dec_g, d_g = ra_g2[m_g], dec_g2[m_g], d_g2[m_g]
        ra_r, dec_r, d_r = ra_r2[m_r], dec_r2[m_r], d_r2[m_r]
        ra_r0, dec_r0, d_r0 = ra_r_rec[m_r], dec_r_rec[m_r], d_r_rec[m_r]

        # 条件分岐: `recon_meta is not None` を満たす経路を評価する。
        if recon_meta is not None:
            try:
                recon_meta = dict(recon_meta)
                spec = dict(recon_meta.get("spec", {}) if isinstance(recon_meta.get("spec", {}), dict) else {})
                spec["weights"] = {
                    "scheme": str(recon_weight_scheme),
                    "galaxy": w_meta_g_recon,
                    "random": w_meta_r_recon,
                }
                spec["z_cut"] = {
                    "z_min_effective": z_min_recon_eff,
                    "z_max_effective": z_max_recon_eff,
                    "z_min_cli": None if (recon_z_min is None) else float(recon_z_min),
                    "z_max_cli": None if (recon_z_max is None) else float(recon_z_max),
                }
                recon_meta["spec"] = spec
            except Exception:
                pass

    xi_estimator: Dict[str, Any]
    # 条件分岐: `recon == "none"` を満たす経路を評価する。
    if recon == "none":
        xi_estimator = {
            "name": "landy_szalay",
            "formula": "xi(s,mu)=(DDn-2DRn+RRn)/RRn",
            "counts": {
                "DD": "galaxy-galaxy (weighted)",
                "DR": "galaxy-random (weighted)",
                "RR": "random-random (weighted)",
            },
        }
    else:
        # Reconstruction estimator (correlation of δ_rec = δ_D - δ_S).
        # D: displaced galaxies, S: shifted randoms, R0: unshifted randoms.
        xi_estimator = {
            "name": "recon_dd_ds_ss_over_rr0",
            "formula": "xi(s,mu)=(DDn-2DSn+SSn)/RR0n",
            "counts": {
                "DD": "displaced galaxies (weighted)",
                "DS": "displaced galaxies × shifted randoms (weighted)",
                "SS": "shifted randoms (weighted)",
                "RR0": "unshifted randoms (weighted)",
            },
            "notes": [
                "Uses RR0 (unshifted randoms) as denominator to keep the survey selection/edges fixed.",
                "This matches the standard reconstruction correlation estimator for δ_rec=δ_D-δ_S.",
            ],
        }

    coordinate_spec: Dict[str, Any] = {
        "redshift": {"definition": str(z_source), "galaxy": z_meta_g, "random": z_meta_r},
        "weights": {"scheme": str(weight_scheme), "galaxy": w_meta_g, "random": w_meta_r},
        "sector_matching": {
            "enabled": bool(match_sectors),
            "sector_key": str(sector_key),
            "applies_to": "recon_inputs",
        },
        "los": {
            "definition": "pairwise_midpoint",
            "backend": ("Corrfunc.DDsmu_mocks" if paircounts_backend == "corrfunc" else "pycorr.TwoPointCorrelationFunction(engine=corrfunc)"),
            "formula": "s=v1-v2, l=(v1+v2)/2, mu=cos(s,l)",
            "cli": str(los),
        },
        "distance": dist_meta,
        "reconstruction": {
            "enabled": bool(recon != "none"),
            "meta": recon_meta,
            "xi_estimator": xi_estimator,
        },
    }

    # Pair counts
    #
    # - recon=none: standard LS using (galaxy, random)
    # - recon!=none: recon estimator using (displaced galaxy, shifted random, unshifted random)
    if paircounts_backend == "pycorr":
        # 条件分岐: `recon != "none"` を満たす経路を評価する。
        if recon != "none":
            raise ValueError("--paircounts-backend pycorr currently supports --recon none only")

        out_counts = _pycorr_paircounts_smu_ls(
            ra_g=ra_g,
            dec_g=dec_g,
            d_g=d_g,
            w_g=w_g,
            ra_r=ra_r,
            dec_r=dec_r,
            d_r=d_r,
            w_r=w_r,
            edges=edges,
            mu_max=mu_max,
            nmu=int(nmu),
            nthreads=int(nthreads),
        )
        dd_w = np.asarray(out_counts["DD_w"], dtype=np.float64)
    else:
        dd_w = _corrfunc_paircounts_smu(
            ra1=ra_g,
            dec1=dec_g,
            dist1=d_g,
            w1=w_g,
            ra2=None,
            dec2=None,
            dist2=None,
            w2=None,
            s_bins_file=s_bins_file,
            mu_max=mu_max,
            nmu=nmu,
            nthreads=nthreads,
            autocorr=1,
        )

    ss_w: np.ndarray | None = None
    # 条件分岐: `recon == "none"` を満たす経路を評価する。
    if recon == "none":
        # 条件分岐: `paircounts_backend == "pycorr"` を満たす経路を評価する。
        if paircounts_backend == "pycorr":
            # 条件分岐: `"DR_w" not in out_counts or "RR_w" not in out_counts` を満たす経路を評価する。
            if "DR_w" not in out_counts or "RR_w" not in out_counts:
                raise ValueError("internal error: missing DR_w/RR_w from pycorr counts")

            dr_w = np.asarray(out_counts["DR_w"], dtype=np.float64)
            rr_w = np.asarray(out_counts["RR_w"], dtype=np.float64)
        else:
            dr_w = _corrfunc_paircounts_smu(
                ra1=ra_g,
                dec1=dec_g,
                dist1=d_g,
                w1=w_g,
                ra2=ra_r,
                dec2=dec_r,
                dist2=d_r,
                w2=w_r,
                s_bins_file=s_bins_file,
                mu_max=mu_max,
                nmu=nmu,
                nthreads=nthreads,
                autocorr=0,
            )
            rr_w = _corrfunc_paircounts_smu(
                ra1=ra_r,
                dec1=dec_r,
                dist1=d_r,
                w1=w_r,
                ra2=None,
                dec2=None,
                dist2=None,
                w2=None,
                s_bins_file=s_bins_file,
                mu_max=mu_max,
                nmu=nmu,
                nthreads=nthreads,
                autocorr=1,
            )
    else:
        # DR_w keeps backward-compatible key name, but is DS in recon mode.
        dr_w = _corrfunc_paircounts_smu(
            ra1=ra_g,
            dec1=dec_g,
            dist1=d_g,
            w1=w_g,
            ra2=ra_r,
            dec2=dec_r,
            dist2=d_r,
            w2=w_r,
            s_bins_file=s_bins_file,
            mu_max=mu_max,
            nmu=nmu,
            nthreads=nthreads,
            autocorr=0,
        )
        ss_w = _corrfunc_paircounts_smu(
            ra1=ra_r,
            dec1=dec_r,
            dist1=d_r,
            w1=w_r,
            ra2=None,
            dec2=None,
            dist2=None,
            w2=None,
            s_bins_file=s_bins_file,
            mu_max=mu_max,
            nmu=nmu,
            nthreads=nthreads,
            autocorr=1,
        )
        # RR_w keeps backward-compatible key name, but is RR0 (unshifted randoms) in recon mode.
        rr_w = _corrfunc_paircounts_smu(
            ra1=ra_r0,
            dec1=dec_r0,
            dist1=d_r0,
            w1=w_r,
            ra2=None,
            dec2=None,
            dist2=None,
            w2=None,
            s_bins_file=s_bins_file,
            mu_max=mu_max,
            nmu=nmu,
            nthreads=nthreads,
            autocorr=1,
        )

    nb = int(edges.size - 1)
    dd_w = dd_w.reshape(nb, int(nmu))
    dr_w = dr_w.reshape(nb, int(nmu))
    rr_w = rr_w.reshape(nb, int(nmu))
    # 条件分岐: `ss_w is not None` を満たす経路を評価する。
    if ss_w is not None:
        ss_w = ss_w.reshape(nb, int(nmu))

    sum_wg = float(np.sum(w_g))
    sum_wg2 = float(np.sum(w_g * w_g))
    sum_wr = float(np.sum(w_r))
    sum_wr2 = float(np.sum(w_r * w_r))

    # Corrfunc autocorr counts ordered pairs (i != j), i.e. twice the unique-pair count.
    # Use the same convention for total weights so that normalized counts represent probabilities.
    dd_tot = (sum_wg * sum_wg - sum_wg2)
    rr_tot = (sum_wr * sum_wr - sum_wr2)
    dr_tot = sum_wg * sum_wr
    ss_tot: float | None = None
    # 条件分岐: `recon != "none"` を満たす経路を評価する。
    if recon != "none":
        ss_tot = rr_tot

    # 条件分岐: `not (dd_tot > 0 and rr_tot > 0 and dr_tot > 0)` を満たす経路を評価する。

    if not (dd_tot > 0 and rr_tot > 0 and dr_tot > 0):
        raise ValueError("invalid total weights (non-positive)")

    ddn = dd_w / dd_tot
    drn = dr_w / dr_tot
    rrn = rr_w / rr_tot
    # 条件分岐: `recon == "none"` を満たす経路を評価する。
    if recon == "none":
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = (ddn - 2.0 * drn + rrn) / rrn
    else:
        # 条件分岐: `ss_w is None or ss_tot is None` を満たす経路を評価する。
        if ss_w is None or ss_tot is None:
            raise ValueError("internal error: recon enabled but SS counts missing")

        ssn = ss_w / float(ss_tot)
        with np.errstate(divide="ignore", invalid="ignore"):
            xi = (ddn - 2.0 * drn + ssn) / rrn

    xi = np.where(np.isfinite(xi), xi, 0.0)

    mu_edges = np.linspace(0.0, float(mu_max), int(nmu) + 1, dtype=float)
    mu_mid = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    dmu = float(mu_edges[1] - mu_edges[0])
    p2 = 0.5 * (3.0 * mu_mid * mu_mid - 1.0)

    xi0 = np.sum(xi * dmu, axis=1)
    xi2 = 5.0 * np.sum(xi * p2[None, :] * dmu, axis=1)

    s_cent = 0.5 * (edges[:-1] + edges[1:])

    return {
        "s": s_cent,
        "xi0": xi0,
        "xi2": xi2,
        # Keep the full xi(s, mu) grid for downstream wedge integration.
        # This is small (nbins*nmu) and avoids relying on truncation to ℓ<=2.
        "xi_mu": xi,
        "mu_edges": mu_edges,
        "counts": {"DD_w": dd_w, "DR_w": dr_w, "RR_w": rr_w, **({"SS_w": ss_w} if (ss_w is not None) else {})},
        "sectors": sector_meta,
        "coordinate_spec": coordinate_spec,
        "totals": {
            "sum_w_gal": sum_wg,
            "sum_w2_gal": sum_wg2,
            "sum_w_rnd": sum_wr,
            "sum_w2_rnd": sum_wr2,
            "dd_tot": dd_tot,
            "dr_tot": dr_tot,
            "rr_tot": rr_tot,
            **({"ss_tot": float(ss_tot)} if (ss_tot is not None) else {}),
        },
        "effective": {
            "z_eff_gal_weighted": float(np.sum(z_g * w_g) / np.sum(w_g)),
            "z_eff_rnd_weighted": float(np.sum(z_r * w_r) / np.sum(w_r)),
        },
        "sizes": {"n_gal": int(z_g.size), "n_rnd": int(z_r.size)},
    }


def _xi_multipoles_from_paircounts(
    *,
    dd_w: np.ndarray,
    dr_w: np.ndarray,
    rr_w: np.ndarray,
    edges: np.ndarray,
    mu_max: float,
    nmu: int,
    dd_tot: float,
    dr_tot: float,
    rr_tot: float,
) -> Dict[str, np.ndarray]:
    nb = int(edges.size - 1)
    dd_w = np.asarray(dd_w, dtype=np.float64).reshape(nb, int(nmu))
    dr_w = np.asarray(dr_w, dtype=np.float64).reshape(nb, int(nmu))
    rr_w = np.asarray(rr_w, dtype=np.float64).reshape(nb, int(nmu))

    ddn = dd_w / float(dd_tot)
    drn = dr_w / float(dr_tot)
    rrn = rr_w / float(rr_tot)

    with np.errstate(divide="ignore", invalid="ignore"):
        xi = (ddn - 2.0 * drn + rrn) / rrn

    xi = np.where(np.isfinite(xi), xi, 0.0)

    mu_edges = np.linspace(0.0, float(mu_max), int(nmu) + 1, dtype=float)
    mu_mid = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    dmu = float(mu_edges[1] - mu_edges[0])
    p2 = 0.5 * (3.0 * mu_mid * mu_mid - 1.0)

    xi0 = np.sum(xi * dmu, axis=1)
    xi2 = 5.0 * np.sum(xi * p2[None, :] * dmu, axis=1)
    s_cent = 0.5 * (edges[:-1] + edges[1:])
    return {
        "s": s_cent,
        "xi0": xi0,
        "xi2": xi2,
        "xi_mu": xi,
        "mu_edges": mu_edges,
    }


def _xi_multipoles_from_recon_paircounts(
    *,
    dd_w: np.ndarray,
    ds_w: np.ndarray,
    ss_w: np.ndarray,
    rr0_w: np.ndarray,
    edges: np.ndarray,
    mu_max: float,
    nmu: int,
    dd_tot: float,
    ds_tot: float,
    ss_tot: float,
    rr0_tot: float,
) -> Dict[str, np.ndarray]:
    """
    Reconstruction multipoles from paircounts.

    Estimator (correlation of δ_rec = δ_D - δ_S):
        xi(s,mu) = (DDn - 2 DSn + SSn) / RR0n

    where:
      - D: displaced galaxies
      - S: shifted randoms
      - R0: unshifted randoms (keeps the survey selection/edges fixed)
    """
    nb = int(edges.size - 1)
    dd_w = np.asarray(dd_w, dtype=np.float64).reshape(nb, int(nmu))
    ds_w = np.asarray(ds_w, dtype=np.float64).reshape(nb, int(nmu))
    ss_w = np.asarray(ss_w, dtype=np.float64).reshape(nb, int(nmu))
    rr0_w = np.asarray(rr0_w, dtype=np.float64).reshape(nb, int(nmu))

    ddn = dd_w / float(dd_tot)
    dsn = ds_w / float(ds_tot)
    ssn = ss_w / float(ss_tot)
    rr0n = rr0_w / float(rr0_tot)

    with np.errstate(divide="ignore", invalid="ignore"):
        xi = (ddn - 2.0 * dsn + ssn) / rr0n

    xi = np.where(np.isfinite(xi), xi, 0.0)

    mu_edges = np.linspace(0.0, float(mu_max), int(nmu) + 1, dtype=float)
    mu_mid = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    dmu = float(mu_edges[1] - mu_edges[0])
    p2 = 0.5 * (3.0 * mu_mid * mu_mid - 1.0)

    xi0 = np.sum(xi * dmu, axis=1)
    xi2 = 5.0 * np.sum(xi * p2[None, :] * dmu, axis=1)
    s_cent = 0.5 * (edges[:-1] + edges[1:])
    return {
        "s": s_cent,
        "xi0": xi0,
        "xi2": xi2,
        "xi_mu": xi,
        "mu_edges": mu_edges,
    }


def _estimate_bao_peak_s2_xi(
    *,
    s: np.ndarray,
    xi: np.ndarray,
    fit_deg: int = 3,
    fit_range: tuple[float, float] = (30.0, 150.0),
    exclude_range: tuple[float, float] = (80.0, 130.0),
    search_range: tuple[float, float] = (80.0, 130.0),
) -> dict[str, float | int | str | list[float]]:
    """
    Rough BAO-peak estimator for Phase A screening.

    Work in y(s) = s^2 * xi(s). Fit a low-order polynomial to the broadband part
    (excluding BAO window), then take the residual peak position within the BAO window.
    """
    s = np.asarray(s, dtype=np.float64)
    xi = np.asarray(xi, dtype=np.float64)
    # 条件分岐: `s.shape != xi.shape` を満たす経路を評価する。
    if s.shape != xi.shape:
        raise ValueError("s/xi shape mismatch")

    # 条件分岐: `int(fit_deg) < 1` を満たす経路を評価する。

    if int(fit_deg) < 1:
        raise ValueError("fit_deg must be >= 1")

    y = (s * s) * xi
    s0, s1 = (float(fit_range[0]), float(fit_range[1]))
    e0, e1 = (float(exclude_range[0]), float(exclude_range[1]))
    w0, w1 = (float(search_range[0]), float(search_range[1]))

    m_fit = (s >= s0) & (s <= s1) & ~((s >= e0) & (s <= e1)) & np.isfinite(y)
    # 条件分岐: `int(np.count_nonzero(m_fit)) < (int(fit_deg) + 2)` を満たす経路を評価する。
    if int(np.count_nonzero(m_fit)) < (int(fit_deg) + 2):
        raise ValueError("insufficient points for broadband fit")

    coef = np.polyfit(s[m_fit], y[m_fit], deg=int(fit_deg))
    y_smooth = np.polyval(coef, s)
    y_res = y - y_smooth

    m_win = (s >= w0) & (s <= w1) & np.isfinite(y_res)
    # 条件分岐: `not np.any(m_win)` を満たす経路を評価する。
    if not np.any(m_win):
        raise ValueError("no points in search window")

    idx_win = np.where(m_win)[0]
    # 条件分岐: `idx_win.size == 0` を満たす経路を評価する。
    if idx_win.size == 0:
        raise ValueError("no points in search window")

    # Prefer a true local maximum within the window to avoid edge-picking when
    # the residual is monotonic/noisy (common for low S/N wedges).

    cand_mask = np.zeros_like(m_win, dtype=bool)
    # 条件分岐: `s.size >= 3` を満たす経路を評価する。
    if s.size >= 3:
        cand_mask[1:-1] = (
            m_win[1:-1]
            & m_win[:-2]
            & m_win[2:]
            & np.isfinite(y_res[1:-1])
            & np.isfinite(y_res[:-2])
            & np.isfinite(y_res[2:])
            & (y_res[1:-1] >= y_res[:-2])
            & (y_res[1:-1] > y_res[2:])
        )

    cand_idxs = np.where(cand_mask)[0]
    # 条件分岐: `cand_idxs.size > 0` を満たす経路を評価する。
    if cand_idxs.size > 0:
        idx = int(cand_idxs[np.argmax(y_res[cand_idxs])])
        idx_choice = "local_max"
    else:
        idx = int(idx_win[np.argmax(y_res[idx_win])])
        idx_choice = "global_max"

    is_window_edge = bool((idx == int(idx_win[0])) or (idx == int(idx_win[-1])))

    # Optional quadratic interpolation when interior point is available.
    s_peak = float(s[idx])
    y_res_peak = float(y_res[idx])
    # 条件分岐: `0 < idx < int(s.size) - 1 and np.all(np.isfinite(y_res[idx - 1 : idx + 2]))` を満たす経路を評価する。
    if 0 < idx < int(s.size) - 1 and np.all(np.isfinite(y_res[idx - 1 : idx + 2])):
        x = s[idx - 1 : idx + 2]
        yy = y_res[idx - 1 : idx + 2]
        try:
            a, b, c = np.polyfit(x, yy, deg=2)
            # 条件分岐: `float(a) < 0.0` を満たす経路を評価する。
            if float(a) < 0.0:
                s_star = -float(b) / (2.0 * float(a))
                # 条件分岐: `float(x[0]) <= s_star <= float(x[-1])` を満たす経路を評価する。
                if float(x[0]) <= s_star <= float(x[-1]):
                    s_peak = float(s_star)
                    y_res_peak = float(np.polyval([a, b, c], s_star))
        except Exception:
            pass

    return {
        "method": "s2_xi_poly_broadband_peak",
        "fit_deg": int(fit_deg),
        "fit_range": [s0, s1],
        "exclude_range": [e0, e1],
        "search_range": [w0, w1],
        "idx_bin": int(idx),
        "idx_choice": str(idx_choice),
        "is_window_edge": bool(is_window_edge),
        "s_peak": s_peak,
        "residual_peak": y_res_peak,
        "y_peak": float(np.interp(s_peak, s, y)),
    }


def _estimate_bao_peak_s2_xi0(
    *,
    s: np.ndarray,
    xi0: np.ndarray,
    fit_deg: int = 3,
    fit_range: tuple[float, float] = (30.0, 150.0),
    exclude_range: tuple[float, float] = (80.0, 130.0),
    search_range: tuple[float, float] = (80.0, 130.0),
) -> dict[str, float | int | str | list[float]]:
    # Backward-compatible wrapper (historical name).
    return _estimate_bao_peak_s2_xi(
        s=s,
        xi=xi0,
        fit_deg=fit_deg,
        fit_range=fit_range,
        exclude_range=exclude_range,
        search_range=search_range,
    )


def _estimate_bao_feature_s2_xi(
    *,
    s: np.ndarray,
    xi: np.ndarray,
    fit_deg: int = 3,
    fit_range: tuple[float, float] = (30.0, 150.0),
    exclude_range: tuple[float, float] = (80.0, 130.0),
    search_range: tuple[float, float] = (80.0, 130.0),
) -> dict[str, float | int | str | list[float]]:
    """
    Rough BAO-feature estimator for Phase A screening.

    Work in y(s)=s^2*xi(s). Fit a low-order polynomial to the broadband part
    (excluding BAO window), then find both the maximum and minimum of the residual
    within the BAO window. Report the "dominant" extremum by absolute residual.

    Intended use:
      - xi0: peak position (use _estimate_bao_peak_s2_xi0)
      - xi2: anisotropy-sensitive feature (sign and position can flip under warping)
    """
    s = np.asarray(s, dtype=np.float64)
    xi = np.asarray(xi, dtype=np.float64)
    # 条件分岐: `s.shape != xi.shape` を満たす経路を評価する。
    if s.shape != xi.shape:
        raise ValueError("s/xi shape mismatch")

    # 条件分岐: `int(fit_deg) < 1` を満たす経路を評価する。

    if int(fit_deg) < 1:
        raise ValueError("fit_deg must be >= 1")

    y = (s * s) * xi
    s0, s1 = (float(fit_range[0]), float(fit_range[1]))
    e0, e1 = (float(exclude_range[0]), float(exclude_range[1]))
    w0, w1 = (float(search_range[0]), float(search_range[1]))

    m_fit = (s >= s0) & (s <= s1) & ~((s >= e0) & (s <= e1)) & np.isfinite(y)
    # 条件分岐: `int(np.count_nonzero(m_fit)) < (int(fit_deg) + 2)` を満たす経路を評価する。
    if int(np.count_nonzero(m_fit)) < (int(fit_deg) + 2):
        raise ValueError("insufficient points for broadband fit")

    coef = np.polyfit(s[m_fit], y[m_fit], deg=int(fit_deg))
    y_smooth = np.polyval(coef, s)
    y_res = y - y_smooth

    m_win = (s >= w0) & (s <= w1) & np.isfinite(y_res)
    # 条件分岐: `not np.any(m_win)` を満たす経路を評価する。
    if not np.any(m_win):
        raise ValueError("no points in search window")

    win_idx = np.where(m_win)[0]

    def _refine(idx: int, *, want_max: bool) -> tuple[float, float]:
        s_star = float(s[idx])
        y_star = float(y_res[idx])
        # 条件分岐: `not (0 < idx < int(s.size) - 1)` を満たす経路を評価する。
        if not (0 < idx < int(s.size) - 1):
            return s_star, y_star

        # 条件分岐: `not np.all(np.isfinite(y_res[idx - 1 : idx + 2]))` を満たす経路を評価する。

        if not np.all(np.isfinite(y_res[idx - 1 : idx + 2])):
            return s_star, y_star

        x = s[idx - 1 : idx + 2]
        yy = y_res[idx - 1 : idx + 2]
        try:
            a, b, c = np.polyfit(x, yy, deg=2)
            a = float(a)
            b = float(b)
            # 条件分岐: `(want_max and a < 0.0) or ((not want_max) and a > 0.0)` を満たす経路を評価する。
            if (want_max and a < 0.0) or ((not want_max) and a > 0.0):
                s_hat = -b / (2.0 * a)
                # 条件分岐: `float(x[0]) <= s_hat <= float(x[-1])` を満たす経路を評価する。
                if float(x[0]) <= s_hat <= float(x[-1]):
                    return float(s_hat), float(np.polyval([a, b, c], s_hat))
        except Exception:
            pass

        return s_star, y_star

    idx_max = int(win_idx[int(np.argmax(y_res[m_win]))])
    idx_min = int(win_idx[int(np.argmin(y_res[m_win]))])
    s_max, y_res_max = _refine(idx_max, want_max=True)
    s_min, y_res_min = _refine(idx_min, want_max=False)

    # Dominant feature by absolute residual
    if abs(float(y_res_max)) >= abs(float(y_res_min)):
        s_abs = float(s_max)
        y_res_abs = float(y_res_max)
    else:
        s_abs = float(s_min)
        y_res_abs = float(y_res_min)

    return {
        "method": "s2_xi_poly_broadband",
        "fit_deg": int(fit_deg),
        "fit_range": [s0, s1],
        "exclude_range": [e0, e1],
        "search_range": [w0, w1],
        "s_abs": s_abs,
        "residual_abs": y_res_abs,
        "y_abs": float(np.interp(s_abs, s, y)),
        "s_max": float(s_max),
        "residual_max": float(y_res_max),
        "y_max": float(np.interp(float(s_max), s, y)),
        "s_min": float(s_min),
        "residual_min": float(y_res_min),
        "y_min": float(np.interp(float(s_min), s, y)),
    }


def _p2_antiderivative(mu: float) -> float:
    # ∫ P2(μ) dμ, where P2(μ)=0.5*(3μ^2-1)
    mu = float(mu)
    return 0.5 * (mu * mu * mu - mu)


def _p2_avg(mu0: float, mu1: float) -> float:
    mu0 = float(mu0)
    mu1 = float(mu1)
    # 条件分岐: `not (mu1 > mu0)` を満たす経路を評価する。
    if not (mu1 > mu0):
        raise ValueError("invalid mu range")

    return (_p2_antiderivative(mu1) - _p2_antiderivative(mu0)) / (mu1 - mu0)


def _xi_wedge_from_multipoles(*, xi0: np.ndarray, xi2: np.ndarray, mu0: float, mu1: float) -> tuple[np.ndarray, float]:
    """
    Approximate ξ_wedge(s) over μ∈[mu0,mu1] using only ξ0 and ξ2:
      ξ(s,μ) ≈ ξ0(s) + ξ2(s) P2(μ)
      ξ_wedge(s) = <ξ> = ξ0(s) + ξ2(s) <P2>_[mu0,mu1]
    """
    c2 = _p2_avg(mu0, mu1)
    return np.asarray(xi0, dtype=np.float64) + float(c2) * np.asarray(xi2, dtype=np.float64), float(c2)


def _float_token(x: float) -> str:
    s = f"{float(x):.6g}"
    return s.replace("-", "m").replace(".", "p")


def _zcut_tag(*, z_bin: str, z_min: float | None, z_max: float | None) -> str:
    # 条件分岐: `z_min is None and z_max is None` を満たす経路を評価する。
    if z_min is None and z_max is None:
        return ""

    # 条件分岐: `z_bin != "none"` を満たす経路を評価する。

    if z_bin != "none":
        return z_bin

    parts: list[str] = []
    # 条件分岐: `z_min is not None` を満たす経路を評価する。
    if z_min is not None:
        parts.append(f"zmin{_float_token(z_min)}")

    # 条件分岐: `z_max is not None` を満たす経路を評価する。

    if z_max is not None:
        parts.append(f"zmax{_float_token(z_max)}")

    return "_".join(parts)


def _sanitize_out_tag(tag: str) -> str:
    t = str(tag).strip()
    # 条件分岐: `not t` を満たす経路を評価する。
    if not t:
        return ""

    out: list[str] = []
    for ch in t:
        # 条件分岐: `ch.isalnum() or ch in ("-", "_", ".")` を満たす経路を評価する。
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")

    s = "".join(out).strip("._-")
    return s[:80]


def _append_out_tag(tag: str, suffix: str) -> str:
    """
    Append a sanitized suffix to out_tag while avoiding duplication.

    Notes:
    - This is used to prevent accidental overwrites when running sensitivity variants
      without explicitly providing `--out-tag`.
    """
    tag = _sanitize_out_tag(tag)
    suffix = _sanitize_out_tag(suffix)
    # 条件分岐: `not suffix` を満たす経路を評価する。
    if not suffix:
        return tag

    # 条件分岐: `not tag` を満たす経路を評価する。

    if not tag:
        return suffix

    # 条件分岐: `suffix in tag` を満たす経路を評価する。

    if suffix in tag:
        return tag

    return _sanitize_out_tag(f"{tag}_{suffix}")


def _reconfigure_stdio_best_effort() -> None:
    """
    Best-effort mitigation for Windows cp932 consoles where argparse help can
    crash on Unicode (e.g., "∇·Ψ"). We keep the user's encoding but replace
    unencodable characters with backslash escapes.
    """
    try:
        for s in (sys.stdout, sys.stderr):
            # 条件分岐: `hasattr(s, "reconfigure")` を満たす経路を評価する。
            if hasattr(s, "reconfigure"):
                s.reconfigure(errors="backslashreplace")
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    _reconfigure_stdio_best_effort()
    ap = argparse.ArgumentParser(
        description=(
            "Compute xi multipoles from galaxy+random catalogs. "
            "Default data source is BOSS DR12v5 (use --data-dir to switch; e.g., eBOSS DR16)."
        )
    )
    ap.add_argument(
        "--data-dir",
        default="data/cosmology/boss_dr12v5_lss",
        help="data directory containing manifest.json (default: data/cosmology/boss_dr12v5_lss)",
    )
    ap.add_argument("--sample", required=True, help="sample id (manifest key prefix; e.g., cmasslowztot, lrgpcmass_rec)")
    ap.add_argument("--caps", choices=["combined", "north", "south"], default="combined", help="sky cap selection")
    ap.add_argument("--dist", choices=["lcdm", "pbg"], default="lcdm", help="distance mapping: lcdm or pbg (static)")
    ap.add_argument("--lcdm-omega-m", type=float, default=0.315, help="LCDM Ωm for z->D_M (default: 0.315)")
    ap.add_argument("--lcdm-n-grid", type=int, default=6000, help="LCDM integration grid points (default: 6000)")
    ap.add_argument("--lcdm-z-grid-max", type=float, default=2.0, help="LCDM integration z_max for grid (default: 2.0; should exceed max(z))")
    ap.add_argument("--z-source", choices=sorted(_Z_SOURCE_TO_COLUMN_CANDIDATES), default="obs", help="redshift definition (default: obs)")
    ap.add_argument("--los", choices=["midpoint"], default="midpoint", help="LOS definition (fixed by Corrfunc; default: midpoint)")
    ap.add_argument(
        "--weight-scheme",
        choices=["boss_default", "desi_default", "fkp_only", "none"],
        default="boss_default",
        help="catalog weight scheme (default: boss_default)",
    )
    ap.add_argument("--random-kind", default="random1", help="random key (default: random1; e.g., random0/random1/random)")
    ap.add_argument(
        "--match-sectors",
        choices=["auto", "on", "off"],
        default="auto",
        help="match galaxy/random footprint by common sectors (default: auto)",
    )
    ap.add_argument(
        "--sector-key",
        choices=["isect", "ipoly_isect"],
        default="isect",
        help="sector key used when --match-sectors is on/auto (default: isect)",
    )
    ap.add_argument(
        "--recon",
        choices=["none", "grid", "mw_multigrid"],
        default="none",
        help="BAO reconstruction (Phase B; default: none). mw_multigrid uses an external finite-difference solver (WSL only).",
    )
    ap.add_argument(
        "--recon-z-min",
        type=float,
        default=None,
        help="z min used for reconstruction density field (default: same as measurement z_min / z-bin)",
    )
    ap.add_argument(
        "--recon-z-max",
        type=float,
        default=None,
        help="z max used for reconstruction density field (default: same as measurement z_max / z-bin; half-open)",
    )
    ap.add_argument(
        "--recon-weight-scheme",
        choices=["same", "boss_recon"],
        default="same",
        help="weights used to build the recon density field (default: same)",
    )
    ap.add_argument("--recon-grid", type=int, default=256, help="recon grid size per axis (default: 256)")
    ap.add_argument("--recon-smoothing", type=float, default=15.0, help="recon Gaussian smoothing R [Mpc/h] (default: 15)")
    ap.add_argument(
        "--recon-assignment",
        choices=["ngp", "cic"],
        default="ngp",
        help="mass assignment & interpolation for recon grid (default: ngp)",
    )
    ap.add_argument(
        "--recon-bias",
        type=float,
        default=2.0,
        help="recon linear bias b for δ/b (default: 2.0; Ross 2016 uses b=1.85 for data)",
    )
    ap.add_argument(
        "--recon-psi-k-sign",
        type=int,
        choices=[-1, 1],
        default=-1,
        help="recon FFT sign in Ψ(k)=(sign) i k δ/(b k^2). Default -1 is legacy; use +1 for δ=-∇·Ψ convention tests.",
    )
    ap.add_argument("--recon-f", type=float, default=None, help="recon growth rate f for RSD term (default: auto from LCDM Ωm at z_eff)")
    ap.add_argument("--recon-mode", choices=["ani", "iso"], default="ani", help="recon shift mode: ani (with RSD) or iso (default: ani)")
    ap.add_argument(
        "--recon-rsd-solver",
        choices=["global_constant", "padmanabhan_pp", "none"],
        default="global_constant",
        help=(
            "RSD-aware solver for displacement field (ani mode only). "
            "global_constant applies 1/(1+β μ^2) with a global LOS (β=f/b; Kaiser inversion); "
            "padmanabhan_pp applies 1/(1+f μ^2) with a global LOS (Padmanabhan 2012 plane-parallel); "
            "none disables (default: global_constant)."
        ),
    )
    ap.add_argument(
        "--recon-los-shift",
        choices=["global_mean_direction", "radial"],
        default="global_mean_direction",
        help=(
            "LOS model for the galaxy shift term in ani mode. "
            "global_mean_direction uses a single LOS for the sample; radial uses per-object LOS (default: global_mean_direction)"
        ),
    )
    ap.add_argument(
        "--recon-rsd-shift",
        type=float,
        default=1.0,
        help="RSD shift factor for ani mode: Ψ_s = Ψ + (factor*f)*(Ψ·LOS)LOS (default: 1.0)",
    )
    ap.add_argument("--recon-pad-fraction", type=float, default=0.1, help="recon box padding fraction (default: 0.1)")
    ap.add_argument(
        "--recon-box-shape",
        choices=["rect", "cube"],
        default="rect",
        help="recon box shape: rect (tight AABB) or cube (max-span cube) (default: rect)",
    )
    ap.add_argument(
        "--recon-box-frame",
        choices=["raw", "pca"],
        default="raw",
        help=(
            "recon box frame: raw (original xyz) or pca (rotate to PCA axes before boxing; invert rotation after recon) "
            "(default: raw)"
        ),
    )
    ap.add_argument(
        "--recon-mask-expected-frac",
        type=float,
        default=None,
        help=(
            "mask threshold for reconstruction: ignore cells with expected(random) <= (frac * median(expected>0)). "
            "Default: auto (ngp=0.0, cic=1.5) to avoid CIC-induced delta blow-up."
        ),
    )
    ap.add_argument(
        "--mw-random-rsd",
        action="store_true",
        help="mw_multigrid: also apply RSD-enhanced LOS shift to shifted randoms (default: off; Padmanabhan 2012 convention).",
    )
    ap.add_argument(
        "--mw-force-rebuild",
        action="store_true",
        help="mw_multigrid: force rebuild of local FFTW and recon_mw binary (default: off).",
    )
    ap.add_argument(
        "--out-tag",
        default="",
        help=(
            "optional suffix for outputs (keeps default filenames when empty; for convergence runs). "
            "Note: when --recon!=none, this tool auto-prefixes 'recon_<mode>' and auto-appends 'assign_<ngp/cic>' "
            "so you normally pass just a short tag like 'iso' or 'iso_s10'."
        ),
    )
    ap.add_argument("--s-min", type=float, default=30.0, help="s min [Mpc/h] (default: 30)")
    ap.add_argument("--s-max", type=float, default=150.0, help="s max [Mpc/h] (default: 150)")
    ap.add_argument("--s-step", type=float, default=5.0, help="s step [Mpc/h] (default: 5)")
    ap.add_argument(
        "--z-bin",
        choices=["none", "b1", "b2", "b3"],
        default="none",
        help="optional redshift bin for CMASSLOWZ studies (default: none). "
        "b1:[0.2,0.5), b2:[0.4,0.6), b3:[0.5,0.75)",
    )
    ap.add_argument("--z-min", type=float, default=None, help="override z min (default: from --z-bin)")
    ap.add_argument("--z-max", type=float, default=None, help="override z max (default: from --z-bin)")
    ap.add_argument("--nmu", type=int, default=120, help="mu bins (default: 120)")
    ap.add_argument("--mu-max", type=float, default=1.0, help="mu max (default: 1.0)")
    ap.add_argument(
        "--mu-split",
        type=float,
        default=0.5,
        help="mu wedge split for transverse/radial screening (default: 0.5; valid when mu_max is close to 1)",
    )
    ap.add_argument("--threads", type=int, default=24, help="Corrfunc threads (WSL rule: default 24)")
    ap.add_argument(
        "--paircounts-backend",
        choices=["corrfunc", "pycorr"],
        default="corrfunc",
        help=(
            "paircounts backend: corrfunc (direct) or pycorr (TwoPointCorrelationFunction; engine=corrfunc). "
            "pycorr requires cosmodesi/Corrfunc@desi in the WSL venv."
        ),
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    data_dir = _resolve_manifest_path(str(args.data_dir))
    manifest_path = data_dir / "manifest.json"
    # 条件分岐: `not manifest_path.exists()` を満たす経路を評価する。
    if not manifest_path.exists():
        raise SystemExit(
            f"manifest not found: {manifest_path} "
            f"(run the corresponding fetch_* script first; e.g., fetch_boss_dr12v5_lss.py)"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    sample = str(args.sample)
    caps = str(args.caps)
    dist = str(args.dist)
    omega_m = float(args.lcdm_omega_m)
    lcdm_n_grid = int(args.lcdm_n_grid)
    lcdm_z_grid_max = float(args.lcdm_z_grid_max)
    z_source = str(args.z_source)
    los = str(args.los)
    weight_scheme = str(args.weight_scheme)
    random_kind = str(args.random_kind)
    match_sectors_arg = str(args.match_sectors)
    sector_key = str(args.sector_key)
    recon = str(args.recon)
    recon_z_min = args.recon_z_min
    recon_z_max = args.recon_z_max
    recon_weight_scheme = str(args.recon_weight_scheme)
    recon_grid = int(args.recon_grid)
    recon_smoothing = float(args.recon_smoothing)
    recon_assignment = str(args.recon_assignment)
    recon_bias = float(args.recon_bias)
    recon_psi_k_sign = int(args.recon_psi_k_sign)
    recon_f = args.recon_f
    recon_f_source = "cli" if (recon_f is not None) else "lcdm_omega_m(z_eff)^0.55"
    recon_mode = str(args.recon_mode)
    recon_rsd_solver = str(args.recon_rsd_solver)
    recon_los_shift = str(args.recon_los_shift)
    recon_rsd_shift = float(args.recon_rsd_shift)
    recon_pad_fraction = float(args.recon_pad_fraction)
    recon_mask_expected_frac = args.recon_mask_expected_frac
    recon_box_shape = str(args.recon_box_shape)
    recon_box_frame = str(args.recon_box_frame)
    mw_random_rsd = bool(args.mw_random_rsd)
    mw_force_rebuild = bool(args.mw_force_rebuild)
    paircounts_backend = str(args.paircounts_backend)
    out_tag = _sanitize_out_tag(str(args.out_tag))
    z_bin = str(args.z_bin)
    z_min = args.z_min
    z_max = args.z_max

    # 条件分岐: `weight_scheme != "boss_default" and not out_tag` を満たす経路を評価する。
    if weight_scheme != "boss_default" and not out_tag:
        # Avoid overwriting the default outputs when running sensitivity variants.
        out_tag = _sanitize_out_tag(f"w_{weight_scheme}")

    # 条件分岐: `recon != "none" and recon_weight_scheme != "same"` を満たす経路を評価する。

    if recon != "none" and recon_weight_scheme != "same":
        suffix = _sanitize_out_tag(f"rw_{recon_weight_scheme}")
        # 条件分岐: `not out_tag` を満たす経路を評価する。
        if not out_tag:
            out_tag = suffix
        # 条件分岐: 前段条件が不成立で、`suffix not in out_tag` を追加評価する。
        elif suffix not in out_tag:
            out_tag = _sanitize_out_tag(f"{out_tag}_{suffix}")

    # 条件分岐: `recon != "none" and ((recon_z_min is not None) or (recon_z_max is not None))` を満たす経路を評価する。

    if recon != "none" and ((recon_z_min is not None) or (recon_z_max is not None)):
        ztag = _zcut_tag(z_bin="none", z_min=recon_z_min, z_max=recon_z_max)
        suffix = _sanitize_out_tag(f"rz_{ztag}")
        # 条件分岐: `suffix and suffix not in out_tag` を満たす経路を評価する。
        if suffix and suffix not in out_tag:
            out_tag = _sanitize_out_tag(f"{out_tag}_{suffix}") if out_tag else suffix

    # 条件分岐: `match_sectors_arg != "auto"` を満たす経路を評価する。

    if match_sectors_arg != "auto":
        out_tag = _append_out_tag(out_tag, f"ms_{match_sectors_arg}")

    # 条件分岐: `str(sector_key) != str(ap.get_default("sector_key"))` を満たす経路を評価する。

    if str(sector_key) != str(ap.get_default("sector_key")):
        out_tag = _append_out_tag(out_tag, f"sk_{sector_key}")

    # 条件分岐: `recon != "none"` を満たす経路を評価する。

    if recon != "none":
        # 条件分岐: `recon_grid < 16` を満たす経路を評価する。
        if recon_grid < 16:
            raise SystemExit("--recon-grid must be >= 16")

        # 条件分岐: `not (recon_smoothing > 0.0)` を満たす経路を評価する。

        if not (recon_smoothing > 0.0):
            raise SystemExit("--recon-smoothing must be > 0")

        # 条件分岐: `recon_assignment not in ("ngp", "cic")` を満たす経路を評価する。

        if recon_assignment not in ("ngp", "cic"):
            raise SystemExit("--recon-assignment must be ngp/cic")

        # 条件分岐: `not (recon_bias > 0.0)` を満たす経路を評価する。

        if not (recon_bias > 0.0):
            raise SystemExit("--recon-bias must be > 0")

        # 条件分岐: `recon_psi_k_sign not in (-1, 1)` を満たす経路を評価する。

        if recon_psi_k_sign not in (-1, 1):
            raise SystemExit("--recon-psi-k-sign must be -1 or +1")

        # 条件分岐: `not (recon_pad_fraction >= 0.0)` を満たす経路を評価する。

        if not (recon_pad_fraction >= 0.0):
            raise SystemExit("--recon-pad-fraction must be >= 0")

        # 条件分岐: `not np.isfinite(recon_rsd_shift)` を満たす経路を評価する。

        if not np.isfinite(recon_rsd_shift):
            raise SystemExit("--recon-rsd-shift must be finite")

        # 条件分岐: `(recon_mask_expected_frac is not None) and (not np.isfinite(float(recon_mask_...` を満たす経路を評価する。

        if (recon_mask_expected_frac is not None) and (not np.isfinite(float(recon_mask_expected_frac))):
            raise SystemExit("--recon-mask-expected-frac must be finite when provided")

        # 条件分岐: `(recon_mask_expected_frac is not None) and (float(recon_mask_expected_frac) <...` を満たす経路を評価する。

        if (recon_mask_expected_frac is not None) and (float(recon_mask_expected_frac) < 0.0):
            raise SystemExit("--recon-mask-expected-frac must be >= 0")

        # 条件分岐: `recon == "mw_multigrid"` を満たす経路を評価する。

        if recon == "mw_multigrid":
            # 条件分岐: `int(recon_grid) != 512` を満たす経路を評価する。
            if int(recon_grid) != 512:
                raise SystemExit("mw_multigrid recon requires --recon-grid 512 (upstream grid is fixed)")
            # Guard: options that are meaningful only for the in-repo grid recon should stay at defaults
            # to avoid mixing "backend differences" with "option differences" in discussions.

            if str(recon_assignment) != str(ap.get_default("recon_assignment")):
                raise SystemExit("mw_multigrid recon does not use --recon-assignment (keep default)")

            # 条件分岐: `int(recon_psi_k_sign) != int(ap.get_default("recon_psi_k_sign"))` を満たす経路を評価する。

            if int(recon_psi_k_sign) != int(ap.get_default("recon_psi_k_sign")):
                raise SystemExit("mw_multigrid recon does not use --recon-psi-k-sign (keep default)")

            # 条件分岐: `not math.isclose(float(recon_pad_fraction), float(ap.get_default("recon_pad_f...` を満たす経路を評価する。

            if not math.isclose(float(recon_pad_fraction), float(ap.get_default("recon_pad_fraction")), rel_tol=0.0, abs_tol=1e-12):
                raise SystemExit("mw_multigrid recon does not use --recon-pad-fraction (keep default)")

            # 条件分岐: `recon_mask_expected_frac is not None` を満たす経路を評価する。

            if recon_mask_expected_frac is not None:
                raise SystemExit("mw_multigrid recon does not use --recon-mask-expected-frac (omit)")

            # 条件分岐: `str(recon_box_shape) != str(ap.get_default("recon_box_shape"))` を満たす経路を評価する。

            if str(recon_box_shape) != str(ap.get_default("recon_box_shape")):
                raise SystemExit("mw_multigrid recon does not use --recon-box-shape (keep default)")

            # 条件分岐: `str(recon_box_frame) != str(ap.get_default("recon_box_frame"))` を満たす経路を評価する。

            if str(recon_box_frame) != str(ap.get_default("recon_box_frame")):
                raise SystemExit("mw_multigrid recon does not use --recon-box-frame (keep default)")

            # 条件分岐: `str(recon_rsd_solver) != str(ap.get_default("recon_rsd_solver"))` を満たす経路を評価する。

            if str(recon_rsd_solver) != str(ap.get_default("recon_rsd_solver")):
                raise SystemExit("mw_multigrid recon does not use --recon-rsd-solver (keep default)")

            # 条件分岐: `str(recon_los_shift) != str(ap.get_default("recon_los_shift"))` を満たす経路を評価する。

            if str(recon_los_shift) != str(ap.get_default("recon_los_shift")):
                raise SystemExit("mw_multigrid recon does not use --recon-los-shift (keep default)")

            # 条件分岐: `not math.isclose(float(recon_rsd_shift), float(ap.get_default("recon_rsd_shif...` を満たす経路を評価する。

            if not math.isclose(float(recon_rsd_shift), float(ap.get_default("recon_rsd_shift")), rel_tol=0.0, abs_tol=1e-12):
                raise SystemExit("mw_multigrid recon does not use --recon-rsd-shift (keep default)")

        # Auto-tag non-default recon parameters so that recon scans do not overwrite each other
        # when the user forgets to specify `--out-tag`.

        default_recon_grid = int(ap.get_default("recon_grid"))
        # 条件分岐: `int(recon_grid) != default_recon_grid` を満たす経路を評価する。
        if int(recon_grid) != default_recon_grid:
            out_tag = _append_out_tag(out_tag, f"g{int(recon_grid)}")

        default_recon_smoothing = float(ap.get_default("recon_smoothing"))
        # 条件分岐: `not math.isclose(float(recon_smoothing), default_recon_smoothing, rel_tol=0.0...` を満たす経路を評価する。
        if not math.isclose(float(recon_smoothing), default_recon_smoothing, rel_tol=0.0, abs_tol=1e-12):
            out_tag = _append_out_tag(out_tag, f"s{_float_token(float(recon_smoothing))}")

        default_recon_bias = float(ap.get_default("recon_bias"))
        # 条件分岐: `not math.isclose(float(recon_bias), default_recon_bias, rel_tol=0.0, abs_tol=...` を満たす経路を評価する。
        if not math.isclose(float(recon_bias), default_recon_bias, rel_tol=0.0, abs_tol=1e-12):
            out_tag = _append_out_tag(out_tag, f"bias{_float_token(float(recon_bias))}")

        default_recon_psi = int(ap.get_default("recon_psi_k_sign"))
        # 条件分岐: `int(recon_psi_k_sign) != default_recon_psi` を満たす経路を評価する。
        if int(recon_psi_k_sign) != default_recon_psi:
            out_tag = _append_out_tag(out_tag, "psi_pos" if int(recon_psi_k_sign) > 0 else "psi_neg")

        # 条件分岐: `recon_mode == "ani"` を満たす経路を評価する。

        if recon_mode == "ani":
            # 条件分岐: `recon_f is not None` を満たす経路を評価する。
            if recon_f is not None:
                out_tag = _append_out_tag(out_tag, f"f{_float_token(float(recon_f))}")

            default_rsd_solver = str(ap.get_default("recon_rsd_solver"))
            # 条件分岐: `str(recon_rsd_solver) != default_rsd_solver` を満たす経路を評価する。
            if str(recon_rsd_solver) != default_rsd_solver:
                out_tag = _append_out_tag(out_tag, "solvernone" if str(recon_rsd_solver) == "none" else f"solver{recon_rsd_solver}")

            default_los_shift = str(ap.get_default("recon_los_shift"))
            # 条件分岐: `str(recon_los_shift) != default_los_shift` を満たす経路を評価する。
            if str(recon_los_shift) != default_los_shift:
                out_tag = _append_out_tag(out_tag, "radial" if str(recon_los_shift) == "radial" else str(recon_los_shift))

            default_rsd_shift = float(ap.get_default("recon_rsd_shift"))
            # 条件分岐: `not math.isclose(float(recon_rsd_shift), default_rsd_shift, rel_tol=0.0, abs_...` を満たす経路を評価する。
            if not math.isclose(float(recon_rsd_shift), default_rsd_shift, rel_tol=0.0, abs_tol=1e-12):
                out_tag = _append_out_tag(out_tag, f"rsdshift{_float_token(float(recon_rsd_shift))}")

        default_pad_fraction = float(ap.get_default("recon_pad_fraction"))
        # 条件分岐: `not math.isclose(float(recon_pad_fraction), default_pad_fraction, rel_tol=0.0...` を満たす経路を評価する。
        if not math.isclose(float(recon_pad_fraction), default_pad_fraction, rel_tol=0.0, abs_tol=1e-12):
            out_tag = _append_out_tag(out_tag, f"pad{_float_token(float(recon_pad_fraction))}")

        # 条件分岐: `recon_mask_expected_frac is not None` を満たす経路を評価する。

        if recon_mask_expected_frac is not None:
            out_tag = _append_out_tag(out_tag, f"maskexp{_float_token(float(recon_mask_expected_frac))}")

        # 条件分岐: `recon_box_shape not in ("rect", "cube")` を満たす経路を評価する。

        if recon_box_shape not in ("rect", "cube"):
            raise SystemExit("--recon-box-shape must be rect/cube")

        # 条件分岐: `recon_box_shape != "rect"` を満たす経路を評価する。

        if recon_box_shape != "rect":
            suffix = _sanitize_out_tag(f"box_{recon_box_shape}")
            # 条件分岐: `suffix not in out_tag` を満たす経路を評価する。
            if suffix not in out_tag:
                out_tag = _sanitize_out_tag(f"{out_tag}_{suffix}") if out_tag else suffix

        # 条件分岐: `recon_box_frame not in ("raw", "pca")` を満たす経路を評価する。

        if recon_box_frame not in ("raw", "pca"):
            raise SystemExit("--recon-box-frame must be raw/pca")

        # 条件分岐: `recon_box_frame != "raw"` を満たす経路を評価する。

        if recon_box_frame != "raw":
            suffix = _sanitize_out_tag(f"frame_{recon_box_frame}")
            # 条件分岐: `suffix not in out_tag` を満たす経路を評価する。
            if suffix not in out_tag:
                out_tag = _sanitize_out_tag(f"{out_tag}_{suffix}") if out_tag else suffix

        # 条件分岐: `recon_assignment != "ngp"` を満たす経路を評価する。

        if recon_assignment != "ngp":
            suffix = _sanitize_out_tag(f"assign_{recon_assignment}")
            # 条件分岐: `suffix not in out_tag` を満たす経路を評価する。
            if suffix not in out_tag:
                out_tag = _sanitize_out_tag(f"{out_tag}_{suffix}") if out_tag else suffix
        # Ensure iso/ani recon outputs are distinct by default.
        # (If a user wants multiple variants, they can still add `--out-tag` and we will prefix it.)

        recon_tag = _sanitize_out_tag(
            f"recon_{recon}_{recon_mode}" + ("_mwrndrsd" if (recon == "mw_multigrid" and mw_random_rsd) else "")
        )
        # 条件分岐: `out_tag.startswith(recon_tag)` を満たす経路を評価する。
        if out_tag.startswith(recon_tag):
            out_tag = out_tag
        else:
            out_tag = _sanitize_out_tag(f"{recon_tag}_{out_tag}") if out_tag else recon_tag

    # 条件分岐: `dist == "lcdm"` を満たす経路を評価する。

    if dist == "lcdm":
        # 条件分岐: `not (lcdm_n_grid >= 10)` を満たす経路を評価する。
        if not (lcdm_n_grid >= 10):
            raise SystemExit("--lcdm-n-grid must be >= 10")

        # 条件分岐: `not (lcdm_z_grid_max > 0.0)` を満たす経路を評価する。

        if not (lcdm_z_grid_max > 0.0):
            raise SystemExit("--lcdm-z-grid-max must be > 0")

    # 条件分岐: `z_bin != "none"` を満たす経路を評価する。

    if z_bin != "none":
        z_map = {
            "b1": (0.2, 0.5),
            "b2": (0.4, 0.6),
            "b3": (0.5, 0.75),
        }
        zb_min, zb_max = z_map[z_bin]
        # 条件分岐: `z_min is None` を満たす経路を評価する。
        if z_min is None:
            z_min = zb_min

        # 条件分岐: `z_max is None` を満たす経路を評価する。

        if z_max is None:
            z_max = zb_max

    # 条件分岐: `(z_min is not None) and (z_max is not None) and not (float(z_min) < float(z_m...` を満たす経路を評価する。

    if (z_min is not None) and (z_max is not None) and not (float(z_min) < float(z_max)):
        raise SystemExit("--z-min must be < --z-max when both are provided")

    caps_to_use = ["north", "south"] if caps == "combined" else [caps]
    gal_paths = []
    rnd_paths = []
    sampling_methods: list[str] = []
    for cap in caps_to_use:
        key = f"{sample}:{cap}"
        it = manifest.get("items", {}).get(key, {})
        # 条件分岐: `"galaxy" not in it or "random" not in it` を満たす経路を評価する。
        if "galaxy" not in it or "random" not in it:
            raise SystemExit(f"missing galaxy/random for {key} in manifest (run fetch)")

        # 条件分岐: `str(it.get("random", {}).get("kind", "")).strip() != random_kind` を満たす経路を評価する。

        if str(it.get("random", {}).get("kind", "")).strip() != random_kind:
            mk = str(it.get("random", {}).get("kind", "")).strip()
            raise SystemExit(
                f"manifest random kind mismatch for {key}: requested {random_kind}, "
                f"but manifest has {mk!r}. "
                f"Either pass --random-kind {mk}, or re-run fetch to extract {random_kind}."
            )

        gal_paths.append(_resolve_manifest_path(it["galaxy"]["npz_path"]))
        rnd_paths.append(_resolve_manifest_path(it["random"]["npz_path"]))
        sampling_methods.append(str(it.get("random", {}).get("extract", {}).get("sampling", {}).get("method", "")))

    # 条件分岐: `match_sectors_arg == "on"` を満たす経路を評価する。

    if match_sectors_arg == "on":
        match_sectors = True
    # 条件分岐: 前段条件が不成立で、`match_sectors_arg == "off"` を追加評価する。
    elif match_sectors_arg == "off":
        match_sectors = False
    else:
        # auto: enable when inputs look subsampled (prefix_rows/cached/unknown)
        m = [s for s in sampling_methods if s]
        match_sectors = (not m) or any(s != "full" for s in m)

    # If f is not explicitly provided, fix it across NGC/SGC to avoid mixing definitions.

    if recon != "none" and recon_f is None and caps == "combined":
        try:
            num = 0.0
            den = 0.0
            for gal_p in gal_paths:
                gal_tmp = _load_npz(gal_p)
                z_tmp, _ = _select_redshift(gal_tmp, z_source=z_source)
                w_tmp, _ = _weights_galaxy(gal_tmp, scheme=weight_scheme)
                m0 = np.isfinite(z_tmp) & (z_tmp > 0.0) & np.isfinite(w_tmp)
                # 条件分岐: `z_min is not None` を満たす経路を評価する。
                if z_min is not None:
                    m0 = m0 & (z_tmp >= float(z_min))

                # 条件分岐: `z_max is not None` を満たす経路を評価する。

                if z_max is not None:
                    m0 = m0 & (z_tmp < float(z_max))

                z_use = np.asarray(z_tmp[m0], dtype=np.float64)
                w_use = np.asarray(w_tmp[m0], dtype=np.float64)
                num += float(np.sum(z_use * w_use))
                den += float(np.sum(w_use))

            # 条件分岐: `den > 0.0` を満たす経路を評価する。

            if den > 0.0:
                z_eff_all = num / den
                recon_f = float(_growth_rate_lcdm(float(omega_m), float(z_eff_all)))
                recon_f_source = f"lcdm_omega_m(z_eff_all={z_eff_all:.3f})^0.55"
        except Exception:
            # Fall back to per-cap z_eff inside reconstruction.
            pass

    out_dir = _ROOT / "output" / "private" / "cosmology"
    out_dir.mkdir(parents=True, exist_ok=True)
    s_bins_file, edges = _make_s_bins_file(out_dir, s_min=float(args.s_min), s_max=float(args.s_max), s_step=float(args.s_step))

    # 条件分岐: `caps != "combined"` を満たす経路を評価する。
    if caps != "combined":
        gal = _load_npz(gal_paths[0])
        rnd = _load_npz(rnd_paths[0])
        pack = _xi_multipoles_from_catalogs(
            gal=gal,
            rnd=rnd,
            weight_scheme=str(args.weight_scheme),
            recon_weight_scheme=recon_weight_scheme,
            dist_model=dist,
            omega_m=omega_m,
            z_source=z_source,
            los=los,
            lcdm_n_grid=lcdm_n_grid,
            lcdm_z_grid_max=lcdm_z_grid_max,
            z_min=z_min,
            z_max=z_max,
            recon_z_min=recon_z_min,
            recon_z_max=recon_z_max,
            s_bins_file=s_bins_file,
            edges=edges,
            mu_max=float(args.mu_max),
            nmu=int(args.nmu),
            nthreads=int(args.threads),
            paircounts_backend=paircounts_backend,
            match_sectors=bool(match_sectors),
            sector_key=sector_key,
            recon=recon,
            recon_grid=recon_grid,
            recon_smoothing=recon_smoothing,
            recon_assignment=recon_assignment,
            recon_bias=recon_bias,
            recon_psi_k_sign=recon_psi_k_sign,
            recon_f=recon_f,
            recon_f_source=recon_f_source,
            recon_mode=recon_mode,
            recon_rsd_solver=recon_rsd_solver,
            recon_los_shift=recon_los_shift,
            recon_rsd_shift=recon_rsd_shift,
            recon_pad_fraction=recon_pad_fraction,
            recon_mask_expected_frac=args.recon_mask_expected_frac,
            recon_box_shape=recon_box_shape,
            recon_box_frame=recon_box_frame,
            mw_random_rsd=bool(mw_random_rsd),
            mw_force_rebuild=bool(mw_force_rebuild),
        )
    else:
        cap_names = ["north", "south"]
        cap_packs = []
        for cap_name, gal_p, rnd_p in zip(cap_names, gal_paths, rnd_paths, strict=True):
            gal = _load_npz(gal_p)
            rnd = _load_npz(rnd_p)
            cap_pack = _xi_multipoles_from_catalogs(
                gal=gal,
                rnd=rnd,
                weight_scheme=str(args.weight_scheme),
                recon_weight_scheme=recon_weight_scheme,
                dist_model=dist,
                omega_m=omega_m,
                z_source=z_source,
                los=los,
                lcdm_n_grid=lcdm_n_grid,
                lcdm_z_grid_max=lcdm_z_grid_max,
                z_min=z_min,
                z_max=z_max,
                recon_z_min=recon_z_min,
                recon_z_max=recon_z_max,
                s_bins_file=s_bins_file,
                edges=edges,
                mu_max=float(args.mu_max),
                nmu=int(args.nmu),
                nthreads=int(args.threads),
                paircounts_backend=paircounts_backend,
                match_sectors=bool(match_sectors),
                sector_key=sector_key,
                recon=recon,
                recon_grid=recon_grid,
                recon_smoothing=recon_smoothing,
                recon_assignment=recon_assignment,
                recon_bias=recon_bias,
                recon_psi_k_sign=recon_psi_k_sign,
                recon_f=recon_f,
                recon_f_source=recon_f_source,
                recon_mode=recon_mode,
                recon_rsd_solver=recon_rsd_solver,
                recon_los_shift=recon_los_shift,
                recon_rsd_shift=recon_rsd_shift,
                recon_pad_fraction=recon_pad_fraction,
                recon_mask_expected_frac=args.recon_mask_expected_frac,
                recon_box_shape=recon_box_shape,
                recon_box_frame=recon_box_frame,
                mw_random_rsd=bool(mw_random_rsd),
                mw_force_rebuild=bool(mw_force_rebuild),
            )
            cap_pack["cap"] = cap_name
            cap_packs.append(cap_pack)

        # Coordinate spec must match across caps; allow per-cap runtime details (e.g. recon box) to differ.

        coord0_sig = _coordinate_spec_signature(cap_packs[0].get("coordinate_spec", {}))
        for p in cap_packs[1:]:
            # 条件分岐: `json.dumps(coord0_sig, sort_keys=True) != json.dumps(_coordinate_spec_signatu...` を満たす経路を評価する。
            if json.dumps(coord0_sig, sort_keys=True) != json.dumps(_coordinate_spec_signature(p.get("coordinate_spec", {})), sort_keys=True):
                raise SystemExit(
                    "coordinate_spec mismatch across caps (check --z-source/--los/--sector-key/--match-sectors/--lcdm-*/--recon-*/--recon-weight-scheme)"
                )

        # Combine by summing paircounts & normalizations per cap (robust for disconnected caps and subsampling).

        dd_w = sum(p["counts"]["DD_w"] for p in cap_packs)

        # If random density differs across caps due to subsampling, LS can be biased when aggregating.
        # Re-scale random weights per cap so that (sum_w_gal / sum_w_rnd) is consistent across caps.
        sum_w_gal_total = sum(float(p["totals"]["sum_w_gal"]) for p in cap_packs)
        sum_w_rnd_total = sum(float(p["totals"]["sum_w_rnd"]) for p in cap_packs)
        ratio_target = sum_w_gal_total / max(1e-30, sum_w_rnd_total)
        cap_scales = []
        dr_w = 0.0
        rr_w = 0.0
        ss_w: float | None = 0.0 if recon != "none" else None
        dr_tot = 0.0
        rr_tot = 0.0
        ss_tot: float | None = 0.0 if recon != "none" else None
        sum_w2_rnd_scaled = 0.0
        for p in cap_packs:
            sum_w_gal_i = float(p["totals"]["sum_w_gal"])
            sum_w_rnd_i = float(p["totals"]["sum_w_rnd"])
            # scale so that (sum_w_gal_i / (f * sum_w_rnd_i)) == ratio_target
            f = (sum_w_gal_i / max(1e-30, sum_w_rnd_i)) / max(1e-30, ratio_target)
            cap_scales.append({"cap": p["cap"], "scale_random_w": float(f)})

            dr_w = dr_w + float(f) * p["counts"]["DR_w"]
            rr_w = rr_w + float(f * f) * p["counts"]["RR_w"]
            dr_tot = dr_tot + float(f) * float(p["totals"]["dr_tot"])
            rr_tot = rr_tot + float(f * f) * float(p["totals"]["rr_tot"])
            sum_w2_rnd_scaled = sum_w2_rnd_scaled + float(f * f) * float(p["totals"]["sum_w2_rnd"])
            # 条件分岐: `ss_w is not None and ss_tot is not None` を満たす経路を評価する。
            if ss_w is not None and ss_tot is not None:
                # 条件分岐: `"SS_w" not in p.get("counts", {})` を満たす経路を評価する。
                if "SS_w" not in p.get("counts", {}):
                    raise SystemExit("recon enabled but missing SS_w in per-cap pack (re-run xi-from-catalogs)")

                # 条件分岐: `"ss_tot" not in p.get("totals", {})` を満たす経路を評価する。

                if "ss_tot" not in p.get("totals", {}):
                    raise SystemExit("recon enabled but missing ss_tot in per-cap totals (re-run xi-from-catalogs)")

                ss_w = ss_w + float(f * f) * p["counts"]["SS_w"]
                ss_tot = ss_tot + float(f * f) * float(p["totals"]["ss_tot"])

        dd_tot = sum(float(p["totals"]["dd_tot"]) for p in cap_packs)
        # 条件分岐: `recon == "none"` を満たす経路を評価する。
        if recon == "none":
            out_xi = _xi_multipoles_from_paircounts(
                dd_w=dd_w,
                dr_w=dr_w,
                rr_w=rr_w,
                edges=edges,
                mu_max=float(args.mu_max),
                nmu=int(args.nmu),
                dd_tot=dd_tot,
                dr_tot=dr_tot,
                rr_tot=rr_tot,
            )
        else:
            # 条件分岐: `ss_w is None or ss_tot is None` を満たす経路を評価する。
            if ss_w is None or ss_tot is None:
                raise SystemExit("internal error: recon enabled but SS counts missing")

            out_xi = _xi_multipoles_from_recon_paircounts(
                dd_w=dd_w,
                ds_w=dr_w,
                ss_w=ss_w,
                rr0_w=rr_w,
                edges=edges,
                mu_max=float(args.mu_max),
                nmu=int(args.nmu),
                dd_tot=dd_tot,
                ds_tot=dr_tot,
                ss_tot=float(ss_tot),
                rr0_tot=rr_tot,
            )

        sum_w_gal = sum_w_gal_total
        sum_w_rnd = sum_w_rnd_total
        z_eff_gal = sum(float(p["totals"]["sum_w_gal"]) * float(p["effective"]["z_eff_gal_weighted"]) for p in cap_packs) / max(
            1e-30, sum_w_gal_total
        )
        z_eff_rnd = sum(float(p["totals"]["sum_w_rnd"]) * float(p["effective"]["z_eff_rnd_weighted"]) for p in cap_packs) / max(
            1e-30, sum_w_rnd_total
        )

        coord0 = _combine_coordinate_spec_by_cap(cap_packs)
        pack = {
            "s": out_xi["s"],
            "xi0": out_xi["xi0"],
            "xi2": out_xi["xi2"],
            "xi_mu": out_xi.get("xi_mu"),
            "mu_edges": out_xi.get("mu_edges"),
            "counts": {"DD_w": dd_w, "DR_w": dr_w, "RR_w": rr_w, **({"SS_w": ss_w} if (ss_w is not None) else {})},
            "coordinate_spec": coord0,
            "totals": {
                "sum_w_gal": sum_w_gal,
                "sum_w2_gal": sum(float(p["totals"]["sum_w2_gal"]) for p in cap_packs),
                "sum_w_rnd": sum_w_rnd,
                "sum_w2_rnd": float(sum_w2_rnd_scaled),
                "dd_tot": dd_tot,
                "dr_tot": dr_tot,
                "rr_tot": rr_tot,
                **({"ss_tot": float(ss_tot)} if (ss_tot is not None) else {}),
            },
            "effective": {"z_eff_gal_weighted": z_eff_gal, "z_eff_rnd_weighted": z_eff_rnd},
            "sizes": {
                "n_gal": int(sum(int(p["sizes"]["n_gal"]) for p in cap_packs)),
                "n_rnd": int(sum(int(p["sizes"]["n_rnd"]) for p in cap_packs)),
            },
            "sectors": {
                "enabled": bool(any(p.get("sectors", {}).get("enabled", False) for p in cap_packs)),
                "by_cap": [{"cap": p["cap"], **p.get("sectors", {"enabled": False})} for p in cap_packs],
                "random_weight_rescale": cap_scales,
            },
        }

    tag = f"{sample}_{caps}_{dist}"
    ztag = _zcut_tag(z_bin=z_bin, z_min=z_min, z_max=z_max)
    # 条件分岐: `ztag` を満たす経路を評価する。
    if ztag:
        tag = f"{tag}_{ztag}"

    # 条件分岐: `out_tag` を満たす経路を評価する。

    if out_tag:
        tag = f"{tag}__{out_tag}"

    out_npz = out_dir / f"cosmology_bao_xi_from_catalogs_{tag}.npz"

    bao_peak = _estimate_bao_peak_s2_xi0(s=pack["s"], xi0=pack["xi0"])
    bao_feature_xi2 = _estimate_bao_feature_s2_xi(s=pack["s"], xi=pack["xi2"])
    try:
        s_ref = float(bao_peak.get("s_peak"))
        s_arr = np.asarray(pack["s"], dtype=float)
        xi2_arr = np.asarray(pack["xi2"], dtype=float)
        bao_feature_xi2["xi2_at_s_peak_xi0"] = float(np.interp(s_ref, s_arr, xi2_arr))
        bao_feature_xi2["s2_xi2_at_s_peak_xi0"] = float(np.interp(s_ref, s_arr, (s_arr * s_arr) * xi2_arr))
    except Exception:
        pass

    bao_wedges: Dict[str, Any] = {"enabled": False}
    xi_wedge_transverse: np.ndarray | None = None
    xi_wedge_radial: np.ndarray | None = None
    try:
        mu_max = float(args.mu_max)
        mu_split = float(args.mu_split)
        # 条件分岐: `not (0.0 < mu_split < mu_max)` を満たす経路を評価する。
        if not (0.0 < mu_split < mu_max):
            raise ValueError("--mu-split must satisfy 0 < mu_split < mu_max")

        # 条件分岐: `abs(mu_max - 1.0) > 1e-6` を満たす経路を評価する。

        if abs(mu_max - 1.0) > 1e-6:
            # The wedge approximation assumes standard multipoles defined over μ∈[0,1].
            # Keep disabled to avoid misinterpretation when μ is truncated.
            bao_wedges = {"enabled": False, "reason": f"mu_max!=1 (mu_max={mu_max:g})"}
        else:
            s_arr = np.asarray(pack["s"], dtype=float)

            # Prefer direct μ-integration from the full xi(s, μ) grid.
            xi_mu = pack.get("xi_mu", None)
            mu_edges = pack.get("mu_edges", None)
            # 条件分岐: `(xi_mu is not None) and (mu_edges is not None)` を満たす経路を評価する。
            if (xi_mu is not None) and (mu_edges is not None):
                xi_mu_arr = np.asarray(xi_mu, dtype=np.float64)
                mu_edges_arr = np.asarray(mu_edges, dtype=np.float64).reshape(-1)
                # 条件分岐: `mu_edges_arr.size < 2` を満たす経路を評価する。
                if mu_edges_arr.size < 2:
                    raise ValueError("invalid mu_edges")

                mu_mid = 0.5 * (mu_edges_arr[:-1] + mu_edges_arr[1:])
                dmu = float(mu_edges_arr[1] - mu_edges_arr[0])
                # 条件分岐: `not np.allclose(np.diff(mu_edges_arr), dmu, rtol=0, atol=1e-12)` を満たす経路を評価する。
                if not np.allclose(np.diff(mu_edges_arr), dmu, rtol=0, atol=1e-12):
                    raise ValueError("non-uniform mu_edges (unexpected)")

                # 条件分岐: `xi_mu_arr.shape != (s_arr.size, mu_mid.size)` を満たす経路を評価する。

                if xi_mu_arr.shape != (s_arr.size, mu_mid.size):
                    raise ValueError(f"xi_mu shape mismatch: {xi_mu_arr.shape} vs (nbins,nmu)=({s_arr.size},{mu_mid.size})")

                m_t = (mu_mid >= 0.0) & (mu_mid < mu_split)
                m_r = (mu_mid >= mu_split) & (mu_mid <= mu_max)
                # 条件分岐: `not (np.any(m_t) and np.any(m_r))` を満たす経路を評価する。
                if not (np.any(m_t) and np.any(m_r)):
                    raise ValueError("empty wedge mask")

                xi_wedge_transverse = (np.sum(xi_mu_arr[:, m_t] * dmu, axis=1) / float(mu_split - 0.0)).astype(np.float64)
                xi_wedge_radial = (np.sum(xi_mu_arr[:, m_r] * dmu, axis=1) / float(mu_max - mu_split)).astype(np.float64)
                wedge_method = "xi_mu_integral"
            else:
                # Fallback: approximate wedge using ℓ=0,2 only.
                xi0_arr = np.asarray(pack["xi0"], dtype=float)
                xi2_arr = np.asarray(pack["xi2"], dtype=float)
                xi_t, _ = _xi_wedge_from_multipoles(xi0=xi0_arr, xi2=xi2_arr, mu0=0.0, mu1=mu_split)
                xi_r, _ = _xi_wedge_from_multipoles(xi0=xi0_arr, xi2=xi2_arr, mu0=mu_split, mu1=mu_max)
                xi_wedge_transverse = np.asarray(xi_t, dtype=np.float64)
                xi_wedge_radial = np.asarray(xi_r, dtype=np.float64)
                wedge_method = "xi_l0l2_approx"

            def _p2_avg(mu0: float, mu1: float) -> float:
                # P2(μ) = (3μ^2-1)/2, ∫P2 dμ = (μ^3-μ)/2
                mu0 = float(mu0)
                mu1 = float(mu1)
                return (((mu1 * mu1 * mu1 - mu1) - (mu0 * mu0 * mu0 - mu0)) / (2.0 * (mu1 - mu0))) if (mu1 != mu0) else float("nan")

            c2_t = _p2_avg(0.0, mu_split)
            c2_r = _p2_avg(mu_split, mu_max)
            peak_t = _estimate_bao_peak_s2_xi(s=s_arr, xi=xi_wedge_transverse)
            peak_r = _estimate_bao_peak_s2_xi(s=s_arr, xi=xi_wedge_radial)
            s_t = float(peak_t.get("s_peak"))
            s_r = float(peak_r.get("s_peak"))

            # Reliability heuristics (Phase A): guard against edge-picking when the
            # wedge peak is ill-defined (common in low S/N subsamples).
            edge_margin = max(2.0 * float(args.s_step), 5.0)

            def _peak_near_edge(peak: Dict[str, Any]) -> bool:
                try:
                    w0, w1 = peak.get("search_range", [None, None])
                    # 条件分岐: `(w0 is None) or (w1 is None)` を満たす経路を評価する。
                    if (w0 is None) or (w1 is None):
                        return True

                    w0 = float(w0)
                    w1 = float(w1)
                    s_pk = float(peak.get("s_peak"))
                    return (s_pk - w0) < edge_margin or (w1 - s_pk) < edge_margin
                except Exception:
                    return True

            wedge_flags: list[str] = []
            # 条件分岐: `_peak_near_edge(peak_t)` を満たす経路を評価する。
            if _peak_near_edge(peak_t):
                wedge_flags.append("transverse_peak_near_edge")

            # 条件分岐: `_peak_near_edge(peak_r)` を満たす経路を評価する。

            if _peak_near_edge(peak_r):
                wedge_flags.append("radial_peak_near_edge")

            # 条件分岐: `bool(peak_t.get("is_window_edge", False))` を満たす経路を評価する。

            if bool(peak_t.get("is_window_edge", False)):
                wedge_flags.append("transverse_peak_at_window_edge")

            # 条件分岐: `bool(peak_r.get("is_window_edge", False))` を満たす経路を評価する。

            if bool(peak_r.get("is_window_edge", False)):
                wedge_flags.append("radial_peak_at_window_edge")

            reliable = len(wedge_flags) == 0

            bao_wedges = {
                "enabled": True,
                "method": wedge_method,
                "mu_split": mu_split,
                "reliable": reliable,
                "edge_margin": float(edge_margin),
                "flags": list(wedge_flags),
                "transverse": {"mu_range": [0.0, mu_split], "p2_avg": c2_t, "peak": peak_t},
                "radial": {"mu_range": [mu_split, mu_max], "p2_avg": c2_r, "peak": peak_r},
                "delta_s_peak": s_r - s_t,
                "ratio_s_peak": (s_r / s_t) if (s_t != 0.0) else float("nan"),
            }
    except Exception as e:
        bao_wedges = {"enabled": False, "reason": f"wedge_estimator_failed: {e}"}

    # Save outputs.
    # Keep backward compatible keys (s/xi0/xi2) and add optional diagnostic arrays for downstream fits.

    save_kwargs: Dict[str, Any] = {
        "s": np.asarray(pack["s"], dtype=float),
        "xi0": np.asarray(pack["xi0"], dtype=float),
        "xi2": np.asarray(pack["xi2"], dtype=float),
        # Always store μ-binning so later stages can propagate errors consistently.
        "mu_edges": np.asarray(pack.get("mu_edges", []), dtype=float),
        "mu_split": float(args.mu_split),
        # Pair counts (weighted sums). These stay small (nbins*nmu) and enable approximate error models.
        "dd_w": np.asarray(pack.get("counts", {}).get("DD_w", []), dtype=float),
        "dr_w": np.asarray(pack.get("counts", {}).get("DR_w", []), dtype=float),
        "rr_w": np.asarray(pack.get("counts", {}).get("RR_w", []), dtype=float),
        # Recon-only pair counts (optional): shifted-random auto counts.
        "ss_w": np.asarray(pack.get("counts", {}).get("SS_w", []), dtype=float),
        # Optional full ξ(s,μ) grid (nbins*nmu) for wedge/fit reuse.
        "xi_mu": np.asarray(pack.get("xi_mu", []), dtype=float),
        # Totals used for normalization (scalars).
        "dd_tot": float(pack.get("totals", {}).get("dd_tot", float("nan"))),
        "dr_tot": float(pack.get("totals", {}).get("dr_tot", float("nan"))),
        "rr_tot": float(pack.get("totals", {}).get("rr_tot", float("nan"))),
        "ss_tot": float(pack.get("totals", {}).get("ss_tot", float("nan"))),
    }
    # 条件分岐: `bao_wedges.get("enabled", False) and (xi_wedge_transverse is not None) and (x...` を満たす経路を評価する。
    if bao_wedges.get("enabled", False) and (xi_wedge_transverse is not None) and (xi_wedge_radial is not None):
        save_kwargs["xi_wedge_transverse"] = xi_wedge_transverse
        save_kwargs["xi_wedge_radial"] = xi_wedge_radial

    np.savez_compressed(out_npz, **save_kwargs)

    coord_sig = _coordinate_spec_signature(pack.get("coordinate_spec", {}) or {})
    coord_sig_hash = hashlib.sha256(
        json.dumps(coord_sig, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    estimator_spec: Dict[str, Any] = {
        "data_selection": {
            "random_kind": str(random_kind),
            "match_sectors": {"cli": str(match_sectors_arg), "enabled": bool(match_sectors), "sector_key": str(sector_key)},
        },
        "paircounts": {
            "backend": str(((pack.get("coordinate_spec", {}) or {}).get("los", {}) or {}).get("backend", "Corrfunc.DDsmu_mocks")),
            "weight_type": "pair_product",
            "pair_sum": "npairs*weightavg",
            "autocorr_convention": "ordered_pairs (i!=j)",
            "totals_convention": {
                "dd_tot": "sum_w^2 - sum_w2",
                "rr_tot": "sum_w^2 - sum_w2",
                "dr_tot": "sum_w_gal * sum_w_rnd",
                "note": "Matches Corrfunc autocorr ordered-pair counting so normalized counts behave like probabilities.",
            },
        },
        "bins": {
            "s": {"builder": "np.arange", "min": float(args.s_min), "max": float(args.s_max), "step": float(args.s_step)},
            "mu": {
                "builder": "np.linspace",
                "nmu": int(args.nmu),
                "mu_max": float(args.mu_max),
                "mu_centers": "midpoint",
            },
        },
        "xi_multipoles": {
            "mu_range": [0.0, float(args.mu_max)],
            "definition": "xi_l(s)=(2l+1)∫ xi(s,mu) P_l(mu) dmu  (0<=mu<=mu_max)",
            "discretization": "riemann_midpoint (uniform mu bins)",
            "p2": "P2(mu)=0.5*(3 mu^2 - 1)",
            "formula_discrete": {
                "xi0": "sum(xi*dmu)",
                "xi2": "5*sum(xi*P2(mu_mid)*dmu)",
            },
            "finite_policy": "xi_mu non-finite -> 0.0",
        },
        "combine_caps": {
            "caps": str(caps),
            "enabled": bool(caps == "combined"),
            "method": "sum_paircounts_then_project",
            "random_weight_rescale": {
                "enabled": bool(caps == "combined"),
                "policy": "rescale random weights per cap to match global (sum_w_gal/sum_w_rnd) before aggregating LS terms",
                "applies_to": ["DR", "RR", "SS"],
                "runtime_by_cap": pack.get("sectors", {}).get("random_weight_rescale", None),
            },
        },
    }

    est_sig = _estimator_spec_signature(estimator_spec)
    est_sig_hash = hashlib.sha256(json.dumps(est_sig, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()

    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "domain": "cosmology",
        "step": "16.4 (BAO primary statistic re-derivation from catalogs)",
        "inputs": {
            "manifest_json": str(manifest_path),
            "galaxy_npz": [str(p) for p in gal_paths],
            "random_npz": [str(p) for p in rnd_paths],
        },
        "params": {
            "sample": sample,
            "caps": caps,
            "distance_model": dist,
            "lcdm_omega_m": omega_m,
            "lcdm_n_grid": lcdm_n_grid,
            "lcdm_z_grid_max": lcdm_z_grid_max,
            "z_source": z_source,
            "los": los,
            "weight_scheme": weight_scheme,
            "random_kind": random_kind,
            "match_sectors": match_sectors_arg,
            "sector_key": sector_key,
            "recon": {
                "mode": recon,
                "weight_scheme": recon_weight_scheme if recon != "none" else None,
                "z_min_cli": (None if (recon == "none" or recon_z_min is None) else float(recon_z_min)),
                "z_max_cli": (None if (recon == "none" or recon_z_max is None) else float(recon_z_max)),
                "z_min_effective": (
                    None if (recon == "none") else (z_min if (recon_z_min is None) else float(recon_z_min))
                ),
                "z_max_effective": (
                    None if (recon == "none") else (z_max if (recon_z_max is None) else float(recon_z_max))
                ),
                "grid": recon_grid if recon != "none" else None,
                "smoothing_mpc_over_h": recon_smoothing if recon != "none" else None,
                "assignment": recon_assignment if recon != "none" else None,
                "mask_expected_frac": (
                    None
                    if recon == "none"
                    else (
                        float(recon_mask_expected_frac)
                        if (recon_mask_expected_frac is not None)
                        else (1.5 if (recon_assignment == "cic") else 0.0)
                    )
                ),
                "bias": recon_bias if recon != "none" else None,
                "psi_k_sign": recon_psi_k_sign if recon != "none" else None,
                "f": float(recon_f) if (recon_f is not None) else None,
                "shift_mode": recon_mode if recon != "none" else None,
                "rsd_solver": recon_rsd_solver if recon != "none" else None,
                "los_shift": recon_los_shift if recon != "none" else None,
                "rsd_shift": recon_rsd_shift if recon != "none" else None,
                "pad_fraction": recon_pad_fraction if recon != "none" else None,
                "box_shape": recon_box_shape if recon != "none" else None,
            },
            "out_tag": out_tag or None,
            "s_bins": {"min": float(args.s_min), "max": float(args.s_max), "step": float(args.s_step)},
            "z_cut": {"bin": z_bin, "z_min": z_min, "z_max": z_max},
            "mu_bins": {"nmu": int(args.nmu), "mu_max": float(args.mu_max), "mu_split": float(args.mu_split)},
            "corrfunc_threads": int(args.threads),
            "coordinate_spec": pack.get("coordinate_spec", {}),
            "coordinate_spec_signature": coord_sig,
            "coordinate_spec_hash": coord_sig_hash,
            "estimator_spec": estimator_spec,
            "estimator_spec_signature": est_sig,
            "estimator_spec_hash": est_sig_hash,
        },
        "derived": {
            "z_eff_gal_weighted": pack["effective"]["z_eff_gal_weighted"],
            "z_eff_rnd_weighted": pack["effective"]["z_eff_rnd_weighted"],
            "bao_peak": bao_peak,
            "bao_feature_xi2": bao_feature_xi2,
            "bao_wedges": bao_wedges,
        },
        "sizes": pack["sizes"],
        "sectors": pack.get("sectors", {"enabled": False}),
        "totals": pack["totals"],
        "outputs": {"npz": str(out_npz)},
        "notes": [],
    }

    def _random_sampling_desc(path: Path) -> str:
        name = str(path.name)
        # 条件分岐: `".prefix_" in name` を満たす経路を評価する。
        if ".prefix_" in name:
            try:
                token = name.split(".prefix_", 1)[1].split(".npz", 1)[0]
                n = int(token)
                return f"prefix_rows（先頭{n:,}行）"
            except Exception:
                return "prefix_rows"

        # 条件分岐: `".reservoir_" in name` を満たす経路を評価する。

        if ".reservoir_" in name:
            try:
                token = name.split(".reservoir_", 1)[1].split(".npz", 1)[0]
                # e.g. "2000000_seed0" or "2000000_seed0_covergal"
                parts = token.split("_")
                n = int(parts[0])
                seed = None
                for p in parts[1:]:
                    # 条件分岐: `p.startswith("seed")` を満たす経路を評価する。
                    if p.startswith("seed"):
                        try:
                            seed = int(p[4:])
                        except Exception:
                            seed = None

                        break

                # 条件分岐: `seed is None` を満たす経路を評価する。

                if seed is None:
                    return f"reservoir（{n:,}行）"

                return f"reservoir（{n:,}行; seed={seed}）"
            except Exception:
                return "reservoir"

        return "full（全量）"

    random_descs = sorted({_random_sampling_desc(p) for p in rnd_paths})
    random_desc = " / ".join(random_descs) if random_descs else "unknown"
    # 条件分岐: `any(d.startswith("prefix_rows") for d in random_descs)` を満たす経路を評価する。
    if any(d.startswith("prefix_rows") for d in random_descs):
        metrics["notes"].append(f"random catalog は fetch 時点で {random_desc} を使用（厳密な一様サンプルではない可能性）。")
    else:
        metrics["notes"].append(f"random catalog は fetch 時点で {random_desc} を使用。")

    sectors_enabled = bool((metrics.get("sectors") or {}).get("enabled"))
    # 条件分岐: `sectors_enabled` を満たす経路を評価する。
    if sectors_enabled:
        metrics["notes"].append("入力が部分抽出の場合、IPOLY/ISECT で共通セクタにそろえてから ξ を計算する（--match-sectors）。")
    else:
        metrics["notes"].append("フットプリント整合（--match-sectors）は適用なし（該当列が無い／全量入力など）。")

    metrics["notes"].append("recon は --recon で有効化できる（Phase B: 簡易Zel'dovich）。")
    out_json = out_dir / f"cosmology_bao_xi_from_catalogs_{tag}_metrics.json"
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    # Plot
    _set_japanese_font()
    import matplotlib.pyplot as plt

    s = np.asarray(pack["s"], dtype=float)
    xi0 = np.asarray(pack["xi0"], dtype=float)
    xi2 = np.asarray(pack["xi2"], dtype=float)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
    ax1.plot(s, (s * s) * xi0, color="#d62728", linewidth=2.0, label="ξ0")
    ax2.plot(s, (s * s) * xi2, color="#1f77b4", linewidth=2.0, label="ξ2")
    try:
        s_peak = float(metrics.get("derived", {}).get("bao_peak", {}).get("s_peak"))
        y0 = (s * s) * xi0
        ax1.axvline(s_peak, color="#888888", linewidth=1.2, linestyle="--")
        ax2.axvline(s_peak, color="#bbbbbb", linewidth=1.0, linestyle="--")
        ax1.text(
            s_peak + 1.0,
            float(np.nanmax(y0)),
            f"s_peak≈{s_peak:.1f}",
            fontsize=9,
            color="#444444",
            va="top",
        )
    except Exception:
        pass

    try:
        wedges = metrics.get("derived", {}).get("bao_wedges", {})
        # 条件分岐: `isinstance(wedges, dict) and wedges.get("enabled", False)` を満たす経路を評価する。
        if isinstance(wedges, dict) and wedges.get("enabled", False):
            s_t = float(wedges.get("transverse", {}).get("peak", {}).get("s_peak"))
            s_r = float(wedges.get("radial", {}).get("peak", {}).get("s_peak"))
            ax1.axvline(s_t, color="#2ca02c", linewidth=1.1, linestyle=":")
            ax1.axvline(s_r, color="#9467bd", linewidth=1.1, linestyle=":")
            ax1.text(
                s_t + 1.0,
                float(np.nanmin(y0)),
                f"s⊥≈{s_t:.1f}",
                fontsize=9,
                color="#2ca02c",
                va="bottom",
            )
            ax1.text(
                s_r + 1.0,
                float(np.nanmin(y0)) + 0.02 * float(np.nanmax(y0) - np.nanmin(y0)),
                f"s∥≈{s_r:.1f}",
                fontsize=9,
                color="#9467bd",
                va="bottom",
            )
    except Exception:
        pass

    try:
        bf2 = metrics.get("derived", {}).get("bao_feature_xi2", {})
        s2 = float(bf2.get("s_abs"))
        r2 = float(bf2.get("residual_abs"))
        y2 = (s * s) * xi2
        color = "#2ca02c" if r2 >= 0.0 else "#9467bd"
        sign = "+" if r2 >= 0.0 else "-"
        ax2.axvline(s2, color=color, linewidth=1.2, linestyle=":")
        ax2.text(
            s2 + 1.0,
            float(np.nanmax(y2)),
            f"s₂*≈{s2:.1f} ({sign})",
            fontsize=9,
            color=color,
            va="top",
        )
    except Exception:
        pass

    ax1.set_xlabel("s [Mpc/h]")
    ax2.set_xlabel("s [Mpc/h]")
    ax1.set_ylabel("s² ξ0")
    ax2.set_ylabel("s² ξ2")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax1.legend()
    ax2.legend()
    z_title = f", zcut={ztag}" if ztag else ""
    recon_title = f", recon={recon}" if recon != "none" else ""
    fig.suptitle(
        f"BOSS DR12v5 LSS：ξℓ（銀河+ランダム再計算） sample={sample}, caps={caps}, dist={dist}{recon_title}{z_title}",
        fontsize=13,
    )
    fig.text(
        0.5,
        0.01,
        f"z_eff(gal,w)={metrics['derived']['z_eff_gal_weighted']:.3f} / n_gal={metrics['sizes']['n_gal']:,} / n_rand={metrics['sizes']['n_rnd']:,}",
        ha="center",
        fontsize=10,
    )
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.93))
    out_png = out_dir / f"cosmology_bao_xi_from_catalogs_{tag}.png"
    fig.savefig(out_png, dpi=200)
    plt.close(fig)

    metrics["outputs"]["png"] = str(out_png)
    out_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[ok] npz : {out_npz}")
    print(f"[ok] png : {out_png}")
    print(f"[ok] json: {out_json}")

    try:
        worklog.append_event(
            {
                "event_type": "cosmology_bao_xi_from_catalogs",
                "argv": sys.argv,
                "metrics": metrics["params"],
                "outputs": {"npz": out_npz, "png": out_png, "metrics_json": out_json},
            }
        )
    except Exception:
        pass

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
