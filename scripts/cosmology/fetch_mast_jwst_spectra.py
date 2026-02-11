#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_mast_jwst_spectra.py

Phase 4（宇宙論）/ Step 4.6（JWST/MAST：スペクトル一次データ）:
MAST（JWST）から x1d（1D spectrum）を取得・キャッシュし、最小のQC図を生成する。

狙い：
- 距離指標（ΛCDM距離など）の二次産物ではなく、一次データ（スペクトル）をローカルに保存し、
  「赤方偏移 z はスペクトルの輝線/吸収線のズレから直接決まる」入口を再現可能にする。
- 本スクリプトは「取得と可視化の土台」を提供し、z推定ロジック（線同定等）は Step 4.6.3 で拡張する。

データソース：
- MAST API（invoke）：https://mast.stsci.edu/api/v0/invoke
- Download（file）：https://mast.stsci.edu/api/v0.1/Download/file?uri=...

出力（固定）：
- data/cosmology/mast/jwst_spectra/<target_slug>/
    - manifest.json（取得条件・観測一覧・選択プロダクト・sha256）
    - raw/（downloaded FITS/PNG 等）
- output/cosmology/
    - jwst_spectra__<target_slug>__x1d_qc.png（取得済みx1dをまとめたQC）

注意：
- 取得対象の一部は proprietary 期間により 401（Unauthorized）になる場合がある。
  その場合も manifest に「見つかったが取得不能」で記録し、議論がブレないようにする。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover - optional dependency for offline/local runs
    requests = None

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.cosmology.boss_dr12v5_fits import read_bintable_columns, read_first_bintable_layout  # noqa: E402
from scripts.summary import worklog  # noqa: E402


_INVOKE_URL = "https://mast.stsci.edu/api/v0/invoke"
_DOWNLOAD_URL = "https://mast.stsci.edu/api/v0.1/Download/file"
_REQ_TIMEOUT = (30, 600)  # (connect, read)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mjd_to_utc_iso(mjd: float) -> str:
    # MJD epoch: 1858-11-17 00:00:00 UTC
    epoch = datetime(1858, 11, 17, tzinfo=timezone.utc)
    return (epoch + timedelta(days=float(mjd))).isoformat()


def _utc_to_mjd(dt: datetime) -> float:
    epoch = datetime(1858, 11, 17, tzinfo=timezone.utc)
    return (dt - epoch).total_seconds() / 86400.0


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _relpath(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    try:
        return path.relative_to(_ROOT).as_posix()
    except Exception:
        return path.as_posix()


def _slugify(s: str) -> str:
    s = (s or "").strip().lower()
    out = []
    for ch in s:
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out).strip("_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "target"


def _try_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _summarize_z_estimate(z: Dict[str, Any], out_json: Path) -> Dict[str, Any]:
    best_in = z.get("best") if isinstance(z.get("best"), dict) else {}
    best: Dict[str, Any] = {}
    for k in ("z", "score", "n_matches", "z_mean_from_matches", "z_std_from_matches"):
        if k in best_in and best_in.get(k) is not None:
            best[k] = best_in.get(k)

    out: Dict[str, Any] = {
        "ok": bool(z.get("ok")),
        "reason": z.get("reason"),
        "generated_utc": z.get("generated_utc"),
        "path": _relpath(out_json),
        "best": best if best else None,
    }
    # Avoid embedding large payloads (peaks/top/spectra etc.) into manifests.
    return out


def _summarize_z_confirmed(z: Dict[str, Any], out_json: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "ok": bool(z.get("ok")),
        "reason": z.get("reason"),
        "generated_utc": z.get("generated_utc"),
        "path": _relpath(out_json),
        "z": z.get("z"),
        "z_total_sigma": z.get("z_total_sigma"),
        "chi2_dof": z.get("chi2_dof"),
        "ok_spectra_n": z.get("ok_spectra_n"),
    }
    return out


def _load_z_estimate_summary(out_json: Path) -> Optional[Dict[str, Any]]:
    if not out_json.exists():
        return None
    obj = _try_read_json(out_json)
    if obj is None:
        return None
    return _summarize_z_estimate(obj, out_json)


def _load_z_confirmed_summary(out_json: Path) -> Optional[Dict[str, Any]]:
    if not out_json.exists():
        return None
    obj = _try_read_json(out_json)
    if obj is None:
        return None
    return _summarize_z_confirmed(obj, out_json)


def _mast_invoke(request_obj: Dict[str, Any]) -> Dict[str, Any]:
    if requests is None:
        raise RuntimeError("requests is required for online MAST queries. Install it or run with --offline.")
    r = requests.post(_INVOKE_URL, data={"request": json.dumps(request_obj)}, timeout=_REQ_TIMEOUT)
    r.raise_for_status()
    try:
        return r.json()
    except Exception as e:
        raise RuntimeError(f"MAST returned non-JSON response: {e}") from e


def _mast_download(uri: str, dst: Path) -> Dict[str, Any]:
    if requests is None:
        raise RuntimeError("requests is required to download MAST products. Install it or place files under data/.../raw/")
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        with requests.get(_DOWNLOAD_URL, params={"uri": uri}, stream=True, timeout=_REQ_TIMEOUT) as r:
            # proprietary or auth-required products may return 401/403
            if int(r.status_code) in (401, 403):
                return {
                    "ok": False,
                    "status_code": int(r.status_code),
                    "reason": "unauthorized (likely proprietary period)",
                }
            if int(r.status_code) == 404:
                return {"ok": False, "status_code": 404, "reason": "not_found"}
            if int(r.status_code) >= 500:
                return {"ok": False, "status_code": int(r.status_code), "reason": f"http_{int(r.status_code)}"}
            r.raise_for_status()
            with tmp.open("wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
        tmp.replace(dst)
        return {"ok": True, "bytes": int(dst.stat().st_size), "sha256": _sha256(dst)}
    except Exception as e:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return {"ok": False, "status_code": None, "reason": f"exception: {type(e).__name__}: {e}"}


def _query_jwst_obs_by_target(
    target_name: str,
    *,
    obs_collection: str,
    extra_filters: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    filters: List[Dict[str, Any]] = [
        {"paramName": "obs_collection", "values": [obs_collection]},
        {"paramName": "target_name", "values": [target_name]},
    ]
    if extra_filters:
        for f in extra_filters:
            if isinstance(f, dict) and f.get("paramName") and f.get("values"):
                filters.append(f)
    req = {
        "service": "Mast.Caom.Filtered",
        "params": {
            "columns": ",".join(
                [
                    "obsid",
                    "obs_id",
                    "obs_collection",
                    "target_name",
                    "dataproduct_type",
                    "calib_level",
                    "instrument_name",
                    "filters",
                    "proposal_id",
                    "proposal_pi",
                    "t_min",
                    "t_max",
                    "t_exptime",
                    "t_obs_release",
                    "s_ra",
                    "s_dec",
                    "obs_title",
                ]
            ),
            "filters": filters,
        },
        "format": "json",
        "pagesize": 200,
        "page": 1,
    }
    obj = _mast_invoke(req)
    rows = obj.get("data") or []
    return [r for r in rows if isinstance(r, dict)]


def _query_jwst_obs_by_box(
    *,
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    obs_collection: str,
    extra_filters: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    d = float(radius_deg)
    # CAOM stores s_ra in degrees (0..360). For small radii, a simple box is sufficient.
    ra_min = float(ra_deg) - d
    ra_max = float(ra_deg) + d
    dec_min = float(dec_deg) - d
    dec_max = float(dec_deg) + d
    filters: List[Dict[str, Any]] = [
        {"paramName": "obs_collection", "values": [obs_collection]},
        {"paramName": "s_ra", "values": [{"min": ra_min, "max": ra_max}]},
        {"paramName": "s_dec", "values": [{"min": dec_min, "max": dec_max}]},
    ]
    if extra_filters:
        for f in extra_filters:
            if isinstance(f, dict) and f.get("paramName") and f.get("values"):
                filters.append(f)

    req = {
        "service": "Mast.Caom.Filtered",
        "params": {
            "columns": ",".join(
                [
                    "obsid",
                    "obs_id",
                    "obs_collection",
                    "target_name",
                    "dataproduct_type",
                    "calib_level",
                    "instrument_name",
                    "filters",
                    "proposal_id",
                    "proposal_pi",
                    "t_min",
                    "t_max",
                    "t_exptime",
                    "t_obs_release",
                    "s_ra",
                    "s_dec",
                    "obs_title",
                ]
            ),
            "filters": filters,
        },
        "format": "json",
        "pagesize": 200,
        "page": 1,
    }
    obj = _mast_invoke(req)
    rows = obj.get("data") or []
    return [r for r in rows if isinstance(r, dict)]


def _read_target_config(path: Path) -> Dict[str, Any]:
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(obj, dict):
        return obj
    return {}


def _query_jwst_obs_with_fallback(
    target_name: str,
    *,
    obs_collection: str,
    cfg: Optional[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Query JWST observations for a target using a small, reproducible fallback chain.

    Fallback order:
      1) exact target_name (or cfg.mast_target_name / cfg.aliases)
      2) position box query (cfg.position.ra_deg/dec_deg/radius_deg) with cfg.extra_filters

    Returns (rows, query_info) where query_info is stored in manifest.json for traceability.
    """
    names: List[str] = []
    if cfg and isinstance(cfg, dict):
        mt = cfg.get("mast_target_name")
        if isinstance(mt, str) and mt.strip():
            names.append(mt.strip())
        aliases = cfg.get("aliases")
        if isinstance(aliases, list):
            for a in aliases:
                if isinstance(a, str) and a.strip():
                    names.append(a.strip())

    if target_name not in names:
        names.insert(0, target_name)

    seen: set[str] = set()
    names_u: List[str] = []
    for n in names:
        if n in seen:
            continue
        seen.add(n)
        names_u.append(n)

    extra_filters: List[Dict[str, Any]] = []
    if cfg and isinstance(cfg, dict):
        ef = cfg.get("extra_filters")
        if isinstance(ef, list):
            extra_filters = [f for f in ef if isinstance(f, dict)]

    # Name-based query: try all names and union observations (some targets use multiple MAST target_name values).
    rows_union: List[Dict[str, Any]] = []
    per_name: List[Dict[str, Any]] = []
    for n in names_u:
        try:
            rows = _query_jwst_obs_by_target(n, obs_collection=obs_collection, extra_filters=extra_filters)
            per_name.append(
                {"target_name": n, "ok": True, "n_obs": int(len(rows)), "extra_filters": extra_filters if extra_filters else None}
            )
        except Exception as e:
            per_name.append({"target_name": n, "ok": False, "reason": f"exception: {e}"})
            continue
        for r in rows:
            if isinstance(r, dict):
                rows_union.append(r)

    # De-duplicate by obsid (fallback to obs_id string).
    uniq: Dict[str, Dict[str, Any]] = {}
    for r in rows_union:
        if not isinstance(r, dict):
            continue
        key = None
        if r.get("obsid") is not None:
            key = f"obsid:{r.get('obsid')}"
        elif r.get("obs_id"):
            key = f"obs_id:{r.get('obs_id')}"
        else:
            continue
        if key not in uniq:
            uniq[key] = r

    if uniq:
        return (
            list(uniq.values()),
            {
                "method": "target_name_union",
                "ok": True,
                "names_tried": per_name,
                "n_obs": int(len(uniq)),
            },
        )

    # Position-based fallback (optional).
    if cfg and isinstance(cfg, dict):
        pos = cfg.get("position")
        if isinstance(pos, dict):
            ra = pos.get("ra_deg")
            dec = pos.get("dec_deg")
            rad = pos.get("radius_deg", 0.02)
            if isinstance(ra, (int, float)) and isinstance(dec, (int, float)) and isinstance(rad, (int, float)) and float(rad) > 0:
                extra_filters: List[Dict[str, Any]] = []
                ef = cfg.get("extra_filters")
                if isinstance(ef, list):
                    extra_filters = [f for f in ef if isinstance(f, dict)]
                try:
                    rows = _query_jwst_obs_by_box(
                        ra_deg=float(ra),
                        dec_deg=float(dec),
                        radius_deg=float(rad),
                        obs_collection=obs_collection,
                        extra_filters=extra_filters,
                    )
                except Exception as e:
                    return [], {"method": "position_box", "position": pos, "ok": False, "reason": f"exception: {e}"}
                if rows:
                    return rows, {"method": "position_box", "position": pos, "ok": True, "n_obs": int(len(rows))}
                return [], {"method": "position_box", "position": pos, "ok": False, "reason": "no_obs_found"}

    return [], {"method": "target_name", "target_name": target_name, "ok": False, "reason": "no_obs_found"}


def _query_products_for_obsid(obsid: int) -> List[Dict[str, Any]]:
    req = {"service": "Mast.Caom.Products", "params": {"obsid": int(obsid)}, "format": "json"}
    obj = _mast_invoke(req)
    rows = obj.get("data") or []
    return [r for r in rows if isinstance(r, dict)]


def _select_x1d_products(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Return (x1d_fits, x1d_png_previews)

    JWST x1d science files typically have:
      - productSubGroupDescription == "X1D"
      - productFilename endswith "_x1d.fits"
    """
    x1d_fits: List[Dict[str, Any]] = []
    previews: List[Dict[str, Any]] = []
    for r in rows:
        fn = str(r.get("productFilename") or "")
        sub = str(r.get("productSubGroupDescription") or "")
        ptype = str(r.get("productType") or "")
        fn_l = fn.lower()
        if fn_l.endswith(".fits") and (fn_l.endswith("_x1d.fits") or sub.upper() == "X1D"):
            x1d_fits.append(r)
            continue
        if fn_l.endswith("_x1d.png") and ptype.upper() == "PREVIEW":
            previews.append(r)
    return x1d_fits, previews


def _read_x1d_spectrum(fits_path: Path) -> Dict[str, np.ndarray]:
    with fits_path.open("rb") as f:
        layout = read_first_bintable_layout(f)
        want = [c for c in ["WAVELENGTH", "FLUX", "FLUX_ERROR", "SURF_BRIGHT", "SB_ERROR"] if c in layout.columns]
        if "WAVELENGTH" not in want:
            raise ValueError(f"x1d missing WAVELENGTH column: {fits_path}")
        cols = read_bintable_columns(f, layout=layout, columns=want, max_rows=None)
    return cols


_REST_LINES_UM: List[Tuple[str, float]] = [
    # UV
    ("Lyα", 0.121567),
    ("C IV", 0.1549),
    ("He II", 0.1640),
    ("C III]", 0.1909),
    # Optical (rest-frame)
    ("[O II]", 0.3727),
    ("[Ne III]", 0.3869),
    ("Hβ", 0.4861),
    ("[O III]4959", 0.4959),
    ("[O III]5007", 0.5007),
    ("Hα", 0.6563),
    ("[N II]", 0.6583),
    ("[S II]6716", 0.6716),
    ("[S II]6731", 0.6731),
]


def _moving_average(y: np.ndarray, win: int) -> np.ndarray:
    win_i = int(win)
    if win_i <= 1:
        return np.asarray(y, dtype=np.float64)
    k = np.ones(win_i, dtype=np.float64) / float(win_i)
    return np.convolve(np.asarray(y, dtype=np.float64), k, mode="same")


def _robust_sigma(y: np.ndarray) -> float:
    yv = np.asarray(y, dtype=np.float64)
    yv = yv[np.isfinite(yv)]
    if yv.size <= 8:
        return float(np.nanstd(yv)) if yv.size else float("nan")
    med = float(np.nanmedian(yv))
    mad = float(np.nanmedian(np.abs(yv - med)))
    if not np.isfinite(mad) or mad <= 0:
        return float(np.nanstd(yv))
    return 1.4826 * mad


def _detect_emission_peaks(
    w_um: np.ndarray,
    y: np.ndarray,
    yerr: Optional[np.ndarray],
    *,
    snr_threshold: float = 6.0,
    snr_threshold_fluxerr: float = 0.0,
    max_peaks: int = 20,
) -> Dict[str, Any]:
    w = np.asarray(w_um, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    m = np.isfinite(w) & np.isfinite(yv)
    if yerr is not None:
        ev = np.asarray(yerr, dtype=np.float64)
        m = m & np.isfinite(ev) & (ev > 0)
    if int(m.sum()) <= 10:
        return {"ok": False, "reason": "too_few_points"}

    w = w[m]
    yv = yv[m]
    ev = np.asarray(yerr, dtype=np.float64)[m] if yerr is not None else None

    # High-pass filter: (short smooth) - (long smooth) to isolate lines from continuum.
    y_s = _moving_average(yv, 7)
    y_base = _moving_average(y_s, max(31, int(round(len(y_s) / 15))))
    y_hp = y_s - y_base
    sigma = _robust_sigma(y_hp)
    snr = y_hp / sigma if np.isfinite(sigma) and sigma > 0 else np.full_like(y_hp, np.nan)
    snr_fluxerr = (y_hp / ev) if ev is not None else None

    # local maxima
    if y_hp.size < 3:
        return {"ok": False, "reason": "too_few_points"}
    core = (y_hp[1:-1] > y_hp[:-2]) & (y_hp[1:-1] >= y_hp[2:]) & np.isfinite(snr[1:-1])
    core = core & (snr[1:-1] >= float(snr_threshold)) & (y_hp[1:-1] > 0)
    if snr_fluxerr is not None and float(snr_threshold_fluxerr) > 0:
        core = core & (snr_fluxerr[1:-1] >= float(snr_threshold_fluxerr))
    idx = np.where(core)[0] + 1
    if idx.size <= 0:
        return {
            "ok": True,
            "peaks": [],
            "sigma_proxy": sigma,
            "snr_threshold": float(snr_threshold),
            "snr_threshold_fluxerr": float(snr_threshold_fluxerr),
            "n_points": int(w.size),
            "w_min_um": float(np.nanmin(w)),
            "w_max_um": float(np.nanmax(w)),
        }

    order = np.argsort(snr[idx])[::-1]
    idx = idx[order[: int(max_peaks)]]
    peaks = [
        {
            "w_um": float(w[i]),
            "snr": float(snr[i]),
            "snr_fluxerr": float(snr_fluxerr[i]) if snr_fluxerr is not None else None,
            "y": float(yv[i]),
            "yerr": float(ev[i]) if ev is not None else None,
        }
        for i in idx
    ]
    peaks = sorted(peaks, key=lambda r: r["w_um"])
    return {
        "ok": True,
        "peaks": peaks,
        "sigma_proxy": sigma,
        "snr_threshold": float(snr_threshold),
        "snr_threshold_fluxerr": float(snr_threshold_fluxerr),
        "n_points": int(w.size),
        "w_min_um": float(np.nanmin(w)),
        "w_max_um": float(np.nanmax(w)),
    }


def _score_redshift_candidates(
    peaks: List[Dict[str, Any]],
    *,
    w_min_um: float,
    w_max_um: float,
    z_min: float,
    z_max: float,
    tol_rel: float = 0.003,
    tol_abs_um: float = 0.004,
    top_n: int = 12,
) -> Dict[str, Any]:
    if not peaks:
        return {"ok": False, "reason": "no_peaks"}

    peak_w = np.array([float(p["w_um"]) for p in peaks], dtype=np.float64)
    peak_snr = np.array([float(p.get("snr") or 0.0) for p in peaks], dtype=np.float64)

    def nearest_peak_candidates(pred_um: float) -> List[int]:
        j = int(np.searchsorted(peak_w, pred_um))
        candidates: List[int] = []
        if 0 <= j < peak_w.size:
            candidates.append(j)
        if 0 <= j - 1 < peak_w.size:
            candidates.append(j - 1)
        if 0 <= j + 1 < peak_w.size:
            candidates.append(j + 1)
        # sort by distance and keep unique indices
        uniq: List[int] = []
        for k in sorted(set(candidates), key=lambda kk: abs(float(peak_w[kk]) - float(pred_um))):
            d = abs(float(peak_w[k]) - float(pred_um))
            tol = max(float(tol_abs_um), float(tol_rel) * float(pred_um))
            if d <= tol:
                uniq.append(int(k))
        return uniq

    # Build candidate set from (peak, rest_line) pairs.
    cand_map: Dict[str, Dict[str, Any]] = {}
    for i in range(int(peak_w.size)):
        wobs = float(peak_w[i])
        for name, lam0 in _REST_LINES_UM:
            if lam0 <= 0:
                continue
            z = wobs / float(lam0) - 1.0
            if not (float(z_min) <= z <= float(z_max)):
                continue
            key = f"{z:.5f}"
            rec = cand_map.get(key)
            if rec is None:
                rec = {"z": float(z), "seed": {"peak_w_um": wobs, "line": name, "line_um": float(lam0)}}
                cand_map[key] = rec

    candidates: List[Dict[str, Any]] = []
    for rec in cand_map.values():
        z = float(rec["z"])
        matches: List[Dict[str, Any]] = []
        score = 0.0
        used_peaks: set[int] = set()
        for name, lam0 in _REST_LINES_UM:
            pred = float(lam0) * (1.0 + z)
            if pred < float(w_min_um) or pred > float(w_max_um):
                continue
            j = None
            for cand in nearest_peak_candidates(pred):
                if int(cand) in used_peaks:
                    continue
                j = int(cand)
                break
            if j is None:
                continue
            used_peaks.add(int(j))
            d = float(peak_w[j]) - pred
            sn = float(peak_snr[j])
            matches.append(
                {
                    "line": name,
                    "line_um": float(lam0),
                    "pred_um": float(pred),
                    "peak_um": float(peak_w[j]),
                    "delta_um": float(d),
                    "peak_snr": sn,
                }
            )
            score += max(0.0, sn)
        matches = sorted(matches, key=lambda r: abs(float(r["delta_um"])))
        if not matches:
            continue
        z_vals = [float(m["peak_um"]) / float(m["line_um"]) - 1.0 for m in matches]
        z_mean = float(np.mean(z_vals))
        z_std = float(np.std(z_vals, ddof=1)) if len(z_vals) >= 2 else 0.0
        candidates.append(
            {
                "z": float(z),
                "score": float(score),
                "n_matches": int(len(matches)),
                "z_mean_from_matches": z_mean,
                "z_std_from_matches": z_std,
                "matches": matches,
            }
        )

    if not candidates:
        return {"ok": False, "reason": "no_candidates_scored"}

    candidates = sorted(candidates, key=lambda r: (float(r["score"]), float(r["n_matches"])), reverse=True)
    return {
        "ok": True,
        "best": candidates[0],
        "top": candidates[: int(top_n)],
        "params": {
            "z_min": float(z_min),
            "z_max": float(z_max),
            "tol_rel": float(tol_rel),
            "tol_abs_um": float(tol_abs_um),
        },
    }


def _plot_target_z_diagnostic(
    target_slug: str,
    *,
    spectrum_path: Path,
    peaks: List[Dict[str, Any]],
    best: Optional[Dict[str, Any]],
    out_png: Path,
) -> Optional[Dict[str, Any]]:
    if plt is None:
        return {"ok": False, "reason": "matplotlib not available"}
    try:
        cols = _read_x1d_spectrum(spectrum_path)
        w = cols["WAVELENGTH"]
        y = cols.get("FLUX")
        if y is None:
            y = cols.get("SURF_BRIGHT")
        if y is None:
            return {"ok": False, "reason": "missing FLUX/SURF_BRIGHT"}
        yerr = cols.get("FLUX_ERROR")
        if yerr is None:
            yerr = cols.get("SB_ERROR")
    except Exception as e:
        return {"ok": False, "reason": f"failed to read spectrum: {e}"}

    m = np.isfinite(w) & np.isfinite(y)
    if yerr is not None:
        m = m & np.isfinite(yerr)
    w = np.asarray(w[m], dtype=np.float64)
    y = np.asarray(y[m], dtype=np.float64)
    yerr_v = np.asarray(yerr[m], dtype=np.float64) if yerr is not None else None

    y_s = _moving_average(y, 7)
    y_base = _moving_average(y_s, max(31, int(round(len(y_s) / 15))))
    y_hp = y_s - y_base
    sigma = _robust_sigma(y_hp)
    snr = y_hp / sigma if np.isfinite(sigma) and sigma > 0 else np.full_like(y_hp, np.nan)
    snr_label = "SNR proxy (HP / robust_sigma)"

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax0, ax1 = axes
    ax0.plot(w, y, lw=0.8, alpha=0.7, label="raw")
    ax0.plot(w, y_s, lw=1.0, alpha=0.9, label="smooth")
    ax0.set_ylabel("flux (raw)")
    ax0.grid(True, alpha=0.25)
    ax0.legend(fontsize=8, loc="best")

    ax1.plot(w, snr, lw=0.9, alpha=0.9, color="#333333")
    ax1.axhline(0.0, color="#888888", lw=0.8)
    ax1.set_ylabel(snr_label)
    ax1.grid(True, alpha=0.25)

    # Peaks
    for p in peaks:
        x = float(p["w_um"])
        ax0.axvline(x, color="#cc0000", lw=0.6, alpha=0.25)
        ax1.plot([x], [0.0], marker="x", color="#cc0000", ms=6, mew=1.2)

    # Best z predicted lines (matched only)
    title = f"JWST x1d z diagnostic : {target_slug}"
    if best is not None:
        z = float(best.get("z_mean_from_matches") or best.get("z") or float("nan"))
        title += f" (best z≈{z:.3f}; matches={int(best.get('n_matches') or 0)})"
        for mm in best.get("matches") or []:
            x = float(mm["pred_um"])
            line = str(mm.get("line") or "")
            ax0.axvline(x, color="#1f77b4", lw=0.9, alpha=0.55)
            ax1.axvline(x, color="#1f77b4", lw=0.9, alpha=0.55)
            if line:
                ax0.text(x, float(np.nanmax(y)), line, rotation=90, va="top", ha="right", fontsize=7, alpha=0.7)

    ax0.set_title(title)
    ax1.set_xlabel("wavelength [µm]")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return {"ok": True, "out_png": _relpath(out_png), "spectrum": _relpath(spectrum_path)}


def _estimate_target_redshift(
    target_slug: str,
    *,
    x1d_paths: List[Path],
    out_dir: Path,
    snr_threshold: float = 6.0,
    snr_threshold_fluxerr: float = 0.0,
    z_min: float = 0.0,
    z_max: float = 25.0,
) -> Dict[str, Any]:
    out_json = out_dir / f"jwst_spectra__{target_slug}__z_estimate.json"
    out_csv = out_dir / f"jwst_spectra__{target_slug}__z_estimate.csv"
    out_png = out_dir / f"jwst_spectra__{target_slug}__z_diagnostic.png"

    rec: Dict[str, Any] = {
        "generated_utc": _utc_now(),
        "target_slug": target_slug,
        "ok": False,
        "reason": None,
        "spectrum_used": None,
        "w_range_um": None,
        "peak_detection": {"snr_threshold": float(snr_threshold), "snr_threshold_fluxerr": float(snr_threshold_fluxerr)},
        "rest_lines_um": [{"line": n, "um": float(l)} for n, l in _REST_LINES_UM],
        "best": None,
        "top": None,
        "plot": None,
    }

    if not x1d_paths:
        rec["reason"] = "no_local_x1d"
        out_json.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return rec

    # Choose a representative spectrum.
    #
    # Previous heuristic: prefer non-"mirimage", then maximize #peaks/SNR/span.
    # Problem: some x1d (esp. long-λ MIRI) can yield many spurious peaks that are
    # incompatible with the current (rest-line list, z-range), causing scoring to fail.
    #
    # New heuristic: prefer spectra that actually yield redshift candidates (score ok),
    # and only then use peaks/SNR/span as tiebreakers. Still prefers non-"mirimage" when possible.
    tried: List[Dict[str, Any]] = []
    cand: List[Dict[str, Any]] = []
    for p in x1d_paths:
        try:
            not_mirimage = "mirimage" not in p.name.lower()
            cols = _read_x1d_spectrum(p)
            w = cols["WAVELENGTH"]
            y = cols.get("FLUX")
            if y is None:
                y = cols.get("SURF_BRIGHT")
            if y is None:
                continue
            yerr = cols.get("FLUX_ERROR")
            if yerr is None:
                yerr = cols.get("SB_ERROR")

            peaks_tmp = _detect_emission_peaks(
                w,
                y,
                yerr,
                snr_threshold=float(snr_threshold),
                snr_threshold_fluxerr=float(snr_threshold_fluxerr),
                max_peaks=20,
            )
            if not bool(peaks_tmp.get("ok")):
                continue

            m = np.isfinite(w) & np.isfinite(y)
            if int(m.sum()) <= 10:
                continue
            wv = np.asarray(w[m], dtype=np.float64)
            span = float(np.nanmax(wv) - np.nanmin(wv))

            peaks_list = list(peaks_tmp.get("peaks") or [])
            n_peaks = int(len(peaks_list))
            snr_sum = float(sum(float(pp.get("snr") or 0.0) for pp in peaks_list))

            score_tmp = None
            score_ok = False
            best_n_matches = 0
            best_score = 0.0
            best_z = None
            score_reason = None
            if peaks_list:
                score_tmp = _score_redshift_candidates(
                    peaks_list,
                    w_min_um=float(peaks_tmp["w_min_um"]),
                    w_max_um=float(peaks_tmp["w_max_um"]),
                    z_min=float(z_min),
                    z_max=float(z_max),
                )
                score_ok = bool(score_tmp.get("ok"))
                if score_ok:
                    best = score_tmp.get("best") or {}
                    best_n_matches = int(best.get("n_matches") or 0)
                    best_score = float(best.get("score") or 0.0)
                    z_show = best.get("z_mean_from_matches")
                    if z_show is None:
                        z_show = best.get("z")
                    if z_show is not None:
                        best_z = float(z_show)
                else:
                    score_reason = str(score_tmp.get("reason") or "scoring_failed")

            meta = {
                "w_min_um": float(np.nanmin(wv)),
                "w_max_um": float(np.nanmax(wv)),
                "n_peaks": n_peaks,
                "snr_sum": snr_sum,
                "not_mirimage": bool(not_mirimage),
                "best_n_matches": int(best_n_matches),
                "best_score": float(best_score),
                "best_z": best_z,
                "score_ok": bool(score_ok),
                "score_reason": score_reason,
            }
            tried.append({"spectrum": _relpath(p), **meta})
            cand.append(
                {
                    "path": p,
                    "name": str(p.name),
                    "not_mirimage": bool(not_mirimage),
                    "span": float(span),
                    "n_peaks": int(n_peaks),
                    "snr_sum": float(snr_sum),
                    "peaks_rec": peaks_tmp,
                    "score_rec": score_tmp,
                    "score_ok": bool(score_ok),
                    "best_n_matches": int(best_n_matches),
                    "best_score": float(best_score),
                    "meta": meta,
                }
            )
        except Exception:
            continue

    if not cand:
        rec["reason"] = "no_readable_x1d"
        out_json.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return rec

    def _pick_best(pool: List[Dict[str, Any]], *, prefer_non_mirimage: bool) -> Optional[Dict[str, Any]]:
        src = pool
        if prefer_non_mirimage:
            src = [c for c in pool if bool(c.get("not_mirimage"))]
        src_scored = [c for c in src if bool(c.get("score_ok"))]
        if src_scored:
            return max(
                src_scored,
                key=lambda c: (
                    int(c.get("best_n_matches") or 0),
                    float(c.get("best_score") or 0.0),
                    int(c.get("n_peaks") or 0),
                    float(c.get("snr_sum") or 0.0),
                    float(c.get("span") or 0.0),
                    str(c.get("name") or ""),
                ),
            )
        return None

    picked = _pick_best(cand, prefer_non_mirimage=True)
    picked_reason = "non_mirimage_scored"
    if picked is None:
        picked = _pick_best(cand, prefer_non_mirimage=False)
        picked_reason = "any_scored"
    if picked is None:
        # Fall back to the previous peak-based heuristic (keeps behavior stable when scoring fails everywhere).
        picked = max(
            cand,
            key=lambda c: (
                int(bool(c.get("not_mirimage"))),
                int(c.get("n_peaks") or 0),
                float(c.get("snr_sum") or 0.0),
                float(c.get("span") or 0.0),
                str(c.get("name") or ""),
            ),
        )
        picked_reason = "fallback_most_peaks"

    best_path = Path(str(picked["path"]))
    best_peaks_rec = picked.get("peaks_rec")
    best_meta = picked.get("meta")
    best_score_rec = picked.get("score_rec")

    tried = sorted(
        tried,
        key=lambda r: (
            int(bool(r.get("score_ok"))),
            int(bool(r.get("not_mirimage"))),
            int(r.get("best_n_matches") or 0),
            float(r.get("best_score") or 0.0),
            int(r.get("n_peaks") or 0),
            float(r.get("snr_sum") or 0.0),
            str(r.get("spectrum") or ""),
        ),
        reverse=True,
    )
    rec["spectrum_selection"] = {
        "method": "score_then_peaks",
        "picked_reason": str(picked_reason),
        "picked_spectrum": _relpath(best_path),
        "picked_not_mirimage": bool(bool(best_meta.get("not_mirimage")) if isinstance(best_meta, dict) else None),
        "tried": tried,
    }

    cols = _read_x1d_spectrum(best_path)
    w = cols["WAVELENGTH"]
    y = cols.get("FLUX")
    if y is None:
        y = cols.get("SURF_BRIGHT")
    yerr = cols.get("FLUX_ERROR")
    if yerr is None:
        yerr = cols.get("SB_ERROR")

    peaks_rec = best_peaks_rec or _detect_emission_peaks(
        w,
        y,
        yerr,
        snr_threshold=float(snr_threshold),
        snr_threshold_fluxerr=float(snr_threshold_fluxerr),
        max_peaks=20,
    )
    rec["spectrum_used"] = _relpath(best_path)
    rec["w_range_um"] = best_meta
    rec["peaks"] = peaks_rec

    peaks: List[Dict[str, Any]] = []
    if not bool(peaks_rec.get("ok")):
        rec["reason"] = str(peaks_rec.get("reason") or "peak_detection_failed")
    else:
        peaks = list(peaks_rec.get("peaks") or [])
        if not peaks:
            rec["reason"] = "no_peaks_over_threshold"
        else:
            score_rec = best_score_rec or _score_redshift_candidates(
                peaks,
                w_min_um=float(peaks_rec["w_min_um"]),
                w_max_um=float(peaks_rec["w_max_um"]),
                z_min=float(z_min),
                z_max=float(z_max),
            )
            if not bool(score_rec.get("ok")):
                rec["reason"] = str(score_rec.get("reason") or "scoring_failed")
            else:
                rec["ok"] = True
                rec["best"] = score_rec.get("best")
                rec["top"] = score_rec.get("top")
                rec["score_params"] = score_rec.get("params")

    # CSV: small view (write even when ok=false so paper references stay stable)
    try:
        lines = ["rank,z,score,n_matches,z_mean_from_matches,z_std_from_matches,matched_lines"]
        for rank, c in enumerate(rec.get("top") or [], start=1):
            matched = ";".join([str(m.get("line") or "") for m in c.get("matches") or []][:8])
            lines.append(
                ",".join(
                    [
                        str(rank),
                        f"{float(c.get('z') or 0.0):.6f}",
                        f"{float(c.get('score') or 0.0):.3f}",
                        str(int(c.get("n_matches") or 0)),
                        f"{float(c.get('z_mean_from_matches') or 0.0):.6f}",
                        f"{float(c.get('z_std_from_matches') or 0.0):.6f}",
                        matched.replace(",", " "),
                    ]
                )
            )
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
        rec["csv"] = _relpath(out_csv)
    except Exception:
        rec["csv"] = None

    try:
        plot = _plot_target_z_diagnostic(
            target_slug,
            spectrum_path=best_path,
            peaks=peaks,
            best=rec.get("best"),
            out_png=out_png,
        )
        if plot is not None:
            rec["plot"] = plot
    except Exception:
        rec["plot"] = {"ok": False, "reason": "plot_failed"}

    out_json.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rec


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rest_um_from_name(line: str) -> Optional[float]:
    n = str(line or "").strip()
    if not n:
        return None
    for name, lam0 in _REST_LINES_UM:
        if str(name) == n:
            return float(lam0)
    return None


def _measure_line_centroid(
    w_um: np.ndarray,
    y: np.ndarray,
    yerr: Optional[np.ndarray],
    *,
    center_hint_um: float,
    window_um: float,
    kind: str,
    bootstrap_n: int,
    seed: int,
    prior_sigma_um: Optional[float] = None,
) -> Dict[str, Any]:
    w = np.asarray(w_um, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    m = np.isfinite(w) & np.isfinite(yv)
    ev = None
    if yerr is not None:
        ev = np.asarray(yerr, dtype=np.float64)
        m = m & np.isfinite(ev) & (ev > 0)
    if int(m.sum()) <= 10:
        return {"ok": False, "reason": "too_few_points"}

    w = w[m]
    yv = yv[m]
    ev = ev[m] if ev is not None else None

    c = float(center_hint_um)
    half = float(window_um) / 2.0
    win = (w >= (c - half)) & (w <= (c + half))
    if int(win.sum()) < 7:
        return {"ok": False, "reason": "too_few_points_in_window"}

    ww = np.asarray(w[win], dtype=np.float64)
    yy = np.asarray(yv[win], dtype=np.float64)
    ee = (np.asarray(ev[win], dtype=np.float64) if ev is not None else None)

    # Sort by wavelength (some JWST x1d are descending).
    order = np.argsort(ww)
    ww = ww[order]
    yy = yy[order]
    if ee is not None:
        ee = ee[order]

    is_abs = str(kind or "").lower().startswith("abs")

    def _trimmed_median(y_local: np.ndarray) -> Optional[float]:
        y_use = np.asarray(y_local, dtype=np.float64)
        y_use = y_use[np.isfinite(y_use)]
        if y_use.size <= 0:
            return None
        # Robust baseline: drop the most extreme 20% on each side (keeps continuum for narrow lines).
        y_sorted = np.sort(y_use)
        k = int(np.floor(0.2 * float(y_sorted.size)))
        if y_sorted.size >= 10 and (y_sorted.size - 2 * k) >= 3 and k > 0:
            y_sorted = y_sorted[k:-k]
        return float(np.nanmedian(y_sorted))

    def _centroid_from_local_window(yy_local: np.ndarray, ee_local: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
        base_local = _trimmed_median(yy_local)
        if base_local is None or not np.isfinite(base_local):
            return None
        resid_local = np.asarray(yy_local, dtype=np.float64) - float(base_local)
        if is_abs:
            prof_use_local = np.clip(-resid_local, 0.0, None)
        else:
            prof_use_local = np.clip(resid_local, 0.0, None)
        if not np.isfinite(prof_use_local).any() or float(np.nanmax(prof_use_local)) <= 0:
            return None

        # Choose the peak closest to the provided hint (Gaussian prior).
        if prior_sigma_um is not None and float(prior_sigma_um) > 0:
            prior_sigma = float(prior_sigma_um)
        else:
            prior_sigma = max(float(window_um) / 6.0, 1e-6)
        weight = np.exp(-0.5 * ((ww - float(center_hint_um)) / prior_sigma) ** 2)
        score = np.where(np.isfinite(prof_use_local), prof_use_local, 0.0) * weight
        if not np.isfinite(score).any() or float(np.nanmax(score)) <= 0:
            return None
        i_peak = int(np.nanargmax(score))
        # Guard: do not "snap" to a far-away stronger peak when the hint is meant
        # to select between nearby candidates (e.g., close doublets).
        #
        # The user can widen prior_sigma_um when the hint itself is uncertain.
        if float(np.abs(ww[i_peak] - float(center_hint_um))) > 3.0 * float(prior_sigma):
            return None
        amp_peak = float(score[i_peak])
        if not np.isfinite(amp_peak) or amp_peak <= 0:
            return None
        thr_peak = 0.5 * amp_peak

        # Use a contiguous region around the selected peak to avoid mixing multiple peaks.
        # NOTE: We use the *prior-weighted* score for the region to ensure that "candidate
        # selection" (near the hint) is not pulled by a nearby stronger feature.
        left = i_peak
        while left - 1 >= 0 and float(score[left - 1]) >= thr_peak:
            left -= 1
        right = i_peak
        while right + 1 < score.size and float(score[right + 1]) >= thr_peak:
            right += 1
        w_sel_local = ww[left : right + 1]
        p_sel_local = prof_use_local[left : right + 1]
        s = float(np.sum(p_sel_local))
        if not np.isfinite(s) or s <= 0:
            return None
        centroid_local = float(np.sum(w_sel_local * p_sel_local) / s)

        # A conservative systematics floor in wavelength (cannot beat sampling).
        step_local = float(np.nanmedian(np.abs(np.diff(w_sel_local)))) if w_sel_local.size >= 3 else float("nan")
        if not np.isfinite(step_local) or step_local <= 0:
            step_local = float(np.nanmedian(np.abs(np.diff(ww)))) if ww.size >= 3 else float("nan")
        sys_floor_local = 0.5 * step_local if np.isfinite(step_local) and step_local > 0 else None

        peak_um = float(ww[i_peak])
        peak_hp = float(resid_local[i_peak])
        peak_snr_fluxerr = None
        snr_int = None
        if ee_local is not None and np.isfinite(float(ee_local[i_peak])) and float(ee_local[i_peak]) > 0:
            peak_snr_fluxerr = float(peak_hp / float(ee_local[i_peak]))
            den = float(np.sqrt(np.sum(np.asarray(ee_local[left : right + 1], dtype=np.float64) ** 2)))
            if np.isfinite(den) and den > 0:
                snr_int = float(np.sum(np.asarray(p_sel_local, dtype=np.float64)) / den)

        return {
            "centroid_um": centroid_local,
            "sys_floor_um": sys_floor_local,
            "peak_um": peak_um,
            "peak_hp": peak_hp,
            "peak_snr_fluxerr": peak_snr_fluxerr,
            "snr_integrated_proxy": snr_int,
            "points_selected": int(w_sel_local.size),
            "prior_sigma_um": float(prior_sigma),
            "baseline_median": float(base_local),
        }

    base = _centroid_from_local_window(yy, ee)
    if base is None:
        return {"ok": False, "reason": "no_signal_over_baseline"}

    centroid = float(base["centroid_um"])
    sys_floor_um = base["sys_floor_um"]
    points_selected = int(base["points_selected"])

    # Bootstrap statistical uncertainty (if FLUX_ERROR exists).
    stat_sigma = None
    if ev is not None and int(bootstrap_n) > 0:
        rng = np.random.default_rng(int(seed))
        cents: List[float] = []
        for _ in range(int(bootstrap_n)):
            noise = rng.normal(0.0, ee)
            yb = yy + noise
            b = _centroid_from_local_window(yb, ee)
            if b is None:
                continue
            cents.append(float(b["centroid_um"]))
        if len(cents) >= 16:
            stat_sigma = float(np.nanstd(np.asarray(cents, dtype=np.float64)))

    return {
        "ok": True,
        "centroid_um": centroid,
        "centroid_stat_sigma_um": stat_sigma,
        "centroid_sys_floor_um": sys_floor_um,
        "window_um": float(window_um),
        "kind": "absorption" if str(kind or "").lower().startswith("abs") else "emission",
        "points_in_window": int(ww.size),
        "points_selected": points_selected,
        "peak_um": base.get("peak_um"),
        "peak_hp": base.get("peak_hp"),
        "peak_snr_fluxerr": base.get("peak_snr_fluxerr"),
        "snr_integrated_proxy": base.get("snr_integrated_proxy"),
        "prior_sigma_um": base.get("prior_sigma_um"),
    }


def _confirm_target_redshift(
    target_slug: str,
    *,
    target_dir: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    """
    Confirm redshift from a manually curated line list (line_id.json).

    This is a *workflow* primitive for Step 4.6.5:
    - The user decides which lines are real (line ID) and provides a window.
    - The code measures line centroids and returns z with stat+sys separated.
    """

    line_id_path = target_dir / "line_id.json"
    out_json = out_dir / f"jwst_spectra__{target_slug}__z_confirmed.json"
    out_csv = out_dir / f"jwst_spectra__{target_slug}__z_confirmed_lines.csv"
    out_png = out_dir / f"jwst_spectra__{target_slug}__z_confirmed.png"

    rec: Dict[str, Any] = {
        "generated_utc": _utc_now(),
        "target_slug": target_slug,
        "ok": False,
        "reason": None,
        "line_id": _relpath(line_id_path) if line_id_path.exists() else None,
        "spectrum_used": None,
        "spectra_used": None,
        "params": None,
        "z": None,
        "z_stat_sigma": None,
        "z_sys_sigma": None,
        "z_total_sigma": None,
        "chi2_dof": None,
        "lines": [],
        "lines_summary": [],
        "spectra": [],
        "csv": None,
        "plot": None,
        "plot_summary": None,
    }

    if not line_id_path.exists():
        rec["reason"] = "no_line_id"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return rec

    cfg = _read_json(line_id_path)
    if not isinstance(cfg, dict):
        rec["reason"] = "invalid_line_id_format"
        out_json.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return rec

    params = cfg.get("params") if isinstance(cfg.get("params"), dict) else {}
    bootstrap_n = int(params.get("bootstrap_n") or 0)
    seed = int(params.get("seed") or 0)
    sys_extra_um = params.get("sigma_sys_um")
    sys_extra_um_f = float(sys_extra_um) if sys_extra_um is not None else None
    min_ok_spectra_raw = params.get("min_ok_spectra")
    if min_ok_spectra_raw is None:
        min_ok_spectra_raw = params.get("min_ok_spectra_for_z")
    try:
        min_ok_spectra = int(min_ok_spectra_raw) if min_ok_spectra_raw is not None else 1
    except Exception:
        min_ok_spectra = 1
    min_ok_spectra = int(max(1, min_ok_spectra))
    max_z_pull_raw = params.get("max_z_pull")
    if max_z_pull_raw is None:
        max_z_pull_raw = params.get("max_z_sigma_pull")
    try:
        max_z_pull = float(max_z_pull_raw) if max_z_pull_raw is not None else None
    except Exception:
        max_z_pull = None
    if max_z_pull is not None and (not np.isfinite(max_z_pull) or float(max_z_pull) <= 0):
        max_z_pull = None

    lines_cfg = cfg.get("lines") if isinstance(cfg.get("lines"), list) else []
    if not lines_cfg:
        rec["reason"] = "no_lines_selected"
        out_json.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return rec

    # spectra (multi-spectrum cross-check). Backward compatible: accept {"spectrum":{...}}.
    spec_entries = cfg.get("spectra") if isinstance(cfg.get("spectra"), list) else None
    spec_paths_s: List[str] = []
    if spec_entries:
        for s in spec_entries:
            if isinstance(s, str):
                spec_paths_s.append(s)
            elif isinstance(s, dict) and s.get("path"):
                spec_paths_s.append(str(s.get("path")))
    else:
        spec = cfg.get("spectrum") if isinstance(cfg.get("spectrum"), dict) else {}
        if spec.get("path"):
            spec_paths_s.append(str(spec.get("path")))

    if not spec_paths_s:
        rec["reason"] = "missing_spectrum_path"
        out_json.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return rec

    spec_paths: List[Path] = []
    missing_specs: List[str] = []
    for s in spec_paths_s:
        p = Path(str(s))
        if not p.is_absolute():
            p = (_ROOT / p).resolve()
        if not p.exists():
            missing_specs.append(str(s))
            continue
        spec_paths.append(p)

    if not spec_paths:
        rec["reason"] = "spectrum_not_found"
        rec["spectrum_used"] = spec_paths_s[0] if spec_paths_s else None
        rec["spectra_used"] = list(spec_paths_s)
        out_json.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return rec

    rec["spectrum_used"] = _relpath(spec_paths[0])
    rec["spectra_used"] = [_relpath(p) for p in spec_paths]
    rec["params"] = {
        "bootstrap_n": bootstrap_n,
        "seed": seed,
        "sigma_sys_um": sys_extra_um_f,
        "min_ok_spectra": int(min_ok_spectra),
        "max_z_pull": float(max_z_pull) if max_z_pull is not None else None,
    }

    used: List[Dict[str, Any]] = []
    spectra_recs: List[Dict[str, Any]] = []

    for sp in spec_paths:
        spec_rec: Dict[str, Any] = {"spectrum": _relpath(sp), "ok": True, "reason": None, "lines": []}
        try:
            cols = _read_x1d_spectrum(sp)
            w = cols["WAVELENGTH"]
            y = cols.get("FLUX")
            if y is None:
                y = cols.get("SURF_BRIGHT")
            yerr = cols.get("FLUX_ERROR")
            if yerr is None:
                yerr = cols.get("SB_ERROR")
            if y is None:
                spec_rec["ok"] = False
                spec_rec["reason"] = "spectrum_missing_flux"
                spectra_recs.append(spec_rec)
                continue
        except Exception as e:
            spec_rec["ok"] = False
            spec_rec["reason"] = f"read_failed: {e}"
            spectra_recs.append(spec_rec)
            continue

        for it in lines_cfg:
            if not isinstance(it, dict):
                continue
            use_raw = it.get("use", True)
            if isinstance(use_raw, bool):
                use_flag = bool(use_raw)
            elif use_raw is None:
                use_flag = True
            elif isinstance(use_raw, (int, float)):
                use_flag = bool(use_raw)
            elif isinstance(use_raw, str):
                use_flag = str(use_raw).strip().lower() not in ("0", "false", "no", "off")
            else:
                use_flag = True
            line_name = str(it.get("line") or "").strip()
            rest_um = it.get("rest_um")
            if rest_um is None:
                rest_um = _rest_um_from_name(line_name)
            try:
                rest_um_f = float(rest_um)
            except Exception:
                continue
            obs_hint = it.get("obs_um")
            if obs_hint is None:
                continue
            try:
                obs_hint_f = float(obs_hint)
            except Exception:
                continue
            win_um = it.get("window_um") or params.get("window_um_default") or 0.2
            try:
                win_um_f = float(win_um)
            except Exception:
                win_um_f = 0.2
            kind = str(it.get("kind") or params.get("kind") or "emission")
            prior_sigma_um = it.get("prior_sigma_um")
            try:
                prior_sigma_um_f = float(prior_sigma_um) if prior_sigma_um is not None else None
            except Exception:
                prior_sigma_um_f = None
            meas = _measure_line_centroid(
                w,
                y,
                yerr,
                center_hint_um=obs_hint_f,
                window_um=win_um_f,
                kind=kind,
                bootstrap_n=bootstrap_n,
                seed=seed,
                prior_sigma_um=prior_sigma_um_f,
            )
            if not bool(meas.get("ok")):
                r = {
                    "spectrum": _relpath(sp),
                    "line": line_name,
                    "rest_um": rest_um_f,
                    "obs_hint_um": obs_hint_f,
                    "use": bool(use_flag),
                    "ok": False,
                    "reason": meas.get("reason"),
                }
                used.append(r)
                spec_rec["lines"].append(r)
                continue
            obs_um = float(meas["centroid_um"])
            obs_stat = meas.get("centroid_stat_sigma_um")
            obs_sys_floor = meas.get("centroid_sys_floor_um")
            if sys_extra_um_f is not None and obs_sys_floor is not None:
                obs_sys = float(max(float(obs_sys_floor), float(sys_extra_um_f)))
            elif sys_extra_um_f is not None:
                obs_sys = float(sys_extra_um_f)
            elif obs_sys_floor is not None:
                obs_sys = float(obs_sys_floor)
            else:
                obs_sys = None

            z = obs_um / rest_um_f - 1.0
            z_stat = (float(obs_stat) / rest_um_f) if obs_stat is not None else None
            z_sys = (float(obs_sys) / rest_um_f) if obs_sys is not None else None

            # Optional acceptance thresholds (per-line) to make "manual" line ID reproducible.
            # If a threshold is provided and the measurement doesn't meet it, keep the measured
            # values but mark ok=false so it won't be used in z combination.
            reject_reason = None
            min_snr_int = it.get("min_snr_integrated_proxy")
            if min_snr_int is None:
                min_snr_int = it.get("min_snr_integrated")
            if min_snr_int is None:
                min_snr_int = it.get("min_snr_int")
            if min_snr_int is not None:
                try:
                    thr = float(min_snr_int)
                except Exception:
                    thr = None
                if thr is not None and thr > 0:
                    snr_val = meas.get("snr_integrated_proxy")
                    try:
                        snr_f = float(snr_val) if snr_val is not None else None
                    except Exception:
                        snr_f = None
                    if snr_f is None or not np.isfinite(snr_f) or float(snr_f) < float(thr):
                        reject_reason = f"below_snr_integrated_proxy:{float(thr):g}"

            min_peak_snr = it.get("min_peak_snr_fluxerr")
            if reject_reason is None and min_peak_snr is not None:
                try:
                    thr = float(min_peak_snr)
                except Exception:
                    thr = None
                if thr is not None and thr > 0:
                    snr_val = meas.get("peak_snr_fluxerr")
                    try:
                        snr_f = float(snr_val) if snr_val is not None else None
                    except Exception:
                        snr_f = None
                    if snr_f is None or not np.isfinite(snr_f) or float(snr_f) < float(thr):
                        reject_reason = f"below_peak_snr_fluxerr:{float(thr):g}"

            r = {
                "spectrum": _relpath(sp),
                "line": line_name,
                "rest_um": rest_um_f,
                "obs_hint_um": obs_hint_f,
                "use": bool(use_flag),
                "obs_um": obs_um,
                "obs_stat_sigma_um": float(obs_stat) if obs_stat is not None else None,
                "obs_sys_sigma_um": float(obs_sys) if obs_sys is not None else None,
                "z": float(z),
                "z_stat_sigma": float(z_stat) if z_stat is not None else None,
                "z_sys_sigma": float(z_sys) if z_sys is not None else None,
                "meas": {k: v for k, v in meas.items() if k not in ("ok", "centroid_um", "centroid_stat_sigma_um", "centroid_sys_floor_um")},
                "ok": reject_reason is None,
                "reason": reject_reason,
            }
            used.append(r)
            spec_rec["lines"].append(r)

        spectra_recs.append(spec_rec)

    rec["lines"] = used
    rec["spectra"] = spectra_recs
    if max_z_pull is not None:
        # Optional outlier clipping (reproducible, data-driven) to avoid combining
        # an obviously inconsistent line measurement into the final z.
        cand = [
            r
            for r in used
            if isinstance(r, dict)
            and bool(r.get("ok"))
            and bool(r.get("use", True))
            and isinstance(r.get("z"), (int, float))
        ]
        if len(cand) >= 3:
            z_c = np.array([float(r["z"]) for r in cand], dtype=np.float64)
            zstat_c = np.array([float(r.get("z_stat_sigma") or np.nan) for r in cand], dtype=np.float64)
            zsys_c = np.array([float(r.get("z_sys_sigma") or np.nan) for r in cand], dtype=np.float64)
            stat_ok_c = np.isfinite(zstat_c) & (zstat_c > 0)
            sys_ok_c = np.isfinite(zsys_c) & (zsys_c > 0)
            var_tot_c = np.full_like(z_c, np.nan, dtype=np.float64)
            var_tot_c = np.where(stat_ok_c & sys_ok_c, (zstat_c**2) + (zsys_c**2), var_tot_c)
            var_tot_c = np.where(stat_ok_c & ~sys_ok_c, (zstat_c**2), var_tot_c)
            var_tot_c = np.where(~stat_ok_c & sys_ok_c, (zsys_c**2), var_tot_c)
            w_tot_c = np.where(np.isfinite(var_tot_c) & (var_tot_c > 0), 1.0 / var_tot_c, 0.0)
            if float(np.sum(w_tot_c)) > 0:
                z0 = float(np.sum(w_tot_c * z_c) / np.sum(w_tot_c))
            else:
                z0 = float(np.nanmedian(z_c))

            rej_n = 0
            for r, z, st, sy in zip(cand, z_c, zstat_c, zsys_c):
                var = np.nan
                if np.isfinite(st) and st > 0 and np.isfinite(sy) and sy > 0:
                    var = float(st**2 + sy**2)
                elif np.isfinite(st) and st > 0:
                    var = float(st**2)
                elif np.isfinite(sy) and sy > 0:
                    var = float(sy**2)
                if not np.isfinite(var) or float(var) <= 0:
                    continue
                pull = float(np.abs(float(z) - float(z0)) / float(np.sqrt(var)))
                if np.isfinite(pull) and float(pull) > float(max_z_pull):
                    r["ok"] = False
                    r["reason"] = f"outlier_z_pull:{pull:.3f} > {float(max_z_pull):g}"
                    rej_n += 1
            rec["outlier_clipping"] = {
                "max_z_pull": float(max_z_pull),
                "z_preclip": float(z0),
                "rejected_n": int(rej_n),
            }

    ok_lines = [
        r
        for r in used
        if isinstance(r, dict)
        and bool(r.get("ok"))
        and bool(r.get("use", True))
        and isinstance(r.get("z"), (int, float))
    ]
    if len(ok_lines) <= 0:
        rec["reason"] = "no_ok_lines"
        out_json.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return rec

    ok_spectra = sorted({str(r.get("spectrum") or "") for r in ok_lines if r.get("spectrum")})
    rec["ok_spectra_n"] = int(len(ok_spectra))
    rec["ok_spectra"] = ok_spectra

    z_vals = np.array([float(r["z"]) for r in ok_lines], dtype=np.float64)
    z_stat_vals = np.array([float(r.get("z_stat_sigma") or np.nan) for r in ok_lines], dtype=np.float64)
    z_sys_vals = np.array([float(r.get("z_sys_sigma") or np.nan) for r in ok_lines], dtype=np.float64)

    # Combine:
    # - z_mean: use *total* variance (stat+sys) when available to avoid over-weighting
    #   pathological near-zero stat sigmas (common when the centroid is dominated by sampling).
    # - z_stat_sigma: still report the inverse-variance combination of stat terms alone.
    stat_ok = np.isfinite(z_stat_vals) & (z_stat_vals > 0)
    sys_ok = np.isfinite(z_sys_vals) & (z_sys_vals > 0)
    var_tot = np.full_like(z_vals, np.nan, dtype=np.float64)
    var_tot = np.where(stat_ok & sys_ok, (z_stat_vals**2) + (z_sys_vals**2), var_tot)
    var_tot = np.where(stat_ok & ~sys_ok, (z_stat_vals**2), var_tot)
    var_tot = np.where(~stat_ok & sys_ok, (z_sys_vals**2), var_tot)
    w_tot = np.where(np.isfinite(var_tot) & (var_tot > 0), 1.0 / var_tot, 0.0)
    if float(np.sum(w_tot)) > 0:
        z_mean = float(np.sum(w_tot * z_vals) / np.sum(w_tot))
    else:
        z_mean = float(np.nanmedian(z_vals))

    w_stat = np.where(stat_ok, 1.0 / (z_stat_vals**2), 0.0)
    z_stat_sigma = float(np.sqrt(1.0 / np.sum(w_stat))) if float(np.sum(w_stat)) > 0 else None

    z_sys_sigma = float(np.nanmax(z_sys_vals)) if np.isfinite(z_sys_vals).any() else None
    if z_stat_sigma is not None and z_sys_sigma is not None:
        z_total_sigma = float(np.sqrt(float(z_stat_sigma) ** 2 + float(z_sys_sigma) ** 2))
    elif z_stat_sigma is not None:
        z_total_sigma = float(z_stat_sigma)
    elif z_sys_sigma is not None:
        z_total_sigma = float(z_sys_sigma)
    else:
        z_total_sigma = None

    # Consistency check (use stat+sys if available, else skip).
    var = (z_stat_vals**2) + (z_sys_vals**2)
    ok_var = np.isfinite(var) & (var > 0)
    if int(ok_var.sum()) >= 2:
        chi2 = float(np.sum(((z_vals[ok_var] - z_mean) ** 2) / var[ok_var]))
        dof = int(ok_var.sum()) - 1
        rec["chi2_dof"] = float(chi2 / float(dof)) if dof > 0 else None

    # If the per-spectrum scatter is larger than the stated (stat+sys) uncertainties,
    # inflate the total uncertainty by the Birge ratio sqrt(chi2/dof). This provides a
    # conservative, reproducible systematic envelope without relying on manual pruning.
    chi2_dof = rec.get("chi2_dof")
    if (
        z_total_sigma is not None
        and isinstance(chi2_dof, (int, float))
        and np.isfinite(float(chi2_dof))
        and float(chi2_dof) > 1.0
    ):
        try:
            birge = float(np.sqrt(float(chi2_dof)))
        except Exception:
            birge = None
        if birge is not None and np.isfinite(birge) and birge > 1.0:
            z_total_sigma_raw = float(z_total_sigma)
            z_total_sigma = float(z_total_sigma_raw * birge)
            if z_stat_sigma is not None and np.isfinite(float(z_stat_sigma)) and float(z_stat_sigma) >= 0:
                z_sys_sigma = float(np.sqrt(max(0.0, float(z_total_sigma) ** 2 - float(z_stat_sigma) ** 2)))
            else:
                z_sys_sigma = float(z_total_sigma)
            rec["uncertainty_inflation"] = {
                "method": "birge_ratio",
                "chi2_dof": float(chi2_dof),
                "birge_ratio": float(birge),
                "z_total_sigma_raw": z_total_sigma_raw,
                "z_total_sigma": float(z_total_sigma),
            }

    rec["ok"] = True
    rec["reason"] = None
    rec["z"] = z_mean
    rec["z_stat_sigma"] = z_stat_sigma
    rec["z_sys_sigma"] = z_sys_sigma
    rec["z_total_sigma"] = z_total_sigma
    if int(len(ok_spectra)) < int(min_ok_spectra):
        rec["ok"] = False
        rec["reason"] = f"insufficient_ok_spectra:{int(min_ok_spectra)} got:{int(len(ok_spectra))}"

    # Per-line summary (combine across spectra)
    by_line: Dict[str, List[Dict[str, Any]]] = {}
    for r in ok_lines:
        key = f"{r.get('line')}@{float(r.get('rest_um') or 0.0):.6f}"
        by_line.setdefault(key, []).append(r)
    line_summary: List[Dict[str, Any]] = []
    for key, grp in sorted(by_line.items(), key=lambda kv: kv[0]):
        z_g = np.array([float(rr["z"]) for rr in grp], dtype=np.float64)
        zstat_g = np.array([float(rr.get("z_stat_sigma") or np.nan) for rr in grp], dtype=np.float64)
        zsys_g = np.array([float(rr.get("z_sys_sigma") or np.nan) for rr in grp], dtype=np.float64)
        stat_ok_g = np.isfinite(zstat_g) & (zstat_g > 0)
        sys_ok_g = np.isfinite(zsys_g) & (zsys_g > 0)
        var_tot_g = np.full_like(z_g, np.nan, dtype=np.float64)
        var_tot_g = np.where(stat_ok_g & sys_ok_g, (zstat_g**2) + (zsys_g**2), var_tot_g)
        var_tot_g = np.where(stat_ok_g & ~sys_ok_g, (zstat_g**2), var_tot_g)
        var_tot_g = np.where(~stat_ok_g & sys_ok_g, (zsys_g**2), var_tot_g)
        w_tot_g = np.where(np.isfinite(var_tot_g) & (var_tot_g > 0), 1.0 / var_tot_g, 0.0)
        if float(np.sum(w_tot_g)) > 0:
            z_m = float(np.sum(w_tot_g * z_g) / np.sum(w_tot_g))
        else:
            z_m = float(np.nanmedian(z_g))

        w_g = np.where(stat_ok_g, 1.0 / (zstat_g**2), 0.0)
        z_s = float(np.sqrt(1.0 / np.sum(w_g))) if float(np.sum(w_g)) > 0 else None
        z_sys = float(np.nanmax(zsys_g)) if np.isfinite(zsys_g).any() else None
        z_tot = float(np.sqrt(float(z_s) ** 2 + float(z_sys) ** 2)) if (z_s is not None and z_sys is not None) else (float(z_s) if z_s is not None else (float(z_sys) if z_sys is not None else None))
        line_summary.append(
            {
                "key": key,
                "line": grp[0].get("line"),
                "rest_um": float(grp[0].get("rest_um") or 0.0),
                "n": int(len(grp)),
                "z": z_m,
                "z_stat_sigma": z_s,
                "z_sys_sigma": z_sys,
                "z_total_sigma": z_tot,
            }
        )
    rec["lines_summary"] = line_summary

    # CSV (per-line)
    try:
        rows = ["spectrum,line,rest_um,obs_um,obs_stat_sigma_um,obs_sys_sigma_um,z,z_stat_sigma,z_sys_sigma,ok,reason"]
        for r in used:
            rows.append(
                ",".join(
                    [
                        str(r.get("spectrum") or "").replace(",", " "),
                        str(r.get("line") or "").replace(",", " "),
                        f"{float(r.get('rest_um') or 0.0):.6f}",
                        f"{float(r.get('obs_um') or 0.0):.6f}" if r.get("obs_um") is not None else "",
                        f"{float(r.get('obs_stat_sigma_um') or 0.0):.6f}" if r.get("obs_stat_sigma_um") is not None else "",
                        f"{float(r.get('obs_sys_sigma_um') or 0.0):.6f}" if r.get("obs_sys_sigma_um") is not None else "",
                        f"{float(r.get('z') or 0.0):.6f}" if r.get("z") is not None else "",
                        f"{float(r.get('z_stat_sigma') or 0.0):.6f}" if r.get("z_stat_sigma") is not None else "",
                        f"{float(r.get('z_sys_sigma') or 0.0):.6f}" if r.get("z_sys_sigma") is not None else "",
                        "1" if bool(r.get("ok")) else "0",
                        str(r.get("reason") or "").replace(",", " "),
                    ]
                )
            )
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")
        rec["csv"] = _relpath(out_csv)
    except Exception:
        rec["csv"] = None

    # Plot
    if plt is not None and spec_paths:
        try:
            # Choose the best spectrum for the plot (max ok lines, then max integrated SNR proxy).
            best_spec = spec_paths[0]
            best_key = None
            for sr in spectra_recs:
                if not isinstance(sr, dict) or not sr.get("spectrum"):
                    continue
                ok_ls = [
                    rr
                    for rr in (sr.get("lines") or [])
                    if isinstance(rr, dict) and bool(rr.get("ok")) and bool(rr.get("use", True))
                ]
                n_ok = int(len(ok_ls))
                snr_sum = float(
                    sum(float((rr.get("meas") or {}).get("snr_integrated_proxy") or 0.0) for rr in ok_ls if isinstance(rr.get("meas"), dict))
                )
                key = (n_ok, snr_sum)
                if best_key is None or key > best_key:
                    best_key = key
                    best_spec = Path(str(sr["spectrum"]))
                    if not best_spec.is_absolute():
                        best_spec = (_ROOT / best_spec).resolve()

            cols = _read_x1d_spectrum(best_spec)
            w = cols["WAVELENGTH"]
            y = cols.get("FLUX")
            if y is None:
                y = cols.get("SURF_BRIGHT")

            fig, ax = plt.subplots(figsize=(12, 5))
            m = np.isfinite(w) & np.isfinite(y)
            ax.plot(w[m], y[m], lw=0.8, alpha=0.75, color="#333333")
            ax.set_title(f"JWST x1d z confirmed : {target_slug} (z≈{z_mean:.3f})")
            ax.set_xlabel("wavelength [µm]")
            ax.set_ylabel("flux (raw units)")
            for r in ok_lines:
                if str(r.get("spectrum") or "") != _relpath(best_spec):
                    continue
                x = float(r.get("obs_um") or 0.0)
                lab = str(r.get("line") or "")
                ax.axvline(x, color="#cc0000", lw=0.9, alpha=0.6)
                if lab:
                    ax.text(x, float(np.nanmax(y[m])), lab, rotation=90, va="top", ha="right", fontsize=7, alpha=0.8)
            ax.grid(True, alpha=0.25)
            out_png.parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            fig.savefig(out_png, dpi=160)
            plt.close(fig)
            rec["plot"] = {"ok": True, "out_png": _relpath(out_png)}
        except Exception:
            rec["plot"] = {"ok": False, "reason": "plot_failed"}

    # Plot summary (z per spectrum)
    if plt is not None and len(spectra_recs) >= 2:
        try:
            out_png2 = out_dir / f"jwst_spectra__{target_slug}__z_confirmed_summary.png"
            names: List[str] = []
            z_spec: List[float] = []
            zerr_spec: List[float] = []
            for sr in spectra_recs:
                sp_name = str(sr.get("spectrum") or "")
                names.append(Path(sp_name).name if sp_name else "spec")
                ok_ls = [
                    rr
                    for rr in (sr.get("lines") or [])
                    if isinstance(rr, dict)
                    and bool(rr.get("ok"))
                    and bool(rr.get("use", True))
                    and isinstance(rr.get("z"), (int, float))
                ]
                if not ok_ls:
                    z_spec.append(float("nan"))
                    zerr_spec.append(float("nan"))
                    continue
                zv = np.array([float(rr["z"]) for rr in ok_ls], dtype=np.float64)
                zsv = np.array([float(rr.get("z_stat_sigma") or np.nan) for rr in ok_ls], dtype=np.float64)
                wv = np.where(np.isfinite(zsv) & (zsv > 0), 1.0 / (zsv**2), 0.0)
                if float(np.sum(wv)) > 0:
                    zm = float(np.sum(wv * zv) / np.sum(wv))
                    ze = float(np.sqrt(1.0 / np.sum(wv)))
                else:
                    zm = float(np.nanmedian(zv))
                    ze = float("nan")
                z_spec.append(zm)
                zerr_spec.append(ze)
            xs = np.arange(len(names), dtype=np.float64)
            fig, ax = plt.subplots(figsize=(12.8, 6.0), dpi=200)
            ax.errorbar(xs, z_spec, yerr=zerr_spec, fmt="o", color="#1f77b4", ecolor="#1f77b4", elinewidth=1.0, capsize=3)
            ax.axhline(float(z_mean), color="#cc0000", lw=1.0, alpha=0.7, label="combined")
            ax.set_xticks(xs)
            ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel("z (per-spectrum; stat err only)")
            ax.set_title(f"JWST z_confirmed summary : {target_slug}")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8, loc="best")
            out_png2.parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            fig.savefig(out_png2, dpi=220, bbox_inches="tight")
            plt.close(fig)
            rec["plot_summary"] = {"ok": True, "out_png": _relpath(out_png2)}
        except Exception:
            rec["plot_summary"] = {"ok": False, "reason": "plot_failed"}

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(rec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return rec


def _plot_target_qc(target_slug: str, *, x1d_paths: List[Path], out_png: Path) -> Optional[Dict[str, Any]]:
    if plt is None:
        return {"ok": False, "reason": "matplotlib not available"}
    if not x1d_paths:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    n_ok = 0
    for p in x1d_paths:
        try:
            cols = _read_x1d_spectrum(p)
            w = cols["WAVELENGTH"]
            y = cols.get("FLUX")
            if y is None:
                y = cols.get("SURF_BRIGHT")
            if y is None:
                continue
            m = np.isfinite(w) & np.isfinite(y)
            if int(m.sum()) <= 3:
                continue
            ax.plot(w[m], y[m], lw=0.8, alpha=0.7, label=p.name)
            n_ok += 1
        except Exception:
            continue

    if n_ok <= 0:
        plt.close(fig)
        return {"ok": False, "reason": "no readable x1d spectra"}

    ax.set_title(f"JWST x1d spectrum (MAST) : {target_slug}")
    ax.set_xlabel("wavelength [µm]")
    ax.set_ylabel("flux (raw units)")
    ax.grid(True, alpha=0.3)
    if n_ok <= 8:
        ax.legend(fontsize=7, loc="best")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return {"ok": True, "spectra_plotted": int(n_ok), "out_png": _relpath(out_png)}


def _scan_local_x1d_fits(raw_dir: Path) -> List[Path]:
    if not raw_dir.exists():
        return []

    patterns = [
        "*_x1d.fits",
        "*_x1d.fits.gz",
        "*_x1d.fits.fz",
        "*x1d*.fits",
        "*x1d*.fits.gz",
        "*x1d*.fits.fz",
    ]
    found: List[Path] = []
    for pat in patterns:
        found.extend(list(raw_dir.glob(pat)))

    out: List[Path] = []
    for p in sorted(set(found)):
        try:
            if not p.is_file():
                continue
        except Exception:
            continue
        nm = p.name.lower()
        if not (nm.endswith(".fits") or nm.endswith(".fits.gz") or nm.endswith(".fits.fz")):
            continue
        out.append(p)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--targets",
        type=str,
        default="ALL",
        help="comma-separated target keys (match targets.json keys). Use ALL/AUTO to run all configured targets.",
    )
    p.add_argument("--obs-collection", type=str, default="JWST")
    p.add_argument("--offline", action="store_true", help="do not query/download; use local cache only")
    p.add_argument("--download-missing", action="store_true", help="download missing x1d FITS/PNG into data/.../raw/")
    p.add_argument("--include-previews", action="store_true", help="also download x1d PNG previews when available")
    p.add_argument("--estimate-z", action="store_true", help="estimate redshift candidates from local x1d spectra (offline ok)")
    p.add_argument("--confirm-z", action="store_true", help="confirm redshift from manual line_id.json (offline ok)")
    p.add_argument("--init-line-id", action="store_true", help="create a line_id.json template under data/.../<target>/ if missing")
    p.add_argument("--z-min", type=float, default=0.0, help="min z for candidate search (default: 0)")
    p.add_argument("--z-max", type=float, default=25.0, help="max z for candidate search (default: 25)")
    p.add_argument("--peak-snr", type=float, default=6.0, help="peak SNR threshold for candidate extraction (default: 6)")
    p.add_argument(
        "--peak-snr-fluxerr",
        type=float,
        default=0.0,
        help="optional additional threshold using FLUX_ERROR: require (high-pass flux / flux_error) >= this value (default: 0=disabled)",
    )
    p.add_argument("--max-obs", type=int, default=0, help="limit number of observations per target (0=all)")
    p.add_argument(
        "--targets-config",
        type=str,
        default="",
        help="optional JSON mapping for target_name -> {mast_target_name, aliases, position, extra_filters}. Default is data/.../jwst_spectra/targets.json if present.",
    )
    args = p.parse_args(argv)

    base_dir = _ROOT / "data" / "cosmology" / "mast" / "jwst_spectra"
    out_dir = _ROOT / "output" / "cosmology"
    base_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    config_path = Path(str(args.targets_config)).expanduser() if str(args.targets_config).strip() else (base_dir / "targets.json")
    target_cfg_map: Dict[str, Any] = _read_target_config(config_path) if config_path.exists() else {}

    targets_raw = str(args.targets).strip()
    if not targets_raw or targets_raw.upper() in {"ALL", "AUTO"}:
        if target_cfg_map:
            targets = sorted(target_cfg_map.keys())
        else:
            targets = ["GN-z11", "JADES-GS-z14-0", "LID-568", "GLASS-z12"]
    else:
        targets = [t.strip() for t in targets_raw.split(",") if t.strip()]
        if any(t.upper() in {"ALL", "AUTO"} for t in targets):
            if target_cfg_map:
                cfg_targets = sorted(target_cfg_map.keys())
            else:
                cfg_targets = ["GN-z11", "JADES-GS-z14-0", "LID-568", "GLASS-z12"]
            expanded: List[str] = []
            for t in targets:
                if t.upper() in {"ALL", "AUTO"}:
                    expanded.extend(cfg_targets)
                else:
                    expanded.append(t)
            targets = list(dict.fromkeys(expanded))
    if not targets:
        raise SystemExit("--targets is empty")

    now_mjd = _utc_to_mjd(datetime.now(timezone.utc))

    all_targets: List[str] = sorted(target_cfg_map.keys()) if target_cfg_map else list(targets)

    manifest_all: Dict[str, Any] = {
        "generated_utc": _utc_now(),
        "targets": all_targets,
        "run_targets": targets,
        "obs_collection": str(args.obs_collection),
        "offline": bool(args.offline),
        "download_missing": bool(args.download_missing),
        "include_previews": bool(args.include_previews),
        "estimate_z": bool(args.estimate_z),
        "confirm_z": bool(getattr(args, "confirm_z", False)),
        "init_line_id": bool(getattr(args, "init_line_id", False)),
        "z_min": float(args.z_min),
        "z_max": float(args.z_max),
        "peak_snr": float(args.peak_snr),
        "peak_snr_fluxerr": float(getattr(args, "peak_snr_fluxerr", 0.0)),
        "targets_config": _relpath(config_path) if config_path.exists() else None,
        "items": {},
    }

    for target in targets:
        slug = _slugify(target)
        tdir = base_dir / slug
        raw_dir = tdir / "raw"
        tdir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = tdir / "manifest.json"
        z_est_out_json = out_dir / f"jwst_spectra__{slug}__z_estimate.json"
        z_conf_out_json = out_dir / f"jwst_spectra__{slug}__z_confirmed.json"

        item: Dict[str, Any] = {
            "target_name": target,
            "target_slug": slug,
            "generated_utc": _utc_now(),
            "obs_collection": str(args.obs_collection),
            "query": None,
            "obs": [],
            "downloads": [],
            "qc": None,
            "line_id": None,
            "z_estimate": None,
            "z_confirmed": None,
        }

        obs_rows: List[Dict[str, Any]] = []
        if args.offline:
            cache_ok = False
            cache_reason: Optional[str] = None
            if manifest_path.exists():
                try:
                    item["cache_manifest"] = _relpath(manifest_path)
                    cached = json.loads(manifest_path.read_text(encoding="utf-8"))
                    cached = cached if isinstance(cached, dict) else {}
                    obs_rows = list(cached.get("obs") or [])
                    # Preserve cached results so --offline does not wipe derived states.
                    item["downloads"] = list(cached.get("downloads") or [])
                    if isinstance(cached.get("qc"), dict):
                        item["qc"] = cached.get("qc")
                    if isinstance(cached.get("line_id"), dict):
                        item["line_id"] = cached.get("line_id")
                    if isinstance(cached.get("z_estimate"), dict):
                        item["z_estimate"] = _summarize_z_estimate(cached.get("z_estimate") or {}, z_est_out_json)
                    if isinstance(cached.get("z_confirmed"), dict):
                        item["z_confirmed"] = _summarize_z_confirmed(cached.get("z_confirmed") or {}, z_conf_out_json)
                    cache_ok = True
                except Exception:
                    obs_rows = []
                    cache_reason = "cache_read_failed"
            else:
                cache_reason = "no_cache_manifest"
            item["query"] = {"method": "offline_cache", "ok": cache_ok, "reason": cache_reason}
        else:
            cfg = target_cfg_map.get(target) if isinstance(target_cfg_map, dict) else None
            obs_rows, qinfo = _query_jwst_obs_with_fallback(target, obs_collection=str(args.obs_collection), cfg=cfg if isinstance(cfg, dict) else None)
            item["query"] = qinfo

        # If derived outputs already exist, keep lightweight summaries in the manifest.
        # This avoids recomputation for dashboards/scoreboards while keeping manifests small.
        if item.get("z_estimate") is None:
            item["z_estimate"] = _load_z_estimate_summary(z_est_out_json)
        if item.get("z_confirmed") is None:
            item["z_confirmed"] = _load_z_confirmed_summary(z_conf_out_json)

        if int(args.max_obs) > 0:
            obs_rows = obs_rows[: int(args.max_obs)]

        x1d_local: List[Path] = []
        for obs in obs_rows:
            obsid = obs.get("obsid")
            if obsid is None:
                continue
            try:
                obsid_int = int(obsid)
            except Exception:
                continue
            obs_rec: Dict[str, Any] = dict(obs)
            obs_rec["obsid"] = obsid_int
            obs_rec["t_obs_release_utc"] = None
            obs_rec["is_released"] = None
            try:
                if obs_rec.get("t_obs_release") is not None:
                    t_rel = float(obs_rec["t_obs_release"])
                    obs_rec["t_obs_release_utc"] = _mjd_to_utc_iso(t_rel)
                    obs_rec["is_released"] = bool(np.isfinite(t_rel) and (t_rel <= now_mjd))
            except Exception:
                obs_rec["t_obs_release_utc"] = None
                obs_rec["is_released"] = None

            x1d_fits_min: List[Dict[str, Any]] = []
            previews_min: List[Dict[str, Any]] = []
            if args.offline:
                # Keep cached product lists when available so offline QC can find local files.
                x1d_fits_min = list(obs_rec.get("x1d_fits") or [])
                previews_min = list(obs_rec.get("x1d_previews") or [])
            else:
                products = _query_products_for_obsid(obsid_int)
                x1d_fits, previews = _select_x1d_products(products)
                obs_rec["x1d_fits_n"] = int(len(x1d_fits))
                obs_rec["x1d_preview_n"] = int(len(previews))
                # Record selected products (lightweight; avoid bloating manifest too much).
                x1d_fits_min = [
                    {
                        "productFilename": r.get("productFilename"),
                        "dataURI": r.get("dataURI"),
                        "calib_level": r.get("calib_level"),
                    }
                    for r in x1d_fits
                ]
                previews_min = [
                    {
                        "productFilename": r.get("productFilename"),
                        "dataURI": r.get("dataURI"),
                        "calib_level": r.get("calib_level"),
                    }
                    for r in previews
                ]
                obs_rec["x1d_fits"] = x1d_fits_min
                obs_rec["x1d_previews"] = previews_min

            if "x1d_fits_n" not in obs_rec:
                obs_rec["x1d_fits_n"] = int(len(x1d_fits_min))
            if "x1d_preview_n" not in obs_rec:
                obs_rec["x1d_preview_n"] = int(len(previews_min))
            if "x1d_fits" not in obs_rec:
                obs_rec["x1d_fits"] = x1d_fits_min
            if "x1d_previews" not in obs_rec:
                obs_rec["x1d_previews"] = previews_min
            item["obs"].append(obs_rec)

            if args.offline or not bool(args.download_missing):
                # collect existing local x1d for QC
                for r in obs_rec.get("x1d_fits") or []:
                    fn = str(r.get("productFilename") or "")
                    if not fn:
                        continue
                    pth = raw_dir / fn
                    if pth.exists():
                        x1d_local.append(pth)
                continue

            # If the product release date is in the future, do not spam download attempts.
            if obs_rec.get("is_released") is False:
                obs_rec["download_blocked_reason"] = "not_released_yet"
                continue

            # download x1d FITS (and optionally PNG previews)
            for r in x1d_fits:
                fn = str(r.get("productFilename") or "")
                uri = str(r.get("dataURI") or "")
                if not fn or not uri:
                    continue
                dst = raw_dir / fn
                if dst.exists():
                    x1d_local.append(dst)
                    continue
                dl = _mast_download(uri, dst)
                dl_rec = {"productFilename": fn, "dataURI": uri, "dst": _relpath(dst)} | dl
                item["downloads"].append(dl_rec)
                if dl.get("ok"):
                    x1d_local.append(dst)

            if bool(args.include_previews):
                for r in previews:
                    fn = str(r.get("productFilename") or "")
                    uri = str(r.get("dataURI") or "")
                    if not fn or not uri:
                        continue
                    dst = raw_dir / fn
                    if dst.exists():
                        continue
                    dl = _mast_download(uri, dst)
                    dl_rec = {"productFilename": fn, "dataURI": uri, "dst": _relpath(dst)} | dl
                    item["downloads"].append(dl_rec)

        # persist per-target manifest
        manifest_path.write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        if args.offline and not x1d_local:
            # In offline mode, cached manifests may not include the full x1d product list even if files exist
            # in data/.../raw/. Fall back to scanning the raw directory so QC/z-estimate are reproducible.
            x1d_local = _scan_local_x1d_fits(raw_dir)

        # QC plot (only from local x1d fits)
        qc_png = out_dir / f"jwst_spectra__{slug}__x1d_qc.png"
        qc = _plot_target_qc(slug, x1d_paths=sorted(set(x1d_local)), out_png=qc_png)
        if qc is not None:
            item["qc"] = qc
            # refresh manifest with qc result
            manifest_path.write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        # Manual line-ID template (Step 4.6.5)
        line_id_path = tdir / "line_id.json"
        if bool(getattr(args, "init_line_id", False)) and not line_id_path.exists():
            try:
                best_path: Optional[Path] = None
                best_key: Optional[Tuple[int, int, str]] = None  # (calib_level, not_mirimage, filename)
                for obs_rec in item.get("obs") or []:
                    if not isinstance(obs_rec, dict):
                        continue
                    for r in obs_rec.get("x1d_fits") or []:
                        if not isinstance(r, dict):
                            continue
                        fn = str(r.get("productFilename") or "")
                        if not fn:
                            continue
                        try:
                            calib = int(r.get("calib_level")) if r.get("calib_level") is not None else -1
                        except Exception:
                            calib = -1
                        not_mir = 0 if "mirimage" in fn.lower() else 1
                        key = (calib, not_mir, fn)
                        pth = raw_dir / fn
                        if not pth.exists():
                            continue
                        if best_key is None or key > best_key:
                            best_key = key
                            best_path = pth

                template: Dict[str, Any] = {
                    "generated_utc": _utc_now(),
                    "target_slug": slug,
                    "spectrum": {
                        "path": _relpath(best_path) if best_path is not None else "",
                        "notes": "Choose a representative x1d FITS and fill lines[].",
                    },
                    "params": {
                        "window_um_default": 0.3,
                        "bootstrap_n": 400,
                        "seed": 0,
                        # Extra systematics floor in wavelength [µm] (optional). Sampling floor is always applied.
                        "sigma_sys_um": None,
                        "kind": "emission",
                    },
                    "line_entry_template": {
                        "line": "Hα",
                        "rest_um": 0.6563,
                        "obs_um": 10.1,
                        "window_um": 0.3,
                        "use": True,
                        # Optional: narrow the centroid selection around obs_um (helps separate close lines).
                        "prior_sigma_um": None,
                        "kind": "emission",
                    },
                    "lines": [],
                    "rest_lines_um": [{"line": n, "um": float(l)} for n, l in _REST_LINES_UM],
                }
                line_id_path.write_text(json.dumps(template, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            except Exception:
                pass

        if line_id_path.exists():
            item["line_id"] = {"path": _relpath(line_id_path)}
            manifest_path.write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        # Redshift candidates (Step 4.6.3; offline from cached x1d)
        if bool(getattr(args, "estimate_z", False)):
            try:
                z_est = _estimate_target_redshift(
                    slug,
                    x1d_paths=sorted(set(x1d_local)),
                    out_dir=out_dir,
                    snr_threshold=float(args.peak_snr),
                    snr_threshold_fluxerr=float(args.peak_snr_fluxerr),
                    z_min=float(args.z_min),
                    z_max=float(args.z_max),
                )
                item["z_estimate"] = _summarize_z_estimate(z_est, z_est_out_json)
                manifest_path.write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            except Exception as e:
                item["z_estimate"] = {"ok": False, "reason": f"exception: {e}"}
                manifest_path.write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        # Confirmed z (manual line ID; Step 4.6.5)
        if bool(getattr(args, "confirm_z", False)):
            try:
                z_conf = _confirm_target_redshift(slug, target_dir=tdir, out_dir=out_dir)
                item["z_confirmed"] = _summarize_z_confirmed(z_conf, z_conf_out_json)
                manifest_path.write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            except Exception as e:
                item["z_confirmed"] = {"ok": False, "reason": f"exception: {e}"}
                manifest_path.write_text(json.dumps(item, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        z_est_summary = None
        if isinstance(item.get("z_estimate"), dict):
            z_est_summary = _summarize_z_estimate(item.get("z_estimate") or {}, z_est_out_json)
        else:
            z_est_summary = _load_z_estimate_summary(z_est_out_json)
        z_conf_summary = None
        if isinstance(item.get("z_confirmed"), dict):
            z_conf_summary = _summarize_z_confirmed(item.get("z_confirmed") or {}, z_conf_out_json)
        else:
            z_conf_summary = _load_z_confirmed_summary(z_conf_out_json)

        manifest_all["items"][slug] = {
            "manifest": _relpath(manifest_path),
            "qc": item.get("qc"),
            "z_estimate": z_est_summary,
            "line_id": item.get("line_id"),
            "z_confirmed": z_conf_summary,
        }

    # Write aggregated manifest (small) for quick inspection.
    manifest_all_path = base_dir / "manifest_all.json"

    # When running on a subset of targets, do not clobber previous entries.
    # This keeps the aggregated manifest usable for dashboards/Table 1 without requiring a full rerun.
    if manifest_all_path.exists():
        try:
            prev = json.loads(manifest_all_path.read_text(encoding="utf-8"))
            prev_items = prev.get("items") if isinstance(prev, dict) else None
            if isinstance(prev_items, dict):
                cur_items = manifest_all.get("items")
                if isinstance(cur_items, dict):
                    for k, v in prev_items.items():
                        if k not in cur_items:
                            cur_items[k] = v
        except Exception:
            pass

    manifest_all_path.write_text(json.dumps(manifest_all, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    try:
        worklog.append_event(
            {
                "event_type": "fetch_mast_jwst_spectra",
                "argv": sys.argv,
                "params": {
                    "targets": targets,
                    "obs_collection": str(args.obs_collection),
                    "offline": bool(args.offline),
                    "download_missing": bool(args.download_missing),
                    "include_previews": bool(args.include_previews),
                    "estimate_z": bool(args.estimate_z),
                    "confirm_z": bool(getattr(args, "confirm_z", False)),
                    "init_line_id": bool(getattr(args, "init_line_id", False)),
                    "z_min": float(args.z_min),
                    "z_max": float(args.z_max),
                    "peak_snr": float(args.peak_snr),
                    "peak_snr_fluxerr": float(getattr(args, "peak_snr_fluxerr", 0.0)),
                    "max_obs": int(args.max_obs),
                },
                "outputs": {"manifest_all": manifest_all_path},
            }
        )
    except Exception:
        pass

    print(f"[ok] wrote: {_relpath(manifest_all_path)}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
