#!/usr/bin/env python3
"""
llr_batch_eval.py

EDC（月次NP2）をまとめて解析し、複数局×複数反射器の LLR を
同一条件で P-model と比較・集計する。

ポイント:
  - Horizons のネットワーク呼び出し回数を抑えるため、全観測の epoch を集約して一括取得する
  - station->Moon (topocentric) も station ごとに epoch を集約して取得する（観測局コード単位）
  - 3段階の幾何モデルを並列に評価して「どこで改善したか」を可視化する
      1) 地球中心→月中心
      2) 観測局→月中心（topocentric）
      3) 観測局→反射器（MOON_PA_DE421 + SPICE）

出力（固定: output/private/llr/batch/）
  - llr_batch_metrics.csv
  - llr_batch_summary.json
  - llr_rms_improvement_overall.png
  - llr_rms_by_station_target.png
  - llr_rms_ablations_overall.png
  - llr_shapiro_ablations_overall.png
  - llr_tide_ablations_overall.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.llr import llr_pmodel_overlay_horizons_noargs as llr  # noqa: E402
from scripts.llr import ocean_loading_harpos as ol  # noqa: E402
from scripts.summary import worklog  # noqa: E402

LLR_SHORT_NAME = "月レーザー測距（LLR: Lunar Laser Ranging）"

# NGLR-1 is new and currently has sparse public NP coverage.
# Keep the global LLR min-points policy (default 30 in run_all) while allowing
# NGLR-1 to be evaluated with a smaller threshold.
_NGLR1_MIN_POINTS_CAP = 6


def _min_points_for_target(target: Any, default_min_points: int) -> int:
    try:
        t = str(target or "").strip().lower()
    except Exception:
        t = ""

    # 条件分岐: `t == "nglr1"` を満たす経路を評価する。

    if t == "nglr1":
        return int(min(int(default_min_points), int(_NGLR1_MIN_POINTS_CAP)))

    return int(default_min_points)


def _repo_root() -> Path:
    return _ROOT


def _set_japanese_font() -> None:
    # Reuse the same policy as other scripts
    try:
        llr._set_japanese_font()  # type: ignore[attr-defined]
    except Exception:
        pass


def _save_placeholder_plot_png(path: Path, title: str, lines: List[str]) -> None:
    _set_japanese_font()
    fig, ax = plt.subplots(figsize=(12.8, 4.8))
    ax.axis("off")
    fig.suptitle(title, fontsize=12)
    y = 0.86
    for line in lines:
        ax.text(0.02, y, str(line), transform=ax.transAxes, fontsize=11, va="top")
        y -= 0.11

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(path, dpi=200)
    plt.close()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_station_xyz_overrides(path: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    diag: Dict[str, Any] = {
        "path": str(path),
        "n_candidates": 0,
        "n_loaded": 0,
        "loaded_stations": [],
    }
    data = _read_json(path)
    out: Dict[str, Dict[str, Any]] = {}

    def _add(station: Any, rec: Any, default_source: str) -> None:
        # 条件分岐: `not isinstance(rec, dict)` を満たす経路を評価する。
        if not isinstance(rec, dict):
            return

        st = str(station or "").strip().upper()
        # 条件分岐: `not st` を満たす経路を評価する。
        if not st:
            return

        try:
            x = float(rec["x_m"])
            y = float(rec["y_m"])
            z = float(rec["z_m"])
        except Exception:
            return

        row: Dict[str, Any] = {"x_m": x, "y_m": y, "z_m": z}
        for k in ("pos_eop_yymmdd", "pos_eop_ref_epoch_utc", "log_file", "log_date", "source_group"):
            # 条件分岐: `rec.get(k) is not None` を満たす経路を評価する。
            if rec.get(k) is not None:
                row[k] = rec.get(k)

        src = str(rec.get("coord_source") or default_source).strip()
        # 条件分岐: `src` を満たす経路を評価する。
        if src:
            row["coord_source"] = src

        out[st] = row
        diag["n_loaded"] = int(diag["n_loaded"]) + 1
        diag["loaded_stations"].append(st)

    # 条件分岐: `isinstance(data, dict)` を満たす経路を評価する。

    if isinstance(data, dict):
        route = data.get("deterministic_merge_route")
        # 条件分岐: `isinstance(route, dict)` を満たす経路を評価する。
        if isinstance(route, dict):
            sel = route.get("selected_xyz")
            # 条件分岐: `isinstance(sel, dict)` を満たす経路を評価する。
            if isinstance(sel, dict):
                diag["n_candidates"] = int(diag["n_candidates"]) + 1
                _add(sel.get("station") or "APOL", sel, default_source=f"merge_route:{path.name}")

        sel0 = data.get("selected_xyz")
        # 条件分岐: `isinstance(sel0, dict)` を満たす経路を評価する。
        if isinstance(sel0, dict):
            diag["n_candidates"] = int(diag["n_candidates"]) + 1
            _add(sel0.get("station") or "APOL", sel0, default_source=f"selected_xyz:{path.name}")

        stations = data.get("stations")
        # 条件分岐: `isinstance(stations, dict)` を満たす経路を評価する。
        if isinstance(stations, dict):
            for st, rec in stations.items():
                diag["n_candidates"] = int(diag["n_candidates"]) + 1
                _add(st, rec, default_source=f"stations:{path.name}")
        else:
            for st, rec in data.items():
                # 条件分岐: `st in ("deterministic_merge_route", "selected_xyz", "target", "version", "sta...` を満たす経路を評価する。
                if st in ("deterministic_merge_route", "selected_xyz", "target", "version", "status"):
                    continue

                # 条件分岐: `isinstance(rec, dict) and all(k in rec for k in ("x_m", "y_m", "z_m"))` を満たす経路を評価する。

                if isinstance(rec, dict) and all(k in rec for k in ("x_m", "y_m", "z_m")):
                    diag["n_candidates"] = int(diag["n_candidates"]) + 1
                    _add(st, rec, default_source=f"mapping:{path.name}")

    diag["loaded_stations"] = sorted(set(str(v) for v in diag["loaded_stations"]))
    return out, diag


def _quantize(dt: datetime) -> datetime:
    return llr._quantize_utc_for_horizons(dt)  # type: ignore[attr-defined]


TimeTagMode = Literal["tx", "rx", "mid", "auto"]


def _time_tag_mode_env() -> str:
    mode = os.environ.get("LLR_TIME_TAG", "").strip().lower()
    return mode


def _build_tx_b_rx(tag_times: List[datetime], tof_s: np.ndarray, mode: str) -> Tuple[List[datetime], List[datetime], List[datetime]]:
    def _sec(x: float) -> timedelta:
        return timedelta(seconds=float(x))

    tx_times: List[datetime] = []
    b_times: List[datetime] = []
    rx_times: List[datetime] = []
    for t_tag, tof in zip(tag_times, tof_s):
        # 条件分岐: `mode == "tx"` を満たす経路を評価する。
        if mode == "tx":
            t_tx = t_tag
            t_b = t_tag + _sec(tof / 2.0)
            t_rx = t_tag + _sec(tof)
        # 条件分岐: 前段条件が不成立で、`mode == "rx"` を追加評価する。
        elif mode == "rx":
            t_rx = t_tag
            t_b = t_tag - _sec(tof / 2.0)
            t_tx = t_tag - _sec(tof)
        # 条件分岐: 前段条件が不成立で、`mode == "mid"` を追加評価する。
        elif mode == "mid":
            t_b = t_tag
            t_tx = t_tag - _sec(tof / 2.0)
            t_rx = t_tag + _sec(tof / 2.0)
        else:
            raise ValueError(f"Invalid LLR_TIME_TAG={mode!r} (expected tx/rx/mid)")

        tx_times.append(_quantize(t_tx))
        b_times.append(_quantize(t_b))
        rx_times.append(_quantize(t_rx))

    return tx_times, b_times, rx_times


def _to_vec_map(vdf: pd.DataFrame) -> Dict[datetime, np.ndarray]:
    out: Dict[datetime, np.ndarray] = {}
    for r in vdf.itertuples(index=False):
        t = getattr(r, "epoch_utc")
        # 条件分岐: `isinstance(t, pd.Timestamp)` を満たす経路を評価する。
        if isinstance(t, pd.Timestamp):
            t = t.to_pydatetime()

        t = _quantize(t)
        out[t] = np.array([getattr(r, "x_km"), getattr(r, "y_km"), getattr(r, "z_km")], dtype=float) * 1000.0

    return out


def _safe_id(s: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "", (s or "").strip().lower()) or "na"


def _rms_ns(arr: np.ndarray) -> float:
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.sqrt(np.mean(a * a))) if len(a) else float("nan")


def _robust_inlier_mask_ns(delta_ns: np.ndarray, *, clip_sigma: float, clip_min_ns: float) -> np.ndarray:
    """
    Robust outlier gate around the median using MAD.

    delta_ns: (obs - model) in nanoseconds; may include NaN/inf.
    """
    x = np.asarray(delta_ns, dtype=float)
    ok = np.isfinite(x)
    # 条件分岐: `not np.any(ok)` を満たす経路を評価する。
    if not np.any(ok):
        return ok

    x0 = x[ok]
    # 条件分岐: `len(x0) < 20` を満たす経路を評価する。
    if len(x0) < 20:
        # Too few points for robust stats; keep all finite.
        return ok

    med = float(np.median(x0))
    mad = float(np.median(np.abs(x0 - med)))
    sigma = 1.4826 * mad if np.isfinite(mad) else float("nan")
    thr = float(clip_min_ns)
    # 条件分岐: `np.isfinite(sigma) and sigma > 0` を満たす経路を評価する。
    if np.isfinite(sigma) and sigma > 0:
        thr = max(thr, float(clip_sigma) * sigma)

    return ok & (np.abs(x - med) <= thr)


def _offset_align_residual_ns(
    obs_s: np.ndarray,
    pred_s: np.ndarray,
    *,
    inlier_mask: np.ndarray,
) -> np.ndarray:
    """
    obs_s, pred_s: seconds
    inlier_mask: boolean mask (same length) selecting points used for offset fit and RMS.

    Returns residuals in ns, with outliers/non-finite points set to NaN.
    """
    obs = np.asarray(obs_s, dtype=float)
    pred = np.asarray(pred_s, dtype=float)
    inl = np.asarray(inlier_mask, dtype=bool)
    inl = inl & np.isfinite(obs) & np.isfinite(pred)
    # 条件分岐: `not np.any(inl)` を満たす経路を評価する。
    if not np.any(inl):
        return np.full_like(obs, np.nan, dtype=float)

    k = float(np.mean(obs[inl] - pred[inl]))
    res = np.full_like(obs, np.nan, dtype=float)
    res[inl] = (obs[inl] - (pred[inl] + k)) * 1e9
    return res


def _offset_align_residual_all_ns(
    obs_s: np.ndarray,
    pred_s: np.ndarray,
    *,
    inlier_mask: np.ndarray,
) -> np.ndarray:
    """
    Like _offset_align_residual_ns, but returns residuals for *all* finite points
    using an offset k fitted on inliers.

    This is used only for diagnostics (e.g., outlier root-cause) and must NOT be
    used for RMS aggregation unless outliers are explicitly gated.
    """
    obs = np.asarray(obs_s, dtype=float)
    pred = np.asarray(pred_s, dtype=float)
    inl = np.asarray(inlier_mask, dtype=bool)
    inl = inl & np.isfinite(obs) & np.isfinite(pred)
    # 条件分岐: `not np.any(inl)` を満たす経路を評価する。
    if not np.any(inl):
        return np.full_like(obs, np.nan, dtype=float)

    k = float(np.mean(obs[inl] - pred[inl]))
    res = np.full_like(obs, np.nan, dtype=float)
    ok = np.isfinite(obs) & np.isfinite(pred)
    res[ok] = (obs[ok] - (pred[ok] + k)) * 1e9
    return res


def _filter_llr_rows_by_tof(
    all_df: pd.DataFrame,
    *,
    tof_min_s: float = 1.0,
    tof_max_s: float = 4.0,
    emit_bad_rows_csv: Optional[Path] = None,
) -> pd.DataFrame:
    """
    EDCのNP/NP2には稀に、LLR往復TOFとして明らかに不正な値が混入することがある。
    例: TOF ~ 1e-8 s（nsオーダー）は物理的にあり得ず、RMSを破壊する。

    ここでは「評価用に」妥当域 (tof_min_s, tof_max_s) のみ残す。
    """
    # 条件分岐: `all_df.empty` を満たす経路を評価する。
    if all_df.empty:
        return all_df

    tof = pd.to_numeric(all_df.get("tof_obs_s"), errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(tof) & (tof > float(tof_min_s)) & (tof < float(tof_max_s))
    # 条件分岐: `bool(np.all(ok))` を満たす経路を評価する。
    if bool(np.all(ok)):
        return all_df

    bad_df = all_df.loc[~ok].copy()
    keep_df = all_df.loc[ok].copy()

    # 条件分岐: `emit_bad_rows_csv is not None` を満たす経路を評価する。
    if emit_bad_rows_csv is not None:
        try:
            emit_bad_rows_csv.parent.mkdir(parents=True, exist_ok=True)
            cols = [
                c
                for c in [
                    "source_file",
                    "file",
                    "lineno",
                    "station",
                    "target",
                    "range_type",
                    "epoch_utc",
                    "seconds_of_day",
                    "tof_obs_s",
                ]
                if c in bad_df.columns
            ]
            bad_df[cols].to_csv(emit_bad_rows_csv, index=False)
        except Exception:
            pass

    return keep_df.reset_index(drop=True)


@dataclass(frozen=True)
class ModePrediction:
    mode: str
    has_station: np.ndarray  # (n,)
    has_reflector: np.ndarray  # (n,)
    tof_gc_raw_s: np.ndarray  # (n,)
    tof_sm_raw_s: np.ndarray  # (n,)
    tof_sr_raw_s: np.ndarray  # (n,)
    tof_sr_raw_no_shapiro_s: np.ndarray  # (n,)
    tof_sr_raw_tropo_s: np.ndarray  # (n,)
    tof_sr_raw_tropo_station_tide_s: np.ndarray  # (n,)
    tof_sr_raw_tropo_moon_tide_s: np.ndarray  # (n,)
    tof_sr_raw_tropo_tide_no_ocean_s: np.ndarray  # (n,)
    tof_sr_raw_tropo_tide_s: np.ndarray  # (n,)
    tof_sr_raw_tropo_no_shapiro_s: np.ndarray  # (n,)
    tof_sr_raw_tropo_earth_shapiro_s: np.ndarray  # (n,)
    tof_sr_raw_iau_s: np.ndarray  # (n,)
    tof_sr_raw_iau_no_shapiro_s: np.ndarray  # (n,)
    elev_up_deg: np.ndarray  # (n,)
    elev_dn_deg: np.ndarray  # (n,)


def _compute_predictions_for_mode(
    *,
    root: Path,
    all_df: pd.DataFrame,
    tof_obs_s: np.ndarray,
    mode: str,
    beta: float,
    offline: bool,
    chunk: int,
    cache_dir: Path,
    station_xyz_override: Optional[Dict[str, Dict[str, Any]]] = None,
    ocean_model: Optional[ol.HarposModel] = None,
    ocean_site_by_station: Optional[Dict[str, str]] = None,
) -> ModePrediction:
    # Compute tx/bounce/rx timestamps for this mode
    tag_times = [t.to_pydatetime() if isinstance(t, pd.Timestamp) else t for t in all_df["epoch_utc"].tolist()]
    tx_times, b_times, rx_times = _build_tx_b_rx(tag_times, tof_obs_s, mode=mode)

    # Union times (Earth->Moon / Earth->Sun)
    times_all = sorted({t.astimezone(timezone.utc) for t in (tx_times + b_times + rx_times)})
    moon_all = llr.fetch_vectors_chunked_cached(
        "301",
        "500@399",
        times_all,
        chunk=chunk,
        cache_dir=cache_dir,
        offline=offline,
        ref_plane="FRAME",
    )
    sun_all = llr.fetch_vectors_chunked_cached(
        "10",
        "500@399",
        times_all,
        chunk=chunk,
        cache_dir=cache_dir,
        offline=offline,
        ref_plane="FRAME",
    )
    moon_map = _to_vec_map(moon_all)
    sun_map = _to_vec_map(sun_all)

    # Station topocentric: station->Moon at tx/rx, per station (site log required)
    stations = sorted({s for s in all_df["station"].unique() if s and s.lower() not in ("na", "nan")})
    station_meta: Dict[str, Optional[Dict[str, Any]]] = {s: llr._load_station_geodetic(root, s) for s in stations}  # type: ignore[attr-defined]
    # 条件分岐: `station_xyz_override` を満たす経路を評価する。
    if station_xyz_override:
        for st, override in station_xyz_override.items():
            # 条件分岐: `st not in station_meta` を満たす経路を評価する。
            if st not in station_meta:
                continue

            meta0 = station_meta.get(st)
            # 条件分岐: `not isinstance(meta0, dict) or not isinstance(override, dict)` を満たす経路を評価する。
            if not isinstance(meta0, dict) or not isinstance(override, dict):
                continue

            # 条件分岐: `not all(k in override for k in ("x_m", "y_m", "z_m"))` を満たす経路を評価する。

            if not all(k in override for k in ("x_m", "y_m", "z_m")):
                continue

            meta = dict(meta0)
            meta["x_m"] = float(override["x_m"])
            meta["y_m"] = float(override["y_m"])
            meta["z_m"] = float(override["z_m"])
            # 条件分岐: `override.get("pos_eop_yymmdd")` を満たす経路を評価する。
            if override.get("pos_eop_yymmdd"):
                meta["station_coord_source"] = "EDC pos+eop (SINEX)"
                meta["pos_eop_yymmdd"] = str(override.get("pos_eop_yymmdd"))
                # 条件分岐: `override.get("pos_eop_ref_epoch_utc")` を満たす経路を評価する。
                if override.get("pos_eop_ref_epoch_utc"):
                    meta["pos_eop_ref_epoch_utc"] = str(override.get("pos_eop_ref_epoch_utc"))
            else:
                meta["station_coord_source"] = "override"

            station_meta[st] = meta

    station_moon_map: Dict[str, Dict[datetime, np.ndarray]] = {}
    for st in stations:
        meta = station_meta.get(st) or None
        # 条件分岐: `not meta` を満たす経路を評価する。
        if not meta:
            continue

        coord_type = "GEODETIC"
        # 条件分岐: `all(meta.get(k) is not None for k in ("x_m", "y_m", "z_m"))` を満たす経路を評価する。
        if all(meta.get(k) is not None for k in ("x_m", "y_m", "z_m")):
            lon_deg, lat_deg, h_m = llr.geodetic_from_ecef(float(meta["x_m"]), float(meta["y_m"]), float(meta["z_m"]))
            site_coord = f"{lon_deg:.10f},{lat_deg:.10f},{h_m/1000.0:.6f}"
        # 条件分岐: 前段条件が不成立で、`all(k in meta for k in ("lat_deg", "lon_deg", "height_m"))` を追加評価する。
        elif all(k in meta for k in ("lat_deg", "lon_deg", "height_m")):
            site_coord = f"{float(meta['lon_deg']):.10f},{float(meta['lat_deg']):.10f},{float(meta['height_m'])/1000.0:.6f}"
        else:
            continue

        st_mask = all_df["station"].to_numpy() == st
        st_tx = [tx_times[i] for i, ok in enumerate(st_mask) if ok]
        st_rx = [rx_times[i] for i, ok in enumerate(st_mask) if ok]
        times_sm = sorted({t.astimezone(timezone.utc) for t in (st_tx + st_rx)})
        # 条件分岐: `not times_sm` を満たす経路を評価する。
        if not times_sm:
            continue

        moon_site = llr.fetch_vectors_chunked_cached(
            "301",
            "coord@399",
            times_sm,
            chunk=chunk,
            cache_dir=cache_dir,
            offline=offline,
            ref_plane="FRAME",
            coord_type=coord_type,
            site_coord=site_coord,
        )
        station_moon_map[st] = _to_vec_map(moon_site)

    # Vectorize core ephemerides lookups

    n = len(all_df)
    r_em_tx = np.stack([moon_map[t] for t in tx_times], axis=0)
    r_em_rx = np.stack([moon_map[t] for t in rx_times], axis=0)
    r_em_b = np.stack([moon_map[t] for t in b_times], axis=0)
    r_es_tx = np.stack([sun_map[t] for t in tx_times], axis=0)
    r_es_rx = np.stack([sun_map[t] for t in rx_times], axis=0)
    r_es_b = np.stack([sun_map[t] for t in b_times], axis=0)

    # Station vectors (fallback: geocenter)
    r_st_tx = np.zeros_like(r_em_tx)
    r_st_rx = np.zeros_like(r_em_rx)
    has_station = np.zeros((n,), dtype=bool)
    stations_list = all_df["station"].tolist()
    for i, st in enumerate(stations_list):
        sm = station_moon_map.get(st)
        # 条件分岐: `sm is None` を満たす経路を評価する。
        if sm is None:
            continue

        try:
            r_sm_tx = sm[tx_times[i]]
            r_sm_rx = sm[rx_times[i]]
        except KeyError:
            continue

        r_st_tx[i] = r_em_tx[i] - r_sm_tx
        r_st_rx[i] = r_em_rx[i] - r_sm_rx
        has_station[i] = True

    # Distances (geocenter / station->moon)

    r_moon_b = r_em_b
    up_gc = np.linalg.norm(r_moon_b, axis=1)
    down_gc = up_gc.copy()
    up_sm = np.linalg.norm(r_moon_b - r_st_tx, axis=1)
    down_sm = np.linalg.norm(r_moon_b - r_st_rx, axis=1)

    # Reflector: station->reflector (Moon PA DE421 -> J2000)
    refl_cat = _read_json(root / "data" / "llr" / "reflectors_de421_pa.json")
    refls = refl_cat.get("reflectors") or {}

    r_refl_b = np.full_like(r_moon_b, np.nan)  # SPICE(MOON_PA_DE421) ※fallbackあり
    r_refl_b_iau = np.full_like(r_moon_b, np.nan)  # IAU近似（比較用）
    has_reflector = np.zeros((n,), dtype=bool)

    # Cache rotation matrices by time to avoid repeated calls
    rot_cache_spice: Dict[datetime, Optional[np.ndarray]] = {}
    rot_cache_iau: Dict[datetime, np.ndarray] = {}

    targets_list = all_df["target"].tolist()
    for target in sorted(set(targets_list)):
        meta = refls.get(target)
        # 条件分岐: `not isinstance(meta, dict) or not all(k in meta for k in ("x_m", "y_m", "z_m"))` を満たす経路を評価する。
        if not isinstance(meta, dict) or not all(k in meta for k in ("x_m", "y_m", "z_m")):
            continue

        pa = np.array([float(meta["x_m"]), float(meta["y_m"]), float(meta["z_m"])], dtype=float)

        # indices for this target
        idx = [i for i, t in enumerate(targets_list) if t == target]
        # 条件分岐: `not idx` を満たす経路を評価する。
        if not idx:
            continue

        mats_spice: List[np.ndarray] = []
        mats_iau: List[np.ndarray] = []
        for i in idx:
            t = b_times[i]
            mat_iau = rot_cache_iau.get(t)
            # 条件分岐: `mat_iau is None` を満たす経路を評価する。
            if mat_iau is None:
                mat_iau = llr.moon_pa_to_icrf_matrix(t)
                rot_cache_iau[t] = mat_iau

            mat_sp = rot_cache_spice.get(t)
            # 条件分岐: `mat_sp is None and t not in rot_cache_spice` を満たす経路を評価する。
            if mat_sp is None and t not in rot_cache_spice:
                mat_sp = llr._moon_pa_de421_to_j2000_matrix(root, t)  # type: ignore[attr-defined]
                rot_cache_spice[t] = mat_sp

            mat_sp_eff = mat_iau if mat_sp is None else mat_sp

            mats_spice.append(mat_sp_eff)
            mats_iau.append(mat_iau)

        rot_sp = np.stack(mats_spice, axis=0)  # (k,3,3)
        rot_iau = np.stack(mats_iau, axis=0)
        refl_icrf_sp = rot_sp @ pa  # (k,3)
        refl_icrf_iau = rot_iau @ pa

        r_refl_b[idx, :] = r_em_b[idx, :] + refl_icrf_sp
        r_refl_b_iau[idx, :] = r_em_b[idx, :] + refl_icrf_iau
        has_reflector[idx] = True

    up_sr = np.linalg.norm(r_refl_b - r_st_tx, axis=1)
    down_sr = np.linalg.norm(r_refl_b - r_st_rx, axis=1)
    up_sr_iau = np.linalg.norm(r_refl_b_iau - r_st_tx, axis=1)
    down_sr_iau = np.linalg.norm(r_refl_b_iau - r_st_rx, axis=1)

    # Two-way geometric TOF
    tof_geo_gc = (up_gc + down_gc) / llr.C
    tof_geo_sm = (up_sm + down_sm) / llr.C
    tof_geo_sr = (up_sr + down_sr) / llr.C
    tof_geo_sr_iau = (up_sr_iau + down_sr_iau) / llr.C

    # Sun Shapiro
    coeff = 2.0 * beta
    r2_gc = np.linalg.norm(r_es_b - r_moon_b, axis=1)
    dt_up_gc = llr.shapiro_oneway_sun(llr.GM_SUN, np.linalg.norm(r_es_tx, axis=1), r2_gc, up_gc, coeff=coeff)
    dt_dn_gc = llr.shapiro_oneway_sun(llr.GM_SUN, np.linalg.norm(r_es_rx, axis=1), r2_gc, down_gc, coeff=coeff)
    dt_two_gc = dt_up_gc + dt_dn_gc
    tof_gc_raw = tof_geo_gc + dt_two_gc

    r1_sm_tx = np.linalg.norm(r_es_tx - r_st_tx, axis=1)
    r1_sm_rx = np.linalg.norm(r_es_rx - r_st_rx, axis=1)
    r2_sm = np.linalg.norm(r_es_b - r_moon_b, axis=1)
    dt_up_sm = llr.shapiro_oneway_sun(llr.GM_SUN, r1_sm_tx, r2_sm, up_sm, coeff=coeff)
    dt_dn_sm = llr.shapiro_oneway_sun(llr.GM_SUN, r1_sm_rx, r2_sm, down_sm, coeff=coeff)
    dt_two_sm = dt_up_sm + dt_dn_sm
    tof_sm_raw = tof_geo_sm + dt_two_sm

    r2_sr = np.linalg.norm(r_es_b - r_refl_b, axis=1)
    dt_up_sr_sun = llr.shapiro_oneway_sun(llr.GM_SUN, r1_sm_tx, r2_sr, up_sr, coeff=coeff)
    dt_dn_sr_sun = llr.shapiro_oneway_sun(llr.GM_SUN, r1_sm_rx, r2_sr, down_sr, coeff=coeff)
    dt_two_sr_sun = dt_up_sr_sun + dt_dn_sr_sun
    tof_sr_raw = tof_geo_sr + dt_two_sr_sun

    # Troposphere (Saastamoinen; use CRD record 20 when available) + simple mapping
    # NOTE:
    #   - LLR NP2 may already include some atmospheric correction; we keep this as an explicit term so
    #     we can quantify whether it reduces station-dependent residuals or double-counts.
    #   - We first use per-normal-point meteo (pressure/temperature/humidity). If missing, fall back to
    #     a standard-atmosphere estimate from station height.
    p_hpa = all_df["pressure_hpa"].to_numpy(dtype=float) if "pressure_hpa" in all_df.columns else np.full((n,), np.nan)
    t_k = all_df["temp_k"].to_numpy(dtype=float) if "temp_k" in all_df.columns else np.full((n,), np.nan)
    rh = all_df["rh_percent"].to_numpy(dtype=float) if "rh_percent" in all_df.columns else np.full((n,), np.nan)

    station_atmos: Dict[str, Dict[str, float]] = {}
    for st, meta in station_meta.items():
        # 条件分岐: `not meta` を満たす経路を評価する。
        if not meta:
            continue

        try:
            lat_deg = float(meta.get("lat_deg")) if meta.get("lat_deg") is not None else float("nan")
            h_m = float(meta.get("height_m")) if meta.get("height_m") is not None else float("nan")
        except Exception:
            continue

        # 条件分岐: `not (np.isfinite(lat_deg) and np.isfinite(h_m))` を満たす経路を評価する。

        if not (np.isfinite(lat_deg) and np.isfinite(h_m)):
            continue

        phi = float(np.deg2rad(lat_deg))
        # Standard atmosphere pressure at height (hPa), used as fallback
        try:
            P_std_hpa = float(1013.25 * (1.0 - 2.25577e-5 * h_m) ** 5.25588)
        except Exception:
            P_std_hpa = float("nan")

        station_atmos[str(st)] = {"phi_rad": phi, "h_m": float(h_m), "P_std_hpa": P_std_hpa}

    # Per-station fallback meteorology (derived from CRD record 20 when available).
    # Many stations have sparse meteo coverage; using station medians as fallback helps avoid
    # systematically skipping wet delay on points with missing temperature/humidity.

    try:
        stations_arr = all_df["station"].fillna("").astype(str).to_numpy(dtype=object)
    except Exception:
        stations_arr = np.array([""] * n, dtype=object)

    for st in list(station_atmos.keys()):
        mask = stations_arr == str(st)
        # 条件分岐: `not bool(np.any(mask))` を満たす経路を評価する。
        if not bool(np.any(mask)):
            station_atmos[str(st)]["met_frac"] = 0.0
            station_atmos[str(st)]["P_med_hpa"] = float("nan")
            station_atmos[str(st)]["T_med_k"] = float("nan")
            station_atmos[str(st)]["RH_med_percent"] = float("nan")
            continue

        P_vals = p_hpa[mask]
        T_vals = t_k[mask]
        RH_vals = rh[mask]

        met_ok = np.isfinite(P_vals) & np.isfinite(T_vals) & np.isfinite(RH_vals)
        station_atmos[str(st)]["met_frac"] = float(np.mean(met_ok)) if int(mask.sum()) else 0.0

        P_ok = P_vals[np.isfinite(P_vals) & (P_vals >= 100.0) & (P_vals <= 1100.0)]
        T_ok = T_vals[np.isfinite(T_vals) & (T_vals >= 150.0) & (T_vals <= 330.0)]
        RH_ok = RH_vals[np.isfinite(RH_vals) & (RH_vals >= 0.0) & (RH_vals <= 100.0)]

        station_atmos[str(st)]["P_med_hpa"] = float(np.median(P_ok)) if len(P_ok) else float("nan")
        station_atmos[str(st)]["T_med_k"] = float(np.median(T_ok)) if len(T_ok) else float("nan")
        station_atmos[str(st)]["RH_med_percent"] = float(np.median(RH_ok)) if len(RH_ok) else float("nan")

    def _saast_zhd_m(pressure_hpa: float, phi_rad: float, height_m: float) -> float:
        # Saastamoinen hydrostatic zenith delay [m]
        h_km = height_m / 1000.0
        denom = 1.0 - 0.00266 * float(np.cos(2.0 * phi_rad)) - 0.00028 * h_km
        # 条件分岐: `denom <= 0` を満たす経路を評価する。
        if denom <= 0:
            return float("nan")

        return float(0.0022768 * pressure_hpa / denom)

    def _sat_vapor_pressure_hpa(temp_k: float) -> float:
        # Tetens formula (hPa)
        t_c = float(temp_k - 273.15)
        return float(6.112 * np.exp((17.67 * t_c) / (t_c + 243.5)))

    def _saast_zwd_m(temp_k: float, rh_percent: float) -> float:
        # Saastamoinen wet zenith delay [m]
        if not (np.isfinite(temp_k) and np.isfinite(rh_percent)):
            return 0.0

        # 条件分岐: `temp_k < 150.0 or temp_k > 330.0` を満たす経路を評価する。

        if temp_k < 150.0 or temp_k > 330.0:
            return 0.0

        # 条件分岐: `rh_percent < 0.0 or rh_percent > 100.0` を満たす経路を評価する。

        if rh_percent < 0.0 or rh_percent > 100.0:
            return 0.0

        e_hpa = (float(rh_percent) / 100.0) * _sat_vapor_pressure_hpa(float(temp_k))
        return float(0.002277 * (1255.0 / float(temp_k) + 0.05) * e_hpa)

    dt_tropo_two_way_s = np.zeros((n,), dtype=float)
    dt_tides_two_way_s = np.zeros((n,), dtype=float)
    dt_tides_no_ocean_two_way_s = np.zeros((n,), dtype=float)
    dt_station_tide_two_way_s = np.zeros((n,), dtype=float)
    dt_moon_tide_two_way_s = np.zeros((n,), dtype=float)
    dt_ocean_loading_two_way_s = np.zeros((n,), dtype=float)
    elev_up_deg = np.full((n,), np.nan, dtype=float)
    elev_dn_deg = np.full((n,), np.nan, dtype=float)
    min_sin_el = float(np.sin(np.deg2rad(5.0)))

    # ------------------------------------------------------------
    # Troposphere mapping function: Niell Mapping Function (NMF)
    # ------------------------------------------------------------
    # Replace the simple 1/sin(E) mapping with Niell (1996)-style hydrostatic/wet mapping
    # + height correction (see e.g. Orekit NiellMappingFunctionModel).
    _NMF_LAT_DEG = [15.0, 30.0, 45.0, 60.0, 75.0]
    _NMF_LAT_RAD = [float(np.deg2rad(x)) for x in _NMF_LAT_DEG]

    _AH_AVG = [1.2769934e-3, 1.2683230e-3, 1.2465397e-3, 1.2196049e-3, 1.2045996e-3]
    _BH_AVG = [2.9153695e-3, 2.9152299e-3, 2.9288445e-3, 2.9022565e-3, 2.9024912e-3]
    _CH_AVG = [62.610505e-3, 62.837393e-3, 63.721774e-3, 63.824265e-3, 64.258455e-3]

    _AH_AMP = [0.0, 1.2709626e-5, 2.6523662e-5, 3.4000452e-5, 4.1202191e-5]
    _BH_AMP = [0.0, 2.1414979e-5, 3.0160779e-5, 7.2562722e-5, 11.723375e-5]
    _CH_AMP = [0.0, 9.0128400e-5, 4.3497037e-5, 84.795348e-5, 170.37206e-5]

    _AW = [5.8021897e-4, 5.6794847e-4, 5.8118019e-4, 5.9727542e-4, 6.1641693e-4]
    _BW = [1.4275268e-3, 1.5138625e-3, 1.4572752e-3, 1.5007428e-3, 1.7599082e-3]
    _CW = [4.3472961e-2, 4.6729510e-2, 4.3908931e-2, 4.4626982e-2, 5.4736038e-2]

    _A_HT = 2.53e-5
    _B_HT = 5.49e-3
    _C_HT = 1.14e-3

    def _interp_lat(abs_lat_rad: float, values: list[float]) -> float:
        x = float(abs_lat_rad)
        # 条件分岐: `x <= _NMF_LAT_RAD[0]` を満たす経路を評価する。
        if x <= _NMF_LAT_RAD[0]:
            return float(values[0])

        # 条件分岐: `x >= _NMF_LAT_RAD[-1]` を満たす経路を評価する。

        if x >= _NMF_LAT_RAD[-1]:
            return float(values[-1])

        for j in range(len(_NMF_LAT_RAD) - 1):
            x0 = _NMF_LAT_RAD[j]
            x1 = _NMF_LAT_RAD[j + 1]
            # 条件分岐: `x0 <= x <= x1` を満たす経路を評価する。
            if x0 <= x <= x1:
                w = (x - x0) / max(x1 - x0, 1e-15)
                return float(values[j] * (1.0 - w) + values[j + 1] * w)

        return float(values[-1])

    def _marini_mapping_from_sin(sin_e: float, a: float, b: float, c: float) -> float:
        # Marini (1972) mapping normalized to 1 at zenith (Niell, 1996)
        s = float(sin_e)
        # Prevent division by ~0 at very low elevations (already clamped by min_sin_el).
        s = max(s, float(min_sin_el))
        num = 1.0 + a / (1.0 + b / (1.0 + c))
        den = s + a / (s + b / (s + c))
        return float(num / den)

    def _nmf_mapping_factors(*, sin_e: float, lat_rad: float, height_m: float, dt_utc: datetime) -> tuple[float, float]:
        # Seasonal term (day of year)
        doy = float(dt_utc.timetuple().tm_yday)
        t0 = 28.0 + (183.0 if float(lat_rad) < 0.0 else 0.0)
        cos_coef = float(np.cos(2.0 * np.pi * ((doy - t0) / 365.25)))

        abs_lat = float(abs(lat_rad))
        abs_lat = float(max(_NMF_LAT_RAD[0], min(_NMF_LAT_RAD[-1], abs_lat)))

        ah = _interp_lat(abs_lat, _AH_AVG) - _interp_lat(abs_lat, _AH_AMP) * cos_coef
        bh = _interp_lat(abs_lat, _BH_AVG) - _interp_lat(abs_lat, _BH_AMP) * cos_coef
        ch = _interp_lat(abs_lat, _CH_AVG) - _interp_lat(abs_lat, _CH_AMP) * cos_coef

        mh = _marini_mapping_from_sin(sin_e, float(ah), float(bh), float(ch))
        mw = _marini_mapping_from_sin(
            sin_e,
            _interp_lat(abs_lat, _AW),
            _interp_lat(abs_lat, _BW),
            _interp_lat(abs_lat, _CW),
        )

        # Height correction applies to hydrostatic mapping only (Niell 1996).
        h = max(0.0, float(height_m))
        corr = ((1.0 / max(float(sin_e), float(min_sin_el))) - _marini_mapping_from_sin(sin_e, _A_HT, _B_HT, _C_HT)) * (h / 1000.0)
        mh = float(mh + corr)
        return (mh, mw)

    # Solid Earth tide (simple, but includes horizontal component): driven by Moon+Sun.
    # Goal: reduce station-dependent systematic residuals at the ns level.

    GM_MOON = 4.9048695e12  # m^3/s^2
    R_E = 6378137.0  # m (WGS84)
    H2_EARTH = 0.6078
    L2_EARTH = 0.0847
    G0 = 9.80665  # m/s^2

    # Lunar body tide at reflector (very simplified): driven mainly by Earth.
    GM_EARTH = 3.986004418e14  # m^3/s^2
    R_MOON = 1737400.0  # m
    H2_MOON = 0.04
    L2_MOON = 0.01
    G_MOON = GM_MOON / (R_MOON**2)

    def _tide_disp_vec(
        *,
        u_r: np.ndarray,
        r_body: np.ndarray,
        GM_body: float,
        R_ref: float,
        h2: float,
        l2: float,
        g_ref: float,
    ) -> np.ndarray:
        rb = float(np.linalg.norm(r_body))
        # 条件分岐: `not np.isfinite(rb) or rb <= 0` を満たす経路を評価する。
        if not np.isfinite(rb) or rb <= 0:
            return np.zeros((3,), dtype=float)

        u_b = r_body / rb
        cospsi = float(np.dot(u_r, u_b))
        cospsi = max(min(cospsi, 1.0), -1.0)
        sinpsi = float(np.sqrt(max(1.0 - cospsi * cospsi, 1e-12)))
        p2 = 0.5 * (3.0 * cospsi * cospsi - 1.0)
        scale = float((GM_body / (rb**3)) * (R_ref**2) / g_ref)

        # Radial displacement
        dr = float(h2 * scale * p2)
        d = dr * u_r

        # Horizontal displacement (direction towards the body projection)
        u_t = (u_b - cospsi * u_r) / sinpsi
        dt = float(3.0 * l2 * scale * cospsi * sinpsi)
        d = d + dt * u_t
        return d

    for i, st in enumerate(stations_list):
        # 条件分岐: `not (has_station[i] and has_reflector[i])` を満たす経路を評価する。
        if not (has_station[i] and has_reflector[i]):
            continue

        # 条件分岐: `not (np.isfinite(up_sr[i]) and np.isfinite(down_sr[i]))` を満たす経路を評価する。

        if not (np.isfinite(up_sr[i]) and np.isfinite(down_sr[i])):
            continue

        meta_atm = station_atmos.get(str(st))
        # 条件分岐: `not meta_atm` を満たす経路を評価する。
        if not meta_atm:
            continue

        phi = float(meta_atm["phi_rad"])
        h_m = float(meta_atm["h_m"])

        P = float(p_hpa[i]) if np.isfinite(p_hpa[i]) else float("nan")
        # 条件分岐: `not (np.isfinite(P) and 100.0 <= P <= 1100.0)` を満たす経路を評価する。
        if not (np.isfinite(P) and 100.0 <= P <= 1100.0):
            P_med = float(meta_atm.get("P_med_hpa", float("nan")))
            # 条件分岐: `np.isfinite(P_med) and 100.0 <= P_med <= 1100.0` を満たす経路を評価する。
            if np.isfinite(P_med) and 100.0 <= P_med <= 1100.0:
                P = P_med
            else:
                P = float(meta_atm.get("P_std_hpa", float("nan")))

        # 条件分岐: `not np.isfinite(P)` を満たす経路を評価する。

        if not np.isfinite(P):
            continue

        # Wet delay requires temperature + humidity; if missing, fall back to station medians
        # when available (otherwise keep NaN so zwd becomes 0).

        T = float(t_k[i]) if np.isfinite(t_k[i]) else float("nan")
        RH = float(rh[i]) if np.isfinite(rh[i]) else float("nan")
        # 条件分岐: `not np.isfinite(T)` を満たす経路を評価する。
        if not np.isfinite(T):
            T = float(meta_atm.get("T_med_k", float("nan")))

        # 条件分岐: `not np.isfinite(RH)` を満たす経路を評価する。

        if not np.isfinite(RH):
            RH = float(meta_atm.get("RH_med_percent", float("nan")))

        zhd_m = _saast_zhd_m(P, phi, h_m)
        zwd_m = _saast_zwd_m(T, RH)
        # 条件分岐: `not np.isfinite(zhd_m)` を満たす経路を評価する。
        if not np.isfinite(zhd_m):
            continue

        # Local zenith (radial)

        zen_tx = r_st_tx[i] / max(float(np.linalg.norm(r_st_tx[i])), 1e-9)
        zen_rx = r_st_rx[i] / max(float(np.linalg.norm(r_st_rx[i])), 1e-9)
        u_up = (r_refl_b[i] - r_st_tx[i]) / max(float(up_sr[i]), 1e-9)
        u_dn = (r_refl_b[i] - r_st_rx[i]) / max(float(down_sr[i]), 1e-9)

        sin_el_up = max(float(np.dot(u_up, zen_tx)), min_sin_el)
        sin_el_dn = max(float(np.dot(u_dn, zen_rx)), min_sin_el)
        # Store the geometric elevation angle (before clamping) for diagnostics.
        try:
            elev_up_deg[i] = float(
                np.rad2deg(np.arcsin(np.clip(float(np.dot(u_up, zen_tx)), -1.0, 1.0)))
            )
            elev_dn_deg[i] = float(
                np.rad2deg(np.arcsin(np.clip(float(np.dot(u_dn, zen_rx)), -1.0, 1.0)))
            )
        except Exception:
            pass

        # Niell mapping factors (hydrostatic + wet) for each leg

        mh_up, mw_up = _nmf_mapping_factors(sin_e=sin_el_up, lat_rad=phi, height_m=h_m, dt_utc=tx_times[i])
        mh_dn, mw_dn = _nmf_mapping_factors(sin_e=sin_el_dn, lat_rad=phi, height_m=h_m, dt_utc=rx_times[i])
        dt_tropo_two_way_s[i] = (zhd_m * (mh_up + mh_dn) + zwd_m * (mw_up + mw_dn)) / llr.C

        # Station solid Earth tide (Moon+Sun)
        try:
            d_tx = _tide_disp_vec(
                u_r=zen_tx,
                r_body=r_em_tx[i],
                GM_body=GM_MOON,
                R_ref=R_E,
                h2=H2_EARTH,
                l2=L2_EARTH,
                g_ref=G0,
            ) + _tide_disp_vec(
                u_r=zen_tx,
                r_body=r_es_tx[i],
                GM_body=llr.GM_SUN,
                R_ref=R_E,
                h2=H2_EARTH,
                l2=L2_EARTH,
                g_ref=G0,
            )
            d_rx = _tide_disp_vec(
                u_r=zen_rx,
                r_body=r_em_rx[i],
                GM_body=GM_MOON,
                R_ref=R_E,
                h2=H2_EARTH,
                l2=L2_EARTH,
                g_ref=G0,
            ) + _tide_disp_vec(
                u_r=zen_rx,
                r_body=r_es_rx[i],
                GM_body=llr.GM_SUN,
                R_ref=R_E,
                h2=H2_EARTH,
                l2=L2_EARTH,
                g_ref=G0,
            )
            dt_station = -(float(np.dot(u_up, d_tx)) + float(np.dot(u_dn, d_rx))) / llr.C
        except Exception:
            dt_station = 0.0

        # Moon body tide at reflector (Earth-driven; very simplified)

        try:
            r_refl_m = r_refl_b[i] - r_em_b[i]
            u_r_m = r_refl_m / max(float(np.linalg.norm(r_refl_m)), 1e-9)
            r_me = -r_em_b[i]  # Moon->Earth
            d_moon = _tide_disp_vec(
                u_r=u_r_m,
                r_body=r_me,
                GM_body=GM_EARTH,
                R_ref=R_MOON,
                h2=H2_MOON,
                l2=L2_MOON,
                g_ref=G_MOON,
            )
            dt_moon = (float(np.dot(u_up, d_moon)) + float(np.dot(u_dn, d_moon))) / llr.C
        except Exception:
            dt_moon = 0.0

        # Ocean loading (TOC harmonics; station displacement)

        dt_ocean = 0.0
        # 条件分岐: `ocean_model is not None and ocean_site_by_station` を満たす経路を評価する。
        if ocean_model is not None and ocean_site_by_station:
            sid = ocean_site_by_station.get(str(st))
            # 条件分岐: `sid` を満たす経路を評価する。
            if sid:
                try:
                    up_tx_m, east_tx_m, north_tx_m = ol.displacement_uen_m(ocean_model, site_id=sid, dt_utc=tx_times[i])
                    up_rx_m, east_rx_m, north_rx_m = ol.displacement_uen_m(ocean_model, site_id=sid, dt_utc=rx_times[i])

                    # Build a simple local ENU basis in the same inertial frame as r_st_*.
                    # Up is the geocentric radial direction (as defined by HARPOS),
                    # East is along increasing longitude (k × Up), North completes the triad.
                    k = np.array([0.0, 0.0, 1.0], dtype=float)
                    east_tx = np.cross(k, zen_tx)
                    # 条件分岐: `float(np.linalg.norm(east_tx)) < 1e-12` を満たす経路を評価する。
                    if float(np.linalg.norm(east_tx)) < 1e-12:
                        east_tx = np.cross(np.array([0.0, 1.0, 0.0], dtype=float), zen_tx)

                    east_tx = east_tx / max(float(np.linalg.norm(east_tx)), 1e-12)
                    north_tx = np.cross(zen_tx, east_tx)

                    east_rx = np.cross(k, zen_rx)
                    # 条件分岐: `float(np.linalg.norm(east_rx)) < 1e-12` を満たす経路を評価する。
                    if float(np.linalg.norm(east_rx)) < 1e-12:
                        east_rx = np.cross(np.array([0.0, 1.0, 0.0], dtype=float), zen_rx)

                    east_rx = east_rx / max(float(np.linalg.norm(east_rx)), 1e-12)
                    north_rx = np.cross(zen_rx, east_rx)

                    d_ol_tx = zen_tx * float(up_tx_m) + east_tx * float(east_tx_m) + north_tx * float(north_tx_m)
                    d_ol_rx = zen_rx * float(up_rx_m) + east_rx * float(east_rx_m) + north_rx * float(north_rx_m)
                    dt_ocean = -(float(np.dot(u_up, d_ol_tx)) + float(np.dot(u_dn, d_ol_rx))) / llr.C
                except Exception:
                    dt_ocean = 0.0

        dt_station_tide_two_way_s[i] = float(dt_station)
        dt_moon_tide_two_way_s[i] = float(dt_moon)
        dt_ocean_loading_two_way_s[i] = float(dt_ocean)
        dt_tides_no_ocean_two_way_s[i] = float(dt_station + dt_moon)
        dt_tides_two_way_s[i] = float(dt_station + dt_moon + dt_ocean)

    tof_sr_raw_tropo = tof_sr_raw + dt_tropo_two_way_s
    tof_sr_raw_tropo_station_tide = tof_sr_raw_tropo + dt_station_tide_two_way_s
    tof_sr_raw_tropo_moon_tide = tof_sr_raw_tropo + dt_moon_tide_two_way_s
    tof_sr_raw_tropo_tide_no_ocean = tof_sr_raw_tropo + dt_tides_no_ocean_two_way_s
    tof_sr_raw_tropo_tide = tof_sr_raw_tropo + dt_tides_two_way_s
    tof_sr_raw_tropo_no_shapiro = tof_geo_sr + dt_tropo_two_way_s

    # Shapiro extension: Earth (two-way, small but ns級では寄与しうる)
    r1_e_tx = np.linalg.norm(r_st_tx, axis=1)
    r1_e_rx = np.linalg.norm(r_st_rx, axis=1)
    r2_e = np.linalg.norm(r_refl_b, axis=1)
    dt_up_earth = llr.shapiro_oneway_sun(GM_EARTH, r1_e_tx, r2_e, up_sr, coeff=coeff)
    dt_dn_earth = llr.shapiro_oneway_sun(GM_EARTH, r1_e_rx, r2_e, down_sr, coeff=coeff)
    dt_two_earth = dt_up_earth + dt_dn_earth
    tof_sr_raw_tropo_earth_shapiro = tof_sr_raw_tropo + dt_two_earth

    # Ablations (station->reflector only): Shapiro OFF / IAU rotation
    tof_sr_raw_no_shapiro = tof_geo_sr

    r2_sr_iau = np.linalg.norm(r_es_b - r_refl_b_iau, axis=1)
    dt_up_sr_iau_sun = llr.shapiro_oneway_sun(llr.GM_SUN, r1_sm_tx, r2_sr_iau, up_sr_iau, coeff=coeff)
    dt_dn_sr_iau_sun = llr.shapiro_oneway_sun(llr.GM_SUN, r1_sm_rx, r2_sr_iau, down_sr_iau, coeff=coeff)
    dt_two_sr_iau_sun = dt_up_sr_iau_sun + dt_dn_sr_iau_sun
    tof_sr_raw_iau = tof_geo_sr_iau + dt_two_sr_iau_sun
    tof_sr_raw_iau_no_shapiro = tof_geo_sr_iau

    return ModePrediction(
        mode=str(mode),
        has_station=has_station,
        has_reflector=has_reflector,
        tof_gc_raw_s=tof_gc_raw,
        tof_sm_raw_s=tof_sm_raw,
        tof_sr_raw_s=tof_sr_raw,
        tof_sr_raw_no_shapiro_s=tof_sr_raw_no_shapiro,
        tof_sr_raw_tropo_s=tof_sr_raw_tropo,
        tof_sr_raw_tropo_station_tide_s=tof_sr_raw_tropo_station_tide,
        tof_sr_raw_tropo_moon_tide_s=tof_sr_raw_tropo_moon_tide,
        tof_sr_raw_tropo_tide_no_ocean_s=tof_sr_raw_tropo_tide_no_ocean,
        tof_sr_raw_tropo_tide_s=tof_sr_raw_tropo_tide,
        tof_sr_raw_tropo_no_shapiro_s=tof_sr_raw_tropo_no_shapiro,
        tof_sr_raw_tropo_earth_shapiro_s=tof_sr_raw_tropo_earth_shapiro,
        tof_sr_raw_iau_s=tof_sr_raw_iau,
        tof_sr_raw_iau_no_shapiro_s=tof_sr_raw_iau_no_shapiro,
        elev_up_deg=elev_up_deg,
        elev_dn_deg=elev_dn_deg,
    )


def _compute_group_metrics(
    *,
    all_df: pd.DataFrame,
    tof_obs_s: np.ndarray,
    preds: ModePrediction,
    beta: float,
    time_tag_mode_by_station: Optional[Dict[str, str]],
    min_points: int,
    max_groups: int,
    clip_sigma: float,
    clip_min_ns: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    groups = sorted({(s, t) for s, t in zip(all_df["station"].tolist(), all_df["target"].tolist())})
    # 条件分岐: `max_groups and max_groups > 0` を満たす経路を評価する。
    if max_groups and max_groups > 0:
        groups = groups[: int(max_groups)]

    for st, target in groups:
        mask = (all_df["station"].to_numpy() == st) & (all_df["target"].to_numpy() == target)
        min_pts = _min_points_for_target(target, min_points)
        # 条件分岐: `int(mask.sum()) < int(min_pts)` を満たす経路を評価する。
        if int(mask.sum()) < int(min_pts):
            continue

        obs = tof_obs_s[mask]
        gc = preds.tof_gc_raw_s[mask]
        sm = preds.tof_sm_raw_s[mask]
        sr = preds.tof_sr_raw_s[mask]
        sr_nosh = preds.tof_sr_raw_no_shapiro_s[mask]
        sr_tropo = preds.tof_sr_raw_tropo_s[mask]
        sr_tropo_station_tide = preds.tof_sr_raw_tropo_station_tide_s[mask]
        sr_tropo_moon_tide = preds.tof_sr_raw_tropo_moon_tide_s[mask]
        sr_tropo_tide_no_ocean = preds.tof_sr_raw_tropo_tide_no_ocean_s[mask]
        sr_tropo_tide = preds.tof_sr_raw_tropo_tide_s[mask]
        sr_tropo_nosh = preds.tof_sr_raw_tropo_no_shapiro_s[mask]
        sr_tropo_earth = preds.tof_sr_raw_tropo_earth_shapiro_s[mask]
        sr_iau = preds.tof_sr_raw_iau_s[mask]
        sr_iau_nosh = preds.tof_sr_raw_iau_no_shapiro_s[mask]
        has_refl = bool(np.all(np.isfinite(sr)))
        has_refl_iau = bool(np.all(np.isfinite(sr_iau)))

        # Outlier gating (station×target). Use the "best" model (tropo+tide) as reference.
        if has_refl:
            delta_best_ns = (obs - sr_tropo_tide) * 1e9
        else:
            delta_best_ns = (obs - sm) * 1e9

        inlier = _robust_inlier_mask_ns(delta_best_ns, clip_sigma=float(clip_sigma), clip_min_ns=float(clip_min_ns))
        n_inlier = int(np.sum(inlier))
        n_finite = int(np.sum(np.isfinite(delta_best_ns)))
        n_outlier = int(max(0, n_finite - n_inlier))

        # Residuals (offset-aligned on inliers only). Outliers are set to NaN and excluded from RMS.
        res_gc_ns = _offset_align_residual_ns(obs, gc, inlier_mask=inlier)
        res_sm_ns = _offset_align_residual_ns(obs, sm, inlier_mask=inlier)
        res_sr_ns = _offset_align_residual_ns(obs, sr, inlier_mask=inlier) if has_refl else np.full_like(res_gc_ns, np.nan)
        res_sr_nosh_ns = _offset_align_residual_ns(obs, sr_nosh, inlier_mask=inlier) if has_refl else np.full_like(res_gc_ns, np.nan)
        res_sr_tropo_ns = _offset_align_residual_ns(obs, sr_tropo, inlier_mask=inlier) if has_refl else np.full_like(res_gc_ns, np.nan)
        res_sr_tropo_station_ns = (
            _offset_align_residual_ns(obs, sr_tropo_station_tide, inlier_mask=inlier) if has_refl else np.full_like(res_gc_ns, np.nan)
        )
        res_sr_tropo_moon_ns = (
            _offset_align_residual_ns(obs, sr_tropo_moon_tide, inlier_mask=inlier) if has_refl else np.full_like(res_gc_ns, np.nan)
        )
        res_sr_tropo_tide_no_ocean_ns = (
            _offset_align_residual_ns(obs, sr_tropo_tide_no_ocean, inlier_mask=inlier) if has_refl else np.full_like(res_gc_ns, np.nan)
        )
        res_sr_tropo_tide_ns = _offset_align_residual_ns(obs, sr_tropo_tide, inlier_mask=inlier) if has_refl else np.full_like(res_gc_ns, np.nan)
        res_sr_tropo_nosh_ns = _offset_align_residual_ns(obs, sr_tropo_nosh, inlier_mask=inlier) if has_refl else np.full_like(res_gc_ns, np.nan)
        res_sr_tropo_earth_ns = _offset_align_residual_ns(obs, sr_tropo_earth, inlier_mask=inlier) if has_refl else np.full_like(res_gc_ns, np.nan)
        res_sr_iau_ns = _offset_align_residual_ns(obs, sr_iau, inlier_mask=inlier) if has_refl_iau else np.full_like(res_gc_ns, np.nan)
        res_sr_iau_nosh_ns = _offset_align_residual_ns(obs, sr_iau_nosh, inlier_mask=inlier) if has_refl_iau else np.full_like(res_gc_ns, np.nan)

        st_mode = None
        # 条件分岐: `time_tag_mode_by_station is not None` を満たす経路を評価する。
        if time_tag_mode_by_station is not None:
            st_mode = time_tag_mode_by_station.get(st)

        rows.append(
            {
                "station": st,
                "target": target,
                "n": int(mask.sum()),
                "n_inlier": int(n_inlier),
                "n_outlier": int(n_outlier),
                "beta": float(beta),
                "time_tag_mode": str(st_mode or preds.mode),
                "has_station": bool(np.all(preds.has_station[mask])),
                "has_reflector": bool(np.all(preds.has_reflector[mask])),
                "rms_gc_ns": _rms_ns(res_gc_ns),
                "rms_sm_ns": _rms_ns(res_sm_ns),
                "rms_sr_ns": _rms_ns(res_sr_ns),
                "rms_sr_no_shapiro_ns": _rms_ns(res_sr_nosh_ns),
                "rms_sr_tropo_ns": _rms_ns(res_sr_tropo_ns),
                "rms_sr_tropo_station_tide_ns": _rms_ns(res_sr_tropo_station_ns),
                "rms_sr_tropo_moon_tide_ns": _rms_ns(res_sr_tropo_moon_ns),
                "rms_sr_tropo_tide_no_ocean_ns": _rms_ns(res_sr_tropo_tide_no_ocean_ns),
                "rms_sr_tropo_tide_ns": _rms_ns(res_sr_tropo_tide_ns),
                "rms_sr_tropo_no_shapiro_ns": _rms_ns(res_sr_tropo_nosh_ns),
                "rms_sr_tropo_earth_shapiro_ns": _rms_ns(res_sr_tropo_earth_ns),
                "rms_sr_iau_ns": _rms_ns(res_sr_iau_ns),
                "rms_sr_iau_no_shapiro_ns": _rms_ns(res_sr_iau_nosh_ns),
            }
        )

    return pd.DataFrame(rows).sort_values(["station", "target"]).reset_index(drop=True)


def _station_weighted_rms_sr_ns(
    *,
    all_df: pd.DataFrame,
    tof_obs_s: np.ndarray,
    preds: ModePrediction,
    min_points: int,
    clip_sigma: float,
    clip_min_ns: float,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    stations = sorted(set(all_df["station"].tolist()))
    targets = sorted(set(all_df["target"].tolist()))
    st_arr = all_df["station"].to_numpy()
    tgt_arr = all_df["target"].to_numpy()
    for st in stations:
        res_all: List[float] = []
        for tgt in targets:
            mask = (st_arr == st) & (tgt_arr == tgt)
            min_pts = _min_points_for_target(tgt, min_points)
            # 条件分岐: `int(mask.sum()) < int(min_pts)` を満たす経路を評価する。
            if int(mask.sum()) < int(min_pts):
                continue

            obs = tof_obs_s[mask]
            sr = preds.tof_sr_raw_s[mask]
            # 条件分岐: `not np.all(np.isfinite(sr))` を満たす経路を評価する。
            if not np.all(np.isfinite(sr)):
                continue

            delta_ns = (obs - sr) * 1e9
            inlier = _robust_inlier_mask_ns(delta_ns, clip_sigma=float(clip_sigma), clip_min_ns=float(clip_min_ns))
            res_ns = _offset_align_residual_ns(obs, sr, inlier_mask=inlier)
            res_all.extend([float(x) for x in res_ns if np.isfinite(x)])

        out[st] = _rms_ns(np.array(res_all, dtype=float)) if res_all else float("nan")

    return out


def _pick_best_time_tag_by_station(
    *,
    all_df: pd.DataFrame,
    tof_obs_s: np.ndarray,
    preds_by_mode: Dict[str, ModePrediction],
    min_points: int,
    clip_sigma: float,
    clip_min_ns: float,
) -> Tuple[Dict[str, str], Dict[str, Dict[str, float]]]:
    stations = sorted(set(all_df["station"].tolist()))
    rms_by_station_mode: Dict[str, Dict[str, float]] = {st: {} for st in stations}

    for mode, preds in preds_by_mode.items():
        st_rms = _station_weighted_rms_sr_ns(
            all_df=all_df,
            tof_obs_s=tof_obs_s,
            preds=preds,
            min_points=min_points,
            clip_sigma=clip_sigma,
            clip_min_ns=clip_min_ns,
        )
        for st in stations:
            rms_by_station_mode[st][mode] = float(st_rms.get(st, float("nan")))

    best: Dict[str, str] = {}
    for st in stations:
        items = [(m, rms_by_station_mode[st].get(m, float("nan"))) for m in sorted(preds_by_mode.keys())]
        items = [(m, v) for m, v in items if np.isfinite(v)]
        # 条件分岐: `not items` を満たす経路を評価する。
        if not items:
            best[st] = "tx"
            continue

        best_mode, _ = min(items, key=lambda t: float(t[1]))
        best[st] = str(best_mode)

    return best, rms_by_station_mode


def _mix_predictions_by_station(
    *,
    all_df: pd.DataFrame,
    preds_by_mode: Dict[str, ModePrediction],
    best_mode_by_station: Dict[str, str],
) -> ModePrediction:
    n = len(all_df)
    stations_arr = all_df["station"].to_numpy()
    chosen = np.array([best_mode_by_station.get(str(st), "tx") for st in stations_arr], dtype=object)

    def _mix_field(field: str) -> np.ndarray:
        out = np.full((n,), np.nan, dtype=float)
        for mode, preds in preds_by_mode.items():
            mask = chosen == mode
            # 条件分岐: `not np.any(mask)` を満たす経路を評価する。
            if not np.any(mask):
                continue

            v = getattr(preds, field)
            out[mask] = v[mask]

        return out

    def _mix_bool(field: str) -> np.ndarray:
        outb = np.zeros((n,), dtype=bool)
        for mode, preds in preds_by_mode.items():
            mask = chosen == mode
            # 条件分岐: `not np.any(mask)` を満たす経路を評価する。
            if not np.any(mask):
                continue

            v = getattr(preds, field)
            outb[mask] = v[mask]

        return outb

    return ModePrediction(
        mode="auto",
        has_station=_mix_bool("has_station"),
        has_reflector=_mix_bool("has_reflector"),
        tof_gc_raw_s=_mix_field("tof_gc_raw_s"),
        tof_sm_raw_s=_mix_field("tof_sm_raw_s"),
        tof_sr_raw_s=_mix_field("tof_sr_raw_s"),
        tof_sr_raw_no_shapiro_s=_mix_field("tof_sr_raw_no_shapiro_s"),
        tof_sr_raw_tropo_s=_mix_field("tof_sr_raw_tropo_s"),
        tof_sr_raw_tropo_station_tide_s=_mix_field("tof_sr_raw_tropo_station_tide_s"),
        tof_sr_raw_tropo_moon_tide_s=_mix_field("tof_sr_raw_tropo_moon_tide_s"),
        tof_sr_raw_tropo_tide_no_ocean_s=_mix_field("tof_sr_raw_tropo_tide_no_ocean_s"),
        tof_sr_raw_tropo_tide_s=_mix_field("tof_sr_raw_tropo_tide_s"),
        tof_sr_raw_tropo_no_shapiro_s=_mix_field("tof_sr_raw_tropo_no_shapiro_s"),
        tof_sr_raw_tropo_earth_shapiro_s=_mix_field("tof_sr_raw_tropo_earth_shapiro_s"),
        tof_sr_raw_iau_s=_mix_field("tof_sr_raw_iau_s"),
        tof_sr_raw_iau_no_shapiro_s=_mix_field("tof_sr_raw_iau_no_shapiro_s"),
        elev_up_deg=_mix_field("elev_up_deg"),
        elev_dn_deg=_mix_field("elev_dn_deg"),
    )


def main() -> int:
    root = _repo_root()

    ap = argparse.ArgumentParser(description="Batch-evaluate LLR (EDC monthly NP2) against the current P-model.")
    ap.add_argument(
        "--manifest",
        type=str,
        default=str(root / "data" / "llr" / "llr_edc_batch_manifest.json"),
        help="Path to batch manifest JSON (generated by fetch_llr_edc_batch.py).",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(Path("output") / "private" / "llr" / "batch"),
        help="Output directory (relative to repo root unless absolute). Default: output/private/llr/batch",
    )
    ap.add_argument("--beta", type=float, default=1.0, help="P-model beta parameter (Shapiro coefficient uses 2*beta).")
    ap.add_argument(
        "--time-tag-mode",
        type=str,
        default="",
        help="Time tag interpretation: tx/rx/mid/auto (auto picks best per station). Default: env LLR_TIME_TAG or tx.",
    )
    ap.add_argument("--min-points", type=int, default=6, help="Minimum points per station×target group to include.")
    ap.add_argument(
        "--outlier-clip-sigma",
        type=float,
        default=8.0,
        help="Outlier gate: keep points within max(clip_ns, clip_sigma*MAD) around median(obs-model).",
    )
    ap.add_argument(
        "--outlier-clip-ns",
        type=float,
        default=100.0,
        help="Outlier gate: minimum absolute threshold in nanoseconds (default: 100 ns; protects RMS from rare bad points).",
    )
    ap.add_argument("--chunk", type=int, default=200, help="Horizons TLIST chunk size (smaller avoids HTTP 414).")
    ap.add_argument("--offline", action="store_true", help="Do not use network; require Horizons cache.")
    ap.add_argument(
        "--station-coords",
        type=str,
        default="auto",
        help="Station coordinate source: slrlog / pos_eop / auto (default: auto).",
    )
    ap.add_argument(
        "--station-override-json",
        type=str,
        default="",
        help="Optional JSON path for station XYZ override (e.g., APOL merge-route selected_xyz).",
    )
    ap.add_argument(
        "--pos-eop-date",
        type=str,
        default="",
        help="Preferred pos+eop date (YYYYMMDD or YYMMDD). If omitted, nearest cached date is used.",
    )
    ap.add_argument(
        "--pos-eop-max-days",
        type=int,
        default=3650,
        help="Max distance (days) when auto-picking cached pos+eop date (default: 3650).",
    )
    ap.add_argument(
        "--ocean-loading",
        type=str,
        default="auto",
        help="Station tidal ocean loading (TOC) model: auto/on/off. Uses data/llr/ocean_loading/toc_fes2014b_harmod.hps when available.",
    )
    ap.add_argument("--max-groups", type=int, default=0, help="Optional cap on number of station×target groups (0=all).")
    args = ap.parse_args()

    out_dir = Path(str(args.out_dir))
    # 条件分岐: `not out_dir.is_absolute()` を満たす経路を評価する。
    if not out_dir.is_absolute():
        out_dir = root / out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(str(args.manifest))
    # 条件分岐: `not manifest_path.is_absolute()` を満たす経路を評価する。
    if not manifest_path.is_absolute():
        manifest_path = (root / manifest_path).resolve()

    # 条件分岐: `not manifest_path.exists()` を満たす経路を評価する。

    if not manifest_path.exists():
        print(f"[err] missing manifest: {manifest_path}")
        return 2

    manifest = _read_json(manifest_path)
    file_recs = manifest.get("files") or []
    # 条件分岐: `not isinstance(file_recs, list) or not file_recs` を満たす経路を評価する。
    if not isinstance(file_recs, list) or not file_recs:
        print(f"[err] empty files in manifest: {manifest_path}")
        return 2

    offline = bool(args.offline) or (os.environ.get("HORIZONS_OFFLINE", "").strip() == "1")
    cache_dir = root / "output" / "private" / "llr" / "horizons_cache"

    dfs: List[pd.DataFrame] = []
    for rec in file_recs:
        rel = rec.get("cached_path")
        # 条件分岐: `not rel` を満たす経路を評価する。
        if not rel:
            continue

        p = root / Path(str(rel))
        # 条件分岐: `not p.exists()` を満たす経路を評価する。
        if not p.exists():
            continue

        df = llr.parse_crd_npt11(p, assume_two_way_if_missing=True)
        # 条件分岐: `df.empty` を満たす経路を評価する。
        if df.empty:
            continue

        df = df.copy()
        df["source_file"] = str(p.relative_to(root)).replace("\\", "/")
        dfs.append(df)

    # 条件分岐: `not dfs` を満たす経路を評価する。

    if not dfs:
        print("[err] no inputs parsed (record 11 not found?)")
        return 2

    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df.dropna(subset=["epoch_utc", "tof_obs_s"]).reset_index(drop=True)

    # Normalize station/target tokens
    all_df["station"] = all_df["station"].astype(str).str.strip().str.upper()
    all_df["target"] = all_df["target"].astype(str).str.strip().str.lower()

    # Filter obviously invalid TOF rows (prevents station-specific RMS blow-ups, e.g. GRSM)
    bad_tof_csv = out_dir / "llr_bad_tof_rows.csv"
    n0 = int(len(all_df))
    all_df = _filter_llr_rows_by_tof(all_df, tof_min_s=1.0, tof_max_s=4.0, emit_bad_rows_csv=bad_tof_csv)
    n1 = int(len(all_df))
    n_bad_tof = int(n0 - n1)
    # 条件分岐: `n_bad_tof` を満たす経路を評価する。
    if n_bad_tof:
        print(f"[warn] dropped invalid TOF rows: {n_bad_tof} (kept {n1}); see {bad_tof_csv}")

    # Deduplicate exact same NP rows (monthly vs daily files can overlap when include_daily=True).

    n1b = int(len(all_df))
    all_df = all_df.drop_duplicates(subset=["station", "target", "epoch_utc", "tof_obs_s"], keep="first").reset_index(drop=True)
    n1c = int(len(all_df))
    n_dup = int(n1b - n1c)
    # 条件分岐: `n_dup` を満たす経路を評価する。
    if n_dup:
        print(f"[info] dropped exact-duplicate rows: {n_dup} (kept {n1c})")

    beta = float(args.beta)
    chunk = int(args.chunk)
    min_points = int(args.min_points)
    clip_sigma = float(getattr(args, "outlier_clip_sigma", 8.0))
    clip_min_ns = float(getattr(args, "outlier_clip_ns", 10000.0))

    # Data coverage (unique points after filtering + dedup).
    # This is intentionally computed before model evaluation so it can be used to
    # track NGLR-1 year/station coverage even when groups are too sparse for RMS metrics.
    try:
        cov_rows: List[Dict[str, Any]] = []
        for (st, tgt), sub in all_df.groupby(["station", "target"], sort=True):
            n_pts = int(len(sub))
            min_pts = int(_min_points_for_target(tgt, min_points))
            cov_rows.append(
                {
                    "station": str(st),
                    "target": str(tgt),
                    "n_unique": n_pts,
                    "min_points_required": min_pts,
                    "included_in_metrics": bool(n_pts >= min_pts),
                }
            )

        coverage_df = pd.DataFrame(cov_rows).sort_values(["station", "target"]).reset_index(drop=True)
        coverage_path = out_dir / "llr_data_coverage.csv"
        coverage_df.to_csv(coverage_path, index=False)
        print(f"[ok] coverage: {coverage_path}")
    except Exception as e:
        print(f"[warn] coverage summary failed: {e}")

    # Time-tag mode (CLI > env > default)

    mode_raw = str(args.time_tag_mode or "").strip().lower()
    # 条件分岐: `not mode_raw` を満たす経路を評価する。
    if not mode_raw:
        mode_raw = _time_tag_mode_env() or "tx"

    # 条件分岐: `mode_raw not in ("tx", "rx", "mid", "auto")` を満たす経路を評価する。

    if mode_raw not in ("tx", "rx", "mid", "auto"):
        print(f"[err] invalid --time-tag-mode {mode_raw!r} (expected tx/rx/mid/auto)")
        return 2

    mode: TimeTagMode = mode_raw  # type: ignore[assignment]

    tof_obs = all_df["tof_obs_s"].to_numpy(dtype=float)

    # Station coordinate source (slrlog vs pos+eop SINEX)
    station_coords_mode = str(getattr(args, "station_coords", "") or "").strip().lower()
    # 条件分岐: `station_coords_mode not in ("slrlog", "pos_eop", "auto")` を満たす経路を評価する。
    if station_coords_mode not in ("slrlog", "pos_eop", "auto"):
        print(f"[err] invalid --station-coords {station_coords_mode!r} (expected slrlog/pos_eop/auto)")
        return 2

    station_override_json = str(getattr(args, "station_override_json", "") or "").strip()
    pos_eop_date = str(getattr(args, "pos_eop_date", "") or "").strip()
    pos_eop_max_days = int(getattr(args, "pos_eop_max_days", 3650) or 3650)

    station_xyz_override: Dict[str, Dict[str, Any]] = {}
    station_coord_summary: Dict[str, Dict[str, Any]] = {}
    station_override_input: Optional[Dict[str, Any]] = None
    # 条件分岐: `station_override_json` を満たす経路を評価する。
    if station_override_json:
        station_override_path = Path(station_override_json)
        # 条件分岐: `not station_override_path.is_absolute()` を満たす経路を評価する。
        if not station_override_path.is_absolute():
            station_override_path = (root / station_override_path).resolve()

        # 条件分岐: `not station_override_path.exists()` を満たす経路を評価する。

        if not station_override_path.exists():
            print(f"[err] missing --station-override-json: {station_override_path}")
            return 2

        try:
            loaded_override, station_override_input = _load_station_xyz_overrides(station_override_path)
        except Exception as e:
            print(f"[err] failed to load --station-override-json: {e}")
            return 2

        # 条件分岐: `loaded_override` を満たす経路を評価する。

        if loaded_override:
            station_xyz_override.update(loaded_override)
            for st, xyz in loaded_override.items():
                dx = dy = dz = dr = None
                meta = llr._load_station_geodetic(root, st)  # type: ignore[attr-defined]
                try:
                    if (
                        isinstance(meta, dict)
                        and all(meta.get(k) is not None for k in ("x_m", "y_m", "z_m"))
                    ):
                        dx = float(xyz["x_m"]) - float(meta["x_m"])
                        dy = float(xyz["y_m"]) - float(meta["y_m"])
                        dz = float(xyz["z_m"]) - float(meta["z_m"])
                        dr = float(np.sqrt(dx * dx + dy * dy + dz * dz))
                except Exception:
                    pass

                station_coord_summary[str(st)] = {
                    "station": str(st),
                    "pad_id": (meta.get("cdp_pad_id") if isinstance(meta, dict) else None),
                    "coord_source": str(xyz.get("coord_source") or f"override:{station_override_path.name}"),
                    "pos_eop_yymmdd": xyz.get("pos_eop_yymmdd"),
                    "pos_eop_ref_epoch_utc": xyz.get("pos_eop_ref_epoch_utc"),
                    "delta_vs_slrlog_m": dr,
                    "dx_m": dx,
                    "dy_m": dy,
                    "dz_m": dz,
                }

            print(
                f"[info] station override: loaded {len(loaded_override)} station(s) from {station_override_path.relative_to(root).as_posix()}"
            )
        else:
            print(f"[warn] station override: no valid xyz rows in {station_override_path}")

    # 条件分岐: `station_coords_mode in ("auto", "pos_eop")` を満たす経路を評価する。

    if station_coords_mode in ("auto", "pos_eop"):
        stations_all = sorted({s for s in all_df["station"].unique() if s and s.lower() not in ("na", "nan")})
        for st in stations_all:
            # 条件分岐: `st in station_xyz_override` を満たす経路を評価する。
            if st in station_xyz_override:
                continue

            meta = llr._load_station_geodetic(root, st)  # type: ignore[attr-defined]
            # 条件分岐: `not isinstance(meta, dict)` を満たす経路を評価する。
            if not isinstance(meta, dict):
                continue

            pad_id = meta.get("cdp_pad_id")
            try:
                pad_id_i = int(pad_id) if pad_id is not None else None
            except Exception:
                pad_id_i = None

            # 条件分岐: `pad_id_i is None` を満たす経路を評価する。

            if pad_id_i is None:
                continue

            st_epochs = all_df.loc[all_df["station"] == st, "epoch_utc"].dropna().tolist()
            # 条件分岐: `not st_epochs` を満たす経路を評価する。
            if not st_epochs:
                continue

            dts: List[datetime] = []
            for t in st_epochs:
                # 条件分岐: `isinstance(t, pd.Timestamp)` を満たす経路を評価する。
                if isinstance(t, pd.Timestamp):
                    dts.append(t.to_pydatetime())
                # 条件分岐: 前段条件が不成立で、`isinstance(t, datetime)` を追加評価する。
                elif isinstance(t, datetime):
                    dts.append(t)

            # 条件分岐: `not dts` を満たす経路を評価する。

            if not dts:
                continue

            dts = [dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc) for dt in dts]
            dts.sort()
            target_dt = dts[len(dts) // 2]

            xyz = llr.load_station_xyz_from_pos_eop(
                root,
                station_code=st,
                pad_id=pad_id_i,
                target_dt=target_dt,
                max_days=pos_eop_max_days,
                preferred_yymmdd=pos_eop_date,
            )
            # 条件分岐: `not isinstance(xyz, dict)` を満たす経路を評価する。
            if not isinstance(xyz, dict):
                continue

            # 条件分岐: `not all(k in xyz for k in ("x_m", "y_m", "z_m"))` を満たす経路を評価する。

            if not all(k in xyz for k in ("x_m", "y_m", "z_m")):
                continue

            station_xyz_override[st] = xyz

            dx = dy = dz = dr = None
            try:
                # 条件分岐: `all(meta.get(k) is not None for k in ("x_m", "y_m", "z_m"))` を満たす経路を評価する。
                if all(meta.get(k) is not None for k in ("x_m", "y_m", "z_m")):
                    dx = float(xyz["x_m"]) - float(meta["x_m"])
                    dy = float(xyz["y_m"]) - float(meta["y_m"])
                    dz = float(xyz["z_m"]) - float(meta["z_m"])
                    dr = float(np.sqrt(dx * dx + dy * dy + dz * dz))
            except Exception:
                pass

            station_coord_summary[st] = {
                "station": st,
                "pad_id": pad_id_i,
                "coord_source": "EDC pos+eop (SINEX)",
                "pos_eop_yymmdd": xyz.get("pos_eop_yymmdd"),
                "pos_eop_ref_epoch_utc": xyz.get("pos_eop_ref_epoch_utc"),
                "delta_vs_slrlog_m": dr,
                "dx_m": dx,
                "dy_m": dy,
                "dz_m": dz,
            }

        # 条件分岐: `station_coords_mode == "pos_eop" and not station_xyz_override` を満たす経路を評価する。

        if station_coords_mode == "pos_eop" and not station_xyz_override:
            print("[err] --station-coords=pos_eop but no cached pos+eop station coords matched. Run scripts/llr/fetch_pos_eop_edc.py first.")
            return 2

        # 条件分岐: `station_coords_mode == "auto"` を満たす経路を評価する。

        if station_coords_mode == "auto":
            # 条件分岐: `station_xyz_override` を満たす経路を評価する。
            if station_xyz_override:
                print(f"[info] station coords: using override/pos+eop for {len(station_xyz_override)} station(s)")
            else:
                print("[info] station coords: pos+eop not available; using slrlog")

    # Ocean loading (TOC harmonics; HARPOS via IMLS)

    ocean_loading_mode = str(getattr(args, "ocean_loading", "") or "auto").strip().lower()
    # 条件分岐: `ocean_loading_mode not in ("auto", "on", "off")` を満たす経路を評価する。
    if ocean_loading_mode not in ("auto", "on", "off"):
        print(f"[err] invalid --ocean-loading {ocean_loading_mode!r} (expected auto/on/off)")
        return 2

    ocean_model: Optional[ol.HarposModel] = None
    ocean_site_by_station: Dict[str, str] = {}
    ocean_site_info: Dict[str, Dict[str, Any]] = {}
    ocean_harpos_path = root / "data" / "llr" / "ocean_loading" / "toc_fes2014b_harmod.hps"
    # 条件分岐: `ocean_loading_mode in ("auto", "on")` を満たす経路を評価する。
    if ocean_loading_mode in ("auto", "on"):
        # 条件分岐: `not ocean_harpos_path.exists()` を満たす経路を評価する。
        if not ocean_harpos_path.exists():
            # 条件分岐: `ocean_loading_mode == "on"` を満たす経路を評価する。
            if ocean_loading_mode == "on":
                print("[err] --ocean-loading=on but HARPOS file is missing.")
                print("      Run: python -B scripts/llr/fetch_ocean_loading_imls.py")
                return 2
        else:
            try:
                ocean_model = ol.parse_harpos(ocean_harpos_path)
                st_for_map = sorted({s for s in all_df["station"].unique() if s and s.lower() not in ("na", "nan")})
                for st in st_for_map:
                    meta = llr._load_station_geodetic(root, st)  # type: ignore[attr-defined]
                    # 条件分岐: `not isinstance(meta, dict)` を満たす経路を評価する。
                    if not isinstance(meta, dict):
                        continue

                    use_pos = bool(
                        station_coords_mode in ("auto", "pos_eop")
                        and isinstance(station_xyz_override.get(st), dict)
                        and all(k in station_xyz_override[st] for k in ("x_m", "y_m", "z_m"))
                    )
                    try:
                        # 条件分岐: `use_pos` を満たす経路を評価する。
                        if use_pos:
                            x_m = float(station_xyz_override[st]["x_m"])
                            y_m = float(station_xyz_override[st]["y_m"])
                            z_m = float(station_xyz_override[st]["z_m"])
                        # 条件分岐: 前段条件が不成立で、`all(meta.get(k) is not None for k in ("x_m", "y_m", "z_m"))` を追加評価する。
                        elif all(meta.get(k) is not None for k in ("x_m", "y_m", "z_m")):
                            x_m = float(meta["x_m"])
                            y_m = float(meta["y_m"])
                            z_m = float(meta["z_m"])
                        # 条件分岐: 前段条件が不成立で、`all(meta.get(k) is not None for k in ("lat_deg", "lon_deg", "height_m"))` を追加評価する。
                        elif all(meta.get(k) is not None for k in ("lat_deg", "lon_deg", "height_m")):
                            v = llr.ecef_from_geodetic(float(meta["lat_deg"]), float(meta["lon_deg"]), float(meta["height_m"]))
                            x_m, y_m, z_m = float(v[0]), float(v[1]), float(v[2])
                        else:
                            continue
                    except Exception:
                        continue

                    sid, dist_m = ol.best_site_by_ecef(ocean_model, x_m=x_m, y_m=y_m, z_m=z_m)
                    # 条件分岐: `not sid` を満たす経路を評価する。
                    if not sid:
                        continue

                    ocean_site_by_station[str(st)] = str(sid)
                    ocean_site_info[str(st)] = {
                        "harpos_site_id": str(sid),
                        "harpos_dist_m": float(dist_m),
                        "validity_radius_m": float(ocean_model.validity_radius_m),
                        "warning_outside_validity": bool(
                            np.isfinite(float(ocean_model.validity_radius_m))
                            and float(ocean_model.validity_radius_m) > 0.0
                            and float(dist_m) > float(ocean_model.validity_radius_m)
                        ),
                    }

                # 条件分岐: `ocean_site_by_station` を満たす経路を評価する。

                if ocean_site_by_station:
                    print(
                        f"[info] ocean loading (TOC): enabled ({ocean_harpos_path.relative_to(root).as_posix()}), mapped {len(ocean_site_by_station)} station(s)"
                    )
                else:
                    print(f"[warn] ocean loading (TOC): HARPOS loaded but no stations matched; continuing without it")
                    ocean_model = None
            except Exception as e:
                print(f"[warn] ocean loading parse failed: {e}; continuing without it")
                ocean_model = None

    time_tag_mode_by_station: Optional[Dict[str, str]] = None
    rms_by_station_mode: Optional[Dict[str, Dict[str, float]]] = None
    preds_final: Optional[ModePrediction] = None

    # 条件分岐: `mode == "auto"` を満たす経路を評価する。
    if mode == "auto":
        best_path = out_dir / "llr_time_tag_best_by_station.json"

        def _load_time_tag_best(path: Path) -> tuple[Optional[Dict[str, str]], Optional[Dict[str, Dict[str, float]]]]:
            # 条件分岐: `not path.exists()` を満たす経路を評価する。
            if not path.exists():
                return None, None

            try:
                d = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return None, None

            # 条件分岐: `not isinstance(d, dict)` を満たす経路を評価する。

            if not isinstance(d, dict):
                return None, None

            best = d.get("best_mode_by_station")
            rms = d.get("rms_by_station_and_mode_ns")
            best_out: Dict[str, str] = {}
            # 条件分岐: `isinstance(best, dict)` を満たす経路を評価する。
            if isinstance(best, dict):
                for k, v in best.items():
                    ks = str(k).strip().upper()
                    vs = str(v).strip().lower()
                    # 条件分岐: `ks and vs in ("tx", "rx", "mid")` を満たす経路を評価する。
                    if ks and vs in ("tx", "rx", "mid"):
                        best_out[ks] = vs

            rms_out: Dict[str, Dict[str, float]] = {}
            # 条件分岐: `isinstance(rms, dict)` を満たす経路を評価する。
            if isinstance(rms, dict):
                for st, rec in rms.items():
                    st_s = str(st).strip().upper()
                    # 条件分岐: `not st_s or not isinstance(rec, dict)` を満たす経路を評価する。
                    if not st_s or not isinstance(rec, dict):
                        continue

                    rec_out: Dict[str, float] = {}
                    for m, val in rec.items():
                        ms = str(m).strip().lower()
                        # 条件分岐: `ms not in ("tx", "rx", "mid")` を満たす経路を評価する。
                        if ms not in ("tx", "rx", "mid"):
                            continue

                        try:
                            rec_out[ms] = float(val)
                        except Exception:
                            continue

                    # 条件分岐: `rec_out` を満たす経路を評価する。

                    if rec_out:
                        rms_out[st_s] = rec_out

            return (best_out or None), (rms_out or None)

        # Offline replay should not fail due to missing "rx/mid" Horizons caches.
        # If a previous run already determined station-best time-tag modes, reuse it.

        if offline:
            cached_best, cached_rms = _load_time_tag_best(best_path)
            # 条件分岐: `cached_best` を満たす経路を評価する。
            if cached_best:
                best_by_station = cached_best
                rms_by_station_mode = cached_rms
                # Ensure every station in this batch has a resolved mode.
                # If a station is not present in the cached selection dictionary,
                # fall back to "tx" (safest default for CRD/NP2 in our pipeline).
                stations_in_batch = {
                    str(s).strip().upper()
                    for s in all_df["station"].astype(str).unique().tolist()
                    if str(s).strip()
                }
                for st in sorted(stations_in_batch):
                    # 条件分岐: `st in ("NA", "NAN")` を満たす経路を評価する。
                    if st in ("NA", "NAN"):
                        continue

                    best_by_station.setdefault(st, "tx")

                time_tag_mode_by_station = best_by_station
                print(f"[info] offline: reuse time-tag selection from {best_path}")

                selected_modes = sorted(set(best_by_station.values()))
                preds_by_mode_full: Dict[str, ModePrediction] = {}
                for m in selected_modes:
                    print(f"[info] evaluating time-tag mode (full): {m}")
                    preds_by_mode_full[m] = _compute_predictions_for_mode(
                        root=root,
                        all_df=all_df,
                        tof_obs_s=tof_obs,
                        mode=m,
                        beta=beta,
                        offline=offline,
                        chunk=chunk,
                        cache_dir=cache_dir,
                        station_xyz_override=station_xyz_override or None,
                        ocean_model=ocean_model,
                        ocean_site_by_station=ocean_site_by_station or None,
                    )

                preds_final = _mix_predictions_by_station(
                    all_df=all_df,
                    preds_by_mode=preds_by_mode_full,
                    best_mode_by_station=best_by_station,
                )
            else:
                print(f"[warn] offline: time-tag auto selection cache not found; falling back to tx (create {best_path} by running once online)")
                mode = "tx"  # type: ignore[assignment]

        # 条件分岐: `mode == "auto" and preds_final is None` を満たす経路を評価する。

        if mode == "auto" and preds_final is None:
            # --------------------------------------------
            # Phase 1 (selection): evaluate tx/rx/mid on a small, time-spread sample per station×target group
            # --------------------------------------------
            sample_per_group = 80
            min_points_selection = int(min(min_points, sample_per_group))
            st_arr = all_df["station"].to_numpy()
            tgt_arr = all_df["target"].to_numpy()
            sample_idx: List[int] = []
            for st, tgt in sorted({(s, t) for s, t in zip(st_arr.tolist(), tgt_arr.tolist())}):
                mask = (st_arr == st) & (tgt_arr == tgt)
                idx = np.flatnonzero(mask)
                min_pts = _min_points_for_target(tgt, min_points)
                # 条件分岐: `len(idx) < int(min_pts)` を満たす経路を評価する。
                if len(idx) < int(min_pts):
                    continue

                # 条件分岐: `len(idx) <= sample_per_group` を満たす経路を評価する。

                if len(idx) <= sample_per_group:
                    pick = idx
                else:
                    # Use a spread sample; ensure uniqueness to avoid accidental undersampling.
                    pos = np.round(np.linspace(0, len(idx) - 1, sample_per_group)).astype(int)
                    pos = np.clip(pos, 0, len(idx) - 1)
                    pos = np.unique(pos)
                    # 条件分岐: `len(pos) < sample_per_group` を満たす経路を評価する。
                    if len(pos) < sample_per_group:
                        remaining = np.setdiff1d(np.arange(len(idx)), pos, assume_unique=False)
                        need = int(sample_per_group - len(pos))
                        # 条件分岐: `len(remaining) > 0 and need > 0` を満たす経路を評価する。
                        if len(remaining) > 0 and need > 0:
                            fill_pos = remaining[np.linspace(0, len(remaining) - 1, min(need, len(remaining)), dtype=int)]
                            pos = np.unique(np.concatenate([pos, fill_pos]))

                    pick = idx[np.sort(pos)]

                sample_idx.extend([int(i) for i in pick])

            sample_idx = sorted(set(sample_idx))
            sample_df = all_df.iloc[sample_idx].reset_index(drop=True)
            sample_tof = tof_obs[sample_idx]

            preds_by_mode_sample: Dict[str, ModePrediction] = {}
            for m in ("tx", "rx", "mid"):
                print(f"[info] evaluating time-tag mode (sample): {m}")
                preds_by_mode_sample[m] = _compute_predictions_for_mode(
                    root=root,
                    all_df=sample_df,
                    tof_obs_s=sample_tof,
                    mode=m,
                    beta=beta,
                    offline=offline,
                    chunk=chunk,
                    cache_dir=cache_dir,
                    station_xyz_override=station_xyz_override or None,
                    ocean_model=ocean_model,
                    ocean_site_by_station=ocean_site_by_station or None,
                )

            best_by_station, rms_by_station_mode = _pick_best_time_tag_by_station(
                all_df=sample_df,
                tof_obs_s=sample_tof,
                preds_by_mode=preds_by_mode_sample,
                min_points=min_points_selection,
                clip_sigma=clip_sigma,
                clip_min_ns=clip_min_ns,
            )
            time_tag_mode_by_station = best_by_station

            best_path.write_text(
                json.dumps(
                    {
                        "generated_utc": datetime.now(timezone.utc).isoformat(),
                        "selection_metric": "weighted_rms_sr_ns (offset aligned per station×target)",
                        "min_points_per_group": int(min_points),
                        "min_points_per_group_selection": int(min_points_selection),
                        "selection_sample_per_group": int(sample_per_group),
                        "selection_sample_points_total": int(len(sample_df)),
                        "best_mode_by_station": best_by_station,
                        "rms_by_station_and_mode_ns": rms_by_station_mode,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"[ok] time-tag selection: {best_path}")

            # Plot: RMS per station per mode
            _set_japanese_font()
            st_list = sorted(best_by_station.keys())
            mode_list = ["tx", "rx", "mid"]
            x = np.arange(len(st_list))
            width = 0.25
            plt.figure(figsize=(10.5, 4.2))
            for j, mm in enumerate(mode_list):
                vals = [float(rms_by_station_mode[st].get(mm, float("nan"))) for st in st_list]  # type: ignore[index]
                plt.bar(x + (j - 1) * width, vals, width=width, label=f"{mm}")

            plt.xticks(x, st_list)
            plt.ylabel("残差RMS [ns]（観測局→反射器, 定数整列, 局内重み付き）")
            plt.title(f"{LLR_SHORT_NAME}：time-tag 最適化（局ごとに tx/rx/mid を選択）")
            plt.grid(True, axis="y", alpha=0.3)
            plt.legend()
            plt.tight_layout()
            p_sel = out_dir / "llr_time_tag_selection_by_station.png"
            plt.savefig(p_sel, dpi=200)
            plt.close()

            # --------------------------------------------
            # Phase 2 (final): compute only the modes actually selected by stations, for the full dataset
            # --------------------------------------------
            selected_modes = sorted(set(best_by_station.values()))
            preds_by_mode_full: Dict[str, ModePrediction] = {}
            for m in selected_modes:
                print(f"[info] evaluating time-tag mode (full): {m}")
                preds_by_mode_full[m] = _compute_predictions_for_mode(
                    root=root,
                    all_df=all_df,
                    tof_obs_s=tof_obs,
                    mode=m,
                    beta=beta,
                    offline=offline,
                    chunk=chunk,
                    cache_dir=cache_dir,
                    station_xyz_override=station_xyz_override or None,
                    ocean_model=ocean_model,
                    ocean_site_by_station=ocean_site_by_station or None,
                )

            preds_final = _mix_predictions_by_station(
                all_df=all_df,
                preds_by_mode=preds_by_mode_full,
                best_mode_by_station=best_by_station,
            )

    # 条件分岐: `preds_final is None` を満たす経路を評価する。

    if preds_final is None:
        preds_final = _compute_predictions_for_mode(
            root=root,
            all_df=all_df,
            tof_obs_s=tof_obs,
            mode=str(mode),
            beta=beta,
            offline=offline,
            chunk=chunk,
            cache_dir=cache_dir,
            station_xyz_override=station_xyz_override or None,
            ocean_model=ocean_model,
            ocean_site_by_station=ocean_site_by_station or None,
        )

    metrics_df = _compute_group_metrics(
        all_df=all_df,
        tof_obs_s=tof_obs,
        preds=preds_final,
        beta=beta,
        time_tag_mode_by_station=time_tag_mode_by_station,
        min_points=min_points,
        max_groups=int(args.max_groups),
        clip_sigma=clip_sigma,
        clip_min_ns=clip_min_ns,
    )
    # 条件分岐: `metrics_df.empty` を満たす経路を評価する。
    if metrics_df.empty:
        print("[err] no groups met the criteria (try lowering --min-points)")
        return 2

    metrics_csv = out_dir / "llr_batch_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    # --------------------------------------------
    # Diagnostics: monthly residual stats (station/target)
    # --------------------------------------------
    try:
        # Record station coordinate sources used (site log provenance)
        try:
            st_meta_out: Dict[str, Dict[str, Any]] = {}
            for st in sorted(set(all_df["station"].tolist())):
                p = root / "data" / "llr" / "stations" / f"{str(st).lower()}.json"
                # 条件分岐: `not p.exists()` を満たす経路を評価する。
                if not p.exists():
                    continue

                d = json.loads(p.read_text(encoding="utf-8"))
                st_key = str(st)
                pos = station_xyz_override.get(st_key)
                use_pos = bool(station_coords_mode in ("auto", "pos_eop") and isinstance(pos, dict) and all(k in pos for k in ("x_m", "y_m", "z_m")))

                x_slr = d.get("x_m")
                y_slr = d.get("y_m")
                z_slr = d.get("z_m")
                x_pos = pos.get("x_m") if use_pos else None
                y_pos = pos.get("y_m") if use_pos else None
                z_pos = pos.get("z_m") if use_pos else None

                x_used = x_pos if use_pos else x_slr
                y_used = y_pos if use_pos else y_slr
                z_used = z_pos if use_pos else z_slr

                st_meta_out[str(st)] = {
                    "site_name": d.get("site_name"),
                    "date_prepared": d.get("date_prepared"),
                    "log_filename": d.get("log_filename"),
                    "log_url": d.get("log_url"),
                    "lat_deg": d.get("lat_deg"),
                    "lon_deg": d.get("lon_deg"),
                    "height_m": d.get("height_m"),
                    # SLRLOG (site log)
                    "x_m": x_slr,
                    "y_m": y_slr,
                    "z_m": z_slr,
                    # pos+eop (SINEX) override candidate
                    "x_m_pos_eop": x_pos,
                    "y_m_pos_eop": y_pos,
                    "z_m_pos_eop": z_pos,
                    # actually used (auto/pos_eop modes)
                    # Keep `station_coord_source_used` as canonical token for downstream audits.
                    "station_coord_source_used": ("pos_eop" if use_pos else "slrlog"),
                    "station_coord_source_used_detail": (
                        str(pos.get("coord_source") or "pos_eop") if use_pos and isinstance(pos, dict) else "slrlog"
                    ),
                    "x_m_used": x_used,
                    "y_m_used": y_used,
                    "z_m_used": z_used,
                }
                extra = station_coord_summary.get(str(st))
                # 条件分岐: `isinstance(extra, dict)` を満たす経路を評価する。
                if isinstance(extra, dict):
                    st_meta_out[str(st)]["pos_eop_yymmdd"] = extra.get("pos_eop_yymmdd")
                    st_meta_out[str(st)]["pos_eop_ref_epoch_utc"] = extra.get("pos_eop_ref_epoch_utc")
                    st_meta_out[str(st)]["delta_vs_slrlog_m"] = extra.get("delta_vs_slrlog_m")

            (out_dir / "llr_station_metadata_used.json").write_text(
                json.dumps(
                    {"generated_utc": datetime.now(timezone.utc).isoformat(), "stations": st_meta_out},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

            # Plot: station coordinate delta (pos+eop vs slrlog), if available
            try:
                # 条件分岐: `station_coord_summary` を満たす経路を評価する。
                if station_coord_summary:
                    _set_japanese_font()
                    xs: List[str] = []
                    ys: List[float] = []
                    for st in sorted(station_coord_summary.keys()):
                        v = station_coord_summary.get(st, {})
                        dr = v.get("delta_vs_slrlog_m")
                        # 条件分岐: `dr is None or not np.isfinite(float(dr))` を満たす経路を評価する。
                        if dr is None or not np.isfinite(float(dr)):
                            continue

                        xs.append(str(st))
                        ys.append(float(dr))

                    # 条件分岐: `xs` を満たす経路を評価する。

                    if xs:
                        width_in = max(12.0, 0.6 * len(xs) + 4.0)
                        fig, ax = plt.subplots(figsize=(width_in, 6.0), dpi=200)
                        ax.bar(xs, ys)
                        ax.set_ylabel("||Δr|| [m]（pos+eop - slrlog）")
                        ax.set_title(f"{LLR_SHORT_NAME}：局座標の差分（一次ソースpos+eopの影響）")
                        ax.grid(True, axis="y", alpha=0.3)
                        for label in ax.get_xticklabels():
                            label.set_rotation(45)
                            label.set_ha("right")

                        fig.tight_layout()
                        fig.savefig(out_dir / "llr_station_coord_delta_pos_eop.png", dpi=220, bbox_inches="tight")
                        plt.close(fig)
            except Exception as e:
                print(f"[warn] station coord delta plot failed: {e}")
        except Exception as e:
            print(f"[warn] station metadata emit failed: {e}")

        st_arr = all_df["station"].to_numpy()
        tgt_arr = all_df["target"].to_numpy()
        n = len(all_df)

        # Residuals (offset-aligned per station×reflector) for multiple model variants,
        # to help isolate station-dependent / period-dependent effects.
        res_sr_ns = np.full((n,), np.nan, dtype=float)
        res_sr_tropo_ns = np.full((n,), np.nan, dtype=float)
        res_sr_tropo_station_tide_ns = np.full((n,), np.nan, dtype=float)
        res_sr_tropo_moon_tide_ns = np.full((n,), np.nan, dtype=float)
        res_sr_tropo_tide_no_ocean_ns = np.full((n,), np.nan, dtype=float)
        res_sr_tropo_tide_ns = np.full((n,), np.nan, dtype=float)
        res_sr_tropo_tide_all_ns = np.full((n,), np.nan, dtype=float)
        inlier_best = np.zeros((n,), dtype=bool)
        processed_best = np.zeros((n,), dtype=bool)
        delta_best_raw_ns = np.full((n,), np.nan, dtype=float)

        for st, tgt in sorted({(s, t) for s, t in zip(st_arr.tolist(), tgt_arr.tolist())}):
            mask = (st_arr == st) & (tgt_arr == tgt)
            min_pts = _min_points_for_target(tgt, min_points)
            # 条件分岐: `int(mask.sum()) < int(min_pts)` を満たす経路を評価する。
            if int(mask.sum()) < int(min_pts):
                continue

            obs = tof_obs[mask]
            pred = preds_final.tof_sr_raw_s[mask]
            pred_tropo = preds_final.tof_sr_raw_tropo_s[mask]
            pred_tropo_station = preds_final.tof_sr_raw_tropo_station_tide_s[mask]
            pred_tropo_moon = preds_final.tof_sr_raw_tropo_moon_tide_s[mask]
            pred_tropo_tide_no_ocean = preds_final.tof_sr_raw_tropo_tide_no_ocean_s[mask]
            pred_tropo_tide = preds_final.tof_sr_raw_tropo_tide_s[mask]

            # Robust outlier gating per station×target, consistent with the group-metrics pipeline.
            # Use the "best" model (tropo+tide) as reference when available, else fall back.
            processed_best[mask] = True
            # 条件分岐: `np.all(np.isfinite(pred_tropo_tide))` を満たす経路を評価する。
            if np.all(np.isfinite(pred_tropo_tide)):
                delta_ref_ns = (obs - pred_tropo_tide) * 1e9
            # 条件分岐: 前段条件が不成立で、`np.all(np.isfinite(pred_tropo))` を追加評価する。
            elif np.all(np.isfinite(pred_tropo)):
                delta_ref_ns = (obs - pred_tropo) * 1e9
            else:
                delta_ref_ns = (obs - pred) * 1e9

            delta_best_raw_ns[mask] = delta_ref_ns
            inlier = _robust_inlier_mask_ns(delta_ref_ns, clip_sigma=float(clip_sigma), clip_min_ns=float(clip_min_ns))
            inlier_best[mask] = inlier

            res_sr_ns[mask] = _offset_align_residual_ns(obs, pred, inlier_mask=inlier)
            res_sr_tropo_ns[mask] = _offset_align_residual_ns(obs, pred_tropo, inlier_mask=inlier)
            res_sr_tropo_station_tide_ns[mask] = _offset_align_residual_ns(obs, pred_tropo_station, inlier_mask=inlier)
            res_sr_tropo_moon_tide_ns[mask] = _offset_align_residual_ns(obs, pred_tropo_moon, inlier_mask=inlier)
            res_sr_tropo_tide_no_ocean_ns[mask] = _offset_align_residual_ns(obs, pred_tropo_tide_no_ocean, inlier_mask=inlier)
            res_sr_tropo_tide_ns[mask] = _offset_align_residual_ns(obs, pred_tropo_tide, inlier_mask=inlier)
            # Diagnostics: keep outliers too (using the offset fitted on inliers) so spikes are explainable.
            res_sr_tropo_tide_all_ns[mask] = _offset_align_residual_all_ns(obs, pred_tropo_tide, inlier_mask=inlier)

        diag_df = pd.DataFrame(
            {
                "epoch_utc": all_df["epoch_utc"],
                "station": all_df["station"],
                "target": all_df["target"],
                "source_file": all_df.get("source_file"),
                "lineno": all_df.get("lineno"),
                "tof_obs_s": tof_obs,
                "delta_best_raw_ns": delta_best_raw_ns,
                "inlier_best": processed_best & inlier_best,
                "outlier_best": processed_best & (~inlier_best),
                "residual_sr_ns": res_sr_ns,
                "residual_sr_tropo_ns": res_sr_tropo_ns,
                "residual_sr_tropo_station_tide_ns": res_sr_tropo_station_tide_ns,
                "residual_sr_tropo_moon_tide_ns": res_sr_tropo_moon_tide_ns,
                "residual_sr_tropo_tide_no_ocean_ns": res_sr_tropo_tide_no_ocean_ns,
                "residual_sr_tropo_tide_ns": res_sr_tropo_tide_ns,
                "residual_sr_tropo_tide_all_ns": res_sr_tropo_tide_all_ns,
                "elev_up_deg": preds_final.elev_up_deg,
                "elev_dn_deg": preds_final.elev_dn_deg,
                "dt_tropo_ns": (preds_final.tof_sr_raw_tropo_s - preds_final.tof_sr_raw_s) * 1e9,
                "dt_tide_station_ns": (preds_final.tof_sr_raw_tropo_station_tide_s - preds_final.tof_sr_raw_tropo_s) * 1e9,
                "dt_tide_moon_ns": (preds_final.tof_sr_raw_tropo_moon_tide_s - preds_final.tof_sr_raw_tropo_s) * 1e9,
                "dt_tide_no_ocean_ns": (preds_final.tof_sr_raw_tropo_tide_no_ocean_s - preds_final.tof_sr_raw_tropo_s) * 1e9,
                "dt_tide_ns": (preds_final.tof_sr_raw_tropo_tide_s - preds_final.tof_sr_raw_tropo_s) * 1e9,
                "dt_ocean_loading_ns": (
                    preds_final.tof_sr_raw_tropo_tide_s - preds_final.tof_sr_raw_tropo_tide_no_ocean_s
                )
                * 1e9,
                "dt_sun_shapiro_ns": (preds_final.tof_sr_raw_s - preds_final.tof_sr_raw_no_shapiro_s) * 1e9,
                "dt_earth_shapiro_ns": (
                    preds_final.tof_sr_raw_tropo_earth_shapiro_s - preds_final.tof_sr_raw_tropo_s
                )
                * 1e9,
            }
        )
        diag_df = diag_df.dropna(subset=["epoch_utc"]).reset_index(drop=True)
        diag_df["year_month"] = pd.to_datetime(diag_df["epoch_utc"], utc=True, errors="coerce", format="mixed").dt.strftime("%Y-%m")
        diag_df["elev_mean_deg"] = (diag_df["elev_up_deg"] + diag_df["elev_dn_deg"]) / 2.0

        # Optional meteo columns (from CRD record 20 parsing)
        for c in ("pressure_hpa", "temp_k", "rh_percent", "met_source"):
            # 条件分岐: `c in all_df.columns` を満たす経路を評価する。
            if c in all_df.columns:
                diag_df[c] = all_df[c]

        # time-tag mode per point (auto selection keeps station-specific modes)

        if time_tag_mode_by_station:
            diag_df["time_tag_mode"] = [str(time_tag_mode_by_station.get(str(s), "tx")) for s in diag_df["station"].tolist()]
        else:
            diag_df["time_tag_mode"] = str(mode)

        # Center the raw delta (obs - model) by the inlier median per station×target.
        # This makes outliers interpretable on an absolute scale without the unknown constant offset.

        try:
            med_df = (
                diag_df.loc[diag_df["inlier_best"] == True]  # noqa: E712
                .groupby(["station", "target"], dropna=False)["delta_best_raw_ns"]
                .median()
                .reset_index()
                .rename(columns={"delta_best_raw_ns": "delta_best_med_ns"})
            )
            diag_df = diag_df.merge(med_df, on=["station", "target"], how="left")
            diag_df["delta_best_centered_ns"] = diag_df["delta_best_raw_ns"] - diag_df["delta_best_med_ns"]
        except Exception:
            diag_df["delta_best_med_ns"] = np.nan
            diag_df["delta_best_centered_ns"] = np.nan

        # Emit per-point diagnostics (small enough for offline use; enables root-cause analysis).

        diag_points_csv = out_dir / "llr_batch_points.csv"
        diag_df.to_csv(diag_points_csv, index=False)

        # Emit outliers for manual review (helps isolate bad files / wrong time-tag / metadata issues).
        try:
            out_df = diag_df[diag_df["outlier_best"] == True].copy()  # noqa: E712
            # 条件分岐: `not out_df.empty` を満たす経路を評価する。
            if not out_df.empty:
                out_df = out_df.sort_values(
                    "delta_best_centered_ns",
                    key=lambda s: np.abs(s.to_numpy(dtype=float)),
                    ascending=False,
                )
                out_df.to_csv(out_dir / "llr_outliers.csv", index=False)
        except Exception as e:
            print(f"[warn] outlier emit failed: {e}")

        # Outlier summary + overview plot (for roadmap step 2: spike root-cause)

        outliers_csv = out_dir / "llr_outliers.csv"
        outliers_summary_path = out_dir / "llr_outliers_summary.json"
        outliers_plot = out_dir / "llr_outliers_overview.png"
        try:
            o = diag_df.loc[diag_df["outlier_best"] == True].copy()  # noqa: E712
            n_out = int(len(o))
            n_inl = int(np.sum(diag_df["inlier_best"].to_numpy(dtype=bool)))
            max_abs = float("nan")
            # 条件分岐: `n_out` を満たす経路を評価する。
            if n_out:
                a = np.abs(pd.to_numeric(o["delta_best_centered_ns"], errors="coerce").to_numpy(dtype=float))
                a = a[np.isfinite(a)]
                max_abs = float(np.max(a)) if len(a) else float("nan")

            out_sum = {
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "n_total": int(len(diag_df)),
                "n_inliers": n_inl,
                "n_outliers": n_out,
                "max_abs_delta_centered_ns": max_abs,
                "clip": {"sigma": float(clip_sigma), "min_ns": float(clip_min_ns)},
                "paths": {
                    "points_csv": str(diag_points_csv.relative_to(root)).replace("\\", "/"),
                    "outliers_csv": str(outliers_csv.relative_to(root)).replace("\\", "/") if outliers_csv.exists() else None,
                    "plot": str(outliers_plot.relative_to(root)).replace("\\", "/"),
                },
                "counts": {
                    "by_station": {str(k): int(v) for k, v in (o["station"].value_counts().to_dict() if n_out else {}).items()},
                    "by_target": {str(k): int(v) for k, v in (o["target"].value_counts().to_dict() if n_out else {}).items()},
                    "by_year_month": {
                        str(k): int(v) for k, v in (o["year_month"].value_counts().sort_index().to_dict() if n_out else {}).items()
                    },
                },
            }
            outliers_summary_path.write_text(json.dumps(out_sum, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

            # Plot: where outliers occur (time/elevation) and how many (station/month)
            _set_japanese_font()
            fig, axs = plt.subplots(2, 2, figsize=(12.5, 7.2))

            for ax in axs.ravel():
                ax.grid(True, alpha=0.25)

            # (1) time vs |delta_centered|

            ax = axs[0, 0]
            # 条件分岐: `n_out` を満たす経路を評価する。
            if n_out:
                t = pd.to_datetime(o["epoch_utc"], utc=True, errors="coerce")
                y = np.abs(pd.to_numeric(o["delta_best_centered_ns"], errors="coerce").to_numpy(dtype=float))
                ok = np.isfinite(y) & t.notna().to_numpy(dtype=bool)
                # 条件分岐: `np.any(ok)` を満たす経路を評価する。
                if np.any(ok):
                    x_time = t[ok].dt.tz_convert(None).to_numpy(dtype="datetime64[ns]")
                    ax.scatter(x_time, y[ok], s=30, alpha=0.85)
                    ax.set_yscale("log")

            ax.set_title("外れ値：時刻 vs |Δ|（中心化, log）")
            ax.set_ylabel("|Δ| [ns]（観測-モデル, 反射器ごと中央値で中心化）")
            ax.set_xlabel("UTC時刻")

            # (2) elevation vs |delta_centered|
            ax = axs[0, 1]
            # 条件分岐: `n_out` を満たす経路を評価する。
            if n_out:
                x = pd.to_numeric(o["elev_mean_deg"], errors="coerce").to_numpy(dtype=float)
                y = np.abs(pd.to_numeric(o["delta_best_centered_ns"], errors="coerce").to_numpy(dtype=float))
                ok = np.isfinite(x) & np.isfinite(y)
                # 条件分岐: `np.any(ok)` を満たす経路を評価する。
                if np.any(ok):
                    ax.scatter(x[ok], y[ok], s=30, alpha=0.85)
                    ax.set_yscale("log")

            ax.set_title("外れ値：平均仰角 vs |Δ|（中心化, log）")
            ax.set_xlabel("平均仰角 [deg]（上り/下りの平均）")
            ax.set_ylabel("|Δ| [ns]")

            # (3) counts by station
            ax = axs[1, 0]
            # 条件分岐: `n_out` を満たす経路を評価する。
            if n_out:
                vc = o["station"].value_counts()
                ax.bar(vc.index.tolist(), vc.to_numpy(dtype=int))

            ax.set_title("外れ値：局別件数")
            ax.set_xlabel("観測局")
            ax.set_ylabel("件数")

            # (4) counts by month
            ax = axs[1, 1]
            # 条件分岐: `n_out` を満たす経路を評価する。
            if n_out:
                vc = o["year_month"].value_counts().sort_index()
                ax.bar(vc.index.tolist(), vc.to_numpy(dtype=int))
                ax.tick_params(axis="x", rotation=35)

            ax.set_title("外れ値：月別件数")
            ax.set_xlabel("年-月")
            ax.set_ylabel("件数")

            frac = (n_out / max(1, n_inl + n_out)) * 100.0
            fig.suptitle(
                f"{LLR_SHORT_NAME}：外れ値（スパイク）概要  n_out={n_out} / n_used={n_inl+n_out}（{frac:.4f}%）, max|Δ|={max_abs:.3g} ns",
                fontsize=12,
            )
            plt.tight_layout(rect=[0, 0.02, 1, 0.95])
            plt.savefig(outliers_plot, dpi=200)
            plt.close()
        except Exception as e:
            print(f"[warn] outlier summary/plot failed: {e}")

        # Diagnostics: outlier root-cause (time-tag sensitivity / clustering)

        try:
            o = diag_df.loc[diag_df["outlier_best"] == True].copy()  # noqa: E712
            # 条件分岐: `o.empty` を満たす経路を評価する。
            if o.empty:
                _save_placeholder_plot_png(
                    out_dir / "llr_outliers_time_tag_sensitivity.png",
                    f"{LLR_SHORT_NAME}：外れ値の time-tag 感度（tx/rx/mid）",
                    [
                        "外れ値が検出されなかったため、time-tag 感度解析は N/A。",
                        f"判定条件: clip_sigma={clip_sigma}, clip_min_ns={clip_min_ns:.0f} ns。",
                        "（この図は論文埋め込み用のプレースホルダとして出力）",
                    ],
                )
                _save_placeholder_plot_png(
                    out_dir / "llr_outliers_target_mixing_sensitivity.png",
                    f"{LLR_SHORT_NAME}：外れ値のターゲット混入感度（現ターゲット vs 推定ターゲット）",
                    [
                        "外れ値が検出されなかったため、ターゲット混入感度解析は N/A。",
                        f"判定条件: clip_sigma={clip_sigma}, clip_min_ns={clip_min_ns:.0f} ns。",
                        "（この図は論文埋め込み用のプレースホルダとして出力）",
                    ],
                )
            else:
                o["delta_centered_ns"] = pd.to_numeric(o["delta_best_centered_ns"], errors="coerce")
                o["abs_delta_centered_ns"] = np.abs(o["delta_centered_ns"].to_numpy(dtype=float))
                o["cluster_station_month_n"] = (
                    o.groupby(["station", "year_month"], dropna=False)["delta_centered_ns"].transform("count").astype(int)
                )
                o["cluster_station_month_med_ns"] = o.groupby(["station", "year_month"], dropna=False)["delta_centered_ns"].transform(
                    "median"
                )

                def _cause_hint(row: pd.Series) -> str:
                    a = row.get("abs_delta_centered_ns")
                    # 条件分岐: `a is None or not np.isfinite(float(a))` を満たす経路を評価する。
                    if a is None or not np.isfinite(float(a)):
                        return "不明（数値欠損）"

                    a = float(a)
                    elev = row.get("elev_mean_deg")
                    elev = float(elev) if elev is not None and np.isfinite(float(elev)) else float("nan")
                    n_cluster = int(row.get("cluster_station_month_n") or 0)
                    # Heuristics (evidence-based categories; not a physical claim)
                    if a >= 1e5:
                        return "巨大スパイク（TOF異常/記録混入の可能性）"

                    # 条件分岐: `n_cluster >= 2 and a >= 100.0` を満たす経路を評価する。

                    if n_cluster >= 2 and a >= 100.0:
                        return "局の系統オフセット段差（同月に複数）"

                    # 条件分岐: `np.isfinite(elev) and elev < 20.0 and a >= 100.0` を満たす経路を評価する。

                    if np.isfinite(elev) and elev < 20.0 and a >= 100.0:
                        return "低仰角（対流圏/測距ノイズ疑い）"

                    # 条件分岐: `a >= 500.0` を満たす経路を評価する。

                    if a >= 500.0:
                        return "中スパイク（単発）"

                    return "小スパイク（単発）"

                o["cause_hint"] = o.apply(_cause_hint, axis=1)

                # Time-tag sensitivity: would another tag (tx/rx/mid) reduce the centered delta?
                # Offline-friendly: first try to reuse Horizons caches (offline=True); if missing and online is allowed,
                # fall back to a live request to populate caches. This enables "2nd run = offline replay".
                modes_try = ["tx", "rx", "mid"]
                time_tag_cols: list[str] = []
                base = o.copy()
                need_cols = ["epoch_utc", "station", "target", "tof_obs_s", "pressure_hpa", "temp_k", "rh_percent", "met_source"]
                base = base[[c for c in need_cols if c in base.columns]].copy()
                # Keep a stable join key (future-proof; currently order is preserved).
                base["_key"] = np.arange(len(base), dtype=int)
                tof_o = pd.to_numeric(base["tof_obs_s"], errors="coerce").to_numpy(dtype=float)

                # Reflector-mixing sensitivity:
                # Check whether an outlier "disappears" if we swap the reflector (target).
                # This catches rare cases where returns are logged under the wrong reflector file.
                try:
                    refl_cat = _read_json(root / "data" / "llr" / "reflectors_de421_pa.json")
                    refls = refl_cat.get("reflectors") if isinstance(refl_cat, dict) else None
                    cand_targets: List[str] = []
                    # 条件分岐: `isinstance(refls, dict)` を満たす経路を評価する。
                    if isinstance(refls, dict):
                        for k, meta in refls.items():
                            # 条件分岐: `not k or not isinstance(meta, dict)` を満たす経路を評価する。
                            if not k or not isinstance(meta, dict):
                                continue

                            # 条件分岐: `all(kk in meta for kk in ("x_m", "y_m", "z_m"))` を満たす経路を評価する。

                            if all(kk in meta for kk in ("x_m", "y_m", "z_m")):
                                cand_targets.append(str(k))

                    cand_targets = sorted(set(cand_targets))
                except Exception:
                    cand_targets = []

                # 条件分岐: `not cand_targets` を満たす経路を評価する。

                if not cand_targets:
                    cand_targets = sorted(
                        {
                            str(t)
                            for t in base.get("target", pd.Series(dtype=object)).tolist()
                            if t and str(t).lower() not in ("nan", "na")
                        }
                    )

                # The current pipeline may use a station-specific mode for each point; keep it per-row.

                if "time_tag_mode" in o.columns:
                    base["_mode"] = o["time_tag_mode"].astype(str).to_numpy(dtype=object)
                else:
                    base["_mode"] = str(mode)

                best_tgt_guess: list[Optional[str]] = [None] * len(base)
                best_tgt_delta_raw_ns = np.full((len(base),), np.nan, dtype=float)
                suspected_target_mixing = np.zeros((len(base),), dtype=bool)

                # 条件分岐: `cand_targets and len(cand_targets) >= 2 and len(base)` を満たす経路を評価する。
                if cand_targets and len(cand_targets) >= 2 and len(base):
                    for mm in sorted({str(x) for x in base["_mode"].tolist()}):
                        mask_mm = base["_mode"].astype(str) == str(mm)
                        # 条件分岐: `not bool(np.any(mask_mm))` を満たす経路を評価する。
                        if not bool(np.any(mask_mm)):
                            continue

                        sub = base.loc[mask_mm].copy()
                        # 条件分岐: `sub.empty` を満たす経路を評価する。
                        if sub.empty:
                            continue

                        reps: List[pd.DataFrame] = []
                        for tgt in cand_targets:
                            tmp = sub.copy()
                            tmp["target"] = str(tgt)
                            tmp["_cand_target"] = str(tgt)
                            reps.append(tmp)

                        test_df = pd.concat(reps, ignore_index=True)
                        tof_test = pd.to_numeric(test_df["tof_obs_s"], errors="coerce").to_numpy(dtype=float)

                        try:
                            # (1) offline-first: reuse cached ephemerides
                            try:
                                preds_tgt = _compute_predictions_for_mode(
                                    root=root,
                                    all_df=test_df.drop(columns=["_cand_target"], errors="ignore"),
                                    tof_obs_s=tof_test,
                                    mode=str(mm),
                                    beta=beta,
                                    offline=True,
                                    chunk=chunk,
                                    cache_dir=cache_dir,
                                    station_xyz_override=station_xyz_override or None,
                                    ocean_model=ocean_model,
                                    ocean_site_by_station=ocean_site_by_station or None,
                                )
                            except Exception:
                                # (2) if global offline is disabled, allow network to fill missing caches
                                if offline:
                                    raise

                                preds_tgt = _compute_predictions_for_mode(
                                    root=root,
                                    all_df=test_df.drop(columns=["_cand_target"], errors="ignore"),
                                    tof_obs_s=tof_test,
                                    mode=str(mm),
                                    beta=beta,
                                    offline=False,
                                    chunk=chunk,
                                    cache_dir=cache_dir,
                                    station_xyz_override=station_xyz_override or None,
                                    ocean_model=ocean_model,
                                    ocean_site_by_station=ocean_site_by_station or None,
                                )

                            pred_test = preds_tgt.tof_sr_raw_tropo_tide_s
                            delta_test = (tof_test - pred_test) * 1e9
                        except Exception as e:
                            print(f"[warn] outlier target-mixing sensitivity failed (mode={mm}): {e}")
                            continue

                        keys = pd.to_numeric(test_df["_key"], errors="coerce").to_numpy(dtype=int)
                        cands = test_df["_cand_target"].astype(str).to_numpy(dtype=object)
                        for key in sorted(set(keys.tolist())):
                            m = keys == int(key)
                            # 条件分岐: `not bool(np.any(m))` を満たす経路を評価する。
                            if not bool(np.any(m)):
                                continue

                            d = delta_test[m]
                            ok = np.isfinite(d)
                            # 条件分岐: `not bool(np.any(ok))` を満たす経路を評価する。
                            if not bool(np.any(ok)):
                                continue

                            d_ok = d[ok]
                            c_ok = cands[m][ok]
                            j = int(np.argmin(np.abs(d_ok)))
                            best_tgt_guess[int(key)] = str(c_ok[j])
                            best_tgt_delta_raw_ns[int(key)] = float(d_ok[j])

                o["best_target_guess"] = best_tgt_guess
                o["best_target_delta_raw_ns"] = best_tgt_delta_raw_ns
                o["best_target_abs_delta_raw_ns"] = np.abs(best_tgt_delta_raw_ns)

                # If a huge spike disappears by switching reflector, treat it as probable target mixing.
                try:
                    cur_abs = np.abs(pd.to_numeric(o.get("delta_best_raw_ns"), errors="coerce").to_numpy(dtype=float))
                    best_abs = np.abs(pd.to_numeric(o.get("best_target_delta_raw_ns"), errors="coerce").to_numpy(dtype=float))
                    tgt_cur = o.get("target").astype(str).to_numpy(dtype=object)
                    tgt_best = o.get("best_target_guess").astype(str).to_numpy(dtype=object)
                    mix_mask = (cur_abs >= 1e5) & np.isfinite(best_abs) & (best_abs <= 1e3) & (tgt_best != tgt_cur)
                    suspected_target_mixing[:] = mix_mask
                    # 条件分岐: `bool(np.any(mix_mask))` を満たす経路を評価する。
                    if bool(np.any(mix_mask)):
                        o.loc[mix_mask, "cause_hint"] = [
                            f"巨大スパイク（ターゲット混入の可能性: 推定={tb}）" for tb in tgt_best[mix_mask].tolist()
                        ]
                except Exception:
                    pass

                o["suspected_target_mixing"] = suspected_target_mixing

                for mm in modes_try:
                    col = f"abs_delta_centered_{mm}_ns"
                    time_tag_cols.append(col)
                    try:
                        # (1) offline-first: reuse a superset Horizons cache if available
                        try:
                            preds_mm = _compute_predictions_for_mode(
                                root=root,
                                all_df=base.drop(columns=["_key"]).copy(),
                                tof_obs_s=tof_o,
                                mode=str(mm),
                                beta=beta,
                                offline=True,
                                chunk=chunk,
                                cache_dir=cache_dir,
                                station_xyz_override=station_xyz_override or None,
                                ocean_model=ocean_model,
                                ocean_site_by_station=ocean_site_by_station or None,
                            )
                        except Exception:
                            # (2) if global offline is disabled, allow network to fill missing caches
                            if offline:
                                raise

                            preds_mm = _compute_predictions_for_mode(
                                root=root,
                                all_df=base.drop(columns=["_key"]).copy(),
                                tof_obs_s=tof_o,
                                mode=str(mm),
                                beta=beta,
                                offline=False,
                                chunk=chunk,
                                cache_dir=cache_dir,
                                station_xyz_override=station_xyz_override or None,
                                ocean_model=ocean_model,
                                ocean_site_by_station=ocean_site_by_station or None,
                            )

                        pred_mm = preds_mm.tof_sr_raw_tropo_tide_s
                        delta_raw_mm = (tof_o - pred_mm) * 1e9
                        # Center by the inlier median offset (best-mode), so values are comparable.
                        med = pd.to_numeric(o["delta_best_med_ns"], errors="coerce").to_numpy(dtype=float)
                        delta_ctr_mm = delta_raw_mm - med
                        o[col] = np.abs(delta_ctr_mm)
                    except Exception as e:
                        o[col] = np.nan
                        print(f"[warn] outlier time-tag sensitivity failed (mode={mm}): {e}")

                # Pick the best mode per outlier

                abs_mat = np.stack([pd.to_numeric(o[c], errors="coerce").to_numpy(dtype=float) for c in time_tag_cols], axis=1)
                best_idx = np.full((len(o),), -1, dtype=int)
                for i in range(len(o)):
                    row = abs_mat[i, :]
                    ok = np.isfinite(row)
                    # 条件分岐: `not np.any(ok)` を満たす経路を評価する。
                    if not np.any(ok):
                        continue

                    best_idx[i] = int(np.argmin(row[ok]))  # argmin over filtered array
                    # map back to original modes_try index
                    best_idx[i] = int(np.arange(len(row))[ok][best_idx[i]])

                o["best_time_tag_mode"] = [modes_try[i] if i >= 0 else None for i in best_idx.tolist()]
                o["best_abs_delta_centered_ns"] = [
                    float(abs_mat[i, j]) if j >= 0 and np.isfinite(abs_mat[i, j]) else float("nan") for i, j in enumerate(best_idx.tolist())
                ]

                # Emit diagnosis CSV + plot
                o = o.sort_values("abs_delta_centered_ns", ascending=False, kind="mergesort").reset_index(drop=True)
                out_diag_csv = out_dir / "llr_outliers_diagnosis.csv"
                o.to_csv(out_diag_csv, index=False)

                # Plot: time-tag sensitivity (abs centered delta)
                try:
                    # 条件分岐: `time_tag_cols and int(len(o)) > 0` を満たす経路を評価する。
                    if time_tag_cols and int(len(o)) > 0:
                        _set_japanese_font()
                        x = np.arange(len(o), dtype=float)
                        width = 0.26
                        plt.figure(figsize=(12.8, 4.8))
                        colors = {"tx": "#1f77b4", "rx": "#ff7f0e", "mid": "#2ca02c"}
                        for j, mm in enumerate(modes_try):
                            y = pd.to_numeric(o[f"abs_delta_centered_{mm}_ns"], errors="coerce").to_numpy(dtype=float)
                            plt.bar(x + (j - 1) * width, y, width=width, label=f"{mm}", color=colors.get(mm))

                        plt.yscale("log")
                        plt.axhline(float(clip_min_ns), color="#333333", lw=1.1, alpha=0.6, linestyle="--", label=f"外れ値閾値 {clip_min_ns:.0f} ns")
                        plt.xticks(x, [str(i + 1) for i in range(len(o))])
                        plt.xlabel("外れ値ID（降順）")
                        plt.ylabel("|Δ| [ns]（観測-モデル, 反射器別中央値で中心化）")
                        plt.title(f"{LLR_SHORT_NAME}：外れ値の time-tag 感度（tx/rx/mid）")
                        plt.legend(ncols=4, fontsize=9)
                        plt.grid(True, axis="y", alpha=0.25)
                        plt.tight_layout()
                        plt.savefig(out_dir / "llr_outliers_time_tag_sensitivity.png", dpi=200)
                        plt.close()
                except Exception as e:
                    print(f"[warn] outlier time-tag sensitivity plot failed: {e}")

                # Plot: target mixing sensitivity (current target vs best alternative target)

                try:
                    # 条件分岐: `"best_target_delta_raw_ns" in o.columns and "delta_best_raw_ns" in o.columns...` を満たす経路を評価する。
                    if "best_target_delta_raw_ns" in o.columns and "delta_best_raw_ns" in o.columns and int(len(o)) > 0:
                        _set_japanese_font()
                        x = np.arange(len(o), dtype=float)
                        width = 0.38
                        cur_abs = np.abs(pd.to_numeric(o["delta_best_raw_ns"], errors="coerce").to_numpy(dtype=float))
                        best_abs = np.abs(pd.to_numeric(o["best_target_delta_raw_ns"], errors="coerce").to_numpy(dtype=float))

                        plt.figure(figsize=(12.8, 4.8))
                        plt.bar(x - width / 2, cur_abs, width=width, label="現在ターゲット |Δ_raw|", color="#1f77b4", alpha=0.9)
                        plt.bar(x + width / 2, best_abs, width=width, label="推定ターゲット |Δ_raw|", color="#ff7f0e", alpha=0.9)
                        plt.yscale("log")
                        plt.axhline(1e5, color="#333333", lw=1.0, alpha=0.5, linestyle="--", label="混入判定: |Δ_raw|≥1e5 ns")
                        plt.axhline(1e3, color="#666666", lw=1.0, alpha=0.35, linestyle="--", label="混入判定: best |Δ_raw|≤1e3 ns")
                        plt.xticks(x, [str(i + 1) for i in range(len(o))])
                        plt.xlabel("外れ値ID（降順）")
                        plt.ylabel("|Δ_raw| [ns]（観測-モデル, オフセット未除去）")
                        plt.title(f"{LLR_SHORT_NAME}：外れ値のターゲット混入感度（現ターゲット vs 推定ターゲット）")
                        plt.legend(ncols=4, fontsize=9)
                        plt.grid(True, axis="y", alpha=0.25)
                        plt.tight_layout()
                        plt.savefig(out_dir / "llr_outliers_target_mixing_sensitivity.png", dpi=200)
                        plt.close()
                except Exception as e:
                    print(f"[warn] outlier target-mixing sensitivity plot failed: {e}")

                # Summary JSON (for report)

                try:
                    by_cause = o["cause_hint"].value_counts().to_dict()

                    # time-tag sensitivity summary
                    best_not_current = 0
                    computed_modes: list[str] = []
                    # 条件分岐: `"best_time_tag_mode" in o.columns and "time_tag_mode" in o.columns` を満たす経路を評価する。
                    if "best_time_tag_mode" in o.columns and "time_tag_mode" in o.columns:
                        cur = o["time_tag_mode"].astype(str).to_numpy(dtype=object)
                        best = o["best_time_tag_mode"].astype(str).to_numpy(dtype=object)
                        for a, b in zip(cur.tolist(), best.tolist()):
                            # 条件分岐: `b and b not in ("None", "nan", "NaN") and str(a).strip().lower() != str(b).st...` を満たす経路を評価する。
                            if b and b not in ("None", "nan", "NaN") and str(a).strip().lower() != str(b).strip().lower():
                                best_not_current += 1

                        for mm in modes_try:
                            c = f"abs_delta_centered_{mm}_ns"
                            # 条件分岐: `c in o.columns` を満たす経路を評価する。
                            if c in o.columns:
                                y = pd.to_numeric(o[c], errors="coerce").to_numpy(dtype=float)
                                # 条件分岐: `np.isfinite(y).any()` を満たす経路を評価する。
                                if np.isfinite(y).any():
                                    computed_modes.append(mm)

                    # target mixing summary

                    n_target_mix = 0
                    # 条件分岐: `"suspected_target_mixing" in o.columns` を満たす経路を評価する。
                    if "suspected_target_mixing" in o.columns:
                        try:
                            n_target_mix = int(np.sum(o["suspected_target_mixing"].to_numpy(dtype=bool)))
                        except Exception:
                            n_target_mix = 0

                    out_diag_sum = {
                        "generated_utc": datetime.now(timezone.utc).isoformat(),
                        "n_outliers": int(len(o)),
                        "by_cause_hint": {str(k): int(v) for k, v in by_cause.items()},
                        "time_tag_sensitivity": {
                            "computed": bool(computed_modes),
                            "computed_modes": computed_modes,
                            "n_best_mode_differs": int(best_not_current),
                            "modes": modes_try,
                        },
                        "target_mixing_sensitivity": {
                            "computed": bool("best_target_guess" in o.columns),
                            "n_suspected": int(n_target_mix),
                            "criteria": {
                                "abs_delta_raw_ge_ns": 1e5,
                                "best_abs_delta_raw_le_ns": 1e3,
                                "best_target_differs": True,
                            },
                            "policy": {
                                "baseline": "exclude",
                                "auto_reassign": False,
                                "note": "疑いがある場合も観測のターゲットラベルは自動では書き換えず、統計から除外し、source_file:lineno で一次データ行へ追跡して確認する。",
                            },
                        },
                        "paths": {
                            "diagnosis_csv": str(out_diag_csv.relative_to(root)).replace("\\", "/"),
                            "time_tag_plot": str((out_dir / "llr_outliers_time_tag_sensitivity.png").relative_to(root)).replace(
                                "\\", "/"
                            ),
                            "target_mixing_plot": str((out_dir / "llr_outliers_target_mixing_sensitivity.png").relative_to(root)).replace(
                                "\\", "/"
                            ),
                        },
                    }
                    (out_dir / "llr_outliers_diagnosis_summary.json").write_text(
                        json.dumps(out_diag_sum, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                    )
                except Exception as e:
                    print(f"[warn] outlier diagnosis summary json failed: {e}")
        except Exception as e:
            print(f"[warn] outlier diagnosis failed: {e}")

        # Plot: residual distribution (obs - model) to make "how different" clear for non-experts.
        # Use inliers only (robust-gated per station×target) and the offset-aligned residuals.

        try:
            inl = diag_df["inlier_best"].to_numpy(dtype=bool)
            res_sr = pd.to_numeric(diag_df.loc[inl, "residual_sr_ns"], errors="coerce").to_numpy(dtype=float)
            res_tropo = pd.to_numeric(diag_df.loc[inl, "residual_sr_tropo_ns"], errors="coerce").to_numpy(dtype=float)
            res_final = pd.to_numeric(diag_df.loc[inl, "residual_sr_tropo_tide_ns"], errors="coerce").to_numpy(dtype=float)
            res_sr = res_sr[np.isfinite(res_sr)]
            res_tropo = res_tropo[np.isfinite(res_tropo)]
            res_final = res_final[np.isfinite(res_final)]

            # 条件分岐: `len(res_final) >= 100` を満たす経路を評価する。
            if len(res_final) >= 100:
                def _rms(x: np.ndarray) -> float:
                    x = x[np.isfinite(x)]
                    return float(np.sqrt(np.mean(x * x))) if len(x) else float("nan")

                def _ecdf_abs(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
                    x = np.abs(x[np.isfinite(x)])
                    # 条件分岐: `not len(x)` を満たす経路を評価する。
                    if not len(x):
                        return np.array([]), np.array([])

                    xs = np.sort(x)
                    ys = np.arange(1, len(xs) + 1, dtype=float) / float(len(xs))
                    return xs, ys

                # Range for histogram based on robust percentiles.

                pool = np.concatenate([np.abs(res_sr), np.abs(res_tropo), np.abs(res_final)]) if len(res_tropo) else np.concatenate([np.abs(res_sr), np.abs(res_final)])
                pool = pool[np.isfinite(pool)]
                p99 = float(np.nanpercentile(pool, 99)) if len(pool) else 30.0
                lim = max(10.0, min(200.0, p99 * 1.2))
                bins = np.linspace(-lim, lim, num=61)

                _set_japanese_font()
                fig, axs = plt.subplots(1, 2, figsize=(12.6, 4.6))
                for ax in axs:
                    ax.grid(True, alpha=0.25)

                # (1) signed residual histogram

                ax = axs[0]
                ax.hist(res_sr, bins=bins, alpha=0.35, label=f"SR（RMS={_rms(res_sr):.3f} ns）")
                # 条件分岐: `len(res_tropo)` を満たす経路を評価する。
                if len(res_tropo):
                    ax.hist(res_tropo, bins=bins, alpha=0.35, label=f"SR+Tropo（RMS={_rms(res_tropo):.3f} ns）")

                ax.hist(res_final, bins=bins, alpha=0.55, label=f"SR+Tropo+Tide（RMS={_rms(res_final):.3f} ns）")
                ax.axvline(0.0, color="#333333", lw=1.2, alpha=0.7)
                ax.set_title("残差分布（観測−モデル）")
                ax.set_xlabel("残差 [ns]（定数オフセット整列後）")
                ax.set_ylabel("件数")
                ax.legend(fontsize=9)

                # (2) ECDF of |residual|
                ax = axs[1]
                xs, ys = _ecdf_abs(res_sr)
                # 条件分岐: `len(xs)` を満たす経路を評価する。
                if len(xs):
                    ax.plot(xs, ys, lw=2.0, label="SR")

                # 条件分岐: `len(res_tropo)` を満たす経路を評価する。

                if len(res_tropo):
                    xs, ys = _ecdf_abs(res_tropo)
                    # 条件分岐: `len(xs)` を満たす経路を評価する。
                    if len(xs):
                        ax.plot(xs, ys, lw=2.0, label="SR+Tropo")

                xs, ys = _ecdf_abs(res_final)
                # 条件分岐: `len(xs)` を満たす経路を評価する。
                if len(xs):
                    ax.plot(xs, ys, lw=2.4, label="SR+Tropo+Tide")

                for q in (0.5, 0.9, 0.95):
                    ax.axhline(q, color="#666666", lw=0.8, alpha=0.35)

                ax.set_xlim(0.0, lim)
                ax.set_ylim(0.0, 1.0)
                ax.set_title("|残差| の累積分布（小さいほど良い）")
                ax.set_xlabel("|残差| [ns]")
                ax.set_ylabel("累積割合")
                ax.legend(fontsize=9)

                n_used = int(len(res_final))
                fig.suptitle(
                    f"{LLR_SHORT_NAME}：観測−P-model の差（inlierのみ, n={n_used}）",
                    fontsize=12,
                )
                plt.tight_layout(rect=[0, 0.02, 1, 0.92])
                plt.savefig(out_dir / "llr_residual_distribution.png", dpi=200)
                plt.close()
        except Exception as e:
            print(f"[warn] residual distribution plot failed: {e}")

        def _grp_rms(x: pd.Series) -> float:
            a = x.to_numpy(dtype=float)
            a = a[np.isfinite(a)]
            return float(np.sqrt(np.mean(a * a))) if len(a) else float("nan")

        def _grp_mean(x: pd.Series) -> float:
            a = x.to_numpy(dtype=float)
            a = a[np.isfinite(a)]
            return float(np.mean(a)) if len(a) else float("nan")

        # Legacy monthly stats (base SR model) for continuity

        by_st_tgt_month = (
            diag_df.groupby(["station", "target", "year_month"], dropna=False)["residual_sr_ns"]
            .agg(n="count", mean_ns=_grp_mean, rms_ns=_grp_rms)
            .reset_index()
        )
        by_st_month = (
            diag_df.groupby(["station", "year_month"], dropna=False)["residual_sr_ns"]
            .agg(n="count", mean_ns=_grp_mean, rms_ns=_grp_rms)
            .reset_index()
        )
        (out_dir / "llr_monthly_station_target_stats.csv").write_text(by_st_tgt_month.to_csv(index=False), encoding="utf-8")
        (out_dir / "llr_monthly_station_stats.csv").write_text(by_st_month.to_csv(index=False), encoding="utf-8")

        # Plot (legacy): station RMS by month (all targets pooled)
        monthly_min_n = 30
        _set_japanese_font()
        plt.figure(figsize=(12, 4.5))
        by_st_month_plot = by_st_month[(by_st_month["n"] >= monthly_min_n) & np.isfinite(by_st_month["rms_ns"])].copy()
        for st in sorted(by_st_month_plot["station"].unique().tolist()):
            sub = by_st_month_plot[by_st_month_plot["station"] == st].copy()
            sub["t"] = pd.to_datetime(sub["year_month"] + "-01", utc=True, errors="coerce")
            sub = sub.dropna(subset=["t"]).sort_values("t")
            # 条件分岐: `sub.empty` を満たす経路を評価する。
            if sub.empty:
                continue

            x_time = sub["t"].dt.tz_convert(None).to_numpy(dtype="datetime64[ns]")
            plt.plot(x_time, sub["rms_ns"].to_numpy(dtype=float), marker="o", label=st)

        plt.ylabel(f"残差RMS [ns]（観測局→反射器, 月ごと, 全反射器, 月内 n≥{monthly_min_n}）")
        plt.title(f"{LLR_SHORT_NAME}：局ごとの残差RMS（期間依存の確認, 月内 n≥{monthly_min_n} のみ表示）")
        plt.grid(True, alpha=0.3)
        plt.legend(ncols=4, fontsize=9)
        plt.tight_layout()
        p_diag_legacy = out_dir / "llr_rms_by_station_month.png"
        plt.savefig(p_diag_legacy, dpi=200)
        plt.close()

        # Plot: residual vs elevation (helps identify troposphere / low-elevation effects)
        # - Keep legacy filename for GRSM for backward compatibility
        # - Also output per-station figures (e.g., APOL) to speed up root-cause analysis
        try:
            diag_el = diag_df[np.isfinite(diag_df["residual_sr_tropo_tide_ns"])].copy()
            diag_el = diag_el[np.isfinite(diag_el["elev_mean_deg"])].copy()
            # 条件分岐: `not diag_el.empty` を満たす経路を評価する。
            if not diag_el.empty:
                # Paper figures may be generated from a subset manifest; keep a low threshold so
                # per-station plots still exist even when n is modest.
                min_points_el_plot = 30
                for st in sorted(diag_el["station"].astype(str).unique().tolist()):
                    ssub = diag_el[diag_el["station"].astype(str) == str(st)].copy()
                    # 条件分岐: `len(ssub) < min_points_el_plot` を満たす経路を評価する。
                    if len(ssub) < min_points_el_plot:
                        continue

                    _set_japanese_font()
                    plt.figure(figsize=(10.5, 5.2))
                    for tgt in sorted(ssub["target"].astype(str).unique().tolist()):
                        sub = ssub[ssub["target"].astype(str) == str(tgt)]
                        # 条件分岐: `sub.empty` を満たす経路を評価する。
                        if sub.empty:
                            continue

                        plt.scatter(
                            sub["elev_mean_deg"].to_numpy(dtype=float),
                            sub["residual_sr_tropo_tide_ns"].to_numpy(dtype=float),
                            s=10,
                            alpha=0.25,
                            label=str(tgt),
                        )
                    # Add a binned median trend line for non-experts (reduces overplot confusion).

                    try:
                        x = ssub["elev_mean_deg"].to_numpy(dtype=float)
                        y = ssub["residual_sr_tropo_tide_ns"].to_numpy(dtype=float)
                        m = np.isfinite(x) & np.isfinite(y)
                        x = x[m]
                        y = y[m]
                        # 条件分岐: `len(x) >= 200` を満たす経路を評価する。
                        if len(x) >= 200:
                            x_min = float(np.nanpercentile(x, 1))
                            x_max = float(np.nanpercentile(x, 99))
                            # 条件分岐: `np.isfinite(x_min) and np.isfinite(x_max) and (x_max - x_min) > 10.0` を満たす経路を評価する。
                            if np.isfinite(x_min) and np.isfinite(x_max) and (x_max - x_min) > 10.0:
                                bins = np.linspace(x_min, x_max, num=11)
                                xc = 0.5 * (bins[:-1] + bins[1:])
                                med = np.full((len(xc),), np.nan, dtype=float)
                                p25 = np.full((len(xc),), np.nan, dtype=float)
                                p75 = np.full((len(xc),), np.nan, dtype=float)
                                for bi in range(len(xc)):
                                    mb = (x >= bins[bi]) & (x < bins[bi + 1])
                                    # 条件分岐: `int(np.sum(mb)) < 50` を満たす経路を評価する。
                                    if int(np.sum(mb)) < 50:
                                        continue

                                    yy = y[mb]
                                    med[bi] = float(np.nanmedian(yy))
                                    p25[bi] = float(np.nanpercentile(yy, 25))
                                    p75[bi] = float(np.nanpercentile(yy, 75))

                                ok = np.isfinite(med)
                                # 条件分岐: `int(np.sum(ok)) >= 3` を満たす経路を評価する。
                                if int(np.sum(ok)) >= 3:
                                    plt.fill_between(xc[ok], p25[ok], p75[ok], color="#111111", alpha=0.08, linewidth=0)
                                    plt.plot(xc[ok], med[ok], color="#111111", lw=2.0, marker="o", ms=4, label="全体中央値（仰角ビン）")
                    except Exception:
                        pass

                    y = ssub["residual_sr_tropo_tide_ns"].to_numpy(dtype=float)
                    y = y[np.isfinite(y)]
                    # 条件分岐: `len(y)` を満たす経路を評価する。
                    if len(y):
                        p99 = float(np.percentile(np.abs(y), 99))
                        lim = max(30.0, min(200.0, p99 * 1.15))
                        plt.ylim(-lim, lim)

                    plt.axhline(0.0, color="#333333", lw=1.0, alpha=0.6)
                    plt.xlabel("平均仰角 [deg]（上り/下りの平均）")
                    plt.ylabel("残差 [ns]（SR+Tropo+Tide, 定数オフセット整列後）")
                    plt.title(f"{LLR_SHORT_NAME}：{st} 残差と仰角（低仰角での系統誤差の有無を確認）")
                    plt.grid(True, alpha=0.25)
                    plt.legend(ncols=3, fontsize=9)
                    plt.tight_layout()
                    # 条件分岐: `str(st).upper() == "GRSM"` を満たす経路を評価する。
                    if str(st).upper() == "GRSM":
                        out_name = "llr_grsm_residual_vs_elevation.png"
                    else:
                        out_name = f"llr_{str(st).strip().lower()}_residual_vs_elevation.png"

                    plt.savefig(out_dir / out_name, dpi=200)
                    plt.close()
        except Exception as e:
            print(f"[warn] residual vs elevation plot failed: {e}")

        # Diagnostics: per-station correlation / worst months (for root-cause analysis)

        try:
            diag_ok = diag_df[np.isfinite(diag_df["residual_sr_tropo_tide_ns"])].copy()
            # 条件分岐: `not diag_ok.empty` を満たす経路を評価する。
            if not diag_ok.empty:
                diag_ok["epoch_utc"] = pd.to_datetime(diag_ok["epoch_utc"], utc=True, errors="coerce", format="mixed")
                diag_ok = diag_ok.dropna(subset=["epoch_utc"])
                diag_ok["ym"] = diag_ok["epoch_utc"].dt.strftime("%Y-%m")

                def _rms_series_ns(x: pd.Series) -> float:
                    a = x.to_numpy(dtype=float)
                    a = a[np.isfinite(a)]
                    return float(np.sqrt(np.mean(a * a))) if len(a) else float("nan")

                def _corr(a: pd.Series, b: pd.Series) -> float:
                    x = a.to_numpy(dtype=float)
                    y = b.to_numpy(dtype=float)
                    m = np.isfinite(x) & np.isfinite(y)
                    # 条件分岐: `int(np.sum(m)) < 3` を満たす経路を評価する。
                    if int(np.sum(m)) < 3:
                        return float("nan")

                    return float(np.corrcoef(x[m], y[m])[0, 1])

                diag_ok["m_1_sin"] = 1.0 / np.maximum(
                    np.sin(np.deg2rad(diag_ok["elev_mean_deg"].to_numpy(dtype=float))),
                    float(np.sin(np.deg2rad(5.0))),
                )

                st_diag: Dict[str, Dict[str, Any]] = {}
                for st in sorted(diag_ok["station"].astype(str).unique().tolist()):
                    ssub = diag_ok[diag_ok["station"].astype(str) == str(st)].copy()
                    # 条件分岐: `ssub.empty` を満たす経路を評価する。
                    if ssub.empty:
                        continue
                    # Worst month×target (require a minimum sample size per cell)

                    by = (
                        ssub.groupby(["ym", "target"], dropna=False)["residual_sr_tropo_tide_ns"]
                        .agg(n="count", rms_ns=_rms_series_ns)
                        .reset_index()
                    )
                    by = by[by["n"] >= 30].sort_values("rms_ns", ascending=False)
                    worst = []
                    for _, r in by.head(8).iterrows():
                        worst.append(
                            {
                                "ym": str(r["ym"]),
                                "target": str(r["target"]),
                                "n": int(r["n"]),
                                "rms_ns": float(r["rms_ns"]),
                            }
                        )

                    st_diag[str(st)] = {
                        "n": int(len(ssub)),
                        "rms_sr_tropo_tide_ns": float(_rms_series_ns(ssub["residual_sr_tropo_tide_ns"])),
                        "corr_res_vs_elev_deg": float(_corr(ssub["residual_sr_tropo_tide_ns"], ssub["elev_mean_deg"])),
                        "corr_res_vs_m_1_sin": float(_corr(ssub["residual_sr_tropo_tide_ns"], ssub["m_1_sin"])),
                        "worst_month_target": worst,
                    }

                diag_path = out_dir / "llr_station_diagnostics.json"
                diag_path.write_text(
                    json.dumps(
                        {"generated_utc": datetime.now(timezone.utc).isoformat(), "station_diagnostics": st_diag},
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
        except Exception as e:
            print(f"[warn] station diagnostics failed: {e}")

        # Extended monthly stats (model variants)

        long_frames: List[pd.DataFrame] = []
        for model, col in [
            ("SR", "residual_sr_ns"),
            ("SR+Tropo", "residual_sr_tropo_ns"),
            ("SR+Tropo+Tide", "residual_sr_tropo_tide_ns"),
        ]:
            tmp = diag_df[["station", "target", "year_month", col]].rename(columns={col: "residual_ns"})
            tmp = tmp.assign(model=model)
            long_frames.append(tmp)

        diag_long = pd.concat(long_frames, ignore_index=True)

        by_st_month_model = (
            diag_long.groupby(["model", "station", "year_month"], dropna=False)["residual_ns"]
            .agg(n="count", mean_ns=_grp_mean, rms_ns=_grp_rms)
            .reset_index()
        )
        by_st_tgt_month_model = (
            diag_long.groupby(["model", "station", "target", "year_month"], dropna=False)["residual_ns"]
            .agg(n="count", mean_ns=_grp_mean, rms_ns=_grp_rms)
            .reset_index()
        )
        by_st_month_model.to_csv(out_dir / "llr_monthly_station_stats_models.csv", index=False)
        by_st_tgt_month_model.to_csv(out_dir / "llr_monthly_station_target_stats_models.csv", index=False)

        # Plot: station RMS by month (subplots; compare model variants)
        _set_japanese_font()
        by_st_month_model_plot = by_st_month_model[
            (by_st_month_model["n"] >= monthly_min_n) & np.isfinite(by_st_month_model["rms_ns"])
        ].copy()
        stations_u = sorted(by_st_month_model_plot["station"].unique().tolist())
        ncols = 2
        nrows = int(np.ceil(len(stations_u) / ncols)) if stations_u else 1
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 4.2 * nrows), squeeze=False)
        for ax in axes.flat:
            ax.grid(True, alpha=0.25)

        for idx, st in enumerate(stations_u):
            ax = axes[idx // ncols][idx % ncols]
            sub = by_st_month_model_plot[by_st_month_model_plot["station"] == st].copy()
            sub["t"] = pd.to_datetime(sub["year_month"] + "-01", utc=True, errors="coerce")
            sub = sub.dropna(subset=["t"]).sort_values("t")
            for model in ["SR", "SR+Tropo", "SR+Tropo+Tide"]:
                s2 = sub[sub["model"] == model].sort_values("t")
                # 条件分岐: `s2.empty` を満たす経路を評価する。
                if s2.empty:
                    continue

                x_time = s2["t"].dt.tz_convert(None).to_numpy(dtype="datetime64[ns]")
                ax.plot(x_time, s2["rms_ns"].to_numpy(dtype=float), marker="o", label=model)

            ax.set_title(str(st))
            ax.set_ylabel(f"残差RMS [ns]（月ごと, 全反射器, 定数整列, 月内 n≥{monthly_min_n}）")
            ax.legend(fontsize=9)

        for j in range(len(stations_u), nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")

        fig.suptitle(f"{LLR_SHORT_NAME}：局ごとの残差RMS（期間依存 × モデル切り分け, 月内 n≥{monthly_min_n} のみ表示）", y=0.995)
        fig.tight_layout()
        p_diag1 = out_dir / "llr_rms_by_station_month_models.png"
        fig.savefig(p_diag1, dpi=200)
        plt.close(fig)

        # Plot: GRSM RMS by month by target (base model; kept for continuity)
        if "GRSM" in set(by_st_tgt_month["station"].tolist()):
            sub = by_st_tgt_month[(by_st_tgt_month["station"] == "GRSM") & (by_st_tgt_month["n"] >= monthly_min_n)].copy()
            sub = sub[np.isfinite(sub["rms_ns"])].copy()
            sub["t"] = pd.to_datetime(sub["year_month"] + "-01", utc=True, errors="coerce")
            sub = sub.dropna(subset=["t"]).sort_values("t")
            # 条件分岐: `not sub.empty` を満たす経路を評価する。
            if not sub.empty:
                plt.figure(figsize=(12, 4.5))
                for tgt in sorted(sub["target"].unique().tolist()):
                    s2 = sub[sub["target"] == tgt].sort_values("t")
                    x_time = s2["t"].dt.tz_convert(None).to_numpy(dtype="datetime64[ns]")
                    plt.plot(x_time, s2["rms_ns"].to_numpy(dtype=float), marker="o", label=tgt)

                plt.ylabel(f"残差RMS [ns]（月ごと, 定数整列, 月内 n≥{monthly_min_n}）")
                plt.title(f"{LLR_SHORT_NAME}：GRSM 残差RMS（反射器別の期間依存, 月内 n≥{monthly_min_n} のみ表示）")
                plt.grid(True, alpha=0.3)
                plt.legend(ncols=5, fontsize=9)
                plt.tight_layout()
                p_diag2 = out_dir / "llr_grsm_rms_by_target_month.png"
                plt.savefig(p_diag2, dpi=200)
                plt.close()

        # Plot: GRSM model ablation by month (targets pooled)

        if "GRSM" in set(by_st_month_model["station"].tolist()):
            sub = by_st_month_model_plot[by_st_month_model_plot["station"] == "GRSM"].copy()
            sub["t"] = pd.to_datetime(sub["year_month"] + "-01", utc=True, errors="coerce")
            sub = sub.dropna(subset=["t"]).sort_values("t")
            # 条件分岐: `not sub.empty` を満たす経路を評価する。
            if not sub.empty:
                plt.figure(figsize=(12, 4.5))
                for model in ["SR", "SR+Tropo", "SR+Tropo+Tide"]:
                    s2 = sub[sub["model"] == model].sort_values("t")
                    # 条件分岐: `s2.empty` を満たす経路を評価する。
                    if s2.empty:
                        continue

                    x_time = s2["t"].dt.tz_convert(None).to_numpy(dtype="datetime64[ns]")
                    plt.plot(x_time, s2["rms_ns"].to_numpy(dtype=float), marker="o", label=model)

                plt.ylabel(f"残差RMS [ns]（月ごと, 全反射器, 定数整列, 月内 n≥{monthly_min_n}）")
                plt.title(f"{LLR_SHORT_NAME}：GRSM 残差RMS（モデル切り分けで原因を見る, 月内 n≥{monthly_min_n} のみ表示）")
                plt.grid(True, alpha=0.3)
                plt.legend(ncols=3, fontsize=9)
                plt.tight_layout()
                p_diag3 = out_dir / "llr_grsm_rms_by_month_models.png"
                plt.savefig(p_diag3, dpi=200)
                plt.close()
    except Exception as e:
        print(f"[warn] diagnostics failed: {e}")

    # Summary JSON

    def _median(col: str) -> float:
        v = metrics_df[col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        return float(np.median(v)) if len(v) else float("nan")

    def _point_weighted_rms(col: str) -> float:
        # 条件分岐: `col not in metrics_df.columns or "n" not in metrics_df.columns` を満たす経路を評価する。
        if col not in metrics_df.columns or "n" not in metrics_df.columns:
            return float("nan")

        rv = pd.to_numeric(metrics_df[col], errors="coerce").to_numpy(dtype=float)
        nv = pd.to_numeric(metrics_df["n"], errors="coerce").to_numpy(dtype=float)
        ok = np.isfinite(rv) & np.isfinite(nv) & (nv > 0)
        # 条件分岐: `not np.any(ok)` を満たす経路を評価する。
        if not np.any(ok):
            return float("nan")

        return float(np.sqrt(np.sum(nv[ok] * rv[ok] * rv[ok]) / np.sum(nv[ok])))

    def _diag_rms(
        *,
        modern_start_year: int = 2023,
    ) -> Dict[str, Any]:
        try:
            d = diag_df.copy()
            d["epoch_utc"] = pd.to_datetime(d["epoch_utc"], utc=True, errors="coerce")
            d["year"] = d["epoch_utc"].dt.year
            d = d[d["inlier_best"] == True]  # noqa: E712
            d["residual_sr_tropo_tide_ns"] = pd.to_numeric(d["residual_sr_tropo_tide_ns"], errors="coerce")
            d = d[np.isfinite(d["residual_sr_tropo_tide_ns"])]
            # 条件分岐: `d.empty` を満たす経路を評価する。
            if d.empty:
                return {
                    "modern_start_year": int(modern_start_year),
                    "all_modern_rms_ns": float("nan"),
                    "all_modern_n": 0,
                    "apol_modern_rms_ns": float("nan"),
                    "apol_modern_n": 0,
                    "apol_modern_ex_nglr1_rms_ns": float("nan"),
                    "apol_modern_ex_nglr1_n": 0,
                }

            dm = d[d["year"] >= int(modern_start_year)].copy()
            def _rms_of(sub: pd.DataFrame) -> float:
                x = pd.to_numeric(sub["residual_sr_tropo_tide_ns"], errors="coerce").to_numpy(dtype=float)
                x = x[np.isfinite(x)]
                return float(np.sqrt(np.mean(x * x))) if len(x) else float("nan")

            ap = dm[dm["station"].astype(str).str.upper() == "APOL"].copy()
            ap_ex_ng = ap[ap["target"].astype(str).str.lower() != "nglr1"].copy()
            return {
                "modern_start_year": int(modern_start_year),
                "all_modern_rms_ns": _rms_of(dm),
                "all_modern_n": int(len(dm)),
                "apol_modern_rms_ns": _rms_of(ap),
                "apol_modern_n": int(len(ap)),
                "apol_modern_ex_nglr1_rms_ns": _rms_of(ap_ex_ng),
                "apol_modern_ex_nglr1_n": int(len(ap_ex_ng)),
            }
        except Exception:
            return {
                "modern_start_year": int(modern_start_year),
                "all_modern_rms_ns": float("nan"),
                "all_modern_n": 0,
                "apol_modern_rms_ns": float("nan"),
                "apol_modern_n": 0,
                "apol_modern_ex_nglr1_rms_ns": float("nan"),
                "apol_modern_ex_nglr1_n": 0,
            }

    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(manifest_path.relative_to(root)).replace("\\", "/"),
        "n_files": int(len(file_recs)),
        "n_points_total": int(len(all_df)),
        "n_bad_tof_dropped": int(n_bad_tof),
        "n_groups": int(len(metrics_df)),
        "beta": beta,
        "time_tag_mode": str(mode),
        "station_coords_mode": station_coords_mode,
        "station_override_json": (station_override_json or None),
        "station_override_input": station_override_input,
        "pos_eop_date_preferred": pos_eop_date or None,
        "pos_eop_max_days": int(pos_eop_max_days),
        "ocean_loading_mode": ocean_loading_mode,
        "ocean_loading_harpos": (
            str(ocean_harpos_path.relative_to(root)).replace("\\", "/") if ocean_model is not None else None
        ),
        "ocean_loading_station_map": (ocean_site_info or None),
        "outlier_clip_sigma": float(clip_sigma),
        "outlier_clip_ns": float(clip_min_ns),
        "station_coord_summary": station_coord_summary,
        "time_tag_mode_by_station": time_tag_mode_by_station,
        "min_points_per_group": min_points,
        "stations": sorted(set(metrics_df["station"].tolist())),
        "targets": sorted(set(metrics_df["target"].tolist())),
        "median_rms_ns": {
            "geocenter_moon": _median("rms_gc_ns"),
            "station_moon": _median("rms_sm_ns"),
            "station_reflector": _median("rms_sr_ns"),
            "station_reflector_no_shapiro": _median("rms_sr_no_shapiro_ns"),
            "station_reflector_tropo": _median("rms_sr_tropo_ns"),
            "station_reflector_tropo_station_tide": _median("rms_sr_tropo_station_tide_ns"),
            "station_reflector_tropo_moon_tide": _median("rms_sr_tropo_moon_tide_ns"),
            "station_reflector_tropo_tide_no_ocean": _median("rms_sr_tropo_tide_no_ocean_ns"),
            "station_reflector_tropo_tide": _median("rms_sr_tropo_tide_ns"),
            "station_reflector_tropo_no_shapiro": _median("rms_sr_tropo_no_shapiro_ns"),
            "station_reflector_tropo_earth_shapiro": _median("rms_sr_tropo_earth_shapiro_ns"),
            "station_reflector_iau": _median("rms_sr_iau_ns"),
        },
        "point_weighted_rms_ns": {
            "station_reflector": _point_weighted_rms("rms_sr_ns"),
            "station_reflector_tropo": _point_weighted_rms("rms_sr_tropo_ns"),
            "station_reflector_tropo_tide": _point_weighted_rms("rms_sr_tropo_tide_ns"),
        },
        "modern_subset_rms_ns": _diag_rms(modern_start_year=2023),
    }
    summary_path = out_dir / "llr_batch_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Plot 1: overall improvement (median RMS)
    _set_japanese_font()
    labels = ["地球中心→月中心", "観測局→月中心", "観測局→反射器"]
    vals_ns = [
        summary["median_rms_ns"]["geocenter_moon"],
        summary["median_rms_ns"]["station_moon"],
        summary["median_rms_ns"]["station_reflector"],
    ]
    plt.figure(figsize=(9, 4.5))
    plt.bar(labels, vals_ns)
    plt.yscale("log")
    plt.ylabel("残差RMS [ns]（定数オフセット整列後, 中央値）")
    plt.title(f"{LLR_SHORT_NAME}：モデル改善の効果（全グループ中央値）")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    p1 = out_dir / "llr_rms_improvement_overall.png"
    plt.savefig(p1, dpi=200)
    plt.close()

    # Plot 2: RMS by station×target (station_reflector)
    piv = metrics_df.pivot(index="station", columns="target", values="rms_sr_ns")
    plt.figure(figsize=(12, 4.8))
    im = plt.imshow(piv.to_numpy(dtype=float), aspect="auto")
    plt.colorbar(im, label="残差RMS [ns]（観測局→反射器）")
    plt.xticks(range(len(piv.columns)), list(piv.columns), rotation=30, ha="right")
    plt.yticks(range(len(piv.index)), list(piv.index))
    plt.title(f"{LLR_SHORT_NAME}：残差RMS（観測局→反射器, 定数整列）  station × reflector")
    plt.tight_layout()
    p2 = out_dir / "llr_rms_by_station_target.png"
    plt.savefig(p2, dpi=200)
    plt.close()

    # Plot 3: Ablations (median RMS, station->reflector only)
    labels2 = ["SPICE+Shapiro", "SPICE+Shapiro+対流圏", "SPICE+Shapiro+対流圏+潮汐(固体+海洋+月)", "SPICEのみ", "IAU+Shapiro"]
    vals2 = [
        summary["median_rms_ns"]["station_reflector"],
        summary["median_rms_ns"]["station_reflector_tropo"],
        summary["median_rms_ns"]["station_reflector_tropo_tide"],
        summary["median_rms_ns"]["station_reflector_no_shapiro"],
        summary["median_rms_ns"]["station_reflector_iau"],
    ]
    plt.figure(figsize=(9, 4.5))
    plt.bar(labels2, vals2)
    plt.yscale("log")
    plt.ylabel("残差RMS [ns]（観測局→反射器, 定数整列後, 中央値）")
    plt.title(f"{LLR_SHORT_NAME}：反射器モデルの切り分け（Shapiro / 対流圏 / 潮汐 / 月回転）")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    p3 = out_dir / "llr_rms_ablations_overall.png"
    plt.savefig(p3, dpi=200)
    plt.close()

    # Plot 4: Shapiro breakdown (with troposphere to isolate Shapiro effect)
    labels3 = ["対流圏のみ（Shapiro OFF）", "対流圏+太陽Shapiro", "対流圏+太陽+地球Shapiro"]
    vals3 = [
        float(summary["median_rms_ns"].get("station_reflector_tropo_no_shapiro", float("nan"))),
        float(summary["median_rms_ns"].get("station_reflector_tropo", float("nan"))),
        float(summary["median_rms_ns"].get("station_reflector_tropo_earth_shapiro", float("nan"))),
    ]
    plt.figure(figsize=(10, 4.5))
    plt.bar(labels3, vals3)
    try:
        vmax = float(np.nanmax(np.array(vals3, dtype=float)))
        # 条件分岐: `np.isfinite(vmax) and vmax > 0` を満たす経路を評価する。
        if np.isfinite(vmax) and vmax > 0:
            plt.ylim(0.0, vmax * 1.08)
            for i, v in enumerate(vals3):
                # 条件分岐: `np.isfinite(v)` を満たす経路を評価する。
                if np.isfinite(v):
                    plt.text(i, v + vmax * 0.03, f"{v:.3f} ns", ha="center", va="bottom", fontsize=9)
    except Exception:
        pass

    plt.ylabel("残差RMS [ns]（観測局→反射器, 定数整列後, 中央値）")
    plt.title(f"{LLR_SHORT_NAME}：重力遅延の寄与（ShapiroのON/OFF, 対流圏あり）")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    p4 = out_dir / "llr_shapiro_ablations_overall.png"
    plt.savefig(p4, dpi=200)
    plt.close()

    # Plot 5: Tide breakdown (with troposphere to isolate tide effect)
    labels5 = ["対流圏のみ", "対流圏+観測局潮汐(固体)", "対流圏+月体潮汐(反射器)", "対流圏+潮汐(固体+月)", "対流圏+潮汐(固体+海洋+月)"]
    vals5 = [
        float(summary["median_rms_ns"].get("station_reflector_tropo", float("nan"))),
        float(summary["median_rms_ns"].get("station_reflector_tropo_station_tide", float("nan"))),
        float(summary["median_rms_ns"].get("station_reflector_tropo_moon_tide", float("nan"))),
        float(summary["median_rms_ns"].get("station_reflector_tropo_tide_no_ocean", float("nan"))),
        float(summary["median_rms_ns"].get("station_reflector_tropo_tide", float("nan"))),
    ]
    plt.figure(figsize=(10, 4.5))
    plt.bar(labels5, vals5)
    plt.xticks(rotation=12, ha="right", fontsize=9)
    try:
        vmax = float(np.nanmax(np.array(vals5, dtype=float)))
        # 条件分岐: `np.isfinite(vmax) and vmax > 0` を満たす経路を評価する。
        if np.isfinite(vmax) and vmax > 0:
            plt.ylim(0.0, vmax * 1.08)
            for i, v in enumerate(vals5):
                # 条件分岐: `np.isfinite(v)` を満たす経路を評価する。
                if np.isfinite(v):
                    plt.text(i, v + vmax * 0.03, f"{v:.3f} ns", ha="center", va="bottom", fontsize=9)
    except Exception:
        pass

    plt.ylabel("残差RMS [ns]（観測局→反射器, 定数整列後, 中央値）")
    plt.title(f"{LLR_SHORT_NAME}：潮汐の寄与（固体潮汐/海洋荷重/月体潮汐, 対流圏あり）")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    p5 = out_dir / "llr_tide_ablations_overall.png"
    plt.savefig(p5, dpi=200)
    plt.close()

    print(f"[ok] metrics: {metrics_csv}")
    print(f"[ok] summary: {summary_path}")
    print(f"[ok] plots  : {p1} / {p2} / {p3} / {p4} / {p5}")

    try:
        def _safe_rel(p: Path) -> str:
            try:
                return str(p.relative_to(root)).replace("\\", "/")
            except Exception:
                return str(p).replace("\\", "/")

        worklog.append_event(
            {
                "event_type": "llr_batch_eval",
                "mode": "offline" if offline else "online",
                "argv": sys.argv,
                "inputs": {
                    "manifest": _safe_rel(manifest_path),
                    "cache_dir": _safe_rel(cache_dir),
                },
                "params": {
                    "beta": float(args.beta),
                    "time_tag_mode": str(mode),
                    "station_coords": str(args.station_coords),
                    "pos_eop_date": str(args.pos_eop_date),
                    "pos_eop_max_days": int(args.pos_eop_max_days),
                    "ocean_loading": str(args.ocean_loading),
                    "min_points": int(args.min_points),
                    "outlier_clip_sigma": float(args.outlier_clip_sigma),
                    "outlier_clip_ns": float(args.outlier_clip_ns),
                    "chunk": int(args.chunk),
                },
                "metrics": {
                    "median_rms_ns_station_reflector": float(summary.get("median_rms_ns", {}).get("station_reflector", float("nan"))),
                    "median_rms_ns_station_reflector_tropo": float(summary.get("median_rms_ns", {}).get("station_reflector_tropo", float("nan"))),
                    "median_rms_ns_station_reflector_tropo_tide": float(summary.get("median_rms_ns", {}).get("station_reflector_tropo_tide", float("nan"))),
                    "n_groups": int(summary.get("n_groups", 0) or 0),
                },
                "outputs": {
                    "out_dir": out_dir,
                    "metrics_csv": metrics_csv,
                    "summary_json": summary_path,
                    "plot_rms_improvement_overall": p1,
                    "plot_rms_by_station_target": p2,
                    "plot_rms_ablations_overall": p3,
                    "plot_shapiro_ablations_overall": p4,
                    "plot_tide_ablations_overall": p5,
                    "outliers_csv": out_dir / "llr_outliers.csv",
                    "outliers_summary_json": out_dir / "llr_outliers_summary.json",
                    "outliers_overview_plot": out_dir / "llr_outliers_overview.png",
                },
            }
        )
    except Exception:
        pass

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
