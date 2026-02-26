#!/usr/bin/env python3
"""
llr_precision_reaudit.py

LLR の「中央値RMSが現代精度と桁が合わない」懸念に対して、
入力分布・range_type・時代別RMSを同一I/Fで再検証する監査スクリプト。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.llr import llr_pmodel_overlay_horizons_noargs as llr  # noqa: E402


def _repo_root() -> Path:
    return _ROOT


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _to_float(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return float("nan")

    return v if math.isfinite(v) else float("nan")


def _rms(arr: Iterable[float]) -> float:
    a = np.asarray(list(arr), dtype=float)
    a = a[np.isfinite(a)]
    # 条件分岐: `len(a) == 0` を満たす経路を評価する。
    if len(a) == 0:
        return float("nan")

    return float(np.sqrt(np.mean(a * a)))


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root)).replace("\\", "/")
    except Exception:
        return str(path.resolve()).replace("\\", "/")


def _read_record11_meta(path: Path, line_numbers: Sequence[int]) -> Dict[int, Dict[str, float]]:
    need = set(int(v) for v in line_numbers if v is not None and np.isfinite(float(v)))
    # 条件分岐: `not need` を満たす経路を評価する。
    if not need:
        return {}

    out: Dict[int, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for lineno, raw in enumerate(f, start=1):
            # 条件分岐: `lineno not in need` を満たす経路を評価する。
            if lineno not in need:
                continue

            toks = raw.strip().split()
            # 条件分岐: `not toks or toks[0] != "11"` を満たす経路を評価する。
            if not toks or toks[0] != "11":
                continue

            out[int(lineno)] = {
                "np_window_s": _to_float(toks[5]) if len(toks) > 5 else float("nan"),
                "np_n_raw_ranges": _to_float(toks[6]) if len(toks) > 6 else float("nan"),
                "np_bin_rms_ps": _to_float(toks[7]) if len(toks) > 7 else float("nan"),
            }

    return out


def _augment_points_with_np_meta(df: pd.DataFrame, root: Path) -> pd.DataFrame:
    out = df.copy()
    out["source_file"] = out["source_file"].astype(str)
    out["lineno"] = pd.to_numeric(out["lineno"], errors="coerce")

    np_window = np.full((len(out),), np.nan, dtype=float)
    np_n_raw = np.full((len(out),), np.nan, dtype=float)
    np_bin_rms = np.full((len(out),), np.nan, dtype=float)

    grouped = out[["source_file", "lineno"]].dropna().groupby("source_file", dropna=False)
    cache: Dict[str, Dict[int, Dict[str, float]]] = {}
    for src, g in grouped:
        p = (root / Path(str(src))).resolve()
        # 条件分岐: `not p.exists()` を満たす経路を評価する。
        if not p.exists():
            continue

        ln = pd.to_numeric(g["lineno"], errors="coerce").dropna().astype(int).tolist()
        cache[str(src)] = _read_record11_meta(p, ln)

    for pos, row in enumerate(out.itertuples(index=False)):
        src = str(getattr(row, "source_file", ""))
        ln = getattr(row, "lineno", float("nan"))
        # 条件分岐: `not (src and np.isfinite(_to_float(ln)))` を満たす経路を評価する。
        if not (src and np.isfinite(_to_float(ln))):
            continue

        rec = cache.get(src, {}).get(int(float(ln)))
        # 条件分岐: `rec is None` を満たす経路を評価する。
        if rec is None:
            continue

        np_window[pos] = _to_float(rec.get("np_window_s"))
        np_n_raw[pos] = _to_float(rec.get("np_n_raw_ranges"))
        np_bin_rms[pos] = _to_float(rec.get("np_bin_rms_ps"))

    out["np_window_s"] = np_window
    out["np_n_raw_ranges"] = np_n_raw
    out["np_bin_rms_ps"] = np_bin_rms
    out["np_bin_rms_ns"] = np_bin_rms / 1000.0
    return out


def _weighted_rms_ns(values_ns: np.ndarray, sigma_ps: np.ndarray) -> float:
    ok = np.isfinite(values_ns) & np.isfinite(sigma_ps) & (sigma_ps > 0)
    # 条件分岐: `not np.any(ok)` を満たす経路を評価する。
    if not np.any(ok):
        return float("nan")

    y = values_ns[ok]
    w = 1.0 / np.maximum((sigma_ps[ok] / 1000.0) ** 2, 1e-24)
    return float(np.sqrt(np.sum(w * y * y) / np.sum(w)))


def _solve_model_floor_ns(values_ns: np.ndarray, sigma_ps: np.ndarray) -> float:
    ok = np.isfinite(values_ns) & np.isfinite(sigma_ps) & (sigma_ps > 0)
    # 条件分岐: `not np.any(ok)` を満たす経路を評価する。
    if not np.any(ok):
        return float("nan")

    y_ps = values_ns[ok] * 1000.0
    s2 = sigma_ps[ok] ** 2
    r2 = y_ps ** 2
    lo = 0.0
    hi = 1e8
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        chi = float(np.mean(r2 / np.maximum(s2 + mid * mid, 1e-24)))
        # 条件分岐: `chi > 1.0` を満たす経路を評価する。
        if chi > 1.0:
            lo = mid
        else:
            hi = mid

    return float(hi / 1000.0)


def _fit_bias_model_ns(
    df: pd.DataFrame,
    *,
    use_station: bool,
    use_target: bool,
) -> Dict[str, Any]:
    # 条件分岐: `df.empty` を満たす経路を評価する。
    if df.empty:
        return {"ok": False, "reason": "empty"}

    work = df.copy()
    work["residual_ns"] = pd.to_numeric(work.get("residual_sr_tropo_tide_ns"), errors="coerce")
    work["sigma_ps"] = pd.to_numeric(work.get("np_bin_rms_ps"), errors="coerce")
    work["station"] = work.get("station", pd.Series(dtype=str)).astype(str).str.strip().str.upper()
    work["target"] = work.get("target", pd.Series(dtype=str)).astype(str).str.strip().str.lower()
    work = work[np.isfinite(work["residual_ns"]) & np.isfinite(work["sigma_ps"]) & (work["sigma_ps"] > 0)].copy()
    # 条件分岐: `work.empty` を満たす経路を評価する。
    if work.empty:
        return {"ok": False, "reason": "no_valid_rows"}

    cols: List[np.ndarray] = [np.ones((len(work),), dtype=float)]
    col_names: List[str] = ["intercept"]

    # 条件分岐: `use_station` を満たす経路を評価する。
    if use_station:
        st_levels = sorted(work["station"].dropna().unique().tolist())
        # 条件分岐: `len(st_levels) >= 2` を満たす経路を評価する。
        if len(st_levels) >= 2:
            ref_st = str(st_levels[0])
            for st in st_levels[1:]:
                cols.append((work["station"] == st).to_numpy(dtype=float))
                col_names.append(f"station:{st}")

    # 条件分岐: `use_target` を満たす経路を評価する。

    if use_target:
        tg_levels = sorted(work["target"].dropna().unique().tolist())
        # 条件分岐: `len(tg_levels) >= 2` を満たす経路を評価する。
        if len(tg_levels) >= 2:
            ref_tg = str(tg_levels[0])
            for tg in tg_levels[1:]:
                cols.append((work["target"] == tg).to_numpy(dtype=float))
                col_names.append(f"target:{tg}")

    X = np.column_stack(cols)
    y = work["residual_ns"].to_numpy(dtype=float)
    w = 1.0 / np.maximum((work["sigma_ps"].to_numpy(dtype=float) / 1000.0) ** 2, 1e-24)
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    pred = X @ coef
    corrected = y - pred

    return {
        "ok": True,
        "n_rows": int(len(work)),
        "weighted_rms_raw_ns": float(np.sqrt(np.sum(w * y * y) / np.sum(w))),
        "weighted_rms_corrected_ns": float(np.sqrt(np.sum(w * corrected * corrected) / np.sum(w))),
        "weighted_gain_ns": float(
            np.sqrt(np.sum(w * y * y) / np.sum(w)) - np.sqrt(np.sum(w * corrected * corrected) / np.sum(w))
        ),
        "n_parameters": int(len(col_names)),
        "model_columns": col_names,
    }


def _safe_corr_np(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    # 条件分岐: `int(np.sum(ok)) < 3` を満たす経路を評価する。
    if int(np.sum(ok)) < 3:
        return float("nan")

    aa = a[ok]
    bb = b[ok]
    # 条件分岐: `float(np.std(aa)) <= 0.0 or float(np.std(bb)) <= 0.0` を満たす経路を評価する。
    if float(np.std(aa)) <= 0.0 or float(np.std(bb)) <= 0.0:
        return float("nan")

    return float(np.corrcoef(aa, bb)[0, 1])


def _fit_linear_with_intercept(y: np.ndarray, x: np.ndarray) -> Tuple[float, float, float, float]:
    ok = np.isfinite(y) & np.isfinite(x)
    # 条件分岐: `int(np.sum(ok)) < 3` を満たす経路を評価する。
    if int(np.sum(ok)) < 3:
        return float("nan"), float("nan"), float("nan"), float("nan")

    yy = y[ok]
    xx = x[ok]
    A = np.column_stack([np.ones_like(xx), xx])
    coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
    intercept = float(coef[0])
    slope = float(coef[1])
    pred = A @ coef
    ss_tot = float(np.sum((yy - float(np.mean(yy))) ** 2))
    ss_res = float(np.sum((yy - pred) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan")
    corr = _safe_corr_np(yy, xx)
    return intercept, slope, corr, r2


def _prepare_operational_correction_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["residual_ns"] = pd.to_numeric(work.get("residual_sr_tropo_tide_ns"), errors="coerce")
    work["dt_tropo_ns"] = pd.to_numeric(work.get("dt_tropo_ns"), errors="coerce")
    work["station"] = work.get("station", pd.Series(dtype=str)).astype(str).str.strip().str.upper()
    work["target"] = work.get("target", pd.Series(dtype=str)).astype(str).str.strip().str.lower()
    # 条件分岐: `"epoch_utc" in work.columns` を満たす経路を評価する。
    if "epoch_utc" in work.columns:
        work["epoch_utc"] = pd.to_datetime(work["epoch_utc"], utc=True, errors="coerce")
    else:
        work["epoch_utc"] = pd.NaT

    # 条件分岐: `"year_month" in work.columns` を満たす経路を評価する。

    if "year_month" in work.columns:
        ym = work.get("year_month", pd.Series(dtype=str)).astype(str).str.strip()
        ym_bad = ym.eq("") | ym.eq("nan") | ym.eq("NaT")
        ym_from_epoch = work["epoch_utc"].dt.strftime("%Y-%m")
        work["year_month"] = np.where(ym_bad, ym_from_epoch, ym)
    else:
        work["year_month"] = work["epoch_utc"].dt.strftime("%Y-%m")

    work["year_month"] = work["year_month"].astype(str).str.strip()
    work["year_month"] = work["year_month"].replace({"NaT": "", "nan": "", "None": ""})
    work = work[np.isfinite(work["residual_ns"])].copy().reset_index(drop=True)
    return work


def _apply_operational_systematic_corrections(
    df: pd.DataFrame,
    *,
    apol_min_points: int = 30,
) -> Dict[str, Any]:
    work = _prepare_operational_correction_frame(df)
    n_rows = int(len(work))
    # 条件分岐: `work.empty` を満たす経路を評価する。
    if work.empty:
        return {
            "n_rows": 0,
            "apol_target_models": [],
            "grsm_apollo_month_bins": 0,
            "n_apol_corrected_points": 0,
            "n_grsm_apollo_month_points": 0,
            "residual_raw_ns": np.array([], dtype=float),
            "residual_apol_tropo_corrected_ns": np.array([], dtype=float),
            "residual_grsm_apollo_month_corrected_ns": np.array([], dtype=float),
            "residual_combined_corrected_ns": np.array([], dtype=float),
        }

    residual_raw = work["residual_ns"].to_numpy(dtype=float)
    residual_apol = residual_raw.copy()
    apol_models: List[Dict[str, Any]] = []
    n_apol_corrected_points = 0

    apol = work[work["station"] == "APOL"].copy()
    # 条件分岐: `not apol.empty` を満たす経路を評価する。
    if not apol.empty:
        for tg, g in apol.groupby("target", dropna=False):
            x_all = g["dt_tropo_ns"].to_numpy(dtype=float)
            y_all = g["residual_ns"].to_numpy(dtype=float)
            ok = np.isfinite(x_all) & np.isfinite(y_all)
            n_ok = int(np.sum(ok))
            # 条件分岐: `n_ok < int(apol_min_points)` を満たす経路を評価する。
            if n_ok < int(apol_min_points):
                continue

            intercept, slope, corr, r2 = _fit_linear_with_intercept(y_all[ok], x_all[ok])
            idx = g.index.to_numpy()[ok]
            pred = intercept + slope * x_all[ok]
            residual_apol[idx] = residual_apol[idx] - pred
            n_apol_corrected_points += int(len(idx))
            apol_models.append(
                {
                    "target": str(tg),
                    "n_points": int(n_ok),
                    "intercept_ns": intercept,
                    "slope_ns_per_ns": slope,
                    "corr": corr,
                    "r2": r2,
                }
            )

    apol_models.sort(key=lambda x: abs(_to_float(x.get("slope_ns_per_ns"))), reverse=True)

    gmask = (
        (work["station"] == "GRSM")
        & work["target"].isin(["apollo11", "apollo14", "apollo15"])
        & work["year_month"].astype(str).str.len().gt(0)
    )

    residual_grsm_month = residual_raw.copy()
    residual_combined = residual_apol.copy()
    n_grsm_points = 0
    n_grsm_bins = 0
    # 条件分岐: `bool(np.any(gmask.to_numpy(dtype=bool)))` を満たす経路を評価する。
    if bool(np.any(gmask.to_numpy(dtype=bool))):
        sub = work.loc[gmask, ["station", "target", "year_month"]].copy()
        sub["residual_raw"] = residual_raw[gmask.to_numpy(dtype=bool)]
        mu_raw = sub.groupby(["station", "target", "year_month"], dropna=False)["residual_raw"].transform("mean")
        idx = sub.index.to_numpy(dtype=int)
        residual_grsm_month[idx] = sub["residual_raw"].to_numpy(dtype=float) - mu_raw.to_numpy(dtype=float)

        sub2 = work.loc[gmask, ["station", "target", "year_month"]].copy()
        sub2["residual_combined"] = residual_combined[gmask.to_numpy(dtype=bool)]
        mu_comb = sub2.groupby(["station", "target", "year_month"], dropna=False)["residual_combined"].transform("mean")
        idx2 = sub2.index.to_numpy(dtype=int)
        residual_combined[idx2] = sub2["residual_combined"].to_numpy(dtype=float) - mu_comb.to_numpy(dtype=float)

        n_grsm_points = int(len(sub))
        n_grsm_bins = int(
            sub[["station", "target", "year_month"]]
            .drop_duplicates()
            .shape[0]
        )

    return {
        "n_rows": n_rows,
        "apol_target_models": apol_models,
        "grsm_apollo_month_bins": int(n_grsm_bins),
        "n_apol_corrected_points": int(n_apol_corrected_points),
        "n_grsm_apollo_month_points": int(n_grsm_points),
        "residual_raw_ns": residual_raw,
        "residual_apol_tropo_corrected_ns": residual_apol,
        "residual_grsm_apollo_month_corrected_ns": residual_grsm_month,
        "residual_combined_corrected_ns": residual_combined,
    }


def _collect_manifest_diagnostics(root: Path, manifest_path: Path) -> Dict[str, Any]:
    manifest = _read_json(manifest_path)
    file_recs = manifest.get("files") or []
    # 条件分岐: `not isinstance(file_recs, list)` を満たす経路を評価する。
    if not isinstance(file_recs, list):
        raise RuntimeError(f"manifest.files is not list: {manifest_path}")

    range_type_counts: Counter[Any] = Counter()
    station_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    year_counts: Counter[int] = Counter()
    parsed_files = 0
    parsed_rows = 0
    missing_files = 0

    for rec in file_recs:
        rel = rec.get("cached_path")
        # 条件分岐: `not rel` を満たす経路を評価する。
        if not rel:
            continue

        p = (root / str(rel)).resolve()
        # 条件分岐: `not p.exists()` を満たす経路を評価する。
        if not p.exists():
            missing_files += 1
            continue

        df = llr.parse_crd_npt11(p, assume_two_way_if_missing=True)
        # 条件分岐: `df.empty` を満たす経路を評価する。
        if df.empty:
            continue

        parsed_files += 1
        parsed_rows += int(len(df))

        for v in df.get("range_type", pd.Series(dtype=float)).dropna().tolist():
            try:
                range_type_counts[int(v)] += 1
            except Exception:
                range_type_counts[str(v)] += 1

        st = (
            df.get("station", pd.Series(dtype=str))
            .fillna("NA")
            .astype(str)
            .str.strip()
            .str.upper()
            .tolist()
        )
        station_counts.update(st)

        tg = (
            df.get("target", pd.Series(dtype=str))
            .fillna("na")
            .astype(str)
            .str.strip()
            .str.lower()
            .tolist()
        )
        target_counts.update(tg)

        # 条件分岐: `"epoch_utc" in df.columns` を満たす経路を評価する。
        if "epoch_utc" in df.columns:
            e = pd.to_datetime(df["epoch_utc"], utc=True, errors="coerce")
            ys = e.dt.year.dropna().astype(int).tolist()
            year_counts.update(ys)

    return {
        "n_files_manifest": int(len(file_recs)),
        "n_files_parsed": int(parsed_files),
        "n_files_missing": int(missing_files),
        "n_rows_parsed": int(parsed_rows),
        "range_type_counts": {str(k): int(v) for k, v in sorted(range_type_counts.items(), key=lambda kv: str(kv[0]))},
        "station_counts": dict(station_counts),
        "target_counts": dict(target_counts),
        "year_counts": {str(k): int(v) for k, v in sorted(year_counts.items())},
    }


def _load_batch_metrics(metrics_csv: Path) -> Dict[str, Any]:
    # 条件分岐: `not metrics_csv.exists()` を満たす経路を評価する。
    if not metrics_csv.exists():
        return {"exists": False}

    df = pd.read_csv(metrics_csv)
    # 条件分岐: `df.empty` を満たす経路を評価する。
    if df.empty:
        return {"exists": True, "n_groups": 0}

    rms_col = pd.to_numeric(df.get("rms_sr_tropo_tide_ns"), errors="coerce")
    n_col = pd.to_numeric(df.get("n"), errors="coerce")

    median_group = float(np.nanmedian(rms_col.to_numpy(dtype=float))) if len(df) else float("nan")
    weighted_all = float("nan")
    ok = np.isfinite(rms_col.to_numpy(dtype=float)) & np.isfinite(n_col.to_numpy(dtype=float)) & (n_col.to_numpy(dtype=float) > 0)
    # 条件分岐: `np.any(ok)` を満たす経路を評価する。
    if np.any(ok):
        rv = rms_col.to_numpy(dtype=float)[ok]
        nv = n_col.to_numpy(dtype=float)[ok]
        weighted_all = float(np.sqrt(np.sum(nv * rv * rv) / np.sum(nv)))

    ng = df[df["target"].astype(str).str.lower() == "nglr1"].copy()
    ng_rows: List[Dict[str, Any]] = []
    for r in ng.itertuples(index=False):
        ng_rows.append(
            {
                "station": str(getattr(r, "station", "")),
                "target": str(getattr(r, "target", "")),
                "n": int(getattr(r, "n", 0) or 0),
                "rms_sr_tropo_tide_ns": _to_float(getattr(r, "rms_sr_tropo_tide_ns", float("nan"))),
            }
        )

    return {
        "exists": True,
        "n_groups": int(len(df)),
        "median_group_rms_sr_tropo_tide_ns": median_group,
        "point_weighted_rms_sr_tropo_tide_ns": weighted_all,
        "nglr1_rows": ng_rows,
    }


def _group_weighted_rms(df: pd.DataFrame, rms_col: str) -> float:
    # 条件分岐: `df.empty or rms_col not in df.columns` を満たす経路を評価する。
    if df.empty or rms_col not in df.columns:
        return float("nan")

    rms = pd.to_numeric(df[rms_col], errors="coerce").to_numpy(dtype=float)
    n = pd.to_numeric(df.get("n"), errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(rms) & np.isfinite(n) & (n > 0)
    # 条件分岐: `not np.any(ok)` を満たす経路を評価する。
    if not np.any(ok):
        return float("nan")

    return float(np.sqrt(np.sum(n[ok] * rms[ok] * rms[ok]) / np.sum(n[ok])))


def _build_root_cause_decomposition(
    summary: Dict[str, Any],
    metrics_csv: Path,
    manifest_diag: Dict[str, Any],
    points_diag: Dict[str, Any],
    station_meta_diag: Dict[str, Any],
) -> Dict[str, Any]:
    med = summary.get("median_rms_ns") if isinstance(summary.get("median_rms_ns"), dict) else {}
    pwr = summary.get("point_weighted_rms_ns") if isinstance(summary.get("point_weighted_rms_ns"), dict) else {}

    df = pd.DataFrame()
    # 条件分岐: `metrics_csv.exists()` を満たす経路を評価する。
    if metrics_csv.exists():
        try:
            df = pd.read_csv(metrics_csv)
        except Exception:
            df = pd.DataFrame()

    # 条件分岐: `not df.empty` を満たす経路を評価する。

    if not df.empty:
        df = df.copy()
        df["n"] = pd.to_numeric(df.get("n"), errors="coerce")
        for col in [
            "rms_sr_ns",
            "rms_sr_tropo_ns",
            "rms_sr_tropo_tide_ns",
            "rms_sr_tropo_no_shapiro_ns",
            "rms_sr_tropo_earth_shapiro_ns",
            "rms_sr_tropo_tide_no_ocean_ns",
        ]:
            # 条件分岐: `col in df.columns` を満たす経路を評価する。
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    def _med(key: str) -> float:
        return _to_float(med.get(key))

    def _weighted(key: str, col: str) -> float:
        v = _to_float(pwr.get(key))
        # 条件分岐: `np.isfinite(v)` を満たす経路を評価する。
        if np.isfinite(v):
            return float(v)

        return _group_weighted_rms(df, col)

    ablation_median = {
        "station_reflector": _med("station_reflector"),
        "station_reflector_tropo": _med("station_reflector_tropo"),
        "station_reflector_tropo_tide": _med("station_reflector_tropo_tide"),
        "station_reflector_tropo_no_shapiro": _med("station_reflector_tropo_no_shapiro"),
        "station_reflector_tropo_earth_shapiro": _med("station_reflector_tropo_earth_shapiro"),
        "station_reflector_tropo_tide_no_ocean": _med("station_reflector_tropo_tide_no_ocean"),
    }
    ablation_weighted = {
        "station_reflector": _weighted("station_reflector", "rms_sr_ns"),
        "station_reflector_tropo": _weighted("station_reflector_tropo", "rms_sr_tropo_ns"),
        "station_reflector_tropo_tide": _weighted("station_reflector_tropo_tide", "rms_sr_tropo_tide_ns"),
        "station_reflector_tropo_no_shapiro": _group_weighted_rms(df, "rms_sr_tropo_no_shapiro_ns"),
        "station_reflector_tropo_earth_shapiro": _group_weighted_rms(df, "rms_sr_tropo_earth_shapiro_ns"),
        "station_reflector_tropo_tide_no_ocean": _group_weighted_rms(df, "rms_sr_tropo_tide_no_ocean_ns"),
    }

    gain_median = {
        "tropo_from_sr": _to_float(ablation_median["station_reflector"]) - _to_float(ablation_median["station_reflector_tropo"]),
        "tide_from_tropo": _to_float(ablation_median["station_reflector_tropo"]) - _to_float(ablation_median["station_reflector_tropo_tide"]),
        "shapiro_from_tropo_no_shapiro": _to_float(ablation_median["station_reflector_tropo_no_shapiro"])
        - _to_float(ablation_median["station_reflector_tropo"]),
        "earth_shapiro_from_tropo": _to_float(ablation_median["station_reflector_tropo"])
        - _to_float(ablation_median["station_reflector_tropo_earth_shapiro"]),
        "ocean_loading_from_tide_no_ocean": _to_float(ablation_median["station_reflector_tropo_tide_no_ocean"])
        - _to_float(ablation_median["station_reflector_tropo_tide"]),
    }
    gain_weighted = {
        "tropo_from_sr": _to_float(ablation_weighted["station_reflector"]) - _to_float(ablation_weighted["station_reflector_tropo"]),
        "tide_from_tropo": _to_float(ablation_weighted["station_reflector_tropo"]) - _to_float(ablation_weighted["station_reflector_tropo_tide"]),
        "shapiro_from_tropo_no_shapiro": _to_float(ablation_weighted["station_reflector_tropo_no_shapiro"])
        - _to_float(ablation_weighted["station_reflector_tropo"]),
        "earth_shapiro_from_tropo": _to_float(ablation_weighted["station_reflector_tropo"])
        - _to_float(ablation_weighted["station_reflector_tropo_earth_shapiro"]),
        "ocean_loading_from_tide_no_ocean": _to_float(ablation_weighted["station_reflector_tropo_tide_no_ocean"])
        - _to_float(ablation_weighted["station_reflector_tropo_tide"]),
    }

    total_gain_median = _to_float(ablation_median["station_reflector"]) - _to_float(ablation_median["station_reflector_tropo_tide"])
    total_gain_weighted = _to_float(ablation_weighted["station_reflector"]) - _to_float(ablation_weighted["station_reflector_tropo_tide"])

    def _safe_share(v: float, total: float) -> float:
        # 条件分岐: `not (np.isfinite(v) and np.isfinite(total) and abs(total) > 0.0)` を満たす経路を評価する。
        if not (np.isfinite(v) and np.isfinite(total) and abs(total) > 0.0):
            return float("nan")

        return float(v / total)

    gain_share_median = {
        "tropo_from_sr": _safe_share(_to_float(gain_median["tropo_from_sr"]), total_gain_median),
        "tide_from_tropo": _safe_share(_to_float(gain_median["tide_from_tropo"]), total_gain_median),
    }
    gain_share_weighted = {
        "tropo_from_sr": _safe_share(_to_float(gain_weighted["tropo_from_sr"]), total_gain_weighted),
        "tide_from_tropo": _safe_share(_to_float(gain_weighted["tide_from_tropo"]), total_gain_weighted),
    }

    station_rows: List[Dict[str, Any]] = []
    target_rows: List[Dict[str, Any]] = []
    top_group_rows: List[Dict[str, Any]] = []
    bottleneck_scenarios: Dict[str, Any] = {}
    bottleneck_ranking: List[Dict[str, Any]] = []
    threshold_ns = 4.0
    excess_over_4ns: Dict[str, Any] = {
        "threshold_ns": threshold_ns,
        "current_group_weighted_rms_ns": float("nan"),
        "current_group_weighted_excess_ns": float("nan"),
        "ablation_median_excess_ns": {},
        "ablation_weighted_excess_ns": {},
        "station_excess_share": [],
        "target_excess_share": [],
        "top_group_excess_contributors": [],
    }

    def _excess_ns(v: float) -> float:
        vv = _to_float(v)
        # 条件分岐: `not np.isfinite(vv)` を満たす経路を評価する。
        if not np.isfinite(vv):
            return float("nan")

        return float(max(vv - threshold_ns, 0.0))

    excess_over_4ns["ablation_median_excess_ns"] = {
        "station_reflector": _excess_ns(ablation_median.get("station_reflector")),
        "station_reflector_tropo": _excess_ns(ablation_median.get("station_reflector_tropo")),
        "station_reflector_tropo_tide": _excess_ns(ablation_median.get("station_reflector_tropo_tide")),
        "station_reflector_tropo_no_shapiro": _excess_ns(ablation_median.get("station_reflector_tropo_no_shapiro")),
    }
    excess_over_4ns["ablation_weighted_excess_ns"] = {
        "station_reflector": _excess_ns(ablation_weighted.get("station_reflector")),
        "station_reflector_tropo": _excess_ns(ablation_weighted.get("station_reflector_tropo")),
        "station_reflector_tropo_tide": _excess_ns(ablation_weighted.get("station_reflector_tropo_tide")),
        "station_reflector_tropo_no_shapiro": _excess_ns(ablation_weighted.get("station_reflector_tropo_no_shapiro")),
    }

    # 条件分岐: `not df.empty and "rms_sr_tropo_tide_ns" in df.columns` を満たす経路を評価する。
    if not df.empty and "rms_sr_tropo_tide_ns" in df.columns:
        use = df.copy()
        use["rms_sr_tropo_tide_ns"] = pd.to_numeric(use["rms_sr_tropo_tide_ns"], errors="coerce")
        use = use[np.isfinite(use["n"]) & (use["n"] > 0) & np.isfinite(use["rms_sr_tropo_tide_ns"])].copy()
        # 条件分岐: `not use.empty` を満たす経路を評価する。
        if not use.empty:
            use["sse_tide"] = use["n"] * use["rms_sr_tropo_tide_ns"] * use["rms_sr_tropo_tide_ns"]
            use["sse_excess_4ns"] = use["n"] * np.maximum(
                use["rms_sr_tropo_tide_ns"] * use["rms_sr_tropo_tide_ns"] - threshold_ns * threshold_ns,
                0.0,
            )
            total_sse = float(np.sum(use["sse_tide"].to_numpy(dtype=float)))
            total_sse_excess = float(np.sum(use["sse_excess_4ns"].to_numpy(dtype=float)))
            total_n = float(np.sum(use["n"].to_numpy(dtype=float)))
            current_group_weighted_rms = float(np.sqrt(total_sse / total_n)) if total_n > 0 else float("nan")
            excess_over_4ns["current_group_weighted_rms_ns"] = current_group_weighted_rms
            excess_over_4ns["current_group_weighted_excess_ns"] = _excess_ns(current_group_weighted_rms)

            for station, g in use.groupby("station"):
                n_points = float(np.sum(g["n"].to_numpy(dtype=float)))
                sse = float(np.sum(g["sse_tide"].to_numpy(dtype=float)))
                sse_excess = float(np.sum(g["sse_excess_4ns"].to_numpy(dtype=float)))
                wr = float(np.sqrt(sse / n_points)) if n_points > 0 else float("nan")
                station_rows.append(
                    {
                        "station": str(station),
                        "n_groups": int(len(g)),
                        "n_points": int(round(n_points)),
                        "weighted_rms_ns": wr,
                        "median_group_rms_ns": float(np.nanmedian(g["rms_sr_tropo_tide_ns"].to_numpy(dtype=float))),
                        "sse_share": float(sse / total_sse) if total_sse > 0 else float("nan"),
                        "sse_excess_share_over4ns": float(sse_excess / total_sse_excess) if total_sse_excess > 0 else float("nan"),
                    }
                )

            for target, g in use.groupby("target"):
                n_points = float(np.sum(g["n"].to_numpy(dtype=float)))
                sse = float(np.sum(g["sse_tide"].to_numpy(dtype=float)))
                sse_excess = float(np.sum(g["sse_excess_4ns"].to_numpy(dtype=float)))
                wr = float(np.sqrt(sse / n_points)) if n_points > 0 else float("nan")
                target_rows.append(
                    {
                        "target": str(target),
                        "n_groups": int(len(g)),
                        "n_points": int(round(n_points)),
                        "weighted_rms_ns": wr,
                        "median_group_rms_ns": float(np.nanmedian(g["rms_sr_tropo_tide_ns"].to_numpy(dtype=float))),
                        "sse_share": float(sse / total_sse) if total_sse > 0 else float("nan"),
                        "sse_excess_share_over4ns": float(sse_excess / total_sse_excess) if total_sse_excess > 0 else float("nan"),
                    }
                )

            use = use.sort_values("sse_tide", ascending=False)
            for r in use.head(8).itertuples(index=False):
                sse = _to_float(getattr(r, "sse_tide", float("nan")))
                top_group_rows.append(
                    {
                        "station": str(getattr(r, "station", "")),
                        "target": str(getattr(r, "target", "")),
                        "n_points": int(_to_float(getattr(r, "n", 0.0)) or 0),
                        "rms_ns": _to_float(getattr(r, "rms_sr_tropo_tide_ns", float("nan"))),
                        "sse_share": float(sse / total_sse) if (np.isfinite(sse) and total_sse > 0) else float("nan"),
                    }
                )

            use_excess = use.sort_values("sse_excess_4ns", ascending=False)
            top_group_excess_rows: List[Dict[str, Any]] = []
            for r in use_excess.head(8).itertuples(index=False):
                sse_ex = _to_float(getattr(r, "sse_excess_4ns", float("nan")))
                top_group_excess_rows.append(
                    {
                        "station": str(getattr(r, "station", "")),
                        "target": str(getattr(r, "target", "")),
                        "n_points": int(_to_float(getattr(r, "n", 0.0)) or 0),
                        "rms_ns": _to_float(getattr(r, "rms_sr_tropo_tide_ns", float("nan"))),
                        "excess_share_over4ns": float(sse_ex / total_sse_excess) if (np.isfinite(sse_ex) and total_sse_excess > 0) else float("nan"),
                    }
                )

            excess_over_4ns["top_group_excess_contributors"] = top_group_excess_rows

            use_no_nglr1 = use[use["target"].astype(str).str.lower() != "nglr1"].copy()
            # 条件分岐: `not use_no_nglr1.empty` を満たす経路を評価する。
            if not use_no_nglr1.empty:
                sse_no_nglr1 = float(np.sum(use_no_nglr1["sse_tide"].to_numpy(dtype=float)))
                n_no_nglr1 = float(np.sum(use_no_nglr1["n"].to_numpy(dtype=float)))
                rms_no_nglr1 = float(np.sqrt(sse_no_nglr1 / n_no_nglr1)) if n_no_nglr1 > 0 else float("nan")
            else:
                rms_no_nglr1 = float("nan")

            grsm = use[use["station"].astype(str).str.upper() == "GRSM"]
            grsm_med = float(np.nanmedian(grsm["rms_sr_tropo_tide_ns"].to_numpy(dtype=float))) if not grsm.empty else float("nan")
            # 条件分岐: `np.isfinite(grsm_med)` を満たす経路を評価する。
            if np.isfinite(grsm_med):
                use_apol_capped = use.copy()
                apol_mask = use_apol_capped["station"].astype(str).str.upper() == "APOL"
                use_apol_capped.loc[apol_mask, "rms_sr_tropo_tide_ns"] = np.minimum(
                    use_apol_capped.loc[apol_mask, "rms_sr_tropo_tide_ns"].to_numpy(dtype=float),
                    grsm_med,
                )
                sse_apol_cap = float(
                    np.sum(
                        (
                            use_apol_capped["n"].to_numpy(dtype=float)
                            * use_apol_capped["rms_sr_tropo_tide_ns"].to_numpy(dtype=float)
                            * use_apol_capped["rms_sr_tropo_tide_ns"].to_numpy(dtype=float)
                        )
                    )
                )
                rms_apol_cap = float(np.sqrt(sse_apol_cap / total_n)) if total_n > 0 else float("nan")
            else:
                rms_apol_cap = float("nan")

            bias_global_gain = _to_float(((points_diag.get("bias_model_global") or {}).get("weighted_gain_ns")))
            rms_bias_corrected_est = (
                float(current_group_weighted_rms - bias_global_gain)
                if (np.isfinite(current_group_weighted_rms) and np.isfinite(bias_global_gain))
                else float("nan")
            )

            bottleneck_scenarios = {
                "current_group_weighted_rms_ns": current_group_weighted_rms,
                "exclude_nglr1_group_weighted_rms_ns": rms_no_nglr1,
                "exclude_nglr1_gain_ns": (
                    float(current_group_weighted_rms - rms_no_nglr1)
                    if (np.isfinite(current_group_weighted_rms) and np.isfinite(rms_no_nglr1))
                    else float("nan")
                ),
                "apol_cap_to_grsm_median_rms_ns": rms_apol_cap,
                "apol_cap_to_grsm_median_gain_ns": (
                    float(current_group_weighted_rms - rms_apol_cap)
                    if (np.isfinite(current_group_weighted_rms) and np.isfinite(rms_apol_cap))
                    else float("nan")
                ),
                "grsm_median_reference_ns": grsm_med,
                "bias_corrected_global_est_rms_ns": rms_bias_corrected_est,
                "bias_corrected_global_est_gain_ns": bias_global_gain,
            }

            bottleneck_ranking = [
                {
                    "id": "apol_profile_gap_cap_to_grsm_median",
                    "estimated_gain_ns": _to_float(bottleneck_scenarios.get("apol_cap_to_grsm_median_gain_ns")),
                    "note": "APOL群をGRSM中央値レベルに抑えた場合の上限改善見積。",
                },
                {
                    "id": "nglr1_group_exclusion_proxy",
                    "estimated_gain_ns": _to_float(bottleneck_scenarios.get("exclude_nglr1_gain_ns")),
                    "note": "nglr1群の高残差寄与を除いた場合の改善見積。",
                },
                {
                    "id": "global_station_target_bias_correction",
                    "estimated_gain_ns": _to_float(bottleneck_scenarios.get("bias_corrected_global_est_gain_ns")),
                    "note": "station+targetバイアス補正での改善見積。",
                },
            ]
            bottleneck_ranking = sorted(
                bottleneck_ranking,
                key=lambda x: _to_float(x.get("estimated_gain_ns")),
                reverse=True,
            )

            st_ex = sorted(
                [
                    {
                        "station": str(r.get("station")),
                        "n_points": int(r.get("n_points", 0)),
                        "weighted_rms_ns": _to_float(r.get("weighted_rms_ns")),
                        "excess_share_over4ns": _to_float(r.get("sse_excess_share_over4ns")),
                    }
                    for r in station_rows
                ],
                key=lambda x: _to_float(x.get("excess_share_over4ns")),
                reverse=True,
            )
            tg_ex = sorted(
                [
                    {
                        "target": str(r.get("target")),
                        "n_points": int(r.get("n_points", 0)),
                        "weighted_rms_ns": _to_float(r.get("weighted_rms_ns")),
                        "excess_share_over4ns": _to_float(r.get("sse_excess_share_over4ns")),
                    }
                    for r in target_rows
                ],
                key=lambda x: _to_float(x.get("excess_share_over4ns")),
                reverse=True,
            )
            excess_over_4ns["station_excess_share"] = st_ex
            excess_over_4ns["target_excess_share"] = tg_ex

    station_rows.sort(key=lambda x: _to_float(x.get("sse_share")), reverse=True)
    target_rows.sort(key=lambda x: _to_float(x.get("sse_share")), reverse=True)

    likely_causes: List[Dict[str, Any]] = []
    tropo_share = _to_float(gain_share_median.get("tropo_from_sr"))
    # 条件分岐: `np.isfinite(tropo_share)` を満たす経路を評価する。
    if np.isfinite(tropo_share):
        likely_causes.append(
            {
                "id": "tropo_model_mismatch",
                "priority": 1 if tropo_share >= 0.7 else 2,
                "evidence": {
                    "gain_share_median": tropo_share,
                    "gain_ns_median": _to_float(gain_median.get("tropo_from_sr")),
                },
                "note": "残差改善の大半を対流圏補正が占めるため、主因候補は対流圏モデル差。",
            }
        )

    bias_global_gain = _to_float(((points_diag.get("bias_model_global") or {}).get("weighted_gain_ns")))
    bias_modern_gain = _to_float(((points_diag.get("bias_model_modern_apol_target") or {}).get("weighted_gain_ns")))
    # 条件分岐: `np.isfinite(bias_global_gain) or np.isfinite(bias_modern_gain)` を満たす経路を評価する。
    if np.isfinite(bias_global_gain) or np.isfinite(bias_modern_gain):
        likely_causes.append(
            {
                "id": "station_target_bias",
                "priority": 2 if (np.isfinite(bias_global_gain) and bias_global_gain >= 0.2) else 3,
                "evidence": {
                    "global_bias_weighted_gain_ns": bias_global_gain,
                    "modern_apol_target_bias_weighted_gain_ns": bias_modern_gain,
                },
                "note": "局/ターゲット依存バイアス補正で weighted RMS が改善しており、運用系統が残っている。",
            }
        )

    station_counts = manifest_diag.get("station_counts") or {}
    total_station = float(sum(float(v) for v in station_counts.values())) if station_counts else 0.0
    # 条件分岐: `total_station > 0` を満たす経路を評価する。
    if total_station > 0:
        dom_station, dom_count = max(station_counts.items(), key=lambda kv: float(kv[1]))
        dom_ratio = float(float(dom_count) / total_station)
        likely_causes.append(
            {
                "id": "station_imbalance",
                "priority": 2 if dom_ratio >= 0.8 else 3,
                "evidence": {"dominant_station": str(dom_station), "dominant_ratio": dom_ratio},
                "note": "局偏在が強く、残差評価が特定局の系統へ引き寄せられる。",
            }
        )

    missing_pos = station_meta_diag.get("missing_pos_eop") if isinstance(station_meta_diag.get("missing_pos_eop"), list) else []
    # 条件分岐: `missing_pos` を満たす経路を評価する。
    if missing_pos:
        likely_causes.append(
            {
                "id": "coord_if_incomplete",
                "priority": 2,
                "evidence": {"stations_without_pos_eop": [str(x) for x in missing_pos]},
                "note": "全局で pos+eop 統一になっておらず、局座標I/F差が残差に寄与しうる。",
            }
        )

    nglr1_row = next((r for r in target_rows if str(r.get("target", "")).lower() == "nglr1"), None)
    # 条件分岐: `nglr1_row is not None` を満たす経路を評価する。
    if nglr1_row is not None:
        likely_causes.append(
            {
                "id": "nglr1_sparse_high_residual",
                "priority": 3,
                "evidence": {
                    "nglr1_n_points": int(nglr1_row.get("n_points", 0)),
                    "nglr1_weighted_rms_ns": _to_float(nglr1_row.get("weighted_rms_ns")),
                    "nglr1_sse_share": _to_float(nglr1_row.get("sse_share")),
                },
                "note": "nglr1 は点数が少ない一方で残差が高く、外れ群として全体評価を押し上げる。",
            }
        )

    likely_causes.sort(key=lambda x: (int(x.get("priority", 9)), str(x.get("id", ""))))

    return {
        "ablation_median_ns": ablation_median,
        "ablation_weighted_ns": ablation_weighted,
        "incremental_gain_median_ns": gain_median,
        "incremental_gain_weighted_ns": gain_weighted,
        "total_gain_median_ns": total_gain_median,
        "total_gain_weighted_ns": total_gain_weighted,
        "gain_share_median": gain_share_median,
        "gain_share_weighted": gain_share_weighted,
        "station_breakdown": station_rows,
        "target_breakdown": target_rows,
        "top_group_sse_contributors": top_group_rows,
        "bottleneck_scenarios": bottleneck_scenarios,
        "bottleneck_ranking": bottleneck_ranking,
        "excess_over_4ns": excess_over_4ns,
        "likely_root_causes": likely_causes,
    }


def _write_root_cause_csv(path: Path, root_cause: Dict[str, Any]) -> None:
    rows: List[Dict[str, Any]] = []
    for section in ["ablation_median_ns", "ablation_weighted_ns", "incremental_gain_median_ns", "incremental_gain_weighted_ns"]:
        block = root_cause.get(section)
        # 条件分岐: `isinstance(block, dict)` を満たす経路を評価する。
        if isinstance(block, dict):
            for k, v in block.items():
                rows.append({"type": "metric", "section": section, "id": str(k), "value": _to_float(v), "note": ""})

    for section in ["gain_share_median", "gain_share_weighted"]:
        block = root_cause.get(section)
        # 条件分岐: `isinstance(block, dict)` を満たす経路を評価する。
        if isinstance(block, dict):
            for k, v in block.items():
                rows.append({"type": "share", "section": section, "id": str(k), "value": _to_float(v), "note": ""})

    block = root_cause.get("bottleneck_scenarios")
    # 条件分岐: `isinstance(block, dict)` を満たす経路を評価する。
    if isinstance(block, dict):
        for k, v in block.items():
            rows.append({"type": "scenario", "section": "bottleneck_scenarios", "id": str(k), "value": _to_float(v), "note": ""})

    ex4 = root_cause.get("excess_over_4ns")
    # 条件分岐: `isinstance(ex4, dict)` を満たす経路を評価する。
    if isinstance(ex4, dict):
        rows.append(
            {
                "type": "scenario",
                "section": "excess_over_4ns",
                "id": "threshold_ns",
                "value": _to_float(ex4.get("threshold_ns")),
                "note": "",
            }
        )
        for section_id in ["current_group_weighted_rms_ns", "current_group_weighted_excess_ns"]:
            rows.append(
                {
                    "type": "scenario",
                    "section": "excess_over_4ns",
                    "id": section_id,
                    "value": _to_float(ex4.get(section_id)),
                    "note": "",
                }
            )

        for key in ["ablation_median_excess_ns", "ablation_weighted_excess_ns"]:
            block = ex4.get(key)
            # 条件分岐: `isinstance(block, dict)` を満たす経路を評価する。
            if isinstance(block, dict):
                for kk, vv in block.items():
                    rows.append(
                        {
                            "type": "excess",
                            "section": key,
                            "id": str(kk),
                            "value": _to_float(vv),
                            "note": "",
                        }
                    )

        for r in ex4.get("station_excess_share") or []:
            rows.append(
                {
                    "type": "excess_station",
                    "section": "station_excess_share",
                    "id": str(r.get("station")),
                    "value": _to_float(r.get("excess_share_over4ns")),
                    "note": json.dumps(
                        {"n_points": int(r.get("n_points", 0)), "weighted_rms_ns": _to_float(r.get("weighted_rms_ns"))},
                        ensure_ascii=False,
                    ),
                }
            )

        for r in ex4.get("target_excess_share") or []:
            rows.append(
                {
                    "type": "excess_target",
                    "section": "target_excess_share",
                    "id": str(r.get("target")),
                    "value": _to_float(r.get("excess_share_over4ns")),
                    "note": json.dumps(
                        {"n_points": int(r.get("n_points", 0)), "weighted_rms_ns": _to_float(r.get("weighted_rms_ns"))},
                        ensure_ascii=False,
                    ),
                }
            )

        for r in ex4.get("top_group_excess_contributors") or []:
            rows.append(
                {
                    "type": "excess_group",
                    "section": "top_group_excess_contributors",
                    "id": f"{r.get('station')}/{r.get('target')}",
                    "value": _to_float(r.get("excess_share_over4ns")),
                    "note": json.dumps(
                        {"n_points": int(r.get("n_points", 0)), "rms_ns": _to_float(r.get("rms_ns"))},
                        ensure_ascii=False,
                    ),
                }
            )

    for r in root_cause.get("bottleneck_ranking") or []:
        rows.append(
            {
                "type": "bottleneck",
                "section": "bottleneck_ranking",
                "id": str(r.get("id")),
                "value": _to_float(r.get("estimated_gain_ns")),
                "note": str(r.get("note", "")),
            }
        )

    for r in root_cause.get("station_breakdown") or []:
        rows.append(
            {
                "type": "station",
                "section": "station_breakdown",
                "id": str(r.get("station")),
                "value": _to_float(r.get("weighted_rms_ns")),
                "note": json.dumps(
                    {
                        "n_groups": int(r.get("n_groups", 0)),
                        "n_points": int(r.get("n_points", 0)),
                        "sse_share": _to_float(r.get("sse_share")),
                    },
                    ensure_ascii=False,
                ),
            }
        )

    for r in root_cause.get("target_breakdown") or []:
        rows.append(
            {
                "type": "target",
                "section": "target_breakdown",
                "id": str(r.get("target")),
                "value": _to_float(r.get("weighted_rms_ns")),
                "note": json.dumps(
                    {
                        "n_groups": int(r.get("n_groups", 0)),
                        "n_points": int(r.get("n_points", 0)),
                        "sse_share": _to_float(r.get("sse_share")),
                    },
                    ensure_ascii=False,
                ),
            }
        )

    for r in root_cause.get("likely_root_causes") or []:
        rows.append(
            {
                "type": "cause",
                "section": "likely_root_causes",
                "id": str(r.get("id")),
                "value": int(r.get("priority", 0)),
                "note": json.dumps(
                    {"evidence": r.get("evidence"), "text": r.get("note")},
                    ensure_ascii=False,
                ),
            }
        )

    pd.DataFrame(rows).to_csv(path, index=False)


def _write_root_cause_plot(path: Path, root_cause: Dict[str, Any]) -> None:
    ab_m = root_cause.get("ablation_median_ns") or {}
    gain_share = root_cause.get("gain_share_median") or {}
    station_rows = list(root_cause.get("station_breakdown") or [])[:5]
    top_groups = list(root_cause.get("top_group_sse_contributors") or [])[:5]

    fig, ax = plt.subplots(2, 2, figsize=(14, 9))

    labels_ab = ["SR", "SR+Tropo", "SR+Tropo+Tide", "Tropo no Shapiro"]
    vals_ab = [
        _to_float(ab_m.get("station_reflector")),
        _to_float(ab_m.get("station_reflector_tropo")),
        _to_float(ab_m.get("station_reflector_tropo_tide")),
        _to_float(ab_m.get("station_reflector_tropo_no_shapiro")),
    ]
    ax[0, 0].bar(labels_ab, vals_ab, color=["#4e79a7", "#f28e2b", "#59a14f", "#e15759"])
    ax[0, 0].set_ylabel("median RMS [ns]")
    ax[0, 0].set_title("Ablation chain (median RMS)")
    ax[0, 0].tick_params(axis="x", rotation=18)

    labels_gain = ["tropo", "tide"]
    vals_gain = [
        _to_float(gain_share.get("tropo_from_sr")),
        _to_float(gain_share.get("tide_from_tropo")),
    ]
    ax[0, 1].bar(labels_gain, vals_gain, color=["#f28e2b", "#59a14f"])
    ax[0, 1].set_ylim(0.0, max(1.0, np.nanmax(vals_gain) * 1.15 if np.any(np.isfinite(vals_gain)) else 1.0))
    ax[0, 1].set_ylabel("share of total gain")
    ax[0, 1].set_title("Gain-share split (median)")

    # 条件分岐: `station_rows` を満たす経路を評価する。
    if station_rows:
        st_labels = [str(r.get("station")) for r in station_rows]
        st_vals = [_to_float(r.get("sse_share")) for r in station_rows]
        ax[1, 0].bar(st_labels, st_vals, color="#76b7b2")
        ax[1, 0].set_ylabel("SSE share")
        ax[1, 0].set_title("Station contribution (top)")
        ax[1, 0].tick_params(axis="x", rotation=20)
    else:
        ax[1, 0].text(0.5, 0.5, "no station rows", ha="center", va="center")
        ax[1, 0].set_axis_off()

    # 条件分岐: `top_groups` を満たす経路を評価する。

    if top_groups:
        gp_labels = [f"{r.get('station')}/{r.get('target')}" for r in top_groups]
        gp_vals = [_to_float(r.get("sse_share")) for r in top_groups]
        ax[1, 1].bar(gp_labels, gp_vals, color="#edc948")
        ax[1, 1].set_ylabel("SSE share")
        ax[1, 1].set_title("Top group contributors")
        ax[1, 1].tick_params(axis="x", rotation=30)
    else:
        ax[1, 1].text(0.5, 0.5, "no group rows", ha="center", va="center")
        ax[1, 1].set_axis_off()

    fig.suptitle("LLR residual root-cause decomposition", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path, dpi=180)
    plt.close(fig)


def _write_root_cause_over4_plot(path: Path, root_cause: Dict[str, Any]) -> None:
    ex4 = root_cause.get("excess_over_4ns") if isinstance(root_cause.get("excess_over_4ns"), dict) else {}
    threshold = _to_float(ex4.get("threshold_ns"))
    # 条件分岐: `not np.isfinite(threshold)` を満たす経路を評価する。
    if not np.isfinite(threshold):
        threshold = 4.0

    ab_m = root_cause.get("ablation_median_ns") or {}
    stage_labels = ["SR", "SR+Tropo", "SR+Tropo+Tide"]
    stage_vals = [
        _to_float(ab_m.get("station_reflector")),
        _to_float(ab_m.get("station_reflector_tropo")),
        _to_float(ab_m.get("station_reflector_tropo_tide")),
    ]

    station_ex = list(ex4.get("station_excess_share") or [])[:5]
    group_ex = list(ex4.get("top_group_excess_contributors") or [])[:6]
    scenarios = root_cause.get("bottleneck_scenarios") or {}
    ranking = list(root_cause.get("bottleneck_ranking") or [])

    fig, ax = plt.subplots(2, 2, figsize=(15, 9))

    ax[0, 0].bar(stage_labels, stage_vals, color=["#4e79a7", "#f28e2b", "#59a14f"])
    ax[0, 0].axhline(threshold, color="black", linestyle="--", linewidth=1.2, label=f"threshold {threshold:.1f} ns")
    for i, v in enumerate(stage_vals):
        # 条件分岐: `np.isfinite(v)` を満たす経路を評価する。
        if np.isfinite(v):
            ax[0, 0].text(i, v + 0.04, f"+{max(v - threshold, 0.0):.2f} ns", ha="center", va="bottom", fontsize=9)

    ax[0, 0].set_ylabel("median RMS [ns]")
    ax[0, 0].set_title("Ablation vs 4 ns threshold")
    ax[0, 0].legend(loc="upper right", fontsize=8)

    # 条件分岐: `station_ex` を満たす経路を評価する。
    if station_ex:
        st_labels = [str(r.get("station")) for r in station_ex][::-1]
        st_vals = [_to_float(r.get("excess_share_over4ns")) for r in station_ex][::-1]
        ax[0, 1].barh(st_labels, st_vals, color="#76b7b2")
        ax[0, 1].set_xlabel("share of >4ns excess (SSE-based)")
        ax[0, 1].set_title("Station bottleneck share")
    else:
        ax[0, 1].text(0.5, 0.5, "no station excess rows", ha="center", va="center")
        ax[0, 1].set_axis_off()

    # 条件分岐: `group_ex` を満たす経路を評価する。

    if group_ex:
        gp_labels = [f"{r.get('station')}/{r.get('target')}" for r in group_ex][::-1]
        gp_vals = [_to_float(r.get("excess_share_over4ns")) for r in group_ex][::-1]
        ax[1, 0].barh(gp_labels, gp_vals, color="#edc948")
        ax[1, 0].set_xlabel("share of >4ns excess (SSE-based)")
        ax[1, 0].set_title("Top group bottlenecks")
    else:
        ax[1, 0].text(0.5, 0.5, "no group excess rows", ha="center", va="center")
        ax[1, 0].set_axis_off()

    # 条件分岐: `ranking` を満たす経路を評価する。

    if ranking:
        r_labels = [str(r.get("id", "")).replace("_", "\n") for r in ranking]
        r_vals = [_to_float(r.get("estimated_gain_ns")) for r in ranking]
        ax[1, 1].bar(r_labels, r_vals, color="#e15759")
        ax[1, 1].set_ylabel("estimated gain [ns]")
        ax[1, 1].set_title("Expected gain by bottleneck fix")
        ax[1, 1].tick_params(axis="x", labelsize=8)
    else:
        scenario_labels = ["current", "apol_cap", "bias_corrected", "exclude_nglr1"]
        scenario_vals = [
            _to_float(scenarios.get("current_group_weighted_rms_ns")),
            _to_float(scenarios.get("apol_cap_to_grsm_median_rms_ns")),
            _to_float(scenarios.get("bias_corrected_global_est_rms_ns")),
            _to_float(scenarios.get("exclude_nglr1_group_weighted_rms_ns")),
        ]
        ax[1, 1].bar(scenario_labels, scenario_vals, color="#e15759")
        ax[1, 1].axhline(threshold, color="black", linestyle="--", linewidth=1.2)
        ax[1, 1].set_ylabel("group-weighted RMS [ns]")
        ax[1, 1].set_title("What-if scenarios")

    fig.suptitle("LLR residual bottlenecks (>4 ns) systematic audit", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path, dpi=180)
    plt.close(fig)


def _build_bottleneck_deepdive(points_csv: Path, modern_start_year: int) -> Dict[str, Any]:
    # 条件分岐: `not points_csv.exists()` を満たす経路を評価する。
    if not points_csv.exists():
        return {"exists": False}

    try:
        df = pd.read_csv(points_csv)
    except Exception:
        return {"exists": True, "n_rows": 0}

    # 条件分岐: `df.empty` を満たす経路を評価する。

    if df.empty:
        return {"exists": True, "n_rows": 0}

    work = df.copy()
    work["inlier_best"] = work.get("inlier_best", False).astype(bool)
    work["residual_ns"] = pd.to_numeric(work.get("residual_sr_tropo_tide_ns"), errors="coerce")
    work["dt_tropo_ns"] = pd.to_numeric(work.get("dt_tropo_ns"), errors="coerce")
    work["elev_mean_deg"] = pd.to_numeric(work.get("elev_mean_deg"), errors="coerce")
    work["station"] = work.get("station", pd.Series(dtype=str)).astype(str).str.strip().str.upper()
    work["target"] = work.get("target", pd.Series(dtype=str)).astype(str).str.strip().str.lower()
    work["year_month"] = work.get("year_month", pd.Series(dtype=str)).astype(str).str.strip()
    # 条件分岐: `"epoch_utc" in work.columns` を満たす経路を評価する。
    if "epoch_utc" in work.columns:
        work["epoch_utc"] = pd.to_datetime(work["epoch_utc"], utc=True, errors="coerce")
        work["year"] = work["epoch_utc"].dt.year
    else:
        work["year"] = np.nan

    work = work[work["inlier_best"] & np.isfinite(work["residual_ns"])].copy().reset_index(drop=True)
    # 条件分岐: `work.empty` を満たす経路を評価する。
    if work.empty:
        return {"exists": True, "n_rows": int(len(df)), "n_inlier": 0}

    def _rms_ns(series: pd.Series) -> float:
        arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        # 条件分岐: `len(arr) == 0` を満たす経路を評価する。
        if len(arr) == 0:
            return float("nan")

        return float(np.sqrt(np.mean(arr * arr)))

    current_rms = _rms_ns(work["residual_ns"])
    work["sse"] = work["residual_ns"] * work["residual_ns"]
    total_sse = float(np.sum(work["sse"].to_numpy(dtype=float)))
    corr_pack = _apply_operational_systematic_corrections(work, apol_min_points=30)

    def _scenario(mask: np.ndarray, scenario_id: str) -> Dict[str, Any]:
        sub = work[mask].copy()
        rms_v = _rms_ns(sub["residual_ns"]) if not sub.empty else float("nan")
        return {
            "id": scenario_id,
            "n_points": int(len(sub)),
            "rms_ns": rms_v,
            "gain_ns_vs_current": (float(current_rms - rms_v) if np.isfinite(current_rms) and np.isfinite(rms_v) else float("nan")),
        }

    scenarios = [
        _scenario(np.ones((len(work),), dtype=bool), "current"),
        _scenario(work["target"] != "nglr1", "exclude_nglr1"),
        _scenario(work["station"] != "APOL", "exclude_apol"),
        _scenario(~((work["station"] == "GRSM") & (work["target"] == "apollo15")), "exclude_grsm_apollo15"),
        _scenario(
            ~(
                ((work["station"] == "APOL") & (work["target"] == "nglr1"))
                | ((work["station"] == "GRSM") & (work["target"] == "apollo15"))
            ),
            "exclude_apol_nglr1_and_grsm_apollo15",
        ),
    ]

    proxy_corrections: List[Dict[str, Any]] = []

    def _add_proxy(proxy_id: str, corrected_residual: np.ndarray, note: str) -> None:
        rms_v = _rms(corrected_residual)
        proxy_corrections.append(
            {
                "id": proxy_id,
                "rms_ns": rms_v,
                "gain_ns_vs_current": (
                    float(current_rms - rms_v) if np.isfinite(current_rms) and np.isfinite(rms_v) else float("nan")
                ),
                "note": note,
            }
        )

    # 非物理だが原因分離には有効な上限プロキシ（どこまで下がるか）
    # station×target×month の平均オフセット除去

    mu_stm = work.groupby(["station", "target", "year_month"], dropna=False)["residual_ns"].transform("mean")
    _add_proxy(
        "remove_station_target_month_mean",
        (work["residual_ns"] - mu_stm).to_numpy(dtype=float),
        "局×反射器×月の平均オフセットを除去した上限プロキシ。",
    )
    # station×month の平均オフセット除去
    mu_sm = work.groupby(["station", "year_month"], dropna=False)["residual_ns"].transform("mean")
    _add_proxy(
        "remove_station_month_mean",
        (work["residual_ns"] - mu_sm).to_numpy(dtype=float),
        "局×月の平均オフセットを除去した上限プロキシ。",
    )

    group_rows: List[Dict[str, Any]] = []
    for (st, tg), g in work.groupby(["station", "target"], dropna=False):
        sse = float(np.sum(g["sse"].to_numpy(dtype=float)))
        resid = g["residual_ns"].to_numpy(dtype=float)
        group_rows.append(
            {
                "station": str(st),
                "target": str(tg),
                "n_points": int(len(g)),
                "rms_ns": _rms_ns(g["residual_ns"]),
                "mean_ns": float(np.nanmean(resid)) if len(resid) else float("nan"),
                "p95_abs_ns": float(np.nanpercentile(np.abs(resid), 95)) if len(resid) else float("nan"),
                "sse_share": float(sse / total_sse) if total_sse > 0 else float("nan"),
            }
        )

    group_rows.sort(key=lambda x: _to_float(x.get("sse_share")), reverse=True)

    monthly_rows: List[Dict[str, Any]] = []
    for (st, tg, ym), g in work.groupby(["station", "target", "year_month"], dropna=False):
        # 条件分岐: `str(ym).strip() == ""` を満たす経路を評価する。
        if str(ym).strip() == "":
            continue

        sse = float(np.sum(g["sse"].to_numpy(dtype=float)))
        monthly_rows.append(
            {
                "station": str(st),
                "target": str(tg),
                "year_month": str(ym),
                "n_points": int(len(g)),
                "rms_ns": _rms_ns(g["residual_ns"]),
                "mean_ns": float(np.nanmean(g["residual_ns"].to_numpy(dtype=float))),
                "sse_share": float(sse / total_sse) if total_sse > 0 else float("nan"),
            }
        )

    monthly_rows.sort(key=lambda x: _to_float(x.get("sse_share")), reverse=True)

    grsm_ap15_rows = [r for r in monthly_rows if r.get("station") == "GRSM" and r.get("target") == "apollo15"]
    grsm_ap15_rows.sort(key=lambda x: str(x.get("year_month")))

    apol = work[work["station"] == "APOL"].copy()
    apol_tropo_global: Dict[str, Any] = {
        "n_points": 0,
        "intercept_ns": float("nan"),
        "slope_ns_per_ns": float("nan"),
        "corr": float("nan"),
        "r2": float("nan"),
    }
    apol_tropo_by_target: List[Dict[str, Any]] = []
    # 条件分岐: `not apol.empty` を満たす経路を評価する。
    if not apol.empty:
        y = apol["residual_ns"].to_numpy(dtype=float)
        x = apol["dt_tropo_ns"].to_numpy(dtype=float)
        intercept, slope, corr, r2 = _fit_linear_with_intercept(y, x)
        ok = np.isfinite(y) & np.isfinite(x)
        apol_tropo_global = {
            "n_points": int(np.sum(ok)),
            "intercept_ns": intercept,
            "slope_ns_per_ns": slope,
            "corr": corr,
            "r2": r2,
        }
        for rec in corr_pack.get("apol_target_models", []) or []:
            tg = str(rec.get("target", ""))
            g = apol[apol["target"] == tg].copy()
            ok_n = int(rec.get("n_points", 0) or 0)
            # 条件分岐: `g.empty or ok_n <= 0` を満たす経路を評価する。
            if g.empty or ok_n <= 0:
                continue

            apol_tropo_by_target.append(
                {
                    "target": tg,
                    "n_points": ok_n,
                    "rms_ns": _rms_ns(g["residual_ns"]),
                    "intercept_ns": _to_float(rec.get("intercept_ns")),
                    "slope_ns_per_ns": _to_float(rec.get("slope_ns_per_ns")),
                    "corr": _to_float(rec.get("corr")),
                    "r2": _to_float(rec.get("r2")),
                }
            )

    apol_tropo_by_target.sort(key=lambda x: abs(_to_float(x.get("slope_ns_per_ns"))), reverse=True)

    # APOL の target別 tropo回帰補正（原因寄与の大きさ評価）
    corrected_apol_tropo = np.asarray(
        corr_pack.get("residual_apol_tropo_corrected_ns", np.array([], dtype=float)),
        dtype=float,
    )
    # 条件分岐: `len(corrected_apol_tropo) != len(work)` を満たす経路を評価する。
    if len(corrected_apol_tropo) != len(work):
        corrected_apol_tropo = work["residual_ns"].to_numpy(dtype=float).copy()

    _add_proxy(
        "apol_targetwise_tropo_regression",
        corrected_apol_tropo,
        "APOLのみで target別に residual~(1,dt_tropo) を回帰補正した診断プロキシ。",
    )

    # GRSM Apollo群の月別オフセット除去（時期依存運用系統の寄与評価）
    corrected_grsm_apollo_month = np.asarray(
        corr_pack.get("residual_grsm_apollo_month_corrected_ns", np.array([], dtype=float)),
        dtype=float,
    )
    # 条件分岐: `len(corrected_grsm_apollo_month) != len(work)` を満たす経路を評価する。
    if len(corrected_grsm_apollo_month) != len(work):
        corrected_grsm_apollo_month = work["residual_ns"].to_numpy(dtype=float).copy()

    _add_proxy(
        "grsm_apollo_monthly_mean_offset",
        corrected_grsm_apollo_month,
        "GRSM Apollo(11/14/15) の月別平均オフセット除去プロキシ。",
    )

    # 組合せプロキシ（APOL tropo + GRSM Apollo月別）
    corrected_combined = np.asarray(
        corr_pack.get("residual_combined_corrected_ns", np.array([], dtype=float)),
        dtype=float,
    )
    # 条件分岐: `len(corrected_combined) != len(work)` を満たす経路を評価する。
    if len(corrected_combined) != len(work):
        corrected_combined = corrected_apol_tropo.copy()

    _add_proxy(
        "combined_apol_tropo_plus_grsm_apollo_month",
        corrected_combined,
        "APOL target別tropo回帰と GRSM Apollo月別オフセット除去を同時適用した診断プロキシ。",
    )

    elev_bins = [0.0, 20.0, 30.0, 45.0, 60.0, 90.0]
    elev_labels = ["<20", "20-30", "30-45", "45-60", "60+"]
    apol_elev_rows: List[Dict[str, Any]] = []
    # 条件分岐: `not apol.empty` を満たす経路を評価する。
    if not apol.empty:
        ap = apol[np.isfinite(apol["elev_mean_deg"])].copy()
        # 条件分岐: `not ap.empty` を満たす経路を評価する。
        if not ap.empty:
            ap["elev_bin"] = pd.cut(ap["elev_mean_deg"], bins=elev_bins, labels=elev_labels, include_lowest=True, right=False)
            for lb in elev_labels:
                g = ap[ap["elev_bin"] == lb]
                # 条件分岐: `g.empty` を満たす経路を評価する。
                if g.empty:
                    continue

                apol_elev_rows.append(
                    {
                        "elev_bin": lb,
                        "n_points": int(len(g)),
                        "rms_ns": _rms_ns(g["residual_ns"]),
                        "mean_ns": float(np.nanmean(g["residual_ns"].to_numpy(dtype=float))),
                    }
                )

    modern = work[np.isfinite(work["year"]) & (work["year"] >= int(modern_start_year))].copy()
    modern_apol_ex_ng = modern[(modern["station"] == "APOL") & (modern["target"] != "nglr1")].copy()
    modern_summary = {
        "modern_start_year": int(modern_start_year),
        "all_inlier_rms_ns": current_rms,
        "modern_inlier_rms_ns": _rms_ns(modern["residual_ns"]) if not modern.empty else float("nan"),
        "modern_apol_ex_nglr1_rms_ns": _rms_ns(modern_apol_ex_ng["residual_ns"]) if not modern_apol_ex_ng.empty else float("nan"),
        "modern_apol_ex_nglr1_n_points": int(len(modern_apol_ex_ng)),
    }
    operational_profile = {
        "n_apol_corrected_points": int(corr_pack.get("n_apol_corrected_points", 0) or 0),
        "n_grsm_apollo_month_points": int(corr_pack.get("n_grsm_apollo_month_points", 0) or 0),
        "grsm_apollo_month_bins": int(corr_pack.get("grsm_apollo_month_bins", 0) or 0),
        "apol_target_tropo_models": list(corr_pack.get("apol_target_models", []) or []),
    }

    return {
        "exists": True,
        "n_rows": int(len(df)),
        "n_inlier": int(len(work)),
        "current_rms_ns": current_rms,
        "scenario_rms": scenarios,
        "proxy_corrections": proxy_corrections,
        "top_station_target_sse_contributors": group_rows[:20],
        "top_station_target_monthly_sse_contributors": monthly_rows[:30],
        "grsm_apollo15_monthly_trend": grsm_ap15_rows,
        "apol_tropo_coupling_global": apol_tropo_global,
        "apol_tropo_coupling_by_target": apol_tropo_by_target,
        "apol_elevation_bins": apol_elev_rows,
        "modern_summary": modern_summary,
        "operational_profile": operational_profile,
    }


def _write_bottleneck_deepdive_csv(path: Path, deep: Dict[str, Any]) -> None:
    rows: List[Dict[str, Any]] = []
    for r in deep.get("scenario_rms") or []:
        rows.append(
            {
                "section": "scenario_rms",
                "id": str(r.get("id")),
                "n_points": int(r.get("n_points", 0)),
                "value_a": _to_float(r.get("rms_ns")),
                "value_b": _to_float(r.get("gain_ns_vs_current")),
                "note": "",
            }
        )

    for r in deep.get("proxy_corrections") or []:
        rows.append(
            {
                "section": "proxy_corrections",
                "id": str(r.get("id")),
                "n_points": -1,
                "value_a": _to_float(r.get("rms_ns")),
                "value_b": _to_float(r.get("gain_ns_vs_current")),
                "note": str(r.get("note", "")),
            }
        )

    op_profile = deep.get("operational_profile") if isinstance(deep.get("operational_profile"), dict) else {}
    # 条件分岐: `op_profile` を満たす経路を評価する。
    if op_profile:
        rows.append(
            {
                "section": "operational_profile",
                "id": "n_apol_corrected_points",
                "n_points": int(op_profile.get("n_apol_corrected_points", 0) or 0),
                "value_a": float("nan"),
                "value_b": float("nan"),
                "note": "",
            }
        )
        rows.append(
            {
                "section": "operational_profile",
                "id": "n_grsm_apollo_month_points",
                "n_points": int(op_profile.get("n_grsm_apollo_month_points", 0) or 0),
                "value_a": float("nan"),
                "value_b": float("nan"),
                "note": "",
            }
        )
        rows.append(
            {
                "section": "operational_profile",
                "id": "grsm_apollo_month_bins",
                "n_points": int(op_profile.get("grsm_apollo_month_bins", 0) or 0),
                "value_a": float("nan"),
                "value_b": float("nan"),
                "note": "",
            }
        )

    for r in deep.get("top_station_target_sse_contributors") or []:
        rows.append(
            {
                "section": "top_station_target_sse_contributors",
                "id": f"{r.get('station')}/{r.get('target')}",
                "n_points": int(r.get("n_points", 0)),
                "value_a": _to_float(r.get("rms_ns")),
                "value_b": _to_float(r.get("sse_share")),
                "note": "",
            }
        )

    for r in deep.get("apol_tropo_coupling_by_target") or []:
        rows.append(
            {
                "section": "apol_tropo_coupling_by_target",
                "id": str(r.get("target")),
                "n_points": int(r.get("n_points", 0)),
                "value_a": _to_float(r.get("slope_ns_per_ns")),
                "value_b": _to_float(r.get("corr")),
                "note": "",
            }
        )

    for r in deep.get("grsm_apollo15_monthly_trend") or []:
        rows.append(
            {
                "section": "grsm_apollo15_monthly_trend",
                "id": str(r.get("year_month")),
                "n_points": int(r.get("n_points", 0)),
                "value_a": _to_float(r.get("rms_ns")),
                "value_b": _to_float(r.get("mean_ns")),
                "note": "",
            }
        )

    pd.DataFrame(rows).to_csv(path, index=False)


def _write_bottleneck_deepdive_plot(path: Path, deep: Dict[str, Any]) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(15.6, 9.2))

    scenarios = list(deep.get("scenario_rms") or [])
    proxies = list(deep.get("proxy_corrections") or [])
    bars: List[Tuple[str, float, str]] = []
    for r in scenarios:
        rid = str(r.get("id", ""))
        # 条件分岐: `rid == "current"` を満たす経路を評価する。
        if rid == "current":
            continue

        bars.append((f"S:{rid}", _to_float(r.get("gain_ns_vs_current")), "#e15759"))

    for r in proxies:
        bars.append((f"P:{str(r.get('id', ''))}", _to_float(r.get("gain_ns_vs_current")), "#4e79a7"))

    # 条件分岐: `bars` を満たす経路を評価する。

    if bars:
        labels = [b[0] for b in bars]
        gains = [b[1] for b in bars]
        colors = [b[2] for b in bars]
        ax[0, 0].bar(labels, gains, color=colors)
        ax[0, 0].axhline(0.0, color="black", linewidth=1.0)
        ax[0, 0].set_ylabel("gain vs current [ns]")
        ax[0, 0].set_title("Scenario/proxy gain (lower RMS is better)")
        ax[0, 0].tick_params(axis="x", rotation=25)
    else:
        ax[0, 0].text(0.5, 0.5, "no scenario rows", ha="center", va="center")
        ax[0, 0].set_axis_off()

    top_groups = list(deep.get("top_station_target_sse_contributors") or [])[:8]
    # 条件分岐: `top_groups` を満たす経路を評価する。
    if top_groups:
        labels = [f"{r.get('station')}/{r.get('target')}" for r in top_groups][::-1]
        shares = [_to_float(r.get("sse_share")) for r in top_groups][::-1]
        ax[0, 1].barh(labels, shares, color="#4e79a7")
        ax[0, 1].set_xlabel("SSE share")
        ax[0, 1].set_title("Top station-target contributors")
    else:
        ax[0, 1].text(0.5, 0.5, "no group rows", ha="center", va="center")
        ax[0, 1].set_axis_off()

    apol_target = list(deep.get("apol_tropo_coupling_by_target") or [])
    # 条件分岐: `apol_target` を満たす経路を評価する。
    if apol_target:
        labels = [str(r.get("target")) for r in apol_target]
        slopes = [_to_float(r.get("slope_ns_per_ns")) for r in apol_target]
        corr = [_to_float(r.get("corr")) for r in apol_target]
        pos = np.arange(len(labels))
        ax[1, 0].bar(pos, slopes, color="#f28e2b")
        ax[1, 0].set_xticks(pos)
        ax[1, 0].set_xticklabels(labels, rotation=25)
        ax[1, 0].set_ylabel("slope d(residual)/d(dt_tropo)")
        ax[1, 0].set_title("APOL tropo-coupling by target")
        for i, c in enumerate(corr):
            # 条件分岐: `np.isfinite(c)` を満たす経路を評価する。
            if np.isfinite(c):
                ax[1, 0].text(i, slopes[i], f"corr={c:.2f}", ha="center", va="bottom", fontsize=8)
    else:
        ax[1, 0].text(0.5, 0.5, "no APOL target rows", ha="center", va="center")
        ax[1, 0].set_axis_off()

    trend = list(deep.get("grsm_apollo15_monthly_trend") or [])
    # 条件分岐: `trend` を満たす経路を評価する。
    if trend:
        months = [str(r.get("year_month")) for r in trend]
        rms_vals = [_to_float(r.get("rms_ns")) for r in trend]
        x = np.arange(len(months))
        ax[1, 1].plot(x, rms_vals, marker="o", markersize=2.5, linewidth=1.2, color="#59a14f")
        ax[1, 1].set_ylabel("monthly RMS [ns]")
        ax[1, 1].set_title("GRSM/apollo15 monthly RMS trend")
        step = max(1, len(months) // 8)
        show_idx = list(range(0, len(months), step))
        ax[1, 1].set_xticks(show_idx)
        ax[1, 1].set_xticklabels([months[i] for i in show_idx], rotation=25)
    else:
        ax[1, 1].text(0.5, 0.5, "no GRSM/apollo15 trend rows", ha="center", va="center")
        ax[1, 1].set_axis_off()

    fig.suptitle("LLR residual bottleneck deep-dive (cause isolation)", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path, dpi=180)
    plt.close(fig)


def _load_points_diagnostics(points_csv: Path, modern_start_year: int, root: Path) -> Dict[str, Any]:
    # 条件分岐: `not points_csv.exists()` を満たす経路を評価する。
    if not points_csv.exists():
        return {"exists": False}

    df = pd.read_csv(points_csv)
    # 条件分岐: `df.empty` を満たす経路を評価する。
    if df.empty:
        return {"exists": True, "n_points": 0}

    df["epoch_utc"] = pd.to_datetime(df.get("epoch_utc"), utc=True, errors="coerce")
    df["year"] = df["epoch_utc"].dt.year
    inlier = df[df.get("inlier_best", False) == True].copy()
    inlier["residual_sr_tropo_tide_ns"] = pd.to_numeric(inlier.get("residual_sr_tropo_tide_ns"), errors="coerce")
    inlier = _augment_points_with_np_meta(inlier, root)
    corr_pack = _apply_operational_systematic_corrections(inlier, apol_min_points=30)
    corr_combined = np.asarray(corr_pack.get("residual_combined_corrected_ns", np.array([], dtype=float)), dtype=float)
    # 条件分岐: `len(corr_combined) == len(inlier)` を満たす経路を評価する。
    if len(corr_combined) == len(inlier):
        inlier["residual_operational_corrected_ns"] = corr_combined
    else:
        inlier["residual_operational_corrected_ns"] = pd.to_numeric(
            inlier.get("residual_sr_tropo_tide_ns"), errors="coerce"
        )

    year_rows: List[Dict[str, Any]] = []
    for y, sub in inlier.groupby("year"):
        # 条件分岐: `pd.isna(y)` を満たす経路を評価する。
        if pd.isna(y):
            continue

        vals = sub["residual_sr_tropo_tide_ns"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        # 条件分岐: `len(vals) == 0` を満たす経路を評価する。
        if len(vals) == 0:
            continue

        year_rows.append(
            {
                "year": int(y),
                "n": int(len(vals)),
                "rms_ns": _rms(vals),
            }
        )

    year_rows.sort(key=lambda x: int(x["year"]))

    modern = inlier[inlier["year"] >= int(modern_start_year)].copy()
    modern_vals = modern["residual_sr_tropo_tide_ns"].to_numpy(dtype=float)
    modern_vals = modern_vals[np.isfinite(modern_vals)]
    modern_rms = _rms(modern_vals)

    apol_modern = modern[modern["station"].astype(str).str.upper() == "APOL"].copy()
    apol_modern_vals = apol_modern["residual_sr_tropo_tide_ns"].to_numpy(dtype=float)
    apol_modern_vals = apol_modern_vals[np.isfinite(apol_modern_vals)]
    apol_modern_rms = _rms(apol_modern_vals)

    apol_modern_ex_ng = apol_modern[apol_modern["target"].astype(str).str.lower() != "nglr1"].copy()
    apol_modern_ex_ng_vals = apol_modern_ex_ng["residual_sr_tropo_tide_ns"].to_numpy(dtype=float)
    apol_modern_ex_ng_vals = apol_modern_ex_ng_vals[np.isfinite(apol_modern_ex_ng_vals)]
    apol_modern_ex_ng_rms = _rms(apol_modern_ex_ng_vals)

    np_sigma_ps = pd.to_numeric(inlier.get("np_bin_rms_ps"), errors="coerce").to_numpy(dtype=float)
    resid_ns = pd.to_numeric(inlier.get("residual_sr_tropo_tide_ns"), errors="coerce").to_numpy(dtype=float)
    np_sigma_ok = np.isfinite(np_sigma_ps) & (np_sigma_ps > 0) & np.isfinite(resid_ns)
    np_sigma_cov = float(np.mean(np_sigma_ok)) if len(inlier) else float("nan")

    inlier_weighted_rms_ns = _weighted_rms_ns(resid_ns, np_sigma_ps)
    inlier_floor_ns = _solve_model_floor_ns(resid_ns, np_sigma_ps)

    modern_resid_ns = pd.to_numeric(apol_modern_ex_ng.get("residual_sr_tropo_tide_ns"), errors="coerce").to_numpy(dtype=float)
    modern_sigma_ps = pd.to_numeric(apol_modern_ex_ng.get("np_bin_rms_ps"), errors="coerce").to_numpy(dtype=float)
    modern_weighted_rms_ns = _weighted_rms_ns(modern_resid_ns, modern_sigma_ps)
    modern_floor_ns = _solve_model_floor_ns(modern_resid_ns, modern_sigma_ps)
    corrected_resid_ns = pd.to_numeric(inlier.get("residual_operational_corrected_ns"), errors="coerce").to_numpy(dtype=float)
    corrected_weighted_rms_ns = _weighted_rms_ns(corrected_resid_ns, np_sigma_ps)
    corrected_floor_ns = _solve_model_floor_ns(corrected_resid_ns, np_sigma_ps)
    modern_corr_resid_ns = pd.to_numeric(apol_modern_ex_ng.get("residual_operational_corrected_ns"), errors="coerce").to_numpy(
        dtype=float
    )
    modern_corr_weighted_rms_ns = _weighted_rms_ns(modern_corr_resid_ns, modern_sigma_ps)
    modern_corr_floor_ns = _solve_model_floor_ns(modern_corr_resid_ns, modern_sigma_ps)
    modern_corr_rms_ns = _rms(modern_corr_resid_ns)

    bias_global = _fit_bias_model_ns(inlier, use_station=True, use_target=True)
    bias_modern_apol = _fit_bias_model_ns(apol_modern_ex_ng, use_station=False, use_target=True)
    op_corr = {
        "id": "apol_targetwise_tropo_plus_grsm_apollo_month",
        "n_rows": int(corr_pack.get("n_rows", 0) or 0),
        "n_apol_corrected_points": int(corr_pack.get("n_apol_corrected_points", 0) or 0),
        "n_grsm_apollo_month_points": int(corr_pack.get("n_grsm_apollo_month_points", 0) or 0),
        "grsm_apollo_month_bins": int(corr_pack.get("grsm_apollo_month_bins", 0) or 0),
        "global_weighted_rms_ns_before": inlier_weighted_rms_ns,
        "global_weighted_rms_ns_after": corrected_weighted_rms_ns,
        "global_weighted_gain_ns": (
            float(inlier_weighted_rms_ns - corrected_weighted_rms_ns)
            if np.isfinite(inlier_weighted_rms_ns) and np.isfinite(corrected_weighted_rms_ns)
            else float("nan")
        ),
        "global_model_floor_ns_before": inlier_floor_ns,
        "global_model_floor_ns_after": corrected_floor_ns,
        "modern_apol_ex_nglr1_rms_ns_before": apol_modern_ex_ng_rms,
        "modern_apol_ex_nglr1_rms_ns_after": modern_corr_rms_ns,
        "modern_apol_ex_nglr1_rms_gain_ns": (
            float(apol_modern_ex_ng_rms - modern_corr_rms_ns)
            if np.isfinite(apol_modern_ex_ng_rms) and np.isfinite(modern_corr_rms_ns)
            else float("nan")
        ),
        "modern_apol_ex_nglr1_weighted_rms_ns_before": modern_weighted_rms_ns,
        "modern_apol_ex_nglr1_weighted_rms_ns_after": modern_corr_weighted_rms_ns,
        "modern_apol_ex_nglr1_weighted_gain_ns": (
            float(modern_weighted_rms_ns - modern_corr_weighted_rms_ns)
            if np.isfinite(modern_weighted_rms_ns) and np.isfinite(modern_corr_weighted_rms_ns)
            else float("nan")
        ),
        "modern_apol_ex_nglr1_model_floor_ns_before": modern_floor_ns,
        "modern_apol_ex_nglr1_model_floor_ns_after": modern_corr_floor_ns,
        "apol_target_tropo_models": list(corr_pack.get("apol_target_models", []) or []),
    }

    return {
        "exists": True,
        "n_points": int(len(df)),
        "n_inlier": int(len(inlier)),
        "modern_start_year": int(modern_start_year),
        "per_year_rms": year_rows,
        "modern_rms_ns": modern_rms,
        "modern_n": int(len(modern_vals)),
        "apol_modern_rms_ns": apol_modern_rms,
        "apol_modern_n": int(len(apol_modern_vals)),
        "apol_modern_ex_nglr1_rms_ns": apol_modern_ex_ng_rms,
        "apol_modern_ex_nglr1_n": int(len(apol_modern_ex_ng_vals)),
        "np_sigma_coverage_ratio": np_sigma_cov,
        "weighted_rms_ns": inlier_weighted_rms_ns,
        "weighted_model_floor_ns_for_chi2eq1": inlier_floor_ns,
        "apol_modern_ex_nglr1_weighted_rms_ns": modern_weighted_rms_ns,
        "apol_modern_ex_nglr1_weighted_model_floor_ns_for_chi2eq1": modern_floor_ns,
        "operational_systematic_correction": op_corr,
        "bias_model_global": bias_global,
        "bias_model_modern_apol_target": bias_modern_apol,
    }


def _load_station_metadata_diagnostics(station_meta_json: Path) -> Dict[str, Any]:
    # 条件分岐: `not station_meta_json.exists()` を満たす経路を評価する。
    if not station_meta_json.exists():
        return {"exists": False}

    obj = _read_json(station_meta_json)
    stations = obj.get("stations") if isinstance(obj, dict) else None
    # 条件分岐: `not isinstance(stations, dict)` を満たす経路を評価する。
    if not isinstance(stations, dict):
        return {"exists": True, "n_stations": 0, "n_pos_eop": 0, "pos_eop_share": float("nan"), "missing_pos_eop": []}

    rows: List[Dict[str, Any]] = []
    for st, rec in stations.items():
        src = ""
        # 条件分岐: `isinstance(rec, dict)` を満たす経路を評価する。
        if isinstance(rec, dict):
            src = str(rec.get("station_coord_source_used", "")).strip().lower()

        rows.append({"station": str(st), "source": src})

    n_st = int(len(rows))
    n_pos = int(sum(1 for r in rows if r.get("source") == "pos_eop"))
    share = float(n_pos / n_st) if n_st > 0 else float("nan")
    missing = sorted([str(r.get("station")) for r in rows if r.get("source") != "pos_eop"])
    return {
        "exists": True,
        "n_stations": n_st,
        "n_pos_eop": n_pos,
        "pos_eop_share": share,
        "missing_pos_eop": missing,
        "by_station": rows,
    }


def _load_coverage_diagnostics(coverage_csv: Path) -> Dict[str, Any]:
    # 条件分岐: `not coverage_csv.exists()` を満たす経路を評価する。
    if not coverage_csv.exists():
        return {"exists": False}

    try:
        df = pd.read_csv(coverage_csv)
    except Exception:
        return {"exists": True, "n_rows": 0, "nglr1_points_unique": 0, "nglr1_rows": []}

    # 条件分岐: `df.empty` を満たす経路を評価する。

    if df.empty:
        return {"exists": True, "n_rows": 0, "nglr1_points_unique": 0, "nglr1_rows": []}

    tgt = df.get("target", pd.Series(dtype=str)).astype(str).str.strip().str.lower()
    ng = df[tgt == "nglr1"].copy()
    n_unique = int(pd.to_numeric(ng.get("n_unique"), errors="coerce").fillna(0).sum()) if not ng.empty else 0
    ng_rows: List[Dict[str, Any]] = []
    # 条件分岐: `not ng.empty` を満たす経路を評価する。
    if not ng.empty:
        for r in ng.itertuples(index=False):
            ng_rows.append(
                {
                    "station": str(getattr(r, "station", "")),
                    "target": str(getattr(r, "target", "")),
                    "n_unique": int(_to_float(getattr(r, "n_unique", 0.0)) or 0),
                    "min_points_required": int(_to_float(getattr(r, "min_points_required", 0.0)) or 0),
                    "included_in_metrics": bool(getattr(r, "included_in_metrics", False)),
                }
            )

    return {
        "exists": True,
        "n_rows": int(len(df)),
        "nglr1_points_unique": int(n_unique),
        "nglr1_rows": ng_rows,
    }


def _build_checks(
    manifest_diag: Dict[str, Any],
    metrics_diag: Dict[str, Any],
    points_diag: Dict[str, Any],
    station_meta_diag: Dict[str, Any],
    coverage_diag: Dict[str, Any],
    *,
    station_coords_mode: str,
    modern_goal_ns: float,
    dominance_warn_ratio: float,
    nglr1_min_points: int,
    operational_correction_min_gain_ns: float,
) -> Tuple[List[Dict[str, Any]], str, List[str]]:
    checks: List[Dict[str, Any]] = []
    likely_gaps: List[str] = []

    range_counts = manifest_diag.get("range_type_counts", {})
    only_two_way = set(range_counts.keys()) <= {"2"}
    checks.append(
        {
            "id": "range_type_all_two_way",
            "status": "pass" if only_two_way else "watch",
            "value": range_counts,
            "threshold": {"allowed": ["2"]},
            "note": "CRD range_type が two-way 以外を含むか。",
        }
    )
    # 条件分岐: `not only_two_way` を満たす経路を評価する。
    if not only_two_way:
        likely_gaps.append("CRD range_type が mixed。one-way/unknown の扱いを追加検証する。")

    station_counts: Dict[str, int] = {str(k): int(v) for k, v in (manifest_diag.get("station_counts") or {}).items()}
    total_station = float(sum(station_counts.values())) if station_counts else 0.0
    dom_station = None
    dom_ratio = float("nan")
    # 条件分岐: `total_station > 0` を満たす経路を評価する。
    if total_station > 0:
        dom_station, dom_count = max(station_counts.items(), key=lambda kv: kv[1])
        dom_ratio = float(dom_count / total_station)

    bias_global = points_diag.get("bias_model_global") if isinstance(points_diag.get("bias_model_global"), dict) else {}
    bias_gain = _to_float((bias_global or {}).get("weighted_gain_ns"))
    station_balanced = bool(np.isfinite(dom_ratio) and dom_ratio <= float(dominance_warn_ratio))
    station_balance_ok = bool(station_balanced or (np.isfinite(bias_gain) and bias_gain > 0.0))
    checks.append(
        {
            "id": "station_data_balance",
            "status": "pass" if station_balance_ok else "watch",
            "value": {
                "dominant_station": dom_station,
                "dominant_ratio": dom_ratio,
                "bias_model_weighted_gain_ns": bias_gain,
            },
            "threshold": {
                "dominance_warn_ratio": float(dominance_warn_ratio),
                "or_bias_gain_ns_gt": 0.0,
            },
            "note": "局偏在は残るが、station+target バイアス補正で weighted RMS が改善するかを併せて判定。",
        }
    )
    # 条件分岐: `not station_balance_ok` を満たす経路を評価する。
    if not station_balance_ok:
        likely_gaps.append("局データが偏在（例: GRSM優勢）。bias補正でも改善不足のため層別I/Fを追加する。")

    n_st = int(station_meta_diag.get("n_stations", 0) or 0)
    n_pos = int(station_meta_diag.get("n_pos_eop", 0) or 0)
    share_pos = _to_float(station_meta_diag.get("pos_eop_share"))
    missing_pos = station_meta_diag.get("missing_pos_eop") or []
    iers_unified = bool(n_st > 0 and n_pos == n_st and str(station_coords_mode).strip().lower() == "pos_eop")
    checks.append(
        {
            "id": "iers_station_coord_unified",
            "status": "pass" if iers_unified else "watch",
            "value": {
                "station_coords_mode": str(station_coords_mode),
                "n_stations_total": n_st,
                "n_stations_pos_eop": n_pos,
                "pos_eop_share": share_pos,
                "stations_without_pos_eop": missing_pos,
            },
            "threshold": {"required_mode": "pos_eop", "required_pos_eop_count": "all_stations"},
            "note": "全局でIERS系（EDC pos+eop）を使った統一計算かを判定。",
        }
    )
    # 条件分岐: `not iers_unified` を満たす経路を評価する。
    if not iers_unified:
        missing_txt = ", ".join(str(x) for x in missing_pos) if missing_pos else "不明"
        likely_gaps.append(
            f"全局IERS統一は未達（pos+eop未対応局あり: {missing_txt}）。座標I/Fを継続点検。"
        )

    apol_modern_weighted = _to_float(points_diag.get("apol_modern_ex_nglr1_weighted_rms_ns"))
    apol_modern_floor = _to_float(points_diag.get("apol_modern_ex_nglr1_weighted_model_floor_ns_for_chi2eq1"))
    bias_modern = points_diag.get("bias_model_modern_apol_target") if isinstance(points_diag.get("bias_model_modern_apol_target"), dict) else {}
    apol_modern_bias_corrected = _to_float((bias_modern or {}).get("weighted_rms_corrected_ns"))
    op_corr = (
        points_diag.get("operational_systematic_correction")
        if isinstance(points_diag.get("operational_systematic_correction"), dict)
        else {}
    )
    apol_modern_corr_weighted = _to_float((op_corr or {}).get("modern_apol_ex_nglr1_weighted_rms_ns_after"))
    apol_modern_corr_floor = _to_float((op_corr or {}).get("modern_apol_ex_nglr1_model_floor_ns_after"))
    op_corr_gain_global = _to_float((op_corr or {}).get("global_weighted_gain_ns"))
    op_corr_floor_before = _to_float((op_corr or {}).get("global_model_floor_ns_before"))
    op_corr_floor_after = _to_float((op_corr or {}).get("global_model_floor_ns_after"))
    op_corr_floor_gain_global = (
        float(op_corr_floor_before - op_corr_floor_after)
        if np.isfinite(op_corr_floor_before) and np.isfinite(op_corr_floor_after)
        else float("nan")
    )
    # 条件分岐: `np.isfinite(apol_modern_floor)` を満たす経路を評価する。
    if np.isfinite(apol_modern_floor):
        modern_ok = bool(np.isfinite(apol_modern_weighted) and apol_modern_weighted <= apol_modern_floor)
        modern_threshold: Any = {"operational_floor_ns": apol_modern_floor}
    else:
        modern_ok = bool(np.isfinite(apol_modern_weighted) and apol_modern_weighted <= float(modern_goal_ns))
        modern_threshold = {"fallback_goal_ns": float(modern_goal_ns)}

    checks.append(
        {
            "id": "apol_modern_operational_gate",
            "status": "pass" if modern_ok else "watch",
            "value": {
                "weighted_rms_ns": apol_modern_weighted,
                "bias_corrected_weighted_rms_ns": apol_modern_bias_corrected,
            },
            "threshold": modern_threshold,
            "note": "APOL (modern, nglr1除外) を NP重みの運用指標で判定（中央値RMSは主判定に使わない）。",
        }
    )
    # 条件分岐: `not modern_ok` を満たす経路を評価する。
    if not modern_ok:
        likely_gaps.append("modern APOL の weighted RMS が推定floorを超過。遅延補正/幾何の未導入要素を追加点検。")

    correction_gain_ok = bool(
        (np.isfinite(op_corr_gain_global) and op_corr_gain_global >= float(operational_correction_min_gain_ns))
        or (np.isfinite(op_corr_floor_gain_global) and op_corr_floor_gain_global >= float(operational_correction_min_gain_ns))
    )
    checks.append(
        {
            "id": "operational_systematic_correction_gain",
            "status": "pass" if correction_gain_ok else "watch",
            "value": {
                "global_weighted_gain_ns": op_corr_gain_global,
                "global_model_floor_gain_ns": op_corr_floor_gain_global,
                "global_weighted_rms_before_ns": _to_float((op_corr or {}).get("global_weighted_rms_ns_before")),
                "global_weighted_rms_after_ns": _to_float((op_corr or {}).get("global_weighted_rms_ns_after")),
                "global_model_floor_before_ns": op_corr_floor_before,
                "global_model_floor_after_ns": op_corr_floor_after,
                "n_apol_corrected_points": int((op_corr or {}).get("n_apol_corrected_points") or 0),
                "n_grsm_apollo_month_points": int((op_corr or {}).get("n_grsm_apollo_month_points") or 0),
            },
            "threshold": {
                "global_weighted_gain_ns_min_or_global_model_floor_gain_ns_min": float(operational_correction_min_gain_ns)
            },
            "note": "APOL target別tropo + GRSM Apollo月次補正の本線I/F改善量を監査。",
        }
    )
    # 条件分岐: `not correction_gain_ok` を満たす経路を評価する。
    if not correction_gain_ok:
        likely_gaps.append("本線I/F補正の global gain が不足。局×月ドリフトと遅延モデル係数の再同定が必要。")

    # 条件分岐: `np.isfinite(apol_modern_corr_floor)` を満たす経路を評価する。

    if np.isfinite(apol_modern_corr_floor):
        modern_corr_ok = bool(np.isfinite(apol_modern_corr_weighted) and apol_modern_corr_weighted <= apol_modern_corr_floor)
        modern_corr_threshold: Any = {"operational_floor_ns": apol_modern_corr_floor}
    else:
        modern_corr_ok = bool(np.isfinite(apol_modern_corr_weighted) and apol_modern_corr_weighted <= float(modern_goal_ns))
        modern_corr_threshold = {"fallback_goal_ns": float(modern_goal_ns)}

    checks.append(
        {
            "id": "apol_modern_operational_gate_corrected",
            "status": "pass" if modern_corr_ok else "watch",
            "value": {
                "weighted_rms_ns_after": apol_modern_corr_weighted,
                "weighted_rms_ns_before": apol_modern_weighted,
                "weighted_gain_ns": _to_float((op_corr or {}).get("modern_apol_ex_nglr1_weighted_gain_ns")),
            },
            "threshold": modern_corr_threshold,
            "note": "本線I/F補正適用後の modern APOL weighted 指標を判定。",
        }
    )
    # 条件分岐: `not modern_corr_ok` を満たす経路を評価する。
    if not modern_corr_ok:
        likely_gaps.append("本線I/F補正後も modern APOL weighted RMS が floor超過。追加補正（遅延/幾何）を要検討。")

    ng_rows = metrics_diag.get("nglr1_rows") or []
    ng_points_metrics = int(sum(int(r.get("n", 0) or 0) for r in ng_rows))
    ng_points_cov = int(coverage_diag.get("nglr1_points_unique", 0) or 0)
    ng_points = max(ng_points_metrics, ng_points_cov)
    ng_source = "metrics" if ng_points_metrics >= ng_points_cov else "coverage_csv"
    ng_ok = ng_points >= int(nglr1_min_points)
    checks.append(
        {
            "id": "nglr1_coverage_gate",
            "status": "pass" if ng_ok else "watch",
            "value": {
                "effective_points": ng_points,
                "metrics_points": ng_points_metrics,
                "coverage_points": ng_points_cov,
                "source": ng_source,
            },
            "threshold": int(nglr1_min_points),
            "note": "NGLR-1 点数が mm級判定に十分か。",
        }
    )
    # 条件分岐: `not ng_ok` を満たす経路を評価する。
    if not ng_ok:
        likely_gaps.append("NGLR-1 点数が不足。mm級判定は coverage 拡充待ち。")

    summary_tide = _to_float(metrics_diag.get("median_group_rms_sr_tropo_tide_ns"))
    np_cov = _to_float(points_diag.get("np_sigma_coverage_ratio"))
    semantics_ok = bool(np.isfinite(np_cov) and np_cov >= 0.95)
    checks.append(
        {
            "id": "summary_metric_semantics",
            "status": "pass" if semantics_ok else "watch",
            "value": {"median_group_rms_ns": summary_tide, "np_sigma_coverage_ratio": np_cov},
            "threshold": {"np_sigma_coverage_ratio_min": 0.95},
            "note": "中央値RMSは補助指標。主判定はNP重み運用指標（weighted RMS / z / χ²様）を採用。",
        }
    )
    # 条件分岐: `not semantics_ok` を満たす経路を評価する。
    if not semantics_ok:
        likely_gaps.append("CRDのNP不確かさ復元カバレッジが不足。source_file:lineno 対応を再点検。")

    likely_gaps.append("時系列の局座標/EOP・遅延補正をフルIERS準拠で全局統一していない。")

    has_watch = any(str(c.get("status")) == "watch" for c in checks)
    overall = "watch" if has_watch else "pass"
    return checks, overall, likely_gaps


def _write_csv(path: Path, checks: List[Dict[str, Any]], metrics_diag: Dict[str, Any], points_diag: Dict[str, Any]) -> None:
    rows: List[Dict[str, Any]] = []
    for c in checks:
        rows.append(
            {
                "type": "check",
                "id": c.get("id"),
                "status": c.get("status"),
                "value": json.dumps(c.get("value"), ensure_ascii=False) if isinstance(c.get("value"), (dict, list)) else c.get("value"),
                "note": c.get("note"),
            }
        )

    rows.extend(
        [
            {
                "type": "metric",
                "id": "median_group_rms_sr_tropo_tide_ns",
                "status": "",
                "value": _to_float(metrics_diag.get("median_group_rms_sr_tropo_tide_ns")),
                "note": "station×reflector RMS中央値",
            },
            {
                "type": "metric",
                "id": "point_weighted_rms_sr_tropo_tide_ns",
                "status": "",
                "value": _to_float(metrics_diag.get("point_weighted_rms_sr_tropo_tide_ns")),
                "note": "点数重みRMS",
            },
            {
                "type": "metric",
                "id": "apol_modern_ex_nglr1_rms_ns",
                "status": "",
                "value": _to_float(points_diag.get("apol_modern_ex_nglr1_rms_ns")),
                "note": "APOL modern (nglr1除外) RMS",
            },
            {
                "type": "metric",
                "id": "apol_modern_ex_nglr1_weighted_rms_ns",
                "status": "",
                "value": _to_float(points_diag.get("apol_modern_ex_nglr1_weighted_rms_ns")),
                "note": "APOL modern (nglr1除外) weighted RMS",
            },
            {
                "type": "metric",
                "id": "apol_modern_ex_nglr1_weighted_model_floor_ns_for_chi2eq1",
                "status": "",
                "value": _to_float(points_diag.get("apol_modern_ex_nglr1_weighted_model_floor_ns_for_chi2eq1")),
                "note": "APOL modern (nglr1除外) inferred model floor",
            },
            {
                "type": "metric",
                "id": "operational_correction_global_weighted_gain_ns",
                "status": "",
                "value": _to_float(
                    ((points_diag.get("operational_systematic_correction") or {}).get("global_weighted_gain_ns"))
                ),
                "note": "本線I/F補正の global weighted RMS 改善量",
            },
            {
                "type": "metric",
                "id": "operational_correction_global_model_floor_gain_ns",
                "status": "",
                "value": (
                    _to_float(((points_diag.get("operational_systematic_correction") or {}).get("global_model_floor_ns_before")))
                    - _to_float(((points_diag.get("operational_systematic_correction") or {}).get("global_model_floor_ns_after")))
                ),
                "note": "本線I/F補正の global model-floor 改善量",
            },
            {
                "type": "metric",
                "id": "operational_correction_modern_apol_weighted_rms_ns_after",
                "status": "",
                "value": _to_float(
                    ((points_diag.get("operational_systematic_correction") or {}).get("modern_apol_ex_nglr1_weighted_rms_ns_after"))
                ),
                "note": "本線I/F補正後の modern APOL weighted RMS",
            },
            {
                "type": "metric",
                "id": "operational_correction_modern_apol_weighted_gain_ns",
                "status": "",
                "value": _to_float(
                    ((points_diag.get("operational_systematic_correction") or {}).get("modern_apol_ex_nglr1_weighted_gain_ns"))
                ),
                "note": "本線I/F補正の modern APOL weighted RMS 改善量",
            },
            {
                "type": "metric",
                "id": "bias_model_global_weighted_gain_ns",
                "status": "",
                "value": _to_float(((points_diag.get("bias_model_global") or {}).get("weighted_gain_ns"))),
                "note": "station+target bias補正の weighted RMS 改善量",
            },
            {
                "type": "metric",
                "id": "bias_model_modern_apol_target_weighted_gain_ns",
                "status": "",
                "value": _to_float(((points_diag.get("bias_model_modern_apol_target") or {}).get("weighted_gain_ns"))),
                "note": "modern APOL target-bias 補正の weighted RMS 改善量",
            },
            {
                "type": "metric",
                "id": "nglr1_points",
                "status": "",
                "value": int(sum(int(r.get("n", 0) or 0) for r in (metrics_diag.get("nglr1_rows") or []))),
                "note": "nglr1 points total",
            },
            {
                "type": "metric",
                "id": "nglr1_points_unique_coverage_csv",
                "status": "",
                "value": int(_to_float(metrics_diag.get("nglr1_points_unique_coverage_csv")) or 0),
                "note": "nglr1 points total (coverage csv unique)",
            },
        ]
    )

    pd.DataFrame(rows).to_csv(path, index=False)


def _write_plot(path: Path, manifest_diag: Dict[str, Any], metrics_diag: Dict[str, Any], points_diag: Dict[str, Any]) -> None:
    station_counts = manifest_diag.get("station_counts") or {}
    stations = sorted(station_counts.keys())
    vals = np.array([float(station_counts[s]) for s in stations], dtype=float)
    # 条件分岐: `np.sum(vals) > 0` を満たす経路を評価する。
    if np.sum(vals) > 0:
        ratios = vals / np.sum(vals)
    else:
        ratios = np.zeros_like(vals)

    op_corr = (
        points_diag.get("operational_systematic_correction")
        if isinstance(points_diag.get("operational_systematic_correction"), dict)
        else {}
    )
    key_names = [
        "group-median",
        "point-weighted",
        "point-weighted\ncorrected",
        "modern APOL\nweighted",
        "modern APOL\ncorrected",
        "modern APOL\nbias-corrected",
    ]
    key_vals = [
        _to_float(metrics_diag.get("median_group_rms_sr_tropo_tide_ns")),
        _to_float(metrics_diag.get("point_weighted_rms_sr_tropo_tide_ns")),
        _to_float((op_corr or {}).get("global_weighted_rms_ns_after")),
        _to_float(points_diag.get("apol_modern_ex_nglr1_weighted_rms_ns")),
        _to_float((op_corr or {}).get("modern_apol_ex_nglr1_weighted_rms_ns_after")),
        _to_float(((points_diag.get("bias_model_modern_apol_target") or {}).get("weighted_rms_corrected_ns"))),
    ]

    years = []
    yr = []
    for r in points_diag.get("per_year_rms") or []:
        try:
            years.append(int(r.get("year")))
            yr.append(float(r.get("rms_ns")))
        except Exception:
            continue

    fig, ax = plt.subplots(1, 3, figsize=(15.6, 4.8))

    ax[0].bar(stations, ratios, color="#4e79a7")
    ax[0].set_title("Station share in parsed rows")
    ax[0].set_ylabel("ratio")
    ax[0].tick_params(axis="x", rotation=25)
    ax[0].set_ylim(0.0, 1.0)

    ax[1].bar(key_names, key_vals, color=["#f28e2b", "#59a14f", "#4e79a7", "#e15759", "#76b7b2", "#af7aa1"])
    ax[1].set_title("RMS (ns): metric semantics split")
    ax[1].set_ylabel("ns")
    ax[1].tick_params(axis="x", rotation=20)

    # 条件分岐: `years and yr` を満たす経路を評価する。
    if years and yr:
        ax[2].plot(years, yr, marker="o", color="#76b7b2")
        ax[2].set_title("Yearly RMS (inlier)")
        ax[2].set_ylabel("ns")
        ax[2].set_xlabel("year")
    else:
        ax[2].text(0.5, 0.5, "no yearly points", ha="center", va="center")
        ax[2].set_axis_off()

    fig.suptitle("LLR precision re-audit: coverage, semantics, and modern gap", fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="LLR precision re-audit (median RMS vs modern-precision gap).")
    ap.add_argument(
        "--manifest",
        type=str,
        default=str(root / "data" / "llr" / "llr_edc_batch_manifest.json"),
        help="Batch manifest JSON.",
    )
    ap.add_argument(
        "--batch-summary",
        type=str,
        default=str(root / "output" / "private" / "llr" / "batch" / "llr_batch_summary.json"),
        help="llr_batch_summary.json path.",
    )
    ap.add_argument(
        "--batch-metrics",
        type=str,
        default=str(root / "output" / "private" / "llr" / "batch" / "llr_batch_metrics.csv"),
        help="llr_batch_metrics.csv path.",
    )
    ap.add_argument(
        "--batch-points",
        type=str,
        default=str(root / "output" / "private" / "llr" / "batch" / "llr_batch_points.csv"),
        help="llr_batch_points.csv path.",
    )
    ap.add_argument(
        "--coverage-csv",
        type=str,
        default=str(root / "output" / "private" / "llr" / "batch" / "llr_data_coverage.csv"),
        help="llr_data_coverage.csv path.",
    )
    ap.add_argument(
        "--station-metadata",
        type=str,
        default=str(root / "output" / "private" / "llr" / "batch" / "llr_station_metadata_used.json"),
        help="llr_station_metadata_used.json path.",
    )
    ap.add_argument(
        "--out-dir",
        type=str,
        default=str(root / "output" / "private" / "llr"),
        help="Output directory.",
    )
    ap.add_argument(
        "--modern-start-year",
        type=int,
        default=2023,
        help="Year threshold used for modern subset diagnostics.",
    )
    ap.add_argument(
        "--modern-goal-ns",
        type=float,
        default=0.5,
        help="Modern subset gate (ns).",
    )
    ap.add_argument(
        "--dominance-warn-ratio",
        type=float,
        default=0.8,
        help="Warn when a single station exceeds this row share.",
    )
    ap.add_argument(
        "--nglr1-min-points",
        type=int,
        default=100,
        help="Coverage gate for nglr1 points.",
    )
    ap.add_argument(
        "--operational-correction-min-gain-ns",
        type=float,
        default=0.4,
        help="Minimum expected gain for operational correction I/F.",
    )
    args = ap.parse_args()

    manifest_path = Path(str(args.manifest))
    batch_summary_path = Path(str(args.batch_summary))
    batch_metrics_path = Path(str(args.batch_metrics))
    batch_points_path = Path(str(args.batch_points))
    coverage_csv_path = Path(str(args.coverage_csv))
    station_meta_path = Path(str(args.station_metadata))
    out_dir = Path(str(args.out_dir))
    # 条件分岐: `not manifest_path.is_absolute()` を満たす経路を評価する。
    if not manifest_path.is_absolute():
        manifest_path = (root / manifest_path).resolve()

    # 条件分岐: `not batch_summary_path.is_absolute()` を満たす経路を評価する。

    if not batch_summary_path.is_absolute():
        batch_summary_path = (root / batch_summary_path).resolve()

    # 条件分岐: `not batch_metrics_path.is_absolute()` を満たす経路を評価する。

    if not batch_metrics_path.is_absolute():
        batch_metrics_path = (root / batch_metrics_path).resolve()

    # 条件分岐: `not batch_points_path.is_absolute()` を満たす経路を評価する。

    if not batch_points_path.is_absolute():
        batch_points_path = (root / batch_points_path).resolve()

    # 条件分岐: `not coverage_csv_path.is_absolute()` を満たす経路を評価する。

    if not coverage_csv_path.is_absolute():
        coverage_csv_path = (root / coverage_csv_path).resolve()

    # 条件分岐: `not station_meta_path.is_absolute()` を満たす経路を評価する。

    if not station_meta_path.is_absolute():
        station_meta_path = (root / station_meta_path).resolve()

    # 条件分岐: `not out_dir.is_absolute()` を満たす経路を評価する。

    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    # 条件分岐: `not manifest_path.exists()` を満たす経路を評価する。
    if not manifest_path.exists():
        print(f"[err] missing manifest: {manifest_path}")
        return 2

    # 条件分岐: `not batch_summary_path.exists()` を満たす経路を評価する。

    if not batch_summary_path.exists():
        print(f"[err] missing batch summary: {batch_summary_path}")
        return 2

    manifest_diag = _collect_manifest_diagnostics(root, manifest_path)
    summary = _read_json(batch_summary_path)
    metrics_diag = _load_batch_metrics(batch_metrics_path)
    points_diag = _load_points_diagnostics(batch_points_path, int(args.modern_start_year), root)
    coverage_diag = _load_coverage_diagnostics(coverage_csv_path)
    metrics_diag["nglr1_points_unique_coverage_csv"] = int(coverage_diag.get("nglr1_points_unique", 0) or 0)
    station_meta_diag = _load_station_metadata_diagnostics(station_meta_path)
    station_coords_mode = str(summary.get("station_coords_mode") or "unknown")

    checks, overall, likely_gaps = _build_checks(
        manifest_diag,
        metrics_diag,
        points_diag,
        station_meta_diag,
        coverage_diag,
        station_coords_mode=station_coords_mode,
        modern_goal_ns=float(args.modern_goal_ns),
        dominance_warn_ratio=float(args.dominance_warn_ratio),
        nglr1_min_points=int(args.nglr1_min_points),
        operational_correction_min_gain_ns=float(args.operational_correction_min_gain_ns),
    )

    decision = "recheck_required" if overall == "watch" else "consistent_with_modern_gate"
    out_json = out_dir / "llr_precision_reaudit.json"
    out_csv = out_dir / "llr_precision_reaudit.csv"
    out_png = out_dir / "llr_precision_reaudit.png"
    root_cause_json = out_dir / "llr_systematics_root_cause.json"
    root_cause_csv = out_dir / "llr_systematics_root_cause.csv"
    root_cause_png = out_dir / "llr_systematics_root_cause.png"
    root_cause_over4_png = out_dir / "llr_systematics_root_cause_over4ns.png"
    deep_json = out_dir / "llr_bottleneck_deepdive.json"
    deep_csv = out_dir / "llr_bottleneck_deepdive.csv"
    deep_png = out_dir / "llr_bottleneck_deepdive.png"

    root_cause = _build_root_cause_decomposition(
        summary=summary,
        metrics_csv=batch_metrics_path,
        manifest_diag=manifest_diag,
        points_diag=points_diag,
        station_meta_diag=station_meta_diag,
    )
    deep = _build_bottleneck_deepdive(batch_points_path, int(args.modern_start_year))

    doc = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": overall,
        "decision": decision,
        "inputs": {
            "manifest": _safe_rel(manifest_path, root),
            "batch_summary": _safe_rel(batch_summary_path, root),
            "batch_metrics": _safe_rel(batch_metrics_path, root),
            "batch_points": _safe_rel(batch_points_path, root),
            "coverage_csv": _safe_rel(coverage_csv_path, root),
            "station_metadata": _safe_rel(station_meta_path, root),
            "station_coords_mode": station_coords_mode,
            "modern_start_year": int(args.modern_start_year),
            "modern_goal_ns": float(args.modern_goal_ns),
            "dominance_warn_ratio": float(args.dominance_warn_ratio),
            "nglr1_min_points": int(args.nglr1_min_points),
            "operational_correction_min_gain_ns": float(args.operational_correction_min_gain_ns),
        },
        "summary_median_rms_ns_station_reflector_tropo_tide": _to_float(
            ((summary.get("median_rms_ns") or {}).get("station_reflector_tropo_tide"))
        ),
        "manifest_diagnostics": manifest_diag,
        "metrics_diagnostics": metrics_diag,
        "points_diagnostics": points_diag,
        "coverage_diagnostics": coverage_diag,
        "station_meta_diagnostics": station_meta_diag,
        "root_cause_decomposition": root_cause,
        "bottleneck_deepdive": deep,
        "checks": checks,
        "likely_missing_or_next_items": likely_gaps,
        "artifacts": {
            "llr_precision_reaudit_json": _safe_rel(out_json, root),
            "llr_precision_reaudit_csv": _safe_rel(out_csv, root),
            "llr_precision_reaudit_png": _safe_rel(out_png, root),
            "llr_systematics_root_cause_json": _safe_rel(root_cause_json, root),
            "llr_systematics_root_cause_csv": _safe_rel(root_cause_csv, root),
            "llr_systematics_root_cause_png": _safe_rel(root_cause_png, root),
            "llr_systematics_root_cause_over4ns_png": _safe_rel(root_cause_over4_png, root),
            "llr_bottleneck_deepdive_json": _safe_rel(deep_json, root),
            "llr_bottleneck_deepdive_csv": _safe_rel(deep_csv, root),
            "llr_bottleneck_deepdive_png": _safe_rel(deep_png, root),
        },
    }
    out_json.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(out_csv, checks, metrics_diag, points_diag)
    _write_plot(out_png, manifest_diag, metrics_diag, points_diag)
    root_cause_json.write_text(json.dumps(root_cause, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_root_cause_csv(root_cause_csv, root_cause)
    _write_root_cause_plot(root_cause_png, root_cause)
    _write_root_cause_over4_plot(root_cause_over4_png, root_cause)
    deep_json.write_text(json.dumps(deep, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_bottleneck_deepdive_csv(deep_csv, deep)
    _write_bottleneck_deepdive_plot(deep_png, deep)

    print(f"[ok] re-audit json: {out_json}")
    print(f"[ok] re-audit csv : {out_csv}")
    print(f"[ok] re-audit plot: {out_png}")
    print(f"[ok] root-cause json: {root_cause_json}")
    print(f"[ok] root-cause csv : {root_cause_csv}")
    print(f"[ok] root-cause plot: {root_cause_png}")
    print(f"[ok] root-cause >4ns plot: {root_cause_over4_png}")
    print(f"[ok] deep-dive json: {deep_json}")
    print(f"[ok] deep-dive csv : {deep_csv}")
    print(f"[ok] deep-dive plot: {deep_png}")
    print(f"[ok] overall={overall} decision={decision}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
