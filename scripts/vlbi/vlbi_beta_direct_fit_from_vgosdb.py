#!/usr/bin/env python3
"""
vlbi_beta_direct_fit_from_vgosdb.py

IVS vgosDb（一次データ）から Group Delay を読み込み、
P-model の光伝播係数 β を直接推定する。

方針:
- 観測量は vgosDb の GroupDelayFull（S/X）を一次データとして使う。
- S/X の ionosphere-free 合成を適用して分散性遅延の影響を抑制する。
- Calc 理論のうち太陽重力成分（Cal-BendSun）だけを切り出して
  P-model 係数 β で置換し、残差線形回帰で β を直接推定する。

モデル:
  τ_obs,IF,i = ionosphere_free(τ_X,i, τ_S,i; f_X,i, f_S,i)
  τ_base,i = τ_theo,i - τ_bendSun,GR,i
  (τ_obs,IF,i - τ_base,i) = c0 + β * τ_bendSun,GR,i + ε_i

入力:
- data/vlbi/sources/vgosdb/<session>/extracted 配下の .nc（netCDF）

出力:
- output/vlbi/vlbi_<session>_beta_direct_fit_points.csv
- output/vlbi/vlbi_<session>_beta_direct_fit_summary.csv
- output/vlbi/vlbi_<session>_beta_direct_fit_metrics.json
- output/vlbi/vlbi_<session>_beta_direct_fit.pdf
- output/vlbi/vlbi_<session>_beta_direct_fit.png
- 併せて output/public/vlbi/ に同期
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.io import netcdf_file

try:
    from netCDF4 import Dataset as _NC4Dataset  # type: ignore
except Exception:
    _NC4Dataset = None


OBS_CANDIDATES = [
    "GroupDelay",
    "GroupDelayFull",
]
OBS_FULL_CANDIDATES = [
    "GroupDelayFull",
]
OBS_FRINGE_CANDIDATES = [
    "GroupDelay",
]
THEO_CANDIDATES = [
    "DelayTheoretical",
    "DelayTheo",
]
BEND_CANDIDATES = [
    "Cal-BendSun",
    "Cal-Bend",
]
PART_CANDIDATES = [
    "Part-Gamma",
    "PartGamma",
    "GammaPart",
    "Part-Bend",
    "PartBend",
    "BendPart",
    "BendSunCal",
]
SIGMA_CANDIDATES = [
    "GroupDelayFullSig",
    "GroupDelaySig",
    "DelayTheoreticalSig",
]
FLAG_CANDIDATES = [
    "DelayFlag",
    "DelayDataFlag",
    "Edit",
    "QualityFlag",
]
SOURCE_CANDIDATES = [
    "Source",
]
BASELINE_CANDIDATES = [
    "Baseline",
]
YMDHM_CANDIDATES = [
    "YMDHM",
]
SECOND_CANDIDATES = [
    "Second",
]
FREQ_GROUP_CANDIDATES = [
    "FreqGroupIono",
]
IONO_CAL_CANDIDATES = [
    "Cal-SlantPathIonoGroup",
]
IONO_CAL_FLAG_CANDIDATES = [
    "Cal-SlantPathIonoGroupDataFlag",
]


# 関数: `_repo_root` の入出力契約と処理意図を定義する。
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# 関数: `_safe_float` の入出力契約と処理意図を定義する。

def _safe_float(value: object, default: float = math.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


# 関数: `_sha256` の入出力契約と処理意図を定義する。

def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


# 関数: `_scan_netcdf_variables` の入出力契約と処理意図を定義する。

def _list_variable_names(path: Path) -> List[str]:
    try:
        with netcdf_file(str(path), mode="r", mmap=False) as nc:
            return [str(k) for k in nc.variables.keys()]
    except Exception:
        if _NC4Dataset is None:
            raise

    with _NC4Dataset(str(path), mode="r") as nc:  # type: ignore[misc]
        return [str(k) for k in nc.variables.keys()]


# 関数: `_scan_netcdf_variables` の入出力契約と処理意図を定義する。

def _scan_netcdf_variables(root: Path) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {}
    for path in sorted(root.rglob("*.nc")):
        try:
            names = _list_variable_names(path)
        except Exception:
            continue

        for name in names:
            key = name.lower()
            if key not in index:
                index[key] = []

            index[key].append(path)

    return index


# 関数: `_pick_variable` の入出力契約と処理意図を定義する。

def _pick_variable(index: Dict[str, List[Path]], candidates: Sequence[str]) -> Tuple[Optional[str], Optional[Path]]:
    for cand in candidates:
        key = cand.lower()
        if key not in index:
            continue

        files = index[key]
        if not files:
            continue

        return cand, files[0]

    return None, None


# 関数: `_pick_variable_preferred` の入出力契約と処理意図を定義する。

def _pick_variable_preferred(
    index: Dict[str, List[Path]],
    candidates: Sequence[str],
    prefer_substring: str,
) -> Tuple[Optional[str], Optional[Path]]:
    pref = str(prefer_substring).strip().lower()
    for cand in candidates:
        key = cand.lower()
        if key not in index:
            continue

        files = index[key]
        if not files:
            continue

        if pref:
            for p in files:
                if pref in str(p).lower():
                    return cand, p

        return cand, files[0]

    return None, None


# 関数: `_read_variable` の入出力契約と処理意図を定義する。

def _read_variable(path: Path, var_name: str) -> np.ndarray:
    key = var_name.lower()
    try:
        with netcdf_file(str(path), mode="r", mmap=False) as nc:
            names = {str(k).lower(): str(k) for k in nc.variables.keys()}
            if key not in names:
                raise KeyError(f"variable not found: {var_name} in {path}")

            arr = np.array(nc.variables[names[key]][:])
            return arr
    except Exception:
        if _NC4Dataset is None:
            raise

    with _NC4Dataset(str(path), mode="r") as nc:  # type: ignore[misc]
        names = {str(k).lower(): str(k) for k in nc.variables.keys()}
        if key not in names:
            raise KeyError(f"variable not found: {var_name} in {path}")

        arr = np.array(nc.variables[names[key]][:])
        return arr


# 関数: `_reduce_to_vector_numeric` の入出力契約と処理意図を定義する。

def _reduce_to_vector_numeric(arr: np.ndarray, band_index: int) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 0:
        return a.reshape(1).astype(np.float64)

    if a.ndim == 1:
        return a.astype(np.float64)

    # 小さい次元を band 軸とみなして1本を選び、残りを観測軸として flatten する。

    axis_band = int(np.argmin(a.shape))
    if a.shape[axis_band] <= 32:
        b = np.moveaxis(a, axis_band, 0)
        idx = max(0, min(int(band_index), b.shape[0] - 1))
        return np.asarray(b[idx]).reshape(-1).astype(np.float64)

    return a.reshape(-1).astype(np.float64)


# 関数: `_reduce_to_vector_flag` の入出力契約と処理意図を定義する。

def _reduce_to_vector_flag(arr: np.ndarray, band_index: int) -> np.ndarray:
    a = np.asarray(arr)
    if a.dtype.kind in {"S", "U"}:
        if a.ndim == 0:
            return np.asarray([str(a.item())], dtype=object)

        if a.ndim == 1:
            return a.astype(object)

        axis_band = int(np.argmin(a.shape))
        if a.shape[axis_band] <= 32:
            b = np.moveaxis(a, axis_band, 0)
            idx = max(0, min(int(band_index), b.shape[0] - 1))
            return np.asarray(b[idx]).reshape(-1).astype(object)

        return np.asarray(a.reshape(-1), dtype=object)

    # 条件分岐: `a.ndim == 0` を満たす経路を評価する。

    if a.ndim == 0:
        return a.reshape(1)

    # 条件分岐: `a.ndim == 1` を満たす経路を評価する。

    if a.ndim == 1:
        return a

    axis_band = int(np.argmin(a.shape))
    if a.shape[axis_band] <= 32:
        b = np.moveaxis(a, axis_band, 0)
        idx = max(0, min(int(band_index), b.shape[0] - 1))
        return np.asarray(b[idx]).reshape(-1)

    return a.reshape(-1)


# 関数: `_align_common_length` の入出力契約と処理意図を定義する。

def _align_common_length(arrays: Sequence[np.ndarray]) -> int:
    lengths = [int(a.shape[0]) for a in arrays if a is not None and a.size > 0]
    if not lengths:
        return 0

    return int(min(lengths))


# 関数: `_decode_char_row` の入出力契約と処理意図を定義する。

def _decode_char_row(arr: np.ndarray) -> str:
    a = np.asarray(arr)
    if a.dtype.kind in {"S", "U"}:
        if a.ndim == 0:
            return str(a.item()).strip()

        if a.dtype.kind == "S":
            return b"".join(a.reshape(-1).tolist()).decode("ascii", "ignore").strip()

        return "".join([str(v) for v in a.reshape(-1).tolist()]).strip()

    return str(a.item()).strip() if a.ndim == 0 else str(a.reshape(-1)[0]).strip()


# 関数: `_read_baseline_vector` の入出力契約と処理意図を定義する。

def _read_baseline_vector(baseline_raw: np.ndarray) -> np.ndarray:
    a = np.asarray(baseline_raw)
    if a.ndim == 3 and a.shape[1] == 2:
        out: List[str] = []
        for row in a:
            s1 = _decode_char_row(np.asarray(row[0]))
            s2 = _decode_char_row(np.asarray(row[1]))
            out.append(f"{s1}-{s2}")

        return np.asarray(out, dtype=object)

    if a.ndim >= 2:
        out = [_decode_char_row(np.asarray(row)) for row in a]
        return np.asarray(out, dtype=object)

    if a.ndim == 1:
        out = [str(v).strip() for v in a.tolist()]
        return np.asarray(out, dtype=object)

    return np.asarray([str(a.item()).strip()], dtype=object)


# 関数: `_read_source_vector` の入出力契約と処理意図を定義する。

def _read_source_vector(source_raw: np.ndarray) -> np.ndarray:
    a = np.asarray(source_raw)
    if a.ndim == 2 and a.dtype.kind in {"S", "U"}:
        out = [_decode_char_row(np.asarray(row)) for row in a]
        return np.asarray(out, dtype=object)

    if a.ndim == 1 and a.dtype.kind in {"S", "U"}:
        out = [str(v.decode("ascii", "ignore") if isinstance(v, (bytes, bytearray)) else v).strip() for v in a.tolist()]
        return np.asarray(out, dtype=object)

    if a.ndim == 1:
        return np.asarray([str(v).strip() for v in a.tolist()], dtype=object)

    if a.ndim >= 2:
        out = [_decode_char_row(np.asarray(row)) for row in a]
        return np.asarray(out, dtype=object)

    return np.asarray([str(a.item()).strip()], dtype=object)


# 関数: `_parse_source_allowlist` の入出力契約と処理意図を定義する。

def _parse_source_allowlist(text: str) -> List[str]:
    tokens = [t.strip() for t in str(text).replace(";", ",").split(",")]
    return [t for t in tokens if t]


# 関数: `_compute_iono_free_group_delay` の入出力契約と処理意図を定義する。

def _compute_iono_free_group_delay(
    tau_x: np.ndarray,
    tau_s: np.ndarray,
    freq_x_mhz: np.ndarray,
    freq_s_mhz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    tx = np.asarray(tau_x, dtype=np.float64)
    ts = np.asarray(tau_s, dtype=np.float64)
    fx = np.asarray(freq_x_mhz, dtype=np.float64)
    fs = np.asarray(freq_s_mhz, dtype=np.float64)
    n = _align_common_length([tx, ts, fx, fs])
    out = np.full(n, np.nan, dtype=np.float64)
    if n <= 0:
        return out, np.zeros(0, dtype=bool)

    tx = tx[:n]
    ts = ts[:n]
    fx = fx[:n]
    fs = fs[:n]
    ax = np.square(fx)
    ass = np.square(fs)
    den = ax - ass
    valid = np.isfinite(tx) & np.isfinite(ts) & np.isfinite(fx) & np.isfinite(fs)
    valid &= (fx > 0.0) & (fs > 0.0) & (np.abs(den) > 1.0e-9)
    if np.any(valid):
        out[valid] = (ax[valid] * tx[valid] - ass[valid] * ts[valid]) / den[valid]

    return out, valid


# 関数: `_build_time_seconds` の入出力契約と処理意図を定義する。

def _build_time_seconds(ymdhm_raw: np.ndarray, second_raw: np.ndarray) -> Optional[np.ndarray]:
    ymdhm = np.asarray(ymdhm_raw)
    sec = np.asarray(second_raw, dtype=np.float64)
    if ymdhm.ndim != 2 or ymdhm.shape[1] < 5:
        return None

    if sec.ndim != 1:
        sec = sec.reshape(-1)

    n = int(min(ymdhm.shape[0], sec.shape[0]))
    if n < 3:
        return None

    y = ymdhm[:n, 0].astype(np.float64)
    mo = ymdhm[:n, 1].astype(np.float64)
    d = ymdhm[:n, 2].astype(np.float64)
    h = ymdhm[:n, 3].astype(np.float64)
    mi = ymdhm[:n, 4].astype(np.float64)
    s = sec[:n]
    # 単調な擬似UTC秒（相対時刻目的）。
    t = (((((y * 12.0) + mo) * 31.0 + d) * 24.0 + h) * 60.0 + mi) * 60.0 + s
    return t


# 関数: `_build_nuisance_matrix` の入出力契約と処理意図を定義する。

def _build_nuisance_matrix(
    index: Dict[str, List[Path]],
    n_common: int,
    keep_idx: np.ndarray,
    nuisance_mode: str,
) -> Tuple[Optional[np.ndarray], Dict[str, object]]:
    info: Dict[str, object] = {
        "requested_mode": nuisance_mode,
        "effective_mode": "none",
        "baseline_groups": 0,
        "nuisance_columns": 0,
    }
    if nuisance_mode == "none":
        return None, info

    bl_name, bl_file = _pick_variable(index, BASELINE_CANDIDATES)
    if bl_name is None or bl_file is None:
        info["reason"] = "baseline_variable_not_found"
        return None, info

    baseline_raw = _read_variable(bl_file, bl_name)
    baseline_vec = _read_baseline_vector(baseline_raw)
    if baseline_vec.size < n_common:
        info["reason"] = "baseline_length_shorter_than_observables"
        return None, info

    bl = baseline_vec[:n_common][keep_idx]
    if bl.size < 3:
        info["reason"] = "not_enough_points_for_baseline_nuisance"
        return None, info

    uniq_bl, ibl = np.unique(bl, return_inverse=True)
    n_bl = int(uniq_bl.size)
    info["baseline_groups"] = n_bl
    info["baseline_variable"] = {"name": bl_name, "file": str(bl_file)}
    if n_bl <= 1:
        info["reason"] = "single_baseline_only"
        return None, info

    n = int(bl.size)
    cols: List[np.ndarray] = []
    h_bl = np.zeros((n, n_bl - 1), dtype=np.float64)
    for j in range(1, n_bl):
        h_bl[:, j - 1] = (ibl == j).astype(np.float64)

    cols.append(h_bl)
    effective_mode = "baseline_intercept"

    if nuisance_mode == "baseline_intercept_linear":
        ymdhm_name, ymdhm_file = _pick_variable_preferred(index, YMDHM_CANDIDATES, prefer_substring="observables")
        if ymdhm_name is None or ymdhm_file is None:
            ymdhm_name, ymdhm_file = _pick_variable(index, YMDHM_CANDIDATES)

        sec_name, sec_file = _pick_variable_preferred(index, SECOND_CANDIDATES, prefer_substring="observables")
        if sec_name is None or sec_file is None:
            sec_name, sec_file = _pick_variable(index, SECOND_CANDIDATES)

        if ymdhm_name is not None and ymdhm_file is not None and sec_name is not None and sec_file is not None:
            ymdhm_raw = _read_variable(ymdhm_file, ymdhm_name)
            sec_raw = _read_variable(sec_file, sec_name)
            t = _build_time_seconds(ymdhm_raw, sec_raw)
            if t is not None and t.size >= n_common:
                tk = t[:n_common][keep_idx]
                tk = (tk - float(np.mean(tk))) / 3600.0
                h_sl = np.zeros((n, n_bl - 1), dtype=np.float64)
                for j in range(1, n_bl):
                    h_sl[:, j - 1] = ((ibl == j).astype(np.float64) * tk)

                cols.append(h_sl)
                effective_mode = "baseline_intercept_linear"
                info["time_variable"] = {
                    "ymdhm_name": ymdhm_name,
                    "ymdhm_file": str(ymdhm_file),
                    "second_name": sec_name,
                    "second_file": str(sec_file),
                }
            else:
                info["time_reason"] = "time_vector_unavailable_or_short"
        else:
            info["time_reason"] = "time_variables_not_found"

    z = np.concatenate(cols, axis=1) if cols else None
    info["effective_mode"] = effective_mode
    info["nuisance_columns"] = 0 if z is None else int(z.shape[1])
    return z, info


# 関数: `_weighted_linear_fit` の入出力契約と処理意図を定義する。

def _weighted_linear_fit(x: np.ndarray, y: np.ndarray, w: np.ndarray, z: Optional[np.ndarray] = None) -> Dict[str, float]:
    n = int(x.size)
    if n < 3:
        raise ValueError("not enough points after filtering")

    X = np.column_stack([np.ones(n, dtype=np.float64), x.astype(np.float64)])
    if z is not None:
        zz = np.asarray(z, dtype=np.float64)
        if zz.ndim == 1:
            zz = zz.reshape(-1, 1)

        if zz.shape[0] != n:
            raise ValueError("nuisance row count mismatch")

        if zz.shape[1] > 0:
            X = np.column_stack([X, zz])

    sw = np.sqrt(w.astype(np.float64))
    Xw = X * sw[:, None]
    yw = y.astype(np.float64) * sw
    theta, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    intercept = float(theta[0])
    slope = float(theta[1])
    yhat = X @ theta
    resid = y - yhat
    chi2 = float(np.sum(w * resid * resid))
    dof = int(max(1, n - X.shape[1]))
    s2 = chi2 / float(dof)
    xtwx = Xw.T @ Xw
    try:
        cov = np.linalg.inv(xtwx) * s2
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(xtwx) * s2

    intercept_sigma = float(math.sqrt(max(0.0, cov[0, 0])))
    slope_sigma = float(math.sqrt(max(0.0, cov[1, 1])))
    rmse = float(math.sqrt(np.mean(resid * resid)))
    wrmse = float(math.sqrt(np.average(resid * resid, weights=w)))
    corr = float(np.corrcoef(x, y)[0, 1]) if n > 2 else math.nan
    return {
        "intercept": intercept,
        "slope": slope,
        "intercept_sigma": intercept_sigma,
        "slope_sigma": slope_sigma,
        "chi2": chi2,
        "dof": float(dof),
        "rmse": rmse,
        "weighted_rmse": wrmse,
        "corr_x_y": corr,
        "n_params": float(X.shape[1]),
    }


# 関数: `_write_points_csv` の入出力契約と処理意図を定義する。

def _write_points_csv(
    path: Path,
    idx: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    yhat: np.ndarray,
    sigma: Optional[np.ndarray],
    flag: Optional[np.ndarray],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "obs_index",
                "gravity_template_s",
                "obs_minus_base_s",
                "fit_s",
                "obs_minus_base_minus_fit_s",
                "sigma_s",
                "flag",
            ]
        )
        for i in range(idx.size):
            sig = math.nan if sigma is None else float(sigma[i])
            flg = "" if flag is None else str(flag[i])
            w.writerow(
                [
                    int(idx[i]),
                    f"{float(x[i]):.16e}",
                    f"{float(y[i]):.16e}",
                    f"{float(yhat[i]):.16e}",
                    f"{float(y[i] - yhat[i]):.16e}",
                    "" if not math.isfinite(sig) else f"{sig:.16e}",
                    flg,
                ]
            )


# 関数: `_write_summary_csv` の入出力契約と処理意図を定義する。

def _write_summary_csv(path: Path, metrics: Dict[str, object]) -> None:
    fit = metrics.get("fit_result", {}) if isinstance(metrics.get("fit_result"), dict) else {}
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key", "value"])
        rows = [
            ("session", str(metrics.get("session") or "")),
            ("n_points", str(int(_safe_float(fit.get("n_points"), 0.0)))),
            ("n_params", str(int(_safe_float(fit.get("n_params"), 0.0)))),
            ("nuisance_mode", str(((fit.get("nuisance_info") or {}) if isinstance(fit.get("nuisance_info"), dict) else {}).get("effective_mode") or "")),
            ("intercept_s", f"{_safe_float(fit.get('intercept_s')):.16e}"),
            ("delta_gamma", f"{_safe_float(fit.get('delta_gamma')):.16e}"),
            ("delta_gamma_sigma", f"{_safe_float(fit.get('delta_gamma_sigma')):.16e}"),
            ("gamma_est", f"{_safe_float(fit.get('gamma_est')):.16e}"),
            ("gamma_sigma", f"{_safe_float(fit.get('gamma_sigma')):.16e}"),
            ("beta_est", f"{_safe_float(fit.get('beta_est')):.16e}"),
            ("beta_sigma", f"{_safe_float(fit.get('beta_sigma')):.16e}"),
            ("delta_beta", f"{_safe_float(fit.get('delta_beta')):.16e}"),
            ("chi2", f"{_safe_float(fit.get('chi2')):.16e}"),
            ("dof", str(int(_safe_float(fit.get("dof"), 0.0)))),
            ("rmse_s", f"{_safe_float(fit.get('rmse_s')):.16e}"),
            ("weighted_rmse_s", f"{_safe_float(fit.get('weighted_rmse_s')):.16e}"),
            ("corr_template_residual", f"{_safe_float(fit.get('corr_template_residual')):.16e}"),
        ]
        for k, v in rows:
            w.writerow([k, v])


# 関数: `_plot_result` の入出力契約と処理意図を定義する。

def _plot_result(
    pdf_path: Path,
    png_path: Path,
    x: np.ndarray,
    y: np.ndarray,
    yhat: np.ndarray,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    x_ps = x * 1.0e12
    y_ps = y * 1.0e12
    yhat_ps = yhat * 1.0e12
    resid_ps = (y - yhat) * 1.0e12
    order = np.argsort(x_ps)
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(11.0, 8.0), gridspec_kw={"height_ratios": [2.2, 1.0]})
    ax0.scatter(x_ps, y_ps, s=8, alpha=0.35, label="residual vs template")
    ax0.plot(x_ps[order], yhat_ps[order], color="tab:red", linewidth=2.0, label="weighted fit")
    ax0.set_xlabel(x_label)
    ax0.set_ylabel(y_label)
    ax0.set_title(title)
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="best")
    ax1.hist(resid_ps, bins=60, color="tab:gray", alpha=0.85)
    ax1.set_xlabel("Residual - fit [ps]")
    ax1.set_ylabel("Count")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(pdf_path))
    fig.savefig(str(png_path), dpi=170)
    plt.close(fig)


# 関数: `_sync_public` の入出力契約と処理意図を定義する。

def _sync_public(root: Path, outputs: Sequence[Path]) -> None:
    dst = root / "output" / "public" / "vlbi"
    dst.mkdir(parents=True, exist_ok=True)
    for path in outputs:
        if path.exists():
            shutil.copy2(path, dst / path.name)


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="Direct beta fit from IVS vgosDb group-delay observables.")
    ap.add_argument("--session", type=str, default="AUA020", help="Session code label used in output filenames.")
    ap.add_argument(
        "--input-root",
        type=Path,
        default=root / "data" / "vlbi" / "sources" / "vgosdb" / "AUA020" / "extracted",
        help="Extracted vgosDb root directory containing .nc files.",
    )
    ap.add_argument(
        "--observable-series",
        type=str,
        default="full",
        choices=["full", "fringe"],
        help="Use GroupDelayFull (full) or GroupDelay (fringe) as input observables.",
    )
    ap.add_argument("--band-index", type=int, default=0, help="Band index to select when arrays are 2D.")
    ap.add_argument(
        "--obs-band",
        type=str,
        default="X",
        choices=["auto", "S", "X", "s", "x"],
        help="Band used when --disable-iono-free is set.",
    )
    ap.add_argument(
        "--min-template-abs",
        type=float,
        default=1.0e-14,
        help="Absolute threshold on |Cal-BendSun template| [s] to reject near-zero template points.",
    )
    ap.add_argument(
        "--min-gamma-part-abs",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    ap.add_argument(
        "--disable-flag-filter",
        action="store_true",
        help="If set, do not enforce DelayDataFlag==0 filtering.",
    )
    ap.add_argument(
        "--uniform-weight",
        action="store_true",
        help="If set, ignore sigma and use uniform weights.",
    )
    ap.add_argument(
        "--nuisance-mode",
        type=str,
        default="baseline_intercept",
        choices=["none", "baseline_intercept", "baseline_intercept_linear"],
        help="Nuisance model to absorb session-level delay offsets before estimating beta.",
    )
    ap.add_argument(
        "--disable-iono-free",
        action="store_true",
        help="If set, skip S/X ionosphere-free combination and fit a single selected band.",
    )
    ap.add_argument(
        "--source-include",
        type=str,
        default="",
        help="Comma-separated source allowlist (e.g. 0229+131,0235+164). Empty means all sources.",
    )
    args = ap.parse_args()

    session = str(args.session).strip()
    session_slug = "".join(ch if ch.isalnum() else "_" for ch in session).lower() or "session"
    input_root = args.input_root.resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"input root not found: {input_root}")

    index = _scan_netcdf_variables(input_root)
    if not index:
        raise RuntimeError(f"no netCDF files found under: {input_root}")

    threshold = float(args.min_template_abs)
    if args.min_gamma_part_abs is not None:
        threshold = float(args.min_gamma_part_abs)

    obs_candidates = OBS_FULL_CANDIDATES if str(args.observable_series) == "full" else OBS_FRINGE_CANDIDATES
    obs_x_name, obs_x_file = _pick_variable_preferred(index, obs_candidates, prefer_substring="_bx")
    obs_s_name, obs_s_file = _pick_variable_preferred(index, obs_candidates, prefer_substring="_bs")
    theo_name, theo_file = _pick_variable(index, THEO_CANDIDATES)
    bend_name, bend_file = _pick_variable(index, BEND_CANDIDATES)
    if obs_x_name is None or obs_x_file is None:
        raise RuntimeError(f"X-band observable variable not found. tried: {obs_candidates}")

    if theo_name is None or theo_file is None:
        raise RuntimeError(f"theoretical variable not found. tried: {THEO_CANDIDATES}")

    if bend_name is None or bend_file is None:
        raise RuntimeError(f"gravity template variable not found. tried: {BEND_CANDIDATES}")

    sig_x_name, sig_x_file = _pick_variable_preferred(index, SIGMA_CANDIDATES, prefer_substring="_bx")
    sig_s_name, sig_s_file = _pick_variable_preferred(index, SIGMA_CANDIDATES, prefer_substring="_bs")
    flag_name, flag_file = _pick_variable(index, FLAG_CANDIDATES)
    source_name, source_file = _pick_variable(index, SOURCE_CANDIDATES)

    freq_x_name, freq_x_file = _pick_variable_preferred(index, FREQ_GROUP_CANDIDATES, prefer_substring="_bx")
    freq_s_name, freq_s_file = _pick_variable_preferred(index, FREQ_GROUP_CANDIDATES, prefer_substring="_bs")

    obs_x_raw = _read_variable(obs_x_file, obs_x_name)
    obs_s_raw = _read_variable(obs_s_file, obs_s_name) if obs_s_name is not None and obs_s_file is not None else None
    theo_raw = _read_variable(theo_file, theo_name)
    bend_raw = _read_variable(bend_file, bend_name)
    sig_x_raw = _read_variable(sig_x_file, sig_x_name) if sig_x_name is not None and sig_x_file is not None else None
    sig_s_raw = _read_variable(sig_s_file, sig_s_name) if sig_s_name is not None and sig_s_file is not None else None
    flag_raw = _read_variable(flag_file, flag_name) if flag_name is not None and flag_file is not None else None
    source_raw = _read_variable(source_file, source_name) if source_name is not None and source_file is not None else None
    freq_x_raw = _read_variable(freq_x_file, freq_x_name) if freq_x_name is not None and freq_x_file is not None else None
    freq_s_raw = _read_variable(freq_s_file, freq_s_name) if freq_s_name is not None and freq_s_file is not None else None

    obs_x = _reduce_to_vector_numeric(obs_x_raw, int(args.band_index))
    obs_s = _reduce_to_vector_numeric(obs_s_raw, int(args.band_index)) if obs_s_raw is not None else None
    theo = _reduce_to_vector_numeric(theo_raw, int(args.band_index))
    bend = _reduce_to_vector_numeric(bend_raw, int(args.band_index))
    sigma_x = _reduce_to_vector_numeric(sig_x_raw, int(args.band_index)) if sig_x_raw is not None else None
    sigma_s = _reduce_to_vector_numeric(sig_s_raw, int(args.band_index)) if sig_s_raw is not None else None
    flag = _reduce_to_vector_flag(flag_raw, int(args.band_index)) if flag_raw is not None else None
    source_vec = _read_source_vector(source_raw) if source_raw is not None else None
    freq_x = _reduce_to_vector_numeric(freq_x_raw, int(args.band_index)) if freq_x_raw is not None else None
    freq_s = _reduce_to_vector_numeric(freq_s_raw, int(args.band_index)) if freq_s_raw is not None else None

    arrays_for_n: List[np.ndarray] = [obs_x, theo, bend]
    if obs_s is not None:
        arrays_for_n.append(obs_s)

    if flag is not None:
        arrays_for_n.append(flag)

    if source_vec is not None:
        arrays_for_n.append(source_vec)

    if freq_x is not None:
        arrays_for_n.append(freq_x)

    if freq_s is not None:
        arrays_for_n.append(freq_s)

    if sigma_x is not None:
        arrays_for_n.append(sigma_x)

    if sigma_s is not None:
        arrays_for_n.append(sigma_s)

    n_common = _align_common_length(arrays_for_n)
    if n_common < 3:
        raise RuntimeError(f"not enough aligned observations: {n_common}")

    obs_x = obs_x[:n_common]
    if obs_s is not None:
        obs_s = obs_s[:n_common]

    theo = theo[:n_common]
    bend = bend[:n_common]
    if sigma_x is not None:
        sigma_x = sigma_x[:n_common]

    if sigma_s is not None:
        sigma_s = sigma_s[:n_common]

    if freq_x is not None:
        freq_x = freq_x[:n_common]

    if freq_s is not None:
        freq_s = freq_s[:n_common]

    if source_vec is not None:
        source_vec = source_vec[:n_common]

    if flag is not None:
        flag = flag[:n_common]

    use_iono_free = not bool(args.disable_iono_free)
    obs_band = str(args.obs_band).upper()
    if obs_band not in {"S", "X"}:
        obs_band = "X"

    if use_iono_free:
        if obs_s is None:
            raise RuntimeError("S-band observable not found; cannot build ionosphere-free observable.")

        if freq_x is None or freq_s is None:
            raise RuntimeError("effective frequency vectors (FreqGroupIono) are required for ionosphere-free mode.")

        obs_used, iono_valid = _compute_iono_free_group_delay(
            tau_x=obs_x,
            tau_s=obs_s,
            freq_x_mhz=freq_x,
            freq_s_mhz=freq_s,
        )
        sigma_all = None
        if sigma_x is not None and sigma_s is not None:
            ax = np.square(freq_x)
            ass = np.square(freq_s)
            den = ax - ass
            sigma_all = np.full(n_common, np.nan, dtype=np.float64)
            ok = np.isfinite(ax) & np.isfinite(ass) & (np.abs(den) > 1.0e-9)
            ok &= np.isfinite(sigma_x) & np.isfinite(sigma_s) & (sigma_x > 0.0) & (sigma_s > 0.0)
            if np.any(ok):
                cx = ax[ok] / den[ok]
                cs = ass[ok] / den[ok]
                sigma_all[ok] = np.sqrt(np.square(cx * sigma_x[ok]) + np.square(cs * sigma_s[ok]))
    else:
        if obs_band == "S":
            if obs_s is None:
                raise RuntimeError("requested S band, but S-band observable not found.")

            obs_used = np.asarray(obs_s, dtype=np.float64)
            sigma_all = None if sigma_s is None else np.asarray(sigma_s, dtype=np.float64)
        else:
            obs_used = np.asarray(obs_x, dtype=np.float64)
            sigma_all = None if sigma_x is None else np.asarray(sigma_x, dtype=np.float64)

        iono_valid = np.isfinite(obs_used)

    tau_base = theo - bend
    obs_minus_base = obs_used - tau_base
    template = bend
    mask = np.isfinite(obs_minus_base) & np.isfinite(template) & np.asarray(iono_valid, dtype=bool)
    mask &= np.abs(template) >= float(threshold)

    if sigma_all is not None and not bool(args.uniform_weight):
        mask &= np.isfinite(sigma_all) & (sigma_all > 0.0)

    if flag is not None and not bool(args.disable_flag_filter):
        if np.issubdtype(flag.dtype, np.number):
            mask &= np.asarray(flag == 0)
        else:
            txt = np.asarray([str(v).strip() for v in flag], dtype=object)
            mask &= np.asarray([(v == "" or v == "0") for v in txt], dtype=bool)

    source_allow = _parse_source_allowlist(str(args.source_include))
    source_filter_applied = len(source_allow) > 0
    if source_filter_applied:
        if source_vec is None:
            raise RuntimeError("source filter requested but Source variable not found in vgosDb.")

        allowed = set(source_allow)
        source_mask = np.asarray([str(s) in allowed for s in source_vec], dtype=bool)
        mask &= source_mask

    keep_idx = np.where(mask)[0]
    if keep_idx.size < 3:
        raise RuntimeError("too few points after filtering")

    x = template[keep_idx]
    y = obs_minus_base[keep_idx]
    if sigma_all is None or bool(args.uniform_weight):
        w = np.ones_like(y, dtype=np.float64)
        sigma_used = None
    else:
        sigma_used = sigma_all[keep_idx]
        w = 1.0 / np.square(sigma_used)

    z, nuisance_info = _build_nuisance_matrix(
        index=index,
        n_common=n_common,
        keep_idx=keep_idx,
        nuisance_mode=str(args.nuisance_mode),
    )
    fit = _weighted_linear_fit(x=x, y=y, w=w, z=z)
    intercept = fit["intercept"]
    beta_est = fit["slope"]
    beta_sigma = fit["slope_sigma"]
    delta_beta = beta_est - 1.0
    gamma_est = (2.0 * beta_est) - 1.0
    gamma_sigma = 2.0 * beta_sigma
    delta_gamma = gamma_est - 1.0
    delta_gamma_sigma = gamma_sigma
    if z is None or z.size == 0:
        yhat = intercept + beta_est * x
    else:
        zz = np.asarray(z, dtype=np.float64)
        if zz.ndim == 1:
            zz = zz.reshape(-1, 1)

        Xtmp = np.column_stack([np.ones_like(x, dtype=np.float64), x.astype(np.float64), zz])
        sw = np.sqrt(w.astype(np.float64))
        theta, _, _, _ = np.linalg.lstsq(Xtmp * sw[:, None], y.astype(np.float64) * sw, rcond=None)
        yhat = Xtmp @ theta

    iono_audit: Dict[str, object] = {
        "enabled": bool(use_iono_free),
        "status": "not_run" if not use_iono_free else "unavailable",
    }
    if use_iono_free:
        iono_audit.update(
            {
                "formula": "(fX^2*tauX - fS^2*tauS)/(fX^2 - fS^2)",
                "n_valid_formula": int(np.sum(iono_valid)),
            }
        )
        iono_cal_name, iono_cal_file = _pick_variable_preferred(index, IONO_CAL_CANDIDATES, prefer_substring="_bx")
        iono_cal_flag_name, iono_cal_flag_file = _pick_variable_preferred(index, IONO_CAL_FLAG_CANDIDATES, prefer_substring="_bx")
        if iono_cal_name is not None and iono_cal_file is not None:
            iono_cal = _reduce_to_vector_numeric(_read_variable(iono_cal_file, iono_cal_name), int(args.band_index))
            iono_cal = iono_cal[:n_common]
            delta_if_minus_x = obs_used - obs_x
            cmp_mask = np.isfinite(delta_if_minus_x) & np.isfinite(iono_cal)
            iono_flag = None
            if iono_cal_flag_name is not None and iono_cal_flag_file is not None:
                iono_flag = _reduce_to_vector_flag(_read_variable(iono_cal_flag_file, iono_cal_flag_name), int(args.band_index))
                iono_flag = iono_flag[:n_common]
                if np.issubdtype(iono_flag.dtype, np.number):
                    cmp_mask &= np.asarray(iono_flag == 0)
                else:
                    txt = np.asarray([str(v).strip() for v in iono_flag], dtype=object)
                    cmp_mask &= np.asarray([(v == "" or v == "0") for v in txt], dtype=bool)

            if np.sum(cmp_mask) >= 3:
                chk = delta_if_minus_x[cmp_mask] + iono_cal[cmp_mask]
                corr = float(np.corrcoef(delta_if_minus_x[cmp_mask], -iono_cal[cmp_mask])[0, 1])
                iono_audit.update(
                    {
                        "status": "ok",
                        "iono_correction_variable": {"name": iono_cal_name, "file": str(iono_cal_file)},
                        "n_compared": int(np.sum(cmp_mask)),
                        "corr_delta_vs_minus_cal": corr,
                        "mean_check_s": float(np.mean(chk)),
                        "std_check_s": float(np.std(chk)),
                        "max_abs_check_s": float(np.max(np.abs(chk))),
                    }
                )
            else:
                iono_audit.update({"status": "insufficient_comparison_points", "n_compared": int(np.sum(cmp_mask))})

    out_dir = root / "output" / "vlbi"
    out_dir.mkdir(parents=True, exist_ok=True)
    points_csv = out_dir / f"vlbi_{session_slug}_beta_direct_fit_points.csv"
    summary_csv = out_dir / f"vlbi_{session_slug}_beta_direct_fit_summary.csv"
    metrics_json = out_dir / f"vlbi_{session_slug}_beta_direct_fit_metrics.json"
    plot_pdf = out_dir / f"vlbi_{session_slug}_beta_direct_fit.pdf"
    plot_png = out_dir / f"vlbi_{session_slug}_beta_direct_fit.png"

    _write_points_csv(
        path=points_csv,
        idx=keep_idx.astype(np.int64),
        x=x,
        y=y,
        yhat=yhat,
        sigma=sigma_used,
        flag=None if flag is None else flag[keep_idx],
    )

    metrics: Dict[str, object] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "session": session,
        "method": {
            "description": "direct fit from vgosDb observables using BendSun substitution (no PPN input)",
            "equation": "ObsMinusBase = c0 + beta * BendSun_GR, where Base = DelayTheo - BendSun_GR",
            "ionosphere_handling": "ionosphere-free combination from GroupDelayFull_bX/bS unless disabled",
            "gr_add_back": "not_applied (observables are not O-C residual products)",
        },
        "input": {
            "root": str(input_root),
            "observable_series": str(args.observable_series),
            "observable_x_variable": {"name": obs_x_name, "file": str(obs_x_file), "sha256": _sha256(obs_x_file)},
            "observable_s_variable": None
            if obs_s_name is None or obs_s_file is None
            else {"name": obs_s_name, "file": str(obs_s_file), "sha256": _sha256(obs_s_file)},
            "theoretical_variable": {"name": theo_name, "file": str(theo_file), "sha256": _sha256(theo_file)},
            "gravity_template_variable": {"name": bend_name, "file": str(bend_file), "sha256": _sha256(bend_file)},
            "sigma_x_variable": None
            if sig_x_name is None or sig_x_file is None
            else {"name": sig_x_name, "file": str(sig_x_file), "sha256": _sha256(sig_x_file)},
            "sigma_s_variable": None
            if sig_s_name is None or sig_s_file is None
            else {"name": sig_s_name, "file": str(sig_s_file), "sha256": _sha256(sig_s_file)},
            "flag_variable": None
            if flag_name is None or flag_file is None
            else {"name": flag_name, "file": str(flag_file), "sha256": _sha256(flag_file)},
            "source_variable": None
            if source_name is None or source_file is None
            else {"name": source_name, "file": str(source_file), "sha256": _sha256(source_file)},
            "freq_group_x_variable": None
            if freq_x_name is None or freq_x_file is None
            else {"name": freq_x_name, "file": str(freq_x_file), "sha256": _sha256(freq_x_file)},
            "freq_group_s_variable": None
            if freq_s_name is None or freq_s_file is None
            else {"name": freq_s_name, "file": str(freq_s_file), "sha256": _sha256(freq_s_file)},
            "band_index": int(args.band_index),
            "obs_band_when_single_band": str(obs_band),
            "min_template_abs_s": float(threshold),
            "flag_filter_enabled": not bool(args.disable_flag_filter),
            "sigma_weight_enabled": not bool(args.uniform_weight),
            "nuisance_mode": str(args.nuisance_mode),
            "ionosphere_free_enabled": bool(use_iono_free),
            "source_filter": source_allow,
        },
        "counts": {
            "n_common_raw": int(n_common),
            "n_kept": int(keep_idx.size),
            "n_rejected": int(n_common - keep_idx.size),
            "n_sources_in_filter": int(np.sum(np.asarray([str(s) in set(source_allow) for s in source_vec], dtype=bool)))
            if source_filter_applied and source_vec is not None
            else None,
        },
        "fit_result": {
            "n_points": int(keep_idx.size),
            "intercept_s": float(intercept),
            "intercept_sigma_s": float(fit["intercept_sigma"]),
            "delta_gamma": float(delta_gamma),
            "delta_gamma_sigma": float(delta_gamma_sigma),
            "gamma_est": float(gamma_est),
            "gamma_sigma": float(gamma_sigma),
            "beta_est": float(beta_est),
            "beta_sigma": float(beta_sigma),
            "delta_beta": float(delta_beta),
            "chi2": float(fit["chi2"]),
            "dof": int(_safe_float(fit["dof"], 0.0)),
            "rmse_s": float(fit["rmse"]),
            "weighted_rmse_s": float(fit["weighted_rmse"]),
            "corr_template_residual": float(fit["corr_x_y"]),
            "n_params": int(_safe_float(fit.get("n_params"), 0.0)),
            "nuisance_info": nuisance_info,
        },
        "ionosphere_audit": iono_audit,
        "outputs": {
            "points_csv": str(points_csv),
            "summary_csv": str(summary_csv),
            "plot_pdf": str(plot_pdf),
            "plot_png": str(plot_png),
        },
    }
    _write_summary_csv(summary_csv, metrics)
    _plot_result(
        pdf_path=plot_pdf,
        png_path=plot_png,
        x=x,
        y=y,
        yhat=yhat,
        title=f"VLBI {session} direct beta fit (vgosDb primary data)",
        x_label="Cal-BendSun template [ps]",
        y_label=("Obs(IF) - Base [ps]" if use_iono_free else "Obs - Base [ps]"),
    )
    metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _sync_public(root, [points_csv, summary_csv, metrics_json, plot_pdf, plot_png])
    print("Wrote:", points_csv)
    print("Wrote:", summary_csv)
    print("Wrote:", metrics_json)
    print("Wrote:", plot_pdf)
    print("Wrote:", plot_png)
    print("Synced:", root / "output" / "public" / "vlbi")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
