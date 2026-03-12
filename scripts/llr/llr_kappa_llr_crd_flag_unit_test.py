#!/usr/bin/env python3
"""
llr_kappa_llr_crd_flag_unit_test.py

Step 8.7.47.16:
- LLR improvement roadmap Step 5 (CRD flag unit test).
- Audit range-type / system-delay / COM / refraction flag sensitivity and
  CRD-vs-MeritII quality inversion sensitivity on kappa fit.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]


# 関数: `_safe_rel` の入出力契約と処理意図を定義する。
def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve()).replace("\\", "/")


# 関数: `_combine_status` の入出力契約と処理意図を定義する。

def _combine_status(values: Iterable[str]) -> str:
    norm_all = [str(v or "").strip().lower() for v in values if str(v or "").strip()]
    norm = [v for v in norm_all if v in {"pass", "watch", "reject"}]
    if not norm:
        return "watch"

    if any(v == "reject" for v in norm):
        return "reject"

    if any(v == "watch" for v in norm):
        return "watch"

    return "pass"


# 関数: `_to_int_or_nan` の入出力契約と処理意図を定義する。

def _to_int_or_nan(tok: str) -> float:
    text = str(tok or "").strip()
    if text.lower() in ("", "na", "nan"):
        return float("nan")

    try:
        return float(int(text))
    except Exception:
        return float("nan")


# 関数: `_to_float_or_nan` の入出力契約と処理意図を定義する。

def _to_float_or_nan(tok: str) -> float:
    text = str(tok or "").strip()
    if text.lower() in ("", "na", "nan"):
        return float("nan")

    try:
        return float(text)
    except Exception:
        return float("nan")


# 関数: `_load_core_module` の入出力契約と処理意図を定義する。

def _load_core_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("llr_kappa_llr_core_flag_unit", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load core module spec: {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# 関数: `_build_cluster_ids` の入出力契約と処理意図を定義する。

def _build_cluster_ids(df: pd.DataFrame) -> np.ndarray:
    epoch = (
        df["epoch_utc"]
        if pd.api.types.is_datetime64_any_dtype(df["epoch_utc"])
        else pd.to_datetime(df["epoch_utc"], utc=True, errors="coerce")
    )
    night = epoch.dt.strftime("%Y-%m-%d").fillna("NA")
    return (df["station"].astype(str) + "|" + df["target"].astype(str) + "|" + night.astype(str)).to_numpy(dtype=object)


# 関数: `_fit_weighted_beta` の入出力契約と処理意図を定義する。

def _fit_weighted_beta(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    w = np.ones(len(y), dtype=float) if sample_weight is None else np.asarray(sample_weight, dtype=float).reshape(-1)
    w = np.where(np.isfinite(w) & (w > 0.0), w, np.nan)
    ok = np.isfinite(w)
    if not np.any(ok):
        raise ValueError("all weights invalid")

    w = np.where(ok, w / float(np.nanmean(w[ok])), 1.0)
    sw = np.sqrt(w)
    x_fit = x * sw[:, None]
    y_fit = y * sw
    beta_hat, _, _, _ = np.linalg.lstsq(x_fit, y_fit, rcond=None)
    resid_fit = y_fit - (x_fit @ beta_hat)
    return beta_hat, resid_fit, x_fit


# 関数: `_sandwich_slope_sigma` の入出力契約と処理意図を定義する。

def _sandwich_slope_sigma(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray],
    cluster_ids: np.ndarray,
) -> float:
    _, resid_fit, x_fit = _fit_weighted_beta(x=x, y=y, sample_weight=sample_weight)
    n = int(x_fit.shape[0])
    k = int(x_fit.shape[1])
    if n <= k:
        return float("nan")

    cluster = np.asarray(cluster_ids, dtype=object).reshape(-1)
    if len(cluster) != n:
        return float("nan")

    keys = pd.Series(cluster).dropna().astype(str).unique().tolist()
    g = int(len(keys))
    if g <= 1:
        return float("nan")

    xtx_inv = np.linalg.pinv(x_fit.T @ x_fit)
    meat = np.zeros((k, k), dtype=float)
    for key in keys:
        mask = cluster.astype(str) == str(key)
        xg = x_fit[mask, :]
        eg = resid_fit[mask]
        ug = xg.T @ eg
        meat += np.outer(ug, ug)

    cov = xtx_inv @ meat @ xtx_inv
    cov *= (g / max(g - 1, 1)) * ((n - 1) / max(n - k, 1))
    var0 = float(cov[0, 0])
    return float(math.sqrt(var0)) if np.isfinite(var0) and var0 >= 0.0 else float("nan")


# 関数: `_jackknife_kappa_sigma` の入出力契約と処理意図を定義する。

def _jackknife_kappa_sigma(
    x: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray],
    cluster_ids: np.ndarray,
) -> float:
    cluster = np.asarray(cluster_ids, dtype=object).reshape(-1)
    keys = pd.Series(cluster).dropna().astype(str).unique().tolist()
    if len(keys) <= 1:
        return float("nan")

    vals: List[float] = []
    for key in keys:
        keep = cluster.astype(str) != str(key)
        if int(np.sum(keep)) <= int(x.shape[1]) + 1:
            continue

        x_sub = x[keep, :]
        y_sub = y[keep]
        w_sub = None if sample_weight is None else np.asarray(sample_weight, dtype=float).reshape(-1)[keep]
        try:
            beta_hat, _, _ = _fit_weighted_beta(x=x_sub, y=y_sub, sample_weight=w_sub)
        except Exception:
            continue

        vals.append(float(1.0 + beta_hat[0]))

    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return float("nan")

    mu = float(np.mean(arr))
    var = float((len(arr) - 1) / len(arr) * np.sum(np.square(arr - mu)))
    return float(math.sqrt(var)) if np.isfinite(var) and var >= 0.0 else float("nan")


# 関数: `_fit_with_cluster_sigma` の入出力契約と処理意図を定義する。

def _fit_with_cluster_sigma(
    core: Any,
    df_sub: pd.DataFrame,
    *,
    mode: str,
    sample_weight: Optional[np.ndarray],
    cluster_ids: np.ndarray,
) -> Optional[Dict[str, Any]]:
    if df_sub.empty:
        return None

    try:
        x, y, _ = core._build_design_matrix(df_sub, mode=mode)
        fr = core._fit_ols(mode=mode, x=x, y=y, sample_weight=sample_weight)
    except Exception:
        return None

    sigma_sand = _sandwich_slope_sigma(x=x, y=y, sample_weight=sample_weight, cluster_ids=cluster_ids)
    sigma_jack = _jackknife_kappa_sigma(x=x, y=y, sample_weight=sample_weight, cluster_ids=cluster_ids)
    sigmas = [float(fr.kappa_sigma)]
    if np.isfinite(sigma_sand) and sigma_sand > 0.0:
        sigmas.append(float(sigma_sand))

    if np.isfinite(sigma_jack) and sigma_jack > 0.0:
        sigmas.append(float(sigma_jack))

    sigma_cluster = float(np.nanmax(np.asarray(sigmas, dtype=float)))
    abs_z = float(abs(fr.kappa_est - 1.0) / sigma_cluster) if np.isfinite(sigma_cluster) and sigma_cluster > 0.0 else float("nan")
    return {
        "kappa_est": float(fr.kappa_est),
        "kappa_sigma_cluster": float(sigma_cluster),
        "kappa_sigma_indep": float(fr.kappa_sigma),
        "kappa_sigma_sandwich": float(sigma_sand),
        "kappa_sigma_jackknife": float(sigma_jack),
        "abs_z_cluster": abs_z,
        "status_cluster": core._status_from_abs_z(abs_z),
        "n_points": int(fr.n_points),
    }


# 関数: `_extract_flags_for_points` の入出力契約と処理意図を定義する。

def _extract_flags_for_points(root: Path, points_df: pd.DataFrame) -> pd.DataFrame:
    need_keys: Set[Tuple[str, int]] = set()
    for rec in points_df[["source_file", "lineno"]].dropna().to_dict(orient="records"):
        need_keys.add((str(rec["source_file"]), int(rec["lineno"])))

    rows: List[Dict[str, Any]] = []
    for source_file in sorted(points_df["source_file"].dropna().astype(str).unique().tolist()):
        src = (root / source_file).resolve()
        if not src.exists():
            continue

        h4_vals: List[float] = [float("nan")] * 8
        try:
            with src.open("r", encoding="utf-8", errors="ignore") as fh:
                for lineno, raw in enumerate(fh, start=1):
                    line = raw.strip()
                    if not line:
                        continue

                    toks = line.split()
                    rec = str(toks[0]).lower()
                    if rec == "h4":
                        # H4 trailing flags in CRD:
                        # [data_release, refraction, COM, amp, system_delay, transponder, range_type, quality_alert]
                        h4_vals = [
                            _to_int_or_nan(toks[idx]) if len(toks) > idx else float("nan")
                            for idx in (14, 15, 16, 17, 18, 19, 20, 21)
                        ]
                        continue

                    if rec != "11":
                        continue

                    key = (source_file, int(lineno))
                    if key not in need_keys:
                        continue

                    np_bin_rms = float("nan")
                    if len(toks) > 8:
                        np_bin_rms = _to_float_or_nan(toks[8])

                    np_last_int = float("nan")
                    if len(toks) > 1:
                        np_last_int = _to_int_or_nan(toks[-1])

                    rows.append(
                        {
                            "source_file": source_file,
                            "lineno": int(lineno),
                            "h4_data_release": float(h4_vals[0]),
                            "h4_refraction_applied": float(h4_vals[1]),
                            "h4_com_applied": float(h4_vals[2]),
                            "h4_amplitude_applied": float(h4_vals[3]),
                            "h4_system_delay_applied": float(h4_vals[4]),
                            "h4_transponder_applied": float(h4_vals[5]),
                            "h4_range_type": float(h4_vals[6]),
                            "h4_quality_alert": float(h4_vals[7]),
                            "np_bin_rms": np_bin_rms,
                            "np_last_int": np_last_int,
                        }
                    )
        except Exception:
            continue

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.drop_duplicates(subset=["source_file", "lineno"], keep="first").reset_index(drop=True)
    return out


# 関数: `_split_fit_test` の入出力契約と処理意図を定義する。

def _split_fit_test(
    core: Any,
    df: pd.DataFrame,
    *,
    fit_mode: str,
    sample_weight: np.ndarray,
    cluster_ids: np.ndarray,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    min_points_group: int,
    test_id: str,
    test_label: str,
    note: str,
) -> Dict[str, Any]:
    n_a = int(np.sum(mask_a))
    n_b = int(np.sum(mask_b))
    row: Dict[str, Any] = {
        "test_id": test_id,
        "test_label": test_label,
        "n_a": n_a,
        "n_b": n_b,
        "fit_mode": fit_mode,
        "note": note,
        "fit_ok": False,
        "status": "reject",
        "reason": "",
        "kappa_a": float("nan"),
        "sigma_a": float("nan"),
        "kappa_b": float("nan"),
        "sigma_b": float("nan"),
        "delta_a_minus_b": float("nan"),
        "sigma_delta": float("nan"),
        "z_delta": float("nan"),
        "abs_z_delta": float("nan"),
    }
    if n_a < int(min_points_group) or n_b < int(min_points_group):
        row["status"] = "not_testable"
        row["reason"] = f"insufficient_contrast(min={int(min_points_group)})"
        return row

    idx_a = np.flatnonzero(mask_a)
    idx_b = np.flatnonzero(mask_b)
    fit_a = _fit_with_cluster_sigma(
        core=core,
        df_sub=df.loc[mask_a].copy().reset_index(drop=True),
        mode=fit_mode,
        sample_weight=np.asarray(sample_weight, dtype=float)[idx_a],
        cluster_ids=np.asarray(cluster_ids, dtype=object)[idx_a],
    )
    fit_b = _fit_with_cluster_sigma(
        core=core,
        df_sub=df.loc[mask_b].copy().reset_index(drop=True),
        mode=fit_mode,
        sample_weight=np.asarray(sample_weight, dtype=float)[idx_b],
        cluster_ids=np.asarray(cluster_ids, dtype=object)[idx_b],
    )
    if fit_a is None or fit_b is None:
        row["status"] = "reject"
        row["reason"] = "fit_failed"
        return row

    k_a = float(fit_a["kappa_est"])
    s_a = float(fit_a["kappa_sigma_cluster"])
    k_b = float(fit_b["kappa_est"])
    s_b = float(fit_b["kappa_sigma_cluster"])
    delta = float(k_a - k_b)
    s_delta = float(math.sqrt(max((s_a * s_a) + (s_b * s_b), 1e-30)))
    z = float(delta / s_delta) if np.isfinite(s_delta) and s_delta > 0.0 else float("nan")
    abs_z = float(abs(z)) if np.isfinite(z) else float("nan")
    status = core._status_from_abs_z(abs_z)
    row.update(
        {
            "fit_ok": True,
            "status": status,
            "reason": "",
            "kappa_a": k_a,
            "sigma_a": s_a,
            "kappa_b": k_b,
            "sigma_b": s_b,
            "delta_a_minus_b": delta,
            "sigma_delta": s_delta,
            "z_delta": z,
            "abs_z_delta": abs_z,
        }
    )
    return row


# 関数: `_write_plot` の入出力契約と処理意図を定義する。

def _write_plot(summary_df: pd.DataFrame, overall_status: str, out_pdf: Path, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.0))
    ax0, ax1, ax2 = axes
    s = summary_df.copy()
    s["idx"] = np.arange(len(s))
    labels = s["test_label"].astype(str).tolist()
    n_a = pd.to_numeric(s["n_a"], errors="coerce").to_numpy(dtype=float)
    n_b = pd.to_numeric(s["n_b"], errors="coerce").to_numpy(dtype=float)
    z = pd.to_numeric(s["abs_z_delta"], errors="coerce").to_numpy(dtype=float)
    status = s["status"].astype(str).str.lower().tolist()

    w = 0.38
    ax0.bar(s["idx"] - w / 2.0, n_a, width=w, color="#1f77b4", label="A")
    ax0.bar(s["idx"] + w / 2.0, n_b, width=w, color="#ff7f0e", label="B")
    ax0.set_xticks(s["idx"].to_numpy(dtype=float))
    ax0.set_xticklabels(labels, rotation=28, ha="right")
    ax0.set_ylabel("n points")
    ax0.set_title("Sample coverage")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.grid(axis="y", alpha=0.25)

    ax1.bar(s["idx"].to_numpy(dtype=float), z, color="#9467bd", width=0.58)
    ax1.axhline(2.0, color="#2ca02c", linestyle="--", linewidth=1.0)
    ax1.axhline(3.0, color="#ff7f0e", linestyle="--", linewidth=1.0)
    ax1.set_xticks(s["idx"].to_numpy(dtype=float))
    ax1.set_xticklabels(labels, rotation=28, ha="right")
    ax1.set_ylabel("|z(delta)|")
    ax1.set_title("Flag sensitivity")
    ax1.grid(axis="y", alpha=0.25)

    map_val = {"pass": 3.0, "watch": 2.0, "not_testable": 1.0, "reject": 0.0}
    y_status = np.asarray([map_val.get(v, 0.0) for v in status], dtype=float)
    colors = [
        "#2ca02c" if v == "pass" else "#ff7f0e" if v == "watch" else "#9aa0a6" if v == "not_testable" else "#d62728"
        for v in status
    ]
    ax2.bar(s["idx"].to_numpy(dtype=float), y_status, color=colors, width=0.58)
    ax2.set_yticks([0.0, 1.0, 2.0, 3.0])
    ax2.set_yticklabels(["reject", "not_testable", "watch", "pass"])
    ax2.set_xticks(s["idx"].to_numpy(dtype=float))
    ax2.set_xticklabels(labels, rotation=28, ha="right")
    ax2.set_ylabel("status")
    ax2.set_title("Per-test gate")
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle(f"LLR CRD flag unit test (8.7.47.16): overall={overall_status}", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


# 関数: `_sync_outputs` の入出力契約と処理意図を定義する。

def _sync_outputs(paths: Iterable[Path], *, private_root: Path, public_root: Path) -> List[str]:
    out: List[str] = []
    for src in paths:
        rel = src.resolve().relative_to(private_root.resolve())
        dst = public_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        out.append(str(dst))

    return out


# 関数: `main` の入出力契約と処理意図を定義する。

def main() -> int:
    ap = argparse.ArgumentParser(description="LLR CRD flag unit test (Step 8.7.47.16).")
    ap.add_argument("--points-csv", type=str, default=str(ROOT / "output" / "private" / "llr" / "batch" / "llr_batch_points.csv"))
    ap.add_argument("--out-dir", type=str, default=str(ROOT / "output" / "private" / "llr"))
    ap.add_argument("--public-dir", type=str, default=str(ROOT / "output" / "public" / "llr"))
    ap.add_argument("--core-script", type=str, default=str(ROOT / "scripts" / "llr" / "llr_kappa_llr_direct_fit.py"))
    ap.add_argument("--fit-mode", type=str, default="station_target_year")
    ap.add_argument("--weight-scheme", type=str, default="inv_station_target")
    ap.add_argument("--weight-floor-station", type=int, default=180)
    ap.add_argument("--weight-floor-target", type=int, default=180)
    ap.add_argument("--weight-floor-station-target", type=int, default=120)
    ap.add_argument("--max-weight-cap", type=float, default=8.0)
    ap.add_argument("--min-points-group", type=int, default=200)
    args = ap.parse_args()

    points_csv = Path(str(args.points_csv))
    out_dir = Path(str(args.out_dir))
    public_dir = Path(str(args.public_dir))
    core_script = Path(str(args.core_script))
    if not points_csv.is_absolute():
        points_csv = (ROOT / points_csv).resolve()

    if not out_dir.is_absolute():
        out_dir = (ROOT / out_dir).resolve()

    if not public_dir.is_absolute():
        public_dir = (ROOT / public_dir).resolve()

    if not core_script.is_absolute():
        core_script = (ROOT / core_script).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)

    core = _load_core_module(core_script)
    df = core._read_points(points_csv)
    if df.empty:
        raise RuntimeError(f"no valid rows from {points_csv}")

    required_extra = ["source_file", "lineno"]
    miss = [c for c in required_extra if c not in df.columns]
    if miss:
        raise RuntimeError(f"points missing required columns for flag audit: {miss}")

    flags_df = _extract_flags_for_points(ROOT, df)
    if flags_df.empty:
        raise RuntimeError("failed to extract any CRD flags from source_file+lineno")

    merged = df.merge(flags_df, on=["source_file", "lineno"], how="left")
    merged["np_quality_code"] = pd.to_numeric(merged["np_last_int"], errors="coerce")
    merged["h4_range_type"] = pd.to_numeric(merged["h4_range_type"], errors="coerce")
    merged["h4_refraction_applied"] = pd.to_numeric(merged["h4_refraction_applied"], errors="coerce")
    merged["h4_com_applied"] = pd.to_numeric(merged["h4_com_applied"], errors="coerce")
    merged["h4_system_delay_applied"] = pd.to_numeric(merged["h4_system_delay_applied"], errors="coerce")

    sample_weight = core._build_imbalance_weight(
        merged,
        scheme=str(args.weight_scheme),
        floor_station=int(args.weight_floor_station),
        floor_target=int(args.weight_floor_target),
        floor_station_target=int(args.weight_floor_station_target),
        max_weight_cap=float(args.max_weight_cap),
    )
    cluster_ids = _build_cluster_ids(merged)

    baseline = _fit_with_cluster_sigma(
        core=core,
        df_sub=merged,
        mode=str(args.fit_mode),
        sample_weight=sample_weight,
        cluster_ids=cluster_ids,
    )
    if baseline is None:
        raise RuntimeError("baseline fit failed")

    tests: List[Dict[str, Any]] = []
    specs = [
        ("range_type", "Range Type (2 vs 1)", "h4_range_type", [2.0], [1.0], "H4 range_type split"),
        (
            "system_delay",
            "System Delay (1 vs 0)",
            "h4_system_delay_applied",
            [1.0],
            [0.0],
            "H4 station system-delay correction flag",
        ),
        ("com", "COM Correction (1 vs 0)", "h4_com_applied", [1.0], [0.0], "H4 center-of-mass correction flag"),
        ("refraction", "Refraction (1 vs 0)", "h4_refraction_applied", [1.0], [0.0], "H4 tropospheric refraction flag"),
    ]
    for test_id, label, col, pos_vals, neg_vals, note in specs:
        vals = pd.to_numeric(merged[col], errors="coerce").to_numpy(dtype=float)
        mask_pos = np.isfinite(vals) & np.isin(vals, np.asarray(pos_vals, dtype=float))
        mask_neg = np.isfinite(vals) & np.isin(vals, np.asarray(neg_vals, dtype=float))
        tests.append(
            _split_fit_test(
                core=core,
                df=merged,
                fit_mode=str(args.fit_mode),
                sample_weight=sample_weight,
                cluster_ids=cluster_ids,
                mask_a=mask_pos,
                mask_b=mask_neg,
                min_points_group=int(args.min_points_group),
                test_id=test_id,
                test_label=label,
                note=note,
            )
        )

    q = pd.to_numeric(merged["np_quality_code"], errors="coerce").to_numpy(dtype=float)
    mask_zero = np.isfinite(q) & (q == 0.0)
    mask_nonzero = np.isfinite(q) & (q != 0.0)
    tests.append(
        _split_fit_test(
            core=core,
            df=merged,
            fit_mode=str(args.fit_mode),
            sample_weight=sample_weight,
            cluster_ids=cluster_ids,
            mask_a=mask_zero,
            mask_b=mask_nonzero,
            min_points_group=int(args.min_points_group),
            test_id="crd_vs_meritii_inversion",
            test_label="Quality inversion (0 vs nonzero)",
            note="CRD-vs-MeritII inversion proxy on record11 quality code",
        )
    )

    summary_df = pd.DataFrame(tests)
    summary_df = summary_df.sort_values(["test_id"]).reset_index(drop=True)
    status_values = summary_df["status"].astype(str).tolist()
    overall_status = _combine_status(status_values)
    n_testable = int(summary_df["status"].astype(str).isin(["pass", "watch", "reject"]).sum())
    n_not_testable = int(summary_df["status"].astype(str).eq("not_testable").sum())

    coverage_rows: List[Dict[str, Any]] = []
    for col in [
        "h4_range_type",
        "h4_system_delay_applied",
        "h4_com_applied",
        "h4_refraction_applied",
        "h4_quality_alert",
        "np_quality_code",
    ]:
        vals = pd.to_numeric(merged[col], errors="coerce")
        vc = vals.value_counts(dropna=False).to_dict()
        for key, count in vc.items():
            if isinstance(key, float) and np.isnan(key):
                val_txt = "NaN"
            else:
                val_txt = str(int(key)) if float(key).is_integer() else str(float(key))

            coverage_rows.append({"flag_name": col, "flag_value": val_txt, "n_points": int(count)})

    coverage_df = pd.DataFrame(coverage_rows).sort_values(["flag_name", "flag_value"]).reset_index(drop=True)

    summary_csv = out_dir / "llr_kappa_llr_crd_flag_unit_test_summary.csv"
    coverage_csv = out_dir / "llr_kappa_llr_crd_flag_unit_test_flag_coverage.csv"
    metrics_json = out_dir / "llr_kappa_llr_crd_flag_unit_test_metrics.json"
    plot_pdf = out_dir / "llr_kappa_llr_crd_flag_unit_test_audit.pdf"
    plot_png = out_dir / "llr_kappa_llr_crd_flag_unit_test_audit.png"
    summary_df.to_csv(summary_csv, index=False)
    coverage_df.to_csv(coverage_csv, index=False)
    _write_plot(summary_df=summary_df, overall_status=overall_status, out_pdf=plot_pdf, out_png=plot_png)

    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase": {"step": "8.7.47.16"},
        "input": {
            "points_csv": _safe_rel(points_csv, ROOT),
            "n_points": int(len(merged)),
            "fit_mode": str(args.fit_mode),
            "weight_scheme": str(args.weight_scheme),
            "weight_floor_station": int(args.weight_floor_station),
            "weight_floor_target": int(args.weight_floor_target),
            "weight_floor_station_target": int(args.weight_floor_station_target),
            "max_weight_cap": float(args.max_weight_cap),
            "min_points_group": int(args.min_points_group),
        },
        "baseline": {
            "kappa_est": float(baseline["kappa_est"]),
            "kappa_sigma_cluster": float(baseline["kappa_sigma_cluster"]),
            "status_cluster": str(baseline["status_cluster"]),
        },
        "tests": summary_df.to_dict(orient="records"),
        "gate_status": {
            "overall_status": overall_status,
            "per_test_status": status_values,
            "n_testable_tests": n_testable,
            "n_not_testable_tests": n_not_testable,
            "non_blocking_statuses": ["not_testable"],
        },
        "outputs": {
            "summary_csv": _safe_rel(summary_csv, ROOT),
            "coverage_csv": _safe_rel(coverage_csv, ROOT),
            "metrics_json": _safe_rel(metrics_json, ROOT),
            "plot_pdf": _safe_rel(plot_pdf, ROOT),
            "plot_png": _safe_rel(plot_png, ROOT),
        },
    }
    metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    produced = [summary_csv, coverage_csv, metrics_json, plot_pdf, plot_png]
    synced = _sync_outputs(paths=produced, private_root=out_dir, public_root=public_dir)
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {coverage_csv}")
    print(f"Wrote: {metrics_json}")
    print(f"Wrote: {plot_pdf}")
    print(f"Wrote: {plot_png}")
    print(f"Synced: {len(synced)} files")
    print(f"Status: {overall_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
