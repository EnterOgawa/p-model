#!/usr/bin/env python3
"""
messenger_beta_stage_g_spe_sensitivity.py

Roadmap Step 8.7.48.7 (SPE cut sensitivity audit) implementation.

Purpose:
- Re-run the Stage D/E joint-fit interface under SPE subset policies.
- Use only primary timestamps and a geometry-derived SPE proxy angle.
- Emit machine-readable Pass/Watch/Reject gates for SPE sensitivity.

Notes:
- SPE is approximated as the Sun-Mercury-Earth angle (MESSENGER ~= Mercury).
- This approximation keeps the workflow theory-native and reproducible with
  generic SPICE kernels only.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import spiceypy as spice

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from scripts.mercury.messenger_beta_stage_d_joint_fit import (
    _aggregate_channel,
    _build_design_matrix,
    _fit_joint,
    _load_channel_csv,
    _sync_to_public,
)
from scripts.summary.worklog import append_event


# Class: Defines per-subset fit outputs and gate status.
@dataclass
class SubsetFitRow:
    subset_key: str
    spe_rule: str
    n_rows: int
    n_range_rows: int
    n_doppler_rows: int
    beta_dyn: float
    beta_sigma: float
    beta_z_from_1: float
    rms_range: float
    rms_doppler: float
    fit_status: str
    sigma_status: str
    z_delta_vs_all: float
    status_delta_vs_all: str


# Function: Converts a path to a repo-relative string when possible.

def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve()).replace("\\", "/")


# Function: Resolves absolute/relative path against repository root.

def _resolve_path(path_str: str, root: Path) -> Path:
    p = Path(str(path_str))
    if p.is_absolute():
        return p

    return (root / p).resolve()


# Function: Combines statuses with reject > watch > pass priority.

def _combine_status(values: Iterable[str]) -> str:
    norm = [str(v or "").strip().lower() for v in values if str(v or "").strip()]
    if len(norm) <= 0:
        return "reject"

    if any(v == "reject" for v in norm):
        return "reject"

    if all(v == "pass" for v in norm):
        return "pass"

    return "watch"


# Function: Threshold gate helper for absolute-value checks.

def _status_from_abs(value: float, pass_thr: float, watch_thr: float) -> str:
    if not math.isfinite(value):
        return "reject"

    if float(value) <= float(pass_thr):
        return "pass"

    if float(value) <= float(watch_thr):
        return "watch"

    return "reject"


# Function: Downloads one URL to destination if missing.

def _download_if_missing(url: str, dst: Path, timeout_sec: float) -> Tuple[bool, str]:
    if dst.exists():
        return (False, "exists")

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "waveP-spe-stage-g/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=float(timeout_sec)) as r, tmp.open("wb") as f:
            shutil.copyfileobj(r, f)
    except urllib.error.HTTPError as e:
        if tmp.exists():
            tmp.unlink()

        return (False, f"http_error:{int(e.code)}")
    except urllib.error.URLError as e:
        if tmp.exists():
            tmp.unlink()

        return (False, f"url_error:{e.reason}")
    except Exception as e:
        if tmp.exists():
            tmp.unlink()

        return (False, f"download_error:{type(e).__name__}")

    tmp.replace(dst)
    return (True, "download")


# Function: Ensures minimal NAIF generic kernels needed for SPE proxy computation.

def _ensure_generic_kernels(
    kernels_dir: Path,
    auto_fetch: bool,
    timeout_sec: float,
) -> Tuple[Dict[str, Path], List[Dict[str, object]]]:
    kernels_dir.mkdir(parents=True, exist_ok=True)
    required = {
        "lsk": (
            "naif0012.tls",
            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/lsk/naif0012.tls",
        ),
        "spk_planet": (
            "de440s.bsp",
            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp",
        ),
    }

    out: Dict[str, Path] = {}
    rows: List[Dict[str, object]] = []
    for key, (name, url) in required.items():
        dst = kernels_dir / name
        if dst.exists():
            status = "exists"
            changed = False
            note = "ok"
        else:
            if bool(auto_fetch):
                changed, note = _download_if_missing(url=url, dst=dst, timeout_sec=float(timeout_sec))
                status = "download" if changed else "reject"
            else:
                changed = False
                status = "reject"
                note = "missing_and_auto_fetch_disabled"

        exists_now = bool(dst.exists())
        if exists_now:
            out[str(key)] = dst

        rows.append(
            {
                "kernel_key": str(key),
                "filename": str(name),
                "status": str(status if exists_now else "reject"),
                "note": str(note),
                "exists": int(1 if exists_now else 0),
                "bytes": int(dst.stat().st_size) if exists_now else 0,
                "path": _safe_rel(dst, _ROOT),
                "url": str(url),
            }
        )

    return (out, rows)


# Function: Loads SPICE kernels and computes Sun-Mercury-Earth angle per epoch.

def _compute_spe_proxy_deg(epochs: pd.Series, kernels: Dict[str, Path]) -> pd.Series:
    parsed = pd.to_datetime(epochs, utc=True, errors="coerce")
    out = pd.Series(np.nan, index=parsed.index, dtype=float)
    if len(parsed) <= 0:
        return out

    valid_idx = parsed.notna()
    if int(valid_idx.sum()) <= 0:
        return out

    iso_full = parsed.dt.strftime("%Y-%m-%dT%H:%M:%S")
    unique_times = sorted(set(iso_full.loc[valid_idx].astype(str).tolist()))

    spice.kclear()
    spice.furnsh(str(kernels["lsk"]))
    spice.furnsh(str(kernels["spk_planet"]))

    angle_map: Dict[str, float] = {}
    for text in unique_times:
        try:
            et = spice.utc2et(str(text))
            vec_sun, _ = spice.spkpos("SUN", et, "J2000", "NONE", "MERCURY BARYCENTER")
            vec_earth, _ = spice.spkpos("EARTH BARYCENTER", et, "J2000", "NONE", "MERCURY BARYCENTER")
            dot = float(np.dot(vec_sun, vec_earth))
            den = float(np.linalg.norm(vec_sun) * np.linalg.norm(vec_earth))
            if den <= 0.0:
                ang = float("nan")
            else:
                cs = float(np.clip(dot / den, -1.0, 1.0))
                ang = float(np.degrees(np.arccos(cs)))
        except Exception:
            ang = float("nan")

        angle_map[str(text)] = float(ang)

    spice.kclear()
    out.loc[valid_idx] = iso_full.loc[valid_idx].map(angle_map).astype(float)
    return out


# Function: Computes z-distance between subset and baseline beta estimates.

def _z_delta_beta(beta: float, sigma: float, beta_ref: float, sigma_ref: float) -> float:
    if (not math.isfinite(beta)) or (not math.isfinite(sigma)):
        return float("nan")

    if (not math.isfinite(beta_ref)) or (not math.isfinite(sigma_ref)):
        return float("nan")

    denom = float(math.sqrt(max(0.0, (sigma * sigma) + (sigma_ref * sigma_ref))))
    if denom <= 0.0:
        return float("nan")

    return float(abs(beta - beta_ref) / denom)


# Function: Applies SPE rule mask to aggregated joint dataframe.

def _mask_by_subset(spe_deg: pd.Series, subset_key: str) -> np.ndarray:
    x = pd.to_numeric(spe_deg, errors="coerce")
    if str(subset_key) == "all_rows":
        return x.notna().to_numpy()

    if str(subset_key) == "spe_gt_90":
        return (x > 90.0).to_numpy()

    if str(subset_key) == "spe_35_90":
        return ((x > 35.0) & (x <= 90.0)).to_numpy()

    if str(subset_key) == "spe_gt_35":
        return (x > 35.0).to_numpy()

    if str(subset_key) == "spe_le_35":
        return (x <= 35.0).to_numpy()

    return np.zeros(len(x), dtype=bool)


# Function: Human-readable rule text for each subset key.

def _subset_rule_text(subset_key: str) -> str:
    if str(subset_key) == "all_rows":
        return "SPE finite (baseline)"

    if str(subset_key) == "spe_gt_90":
        return "SPE > 90 deg (primary)"

    if str(subset_key) == "spe_35_90":
        return "35 < SPE <= 90 deg (extended)"

    if str(subset_key) == "spe_gt_35":
        return "SPE > 35 deg (operational)"

    if str(subset_key) == "spe_le_35":
        return "SPE <= 35 deg (excluded policy branch)"

    return "unknown"


# Function: Runs Stage D-equivalent fit on one subset.

def _fit_subset(
    df_joint: pd.DataFrame,
    orbital_period_days: float,
    max_station_bias_per_channel: int,
    min_joint_rows: int,
    sigma_watch_threshold: float,
) -> Dict[str, object]:
    if len(df_joint) <= 0:
        return {
            "n_rows": 0,
            "n_range_rows": 0,
            "n_doppler_rows": 0,
            "beta_dyn": float("nan"),
            "beta_sigma": float("nan"),
            "beta_z_from_1": float("nan"),
            "rms_range": float("nan"),
            "rms_doppler": float("nan"),
            "fit_status": "reject",
            "sigma_status": "reject",
        }

    X, y_norm, y_obs, labels, _meta, work = _build_design_matrix(
        df_joint,
        orbital_period_days=float(orbital_period_days),
        max_station_bias_per_channel=int(max_station_bias_per_channel),
    )
    channels = work["channel"].astype(str).to_numpy()
    fit, _coef, _fit_norm, _residual_norm = _fit_joint(
        X=X,
        y_norm=y_norm,
        y_obs=y_obs,
        scale_by_row=work["scale_by_row"].to_numpy(dtype=float),
        labels=labels,
        channels=channels,
        min_rows=int(min_joint_rows),
        sigma_watch_threshold=float(sigma_watch_threshold),
    )
    return {
        "n_rows": int(fit.n_rows),
        "n_range_rows": int(fit.n_range_rows),
        "n_doppler_rows": int(fit.n_doppler_rows),
        "beta_dyn": float(fit.beta_dyn),
        "beta_sigma": float(fit.beta_sigma),
        "beta_z_from_1": float(fit.beta_z_from_1),
        "rms_range": float(fit.rms_range),
        "rms_doppler": float(fit.rms_doppler),
        "fit_status": str(fit.status_data),
        "sigma_status": str(fit.status_sigma),
    }


# Function: Saves kernel status rows to CSV.

def _write_kernel_status_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["kernel_key", "filename", "status", "note", "exists", "bytes", "path", "url"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# Function: Writes SPE profile rows for reproducibility and diagnostics.

def _write_spe_profile_csv(path: Path, df_joint: pd.DataFrame) -> None:
    cols = [
        "epoch_utc",
        "channel",
        "station_id",
        "observable_value",
        "spe_proxy_deg",
    ]
    keep = [c for c in cols if c in df_joint.columns]
    path.parent.mkdir(parents=True, exist_ok=True)
    df_joint[keep].to_csv(path, index=False)


# Function: Creates SPE histogram + subset beta error-bar summary plot.

def _make_plot(
    df_joint: pd.DataFrame,
    rows_df: pd.DataFrame,
    out_pdf: Path,
    out_png: Path,
) -> Optional[str]:
    if plt is None:
        return "matplotlib_unavailable"

    if len(df_joint) <= 0 or len(rows_df) <= 0:
        return "no_data"

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13.2, 5.4), constrained_layout=True)
    spe = pd.to_numeric(df_joint["spe_proxy_deg"], errors="coerce").to_numpy(dtype=float)
    spe = spe[np.isfinite(spe)]
    if len(spe) > 0:
        ax0.hist(spe, bins=32, color="#2A6EA6", alpha=0.85)
        ax0.axvline(35.0, color="#7A7A7A", linestyle="--", linewidth=1.2)
        ax0.axvline(90.0, color="#7A7A7A", linestyle="--", linewidth=1.2)

    ax0.set_xlabel("SPE proxy (deg)")
    ax0.set_ylabel("Count")
    ax0.set_title("Stage 8.7.48.7: SPE proxy distribution")
    ax0.grid(alpha=0.28)

    view = rows_df.loc[rows_df["subset_key"] != "all_rows"].copy()
    view = view.reset_index(drop=True)
    x = np.arange(len(view), dtype=float)
    y = pd.to_numeric(view["beta_dyn"], errors="coerce").to_numpy(dtype=float)
    s = pd.to_numeric(view["beta_sigma"], errors="coerce").to_numpy(dtype=float)
    ax1.axhline(1.0, color="#7A7A7A", linestyle="--", linewidth=1.2)
    if len(view) > 0:
        ax1.errorbar(x, y, yerr=s, fmt="o", capsize=4, color="#C23B22")
        ax1.set_xticks(x, view["subset_key"].astype(str).tolist(), rotation=20, ha="right")

    ax1.set_ylabel("beta_dyn")
    ax1.set_title("Subset beta estimates")
    ax1.grid(alpha=0.28)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return None


# Function: Main entrypoint for roadmap step 8.7.48.7.

def main() -> int:
    ap = argparse.ArgumentParser(description="Roadmap 8.7.48.7: SPE cut sensitivity audit.")
    ap.add_argument("--data-root", type=str, default=str(_ROOT / "data" / "mercury" / "messenger"))
    ap.add_argument("--doppler-csv", type=str, default="")
    ap.add_argument("--range-csv", type=str, default="")
    ap.add_argument("--source-branch", type=str, default="tnf", choices=("tnf", "odf"))
    ap.add_argument(
        "--kernels-dir",
        type=str,
        default=str(_ROOT / "data" / "mercury" / "messenger" / "spice" / "generic"),
    )
    ap.add_argument("--auto-fetch-kernels", action="store_true")
    ap.add_argument("--timeout-sec", type=float, default=120.0)
    ap.add_argument("--doppler-bin-minutes", type=int, default=30)
    ap.add_argument("--range-bin-minutes", type=int, default=30)
    ap.add_argument("--min-joint-rows", type=int, default=300)
    ap.add_argument("--max-station-bias-per-channel", type=int, default=8)
    ap.add_argument("--orbital-period-days", type=float, default=87.9691)
    ap.add_argument("--sigma-watch-threshold", type=float, default=0.1)
    ap.add_argument("--out-dir", type=str, default=str(_ROOT / "output" / "private" / "mercury"))
    ap.add_argument("--public-dir", type=str, default=str(_ROOT / "output" / "public" / "mercury"))
    args = ap.parse_args()

    data_root = _resolve_path(args.data_root, _ROOT)
    if str(args.doppler_csv).strip():
        doppler_csv = _resolve_path(args.doppler_csv, _ROOT)
    else:
        if str(args.source_branch) == "odf":
            doppler_csv = data_root / "derived" / "odf_doppler_observations.csv"
        else:
            doppler_csv = data_root / "derived" / "tnf_doppler_observations.csv"

    if str(args.range_csv).strip():
        range_csv = _resolve_path(args.range_csv, _ROOT)
    else:
        if str(args.source_branch) == "odf":
            range_csv = data_root / "derived" / "odf_range_observations.csv"
        else:
            range_csv = data_root / "derived" / "tnf_range_observations.csv"

    kernels_dir = _resolve_path(args.kernels_dir, _ROOT)
    out_dir = _resolve_path(args.out_dir, _ROOT)
    public_dir = _resolve_path(args.public_dir, _ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_summary_csv = out_dir / "messenger_beta_stage_g_spe_sensitivity_summary.csv"
    out_profile_csv = out_dir / "messenger_beta_stage_g_spe_profile.csv"
    out_kernel_csv = out_dir / "messenger_beta_stage_g_spe_kernel_status.csv"
    out_metrics_json = out_dir / "messenger_beta_stage_g_spe_sensitivity_metrics.json"
    out_plot_pdf = out_dir / "messenger_beta_stage_g_spe_sensitivity.pdf"
    out_plot_png = out_dir / "messenger_beta_stage_g_spe_sensitivity.png"

    if (not doppler_csv.exists()) or (not range_csv.exists()):
        payload = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "phase_step": "8.7.48.7",
            "overall_status": "reject",
            "reason": "input_missing",
            "source_branch": str(args.source_branch),
            "doppler_csv": _safe_rel(doppler_csv, _ROOT),
            "range_csv": _safe_rel(range_csv, _ROOT),
        }
        out_metrics_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        synced = _sync_to_public([out_metrics_json], private_root=out_dir, public_root=public_dir)
        append_event(
            {
                "event": "run_script",
                "script": "scripts/mercury/messenger_beta_stage_g_spe_sensitivity.py",
                "phase_step": "8.7.48.7",
                "status": "reject",
                "input": f"{_safe_rel(doppler_csv, _ROOT)}|{_safe_rel(range_csv, _ROOT)}",
                "outputs": [_safe_rel(out_metrics_json, _ROOT)],
                "metrics": {"reason": "input_missing"},
            }
        )
        print("[warn] Stage G skipped: required input CSV missing.")
        print(f"[ok] wrote: {out_metrics_json}")
        print(f"[ok] synced_to_public={len(synced)}")
        return 0

    kernels, kernel_rows = _ensure_generic_kernels(
        kernels_dir=kernels_dir,
        auto_fetch=bool(args.auto_fetch_kernels),
        timeout_sec=float(args.timeout_sec),
    )
    _write_kernel_status_csv(out_kernel_csv, kernel_rows)
    kernel_status = "pass" if ("lsk" in kernels and "spk_planet" in kernels) else "reject"
    if kernel_status == "reject":
        payload = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "phase_step": "8.7.48.7",
            "overall_status": "reject",
            "reason": "kernel_missing",
            "source_branch": str(args.source_branch),
            "kernels_dir": _safe_rel(kernels_dir, _ROOT),
            "kernel_status": kernel_rows,
            "doppler_csv": _safe_rel(doppler_csv, _ROOT),
            "range_csv": _safe_rel(range_csv, _ROOT),
            "outputs_private": [_safe_rel(out_kernel_csv, _ROOT)],
        }
        out_metrics_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        synced = _sync_to_public([out_kernel_csv, out_metrics_json], private_root=out_dir, public_root=public_dir)
        append_event(
            {
                "event": "run_script",
                "script": "scripts/mercury/messenger_beta_stage_g_spe_sensitivity.py",
                "phase_step": "8.7.48.7",
                "status": "reject",
                "input": f"{_safe_rel(doppler_csv, _ROOT)}|{_safe_rel(range_csv, _ROOT)}",
                "outputs": [_safe_rel(out_kernel_csv, _ROOT), _safe_rel(out_metrics_json, _ROOT)],
                "metrics": {"reason": "kernel_missing"},
            }
        )
        print("[warn] Stage G skipped: required SPICE kernels missing.")
        print(f"[ok] wrote: {out_metrics_json}")
        print(f"[ok] synced_to_public={len(synced)}")
        return 0

    doppler_df = _load_channel_csv(doppler_csv, channel="doppler")
    range_df = _load_channel_csv(range_csv, channel="range")
    doppler_agg = _aggregate_channel(doppler_df, bin_minutes=int(args.doppler_bin_minutes))
    range_agg = _aggregate_channel(range_df, bin_minutes=int(args.range_bin_minutes))
    joint_df = pd.concat([range_agg, doppler_agg], ignore_index=True).sort_values("epoch_utc").reset_index(drop=True)
    joint_df["spe_proxy_deg"] = _compute_spe_proxy_deg(joint_df["epoch_utc"], kernels=kernels)
    _write_spe_profile_csv(out_profile_csv, joint_df)

    subset_keys = [
        "all_rows",
        "spe_gt_90",
        "spe_35_90",
        "spe_gt_35",
        "spe_le_35",
    ]
    fit_map: Dict[str, Dict[str, object]] = {}
    for key in subset_keys:
        mask = _mask_by_subset(joint_df["spe_proxy_deg"], subset_key=key)
        df_sub = joint_df.loc[mask].copy().reset_index(drop=True)
        fit_map[str(key)] = _fit_subset(
            df_joint=df_sub,
            orbital_period_days=float(args.orbital_period_days),
            max_station_bias_per_channel=int(args.max_station_bias_per_channel),
            min_joint_rows=int(args.min_joint_rows),
            sigma_watch_threshold=float(args.sigma_watch_threshold),
        )

    base_beta = float(fit_map["all_rows"]["beta_dyn"])
    base_sigma = float(fit_map["all_rows"]["beta_sigma"])
    rows: List[SubsetFitRow] = []
    for key in subset_keys:
        fit = fit_map[str(key)]
        z_delta = _z_delta_beta(
            beta=float(fit["beta_dyn"]),
            sigma=float(fit["beta_sigma"]),
            beta_ref=base_beta,
            sigma_ref=base_sigma,
        )
        if key == "all_rows":
            z_delta = 0.0
            status_delta = "pass"
        else:
            status_delta = _status_from_abs(z_delta, pass_thr=2.0, watch_thr=5.0)

        rows.append(
            SubsetFitRow(
                subset_key=str(key),
                spe_rule=_subset_rule_text(str(key)),
                n_rows=int(fit["n_rows"]),
                n_range_rows=int(fit["n_range_rows"]),
                n_doppler_rows=int(fit["n_doppler_rows"]),
                beta_dyn=float(fit["beta_dyn"]),
                beta_sigma=float(fit["beta_sigma"]),
                beta_z_from_1=float(fit["beta_z_from_1"]),
                rms_range=float(fit["rms_range"]),
                rms_doppler=float(fit["rms_doppler"]),
                fit_status=str(fit["fit_status"]),
                sigma_status=str(fit["sigma_status"]),
                z_delta_vs_all=float(z_delta),
                status_delta_vs_all=str(status_delta),
            )
        )

    rows_df = pd.DataFrame(
        [
            {
                "subset_key": r.subset_key,
                "spe_rule": r.spe_rule,
                "n_rows": r.n_rows,
                "n_range_rows": r.n_range_rows,
                "n_doppler_rows": r.n_doppler_rows,
                "beta_dyn": r.beta_dyn,
                "beta_sigma": r.beta_sigma,
                "beta_z_from_1": r.beta_z_from_1,
                "rms_range": r.rms_range,
                "rms_doppler": r.rms_doppler,
                "fit_status": r.fit_status,
                "sigma_status": r.sigma_status,
                "z_delta_vs_all": r.z_delta_vs_all,
                "status_delta_vs_all": r.status_delta_vs_all,
            }
            for r in rows
        ]
    )
    rows_df.to_csv(out_summary_csv, index=False)

    # Function: Computes subset gate status with insufficient-row watch policy.
    def _status_for_subset(key: str) -> str:
        row = rows_df.loc[rows_df["subset_key"] == str(key)]
        if len(row) <= 0:
            return "reject"

        fit_status = str(row["fit_status"].iloc[0])
        delta_status = str(row["status_delta_vs_all"].iloc[0])
        n_rows = int(pd.to_numeric(row["n_rows"], errors="coerce").fillna(0).iloc[0])
        if n_rows < int(args.min_joint_rows):
            return "watch"

        return _combine_status([fit_status, delta_status])

    status_primary = _status_for_subset("spe_gt_90")
    status_extended = _status_for_subset("spe_35_90")
    status_operational = _status_for_subset("spe_gt_35")
    status_excluded = _status_for_subset("spe_le_35")

    # Extended branch is diagnostic-only because it is not the primary gate.
    if str(status_extended) == "reject":
        status_extended_policy = "watch"
    else:
        status_extended_policy = str(status_extended)

    # Excluded branch is diagnostic-only; it should not dominate primary gate.

    if str(status_excluded) == "reject":
        status_excluded_policy = "watch"
    else:
        status_excluded_policy = str(status_excluded)

    overall = _combine_status(
        [
            kernel_status,
            status_primary,
            status_operational,
        ]
    )

    plot_note = _make_plot(df_joint=joint_df, rows_df=rows_df, out_pdf=out_plot_pdf, out_png=out_plot_png)
    produced: List[Path] = [out_summary_csv, out_profile_csv, out_kernel_csv, out_metrics_json]
    if plot_note is None:
        produced.extend([out_plot_pdf, out_plot_png])

    spe_vals = pd.to_numeric(joint_df["spe_proxy_deg"], errors="coerce").to_numpy(dtype=float)
    finite_spe = spe_vals[np.isfinite(spe_vals)]
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase_step": "8.7.48.7",
        "overall_status": overall,
        "source_branch": str(args.source_branch),
        "doppler_csv": _safe_rel(doppler_csv, _ROOT),
        "range_csv": _safe_rel(range_csv, _ROOT),
        "kernels_dir": _safe_rel(kernels_dir, _ROOT),
        "kernel_status": kernel_status,
        "kernel_rows": kernel_rows,
        "n_rows_joint": int(len(joint_df)),
        "n_rows_range": int(np.sum(joint_df["channel"].astype(str).to_numpy() == "range")),
        "n_rows_doppler": int(np.sum(joint_df["channel"].astype(str).to_numpy() == "doppler")),
        "spe_proxy_stats": {
            "n_finite": int(len(finite_spe)),
            "min_deg": float(np.min(finite_spe)) if len(finite_spe) > 0 else float("nan"),
            "median_deg": float(np.median(finite_spe)) if len(finite_spe) > 0 else float("nan"),
            "max_deg": float(np.max(finite_spe)) if len(finite_spe) > 0 else float("nan"),
        },
        "subset_status": {
            "spe_gt_90": status_primary,
            "spe_35_90_diagnostic": status_extended_policy,
            "spe_gt_35": status_operational,
            "spe_le_35_diagnostic": status_excluded_policy,
        },
        "gating_policy": {
            "primary_subset": "SPE > 90",
            "extended_subset": "35 < SPE <= 90",
            "operational_subset": "SPE > 35",
            "excluded_subset": "SPE <= 35",
            "required_subsets_for_overall": ["SPE > 90", "SPE > 35"],
            "diagnostic_only_subsets": ["35 < SPE <= 90", "SPE <= 35"],
            "subset_delta_pass_abs_z": 2.0,
            "subset_delta_watch_abs_z": 5.0,
            "insufficient_rows_policy": "watch",
            "spe_definition": "Sun-Mercury-Earth angle (Mercury proxy for MESSENGER)",
        },
        "outputs_private": [_safe_rel(p, _ROOT) for p in produced if p != out_metrics_json],
        "plot": "generated" if plot_note is None else plot_note,
    }
    out_metrics_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    synced = _sync_to_public(produced, private_root=out_dir, public_root=public_dir)
    payload["outputs_public"] = [_safe_rel(p, _ROOT) for p in synced if p.name != out_metrics_json.name]
    out_metrics_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _sync_to_public([out_metrics_json], private_root=out_dir, public_root=public_dir)

    append_event(
        {
            "event": "run_script",
            "script": "scripts/mercury/messenger_beta_stage_g_spe_sensitivity.py",
            "phase_step": "8.7.48.7",
            "status": overall,
            "input": f"{_safe_rel(doppler_csv, _ROOT)}|{_safe_rel(range_csv, _ROOT)}",
            "outputs": [_safe_rel(p, _ROOT) for p in produced],
            "metrics": {
                "n_rows_joint": int(len(joint_df)),
                "status_primary": status_primary,
                "status_operational": status_operational,
                "status_extended_policy": status_extended_policy,
                "status_excluded_policy": status_excluded_policy,
            },
        }
    )

    print(f"[ok] stage_g_overall={overall}")
    print(f"[ok] source_branch={args.source_branch} n_rows_joint={int(len(joint_df))}")
    print(
        "[ok] subset_status primary={} operational={} extended={} excluded={}".format(
            status_primary,
            status_operational,
            status_extended_policy,
            status_excluded_policy,
        )
    )
    print(f"[ok] wrote: {out_summary_csv}")
    print(f"[ok] wrote: {out_metrics_json}")
    if plot_note is None:
        print(f"[ok] wrote: {out_plot_pdf}")
        print(f"[ok] wrote: {out_plot_png}")
    else:
        print(f"[warn] plot skipped: {plot_note}")

    print(f"[ok] synced_to_public={len(synced)}")
    return 0


# Condition: Evaluate script main path for CLI execution.

if __name__ == "__main__":
    raise SystemExit(main())
