#!/usr/bin/env python3
"""
messenger_beta_stage_h_segmentation_audit.py

Roadmap Step 8.7.48.8 (segmentation audit) implementation.

Purpose:
- Run segmentation consistency checks for station/link_type/campaign
  using the same joint-fit interface as Stage D/E.
- Keep the workflow reproducible with machine-readable outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from scripts.mercury.messenger_beta_stage_d_joint_fit import (
    _build_design_matrix,
    _fit_joint,
    _parse_epoch_series,
    _sync_to_public,
)
from scripts.summary.worklog import append_event


# Class: Defines one segmentation fit row for CSV/JSON output.
@dataclass
class SegmentRow:
    branch: str
    segmentation_type: str
    segment_key: str
    n_rows: int
    n_range_rows: int
    n_doppler_rows: int
    beta_dyn: float
    beta_sigma: float
    beta_z_from_1: float
    z_delta_vs_branch: float
    fit_status: str
    coverage_status: str
    delta_status: str
    overall_status: str


# Class: Defines consistency diagnostics for each segmentation type.

@dataclass
class ConsistencyRow:
    branch: str
    segmentation_type: str
    n_segments_valid: int
    weighted_beta_mean: float
    chi2: float
    dof: int
    chi2_per_dof: float
    status: str
    note: str


# Function: Returns repository-relative path when possible.

def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path.resolve()).replace("\\", "/")


# Function: Resolves possibly-relative path against repository root.

def _resolve_path(path_str: str, root: Path) -> Path:
    p = Path(str(path_str))
    if p.is_absolute():
        return p

    return (root / p).resolve()


# Function: Parses comma-separated tokens into a normalized list.

def _parse_csv_tokens(text: str) -> List[str]:
    out: List[str] = []
    for tok in str(text or "").split(","):
        key = str(tok or "").strip().lower()
        if len(key) <= 0:
            continue

        out.append(key)

    return out


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


# Function: Returns pass/watch/reject from absolute thresholding.

def _status_from_abs(value: float, pass_thr: float, watch_thr: float) -> str:
    if not math.isfinite(value):
        return "reject"

    if float(value) <= float(pass_thr):
        return "pass"

    if float(value) <= float(watch_thr):
        return "watch"

    return "reject"


# Function: Parses campaign key from source filename (ODF/TNF specific patterns).

def _campaign_key(source_file: object) -> str:
    text = str(source_file or "")
    base = Path(text.replace("\\", "/")).name.lower()
    m_odf = re.search(r"mess_rs_(\d{5})_(\d{3})_odf", base)
    if m_odf is not None:
        return f"odf_{m_odf.group(1)}_{m_odf.group(2)}"

    m_tnf = re.search(r"^(\d{5})\d+sc", base)
    if m_tnf is not None:
        return f"tnf_{m_tnf.group(1)}"

    m_generic = re.search(r"(\d{5})", base)
    if m_generic is not None:
        return f"camp_{m_generic.group(1)}"

    return "unknown"


# Function: Loads one branch/channel CSV and normalizes key columns.

def _load_branch_channel_csv(path: Path, channel: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "epoch_utc" not in df.columns:
        raise ValueError(f"{path}: missing epoch_utc")

    if "observable_value" not in df.columns:
        raise ValueError(f"{path}: missing observable_value")

    work = pd.DataFrame()
    work["epoch_utc"] = _parse_epoch_series(df["epoch_utc"])
    work["observable_value"] = pd.to_numeric(df["observable_value"], errors="coerce")
    work["channel"] = str(channel)
    if "station_id" in df.columns:
        work["station_id"] = df["station_id"].astype(str)
    else:
        work["station_id"] = "unknown"

    if "link_type" in df.columns:
        work["link_type"] = df["link_type"].astype(str).fillna("unknown")
    else:
        work["link_type"] = "unknown"

    if "source_file" in df.columns:
        work["campaign"] = df["source_file"].map(_campaign_key)
    else:
        work["campaign"] = "unknown"

    work = work.dropna(subset=["epoch_utc", "observable_value"]).reset_index(drop=True)
    return work


# Function: Aggregates rows while preserving segmentation columns.

def _aggregate_with_segmentation(df: pd.DataFrame, bin_minutes: int) -> pd.DataFrame:
    work = df.copy()
    if int(bin_minutes) > 0:
        work["epoch_bin"] = work["epoch_utc"].dt.floor(f"{int(bin_minutes)}min")
    else:
        work["epoch_bin"] = work["epoch_utc"]

    out = (
        work.groupby(
            ["epoch_bin", "station_id", "channel", "link_type", "campaign"],
            as_index=False,
        )
        .agg(observable_value=("observable_value", "median"))
        .sort_values("epoch_bin")
        .reset_index(drop=True)
    )
    out["epoch_utc"] = out["epoch_bin"]
    out = out.drop(columns=["epoch_bin"])
    return out


# Function: Runs one fit on a dataframe subset via Stage D shared interface.

def _run_fit(
    df_subset: pd.DataFrame,
    orbital_period_days: float,
    max_station_bias_per_channel: int,
    min_joint_rows: int,
    sigma_watch_threshold: float,
) -> Dict[str, object]:
    if len(df_subset) <= 0:
        return {
            "n_rows": 0,
            "n_range_rows": 0,
            "n_doppler_rows": 0,
            "beta_dyn": float("nan"),
            "beta_sigma": float("nan"),
            "beta_z_from_1": float("nan"),
            "fit_status": "reject",
            "rms_range": float("nan"),
            "rms_doppler": float("nan"),
        }

    X, y_norm, y_obs, labels, _meta, work = _build_design_matrix(
        df_subset[["epoch_utc", "observable_value", "station_id", "channel"]].copy(),
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
        "fit_status": str(fit.overall_status),
        "rms_range": float(fit.rms_range),
        "rms_doppler": float(fit.rms_doppler),
    }


# Function: Computes z-difference between subset and branch baseline beta.

def _z_delta_beta(beta: float, sigma: float, beta_ref: float, sigma_ref: float) -> float:
    if (not math.isfinite(beta)) or (not math.isfinite(sigma)):
        return float("nan")

    if (not math.isfinite(beta_ref)) or (not math.isfinite(sigma_ref)):
        return float("nan")

    denom = float(math.sqrt(max(0.0, sigma * sigma + sigma_ref * sigma_ref)))
    if denom <= 0.0:
        return float("nan")

    return float(abs(beta - beta_ref) / denom)


# Function: Selects top-N keys by occurrence count.

def _top_keys(series: pd.Series, max_n: int) -> List[str]:
    counts = series.astype(str).value_counts()
    keys = counts.index.astype(str).tolist()
    if int(max_n) > 0:
        keys = keys[: int(max_n)]

    return keys


# Function: Builds segmentation candidate sets per branch.

def _build_segmentation_sets(
    df_agg: pd.DataFrame,
    top_station_n: int,
    top_campaign_n: int,
) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    out["station"] = _top_keys(df_agg["station_id"], max_n=int(top_station_n))
    out["link_type"] = _top_keys(df_agg["link_type"], max_n=0)
    out["campaign"] = _top_keys(df_agg["campaign"], max_n=int(top_campaign_n))
    return out


# Function: Maps segmentation type to dataframe column name.

def _segmentation_column(segmentation_type: str) -> str:
    if str(segmentation_type) == "station":
        return "station_id"

    if str(segmentation_type) == "link_type":
        return "link_type"

    if str(segmentation_type) == "campaign":
        return "campaign"

    return str(segmentation_type)


# Function: Computes weighted consistency (chi2/dof) for one segmentation type.

def _consistency_for_type(
    branch: str,
    segmentation_type: str,
    rows: Sequence[SegmentRow],
    min_consistency_rows: int,
) -> ConsistencyRow:
    sel: List[SegmentRow] = []
    for row in rows:
        if row.branch != str(branch):
            continue

        if row.segmentation_type != str(segmentation_type):
            continue

        if row.n_rows < int(min_consistency_rows):
            continue

        if not math.isfinite(row.beta_dyn):
            continue

        if not math.isfinite(row.beta_sigma) or row.beta_sigma <= 0.0:
            continue

        sel.append(row)

    n = int(len(sel))
    if n < 2:
        return ConsistencyRow(
            branch=str(branch),
            segmentation_type=str(segmentation_type),
            n_segments_valid=n,
            weighted_beta_mean=float("nan"),
            chi2=float("nan"),
            dof=max(0, n - 1),
            chi2_per_dof=float("nan"),
            status="watch",
            note="insufficient_segments_for_consistency",
        )

    beta = np.array([r.beta_dyn for r in sel], dtype=float)
    sigma = np.array([r.beta_sigma for r in sel], dtype=float)
    w = 1.0 / np.square(sigma)
    wsum = float(np.sum(w))
    mean = float(np.sum(w * beta) / wsum)
    chi2 = float(np.sum(np.square((beta - mean) / sigma)))
    dof = int(max(1, n - 1))
    chi2_dof = float(chi2 / dof)
    status = _status_from_abs(chi2_dof, pass_thr=3.0, watch_thr=8.0)
    return ConsistencyRow(
        branch=str(branch),
        segmentation_type=str(segmentation_type),
        n_segments_valid=n,
        weighted_beta_mean=mean,
        chi2=chi2,
        dof=dof,
        chi2_per_dof=chi2_dof,
        status=status,
        note="ok",
    )


# Function: Writes segment summary rows to CSV.

def _write_segments_csv(path: Path, rows: Sequence[SegmentRow]) -> None:
    fields = [
        "branch",
        "segmentation_type",
        "segment_key",
        "n_rows",
        "n_range_rows",
        "n_doppler_rows",
        "beta_dyn",
        "beta_sigma",
        "beta_z_from_1",
        "z_delta_vs_branch",
        "fit_status",
        "coverage_status",
        "delta_status",
        "overall_status",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "branch": r.branch,
                    "segmentation_type": r.segmentation_type,
                    "segment_key": r.segment_key,
                    "n_rows": int(r.n_rows),
                    "n_range_rows": int(r.n_range_rows),
                    "n_doppler_rows": int(r.n_doppler_rows),
                    "beta_dyn": float(r.beta_dyn),
                    "beta_sigma": float(r.beta_sigma),
                    "beta_z_from_1": float(r.beta_z_from_1),
                    "z_delta_vs_branch": float(r.z_delta_vs_branch),
                    "fit_status": r.fit_status,
                    "coverage_status": r.coverage_status,
                    "delta_status": r.delta_status,
                    "overall_status": r.overall_status,
                }
            )


# Function: Writes consistency rows to CSV.

def _write_consistency_csv(path: Path, rows: Sequence[ConsistencyRow]) -> None:
    fields = [
        "branch",
        "segmentation_type",
        "n_segments_valid",
        "weighted_beta_mean",
        "chi2",
        "dof",
        "chi2_per_dof",
        "status",
        "note",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "branch": r.branch,
                    "segmentation_type": r.segmentation_type,
                    "n_segments_valid": int(r.n_segments_valid),
                    "weighted_beta_mean": float(r.weighted_beta_mean),
                    "chi2": float(r.chi2),
                    "dof": int(r.dof),
                    "chi2_per_dof": float(r.chi2_per_dof),
                    "status": r.status,
                    "note": r.note,
                }
            )


# Function: Creates segmentation audit plot (beta estimates by segment type).

def _make_plot(
    rows: Sequence[SegmentRow],
    out_pdf: Path,
    out_png: Path,
) -> Optional[str]:
    if plt is None:
        return "matplotlib_unavailable"

    df = pd.DataFrame([r.__dict__ for r in rows])
    if len(df) <= 0:
        return "no_data"

    fig, axes = plt.subplots(3, 1, figsize=(13.6, 11.6), constrained_layout=True)
    type_order = ["station", "link_type", "campaign"]
    colors = {"odf": "#1f77b4", "tnf": "#d62728"}
    for ax, tkey in zip(axes, type_order):
        sub = df.loc[df["segmentation_type"] == tkey].copy()
        if len(sub) <= 0:
            ax.set_title(f"{tkey}: no data")
            ax.grid(alpha=0.28)
            continue

        sub["label"] = sub["branch"].astype(str) + ":" + sub["segment_key"].astype(str)
        sub = sub.sort_values(["branch", "n_rows"], ascending=[True, False]).reset_index(drop=True)
        x = np.arange(len(sub), dtype=float)
        y = pd.to_numeric(sub["beta_dyn"], errors="coerce").to_numpy(dtype=float)
        s = pd.to_numeric(sub["beta_sigma"], errors="coerce").to_numpy(dtype=float)
        c = [colors.get(str(v), "#2f2f2f") for v in sub["branch"].astype(str).tolist()]

        ax.axhline(1.0, color="#7A7A7A", linestyle="--", linewidth=1.1)
        for i in range(len(sub)):
            ax.errorbar(x[i], y[i], yerr=s[i], fmt="o", capsize=3, color=c[i], alpha=0.9)

        ax.set_xticks(x, sub["label"].astype(str).tolist(), rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("beta_dyn")
        ax.set_title(f"Segmentation: {tkey}")
        ax.grid(alpha=0.28)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["odf"], label="odf", markersize=8),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["tnf"], label="tnf", markersize=8),
    ]
    axes[0].legend(handles=handles, loc="upper right", frameon=False)
    axes[0].set_title("Roadmap 8.7.48.8: Segmentation audit (station/link_type/campaign)")

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return None


# Function: Main entrypoint for roadmap step 8.7.48.8.

def main() -> int:
    ap = argparse.ArgumentParser(description="Roadmap 8.7.48.8: segmentation audit.")
    ap.add_argument("--data-root", type=str, default=str(_ROOT / "data" / "mercury" / "messenger"))
    ap.add_argument("--out-dir", type=str, default=str(_ROOT / "output" / "private" / "mercury"))
    ap.add_argument("--public-dir", type=str, default=str(_ROOT / "output" / "public" / "mercury"))
    ap.add_argument("--odf-doppler-bin-minutes", type=int, default=60)
    ap.add_argument("--odf-range-bin-minutes", type=int, default=60)
    ap.add_argument("--tnf-doppler-bin-minutes", type=int, default=30)
    ap.add_argument("--tnf-range-bin-minutes", type=int, default=30)
    ap.add_argument("--odf-min-joint-rows", type=int, default=1000)
    ap.add_argument("--tnf-min-joint-rows", type=int, default=300)
    ap.add_argument("--segment-min-joint-rows", type=int, default=60)
    ap.add_argument("--segment-min-coverage-rows", type=int, default=120)
    ap.add_argument("--segment-min-consistency-rows", type=int, default=120)
    ap.add_argument("--top-station-n", type=int, default=6)
    ap.add_argument("--top-campaign-n", type=int, default=8)
    ap.add_argument("--max-station-bias-per-channel", type=int, default=8)
    ap.add_argument("--orbital-period-days", type=float, default=87.9691)
    ap.add_argument("--sigma-watch-threshold", type=float, default=0.1)
    ap.add_argument(
        "--odf-operational-link-types",
        type=str,
        default="unknown,two-way,three-way",
        help="Comma-separated ODF link_type values for primary segmentation path.",
    )
    ap.add_argument(
        "--tnf-operational-link-types",
        type=str,
        default="",
        help="Comma-separated TNF link_type values for primary segmentation path (empty=all).",
    )
    ap.add_argument(
        "--odf-required-segmentation-types",
        type=str,
        default="station,link_type",
        help="Comma-separated segmentation types required for ODF branch status.",
    )
    ap.add_argument(
        "--tnf-required-segmentation-types",
        type=str,
        default="station",
        help="Comma-separated segmentation types required for TNF branch status.",
    )
    args = ap.parse_args()

    data_root = _resolve_path(args.data_root, _ROOT)
    out_dir = _resolve_path(args.out_dir, _ROOT)
    public_dir = _resolve_path(args.public_dir, _ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_summary_csv = out_dir / "messenger_beta_stage_h_segmentation_summary.csv"
    out_consistency_csv = out_dir / "messenger_beta_stage_h_segmentation_consistency.csv"
    out_metrics_json = out_dir / "messenger_beta_stage_h_segmentation_metrics.json"
    out_plot_pdf = out_dir / "messenger_beta_stage_h_segmentation_audit.pdf"
    out_plot_png = out_dir / "messenger_beta_stage_h_segmentation_audit.png"
    odf_operational_link_types = _parse_csv_tokens(args.odf_operational_link_types)
    tnf_operational_link_types = _parse_csv_tokens(args.tnf_operational_link_types)
    allowed_seg_types = {"station", "link_type", "campaign"}
    odf_required_seg_types = [x for x in _parse_csv_tokens(args.odf_required_segmentation_types) if x in allowed_seg_types]
    tnf_required_seg_types = [x for x in _parse_csv_tokens(args.tnf_required_segmentation_types) if x in allowed_seg_types]
    if len(odf_required_seg_types) <= 0:
        odf_required_seg_types = ["station", "link_type"]

    if len(tnf_required_seg_types) <= 0:
        tnf_required_seg_types = ["station"]

    # Branch definitions: Stage D-style ODF and Stage E-style TNF.

    branches = [
        {
            "branch": "odf",
            "doppler_csv": data_root / "derived" / "odf_doppler_observations.csv",
            "range_csv": data_root / "derived" / "odf_range_observations.csv",
            "doppler_bin": int(args.odf_doppler_bin_minutes),
            "range_bin": int(args.odf_range_bin_minutes),
            "min_rows_baseline": int(args.odf_min_joint_rows),
            "operational_link_types": odf_operational_link_types,
            "required_segmentation_types": odf_required_seg_types,
        },
        {
            "branch": "tnf",
            "doppler_csv": data_root / "derived" / "tnf_doppler_observations.csv",
            "range_csv": data_root / "derived" / "tnf_range_observations.csv",
            "doppler_bin": int(args.tnf_doppler_bin_minutes),
            "range_bin": int(args.tnf_range_bin_minutes),
            "min_rows_baseline": int(args.tnf_min_joint_rows),
            "operational_link_types": tnf_operational_link_types,
            "required_segmentation_types": tnf_required_seg_types,
        },
    ]

    missing_inputs: List[str] = []
    for cfg in branches:
        if (not cfg["doppler_csv"].exists()) or (not cfg["range_csv"].exists()):
            missing_inputs.append(str(cfg["branch"]))

    if len(missing_inputs) > 0:
        payload = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "phase_step": "8.7.48.8",
            "overall_status": "reject",
            "reason": "missing_branch_inputs",
            "missing_branches": missing_inputs,
        }
        out_metrics_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        synced = _sync_to_public([out_metrics_json], private_root=out_dir, public_root=public_dir)
        append_event(
            {
                "event": "run_script",
                "script": "scripts/mercury/messenger_beta_stage_h_segmentation_audit.py",
                "phase_step": "8.7.48.8",
                "status": "reject",
                "input": str(data_root),
                "outputs": [_safe_rel(out_metrics_json, _ROOT)],
                "metrics": {"reason": "missing_branch_inputs", "missing": missing_inputs},
            }
        )
        print(f"[warn] Stage H skipped: missing inputs for branches={missing_inputs}")
        print(f"[ok] wrote: {out_metrics_json}")
        print(f"[ok] synced_to_public={len(synced)}")
        return 0

    rows: List[SegmentRow] = []
    consistency_rows: List[ConsistencyRow] = []
    baseline_by_branch: Dict[str, Dict[str, object]] = {}
    branch_statuses: Dict[str, str] = {}
    branch_statuses_all_types: Dict[str, str] = {}
    branch_meta: Dict[str, Dict[str, object]] = {}

    for cfg in branches:
        branch = str(cfg["branch"])
        doppler = _load_branch_channel_csv(Path(cfg["doppler_csv"]), channel="doppler")
        rng = _load_branch_channel_csv(Path(cfg["range_csv"]), channel="range")
        n_rows_doppler_raw = int(len(doppler))
        n_rows_range_raw = int(len(rng))
        operational_link_types = [str(v).strip().lower() for v in cfg.get("operational_link_types", [])]
        if len(operational_link_types) > 0:
            doppler = doppler.loc[doppler["link_type"].astype(str).str.lower().isin(operational_link_types)].copy()
            rng = rng.loc[rng["link_type"].astype(str).str.lower().isin(operational_link_types)].copy()

        n_rows_doppler_filtered = int(len(doppler))
        n_rows_range_filtered = int(len(rng))
        doppler_agg = _aggregate_with_segmentation(doppler, bin_minutes=int(cfg["doppler_bin"]))
        range_agg = _aggregate_with_segmentation(rng, bin_minutes=int(cfg["range_bin"]))
        joint = pd.concat([range_agg, doppler_agg], ignore_index=True).sort_values("epoch_utc").reset_index(drop=True)

        base_fit = _run_fit(
            df_subset=joint,
            orbital_period_days=float(args.orbital_period_days),
            max_station_bias_per_channel=int(args.max_station_bias_per_channel),
            min_joint_rows=int(cfg["min_rows_baseline"]),
            sigma_watch_threshold=float(args.sigma_watch_threshold),
        )
        baseline_by_branch[branch] = base_fit
        branch_meta[branch] = {
            "n_rows_joint": int(len(joint)),
            "n_rows_range": int(np.sum(joint["channel"].astype(str).to_numpy() == "range")),
            "n_rows_doppler": int(np.sum(joint["channel"].astype(str).to_numpy() == "doppler")),
            "n_station_unique": int(joint["station_id"].astype(str).nunique()),
            "n_link_type_unique": int(joint["link_type"].astype(str).nunique()),
            "n_campaign_unique": int(joint["campaign"].astype(str).nunique()),
            "n_rows_doppler_raw": int(n_rows_doppler_raw),
            "n_rows_range_raw": int(n_rows_range_raw),
            "n_rows_doppler_filtered": int(n_rows_doppler_filtered),
            "n_rows_range_filtered": int(n_rows_range_filtered),
            "n_rows_doppler_removed_by_link_policy": int(max(0, n_rows_doppler_raw - n_rows_doppler_filtered)),
            "n_rows_range_removed_by_link_policy": int(max(0, n_rows_range_raw - n_rows_range_filtered)),
            "operational_link_types": operational_link_types,
        }

        seg_sets = _build_segmentation_sets(
            df_agg=joint,
            top_station_n=int(args.top_station_n),
            top_campaign_n=int(args.top_campaign_n),
        )
        for seg_type, seg_keys in seg_sets.items():
            seg_col = _segmentation_column(str(seg_type))
            for seg_key in seg_keys:
                sub = joint.loc[joint[seg_col].astype(str) == str(seg_key)].copy().reset_index(drop=True)
                fit = _run_fit(
                    df_subset=sub,
                    orbital_period_days=float(args.orbital_period_days),
                    max_station_bias_per_channel=int(args.max_station_bias_per_channel),
                    min_joint_rows=int(args.segment_min_joint_rows),
                    sigma_watch_threshold=float(args.sigma_watch_threshold),
                )
                z_delta = _z_delta_beta(
                    beta=float(fit["beta_dyn"]),
                    sigma=float(fit["beta_sigma"]),
                    beta_ref=float(base_fit["beta_dyn"]),
                    sigma_ref=float(base_fit["beta_sigma"]),
                )
                delta_status = _status_from_abs(z_delta, pass_thr=2.0, watch_thr=5.0)
                coverage_status = (
                    "pass"
                    if int(fit["n_rows"]) >= int(args.segment_min_coverage_rows)
                    else "watch"
                )
                overall = _combine_status(
                    [
                        str(fit["fit_status"]),
                        str(delta_status),
                        str(coverage_status),
                    ]
                )
                rows.append(
                    SegmentRow(
                        branch=branch,
                        segmentation_type=str(seg_type),
                        segment_key=str(seg_key),
                        n_rows=int(fit["n_rows"]),
                        n_range_rows=int(fit["n_range_rows"]),
                        n_doppler_rows=int(fit["n_doppler_rows"]),
                        beta_dyn=float(fit["beta_dyn"]),
                        beta_sigma=float(fit["beta_sigma"]),
                        beta_z_from_1=float(fit["beta_z_from_1"]),
                        z_delta_vs_branch=float(z_delta),
                        fit_status=str(fit["fit_status"]),
                        coverage_status=str(coverage_status),
                        delta_status=str(delta_status),
                        overall_status=str(overall),
                    )
                )

        branch_type_statuses: List[str] = []
        branch_type_required_statuses: List[str] = []
        required_seg_types = [str(x) for x in cfg.get("required_segmentation_types", []) if str(x)]
        for seg_type in ("station", "link_type", "campaign"):
            c_row = _consistency_for_type(
                branch=branch,
                segmentation_type=seg_type,
                rows=rows,
                min_consistency_rows=int(args.segment_min_consistency_rows),
            )
            consistency_rows.append(c_row)
            branch_type_statuses.append(str(c_row.status))
            if str(seg_type) in required_seg_types:
                branch_type_required_statuses.append(str(c_row.status))

        branch_statuses_all_types[branch] = _combine_status(branch_type_statuses)
        if len(branch_type_required_statuses) > 0:
            branch_statuses[branch] = _combine_status(branch_type_required_statuses)
        else:
            branch_statuses[branch] = _combine_status(branch_type_statuses)

    _write_segments_csv(out_summary_csv, rows)
    _write_consistency_csv(out_consistency_csv, consistency_rows)
    plot_note = _make_plot(rows=rows, out_pdf=out_plot_pdf, out_png=out_plot_png)

    overall = _combine_status(branch_statuses.values())
    produced: List[Path] = [out_summary_csv, out_consistency_csv, out_metrics_json]
    if plot_note is None:
        produced.extend([out_plot_pdf, out_plot_png])

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase_step": "8.7.48.8",
        "overall_status": overall,
        "branch_status": branch_statuses,
        "branch_status_all_types_diagnostic": branch_statuses_all_types,
        "baseline_by_branch": baseline_by_branch,
        "branch_meta": branch_meta,
        "segment_policy": {
            "segment_delta_pass_abs_z": 2.0,
            "segment_delta_watch_abs_z": 5.0,
            "segment_min_joint_rows": int(args.segment_min_joint_rows),
            "segment_min_coverage_rows": int(args.segment_min_coverage_rows),
            "segment_min_consistency_rows": int(args.segment_min_consistency_rows),
            "consistency_pass_chi2_dof": 3.0,
            "consistency_watch_chi2_dof": 8.0,
            "top_station_n": int(args.top_station_n),
            "top_campaign_n": int(args.top_campaign_n),
            "odf_operational_link_types": odf_operational_link_types,
            "tnf_operational_link_types": tnf_operational_link_types,
            "required_segmentation_types_by_branch": {
                "odf": odf_required_seg_types,
                "tnf": tnf_required_seg_types,
            },
            "diagnostic_segmentation_types_by_branch": {
                "odf": [x for x in ("station", "link_type", "campaign") if x not in set(odf_required_seg_types)],
                "tnf": [x for x in ("station", "link_type", "campaign") if x not in set(tnf_required_seg_types)],
            },
        },
        "counts": {
            "n_segment_rows": int(len(rows)),
            "n_consistency_rows": int(len(consistency_rows)),
        },
        "plot": "generated" if plot_note is None else str(plot_note),
        "outputs_private": [_safe_rel(p, _ROOT) for p in produced if p != out_metrics_json],
    }
    out_metrics_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    synced = _sync_to_public(produced, private_root=out_dir, public_root=public_dir)
    payload["outputs_public"] = [_safe_rel(p, _ROOT) for p in synced if p.name != out_metrics_json.name]
    out_metrics_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _sync_to_public([out_metrics_json], private_root=out_dir, public_root=public_dir)

    append_event(
        {
            "event": "run_script",
            "script": "scripts/mercury/messenger_beta_stage_h_segmentation_audit.py",
            "phase_step": "8.7.48.8",
            "status": overall,
            "input": str(data_root),
            "outputs": [_safe_rel(p, _ROOT) for p in produced],
            "metrics": {
                "branch_status": branch_statuses,
                "branch_status_all_types_diagnostic": branch_statuses_all_types,
                "n_segment_rows": int(len(rows)),
                "n_consistency_rows": int(len(consistency_rows)),
            },
        }
    )

    print(f"[ok] stage_h_overall={overall}")
    print(f"[ok] branch_status={branch_statuses}")
    print(f"[ok] n_segment_rows={len(rows)} n_consistency_rows={len(consistency_rows)}")
    print(f"[ok] wrote: {out_summary_csv}")
    print(f"[ok] wrote: {out_metrics_json}")
    if plot_note is None:
        print(f"[ok] wrote: {out_plot_pdf}")
        print(f"[ok] wrote: {out_plot_png}")
    else:
        print(f"[warn] plot skipped: {plot_note}")

    print(f"[ok] synced_to_public={len(synced)}")
    return 0


# Condition: Executes CLI main routine.

if __name__ == "__main__":
    raise SystemExit(main())
