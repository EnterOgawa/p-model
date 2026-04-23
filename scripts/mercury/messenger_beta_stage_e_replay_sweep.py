#!/usr/bin/env python3
"""
messenger_beta_stage_e_replay_sweep.py

Roadmap Step 8.7.48 follow-up utility for Stage E replay watch reduction.

Purpose:
- Sweep Stage E replay fit hyper-parameters over existing TNF-derived CSVs.
- Evaluate ODF-vs-TNF replay consistency (z_delta_beta) and beta sigma.
- Emit machine-readable summary and recommended configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.mercury.messenger_beta_stage_d_joint_fit import (
    _aggregate_channel,
    _build_design_matrix,
    _fit_joint,
    _load_channel_csv,
    _sync_to_public,
)
from scripts.mercury.messenger_beta_stage_e_tnf_replay import _compare_with_odf
from scripts.summary.worklog import append_event


# Class: Holds one replay sweep trial result row.
@dataclass
class SweepRow:
    doppler_bin_minutes: int
    range_bin_minutes: int
    min_joint_rows: int
    max_station_bias_per_channel: int
    n_rows_joint: int
    n_rows_range: int
    n_rows_doppler: int
    beta_dyn: float
    beta_sigma: float
    beta_z_from_1: float
    status_data: str
    status_sigma: str
    replay_status: str
    replay_dyn_status: str
    replay_lt_status: str
    replay_z_delta_beta: float
    replay_dyn_z_delta_beta: float
    replay_lt_z_delta_beta: float
    replay_delta_beta: float
    score: float


# Function: Converts path to repo-relative string when possible.

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


# Function: Parses comma-separated integer lists for sweep parameters.

def _parse_int_list(text: str) -> List[int]:
    out: List[int] = []
    for tok in str(text).split(","):
        s = str(tok).strip()
        if not s:
            continue

        try:
            out.append(int(s))
        except Exception:
            continue

    uniq = sorted(set(out))
    return uniq


# Function: Maps status to numeric rank for stable trial scoring.

def _status_rank(status: str) -> int:
    t = str(status).strip().lower()
    if t == "pass":
        return 0

    if t == "watch":
        return 1

    return 2


# Function: Iterates all sweep parameter combinations.

def _iter_grid(
    doppler_bins: Sequence[int],
    range_bins: Sequence[int],
    min_rows_list: Sequence[int],
    bias_caps: Sequence[int],
) -> Iterable[tuple[int, int, int, int]]:
    for db in doppler_bins:
        for rb in range_bins:
            for mr in min_rows_list:
                for cap in bias_caps:
                    yield (int(db), int(rb), int(mr), int(cap))


# Function: Writes sweep rows into CSV for audit and reproducibility.

def _write_rows_csv(path: Path, rows: Sequence[SweepRow]) -> None:
    fields = [
        "doppler_bin_minutes",
        "range_bin_minutes",
        "min_joint_rows",
        "max_station_bias_per_channel",
        "n_rows_joint",
        "n_rows_range",
        "n_rows_doppler",
        "beta_dyn",
        "beta_sigma",
        "beta_z_from_1",
        "status_data",
        "status_sigma",
        "replay_status",
        "replay_dyn_status",
        "replay_lt_status",
        "replay_z_delta_beta",
        "replay_dyn_z_delta_beta",
        "replay_lt_z_delta_beta",
        "replay_delta_beta",
        "score",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "doppler_bin_minutes": int(r.doppler_bin_minutes),
                    "range_bin_minutes": int(r.range_bin_minutes),
                    "min_joint_rows": int(r.min_joint_rows),
                    "max_station_bias_per_channel": int(r.max_station_bias_per_channel),
                    "n_rows_joint": int(r.n_rows_joint),
                    "n_rows_range": int(r.n_rows_range),
                    "n_rows_doppler": int(r.n_rows_doppler),
                    "beta_dyn": float(r.beta_dyn),
                    "beta_sigma": float(r.beta_sigma),
                    "beta_z_from_1": float(r.beta_z_from_1),
                    "status_data": str(r.status_data),
                    "status_sigma": str(r.status_sigma),
                    "replay_status": str(r.replay_status),
                    "replay_dyn_status": str(r.replay_dyn_status),
                    "replay_lt_status": str(r.replay_lt_status),
                    "replay_z_delta_beta": float(r.replay_z_delta_beta),
                    "replay_dyn_z_delta_beta": float(r.replay_dyn_z_delta_beta),
                    "replay_lt_z_delta_beta": float(r.replay_lt_z_delta_beta),
                    "replay_delta_beta": float(r.replay_delta_beta),
                    "score": float(r.score),
                }
            )


# Function: Main entrypoint for Stage E replay sweep.

def main() -> int:
    ap = argparse.ArgumentParser(description="Roadmap 8.7.48: Stage E replay hyper-parameter sweep.")
    ap.add_argument("--data-root", type=str, default=str(_ROOT / "data" / "mercury" / "messenger"))
    ap.add_argument(
        "--odf-stage-d-metrics",
        type=str,
        default=str(_ROOT / "output" / "public" / "mercury" / "messenger_beta_stage_d_joint_metrics.json"),
    )
    ap.add_argument("--out-dir", type=str, default=str(_ROOT / "output" / "private" / "mercury"))
    ap.add_argument("--public-dir", type=str, default=str(_ROOT / "output" / "public" / "mercury"))
    ap.add_argument("--doppler-bin-list", type=str, default="20,30,45,60")
    ap.add_argument("--range-bin-list", type=str, default="20,30,45")
    ap.add_argument("--min-joint-rows-list", type=str, default="200,300,400")
    ap.add_argument("--station-bias-cap-list", type=str, default="6,8,10,12")
    ap.add_argument("--orbital-period-days", type=float, default=87.9691)
    ap.add_argument("--sigma-watch-threshold", type=float, default=0.1)
    ap.add_argument(
        "--beta-split-mode",
        type=str,
        choices=("auto", "coupled", "split"),
        default="auto",
        help="Beta split mode for replay sweep. 'auto' follows ODF Stage D metrics.",
    )
    args = ap.parse_args()

    data_root = _resolve_path(args.data_root, _ROOT)
    out_dir = _resolve_path(args.out_dir, _ROOT)
    public_dir = _resolve_path(args.public_dir, _ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    derived_root = data_root / "derived"
    tnf_doppler_csv = derived_root / "tnf_doppler_observations.csv"
    tnf_range_csv = derived_root / "tnf_range_observations.csv"
    odf_stage_d_metrics = _resolve_path(args.odf_stage_d_metrics, _ROOT)

    out_summary_csv = out_dir / "messenger_beta_stage_e_replay_sweep_summary.csv"
    out_metrics_json = out_dir / "messenger_beta_stage_e_replay_sweep_metrics.json"

    if (not tnf_doppler_csv.exists()) or (not tnf_range_csv.exists()):
        payload = {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "phase_step": "8.7.48.5_sweep",
            "overall_status": "reject",
            "reason": "tnf_input_missing",
            "tnf_doppler_csv": _safe_rel(tnf_doppler_csv, _ROOT),
            "tnf_range_csv": _safe_rel(tnf_range_csv, _ROOT),
        }
        out_metrics_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        synced = _sync_to_public([out_metrics_json], private_root=out_dir, public_root=public_dir)
        print("[warn] TNF input CSV missing.")
        print(f"[ok] wrote: {out_metrics_json}")
        print(f"[ok] synced_to_public={len(synced)}")
        return 0

    doppler_bins = _parse_int_list(args.doppler_bin_list)
    range_bins = _parse_int_list(args.range_bin_list)
    min_rows_list = _parse_int_list(args.min_joint_rows_list)
    bias_caps = _parse_int_list(args.station_bias_cap_list)
    if len(doppler_bins) <= 0 or len(range_bins) <= 0 or len(min_rows_list) <= 0 or len(bias_caps) <= 0:
        raise ValueError("sweep lists must contain at least one integer.")

    odf_metrics: Dict[str, object] = {}
    if odf_stage_d_metrics.exists():
        odf_metrics = json.loads(odf_stage_d_metrics.read_text(encoding="utf-8"))

    req_split_mode = str(args.beta_split_mode).strip().lower()
    if req_split_mode == "auto":
        odf_split_mode = str(odf_metrics.get("beta_split_mode", "coupled")).strip().lower()
        beta_split_mode = odf_split_mode if odf_split_mode in {"coupled", "split"} else "coupled"
    else:
        beta_split_mode = req_split_mode

    doppler_df = _load_channel_csv(tnf_doppler_csv, channel="doppler")
    range_df = _load_channel_csv(tnf_range_csv, channel="range")

    rows: List[SweepRow] = []
    for db, rb, mr, cap in _iter_grid(doppler_bins, range_bins, min_rows_list, bias_caps):
        doppler_agg = _aggregate_channel(doppler_df, bin_minutes=int(db))
        range_agg = _aggregate_channel(range_df, bin_minutes=int(rb))
        joint_df = pd.concat([range_agg, doppler_agg], ignore_index=True).sort_values("epoch_utc").reset_index(drop=True)

        X, y_norm, y_obs, labels, _meta, work = _build_design_matrix(
            joint_df,
            orbital_period_days=float(args.orbital_period_days),
            max_station_bias_per_channel=int(cap),
            split_beta_lt=(beta_split_mode == "split"),
        )
        channels = work["channel"].astype(str).to_numpy()
        fit, _coef, _fit_norm, _residual_norm = _fit_joint(
            X=X,
            y_norm=y_norm,
            y_obs=y_obs,
            scale_by_row=work["scale_by_row"].to_numpy(dtype=float),
            labels=labels,
            channels=channels,
            min_rows=int(mr),
            sigma_watch_threshold=float(args.sigma_watch_threshold),
        )
        replay_cmp = _compare_with_odf(
            tnf_beta_dyn=float(fit.beta_dyn),
            tnf_sigma_dyn=float(fit.beta_sigma),
            tnf_beta_lt=float(fit.beta_lt),
            tnf_sigma_lt=float(fit.beta_lt_sigma),
            beta_split_mode=str(fit.beta_split_mode),
            odf_metrics=odf_metrics,
        )
        replay_status = str(replay_cmp.get("status", "watch"))
        replay_dyn_status = str(replay_cmp.get("replay_vs_odf_beta_dyn", {}).get("status", "watch"))
        replay_lt_status = str(replay_cmp.get("replay_vs_odf_beta_lt", {}).get("status", "watch"))
        replay_dyn_z = float(replay_cmp.get("replay_vs_odf_beta_dyn", {}).get("z_delta_beta", float("nan")))
        replay_lt_z = float(replay_cmp.get("replay_vs_odf_beta_lt", {}).get("z_delta_beta", float("nan")))
        replay_z = float(replay_cmp.get("z_delta_beta", replay_dyn_z))
        replay_delta = float(replay_cmp.get("delta_beta", float("nan")))

        # Prefer pass replay status on beta_dyn, then smaller beta_dyn replay z, then smaller beta sigma.
        score = float(_status_rank(replay_dyn_status) * 1.0e6)
        if math.isfinite(replay_dyn_z):
            score += float(abs(replay_dyn_z) * 1.0e3)
        else:
            score += 9.9e5

        if math.isfinite(float(fit.beta_sigma)):
            score += float(abs(fit.beta_sigma))
        else:
            score += 9.9e5

        rows.append(
            SweepRow(
                doppler_bin_minutes=int(db),
                range_bin_minutes=int(rb),
                min_joint_rows=int(mr),
                max_station_bias_per_channel=int(cap),
                n_rows_joint=int(fit.n_rows),
                n_rows_range=int(fit.n_range_rows),
                n_rows_doppler=int(fit.n_doppler_rows),
                beta_dyn=float(fit.beta_dyn),
                beta_sigma=float(fit.beta_sigma),
                beta_z_from_1=float(fit.beta_z_from_1),
                status_data=str(fit.status_data),
                status_sigma=str(fit.status_sigma),
                replay_status=replay_status,
                replay_dyn_status=replay_dyn_status,
                replay_lt_status=replay_lt_status,
                replay_z_delta_beta=float(replay_z),
                replay_dyn_z_delta_beta=float(replay_dyn_z),
                replay_lt_z_delta_beta=float(replay_lt_z),
                replay_delta_beta=float(replay_delta),
                score=float(score),
            )
        )

    rows_sorted = sorted(rows, key=lambda r: (r.score, abs(r.replay_z_delta_beta) if math.isfinite(r.replay_z_delta_beta) else 1.0e9))
    _write_rows_csv(out_summary_csv, rows_sorted)

    best = rows_sorted[0] if len(rows_sorted) > 0 else None
    if best is None:
        overall = "reject"
    else:
        if str(best.replay_status) == "pass":
            overall = "pass"
        elif str(best.replay_status) == "watch":
            overall = "watch"
        else:
            overall = "reject"

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase_step": "8.7.48.5_sweep",
        "overall_status": overall,
        "inputs": {
            "tnf_doppler_csv": _safe_rel(tnf_doppler_csv, _ROOT),
            "tnf_range_csv": _safe_rel(tnf_range_csv, _ROOT),
            "odf_stage_d_metrics": _safe_rel(odf_stage_d_metrics, _ROOT),
            "beta_split_mode": str(beta_split_mode),
        },
        "grid": {
            "doppler_bin_list": doppler_bins,
            "range_bin_list": range_bins,
            "min_joint_rows_list": min_rows_list,
            "station_bias_cap_list": bias_caps,
        },
        "counts": {
            "n_trials": int(len(rows_sorted)),
            "replay_status_counts": {
                "pass": int(sum(1 for r in rows_sorted if str(r.replay_status) == "pass")),
                "watch": int(sum(1 for r in rows_sorted if str(r.replay_status) == "watch")),
                "reject": int(sum(1 for r in rows_sorted if str(r.replay_status) == "reject")),
            },
        },
        "best_trial": (best.__dict__ if best is not None else None),
        "recommended_stage_e_args": (
            {
                "doppler_bin_minutes": int(best.doppler_bin_minutes),
                "range_bin_minutes": int(best.range_bin_minutes),
                "min_joint_rows": int(best.min_joint_rows),
                "max_station_bias_per_channel": int(best.max_station_bias_per_channel),
            }
            if best is not None
            else {}
        ),
        "outputs_private": [_safe_rel(out_summary_csv, _ROOT)],
    }
    out_metrics_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    synced = _sync_to_public([out_summary_csv, out_metrics_json], private_root=out_dir, public_root=public_dir)
    payload["outputs_public"] = [_safe_rel(p, _ROOT) for p in synced]
    out_metrics_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _sync_to_public([out_metrics_json], private_root=out_dir, public_root=public_dir)

    append_event(
        {
            "event": "run_script",
            "script": "scripts/mercury/messenger_beta_stage_e_replay_sweep.py",
            "phase_step": "8.7.48.5_sweep",
            "status": overall,
            "input": f"{_safe_rel(tnf_doppler_csv, _ROOT)}|{_safe_rel(tnf_range_csv, _ROOT)}",
            "outputs": [_safe_rel(out_summary_csv, _ROOT), _safe_rel(out_metrics_json, _ROOT)],
            "metrics": {
                "n_trials": int(len(rows_sorted)),
                "best_replay_status": (str(best.replay_status) if best is not None else "n/a"),
                "best_replay_z": (float(best.replay_z_delta_beta) if best is not None else float("nan")),
            },
        }
    )

    print(f"[ok] stage_e_sweep_overall={overall}")
    if best is not None:
        print(
            "[ok] best_trial replay_status={} replay_z={:.6f} db={} rb={} min_rows={} bias_cap={}".format(
                best.replay_status,
                float(best.replay_z_delta_beta),
                int(best.doppler_bin_minutes),
                int(best.range_bin_minutes),
                int(best.min_joint_rows),
                int(best.max_station_bias_per_channel),
            )
        )

    print(f"[ok] wrote: {out_summary_csv}")
    print(f"[ok] wrote: {out_metrics_json}")
    print(f"[ok] synced_to_public={len(synced)}")
    return 0


# Condition: Executes CLI main routine.

if __name__ == "__main__":
    raise SystemExit(main())
