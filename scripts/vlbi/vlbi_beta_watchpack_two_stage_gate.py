#!/usr/bin/env python3
"""
vlbi_beta_watchpack_two_stage_gate.py

Two-stage gate for session-level watchpack decisions.

Stage 1 (all-sky gain):
- Exclusion candidate must improve all-sky high-sensitivity chi2/dof.

Stage 2 (stable/timeband non-regression):
- The same exclusion must not worsen stable-source or timeband consistency.

This script fixes a machine-readable decision:
- selected_policy: keep_all or exclude_session:<SESSION>
- selected_scenario: keep_all or drop_<SESSION>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence


# Function: Resolve repository root from script path.
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# Function: Read JSON payload as dictionary.

def _read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"json not found: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"json root is not an object: {path}")

    return payload


# Function: Build scenario key for a dropped session.

def _drop_scenario(session: str) -> str:
    return f"drop_{str(session).strip()}"


# Function: Extract one scenario row from rows list.

def _find_scenario_row(rows: List[Dict[str, object]], scenario: str) -> Dict[str, object]:
    found = [r for r in rows if str(r.get("scenario") or "").strip() == scenario]
    if not found:
        raise RuntimeError(f"scenario row not found: {scenario}")

    return found[0]


# Function: Convert arbitrary numeric field to float.

def _as_float(row: Dict[str, object], key: str) -> float:
    val = row.get(key, math.nan)
    try:
        return float(val)  # type: ignore[arg-type]
    except Exception:
        return math.nan


# Function: Sync generated files to output/public/vlbi.

def _sync_public(root: Path, outputs: Sequence[Path]) -> None:
    dst = root / "output" / "public" / "vlbi"
    dst.mkdir(parents=True, exist_ok=True)
    for p in outputs:
        if p.exists():
            shutil.copy2(p, dst / p.name)


# Function: Render a compact decision plot.

def _plot_decision(pdf_path: Path, png_path: Path, values: Dict[str, float]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    labels = ["allsky_rel_improve", "stable_delta", "timeband_delta"]
    vals = [
        float(values.get("allsky_relative_improvement", math.nan)),
        float(values.get("stable_delta_chi2_dof", math.nan)),
        float(values.get("timeband_delta_chi2_dof", math.nan)),
    ]
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    colors = ["tab:green", "tab:blue", "tab:orange"]
    ax.bar(labels, vals, color=colors, alpha=0.88)
    ax.axhline(0.0, color="tab:gray", linestyle="--", linewidth=1.0)
    ax.set_ylabel("value")
    ax.set_title("Two-stage gate diagnostics")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(str(pdf_path))
    fig.savefig(str(png_path), dpi=170)
    plt.close(fig)


# Function: Main entrypoint for two-stage gate decision.

def main() -> int:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="Fix two-stage gate for session watchpack exclusion decisions.")
    ap.add_argument(
        "--watchpack-comparison-metrics",
        type=Path,
        default=root / "output" / "public" / "vlbi" / "vlbi_beta_watchpack_condition_refit_comparison_metrics.json",
        help="Comparison metrics from step 8.7.46.21.",
    )
    ap.add_argument(
        "--threshold-comparison-metrics",
        type=Path,
        default=root / "output" / "public" / "vlbi" / "vlbi_high_sensitivity_threshold_sweep_condition_comparison_metrics.json",
        help="Threshold comparison metrics from step 8.7.46.22.",
    )
    ap.add_argument(
        "--candidate-session",
        type=str,
        default="20MAY04XA",
        help="Exclusion candidate session label for two-stage gate.",
    )
    ap.add_argument(
        "--min-allsky-relative-improvement",
        type=float,
        default=0.05,
        help="Minimum relative all-sky chi2/dof improvement required in stage 1.",
    )
    ap.add_argument(
        "--max-stable-delta-chi2-dof",
        type=float,
        default=0.0,
        help="Maximum allowed stable chi2/dof delta versus keep_all in stage 2.",
    )
    ap.add_argument(
        "--max-timeband-delta-chi2-dof",
        type=float,
        default=0.0,
        help="Maximum allowed timeband chi2/dof delta versus keep_all in stage 2.",
    )
    args = ap.parse_args()

    watchpack_payload = _read_json(args.watchpack_comparison_metrics.resolve())
    threshold_payload = _read_json(args.threshold_comparison_metrics.resolve())
    watch_rows = list(watchpack_payload.get("rows") or [])
    thr_rows = list(threshold_payload.get("rows") or [])
    if not watch_rows or not thr_rows:
        raise RuntimeError("comparison metrics rows are empty.")

    keep_watch = _find_scenario_row(watch_rows, "keep_all")
    keep_thr = _find_scenario_row(thr_rows, "keep_all")
    candidate = str(args.candidate_session).strip()
    drop_scenario = _drop_scenario(candidate)
    drop_watch = _find_scenario_row(watch_rows, drop_scenario)
    drop_thr = _find_scenario_row(thr_rows, drop_scenario)

    keep_thr_chi = _as_float(keep_thr, "recommended_chi2_dof")
    drop_thr_chi = _as_float(drop_thr, "recommended_chi2_dof")
    allsky_rel_improvement = (
        float((keep_thr_chi - drop_thr_chi) / keep_thr_chi)
        if (math.isfinite(keep_thr_chi) and keep_thr_chi > 0.0 and math.isfinite(drop_thr_chi))
        else math.nan
    )
    stable_delta = _as_float(drop_watch, "stable_chi2_dof") - _as_float(keep_watch, "stable_chi2_dof")
    timeband_delta = _as_float(drop_watch, "timeband_session_chi2_dof") - _as_float(
        keep_watch, "timeband_session_chi2_dof"
    )
    keep_status = str(keep_watch.get("stable_status") or "")
    drop_status = str(drop_watch.get("stable_status") or "")
    keep_time_status = str(keep_watch.get("timeband_session_status") or "")
    drop_time_status = str(drop_watch.get("timeband_session_status") or "")

    stage1_pass = bool(
        math.isfinite(allsky_rel_improvement)
        and (allsky_rel_improvement >= float(args.min_allsky_relative_improvement))
    )
    stage2_pass = bool(
        math.isfinite(stable_delta)
        and math.isfinite(timeband_delta)
        and (stable_delta <= float(args.max_stable_delta_chi2_dof))
        and (timeband_delta <= float(args.max_timeband_delta_chi2_dof))
    )
    status_regression = bool(
        (keep_status in {"pass", "watch"} and drop_status == "reject")
        or (keep_time_status in {"pass", "watch"} and drop_time_status == "reject")
    )
    overall_pass = bool(stage1_pass and stage2_pass and (not status_regression))
    selected_scenario = drop_scenario if overall_pass else "keep_all"
    selected_policy = f"exclude_session:{candidate}" if overall_pass else "keep_all"

    selected_watch = _find_scenario_row(watch_rows, selected_scenario)
    selected_thr = _find_scenario_row(thr_rows, selected_scenario)
    selected_allsky_csv = str(selected_watch.get("allsky_summary_csv") or "")
    selected_threshold_ns = _as_float(selected_thr, "recommended_threshold_ns")

    out_dir = root / "output" / "vlbi"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "vlbi_beta_watchpack_two_stage_gate_summary.csv"
    metrics_path = out_dir / "vlbi_beta_watchpack_two_stage_gate_metrics.json"
    plot_pdf = out_dir / "vlbi_beta_watchpack_two_stage_gate.pdf"
    plot_png = out_dir / "vlbi_beta_watchpack_two_stage_gate.png"

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        cols = [
            "candidate_session",
            "drop_scenario",
            "allsky_relative_improvement",
            "stable_delta_chi2_dof",
            "timeband_delta_chi2_dof",
            "stage1_pass",
            "stage2_pass",
            "status_regression",
            "overall_pass",
            "selected_scenario",
            "selected_policy",
            "selected_allsky_summary_csv",
            "selected_threshold_ns",
        ]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerow(
            {
                "candidate_session": candidate,
                "drop_scenario": drop_scenario,
                "allsky_relative_improvement": f"{allsky_rel_improvement:.16e}",
                "stable_delta_chi2_dof": f"{stable_delta:.16e}",
                "timeband_delta_chi2_dof": f"{timeband_delta:.16e}",
                "stage1_pass": str(stage1_pass).lower(),
                "stage2_pass": str(stage2_pass).lower(),
                "status_regression": str(status_regression).lower(),
                "overall_pass": str(overall_pass).lower(),
                "selected_scenario": selected_scenario,
                "selected_policy": selected_policy,
                "selected_allsky_summary_csv": selected_allsky_csv,
                "selected_threshold_ns": f"{selected_threshold_ns:.16e}",
            }
        )

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "step": "8.7.46.23",
        "candidate_session": candidate,
        "drop_scenario": drop_scenario,
        "gates": {
            "stage1_allsky": {
                "metric": "allsky_relative_improvement",
                "value": float(allsky_rel_improvement),
                "threshold_min": float(args.min_allsky_relative_improvement),
                "pass": bool(stage1_pass),
            },
            "stage2_stable_timeband": {
                "stable_delta_chi2_dof": float(stable_delta),
                "timeband_delta_chi2_dof": float(timeband_delta),
                "stable_threshold_max": float(args.max_stable_delta_chi2_dof),
                "timeband_threshold_max": float(args.max_timeband_delta_chi2_dof),
                "pass": bool(stage2_pass),
            },
            "status_regression": {
                "keep_stable_status": keep_status,
                "drop_stable_status": drop_status,
                "keep_timeband_status": keep_time_status,
                "drop_timeband_status": drop_time_status,
                "regression": bool(status_regression),
            },
        },
        "decision": {
            "overall_pass": bool(overall_pass),
            "selected_scenario": selected_scenario,
            "selected_policy": selected_policy,
            "selected_allsky_summary_csv": selected_allsky_csv,
            "selected_threshold_ns": float(selected_threshold_ns),
            "reason": (
                "exclude_candidate_passed_two_stage_gate"
                if overall_pass
                else "exclude_candidate_failed_stable_timeband_non_regression_gate"
            ),
        },
        "references": {
            "watchpack_comparison_metrics": str(args.watchpack_comparison_metrics.resolve()),
            "threshold_comparison_metrics": str(args.threshold_comparison_metrics.resolve()),
        },
        "outputs": {
            "summary_csv": str(csv_path.resolve()),
            "metrics_json": str(metrics_path.resolve()),
            "plot_pdf": str(plot_pdf.resolve()),
            "plot_png": str(plot_png.resolve()),
        },
    }
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    _plot_decision(
        pdf_path=plot_pdf,
        png_path=plot_png,
        values={
            "allsky_relative_improvement": float(allsky_rel_improvement),
            "stable_delta_chi2_dof": float(stable_delta),
            "timeband_delta_chi2_dof": float(timeband_delta),
        },
    )

    outputs = [csv_path, metrics_path, plot_pdf, plot_png]
    _sync_public(root, outputs)
    for p in outputs:
        print(f"Wrote: {p}")

    print("Synced:", root / "output" / "public" / "vlbi")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

