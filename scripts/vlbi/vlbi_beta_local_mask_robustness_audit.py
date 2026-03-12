#!/usr/bin/env python3
"""
vlbi_beta_local_mask_robustness_audit.py

Robustness audit for local-mask two-stage selection under expanded top-N candidates.

This script:
1) Runs `vlbi_beta_local_mask_two_stage_sweep.py` for multiple (top_sources, top_time_quartiles) combos.
2) Snapshots per-combo sweep artifacts with explicit suffixes.
3) Compares selected scenario stability across combos and emits a robustness summary.
4) Restores canonical chain by re-applying the pre-audit gate metrics decision.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(ROOT) not in sys.path` を満たす場合のみ検索パスへ追加する。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summary.worklog import append_event


# Function: Resolve repository root from script path.
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# Function: Parse combo tokens formatted as "<top_sources>x<top_time_quartiles>".

def _parse_combos(text: str) -> List[Tuple[int, int]]:
    combos: List[Tuple[int, int]] = []
    for raw in [t.strip() for t in str(text).split(",") if t.strip()]:
        m = re.fullmatch(r"(\d+)\s*[xX]\s*(\d+)", raw)
        if not m:
            raise ValueError(f"invalid combo token: {raw!r} (expected format: 3x2)")

        s = int(m.group(1))
        q = int(m.group(2))
        if s <= 0 or q <= 0:
            raise ValueError(f"combo values must be positive: {raw!r}")

        combos.append((s, q))

    if not combos:
        raise ValueError("no valid combos provided")

    return combos


# Function: Execute subprocess command and capture short diagnostics.

def _run(cmd: Sequence[str], cwd: Path) -> Dict[str, object]:
    cp = subprocess.run(
        list(cmd),
        cwd=str(cwd),
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "cmd": list(cmd),
        "returncode": int(cp.returncode),
        "ok": bool(cp.returncode == 0),
        "stdout_tail": "\n".join((cp.stdout or "").splitlines()[-12:]),
        "stderr_tail": "\n".join((cp.stderr or "").splitlines()[-12:]),
    }


# Function: Read JSON file from disk.

def _read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


# Function: Convert unknown value to finite float or NaN.

def _to_float(value: object, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


# Function: Copy one file if present.

def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False

    try:
        if src.resolve() == dst.resolve():
            return True
    except OSError:
        pass

    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


# Function: Snapshot local-mask sweep outputs for one combo tag.

def _snapshot_combo_outputs(root: Path, combo_tag: str) -> Dict[str, str]:
    private_dir = root / "output" / "vlbi"
    public_dir = root / "output" / "public" / "vlbi"
    names = (
        "vlbi_beta_local_mask_two_stage_sweep_summary.csv",
        "vlbi_beta_local_mask_two_stage_sweep_metrics.json",
        "vlbi_beta_local_mask_two_stage_sweep.pdf",
        "vlbi_beta_local_mask_two_stage_sweep.png",
    )
    copied: Dict[str, str] = {}
    for name in names:
        src_public = public_dir / name
        suffix_name = name.replace("vlbi_beta_local_mask_two_stage_sweep", f"vlbi_beta_local_mask_two_stage_sweep_{combo_tag}")
        dst_public = public_dir / suffix_name
        dst_private = private_dir / suffix_name
        if _copy_if_exists(src_public, dst_public):
            _copy_if_exists(src_public, dst_private)
            copied[name] = str(dst_public)

    return copied


# Function: Build per-combo summary row from metrics JSON.

def _build_combo_row(top_sources: int, top_time_quartiles: int, metrics: Dict[str, object]) -> Dict[str, object]:
    decision = metrics.get("decision", {})
    rows = metrics.get("rows", [])
    selected = str((decision or {}).get("selected_scenario") or "")
    selected_row: Dict[str, object] = {}
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and str(row.get("scenario") or "") == selected:
                selected_row = row
                break

    pass_count = 0
    if isinstance(rows, list):
        pass_count = sum(1 for row in rows if isinstance(row, dict) and bool(row.get("overall_pass")))

    return {
        "combo_tag": f"ts{top_sources}_tq{top_time_quartiles}",
        "top_sources": int(top_sources),
        "top_time_quartiles": int(top_time_quartiles),
        "selected_scenario": selected,
        "selected_policy": str((decision or {}).get("selected_policy") or ""),
        "selected_removed_fraction": _to_float(selected_row.get("removed_fraction")),
        "selected_allsky_relative_improvement": _to_float(selected_row.get("allsky_relative_improvement")),
        "selected_stable_chi2_dof": _to_float(selected_row.get("stable_chi2_dof")),
        "selected_timeband_chi2_dof": _to_float(selected_row.get("timeband_chi2_dof")),
        "selected_threshold_chi2_dof": _to_float(selected_row.get("threshold_chi2_dof")),
        "n_candidates": int((metrics.get("candidate_generation", {}) or {}).get("n_candidates") or 0),
        "n_pass_candidates": int(pass_count),
    }


# Function: Write CSV summary rows.

def _write_summary_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    cols = [
        "combo_tag",
        "top_sources",
        "top_time_quartiles",
        "selected_scenario",
        "selected_policy",
        "selected_removed_fraction",
        "selected_allsky_relative_improvement",
        "selected_stable_chi2_dof",
        "selected_timeband_chi2_dof",
        "selected_threshold_chi2_dof",
        "n_candidates",
        "n_pass_candidates",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in cols})


# Function: Generate a PDF/PNG plot for robustness overview.

def _plot_summary(pdf_path: Path, png_path: Path, rows: Sequence[Dict[str, object]]) -> None:
    labels = [str(r.get("combo_tag") or "") for r in rows]
    improvements = [_to_float(r.get("selected_allsky_relative_improvement")) for r in rows]
    removed = [_to_float(r.get("selected_removed_fraction")) for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(11.0, 7.5), constrained_layout=True)
    x = list(range(len(labels)))

    ax0 = axes[0]
    ax0.plot(x, improvements, marker="o", linewidth=1.6, color="#0B5394")
    ax0.axhline(0.05, color="#CC0000", linestyle="--", linewidth=1.0)
    ax0.set_ylabel("allsky relative improvement")
    ax0.set_title("Local-mask top-N robustness audit")
    ax0.set_xticks(x, labels, rotation=0)
    ax0.grid(alpha=0.25, linestyle="--", linewidth=0.6)

    ax1 = axes[1]
    ax1.bar(x, removed, color="#6AA84F")
    ax1.set_ylabel("selected removed fraction")
    ax1.set_xlabel("top-N combo")
    ax1.set_xticks(x, labels, rotation=0)
    ax1.grid(alpha=0.25, linestyle="--", linewidth=0.6)

    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=160)
    plt.close(fig)


# Function: Sync generated artifacts to public directory.

def _sync_public(root: Path, outputs: Sequence[Path]) -> None:
    public_dir = root / "output" / "public" / "vlbi"
    private_dir = root / "output" / "vlbi"
    public_dir.mkdir(parents=True, exist_ok=True)
    for out in outputs:
        src = out if out.exists() else (private_dir / out.name)
        if not src.exists():
            continue

        if src.parent.resolve() == public_dir.resolve():
            _copy_if_exists(src, private_dir / src.name)
            continue

        _copy_if_exists(src, public_dir / src.name)
        _copy_if_exists(src, private_dir / src.name)


# Function: Main entrypoint for 8.7.46.27 robustness audit.

def main() -> int:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="Robustness audit of local-mask two-stage selection under top-N expansion.")
    ap.add_argument(
        "--combos",
        type=str,
        default="3x2,5x4",
        help="Comma-separated top-N combos in '<sources>x<time_quartiles>' format.",
    )
    ap.add_argument(
        "--restore-chain",
        action="store_true",
        help="Restore canonical chain using pre-audit gate metrics after all combo runs.",
    )
    ap.add_argument(
        "--skip-runs",
        action="store_true",
        help="Skip running sweeps and only summarize existing combo snapshots.",
    )
    args = ap.parse_args()

    combos = _parse_combos(args.combos)
    script_sweep = root / "scripts" / "vlbi" / "vlbi_beta_local_mask_two_stage_sweep.py"
    script_apply = root / "scripts" / "vlbi" / "vlbi_beta_watchpack_apply_chain.py"
    out_private = root / "output" / "vlbi"
    out_public = root / "output" / "public" / "vlbi"
    out_private.mkdir(parents=True, exist_ok=True)
    out_public.mkdir(parents=True, exist_ok=True)

    canonical_metrics = out_public / "vlbi_beta_local_mask_two_stage_sweep_metrics.json"
    if not canonical_metrics.exists():
        raise FileNotFoundError(f"missing canonical local-mask metrics: {canonical_metrics}")

    pre_audit_metrics = out_private / "vlbi_beta_local_mask_two_stage_sweep_metrics_pre_robustness.json"
    shutil.copy2(canonical_metrics, pre_audit_metrics)

    run_logs: List[Dict[str, object]] = []
    rows: List[Dict[str, object]] = []
    snapshots: Dict[str, Dict[str, str]] = {}
    for top_sources, top_time in combos:
        combo_tag = f"ts{top_sources}_tq{top_time}"
        if not args.skip_runs:
            cmd = [
                sys.executable,
                "-B",
                str(script_sweep.resolve()),
                "--top-sources",
                str(int(top_sources)),
                "--top-time-quartiles",
                str(int(top_time)),
            ]
            result = _run(cmd, root)
            run_logs.append({"combo_tag": combo_tag, **result})
            if int(result["returncode"]) != 0:
                raise RuntimeError(
                    "local-mask sweep failed "
                    f"for {combo_tag}\nstdout:\n{result['stdout_tail']}\nstderr:\n{result['stderr_tail']}"
                )

        if args.skip_runs:
            combo_metrics_path = out_public / f"vlbi_beta_local_mask_two_stage_sweep_{combo_tag}_metrics.json"
            if not combo_metrics_path.exists():
                combo_metrics_path = out_private / f"vlbi_beta_local_mask_two_stage_sweep_{combo_tag}_metrics.json"

            if not combo_metrics_path.exists():
                raise FileNotFoundError(
                    "skip-runs requested but combo snapshot is missing: "
                    f"{combo_tag} ({combo_metrics_path})"
                )

            sweep_metrics = _read_json(combo_metrics_path)
            snapshots[combo_tag] = {
                "summary_csv": str((out_public / f"vlbi_beta_local_mask_two_stage_sweep_{combo_tag}_summary.csv").resolve()),
                "metrics_json": str(combo_metrics_path.resolve()),
                "plot_pdf": str((out_public / f"vlbi_beta_local_mask_two_stage_sweep_{combo_tag}.pdf").resolve()),
                "plot_png": str((out_public / f"vlbi_beta_local_mask_two_stage_sweep_{combo_tag}.png").resolve()),
            }
        else:
            sweep_metrics = _read_json(canonical_metrics)
            snapshots[combo_tag] = _snapshot_combo_outputs(root, combo_tag)

        rows.append(_build_combo_row(top_sources=top_sources, top_time_quartiles=top_time, metrics=sweep_metrics))

    ref_selected = str(rows[0].get("selected_scenario") or "")
    ref_policy = str(rows[0].get("selected_policy") or "")
    robust_selection_same = all(str(r.get("selected_scenario") or "") == ref_selected for r in rows)
    robust_policy_same = all(str(r.get("selected_policy") or "") == ref_policy for r in rows)
    robust_pass = bool(robust_selection_same and robust_policy_same)

    summary_csv = out_private / "vlbi_beta_local_mask_robustness_audit_summary.csv"
    metrics_json = out_private / "vlbi_beta_local_mask_robustness_audit_metrics.json"
    plot_pdf = out_private / "vlbi_beta_local_mask_robustness_audit.pdf"
    plot_png = out_private / "vlbi_beta_local_mask_robustness_audit.png"
    _write_summary_csv(summary_csv, rows)
    _plot_summary(plot_pdf, plot_png, rows)

    restore_result: Dict[str, object] = {}
    if args.restore_chain:
        cmd = [
            sys.executable,
            "-B",
            str(script_apply.resolve()),
            "--gate-metrics",
            str(pre_audit_metrics.resolve()),
        ]
        restore_result = _run(cmd, root)
        if int(restore_result["returncode"]) != 0:
            raise RuntimeError(
                "chain restore failed\n"
                f"stdout:\n{restore_result.get('stdout_tail','')}\n"
                f"stderr:\n{restore_result.get('stderr_tail','')}"
            )

    metrics = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "step": "8.7.46.27",
        "combos": [{"top_sources": int(s), "top_time_quartiles": int(q)} for s, q in combos],
        "rows": rows,
        "robustness": {
            "reference_combo": str(rows[0].get("combo_tag") or ""),
            "reference_selected_scenario": ref_selected,
            "reference_selected_policy": ref_policy,
            "selection_invariant": bool(robust_selection_same),
            "policy_invariant": bool(robust_policy_same),
            "robust_pass": bool(robust_pass),
        },
        "snapshots": snapshots,
        "runs": run_logs,
        "restore_chain": {
            "enabled": bool(args.restore_chain),
            "gate_metrics_used": str(pre_audit_metrics.resolve()),
            "result": restore_result,
        },
        "outputs": {
            "summary_csv": str(summary_csv.resolve()),
            "metrics_json": str(metrics_json.resolve()),
            "plot_pdf": str(plot_pdf.resolve()),
            "plot_png": str(plot_png.resolve()),
        },
    }
    metrics_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    _sync_public(root, [summary_csv, metrics_json, plot_pdf, plot_png, pre_audit_metrics])

    append_event(
        {
            "source": "vlbi_beta_local_mask_robustness_audit",
            "description": "local-mask top-N robustness audit across multiple combos",
            "step": "8.7.46.27",
            "combos": [f"{s}x{q}" for s, q in combos],
            "robust_pass": bool(robust_pass),
            "reference_selected_scenario": ref_selected,
        }
    )

    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {metrics_json}")
    print(f"Wrote: {plot_pdf}")
    print(f"Wrote: {plot_png}")
    print(f"Robust pass: {robust_pass}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
