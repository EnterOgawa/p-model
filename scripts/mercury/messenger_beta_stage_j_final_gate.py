#!/usr/bin/env python3
"""
messenger_beta_stage_j_final_gate.py

Roadmap Step 8.7.48.10 (final gate) implementation.

Purpose:
- Aggregate Stage B-I machine-readable metrics and freeze final gate status
  for MESSENGER theory-native beta pipeline.
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
from typing import Dict, Iterable, List, Optional

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from scripts.mercury.messenger_beta_stage_d_joint_fit import _sync_to_public
from scripts.summary.worklog import append_event


# Class: Defines one gate row for final checklist output.
@dataclass
class GateRow:
    gate_id: str
    source_metrics: str
    required: str
    status: str
    value: str
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


# Function: Loads one metrics JSON safely.

def _load_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}

    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

    if isinstance(obj, dict):
        return obj

    return {}


# Function: Maps status string into pass/watch/reject.

def _normalize_status(value: object) -> str:
    text = str(value or "").strip().lower()
    if text in {"pass", "watch", "reject"}:
        return text

    return "reject"


# Function: Formats finite float as compact string.

def _fmt_float(value: object, digits: int = 6) -> str:
    try:
        v = float(value)
    except Exception:
        return "nan"

    if not math.isfinite(v):
        return "nan"

    return f"{v:.{int(digits)}g}"


# Function: Writes gate checklist rows to CSV.

def _write_csv(path: Path, rows: List[GateRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["gate_id", "source_metrics", "required", "status", "value", "note"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "gate_id": row.gate_id,
                    "source_metrics": row.source_metrics,
                    "required": row.required,
                    "status": row.status,
                    "value": row.value,
                    "note": row.note,
                }
            )


# Function: Creates final gate status plot.

def _make_plot(rows: List[GateRow], out_pdf: Path, out_png: Path) -> Optional[str]:
    if plt is None:
        return "matplotlib_unavailable"

    if len(rows) <= 0:
        return "no_data"

    color_map = {"pass": "#2ca02c", "watch": "#ff7f0e", "reject": "#d62728"}
    labels = [r.gate_id for r in rows]
    status = [r.status for r in rows]
    y = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(12.8, 0.72 * max(4, len(rows))))
    for yi, st in zip(y, status):
        ax.barh(yi, 1.0, color=color_map.get(st, "#7f7f7f"), alpha=0.85)
        ax.text(1.02, yi, st, va="center", ha="left", fontsize=10)

    ax.set_yticks(y, labels)
    ax.set_xlim(0.0, 1.2)
    ax.set_xlabel("Gate status")
    ax.set_title("Roadmap 8.7.48.10: MESSENGER final gate checklist")
    ax.grid(axis="x", alpha=0.22)
    fig.tight_layout()

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return None


# Function: Main entrypoint for roadmap step 8.7.48.10.

def main() -> int:
    ap = argparse.ArgumentParser(description="Roadmap 8.7.48.10: MESSENGER final gate checklist.")
    ap.add_argument("--public-dir", type=str, default=str(_ROOT / "output" / "public" / "mercury"))
    ap.add_argument("--out-dir", type=str, default=str(_ROOT / "output" / "private" / "mercury"))
    args = ap.parse_args()

    public_dir = _resolve_path(args.public_dir, _ROOT)
    out_dir = _resolve_path(args.out_dir, _ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage_d = _load_json(public_dir / "messenger_beta_stage_d_joint_metrics.json")
    stage_e = _load_json(public_dir / "messenger_beta_stage_e_tnf_replay_metrics.json")
    stage_f = _load_json(public_dir / "messenger_beta_stage_f_injection_recovery_metrics.json")
    stage_g = _load_json(public_dir / "messenger_beta_stage_g_spe_sensitivity_metrics.json")
    stage_h = _load_json(public_dir / "messenger_beta_stage_h_segmentation_metrics.json")
    stage_i = _load_json(public_dir / "messenger_beta_stage_i_nuisance_sensitivity_metrics.json")

    rows: List[GateRow] = []

    st_f = _normalize_status(stage_f.get("overall_status"))
    rows.append(
        GateRow(
            gate_id="gate_injection_recovery",
            source_metrics="messenger_beta_stage_f_injection_recovery_metrics.json",
            required="pass",
            status=st_f if st_f == "pass" else "reject",
            value=f"slope={_fmt_float(stage_f.get('linearity_fit', {}).get('slope'))}, max|z|={_fmt_float(stage_f.get('max_abs_z_error'))}",
            note="hard gate",
        )
    )

    st_d = _normalize_status(stage_d.get("overall_status"))
    st_d_data = _normalize_status(stage_d.get("status_components", {}).get("data"))
    st_d_sigma = _normalize_status(stage_d.get("status_components", {}).get("sigma"))
    if st_d_data == "unknown":
        st_d_gate = st_d
    else:
        st_d_gate = st_d_data

    rows.append(
        GateRow(
            gate_id="gate_odf_joint_stability",
            source_metrics="messenger_beta_stage_d_joint_metrics.json",
            required="watch_or_better",
            status="reject" if st_d_gate == "reject" else st_d_gate,
            value=(
                f"beta={_fmt_float(stage_d.get('beta_dyn_estimate'))}, "
                f"sigma={_fmt_float(stage_d.get('beta_sigma'))}, "
                f"data={st_d_data}, sigma_status={st_d_sigma}"
            ),
            note="stage D baseline",
        )
    )

    st_e = _normalize_status(stage_e.get("overall_status"))
    st_e_data = _normalize_status(stage_e.get("status_components", {}).get("data"))
    st_e_replay = _normalize_status(stage_e.get("replay_vs_odf", {}).get("status"))
    if st_e_data == "reject":
        st_e_gate = "reject"
    else:
        st_e_gate = st_e_replay

    if st_e_gate == "unknown":
        st_e_gate = st_e

    rows.append(
        GateRow(
            gate_id="gate_tnf_replay",
            source_metrics="messenger_beta_stage_e_tnf_replay_metrics.json",
            required="watch_or_better",
            status="reject" if st_e_gate == "reject" else st_e_gate,
            value=(
                f"z_delta={_fmt_float(stage_e.get('replay_vs_odf', {}).get('z_delta_beta'))}, "
                f"status={stage_e.get('replay_vs_odf', {}).get('status')}"
            ),
            note="stage E replay",
        )
    )

    beta_split_mode = str(stage_d.get("beta_split_mode", "coupled")).strip().lower()
    if beta_split_mode != "split":
        st_sep = "watch"
        sep_value = f"beta_split_mode={beta_split_mode or 'coupled'}"
        sep_note = "beta_dyn/beta_lt separation pending"
    else:
        st_d_sep = _normalize_status(stage_d.get("beta_dyn_lt_consistency_status"))
        st_e_sep = _normalize_status(
            stage_e.get("replay_vs_odf", {}).get("replay_vs_odf_beta_lt", {}).get("status")
        )
        if st_d_sep == "reject" or st_e_sep == "reject":
            st_sep = "reject"
        elif st_d_sep == "pass" and st_e_sep == "pass":
            st_sep = "pass"
        else:
            st_sep = "watch"

        sep_value = (
            f"delta_dyn_lt={_fmt_float(stage_d.get('beta_dyn_lt_delta'))}, "
            f"z_dyn_lt={_fmt_float(stage_d.get('beta_dyn_lt_consistency_z'))}, "
            f"replay_z_lt={_fmt_float(stage_e.get('replay_vs_odf', {}).get('replay_vs_odf_beta_lt', {}).get('z_delta_beta'))}"
        )
        sep_note = f"stageD={st_d_sep}, stageE={st_e_sep}"

    rows.append(
        GateRow(
            gate_id="gate_beta_dyn_lt_separation",
            source_metrics="messenger_beta_stage_d_joint_metrics.json|messenger_beta_stage_e_tnf_replay_metrics.json",
            required="watch_or_better",
            status="reject" if st_sep == "reject" else st_sep,
            value=sep_value,
            note=sep_note,
        )
    )

    st_g = _normalize_status(stage_g.get("overall_status"))
    rows.append(
        GateRow(
            gate_id="gate_spe_sensitivity",
            source_metrics="messenger_beta_stage_g_spe_sensitivity_metrics.json",
            required="watch_or_better",
            status="reject" if st_g == "reject" else st_g,
            value=f"branch={stage_g.get('source_branch')}",
            note="stage G SPE subsets",
        )
    )

    st_h = _normalize_status(stage_h.get("overall_status"))
    rows.append(
        GateRow(
            gate_id="gate_segmentation",
            source_metrics="messenger_beta_stage_h_segmentation_metrics.json",
            required="watch_or_better",
            status="reject" if st_h == "reject" else st_h,
            value=f"branch_status={stage_h.get('branch_status')}",
            note="stage H station/link/campaign",
        )
    )

    st_i = _normalize_status(stage_i.get("overall_status"))
    rows.append(
        GateRow(
            gate_id="gate_nuisance_sensitivity",
            source_metrics="messenger_beta_stage_i_nuisance_sensitivity_metrics.json",
            required="watch_or_better",
            status="reject" if st_i == "reject" else st_i,
            value=f"branch_status={stage_i.get('branch_status')}",
            note="stage I nuisance sweep",
        )
    )

    out_csv = out_dir / "messenger_beta_stage_j_final_gate_summary.csv"
    out_json = out_dir / "messenger_beta_stage_j_final_gate_metrics.json"
    out_pdf = out_dir / "messenger_beta_stage_j_final_gate_audit.pdf"
    out_png = out_dir / "messenger_beta_stage_j_final_gate_audit.png"

    _write_csv(out_csv, rows)

    gate_statuses = [r.status for r in rows]
    overall = _combine_status(gate_statuses)
    hard_fail = any(r.gate_id == "gate_injection_recovery" and r.status != "pass" for r in rows)
    if hard_fail:
        overall = "reject"

    plot_note = _make_plot(rows=rows, out_pdf=out_pdf, out_png=out_png)
    produced = [out_csv, out_json]
    if plot_note is None:
        produced.extend([out_pdf, out_png])

    payload: Dict[str, object] = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "phase_step": "8.7.48.10",
        "overall_status": overall,
        "hard_fail": bool(hard_fail),
        "gate_rows": [r.__dict__ for r in rows],
        "status_counts": {
            "pass": int(sum(1 for s in gate_statuses if s == "pass")),
            "watch": int(sum(1 for s in gate_statuses if s == "watch")),
            "reject": int(sum(1 for s in gate_statuses if s == "reject")),
        },
        "inputs": {
            "stage_d": _safe_rel(public_dir / "messenger_beta_stage_d_joint_metrics.json", _ROOT),
            "stage_e": _safe_rel(public_dir / "messenger_beta_stage_e_tnf_replay_metrics.json", _ROOT),
            "stage_f": _safe_rel(public_dir / "messenger_beta_stage_f_injection_recovery_metrics.json", _ROOT),
            "stage_g": _safe_rel(public_dir / "messenger_beta_stage_g_spe_sensitivity_metrics.json", _ROOT),
            "stage_h": _safe_rel(public_dir / "messenger_beta_stage_h_segmentation_metrics.json", _ROOT),
            "stage_i": _safe_rel(public_dir / "messenger_beta_stage_i_nuisance_sensitivity_metrics.json", _ROOT),
        },
        "plot": "generated" if plot_note is None else str(plot_note),
        "outputs_private": [_safe_rel(p, _ROOT) for p in produced if p != out_json],
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    synced = _sync_to_public(produced, private_root=out_dir, public_root=public_dir)
    payload["outputs_public"] = [_safe_rel(p, _ROOT) for p in synced if p.name != out_json.name]
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _sync_to_public([out_json], private_root=out_dir, public_root=public_dir)

    append_event(
        {
            "event": "run_script",
            "script": "scripts/mercury/messenger_beta_stage_j_final_gate.py",
            "phase_step": "8.7.48.10",
            "status": overall,
            "input": str(public_dir),
            "outputs": [_safe_rel(p, _ROOT) for p in produced],
            "metrics": {
                "hard_fail": bool(hard_fail),
                "status_counts": payload["status_counts"],
            },
        }
    )

    print(f"[ok] stage_j_overall={overall}")
    print(f"[ok] hard_fail={hard_fail}")
    print(f"[ok] wrote: {out_csv}")
    print(f"[ok] wrote: {out_json}")
    if plot_note is None:
        print(f"[ok] wrote: {out_pdf}")
        print(f"[ok] wrote: {out_png}")
    else:
        print(f"[warn] plot skipped: {plot_note}")

    print(f"[ok] synced_to_public={len(synced)}")
    return 0


# Condition: Executes CLI main routine.

if __name__ == "__main__":
    raise SystemExit(main())
