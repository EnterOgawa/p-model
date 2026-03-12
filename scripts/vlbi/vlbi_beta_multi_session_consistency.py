#!/usr/bin/env python3
"""
vlbi_beta_multi_session_consistency.py

Run multi-session VLBI beta direct-fit consistency checks from local archives.

Purpose:
- Expand the same primary-data direct-fit pipeline to multiple sessions.
- Produce one consolidated cross-session table for beta and source-filter
  sensitivity.

Workflow per session archive:
1) fetch_ivs_vgosdb_session.py (local input archive mode)
2) vlbi_session_identity_audit.py
3) vlbi_beta_nuisance_sensitivity.py (selected sources)
4) vlbi_beta_nuisance_sensitivity.py (all sources)
5) vlbi_beta_source_filter_sensitivity.py
6) vlbi_beta_source_filter_decomposition.py

Output:
- output/vlbi/vlbi_multi_session_consistency_summary.csv
- output/vlbi/vlbi_multi_session_consistency_metrics.json
- output/vlbi/vlbi_multi_session_consistency.pdf
- output/vlbi/vlbi_multi_session_consistency.png
- Synced copies under output/public/vlbi/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


# Function: Resolve repository root from this script location.
def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


# Function: Normalize labels for stable output filenames.

def _slugify(text: str) -> str:
    value = "".join(ch if ch.isalnum() else "_" for ch in str(text).strip().lower())
    return value or "session"


# Function: Parse archive paths from a directory and optional explicit list.

def _resolve_archives(temp_dir: Path, explicit: str) -> List[Path]:
    archives: List[Path] = []
    if str(explicit).strip():
        for token in str(explicit).replace(";", ",").split(","):
            t = token.strip()
            if not t:
                continue

            p = Path(t).expanduser().resolve()
            if p.exists() and p.is_file():
                archives.append(p)

    if temp_dir.exists():
        for p in sorted(temp_dir.glob("*.tgz")):
            if p not in archives:
                archives.append(p.resolve())

    return archives


# Function: Derive session label from archive filename.

def _session_from_archive(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".tar.gz"):
        stem = name[:-7]
    elif "." in name:
        stem = name.rsplit(".", 1)[0]
    else:
        stem = name

    session = "".join(ch if ch.isalnum() else "_" for ch in stem).upper()
    return session or "SESSION"


# Function: Run a subprocess command and capture status without throwing.

def _run_cmd(cmd: List[str]) -> Dict[str, object]:
    try:
        cp = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception as exc:
        return {
            "ok": False,
            "returncode": -1,
            "stdout": "",
            "stderr": str(exc),
            "cmd": cmd,
        }

    return {
        "ok": cp.returncode == 0,
        "returncode": int(cp.returncode),
        "stdout": cp.stdout or "",
        "stderr": cp.stderr or "",
        "cmd": cmd,
    }


# Function: Read JSON file if available and valid.

def _read_json(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if isinstance(payload, dict):
        return payload

    return None


# Function: Extract best-mode beta from nuisance sensitivity payload.

def _extract_best_mode(payload: Optional[Dict[str, object]]) -> Dict[str, object]:
    out: Dict[str, object] = {
        "best_mode": "",
        "beta_est": math.nan,
        "beta_sigma": math.nan,
        "weighted_rmse_s": math.nan,
    }
    if payload is None:
        return out

    summary = payload.get("summary")
    rows_raw = payload.get("rows")
    if not isinstance(summary, dict) or not isinstance(rows_raw, list):
        return out

    best_mode = str(summary.get("mode_best_weighted_rmse") or "")
    if not best_mode and isinstance(summary.get("mode_best_aic_like"), str):
        best_mode = str(summary.get("mode_best_aic_like") or "")

    out["best_mode"] = best_mode
    for row in rows_raw:
        if not isinstance(row, dict):
            continue

        if str(row.get("mode") or "") != best_mode:
            continue

        out["beta_est"] = float(row.get("beta_est", math.nan))
        out["beta_sigma"] = float(row.get("beta_sigma", math.nan))
        out["weighted_rmse_s"] = float(row.get("weighted_rmse_s", math.nan))
        return out

    return out


# Function: Render cross-session delta-beta summary figure.

def _plot_summary(pdf_path: Path, png_path: Path, rows: List[Dict[str, object]]) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    if not rows:
        return

    labels: List[str] = []
    deltas: List[float] = []
    sigmas: List[float] = []
    for row in rows:
        z = float(row.get("source_filter_abs_z_delta_beta", math.nan))
        d = float(row.get("source_filter_delta_beta_all_minus_selected", math.nan))
        s = float(row.get("source_filter_sigma_combined", math.nan))
        if not (math.isfinite(z) and math.isfinite(d) and math.isfinite(s)):
            continue

        labels.append(str(row.get("session") or ""))
        deltas.append(d)
        sigmas.append(s)

    if not labels:
        return

    x = np.arange(len(labels), dtype=np.float64)
    dd = np.asarray(deltas, dtype=np.float64)
    ss = np.asarray(sigmas, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(12.8, 6.6))
    ax.errorbar(x, dd, yerr=ss, fmt="o", capsize=4, color="tab:blue")
    ax.axhline(0.0, color="tab:gray", linestyle="--", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("delta beta (all - selected)")
    ax.set_title("VLBI multi-session source-filter delta beta consistency")
    ax.grid(True, alpha=0.28)
    fig.tight_layout()
    fig.savefig(str(pdf_path))
    fig.savefig(str(png_path), dpi=170)
    plt.close(fig)


# Function: Copy generated artifacts to output/public/vlbi.

def _sync_public(root: Path, outputs: Sequence[Path]) -> None:
    dst = root / "output" / "public" / "vlbi"
    dst.mkdir(parents=True, exist_ok=True)
    for path in outputs:
        if path.exists():
            shutil.copy2(path, dst / path.name)


# Function: Main entrypoint for multi-session consistency run.

def main() -> int:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="Run VLBI beta consistency checks across multiple local session archives.")
    ap.add_argument(
        "--temp-dir",
        type=Path,
        default=root / "data" / "temp",
        help="Directory containing local .tgz session archives.",
    )
    ap.add_argument(
        "--archives",
        type=str,
        default="",
        help="Comma-separated explicit archive file paths (optional).",
    )
    ap.add_argument(
        "--source-include",
        type=str,
        default="0229+131,0235+164",
        help="Comma-separated source allowlist for selected-source branch.",
    )
    ap.add_argument(
        "--modes",
        type=str,
        default="none,baseline_intercept,baseline_intercept_linear",
        help="Nuisance modes passed to nuisance sensitivity script.",
    )
    ap.add_argument(
        "--best-mode",
        type=str,
        default="baseline_intercept_linear",
        help="Best mode label passed to source-filter sensitivity script.",
    )
    args = ap.parse_args()

    temp_dir = args.temp_dir.resolve()
    archives = _resolve_archives(temp_dir=temp_dir, explicit=str(args.archives))
    if not archives:
        raise FileNotFoundError(f"no archives found in temp-dir={temp_dir}")

    rows: List[Dict[str, object]] = []
    runlog: List[Dict[str, object]] = []
    for archive in archives:
        session = _session_from_archive(archive)
        session_slug = _slugify(session)
        session_root = root / "data" / "vlbi" / "sources" / "vgosdb" / session
        extracted = session_root / "extracted"
        row: Dict[str, object] = {
            "session": session,
            "archive_path": str(archive),
            "archive_bytes": int(archive.stat().st_size),
        }

        cmd_fetch = [
            sys.executable,
            "-B",
            str(root / "scripts" / "vlbi" / "fetch_ivs_vgosdb_session.py"),
            "--session",
            session,
            "--input-archive",
            str(archive),
            "--force",
        ]
        r_fetch = _run_cmd(cmd_fetch)
        row["fetch_ok"] = bool(r_fetch["ok"])
        runlog.append({"session": session, "step": "fetch", **r_fetch})
        if not bool(r_fetch["ok"]):
            rows.append(row)
            continue

        cmd_identity = [
            sys.executable,
            "-B",
            str(root / "scripts" / "vlbi" / "vlbi_session_identity_audit.py"),
            "--session-label",
            session,
            "--input-root",
            str(extracted),
        ]
        r_identity = _run_cmd(cmd_identity)
        row["identity_ok"] = bool(r_identity["ok"])
        runlog.append({"session": session, "step": "identity", **r_identity})
        identity_json = root / "output" / "public" / "vlbi" / f"vlbi_{session_slug}_session_identity_audit.json"
        identity_payload = _read_json(identity_json)
        if identity_payload is not None:
            scan = identity_payload.get("scan_summary")
            if isinstance(scan, dict):
                row["identity_status"] = str(scan.get("status") or "")
                vals = scan.get("observed_session_values")
                if isinstance(vals, list):
                    row["identity_unique_session_count"] = int(len(vals))
                    row["identity_session_values"] = ",".join([str(v) for v in vals])

        cmd_ns_selected = [
            sys.executable,
            "-B",
            str(root / "scripts" / "vlbi" / "vlbi_beta_nuisance_sensitivity.py"),
            "--session",
            session,
            "--input-root",
            str(extracted),
            "--source-include",
            str(args.source_include),
            "--modes",
            str(args.modes),
        ]
        r_ns_sel = _run_cmd(cmd_ns_selected)
        row["nuisance_selected_ok"] = bool(r_ns_sel["ok"])
        row["nuisance_selected_rc"] = int(r_ns_sel.get("returncode", -1))
        runlog.append({"session": session, "step": "nuisance_selected", **r_ns_sel})
        sel_metrics = root / "output" / "public" / "vlbi" / f"vlbi_{session_slug}_beta_nuisance_sensitivity_metrics.json"
        sel_payload = _read_json(sel_metrics)
        sel_best = _extract_best_mode(sel_payload)
        row["selected_best_mode"] = str(sel_best.get("best_mode") or "")
        row["selected_beta_best"] = float(sel_best.get("beta_est", math.nan))
        row["selected_beta_sigma_best"] = float(sel_best.get("beta_sigma", math.nan))
        row["selected_wrmse_best_s"] = float(sel_best.get("weighted_rmse_s", math.nan))

        session_all = f"{session}_ALL"
        session_all_slug = _slugify(session_all)
        cmd_ns_all = [
            sys.executable,
            "-B",
            str(root / "scripts" / "vlbi" / "vlbi_beta_nuisance_sensitivity.py"),
            "--session",
            session_all,
            "--input-root",
            str(extracted),
            "--modes",
            str(args.modes),
        ]
        r_ns_all = _run_cmd(cmd_ns_all)
        row["nuisance_all_ok"] = bool(r_ns_all["ok"])
        row["nuisance_all_rc"] = int(r_ns_all.get("returncode", -1))
        runlog.append({"session": session, "step": "nuisance_all", **r_ns_all})
        all_metrics = root / "output" / "public" / "vlbi" / f"vlbi_{session_all_slug}_beta_nuisance_sensitivity_metrics.json"
        all_payload = _read_json(all_metrics)
        all_best = _extract_best_mode(all_payload)
        row["all_best_mode"] = str(all_best.get("best_mode") or "")
        row["all_beta_best"] = float(all_best.get("beta_est", math.nan))
        row["all_beta_sigma_best"] = float(all_best.get("beta_sigma", math.nan))
        row["all_wrmse_best_s"] = float(all_best.get("weighted_rmse_s", math.nan))

        cmd_sf = [
            sys.executable,
            "-B",
            str(root / "scripts" / "vlbi" / "vlbi_beta_source_filter_sensitivity.py"),
            "--session",
            session,
            "--selected-metrics",
            str(sel_metrics),
            "--all-metrics",
            str(all_metrics),
            "--best-mode",
            str(args.best_mode),
        ]
        r_sf = _run_cmd(cmd_sf)
        row["source_filter_ok"] = bool(r_sf["ok"])
        row["source_filter_rc"] = int(r_sf.get("returncode", -1))
        runlog.append({"session": session, "step": "source_filter", **r_sf})
        sf_json = root / "output" / "public" / "vlbi" / f"vlbi_{session_slug}_beta_source_filter_sensitivity_metrics.json"
        sf_payload = _read_json(sf_json)
        if sf_payload is not None:
            summary = sf_payload.get("summary")
            if isinstance(summary, dict):
                row["source_filter_status"] = str(summary.get("best_mode_status") or "")
                row["source_filter_delta_beta_all_minus_selected"] = float(
                    summary.get("best_mode_delta_beta_all_minus_selected", math.nan)
                )
                row["source_filter_sigma_combined"] = float(summary.get("best_mode_sigma_combined", math.nan))
                row["source_filter_abs_z_delta_beta"] = float(summary.get("best_mode_abs_z_delta_beta", math.nan))

        cmd_decomp = [
            sys.executable,
            "-B",
            str(root / "scripts" / "vlbi" / "vlbi_beta_source_filter_decomposition.py"),
            "--session",
            session,
            "--input-root",
            str(extracted),
            "--source-include",
            str(args.source_include),
            "--nuisance-mode",
            str(args.best_mode),
        ]
        r_decomp = _run_cmd(cmd_decomp)
        row["decomposition_ok"] = bool(r_decomp["ok"])
        row["decomposition_rc"] = int(r_decomp.get("returncode", -1))
        runlog.append({"session": session, "step": "decomposition", **r_decomp})
        decomp_json = root / "output" / "public" / "vlbi" / f"vlbi_{session_slug}_beta_source_filter_decomposition_metrics.json"
        decomp_payload = _read_json(decomp_json)
        if decomp_payload is not None:
            row["decomposition_status"] = str(decomp_payload.get("status") or "")
            ds = decomp_payload.get("decomposition_summary")
            if isinstance(ds, dict):
                row["decomposition_max_abs_z_component"] = float(ds.get("max_abs_z_component", math.nan))
                row["decomposition_max_abs_impact_component"] = float(ds.get("max_abs_impact_component", math.nan))

        rows.append(row)

    out_dir = root / "output" / "vlbi"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "vlbi_multi_session_consistency_summary.csv"
    metrics_json = out_dir / "vlbi_multi_session_consistency_metrics.json"
    plot_pdf = out_dir / "vlbi_multi_session_consistency.pdf"
    plot_png = out_dir / "vlbi_multi_session_consistency.png"

    cols = [
        "session",
        "archive_path",
        "archive_bytes",
        "fetch_ok",
        "identity_ok",
        "identity_status",
        "identity_unique_session_count",
        "identity_session_values",
        "nuisance_selected_ok",
        "nuisance_selected_rc",
        "selected_best_mode",
        "selected_beta_best",
        "selected_beta_sigma_best",
        "selected_wrmse_best_s",
        "nuisance_all_ok",
        "nuisance_all_rc",
        "all_best_mode",
        "all_beta_best",
        "all_beta_sigma_best",
        "all_wrmse_best_s",
        "source_filter_ok",
        "source_filter_rc",
        "source_filter_status",
        "source_filter_delta_beta_all_minus_selected",
        "source_filter_sigma_combined",
        "source_filter_abs_z_delta_beta",
        "decomposition_ok",
        "decomposition_rc",
        "decomposition_status",
        "decomposition_max_abs_z_component",
        "decomposition_max_abs_impact_component",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for row in rows:
            out = []
            for col in cols:
                val = row.get(col, "")
                if isinstance(val, float):
                    if math.isfinite(val):
                        out.append(f"{val:.16e}")
                    else:
                        out.append("nan")
                else:
                    out.append(val)

            w.writerow(out)

    _plot_summary(pdf_path=plot_pdf, png_path=plot_png, rows=rows)
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "temp_dir": str(temp_dir),
            "archives_count": len(archives),
            "source_include": str(args.source_include),
            "modes": str(args.modes),
            "best_mode": str(args.best_mode),
        },
        "rows": rows,
        "runlog": runlog,
        "outputs": {
            "summary_csv": str(summary_csv),
            "metrics_json": str(metrics_json),
            "plot_pdf": str(plot_pdf),
            "plot_png": str(plot_png),
        },
    }
    metrics_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _sync_public(root, [summary_csv, metrics_json, plot_pdf, plot_png])
    print("Wrote:", summary_csv)
    print("Wrote:", metrics_json)
    print("Wrote:", plot_pdf)
    print("Wrote:", plot_png)
    print("Synced:", root / "output" / "public" / "vlbi")
    return 0


# Branch: Execute CLI entrypoint when this file is invoked directly.

if __name__ == "__main__":
    raise SystemExit(main())
