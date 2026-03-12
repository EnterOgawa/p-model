#!/usr/bin/env python3
"""
vlbi_beta_allsky_consistency.py

All-sky VLBI beta consistency audit across multiple sessions.

Policy:
- Do not use source allowlists.
- Fit beta with all available observations after base quality filtering.
- Use one fixed nuisance mode across sessions (default: baseline_intercept_linear).
- Evaluate cross-session consistency via weighted-mean chi2/dof.

Outputs:
- output/vlbi/vlbi_allsky_beta_consistency_summary.csv
- output/vlbi/vlbi_allsky_beta_consistency_metrics.json
- output/vlbi/vlbi_allsky_beta_consistency.pdf
- output/vlbi/vlbi_allsky_beta_consistency.png
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
from typing import Dict, List, Optional, Sequence, Tuple

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


# Function: Run a subprocess command and capture status without raising.

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


# Function: Compute max abs gravity template [ns] from points CSV.

def _max_abs_template_ns(points_csv: Path) -> float:
    if not points_csv.exists():
        return math.nan

    max_abs = 0.0
    with points_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                val = float(row.get("gravity_template_s", "nan"))
            except Exception:
                continue

            if not math.isfinite(val):
                continue

            aval = abs(val)
            if aval > max_abs:
                max_abs = aval

    return max_abs * 1.0e9


# Function: Compute weighted mean and chi2/dof for session-level beta values.

def _weighted_consistency(beta: np.ndarray, sigma: np.ndarray) -> Dict[str, float]:
    b = np.asarray(beta, dtype=np.float64)
    s = np.asarray(sigma, dtype=np.float64)
    mask = np.isfinite(b) & np.isfinite(s) & (s > 0.0)
    if int(np.sum(mask)) < 2:
        return {
            "n_valid": int(np.sum(mask)),
            "beta_weighted_mean": math.nan,
            "beta_weighted_sigma": math.nan,
            "chi2": math.nan,
            "dof": math.nan,
            "chi2_dof": math.nan,
        }

    b = b[mask]
    s = s[mask]
    w = 1.0 / np.square(s)
    wsum = float(np.sum(w))
    bbar = float(np.sum(w * b) / wsum)
    sig_bar = float(math.sqrt(1.0 / wsum))
    chi2 = float(np.sum(np.square((b - bbar) / s)))
    dof = float(max(1, int(b.size - 1)))
    chi2_dof = float(chi2 / dof)
    return {
        "n_valid": int(b.size),
        "beta_weighted_mean": bbar,
        "beta_weighted_sigma": sig_bar,
        "chi2": chi2,
        "dof": dof,
        "chi2_dof": chi2_dof,
    }


# Function: Build a coarse status from weighted consistency metrics.

def _consistency_status(chi2_dof: float) -> str:
    if not math.isfinite(chi2_dof):
        return "watch"

    if chi2_dof <= 2.0:
        return "pass"

    if chi2_dof <= 5.0:
        return "watch"

    return "reject"


# Function: Render all-sky beta consistency figure.

def _plot_summary(
    pdf_path: Path,
    png_path: Path,
    rows: List[Dict[str, object]],
    beta_mean: float,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return

    valid_rows = []
    for row in rows:
        b = float(row.get("beta_est", math.nan))
        s = float(row.get("beta_sigma", math.nan))
        if math.isfinite(b) and math.isfinite(s) and s > 0.0:
            valid_rows.append(row)

    if not valid_rows:
        return

    labels = [str(r.get("session") or "") for r in valid_rows]
    beta = np.asarray([float(r["beta_est"]) for r in valid_rows], dtype=np.float64)
    sigma = np.asarray([float(r["beta_sigma"]) for r in valid_rows], dtype=np.float64)
    sens = np.asarray([float(r.get("max_abs_bendsun_ns", math.nan)) for r in valid_rows], dtype=np.float64)
    x = np.arange(len(labels), dtype=np.float64)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12.8, 8.8), gridspec_kw={"height_ratios": [2.1, 1.2]})
    ax0.errorbar(x, beta, yerr=sigma, fmt="o", capsize=4, color="tab:blue", label="beta per session")
    ax0.axhline(1.0, color="tab:gray", linestyle="--", linewidth=1.1, label="beta=1")
    if math.isfinite(beta_mean):
        ax0.axhline(beta_mean, color="tab:red", linestyle="-", linewidth=1.2, label="weighted mean")

    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=30, ha="right")
    ax0.set_ylabel("beta estimate")
    ax0.set_title("All-sky VLBI beta consistency")
    ax0.grid(True, alpha=0.28)
    ax0.legend(loc="best")

    ax1.bar(x, sens, color="tab:orange", alpha=0.85)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right")
    ax1.set_ylabel("max |Cal-BendSun| [ns]")
    ax1.set_xlabel("Session")
    ax1.grid(True, axis="y", alpha=0.25)
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


# Function: Main entrypoint for all-sky multi-session audit.

def main() -> int:
    root = _repo_root()
    ap = argparse.ArgumentParser(description="Run all-sky VLBI beta consistency across multiple local archives.")
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
        "--nuisance-mode",
        type=str,
        default="baseline_intercept_linear",
        choices=["none", "baseline_intercept", "baseline_intercept_linear"],
        help="Fixed nuisance mode used for all sessions.",
    )
    ap.add_argument(
        "--observable-series",
        type=str,
        default="full",
        choices=["full", "fringe"],
        help="Observable series passed to direct-fit script.",
    )
    args = ap.parse_args()

    archives = _resolve_archives(temp_dir=args.temp_dir.resolve(), explicit=str(args.archives))
    if not archives:
        raise FileNotFoundError(f"no archives found (temp-dir={args.temp_dir})")

    rows: List[Dict[str, object]] = []
    runlog: List[Dict[str, object]] = []
    for archive in archives:
        session = _session_from_archive(archive)
        session_slug = _slugify(session)
        session_fit_label = f"{session}_ALLSKY_{str(args.nuisance_mode).upper()}"
        session_fit_slug = _slugify(session_fit_label)
        extracted = root / "data" / "vlbi" / "sources" / "vgosdb" / session / "extracted"

        row: Dict[str, object] = {
            "session": session,
            "archive_path": str(archive),
            "archive_bytes": int(archive.stat().st_size),
            "fit_label": session_fit_label,
            "nuisance_mode": str(args.nuisance_mode),
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
                    row["identity_session_values"] = ",".join([str(v) for v in vals])

        cmd_fit = [
            sys.executable,
            "-B",
            str(root / "scripts" / "vlbi" / "vlbi_beta_direct_fit_from_vgosdb.py"),
            "--session",
            session_fit_label,
            "--input-root",
            str(extracted),
            "--nuisance-mode",
            str(args.nuisance_mode),
            "--observable-series",
            str(args.observable_series),
        ]
        r_fit = _run_cmd(cmd_fit)
        row["fit_ok"] = bool(r_fit["ok"])
        row["fit_rc"] = int(r_fit.get("returncode", -1))
        runlog.append({"session": session, "step": "direct_fit_allsky", **r_fit})
        if not bool(r_fit["ok"]):
            rows.append(row)
            continue

        metrics_json = root / "output" / "public" / "vlbi" / f"vlbi_{session_fit_slug}_beta_direct_fit_metrics.json"
        payload = _read_json(metrics_json)
        if payload is None:
            row["fit_ok"] = False
            row["fit_rc"] = -2
            rows.append(row)
            continue

        fit = payload.get("fit_result")
        if isinstance(fit, dict):
            row["n_points"] = int(fit.get("n_points", 0))
            row["beta_est"] = float(fit.get("beta_est", math.nan))
            row["beta_sigma"] = float(fit.get("beta_sigma", math.nan))
            row["delta_beta"] = float(fit.get("delta_beta", math.nan))
            row["chi2"] = float(fit.get("chi2", math.nan))
            row["dof"] = int(fit.get("dof", 0))
            row["weighted_rmse_s"] = float(fit.get("weighted_rmse_s", math.nan))

        outputs = payload.get("outputs")
        if isinstance(outputs, dict):
            points_csv_path = Path(str(outputs.get("points_csv", "")))
            row["max_abs_bendsun_ns"] = _max_abs_template_ns(points_csv_path)

        rows.append(row)

    beta_arr = np.asarray([float(r.get("beta_est", math.nan)) for r in rows], dtype=np.float64)
    sigma_arr = np.asarray([float(r.get("beta_sigma", math.nan)) for r in rows], dtype=np.float64)
    consistency = _weighted_consistency(beta=beta_arr, sigma=sigma_arr)
    status = _consistency_status(float(consistency.get("chi2_dof", math.nan)))

    beta_mean = float(consistency.get("beta_weighted_mean", math.nan))
    for row in rows:
        b = float(row.get("beta_est", math.nan))
        s = float(row.get("beta_sigma", math.nan))
        if math.isfinite(b) and math.isfinite(s) and s > 0.0 and math.isfinite(beta_mean):
            row["pull_vs_weighted_mean"] = float((b - beta_mean) / s)
            row["chi2_contrib"] = float(((b - beta_mean) / s) ** 2)
        else:
            row["pull_vs_weighted_mean"] = math.nan
            row["chi2_contrib"] = math.nan

    out_dir = root / "output" / "vlbi"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "vlbi_allsky_beta_consistency_summary.csv"
    metrics_json = out_dir / "vlbi_allsky_beta_consistency_metrics.json"
    plot_pdf = out_dir / "vlbi_allsky_beta_consistency.pdf"
    plot_png = out_dir / "vlbi_allsky_beta_consistency.png"

    cols = [
        "session",
        "archive_path",
        "archive_bytes",
        "identity_status",
        "identity_session_values",
        "fit_ok",
        "fit_rc",
        "n_points",
        "beta_est",
        "beta_sigma",
        "delta_beta",
        "chi2",
        "dof",
        "weighted_rmse_s",
        "max_abs_bendsun_ns",
        "pull_vs_weighted_mean",
        "chi2_contrib",
    ]
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for row in rows:
            out: List[object] = []
            for col in cols:
                val = row.get(col, "")
                if isinstance(val, float):
                    out.append(f"{val:.16e}" if math.isfinite(val) else "nan")
                else:
                    out.append(val)

            w.writerow(out)

    _plot_summary(pdf_path=plot_pdf, png_path=plot_png, rows=rows, beta_mean=beta_mean)
    payload_out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "all_sky_mode": True,
            "source_filter": "disabled",
            "nuisance_mode": str(args.nuisance_mode),
            "observable_series": str(args.observable_series),
            "consistency_metric": "weighted_mean + chi2/dof across sessions",
        },
        "input": {
            "archives_count": len(archives),
            "archives": [str(p) for p in archives],
        },
        "consistency": {
            **consistency,
            "status": status,
            "criterion_note": "target chi2/dof ~ 1; pass if <=2, watch if <=5, else reject",
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
    metrics_json.write_text(json.dumps(payload_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
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
