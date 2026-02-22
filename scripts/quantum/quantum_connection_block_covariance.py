#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
quantum_connection_block_covariance.py

Step 7.21.4:
Bell の cross-dataset covariance と、干渉 / 物性・熱 holdout の要約を
公開済みデータ範囲で block covariance（チャネル横断）として統合する。

出力:
  - output/public/quantum/quantum_connection_block_covariance.json
  - output/public/quantum/quantum_connection_block_covariance.csv
  - output/public/quantum/quantum_connection_block_covariance.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summary import worklog  # noqa: E402


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT)).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        number = float(value)
        if math.isfinite(number):
            return number
    return None


def _resample(series: np.ndarray, n_points: int) -> np.ndarray:
    if series.ndim != 1:
        series = series.ravel()
    if series.size == 0:
        raise ValueError("empty series cannot be resampled")
    if n_points <= 1:
        return np.array([float(series[0])], dtype=float)
    if series.size == 1:
        return np.full((n_points,), float(series[0]), dtype=float)
    x_old = np.linspace(0.0, 1.0, series.size)
    x_new = np.linspace(0.0, 1.0, n_points)
    return np.interp(x_new, x_old, series).astype(float)


def _safe_log1p(series: np.ndarray) -> np.ndarray:
    clipped = np.clip(series.astype(float), a_min=0.0, a_max=None)
    return np.log1p(clipped)


def _eigen_summary(matrix: np.ndarray, labels: List[str]) -> Dict[str, Any]:
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        return {"supported": False, "reason": "matrix must be non-empty square"}
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    dominant = eigenvectors[:, 0]
    dominant_abs = np.abs(dominant)
    idx = int(np.argmax(dominant_abs))
    return {
        "supported": True,
        "eigenvalues_desc": [float(v) for v in eigenvalues.tolist()],
        "dominant_mode": {
            "eigenvalue": float(eigenvalues[0]),
            "dominant_channel": labels[idx],
            "dominant_loading": float(dominant[idx]),
            "channel_loadings": {labels[i]: float(dominant[i]) for i in range(len(labels))},
        },
    }


def _extract_bell_series(cross_cov: Dict[str, Any]) -> np.ndarray:
    matrices = cross_cov.get("matrices") if isinstance(cross_cov.get("matrices"), dict) else {}
    profile_cov = np.array(matrices.get("profile_cov") or [], dtype=float)
    if profile_cov.ndim != 2 or profile_cov.shape[0] != profile_cov.shape[1] or profile_cov.shape[0] == 0:
        raise ValueError("bell cross_dataset profile_cov is missing or invalid")
    sigma = np.sqrt(np.clip(np.diag(profile_cov), a_min=0.0, a_max=None))
    return sigma.astype(float)


def _extract_interference_series(born_ab: Dict[str, Any]) -> np.ndarray:
    criteria = born_ab.get("criteria") if isinstance(born_ab.get("criteria"), list) else []
    values: List[float] = []
    for row in criteria:
        if not isinstance(row, dict):
            continue
        channel = str(row.get("channel") or "")
        if channel == "bell_selection":
            continue
        score = _as_float(row.get("normalized_score"))
        if score is None:
            continue
        values.append(score)
    if not values:
        raise ValueError("interference criteria not found in born_ab gate payload")
    return np.array(values, dtype=float)


def _extract_condensed_series(condensed: Dict[str, Any]) -> np.ndarray:
    summary = condensed.get("summary") if isinstance(condensed.get("summary"), dict) else {}
    datasets = summary.get("datasets") if isinstance(summary.get("datasets"), list) else []
    values: List[float] = []
    for item in datasets:
        if not isinstance(item, dict):
            continue
        gates = item.get("audit_gates") if isinstance(item.get("audit_gates"), dict) else {}
        model = (
            gates.get("recommended_model_by_minimax_test_max_abs_z")
            if isinstance(gates.get("recommended_model_by_minimax_test_max_abs_z"), dict)
            else {}
        )
        worst = _as_float(model.get("worst_test_max_abs_z"))
        if worst is None:
            continue
        values.append(worst / 3.0)
    if not values:
        raise ValueError("condensed holdout summary lacks usable worst_test_max_abs_z values")
    return np.array(values, dtype=float)


def build_payload(
    *,
    bell_cross_cov_json: Path,
    born_ab_gate_json: Path,
    condensed_holdout_summary_json: Path,
    shared_kpi_json: Path,
) -> Dict[str, Any]:
    bell_cross = _read_json(bell_cross_cov_json)
    born_ab = _read_json(born_ab_gate_json)
    condensed = _read_json(condensed_holdout_summary_json)
    shared_kpi = _read_json(shared_kpi_json)

    bell_raw = _extract_bell_series(bell_cross)
    interference_raw = _extract_interference_series(born_ab)
    condensed_raw = _extract_condensed_series(condensed)

    grid_n = 81
    transformed = {
        "bell": _safe_log1p(bell_raw),
        "interference": _safe_log1p(interference_raw),
        "condensed": _safe_log1p(condensed_raw),
    }
    aligned = {
        name: _resample(series, grid_n)
        for name, series in transformed.items()
    }

    channel_labels = ["bell", "interference", "condensed"]
    channel_matrix = np.vstack([aligned[label] for label in channel_labels]).astype(float)
    channel_cov = np.cov(channel_matrix, bias=False)
    channel_corr = np.corrcoef(channel_matrix)

    cov_eigen = _eigen_summary(channel_cov, channel_labels)
    corr_eigen = _eigen_summary(channel_corr, channel_labels)

    status_by_channel: Dict[str, str] = {}
    for entry in shared_kpi.get("channels") if isinstance(shared_kpi.get("channels"), list) else []:
        if not isinstance(entry, dict):
            continue
        key = str(entry.get("channel") or "")
        status = str(entry.get("status") or "")
        if key:
            status_by_channel[key] = status

    bell_eigen_src = (
        bell_cross.get("matrices", {}).get("profile_cov_eigen")
        if isinstance(bell_cross.get("matrices"), dict) and isinstance(bell_cross.get("matrices", {}).get("profile_cov_eigen"), dict)
        else {}
    )
    bell_top_eigenvalue = _as_float((bell_eigen_src or {}).get("eigenvalues_desc", [None])[0] if isinstance((bell_eigen_src or {}).get("eigenvalues_desc"), list) and (bell_eigen_src or {}).get("eigenvalues_desc") else None)

    channels_summary: List[Dict[str, Any]] = []
    for label in channel_labels:
        raw = {"bell": bell_raw, "interference": interference_raw, "condensed": condensed_raw}[label]
        tr = transformed[label]
        al = aligned[label]
        status_key = label if label != "condensed" else "condensed_thermal_holdout"
        channels_summary.append(
            {
                "channel": label,
                "status_from_shared_kpi": status_by_channel.get(status_key),
                "raw_points_n": int(raw.size),
                "raw_min": float(np.min(raw)),
                "raw_max": float(np.max(raw)),
                "raw_median": float(np.median(raw)),
                "transformed_median": float(np.median(tr)),
                "aligned_mean": float(np.mean(al)),
                "aligned_std": float(np.std(al, ddof=1)) if al.size > 1 else 0.0,
            }
        )

    dominant_mode = cov_eigen.get("dominant_mode") if isinstance(cov_eigen, dict) else {}
    if not isinstance(dominant_mode, dict):
        dominant_mode = {}
    dominant_channel = str(dominant_mode.get("dominant_channel") or "")
    dominant_reason = {
        "bell": "Bell cross-dataset covariance block",
        "interference": "interference proxy gate block",
        "condensed": "condensed holdout block",
    }.get(dominant_channel, "not identified")

    return {
        "generated_utc": _iso_utc_now(),
        "phase": {"phase": 7, "step": "7.21.4", "name": "quantum cross-channel block covariance"},
        "intent": "Integrate Bell covariance with interference/condensed summaries and identify dominant eigen-mode within public data scope.",
        "inputs": {
            "bell_cross_dataset_covariance_json": _rel(bell_cross_cov_json),
            "born_ab_gate_json": _rel(born_ab_gate_json),
            "condensed_holdout_audit_summary_json": _rel(condensed_holdout_summary_json),
            "quantum_connection_shared_kpi_json": _rel(shared_kpi_json),
        },
        "method": {
            "transform": "log1p on non-negative normalized proxies",
            "alignment": f"linear interpolation to common grid (n={grid_n})",
            "covariance": "sample covariance across aligned channel series",
            "note": "Operational block covariance for cross-channel auditing (not a universal physical covariance).",
        },
        "channels": channels_summary,
        "matrices": {
            "channel_order": channel_labels,
            "channel_covariance": channel_cov.tolist(),
            "channel_correlation": channel_corr.tolist(),
            "channel_covariance_eigen": cov_eigen,
            "channel_correlation_eigen": corr_eigen,
            "bell_profile_covariance_top_eigenvalue": bell_top_eigenvalue,
        },
        "diagnostics": {
            "dominant_mode_reason": dominant_reason,
            "aligned_grid_points": grid_n,
        },
    }


def _write_csv(path: Path, payload: Dict[str, Any]) -> None:
    matrices = payload.get("matrices") if isinstance(payload.get("matrices"), dict) else {}
    labels = matrices.get("channel_order") if isinstance(matrices.get("channel_order"), list) else []
    cov = np.array(matrices.get("channel_covariance") or [], dtype=float)
    corr = np.array(matrices.get("channel_correlation") or [], dtype=float)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "row_channel",
                "col_channel",
                "covariance",
                "correlation",
            ],
        )
        writer.writeheader()
        for i, row_label in enumerate(labels):
            for j, col_label in enumerate(labels):
                writer.writerow(
                    {
                        "row_channel": row_label,
                        "col_channel": col_label,
                        "covariance": float(cov[i, j]) if cov.size else None,
                        "correlation": float(corr[i, j]) if corr.size else None,
                    }
                )


def _plot(path: Path, payload: Dict[str, Any]) -> None:
    matrices = payload.get("matrices") if isinstance(payload.get("matrices"), dict) else {}
    labels = matrices.get("channel_order") if isinstance(matrices.get("channel_order"), list) else []
    cov = np.array(matrices.get("channel_covariance") or [], dtype=float)
    corr = np.array(matrices.get("channel_correlation") or [], dtype=float)
    eig = (
        matrices.get("channel_covariance_eigen")
        if isinstance(matrices.get("channel_covariance_eigen"), dict)
        else {}
    )
    eig_vals = (
        [float(v) for v in eig.get("eigenvalues_desc") if isinstance(v, (int, float))]
        if isinstance(eig.get("eigenvalues_desc"), list)
        else []
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), dpi=180)

    im0 = axes[0, 0].imshow(cov, cmap="viridis")
    axes[0, 0].set_title("Channel covariance")
    axes[0, 0].set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    axes[0, 0].set_yticks(range(len(labels)), labels)
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axes[0, 1].set_title("Channel correlation")
    axes[0, 1].set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    axes[0, 1].set_yticks(range(len(labels)), labels)
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    axes[1, 0].bar(range(len(eig_vals)), eig_vals, color="#4c6ef5")
    axes[1, 0].set_title("Covariance eigenvalues")
    axes[1, 0].set_xlabel("mode index")
    axes[1, 0].set_ylabel("eigenvalue")
    axes[1, 0].grid(axis="y", alpha=0.25, linestyle=":")

    channel_info = payload.get("channels") if isinstance(payload.get("channels"), list) else []
    x = np.arange(len(channel_info))
    means = [float(item.get("aligned_mean") or 0.0) for item in channel_info if isinstance(item, dict)]
    stds = [float(item.get("aligned_std") or 0.0) for item in channel_info if isinstance(item, dict)]
    labels2 = [str(item.get("channel") or "") for item in channel_info if isinstance(item, dict)]
    axes[1, 1].bar(x, means, yerr=stds, capsize=4, color="#51cf66")
    axes[1, 1].set_xticks(x, labels2, rotation=20, ha="right")
    axes[1, 1].set_title("Aligned channel mean ± std")
    axes[1, 1].set_ylabel("log1p-normalized scale")
    axes[1, 1].grid(axis="y", alpha=0.25, linestyle=":")

    fig.suptitle("Quantum connection block covariance (Step 7.21.4)", y=0.99)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build block covariance across Bell/interference/condensed channels.")
    parser.add_argument(
        "--bell-cross-cov",
        default=str(ROOT / "output" / "public" / "quantum" / "bell" / "cross_dataset_covariance.json"),
        help="Input Bell cross-dataset covariance JSON.",
    )
    parser.add_argument(
        "--born-ab-gate",
        default=str(ROOT / "output" / "public" / "quantum" / "quantum_connection_born_ab_gate.json"),
        help="Input Step 7.21.3 Born A/B gate JSON.",
    )
    parser.add_argument(
        "--condensed-holdout",
        default=str(ROOT / "output" / "public" / "quantum" / "condensed_holdout_audit_summary.json"),
        help="Input condensed holdout summary JSON.",
    )
    parser.add_argument(
        "--shared-kpi",
        default=str(ROOT / "output" / "public" / "quantum" / "quantum_connection_shared_kpi.json"),
        help="Input shared KPI JSON.",
    )
    parser.add_argument(
        "--out-json",
        default=str(ROOT / "output" / "public" / "quantum" / "quantum_connection_block_covariance.json"),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--out-csv",
        default=str(ROOT / "output" / "public" / "quantum" / "quantum_connection_block_covariance.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--out-png",
        default=str(ROOT / "output" / "public" / "quantum" / "quantum_connection_block_covariance.png"),
        help="Output PNG path.",
    )
    args = parser.parse_args(argv)

    def _resolve(path_text: str) -> Path:
        path = Path(path_text)
        if path.is_absolute():
            return path.resolve()
        return (ROOT / path).resolve()

    input_paths = {
        "bell_cross_cov": _resolve(args.bell_cross_cov),
        "born_ab_gate": _resolve(args.born_ab_gate),
        "condensed_holdout": _resolve(args.condensed_holdout),
        "shared_kpi": _resolve(args.shared_kpi),
    }
    for key, input_path in input_paths.items():
        if not input_path.exists():
            raise FileNotFoundError(f"required input not found ({key}): {_rel(input_path)}")

    out_json = _resolve(args.out_json)
    out_csv = _resolve(args.out_csv)
    out_png = _resolve(args.out_png)

    payload = build_payload(
        bell_cross_cov_json=input_paths["bell_cross_cov"],
        born_ab_gate_json=input_paths["born_ab_gate"],
        condensed_holdout_summary_json=input_paths["condensed_holdout"],
        shared_kpi_json=input_paths["shared_kpi"],
    )

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(out_csv, payload)
    _plot(out_png, payload)

    print(f"[ok] wrote: {_rel(out_json)}")
    print(f"[ok] wrote: {_rel(out_csv)}")
    print(f"[ok] wrote: {_rel(out_png)}")

    try:
        worklog.append_event(
            {
                "event_type": "quantum_connection_block_covariance",
                "phase": "7.21.4",
                "inputs": payload.get("inputs"),
                "outputs": {
                    "quantum_connection_block_covariance_json": _rel(out_json),
                    "quantum_connection_block_covariance_csv": _rel(out_csv),
                    "quantum_connection_block_covariance_png": _rel(out_png),
                },
                "diagnostics": payload.get("diagnostics"),
                "dominant_mode": (
                    payload.get("matrices", {}).get("channel_covariance_eigen", {}).get("dominant_mode")
                    if isinstance(payload.get("matrices"), dict)
                    else None
                ),
            }
        )
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
