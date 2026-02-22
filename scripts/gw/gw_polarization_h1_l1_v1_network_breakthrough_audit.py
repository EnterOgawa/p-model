from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog  # noqa: E402


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_japanese_font() -> None:
    try:
        import matplotlib as mpl
        import matplotlib.font_manager as font_manager

        preferred_fonts = [
            "Yu Gothic",
            "Meiryo",
            "BIZ UDGothic",
            "MS Gothic",
            "Yu Mincho",
            "MS Mincho",
        ]
        available_fonts = {font.name for font in font_manager.fontManager.ttflist}
        selected_fonts = [name for name in preferred_fonts if name in available_fonts]
        if not selected_fonts:
            return
        mpl.rcParams["font.family"] = selected_fonts + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        return


def _safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out


def _fmt(value: Any, digits: int = 7) -> str:
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if not isinstance(value, (float, np.floating)):
        return str(value)
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        return ""
    if numeric_value == 0.0:
        return "0"
    abs_value = abs(numeric_value)
    if abs_value >= 1e4 or abs_value < 1e-3:
        return f"{numeric_value:.{digits}g}"
    return f"{numeric_value:.{digits}f}".rstrip("0").rstrip(".")


def _parse_float_grid(text: str) -> List[float]:
    out: List[float] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        out.append(float(token))
    return out


def _parse_int_grid(text: str) -> List[int]:
    out: List[int] = []
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        out.append(int(token))
    return out


def _parse_bool_grid(text: str) -> List[bool]:
    mapping = {"0": False, "1": True, "false": False, "true": True, "no": False, "yes": True}
    out: List[bool] = []
    for token in str(text).split(","):
        token = token.strip().lower()
        if not token:
            continue
        if token not in mapping:
            raise ValueError(f"invalid bool token in grid: {token}")
        out.append(bool(mapping[token]))
    return out


def _status_bucket(status: str) -> str:
    text = str(status or "")
    if text.startswith("pass"):
        return "pass"
    if text.startswith("watch"):
        return "watch"
    if text.startswith("reject"):
        return "reject"
    if text.startswith("inconclusive"):
        return "inconclusive"
    return "other"


def _status_rank(status: str) -> int:
    bucket = _status_bucket(status)
    if bucket == "pass":
        return 3
    if bucket == "watch":
        return 2
    if bucket == "reject":
        return 1
    if bucket == "inconclusive":
        return 0
    return -1


def _scalar_rank(value: float) -> float:
    if not math.isfinite(value):
        return -1.0e12
    return -float(value)


def _objective_key(row: Dict[str, Any]) -> Tuple[int, int, int, float, int]:
    status = str(row.get("status", ""))
    n_usable_events = int(row.get("n_usable_events", 0))
    tensor_reject_events = int(row.get("tensor_reject_events", 0))
    scalar_overlap_proxy = _safe_float(row.get("scalar_overlap_proxy"))
    pair_pruned_events = int(row.get("n_pair_pruned_events", 0))
    return (
        _status_rank(status),
        n_usable_events,
        -tensor_reject_events,
        _scalar_rank(scalar_overlap_proxy),
        -pair_pruned_events,
    )


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    headers = [
        "trial_id",
        "status",
        "reason",
        "n_usable_events",
        "n_pair_pruned_events",
        "tensor_reject_events",
        "watch_events",
        "scalar_overlap_proxy",
        "corr_use_min",
        "response_floor_frac",
        "min_ring_directions",
        "geometry_relax_factor",
        "geometry_delay_floor_ms",
        "allow_pair_pruning",
        "top_bottleneck_pair",
        "top_bottleneck_share",
        "returncode",
        "trial_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        for row in rows:
            values: List[Any] = []
            for key in headers:
                value = row.get(key, "")
                values.append(_fmt(value) if isinstance(value, float) else value)
            writer.writerow(values)


def _plot(rows: List[Dict[str, Any]], status_counts: Dict[str, int], pair_summary: List[Dict[str, Any]], out_png: Path) -> None:
    _set_japanese_font()
    color_map = {
        "pass": "#2ca02c",
        "watch": "#f2c744",
        "reject": "#d62728",
        "inconclusive": "#9aa0a6",
        "other": "#555555",
    }

    figure, axes = plt.subplots(3, 1, figsize=(13.5, 10.2))
    axis_scatter, axis_status, axis_pair = axes

    if rows:
        usable_values = np.asarray([int(row.get("n_usable_events", 0)) for row in rows], dtype=float)
        scalar_values = np.asarray([_safe_float(row.get("scalar_overlap_proxy")) for row in rows], dtype=float)
        status_values = [str(row.get("status", "")) for row in rows]
        colors = [color_map[_status_bucket(status)] for status in status_values]
        axis_scatter.scatter(usable_values, scalar_values, c=colors, s=34, alpha=0.85, edgecolor="none")
    axis_scatter.axhline(0.0, color="#666666", linestyle="--", linewidth=1.0, alpha=0.8)
    axis_scatter.set_xlabel("n usable events")
    axis_scatter.set_ylabel("scalar overlap proxy")
    axis_scatter.set_title("Breakthrough envelope: usable events vs scalar overlap")
    axis_scatter.grid(True, alpha=0.25)

    status_order = ["pass", "watch", "reject", "inconclusive", "other"]
    status_values = [int(status_counts.get(name, 0)) for name in status_order]
    axis_status.bar(np.arange(len(status_order)), status_values, color=[color_map[name] for name in status_order], alpha=0.9)
    axis_status.set_xticks(np.arange(len(status_order)))
    axis_status.set_xticklabels(status_order)
    axis_status.set_ylabel("trial count")
    axis_status.set_title("Trial status counts")
    axis_status.grid(True, axis="y", alpha=0.25)

    if pair_summary:
        labels = [str(row.get("pair", "")) for row in pair_summary[:4]]
        shares = np.asarray([_safe_float(row.get("weighted_share_sum")) for row in pair_summary[:4]], dtype=float)
        fail_rates = np.asarray([_safe_float(row.get("tensor_fail_rate_mean")) for row in pair_summary[:4]], dtype=float)
        x = np.arange(len(labels), dtype=float)
        width = 0.36
        axis_pair.bar(x - width / 2.0, shares, width=width, color="#9467bd", alpha=0.9, label="weighted reject share")
        axis_pair.bar(x + width / 2.0, fail_rates, width=width, color="#1f77b4", alpha=0.9, label="mean tensor fail rate")
        axis_pair.set_xticks(x)
        axis_pair.set_xticklabels(labels)
        axis_pair.legend(loc="best", fontsize=9)
    else:
        axis_pair.text(0.5, 0.5, "No pair decomposition available", ha="center", va="center", fontsize=12)
        axis_pair.set_xticks([])
    axis_pair.set_ylabel("normalized score")
    axis_pair.set_title("Dominant bottleneck pairs")
    axis_pair.grid(True, axis="y", alpha=0.25)

    figure.suptitle("Step 8.7.32.5: H1/L1/V1 network breakthrough audit", fontsize=14)
    figure.tight_layout(rect=[0.02, 0.02, 0.98, 0.96])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(figure)


def _aggregate_pair_bottlenecks(trial_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pair_accumulator: Dict[str, Dict[str, float]] = {}
    for row in trial_rows:
        pair = str(row.get("top_bottleneck_pair", "")).strip()
        if not pair:
            continue
        share = _safe_float(row.get("top_bottleneck_share"))
        fail_rate = _safe_float(row.get("top_bottleneck_fail_rate"))
        count = pair_accumulator.setdefault(pair, {"weighted_share_sum": 0.0, "fail_rate_sum": 0.0, "n": 0.0})
        if math.isfinite(share):
            count["weighted_share_sum"] += float(max(0.0, share))
        if math.isfinite(fail_rate):
            count["fail_rate_sum"] += float(max(0.0, fail_rate))
        count["n"] += 1.0

    output: List[Dict[str, Any]] = []
    for pair, values in pair_accumulator.items():
        n_samples = int(values["n"])
        output.append(
            {
                "pair": pair,
                "n_trials": n_samples,
                "weighted_share_sum": float(values["weighted_share_sum"]),
                "tensor_fail_rate_mean": float(values["fail_rate_sum"] / max(float(n_samples), 1.0)),
            }
        )
    output.sort(key=lambda row: (_safe_float(row.get("weighted_share_sum")), _safe_float(row.get("tensor_fail_rate_mean"))), reverse=True)
    for index, row in enumerate(output, start=1):
        row["bottleneck_rank"] = int(index)
    return output


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Step 8.7.32.5: H1/L1/V1 network breakthrough audit. "
            "Searches a fixed parameter envelope and records pass conditions "
            "or an explicit non-pass certificate inside that envelope."
        )
    )
    parser.add_argument("--events", type=str, default="GW200129_065458,GW200224_222234,GW200115_042309,GW200311_115853")
    parser.add_argument("--detectors", type=str, default="H1,L1,V1")
    parser.add_argument("--corr-use-min-grid", type=str, default="0.03,0.05,0.07")
    parser.add_argument("--response-floor-grid", type=str, default="0.08,0.10")
    parser.add_argument("--min-ring-grid", type=str, default="6,8,10")
    parser.add_argument("--geometry-relax-grid", type=str, default="2.0,4.0")
    parser.add_argument("--geometry-delay-floor-ms-grid", type=str, default="0.25")
    parser.add_argument("--pair-pruning-grid", type=str, default="0,1")
    parser.add_argument("--sky-samples", type=int, default=5000)
    parser.add_argument("--psi-samples", type=int, default=36)
    parser.add_argument("--cosi-samples", type=int, default=41)
    parser.add_argument("--keep-trial-artifacts", action="store_true")
    parser.add_argument("--trial-prefix", type=str, default="tmp_ng_breakthrough")
    parser.add_argument("--outdir", type=str, default=str(_ROOT / "output" / "private" / "gw"))
    parser.add_argument("--public-outdir", type=str, default=str(_ROOT / "output" / "public" / "gw"))
    parser.add_argument("--prefix", type=str, default="gw_polarization_h1_l1_v1_network_breakthrough_audit")
    args = parser.parse_args(list(argv) if argv is not None else None)

    corr_use_min_grid = _parse_float_grid(str(args.corr_use_min_grid))
    response_floor_grid = _parse_float_grid(str(args.response_floor_grid))
    min_ring_grid = _parse_int_grid(str(args.min_ring_grid))
    geometry_relax_grid = _parse_float_grid(str(args.geometry_relax_grid))
    geometry_delay_floor_ms_grid = _parse_float_grid(str(args.geometry_delay_floor_ms_grid))
    pair_pruning_grid = _parse_bool_grid(str(args.pair_pruning_grid))
    if not (corr_use_min_grid and response_floor_grid and min_ring_grid and geometry_relax_grid and geometry_delay_floor_ms_grid and pair_pruning_grid):
        print("[err] tuning grids must not be empty")
        return 2

    outdir = Path(str(args.outdir))
    public_outdir = Path(str(args.public_outdir))
    outdir.mkdir(parents=True, exist_ok=True)
    public_outdir.mkdir(parents=True, exist_ok=True)

    network_script = _ROOT / "scripts" / "gw" / "gw_polarization_h1_l1_v1_network_audit.py"
    if not network_script.exists():
        print(f"[err] missing script: {network_script}")
        return 2

    trial_rows: List[Dict[str, Any]] = []
    trial_counter = 0
    for corr_use_min, response_floor, min_ring_directions, geometry_relax_factor, geometry_delay_floor_ms, allow_pair_pruning in product(
        corr_use_min_grid,
        response_floor_grid,
        min_ring_grid,
        geometry_relax_grid,
        geometry_delay_floor_ms_grid,
        pair_pruning_grid,
    ):
        trial_counter += 1
        trial_id = f"{args.trial_prefix}_{trial_counter:04d}"
        trial_json = outdir / f"{trial_id}.json"
        trial_csv = outdir / f"{trial_id}.csv"
        trial_png = outdir / f"{trial_id}.png"
        trial_reject_detail_csv = outdir / f"{trial_id}_reject_factor_details.csv"
        trial_reject_summary_csv = outdir / f"{trial_id}_reject_factor_decomposition.csv"
        trial_reject_summary_json = outdir / f"{trial_id}_reject_factor_decomposition.json"
        trial_reject_summary_png = outdir / f"{trial_id}_reject_factor_decomposition.png"

        command: List[str] = [
            sys.executable,
            "-B",
            str(network_script),
            "--events",
            str(args.events),
            "--detectors",
            str(args.detectors),
            "--corr-use-min",
            str(float(corr_use_min)),
            "--sky-samples",
            str(int(args.sky_samples)),
            "--psi-samples",
            str(int(args.psi_samples)),
            "--cosi-samples",
            str(int(args.cosi_samples)),
            "--response-floor-frac",
            str(float(response_floor)),
            "--min-ring-directions",
            str(int(min_ring_directions)),
            "--geometry-relax-factor",
            str(float(geometry_relax_factor)),
            "--geometry-delay-floor-ms",
            str(float(geometry_delay_floor_ms)),
            "--outdir",
            str(outdir),
            "--public-outdir",
            str(public_outdir),
            "--prefix",
            str(trial_id),
        ]
        if bool(allow_pair_pruning):
            command.append("--allow-pair-pruning")

        process = subprocess.run(command, cwd=str(_ROOT), capture_output=True, text=True)
        status = "subprocess_error"
        reason = "subprocess_nonzero_exit"
        n_usable_events = 0
        n_pair_pruned_events = 0
        tensor_reject_events = 0
        watch_events = 0
        scalar_overlap_proxy = float("nan")
        top_bottleneck_pair = ""
        top_bottleneck_share = float("nan")
        top_bottleneck_fail_rate = float("nan")

        if process.returncode == 0 and trial_json.exists():
            try:
                payload = _load_json(trial_json)
                summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
                rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
                status = str(summary.get("overall_status", "inconclusive"))
                reason = str(summary.get("overall_reason", "no_reason"))
                n_usable_events = int(_safe_float(summary.get("n_usable_events")))
                n_pair_pruned_events = int(_safe_float(summary.get("n_pair_pruned_events")))
                scalar_overlap_proxy = _safe_float(summary.get("scalar_only_mode_global_upper_bound_proxy"))
                tensor_reject_events = int(
                    sum(1 for row in rows if isinstance(row, dict) and str(row.get("status", "")) == "reject_tensor_response")
                )
                watch_events = int(
                    sum(1 for row in rows if isinstance(row, dict) and str(row.get("status", "")).startswith("watch_"))
                )

                outputs = payload.get("outputs") if isinstance(payload.get("outputs"), dict) else {}
                reject_summary_path_raw = str(outputs.get("reject_factor_summary_json", "") or "")
                if reject_summary_path_raw:
                    reject_summary_path = Path(reject_summary_path_raw.replace("/", "\\"))
                    if reject_summary_path.exists():
                        reject_summary_payload = _load_json(reject_summary_path)
                        pair_rows = (
                            reject_summary_payload.get("pair_summary_rows")
                            if isinstance(reject_summary_payload.get("pair_summary_rows"), list)
                            else []
                        )
                        if pair_rows:
                            first_pair = pair_rows[0] if isinstance(pair_rows[0], dict) else {}
                            top_bottleneck_pair = str(first_pair.get("pair", ""))
                            top_bottleneck_share = _safe_float(first_pair.get("mean_reject_tensor_mismatch_share"))
                            top_bottleneck_fail_rate = _safe_float(first_pair.get("tensor_fail_rate"))
            except Exception:
                status = "parse_error"
                reason = "trial_json_parse_failed"

        trial_rows.append(
            {
                "trial_id": trial_id,
                "status": status,
                "reason": reason,
                "n_usable_events": int(n_usable_events),
                "n_pair_pruned_events": int(n_pair_pruned_events),
                "tensor_reject_events": int(tensor_reject_events),
                "watch_events": int(watch_events),
                "scalar_overlap_proxy": float(scalar_overlap_proxy),
                "corr_use_min": float(corr_use_min),
                "response_floor_frac": float(response_floor),
                "min_ring_directions": int(min_ring_directions),
                "geometry_relax_factor": float(geometry_relax_factor),
                "geometry_delay_floor_ms": float(geometry_delay_floor_ms),
                "allow_pair_pruning": bool(allow_pair_pruning),
                "top_bottleneck_pair": top_bottleneck_pair,
                "top_bottleneck_share": float(top_bottleneck_share),
                "top_bottleneck_fail_rate": float(top_bottleneck_fail_rate),
                "returncode": int(process.returncode),
                "trial_json": str(trial_json).replace("\\", "/"),
            }
        )

        if not bool(args.keep_trial_artifacts):
            cleanup_paths = [
                trial_json,
                trial_csv,
                trial_png,
                trial_reject_detail_csv,
                trial_reject_summary_csv,
                trial_reject_summary_json,
                trial_reject_summary_png,
                public_outdir / f"{trial_id}.json",
                public_outdir / f"{trial_id}.csv",
                public_outdir / f"{trial_id}.png",
                public_outdir / f"{trial_id}_reject_factor_details.csv",
                public_outdir / f"{trial_id}_reject_factor_decomposition.csv",
                public_outdir / f"{trial_id}_reject_factor_decomposition.json",
                public_outdir / f"{trial_id}_reject_factor_decomposition.png",
            ]
            for path in cleanup_paths:
                if path.exists():
                    path.unlink()

    sorted_rows = sorted(trial_rows, key=_objective_key, reverse=True)
    status_counts: Dict[str, int] = {}
    for row in trial_rows:
        bucket = _status_bucket(str(row.get("status", "")))
        status_counts[bucket] = int(status_counts.get(bucket, 0)) + 1

    pass_found = bool(any(_status_bucket(str(row.get("status", ""))) == "pass" for row in trial_rows))
    watch_found = bool(any(_status_bucket(str(row.get("status", ""))) == "watch" for row in trial_rows))

    best_by_objective = sorted_rows[0] if sorted_rows else {}
    finite_scalar_rows = [row for row in trial_rows if math.isfinite(_safe_float(row.get("scalar_overlap_proxy")))]
    finite_scalar_rows.sort(
        key=lambda row: (
            _safe_float(row.get("scalar_overlap_proxy")),
            -int(row.get("n_usable_events", 0)),
            -_status_rank(str(row.get("status", ""))),
        )
    )
    best_scalar_any = finite_scalar_rows[0] if finite_scalar_rows else {}
    usable2_rows = [row for row in finite_scalar_rows if int(row.get("n_usable_events", 0)) >= 2]
    best_scalar_usable2 = usable2_rows[0] if usable2_rows else {}

    pair_bottleneck_summary = _aggregate_pair_bottlenecks(trial_rows)

    if pass_found:
        decision_status = "pass"
        decision = "network_breakthrough_pass"
        decision_reason = "at_least_one_trial_reaches_pass_with_three_detector_constraints"
    elif watch_found:
        decision_status = "watch"
        decision = "network_breakthrough_watch"
        decision_reason = "no_pass_found_in_search_envelope_but_watch_trials_exist"
    else:
        decision_status = "reject"
        decision = "network_breakthrough_reject_in_envelope"
        decision_reason = "no_pass_or_watch_found_in_search_envelope"

    out_json = outdir / f"{args.prefix}.json"
    out_csv = outdir / f"{args.prefix}.csv"
    out_png = outdir / f"{args.prefix}.png"

    _write_csv(out_csv, sorted_rows)
    _plot(sorted_rows, status_counts, pair_bottleneck_summary, out_png)

    payload: Dict[str, Any] = {
        "generated_utc": _iso_utc_now(),
        "schema": "wavep.gw.polarization.network_breakthrough_audit.v1",
        "phase": 8,
        "step": "8.7.32.5",
        "intent": (
            "Search H1/L1/V1 network-gate envelope and fix either "
            "a breakthrough pass condition or a non-pass certificate "
            "for the scanned envelope."
        ),
        "inputs": {
            "events": [item.strip() for item in str(args.events).split(",") if item.strip()],
            "detectors": [item.strip().upper() for item in str(args.detectors).split(",") if item.strip()],
            "corr_use_min_grid": [float(value) for value in corr_use_min_grid],
            "response_floor_grid": [float(value) for value in response_floor_grid],
            "min_ring_grid": [int(value) for value in min_ring_grid],
            "geometry_relax_grid": [float(value) for value in geometry_relax_grid],
            "geometry_delay_floor_ms_grid": [float(value) for value in geometry_delay_floor_ms_grid],
            "pair_pruning_grid": [bool(value) for value in pair_pruning_grid],
            "sky_samples": int(args.sky_samples),
            "psi_samples": int(args.psi_samples),
            "cosi_samples": int(args.cosi_samples),
            "n_trials": int(len(trial_rows)),
        },
        "summary": {
            "status_counts": status_counts,
            "pass_found": bool(pass_found),
            "watch_found": bool(watch_found),
            "best_by_objective": best_by_objective,
            "best_scalar_proxy_any": best_scalar_any,
            "best_scalar_proxy_usable2": best_scalar_usable2,
            "dominant_bottleneck_pairs": pair_bottleneck_summary[:4],
        },
        "decision": {
            "overall_status": decision_status,
            "decision": decision,
            "reason": decision_reason,
            "certificate_in_scope": {
                "scope": "scanned_parameter_envelope_only",
                "non_pass_certificate": bool(not pass_found),
                "non_pass_certificate_reason": (
                    "no trial reached pass in scanned envelope"
                    if not pass_found
                    else "not_applicable"
                ),
            },
        },
        "rows": sorted_rows,
        "outputs": {
            "audit_json": str(out_json).replace("\\", "/"),
            "audit_csv": str(out_csv).replace("\\", "/"),
            "audit_png": str(out_png).replace("\\", "/"),
        },
        "falsification_gate": {
            "pass_if": [
                "any trial status is pass",
            ],
            "watch_if": [
                "no pass but at least one watch trial in scanned envelope",
            ],
            "reject_if": [
                "no pass and no watch trial in scanned envelope",
            ],
        },
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    public_copies: List[str] = []
    for source in [out_json, out_csv, out_png]:
        destination = public_outdir / source.name
        shutil.copy2(source, destination)
        public_copies.append(str(destination).replace("\\", "/"))
    payload["outputs"]["public_copies"] = public_copies
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(out_json, public_outdir / out_json.name)

    worklog.append_event(
        {
            "event_type": "gw_polarization_h1_l1_v1_network_breakthrough_audit",
            "generated_utc": payload["generated_utc"],
            "summary": payload["summary"],
            "decision": payload["decision"],
            "outputs": payload["outputs"],
        }
    )

    print(f"[ok] trials={len(trial_rows)} pass_found={int(pass_found)} watch_found={int(watch_found)}")
    print(
        "[ok] best status={status} usable={usable} scalar={scalar} cfg=(corr={corr}, floor={floor}, ring={ring}, relax={relax}, delay={delay}, prune={prune})".format(
            status=str(best_by_objective.get("status", "")),
            usable=int(best_by_objective.get("n_usable_events", 0)) if best_by_objective else 0,
            scalar=_fmt(_safe_float(best_by_objective.get("scalar_overlap_proxy")) if best_by_objective else float("nan")),
            corr=_fmt(_safe_float(best_by_objective.get("corr_use_min")) if best_by_objective else float("nan")),
            floor=_fmt(_safe_float(best_by_objective.get("response_floor_frac")) if best_by_objective else float("nan")),
            ring=int(best_by_objective.get("min_ring_directions", 0)) if best_by_objective else 0,
            relax=_fmt(_safe_float(best_by_objective.get("geometry_relax_factor")) if best_by_objective else float("nan")),
            delay=_fmt(_safe_float(best_by_objective.get("geometry_delay_floor_ms")) if best_by_objective else float("nan")),
            prune=int(bool(best_by_objective.get("allow_pair_pruning", False))) if best_by_objective else 0,
        )
    )
    print(f"[ok] decision={decision} status={decision_status}")
    print(f"[ok] json: {out_json}")
    print(f"[ok] csv : {out_csv}")
    print(f"[ok] png : {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
