from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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


def _safe_float(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return float("nan")
    return x


def _status_rank(status: str) -> int:
    s = str(status or "")
    if s.startswith("pass"):
        return 3
    if s.startswith("watch"):
        return 2
    if s.startswith("reject"):
        return 1
    if s.startswith("inconclusive"):
        return 0
    return -1


def _fmt(v: Any, digits: int = 6) -> str:
    x = _safe_float(v)
    if not math.isfinite(x):
        return ""
    if x == 0.0:
        return "0"
    ax = abs(x)
    if ax >= 1e4 or ax < 1e-3:
        return f"{x:.{digits}g}"
    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


def _parse_events(raw: str) -> List[str]:
    return [s.strip() for s in str(raw).split(",") if s.strip()]


def _parse_float_grid(raw: str) -> List[float]:
    out: List[float] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(float(token))
        except Exception:
            continue
    return sorted(set(out))


def _parse_int_grid(raw: str) -> List[int]:
    out: List[int] = []
    for token in str(raw).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            out.append(int(token))
        except Exception:
            continue
    return sorted(set(out))


def _subset_iter(events: List[str], min_size: int, max_size: int, anchor_event: str) -> Iterable[Tuple[str, ...]]:
    n = len(events)
    lo = max(1, min_size)
    hi = max(lo, min(max_size, n))
    anchor = str(anchor_event or "").strip()
    for size in range(lo, hi + 1):
        for subset in combinations(events, size):
            if anchor and anchor not in subset:
                continue
            yield subset


def _row_key_for_gate(row: Dict[str, Any]) -> Tuple[float, int, int]:
    return (
        _safe_float(row.get("scalar_overlap_proxy")),
        -int(row.get("n_usable_events", 0)),
        -_status_rank(str(row.get("status", ""))),
    )


def _row_key_for_coverage(row: Dict[str, Any]) -> Tuple[int, int, float]:
    return (
        int(row.get("n_usable_events", 0)),
        _status_rank(str(row.get("status", ""))),
        -_safe_float(row.get("scalar_overlap_proxy")),
    )


def _pick_best(rows: List[Dict[str, Any]], pred) -> Optional[Dict[str, Any]]:
    cand = [row for row in rows if pred(row)]
    if not cand:
        return None
    cand.sort(key=_row_key_for_gate)
    return cand[0]


def _pick_best_coverage(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    rows_sorted = sorted(rows, key=_row_key_for_coverage, reverse=True)
    return rows_sorted[0]


def _pick_best_subset_coverage(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            int(r.get("best_coverage_n_usable_events", 0)),
            _status_rank(str(r.get("best_coverage_status", ""))),
            -_safe_float(r.get("best_coverage_scalar_overlap_proxy")),
        ),
        reverse=True,
    )
    return rows_sorted[0]


def _as_metric_row(subset: Tuple[str, ...], payload: Dict[str, Any], gate_scalar_max: float, gate_usable_min: int) -> Dict[str, Any]:
    rows = [row for row in payload.get("rows", []) if isinstance(row, dict)]
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    best_gate = _pick_best(
        rows,
        lambda r: int(r.get("n_usable_events", 0)) >= int(gate_usable_min)
        and math.isfinite(_safe_float(r.get("scalar_overlap_proxy")))
        and _safe_float(r.get("scalar_overlap_proxy")) < float(gate_scalar_max),
    )
    best_u2 = _pick_best(
        rows,
        lambda r: int(r.get("n_usable_events", 0)) >= int(gate_usable_min)
        and math.isfinite(_safe_float(r.get("scalar_overlap_proxy"))),
    )
    best_cov = _pick_best_coverage(rows)
    pass_found = bool(summary.get("pass_found"))
    status_counts = summary.get("status_counts") if isinstance(summary.get("status_counts"), dict) else {}
    return {
        "subset_events": ",".join(subset),
        "subset_size": int(len(subset)),
        "n_trials": int(summary.get("n_trials", len(rows))),
        "pass_found": int(pass_found),
        "status_counts_reject": int(status_counts.get("reject", 0)),
        "status_counts_watch": int(status_counts.get("watch", 0)),
        "status_counts_pass": int(status_counts.get("pass", 0)),
        "status_counts_inconclusive": int(status_counts.get("inconclusive", 0)),
        "best_gate_found": int(best_gate is not None),
        "best_gate_scalar_overlap_proxy": _safe_float(best_gate.get("scalar_overlap_proxy")) if best_gate else float("nan"),
        "best_gate_n_usable_events": int(best_gate.get("n_usable_events", 0)) if best_gate else 0,
        "best_gate_status": str(best_gate.get("status", "")) if best_gate else "",
        "best_u2_scalar_overlap_proxy": _safe_float(best_u2.get("scalar_overlap_proxy")) if best_u2 else float("nan"),
        "best_u2_n_usable_events": int(best_u2.get("n_usable_events", 0)) if best_u2 else 0,
        "best_u2_status": str(best_u2.get("status", "")) if best_u2 else "",
        "best_coverage_n_usable_events": int(best_cov.get("n_usable_events", 0)) if best_cov else 0,
        "best_coverage_scalar_overlap_proxy": _safe_float(best_cov.get("scalar_overlap_proxy")) if best_cov else float("nan"),
        "best_coverage_status": str(best_cov.get("status", "")) if best_cov else "",
        "best_coverage_corr_use_min": _safe_float(best_cov.get("corr_use_min")) if best_cov else float("nan"),
        "best_coverage_response_floor_frac": _safe_float(best_cov.get("response_floor_frac")) if best_cov else float("nan"),
        "best_coverage_min_ring_directions": int(best_cov.get("min_ring_directions", 0)) if best_cov else 0,
        "best_coverage_geometry_relax_factor": _safe_float(best_cov.get("geometry_relax_factor")) if best_cov else float("nan"),
        "best_coverage_allow_pair_pruning": int(bool(best_cov.get("allow_pair_pruning", False))) if best_cov else 0,
    }


def _plot(rows: List[Dict[str, Any]], gate_scalar_max: float, out_png: Path) -> None:
    labels = [str(row.get("subset_events", "")) for row in rows]
    y_scalar = np.asarray([_safe_float(row.get("best_u2_scalar_overlap_proxy")) for row in rows], dtype=float)
    y_usable = np.asarray([float(row.get("best_coverage_n_usable_events", 0)) for row in rows], dtype=float)
    pass_found = np.asarray([float(row.get("best_gate_found", 0)) for row in rows], dtype=float)

    x = np.arange(len(rows), dtype=float)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    colors = []
    for value in y_scalar:
        if not math.isfinite(float(value)):
            colors.append("#9aa0a6")
        elif value < float(gate_scalar_max):
            colors.append("#2ca02c")
        elif value <= 0.5:
            colors.append("#f2c744")
        else:
            colors.append("#d62728")
    ax1.bar(x, y_scalar, color=colors, alpha=0.9)
    ax1.axhline(float(gate_scalar_max), color="#2ca02c", linestyle="--", linewidth=1.4, label=f"gate scalar<{_fmt(gate_scalar_max)}")
    ax1.axhline(0.5, color="#f2c744", linestyle=":", linewidth=1.2, label="watch boundary = 0.5")
    ax1.set_ylabel("best scalar_overlap_proxy (usable>=2)")
    ax1.set_title("GW polarization coverage expansion audit (subset sweep)")
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend(loc="upper right", frameon=True)

    width = 0.42
    ax2.bar(x - width / 2.0, y_usable, width=width, color="#4c78a8", alpha=0.9, label="best usable events")
    ax2.bar(x + width / 2.0, pass_found, width=width, color="#59a14f", alpha=0.9, label="gate_found(0/1)")
    ax2.axhline(2.0, color="#7f7f7f", linestyle="--", linewidth=1.0, label="coverage target usable>=2")
    ax2.set_ylabel("coverage / gate")
    ax2.set_ylim(0.0, max(3.0, float(np.nanmax(y_usable)) + 0.8 if y_usable.size else 3.0))
    ax2.grid(True, axis="y", alpha=0.25)
    ax2.legend(loc="upper right", frameon=True)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=35, ha="right")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            row_out = {
                key: (
                    _fmt(val)
                    if isinstance(val, float)
                    else str(int(val))
                    if isinstance(val, bool)
                    else val
                )
                for key, val in row.items()
            }
            writer.writerow(row_out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Step 8.7.19.9: coverage expansion audit for H1/L1/V1 polarization network. "
            "Sweeps event subsets (anchor-constrained) and fixed local tuning envelope."
        )
    )
    ap.add_argument("--events", type=str, default="GW200129_065458,GW200224_222234,GW200115_042309,GW200311_115853")
    ap.add_argument("--anchor-event", type=str, default="GW200129_065458")
    ap.add_argument("--subset-min-size", type=int, default=2)
    ap.add_argument("--subset-max-size", type=int, default=4)
    ap.add_argument("--corr-grid", type=str, default="0.05,0.06,0.07")
    ap.add_argument("--response-floor-grid", type=str, default="0.05,0.08")
    ap.add_argument("--min-ring-grid", type=str, default="6,8")
    ap.add_argument("--relax-grid", type=str, default="2.0,4.0")
    ap.add_argument("--pair-pruning-grid", type=str, default="0")
    ap.add_argument("--geometry-delay-floor-ms", type=float, default=0.25)
    ap.add_argument("--gate-usable-min", type=int, default=2)
    ap.add_argument("--gate-scalar-max", type=float, default=0.5)
    ap.add_argument("--sky-samples", type=int, default=5000)
    ap.add_argument("--psi-samples", type=int, default=36)
    ap.add_argument("--cosi-samples", type=int, default=41)
    ap.add_argument("--outdir", type=str, default=str(_ROOT / "output" / "private" / "gw"))
    ap.add_argument("--public-outdir", type=str, default=str(_ROOT / "output" / "public" / "gw"))
    ap.add_argument("--prefix", type=str, default="gw_polarization_network_coverage_expansion_audit")
    args = ap.parse_args(list(argv) if argv is not None else None)

    events = _parse_events(args.events)
    if len(events) < 2:
        print("[err] need at least two events")
        return 2
    if args.anchor_event and args.anchor_event not in events:
        print(f"[err] anchor event not in events: {args.anchor_event}")
        return 2

    corr_grid = _parse_float_grid(args.corr_grid)
    floor_grid = _parse_float_grid(args.response_floor_grid)
    min_ring_grid = _parse_int_grid(args.min_ring_grid)
    relax_grid = _parse_float_grid(args.relax_grid)
    if not (corr_grid and floor_grid and min_ring_grid and relax_grid):
        print("[err] tuning grids must not be empty")
        return 2

    outdir = Path(args.outdir)
    public_outdir = Path(args.public_outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    public_outdir.mkdir(parents=True, exist_ok=True)
    tuning_script = _ROOT / "scripts" / "gw" / "gw_polarization_h1_l1_v1_network_tuning_audit.py"
    if not tuning_script.exists():
        print(f"[err] missing tuning script: {tuning_script}")
        return 2

    subset_rows: List[Dict[str, Any]] = []
    subset_payloads: List[Dict[str, Any]] = []
    subsets = list(_subset_iter(events, int(args.subset_min_size), int(args.subset_max_size), str(args.anchor_event)))
    if not subsets:
        print("[err] no subsets generated (check subset range / anchor)")
        return 2

    for index, subset in enumerate(subsets, start=1):
        subset_id = f"tmp_covexp_{index:03d}"
        subset_prefix = f"{subset_id}_tuning"
        subset_trial_prefix = f"tmp_covexp_trial_{index:03d}"
        cmd = [
            sys.executable,
            "-B",
            str(tuning_script),
            "--events",
            ",".join(subset),
            "--corr-grid",
            ",".join(_fmt(v, 4) for v in corr_grid),
            "--response-floor-grid",
            ",".join(_fmt(v, 4) for v in floor_grid),
            "--min-ring-grid",
            ",".join(str(v) for v in min_ring_grid),
            "--relax-grid",
            ",".join(_fmt(v, 4) for v in relax_grid),
            "--pair-pruning-grid",
            str(args.pair_pruning_grid),
            "--geometry-delay-floor-ms",
            _fmt(args.geometry_delay_floor_ms, 4),
            "--sky-samples",
            str(int(args.sky_samples)),
            "--psi-samples",
            str(int(args.psi_samples)),
            "--cosi-samples",
            str(int(args.cosi_samples)),
            "--trial-prefix",
            subset_trial_prefix,
            "--prefix",
            subset_prefix,
            "--outdir",
            str(outdir),
            "--public-outdir",
            str(public_outdir),
        ]
        proc = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True, check=False)
        tuning_json = outdir / f"{subset_prefix}.json"
        if proc.returncode != 0 or not tuning_json.exists():
            subset_rows.append(
                {
                    "subset_events": ",".join(subset),
                    "subset_size": int(len(subset)),
                    "n_trials": 0,
                    "pass_found": 0,
                    "status_counts_reject": 0,
                    "status_counts_watch": 0,
                    "status_counts_pass": 0,
                    "status_counts_inconclusive": 0,
                    "best_gate_found": 0,
                    "best_gate_scalar_overlap_proxy": float("nan"),
                    "best_gate_n_usable_events": 0,
                    "best_gate_status": "",
                    "best_u2_scalar_overlap_proxy": float("nan"),
                    "best_u2_n_usable_events": 0,
                    "best_u2_status": "",
                    "best_coverage_n_usable_events": 0,
                    "best_coverage_scalar_overlap_proxy": float("nan"),
                    "best_coverage_status": "inconclusive",
                    "best_coverage_corr_use_min": float("nan"),
                    "best_coverage_response_floor_frac": float("nan"),
                    "best_coverage_min_ring_directions": 0,
                    "best_coverage_geometry_relax_factor": float("nan"),
                    "best_coverage_allow_pair_pruning": 0,
                }
            )
            subset_payloads.append(
                {
                    "subset_events": list(subset),
                    "returncode": int(proc.returncode),
                    "error": "tuning_failed_or_missing_json",
                    "stdout_tail": str(proc.stdout or "")[-2000:],
                    "stderr_tail": str(proc.stderr or "")[-2000:],
                }
            )
            continue

        payload = json.loads(tuning_json.read_text(encoding="utf-8"))
        metric_row = _as_metric_row(
            subset=subset,
            payload=payload,
            gate_scalar_max=float(args.gate_scalar_max),
            gate_usable_min=int(args.gate_usable_min),
        )
        subset_rows.append(metric_row)
        subset_payloads.append(
            {
                "subset_events": list(subset),
                "returncode": int(proc.returncode),
                "summary": payload.get("summary", {}),
            }
        )
        for suffix in (".json", ".csv", ".png"):
            for base_dir in (outdir, public_outdir):
                tmp_path = base_dir / f"{subset_prefix}{suffix}"
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except Exception:
                        pass
        for base_dir in (outdir, public_outdir):
            for trial_path in base_dir.glob(f"{subset_trial_prefix}_*"):
                if not trial_path.is_file():
                    continue
                try:
                    trial_path.unlink()
                except Exception:
                    pass

    gate_hits = [row for row in subset_rows if int(row.get("best_gate_found", 0)) > 0]
    with_u2 = [row for row in subset_rows if int(row.get("best_u2_n_usable_events", 0)) >= int(args.gate_usable_min)]
    best_u2_all = None
    if with_u2:
        best_u2_all = sorted(
            with_u2,
            key=lambda r: (
                _safe_float(r.get("best_u2_scalar_overlap_proxy")),
                -int(r.get("best_u2_n_usable_events", 0)),
                -_status_rank(str(r.get("best_u2_status", ""))),
            ),
        )[0]
    best_coverage = _pick_best_subset_coverage(subset_rows)

    if gate_hits:
        overall_status = "pass"
        decision = "coverage_gate_pass_found"
        reason = "at_least_one_subset_reaches_scalar_gate_under_usable_coverage"
    elif with_u2:
        overall_status = "watch"
        decision = "coverage_gate_not_reached_boundary_locked"
        reason = "usable_coverage_exists_but_scalar_gate_not_reached"
    else:
        overall_status = "reject"
        decision = "coverage_not_reached"
        reason = "no_subset_reaches_minimum_usable_coverage"

    out_json = outdir / f"{args.prefix}.json"
    out_csv = outdir / f"{args.prefix}.csv"
    out_png = outdir / f"{args.prefix}.png"
    _write_csv(out_csv, subset_rows)
    _plot(subset_rows, gate_scalar_max=float(args.gate_scalar_max), out_png=out_png)

    payload_out: Dict[str, Any] = {
        "generated_utc": _iso_utc_now(),
        "schema": "wavep.gw.polarization.coverage_expansion_audit.v1",
        "phase": 8,
        "step": "8.7.19.9",
        "inputs": {
            "events": events,
            "anchor_event": str(args.anchor_event),
            "subset_min_size": int(args.subset_min_size),
            "subset_max_size": int(args.subset_max_size),
            "corr_grid": corr_grid,
            "response_floor_grid": floor_grid,
            "min_ring_grid": min_ring_grid,
            "relax_grid": relax_grid,
            "pair_pruning_grid": str(args.pair_pruning_grid),
            "geometry_delay_floor_ms": float(args.geometry_delay_floor_ms),
            "gate_usable_min": int(args.gate_usable_min),
            "gate_scalar_max": float(args.gate_scalar_max),
            "sky_samples": int(args.sky_samples),
            "psi_samples": int(args.psi_samples),
            "cosi_samples": int(args.cosi_samples),
        },
        "summary": {
            "overall_status": overall_status,
            "decision": decision,
            "reason": reason,
            "n_subsets_tested": int(len(subset_rows)),
            "n_gate_hits": int(len(gate_hits)),
            "n_subsets_with_usable_min": int(len(with_u2)),
            "best_u2_all": best_u2_all,
            "best_coverage_all": best_coverage,
        },
        "rows": subset_rows,
        "subset_payloads": subset_payloads,
        "outputs": {
            "audit_json": str(out_json).replace("\\", "/"),
            "audit_csv": str(out_csv).replace("\\", "/"),
            "audit_png": str(out_png).replace("\\", "/"),
        },
        "falsification": {
            "hard_pass_if": [
                f"exists subset with n_usable_events>={int(args.gate_usable_min)} and scalar_overlap_proxy<{_fmt(args.gate_scalar_max, 4)}"
            ],
            "watch_if": [
                f"no hard pass but at least one subset with n_usable_events>={int(args.gate_usable_min)}"
            ],
            "reject_if": [
                f"all subsets have n_usable_events<{int(args.gate_usable_min)}"
            ],
        },
    }
    out_json.write_text(json.dumps(payload_out, ensure_ascii=False, indent=2), encoding="utf-8")

    copied: List[str] = []
    for src in [out_json, out_csv, out_png]:
        dst = public_outdir / src.name
        shutil.copy2(src, dst)
        copied.append(str(dst).replace("\\", "/"))
    payload_out["outputs"]["public_copies"] = copied
    out_json.write_text(json.dumps(payload_out, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(out_json, public_outdir / out_json.name)

    try:
        worklog.append_event(
            {
                "event_type": "gw_polarization_network_coverage_expansion_audit",
                "argv": list(sys.argv),
                "outputs": {"audit_json": out_json, "audit_csv": out_csv, "audit_png": out_png},
                "metrics": {
                    "overall_status": overall_status,
                    "decision": decision,
                    "n_subsets_tested": int(len(subset_rows)),
                    "n_gate_hits": int(len(gate_hits)),
                    "best_u2_scalar_overlap_proxy": _safe_float(best_u2_all.get("best_u2_scalar_overlap_proxy")) if isinstance(best_u2_all, dict) else float("nan"),
                    "best_u2_n_usable_events": int(best_u2_all.get("best_u2_n_usable_events", 0)) if isinstance(best_u2_all, dict) else 0,
                },
            }
        )
    except Exception:
        pass

    print(f"[ok] subsets={len(subset_rows)} gate_hits={len(gate_hits)} status={overall_status}")
    if isinstance(best_u2_all, dict):
        print(
            "[ok] best_u2 subset={subset} scalar={scalar} usable={usable}".format(
                subset=best_u2_all.get("subset_events", ""),
                scalar=_fmt(best_u2_all.get("best_u2_scalar_overlap_proxy")),
                usable=int(best_u2_all.get("best_u2_n_usable_events", 0)),
            )
        )
    print(f"[ok] json: {out_json}")
    print(f"[ok] csv : {out_csv}")
    print(f"[ok] png : {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
