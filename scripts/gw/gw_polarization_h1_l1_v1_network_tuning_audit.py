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
        import matplotlib.font_manager as fm

        preferred = [
            "Yu Gothic",
            "Meiryo",
            "BIZ UDGothic",
            "MS Gothic",
            "Yu Mincho",
            "MS Mincho",
        ]
        available = {f.name for f in fm.fontManager.ttflist}
        chosen = [name for name in preferred if name in available]
        if not chosen:
            return
        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
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
    x = float(value)
    if not math.isfinite(x):
        return ""
    if x == 0.0:
        return "0"
    ax = abs(x)
    if ax >= 1e4 or ax < 1e-3:
        return f"{x:.{digits}g}"
    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


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


def _status_rank(status: str) -> int:
    s = str(status or "")
    if s == "pass":
        return 3
    if s == "watch":
        return 2
    if s == "reject":
        return 1
    if s == "inconclusive":
        return 0
    return -1


def _status_bucket(status: str) -> str:
    s = str(status or "")
    if s.startswith("pass"):
        return "pass"
    if s.startswith("watch"):
        return "watch"
    if s.startswith("reject"):
        return "reject"
    if s.startswith("inconclusive"):
        return "inconclusive"
    return "other"


def _trial_sort_key(row: Dict[str, Any]) -> Tuple[int, int, float]:
    status = str(row.get("status", ""))
    n_usable = int(row.get("n_usable_events", 0))
    scalar = _safe_float(row.get("scalar_overlap_proxy"))
    scalar_term = scalar if math.isfinite(scalar) else 9.9e9
    return (_status_rank(status), n_usable, -scalar_term)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    headers = [
        "trial_id",
        "status",
        "reason",
        "n_usable_events",
        "n_pair_pruned_events",
        "scalar_overlap_proxy",
        "corr_use_min",
        "response_floor_frac",
        "min_ring_directions",
        "geometry_relax_factor",
        "geometry_delay_floor_ms",
        "allow_pair_pruning",
        "returncode",
        "trial_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for row in rows:
            vals: List[Any] = []
            for key in headers:
                val = row.get(key, "")
                if isinstance(val, float):
                    vals.append(_fmt(val))
                else:
                    vals.append(val)
            w.writerow(vals)


def _plot(rows: List[Dict[str, Any]], out_png: Path) -> None:
    _set_japanese_font()
    if not rows:
        fig, ax = plt.subplots(1, 1, figsize=(10.0, 5.0))
        ax.text(0.5, 0.5, "No tuning rows", ha="center", va="center", fontsize=14)
        ax.axis("off")
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return

    x = np.array([int(r.get("n_usable_events", 0)) for r in rows], dtype=float)
    y = np.array([_safe_float(r.get("scalar_overlap_proxy")) for r in rows], dtype=float)
    prune = np.array([int(bool(r.get("allow_pair_pruning", False))) for r in rows], dtype=int)
    status = [str(r.get("status", "")) for r in rows]
    colors = [{"pass": "#2ca02c", "watch": "#f2c744", "reject": "#d62728", "inconclusive": "#9aa0a6"}.get(_status_bucket(s), "#555555") for s in status]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12.8, 8.4))
    ax0.scatter(x + 0.02 * prune, y, c=colors, s=36, alpha=0.85, edgecolor="none")
    ax0.axhline(0.0, color="#666666", ls="--", lw=1.0, alpha=0.8)
    ax0.set_xlabel("n usable events")
    ax0.set_ylabel("scalar overlap proxy (max)")
    ax0.set_title("Network gate tuning space")
    ax0.grid(True, alpha=0.25)

    status_order = ["pass", "watch", "reject", "inconclusive", "other"]
    counts = [sum(1 for s in status if _status_bucket(s) == k) for k in status_order]
    ax1.bar(np.arange(len(status_order)), counts, color=["#2ca02c", "#f2c744", "#d62728", "#9aa0a6", "#555555"], alpha=0.9)
    ax1.set_xticks(np.arange(len(status_order)))
    ax1.set_xticklabels(status_order)
    ax1.set_ylabel("trial count")
    ax1.set_title("Trial status counts")
    ax1.grid(True, axis="y", alpha=0.25)

    fig.suptitle("GW polarization network tuning audit (Step 8.7.32.3 next)", fontsize=14)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Step 8.7.32.3 next: tune H1/L1/V1 network gate parameters and fix best-achievable region.")
    ap.add_argument("--events", type=str, default="GW200129_065458,GW200224_222234,GW200115_042309,GW200311_115853")
    ap.add_argument("--detectors", type=str, default="H1,L1,V1")
    ap.add_argument("--corr-grid", type=str, default="0.03,0.05")
    ap.add_argument("--response-floor-grid", type=str, default="0.05,0.10,0.15")
    ap.add_argument("--min-ring-grid", type=str, default="8,12")
    ap.add_argument("--relax-grid", type=str, default="2.0,4.0")
    ap.add_argument("--pair-pruning-grid", type=str, default="0,1")
    ap.add_argument("--geometry-delay-floor-ms", type=float, default=0.25)
    ap.add_argument("--sky-samples", type=int, default=5000)
    ap.add_argument("--psi-samples", type=int, default=36)
    ap.add_argument("--cosi-samples", type=int, default=41)
    ap.add_argument("--keep-trial-artifacts", action="store_true")
    ap.add_argument("--trial-prefix", type=str, default="tmp_ng_tune")
    ap.add_argument("--outdir", type=str, default=str(_ROOT / "output" / "private" / "gw"))
    ap.add_argument("--public-outdir", type=str, default=str(_ROOT / "output" / "public" / "gw"))
    ap.add_argument("--prefix", type=str, default="gw_polarization_h1_l1_v1_network_tuning_audit")
    args = ap.parse_args(list(argv) if argv is not None else None)

    corr_grid = _parse_float_grid(str(args.corr_grid))
    floor_grid = _parse_float_grid(str(args.response_floor_grid))
    min_ring_grid = _parse_int_grid(str(args.min_ring_grid))
    relax_grid = _parse_float_grid(str(args.relax_grid))
    prune_grid = _parse_bool_grid(str(args.pair_pruning_grid))
    if not corr_grid or not floor_grid or not min_ring_grid or not relax_grid or not prune_grid:
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

    rows: List[Dict[str, Any]] = []
    trial_index = 0
    for corr_use_min, response_floor, min_ring, relax_factor, allow_prune in product(
        corr_grid, floor_grid, min_ring_grid, relax_grid, prune_grid
    ):
        trial_index += 1
        trial_id = f"{args.trial_prefix}_{trial_index:04d}"
        trial_json = outdir / f"{trial_id}.json"
        trial_csv = outdir / f"{trial_id}.csv"
        trial_png = outdir / f"{trial_id}.png"
        cmd = [
            sys.executable,
            "-B",
            str(network_script),
            "--events",
            str(args.events),
            "--detectors",
            str(args.detectors),
            "--corr-use-min",
            str(float(corr_use_min)),
            "--response-floor-frac",
            str(float(response_floor)),
            "--min-ring-directions",
            str(int(min_ring)),
            "--geometry-relax-factor",
            str(float(relax_factor)),
            "--geometry-delay-floor-ms",
            str(float(args.geometry_delay_floor_ms)),
            "--sky-samples",
            str(int(args.sky_samples)),
            "--psi-samples",
            str(int(args.psi_samples)),
            "--cosi-samples",
            str(int(args.cosi_samples)),
            "--outdir",
            str(outdir),
            "--public-outdir",
            str(public_outdir),
            "--prefix",
            trial_id,
        ]
        if bool(allow_prune):
            cmd.append("--allow-pair-pruning")

        proc = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True)
        status = "subprocess_error"
        reason = "subprocess_nonzero_exit"
        n_usable = 0
        n_pruned = 0
        scalar_proxy = float("nan")
        if proc.returncode == 0 and trial_json.exists():
            try:
                payload = json.loads(trial_json.read_text(encoding="utf-8"))
                summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
                status = str(summary.get("overall_status", "inconclusive"))
                reason = str(summary.get("overall_reason", "no_reason"))
                n_usable = int(_safe_float(summary.get("n_usable_events")))
                n_pruned = int(_safe_float(summary.get("n_pair_pruned_events")))
                scalar_proxy = _safe_float(summary.get("scalar_only_mode_global_upper_bound_proxy"))
            except Exception:
                status = "parse_error"
                reason = "trial_json_parse_failed"

        rows.append(
            {
                "trial_id": trial_id,
                "status": status,
                "reason": reason,
                "n_usable_events": int(n_usable),
                "n_pair_pruned_events": int(n_pruned),
                "scalar_overlap_proxy": float(scalar_proxy),
                "corr_use_min": float(corr_use_min),
                "response_floor_frac": float(response_floor),
                "min_ring_directions": int(min_ring),
                "geometry_relax_factor": float(relax_factor),
                "geometry_delay_floor_ms": float(args.geometry_delay_floor_ms),
                "allow_pair_pruning": bool(allow_prune),
                "returncode": int(proc.returncode),
                "trial_json": str(trial_json).replace("\\", "/"),
            }
        )

        if not bool(args.keep_trial_artifacts):
            for path in [trial_json, trial_csv, trial_png]:
                if path.exists():
                    path.unlink()
            for dst in [public_outdir / f"{trial_id}.json", public_outdir / f"{trial_id}.csv", public_outdir / f"{trial_id}.png"]:
                if dst.exists():
                    dst.unlink()

    sorted_rows = sorted(rows, key=_trial_sort_key, reverse=True)
    finite_scalar_rows = [r for r in rows if math.isfinite(_safe_float(r.get("scalar_overlap_proxy")))]
    finite_scalar_rows.sort(key=lambda r: (_safe_float(r.get("scalar_overlap_proxy")), -int(r.get("n_usable_events", 0)), -_status_rank(str(r.get("status", "")))))
    top_scalar_rows = finite_scalar_rows[:5]

    default_row = None
    for row in rows:
        if (
            abs(float(row["corr_use_min"]) - 0.05) < 1e-12
            and abs(float(row["response_floor_frac"]) - 0.10) < 1e-12
            and int(row["min_ring_directions"]) == 8
            and abs(float(row["geometry_relax_factor"]) - 4.0) < 1e-12
            and not bool(row["allow_pair_pruning"])
        ):
            default_row = row
            break

    status_counts: Dict[str, int] = {}
    for row in rows:
        bucket = _status_bucket(str(row.get("status", "")))
        status_counts[bucket] = int(status_counts.get(bucket, 0)) + 1

    out_json = outdir / f"{args.prefix}.json"
    out_csv = outdir / f"{args.prefix}.csv"
    out_png = outdir / f"{args.prefix}.png"
    _write_csv(out_csv, sorted_rows)
    _plot(sorted_rows, out_png)

    best_coverage_row = sorted_rows[0] if sorted_rows else {}
    best_scalar_any = finite_scalar_rows[0] if finite_scalar_rows else {}
    usable2_rows = [r for r in finite_scalar_rows if int(r.get("n_usable_events", 0)) >= 2]
    best_scalar_min_usable2 = usable2_rows[0] if usable2_rows else {}

    payload: Dict[str, Any] = {
        "generated_utc": _iso_utc_now(),
        "schema": "wavep.gw.polarization.network_tuning_audit.v1",
        "phase": 8,
        "step": "8.7.32.3",
        "inputs": {
            "events": [s.strip() for s in str(args.events).split(",") if s.strip()],
            "detectors": [s.strip().upper() for s in str(args.detectors).split(",") if s.strip()],
            "corr_grid": [float(x) for x in corr_grid],
            "response_floor_grid": [float(x) for x in floor_grid],
            "min_ring_grid": [int(x) for x in min_ring_grid],
            "relax_grid": [float(x) for x in relax_grid],
            "pair_pruning_grid": [bool(x) for x in prune_grid],
            "geometry_delay_floor_ms": float(args.geometry_delay_floor_ms),
            "sky_samples": int(args.sky_samples),
            "psi_samples": int(args.psi_samples),
            "cosi_samples": int(args.cosi_samples),
            "keep_trial_artifacts": bool(args.keep_trial_artifacts),
        },
        "summary": {
            "n_trials": int(len(rows)),
            "status_counts": status_counts,
            "best_by_status_then_coverage": best_coverage_row,
            "best_scalar_proxy_any": best_scalar_any,
            "best_scalar_proxy_min_usable2": best_scalar_min_usable2,
            "top_scalar_proxy_rows": top_scalar_rows,
            "default_config_row": default_row or {},
            "pass_found": bool(any(str(r.get("status")) == "pass" for r in rows)),
        },
        "rows": sorted_rows,
        "outputs": {
            "audit_json": str(out_json).replace("\\", "/"),
            "audit_csv": str(out_csv).replace("\\", "/"),
            "audit_png": str(out_png).replace("\\", "/"),
        },
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    copied: List[str] = []
    for src in [out_json, out_csv, out_png]:
        dst = public_outdir / src.name
        shutil.copy2(src, dst)
        copied.append(str(dst).replace("\\", "/"))
    payload["outputs"]["public_copies"] = copied
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copy2(out_json, public_outdir / out_json.name)

    event = {
        "event_type": "gw_polarization_h1_l1_v1_network_tuning_audit",
        "generated_utc": payload["generated_utc"],
        "summary": payload["summary"],
        "outputs": payload["outputs"],
    }
    worklog.append_event(event)

    print(f"[ok] trials={len(rows)} pass_found={int(payload['summary']['pass_found'])}")
    best = payload["summary"]["best_by_status_then_coverage"] if isinstance(payload["summary"]["best_by_status_then_coverage"], dict) else {}
    best_scalar = payload["summary"]["best_scalar_proxy_any"] if isinstance(payload["summary"]["best_scalar_proxy_any"], dict) else {}
    best_scalar_u2 = payload["summary"]["best_scalar_proxy_min_usable2"] if isinstance(payload["summary"]["best_scalar_proxy_min_usable2"], dict) else {}
    print(
        "[ok] best(coverage) status={status} usable={usable} scalar={scalar} cfg=(corr={corr}, floor={floor}, min_ring={mr}, relax={relax}, prune={prune})".format(
            status=best.get("status", ""),
            usable=best.get("n_usable_events", ""),
            scalar=_fmt(_safe_float(best.get("scalar_overlap_proxy"))),
            corr=_fmt(_safe_float(best.get("corr_use_min"))),
            floor=_fmt(_safe_float(best.get("response_floor_frac"))),
            mr=best.get("min_ring_directions", ""),
            relax=_fmt(_safe_float(best.get("geometry_relax_factor"))),
            prune=int(bool(best.get("allow_pair_pruning", False))),
        )
    )
    if best_scalar:
        print(
            "[ok] best(scalar-any) status={status} usable={usable} scalar={scalar} cfg=(corr={corr}, floor={floor}, min_ring={mr}, relax={relax}, prune={prune})".format(
                status=best_scalar.get("status", ""),
                usable=best_scalar.get("n_usable_events", ""),
                scalar=_fmt(_safe_float(best_scalar.get("scalar_overlap_proxy"))),
                corr=_fmt(_safe_float(best_scalar.get("corr_use_min"))),
                floor=_fmt(_safe_float(best_scalar.get("response_floor_frac"))),
                mr=best_scalar.get("min_ring_directions", ""),
                relax=_fmt(_safe_float(best_scalar.get("geometry_relax_factor"))),
                prune=int(bool(best_scalar.get("allow_pair_pruning", False))),
            )
        )
    if best_scalar_u2:
        print(
            "[ok] best(scalar usable>=2) status={status} usable={usable} scalar={scalar} cfg=(corr={corr}, floor={floor}, min_ring={mr}, relax={relax}, prune={prune})".format(
                status=best_scalar_u2.get("status", ""),
                usable=best_scalar_u2.get("n_usable_events", ""),
                scalar=_fmt(_safe_float(best_scalar_u2.get("scalar_overlap_proxy"))),
                corr=_fmt(_safe_float(best_scalar_u2.get("corr_use_min"))),
                floor=_fmt(_safe_float(best_scalar_u2.get("response_floor_frac"))),
                mr=best_scalar_u2.get("min_ring_directions", ""),
                relax=_fmt(_safe_float(best_scalar_u2.get("geometry_relax_factor"))),
                prune=int(bool(best_scalar_u2.get("allow_pair_pruning", False))),
            )
        )
    print(f"[ok] json: {out_json}")
    print(f"[ok] csv : {out_csv}")
    print(f"[ok] png : {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
