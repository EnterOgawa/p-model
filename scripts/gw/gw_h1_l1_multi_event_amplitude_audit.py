from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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


def _slugify(s: str) -> str:
    out = "".join(ch.lower() if ch.isalnum() else "_" for ch in (s or "").strip())
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "event"


def _fmt(v: float, digits: int = 7) -> str:
    if not math.isfinite(float(v)):
        return ""
    x = float(v)
    if x == 0.0:
        return "0"
    ax = abs(x)
    if ax >= 1e4 or ax < 1e-3:
        return f"{x:.{digits}g}"
    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


def _load_metrics_json(event: str, slug: str) -> Optional[Dict[str, Any]]:
    candidates = [
        _ROOT / "output" / "public" / "gw" / f"{slug}_h1_l1_amplitude_ratio_metrics.json",
        _ROOT / "output" / "private" / "gw" / f"{slug}_h1_l1_amplitude_ratio_metrics.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["_metrics_path"] = str(path).replace("\\", "/")
                payload["_event_name_requested"] = event
                return payload
            except Exception:
                continue
    return None


def _event_row(event: str, payload: Optional[Dict[str, Any]], corr_use_min: float) -> Dict[str, Any]:
    if payload is None:
        return {
            "event": event,
            "slug": _slugify(event),
            "status": "missing",
            "quality": "missing",
            "best_lag_ms": "",
            "abs_corr": "",
            "ratio_median": "",
            "ratio_p16": "",
            "ratio_p84": "",
            "ratio_halfspan_over_median": "",
            "metrics_path": "",
        }

    event_meta = payload.get("event") if isinstance(payload.get("event"), dict) else {}
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    ratio = metrics.get("ratio") if isinstance(metrics.get("ratio"), dict) else {}
    gate = payload.get("gate") if isinstance(payload.get("gate"), dict) else {}
    status = str(gate.get("status") or "unknown")
    reasons = gate.get("status_reasons")
    reasons_text = ";".join(str(v) for v in reasons) if isinstance(reasons, list) else str(reasons or "")

    abs_corr = float(metrics.get("abs_best_corr")) if metrics.get("abs_best_corr") is not None else float("nan")
    ratio_median = float(ratio.get("median")) if ratio.get("median") is not None else float("nan")
    ratio_p16 = float(ratio.get("p16")) if ratio.get("p16") is not None else float("nan")
    ratio_p84 = float(ratio.get("p84")) if ratio.get("p84") is not None else float("nan")
    ratio_halfspan_over_median = (
        float((abs(ratio_p84 - ratio_p16) / 2.0) / ratio_median)
        if math.isfinite(ratio_median) and ratio_median != 0.0 and math.isfinite(ratio_p16) and math.isfinite(ratio_p84)
        else float("nan")
    )

    quality = "usable" if math.isfinite(abs_corr) and abs_corr >= float(corr_use_min) else "low_corr"
    return {
        "event": str(event_meta.get("name") or event),
        "slug": str(event_meta.get("slug") or _slugify(event)),
        "status": status,
        "status_reason": reasons_text,
        "quality": quality,
        "best_lag_ms": float(metrics.get("best_lag_ms_apply_to_h1"))
        if metrics.get("best_lag_ms_apply_to_h1") is not None
        else float("nan"),
        "abs_corr": abs_corr,
        "ratio_median": ratio_median,
        "ratio_p16": ratio_p16,
        "ratio_p84": ratio_p84,
        "ratio_halfspan_over_median": ratio_halfspan_over_median,
        "metrics_path": str(payload.get("_metrics_path") or ""),
    }


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "event",
        "slug",
        "status",
        "status_reason",
        "quality",
        "best_lag_ms",
        "abs_corr",
        "ratio_median",
        "ratio_p16",
        "ratio_p84",
        "ratio_halfspan_over_median",
        "metrics_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for row in rows:
            w.writerow([_fmt(row.get(h, float("nan"))) if isinstance(row.get(h), float) else row.get(h, "") for h in headers])


def _plot(rows: List[Dict[str, Any]], out_png: Path, corr_use_min: float) -> None:
    _set_japanese_font()
    labels = [str(r.get("event", "")) for r in rows]
    x = np.arange(len(rows), dtype=float)
    abs_corr = np.array([float(r.get("abs_corr", float("nan"))) for r in rows], dtype=float)
    ratio_med = np.array([float(r.get("ratio_median", float("nan"))) for r in rows], dtype=float)
    ratio_p16 = np.array([float(r.get("ratio_p16", float("nan"))) for r in rows], dtype=float)
    ratio_p84 = np.array([float(r.get("ratio_p84", float("nan"))) for r in rows], dtype=float)
    status = [str(r.get("status", "unknown")) for r in rows]

    colors = []
    for s in status:
        if s == "pass":
            colors.append("#2ca02c")
        elif s == "watch":
            colors.append("#f2c744")
        elif s == "reject":
            colors.append("#d62728")
        else:
            colors.append("#9c9c9c")

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(13.0, 7.8), sharex=True)

    ax0.bar(x, abs_corr, color=colors, alpha=0.9)
    ax0.axhline(float(corr_use_min), color="#444444", linestyle="--", linewidth=1.0, label=f"usable corr ≥ {corr_use_min:.2f}")
    ax0.set_ylabel("|corr| (H1/L1)")
    ax0.set_ylim(0.0, max(1.0, float(np.nanmax(abs_corr)) + 0.05 if np.isfinite(np.nanmax(abs_corr)) else 1.0))
    ax0.grid(True, axis="y", alpha=0.3)
    ax0.legend(loc="upper right")

    valid = np.isfinite(ratio_med) & np.isfinite(ratio_p16) & np.isfinite(ratio_p84)
    if np.any(valid):
        yerr = np.vstack(
            [
                np.maximum(ratio_med - ratio_p16, 0.0),
                np.maximum(ratio_p84 - ratio_med, 0.0),
            ]
        )
        ax1.errorbar(
            x[valid],
            ratio_med[valid],
            yerr=yerr[:, valid],
            fmt="o",
            color="#1f77b4",
            ecolor="#1f77b4",
            capsize=4.0,
            linewidth=1.3,
            markersize=5.0,
            label="|H1|/|L1| (median, p16–p84)",
        )
    ax1.set_ylabel("envelope ratio")
    ax1.set_xlabel("event")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend(loc="upper right")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=0, fontsize=9)
    fig.suptitle("GW multi-event H1/L1 amplitude-ratio audit (event-dependency check)", fontsize=14)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Step 8.7.19.1: multi-event H1/L1 amplitude-ratio audit.")
    ap.add_argument(
        "--events",
        type=str,
        default="GW150914,GW151226,GW170104,GW200129_065458",
        help="Comma-separated event names.",
    )
    ap.add_argument(
        "--corr-use-min",
        type=float,
        default=0.6,
        help="Minimum |corr| to count event as usable for event-dependency assessment.",
    )
    ap.add_argument("--outdir", type=str, default=str(_ROOT / "output" / "private" / "gw"))
    ap.add_argument("--public-outdir", type=str, default=str(_ROOT / "output" / "public" / "gw"))
    ap.add_argument("--prefix", type=str, default="gw_h1_l1_multi_event_amplitude_audit")
    args = ap.parse_args(list(argv) if argv is not None else None)

    events = [s.strip() for s in str(args.events).split(",") if s.strip()]
    if not events:
        print("[err] --events is empty")
        return 2

    rows: List[Dict[str, Any]] = []
    missing_events: List[str] = []
    for event in events:
        payload = _load_metrics_json(event, _slugify(event))
        if payload is None:
            missing_events.append(event)
        rows.append(_event_row(event, payload, float(args.corr_use_min)))

    usable_rows = [r for r in rows if str(r.get("quality")) == "usable"]
    ratio_vals = [float(r.get("ratio_median")) for r in usable_rows if math.isfinite(float(r.get("ratio_median", float("nan"))))]
    corr_vals = [float(r.get("abs_corr")) for r in usable_rows if math.isfinite(float(r.get("abs_corr", float("nan"))))]
    status_counts: Dict[str, int] = {}
    for row in rows:
        k = str(row.get("status", "unknown"))
        status_counts[k] = int(status_counts.get(k, 0)) + 1

    event_dependency_index = float(np.std(np.array(ratio_vals, dtype=float)) / np.mean(np.array(ratio_vals, dtype=float))) if len(ratio_vals) >= 2 and float(np.mean(np.array(ratio_vals, dtype=float))) != 0 else float("nan")
    overall_status = "pass" if len(usable_rows) >= 2 else ("watch" if len(usable_rows) == 1 else "reject")

    outdir = Path(str(args.outdir))
    public_outdir = Path(str(args.public_outdir))
    outdir.mkdir(parents=True, exist_ok=True)
    public_outdir.mkdir(parents=True, exist_ok=True)
    out_json = outdir / f"{args.prefix}.json"
    out_csv = outdir / f"{args.prefix}.csv"
    out_png = outdir / f"{args.prefix}.png"

    _write_csv(out_csv, rows)
    _plot(rows, out_png, corr_use_min=float(args.corr_use_min))

    payload: Dict[str, Any] = {
        "generated_utc": _iso_utc_now(),
        "schema": "wavep.gw.h1_l1.multi_event_audit.v1",
        "phase": 8,
        "step": "8.7.19.1",
        "inputs": {
            "events": events,
            "corr_use_min": float(args.corr_use_min),
        },
        "summary": {
            "n_events_requested": int(len(events)),
            "n_missing_metrics": int(len(missing_events)),
            "n_usable_events": int(len(usable_rows)),
            "status_counts": status_counts,
            "median_abs_corr_usable": float(np.median(np.array(corr_vals, dtype=float))) if corr_vals else float("nan"),
            "event_dependency_index_ratio_cv": event_dependency_index,
            "overall_status": overall_status,
            "overall_reason": "usable_events>=2" if len(usable_rows) >= 2 else "insufficient_usable_events",
        },
        "rows": rows,
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

    try:
        worklog.append_event(
            {
                "event_type": "gw_h1_l1_multi_event_amplitude_audit",
                "argv": list(sys.argv),
                "outputs": {
                    "audit_json": out_json,
                    "audit_csv": out_csv,
                    "audit_png": out_png,
                },
                "metrics": {
                    "overall_status": overall_status,
                    "n_usable_events": int(len(usable_rows)),
                    "n_missing_metrics": int(len(missing_events)),
                },
            }
        )
    except Exception:
        pass

    print(f"[ok] json: {out_json}")
    print(f"[ok] csv : {out_csv}")
    print(f"[ok] png : {out_png}")
    print(f"[ok] public copies: {len(copied)} files -> {public_outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
