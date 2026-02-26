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
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.gw import gw_polarization_h1_l1_v1_network_audit as network_audit  # noqa: E402
from scripts.summary import worklog  # noqa: E402


# 関数: `_iso_utc_now` の入出力契約と処理意図を定義する。
def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# 関数: `_set_japanese_font` の入出力契約と処理意図を定義する。

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
        # 条件分岐: `not chosen` を満たす経路を評価する。
        if not chosen:
            return

        mpl.rcParams["font.family"] = chosen + ["DejaVu Sans"]
        mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        return


# 関数: `_fmt` の入出力契約と処理意図を定義する。

def _fmt(v: float, digits: int = 7) -> str:
    # 条件分岐: `not math.isfinite(float(v))` を満たす経路を評価する。
    if not math.isfinite(float(v)):
        return ""

    x = float(v)
    # 条件分岐: `x == 0.0` を満たす経路を評価する。
    if x == 0.0:
        return "0"

    ax = abs(x)
    # 条件分岐: `ax >= 1e4 or ax < 1e-3` を満たす経路を評価する。
    if ax >= 1e4 or ax < 1e-3:
        return f"{x:.{digits}g}"

    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


# 関数: `_parse_corr_grid` の入出力契約と処理意図を定義する。

def _parse_corr_grid(spec: str) -> List[float]:
    vals: List[float] = []
    for token in str(spec).split(","):
        t = token.strip()
        # 条件分岐: `not t` を満たす経路を評価する。
        if not t:
            continue

        try:
            v = float(t)
        except Exception:
            continue

        # 条件分岐: `not math.isfinite(v) or v < 0.0` を満たす経路を評価する。

        if not math.isfinite(v) or v < 0.0:
            continue

        vals.append(v)

    uniq = sorted(set(round(v, 6) for v in vals))
    return [float(v) for v in uniq]


# 関数: `_corr_tag` の入出力契約と処理意図を定義する。

def _corr_tag(corr: float) -> str:
    return f"{int(round(float(corr) * 1000.0)):04d}"


# 関数: `_status_bucket` の入出力契約と処理意図を定義する。

def _status_bucket(status: str) -> str:
    s = str(status or "")
    # 条件分岐: `s.startswith("reject")` を満たす経路を評価する。
    if s.startswith("reject"):
        return "reject"

    # 条件分岐: `s.startswith("watch")` を満たす経路を評価する。

    if s.startswith("watch"):
        return "watch"

    # 条件分岐: `s.startswith("pass")` を満たす経路を評価する。

    if s.startswith("pass"):
        return "pass"

    return "other"


# 関数: `_event_name_list` の入出力契約と処理意図を定義する。

def _event_name_list(rows: List[Dict[str, Any]], predicate) -> List[str]:
    out: List[str] = []
    for row in rows:
        # 条件分岐: `predicate(row)` を満たす経路を評価する。
        if predicate(row):
            out.append(str(row.get("event", "")))

    return sorted([v for v in out if v])


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    headers = [
        "corr_use_min",
        "run_prefix",
        "n_rows",
        "n_usable_events",
        "overall_status",
        "overall_reason",
        "usable_events",
        "reject_events",
        "watch_events",
        "pass_events",
        "inconclusive_events",
        "network_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for row in rows:
            vals: List[Any] = []
            for h in headers:
                v = row.get(h, "")
                # 条件分岐: `isinstance(v, float)` を満たす経路を評価する。
                if isinstance(v, float):
                    vals.append(_fmt(v))
                # 条件分岐: 前段条件が不成立で、`isinstance(v, list)` を追加評価する。
                elif isinstance(v, list):
                    vals.append(";".join(str(x) for x in v))
                else:
                    vals.append(v)

            w.writerow(vals)


# 関数: `_plot` の入出力契約と処理意図を定義する。

def _plot(rows: List[Dict[str, Any]], out_png: Path, locked_corr: float) -> None:
    _set_japanese_font()
    corr = np.asarray([float(r.get("corr_use_min", float("nan"))) for r in rows], dtype=float)
    n_use = np.asarray([float(r.get("n_usable_events", 0.0)) for r in rows], dtype=float)
    n_reject = np.asarray([float(len(r.get("reject_events") or [])) for r in rows], dtype=float)
    n_watch = np.asarray([float(len(r.get("watch_events") or [])) for r in rows], dtype=float)
    n_pass = np.asarray([float(len(r.get("pass_events") or [])) for r in rows], dtype=float)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12.6, 8.2), sharex=True)
    ax0.plot(corr, n_use, marker="o", color="#1f77b4", linewidth=1.8, label="usable events")
    ax0.set_ylabel("count")
    ax0.set_title("Three-detector usable-event count vs corr gate")
    ax0.grid(True, axis="both", alpha=0.25)
    # 条件分岐: `math.isfinite(float(locked_corr))` を満たす経路を評価する。
    if math.isfinite(float(locked_corr)):
        ax0.axvline(float(locked_corr), color="#444444", linestyle="--", linewidth=1.0, label=f"locked corr={locked_corr:.3f}")

    ax0.legend(loc="upper right", fontsize=9)

    width = 0.02
    ax1.bar(corr, n_reject, width=width, color="#d62728", alpha=0.85, label="reject")
    ax1.bar(corr, n_watch, width=width, bottom=n_reject, color="#f2c744", alpha=0.85, label="watch")
    ax1.bar(corr, n_pass, width=width, bottom=n_reject + n_watch, color="#2ca02c", alpha=0.85, label="pass")
    ax1.set_ylabel("usable status counts")
    ax1.set_xlabel("corr_use_min")
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend(loc="upper right", fontsize=9)
    # 条件分岐: `math.isfinite(float(locked_corr))` を満たす経路を評価する。
    if math.isfinite(float(locked_corr)):
        ax1.axvline(float(locked_corr), color="#444444", linestyle="--", linewidth=1.0)

    fig.suptitle("GW polarization network corr-gate staging audit (Step 8.7.19.4)", fontsize=14)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Step 8.7.19.4: stage corr_use_min gate for 3-detector GW polarization audit.")
    ap.add_argument("--events", type=str, default="GW200115_042309,GW200129_065458,GW200311_115853")
    ap.add_argument("--detectors", type=str, default="H1,L1,V1")
    ap.add_argument("--corr-grid", type=str, default="0.05,0.08,0.10,0.12,0.15,0.20,0.25,0.30")
    ap.add_argument("--min-usable-events", type=int, default=1)
    ap.add_argument("--sky-samples", type=int, default=5000)
    ap.add_argument("--psi-samples", type=int, default=36)
    ap.add_argument("--cosi-samples", type=int, default=41)
    ap.add_argument("--response-floor-frac", type=float, default=0.1)
    ap.add_argument("--outdir", type=str, default=str(_ROOT / "output" / "private" / "gw"))
    ap.add_argument("--public-outdir", type=str, default=str(_ROOT / "output" / "public" / "gw"))
    ap.add_argument("--prefix", type=str, default="gw_polarization_corr_gate_stage_audit")
    args = ap.parse_args(list(argv) if argv is not None else None)

    events = [s.strip() for s in str(args.events).split(",") if s.strip()]
    corr_grid = _parse_corr_grid(str(args.corr_grid))
    # 条件分岐: `not events` を満たす経路を評価する。
    if not events:
        print("[err] --events is empty")
        return 2

    # 条件分岐: `not corr_grid` を満たす経路を評価する。

    if not corr_grid:
        print("[err] --corr-grid is empty")
        return 2

    outdir = Path(str(args.outdir))
    public_outdir = Path(str(args.public_outdir))
    run_outdir = outdir / "gw_polarization_corr_gate_stage_runs"
    run_public_cache = outdir / "gw_polarization_corr_gate_stage_runs_public_cache"
    outdir.mkdir(parents=True, exist_ok=True)
    public_outdir.mkdir(parents=True, exist_ok=True)
    run_outdir.mkdir(parents=True, exist_ok=True)
    run_public_cache.mkdir(parents=True, exist_ok=True)

    stage_rows: List[Dict[str, Any]] = []
    for corr in corr_grid:
        tag = _corr_tag(corr)
        run_prefix = f"{args.prefix}_corr{tag}"
        rc = network_audit.main(
            [
                "--events",
                ",".join(events),
                "--detectors",
                str(args.detectors),
                "--corr-use-min",
                str(corr),
                "--sky-samples",
                str(int(args.sky_samples)),
                "--psi-samples",
                str(int(args.psi_samples)),
                "--cosi-samples",
                str(int(args.cosi_samples)),
                "--response-floor-frac",
                str(float(args.response_floor_frac)),
                "--outdir",
                str(run_outdir),
                "--public-outdir",
                str(run_public_cache),
                "--prefix",
                run_prefix,
            ]
        )
        run_json = run_outdir / f"{run_prefix}.json"
        row_base: Dict[str, Any] = {
            "corr_use_min": float(corr),
            "run_prefix": run_prefix,
            "network_json": str(run_json).replace("\\", "/"),
        }
        # 条件分岐: `int(rc) != 0 or not run_json.exists()` を満たす経路を評価する。
        if int(rc) != 0 or not run_json.exists():
            row_base.update(
                {
                    "n_rows": 0,
                    "n_usable_events": 0,
                    "overall_status": "run_error",
                    "overall_reason": f"network_audit_rc={rc}",
                    "usable_events": [],
                    "reject_events": [],
                    "watch_events": [],
                    "pass_events": [],
                    "inconclusive_events": events,
                }
            )
            stage_rows.append(row_base)
            continue

        payload = json.loads(run_json.read_text(encoding="utf-8"))
        rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []
        usable = _event_name_list(
            rows,
            lambda r: str(r.get("quality")) == "usable" and _status_bucket(str(r.get("status"))) in {"reject", "watch", "pass"},
        )
        reject_events = _event_name_list(rows, lambda r: _status_bucket(str(r.get("status"))) == "reject")
        watch_events = _event_name_list(rows, lambda r: _status_bucket(str(r.get("status"))) == "watch")
        pass_events = _event_name_list(rows, lambda r: _status_bucket(str(r.get("status"))) == "pass")
        inconclusive_events = _event_name_list(rows, lambda r: str(r.get("status", "")).startswith("inconclusive"))
        summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
        row_base.update(
            {
                "n_rows": int(len(rows)),
                "n_usable_events": int(len(usable)),
                "overall_status": str(summary.get("overall_status") or "unknown"),
                "overall_reason": str(summary.get("overall_reason") or ""),
                "usable_events": usable,
                "reject_events": reject_events,
                "watch_events": watch_events,
                "pass_events": pass_events,
                "inconclusive_events": inconclusive_events,
            }
        )
        stage_rows.append(row_base)

    stage_rows = sorted(stage_rows, key=lambda r: float(r.get("corr_use_min", 0.0)))
    min_usable_events = int(max(1, int(args.min_usable_events)))
    candidates = [r for r in stage_rows if int(r.get("n_usable_events", 0)) >= min_usable_events]
    # 条件分岐: `not candidates` を満たす経路を評価する。
    if not candidates:
        locked_corr = float("nan")
        locked_events: List[str] = []
        decision = "inconclusive_no_corr_gate_candidate"
        decision_reason = "no_threshold_keeps_minimum_usable_events"
        stable_signature = False
    else:
        locked = max(candidates, key=lambda r: float(r.get("corr_use_min", 0.0)))
        locked_corr = float(locked.get("corr_use_min", float("nan")))
        locked_events = list(locked.get("usable_events") or [])
        stable_signature = True
        for row in candidates:
            # 条件分岐: `set(row.get("usable_events") or []) != set(locked_events)` を満たす経路を評価する。
            if set(row.get("usable_events") or []) != set(locked_events):
                stable_signature = False
                break

        decision = "lock_corr_gate"
        decision_reason = (
            "strictest_threshold_with_minimum_usable_events_and_stable_selection"
            if stable_signature
            else "strictest_threshold_with_minimum_usable_events_selection_changes_across_grid"
        )

    out_json = outdir / f"{args.prefix}.json"
    out_csv = outdir / f"{args.prefix}.csv"
    out_png = outdir / f"{args.prefix}.png"
    _write_csv(out_csv, stage_rows)
    _plot(stage_rows, out_png, locked_corr=locked_corr)

    payload: Dict[str, Any] = {
        "generated_utc": _iso_utc_now(),
        "schema": "wavep.gw.polarization.corr_gate_stage_audit.v1",
        "phase": 8,
        "step": "8.7.19.4",
        "inputs": {
            "events": events,
            "detectors": [s.strip().upper() for s in str(args.detectors).split(",") if s.strip()],
            "corr_grid": corr_grid,
            "min_usable_events": min_usable_events,
            "sky_samples": int(args.sky_samples),
            "psi_samples": int(args.psi_samples),
            "cosi_samples": int(args.cosi_samples),
            "response_floor_frac": float(args.response_floor_frac),
        },
        "summary": {
            "n_thresholds": int(len(stage_rows)),
            "n_candidate_thresholds": int(len(candidates)),
            "decision": decision,
            "decision_reason": decision_reason,
            "locked_corr_use_min": locked_corr,
            "locked_events": locked_events,
            "selection_signature_stable_across_candidates": bool(stable_signature),
            "notes": [
                "corr_use_min is staged to separate usable-vs-inconclusive network events.",
                "Locked threshold is the strictest setting that keeps minimum usable events.",
            ],
        },
        "rows": stage_rows,
        "outputs": {
            "audit_json": str(out_json).replace("\\", "/"),
            "audit_csv": str(out_csv).replace("\\", "/"),
            "audit_png": str(out_png).replace("\\", "/"),
            "network_stage_runs_dir": str(run_outdir).replace("\\", "/"),
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
                "event_type": "gw_polarization_corr_gate_stage_audit",
                "argv": list(sys.argv),
                "outputs": {
                    "audit_json": out_json,
                    "audit_csv": out_csv,
                    "audit_png": out_png,
                },
                "metrics": {
                    "decision": decision,
                    "locked_corr_use_min": locked_corr,
                    "locked_events_count": int(len(locked_events)),
                    "n_thresholds": int(len(stage_rows)),
                },
            }
        )
    except Exception:
        pass

    print(f"[ok] json: {out_json}")
    print(f"[ok] csv : {out_csv}")
    print(f"[ok] png : {out_png}")
    # 条件分岐: `math.isfinite(float(locked_corr))` を満たす経路を評価する。
    if math.isfinite(float(locked_corr)):
        print(f"[ok] locked corr_use_min: {locked_corr:.3f} | events={','.join(locked_events) if locked_events else '-'}")
    else:
        print("[warn] no locked corr_use_min candidate")

    print(f"[ok] public copies: {len(copied)} files -> {public_outdir}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
