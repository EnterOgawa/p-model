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


# 関数: `_safe_float` の入出力契約と処理意図を定義する。

def _safe_float(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return float("nan")

    return x


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

    # 条件分岐: `s.startswith("inconclusive")` を満たす経路を評価する。

    if s.startswith("inconclusive"):
        return "inconclusive"

    return "other"


# 関数: `_is_decisive` の入出力契約と処理意図を定義する。

def _is_decisive(status: str) -> bool:
    return _status_bucket(status) in {"reject", "watch", "pass"}


# 関数: `_load_payload` の入出力契約と処理意図を定義する。

def _load_payload(path: Path) -> Dict[str, Any]:
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        raise FileNotFoundError(f"missing audit json: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_index_rows` の入出力契約と処理意図を定義する。

def _index_rows(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        event = str(row.get("event") or "").strip()
        # 条件分岐: `not event` を満たす経路を評価する。
        if not event:
            continue

        out[event] = row

    return out


# 関数: `_count_summary` の入出力契約と処理意図を定義する。

def _count_summary(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    status_counts = {"reject": 0, "watch": 0, "pass": 0, "inconclusive": 0, "other": 0}
    for row in rows:
        bucket = _status_bucket(str(row.get("status") or ""))
        status_counts[bucket] = int(status_counts.get(bucket, 0)) + 1

    decisive = int(status_counts["reject"] + status_counts["watch"] + status_counts["pass"])
    return {
        "n_rows": int(len(rows)),
        "n_decisive": decisive,
        "n_inconclusive": int(status_counts["inconclusive"]),
        "n_reject": int(status_counts["reject"]),
        "n_watch": int(status_counts["watch"]),
        "n_pass": int(status_counts["pass"]),
        "n_other": int(status_counts["other"]),
    }


# 関数: `_transition_class` の入出力契約と処理意図を定義する。

def _transition_class(before_status: str, after_status: str) -> str:
    before_decisive = _is_decisive(before_status)
    after_decisive = _is_decisive(after_status)
    # 条件分岐: `not before_decisive and after_decisive` を満たす経路を評価する。
    if not before_decisive and after_decisive:
        return "inconclusive_to_decisive"

    # 条件分岐: `before_decisive and not after_decisive` を満たす経路を評価する。

    if before_decisive and not after_decisive:
        return "decisive_to_inconclusive"

    # 条件分岐: `before_status == after_status` を満たす経路を評価する。

    if before_status == after_status:
        return "status_unchanged"

    return "status_changed_same_decisiveness"


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    headers = [
        "event",
        "before_status",
        "after_status",
        "before_quality",
        "after_quality",
        "before_usable_pair_count",
        "after_usable_pair_count",
        "before_ring_directions_used",
        "after_ring_directions_used",
        "before_bucket",
        "after_bucket",
        "before_decisive",
        "after_decisive",
        "transition_class",
        "status_changed",
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
                else:
                    vals.append(v)

            w.writerow(vals)


# 関数: `_plot` の入出力契約と処理意図を定義する。

def _plot(
    *,
    baseline: Dict[str, int],
    refined: Dict[str, int],
    transition_rows: List[Dict[str, Any]],
    out_png: Path,
) -> None:
    _set_japanese_font()
    labels = ["usable(decisive)", "inconclusive", "reject", "watch", "pass"]
    b_vals = [
        float(baseline["n_decisive"]),
        float(baseline["n_inconclusive"]),
        float(baseline["n_reject"]),
        float(baseline["n_watch"]),
        float(baseline["n_pass"]),
    ]
    r_vals = [
        float(refined["n_decisive"]),
        float(refined["n_inconclusive"]),
        float(refined["n_reject"]),
        float(refined["n_watch"]),
        float(refined["n_pass"]),
    ]

    events = [str(r.get("event", "")) for r in transition_rows]
    ring_before = np.array([_safe_float(r.get("before_ring_directions_used")) for r in transition_rows], dtype=float)
    ring_after = np.array([_safe_float(r.get("after_ring_directions_used")) for r in transition_rows], dtype=float)

    x0 = np.arange(len(labels), dtype=float)
    x1 = np.arange(len(events), dtype=float)
    width = 0.36

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12.8, 8.4))
    ax0.bar(x0 - width / 2.0, b_vals, width=width, color="#4c78a8", alpha=0.88, label="baseline")
    ax0.bar(x0 + width / 2.0, r_vals, width=width, color="#f58518", alpha=0.88, label="refined")
    ax0.set_xticks(x0)
    ax0.set_xticklabels(labels, fontsize=9)
    ax0.set_ylabel("event count")
    ax0.grid(True, axis="y", alpha=0.25)
    ax0.legend(loc="upper right", fontsize=9)
    ax0.set_title("Network-decision count comparison (Step 8.7.19.5)")

    ax1.bar(x1 - width / 2.0, ring_before, width=width, color="#7aa6d9", alpha=0.9, label="baseline ring dirs")
    ax1.bar(x1 + width / 2.0, ring_after, width=width, color="#ffb066", alpha=0.9, label="refined ring dirs")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(events, fontsize=9)
    ax1.set_ylabel("ring directions used")
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_xlabel("event")

    fig.suptitle("GW polarization inconclusive-reduction audit", fontsize=14)
    fig.tight_layout(rect=[0.02, 0.02, 0.98, 0.95])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Step 8.7.19.5: compare baseline/refined network audits and quantify inconclusive reduction.")
    ap.add_argument(
        "--baseline-json",
        type=str,
        default=str(_ROOT / "output" / "public" / "gw" / "gw_polarization_h1_l1_v1_network_audit_corr005.json"),
    )
    ap.add_argument(
        "--refined-json",
        type=str,
        default=str(_ROOT / "output" / "public" / "gw" / "gw_polarization_h1_l1_v1_network_audit_corr005_whitenrefresh_sky50k.json"),
    )
    ap.add_argument(
        "--refined-label",
        type=str,
        default="corr=0.05 + whiten refresh + sky50k",
    )
    ap.add_argument("--outdir", type=str, default=str(_ROOT / "output" / "private" / "gw"))
    ap.add_argument("--public-outdir", type=str, default=str(_ROOT / "output" / "public" / "gw"))
    ap.add_argument("--prefix", type=str, default="gw_polarization_inconclusive_reduction_audit")
    args = ap.parse_args(list(argv) if argv is not None else None)

    baseline_path = Path(str(args.baseline_json))
    refined_path = Path(str(args.refined_json))
    baseline_payload = _load_payload(baseline_path)
    refined_payload = _load_payload(refined_path)

    baseline_rows = baseline_payload.get("rows") if isinstance(baseline_payload.get("rows"), list) else []
    refined_rows = refined_payload.get("rows") if isinstance(refined_payload.get("rows"), list) else []
    baseline_idx = _index_rows(baseline_rows)
    refined_idx = _index_rows(refined_rows)

    baseline_events = list(baseline_payload.get("inputs", {}).get("events") or [])
    refined_events = list(refined_payload.get("inputs", {}).get("events") or [])
    event_set_match = set(baseline_events) == set(refined_events)
    all_events = sorted(set(list(baseline_idx.keys()) + list(refined_idx.keys())))

    transition_rows: List[Dict[str, Any]] = []
    for event in all_events:
        before = baseline_idx.get(event, {})
        after = refined_idx.get(event, {})
        before_status = str(before.get("status") or "missing")
        after_status = str(after.get("status") or "missing")
        before_bucket = _status_bucket(before_status)
        after_bucket = _status_bucket(after_status)
        before_decisive = _is_decisive(before_status)
        after_decisive = _is_decisive(after_status)
        transition_rows.append(
            {
                "event": event,
                "before_status": before_status,
                "after_status": after_status,
                "before_quality": str(before.get("quality") or ""),
                "after_quality": str(after.get("quality") or ""),
                "before_usable_pair_count": int(_safe_float(before.get("usable_pair_count"))) if before else 0,
                "after_usable_pair_count": int(_safe_float(after.get("usable_pair_count"))) if after else 0,
                "before_ring_directions_used": int(_safe_float(before.get("ring_directions_used"))) if before else 0,
                "after_ring_directions_used": int(_safe_float(after.get("ring_directions_used"))) if after else 0,
                "before_bucket": before_bucket,
                "after_bucket": after_bucket,
                "before_decisive": int(before_decisive),
                "after_decisive": int(after_decisive),
                "transition_class": _transition_class(before_status, after_status),
                "status_changed": int(before_status != after_status),
            }
        )

    baseline_counts = _count_summary(baseline_rows)
    refined_counts = _count_summary(refined_rows)
    inconclusive_delta = int(refined_counts["n_inconclusive"] - baseline_counts["n_inconclusive"])
    decisive_delta = int(refined_counts["n_decisive"] - baseline_counts["n_decisive"])
    usable_before = int(_safe_float((baseline_payload.get("summary") or {}).get("n_usable_events")))
    usable_after = int(_safe_float((refined_payload.get("summary") or {}).get("n_usable_events")))
    usable_delta = int(usable_after - usable_before)

    transitioned_to_decisive = sorted(
        [str(r.get("event")) for r in transition_rows if str(r.get("transition_class")) == "inconclusive_to_decisive"]
    )
    transitioned_to_inconclusive = sorted(
        [str(r.get("event")) for r in transition_rows if str(r.get("transition_class")) == "decisive_to_inconclusive"]
    )

    # 条件分岐: `decisive_delta > 0 and inconclusive_delta < 0` を満たす経路を評価する。
    if decisive_delta > 0 and inconclusive_delta < 0:
        overall_status = "pass"
        decision = "inconclusive_reduced_and_decisive_events_increased"
    # 条件分岐: 前段条件が不成立で、`decisive_delta > 0 or inconclusive_delta < 0` を追加評価する。
    elif decisive_delta > 0 or inconclusive_delta < 0:
        overall_status = "watch"
        decision = "partial_improvement_only"
    else:
        overall_status = "watch"
        decision = "no_improvement_detected"

    outdir = Path(str(args.outdir))
    public_outdir = Path(str(args.public_outdir))
    outdir.mkdir(parents=True, exist_ok=True)
    public_outdir.mkdir(parents=True, exist_ok=True)
    out_json = outdir / f"{args.prefix}.json"
    out_csv = outdir / f"{args.prefix}.csv"
    out_png = outdir / f"{args.prefix}.png"

    transition_rows = sorted(transition_rows, key=lambda r: str(r.get("event", "")))
    _write_csv(out_csv, transition_rows)
    _plot(baseline=baseline_counts, refined=refined_counts, transition_rows=transition_rows, out_png=out_png)

    payload: Dict[str, Any] = {
        "generated_utc": _iso_utc_now(),
        "schema": "wavep.gw.polarization.inconclusive_reduction_audit.v1",
        "phase": 8,
        "step": "8.7.19.5",
        "inputs": {
            "baseline_json": str(baseline_path).replace("\\", "/"),
            "refined_json": str(refined_path).replace("\\", "/"),
            "refined_label": str(args.refined_label),
            "event_set_match": bool(event_set_match),
            "baseline_events": [str(x) for x in baseline_events],
            "refined_events": [str(x) for x in refined_events],
        },
        "summary": {
            "overall_status": overall_status,
            "decision": decision,
            "n_events_compared": int(len(all_events)),
            "baseline_n_usable_events": usable_before,
            "refined_n_usable_events": usable_after,
            "usable_delta": usable_delta,
            "baseline_n_inconclusive": int(baseline_counts["n_inconclusive"]),
            "refined_n_inconclusive": int(refined_counts["n_inconclusive"]),
            "inconclusive_delta": inconclusive_delta,
            "baseline_n_decisive": int(baseline_counts["n_decisive"]),
            "refined_n_decisive": int(refined_counts["n_decisive"]),
            "decisive_delta": decisive_delta,
            "transitioned_to_decisive": transitioned_to_decisive,
            "transitioned_to_inconclusive": transitioned_to_inconclusive,
        },
        "counts": {
            "baseline": baseline_counts,
            "refined": refined_counts,
        },
        "rows": transition_rows,
        "notes": {
            "preprocess_output_collision_caution": "detector-pair amplitude output names do not encode preprocess mode; refresh runs can overwrite pair metrics when the same event+pair is reused."
        },
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
                "event_type": "gw_polarization_inconclusive_reduction_audit",
                "argv": list(sys.argv),
                "outputs": {"audit_json": out_json, "audit_csv": out_csv, "audit_png": out_png},
                "metrics": payload.get("summary", {}),
            }
        )
    except Exception:
        pass

    print(f"[ok] json: {out_json}")
    print(f"[ok] csv : {out_csv}")
    print(f"[ok] png : {out_png}")
    print(f"[ok] public copies: {len(copied)} files -> {public_outdir}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
