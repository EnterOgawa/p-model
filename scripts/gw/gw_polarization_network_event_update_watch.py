from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

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


# 関数: `_safe_float` の入出力契約と処理意図を定義する。

def _safe_float(v: Any) -> float:
    try:
        x = float(v)
    except Exception:
        return float("nan")

    return x


# 関数: `_fmt` の入出力契約と処理意図を定義する。

def _fmt(v: Any, digits: int = 6) -> str:
    x = _safe_float(v)
    # 条件分岐: `not math.isfinite(x)` を満たす経路を評価する。
    if not math.isfinite(x):
        return ""

    # 条件分岐: `x == 0.0` を満たす経路を評価する。

    if x == 0.0:
        return "0"

    ax = abs(x)
    # 条件分岐: `ax >= 1e4 or ax < 1e-3` を満たす経路を評価する。
    if ax >= 1e4 or ax < 1e-3:
        return f"{x:.{digits}g}"

    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


# 関数: `_read_json` の入出力契約と処理意図を定義する。

def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_write_json` の入出力契約と処理意図を定義する。

def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # 条件分岐: `not rows` を満たす経路を評価する。
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


# 関数: `_parse_detectors` の入出力契約と処理意図を定義する。

def _parse_detectors(raw: Any) -> Set[str]:
    text = str(raw or "").strip()
    # 条件分岐: `not text` を満たす経路を評価する。
    if not text:
        return set()

    for sep in [";", "/", "|", " "]:
        text = text.replace(sep, ",")

    out = {token.strip().upper() for token in text.split(",") if token.strip()}
    return out


# 関数: `_inventory_for_slug` の入出力契約と処理意図を定義する。

def _inventory_for_slug(slug: str, max_files: int = 128) -> Dict[str, Any]:
    folder = _ROOT / "data" / "gw" / str(slug)
    # 条件分岐: `not folder.exists() or not folder.is_dir()` を満たす経路を評価する。
    if not folder.exists() or not folder.is_dir():
        return {"exists": False, "n_files": 0, "files": []}

    files: List[Dict[str, Any]] = []
    for idx, fp in enumerate(sorted(folder.rglob("*"))):
        # 条件分岐: `not fp.is_file()` を満たす経路を評価する。
        if not fp.is_file():
            continue

        # 条件分岐: `len(files) >= max_files` を満たす経路を評価する。

        if len(files) >= max_files:
            break

        files.append(
            {
                "relpath": str(fp.relative_to(folder)).replace("\\", "/"),
                "size": int(fp.stat().st_size),
            }
        )
        # 条件分岐: `idx >= max_files * 2` を満たす経路を評価する。
        if idx >= max_files * 2:
            break

    total_files = int(sum(1 for fp in folder.rglob("*") if fp.is_file()))
    return {
        "exists": True,
        "n_files": total_files,
        "files": files,
        "truncated": bool(total_files > len(files)),
    }


# 関数: `_load_three_detector_events` の入出力契約と処理意図を定義する。

def _load_three_detector_events(event_list_path: Path) -> List[Dict[str, Any]]:
    # 条件分岐: `not event_list_path.exists()` を満たす経路を評価する。
    if not event_list_path.exists():
        return []

    root = _read_json(event_list_path)
    rows = root.get("events")
    # 条件分岐: `not isinstance(rows, list)` を満たす経路を評価する。
    if not isinstance(rows, list):
        return []

    out: List[Dict[str, Any]] = []
    required = {"H1", "L1", "V1"}
    for row in rows:
        # 条件分岐: `not isinstance(row, dict)` を満たす経路を評価する。
        if not isinstance(row, dict):
            continue

        name = str(row.get("name") or "").strip()
        slug = str(row.get("slug") or "").strip()
        # 条件分岐: `not (name and slug)` を満たす経路を評価する。
        if not (name and slug):
            continue

        detectors = _parse_detectors(row.get("detectors"))
        # 条件分岐: `not required.issubset(detectors)` を満たす経路を評価する。
        if not required.issubset(detectors):
            continue

        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        event = {
            "name": name,
            "slug": slug,
            "type": str(row.get("type") or "").strip(),
            "profile": str(row.get("profile") or "").strip(),
            "catalog": str(row.get("catalog") or "").strip(),
            "detectors": sorted(detectors),
            "network_snr": _safe_float(meta.get("network_snr")),
            "p_astro": _safe_float(meta.get("p_astro")),
            "far_yr": _safe_float(meta.get("far_yr")),
            "inventory": _inventory_for_slug(slug),
        }
        out.append(event)

    out.sort(key=lambda r: (-_safe_float(r.get("network_snr")), str(r.get("name", ""))))
    return out


# 関数: `_input_signature` の入出力契約と処理意図を定義する。

def _input_signature(events: List[Dict[str, Any]], event_list_path: Path) -> Dict[str, Any]:
    canonical_events = [
        {
            "name": str(e.get("name", "")),
            "slug": str(e.get("slug", "")),
            "type": str(e.get("type", "")),
            "profile": str(e.get("profile", "")),
            "catalog": str(e.get("catalog", "")),
            "detectors": list(e.get("detectors") or []),
            "network_snr": _safe_float(e.get("network_snr")),
            "p_astro": _safe_float(e.get("p_astro")),
            "far_yr": _safe_float(e.get("far_yr")),
            "inventory": e.get("inventory") if isinstance(e.get("inventory"), dict) else {},
        }
        for e in events
    ]
    canonical_obj = {
        "event_list_path": str(event_list_path).replace("\\", "/"),
        "events": canonical_events,
    }
    canonical_text = json.dumps(canonical_obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()
    return {
        "digest_sha256": digest,
        "canonical_event_count": int(len(canonical_events)),
        "canonical_payload": canonical_obj,
    }


# 関数: `_load_state` の入出力契約と処理意図を定義する。

def _load_state(path: Path) -> Dict[str, Any]:
    # 条件分岐: `not path.exists()` を満たす経路を評価する。
    if not path.exists():
        return {}

    try:
        obj = _read_json(path)
    except Exception:
        return {}

    return obj if isinstance(obj, dict) else {}


# 関数: `_extract_coverage_metrics` の入出力契約と処理意図を定義する。

def _extract_coverage_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    best_u2_all = summary.get("best_u2_all") if isinstance(summary.get("best_u2_all"), dict) else {}
    return {
        "overall_status": str(summary.get("overall_status", "")),
        "decision": str(summary.get("decision", "")),
        "reason": str(summary.get("reason", "")),
        "n_subsets_tested": int(summary.get("n_subsets_tested", 0)),
        "n_gate_hits": int(summary.get("n_gate_hits", 0)),
        "n_subsets_with_usable_min": int(summary.get("n_subsets_with_usable_min", 0)),
        "best_u2_scalar_overlap_proxy": _safe_float(best_u2_all.get("best_u2_scalar_overlap_proxy")),
        "best_u2_n_usable_events": int(best_u2_all.get("best_u2_n_usable_events", 0)),
        "best_u2_subset_events": str(best_u2_all.get("subset_events", "")),
    }


# 関数: `_run_coverage_expansion` の入出力契約と処理意図を定義する。

def _run_coverage_expansion(
    *,
    events: List[Dict[str, Any]],
    anchor_event: str,
    outdir: Path,
    public_outdir: Path,
    coverage_prefix: str,
) -> Dict[str, Any]:
    coverage_script = _ROOT / "scripts" / "gw" / "gw_polarization_network_coverage_expansion_audit.py"
    # 条件分岐: `not coverage_script.exists()` を満たす経路を評価する。
    if not coverage_script.exists():
        return {
            "ok": False,
            "error": "missing_coverage_script",
            "returncode": 2,
            "stdout_tail": "",
            "stderr_tail": str(coverage_script),
        }

    event_names = [str(e.get("name", "")).strip() for e in events if str(e.get("name", "")).strip()]
    # 条件分岐: `len(event_names) < 2` を満たす経路を評価する。
    if len(event_names) < 2:
        return {
            "ok": False,
            "error": "insufficient_three_detector_events",
            "returncode": 2,
            "stdout_tail": "",
            "stderr_tail": f"n_three_detector_events={len(event_names)}",
        }

    # 条件分岐: `anchor_event not in event_names` を満たす経路を評価する。

    if anchor_event not in event_names:
        anchor = event_names[0]
    else:
        anchor = anchor_event

    cmd = [
        sys.executable,
        "-B",
        str(coverage_script),
        "--events",
        ",".join(event_names),
        "--anchor-event",
        str(anchor),
        "--prefix",
        str(coverage_prefix),
        "--outdir",
        str(outdir),
        "--public-outdir",
        str(public_outdir),
    ]
    proc = subprocess.run(cmd, cwd=str(_ROOT), capture_output=True, text=True, check=False)
    coverage_json = outdir / f"{coverage_prefix}.json"
    # 条件分岐: `proc.returncode != 0 or not coverage_json.exists()` を満たす経路を評価する。
    if proc.returncode != 0 or not coverage_json.exists():
        return {
            "ok": False,
            "error": "coverage_audit_failed",
            "returncode": int(proc.returncode),
            "stdout_tail": str(proc.stdout or "")[-2000:],
            "stderr_tail": str(proc.stderr or "")[-2000:],
        }

    payload = _read_json(coverage_json)
    metrics = _extract_coverage_metrics(payload)
    return {
        "ok": True,
        "returncode": int(proc.returncode),
        "anchor_event_used": anchor,
        "event_names": event_names,
        "coverage_json": str(coverage_json).replace("\\", "/"),
        "coverage_metrics": metrics,
        "stdout_tail": str(proc.stdout or "")[-2000:],
        "stderr_tail": str(proc.stderr or "")[-2000:],
    }


# 関数: `_plot_summary` の入出力契約と処理意図を定義する。

def _plot_summary(row: Dict[str, Any], out_png: Path) -> None:
    labels0 = ["3-det events", "hash changed", "rerun"]
    vals0 = np.asarray(
        [
            float(row.get("n_three_detector_events", 0)),
            float(row.get("update_event_detected", 0)),
            float(row.get("rerun_triggered", 0)),
        ],
        dtype=float,
    )
    labels1 = ["gate hits (prev)", "gate hits (curr)", "best_u2 (prev)", "best_u2 (curr)"]
    vals1 = np.asarray(
        [
            float(row.get("prev_n_gate_hits", 0)),
            float(row.get("curr_n_gate_hits", 0)),
            _safe_float(row.get("prev_best_u2_scalar_overlap_proxy")),
            _safe_float(row.get("curr_best_u2_scalar_overlap_proxy")),
        ],
        dtype=float,
    )

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [1, 1]})
    x0 = np.arange(len(labels0), dtype=float)
    ax0.bar(x0, vals0, color=["#4c78a8", "#f58518", "#54a24b"], alpha=0.9)
    ax0.set_xticks(x0)
    ax0.set_xticklabels(labels0)
    ax0.set_ylabel("count / flag")
    ax0.set_title("GW polarization event-update watch (3-detector)")
    ax0.grid(True, axis="y", alpha=0.25)

    x1 = np.arange(len(labels1), dtype=float)
    colors = ["#9aa0a6", "#4c78a8", "#9aa0a6", "#f2c744"]
    for idx, val in enumerate(vals1):
        # 条件分岐: `idx >= 2 and not math.isfinite(float(val))` を満たす経路を評価する。
        if idx >= 2 and not math.isfinite(float(val)):
            colors[idx] = "#c7c7c7"

    ax1.bar(x1, vals1, color=colors, alpha=0.9)
    ax1.axhline(0.5, color="#f2c744", linestyle=":", linewidth=1.2, label="watch boundary (best_u2)")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(labels1, rotation=10)
    ax1.set_ylabel("value")
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Step 8.7.19.10: monitor three-detector public-event input hash and "
            "rerun coverage expansion audit only when input hash changes."
        )
    )
    ap.add_argument("--event-list", type=str, default=str(_ROOT / "data" / "gw" / "event_list.json"))
    ap.add_argument("--anchor-event", type=str, default="GW200129_065458")
    ap.add_argument("--force-rerun", action="store_true")
    ap.add_argument("--coverage-prefix", type=str, default="gw_polarization_network_coverage_expansion_audit")
    ap.add_argument(
        "--state-path",
        type=str,
        default=str(_ROOT / "output" / "private" / "gw" / "gw_polarization_network_event_update_watch_state.json"),
    )
    ap.add_argument("--outdir", type=str, default=str(_ROOT / "output" / "private" / "gw"))
    ap.add_argument("--public-outdir", type=str, default=str(_ROOT / "output" / "public" / "gw"))
    ap.add_argument("--prefix", type=str, default="gw_polarization_network_event_update_watch")
    ap.add_argument("--step-tag", type=str, default="8.7.19.10")
    args = ap.parse_args(list(argv) if argv is not None else None)

    outdir = Path(args.outdir)
    public_outdir = Path(args.public_outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    public_outdir.mkdir(parents=True, exist_ok=True)

    event_list_path = Path(args.event_list)
    events = _load_three_detector_events(event_list_path)
    signature = _input_signature(events, event_list_path)
    digest = str(signature.get("digest_sha256", ""))

    state_path = Path(args.state_path)
    prev_state = _load_state(state_path)
    prev_digest = str(prev_state.get("last_digest_sha256", ""))
    prev_counter = int(prev_state.get("event_counter", 0))
    prev_metrics = prev_state.get("last_coverage_metrics") if isinstance(prev_state.get("last_coverage_metrics"), dict) else {}

    # 条件分岐: `bool(args.force_rerun)` を満たす経路を評価する。
    if bool(args.force_rerun):
        update_event_type = "forced_rerun"
        update_event_detected = bool(digest != prev_digest)
    # 条件分岐: 前段条件が不成立で、`not prev_digest` を追加評価する。
    elif not prev_digest:
        update_event_type = "initial_lock"
        update_event_detected = True
    # 条件分岐: 前段条件が不成立で、`digest != prev_digest` を追加評価する。
    elif digest != prev_digest:
        update_event_type = "hash_changed"
        update_event_detected = True
    else:
        update_event_type = "no_change"
        update_event_detected = False

    event_counter = prev_counter + (1 if update_event_detected else 0)
    rerun_triggered = bool(update_event_detected or bool(args.force_rerun))

    coverage_run: Dict[str, Any] = {}
    curr_metrics: Dict[str, Any] = {}
    # 条件分岐: `rerun_triggered` を満たす経路を評価する。
    if rerun_triggered:
        coverage_run = _run_coverage_expansion(
            events=events,
            anchor_event=str(args.anchor_event),
            outdir=outdir,
            public_outdir=public_outdir,
            coverage_prefix=str(args.coverage_prefix),
        )
        # 条件分岐: `bool(coverage_run.get("ok"))` を満たす経路を評価する。
        if bool(coverage_run.get("ok")):
            curr_metrics = coverage_run.get("coverage_metrics") if isinstance(coverage_run.get("coverage_metrics"), dict) else {}
    else:
        existing_coverage = outdir / f"{args.coverage_prefix}.json"
        # 条件分岐: `existing_coverage.exists()` を満たす経路を評価する。
        if existing_coverage.exists():
            curr_metrics = _extract_coverage_metrics(_read_json(existing_coverage))

    # 条件分岐: `not curr_metrics and prev_metrics` を満たす経路を評価する。

    if not curr_metrics and prev_metrics:
        curr_metrics = dict(prev_metrics)

    has_prev_metrics = bool(prev_metrics)
    prev_gate_hits = int(prev_metrics.get("n_gate_hits", 0))
    curr_gate_hits = int(curr_metrics.get("n_gate_hits", 0))
    prev_best_u2 = _safe_float(prev_metrics.get("best_u2_scalar_overlap_proxy"))
    curr_best_u2 = _safe_float(curr_metrics.get("best_u2_scalar_overlap_proxy"))

    improved_gate_hits = bool(has_prev_metrics and (curr_gate_hits > prev_gate_hits))
    improved_best_u2 = bool(
        has_prev_metrics
        and
        math.isfinite(curr_best_u2)
        and (
            (not math.isfinite(prev_best_u2))
            or (curr_best_u2 < prev_best_u2 - 1e-12)
        )
    )
    improvement_detected = bool(improved_gate_hits or improved_best_u2)

    coverage_ok = bool(not rerun_triggered or coverage_run.get("ok"))
    coverage_status = str(curr_metrics.get("overall_status", "watch"))
    # 条件分岐: `not coverage_status` を満たす経路を評価する。
    if not coverage_status:
        coverage_status = "watch"

    # 条件分岐: `rerun_triggered and not coverage_ok` を満たす経路を評価する。

    if rerun_triggered and not coverage_ok:
        overall_status = "watch"
        decision = "coverage_rerun_failed"
        next_action = "fix_coverage_rerun_failure_then_repeat_step_8_7_19_10"
    # 条件分岐: 前段条件が不成立で、`update_event_type == "no_change"` を追加評価する。
    elif update_event_type == "no_change":
        overall_status = coverage_status
        decision = "no_change_hold"
        next_action = "wait_for_three_detector_input_hash_change_then_rerun_step_8_7_19_10"
    # 条件分岐: 前段条件が不成立で、`update_event_type == "initial_lock"` を追加評価する。
    elif update_event_type == "initial_lock":
        overall_status = coverage_status
        decision = "initial_lock_rerun"
        next_action = "wait_for_three_detector_input_hash_change_then_rerun_step_8_7_19_10"
    # 条件分岐: 前段条件が不成立で、`improvement_detected` を追加評価する。
    elif improvement_detected:
        overall_status = coverage_status
        decision = "hash_update_rerun_improved"
        next_action = "lock_updated_coverage_metrics_and_continue_event_monitoring"
    else:
        overall_status = "watch" if coverage_status != "pass" else "pass"
        decision = "hash_update_rerun_no_improvement"
        next_action = "keep_watch_and_wait_for_more_three_detector_public_events"

    row = {
        "generated_utc": _iso_utc_now(),
        "update_event_type": update_event_type,
        "update_event_detected": int(update_event_detected),
        "rerun_triggered": int(rerun_triggered),
        "rerun_ok": int(bool(coverage_ok)),
        "n_three_detector_events": int(len(events)),
        "prev_n_gate_hits": int(prev_gate_hits),
        "curr_n_gate_hits": int(curr_gate_hits),
        "prev_best_u2_scalar_overlap_proxy": float(prev_best_u2) if math.isfinite(prev_best_u2) else float("nan"),
        "curr_best_u2_scalar_overlap_proxy": float(curr_best_u2) if math.isfinite(curr_best_u2) else float("nan"),
        "improved_gate_hits": int(improved_gate_hits),
        "improved_best_u2": int(improved_best_u2),
        "improvement_detected": int(improvement_detected),
        "overall_status": overall_status,
        "decision": decision,
        "event_counter": int(event_counter),
        "next_action": next_action,
        "input_hash": digest,
    }

    watch_boundary = 0.5
    usable_subset_count = int(curr_metrics.get("n_subsets_with_usable_min", 0))
    best_u2_watch_value = float(curr_best_u2) if math.isfinite(curr_best_u2) else float("nan")
    needed_gate_hits = max(0, 1 - curr_gate_hits)
    needed_best_u2_drop = (
        max(0.0, best_u2_watch_value - (watch_boundary - 1e-12))
        if math.isfinite(best_u2_watch_value)
        else float("nan")
    )
    checklist_rows: List[Dict[str, Any]] = [
        {
            "requirement_id": "polarization::usable_subsets_with_min_coverage",
            "metric": "n_subsets_with_usable_min",
            "value": usable_subset_count,
            "expected": ">=1",
            "status": "pass" if usable_subset_count >= 1 else "watch",
            "required_action": (
                "none"
                if usable_subset_count >= 1
                else "wait_for_more_three_detector_public_events_and_rerun_network_coverage"
            ),
        },
        {
            "requirement_id": "polarization::coverage_gate_hits",
            "metric": "n_gate_hits",
            "value": curr_gate_hits,
            "expected": ">=1",
            "status": "pass" if curr_gate_hits >= 1 else "watch",
            "required_action": (
                "none"
                if curr_gate_hits >= 1
                else "need_additional_high_quality_three_detector_events_to_raise_gate_hits"
            ),
        },
        {
            "requirement_id": "polarization::best_u2_below_watch_boundary",
            "metric": "best_u2_scalar_overlap_proxy",
            "value": best_u2_watch_value if math.isfinite(best_u2_watch_value) else "",
            "expected": "<0.5 (usable>=2)",
            "status": (
                "pass"
                if (math.isfinite(best_u2_watch_value) and best_u2_watch_value < watch_boundary)
                else "watch"
            ),
            "required_action": (
                "none"
                if (math.isfinite(best_u2_watch_value) and best_u2_watch_value < watch_boundary)
                else "need_best_u2_reduction_with_usable_events_kept_at_least_2"
            ),
        },
    ]
    blocked_priority_class = (
        "low_blocked_missing_high_quality_three_detector_events"
        if (decision in {"no_change_hold", "hash_update_rerun_no_improvement"} and curr_gate_hits < 1)
        else "active"
    )

    out_json = outdir / f"{args.prefix}.json"
    out_csv = outdir / f"{args.prefix}.csv"
    out_png = outdir / f"{args.prefix}.png"
    out_checklist_csv = outdir / f"{args.prefix}_checklist.csv"
    _write_csv(out_csv, [row])
    _write_csv(out_checklist_csv, checklist_rows)
    _plot_summary(row, out_png)

    payload: Dict[str, Any] = {
        "generated_utc": row["generated_utc"],
        "schema": "wavep.gw.polarization.network_event_update_watch.v1",
        "phase": 8,
        "step": str(args.step_tag),
        "inputs": {
            "event_list": str(event_list_path).replace("\\", "/"),
            "anchor_event": str(args.anchor_event),
            "coverage_prefix": str(args.coverage_prefix),
            "force_rerun": bool(args.force_rerun),
        },
        "summary": {
            "overall_status": overall_status,
            "decision": decision,
            "next_action": next_action,
            "update_event_type": update_event_type,
            "update_event_detected": bool(update_event_detected),
            "rerun_triggered": bool(rerun_triggered),
            "rerun_ok": bool(coverage_ok),
            "event_counter": int(event_counter),
            "n_three_detector_events": int(len(events)),
            "current_input_hash": digest,
            "previous_input_hash": prev_digest,
            "gate_hits_previous": int(prev_gate_hits),
            "gate_hits_current": int(curr_gate_hits),
            "best_u2_previous": float(prev_best_u2) if math.isfinite(prev_best_u2) else None,
            "best_u2_current": float(curr_best_u2) if math.isfinite(curr_best_u2) else None,
            "improvement_detected": bool(improvement_detected),
            "needed_gate_hits_for_pass": int(needed_gate_hits),
            "needed_best_u2_drop_for_pass": float(needed_best_u2_drop) if math.isfinite(needed_best_u2_drop) else None,
            "blocked_priority_class": blocked_priority_class,
        },
        "coverage_gap_checklist": checklist_rows,
        "three_detector_events": events,
        "coverage_metrics_previous": prev_metrics,
        "coverage_metrics_current": curr_metrics,
        "coverage_rerun": coverage_run,
        "input_signature": signature,
        "outputs": {
            "audit_json": str(out_json).replace("\\", "/"),
            "audit_csv": str(out_csv).replace("\\", "/"),
            "audit_png": str(out_png).replace("\\", "/"),
            "audit_checklist_csv": str(out_checklist_csv).replace("\\", "/"),
        },
        "falsification": {
            "hard_pass_if": [
                "coverage_metrics_current.n_gate_hits increases and best_u2 decreases below watch boundary under usable>=2",
            ],
            "watch_if": [
                "three-detector hash is stable (no change) and coverage remains unchanged",
                "hash changed but rerun keeps best_u2 at watch boundary with usable>=2",
            ],
            "reject_if": [
                "coverage rerun repeatedly fails after hash change",
            ],
        },
    }
    _write_json(out_json, payload)

    state_payload = {
        "generated_utc": row["generated_utc"],
        "last_digest_sha256": digest,
        "event_counter": int(event_counter),
        "last_update_event_type": update_event_type,
        "last_update_event_detected": bool(update_event_detected),
        "last_rerun_triggered": bool(rerun_triggered),
        "last_rerun_ok": bool(coverage_ok),
        "last_coverage_metrics": curr_metrics,
        "last_coverage_prefix": str(args.coverage_prefix),
    }
    _write_json(state_path, state_payload)

    copied: List[str] = []
    for src in [out_json, out_csv, out_png, out_checklist_csv]:
        dst = public_outdir / src.name
        shutil.copy2(src, dst)
        copied.append(str(dst).replace("\\", "/"))

    payload["outputs"]["public_copies"] = copied
    _write_json(out_json, payload)
    shutil.copy2(out_json, public_outdir / out_json.name)

    try:
        worklog.append_event(
            {
                "event_type": "gw_polarization_network_event_update_watch",
                "argv": list(sys.argv),
                    "outputs": {"audit_json": out_json, "audit_csv": out_csv, "audit_png": out_png, "state_json": state_path},
                    "metrics": {
                    "overall_status": overall_status,
                    "decision": decision,
                    "update_event_type": update_event_type,
                    "update_event_detected": bool(update_event_detected),
                    "rerun_triggered": bool(rerun_triggered),
                    "rerun_ok": bool(coverage_ok),
                    "n_three_detector_events": int(len(events)),
                        "n_gate_hits": int(curr_gate_hits),
                        "best_u2_scalar_overlap_proxy": float(curr_best_u2) if math.isfinite(curr_best_u2) else float("nan"),
                        "event_counter": int(event_counter),
                        "needed_gate_hits_for_pass": int(needed_gate_hits),
                        "needed_best_u2_drop_for_pass": float(needed_best_u2_drop) if math.isfinite(needed_best_u2_drop) else float("nan"),
                    },
                }
            )
    except Exception:
        pass

    print(
        "[ok] update_event_type={evt} detected={det} rerun={rerun} ok={ok}".format(
            evt=update_event_type,
            det=int(update_event_detected),
            rerun=int(rerun_triggered),
            ok=int(bool(coverage_ok)),
        )
    )
    print(
        "[ok] gate_hits prev={prev} curr={curr} | best_u2 prev={p} curr={c}".format(
            prev=prev_gate_hits,
            curr=curr_gate_hits,
            p=_fmt(prev_best_u2),
            c=_fmt(curr_best_u2),
        )
    )
    print(f"[ok] json: {out_json}")
    print(f"[ok] csv : {out_csv}")
    print(f"[ok] checklist_csv : {out_checklist_csv}")
    print(f"[ok] png : {out_png}")
    print(f"[ok] state: {state_path}")
    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
