from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(ROOT) not in sys.path` を満たす経路を評価する。
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.summary import worklog


# 関数: `_read_json` の入出力契約と処理意図を定義する。
def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# 関数: `_write_json` の入出力契約と処理意図を定義する。

def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# 関数: `_fmt_float` の入出力契約と処理意図を定義する。

def _fmt_float(x: float, digits: int = 6) -> str:
    # 条件分岐: `x == 0.0` を満たす経路を評価する。
    if x == 0.0:
        return "0"

    ax = abs(x)
    # 条件分岐: `ax >= 1e4 or ax < 1e-3` を満たす経路を評価する。
    if ax >= 1e4 or ax < 1e-3:
        return f"{x:.{digits}g}"

    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


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


# 関数: `_sigma_median` の入出力契約と処理意図を定義する。

def _sigma_median(arr: np.ndarray) -> Optional[float]:
    data = np.asarray(arr, dtype=float)
    n = int(data.size)
    # 条件分岐: `n < 5` を満たす経路を評価する。
    if n < 5:
        return None

    q25, q75 = np.quantile(data, [0.25, 0.75])
    iqr = float(q75 - q25)
    # 条件分岐: `iqr > 0` を満たす経路を評価する。
    if iqr > 0:
        sigma = iqr / 1.349
    else:
        sigma = float(np.std(data, ddof=1))

    # 条件分岐: `not math.isfinite(sigma) or sigma <= 0.0` を満たす経路を評価する。

    if not math.isfinite(sigma) or sigma <= 0.0:
        return None

    return float(1.2533141373155001 * sigma / math.sqrt(float(n)))


# 関数: `_delay_stat` の入出力契約と処理意図を定義する。

def _delay_stat(
    dt0_in: np.ndarray,
    dt1_in: np.ndarray,
    *,
    window_ns: Optional[float],
    offset_ns: float,
    clip_quantile: Optional[float],
    min_n: int,
) -> Dict[str, Any]:
    dt0 = np.asarray(dt0_in, dtype=float)
    dt1 = np.asarray(dt1_in, dtype=float) + float(offset_ns)
    dt1 = dt1[dt1 >= 0.0]

    # 条件分岐: `window_ns is not None and math.isfinite(float(window_ns))` を満たす経路を評価する。
    if window_ns is not None and math.isfinite(float(window_ns)):
        w = float(window_ns)
        dt0 = dt0[dt0 <= w]
        dt1 = dt1[dt1 <= w]

    # 条件分岐: `clip_quantile is not None` を満たす経路を評価する。

    if clip_quantile is not None:
        q = float(clip_quantile)
        # 条件分岐: `0.0 < q < 1.0` を満たす経路を評価する。
        if 0.0 < q < 1.0:
            # 条件分岐: `dt0.size > 0` を満たす経路を評価する。
            if dt0.size > 0:
                q0 = float(np.quantile(dt0, q))
                dt0 = dt0[dt0 <= q0]

            # 条件分岐: `dt1.size > 0` を満たす経路を評価する。

            if dt1.size > 0:
                q1 = float(np.quantile(dt1, q))
                dt1 = dt1[dt1 <= q1]

    n0 = int(dt0.size)
    n1 = int(dt1.size)
    # 条件分岐: `n0 < min_n or n1 < min_n` を満たす経路を評価する。
    if n0 < min_n or n1 < min_n:
        return {
            "n0": n0,
            "n1": n1,
            "median0_ns": None,
            "median1_ns": None,
            "delta_median_ns": None,
            "sigma_delta_ns": None,
            "z_delta_median": None,
        }

    median0 = float(np.median(dt0))
    median1 = float(np.median(dt1))
    delta = median0 - median1
    s0 = _sigma_median(dt0)
    s1 = _sigma_median(dt1)
    # 条件分岐: `s0 is None or s1 is None` を満たす経路を評価する。
    if s0 is None or s1 is None:
        return {
            "n0": n0,
            "n1": n1,
            "median0_ns": median0,
            "median1_ns": median1,
            "delta_median_ns": delta,
            "sigma_delta_ns": None,
            "z_delta_median": None,
        }

    sigma_delta = float(math.sqrt(s0 * s0 + s1 * s1))
    # 条件分岐: `sigma_delta <= 0.0` を満たす経路を評価する。
    if sigma_delta <= 0.0:
        z = None
    else:
        z = float(delta / sigma_delta)

    return {
        "n0": n0,
        "n1": n1,
        "median0_ns": median0,
        "median1_ns": median1,
        "delta_median_ns": delta,
        "sigma_delta_ns": sigma_delta,
        "z_delta_median": z,
    }


# 関数: `_max_abs_z` の入出力契約と処理意図を定義する。

def _max_abs_z(entry: Dict[str, Any]) -> Optional[float]:
    az = entry.get("alice", {}).get("z_delta_median") if isinstance(entry.get("alice"), dict) else None
    bz = entry.get("bob", {}).get("z_delta_median") if isinstance(entry.get("bob"), dict) else None
    vals = []
    for value in (az, bz):
        # 条件分岐: `isinstance(value, (int, float)) and math.isfinite(float(value))` を満たす経路を評価する。
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            vals.append(abs(float(value)))

    # 条件分岐: `not vals` を満たす経路を評価する。

    if not vals:
        return None

    return max(vals)


# 関数: `_run_audit` の入出力契約と処理意図を定義する。

def _run_audit(
    *,
    delay_npz: Path,
    window_metrics_json: Path,
    z_threshold: float,
    clip_quantile_default: float,
    offset_span_ns: float,
    offset_step_ns: float,
    min_n: int,
) -> Dict[str, Any]:
    metrics = _read_json(window_metrics_json)
    windows = metrics.get("config", {}).get("windows_ns")
    # 条件分岐: `not isinstance(windows, list) or not windows` を満たす経路を評価する。
    if not isinstance(windows, list) or not windows:
        windows = metrics.get("window_sweep", {}).get("values")

    windows_ns = [float(w) for w in windows if isinstance(w, (int, float))]
    windows_ns = sorted(set(windows_ns))

    natural_window = metrics.get("natural_window", {}).get("recommended_window_ns")
    # 条件分岐: `not isinstance(natural_window, (int, float))` を満たす経路を評価する。
    if not isinstance(natural_window, (int, float)):
        natural_window = windows_ns[-1] if windows_ns else None

    natural_window_ns = float(natural_window) if natural_window is not None else None

    npz = np.load(delay_npz)
    a0 = np.asarray(npz["alice_setting0_dt_ns"], dtype=float)
    a1 = np.asarray(npz["alice_setting1_dt_ns"], dtype=float)
    b0 = np.asarray(npz["bob_setting0_dt_ns"], dtype=float)
    b1 = np.asarray(npz["bob_setting1_dt_ns"], dtype=float)

    baseline = {
        "window_ns": natural_window_ns,
        "offset_ns": 0.0,
        "clip_quantile": None,
        "alice": _delay_stat(a0, a1, window_ns=natural_window_ns, offset_ns=0.0, clip_quantile=None, min_n=min_n),
        "bob": _delay_stat(b0, b1, window_ns=natural_window_ns, offset_ns=0.0, clip_quantile=None, min_n=min_n),
    }
    baseline["max_abs_z"] = _max_abs_z(baseline)

    window_sweep: List[Dict[str, Any]] = []
    for w in windows_ns:
        item = {
            "window_ns": float(w),
            "offset_ns": 0.0,
            "clip_quantile": None,
            "alice": _delay_stat(a0, a1, window_ns=float(w), offset_ns=0.0, clip_quantile=None, min_n=min_n),
            "bob": _delay_stat(b0, b1, window_ns=float(w), offset_ns=0.0, clip_quantile=None, min_n=min_n),
        }
        item["max_abs_z"] = _max_abs_z(item)
        window_sweep.append(item)

    clipped_window_sweep: List[Dict[str, Any]] = []
    for w in windows_ns:
        item = {
            "window_ns": float(w),
            "offset_ns": 0.0,
            "clip_quantile": float(clip_quantile_default),
            "alice": _delay_stat(
                a0,
                a1,
                window_ns=float(w),
                offset_ns=0.0,
                clip_quantile=float(clip_quantile_default),
                min_n=min_n,
            ),
            "bob": _delay_stat(
                b0,
                b1,
                window_ns=float(w),
                offset_ns=0.0,
                clip_quantile=float(clip_quantile_default),
                min_n=min_n,
            ),
        }
        item["max_abs_z"] = _max_abs_z(item)
        clipped_window_sweep.append(item)

    offsets = np.arange(-float(offset_span_ns), float(offset_span_ns) + 0.5 * float(offset_step_ns), float(offset_step_ns))
    offset_sweep: List[Dict[str, Any]] = []
    for off in offsets.tolist():
        item = {
            "window_ns": natural_window_ns,
            "offset_ns": float(off),
            "clip_quantile": None,
            "alice": _delay_stat(a0, a1, window_ns=natural_window_ns, offset_ns=float(off), clip_quantile=None, min_n=min_n),
            "bob": _delay_stat(b0, b1, window_ns=natural_window_ns, offset_ns=float(off), clip_quantile=None, min_n=min_n),
        }
        item["max_abs_z"] = _max_abs_z(item)
        offset_sweep.append(item)

    accidental_quantiles = [1.0, 0.98, 0.95, 0.9, 0.85]
    accidental_sweep: List[Dict[str, Any]] = []
    for q in accidental_quantiles:
        clip = None if q >= 0.999 else float(q)
        item = {
            "window_ns": natural_window_ns,
            "offset_ns": 0.0,
            "clip_quantile": clip,
            "alice": _delay_stat(a0, a1, window_ns=natural_window_ns, offset_ns=0.0, clip_quantile=clip, min_n=min_n),
            "bob": _delay_stat(b0, b1, window_ns=natural_window_ns, offset_ns=0.0, clip_quantile=clip, min_n=min_n),
        }
        item["max_abs_z"] = _max_abs_z(item)
        accidental_sweep.append(item)

    # 関数: `_max_z` の入出力契約と処理意図を定義する。

    def _max_z(items: List[Dict[str, Any]]) -> Optional[float]:
        vals: List[float] = []
        for item in items:
            z = item.get("max_abs_z")
            # 条件分岐: `isinstance(z, (int, float)) and math.isfinite(float(z))` を満たす経路を評価する。
            if isinstance(z, (int, float)) and math.isfinite(float(z)):
                vals.append(float(z))

        return max(vals) if vals else None

    max_window_raw = _max_z(window_sweep)
    max_window_clip = _max_z(clipped_window_sweep)
    max_offset = _max_z(offset_sweep)
    max_accidental = _max_z(accidental_sweep)
    all_candidates = [v for v in [max_window_raw, max_window_clip, max_offset, max_accidental] if v is not None]
    max_any = max(all_candidates) if all_candidates else None

    hard_gate_applicable = bool(max_any is not None and max_any >= float(z_threshold))
    decision = "hard_gate_candidate" if hard_gate_applicable else "keep_watch_nonhard_gate"

    return {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "schema": "wavep.quantum.bell_kwiat_delay_signature_watch_audit.v1",
        "dataset_id": "kwiat2013_prl111_130406_05082013_15",
        "inputs": {
            "delay_signature_ref_samples_npz": str(delay_npz).replace("\\", "/"),
            "window_sweep_metrics_json": str(window_metrics_json).replace("\\", "/"),
        },
        "thresholds": {
            "z_hard_gate_min": float(z_threshold),
            "clip_quantile_default": float(clip_quantile_default),
            "offset_span_ns": float(offset_span_ns),
            "offset_step_ns": float(offset_step_ns),
            "min_samples_per_setting": int(min_n),
        },
        "baseline": baseline,
        "window_sweep_raw": window_sweep,
        "window_sweep_tail_clipped": clipped_window_sweep,
        "offset_sweep": offset_sweep,
        "accidental_policy_sweep": accidental_sweep,
        "summary": {
            "baseline_max_abs_z": baseline.get("max_abs_z"),
            "max_window_raw_abs_z": max_window_raw,
            "max_window_tail_clipped_abs_z": max_window_clip,
            "max_offset_abs_z": max_offset,
            "max_accidental_policy_abs_z": max_accidental,
            "max_abs_z_any": max_any,
            "hard_gate_applicable": hard_gate_applicable,
            "decision": decision,
            "decision_note": (
                "window/offset/accidental-like sweeps stay below |z| threshold; keep as non-hard watch."
                if not hard_gate_applicable
                else "at least one sweep exceeds hard threshold; review hard-gate promotion."
            ),
        },
    }


# 関数: `_write_csv` の入出力契約と処理意図を定義する。

def _write_csv(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    # 関数: `_push` の入出力契約と処理意図を定義する。
    def _push(tag: str, items: List[Dict[str, Any]]) -> None:
        for item in items:
            rows.append(
                {
                    "sweep": tag,
                    "window_ns": item.get("window_ns"),
                    "offset_ns": item.get("offset_ns"),
                    "clip_quantile": item.get("clip_quantile"),
                    "max_abs_z": item.get("max_abs_z"),
                    "alice_z": item.get("alice", {}).get("z_delta_median") if isinstance(item.get("alice"), dict) else None,
                    "bob_z": item.get("bob", {}).get("z_delta_median") if isinstance(item.get("bob"), dict) else None,
                    "alice_n0": item.get("alice", {}).get("n0") if isinstance(item.get("alice"), dict) else None,
                    "alice_n1": item.get("alice", {}).get("n1") if isinstance(item.get("alice"), dict) else None,
                    "bob_n0": item.get("bob", {}).get("n0") if isinstance(item.get("bob"), dict) else None,
                    "bob_n1": item.get("bob", {}).get("n1") if isinstance(item.get("bob"), dict) else None,
                }
            )

    baseline = payload.get("baseline") if isinstance(payload.get("baseline"), dict) else {}
    # 条件分岐: `baseline` を満たす経路を評価する。
    if baseline:
        _push("baseline", [baseline])

    _push("window_raw", payload.get("window_sweep_raw") if isinstance(payload.get("window_sweep_raw"), list) else [])
    _push(
        "window_tail_clipped",
        payload.get("window_sweep_tail_clipped") if isinstance(payload.get("window_sweep_tail_clipped"), list) else [],
    )
    _push("offset", payload.get("offset_sweep") if isinstance(payload.get("offset_sweep"), list) else [])
    _push(
        "accidental_policy",
        payload.get("accidental_policy_sweep") if isinstance(payload.get("accidental_policy_sweep"), list) else [],
    )

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "sweep",
                "window_ns",
                "offset_ns",
                "clip_quantile",
                "max_abs_z",
                "alice_z",
                "bob_z",
                "alice_n0",
                "alice_n1",
                "bob_n0",
                "bob_n1",
            ]
        )
        for row in rows:
            w.writerow(
                [
                    row["sweep"],
                    "" if row["window_ns"] is None else _fmt_float(float(row["window_ns"]), 6),
                    "" if row["offset_ns"] is None else _fmt_float(float(row["offset_ns"]), 6),
                    "" if row["clip_quantile"] is None else _fmt_float(float(row["clip_quantile"]), 6),
                    "" if row["max_abs_z"] is None else _fmt_float(float(row["max_abs_z"]), 6),
                    "" if row["alice_z"] is None else _fmt_float(float(row["alice_z"]), 6),
                    "" if row["bob_z"] is None else _fmt_float(float(row["bob_z"]), 6),
                    row["alice_n0"] if row["alice_n0"] is not None else "",
                    row["alice_n1"] if row["alice_n1"] is not None else "",
                    row["bob_n0"] if row["bob_n0"] is not None else "",
                    row["bob_n1"] if row["bob_n1"] is not None else "",
                ]
            )


# 関数: `_plot` の入出力契約と処理意図を定義する。

def _plot(path: Path, payload: Dict[str, Any]) -> None:
    _set_japanese_font()
    z_thr = float(payload.get("thresholds", {}).get("z_hard_gate_min", 3.0))

    window_raw = payload.get("window_sweep_raw") if isinstance(payload.get("window_sweep_raw"), list) else []
    window_clip = payload.get("window_sweep_tail_clipped") if isinstance(payload.get("window_sweep_tail_clipped"), list) else []
    offset_rows = payload.get("offset_sweep") if isinstance(payload.get("offset_sweep"), list) else []
    accidental_rows = payload.get("accidental_policy_sweep") if isinstance(payload.get("accidental_policy_sweep"), list) else []

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.6), dpi=170)

    ax0 = axes[0]
    xy_raw = [
        (float(r.get("window_ns")), float(r.get("max_abs_z")))
        for r in window_raw
        if isinstance(r.get("window_ns"), (int, float)) and isinstance(r.get("max_abs_z"), (int, float))
    ]
    xy_clp = [
        (float(r.get("window_ns")), float(r.get("max_abs_z")))
        for r in window_clip
        if isinstance(r.get("window_ns"), (int, float)) and isinstance(r.get("max_abs_z"), (int, float))
    ]
    x_raw = [x for x, _ in xy_raw]
    y_raw = [y for _, y in xy_raw]
    x_clp = [x for x, _ in xy_clp]
    y_clp = [y for _, y in xy_clp]
    # 条件分岐: `x_raw and y_raw` を満たす経路を評価する。
    if x_raw and y_raw:
        ax0.plot(x_raw, y_raw, marker="o", linewidth=1.8, label="raw")

    # 条件分岐: `x_clp and y_clp` を満たす経路を評価する。

    if x_clp and y_clp:
        ax0.plot(x_clp, y_clp, marker="s", linewidth=1.5, label="tail-clipped")

    ax0.axhline(z_thr, color="#333333", linestyle="--", linewidth=1.1)
    ax0.set_xlabel("window half-width (ns)")
    ax0.set_ylabel("max abs(z_delay)")
    ax0.set_title("Window sweep")
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="upper right")

    ax1 = axes[1]
    x_off = [float(r.get("offset_ns")) for r in offset_rows if isinstance(r.get("offset_ns"), (int, float))]
    y_off = [float(r.get("max_abs_z")) for r in offset_rows if isinstance(r.get("max_abs_z"), (int, float))]
    # 条件分岐: `x_off and y_off` を満たす経路を評価する。
    if x_off and y_off:
        ax1.plot(x_off, y_off, marker="o", linewidth=1.8, color="#1f77b4")

    ax1.axhline(z_thr, color="#333333", linestyle="--", linewidth=1.1)
    ax1.set_xlabel("relative offset applied to setting-1 (ns)")
    ax1.set_ylabel("max abs(z_delay)")
    ax1.set_title("Offset sweep")
    ax1.grid(True, alpha=0.25)

    ax2 = axes[2]
    x_acc = []
    y_acc = []
    for r in accidental_rows:
        q = r.get("clip_quantile")
        z = r.get("max_abs_z")
        # 条件分岐: `not isinstance(z, (int, float))` を満たす経路を評価する。
        if not isinstance(z, (int, float)):
            continue

        q_plot = 1.0 if q is None else float(q)
        x_acc.append(q_plot)
        y_acc.append(float(z))

    # 条件分岐: `x_acc and y_acc` を満たす経路を評価する。

    if x_acc and y_acc:
        ax2.plot(x_acc, y_acc, marker="o", linewidth=1.8, color="#ff7f0e")

    ax2.axhline(z_thr, color="#333333", linestyle="--", linewidth=1.1)
    ax2.set_xlabel("tail clip quantile (1.0=no clip)")
    ax2.set_ylabel("max abs(z_delay)")
    ax2.set_title("Accidental-like policy sweep")
    ax2.grid(True, alpha=0.25)

    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    decision = str(summary.get("decision") or "unknown")
    fig.suptitle(f"Kwiat delay-signature watch audit ({decision})", y=1.02)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# 関数: `main` の入出力契約と処理意図を定義する。

def main(argv: Optional[Sequence[str]] = None) -> int:
    default_npz = ROOT / "output" / "public" / "quantum" / "bell" / "kwiat2013_prl111_130406_05082013_15" / "delay_signature_ref_samples.npz"
    default_window_json = ROOT / "output" / "public" / "quantum" / "bell" / "kwiat2013_prl111_130406_05082013_15" / "window_sweep_metrics.json"
    default_outdir = ROOT / "output" / "private" / "quantum"
    default_public_outdir = ROOT / "output" / "public" / "quantum"

    ap = argparse.ArgumentParser(description="Audit Kwiat delay_signature watch convergence with window/offset/accidental policy sweeps.")
    ap.add_argument("--delay-npz", type=str, default=str(default_npz), help="Input delay signature samples NPZ.")
    ap.add_argument("--window-metrics-json", type=str, default=str(default_window_json), help="Input window sweep metrics JSON.")
    ap.add_argument("--outdir", type=str, default=str(default_outdir), help="Private output directory.")
    ap.add_argument("--public-outdir", type=str, default=str(default_public_outdir), help="Public output directory.")
    ap.add_argument("--z-threshold", type=float, default=3.0, help="Hard gate threshold on abs(z).")
    ap.add_argument("--clip-quantile-default", type=float, default=0.9, help="Default tail clip quantile for accidental-like sweep.")
    ap.add_argument("--offset-span-ns", type=float, default=20.0, help="Offset sweep span in ns (symmetric).")
    ap.add_argument("--offset-step-ns", type=float, default=5.0, help="Offset sweep step in ns.")
    ap.add_argument("--min-n", type=int, default=30, help="Minimum samples per setting.")
    ap.add_argument("--no-public-copy", action="store_true", help="Do not copy outputs to public directory.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    delay_npz = Path(args.delay_npz)
    window_json = Path(args.window_metrics_json)
    outdir = Path(args.outdir)
    public_outdir = Path(args.public_outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    public_outdir.mkdir(parents=True, exist_ok=True)

    payload = _run_audit(
        delay_npz=delay_npz,
        window_metrics_json=window_json,
        z_threshold=float(args.z_threshold),
        clip_quantile_default=float(args.clip_quantile_default),
        offset_span_ns=float(args.offset_span_ns),
        offset_step_ns=float(args.offset_step_ns),
        min_n=int(args.min_n),
    )

    out_json = outdir / "bell_kwiat_delay_signature_watch_audit.json"
    out_csv = outdir / "bell_kwiat_delay_signature_watch_audit.csv"
    out_png = outdir / "bell_kwiat_delay_signature_watch_audit.png"

    _write_json(out_json, payload)
    _write_csv(out_csv, payload)
    _plot(out_png, payload)

    copied: List[Path] = []
    # 条件分岐: `not args.no_public_copy` を満たす経路を評価する。
    if not args.no_public_copy:
        for src in (out_json, out_csv, out_png):
            dst = public_outdir / src.name
            shutil.copy2(src, dst)
            copied.append(dst)

    try:
        worklog.append_event(
            {
                "event_type": "quantum_bell_kwiat_delay_signature_watch_audit",
                "argv": sys.argv,
                "inputs": {"delay_npz": delay_npz, "window_metrics_json": window_json},
                "outputs": {
                    "json": out_json,
                    "csv": out_csv,
                    "png": out_png,
                    "public_copies": copied,
                },
                "metrics": payload.get("summary"),
            }
        )
    except Exception:
        pass

    print(f"[ok] json : {out_json}")
    print(f"[ok] csv  : {out_csv}")
    print(f"[ok] png  : {out_png}")
    # 条件分岐: `copied` を満たす経路を評価する。
    if copied:
        print(f"[ok] public copies: {len(copied)} files -> {public_outdir}")

    return 0


# 条件分岐: `__name__ == "__main__"` を満たす経路を評価する。

if __name__ == "__main__":
    raise SystemExit(main())
