from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
# 条件分岐: `str(_ROOT) not in sys.path` を満たす経路を評価する。
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.summary import worklog


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


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _fmt_float(x: float, digits: int = 6) -> str:
    # 条件分岐: `x == 0.0` を満たす経路を評価する。
    if x == 0.0:
        return "0"

    ax = abs(x)
    # 条件分岐: `ax >= 1e4 or ax < 1e-3` を満たす経路を評価する。
    if ax >= 1e4 or ax < 1e-3:
        return f"{x:.{digits}g}"

    return f"{x:.{digits}f}".rstrip("0").rstrip(".")


def _to_float(v: Any) -> Optional[float]:
    try:
        val = float(v)
    except Exception:
        return None

    # 条件分岐: `math.isnan(val) or math.isinf(val)` を満たす経路を評価する。

    if math.isnan(val) or math.isinf(val):
        return None

    return val


def _compute_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    reject_z = _to_float((payload.get("gate") or {}).get("z_reject"))
    z_gate = 3.0 if reject_z is None else float(reject_z)

    for ch in payload.get("channels") or []:
        observed = _to_float(ch.get("observed"))
        sigma = _to_float(ch.get("observed_sigma"))
        pred_scalar = _to_float(ch.get("pmodel_scalar_prediction"))
        pred_ref = _to_float(ch.get("reference_prediction"))
        residual_scalar = None
        z_scalar = None
        status = "inconclusive"

        # 条件分岐: `observed is not None and pred_scalar is not None` を満たす経路を評価する。
        if observed is not None and pred_scalar is not None:
            residual_scalar = observed - pred_scalar
            # 条件分岐: `sigma is not None and sigma > 0` を満たす経路を評価する。
            if sigma is not None and sigma > 0:
                z_scalar = residual_scalar / sigma
                status = "reject" if abs(z_scalar) > z_gate else "pass"

        rows.append(
            {
                "id": str(ch.get("id") or ""),
                "label": str(ch.get("label") or ch.get("id") or ""),
                "unit": str(ch.get("unit") or ""),
                "observed": observed,
                "observed_sigma": sigma,
                "pmodel_scalar_prediction": pred_scalar,
                "reference_prediction": pred_ref,
                "residual_scalar": residual_scalar,
                "z_scalar": z_scalar,
                "status": status,
                "note": str(ch.get("note") or ""),
            }
        )

    return rows


def _build_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    n_pass = sum(1 for r in rows if r.get("status") == "pass")
    n_reject = sum(1 for r in rows if r.get("status") == "reject")
    n_inconclusive = sum(1 for r in rows if r.get("status") == "inconclusive")

    by_id = {str(r.get("id")): r for r in rows}
    geodetic_status = str((by_id.get("geodetic_precession") or {}).get("status") or "inconclusive")
    frame_status = str((by_id.get("frame_dragging") or {}).get("status") or "inconclusive")

    overall = "inconclusive"
    # 条件分岐: `frame_status == "reject"` を満たす経路を評価する。
    if frame_status == "reject":
        overall = "reject"
    # 条件分岐: 前段条件が不成立で、`n_reject == 0 and n_inconclusive == 0` を追加評価する。
    elif n_reject == 0 and n_inconclusive == 0:
        overall = "pass"

    return {
        "channels_n": len(rows),
        "pass_n": n_pass,
        "reject_n": n_reject,
        "inconclusive_n": n_inconclusive,
        "geodetic_status": geodetic_status,
        "frame_dragging_status": frame_status,
        "overall_status": overall,
        "decision": (
            "scalar_limit_rejected_by_frame_dragging"
            if overall == "reject"
            else "scalar_limit_not_rejected_by_current_gate"
        ),
    }


def _plot(rows: Sequence[Dict[str, Any]], out_png: Path) -> None:
    _set_japanese_font()
    labels = [str(r.get("label") or r.get("id") or "") for r in rows]
    obs = np.array([float(r["observed"]) if r.get("observed") is not None else np.nan for r in rows], dtype=float)
    sig = np.array([float(r["observed_sigma"]) if r.get("observed_sigma") is not None else np.nan for r in rows], dtype=float)
    pred = np.array(
        [float(r["pmodel_scalar_prediction"]) if r.get("pmodel_scalar_prediction") is not None else np.nan for r in rows],
        dtype=float,
    )
    zvals = np.array([float(r["z_scalar"]) if r.get("z_scalar") is not None else np.nan for r in rows], dtype=float)

    x = np.arange(len(rows), dtype=float)
    width = 0.35

    fig = plt.figure(figsize=(11.0, 5.8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.bar(x - width / 2.0, obs, width=width, color="#1f77b4", alpha=0.9, label="observed (GP-B)")
    ax1.bar(x + width / 2.0, pred, width=width, color="#ff7f0e", alpha=0.9, label="P-model scalar")
    for i, (xi, yi, si) in enumerate(zip(x, obs, sig)):
        # 条件分岐: `math.isfinite(float(yi)) and math.isfinite(float(si)) and float(si) > 0` を満たす経路を評価する。
        if math.isfinite(float(yi)) and math.isfinite(float(si)) and float(si) > 0:
            ax1.errorbar(
                [xi - width / 2.0],
                [yi],
                yerr=[si],
                fmt="none",
                ecolor="#1f77b4",
                elinewidth=1.5,
                capsize=4,
            )

        # 条件分岐: `math.isfinite(float(yi))` を満たす経路を評価する。

        if math.isfinite(float(yi)):
            ax1.text(xi - width / 2.0, yi, _fmt_float(float(yi), 4), ha="center", va="bottom", fontsize=9)

        # 条件分岐: `math.isfinite(float(pred[i]))` を満たす経路を評価する。

        if math.isfinite(float(pred[i])):
            ax1.text(xi + width / 2.0, pred[i], _fmt_float(float(pred[i]), 4), ha="center", va="bottom", fontsize=9)

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=12, ha="right")
    unit = str(rows[0].get("unit") or "") if rows else ""
    ax1.set_ylabel(f"value [{unit}]" if unit else "value")
    ax1.set_title("GP-B observables vs scalar-limit prediction")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper right")

    ax2 = fig.add_subplot(1, 2, 2)
    colors = ["#2ca02c" if (math.isfinite(float(z)) and abs(float(z)) <= 3.0) else "#d62728" for z in zvals]
    ax2.bar(x, np.nan_to_num(zvals, nan=0.0), color=colors, alpha=0.92)
    ax2.axhline(3.0, color="#333333", linestyle="--", linewidth=1.1)
    ax2.axhline(-3.0, color="#333333", linestyle="--", linewidth=1.1)
    for xi, zi in zip(x, zvals):
        # 条件分岐: `math.isfinite(float(zi))` を満たす経路を評価する。
        if math.isfinite(float(zi)):
            ax2.text(xi, zi, _fmt_float(float(zi), 3), ha="center", va="bottom", fontsize=9)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=12, ha="right")
    ax2.set_ylabel("z = (obs - pred_scalar) / sigma_obs")
    ax2.set_title("Scalar-limit rejection gate")
    ax2.grid(True, alpha=0.25)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "id",
                "label",
                "unit",
                "observed",
                "observed_sigma",
                "pmodel_scalar_prediction",
                "reference_prediction",
                "residual_scalar",
                "z_scalar",
                "status",
                "note",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.get("id", ""),
                    r.get("label", ""),
                    r.get("unit", ""),
                    "" if r.get("observed") is None else _fmt_float(float(r["observed"]), 6),
                    "" if r.get("observed_sigma") is None else _fmt_float(float(r["observed_sigma"]), 6),
                    ""
                    if r.get("pmodel_scalar_prediction") is None
                    else _fmt_float(float(r["pmodel_scalar_prediction"]), 6),
                    "" if r.get("reference_prediction") is None else _fmt_float(float(r["reference_prediction"]), 6),
                    "" if r.get("residual_scalar") is None else _fmt_float(float(r["residual_scalar"]), 6),
                    "" if r.get("z_scalar") is None else _fmt_float(float(r["z_scalar"]), 6),
                    r.get("status", ""),
                    r.get("note", ""),
                ]
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    root = _ROOT
    default_data = root / "data" / "theory" / "gpb_scalar_limit_audit.json"
    default_outdir = root / "output" / "private" / "theory"
    default_public_outdir = root / "output" / "public" / "theory"

    ap = argparse.ArgumentParser(description="GP-B scalar-limit audit (geodetic vs frame-dragging).")
    ap.add_argument("--data", type=str, default=str(default_data), help="Input JSON.")
    ap.add_argument("--outdir", type=str, default=str(default_outdir), help="Output directory.")
    ap.add_argument(
        "--public-outdir",
        type=str,
        default=str(default_public_outdir),
        help="Public output directory.",
    )
    ap.add_argument(
        "--no-public-copy",
        action="store_true",
        help="Do not copy outputs to public directory.",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)

    data_path = Path(args.data)
    outdir = Path(args.outdir)
    public_outdir = Path(args.public_outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    public_outdir.mkdir(parents=True, exist_ok=True)

    payload_in = _read_json(data_path)
    rows = _compute_rows(payload_in)
    summary = _build_summary(rows)

    out_json = outdir / "gpb_scalar_limit_audit.json"
    out_csv = outdir / "gpb_scalar_limit_audit.csv"
    out_png = outdir / "gpb_scalar_limit_audit.png"

    payload_out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "schema": payload_in.get("schema") or "wavep.theory.gpb_scalar_limit_audit.v1",
        "title": payload_in.get("title") or "GP-B scalar-limit audit",
        "input": str(data_path).replace("\\", "/"),
        "gate": payload_in.get("gate") or {"z_reject": 3.0},
        "source": payload_in.get("source") or {},
        "rows": rows,
        "summary": summary,
        "outputs": {
            "rows_json": str(out_json).replace("\\", "/"),
            "rows_csv": str(out_csv).replace("\\", "/"),
            "plot_png": str(out_png).replace("\\", "/"),
        },
    }

    _plot(rows, out_png)
    _write_json(out_json, payload_out)
    _write_csv(out_csv, rows)

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
                "event_type": "theory_gpb_scalar_limit_audit",
                "argv": sys.argv,
                "inputs": {"data": data_path},
                "outputs": {
                    "rows_json": out_json,
                    "rows_csv": out_csv,
                    "plot_png": out_png,
                    "public_copies": copied,
                },
                "metrics": {
                    "channels_n": summary.get("channels_n"),
                    "pass_n": summary.get("pass_n"),
                    "reject_n": summary.get("reject_n"),
                    "overall_status": summary.get("overall_status"),
                },
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
